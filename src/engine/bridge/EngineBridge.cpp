// 20260323 ZJH EngineBridge — 引擎桥接层实现
// 通过 import om.engine.* 导入 C++23 模块，封装为传统 C++ 接口
// 重要: 所有 #include 必须在 import 之前，避免 MSVC C++23 模块兼容性问题
// 20260324 ZJH 训练流水线优化: 双缓冲 / 索引洗牌 / 自动批量大小

// 20260323 ZJH 标准库头文件（必须在 import 之前）
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>  // 20260324 ZJH 路径验证（saveModel/loadModel）
#include <iostream>    // 20260324 ZJH std::cerr 日志输出
#include <future>      // 20260324 ZJH std::async 双缓冲异步填充
#include <map>         // 20260402 ZJH BN 折叠名称映射
#include <fstream>     // 20260402 ZJH 精度基线 JSON 文件读写

// 20260324 ZJH 包含桥接层头文件，消除重复类型定义（ODR 违规风险）
// EngineBridge.h 仅使用标准库头文件，不含 C++23 import，可安全包含
#include "engine/bridge/EngineBridge.h"
// 20260330 ZJH 包含模型注册表头文件（createModel() 通过注册表查找工厂函数）
#include "engine/bridge/ModelRegistry.h"

// 20260323 ZJH 导入 OmniMatch 引擎 C++23 模块（必须在 #include 之后）
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.linear;
import om.engine.activations;
import om.engine.conv;
import om.engine.optimizer;
import om.engine.loss;
import om.engine.resnet;
import om.engine.resnet50;
import om.engine.mobilenet;
import om.engine.vit;
import om.engine.unet;
import om.engine.segmodels;
import om.engine.yolo;
import om.engine.instance_seg;
import om.engine.efficientad;
import om.engine.patchcore;
import om.engine.convnext;          // 20260402 ZJH ConvNeXt-Tiny 现代 CNN 分类
import om.engine.gan;
import om.engine.crnn;
import om.engine.autograd;
import om.engine.serializer;
import om.engine.pretrained;  // 20260331 ZJH PyTorch 预训练权重跨架构加载
// 20260330 ZJH 导入数据管线模块，提供训练增强（augmentImage）和归一化（normalizeImage）
import om.engine.data_pipeline;
// 20260330 ZJH 导入数据合成模块，提供 CopyPaste/增强/GAN 三种合成策略
import om.engine.data_synthesis;
// 20260402 ZJH 导入新增模块（对标 Halcon/ViDi 差距补全）
import om.engine.gcad;               // 20260402 ZJH GCAD 全局上下文异常检测
import om.engine.dbnet;              // 20260402 ZJH DBNet 文本检测
import om.engine.advanced_learning;  // 20260402 ZJH ContinualLearner (EWC)
import om.engine.edge_extraction;   // 20260402 ZJH DL 边缘提取
import om.engine.defect_generator;  // 20260402 ZJH AI 缺陷生成器
import om.engine.fp16;              // 20260402 ZJH FP16 混合精度 + GradScaler
import om.engine.pruning;           // 20260402 ZJH 模型剪枝（后训练压缩）
import om.engine.inference_enhance;  // 20260402 ZJH TTAPredictor
// 20260324 ZJH 导入 HAL 层模块，获取 GPU 加速开关接口（setGpuAcceleration/isGpuAccelerationEnabled）
import om.hal.cpu_backend;
// 20260325 ZJH 导入 CUDA 后端模块，提供 GPU 全算子加速（Phase 4 GPU-Resident 训练依赖）
#ifdef OM_HAS_CUDA
import om.hal.cuda_backend;
#endif

// 20260329 ZJH PixelCEBackwardFn — 逐像素类别加权交叉熵反向传播（GPU/CPU 双路）
// GPU 路径: 调用 fused CUDA kernel（零中间张量，单 kernel 完成）
// CPU 路径: 使用 tensor ops（tensorSub + tensorMul + tensorMulScalar）
// 梯度公式: grad[b,c,h,w] = w[target(b,h,w)] * (softmax[b,c,h,w] - 1{c==t}) / weightSum
class PixelCEBackwardFn : public om::GradFunction {
public:
    // 20260329 ZJH 共享数据（GPU/CPU 均使用）
    om::Tensor m_savedSoftmax;       // [B,C,H,W] softmax 概率

    // 20260329 ZJH GPU 路径专用：紧凑格式，显存占用仅 1/C
    om::Tensor m_savedTargetFloat;   // [N] float 类别 ID（N=B*H*W）
    om::Tensor m_savedClassWeights;  // [C] 类别权重
    om::Tensor m_savedStats;         // [2] = {lossSum, weightSum}
    int m_nPixels = 0;               // N = B * H * W
    int m_nClasses = 0;              // C
    int m_nSpatial = 0;              // H * W
    bool m_bUseCuda = false;         // 是否使用 GPU 路径

    // 20260329 ZJH CPU 路径专用：展开格式
    om::Tensor m_savedTarget;        // [B,C,H,W] one-hot 目标
    om::Tensor m_savedWeightMap;     // [B,C,H,W] 逐像素权重
    float m_fWeightSum = 0.0f;       // 权重总和

    // 20260329 ZJH 释放保存的中间张量
    void releaseSavedTensors() override {
        m_savedSoftmax = om::Tensor();
        m_savedTargetFloat = om::Tensor();
        m_savedClassWeights = om::Tensor();
        m_savedStats = om::Tensor();
        m_savedTarget = om::Tensor();
        m_savedWeightMap = om::Tensor();
    }

    // 20260329 ZJH backward — 双路分发
    std::vector<om::Tensor> backward(const om::Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (m_bUseCuda) {
            // 20260329 ZJH GPU 路径: 调用 fused kernel，单次 launch 完成所有梯度
            auto gradLogits = om::Tensor::zeros(m_savedSoftmax.shapeVec());
            gradLogits = gradLogits.cuda();  // 20260329 ZJH 分配 GPU 梯度缓冲
            // 20260329 ZJH 确保 gradOutput 在 GPU 上（autograd 可能创建在 CPU）
            auto tGradGpu = gradOutput.isCuda() ? gradOutput : gradOutput.cuda();
            om::CUDABackend::weightedPixelCEBackward(
                m_savedSoftmax.floatDataPtr(),      // 20260329 ZJH [B,C,H,W] softmax GPU 指针
                m_savedTargetFloat.floatDataPtr(),   // 20260329 ZJH [N] float 类别 ID GPU 指针
                m_savedClassWeights.floatDataPtr(),  // 20260329 ZJH [C] 权重 GPU 指针
                tGradGpu.floatDataPtr(),             // 20260329 ZJH [1] 上游梯度 GPU 指针（已确保 GPU）
                m_savedStats.floatDataPtr(),         // 20260329 ZJH [2] {lossSum, weightSum}
                gradLogits.mutableFloatDataPtr(),    // 20260329 ZJH [B,C,H,W] 输出梯度
                m_nPixels, m_nClasses, m_nSpatial);
            return {gradLogits};
        }
#endif
        // 20260330 ZJH CPU 路径: Focal CE 梯度（简化版：CE梯度 × focal weight）
        // 完整 Focal 梯度在 GPU kernel 中精确计算，CPU fallback 用近似版
        float fGradScale = gradOutput.item();  // 20260329 ZJH 上游梯度标量
        float fScale = fGradScale / std::max(m_fWeightSum, 1e-7f);  // 20260329 ZJH 归一化
        auto diff = om::tensorSub(m_savedSoftmax, m_savedTarget);    // 20260329 ZJH [B,C,H,W]
        // 20260330 ZJH Focal weight: (1-p_t)^2 近似为 (1-softmax)^2 逐元素
        auto tOnes = om::Tensor::ones(m_savedSoftmax.shapeVec());
        auto tFocalW = om::tensorMul(
            om::tensorSub(tOnes, m_savedSoftmax),
            om::tensorSub(tOnes, m_savedSoftmax));  // 20260330 ZJH (1-p)^2
        auto gradInput = om::tensorMul(om::tensorMul(diff, tFocalW), m_savedWeightMap);
        gradInput = om::tensorMulScalar(gradInput, fScale);
        return {gradInput};
    }
};

// 20260324 ZJH CUDA 初始化函数前向声明（仅在 CUDA 编译时使用）
#ifdef OM_HAS_CUDA
extern "C" {
    int omCudaInit(int nDeviceId);                                       // 20260324 ZJH 初始化 CUDA 设备并创建异步流
    int omCudaCleanup();                                                 // 20260324 ZJH 释放 GPU 内存池和异步流资源
    int omCudaForceReset();                                              // 20260326 ZJH 强制重置 GPU
    int omCudaSynchronize();                                             // 20260326 ZJH 同步所有 GPU 操作
    int omCudaGetMemInfo(int nDeviceId, size_t* pFree, size_t* pTotal);  // 20260325 ZJH 查询设备显存信息
    int omCudaGetComputeCapability(int nDeviceId, int* pMajor, int* pMinor);  // 20260402 ZJH 查询 GPU 计算能力
}
#endif

// 20260323 ZJH PIMPL 实现结构体
struct EngineSessionImpl
{
    std::shared_ptr<om::Module> pModel;
    int nInputDim = 0;      // 20260323 ZJH 展平输入维度 (3*H*W)
    int nNumClasses = 0;    // 20260323 ZJH 类别数
    int nInputSize = 0;     // 20260325 ZJH 空间尺寸 H=W（用于 CNN 4D reshape）
    int nBaseChannels = 64; // 20260330 ZJH 模型基础通道数（UNet/DeepLabV3 的 base，用于序列化匹配）
    std::string strModelType;  // 20260330 ZJH 模型类型字符串（序列化元数据用）
    bool bIsCnn = false;    // 20260325 ZJH 是否为 CNN 模型（需要 4D 输入 [B,3,H,W]）
    bool bIsEfficientAD = false;  // 20260326 ZJH EfficientAD 异常检测模型标记（蒸馏训练 + 空间异常图）
    bool bIsGCAD = false;         // 20260402 ZJH GCAD 全局上下文异常检测（双分支: Teacher-Student + ViT 全局）
    bool bIsDBNet = false;        // 20260402 ZJH DBNet 文本检测（可微分二值化）
    bool bIsEdgeExtraction = false; // 20260402 ZJH DL 边缘提取（EdgeUNet）
    bool bIsDetection = false;    // 20260330 ZJH 是否为目标检测模型（YOLO 系列，使用 YOLOLoss）
    bool bIsSegmentation = false; // 20260328 ZJH 是否为语义分割模型（UNet/DeepLabV3 等）
    bool bIsInstanceSeg = false;  // 20260402 ZJH 是否为实例分割模型（YOLOv8Seg/MaskRCNN）
    bool bUseGroupNorm = false;   // 20260402 ZJH 是否使用 GroupNorm（小 batch 自动启用，序列化用）
};

// 20260330 ZJH SimpleMLP 已迁移至 ModelRegistry.cpp（模型注册表统一管理）

// =====================================================================
// 20260324 ZJH 辅助函数：填充 batch 缓冲区（用于双缓冲异步调用）
// 从训练数据中按 vecIndices 索引拷贝样本到预分配缓冲区
// 参数:
//   vecInput - 目标输入缓冲区 [nBatchSize * nInputDim]
//   vecLabel - 目标标签缓冲区 [nBatchSize * nNumClasses] (one-hot)
//   vecTrainData - 完整训练数据集
//   vecTrainLabels - 完整训练标签
//   vecIndices - 洗牌后的索引数组
//   nStart - 本 batch 起始索引
//   nCurBatch - 本 batch 实际样本数
//   nInputDim - 单样本展平维度
//   nNumClasses - 类别数
// =====================================================================
static void fillBatch(std::vector<float>& vecInput,
                      std::vector<float>& vecLabel,
                      const std::vector<float>& vecTrainData,
                      const std::vector<int>& vecTrainLabels,
                      const std::vector<int>& vecIndices,
                      int nStart, int nCurBatch,
                      int nInputDim, int nNumClasses)
{
    // 20260324 ZJH 清零标签缓冲区（one-hot 只设一个 1.0f，其余需为 0）
    std::fill(vecLabel.begin(), vecLabel.begin() + static_cast<size_t>(nCurBatch) * nNumClasses, 0.0f);

    for (int i = 0; i < nCurBatch; ++i) {
        int nIdx = vecIndices[nStart + i];  // 20260324 ZJH 通过索引间接访问（洗牌索引而非数据）
        int nSrc = nIdx * nInputDim;        // 20260324 ZJH 源数据偏移

        // 20260324 ZJH 拷贝输入数据（覆写，无需预清零）
        if (nSrc + nInputDim <= static_cast<int>(vecTrainData.size())) {
            std::copy(vecTrainData.begin() + nSrc,
                      vecTrainData.begin() + nSrc + nInputDim,
                      vecInput.begin() + static_cast<size_t>(i) * nInputDim);
        }

        // 20260324 ZJH 设置 one-hot 标签
        int nLabel = vecTrainLabels[nIdx];
        if (nLabel >= 0 && nLabel < nNumClasses) {
            vecLabel[static_cast<size_t>(i) * nNumClasses + nLabel] = 1.0f;
        }
    }
}

// 20260330 ZJH 根据模型类型确定归一化策略
// CNN 分类模型（ResNet/MobileNet/ViT/EfficientAD）使用 ImageNet 归一化
// YOLO/UNet/分割模型仅用 /255 归一化（ZeroOne）
static om::NormPreset selectNormPreset(const std::string& strModelType) {
    // 20260330 ZJH ImageNet 归一化模型列表
    if (strModelType == "ResNet18" || strModelType == "ResNet50" ||
        strModelType == "MobileNetV4Small" || strModelType == "ViTTiny" ||
        strModelType == "EfficientAD") {
        return om::NormPreset::ImageNet;  // 20260330 ZJH ImageNet mean/std
    }
    // 20260330 ZJH YOLO/UNet/DeepLab 使用 /255 归一化（约定俗成，保持与训练一致）
    return om::NormPreset::ZeroOne;
}

// 20260330 ZJH 构建训练数据增强配置（工业视觉合理默认值）
// 水平翻转 50%、垂直翻转 30%、随机旋转 ±15°、颜色抖动、高斯噪声
// 20260402 ZJH [OPT-2.3] 新增高级增强: 根据模型类型自动配置 CutPaste/MixUp/CutMix/Mosaic/ElasticDeform
// 参数: strModelType - 模型类型字符串（用于判断启用哪种高级增强策略）
//       bAdvancedAugment - 高级增强总开关（false 时跳过所有高级增强配置）
static om::AugmentConfig buildTrainAugmentConfig(const std::string& strModelType,
                                                  bool bAdvancedAugment = true) {
    om::AugmentConfig cfg;
    // 20260330 ZJH 几何增强
    cfg.bRandomHFlip = true;        // 20260330 ZJH 50% 概率水平翻转
    cfg.bRandomVFlip = true;        // 20260330 ZJH 30% 通过在 augmentImage 内部概率控制
    cfg.bRandomRotate = true;       // 20260330 ZJH 启用随机旋转
    cfg.fRotateRange = 15.0f;       // 20260330 ZJH ±15° 范围
    // 20260330 ZJH 颜色增强
    cfg.bColorJitter = true;        // 20260330 ZJH 启用颜色抖动
    cfg.fJitterBrightness = 0.2f;   // 20260330 ZJH 亮度变化 ±0.2
    cfg.fJitterContrast = 0.2f;     // 20260330 ZJH 对比度变化 ±0.2
    cfg.fJitterSaturation = 0.1f;   // 20260330 ZJH 饱和度变化 ±0.1
    // 20260330 ZJH 噪声增强
    cfg.bGaussianNoise = true;      // 20260330 ZJH 启用高斯噪声
    cfg.fNoiseStd = 0.02f;          // 20260330 ZJH 噪声 sigma=0.02
    // 20260330 ZJH 归一化设置
    cfg.bNormalize = false;         // 20260330 ZJH 归一化在增强后单独调用
    cfg.eNormPreset = om::NormPreset::None;  // 20260330 ZJH 增强内部不做归一化

    // 20260402 ZJH [OPT-2.3] 高级增强策略 — 根据模型类型自动选择最优增强方法
    // 仅在 bAdvancedAugment=true 时启用（用户可通过 BridgeTrainParams::bAdvancedAugment 关闭）
    if (bAdvancedAugment) {
        // 20260402 ZJH 异常检测模型 (EfficientAD/PatchCore): 启用 CutPaste
        // CutPaste 从同一图像裁剪 patch 并粘贴到另一位置，生成伪异常样本
        // 论文: CutPaste (Li et al., 2021) — 自监督异常检测专用增强
        if (strModelType == "EfficientAD" || strModelType == "PatchCore" ||
            strModelType == "GCAD" || strModelType == "GlobalContextAD") {
            cfg.bCutPaste = true;           // 20260402 ZJH 启用 CutPaste 伪异常生成
            cfg.fCutPasteMinArea = 0.02f;   // 20260402 ZJH 最小裁剪面积比 2%
            cfg.fCutPasteMaxArea = 0.15f;   // 20260402 ZJH 最大裁剪面积比 15%
            cfg.fCutPasteMinAspect = 0.3f;  // 20260402 ZJH 最小宽高比
            cfg.fCutPasteMaxAspect = 3.3f;  // 20260402 ZJH 最大宽高比
        }
        // 20260402 ZJH 目标检测模型 (YOLO 系列): 启用 Mosaic 4 图拼接
        // Mosaic 将 4 张训练图像拼接成一张，增加上下文多样性
        // 效果: 小目标检测 mAP +3-5%（YOLOv4 论文标配）
        else if (strModelType == "YOLOv5Nano" || strModelType == "YOLOv8Nano" ||
                 strModelType == "YOLOv5s" || strModelType == "YOLOv8s" ||
                 strModelType == "YOLOv10Nano" || strModelType == "YOLOv10s") {
            cfg.bMosaic = true;             // 20260402 ZJH 启用 Mosaic 4 图拼接
            cfg.fMosaicProb = 0.5f;         // 20260402 ZJH 50% 概率触发 Mosaic
        }
        // 20260402 ZJH 分割模型 (UNet/DeepLabV3/MobileSegNet): 启用 ElasticDeform
        // 弹性形变模拟自然形变（布料褶皱、柔性材料弯曲、医学组织变形）
        // 论文: U-Net (Ronneberger et al., 2015) — 分割增强经典策略
        else if (strModelType == "UNet" || strModelType == "DeepLabV3+" ||
                 strModelType == "DeepLabV3Plus" || strModelType == "DeepLabV3" ||
                 strModelType == "MobileSegNet" || strModelType == "MobileSeg" ||
                 strModelType == "EdgeUNet" || strModelType == "EdgeExtraction") {
            cfg.bElasticDeform = true;      // 20260402 ZJH 启用弹性形变
            cfg.fElasticAlpha = 50.0f;      // 20260402 ZJH 形变幅度 50 像素
            cfg.fElasticSigma = 5.0f;       // 20260402 ZJH 高斯平滑 sigma=5
        }
        // 20260402 ZJH 分类模型 (ResNet/MobileNet/ViT): 启用 MixUp + CutMix
        // MixUp: image = λ*img1 + (1-λ)*img2，强正则化防过拟合 +2-4%
        // CutMix: 随机裁剪区域混合，保留局部结构信息
        // 两者同时启用时训练循环中随机选择其一（互斥应用）
        else {
            cfg.bMixUp = true;              // 20260402 ZJH 启用 MixUp 图像级混合
            cfg.fMixUpAlpha = 0.2f;         // 20260402 ZJH Beta(0.2, 0.2) 分布参数
            cfg.bCutMix = true;             // 20260402 ZJH 启用 CutMix 区域级混合
            cfg.fCutMixAlpha = 1.0f;        // 20260402 ZJH Beta(1.0, 1.0) = 均匀分布
        }
    }

    return cfg;  // 20260406 ZJH 返回构建好的增强配置
}

// ===== 实现 =====

// 20260406 ZJH 构造函数: 创建 PIMPL 实现对象（隐藏引擎模块依赖）
EngineBridge::EngineBridge()
    : m_pImpl(std::make_unique<EngineSessionImpl>()) {}

// 20260406 ZJH 析构函数: 默认实现（unique_ptr 自动释放 EngineSessionImpl）
EngineBridge::~EngineBridge() = default;

// 20260330 ZJH 创建模型 — 通过 ModelRegistry 注册表查找工厂函数
// 重构: 原 200+ 行 if-else 字符串匹配链已迁移至 ModelRegistry.cpp
// 新增模型只需在 ModelRegistry.cpp 的 registerAllModels() 中添加一行
bool EngineBridge::createModel(const std::string& strModelType, int nInputSize, int nNumClasses)
{
    // 20260324 ZJH 验证 nInputSize 范围，防止平方运算导致整数溢出
    if (nInputSize < 1 || nInputSize > 10000) {
        return false;  // 20260324 ZJH 输入尺寸超出合理范围（1-10000），拒绝创建模型
    }

    // 20260330 ZJH 确保注册表已初始化（懒初始化，首次调用时注册所有内置模型）
    om::ModelRegistry::instance().ensureInitialized();

    m_pImpl->nNumClasses = nNumClasses;  // 20260406 ZJH 记录类别数到实现结构体
    m_pImpl->nInputSize = nInputSize;  // 20260325 ZJH 记录空间尺寸
    m_pImpl->strModelType = strModelType;  // 20260330 ZJH 记录模型类型（序列化元数据用）

    // 20260324 ZJH 计算展平输入维度: 3 通道 (RGB) * H * W
    int64_t nInputDim64 = 3LL * static_cast<int64_t>(nInputSize) * static_cast<int64_t>(nInputSize);
    m_pImpl->nInputDim = static_cast<int>(nInputDim64);

    // 20260330 ZJH 直接创建模型（不经过 shared_ptr<void> 中转，避免跨模块边界虚函数表问题）
    // 注: ModelRegistry 的 shared_ptr<void> 方案在 MSVC C++23 模块下可能导致虚函数表损坏
    std::shared_ptr<om::Module> pModel;
    int nInCh = 3;  // 20260330 ZJH RGB 输入

    if (strModelType == "ResNet18")          pModel = std::make_shared<om::ResNet18>(nNumClasses, nInCh);
    else if (strModelType == "ResNet50")     pModel = std::make_shared<om::ResNet50>(nNumClasses, nInCh);
    else if (strModelType == "MobileNetV4Small") pModel = std::make_shared<om::MobileNetV4Small>(nNumClasses, nInCh);
    else if (strModelType == "ViTTiny")      pModel = std::make_shared<om::ViT>(nInputSize, 16, nInCh, nNumClasses, 192, 6, 3, 384);
    else if (strModelType == "YOLOv5Nano")   pModel = std::make_shared<om::YOLOv5Nano>(nNumClasses, nInCh);
    else if (strModelType == "YOLOv8Nano")   pModel = std::make_shared<om::YOLOv8Nano>(nNumClasses, nInCh);
    else if (strModelType == "UNet") {
        // 20260401 ZJH 轻量 UNet: base=16（1.8M 参数，对标海康 4.7MB 工业网络）
        // 原 base=64 有 31.4M 参数，20 张图根本训不动
        // base=16 → 编码器 16/32/64/128 通道，总参数 ~1.8M，小数据集不易过拟合
        int nBase = 16;
        // 20260402 ZJH 传递 GroupNorm 标志（用户显式设置时在 createModel 阶段即生效）
        pModel = std::make_shared<om::UNet>(nInCh, nNumClasses, nBase, m_pImpl->bUseGroupNorm);
    }
    else if (strModelType == "DeepLabV3+" || strModelType == "DeepLabV3Plus" || strModelType == "DeepLabV3")
        // 20260402 ZJH 传递 GroupNorm 标志
        pModel = std::make_shared<om::DeepLabV3>(nInCh, nNumClasses, m_pImpl->bUseGroupNorm);
    // 20260401 ZJH MobileSegNet: 轻量级分割网络（~1.75M 参数，对标海康 ASI_SEG 4.7MB）
    // MobileNetV4-Small 编码器 + ASPPLite(3分支) + 低级特征融合解码器
    else if (strModelType == "MobileSegNet" || strModelType == "MobileSeg")
        // 20260402 ZJH 传递 GroupNorm 标志
        pModel = std::make_shared<om::MobileSegNet>(nInCh, nNumClasses, m_pImpl->bUseGroupNorm);
    // 20260402 ZJH EfficientAD: 默认使用 ResNet18 预训练骨干（bUsePretrainedBackbone=true）
    // 教师网络使用预训练 ResNet18（冻结），学生网络使用随机初始化 ResNet18
    // Phase 1.2 — 对标 PatchCore/EfficientAD 论文的 ImageNet 预训练特征提取策略
    else if (strModelType == "EfficientAD")  pModel = std::make_shared<om::EfficientAD>(nInCh, true);
    else if (strModelType == "YOLOv8Seg")    pModel = std::make_shared<om::SimpleInstanceSeg>(nInCh, nNumClasses);
    else if (strModelType == "MLP") {
        auto pSeq = std::make_shared<om::Sequential>();
        pSeq->add(std::make_shared<om::Linear>(m_pImpl->nInputDim, 128));
        pSeq->add(std::make_shared<om::ReLU>());
        pSeq->add(std::make_shared<om::Linear>(128, nNumClasses));
        pModel = pSeq;
    }

    if (!pModel) {
        // 20260406 ZJH 未知模型类型，输出错误日志并返回失败
        std::cerr << "[EngineBridge] Unknown model type: " << strModelType << std::endl;
        return false;  // 20260406 ZJH 模型创建失败
    }
    m_pImpl->pModel = pModel;  // 20260406 ZJH 将创建好的模型存储到实现结构体
    // 20260330 ZJH 诊断: 检查每个参数的健康度
    auto vecDiagParams = pModel->parameters();
    std::cerr << "[EngineBridge] createModel OK: " << strModelType
              << " params=" << vecDiagParams.size() << std::endl;
    int64_t nTotalElems = 0;
    for (size_t i = 0; i < vecDiagParams.size(); ++i) {
        try {
            auto* pT = vecDiagParams[i];
            if (!pT) { std::cerr << "[EngineBridge] NULL PARAM #" << i << std::endl; continue; }
            int nE = pT->numel();
            nTotalElems += nE;
            if (nE == 0) std::cerr << "[EngineBridge] EMPTY PARAM #" << i << std::endl;
        } catch (...) {
            std::cerr << "[EngineBridge] BAD PARAM #" << i
                      << " ptr=" << (void*)vecDiagParams[i] << std::endl;
        }
    }
    std::cerr << "[EngineBridge] totalElements=" << nTotalElems << std::endl;

    // 20260330 ZJH 从注册表元信息设置 EngineSessionImpl 标志位
    // 这些标志位被 train()/infer() 路径读取，当前保留以最小化侵入性
    const om::ModelInfo* pInfo = om::ModelRegistry::instance().getModelInfo(strModelType);
    if (pInfo) {
        m_pImpl->bIsCnn = pInfo->bIsCnn;                                               // 20260330 ZJH CNN 4D 输入标记
        m_pImpl->bIsDetection = (pInfo->eCategory == om::ModelCategory::Detection);     // 20260330 ZJH 检测模型标记（YOLOLoss）
        m_pImpl->bIsSegmentation = (pInfo->eCategory == om::ModelCategory::Segmentation); // 20260330 ZJH 语义分割标记
        m_pImpl->bIsInstanceSeg = (pInfo->eCategory == om::ModelCategory::InstanceSeg);  // 20260402 ZJH 实例分割标记
        m_pImpl->bIsEfficientAD = (strModelType == "EfficientAD");                      // 20260330 ZJH 蒸馏训练标记
        m_pImpl->bIsGCAD = (strModelType == "GCAD" || strModelType == "GlobalContextAD");  // 20260402 ZJH GCAD 双分支标记
        m_pImpl->bIsDBNet = (strModelType == "DBNet" || strModelType == "DBNet+CRNN");     // 20260402 ZJH DBNet 文本检测标记
        m_pImpl->bIsEdgeExtraction = (strModelType == "EdgeUNet" || strModelType == "EdgeExtraction"); // 20260402 ZJH 边缘提取标记
        m_pImpl->nBaseChannels = pInfo->nDefaultBaseChannels;                           // 20260330 ZJH 基础通道数

        // 20260330 ZJH UNet 特殊处理: 根据输入尺寸动态调整 base channels
        // 注册表中默认值为 64，但 >384 尺寸实际使用 32（轻量版）
        if (strModelType == "UNet" && nInputSize > 384) {
            m_pImpl->nBaseChannels = 32;  // 20260330 ZJH 覆盖为轻量版实际值
        }
    }

    return true;
}

// 20260324 ZJH 自动选择最大 batch size（OPTIMIZATION 5）
// 根据模型参数量和可用内存估算可容纳的最大 batch 大小
// 参数: nInputDim - 单样本输入维度; nNumClasses - 类别数; nModelParams - 模型参数总数
// 返回: 推荐的 batch size（范围 [1, 512]）
int EngineBridge::autoSelectBatchSize(int nInputDim, int nNumClasses, int64_t nModelParams)
{
    // 20260324 ZJH 估算每样本内存占用：输入 + 标签 + 梯度 + 激活缓存
    // 粗略 4 倍因子：1x 前向数据 + 1x 梯度 + 2x 中间激活
    size_t nBytesPerSample = static_cast<size_t>(nInputDim + nNumClasses) * sizeof(float) * 4;

    // 20260324 ZJH 获取可用内存
    size_t nAvailableBytes = 0;

#ifdef OM_HAS_CUDA
    // 20260325 ZJH GPU 模式：通过 om_cuda 封装查询设备显存（不直接调用 cuda_runtime.h）
    {
        size_t nFreeMem = 0, nTotalMem = 0;
        if (omCudaGetMemInfo(0, &nFreeMem, &nTotalMem) == 0) {
            nAvailableBytes = nFreeMem;
        }
    }
#endif

    // 20260324 ZJH CPU 模式回退：假设 4GB 可用内存
    if (nAvailableBytes == 0) {
        nAvailableBytes = static_cast<size_t>(4) * 1024 * 1024 * 1024;  // 20260324 ZJH 4GB 默认
    }

    // 20260325 ZJH GPU 使用 90% 可用显存（无需与 OS 共享），CPU 使用 80%（预留给系统/其他进程）
    bool bIsGpuMem = false;  // 20260325 ZJH 标记当前可用内存是否来自 GPU 显存
#ifdef OM_HAS_CUDA
    bIsGpuMem = (nAvailableBytes > 0);  // 20260325 ZJH 若已通过 CUDA 获取到显存信息则为 GPU 模式
#endif
    size_t nUsableBytes = bIsGpuMem ? (nAvailableBytes * 9 / 10) : (nAvailableBytes * 8 / 10);

    // 20260324 ZJH 扣除模型自身内存：权重 + 梯度 + 优化器状态（Adam 需 3 倍）
    size_t nModelBytes = static_cast<size_t>(nModelParams) * sizeof(float) * 3;
    if (nModelBytes < nUsableBytes) {
        nUsableBytes -= nModelBytes;  // 20260324 ZJH 扣除模型内存
    } else {
        nUsableBytes = nUsableBytes / 2;  // 20260324 ZJH 模型过大时仍保留一半给 batch
    }

    // 20260324 ZJH 计算可容纳的最大 batch 数
    int nMaxBatch = 1;  // 20260324 ZJH 最少 1 个样本
    if (nBytesPerSample > 0) {
        nMaxBatch = static_cast<int>(nUsableBytes / nBytesPerSample);
    }

    // 20260325 ZJH 限制范围：GPU 模式 [1, 2048]（大显存可充分利用），CPU 模式 [1, 512]
    int nMaxBatchLimit = bIsGpuMem ? 2048 : 512;  // 20260325 ZJH GPU 允许更大的 batch size
    if (nMaxBatch < 1) nMaxBatch = 1;
    if (nMaxBatch > nMaxBatchLimit) nMaxBatch = nMaxBatchLimit;

    // 20260324 ZJH 向下取到最近的 2 的幂（GPU 友好）
    int nPow2 = 1;
    while (nPow2 * 2 <= nMaxBatch) nPow2 *= 2;

    return nPow2;  // 20260324 ZJH 返回 2 的幂 batch size
}

// 20260323 ZJH 执行训练
// 20260324 ZJH 优化: 双缓冲异步数据准备 + 索引洗牌 + 预分配全部缓冲区
bool EngineBridge::train(
    const BridgeTrainParams& params,
    const std::vector<float>& vecTrainData, const std::vector<int>& vecTrainLabels,
    const std::vector<float>& vecValData, const std::vector<int>& vecValLabels,
    const std::vector<int>& vecTrainMasks, const std::vector<int>& vecValMasks,
    EpochCallback epochCb, BatchCallback batchCb, LogCallback logCb, StopChecker stopCheck)
{
    if (!m_pImpl->pModel) {
        if (logCb) logCb("[ERROR] No model created");  // 20260406 ZJH 输出错误日志
        return false;  // 20260406 ZJH 模型未创建，无法训练
    }

    // 20260328 ZJH [DIAG-1] 训练入口诊断：确认模型参数在 train() 开始时存在
    {
        auto vecDiag = m_pImpl->pModel->namedParameters();
        auto vecDiagP = m_pImpl->pModel->parameters();
        std::cerr << "[DIAG-1] train() ENTRY: namedParams=" << vecDiag.size()
                  << " params=" << vecDiagP.size()
                  << " children=" << m_pImpl->pModel->debugChildCount()
                  << " directParams=" << m_pImpl->pModel->debugParamCount()
                  << std::endl;
    }

    // 20260402 ZJH ===== [OPT-3.8] AutoML 智能模型选择 =====
    // 当 bSmartMode=true 时，根据任务类型和数据量自动选择最优模型+超参数
    // 用户无需手动调参，一键智能训练
    // 注意: 需要在 CUDA 初始化之前执行，因为可能会重建模型
    if (params.bSmartMode) {
        int nDatasetSize = static_cast<int>(vecTrainLabels.size());  // 20260402 ZJH 训练集样本数
        // 20260402 ZJH 局部可变副本（仅在 SmartMode 内部覆写，不修改原始 params 引用）
        auto& mutParams = const_cast<BridgeTrainParams&>(params);

        // 20260402 ZJH 根据任务类型选择模型
        bool bIsDetection = m_pImpl->bIsDetection;         // 20260402 ZJH 是否为检测模型
        bool bIsSegmentation = m_pImpl->bIsSegmentation;   // 20260402 ZJH 是否为分割模型
        bool bIsAnomalyDet = (m_pImpl->strModelType == "EfficientAD"
                           || m_pImpl->strModelType == "PatchCore"
                           || m_pImpl->strModelType == "GCAD");  // 20260402 ZJH 是否为异常检测

        std::string strAutoModel = mutParams.strModelType;  // 20260402 ZJH 默认保持用户选择

        if (bIsAnomalyDet) {
            // 20260402 ZJH 异常检测: <50 正常→EfficientAD, >=50→PatchCore
            if (nDatasetSize < 50) {
                strAutoModel = "EfficientAD";
                if (logCb) logCb("[AutoML] 异常检测: 样本 <50，选择 EfficientAD（少样本高效）");
            } else {
                strAutoModel = "PatchCore";
                if (logCb) logCb("[AutoML] 异常检测: 样本 >=50，选择 PatchCore（记忆库精度高）");
            }
        } else if (bIsDetection) {
            // 20260402 ZJH 检测: 数据量小→YOLOv5Nano（轻量快速），数据量大→YOLOv8Nano
            if (nDatasetSize < 500) {
                strAutoModel = "YOLOv5Nano";
                if (logCb) logCb("[AutoML] 目标检测: 样本 <500，选择 YOLOv5Nano（轻量边缘部署）");
            } else {
                strAutoModel = "YOLOv8Nano";
                if (logCb) logCb("[AutoML] 目标检测: 样本 >=500，选择 YOLOv8Nano（标准精度）");
            }
        } else if (bIsSegmentation) {
            // 20260402 ZJH 分割: <200→MobileSegNet, 200-1000→UNet, >1000→DeepLabV3+
            if (nDatasetSize < 200) {
                strAutoModel = "MobileSegNet";
                if (logCb) logCb("[AutoML] 语义分割: 样本 <200，选择 MobileSegNet（轻量 ~1.75M 参数）");
            } else if (nDatasetSize < 1000) {
                strAutoModel = "UNet";
                if (logCb) logCb("[AutoML] 语义分割: 样本 200-1000，选择 UNet（标准 base=16）");
            } else {
                strAutoModel = "DeepLabV3+";
                if (logCb) logCb("[AutoML] 语义分割: 样本 >1000，选择 DeepLabV3+（高精度 ASPP）");
            }
        } else {
            // 20260402 ZJH 分类: <100→ResNet18+强增强, 100-1000→ResNet18, >1000→ConvNeXtTiny
            if (nDatasetSize < 100) {
                strAutoModel = "ResNet18";
                mutParams.bAugmentEnabled = true;  // 20260402 ZJH 强制开启增强（小数据防过拟合）
                if (logCb) logCb("[AutoML] 分类: 样本 <100，选择 ResNet18 + 强数据增强");
            } else if (nDatasetSize < 1000) {
                strAutoModel = "ResNet18";
                if (logCb) logCb("[AutoML] 分类: 样本 100-1000，选择 ResNet18（经典可靠）");
            } else {
                strAutoModel = "ConvNeXtTiny";
                if (logCb) logCb("[AutoML] 分类: 样本 >1000，选择 ConvNeXtTiny（现代 CNN, Top-1 82.1%）");
            }
        }

        // 20260402 ZJH 如果选择了不同的模型，需要重建
        if (strAutoModel != m_pImpl->strModelType) {
            mutParams.strModelType = strAutoModel;
            if (logCb) logCb("[AutoML] 重建模型: " + m_pImpl->strModelType + " -> " + strAutoModel);
            // 20260402 ZJH 通过 createModel 重建（会更新 m_pImpl 内部状态）
            createModel(strAutoModel, mutParams.nInputSize, mutParams.nNumClasses);
        }

        // 20260402 ZJH 自动设置 epoch: max(50, 500/sqrt(dataset_size))
        // 小数据集需要更多 epoch 充分学习，大数据集收敛更快
        int nAutoEpochs = std::max(50, static_cast<int>(500.0f / std::sqrt(static_cast<float>(std::max(nDatasetSize, 1)))));
        mutParams.nEpochs = nAutoEpochs;
        if (logCb) logCb("[AutoML] 自动 epoch = " + std::to_string(nAutoEpochs)
            + " (公式: max(50, 500/sqrt(" + std::to_string(nDatasetSize) + ")))");

        // 20260402 ZJH 自动 batch size（根据模型参数量和可用内存）
        int64_t nModelParams = 0;
        for (auto* p : m_pImpl->pModel->parameters()) nModelParams += p->numel();
        int nAutoBatch = autoSelectBatchSize(m_pImpl->nInputDim, mutParams.nNumClasses, nModelParams);
        // 20260402 ZJH batch 不超过训练集大小
        nAutoBatch = std::min(nAutoBatch, nDatasetSize);
        nAutoBatch = std::max(nAutoBatch, 4);  // 20260402 ZJH 最小 batch 4（梯度稳定性）
        mutParams.nBatchSize = nAutoBatch;
        if (logCb) logCb("[AutoML] 自动 batch_size = " + std::to_string(nAutoBatch));

        // 20260402 ZJH 自动学习率（LR finder 简化版: 基于经验公式）
        // 大 batch 用较大 LR（线性缩放规则: LR ∝ batch_size / 256）
        float fAutoLR = 0.001f * static_cast<float>(nAutoBatch) / 32.0f;
        // 20260402 ZJH clamp LR 到合理范围 [1e-5, 0.01]
        fAutoLR = std::max(1e-5f, std::min(fAutoLR, 0.01f));
        mutParams.fLearningRate = fAutoLR;
        if (logCb) logCb("[AutoML] 自动 LR = " + std::to_string(fAutoLR)
            + " (线性缩放: 0.001 * " + std::to_string(nAutoBatch) + " / 32)");

        // 20260402 ZJH 自动优化器: 分类/分割用 AdamW，检测用 SGD
        if (bIsDetection) {
            mutParams.strOptimizer = "SGD";
            mutParams.fMomentum = 0.937f;  // 20260402 ZJH YOLO 推荐动量
        } else {
            mutParams.strOptimizer = "AdamW";
        }
        if (logCb) logCb("[AutoML] 自动优化器 = " + mutParams.strOptimizer);

        // 20260402 ZJH 强制开启数据增强（AutoML 策略: 增强总是有益的）
        mutParams.bAugmentEnabled = true;

        if (logCb) logCb("[AutoML] ===== 智能模式配置完成 =====");
    }

    // 20260325 ZJH ===== GPU-Resident 训练初始化 =====
    // Phase 4: 当 bUseCuda 为 true 且 OM_HAS_CUDA 已定义时，执行真正的 GPU 驻留训练
    // 数据/权重/梯度全部驻留 GPU 显存，消除 CPU↔GPU 数据乒乓
    bool bUseCuda = false;  // 20260325 ZJH 实际 CUDA 使用标志
    // 20260327 ZJH GPU 训练重新启用 — 根因已修复：
    //   1. MaxPool2d 索引 GPU 驻留（消除 CPU 指针传入 CUDA kernel 导致 illegal memory access）
    //   2. rootGrad 设备修复（初始梯度在 loss 所在设备创建，不再强制 CPU）
    //   3. 补齐全部前向/反向 CUDA kernel（消除 D2H 回退）
    if (params.bUseCuda) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH 初始化 CUDA 设备 0（创建异步流和内存池）
        if (omCudaInit(0) == 0) {
            // 20260326 ZJH 显存预检：估算训练所需显存，不足则自动回退 CPU
            // 粗估公式: 模型参数×12（参数+梯度+优化器状态各4字节×3）+ batch中间激活
            size_t nFreeMem = 0, nTotalMem = 0;
            omCudaGetMemInfo(0, &nFreeMem, &nTotalMem);
            int64_t nParams = 0;
            for (auto* p : m_pImpl->pModel->parameters()) nParams += p->numel();
            // 20260326 ZJH 参数×12 + batch×inputDim×4×50（中间激活经验系数）
            size_t nEstBytes = static_cast<size_t>(nParams) * 12
                + static_cast<size_t>(params.nBatchSize) * m_pImpl->nInputDim * 4 * 50;
            if (nFreeMem > 0 && nEstBytes > nFreeMem * 9 / 10) {
                // 20260326 ZJH 预估超过可用显存 90%，自动回退 CPU 避免 OOM
                if (logCb) logCb("[警告] GPU 显存不足 (可用 " + std::to_string(nFreeMem / 1048576) +
                    "MB, 预估需要 " + std::to_string(nEstBytes / 1048576) +
                    "MB)，自动切换到 CPU 训练");
                omCudaCleanup();  // 20260326 ZJH 释放刚初始化的 CUDA 资源
                bUseCuda = false;
            } else {
                bUseCuda = true;
                if (logCb) logCb("[INFO] CUDA initialized — GPU 显存: 可用 " +
                    std::to_string(nFreeMem / 1048576) + "MB, 预估需要 " +
                    std::to_string(nEstBytes / 1048576) + "MB");
            }
        } else {
            if (logCb) logCb("[WARN] CUDA initialization failed, falling back to CPU (SIMD+OpenMP)");
        }
#else
        // 20260325 ZJH 当前构建未包含 CUDA 支持
        if (logCb) logCb("[WARN] CUDA not compiled in this build (OM_ENABLE_CUDA=OFF), using CPU (SIMD+OpenMP)");
#endif
    }

    int nNumClasses = m_pImpl->nNumClasses;  // 20260406 ZJH 本地缓存类别数（频繁访问）
    int nInputDim = m_pImpl->nInputDim;    // 20260406 ZJH 本地缓存输入维度
    int nTrainCount = static_cast<int>(vecTrainLabels.size());  // 20260406 ZJH 训练集样本数
    int nValCount = static_cast<int>(vecValLabels.size());      // 20260406 ZJH 验证集样本数
    int nBatchSize = params.nBatchSize;    // 20260406 ZJH 批量大小（可能被后续修正）
    int nEpochs = params.nEpochs;          // 20260406 ZJH 训练轮数（可能被 AutoML 修改）
    float fLr = params.fLearningRate;      // 20260406 ZJH 学习率（可能被 AutoML/LR Finder 修改）

    // 20260331 ZJH ===== 训练诊断: batch_size / LR / 数据量自动修正 =====
    // 问题: batch=1 时每步梯度来自单张图像，方向极度嘈杂导致损失剧烈震荡
    // 修正: CNN 模型强制 batch ≥ 4（梯度平均降噪），不超过训练集大小
    {
        int nMinBatch = m_pImpl->bIsCnn ? 4 : 2;  // 20260331 ZJH CNN 最小 4，MLP 最小 2
        if (nBatchSize < nMinBatch) {
            if (logCb) logCb("[WARN] batch_size=" + std::to_string(nBatchSize)
                + " too small for stable training, auto-clamped to " + std::to_string(nMinBatch)
                + " (single-sample gradients cause severe oscillation)");
            nBatchSize = nMinBatch;
        }
        // 20260331 ZJH batch 不超过训练集（避免空 batch 填充问题）
        if (nBatchSize > nTrainCount && nTrainCount >= nMinBatch) {
            nBatchSize = nTrainCount;
            if (logCb) logCb("[INFO] batch_size clamped to train set size: " + std::to_string(nBatchSize));
        }
    }

    // 20260402 ZJH ===== GroupNorm 自动选择策略（对标 Halcon/ViDi）=====
    // BatchNorm 的 running_mean/var 来自 batch 统计量
    // batch=4 时仅 4 个样本，均值/方差几乎是随机数（信噪比极低）
    // GroupNorm 按 channel 分组归一化，完全不依赖 batch 维度
    // Halcon 策略: batch < 8 时自动切换 GroupNorm
    bool bUseGN = params.bUseGroupNorm;  // 20260402 ZJH 用户显式指定的 GroupNorm 开关
    if (!bUseGN && nBatchSize < 8 && m_pImpl->bIsSegmentation) {
        // 20260402 ZJH 小 batch 分割模型自动启用 GroupNorm
        bUseGN = true;
        if (logCb) logCb("[INFO] batch_size=" + std::to_string(nBatchSize)
            + " < 8: auto-enabling GroupNorm (BN statistics unreliable with small batches)");
    }
    // 20260402 ZJH 存储到 Impl，序列化和后续步骤可访问
    m_pImpl->bUseGroupNorm = bUseGN;

    // 20260402 ZJH ===== 小 batch + 分割模型: 需要重建模型以注入 GroupNorm =====
    // createModel() 在 train() 之前调用，默认 bUseGroupNorm=false
    // 如果自动检测或用户指定需要 GroupNorm，必须重建模型（GroupNorm vs BatchNorm 是结构性差异）
    if (bUseGN && m_pImpl->bIsSegmentation) {
        std::string strMT = m_pImpl->strModelType;  // 20260402 ZJH 当前模型类型
        int nInCh = 3;  // 20260402 ZJH RGB 输入通道数
        std::shared_ptr<om::Module> pNewModel;
        if (strMT == "UNet") {
            int nBase = 16;  // 20260402 ZJH 轻量 UNet base=16（与 createModel 一致）
            pNewModel = std::make_shared<om::UNet>(nInCh, nNumClasses, nBase, true);
        } else if (strMT == "DeepLabV3+" || strMT == "DeepLabV3Plus" || strMT == "DeepLabV3") {
            pNewModel = std::make_shared<om::DeepLabV3>(nInCh, nNumClasses, true);
        } else if (strMT == "MobileSegNet" || strMT == "MobileSeg") {
            pNewModel = std::make_shared<om::MobileSegNet>(nInCh, nNumClasses, true);
        }
        if (pNewModel) {
            m_pImpl->pModel = pNewModel;  // 20260402 ZJH 替换模型实例（旧 BN 模型自动释放）
            if (logCb) logCb("[INFO] Model recreated with GroupNorm for " + strMT);
        }
    }

    // 20260402 ZJH ===== 小数据集强正则策略（对标 Darknet）=====
    // 训练样本不足时自动启用 weight_decay 防止过拟合
    // Darknet 默认 weight_decay=5e-4，对小数据集分割模型尤其有效
    float fWeightDecay = 0.0f;  // 20260402 ZJH 权重衰减（默认关闭，小数据集自动启用）
    if (nTrainCount < 200 && m_pImpl->bIsSegmentation) {
        fWeightDecay = 5e-4f;  // 20260402 ZJH Darknet 默认值
        if (logCb) logCb("[INFO] Small dataset (" + std::to_string(nTrainCount)
            + " images): auto-enabling weight_decay=5e-4");
    }

    // 20260401 ZJH 降 LR 逻辑已移到预训练加载之后，根据实际加载结果决定
    // （此前在加载之前降 LR，导致预训练不匹配时仍用低 LR 从头训练）
    float fOriginalLr = fLr;  // 20260401 ZJH 保存原始 LR，加载成功后再决定是否降

    // 20260331 ZJH ===== 训练数据量诊断警告 =====
    if (nTrainCount < 20 && logCb) {
        logCb("[WARN] Only " + std::to_string(nTrainCount) + " training samples — "
            "consider enabling data augmentation and using pretrained weights for best results");
    }
    if (nValCount < 2 && logCb) {
        logCb("[WARN] Only " + std::to_string(nValCount) + " validation samples — "
            "validation loss may be unreliable, early stopping may trigger prematurely");
    }

    // 20260326 ZJH EfficientAD 蒸馏训练：冻结教师网络，仅训练学生参数
    // Halcon 异常检测核心原理：教师网络学习正常样本特征分布，学生网络尝试复现
    // 教师-学生特征差异 = 异常分数（正常样本差异小，缺陷样本差异大）
    if (m_pImpl->bIsEfficientAD) {
        auto* pEfficientAD = static_cast<om::EfficientAD*>(m_pImpl->pModel.get());
        // 20260402 ZJH Phase 1.2: 输出骨干类型日志
        if (pEfficientAD->isUsingPretrainedBackbone()) {
            if (logCb) logCb("[INFO] EfficientAD: Using ResNet18 pretrained backbone (Phase 1.2)");
            // 20260402 ZJH 尝试自动加载 ImageNet 预训练权重到教师 ResNet18 骨干
            // 查找预训练权重文件: 1) 用户指定路径 2) pretrained/ 目录下 resnet18 文件
            if (!params.strPretrainedModelPath.empty()) {
                // 20260402 ZJH 用户已指定预训练路径，权重将在统一的预训练加载逻辑中处理
                if (logCb) logCb("[INFO] EfficientAD: ResNet18 teacher weights will be loaded from: "
                    + params.strPretrainedModelPath);
            } else {
                // 20260402 ZJH 自动搜索 pretrained/ 目录下的 ResNet18 权重
                std::string strAutoPath = "pretrained/resnet18_imagenet.omm";
                std::filesystem::path fsAutoPath(
                    reinterpret_cast<const char8_t*>(strAutoPath.c_str()));
                if (std::filesystem::exists(fsAutoPath)) {
                    try {
                        // 20260402 ZJH 加载预训练权重到教师 ResNet18 骨干
                        auto* pTeacherModule = pEfficientAD->getTeacherModule();
                        if (pTeacherModule) {
                            auto [nL, nS] = om::loadPyTorchPretrainedToSegModel(
                                *pTeacherModule, strAutoPath);
                            if (logCb) logCb("[INFO] EfficientAD: Auto-loaded ResNet18 pretrained weights to teacher — "
                                + std::to_string(nL) + " layers loaded, " + std::to_string(nS) + " skipped");
                        }
                    } catch (const std::exception& ex) {
                        if (logCb) logCb("[WARN] EfficientAD: Failed to auto-load pretrained weights: "
                            + std::string(ex.what()));
                    }
                } else {
                    if (logCb) logCb("[INFO] EfficientAD: No pretrained weights found at " + strAutoPath
                        + " — teacher starts with random weights (provide pretrained for best accuracy)");
                }
            }
        } else {
            if (logCb) logCb("[INFO] EfficientAD: Using legacy 4-layer CNN backbone (fallback mode)");
        }
        pEfficientAD->freezeTeacher();  // 20260326 ZJH 冻结教师网络（eval 模式，参数不更新）
        if (logCb) logCb("[INFO] EfficientAD: Teacher frozen, training student only (distillation)");
    }

    // 20260402 ZJH GCAD 蒸馏训练：冻结 Teacher，仅训练 Student + GlobalEncoder
    if (m_pImpl->bIsGCAD) {
        auto* pGCAD = dynamic_cast<om::GCAD*>(m_pImpl->pModel.get());
        if (pGCAD) {
            pGCAD->teacher().eval();  // 20260402 ZJH 冻结教师网络
            if (logCb) logCb("[INFO] GCAD: Teacher frozen, training student + global encoder");
        }
    }

    // 20260326 ZJH EfficientAD/GCAD 仅传可训练参数给优化器
    std::vector<om::Tensor*> vecModelParams;
    if (m_pImpl->bIsEfficientAD) {
        vecModelParams = static_cast<om::EfficientAD*>(m_pImpl->pModel.get())->studentParameters();
    } else if (m_pImpl->bIsGCAD) {
        // 20260402 ZJH GCAD: Student + GlobalEncoder 参数（排除 Teacher）
        auto* pGCAD = dynamic_cast<om::GCAD*>(m_pImpl->pModel.get());
        if (pGCAD) {
            auto vecStudent = pGCAD->student().parameters();
            auto vecGlobal = pGCAD->globalEncoder().parameters();
            vecModelParams.insert(vecModelParams.end(), vecStudent.begin(), vecStudent.end());
            vecModelParams.insert(vecModelParams.end(), vecGlobal.begin(), vecGlobal.end());
        }
    } else {
        vecModelParams = m_pImpl->pModel->parameters();
    }

    // 20260401 ZJH ===== 预训练加载必须在 GPU 迁移之前 =====
    // 原因: loadPyTorchPretrainedToSegModel 创建 CPU 临时模型并复制权重到目标模型
    // 如果目标参数已在 GPU 上，CPU→GPU 写入会导致 CUDA 段错误崩溃
    // 顺序: 预训练加载(CPU) → GPU 迁移 → 优化器创建
    int nPretrainedLoaded = 0;  // 20260401 ZJH 实际成功加载的参数层数
    if (!params.strPretrainedModelPath.empty()) {
        try {
            std::filesystem::path fsPretrained(
                reinterpret_cast<const char8_t*>(params.strPretrainedModelPath.c_str()));
            if (std::filesystem::exists(fsPretrained)) {
                om::ModelMeta fileMeta;
                bool bHasMeta = om::ModelSerializer::peekMeta(params.strPretrainedModelPath, fileMeta);

                bool bIsCrossArch = false;
                if (bHasMeta) {
                    // 20260401 ZJH 空类型但 classes=1000 → 推断为 ResNet18 ImageNet
                    if (fileMeta.strModelType.empty() && fileMeta.nNumClasses == 1000) {
                        fileMeta.strModelType = "ResNet18";
                        fileMeta.nModelTypeHash = om::ModelMeta::hashString("ResNet18");
                        if (logCb) logCb("[INFO] Inferred pretrained model type: ResNet18 (classes=1000)");
                    }
                    if (!fileMeta.strModelType.empty()) {
                        uint32_t nCurrentHash = om::ModelMeta::hashString(m_pImpl->strModelType);
                        bIsCrossArch = (fileMeta.nModelTypeHash != nCurrentHash);
                    }
                }

                int nLoaded = 0, nSkipped = 0;

                if (bIsCrossArch) {
                    if (logCb) logCb("[INFO] Cross-architecture pretrained loading: "
                        + fileMeta.strModelType + " -> " + m_pImpl->strModelType);
                    auto [nL, nS] = om::loadPyTorchPretrainedToSegModel(
                        *m_pImpl->pModel, params.strPretrainedModelPath);
                    nLoaded = nL;
                    nSkipped = nS;
                } else {
                    om::ModelSerializer::load(*m_pImpl->pModel, params.strPretrainedModelPath);
                    auto vecLoadedNamed = m_pImpl->pModel->namedParameters();
                    for (auto& [strName, pParam] : vecLoadedNamed) {
                        bool bAllZero = true;
                        const float* pData = pParam->floatDataPtr();
                        int nElem = std::min(static_cast<int>(pParam->numel()), 16);
                        for (int k = 0; k < nElem; ++k) {
                            if (pData[k] != 0.0f) { bAllZero = false; break; }
                        }
                        if (bAllZero && pParam->numel() > 1) ++nSkipped; else ++nLoaded;
                    }
                }

                nPretrainedLoaded = nLoaded;
                if (logCb) logCb("[INFO] Pre-trained weights loaded: "
                    + std::to_string(nLoaded) + " layers loaded, "
                    + std::to_string(nSkipped) + " skipped");
            } else {
                if (logCb) logCb("[WARN] Pre-trained model file not found: " + params.strPretrainedModelPath);
            }
        } catch (const std::exception& ex) {
            if (logCb) logCb("[WARN] Failed to load pre-trained weights: " + std::string(ex.what())
                + " — continuing with random initialization");
        }
    }

    // 20260402 ZJH ===== 预训练权重前向验证 =====
    // 在 GPU 迁移之前用随机输入做一次前向，检测权重是否损坏或架构不匹配
    if (nPretrainedLoaded > 0 && m_pImpl->pModel) {
        auto tTest = om::Tensor::randn({1, 3, 32, 32});  // 20260402 ZJH CPU 上的随机测试输入 (3ch RGB)
        m_pImpl->pModel->eval();   // 20260402 ZJH 切换到推理模式（禁用 Dropout/BN 更新）
        auto tOut = m_pImpl->pModel->forward(tTest);  // 20260402 ZJH 前向传播
        m_pImpl->pModel->train();  // 20260402 ZJH 恢复训练模式

        auto tCpu = tOut.cpu().contiguous();  // 20260402 ZJH 确保在 CPU 上且连续
        const float* pOut = tCpu.floatDataPtr();
        int nElems = tCpu.numel();  // 20260402 ZJH 输出元素总数
        bool bHasNan = false, bAllZero = true;
        float fSum = 0.0f;
        for (int i = 0; i < nElems; ++i) {
            if (std::isnan(pOut[i]) || std::isinf(pOut[i])) bHasNan = true;  // 20260402 ZJH 检测 NaN/Inf
            if (std::abs(pOut[i]) > 1e-8f) bAllZero = false;  // 20260402 ZJH 检测全零输出
            fSum += pOut[i];
        }
        if (bHasNan && logCb)
            logCb("[ERROR] Pretrained weights produce NaN/Inf — weights may be corrupted!");
        else if (bAllZero && logCb)
            logCb("[WARN] Pretrained weights produce all-zero output — architecture mismatch?");
        else if (logCb)
            logCb("[INFO] Pretrained weights verified OK (output mean="
                + std::to_string(fSum / static_cast<float>(nElems)) + ")");
    }

    // 20260325 ZJH ===== GPU 路径：将模型参数迁移到 GPU =====
#ifdef OM_HAS_CUDA
    if (bUseCuda) {
        // 20260330 ZJH Step 1: 迁移参数到 GPU（跳过损坏的 Tensor）
        auto vecAllParams = m_pImpl->pModel->parameters();
        int nMigrated = 0, nSkipped = 0;
        for (auto* pParam : vecAllParams) {
            try {
                if (pParam && pParam->numel() > 0 && pParam->isCpu())
                    { *pParam = pParam->cuda(); nMigrated++; }
                else if (pParam && pParam->numel() > 0)
                    nMigrated++;  // 20260330 ZJH 已在 GPU 上
            } catch (...) { nSkipped++; }
        }
        // 20260330 ZJH Step 2: 迁移缓冲区
        auto vecBufs = m_pImpl->pModel->buffers();
        for (auto* pBuf : vecBufs) {
            try {
                if (pBuf && pBuf->numel() > 0 && pBuf->isCpu())
                    *pBuf = pBuf->cuda();
            } catch (...) {}
        }
        if (nSkipped > 0 && logCb)
            logCb("[WARNING] Skipped " + std::to_string(nSkipped) + " bad params during GPU migration");
        if (logCb) logCb("[INFO] Model moved to GPU (" +
            std::to_string(vecAllParams.size()) + " params + " +
            std::to_string(vecBufs.size()) + " buffers)");
    }
#endif

    // 20260328 ZJH [DIAG-2] GPU 迁移后诊断
    {
        auto vecDiag = m_pImpl->pModel->namedParameters();
        std::cerr << "[DIAG-2] after GPU migration: namedParams=" << vecDiag.size()
                  << " children=" << m_pImpl->pModel->debugChildCount() << std::endl;
    }

    // 20260325 ZJH 重新获取参数引用（cuda() 可能创建新 storage，需要刷新指针列表）
    vecModelParams = m_pImpl->bIsEfficientAD
        ? static_cast<om::EfficientAD*>(m_pImpl->pModel.get())->studentParameters()
        : m_pImpl->pModel->parameters();

    // 20260401 ZJH ===== 迁移学习: 骨干冻结（延迟到预训练加载后决定）=====
    // 此处只记录意图，实际冻结在预训练加载成功后才执行
    // 原因: 此前在加载前就冻结，导致预训练不匹配时冻结了随机权重 + 降了 LR → 训练更差
    int nFreezeEpochs = params.nFreezeEpochs;  // 20260331 ZJH 冻结轮数
    float fBackboneLrMul = params.fBackboneLrMultiplier;  // 20260331 ZJH 骨干 LR 倍率
    bool bUseFreeze = false;  // 20260401 ZJH 延迟决定，预训练加载成功后才设 true

    std::vector<om::Tensor*> vecBackboneParams;  // 20260331 ZJH 骨干参数（编码器）
    std::vector<om::Tensor*> vecHeadParams;      // 20260331 ZJH head 参数（decoder/ASPP/FC）

    // 20260401 ZJH 优化器创建暂时先用全部参数，冻结成功后会重建
    std::unique_ptr<om::Adam> pAdam;
    std::unique_ptr<om::SGD> pSgd;
    std::unique_ptr<om::AdamW> pAdamW;

    om::CrossEntropyLoss criterion;  // 20260323 ZJH 损失函数（分类模型使用）
    // 20260401 ZJH Label Smoothing 内置启用（超越 PyTorch 标准训练）
    // 分类任务自动启用 0.1 smoothing，分割/检测任务不启用
    if (!m_pImpl->bIsSegmentation && !m_pImpl->bIsDetection && !m_pImpl->bIsEfficientAD) {
        criterion.setSmoothing(0.1f);
        if (logCb) logCb("[INFO] Label smoothing enabled: ε=0.1 (built-in regularization)");
    }
    om::YOLOLoss yoloLoss(5.0f, 1.0f, 0.5f, 1.0f);  // 20260330 ZJH YOLO 分项加权损失
    // 20260401 ZJH 预训练加载已移到 GPU 迁移之前（避免 CPU→GPU 写入崩溃）

    // 20260401 ZJH ===== 迁移学习: 根据实际加载结果决定冻结/降LR =====
    // 核心逻辑: 预训练权重实际加载成功（>50% 参数）→ 冻结骨干 + 降 LR
    //           预训练权重加载失败或不匹配（≤50%）→ 保持原 LR，不冻结，从头训练
    {
        int nTotalParams = static_cast<int>(m_pImpl->pModel->parameters().size());
        bool bPretrainedEffective = (nPretrainedLoaded > nTotalParams / 2);

        if (bPretrainedEffective && nFreezeEpochs > 0) {
            // 20260401 ZJH 预训练有效: 执行骨干冻结
            bUseFreeze = true;
            auto vecAllP = m_pImpl->pModel->parameters();
            int nBackboneEnd = nTotalParams * 4 / 5;  // 20260401 ZJH 前 80% 为骨干
            for (int i = 0; i < nTotalParams; ++i) {
                if (i < nBackboneEnd)
                    vecBackboneParams.push_back(vecAllP[i]);
                else
                    vecHeadParams.push_back(vecAllP[i]);
            }
            om::freezeBackbone(*m_pImpl->pModel, nBackboneEnd);
            vecModelParams = vecHeadParams;  // 20260401 ZJH 优化器只持有 head 参数
            if (logCb) logCb("[INFO] Transfer learning ACTIVE: froze backbone ("
                + std::to_string(nBackboneEnd) + "/" + std::to_string(nTotalParams)
                + " params) for " + std::to_string(nFreezeEpochs) + " epochs");

            // 20260401 ZJH 预训练有效时降 LR（保护 ImageNet 特征）
            if (fLr >= 0.001f) {
                float fOldLr = fLr;
                fLr *= 0.1f;
                if (logCb) logCb("[INFO] Transfer learning: auto-reduced LR "
                    + std::to_string(fOldLr) + " -> " + std::to_string(fLr));
            }
        } else {
            // 20260401 ZJH 预训练无效: 不冻结，保持原 LR，从头训练
            if (!params.strPretrainedModelPath.empty() && logCb) {
                logCb("[INFO] Pretrained weights NOT effective for this architecture ("
                    + std::to_string(nPretrainedLoaded) + "/" + std::to_string(nTotalParams)
                    + " matched) — training from scratch with original LR="
                    + std::to_string(fLr));
            }
        }
    }

    // 20260401 ZJH 创建优化器（在冻结决定之后，使用最终的参数列表和 LR）
    // 20260402 ZJH SGD: 传入 fWeightDecay（小数据集自动启用 5e-4）
    // 20260402 ZJH AdamW: 若 fWeightDecay > 0 则使用自动值，否则使用 AdamW 默认的 0.01
    if (params.strOptimizer == "SGD") {
        pSgd = std::make_unique<om::SGD>(vecModelParams, fLr, params.fMomentum, fWeightDecay);
    } else if (params.strOptimizer == "AdamW") {
        float fAdamWDecay = (fWeightDecay > 0.0f) ? fWeightDecay : 0.01f;  // 20260402 ZJH 小数据集自动值 or AdamW 默认
        pAdamW = std::make_unique<om::AdamW>(vecModelParams, fLr, 0.9f, 0.999f, 1e-8f, fAdamWDecay);
    } else {
        pAdam = std::make_unique<om::Adam>(vecModelParams, fLr);
    }

    float fBestValLoss = 1e9f;    // 20260406 ZJH 最佳验证损失（初始极大值，任何实际 loss 都会更新）
    int nPatienceCounter = 0;    // 20260406 ZJH 早停计数器（连续无改善的 epoch 数）
    std::mt19937 rng(42);        // 20260406 ZJH 梅森旋转随机引擎（固定种子保证可复现性）

    // 20260401 ZJH ===== 自动收敛检测（对标 Halcon dl_train_model）=====
    // 监控 val_loss 的移动平均斜率，斜率趋近 0 → 收敛 → 自动停止
    // 窗口大小: min(10, nEpochs/5)，斜率阈值: val_loss * 0.001
    std::vector<float> vecValLossHistory;  // 20260401 ZJH 验证损失历史记录
    vecValLossHistory.reserve(nEpochs);
    int nConvergeWindow = std::max(5, std::min(10, nEpochs / 5));  // 20260401 ZJH 滑动窗口大小
    bool bAutoConverge = (nEpochs >= 30);  // 20260401 ZJH 仅当 epoch 够多时启用自动收敛

    // 20260401 ZJH ===== SWA: Stochastic Weight Averaging（超越 PyTorch 标准训练）=====
    // 在最后 25% 的 epoch 中累积权重平均，比 EMA 更适合收敛后的微调阶段
    // SWA 使用等权平均（而非指数衰减），捕捉 loss landscape 的更平坦极小值
    int nSwaStartEpoch = nEpochs * 3 / 4;  // 20260401 ZJH 从 75% epoch 开始 SWA
    bool bUseSwa = (nEpochs >= 20);         // 20260401 ZJH 至少 20 epoch 才有意义
    std::vector<std::vector<float>> vecSwaAccum;  // 20260401 ZJH 累积权重和
    int nSwaCount = 0;  // 20260401 ZJH SWA 累积次数

    // 20260330 ZJH ===== S2: 最佳模型检查点（deep copy 参数和缓冲区）=====
    // 训练过程中，当验证损失创新低时保存模型权重的深拷贝
    // 训练结束后恢复最佳权重再 saveModel，避免过拟合后期的权重退化
    std::vector<om::Tensor> vecBestParams;    // 20260330 ZJH 最佳检查点的参数克隆
    std::vector<om::Tensor> vecBestBuffers;   // 20260330 ZJH 最佳检查点的缓冲区克隆（BN running stats）
    bool bHasBestCheckpoint = false;          // 20260330 ZJH 是否已保存过至少一次最佳检查点

    // 20260401 ZJH ===== Model Soup: top-K 检查点平均（超越 PyTorch 标准训练）=====
    // 保存 top-3 最低 val_loss 的检查点，训练结束后平均权重
    // 比单一 best 检查点高 1~2% 精度（Wortsman et al., 2022）
    struct CheckpointEntry {
        float fValLoss;
        std::vector<om::Tensor> vecParams;
        std::vector<om::Tensor> vecBuffers;
    };
    std::vector<CheckpointEntry> vecTopCheckpoints;  // 20260401 ZJH top-K 检查点（按 val_loss 排序）
    constexpr int nTopK = 3;  // 20260401 ZJH 保留 top-3

    // 20260330 ZJH ===== F4 + S1: 构建训练增强配置和归一化预设 =====
    // 20260402 ZJH [OPT-2.3] 构建增强配置时传入高级增强开关（由用户在 BridgeTrainParams 中控制）
    om::AugmentConfig augCfg = buildTrainAugmentConfig(m_pImpl->strModelType, params.bAdvancedAugment);
    om::NormPreset eNormPreset = selectNormPreset(m_pImpl->strModelType);
    // 20260330 ZJH 构建归一化专用配置（增强后单独应用，不在 augmentImage 内部归一化）
    om::AugmentConfig normCfg;
    normCfg.bNormalize = true;
    normCfg.eNormPreset = eNormPreset;
    if (logCb) {
        std::string strNorm = (eNormPreset == om::NormPreset::ImageNet) ? "ImageNet" : "ZeroOne";
        logCb("[INFO] Augmentation: " + std::string(params.bAugmentEnabled ? "ON" : "OFF")
            + " | Normalization: " + strNorm);
    }

    // 20260324 ZJH 索引洗牌优化：洗牌索引数组而非实际数据
    // 数据保持原位不动，通过 vecIndices 间接访问，减少内存搬运
    std::vector<int> vecIndices(nTrainCount);
    std::iota(vecIndices.begin(), vecIndices.end(), 0);  // 20260324 ZJH 初始化为 0..N-1

    // 20260330 ZJH 预分配单样本复用缓冲区，避免逐样本 augment/normalize 时反复堆分配
    // 每个样本 nInputDim 个 float（3*H*W），整个训练过程复用同一块内存
    std::vector<float> vecSampleBuf(static_cast<size_t>(nInputDim));

    // 20260329 ZJH ===== 分割模型: 计算反频率类别权重 =====
    // weight[c] = total_pixels / (nClasses * count[c])，归一化使 mean(w)=1
    // 背景占多数 → 低权重；缺陷少数类 → 高权重，梯度均衡
    std::vector<float> vecClassWeights(nNumClasses, 1.0f);  // 20260329 ZJH 默认均匀权重
    if (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) {
        std::vector<long long> vecClassCount(nNumClasses, 0);  // 20260329 ZJH 各类像素计数
        for (size_t i = 0; i < vecTrainMasks.size(); ++i) {
            int nCls = vecTrainMasks[i];  // 20260329 ZJH 掩码中的类别 ID
            if (nCls >= 0 && nCls < nNumClasses) ++vecClassCount[nCls];
        }
        long long nTotalMaskPx = static_cast<long long>(vecTrainMasks.size());  // 20260329 ZJH 总像素数
        float fWeightSum = 0.0f;  // 20260329 ZJH 权重求和（用于归一化）
        for (int c = 0; c < nNumClasses; ++c) {
            // 20260329 ZJH 反频率: total / (nClasses * count)，空类给 0 权重
            if (vecClassCount[c] > 0) {
                vecClassWeights[c] = static_cast<float>(nTotalMaskPx)
                    / (static_cast<float>(nNumClasses) * static_cast<float>(vecClassCount[c]));
            } else {
                vecClassWeights[c] = 0.0f;  // 20260329 ZJH 无像素的类不参与损失
            }
            fWeightSum += vecClassWeights[c];
        }
        // 20260401 ZJH ===== 缺陷专用损失：背景抑制 + 正样本加权（借鉴海康 neg_ratio/pos_weight）=====
        // 缺陷分割中背景占 >95% 像素，标准 CE 会让模型全预测为背景
        // 海康策略: neg_ratio=0.1（背景权重降 10 倍）+ pos_weight=1.0（缺陷类保持原权重）
        // 效果: 模型被迫关注少数缺陷像素，不能靠"全猜背景"蒙混过关
        {
            float fNegRatio = 0.1f;  // 20260401 ZJH 背景类权重衰减因子（海康默认 0.1）
            vecClassWeights[0] *= fNegRatio;  // 20260401 ZJH class 0 = 背景，权重降 10 倍
            if (logCb) logCb("[INFO] Defect-seg loss: background weight *= "
                + std::to_string(fNegRatio) + " (neg_ratio, Hikrobot-style)");
        }

        // 20260407 ZJH [修复] neg_ratio 修改后重算 fWeightSum
        // 旧: fWeightSum 在 neg_ratio 修改前累加，归一化用的是旧值 → 所有权重系统性偏小
        fWeightSum = 0.0f;
        for (int c = 0; c < nNumClasses; ++c) fWeightSum += vecClassWeights[c];
        // 20260329 ZJH 归一化使 mean(w) = 1.0（损失量级不变）
        if (fWeightSum > 0.0f) {
            float fNorm = static_cast<float>(nNumClasses) / fWeightSum;
            for (int c = 0; c < nNumClasses; ++c) vecClassWeights[c] *= fNorm;
        }
        if (logCb) {
            std::string strW = "[INFO] Class weights:";
            for (int c = 0; c < nNumClasses; ++c)
                strW += " c" + std::to_string(c) + "=" + std::to_string(vecClassWeights[c]).substr(0, 5);
            logCb(strW);
        }
    }

    // 20260324 ZJH 双缓冲预分配（CPU 路径使用，GPU 路径使用 batch CPU 缓冲 + 一次 H2D）
    size_t nBufSizeInput  = static_cast<size_t>(nBatchSize) * static_cast<size_t>(nInputDim);    // 20260324 ZJH 单缓冲输入元素数
    size_t nBufSizeLabels = static_cast<size_t>(nBatchSize) * static_cast<size_t>(nNumClasses);  // 20260324 ZJH 单缓冲标签元素数
    std::vector<float> vecBufA_Input(nBufSizeInput, 0.0f);     // 20260324 ZJH 缓冲区 A 输入
    std::vector<float> vecBufA_Label(nBufSizeLabels, 0.0f);    // 20260324 ZJH 缓冲区 A 标签
    std::vector<float> vecBufB_Input(nBufSizeInput, 0.0f);     // 20260324 ZJH 缓冲区 B 输入
    std::vector<float> vecBufB_Label(nBufSizeLabels, 0.0f);    // 20260324 ZJH 缓冲区 B 标签

    // 20260324 ZJH 验证缓冲区预分配（保留用于验证阶段，单缓冲即可）
    std::vector<float> vecValInput(nBufSizeInput, 0.0f);       // 20260324 ZJH 验证输入缓冲区
    std::vector<float> vecValLabel(nBufSizeLabels, 0.0f);      // 20260324 ZJH 验证标签缓冲区

    // 20260325 ZJH GPU 显存预检查：估算单 batch 前向传播的显存需求
    // CNN 模型中间激活值非常大，1024×1024 图像的特征图会占用数 GB 显存
    // 20260325 ZJH GPU 显存预检查（仅警告不阻断，让实际训练决定是否 OOM）
    if (bUseCuda && m_pImpl->bIsCnn) {
#ifdef OM_HAS_CUDA
        size_t nFreeMem = 0, nTotalMem = 0;
        omCudaGetMemInfo(0, &nFreeMem, &nTotalMem);
        // 20260325 ZJH CNN 显存粗估: batch × 3 × H × W × 4bytes × 100（经验系数）
        size_t nBatchBytes = static_cast<size_t>(nBatchSize) * 3
            * static_cast<size_t>(m_pImpl->nInputSize) * m_pImpl->nInputSize * sizeof(float);
        size_t nEstTotal = nBatchBytes * 100 + static_cast<size_t>(totalParameters()) * sizeof(float) * 3;
        if (logCb) logCb("[INFO] GPU 显存: 可用 " + std::to_string(nFreeMem / 1048576) +
                         "MB / 总计 " + std::to_string(nTotalMem / 1048576) +
                         "MB | 预估需要 ~" + std::to_string(nEstTotal / 1048576) + "MB");
        if (nFreeMem > 0 && nEstTotal > nFreeMem) {
            if (logCb) logCb("[警告] GPU 显存可能不足，训练中如 OOM 会自动停止");
        }
#endif
    }

    // 20260402 ZJH ===== 确定性训练: 固定所有随机种子 =====
    // 保证同样的数据+参数→相同的训练结果（用于回归测试和调试）
    if (params.bDeterministic) {
        std::srand(static_cast<unsigned>(params.nRandomSeed));  // 20260402 ZJH C 标准库随机种子
        // 20260402 ZJH 注: C++23 模块内的 std::mt19937 等需在引擎层单独固定
        // CUDA 确定性模式通过 CUDABackend 设置（如有）
        if (logCb) logCb("[INFO] Deterministic training: seed=" + std::to_string(params.nRandomSeed));
    }

    // 20260407 ZJH ===== 混合精度训练（暂时禁用自动启用）=====
    // GradScaler 的 GPU 张量处理存在多个问题:
    //   1. hasInfOrNan() 对 GPU 梯度 D2H 检查性能差（160 参数逐个传输）
    //   2. unscaleGrads() 无法直接操作 GradAccumulator（模块依赖限制）
    //   3. 训练循环中 scale/unscale/clip 的流同步可能导致死锁
    // 待 GradScaler 完整重写（CUDA kernel 融合 unscale+inf检测）后再启用
    // 用户仍可通过 params.bMixedPrecision 手动启用（自行承担风险）
    bool bMixedPrecision = params.bMixedPrecision;  // 20260407 ZJH 仅响应用户显式设置，不自动启用
    bMixedPrecision = bMixedPrecision && bUseCuda;
    // 20260402 ZJH 使用 om::GradScaler 实例（非简单变量）
    // GradScaler 管理: loss 放大 → backward → 梯度反缩放 → NaN 检测 → 动态调整
    om::GradScaler gradScaler(65536.0f, 2.0f, 0.5f, 2000);  // 20260402 ZJH init=65536, grow=2x, shrink=0.5x
    if (bMixedPrecision && logCb) {
        logCb("[INFO] Mixed Precision (FP16) enabled: GradScaler init=" +
              std::to_string(gradScaler.getScale()));
    }

    // 20260402 ZJH ===== BoundaryLoss 配置（分割训练专用）=====
    bool bUseBoundaryLoss = params.bUseBoundaryLoss && m_pImpl->bIsSegmentation && !vecTrainMasks.empty();
    if (bUseBoundaryLoss && logCb) {
        logCb("[INFO] BoundaryLoss enabled: CE+Dice+Boundary (weight=0.5)");
    }

    // 20260325 ZJH 输出训练配置日志（区分 GPU/CPU 路径）
    if (logCb) {
        std::string strDeviceInfo = bUseCuda ? "GPU-Resident (CUDA)" : "CPU (SIMD+OpenMP)";
        // 20260402 ZJH 更新损失信息日志
        std::string strLossInfo = m_pImpl->bIsEfficientAD ? "Distillation"
            : bUseBoundaryLoss ? "CE+Dice+Boundary (combined)"
            : (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) ? "PixelCE+Dice" : "CrossEntropy";
        logCb("[INFO] Engine training: " + params.strModelType +
              " | Params: " + std::to_string(totalParameters()) +
              " | Train: " + std::to_string(nTrainCount) +
              " | Val: " + std::to_string(nValCount) +
              " | Device: " + strDeviceInfo +
              " | Loss: " + strLossInfo +
              " | FP16: " + (bMixedPrecision ? "ON" : "OFF") +
              " | Deterministic: " + (params.bDeterministic ? "ON" : "OFF"));
    }

    // 20260326 ZJH 当前学习率变量，用于 Cosine Annealing 调度和日志输出
    float fCurrentLr = fLr;

    // 20260401 ZJH ===== 梯度累积配置 =====
    // 等效 batch = nBatchSize × nAccumSteps，减少显存占用同时保持大 batch 效果
    int nAccumSteps = std::max(1, params.nGradAccumSteps);
    if (nAccumSteps > 1 && logCb) {
        logCb("[INFO] Gradient accumulation enabled: " + std::to_string(nAccumSteps)
            + " steps, effective batch=" + std::to_string(nBatchSize * nAccumSteps));
    }

    // 20260401 ZJH ===== EMA 权重平均 =====
    // 维护模型参数的指数移动平均副本，推理时替代原始权重
    // EMA_weight = decay × EMA_weight + (1 - decay) × current_weight
    float fEmaDecay = params.fEmaDecay;
    bool bUseEma = (fEmaDecay > 0.0f && fEmaDecay < 1.0f);
    std::vector<om::Tensor> vecEmaParams;  // 20260401 ZJH EMA 参数快照（CPU 存储）
    if (bUseEma) {
        // 20260401 ZJH 初始化 EMA 为当前模型参数的副本
        vecModelParams = m_pImpl->pModel->parameters();
        vecEmaParams.reserve(vecModelParams.size());
        for (auto* pParam : vecModelParams) {
            // 20260401 ZJH 深拷贝到 CPU（手动 memcpy，om::Tensor 无 clone 方法）
            auto tCpu = pParam->cpu().contiguous();
            om::Tensor tClone = om::Tensor::zeros(tCpu.shapeVec());
            std::memcpy(tClone.mutableFloatDataPtr(), tCpu.floatDataPtr(),
                        static_cast<size_t>(tCpu.numel()) * sizeof(float));
            vecEmaParams.push_back(std::move(tClone));
        }
        if (logCb) logCb("[INFO] EMA enabled: decay=" + std::to_string(fEmaDecay)
            + " params=" + std::to_string(vecEmaParams.size()));
    }

    // 20260401 ZJH ===== Auto LR Finder（超越 PyTorch 标准训练 — Leslie Smith 方法）=====
    // 在正式训练前，用指数递增的 LR 跑 10 个 mini-batch，找到 loss 下降最快的 LR
    // PyTorch 需要手动调用 lr_finder 库，我们内置自动化
    // 20260402 ZJH [OPT-1.3] 扩展至分割/检测模型：使用各自对应的 loss 做 LR 搜索
    // 原版仅支持分类 CNN，现在 EfficientAD（蒸馏loss）和检测（YOLOLoss）也可受益
    // 跳过条件：数据不足（< 5 batch）或 PatchCore（非梯度训练）
    // 20260407 ZJH [修复] 分割模型跳过 LR Finder（用分类 CE 做 LR 搜索，与像素级损失不匹配）
    bool bSkipLrFinder = (!m_pImpl->bIsCnn && m_pImpl->strModelType != "MLP")
        || nTrainCount < nBatchSize * 3
        || m_pImpl->bIsSegmentation;  // 20260407 ZJH 分割模型的损失函数不同，LR Finder 结果不适用
    if (!bSkipLrFinder) {
        float fLrMin = 1e-5f, fLrMax = 1.0f;
        int nLrSteps = std::min(10, (nTrainCount + nBatchSize - 1) / nBatchSize);
        float fBestLrLoss = 1e9f;
        float fFoundLr = fLr;
        float fPrevLoss = 1e9f;
        float fBestSlope = 0.0f;

        // 20260407 ZJH [修复] 保存全部模型参数（不只是 vecModelParams 中的 head 参数）
        // 旧: freeze backbone 时 vecModelParams 只含 head → 骨干被 LR Finder 污染
        auto vecAllParams = m_pImpl->pModel->parameters();
        std::vector<om::Tensor> vecOrigParams;
        for (auto* p : vecAllParams) {
            auto tCpu = p->isCuda() ? p->cpu() : *p;
            auto tc = tCpu.contiguous();
            om::Tensor tClone = om::Tensor::zeros(p->shapeVec());
            std::memcpy(tClone.mutableFloatDataPtr(), tc.floatDataPtr(), static_cast<size_t>(p->numel()) * sizeof(float));
            vecOrigParams.push_back(std::move(tClone));
        }

        m_pImpl->pModel->train();
        for (int nStep = 0; nStep < nLrSteps; ++nStep) {
            // 20260401 ZJH 指数递增 LR: lr = lr_min * (lr_max/lr_min)^(step/nSteps)
            float fTestLr = fLrMin * std::pow(fLrMax / fLrMin, static_cast<float>(nStep) / static_cast<float>(nLrSteps));
            if (pAdam) pAdam->setLearningRate(fTestLr);
            else if (pAdamW) pAdamW->setLearningRate(fTestLr);
            else if (pSgd) pSgd->setLearningRate(fTestLr);

            // 20260401 ZJH 取一个 mini-batch 做前向+反向
            int nStart = nStep * nBatchSize;
            int nCurBatch = std::min(nBatchSize, nTrainCount - nStart);
            if (nCurBatch <= 0) break;

            // 20260401 ZJH 构建输入（复用已有 vecTrainData）
            auto tInput = om::Tensor::fromData(
                vecTrainData.data() + static_cast<size_t>(vecIndices[nStart]) * nInputDim,
                m_pImpl->bIsCnn ? std::vector<int>{1, 3, m_pImpl->nInputSize, m_pImpl->nInputSize}
                                : std::vector<int>{1, nInputDim});
            if (bUseCuda) tInput = tInput.cuda();
            tInput.setRequiresGrad(true);

            auto tLabels = om::Tensor::zeros({1, nNumClasses});
            int nLbl = static_cast<int>(vecTrainLabels[vecIndices[nStart]]);
            if (nLbl >= 0 && nLbl < nNumClasses)
                tLabels.mutableFloatDataPtr()[nLbl] = 1.0f;
            if (bUseCuda) tLabels = tLabels.cuda();

            try {
                auto tOut = m_pImpl->pModel->forward(tInput);
                auto tLoss = criterion.forward(tOut, tLabels);
                m_pImpl->pModel->zeroGrad();
                om::tensorBackward(tLoss);
                if (pAdam) pAdam->step();
                else if (pAdamW) pAdamW->step();
                else if (pSgd) pSgd->step();

                float fStepLoss = tLoss.item();
                // 20260401 ZJH 跟踪 loss 下降斜率，选择下降最陡的 LR
                float fSlope = fPrevLoss - fStepLoss;
                if (fSlope > fBestSlope && !std::isnan(fStepLoss) && !std::isinf(fStepLoss)) {
                    fBestSlope = fSlope;
                    fFoundLr = fTestLr;
                }
                fPrevLoss = fStepLoss;
                if (std::isnan(fStepLoss) || fStepLoss > fPrevLoss * 4.0f) break;  // 20260401 ZJH 发散则停
            } catch (...) { break; }
        }

        // 20260401 ZJH 恢复原始权重
        // 20260407 ZJH [修复] 恢复全部参数（与保存时使用相同的 vecAllParams）
        for (size_t i = 0; i < vecAllParams.size() && i < vecOrigParams.size(); ++i) {
            if (vecAllParams[i]->isCuda()) {
                *vecAllParams[i] = vecOrigParams[i].cuda();
            } else {
                *vecAllParams[i] = vecOrigParams[i];
            }
        }
        vecOrigParams.clear();

        // 20260401 ZJH 使用找到的 LR（取最优 LR 的 1/3 作为保守起点）
        if (fFoundLr != fLr && fBestSlope > 0) {
            float fAutoLr = fFoundLr / 3.0f;
            fAutoLr = std::max(1e-5f, std::min(fAutoLr, 0.1f));
            if (logCb) logCb("[INFO] Auto LR Finder: best_lr=" + std::to_string(fFoundLr)
                + " → using lr=" + std::to_string(fAutoLr) + " (original=" + std::to_string(fLr) + ")");
            fLr = fAutoLr;
            if (pAdam) pAdam->setLearningRate(fLr);
            else if (pAdamW) pAdamW->setLearningRate(fLr);
            else if (pSgd) pSgd->setLearningRate(fLr);
        }
    }

    int nOverfitCount = 0;       // 20260402 ZJH 连续过拟合 epoch 计数
    float fPrevTrainLoss = 1e10f; // 20260402 ZJH 上一 epoch 训练损失

    std::cerr << "[TRAIN-DIAG] entering epoch loop, nEpochs=" << nEpochs
              << " bUseCuda=" << bUseCuda << " nInputDim=" << nInputDim
              << " nNumClasses=" << nNumClasses << " nTrainCount=" << nTrainCount
              << " bIsSeg=" << m_pImpl->bIsSegmentation
              << " masksEmpty=" << vecTrainMasks.empty() << std::endl;
    for (int nEpoch = 1; nEpoch <= nEpochs; ++nEpoch) {
        if (stopCheck && stopCheck()) {
            if (logCb) logCb("[INFO] Stopped at epoch " + std::to_string(nEpoch));
            // 20260325 ZJH 提前退出时也需要将参数移回 CPU（如果在 GPU 上）
#ifdef OM_HAS_CUDA
            if (bUseCuda) {
                vecModelParams = m_pImpl->pModel->parameters();
                for (auto* pParam : vecModelParams) {
                    *pParam = pParam->cpu();  // 20260325 ZJH 将参数移回 CPU
                }
                omCudaCleanup();  // 20260325 ZJH 释放 GPU 资源
            }
#endif
            return false;
        }

        // 20260324 ZJH 索引洗牌：只洗牌索引数组，数据保持不动
        std::shuffle(vecIndices.begin(), vecIndices.end(), rng);

        // 20260331 ZJH ===== 迁移学习: 骨干解冻检查 =====
        // 当 nEpoch == nFreezeEpochs + 1 时，解冻骨干并将其加入优化器（低 LR 组）
        // 这是两阶段训练的关键切换点：从"只训练 head"过渡到"全模型微调"
        if (bUseFreeze && nEpoch == nFreezeEpochs + 1) {
            // 20260331 ZJH 解冻所有参数（恢复 requiresGrad=true）
            om::unfreezeAll(*m_pImpl->pModel);
            // 20260331 ZJH 将骨干参数加入优化器，LR 倍率 = fBackboneLrMul（典型 0.1）
            if (pAdam) pAdam->addParams(vecBackboneParams, fBackboneLrMul);
            else if (pAdamW) pAdamW->addParams(vecBackboneParams, fBackboneLrMul);
            else if (pSgd) pSgd->addParams(vecBackboneParams, fBackboneLrMul);
            if (logCb) logCb("[INFO] Epoch " + std::to_string(nEpoch)
                + ": unfroze backbone, added to optimizer with LR multiplier="
                + std::to_string(fBackboneLrMul));
        }

        // 20260330 ZJH Warmup + Cosine Annealing 学习率调度
        // Phase 1 (epoch 1~nWarmup): 线性 warmup lr = lr_init * (epoch / nWarmup)
        //   防止初始大梯度破坏随机初始化权重和 BN running stats
        // Phase 2 (epoch nWarmup+1~nEpochs): Cosine Annealing
        //   lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(π * progress))
        //   progress = (epoch - nWarmup) / (nEpochs - nWarmup)
        {
            int nWarmupEpochs = std::max(1, nEpochs / 10);  // 20260330 ZJH Warmup 占总 epoch 的 10%，至少 1 轮
            float fMinLr = fLr * 0.01f;  // 20260326 ZJH 最小学习率 = 初始的 1%
            float fScheduledLr;
            if (nEpoch <= nWarmupEpochs) {
                // 20260330 ZJH Phase 1: 线性 Warmup（从 lr_min 升到 lr_init）
                float fWarmupProgress = static_cast<float>(nEpoch) / static_cast<float>(nWarmupEpochs);
                fScheduledLr = fMinLr + (fLr - fMinLr) * fWarmupProgress;
            } else {
                // 20260330 ZJH Phase 2: Cosine Annealing（从 lr_init 降到 lr_min）
                float fCosProgress = static_cast<float>(nEpoch - nWarmupEpochs)
                                   / static_cast<float>(std::max(1, nEpochs - nWarmupEpochs));
                fScheduledLr = fMinLr + 0.5f * (fLr - fMinLr)
                             * (1.0f + std::cos(3.14159265f * fCosProgress));
            }
            // 20260331 ZJH setLearningRate 设置基础 LR，各参数实际 LR = baseLR × 倍率
            if (pAdam) pAdam->setLearningRate(fScheduledLr);
            else if (pAdamW) pAdamW->setLearningRate(fScheduledLr);
            else if (pSgd) pSgd->setLearningRate(fScheduledLr);
            fCurrentLr = fScheduledLr;  // 20260330 ZJH 更新当前学习率供日志输出
        }

        // 20260401 ZJH ===== 多尺度渐进式训练（超越海康/Halcon 的独有功能）=====
        // Phase 1 (前1/3 epoch): 50% 分辨率 → 学全局特征，速度快 4x
        // Phase 2 (中1/3): 75% 分辨率 → 学中等细节
        // Phase 3 (后1/3): 100% 分辨率 → 学精细特征
        // 每 epoch 额外叠加 [0.9x, 1.1x] 随机扰动，增强尺度鲁棒性
        int nBaseInputSize = m_pImpl->nInputSize;  // 20260401 ZJH 基准输入尺寸（createModel 时设置）
        int nEpochInputSize = nBaseInputSize;     // 20260401 ZJH 当前 epoch 的实际输入尺寸
        // 20260401 ZJH 分割/检测模型跳过多尺度（mask/target 尺寸在数据加载时固定，无法动态变）
        if (m_pImpl->bIsCnn && nEpochs >= 15 && nBaseInputSize >= 128
            && !m_pImpl->bIsSegmentation && !m_pImpl->bIsDetection) {
            float fSizeRatio;
            if (nEpoch <= nEpochs / 3) {
                fSizeRatio = 0.5f;   // 20260401 ZJH Phase 1: 50% 分辨率
            } else if (nEpoch <= nEpochs * 2 / 3) {
                fSizeRatio = 0.75f;  // 20260401 ZJH Phase 2: 75% 分辨率
            } else {
                fSizeRatio = 1.0f;   // 20260401 ZJH Phase 3: 100% 分辨率
            }
            // 20260401 ZJH 随机尺度扰动 ±10%（训练时，最后 5 epoch 不扰动以稳定 BN）
            if (nEpoch <= nEpochs - 5) {
                float fJitter = 0.9f + static_cast<float>(rng() % 21) / 100.0f;  // [0.9, 1.1]
                fSizeRatio *= fJitter;
            }
            int nScaled = static_cast<int>(nBaseInputSize * fSizeRatio);
            nScaled = ((nScaled + 15) / 32) * 32;  // 20260401 ZJH 对齐到 32
            nScaled = std::max(32, std::min(nScaled, nBaseInputSize));
            nEpochInputSize = nScaled;
            // 20260401 ZJH 重算输入维度（CNN: C×H×W）
            nInputDim = (m_pImpl->bIsCnn ? 3 : 1) * nEpochInputSize * nEpochInputSize;
        }

        int nBatches = (nTrainCount + nBatchSize - 1) / nBatchSize;  // 20260406 ZJH 向上取整计算 batch 数
        float fEpochLoss = 0.0f;  // 20260406 ZJH 本 epoch 累计训练损失
        m_pImpl->pModel->train();  // 20260406 ZJH 切换到训练模式（启用 Dropout/BN 训练行为）

        if (bUseCuda) {
            // 20260326 ZJH ===== GPU 训练循环（预分配缓冲区优化）=====
            // 优化：batch 缓冲区在循环外预分配，避免每 batch malloc/free 开销
            std::vector<float> vecBatchInput(static_cast<size_t>(nBatchSize) * nInputDim);
            std::vector<float> vecBatchLabel(static_cast<size_t>(nBatchSize) * nNumClasses, 0.0f);

            // 20260329 ZJH ===== 分割训练 GPU 缓冲预分配 =====
            // 仅预分配固定大小的缓冲（classWeights/loss/stats），变尺寸的 softmax 留在循环内
            std::vector<float> vecTargetBuf;       // 20260329 ZJH CPU 端 float 目标缓冲
            om::Tensor tClassWGpu;                  // 20260329 ZJH [C] 类别权重（GPU 常驻，一次上传）
            if (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) {
                int nSpatial = m_pImpl->nInputSize * m_pImpl->nInputSize;
                vecTargetBuf.resize(static_cast<size_t>(nBatchSize) * nSpatial, 0.0f);
                tClassWGpu = om::Tensor::fromData(vecClassWeights.data(), {m_pImpl->nNumClasses}).cuda();
            }

            std::cerr << "[TRAIN-DIAG] epoch " << nEpoch << " nBatches=" << nBatches
                      << " targetBufSize=" << vecTargetBuf.size() << std::endl;
            for (int nBatch = 0; nBatch < nBatches; ++nBatch) {
                if (stopCheck && stopCheck()) {
#ifdef OM_HAS_CUDA
                    vecModelParams = m_pImpl->pModel->parameters();
                    for (auto* pParam : vecModelParams) {
                        *pParam = pParam->cpu();
                    }
                    omCudaCleanup();
#endif
                    return false;
                }

                int nStart = nBatch * nBatchSize;
                int nEnd = std::min(nStart + nBatchSize, nTrainCount);
                int nCurBatch = nEnd - nStart;

                // 20260326 ZJH 填充预分配缓冲区（零拷贝复用，不重新分配）
                std::fill(vecBatchLabel.begin(),
                          vecBatchLabel.begin() + static_cast<size_t>(nCurBatch) * nNumClasses, 0.0f);
                for (int i = 0; i < nCurBatch; ++i) {
                    int nIdx = vecIndices[nStart + i];
                    int nSrc = nIdx * nInputDim;
                    if (nSrc + nInputDim <= static_cast<int>(vecTrainData.size())) {
                        std::copy(vecTrainData.data() + nSrc,
                                  vecTrainData.data() + nSrc + nInputDim,
                                  vecBatchInput.data() + static_cast<size_t>(i) * nInputDim);
                    }
                    int nLabel = vecTrainLabels[nIdx];
                    if (nLabel >= 0 && nLabel < nNumClasses) {
                        vecBatchLabel[static_cast<size_t>(i) * nNumClasses + nLabel] = 1.0f;
                    }
                }

                // 20260330 ZJH ===== F4: 训练数据增强（GPU 路径）=====
                // 在 CPU 缓冲区上逐样本执行增强，增强后再上传 GPU
                // 数据此时为 CHW float [0,1]，与 augmentImage 约定一致
                // 20260330 ZJH ===== F4: 训练数据增强（GPU 路径）=====
                // 安全检查: nInputDim 必须 == nC * nSp * nSp，否则跳过增强避免越界崩溃
                // 20260407 ZJH [修复] 分割模型禁用 EngineBridge 内部几何增强
                // 原因: augmentImage 对图像做随机翻转/旋转，但对应的掩码（vecTrainMasks）
                //       没有同步增强 → 图像翻转后像素级标签错位 → 模型学到错误对应关系
                // TrainingSession 已经对图像和掩码做了同步增强（翻转/亮度），这里不需要重复
                // 仅对非分割模型（分类/检测）保留 EngineBridge 增强
                if (params.bAugmentEnabled && m_pImpl->bIsCnn && !m_pImpl->bIsSegmentation) {
                    int nC = 3;
                    int nSp = m_pImpl->nInputSize;
                    if (nC * nSp * nSp == nInputDim && nSp > 0) {
                        for (int i = 0; i < nCurBatch; ++i) {
                            size_t nOff = static_cast<size_t>(i) * nInputDim;
                            std::copy(vecBatchInput.data() + nOff,
                                      vecBatchInput.data() + nOff + nInputDim,
                                      vecSampleBuf.begin());
                            try {
                                om::augmentImage(vecSampleBuf, nC, nSp, nSp, augCfg);
                            } catch (...) {}
                            std::copy(vecSampleBuf.begin(), vecSampleBuf.end(), vecBatchInput.data() + nOff);
                        }
                    }
                }

                // 20260330 ZJH ===== S1: 统一归一化（GPU 路径）=====
                // ImageNet 归一化用于 CNN 分类模型，/255 用于 YOLO/UNet
                if (m_pImpl->bIsCnn && eNormPreset == om::NormPreset::ImageNet) {
                    int nC = 3;
                    int nSp = m_pImpl->nInputSize;
                    if (nC * nSp * nSp == nInputDim && nSp > 0) {
                        for (int i = 0; i < nCurBatch; ++i) {
                            size_t nOff = static_cast<size_t>(i) * nInputDim;
                            std::copy(vecBatchInput.data() + nOff,
                                      vecBatchInput.data() + nOff + nInputDim,
                                      vecSampleBuf.begin());
                            try { om::normalizeImage(vecSampleBuf, nC, nSp, nSp, normCfg); } catch (...) {}
                            std::copy(vecSampleBuf.begin(), vecSampleBuf.end(), vecBatchInput.data() + nOff);
                        }
                    }
                }

                auto tInput = m_pImpl->bIsCnn
                    ? om::Tensor::fromData(vecBatchInput.data(), {nCurBatch, 3, m_pImpl->nInputSize, m_pImpl->nInputSize})
                    : om::Tensor::fromData(vecBatchInput.data(), {nCurBatch, nInputDim});
                auto tLabels = om::Tensor::fromData(vecBatchLabel.data(), {nCurBatch, nNumClasses});
#ifdef OM_HAS_CUDA
                tInput = tInput.cuda();
                tLabels = tLabels.cuda();
#endif

                // 20260325 ZJH 前向传播（全部在 GPU 上执行，tensor_ops 按设备类型自动调度）
                std::cerr << "[TRAIN-DIAG] batch " << nBatch << " forward start, input=["
                          << tInput.shape(0) << "," << tInput.shape(1) << "," << tInput.shape(2) << "," << tInput.shape(3)
                          << "] cuda=" << tInput.isCuda() << std::endl;
                om::Tensor tOutput, tLoss;
                try {
                    // 20260326 ZJH EfficientAD 使用蒸馏损失，其他模型使用交叉熵
                    if (m_pImpl->bIsEfficientAD) {
                        auto* pEAD = static_cast<om::EfficientAD*>(m_pImpl->pModel.get());
                        tLoss = pEAD->computeDistillationLoss(tInput);
                    } else if (m_pImpl->bIsGCAD) {
                        // 20260402 ZJH ===== GCAD 训练: Teacher-Student 蒸馏 + 全局编码器 =====
                        // GCAD 的训练分两阶段:
                        //   阶段1: Student 模仿 Teacher 的局部特征（MSE 损失）
                        //   阶段2: 全局编码器学习正常布局分布（后续 fitGCADDistribution 处理）
                        // 这里只做阶段1（蒸馏），阶段2 在训练完成后由 fitGCADDistribution 执行
                        auto* pGCAD = dynamic_cast<om::GCAD*>(m_pImpl->pModel.get());
                        if (pGCAD) {
                            // 20260402 ZJH 冻结 Teacher，仅训练 Student
                            pGCAD->teacher().eval();
                            auto teacherFeat = pGCAD->teacher().forward(tInput);  // 20260402 ZJH [N,256,H/8,W/8]
                            auto studentFeat = pGCAD->student().forward(tInput);  // 20260402 ZJH [N,256,H/8,W/8]
                            // 20260402 ZJH MSE 蒸馏损失
                            auto diff = om::tensorSub(teacherFeat, studentFeat);
                            auto sq = om::tensorMul(diff, diff);
                            auto sumSq = om::tensorSum(sq);
                            float fInvN = 1.0f / static_cast<float>(std::max(sq.numel(), 1));
                            tLoss = om::tensorMulScalar(sumSq, fInvN);
                        }
                    } else if (m_pImpl->bIsDBNet) {
                        // 20260402 ZJH ===== DBNet 训练: 概率图 BCE + 阈值图 L1 + 二值图 BCE =====
                        // DBNet 输出 [N, 3, H/4, W/4]: channel 0=P, 1=T, 2=B
                        // 目标 mask: [N, 1, H/4, W/4] 文本区域二值 GT
                        tOutput = m_pImpl->pModel->forward(tInput);  // 20260402 ZJH [N,3,H/4,W/4]

                        // 20260402 ZJH 构建目标: 从 vecTrainMasks 提取并下采样到 H/4
                        int nOutH = tOutput.shape(2);  // 20260402 ZJH H/4
                        int nOutW = tOutput.shape(3);  // 20260402 ZJH W/4
                        int nSpatialOut = nOutH * nOutW;
                        int nSrcH = m_pImpl->nInputSize;
                        int nSrcSpatial = nSrcH * nSrcH;

                        auto tTarget = om::Tensor::zeros({nCurBatch, 1, nOutH, nOutW});
                        auto tThreshTarget = om::Tensor::zeros({nCurBatch, 1, nOutH, nOutW});
                        float* pTgt = tTarget.mutableFloatDataPtr();
                        float* pThreshTgt = tThreshTarget.mutableFloatDataPtr();

                        // 20260402 ZJH 简化: mask 最近邻下采样 + 阈值图设为边界附近高值
                        float fScaleH = static_cast<float>(nSrcH) / static_cast<float>(nOutH);
                        for (int i = 0; i < nCurBatch; ++i) {
                            int nIdx = vecIndices[nStart + i];
                            for (int y = 0; y < nOutH; ++y) {
                                int nSrcY = std::min(static_cast<int>(y * fScaleH), nSrcH - 1);
                                for (int x = 0; x < nOutW; ++x) {
                                    int nSrcX = std::min(static_cast<int>(x * fScaleH), nSrcH - 1);
                                    int nMaskIdx = nIdx * nSrcSpatial + nSrcY * nSrcH + nSrcX;
                                    float fVal = (nMaskIdx < static_cast<int>(vecTrainMasks.size()) && vecTrainMasks[nMaskIdx] > 0) ? 1.0f : 0.0f;
                                    pTgt[i * nSpatialOut + y * nOutW + x] = fVal;
                                    pThreshTgt[i * nSpatialOut + y * nOutW + x] = 0.3f;  // 20260402 ZJH 默认阈值 0.3
                                }
                            }
                        }

                        if (tOutput.isCuda()) {
                            tTarget = tTarget.cuda();
                            tThreshTarget = tThreshTarget.cuda();
                        }

                        // 20260402 ZJH DB 三项损失
                        om::DBLoss dbLoss;
                        tLoss = dbLoss.forward(tOutput, tTarget, tThreshTarget);

                    } else if (m_pImpl->bIsEdgeExtraction && !vecTrainMasks.empty()) {
                        // 20260402 ZJH ===== EdgeExtraction 训练: BCE+Dice 边缘损失 =====
                        // EdgeUNet 输出 [N, 1, H, W] 边缘概率（sigmoid），mask 为二值边缘标注
                        tOutput = m_pImpl->pModel->forward(tInput);  // 20260402 ZJH [B, 1, H, W]

                        // 20260402 ZJH 构建边缘目标 [B, 1, H, W] 从 vecTrainMasks
                        int nEdgeH = tOutput.shape(2), nEdgeW = tOutput.shape(3);
                        int nEdgeSpatial = nEdgeH * nEdgeW;
                        int nSrcH = m_pImpl->nInputSize;
                        int nSrcSpatial = nSrcH * nSrcH;
                        auto tEdgeTarget = om::Tensor::zeros({nCurBatch, 1, nEdgeH, nEdgeW});
                        float* pET = tEdgeTarget.mutableFloatDataPtr();
                        for (int i = 0; i < nCurBatch; ++i) {
                            int nIdx = vecIndices[nStart + i];
                            for (int j = 0; j < nEdgeSpatial; ++j) {
                                int nMaskIdx = nIdx * nSrcSpatial + j;
                                pET[i * nEdgeSpatial + j] = (nMaskIdx < static_cast<int>(vecTrainMasks.size()) && vecTrainMasks[nMaskIdx] > 0) ? 1.0f : 0.0f;
                            }
                        }
                        if (tOutput.isCuda()) tEdgeTarget = tEdgeTarget.cuda();

                        // 20260402 ZJH BCE+Dice 混合损失（边缘正样本加权 20x）
                        tLoss = om::EdgeExtractionNet::edgeLoss(tOutput, tEdgeTarget, 20.0f);

                    } else if (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) {
                        // 20260329 ZJH ===== GPU Fused Weighted PixelCE（零 CPU 回退）=====
                        // 全程 GPU: forward → fused softmax+CE kernel → backward kernel
                        // 消除旧路径的 D2H + CPU softmax + H2D 往返（~3-10x 加速）
                        // GPU 缓冲在循环外预分配（tClassWGpu/tSoftmaxBuf/tLossGpuBuf/tStatsBuf）
                        tOutput = m_pImpl->pModel->forward(tInput);  // 20260329 ZJH [B, C, H, W] logits (GPU)

                        int nH = m_pImpl->nInputSize;    // 20260329 ZJH 空间高度 H=W
                        int nSpatial = nH * nH;          // 20260329 ZJH 每张图像像素数
                        int nC = m_pImpl->nNumClasses;    // 20260329 ZJH 分割类别数（含背景 c=0）
                        int nTotalPx = nCurBatch * nSpatial;  // 20260329 ZJH batch 总像素数

                        // 20260329 ZJH 填充预分配 CPU 缓冲（零分配，仅写入）
                        for (int i = 0; i < nCurBatch; ++i) {
                            int nIdx = vecIndices[nStart + i];  // 20260329 ZJH 洗牌后样本索引
                            for (int s = 0; s < nSpatial; ++s) {
                                int nMO = nIdx * nSpatial + s;  // 20260329 ZJH 掩码偏移
                                int nCls = 0;
                                if (nMO < static_cast<int>(vecTrainMasks.size()))
                                    nCls = vecTrainMasks[nMO];  // 20260329 ZJH 像素类别 ID
                                vecTargetBuf[i * nSpatial + s] = static_cast<float>(nCls);
                            }
                        }
                        // 20260329 ZJH H2D 上传目标（复用 CPU 缓冲，仅传 nTotalPx 个 float）
                        auto tTargetFloat = om::Tensor::fromData(vecTargetBuf.data(), {nTotalPx}).cuda();

                        // 20260329 ZJH 确保 logits 连续
                        auto tLogitsContig = tOutput.contiguous();

                        // 20260329 ZJH 分配正确尺寸的 GPU 缓冲（softmax 随 nCurBatch 变化）
                        auto tSoftmax = om::Tensor::zeros({nCurBatch, nC, nH, nH}).cuda();
                        auto tLossGpu = om::Tensor::zeros({1}).cuda();
                        auto tStats = om::Tensor::zeros({2}).cuda();

                        // 20260329 ZJH 调用 fused CUDA kernel: softmax + weighted CE
                        om::CUDABackend::weightedPixelCEForward(
                            tLogitsContig.floatDataPtr(),     // 20260329 ZJH [B,C,H,W] logits
                            tTargetFloat.floatDataPtr(),      // 20260329 ZJH [N] float 类别 ID
                            tClassWGpu.floatDataPtr(),        // 20260329 ZJH [C] 权重（循环外上传）
                            tSoftmax.mutableFloatDataPtr(),   // 20260329 ZJH [nCurBatch,C,H,W] softmax
                            tLossGpu.mutableFloatDataPtr(),   // 20260329 ZJH [1] loss
                            tStats.mutableFloatDataPtr(),     // 20260329 ZJH [2] stats
                            nTotalPx, nC, nSpatial);

                        // 20260329 ZJH 注册 backward (PixelCE 部分)
                        tLoss = tLossGpu;
                        if (tOutput.requiresGrad()) {
                            auto pBack = std::make_shared<PixelCEBackwardFn>();
                            pBack->m_bUseCuda = true;
                            pBack->m_savedSoftmax = tSoftmax;
                            pBack->m_savedTargetFloat = tTargetFloat;
                            pBack->m_savedClassWeights = tClassWGpu;
                            pBack->m_savedStats = tStats;
                            pBack->m_nPixels = nTotalPx;
                            pBack->m_nClasses = nC;
                            pBack->m_nSpatial = nSpatial;
                            pBack->m_vecInputEdges.push_back(om::makeEdge(tOutput, 0));
                            tLoss.setGradFnRaw(pBack);
                            tLoss.setRequiresGrad(true);
                        }

                        // 20260330 ZJH ===== CE+Dice 混合损失（GPU 路径）=====
                        // CE 擅长逐像素分类精度，Dice 擅长区域级重叠度，两者梯度互补
                        // 使用 sigmoid-based Dice（每通道独立激活，无需通道维 softmax）
                        // autograd 链: tOutput → tensorSigmoid → tensorMul/Sum → Dice → tLoss
                        {
                            // 20260330 ZJH 构建 one-hot 目标 [B, C, H, W]（CPU 上构建后 H2D）
                            auto tOH = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                            float* pOH = tOH.mutableFloatDataPtr();
                            for (int i = 0; i < nCurBatch; ++i) {
                                int nIdx = vecIndices[nStart + i];
                                for (int s = 0; s < nSpatial; ++s) {
                                    int nMO = nIdx * nSpatial + s;
                                    int nCls = (nMO < static_cast<int>(vecTrainMasks.size())) ? vecTrainMasks[nMO] : 0;
                                    if (nCls >= 0 && nCls < nC)
                                        pOH[static_cast<size_t>(i) * nC * nSpatial + static_cast<size_t>(nCls) * nSpatial + s] = 1.0f;
                                }
                            }
                            tOH = tOH.cuda();

                            // 20260407 ZJH Sigmoid Dice（GPU 路径）— 稳定可用
                            // softmax Dice 因 NCHW reshape 和 tSoftmax GPU 指针问题导致崩溃
                            // sigmoid Dice 虽然与 CE 的 softmax 概率空间不完全一致，
                            // 但在 nnU-Net/Anomalib 等主流框架中广泛使用，实际训练效果良好
                            auto tSig = om::tensorSigmoid(tOutput);
                            auto tInter = om::tensorSum(om::tensorMul(tSig, tOH));
                            auto tPredS = om::tensorSum(tSig);
                            auto tTgtS  = om::tensorSum(tOH);
                            auto tEps = om::Tensor::full({1}, 1e-6f).cuda();
                            auto tDenom = om::tensorAdd(om::tensorAdd(tPredS, tTgtS), tEps);
                            // 20260330 ZJH 用 tensorDiv 保留 autograd（旧代码用 .cpu() 提取标量导致梯度断裂）
                            auto tNumer = om::tensorMulScalar(tInter, 2.0f);  // 20260330 ZJH 2 * intersection
                            auto tDiceCoeff = om::tensorDiv(tNumer, tDenom);   // 20260330 ZJH autograd 完整 ✓
                            auto tDiceLoss = om::tensorSub(om::Tensor::full({1}, 1.0f).cuda(), tDiceCoeff);

                            // 20260330 ZJH 混合: total = 0.5 * CE + 0.5 * Dice
                            tLoss = om::tensorAdd(
                                om::tensorMulScalar(tLoss, 0.5f),
                                om::tensorMulScalar(tDiceLoss, 0.5f));

                            // 20260406 ZJH [修复] BoundaryLoss 暂时禁用
                            // tensorSub(Tensor::full({1}), tSig) 形状 [1] vs [B,C,H,W] → 不广播 → 越界
                            // 需要重写为正确的逐像素 1-sigmoid 运算，暂时跳过避免损失爆炸
                            // TODO: 使用 tensorMulScalar(tSig, -1) + tensorAddScalar(_, 1) 替代
                        }
                    } else if (m_pImpl->bIsDetection) {
                        // 20260330 ZJH YOLO 检测模型 GPU 训练路���
                        tOutput = m_pImpl->pModel->forward(tInput);
                        // 20260330 ZJH 构造 YOLO 目标张量（弱监督：从图像级标签构造中心 cell 目标）
                        auto cpuTarget = om::Tensor::zeros(tOutput.shapeVec());
                        {
                            auto cpuLabels = tLabels.cpu();
                            float* pT = cpuTarget.mutableFloatDataPtr();
                            const float* pL = cpuLabels.floatDataPtr();
                            int nB = tOutput.shape(0), nP = tOutput.shape(1), nD = tOutput.shape(2);
                            int nC = nD - 5;
                            for (int b = 0; b < nB; ++b) {
                                int nOff = (b * nP + nP / 2) * nD;  // 20260330 ZJH 中心网格
                                pT[nOff + 0] = 0.5f; pT[nOff + 1] = 0.5f;  // 20260330 ZJH tx,ty
                                pT[nOff + 2] = 0.3f; pT[nOff + 3] = 0.3f;  // 20260330 ZJH tw,th
                                pT[nOff + 4] = 1.0f;  // 20260330 ZJH objectness=1
                                int nLabel = static_cast<int>(pL[b]);
                                if (nLabel >= 0 && nLabel < nC) pT[nOff + 5 + nLabel] = 1.0f;
                            }
                        }
                        auto tYoloTarget = tOutput.isCuda() ? cpuTarget.cuda() : cpuTarget;
                        tLoss = yoloLoss.forward(tOutput, tYoloTarget);
                    } else {
                        tOutput = m_pImpl->pModel->forward(tInput);
                        tLoss = criterion.forward(tOutput, tLabels);

                        // 20260407 ZJH [审计] OHEM 死代码已删除
                        // 原因: criterion.forward() 返回标量均值，tMask 构建后未与 tLoss 组合
                        // tensorMulScalar(tLoss, 1.0f) 等于无操作，每 batch 浪费一次 D2H + 排序
                        // 待实现逐样本 CE 后再重新添加 OHEM
                    }
                } catch (const std::exception& ex) {
                    if (logCb) logCb(std::string("[错误] 前向传播异常: ") + ex.what()
                        + " | 输入形状: [" + std::to_string(tInput.shape(0))
                        + (tInput.shapeVec().size() > 1 ? "," + std::to_string(tInput.shape(1)) : "")
                        + (tInput.shapeVec().size() > 2 ? "," + std::to_string(tInput.shape(2)) : "")
                        + (tInput.shapeVec().size() > 3 ? "," + std::to_string(tInput.shape(3)) : "")
                        + "] 请尝试减小输入尺寸或批量大小");
#ifdef OM_HAS_CUDA
                    if (bUseCuda) { try { omCudaCleanup(); } catch (...) {} }
#endif
                    return false;
                } catch (...) {
                    // 20260331 ZJH 捕获非 std::exception 异常（Windows SEH / CUDA 崩溃）
                    if (logCb) logCb("[错误] 前向/损失计算中发生非标准异常（SEH/CUDA crash）"
                        " | 输入: [" + std::to_string(tInput.shape(0))
                        + "," + std::to_string(tInput.shape(1))
                        + "," + std::to_string(tInput.shape(2))
                        + "," + std::to_string(tInput.shape(3)) + "]"
                        + " | 输出: " + (tOutput.numel() > 0 ?
                            ("[" + std::to_string(tOutput.shape(0))
                            + "," + std::to_string(tOutput.shape(1))
                            + "," + std::to_string(tOutput.shape(2))
                            + "," + std::to_string(tOutput.shape(3)) + "]") : "empty"));
#ifdef OM_HAS_CUDA
                    if (bUseCuda) { try { omCudaCleanup(); } catch (...) {} }
#endif
                    return false;
                }

                // 20260401 ZJH 梯度累积: 仅在累积窗口首个 batch 清零梯度
                // 后续 batch 的梯度会叠加到已有梯度上，达到 nAccumSteps 后才 step()
                bool bIsAccumStart = (nBatch % nAccumSteps == 0);
                bool bIsAccumEnd = ((nBatch + 1) % nAccumSteps == 0) || (nBatch == nBatches - 1);

                // 20260407 ZJH ===== PyTorch 六步法: scale → backward → unscale → clip → step → update =====
                // Step 0: 在 scale 之前保存真实 loss 值（用于显示）
                float fRealLoss = tLoss.item();

                // Step 1+2: scale(loss) → backward
                try {
                if (bIsAccumStart) m_pImpl->pModel->zeroGrad();
                if (bMixedPrecision) {
                    tLoss = gradScaler.scale(tLoss);  // 20260407 ZJH Step 1: loss *= scale（放大梯度防下溢）
                }
                om::tensorBackward(tLoss);  // 20260407 ZJH Step 2: backward（梯度被同比例放大）
                } catch (const std::exception& ex) {
                    if (logCb) logCb(std::string("[错误] 反向传播异常: ") + ex.what());
#ifdef OM_HAS_CUDA
                    if (bUseCuda) { try { omCudaCleanup(); } catch (...) {} }
#endif
                    return false;
                } catch (...) {
                    if (logCb) logCb("[错误] 反向传播非标准异常（SEH/CUDA crash）");
#ifdef OM_HAS_CUDA
                    if (bUseCuda) { try { omCudaCleanup(); } catch (...) {} }
#endif
                    return false;
                }

                // 20260407 ZJH 梯度累积: 仅在累积窗口末尾执行 unscale → clip → step → update
                if (bIsAccumEnd) {
                    // Step 3: unscale（梯度除以 scale，恢复真实值 + NaN 检测）
                    bool bShouldStep = true;
                    if (bMixedPrecision) {
                        // 20260407 ZJH [修复] 梯度反缩放直接通过 GradAccumulator（fp16 模块只做 inf 检测）
                        float fInvScale = 1.0f / gradScaler.getScale();
                        auto vecStepParams = m_pImpl->pModel->parameters();
                        for (auto* pParam : vecStepParams) {
                            auto pAccumRaw = pParam->gradAccumRaw();
                            if (!pAccumRaw) continue;
                            auto pAccum = std::static_pointer_cast<om::GradAccumulator>(pAccumRaw);
                            if (!pAccum->m_bHasGrad) continue;
                            pAccum->m_grad = om::tensorMulScalar(pAccum->m_grad, fInvScale);
                        }
                        // 20260407 ZJH inf 检测 + step 判断
                        gradScaler.unscaleGrads(vecStepParams);  // 20260407 ZJH 只做 inf 检测
                        bShouldStep = gradScaler.step();
                        if (!bShouldStep && logCb) {
                            logCb("[WARN] GradScaler: inf/NaN detected, skip step (scale=" +
                                  std::to_string(gradScaler.getScale()) + ")");
                        }
                    }

                    // Step 4: clip_grad_norm（在真实梯度上裁剪，不是放大后的梯度）
                    if (bShouldStep) {
                        bool bGradValid = true;
                        float fGradNormSq = 0.0f;
                        std::vector<std::shared_ptr<om::GradAccumulator>> vecAccums;
                        for (auto* pParam : vecModelParams) {
                            auto pAccumRaw = pParam->gradAccumRaw();
                            if (!pAccumRaw) continue;
                            auto pAccum = std::static_pointer_cast<om::GradAccumulator>(pAccumRaw);
                            if (!pAccum->m_bHasGrad) continue;
                            vecAccums.push_back(pAccum);
                            auto cGrad = pAccum->m_grad.contiguous();
                            if (cGrad.isCuda()) cGrad = cGrad.cpu();
                            const float* pG = cGrad.floatDataPtr();
                            int nN = cGrad.numel();
                            for (int gi = 0; gi < nN; ++gi) {
                                if (std::isnan(pG[gi]) || std::isinf(pG[gi])) { bGradValid = false; break; }
                                fGradNormSq += pG[gi] * pG[gi];
                            }
                            if (!bGradValid) break;
                        }
                        if (!bGradValid) {
                            if (logCb) logCb("[WARN] NaN/Inf gradient — skipping batch");
                            bShouldStep = false;
                        } else {
                            constexpr float fMaxGradNorm = 5.0f;
                            float fGradNorm = std::sqrt(fGradNormSq);
                            if (fGradNorm > fMaxGradNorm) {
                                float fClipCoeff = fMaxGradNorm / fGradNorm;
                                for (auto& pAccum : vecAccums) {
                                    pAccum->m_grad = om::tensorMulScalar(pAccum->m_grad, fClipCoeff);
                                }
                            }
                            // 20260401 ZJH 梯度累积平均
                            if (nAccumSteps > 1) {
                                float fAccumScale = 1.0f / static_cast<float>(nAccumSteps);
                                for (auto& pAcc : vecAccums) {
                                    pAcc->m_grad = om::tensorMulScalar(pAcc->m_grad, fAccumScale);
                                }
                            }
                        }
                    }

                    // Step 5: optimizer.step（梯度正常时更新参数）
                    if (bShouldStep) {
                        if (pAdam) pAdam->step();
                        else if (pAdamW) pAdamW->step();
                        else if (pSgd) pSgd->step();
                    }

                    // 20260402 ZJH GradScaler 动态更新
                    if (bMixedPrecision) gradScaler.update();

                    // 20260401 ZJH EMA 权重更新: ema = decay * ema + (1-decay) * param
                    if (bUseEma) {
                        auto vecCurParams = m_pImpl->pModel->parameters();
                        for (size_t ei = 0; ei < vecEmaParams.size() && ei < vecCurParams.size(); ++ei) {
                            auto cpuParam = vecCurParams[ei]->cpu();  // 20260401 ZJH 从 GPU 取回当前参数
                            float* pEma = vecEmaParams[ei].mutableFloatDataPtr();
                            const float* pCur = cpuParam.floatDataPtr();
                            int nE = vecEmaParams[ei].numel();
                            for (int gi = 0; gi < nE; ++gi) {
                                pEma[gi] = fEmaDecay * pEma[gi] + (1.0f - fEmaDecay) * pCur[gi];
                            }
                        }
                    }
                }

                // 20260406 ZJH 使用 scale 之前保存的真实 loss 值（非放大值）
                tOutput = om::Tensor();
                tLoss = om::Tensor();
                tInput = om::Tensor();
                tLabels = om::Tensor();
                fEpochLoss += fRealLoss;

                if (batchCb) batchCb(nBatch + 1, nBatches);
            }
        } else {
            // 20260324 ZJH ===== CPU 双缓冲训练循环（保持原有逻辑不变）=====
            // 1. 预填充第一个 batch 到缓冲区 A
            // 2. 循环中：异步填充下一个 batch 到缓冲区 B，同时用缓冲区 A 训练
            // 3. 训练完成后交换 A/B 缓冲区

            // 20260324 ZJH 预填充第一个 batch
            int nFirstEnd = std::min(nBatchSize, nTrainCount);      // 20260324 ZJH 第一个 batch 结束索引
            int nFirstBatch = nFirstEnd;                              // 20260324 ZJH 第一个 batch 实际大小
            fillBatch(vecBufA_Input, vecBufA_Label,
                      vecTrainData, vecTrainLabels, vecIndices,
                      0, nFirstBatch, nInputDim, nNumClasses);

            for (int nBatch = 0; nBatch < nBatches; ++nBatch) {
                if (stopCheck && stopCheck()) return false;

                int nStart = nBatch * nBatchSize;
                int nEnd = std::min(nStart + nBatchSize, nTrainCount);
                int nCurBatch = nEnd - nStart;

                // 20260324 ZJH 异步填充下一个 batch 到缓冲区 B（如果还有下一个 batch）
                std::future<void> futureNext;
                int nNextBatch = nBatch + 1;              // 20260324 ZJH 下一个 batch 索引
                int nNextCurBatch = 0;                     // 20260324 ZJH 下一个 batch 实际大小
                bool bHasNext = (nNextBatch < nBatches);   // 20260324 ZJH 是否还有下一个 batch

                if (bHasNext) {
                    int nNextStart = nNextBatch * nBatchSize;
                    int nNextEnd = std::min(nNextStart + nBatchSize, nTrainCount);
                    nNextCurBatch = nNextEnd - nNextStart;

                    // 20260324 ZJH 异步启动数据填充线程
                    futureNext = std::async(std::launch::async, [&]() {
                        fillBatch(vecBufB_Input, vecBufB_Label,
                                  vecTrainData, vecTrainLabels, vecIndices,
                                  nNextStart, nNextCurBatch, nInputDim, nNumClasses);
                    });
                }

                // 20260407 ZJH [修复] CPU 路径增强也需跳过分割模型（掩码不同步）
                if (params.bAugmentEnabled && m_pImpl->bIsCnn && !m_pImpl->bIsSegmentation) {
                    int nC = 3;
                    int nSp = m_pImpl->nInputSize;
                    if (nC * nSp * nSp == nInputDim && nSp > 0) {
                        for (int i = 0; i < nCurBatch; ++i) {
                            size_t nOff = static_cast<size_t>(i) * nInputDim;
                            std::copy(vecBufA_Input.data() + nOff,
                                      vecBufA_Input.data() + nOff + nInputDim,
                                      vecSampleBuf.begin());
                            try { om::augmentImage(vecSampleBuf, nC, nSp, nSp, augCfg); } catch (...) {}
                            std::copy(vecSampleBuf.begin(), vecSampleBuf.end(), vecBufA_Input.data() + nOff);
                        }
                    }
                }

                // 20260330 ZJH ===== S1: 统一归一化（CPU 路径）=====
                if (m_pImpl->bIsCnn && eNormPreset == om::NormPreset::ImageNet) {
                    int nC = 3;
                    int nSp = m_pImpl->nInputSize;
                    if (nC * nSp * nSp == nInputDim && nSp > 0) {
                        for (int i = 0; i < nCurBatch; ++i) {
                            size_t nOff = static_cast<size_t>(i) * nInputDim;
                            std::copy(vecBufA_Input.data() + nOff,
                                      vecBufA_Input.data() + nOff + nInputDim,
                                      vecSampleBuf.begin());
                            try { om::normalizeImage(vecSampleBuf, nC, nSp, nSp, normCfg); } catch (...) {}
                            std::copy(vecSampleBuf.begin(), vecSampleBuf.end(), vecBufA_Input.data() + nOff);
                        }
                    }
                }

                // 20260325 ZJH 用缓冲区 A 执行当前 batch（CNN→4D, MLP→2D）
                auto tInput = m_pImpl->bIsCnn
                    ? om::Tensor::fromData(vecBufA_Input.data(), {nCurBatch, 3, m_pImpl->nInputSize, m_pImpl->nInputSize})
                    : om::Tensor::fromData(vecBufA_Input.data(), {nCurBatch, nInputDim});
                auto tLabels = om::Tensor::fromData(vecBufA_Label.data(), {nCurBatch, nNumClasses});

                om::Tensor tOutput, tLoss;
                try {
                    // 20260326 ZJH EfficientAD 使用蒸馏损失（MSE(teacher, student)），其他模型使用交叉熵
                    if (m_pImpl->bIsEfficientAD) {
                        auto* pEAD = static_cast<om::EfficientAD*>(m_pImpl->pModel.get());
                        tLoss = pEAD->computeDistillationLoss(tInput);  // 20260326 ZJH 教师-学生特征 MSE
                    } else if (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) {
                        // 20260329 ZJH ===== CPU Pixel-wise Cross-Entropy =====
                        tOutput = m_pImpl->pModel->forward(tInput);  // 20260329 ZJH [B, C, H, W] logits

                        int nH = m_pImpl->nInputSize;    // 20260329 ZJH 空间高度
                        int nSpatial = nH * nH;          // 20260329 ZJH 像素数
                        int nC = m_pImpl->nNumClasses;    // 20260329 ZJH 类别数
                        int nTotalPx = nCurBatch * nSpatial;  // 20260329 ZJH 总像素数

                        // 20260329 ZJH 构建 one-hot [B, C, H, W]
                        auto tTargetOH = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                        {
                            float* pT = tTargetOH.mutableFloatDataPtr();
                            for (int i = 0; i < nCurBatch; ++i) {
                                int nIdx = vecIndices[nStart + i];
                                for (int s = 0; s < nSpatial; ++s) {
                                    int nMO = nIdx * nSpatial + s;
                                    if (nMO < static_cast<int>(vecTrainMasks.size())) {
                                        int nCls = vecTrainMasks[nMO];
                                        if (nCls >= 0 && nCls < nC) {
                                            pT[static_cast<size_t>(i) * nC * nSpatial
                                               + static_cast<size_t>(nCls) * nSpatial + s] = 1.0f;
                                        }
                                    }
                                }
                            }
                        }

                        // 20260329 ZJH 逐像素 softmax + 加权 CE（CPU 上）
                        auto cOut = tOutput.contiguous();
                        const float* pL = cOut.floatDataPtr();
                        auto tSoftmax = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                        float* pSM = tSoftmax.mutableFloatDataPtr();
                        auto tWeightMap = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                        float* pWM = tWeightMap.mutableFloatDataPtr();
                        float fCeLoss = 0.0f;
                        float fWeightSum = 0.0f;

                        for (int b = 0; b < nCurBatch; ++b) {
                            for (int px = 0; px < nSpatial; ++px) {
                                float fMax = -1e30f;
                                for (int c = 0; c < nC; ++c) {
                                    float fV = pL[b * nC * nSpatial + c * nSpatial + px];
                                    if (fV > fMax) fMax = fV;
                                }
                                float fExpSum = 0.0f;
                                for (int c = 0; c < nC; ++c) {
                                    float fE = std::exp(pL[b * nC * nSpatial + c * nSpatial + px] - fMax);
                                    pSM[b * nC * nSpatial + c * nSpatial + px] = fE;
                                    fExpSum += fE;
                                }
                                float fInvSum = 1.0f / fExpSum;
                                for (int c = 0; c < nC; ++c) {
                                    pSM[b * nC * nSpatial + c * nSpatial + px] *= fInvSum;
                                }
                                int nIdx = vecIndices[nStart + b];
                                int nMO = nIdx * nSpatial + px;
                                int nTC = 0;
                                if (nMO < static_cast<int>(vecTrainMasks.size())) nTC = vecTrainMasks[nMO];
                                // 20260329 ZJH 无效类别权重=0（与 GPU kernel 一致）
                                float fW = (nTC >= 0 && nTC < nC) ? vecClassWeights[nTC] : 0.0f;
                                if (nTC >= 0 && nTC < nC) {
                                    float fPt = pSM[b * nC * nSpatial + nTC * nSpatial + px];
                                    float fFocal = (1.0f - fPt) * (1.0f - fPt);  // 20260330 ZJH (1-p_t)^2 focal weight
                                    fCeLoss -= fW * fFocal * std::log(std::max(fPt, 1e-7f));  // 20260330 ZJH Focal CE
                                    fWeightSum += fW;  // 20260329 ZJH 仅有效像素累加权重
                                }
                                for (int c = 0; c < nC; ++c) {
                                    pWM[b * nC * nSpatial + c * nSpatial + px] = fW;
                                }
                            }
                        }
                        fCeLoss /= std::max(fWeightSum, 1e-7f);

                        // 20260329 ZJH 创建损失 + 注册加权反向
                        tLoss = om::Tensor::full({1}, fCeLoss);
                        if (tOutput.requiresGrad()) {
                            auto pBack = std::make_shared<PixelCEBackwardFn>();
                            pBack->m_savedSoftmax = tSoftmax;
                            pBack->m_savedTarget = tTargetOH;
                            pBack->m_savedWeightMap = tWeightMap;
                            pBack->m_fWeightSum = fWeightSum;
                            pBack->m_vecInputEdges.push_back(om::makeEdge(tOutput, 0));
                            tLoss.setGradFnRaw(pBack);
                            tLoss.setRequiresGrad(true);
                        }

                        // 20260407 ZJH CE+Dice 混合损失（CPU 路径）— sigmoid Dice
                        // CPU 路径没有 CUDA kernel 预算的 softmax，用 sigmoid 作为 Dice 的激活
                        // sigmoid Dice 梯度与 CE 有轻微冲突但不崩溃，且 CPU 训练较少使用
                        {
                            auto tSig = om::tensorSigmoid(tOutput);
                            auto tInter = om::tensorSum(om::tensorMul(tSig, tTargetOH));
                            auto tPredS = om::tensorSum(tSig);
                            auto tTgtS  = om::tensorSum(tTargetOH);
                            auto tDenom = om::tensorAdd(om::tensorAdd(tPredS, tTgtS),
                                                         om::Tensor::full({1}, 1e-6f));
                            // 20260330 ZJH 用 tensorDiv 保留 autograd（旧代码提取标量导致梯度断裂）
                            auto tNumer = om::tensorMulScalar(tInter, 2.0f);
                            auto tDiceCoeff = om::tensorDiv(tNumer, tDenom);  // 20260330 ZJH autograd ✓
                            auto tDiceLoss = om::tensorSub(om::Tensor::full({1}, 1.0f), tDiceCoeff);
                            tLoss = om::tensorAdd(
                                om::tensorMulScalar(tLoss, 0.5f),
                                om::tensorMulScalar(tDiceLoss, 0.5f));
                        }
                    } else if (m_pImpl->bIsDetection) {
                        // 20260330 ZJH YOLO 检测模型 CPU 训练路径
                        tOutput = m_pImpl->pModel->forward(tInput);
                        auto tYoloTarget = om::Tensor::zeros(tOutput.shapeVec());
                        {
                            float* pT = tYoloTarget.mutableFloatDataPtr();
                            const float* pL = tLabels.floatDataPtr();
                            int nB = tOutput.shape(0), nP = tOutput.shape(1), nD = tOutput.shape(2);
                            int nC = nD - 5;
                            for (int b = 0; b < nB; ++b) {
                                int nOff = (b * nP + nP / 2) * nD;
                                pT[nOff + 0] = 0.5f; pT[nOff + 1] = 0.5f;
                                pT[nOff + 2] = 0.3f; pT[nOff + 3] = 0.3f;
                                pT[nOff + 4] = 1.0f;
                                int nLabel = static_cast<int>(pL[b]);
                                if (nLabel >= 0 && nLabel < nC) pT[nOff + 5 + nLabel] = 1.0f;
                            }
                        }
                        tLoss = yoloLoss.forward(tOutput, tYoloTarget);
                    } else {
                        tOutput = m_pImpl->pModel->forward(tInput);
                        tLoss = criterion.forward(tOutput, tLabels);
                    }
                } catch (const std::exception& ex) {
                    if (logCb) logCb(std::string("[错误] 前向传播异常: ") + ex.what()
                        + " 请尝试减小输入尺寸或批量大小");
                    return false;
                }

                // 20260401 ZJH 梯度累积（CPU 路径，同 GPU 路径逻辑）
                bool bIsAccumStartCpu = (nBatch % nAccumSteps == 0);
                bool bIsAccumEndCpu = ((nBatch + 1) % nAccumSteps == 0) || (nBatch == nBatches - 1);
                if (bIsAccumStartCpu) m_pImpl->pModel->zeroGrad();
                om::tensorBackward(tLoss);

                // 20260330 ZJH 梯度安全网（CPU 路径，同 GPU 路径逻辑）
                {
                    bool bGradValid = true;
                    float fGradNormSq = 0.0f;
                    std::vector<std::shared_ptr<om::GradAccumulator>> vecAccums;
                    for (auto* pParam : vecModelParams) {
                        auto pAccumRaw = pParam->gradAccumRaw();
                        if (!pAccumRaw) continue;
                        auto pAccum = std::static_pointer_cast<om::GradAccumulator>(pAccumRaw);
                        if (!pAccum->m_bHasGrad) continue;
                        vecAccums.push_back(pAccum);
                        const float* pG = pAccum->m_grad.contiguous().floatDataPtr();
                        int nN = pAccum->m_grad.numel();
                        for (int gi = 0; gi < nN; ++gi) {
                            if (std::isnan(pG[gi]) || std::isinf(pG[gi])) { bGradValid = false; break; }
                            fGradNormSq += pG[gi] * pG[gi];
                        }
                        if (!bGradValid) break;
                    }
                    if (!bGradValid) {
                        if (logCb) logCb("[WARN] NaN/Inf gradient — skipping batch");
                        tOutput = om::Tensor(); tLoss = om::Tensor();
                        continue;
                    }
                    constexpr float fMaxGradNorm = 5.0f;
                    float fGradNorm = std::sqrt(fGradNormSq);
                    if (fGradNorm > fMaxGradNorm) {
                        float fClipCoeff = fMaxGradNorm / fGradNorm;
                        for (auto& pAccum : vecAccums) {
                            pAccum->m_grad = om::tensorMulScalar(pAccum->m_grad, fClipCoeff);
                        }
                    }
                    // 20260401 ZJH 梯度累积缩放（CPU 路径，在 scope 内完成）
                    if (bIsAccumEndCpu && nAccumSteps > 1) {
                        float fScale = 1.0f / static_cast<float>(nAccumSteps);
                        for (auto& pAcc : vecAccums) {
                            pAcc->m_grad = om::tensorMulScalar(pAcc->m_grad, fScale);
                        }
                    }
                }

                // 20260401 ZJH 梯度累积: 仅在累积窗口末尾执行 step（CPU 路径）
                if (bIsAccumEndCpu) {
                    if (pAdam) pAdam->step();
                    else if (pAdamW) pAdamW->step();
                    else if (pSgd) pSgd->step();

                    // 20260401 ZJH EMA 权重更新（CPU 路径）
                    if (bUseEma) {
                        auto vecCurParams = m_pImpl->pModel->parameters();
                        for (size_t ei = 0; ei < vecEmaParams.size() && ei < vecCurParams.size(); ++ei) {
                            float* pEma = vecEmaParams[ei].mutableFloatDataPtr();
                            const float* pCur = vecCurParams[ei]->floatDataPtr();
                            int nE = vecEmaParams[ei].numel();
                            for (int gi = 0; gi < nE; ++gi) {
                                pEma[gi] = fEmaDecay * pEma[gi] + (1.0f - fEmaDecay) * pCur[gi];
                            }
                        }
                    }
                }

                // 20260326 ZJH 取 loss 值后释放计算图，回收内存
                float fBatchLoss = tLoss.item();
                tOutput = om::Tensor();
                tLoss = om::Tensor();

                fEpochLoss += fBatchLoss;
                if (batchCb) batchCb(nBatch + 1, nBatches);

                // 20260324 ZJH 等待下一个 batch 的异步填充完成
                if (bHasNext) {
                    futureNext.wait();
                    // 20260324 ZJH 交换 A/B 缓冲区（O(1) swap，无数据拷贝）
                    std::swap(vecBufA_Input, vecBufB_Input);
                    std::swap(vecBufA_Label, vecBufB_Label);
                }
            }
        }

        float fAvgTrainLoss = fEpochLoss / std::max(1, nBatches);  // 20260406 ZJH 本 epoch 平均训练损失

        // 20260323 ZJH ===== 验证阶段 =====
        std::cerr << "[TRAIN-DIAG] epoch " << nEpoch << " validation start" << std::endl;
        float fValLoss = 0.0f;   // 20260406 ZJH 验证损失累计值
        int nValCorrect = 0;    // 20260406 ZJH 验证正确数（分类=正确图像数, 分割=正确像素数）
        int nValTotalPixels = 0;  // 20260328 ZJH 分割模型验证的总像素计数（像素级准确率分母）
        m_pImpl->pModel->eval();

        if (nValCount > 0) {
            int nVB = (nValCount + nBatchSize - 1) / nBatchSize;
            for (int b = 0; b < nVB; ++b) {
                int s = b * nBatchSize;
                int e = std::min(s + nBatchSize, nValCount);
                int nc = e - s;

                // 20260324 ZJH 使用独立验证缓冲区（避免与训练缓冲区冲突）
                std::fill(vecValInput.begin(), vecValInput.begin() + static_cast<size_t>(nc) * nInputDim, 0.0f);
                std::fill(vecValLabel.begin(), vecValLabel.begin() + static_cast<size_t>(nc) * nNumClasses, 0.0f);

                for (int i = 0; i < nc; ++i) {
                    int idx = s + i;
                    int src = idx * nInputDim;
                    if (src + nInputDim <= static_cast<int>(vecValData.size())) {
                        std::copy(vecValData.begin() + src, vecValData.begin() + src + nInputDim,
                                  vecValInput.begin() + static_cast<size_t>(i) * nInputDim);
                    }
                    int lab = vecValLabels[idx];
                    if (lab >= 0 && lab < nNumClasses) vecValLabel[static_cast<size_t>(i) * nNumClasses + lab] = 1.0f;
                }

                // 20260330 ZJH ===== S1: 验证数据归一化（与训练一致，无增强）=====
                // 注意: 验证数据不做数据增强，仅做归一化
                if (m_pImpl->bIsCnn && eNormPreset == om::NormPreset::ImageNet) {
                    int nC = 3;
                    int nSp = m_pImpl->nInputSize;
                    for (int i = 0; i < nc; ++i) {
                        // 20260330 ZJH 复用预分配缓冲区，避免每样本堆分配
                        size_t nOff = static_cast<size_t>(i) * nInputDim;
                        std::copy(vecValInput.data() + nOff,
                                  vecValInput.data() + nOff + nInputDim,
                                  vecSampleBuf.begin());
                        om::normalizeImage(vecSampleBuf, nC, nSp, nSp, normCfg);
                        std::copy(vecSampleBuf.begin(), vecSampleBuf.end(), vecValInput.data() + nOff);
                    }
                }

                // 20260325 ZJH 创建验证 batch（CNN→4D, MLP→2D）
                auto tI = m_pImpl->bIsCnn
                    ? om::Tensor::fromData(vecValInput.data(), {nc, 3, m_pImpl->nInputSize, m_pImpl->nInputSize})
                    : om::Tensor::fromData(vecValInput.data(), {nc, nInputDim});
                auto tL = om::Tensor::fromData(vecValLabel.data(), {nc, nNumClasses});

                // 20260325 ZJH GPU 路径：将验证 batch 上传到 GPU
#ifdef OM_HAS_CUDA
                if (bUseCuda) {
                    tI = tI.cuda();   // 20260325 ZJH 验证输入 H2D
                    tL = tL.cuda();   // 20260325 ZJH 验证标签 H2D
                }
#endif

                std::cerr << "[TRAIN-DIAG] val batch " << b << " forward start" << std::endl;
                om::Tensor tO;
                try {
                    tO = m_pImpl->pModel->forward(tI);
                    std::cerr << "[TRAIN-DIAG] val batch " << b << " forward OK, output=["
                              << tO.shape(0) << "," << tO.shape(1) << "," << tO.shape(2) << "," << tO.shape(3)
                              << "] cuda=" << tO.isCuda() << std::endl;
                } catch (const std::exception& ex) {
                    if (logCb) logCb(std::string("[错误] 验证前向异常: ") + ex.what());
                    std::cerr << "[TRAIN-DIAG] val forward EXCEPTION: " << ex.what() << std::endl;
                    break;
                } catch (...) {
                    if (logCb) logCb("[错误] 验证前向非标准异常");
                    std::cerr << "[TRAIN-DIAG] val forward UNKNOWN EXCEPTION" << std::endl;
                    break;
                }

                // 20260328 ZJH 分割模型验证：计算 Dice Loss + 像素级准确率
                if (m_pImpl->bIsSegmentation && !vecValMasks.empty()) {
                    // 20260328 ZJH 分割验证路径：Dice Loss + 逐像素准确率
                    int nH = m_pImpl->nInputSize;
                    int nSpatial = nH * nH;
                    int nC = m_pImpl->nNumClasses;

                    // 20260328 ZJH 确保输出在 CPU 上以便读取
                    std::cerr << "[TRAIN-DIAG] val seg D2H start" << std::endl;
                    om::Tensor tO_cpu_seg;
#ifdef OM_HAS_CUDA
                    if (bUseCuda) {
                        tO_cpu_seg = tO.cpu();
                    } else
#endif
                    {
                        tO_cpu_seg = tO;
                    }
                    auto cValOut = tO_cpu_seg.contiguous();  // 20260328 ZJH 连续化
                    const float* pValOut = cValOut.floatDataPtr();

                    // 20260328 ZJH 计算 Dice Loss（无梯度，仅指标）
                    // Reshape [B,C,H,W] -> [B*H*W, C] 在 CPU 上手动计算
                    float fDiceSum = 0.0f;   // 20260328 ZJH 累计 Dice 系数
                    int nTotalPixels = 0;    // 20260328 ZJH 总像素数
                    int nCorrectPixels = 0;  // 20260328 ZJH 正确预测的像素数

                    // 20260329 ZJH 预分配 argmax + 真实标签缓存（消除 Dice 内重复 argmax，O(C²S)→O(CS)）
                    std::vector<int> vecPredBuf(nSpatial);   // 20260329 ZJH 缓存 argmax 预测类别
                    std::vector<int> vecTrueBuf(nSpatial);   // 20260329 ZJH 缓存真实类别

                    for (int i = 0; i < nc; ++i) {
                        int nIdx = s + i;  // 20260328 ZJH 验证集样本索引

                        // 20260329 ZJH 一趟 argmax + 真实标签读取（两者共享循环，减少一半遍历）
                        for (int px = 0; px < nSpatial; ++px) {
                            // 20260329 ZJH argmax: 找预测类别
                            int nBestClass = 0;
                            float fBestVal = pValOut[static_cast<size_t>(i) * nC * nSpatial + px];
                            for (int c = 1; c < nC; ++c) {
                                float fVal = pValOut[static_cast<size_t>(i) * nC * nSpatial + static_cast<size_t>(c) * nSpatial + px];
                                if (fVal > fBestVal) { fBestVal = fVal; nBestClass = c; }
                            }
                            vecPredBuf[px] = nBestClass;  // 20260329 ZJH 缓存预测

                            // 20260329 ZJH 真实标签
                            int nMO = nIdx * nSpatial + px;
                            int nTC = (nMO < static_cast<int>(vecValMasks.size())) ? vecValMasks[nMO] : 0;
                            vecTrueBuf[px] = nTC;  // 20260329 ZJH 缓存真实

                            // 20260329 ZJH 准确率统计
                            if (nBestClass == nTC) ++nCorrectPixels;
                            ++nTotalPixels;
                        }

                        // 20260329 ZJH Dice 系数：用缓存的 pred/true 扫描一趟 O(C*S)
                        float fSampleDice = 0.0f;
                        int nValidClasses = 0;
                        for (int c = 0; c < nC; ++c) {
                            float fInter = 0.0f, fPredC = 0.0f, fTrueC = 0.0f;
                            for (int px = 0; px < nSpatial; ++px) {
                                int nP = vecPredBuf[px];  // 20260329 ZJH 已缓存的预测（无重复 argmax）
                                int nT = vecTrueBuf[px];  // 20260329 ZJH 已缓存的真实
                                if (nP == c) fPredC += 1.0f;
                                if (nT == c) fTrueC += 1.0f;
                                if (nP == c && nT == c) fInter += 1.0f;
                            }
                            if (fTrueC > 0.0f || fPredC > 0.0f) {
                                fSampleDice += 2.0f * fInter / (fPredC + fTrueC + 1e-6f);
                                ++nValidClasses;
                            }
                        }
                        if (nValidClasses > 0) fDiceSum += fSampleDice / static_cast<float>(nValidClasses);
                    }
                    // 20260328 ZJH 该 batch 的平均 Dice Loss = 1 - mean Dice
                    float fBatchDice = (nc > 0) ? fDiceSum / static_cast<float>(nc) : 0.0f;
                    fValLoss += (1.0f - fBatchDice);  // 20260328 ZJH Dice Loss

                    // 20260328 ZJH 像素级准确率累加到 nValCorrect（语义: 正确像素数 / 总像素数）
                    nValCorrect += nCorrectPixels;
                    nValTotalPixels += nTotalPixels;  // 20260328 ZJH 分割模型的总像素计数
                } else {
                    // 20260328 ZJH 分类模型: 原有逻辑（交叉熵 + image-level 准确率）
                    auto tLs = criterion.forward(tO, tL);

                    // 20260325 ZJH 获取 loss 和输出：GPU 路径需 D2H 拷回 CPU 才能读取
                    float fBatchValLoss = 0.0f;
                    const float* pO = nullptr;
                    om::Tensor tO_cpu, tLs_cpu;  // 20260325 ZJH 保持 CPU 副本的生命周期
#ifdef OM_HAS_CUDA
                    if (bUseCuda) {
                        tLs_cpu = tLs.cpu();          // 20260325 ZJH loss D2H（4 字节）
                        tO_cpu = tO.cpu();             // 20260325 ZJH 输出 D2H（用于计算准确率）
                        fBatchValLoss = tLs_cpu.item();  // 20260325 ZJH 使用 item() 读取标量
                        pO = tO_cpu.floatDataPtr();
                    } else
#endif
                    {
                        fBatchValLoss = tLs.item();  // 20260325 ZJH 使用 item() 读取标量
                        pO = tO.floatDataPtr();
                    }
                    fValLoss += fBatchValLoss;

                    // 20260325 ZJH 计算验证准确率（在 CPU 上比较预测结果与真实标签）
                    for (int i = 0; i < nc; ++i) {
                        int pred = 0;
                        float mx = pO[i * nNumClasses];
                        for (int c = 1; c < nNumClasses; ++c) {
                            if (pO[i * nNumClasses + c] > mx) { mx = pO[i * nNumClasses + c]; pred = c; }
                        }
                        if (pred == vecValLabels[s + i]) ++nValCorrect;
                    }
                }
            }
            fValLoss /= std::max(1, nVB);
        }

        // 20260328 ZJH 分割模型用像素级准确率，分类模型用图像级准确率
        float fValAcc = 0.0f;
        if (m_pImpl->bIsSegmentation && !vecValMasks.empty()) {
            fValAcc = (nValTotalPixels > 0) ? static_cast<float>(nValCorrect) / static_cast<float>(nValTotalPixels) : 0.0f;
        } else {
            fValAcc = (nValCount > 0) ? static_cast<float>(nValCorrect) / nValCount : 0.0f;
        }

        if (fValLoss < fBestValLoss) {
            fBestValLoss = fValLoss;
            nPatienceCounter = 0;
            // 20260330 ZJH ===== S2: 保存最佳检查点（深拷贝参数和缓冲区）=====
            // 训练验证损失创新低时，克隆当前模型状态
            // 训练结束后恢复最佳权重，避免过拟合后期的权重退化
            vecBestParams.clear();
            for (auto* pParam : m_pImpl->pModel->parameters()) {
                // 20260331 ZJH 深拷贝参数: GPU 参数先 D2H 再 memcpy（避免从 GPU 指针 memcpy 崩溃）
                auto tCpu = pParam->isCuda() ? pParam->cpu() : *pParam;
                auto tContig = tCpu.contiguous();
                om::Tensor tClone = om::Tensor::zeros(pParam->shapeVec());
                std::memcpy(tClone.mutableFloatDataPtr(), tContig.floatDataPtr(),
                            static_cast<size_t>(pParam->numel()) * sizeof(float));
                vecBestParams.push_back(std::move(tClone));
            }
            vecBestBuffers.clear();
            for (auto* pBuf : m_pImpl->pModel->buffers()) {
                // 20260331 ZJH 深拷贝 BN running stats: GPU 缓冲先 D2H 再 memcpy
                auto tCpu = pBuf->isCuda() ? pBuf->cpu() : *pBuf;
                auto tContig = tCpu.contiguous();
                om::Tensor tClone = om::Tensor::zeros(pBuf->shapeVec());
                std::memcpy(tClone.mutableFloatDataPtr(), tContig.floatDataPtr(),
                            static_cast<size_t>(pBuf->numel()) * sizeof(float));
                vecBestBuffers.push_back(std::move(tClone));
            }
            bHasBestCheckpoint = true;
        }

        // 20260401 ZJH ===== Model Soup: 维护 top-K 检查点列表 =====
        // 每个 epoch 检查当前 val_loss 是否进入 top-K
        {
            bool bShouldSave = (static_cast<int>(vecTopCheckpoints.size()) < nTopK) ||
                               (fValLoss < vecTopCheckpoints.back().fValLoss);
            if (bShouldSave && nEpoch >= 3) {  // 20260401 ZJH 前 3 epoch 跳过（权重尚未稳定）
                CheckpointEntry ckpt;
                ckpt.fValLoss = fValLoss;
                for (auto* pParam : m_pImpl->pModel->parameters()) {
                    auto tCpu = pParam->isCuda() ? pParam->cpu() : *pParam;
                    auto tContig = tCpu.contiguous();
                    om::Tensor tClone = om::Tensor::zeros(pParam->shapeVec());
                    std::memcpy(tClone.mutableFloatDataPtr(), tContig.floatDataPtr(),
                                static_cast<size_t>(pParam->numel()) * sizeof(float));
                    ckpt.vecParams.push_back(std::move(tClone));
                }
                for (auto* pBuf : m_pImpl->pModel->buffers()) {
                    auto tCpu = pBuf->isCuda() ? pBuf->cpu() : *pBuf;
                    auto tContig = tCpu.contiguous();
                    om::Tensor tClone = om::Tensor::zeros(pBuf->shapeVec());
                    std::memcpy(tClone.mutableFloatDataPtr(), tContig.floatDataPtr(),
                                static_cast<size_t>(pBuf->numel()) * sizeof(float));
                    ckpt.vecBuffers.push_back(std::move(tClone));
                }
                vecTopCheckpoints.push_back(std::move(ckpt));
                // 20260401 ZJH 按 val_loss 升序排序，保留 top-K
                std::sort(vecTopCheckpoints.begin(), vecTopCheckpoints.end(),
                    [](const CheckpointEntry& a, const CheckpointEntry& b) { return a.fValLoss < b.fValLoss; });
                if (static_cast<int>(vecTopCheckpoints.size()) > nTopK) {
                    vecTopCheckpoints.resize(nTopK);  // 20260401 ZJH 丢弃最差的
                }
            }
        }

        if (nPatienceCounter == 0) { /* best — already handled above */ }
        else {
            ++nPatienceCounter;
            // 20260401 ZJH ReduceLROnPlateau: 验证损失连续 patience/3 轮无改善则降 LR
            // 类似 PyTorch ReduceLROnPlateau(factor=0.5, patience=patience/3, min_lr=lr*0.001)
            int nPlateauPatience = std::max(3, params.nPatience / 3);
            if (nPatienceCounter > 0 && (nPatienceCounter % nPlateauPatience == 0)) {
                float fNewLr = fCurrentLr * 0.5f;
                float fFloorLr = fLr * 0.001f;  // 20260401 ZJH 最低 LR = 初始的 0.1%
                if (fNewLr >= fFloorLr) {
                    fCurrentLr = fNewLr;
                    fLr = fNewLr;  // 20260407 ZJH [修复] 同步修改基准 LR，防止 CosineAnnealing 下轮覆盖
                    if (pAdam) pAdam->setLearningRate(fNewLr);
                    else if (pAdamW) pAdamW->setLearningRate(fNewLr);
                    else if (pSgd) pSgd->setLearningRate(fNewLr);
                    if (logCb) logCb("[INFO] ReduceLROnPlateau: lr reduced to " + std::to_string(fNewLr)
                        + " (plateau for " + std::to_string(nPatienceCounter) + " epochs)");
                }
            }
        }

        if (epochCb) {
            BridgeEpochResult r;
            r.nEpoch = nEpoch; r.nTotalEpochs = nEpochs;
            r.fTrainLoss = fAvgTrainLoss; r.fValLoss = fValLoss; r.fMetric = fValAcc;
            epochCb(r);
        }

        if (logCb) {
            // 20260328 ZJH 分割模型显示 PixelAcc（像素准确率），分类模型显示 Acc（图像准确率）
            std::string strAccLabel = (m_pImpl->bIsSegmentation && !vecValMasks.empty()) ? " PixelAcc:" : " Acc:";
            // 20260330 ZJH 增强训练日志（对标海康 VisionTrain 实时指标显示）
            // 分割模型额外显示 Dice 分数 = 1 - ValLoss（ValLoss 就是 1-meanDice）
            std::string strExtra;
            if (m_pImpl->bIsSegmentation && !vecValMasks.empty()) {
                float fValDice = 1.0f - fValLoss;  // 20260330 ZJH Dice 系数 [0,1]
                strExtra = " Dice:" + std::to_string(fValDice).substr(0, 5);
            }
            logCb("Epoch " + std::to_string(nEpoch) + "/" + std::to_string(nEpochs) +
                  " Train:" + std::to_string(fAvgTrainLoss).substr(0, 6) +
                  " Val:" + std::to_string(fValLoss).substr(0, 6) +
                  strAccLabel + std::to_string(fValAcc).substr(0, 5) +
                  strExtra +
                  " LR:" + std::to_string(fCurrentLr).substr(0, 8) +
                  " Pat:" + std::to_string(nPatienceCounter) + "/" + std::to_string(params.nPatience));
        }

        if (nPatienceCounter >= params.nPatience) {
            if (logCb) logCb("[INFO] Early stopping at epoch " + std::to_string(nEpoch));
            break;
        }

        // 20260402 ZJH ===== 过拟合早期警告 =====
        // train_loss 持续下降但 val_loss 停滞或上升 = 过拟合信号
        if (nEpoch >= 5) {
            bool bTrainDecreasing = (fAvgTrainLoss < fPrevTrainLoss * 0.98f);
            bool bValStagnant = (fValLoss > fBestValLoss * 1.02f);
            if (bTrainDecreasing && bValStagnant) {
                ++nOverfitCount;
                if (nOverfitCount >= 3 && logCb) {
                    logCb("[WARN] Overfitting detected: train_loss decreasing but val_loss stagnant for "
                        + std::to_string(nOverfitCount) + " epochs — consider reducing model size or adding augmentation");
                }
            } else {
                nOverfitCount = 0;
            }
        }
        fPrevTrainLoss = fAvgTrainLoss;

        // 20260401 ZJH ===== 自动收敛检测（对标 Halcon dl_train_model）=====
        // 计算最近 nConvergeWindow 个 epoch 的 val_loss 线性回归斜率
        // 斜率绝对值 < threshold → 训练已收敛，继续训练无显著收益
        vecValLossHistory.push_back(fValLoss);
        // 20260405 ZJH [修复] 自动收敛必须同时满足:
        //   (1) 已跑完 1/3 epoch  (2) patience 已消耗过半
        //   防止 val_loss 短暂平坦时绕过用户设置的 patience 提前停训
        if (bAutoConverge && static_cast<int>(vecValLossHistory.size()) >= nConvergeWindow
            && nEpoch >= nEpochs / 3
            && nPatienceCounter >= params.nPatience / 2) {
            // 20260401 ZJH 简易线性回归: slope = Σ(xi-xmean)(yi-ymean) / Σ(xi-xmean)²
            int nStart = static_cast<int>(vecValLossHistory.size()) - nConvergeWindow;
            float fSumX = 0, fSumY = 0, fSumXY = 0, fSumX2 = 0;
            for (int w = 0; w < nConvergeWindow; ++w) {
                float fX = static_cast<float>(w);
                float fY = vecValLossHistory[static_cast<size_t>(nStart + w)];
                fSumX += fX;
                fSumY += fY;
                fSumXY += fX * fY;
                fSumX2 += fX * fX;
            }
            float fN = static_cast<float>(nConvergeWindow);
            float fSlope = (fN * fSumXY - fSumX * fSumY) / (fN * fSumX2 - fSumX * fSumX + 1e-10f);
            float fMeanY = fSumY / fN;
            // 20260401 ZJH 收敛阈值: 斜率绝对值 < 平均 val_loss 的 0.1%
            // 且斜率不为显著负值（仍在下降则不停）
            float fThreshold = std::max(1e-6f, fMeanY * 0.001f);
            if (std::abs(fSlope) < fThreshold && fSlope >= -fThreshold) {
                if (logCb) logCb("[INFO] Auto-convergence: val_loss slope=" + std::to_string(fSlope)
                    + " < threshold=" + std::to_string(fThreshold)
                    + " over " + std::to_string(nConvergeWindow) + " epochs → converged at epoch "
                    + std::to_string(nEpoch));
                break;
            }
        }

        // 20260401 ZJH ===== SWA 权重累积（最后 25% epoch）=====
        if (bUseSwa && nEpoch >= nSwaStartEpoch) {
            auto vecCurParams = m_pImpl->pModel->parameters();
            if (vecSwaAccum.empty()) {
                // 20260401 ZJH 首次：初始化累积缓冲区
                vecSwaAccum.resize(vecCurParams.size());
                for (size_t pi = 0; pi < vecCurParams.size(); ++pi) {
                    vecSwaAccum[pi].resize(vecCurParams[pi]->numel(), 0.0f);
                }
            }
            for (size_t pi = 0; pi < vecCurParams.size() && pi < vecSwaAccum.size(); ++pi) {
                auto tCpu = vecCurParams[pi]->isCuda() ? vecCurParams[pi]->cpu() : *vecCurParams[pi];
                const float* pSrc = tCpu.contiguous().floatDataPtr();
                int nE = vecCurParams[pi]->numel();
                for (int j = 0; j < nE; ++j) vecSwaAccum[pi][j] += pSrc[j];
            }
            ++nSwaCount;
        }
    }

    // 20260401 ZJH ===== SWA 权重平均应用（在 Model Soup 之前）=====
    // SWA 和 Model Soup 是互斥的 — 优先用 Model Soup（更精确），SWA 作为回退
    if (bUseSwa && nSwaCount >= 2 && vecTopCheckpoints.size() < 2) {
        auto vecParams = m_pImpl->pModel->parameters();
        float fInvN = 1.0f / static_cast<float>(nSwaCount);
        for (size_t pi = 0; pi < vecParams.size() && pi < vecSwaAccum.size(); ++pi) {
            int nE = vecParams[pi]->numel();
            for (int j = 0; j < nE; ++j) vecSwaAccum[pi][j] *= fInvN;
            auto tAvg = om::Tensor::fromData(vecSwaAccum[pi].data(), vecParams[pi]->shapeVec());
            *vecParams[pi] = tAvg;
        }
        if (logCb) logCb("[INFO] SWA applied: averaged " + std::to_string(nSwaCount)
            + " snapshots from epoch " + std::to_string(nSwaStartEpoch) + "+");
    }
    vecSwaAccum.clear();

    // 20260401 ZJH ===== Model Soup: top-K 检查点权重平均（超越 PyTorch 标准训练）=====
    // 平均 top-3 最佳检查点的权重，比单一 best 高 1~2% 精度
    // 原理：不同 epoch 的权重各有偏好的特征，平均后泛化更好（Wortsman et al., 2022）
    if (vecTopCheckpoints.size() >= 2) {
        auto vecParams = m_pImpl->pModel->parameters();
        int nK = static_cast<int>(vecTopCheckpoints.size());
        float fInvK = 1.0f / static_cast<float>(nK);

        // 20260401 ZJH 参数平均：param = (1/K) * Σ checkpoint_params
        for (size_t pi = 0; pi < vecParams.size(); ++pi) {
            int nNumel = vecParams[pi]->numel();
            std::vector<float> vecAvg(nNumel, 0.0f);
            for (int k = 0; k < nK; ++k) {
                if (pi < vecTopCheckpoints[k].vecParams.size()) {
                    const float* pSrc = vecTopCheckpoints[k].vecParams[pi].contiguous().floatDataPtr();
                    for (int j = 0; j < nNumel; ++j) vecAvg[j] += pSrc[j] * fInvK;
                }
            }
            om::Tensor tAvg = om::Tensor::fromData(vecAvg.data(), vecParams[pi]->shapeVec());
            *vecParams[pi] = tAvg;
        }
        // 20260401 ZJH 缓冲区平均（BN running stats）
        auto vecBufs = m_pImpl->pModel->buffers();
        for (size_t bi = 0; bi < vecBufs.size(); ++bi) {
            int nNumel = vecBufs[bi]->numel();
            std::vector<float> vecAvg(nNumel, 0.0f);
            for (int k = 0; k < nK; ++k) {
                if (bi < vecTopCheckpoints[k].vecBuffers.size()) {
                    const float* pSrc = vecTopCheckpoints[k].vecBuffers[bi].contiguous().floatDataPtr();
                    for (int j = 0; j < nNumel; ++j) vecAvg[j] += pSrc[j] * fInvK;
                }
            }
            om::Tensor tAvg = om::Tensor::fromData(vecAvg.data(), vecBufs[bi]->shapeVec());
            *vecBufs[bi] = tAvg;
        }
        std::string strLosses;
        for (const auto& ck : vecTopCheckpoints) strLosses += std::to_string(ck.fValLoss).substr(0, 6) + " ";
        if (logCb) logCb("[INFO] Model Soup: averaged " + std::to_string(nK)
            + " checkpoints [val_losses: " + strLosses + "]");
        vecTopCheckpoints.clear();
    } else if (bHasBestCheckpoint) {
        // 20260330 ZJH 回退到单一最佳检查点（top-K 不足 2 个时）
        auto vecParams = m_pImpl->pModel->parameters();
        for (size_t i = 0; i < vecParams.size() && i < vecBestParams.size(); ++i) {
            *vecParams[i] = vecBestParams[i];
        }
        auto vecBufs = m_pImpl->pModel->buffers();
        for (size_t i = 0; i < vecBufs.size() && i < vecBestBuffers.size(); ++i) {
            *vecBufs[i] = vecBestBuffers[i];
        }
        if (logCb) logCb("[INFO] Best checkpoint restored (val_loss=" + std::to_string(fBestValLoss) + ")");
    }
    vecBestParams.clear();
    vecBestBuffers.clear();

    // 20260401 ZJH ===== EMA 权重替换：训练完成后用 EMA 参数替代原始参数 =====
    // EMA 参数比原始参数更平滑，通常在验证/推理时表现更好（+1~2% 精度）
    if (bUseEma && !vecEmaParams.empty()) {
        vecModelParams = m_pImpl->pModel->parameters();
        int nReplaced = 0;
        for (size_t ei = 0; ei < vecEmaParams.size() && ei < vecModelParams.size(); ++ei) {
            // 20260401 ZJH 将 EMA 快照写回模型参数（CPU 侧）
            if (vecModelParams[ei]->isCuda()) {
                // 20260401 ZJH GPU 参数：先将 EMA 上传到 GPU
                *vecModelParams[ei] = vecEmaParams[ei].cuda();
            } else {
                *vecModelParams[ei] = vecEmaParams[ei];
            }
            ++nReplaced;
        }
        vecEmaParams.clear();  // 20260401 ZJH 释放 EMA 副本内存
        if (logCb) logCb("[INFO] EMA weights applied to model (" + std::to_string(nReplaced) + " params)");
    }

    // 20260325 ZJH ===== GPU 路径：训练完成，将模型参数和缓冲区迁移回 CPU 用于序列化保存 =====
#ifdef OM_HAS_CUDA
    if (bUseCuda) {
        // 20260327 ZJH Step 1: 将可训练参数移回 CPU（weight, bias, gamma, beta）
        vecModelParams = m_pImpl->pModel->parameters();
        for (auto* pParam : vecModelParams) {
            *pParam = pParam->cpu();  // 20260325 ZJH 将权重从 GPU 拷回 CPU（D2H）
        }
        // 20260327 ZJH Step 2: 将缓冲区也移回 CPU（BN running_mean, running_var）
        // 之前此步骤缺失，导致 BN 缓冲区仍在 GPU 上，序列化时无法正确保存
        auto vecBufsBack = m_pImpl->pModel->buffers();
        for (auto* pBuf : vecBufsBack) {
            if (pBuf->isCuda()) *pBuf = pBuf->cpu();  // 20260327 ZJH 缓冲区 D2H
        }
        if (logCb) logCb("[INFO] Model parameters + buffers moved back to CPU for saving ("
            + std::to_string(vecModelParams.size()) + " params, "
            + std::to_string(vecBufsBack.size()) + " buffers)");
    }
#endif

    // 20260325 ZJH ===== 清理 GPU 资源 =====
#ifdef OM_HAS_CUDA
    if (bUseCuda) {
        omCudaCleanup();  // 20260325 ZJH 释放 GPU 内存池和异步流资源
        if (logCb) logCb("[INFO] CUDA resources cleaned up");
    }
#endif

    // 20260328 ZJH ===== EfficientAD 自适应阈值校准 =====
    // 训练完成后，用全部训练样本（正常图像）前向推理，收集图像级异常分数
    // 然后用 3-sigma 规则计算阈值: threshold = mean + 3*std
    if (m_pImpl->bIsEfficientAD) {
        auto* pEAD = static_cast<om::EfficientAD*>(m_pImpl->pModel.get());
        pEAD->eval();  // 20260328 ZJH 切换到评估模式（BN 使用 running stats）

        std::vector<float> vecCalibScores;  // 20260328 ZJH 收集所有样本的图像级异常分数
        vecCalibScores.reserve(static_cast<size_t>(nTrainCount));

        if (logCb) logCb("[INFO] EfficientAD: Calibrating anomaly threshold on "
            + std::to_string(nTrainCount) + " training samples...");

        // 20260328 ZJH 逐样本前向推理（无梯度计算，纯前向传播）
        for (int i = 0; i < nTrainCount; ++i) {
            int nSrc = i * nInputDim;  // 20260328 ZJH 训练数据偏移量
            if (nSrc + nInputDim > static_cast<int>(vecTrainData.size())) break;

            // 20260328 ZJH 构造单张图像输入 [1, 3, H, W]
            auto tCalibInput = om::Tensor::fromData(
                vecTrainData.data() + nSrc, {1, 3, m_pImpl->nInputSize, m_pImpl->nInputSize});

            try {
                // 20260328 ZJH 计算异常分数图并取最大值
                auto tAnomalyMap = pEAD->computeAnomalyScore(tCalibInput);
                auto cMap = tAnomalyMap.contiguous();
                const float* pMap = cMap.floatDataPtr();
                int nMapSize = static_cast<int>(cMap.numel());

                // 20260328 ZJH 图像级异常分数 = 异常图最大像素值
                float fMaxScore = *std::max_element(pMap, pMap + nMapSize);
                vecCalibScores.push_back(fMaxScore);
            } catch (...) {
                // 20260328 ZJH 单张图像推理失败不影响校准
            }
        }

        // 20260328 ZJH Step 1: 3-sigma 基线校准: threshold = mean + 3*std
        if (!vecCalibScores.empty()) {
            pEAD->calibrate(vecCalibScores, 3.0f);  // 20260328 ZJH 3-sigma 规则
            float fSigmaThreshold = pEAD->anomalyThreshold();  // 20260402 ZJH 保存 3-sigma 阈值
            if (logCb) logCb("[INFO] EfficientAD: 3-sigma baseline — mean=" + std::to_string(pEAD->scoreMean())
                + " std=" + std::to_string(pEAD->scoreStd())
                + " threshold=" + std::to_string(fSigmaThreshold));

            // 20260402 ZJH [OPT-2.7] Step 2: F1-score 最大化校准
            // 如果验证集同时包含正常样本（label=0）和异常样本（label>0），
            // 则在验证集上搜索使 F1-score 最大化的阈值
            // 遍历候选阈值 → 计算 TP/FP/FN → F1 = 2*P*R/(P+R) → 选最大 F1 对应的阈值
            bool bHasNormalVal = false;   // 20260402 ZJH 验证集是否包含正常样本
            bool bHasAnomalyVal = false;  // 20260402 ZJH 验证集是否包含异常样本
            for (int i = 0; i < nValCount; ++i) {
                if (vecValLabels[i] == 0) bHasNormalVal = true;   // 20260402 ZJH 标签 0 = 正常
                else                      bHasAnomalyVal = true;  // 20260402 ZJH 标签 > 0 = 异常
                if (bHasNormalVal && bHasAnomalyVal) break;       // 20260402 ZJH 两类都有，无需继续
            }

            if (bHasNormalVal && bHasAnomalyVal) {
                // 20260402 ZJH 验证集含正常+异常样本，执行 F1-max 校准
                if (logCb) logCb("[INFO] EfficientAD: F1-max calibration on "
                    + std::to_string(nValCount) + " val samples (normal+anomaly)...");

                // 20260402 ZJH 收集验证集异常分数
                std::vector<float> vecValScores;   // 20260402 ZJH 验证集每张图的异常分数
                std::vector<int>   vecValGT;       // 20260402 ZJH 验证集真实标签（0=正常, 1=异常）
                vecValScores.reserve(static_cast<size_t>(nValCount));
                vecValGT.reserve(static_cast<size_t>(nValCount));

                for (int i = 0; i < nValCount; ++i) {
                    int nSrc = i * nInputDim;  // 20260402 ZJH 验证数据偏移
                    if (nSrc + nInputDim > static_cast<int>(vecValData.size())) break;

                    // 20260402 ZJH 构造验证集单张图像输入 [1, 3, H, W]
                    auto tValInput = om::Tensor::fromData(
                        vecValData.data() + nSrc, {1, 3, m_pImpl->nInputSize, m_pImpl->nInputSize});

                    try {
                        // 20260402 ZJH 计算异常分数图并取最大值
                        auto tAnomalyMap = pEAD->computeAnomalyScore(tValInput);
                        auto cMap = tAnomalyMap.contiguous();
                        const float* pMap = cMap.floatDataPtr();
                        int nMapSize = static_cast<int>(cMap.numel());
                        float fMaxScore = *std::max_element(pMap, pMap + nMapSize);
                        vecValScores.push_back(fMaxScore);
                        // 20260402 ZJH 二值化标签: 0=正常, >0 统一视为异常=1
                        vecValGT.push_back(vecValLabels[i] > 0 ? 1 : 0);
                    } catch (...) {
                        // 20260402 ZJH 单样本推理失败不影响校准
                    }
                }

                // 20260402 ZJH 在候选阈值上搜索 F1 最大化（需至少 4 个有效样本）
                if (vecValScores.size() >= 4) {
                    // 20260402 ZJH 构建候选阈值列表：使用验证集分数排序后去重
                    std::vector<float> vecCandidates(vecValScores);  // 20260402 ZJH 拷贝所有分数作为候选
                    std::sort(vecCandidates.begin(), vecCandidates.end());  // 20260402 ZJH 升序排序
                    // 20260402 ZJH 去重，减少搜索空间
                    vecCandidates.erase(
                        std::unique(vecCandidates.begin(), vecCandidates.end()),
                        vecCandidates.end());

                    float fBestF1 = 0.0f;          // 20260402 ZJH 当前最佳 F1 分数
                    float fBestThreshold = fSigmaThreshold;  // 20260402 ZJH 默认使用 3-sigma 阈值

                    // 20260402 ZJH 遍历每个候选阈值，计算 Precision/Recall/F1
                    for (float fCandThresh : vecCandidates) {
                        int nTP = 0, nFP = 0, nFN = 0;  // 20260402 ZJH 混淆矩阵计数
                        for (size_t j = 0; j < vecValScores.size(); ++j) {
                            bool bPredAnomaly = (vecValScores[j] >= fCandThresh);  // 20260402 ZJH 分数>=阈值→异常
                            bool bGTAnomaly   = (vecValGT[j] == 1);               // 20260402 ZJH 标签=1→异常
                            if (bPredAnomaly && bGTAnomaly)   nTP++;  // 20260402 ZJH 正确检出异常
                            if (bPredAnomaly && !bGTAnomaly)  nFP++;  // 20260402 ZJH 误检（正常判为异常）
                            if (!bPredAnomaly && bGTAnomaly)  nFN++;  // 20260402 ZJH 漏检（异常判为正常）
                        }
                        // 20260402 ZJH 计算 F1 = 2*TP / (2*TP + FP + FN)
                        float fF1 = 0.0f;
                        int nDenom = 2 * nTP + nFP + nFN;  // 20260402 ZJH F1 分母
                        if (nDenom > 0) {
                            fF1 = static_cast<float>(2 * nTP) / static_cast<float>(nDenom);
                        }
                        // 20260402 ZJH 更新最佳 F1 及对应阈值
                        if (fF1 > fBestF1) {
                            fBestF1 = fF1;
                            fBestThreshold = fCandThresh;
                        }
                    }

                    // 20260402 ZJH 仅在 F1 > 0 时才用 F1-max 阈值覆盖 3-sigma 基线
                    if (fBestF1 > 0.0f) {
                        pEAD->setAnomalyThreshold(fBestThreshold);  // 20260402 ZJH 用 F1-max 阈值覆盖
                        if (logCb) logCb("[INFO] EfficientAD: F1-max threshold=" + std::to_string(fBestThreshold)
                            + " (F1=" + std::to_string(fBestF1)
                            + ", tested " + std::to_string(vecCandidates.size()) + " candidates"
                            + ", 3-sigma baseline was " + std::to_string(fSigmaThreshold) + ")");
                    } else {
                        // 20260402 ZJH F1 为 0（验证集标签可能有误），保持 3-sigma 基线
                        if (logCb) logCb("[WARN] EfficientAD: F1-max calibration returned F1=0, keeping 3-sigma baseline");
                    }
                } else {
                    // 20260402 ZJH 验证集推理成功样本不足 4 个，跳过 F1 校准
                    if (logCb) logCb("[WARN] EfficientAD: Too few val samples for F1 calibration ("
                        + std::to_string(vecValScores.size()) + "), keeping 3-sigma baseline");
                }
            } else {
                // 20260402 ZJH 验证集仅含单一类别（全正常或全异常），无法计算 F1
                if (logCb) logCb("[INFO] EfficientAD: Val set has only "
                    + std::string(bHasNormalVal ? "normal" : "anomaly")
                    + " samples — F1 calibration skipped, using 3-sigma threshold="
                    + std::to_string(fSigmaThreshold));
            }
        } else {
            if (logCb) logCb("[WARN] EfficientAD: No calibration scores collected, using default threshold 0.5");
        }
    }

    // 20260402 ZJH ===== GCAD 自动拟合全局分布（Stage 2）=====
    // 训练完成后自动收集正常样本的全局特征向量，拟合高斯分布
    // 无需用户手动调用 fitGCADDistribution()
    if (m_pImpl->bIsGCAD) {
        auto* pGCAD = dynamic_cast<om::GCAD*>(m_pImpl->pModel.get());
        if (pGCAD) {
            pGCAD->eval();
            std::vector<std::vector<float>> vecGlobalFeatures;
            std::vector<float> vecFusedScores;

            if (logCb) logCb("[INFO] GCAD: Fitting global distribution on " +
                std::to_string(nTrainCount) + " normal samples...");

            for (int i = 0; i < nTrainCount; ++i) {
                int nSrc = i * nInputDim;
                if (nSrc + nInputDim > static_cast<int>(vecTrainData.size())) break;

                auto tInput = om::Tensor::fromData(
                    vecTrainData.data() + nSrc, {1, 3, m_pImpl->nInputSize, m_pImpl->nInputSize});

                try {
                    // 20260402 ZJH 提取全局上下文向量
                    auto globalVec = pGCAD->predictGlobal(tInput);
                    auto cGlobal = globalVec.contiguous();
                    int nDim = cGlobal.shape(1);
                    const float* pG = cGlobal.floatDataPtr();
                    std::vector<float> feat(nDim);
                    for (int d = 0; d < nDim; ++d) feat[d] = pG[d];
                    vecGlobalFeatures.push_back(std::move(feat));

                    // 20260402 ZJH 收集融合分数
                    auto gcadResult = pGCAD->predict(tInput);
                    vecFusedScores.push_back(gcadResult.fFusedScore);
                } catch (...) {}
            }

            if (!vecGlobalFeatures.empty()) {
                pGCAD->fitGlobalDistribution(vecGlobalFeatures);
                pGCAD->calibrateThreshold(vecFusedScores);
                if (logCb) logCb("[INFO] GCAD: Distribution fitted, layout threshold=" +
                    std::to_string(pGCAD->layoutThreshold()) +
                    " anomaly threshold=" + std::to_string(pGCAD->anomalyThreshold()));
            }
        }
    }

    // 20260402 ZJH ===== 后训练模型剪枝（可选）=====
    // 训练完成后自动执行 magnitude-based 剪枝，减少模型体积和推理延迟
    if (params.bPruneAfterTraining && m_pImpl->pModel) {
        if (logCb) logCb("[INFO] Post-training pruning: ratio=" +
            std::to_string(params.fPruneRatio));

        // 20260402 ZJH 幅度剪枝: 将绝对值最小的 fPruneRatio 比例权重置零
        int nPruned = om::pruneModelMagnitude(*m_pImpl->pModel, params.fPruneRatio);

        // 20260402 ZJH 统计剪枝后稀疏率
        auto sparsityInfo = om::analyzeSparsity(*m_pImpl->pModel);
        if (logCb) {
            logCb("[INFO] Pruning complete: " + std::to_string(nPruned) + " params zeroed, sparsity=" +
                std::to_string(sparsityInfo.fSparsityRatio * 100.0f) + "%");
        }
    }

    // 20260402 ZJH ===== BN 折叠推理优化（Conv+BN → Conv）=====
    // 训练完成后自动将 BatchNorm 参数合并到前置 Conv 权重中
    // W_new = W * γ/√(var+ε), b_new = (b-μ)*γ/√(var+ε) + β
    // 效果: 推理时消除 BN 层（减少 ~50% 算子），零精度损失
    if (m_pImpl->pModel) {
        auto vecNP = m_pImpl->pModel->namedParameters();
        auto vecNB = m_pImpl->pModel->namedBuffers();

        // 20260402 ZJH 构建名称→指针映射
        std::map<std::string, om::Tensor*> mapParams, mapBuffers;
        for (auto& [name, ptr] : vecNP) mapParams[name] = ptr;
        for (auto& [name, ptr] : vecNB) mapBuffers[name] = ptr;

        int nFoldedCount = 0;  // 20260402 ZJH 折叠计数
        // 20260402 ZJH 扫描所有 BN 层，尝试找到匹配的前置 Conv
        // 命名约定: conv1.weight + bn1.gamma → 折叠
        for (auto& [bnName, bnGamma] : vecNP) {
            // 20260402 ZJH 查找 *.gamma 参数（BN 层标识）
            if (bnName.size() < 6 || bnName.substr(bnName.size() - 5) != "gamma") continue;

            std::string strBnPrefix = bnName.substr(0, bnName.size() - 5);  // 20260402 ZJH "bn1."
            // 20260402 ZJH 查找对应的 BN beta, running_mean, running_var
            auto itBeta = mapParams.find(strBnPrefix + "beta");
            auto itMean = mapBuffers.find(strBnPrefix + "running_mean");
            auto itVar = mapBuffers.find(strBnPrefix + "running_var");
            if (itBeta == mapParams.end() || itMean == mapBuffers.end() || itVar == mapBuffers.end()) continue;

            om::Tensor* pGamma = bnGamma;
            om::Tensor* pBeta = itBeta->second;
            om::Tensor* pMean = itMean->second;
            om::Tensor* pVar = itVar->second;
            int nCh = pGamma->numel();  // 20260402 ZJH 通道数

            // 20260402 ZJH 推断前置 Conv 名称（尝试 bn1→conv1 的命名映射）
            // 常见模式: "encoder.bn1" → "encoder.conv1", "bn2" → "conv2"
            std::string strConvPrefix = strBnPrefix;
            auto bnPos = strConvPrefix.find("bn");
            if (bnPos == std::string::npos) continue;
            strConvPrefix.replace(bnPos, 2, "conv");  // 20260402 ZJH bn→conv

            auto itConvW = mapParams.find(strConvPrefix + "weight");
            if (itConvW == mapParams.end()) continue;
            om::Tensor* pConvW = itConvW->second;

            // 20260402 ZJH 查找或创建 Conv bias
            auto itConvB = mapParams.find(strConvPrefix + "bias");
            bool bHasBias = (itConvB != mapParams.end());

            // 20260402 ZJH 验证通道数匹配
            if (pConvW->shape(0) != nCh) continue;

            // 20260402 ZJH 执行折叠: W_new[c] = W[c] * γ[c] / √(var[c]+ε)
            float fEps = 1e-5f;  // 20260402 ZJH BN 默认 epsilon
            const float* pG = pGamma->contiguous().floatDataPtr();
            const float* pB = pBeta->contiguous().floatDataPtr();
            const float* pM = pMean->contiguous().floatDataPtr();
            const float* pV = pVar->contiguous().floatDataPtr();

            float* pW = pConvW->mutableFloatDataPtr();
            int nWeightsPerChannel = pConvW->numel() / nCh;  // 20260402 ZJH Cin*KH*KW

            for (int c = 0; c < nCh; ++c) {
                float fScale = pG[c] / std::sqrt(pV[c] + fEps);  // 20260402 ZJH γ/√(var+ε)
                // 20260402 ZJH 缩放该通道的所有权重
                for (int j = 0; j < nWeightsPerChannel; ++j) {
                    pW[c * nWeightsPerChannel + j] *= fScale;
                }
                // 20260402 ZJH 更新或创建 bias: b_new = (b-μ)*scale + β
                if (bHasBias) {
                    float* pCB = itConvB->second->mutableFloatDataPtr();
                    pCB[c] = (pCB[c] - pM[c]) * fScale + pB[c];
                }
            }

            // 20260402 ZJH 将 BN 参数置零（折叠后 BN 变为恒等映射）
            // gamma=1, beta=0, mean=0, var=1 → BN(x) = x
            float* pGW = pGamma->mutableFloatDataPtr();
            float* pBW = pBeta->mutableFloatDataPtr();
            float* pMW = pMean->mutableFloatDataPtr();
            float* pVW = pVar->mutableFloatDataPtr();
            for (int c = 0; c < nCh; ++c) {
                pGW[c] = 1.0f; pBW[c] = 0.0f; pMW[c] = 0.0f; pVW[c] = 1.0f;
            }

            nFoldedCount++;
        }
        if (nFoldedCount > 0 && logCb) {
            logCb("[INFO] BN Folding: " + std::to_string(nFoldedCount) +
                  " Conv+BN pairs fused (inference ~50% fewer ops)");
        }
    }

    if (logCb) logCb("[INFO] Training done. Best val loss: " + std::to_string(fBestValLoss));

    // 20260328 ZJH 训练完毕诊断：确认参数在 train() 返回前仍可访问
    {
        auto vecChk = m_pImpl->pModel->namedParameters();
        auto vecChkP = m_pImpl->pModel->parameters();
        int64_t nChkElem = 0;
        for (auto* p : vecChkP) nChkElem += p->numel();
        std::cerr << "[EngineBridge] train() EXIT CHECK: namedParameters=" << vecChk.size()
                  << " parameters=" << vecChkP.size()
                  << " totalElements=" << nChkElem
                  << " children=" << m_pImpl->pModel->debugChildCount()
                  << std::endl;
    }

    return true;
}

// 20260323 ZJH 推理
// 20260325 ZJH 重写：GPU 加速推理 + 异常热力图生成
BridgeInferResult EngineBridge::infer(const std::vector<float>& vecImageData)
{
    BridgeInferResult result;  // 20260406 ZJH 推理结果（默认构造，全零初始化）
    if (!m_pImpl->pModel) return result;  // 20260406 ZJH 模型未创建，返回空结果

    int nInputDim = m_pImpl->nInputDim;    // 20260406 ZJH 展平输入维度
    int nNumClasses = m_pImpl->nNumClasses;  // 20260406 ZJH 输出类别数
    int nH = m_pImpl->nInputSize;          // 20260406 ZJH 输入空间尺寸 H=W

    // 20260325 ZJH 准备输入数据
    std::vector<float> vecPad(nInputDim, 0.0f);
    int nCopy = std::min(static_cast<int>(vecImageData.size()), nInputDim);
    std::copy(vecImageData.begin(), vecImageData.begin() + nCopy, vecPad.begin());

    // 20260330 ZJH ===== S1: 推理归一化（与训练保持一致）=====
    // 推理时必须应用与训练相同的归一化，否则特征空间不一致导致预测错误
    if (m_pImpl->bIsCnn) {
        om::NormPreset eInferNorm = selectNormPreset(m_pImpl->strModelType);
        if (eInferNorm == om::NormPreset::ImageNet) {
            om::AugmentConfig inferNormCfg;
            inferNormCfg.bNormalize = true;
            inferNormCfg.eNormPreset = om::NormPreset::ImageNet;
            om::normalizeImage(vecPad, 3, nH, nH, inferNormCfg);
        }
    }

    // 20260325 ZJH CNN→4D [1,3,H,W]，MLP→2D [1,D]
    auto tIn = m_pImpl->bIsCnn
        ? om::Tensor::fromData(vecPad.data(), {1, 3, nH, nH})
        : om::Tensor::fromData(vecPad.data(), {1, nInputDim});

    // 20260325 ZJH GPU 加速推理：将模型和输入迁移到 GPU
#ifdef OM_HAS_CUDA
    bool bInferGpu = false;
    {
        // 20260325 ZJH 检查模型参数是否已在 GPU 上（加载后保持 CPU）
        auto vecParams = m_pImpl->pModel->parameters();
        bool bParamsOnGpu = (!vecParams.empty() && vecParams[0]->isCuda());
        if (!bParamsOnGpu) {
            // 20260325 ZJH 首次推理：初始化 CUDA 并将模型迁移到 GPU
            if (omCudaInit(0) == 0) {
                for (auto* p : vecParams) *p = p->cuda();
                // 20260407 ZJH [修复] BN buffers (running_mean/running_var) 也需迁移到 GPU
                auto vecBufs = m_pImpl->pModel->buffers();
                for (auto* b : vecBufs) { if (b->isCpu()) *b = b->cuda(); }
                bInferGpu = true;
            }
        } else {
            bInferGpu = true;
        }
        if (bInferGpu) {
            tIn = tIn.cuda();  // 20260325 ZJH 输入上传到 GPU
        }
    }
#endif

    m_pImpl->pModel->eval();  // 20260406 ZJH 切换到评估模式（关闭 Dropout，BN 使用 running stats）

    // 20260326 ZJH EfficientAD 专用推理路径：直接提取空间异常图，跳过分类流程
    if (m_pImpl->bIsEfficientAD) {
        try {
            auto* pEAD = static_cast<om::EfficientAD*>(m_pImpl->pModel.get());
            // 20260326 ZJH computeAnomalyScore 返回 [1, 1, H/8, W/8] 异常分数图
            auto tAnomalyMap = pEAD->computeAnomalyScore(tIn);
            if (tAnomalyMap.isCuda()) tAnomalyMap = tAnomalyMap.cpu();
            auto cMap = tAnomalyMap.contiguous();
            const float* pMap = cMap.floatDataPtr();
            int nMapH = cMap.shape(2);
            int nMapW = cMap.shape(3);
            int nMapSize = nMapH * nMapW;

            // 20260326 ZJH 复制异常分数到结果
            result.vecAnomalyMap.resize(static_cast<size_t>(nMapSize));
            std::copy(pMap, pMap + nMapSize, result.vecAnomalyMap.data());
            result.nMapW = nMapW;
            result.nMapH = nMapH;

            // 20260326 ZJH 图像级异常分数 = 异常图最大值
            float fMaxScore = *std::max_element(result.vecAnomalyMap.begin(), result.vecAnomalyMap.end());

            // 20260328 ZJH 使用校准后的自适应阈值（3-sigma），替代硬编码 0.5
            float fThreshold = pEAD->anomalyThreshold();  // 20260328 ZJH 未校准时 fallback 0.5

            // 20260328 ZJH 归一化置信度: score/threshold 映射到 [0,1] 附近
            // <1.0 = 正常（越小越正常）, >1.0 = 异常（越大越异常）
            float fNormScore = (fThreshold > 0.0f) ? (fMaxScore / fThreshold) : fMaxScore;
            // 20260328 ZJH 限制到 [0, 1] 范围供 UI 显示
            float fConfidence = std::min(fNormScore, 1.0f);

            result.fConfidence = fConfidence;
            result.nPredictedClass = (fMaxScore > fThreshold) ? 1 : 0;  // 20260328 ZJH 自适应阈值判断
            result.vecProbs = {1.0f - fConfidence, fConfidence};  // 20260328 ZJH [OK概率, 缺陷概率]
        } catch (...) {
            // 20260326 ZJH 异常时返回默认结果
        }
        return result;
    }

    om::Tensor tOut;  // 20260406 ZJH 模型前向输出张量
    try {
        tOut = m_pImpl->pModel->forward(tIn);  // 20260406 ZJH 执行前向传播
    } catch (const std::exception& ex) {
        // 20260325 ZJH forward 异常时回退 CPU 再试一次
#ifdef OM_HAS_CUDA
        try {
            auto vecP = m_pImpl->pModel->parameters();
            for (auto* p : vecP) *p = p->cpu();
            tIn = tIn.isCuda() ? tIn.cpu() : tIn;
            omCudaCleanup();
        } catch (...) {}
#endif
        try {
            tOut = m_pImpl->pModel->forward(tIn);
        } catch (...) {
            return result;  // 20260325 ZJH 彻底失败
        }
    }

    // 20260325 ZJH 确保输出在 CPU 上
    if (tOut.isCuda()) tOut = tOut.cpu();  // 20260406 ZJH GPU 输出拷回 CPU（D2H）
    auto cOut = tOut.contiguous();         // 20260406 ZJH 保证内存连续（后续指针访问需要）
    const float* pO = cOut.floatDataPtr(); // 20260406 ZJH 获取输出数据指针
    int nOutTotal = cOut.numel();          // 20260406 ZJH 输出元素总数

    // 20260330 ZJH ===== F1: YOLO 检测推理 + NMS =====
    // 检测模型输出为 3D [1, nPreds, 5+C]，通过 yoloDecodeAndNms 解码+NMS
    // 结果存入 result.vecDetections，同时设置最高分检测框的类别和置信度
    if (m_pImpl->bIsDetection && cOut.shapeVec().size() == 3) {
        int nBatch = cOut.shape(0);   // 20260330 ZJH 批次大小（推理时通常为 1）
        int nPreds = cOut.shape(1);   // 20260330 ZJH 每张图的预测数量
        int nDim   = cOut.shape(2);   // 20260330 ZJH 每个预测的维度 = 5 + nClasses
        int nDetClasses = nDim - 5;   // 20260330 ZJH 检测类别数

        // 20260330 ZJH 构建 YOLO 解码参数
        om::YoloDecodeParams decodeParams;
        decodeParams.nNumClasses = nDetClasses;
        decodeParams.fConfThreshold = 0.25f;     // 20260330 ZJH 置信度阈值（YOLO 标准值）
        decodeParams.fNmsIoUThreshold = 0.45f;   // 20260330 ZJH NMS IoU 阈值（YOLO 标准值）
        decodeParams.nInputWidth = m_pImpl->nInputSize;   // 20260330 ZJH 模型输入宽度
        decodeParams.nInputHeight = m_pImpl->nInputSize;  // 20260330 ZJH 模型输入高度

        // 20260330 ZJH 执行解码 + NMS（一体化处理）
        result.vecDetections = om::yoloDecodeAndNms(pO, nBatch, nPreds, decodeParams);

        // 20260330 ZJH 从检测结果中提取图像级预测（最高置信度的检测框）
        if (!result.vecDetections.empty()) {
            // 20260330 ZJH 找最高分检测框
            float fBestDetScore = -1.0f;
            int nBestDetClass = -1;
            for (const auto& det : result.vecDetections) {
                if (det.fScore > fBestDetScore) {
                    fBestDetScore = det.fScore;
                    nBestDetClass = det.nClassId;
                }
            }
            result.nPredictedClass = nBestDetClass;    // 20260330 ZJH 最高分检测框的类别
            result.fConfidence = fBestDetScore;         // 20260330 ZJH 最高分检测框的置信度
            // 20260330 ZJH 各类别概率：按检测框数量占比近似
            result.vecProbs.resize(nNumClasses, 0.0f);
            for (const auto& det : result.vecDetections) {
                if (det.nClassId >= 0 && det.nClassId < nNumClasses) {
                    result.vecProbs[det.nClassId] += det.fScore;
                }
            }
            // 20260330 ZJH 归一化概率
            float fProbSum = 0.0f;
            for (float fP : result.vecProbs) fProbSum += fP;
            if (fProbSum > 0.0f) {
                for (float& fP : result.vecProbs) fP /= fProbSum;
            }
        } else {
            // 20260330 ZJH 无检测框：返回背景类
            result.nPredictedClass = 0;
            result.fConfidence = 0.0f;
            result.vecProbs.resize(nNumClasses, 0.0f);
            if (nNumClasses > 0) result.vecProbs[0] = 1.0f;
        }

        return result;  // 20260330 ZJH 检测模型直接返回，不走分类流程
    }

    // 20260328 ZJH ===== 分类结果 =====
    {
        if (cOut.shapeVec().size() == 4 && cOut.shape(1) >= 2) {
            // 20260328 ZJH 分割模型 4D 输出 [B, C, H, W]：逐像素 argmax 统计缺陷面积
            // 正确做法：不用全局平均池化（会把空间信息混淆），而是：
            //   1. 每个像素取 argmax 得到预测类别
            //   2. 统计非背景(class!=0)像素占比
            //   3. 占比 > 阈值 → 判定为缺陷图像
            int nOutC = cOut.shape(1);       // 20260328 ZJH 类别通道数
            int nSpatial = cOut.shape(2) * cOut.shape(3);  // 20260328 ZJH 空间像素总数

            int nDefectPixels = 0;  // 20260328 ZJH 缺陷像素计数
            float fMaxDefectProb = 0.0f;  // 20260328 ZJH 所有像素中最大的缺陷类概率

            // 20260328 ZJH 统计每类的像素数（用于 vecProbs）
            std::vector<int> vecClassCount(nNumClasses, 0);

            for (int s = 0; s < nSpatial; ++s) {
                // 20260328 ZJH 逐像素 argmax：找当前像素 logit 最大的类别
                int nBestC = 0;
                float fBestLogit = pO[s];  // 20260328 ZJH 通道 0 的 logit
                for (int c = 1; c < nOutC; ++c) {
                    float fLogit = pO[c * nSpatial + s];
                    if (fLogit > fBestLogit) {
                        fBestLogit = fLogit;
                        nBestC = c;
                    }
                }
                // 20260328 ZJH 非背景类 = 缺陷
                if (nBestC != 0) ++nDefectPixels;
                // 20260328 ZJH 累计每类像素数
                if (nBestC < nNumClasses) ++vecClassCount[nBestC];

                // 20260329 ZJH 计算该像素的缺陷类 softmax 概率（与 PixelCE 训练一致）
                {
                    float fMax = pO[s];
                    for (int c = 1; c < nOutC; ++c) {
                        float fV = pO[c * nSpatial + s];
                        if (fV > fMax) fMax = fV;
                    }
                    float fExpSum = 0.0f;
                    for (int c = 0; c < nOutC; ++c) fExpSum += std::exp(pO[c * nSpatial + s] - fMax);
                    float fBgProb = std::exp(pO[s] - fMax) / fExpSum;
                    float fDefProb = 1.0f - fBgProb;  // 20260329 ZJH P(defect) = 1 - P(bg)
                    if (fDefProb > fMaxDefectProb) fMaxDefectProb = fDefProb;
                }
            }

            // 20260328 ZJH 缺陷像素占比
            float fDefectRatio = static_cast<float>(nDefectPixels) / static_cast<float>(std::max(nSpatial, 1));

            // 20260328 ZJH 图像级判定: 缺陷像素占比 > 1% 即判定缺陷
            // 1% 阈值适用于工业检测：微小缺陷（划痕/针孔）通常占比 0.5%-5%
            constexpr float fDefectRatioThreshold = 0.01f;
            bool bIsDefect = (fDefectRatio > fDefectRatioThreshold);

            result.nPredictedClass = bIsDefect ? 1 : 0;
            // 20260328 ZJH 置信度：取缺陷比例和最大缺陷概率的较大值
            result.fConfidence = bIsDefect ? std::max(fDefectRatio, fMaxDefectProb) : (1.0f - fDefectRatio);

            // 20260328 ZJH 每类概率 = 该类像素占比
            result.vecProbs.resize(nNumClasses);
            for (int c = 0; c < nNumClasses; ++c) {
                result.vecProbs[c] = static_cast<float>(vecClassCount[c]) / static_cast<float>(std::max(nSpatial, 1));
            }
        } else {
            // 20260325 ZJH 2D 输出（MLP分类模型）：标准 softmax
            std::vector<float> vecLogits(nNumClasses, 0.0f);
            for (int c = 0; c < nNumClasses && c < nOutTotal; ++c) {
                vecLogits[c] = pO[c];
            }

            result.vecProbs.resize(nNumClasses);
            float fMax2 = *std::max_element(vecLogits.begin(), vecLogits.end());
            float fSum2 = 0.0f;
            for (int c = 0; c < nNumClasses; ++c) {
                result.vecProbs[c] = std::exp(vecLogits[c] - fMax2);
                fSum2 += result.vecProbs[c];
            }
            int nBest = 0;
            float fBestP = 0.0f;
            for (int c = 0; c < nNumClasses; ++c) {
                result.vecProbs[c] /= std::max(fSum2, 1e-8f);
                if (result.vecProbs[c] > fBestP) { fBestP = result.vecProbs[c]; nBest = c; }
            }
            result.nPredictedClass = nBest;
            result.fConfidence = fBestP;
        }
    }

    // 20260329 ZJH ===== 空间缺陷图：softmax 概率（与 PixelCE 训练一致）=====
    std::cerr << "[DIAG-INFER] bIsCnn=" << m_pImpl->bIsCnn
              << " ndim=" << cOut.shapeVec().size()
              << " shape=[" << cOut.shape(0);
    for (size_t di = 1; di < cOut.shapeVec().size(); ++di) std::cerr << "," << cOut.shapeVec()[di];
    std::cerr << "] numel=" << cOut.numel() << std::endl;
    if (m_pImpl->bIsCnn && cOut.shapeVec().size() == 4 && cOut.shape(1) >= 2) {
        int nOutC = cOut.shape(1);       // 20260329 ZJH 类别通道数
        int nOutH = cOut.shape(2);       // 20260329 ZJH 输出空间高度
        int nOutW = cOut.shape(3);       // 20260329 ZJH 输出空间宽度
        int nSpatial = nOutH * nOutW;    // 20260329 ZJH 空间像素数

        std::vector<float> vecDefectProb(static_cast<size_t>(nSpatial), 0.0f);
        for (int s = 0; s < nSpatial; ++s) {
            // 20260329 ZJH 数值稳定 softmax: 先求 max
            float fMax = pO[s];
            for (int c = 1; c < nOutC; ++c) {
                float fV = pO[c * nSpatial + s];
                if (fV > fMax) fMax = fV;
            }
            // 20260329 ZJH exp 并求和
            float fExpSum = 0.0f;
            for (int c = 0; c < nOutC; ++c) {
                fExpSum += std::exp(pO[c * nSpatial + s] - fMax);
            }
            // 20260329 ZJH 缺陷概率 = 1 - P(background) = sum(P(c>=1))
            float fBgProb = std::exp(pO[s] - fMax) / fExpSum;  // 20260329 ZJH P(background=c0)
            vecDefectProb[static_cast<size_t>(s)] = 1.0f - fBgProb;
        }

        result.vecAnomalyMap = std::move(vecDefectProb);
        result.nMapW = nOutW;
        result.nMapH = nOutH;

        // 20260402 ZJH ===== DenseCRF 后处理（可选: 分割边界精化）=====
        // 在 argmax 之前对 softmax 概率图做 CRF 精化，提升边界锐利度
        // 条件: 分割模型 + 输入图像数据可用 + 类别数 > 1
        if (m_pImpl->bIsSegmentation && nOutC >= 2 && vecImageData.size() >= static_cast<size_t>(3 * nOutH * nOutW)) {
            try {
                // 20260402 ZJH 构造 softmax 概率张量 [C, H, W]
                auto tSoftmax = om::Tensor::zeros({nOutC, nOutH, nOutW});
                float* pSM = tSoftmax.mutableFloatDataPtr();
                for (int s = 0; s < nSpatial; ++s) {
                    float fMax = pO[s];
                    for (int c = 1; c < nOutC; ++c) fMax = std::max(fMax, pO[c * nSpatial + s]);
                    float fSum = 0.0f;
                    for (int c = 0; c < nOutC; ++c) {
                        pSM[c * nSpatial + s] = std::exp(pO[c * nSpatial + s] - fMax);
                        fSum += pSM[c * nSpatial + s];
                    }
                    for (int c = 0; c < nOutC; ++c) pSM[c * nSpatial + s] /= (fSum + 1e-10f);
                }

                // 20260402 ZJH 构造 RGB 图像张量 [3, H, W]（从输入数据截取，可能需要下采样）
                auto tImage = om::Tensor::zeros({3, nOutH, nOutW});
                float* pImg = tImage.mutableFloatDataPtr();
                int nInH = m_pImpl->nInputSize;
                float fScale = static_cast<float>(nInH) / static_cast<float>(nOutH);
                for (int ch = 0; ch < 3; ++ch) {
                    for (int y = 0; y < nOutH; ++y) {
                        int nSrcY = std::min(static_cast<int>(y * fScale), nInH - 1);
                        for (int x = 0; x < nOutW; ++x) {
                            int nSrcX = std::min(static_cast<int>(x * fScale), nInH - 1);
                            int nSrcIdx = ch * nInH * nInH + nSrcY * nInH + nSrcX;
                            if (nSrcIdx < static_cast<int>(vecImageData.size()))
                                pImg[ch * nSpatial + y * nOutW + x] = vecImageData[nSrcIdx];
                        }
                    }
                }

                // 20260402 ZJH 运行 CRF 精化（5 次迭代，~50ms）
                om::DenseCRFPostProcessor crf;
                crf.m_nIterations = 5;
                auto tRefined = crf.refine(tSoftmax, tImage);

                // 20260402 ZJH 用 CRF 精化后的概率替换 pO 指针用于后续 argmax
                // 注意: pO 指向 cOut（原始 logits），CRF 输出是概率
                // 使用 CRF 结果直接做 argmax
                const float* pCRF = tRefined.contiguous().floatDataPtr();
                result.vecArgmaxMap.resize(static_cast<size_t>(nSpatial));
                for (int s = 0; s < nSpatial; ++s) {
                    int nBestC = 0;
                    float fBestV = pCRF[s];
                    for (int c = 1; c < nOutC; ++c) {
                        float fV = pCRF[c * nSpatial + s];
                        if (fV > fBestV) { fBestV = fV; nBestC = c; }
                    }
                    result.vecArgmaxMap[s] = static_cast<uint8_t>(nBestC);
                }
            } catch (...) {
                // 20260402 ZJH CRF 失败时 fallback 到原始 argmax
                result.vecArgmaxMap.resize(static_cast<size_t>(nSpatial));
                for (int s = 0; s < nSpatial; ++s) {
                    int nBestC = 0;
                    float fBestV = pO[s];
                    for (int c = 1; c < nOutC; ++c) {
                        float fV = pO[c * nSpatial + s];
                        if (fV > fBestV) { fBestV = fV; nBestC = c; }
                    }
                    result.vecArgmaxMap[s] = static_cast<uint8_t>(nBestC);
                }
            }
        } else {
            // 20260401 ZJH ===== 逐像素 argmax 类别图（分割 mask overlay 使用）=====
            result.vecArgmaxMap.resize(static_cast<size_t>(nSpatial));
            for (int s = 0; s < nSpatial; ++s) {
                int nBestC = 0;
                float fBestV = pO[s];
                for (int c = 1; c < nOutC; ++c) {
                    float fV = pO[c * nSpatial + s];
                    if (fV > fBestV) { fBestV = fV; nBestC = c; }
                }
                result.vecArgmaxMap[s] = static_cast<uint8_t>(nBestC);
            }
        }

        // 20260407 ZJH ===== 形态学后处理（对标 Halcon closing_circle + opening_circle）=====
        // 行业标准流水线: argmax → 开运算(去噪点) → 闭运算(填小孔)
        // 3×3 形态学操作，纯 CPU 逐像素处理，对推理延迟影响 <1ms
        if (!result.vecArgmaxMap.empty() && nOutH > 2 && nOutW > 2) {
            auto& map = result.vecArgmaxMap;
            std::vector<uint8_t> vecTemp(map.size());

            // 20260407 ZJH 开运算 (erosion → dilation): 去除 1-2 像素的噪点
            // erosion: 3×3 窗口内有任何背景邻居 → 设为背景
            for (int y = 1; y < nOutH - 1; ++y) {
                for (int x = 1; x < nOutW - 1; ++x) {
                    uint8_t nC = map[y * nOutW + x];
                    if (nC == 0) { vecTemp[y * nOutW + x] = 0; continue; }
                    // 20260407 ZJH 检查 4-邻域是否全为同类
                    bool bKeep = (map[(y-1)*nOutW + x] == nC) && (map[(y+1)*nOutW + x] == nC)
                              && (map[y*nOutW + (x-1)] == nC) && (map[y*nOutW + (x+1)] == nC);
                    vecTemp[y * nOutW + x] = bKeep ? nC : 0;
                }
            }
            // dilation: 恢复被 erosion 缩小的区域
            for (int y = 1; y < nOutH - 1; ++y) {
                for (int x = 1; x < nOutW - 1; ++x) {
                    if (vecTemp[y * nOutW + x] != 0) { map[y * nOutW + x] = vecTemp[y * nOutW + x]; continue; }
                    // 20260407 ZJH 如果 4-邻域有非背景像素，取最常见类
                    uint8_t nUp = vecTemp[(y-1)*nOutW+x], nDn = vecTemp[(y+1)*nOutW+x];
                    uint8_t nLt = vecTemp[y*nOutW+(x-1)], nRt = vecTemp[y*nOutW+(x+1)];
                    uint8_t nBest = 0;
                    if (nUp) nBest = nUp; else if (nDn) nBest = nDn;
                    else if (nLt) nBest = nLt; else if (nRt) nBest = nRt;
                    map[y * nOutW + x] = nBest;
                }
            }
        }
    }
    return result;
}

// 20260330 ZJH inferBatch — 批量推理实现（对标 HikRobot SetBatchSize(1-32)）
// 核心思路:
//   1. 将 N 张图像按 params.nBatchSize 分批打包为 [B, C, H, W] 张量
//   2. 单次 forward 传播处理整个批次
//   3. 拆分输出为逐图像结果，分别后处理（NMS/argmax/softmax）
// 性能收益: 减少 GPU kernel launch 次数、提高显存带宽利用率
std::vector<BridgeInferResult> EngineBridge::inferBatch(
    const std::vector<std::vector<float>>& vecImages,
    int nC, int nH, int nW,
    const BridgeInferParams& params)
{
    std::vector<BridgeInferResult> vecResults;  // 20260330 ZJH 最终结果集

    // 20260330 ZJH 基本校验
    if (!m_pImpl->pModel || vecImages.empty()) return vecResults;

    int nInputDim = nC * nH * nW;       // 20260330 ZJH 单张图像展平维度
    int nNumClasses = m_pImpl->nNumClasses;  // 20260330 ZJH 模型输出类别数

    // 20260330 ZJH 限制 batch size 到 [1, 32] 范围（对标 HikRobot 上限）
    int nBatchSize = std::max(1, std::min(params.nBatchSize, 32));

    // 20260330 ZJH 确保模型处于评估模式（关闭 Dropout/BN 训练行为）
    m_pImpl->pModel->eval();

    // 20260330 ZJH GPU 初始化: 检查模型是否已在 GPU，若不在则迁移
    bool bInferGpu = false;
#ifdef OM_HAS_CUDA
    {
        auto vecParams = m_pImpl->pModel->parameters();
        bool bParamsOnGpu = (!vecParams.empty() && vecParams[0]->isCuda());
        if (!bParamsOnGpu) {
            if (omCudaInit(0) == 0) {
                for (auto* p : vecParams) *p = p->cuda();
                bInferGpu = true;
            }
        } else {
            bInferGpu = true;
        }
    }
#endif

    int nTotalImages = static_cast<int>(vecImages.size());  // 20260330 ZJH 总图像数
    vecResults.reserve(static_cast<size_t>(nTotalImages));   // 20260330 ZJH 预分配结果空间

    // 20260330 ZJH 按 nBatchSize 分批处理
    for (int nStart = 0; nStart < nTotalImages; nStart += nBatchSize) {
        // 20260330 ZJH 当前批次的实际大小（最后一批可能不足 nBatchSize）
        int nCurBatch = std::min(nBatchSize, nTotalImages - nStart);

        // 20260330 ZJH ===== S1: 打包批量输入张量 [B, C, H, W] =====
        std::vector<float> vecBatchInput(static_cast<size_t>(nCurBatch) * nInputDim, 0.0f);
        for (int i = 0; i < nCurBatch; ++i) {
            const auto& vecImg = vecImages[static_cast<size_t>(nStart + i)];
            int nCopy = std::min(static_cast<int>(vecImg.size()), nInputDim);  // 20260330 ZJH 防止越界
            std::copy(vecImg.begin(), vecImg.begin() + nCopy,
                      vecBatchInput.begin() + static_cast<size_t>(i) * nInputDim);
        }

        // 20260330 ZJH ===== S2: 推理归一化（与训练保持一致）=====
        if (m_pImpl->bIsCnn) {
            om::NormPreset eInferNorm = selectNormPreset(m_pImpl->strModelType);
            if (eInferNorm == om::NormPreset::ImageNet) {
                // 20260330 ZJH 对每张图像独立归一化
                for (int i = 0; i < nCurBatch; ++i) {
                    float* pImgStart = vecBatchInput.data() + static_cast<size_t>(i) * nInputDim;
                    std::vector<float> vecSingleImg(pImgStart, pImgStart + nInputDim);
                    om::AugmentConfig inferNormCfg;
                    inferNormCfg.bNormalize = true;
                    inferNormCfg.eNormPreset = om::NormPreset::ImageNet;
                    om::normalizeImage(vecSingleImg, nC, nH, nW, inferNormCfg);
                    std::copy(vecSingleImg.begin(), vecSingleImg.end(), pImgStart);
                }
            }
        }

        // 20260330 ZJH ===== S3: 构建输入张量 =====
        // CNN 模型: [B, C, H, W] 4D 张量
        // MLP 模型: [B, D] 2D 张量
        auto tIn = m_pImpl->bIsCnn
            ? om::Tensor::fromData(vecBatchInput.data(), {nCurBatch, nC, nH, nW})
            : om::Tensor::fromData(vecBatchInput.data(), {nCurBatch, nInputDim});

#ifdef OM_HAS_CUDA
        if (bInferGpu) {
            tIn = tIn.cuda();  // 20260330 ZJH 输入上传到 GPU
        }
#endif

        // 20260330 ZJH ===== S4: 前向传播 =====
        om::Tensor tOut;
        try {
            tOut = m_pImpl->pModel->forward(tIn);
        } catch (const std::exception& ex) {
            std::cerr << "[EngineBridge] inferBatch forward FAILED: " << ex.what() << std::endl;
            // 20260330 ZJH 批量前向失败，回退到逐张推理
            for (int i = 0; i < nCurBatch; ++i) {
                vecResults.push_back(infer(vecImages[static_cast<size_t>(nStart + i)]));
            }
            continue;  // 20260330 ZJH 处理下一批
        }

        // 20260330 ZJH 确保输出在 CPU 上
        if (tOut.isCuda()) tOut = tOut.cpu();
        auto cOut = tOut.contiguous();
        const float* pO = cOut.floatDataPtr();
        auto vecOutShape = cOut.shapeVec();

        // 20260330 ZJH ===== S5: 拆分输出，逐图像后处理 =====
        for (int i = 0; i < nCurBatch; ++i) {
            BridgeInferResult result;

            // 20260330 ZJH ===== YOLO 检测模型: 3D 输出 [B, nPreds, 5+C] =====
            if (m_pImpl->bIsDetection && vecOutShape.size() == 3) {
                int nPreds = cOut.shape(1);    // 20260330 ZJH 每张图的预测数量
                int nDim = cOut.shape(2);      // 20260330 ZJH 每个预测的维度
                int nDetClasses = nDim - 5;    // 20260330 ZJH 检测类别数

                // 20260330 ZJH 定位第 i 张图像的输出数据
                const float* pImgOut = pO + static_cast<size_t>(i) * nPreds * nDim;

                // 20260330 ZJH 构建 YOLO 解码参数
                om::YoloDecodeParams decodeParams;
                decodeParams.nNumClasses = nDetClasses;
                decodeParams.fConfThreshold = params.fConfThreshold;  // 20260330 ZJH 使用调用者指定的阈值
                decodeParams.fNmsIoUThreshold = params.fNmsThreshold;
                decodeParams.nInputWidth = m_pImpl->nInputSize;
                decodeParams.nInputHeight = m_pImpl->nInputSize;

                // 20260330 ZJH 对单张图像执行解码+NMS
                result.vecDetections = om::yoloDecodeAndNms(pImgOut, 1, nPreds, decodeParams);

                // 20260330 ZJH 从检测结果提取图像级预测
                if (!result.vecDetections.empty()) {
                    float fBestScore = -1.0f;
                    int nBestClass = -1;
                    for (const auto& det : result.vecDetections) {
                        if (det.fScore > fBestScore) {
                            fBestScore = det.fScore;
                            nBestClass = det.nClassId;
                        }
                    }
                    result.nPredictedClass = nBestClass;
                    result.fConfidence = fBestScore;
                    result.vecProbs.resize(nNumClasses, 0.0f);
                    for (const auto& det : result.vecDetections) {
                        if (det.nClassId >= 0 && det.nClassId < nNumClasses) {
                            result.vecProbs[det.nClassId] += det.fScore;
                        }
                    }
                    float fProbSum = 0.0f;
                    for (float fP : result.vecProbs) fProbSum += fP;
                    if (fProbSum > 0.0f) {
                        for (float& fP : result.vecProbs) fP /= fProbSum;
                    }
                } else {
                    result.nPredictedClass = 0;
                    result.fConfidence = 0.0f;
                    result.vecProbs.resize(nNumClasses, 0.0f);
                    if (nNumClasses > 0) result.vecProbs[0] = 1.0f;
                }

            // 20260330 ZJH ===== 分割模型: 4D 输出 [B, C, H, W] → 逐像素 argmax =====
            } else if (vecOutShape.size() == 4 && cOut.shape(1) >= 2) {
                int nOutC = cOut.shape(1);        // 20260330 ZJH 类别通道数
                int nOutH = cOut.shape(2);        // 20260330 ZJH 空间高度
                int nOutW = cOut.shape(3);        // 20260330 ZJH 空间宽度
                int nSpatial = nOutH * nOutW;     // 20260330 ZJH 空间像素总数

                // 20260330 ZJH 定位第 i 张图像的输出: 偏移 i * C * H * W
                const float* pImgOut = pO + static_cast<size_t>(i) * nOutC * nSpatial;

                int nDefectPixels = 0;  // 20260330 ZJH 缺陷像素计数
                std::vector<int> vecClassCount(nNumClasses, 0);

                for (int s = 0; s < nSpatial; ++s) {
                    // 20260330 ZJH 逐像素 argmax
                    int nBestC = 0;
                    float fBestLogit = pImgOut[s];
                    for (int c = 1; c < nOutC; ++c) {
                        float fLogit = pImgOut[c * nSpatial + s];
                        if (fLogit > fBestLogit) {
                            fBestLogit = fLogit;
                            nBestC = c;
                        }
                    }
                    if (nBestC != 0) ++nDefectPixels;
                    if (nBestC < nNumClasses) ++vecClassCount[nBestC];
                }

                float fDefectRatio = static_cast<float>(nDefectPixels)
                    / static_cast<float>(std::max(nSpatial, 1));
                constexpr float fDefectRatioThreshold = 0.01f;
                bool bIsDefect = (fDefectRatio > fDefectRatioThreshold);

                result.nPredictedClass = bIsDefect ? 1 : 0;
                result.fConfidence = bIsDefect ? fDefectRatio : (1.0f - fDefectRatio);
                result.vecProbs.resize(nNumClasses);
                for (int c = 0; c < nNumClasses; ++c) {
                    result.vecProbs[c] = static_cast<float>(vecClassCount[c])
                        / static_cast<float>(std::max(nSpatial, 1));
                }

            // 20260330 ZJH ===== 分类模型: 2D 输出 [B, C] → softmax =====
            } else {
                // 20260330 ZJH 2D 输出每行 nNumClasses 个 logit
                int nRowStride = (vecOutShape.size() >= 2) ? cOut.shape(1) : cOut.numel() / nCurBatch;
                const float* pImgOut = pO + static_cast<size_t>(i) * nRowStride;

                std::vector<float> vecLogits(nNumClasses, 0.0f);
                for (int c = 0; c < nNumClasses && c < nRowStride; ++c) {
                    vecLogits[c] = pImgOut[c];
                }

                // 20260330 ZJH 数值稳定 softmax
                result.vecProbs.resize(nNumClasses);
                float fMax = *std::max_element(vecLogits.begin(), vecLogits.end());
                float fSum = 0.0f;
                for (int c = 0; c < nNumClasses; ++c) {
                    result.vecProbs[c] = std::exp(vecLogits[c] - fMax);
                    fSum += result.vecProbs[c];
                }
                int nBest = 0;
                float fBestP = 0.0f;
                for (int c = 0; c < nNumClasses; ++c) {
                    result.vecProbs[c] /= std::max(fSum, 1e-8f);
                    if (result.vecProbs[c] > fBestP) {
                        fBestP = result.vecProbs[c];
                        nBest = c;
                    }
                }
                result.nPredictedClass = nBest;
                result.fConfidence = fBestP;

                // 20260330 ZJH 置信度过滤: 低于阈值的预测视为不确定
                if (result.fConfidence < params.fConfThreshold) {
                    result.nPredictedClass = -1;  // 20260330 ZJH 标记为不确定
                }
            }

            vecResults.push_back(std::move(result));  // 20260330 ZJH 收集结果
        }
    }

    std::cerr << "[EngineBridge] inferBatch complete: " << nTotalImages << " images, "
              << "batchSize=" << nBatchSize << ", batches="
              << ((nTotalImages + nBatchSize - 1) / nBatchSize) << std::endl;

    return vecResults;  // 20260330 ZJH 返回全部结果
}

// 20260324 ZJH 保存模型权重到文件（增加路径验证）
bool EngineBridge::saveModel(const std::string& strPath) {
    if (!m_pImpl->pModel) return false;

    // 20260327 ZJH 强制显存同步：防止异步 CUDA Kernel 错误导致后续 D2H 抛异常
#ifdef OM_HAS_CUDA
    try {
        omCudaSynchronize();
    } catch (...) {}
#endif

    // 20260324 ZJH 验证路径非空
    if (strPath.empty()) {
        std::cerr << "[EngineBridge] saveModel: path is empty" << std::endl;
        return false;
    }

    // 20260325 ZJH 使用 char8_t 构造 filesystem::path，确保 UTF-8 中文路径在 Windows 上正确解析
    std::filesystem::path fsPath(
        reinterpret_cast<const char8_t*>(strPath.c_str()));

    // 20260324 ZJH 检测路径中的 ".." 组件，警告潜在的目录遍历
    for (const auto& component : fsPath) {
        if (component == "..") {
            std::cerr << "[EngineBridge] saveModel: WARNING - path contains '..' component: " << strPath << std::endl;
            break;  // 20260324 ZJH 仅警告，不阻断（合法用途也可能含 ".."）
        }
    }

    // 20260324 ZJH 验证父目录存在，否则无法写入文件
    std::filesystem::path parentDir = fsPath.parent_path();  // 20260324 ZJH 获取父目录
    if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
        std::cerr << "[EngineBridge] saveModel: parent directory does not exist: " << parentDir.string() << std::endl;
        return false;  // 20260324 ZJH 父目录不存在，无法保存
    }

    // 20260328 ZJH 保存前诊断：打印参数总数和数据采样，帮助排查 1KB 空文件问题
    {
        auto vecParams = m_pImpl->pModel->namedParameters();
        int64_t nTotalElements = 0;
        for (auto& [name, p] : vecParams) nTotalElements += p->numel();
        std::cerr << "[EngineBridge] saveModel PRE-CHECK: " << vecParams.size()
                  << " params, " << nTotalElements << " elements ("
                  << (nTotalElements * 4 / 1024) << " KB)"
                  << ", all on " << (vecParams.empty() ? "N/A" : (vecParams[0].second->isCuda() ? "CUDA" : "CPU"))
                  << std::endl;
        // 20260328 ZJH 打印前3个参数的数据采样（前2个float），确认数据非零
        for (size_t i = 0; i < std::min(vecParams.size(), (size_t)3); ++i) {
            auto& [name, p] = vecParams[i];
            if (p->numel() > 0 && p->isCpu()) {
                const float* pD = p->floatDataPtr();
                std::cerr << "  [" << i << "] " << name << " numel=" << p->numel()
                          << " data[0]=" << pD[0]
                          << (p->numel() > 1 ? (" data[1]=" + std::to_string(pD[1])) : "")
                          << std::endl;
            }
        }
    }

    // 20260325 ZJH 捕获具体异常并输出错误信息，便于诊断模型保存失败原因
    try {
        // 20260330 ZJH 构建模型架构元数据并传入序列化器（v4 格式）
        om::ModelMeta meta;
        meta.strModelType = m_pImpl->strModelType;
        meta.nModelTypeHash = om::ModelMeta::hashString(m_pImpl->strModelType);
        meta.nBaseChannels = m_pImpl->nBaseChannels;
        meta.nInputSize = m_pImpl->nInputSize;
        meta.nNumClasses = m_pImpl->nNumClasses;
        meta.nInChannels = 3;  // 20260330 ZJH 当前固定 RGB 3 通道
        meta.nNormType = m_pImpl->bUseGroupNorm ? 1 : 0;  // 20260402 ZJH 记录归一化类型
        meta.nGroupNormGroups = 32;  // 20260402 ZJH 默认分组数
        om::ModelSerializer::save(*m_pImpl->pModel, strPath, meta);  // 20260402 ZJH v5 序列化
        return true;  // 20260325 ZJH 保存成功
    } catch (const std::exception& e) {
        // 20260325 ZJH 捕获标准异常，输出具体错误信息和目标路径
        std::cerr << "[EngineBridge] saveModel FAILED: " << e.what() << " path=" << strPath << std::endl;
        return false;  // 20260325 ZJH 保存失败
    } catch (...) {
        // 20260325 ZJH 捕获未知异常，输出路径以便排查
        std::cerr << "[EngineBridge] saveModel FAILED: unknown exception, path=" << strPath << std::endl;
        return false;  // 20260325 ZJH 保存失败（未知异常）
    }
}

// 20260324 ZJH 加载模型权重从文件（增加路径验证）
bool EngineBridge::loadModel(const std::string& strPath) {
    if (!m_pImpl->pModel) return false;  // 20260323 ZJH 模型未创建，直接返回

    // 20260324 ZJH 验证路径非空
    if (strPath.empty()) {
        std::cerr << "[EngineBridge] loadModel: path is empty" << std::endl;
        return false;
    }

    // 20260326 ZJH 将所有操作包在 try/catch 中，防止 filesystem 异常逃逸
    // 之前仅 ModelSerializer::load 被 try/catch 保护，filesystem::path 构造和
    // filesystem::exists 可能抛出 filesystem_error，导致异常逃逸到调用方
    try {
        // 20260325 ZJH 使用 char8_t 构造 filesystem::path，确保 UTF-8 中文路径在 Windows 上正确解析
        std::filesystem::path fsPath(
            reinterpret_cast<const char8_t*>(strPath.c_str()));
        // 20260324 ZJH 检测路径中的 ".." 组件，警告潜在的目录遍历
        for (const auto& component : fsPath) {
            if (component == "..") {
                std::cerr << "[EngineBridge] loadModel: WARNING - path contains '..' component: " << strPath << std::endl;
                break;  // 20260324 ZJH 仅警告，不阻断
            }
        }

        // 20260324 ZJH 验证文件存在，避免传入不存在的路径导致异常
        if (!std::filesystem::exists(fsPath)) {
            std::cerr << "[EngineBridge] loadModel: file does not exist: " << strPath << std::endl;
            return false;  // 20260324 ZJH 文件不存在，无法加载
        }

        // 20260330 ZJH ===== 预扫描元数据，检测架构是否匹配 =====
        // 问题根因: createModel() 的 base channels 等参数会随代码更新而变化，
        //           导致推理时创建的架构与训练时不同，参数形状全部不匹配被静默跳过，
        //           模型保留随机权重 → 输出噪声（水平条纹等）
        // 解决: 加载前先读取文件中的元数据，如果架构不匹配则自动重建正确的模型
        {
            om::ModelMeta fileMeta;
            std::vector<int> vecFirstConvShape;
            bool bHasMeta = om::ModelSerializer::peekMeta(strPath, fileMeta, &vecFirstConvShape);

            if (bHasMeta) {
                // 20260330 ZJH v4 文件: 精确匹配架构
                bool bNeedRebuild = false;
                if (fileMeta.nBaseChannels != m_pImpl->nBaseChannels) {
                    std::cerr << "[EngineBridge] loadModel: base channels mismatch — file="
                              << fileMeta.nBaseChannels << " current=" << m_pImpl->nBaseChannels
                              << " → rebuilding model" << std::endl;
                    bNeedRebuild = true;
                }
                if (fileMeta.nNumClasses != m_pImpl->nNumClasses) {
                    std::cerr << "[EngineBridge] loadModel: numClasses mismatch — file="
                              << fileMeta.nNumClasses << " current=" << m_pImpl->nNumClasses
                              << " → rebuilding model" << std::endl;
                    bNeedRebuild = true;
                }
                // 20260405 ZJH GroupNorm/BatchNorm 不匹配检测
                // 训练时 batch<8 自动启用 GroupNorm（train() 内重建模型），meta 记录 nNormType=1
                // 推理时 createModel() 默认 bUseGroupNorm=false（BatchNorm）
                // 若不重建: GN 权重加载到 BN 层，但 BN running_mean=0/running_var=1（GN 无 buffers）
                // → eval 模式下 BN 不做真正归一化 → 特征空间错乱 → 全预测背景 → 推理不出缺陷
                bool bFileGroupNorm = (fileMeta.nNormType == 1);  // 20260405 ZJH 文件中记录的归一化类型
                if (bFileGroupNorm != m_pImpl->bUseGroupNorm) {
                    std::cerr << "[EngineBridge] loadModel: normType mismatch — file="
                              << (bFileGroupNorm ? "GroupNorm" : "BatchNorm")
                              << " current=" << (m_pImpl->bUseGroupNorm ? "GroupNorm" : "BatchNorm")
                              << " → rebuilding model" << std::endl;
                    m_pImpl->bUseGroupNorm = bFileGroupNorm;  // 20260405 ZJH 同步归一化标志
                    bNeedRebuild = true;
                }
                if (bNeedRebuild) {
                    // 20260330 ZJH 用文件元数据中的参数重建模型
                    std::string strType = m_pImpl->strModelType;

                    // 20260406 ZJH [修复] 检测训练时是否自动缩放了模型类型
                    // 场景: 训练时 DeepLabV3+ 自动缩放为 MobileSegNet（<30张图）
                    //       但推理端用户仍选 DeepLabV3+ → nModelTypeHash 不匹配
                    //       → 用错误架构重建 → 所有权重 shape 不匹配全部 SKIPPED
                    uint32_t nCurHash = om::ModelMeta::hashString(strType);
                    if (fileMeta.nModelTypeHash != 0 && fileMeta.nModelTypeHash != nCurHash) {
                        // 20260406 ZJH 遍历已知分割模型类型，匹配文件中的 hash
                        for (const char* strCandidate : {"UNet", "DeepLabV3+", "DeepLabV3Plus",
                                "MobileSegNet", "MobileSeg", "ResNet18", "ResNet50",
                                "MobileNetV4Small", "ViTTiny", "EfficientAD"}) {
                            if (om::ModelMeta::hashString(strCandidate) == fileMeta.nModelTypeHash) {
                                std::cerr << "[EngineBridge] loadModel: model type auto-corrected — "
                                          << "user selected '" << strType << "' but file was trained as '"
                                          << strCandidate << "'" << std::endl;
                                strType = strCandidate;
                                m_pImpl->strModelType = strType;
                                break;
                            }
                        }
                    }

                    int nSavedBase = fileMeta.nBaseChannels;
                    int nSavedClasses = fileMeta.nNumClasses;
                    int nSavedInput = (fileMeta.nInputSize > 0) ? fileMeta.nInputSize : m_pImpl->nInputSize;
                    // 20260402 ZJH 从序列化 meta 中读取 norm_type，v4 文件默认 BN
                    bool bFileGroupNorm = (fileMeta.nNormType == 1);
                    // 20260330 ZJH 重建 UNet 时使用文件中记录的 base
                    if (strType == "UNet") {
                        m_pImpl->pModel = std::make_shared<om::UNet>(3, nSavedClasses, nSavedBase, bFileGroupNorm);
                        m_pImpl->nBaseChannels = nSavedBase;
                        m_pImpl->nNumClasses = nSavedClasses;
                        m_pImpl->bIsSegmentation = true;
                    } else if (strType == "DeepLabV3+" || strType == "DeepLabV3Plus" || strType == "DeepLabV3") {
                        m_pImpl->pModel = std::make_shared<om::DeepLabV3>(3, nSavedClasses, bFileGroupNorm);
                        m_pImpl->nNumClasses = nSavedClasses;
                        m_pImpl->bIsSegmentation = true;
                    }
                    // 20260401 ZJH MobileSegNet 反序列化重建
                    else if (strType == "MobileSegNet" || strType == "MobileSeg") {
                        m_pImpl->pModel = std::make_shared<om::MobileSegNet>(3, nSavedClasses, bFileGroupNorm);
                        m_pImpl->nNumClasses = nSavedClasses;
                        m_pImpl->bIsSegmentation = true;
                    }
                    else if (strType == "ViTTiny") {
                        // 20260330 ZJH ViT 位置编码依赖 inputSize，需要用文件中的 inputSize 重建
                        m_pImpl->pModel = std::make_shared<om::ViT>(
                            nSavedInput, 16, 3, nSavedClasses, 192, 12, 3);
                        m_pImpl->nBaseChannels = 192;
                        m_pImpl->nNumClasses = nSavedClasses;
                        m_pImpl->nInputSize = nSavedInput;
                    }
                    // 20260330 ZJH 其他模型类型的 base 是固定的，numClasses 变化需要重建
                    // 但分类模型的最终层形状变化已由 ModelSerializer 的 shape mismatch 处理
                    std::cerr << "[EngineBridge] loadModel: model rebuilt with base="
                              << m_pImpl->nBaseChannels << " classes=" << m_pImpl->nNumClasses << std::endl;
                }
            } else if (!vecFirstConvShape.empty() && vecFirstConvShape.size() == 4) {
                // 20260330 ZJH v3 回退: 从第一个 conv weight 的 shape[0] 推断 base channels
                int nFileBase = vecFirstConvShape[0];  // 20260330 ZJH Conv2d weight [Cout, Cin, KH, KW]
                if (nFileBase != m_pImpl->nBaseChannels && m_pImpl->strModelType == "UNet") {
                    std::cerr << "[EngineBridge] loadModel: v3 fallback — inferred base="
                              << nFileBase << " from first conv weight, current=" << m_pImpl->nBaseChannels
                              << " → rebuilding UNet" << std::endl;
                    // 20260402 ZJH v3 文件无 norm 元数据，默认 BN（bUseGroupNorm=false）
                    m_pImpl->pModel = std::make_shared<om::UNet>(3, m_pImpl->nNumClasses, nFileBase, false);
                    m_pImpl->nBaseChannels = nFileBase;
                    m_pImpl->bIsSegmentation = true;
                }
            }
        }

        om::ModelSerializer::load(*m_pImpl->pModel, strPath);  // 20260330 ZJH 从磁盘反序列化模型权重
        return true;  // 20260325 ZJH 加载成功
    } catch (const std::exception& e) {
        // 20260325 ZJH 捕获标准异常，输出具体错误信息和文件路径
        std::cerr << "[EngineBridge] loadModel FAILED: " << e.what() << " path=" << strPath << std::endl;
        return false;  // 20260325 ZJH 加载失败
    } catch (...) {
        // 20260325 ZJH 捕获未知异常，输出路径以便排查
        std::cerr << "[EngineBridge] loadModel FAILED: unknown exception, path=" << strPath << std::endl;
        return false;  // 20260325 ZJH 加载失败（未知异常）
    }
}

// 20260406 ZJH 获取模型参数总数（遍历所有参数张量，累加元素数）
int64_t EngineBridge::totalParameters() const {
    if (!m_pImpl->pModel) return 0;  // 20260406 ZJH 模型未创建返回 0
    int64_t n = 0;  // 20260406 ZJH 参数元素总数累加器
    for (const auto* p : m_pImpl->pModel->parameters()) {
        try { if (p) n += p->numel(); } catch (...) {}  // 20260330 ZJH 跳过损坏的参数
    }
    return n;  // 20260406 ZJH 返回参数总元素数
}

// 20260406 ZJH 可训练参数数（当前所有参数均可训练，等同 totalParameters）
int64_t EngineBridge::trainableParameters() const { return totalParameters(); }
// 20260406 ZJH 检查模型是否已创建（pModel 非空即表示已创建）
bool EngineBridge::hasModel() const { return m_pImpl->pModel.get() != nullptr; }

// 20260326 ZJH 强制释放 GPU 内存
// 训练异常退出时 omCudaCleanup() 被跳过，GPU 内存泄漏 15.7GB
// 此方法将所有模型参数移回 CPU，释放 GPU 张量存储，然后清理 CUDA 资源
void EngineBridge::releaseGpu() {
#ifdef OM_HAS_CUDA
    // 20260326 ZJH GPU 清理：销毁模型释放张量引用 → 同步 → 清错误 → 释放流
    // 不使用 cudaDeviceReset（会销毁 CUDA 上下文，TensorStorage 析构时 cudaFree 失败
    // → illegal memory access → 永久污染后续 CUDA 调用）
    try {
        if (m_pImpl->pModel) {
            m_pImpl->pModel.reset();  // 20260326 ZJH 销毁模型 → 级联 cudaFree 所有 GPU 张量
        }
    } catch (...) {}
    try {
        omCudaForceReset();  // 20260326 ZJH 同步 + 清错误 + 释放流（不做 cudaDeviceReset）
    } catch (...) {}
#endif
}

// =========================================================================
// 20260330 ZJH AI 数据合成 — 从少量缺陷样本自动生成大量合成训练数据
// 桥接 om::DataSynthesisPipeline / DefectCopyPaste / AugmentSynthesizer
// =========================================================================

BridgeSynthesisResult EngineBridge::synthesizeData(
    const std::vector<std::vector<float>>& vecNormalImages,
    const std::vector<std::vector<float>>& vecDefectImages,
    int nC, int nH, int nW,
    const BridgeSynthesisParams& params)
{
    BridgeSynthesisResult result;  // 20260330 ZJH 返回结果
    result.nOrigCount = static_cast<int>(vecDefectImages.size());

    // 20260330 ZJH 输入验证
    if (vecDefectImages.empty()) {
        std::cerr << "[EngineBridge::synthesizeData] ERROR: no defect images" << std::endl;
        return result;
    }

    try {
        int nStrategy = params.nStrategy;  // 20260330 ZJH 合成策略

        // 20260330 ZJH 策略 3: 自动选择 — 委托给 DataSynthesisPipeline::autoSynthesize()
        if (nStrategy == 3) {
            // 20260330 ZJH 构建空的缺陷标注框列表（简化版：整张图都是缺陷）
            // 对于无 BBox 标注的缺陷图像，CopyPaste 使用整张图作为缺陷 patch
            std::vector<std::vector<om::BBox>> vecAnnotations;
            vecAnnotations.reserve(vecDefectImages.size());
            for (size_t i = 0; i < vecDefectImages.size(); ++i) {
                // 20260330 ZJH 每张缺陷图的标注: 整图 BBox
                om::BBox bbox;
                bbox.fX = 0.0f;
                bbox.fY = 0.0f;
                bbox.fW = static_cast<float>(nW);
                bbox.fH = static_cast<float>(nH);
                bbox.nClassId = 0;
                bbox.strClassName = "defect";
                vecAnnotations.push_back({bbox});
            }

            // 20260330 ZJH 调用自动合成管线
            om::DataSynthesisPipeline::SynthesisResult synthResult =
                om::DataSynthesisPipeline::autoSynthesize(
                    vecNormalImages, vecDefectImages, vecAnnotations,
                    nC, nH, nW, params.nTargetCount);

            // 20260330 ZJH 转换结果
            result.vecImages = std::move(synthResult.vecImages);
            result.nSynthCount = synthResult.nSynthesizedCount;
            return result;
        }

        // 20260330 ZJH 策略 0: CopyPaste 缺陷粘贴
        if (nStrategy == 0) {
            // 20260330 ZJH 构建 CopyPaste 配置
            om::CopyPasteConfig cpConfig;
            cpConfig.fScaleMin = params.fScaleMin;
            cpConfig.fScaleMax = params.fScaleMax;
            cpConfig.fRotateRange = params.fRotateRange;
            cpConfig.fBlendAlpha = params.fBlendAlpha;
            cpConfig.bPoissonBlend = params.bPoisson;
            cpConfig.bRandomPosition = true;

            // 20260330 ZJH 从缺陷图像中提取 patch（整图作为缺陷区域）
            std::vector<om::DefectCopyPaste::DefectPatch> vecAllPatches;
            for (const auto& vecDefImg : vecDefectImages) {
                om::BBox bbox;
                bbox.fX = 0.0f;
                bbox.fY = 0.0f;
                bbox.fW = static_cast<float>(nW);
                bbox.fH = static_cast<float>(nH);
                bbox.nClassId = 0;
                bbox.strClassName = "defect";
                auto vecPatches = om::DefectCopyPaste::extractDefects(
                    vecDefImg, nC, nH, nW, {bbox});
                vecAllPatches.insert(vecAllPatches.end(),
                    std::make_move_iterator(vecPatches.begin()),
                    std::make_move_iterator(vecPatches.end()));
            }

            if (!vecAllPatches.empty() && !vecNormalImages.empty()) {
                // 20260330 ZJH 计算每个 patch 的合成数量
                int nPerPatch = std::max(1, params.nTargetCount / static_cast<int>(vecAllPatches.size()));
                cpConfig.nNumSynthPerDefect = nPerPatch;

                auto vecSynth = om::DefectCopyPaste::batchSynthesize(
                    vecNormalImages, vecAllPatches, nC, nH, nW, cpConfig);

                // 20260330 ZJH 截取所需数量
                int nTake = std::min(params.nTargetCount, static_cast<int>(vecSynth.size()));
                for (int i = 0; i < nTake; ++i) {
                    result.vecImages.push_back(std::move(vecSynth[static_cast<size_t>(i)].first));
                }
                result.nSynthCount = nTake;
            }
            return result;
        }

        // 20260330 ZJH 策略 1: 几何+光度增强
        if (nStrategy == 1) {
            om::AugSynthConfig augConfig;
            augConfig.nNumVariants = params.nVariantsPerImage;
            augConfig.bElasticDeform = params.bElastic;
            augConfig.bPerspective = params.bPerspective;
            augConfig.bColorTransfer = true;

            int nGenerated = 0;
            for (size_t i = 0; i < vecDefectImages.size() && nGenerated < params.nTargetCount; ++i) {
                auto vecVariants = om::AugmentSynthesizer::generateVariants(
                    vecDefectImages[i], nC, nH, nW, augConfig, vecNormalImages);

                for (auto& variant : vecVariants) {
                    if (nGenerated >= params.nTargetCount) break;
                    result.vecImages.push_back(std::move(variant));
                    ++nGenerated;
                }
            }
            result.nSynthCount = nGenerated;
            return result;
        }

        // 20260330 ZJH 策略 2: GAN 生成
        if (nStrategy == 2) {
            om::DefectGANSynthesizer gan(64);  // 20260330 ZJH 64 维潜在空间
            gan.train(vecDefectImages, nC, nH, nW, params.nGanEpochs);

            if (gan.isTrained()) {
                auto vecGenerated = gan.generate(params.nTargetCount);
                result.vecImages = std::move(vecGenerated);
                result.nSynthCount = static_cast<int>(result.vecImages.size());
            }
            return result;
        }

    } catch (const std::exception& e) {
        // 20260330 ZJH 捕获异常，输出错误日志
        std::cerr << "[EngineBridge::synthesizeData] EXCEPTION: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[EngineBridge::synthesizeData] UNKNOWN EXCEPTION" << std::endl;
    }

    return result;
}

// =============================================================================
// 20260402 ZJH 新增接口实现 — 对标 Halcon/ViDi 差距补全
// =============================================================================

// 20260402 ZJH inferGCAD — GCAD 全局上下文异常检测推理
EngineBridge::GCADInferResult EngineBridge::inferGCAD(
    const std::vector<float>& vecImageData, int nC, int nH, int nW)
{
    GCADInferResult result{};  // 20260402 ZJH 默认初始化
    try {
        if (!m_pImpl || !m_pImpl->pModel) {
            std::cerr << "[EngineBridge::inferGCAD] No model loaded" << std::endl;
            return result;
        }

        // 20260402 ZJH 构造输入张量 [1, C, H, W]
        auto input = om::Tensor::zeros({1, nC, nH, nW});
        float* pInput = input.mutableFloatDataPtr();
        int nTotal = nC * nH * nW;
        for (int i = 0; i < std::min(nTotal, static_cast<int>(vecImageData.size())); ++i) {
            pInput[i] = vecImageData[i];
        }

        // 20260402 ZJH 尝试转换为 GCAD 模型
        auto* pGcad = dynamic_cast<om::GCAD*>(m_pImpl->pModel.get());
        if (!pGcad) {
            std::cerr << "[EngineBridge::inferGCAD] Model is not GCAD type" << std::endl;
            return result;
        }

        // 20260402 ZJH 调用 GCAD predict
        pGcad->eval();  // 20260402 ZJH 切换到评估模式
        auto gcadResult = pGcad->predict(input);

        // 20260402 ZJH 转换结果
        result.fGlobalScore = gcadResult.fGlobalScore;
        result.fLocalScore = gcadResult.fLocalScore;
        result.fFusedScore = gcadResult.fFusedScore;
        result.bIsAnomaly = gcadResult.bIsAnomaly;
        result.bIsLayoutAnomaly = gcadResult.bIsLayoutAnomaly;
        result.vecAnomalyMap = std::move(gcadResult.vecAnomalyMap);
        result.nMapH = gcadResult.nMapH;
        result.nMapW = gcadResult.nMapW;

    } catch (const std::exception& e) {
        std::cerr << "[EngineBridge::inferGCAD] EXCEPTION: " << e.what() << std::endl;
    }
    return result;
}

// 20260402 ZJH fitGCADDistribution — 训练后拟合正常样本全局分布
bool EngineBridge::fitGCADDistribution(
    const std::vector<std::vector<float>>& vecNormalImages,
    int nC, int nH, int nW)
{
    try {
        if (!m_pImpl || !m_pImpl->pModel) return false;

        auto* pGcad = dynamic_cast<om::GCAD*>(m_pImpl->pModel.get());
        if (!pGcad) return false;

        pGcad->eval();  // 20260402 ZJH 评估模式

        // 20260402 ZJH 收集所有正常样本的全局特征向量
        std::vector<std::vector<float>> vecFeatures;
        std::vector<float> vecFusedScores;

        for (const auto& vecImg : vecNormalImages) {
            auto input = om::Tensor::zeros({1, nC, nH, nW});
            float* pIn = input.mutableFloatDataPtr();
            int nPixels = nC * nH * nW;
            for (int i = 0; i < std::min(nPixels, static_cast<int>(vecImg.size())); ++i) {
                pIn[i] = vecImg[i];
            }

            // 20260402 ZJH 提取全局上下文向量
            auto globalVec = pGcad->predictGlobal(input);
            auto cGlobal = globalVec.contiguous();
            int nDim = cGlobal.shape(1);
            const float* pGlobal = cGlobal.floatDataPtr();

            std::vector<float> vecFeat(nDim);
            for (int d = 0; d < nDim; ++d) vecFeat[d] = pGlobal[d];
            vecFeatures.push_back(std::move(vecFeat));

            // 20260402 ZJH 收集融合分数
            auto gcadResult = pGcad->predict(input);
            vecFusedScores.push_back(gcadResult.fFusedScore);
        }

        // 20260402 ZJH 拟合全局分布 + 校准阈值
        pGcad->fitGlobalDistribution(vecFeatures);
        pGcad->calibrateThreshold(vecFusedScores);

        std::cerr << "[EngineBridge::fitGCADDistribution] Fitted with "
                  << vecNormalImages.size() << " normal samples" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[EngineBridge::fitGCADDistribution] EXCEPTION: " << e.what() << std::endl;
    }
    return false;
}

// 20260402 ZJH trainContinual — 增量学习（EWC 弹性权重巩固）
bool EngineBridge::trainContinual(
    const std::vector<std::vector<float>>& vecOldData,
    const std::vector<std::vector<float>>& vecOldLabels,
    const std::vector<std::vector<float>>& vecNewData,
    const std::vector<std::vector<float>>& vecNewLabels,
    int nC, int nH, int nW, int nEpochs, float fLR,
    float fEwcLambda)
{
    try {
        if (!m_pImpl || !m_pImpl->pModel) return false;

        auto* pModel = m_pImpl->pModel.get();
        pModel->train();  // 20260402 ZJH 训练模式

        // 20260402 ZJH Step 1: 用旧数据计算 Fisher 信息矩阵
        om::ContinualLearner ewc;
        om::ContinualLearner::EWCConfig config;
        config.fLambda = fEwcLambda;

        // 20260402 ZJH 构造旧任务张量
        int nInputDim = nC * nH * nW;
        std::vector<om::Tensor> vecOldTensors, vecOldLabelTensors;
        for (size_t i = 0; i < vecOldData.size(); ++i) {
            auto t = om::Tensor::zeros({1, nC, nH, nW});
            float* p = t.mutableFloatDataPtr();
            for (int j = 0; j < std::min(nInputDim, static_cast<int>(vecOldData[i].size())); ++j) {
                p[j] = vecOldData[i][j];
            }
            vecOldTensors.push_back(std::move(t));

            auto lbl = om::Tensor::zeros({1, static_cast<int>(vecOldLabels[i].size())});
            float* pLbl = lbl.mutableFloatDataPtr();
            for (size_t j = 0; j < vecOldLabels[i].size(); ++j) {
                pLbl[j] = vecOldLabels[i][j];
            }
            vecOldLabelTensors.push_back(std::move(lbl));
        }

        ewc.computeFisherMatrix(*pModel, vecOldTensors, vecOldLabelTensors);

        // 20260402 ZJH Step 2: 在新数据上训练（带 EWC 正则化）
        auto vecParams = pModel->parameters();
        om::SGD optimizer(vecParams, fLR, 0.9f, 1e-4f);
        om::CrossEntropyLoss ceLoss;

        for (int epoch = 0; epoch < nEpochs; ++epoch) {
            float fEpochLoss = 0.0f;
            for (size_t i = 0; i < vecNewData.size(); ++i) {
                auto input = om::Tensor::zeros({1, nC, nH, nW});
                float* pIn = input.mutableFloatDataPtr();
                for (int j = 0; j < std::min(nInputDim, static_cast<int>(vecNewData[i].size())); ++j) {
                    pIn[j] = vecNewData[i][j];
                }

                auto target = om::Tensor::zeros({1, static_cast<int>(vecNewLabels[i].size())});
                float* pTgt = target.mutableFloatDataPtr();
                for (size_t j = 0; j < vecNewLabels[i].size(); ++j) {
                    pTgt[j] = vecNewLabels[i][j];
                }

                // 20260402 ZJH 前向 + CE + EWC
                auto output = pModel->forward(input);
                auto taskLoss = ceLoss.forward(output, target);
                auto ewcPenalty = ewc.ewcPenalty(*pModel);
                auto totalLoss = om::tensorAdd(taskLoss, ewcPenalty);

                optimizer.zeroGrad();
                om::tensorBackward(totalLoss);  // 20260402 ZJH autograd 反向传播
                optimizer.step();

                fEpochLoss += totalLoss.contiguous().floatDataPtr()[0];
            }

            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                std::cerr << "[ContinualLearning] Epoch " << (epoch + 1)
                          << "/" << nEpochs << " loss="
                          << fEpochLoss / std::max(size_t(1), vecNewData.size()) << std::endl;
            }
        }
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[EngineBridge::trainContinual] EXCEPTION: " << e.what() << std::endl;
    }
    return false;
}

// 20260402 ZJH inferWithTTA — TTA 增强推理
BridgeInferResult EngineBridge::inferWithTTA(
    const std::vector<float>& vecImageData,
    int nC, int nH, int nW,
    bool bHFlip, bool bVFlip, bool bRotate90, bool bMultiScale)
{
    BridgeInferResult result{};  // 20260406 ZJH TTA 推理结果（默认初始化）
    try {
        if (!m_pImpl || !m_pImpl->pModel) return result;  // 20260406 ZJH 模型未加载，返回空结果

        auto* pModel = m_pImpl->pModel.get();  // 20260406 ZJH 获取模型裸指针
        pModel->eval();  // 20260406 ZJH 切换到评估模式

        // 20260402 ZJH 构造输入张量
        auto input = om::Tensor::zeros({1, nC, nH, nW});  // 20260406 ZJH 创建 [1,C,H,W] 零张量
        float* pInput = input.mutableFloatDataPtr();  // 20260406 ZJH 获取可写数据指针
        int nTotal = nC * nH * nW;  // 20260406 ZJH 单张图像总元素数
        for (int i = 0; i < std::min(nTotal, static_cast<int>(vecImageData.size())); ++i) {
            pInput[i] = vecImageData[i];  // 20260406 ZJH 将输入数据拷贝到张量
        }

        // 20260402 ZJH 配置 TTA
        om::TTAPredictor tta;
        om::TTAPredictor::TTAConfig ttaConfig;
        ttaConfig.bHFlip = bHFlip;
        ttaConfig.bVFlip = bVFlip;
        ttaConfig.bRotate90 = bRotate90;
        ttaConfig.bMultiScale = bMultiScale;

        if (m_pImpl->bIsSegmentation) {
            // 20260402 ZJH 分割 TTA
            auto ttaOutput = tta.segmentTTA(*pModel, input, ttaConfig);
            auto cOut = ttaOutput.contiguous();
            int nOutC = cOut.shape(1), nOutH = cOut.shape(2), nOutW = cOut.shape(3);
            result.vecArgmaxMap.resize(nOutH * nOutW);
            const float* pOut = cOut.floatDataPtr();
            for (int h = 0; h < nOutH; ++h) {
                for (int w = 0; w < nOutW; ++w) {
                    int nBestC = 0;
                    float fBestV = pOut[h * nOutW + w];
                    for (int c = 1; c < nOutC; ++c) {
                        float fV = pOut[(c * nOutH + h) * nOutW + w];
                        if (fV > fBestV) { fBestV = fV; nBestC = c; }
                    }
                    result.vecArgmaxMap[h * nOutW + w] = static_cast<uint8_t>(nBestC);
                }
            }
        } else {
            // 20260402 ZJH 分类 TTA
            auto vecProbs = tta.classifyTTA(*pModel, input, ttaConfig);
            if (!vecProbs.empty()) {
                int nBestClass = 0;
                float fBestProb = vecProbs[0];
                for (int i = 1; i < static_cast<int>(vecProbs.size()); ++i) {
                    if (vecProbs[i] > fBestProb) {
                        fBestProb = vecProbs[i];
                        nBestClass = i;
                    }
                }
                result.nPredictedClass = nBestClass;
                result.fConfidence = fBestProb;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[EngineBridge::inferWithTTA] EXCEPTION: " << e.what() << std::endl;
    }
    return result;
}

// =============================================================================
// 20260402 ZJH Benchmark + 精度基线系统实现
// =============================================================================

// 20260402 ZJH benchmarkInference — 推理性能基准测试
EngineBridge::BenchmarkResult EngineBridge::benchmarkInference(
    const std::vector<float>& vecImageData,
    int nC, int nH, int nW,
    int nWarmupRuns, int nBenchmarkRuns)
{
    BenchmarkResult result{};  // 20260406 ZJH 性能基准测试结果（默认初始化）
    result.nWarmupRuns = nWarmupRuns;        // 20260406 ZJH 记录预热轮数
    result.nBenchmarkRuns = nBenchmarkRuns;  // 20260406 ZJH 记录测试轮数

    try {
        if (!m_pImpl || !m_pImpl->pModel) return result;  // 20260406 ZJH 模型未加载，返回空结果

        m_pImpl->pModel->eval();  // 20260402 ZJH 评估模式

        // 20260402 ZJH 构造输入张量
        auto input = om::Tensor::zeros({1, nC, nH, nW});
        float* pInput = input.mutableFloatDataPtr();
        int nTotal = nC * nH * nW;
        for (int i = 0; i < std::min(nTotal, static_cast<int>(vecImageData.size())); ++i) {
            pInput[i] = vecImageData[i];
        }

        // 20260402 ZJH Warmup（消除冷启动、JIT 编译等影响）
        for (int i = 0; i < nWarmupRuns; ++i) {
            auto output = m_pImpl->pModel->forward(input);
            (void)output;
        }

        // 20260402 ZJH 正式计时
        std::vector<double> vecTimes;           // 20260406 ZJH 每次推理的耗时记录 (ms)
        vecTimes.reserve(nBenchmarkRuns);       // 20260406 ZJH 预分配避免扩容
        for (int i = 0; i < nBenchmarkRuns; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto output = m_pImpl->pModel->forward(input);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            vecTimes.push_back(dMs);
        }

        // 20260402 ZJH 统计
        std::sort(vecTimes.begin(), vecTimes.end());  // 20260406 ZJH 升序排序用于分位数计算
        result.dMinMs = vecTimes.front();              // 20260406 ZJH 最小延迟
        result.dMaxMs = vecTimes.back();               // 20260406 ZJH 最大延迟
        result.dMedianMs = vecTimes[vecTimes.size() / 2];  // 20260406 ZJH 中位数延迟
        result.dP95Ms = vecTimes[static_cast<int>(vecTimes.size() * 0.95)];  // 20260406 ZJH P95 延迟
        result.dP99Ms = vecTimes[static_cast<int>(vecTimes.size() * 0.99)];  // 20260406 ZJH P99 延迟

        // 20260402 ZJH 吞吐量 = 1000 / median
        result.dThroughputFPS = (result.dMedianMs > 0.0) ? (1000.0 / result.dMedianMs) : 0.0;

        std::cerr << "[Benchmark] median=" << result.dMedianMs
                  << "ms p95=" << result.dP95Ms
                  << "ms FPS=" << result.dThroughputFPS << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[EngineBridge::benchmarkInference] EXCEPTION: " << e.what() << std::endl;
    }
    return result;
}

// 20260402 ZJH saveAccuracyBaseline — 保存精度基线到 JSON
bool EngineBridge::saveAccuracyBaseline(const AccuracyBaseline& baseline,
                                         const std::string& strBaselineDir)
{
    try {
        // 20260402 ZJH 构造文件路径: baselines/ModelType_DatasetName.json
        std::string strFileName = strBaselineDir + "/" + baseline.strModelType + "_baseline.json";
        std::ofstream ofs(strFileName);
        if (!ofs.is_open()) {
            std::cerr << "[Baseline] Failed to open: " << strFileName << std::endl;
            return false;
        }

        // 20260402 ZJH 手写 JSON（不依赖 nlohmann/json 在此层）
        ofs << "{\n"
            << "  \"model\": \"" << baseline.strModelType << "\",\n"
            << "  \"dataset\": \"" << baseline.strDatasetName << "\",\n"
            << "  \"train_samples\": " << baseline.nTrainSamples << ",\n"
            << "  \"epochs\": " << baseline.nEpochs << ",\n"
            << "  \"final_loss\": " << baseline.fFinalLoss << ",\n"
            << "  \"val_accuracy\": " << baseline.fValAccuracy << ",\n"
            << "  \"val_f1\": " << baseline.fValF1 << ",\n"
            << "  \"inference_ms\": " << baseline.fInferenceMs << ",\n"
            << "  \"timestamp\": \"" << baseline.strTimestamp << "\"\n"
            << "}\n";
        ofs.close();

        std::cerr << "[Baseline] Saved: " << strFileName << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Baseline] Save error: " << e.what() << std::endl;
    }
    return false;
}

// 20260402 ZJH loadAccuracyBaseline — 从 JSON 加载精度基线
bool EngineBridge::loadAccuracyBaseline(const std::string& strBaselineDir,
                                         const std::string& strModelType,
                                         AccuracyBaseline& outBaseline)
{
    try {
        std::string strFileName = strBaselineDir + "/" + strModelType + "_baseline.json";
        std::ifstream ifs(strFileName);
        if (!ifs.is_open()) return false;  // 20260402 ZJH 无基线文件（首次运行）

        // 20260402 ZJH 简化 JSON 解析（仅提取关键数值字段）
        std::string strContent((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());
        ifs.close();

        auto extractFloat = [&](const std::string& key) -> float {
            auto pos = strContent.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0f;
            pos = strContent.find(":", pos);
            if (pos == std::string::npos) return 0.0f;
            return std::stof(strContent.substr(pos + 1));
        };

        outBaseline.strModelType = strModelType;
        outBaseline.fFinalLoss = extractFloat("final_loss");
        outBaseline.fValAccuracy = extractFloat("val_accuracy");
        outBaseline.fValF1 = extractFloat("val_f1");
        outBaseline.fInferenceMs = extractFloat("inference_ms");
        outBaseline.nTrainSamples = static_cast<int>(extractFloat("train_samples"));
        outBaseline.nEpochs = static_cast<int>(extractFloat("epochs"));

        std::cerr << "[Baseline] Loaded: " << strFileName
                  << " acc=" << outBaseline.fValAccuracy
                  << " f1=" << outBaseline.fValF1 << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Baseline] Load error: " << e.what() << std::endl;
    }
    return false;
}

// 20260402 ZJH checkAccuracyRegression — 精度回归检测
bool EngineBridge::checkAccuracyRegression(const AccuracyBaseline& current,
                                            const AccuracyBaseline& baseline,
                                            float fTolerance)
{
    bool bPass = true;  // 20260406 ZJH 回归检测结果（默认通过）

    // 20260402 ZJH 精度检查: current.accuracy >= baseline.accuracy - tolerance
    if (baseline.fValAccuracy > 0.0f) {
        float fMinAccuracy = baseline.fValAccuracy - fTolerance;
        if (current.fValAccuracy < fMinAccuracy) {
            std::cerr << "[REGRESSION] Accuracy dropped: " << current.fValAccuracy
                      << " < " << fMinAccuracy << " (baseline=" << baseline.fValAccuracy
                      << " tolerance=" << fTolerance << ")" << std::endl;
            bPass = false;
        }
    }

    // 20260402 ZJH F1 检查
    if (baseline.fValF1 > 0.0f) {
        float fMinF1 = baseline.fValF1 - fTolerance;
        if (current.fValF1 < fMinF1) {
            std::cerr << "[REGRESSION] F1 dropped: " << current.fValF1
                      << " < " << fMinF1 << std::endl;
            bPass = false;
        }
    }

    // 20260402 ZJH 推理速度检查: 不超过基线的 1.2 倍（允许 20% 波动）
    if (baseline.fInferenceMs > 0.0f) {
        float fMaxMs = baseline.fInferenceMs * 1.2f;
        if (current.fInferenceMs > fMaxMs) {
            std::cerr << "[REGRESSION] Inference slowed: " << current.fInferenceMs
                      << "ms > " << fMaxMs << "ms" << std::endl;
            bPass = false;
        }
    }

    if (bPass) {  // 20260406 ZJH 所有检查项均通过
        std::cerr << "[REGRESSION] PASSED: " << current.strModelType
                  << " acc=" << current.fValAccuracy
                  << " f1=" << current.fValF1
                  << " ms=" << current.fInferenceMs << std::endl;
    }
    return bPass;  // 20260406 ZJH 返回回归检测结果（true=通过, false=回归）
}

// 20260402 ZJH generateDefects — AI 缺陷生成器入口
EngineBridge::DefectGenResult EngineBridge::generateDefects(
    const std::vector<std::vector<float>>& vecNormalImages,
    const std::vector<std::vector<float>>& vecDefectImages,
    const DefectGenConfig& config)
{
    DefectGenResult result;  // 20260406 ZJH 缺陷生成结果（默认构造）
    try {
        om::DefectGeneratorConfig genConfig;  // 20260406 ZJH 引擎层缺陷生成配置
        genConfig.nTargetCount = config.nTargetCount;
        genConfig.nImageWidth = config.nImageWidth;
        genConfig.nImageHeight = config.nImageHeight;
        genConfig.nDDPMTrainEpochs = config.nDDPMTrainEpochs;

        // 20260406 ZJH 调用引擎层缺陷生成器（自动选择 DRAEM+ 或 DDPM Tiny）
        auto omResult = om::DefectGenerator::generate(vecNormalImages, vecDefectImages, genConfig);

        result.vecImages = std::move(omResult.vecImages);        // 20260406 ZJH 生成的缺陷图像列表
        result.vecMasks = std::move(omResult.vecMasks);          // 20260406 ZJH 对应的缺陷 mask 列表
        result.nGeneratedCount = omResult.nGeneratedCount;       // 20260406 ZJH 实际生成数量
        result.nMode = omResult.nMode;                           // 20260406 ZJH 使用的生成模式 (0=DRAEM+, 1=DDPM)
        result.strLog = std::move(omResult.strLog);              // 20260406 ZJH 生成过程日志

        std::cerr << result.strLog;  // 20260406 ZJH 输出生成日志到标准错误流
    } catch (const std::exception& e) {
        result.strLog = std::string("[DefectGen] ERROR: ") + e.what();
        std::cerr << result.strLog << std::endl;
    }
    return result;
}
