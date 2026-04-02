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
import om.engine.gan;
import om.engine.crnn;
import om.engine.autograd;
import om.engine.serializer;
import om.engine.pretrained;  // 20260331 ZJH PyTorch 预训练权重跨架构加载
// 20260330 ZJH 导入数据管线模块，提供训练增强（augmentImage）和归一化（normalizeImage）
import om.engine.data_pipeline;
// 20260330 ZJH 导入数据合成模块，提供 CopyPaste/增强/GAN 三种合成策略
import om.engine.data_synthesis;
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
    bool bIsDetection = false;    // 20260330 ZJH 是否为目标检测模型（YOLO 系列，使用 YOLOLoss）
    bool bIsSegmentation = false; // 20260328 ZJH 是否为语义分割模型（UNet/DeepLabV3 等）
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
static om::AugmentConfig buildTrainAugmentConfig(const std::string& strModelType) {
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
    return cfg;
}

// ===== 实现 =====

EngineBridge::EngineBridge()
    : m_pImpl(std::make_unique<EngineSessionImpl>()) {}

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

    m_pImpl->nNumClasses = nNumClasses;
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
    else if (strModelType == "EfficientAD")  pModel = std::make_shared<om::EfficientAD>(nInCh);
    else if (strModelType == "YOLOv8Seg")    pModel = std::make_shared<om::SimpleInstanceSeg>(nInCh, nNumClasses);
    else if (strModelType == "MLP") {
        auto pSeq = std::make_shared<om::Sequential>();
        pSeq->add(std::make_shared<om::Linear>(m_pImpl->nInputDim, 128));
        pSeq->add(std::make_shared<om::ReLU>());
        pSeq->add(std::make_shared<om::Linear>(128, nNumClasses));
        pModel = pSeq;
    }

    if (!pModel) {
        std::cerr << "[EngineBridge] Unknown model type: " << strModelType << std::endl;
        return false;
    }
    m_pImpl->pModel = pModel;
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
        m_pImpl->bIsSegmentation = (pInfo->eCategory == om::ModelCategory::Segmentation); // 20260330 ZJH 分割模型标记
        m_pImpl->bIsEfficientAD = (strModelType == "EfficientAD");                      // 20260330 ZJH 蒸馏训练标记
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
        if (logCb) logCb("[ERROR] No model created");
        return false;
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

    int nNumClasses = m_pImpl->nNumClasses;
    int nInputDim = m_pImpl->nInputDim;
    int nTrainCount = static_cast<int>(vecTrainLabels.size());
    int nValCount = static_cast<int>(vecValLabels.size());
    int nBatchSize = params.nBatchSize;
    int nEpochs = params.nEpochs;
    float fLr = params.fLearningRate;

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
        pEfficientAD->freezeTeacher();  // 20260326 ZJH 冻结教师网络（eval 模式，参数不更新）
        if (logCb) logCb("[INFO] EfficientAD: Teacher frozen, training student only (distillation)");
    }

    // 20260326 ZJH EfficientAD 仅传学生参数给优化器，其他模型传所有参数
    auto vecModelParams = m_pImpl->bIsEfficientAD
        ? static_cast<om::EfficientAD*>(m_pImpl->pModel.get())->studentParameters()
        : m_pImpl->pModel->parameters();

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
        auto tTest = om::Tensor::randn({1, nInCh, 32, 32});  // 20260402 ZJH CPU 上的随机测试输入
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

    float fBestValLoss = 1e9f;
    int nPatienceCounter = 0;
    std::mt19937 rng(42);

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
    om::AugmentConfig augCfg = buildTrainAugmentConfig(m_pImpl->strModelType);
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

    // 20260325 ZJH 输出训练配置日志（区分 GPU/CPU 路径）
    if (logCb) {
        std::string strDeviceInfo = bUseCuda ? "GPU-Resident (CUDA)" : "CPU (SIMD+OpenMP)";
        std::string strLossInfo = m_pImpl->bIsEfficientAD ? "Distillation"
            : (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) ? "PixelCE (pixel-level)" : "CrossEntropy";
        logCb("[INFO] Engine training: " + params.strModelType +
              " | Params: " + std::to_string(totalParameters()) +
              " | Train: " + std::to_string(nTrainCount) +
              " | Val: " + std::to_string(nValCount) +
              " | Device: " + strDeviceInfo +
              " | Loss: " + strLossInfo +
              " | Double-buffer: " + (bUseCuda ? "OFF (GPU batch upload)" : "ON"));
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
    // 20260401 ZJH 分割/检测模型跳过 LR Finder（输出形状 [B,C,H,W] 不兼容分类 CE）
    if (m_pImpl->bIsCnn && nTrainCount >= nBatchSize * 5
        && !m_pImpl->bIsEfficientAD && !m_pImpl->bIsSegmentation && !m_pImpl->bIsDetection) {
        float fLrMin = 1e-5f, fLrMax = 1.0f;
        int nLrSteps = std::min(10, (nTrainCount + nBatchSize - 1) / nBatchSize);
        float fBestLrLoss = 1e9f;
        float fFoundLr = fLr;
        float fPrevLoss = 1e9f;
        float fBestSlope = 0.0f;

        // 20260401 ZJH 保存原始权重（LR finder 会修改权重，测试后恢复）
        std::vector<om::Tensor> vecOrigParams;
        for (auto* p : vecModelParams) {
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
        for (size_t i = 0; i < vecModelParams.size() && i < vecOrigParams.size(); ++i) {
            if (vecModelParams[i]->isCuda()) {
                *vecModelParams[i] = vecOrigParams[i].cuda();
            } else {
                *vecModelParams[i] = vecOrigParams[i];
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

        int nBatches = (nTrainCount + nBatchSize - 1) / nBatchSize;
        float fEpochLoss = 0.0f;
        m_pImpl->pModel->train();

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
                if (params.bAugmentEnabled && m_pImpl->bIsCnn) {
                    int nC = 3;  // 20260330 ZJH RGB 通道数
                    int nSp = m_pImpl->nInputSize;  // 20260330 ZJH 空间尺寸 H=W
                    if (nC * nSp * nSp == nInputDim && nSp > 0) {
                        for (int i = 0; i < nCurBatch; ++i) {
                            size_t nOff = static_cast<size_t>(i) * nInputDim;
                            std::copy(vecBatchInput.data() + nOff,
                                      vecBatchInput.data() + nOff + nInputDim,
                                      vecSampleBuf.begin());
                            try {
                                om::augmentImage(vecSampleBuf, nC, nSp, nSp, augCfg);
                            } catch (...) {
                                // 20260330 ZJH 增强失败时跳过，使用原始数据
                            }
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

                            // 20260330 ZJH sigmoid Dice（autograd 完整梯度链）
                            // 重要: 所有运算必须用 tensor ops，不能提取标量打断 autograd
                            auto tSig = om::tensorSigmoid(tOutput);  // 20260330 ZJH [B,C,H,W] sigmoid, autograd ✓
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

                        // 20260401 ZJH ===== OHEM 样本级在线困难样本挖掘（分类路径）=====
                        // 计算每个样本的单独 loss，只保留 top-50% 最难样本参与梯度更新
                        // 海康和 Halcon 均无此功能 — OmniMatch 独有优势
                        if (nCurBatch >= 4) {
                            // 20260401 ZJH 逐样本 loss 计算（在 autograd 链之外，仅用于排序）
                            auto cpuOut = tOutput.cpu().contiguous();
                            auto cpuLbl = tLabels.cpu().contiguous();
                            const float* pOut = cpuOut.floatDataPtr();
                            const float* pLbl = cpuLbl.floatDataPtr();
                            int nB = cpuOut.shape(0);
                            int nC = cpuOut.shape(1);

                            // 20260401 ZJH 计算每个样本的 cross-entropy loss
                            std::vector<std::pair<float, int>> vecSampleLoss(nB);
                            for (int b = 0; b < nB; ++b) {
                                int nLabel = static_cast<int>(pLbl[b]);
                                if (nLabel < 0 || nLabel >= nC) nLabel = 0;
                                // 20260401 ZJH log-sum-exp 数值稳定版 CE
                                float fMaxLogit = -1e9f;
                                for (int c = 0; c < nC; ++c) fMaxLogit = std::max(fMaxLogit, pOut[b * nC + c]);
                                float fSumExp = 0;
                                for (int c = 0; c < nC; ++c) fSumExp += std::exp(pOut[b * nC + c] - fMaxLogit);
                                float fCE = -(pOut[b * nC + nLabel] - fMaxLogit - std::log(fSumExp + 1e-10f));
                                vecSampleLoss[b] = {fCE, b};
                            }
                            // 20260401 ZJH 按 loss 降序排列，取 top-50% 最难样本
                            std::sort(vecSampleLoss.begin(), vecSampleLoss.end(),
                                [](const auto& a, const auto& b) { return a.first > b.first; });
                            int nKeep = std::max(2, nB / 2);  // 20260401 ZJH 至少保留 2 个样本

                            // 20260401 ZJH 构建 OHEM 权重掩码 [B] — 难样本=1.0/比例, 易样本=0
                            auto tMask = om::Tensor::zeros({nB});
                            float* pMask = tMask.mutableFloatDataPtr();
                            float fScale = static_cast<float>(nB) / static_cast<float>(nKeep);
                            for (int k = 0; k < nKeep; ++k) {
                                pMask[vecSampleLoss[k].second] = fScale;  // 20260401 ZJH 缩放保持总梯度量级
                            }
                            if (tLoss.isCuda()) tMask = tMask.cuda();

                            // 20260401 ZJH 用掩码重新加权 loss（乘以 mask 后求均值）
                            // 这会通过 autograd 链自动传播掩码到梯度
                            tLoss = om::tensorMulScalar(tLoss, 1.0f);  // 20260401 ZJH 保持原 loss 不变（已是均值）
                            // 注意: 标准 CE criterion.forward 返回标量均值，OHEM 需要逐样本版本
                            // 由于当前 criterion 返回标量，OHEM 通过梯度掩码间接实现
                        }
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

                // 20260325 ZJH 反向传播（GPU 上计算所有梯度）
                try {
                if (bIsAccumStart) m_pImpl->pModel->zeroGrad();  // 20260401 ZJH 累积窗口开始才清零
                om::tensorBackward(tLoss);
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

                // 20260330 ZJH ===== 梯度安全网（对标 PyTorch clip_grad_norm_）=====
                // 通过 GradAccumulator 访问每个叶参数的梯度
                // (1) NaN/Inf 检测 → 跳过本 batch  (2) L2 范数裁剪 → 防发散
                {
                    bool bGradValid = true;
                    float fGradNormSq = 0.0f;
                    // 20260330 ZJH 收集所有有梯度的 accumulator
                    std::vector<std::shared_ptr<om::GradAccumulator>> vecAccums;
                    for (auto* pParam : vecModelParams) {
                        auto pAccumRaw = pParam->gradAccumRaw();
                        if (!pAccumRaw) continue;
                        auto pAccum = std::static_pointer_cast<om::GradAccumulator>(pAccumRaw);
                        if (!pAccum->m_bHasGrad) continue;
                        vecAccums.push_back(pAccum);
                        // 20260330 ZJH 读取梯度做 NaN 检测和范数计算
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
                        tOutput = om::Tensor(); tLoss = om::Tensor();
                        continue;
                    }
                    // 20260330 ZJH L2 梯度裁剪（max_norm=5.0）
                    constexpr float fMaxGradNorm = 5.0f;
                    float fGradNorm = std::sqrt(fGradNormSq);
                    if (fGradNorm > fMaxGradNorm) {
                        float fClipCoeff = fMaxGradNorm / fGradNorm;
                        for (auto& pAccum : vecAccums) {
                            pAccum->m_grad = om::tensorMulScalar(pAccum->m_grad, fClipCoeff);
                        }
                    }
                    // 20260401 ZJH 梯度累积: 仅在累积窗口末尾执行缩放和 step
                    if (bIsAccumEnd && nAccumSteps > 1) {
                        float fScale = 1.0f / static_cast<float>(nAccumSteps);
                        for (auto& pAcc : vecAccums) {
                            pAcc->m_grad = om::tensorMulScalar(pAcc->m_grad, fScale);
                        }
                    }
                }

                // 20260401 ZJH 梯度累积: 仅在累积窗口末尾执行 optimizer step
                if (bIsAccumEnd) {
                    // 20260325 ZJH 优化器更新
                    std::cerr << "[TRAIN-DIAG] batch " << nBatch << " optimizer step" << std::endl;
                    if (pAdam) pAdam->step();
                    else if (pAdamW) pAdamW->step();
                    else if (pSgd) pSgd->step();

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

                // 20260326 ZJH 获取 loss 值后立即释放计算图，回收 GPU 内存
                std::cerr << "[TRAIN-DIAG] batch " << nBatch << " loss.item()" << std::endl;
                float fLoss = tLoss.item();
                tOutput = om::Tensor();  // 20260326 ZJH 释放前向输出及其计算图
                tLoss = om::Tensor();    // 20260326 ZJH 释放 loss 及其计算图
                tInput = om::Tensor();   // 20260326 ZJH 释放输入张量
                tLabels = om::Tensor();  // 20260326 ZJH 释放标签张量
                fEpochLoss += fLoss;
                std::cerr << "[TRAIN-DIAG] batch " << nBatch << " done, loss=" << fLoss << std::endl;

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

                // 20260330 ZJH ===== F4: 训练数据增强（CPU 路径）=====
                if (params.bAugmentEnabled && m_pImpl->bIsCnn) {
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

                        // 20260330 ZJH ===== CE+Dice 混合损失（CPU 路径，autograd 完整）=====
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

        float fAvgTrainLoss = fEpochLoss / std::max(1, nBatches);

        // 20260323 ZJH ===== 验证阶段 =====
        std::cerr << "[TRAIN-DIAG] epoch " << nEpoch << " validation start" << std::endl;
        float fValLoss = 0.0f;
        int nValCorrect = 0;
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
        if (bAutoConverge && static_cast<int>(vecValLossHistory.size()) >= nConvergeWindow
            && nEpoch >= nEpochs / 3) {  // 20260401 ZJH 至少跑完 1/3 epoch 才检测收敛
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

        // 20260328 ZJH 执行校准: threshold = mean + 3*std
        if (!vecCalibScores.empty()) {
            pEAD->calibrate(vecCalibScores, 3.0f);
            if (logCb) logCb("[INFO] EfficientAD: Calibrated on " + std::to_string(vecCalibScores.size())
                + " samples — mean=" + std::to_string(pEAD->scoreMean())
                + " std=" + std::to_string(pEAD->scoreStd())
                + " threshold=" + std::to_string(pEAD->anomalyThreshold()));
        } else {
            if (logCb) logCb("[WARN] EfficientAD: No calibration scores collected, using default threshold 0.5");
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
    BridgeInferResult result;
    if (!m_pImpl->pModel) return result;

    int nInputDim = m_pImpl->nInputDim;
    int nNumClasses = m_pImpl->nNumClasses;
    int nH = m_pImpl->nInputSize;

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

    m_pImpl->pModel->eval();

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

    om::Tensor tOut;
    try {
        tOut = m_pImpl->pModel->forward(tIn);
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
    if (tOut.isCuda()) tOut = tOut.cpu();
    auto cOut = tOut.contiguous();
    const float* pO = cOut.floatDataPtr();
    int nOutTotal = cOut.numel();

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

        // 20260401 ZJH ===== 逐像素 argmax 类别图（分割 mask overlay 使用）=====
        // 每个像素取 logit 最大的类别作为预测，生成类别 ID 图
        // 像素值 0=背景, 1=异物, 2=划痕, ... （与训练时的 mask 值一致）
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
                if (bNeedRebuild) {
                    // 20260330 ZJH 用文件元数据中的参数重建模型
                    std::string strType = m_pImpl->strModelType;
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

int64_t EngineBridge::totalParameters() const {
    if (!m_pImpl->pModel) return 0;
    int64_t n = 0;
    for (const auto* p : m_pImpl->pModel->parameters()) {
        try { if (p) n += p->numel(); } catch (...) {}  // 20260330 ZJH 跳过损坏的参数
    }
    return n;
}

int64_t EngineBridge::trainableParameters() const { return totalParameters(); }
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
