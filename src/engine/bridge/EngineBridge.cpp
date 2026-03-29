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
// 20260324 ZJH 导入 HAL 层模块，获取 GPU 加速开关接口（setGpuAcceleration/isGpuAccelerationEnabled）
import om.hal.cpu_backend;
// 20260325 ZJH 导入 CUDA 后端模块，提供 GPU 全算子加速（Phase 4 GPU-Resident 训练依赖）
#ifdef OM_HAS_CUDA
import om.hal.cuda_backend;
#endif

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
    bool bIsCnn = false;    // 20260325 ZJH 是否为 CNN 模型（需要 4D 输入 [B,3,H,W]）
    bool bIsEfficientAD = false;  // 20260326 ZJH EfficientAD 异常检测模型标记（蒸馏训练 + 空间异常图）
    bool bIsSegmentation = false; // 20260328 ZJH 是否为语义分割模型（UNet/DeepLabV3 等，逐像素 Dice Loss 训练）
};

// 20260323 ZJH 简单 MLP 分类器
class SimpleMLP : public om::Module {
public:
    SimpleMLP(int nInputDim, int nNumClasses)
        : m_fc1(nInputDim, 256), m_fc2(256, 128), m_fc3(128, nNumClasses)
    {
        registerModule("fc1", std::make_shared<om::Linear>(m_fc1));
        registerModule("fc2", std::make_shared<om::Linear>(m_fc2));
        registerModule("fc3", std::make_shared<om::Linear>(m_fc3));
    }
    om::Tensor forward(const om::Tensor& input) override {
        auto x = m_fc1.forward(input);
        x = m_relu.forward(x);
        x = m_fc2.forward(x);
        x = m_relu.forward(x);
        x = m_fc3.forward(x);
        return x;
    }
private:
    om::Linear m_fc1, m_fc2, m_fc3;
    om::ReLU m_relu;
};

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

// ===== 实现 =====

EngineBridge::EngineBridge()
    : m_pImpl(std::make_unique<EngineSessionImpl>()) {}

EngineBridge::~EngineBridge() = default;

// 20260323 ZJH 创建模型
bool EngineBridge::createModel(const std::string& strModelType, int nInputSize, int nNumClasses)
{
    // 20260324 ZJH 验证 nInputSize 范围，防止平方运算导致整数溢出
    if (nInputSize < 1 || nInputSize > 10000) {
        return false;  // 20260324 ZJH 输入尺寸超出合理范围（1-10000），拒绝创建模型
    }

    m_pImpl->nNumClasses = nNumClasses;
    m_pImpl->nInputSize = nInputSize;  // 20260325 ZJH 记录空间尺寸

    // 20260324 ZJH 计算展平输入维度: 3 通道 (RGB) * H * W
    int64_t nInputDim64 = 3LL * static_cast<int64_t>(nInputSize) * static_cast<int64_t>(nInputSize);
    m_pImpl->nInputDim = static_cast<int>(nInputDim64);

    // 20260325 ZJH 根据模型类型字符串创建对应的模型实例
    // 支持 UI 显示名称（去除连字符和空格后）和内部名称两种格式

    // ===== 分类模型 =====
    if (strModelType == "MLP" || strModelType == "SimpleMLP") {
        m_pImpl->bIsCnn = false;  // 20260325 ZJH MLP 使用展平输入
        m_pImpl->pModel = std::make_shared<SimpleMLP>(m_pImpl->nInputDim, nNumClasses);
        return true;
    }
    // 20260325 ZJH 以下均为 CNN 模型，需要 4D 输入 [B, 3, H, W]
    m_pImpl->bIsCnn = true;
    if (strModelType == "ResNet18" || strModelType == "ResNet18") {
        m_pImpl->pModel = std::make_shared<om::ResNet18>(nNumClasses);
        return true;
    }
    if (strModelType == "ResNet50" || strModelType == "ResNet50") {
        m_pImpl->pModel = std::make_shared<om::ResNet50>(nNumClasses);
        return true;
    }
    if (strModelType == "MobileNetV4Small" || strModelType == "MobileNetV4Small") {
        m_pImpl->pModel = std::make_shared<om::MobileNetV4Small>(nNumClasses);
        return true;
    }
    if (strModelType == "ViTTiny" || strModelType == "ViTTiny") {
        // 20260325 ZJH ViT 需要 patch_size=16, embed_dim=192, depth=12, heads=3
        m_pImpl->pModel = std::make_shared<om::ViT>(
            nInputSize, 16, 3, nNumClasses, 192, 12, 3);
        return true;
    }

    // ===== 目标检测模型 =====
    if (strModelType == "YOLOv5Nano" || strModelType == "YOLOv5Nano") {
        m_pImpl->pModel = std::make_shared<om::YOLOv5Nano>(nNumClasses);
        return true;
    }
    if (strModelType == "YOLOv8Nano" || strModelType == "YOLOv8Nano") {
        m_pImpl->pModel = std::make_shared<om::YOLOv8Nano>(nNumClasses);
        return true;
    }

    // ===== 语义分割模型 =====
    if (strModelType == "UNet" || strModelType == "UNet") {
        m_pImpl->pModel = std::make_shared<om::UNet>(3, nNumClasses);
        m_pImpl->bIsSegmentation = true;  // 20260328 ZJH 标记走分割训练路径（逐像素 Dice Loss）
        return true;
    }
    if (strModelType == "DeepLabV3+" || strModelType == "DeepLabV3Plus" || strModelType == "DeepLabV3") {
        m_pImpl->pModel = std::make_shared<om::DeepLabV3>(nNumClasses);
        m_pImpl->bIsSegmentation = true;  // 20260328 ZJH 标记走分割训练路径（逐像素 Dice Loss）
        return true;
    }

    // ===== 异常检测模型 =====
    if (strModelType == "EfficientAD") {
        m_pImpl->pModel = std::make_shared<om::EfficientAD>();
        m_pImpl->bIsEfficientAD = true;  // 20260326 ZJH 标记走蒸馏训练路径
        return true;
    }

    // ===== 实例分割模型 =====
    if (strModelType == "YOLOv8Seg" || strModelType == "YOLOv8InstanceSeg") {
        // 20260325 ZJH SimpleInstanceSeg(nInChannels=3, nNumClasses, nNumPrototypes=32)
        m_pImpl->pModel = std::make_shared<om::SimpleInstanceSeg>(3, nNumClasses, 32);
        return true;
    }
    if (strModelType == "MaskRCNN" || strModelType == "MaskRCNN") {
        // 20260325 ZJH Mask R-CNN → SimpleInstanceSeg(nInChannels=3, nNumClasses, nNumPrototypes=32)
        m_pImpl->pModel = std::make_shared<om::SimpleInstanceSeg>(3, nNumClasses, 32);
        return true;
    }

    // ===== 未实现的架构：明确返回 false =====
    // 20260325 ZJH 以下架构暂未在引擎中实现，UI 层会显示"无法创建模型"提示
    // EfficientNetB0, MobileNetV4Medium, ConvNeXtTiny, RepVGGA0,
    // PaDiM, PatchCore, FastFlow, YOLOv11Nano, RTDETR,
    // PSPNet, SegFormer, PaddleOCRv4, PPOCR,
    // WinCLIP, AnomalyCLIP, GroundingDINO, YOLOWorld
    return false;
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

    // 20260325 ZJH ===== GPU 路径：将模型参数迁移到 GPU =====
#ifdef OM_HAS_CUDA
    if (bUseCuda) {
        // 20260326 ZJH Step 1: 迁移可学习参数（weight, bias, gamma, beta）
        auto vecAllParams = m_pImpl->pModel->parameters();
        for (auto* pParam : vecAllParams) {
            if (pParam->isCpu()) *pParam = pParam->cuda();
        }
        // 20260326 ZJH Step 2: 迁移缓冲区（BN running_mean, running_var）
        // buffers() 独立于 parameters()，不影响优化器
        auto vecBufs = m_pImpl->pModel->buffers();
        for (auto* pBuf : vecBufs) {
            if (pBuf->isCpu()) *pBuf = pBuf->cuda();
        }
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

    // 20260323 ZJH 创建优化器（使用迁移后的参数列表，确保优化器持有 GPU 参数引用）
    std::unique_ptr<om::Adam> pAdam;
    std::unique_ptr<om::SGD> pSgd;
    if (params.strOptimizer == "SGD") {
        pSgd = std::make_unique<om::SGD>(vecModelParams, fLr, params.fMomentum);
    } else {
        pAdam = std::make_unique<om::Adam>(vecModelParams, fLr);
    }

    om::CrossEntropyLoss criterion;  // 20260323 ZJH 损失函数（分类模型使用）

    float fBestValLoss = 1e9f;
    int nPatienceCounter = 0;
    std::mt19937 rng(42);

    // 20260324 ZJH 索引洗牌优化：洗牌索引数组而非实际数据
    // 数据保持原位不动，通过 vecIndices 间接访问，减少内存搬运
    std::vector<int> vecIndices(nTrainCount);
    std::iota(vecIndices.begin(), vecIndices.end(), 0);  // 20260324 ZJH 初始化为 0..N-1

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
            : (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) ? "Dice (pixel-level)" : "CrossEntropy";
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

        // 20260326 ZJH Cosine Annealing 学习率调度（Halcon 级训练标配）
        // lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(π * epoch / T_max))
        // lr_min = lr_init * 0.01（最终衰减到初始值的 1%）
        {
            float fMinLr = fLr * 0.01f;  // 20260326 ZJH 最小学习率 = 初始的 1%
            float fCosLr = fMinLr + 0.5f * (fLr - fMinLr)
                         * (1.0f + std::cos(3.14159265f * static_cast<float>(nEpoch) / static_cast<float>(nEpochs)));
            if (pAdam) pAdam->setLearningRate(fCosLr);
            else if (pSgd) pSgd->setLearningRate(fCosLr);
            fCurrentLr = fCosLr;  // 20260326 ZJH 更新当前学习率供日志输出
        }

        int nBatches = (nTrainCount + nBatchSize - 1) / nBatchSize;
        float fEpochLoss = 0.0f;
        m_pImpl->pModel->train();

        if (bUseCuda) {
            // 20260326 ZJH ===== GPU 训练循环（预分配缓冲区优化）=====
            // 优化：batch 缓冲区在循环外预分配，避免每 batch malloc/free 开销
            std::vector<float> vecBatchInput(static_cast<size_t>(nBatchSize) * nInputDim);
            std::vector<float> vecBatchLabel(static_cast<size_t>(nBatchSize) * nNumClasses, 0.0f);

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

                auto tInput = m_pImpl->bIsCnn
                    ? om::Tensor::fromData(vecBatchInput.data(), {nCurBatch, 3, m_pImpl->nInputSize, m_pImpl->nInputSize})
                    : om::Tensor::fromData(vecBatchInput.data(), {nCurBatch, nInputDim});
                auto tLabels = om::Tensor::fromData(vecBatchLabel.data(), {nCurBatch, nNumClasses});
#ifdef OM_HAS_CUDA
                tInput = tInput.cuda();
                tLabels = tLabels.cuda();
#endif

                // 20260325 ZJH 前向传播（全部在 GPU 上执行，tensor_ops 按设备类型自动调度）
                om::Tensor tOutput, tLoss;
                try {
                    // 20260326 ZJH EfficientAD 使用蒸馏损失，其他模型使用交叉熵
                    if (m_pImpl->bIsEfficientAD) {
                        auto* pEAD = static_cast<om::EfficientAD*>(m_pImpl->pModel.get());
                        tLoss = pEAD->computeDistillationLoss(tInput);
                    } else if (m_pImpl->bIsSegmentation && !vecTrainMasks.empty()) {
                        // 20260328 ZJH ===== 分割模型 Fused Dice Loss（1 kernel 替代 10 步 ops）=====
                        tOutput = m_pImpl->pModel->forward(tInput);  // 20260328 ZJH [B, C, H, W] logits

                        int nH = m_pImpl->nInputSize;   // 20260328 ZJH 空间高度
                        int nSpatial = nH * nH;         // 20260328 ZJH 每张图像的像素数 H*W
                        int nC = m_pImpl->nNumClasses;   // 20260328 ZJH 分割类别数

                        // 20260328 ZJH 构建 one-hot 目标掩码 [B, C, H, W]（CPU 上构建后上传 GPU）
                        auto tMaskOneHot = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                        float* pMaskOH = tMaskOneHot.mutableFloatDataPtr();  // 20260328 ZJH 可写指针
                        for (int i = 0; i < nCurBatch; ++i) {
                            int nIdx = vecIndices[nStart + i];  // 20260328 ZJH 洗牌后的样本索引
                            for (int s = 0; s < nSpatial; ++s) {
                                int nMaskOffset = nIdx * nSpatial + s;  // 20260328 ZJH 像素偏移
                                if (nMaskOffset < static_cast<int>(vecTrainMasks.size())) {
                                    int nClass = vecTrainMasks[nMaskOffset];  // 20260328 ZJH 类别 ID
                                    if (nClass >= 0 && nClass < nC) {
                                        pMaskOH[static_cast<size_t>(i) * nC * nSpatial
                                                + static_cast<size_t>(nClass) * nSpatial + s] = 1.0f;  // 20260328 ZJH one-hot 赋值
                                    }
                                }
                            }
                        }
#ifdef OM_HAS_CUDA
                        if (bUseCuda) {
                            tMaskOneHot = tMaskOneHot.cuda();  // 20260328 ZJH H2D 上传掩码
                        }
#endif
                        // 20260328 ZJH ===== 逐类 Dice Loss（Per-Class Mean Dice）=====
                        // 全局 Dice 被背景 99.9% 像素淹没缺陷 0.1% 信号
                        // 逐类 Dice：每个类别独立计算 Dice 后取平均，权重均等
                        // 20260328 ZJH ===== 多类分割 Dice Loss: 逐非背景类，跳过空类 =====
                        // 对每个非背景通道(c>=1)独立计算 Dice，空类跳过，最后取平均
                        // autograd 完整: selector mask 是常量，梯度只传给 tSig
                        auto tSig = om::tensorSigmoid(tOutput);  // 20260328 ZJH [B,C,H,W], autograd ✓

                        int nValidClasses = 0;  // 20260328 ZJH 有像素的非背景类计数
                        om::Tensor tTotalDice = om::Tensor::full({1}, 0.0f);
#ifdef OM_HAS_CUDA
                        if (bUseCuda) tTotalDice = tTotalDice.cuda();
#endif
                        for (int c = 1; c < nC; ++c) {  // 20260328 ZJH 跳过 c=0 背景
                            // 20260328 ZJH 先检查该类在 batch 中是否有 target 像素
                            // 从 CPU 端 one-hot 数据检查（tMaskOneHot 构建在 CPU 上再上传）
                            bool bHasPixels = false;
                            {
                                // 20260328 ZJH 快速扫描 one-hot 的 channel c 是否有 1.0
                                auto cMaskCpu = tMaskOneHot.isCuda() ? tMaskOneHot.cpu().contiguous() : tMaskOneHot.contiguous();
                                const float* pM = cMaskCpu.floatDataPtr();
                                for (int b = 0; b < nCurBatch && !bHasPixels; ++b) {
                                    for (int s = 0; s < nSpatial && !bHasPixels; ++s) {
                                        if (pM[b * nC * nSpatial + c * nSpatial + s] > 0.5f) bHasPixels = true;
                                    }
                                }
                            }
                            if (!bHasPixels) continue;  // 20260328 ZJH 该类无标注像素，跳过

                            // 20260328 ZJH 构建 channel c selector [B,C,H,W]
                            auto tChSel = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                            {
                                float* pSel = tChSel.mutableFloatDataPtr();
                                for (int b = 0; b < nCurBatch; ++b) {
                                    for (int s = 0; s < nSpatial; ++s) {
                                        pSel[b * nC * nSpatial + c * nSpatial + s] = 1.0f;
                                    }
                                }
                            }
#ifdef OM_HAS_CUDA
                            if (bUseCuda) tChSel = tChSel.cuda();
#endif
                            // 20260328 ZJH 该类的预测和目标（autograd 传导到 tSig）
                            auto tPredC = om::tensorMul(tSig, tChSel);
                            auto tTargC = om::tensorMul(tMaskOneHot, tChSel);
                            auto tInterC = om::tensorSum(om::tensorMul(tPredC, tTargC));
                            auto tPredSumC = om::tensorSum(tPredC);
                            auto tTargSumC = om::tensorSum(tTargC);
                            auto tDenomC = om::tensorAdd(tPredSumC, tTargSumC);
                            auto tEpsC = om::Tensor::full({1}, 1e-6f);
#ifdef OM_HAS_CUDA
                            if (bUseCuda) tEpsC = tEpsC.cuda();
#endif
                            tDenomC = om::tensorAdd(tDenomC, tEpsC);
                            auto tNumerC = om::tensorMulScalar(tInterC, 2.0f);
                            float fDenomVal = tDenomC.isCuda() ? tDenomC.cpu().contiguous().floatDataPtr()[0]
                                                                : tDenomC.contiguous().floatDataPtr()[0];
                            auto tDiceC = om::tensorMulScalar(tNumerC, 1.0f / fDenomVal);
                            tTotalDice = om::tensorAdd(tTotalDice, tDiceC);
                            ++nValidClasses;
                        }
                        // 20260328 ZJH mean Dice over valid classes → loss = 1 - meanDice
                        float fDivisor = (nValidClasses > 0) ? static_cast<float>(nValidClasses) : 1.0f;
                        auto tMeanDice = om::tensorMulScalar(tTotalDice, 1.0f / fDivisor);
                        auto tOne = om::Tensor::full({1}, 1.0f);
#ifdef OM_HAS_CUDA
                        if (bUseCuda) tOne = tOne.cuda();
#endif
                        tLoss = om::tensorSub(tOne, tMeanDice);
                    } else {
                        tOutput = m_pImpl->pModel->forward(tInput);
                        tLoss = criterion.forward(tOutput, tLabels);
                    }
                } catch (const std::exception& ex) {
                    if (logCb) logCb(std::string("[错误] 前向传播异常: ") + ex.what()
                        + " | 输入形状: [" + std::to_string(tInput.shape(0))
                        + (tInput.shapeVec().size() > 1 ? "," + std::to_string(tInput.shape(1)) : "")
                        + (tInput.shapeVec().size() > 2 ? "," + std::to_string(tInput.shape(2)) : "")
                        + (tInput.shapeVec().size() > 3 ? "," + std::to_string(tInput.shape(3)) : "")
                        + "] 请尝试减小输入尺寸或批量大小");
                    // 20260325 ZJH GPU OOM 时不尝试 D2H（会再次失败），直接清理
#ifdef OM_HAS_CUDA
                    if (bUseCuda) { try { omCudaCleanup(); } catch (...) {} }
#endif
                    return false;
                }

                // 20260325 ZJH 反向传播（GPU 上计算所有梯度）
                m_pImpl->pModel->zeroGrad();
                om::tensorBackward(tLoss);

                // 20260325 ZJH 优化器更新
                if (pAdam) pAdam->step();
                else if (pSgd) pSgd->step();

                // 20260326 ZJH 获取 loss 值后立即释放计算图，回收 GPU 内存
                // 前向传播的 autograd 图保存所有中间张量（数百 MB），不释放下个 batch 会 OOM
                float fLoss = tLoss.item();
                tOutput = om::Tensor();  // 20260326 ZJH 释放前向输出及其计算图
                tLoss = om::Tensor();    // 20260326 ZJH 释放 loss 及其计算图
                tInput = om::Tensor();   // 20260326 ZJH 释放输入张量
                tLabels = om::Tensor();  // 20260326 ZJH 释放标签张量
                fEpochLoss += fLoss;

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
                        // 20260328 ZJH ===== CPU 分割模型 Fused Dice Loss =====
                        tOutput = m_pImpl->pModel->forward(tInput);  // 20260328 ZJH [B, C, H, W] logits

                        int nH = m_pImpl->nInputSize;   // 20260328 ZJH 空间高度
                        int nSpatial = nH * nH;         // 20260328 ZJH 每张图像的像素数 H*W
                        int nC = m_pImpl->nNumClasses;   // 20260328 ZJH 分割类别数

                        // 20260328 ZJH 构建 one-hot 目标掩码 [B, C, H, W]
                        auto tMaskOneHot = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                        float* pMaskOH = tMaskOneHot.mutableFloatDataPtr();  // 20260328 ZJH 可写指针
                        for (int i = 0; i < nCurBatch; ++i) {
                            int nIdx = vecIndices[nStart + i];  // 20260328 ZJH 洗牌后的样本索引
                            for (int s = 0; s < nSpatial; ++s) {
                                int nMaskOffset = nIdx * nSpatial + s;  // 20260328 ZJH 像素偏移
                                if (nMaskOffset < static_cast<int>(vecTrainMasks.size())) {
                                    int nClass = vecTrainMasks[nMaskOffset];  // 20260328 ZJH 类别 ID
                                    if (nClass >= 0 && nClass < nC) {
                                        pMaskOH[static_cast<size_t>(i) * nC * nSpatial
                                                + static_cast<size_t>(nClass) * nSpatial + s] = 1.0f;  // 20260328 ZJH one-hot 赋值
                                    }
                                }
                            }
                        }
                        // 20260328 ZJH ===== CPU: 多类 Dice Loss (逐非背景类，跳过空类) =====
                        auto tSig = om::tensorSigmoid(tOutput);
                        int nValidClasses = 0;
                        om::Tensor tTotalDice = om::Tensor::full({1}, 0.0f);
                        for (int c = 1; c < nC; ++c) {
                            // 20260328 ZJH 检查该类是否有像素
                            bool bHasPixels = false;
                            {
                                auto cM = tMaskOneHot.contiguous();
                                const float* pM = cM.floatDataPtr();
                                for (int b = 0; b < nCurBatch && !bHasPixels; ++b)
                                    for (int s = 0; s < nSpatial && !bHasPixels; ++s)
                                        if (pM[b * nC * nSpatial + c * nSpatial + s] > 0.5f) bHasPixels = true;
                            }
                            if (!bHasPixels) continue;
                            auto tChSel = om::Tensor::zeros({nCurBatch, nC, nH, nH});
                            {
                                float* pSel = tChSel.mutableFloatDataPtr();
                                for (int b = 0; b < nCurBatch; ++b)
                                    for (int s = 0; s < nSpatial; ++s)
                                        pSel[b * nC * nSpatial + c * nSpatial + s] = 1.0f;
                            }
                            auto tPredC = om::tensorMul(tSig, tChSel);
                            auto tTargC = om::tensorMul(tMaskOneHot, tChSel);
                            auto tInterC = om::tensorSum(om::tensorMul(tPredC, tTargC));
                            auto tPredSumC = om::tensorSum(tPredC);
                            auto tTargSumC = om::tensorSum(tTargC);
                            auto tDenomC = om::tensorAdd(om::tensorAdd(tPredSumC, tTargSumC), om::Tensor::full({1}, 1e-6f));
                            auto tNumerC = om::tensorMulScalar(tInterC, 2.0f);
                            float fDenomVal = tDenomC.contiguous().floatDataPtr()[0];
                            auto tDiceC = om::tensorMulScalar(tNumerC, 1.0f / fDenomVal);
                            tTotalDice = om::tensorAdd(tTotalDice, tDiceC);
                            ++nValidClasses;
                        }
                        float fDiv = (nValidClasses > 0) ? static_cast<float>(nValidClasses) : 1.0f;
                        auto tMeanDice = om::tensorMulScalar(tTotalDice, 1.0f / fDiv);
                        tLoss = om::tensorSub(om::Tensor::full({1}, 1.0f), tMeanDice);
                    } else {
                        tOutput = m_pImpl->pModel->forward(tInput);
                        tLoss = criterion.forward(tOutput, tLabels);
                    }
                } catch (const std::exception& ex) {
                    if (logCb) logCb(std::string("[错误] 前向传播异常: ") + ex.what()
                        + " 请尝试减小输入尺寸或批量大小");
                    return false;
                }

                m_pImpl->pModel->zeroGrad();
                om::tensorBackward(tLoss);

                if (pAdam) pAdam->step();
                else if (pSgd) pSgd->step();

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

                auto tO = m_pImpl->pModel->forward(tI);

                // 20260328 ZJH 分割模型验证：计算 Dice Loss + 像素级准确率
                if (m_pImpl->bIsSegmentation && !vecValMasks.empty()) {
                    // 20260328 ZJH 分割验证路径：Dice Loss + 逐像素准确率
                    int nH = m_pImpl->nInputSize;
                    int nSpatial = nH * nH;
                    int nC = m_pImpl->nNumClasses;

                    // 20260328 ZJH 确保输出在 CPU 上以便读取
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

                    for (int i = 0; i < nc; ++i) {
                        int nIdx = s + i;  // 20260328 ZJH 验证集样本索引
                        // 20260328 ZJH 逐像素 argmax 计算准确率
                        for (int px = 0; px < nSpatial; ++px) {
                            // 20260328 ZJH 模型输出布局: [B, C, H, W] -> pValOut[i*C*spatial + c*spatial + px]
                            int nBestClass = 0;
                            float fBestVal = pValOut[static_cast<size_t>(i) * nC * nSpatial + px];
                            for (int c = 1; c < nC; ++c) {
                                float fVal = pValOut[static_cast<size_t>(i) * nC * nSpatial + static_cast<size_t>(c) * nSpatial + px];
                                if (fVal > fBestVal) {
                                    fBestVal = fVal;
                                    nBestClass = c;
                                }
                            }
                            // 20260328 ZJH 取真实标签并比较
                            int nMaskOffset = nIdx * nSpatial + px;
                            if (nMaskOffset < static_cast<int>(vecValMasks.size())) {
                                int nTrueClass = vecValMasks[nMaskOffset];
                                if (nBestClass == nTrueClass) ++nCorrectPixels;
                            }
                            ++nTotalPixels;
                        }

                        // 20260328 ZJH 计算该样本的 Dice 系数（按类别平均）
                        // 逐类统计 intersection 和 union
                        float fSampleDice = 0.0f;
                        int nValidClasses = 0;
                        for (int c = 0; c < nC; ++c) {
                            float fInter = 0.0f, fPredC = 0.0f, fTrueC = 0.0f;
                            for (int px = 0; px < nSpatial; ++px) {
                                // 20260328 ZJH 预测: softmax argmax == c
                                int nPredClass = 0;
                                float fBV = pValOut[static_cast<size_t>(i) * nC * nSpatial + px];
                                for (int cc = 1; cc < nC; ++cc) {
                                    float fV = pValOut[static_cast<size_t>(i) * nC * nSpatial + static_cast<size_t>(cc) * nSpatial + px];
                                    if (fV > fBV) { fBV = fV; nPredClass = cc; }
                                }
                                // 20260328 ZJH 真实标签
                                int nMO = nIdx * nSpatial + px;
                                int nTC = (nMO < static_cast<int>(vecValMasks.size())) ? vecValMasks[nMO] : 0;
                                // 20260328 ZJH 累计
                                if (nPredClass == c) fPredC += 1.0f;
                                if (nTC == c) fTrueC += 1.0f;
                                if (nPredClass == c && nTC == c) fInter += 1.0f;
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

        if (fValLoss < fBestValLoss) { fBestValLoss = fValLoss; nPatienceCounter = 0; }
        else ++nPatienceCounter;

        if (epochCb) {
            BridgeEpochResult r;
            r.nEpoch = nEpoch; r.nTotalEpochs = nEpochs;
            r.fTrainLoss = fAvgTrainLoss; r.fValLoss = fValLoss; r.fMetric = fValAcc;
            epochCb(r);
        }

        if (logCb) {
            // 20260328 ZJH 分割模型显示 PixelAcc（像素准确率），分类模型显示 Acc（图像准确率）
            std::string strAccLabel = (m_pImpl->bIsSegmentation && !vecValMasks.empty()) ? " PixelAcc:" : " Acc:";
            logCb("Epoch " + std::to_string(nEpoch) + "/" + std::to_string(nEpochs) +
                  " Train:" + std::to_string(fAvgTrainLoss) +
                  " Val:" + std::to_string(fValLoss) +
                  strAccLabel + std::to_string(fValAcc) +
                  " LR:" + std::to_string(fCurrentLr) +
                  " Pat:" + std::to_string(nPatienceCounter) + "/" + std::to_string(params.nPatience));
        }

        if (nPatienceCounter >= params.nPatience) {
            if (logCb) logCb("[INFO] Early stopping at epoch " + std::to_string(nEpoch));
            break;
        }
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

                // 20260328 ZJH 计算该像素的缺陷类 sigmoid 概率（与训练一致）
                for (int c = 1; c < nOutC; ++c) {
                    float fLogit = pO[c * nSpatial + s];
                    float fSig = 1.0f / (1.0f + std::exp(-fLogit));  // 20260328 ZJH sigmoid
                    if (fSig > fMaxDefectProb) fMaxDefectProb = fSig;
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

    // 20260328 ZJH ===== 空间缺陷图：仅对 4D 分割输出生成 =====
    // 分割模型输出 [B, C, H, W]：生成逐像素缺陷概率图
    // 每个像素的值 = P(defect)，0=正常，1=缺陷
    if (m_pImpl->bIsCnn && cOut.shapeVec().size() == 4 && cOut.shape(1) >= 2) {
        int nOutC = cOut.shape(1);       // 20260328 ZJH 类别通道数
        int nOutH = cOut.shape(2);       // 20260328 ZJH 输出空间高度
        int nOutW = cOut.shape(3);       // 20260328 ZJH 输出空间宽度
        int nSpatial = nOutH * nOutW;    // 20260328 ZJH 空间像素数

        // 20260328 ZJH 逐像素计算缺陷概率 — 使用 sigmoid（与训练一致）
        // 训练用 sigmoid + Dice Loss，推理也必须用 sigmoid 保持一致
        // 取所有非背景通道(class>=1) sigmoid 输出的最大值作为缺陷概率
        std::vector<float> vecDefectProb(static_cast<size_t>(nSpatial), 0.0f);
        for (int s = 0; s < nSpatial; ++s) {
            float fMaxSigmoid = 0.0f;  // 20260328 ZJH 最大缺陷类 sigmoid 概率
            for (int c = 1; c < nOutC; ++c) {
                // 20260328 ZJH sigmoid(logit) = 1 / (1 + exp(-logit))
                float fLogit = pO[c * nSpatial + s];
                float fSig = 1.0f / (1.0f + std::exp(-fLogit));
                if (fSig > fMaxSigmoid) fMaxSigmoid = fSig;
            }
            vecDefectProb[static_cast<size_t>(s)] = fMaxSigmoid;
        }

        result.vecAnomalyMap = std::move(vecDefectProb);
        result.nMapW = nOutW;
        result.nMapH = nOutH;
    }
    return result;
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
        om::ModelSerializer::save(*m_pImpl->pModel, strPath);  // 20260328 ZJH 序列化模型权重到磁盘
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

        om::ModelSerializer::load(*m_pImpl->pModel, strPath);  // 20260325 ZJH 从磁盘反序列化模型权重
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
    for (const auto* p : m_pImpl->pModel->parameters()) n += p->numel();
    return n;
}

int64_t EngineBridge::trainableParameters() const { return totalParameters(); }
bool EngineBridge::hasModel() const { return m_pImpl->pModel != nullptr; }

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
