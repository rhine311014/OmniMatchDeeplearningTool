// 20260323 ZJH EngineBridge — OmniMatch 引擎桥接层
// 将 C++23 模块化引擎 (om::Module/Tensor/Optimizer) 暴露为传统 C++ 接口
// 供 Qt6 应用层 (TrainingSession/EvaluationWorker) 调用
// 桥接方式: .cpp 中 import om.engine.*, 通过不透明指针和回调传递数据
#pragma once

#include <string>       // 20260323 ZJH std::string
#include <vector>       // 20260323 ZJH std::vector
#include <functional>   // 20260323 ZJH std::function 回调
#include <memory>       // 20260323 ZJH unique_ptr

// 20260330 ZJH NMS 工具（DetectionBox 结构、yoloDecodeAndNms 等）
#include "core/NmsUtils.h"

// 20260323 ZJH 前向声明引擎内部实现（PIMPL 模式）
struct EngineSessionImpl;

// 20260323 ZJH 训练超参数（桥接结构体，不依赖 Qt 或 om 命名空间）
struct BridgeTrainParams
{
    std::string strModelType;     // 20260323 ZJH 模型类型名称 (如 "ResNet18", "MLP")
    int nInputSize = 224;         // 20260323 ZJH 输入图像尺寸
    int nNumClasses = 10;         // 20260323 ZJH 类别数量
    int nEpochs = 50;             // 20260323 ZJH 训练轮次
    int nBatchSize = 32;          // 20260323 ZJH 批量大小
    float fLearningRate = 0.001f; // 20260323 ZJH 学习率
    float fMomentum = 0.9f;       // 20260323 ZJH SGD 动量
    std::string strOptimizer;     // 20260323 ZJH 优化器名称 ("Adam"/"SGD"/"AdamW")
    int nPatience = 10;           // 20260323 ZJH 早停耐心
    bool bUseCuda = false;        // 20260323 ZJH 是否使用 CUDA
    // 20260330 ZJH 预训练权重路径（迁移学习: 加载已有权重，跳过形状不匹配的层）
    std::string strPretrainedModelPath;
    // 20260330 ZJH 是否启用训练数据增强（水平翻转/颜色抖动/高斯噪声等）
    bool bAugmentEnabled = true;
    // 20260331 ZJH 迁移学习: 骨干冻结轮数（前 N epoch 只训练 head，不更新编码器权重）
    int nFreezeEpochs = 0;
    // 20260331 ZJH 迁移学习: 骨干学习率倍率（解冻后骨干 LR = baseLR × 此值，典型 0.1）
    float fBackboneLrMultiplier = 0.1f;
    // 20260401 ZJH 梯度累积步数（1=不累积, >1=每 N 个 mini-batch 累积梯度后再 step）
    // 等效 batch_size = nBatchSize × nGradAccumSteps，用于小显存模拟大 batch 训练
    int nGradAccumSteps = 1;
    // 20260401 ZJH EMA 权重平均衰减系数（0=关闭, 典型 0.999~0.9999）
    // 推理时用 EMA 权重替代原始权重，平滑权重更新，通常提升 1~2% 精度
    float fEmaDecay = 0.0f;
    // 20260402 ZJH 是否使用 GroupNorm（由 Auto 策略决定）
    // BatchNorm 依赖 batch 统计量，小 batch 时不稳定；GroupNorm 按 channel 分组，不受 batch 影响
    bool bUseGroupNorm = false;
    // 20260402 ZJH GroupNorm 分组数（默认 32，需整除通道数）
    int nGroupNormGroups = 32;
    // 20260402 ZJH 混合精度训练（FP16 前向 + FP32 反向 + GradScaler 梯度缩放）
    // 减少 ~40% 显存占用，加速 ~1.5x（Tensor Core 硬件），默认关闭
    bool bMixedPrecision = false;
    // 20260402 ZJH 确定性训练（固定所有随机种子 + cuDNN 确定性算法）
    // 保证同样的数据+参数→相同的训练结果，用于回归测试和调试
    // 代价: 训练速度 ~10-20% 降低（cuDNN 不使用最快但非确定性算法）
    bool bDeterministic = false;
    // 20260402 ZJH 确定性训练随机种子（bDeterministic=true 时生效）
    int nRandomSeed = 42;
    // 20260402 ZJH 分割训练是否使用 BoundaryLoss（CE+Dice+Boundary 三项组合损失）
    // 默认启用，显著提升边界精度（Hausdorff -10~30%），无距离图时自动退化为 CE+Dice
    bool bUseBoundaryLoss = true;
    // 20260402 ZJH [OPT-3.1] 渐进式分辨率训练（Progressive Resizing）
    // 前 30% epoch 用 50% 分辨率，中 40% 用 75%，后 30% 用 100%
    // 低分辨率阶段 batch 可开更大（4x），FLOP 减少 40-60%
    // 效果: 先学粗粒度特征，再精调细节，对小数据集尤其有效
    bool bProgressiveResize = false;
    // 20260402 ZJH 后训练模型剪枝（训练完成后自动执行）
    // 减少模型体积和推理延迟，精度损失 <1%（需要微调恢复）
    bool bPruneAfterTraining = false;
    // 20260402 ZJH 剪枝率（0.1~0.8，默认 0.3 = 裁剪 30% 参数）
    float fPruneRatio = 0.3f;
    // 20260402 ZJH 半监督训练（FixMatch: 少量标注 + 大量无标注混合训练）
    // 来源: Semi-Supervised Seg (Nature 2024) — 比 SOTA 高 3-4% mIoU，标注量减少 80%
    bool bSemiSupervised = false;
    // 20260402 ZJH 半监督伪标签置信度阈值（高于此值的无标注样本才参与训练）
    float fPseudoLabelThreshold = 0.95f;
    // 20260402 ZJH [OPT-2.3] 高级增强总开关（CutPaste/MixUp/CutMix/Mosaic/ElasticDeform）
    // 默认启用，由 buildTrainAugmentConfig() 根据模型类型自动选择具体增强策略
    bool bAdvancedAugment = true;
    // 20260402 ZJH [OPT-2.5] Continual Learning 增量训练
    // 在已有模型基础上增量训练，使用 EWC 正则化防止灾难性遗忘
    bool bContinualLearning = false;
    // 20260402 ZJH EWC 正则化系数（λ 越大，对旧任务参数的保护越强）
    // 典型值 100~10000，工业场景推荐 1000
    float fEwcLambda = 1000.0f;
    // 20260402 ZJH [OPT-3.8] AutoML 智能训练模式
    // 勾选后隐藏所有超参数，自动选择模型+LR+batch+epoch
    // 分类: <100 张→ResNet18+强增强, 100-1000 张→ResNet18 预训练, >1000→ResNet50/ViT
    // 异常检测: <50 正常→EfficientAD, >50→PatchCore
    // 分割: 轻量→MobileSegNet, 标准→UNet(base=16), 高精度→DeepLabV3+
    // 检测: 边缘→YOLOv5Nano, 标准→YOLOv8Nano
    bool bSmartMode = false;
};

// 20260323 ZJH 单 Epoch 训练结果
struct BridgeEpochResult
{
    int nEpoch = 0;           // 20260323 ZJH 当前 Epoch（1-based）
    int nTotalEpochs = 0;     // 20260323 ZJH 总 Epoch 数
    float fTrainLoss = 0.0f;  // 20260323 ZJH 训练损失
    float fValLoss = 0.0f;    // 20260323 ZJH 验证损失
    float fMetric = 0.0f;     // 20260323 ZJH 主评估指标 (Accuracy/mAP 等)
};

// 20260323 ZJH 推理结果
struct BridgeInferResult
{
    int nPredictedClass = -1;     // 20260323 ZJH 预测类别 ID
    float fConfidence = 0.0f;     // 20260323 ZJH 置信度
    std::vector<float> vecProbs;  // 20260323 ZJH 各类别概率
    // 20260325 ZJH 异常/缺陷热力图 — 与输入图像同尺寸的单通道浮点图 [0,1]
    // 值越大表示越可能是异物/缺陷，空表示模型不支持空间预测
    std::vector<float> vecAnomalyMap;
    int nMapW = 0;  // 20260325 ZJH 热力图宽度
    int nMapH = 0;  // 20260325 ZJH 热力图高度
    // 20260401 ZJH 逐像素 argmax 类别图（分割模型专用）
    // 像素值 = 类别 ID（0=背景, 1+=缺陷类），与 nMapW×nMapH 同尺寸
    // 用于 mask overlay 的真实类别着色（区分异物/划痕/脏污不同颜色）
    std::vector<uint8_t> vecArgmaxMap;
    // 20260330 ZJH YOLO 检测结果（目标检测模型推理后的 NMS 去重检测框列表）
    std::vector<om::DetectionBox> vecDetections;
};

// 20260330 ZJH 批量推理参数（对标 HikRobot SetBatchSize(1-32) 和 MVTec 批量优化）
// 单次前向传播处理多张图像，GPU 吞吐量提升 2-8x（取决于模型和显存）
struct BridgeInferParams {
    int nBatchSize = 1;          // 20260330 ZJH 批量大小 [1, 32]，对标 HikRobot
    float fConfThreshold = 0.5f; // 20260330 ZJH 置信度阈值（分类/检测共用）
    float fNmsThreshold = 0.45f; // 20260330 ZJH NMS IoU 阈值（仅检测模型使用）
};

// 20260330 ZJH AI 数据合成参数（桥接结构体，对应 om::DataSynthesisPipeline）
// 三种策略: CopyPaste缺陷粘贴 / 几何+光度增强 / GAN生成 / 自动选择
struct BridgeSynthesisParams {
    int nStrategy = 3;              // 20260330 ZJH 合成策略 (0=CopyPaste, 1=Augment, 2=GAN, 3=Auto)
    int nTargetCount = 500;         // 20260330 ZJH 目标合成数量
    float fScaleMin = 0.7f;         // 20260330 ZJH CopyPaste: 缺陷缩放下限
    float fScaleMax = 1.3f;         // 20260330 ZJH CopyPaste: 缺陷缩放上限
    float fRotateRange = 30.0f;     // 20260330 ZJH CopyPaste: 旋转角度范围（±度）
    float fBlendAlpha = 0.9f;       // 20260330 ZJH CopyPaste: 融合透明度
    bool bPoisson = true;           // 20260330 ZJH CopyPaste: 泊松融合开关
    int nVariantsPerImage = 20;     // 20260330 ZJH Augment: 每张图像变体数
    bool bElastic = true;           // 20260330 ZJH Augment: 弹性变形开关
    bool bPerspective = true;       // 20260330 ZJH Augment: 透视变换开关
    int nGanEpochs = 200;           // 20260330 ZJH GAN: 训练轮数
};

// 20260330 ZJH AI 数据合成结果（桥接结构体）
struct BridgeSynthesisResult {
    std::vector<std::vector<float>> vecImages;  // 20260330 ZJH 合成图像列表 [C*H*W]
    int nOrigCount = 0;     // 20260330 ZJH 原始样本数
    int nSynthCount = 0;    // 20260330 ZJH 合成生成数
};

// 20260323 ZJH 回调函数类型
using EpochCallback = std::function<void(const BridgeEpochResult&)>;
using BatchCallback = std::function<void(int nBatch, int nTotalBatches)>;
using LogCallback = std::function<void(const std::string&)>;
using StopChecker = std::function<bool()>;

// 20260323 ZJH EngineBridge — 引擎桥接接口
// 封装 om::Module、om::Adam/SGD、om::CrossEntropyLoss 的完整训练/推理流程
// Qt 层通过此接口与引擎交互，无需直接 import C++23 模块
class EngineBridge
{
public:
    // 20260323 ZJH 构造函数
    EngineBridge();

    // 20260323 ZJH 析构函数
    ~EngineBridge();

    // 20260323 ZJH 不可拷贝
    EngineBridge(const EngineBridge&) = delete;
    EngineBridge& operator=(const EngineBridge&) = delete;

    // 20260323 ZJH 创建模型
    // 根据 strModelType 创建对应的 om::Module (ResNet18/MLP/VGG 等)
    // 返回: true 表示成功
    bool createModel(const std::string& strModelType, int nInputSize, int nNumClasses);

    // 20260323 ZJH 执行训练
    // 参数: params - 训练超参数
    //       vecTrainData - 训练数据 [N, C*H*W] 展平的浮点向量
    //       vecTrainLabels - 训练标签 [N] 类别 ID
    //       vecValData - 验证数据（格式同上）
    //       vecValLabels - 验证标签
    //       epochCb - 每 Epoch 完成回调
    //       batchCb - 每 Batch 完成回调
    //       logCb - 日志回调
    //       stopCheck - 外部停止检查（返回 true 则中断训练）
    // 返回: true 表示训练正常完成
    // 20260328 ZJH 新增 vecTrainMasks/vecValMasks: 语义分割模型的逐像素标注
    // vecTrainMasks: [N_train * H * W] 每个像素的类别 ID (0=背景)
    // vecValMasks:   [N_val * H * W]   验证集像素标注
    // 对非分割模型可传空向量，行为不变
    bool train(
        const BridgeTrainParams& params,
        const std::vector<float>& vecTrainData,
        const std::vector<int>& vecTrainLabels,
        const std::vector<float>& vecValData,
        const std::vector<int>& vecValLabels,
        const std::vector<int>& vecTrainMasks = {},
        const std::vector<int>& vecValMasks = {},
        EpochCallback epochCb = nullptr,
        BatchCallback batchCb = nullptr,
        LogCallback logCb = nullptr,
        StopChecker stopCheck = nullptr);

    // 20260323 ZJH 单张图像推理
    // 参数: vecImageData - 图像数据 [C*H*W] 展平的浮点向量
    // 返回: 推理结果
    BridgeInferResult infer(const std::vector<float>& vecImageData);

    // 20260330 ZJH 批量��理 — 单次前向传播处理多张图像（对标 HikRobot SetBatchSize）
    // 将多张图像打包为 [B, C, H, W] 批量张量，一次 forward 完成推理
    // GPU 场景下吞吐量可提升 2-8x（batch=1 → batch=N 减少 kernel launch 开销）
    // 参数: vecImages - 每张图像的展平浮点数据 [N][C*H*W]
    //       nC - 通道数（通常为 3）
    //       nH - 图像高度
    //       nW - 图像宽度
    //       params - 批量推理参数（batchSize/confThreshold/nmsThreshold）
    // 返回: 每张图像的推理结果向量（与输入顺序一一对应）
    std::vector<BridgeInferResult> inferBatch(
        const std::vector<std::vector<float>>& vecImages,
        int nC, int nH, int nW,
        const BridgeInferParams& params = {});

    // 20260323 ZJH 保存模型权重到文件
    bool saveModel(const std::string& strPath);

    // 20260323 ZJH 加载模型权重从文件
    bool loadModel(const std::string& strPath);

    // 20260323 ZJH 获取模型参数总数
    int64_t totalParameters() const;

    // 20260323 ZJH 获取可训练参数总数
    int64_t trainableParameters() const;

    // 20260323 ZJH 检查模型是否已创建
    bool hasModel() const;

    // 20260326 ZJH 强制释放 GPU 内存（将模型参数移回 CPU 并清理 CUDA 资源）
    void releaseGpu();

    // 20260324 ZJH 自动选择最大 batch size（根据可用内存和模型大小）
    // 参数: nInputDim - 单样本展平维度
    //       nNumClasses - 类别数
    //       nModelParams - 模型参数总数
    // 返回: 推荐 batch size（2 的幂，范围 [1, 512]）
    static int autoSelectBatchSize(int nInputDim, int nNumClasses, int64_t nModelParams);

    // 20260330 ZJH AI 数据合成 — 从少量缺陷样本自动生成大量合成训练数据
    // 通过 om::DataSynthesisPipeline 调用引擎层合成模块
    // 参数: vecNormalImages - 正常（无缺陷）图像列表，每张 [C*H*W] 展平 float [0,1]
    //       vecDefectImages - 缺陷图像列表，每张 [C*H*W] 展平 float [0,1]
    //       nC, nH, nW - 通道数、高度、宽度
    //       params - 合成参数（策略/数量/缩放/旋转/融合等）
    // 返回: 合成结果（图像列表 + 统计信息）
    BridgeSynthesisResult synthesizeData(
        const std::vector<std::vector<float>>& vecNormalImages,
        const std::vector<std::vector<float>>& vecDefectImages,
        int nC, int nH, int nW,
        const BridgeSynthesisParams& params);

    // ===================================================================
    // 20260402 ZJH 新增接口 — 对标 Halcon/ViDi 差距补全
    // ===================================================================

    // 20260402 ZJH GCAD 推理 — 全局上下文异常检测（布局+纹理双分支）
    // 返回 GCADResult: 全局分数 + 局部分数 + 融合分数 + 热力图 + 布局异常标志
    struct GCADInferResult {
        float fGlobalScore;        // 20260402 ZJH 全局异常分数（马氏距离）
        float fLocalScore;         // 20260402 ZJH 局部异常分数（最大像素异常值）
        float fFusedScore;         // 20260402 ZJH 融合异常分数
        bool bIsAnomaly;           // 20260402 ZJH 是否为异常
        bool bIsLayoutAnomaly;     // 20260402 ZJH 是否为布局异常（GCAD 独有）
        std::vector<float> vecAnomalyMap;  // 20260402 ZJH 热力图
        int nMapH, nMapW;          // 20260402 ZJH 热力图尺寸
    };
    GCADInferResult inferGCAD(const std::vector<float>& vecImageData, int nC, int nH, int nW);

    // 20260402 ZJH GCAD 训练后拟合全局分布 — 收集正常样本特征
    bool fitGCADDistribution(const std::vector<std::vector<float>>& vecNormalImages,
                              int nC, int nH, int nW);

    // 20260402 ZJH Continual Learning（增量学习）— 对标 Halcon 25.11
    // 在已有模型基础上增量训练新类别，不遗忘旧类别（EWC 正则化）
    // vecOldData/Labels: 旧任务少量样本（用于计算 Fisher 信息矩阵）
    // vecNewData/Labels: 新类别训练数据
    bool trainContinual(const std::vector<std::vector<float>>& vecOldData,
                        const std::vector<std::vector<float>>& vecOldLabels,
                        const std::vector<std::vector<float>>& vecNewData,
                        const std::vector<std::vector<float>>& vecNewLabels,
                        int nC, int nH, int nW, int nEpochs, float fLR,
                        float fEwcLambda = 1000.0f);

    // 20260402 ZJH TTA 推理（Test-Time Augmentation）— 通用精度提升
    // 对输入图像进行多种增强变换，分别推理后融合
    // 返回: 增强后的推理结果（分类/分割通用）
    BridgeInferResult inferWithTTA(const std::vector<float>& vecImageData,
                                    int nC, int nH, int nW,
                                    bool bHFlip = true, bool bVFlip = false,
                                    bool bRotate90 = false, bool bMultiScale = false);

    // ===================================================================
    // 20260402 ZJH 推理 Benchmark + 精度基线系统
    // 对标 Halcon 内部回归测试 + 工业级推理速度要求
    // ===================================================================

    // 20260402 ZJH BenchmarkResult — 推理性能基准测试结果
    struct BenchmarkResult {
        double dMedianMs;      // 20260402 ZJH 中位数延迟 (ms)
        double dP95Ms;         // 20260402 ZJH P95 延迟 (ms)
        double dP99Ms;         // 20260402 ZJH P99 延迟 (ms)
        double dMinMs;         // 20260402 ZJH 最小延迟 (ms)
        double dMaxMs;         // 20260402 ZJH 最大延迟 (ms)
        double dThroughputFPS; // 20260402 ZJH 吞吐量 (帧/秒)
        int nWarmupRuns;       // 20260402 ZJH 预热轮数
        int nBenchmarkRuns;    // 20260402 ZJH 测试轮数
    };

    // 20260402 ZJH benchmarkInference — 推理性能基准测试
    // 输入: 测试图像数据 + 重复次数
    // 输出: 延迟统计（median/p95/p99/min/max）+ 吞吐量 FPS
    // 用途: (1) OCR 推理优化验证 <10ms/帧目标
    //       (2) TensorRT vs NativeCpp 速度对比
    //       (3) 版本更新后推理速度回归检测
    BenchmarkResult benchmarkInference(const std::vector<float>& vecImageData,
                                        int nC, int nH, int nW,
                                        int nWarmupRuns = 5, int nBenchmarkRuns = 50);

    // 20260402 ZJH AccuracyBaseline — 精度基线记录
    struct AccuracyBaseline {
        std::string strModelType;   // 20260402 ZJH 模型类型
        std::string strDatasetName; // 20260402 ZJH 数据集标识
        int nTrainSamples;          // 20260402 ZJH 训练样本数
        int nEpochs;                // 20260402 ZJH 训练轮次
        float fFinalLoss;           // 20260402 ZJH 最终训练损失
        float fValAccuracy;         // 20260402 ZJH 验证精度
        float fValF1;               // 20260402 ZJH 验证 F1
        float fInferenceMs;         // 20260402 ZJH 推理延迟 (ms)
        std::string strTimestamp;   // 20260402 ZJH 记录时间
    };

    // 20260402 ZJH saveAccuracyBaseline — 保存精度基线到 JSON 文件
    // 训练完成后调用，记录当前精度作为后续版本的回归基准
    static bool saveAccuracyBaseline(const AccuracyBaseline& baseline,
                                      const std::string& strBaselineDir);

    // 20260402 ZJH loadAccuracyBaseline — 加载精度基线
    static bool loadAccuracyBaseline(const std::string& strBaselineDir,
                                      const std::string& strModelType,
                                      AccuracyBaseline& outBaseline);

    // 20260402 ZJH checkAccuracyRegression — 对比当前精度与基线
    // 返回: true = 精度未下降（允许 ±fTolerance 波动）
    //       false = 精度回归（当前精度低于基线 - fTolerance）
    static bool checkAccuracyRegression(const AccuracyBaseline& current,
                                         const AccuracyBaseline& baseline,
                                         float fTolerance = 0.02f);

    // ===================================================================
    // 20260402 ZJH AI 缺陷生成器 — 内置缺陷合成（解决残次品不足问题）
    // 自动选择路线: <5 张缺陷用 DRAEM+, ≥5 张用 DDPM Tiny
    // ===================================================================
    struct DefectGenConfig {
        int nTargetCount = 100;        // 20260402 ZJH 目标生成数量
        int nImageWidth = 224;         // 20260402 ZJH 图像宽度
        int nImageHeight = 224;        // 20260402 ZJH 图像高度
        int nDDPMTrainEpochs = 20;    // 20260402 ZJH DDPM 训练轮次
    };

    struct DefectGenResult {
        std::vector<std::vector<float>> vecImages;  // 20260402 ZJH 生成图 [C*H*W]
        std::vector<std::vector<float>> vecMasks;   // 20260402 ZJH 缺陷 mask [H*W]
        int nGeneratedCount = 0;
        int nMode = 0;  // 20260402 ZJH 0=DRAEM+, 1=DDPM
        std::string strLog;
    };

    // 20260402 ZJH 生成缺陷图（自动选择 DRAEM+ 或 DDPM Tiny）
    static DefectGenResult generateDefects(
        const std::vector<std::vector<float>>& vecNormalImages,
        const std::vector<std::vector<float>>& vecDefectImages,
        const DefectGenConfig& config);

private:
    // 20260323 ZJH PIMPL 实现指针（隐藏 C++23 模块依赖）
    std::unique_ptr<EngineSessionImpl> m_pImpl;
};
