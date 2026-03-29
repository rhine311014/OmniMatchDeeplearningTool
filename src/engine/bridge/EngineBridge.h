// 20260323 ZJH EngineBridge — OmniMatch 引擎桥接层
// 将 C++23 模块化引擎 (om::Module/Tensor/Optimizer) 暴露为传统 C++ 接口
// 供 Qt6 应用层 (TrainingSession/EvaluationWorker) 调用
// 桥接方式: .cpp 中 import om.engine.*, 通过不透明指针和回调传递数据
#pragma once

#include <string>       // 20260323 ZJH std::string
#include <vector>       // 20260323 ZJH std::vector
#include <functional>   // 20260323 ZJH std::function 回调
#include <memory>       // 20260323 ZJH unique_ptr

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

private:
    // 20260323 ZJH PIMPL 实现指针（隐藏 C++23 模块依赖）
    std::unique_ptr<EngineSessionImpl> m_pImpl;
};
