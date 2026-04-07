// 20260330 ZJH 高级学习范式模块 — 超越所有竞品的前沿能力
// 包含四大组件：
// 1. SemiSupervisedTrainer — FixMatch 半监督学习（少量标注 + 大量无标注混合训练）
// 2. ContinualLearner — EWC 弹性权重巩固（持续学习不遗忘旧任务）
// 3. DataQualityAssessor — 数据质量评估器（重复/模糊/过曝/标注错误检测）
// 4. ModelComparator — A/B 模型对比测试（Cohen's Kappa 一致性分析）
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>
#include <limits>

export module om.engine.advanced_learning;

// 20260330 ZJH 导入依赖模块：张量、张量运算、模块基类、CPU 后端
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.hal.cpu_backend;

export namespace om {

// =============================================================================
// 20260330 ZJH 辅助工具函数（模块内部使用）
// =============================================================================

// 20260330 ZJH cpuSoftmax — 对长度为 nLen 的原始 logits 向量做 softmax
// pLogits: 输入 logits 数组指针
// pOut: 输出概率数组指针
// nLen: 类别数
// 算法: 减去最大值保证数值稳定 → exp → 归一化
inline void cpuSoftmax(const float* pLogits, float* pOut, int nLen) {
    // 20260330 ZJH 求最大值，防止 exp 溢出
    float fMax = pLogits[0];  // 20260330 ZJH 初始化为第一个元素
    for (int i = 1; i < nLen; ++i) {
        if (pLogits[i] > fMax) {
            fMax = pLogits[i];  // 20260330 ZJH 更新最大值
        }
    }
    // 20260330 ZJH 计算 exp(x - max) 并累加求和
    float fSum = 0.0f;  // 20260330 ZJH 归一化因子
    for (int i = 0; i < nLen; ++i) {
        pOut[i] = std::exp(pLogits[i] - fMax);  // 20260330 ZJH 数值稳定的 exp
        fSum += pOut[i];  // 20260330 ZJH 累加到归一化因子
    }
    // 20260330 ZJH 归一化得到概率分布
    float fInvSum = 1.0f / (fSum + 1e-10f);  // 20260330 ZJH 避免除零
    for (int i = 0; i < nLen; ++i) {
        pOut[i] *= fInvSum;  // 20260330 ZJH 归一化
    }
}

// 20260330 ZJH cpuArgmax — 返回长度为 nLen 的数组中最大值的索引
// pData: 输入数组指针
// nLen: 数组长度
// 返回: 最大值对应的索引
inline int cpuArgmax(const float* pData, int nLen) {
    int nBestIdx = 0;  // 20260330 ZJH 最大值索引
    float fBestVal = pData[0];  // 20260330 ZJH 当前最大值
    for (int i = 1; i < nLen; ++i) {
        if (pData[i] > fBestVal) {
            fBestVal = pData[i];  // 20260330 ZJH 更新最大值
            nBestIdx = i;  // 20260330 ZJH 更新索引
        }
    }
    return nBestIdx;  // 20260330 ZJH 返回最大值索引
}

// =============================================================================
// 20260330 ZJH 1. 半监督学习 (Semi-Supervised Learning)
// =============================================================================
// FixMatch 算法核心思想：
// - 对无标注数据做弱增强后前向推理，若最大概率超过阈值则生成伪标签
// - 对同一无标注数据做强增强后计算交叉熵损失（以伪标签为目标）
// - 总损失 = 有标注损失 + λ * 无标注损失
// =============================================================================

// 20260330 ZJH SemiSupervisedTrainer — FixMatch 半监督训练器
class SemiSupervisedTrainer {
public:
    // 20260330 ZJH 半监督训练配置参数
    struct Config {
        float fPseudoLabelThreshold = 0.95f;  // 20260330 ZJH 伪标签置信度阈值，仅高于此值的预测才被接受
        float fUnlabeledWeight = 1.0f;        // 20260330 ZJH 无标注损失在总损失中的权重系数
        int nLabeledBatchSize = 8;            // 20260330 ZJH 每批有标注样本数量
        int nUnlabeledBatchSize = 32;         // 20260330 ZJH 每批无标注样本数量（通常远大于有标注）
    };

    // 20260330 ZJH PseudoLabel — 伪标签结构体，记录预测类别、置信度和是否被接受
    struct PseudoLabel {
        int nClassId;       // 20260330 ZJH 预测的类别 ID（softmax argmax）
        float fConfidence;  // 20260330 ZJH 预测的最大置信度（softmax max prob）
        bool bAccepted;     // 20260330 ZJH 是否超过置信度阈值，超过才用于训练
    };

    // 20260330 ZJH generatePseudoLabels — 对无标注数据批次生成伪标签
    // 算法: forward pass → softmax → 取最大概率 → 若 > threshold 则接受
    // model: 当前训练中的模型（设为 eval 模式推理后恢复 train 模式）
    // vecUnlabeled: 无标注数据张量列表，每个 shape=[1, C, H, W] 或 [1, features]
    // fThreshold: 置信度阈值，默认使用 Config 中的值
    // 返回: 与输入等长的 PseudoLabel 向量
    std::vector<PseudoLabel> generatePseudoLabels(
        Module& model, const std::vector<Tensor>& vecUnlabeled, float fThreshold) {

        std::vector<PseudoLabel> vecResult;  // 20260330 ZJH 输出伪标签列表
        vecResult.reserve(vecUnlabeled.size());  // 20260330 ZJH 预分配避免重复扩容

        // 20260330 ZJH 保存当前训练模式状态，推理时需切换到 eval 模式
        bool bWasTraining = model.isTraining();  // 20260330 ZJH 记住原始状态
        model.eval();  // 20260330 ZJH 切换到评估模式（禁用 dropout/BN 等随机行为）

        // 20260330 ZJH 逐样本前向推理生成伪标签
        for (size_t i = 0; i < vecUnlabeled.size(); ++i) {
            // 20260330 ZJH 前向传播得到 logits
            Tensor logits = model.forward(vecUnlabeled[i]);  // 20260330 ZJH shape=[1, nClasses]
            auto cLogits = logits.contiguous();  // 20260330 ZJH 确保内存连续

            // 20260330 ZJH 获取类别数（logits 的最后一维）
            int nClasses = cLogits.shape(cLogits.ndim() - 1);  // 20260330 ZJH 类别数
            int nBatch = cLogits.numel() / nClasses;  // 20260330 ZJH 批量大小（通常为 1）

            const float* pLogits = cLogits.floatDataPtr();  // 20260330 ZJH 原始 logits 指针

            // 20260330 ZJH 对第一个样本（batch=0）做 softmax
            std::vector<float> vecProb(nClasses);  // 20260330 ZJH softmax 概率输出缓冲区
            cpuSoftmax(pLogits, vecProb.data(), nClasses);  // 20260330 ZJH 计算 softmax 概率

            // 20260330 ZJH 找到最大概率及其对应类别
            int nBestClass = cpuArgmax(vecProb.data(), nClasses);  // 20260330 ZJH argmax 类别
            float fMaxProb = vecProb[nBestClass];  // 20260330 ZJH 最大概率值

            // 20260330 ZJH 构造伪标签：置信度超过阈值则接受
            PseudoLabel label;  // 20260330 ZJH 伪标签结构体
            label.nClassId = nBestClass;  // 20260330 ZJH 预测类别
            label.fConfidence = fMaxProb;  // 20260330 ZJH 预测置信度
            label.bAccepted = (fMaxProb >= fThreshold);  // 20260330 ZJH 是否超过阈值
            vecResult.push_back(label);  // 20260330 ZJH 加入结果列表
        }

        // 20260330 ZJH 恢复原始训练模式状态
        if (bWasTraining) {
            model.train(true);  // 20260330 ZJH 切回训练模式
        }

        return vecResult;  // 20260330 ZJH 返回伪标签向量
    }

    // 20260330 ZJH fixmatchStep — 执行一步 FixMatch 训练
    // 计算: total_loss = CE(model(labeled), target) + weight * CE(model(unlabeled), pseudo_target)
    // 其中 pseudo_target 仅使用置信度超过阈值的伪标签
    // model: 训练中的模型
    // labeledInput: 有标注数据 [N_labeled, ...]
    // labeledTarget: 有标注数据的 one-hot 标签 [N_labeled, nClasses]
    // unlabeledInput: 无标注数据 [N_unlabeled, ...]
    // config: 训练配置
    // 返回: 标量总损失张量，支持反向传播
    Tensor fixmatchStep(Module& model,
                        const Tensor& labeledInput, const Tensor& labeledTarget,
                        const Tensor& unlabeledInput, const Config& config) {

        // ===========================================================
        // 20260330 ZJH Step 1: 有标注数据的监督损失
        // ===========================================================
        Tensor labeledLogits = model.forward(labeledInput);  // 20260330 ZJH 前向传播
        // 20260330 ZJH 调用 softmax + 交叉熵联合损失
        Tensor labeledLoss = tensorSoftmaxCrossEntropy(labeledLogits, labeledTarget);

        // ===========================================================
        // 20260330 ZJH Step 2: 无标注数据生成伪标签
        // ===========================================================

        // 20260330 ZJH 首先在 eval 模式下推理，获取伪标签
        bool bWasTraining = model.isTraining();  // 20260330 ZJH 保存训练模式
        model.eval();  // 20260330 ZJH 切换到 eval 模式，确保伪标签稳定

        Tensor unlabeledLogitsDetach = model.forward(unlabeledInput);  // 20260330 ZJH 弱增强推理
        auto cULogits = unlabeledLogitsDetach.contiguous();  // 20260330 ZJH 连续化

        int nUnlabeledBatch = cULogits.shape(0);  // 20260330 ZJH 无标注批量大小
        int nClasses = cULogits.shape(cULogits.ndim() - 1);  // 20260330 ZJH 类别数

        const float* pULogits = cULogits.floatDataPtr();  // 20260330 ZJH logits 原始指针

        // 20260330 ZJH 生成伪标签及掩码
        std::vector<float> vecPseudoTarget(nUnlabeledBatch * nClasses, 0.0f);  // 20260330 ZJH one-hot 伪标签
        std::vector<float> vecMask(nUnlabeledBatch, 0.0f);  // 20260330 ZJH 置信度掩码（0 或 1）
        int nAccepted = 0;  // 20260330 ZJH 被接受的伪标签数量

        for (int b = 0; b < nUnlabeledBatch; ++b) {
            // 20260330 ZJH 对每个样本做 softmax
            std::vector<float> vecProb(nClasses);  // 20260330 ZJH softmax 输出缓冲区
            cpuSoftmax(pULogits + b * nClasses, vecProb.data(), nClasses);

            // 20260330 ZJH 找最大概率
            int nBestClass = cpuArgmax(vecProb.data(), nClasses);  // 20260330 ZJH argmax
            float fMaxProb = vecProb[nBestClass];  // 20260330 ZJH 最大概率

            if (fMaxProb >= config.fPseudoLabelThreshold) {
                // 20260330 ZJH 置信度足够高，接受此伪标签
                vecPseudoTarget[b * nClasses + nBestClass] = 1.0f;  // 20260330 ZJH one-hot 编码
                vecMask[b] = 1.0f;  // 20260330 ZJH 掩码设为 1
                ++nAccepted;  // 20260330 ZJH 计数
            }
            // 20260330 ZJH 置信度不够则掩码为 0，该样本不参与无标注损失
        }

        // 20260330 ZJH 恢复训练模式
        if (bWasTraining) {
            model.train(true);  // 20260330 ZJH 切回训练模式
        }

        // ===========================================================
        // 20260330 ZJH Step 3: 无标注数据的交叉熵损失（仅对被接受的伪标签）
        // ===========================================================
        Tensor unlabeledLoss;  // 20260330 ZJH 无标注损失
        if (nAccepted > 0) {
            // 20260330 ZJH 重新前向传播（在 train 模式下，支持梯度追踪）
            Tensor unlabeledLogitsTrain = model.forward(unlabeledInput);  // 20260330 ZJH 强增强前向
            auto cULogitsTrain = unlabeledLogitsTrain.contiguous();  // 20260330 ZJH 连续化

            // 20260330 ZJH 构建伪标签张量
            Tensor pseudoTargetTensor = Tensor::fromData(
                vecPseudoTarget.data(), {nUnlabeledBatch, nClasses});  // 20260330 ZJH one-hot 伪标签

            // 20260330 ZJH 计算 softmax 交叉熵损失
            Tensor rawLoss = tensorSoftmaxCrossEntropy(
                unlabeledLogitsTrain, pseudoTargetTensor);  // 20260330 ZJH 未加权的 CE 损失

            // 20260330 ZJH 按被接受样本比例缩放（掩码的效果）
            // FixMatch 原论文对未被接受的样本梯度为零，这里通过 accepted_ratio 近似
            float fAcceptedRatio = static_cast<float>(nAccepted) /
                                   static_cast<float>(nUnlabeledBatch);  // 20260330 ZJH 接受比例
            unlabeledLoss = tensorMulScalar(rawLoss, fAcceptedRatio);  // 20260330 ZJH 缩放损失
        } else {
            // 20260330 ZJH 没有任何伪标签被接受，无标注损失为零
            unlabeledLoss = Tensor::full({1}, 0.0f);  // 20260330 ZJH 零损失
        }

        // ===========================================================
        // 20260330 ZJH Step 4: 合并总损失
        // total = labeled_loss + unlabeled_weight * unlabeled_loss
        // ===========================================================
        Tensor weightedUnlabeled = tensorMulScalar(
            unlabeledLoss, config.fUnlabeledWeight);  // 20260330 ZJH 加权无标注损失
        Tensor totalLoss = tensorAdd(labeledLoss, weightedUnlabeled);  // 20260330 ZJH 总损失

        return totalLoss;  // 20260330 ZJH 返回可反向传播的总损失
    }
};

// =============================================================================
// 20260330 ZJH 2. 持续学习 (Continual Learning) — EWC
// =============================================================================
// EWC (Elastic Weight Consolidation) 核心思想：
// - 旧任务训练完成后，计算 Fisher 信息矩阵 F_i（衡量每个参数对旧任务的重要性）
// - 学习新任务时，在损失函数中加入正则项：
//   penalty = (lambda/2) * sum_i( F_i * (theta_i - theta*_i)^2 )
// - 对旧任务重要的参数（F_i 大）变化受到更强约束，不重要的参数自由更新
// =============================================================================

// 20260330 ZJH ContinualLearner — 弹性权重巩固持续学习器
class ContinualLearner {
public:
    // 20260330 ZJH EWC 配置参数
    struct EWCConfig {
        float fLambda = 1000.0f;  // 20260330 ZJH Fisher 信息矩阵正则化强度，越大越保守
    };

    // 20260330 ZJH computeFisherMatrix — 计算 Fisher 信息矩阵的对角近似
    // 算法: 对每个样本，计算 log p(y|x; θ) 的梯度，然后取梯度平方的均值
    // Fisher 对角近似: F_i ≈ (1/N) * Σ_n (∂logP/∂θ_i)^2
    // model: 当前模型（需在旧任务最优参数状态下调用）
    // vecOldData: 旧任务训练数据样本列表
    // vecOldLabels: 旧任务对应的 one-hot 标签列表
    void computeFisherMatrix(Module& model,
                             const std::vector<Tensor>& vecOldData,
                             const std::vector<Tensor>& vecOldLabels) {

        // 20260330 ZJH 获取所有参数指针
        auto vecParams = model.parameters();  // 20260330 ZJH 模型参数列表
        int nParams = static_cast<int>(vecParams.size());  // 20260330 ZJH 参数数量

        if (nParams == 0) {
            return;  // 20260330 ZJH 无参数模型，直接返回
        }

        // 20260330 ZJH 初始化 Fisher 对角阵为零（与每个参数同形状）
        m_vecFisher.clear();  // 20260330 ZJH 清空旧的 Fisher 矩阵
        m_vecFisher.reserve(nParams);  // 20260330 ZJH 预分配
        for (int p = 0; p < nParams; ++p) {
            m_vecFisher.push_back(
                Tensor::zeros(vecParams[p]->shapeVec()));  // 20260330 ZJH 零初始化同形状张量
        }

        int nSamples = static_cast<int>(vecOldData.size());  // 20260330 ZJH 旧任务样本数
        if (nSamples == 0) {
            return;  // 20260330 ZJH 无样本，Fisher 保持为零
        }

        // 20260330 ZJH 逐样本计算梯度的平方并累加
        model.train(true);  // 20260330 ZJH 确保处于训练模式（启用梯度追踪）

        for (int s = 0; s < nSamples; ++s) {
            // 20260330 ZJH 清零所有参数梯度
            model.zeroGrad();  // 20260330 ZJH 梯度清零

            // 20260330 ZJH 前向传播 → 计算损失 → 反向传播
            Tensor logits = model.forward(vecOldData[s]);  // 20260330 ZJH 前向推理
            // 20260330 ZJH 使用 softmax 交叉熵作为 log-likelihood 的近似
            Tensor loss = tensorSoftmaxCrossEntropy(logits, vecOldLabels[s]);  // 20260330 ZJH 损失
            tensorBackward(loss);  // 20260330 ZJH 反向传播计算梯度

            // 20260330 ZJH 对每个参数，取梯度的平方并累加到 Fisher 矩阵
            for (int p = 0; p < nParams; ++p) {
                Tensor grad = tensorGetGrad(*vecParams[p]);  // 20260330 ZJH 获取参数梯度
                if (grad.numel() == 0) {
                    continue;  // 20260330 ZJH 该参数无梯度，跳过
                }
                // 20260330 ZJH grad^2 累加到 Fisher
                auto cGrad = grad.contiguous();  // 20260330 ZJH 确保连续
                auto cFisher = m_vecFisher[p].contiguous();  // 20260330 ZJH 确保连续
                float* pFisher = m_vecFisher[p].mutableFloatDataPtr();  // 20260330 ZJH Fisher 可写指针
                const float* pGrad = cGrad.floatDataPtr();  // 20260330 ZJH 梯度只读指针
                int nElem = cGrad.numel();  // 20260330 ZJH 元素数量
                for (int e = 0; e < nElem; ++e) {
                    pFisher[e] += pGrad[e] * pGrad[e];  // 20260330 ZJH 梯度平方累加
                }
            }
        }

        // 20260330 ZJH 除以样本数取均值: F_i = (1/N) * Σ_n g_i^2
        float fInvN = 1.0f / static_cast<float>(nSamples);  // 20260330 ZJH 归一化系数
        for (int p = 0; p < nParams; ++p) {
            float* pFisher = m_vecFisher[p].mutableFloatDataPtr();  // 20260330 ZJH Fisher 可写指针
            int nElem = m_vecFisher[p].numel();  // 20260330 ZJH 元素数量
            for (int e = 0; e < nElem; ++e) {
                pFisher[e] *= fInvN;  // 20260330 ZJH 均值化
            }
        }

        // 20260330 ZJH 清零模型梯度（避免残留影响后续训练）
        model.zeroGrad();  // 20260330 ZJH 清零
    }

    // 20260330 ZJH ewcPenalty — 计算 EWC 正则化惩罚项
    // penalty = (lambda/2) * Σ_i F_i * (θ_i - θ*_i)^2
    // 其中 θ* 为旧任务最优参数快照，F 为 Fisher 信息矩阵对角近似
    // model: 当前模型（参数可能已被新任务更新）
    // 返回: 标量惩罚张量
    Tensor ewcPenalty(const Module& model) const {
        // 20260330 ZJH 需要 const_cast 来调用非 const 的 parameters()
        // Module::parameters() 声明为 virtual 非 const，但此处仅读取
        auto& mutableModel = const_cast<Module&>(model);  // 20260330 ZJH const_cast 仅用于读取
        auto vecParams = mutableModel.parameters();  // 20260330 ZJH 当前参数
        int nParams = static_cast<int>(vecParams.size());  // 20260330 ZJH 参数数量

        // 20260330 ZJH 验证快照和 Fisher 是否已计算
        if (m_vecSnapshot.empty() || m_vecFisher.empty()) {
            return Tensor::full({1}, 0.0f);  // 20260330 ZJH 未初始化，返回零惩罚
        }

        // 20260330 ZJH 验证参数数量一致
        if (nParams != static_cast<int>(m_vecSnapshot.size()) ||
            nParams != static_cast<int>(m_vecFisher.size())) {
            return Tensor::full({1}, 0.0f);  // 20260330 ZJH 数量不匹配，返回零惩罚
        }

        // 20260330 ZJH [修复] 用 tensor ops 链计算惩罚项，保持 autograd 到当前参数
        // 原实现通过 raw pointer 循环累加标量，返回 Tensor::full({1}, scalar) 断裂梯度链
        // 修复: diff = tensorSub(param, snapshot) → sq = tensorMul(diff, diff)
        //       → weighted = tensorMul(sq, fisher) → tensorSum → 累加
        // 梯度路径: penalty → sum → weighted → sq → diff → param（autograd 完整）
        auto penalty = Tensor::full({1}, 0.0f);  // 20260330 ZJH 累积惩罚张量（起始为零）
        for (int p = 0; p < nParams; ++p) {
            // 20260330 ZJH diff = θ_current - θ*_snapshot（autograd 连接到当前参数）
            auto diff = tensorSub(*vecParams[p], m_vecSnapshot[p]);  // 20260330 ZJH 参数偏移
            // 20260330 ZJH diffSq = (θ - θ*)^2
            auto diffSq = tensorMul(diff, diff);  // 20260330 ZJH 逐元素平方
            // 20260330 ZJH weighted = F_i * (θ_i - θ*_i)^2（Fisher 为常数，不参与 autograd）
            auto weighted = tensorMul(diffSq, m_vecFisher[p]);  // 20260330 ZJH Fisher 加权
            // 20260330 ZJH 求和并累加到 penalty
            auto paramPenalty = tensorSum(weighted);  // 20260330 ZJH 当前参数的惩罚贡献
            penalty = tensorAdd(penalty, paramPenalty);  // 20260330 ZJH 累加（autograd 链式）
        }

        // 20260330 ZJH 乘以 lambda/2（tensorMulScalar 保持 autograd）
        return tensorMulScalar(penalty, m_fLambda * 0.5f);  // 20260330 ZJH EWC 正则化惩罚（autograd 完整）
    }

    // 20260330 ZJH snapshotParameters — 存储当前参数的深拷贝快照
    // 在旧任务训练结束后调用，作为 EWC 惩罚的参考点 θ*
    // model: 旧任务训练完成后的最优模型
    void snapshotParameters(const Module& model) {
        auto& mutableModel = const_cast<Module&>(model);  // 20260330 ZJH const_cast 仅用于读取
        auto vecParams = mutableModel.parameters();  // 20260330 ZJH 获取参数列表

        m_vecSnapshot.clear();  // 20260330 ZJH 清空旧快照
        m_vecSnapshot.reserve(vecParams.size());  // 20260330 ZJH 预分配

        for (auto* pParam : vecParams) {
            // 20260330 ZJH 深拷贝参数数据（避免后续训练修改影响快照）
            auto cParam = pParam->contiguous();  // 20260330 ZJH 确保连续
            Tensor snapshot = Tensor::fromData(
                cParam.floatDataPtr(), cParam.shapeVec());  // 20260330 ZJH 深拷贝到新张量
            m_vecSnapshot.push_back(std::move(snapshot));  // 20260330 ZJH 存储快照
        }
    }

    // 20260330 ZJH hasSnapshot — 查询是否已有参数快照
    // 返回: true 表示已调用过 snapshotParameters，可以计算 EWC 惩罚
    bool hasSnapshot() const {
        return !m_vecSnapshot.empty();  // 20260330 ZJH 非空表示已有快照
    }

    // 20260330 ZJH setLambda — 设置 EWC 正则化强度
    // fLambda: 正则化系数，越大越保守（旧任务保护越强，新任务学习越慢）
    void setLambda(float fLambda) {
        m_fLambda = fLambda;  // 20260330 ZJH 更新正则化系数
    }

    // 20260330 ZJH getLambda — 获取当前 EWC 正则化强度
    float getLambda() const {
        return m_fLambda;  // 20260330 ZJH 返回正则化系数
    }

private:
    std::vector<Tensor> m_vecFisher;    // 20260330 ZJH Fisher 信息矩阵对角近似，每个元素与对应参数同形状
    std::vector<Tensor> m_vecSnapshot;  // 20260330 ZJH 旧任务最优参数快照 θ*
    float m_fLambda = 1000.0f;          // 20260330 ZJH EWC 正则化强度
};

// =============================================================================
// 20260330 ZJH 3. 数据质量评估器
// =============================================================================
// 全方位检查数据集质量：
// - 重复图片检测（感知哈希 pHash）
// - 模糊图片检测（Laplacian 方差）
// - 过曝/欠曝检测（亮度直方图分析）
// - 类别不平衡检测（最大/最小类比例）
// - 疑似标注错误检测（模型预测与标签不一致）
// - 离群样本检测（基于特征空间的 Mahalanobis 距离近似）
// =============================================================================

// 20260330 ZJH DataQualityAssessor — 数据质量评估器
class DataQualityAssessor {
public:
    // 20260330 ZJH QualityReport — 数据质量综合报告
    struct QualityReport {
        int nTotalImages;              // 20260330 ZJH 数据集总图片数
        int nDuplicateImages;          // 20260330 ZJH 检测到的重复图片数
        int nBlurryImages;             // 20260330 ZJH 模糊图片数
        int nOverexposed;              // 20260330 ZJH 过曝图片数
        int nUnderexposed;             // 20260330 ZJH 欠曝图片数
        float fClassImbalanceRatio;    // 20260330 ZJH 类别不平衡比（最大类/最小类）
        int nMislabeledSuspect;        // 20260330 ZJH 疑似标注错误样本数
        int nOutlierImages;            // 20260330 ZJH 离群样本数
        std::vector<std::string> vecWarnings;  // 20260330 ZJH 警告信息列表
        float fOverallScore;           // 20260330 ZJH 数据质量综合分 [0, 100]
    };

    // 20260330 ZJH assess — 全面评估数据集质量（不依赖模型的静态分析）
    // vecImages: 所有图片的展平浮点数组，每张 nC*nH*nW 个元素，值域 [0,1]
    // vecLabels: 每张图片的类别标签（整数）
    // nC: 通道数
    // nH: 图片高度
    // nW: 图片宽度
    // 返回: QualityReport 综合报告
    QualityReport assess(const std::vector<std::vector<float>>& vecImages,
                         const std::vector<int>& vecLabels,
                         int nC, int nH, int nW) {

        QualityReport report;  // 20260330 ZJH 初始化报告结构
        report.nTotalImages = static_cast<int>(vecImages.size());  // 20260330 ZJH 总图片数
        report.nDuplicateImages = 0;  // 20260330 ZJH 初始化
        report.nBlurryImages = 0;     // 20260330 ZJH 初始化
        report.nOverexposed = 0;      // 20260330 ZJH 初始化
        report.nUnderexposed = 0;     // 20260330 ZJH 初始化
        report.fClassImbalanceRatio = 1.0f;  // 20260330 ZJH 初始化
        report.nMislabeledSuspect = 0;  // 20260330 ZJH 初始化（需模型，静态分析无法检测）
        report.nOutlierImages = 0;    // 20260330 ZJH 初始化
        report.fOverallScore = 100.0f;  // 20260330 ZJH 初始满分

        if (vecImages.empty()) {
            report.vecWarnings.push_back("Dataset is empty");  // 20260330 ZJH 空数据集警告
            report.fOverallScore = 0.0f;  // 20260330 ZJH 零分
            return report;  // 20260330 ZJH 直接返回
        }

        // ===========================================================
        // 20260330 ZJH Step 1: 重复图片检测（感知哈希）
        // ===========================================================
        auto vecDuplicates = findDuplicates(vecImages, nC, nH, nW, 0.95f);  // 20260330 ZJH 默认阈值
        report.nDuplicateImages = static_cast<int>(vecDuplicates.size());  // 20260330 ZJH 重复对数

        if (report.nDuplicateImages > 0) {
            report.vecWarnings.push_back(
                "Found " + std::to_string(report.nDuplicateImages) +
                " duplicate image pairs");  // 20260330 ZJH 重复警告
        }

        // ===========================================================
        // 20260330 ZJH Step 2: 模糊图片检测（Laplacian 方差）
        // ===========================================================
        auto vecBlurry = findBlurry(vecImages, nC, nH, nW, 100.0f);  // 20260330 ZJH 默认阈值
        report.nBlurryImages = static_cast<int>(vecBlurry.size());  // 20260330 ZJH 模糊数量

        if (report.nBlurryImages > 0) {
            report.vecWarnings.push_back(
                "Found " + std::to_string(report.nBlurryImages) +
                " blurry images (Laplacian variance < 100)");  // 20260330 ZJH 模糊警告
        }

        // ===========================================================
        // 20260330 ZJH Step 3: 过曝/欠曝检测（亮度分析）
        // ===========================================================
        int nPixelsPerImage = nC * nH * nW;  // 20260330 ZJH 每张图片总像素数
        for (size_t i = 0; i < vecImages.size(); ++i) {
            if (static_cast<int>(vecImages[i].size()) != nPixelsPerImage) {
                continue;  // 20260330 ZJH 尺寸不匹配，跳过
            }
            // 20260330 ZJH 计算灰度均值（多通道取均值，单通道直接用）
            float fMeanBrightness = 0.0f;  // 20260330 ZJH 平均亮度
            for (int px = 0; px < nPixelsPerImage; ++px) {
                fMeanBrightness += vecImages[i][px];  // 20260330 ZJH 累加所有像素值
            }
            fMeanBrightness /= static_cast<float>(nPixelsPerImage);  // 20260330 ZJH 取均值

            // 20260330 ZJH 判断过曝/欠曝（值域 [0,1]，阈值经验值）
            if (fMeanBrightness > 0.9f) {
                ++report.nOverexposed;  // 20260330 ZJH 过曝计数
            } else if (fMeanBrightness < 0.1f) {
                ++report.nUnderexposed;  // 20260330 ZJH 欠曝计数
            }
        }

        if (report.nOverexposed > 0) {
            report.vecWarnings.push_back(
                "Found " + std::to_string(report.nOverexposed) +
                " overexposed images (mean brightness > 0.9)");  // 20260330 ZJH 过曝警告
        }
        if (report.nUnderexposed > 0) {
            report.vecWarnings.push_back(
                "Found " + std::to_string(report.nUnderexposed) +
                " underexposed images (mean brightness < 0.1)");  // 20260330 ZJH 欠曝警告
        }

        // ===========================================================
        // 20260330 ZJH Step 4: 类别不平衡检测
        // ===========================================================
        if (!vecLabels.empty()) {
            std::unordered_map<int, int> mapClassCount;  // 20260330 ZJH 类别计数映射
            for (int nLabel : vecLabels) {
                ++mapClassCount[nLabel];  // 20260330 ZJH 累加每个类别的样本数
            }
            // 20260330 ZJH 找最大和最小类别
            int nMaxCount = 0;   // 20260330 ZJH 最大类样本数
            int nMinCount = std::numeric_limits<int>::max();  // 20260330 ZJH 最小类样本数
            for (const auto& [nClass, nCount] : mapClassCount) {
                if (nCount > nMaxCount) nMaxCount = nCount;  // 20260330 ZJH 更新最大
                if (nCount < nMinCount) nMinCount = nCount;  // 20260330 ZJH 更新最小
            }
            // 20260330 ZJH 计算不平衡比
            if (nMinCount > 0) {
                report.fClassImbalanceRatio = static_cast<float>(nMaxCount) /
                                              static_cast<float>(nMinCount);  // 20260330 ZJH 比值
            } else {
                report.fClassImbalanceRatio = static_cast<float>(nMaxCount);  // 20260330 ZJH 最小类为零
            }

            if (report.fClassImbalanceRatio > 10.0f) {
                report.vecWarnings.push_back(
                    "Severe class imbalance: ratio = " +
                    std::to_string(report.fClassImbalanceRatio));  // 20260330 ZJH 严重不平衡
            } else if (report.fClassImbalanceRatio > 3.0f) {
                report.vecWarnings.push_back(
                    "Moderate class imbalance: ratio = " +
                    std::to_string(report.fClassImbalanceRatio));  // 20260330 ZJH 中度不平衡
            }
        }

        // ===========================================================
        // 20260330 ZJH Step 5: 离群样本检测（基于像素统计的简单方法）
        // ===========================================================
        // 20260330 ZJH 计算所有图片的全局均值和标准差
        if (vecImages.size() >= 10) {  // 20260330 ZJH 样本足够多才有意义
            // 20260330 ZJH 计算每张图片的均值作为特征
            std::vector<float> vecMeans(vecImages.size());  // 20260330 ZJH 各图片均值
            float fGlobalMean = 0.0f;  // 20260330 ZJH 全局均值
            for (size_t i = 0; i < vecImages.size(); ++i) {
                float fSum = 0.0f;  // 20260330 ZJH 图片像素累加
                for (float fVal : vecImages[i]) {
                    fSum += fVal;  // 20260330 ZJH 累加
                }
                vecMeans[i] = fSum / static_cast<float>(vecImages[i].size());  // 20260330 ZJH 均值
                fGlobalMean += vecMeans[i];  // 20260330 ZJH 累加全局
            }
            fGlobalMean /= static_cast<float>(vecImages.size());  // 20260330 ZJH 全局均值

            // 20260330 ZJH 计算标准差
            float fVariance = 0.0f;  // 20260330 ZJH 方差
            for (size_t i = 0; i < vecImages.size(); ++i) {
                float fDiff = vecMeans[i] - fGlobalMean;  // 20260330 ZJH 偏差
                fVariance += fDiff * fDiff;  // 20260330 ZJH 平方累加
            }
            fVariance /= static_cast<float>(vecImages.size());  // 20260330 ZJH 均值化
            float fStdDev = std::sqrt(fVariance + 1e-10f);  // 20260330 ZJH 标准差

            // 20260330 ZJH 超过 3 sigma 的样本视为离群点
            float fLower = fGlobalMean - 3.0f * fStdDev;  // 20260330 ZJH 下界
            float fUpper = fGlobalMean + 3.0f * fStdDev;  // 20260330 ZJH 上界
            for (size_t i = 0; i < vecImages.size(); ++i) {
                if (vecMeans[i] < fLower || vecMeans[i] > fUpper) {
                    ++report.nOutlierImages;  // 20260330 ZJH 离群计数
                }
            }

            if (report.nOutlierImages > 0) {
                report.vecWarnings.push_back(
                    "Found " + std::to_string(report.nOutlierImages) +
                    " outlier images (> 3 sigma from mean)");  // 20260330 ZJH 离群警告
            }
        }

        // ===========================================================
        // 20260330 ZJH Step 6: 计算综合质量分数 [0, 100]
        // ===========================================================
        float fScore = 100.0f;  // 20260330 ZJH 从满分开始扣减
        int nTotal = report.nTotalImages;  // 20260330 ZJH 总数

        // 20260330 ZJH 重复图片扣分：每 1% 重复扣 5 分
        if (nTotal > 0) {
            float fDupRatio = static_cast<float>(report.nDuplicateImages) /
                              static_cast<float>(nTotal);  // 20260330 ZJH 重复比例
            fScore -= fDupRatio * 500.0f;  // 20260330 ZJH 最多扣 50 分左右
        }

        // 20260330 ZJH 模糊图片扣分：每 1% 模糊扣 3 分
        if (nTotal > 0) {
            float fBlurRatio = static_cast<float>(report.nBlurryImages) /
                               static_cast<float>(nTotal);  // 20260330 ZJH 模糊比例
            fScore -= fBlurRatio * 300.0f;  // 20260330 ZJH 扣分
        }

        // 20260330 ZJH 过曝/欠曝扣分：每 1% 扣 2 分
        if (nTotal > 0) {
            float fExposureRatio = static_cast<float>(report.nOverexposed + report.nUnderexposed) /
                                   static_cast<float>(nTotal);  // 20260330 ZJH 曝光异常比例
            fScore -= fExposureRatio * 200.0f;  // 20260330 ZJH 扣分
        }

        // 20260330 ZJH 类别不平衡扣分
        if (report.fClassImbalanceRatio > 10.0f) {
            fScore -= 20.0f;  // 20260330 ZJH 严重不平衡扣 20 分
        } else if (report.fClassImbalanceRatio > 3.0f) {
            fScore -= 10.0f;  // 20260330 ZJH 中度不平衡扣 10 分
        }

        // 20260330 ZJH 离群样本扣分
        if (nTotal > 0) {
            float fOutlierRatio = static_cast<float>(report.nOutlierImages) /
                                  static_cast<float>(nTotal);  // 20260330 ZJH 离群比例
            fScore -= fOutlierRatio * 200.0f;  // 20260330 ZJH 扣分
        }

        // 20260330 ZJH 限制分数范围 [0, 100]
        report.fOverallScore = std::max(0.0f, std::min(100.0f, fScore));  // 20260330 ZJH 裁剪

        return report;  // 20260330 ZJH 返回综合报告
    }

    // 20260330 ZJH findDuplicates — 检测重复图片（基于感知哈希 pHash）
    // 算法: 将每张图片缩小到 8x8 灰度 → 计算均值 → 大于均值为 1 否则 0 → 64-bit hash
    // 两张图片的汉明距离小于 (1 - threshold) * 64 视为重复
    // vecImages: 展平浮点图像列表
    // nC, nH, nW: 通道数、高度、宽度
    // fThreshold: 相似度阈值 [0, 1]，默认 0.95
    // 返回: 重复图片对的索引列表
    std::vector<std::pair<int, int>> findDuplicates(
        const std::vector<std::vector<float>>& vecImages,
        int nC, int nH, int nW, float fThreshold = 0.95f) {

        std::vector<std::pair<int, int>> vecDuplicates;  // 20260330 ZJH 结果列表
        int nImages = static_cast<int>(vecImages.size());  // 20260330 ZJH 图片数量

        if (nImages < 2) {
            return vecDuplicates;  // 20260330 ZJH 少于两张无法比对
        }

        // 20260330 ZJH 为每张图片计算 64-bit 感知哈希
        std::vector<uint64_t> vecHashes(nImages, 0);  // 20260330 ZJH 哈希值列表
        for (int i = 0; i < nImages; ++i) {
            vecHashes[i] = computePerceptualHash(
                vecImages[i], nC, nH, nW);  // 20260330 ZJH 计算 pHash
        }

        // 20260330 ZJH 计算汉明距离阈值
        // 相似度 threshold=0.95 → 最多允许 64*(1-0.95) = 3.2 ≈ 3 位不同
        int nMaxHammingDist = static_cast<int>(
            (1.0f - fThreshold) * 64.0f + 0.5f);  // 20260330 ZJH 四舍五入

        // 20260330 ZJH O(n^2) 两两比对汉明距离
        for (int i = 0; i < nImages; ++i) {
            for (int j = i + 1; j < nImages; ++j) {
                // 20260330 ZJH 异或后数 1 的个数 = 汉明距离
                uint64_t nXor = vecHashes[i] ^ vecHashes[j];  // 20260330 ZJH 按位异或
                int nHammingDist = popcount64(nXor);  // 20260330 ZJH 计算 1 的个数

                if (nHammingDist <= nMaxHammingDist) {
                    vecDuplicates.push_back({i, j});  // 20260330 ZJH 记录重复对
                }
            }
        }

        return vecDuplicates;  // 20260330 ZJH 返回重复对列表
    }

    // 20260330 ZJH findBlurry — 检测模糊图片（基于 Laplacian 方差）
    // 算法: 3x3 Laplacian 卷积核 → 计算卷积结果的方差 → 方差低于阈值视为模糊
    // Laplacian 核: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    // vecImages: 展平浮点图像列表
    // nC, nH, nW: 通道数、高度、宽度
    // fThreshold: Laplacian 方差阈值，低于此值视为模糊
    // 返回: 模糊图片的索引列表
    std::vector<int> findBlurry(const std::vector<std::vector<float>>& vecImages,
                                 int nC, int nH, int nW, float fThreshold = 100.0f) {

        std::vector<int> vecBlurryIndices;  // 20260330 ZJH 模糊图片索引列表
        int nImages = static_cast<int>(vecImages.size());  // 20260330 ZJH 图片数量
        int nPixelsPerChannel = nH * nW;  // 20260330 ZJH 每通道像素数

        for (int i = 0; i < nImages; ++i) {
            int nExpectedSize = nC * nPixelsPerChannel;  // 20260330 ZJH 期望像素数
            if (static_cast<int>(vecImages[i].size()) != nExpectedSize) {
                continue;  // 20260330 ZJH 尺寸不匹配，跳过
            }

            // 20260330 ZJH 先转灰度（多通道取均值，单通道直接用）
            std::vector<float> vecGray(nPixelsPerChannel, 0.0f);  // 20260330 ZJH 灰度图缓冲
            if (nC == 1) {
                // 20260330 ZJH 单通道直接拷贝
                for (int px = 0; px < nPixelsPerChannel; ++px) {
                    vecGray[px] = vecImages[i][px];  // 20260330 ZJH 直接赋值
                }
            } else {
                // 20260330 ZJH 多通道取均值（CHW 格式）
                for (int c = 0; c < nC; ++c) {
                    for (int px = 0; px < nPixelsPerChannel; ++px) {
                        vecGray[px] += vecImages[i][c * nPixelsPerChannel + px];  // 20260330 ZJH 通道累加
                    }
                }
                float fInvC = 1.0f / static_cast<float>(nC);  // 20260330 ZJH 通道数倒数
                for (int px = 0; px < nPixelsPerChannel; ++px) {
                    vecGray[px] *= fInvC;  // 20260330 ZJH 取均值
                }
            }

            // 20260330 ZJH 计算 Laplacian 方差
            float fLapVar = computeLaplacianVariance(
                vecGray.data(), nH, nW);  // 20260330 ZJH Laplacian 方差

            if (fLapVar < fThreshold) {
                vecBlurryIndices.push_back(i);  // 20260330 ZJH 低于阈值，标记为模糊
            }
        }

        return vecBlurryIndices;  // 20260330 ZJH 返回模糊图片索引
    }

    // 20260330 ZJH findSuspectLabels — 检测疑似标注错误的样本
    // 算法: 用训练好的模型对每个样本推理，若预测类别与标注不一致则标记为疑似错误
    // model: 训练好的分类模型
    // vecImages: 输入图像张量列表
    // vecLabels: 对应的标注标签
    // 返回: 疑似标注错误的样本索引列表
    std::vector<int> findSuspectLabels(Module& model,
                                       const std::vector<Tensor>& vecImages,
                                       const std::vector<int>& vecLabels) {

        std::vector<int> vecSuspect;  // 20260330 ZJH 疑似错误索引列表

        if (vecImages.size() != vecLabels.size()) {
            return vecSuspect;  // 20260330 ZJH 数量不匹配，返回空
        }

        // 20260330 ZJH 切换到评估模式
        bool bWasTraining = model.isTraining();  // 20260330 ZJH 保存训练状态
        model.eval();  // 20260330 ZJH 评估模式

        for (size_t i = 0; i < vecImages.size(); ++i) {
            // 20260330 ZJH 前向推理
            Tensor logits = model.forward(vecImages[i]);  // 20260330 ZJH 前向传播
            auto cLogits = logits.contiguous();  // 20260330 ZJH 连续化

            int nClasses = cLogits.shape(cLogits.ndim() - 1);  // 20260330 ZJH 类别数
            const float* pLogits = cLogits.floatDataPtr();  // 20260330 ZJH logits 指针

            // 20260330 ZJH softmax + argmax
            std::vector<float> vecProb(nClasses);  // 20260330 ZJH 概率缓冲
            cpuSoftmax(pLogits, vecProb.data(), nClasses);  // 20260330 ZJH softmax
            int nPredClass = cpuArgmax(vecProb.data(), nClasses);  // 20260330 ZJH argmax
            float fPredConf = vecProb[nPredClass];  // 20260330 ZJH 预测置信度

            // 20260330 ZJH 高置信度预测与标注不一致 → 疑似标注错误
            // 阈值 0.8: 模型非常确信但与标注不同，标注更可能有问题
            if (nPredClass != vecLabels[i] && fPredConf > 0.8f) {
                vecSuspect.push_back(static_cast<int>(i));  // 20260330 ZJH 记录疑似错误索引
            }
        }

        // 20260330 ZJH 恢复训练模式
        if (bWasTraining) {
            model.train(true);  // 20260330 ZJH 切回训练模式
        }

        return vecSuspect;  // 20260330 ZJH 返回疑似错误索引列表
    }

private:
    // 20260330 ZJH computePerceptualHash — 计算感知哈希（pHash 简化版）
    // 算法: 将图片缩小到 8x8 灰度 → 计算均值 → 大于均值为 1 否则 0 → 64-bit hash
    // vecImage: 展平浮点图像（CHW 格式，值域 [0,1]）
    // nC, nH, nW: 通道数、高度、宽度
    // 返回: 64-bit 感知哈希值
    uint64_t computePerceptualHash(const std::vector<float>& vecImage,
                                   int nC, int nH, int nW) const {

        int nPixelsPerChannel = nH * nW;  // 20260330 ZJH 每通道像素数

        // 20260330 ZJH Step 1: 转灰度
        std::vector<float> vecGray(nPixelsPerChannel, 0.0f);  // 20260330 ZJH 灰度图
        if (nC == 1) {
            for (int px = 0; px < nPixelsPerChannel; ++px) {
                vecGray[px] = vecImage[px];  // 20260330 ZJH 直接拷贝
            }
        } else {
            // 20260330 ZJH 多通道取均值
            for (int c = 0; c < nC; ++c) {
                for (int px = 0; px < nPixelsPerChannel; ++px) {
                    vecGray[px] += vecImage[c * nPixelsPerChannel + px];  // 20260330 ZJH 通道累加
                }
            }
            float fInvC = 1.0f / static_cast<float>(nC);  // 20260330 ZJH 通道数倒数
            for (int px = 0; px < nPixelsPerChannel; ++px) {
                vecGray[px] *= fInvC;  // 20260330 ZJH 取均值
            }
        }

        // 20260330 ZJH Step 2: 缩小到 8x8（最近邻插值）
        constexpr int s_nHashSize = 8;  // 20260330 ZJH 哈希图大小
        float arrSmall[s_nHashSize * s_nHashSize];  // 20260330 ZJH 缩小后的 8x8 灰度图
        for (int sy = 0; sy < s_nHashSize; ++sy) {
            for (int sx = 0; sx < s_nHashSize; ++sx) {
                // 20260330 ZJH 最近邻插值：计算源图坐标
                int nSrcY = static_cast<int>(
                    static_cast<float>(sy) / static_cast<float>(s_nHashSize) *
                    static_cast<float>(nH));  // 20260330 ZJH 映射行坐标
                int nSrcX = static_cast<int>(
                    static_cast<float>(sx) / static_cast<float>(s_nHashSize) *
                    static_cast<float>(nW));  // 20260330 ZJH 映射列坐标

                // 20260330 ZJH 裁剪到合法范围
                nSrcY = std::min(nSrcY, nH - 1);  // 20260330 ZJH 上界裁剪
                nSrcX = std::min(nSrcX, nW - 1);  // 20260330 ZJH 上界裁剪

                arrSmall[sy * s_nHashSize + sx] = vecGray[nSrcY * nW + nSrcX];  // 20260330 ZJH 取值
            }
        }

        // 20260330 ZJH Step 3: 计算 8x8 图的均值
        float fMean = 0.0f;  // 20260330 ZJH 均值
        for (int k = 0; k < s_nHashSize * s_nHashSize; ++k) {
            fMean += arrSmall[k];  // 20260330 ZJH 累加
        }
        fMean /= static_cast<float>(s_nHashSize * s_nHashSize);  // 20260330 ZJH 取均值

        // 20260330 ZJH Step 4: 生成 64-bit 哈希（大于均值为 1，否则为 0）
        uint64_t nHash = 0;  // 20260330 ZJH 哈希值
        for (int k = 0; k < s_nHashSize * s_nHashSize; ++k) {
            if (arrSmall[k] > fMean) {
                nHash |= (static_cast<uint64_t>(1) << k);  // 20260330 ZJH 设置第 k 位
            }
        }

        return nHash;  // 20260330 ZJH 返回 64-bit 感知哈希
    }

    // 20260330 ZJH popcount64 — 计算 64-bit 整数中 1 的个数（汉明重量）
    // 使用分治法（Brian Kernighan's algorithm）高效计算
    // nVal: 输入 64 位整数
    // 返回: 二进制中 1 的个数
    static int popcount64(uint64_t nVal) {
        int nCount = 0;  // 20260330 ZJH 1 的计数器
        while (nVal) {
            nVal &= (nVal - 1);  // 20260330 ZJH 消除最低位的 1（Brian Kernighan 算法）
            ++nCount;  // 20260330 ZJH 计数
        }
        return nCount;  // 20260330 ZJH 返回 1 的个数
    }

    // 20260330 ZJH computeLaplacianVariance — 计算灰度图的 Laplacian 方差
    // Laplacian 核: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    // 方差越低表示图像越模糊（高频分量越少）
    // pGray: 灰度图像数据指针（行优先存储）
    // nH: 图像高度
    // nW: 图像宽度
    // 返回: Laplacian 响应的方差
    float computeLaplacianVariance(const float* pGray, int nH, int nW) const {
        if (nH < 3 || nW < 3) {
            return 0.0f;  // 20260330 ZJH 图像太小无法卷积
        }

        // 20260330 ZJH 对内部像素（去掉边缘 1 像素）施加 Laplacian 卷积核
        int nInnerH = nH - 2;  // 20260330 ZJH 内部高度
        int nInnerW = nW - 2;  // 20260330 ZJH 内部宽度
        int nInnerCount = nInnerH * nInnerW;  // 20260330 ZJH 内部像素总数

        float fSum = 0.0f;    // 20260330 ZJH Laplacian 响应累加
        float fSumSq = 0.0f;  // 20260330 ZJH Laplacian 响应平方累加

        for (int y = 1; y < nH - 1; ++y) {
            for (int x = 1; x < nW - 1; ++x) {
                // 20260330 ZJH Laplacian: L(y,x) = -4*center + top + bottom + left + right
                float fCenter = pGray[y * nW + x];          // 20260330 ZJH 中心像素
                float fTop    = pGray[(y - 1) * nW + x];    // 20260330 ZJH 上方像素
                float fBottom = pGray[(y + 1) * nW + x];    // 20260330 ZJH 下方像素
                float fLeft   = pGray[y * nW + (x - 1)];    // 20260330 ZJH 左方像素
                float fRight  = pGray[y * nW + (x + 1)];    // 20260330 ZJH 右方像素

                float fLap = fTop + fBottom + fLeft + fRight -
                             4.0f * fCenter;  // 20260330 ZJH Laplacian 响应值

                fSum += fLap;           // 20260330 ZJH 累加响应
                fSumSq += fLap * fLap;  // 20260330 ZJH 累加响应平方
            }
        }

        // 20260330 ZJH 方差 = E[X^2] - (E[X])^2
        float fMean = fSum / static_cast<float>(nInnerCount);  // 20260330 ZJH 均值
        float fMeanSq = fSumSq / static_cast<float>(nInnerCount);  // 20260330 ZJH 平方均值
        float fVariance = fMeanSq - fMean * fMean;  // 20260330 ZJH 方差

        // 20260330 ZJH 确保非负（浮点精度可能导致微小负值）
        return std::max(0.0f, fVariance);  // 20260330 ZJH 返回方差
    }
};

// =============================================================================
// 20260330 ZJH 4. A/B 模型对比测试
// =============================================================================
// 在同一测试集上对比两个模型的准确率、延迟和一致性
// 使用 Cohen's Kappa 系数评估两模型预测的一致性（排除随机一致的影响）
// κ = (p_o - p_e) / (1 - p_e)
//   p_o = 实际一致率
//   p_e = 随机一致率（基于各类别的边际概率）
// =============================================================================

// 20260330 ZJH ModelComparator — A/B 模型对比测试器
class ModelComparator {
public:
    // 20260330 ZJH CompareResult — 对比测试结果
    struct CompareResult {
        std::string strModelA, strModelB;  // 20260330 ZJH 模型名称标识
        float fAccuracyA, fAccuracyB;      // 20260330 ZJH 两个模型的准确率
        float fLatencyA, fLatencyB;        // 20260330 ZJH 两个模型的平均推理延迟（毫秒）
        int nAgreements, nDisagreements;   // 20260330 ZJH 两模型预测一致/不一致的样本数
        float fCohenKappa;                 // 20260330 ZJH Cohen's Kappa 一致性系数
        std::string strWinner;             // 20260330 ZJH 综合胜出的模型
        std::vector<int> vecDisagreeIndices;  // 20260330 ZJH 预测不一致的样本索引
    };

    // 20260330 ZJH compare — 在同一测试集上对比两个模型
    // modelA: 模型 A
    // modelB: 模型 B
    // vecTestData: 测试数据张量列表
    // vecTestLabels: 测试数据的真实标签
    // 返回: CompareResult 详细对比结果
    CompareResult compare(Module& modelA, Module& modelB,
                          const std::vector<Tensor>& vecTestData,
                          const std::vector<int>& vecTestLabels) {

        CompareResult result;  // 20260330 ZJH 初始化结果
        result.strModelA = "ModelA";  // 20260330 ZJH 默认名称
        result.strModelB = "ModelB";  // 20260330 ZJH 默认名称
        result.fAccuracyA = 0.0f;  // 20260330 ZJH 初始化
        result.fAccuracyB = 0.0f;  // 20260330 ZJH 初始化
        result.fLatencyA = 0.0f;   // 20260330 ZJH 初始化
        result.fLatencyB = 0.0f;   // 20260330 ZJH 初始化
        result.nAgreements = 0;    // 20260330 ZJH 初始化
        result.nDisagreements = 0; // 20260330 ZJH 初始化
        result.fCohenKappa = 0.0f; // 20260330 ZJH 初始化

        int nSamples = static_cast<int>(vecTestData.size());  // 20260330 ZJH 测试样本数

        if (nSamples == 0 || vecTestData.size() != vecTestLabels.size()) {
            result.strWinner = "N/A";  // 20260330 ZJH 无效输入
            return result;  // 20260330 ZJH 返回空结果
        }

        // 20260330 ZJH 切换到评估模式
        modelA.eval();  // 20260330 ZJH 模型 A 评估模式
        modelB.eval();  // 20260330 ZJH 模型 B 评估模式

        // 20260330 ZJH 存储两个模型的预测结果
        std::vector<int> vecPredsA(nSamples);  // 20260330 ZJH 模型 A 预测
        std::vector<int> vecPredsB(nSamples);  // 20260330 ZJH 模型 B 预测
        int nCorrectA = 0;  // 20260330 ZJH 模型 A 正确数
        int nCorrectB = 0;  // 20260330 ZJH 模型 B 正确数

        // ===========================================================
        // 20260330 ZJH Step 1: 模型 A 推理及计时
        // ===========================================================
        {
            auto tStart = std::chrono::high_resolution_clock::now();  // 20260330 ZJH 开始计时

            for (int i = 0; i < nSamples; ++i) {
                Tensor logits = modelA.forward(vecTestData[i]);  // 20260330 ZJH 前向传播
                auto cLogits = logits.contiguous();  // 20260330 ZJH 连续化
                int nClasses = cLogits.shape(cLogits.ndim() - 1);  // 20260330 ZJH 类别数
                const float* pLogits = cLogits.floatDataPtr();  // 20260330 ZJH 数据指针

                // 20260330 ZJH softmax + argmax
                std::vector<float> vecProb(nClasses);  // 20260330 ZJH 概率缓冲
                cpuSoftmax(pLogits, vecProb.data(), nClasses);  // 20260330 ZJH softmax
                vecPredsA[i] = cpuArgmax(vecProb.data(), nClasses);  // 20260330 ZJH argmax

                if (vecPredsA[i] == vecTestLabels[i]) {
                    ++nCorrectA;  // 20260330 ZJH 正确计数
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();  // 20260330 ZJH 结束计时
            double dTotalMs = std::chrono::duration<double, std::milli>(
                tEnd - tStart).count();  // 20260330 ZJH 总耗时（毫秒）
            result.fLatencyA = static_cast<float>(
                dTotalMs / static_cast<double>(nSamples));  // 20260330 ZJH 平均延迟
        }

        // ===========================================================
        // 20260330 ZJH Step 2: 模型 B 推理及计时
        // ===========================================================
        {
            auto tStart = std::chrono::high_resolution_clock::now();  // 20260330 ZJH 开始计时

            for (int i = 0; i < nSamples; ++i) {
                Tensor logits = modelB.forward(vecTestData[i]);  // 20260330 ZJH 前向传播
                auto cLogits = logits.contiguous();  // 20260330 ZJH 连续化
                int nClasses = cLogits.shape(cLogits.ndim() - 1);  // 20260330 ZJH 类别数
                const float* pLogits = cLogits.floatDataPtr();  // 20260330 ZJH 数据指针

                // 20260330 ZJH softmax + argmax
                std::vector<float> vecProb(nClasses);  // 20260330 ZJH 概率缓冲
                cpuSoftmax(pLogits, vecProb.data(), nClasses);  // 20260330 ZJH softmax
                vecPredsB[i] = cpuArgmax(vecProb.data(), nClasses);  // 20260330 ZJH argmax

                if (vecPredsB[i] == vecTestLabels[i]) {
                    ++nCorrectB;  // 20260330 ZJH 正确计数
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();  // 20260330 ZJH 结束计时
            double dTotalMs = std::chrono::duration<double, std::milli>(
                tEnd - tStart).count();  // 20260330 ZJH 总耗时（毫秒）
            result.fLatencyB = static_cast<float>(
                dTotalMs / static_cast<double>(nSamples));  // 20260330 ZJH 平均延迟
        }

        // ===========================================================
        // 20260330 ZJH Step 3: 计算准确率
        // ===========================================================
        result.fAccuracyA = static_cast<float>(nCorrectA) /
                            static_cast<float>(nSamples);  // 20260330 ZJH 模型 A 准确率
        result.fAccuracyB = static_cast<float>(nCorrectB) /
                            static_cast<float>(nSamples);  // 20260330 ZJH 模型 B 准确率

        // ===========================================================
        // 20260330 ZJH Step 4: 计算一致性和不一致样本
        // ===========================================================
        for (int i = 0; i < nSamples; ++i) {
            if (vecPredsA[i] == vecPredsB[i]) {
                ++result.nAgreements;  // 20260330 ZJH 预测一致
            } else {
                ++result.nDisagreements;  // 20260330 ZJH 预测不一致
                result.vecDisagreeIndices.push_back(i);  // 20260330 ZJH 记录不一致索引
            }
        }

        // ===========================================================
        // 20260330 ZJH Step 5: 计算 Cohen's Kappa 系数
        // κ = (p_o - p_e) / (1 - p_e)
        //   p_o = 实际一致率（两模型预测相同的比例）
        //   p_e = 随机一致率（假设两模型独立时预期的一致率）
        // ===========================================================
        result.fCohenKappa = computeCohenKappa(
            vecPredsA, vecPredsB, nSamples);  // 20260330 ZJH 计算 Kappa

        // ===========================================================
        // 20260330 ZJH Step 6: 综合判定胜者
        // 规则: 准确率差异 > 1% 以准确率为准，否则看延迟
        // ===========================================================
        float fAccDiff = result.fAccuracyA - result.fAccuracyB;  // 20260330 ZJH 准确率差异
        if (fAccDiff > 0.01f) {
            result.strWinner = "ModelA";  // 20260330 ZJH A 准确率显著更高
        } else if (fAccDiff < -0.01f) {
            result.strWinner = "ModelB";  // 20260330 ZJH B 准确率显著更高
        } else {
            // 20260330 ZJH 准确率接近，以延迟为辅助判定
            if (result.fLatencyA < result.fLatencyB) {
                result.strWinner = "ModelA";  // 20260330 ZJH A 更快
            } else if (result.fLatencyB < result.fLatencyA) {
                result.strWinner = "ModelB";  // 20260330 ZJH B 更快
            } else {
                result.strWinner = "Tie";  // 20260330 ZJH 完全持平
            }
        }

        return result;  // 20260330 ZJH 返回完整对比结果
    }

private:
    // 20260330 ZJH computeCohenKappa — 计算 Cohen's Kappa 一致性系数
    // 公式: κ = (p_o - p_e) / (1 - p_e)
    //   p_o: 观测到的一致率（两预测相同的比例）
    //   p_e: 期望随机一致率（各类别的边际概率乘积之和）
    // vecPredsA: 模型 A 的预测列表
    // vecPredsB: 模型 B 的预测列表
    // nSamples: 样本总数
    // 返回: Kappa 系数，[-1, 1]，1=完全一致，0=随机一致，<0=低于随机
    float computeCohenKappa(const std::vector<int>& vecPredsA,
                            const std::vector<int>& vecPredsB,
                            int nSamples) const {

        if (nSamples == 0) {
            return 0.0f;  // 20260330 ZJH 无样本，返回 0
        }

        // 20260330 ZJH 找到所有出现过的类别（两个模型的并集）
        std::unordered_set<int> setClasses;  // 20260330 ZJH 类别集合
        for (int i = 0; i < nSamples; ++i) {
            setClasses.insert(vecPredsA[i]);  // 20260330 ZJH 加入 A 的类别
            setClasses.insert(vecPredsB[i]);  // 20260330 ZJH 加入 B 的类别
        }

        // 20260330 ZJH 计算 p_o: 观测一致率
        int nAgree = 0;  // 20260330 ZJH 一致计数
        for (int i = 0; i < nSamples; ++i) {
            if (vecPredsA[i] == vecPredsB[i]) {
                ++nAgree;  // 20260330 ZJH 计数
            }
        }
        float fPo = static_cast<float>(nAgree) /
                     static_cast<float>(nSamples);  // 20260330 ZJH 观测一致率

        // 20260330 ZJH 计算 p_e: 期望随机一致率
        // p_e = Σ_k (n_kA / N) * (n_kB / N)
        // 其中 n_kA 是模型 A 预测为类别 k 的次数，n_kB 同理
        float fPe = 0.0f;  // 20260330 ZJH 期望一致率

        // 20260330 ZJH 统计每个模型对每个类别的预测次数
        std::unordered_map<int, int> mapCountA, mapCountB;  // 20260330 ZJH 类别计数
        for (int i = 0; i < nSamples; ++i) {
            ++mapCountA[vecPredsA[i]];  // 20260330 ZJH A 的类别计数
            ++mapCountB[vecPredsB[i]];  // 20260330 ZJH B 的类别计数
        }

        float fN = static_cast<float>(nSamples);  // 20260330 ZJH 总样本数（浮点）
        for (int nClass : setClasses) {
            // 20260330 ZJH 获取两个模型对该类别的预测次数
            float fCountA = 0.0f;  // 20260330 ZJH 模型 A 预测为该类的次数
            float fCountB = 0.0f;  // 20260330 ZJH 模型 B 预测为该类的次数
            auto itA = mapCountA.find(nClass);  // 20260330 ZJH 查找 A
            if (itA != mapCountA.end()) {
                fCountA = static_cast<float>(itA->second);  // 20260330 ZJH 取值
            }
            auto itB = mapCountB.find(nClass);  // 20260330 ZJH 查找 B
            if (itB != mapCountB.end()) {
                fCountB = static_cast<float>(itB->second);  // 20260330 ZJH 取值
            }
            fPe += (fCountA / fN) * (fCountB / fN);  // 20260330 ZJH 边际概率乘积累加
        }

        // 20260330 ZJH 计算 Kappa
        if (std::abs(1.0f - fPe) < 1e-10f) {
            // 20260330 ZJH 分母接近零，说明几乎所有预测都集中在一个类别
            return (fPo >= 1.0f - 1e-10f) ? 1.0f : 0.0f;  // 20260330 ZJH 特殊处理
        }

        float fKappa = (fPo - fPe) / (1.0f - fPe);  // 20260330 ZJH Cohen's Kappa 公式

        return fKappa;  // 20260330 ZJH 返回 Kappa 系数
    }
};

}  // namespace om
