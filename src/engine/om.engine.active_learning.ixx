// 20260330 ZJH 主动学习模块 — 对标 Keyence 自动再训练提示
// 从未标注池中选择最有价值的样本请求人工标注，显著提升标注效率
// 策略:
//   1. Uncertainty (不确定性采样): 选择模型最不确定的样本（最大熵 / 最低置信度）
//   2. Diversity (多样性采样): K-Center-Greedy 在特征空间中选择最分散的样本
//   3. ExpectedGradient (预期梯度): 选择预期梯度范数最大的样本（信息增益最大）
//   4. Combined (综合策略): 加权组合不确定性和多样性分数
//   5. Random (随机基线): 用于对比实验
// 工业场景: 新产品上线时只有少量标注，主动学习可以用最少标注达到最高精度
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <stdexcept>
#include <iostream>

export module om.engine.active_learning;

// 20260330 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260330 ZJH SamplingStrategy — 主动学习采样策略枚举
// =========================================================
enum class SamplingStrategy {
    Uncertainty,       // 20260330 ZJH 不确定性采样：最大熵 / 最低置信度 / 最大 margin
    Diversity,         // 20260330 ZJH 多样性采样：K-Center-Greedy（CoreSet）
    ExpectedGradient,  // 20260330 ZJH 预期梯度长度：EGL (Expected Gradient Length)
    Combined,          // 20260330 ZJH 综合策略：不确定性 × 多样性加权组合
    Random             // 20260330 ZJH 随机基线：用于消融实验对比
};

// =========================================================
// 20260330 ZJH ActiveSample — 主动学习样本评分结构
// 存储每个未标注样本的各项分数，用于排序和选择
// =========================================================
struct ActiveSample {
    int nImageIndex = -1;     // 20260330 ZJH 样本在未标注池中的索引
    float fUncertainty = 0.0f;  // 20260330 ZJH 不确定性分数（熵/1-max_prob/margin）
    float fDiversity = 0.0f;    // 20260330 ZJH 多样性分数（到最近已选样本的距离）
    float fGradientNorm = 0.0f; // 20260330 ZJH 预期梯度范数
    float fCombinedScore = 0.0f;  // 20260330 ZJH 综合分数（最终排序依据）
};

// =========================================================
// 20260330 ZJH UncertaintyMetric — 不确定性度量方式
// =========================================================
enum class UncertaintyMetric {
    Entropy,          // 20260330 ZJH 最大熵: H(p) = -sum(p_i * log(p_i))，信息量最大
    LeastConfidence,  // 20260330 ZJH 最低置信度: 1 - max(p_i)，最不确定
    MarginSampling    // 20260330 ZJH 最小 margin: 1 - (p_max1 - p_max2)，两个最高概率差最小
};

// =========================================================
// 20260330 ZJH ActiveLearner — 主动学习管理器
// 核心功能:
//   1. 对未标注池中的每个样本计算各维度分数
//   2. 按指定策略排序并返回 Top-K 最有价值的样本
//   3. 支持增量式选择（已标注池逐步增大）
// =========================================================
class ActiveLearner {
public:
    // 20260330 ZJH 构造函数
    // eDefaultMetric: 默认不确定性度量方式
    // fDiversityWeight: 综合策略中多样性权重（0-1），默认 0.5
    ActiveLearner(UncertaintyMetric eDefaultMetric = UncertaintyMetric::Entropy,
                  float fDiversityWeight = 0.5f)
        : m_eMetric(eDefaultMetric),
          m_fDiversityWeight(fDiversityWeight)
    {}

    // 20260330 ZJH rankUnlabeled — 对未标注样本评分并排序
    // 使用模型对未标注池中的每个样本进行推理，计算不确定性/多样性/梯度分数
    // model: 已训练的模型（用于推理获取预测概率）
    // vecUnlabeled: 未标注样本张量列表，每个 [1,C,H,W]
    // eStrategy: 采样策略
    // nTopK: 返回前 K 个最有价值的样本
    // 返回: 按综合分数降序排列的 ActiveSample 向量
    std::vector<ActiveSample> rankUnlabeled(
        Module& model,
        const std::vector<Tensor>& vecUnlabeled,
        SamplingStrategy eStrategy,
        int nTopK = 10)
    {
        // 20260330 ZJH 校验输入
        if (vecUnlabeled.empty()) return {};
        nTopK = std::min(nTopK, static_cast<int>(vecUnlabeled.size()));  // 20260330 ZJH 限制 TopK 不超过池大小

        // 20260330 ZJH 切换模型到评估模式（BN 使用 running stats，Dropout 关闭）
        model.eval();

        // 20260330 ZJH Step 1: 对每个未标注样本进行推理，获取预测概率分布
        std::vector<std::vector<float>> vecAllPreds;  // 20260330 ZJH 所有样本的 softmax 概率
        std::vector<std::vector<float>> vecAllFeatures;  // 20260330 ZJH 所有样本的特征向量（用于多样性）
        vecAllPreds.reserve(vecUnlabeled.size());
        vecAllFeatures.reserve(vecUnlabeled.size());

        for (size_t i = 0; i < vecUnlabeled.size(); ++i) {
            // 20260330 ZJH 前向推理: [1,C,H,W] → [1, nClasses]
            auto output = model.forward(vecUnlabeled[i]);
            auto cOutput = output.contiguous();
            int nClasses = cOutput.numel();  // 20260330 ZJH 类别数

            // 20260330 ZJH 计算 softmax 概率
            const float* pOut = cOutput.floatDataPtr();
            std::vector<float> vecProbs(nClasses);  // 20260330 ZJH softmax 概率向量
            float fMax = *std::max_element(pOut, pOut + nClasses);  // 20260330 ZJH 稳定化最大值
            float fSum = 0.0f;
            for (int c = 0; c < nClasses; ++c) {
                vecProbs[c] = std::exp(pOut[c] - fMax);  // 20260330 ZJH exp(logit - max) 防溢出
                fSum += vecProbs[c];
            }
            for (int c = 0; c < nClasses; ++c) {
                vecProbs[c] /= fSum;  // 20260330 ZJH 归一化为概率
            }

            vecAllPreds.push_back(vecProbs);  // 20260330 ZJH 存储概率
            vecAllFeatures.push_back(vecProbs);  // 20260330 ZJH 用概率向量作为特征（简化版；完整版应取中间特征）
        }

        // 20260330 ZJH Step 2: 根据策略计算各维度分数
        std::vector<ActiveSample> vecSamples(vecUnlabeled.size());
        for (size_t i = 0; i < vecUnlabeled.size(); ++i) {
            vecSamples[i].nImageIndex = static_cast<int>(i);
        }

        // 20260330 ZJH 计算不确定性分数（除 Random 策略外都需要）
        if (eStrategy != SamplingStrategy::Random) {
            computeUncertainty(vecAllPreds, vecSamples);
        }

        // 20260330 ZJH 根据策略计算综合分数
        switch (eStrategy) {
            case SamplingStrategy::Uncertainty: {
                // 20260330 ZJH 仅用不确定性排序
                for (auto& sample : vecSamples) {
                    sample.fCombinedScore = sample.fUncertainty;  // 20260330 ZJH 直接用不确定性作为综合分数
                }
                break;
            }
            case SamplingStrategy::Diversity: {
                // 20260330 ZJH K-Center-Greedy 贪心多样性选择
                auto vecSelected = selectByDiversity(vecAllFeatures, nTopK);
                // 20260330 ZJH 将被选中的样本综合分数设为高值（按选择顺序递减）
                for (int k = 0; k < static_cast<int>(vecSelected.size()); ++k) {
                    vecSamples[vecSelected[k]].fDiversity = 1.0f;
                    vecSamples[vecSelected[k]].fCombinedScore = static_cast<float>(nTopK - k);
                }
                break;
            }
            case SamplingStrategy::ExpectedGradient: {
                // 20260330 ZJH 预期梯度长度: 使用预测分布的离散度近似
                // EGL 完整版需要实际计算梯度，此处用近似法：
                //   EGL ≈ sum_c p_c * ||grad L(x, c)||
                //   近似为 1 - max(p)^2，因为高置信度样本梯度小
                for (size_t i = 0; i < vecSamples.size(); ++i) {
                    float fMaxProb = *std::max_element(vecAllPreds[i].begin(), vecAllPreds[i].end());
                    vecSamples[i].fGradientNorm = 1.0f - fMaxProb * fMaxProb;  // 20260330 ZJH 近似 EGL
                    vecSamples[i].fCombinedScore = vecSamples[i].fGradientNorm;
                }
                break;
            }
            case SamplingStrategy::Combined: {
                // 20260330 ZJH 综合策略: 先按不确定性预筛选 2*TopK，再从中选多样性最高的
                // 20260330 ZJH 第一步：不确定性排序
                std::vector<int> vecUncertIndices(vecSamples.size());
                std::iota(vecUncertIndices.begin(), vecUncertIndices.end(), 0);
                std::sort(vecUncertIndices.begin(), vecUncertIndices.end(),
                    [&](int a, int b) {
                        return vecSamples[a].fUncertainty > vecSamples[b].fUncertainty;
                    });

                // 20260330 ZJH 取前 2*TopK 作为候选池
                int nCandidateSize = std::min(2 * nTopK, static_cast<int>(vecUncertIndices.size()));
                std::vector<std::vector<float>> vecCandFeatures;  // 20260330 ZJH 候选特征
                std::vector<int> vecCandOrigIdx;  // 20260330 ZJH 候选样本在原始池中的索引
                for (int k = 0; k < nCandidateSize; ++k) {
                    int nIdx = vecUncertIndices[k];
                    vecCandFeatures.push_back(vecAllFeatures[nIdx]);
                    vecCandOrigIdx.push_back(nIdx);
                }

                // 20260330 ZJH 第二步：从候选池中用多样性选择 TopK
                auto vecDivSelected = selectByDiversity(vecCandFeatures, nTopK);
                for (int k = 0; k < static_cast<int>(vecDivSelected.size()); ++k) {
                    int nOrigIdx = vecCandOrigIdx[vecDivSelected[k]];
                    vecSamples[nOrigIdx].fDiversity = 1.0f;
                    // 20260330 ZJH 综合分数 = 不确定性权重 × 不确定性 + 多样性权重 × 排名分
                    vecSamples[nOrigIdx].fCombinedScore =
                        (1.0f - m_fDiversityWeight) * vecSamples[nOrigIdx].fUncertainty +
                        m_fDiversityWeight * static_cast<float>(nTopK - k) / static_cast<float>(nTopK);
                }
                break;
            }
            case SamplingStrategy::Random: {
                // 20260330 ZJH 随机分数
                std::mt19937 rng(42);
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (auto& sample : vecSamples) {
                    sample.fCombinedScore = dist(rng);  // 20260330 ZJH 随机分数
                }
                break;
            }
        }

        // 20260330 ZJH Step 3: 按综合分数降序排序，返回 Top-K
        std::sort(vecSamples.begin(), vecSamples.end(),
            [](const ActiveSample& a, const ActiveSample& b) {
                return a.fCombinedScore > b.fCombinedScore;  // 20260330 ZJH 降序排列
            });

        // 20260330 ZJH 截取前 nTopK 个
        if (static_cast<int>(vecSamples.size()) > nTopK) {
            vecSamples.resize(nTopK);
        }

        return vecSamples;  // 20260330 ZJH 返回最有价值的 Top-K 样本
    }

    // 20260330 ZJH selectByUncertainty — 基于不确定性的样本选择（静态工具函数）
    // 从预测概率分布中计算不确定性，返回不确定性最高的 nSelect 个样本索引
    // vecPredictions: 每个样本的 softmax 概率向量
    // nSelect: 选择数量
    // eMetric: 不确定性度量方式
    // 返回: 被选样本的索引（不确定性降序）
    static std::vector<int> selectByUncertainty(
        const std::vector<std::vector<float>>& vecPredictions,
        int nSelect,
        UncertaintyMetric eMetric = UncertaintyMetric::Entropy)
    {
        int nN = static_cast<int>(vecPredictions.size());  // 20260330 ZJH 样本总数
        nSelect = std::min(nSelect, nN);  // 20260330 ZJH 限制选择数量

        // 20260330 ZJH 计算每个样本的不确定性分数
        std::vector<std::pair<float, int>> vecScores;  // 20260330 ZJH {分数, 索引}
        vecScores.reserve(nN);

        for (int i = 0; i < nN; ++i) {
            float fScore = 0.0f;
            const auto& probs = vecPredictions[i];

            switch (eMetric) {
                case UncertaintyMetric::Entropy: {
                    // 20260330 ZJH 信息熵: H(p) = -sum(p_i * log(p_i))
                    for (float p : probs) {
                        if (p > 1e-8f) {
                            fScore -= p * std::log(p);  // 20260330 ZJH 逐类别累加 -p*log(p)
                        }
                    }
                    break;
                }
                case UncertaintyMetric::LeastConfidence: {
                    // 20260330 ZJH 最低置信度: 1 - max(p_i)
                    float fMax = *std::max_element(probs.begin(), probs.end());
                    fScore = 1.0f - fMax;  // 20260330 ZJH max_prob 越小越不确定
                    break;
                }
                case UncertaintyMetric::MarginSampling: {
                    // 20260330 ZJH 最小 margin: 1 - (p_max1 - p_max2)
                    auto sorted = probs;
                    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
                    float fMargin = (sorted.size() >= 2) ? (sorted[0] - sorted[1]) : sorted[0];
                    fScore = 1.0f - fMargin;  // 20260330 ZJH margin 越小越不确定
                    break;
                }
            }

            vecScores.push_back({fScore, i});  // 20260330 ZJH 记录分数和索引
        }

        // 20260330 ZJH 按分数降序排序
        std::sort(vecScores.begin(), vecScores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // 20260330 ZJH 取前 nSelect 个索引
        std::vector<int> vecResult;
        vecResult.reserve(nSelect);
        for (int k = 0; k < nSelect; ++k) {
            vecResult.push_back(vecScores[k].second);
        }

        return vecResult;  // 20260330 ZJH 返回最不确定的样本索引
    }

    // 20260330 ZJH selectByDiversity — K-Center-Greedy 多样性选择（静态工具函数）
    // 论文: "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
    // 贪心算法: 每次选择距离已选集合最远的样本，最大化空间覆盖
    // vecFeatures: 每个样本的特征向量（嵌入或概率向量）
    // nSelect: 选择数量
    // 返回: 被选样本的索引（按选择顺序）
    static std::vector<int> selectByDiversity(
        const std::vector<std::vector<float>>& vecFeatures,
        int nSelect)
    {
        int nN = static_cast<int>(vecFeatures.size());  // 20260330 ZJH 样本总数
        nSelect = std::min(nSelect, nN);  // 20260330 ZJH 限制选择数量

        if (nN == 0 || nSelect == 0) return {};  // 20260330 ZJH 空输入快速返回

        std::vector<int> vecSelected;  // 20260330 ZJH 已选索引列表
        vecSelected.reserve(nSelect);
        std::vector<bool> vecChosen(nN, false);  // 20260330 ZJH 标记是否已选

        // 20260330 ZJH 初始化: 选择第一个样本（选择离"中心"最远的，这里简化为选第 0 个）
        // 完整版应选离已标注集最远的点，此处无已标注集，取第 0 个
        vecSelected.push_back(0);
        vecChosen[0] = true;

        // 20260330 ZJH 维护每个未选样本到已选集合的最小距离
        int nDim = static_cast<int>(vecFeatures[0].size());  // 20260330 ZJH 特征维度
        std::vector<float> vecMinDist(nN, std::numeric_limits<float>::max());

        // 20260330 ZJH 初始化距离: 所有样本到第 0 个的距离
        for (int i = 1; i < nN; ++i) {
            float fDist = 0.0f;
            for (int d = 0; d < nDim; ++d) {
                float fDiff = vecFeatures[i][d] - vecFeatures[0][d];
                fDist += fDiff * fDiff;  // 20260330 ZJH 欧氏距离平方
            }
            vecMinDist[i] = fDist;
        }
        vecMinDist[0] = -1.0f;  // 20260330 ZJH 已选样本标记为负数

        // 20260330 ZJH 贪心选择: 每次选 minDist 最大的样本
        for (int k = 1; k < nSelect; ++k) {
            // 20260330 ZJH 找到距离已选集合最远的未选样本
            int nBestIdx = -1;
            float fBestDist = -1.0f;
            for (int i = 0; i < nN; ++i) {
                if (!vecChosen[i] && vecMinDist[i] > fBestDist) {
                    fBestDist = vecMinDist[i];
                    nBestIdx = i;
                }
            }

            if (nBestIdx < 0) break;  // 20260330 ZJH 无更多可选样本

            // 20260330 ZJH 选中该样本
            vecSelected.push_back(nBestIdx);
            vecChosen[nBestIdx] = true;
            vecMinDist[nBestIdx] = -1.0f;  // 20260330 ZJH 标记已选

            // 20260330 ZJH 更新所有未选样本的最小距离
            for (int i = 0; i < nN; ++i) {
                if (vecChosen[i]) continue;  // 20260330 ZJH 跳过已选
                float fDist = 0.0f;
                for (int d = 0; d < nDim; ++d) {
                    float fDiff = vecFeatures[i][d] - vecFeatures[nBestIdx][d];
                    fDist += fDiff * fDiff;  // 20260330 ZJH 到新选样本的距离平方
                }
                // 20260330 ZJH 取与已有最小距离的较小值
                vecMinDist[i] = std::min(vecMinDist[i], fDist);
            }
        }

        return vecSelected;  // 20260330 ZJH 返回被选样本索引（按选择顺序）
    }

    // 20260330 ZJH computeAnnotationBudget — 计算标注预算建议
    // 基于当前模型精度和目标精度，估算还需要标注多少样本
    // fCurrentAccuracy: 当前模型精度 (0-1)
    // fTargetAccuracy: 目标精度 (0-1)
    // nCurrentLabeled: 当前已标注样本数
    // nUnlabeledPool: 未标注池大小
    // 返回: 建议标注的样本数
    static int computeAnnotationBudget(
        float fCurrentAccuracy, float fTargetAccuracy,
        int nCurrentLabeled, int nUnlabeledPool)
    {
        // 20260330 ZJH 校验输入
        if (fCurrentAccuracy >= fTargetAccuracy) return 0;  // 20260330 ZJH 已达标
        if (fTargetAccuracy > 1.0f) fTargetAccuracy = 1.0f;  // 20260330 ZJH 上限截断

        // 20260330 ZJH 经验公式: 精度提升与标注量的对数关系
        // accuracy ≈ a * log(n) + b（学习曲线近似）
        // 额外需要: n_target / n_current ≈ exp((target - current) / slope)
        // 使用保守估计 slope = 0.1
        float fSlope = 0.1f;  // 20260330 ZJH 学习曲线斜率（经验值）
        float fGap = fTargetAccuracy - fCurrentAccuracy;  // 20260330 ZJH 精度差距
        float fMultiplier = std::exp(fGap / fSlope);  // 20260330 ZJH 需要的数据倍数
        int nNeeded = static_cast<int>(std::ceil(static_cast<float>(nCurrentLabeled) * (fMultiplier - 1.0f)));

        // 20260330 ZJH 限制在未标注池范围内
        nNeeded = std::min(nNeeded, nUnlabeledPool);
        nNeeded = std::max(nNeeded, 1);  // 20260330 ZJH 至少建议 1 个

        return nNeeded;  // 20260330 ZJH 返回建议标注数
    }

    // 20260330 ZJH setUncertaintyMetric — 设置不确定性度量方式
    void setUncertaintyMetric(UncertaintyMetric eMetric) {
        m_eMetric = eMetric;
    }

    // 20260330 ZJH setDiversityWeight — 设置综合策略中的多样性权重
    void setDiversityWeight(float fWeight) {
        m_fDiversityWeight = std::max(0.0f, std::min(1.0f, fWeight));  // 20260330 ZJH 限制在 [0,1]
    }

private:
    // 20260330 ZJH computeUncertainty — 计算所有样本的不确定性分数（内部辅助函数）
    void computeUncertainty(const std::vector<std::vector<float>>& vecPreds,
                            std::vector<ActiveSample>& vecSamples) {
        for (size_t i = 0; i < vecPreds.size(); ++i) {
            const auto& probs = vecPreds[i];  // 20260330 ZJH 当前样本的概率分布
            float fScore = 0.0f;

            switch (m_eMetric) {
                case UncertaintyMetric::Entropy: {
                    // 20260330 ZJH 信息熵
                    for (float p : probs) {
                        if (p > 1e-8f) {
                            fScore -= p * std::log(p);
                        }
                    }
                    break;
                }
                case UncertaintyMetric::LeastConfidence: {
                    // 20260330 ZJH 最低置信度
                    float fMax = *std::max_element(probs.begin(), probs.end());
                    fScore = 1.0f - fMax;
                    break;
                }
                case UncertaintyMetric::MarginSampling: {
                    // 20260330 ZJH 最小 margin
                    auto sorted = probs;
                    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
                    float fMargin = (sorted.size() >= 2) ? (sorted[0] - sorted[1]) : sorted[0];
                    fScore = 1.0f - fMargin;
                    break;
                }
            }

            vecSamples[i].fUncertainty = fScore;  // 20260330 ZJH 写入不确定性分数
        }
    }

    UncertaintyMetric m_eMetric;     // 20260330 ZJH 不确定性度量方式
    float m_fDiversityWeight;         // 20260330 ZJH 综合策略中多样性权重
};

// =========================================================
// 20260330 ZJH ActiveLearningPipeline — 主动学习完整流水线
// 封装从"初始标注→训练→选择→标注→重训练"的完整迭代流程
// 工业场景典型用法:
//   1. 用少量初始标注训练基础模型
//   2. 主动学习选择最有价值的未标注样本
//   3. 人工标注这些样本
//   4. 重新训练模型
//   5. 重复直到达到目标精度
// =========================================================
class ActiveLearningPipeline {
public:
    // 20260330 ZJH 构造函数
    // nBatchSize: 每轮选择的标注数量（标注预算）
    // eStrategy: 采样策略
    ActiveLearningPipeline(int nBatchSize = 10,
                           SamplingStrategy eStrategy = SamplingStrategy::Combined)
        : m_nBatchSize(nBatchSize),
          m_eStrategy(eStrategy),
          m_nCurrentRound(0)
    {}

    // 20260330 ZJH setUnlabeledPool — 设置未标注样本池
    void setUnlabeledPool(const std::vector<Tensor>& vecUnlabeled) {
        m_vecUnlabeled = vecUnlabeled;
        // 20260330 ZJH 初始化所有样本为未标注状态
        m_vecIsLabeled.assign(vecUnlabeled.size(), false);
    }

    // 20260330 ZJH selectNextBatch — 选择下一批要标注的样本
    // model: 当前已训练的模型
    // 返回: 应标注的样本索引列表
    std::vector<int> selectNextBatch(Module& model) {
        // 20260330 ZJH 收集未标注样本
        std::vector<Tensor> vecCandidates;  // 20260330 ZJH 未标注候选样本
        std::vector<int> vecCandOrigIdx;    // 20260330 ZJH 候选在原始池中的索引
        for (size_t i = 0; i < m_vecUnlabeled.size(); ++i) {
            if (!m_vecIsLabeled[i]) {
                vecCandidates.push_back(m_vecUnlabeled[i]);
                vecCandOrigIdx.push_back(static_cast<int>(i));
            }
        }

        if (vecCandidates.empty()) return {};  // 20260330 ZJH 无未标注样本

        // 20260330 ZJH 使用 ActiveLearner 评分和排序
        ActiveLearner learner;
        auto vecRanked = learner.rankUnlabeled(model, vecCandidates, m_eStrategy, m_nBatchSize);

        // 20260330 ZJH 将候选索引映射回原始池索引
        std::vector<int> vecResult;
        vecResult.reserve(vecRanked.size());
        for (const auto& sample : vecRanked) {
            int nOrigIdx = vecCandOrigIdx[sample.nImageIndex];  // 20260330 ZJH 映射回原始索引
            vecResult.push_back(nOrigIdx);
        }

        return vecResult;  // 20260330 ZJH 返回应标注的原始池索引
    }

    // 20260330 ZJH markLabeled — 标记指定样本已完成标注
    // vecIndices: 已标注的样本索引列表
    void markLabeled(const std::vector<int>& vecIndices) {
        for (int idx : vecIndices) {
            if (idx >= 0 && idx < static_cast<int>(m_vecIsLabeled.size())) {
                m_vecIsLabeled[idx] = true;  // 20260330 ZJH 标记为已标注
            }
        }
        ++m_nCurrentRound;  // 20260330 ZJH 轮次+1
    }

    // 20260330 ZJH getLabeledCount — 获取当前已标注样本数
    int getLabeledCount() const {
        int nCount = 0;
        for (bool b : m_vecIsLabeled) {
            if (b) ++nCount;
        }
        return nCount;
    }

    // 20260330 ZJH getUnlabeledCount — 获取当前未标注样本数
    int getUnlabeledCount() const {
        return static_cast<int>(m_vecUnlabeled.size()) - getLabeledCount();
    }

    // 20260330 ZJH getCurrentRound — 获取当前主动学习轮次
    int getCurrentRound() const { return m_nCurrentRound; }

    // 20260330 ZJH setBatchSize — 设置每轮标注预算
    void setBatchSize(int nBatchSize) { m_nBatchSize = nBatchSize; }

    // 20260330 ZJH setStrategy — 设置采样策略
    void setStrategy(SamplingStrategy eStrategy) { m_eStrategy = eStrategy; }

private:
    int m_nBatchSize;                    // 20260330 ZJH 每轮选择的标注数量
    SamplingStrategy m_eStrategy;        // 20260330 ZJH 采样策略
    int m_nCurrentRound;                 // 20260330 ZJH 当前轮次
    std::vector<Tensor> m_vecUnlabeled;  // 20260330 ZJH 未标注样本池
    std::vector<bool> m_vecIsLabeled;    // 20260330 ZJH 标注状态追踪
};

}  // namespace om
