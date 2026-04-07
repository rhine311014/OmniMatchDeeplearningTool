// 20260330 ZJH 超参数搜索模块 — 网格搜索 / 随机搜索 / 贝叶斯优化
// 支持三种搜索策略：GridSearch（穷举笛卡尔积）、RandomSearch（均匀/对数采样）、
// BayesianSearch（简化高斯过程 + RBF 核 + Expected Improvement 采集函数）
// 每次试验记录参数组合和验证指标，自动追踪最优参数
module;

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <cassert>

export module om.engine.hypersearch;

export namespace om {

// 20260330 ZJH 数学常量：圆周率，用于高斯过程的正态分布 CDF/PDF 近似
constexpr float s_fHSPI = 3.14159265358979323846f;

// 20260330 ZJH 超参数搜索策略枚举
// GridSearch: 穷举所有离散组合（笛卡尔积），适合搜索空间小（<1000 组合）的场景
// RandomSearch: 在连续范围内随机采样，适合高维搜索空间（Bergstra & Bengio 2012）
// BayesianSearch: 用高斯过程建模目标函数，Expected Improvement 选下一点，适合评估代价高的场景
enum class SearchStrategy {
    GridSearch,       // 20260330 ZJH 网格搜索：穷举所有组合
    RandomSearch,     // 20260330 ZJH 随机搜索：在范围内随机采样
    BayesianSearch    // 20260330 ZJH 贝叶斯优化：高斯过程 + EI 采集函数
};

// 20260330 ZJH 搜索空间中单个超参数的定义
// Float: 连续浮点范围 [fMin, fMax]，可选对数尺度
// Int: 离散整数范围 [fMin, fMax]（采样后取整）
// Choice: 从候选值列表中选取
struct HyperParam {
    std::string strName;                // 20260330 ZJH 参数名称（如 "lr", "batch_size"）
    enum Type { Float, Int, Choice } eType;  // 20260330 ZJH 参数类型
    float fMin = 0.0f;                  // 20260330 ZJH Float/Int 的最小值
    float fMax = 1.0f;                  // 20260330 ZJH Float/Int 的最大值
    std::vector<float> vecChoices;      // 20260330 ZJH Choice 类型的候选值列表
    bool bLogScale = false;             // 20260330 ZJH 是否在对数空间采样（学习率常用 1e-5~1e-1）
};

// 20260330 ZJH 单次试验的完整结果记录
struct TrialResult {
    int nTrialId = 0;                          // 20260330 ZJH 试验序号（从 0 开始递增）
    std::map<std::string, float> mapParams;    // 20260330 ZJH 参数名 → 采样值
    float fValLoss = 0.0f;                     // 20260330 ZJH 验证集损失
    float fValAccuracy = 0.0f;                 // 20260330 ZJH 验证集准确率
    float fTrainLoss = 0.0f;                   // 20260330 ZJH 训练集损失
    int nEpochsTrained = 0;                    // 20260330 ZJH 实际训练的 epoch 数
    double dDurationSec = 0.0;                 // 20260330 ZJH 训练耗时（秒）
};

// 20260330 ZJH 搜索配置：控制搜索策略、最大试验数、提前终止等
struct SearchConfig {
    SearchStrategy eStrategy = SearchStrategy::RandomSearch;  // 20260330 ZJH 搜索策略
    int nMaxTrials = 20;              // 20260330 ZJH 最大试验次数
    int nEpochsPerTrial = 10;         // 20260330 ZJH 每次试验训练的 epoch 数（快速评估用）
    bool bEarlyStopBadTrials = true;  // 20260330 ZJH 是否提前终止差劲的试验
    float fEarlyStopThreshold = 2.0f; // 20260330 ZJH 损失超过当前最优 N 倍则终止该试验
};

// 20260330 ZJH 超参数搜索引擎
// 管理搜索空间定义、参数采样、试验结果记录、最优参数追踪
// 三种策略：
//   Grid: 递归笛卡尔积生成所有组合，顺序遍历
//   Random: 均匀采样或对数均匀采样
//   Bayesian: 维护 (参数, 损失) 观测对，用 GP 预测均值/方差，EI 选下一点
class HyperSearch {
public:
    // 20260330 ZJH 默认构造函数
    HyperSearch() = default;

    // 20260330 ZJH setConfig — 设置搜索配置
    // config: 搜索策略、最大试验数等
    void setConfig(const SearchConfig& config) {
        m_config = config;  // 20260330 ZJH 保存配置副本
    }

    // 20260330 ZJH addParam — 向搜索空间添加一个超参数维度
    // param: 超参数定义（名称、类型、范围/候选值）
    void addParam(const HyperParam& param) {
        m_vecParams.push_back(param);  // 20260330 ZJH 追加到参数列表
        m_bGridGenerated = false;      // 20260330 ZJH 参数变化后需重新生成网格
    }

    // 20260330 ZJH nextParams — 根据当前策略生成下一组超参数
    // 返回: 参数名 → 采样值的映射
    // Grid 模式下，遍历完所有组合后会抛出 runtime_error
    std::map<std::string, float> nextParams() {
        // 20260330 ZJH 根据搜索策略分发到对应的采样方法
        switch (m_config.eStrategy) {
            case SearchStrategy::GridSearch:
                return nextGridParams();      // 20260330 ZJH 网格搜索：取下一个笛卡尔积组合
            case SearchStrategy::RandomSearch:
                return nextRandomParams();    // 20260330 ZJH 随机搜索：在范围内随机采样
            case SearchStrategy::BayesianSearch:
                return nextBayesianParams();  // 20260330 ZJH 贝叶斯优化：GP + EI 选点
            default:
                return nextRandomParams();    // 20260330 ZJH 默认回退到随机搜索
        }
    }

    // 20260330 ZJH reportResult — 记录一次试验的结果
    // 贝叶斯搜索需要这些观测数据来更新高斯过程模型
    // result: 包含参数组合和对应的验证指标
    void reportResult(const TrialResult& result) {
        m_vecTrials.push_back(result);  // 20260330 ZJH 追加到历史记录
    }

    // 20260330 ZJH getBestTrial — 获取当前最优试验（按验证损失最小）
    // 返回: 验证损失最小的 TrialResult
    // 无试验时抛出 runtime_error
    TrialResult getBestTrial() const {
        if (m_vecTrials.empty()) {
            // 20260330 ZJH 尚未完成任何试验，无法返回最优结果
            throw std::runtime_error("HyperSearch::getBestTrial — no trials completed yet");
        }
        // 20260330 ZJH 遍历所有试验，找到 fValLoss 最小的一条
        const TrialResult* pBest = &m_vecTrials[0];  // 20260330 ZJH 初始化指向第一条记录
        for (size_t i = 1; i < m_vecTrials.size(); ++i) {
            if (m_vecTrials[i].fValLoss < pBest->fValLoss) {
                pBest = &m_vecTrials[i];  // 20260330 ZJH 发现更小的损失，更新最优指针
            }
        }
        return *pBest;  // 20260330 ZJH 返回最优试验的拷贝
    }

    // 20260330 ZJH getAllTrials — 获取所有历史试验记录
    // 返回: 按试验顺序排列的结果向量
    std::vector<TrialResult> getAllTrials() const {
        return m_vecTrials;  // 20260330 ZJH 返回副本
    }

    // 20260330 ZJH getConfig — 获取当前搜索配置（只读）
    const SearchConfig& getConfig() const {
        return m_config;  // 20260330 ZJH 返回配置引用
    }

    // 20260330 ZJH getSearchSpace — 获取当前搜索空间定义（只读）
    const std::vector<HyperParam>& getSearchSpace() const {
        return m_vecParams;  // 20260330 ZJH 返回参数列表引用
    }

private:
    // =========================================================
    // 20260330 ZJH Grid Search — 笛卡尔积网格搜索
    // =========================================================

    // 20260330 ZJH generateGrid — 递归生成所有参数组合的笛卡尔积
    // 对每个参数维度：
    //   Float/Int: 在 [fMin, fMax] 范围内等间距取 nGridSteps 个点
    //   Choice: 使用所有候选值
    // 结果存入 m_vecGridCombinations
    void generateGrid() {
        if (m_bGridGenerated) return;  // 20260330 ZJH 避免重复生成

        // 20260330 ZJH 为每个参数维度生成离散候选值列表
        std::vector<std::vector<float>> vecDimValues;  // 20260330 ZJH 各维度候选值
        for (const auto& param : m_vecParams) {
            std::vector<float> vecValues;  // 20260330 ZJH 当前维度的候选值
            if (param.eType == HyperParam::Choice) {
                // 20260330 ZJH Choice 类型：直接使用所有候选值
                vecValues = param.vecChoices;
            } else {
                // 20260330 ZJH Float/Int 类型：在范围内等间距生成 nGridSteps 个点
                int nSteps = s_nGridSteps;  // 20260330 ZJH 每维度的网格点数（默认 5）
                for (int i = 0; i < nSteps; ++i) {
                    // 20260330 ZJH 计算归一化位置 t ∈ [0, 1]
                    float fT = (nSteps == 1) ? 0.5f
                               : static_cast<float>(i) / static_cast<float>(nSteps - 1);
                    float fVal = 0.0f;  // 20260330 ZJH 当前采样值
                    if (param.bLogScale) {
                        // 20260330 ZJH 对数尺度采样：在 log 空间线性插值
                        // 用于学习率等跨数量级的参数（如 1e-5 ~ 1e-1）
                        float fLogMin = std::log(param.fMin + 1e-30f);  // 20260330 ZJH 防止 log(0)
                        float fLogMax = std::log(param.fMax + 1e-30f);  // 20260330 ZJH 防止 log(0)
                        fVal = std::exp(fLogMin + fT * (fLogMax - fLogMin));  // 20260330 ZJH 指数映射回线性空间
                    } else {
                        // 20260330 ZJH 线性尺度采样：直接线性插值
                        fVal = param.fMin + fT * (param.fMax - param.fMin);
                    }
                    // 20260330 ZJH Int 类型需四舍五入到整数
                    if (param.eType == HyperParam::Int) {
                        fVal = std::round(fVal);
                    }
                    vecValues.push_back(fVal);  // 20260330 ZJH 添加到候选值列表
                }
            }
            vecDimValues.push_back(vecValues);  // 20260330 ZJH 追加当前维度的候选值
        }

        // 20260330 ZJH 递归生成笛卡尔积
        m_vecGridCombinations.clear();  // 20260330 ZJH 清空旧组合
        std::map<std::string, float> mapCurrent;  // 20260330 ZJH 当前正在构建的组合
        cartesianProduct(vecDimValues, 0, mapCurrent);  // 20260330 ZJH 从第 0 维开始递归

        m_nGridIndex = 0;        // 20260330 ZJH 重置遍历索引
        m_bGridGenerated = true;  // 20260330 ZJH 标记已生成
    }

    // 20260330 ZJH cartesianProduct — 递归生成笛卡尔积
    // vecDimValues: 各维度的候选值列表
    // nDim: 当前处理的维度索引
    // mapCurrent: 当前正在构建的参数组合（递归传递）
    void cartesianProduct(const std::vector<std::vector<float>>& vecDimValues,
                          size_t nDim,
                          std::map<std::string, float>& mapCurrent) {
        // 20260330 ZJH 递归终止条件：所有维度已选定值
        if (nDim >= vecDimValues.size()) {
            m_vecGridCombinations.push_back(mapCurrent);  // 20260330 ZJH 保存完整组合
            return;
        }
        // 20260330 ZJH 遍历当前维度的所有候选值
        for (float fVal : vecDimValues[nDim]) {
            mapCurrent[m_vecParams[nDim].strName] = fVal;  // 20260330 ZJH 设置当前维度的值
            cartesianProduct(vecDimValues, nDim + 1, mapCurrent);  // 20260330 ZJH 递归处理下一维度
        }
    }

    // 20260330 ZJH nextGridParams — 取下一个网格组合
    // 首次调用时自动生成笛卡尔积；遍历完所有组合后抛异常
    std::map<std::string, float> nextGridParams() {
        generateGrid();  // 20260330 ZJH 确保网格已生成（幂等）
        if (m_nGridIndex >= static_cast<int>(m_vecGridCombinations.size())) {
            // 20260330 ZJH 所有网格组合已遍历完毕
            throw std::runtime_error("HyperSearch::nextGridParams — all grid combinations exhausted");
        }
        // 20260330 ZJH 返回当前索引对应的组合，并递增索引
        return m_vecGridCombinations[m_nGridIndex++];
    }

    // =========================================================
    // 20260330 ZJH Random Search — 随机搜索
    // =========================================================

    // 20260330 ZJH nextRandomParams — 在搜索空间内随机采样一组参数
    // Float: 均匀分布 U[fMin, fMax]（bLogScale=true 时在 log 空间采样）
    // Int: 均匀分布后四舍五入
    // Choice: 等概率选取一个候选值
    std::map<std::string, float> nextRandomParams() {
        std::map<std::string, float> mapResult;  // 20260330 ZJH 结果参数映射
        for (const auto& param : m_vecParams) {
            float fVal = 0.0f;  // 20260330 ZJH 当前参数的采样值
            if (param.eType == HyperParam::Choice) {
                // 20260330 ZJH Choice 类型：在候选值中均匀随机选取
                if (param.vecChoices.empty()) {
                    // 20260330 ZJH 空候选列表，回退到 0
                    fVal = 0.0f;
                } else {
                    std::uniform_int_distribution<int> dist(
                        0, static_cast<int>(param.vecChoices.size()) - 1);  // 20260330 ZJH 索引分布
                    int nIdx = dist(m_rng);  // 20260330 ZJH 随机索引
                    fVal = param.vecChoices[nIdx];  // 20260330 ZJH 取对应候选值
                }
            } else if (param.bLogScale) {
                // 20260330 ZJH 对数尺度采样：先在 [log(fMin), log(fMax)] 均匀采样，再取指数
                // 使得采样在对数空间均匀分布（如 1e-5 和 1e-3 之间的概率与 1e-3 和 1e-1 之间相同）
                float fLogMin = std::log(param.fMin + 1e-30f);  // 20260330 ZJH 防止 log(0)
                float fLogMax = std::log(param.fMax + 1e-30f);  // 20260330 ZJH 防止 log(0)
                std::uniform_real_distribution<float> dist(fLogMin, fLogMax);  // 20260330 ZJH log 空间均匀分布
                fVal = std::exp(dist(m_rng));  // 20260330 ZJH 指数映射回线性空间
                // 20260330 ZJH Int 类型需四舍五入
                if (param.eType == HyperParam::Int) {
                    fVal = std::round(fVal);
                }
            } else {
                // 20260330 ZJH 线性尺度均匀采样
                std::uniform_real_distribution<float> dist(param.fMin, param.fMax);  // 20260330 ZJH 均匀分布
                fVal = dist(m_rng);  // 20260330 ZJH 采样
                // 20260330 ZJH Int 类型需四舍五入
                if (param.eType == HyperParam::Int) {
                    fVal = std::round(fVal);
                }
            }
            mapResult[param.strName] = fVal;  // 20260330 ZJH 存入结果映射
        }
        return mapResult;  // 20260330 ZJH 返回完整参数组合
    }

    // =========================================================
    // 20260330 ZJH Bayesian Search — 简化高斯过程 + Expected Improvement
    // =========================================================
    //
    // 算法流程：
    // 1. 前 s_nInitRandomTrials 次试验使用随机搜索（冷启动，收集初始观测）
    // 2. 之后每次：
    //    a. 随机生成 s_nCandidatePoints 个候选参数组合
    //    b. 对每个候选点，用 GP 预测均值 μ 和方差 σ²
    //    c. 计算 Expected Improvement: EI = σ·φ(z) + (f_best - μ)·Φ(z)，其中 z = (f_best - μ)/σ
    //    d. 选取 EI 最大的候选点作为下一组参数

    // 20260330 ZJH nextBayesianParams — 贝叶斯优化选择下一组参数
    std::map<std::string, float> nextBayesianParams() {
        // 20260330 ZJH 冷启动阶段：观测数据不足时使用随机搜索
        if (static_cast<int>(m_vecTrials.size()) < s_nInitRandomTrials) {
            return nextRandomParams();  // 20260330 ZJH 前 N 次随机采样
        }

        // 20260330 ZJH 找到当前最优验证损失（用于 EI 计算）
        float fBestLoss = std::numeric_limits<float>::max();  // 20260330 ZJH 初始化为最大值
        for (const auto& trial : m_vecTrials) {
            if (trial.fValLoss < fBestLoss) {
                fBestLoss = trial.fValLoss;  // 20260330 ZJH 更新最优损失
            }
        }

        // 20260330 ZJH 构建观测数据矩阵用于 GP 预测
        // m_vecTrials 中已有的 (参数向量, 验证损失) 对
        int nObs = static_cast<int>(m_vecTrials.size());  // 20260330 ZJH 观测数量
        int nDim = static_cast<int>(m_vecParams.size());   // 20260330 ZJH 参数空间维度

        // 20260330 ZJH 将观测参数展平为 [nObs x nDim] 矩阵（行优先）
        std::vector<float> vecObsX(nObs * nDim);   // 20260330 ZJH 观测参数矩阵
        std::vector<float> vecObsY(nObs);           // 20260330 ZJH 观测损失向量
        for (int i = 0; i < nObs; ++i) {
            vecObsY[i] = m_vecTrials[i].fValLoss;  // 20260330 ZJH 第 i 次试验的验证损失
            for (int d = 0; d < nDim; ++d) {
                // 20260330 ZJH 按参数名查找对应值，归一化到 [0,1] 区间
                auto it = m_vecTrials[i].mapParams.find(m_vecParams[d].strName);
                float fRaw = (it != m_vecTrials[i].mapParams.end()) ? it->second : 0.0f;
                vecObsX[i * nDim + d] = normalizeParam(d, fRaw);  // 20260330 ZJH 归一化
            }
        }

        // 20260330 ZJH 预计算 RBF 核矩阵 K(X, X) + σ²_noise·I
        // K[i][j] = σ²_f · exp(-0.5 · ||x_i - x_j||² / l²)
        // 加上噪声方差 σ²_noise 保证矩阵正定（数值稳定）
        std::vector<float> vecK(nObs * nObs);  // 20260330 ZJH 核矩阵 [nObs x nObs]
        for (int i = 0; i < nObs; ++i) {
            for (int j = 0; j < nObs; ++j) {
                float fDist2 = 0.0f;  // 20260330 ZJH 两观测点之间的欧氏距离平方
                for (int d = 0; d < nDim; ++d) {
                    float fDiff = vecObsX[i * nDim + d] - vecObsX[j * nDim + d];
                    fDist2 += fDiff * fDiff;  // 20260330 ZJH 累加各维度差值平方
                }
                // 20260330 ZJH RBF 核值 = σ²_f · exp(-0.5 · dist² / l²)
                vecK[i * nObs + j] = s_fSignalVariance * std::exp(-0.5f * fDist2 / (s_fLengthScale * s_fLengthScale));
                // 20260330 ZJH 对角线加噪声方差，确保正定
                if (i == j) {
                    vecK[i * nObs + j] += s_fNoiseVariance;
                }
            }
        }

        // 20260330 ZJH Cholesky 分解 K = L·L^T（用于高效求解 K^{-1}·y）
        // 下三角矩阵 L 存储在 vecL 中
        std::vector<float> vecL(nObs * nObs, 0.0f);  // 20260330 ZJH Cholesky 下三角矩阵
        bool bCholeskyOk = choleskyDecompose(vecK, vecL, nObs);  // 20260330 ZJH 执行分解
        if (!bCholeskyOk) {
            // 20260330 ZJH Cholesky 分解失败（核矩阵不正定），回退到随机搜索
            return nextRandomParams();
        }

        // 20260330 ZJH 求解 α = K^{-1}·y，通过两次三角求解: L·z=y, L^T·α=z
        std::vector<float> vecAlpha(nObs);  // 20260330 ZJH α = K^{-1}·y
        choleskySolve(vecL, vecObsY, vecAlpha, nObs);  // 20260330 ZJH 三角求解

        // 20260330 ZJH 随机生成 s_nCandidatePoints 个候选点，计算 EI，选最优
        float fBestEI = -std::numeric_limits<float>::max();  // 20260330 ZJH 最高 EI 初始化为负无穷
        std::map<std::string, float> mapBestCandidate;  // 20260330 ZJH 最优候选参数组合

        for (int c = 0; c < s_nCandidatePoints; ++c) {
            // 20260330 ZJH 随机采样一个候选参数组合
            auto mapCandidate = nextRandomParams();

            // 20260330 ZJH 将候选点归一化为向量
            std::vector<float> vecXStar(nDim);  // 20260330 ZJH 归一化后的候选点
            for (int d = 0; d < nDim; ++d) {
                auto it = mapCandidate.find(m_vecParams[d].strName);
                float fRaw = (it != mapCandidate.end()) ? it->second : 0.0f;
                vecXStar[d] = normalizeParam(d, fRaw);  // 20260330 ZJH 归一化
            }

            // 20260330 ZJH 计算候选点与所有观测点的核向量 k(X, x*)
            std::vector<float> vecKStar(nObs);  // 20260330 ZJH k(X, x*) 向量
            for (int i = 0; i < nObs; ++i) {
                float fDist2 = 0.0f;  // 20260330 ZJH 候选点与第 i 个观测点的距离平方
                for (int d = 0; d < nDim; ++d) {
                    float fDiff = vecObsX[i * nDim + d] - vecXStar[d];
                    fDist2 += fDiff * fDiff;  // 20260330 ZJH 累加
                }
                // 20260330 ZJH RBF 核值
                vecKStar[i] = s_fSignalVariance * std::exp(-0.5f * fDist2 / (s_fLengthScale * s_fLengthScale));
            }

            // 20260330 ZJH GP 预测均值: μ(x*) = k(X, x*)^T · α
            float fMu = 0.0f;  // 20260330 ZJH 预测均值
            for (int i = 0; i < nObs; ++i) {
                fMu += vecKStar[i] * vecAlpha[i];  // 20260330 ZJH 内积
            }

            // 20260330 ZJH GP 预测方差: σ²(x*) = k(x*, x*) - k^T · K^{-1} · k
            // 其中 k(x*, x*) = s_fSignalVariance（自核值）
            // K^{-1}·k 通过 Cholesky: L·v = k, σ² = k** - v^T·v
            std::vector<float> vecV(nObs);  // 20260330 ZJH v = L^{-1} · k*
            choleskyForwardSolve(vecL, vecKStar, vecV, nObs);  // 20260330 ZJH 前向求解

            float fVarReduce = 0.0f;  // 20260330 ZJH v^T·v（方差缩减量）
            for (int i = 0; i < nObs; ++i) {
                fVarReduce += vecV[i] * vecV[i];  // 20260330 ZJH 内积
            }
            // 20260330 ZJH 预测方差 = 先验方差 - 方差缩减，钳位到非负
            float fVar = std::max(0.0f, s_fSignalVariance - fVarReduce);
            float fSigma = std::sqrt(fVar + 1e-10f);  // 20260330 ZJH 预测标准差（加 eps 防止除零）

            // 20260330 ZJH 计算 Expected Improvement (EI)
            // EI = σ·φ(z) + (f_best - μ)·Φ(z)
            // 其中 z = (f_best - μ) / σ
            // φ(z): 标准正态 PDF
            // Φ(z): 标准正态 CDF
            float fImprovement = fBestLoss - fMu;  // 20260330 ZJH f_best - μ
            float fZ = fImprovement / fSigma;       // 20260330 ZJH 标准化的改善量
            float fPhi = normalPdf(fZ);              // 20260330 ZJH 标准正态 PDF
            float fCdfZ = normalCdf(fZ);             // 20260330 ZJH 标准正态 CDF
            float fEI = fSigma * fPhi + fImprovement * fCdfZ;  // 20260330 ZJH Expected Improvement

            // 20260330 ZJH 更新最优候选点
            if (fEI > fBestEI) {
                fBestEI = fEI;                      // 20260330 ZJH 更新最高 EI
                mapBestCandidate = mapCandidate;    // 20260330 ZJH 记录对应参数组合
            }
        }

        return mapBestCandidate;  // 20260330 ZJH 返回 EI 最高的候选参数组合
    }

    // =========================================================
    // 20260330 ZJH 辅助函数
    // =========================================================

    // 20260330 ZJH normalizeParam — 将参数值归一化到 [0, 1] 区间
    // 对数尺度参数在 log 空间归一化，线性尺度直接线性归一化
    // nDimIdx: 参数维度索引
    // fValue: 原始参数值
    // 返回: 归一化后的值 ∈ [0, 1]
    float normalizeParam(int nDimIdx, float fValue) const {
        const auto& param = m_vecParams[nDimIdx];  // 20260330 ZJH 获取参数定义
        if (param.eType == HyperParam::Choice) {
            // 20260330 ZJH Choice 类型：找到在候选列表中的索引，归一化为 [0,1]
            if (param.vecChoices.size() <= 1) return 0.0f;  // 20260330 ZJH 单值或空列表
            for (size_t i = 0; i < param.vecChoices.size(); ++i) {
                if (std::abs(param.vecChoices[i] - fValue) < 1e-6f) {
                    // 20260330 ZJH 找到匹配值，按索引归一化
                    return static_cast<float>(i) / static_cast<float>(param.vecChoices.size() - 1);
                }
            }
            return 0.0f;  // 20260330 ZJH 未匹配到（不应发生），返回 0
        }
        if (param.bLogScale) {
            // 20260330 ZJH 对数尺度归一化
            float fLogMin = std::log(param.fMin + 1e-30f);  // 20260330 ZJH 防止 log(0)
            float fLogMax = std::log(param.fMax + 1e-30f);  // 20260330 ZJH 防止 log(0)
            float fLogVal = std::log(fValue + 1e-30f);      // 20260330 ZJH 参数值取对数
            float fRange = fLogMax - fLogMin;                // 20260330 ZJH log 空间范围
            // 20260330 ZJH 避免除零
            return (fRange > 1e-10f) ? (fLogVal - fLogMin) / fRange : 0.0f;
        }
        // 20260330 ZJH 线性尺度归一化
        float fRange = param.fMax - param.fMin;  // 20260330 ZJH 参数范围
        return (fRange > 1e-10f) ? (fValue - param.fMin) / fRange : 0.0f;  // 20260330 ZJH 线性归一化
    }

    // 20260330 ZJH normalPdf — 标准正态分布概率密度函数 φ(x) = exp(-x²/2) / √(2π)
    static float normalPdf(float fX) {
        return std::exp(-0.5f * fX * fX) / std::sqrt(2.0f * s_fHSPI);  // 20260330 ZJH 高斯钟形曲线
    }

    // 20260330 ZJH normalCdf — 标准正态分布累积分布函数 Φ(x) 的近似
    // 使用 erfc 函数: Φ(x) = 0.5 · erfc(-x/√2)
    // 精度优于 Abramowitz & Stegun 多项式近似
    static float normalCdf(float fX) {
        return 0.5f * std::erfc(-fX / std::sqrt(2.0f));  // 20260330 ZJH erfc 近似
    }

    // 20260330 ZJH choleskyDecompose — Cholesky 分解 A = L·L^T
    // vecA: 输入对称正定矩阵 [n x n]（行优先，会被读取但不修改）
    // vecL: 输出下三角矩阵 [n x n]
    // n: 矩阵维度
    // 返回: 分解是否成功（对角元素必须为正）
    static bool choleskyDecompose(const std::vector<float>& vecA,
                                  std::vector<float>& vecL,
                                  int n) {
        // 20260330 ZJH 逐行逐列计算下三角矩阵 L
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                float fSum = 0.0f;  // 20260330 ZJH 累加 L[i][k]*L[j][k] 的结果
                for (int k = 0; k < j; ++k) {
                    fSum += vecL[i * n + k] * vecL[j * n + k];  // 20260330 ZJH 内积累加
                }
                if (i == j) {
                    // 20260330 ZJH 对角元素: L[i][i] = sqrt(A[i][i] - sum(L[i][k]²))
                    float fDiag = vecA[i * n + i] - fSum;  // 20260330 ZJH 对角余量
                    if (fDiag <= 0.0f) {
                        return false;  // 20260330 ZJH 矩阵不正定，分解失败
                    }
                    vecL[i * n + j] = std::sqrt(fDiag);  // 20260330 ZJH 对角元素
                } else {
                    // 20260330 ZJH 非对角元素: L[i][j] = (A[i][j] - sum) / L[j][j]
                    float fDenom = vecL[j * n + j];  // 20260330 ZJH 对角元素（除数）
                    if (std::abs(fDenom) < 1e-30f) {
                        return false;  // 20260330 ZJH 对角元素过小，数值不稳定
                    }
                    vecL[i * n + j] = (vecA[i * n + j] - fSum) / fDenom;  // 20260330 ZJH 非对角元素
                }
            }
        }
        return true;  // 20260330 ZJH 分解成功
    }

    // 20260330 ZJH choleskyForwardSolve — 前向替代求解 L·x = b
    // vecL: 下三角矩阵 [n x n]
    // vecB: 右端向量 [n]
    // vecX: 输出解向量 [n]
    // n: 维度
    static void choleskyForwardSolve(const std::vector<float>& vecL,
                                     const std::vector<float>& vecB,
                                     std::vector<float>& vecX,
                                     int n) {
        for (int i = 0; i < n; ++i) {
            float fSum = 0.0f;  // 20260330 ZJH 累加 L[i][j]*x[j]
            for (int j = 0; j < i; ++j) {
                fSum += vecL[i * n + j] * vecX[j];  // 20260330 ZJH 前向累加
            }
            float fDenom = vecL[i * n + i];  // 20260330 ZJH 对角元素
            // 20260330 ZJH x[i] = (b[i] - sum) / L[i][i]
            vecX[i] = (std::abs(fDenom) > 1e-30f) ? (vecB[i] - fSum) / fDenom : 0.0f;
        }
    }

    // 20260330 ZJH choleskyBackwardSolve — 后向替代求解 L^T·x = b
    // vecL: 下三角矩阵 [n x n]（使用其转置）
    // vecB: 右端向量 [n]
    // vecX: 输出解向量 [n]
    // n: 维度
    static void choleskyBackwardSolve(const std::vector<float>& vecL,
                                      const std::vector<float>& vecB,
                                      std::vector<float>& vecX,
                                      int n) {
        for (int i = n - 1; i >= 0; --i) {
            float fSum = 0.0f;  // 20260330 ZJH 累加 L[j][i]*x[j]（L^T 的行即 L 的列）
            for (int j = i + 1; j < n; ++j) {
                fSum += vecL[j * n + i] * vecX[j];  // 20260330 ZJH 后向累加
            }
            float fDenom = vecL[i * n + i];  // 20260330 ZJH 对角元素
            // 20260330 ZJH x[i] = (b[i] - sum) / L[i][i]
            vecX[i] = (std::abs(fDenom) > 1e-30f) ? (vecB[i] - fSum) / fDenom : 0.0f;
        }
    }

    // 20260330 ZJH choleskySolve — 求解 K·x = b（通过 L·L^T 分解）
    // 步骤: L·z = b（前向替代），L^T·x = z（后向替代）
    static void choleskySolve(const std::vector<float>& vecL,
                              const std::vector<float>& vecB,
                              std::vector<float>& vecX,
                              int n) {
        std::vector<float> vecZ(n);  // 20260330 ZJH 中间变量 z = L^{-1}·b
        choleskyForwardSolve(vecL, vecB, vecZ, n);   // 20260330 ZJH 前向求解
        choleskyBackwardSolve(vecL, vecZ, vecX, n);   // 20260330 ZJH 后向求解
    }

    // =========================================================
    // 20260330 ZJH 成员变量
    // =========================================================

    SearchConfig m_config;  // 20260330 ZJH 搜索配置

    std::vector<HyperParam> m_vecParams;   // 20260330 ZJH 搜索空间参数定义列表
    std::vector<TrialResult> m_vecTrials;  // 20260330 ZJH 所有已完成试验的记录

    // 20260330 ZJH Grid Search 相关
    std::vector<std::map<std::string, float>> m_vecGridCombinations;  // 20260330 ZJH 笛卡尔积所有组合
    int m_nGridIndex = 0;       // 20260330 ZJH 当前遍历到的组合索引
    bool m_bGridGenerated = false;  // 20260330 ZJH 网格是否已生成

    // 20260330 ZJH 随机数生成器（线程局部安全由调用方保证）
    std::mt19937 m_rng{42};  // 20260330 ZJH 默认种子 42，保证可复现

    // 20260330 ZJH GP 超参数（简化为固定值，实际应通过最大似然估计优化）
    static constexpr float s_fLengthScale = 0.3f;      // 20260330 ZJH RBF 核的长度尺度 l
    static constexpr float s_fSignalVariance = 1.0f;    // 20260330 ZJH RBF 核的信号方差 σ²_f
    static constexpr float s_fNoiseVariance = 0.01f;    // 20260330 ZJH 观测噪声方差 σ²_noise

    // 20260330 ZJH 贝叶斯搜索常量
    static constexpr int s_nInitRandomTrials = 5;    // 20260330 ZJH 冷启动随机试验次数
    static constexpr int s_nCandidatePoints = 100;   // 20260330 ZJH 每次 EI 评估的候选点数
    static constexpr int s_nGridSteps = 5;           // 20260330 ZJH 网格搜索每维度的离散步数
};

}  // namespace om
