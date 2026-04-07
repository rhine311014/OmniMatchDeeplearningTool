// 20260330 ZJH 生产监控引擎 — 对标 Keyence 检测余裕监控 + 漂移检测 + SPC集成
// 三大模块:
//   1. MarginTracker — 检测余裕跟踪与趋势告警
//   2. DriftDetector — Page-Hinkley 模型漂移在线检测
//   3. SPCAnalyzer  — X-bar 控制图 + Western Electric 规则 + Cpk + Pareto
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cstdint>

export module om.engine.production_monitor;

export namespace om {

// =========================================================
// 检测余裕 (Detection Margin)
// =========================================================

// 20260330 ZJH DetectionMargin — 单次判定的余裕记录
// 量化每次判定距离阈值的余裕，用于趋势图展示稳定性
struct DetectionMargin {
    float fScore;           // 20260330 ZJH 模型输出分数（置信度或异常分数）
    float fThreshold;       // 20260330 ZJH 判定阈值（OK/NG 分界线）
    float fMargin;          // 20260330 ZJH 绝对余裕 = |score - threshold|
    float fMarginPercent;   // 20260330 ZJH 百分比余裕 = margin / threshold * 100
    bool bPass;             // 20260330 ZJH 判定结果：true=OK, false=NG
    int64_t nTimestamp;     // 20260330 ZJH Unix 时间戳（毫秒）
};

// 20260330 ZJH MarginTracker — 检测余裕跟踪器
// 持续记录每次检测的余裕数据，提供统计分析和趋势告警
// 当余裕持续缩小时（得分逼近阈值），提前发出预警，防止漏检
class MarginTracker {
public:
    // 20260330 ZJH addResult — 添加一条检测余裕记录
    // margin: 包含分数、阈值、余裕等完整信息的结构体
    void addResult(const DetectionMargin& margin) {
        // 20260330 ZJH 超过最大历史容量时移除最早记录，保持内存受控
        if (static_cast<int>(m_vecHistory.size()) >= s_nMaxHistory) {
            m_vecHistory.erase(m_vecHistory.begin());  // 20260330 ZJH 移除最旧记录
        }
        m_vecHistory.push_back(margin);  // 20260330 ZJH 追加最新记录
    }

    // 20260330 ZJH getMeanMargin — 计算所有记录的平均余裕
    // 返回: 平均绝对余裕值，无记录时返回 0
    float getMeanMargin() const {
        if (m_vecHistory.empty()) return 0.0f;  // 20260330 ZJH 空历史返回 0
        float fSum = 0.0f;  // 20260330 ZJH 累加器
        for (const auto& entry : m_vecHistory) {
            fSum += entry.fMargin;  // 20260330 ZJH 累加每条记录的绝对余裕
        }
        return fSum / static_cast<float>(m_vecHistory.size());  // 20260330 ZJH 均值
    }

    // 20260330 ZJH getMinMargin — 获取历史最小余裕（最危险的一次判定）
    // 返回: 最小绝对余裕值，无记录时返回 0
    float getMinMargin() const {
        if (m_vecHistory.empty()) return 0.0f;  // 20260330 ZJH 空历史返回 0
        float fMin = m_vecHistory[0].fMargin;  // 20260330 ZJH 初始化为第一条记录
        for (size_t i = 1; i < m_vecHistory.size(); ++i) {
            if (m_vecHistory[i].fMargin < fMin) {
                fMin = m_vecHistory[i].fMargin;  // 20260330 ZJH 更新最小值
            }
        }
        return fMin;  // 20260330 ZJH 返回历史最小余裕
    }

    // 20260330 ZJH getStdDevMargin — 计算余裕的标准差（波动性）
    // 标准差大说明检测不稳定，可能有周期性干扰
    // 返回: 余裕标准差，不足 2 条记录时返回 0
    float getStdDevMargin() const {
        if (m_vecHistory.size() < 2) return 0.0f;  // 20260330 ZJH 不足 2 条无法计算
        float fMean = getMeanMargin();  // 20260330 ZJH 先计算均值
        float fSumSqDev = 0.0f;  // 20260330 ZJH 偏差平方和累加器
        for (const auto& entry : m_vecHistory) {
            float fDev = entry.fMargin - fMean;  // 20260330 ZJH 偏差
            fSumSqDev += fDev * fDev;  // 20260330 ZJH 累加偏差平方
        }
        // 20260330 ZJH 样本标准差（除以 n-1）
        return std::sqrt(fSumSqDev / static_cast<float>(m_vecHistory.size() - 1));
    }

    // 20260330 ZJH getConsecutiveNarrowCount — 获取连续窄余裕的次数
    // 从最新记录向前计算，连续多少次余裕低于阈值
    // fNarrowThreshold: 窄余裕判定阈值（默认 0.1，即余裕 < 10% 判为窄）
    // 返回: 连续窄余裕计数
    int getConsecutiveNarrowCount(float fNarrowThreshold = 0.1f) const {
        int nCount = 0;  // 20260330 ZJH 连续计数器
        // 20260330 ZJH 从最新记录向前遍历
        for (int i = static_cast<int>(m_vecHistory.size()) - 1; i >= 0; --i) {
            if (m_vecHistory[i].fMarginPercent < fNarrowThreshold * 100.0f) {
                ++nCount;  // 20260330 ZJH 该记录属于窄余裕，计数+1
            } else {
                break;  // 20260330 ZJH 遇到非窄余裕，中断连续
            }
        }
        return nCount;  // 20260330 ZJH 返回连续窄余裕次数
    }

    // 20260330 ZJH getMarginHistory — 获取最近 N 条余裕值（用于UI趋势图）
    // nLastN: 返回最近的记录条数（默认 100）
    // 返回: 余裕值序列，按时间正序排列
    std::vector<float> getMarginHistory(int nLastN = 100) const {
        std::vector<float> vecResult;  // 20260330 ZJH 输出序列
        int nStart = static_cast<int>(m_vecHistory.size()) - nLastN;  // 20260330 ZJH 起始索引
        if (nStart < 0) nStart = 0;  // 20260330 ZJH 不足 N 条时从头开始
        vecResult.reserve(m_vecHistory.size() - nStart);  // 20260330 ZJH 预分配
        for (int i = nStart; i < static_cast<int>(m_vecHistory.size()); ++i) {
            vecResult.push_back(m_vecHistory[i].fMargin);  // 20260330 ZJH 逐条提取余裕值
        }
        return vecResult;  // 20260330 ZJH 返回按时间正序的余裕序列
    }

    // 20260330 ZJH getScoreHistory — 获取最近 N 条分数值（用于UI趋势图叠加显示）
    // nLastN: 返回最近的记录条数（默认 100）
    // 返回: 分数值序列，按时间正序排列
    std::vector<float> getScoreHistory(int nLastN = 100) const {
        std::vector<float> vecResult;  // 20260330 ZJH 输出序列
        int nStart = static_cast<int>(m_vecHistory.size()) - nLastN;  // 20260330 ZJH 起始索引
        if (nStart < 0) nStart = 0;  // 20260330 ZJH 不足 N 条时从头开始
        vecResult.reserve(m_vecHistory.size() - nStart);  // 20260330 ZJH 预分配
        for (int i = nStart; i < static_cast<int>(m_vecHistory.size()); ++i) {
            vecResult.push_back(m_vecHistory[i].fScore);  // 20260330 ZJH 逐条提取分数
        }
        return vecResult;  // 20260330 ZJH 返回按时间正序的分数序列
    }

    // 20260330 ZJH isMarginDeclining — 检测余裕是否持续下降（线性回归斜率为负）
    // 使用最近 nWindowSize 条记录做最小二乘线性回归，若斜率显著为负则告警
    // nWindowSize: 分析窗口大小（默认 50）
    // 返回: true 表示余裕在下降趋势中
    bool isMarginDeclining(int nWindowSize = 50) const {
        int nN = static_cast<int>(m_vecHistory.size());  // 20260330 ZJH 总记录数
        if (nN < nWindowSize) return false;  // 20260330 ZJH 数据不足，无法判定趋势

        // 20260330 ZJH 取最近 nWindowSize 条记录做线性回归
        int nStart = nN - nWindowSize;  // 20260330 ZJH 窗口起始索引
        // 20260330 ZJH 最小二乘法：y = a*x + b，计算斜率 a
        // a = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
        float fSumX = 0.0f, fSumY = 0.0f;  // 20260330 ZJH x 和 y 的累加
        float fSumXY = 0.0f, fSumX2 = 0.0f;  // 20260330 ZJH xy 和 x² 的累加
        for (int i = 0; i < nWindowSize; ++i) {
            float fX = static_cast<float>(i);  // 20260330 ZJH 时间序号作为 x
            float fY = m_vecHistory[nStart + i].fMargin;  // 20260330 ZJH 余裕值作为 y
            fSumX += fX;      // 20260330 ZJH 累加 x
            fSumY += fY;      // 20260330 ZJH 累加 y
            fSumXY += fX * fY;  // 20260330 ZJH 累加 xy
            fSumX2 += fX * fX;  // 20260330 ZJH 累加 x²
        }
        float fW = static_cast<float>(nWindowSize);  // 20260330 ZJH 窗口大小浮点
        float fDenom = fW * fSumX2 - fSumX * fSumX;  // 20260330 ZJH 分母
        if (std::abs(fDenom) < 1e-12f) return false;  // 20260330 ZJH 分母为零（退化情况）
        float fSlope = (fW * fSumXY - fSumX * fSumY) / fDenom;  // 20260330 ZJH 回归斜率

        // 20260330 ZJH 斜率为负且绝对值超过均值的 1% 视为显著下降
        float fMean = getMeanMargin();  // 20260330 ZJH 当前均值作为基准
        float fSlopeThreshold = fMean * 0.01f;  // 20260330 ZJH 显著性阈值: 均值的 1%
        return fSlope < -fSlopeThreshold;  // 20260330 ZJH 返回是否下降
    }

    // 20260330 ZJH getWarningMessage — 生成当前状态的告警文本
    // 根据余裕统计自动判断告警级别，返回人类可读的告警消息
    // 返回: 告警字符串，无告警时返回空字符串
    std::string getWarningMessage() const {
        if (m_vecHistory.empty()) return "";  // 20260330 ZJH 无数据无告警

        // 20260330 ZJH 检查连续窄余裕
        int nNarrow = getConsecutiveNarrowCount(0.1f);  // 20260330 ZJH 余裕<10% 的连续次数
        if (nNarrow >= 10) {
            // 20260330 ZJH 连续 10 次以上窄余裕，严重告警
            return "[CRITICAL] " + std::to_string(nNarrow)
                + " consecutive narrow margins (<10%). Model re-training strongly recommended.";
        }
        if (nNarrow >= 5) {
            // 20260330 ZJH 连续 5 次以上窄余裕，警告
            return "[WARNING] " + std::to_string(nNarrow)
                + " consecutive narrow margins (<10%). Monitor closely.";
        }

        // 20260330 ZJH 检查趋势下降
        if (isMarginDeclining(50)) {
            return "[WARNING] Detection margin is declining over the last 50 samples. Possible model drift.";
        }

        // 20260330 ZJH 检查最小余裕
        float fMin = getMinMargin();  // 20260330 ZJH 历史最小余裕
        if (fMin < 0.01f) {
            return "[INFO] Historical minimum margin is very small ("
                + std::to_string(fMin) + "). Borderline detection occurred.";
        }

        return "";  // 20260330 ZJH 无告警
    }

    // 20260330 ZJH clear — 清空全部历史记录
    void clear() {
        m_vecHistory.clear();  // 20260330 ZJH 释放所有记录
    }

    // 20260330 ZJH getCount — 获取当前历史记录总数
    // 返回: 记录条数
    int getCount() const {
        return static_cast<int>(m_vecHistory.size());  // 20260330 ZJH 返回容器大小
    }

private:
    std::vector<DetectionMargin> m_vecHistory;  // 20260330 ZJH 检测余裕历史记录
    static constexpr int s_nMaxHistory = 10000;  // 20260330 ZJH 最大历史容量（防止内存无限增长）
};

// =========================================================
// 模型漂移检测 (Model Drift Detection)
// =========================================================

// 20260330 ZJH DriftAlert — 漂移告警结构体
// 描述漂移检测的结果、严重程度和建议操作
struct DriftAlert {
    // 20260330 ZJH 告警级别: None=正常, Warning=轻微漂移, Critical=严重漂移
    enum Level { None, Warning, Critical } eLevel;
    std::string strMessage;       // 20260330 ZJH 告警描述文本
    float fDriftScore;            // 20260330 ZJH 漂移程度 [0,1]，0=无漂移，1=完全漂移
    int nSamplesSinceLast;        // 20260330 ZJH 距上次报警经过的样本数
    std::string strSuggestion;    // 20260330 ZJH 建议操作（如"重新训练"）
};

// 20260330 ZJH DriftDetector — Page-Hinkley 在线漂移检测器
// Page-Hinkley Test 是经典的在线均值偏移检测算法:
//   - 维护累积和 U_t = Σ(x_i - mean - delta)
//   - 计算 m_t = min(U_1...U_t)
//   - 当 U_t - m_t > threshold 时判定漂移
// 优点: 内存 O(1)、计算 O(1)/样本、灵敏度可调
class DriftDetector {
public:
    // 20260330 ZJH addScore — 添加一个新的检测分数
    // fScore: 模型输出分数（置信度或异常分数）
    void addScore(float fScore) {
        m_nCount++;  // 20260330 ZJH 样本计数+1
        // 20260330 ZJH 增量更新在线均值: mean_new = mean_old + (x - mean_old) / n
        m_fMean += (fScore - m_fMean) / static_cast<float>(m_nCount);

        // 20260330 ZJH Burn-in 阶段: 同步更新参考均值（用于累积和基准）
        if (m_nCount <= m_nMinSamples) {
            m_fReferenceMean = m_fMean;  // 20260330 ZJH burn-in 期间持续跟踪均值
        }
        // 20260330 ZJH burn-in 结束后冻结 m_fReferenceMean，使用它计算累积和
        // 这样 m_fMean 继续漂移不会污染累积和基准

        // 20260330 ZJH 更新 Page-Hinkley 累积和
        // delta(0.005) 是容忍的最小偏移量，低于此值不触发
        m_fSum += (fScore - m_fReferenceMean - 0.005f);
        // 20260330 ZJH 跟踪累积和的历史最小值
        if (m_fSum < m_fMinValue) {
            m_fMinValue = m_fSum;  // 20260330 ZJH 更新最小值
        }
    }

    // 20260330 ZJH checkDrift — 检查当前是否发生漂移
    // 返回: DriftAlert 结构体，包含告警级别和建议
    DriftAlert checkDrift() const {
        DriftAlert alert;  // 20260330 ZJH 输出告警结构体
        alert.eLevel = DriftAlert::None;  // 20260330 ZJH 默认无告警
        alert.fDriftScore = 0.0f;  // 20260330 ZJH 默认漂移程度 0
        alert.nSamplesSinceLast = m_nCount;  // 20260330 ZJH 距上次报警的样本数
        alert.strMessage = "No drift detected.";  // 20260330 ZJH 默认消息
        alert.strSuggestion = "";  // 20260330 ZJH 默认无建议

        // 20260330 ZJH 未达到最小样本数时不进行检测（避免噪声触发）
        if (m_nCount < m_nMinSamples) {
            alert.strMessage = "Insufficient samples (" + std::to_string(m_nCount)
                + "/" + std::to_string(m_nMinSamples) + ").";
            return alert;  // 20260330 ZJH 样本不足，返回默认
        }

        // 20260330 ZJH Page-Hinkley 统计量: PH = U_t - m_t
        float fPH = m_fSum - m_fMinValue;  // 20260330 ZJH 当前 PH 值

        // 20260330 ZJH 归一化漂移分数到 [0,1] 区间
        // 使用 sigmoid 风格映射: score = PH / (PH + threshold)
        float fDriftScore = fPH / (fPH + m_fThreshold + 1e-9f);  // 20260330 ZJH 防除零
        alert.fDriftScore = std::clamp(fDriftScore, 0.0f, 1.0f);  // 20260330 ZJH 裁剪到 [0,1]

        // 20260330 ZJH 根据 PH 值与阈值的关系判定级别
        if (fPH > m_fThreshold * 2.0f) {
            // 20260330 ZJH PH 值超过阈值 2 倍: 严重漂移
            alert.eLevel = DriftAlert::Critical;
            alert.strMessage = "Critical drift detected! PH=" + std::to_string(fPH)
                + " >> threshold=" + std::to_string(m_fThreshold);
            alert.strSuggestion = "Immediate model re-training required. "
                "Production scores have significantly shifted from training distribution.";
        } else if (fPH > m_fThreshold) {
            // 20260330 ZJH PH 值超过阈值: 轻微漂移
            alert.eLevel = DriftAlert::Warning;
            alert.strMessage = "Drift warning: PH=" + std::to_string(fPH)
                + " > threshold=" + std::to_string(m_fThreshold);
            alert.strSuggestion = "Schedule model re-training. "
                "Collect recent production images for fine-tuning dataset.";
        }

        return alert;  // 20260330 ZJH 返回告警结果
    }

    // 20260330 ZJH setThreshold — 设置漂移判定阈值
    // fThreshold: PH 统计量超过此值判定为漂移（越小越灵敏）
    void setThreshold(float fThreshold) {
        m_fThreshold = fThreshold;  // 20260330 ZJH 更新阈值
    }

    // 20260330 ZJH setMinSamples — 设置开始检测所需的最少样本数
    // nMin: 最少样本数（默认 30），低于此数不进行漂移检测
    void setMinSamples(int nMin) {
        m_nMinSamples = nMin;  // 20260330 ZJH 更新最小样本数
    }

    // 20260330 ZJH reset — 重置检测器状态（换模型或更新阈值后调用）
    void reset() {
        m_fSum = 0.0f;            // 20260330 ZJH 累积和归零
        m_fMean = 0.0f;           // 20260330 ZJH 在线均值归零
        m_fReferenceMean = 0.0f;  // 20260330 ZJH 参考均值归零
        m_fMinValue = 1e9f;       // 20260330 ZJH 最小值重置为极大值
        m_nCount = 0;             // 20260330 ZJH 样本计数归零
    }

private:
    // 20260330 ZJH Page-Hinkley 内部状态
    float m_fSum = 0.0f;            // 20260330 ZJH 累积和 U_t = Σ(x_i - refMean - delta)
    float m_fMean = 0.0f;           // 20260330 ZJH 在线增量均值（持续更新）
    float m_fReferenceMean = 0.0f;  // 20260330 ZJH 参考均值（burn-in 后冻结，用于累积和基准）
    float m_fMinValue = 1e9f;       // 20260330 ZJH 累积和历史最小值 m_t = min(U_1..U_t)
    float m_fThreshold = 0.05f;     // 20260330 ZJH 漂移判定阈值（PH > threshold 则告警）
    int m_nCount = 0;               // 20260330 ZJH 已接收样本总数
    int m_nMinSamples = 30;         // 20260330 ZJH 最少样本数后开始检测
};

// =========================================================
// SPC 统计过程控制 (Statistical Process Control)
// =========================================================

// 20260330 ZJH SPCChart — 控制图数据结构
// 包含测量值序列、控制限和失控点信息
struct SPCChart {
    std::vector<float> vecValues;     // 20260330 ZJH 测量值序列（子组均值或个值）
    float fUCL = 0.0f;               // 20260330 ZJH 上控制限 (Upper Control Limit)
    float fLCL = 0.0f;               // 20260330 ZJH 下控制限 (Lower Control Limit)
    float fCL = 0.0f;                // 20260330 ZJH 中心线 (Center Line)，即总均值
    int nOutOfControl = 0;            // 20260330 ZJH 失控点数（超出控制限的点）
    std::vector<int> vecViolations;   // 20260330 ZJH 失控点索引列表
    std::string strRuleName;          // 20260330 ZJH 违反的 SPC 规则名称
};

// 20260330 ZJH SPCAnalyzer — SPC 统计过程控制分析器
// 提供 X-bar 控制图、Western Electric 规则检查、帕累托分析、Cpk 计算
class SPCAnalyzer {
public:
    // 20260330 ZJH computeXBarChart — 计算 X-bar 控制图
    // 将分数序列按子组大小分组，计算各子组均值，建立控制限
    // vecScores: 原始检测分数序列
    // nSubgroupSize: 子组大小（默认 5）
    // 返回: SPCChart 结构体，包含子组均值序列和 3σ 控制限
    SPCChart computeXBarChart(const std::vector<float>& vecScores, int nSubgroupSize = 5) {
        SPCChart chart;  // 20260330 ZJH 输出控制图

        // 20260330 ZJH 数据量不足一个子组时返回空图
        if (static_cast<int>(vecScores.size()) < nSubgroupSize || nSubgroupSize < 2) {
            return chart;
        }

        // 20260330 ZJH 按子组大小分组，计算每组均值
        int nNumSubgroups = static_cast<int>(vecScores.size()) / nSubgroupSize;  // 20260330 ZJH 完整子组数
        chart.vecValues.reserve(nNumSubgroups);  // 20260330 ZJH 预分配子组均值容器

        std::vector<float> vecRanges;  // 20260330 ZJH 各子组极差（用于估计 σ）
        vecRanges.reserve(nNumSubgroups);  // 20260330 ZJH 预分配极差容器

        for (int g = 0; g < nNumSubgroups; ++g) {
            float fSubgroupSum = 0.0f;  // 20260330 ZJH 子组内累加器
            float fMin = vecScores[g * nSubgroupSize];  // 20260330 ZJH 子组最小值
            float fMax = vecScores[g * nSubgroupSize];  // 20260330 ZJH 子组最大值
            for (int j = 0; j < nSubgroupSize; ++j) {
                float fVal = vecScores[g * nSubgroupSize + j];  // 20260330 ZJH 取子组内第 j 个值
                fSubgroupSum += fVal;  // 20260330 ZJH 累加
                if (fVal < fMin) fMin = fVal;  // 20260330 ZJH 更新子组最小值
                if (fVal > fMax) fMax = fVal;  // 20260330 ZJH 更新子组最大值
            }
            float fMean = fSubgroupSum / static_cast<float>(nSubgroupSize);  // 20260330 ZJH 子组均值
            chart.vecValues.push_back(fMean);  // 20260330 ZJH 记录子组均值
            vecRanges.push_back(fMax - fMin);  // 20260330 ZJH 记录子组极差
        }

        // 20260330 ZJH 计算中心线 CL = 所有子组均值的总平均
        float fGrandMean = 0.0f;  // 20260330 ZJH 总均值累加器
        for (float fVal : chart.vecValues) {
            fGrandMean += fVal;  // 20260330 ZJH 累加子组均值
        }
        fGrandMean /= static_cast<float>(nNumSubgroups);  // 20260330 ZJH 总均值
        chart.fCL = fGrandMean;  // 20260330 ZJH 中心线 = 总均值

        // 20260330 ZJH 计算平均极差 R-bar
        float fRBar = 0.0f;  // 20260330 ZJH 平均极差累加器
        for (float fR : vecRanges) {
            fRBar += fR;  // 20260330 ZJH 累加极差
        }
        fRBar /= static_cast<float>(nNumSubgroups);  // 20260330 ZJH 平均极差

        // 20260330 ZJH A2 常数表（子组大小 2~10 对应的 A2 值）
        // A2 用于从 R-bar 推算 X-bar 控制限: UCL = X̄ + A2*R̄, LCL = X̄ - A2*R̄
        // 来源: ASTM E2587 标准表
        static constexpr float s_arrA2[] = {
            0.0f, 0.0f,   // 20260330 ZJH 占位（n=0,1 不使用）
            1.880f,        // 20260330 ZJH n=2
            1.023f,        // 20260330 ZJH n=3
            0.729f,        // 20260330 ZJH n=4
            0.577f,        // 20260330 ZJH n=5
            0.483f,        // 20260330 ZJH n=6
            0.419f,        // 20260330 ZJH n=7
            0.373f,        // 20260330 ZJH n=8
            0.337f,        // 20260330 ZJH n=9
            0.308f         // 20260330 ZJH n=10
        };

        // 20260330 ZJH 获取对应子组大小的 A2 值
        float fA2 = 0.577f;  // 20260330 ZJH 默认 n=5 的 A2 值
        if (nSubgroupSize >= 2 && nSubgroupSize <= 10) {
            fA2 = s_arrA2[nSubgroupSize];  // 20260330 ZJH 查表
        }

        // 20260330 ZJH 计算 3σ 控制限
        chart.fUCL = fGrandMean + fA2 * fRBar;  // 20260330 ZJH 上控制限 = X̄ + A2*R̄
        chart.fLCL = fGrandMean - fA2 * fRBar;  // 20260330 ZJH 下控制限 = X̄ - A2*R̄

        // 20260330 ZJH 标记超出控制限的失控点
        chart.nOutOfControl = 0;  // 20260330 ZJH 失控点计数器
        for (int i = 0; i < static_cast<int>(chart.vecValues.size()); ++i) {
            if (chart.vecValues[i] > chart.fUCL || chart.vecValues[i] < chart.fLCL) {
                chart.vecViolations.push_back(i);  // 20260330 ZJH 记录失控点索引
                chart.nOutOfControl++;  // 20260330 ZJH 计数+1
            }
        }

        // 20260330 ZJH 设置规则名称
        if (chart.nOutOfControl > 0) {
            chart.strRuleName = "Points beyond 3-sigma control limits";  // 20260330 ZJH 基本控制限规则
        }

        return chart;  // 20260330 ZJH 返回完整控制图
    }

    // 20260330 ZJH WERuleViolation — Western Electric 规则违反记录
    struct WERuleViolation {
        int nRule;              // 20260330 ZJH 规则编号 (1-4)
        int nIndex;             // 20260330 ZJH 违反发生的数据点索引
        std::string strDesc;    // 20260330 ZJH 规则描述
    };

    // 20260330 ZJH checkWesternElectricRules — 检查 Western Electric 规则
    // WE 规则是工业 SPC 的核心判定规则（1956年由 Western Electric 提出）:
    //   规则1: 单点超出 3σ 控制限
    //   规则2: 连续 9 点在中心线同侧
    //   规则3: 连续 6 点递增或递减
    //   规则4: 连续 14 点交替上下
    // chart: 已计算的控制图
    // 返回: 所有违反记录的向量
    std::vector<WERuleViolation> checkWesternElectricRules(const SPCChart& chart) {
        std::vector<WERuleViolation> vecViolations;  // 20260330 ZJH 输出违反记录列表
        const auto& vecVals = chart.vecValues;  // 20260330 ZJH 引用子组均值序列
        int nN = static_cast<int>(vecVals.size());  // 20260330 ZJH 数据点总数

        if (nN == 0) return vecViolations;  // 20260330 ZJH 空数据直接返回

        // ---- 规则1: 单点超出 3σ ----
        // 20260330 ZJH 任何一个点落在 UCL/LCL 之外即违反
        for (int i = 0; i < nN; ++i) {
            if (vecVals[i] > chart.fUCL || vecVals[i] < chart.fLCL) {
                vecViolations.push_back({
                    1, i,
                    "Rule 1: Point " + std::to_string(i) + " beyond 3-sigma limits"
                });  // 20260330 ZJH 记录规则1违反
            }
        }

        // ---- 规则2: 连续 9 点在中心线同侧 ----
        // 20260330 ZJH 连续 9 个点全部在 CL 之上或之下，说明过程偏移
        if (nN >= 9) {
            for (int i = 8; i < nN; ++i) {
                bool bAllAbove = true;   // 20260330 ZJH 假设全部在 CL 之上
                bool bAllBelow = true;   // 20260330 ZJH 假设全部在 CL 之下
                for (int j = i - 8; j <= i; ++j) {
                    if (vecVals[j] <= chart.fCL) bAllAbove = false;  // 20260330 ZJH 有一个不在上方
                    if (vecVals[j] >= chart.fCL) bAllBelow = false;  // 20260330 ZJH 有一个不在下方
                }
                if (bAllAbove || bAllBelow) {
                    vecViolations.push_back({
                        2, i,
                        "Rule 2: 9 consecutive points on same side of CL ending at " + std::to_string(i)
                    });  // 20260330 ZJH 记录规则2违反
                }
            }
        }

        // ---- 规则3: 连续 6 点递增或递减 ----
        // 20260330 ZJH 连续 6 个点单调递增或递减，说明趋势偏移
        if (nN >= 6) {
            for (int i = 5; i < nN; ++i) {
                bool bAllInc = true;  // 20260330 ZJH 假设全部递增
                bool bAllDec = true;  // 20260330 ZJH 假设全部递减
                for (int j = i - 4; j <= i; ++j) {
                    if (vecVals[j] <= vecVals[j - 1]) bAllInc = false;  // 20260330 ZJH 非递增
                    if (vecVals[j] >= vecVals[j - 1]) bAllDec = false;  // 20260330 ZJH 非递减
                }
                if (bAllInc || bAllDec) {
                    vecViolations.push_back({
                        3, i,
                        "Rule 3: 6 consecutive points " + std::string(bAllInc ? "increasing" : "decreasing")
                        + " ending at " + std::to_string(i)
                    });  // 20260330 ZJH 记录规则3违反
                }
            }
        }

        // ---- 规则4: 连续 14 点交替上下 ----
        // 20260330 ZJH 连续 14 个点意味着 13 个相邻差值，每对相邻差值符号交替
        if (nN >= 14) {
            for (int i = 13; i < nN; ++i) {
                bool bAlternating = true;  // 20260330 ZJH 假设全部交替
                // 20260330 ZJH 检查从 (i-13) 到 i 共 14 个点之间的 13 个差值
                // 要求每对相邻差值符号相反
                for (int j = i - 12; j <= i; ++j) {
                    float fPrev = vecVals[j - 1] - vecVals[j - 2];  // 20260330 ZJH 前一步差值
                    float fCurr = vecVals[j] - vecVals[j - 1];      // 20260330 ZJH 当前步差值
                    // 20260330 ZJH 同号（含零）= 非交替，跳出
                    if (fPrev * fCurr >= 0.0f) {
                        bAlternating = false;
                        break;
                    }
                }
                if (bAlternating) {
                    vecViolations.push_back({
                        4, i,
                        "Rule 4: 14 consecutive alternating points ending at " + std::to_string(i)
                    });  // 20260330 ZJH 记录规则4违反
                }
            }
        }

        return vecViolations;  // 20260330 ZJH 返回所有违反记录
    }

    // 20260330 ZJH ParetoItem — 帕累托分析条目
    struct ParetoItem {
        std::string strDefectType;  // 20260330 ZJH 缺陷类型名称
        int nCount;                 // 20260330 ZJH 出现次数
        float fPercent;             // 20260330 ZJH 占比百分比
        float fCumPercent;          // 20260330 ZJH 累积百分比
    };

    // 20260330 ZJH paretoAnalysis — 帕累托分析（良率帕累托图数据）
    // 统计各缺陷类型的出现频次，按频次降序排列并计算累积百分比
    // 80/20 法则: 通常 20% 的缺陷类型导致 80% 的不良
    // vecDefectTypes: 每次不良的缺陷类型标签列表
    // 返回: 按频次降序排列的帕累托分析结果
    std::vector<ParetoItem> paretoAnalysis(const std::vector<std::string>& vecDefectTypes) {
        // 20260330 ZJH 统计各类型计数
        std::map<std::string, int> mapCounts;  // 20260330 ZJH 类型→计数映射
        for (const auto& strType : vecDefectTypes) {
            mapCounts[strType]++;  // 20260330 ZJH 递增计数
        }

        // 20260330 ZJH 转为 ParetoItem 向量
        std::vector<ParetoItem> vecItems;  // 20260330 ZJH 输出向量
        int nTotal = static_cast<int>(vecDefectTypes.size());  // 20260330 ZJH 总不良数
        for (const auto& [strType, nCount] : mapCounts) {
            ParetoItem item;  // 20260330 ZJH 新建条目
            item.strDefectType = strType;  // 20260330 ZJH 类型名称
            item.nCount = nCount;  // 20260330 ZJH 出现次数
            item.fPercent = (nTotal > 0) ? (static_cast<float>(nCount) / static_cast<float>(nTotal) * 100.0f) : 0.0f;  // 20260330 ZJH 占比
            item.fCumPercent = 0.0f;  // 20260330 ZJH 累积百分比稍后计算
            vecItems.push_back(item);  // 20260330 ZJH 加入列表
        }

        // 20260330 ZJH 按频次降序排序
        std::sort(vecItems.begin(), vecItems.end(),
            [](const ParetoItem& a, const ParetoItem& b) {
                return a.nCount > b.nCount;  // 20260330 ZJH 频次高的排前面
            });

        // 20260330 ZJH 计算累积百分比
        float fCum = 0.0f;  // 20260330 ZJH 累积累加器
        for (auto& item : vecItems) {
            fCum += item.fPercent;  // 20260330 ZJH 累加当前占比
            item.fCumPercent = fCum;  // 20260330 ZJH 赋值累积百分比
        }

        return vecItems;  // 20260330 ZJH 返回帕累托分析结果
    }

    // 20260330 ZJH computeCpk — 计算过程能力指数 Cpk
    // Cpk = min((USL - μ), (μ - LSL)) / (3σ)
    // Cpk >= 1.33: 过程能力充足
    // Cpk >= 1.00: 过程能力勉强
    // Cpk <  1.00: 过程能力不足，需改进
    // vecValues: 测量值序列
    // fUSL: 上规格限 (Upper Specification Limit)
    // fLSL: 下规格限 (Lower Specification Limit)
    // 返回: Cpk 值
    float computeCpk(const std::vector<float>& vecValues, float fUSL, float fLSL) {
        if (vecValues.size() < 2) return 0.0f;  // 20260330 ZJH 数据不足

        // 20260330 ZJH 计算均值
        float fSum = 0.0f;  // 20260330 ZJH 累加器
        for (float fVal : vecValues) {
            fSum += fVal;  // 20260330 ZJH 累加
        }
        float fMean = fSum / static_cast<float>(vecValues.size());  // 20260330 ZJH 均值

        // 20260330 ZJH 计算标准差（样本标准差 s）
        float fSumSqDev = 0.0f;  // 20260330 ZJH 偏差平方和
        for (float fVal : vecValues) {
            float fDev = fVal - fMean;  // 20260330 ZJH 偏差
            fSumSqDev += fDev * fDev;  // 20260330 ZJH 累加偏差平方
        }
        float fSigma = std::sqrt(fSumSqDev / static_cast<float>(vecValues.size() - 1));  // 20260330 ZJH 样本标准差

        // 20260330 ZJH σ 为零时返回极大值（过程完全稳定，所有值相同）
        if (fSigma < 1e-12f) return 99.99f;

        // 20260330 ZJH 计算 Cpu 和 Cpl
        float fCpu = (fUSL - fMean) / (3.0f * fSigma);  // 20260330 ZJH 上侧能力指数
        float fCpl = (fMean - fLSL) / (3.0f * fSigma);  // 20260330 ZJH 下侧能力指数

        // 20260330 ZJH Cpk = min(Cpu, Cpl)
        return std::min(fCpu, fCpl);  // 20260330 ZJH 返回较小值即为 Cpk
    }
};

}  // namespace om
