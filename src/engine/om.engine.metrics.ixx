// 20260320 ZJH 模型评估指标模块 — Phase 5
// 实现分类/检测/分割/异常检测的评估指标
// Accuracy / Precision / Recall / F1 / mIoU / mAP / AUC / ConfusionMatrix
module;

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

export module om.engine.metrics;

export namespace om {

// =========================================================
// 分类指标
// =========================================================

// 20260320 ZJH ConfusionMatrix — 混淆矩阵
// 行 = 真实类别，列 = 预测类别
class ConfusionMatrix {
public:
    // 20260320 ZJH 构造函数
    // nNumClasses: 类别数
    explicit ConfusionMatrix(int nNumClasses)
        : m_nNumClasses(nNumClasses),
          m_vecMatrix(static_cast<size_t>(nNumClasses * nNumClasses), 0)
    {}

    // 20260320 ZJH update — 更新混淆矩阵
    // nTrue: 真实类别
    // nPred: 预测类别
    void update(int nTrue, int nPred) {
        if (nTrue >= 0 && nTrue < m_nNumClasses && nPred >= 0 && nPred < m_nNumClasses) {
            m_vecMatrix[static_cast<size_t>(nTrue * m_nNumClasses + nPred)]++;
        }
    }

    // 20260320 ZJH batchUpdate — 批量更新
    void batchUpdate(const std::vector<int>& vecTrue, const std::vector<int>& vecPred) {
        size_t n = std::min(vecTrue.size(), vecPred.size());
        for (size_t i = 0; i < n; ++i) update(vecTrue[i], vecPred[i]);
    }

    // 20260320 ZJH accuracy — 总体准确率
    float accuracy() const {
        int nCorrect = 0, nTotal = 0;
        for (int i = 0; i < m_nNumClasses; ++i) {
            for (int j = 0; j < m_nNumClasses; ++j) {
                int v = m_vecMatrix[static_cast<size_t>(i * m_nNumClasses + j)];
                nTotal += v;
                if (i == j) nCorrect += v;
            }
        }
        return nTotal > 0 ? static_cast<float>(nCorrect) / static_cast<float>(nTotal) : 0.0f;
    }

    // 20260320 ZJH precision — 每个类别的精确率
    // Precision(c) = TP(c) / (TP(c) + FP(c))
    std::vector<float> precision() const {
        std::vector<float> vecPrec(static_cast<size_t>(m_nNumClasses), 0.0f);
        for (int c = 0; c < m_nNumClasses; ++c) {
            int nTP = m_vecMatrix[static_cast<size_t>(c * m_nNumClasses + c)];  // 对角线
            int nColSum = 0;  // 列和 = TP + FP
            for (int i = 0; i < m_nNumClasses; ++i)
                nColSum += m_vecMatrix[static_cast<size_t>(i * m_nNumClasses + c)];
            vecPrec[static_cast<size_t>(c)] = nColSum > 0 ? static_cast<float>(nTP) / static_cast<float>(nColSum) : 0.0f;
        }
        return vecPrec;
    }

    // 20260320 ZJH recall — 每个类别的召回率
    // Recall(c) = TP(c) / (TP(c) + FN(c))
    std::vector<float> recall() const {
        std::vector<float> vecRecall(static_cast<size_t>(m_nNumClasses), 0.0f);
        for (int c = 0; c < m_nNumClasses; ++c) {
            int nTP = m_vecMatrix[static_cast<size_t>(c * m_nNumClasses + c)];
            int nRowSum = 0;  // 行和 = TP + FN
            for (int j = 0; j < m_nNumClasses; ++j)
                nRowSum += m_vecMatrix[static_cast<size_t>(c * m_nNumClasses + j)];
            vecRecall[static_cast<size_t>(c)] = nRowSum > 0 ? static_cast<float>(nTP) / static_cast<float>(nRowSum) : 0.0f;
        }
        return vecRecall;
    }

    // 20260320 ZJH f1Score — 每个类别的 F1 分数
    // F1(c) = 2 * Precision(c) * Recall(c) / (Precision(c) + Recall(c))
    std::vector<float> f1Score() const {
        auto vecPrec = precision();
        auto vecRec = recall();
        std::vector<float> vecF1(static_cast<size_t>(m_nNumClasses), 0.0f);
        for (int c = 0; c < m_nNumClasses; ++c) {
            float fSum = vecPrec[static_cast<size_t>(c)] + vecRec[static_cast<size_t>(c)];
            vecF1[static_cast<size_t>(c)] = fSum > 0
                ? 2.0f * vecPrec[static_cast<size_t>(c)] * vecRec[static_cast<size_t>(c)] / fSum
                : 0.0f;
        }
        return vecF1;
    }

    // 20260320 ZJH macroF1 — 宏平均 F1
    float macroF1() const {
        auto vecF1 = f1Score();
        float fSum = 0.0f;
        for (auto f : vecF1) fSum += f;
        return m_nNumClasses > 0 ? fSum / static_cast<float>(m_nNumClasses) : 0.0f;
    }

    // 20260320 ZJH weightedF1 — 加权平均 F1（按各类样本数加权）
    float weightedF1() const {
        auto vecF1 = f1Score();
        float fSum = 0.0f, fTotalWeight = 0.0f;
        for (int c = 0; c < m_nNumClasses; ++c) {
            int nRowSum = 0;
            for (int j = 0; j < m_nNumClasses; ++j)
                nRowSum += m_vecMatrix[static_cast<size_t>(c * m_nNumClasses + j)];
            fSum += vecF1[static_cast<size_t>(c)] * static_cast<float>(nRowSum);
            fTotalWeight += static_cast<float>(nRowSum);
        }
        return fTotalWeight > 0 ? fSum / fTotalWeight : 0.0f;
    }

    // 20260320 ZJH 获取矩阵值
    int get(int nRow, int nCol) const {
        return m_vecMatrix[static_cast<size_t>(nRow * m_nNumClasses + nCol)];
    }

    // 20260320 ZJH 重置矩阵
    void reset() { std::fill(m_vecMatrix.begin(), m_vecMatrix.end(), 0); }

    int numClasses() const { return m_nNumClasses; }

private:
    int m_nNumClasses;
    std::vector<int> m_vecMatrix;
};

// =========================================================
// 早停机制
// =========================================================

// 20260320 ZJH EarlyStopping — 早停策略
// 当验证损失连续 patience 个 epoch 不改善时触发早停
class EarlyStopping {
public:
    // 20260320 ZJH 构造函数
    // nPatience: 耐心值（最大允许不改善的 epoch 数）
    // fMinDelta: 最小改善量（损失降低小于此值不算改善）
    EarlyStopping(int nPatience = 10, float fMinDelta = 1e-4f)
        : m_nPatience(nPatience), m_fMinDelta(fMinDelta) {}

    // 20260320 ZJH step — 每个 epoch 结束时调用
    // fValLoss: 当前验证损失
    // 返回: true 表示应该停止训练
    bool step(float fValLoss) {
        if (fValLoss < m_fBestLoss - m_fMinDelta) {
            // 20260320 ZJH 损失改善
            m_fBestLoss = fValLoss;
            m_nCounter = 0;
            m_bImproved = true;
            return false;
        }
        // 20260320 ZJH 损失未改善
        m_nCounter++;
        m_bImproved = false;
        return m_nCounter >= m_nPatience;
    }

    // 20260320 ZJH 是否刚刚改善
    bool improved() const { return m_bImproved; }

    // 20260320 ZJH 获取最佳损失
    float bestLoss() const { return m_fBestLoss; }

    // 20260320 ZJH 获取当前计数器
    int counter() const { return m_nCounter; }

    // 20260320 ZJH 重置
    void reset() { m_fBestLoss = 1e30f; m_nCounter = 0; m_bImproved = false; }

private:
    int m_nPatience;
    float m_fMinDelta;
    float m_fBestLoss = 1e30f;
    int m_nCounter = 0;
    bool m_bImproved = false;
};

// =========================================================
// 检测指标
// =========================================================

// 20260320 ZJH DetectionMetrics — 目标检测评估
struct DetectionBox {
    float fX1, fY1, fX2, fY2;  // 20260320 ZJH 边界框坐标
    int nClassId;                // 20260320 ZJH 类别 ID
    float fConfidence;           // 20260320 ZJH 置信度
};

// 20260320 ZJH computeIoU — 计算两个边界框的交并比
float computeIoU(const DetectionBox& a, const DetectionBox& b) {
    float fX1 = std::max(a.fX1, b.fX1);
    float fY1 = std::max(a.fY1, b.fY1);
    float fX2 = std::min(a.fX2, b.fX2);
    float fY2 = std::min(a.fY2, b.fY2);
    float fIntersection = std::max(0.0f, fX2 - fX1) * std::max(0.0f, fY2 - fY1);
    float fAreaA = (a.fX2 - a.fX1) * (a.fY2 - a.fY1);
    float fAreaB = (b.fX2 - b.fX1) * (b.fY2 - b.fY1);
    float fUnion = fAreaA + fAreaB - fIntersection;
    return fUnion > 0 ? fIntersection / fUnion : 0.0f;
}

// 20260320 ZJH computeAP — 计算单个类别的 Average Precision (AP)
// 使用 11 点插值法
float computeAP(const std::vector<float>& vecPrecision,
                const std::vector<float>& vecRecall) {
    if (vecPrecision.empty()) return 0.0f;
    float fAP = 0.0f;
    for (float fT = 0.0f; fT <= 1.0f; fT += 0.1f) {
        float fMaxPrec = 0.0f;
        for (size_t i = 0; i < vecRecall.size(); ++i) {
            if (vecRecall[i] >= fT && vecPrecision[i] > fMaxPrec) {
                fMaxPrec = vecPrecision[i];
            }
        }
        fAP += fMaxPrec;
    }
    return fAP / 11.0f;
}

// =========================================================
// 分割指标
// =========================================================

// 20260320 ZJH computeMeanIoU — 计算语义分割的 mIoU
// vecPredMask/vecTrueMask: 平铺的像素级类别 ID，长度 = H*W
float computeMeanIoU(const std::vector<int>& vecPredMask,
                      const std::vector<int>& vecTrueMask,
                      int nNumClasses) {
    std::vector<int> vecIntersection(static_cast<size_t>(nNumClasses), 0);
    std::vector<int> vecUnion(static_cast<size_t>(nNumClasses), 0);

    for (size_t i = 0; i < vecPredMask.size(); ++i) {
        int nPred = vecPredMask[i];
        int nTrue = vecTrueMask[i];
        if (nPred == nTrue) vecIntersection[static_cast<size_t>(nPred)]++;
        if (nPred >= 0 && nPred < nNumClasses) vecUnion[static_cast<size_t>(nPred)]++;
        if (nTrue >= 0 && nTrue < nNumClasses) vecUnion[static_cast<size_t>(nTrue)]++;
        if (nPred == nTrue && nPred >= 0) vecUnion[static_cast<size_t>(nPred)]--;  // 20260320 ZJH 交集被加了两次
    }

    float fMeanIoU = 0.0f;
    int nValidClasses = 0;
    for (int c = 0; c < nNumClasses; ++c) {
        if (vecUnion[static_cast<size_t>(c)] > 0) {
            fMeanIoU += static_cast<float>(vecIntersection[static_cast<size_t>(c)])
                        / static_cast<float>(vecUnion[static_cast<size_t>(c)]);
            nValidClasses++;
        }
    }
    return nValidClasses > 0 ? fMeanIoU / static_cast<float>(nValidClasses) : 0.0f;
}

// =========================================================
// 异常检测指标
// =========================================================

// 20260320 ZJH computeROCAUC — 计算 ROC AUC
// vecScores: 异常分数（越高越可能异常）
// vecLabels: 真实标签（0=正常, 1=异常）
float computeROCAUC(const std::vector<float>& vecScores,
                     const std::vector<int>& vecLabels) {
    if (vecScores.empty()) return 0.5f;

    // 20260320 ZJH 按分数降序排序
    std::vector<size_t> vecIdx(vecScores.size());
    std::iota(vecIdx.begin(), vecIdx.end(), 0);
    std::sort(vecIdx.begin(), vecIdx.end(), [&](size_t a, size_t b) {
        return vecScores[a] > vecScores[b];
    });

    int nPositive = 0, nNegative = 0;
    for (auto lbl : vecLabels) {
        if (lbl == 1) nPositive++;
        else nNegative++;
    }
    if (nPositive == 0 || nNegative == 0) return 0.5f;

    // 20260320 ZJH 遍历排序后的样本计算 AUC（梯形法则）
    float fAUC = 0.0f;
    int nTP = 0, nFP = 0;
    float fPrevTPR = 0.0f, fPrevFPR = 0.0f;

    for (size_t i = 0; i < vecIdx.size(); ++i) {
        if (vecLabels[vecIdx[i]] == 1) nTP++;
        else nFP++;
        float fTPR = static_cast<float>(nTP) / static_cast<float>(nPositive);
        float fFPR = static_cast<float>(nFP) / static_cast<float>(nNegative);
        fAUC += 0.5f * (fTPR + fPrevTPR) * (fFPR - fPrevFPR);  // 20260320 ZJH 梯形面积
        fPrevTPR = fTPR;
        fPrevFPR = fFPR;
    }

    return fAUC;
}

// 20260320 ZJH findOptimalThreshold — 寻找最优异常阈值（Youden's J）
// 返回使 (TPR - FPR) 最大化的阈值
float findOptimalThreshold(const std::vector<float>& vecScores,
                            const std::vector<int>& vecLabels) {
    if (vecScores.empty()) return 0.5f;

    std::vector<size_t> vecIdx(vecScores.size());
    std::iota(vecIdx.begin(), vecIdx.end(), 0);
    std::sort(vecIdx.begin(), vecIdx.end(), [&](size_t a, size_t b) {
        return vecScores[a] > vecScores[b];
    });

    int nPositive = 0, nNegative = 0;
    for (auto lbl : vecLabels) { if (lbl == 1) nPositive++; else nNegative++; }
    if (nPositive == 0 || nNegative == 0) return 0.5f;

    float fBestJ = -1.0f;
    float fBestThresh = 0.5f;
    int nTP = 0, nFP = 0;

    for (size_t i = 0; i < vecIdx.size(); ++i) {
        if (vecLabels[vecIdx[i]] == 1) nTP++;
        else nFP++;
        float fTPR = static_cast<float>(nTP) / static_cast<float>(nPositive);
        float fFPR = static_cast<float>(nFP) / static_cast<float>(nNegative);
        float fJ = fTPR - fFPR;  // 20260320 ZJH Youden's J 统计量
        if (fJ > fBestJ) {
            fBestJ = fJ;
            fBestThresh = vecScores[vecIdx[i]];
        }
    }
    return fBestThresh;
}

}  // namespace om
