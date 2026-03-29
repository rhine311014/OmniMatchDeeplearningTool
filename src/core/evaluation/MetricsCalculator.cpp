// 20260323 ZJH MetricsCalculator — 评估指标计算器实现
// 从预测结果和标注真值计算精确率/召回率/F1/准确率/混淆矩阵/ROC/PR 曲线

#include "core/evaluation/MetricsCalculator.h"

#include <algorithm>  // 20260323 ZJH std::sort / std::max
#include <numeric>    // 20260323 ZJH std::accumulate
#include <cmath>      // 20260323 ZJH std::abs

// 20260323 ZJH 从分类预测结果计算完整的评估指标
EvaluationResult MetricsCalculator::computeClassificationMetrics(
    const QVector<PredictionEntry>& vecPredictions,
    const QStringList& classNames,
    int nNumClasses)
{
    EvaluationResult result;  // 20260323 ZJH 初始化返回结果

    // 20260323 ZJH 保存类别名称
    result.vecClassNames = classNames;

    // 20260323 ZJH 计算混淆矩阵
    result.matConfusion = computeConfusionMatrix(vecPredictions, nNumClasses);

    // 20260323 ZJH 从混淆矩阵计算每类精确率/召回率/F1
    QVector<double> vecPrecision = computePrecisionPerClass(result.matConfusion);
    QVector<double> vecRecall = computeRecallPerClass(result.matConfusion);
    QVector<double> vecF1 = computeF1PerClass(vecPrecision, vecRecall);

    // 20260324 ZJH 将每类指标存入结果结构体，供 ReportGenerator 直接使用
    // 消除 ReportGenerator 中从混淆矩阵重复计算每类指标的冗余逻辑
    result.vecPrecisionPerClass = vecPrecision;
    result.vecRecallPerClass = vecRecall;
    result.vecF1PerClass = vecF1;

    // 20260323 ZJH 准确率
    result.dAccuracy = computeAccuracy(result.matConfusion);

    // 20260323 ZJH 宏平均指标
    result.dPrecision = macroAverage(vecPrecision);
    result.dRecall = macroAverage(vecRecall);
    result.dF1Score = macroAverage(vecF1);

    // 20260323 ZJH 统计摘要（从混淆矩阵对角线求正确数，避免冗余遍历）
    result.nTotalImages = vecPredictions.size();
    int nCorrect = 0;
    for (int i = 0; i < nNumClasses; ++i) {
        nCorrect += result.matConfusion[i][i];  // 20260323 ZJH 对角线之和 = 正确预测总数
    }
    result.nCorrect = nCorrect;

    return result;  // 20260323 ZJH 返回完整评估结果
}

// 20260323 ZJH 计算混淆矩阵
QVector<QVector<int>> MetricsCalculator::computeConfusionMatrix(
    const QVector<PredictionEntry>& vecPredictions,
    int nNumClasses)
{
    // 20260323 ZJH 初始化 nNumClasses x nNumClasses 零矩阵
    QVector<QVector<int>> matrix(nNumClasses, QVector<int>(nNumClasses, 0));

    // 20260323 ZJH 遍历所有预测，填充矩阵: matrix[真实][预测]++
    for (const auto& pred : vecPredictions) {
        int nTrue = pred.nTrueLabel;    // 20260323 ZJH 真实标签
        int nPred = pred.nPredictedLabel;  // 20260323 ZJH 预测标签
        // 20260323 ZJH 边界检查防止越界
        if (nTrue >= 0 && nTrue < nNumClasses &&
            nPred >= 0 && nPred < nNumClasses) {
            matrix[nTrue][nPred]++;
        }
    }

    return matrix;  // 20260323 ZJH 返回混淆矩阵
}

// 20260323 ZJH 从混淆矩阵计算每个类别的精确率
// 精确率 = TP / (TP + FP) = matrix[i][i] / sum(matrix[*][i])
QVector<double> MetricsCalculator::computePrecisionPerClass(
    const QVector<QVector<int>>& confMatrix)
{
    int nClasses = confMatrix.size();  // 20260323 ZJH 类别总数
    QVector<double> vecPrecision(nClasses, 0.0);

    for (int nCol = 0; nCol < nClasses; ++nCol) {
        int nTP = confMatrix[nCol][nCol];  // 20260323 ZJH 真阳性 = 对角线元素
        int nColSum = 0;  // 20260323 ZJH 该列总和 (TP + FP)
        for (int nRow = 0; nRow < nClasses; ++nRow) {
            nColSum += confMatrix[nRow][nCol];
        }
        // 20260323 ZJH 列总和为 0 时精确率定义为 0
        vecPrecision[nCol] = (nColSum > 0) ? static_cast<double>(nTP) / nColSum : 0.0;
    }

    return vecPrecision;  // 20260323 ZJH 返回各类别精确率
}

// 20260323 ZJH 从混淆矩阵计算每个类别的召回率
// 召回率 = TP / (TP + FN) = matrix[i][i] / sum(matrix[i][*])
QVector<double> MetricsCalculator::computeRecallPerClass(
    const QVector<QVector<int>>& confMatrix)
{
    int nClasses = confMatrix.size();  // 20260323 ZJH 类别总数
    QVector<double> vecRecall(nClasses, 0.0);

    for (int nRow = 0; nRow < nClasses; ++nRow) {
        int nTP = confMatrix[nRow][nRow];  // 20260323 ZJH 真阳性
        int nRowSum = 0;  // 20260323 ZJH 该行总和 (TP + FN)
        for (int nCol = 0; nCol < nClasses; ++nCol) {
            nRowSum += confMatrix[nRow][nCol];
        }
        // 20260323 ZJH 行总和为 0 时召回率定义为 0
        vecRecall[nRow] = (nRowSum > 0) ? static_cast<double>(nTP) / nRowSum : 0.0;
    }

    return vecRecall;  // 20260323 ZJH 返回各类别召回率
}

// 20260323 ZJH 从精确率和召回率计算 F1 分数
// F1 = 2 * P * R / (P + R)
QVector<double> MetricsCalculator::computeF1PerClass(
    const QVector<double>& vecPrecision,
    const QVector<double>& vecRecall)
{
    int nClasses = vecPrecision.size();  // 20260323 ZJH 类别数
    QVector<double> vecF1(nClasses, 0.0);

    for (int i = 0; i < nClasses; ++i) {
        double dP = vecPrecision[i];  // 20260323 ZJH 精确率
        double dR = vecRecall[i];     // 20260323 ZJH 召回率
        double dSum = dP + dR;        // 20260323 ZJH 分母
        // 20260323 ZJH 分母为 0 时 F1 定义为 0
        vecF1[i] = (dSum > 1e-12) ? (2.0 * dP * dR / dSum) : 0.0;
    }

    return vecF1;  // 20260323 ZJH 返回各类别 F1 分数
}

// 20260323 ZJH 计算整体准确率 = 对角线元素之和 / 矩阵所有元素之和
double MetricsCalculator::computeAccuracy(
    const QVector<QVector<int>>& confMatrix)
{
    int nClasses = confMatrix.size();  // 20260323 ZJH 类别数
    int nCorrect = 0;  // 20260323 ZJH 正确预测总数（对角线之和）
    int nTotal = 0;    // 20260323 ZJH 总样本数（所有元素之和）

    for (int i = 0; i < nClasses; ++i) {
        for (int j = 0; j < nClasses; ++j) {
            nTotal += confMatrix[i][j];
            if (i == j) {
                nCorrect += confMatrix[i][j];  // 20260323 ZJH 对角线累加
            }
        }
    }

    // 20260323 ZJH 总数为 0 时返回 0
    return (nTotal > 0) ? static_cast<double>(nCorrect) / nTotal : 0.0;
}

// 20260323 ZJH 计算宏平均值（所有值的算术平均）
double MetricsCalculator::macroAverage(const QVector<double>& vecValues)
{
    if (vecValues.isEmpty()) return 0.0;  // 20260323 ZJH 空向量返回 0

    double dSum = 0.0;  // 20260323 ZJH 求和
    for (double dVal : vecValues) {
        dSum += dVal;
    }
    return dSum / vecValues.size();  // 20260323 ZJH 算术平均
}

// 20260323 ZJH 计算 ROC 曲线数据点
// 将预测按置信度降序排列，逐步计算 TPR 和 FPR
QVector<QPair<double, double>> MetricsCalculator::computeROCCurve(
    const QVector<PredictionEntry>& vecPredictions,
    int nPositiveClass)
{
    // 20260323 ZJH 复制并按置信度降序排序
    QVector<PredictionEntry> vecSorted = vecPredictions;
    std::sort(vecSorted.begin(), vecSorted.end(),
              [](const PredictionEntry& a, const PredictionEntry& b) {
                  return a.dConfidence > b.dConfidence;  // 20260323 ZJH 降序排列
              });

    // 20260323 ZJH 统计正样本和负样本总数
    int nPositive = 0;  // 20260323 ZJH 真实正样本总数
    int nNegative = 0;  // 20260323 ZJH 真实负样本总数
    for (const auto& pred : vecSorted) {
        if (pred.nTrueLabel == nPositiveClass) {
            ++nPositive;
        } else {
            ++nNegative;
        }
    }

    QVector<QPair<double, double>> vecCurve;  // 20260323 ZJH ROC 曲线点列表
    vecCurve.append({0.0, 0.0});  // 20260323 ZJH 起始点 (0, 0)

    // 20260323 ZJH 防止除零
    if (nPositive == 0 || nNegative == 0) {
        vecCurve.append({1.0, 1.0});
        return vecCurve;
    }

    int nTP = 0;  // 20260323 ZJH 累计真阳性
    int nFP = 0;  // 20260323 ZJH 累计假阳性

    // 20260323 ZJH 逐个样本遍历，计算每个阈值下的 TPR 和 FPR
    for (const auto& pred : vecSorted) {
        if (pred.nTrueLabel == nPositiveClass) {
            ++nTP;  // 20260323 ZJH 真阳性 +1
        } else {
            ++nFP;  // 20260323 ZJH 假阳性 +1
        }
        double dTPR = static_cast<double>(nTP) / nPositive;  // 20260323 ZJH 真阳性率
        double dFPR = static_cast<double>(nFP) / nNegative;  // 20260323 ZJH 假阳性率
        vecCurve.append({dFPR, dTPR});
    }

    return vecCurve;  // 20260323 ZJH 返回 ROC 曲线点列表
}

// 20260323 ZJH 计算 PR 曲线数据点
QVector<QPair<double, double>> MetricsCalculator::computePRCurve(
    const QVector<PredictionEntry>& vecPredictions,
    int nPositiveClass)
{
    // 20260323 ZJH 复制并按置信度降序排序
    QVector<PredictionEntry> vecSorted = vecPredictions;
    std::sort(vecSorted.begin(), vecSorted.end(),
              [](const PredictionEntry& a, const PredictionEntry& b) {
                  return a.dConfidence > b.dConfidence;
              });

    // 20260323 ZJH 统计正样本总数
    int nPositive = 0;
    for (const auto& pred : vecSorted) {
        if (pred.nTrueLabel == nPositiveClass) {
            ++nPositive;
        }
    }

    QVector<QPair<double, double>> vecCurve;  // 20260323 ZJH PR 曲线点列表
    vecCurve.append({0.0, 1.0});  // 20260323 ZJH 起始点 (Recall=0, Precision=1)

    if (nPositive == 0) {
        vecCurve.append({1.0, 0.0});
        return vecCurve;
    }

    int nTP = 0;  // 20260323 ZJH 累计真阳性
    int nFP = 0;  // 20260323 ZJH 累计假阳性

    for (const auto& pred : vecSorted) {
        if (pred.nTrueLabel == nPositiveClass) {
            ++nTP;
        } else {
            ++nFP;
        }
        double dPrecision = static_cast<double>(nTP) / (nTP + nFP);  // 20260323 ZJH 精确率
        double dRecall = static_cast<double>(nTP) / nPositive;        // 20260323 ZJH 召回率
        vecCurve.append({dRecall, dPrecision});
    }

    return vecCurve;  // 20260323 ZJH 返回 PR 曲线点列表
}

// 20260323 ZJH 计算曲线下面积（AUC）—— 梯形法则积分
double MetricsCalculator::computeAUC(
    const QVector<QPair<double, double>>& vecCurve)
{
    if (vecCurve.size() < 2) return 0.0;  // 20260323 ZJH 点数不足无法积分

    double dArea = 0.0;  // 20260323 ZJH 累计面积

    // 20260323 ZJH 梯形积分: 相邻两点之间的梯形面积累加
    for (int i = 1; i < vecCurve.size(); ++i) {
        double dDx = vecCurve[i].first - vecCurve[i - 1].first;    // 20260323 ZJH X 轴差
        double dAvgY = (vecCurve[i].second + vecCurve[i - 1].second) / 2.0;  // 20260323 ZJH Y 轴平均高度
        dArea += dDx * dAvgY;  // 20260323 ZJH 累加梯形面积
    }

    return std::abs(dArea);  // 20260323 ZJH 取绝对值防止方向问题
}
