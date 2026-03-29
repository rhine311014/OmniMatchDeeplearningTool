// 20260323 ZJH MetricsCalculator — 评估指标计算器
// 从预测结果和标注真值计算各项评估指标
// 支持分类指标(精确率/召回率/F1/准确率)和检测指标(mAP/IoU)
#pragma once

#include "core/evaluation/EvaluationResult.h"  // 20260323 ZJH 评估结果数据结构

#include <QVector>      // 20260323 ZJH 动态数组
#include <QString>      // 20260323 ZJH 字符串
#include <QStringList>  // 20260323 ZJH 字符串列表

// 20260323 ZJH 单张图像的预测结果
struct PredictionEntry
{
    int nTrueLabel = -1;        // 20260323 ZJH 真实标签 ID
    int nPredictedLabel = -1;   // 20260323 ZJH 预测标签 ID
    double dConfidence = 0.0;   // 20260323 ZJH 预测置信度 [0, 1]
};

// 20260323 ZJH 检测框预测结果
struct DetectionPrediction
{
    int nTrueLabel = -1;        // 20260323 ZJH 真实标签 ID
    int nPredictedLabel = -1;   // 20260323 ZJH 预测标签 ID
    double dConfidence = 0.0;   // 20260323 ZJH 预测置信度
    double dIoU = 0.0;          // 20260323 ZJH 与真实框的 IoU
};

// 20260323 ZJH 评估指标计算器
// 提供从预测结果到 EvaluationResult 的完整计算流程
class MetricsCalculator
{
public:
    // 20260323 ZJH 从分类预测结果计算完整的评估指标
    // 参数: vecPredictions - 全部图像的预测结果列表
    //       classNames - 类别名称列表
    //       nNumClasses - 类别总数
    // 返回: 包含所有指标的 EvaluationResult
    static EvaluationResult computeClassificationMetrics(
        const QVector<PredictionEntry>& vecPredictions,
        const QStringList& classNames,
        int nNumClasses);

    // 20260323 ZJH 计算混淆矩阵
    // 参数: vecPredictions - 预测结果列表
    //       nNumClasses - 类别总数
    // 返回: nNumClasses x nNumClasses 混淆矩阵，行=真实，列=预测
    static QVector<QVector<int>> computeConfusionMatrix(
        const QVector<PredictionEntry>& vecPredictions,
        int nNumClasses);

    // 20260323 ZJH 从混淆矩阵计算每个类别的精确率
    // 参数: confMatrix - 混淆矩阵
    // 返回: 各类别精确率列表
    static QVector<double> computePrecisionPerClass(
        const QVector<QVector<int>>& confMatrix);

    // 20260323 ZJH 从混淆矩阵计算每个类别的召回率
    static QVector<double> computeRecallPerClass(
        const QVector<QVector<int>>& confMatrix);

    // 20260323 ZJH 从精确率和召回率计算 F1 分数
    static QVector<double> computeF1PerClass(
        const QVector<double>& vecPrecision,
        const QVector<double>& vecRecall);

    // 20260323 ZJH 计算整体准确率
    static double computeAccuracy(
        const QVector<QVector<int>>& confMatrix);

    // 20260323 ZJH 计算宏平均指标
    static double macroAverage(const QVector<double>& vecValues);

    // 20260323 ZJH 计算 ROC 曲线数据点 (FPR, TPR) 用于绘图
    // 参数: vecPredictions - 预测结果
    //       nPositiveClass - 正类标签 ID
    // 返回: 点对列表 [(fpr, tpr), ...]，从 (0,0) 到 (1,1)
    static QVector<QPair<double, double>> computeROCCurve(
        const QVector<PredictionEntry>& vecPredictions,
        int nPositiveClass);

    // 20260323 ZJH 计算 PR 曲线数据点 (Recall, Precision) 用于绘图
    static QVector<QPair<double, double>> computePRCurve(
        const QVector<PredictionEntry>& vecPredictions,
        int nPositiveClass);

    // 20260323 ZJH 计算 AUC（梯形法则积分）
    static double computeAUC(
        const QVector<QPair<double, double>>& vecCurve);
};
