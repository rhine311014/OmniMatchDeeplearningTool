// 20260322 ZJH EvaluationResult — 模型评估结果数据结构
// 持有一次评估的全部指标：准确率/精确率/召回率/F1/mAP/mIoU/AUC
// 以及混淆矩阵、性能统计（延迟/吞吐量）和统计摘要

#pragma once

#include <QVector>       // 20260322 ZJH 混淆矩阵行列存储
#include <QStringList>   // 20260322 ZJH 类别名称列表

// 20260322 ZJH 模型评估结果结构体
// 包含分类/检测/分割任务通用的评估指标
struct EvaluationResult
{
    // ===== 核心指标 =====

    double dAccuracy   = 0;  // 20260322 ZJH 准确率（分类任务核心指标，正确数 / 总数）
    double dPrecision  = 0;  // 20260322 ZJH 精确率（TP / (TP + FP)，宏平均）
    double dRecall     = 0;  // 20260322 ZJH 召回率（TP / (TP + FN)，宏平均）
    double dF1Score    = 0;  // 20260322 ZJH F1 分数（精确率与召回率的调和平均）
    double dMeanAP     = 0;  // 20260322 ZJH mAP（检测任务核心指标，平均精度均值）
    double dMIoU       = 0;  // 20260322 ZJH mIoU（分割任务核心指标，平均交并比）
    double dAUC        = 0;  // 20260322 ZJH AUC（ROC 曲线下面积，二分类评价指标）

    // ===== 性能统计 =====

    double dAvgLatencyMs  = 0;  // 20260322 ZJH 平均推理延迟（毫秒/张）
    double dThroughputFPS = 0;  // 20260322 ZJH 推理吞吐量（帧/秒）

    // ===== 每类指标 =====

    // 20260324 ZJH 每类精确率/召回率/F1 向量（与 vecClassNames 索引对齐）
    // 由 MetricsCalculator::computeClassificationMetrics 计算后存储，消除 ReportGenerator 重复计算
    QVector<double> vecPrecisionPerClass;  // 20260324 ZJH 各类别精确率（TP / (TP + FP)）
    QVector<double> vecRecallPerClass;     // 20260324 ZJH 各类别召回率（TP / (TP + FN)）
    QVector<double> vecF1PerClass;         // 20260324 ZJH 各类别 F1 分数

    // ===== 混淆矩阵 =====

    QVector<QVector<int>> matConfusion;  // 20260322 ZJH 混淆矩阵（行=真实类别，列=预测类别）
    QStringList vecClassNames;           // 20260322 ZJH 类别名称列表（与矩阵行列对应）

    // ===== 统计摘要 =====

    int    nTotalImages      = 0;  // 20260322 ZJH 评估的图像总数
    int    nCorrect          = 0;  // 20260322 ZJH 正确预测数量
    double dTotalTimeSeconds = 0;  // 20260322 ZJH 评估总耗时（秒）
};
