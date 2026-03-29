// 20260323 ZJH ReportGenerator — 评估报告生成器
// 将 EvaluationResult 导出为 HTML 可视化报告或 CSV 数据表
#pragma once

#include "core/evaluation/EvaluationResult.h"  // 20260323 ZJH 评估结果数据结构

#include <QString>  // 20260323 ZJH 字符串

// 20260323 ZJH 评估报告生成器
// 支持 HTML（含内嵌 CSS 样式的独立网页）和 CSV 两种格式
class ReportGenerator
{
public:
    // 20260323 ZJH 生成 HTML 报告并保存到文件
    // 参数: result - 评估结果
    //       strProjectName - 项目名称
    //       strOutputPath - 输出文件路径 (.html)
    // 返回: true 表示保存成功
    static bool generateHtmlReport(
        const EvaluationResult& result,
        const QString& strProjectName,
        const QString& strOutputPath);

    // 20260323 ZJH 生成 CSV 数据表并保存到文件
    // 参数: result - 评估结果
    //       strOutputPath - 输出文件路径 (.csv)
    // 返回: true 表示保存成功
    static bool generateCsvReport(
        const EvaluationResult& result,
        const QString& strOutputPath);

    // 20260323 ZJH 生成 HTML 报告的字符串内容（不保存文件）
    static QString buildHtmlContent(
        const EvaluationResult& result,
        const QString& strProjectName);

    // 20260323 ZJH 生成 CSV 报告的字符串内容（不保存文件）
    static QString buildCsvContent(
        const EvaluationResult& result);

private:
    // 20260323 ZJH 生成 HTML 报告头部（含 CSS 样式）
    static QString buildHtmlHead(const QString& strTitle);

    // 20260323 ZJH 生成指标概览卡片 HTML
    static QString buildMetricCards(const EvaluationResult& result);

    // 20260323 ZJH 生成混淆矩阵 HTML 表格
    static QString buildConfusionMatrixTable(const EvaluationResult& result);

    // 20260323 ZJH 生成详细指标表格
    static QString buildDetailedMetricsTable(const EvaluationResult& result);
};
