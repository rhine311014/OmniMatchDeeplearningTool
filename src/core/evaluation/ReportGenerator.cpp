// 20260323 ZJH ReportGenerator — 评估报告生成器实现
// 将 EvaluationResult 导出为 HTML 可视化报告或 CSV 数据表

#include "core/evaluation/ReportGenerator.h"

#include <QFile>        // 20260323 ZJH 文件读写
#include <QTextStream>  // 20260323 ZJH 文本流
#include <QDateTime>    // 20260323 ZJH 时间戳
#include <algorithm>    // 20260323 ZJH std::max

// 20260324 ZJH HTML 转义：防止用户输入注入 HTML/JS（CWE-79 XSS 防护）
// 参数: strInput - 可能包含 HTML 特殊字符的原始字符串
// 返回: 经过转义的安全字符串，<>&"' 等字符被替换为 HTML 实体
static QString htmlEscape(const QString& strInput)
{
    return strInput.toHtmlEscaped();  // 20260324 ZJH Qt 内置转义，处理 < > & " 等字符
}

// 20260324 ZJH CSV 注入防护：阻止以公式字符开头的单元格被 Excel 解析为公式（CWE-1236）
// 参数: strField - CSV 字段原始值
// 返回: 经过清理和 RFC 4180 引用的安全 CSV 字段
static QString csvSanitize(const QString& strField)
{
    QString strSafe = strField;  // 20260324 ZJH 复制输入，避免修改原始值

    // 20260324 ZJH 检测危险前缀字符：= + - @ 在 Excel 中会被当作公式执行
    if (!strSafe.isEmpty()) {
        QChar chFirst = strSafe.at(0);  // 20260324 ZJH 取首字符判断是否为公式触发符
        if (chFirst == '=' || chFirst == '+' || chFirst == '-' || chFirst == '@') {
            strSafe.prepend('\'');  // 20260324 ZJH 前置单引号使 Excel 将其视为文本而非公式
        }
    }

    // 20260324 ZJH RFC 4180 合规：包含逗号、双引号或换行的字段需用双引号包裹
    bool bNeedsQuoting = strSafe.contains(',') || strSafe.contains('"') || strSafe.contains('\n');
    if (bNeedsQuoting) {
        strSafe.replace('"', "\"\"");  // 20260324 ZJH 双引号转义：每个 " 变为 ""
        strSafe = '"' + strSafe + '"';  // 20260324 ZJH 整个字段用双引号包裹
    }

    return strSafe;  // 20260324 ZJH 返回安全的 CSV 字段值
}

// 20260323 ZJH 生成 HTML 报告并保存到文件
bool ReportGenerator::generateHtmlReport(
    const EvaluationResult& result,
    const QString& strProjectName,
    const QString& strOutputPath)
{
    // 20260323 ZJH 构建 HTML 内容
    QString strHtml = buildHtmlContent(result, strProjectName);

    // 20260323 ZJH 写入文件
    QFile file(strOutputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;  // 20260323 ZJH 无法打开文件
    }

    QTextStream stream(&file);  // 20260323 ZJH 文本流
    stream.setEncoding(QStringConverter::Utf8);  // 20260323 ZJH UTF-8 编码
    stream << strHtml;
    file.close();

    return true;  // 20260323 ZJH 保存成功
}

// 20260323 ZJH 生成 CSV 数据表并保存到文件
bool ReportGenerator::generateCsvReport(
    const EvaluationResult& result,
    const QString& strOutputPath)
{
    QString strCsv = buildCsvContent(result);  // 20260323 ZJH 构建 CSV 内容

    QFile file(strOutputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    stream << strCsv;
    file.close();

    return true;
}

// 20260323 ZJH 生成 HTML 报告字符串
// 20260324 ZJH 重构：使用 QStringList + join() 替代 repeated QString +=，减少内存重分配
QString ReportGenerator::buildHtmlContent(
    const EvaluationResult& result,
    const QString& strProjectName)
{
    QStringList vecParts;  // 20260324 ZJH 使用列表收集片段，最后 join 一次性拼接

    // 20260323 ZJH 文档头部
    vecParts << QStringLiteral("<!DOCTYPE html>\n<html lang=\"zh-CN\">\n");
    // 20260324 ZJH 对项目名称进行 HTML 转义，防止 XSS 注入攻击
    vecParts << buildHtmlHead(htmlEscape(strProjectName) + " - OmniMatch Evaluation Report");

    vecParts << QStringLiteral("<body>\n");

    // 20260323 ZJH 标题区域
    vecParts << QStringLiteral("<div class=\"header\">\n");
    vecParts << QStringLiteral("  <h1>OmniMatch Evaluation Report</h1>\n");
    // 20260324 ZJH 对项目名称进行 HTML 转义，防止 XSS 注入攻击
    vecParts << ("  <p>Project: " + htmlEscape(strProjectName) + "</p>\n");
    vecParts << ("  <p>Generated: " + QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss") + "</p>\n");
    vecParts << QStringLiteral("</div>\n");

    // 20260323 ZJH 指标概览卡片
    vecParts << QStringLiteral("<h2>Metrics Overview</h2>\n");
    vecParts << buildMetricCards(result);

    // 20260323 ZJH 详细指标表格
    vecParts << QStringLiteral("<h2>Per-Class Metrics</h2>\n");
    vecParts << buildDetailedMetricsTable(result);

    // 20260323 ZJH 混淆矩阵表格
    vecParts << QStringLiteral("<h2>Confusion Matrix</h2>\n");
    vecParts << buildConfusionMatrixTable(result);

    // 20260323 ZJH 性能统计
    vecParts << QStringLiteral("<h2>Performance</h2>\n");
    vecParts << QStringLiteral("<table>\n<tr><th>Metric</th><th>Value</th></tr>\n");
    vecParts << ("<tr><td>Average Latency</td><td>" + QString::number(result.dAvgLatencyMs, 'f', 2) + " ms</td></tr>\n");
    vecParts << ("<tr><td>Throughput</td><td>" + QString::number(result.dThroughputFPS, 'f', 1) + " FPS</td></tr>\n");
    vecParts << QStringLiteral("</table>\n");

    // 20260323 ZJH 统计摘要
    vecParts << QStringLiteral("<h2>Summary</h2>\n");
    vecParts << ("<p>Total: " + QString::number(result.nTotalImages) +
               " | Correct: " + QString::number(result.nCorrect) +
               " | Time: " + QString::number(result.dTotalTimeSeconds, 'f', 2) + "s</p>\n");

    // 20260324 ZJH 使用 CMake 注入的 OM_VERSION 宏替代硬编码版本号
    vecParts << (QStringLiteral("<div class=\"footer\">Generated by OmniMatch v") + QStringLiteral(OM_VERSION) + QStringLiteral("</div>\n"));
    vecParts << QStringLiteral("</body>\n</html>\n");

    return vecParts.join(QString());  // 20260324 ZJH 一次性拼接所有片段，避免多次 QString 重分配
}

// 20260323 ZJH 生成 CSV 报告字符串
// 20260324 ZJH 优先使用 EvaluationResult 中预计算的每类指标，消除重复计算
// 20260324 ZJH 重构：使用 QStringList + join() 替代 repeated QString +=，减少内存重分配
QString ReportGenerator::buildCsvContent(const EvaluationResult& result)
{
    QStringList vecLines;  // 20260324 ZJH 使用列表收集行，最后 join 一次性拼接

    // 20260323 ZJH CSV 表头
    vecLines << QStringLiteral("Class,Precision,Recall,F1,Support");

    int nClasses = result.vecClassNames.size();  // 20260323 ZJH 类别数量

    // 20260324 ZJH 判断是否有预计算的每类指标可用
    bool bHasPerClassMetrics = (result.vecPrecisionPerClass.size() == nClasses &&
                                result.vecRecallPerClass.size() == nClasses &&
                                result.vecF1PerClass.size() == nClasses);

    // 20260323 ZJH 逐类别输出
    for (int i = 0; i < nClasses; ++i) {
        QString strName = (i < result.vecClassNames.size()) ? result.vecClassNames[i] : QString::number(i);

        // 20260323 ZJH 从混淆矩阵计算该类别的支持数（该行总和）
        int nSupport = 0;
        if (i < result.matConfusion.size()) {
            for (int j = 0; j < result.matConfusion[i].size(); ++j) {
                nSupport += result.matConfusion[i][j];
            }
        }

        double dPrecision = 0.0;  // 20260324 ZJH 精确率
        double dRecall = 0.0;     // 20260324 ZJH 召回率
        double dF1 = 0.0;         // 20260324 ZJH F1 分数

        if (bHasPerClassMetrics) {
            // 20260324 ZJH 直接使用预计算的每类指标
            dPrecision = result.vecPrecisionPerClass[i];
            dRecall = result.vecRecallPerClass[i];
            dF1 = result.vecF1PerClass[i];
        } else if (!result.matConfusion.isEmpty()) {
            // 20260324 ZJH 回退：从混淆矩阵重新计算（兼容旧版数据）
            int nTP = result.matConfusion[i][i];  // 20260323 ZJH 真阳性
            int nColSum = 0;
            for (int r = 0; r < nClasses; ++r) nColSum += result.matConfusion[r][i];
            dPrecision = (nColSum > 0) ? static_cast<double>(nTP) / nColSum : 0.0;
            dRecall = (nSupport > 0) ? static_cast<double>(nTP) / nSupport : 0.0;
            double dSum = dPrecision + dRecall;
            dF1 = (dSum > 1e-12) ? (2.0 * dPrecision * dRecall / dSum) : 0.0;
        }

        // 20260324 ZJH 对类别名称进行 CSV 注入防护，阻止公式注入攻击
        vecLines << (csvSanitize(strName) + "," +
                  QString::number(dPrecision, 'f', 4) + "," +
                  QString::number(dRecall, 'f', 4) + "," +
                  QString::number(dF1, 'f', 4) + "," +
                  QString::number(nSupport));
    }

    // 20260323 ZJH 总体指标行
    vecLines << QString();  // 20260324 ZJH 空行分隔
    vecLines << QStringLiteral("Overall Metrics");
    vecLines << ("Accuracy," + QString::number(result.dAccuracy, 'f', 4));
    vecLines << ("Macro Precision," + QString::number(result.dPrecision, 'f', 4));
    vecLines << ("Macro Recall," + QString::number(result.dRecall, 'f', 4));
    vecLines << ("Macro F1," + QString::number(result.dF1Score, 'f', 4));

    return vecLines.join(QStringLiteral("\n")) + QStringLiteral("\n");  // 20260324 ZJH 一次性拼接，末尾加换行
}

// 20260323 ZJH 生成 HTML 头部（含 CSS 样式表）
// 20260324 ZJH 重构：使用 QStringList + join() 替代 repeated QString +=，减少内存重分配
QString ReportGenerator::buildHtmlHead(const QString& strTitle)
{
    QStringList vecParts;  // 20260324 ZJH 使用列表收集片段
    vecParts << QStringLiteral("<head>\n");
    vecParts << QStringLiteral("<meta charset=\"UTF-8\">\n");
    // 20260324 ZJH 对标题进行 HTML 转义，防止 XSS 注入（双重保险）
    vecParts << ("<title>" + htmlEscape(strTitle) + "</title>\n");
    vecParts << QStringLiteral("<style>\n");
    vecParts << QStringLiteral("  body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1d23; color: #e2e8f0; margin: 20px; }\n");
    vecParts << QStringLiteral("  .header { text-align: center; padding: 20px; border-bottom: 2px solid #2563eb; margin-bottom: 30px; }\n");
    vecParts << QStringLiteral("  h1 { color: #ffffff; font-size: 28px; margin: 0; }\n");
    vecParts << QStringLiteral("  h2 { color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }\n");
    vecParts << QStringLiteral("  .cards { display: flex; gap: 16px; margin: 20px 0; }\n");
    vecParts << QStringLiteral("  .card { background: #22262e; border-radius: 8px; padding: 20px; flex: 1; text-align: center; }\n");
    vecParts << QStringLiteral("  .card .value { font-size: 32px; font-weight: bold; }\n");
    vecParts << QStringLiteral("  .card .label { color: #94a3b8; font-size: 14px; margin-top: 4px; }\n");
    vecParts << QStringLiteral("  .card.blue .value { color: #3b82f6; }\n");
    vecParts << QStringLiteral("  .card.orange .value { color: #f59e0b; }\n");
    vecParts << QStringLiteral("  .card.green .value { color: #10b981; }\n");
    vecParts << QStringLiteral("  .card.purple .value { color: #8b5cf6; }\n");
    vecParts << QStringLiteral("  table { border-collapse: collapse; width: 100%; margin: 16px 0; }\n");
    vecParts << QStringLiteral("  th, td { padding: 10px 16px; text-align: left; border-bottom: 1px solid #334155; }\n");
    vecParts << QStringLiteral("  th { background: #22262e; color: #94a3b8; font-weight: 600; }\n");
    vecParts << QStringLiteral("  td { color: #e2e8f0; }\n");
    vecParts << QStringLiteral("  .footer { text-align: center; color: #64748b; margin-top: 40px; padding: 16px; border-top: 1px solid #334155; }\n");
    vecParts << QStringLiteral("  .cm-cell { padding: 6px 10px; text-align: center; min-width: 50px; }\n");
    vecParts << QStringLiteral("</style>\n");
    vecParts << QStringLiteral("</head>\n");
    return vecParts.join(QString());  // 20260324 ZJH 一次性拼接
}

// 20260323 ZJH 生成指标概览卡片
// 20260324 ZJH 重构：使用 QStringList + join() 替代 repeated QString +=，减少内存重分配
QString ReportGenerator::buildMetricCards(const EvaluationResult& result)
{
    QStringList vecParts;  // 20260324 ZJH 使用列表收集片段
    vecParts << QStringLiteral("<div class=\"cards\">\n");
    vecParts << ("  <div class=\"card blue\"><div class=\"value\">" +
                QString::number(result.dAccuracy * 100, 'f', 1) + "%</div><div class=\"label\">Accuracy</div></div>\n");
    vecParts << ("  <div class=\"card orange\"><div class=\"value\">" +
                QString::number(result.dPrecision * 100, 'f', 1) + "%</div><div class=\"label\">Precision</div></div>\n");
    vecParts << ("  <div class=\"card green\"><div class=\"value\">" +
                QString::number(result.dF1Score * 100, 'f', 1) + "%</div><div class=\"label\">F1 Score</div></div>\n");
    vecParts << ("  <div class=\"card purple\"><div class=\"value\">" +
                QString::number(result.dRecall * 100, 'f', 1) + "%</div><div class=\"label\">Recall</div></div>\n");
    vecParts << ("  <div class=\"card blue\"><div class=\"value\">" +
                QString::number(result.dMIoU * 100, 'f', 1) + "%</div><div class=\"label\">mIoU</div></div>\n");
    vecParts << QStringLiteral("</div>\n");
    return vecParts.join(QString());  // 20260324 ZJH 一次性拼接
}

// 20260323 ZJH 生成混淆矩阵 HTML 表格
// 20260324 ZJH 重构：使用 QStringList + join() 替代 repeated QString +=，减少内存重分配
QString ReportGenerator::buildConfusionMatrixTable(const EvaluationResult& result)
{
    if (result.matConfusion.isEmpty()) {
        return "<p>No confusion matrix data.</p>\n";
    }

    int nClasses = result.matConfusion.size();  // 20260323 ZJH 类别数

    QStringList vecParts;  // 20260324 ZJH 使用列表收集片段
    vecParts << QStringLiteral("<table>\n<tr><th></th>");

    // 20260323 ZJH 列标签
    for (int i = 0; i < nClasses; ++i) {
        QString strName = (i < result.vecClassNames.size()) ? result.vecClassNames[i] : QString::number(i);
        // 20260324 ZJH 对类别名称进行 HTML 转义，防止 XSS 注入攻击
        vecParts << ("<th class=\"cm-cell\">" + htmlEscape(strName) + "</th>");
    }
    vecParts << QStringLiteral("</tr>\n");

    // 20260323 ZJH 矩阵行
    for (int i = 0; i < nClasses; ++i) {
        QString strName = (i < result.vecClassNames.size()) ? result.vecClassNames[i] : QString::number(i);
        // 20260324 ZJH 对类别名称进行 HTML 转义，防止 XSS 注入攻击
        vecParts << ("<tr><th>" + htmlEscape(strName) + "</th>");

        // 20260323 ZJH 查找该行最大值用于着色
        int nRowMax = 0;
        for (int j = 0; j < nClasses; ++j) {
            nRowMax = std::max(nRowMax, result.matConfusion[i][j]);
        }

        for (int j = 0; j < nClasses; ++j) {
            int nVal = result.matConfusion[i][j];
            // 20260323 ZJH 对角线元素（正确预测）用绿色背景，其他用红色深浅
            double dIntensity = (nRowMax > 0) ? static_cast<double>(nVal) / nRowMax : 0.0;
            QString strBg;
            if (i == j) {
                // 20260323 ZJH 对角线：绿色系（越大越深）
                strBg = QString("rgba(16, 185, 129, %1)").arg(0.1 + 0.6 * dIntensity, 0, 'f', 2);
            } else {
                // 20260323 ZJH 非对角线：红色系（越大越深）
                strBg = QString("rgba(239, 68, 68, %1)").arg(0.05 + 0.4 * dIntensity, 0, 'f', 2);
            }
            vecParts << ("<td class=\"cm-cell\" style=\"background:" + strBg + "\">" + QString::number(nVal) + "</td>");
        }
        vecParts << QStringLiteral("</tr>\n");
    }

    vecParts << QStringLiteral("</table>\n");
    return vecParts.join(QString());  // 20260324 ZJH 一次性拼接
}

// 20260323 ZJH 生成详细指标表格
// 20260324 ZJH 优先使用 EvaluationResult 中预计算的 vecPrecisionPerClass/vecRecallPerClass/vecF1PerClass
// 仅在预计算向量为空时回退到从混淆矩阵重新计算（兼容旧数据）
// 20260324 ZJH 重构：使用 QStringList + join() 替代 repeated QString +=，减少内存重分配
QString ReportGenerator::buildDetailedMetricsTable(const EvaluationResult& result)
{
    if (result.matConfusion.isEmpty()) {
        return "<p>No metrics data.</p>\n";
    }

    int nClasses = result.matConfusion.size();  // 20260323 ZJH 类别数

    // 20260324 ZJH 判断是否有预计算的每类指标可用
    bool bHasPerClassMetrics = (result.vecPrecisionPerClass.size() == nClasses &&
                                result.vecRecallPerClass.size() == nClasses &&
                                result.vecF1PerClass.size() == nClasses);

    QStringList vecParts;  // 20260324 ZJH 使用列表收集片段
    vecParts << QStringLiteral("<table>\n");
    vecParts << QStringLiteral("<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>\n");

    for (int i = 0; i < nClasses; ++i) {
        QString strName = (i < result.vecClassNames.size()) ? result.vecClassNames[i] : QString::number(i);

        double dP = 0.0;  // 20260324 ZJH 精确率
        double dR = 0.0;  // 20260324 ZJH 召回率
        double dF = 0.0;  // 20260324 ZJH F1 分数

        // 20260324 ZJH 计算该行支持数（该类别真实样本总数）
        int nRowSum = 0;
        for (int j = 0; j < nClasses; ++j) {
            nRowSum += result.matConfusion[i][j];
        }

        if (bHasPerClassMetrics) {
            // 20260324 ZJH 直接使用预计算的每类指标，消除重复计算
            dP = result.vecPrecisionPerClass[i];
            dR = result.vecRecallPerClass[i];
            dF = result.vecF1PerClass[i];
        } else {
            // 20260324 ZJH 回退：从混淆矩阵重新计算（兼容旧版 EvaluationResult）
            int nTP = result.matConfusion[i][i];
            int nColSum = 0;
            for (int j = 0; j < nClasses; ++j) {
                nColSum += result.matConfusion[j][i];
            }
            dP = (nColSum > 0) ? static_cast<double>(nTP) / nColSum : 0.0;
            dR = (nRowSum > 0) ? static_cast<double>(nTP) / nRowSum : 0.0;
            double dSum = dP + dR;
            dF = (dSum > 1e-12) ? (2.0 * dP * dR / dSum) : 0.0;
        }

        // 20260324 ZJH 对类别名称进行 HTML 转义，防止 XSS 注入攻击
        vecParts << ("<tr><td>" + htmlEscape(strName) + "</td>"
                    "<td>" + QString::number(dP, 'f', 4) + "</td>"
                    "<td>" + QString::number(dR, 'f', 4) + "</td>"
                    "<td>" + QString::number(dF, 'f', 4) + "</td>"
                    "<td>" + QString::number(nRowSum) + "</td></tr>\n");
    }

    vecParts << QStringLiteral("</table>\n");
    return vecParts.join(QString());  // 20260324 ZJH 一次性拼接
}
