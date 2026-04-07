// 20260330 ZJH 评估报告导出器 — 对标 MVTec report_templates (HTML/CSV/JSON 三格式)
// 纯 C++ 实现，无 Qt 依赖，支持自包含 HTML（内联 CSS + SVG 图表 + 混淆矩阵热力图）
// OmniMatch 品牌暗色主题: 主色 #1a1a2e, 强调色 #00bcd4
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace om {

// =========================================================
// ReportData — 报告数据结构
// =========================================================

// 20260330 ZJH ReportData — 聚合训练 + 评估 + 性能三方面数据
// 用于生成完整的评估报告，对标 MVTec Deep Learning Tool 的导出报告
struct ReportData {
    // ------ 训练信息 ------
    std::string strModelName;     // 20260330 ZJH 模型名称（如 "ResNet18_AnomalyDet"）
    std::string strTaskType;      // 20260330 ZJH 任务类型（classification/detection/segmentation/anomaly_detection/instance_seg/ocr）
    std::string strTrainDate;     // 20260330 ZJH 训练完成日期（如 "2026-03-30"）
    int nEpochs = 0;              // 20260330 ZJH 总训练轮数
    int nBatchSize = 0;           // 20260330 ZJH 训练批次大小
    float fFinalLoss = 0.0f;      // 20260330 ZJH 最终训练损失
    float fBestValLoss = 0.0f;    // 20260330 ZJH 最佳验证损失
    std::string strOptimizer;     // 20260330 ZJH 优化器名称（如 "Adam", "SGD"）
    float fLearningRate = 0.0f;   // 20260330 ZJH 初始学习率

    // ------ 评估指标 ------
    float fAccuracy = 0.0f;       // 20260330 ZJH 总体准确率 [0, 1]
    float fPrecision = 0.0f;      // 20260330 ZJH 宏平均精确率 [0, 1]
    float fRecall = 0.0f;         // 20260330 ZJH 宏平均召回率 [0, 1]
    float fF1 = 0.0f;             // 20260330 ZJH 宏平均 F1 分数 [0, 1]
    float fMeanIoU = 0.0f;        // 20260330 ZJH 平均交并比（分割任务）[0, 1]
    float fMeanAP = 0.0f;         // 20260330 ZJH 平均精度均值（检测任务）[0, 1]
    std::vector<std::vector<int>> matConfusion;  // 20260330 ZJH 混淆矩阵 [nClasses x nClasses]
    std::vector<std::string> vecClassNames;      // 20260330 ZJH 类别名称列表

    // ------ 训练曲线数据 ------
    std::vector<float> vecTrainLoss;  // 20260330 ZJH 每轮训练损失
    std::vector<float> vecValLoss;    // 20260330 ZJH 每轮验证损失

    // ------ 推理性能 ------
    float fInferenceTimeMs = 0.0f;  // 20260330 ZJH 单张推理耗时（毫秒）
    int nModelSizeMB = 0;           // 20260330 ZJH 模型文件大小（MB）
};

// =========================================================
// 内部工具函数（匿名命名空间，头文件内联实现）
// =========================================================
namespace detail {

// 20260330 ZJH htmlEscape — HTML 特殊字符转义（防 XSS 注入）
// 参数: strInput - 可能包含 HTML 特殊字符的原始字符串
// 返回: 经过转义的安全字符串
inline std::string htmlEscape(const std::string& strInput) {
    std::string strResult;
    strResult.reserve(strInput.size() + 16);  // 20260330 ZJH 预分配略大空间
    for (char ch : strInput) {
        switch (ch) {
            case '&':  strResult += "&amp;";  break;   // 20260330 ZJH & 号转义
            case '<':  strResult += "&lt;";   break;   // 20260330 ZJH 左尖括号转义
            case '>':  strResult += "&gt;";   break;   // 20260330 ZJH 右尖括号转义
            case '"':  strResult += "&quot;"; break;    // 20260330 ZJH 双引号转义
            case '\'': strResult += "&#39;";  break;    // 20260330 ZJH 单引号转义
            default:   strResult += ch;        break;   // 20260330 ZJH 普通字符直接输出
        }
    }
    return strResult;
}

// 20260330 ZJH jsonEscape — JSON 字符串转义
// 参数: strInput - 原始字符串
// 返回: JSON 安全的字符串（不含外层引号）
inline std::string jsonEscape(const std::string& strInput) {
    std::string strResult;
    strResult.reserve(strInput.size() + 8);
    for (char ch : strInput) {
        switch (ch) {
            case '"':  strResult += "\\\""; break;  // 20260330 ZJH 双引号转义
            case '\\': strResult += "\\\\"; break;  // 20260330 ZJH 反斜杠转义
            case '\n': strResult += "\\n";  break;  // 20260330 ZJH 换行符转义
            case '\r': strResult += "\\r";  break;  // 20260330 ZJH 回车符转义
            case '\t': strResult += "\\t";  break;  // 20260330 ZJH 制表符转义
            default:   strResult += ch;      break;
        }
    }
    return strResult;
}

// 20260330 ZJH csvSanitize — CSV 注入防护（CWE-1236）
// 阻止以 = + - @ 开头的单元格被 Excel 解析为公式
inline std::string csvSanitize(const std::string& strField) {
    std::string strSafe = strField;
    if (!strSafe.empty()) {
        char chFirst = strSafe[0];
        // 20260330 ZJH 检测危险前缀字符
        if (chFirst == '=' || chFirst == '+' || chFirst == '-' || chFirst == '@') {
            strSafe = "'" + strSafe;  // 20260330 ZJH 前置单引号使 Excel 视为文本
        }
    }
    // 20260330 ZJH RFC 4180 合规：包含逗号、双引号或换行的字段用双引号包裹
    bool bNeedsQuoting = strSafe.find(',') != std::string::npos ||
                         strSafe.find('"') != std::string::npos ||
                         strSafe.find('\n') != std::string::npos;
    if (bNeedsQuoting) {
        std::string strQuoted = "\"";
        for (char ch : strSafe) {
            if (ch == '"') strQuoted += "\"\"";  // 20260330 ZJH 双引号转义
            else strQuoted += ch;
        }
        strQuoted += "\"";
        return strQuoted;
    }
    return strSafe;
}

// 20260330 ZJH floatToStr — 格式化浮点数为字符串
// fValue: 浮点值
// nDecimals: 小数位数
// 返回: 格式化字符串
inline std::string floatToStr(float fValue, int nDecimals = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(nDecimals) << fValue;
    return oss.str();
}

// 20260330 ZJH percentStr — 浮点数转百分比字符串
inline std::string percentStr(float fValue, int nDecimals = 1) {
    return floatToStr(fValue * 100.0f, nDecimals) + "%";
}

// 20260330 ZJH getCurrentDateTime — 获取当前日期时间字符串
inline std::string getCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    auto tTime = std::chrono::system_clock::to_time_t(now);
    struct tm tmBuf;
#ifdef _WIN32
    localtime_s(&tmBuf, &tTime);  // 20260330 ZJH Windows 线程安全版本
#else
    localtime_r(&tTime, &tmBuf);  // 20260330 ZJH POSIX 线程安全版本
#endif
    char szBuf[64];
    std::strftime(szBuf, sizeof(szBuf), "%Y-%m-%d %H:%M:%S", &tmBuf);
    return std::string(szBuf);
}

// 20260330 ZJH computePerClassMetrics — 从混淆矩阵计算每类 Precision/Recall/F1
// matConfusion: 混淆矩阵
// vecPrecision: 输出精确率向量
// vecRecall: 输出召回率向量
// vecF1: 输出 F1 向量
// vecSupport: 输出每类样本数向量
inline void computePerClassMetrics(
    const std::vector<std::vector<int>>& matConfusion,
    std::vector<float>& vecPrecision,
    std::vector<float>& vecRecall,
    std::vector<float>& vecF1,
    std::vector<int>& vecSupport)
{
    int nClasses = static_cast<int>(matConfusion.size());
    vecPrecision.resize(nClasses, 0.0f);
    vecRecall.resize(nClasses, 0.0f);
    vecF1.resize(nClasses, 0.0f);
    vecSupport.resize(nClasses, 0);

    for (int i = 0; i < nClasses; ++i) {
        // 20260330 ZJH 计算行和（该类真实样本总数 = support）
        int nRowSum = 0;
        for (int j = 0; j < nClasses; ++j) {
            nRowSum += matConfusion[i][j];
        }
        vecSupport[i] = nRowSum;

        // 20260330 ZJH 计算列和（被预测为该类的总数）
        int nColSum = 0;
        for (int j = 0; j < nClasses; ++j) {
            nColSum += matConfusion[j][i];
        }

        int nTP = matConfusion[i][i];  // 20260330 ZJH 真阳性
        float fP = (nColSum > 0) ? static_cast<float>(nTP) / nColSum : 0.0f;
        float fR = (nRowSum > 0) ? static_cast<float>(nTP) / nRowSum : 0.0f;
        float fSum = fP + fR;
        float fF = (fSum > 1e-7f) ? (2.0f * fP * fR / fSum) : 0.0f;

        vecPrecision[i] = fP;
        vecRecall[i] = fR;
        vecF1[i] = fF;
    }
}

// 20260330 ZJH heatmapColor — 根据归一化值 [0,1] 生成热力图 RGB 颜色
// fNorm: 归一化值
// bDiagonal: 是否对角线元素（对角线用绿色系，非对角线用红色系）
// 返回: CSS rgba 颜色字符串
inline std::string heatmapColor(float fNorm, bool bDiagonal) {
    std::ostringstream oss;
    if (bDiagonal) {
        // 20260330 ZJH 对角线：青色/绿色系（OmniMatch 品牌色 #00bcd4）
        float fAlpha = 0.1f + 0.7f * fNorm;
        oss << "rgba(0, 188, 212, " << std::fixed << std::setprecision(2) << fAlpha << ")";
    } else {
        // 20260330 ZJH 非对角线：红色系（错误预测）
        float fAlpha = 0.05f + 0.5f * fNorm;
        oss << "rgba(239, 68, 68, " << std::fixed << std::setprecision(2) << fAlpha << ")";
    }
    return oss.str();
}

}  // namespace detail

// =========================================================
// HTML 报告导出
// =========================================================

// 20260330 ZJH exportHTML — 生成自包含 HTML 评估报告
// 包含: 标题栏 + 训练摘要表 + 混淆矩阵热力图(SVG+CSS) + 训练损失曲线(内联SVG)
//       + 每类 P/R/F1 表 + 推理性能摘要
// data: 报告数据
// strOutputPath: 输出 HTML 文件路径
// 返回: 导出成功返回 true
inline bool exportHTML(const ReportData& data, const std::string& strOutputPath) {
    // 20260330 ZJH 确保输出目录存在
    std::filesystem::path outputPath(strOutputPath);
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    std::ofstream file(strOutputPath);
    if (!file.is_open()) {
        return false;  // 20260330 ZJH 文件打开失败
    }

    // 20260330 ZJH 从混淆矩阵计算每类指标
    std::vector<float> vecPrecision, vecRecall, vecF1;
    std::vector<int> vecSupport;
    if (!data.matConfusion.empty()) {
        detail::computePerClassMetrics(data.matConfusion, vecPrecision, vecRecall, vecF1, vecSupport);
    }

    std::ostringstream html;  // 20260330 ZJH 使用 ostringstream 收集所有 HTML 内容

    // ====== 1. HTML 头部 + CSS ======
    html << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n";
    html << "<meta charset=\"UTF-8\">\n";
    html << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
    html << "<title>" << detail::htmlEscape(data.strModelName) << " - OmniMatch Evaluation Report</title>\n";

    // 20260330 ZJH 内联 CSS — OmniMatch 品牌暗色主题
    html << "<style>\n";
    // 20260330 ZJH 全局样式: 暗色背景 #1a1a2e, 浅色文字 #e2e8f0
    html << "  * { margin: 0; padding: 0; box-sizing: border-box; }\n";
    html << "  body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; "
            "background: #1a1a2e; color: #e2e8f0; line-height: 1.6; padding: 0; }\n";
    // 20260330 ZJH 顶部导航栏
    html << "  .navbar { background: #16213e; padding: 16px 32px; border-bottom: 2px solid #00bcd4; "
            "display: flex; justify-content: space-between; align-items: center; }\n";
    html << "  .navbar .brand { font-size: 20px; font-weight: 700; color: #00bcd4; letter-spacing: 1px; }\n";
    html << "  .navbar .meta { font-size: 13px; color: #94a3b8; }\n";
    // 20260330 ZJH 内容容器
    html << "  .container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }\n";
    // 20260330 ZJH 报告头部
    html << "  .report-header { text-align: center; margin-bottom: 40px; }\n";
    html << "  .report-header h1 { font-size: 28px; color: #ffffff; margin-bottom: 8px; }\n";
    html << "  .report-header .subtitle { font-size: 15px; color: #94a3b8; }\n";
    html << "  .report-header .task-badge { display: inline-block; background: #00bcd4; color: #1a1a2e; "
            "padding: 4px 16px; border-radius: 16px; font-weight: 600; font-size: 13px; margin-top: 8px; }\n";
    // 20260330 ZJH 段落标题
    html << "  h2 { color: #00bcd4; font-size: 20px; margin: 32px 0 16px; padding-bottom: 8px; "
            "border-bottom: 1px solid #2a2a4a; }\n";
    // 20260330 ZJH 指标卡片
    html << "  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin: 20px 0; }\n";
    html << "  .card { background: #16213e; border-radius: 10px; padding: 20px; text-align: center; "
            "border: 1px solid #2a2a4a; transition: transform 0.2s; }\n";
    html << "  .card:hover { transform: translateY(-2px); border-color: #00bcd4; }\n";
    html << "  .card .value { font-size: 32px; font-weight: 700; }\n";
    html << "  .card .label { color: #94a3b8; font-size: 13px; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }\n";
    html << "  .card.cyan .value { color: #00bcd4; }\n";
    html << "  .card.green .value { color: #10b981; }\n";
    html << "  .card.amber .value { color: #f59e0b; }\n";
    html << "  .card.purple .value { color: #8b5cf6; }\n";
    html << "  .card.blue .value { color: #3b82f6; }\n";
    html << "  .card.rose .value { color: #f43f5e; }\n";
    // 20260330 ZJH 表格样式
    html << "  table { width: 100%; border-collapse: collapse; margin: 16px 0; background: #16213e; border-radius: 8px; overflow: hidden; }\n";
    html << "  th { background: #0f3460; color: #00bcd4; padding: 12px 16px; text-align: left; "
            "font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }\n";
    html << "  td { padding: 10px 16px; border-bottom: 1px solid #2a2a4a; font-size: 14px; }\n";
    html << "  tr:last-child td { border-bottom: none; }\n";
    html << "  tr:hover td { background: rgba(0, 188, 212, 0.05); }\n";
    // 20260330 ZJH 混淆矩阵样式
    html << "  .cm-table { text-align: center; }\n";
    html << "  .cm-table th, .cm-table td { padding: 8px 12px; min-width: 60px; font-size: 13px; }\n";
    html << "  .cm-table th { background: #0f3460; }\n";
    html << "  .cm-corner { background: #1a1a2e !important; }\n";
    // 20260330 ZJH SVG 图表容器
    html << "  .chart-container { background: #16213e; border-radius: 10px; padding: 24px; margin: 20px 0; border: 1px solid #2a2a4a; }\n";
    html << "  .chart-container svg { display: block; margin: 0 auto; }\n";
    // 20260330 ZJH 训练摘要网格
    html << "  .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }\n";
    html << "  @media (max-width: 768px) { .summary-grid { grid-template-columns: 1fr; } }\n";
    // 20260330 ZJH 页脚
    html << "  .footer { text-align: center; color: #64748b; margin-top: 48px; padding: 24px; "
            "border-top: 1px solid #2a2a4a; font-size: 13px; }\n";
    html << "  .footer a { color: #00bcd4; text-decoration: none; }\n";
    // 20260330 ZJH 进度条样式（用于每类指标可视化）
    html << "  .bar-bg { background: #2a2a4a; border-radius: 4px; height: 8px; width: 100px; display: inline-block; vertical-align: middle; margin-left: 8px; }\n";
    html << "  .bar-fill { height: 8px; border-radius: 4px; }\n";
    html << "  .bar-fill.cyan { background: #00bcd4; }\n";
    html << "  .bar-fill.green { background: #10b981; }\n";
    html << "  .bar-fill.amber { background: #f59e0b; }\n";
    html << "</style>\n</head>\n<body>\n";

    // ====== 2. 导航栏 ======
    html << "<div class=\"navbar\">\n";
    html << "  <div class=\"brand\">OmniMatch</div>\n";
    html << "  <div class=\"meta\">Evaluation Report &mdash; " << detail::getCurrentDateTime() << "</div>\n";
    html << "</div>\n";

    // ====== 3. 报告头部 ======
    html << "<div class=\"container\">\n";
    html << "<div class=\"report-header\">\n";
    html << "  <h1>" << detail::htmlEscape(data.strModelName) << "</h1>\n";
    html << "  <div class=\"subtitle\">Training Date: " << detail::htmlEscape(data.strTrainDate) << "</div>\n";
    html << "  <div class=\"task-badge\">" << detail::htmlEscape(data.strTaskType) << "</div>\n";
    html << "</div>\n";

    // ====== 4. 指标概览卡片 ======
    html << "<h2>Metrics Overview</h2>\n";
    html << "<div class=\"cards\">\n";
    html << "  <div class=\"card cyan\"><div class=\"value\">" << detail::percentStr(data.fAccuracy)
         << "</div><div class=\"label\">Accuracy</div></div>\n";
    html << "  <div class=\"card green\"><div class=\"value\">" << detail::percentStr(data.fPrecision)
         << "</div><div class=\"label\">Precision</div></div>\n";
    html << "  <div class=\"card amber\"><div class=\"value\">" << detail::percentStr(data.fRecall)
         << "</div><div class=\"label\">Recall</div></div>\n";
    html << "  <div class=\"card purple\"><div class=\"value\">" << detail::percentStr(data.fF1)
         << "</div><div class=\"label\">F1 Score</div></div>\n";
    if (data.fMeanIoU > 0.0f) {
        html << "  <div class=\"card blue\"><div class=\"value\">" << detail::percentStr(data.fMeanIoU)
             << "</div><div class=\"label\">mIoU</div></div>\n";
    }
    if (data.fMeanAP > 0.0f) {
        html << "  <div class=\"card rose\"><div class=\"value\">" << detail::percentStr(data.fMeanAP)
             << "</div><div class=\"label\">mAP</div></div>\n";
    }
    html << "</div>\n";

    // ====== 5. 训练摘要表 ======
    html << "<h2>Training Summary</h2>\n";
    html << "<div class=\"summary-grid\">\n";
    html << "<table>\n";
    html << "  <tr><th colspan=\"2\">Training Configuration</th></tr>\n";
    html << "  <tr><td>Epochs</td><td>" << data.nEpochs << "</td></tr>\n";
    html << "  <tr><td>Batch Size</td><td>" << data.nBatchSize << "</td></tr>\n";
    html << "  <tr><td>Optimizer</td><td>" << detail::htmlEscape(data.strOptimizer) << "</td></tr>\n";
    html << "  <tr><td>Learning Rate</td><td>" << detail::floatToStr(data.fLearningRate, 6) << "</td></tr>\n";
    html << "</table>\n";
    html << "<table>\n";
    html << "  <tr><th colspan=\"2\">Training Results</th></tr>\n";
    html << "  <tr><td>Final Training Loss</td><td>" << detail::floatToStr(data.fFinalLoss, 6) << "</td></tr>\n";
    html << "  <tr><td>Best Validation Loss</td><td>" << detail::floatToStr(data.fBestValLoss, 6) << "</td></tr>\n";
    html << "  <tr><td>Model Size</td><td>" << data.nModelSizeMB << " MB</td></tr>\n";
    html << "  <tr><td>Inference Time</td><td>" << detail::floatToStr(data.fInferenceTimeMs, 2) << " ms</td></tr>\n";
    html << "</table>\n";
    html << "</div>\n";

    // ====== 6. 训练损失曲线 (内联 SVG) ======
    if (!data.vecTrainLoss.empty()) {
        html << "<h2>Training Loss Curve</h2>\n";
        html << "<div class=\"chart-container\">\n";

        // 20260330 ZJH SVG 图表参数
        const int nSvgW = 800, nSvgH = 400;           // 20260330 ZJH SVG 画布尺寸
        const int nPadL = 60, nPadR = 140, nPadT = 30, nPadB = 50;  // 20260330 ZJH 边距
        const int nPlotW = nSvgW - nPadL - nPadR;     // 20260330 ZJH 绘图区宽度
        const int nPlotH = nSvgH - nPadT - nPadB;     // 20260330 ZJH 绘图区高度

        // 20260330 ZJH 计算 Y 轴范围
        float fMaxLoss = *std::max_element(data.vecTrainLoss.begin(), data.vecTrainLoss.end());
        if (!data.vecValLoss.empty()) {
            fMaxLoss = std::max(fMaxLoss, *std::max_element(data.vecValLoss.begin(), data.vecValLoss.end()));
        }
        fMaxLoss *= 1.1f;  // 20260330 ZJH 上方留 10% 余量
        if (fMaxLoss < 1e-6f) fMaxLoss = 1.0f;  // 20260330 ZJH 防除零

        int nEpochs = static_cast<int>(data.vecTrainLoss.size());

        html << "<svg width=\"" << nSvgW << "\" height=\"" << nSvgH << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

        // 20260330 ZJH 背景
        html << "  <rect width=\"" << nSvgW << "\" height=\"" << nSvgH << "\" fill=\"#16213e\" rx=\"8\"/>\n";

        // 20260330 ZJH 绘图区背景
        html << "  <rect x=\"" << nPadL << "\" y=\"" << nPadT << "\" width=\"" << nPlotW
             << "\" height=\"" << nPlotH << "\" fill=\"#1a1a2e\"/>\n";

        // 20260330 ZJH Y 轴网格线和标签
        int nYTicks = 5;
        for (int i = 0; i <= nYTicks; ++i) {
            float fVal = fMaxLoss * (1.0f - static_cast<float>(i) / nYTicks);
            int nY = nPadT + static_cast<int>(static_cast<float>(i) / nYTicks * nPlotH);
            html << "  <line x1=\"" << nPadL << "\" y1=\"" << nY << "\" x2=\"" << (nPadL + nPlotW)
                 << "\" y2=\"" << nY << "\" stroke=\"#2a2a4a\" stroke-width=\"1\"/>\n";
            html << "  <text x=\"" << (nPadL - 8) << "\" y=\"" << (nY + 4)
                 << "\" fill=\"#94a3b8\" font-size=\"11\" text-anchor=\"end\">"
                 << detail::floatToStr(fVal, 3) << "</text>\n";
        }

        // 20260330 ZJH X 轴标签
        int nXTickInterval = std::max(1, nEpochs / 10);  // 20260330 ZJH 约 10 个刻度
        for (int i = 0; i < nEpochs; i += nXTickInterval) {
            float fX = nPadL + static_cast<float>(i) / std::max(1, nEpochs - 1) * nPlotW;  // 20260330 ZJH 防止 nEpochs==1 时除零
            html << "  <text x=\"" << static_cast<int>(fX) << "\" y=\"" << (nSvgH - nPadB + 20)
                 << "\" fill=\"#94a3b8\" font-size=\"11\" text-anchor=\"middle\">"
                 << (i + 1) << "</text>\n";
        }

        // 20260330 ZJH 轴标题
        html << "  <text x=\"" << (nPadL + nPlotW / 2) << "\" y=\"" << (nSvgH - 5)
             << "\" fill=\"#94a3b8\" font-size=\"12\" text-anchor=\"middle\">Epoch</text>\n";
        html << "  <text x=\"15\" y=\"" << (nPadT + nPlotH / 2)
             << "\" fill=\"#94a3b8\" font-size=\"12\" text-anchor=\"middle\" "
             << "transform=\"rotate(-90 15 " << (nPadT + nPlotH / 2) << ")\">Loss</text>\n";

        // 20260330 ZJH 辅助 lambda: 生成折线路径
        auto generatePath = [&](const std::vector<float>& vecLoss, const std::string& strColor) {
            std::ostringstream path;
            path << "  <polyline points=\"";
            for (int i = 0; i < static_cast<int>(vecLoss.size()); ++i) {
                float fX = nPadL + static_cast<float>(i) / std::max(1, nEpochs - 1) * nPlotW;  // 20260330 ZJH 防止 nEpochs==1 时除零
                float fY = nPadT + (1.0f - vecLoss[i] / fMaxLoss) * nPlotH;
                if (i > 0) path << " ";
                path << static_cast<int>(fX) << "," << static_cast<int>(fY);
            }
            path << "\" fill=\"none\" stroke=\"" << strColor
                 << "\" stroke-width=\"2\" stroke-linejoin=\"round\"/>\n";
            return path.str();
        };

        // 20260330 ZJH 绘制训练损失曲线（青色 #00bcd4）
        html << generatePath(data.vecTrainLoss, "#00bcd4");

        // 20260330 ZJH 绘制验证损失曲线（琥珀色 #f59e0b）
        if (!data.vecValLoss.empty()) {
            html << generatePath(data.vecValLoss, "#f59e0b");
        }

        // 20260330 ZJH 图例
        int nLegendX = nPadL + nPlotW + 16;
        int nLegendY = nPadT + 20;
        html << "  <rect x=\"" << nLegendX << "\" y=\"" << nLegendY
             << "\" width=\"12\" height=\"12\" fill=\"#00bcd4\" rx=\"2\"/>\n";
        html << "  <text x=\"" << (nLegendX + 18) << "\" y=\"" << (nLegendY + 10)
             << "\" fill=\"#e2e8f0\" font-size=\"12\">Train Loss</text>\n";
        if (!data.vecValLoss.empty()) {
            html << "  <rect x=\"" << nLegendX << "\" y=\"" << (nLegendY + 24)
                 << "\" width=\"12\" height=\"12\" fill=\"#f59e0b\" rx=\"2\"/>\n";
            html << "  <text x=\"" << (nLegendX + 18) << "\" y=\"" << (nLegendY + 34)
                 << "\" fill=\"#e2e8f0\" font-size=\"12\">Val Loss</text>\n";
        }

        html << "</svg>\n</div>\n";
    }

    // ====== 7. 混淆矩阵热力图 ======
    if (!data.matConfusion.empty()) {
        int nClasses = static_cast<int>(data.matConfusion.size());

        html << "<h2>Confusion Matrix</h2>\n";
        html << "<div class=\"chart-container\">\n";
        html << "<table class=\"cm-table\">\n";

        // 20260330 ZJH 查找矩阵全局最大值（用于归一化颜色强度）
        int nGlobalMax = 0;
        for (const auto& row : data.matConfusion) {
            for (int nVal : row) {
                nGlobalMax = std::max(nGlobalMax, nVal);
            }
        }
        if (nGlobalMax == 0) nGlobalMax = 1;  // 20260330 ZJH 防除零

        // 20260330 ZJH 表头: 列标签 = 预测类别
        html << "  <tr><th class=\"cm-corner\">Actual \\ Predicted</th>";
        for (int j = 0; j < nClasses; ++j) {
            std::string strName = (j < static_cast<int>(data.vecClassNames.size()))
                ? data.vecClassNames[j] : std::to_string(j);
            html << "<th>" << detail::htmlEscape(strName) << "</th>";
        }
        html << "</tr>\n";

        // 20260330 ZJH 矩阵数据行
        for (int i = 0; i < nClasses; ++i) {
            std::string strRowName = (i < static_cast<int>(data.vecClassNames.size()))
                ? data.vecClassNames[i] : std::to_string(i);
            html << "  <tr><th>" << detail::htmlEscape(strRowName) << "</th>";

            for (int j = 0; j < nClasses; ++j) {
                int nVal = data.matConfusion[i][j];
                float fNorm = static_cast<float>(nVal) / nGlobalMax;
                std::string strBg = detail::heatmapColor(fNorm, i == j);
                html << "<td style=\"background:" << strBg << ";font-weight:"
                     << (i == j ? "700" : "400") << "\">" << nVal << "</td>";
            }
            html << "</tr>\n";
        }
        html << "</table>\n</div>\n";
    }

    // ====== 8. 每类 Precision/Recall/F1 表 ======
    if (!vecPrecision.empty()) {
        int nClasses = static_cast<int>(vecPrecision.size());

        html << "<h2>Per-Class Metrics</h2>\n";
        html << "<table>\n";
        html << "  <tr><th>Class</th><th>Precision</th><th></th><th>Recall</th><th></th>"
                "<th>F1 Score</th><th></th><th>Support</th></tr>\n";

        for (int i = 0; i < nClasses; ++i) {
            std::string strName = (i < static_cast<int>(data.vecClassNames.size()))
                ? data.vecClassNames[i] : std::to_string(i);

            // 20260330 ZJH 构建带进度条的行
            html << "  <tr>";
            html << "<td>" << detail::htmlEscape(strName) << "</td>";

            // 20260330 ZJH Precision 值 + 进度条
            html << "<td>" << detail::percentStr(vecPrecision[i]) << "</td>";
            html << "<td><div class=\"bar-bg\"><div class=\"bar-fill cyan\" style=\"width:"
                 << static_cast<int>(vecPrecision[i] * 100) << "%\"></div></div></td>";

            // 20260330 ZJH Recall 值 + 进度条
            html << "<td>" << detail::percentStr(vecRecall[i]) << "</td>";
            html << "<td><div class=\"bar-bg\"><div class=\"bar-fill green\" style=\"width:"
                 << static_cast<int>(vecRecall[i] * 100) << "%\"></div></div></td>";

            // 20260330 ZJH F1 值 + 进度条
            html << "<td>" << detail::percentStr(vecF1[i]) << "</td>";
            html << "<td><div class=\"bar-bg\"><div class=\"bar-fill amber\" style=\"width:"
                 << static_cast<int>(vecF1[i] * 100) << "%\"></div></div></td>";

            // 20260330 ZJH Support（样本数）
            html << "<td>" << vecSupport[i] << "</td>";
            html << "</tr>\n";
        }

        html << "</table>\n";
    }

    // ====== 9. 推理性能摘要 ======
    html << "<h2>Inference Performance</h2>\n";
    html << "<div class=\"cards\">\n";
    html << "  <div class=\"card cyan\"><div class=\"value\">" << detail::floatToStr(data.fInferenceTimeMs, 2)
         << "</div><div class=\"label\">Inference (ms)</div></div>\n";
    float fFPS = (data.fInferenceTimeMs > 0.01f) ? 1000.0f / data.fInferenceTimeMs : 0.0f;
    html << "  <div class=\"card green\"><div class=\"value\">" << detail::floatToStr(fFPS, 1)
         << "</div><div class=\"label\">Throughput (FPS)</div></div>\n";
    html << "  <div class=\"card amber\"><div class=\"value\">" << data.nModelSizeMB
         << "</div><div class=\"label\">Model Size (MB)</div></div>\n";
    html << "</div>\n";

    // ====== 10. 页脚 ======
    html << "<div class=\"footer\">\n";
    html << "  Generated by <a href=\"#\">OmniMatch Deep Learning Tool</a><br>\n";
    html << "  " << detail::getCurrentDateTime() << "\n";
    html << "</div>\n";

    html << "</div>\n</body>\n</html>\n";

    // 20260330 ZJH 写入文件
    file << html.str();
    file.close();
    return true;
}

// =========================================================
// CSV 报告导出
// =========================================================

// 20260330 ZJH exportCSV — 生成 CSV 指标摘要
// 包含: 总体指标 + 每类 P/R/F1/Support + 训练配置
// data: 报告数据
// strOutputPath: 输出 CSV 文件路径
// 返回: 导出成功返回 true
inline bool exportCSV(const ReportData& data, const std::string& strOutputPath) {
    std::ofstream file(strOutputPath);
    if (!file.is_open()) return false;

    // 20260330 ZJH 总体指标段
    file << "OmniMatch Evaluation Report\n";
    file << "Model," << detail::csvSanitize(data.strModelName) << "\n";
    file << "Task," << detail::csvSanitize(data.strTaskType) << "\n";
    file << "Date," << detail::csvSanitize(data.strTrainDate) << "\n";
    file << "\n";

    // 20260330 ZJH 训练配置
    file << "Training Configuration\n";
    file << "Epochs," << data.nEpochs << "\n";
    file << "Batch Size," << data.nBatchSize << "\n";
    file << "Optimizer," << detail::csvSanitize(data.strOptimizer) << "\n";
    file << "Learning Rate," << detail::floatToStr(data.fLearningRate, 6) << "\n";
    file << "Final Loss," << detail::floatToStr(data.fFinalLoss, 6) << "\n";
    file << "Best Val Loss," << detail::floatToStr(data.fBestValLoss, 6) << "\n";
    file << "\n";

    // 20260330 ZJH 总体评估指标
    file << "Overall Metrics\n";
    file << "Accuracy," << detail::floatToStr(data.fAccuracy) << "\n";
    file << "Precision," << detail::floatToStr(data.fPrecision) << "\n";
    file << "Recall," << detail::floatToStr(data.fRecall) << "\n";
    file << "F1," << detail::floatToStr(data.fF1) << "\n";
    if (data.fMeanIoU > 0.0f) file << "mIoU," << detail::floatToStr(data.fMeanIoU) << "\n";
    if (data.fMeanAP > 0.0f) file << "mAP," << detail::floatToStr(data.fMeanAP) << "\n";
    file << "\n";

    // 20260330 ZJH 每类指标
    if (!data.matConfusion.empty()) {
        std::vector<float> vecP, vecR, vecF;
        std::vector<int> vecS;
        detail::computePerClassMetrics(data.matConfusion, vecP, vecR, vecF, vecS);

        int nClasses = static_cast<int>(vecP.size());
        file << "Per-Class Metrics\n";
        file << "Class,Precision,Recall,F1,Support\n";
        for (int i = 0; i < nClasses; ++i) {
            std::string strName = (i < static_cast<int>(data.vecClassNames.size()))
                ? data.vecClassNames[i] : std::to_string(i);
            file << detail::csvSanitize(strName) << ","
                 << detail::floatToStr(vecP[i]) << ","
                 << detail::floatToStr(vecR[i]) << ","
                 << detail::floatToStr(vecF[i]) << ","
                 << vecS[i] << "\n";
        }
        file << "\n";

        // 20260330 ZJH 混淆矩阵
        file << "Confusion Matrix\n";
        file << ",";
        for (int j = 0; j < nClasses; ++j) {
            std::string strName = (j < static_cast<int>(data.vecClassNames.size()))
                ? data.vecClassNames[j] : std::to_string(j);
            file << detail::csvSanitize(strName);
            if (j < nClasses - 1) file << ",";
        }
        file << "\n";
        for (int i = 0; i < nClasses; ++i) {
            std::string strName = (i < static_cast<int>(data.vecClassNames.size()))
                ? data.vecClassNames[i] : std::to_string(i);
            file << detail::csvSanitize(strName);
            for (int j = 0; j < nClasses; ++j) {
                file << "," << data.matConfusion[i][j];
            }
            file << "\n";
        }
        file << "\n";
    }

    // 20260330 ZJH 推理性能
    file << "Performance\n";
    file << "Inference Time (ms)," << detail::floatToStr(data.fInferenceTimeMs, 2) << "\n";
    float fFPS = (data.fInferenceTimeMs > 0.01f) ? 1000.0f / data.fInferenceTimeMs : 0.0f;
    file << "Throughput (FPS)," << detail::floatToStr(fFPS, 1) << "\n";
    file << "Model Size (MB)," << data.nModelSizeMB << "\n";

    file.close();
    return true;
}

// =========================================================
// JSON 报告导出
// =========================================================

// 20260330 ZJH exportJSON — 生成 JSON 格式评估报告（机器可读）
// data: 报告数据
// strOutputPath: 输出 JSON 文件路径
// 返回: 导出成功返回 true
inline bool exportJSON(const ReportData& data, const std::string& strOutputPath) {
    std::ofstream file(strOutputPath);
    if (!file.is_open()) return false;

    // 20260330 ZJH 从混淆矩阵计算每类指标
    std::vector<float> vecP, vecR, vecF;
    std::vector<int> vecS;
    if (!data.matConfusion.empty()) {
        detail::computePerClassMetrics(data.matConfusion, vecP, vecR, vecF, vecS);
    }

    // 20260330 ZJH 手写 JSON 输出（避免引入第三方 JSON 库）
    file << "{\n";
    file << "  \"report_version\": \"1.0\",\n";
    file << "  \"generator\": \"OmniMatch Deep Learning Tool\",\n";
    file << "  \"generated_at\": \"" << detail::jsonEscape(detail::getCurrentDateTime()) << "\",\n";

    // 20260330 ZJH 模型信息
    file << "  \"model\": {\n";
    file << "    \"name\": \"" << detail::jsonEscape(data.strModelName) << "\",\n";
    file << "    \"task_type\": \"" << detail::jsonEscape(data.strTaskType) << "\",\n";
    file << "    \"train_date\": \"" << detail::jsonEscape(data.strTrainDate) << "\",\n";
    file << "    \"size_mb\": " << data.nModelSizeMB << "\n";
    file << "  },\n";

    // 20260330 ZJH 训练配置
    file << "  \"training\": {\n";
    file << "    \"epochs\": " << data.nEpochs << ",\n";
    file << "    \"batch_size\": " << data.nBatchSize << ",\n";
    file << "    \"optimizer\": \"" << detail::jsonEscape(data.strOptimizer) << "\",\n";
    file << "    \"learning_rate\": " << detail::floatToStr(data.fLearningRate, 6) << ",\n";
    file << "    \"final_loss\": " << detail::floatToStr(data.fFinalLoss, 6) << ",\n";
    file << "    \"best_val_loss\": " << detail::floatToStr(data.fBestValLoss, 6) << "\n";
    file << "  },\n";

    // 20260330 ZJH 评估指标
    file << "  \"metrics\": {\n";
    file << "    \"accuracy\": " << detail::floatToStr(data.fAccuracy) << ",\n";
    file << "    \"precision\": " << detail::floatToStr(data.fPrecision) << ",\n";
    file << "    \"recall\": " << detail::floatToStr(data.fRecall) << ",\n";
    file << "    \"f1\": " << detail::floatToStr(data.fF1) << ",\n";
    file << "    \"mean_iou\": " << detail::floatToStr(data.fMeanIoU) << ",\n";
    file << "    \"mean_ap\": " << detail::floatToStr(data.fMeanAP) << "\n";
    file << "  },\n";

    // 20260330 ZJH 每类指标
    file << "  \"per_class\": [\n";
    int nClasses = static_cast<int>(vecP.size());
    for (int i = 0; i < nClasses; ++i) {
        std::string strName = (i < static_cast<int>(data.vecClassNames.size()))
            ? data.vecClassNames[i] : std::to_string(i);
        file << "    {\n";
        file << "      \"class\": \"" << detail::jsonEscape(strName) << "\",\n";
        file << "      \"precision\": " << detail::floatToStr(vecP[i]) << ",\n";
        file << "      \"recall\": " << detail::floatToStr(vecR[i]) << ",\n";
        file << "      \"f1\": " << detail::floatToStr(vecF[i]) << ",\n";
        file << "      \"support\": " << vecS[i] << "\n";
        file << "    }" << (i < nClasses - 1 ? "," : "") << "\n";
    }
    file << "  ],\n";

    // 20260330 ZJH 混淆矩阵
    file << "  \"confusion_matrix\": [\n";
    for (int i = 0; i < static_cast<int>(data.matConfusion.size()); ++i) {
        file << "    [";
        for (int j = 0; j < static_cast<int>(data.matConfusion[i].size()); ++j) {
            file << data.matConfusion[i][j];
            if (j < static_cast<int>(data.matConfusion[i].size()) - 1) file << ", ";
        }
        file << "]" << (i < static_cast<int>(data.matConfusion.size()) - 1 ? "," : "") << "\n";
    }
    file << "  ],\n";

    // 20260330 ZJH 类别名称
    file << "  \"class_names\": [";
    for (int i = 0; i < static_cast<int>(data.vecClassNames.size()); ++i) {
        file << "\"" << detail::jsonEscape(data.vecClassNames[i]) << "\"";
        if (i < static_cast<int>(data.vecClassNames.size()) - 1) file << ", ";
    }
    file << "],\n";

    // 20260330 ZJH 训练曲线数据
    file << "  \"train_loss_curve\": [";
    for (int i = 0; i < static_cast<int>(data.vecTrainLoss.size()); ++i) {
        file << detail::floatToStr(data.vecTrainLoss[i], 6);
        if (i < static_cast<int>(data.vecTrainLoss.size()) - 1) file << ", ";
    }
    file << "],\n";

    file << "  \"val_loss_curve\": [";
    for (int i = 0; i < static_cast<int>(data.vecValLoss.size()); ++i) {
        file << detail::floatToStr(data.vecValLoss[i], 6);
        if (i < static_cast<int>(data.vecValLoss.size()) - 1) file << ", ";
    }
    file << "],\n";

    // 20260330 ZJH 推理性能
    file << "  \"performance\": {\n";
    file << "    \"inference_time_ms\": " << detail::floatToStr(data.fInferenceTimeMs, 2) << ",\n";
    float fFPS = (data.fInferenceTimeMs > 0.01f) ? 1000.0f / data.fInferenceTimeMs : 0.0f;
    file << "    \"throughput_fps\": " << detail::floatToStr(fFPS, 1) << ",\n";
    file << "    \"model_size_mb\": " << data.nModelSizeMB << "\n";
    file << "  }\n";

    file << "}\n";
    file.close();
    return true;
}

}  // namespace om
