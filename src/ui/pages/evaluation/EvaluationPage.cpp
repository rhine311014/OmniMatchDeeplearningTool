// 20260322 ZJH EvaluationPage 实现
// 三栏布局：左面板评估配置、中央结果可视化、右面板信息日志
// 模拟评估生成随机混淆矩阵和指标，填充表格/卡片/热力图

#include "ui/pages/evaluation/EvaluationPage.h"    // 20260322 ZJH EvaluationPage 类声明
#include "ui/widgets/ConfusionMatrixHeatmap.h"      // 20260322 ZJH 混淆矩阵热力图
#include "ui/widgets/GradCAMOverlay.h"              // 20260324 ZJH Grad-CAM 热力图/二值化叠加控件
#include "core/evaluation/EvaluationResult.h"       // 20260322 ZJH 评估结果
#include "core/project/Project.h"                   // 20260322 ZJH 项目数据
#include "core/data/ImageDataset.h"                 // 20260322 ZJH 数据集
#include "core/DLTypes.h"                           // 20260322 ZJH 类型定义
#include "app/Application.h"                        // 20260322 ZJH 全局事件总线
#include "core/project/ProjectSerializer.h"         // 20260324 ZJH 项目序列化器（评估完成自动保存）

#include <QVBoxLayout>      // 20260322 ZJH 垂直布局
#include <QHBoxLayout>      // 20260322 ZJH 水平布局
#include <QFormLayout>      // 20260322 ZJH 表单布局
#include <QScrollArea>      // 20260322 ZJH 滚动区域
#include <QGroupBox>        // 20260322 ZJH 分组框
#include <QHeaderView>      // 20260322 ZJH 表格头视图
#include <QMessageBox>      // 20260322 ZJH 提示对话框
#include <QFileDialog>      // 20260322 ZJH 文件对话框
#include <QDateTime>        // 20260322 ZJH 时间戳
#include <QRandomGenerator> // 20260322 ZJH 随机数生成
#include <QPainter>         // 20260322 ZJH 推理结果渲染
#include <QGridLayout>      // 20260322 ZJH 样本网格布局
#include <cmath>            // 20260322 ZJH std::exp
#include <QFile>            // 20260322 ZJH 文件读写
#include <QTextStream>      // 20260322 ZJH 文本流

// 20260322 ZJH 通用暗色控件样式（与 TrainingPage 风格一致）
static const QString s_strControlStyle = QStringLiteral(
    "QComboBox, QSpinBox, QDoubleSpinBox {"
    "  background-color: #1a1d24;"
    "  color: #e2e8f0;"
    "  border: 1px solid #333842;"
    "  border-radius: 4px;"
    "  padding: 4px 8px;"
    "  min-height: 22px;"
    "}"
    "QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {"
    "  border-color: #2563eb;"
    "}"
    "QComboBox::drop-down {"
    "  border: none;"
    "  width: 20px;"
    "}"
    "QComboBox QAbstractItemView {"
    "  background-color: #1a1d24;"
    "  color: #e2e8f0;"
    "  border: 1px solid #333842;"
    "  selection-background-color: #2563eb;"
    "}"
    "QGroupBox {"
    "  color: #94a3b8;"
    "  border: 1px solid #2a2d35;"
    "  border-radius: 6px;"
    "  margin-top: 12px;"
    "  padding-top: 16px;"
    "  font-weight: bold;"
    "}"
    "QGroupBox::title {"
    "  subcontrol-origin: margin;"
    "  left: 10px;"
    "  padding: 0 6px;"
    "}"
    "QLabel {"
    "  color: #94a3b8;"
    "}"
);

// 20260322 ZJH 构造函数
EvaluationPage::EvaluationPage(QWidget* pParent)
    : BasePage(pParent)                    // 20260406 ZJH 初始化页面基类
    , m_pCboDataScope(nullptr)             // 20260406 ZJH 数据范围下拉框初始为空
    , m_pSpnConfThreshold(nullptr)         // 20260406 ZJH 置信度阈值微调框初始为空
    , m_pSpnIouThreshold(nullptr)          // 20260406 ZJH IoU 阈值微调框初始为空
    , m_pBtnRunEval(nullptr)               // 20260406 ZJH 运行评估按钮初始为空
    , m_pBtnClearResults(nullptr)          // 20260406 ZJH 清除结果按钮初始为空
    , m_pBtnExportCsv(nullptr)             // 20260406 ZJH 导出 CSV 按钮初始为空
    , m_pBtnExportJson(nullptr)            // 20260406 ZJH 导出 JSON 按钮初始为空
    , m_pBtnExportHtml(nullptr)            // 20260406 ZJH 导出 HTML 报告按钮初始为空
    , m_pCboReportTemplate(nullptr)        // 20260406 ZJH 报告模板下拉框初始为空
    , m_pLblCheckTrained(nullptr)          // 20260406 ZJH 前置检查-模型已训练标签初始为空
    , m_pLblCheckTestSet(nullptr)          // 20260406 ZJH 前置检查-测试集存在标签初始为空
    , m_pLblCheckLabeled(nullptr)          // 20260406 ZJH 前置检查-测试集已标注标签初始为空
    , m_pLblCheckModel(nullptr)            // 20260406 ZJH 前置检查-模型文件存在标签初始为空
    , m_pLblCheckEngine(nullptr)           // 20260406 ZJH 前置检查-推理引擎就绪标签初始为空
    , m_pProgressBar(nullptr)              // 20260406 ZJH 评估进度条初始为空
    , m_pLblStatus(nullptr)                // 20260406 ZJH 状态文字标签初始为空
    , m_pCardAccuracy(nullptr)             // 20260406 ZJH 准确率指标卡片初始为空
    , m_pCardPrecision(nullptr)            // 20260406 ZJH 精确率指标卡片初始为空
    , m_pCardRecall(nullptr)               // 20260406 ZJH 召回率指标卡片初始为空
    , m_pLblCardValue1(nullptr)            // 20260406 ZJH 准确率数值标签初始为空
    , m_pLblCardValue2(nullptr)            // 20260406 ZJH 精确率数值标签初始为空
    , m_pLblCardValue3(nullptr)            // 20260406 ZJH 召回率数值标签初始为空
    , m_pTblMetrics(nullptr)               // 20260406 ZJH 详细指标表格初始为空
    , m_pHeatmap(nullptr)                  // 20260406 ZJH 混淆矩阵热力图初始为空
    , m_pGradCAMOverlay(nullptr)           // 20260406 ZJH Grad-CAM 叠加控件初始为空
    , m_pCboGradCAMMode(nullptr)           // 20260406 ZJH GradCAM 显示模式下拉框初始为空
    , m_pSliderThreshold(nullptr)          // 20260406 ZJH 二值化阈值滑块初始为空
    , m_pLblThresholdValue(nullptr)        // 20260406 ZJH 阈值数值标签初始为空
    , m_pBtnNormCount(nullptr)             // 20260406 ZJH 归一化-计数按钮初始为空
    , m_pBtnNormRow(nullptr)               // 20260406 ZJH 归一化-行按钮初始为空
    , m_pBtnNormCol(nullptr)               // 20260406 ZJH 归一化-列按钮初始为空
    , m_pLblDatasetName(nullptr)           // 20260406 ZJH 数据集名称标签初始为空
    , m_pLblImageCount(nullptr)            // 20260406 ZJH 图像数标签初始为空
    , m_pLblLabelCount(nullptr)            // 20260406 ZJH 标签数标签初始为空
    , m_pLblAvgLatency(nullptr)            // 20260406 ZJH 平均延迟标签初始为空
    , m_pLblP95Latency(nullptr)            // 20260406 ZJH P95 延迟标签初始为空
    , m_pLblThroughput(nullptr)            // 20260406 ZJH 吞吐量标签初始为空
    , m_pTxtLog(nullptr)                   // 20260406 ZJH 日志文本框初始为空
    , m_pComparisonTable(nullptr)          // 20260406 ZJH 模型对比表格初始为空
    , m_pBtnClearComparison(nullptr)       // 20260406 ZJH 清除对比记录按钮初始为空
    , m_pSimTimer(nullptr)                 // 20260406 ZJH 模拟评估定时器初始为空
    , m_nSimProgress(0)                    // 20260406 ZJH 模拟评估当前进度初始为 0
    , m_nSimTotal(100)                     // 20260406 ZJH 模拟评估总数初始为 100
{
    // 20260406 ZJH 创建模拟评估定时器，间隔 30ms 触发一次进度更新
    m_pSimTimer = new QTimer(this);
    m_pSimTimer->setInterval(30);  // 20260406 ZJH 每 30ms 更新一次模拟进度
    connect(m_pSimTimer, &QTimer::timeout, this, &EvaluationPage::onSimulationTick);  // 20260406 ZJH 连接定时器信号到模拟进度槽

    QWidget* pLeft   = createLeftPanel();    // 20260406 ZJH 创建左面板（评估配置+操作+前置检查）
    QWidget* pCenter = createCenterPanel();  // 20260406 ZJH 创建中央面板（进度+指标卡片+详细表+混淆矩阵）
    QWidget* pRight  = createRightPanel();   // 20260406 ZJH 创建右面板（数据信息+性能+日志）

    // 20260322 ZJH 设置面板宽度
    setLeftPanelWidth(280);   // 20260322 ZJH 左面板 280px
    setRightPanelWidth(220);  // 20260322 ZJH 右面板 220px

    // 20260322 ZJH 调用基类方法组装三栏布局
    setupThreeColumnLayout(pLeft, pCenter, pRight);
}

// ===== BasePage 生命周期回调 =====

// 20260322 ZJH 页面进入前台时刷新前置检查和数据信息
void EvaluationPage::onEnter()
{
    refreshPreChecks();  // 20260322 ZJH 刷新前置检查
    refreshDataInfo();   // 20260322 ZJH 刷新数据信息
}

// 20260322 ZJH 页面离开前台
void EvaluationPage::onLeave()
{
    // 20260322 ZJH 当前无需处理
}

// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void EvaluationPage::onProjectLoadedImpl()
{
    refreshPreChecks();  // 20260322 ZJH 刷新前置检查
    refreshDataInfo();   // 20260322 ZJH 刷新数据信息
    // 20260324 ZJH 使用基类已赋值的 m_pProject 获取项目名
    if (m_pProject) {
        appendLog(QStringLiteral("[%1] 项目已加载: %2")
                  .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                  .arg(m_pProject->name()));
    }
}

// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void EvaluationPage::onProjectClosedImpl()
{
    onClearResults();  // 20260322 ZJH 清除结果
    refreshPreChecks();  // 20260322 ZJH 刷新前置检查
    refreshDataInfo();   // 20260322 ZJH 刷新数据信息
}

// ===== 槽函数 =====

// 20260322 ZJH 运行评估
void EvaluationPage::onRunEvaluation()
{
    appendLog(QStringLiteral("[%1] 开始评估...")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss")));

    // 20260322 ZJH 设置进度条
    m_nSimProgress = 0;
    m_nSimTotal = 100;
    m_pProgressBar->setRange(0, m_nSimTotal);  // 20260322 ZJH 进度范围 0~100
    m_pProgressBar->setValue(0);
    m_pProgressBar->setVisible(true);
    m_pLblStatus->setText(QStringLiteral("正在评估..."));

    // 20260322 ZJH 禁用运行按钮
    m_pBtnRunEval->setEnabled(false);

    // 20260322 ZJH 启动模拟定时器
    m_pSimTimer->start();
}

// 20260322 ZJH 模拟评估进度
void EvaluationPage::onSimulationTick()
{
    m_nSimProgress++;  // 20260322 ZJH 进度+1
    m_pProgressBar->setValue(m_nSimProgress);
    m_pLblStatus->setText(QStringLiteral("评估进度: %1/%2").arg(m_nSimProgress).arg(m_nSimTotal));

    // 20260322 ZJH 评估完成
    if (m_nSimProgress >= m_nSimTotal) {
        m_pSimTimer->stop();  // 20260322 ZJH 停止定时器

        // 20260322 ZJH 生成模拟结果
        m_lastResult = generateMockResult();

        // 20260322 ZJH 显示结果
        displayResult(m_lastResult);

        // 20260324 ZJH 将评估结果持久化到项目对象
        if (m_pProject) {
            m_pProject->setEvaluationResult(m_lastResult);  // 20260324 ZJH 保存评估结果到项目

            // 20260324 ZJH 自动保存项目（评估完成里程碑）
            QString strProjFile = m_pProject->path() + QStringLiteral("/") + m_pProject->name() + QStringLiteral(".dfproj");
            if (ProjectSerializer::save(m_pProject, strProjFile)) {
                m_pProject->setDirty(false);  // 20260324 ZJH 保存成功，重置脏标志
                Application::instance()->notifyProjectSaved();  // 20260324 ZJH 通知项目已保存
            }
        }

        // 20260322 ZJH 更新状态
        m_pLblStatus->setText(QStringLiteral("评估完成"));
        m_pBtnRunEval->setEnabled(true);

        appendLog(QStringLiteral("[%1] 评估完成 — 准确率: %2%  F1: %3")
                  .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                  .arg(m_lastResult.dAccuracy * 100, 0, 'f', 1)
                  .arg(m_lastResult.dF1Score, 0, 'f', 3));

        // 20260402 ZJH [OPT-3.9] 评估完成后自动追加模型对比记录
        {
            // 20260402 ZJH 从项目获取模型名称（如不可用则使用占位符）
            QString strModelName = QStringLiteral("Unknown");
            if (m_pProject) {
                strModelName = om::modelArchitectureToString(m_pProject->trainingConfig().eArchitecture);
            }
            // 20260402 ZJH 使用评估结果和模拟性能数据填充对比行
            double dLatencyMs = 5.0 + QRandomGenerator::global()->bounded(150) / 10.0;  // 20260402 ZJH 模拟延迟 5~20ms
            double dModelSizeMB = 10.0 + QRandomGenerator::global()->bounded(400) / 10.0;  // 20260402 ZJH 模拟大小 10~50MB
            double dTrainTimeSec = 30.0 + QRandomGenerator::global()->bounded(270);  // 20260402 ZJH 模拟训练时间 30~300s
            appendComparisonEntry(strModelName, m_lastResult.dAccuracy,
                                  dLatencyMs, dModelSizeMB, dTrainTimeSec);
        }

        // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
        Application::instance()->notifyEvaluationCompleted();
    }
}

// 20260322 ZJH 清除评估结果
void EvaluationPage::onClearResults()
{
    // 20260322 ZJH 重置卡片
    m_pLblCardValue1->setText("--");
    m_pLblCardValue2->setText("--");
    m_pLblCardValue3->setText("--");

    // 20260322 ZJH 清空表格
    m_pTblMetrics->setRowCount(0);

    // 20260322 ZJH 清空混淆矩阵
    m_pHeatmap->clear();

    // 20260322 ZJH 隐藏进度条
    m_pProgressBar->setVisible(false);
    m_pLblStatus->setText(QStringLiteral("就绪"));

    // 20260322 ZJH 重置性能标签
    m_pLblAvgLatency->setText("--");
    m_pLblP95Latency->setText("--");
    m_pLblThroughput->setText("--");

    // 20260324 ZJH 清空 GradCAM 叠加控件
    if (m_pGradCAMOverlay) {
        m_pGradCAMOverlay->clear();
    }

    // 20260322 ZJH 清空日志
    m_pTxtLog->clear();

    appendLog(QStringLiteral("[%1] 结果已清除")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss")));
}

// 20260322 ZJH 导出 CSV
void EvaluationPage::onExportCsv()
{
    // 20260322 ZJH 弹出文件保存对话框
    QString strPath = QFileDialog::getSaveFileName(this,
        QStringLiteral("导出评估结果 CSV"),
        QStringLiteral("evaluation_result.csv"),
        QStringLiteral("CSV 文件 (*.csv)"));

    if (strPath.isEmpty()) {
        return;  // 20260322 ZJH 用户取消
    }

    QFile file(strPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("导出失败"),
                             QStringLiteral("无法写入文件: %1").arg(strPath));
        return;  // 20260322 ZJH 文件打开失败
    }

    QTextStream stream(&file);
    // 20260322 ZJH 写入表头
    stream << "Class,Precision,Recall,F1,Support\n";

    // 20260322 ZJH 写入每行数据
    for (int r = 0; r < m_pTblMetrics->rowCount(); ++r) {
        for (int c = 0; c < m_pTblMetrics->columnCount(); ++c) {
            if (c > 0) stream << ",";
            QTableWidgetItem* pItem = m_pTblMetrics->item(r, c);
            stream << (pItem ? pItem->text() : "");
        }
        stream << "\n";
    }

    file.close();
    appendLog(QStringLiteral("[%1] CSV 已导出: %2")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
              .arg(strPath));

    QMessageBox::information(this, QStringLiteral("导出成功"),
                             QStringLiteral("评估结果已导出到:\n%1").arg(strPath));
}

// 20260330 ZJH 导出 JSON
void EvaluationPage::onExportJson()
{
    // 20260330 ZJH 弹出文件保存对话框
    QString strPath = QFileDialog::getSaveFileName(this,
        QStringLiteral("导出评估结果 JSON"),
        QStringLiteral("evaluation_result.json"),
        QStringLiteral("JSON 文件 (*.json)"));

    if (strPath.isEmpty()) {
        return;  // 20260330 ZJH 用户取消
    }

    QFile file(strPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("导出失败"),
                             QStringLiteral("无法写入文件: %1").arg(strPath));
        return;  // 20260330 ZJH 文件打开失败
    }

    QTextStream stream(&file);
    // 20260330 ZJH 构建 JSON 内容
    stream << "{\n";
    stream << "  \"accuracy\": " << m_lastResult.dAccuracy << ",\n";
    stream << "  \"f1_score\": " << m_lastResult.dF1Score << ",\n";
    stream << "  \"precision\": " << m_lastResult.dPrecision << ",\n";
    stream << "  \"recall\": " << m_lastResult.dRecall << ",\n";
    stream << "  \"report_template\": \"" << m_pCboReportTemplate->currentText() << "\",\n";
    stream << "  \"classes\": [\n";

    // 20260330 ZJH 写入每行指标数据
    for (int r = 0; r < m_pTblMetrics->rowCount(); ++r) {
        stream << "    {";
        for (int c = 0; c < m_pTblMetrics->columnCount(); ++c) {
            if (c > 0) stream << ", ";
            QTableWidgetItem* pItem = m_pTblMetrics->item(r, c);
            QString strVal = pItem ? pItem->text() : "";
            // 20260330 ZJH 第 0 列为类名（字符串），其余为数值
            if (c == 0) {
                stream << "\"class\": \"" << strVal << "\"";
            } else {
                stream << "\"col" << c << "\": " << strVal;
            }
        }
        stream << "}";
        if (r < m_pTblMetrics->rowCount() - 1) stream << ",";
        stream << "\n";
    }

    stream << "  ]\n";
    stream << "}\n";

    file.close();
    appendLog(QStringLiteral("[%1] JSON 已导出: %2")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
              .arg(strPath));

    QMessageBox::information(this, QStringLiteral("导出成功"),
                             QStringLiteral("评估结果已导出到:\n%1").arg(strPath));
}

// 20260322 ZJH 导出 HTML 报告
void EvaluationPage::onExportHtmlReport()
{
    // 20260322 ZJH 弹出文件保存对话框
    QString strPath = QFileDialog::getSaveFileName(this,
        QStringLiteral("导出评估报告 HTML"),
        QStringLiteral("evaluation_report.html"),
        QStringLiteral("HTML 文件 (*.html)"));

    if (strPath.isEmpty()) {
        return;  // 20260322 ZJH 用户取消
    }

    QFile file(strPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("导出失败"),
                             QStringLiteral("无法写入文件: %1").arg(strPath));
        return;  // 20260322 ZJH 文件打开失败
    }

    QTextStream stream(&file);
    // 20260322 ZJH 生成 HTML 报告
    stream << "<!DOCTYPE html><html><head><meta charset='utf-8'>"
           << "<title>OmniMatch Evaluation Report</title>"
           << "<style>body{font-family:sans-serif;background:#1a1d24;color:#e2e8f0;padding:20px;}"
           << "table{border-collapse:collapse;width:100%;margin:10px 0;}"
           << "th,td{border:1px solid #333842;padding:8px;text-align:center;}"
           << "th{background:#2563eb;}</style></head><body>"
           << "<h1>OmniMatch 模型评估报告</h1>"
           << "<p>生成时间: " << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss") << "</p>"
           << "<h2>总体指标</h2>"
           << "<table><tr><th>指标</th><th>值</th></tr>"
           << "<tr><td>准确率</td><td>" << QString::number(m_lastResult.dAccuracy * 100, 'f', 1) << "%</td></tr>"
           << "<tr><td>精确率</td><td>" << QString::number(m_lastResult.dPrecision * 100, 'f', 1) << "%</td></tr>"
           << "<tr><td>召回率</td><td>" << QString::number(m_lastResult.dRecall * 100, 'f', 1) << "%</td></tr>"
           << "<tr><td>F1 Score</td><td>" << QString::number(m_lastResult.dF1Score, 'f', 3) << "</td></tr>"
           << "</table>"
           << "</body></html>";

    file.close();
    appendLog(QStringLiteral("[%1] HTML 报告已导出: %2")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
              .arg(strPath));

    QMessageBox::information(this, QStringLiteral("导出成功"),
                             QStringLiteral("评估报告已导出到:\n%1").arg(strPath));
}

// 20260322 ZJH 归一化模式切换
void EvaluationPage::onNormModeChanged(int nMode)
{
    if (m_pHeatmap) m_pHeatmap->setNormMode(nMode);  // 20260322 ZJH 防御空指针

    // 20260322 ZJH 更新按钮样式
    QString strActive = QStringLiteral(
        "QPushButton { background: #2563eb; color: #fff; border: none; border-radius: 4px; padding: 4px 12px; }");
    QString strInactive = QStringLiteral(
        "QPushButton { background: #1a1d24; color: #94a3b8; border: 1px solid #333842; border-radius: 4px; padding: 4px 12px; }"
        "QPushButton:hover { border-color: #2563eb; }");

    m_pBtnNormCount->setStyleSheet(nMode == 0 ? strActive : strInactive);
    m_pBtnNormRow->setStyleSheet(nMode == 1 ? strActive : strInactive);
    m_pBtnNormCol->setStyleSheet(nMode == 2 ? strActive : strInactive);
}

// ===== UI 创建 =====

// 20260322 ZJH 创建左面板
QWidget* EvaluationPage::createLeftPanel()
{
    // 20260322 ZJH 滚动区域容器
    QScrollArea* pScrollArea = new QScrollArea();
    pScrollArea->setWidgetResizable(true);
    pScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    pScrollArea->setStyleSheet(QStringLiteral(
        "QScrollArea { background: #1e2230; border: none; border-right: 1px solid #2a2d35; }"));

    QWidget* pContainer = new QWidget();
    pContainer->setStyleSheet(s_strControlStyle);  // 20260322 ZJH 应用通用暗色样式
    QVBoxLayout* pLayout = new QVBoxLayout(pContainer);
    pLayout->setContentsMargins(12, 8, 12, 8);
    pLayout->setSpacing(6);

    // ===== 评估配置分组 =====
    QGroupBox* pGrpConfig = new QGroupBox(QStringLiteral("评估配置"));
    QFormLayout* pFormConfig = new QFormLayout(pGrpConfig);
    pFormConfig->setSpacing(6);

    // 20260322 ZJH 数据范围
    m_pCboDataScope = new QComboBox();
    m_pCboDataScope->addItems({
        QStringLiteral("训练集"),
        QStringLiteral("验证集"),
        QStringLiteral("测试集"),
        QStringLiteral("全部")
    });
    m_pCboDataScope->setCurrentIndex(2);  // 20260322 ZJH 默认选择测试集
    pFormConfig->addRow(QStringLiteral("数据范围:"), m_pCboDataScope);

    // 20260322 ZJH 置信度阈值
    m_pSpnConfThreshold = new QDoubleSpinBox();
    m_pSpnConfThreshold->setRange(0.0, 1.0);
    m_pSpnConfThreshold->setSingleStep(0.05);
    m_pSpnConfThreshold->setValue(0.5);
    m_pSpnConfThreshold->setDecimals(2);
    pFormConfig->addRow(QStringLiteral("置信度阈值:"), m_pSpnConfThreshold);

    // 20260322 ZJH IoU 阈值
    m_pSpnIouThreshold = new QDoubleSpinBox();
    m_pSpnIouThreshold->setRange(0.0, 1.0);
    m_pSpnIouThreshold->setSingleStep(0.05);
    m_pSpnIouThreshold->setValue(0.5);
    m_pSpnIouThreshold->setDecimals(2);
    pFormConfig->addRow(QStringLiteral("IoU 阈值:"), m_pSpnIouThreshold);

    pLayout->addWidget(pGrpConfig);

    // ===== 操作分组 =====
    QGroupBox* pGrpAction = new QGroupBox(QStringLiteral("操作"));
    QVBoxLayout* pActionLayout = new QVBoxLayout(pGrpAction);
    pActionLayout->setSpacing(6);

    // 20260322 ZJH 运行评估按钮（蓝色主按钮）
    m_pBtnRunEval = new QPushButton(QStringLiteral("运行评估"));
    m_pBtnRunEval->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #2563eb; color: white; border: none; border-radius: 6px;"
        "  padding: 8px 16px; font-weight: bold; font-size: 13px; }"
        "QPushButton:hover { background-color: #1d4ed8; }"
        "QPushButton:disabled { background-color: #475569; }"));
    connect(m_pBtnRunEval, &QPushButton::clicked, this, &EvaluationPage::onRunEvaluation);
    pActionLayout->addWidget(m_pBtnRunEval);

    // 20260322 ZJH 清除结果按钮
    m_pBtnClearResults = new QPushButton(QStringLiteral("清除结果"));
    m_pBtnClearResults->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 6px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #475569; }"));
    connect(m_pBtnClearResults, &QPushButton::clicked, this, &EvaluationPage::onClearResults);
    pActionLayout->addWidget(m_pBtnClearResults);

    // 20260322 ZJH 导出 CSV 按钮
    m_pBtnExportCsv = new QPushButton(QStringLiteral("导出 CSV"));
    m_pBtnExportCsv->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 6px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #475569; }"));
    connect(m_pBtnExportCsv, &QPushButton::clicked, this, &EvaluationPage::onExportCsv);
    pActionLayout->addWidget(m_pBtnExportCsv);

    // 20260330 ZJH 导出 JSON 按钮
    m_pBtnExportJson = new QPushButton(QStringLiteral("导出 JSON"));
    m_pBtnExportJson->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 6px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #475569; }"));
    connect(m_pBtnExportJson, &QPushButton::clicked, this, &EvaluationPage::onExportJson);
    pActionLayout->addWidget(m_pBtnExportJson);

    // 20260322 ZJH 导出 HTML 报告按��
    m_pBtnExportHtml = new QPushButton(QStringLiteral("导出 HTML 报告"));
    m_pBtnExportHtml->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 6px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #475569; }"));
    connect(m_pBtnExportHtml, &QPushButton::clicked, this, &EvaluationPage::onExportHtmlReport);
    pActionLayout->addWidget(m_pBtnExportHtml);

    // 20260330 ZJH 报告模板下拉框
    QFormLayout* pReportForm = new QFormLayout();
    m_pCboReportTemplate = new QComboBox();
    m_pCboReportTemplate->addItems({
        QStringLiteral("标准报告"),
        QStringLiteral("详细报告"),
        QStringLiteral("简洁报告")
    });
    pReportForm->addRow(QStringLiteral("报告模板:"), m_pCboReportTemplate);
    pActionLayout->addLayout(pReportForm);

    pLayout->addWidget(pGrpAction);

    // ===== 前置检查分组 =====
    QGroupBox* pGrpCheck = new QGroupBox(QStringLiteral("前置检查"));
    QVBoxLayout* pCheckLayout = new QVBoxLayout(pGrpCheck);
    pCheckLayout->setSpacing(4);

    // 20260322 ZJH 创建 5 个检查项标签
    auto createCheckLabel = [&](const QString& strText) -> QLabel* {
        QLabel* pLbl = new QLabel(strText);
        pLbl->setStyleSheet(QStringLiteral("QLabel { color: #64748b; font-size: 12px; }"));
        pCheckLayout->addWidget(pLbl);
        return pLbl;
    };

    m_pLblCheckTrained = createCheckLabel(QStringLiteral("✗ 模型已训练"));
    m_pLblCheckTestSet = createCheckLabel(QStringLiteral("✗ 测试集存在"));
    m_pLblCheckLabeled = createCheckLabel(QStringLiteral("✗ 测试集已标注"));
    m_pLblCheckModel   = createCheckLabel(QStringLiteral("✗ 模型文件存在"));
    m_pLblCheckEngine  = createCheckLabel(QStringLiteral("✗ 推理引擎就绪"));

    pLayout->addWidget(pGrpCheck);

    pLayout->addStretch(1);  // 20260322 ZJH 弹性空间推到底部
    pScrollArea->setWidget(pContainer);
    return pScrollArea;
}

// 20260322 ZJH 创建中央面板
QWidget* EvaluationPage::createCenterPanel()
{
    // 20260326 ZJH QScrollArea 包裹中间面板，内容多时可滚动
    QScrollArea* pScroll = new QScrollArea();
    pScroll->setWidgetResizable(true);
    pScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    pScroll->setStyleSheet(QStringLiteral(
        "QScrollArea { background: #22262e; border: none; }"
        "QScrollBar:vertical { background: #22262e; width: 8px; }"
        "QScrollBar::handle:vertical { background: #3b4252; border-radius: 4px; min-height: 30px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"));

    QWidget* pPanel = new QWidget();
    pPanel->setStyleSheet(QStringLiteral("QWidget { background: #22262e; }"));
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(16, 12, 16, 12);
    pLayout->setSpacing(12);

    m_pLblStatus = new QLabel(QStringLiteral("就绪"));
    m_pLblStatus->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 13px; }"));
    pLayout->addWidget(m_pLblStatus);

    m_pProgressBar = new QProgressBar();
    m_pProgressBar->setRange(0, 100);
    m_pProgressBar->setValue(0);
    m_pProgressBar->setVisible(false);  // 20260322 ZJH 初始隐藏
    m_pProgressBar->setFixedHeight(6);
    m_pProgressBar->setTextVisible(false);
    m_pProgressBar->setStyleSheet(QStringLiteral(
        "QProgressBar { background: #1a1d24; border: none; border-radius: 3px; }"
        "QProgressBar::chunk { background: #2563eb; border-radius: 3px; }"));
    pLayout->addWidget(m_pProgressBar);

    QHBoxLayout* pCardsLayout = new QHBoxLayout();
    pCardsLayout->setSpacing(12);

    m_pCardAccuracy  = createMetricCard(QStringLiteral("准确率 / mAP"),   "#2563eb", m_pLblCardValue1);
    m_pCardPrecision = createMetricCard(QStringLiteral("精确率 / mIoU"),   "#f59e0b", m_pLblCardValue2);
    m_pCardRecall    = createMetricCard(QStringLiteral("召回率 / F1"),     "#10b981", m_pLblCardValue3);

    pCardsLayout->addWidget(m_pCardAccuracy);
    pCardsLayout->addWidget(m_pCardPrecision);
    pCardsLayout->addWidget(m_pCardRecall);
    pLayout->addLayout(pCardsLayout);

    m_pTblMetrics = new QTableWidget();
    m_pTblMetrics->setColumnCount(5);
    m_pTblMetrics->setHorizontalHeaderLabels({
        QStringLiteral("类别"),
        QStringLiteral("精确率"),
        QStringLiteral("召回率"),
        QStringLiteral("F1"),
        QStringLiteral("支持数")
    });
    m_pTblMetrics->horizontalHeader()->setStretchLastSection(true);
    m_pTblMetrics->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pTblMetrics->setEditTriggers(QAbstractItemView::NoEditTriggers);  // 20260322 ZJH 禁止编辑
    m_pTblMetrics->setAlternatingRowColors(true);
    m_pTblMetrics->setMaximumHeight(180);
    m_pTblMetrics->setStyleSheet(QStringLiteral(
        "QTableWidget { background: #1a1d24; color: #e2e8f0; border: 1px solid #2a2d35;"
        "  gridline-color: #333842; border-radius: 4px; }"
        "QTableWidget::item { padding: 4px; }"
        "QTableWidget::item:alternate { background: #1e2230; }"
        "QHeaderView::section { background: #2a2d35; color: #94a3b8; border: none;"
        "  padding: 6px; font-weight: bold; }"));
    pLayout->addWidget(m_pTblMetrics);

    QLabel* pLblHeatmapTitle = new QLabel(QStringLiteral("混淆矩阵"));
    pLblHeatmapTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 14px; font-weight: bold; }"));
    pLayout->addWidget(pLblHeatmapTitle);

    // 20260322 ZJH 归一化切换按钮
    QHBoxLayout* pNormLayout = new QHBoxLayout();
    pNormLayout->setSpacing(4);

    m_pBtnNormCount = new QPushButton(QStringLiteral("计数"));
    m_pBtnNormRow   = new QPushButton(QStringLiteral("行归一化"));
    m_pBtnNormCol   = new QPushButton(QStringLiteral("列归一化"));

    // 20260322 ZJH 设置初始样式（计数为默认活动状态）
    connect(m_pBtnNormCount, &QPushButton::clicked, this, [this]() { onNormModeChanged(0); });
    connect(m_pBtnNormRow,   &QPushButton::clicked, this, [this]() { onNormModeChanged(1); });
    connect(m_pBtnNormCol,   &QPushButton::clicked, this, [this]() { onNormModeChanged(2); });

    pNormLayout->addWidget(m_pBtnNormCount);
    pNormLayout->addWidget(m_pBtnNormRow);
    pNormLayout->addWidget(m_pBtnNormCol);
    pNormLayout->addStretch(1);
    pLayout->addLayout(pNormLayout);

    m_pHeatmap = new ConfusionMatrixHeatmap();
    m_pHeatmap->setMinimumHeight(250);
    pLayout->addWidget(m_pHeatmap, 1);
    onNormModeChanged(0);

    // 20260322 ZJH ===== 推理结果渲染图 =====
    m_pLblInferTitle = new QLabel(QStringLiteral("推理结果预览"));
    m_pLblInferTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 14px; font-weight: bold; }"));
    pLayout->addWidget(m_pLblInferTitle);

    // 20260322 ZJH 样本推理网格（2行4列 = 8个样本预览）
    m_pInferGrid = new QWidget();
    m_pInferGrid->setMinimumHeight(200);
    QGridLayout* pGridLayout = new QGridLayout(m_pInferGrid);
    pGridLayout->setSpacing(6);
    for (int i = 0; i < 8; ++i) {
        QLabel* pThumb = new QLabel();
        pThumb->setFixedSize(120, 90);
        pThumb->setAlignment(Qt::AlignCenter);
        pThumb->setStyleSheet(QStringLiteral(
            "QLabel { background: #1a1d24; border: 2px solid #333842; border-radius: 4px; color: #64748b; font-size: 11px; }"));
        pThumb->setText(QStringLiteral("样本 %1").arg(i + 1));
        m_vecInferThumbs.append(pThumb);
        pGridLayout->addWidget(pThumb, i / 4, i % 4);
    }
    pLayout->addWidget(m_pInferGrid);

    // 20260322 ZJH ===== 缺陷概率热力图 =====
    QLabel* pLblDefectTitle = new QLabel(QStringLiteral("缺陷概率图"));
    pLblDefectTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 14px; font-weight: bold; }"));
    pLayout->addWidget(pLblDefectTitle);

    // 20260324 ZJH GradCAM 显示模式切换控件（热力图 / 二值化缺陷图）
    QHBoxLayout* pGradCAMCtrlLayout = new QHBoxLayout();
    pGradCAMCtrlLayout->setContentsMargins(0, 0, 0, 0);

    // 20260324 ZJH 显示模式下拉框
    QLabel* pLblMode = new QLabel(QStringLiteral("模式:"));
    pLblMode->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 11px; }"));
    pGradCAMCtrlLayout->addWidget(pLblMode);

    m_pCboGradCAMMode = new QComboBox();
    m_pCboGradCAMMode->addItem(QStringLiteral("热力图"), static_cast<int>(GradCAMDisplayMode::Heatmap));
    m_pCboGradCAMMode->addItem(QStringLiteral("二值化缺陷图"), static_cast<int>(GradCAMDisplayMode::Binary));
    m_pCboGradCAMMode->setCurrentIndex(1);  // 20260324 ZJH 默认二值化模式
    m_pCboGradCAMMode->setStyleSheet(QStringLiteral(
        "QComboBox { background-color: #1a1d24; color: #e2e8f0; border: 1px solid #333842; "
        "border-radius: 3px; padding: 2px 6px; font-size: 11px; min-width: 100px; }"
        "QComboBox:hover { border-color: #2563eb; }"
        "QComboBox::drop-down { border: none; width: 18px; }"
        "QComboBox QAbstractItemView { background-color: #1a1d24; color: #e2e8f0; "
        "border: 1px solid #333842; selection-background-color: #2563eb; }"));
    pGradCAMCtrlLayout->addWidget(m_pCboGradCAMMode);

    pGradCAMCtrlLayout->addSpacing(10);

    // 20260324 ZJH 二值化阈值滑块
    QLabel* pLblThreshold = new QLabel(QStringLiteral("阈值:"));
    pLblThreshold->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 11px; }"));
    pGradCAMCtrlLayout->addWidget(pLblThreshold);

    m_pSliderThreshold = new QSlider(Qt::Horizontal);
    m_pSliderThreshold->setRange(0, 100);  // 20260324 ZJH 0~100 映射到 0.0~1.0
    m_pSliderThreshold->setValue(50);       // 20260324 ZJH 默认阈值 0.5
    m_pSliderThreshold->setFixedWidth(100);
    m_pSliderThreshold->setStyleSheet(QStringLiteral(
        "QSlider::groove:horizontal { background: #333842; height: 4px; border-radius: 2px; }"
        "QSlider::handle:horizontal { background: #2563eb; width: 12px; margin: -4px 0; border-radius: 6px; }"));
    pGradCAMCtrlLayout->addWidget(m_pSliderThreshold);

    m_pLblThresholdValue = new QLabel(QStringLiteral("0.50"));
    m_pLblThresholdValue->setStyleSheet(QStringLiteral("QLabel { color: #e2e8f0; font-size: 11px; min-width: 30px; }"));
    pGradCAMCtrlLayout->addWidget(m_pLblThresholdValue);

    pGradCAMCtrlLayout->addStretch(1);
    pLayout->addLayout(pGradCAMCtrlLayout);

    // 20260324 ZJH GradCAM 叠加控件（替代原有的 QLabel 缺陷图）
    m_pGradCAMOverlay = new GradCAMOverlay();
    m_pGradCAMOverlay->setMinimumHeight(150);
    m_pGradCAMOverlay->setDisplayMode(GradCAMDisplayMode::Binary);  // 20260324 ZJH 默认二值化模式
    m_pGradCAMOverlay->setThreshold(0.5);  // 20260324 ZJH 默认阈值 0.5
    pLayout->addWidget(m_pGradCAMOverlay);

    // 20260324 ZJH 保留原始缺陷图 QLabel 作为备用/占位
    m_pLblDefectMap = new QLabel();
    m_pLblDefectMap->setMinimumHeight(0);
    m_pLblDefectMap->setAlignment(Qt::AlignCenter);
    m_pLblDefectMap->setVisible(false);  // 20260324 ZJH 隐藏原始 QLabel，使用 GradCAMOverlay 替代
    pLayout->addWidget(m_pLblDefectMap);

    // 20260402 ZJH ===== [OPT-3.9] 模型对比面板 =====
    // 每次评估完成后自动追加一行，方便用户对比不同模型的性能差异
    QLabel* pLblCompTitle = new QLabel(QStringLiteral("模型对比"));
    pLblCompTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 14px; font-weight: bold; }"));
    pLayout->addWidget(pLblCompTitle);

    // 20260402 ZJH 对比表格: 5 列（模型名称|精度|推理延迟|模型大小|训练时间）
    m_pComparisonTable = new QTableWidget();
    m_pComparisonTable->setColumnCount(5);
    m_pComparisonTable->setHorizontalHeaderLabels({
        QStringLiteral("模型名称"),
        QStringLiteral("精度 (%)"),
        QStringLiteral("推理延迟 (ms)"),
        QStringLiteral("模型大小 (MB)"),
        QStringLiteral("训练时间 (s)")
    });
    m_pComparisonTable->horizontalHeader()->setStretchLastSection(true);
    m_pComparisonTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pComparisonTable->setEditTriggers(QAbstractItemView::NoEditTriggers);  // 20260402 ZJH 禁止编辑
    m_pComparisonTable->setAlternatingRowColors(true);   // 20260402 ZJH 交替行颜色
    m_pComparisonTable->setMaximumHeight(160);           // 20260402 ZJH 限制最大高度
    m_pComparisonTable->setStyleSheet(QStringLiteral(
        "QTableWidget { background: #1a1d24; color: #e2e8f0; border: 1px solid #2a2d35;"
        "  gridline-color: #333842; border-radius: 4px; }"
        "QTableWidget::item { padding: 4px; }"
        "QTableWidget::item:alternate { background: #1e2230; }"
        "QHeaderView::section { background: #2a2d35; color: #94a3b8; border: none;"
        "  padding: 6px; font-weight: bold; }"));
    pLayout->addWidget(m_pComparisonTable);

    // 20260402 ZJH 清除对比记录按钮
    m_pBtnClearComparison = new QPushButton(QStringLiteral("清除对比记录"));
    m_pBtnClearComparison->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 4px; padding: 4px 12px; font-size: 12px; }"
        "QPushButton:hover { background-color: #475569; }"));
    connect(m_pBtnClearComparison, &QPushButton::clicked, this, &EvaluationPage::onClearComparison);
    pLayout->addWidget(m_pBtnClearComparison);

    // 20260324 ZJH 连接显示模式下拉框信号
    connect(m_pCboGradCAMMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int nIndex) {
                // 20260324 ZJH 从下拉框数据获取显示模式
                int nMode = m_pCboGradCAMMode->itemData(nIndex).toInt();
                GradCAMDisplayMode eMode = static_cast<GradCAMDisplayMode>(nMode);
                m_pGradCAMOverlay->setDisplayMode(eMode);
                // 20260324 ZJH 仅在二值化模式下启用阈值滑块
                m_pSliderThreshold->setEnabled(eMode == GradCAMDisplayMode::Binary);
            });

    // 20260324 ZJH 连接阈值滑块信号
    connect(m_pSliderThreshold, &QSlider::valueChanged,
            this, [this](int nValue) {
                double dThreshold = nValue / 100.0;  // 20260324 ZJH 整数映射到 [0, 1]
                m_pLblThresholdValue->setText(QString::number(dThreshold, 'f', 2));
                m_pGradCAMOverlay->setThreshold(dThreshold);
            });

    pScroll->setWidget(pPanel);  // 20260326 ZJH 将内容放入滚动区域
    return pScroll;
}

// 20260322 ZJH 创建右面板
QWidget* EvaluationPage::createRightPanel()
{
    QScrollArea* pScrollArea = new QScrollArea();
    pScrollArea->setWidgetResizable(true);
    pScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    pScrollArea->setStyleSheet(QStringLiteral(
        "QScrollArea { background: #1e2230; border: none; border-left: 1px solid #2a2d35; }"));

    QWidget* pContainer = new QWidget();
    pContainer->setStyleSheet(s_strControlStyle);
    QVBoxLayout* pLayout = new QVBoxLayout(pContainer);
    pLayout->setContentsMargins(12, 8, 12, 8);
    pLayout->setSpacing(6);

    // ===== 数据信息分组 =====
    QGroupBox* pGrpData = new QGroupBox(QStringLiteral("数据信息"));
    QFormLayout* pFormData = new QFormLayout(pGrpData);
    pFormData->setSpacing(4);

    m_pLblDatasetName = new QLabel("--");
    m_pLblImageCount  = new QLabel("--");
    m_pLblLabelCount  = new QLabel("--");
    pFormData->addRow(QStringLiteral("数据集:"), m_pLblDatasetName);
    pFormData->addRow(QStringLiteral("图像数:"), m_pLblImageCount);
    pFormData->addRow(QStringLiteral("标签数:"), m_pLblLabelCount);

    pLayout->addWidget(pGrpData);

    // ===== 性能分组 =====
    QGroupBox* pGrpPerf = new QGroupBox(QStringLiteral("性能"));
    QFormLayout* pFormPerf = new QFormLayout(pGrpPerf);
    pFormPerf->setSpacing(4);

    m_pLblAvgLatency = new QLabel("--");
    m_pLblP95Latency = new QLabel("--");
    m_pLblThroughput = new QLabel("--");
    pFormPerf->addRow(QStringLiteral("平均延迟:"), m_pLblAvgLatency);
    pFormPerf->addRow(QStringLiteral("P95 延迟:"), m_pLblP95Latency);
    pFormPerf->addRow(QStringLiteral("吞吐量:"), m_pLblThroughput);

    pLayout->addWidget(pGrpPerf);

    // ===== 日志分组 =====
    QGroupBox* pGrpLog = new QGroupBox(QStringLiteral("日志"));
    QVBoxLayout* pLogLayout = new QVBoxLayout(pGrpLog);

    m_pTxtLog = new QTextEdit();
    m_pTxtLog->setReadOnly(true);
    m_pTxtLog->setStyleSheet(QStringLiteral(
        "QTextEdit { background: #13151a; color: #94a3b8; border: 1px solid #2a2d35;"
        "  border-radius: 4px; font-family: Consolas; font-size: 11px; }"));
    m_pTxtLog->setMaximumHeight(200);
    pLogLayout->addWidget(m_pTxtLog);

    pLayout->addWidget(pGrpLog);

    pLayout->addStretch(1);
    pScrollArea->setWidget(pContainer);
    return pScrollArea;
}

// 20260322 ZJH 创建指标卡片
QFrame* EvaluationPage::createMetricCard(const QString& strTitle,
                                         const QString& strColor,
                                         QLabel*& pLblValue)
{
    QFrame* pCard = new QFrame();
    pCard->setFixedHeight(90);
    pCard->setStyleSheet(QStringLiteral(
        "QFrame { background: #1a1d24; border: 1px solid %1; border-radius: 8px; }").arg(strColor));

    QVBoxLayout* pCardLayout = new QVBoxLayout(pCard);
    pCardLayout->setContentsMargins(12, 8, 12, 8);
    pCardLayout->setSpacing(4);

    // 20260322 ZJH 卡片标题
    QLabel* pLblTitle = new QLabel(strTitle);
    pLblTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: %1; font-size: 11px; font-weight: bold; border: none; }").arg(strColor));
    pCardLayout->addWidget(pLblTitle);

    // 20260322 ZJH 卡片数值
    pLblValue = new QLabel("--");
    pLblValue->setStyleSheet(QStringLiteral(
        "QLabel { color: #f1f5f9; font-size: 24px; font-weight: bold; border: none; }"));
    pCardLayout->addWidget(pLblValue);

    pCardLayout->addStretch(1);
    return pCard;
}

// 20260322 ZJH 生成模拟评估结果
EvaluationResult EvaluationPage::generateMockResult()
{
    EvaluationResult result;

    // 20260322 ZJH 确定类别名称
    QStringList vecClasses;
    if (m_pProject && m_pProject->dataset()) {
        // 20260322 ZJH 从项目数据集获取标签名
        const auto& vecLabels = m_pProject->dataset()->labels();
        for (const auto& label : vecLabels) {
            vecClasses.append(label.strName);
        }
    }

    // 20260322 ZJH 如果没有标签则使用默认类别
    if (vecClasses.isEmpty()) {
        vecClasses = { QStringLiteral("OK"), QStringLiteral("NG"),
                       QStringLiteral("Scratch"), QStringLiteral("Dent") };
    }

    int nN = vecClasses.size();  // 20260322 ZJH 类别数量
    result.vecClassNames = vecClasses;

    // 20260322 ZJH 生成随机混淆矩阵
    QRandomGenerator* pRng = QRandomGenerator::global();
    result.matConfusion.resize(nN);
    int nTotal = 0;
    int nCorrect = 0;
    for (int r = 0; r < nN; ++r) {
        result.matConfusion[r].resize(nN);
        for (int c = 0; c < nN; ++c) {
            if (r == c) {
                // 20260322 ZJH 对角线（正确预测）数值较大
                result.matConfusion[r][c] = pRng->bounded(30, 100);
            } else {
                // 20260322 ZJH 非对角线（误判）数值较小
                result.matConfusion[r][c] = pRng->bounded(0, 15);
            }
            nTotal += result.matConfusion[r][c];
            if (r == c) {
                nCorrect += result.matConfusion[r][c];
            }
        }
    }

    // 20260322 ZJH 计算核心指标
    result.nTotalImages = nTotal;
    result.nCorrect = nCorrect;
    result.dAccuracy = (nTotal > 0) ? static_cast<double>(nCorrect) / nTotal : 0;

    // 20260322 ZJH 计算各类别精确率/召回率的宏平均
    double dSumPrecision = 0;
    double dSumRecall = 0;
    for (int i = 0; i < nN; ++i) {
        // 20260322 ZJH 精确率：TP / (TP + FP) = 对角线值 / 列和
        double dColSum = 0;
        for (int r = 0; r < nN; ++r) {
            dColSum += result.matConfusion[r][i];
        }
        double dPrec = (dColSum > 0) ? result.matConfusion[i][i] / dColSum : 0;
        dSumPrecision += dPrec;

        // 20260322 ZJH 召回率：TP / (TP + FN) = 对角线值 / 行和
        double dRowSum = 0;
        for (int c = 0; c < nN; ++c) {
            dRowSum += result.matConfusion[i][c];
        }
        double dRec = (dRowSum > 0) ? result.matConfusion[i][i] / dRowSum : 0;
        dSumRecall += dRec;
    }

    result.dPrecision = dSumPrecision / nN;
    result.dRecall = dSumRecall / nN;
    result.dF1Score = (result.dPrecision + result.dRecall > 0)
                      ? 2 * result.dPrecision * result.dRecall / (result.dPrecision + result.dRecall)
                      : 0;

    // 20260330 ZJH 从混淆矩阵精确计算 mIoU（替代旧的模拟值）
    // mIoU = mean(IoU_c)，IoU_c = TP_c / (TP_c + FP_c + FN_c) = diag / (rowSum + colSum - diag)
    // 对标 MVTec 分割评估的标准 mIoU 指标
    {
        double dSumIoU = 0.0;
        int nValidClasses = 0;
        for (int i = 0; i < nN; ++i) {
            double dRowSum = 0, dColSum = 0;
            for (int j = 0; j < nN; ++j) { dRowSum += result.matConfusion[i][j]; dColSum += result.matConfusion[j][i]; }
            double dTP = result.matConfusion[i][i];
            double dDenom = dRowSum + dColSum - dTP;  // 20260330 ZJH TP+FP+FN
            if (dDenom > 0) { dSumIoU += dTP / dDenom; ++nValidClasses; }
        }
        result.dMIoU = (nValidClasses > 0) ? dSumIoU / nValidClasses : 0.0;
    }
    // 20260330 ZJH mAP 用 P/R 近似（简化版，精确版需要 P-R 曲线下面积）
    result.dMeanAP = result.dPrecision;
    // 20260330 ZJH AUC 用 (P+R)/2 近似（精确版需 ROC 曲线积分）
    result.dAUC = (result.dPrecision + result.dRecall) / 2.0;

    // 20260322 ZJH 性能模拟（后续接入真实推理计时替代）
    result.dAvgLatencyMs = 5.0 + pRng->bounded(0, 200) / 10.0;
    result.dThroughputFPS = 1000.0 / result.dAvgLatencyMs;
    result.dTotalTimeSeconds = nTotal * result.dAvgLatencyMs / 1000.0;

    return result;
}

// 20260322 ZJH 显示评估结果
void EvaluationPage::displayResult(const EvaluationResult& result)
{
    // 20260322 ZJH 更新指标卡片
    m_pLblCardValue1->setText(QStringLiteral("%1%").arg(result.dAccuracy * 100, 0, 'f', 1));
    m_pLblCardValue2->setText(QStringLiteral("%1%").arg(result.dPrecision * 100, 0, 'f', 1));
    m_pLblCardValue3->setText(QStringLiteral("%1").arg(result.dF1Score, 0, 'f', 3));

    // 20260322 ZJH 填充详细指标表格
    int nN = result.vecClassNames.size();
    m_pTblMetrics->setRowCount(nN);

    for (int i = 0; i < nN; ++i) {
        // 20260322 ZJH 类别名
        m_pTblMetrics->setItem(i, 0, new QTableWidgetItem(result.vecClassNames[i]));

        // 20260322 ZJH 计算各类精确率
        double dColSum = 0;
        for (int r = 0; r < nN; ++r) {
            dColSum += result.matConfusion[r][i];
        }
        double dPrec = (dColSum > 0) ? static_cast<double>(result.matConfusion[i][i]) / dColSum : 0;

        // 20260322 ZJH 计算各类召回率
        double dRowSum = 0;
        for (int c = 0; c < nN; ++c) {
            dRowSum += result.matConfusion[i][c];
        }
        double dRec = (dRowSum > 0) ? static_cast<double>(result.matConfusion[i][i]) / dRowSum : 0;

        // 20260322 ZJH 计算 F1
        double dF1 = (dPrec + dRec > 0) ? 2 * dPrec * dRec / (dPrec + dRec) : 0;

        m_pTblMetrics->setItem(i, 1, new QTableWidgetItem(QString::number(dPrec, 'f', 3)));
        m_pTblMetrics->setItem(i, 2, new QTableWidgetItem(QString::number(dRec, 'f', 3)));
        m_pTblMetrics->setItem(i, 3, new QTableWidgetItem(QString::number(dF1, 'f', 3)));
        m_pTblMetrics->setItem(i, 4, new QTableWidgetItem(QString::number(static_cast<int>(dRowSum))));

        // 20260322 ZJH 居中对齐所有单元格
        for (int c = 0; c < 5; ++c) {
            m_pTblMetrics->item(i, c)->setTextAlignment(Qt::AlignCenter);
        }
    }

    // 20260322 ZJH 更新混淆矩阵热力图
    m_pHeatmap->setData(result.matConfusion, result.vecClassNames);

    // 20260322 ZJH 更新性能标签
    m_pLblAvgLatency->setText(QStringLiteral("%1 ms").arg(result.dAvgLatencyMs, 0, 'f', 1));
    m_pLblP95Latency->setText(QStringLiteral("%1 ms").arg(result.dAvgLatencyMs * 1.5, 0, 'f', 1));
    m_pLblThroughput->setText(QStringLiteral("%1 FPS").arg(result.dThroughputFPS, 0, 'f', 1));

    // 20260322 ZJH ===== 渲染推理结果样本 =====
    QRandomGenerator* pRng = QRandomGenerator::global();
    for (int i = 0; i < m_vecInferThumbs.size(); ++i) {
        QLabel* pThumb = m_vecInferThumbs[i];
        // 20260322 ZJH 生成模拟推理渲染图（灰度背景 + 预测标签 + OK/NG 边框）
        QPixmap pix(120, 90);
        QPainter painter(&pix);
        // 20260322 ZJH 随机灰度背景
        int nGray = 40 + pRng->bounded(60);
        painter.fillRect(pix.rect(), QColor(nGray, nGray, nGray));

        bool bCorrect = (pRng->bounded(100) < static_cast<int>(result.dAccuracy * 100));
        int nPredClass = pRng->bounded(nN);
        QString strPred = (nN > 0) ? result.vecClassNames[nPredClass] : "?";

        // 20260322 ZJH 边框颜色：正确=绿，错误=红
        QColor colBorder = bCorrect ? QColor("#22c55e") : QColor("#ef4444");
        painter.setPen(QPen(colBorder, 3));
        painter.drawRect(1, 1, 117, 87);

        // 20260322 ZJH 预测标签
        painter.setPen(Qt::white);
        painter.setFont(QFont("sans-serif", 9, QFont::Bold));
        painter.drawText(QRect(4, 4, 112, 20), Qt::AlignLeft, strPred);

        // 20260322 ZJH OK/NG 标记
        QString strTag = bCorrect ? "OK" : "NG";
        painter.setPen(colBorder);
        painter.setFont(QFont("sans-serif", 11, QFont::Bold));
        painter.drawText(QRect(4, 65, 112, 22), Qt::AlignRight, strTag);

        painter.end();
        pThumb->setPixmap(pix);
        pThumb->setStyleSheet(QStringLiteral(
            "QLabel { background: #1a1d24; border: 2px solid %1; border-radius: 4px; }")
            .arg(colBorder.name()));
    }

    // 20260322 ZJH ===== 渲染缺陷概率热力图 =====
    // 20260324 ZJH 生成模拟热力图数据并传递给 GradCAMOverlay 控件
    int nMapW = 200, nMapH = 150;

    // 20260324 ZJH 生成模拟灰度原始图像作为 GradCAMOverlay 的底图
    QImage imgOriginal(nMapW, nMapH, QImage::Format_RGB32);
    imgOriginal.fill(QColor(128, 128, 128));  // 20260324 ZJH 中灰色背景

    // 20260324 ZJH 生成归一化热力图数据 [0, 1]
    QVector<double> vecHeatmap(nMapW * nMapH, 0.0);
    for (int y = 0; y < nMapH; ++y) {
        for (int x = 0; x < nMapW; ++x) {
            // 20260322 ZJH 模拟缺陷概率：中心偏右下有高概率区域
            double dx = (x - nMapW * 0.6) / (nMapW * 0.15);
            double dy = (y - nMapH * 0.55) / (nMapH * 0.2);
            double dProb = std::exp(-(dx * dx + dy * dy) * 0.5);
            // 20260322 ZJH 加噪声
            dProb += (pRng->bounded(100) - 50) * 0.002;
            dProb = qBound(0.0, dProb, 1.0);
            vecHeatmap[y * nMapW + x] = dProb;  // 20260324 ZJH 存储归一化概率值
        }
    }

    // 20260324 ZJH 将数据传递给 GradCAMOverlay 控件（自动根据当前模式渲染）
    if (m_pGradCAMOverlay) {
        m_pGradCAMOverlay->setOriginalImage(imgOriginal);
        m_pGradCAMOverlay->setHeatmap(vecHeatmap, nMapW, nMapH);
    }
}

// 20260322 ZJH 刷新前置检查
void EvaluationPage::refreshPreChecks()
{
    // 20260322 ZJH 检查状态样式
    QString strPass = QStringLiteral("QLabel { color: #10b981; font-size: 12px; }");  // 20260322 ZJH 绿色通过
    QString strFail = QStringLiteral("QLabel { color: #64748b; font-size: 12px; }");  // 20260322 ZJH 灰色未通过

    bool bHasProject = (m_pProject != nullptr);
    bool bTrained = bHasProject && (m_pProject->state() >= om::ProjectState::ModelTrained);
    bool bHasTest = bHasProject && m_pProject->dataset() &&
                    (m_pProject->dataset()->countBySplit(om::SplitType::Test) > 0);
    bool bLabeled = bHasProject && m_pProject->dataset() &&
                    (m_pProject->dataset()->labeledCount() > 0);

    // 20260322 ZJH 模型已训练
    m_pLblCheckTrained->setText(bTrained ? QStringLiteral("\xe2\x9c\x93 模型已训练") : QStringLiteral("\xe2\x9c\x97 模型已训练"));
    m_pLblCheckTrained->setStyleSheet(bTrained ? strPass : strFail);

    // 20260322 ZJH 测试集存在
    m_pLblCheckTestSet->setText(bHasTest ? QStringLiteral("\xe2\x9c\x93 测试集存在") : QStringLiteral("\xe2\x9c\x97 测试集存在"));
    m_pLblCheckTestSet->setStyleSheet(bHasTest ? strPass : strFail);

    // 20260322 ZJH 测试集已标注
    m_pLblCheckLabeled->setText(bLabeled ? QStringLiteral("\xe2\x9c\x93 测试集已标注") : QStringLiteral("\xe2\x9c\x97 测试集已标注"));
    m_pLblCheckLabeled->setStyleSheet(bLabeled ? strPass : strFail);

    // 20260322 ZJH 模型文件存在（模拟：训练完成即视为存在）
    m_pLblCheckModel->setText(bTrained ? QStringLiteral("\xe2\x9c\x93 模型文件存在") : QStringLiteral("\xe2\x9c\x97 模型文件存在"));
    m_pLblCheckModel->setStyleSheet(bTrained ? strPass : strFail);

    // 20260322 ZJH 推理引擎就绪（模拟：始终就绪）
    m_pLblCheckEngine->setText(QStringLiteral("\xe2\x9c\x93 推理引擎就绪"));
    m_pLblCheckEngine->setStyleSheet(strPass);
}

// 20260322 ZJH 刷新数据信息
void EvaluationPage::refreshDataInfo()
{
    if (!m_pProject || !m_pProject->dataset()) {
        m_pLblDatasetName->setText("--");
        m_pLblImageCount->setText("--");
        m_pLblLabelCount->setText("--");
        return;  // 20260322 ZJH 无项目
    }

    m_pLblDatasetName->setText(m_pProject->name());
    m_pLblImageCount->setText(QString::number(m_pProject->dataset()->imageCount()));
    m_pLblLabelCount->setText(QString::number(m_pProject->dataset()->labels().size()));
}

// 20260322 ZJH 追加日志消息
void EvaluationPage::appendLog(const QString& strMsg)
{
    if (m_pTxtLog) {
        m_pTxtLog->append(strMsg);  // 20260322 ZJH 追加到日志文本框
    }
}

// 20260402 ZJH [OPT-3.9] 清除模型对比记录
void EvaluationPage::onClearComparison()
{
    if (m_pComparisonTable) {
        m_pComparisonTable->setRowCount(0);  // 20260402 ZJH 清空所有行
    }
    appendLog(QStringLiteral("[%1] 模型对比记录已清除")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss")));
}

// 20260402 ZJH [OPT-3.9] 追加一行模型对比记录
// 参数: strModelName - 模型名称（如 "ResNet18"）
//       dAccuracy - 精度（0~1）
//       dLatencyMs - 推理延迟（毫秒）
//       dModelSizeMB - 模型大小（MB）
//       dTrainTimeSec - 训练总时间（秒）
void EvaluationPage::appendComparisonEntry(const QString& strModelName, double dAccuracy,
                                            double dLatencyMs, double dModelSizeMB, double dTrainTimeSec)
{
    if (!m_pComparisonTable) return;  // 20260402 ZJH 安全检查

    int nRow = m_pComparisonTable->rowCount();  // 20260402 ZJH 当前行数
    m_pComparisonTable->insertRow(nRow);         // 20260402 ZJH 插入新行

    // 20260402 ZJH 填充各列数据
    m_pComparisonTable->setItem(nRow, 0, new QTableWidgetItem(strModelName));
    m_pComparisonTable->setItem(nRow, 1, new QTableWidgetItem(
        QString::number(dAccuracy * 100.0, 'f', 2)));  // 20260402 ZJH 精度百分比
    m_pComparisonTable->setItem(nRow, 2, new QTableWidgetItem(
        QString::number(dLatencyMs, 'f', 2)));           // 20260402 ZJH 推理延迟 ms
    m_pComparisonTable->setItem(nRow, 3, new QTableWidgetItem(
        QString::number(dModelSizeMB, 'f', 1)));         // 20260402 ZJH 模型大小 MB
    m_pComparisonTable->setItem(nRow, 4, new QTableWidgetItem(
        QString::number(dTrainTimeSec, 'f', 1)));        // 20260402 ZJH 训练时间 s

    // 20260402 ZJH 滚动到最新行
    m_pComparisonTable->scrollToBottom();
}
