// 20260322 ZJH EvaluationPage — 模型评估页面
// BasePage 子类，三栏布局：
//   左面板(280px): 评估配置 + 操作按钮 + 前置检查
//   中央面板: 进度条 + 指标卡片 + 详细指标表 + 混淆矩阵热力图
//   右面板(220px): 数据信息 + 性能 + 日志

#pragma once

#include "ui/pages/BasePage.h"                     // 20260322 ZJH 页面基类
#include "core/evaluation/EvaluationResult.h"      // 20260322 ZJH 评估结果结构体

#include <QComboBox>         // 20260322 ZJH 下拉选择框
#include <QDoubleSpinBox>    // 20260322 ZJH 浮点数微调框
#include <QPushButton>       // 20260322 ZJH 按钮
#include <QLabel>            // 20260322 ZJH 文本标签
#include <QProgressBar>      // 20260322 ZJH 进度条
#include <QTextEdit>         // 20260322 ZJH 日志显示文本框
#include <QTableWidget>      // 20260322 ZJH 详细指标表格
#include <QFrame>            // 20260322 ZJH 指标卡片容器
#include <QTimer>            // 20260322 ZJH 模拟评估计时器
#include <QSlider>           // 20260324 ZJH 滑块（二值化阈值调节）

// 20260322 ZJH 前向声明
class ConfusionMatrixHeatmap;
class GradCAMOverlay;  // 20260324 ZJH Grad-CAM 热力图/二值化叠加控件

// 20260322 ZJH 模型评估页面
// 提供评估配置、运行评估、结果可视化（混淆矩阵/指标卡片/详细表格）
class EvaluationPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，创建三栏布局并初始化全部控件
    // 参数: pParent - 父控件指针
    explicit EvaluationPage(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~EvaluationPage() override = default;

    // ===== BasePage 生命周期回调 =====

    // 20260322 ZJH 页面切换到前台时调用
    void onEnter() override;

    // 20260322 ZJH 页面离开前台时调用
    void onLeave() override;

    // 20260324 ZJH 项目加载后调用，刷新前置检查和数据信息（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时调用，清空显示（Template Method 扩展点）
    void onProjectClosedImpl() override;

private slots:
    // 20260322 ZJH 运行评估按钮点击
    void onRunEvaluation();

    // 20260322 ZJH 清除结果按钮点击
    void onClearResults();

    // 20260322 ZJH 导出 CSV 按钮点击
    void onExportCsv();

    // 20260322 ZJH 导出 HTML 报告按钮点击
    void onExportHtmlReport();

    // 20260330 ZJH 导出 JSON 按钮点击
    void onExportJson();

    // 20260322 ZJH 归一化模式切换
    // 参数: nMode - 0=计数, 1=行归一化, 2=列归一化
    void onNormModeChanged(int nMode);

    // 20260322 ZJH 模拟评估进度更新（定时器触发）
    void onSimulationTick();

private:
    // ===== UI 创建辅助方法 =====

    // 20260322 ZJH 创建左面板（评估配置 + 操作 + 前置检查）
    QWidget* createLeftPanel();

    // 20260322 ZJH 创建中央面板（进度 + 指标卡片 + 详细表 + 混淆矩阵）
    QWidget* createCenterPanel();

    // 20260322 ZJH 创建右面板（数据信息 + 性能 + 日志）
    QWidget* createRightPanel();

    // 20260322 ZJH 创建单个指标卡片
    // 参数: strTitle - 卡片标题; strColor - 强调色（十六进制）
    // 返回: 包含标题和数值标签的 QFrame 指针
    QFrame* createMetricCard(const QString& strTitle, const QString& strColor,
                             QLabel*& pLblValue);

    // 20260322 ZJH 生成模拟评估结果（随机混淆矩阵 + 指标）
    // 返回: 模拟的 EvaluationResult
    EvaluationResult generateMockResult();

    // 20260322 ZJH 用评估结果填充所有 UI 控件
    // 参数: result - 评估结果数据
    void displayResult(const EvaluationResult& result);

    // 20260322 ZJH 刷新前置检查标签
    void refreshPreChecks();

    // 20260322 ZJH 刷新数据信息标签
    void refreshDataInfo();

    // 20260322 ZJH 追加日志消息
    // 参数: strMsg - 日志文本
    void appendLog(const QString& strMsg);

    // ===== 左面板控件 =====

    QComboBox*      m_pCboDataScope;         // 20260322 ZJH 数据范围下拉框（训练集/验证集/测试集/全部）
    QDoubleSpinBox* m_pSpnConfThreshold;     // 20260322 ZJH 置信度阈值微调框
    QDoubleSpinBox* m_pSpnIouThreshold;      // 20260322 ZJH IoU 阈值微调框

    QPushButton* m_pBtnRunEval;       // 20260322 ZJH 运行评估按钮
    QPushButton* m_pBtnClearResults;  // 20260322 ZJH 清除结果按钮
    QPushButton* m_pBtnExportCsv;     // 20260322 ZJH 导出 CSV 按钮
    QPushButton* m_pBtnExportJson;    // 20260330 ZJH 导出 JSON 按钮
    QPushButton* m_pBtnExportHtml;    // 20260322 ZJH 导出 HTML 报告按钮
    QComboBox*   m_pCboReportTemplate;  // 20260330 ZJH 报告模板下拉框（标准/详细/简洁）

    // 20260322 ZJH 前置检查标签（5 项）
    QLabel* m_pLblCheckTrained;   // 20260322 ZJH 模型已训练
    QLabel* m_pLblCheckTestSet;   // 20260322 ZJH 测试集存在
    QLabel* m_pLblCheckLabeled;   // 20260322 ZJH 测试集已标注
    QLabel* m_pLblCheckModel;     // 20260322 ZJH 模型文件存在
    QLabel* m_pLblCheckEngine;    // 20260322 ZJH 推理引擎就绪

    // ===== 中央面板控件 =====

    QProgressBar* m_pProgressBar;  // 20260322 ZJH 进度条
    QLabel*       m_pLblStatus;    // 20260322 ZJH 状态文字标签

    // 20260322 ZJH 三个指标卡片
    QFrame* m_pCardAccuracy;    // 20260322 ZJH 卡片1: 准确率/mAP
    QFrame* m_pCardPrecision;   // 20260322 ZJH 卡片2: 精确率/mIoU
    QFrame* m_pCardRecall;      // 20260322 ZJH 卡片3: 召回率/F1
    QLabel* m_pLblCardValue1;   // 20260322 ZJH 卡片1 数值标签
    QLabel* m_pLblCardValue2;   // 20260322 ZJH 卡片2 数值标签
    QLabel* m_pLblCardValue3;   // 20260322 ZJH 卡片3 数值标签

    QTableWidget*          m_pTblMetrics;    // 20260322 ZJH 详细指标表格
    ConfusionMatrixHeatmap* m_pHeatmap;      // 20260322 ZJH 混淆矩阵热力图

    // 20260322 ZJH 推理结果渲染图（样本预测 + 缺陷概率热力图）
    QLabel* m_pLblInferTitle;         // 20260406 ZJH 推理结果渲染区域标题标签
    QWidget* m_pInferGrid;            // 20260406 ZJH 推理样本网格容器（展示预测结果缩略图）
    QVector<QLabel*> m_vecInferThumbs;  // 20260406 ZJH 推理缩略图标签列表（每个标签显示一张推理结果）
    QLabel* m_pLblDefectMap;          // 20260406 ZJH 缺陷概率热力图标签（异常检测时展示逐像素缺陷概率）

    // 20260324 ZJH GradCAM 显示模式控件
    GradCAMOverlay* m_pGradCAMOverlay;    // 20260324 ZJH Grad-CAM 叠加控件
    QComboBox* m_pCboGradCAMMode;         // 20260324 ZJH 显示模式下拉框（热力图/二值化）
    QSlider* m_pSliderThreshold;          // 20260324 ZJH 二值化阈值滑块
    QLabel* m_pLblThresholdValue;         // 20260324 ZJH 阈值数值标签

    // 20260322 ZJH 归一化模式切换按钮组
    QPushButton* m_pBtnNormCount;   // 20260322 ZJH 计数按钮
    QPushButton* m_pBtnNormRow;     // 20260322 ZJH 行归一化按钮
    QPushButton* m_pBtnNormCol;     // 20260322 ZJH 列归一化按钮

    // ===== 右面板控件 =====

    // 20260322 ZJH 数据信息
    QLabel* m_pLblDatasetName;  // 20260322 ZJH 数据集名称
    QLabel* m_pLblImageCount;   // 20260322 ZJH 图像数
    QLabel* m_pLblLabelCount;   // 20260322 ZJH 标签数

    // 20260322 ZJH 性能
    QLabel* m_pLblAvgLatency;   // 20260322 ZJH 平均延迟
    QLabel* m_pLblP95Latency;   // 20260322 ZJH P95 延迟
    QLabel* m_pLblThroughput;   // 20260322 ZJH 吞吐量

    QTextEdit* m_pTxtLog;  // 20260322 ZJH 日志文本框

    // 20260402 ZJH [OPT-3.9] 模型对比面板
    QTableWidget* m_pComparisonTable;  // 20260402 ZJH 模型对比表格（模型名称|精度|推理延迟|模型大小|训练时间）
    QPushButton*  m_pBtnClearComparison;  // 20260402 ZJH 清除对比记录按钮

private slots:
    // 20260402 ZJH 清除模型对比记录
    void onClearComparison();

private:
    // 20260402 ZJH 追加一行模型对比记录（评估完成后自动调用）
    void appendComparisonEntry(const QString& strModelName, double dAccuracy,
                               double dLatencyMs, double dModelSizeMB, double dTrainTimeSec);

    // ===== 评估状态 =====

    EvaluationResult m_lastResult;     // 20260322 ZJH 最近一次评估结果
    QTimer*          m_pSimTimer;      // 20260322 ZJH 模拟评估计时器
    int              m_nSimProgress;   // 20260322 ZJH 模拟评估当前进度
    int              m_nSimTotal;      // 20260322 ZJH 模拟评估总数
};
