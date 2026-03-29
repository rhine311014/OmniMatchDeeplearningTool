// 20260322 ZJH ExportPage 实现
// 三栏布局：左面板导出配置、中央结果展示、右面板模型信息日志
// 模拟导出流程：加载模型 → 优化 → 转换 → 保存，进度条实时更新

#include "ui/pages/export/ExportPage.h"        // 20260322 ZJH ExportPage 类声明
#include "core/project/Project.h"               // 20260322 ZJH 项目数据
#include "core/data/ImageDataset.h"             // 20260322 ZJH 数据集
#include "core/DLTypes.h"                       // 20260322 ZJH 类型定义
#include "core/training/TrainingConfig.h"       // 20260324 ZJH 训练配置（refreshModelInfo 需要读取架构信息）
#include "app/Application.h"                    // 20260322 ZJH 全局事件总线

#include <QVBoxLayout>      // 20260322 ZJH 垂直布局
#include <QHBoxLayout>      // 20260322 ZJH 水平布局
#include <QFormLayout>      // 20260322 ZJH 表单布局
#include <QScrollArea>      // 20260322 ZJH 滚动区域
#include <QGroupBox>        // 20260322 ZJH 分组框
#include <QHeaderView>      // 20260322 ZJH 表格头视图
#include <QMessageBox>      // 20260322 ZJH 提示对话框
#include <QFileDialog>      // 20260322 ZJH 文件对话框
#include <QFileInfo>        // 20260324 ZJH QFileInfo 获取模型文件大小
#include <QDateTime>        // 20260322 ZJH 时间戳
#include <QDesktopServices> // 20260322 ZJH 打开文件夹
#include <QUrl>             // 20260322 ZJH URL 构造

// 20260322 ZJH 通用暗色控件样式（与其他页面一致）
static const QString s_strControlStyle = QStringLiteral(
    "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {"
    "  background-color: #1a1d24;"
    "  color: #e2e8f0;"
    "  border: 1px solid #333842;"
    "  border-radius: 4px;"
    "  padding: 4px 8px;"
    "  min-height: 22px;"
    "}"
    "QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {"
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
    "QCheckBox {"
    "  color: #e2e8f0;"
    "  spacing: 6px;"
    "}"
    "QCheckBox::indicator {"
    "  width: 16px;"
    "  height: 16px;"
    "  border: 1px solid #333842;"
    "  border-radius: 3px;"
    "  background: #1a1d24;"
    "}"
    "QCheckBox::indicator:checked {"
    "  background: #2563eb;"
    "  border-color: #2563eb;"
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
ExportPage::ExportPage(QWidget* pParent)
    : BasePage(pParent)
    , m_pCboFormat(nullptr)
    , m_pEdtModelName(nullptr)
    , m_pCboPrecision(nullptr)
    , m_pChkDynBatch(nullptr)
    , m_pSpnBatchMin(nullptr)
    , m_pSpnBatchMax(nullptr)
    , m_pEdtOutputDir(nullptr)
    , m_pBtnBrowse(nullptr)
    , m_pBtnStartExport(nullptr)
    , m_pBtnOpenDir(nullptr)
    , m_pLblCheckTrained(nullptr)
    , m_pLblCheckModel(nullptr)
    , m_pProgressBar(nullptr)
    , m_pLblStatus(nullptr)
    , m_pLblPhase(nullptr)
    , m_pCardResult(nullptr)
    , m_pLblResultPath(nullptr)
    , m_pLblResultSize(nullptr)
    , m_pLblResultTime(nullptr)
    , m_pTblCompat(nullptr)
    , m_pTblHistory(nullptr)
    , m_pLblArch(nullptr)
    , m_pLblParams(nullptr)
    , m_pLblFLOPs(nullptr)
    , m_pLblMemory(nullptr)
    , m_pTxtLog(nullptr)
    , m_pSimTimer(nullptr)
    , m_nSimProgress(0)
    , m_nSimTotal(100)
{
    // 20260322 ZJH 创建模拟导出定时器
    m_pSimTimer = new QTimer(this);
    m_pSimTimer->setInterval(40);  // 20260322 ZJH 每 40ms 触发一次进度更新
    connect(m_pSimTimer, &QTimer::timeout, this, &ExportPage::onExportSimTick);

    // 20260322 ZJH 创建三栏布局
    QWidget* pLeft   = createLeftPanel();    // 20260322 ZJH 左面板
    QWidget* pCenter = createCenterPanel();  // 20260322 ZJH 中央面板
    QWidget* pRight  = createRightPanel();   // 20260322 ZJH 右面板

    // 20260322 ZJH 设置面板宽度
    setLeftPanelWidth(280);   // 20260322 ZJH 左面板 280px
    setRightPanelWidth(220);  // 20260322 ZJH 右面板 220px

    // 20260322 ZJH 调用基类方法组装三栏布局
    setupThreeColumnLayout(pLeft, pCenter, pRight);
}

// ===== BasePage 生命周期回调 =====

// 20260322 ZJH 页面进入前台
void ExportPage::onEnter()
{
    refreshPreChecks();  // 20260322 ZJH 刷新前置检查
    refreshModelInfo();  // 20260322 ZJH 刷新模型信息
}

// 20260322 ZJH 页面离开前台
void ExportPage::onLeave()
{
    // 20260322 ZJH 当前无需处理
}

// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void ExportPage::onProjectLoadedImpl()
{
    refreshPreChecks();  // 20260322 ZJH 刷新前置检查
    refreshModelInfo();  // 20260322 ZJH 刷新模型信息

    // 20260322 ZJH 设置默认模型名称
    if (m_pEdtModelName && m_pProject) {
        m_pEdtModelName->setText(m_pProject->name() + "_model");
    }

    // 20260322 ZJH 设置默认输出目录
    if (m_pEdtOutputDir && m_pProject) {
        m_pEdtOutputDir->setText(m_pProject->path() + "/export");
    }

    // 20260324 ZJH 使用基类已赋值的 m_pProject 获取项目名
    if (m_pProject) {
        appendLog(QStringLiteral("[%1] 项目已加载: %2")
                  .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                  .arg(m_pProject->name()));
    }
}

// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void ExportPage::onProjectClosedImpl()
{
    refreshPreChecks();
    refreshModelInfo();
    m_pEdtModelName->clear();
    m_pEdtOutputDir->clear();
}

// ===== 槽函数 =====

// 20260322 ZJH 开始导出
void ExportPage::onStartExport()
{
    // 20260322 ZJH 获取导出格式
    QString strFormat = m_pCboFormat->currentText();

    appendLog(QStringLiteral("[%1] 开始导出 %2 格式...")
              .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
              .arg(strFormat));

    // 20260322 ZJH 设置进度条
    m_nSimProgress = 0;
    m_nSimTotal = 100;
    m_pProgressBar->setRange(0, m_nSimTotal);
    m_pProgressBar->setValue(0);
    m_pProgressBar->setVisible(true);
    m_pLblStatus->setText(QStringLiteral("正在导出..."));
    m_pLblPhase->setText(QStringLiteral("阶段: 加载模型"));

    // 20260322 ZJH 隐藏结果卡片
    m_pCardResult->setVisible(false);

    // 20260322 ZJH 禁用导出按钮
    m_pBtnStartExport->setEnabled(false);

    // 20260322 ZJH 启动模拟定时器
    m_pSimTimer->start();
}

// 20260322 ZJH 打开输出目录
void ExportPage::onOpenOutputDir()
{
    QString strDir = m_pEdtOutputDir->text();
    if (strDir.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("未设置输出目录。"));
        return;
    }
    // 20260322 ZJH 使用系统默认文件管理器打开目录
    QDesktopServices::openUrl(QUrl::fromLocalFile(strDir));
}

// 20260322 ZJH 浏览输出目录
void ExportPage::onBrowseOutputDir()
{
    QString strDir = QFileDialog::getExistingDirectory(this,
        QStringLiteral("选择输出目录"),
        m_pEdtOutputDir->text());
    if (!strDir.isEmpty()) {
        m_pEdtOutputDir->setText(strDir);
    }
}

// 20260322 ZJH 动态批量复选框切换
void ExportPage::onDynamicBatchToggled(bool bChecked)
{
    m_pSpnBatchMin->setEnabled(bChecked);  // 20260322 ZJH 启用/禁用最小批量
    m_pSpnBatchMax->setEnabled(bChecked);  // 20260322 ZJH 启用/禁用最大批量
}

// 20260322 ZJH 模拟导出进度
void ExportPage::onExportSimTick()
{
    m_nSimProgress++;
    m_pProgressBar->setValue(m_nSimProgress);

    // 20260322 ZJH 根据进度更新阶段显示
    if (m_nSimProgress < 20) {
        m_pLblPhase->setText(QStringLiteral("阶段: 加载模型"));
    } else if (m_nSimProgress < 50) {
        m_pLblPhase->setText(QStringLiteral("阶段: 模型优化"));
    } else if (m_nSimProgress < 80) {
        m_pLblPhase->setText(QStringLiteral("阶段: 格式转换"));
    } else {
        m_pLblPhase->setText(QStringLiteral("阶段: 保存文件"));
    }

    m_pLblStatus->setText(QStringLiteral("导出进度: %1%").arg(m_nSimProgress));

    // 20260322 ZJH 导出完成
    if (m_nSimProgress >= m_nSimTotal) {
        m_pSimTimer->stop();

        // 20260322 ZJH 获取导出信息
        QString strFormat = m_pCboFormat->currentText();
        QString strModelName = m_pEdtModelName->text();
        if (strModelName.isEmpty()) {
            strModelName = "model";
        }

        // 20260322 ZJH 构造模拟文件路径和大小
        QString strExt;
        if (strFormat == "ONNX") strExt = ".onnx";
        else if (strFormat == "TensorRT") strExt = ".engine";
        else if (strFormat == "OpenVINO") strExt = ".xml";
        else strExt = ".dfm";

        QString strOutputPath = m_pEdtOutputDir->text() + "/" + strModelName + strExt;
        QString strSize = QStringLiteral("42.5 MB");
        double dTime = 4.0 + m_nSimTotal * 0.04;  // 20260322 ZJH 模拟耗时

        // 20260322 ZJH 显示结果卡片
        m_pLblResultPath->setText(strOutputPath);
        m_pLblResultSize->setText(strSize);
        m_pLblResultTime->setText(QStringLiteral("%1 秒").arg(dTime, 0, 'f', 1));
        m_pCardResult->setVisible(true);

        // 20260322 ZJH 添加历史记录
        addHistoryEntry(strFormat, strOutputPath, strSize, dTime);

        // 20260322 ZJH 更新状态
        m_pLblStatus->setText(QStringLiteral("导出完成"));
        m_pLblPhase->setText(QStringLiteral("阶段: 完成"));
        m_pBtnStartExport->setEnabled(true);

        appendLog(QStringLiteral("[%1] 导出完成: %2 → %3")
                  .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                  .arg(strFormat)
                  .arg(strOutputPath));
    }
}

// ===== UI 创建 =====

// 20260322 ZJH 创建左面板
QWidget* ExportPage::createLeftPanel()
{
    QScrollArea* pScrollArea = new QScrollArea();
    pScrollArea->setWidgetResizable(true);
    pScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    pScrollArea->setStyleSheet(QStringLiteral(
        "QScrollArea { background: #1e2230; border: none; border-right: 1px solid #2a2d35; }"));

    QWidget* pContainer = new QWidget();
    pContainer->setStyleSheet(s_strControlStyle);
    QVBoxLayout* pLayout = new QVBoxLayout(pContainer);
    pLayout->setContentsMargins(12, 8, 12, 8);
    pLayout->setSpacing(6);

    // ===== 导出配置分组 =====
    QGroupBox* pGrpConfig = new QGroupBox(QStringLiteral("导出配置"));
    QFormLayout* pFormConfig = new QFormLayout(pGrpConfig);
    pFormConfig->setSpacing(6);

    // 20260322 ZJH 导出格式
    m_pCboFormat = new QComboBox();
    m_pCboFormat->addItems({
        QStringLiteral("ONNX"),
        QStringLiteral("TensorRT"),
        QStringLiteral("OpenVINO"),
        QStringLiteral("自研DFM")
    });
    pFormConfig->addRow(QStringLiteral("导出格式:"), m_pCboFormat);

    // 20260322 ZJH 模型名称
    m_pEdtModelName = new QLineEdit();
    m_pEdtModelName->setPlaceholderText(QStringLiteral("输入模型名称"));
    pFormConfig->addRow(QStringLiteral("模型名称:"), m_pEdtModelName);

    // 20260322 ZJH 精度
    m_pCboPrecision = new QComboBox();
    m_pCboPrecision->addItems({
        QStringLiteral("FP32"),
        QStringLiteral("FP16"),
        QStringLiteral("INT8")
    });
    pFormConfig->addRow(QStringLiteral("精度:"), m_pCboPrecision);

    // 20260322 ZJH 动态批量
    QHBoxLayout* pDynBatchLayout = new QHBoxLayout();
    m_pChkDynBatch = new QCheckBox(QStringLiteral("动态批量"));
    connect(m_pChkDynBatch, &QCheckBox::toggled, this, &ExportPage::onDynamicBatchToggled);
    pDynBatchLayout->addWidget(m_pChkDynBatch);

    m_pSpnBatchMin = new QSpinBox();
    m_pSpnBatchMin->setRange(1, 64);
    m_pSpnBatchMin->setValue(1);
    m_pSpnBatchMin->setPrefix(QStringLiteral("min: "));
    m_pSpnBatchMin->setEnabled(false);  // 20260322 ZJH 默认禁用
    pDynBatchLayout->addWidget(m_pSpnBatchMin);

    m_pSpnBatchMax = new QSpinBox();
    m_pSpnBatchMax->setRange(1, 128);
    m_pSpnBatchMax->setValue(16);
    m_pSpnBatchMax->setPrefix(QStringLiteral("max: "));
    m_pSpnBatchMax->setEnabled(false);  // 20260322 ZJH 默认禁用
    pDynBatchLayout->addWidget(m_pSpnBatchMax);

    pFormConfig->addRow(pDynBatchLayout);

    pLayout->addWidget(pGrpConfig);

    // ===== 输出目录 =====
    QGroupBox* pGrpOutput = new QGroupBox(QStringLiteral("输出目录"));
    QHBoxLayout* pOutputLayout = new QHBoxLayout(pGrpOutput);

    m_pEdtOutputDir = new QLineEdit();
    m_pEdtOutputDir->setPlaceholderText(QStringLiteral("选择输出目录"));
    pOutputLayout->addWidget(m_pEdtOutputDir);

    m_pBtnBrowse = new QPushButton(QStringLiteral("..."));
    m_pBtnBrowse->setFixedWidth(30);
    m_pBtnBrowse->setStyleSheet(QStringLiteral(
        "QPushButton { background: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 4px; padding: 4px; }"
        "QPushButton:hover { background: #475569; }"));
    connect(m_pBtnBrowse, &QPushButton::clicked, this, &ExportPage::onBrowseOutputDir);
    pOutputLayout->addWidget(m_pBtnBrowse);

    pLayout->addWidget(pGrpOutput);

    // ===== 操作分组 =====
    QGroupBox* pGrpAction = new QGroupBox(QStringLiteral("操作"));
    QVBoxLayout* pActionLayout = new QVBoxLayout(pGrpAction);
    pActionLayout->setSpacing(6);

    // 20260322 ZJH 开始导出按钮（蓝色主按钮）
    m_pBtnStartExport = new QPushButton(QStringLiteral("开始导出"));
    m_pBtnStartExport->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #2563eb; color: white; border: none; border-radius: 6px;"
        "  padding: 8px 16px; font-weight: bold; font-size: 13px; }"
        "QPushButton:hover { background-color: #1d4ed8; }"
        "QPushButton:disabled { background-color: #475569; }"));
    connect(m_pBtnStartExport, &QPushButton::clicked, this, &ExportPage::onStartExport);
    pActionLayout->addWidget(m_pBtnStartExport);

    // 20260322 ZJH 打开输出目录按钮
    m_pBtnOpenDir = new QPushButton(QStringLiteral("打开输出目录"));
    m_pBtnOpenDir->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: #334155; color: #e2e8f0; border: 1px solid #475569;"
        "  border-radius: 6px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #475569; }"));
    connect(m_pBtnOpenDir, &QPushButton::clicked, this, &ExportPage::onOpenOutputDir);
    pActionLayout->addWidget(m_pBtnOpenDir);

    pLayout->addWidget(pGrpAction);

    // ===== 前置检查分组 =====
    QGroupBox* pGrpCheck = new QGroupBox(QStringLiteral("前置检查"));
    QVBoxLayout* pCheckLayout = new QVBoxLayout(pGrpCheck);
    pCheckLayout->setSpacing(4);

    m_pLblCheckTrained = new QLabel(QStringLiteral("\xe2\x9c\x97 模型已训练"));
    m_pLblCheckTrained->setStyleSheet(QStringLiteral("QLabel { color: #64748b; font-size: 12px; }"));
    pCheckLayout->addWidget(m_pLblCheckTrained);

    m_pLblCheckModel = new QLabel(QStringLiteral("\xe2\x9c\x97 模型文件存在"));
    m_pLblCheckModel->setStyleSheet(QStringLiteral("QLabel { color: #64748b; font-size: 12px; }"));
    pCheckLayout->addWidget(m_pLblCheckModel);

    pLayout->addWidget(pGrpCheck);

    pLayout->addStretch(1);
    pScrollArea->setWidget(pContainer);
    return pScrollArea;
}

// 20260322 ZJH 创建中央面板
QWidget* ExportPage::createCenterPanel()
{
    // 20260326 ZJH 用 QScrollArea 包裹中间面板，内容多时可滚动
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

    // ===== 进度 + 状态 + 阶段 =====
    m_pLblStatus = new QLabel(QStringLiteral("就绪"));
    m_pLblStatus->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 13px; }"));
    pLayout->addWidget(m_pLblStatus);

    m_pLblPhase = new QLabel(QStringLiteral("阶段: --"));
    m_pLblPhase->setStyleSheet(QStringLiteral("QLabel { color: #64748b; font-size: 12px; }"));
    pLayout->addWidget(m_pLblPhase);

    m_pProgressBar = new QProgressBar();
    m_pProgressBar->setRange(0, 100);
    m_pProgressBar->setValue(0);
    m_pProgressBar->setVisible(false);
    m_pProgressBar->setFixedHeight(6);
    m_pProgressBar->setTextVisible(false);
    m_pProgressBar->setStyleSheet(QStringLiteral(
        "QProgressBar { background: #1a1d24; border: none; border-radius: 3px; }"
        "QProgressBar::chunk { background: #2563eb; border-radius: 3px; }"));
    pLayout->addWidget(m_pProgressBar);

    // ===== 结果卡片 =====
    m_pCardResult = new QFrame();
    m_pCardResult->setVisible(false);  // 20260322 ZJH 初始隐藏
    m_pCardResult->setStyleSheet(QStringLiteral(
        "QFrame { background: #1a1d24; border: 1px solid #10b981; border-radius: 8px; }"));
    QFormLayout* pResultLayout = new QFormLayout(m_pCardResult);
    pResultLayout->setContentsMargins(12, 8, 12, 8);
    pResultLayout->setSpacing(6);

    m_pLblResultPath = new QLabel("--");
    m_pLblResultPath->setStyleSheet(QStringLiteral("QLabel { color: #e2e8f0; border: none; }"));
    m_pLblResultPath->setWordWrap(true);
    pResultLayout->addRow(QStringLiteral("模型路径:"), m_pLblResultPath);

    m_pLblResultSize = new QLabel("--");
    m_pLblResultSize->setStyleSheet(QStringLiteral("QLabel { color: #e2e8f0; border: none; }"));
    pResultLayout->addRow(QStringLiteral("文件大小:"), m_pLblResultSize);

    m_pLblResultTime = new QLabel("--");
    m_pLblResultTime->setStyleSheet(QStringLiteral("QLabel { color: #e2e8f0; border: none; }"));
    pResultLayout->addRow(QStringLiteral("导出耗时:"), m_pLblResultTime);

    pLayout->addWidget(m_pCardResult);

    // ===== 格式兼容性表 =====
    QLabel* pLblCompat = new QLabel(QStringLiteral("格式兼容性"));
    pLblCompat->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 14px; font-weight: bold; }"));
    pLayout->addWidget(pLblCompat);

    m_pTblCompat = new QTableWidget();
    m_pTblCompat->setColumnCount(4);
    m_pTblCompat->setHorizontalHeaderLabels({
        QStringLiteral("格式"),
        QStringLiteral("精度"),
        QStringLiteral("GPU支持"),
        QStringLiteral("备注")
    });
    m_pTblCompat->horizontalHeader()->setStretchLastSection(true);
    m_pTblCompat->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pTblCompat->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pTblCompat->setAlternatingRowColors(true);
    m_pTblCompat->setMaximumHeight(160);
    m_pTblCompat->setStyleSheet(QStringLiteral(
        "QTableWidget { background: #1a1d24; color: #e2e8f0; border: 1px solid #2a2d35;"
        "  gridline-color: #333842; border-radius: 4px; }"
        "QTableWidget::item { padding: 4px; }"
        "QTableWidget::item:alternate { background: #1e2230; }"
        "QHeaderView::section { background: #2a2d35; color: #94a3b8; border: none;"
        "  padding: 6px; font-weight: bold; }"));
    populateCompatibilityTable();  // 20260322 ZJH 填充初始数据
    pLayout->addWidget(m_pTblCompat);

    // ===== 导出历史表 =====
    QLabel* pLblHistory = new QLabel(QStringLiteral("导出历史"));
    pLblHistory->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 14px; font-weight: bold; }"));
    pLayout->addWidget(pLblHistory);

    m_pTblHistory = new QTableWidget();
    m_pTblHistory->setColumnCount(4);
    m_pTblHistory->setHorizontalHeaderLabels({
        QStringLiteral("格式"),
        QStringLiteral("路径"),
        QStringLiteral("大小"),
        QStringLiteral("耗时")
    });
    m_pTblHistory->horizontalHeader()->setStretchLastSection(true);
    m_pTblHistory->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pTblHistory->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pTblHistory->setAlternatingRowColors(true);
    m_pTblHistory->setStyleSheet(QStringLiteral(
        "QTableWidget { background: #1a1d24; color: #e2e8f0; border: 1px solid #2a2d35;"
        "  gridline-color: #333842; border-radius: 4px; }"
        "QTableWidget::item { padding: 4px; }"
        "QTableWidget::item:alternate { background: #1e2230; }"
        "QHeaderView::section { background: #2a2d35; color: #94a3b8; border: none;"
        "  padding: 6px; font-weight: bold; }"));
    pLayout->addWidget(m_pTblHistory, 1);  // 20260322 ZJH stretch=1 让历史表占剩余空间

    pScroll->setWidget(pPanel);  // 20260326 ZJH 将内容放入滚动区域
    return pScroll;  // 20260326 ZJH 返回可滚动面板
}

// 20260322 ZJH 创建右面板
QWidget* ExportPage::createRightPanel()
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

    // ===== 模型信息分组 =====
    QGroupBox* pGrpModel = new QGroupBox(QStringLiteral("模型信息"));
    QFormLayout* pFormModel = new QFormLayout(pGrpModel);
    pFormModel->setSpacing(4);

    m_pLblArch   = new QLabel("--");
    m_pLblParams = new QLabel("--");
    m_pLblFLOPs  = new QLabel("--");
    m_pLblMemory = new QLabel("--");

    pFormModel->addRow(QStringLiteral("架构:"),   m_pLblArch);
    pFormModel->addRow(QStringLiteral("参数量:"), m_pLblParams);
    pFormModel->addRow(QStringLiteral("FLOPs:"),  m_pLblFLOPs);
    pFormModel->addRow(QStringLiteral("内存:"),   m_pLblMemory);

    pLayout->addWidget(pGrpModel);

    // ===== 日志分组 =====
    QGroupBox* pGrpLog = new QGroupBox(QStringLiteral("日志"));
    QVBoxLayout* pLogLayout = new QVBoxLayout(pGrpLog);

    m_pTxtLog = new QTextEdit();
    m_pTxtLog->setReadOnly(true);
    m_pTxtLog->setStyleSheet(QStringLiteral(
        "QTextEdit { background: #13151a; color: #94a3b8; border: 1px solid #2a2d35;"
        "  border-radius: 4px; font-family: Consolas; font-size: 11px; }"));
    pLogLayout->addWidget(m_pTxtLog);

    pLayout->addWidget(pGrpLog, 1);  // 20260322 ZJH stretch=1 让日志区域可伸展

    pLayout->addStretch(0);
    pScrollArea->setWidget(pContainer);
    return pScrollArea;
}

// 20260322 ZJH 填充格式兼容性表
void ExportPage::populateCompatibilityTable()
{
    // 20260322 ZJH 4 种格式的兼容性数据
    struct CompatEntry {
        QString strFormat;
        QString strPrecision;
        QString strGpu;
        QString strNote;
    };

    const CompatEntry arrEntries[] = {
        { "ONNX",      "FP32/FP16", "\xe2\x9c\x93", QStringLiteral("通用跨平台格式")     },
        { "TensorRT",  "FP32/FP16/INT8", "\xe2\x9c\x93", QStringLiteral("NVIDIA GPU 专用优化") },
        { "OpenVINO",  "FP32/FP16/INT8", "\xe2\x9c\x97", QStringLiteral("Intel CPU/iGPU 优化") },
        { QStringLiteral("自研DFM"), "FP32/FP16", "\xe2\x9c\x93", QStringLiteral("OmniMatch 原生格式") }
    };

    m_pTblCompat->setRowCount(4);
    for (int r = 0; r < 4; ++r) {
        m_pTblCompat->setItem(r, 0, new QTableWidgetItem(arrEntries[r].strFormat));
        m_pTblCompat->setItem(r, 1, new QTableWidgetItem(arrEntries[r].strPrecision));
        m_pTblCompat->setItem(r, 2, new QTableWidgetItem(arrEntries[r].strGpu));
        m_pTblCompat->setItem(r, 3, new QTableWidgetItem(arrEntries[r].strNote));

        // 20260322 ZJH 居中对齐
        for (int c = 0; c < 4; ++c) {
            m_pTblCompat->item(r, c)->setTextAlignment(Qt::AlignCenter);
        }
    }
}

// 20260322 ZJH 添加导出历史记录
void ExportPage::addHistoryEntry(const QString& strFormat, const QString& strPath,
                                 const QString& strSize, double dTimeS)
{
    int nRow = m_pTblHistory->rowCount();
    m_pTblHistory->insertRow(nRow);

    m_pTblHistory->setItem(nRow, 0, new QTableWidgetItem(strFormat));
    m_pTblHistory->setItem(nRow, 1, new QTableWidgetItem(strPath));
    m_pTblHistory->setItem(nRow, 2, new QTableWidgetItem(strSize));
    m_pTblHistory->setItem(nRow, 3, new QTableWidgetItem(
        QStringLiteral("%1s").arg(dTimeS, 0, 'f', 1)));

    // 20260322 ZJH 居中对齐
    for (int c = 0; c < 4; ++c) {
        m_pTblHistory->item(nRow, c)->setTextAlignment(Qt::AlignCenter);
    }
}

// 20260324 ZJH 刷新前置检查（基于真实项目状态和模型文件存在性）
void ExportPage::refreshPreChecks()
{
    QString strPass = QStringLiteral("QLabel { color: #10b981; font-size: 12px; }");  // 20260322 ZJH 绿色通过样式
    QString strFail = QStringLiteral("QLabel { color: #64748b; font-size: 12px; }");  // 20260322 ZJH 灰色未通过样式

    bool bHasProject = (m_pProject != nullptr);  // 20260324 ZJH 是否有项目
    bool bTrained = bHasProject && (m_pProject->state() >= om::ProjectState::ModelTrained);  // 20260324 ZJH 项目状态是否已训练

    // 20260324 ZJH 检查 1: 模型是否已训练
    m_pLblCheckTrained->setText(bTrained ? QStringLiteral("\xe2\x9c\x93 模型已训练") : QStringLiteral("\xe2\x9c\x97 模型已训练"));
    m_pLblCheckTrained->setStyleSheet(bTrained ? strPass : strFail);

    // 20260324 ZJH 检查 2: 模型文件是否真实存在于磁盘（而非仅检查项目状态）
    bool bModelExists = false;
    if (bHasProject) {
        QString strModelPath = m_pProject->bestModelPath();
        bModelExists = !strModelPath.isEmpty() && QFileInfo::exists(strModelPath);
    }
    m_pLblCheckModel->setText(bModelExists ? QStringLiteral("\xe2\x9c\x93 模型文件存在") : QStringLiteral("\xe2\x9c\x97 模型文件存在"));
    m_pLblCheckModel->setStyleSheet(bModelExists ? strPass : strFail);
}

// 20260324 ZJH 刷新模型信息（从项目训练配置和实际模型文件获取真实信息）
void ExportPage::refreshModelInfo()
{
    if (!m_pProject) {
        // 20260324 ZJH 无项目时清空所有标签
        m_pLblArch->setText("--");
        m_pLblParams->setText("--");
        m_pLblFLOPs->setText("--");
        m_pLblMemory->setText("--");
        return;
    }

    // 20260324 ZJH 从项目训练配置获取模型架构（而非从任务类型推断默认值）
    const TrainingConfig& config = m_pProject->trainingConfig();
    m_pLblArch->setText(om::modelArchitectureToString(config.eArchitecture));

    // 20260324 ZJH 根据架构估算参数量（精确值需加载模型，此处基于已知架构给出估算）
    QString strParamEstimate;
    switch (config.eArchitecture) {
        case om::ModelArchitecture::ResNet18:
            strParamEstimate = QStringLiteral("~11.7 M");  // 20260324 ZJH ResNet-18 约 11.7M 参数
            break;
        case om::ModelArchitecture::ResNet50:
            strParamEstimate = QStringLiteral("~25.6 M");  // 20260324 ZJH ResNet-50 约 25.6M 参数
            break;
        case om::ModelArchitecture::EfficientNetB0:
            strParamEstimate = QStringLiteral("~5.3 M");   // 20260324 ZJH EfficientNet-B0 约 5.3M
            break;
        case om::ModelArchitecture::MobileNetV4Small:
            strParamEstimate = QStringLiteral("~3.4 M");   // 20260324 ZJH MobileNetV4-Small 约 3.4M
            break;
        case om::ModelArchitecture::ViTTiny:
            strParamEstimate = QStringLiteral("~5.7 M");   // 20260324 ZJH ViT-Tiny 约 5.7M
            break;
        default:
            strParamEstimate = QStringLiteral("--");        // 20260324 ZJH 未知架构不显示估算
            break;
    }
    m_pLblParams->setText(strParamEstimate);

    // 20260324 ZJH 输入尺寸信息显示为 FLOPs 替代（真实 FLOPs 需要推理计算，暂用输入尺寸代替）
    m_pLblFLOPs->setText(QStringLiteral("输入: %1x%1").arg(config.nInputSize));

    // 20260324 ZJH 检查模型文件是否存在，显示真实文件大小
    QString strModelPath = m_pProject->bestModelPath();
    if (!strModelPath.isEmpty() && QFileInfo::exists(strModelPath)) {
        // 20260324 ZJH 模型文件存在，显示真实文件大小
        QFileInfo fi(strModelPath);
        double dSizeMB = fi.size() / (1024.0 * 1024.0);
        m_pLblMemory->setText(QStringLiteral("%1 MB").arg(dSizeMB, 0, 'f', 1));
    } else if (!strModelPath.isEmpty()) {
        // 20260324 ZJH 路径已设置但文件不存在
        m_pLblMemory->setText(QStringLiteral("文件不存在"));
    } else {
        // 20260324 ZJH 没有训练过的模型
        m_pLblMemory->setText(QStringLiteral("未训练"));
    }
}

// 20260322 ZJH 追加日志消息
void ExportPage::appendLog(const QString& strMsg)
{
    if (m_pTxtLog) {
        m_pTxtLog->append(strMsg);
    }
}
