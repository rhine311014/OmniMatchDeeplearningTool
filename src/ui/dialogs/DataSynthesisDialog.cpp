// 20260330 ZJH AI 数据合成对话框实现
// 通过 EngineBridge::synthesizeData() 调用 om.engine.data_synthesis 引擎模块
// 三栏布局: 左=策略+参数 | 中=预览(原图/合成对比) | 右=进度+日志
// 后台线程执行合成，避免阻塞 UI

#include "ui/dialogs/DataSynthesisDialog.h"  // 20260330 ZJH DataSynthesisDialog 类声明
#include "engine/bridge/EngineBridge.h"       // 20260330 ZJH 引擎桥接层（synthesizeData）

#include <QVBoxLayout>        // 20260330 ZJH 垂直布局
#include <QHBoxLayout>        // 20260330 ZJH 水平布局
#include <QFormLayout>        // 20260330 ZJH 表单布局
#include <QSplitter>          // 20260330 ZJH 三栏分隔器
#include <QGroupBox>          // 20260330 ZJH 参数分组框
#include <QScrollArea>        // 20260330 ZJH 左面板滚动区域
#include <QMessageBox>        // 20260330 ZJH 错误/警告对话框
#include <QDateTime>          // 20260330 ZJH 日志时间戳
#include <QTimer>             // 20260330 ZJH 进度条渐进更新定时器
#include <QtConcurrent>       // 20260330 ZJH QtConcurrent::run 后台线程
#include <QFutureWatcher>     // 20260330 ZJH 监视异步任务完成
#include <iostream>           // 20260330 ZJH std::cerr 调试输出

// =========================================================================
// 20260330 ZJH 暗色主题 QSS 样式表常量（与主应用一致 #1a1a2e + #00bcd4）
// =========================================================================

// 20260330 ZJH 对话框全局样式
static const char* s_strDialogStyle = R"(
    QDialog {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    QLabel {
        color: #e0e0e0;
        font-size: 12px;
    }
    QGroupBox {
        color: #00bcd4;
        border: 1px solid #3d4455;
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 16px;
        font-weight: bold;
        font-size: 12px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
    }
    QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #16213e;
        color: #e0e0e0;
        border: 1px solid #3d4455;
        border-radius: 3px;
        padding: 4px 8px;
        min-height: 22px;
        font-size: 12px;
    }
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
        border-color: #00bcd4;
    }
    QComboBox QAbstractItemView {
        background-color: #16213e;
        color: #e0e0e0;
        selection-background-color: #0a3d62;
    }
    QCheckBox {
        color: #e0e0e0;
        font-size: 12px;
        spacing: 6px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
    }
    QPushButton {
        background-color: #1a1d24;
        color: #e0e0e0;
        border: 1px solid #3d4455;
        border-radius: 4px;
        padding: 6px 16px;
        font-size: 12px;
        min-height: 24px;
    }
    QPushButton:hover {
        border-color: #00bcd4;
        color: #00bcd4;
    }
    QPushButton:pressed {
        background-color: #00bcd4;
        color: #1a1a2e;
    }
    QPushButton:disabled {
        color: #555;
        border-color: #2a2a3e;
    }
    QProgressBar {
        background-color: #16213e;
        border: 1px solid #3d4455;
        border-radius: 4px;
        text-align: center;
        color: #e0e0e0;
        font-size: 11px;
        min-height: 18px;
    }
    QProgressBar::chunk {
        background-color: #00bcd4;
        border-radius: 3px;
    }
    QTextEdit {
        background-color: #0f0f23;
        color: #aab2c0;
        border: 1px solid #3d4455;
        border-radius: 4px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 11px;
        padding: 4px;
    }
    QScrollArea {
        background-color: transparent;
        border: none;
    }
)";

// 20260330 ZJH 开始按钮特殊样式（青色强调色背景）
static const char* s_strStartBtnStyle = R"(
    QPushButton {
        background-color: #00838f;
        color: white;
        border: 1px solid #00bcd4;
        border-radius: 4px;
        padding: 8px 20px;
        font-size: 13px;
        font-weight: bold;
        min-height: 28px;
    }
    QPushButton:hover {
        background-color: #00bcd4;
    }
    QPushButton:pressed {
        background-color: #00e5ff;
        color: #1a1a2e;
    }
    QPushButton:disabled {
        background-color: #2a2a3e;
        color: #555;
        border-color: #3d4455;
    }
)";

// 20260330 ZJH 预览标签样式（固定尺寸带边框）
static const char* s_strPreviewLabelStyle = R"(
    QLabel {
        background-color: #0f0f23;
        border: 1px solid #3d4455;
        border-radius: 4px;
        padding: 2px;
    }
)";

// =========================================================================
// 20260330 ZJH 构造函数
// =========================================================================

DataSynthesisDialog::DataSynthesisDialog(QWidget* pParent)
    : QDialog(pParent)  // 20260330 ZJH 调用 QDialog 基类构造
{
    // 20260330 ZJH 设置对话框基本属性
    setWindowTitle(QStringLiteral("AI \u6570\u636e\u5408\u6210"));  // "AI 数据合成"
    setMinimumSize(900, 600);   // 20260330 ZJH 最小尺寸
    resize(1000, 650);          // 20260330 ZJH 默认尺寸

    // 20260330 ZJH 应用暗色主题样式表
    setStyleSheet(QString::fromUtf8(s_strDialogStyle));

    // 20260330 ZJH 创建 UI 布局
    setupUI();

    // 20260330 ZJH 连接内部信号到槽函数（跨线程安全的 queued 连接）
    connect(this, &DataSynthesisDialog::synthesisProgress,
            this, &DataSynthesisDialog::onProgressUpdated, Qt::QueuedConnection);
    connect(this, &DataSynthesisDialog::synthesisFinished,
            this, &DataSynthesisDialog::onSynthesisComplete, Qt::QueuedConnection);
}

// 20260330 ZJH 析构函数
DataSynthesisDialog::~DataSynthesisDialog() = default;

// =========================================================================
// 20260330 ZJH UI 布局创建
// =========================================================================

void DataSynthesisDialog::setupUI()
{
    // 20260330 ZJH 顶层垂直布局
    auto* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(8, 8, 8, 8);  // 20260330 ZJH 外边距
    pMainLayout->setSpacing(8);                     // 20260330 ZJH 控件间距

    // 20260330 ZJH 标题标签
    auto* pLblTitle = new QLabel(QStringLiteral(
        "<span style='font-size:15px; color:#00bcd4; font-weight:bold;'>"
        "\u2728 AI \u6570\u636e\u5408\u6210</span>"
        "<span style='font-size:11px; color:#888;'>"
        "  \u2014 \u4ece\u5c11\u91cf\u7f3a\u9677\u6837\u672c\u81ea\u52a8\u751f\u6210\u5927\u91cf\u5408\u6210\u8bad\u7ec3\u6570\u636e</span>"));
    // 上面是: "✨ AI 数据合成" + " — 从少量缺陷样本自动生成大量合成训练数据"
    pMainLayout->addWidget(pLblTitle);

    // 20260330 ZJH 三栏分隔器
    auto* pSplitter = new QSplitter(Qt::Horizontal, this);
    pSplitter->setChildrenCollapsible(false);  // 20260330 ZJH 防止面板折叠

    // 20260330 ZJH 创建三个面板并添加到分隔器
    std::cerr << "[DataSynth-UI] creating left panel..." << std::endl;
    pSplitter->addWidget(createLeftPanel());
    std::cerr << "[DataSynth-UI] creating center panel..." << std::endl;
    pSplitter->addWidget(createCenterPanel());
    std::cerr << "[DataSynth-UI] creating right panel..." << std::endl;
    pSplitter->addWidget(createRightPanel());
    std::cerr << "[DataSynth-UI] all panels created OK" << std::endl;

    // 20260330 ZJH 设置初始面板比例 (3:4:3)
    pSplitter->setStretchFactor(0, 3);  // 20260330 ZJH 左面板
    pSplitter->setStretchFactor(1, 4);  // 20260330 ZJH 中面板
    pSplitter->setStretchFactor(2, 3);  // 20260330 ZJH 右面板

    pMainLayout->addWidget(pSplitter, 1);  // 20260330 ZJH 分隔器占满剩余空间
}

// =========================================================================
// 20260330 ZJH 左面板 — 策略选择 + 参数配置
// =========================================================================

QWidget* DataSynthesisDialog::createLeftPanel()
{
    // 20260330 ZJH 创建滚动区域（参数较多时可滚动）
    auto* pScrollArea = new QScrollArea();
    pScrollArea->setWidgetResizable(true);  // 20260330 ZJH 内容自动填充
    pScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // 20260330 ZJH 滚动区域内部容器
    auto* pContainer = new QWidget();
    auto* pLayout = new QVBoxLayout(pContainer);
    pLayout->setContentsMargins(6, 6, 6, 6);
    pLayout->setSpacing(8);

    // ===== 策略选择 =====
    {
        auto* pGrp = new QGroupBox(QStringLiteral("\u5408\u6210\u7b56\u7565"));  // "合成策略"
        auto* pForm = new QFormLayout(pGrp);
        pForm->setContentsMargins(8, 16, 8, 8);
        pForm->setSpacing(6);

        // 20260330 ZJH 策略下拉框
        m_pCboStrategy = new QComboBox();
        m_pCboStrategy->addItem(QStringLiteral("CopyPaste \u7f3a\u9677\u7c98\u8d34"));    // "CopyPaste 缺陷粘贴"
        m_pCboStrategy->addItem(QStringLiteral("\u51e0\u4f55+\u5149\u5ea6\u589e\u5f3a"));  // "几何+光度增强"
        m_pCboStrategy->addItem(QStringLiteral("GAN \u751f\u6210"));                        // "GAN 生成"
        m_pCboStrategy->addItem(QStringLiteral("\u81ea\u52a8\u9009\u62e9 (\u63a8\u8350)")); // "自动选择 (推荐)"
        m_pCboStrategy->setCurrentIndex(3);  // 20260330 ZJH 默认自动选择
        pForm->addRow(QStringLiteral("\u7b56\u7565:"), m_pCboStrategy);  // "策略:"

        // 20260330 ZJH 目标合成数量
        m_pSpnTargetCount = new QSpinBox();
        m_pSpnTargetCount->setRange(10, 10000);   // 20260330 ZJH 范围 [10, 10000]
        m_pSpnTargetCount->setValue(500);          // 20260330 ZJH 默认 500
        m_pSpnTargetCount->setSingleStep(50);      // 20260330 ZJH 步进 50
        m_pSpnTargetCount->setSuffix(QStringLiteral(" \u5f20"));  // " 张"
        pForm->addRow(QStringLiteral("\u76ee\u6807\u6570\u91cf:"), m_pSpnTargetCount);  // "目标数量:"

        pLayout->addWidget(pGrp);
    }

    // ===== CopyPaste 参数 =====
    createCopyPasteGroup(pLayout);

    // ===== 增强合成参数 =====
    createAugmentGroup(pLayout);

    // ===== GAN 参数 =====
    createGanGroup(pLayout);

    // 20260330 ZJH 连接策略切换信号
    connect(m_pCboStrategy, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DataSynthesisDialog::onStrategyChanged);

    // 20260330 ZJH 初始状态：自动选择模式隐藏所有参数分组
    onStrategyChanged(3);

    // 20260330 ZJH 弹簧撑底
    pLayout->addStretch(1);

    pScrollArea->setWidget(pContainer);
    return pScrollArea;
}

// 20260330 ZJH 创建 CopyPaste 缺陷粘贴参数分组框
void DataSynthesisDialog::createCopyPasteGroup(QVBoxLayout* pLayout)
{
    m_pGrpCopyPaste = new QGroupBox(QStringLiteral("CopyPaste \u53c2\u6570"));  // "CopyPaste 参数"
    auto* pForm = new QFormLayout(m_pGrpCopyPaste);
    pForm->setContentsMargins(8, 16, 8, 8);
    pForm->setSpacing(6);

    // 20260330 ZJH 缩放范围 — 控制缺陷 patch 粘贴时的随机缩放比例
    m_pSpnScaleMin = new QDoubleSpinBox();
    m_pSpnScaleMin->setRange(0.3, 2.0);    // 20260330 ZJH 最小缩放范围
    m_pSpnScaleMin->setValue(0.7);          // 20260330 ZJH 默认 0.7x
    m_pSpnScaleMin->setSingleStep(0.05);
    m_pSpnScaleMin->setDecimals(2);
    pForm->addRow(QStringLiteral("\u7f29\u653e\u6700\u5c0f:"), m_pSpnScaleMin);  // "缩放最小:"

    m_pSpnScaleMax = new QDoubleSpinBox();
    m_pSpnScaleMax->setRange(0.5, 3.0);    // 20260330 ZJH 最大缩放范围
    m_pSpnScaleMax->setValue(1.3);          // 20260330 ZJH 默认 1.3x
    m_pSpnScaleMax->setSingleStep(0.05);
    m_pSpnScaleMax->setDecimals(2);
    pForm->addRow(QStringLiteral("\u7f29\u653e\u6700\u5927:"), m_pSpnScaleMax);  // "缩放最大:"

    // 20260330 ZJH 旋转范围 — 缺陷 patch 粘贴前的随机旋转角度（±度）
    m_pSpnRotateRange = new QDoubleSpinBox();
    m_pSpnRotateRange->setRange(0.0, 180.0);  // 20260330 ZJH 范围 [0, 180] 度
    m_pSpnRotateRange->setValue(30.0);         // 20260330 ZJH 默认 ±30 度
    m_pSpnRotateRange->setSingleStep(5.0);
    m_pSpnRotateRange->setDecimals(1);
    m_pSpnRotateRange->setSuffix(QStringLiteral("\u00b0"));  // "°"
    pForm->addRow(QStringLiteral("\u65cb\u8f6c\u8303\u56f4:"), m_pSpnRotateRange);  // "旋转范围:"

    // 20260330 ZJH 融合透明度 — 缺陷与背景的混合比例
    m_pSpnBlendAlpha = new QDoubleSpinBox();
    m_pSpnBlendAlpha->setRange(0.1, 1.0);  // 20260330 ZJH 范围 [0.1, 1.0]
    m_pSpnBlendAlpha->setValue(0.9);        // 20260330 ZJH 默认 0.9（接近完全不透明）
    m_pSpnBlendAlpha->setSingleStep(0.05);
    m_pSpnBlendAlpha->setDecimals(2);
    pForm->addRow(QStringLiteral("\u878d\u5408\u900f\u660e\u5ea6:"), m_pSpnBlendAlpha);  // "融合透明度:"

    // 20260330 ZJH 泊松融合开关 — 使用梯度域融合消除拼贴边缘
    m_pChkPoisson = new QCheckBox(QStringLiteral(
        "\u6cca\u677e\u878d\u5408 (\u65e0\u7f1d\u8fb9\u7f18)"));  // "泊松融合 (无缝边缘)"
    m_pChkPoisson->setChecked(true);  // 20260330 ZJH 默认启用
    pForm->addRow(m_pChkPoisson);

    // 20260330 ZJH 随机位置开关 — 缺陷粘贴到背景的随机位置
    m_pChkRandomPos = new QCheckBox(QStringLiteral(
        "\u968f\u673a\u7c98\u8d34\u4f4d\u7f6e"));  // "随机粘贴位置"
    m_pChkRandomPos->setChecked(true);  // 20260330 ZJH 默认启用
    pForm->addRow(m_pChkRandomPos);

    pLayout->addWidget(m_pGrpCopyPaste);
}

// 20260330 ZJH 创建几何+光度增强参数分组框
void DataSynthesisDialog::createAugmentGroup(QVBoxLayout* pLayout)
{
    m_pGrpAugment = new QGroupBox(QStringLiteral(
        "\u589e\u5f3a\u5408\u6210\u53c2\u6570"));  // "增强合成参数"
    auto* pForm = new QFormLayout(m_pGrpAugment);
    pForm->setContentsMargins(8, 16, 8, 8);
    pForm->setSpacing(6);

    // 20260330 ZJH 每张图像变体数
    m_pSpnVariants = new QSpinBox();
    m_pSpnVariants->setRange(1, 100);   // 20260330 ZJH 范围 [1, 100]
    m_pSpnVariants->setValue(20);       // 20260330 ZJH 默认 20 个变体
    m_pSpnVariants->setSingleStep(5);
    pForm->addRow(QStringLiteral(
        "\u6bcf\u5f20\u53d8\u4f53\u6570:"), m_pSpnVariants);  // "每张变体数:"

    // 20260330 ZJH 弹性变形开关 — 模拟材料/产品的真实形变
    m_pChkElastic = new QCheckBox(QStringLiteral(
        "\u5f39\u6027\u53d8\u5f62"));  // "弹性变形"
    m_pChkElastic->setChecked(true);
    pForm->addRow(m_pChkElastic);

    // 20260330 ZJH 透视变换开关 — 模拟相机角度变化
    m_pChkPerspective = new QCheckBox(QStringLiteral(
        "\u900f\u89c6\u53d8\u6362"));  // "透视变换"
    m_pChkPerspective->setChecked(true);
    pForm->addRow(m_pChkPerspective);

    // 20260330 ZJH 颜色迁移开关 — 匹配目标图像的色调
    m_pChkColorTransfer = new QCheckBox(QStringLiteral(
        "\u989c\u8272\u8fc1\u79fb"));  // "颜色迁移"
    m_pChkColorTransfer->setChecked(true);
    pForm->addRow(m_pChkColorTransfer);

    pLayout->addWidget(m_pGrpAugment);
}

// 20260330 ZJH 创建 GAN 合成参数分组框
void DataSynthesisDialog::createGanGroup(QVBoxLayout* pLayout)
{
    m_pGrpGAN = new QGroupBox(QStringLiteral("GAN \u53c2\u6570"));  // "GAN 参数"
    auto* pForm = new QFormLayout(m_pGrpGAN);
    pForm->setContentsMargins(8, 16, 8, 8);
    pForm->setSpacing(6);

    // 20260330 ZJH GAN 训练轮数
    m_pSpnGanEpochs = new QSpinBox();
    m_pSpnGanEpochs->setRange(50, 1000);  // 20260330 ZJH 范围 [50, 1000]
    m_pSpnGanEpochs->setValue(200);        // 20260330 ZJH 默认 200 轮
    m_pSpnGanEpochs->setSingleStep(50);
    pForm->addRow(QStringLiteral(
        "\u8bad\u7ec3\u8f6e\u6570:"), m_pSpnGanEpochs);  // "训练轮数:"

    // 20260330 ZJH GAN 状态提示
    m_pLblGanStatus = new QLabel(QStringLiteral(
        "<i style='color:#ff9800;'>"
        "\u26a0 GAN \u9700\u8981 \u226550 \u5f20\u7f3a\u9677\u6837\u672c\u624d\u80fd\u6536\u655b</i>"));
    // "⚠ GAN 需要 ≥50 张缺陷样本才能收敛"
    m_pLblGanStatus->setWordWrap(true);
    pForm->addRow(m_pLblGanStatus);

    pLayout->addWidget(m_pGrpGAN);
}

// =========================================================================
// 20260330 ZJH 中央面板 — 预览区域
// =========================================================================

QWidget* DataSynthesisDialog::createCenterPanel()
{
    auto* pWidget = new QWidget();
    auto* pLayout = new QVBoxLayout(pWidget);
    pLayout->setContentsMargins(6, 6, 6, 6);
    pLayout->setSpacing(8);

    // 20260330 ZJH 预览标题
    auto* pLblTitle = new QLabel(QStringLiteral(
        "<b style='color:#00bcd4;'>"
        "\u9884\u89c8\u5bf9\u6bd4</b>"));  // "预览对比"
    pLayout->addWidget(pLblTitle);

    // 20260330 ZJH 预览图像区域（左=原图，右=合成）
    auto* pPreviewLayout = new QHBoxLayout();
    pPreviewLayout->setSpacing(8);

    // 20260330 ZJH 原始图像预览标签
    auto* pBeforeLayout = new QVBoxLayout();
    auto* pLblBeforeTitle = new QLabel(QStringLiteral(
        "<span style='color:#888;'>\u539f\u59cb\u56fe\u50cf</span>"));  // "原始图像"
    pLblBeforeTitle->setAlignment(Qt::AlignCenter);
    pBeforeLayout->addWidget(pLblBeforeTitle);

    m_pLblPreviewBefore = new QLabel();
    m_pLblPreviewBefore->setFixedSize(200, 200);  // 20260330 ZJH 固定预览尺寸
    m_pLblPreviewBefore->setAlignment(Qt::AlignCenter);
    m_pLblPreviewBefore->setStyleSheet(QString::fromUtf8(s_strPreviewLabelStyle));
    m_pLblPreviewBefore->setText(QStringLiteral(
        "<span style='color:#555;'>\u65e0\u6570\u636e</span>"));  // "无数据"
    pBeforeLayout->addWidget(m_pLblPreviewBefore, 0, Qt::AlignCenter);
    pPreviewLayout->addLayout(pBeforeLayout);

    // 20260330 ZJH 箭头分隔
    auto* pLblArrow = new QLabel(QStringLiteral(
        "<span style='font-size:24px; color:#00bcd4;'>\u2192</span>"));  // "→"
    pLblArrow->setAlignment(Qt::AlignCenter);
    pPreviewLayout->addWidget(pLblArrow);

    // 20260330 ZJH 合成结果预览标签
    auto* pAfterLayout = new QVBoxLayout();
    auto* pLblAfterTitle = new QLabel(QStringLiteral(
        "<span style='color:#888;'>\u5408\u6210\u7ed3\u679c</span>"));  // "合成结果"
    pLblAfterTitle->setAlignment(Qt::AlignCenter);
    pAfterLayout->addWidget(pLblAfterTitle);

    m_pLblPreviewAfter = new QLabel();
    m_pLblPreviewAfter->setFixedSize(200, 200);  // 20260330 ZJH 固定预览尺寸
    m_pLblPreviewAfter->setAlignment(Qt::AlignCenter);
    m_pLblPreviewAfter->setStyleSheet(QString::fromUtf8(s_strPreviewLabelStyle));
    m_pLblPreviewAfter->setText(QStringLiteral(
        "<span style='color:#555;'>\u70b9\u51fb\u9884\u89c8</span>"));  // "点击预览"
    pAfterLayout->addWidget(m_pLblPreviewAfter, 0, Qt::AlignCenter);
    pPreviewLayout->addLayout(pAfterLayout);

    pLayout->addLayout(pPreviewLayout);

    // 20260330 ZJH 预览信息文本
    m_pLblPreviewInfo = new QLabel();
    m_pLblPreviewInfo->setAlignment(Qt::AlignCenter);
    m_pLblPreviewInfo->setWordWrap(true);
    m_pLblPreviewInfo->setStyleSheet(QStringLiteral("color: #888; font-size: 11px;"));
    pLayout->addWidget(m_pLblPreviewInfo);

    // 20260330 ZJH 预览按钮
    m_pBtnPreview = new QPushButton(QStringLiteral(
        "\u751f\u6210\u9884\u89c8"));  // "生成预览"
    m_pBtnPreview->setToolTip(QStringLiteral(
        "\u7528\u5f53\u524d\u53c2\u6570\u751f\u62101\u5f20\u5408\u6210\u56fe\u50cf\u9884\u89c8"));
    // "用当前参数生成1张合成图像预览"
    connect(m_pBtnPreview, &QPushButton::clicked,
            this, &DataSynthesisDialog::onPreview);
    pLayout->addWidget(m_pBtnPreview);

    // 20260330 ZJH 弹簧撑底
    pLayout->addStretch(1);

    return pWidget;
}

// =========================================================================
// 20260330 ZJH 右面板 — 进度 + 日志
// =========================================================================

QWidget* DataSynthesisDialog::createRightPanel()
{
    auto* pWidget = new QWidget();
    auto* pLayout = new QVBoxLayout(pWidget);
    pLayout->setContentsMargins(6, 6, 6, 6);
    pLayout->setSpacing(8);

    // 20260330 ZJH 数据概览
    auto* pGrpData = new QGroupBox(QStringLiteral(
        "\u6570\u636e\u6982\u89c8"));  // "数据概览"
    auto* pDataLayout = new QVBoxLayout(pGrpData);
    pDataLayout->setContentsMargins(8, 16, 8, 8);

    // 20260330 ZJH 数据统计标签（延迟到 setSourceData 时更新）
    m_pLblProgress = new QLabel(QStringLiteral(
        "\u5c1a\u672a\u52a0\u8f7d\u6570\u636e"));  // "尚未加载数据"
    m_pLblProgress->setWordWrap(true);
    pDataLayout->addWidget(m_pLblProgress);
    pLayout->addWidget(pGrpData);

    // 20260330 ZJH 进度条
    m_pBarProgress = new QProgressBar();
    m_pBarProgress->setRange(0, 100);  // 20260330 ZJH 百分比范围
    m_pBarProgress->setValue(0);
    pLayout->addWidget(m_pBarProgress);

    // 20260330 ZJH 操作按钮区域
    auto* pBtnLayout = new QHBoxLayout();
    pBtnLayout->setSpacing(8);

    m_pBtnStart = new QPushButton(QStringLiteral(
        "\u5f00\u59cb\u5408\u6210"));  // "开始合成"
    m_pBtnStart->setStyleSheet(QString::fromUtf8(s_strStartBtnStyle));
    m_pBtnStart->setToolTip(QStringLiteral(
        "\u542f\u52a8\u540e\u53f0\u7ebf\u7a0b\u6267\u884c AI \u6570\u636e\u5408\u6210"));
    // "启动后台线程执行 AI 数据合成"
    connect(m_pBtnStart, &QPushButton::clicked,
            this, &DataSynthesisDialog::onStartSynthesis);
    pBtnLayout->addWidget(m_pBtnStart);

    m_pBtnCancel = new QPushButton(QStringLiteral(
        "\u53d6\u6d88"));  // "取消"
    connect(m_pBtnCancel, &QPushButton::clicked,
            this, &QDialog::reject);
    pBtnLayout->addWidget(m_pBtnCancel);

    pLayout->addLayout(pBtnLayout);

    // 20260330 ZJH 日志输出区域
    auto* pLblLog = new QLabel(QStringLiteral(
        "<b style='color:#00bcd4;'>\u65e5\u5fd7</b>"));  // "日志"
    pLayout->addWidget(pLblLog);

    m_pTxtLog = new QTextEdit();
    m_pTxtLog->setReadOnly(true);  // 20260330 ZJH 只读
    m_pTxtLog->setPlaceholderText(QStringLiteral(
        "\u5408\u6210\u65e5\u5fd7\u5c06\u663e\u793a\u5728\u6b64\u5904..."));
    // "合成日志将显示在此处..."
    pLayout->addWidget(m_pTxtLog, 1);  // 20260330 ZJH 日志占满剩余空间

    return pWidget;
}

// =========================================================================
// 20260330 ZJH 数据设置
// =========================================================================

void DataSynthesisDialog::setSourceData(
    const std::vector<std::vector<float>>& vecNormalImages,
    const std::vector<std::vector<float>>& vecDefectImages,
    int nC, int nH, int nW)
{
    // 20260330 ZJH 保存数据引用
    m_vecNormalImages = vecNormalImages;
    m_vecDefectImages = vecDefectImages;
    m_nC = nC;
    m_nH = nH;
    m_nW = nW;

    // 20260330 ZJH 更新数据概览标签
    m_pLblProgress->setText(QStringLiteral(
        "\u6b63\u5e38\u6837\u672c: <b>%1</b> \u5f20 | "         // "正常样本: %1 张 | "
        "\u7f3a\u9677\u6837\u672c: <b>%2</b> \u5f20<br>"        // "缺陷样本: %2 张"
        "\u56fe\u50cf\u5c3a\u5bf8: %3\u00d7%4\u00d7%5")          // "图像尺寸: CxHxW"
        .arg(vecNormalImages.size())
        .arg(vecDefectImages.size())
        .arg(nC).arg(nH).arg(nW));

    // 20260330 ZJH 显示第一张缺陷图像作为原始预览
    if (!vecDefectImages.empty() && nH > 0 && nW > 0) {
        QImage qimg = floatImageToQImage(vecDefectImages[0], nC, nH, nW);
        if (!qimg.isNull()) {
            m_pLblPreviewBefore->setPixmap(
                QPixmap::fromImage(qimg).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

    // 20260330 ZJH GAN 状态更新
    if (vecDefectImages.size() >= 50) {
        m_pLblGanStatus->setText(QStringLiteral(
            "<i style='color:#4caf50;'>"
            "\u2705 \u7f3a\u9677\u6837\u672c\u5145\u8db3 (%1 \u5f20)\uff0cGAN \u53ef\u7528</i>")
            .arg(vecDefectImages.size()));
        // "✅ 缺陷样本充足 (N 张)，GAN 可用"
    } else {
        m_pLblGanStatus->setText(QStringLiteral(
            "<i style='color:#ff9800;'>"
            "\u26a0 \u7f3a\u9677\u6837\u672c %1 \u5f20 < 50\uff0cGAN \u96be\u4ee5\u6536\u655b</i>")
            .arg(vecDefectImages.size()));
        // "⚠ 缺陷样本 N 张 < 50，GAN 难以收敛"
    }

    // 20260330 ZJH 日志记录
    appendLog(QStringLiteral("[%1] \u6570\u636e\u5df2\u52a0\u8f7d: "
                             "\u6b63\u5e38 %2 \u5f20, \u7f3a\u9677 %3 \u5f20, "
                             "\u5c3a\u5bf8 %4x%5x%6")
              .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")))
              .arg(vecNormalImages.size()).arg(vecDefectImages.size())
              .arg(nC).arg(nH).arg(nW));
    // "[HH:mm:ss] 数据已加载: 正常 N 张, 缺陷 M 张, 尺寸 CxHxW"
}

// =========================================================================
// 20260330 ZJH 获取合成结果
// =========================================================================

DataSynthesisDialog::SynthResult DataSynthesisDialog::getResult() const
{
    return m_result;  // 20260330 ZJH 返回合成结果（合成完成后已填充）
}

// =========================================================================
// 20260330 ZJH 策略切换
// =========================================================================

void DataSynthesisDialog::onStrategyChanged(int nIndex)
{
    // 20260330 ZJH 根据策略索引显示/隐藏对应参数分组框
    // 0=CopyPaste, 1=增强, 2=GAN, 3=自动(全隐藏)
    // 20260330 ZJH 空指针保护（构造期间 createLeftPanel 先于 createCenterPanel 调用）
    if (m_pGrpCopyPaste) m_pGrpCopyPaste->setVisible(nIndex == 0);
    if (m_pGrpAugment)   m_pGrpAugment->setVisible(nIndex == 1);
    if (m_pGrpGAN)       m_pGrpGAN->setVisible(nIndex == 2);

    // 20260330 ZJH 自动模式说明
    if (nIndex == 3 && m_pLblPreviewInfo) {
        m_pLblPreviewInfo->setText(QStringLiteral(
            "\u81ea\u52a8\u6a21\u5f0f: \u6839\u636e\u6570\u636e\u91cf\u81ea\u52a8"
            "\u9009\u62e9 CopyPaste + \u589e\u5f3a + GAN \u7684\u6700\u4f18\u7ec4\u5408"));
        // "自动模式: 根据数据量自动选择 CopyPaste + 增强 + GAN 的最优组合"
    } else if (m_pLblPreviewInfo) {
        m_pLblPreviewInfo->clear();
    }
}

// =========================================================================
// 20260330 ZJH 预览
// =========================================================================

void DataSynthesisDialog::onPreview()
{
    // 20260330 ZJH 输入验证
    if (m_vecDefectImages.empty()) {
        QMessageBox::warning(this,
            QStringLiteral("AI \u6570\u636e\u5408\u6210"),  // "AI 数据合成"
            QStringLiteral("\u8bf7\u5148\u52a0\u8f7d\u7f3a\u9677\u6837\u672c\u6570\u636e"));
        // "请先加载缺陷样本数据"
        return;
    }

    // 20260330 ZJH 构建合成参数（只生成1张用于预览）
    BridgeSynthesisParams params;
    params.nStrategy = m_pCboStrategy->currentIndex();  // 20260330 ZJH 策略索引
    params.nTargetCount = 1;                            // 20260330 ZJH 预览只需1张
    params.fScaleMin = static_cast<float>(m_pSpnScaleMin->value());
    params.fScaleMax = static_cast<float>(m_pSpnScaleMax->value());
    params.fRotateRange = static_cast<float>(m_pSpnRotateRange->value());
    params.fBlendAlpha = static_cast<float>(m_pSpnBlendAlpha->value());
    params.bPoisson = m_pChkPoisson->isChecked();
    params.nVariantsPerImage = m_pSpnVariants->value();
    params.bElastic = m_pChkElastic->isChecked();
    params.bPerspective = m_pChkPerspective->isChecked();
    params.nGanEpochs = m_pSpnGanEpochs->value();

    appendLog(QStringLiteral("[%1] \u6b63\u5728\u751f\u6210\u9884\u89c8...")
              .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"))));
    // "[HH:mm:ss] 正在生成预览..."

    // 20260330 ZJH 调用引擎桥接层合成1张图像
    EngineBridge bridge;
    BridgeSynthesisResult synthResult = bridge.synthesizeData(
        m_vecNormalImages, m_vecDefectImages,
        m_nC, m_nH, m_nW, params);

    // 20260330 ZJH 显示预览结果
    if (!synthResult.vecImages.empty() && m_nH > 0 && m_nW > 0) {
        QImage qimg = floatImageToQImage(synthResult.vecImages[0], m_nC, m_nH, m_nW);
        if (!qimg.isNull()) {
            m_pLblPreviewAfter->setPixmap(
                QPixmap::fromImage(qimg).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        m_pLblPreviewInfo->setText(QStringLiteral(
            "\u9884\u89c8\u6210\u529f \u2014 "
            "\u5408\u6210 %1 \u5f20\u56fe\u50cf")
            .arg(synthResult.nSynthCount));
        // "预览成功 — 合成 1 张图像"
        appendLog(QStringLiteral("[%1] \u9884\u89c8\u5b8c\u6210")
                  .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"))));
        // "[HH:mm:ss] 预览完成"
    } else {
        m_pLblPreviewInfo->setText(QStringLiteral(
            "<span style='color:#f44336;'>\u9884\u89c8\u5931\u8d25 \u2014 \u672a\u751f\u6210\u56fe\u50cf</span>"));
        // "预览失败 — 未生成图像"
        appendLog(QStringLiteral("[%1] \u9884\u89c8\u5931\u8d25: \u672a\u751f\u6210\u56fe\u50cf")
                  .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"))));
        // "[HH:mm:ss] 预览失败: 未生成图像"
    }
}

// =========================================================================
// 20260330 ZJH 开始合成
// =========================================================================

void DataSynthesisDialog::onStartSynthesis()
{
    // 20260330 ZJH 防止重复启动
    if (m_bRunning) return;

    // 20260330 ZJH 输入验证
    if (m_vecDefectImages.empty()) {
        QMessageBox::warning(this,
            QStringLiteral("AI \u6570\u636e\u5408\u6210"),
            QStringLiteral("\u7f3a\u9677\u6837\u672c\u4e3a\u7a7a\uff0c"
                           "\u65e0\u6cd5\u6267\u884c\u5408\u6210\u3002\n\n"
                           "\u8bf7\u5148\u5728\u56fe\u5e93\u9875\u5bfc\u5165\u5e76\u6807\u6ce8\u7f3a\u9677\u56fe\u50cf\u3002"));
        // "缺陷样本为空，无法执行合成。\n\n请先在图库页导入并标注缺陷图像。"
        return;
    }

    // 20260330 ZJH 锁定 UI
    m_bRunning = true;
    m_pBtnStart->setEnabled(false);
    m_pBtnPreview->setEnabled(false);
    m_pCboStrategy->setEnabled(false);
    m_pSpnTargetCount->setEnabled(false);
    m_pBarProgress->setValue(0);

    // 20260330 ZJH 构建合成参数
    BridgeSynthesisParams params;
    params.nStrategy = m_pCboStrategy->currentIndex();
    params.nTargetCount = m_pSpnTargetCount->value();
    params.fScaleMin = static_cast<float>(m_pSpnScaleMin->value());
    params.fScaleMax = static_cast<float>(m_pSpnScaleMax->value());
    params.fRotateRange = static_cast<float>(m_pSpnRotateRange->value());
    params.fBlendAlpha = static_cast<float>(m_pSpnBlendAlpha->value());
    params.bPoisson = m_pChkPoisson->isChecked();
    params.nVariantsPerImage = m_pSpnVariants->value();
    params.bElastic = m_pChkElastic->isChecked();
    params.bPerspective = m_pChkPerspective->isChecked();
    params.nGanEpochs = m_pSpnGanEpochs->value();

    appendLog(QStringLiteral("[%1] \u5f00\u59cb\u5408\u6210: \u7b56\u7565=%2, "
                             "\u76ee\u6807=%3 \u5f20")
              .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")))
              .arg(m_pCboStrategy->currentText())
              .arg(params.nTargetCount));
    // "[HH:mm:ss] 开始合成: 策略=XXX, 目标=N 张"

    // 20260330 ZJH 捕获成员数据的副本（线程安全）
    auto vecNormals = m_vecNormalImages;
    auto vecDefects = m_vecDefectImages;
    int nC = m_nC, nH = m_nH, nW = m_nW;

    // 20260330 ZJH 使用 QtConcurrent::run 在后台线程执行合成
    auto* pWatcher = new QFutureWatcher<BridgeSynthesisResult>(this);
    connect(pWatcher, &QFutureWatcher<BridgeSynthesisResult>::finished, this,
            [this, pWatcher]()
    {
        // 20260330 ZJH 获取合成结果
        BridgeSynthesisResult synthResult = pWatcher->result();
        pWatcher->deleteLater();  // 20260330 ZJH 释放 watcher

        // 20260330 ZJH 存储结果
        m_result.vecImages = std::move(synthResult.vecImages);
        m_result.nOrigCount = synthResult.nOrigCount;
        m_result.nSynthCount = synthResult.nSynthCount;

        // 20260330 ZJH 发射完成信号
        bool bSuccess = (m_result.nSynthCount > 0);
        QString strMsg = bSuccess
            ? QStringLiteral("\u5408\u6210\u5b8c\u6210: \u751f\u6210\u4e86 %1 \u5f20\u8bad\u7ec3\u56fe\u50cf")
              .arg(m_result.nSynthCount)
            // "合成完成: 生成了 N 张训练图像"
            : QStringLiteral("\u5408\u6210\u5931\u8d25: \u672a\u751f\u6210\u4efb\u4f55\u56fe\u50cf");
            // "合成失败: 未生成任何图像"
        emit synthesisFinished(bSuccess, strMsg);
    });

    // 20260330 ZJH 启动后台合成任务
    QFuture<BridgeSynthesisResult> future = QtConcurrent::run(
        [vecNormals, vecDefects, nC, nH, nW, params, this]() -> BridgeSynthesisResult
    {
        // 20260330 ZJH 发射进度更新（10%: 开始）
        emit synthesisProgress(10, QStringLiteral(
            "\u6b63\u5728\u521d\u59cb\u5316\u5408\u6210\u5f15\u64ce..."));  // "正在初始化合成引擎..."

        // 20260330 ZJH 调用引擎桥接层
        EngineBridge bridge;
        BridgeSynthesisResult result = bridge.synthesizeData(
            vecNormals, vecDefects, nC, nH, nW, params);

        // 20260330 ZJH 发射进度更新（100%: 完成）
        emit synthesisProgress(100, QStringLiteral(
            "\u5408\u6210\u5b8c\u6210"));  // "合成完成"

        return result;
    });

    pWatcher->setFuture(future);

    // 20260330 ZJH 模拟中间进度（实际合成在后台一次完成）
    // 使用 timer 渐进更新进度条，给用户反馈
    auto* pTimer = new QTimer(this);
    connect(pTimer, &QTimer::timeout, this, [this, pTimer]() {
        if (!m_bRunning) {
            pTimer->stop();
            pTimer->deleteLater();
            return;
        }
        // 20260330 ZJH 缓慢递增进度条到 90%（实际完成时跳到 100%）
        int nCur = m_pBarProgress->value();
        if (nCur < 90) {
            m_pBarProgress->setValue(nCur + 2);  // 20260330 ZJH 每次 +2%
        }
    });
    pTimer->start(500);  // 20260330 ZJH 每 500ms 更新一次
}

// =========================================================================
// 20260330 ZJH 进度更新槽
// =========================================================================

void DataSynthesisDialog::onProgressUpdated(int nPercent, const QString& strStatus)
{
    m_pBarProgress->setValue(nPercent);  // 20260330 ZJH 更新进度条
    appendLog(QStringLiteral("[%1] %2 (%3%)")
              .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")))
              .arg(strStatus)
              .arg(nPercent));
}

// =========================================================================
// 20260330 ZJH 合成完成槽
// =========================================================================

void DataSynthesisDialog::onSynthesisComplete(bool bSuccess, const QString& strMessage)
{
    // 20260330 ZJH 解锁 UI
    m_bRunning = false;
    m_pBtnStart->setEnabled(true);
    m_pBtnPreview->setEnabled(true);
    m_pCboStrategy->setEnabled(true);
    m_pSpnTargetCount->setEnabled(true);
    m_pBarProgress->setValue(100);

    appendLog(QStringLiteral("[%1] %2")
              .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")))
              .arg(strMessage));

    if (bSuccess) {
        // 20260330 ZJH 显示最后一张合成图像作为预览
        if (!m_result.vecImages.empty() && m_nH > 0 && m_nW > 0) {
            QImage qimg = floatImageToQImage(m_result.vecImages.back(), m_nC, m_nH, m_nW);
            if (!qimg.isNull()) {
                m_pLblPreviewAfter->setPixmap(
                    QPixmap::fromImage(qimg).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
            }
        }

        // 20260330 ZJH 询问用户是否接受合成结果
        auto nRet = QMessageBox::question(this,
            QStringLiteral("AI \u6570\u636e\u5408\u6210"),  // "AI 数据合成"
            QStringLiteral("\u5408\u6210\u5b8c\u6210\uff01\u5171\u751f\u6210 %1 \u5f20\u8bad\u7ec3\u56fe\u50cf\u3002\n\n"
                           "\u662f\u5426\u5c06\u5408\u6210\u6570\u636e\u6dfb\u52a0\u5230\u8bad\u7ec3\u96c6\uff1f")
                .arg(m_result.nSynthCount),
            // "合成完成！共生成 N 张训练图像。\n\n是否将合成数据添加到训练集？"
            QMessageBox::Yes | QMessageBox::No);

        if (nRet == QMessageBox::Yes) {
            accept();   // 20260330 ZJH 关闭对话框并返回 Accepted
        }
    } else {
        QMessageBox::warning(this,
            QStringLiteral("AI \u6570\u636e\u5408\u6210"),
            strMessage);
    }
}

// =========================================================================
// 20260330 ZJH 工具方法
// =========================================================================

QImage DataSynthesisDialog::floatImageToQImage(
    const std::vector<float>& vecImage,
    int nC, int nH, int nW) const
{
    // 20260330 ZJH 验证数据大小
    size_t nExpected = static_cast<size_t>(nC) * static_cast<size_t>(nH) * static_cast<size_t>(nW);
    if (vecImage.size() < nExpected || nH <= 0 || nW <= 0) {
        return QImage();  // 20260330 ZJH 数据不足返回空图像
    }

    if (nC == 1) {
        // 20260330 ZJH 单通道灰度图
        QImage qimg(nW, nH, QImage::Format_Grayscale8);
        for (int y = 0; y < nH; ++y) {
            auto* pLine = qimg.scanLine(y);  // 20260330 ZJH 获取行指针
            for (int x = 0; x < nW; ++x) {
                // 20260330 ZJH CHW 格式索引: channel * H * W + y * W + x
                float fVal = vecImage[static_cast<size_t>(y * nW + x)];
                // 20260330 ZJH float [0,1] → uint8 [0,255]
                pLine[x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, fVal * 255.0f)));
            }
        }
        return qimg;
    } else if (nC >= 3) {
        // 20260330 ZJH RGB 三通道图像
        QImage qimg(nW, nH, QImage::Format_RGB888);
        size_t nPixels = static_cast<size_t>(nH * nW);
        for (int y = 0; y < nH; ++y) {
            auto* pLine = qimg.scanLine(y);  // 20260330 ZJH 获取行指针
            for (int x = 0; x < nW; ++x) {
                size_t nIdx = static_cast<size_t>(y * nW + x);
                // 20260330 ZJH CHW → RGB: R=channel0, G=channel1, B=channel2
                float fR = vecImage[0 * nPixels + nIdx];
                float fG = vecImage[1 * nPixels + nIdx];
                float fB = vecImage[2 * nPixels + nIdx];
                pLine[x * 3 + 0] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, fR * 255.0f)));
                pLine[x * 3 + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, fG * 255.0f)));
                pLine[x * 3 + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, fB * 255.0f)));
            }
        }
        return qimg;
    }

    return QImage();  // 20260330 ZJH 不支持的通道数返回空图像
}

void DataSynthesisDialog::appendLog(const QString& strMessage)
{
    m_pTxtLog->append(strMessage);  // 20260330 ZJH 追加日志行
    // 20260330 ZJH 自动滚动到底部
    auto cursor = m_pTxtLog->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_pTxtLog->setTextCursor(cursor);
}
