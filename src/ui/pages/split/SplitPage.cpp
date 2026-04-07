// 20260322 ZJH SplitPage 实现
// 三栏布局：左侧配置面板 + 中央统计视图 + 右侧说明面板
// 滑块联动：训练集 + 验证集 + 测试集（100% 约束）
// 执行拆分后刷新统计卡片、标签分布表格和条形图

#include "ui/pages/split/SplitPage.h"           // 20260322 ZJH 自身声明
#include "ui/widgets/ClassDistributionChart.h"  // 20260322 ZJH 类别分布图控件
#include "core/data/ImageDataset.h"             // 20260322 ZJH 数据集管理（autoSplit/countBySplit）
#include "core/data/LabelInfo.h"                // 20260322 ZJH 标签信息（nId/strName）
#include "core/data/ImageEntry.h"               // 20260322 ZJH 图像条目（splitType）
#include "core/project/Project.h"              // 20260322 ZJH 项目类（dataset() 访问器）
#include "core/DLTypes.h"                       // 20260322 ZJH SplitType 枚举

// 20260322 ZJH Qt 布局与控件
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QSlider>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QFrame>
#include <QScrollArea>
#include <QSizePolicy>
#include <QFont>
#include <QMessageBox>
#include <QSpacerItem>

// ============================================================================
// 20260322 ZJH 内联 QSS 样式常量，统一控件外观
// ============================================================================

// 20260322 ZJH GroupBox 样式（与暗色主题一致）
static const QString s_strGroupBoxStyle = QStringLiteral(
    "QGroupBox {"
    "  color: #94a3b8;"
    "  font-size: 11px;"
    "  font-weight: 600;"
    "  border: 1px solid #2a2d35;"
    "  border-radius: 4px;"
    "  margin-top: 16px;"
    "  padding-top: 8px;"
    "}"
    "QGroupBox::title {"
    "  subcontrol-origin: margin;"
    "  subcontrol-position: top left;"
    "  padding: 0 6px;"
    "  left: 8px;"
    "}"
);

// 20260322 ZJH 主按钮（蓝色 — 执行拆分）样式
static const QString s_strPrimaryBtnStyle = QStringLiteral(
    "QPushButton {"
    "  background-color: #2563eb;"
    "  color: #ffffff;"
    "  border: none;"
    "  border-radius: 4px;"
    "  padding: 6px 12px;"
    "  font-weight: 600;"
    "  font-size: 12px;"
    "}"
    "QPushButton:hover  { background-color: #1d4ed8; }"
    "QPushButton:pressed { background-color: #1e40af; }"
);

// 20260322 ZJH 次要按钮（灰边框）样式
static const QString s_strSecondaryBtnStyle = QStringLiteral(
    "QPushButton {"
    "  background-color: transparent;"
    "  color: #94a3b8;"
    "  border: 1px solid #2a2d35;"
    "  border-radius: 4px;"
    "  padding: 4px 10px;"
    "  font-size: 11px;"
    "}"
    "QPushButton:hover  { background-color: #1e2230; border-color: #3b82f6; color: #e2e8f0; }"
    "QPushButton:pressed { background-color: #2a2d35; }"
);

// ============================================================================
// 20260322 ZJH 构造函数
// ============================================================================
SplitPage::SplitPage(QWidget* pParent)
    : BasePage(pParent)                   // 20260406 ZJH 初始化页面基类
    , m_pSplitNameEdit(nullptr)           // 20260406 ZJH 拆分名称输入框初始为空
    , m_pTrainSlider(nullptr)             // 20260406 ZJH 训练集比例滑块初始为空
    , m_pTrainSpin(nullptr)               // 20260406 ZJH 训练集比例微调框初始为空
    , m_pValSlider(nullptr)               // 20260406 ZJH 验证集比例滑块初始为空
    , m_pValSpin(nullptr)                 // 20260406 ZJH 验证集比例微调框初始为空
    , m_pTestLabel(nullptr)               // 20260406 ZJH 测试集比例显示标签初始为空
    , m_pStratifiedCheck(nullptr)         // 20260406 ZJH 分层采样复选框初始为空
    , m_pPreset701515Btn(nullptr)         // 20260406 ZJH 70/15/15 预设按钮初始为空
    , m_pPreset801010Btn(nullptr)         // 20260406 ZJH 80/10/10 预设按钮初始为空
    , m_pPreset602020Btn(nullptr)         // 20260406 ZJH 60/20/20 预设按钮初始为空
    , m_pExecuteBtn(nullptr)              // 20260406 ZJH 执行拆分按钮初始为空
    , m_pResetBtn(nullptr)                // 20260406 ZJH 重置拆分按钮初始为空
    , m_pTrainCount(nullptr)              // 20260406 ZJH 训练集统计卡片数量标签初始为空
    , m_pTrainPct(nullptr)                // 20260406 ZJH 训练集统计卡片百分比标签初始为空
    , m_pValCount(nullptr)                // 20260406 ZJH 验证集统计卡片数量标签初始为空
    , m_pValPct(nullptr)                  // 20260406 ZJH 验证集统计卡片百分比标签初始为空
    , m_pTestCount(nullptr)               // 20260406 ZJH 测试集统计卡片数量标签初始为空
    , m_pTestPct(nullptr)                 // 20260406 ZJH 测试集统计卡片百分比标签初始为空
    , m_pDistTable(nullptr)               // 20260406 ZJH 标签分布表格初始为空
    , m_pChart(nullptr)                   // 20260406 ZJH 类别分布条形图初始为空
    , m_pSplitStatusLabel(nullptr)        // 20260406 ZJH 拆分状态标签初始为空
    , m_bUpdating(false)                  // 20260406 ZJH 联动防递归标志初始为 false
{
    // 20260322 ZJH 构建三栏子控件
    QWidget* pLeft   = buildLeftPanel();    // 20260322 ZJH 左侧配置面板
    QWidget* pCenter = buildCenterPanel();  // 20260322 ZJH 中央统计视图
    QWidget* pRight  = buildRightPanel();   // 20260322 ZJH 右侧说明面板

    // 20260322 ZJH 设置三栏布局，左 280px，右 220px
    setLeftPanelWidth(280);
    setRightPanelWidth(220);
    setupThreeColumnLayout(pLeft, pCenter, pRight);
}

// ============================================================================
// 20260322 ZJH 构建左侧配置面板
// 包含：拆分配置分组框 + 快速预设分组框 + 操作按钮分组框
// ============================================================================
QWidget* SplitPage::buildLeftPanel()
{
    // 20260322 ZJH 左侧容器（可滚动，防止内容超出）
    QWidget* pContainer = new QWidget(this);
    pContainer->setMinimumWidth(200);
    pContainer->setMaximumWidth(320);

    QVBoxLayout* pMainLayout = new QVBoxLayout(pContainer);
    pMainLayout->setContentsMargins(8, 8, 8, 8);
    pMainLayout->setSpacing(8);

    // ========================
    // 20260322 ZJH 拆分配置分组框
    // ========================
    QGroupBox* pConfigGroup = new QGroupBox(QStringLiteral("拆分配置"), pContainer);
    pConfigGroup->setStyleSheet(s_strGroupBoxStyle);
    QVBoxLayout* pConfigLayout = new QVBoxLayout(pConfigGroup);
    pConfigLayout->setSpacing(8);
    pConfigLayout->setContentsMargins(8, 16, 8, 8);

    // 20260322 ZJH 拆分名称输入行
    QHBoxLayout* pNameRow = new QHBoxLayout();
    QLabel* pNameLabel = new QLabel(QStringLiteral("拆分名称"), pConfigGroup);
    pNameLabel->setStyleSheet(QStringLiteral("color: #94a3b8; font-size: 11px;"));
    m_pSplitNameEdit = new QLineEdit(QStringLiteral("default"), pConfigGroup);
    m_pSplitNameEdit->setStyleSheet(QStringLiteral(
        "QLineEdit { background:#1e2230; border:1px solid #2a2d35; border-radius:3px;"
        "            color:#e2e8f0; padding:3px 6px; font-size:11px; }"
        "QLineEdit:focus { border-color:#2563eb; }"
    ));
    pNameRow->addWidget(pNameLabel);
    pNameRow->addWidget(m_pSplitNameEdit);
    pConfigLayout->addLayout(pNameRow);

    // ------- 训练集滑块行 -------
    QLabel* pTrainTitleLabel = new QLabel(QStringLiteral("训练集"), pConfigGroup);
    pTrainTitleLabel->setStyleSheet(QStringLiteral("color: #60a5fa; font-size: 11px; font-weight:600;"));
    pConfigLayout->addWidget(pTrainTitleLabel);

    QHBoxLayout* pTrainRow = new QHBoxLayout();
    m_pTrainSlider = new QSlider(Qt::Horizontal, pConfigGroup);
    m_pTrainSlider->setRange(50, 95);   // 20260322 ZJH 训练集范围 50~95%
    m_pTrainSlider->setValue(80);       // 20260322 ZJH 默认 80%
    m_pTrainSlider->setStyleSheet(QStringLiteral(
        "QSlider::groove:horizontal { height:4px; background:#2a2d35; border-radius:2px; }"
        "QSlider::handle:horizontal { width:12px; height:12px; border-radius:6px;"
        "    background:#2563eb; margin:-4px 0; }"
        "QSlider::sub-page:horizontal { background:#2563eb; border-radius:2px; }"
    ));

    m_pTrainSpin = new QSpinBox(pConfigGroup);
    m_pTrainSpin->setRange(50, 95);     // 20260322 ZJH 同步范围
    m_pTrainSpin->setValue(80);
    m_pTrainSpin->setSuffix(QStringLiteral("%"));
    m_pTrainSpin->setFixedWidth(58);
    m_pTrainSpin->setStyleSheet(QStringLiteral(
        "QSpinBox { background:#1e2230; border:1px solid #2a2d35; border-radius:3px;"
        "           color:#e2e8f0; padding:2px 4px; font-size:11px; }"
        "QSpinBox:focus { border-color:#2563eb; }"
        "QSpinBox::up-button, QSpinBox::down-button { width:14px; background:#2a2d35; }"
    ));
    pTrainRow->addWidget(m_pTrainSlider, 1);
    pTrainRow->addWidget(m_pTrainSpin);
    pConfigLayout->addLayout(pTrainRow);

    // ------- 验证集滑块行 -------
    QLabel* pValTitleLabel = new QLabel(QStringLiteral("验证集"), pConfigGroup);
    pValTitleLabel->setStyleSheet(QStringLiteral("color: #f59e0b; font-size: 11px; font-weight:600;"));
    pConfigLayout->addWidget(pValTitleLabel);

    QHBoxLayout* pValRow = new QHBoxLayout();
    m_pValSlider = new QSlider(Qt::Horizontal, pConfigGroup);
    m_pValSlider->setRange(0, 30);      // 20260322 ZJH 验证集范围 0~30%
    m_pValSlider->setValue(10);         // 20260322 ZJH 默认 10%
    m_pValSlider->setStyleSheet(QStringLiteral(
        "QSlider::groove:horizontal { height:4px; background:#2a2d35; border-radius:2px; }"
        "QSlider::handle:horizontal { width:12px; height:12px; border-radius:6px;"
        "    background:#f59e0b; margin:-4px 0; }"
        "QSlider::sub-page:horizontal { background:#f59e0b; border-radius:2px; }"
    ));

    m_pValSpin = new QSpinBox(pConfigGroup);
    m_pValSpin->setRange(0, 30);
    m_pValSpin->setValue(10);
    m_pValSpin->setSuffix(QStringLiteral("%"));
    m_pValSpin->setFixedWidth(58);
    m_pValSpin->setStyleSheet(m_pTrainSpin->styleSheet());  // 20260322 ZJH 复用样式
    pValRow->addWidget(m_pValSlider, 1);
    pValRow->addWidget(m_pValSpin);
    pConfigLayout->addLayout(pValRow);

    // ------- 测试集显示行（只读计算值） -------
    QHBoxLayout* pTestRow = new QHBoxLayout();
    QLabel* pTestTitleLabel = new QLabel(QStringLiteral("测试集"), pConfigGroup);
    pTestTitleLabel->setStyleSheet(QStringLiteral("color: #10b981; font-size: 11px; font-weight:600;"));
    m_pTestLabel = new QLabel(QStringLiteral("10%"), pConfigGroup);
    m_pTestLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_pTestLabel->setStyleSheet(QStringLiteral("color: #10b981; font-size: 12px; font-weight:700;"));
    pTestRow->addWidget(pTestTitleLabel);
    pTestRow->addStretch();
    pTestRow->addWidget(m_pTestLabel);
    pConfigLayout->addLayout(pTestRow);

    // 20260322 ZJH 分层采样复选框
    m_pStratifiedCheck = new QCheckBox(QStringLiteral("分层采样（按标签均匀分配）"), pConfigGroup);
    m_pStratifiedCheck->setChecked(true);  // 20260322 ZJH 默认勾选
    m_pStratifiedCheck->setStyleSheet(QStringLiteral(
        "QCheckBox { color: #94a3b8; font-size: 11px; }"
        "QCheckBox::indicator { width:14px; height:14px; border:1px solid #2a2d35;"
        "    border-radius:2px; background:#1e2230; }"
        "QCheckBox::indicator:checked { background:#2563eb; border-color:#2563eb; }"
    ));
    pConfigLayout->addWidget(m_pStratifiedCheck);

    pMainLayout->addWidget(pConfigGroup);

    // ========================
    // 20260322 ZJH 快速预设分组框
    // ========================
    QGroupBox* pPresetGroup = new QGroupBox(QStringLiteral("快速预设"), pContainer);
    pPresetGroup->setStyleSheet(s_strGroupBoxStyle);
    QVBoxLayout* pPresetLayout = new QVBoxLayout(pPresetGroup);
    pPresetLayout->setSpacing(6);
    pPresetLayout->setContentsMargins(8, 16, 8, 8);

    // 20260322 ZJH 三个预设按钮（70/15/15 / 80/10/10 / 60/20/20）
    m_pPreset701515Btn = new QPushButton(QStringLiteral("70 / 15 / 15"), pPresetGroup);
    m_pPreset801010Btn = new QPushButton(QStringLiteral("80 / 10 / 10"), pPresetGroup);
    m_pPreset602020Btn = new QPushButton(QStringLiteral("60 / 20 / 20"), pPresetGroup);

    // 20260322 ZJH 统一设置预设按钮样式
    for (QPushButton* pBtn : {m_pPreset701515Btn, m_pPreset801010Btn, m_pPreset602020Btn}) {
        pBtn->setStyleSheet(s_strSecondaryBtnStyle);
    }

    pPresetLayout->addWidget(m_pPreset701515Btn);
    pPresetLayout->addWidget(m_pPreset801010Btn);
    pPresetLayout->addWidget(m_pPreset602020Btn);

    pMainLayout->addWidget(pPresetGroup);

    // ========================
    // 20260322 ZJH 操作按钮分组框
    // ========================
    QGroupBox* pActionGroup = new QGroupBox(QStringLiteral("操作"), pContainer);
    pActionGroup->setStyleSheet(s_strGroupBoxStyle);
    QVBoxLayout* pActionLayout = new QVBoxLayout(pActionGroup);
    pActionLayout->setSpacing(6);
    pActionLayout->setContentsMargins(8, 16, 8, 8);

    m_pExecuteBtn = new QPushButton(QStringLiteral("执行拆分"), pActionGroup);
    m_pExecuteBtn->setStyleSheet(s_strPrimaryBtnStyle);
    m_pExecuteBtn->setMinimumHeight(32);

    m_pResetBtn = new QPushButton(QStringLiteral("重置拆分"), pActionGroup);
    m_pResetBtn->setStyleSheet(s_strSecondaryBtnStyle);
    m_pResetBtn->setMinimumHeight(28);

    pActionLayout->addWidget(m_pExecuteBtn);
    pActionLayout->addWidget(m_pResetBtn);

    pMainLayout->addWidget(pActionGroup);

    // 20260322 ZJH 底部弹性空间，防止控件上移
    pMainLayout->addStretch();

    // ===== 连接信号槽 =====

    // 20260322 ZJH 训练集滑块 ↔ 微调框双向联动
    connect(m_pTrainSlider, &QSlider::valueChanged,
            this, &SplitPage::onTrainSliderChanged);
    connect(m_pTrainSpin, &QSpinBox::valueChanged,
            this, &SplitPage::onTrainSpinChanged);

    // 20260322 ZJH 验证集滑块 ↔ 微调框双向联动
    connect(m_pValSlider, &QSlider::valueChanged,
            this, &SplitPage::onValSliderChanged);
    connect(m_pValSpin, &QSpinBox::valueChanged,
            this, &SplitPage::onValSpinChanged);

    // 20260322 ZJH 预设按钮
    connect(m_pPreset701515Btn, &QPushButton::clicked, this, &SplitPage::onPreset701515);
    connect(m_pPreset801010Btn, &QPushButton::clicked, this, &SplitPage::onPreset801010);
    connect(m_pPreset602020Btn, &QPushButton::clicked, this, &SplitPage::onPreset602020);

    // 20260322 ZJH 操作按钮
    connect(m_pExecuteBtn, &QPushButton::clicked, this, &SplitPage::onExecuteSplit);
    connect(m_pResetBtn,   &QPushButton::clicked, this, &SplitPage::onResetSplit);

    return pContainer;
}

// ============================================================================
// 20260322 ZJH 构建中央内容区
// 包含：统计卡片行 + 标签分布表格 + 分布条形图
// ============================================================================
QWidget* SplitPage::buildCenterPanel()
{
    QWidget* pContainer = new QWidget(this);
    QVBoxLayout* pMainLayout = new QVBoxLayout(pContainer);
    pMainLayout->setContentsMargins(12, 12, 12, 12);
    pMainLayout->setSpacing(12);

    // ========================
    // 20260322 ZJH 三个统计卡片水平排列
    // ========================
    QHBoxLayout* pCardRow = new QHBoxLayout();
    pCardRow->setSpacing(12);

    // 20260322 ZJH 训练集卡片（蓝色）
    QFrame* pTrainCard = createStatCard(
        QStringLiteral("训练集"), QColor(0x26, 0x63, 0xeb),
        &m_pTrainCount, &m_pTrainPct);

    // 20260322 ZJH 验证集卡片（橙色）
    QFrame* pValCard = createStatCard(
        QStringLiteral("验证集"), QColor(0xf5, 0x9e, 0x0b),
        &m_pValCount, &m_pValPct);

    // 20260322 ZJH 测试集卡片（绿色）
    QFrame* pTestCard = createStatCard(
        QStringLiteral("测试集"), QColor(0x10, 0xb9, 0x81),
        &m_pTestCount, &m_pTestPct);

    pCardRow->addWidget(pTrainCard, 1);  // 20260322 ZJH stretch=1，三列均等
    pCardRow->addWidget(pValCard,   1);
    pCardRow->addWidget(pTestCard,  1);

    pMainLayout->addLayout(pCardRow);

    // ========================
    // 20260322 ZJH 标签分布表格
    // ========================
    QLabel* pTableTitle = new QLabel(QStringLiteral("标签分布明细"), pContainer);
    pTableTitle->setStyleSheet(QStringLiteral("color:#e2e8f0; font-size:12px; font-weight:600;"));
    pMainLayout->addWidget(pTableTitle);

    m_pDistTable = new QTableWidget(0, 5, pContainer);  // 20260322 ZJH 0行5列：标签|训练|验证|测试|总计
    m_pDistTable->setHorizontalHeaderLabels({
        QStringLiteral("标签名"),
        QStringLiteral("训练"),
        QStringLiteral("验证"),
        QStringLiteral("测试"),
        QStringLiteral("总计")
    });

    // 20260322 ZJH 表格样式（暗色主题一致）
    m_pDistTable->setStyleSheet(QStringLiteral(
        "QTableWidget {"
        "  background-color: #1e2230;"
        "  gridline-color: #2a2d35;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  font-size: 11px;"
        "}"
        "QTableWidget::item { padding: 4px 8px; }"
        "QTableWidget::item:selected { background-color: #2563eb; color:#fff; }"
        "QHeaderView::section {"
        "  background-color: #13151a;"
        "  color: #94a3b8;"
        "  padding: 4px 8px;"
        "  border: none;"
        "  border-bottom: 1px solid #2a2d35;"
        "  font-size: 11px;"
        "  font-weight: 600;"
        "}"
        "QScrollBar:vertical { background:#13151a; width:8px; }"
        "QScrollBar::handle:vertical { background:#2a2d35; border-radius:4px; min-height:20px; }"
    ));
    m_pDistTable->horizontalHeader()->setStretchLastSection(false);
    m_pDistTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);  // 20260322 ZJH 标签列拉伸
    for (int i = 1; i < 5; ++i) {
        m_pDistTable->horizontalHeader()->setSectionResizeMode(i, QHeaderView::Fixed);
        m_pDistTable->setColumnWidth(i, 64);  // 20260322 ZJH 数量列固定宽度
    }
    m_pDistTable->verticalHeader()->setVisible(false);  // 20260322 ZJH 隐藏行号
    m_pDistTable->setEditTriggers(QAbstractItemView::NoEditTriggers);  // 20260322 ZJH 不可编辑
    m_pDistTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_pDistTable->setAlternatingRowColors(true);
    m_pDistTable->setMinimumHeight(100);
    m_pDistTable->setMaximumHeight(200);

    pMainLayout->addWidget(m_pDistTable);

    // ========================
    // 20260322 ZJH 类别分布条形图
    // ========================
    QLabel* pChartTitle = new QLabel(QStringLiteral("类别分布图"), pContainer);
    pChartTitle->setStyleSheet(QStringLiteral("color:#e2e8f0; font-size:12px; font-weight:600;"));
    pMainLayout->addWidget(pChartTitle);

    m_pChart = new ClassDistributionChart(pContainer);
    m_pChart->setMinimumHeight(160);
    pMainLayout->addWidget(m_pChart, 1);  // 20260322 ZJH stretch=1，占满剩余空间

    return pContainer;
}

// ============================================================================
// 20260322 ZJH 构建右侧说明面板
// 包含：拆分建议文字 + 拆分状态标签
// ============================================================================
QWidget* SplitPage::buildRightPanel()
{
    QWidget* pContainer = new QWidget(this);
    pContainer->setMinimumWidth(180);
    pContainer->setMaximumWidth(280);

    QVBoxLayout* pMainLayout = new QVBoxLayout(pContainer);
    pMainLayout->setContentsMargins(8, 8, 8, 8);
    pMainLayout->setSpacing(8);

    // 20260322 ZJH 说明文字分组框
    QGroupBox* pHintGroup = new QGroupBox(QStringLiteral("使用说明"), pContainer);
    pHintGroup->setStyleSheet(s_strGroupBoxStyle);
    QVBoxLayout* pHintLayout = new QVBoxLayout(pHintGroup);
    pHintLayout->setContentsMargins(8, 16, 8, 8);

    QLabel* pHintLabel = new QLabel(pHintGroup);
    pHintLabel->setWordWrap(true);       // 20260322 ZJH 自动换行
    pHintLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    pHintLabel->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 11px; line-height: 1.6; background:transparent; border:none; }"
    ));
    pHintLabel->setText(QStringLiteral(
        "数据集拆分建议：\n\n"
        "• 训练集（70~80%）：\n"
        "  用于模型参数学习，越多越好。\n\n"
        "• 验证集（10~15%）：\n"
        "  用于超参数调节和过拟合监控。\n\n"
        "• 测试集（10~20%）：\n"
        "  最终性能评估，执行拆分后不可更改。\n\n"
        "• 分层采样：\n"
        "  保持各类别在三个子集中比例一致，"
        "推荐在类别不均衡时启用。"
    ));
    pHintLayout->addWidget(pHintLabel);
    pMainLayout->addWidget(pHintGroup);

    // 20260322 ZJH 拆分状态分组框
    QGroupBox* pStatusGroup = new QGroupBox(QStringLiteral("拆分状态"), pContainer);
    pStatusGroup->setStyleSheet(s_strGroupBoxStyle);
    QVBoxLayout* pStatusLayout = new QVBoxLayout(pStatusGroup);
    pStatusLayout->setContentsMargins(8, 16, 8, 8);

    m_pSplitStatusLabel = new QLabel(QStringLiteral("未拆分"), pStatusGroup);
    m_pSplitStatusLabel->setAlignment(Qt::AlignCenter);
    m_pSplitStatusLabel->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #94a3b8;"
        "  font-size: 13px;"
        "  font-weight: 600;"
        "  background: #13151a;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 8px 4px;"
        "}"
    ));
    pStatusLayout->addWidget(m_pSplitStatusLabel);
    pMainLayout->addWidget(pStatusGroup);

    // 20260322 ZJH 底部弹性空间
    pMainLayout->addStretch();

    return pContainer;
}

// ============================================================================
// 20260322 ZJH 创建统计卡片
// 参数: strTitle — 卡片标题（训练集/验证集/测试集）
//       color    — 主题颜色（标题/大数字颜色）
//       ppCount  — 输出：数量 QLabel 指针
//       ppPct    — 输出：百分比 QLabel 指针
// ============================================================================
QFrame* SplitPage::createStatCard(const QString& strTitle, const QColor& color,
                                   QLabel** ppCount, QLabel** ppPct)
{
    // 20260322 ZJH 卡片容器 QFrame，圆角边框
    QFrame* pCard = new QFrame(this);
    pCard->setFrameShape(QFrame::StyledPanel);
    pCard->setStyleSheet(QStringLiteral(
        "QFrame {"
        "  background-color: #1e2230;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 6px;"
        "  padding: 4px;"
        "}"
    ));
    pCard->setMinimumHeight(80);

    QVBoxLayout* pCardLayout = new QVBoxLayout(pCard);
    pCardLayout->setContentsMargins(10, 8, 10, 8);
    pCardLayout->setSpacing(4);

    // 20260322 ZJH 卡片标题（彩色）
    QLabel* pTitleLbl = new QLabel(strTitle, pCard);
    pTitleLbl->setAlignment(Qt::AlignCenter);
    pTitleLbl->setStyleSheet(QString(
        "QLabel { color: %1; font-size: 11px; font-weight: 600; background:transparent; border:none; }"
    ).arg(color.name()));
    pCardLayout->addWidget(pTitleLbl);

    // 20260322 ZJH 数量大字（彩色）
    *ppCount = new QLabel(QStringLiteral("0"), pCard);
    (*ppCount)->setAlignment(Qt::AlignCenter);
    (*ppCount)->setStyleSheet(QString(
        "QLabel { color: %1; font-size: 22px; font-weight: 700; background:transparent; border:none; }"
    ).arg(color.name()));
    pCardLayout->addWidget(*ppCount);

    // 20260322 ZJH 百分比小字（灰色）
    *ppPct = new QLabel(QStringLiteral("0.0%"), pCard);
    (*ppPct)->setAlignment(Qt::AlignCenter);
    (*ppPct)->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 11px; background:transparent; border:none; }"
    ));
    pCardLayout->addWidget(*ppPct);

    return pCard;
}

// ============================================================================
// 20260322 ZJH 生命周期回调 — 页面切换到前台
// ============================================================================
void SplitPage::onEnter()
{
    // 20260322 ZJH 每次进入页面时刷新统计（项目数据可能已变更）
    refreshStats();
}

// ============================================================================
// 20260322 ZJH 生命周期回调 — 项目加载完成
// ============================================================================
// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void SplitPage::onProjectLoadedImpl()
{
    refreshStats();          // 20260322 ZJH 读取当前拆分状态并刷新 UI
}

// ============================================================================
// 20260322 ZJH 生命周期回调 — 项目关闭
// ============================================================================
// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void SplitPage::onProjectClosedImpl()
{
    // 20260322 ZJH 重置统计卡片为 0
    for (QLabel* pLbl : {m_pTrainCount, m_pValCount, m_pTestCount}) {
        if (pLbl) pLbl->setText(QStringLiteral("0"));
    }
    for (QLabel* pLbl : {m_pTrainPct, m_pValPct, m_pTestPct}) {
        if (pLbl) pLbl->setText(QStringLiteral("0.0%"));
    }

    // 20260322 ZJH 清空表格和图表
    if (m_pDistTable) m_pDistTable->setRowCount(0);
    if (m_pChart)     m_pChart->clear();

    // 20260322 ZJH 重置状态标签
    if (m_pSplitStatusLabel) {
        m_pSplitStatusLabel->setText(QStringLiteral("未拆分"));
        m_pSplitStatusLabel->setStyleSheet(QStringLiteral(
            "QLabel { color:#94a3b8; font-size:13px; font-weight:600;"
            "         background:#13151a; border:1px solid #2a2d35; border-radius:4px; padding:8px 4px; }"
        ));
    }
}

// ============================================================================
// 20260322 ZJH 刷新统计显示（统计卡片 + 分布表格 + 条形图）
// ============================================================================
void SplitPage::refreshStats()
{
    // 20260322 ZJH 无项目时清空显示
    if (!m_pProject) {
        return;
    }

    // 20260322 ZJH 获取数据集引用
    ImageDataset* pDataset = m_pProject->dataset();
    if (!pDataset) {
        return;
    }

    // 20260322 ZJH 各拆分类型的图像数量
    int nTrain  = pDataset->countBySplit(om::SplitType::Train);
    int nVal    = pDataset->countBySplit(om::SplitType::Validation);
    int nTest   = pDataset->countBySplit(om::SplitType::Test);
    int nTotal  = pDataset->imageCount();  // 20260322 ZJH 全部图像总数

    // 20260322 ZJH 计算百分比（避免除零）
    double dTotalD = nTotal > 0 ? static_cast<double>(nTotal) : 1.0;
    double dTrainPct = nTrain / dTotalD * 100.0;
    double dValPct   = nVal   / dTotalD * 100.0;
    double dTestPct  = nTest  / dTotalD * 100.0;

    // 20260322 ZJH 更新统计卡片
    m_pTrainCount->setText(QString::number(nTrain));
    m_pValCount->setText(QString::number(nVal));
    m_pTestCount->setText(QString::number(nTest));
    m_pTrainPct->setText(QStringLiteral("%1%").arg(dTrainPct, 0, 'f', 1));
    m_pValPct->setText(QStringLiteral("%1%").arg(dValPct,   0, 'f', 1));
    m_pTestPct->setText(QStringLiteral("%1%").arg(dTestPct, 0, 'f', 1));

    // ========================
    // 20260322 ZJH 更新标签分布表格
    // ========================
    const QVector<LabelInfo>& vecLabels = pDataset->labels();
    const QVector<ImageEntry>& vecImages = pDataset->images();

    // 20260322 ZJH 按标签 ID 统计各拆分类型的数量
    // 外层 key: labelId, 内层 key: SplitType(0/1/2)
    QMap<int, int> mapTrainCnt, mapValCnt, mapTestCnt, mapTotalCnt;

    // 20260322 ZJH 初始化所有标签计数为 0
    for (const LabelInfo& label : vecLabels) {
        mapTrainCnt[label.nId] = 0;
        mapValCnt[label.nId]   = 0;
        mapTestCnt[label.nId]  = 0;
        mapTotalCnt[label.nId] = 0;
    }

    // 20260405 ZJH [修复] 智能计数策略（区分两种任务类型）:
    //   语义分割/实例分割/目标检测: 按标注实例计数（一张图 3 个标注 = 3 次）
    //   分类/异常检测: 按图像计数（一张图 = 1 次）
    // 语义分割每张图可有多类标注（异物+划伤+脏污），必须按标注计数才能与图库页一致
    bool bCountPerAnnotation = false;  // 20260405 ZJH 默认按图像计数
    if (m_pProject) {
        auto eTask = m_pProject->taskType();
        bCountPerAnnotation = (eTask == om::TaskType::SemanticSegmentation ||
                               eTask == om::TaskType::InstanceSegmentation ||
                               eTask == om::TaskType::ObjectDetection);
    }

    for (const ImageEntry& entry : vecImages) {
        if (bCountPerAnnotation && !entry.vecAnnotations.isEmpty()) {
            // 20260402 ZJH 实例分割/检测: 按每个标注独立计数
            for (const auto& annotation : entry.vecAnnotations) {
                int nLabelId = annotation.nLabelId;
                if (nLabelId < 0 || !mapTotalCnt.contains(nLabelId)) continue;

                switch (entry.eSplit) {
                    case om::SplitType::Train:      ++mapTrainCnt[nLabelId]; break;
                    case om::SplitType::Validation:  ++mapValCnt[nLabelId];  break;
                    case om::SplitType::Test:        ++mapTestCnt[nLabelId]; break;
                    default: break;
                }
                ++mapTotalCnt[nLabelId];
            }
        } else if (!bCountPerAnnotation && !entry.vecAnnotations.isEmpty()) {
            // 20260402 ZJH 语义分割: 有标注但按图像计数（取第一个标注的标签）
            int nLabelId = entry.vecAnnotations.first().nLabelId;
            if (nLabelId < 0 || !mapTotalCnt.contains(nLabelId)) {
                // 20260402 ZJH fallback 到图像级标签
                nLabelId = entry.nLabelId;
                if (nLabelId < 0 || !mapTotalCnt.contains(nLabelId)) continue;
            }

            switch (entry.eSplit) {
                case om::SplitType::Train:      ++mapTrainCnt[nLabelId]; break;
                case om::SplitType::Validation:  ++mapValCnt[nLabelId];  break;
                case om::SplitType::Test:        ++mapTestCnt[nLabelId]; break;
                default: break;
            }
            ++mapTotalCnt[nLabelId];
        } else if (entry.nLabelId >= 0) {
            // 20260402 ZJH 无标注、仅图像级标签 → 按图像计数（分类/异常检测任务）
            int nLabelId = entry.nLabelId;
            if (!mapTotalCnt.contains(nLabelId)) continue;

            switch (entry.eSplit) {
                case om::SplitType::Train:      ++mapTrainCnt[nLabelId]; break;
                case om::SplitType::Validation:  ++mapValCnt[nLabelId];  break;
                case om::SplitType::Test:        ++mapTestCnt[nLabelId]; break;
                default: break;
            }
            ++mapTotalCnt[nLabelId];
        }
        // 20260402 ZJH 未标注图像（nLabelId<0 且无标注）不参与标签统计
    }

    // 20260322 ZJH 填充表格行
    m_pDistTable->setRowCount(vecLabels.size());
    QMap<QString, int> mapChartData;  // 20260322 ZJH 同步准备条形图数据（标签名 -> 总数）

    for (int nRow = 0; nRow < vecLabels.size(); ++nRow) {
        const LabelInfo& label = vecLabels[nRow];
        int nLblTrain = mapTrainCnt.value(label.nId, 0);
        int nLblVal   = mapValCnt.value(label.nId, 0);
        int nLblTest  = mapTestCnt.value(label.nId, 0);
        int nLblTotal = mapTotalCnt.value(label.nId, 0);

        // 20260322 ZJH 标签名列（带颜色圆点，用 HTML 富文本实现）
        QTableWidgetItem* pNameItem = new QTableWidgetItem(label.strName);
        pNameItem->setForeground(label.color);  // 20260322 ZJH 文字颜色与标签颜色一致
        m_pDistTable->setItem(nRow, 0, pNameItem);

        // 20260322 ZJH 训练/验证/测试/总计数量列
        m_pDistTable->setItem(nRow, 1, new QTableWidgetItem(QString::number(nLblTrain)));
        m_pDistTable->setItem(nRow, 2, new QTableWidgetItem(QString::number(nLblVal)));
        m_pDistTable->setItem(nRow, 3, new QTableWidgetItem(QString::number(nLblTest)));
        m_pDistTable->setItem(nRow, 4, new QTableWidgetItem(QString::number(nLblTotal)));

        // 20260322 ZJH 数量列居中对齐
        for (int nCol = 1; nCol < 5; ++nCol) {
            if (m_pDistTable->item(nRow, nCol)) {
                m_pDistTable->item(nRow, nCol)->setTextAlignment(Qt::AlignCenter);
            }
        }

        // 20260322 ZJH 准备条形图数据
        mapChartData[label.strName] = nLblTotal;
    }

    // 20260322 ZJH 刷新条形图
    m_pChart->updateData(mapChartData);

    // ========================
    // 20260322 ZJH 更新拆分状态标签
    // ========================
    bool bSplitDone = (nTrain + nVal + nTest) > 0;  // 20260322 ZJH 至少有一张已分配
    if (bSplitDone) {
        m_pSplitStatusLabel->setText(QStringLiteral("已拆分"));
        m_pSplitStatusLabel->setStyleSheet(QStringLiteral(
            "QLabel { color:#10b981; font-size:13px; font-weight:600;"
            "         background:#13151a; border:1px solid #10b981; border-radius:4px; padding:8px 4px; }"
        ));
    } else {
        m_pSplitStatusLabel->setText(QStringLiteral("未拆分"));
        m_pSplitStatusLabel->setStyleSheet(QStringLiteral(
            "QLabel { color:#94a3b8; font-size:13px; font-weight:600;"
            "         background:#13151a; border:1px solid #2a2d35; border-radius:4px; padding:8px 4px; }"
        ));
    }
}

// ============================================================================
// 20260322 ZJH 更新测试集显示（= 100 - 训练集 - 验证集）
// ============================================================================
void SplitPage::updateTestLabel()
{
    int nTest = 100 - m_pTrainSpin->value() - m_pValSpin->value();
    // 20260322 ZJH 限制测试集不为负数（极端情况下 train+val > 100）
    if (nTest < 0) nTest = 0;
    m_pTestLabel->setText(QStringLiteral("%1%").arg(nTest));
}

// ============================================================================
// 20260322 ZJH 滑块 ↔ 微调框联动槽函数（防递归标志 m_bUpdating）
// ============================================================================

// 20260322 ZJH 训练集滑块变化 → 同步微调框
void SplitPage::onTrainSliderChanged(int nValue)
{
    if (m_bUpdating) return;  // 20260322 ZJH 防止递归触发
    m_bUpdating = true;
    m_pTrainSpin->setValue(nValue);  // 20260322 ZJH 同步微调框
    m_bUpdating = false;
    updateTestLabel();               // 20260322 ZJH 更新测试集显示
}

// 20260322 ZJH 训练集微调框变化 → 同步滑块
void SplitPage::onTrainSpinChanged(int nValue)
{
    if (m_bUpdating) return;
    m_bUpdating = true;
    m_pTrainSlider->setValue(nValue);  // 20260322 ZJH 同步滑块
    m_bUpdating = false;
    updateTestLabel();
}

// 20260322 ZJH 验证集滑块变化 → 同步微调框
void SplitPage::onValSliderChanged(int nValue)
{
    if (m_bUpdating) return;
    m_bUpdating = true;
    m_pValSpin->setValue(nValue);
    m_bUpdating = false;
    updateTestLabel();
}

// 20260322 ZJH 验证集微调框变化 → 同步滑块
void SplitPage::onValSpinChanged(int nValue)
{
    if (m_bUpdating) return;
    m_bUpdating = true;
    m_pValSlider->setValue(nValue);
    m_bUpdating = false;
    updateTestLabel();
}

// ============================================================================
// 20260322 ZJH 预设比例槽函数
// ============================================================================

// 20260322 ZJH 应用预设（抑制信号防止多次 updateTestLabel 调用）
void SplitPage::applyPreset(int nTrain, int nVal)
{
    m_bUpdating = true;  // 20260322 ZJH 批量设置期间抑制联动信号
    m_pTrainSlider->setValue(nTrain);
    m_pTrainSpin->setValue(nTrain);
    m_pValSlider->setValue(nVal);
    m_pValSpin->setValue(nVal);
    m_bUpdating = false;
    updateTestLabel();   // 20260322 ZJH 设置完成后统一更新测试集显示
}

void SplitPage::onPreset701515()
{
    applyPreset(70, 15);  // 20260322 ZJH 70/15/15 预设
}

void SplitPage::onPreset801010()
{
    applyPreset(80, 10);  // 20260322 ZJH 80/10/10 预设
}

void SplitPage::onPreset602020()
{
    applyPreset(60, 20);  // 20260322 ZJH 60/20/20 预设
}

// ============================================================================
// 20260322 ZJH 执行拆分
// 读取 UI 配置，调用 ImageDataset::autoSplit，刷新统计
// ============================================================================
void SplitPage::onExecuteSplit()
{
    // 20260322 ZJH 检查项目有效性
    if (!m_pProject) {
        QMessageBox::warning(this,
            QStringLiteral("执行拆分"),
            QStringLiteral("请先打开或创建一个项目。"));
        return;
    }

    ImageDataset* pDataset = m_pProject->dataset();
    if (!pDataset || pDataset->imageCount() == 0) {
        QMessageBox::warning(this,
            QStringLiteral("执行拆分"),
            QStringLiteral("数据集为空，请先导入图像。"));
        return;
    }

    // 20260322 ZJH 从 UI 读取比例参数
    int nTrainPct = m_pTrainSpin->value();   // 20260322 ZJH 训练集百分比整数
    int nValPct   = m_pValSpin->value();     // 20260322 ZJH 验证集百分比整数
    bool bStratified = m_pStratifiedCheck->isChecked();  // 20260322 ZJH 是否分层采样

    // 20260322 ZJH 转换为 0.0~1.0 浮点比例
    float fTrainRatio = static_cast<float>(nTrainPct) / 100.0f;
    float fValRatio   = static_cast<float>(nValPct)   / 100.0f;

    // 20260322 ZJH 调用 ImageDataset::autoSplit 执行实际拆分
    pDataset->autoSplit(fTrainRatio, fValRatio, bStratified);

    // 20260322 ZJH 拆分完成后刷新统计显示
    refreshStats();

    // 20260322 ZJH 提示用户拆分完成
    int nTotal = pDataset->imageCount();
    QMessageBox::information(this,
        QStringLiteral("拆分完成"),
        QStringLiteral("数据集拆分完成。\n\n"
                        "训练集：%1 张\n"
                        "验证集：%2 张\n"
                        "测试集：%3 张\n"
                        "总计：%4 张")
            .arg(pDataset->countBySplit(om::SplitType::Train))
            .arg(pDataset->countBySplit(om::SplitType::Validation))
            .arg(pDataset->countBySplit(om::SplitType::Test))
            .arg(nTotal));
}

// ============================================================================
// 20260322 ZJH 重置拆分 — 将所有图像设置为 Unassigned，刷新统计
// ============================================================================
void SplitPage::onResetSplit()
{
    // 20260322 ZJH 检查项目有效性
    if (!m_pProject) {
        return;
    }

    ImageDataset* pDataset = m_pProject->dataset();
    if (!pDataset) {
        return;
    }

    // 20260322 ZJH 确认操作（防止误操作）
    int nRet = QMessageBox::question(this,
        QStringLiteral("重置拆分"),
        QStringLiteral("确定要重置所有图像的拆分状态吗？\n此操作将清除所有训练/验证/测试分配。"),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (nRet != QMessageBox::Yes) {
        return;  // 20260322 ZJH 用户取消，不执行重置
    }

    // 20260322 ZJH 遍历所有图像，将拆分类型设置为 Unassigned
    const QVector<ImageEntry>& vecImages = pDataset->images();
    for (const ImageEntry& entry : vecImages) {
        pDataset->assignSplit(entry.strUuid, om::SplitType::Unassigned);  // 20260322 ZJH 使用公共成员 strUuid
    }

    // 20260322 ZJH 刷新统计显示
    refreshStats();
}
