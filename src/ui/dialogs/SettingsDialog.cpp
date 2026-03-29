// 20260322 ZJH SettingsDialog 实现
// 三标签页设置对话框：常规/外观/GPU

#include "ui/dialogs/SettingsDialog.h"  // 20260322 ZJH 类声明
#include "app/ThemeManager.h"           // 20260322 ZJH 主题管理器
#include "app/Application.h"            // 20260322 ZJH 全局事件总线

#include <QVBoxLayout>       // 20260322 ZJH 垂直布局
#include <QFormLayout>       // 20260322 ZJH 表单布局
#include <QGroupBox>         // 20260322 ZJH 分组框
#include <QMessageBox>       // 20260322 ZJH 消息对话框
#include <QPushButton>       // 20260322 ZJH 按钮（QDialogButtonBox 内部使用）

// 20260322 ZJH 构造函数
SettingsDialog::SettingsDialog(QWidget* pParent)
    : QDialog(pParent)
    , m_pCmbLanguage(nullptr)
    , m_pSpnAutoSave(nullptr)
    , m_pSpnRecentProjects(nullptr)
    , m_pCmbTheme(nullptr)
    , m_pSpnFontSize(nullptr)
    , m_pLblGpuInfo(nullptr)
    , m_pCmbDevice(nullptr)
    , m_pButtonBox(nullptr)
{
    // 20260322 ZJH 设置对话框标题和大小
    setWindowTitle(QStringLiteral("设置"));
    setMinimumSize(480, 400);
    resize(520, 440);

    // 20260322 ZJH 暗色主题样式
    setStyleSheet(QStringLiteral(
        "QDialog {"
        "  background-color: #1a1d24;"
        "  color: #e2e8f0;"
        "}"
        "QTabWidget::pane {"
        "  border: 1px solid #2a2d35;"
        "  background-color: #1a1d24;"
        "}"
        "QTabBar::tab {"
        "  background-color: #13151a;"
        "  color: #94a3b8;"
        "  padding: 8px 20px;"
        "  border: 1px solid #2a2d35;"
        "  border-bottom: none;"
        "  border-top-left-radius: 4px;"
        "  border-top-right-radius: 4px;"
        "  margin-right: 2px;"
        "}"
        "QTabBar::tab:selected {"
        "  background-color: #1a1d24;"
        "  color: #e2e8f0;"
        "  border-bottom: 2px solid #2563eb;"
        "}"
        "QTabBar::tab:hover {"
        "  color: #e2e8f0;"
        "}"
        "QGroupBox {"
        "  font-weight: bold;"
        "  color: #94a3b8;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  margin-top: 12px;"
        "  padding-top: 16px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px;"
        "}"
        "QLabel {"
        "  color: #94a3b8;"
        "  border: none;"
        "}"
        "QComboBox {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 4px 8px;"
        "  min-height: 24px;"
        "}"
        "QComboBox::drop-down {"
        "  border: none;"
        "  width: 20px;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  selection-background-color: #2563eb;"
        "  border: 1px solid #2a2d35;"
        "}"
        "QSpinBox {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 4px 8px;"
        "  min-height: 24px;"
        "}"
        "QPushButton {"
        "  background-color: #2a2d35;"
        "  color: #e2e8f0;"
        "  border: 1px solid #3a3d45;"
        "  border-radius: 4px;"
        "  padding: 6px 16px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3a3d45;"
        "}"
    ));

    // 20260322 ZJH 主布局
    QVBoxLayout* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(12, 12, 12, 12);
    pMainLayout->setSpacing(12);

    // 20260322 ZJH 创建标签页容器
    QTabWidget* pTabWidget = new QTabWidget(this);
    pTabWidget->addTab(createGeneralTab(),    QStringLiteral("常规"));
    pTabWidget->addTab(createAppearanceTab(), QStringLiteral("外观"));
    pTabWidget->addTab(createGpuTab(),        QStringLiteral("GPU"));
    pMainLayout->addWidget(pTabWidget, 1);

    // 20260322 ZJH 创建按钮组（OK/Cancel/Apply）
    m_pButtonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply,
        this);
    pMainLayout->addWidget(m_pButtonBox);

    // 20260322 ZJH 连接按钮信号
    connect(m_pButtonBox, &QDialogButtonBox::accepted,
            this, &SettingsDialog::onAccept);
    connect(m_pButtonBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);
    connect(m_pButtonBox->button(QDialogButtonBox::Apply), &QPushButton::clicked,
            this, &SettingsDialog::onApply);
}

// 20260322 ZJH 创建 "常规" 标签页
QWidget* SettingsDialog::createGeneralTab()
{
    QWidget* pTab = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pTab);
    pLayout->setContentsMargins(12, 12, 12, 12);
    pLayout->setSpacing(12);

    // 20260322 ZJH 表单布局
    QFormLayout* pForm = new QFormLayout();
    pForm->setSpacing(12);
    pForm->setLabelAlignment(Qt::AlignRight);

    // 20260322 ZJH 语言选择
    m_pCmbLanguage = new QComboBox(pTab);
    m_pCmbLanguage->addItem(QStringLiteral("简体中文"), QStringLiteral("zh_CN"));
    m_pCmbLanguage->addItem(QStringLiteral("English"),  QStringLiteral("en_US"));
    pForm->addRow(QStringLiteral("语言:"), m_pCmbLanguage);

    // 20260322 ZJH 自动保存间隔（分钟）
    m_pSpnAutoSave = new QSpinBox(pTab);
    m_pSpnAutoSave->setRange(1, 60);    // 20260322 ZJH 1~60 分钟
    m_pSpnAutoSave->setValue(5);         // 20260322 ZJH 默认 5 分钟
    m_pSpnAutoSave->setSuffix(QStringLiteral(" 分钟"));
    pForm->addRow(QStringLiteral("自动保存间隔:"), m_pSpnAutoSave);

    // 20260322 ZJH 最近项目数
    m_pSpnRecentProjects = new QSpinBox(pTab);
    m_pSpnRecentProjects->setRange(1, 20);   // 20260322 ZJH 1~20 个
    m_pSpnRecentProjects->setValue(10);       // 20260322 ZJH 默认 10 个
    m_pSpnRecentProjects->setSuffix(QStringLiteral(" 个"));
    pForm->addRow(QStringLiteral("最近项目数:"), m_pSpnRecentProjects);

    pLayout->addLayout(pForm);
    pLayout->addStretch(1);  // 20260322 ZJH 弹性空间

    return pTab;
}

// 20260322 ZJH 创建 "外观" 标签页
QWidget* SettingsDialog::createAppearanceTab()
{
    QWidget* pTab = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pTab);
    pLayout->setContentsMargins(12, 12, 12, 12);
    pLayout->setSpacing(12);

    // 20260322 ZJH 表单布局
    QFormLayout* pForm = new QFormLayout();
    pForm->setSpacing(12);
    pForm->setLabelAlignment(Qt::AlignRight);

    // 20260322 ZJH 主题选择
    m_pCmbTheme = new QComboBox(pTab);
    m_pCmbTheme->addItem(QStringLiteral("暗色主题"), static_cast<int>(OmniMatch::ThemeManager::Theme::Dark));
    m_pCmbTheme->addItem(QStringLiteral("亮色主题"), static_cast<int>(OmniMatch::ThemeManager::Theme::Light));

    // 20260322 ZJH 初始化为当前主题
    auto* pThemeMgr = OmniMatch::ThemeManager::instance();
    if (pThemeMgr->currentTheme() == OmniMatch::ThemeManager::Theme::Light) {
        m_pCmbTheme->setCurrentIndex(1);  // 20260322 ZJH 亮色主题对应索引 1
    } else {
        m_pCmbTheme->setCurrentIndex(0);  // 20260322 ZJH 暗色主题对应索引 0
    }
    pForm->addRow(QStringLiteral("主题:"), m_pCmbTheme);

    // 20260322 ZJH 字体大小
    m_pSpnFontSize = new QSpinBox(pTab);
    m_pSpnFontSize->setRange(8, 24);     // 20260322 ZJH 8~24 pt
    m_pSpnFontSize->setValue(12);        // 20260322 ZJH 默认 12pt
    m_pSpnFontSize->setSuffix(QStringLiteral(" pt"));
    pForm->addRow(QStringLiteral("字体大小:"), m_pSpnFontSize);

    pLayout->addLayout(pForm);
    pLayout->addStretch(1);  // 20260322 ZJH 弹性空间

    return pTab;
}

// 20260322 ZJH 创建 "GPU" 标签页
QWidget* SettingsDialog::createGpuTab()
{
    QWidget* pTab = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pTab);
    pLayout->setContentsMargins(12, 12, 12, 12);
    pLayout->setSpacing(12);

    // 20260322 ZJH GPU 信息分组
    QGroupBox* pGpuGroup = new QGroupBox(QStringLiteral("GPU 信息"), pTab);
    QVBoxLayout* pGpuLayout = new QVBoxLayout(pGpuGroup);

    // 20260322 ZJH 显示检测到的 GPU 信息
    m_pLblGpuInfo = new QLabel(pGpuGroup);
    m_pLblGpuInfo->setWordWrap(true);  // 20260322 ZJH 自动换行

    auto* pApp = Application::instance();
    if (pApp->hasGpu()) {
        // 20260322 ZJH 显示 GPU 名称和显存
        m_pLblGpuInfo->setText(QStringLiteral(
            "检测到 GPU:\n\n"
            "  名称: %1\n"
            "  显存: %2 MB\n"
            "  状态: 可用")
            .arg(pApp->gpuName())
            .arg(pApp->gpuVramMB()));
        m_pLblGpuInfo->setStyleSheet(QStringLiteral(
            "QLabel { color: #22c55e; font-size: 12px; border: none; }"));
    } else {
        // 20260322 ZJH 未检测到 GPU
        m_pLblGpuInfo->setText(QStringLiteral(
            "未检测到 NVIDIA GPU\n\n"
            "将使用 CPU 进行训练和推理。\n"
            "安装 NVIDIA 驱动可启用 GPU 加速。"));
        m_pLblGpuInfo->setStyleSheet(QStringLiteral(
            "QLabel { color: #f59e0b; font-size: 12px; border: none; }"));
    }
    pGpuLayout->addWidget(m_pLblGpuInfo);
    pLayout->addWidget(pGpuGroup);

    // 20260322 ZJH 设备选择
    QFormLayout* pForm = new QFormLayout();
    pForm->setSpacing(12);
    pForm->setLabelAlignment(Qt::AlignRight);

    m_pCmbDevice = new QComboBox(pTab);
    m_pCmbDevice->addItem(QStringLiteral("CPU"), 0);
    if (pApp->hasGpu()) {
        m_pCmbDevice->addItem(QStringLiteral("GPU (CUDA)"), 1);
        m_pCmbDevice->setCurrentIndex(1);  // 20260322 ZJH 有 GPU 时默认选择 GPU
    }
    pForm->addRow(QStringLiteral("训练设备:"), m_pCmbDevice);

    pLayout->addLayout(pForm);
    pLayout->addStretch(1);  // 20260322 ZJH 弹性空间

    return pTab;
}

// 20260322 ZJH "应用" 按钮点击
void SettingsDialog::onApply()
{
    applySettings();  // 20260322 ZJH 应用设置
}

// 20260322 ZJH "确定" 按钮点击
void SettingsDialog::onAccept()
{
    applySettings();  // 20260322 ZJH 先应用设置
    accept();          // 20260322 ZJH 关闭对话框
}

// 20260322 ZJH 应用当前设置到系统
void SettingsDialog::applySettings()
{
    // 20260322 ZJH 1. 应用主题
    if (m_pCmbTheme) {
        int nThemeValue = m_pCmbTheme->currentData().toInt();
        auto eTheme = static_cast<OmniMatch::ThemeManager::Theme>(nThemeValue);
        auto* pThemeMgr = OmniMatch::ThemeManager::instance();
        if (pThemeMgr->currentTheme() != eTheme) {
            pThemeMgr->applyTheme(eTheme);  // 20260322 ZJH 切换主题
        }
    }

    // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
    Application::instance()->notifyGlobalSettingsChanged();
}
