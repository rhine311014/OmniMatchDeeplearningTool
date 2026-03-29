// 20260322 ZJH SplashScreen 实现
// 无边框深色启动画面，展示品牌信息和加载进度
// 完成后通过 QPropertyAnimation 渐隐 windowOpacity 过渡到主窗口

#include "ui/widgets/SplashScreen.h"  // 20260322 ZJH SplashScreen 类声明

#include <QVBoxLayout>         // 20260322 ZJH 垂直布局
#include <QPropertyAnimation>  // 20260322 ZJH 渐隐动画
#include <QScreen>             // 20260322 ZJH 获取屏幕尺寸用于居中
#include <QGuiApplication>     // 20260322 ZJH 获取主屏幕

// 20260322 ZJH 启动画面固定尺寸常量
static constexpr int s_nSplashWidth  = 480;  // 20260322 ZJH 启动画面宽度（像素）
static constexpr int s_nSplashHeight = 320;  // 20260322 ZJH 启动画面高度（像素）

// 20260322 ZJH 构造函数，创建无边框置顶窗口及全部子控件
SplashScreen::SplashScreen(QWidget* pParent)
    : QWidget(pParent)
{
    // 20260322 ZJH 设置窗口标志：无边框 + 始终置顶，启动画面不需要标题栏
    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);

    // 20260322 ZJH 启用透明度属性，使 setWindowOpacity() 生效用于渐隐动画
    setAttribute(Qt::WA_TranslucentBackground, false);

    // 20260322 ZJH 固定窗口大小 480x320，不允许用户拖拽调整
    setFixedSize(s_nSplashWidth, s_nSplashHeight);

    // 20260322 ZJH 设置深色背景和圆角边框样式
    setStyleSheet(QStringLiteral(
        "SplashScreen {"
        "  background-color: #1a1d23;"           // 深色背景
        "  border: 1px solid #2a2d35;"           // 微弱边框，增加层次感
        "  border-radius: 8px;"                  // 圆角
        "}"
    ));

    // 20260322 ZJH 创建垂直布局，子控件垂直排列居中
    QVBoxLayout* pLayout = new QVBoxLayout(this);
    pLayout->setContentsMargins(40, 40, 40, 30);  // 20260322 ZJH 四周内边距
    pLayout->setSpacing(8);                         // 20260322 ZJH 控件间距

    // 20260322 ZJH 顶部弹性空间，将标题推到垂直居中偏上的位置
    pLayout->addStretch(2);

    // 20260328 ZJH "OmniMatch" 大标题 — 白色 32pt 粗体
    m_pLblTitle = new QLabel(QStringLiteral("OmniMatch"), this);
    m_pLblTitle->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 水平居中
    m_pLblTitle->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #ffffff;"         // 白色文字
        "  font-size: 32pt;"        // 32 磅大字
        "  font-weight: bold;"      // 粗体
        "  background: transparent;"  // 透明背景，不遮挡父控件
        "  border: none;"           // 无边框
        "}"
    ));
    pLayout->addWidget(m_pLblTitle);

    // 20260324 ZJH 副标题 — 描述项目定位（汉化）
    m_pLblSubtitle = new QLabel(
        QStringLiteral("纯 C++ 深度学习视觉平台"), this);
    m_pLblSubtitle->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 水平居中
    m_pLblSubtitle->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #94a3b8;"         // 灰色文字，与主标题形成层次
        "  font-size: 11pt;"        // 11 磅中号字
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(m_pLblSubtitle);

    // 20260324 ZJH 版本号标签（使用 CMake 注入的 OM_VERSION 宏）
    m_pLblVersion = new QLabel(QStringLiteral("v") + QStringLiteral(OM_VERSION), this);
    m_pLblVersion->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 水平居中
    m_pLblVersion->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #64748b;"         // 更暗的灰色，视觉权重最低
        "  font-size: 10pt;"        // 10 磅小号字
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(m_pLblVersion);

    // 20260322 ZJH 中间弹性空间
    pLayout->addStretch(3);

    // 20260322 ZJH 底部进度条 — 不确定模式（无限循环动画条）
    m_pProgressBar = new QProgressBar(this);
    m_pProgressBar->setMinimum(0);   // 20260322 ZJH 最小值和最大值都为 0 = 不确定模式
    m_pProgressBar->setMaximum(0);   // 20260322 ZJH 不确定模式，进度条无限循环滚动
    m_pProgressBar->setFixedHeight(4);  // 20260322 ZJH 极细进度条，4px 高
    m_pProgressBar->setTextVisible(false);  // 20260322 ZJH 不显示百分比文字
    m_pProgressBar->setStyleSheet(QStringLiteral(
        "QProgressBar {"
        "  background-color: #2a2d35;"   // 进度条底色
        "  border: none;"
        "  border-radius: 2px;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: #2563eb;"   // 蓝色滚动块
        "  border-radius: 2px;"
        "}"
    ));
    pLayout->addWidget(m_pProgressBar);

    // 20260322 ZJH 底部状态标签 — 显示当前加载步骤
    m_pLblStatus = new QLabel(this);
    m_pLblStatus->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 水平居中
    m_pLblStatus->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #64748b;"         // 灰色状态文字
        "  font-size: 9pt;"         // 9 磅小号字
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(m_pLblStatus);

    // 20260322 ZJH 将窗口居中到屏幕中央
    if (QScreen* pScreen = QGuiApplication::primaryScreen()) {
        QRect screenGeometry = pScreen->availableGeometry();  // 20260322 ZJH 获取可用屏幕区域
        int nX = (screenGeometry.width() - s_nSplashWidth) / 2 + screenGeometry.x();    // 20260322 ZJH 水平居中
        int nY = (screenGeometry.height() - s_nSplashHeight) / 2 + screenGeometry.y();  // 20260322 ZJH 垂直居中
        move(nX, nY);  // 20260322 ZJH 移动窗口到屏幕中央
    }
}

// 20260322 ZJH 设置底部状态文本，用于展示加载步骤（如 "正在初始化..."）
void SplashScreen::setStatus(const QString& strMsg)
{
    m_pLblStatus->setText(strMsg);  // 20260322 ZJH 更新状态标签文本
}

// 20260322 ZJH 启动窗口渐隐动画
// 参数 nDurationMs: 渐隐持续时间（毫秒），默认 500ms
// 动画完成后发射 fadeOutFinished 信号，通知调用者显示主窗口
void SplashScreen::fadeOut(int nDurationMs)
{
    // 20260322 ZJH 创建 windowOpacity 属性动画
    QPropertyAnimation* pAnim = new QPropertyAnimation(this, "opacity", this);
    pAnim->setDuration(nDurationMs);     // 20260322 ZJH 动画持续时间
    pAnim->setStartValue(1.0);           // 20260322 ZJH 起始完全不透明
    pAnim->setEndValue(0.0);             // 20260322 ZJH 结束完全透明
    pAnim->setEasingCurve(QEasingCurve::OutQuad);  // 20260322 ZJH 减速缓出曲线，视觉更自然

    // 20260322 ZJH 动画完成后发射 fadeOutFinished 信号
    connect(pAnim, &QPropertyAnimation::finished, this, [this]() {
        emit fadeOutFinished();  // 20260322 ZJH 通知外部渐隐完成，可以显示主窗口
    });

    pAnim->start(QAbstractAnimation::DeleteWhenStopped);  // 20260322 ZJH 动画结束后自动删除动画对象
}
