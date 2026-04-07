// 20260403 ZJH SplashScreen 实现
// 后台线程持续渲染粒子到 QImage，主线程 paintEvent 只做 drawImage
// 主线程阻塞期间粒子位置在后台线程持续更新，下一次 repaint 时粒子位置连续

#include "ui/widgets/SplashScreen.h"

#include <QVBoxLayout>
#include <QPropertyAnimation>
#include <QScreen>
#include <QGuiApplication>
#include <QApplication>        // 20260403 ZJH processEvents() 强制刷新事件队列
#include <QPainter>
#include <QPen>
#include <QRandomGenerator>
#include <QElapsedTimer>
#include <cmath>

static constexpr int s_nSplashWidth  = 560;
static constexpr int s_nSplashHeight = 380;
static constexpr int s_nParticleCount = 40;
static constexpr float s_fConnectDist = 100.0f;

// ===== ParticleRenderThread =====

ParticleRenderThread::ParticleRenderThread(int nWidth, int nHeight, QObject* pParent)
    : QThread(pParent)
    , m_nWidth(nWidth)
    , m_nHeight(nHeight)
    , m_bStop(0)
    , m_frame(nWidth, nHeight, QImage::Format_ARGB32_Premultiplied)
{
    m_frame.fill(Qt::transparent);
    initParticles();
}

void ParticleRenderThread::requestStop()
{
    m_bStop.storeRelaxed(1);
}

QImage ParticleRenderThread::currentFrame()
{
    QMutexLocker lock(&m_mutex);
    return m_frame.copy();  // 20260403 ZJH 深拷贝，线程安全
}

void ParticleRenderThread::initParticles()
{
    auto* pRng = QRandomGenerator::global();
    m_vecParticles.resize(s_nParticleCount);
    for (int i = 0; i < s_nParticleCount; ++i) {
        auto& p = m_vecParticles[i];
        p.pos = QPointF(pRng->bounded(m_nWidth), pRng->bounded(m_nHeight));
        float fSpeed = 0.2f + pRng->bounded(60) / 100.0f;
        float fAngle = pRng->bounded(628) / 100.0f;
        p.vel = QPointF(fSpeed * std::cos(fAngle), fSpeed * std::sin(fAngle));
        p.fRadius = 1.5f + pRng->bounded(35) / 10.0f;
        p.fAlpha = 0.15f + pRng->bounded(35) / 100.0f;
    }
}

void ParticleRenderThread::updateParticles()
{
    for (auto& p : m_vecParticles) {
        p.pos += p.vel;
        if (p.pos.x() < 0 || p.pos.x() > m_nWidth) {
            p.vel.setX(-p.vel.x());
            p.pos.setX(qBound(0.0, p.pos.x(), static_cast<double>(m_nWidth)));
        }
        if (p.pos.y() < 0 || p.pos.y() > m_nHeight) {
            p.vel.setY(-p.vel.y());
            p.pos.setY(qBound(0.0, p.pos.y(), static_cast<double>(m_nHeight)));
        }
    }
}

void ParticleRenderThread::renderFrame()
{
    // 20260403 ZJH 绘制到本地临时 QImage（不持锁，避免阻塞主线程）
    QImage tempFrame(m_nWidth, m_nHeight, QImage::Format_ARGB32_Premultiplied);
    tempFrame.fill(Qt::transparent);

    QPainter painter(&tempFrame);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // 20260403 ZJH 绘制连线
    for (int i = 0; i < m_vecParticles.size(); ++i) {
        for (int j = i + 1; j < m_vecParticles.size(); ++j) {
            float fDx = static_cast<float>(m_vecParticles[i].pos.x() - m_vecParticles[j].pos.x());
            float fDy = static_cast<float>(m_vecParticles[i].pos.y() - m_vecParticles[j].pos.y());
            float fDist = std::sqrt(fDx * fDx + fDy * fDy);
            if (fDist < s_fConnectDist) {
                float fLineAlpha = (1.0f - fDist / s_fConnectDist) * 0.15f;
                painter.setPen(QPen(QColor(59, 130, 246, static_cast<int>(fLineAlpha * 255)), 0.8));
                painter.drawLine(m_vecParticles[i].pos, m_vecParticles[j].pos);
            }
        }
    }

    // 20260403 ZJH 绘制粒子光点
    for (const auto& p : m_vecParticles) {
        QColor glowColor(59, 130, 246, static_cast<int>(p.fAlpha * 80));
        painter.setPen(Qt::NoPen);
        painter.setBrush(glowColor);
        painter.drawEllipse(p.pos, p.fRadius * 2.5, p.fRadius * 2.5);

        QColor coreColor(147, 197, 253, static_cast<int>(p.fAlpha * 255));
        painter.setBrush(coreColor);
        painter.drawEllipse(p.pos, p.fRadius, p.fRadius);
    }
    painter.end();

    // 20260403 ZJH 持锁交换帧缓冲区（仅赋值，极快）
    {
        QMutexLocker lock(&m_mutex);
        m_frame = tempFrame;
    }
}

void ParticleRenderThread::run()
{
    // 20260403 ZJH 后台线程主循环：16ms 更新 + 渲染 + 通知主线程
    QElapsedTimer timer;
    timer.start();

    while (!m_bStop.loadRelaxed()) {
        updateParticles();
        renderFrame();
        emit frameReady();  // 20260403 ZJH 通知主线程有新帧

        // 20260403 ZJH 精确 16ms 间隔（减去渲染耗时）
        qint64 nElapsed = timer.elapsed();
        qint64 nSleep = 16 - nElapsed;
        if (nSleep > 0) {
            msleep(static_cast<unsigned long>(nSleep));
        }
        timer.restart();
    }
}

// ===== SplashScreen =====

SplashScreen::SplashScreen(QWidget* pParent)
    : QWidget(pParent)
    , m_pRenderThread(nullptr)
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setAttribute(Qt::WA_TranslucentBackground, false);
    setFixedSize(s_nSplashWidth, s_nSplashHeight);

    setStyleSheet(QStringLiteral(
        "SplashScreen {"
        "  background-color: #1a1d23;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 12px;"
        "}"
    ));

    QVBoxLayout* pLayout = new QVBoxLayout(this);
    pLayout->setContentsMargins(50, 50, 50, 35);
    pLayout->setSpacing(8);

    pLayout->addStretch(2);

    m_pLblTitle = new QLabel(QStringLiteral("OmniMatch"), this);
    m_pLblTitle->setAlignment(Qt::AlignCenter);
    m_pLblTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #ffffff; font-size: 36pt; font-weight: bold;"
        " background: transparent; border: none; letter-spacing: 2px; }"));
    pLayout->addWidget(m_pLblTitle);

    m_pLblSubtitle = new QLabel(QStringLiteral("纯 C++ 深度学习视觉平台"), this);
    m_pLblSubtitle->setAlignment(Qt::AlignCenter);
    m_pLblSubtitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 11pt; background: transparent; border: none; }"));
    pLayout->addWidget(m_pLblSubtitle);

    m_pLblVersion = new QLabel(QStringLiteral("v") + QStringLiteral(OM_VERSION), this);
    m_pLblVersion->setAlignment(Qt::AlignCenter);
    m_pLblVersion->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 10pt; background: transparent; border: none; }"));
    pLayout->addWidget(m_pLblVersion);

    pLayout->addStretch(3);

    m_pProgressBar = new QProgressBar(this);
    m_pProgressBar->setMinimum(0);
    m_pProgressBar->setMaximum(0);  // 20260403 ZJH 初始不确定模式
    m_pProgressBar->setFixedHeight(3);
    m_pProgressBar->setTextVisible(false);
    m_pProgressBar->setStyleSheet(QStringLiteral(
        "QProgressBar { background-color: #2a2d35; border: none; border-radius: 1px; }"
        "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        "  stop:0 #1e40af, stop:0.5 #3b82f6, stop:1 #1e40af); border-radius: 1px; }"));
    pLayout->addWidget(m_pProgressBar);

    m_pLblStatus = new QLabel(this);
    m_pLblStatus->setAlignment(Qt::AlignCenter);
    m_pLblStatus->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; background: transparent; border: none; }"));
    pLayout->addWidget(m_pLblStatus);

    // 20260403 ZJH 居中到屏幕
    if (QScreen* pScreen = QGuiApplication::primaryScreen()) {
        QRect screenGeometry = pScreen->availableGeometry();
        int nX = (screenGeometry.width() - s_nSplashWidth) / 2 + screenGeometry.x();
        int nY = (screenGeometry.height() - s_nSplashHeight) / 2 + screenGeometry.y();
        move(nX, nY);
    }

    // 20260403 ZJH 启动后台粒子渲染线程
    m_pRenderThread = new ParticleRenderThread(s_nSplashWidth, s_nSplashHeight, this);
    // 20260403 ZJH frameReady 信号用 QueuedConnection 投递到主线程事件队列
    // 主线程空闲时处理 → update() → paintEvent；主线程阻塞时信号排队不丢失
    connect(m_pRenderThread, &ParticleRenderThread::frameReady,
            this, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    m_pRenderThread->start();
}

SplashScreen::~SplashScreen()
{
    // 20260403 ZJH 停止后台线程并等待退出
    if (m_pRenderThread) {
        m_pRenderThread->requestStop();
        m_pRenderThread->wait(2000);  // 20260403 ZJH 最多等 2 秒
    }
}

void SplashScreen::setStatus(const QString& strMsg)
{
    m_pLblStatus->setText(strMsg);
    m_pLblStatus->repaint();
}

void SplashScreen::setProgress(int nPercent)
{
    if (m_pProgressBar->maximum() == 0) {
        m_pProgressBar->setMaximum(100);
    }
    m_pProgressBar->setValue(qBound(0, nPercent, 100));
    m_pProgressBar->repaint();
}

void SplashScreen::tickAnimation()
{
    // 20260403 ZJH 强制处理所有挂起事件（包括窗口合成/DWM刷新）
    // processEvents 让 Qt 处理后台线程排队的 update() 信号和 DWM 合成
    // repaint 直接调用 paintEvent 绘制最新帧到窗口
    QApplication::processEvents();
    repaint();
    QApplication::processEvents();  // 20260403 ZJH 再次处理，确保 DWM 完成合成
}

void SplashScreen::paintEvent(QPaintEvent* pEvent)
{
    // 20260403 ZJH 先绘制背景（QSS 样式）
    QWidget::paintEvent(pEvent);

    // 20260403 ZJH 从后台线程获取最新粒子帧并绘制（~1ms drawImage）
    if (m_pRenderThread) {
        QImage frame = m_pRenderThread->currentFrame();
        QPainter painter(this);
        painter.drawImage(0, 0, frame);
    }
}

void SplashScreen::fadeOut(int nDurationMs)
{
    // 20260403 ZJH 渐隐期间停止后台线程
    if (m_pRenderThread) {
        m_pRenderThread->requestStop();
    }

    QPropertyAnimation* pAnim = new QPropertyAnimation(this, "opacity", this);
    pAnim->setDuration(nDurationMs);
    pAnim->setStartValue(1.0);
    pAnim->setEndValue(0.0);
    pAnim->setEasingCurve(QEasingCurve::OutQuad);

    connect(pAnim, &QPropertyAnimation::finished, this, [this]() {
        emit fadeOutFinished();
    });

    pAnim->start(QAbstractAnimation::DeleteWhenStopped);
}
