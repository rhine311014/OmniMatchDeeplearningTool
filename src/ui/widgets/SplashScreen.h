// 20260403 ZJH SplashScreen — OmniMatch 启动动画窗口
// 后台线程持续渲染粒子到 QImage，主线程 paintEvent 只做 drawImage 拷贝
// 确保主线程任何阻塞（页面构造/GPU检测）都不影响粒子动画流畅度
#pragma once

#include <QWidget>              // 20260322 ZJH 基础窗口控件
#include <QLabel>               // 20260322 ZJH 标题/副标题/版本/状态文本标签
#include <QProgressBar>         // 20260322 ZJH 底部进度条
#include <QPropertyAnimation>   // 20260322 ZJH windowOpacity 渐隐动画
#include <QThread>              // 20260403 ZJH 粒子渲染后台线程
#include <QMutex>               // 20260403 ZJH 帧缓冲区互斥锁
#include <QImage>               // 20260403 ZJH 离屏粒子帧缓冲区
#include <QVector>              // 20260401 ZJH 粒子列表
#include <QPointF>              // 20260401 ZJH 粒子位置
#include <QAtomicInt>           // 20260403 ZJH 线程停止标志

// 20260401 ZJH 单个粒子的状态
struct SplashParticle {
    QPointF pos;       // 20260401 ZJH 当前位置
    QPointF vel;       // 20260401 ZJH 速度向量（像素/帧）
    float fRadius;     // 20260401 ZJH 半径
    float fAlpha;      // 20260401 ZJH 透明度
};

// 20260403 ZJH 粒子渲染后台线程
// 独立线程中 16ms 循环: 更新粒子位置 → 绘制到 QImage → 通知主线程重绘
class ParticleRenderThread : public QThread
{
    Q_OBJECT
public:
    // 20260403 ZJH 构造函数，初始化帧缓冲区大小
    ParticleRenderThread(int nWidth, int nHeight, QObject* pParent = nullptr);

    // 20260403 ZJH 请求线程停止
    void requestStop();

    // 20260403 ZJH 获取当前帧（线程安全拷贝）
    QImage currentFrame();

signals:
    // 20260403 ZJH 新帧就绪信号，通知 SplashScreen 重绘
    void frameReady();

protected:
    // 20260403 ZJH 线程主循环
    void run() override;

private:
    // 20260403 ZJH 初始化粒子
    void initParticles();
    // 20260403 ZJH 更新粒子位置
    void updateParticles();
    // 20260403 ZJH 绘制粒子到帧缓冲区
    void renderFrame();

    int m_nWidth;       // 20260403 ZJH 帧宽度
    int m_nHeight;      // 20260403 ZJH 帧高度
    QAtomicInt m_bStop; // 20260403 ZJH 停止标志
    QMutex m_mutex;     // 20260403 ZJH 帧缓冲区互斥锁
    QImage m_frame;     // 20260403 ZJH 当前帧缓冲区
    QVector<SplashParticle> m_vecParticles;  // 20260403 ZJH 粒子列表
};

// 20260403 ZJH 启动画面窗口
class SplashScreen : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(qreal opacity READ windowOpacity WRITE setWindowOpacity)

public:
    explicit SplashScreen(QWidget* pParent = nullptr);
    ~SplashScreen() override;

    // 20260322 ZJH 设置底部状态文本
    void setStatus(const QString& strMsg);
    // 20260403 ZJH 设置进度条百分比（0~100）
    void setProgress(int nPercent);
    // 20260403 ZJH 同步刷新（兼容旧接口，现在只做 repaint）
    void tickAnimation();
    // 20260322 ZJH 启动渐隐动画
    void fadeOut(int nDurationMs = 500);

signals:
    void fadeOutFinished();

protected:
    void paintEvent(QPaintEvent* pEvent) override;

private:
    QLabel*       m_pLblTitle;
    QLabel*       m_pLblSubtitle;
    QLabel*       m_pLblVersion;
    QLabel*       m_pLblStatus;
    QProgressBar* m_pProgressBar;

    // 20260403 ZJH 后台粒子渲染线程
    ParticleRenderThread* m_pRenderThread;
};
