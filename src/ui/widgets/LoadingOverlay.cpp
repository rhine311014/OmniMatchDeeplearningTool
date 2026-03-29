// 20260323 ZJH LoadingOverlay — 半透明加载遮罩层实现

#include "ui/widgets/LoadingOverlay.h"
#include "ui/widgets/ThemeColors.h"    // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>     // 20260323 ZJH 绘图引擎
#include <QEvent>       // 20260323 ZJH 事件类型
#include <cmath>        // 20260323 ZJH 数学函数

// 20260323 ZJH 构造函数
LoadingOverlay::LoadingOverlay(QWidget* pParent)
    : QWidget(pParent)
    , m_pAnimTimer(new QTimer(this))
{
    // 20260323 ZJH 透明背景+无边框+置顶
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_TranslucentBackground);

    // 20260323 ZJH 安装事件过滤器以跟踪父控件大小变化
    if (pParent) {
        pParent->installEventFilter(this);
        resize(pParent->size());  // 20260323 ZJH 初始大小与父控件一致
    }

    // 20260323 ZJH 动画定时器：每 30ms 更新旋转角度
    connect(m_pAnimTimer, &QTimer::timeout, this, [this]() {
        m_nRotation = (m_nRotation + 8) % 360;  // 20260323 ZJH 每帧旋转 8 度
        update();  // 20260323 ZJH 触发重绘
    });

    hide();  // 20260323 ZJH 默认隐藏
}

// 20260323 ZJH 显示遮罩层
void LoadingOverlay::showWithMessage(const QString& strMessage)
{
    m_strMessage = strMessage;
    m_nRotation = 0;
    raise();               // 20260323 ZJH 置顶显示
    show();                // 20260323 ZJH 显示
    m_pAnimTimer->start(30);  // 20260323 ZJH 启动动画
}

// 20260323 ZJH 更新状态文字
void LoadingOverlay::setMessage(const QString& strMessage)
{
    m_strMessage = strMessage;
    update();
}

// 20260323 ZJH 隐藏遮罩层
void LoadingOverlay::hideOverlay()
{
    m_pAnimTimer->stop();  // 20260323 ZJH 停止动画
    hide();                // 20260323 ZJH 隐藏
}

// 20260323 ZJH 绘制事件
void LoadingOverlay::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 20260323 ZJH 半透明黑色背景
    painter.fillRect(rect(), QColor(0, 0, 0, 160));

    int nCx = width() / 2;    // 20260323 ZJH 中心 X
    int nCy = height() / 2;   // 20260323 ZJH 中心 Y

    // 20260323 ZJH 旋转圆弧（spinner）
    int nRadius = 24;  // 20260323 ZJH 圆弧半径
    QPen penArc(QColor("#3b82f6"), 4, Qt::SolidLine, Qt::RoundCap);
    painter.setPen(penArc);

    // 20260323 ZJH 绘制 270 度圆弧，旋转角度由 m_nRotation 驱动
    QRectF arcRect(nCx - nRadius, nCy - nRadius - 20, nRadius * 2, nRadius * 2);
    painter.drawArc(arcRect, m_nRotation * 16, 270 * 16);

    // 20260323 ZJH 状态文字
    painter.setPen(QColor("#e2e8f0"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 12));  // 20260324 ZJH 使用共享字体族名
    QRectF textRect(0, nCy + 20, width(), 30);
    painter.drawText(textRect, Qt::AlignCenter, m_strMessage);
}

// 20260323 ZJH 事件过滤器：父控件 resize 时同步大小
bool LoadingOverlay::eventFilter(QObject* pObj, QEvent* pEvent)
{
    if (pObj == parent() && pEvent->type() == QEvent::Resize) {
        QWidget* pParent = qobject_cast<QWidget*>(parent());
        if (pParent) {
            resize(pParent->size());  // 20260323 ZJH 同步大小
        }
    }
    return QWidget::eventFilter(pObj, pEvent);
}
