// 20260323 ZJH ToastNotification — 弹出式通知控件实现

#include "ui/widgets/ToastNotification.h"
#include "ui/widgets/ThemeColors.h"        // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>         // 20260323 ZJH 绘图引擎
#include <QPainterPath>     // 20260323 ZJH 圆角路径
#include <QApplication>     // 20260323 ZJH 应用程序实例
#include <QFontMetrics>     // 20260323 ZJH 字体测量
#include <QEvent>           // 20260324 ZJH 事件类型（Resize/Move）

// 20260323 ZJH 构造函数
ToastNotification::ToastNotification(QWidget* pParent)
    : QWidget(pParent)
    , m_pAutoCloseTimer(new QTimer(this))
    , m_pFadeAnim(new QPropertyAnimation(this, "opacity", this))
{
    // 20260323 ZJH 无边框、置顶、不聚焦
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool | Qt::WindowStaysOnTopHint);
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);

    // 20260323 ZJH 自动关闭定时器单次触发
    m_pAutoCloseTimer->setSingleShot(true);
    connect(m_pAutoCloseTimer, &QTimer::timeout, this, &ToastNotification::fadeOut);

    // 20260323 ZJH 淡出动画完成后关闭
    m_pFadeAnim->setDuration(300);  // 20260323 ZJH 300ms 淡出
    connect(m_pFadeAnim, &QPropertyAnimation::finished, this, &ToastNotification::close);
}

// 20260323 ZJH 显示通知
void ToastNotification::showToast(const QString& strMessage, ToastType type, int nDurationMs)
{
    m_strMessage = strMessage;
    m_type = type;

    // 20260323 ZJH 根据文字长度计算控件尺寸
    QFontMetrics fm(QFont(ThemeColors::s_strFontFamily, 10));  // 20260324 ZJH 使用共享字体族名
    int nTextWidth = fm.horizontalAdvance(strMessage) + 60;  // 20260323 ZJH 图标+边距
    int nW = qMax(nTextWidth, 200);
    int nH = 44;
    resize(nW, nH);

    // 20260323 ZJH 定位到父窗口右上角
    if (parentWidget()) {
        // 20260324 ZJH 安装事件过滤器以跟踪父窗口移动和调整大小
        parentWidget()->installEventFilter(this);
        QPoint ptParentTopRight = parentWidget()->mapToGlobal(
            QPoint(parentWidget()->width() - nW - 20, 60));
        move(ptParentTopRight);
    }

    setWindowOpacity(1.0);  // 20260323 ZJH 完全不透明
    show();
    raise();

    // 20260323 ZJH 启动自动关闭定时器
    m_pAutoCloseTimer->start(nDurationMs);
}

// 20260323 ZJH 静态便捷方法
void ToastNotification::showMessage(QWidget* pParent, const QString& strMessage,
                                    ToastType type, int nDurationMs)
{
    // 20260323 ZJH 创建 Toast 实例（自动释放）
    auto* pToast = new ToastNotification(pParent);
    pToast->setAttribute(Qt::WA_DeleteOnClose);  // 20260323 ZJH 关闭时自动删除
    pToast->showToast(strMessage, type, nDurationMs);
}

// 20260323 ZJH 绘制事件
void ToastNotification::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 20260323 ZJH 圆角背景
    QPainterPath path;
    path.addRoundedRect(rect().adjusted(1, 1, -1, -1), 8, 8);

    QColor bgColor = colorForType(m_type);
    painter.fillPath(path, bgColor);

    // 20260323 ZJH 左侧色条
    QRectF barRect(1, 1, 4, height() - 2);
    QColor barColor = bgColor.lighter(150);
    painter.fillRect(barRect, barColor);

    // 20260323 ZJH 图标
    painter.setPen(QColor("#ffffff"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 14));  // 20260324 ZJH 使用共享字体族名
    painter.drawText(QRectF(12, 0, 24, height()), Qt::AlignCenter, iconForType(m_type));

    // 20260323 ZJH 消息文字
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 10));  // 20260324 ZJH 使用共享字体族名
    painter.drawText(QRectF(40, 0, width() - 50, height()),
                    Qt::AlignVCenter | Qt::AlignLeft, m_strMessage);
}

// 20260324 ZJH 事件过滤器：父窗口移动或调整大小时重新定位 toast
bool ToastNotification::eventFilter(QObject* pObj, QEvent* pEvent)
{
    // 20260324 ZJH 仅处理父窗口的 Move 和 Resize 事件
    if (pObj == parentWidget() &&
        (pEvent->type() == QEvent::Move || pEvent->type() == QEvent::Resize))
    {
        repositionToParent();  // 20260324 ZJH 重新计算位置
    }
    return QWidget::eventFilter(pObj, pEvent);  // 20260324 ZJH 调用基类处理
}

// 20260324 ZJH 重新计算 toast 相对于父窗口右上角的位置
void ToastNotification::repositionToParent()
{
    if (!parentWidget()) {
        return;  // 20260324 ZJH 无父窗口时无需定位
    }
    // 20260324 ZJH 将 toast 锚定到父窗口右上角，与 showToast 中相同的偏移量
    QPoint ptParentTopRight = parentWidget()->mapToGlobal(
        QPoint(parentWidget()->width() - width() - 20, 60));
    move(ptParentTopRight);  // 20260324 ZJH 移动到新位置
}

// 20260323 ZJH 淡出动画
void ToastNotification::fadeOut()
{
    m_pFadeAnim->setStartValue(1.0);
    m_pFadeAnim->setEndValue(0.0);
    m_pFadeAnim->start();
}

// 20260323 ZJH 根据类型获取背景颜色
QColor ToastNotification::colorForType(ToastType type) const
{
    switch (type) {
        case ToastType::Info:    return QColor(37, 99, 235, 220);   // 20260323 ZJH 蓝色
        case ToastType::Success: return QColor(16, 185, 129, 220);  // 20260323 ZJH 绿色
        case ToastType::Warning: return QColor(245, 158, 11, 220);  // 20260323 ZJH 橙色
        case ToastType::Error:   return QColor(239, 68, 68, 220);   // 20260323 ZJH 红色
    }
    return QColor(37, 99, 235, 220);
}

// 20260323 ZJH 根据类型获取图标字符
QString ToastNotification::iconForType(ToastType type) const
{
    switch (type) {
        case ToastType::Info:    return "i";     // 20260323 ZJH 信息图标
        case ToastType::Success: return "\u2713"; // 20260323 ZJH 勾号 ✓
        case ToastType::Warning: return "!";     // 20260323 ZJH 感叹号
        case ToastType::Error:   return "\u2717"; // 20260323 ZJH 叉号 ✗
    }
    return "i";
}
