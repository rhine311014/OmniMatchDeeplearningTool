// 20260323 ZJH ToastNotification — 轻量弹出式通知控件
// 在窗口右上角弹出短暂的通知消息
// 支持 Info/Success/Warning/Error 四种类型，自动淡出消失
#pragma once

#include <QWidget>             // 20260323 ZJH 控件基类
#include <QTimer>              // 20260323 ZJH 自动关闭定时器
#include <QPropertyAnimation>  // 20260323 ZJH 淡入淡出动画

// 20260323 ZJH 通知类型枚举
enum class ToastType
{
    Info = 0,    // 20260323 ZJH 信息（蓝色）
    Success,     // 20260323 ZJH 成功（绿色）
    Warning,     // 20260323 ZJH 警告（橙色）
    Error        // 20260323 ZJH 错误（红色）
};

// 20260323 ZJH 弹出式通知控件
class ToastNotification : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(qreal opacity READ windowOpacity WRITE setWindowOpacity)

public:
    // 20260323 ZJH 构造函数
    explicit ToastNotification(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~ToastNotification() override = default;

    // 20260323 ZJH 显示通知
    // 参数: strMessage - 消息文字
    //       type - 通知类型（Info/Success/Warning/Error）
    //       nDurationMs - 显示时长（毫秒，默认 3000）
    void showToast(const QString& strMessage, ToastType type = ToastType::Info, int nDurationMs = 3000);

    // 20260323 ZJH 静态便捷方法：在指定父窗口弹出通知
    static void showMessage(QWidget* pParent, const QString& strMessage,
                            ToastType type = ToastType::Info, int nDurationMs = 3000);

protected:
    // 20260323 ZJH 绘制事件：圆角背景 + 图标 + 文字
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260324 ZJH 事件过滤器：父窗口移动或调整大小时重新定位 toast
    bool eventFilter(QObject* pObj, QEvent* pEvent) override;

private:
    // 20260324 ZJH 重新计算 toast 相对于父窗口右上角的位置
    void repositionToParent();

    // 20260323 ZJH 开始淡出动画
    void fadeOut();

    // 20260323 ZJH 根据类型获取背景颜色
    QColor colorForType(ToastType type) const;

    // 20260323 ZJH 根据类型获取图标字符
    QString iconForType(ToastType type) const;

    QString m_strMessage;           // 20260323 ZJH 消息文字
    ToastType m_type = ToastType::Info;  // 20260323 ZJH 通知类型
    QTimer* m_pAutoCloseTimer;      // 20260323 ZJH 自动关闭定时器
    QPropertyAnimation* m_pFadeAnim;  // 20260323 ZJH 淡出动画
};
