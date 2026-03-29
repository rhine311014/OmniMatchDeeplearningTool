// 20260323 ZJH LoadingOverlay — 半透明加载遮罩层
// 覆盖在父控件上方，显示旋转动画和状态文字
// 用于长时间操作（导入/训练/评估）时阻止用户交互
#pragma once

#include <QWidget>            // 20260323 ZJH 控件基类
#include <QTimer>             // 20260323 ZJH 动画驱动定时器
#include <QString>            // 20260323 ZJH 字符串

// 20260323 ZJH 半透明加载遮罩层
class LoadingOverlay : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    // 参数: pParent - 父控件（遮罩将覆盖父控件整个区域）
    explicit LoadingOverlay(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~LoadingOverlay() override = default;

    // 20260323 ZJH 显示遮罩层并开始动画
    // 参数: strMessage - 显示的状态文字（如 "Loading..."）
    void showWithMessage(const QString& strMessage);

    // 20260323 ZJH 更新状态文字
    void setMessage(const QString& strMessage);

    // 20260323 ZJH 隐藏遮罩层并停止动画
    void hideOverlay();

protected:
    // 20260323 ZJH 绘制事件：半透明背景 + 旋转圆弧 + 文字
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260323 ZJH 父控件调整大小时同步覆盖
    bool eventFilter(QObject* pObj, QEvent* pEvent) override;

private:
    QString m_strMessage;       // 20260323 ZJH 显示的状态文字
    QTimer* m_pAnimTimer;       // 20260323 ZJH 动画定时器（30ms 周期）
    int m_nRotation = 0;        // 20260323 ZJH 旋转角度（0~360 循环）
};
