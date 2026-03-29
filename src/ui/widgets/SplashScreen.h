// 20260322 ZJH SplashScreen — OmniMatch 启动动画窗口
// 无边框置顶窗口，展示应用名称/版本和加载进度
// 启动完成后通过 fadeOut() 渐隐动画过渡到主窗口

#pragma once

#include <QWidget>              // 20260322 ZJH 基础窗口控件
#include <QLabel>               // 20260322 ZJH 标题/副标题/版本/状态文本标签
#include <QProgressBar>         // 20260322 ZJH 底部不确定模式进度条
#include <QPropertyAnimation>   // 20260322 ZJH windowOpacity 渐隐动画

// 20260322 ZJH 启动画面窗口
// 功能：
//   1. 无边框、置顶、居中显示 480x320 深色启动画面
//   2. 展示 "OmniMatch" 大标题 + 副标题 + 版本号
//   3. 底部不确定模式进度条 + 状态文本（如 "正在初始化..."）
//   4. fadeOut() 方法触发 windowOpacity 渐隐动画，完成后发射 fadeOutFinished
class SplashScreen : public QWidget
{
    Q_OBJECT

    // 20260322 ZJH 声明 opacity 属性，用于 QPropertyAnimation 驱动窗口渐隐
    Q_PROPERTY(qreal opacity READ windowOpacity WRITE setWindowOpacity)

public:
    // 20260322 ZJH 构造函数，创建全部子控件并设置窗口属性
    explicit SplashScreen(QWidget* pParent = nullptr);

    // 20260322 ZJH 设置底部状态文本（如 "正在加载主题..."、"启动完成"）
    void setStatus(const QString& strMsg);

    // 20260322 ZJH 启动渐隐动画，默认 500ms
    // 动画完成后发射 fadeOutFinished 信号
    void fadeOut(int nDurationMs = 500);

signals:
    // 20260322 ZJH 渐隐动画完成信号，主窗口监听此信号后显示自身
    void fadeOutFinished();

private:
    QLabel*       m_pLblTitle;      // 20260322 ZJH "OmniMatch" 大标题标签
    QLabel*       m_pLblSubtitle;   // 20260322 ZJH 副标题标签
    QLabel*       m_pLblVersion;    // 20260322 ZJH 版本号标签
    QLabel*       m_pLblStatus;     // 20260322 ZJH 底部加载状态标签
    QProgressBar* m_pProgressBar;   // 20260322 ZJH 不确定模式进度条（无限循环动画）
};
