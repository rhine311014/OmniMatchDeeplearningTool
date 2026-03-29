// 20260322 ZJH ShortcutHelpOverlay — 快捷键参考面板
// QWidget 子类，半透明黑色遮罩覆盖主窗口，中央卡片显示快捷键表格
// F1 触发显示，按任意键或点击关闭
#pragma once

#include <QWidget>    // 20260322 ZJH 控件基类

// 20260322 ZJH 快捷键参考面板
// 功能：
//   - 半透明黑色遮罩覆盖父窗口
//   - 中央白色/暗色卡片显示分组快捷键表格
//   - 分组：文件 / 页面导航 / 标注工具 / 训练控制
//   - 按任意键或鼠标点击关闭
//   - 20260324 ZJH 安装事件过滤器跟踪父控件尺寸变化，自动同步大小
class ShortcutHelpOverlay : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父窗口（通常为 MainWindow）
    explicit ShortcutHelpOverlay(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~ShortcutHelpOverlay() override = default;

    // 20260322 ZJH 显示遮罩
    void showOverlay();

protected:
    // 20260322 ZJH 自绘事件：绘制半透明遮罩 + 中央卡片 + 快捷键表格
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260322 ZJH 按键事件：任意键关闭
    void keyPressEvent(QKeyEvent* pEvent) override;

    // 20260322 ZJH 鼠标点击事件：点击关闭
    void mousePressEvent(QMouseEvent* pEvent) override;

    // 20260324 ZJH 事件过滤器：监听父控件 Resize 事件，同步调整遮罩大小
    bool eventFilter(QObject* pWatched, QEvent* pEvent) override;
};
