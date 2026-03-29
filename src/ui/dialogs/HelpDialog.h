// 20260323 ZJH HelpDialog — 帮助对话框
// 显示 OmniMatch 使用说明、快捷键参考和版本信息
// QTabWidget 三个标签页：使用指南 / 快捷键 / 关于
#pragma once

#include <QDialog>     // 20260323 ZJH 对话框基类
#include <QTabWidget>  // 20260323 ZJH 标签页容器

// 20260323 ZJH 帮助对话框
class HelpDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit HelpDialog(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~HelpDialog() override = default;

private:
    // 20260323 ZJH 创建 "使用指南" 标签页
    QWidget* createGuideTab();

    // 20260323 ZJH 创建 "快捷键" 标签页
    QWidget* createShortcutsTab();

    // 20260323 ZJH 创建 "关于" 标签页
    QWidget* createAboutTab();
};
