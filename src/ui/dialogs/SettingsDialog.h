// 20260322 ZJH SettingsDialog — 应用设置对话框
// QDialog 子类，QTabWidget 三个标签页：常规、外观、GPU
// 提供语言选择、自动保存、主题切换、字体大小、GPU 设备选择等功能
#pragma once

#include <QDialog>           // 20260322 ZJH 对话框基类
#include <QTabWidget>        // 20260322 ZJH 标签页容器
#include <QComboBox>         // 20260322 ZJH 下拉选择框
#include <QSpinBox>          // 20260322 ZJH 整数微调框
#include <QLabel>            // 20260322 ZJH 文本标签
#include <QDialogButtonBox>  // 20260322 ZJH 标准按钮组

// 20260322 ZJH 应用设置对话框
// 三个标签页：
//   常规 — 语言/自动保存间隔/最近项目数
//   外观 — 主题选择/字体大小
//   GPU  — 显示 GPU 信息/设备选择
class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父窗口（通常为 MainWindow）
    explicit SettingsDialog(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~SettingsDialog() override = default;

private slots:
    // 20260322 ZJH "应用" 按钮点击
    void onApply();

    // 20260322 ZJH "确定" 按钮点击（应用并关闭）
    void onAccept();

private:
    // 20260322 ZJH 创建 "常规" 标签页
    QWidget* createGeneralTab();

    // 20260322 ZJH 创建 "外观" 标签页
    QWidget* createAppearanceTab();

    // 20260322 ZJH 创建 "GPU" 标签页
    QWidget* createGpuTab();

    // 20260322 ZJH 应用当前设置到系统
    void applySettings();

    // ===== 常规标签页控件 =====

    QComboBox* m_pCmbLanguage;         // 20260322 ZJH 语言选择（中文/英文）
    QSpinBox*  m_pSpnAutoSave;         // 20260322 ZJH 自动保存间隔（分钟）
    QSpinBox*  m_pSpnRecentProjects;   // 20260322 ZJH 最近项目数

    // ===== 外观标签页控件 =====

    QComboBox* m_pCmbTheme;            // 20260322 ZJH 主题选择（暗色/亮色）
    QSpinBox*  m_pSpnFontSize;         // 20260322 ZJH 字体大小

    // ===== GPU 标签页控件 =====

    QLabel*    m_pLblGpuInfo;          // 20260322 ZJH GPU 信息标签
    QComboBox* m_pCmbDevice;           // 20260322 ZJH 设备选择（CPU/GPU）

    // ===== 按钮 =====

    QDialogButtonBox* m_pButtonBox;    // 20260322 ZJH OK/Cancel/Apply 按钮组
};
