// 20260322 ZJH NewProjectDialog — 新建项目对话框
// 提供项目名称、任务类型、存储路径三项输入
// 确认后由 ProjectPage 读取配置并创建项目
#pragma once

#include <QDialog>          // 20260322 ZJH 对话框基类
#include <QLineEdit>        // 20260322 ZJH 文本输入框
#include <QComboBox>        // 20260322 ZJH 下拉选择框
#include <QPushButton>      // 20260322 ZJH 按钮
#include <QDialogButtonBox> // 20260322 ZJH 标准确定/取消按钮框

#include "core/DLTypes.h"   // 20260322 ZJH TaskType 枚举

// 20260322 ZJH 新建项目对话框
// 布局：项目名称 + 任务类型下拉 + 项目路径（浏览按钮） + 确定/取消
class NewProjectDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化对话框 UI
    // 参数: pParent - 父窗口
    explicit NewProjectDialog(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~NewProjectDialog() override = default;

    // ===== 数据访问 =====

    // 20260322 ZJH 获取用户输入的项目名称
    // 返回: 项目名称字符串
    QString projectName() const;

    // 20260322 ZJH 获取用户选择的任务类型
    // 返回: TaskType 枚举值
    om::TaskType taskType() const;

    // 20260322 ZJH 获取用户选择的项目路径
    // 返回: 项目存储文件夹绝对路径
    QString projectPath() const;

private slots:
    // 20260322 ZJH 浏览按钮点击：弹出文件夹选择对话框
    void onBrowsePath();

    // 20260322 ZJH 验证输入合法性并接受对话框
    void onAccept();

private:
    // 20260322 ZJH 初始化对话框布局和控件
    void setupUi();

    QLineEdit*        m_pEdtName;      // 20260322 ZJH 项目名称输入框
    QComboBox*        m_pCmbTaskType;  // 20260322 ZJH 任务类型下拉框（8 种任务）
    QLineEdit*        m_pEdtPath;      // 20260322 ZJH 项目路径输入框
    QPushButton*      m_pBtnBrowse;    // 20260322 ZJH 路径浏览按钮
    QDialogButtonBox* m_pBtnBox;       // 20260322 ZJH 确定/取消按钮框
};
