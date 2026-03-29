// 20260322 ZJH ProjectPage — 项目管理页面
// 使用 QStackedWidget 切换两种视图：
//   1. 欢迎屏（项目未加载时）：品牌展示 + 新建/打开/最近项目
//   2. 项目信息（项目已加载时）：三栏布局显示项目详情和快捷操作
// 替换 MainWindow 中的第 0 个占位页面
#pragma once

#include "ui/pages/BasePage.h"  // 20260322 ZJH 页面基类

#include <QStackedWidget>    // 20260322 ZJH 双视图切换
#include <QListWidget>       // 20260322 ZJH 最近项目列表
#include <QLabel>            // 20260322 ZJH 文本标签
#include <QPushButton>       // 20260322 ZJH 按钮

// 20260322 ZJH 前向声明
class ProjectManager;
class Project;

// 20260322 ZJH 项目管理页面
// 视图 1（欢迎屏）：OmniMatch 品牌 + 新建/打开/打开文件夹 + 最近项目
// 视图 2（项目信息）：左侧项目详情 + 中央概览卡片和快捷按钮
class ProjectPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父控件
    explicit ProjectPage(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~ProjectPage() override = default;

    // ===== 生命周期回调 =====

    // 20260322 ZJH 页面切换到前台时调用（刷新最近项目列表和项目信息）
    void onEnter() override;

    // 20260324 ZJH 项目加载后调用，切换到项目信息视图（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时调用，切换回欢迎屏（Template Method 扩展点）
    void onProjectClosedImpl() override;

    // ===== 公共操作（供 MainWindow 菜单调用） =====

    // 20260322 ZJH 触发新建项目流程（弹出 NewProjectDialog）
    void triggerNewProject();

    // 20260322 ZJH 触发打开项目流程（弹出文件对话框选择 .dfproj）
    void triggerOpenProject();

private slots:
    // 20260322 ZJH 新建项目按钮点击
    void onNewProject();

    // 20260322 ZJH 打开项目按钮点击
    void onOpenProject();

    // 20260322 ZJH 打开文件夹按钮点击（自动创建项目并导入图像）
    void onOpenFolder();

    // 20260322 ZJH 最近项目列表双击打开
    void onRecentProjectDoubleClicked(QListWidgetItem* pItem);

    // 20260322 ZJH "导入图像" 快捷按钮点击 → 跳转画廊页
    void onGoToGallery();

    // 20260322 ZJH "开始标注" 快捷按钮点击 → 跳转图像页
    void onGoToImage();

    // 20260322 ZJH "查看拆分" 快捷按钮点击 → 跳转拆分页
    void onGoToSplit();

private:
    // 20260322 ZJH 创建欢迎屏视图
    QWidget* createWelcomeView();

    // 20260322 ZJH 创建项目信息视图
    QWidget* createProjectInfoView();

    // 20260322 ZJH 刷新最近项目列表
    void refreshRecentList();

    // 20260322 ZJH 刷新项目信息显示
    void refreshProjectInfo();

    // 20260322 ZJH 创建样式化的大按钮
    // 参数: strText - 按钮文字; strColor - 背景色; nWidth - 宽度; nHeight - 高度
    // 返回: 创建的 QPushButton 指针
    QPushButton* createStyledButton(const QString& strText, const QString& strColor,
                                    int nWidth = 120, int nHeight = 40);

    // 20260322 ZJH 创建统计卡片控件
    // 参数: strTitle - 卡片标题; strValue - 卡片数值; strColor - 边框颜色
    // 返回: 创建的 QWidget 指针
    QWidget* createStatCard(const QString& strTitle, const QString& strValue,
                            const QString& strColor);

    // ===== 组件 =====

    ProjectManager*  m_pProjectManager;   // 20260322 ZJH 项目管理器（生命周期由本对象管理）
    QStackedWidget*  m_pViewStack;        // 20260322 ZJH 欢迎屏/项目信息双视图切换

    // --- 欢迎屏控件 ---
    QListWidget*     m_pRecentList;       // 20260322 ZJH 最近项目列表

    // --- 项目信息视图控件 ---
    QLabel*          m_pLblProjectName;   // 20260322 ZJH 项目名称标签
    QLabel*          m_pLblTaskType;      // 20260322 ZJH 任务类型标签
    QLabel*          m_pLblProjectPath;   // 20260322 ZJH 项目路径标签
    QLabel*          m_pLblCreateTime;    // 20260322 ZJH 创建时间标签
    QLabel*          m_pLblImageCount;    // 20260322 ZJH 图像数量标签
    QLabel*          m_pLblLabeledCount;  // 20260322 ZJH 已标注数量标签
    QWidget*         m_pLabelListArea;    // 20260322 ZJH 标签列表区域

    // --- 统计卡片 ---
    QLabel*          m_pLblStatImages;    // 20260322 ZJH 图像数统计卡片值
    QLabel*          m_pLblStatLabeled;   // 20260322 ZJH 已标注统计卡片值
    QLabel*          m_pLblStatSplit;     // 20260322 ZJH 拆分状态统计卡片值
};
