// 20260322 ZJH MainWindow — OmniMatch 主窗口
// 组装导航栏（顶部）、页面堆叠（中央）、状态栏（底部）
// 管理全部工作流页面的切换动画和菜单栏操作（页面数由 NavigationBar::kPageCount 定义）

#pragma once

#include <QMainWindow>           // 20260322 ZJH 主窗口基类（含菜单栏/状态栏/中央控件）
#include <QStackedWidget>        // 20260322 ZJH 页面堆叠控件，同一时刻只显示一个页面
#include <QAction>               // 20260322 ZJH 菜单项动作
#include <QGraphicsOpacityEffect>  // 20260322 ZJH 页面切换淡入透明度效果
#include <QPropertyAnimation>    // 20260322 ZJH 驱动透明度动画
#include "app/NavigationBar.h"   // 20260324 ZJH 引入 NavigationBar::kPageCount 常量

// 20260322 ZJH 前向声明，避免头文件循环依赖
namespace OmniMatch {
class StatusBar;
}
class BasePage;
class ProjectPage;         // 20260322 ZJH 项目管理页面前向声明
class GalleryPage;         // 20260322 ZJH 图库浏览页面前向声明
class ImagePage;           // 20260322 ZJH 图像标注页面前向声明
class InspectionPage;      // 20260322 ZJH 数据检查页面前向声明
class SplitPage;           // 20260322 ZJH 数据集拆分页面前向声明
class TrainingPage;        // 20260322 ZJH 训练配置页面前向声明
class EvaluationPage;      // 20260322 ZJH 评估分析页面前向声明
class ExportPage;          // 20260322 ZJH 模型导出页面前向声明
class ProjectManager;      // 20260322 ZJH 项目管理器前向声明
class ShortcutHelpOverlay; // 20260322 ZJH 快捷键参考面板前向声明

// 20260322 ZJH OmniMatch 主窗口
// 布局：
//   [菜单栏] — setupMenuBar() 创建
//   [导航栏] — NavigationBar（8 个页签）
//   [页面区] — QStackedWidget（8 个 BasePage 页面）
//   [状态栏] — StatusBar（消息/进度/GPU 信息）
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化全部子组件并组装布局
    explicit MainWindow(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构，子控件由 Qt 对象树管理
    ~MainWindow() override = default;

protected:
    // 20260324 ZJH 窗口关闭事件拦截 — 检查未保存修改并提示用户
    void closeEvent(QCloseEvent* pEvent) override;

public slots:
    // 20260322 ZJH 切换到指定索引的页面（0-7）
    // 触发当前页面 onLeave() + 新页面 onEnter() + 淡入动画
    void switchToPage(int nIndex);

private slots:
    // ===== 菜单槽函数 =====

    // 20260322 ZJH 文件 > 新建项目（Phase 2 实现，当前占位）
    void onMenuNewProject();

    // 20260322 ZJH 文件 > 打开项目（Phase 2 实现，当前占位）
    void onMenuOpenProject();

    // 20260322 ZJH 文件 > 保存项目（Phase 2 实现，当前占位）
    void onMenuSaveProject();

    // 20260322 ZJH 文件 > 关闭项目（Phase 2 实现，当前占位）
    void onMenuCloseProject();

    // 20260322 ZJH 帮助 > 关于（弹出版本信息对话框）
    void onMenuAbout();

    // 20260322 ZJH 视图 > 设置（Phase 2 实现，当前占位）
    void onMenuSettings();

    // 20260322 ZJH 视图 > 切换主题（Dark ↔ Light 切换）
    void onMenuToggleTheme();

    // 20260322 ZJH 更新窗口标题（含项目名和页面名）
    void updateWindowTitle();

private:
    // 20260322 ZJH 创建菜单栏及全部菜单项
    void setupMenuBar();

    // 20260322 ZJH 初始化 8 个页面到 QStackedWidget
    void setupPages();

    // 20260322 ZJH 创建 8 个占位页面（每个页面包含一个居中 QLabel 显示页面名）
    void createPlaceholderPages();

    // ===== 核心组件 =====

    OmniMatch::NavigationBar* m_pNavBar;     // 20260322 ZJH 顶部导航栏（8 个页签按钮）
    QStackedWidget*           m_pPageStack;  // 20260322 ZJH 中央页面堆叠控件
    OmniMatch::StatusBar*     m_pStatusBar;  // 20260322 ZJH 底部状态栏

    // 20260324 ZJH 页面指针数组，大小引用 NavigationBar::kPageCount，消除魔数 8
    BasePage* m_arrPages[OmniMatch::NavigationBar::kPageCount];

    // 20260322 ZJH 项目管理页面指针（m_arrPages[0] 的具体类型，方便直接调用 ProjectPage 方法）
    ProjectPage* m_pProjectPage;

    // 20260322 ZJH 图库浏览页面指针（m_arrPages[1] 的具体类型，方便直接调用 GalleryPage 方法）
    GalleryPage* m_pGalleryPage;

    // 20260322 ZJH 图像标注页面指针（m_arrPages[2] 的具体类型）
    ImagePage* m_pImagePage;

    // 20260322 ZJH 数据检查页面指针（m_arrPages[3] 的具体类型，PageIndex::Inspection = 3）
    InspectionPage* m_pInspectionPage;

    // 20260322 ZJH 数据集拆分页面指针（m_arrPages[4] 的具体类型，PageIndex::Split = 4）
    SplitPage* m_pSplitPage;

    // 20260322 ZJH 训练配置页面指针（m_arrPages[5] 的具体类型，PageIndex::Training = 5）
    TrainingPage* m_pTrainingPage;

    // 20260322 ZJH 评估分析页面指针（m_arrPages[6] 的具体类型，PageIndex::Evaluation = 6）
    EvaluationPage* m_pEvaluationPage;

    // 20260322 ZJH 模型导出页面指针（m_arrPages[7] 的具体类型，PageIndex::Export = 7）
    ExportPage* m_pExportPage;

    // ===== 菜单 Actions =====

    QAction* m_pActNew;         // 20260322 ZJH 文件 > 新建（Ctrl+N）
    QAction* m_pActOpen;        // 20260322 ZJH 文件 > 打开（Ctrl+O）
    QAction* m_pActSave;        // 20260322 ZJH 文件 > 保存（Ctrl+S）
    QAction* m_pActClose;       // 20260322 ZJH 文件 > 关闭（Ctrl+W）
    QAction* m_pActExit;        // 20260322 ZJH 文件 > 退出（Ctrl+Q）
    QAction* m_pActUndo;        // 20260322 ZJH 编辑 > 撤销（Ctrl+Z）
    QAction* m_pActRedo;        // 20260322 ZJH 编辑 > 重做（Ctrl+Y）
    QAction* m_pActFullscreen;  // 20260322 ZJH 视图 > 全屏（F11）
    QAction* m_pActTheme;       // 20260322 ZJH 视图 > 切换主题
    QAction* m_pActHelp;        // 20260322 ZJH 帮助 > 快捷键参考（F1）
    QAction* m_pActAbout;       // 20260322 ZJH 帮助 > 关于

    // ===== 页面切换动画 =====

    QGraphicsOpacityEffect* m_pFadeEffect;  // 20260322 ZJH 页面堆叠的透明度效果
    QPropertyAnimation*     m_pFadeAnim;    // 20260322 ZJH 驱动透明度从 0 到 1 的动画

    // 20260322 ZJH 快捷键参考面板（F1 触发显示/隐藏）
    ShortcutHelpOverlay* m_pShortcutOverlay;

    // 20260322 ZJH 当前显示的页面索引（0-7）
    int m_nCurrentPage = 0;
};
