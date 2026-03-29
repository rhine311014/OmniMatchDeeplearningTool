// 20260322 ZJH MainWindow 实现
// 组装导航栏/页面堆叠/状态栏，配置菜单栏，实现页面切换动画

#include "app/MainWindow.h"          // 20260322 ZJH MainWindow 类声明
#include "app/Application.h"         // 20260322 ZJH 全局事件总线单例
#include "app/ThemeManager.h"        // 20260322 ZJH 主题管理器（切换主题）
#include "app/NavigationBar.h"       // 20260322 ZJH 顶部导航栏
#include "app/StatusBar.h"           // 20260322 ZJH 底部状态栏
#include "ui/pages/BasePage.h"       // 20260322 ZJH 页面基类
#include "ui/pages/project/ProjectPage.h"  // 20260322 ZJH 项目管理页面
#include "ui/pages/gallery/GalleryPage.h"  // 20260322 ZJH 图库浏览页面
#include "ui/pages/image/ImagePage.h"      // 20260322 ZJH 图像标注页面
#include "ui/pages/inspection/InspectionPage.h"  // 20260322 ZJH 数据检查页面（第 3 页）
#include "ui/pages/split/SplitPage.h"             // 20260322 ZJH 数据集拆分页面（第 4 页）
#include "ui/pages/training/TrainingPage.h"       // 20260322 ZJH 训练配置页面（第 5 页）
#include "ui/pages/evaluation/EvaluationPage.h"   // 20260322 ZJH 评估分析页面（第 6 页）
#include "ui/pages/export/ExportPage.h"            // 20260322 ZJH 模型导出页面（第 7 页）
#include "core/project/ProjectManager.h"     // 20260322 ZJH 项目管理器
#include "core/project/Project.h"            // 20260322 ZJH 项目数据类
#include "core/project/ProjectSerializer.h"  // 20260322 ZJH 项目序列化器（菜单保存用）
#include "ui/dialogs/SettingsDialog.h"       // 20260322 ZJH 设置对话框
#include "ui/widgets/ShortcutHelpOverlay.h"  // 20260322 ZJH 快捷键参考面板

#include <QVBoxLayout>     // 20260322 ZJH 垂直布局
#include <QMenuBar>        // 20260322 ZJH 菜单栏
#include <QMenu>           // 20260322 ZJH 菜单
#include <QMessageBox>     // 20260322 ZJH 对话框（关于/占位提示）
#include <QLabel>          // 20260322 ZJH 占位页面中的文本标签
#include <QEasingCurve>    // 20260322 ZJH 动画缓动曲线
#include <QFileDialog>     // 20260322 ZJH 文件对话框（保存项目另存为）
#include <QCloseEvent>     // 20260324 ZJH 窗口关闭事件

// 20260324 ZJH 页签总数快捷别名（引用 NavigationBar::kPageCount，消除魔数 8）
static constexpr int s_nPageCount = OmniMatch::NavigationBar::kPageCount;

// 20260322 ZJH 构造函数，初始化主窗口全部组件
// 20260324 ZJH 初始化列表顺序严格匹配 MainWindow.h 中成员声明顺序
MainWindow::MainWindow(QWidget* pParent)
    : QMainWindow(pParent)
    , m_pNavBar(nullptr)       // 20260322 ZJH 导航栏指针，setupPages 中创建
    , m_pPageStack(nullptr)    // 20260322 ZJH 页面堆叠指针
    , m_pStatusBar(nullptr)    // 20260322 ZJH 状态栏指针
    , m_pProjectPage(nullptr)  // 20260324 ZJH 项目管理页面指针（与头文件声明顺序一致）
    , m_pGalleryPage(nullptr)  // 20260324 ZJH 图库浏览页面指针
    , m_pImagePage(nullptr)    // 20260324 ZJH 图像标注页面指针
    , m_pInspectionPage(nullptr) // 20260324 ZJH 数据检查页面指针
    , m_pSplitPage(nullptr)    // 20260324 ZJH 数据集拆分页面指针
    , m_pTrainingPage(nullptr)   // 20260324 ZJH 训练配置页面指针
    , m_pEvaluationPage(nullptr) // 20260324 ZJH 评估分析页面指针
    , m_pExportPage(nullptr)     // 20260324 ZJH 模型导出页面指针
    , m_pActNew(nullptr)       // 20260324 ZJH 菜单 Action 初始化（与头文件声明顺序一致）
    , m_pActOpen(nullptr)
    , m_pActSave(nullptr)
    , m_pActClose(nullptr)
    , m_pActExit(nullptr)
    , m_pActUndo(nullptr)
    , m_pActRedo(nullptr)
    , m_pActFullscreen(nullptr)
    , m_pActTheme(nullptr)
    , m_pActHelp(nullptr)
    , m_pActAbout(nullptr)
    , m_pFadeEffect(nullptr)   // 20260322 ZJH 淡入效果指针
    , m_pFadeAnim(nullptr)     // 20260322 ZJH 淡入动画指针
    , m_pShortcutOverlay(nullptr) // 20260324 ZJH 快捷键参考面板指针（补充到初始化列表）
    , m_nCurrentPage(0)        // 20260322 ZJH 初始页面索引为 0（项目页）
{
    // 20260324 ZJH 初始化全部页面指针为空（使用 s_nPageCount 消除魔数）
    for (int i = 0; i < s_nPageCount; ++i) {
        m_arrPages[i] = nullptr;
    }

    // 20260328 ZJH 1. 设置窗口标题（使用 CMake 注入的 OM_VERSION 宏）
    setWindowTitle(QStringLiteral("OmniMatch Deep Learning Tool v") + QStringLiteral(OM_VERSION));

    // 20260322 ZJH 2. 设置初始窗口大小 1280x800（参考 MVTec DL Tool）
    resize(1280, 800);
    // 20260325 ZJH 设置最小窗口尺寸，防止缩小后内容被裁剪
    setMinimumSize(960, 600);

    // 20260322 ZJH 3. 创建菜单栏
    setupMenuBar();

    // 20260322 ZJH 4. 创建中央区域容器（导航栏 + 页面堆叠 + 状态栏垂直排列）
    QWidget* pCentralWidget = new QWidget(this);
    QVBoxLayout* pCentralLayout = new QVBoxLayout(pCentralWidget);
    pCentralLayout->setContentsMargins(0, 0, 0, 0);  // 20260322 ZJH 无边距，最大化利用空间
    pCentralLayout->setSpacing(0);                     // 20260322 ZJH 组件间无间距

    // 20260322 ZJH 5. 创建顶部导航栏
    m_pNavBar = new OmniMatch::NavigationBar(pCentralWidget);
    pCentralLayout->addWidget(m_pNavBar);  // 20260322 ZJH 导航栏固定高度 40px

    // 20260322 ZJH 6. 创建中央页面堆叠
    m_pPageStack = new QStackedWidget(pCentralWidget);
    pCentralLayout->addWidget(m_pPageStack, 1);  // 20260322 ZJH stretch=1，占满剩余空间

    // 20260322 ZJH 7. 创建底部状态栏
    m_pStatusBar = new OmniMatch::StatusBar(pCentralWidget);
    pCentralLayout->addWidget(m_pStatusBar);  // 20260322 ZJH 状态栏固定高度 28px

    // 20260322 ZJH 设置中央控件
    setCentralWidget(pCentralWidget);

    // 20260322 ZJH 8. 创建全部工作流页面
    createPlaceholderPages();

    // 20260322 ZJH 9. 连接导航栏页面切换信号到 switchToPage 槽
    connect(m_pNavBar, &OmniMatch::NavigationBar::pageChanged,
            this, &MainWindow::switchToPage);

    // 20260322 ZJH 10. 连接 Application 全局导航请求信号
    connect(Application::instance(), &Application::requestNavigateToPage,
            this, &MainWindow::switchToPage);

    // 20260322 ZJH 10b. 连接 requestOpenImage 信号：切换到图像页并加载图像
    connect(Application::instance(), &Application::requestOpenImage,
            this, [this](const QString& strImageUuid) {
                switchToPage(2);  // 20260322 ZJH 切换到图像标注页
                if (m_pImagePage) {
                    m_pImagePage->loadImage(strImageUuid);  // 20260322 ZJH 加载指定图像
                }
            });

    // 20260322 ZJH 11. 初始化页面切换淡入动画
    m_pFadeEffect = new QGraphicsOpacityEffect(m_pPageStack);
    m_pFadeEffect->setOpacity(1.0);  // 20260322 ZJH 初始完全不透明
    m_pPageStack->setGraphicsEffect(m_pFadeEffect);

    // 20260322 ZJH 创建 opacity 属性动画
    m_pFadeAnim = new QPropertyAnimation(m_pFadeEffect, "opacity", this);
    m_pFadeAnim->setDuration(200);                    // 20260322 ZJH 淡入持续 200ms
    m_pFadeAnim->setStartValue(0.0);                  // 20260322 ZJH 从完全透明开始
    m_pFadeAnim->setEndValue(1.0);                    // 20260322 ZJH 到完全不透明结束
    m_pFadeAnim->setEasingCurve(QEasingCurve::OutQuad);  // 20260322 ZJH 减速缓出

    // 20260322 ZJH 12. 初始化 GPU 信息到状态栏
    if (Application::instance()->hasGpu()) {
        // 20260322 ZJH 显示 GPU 名称和显存大小
        QString strGpuInfo = QStringLiteral("GPU: %1 | %2 MB")
            .arg(Application::instance()->gpuName())
            .arg(Application::instance()->gpuVramMB());
        m_pStatusBar->setGpuInfo(strGpuInfo);
    } else {
        m_pStatusBar->setGpuInfo(QStringLiteral("CPU Only"));  // 20260322 ZJH 无 GPU 时显示 CPU 模式
    }

    // 20260322 ZJH 13. 设置状态栏初始消息
    m_pStatusBar->setMessage(QStringLiteral("就绪"));

    // 20260322 ZJH 14. 创建快捷键参考面板（覆盖在中央控件之上）
    m_pShortcutOverlay = new ShortcutHelpOverlay(pCentralWidget);
    m_pShortcutOverlay->hide();  // 20260322 ZJH 初始隐藏

    // 20260322 ZJH 15. 更新窗口标题
    updateWindowTitle();
}

// ===== 菜单栏 =====

// 20260322 ZJH 创建菜单栏及全部菜单项
void MainWindow::setupMenuBar()
{
    // 20260322 ZJH 获取主窗口菜单栏（QMainWindow 自带，无需手动创建）
    QMenuBar* pMenuBar = menuBar();

    // 20260322 ZJH 设置菜单栏样式，与暗色主题协调
    pMenuBar->setStyleSheet(QStringLiteral(
        "QMenuBar {"
        "  background-color: #13151a;"    // 深色背景
        "  color: #e2e8f0;"               // 浅色文字
        "  border-bottom: 1px solid #1e2230;"  // 底部分割线
        "  padding: 2px 0px;"
        "}"
        "QMenuBar::item {"
        "  padding: 4px 10px;"
        "  background: transparent;"
        "}"
        "QMenuBar::item:selected {"
        "  background-color: #2563eb;"    // 选中蓝色高亮
        "  border-radius: 3px;"
        "}"
        "QMenu {"
        "  background-color: #1e2230;"    // 下拉菜单深色背景
        "  color: #e2e8f0;"               // 浅色文字
        "  border: 1px solid #2a2d35;"    // 边框
        "  padding: 4px 0px;"
        "}"
        "QMenu::item {"
        "  padding: 6px 24px;"
        "}"
        "QMenu::item:selected {"
        "  background-color: #2563eb;"    // 选中蓝色高亮
        "}"
        "QMenu::separator {"
        "  height: 1px;"
        "  background: #2a2d35;"          // 分割线颜色
        "  margin: 4px 8px;"
        "}"
    ));

    // ===== 文件菜单 =====
    QMenu* pFileMenu = pMenuBar->addMenu(QStringLiteral("文件(&F)"));

    // 20260322 ZJH 新建项目 Ctrl+N
    m_pActNew = pFileMenu->addAction(QStringLiteral("新建项目(&N)"));
    m_pActNew->setShortcut(QKeySequence(QStringLiteral("Ctrl+N")));
    connect(m_pActNew, &QAction::triggered, this, &MainWindow::onMenuNewProject);

    // 20260322 ZJH 打开项目 Ctrl+O
    m_pActOpen = pFileMenu->addAction(QStringLiteral("打开项目(&O)"));
    m_pActOpen->setShortcut(QKeySequence(QStringLiteral("Ctrl+O")));
    connect(m_pActOpen, &QAction::triggered, this, &MainWindow::onMenuOpenProject);

    // 20260322 ZJH 保存项目 Ctrl+S
    m_pActSave = pFileMenu->addAction(QStringLiteral("保存项目(&S)"));
    m_pActSave->setShortcut(QKeySequence(QStringLiteral("Ctrl+S")));
    connect(m_pActSave, &QAction::triggered, this, &MainWindow::onMenuSaveProject);

    // 20260322 ZJH 关闭项目 Ctrl+W
    m_pActClose = pFileMenu->addAction(QStringLiteral("关闭项目(&C)"));
    m_pActClose->setShortcut(QKeySequence(QStringLiteral("Ctrl+W")));
    connect(m_pActClose, &QAction::triggered, this, &MainWindow::onMenuCloseProject);

    // 20260322 ZJH 分割线
    pFileMenu->addSeparator();

    // 20260322 ZJH 退出 Ctrl+Q
    m_pActExit = pFileMenu->addAction(QStringLiteral("退出(&X)"));
    m_pActExit->setShortcut(QKeySequence(QStringLiteral("Ctrl+Q")));
    connect(m_pActExit, &QAction::triggered, this, &QMainWindow::close);

    // ===== 编辑菜单 =====
    QMenu* pEditMenu = pMenuBar->addMenu(QStringLiteral("编辑(&E)"));

    // 20260322 ZJH 撤销 Ctrl+Z
    m_pActUndo = pEditMenu->addAction(QStringLiteral("撤销(&U)"));
    m_pActUndo->setShortcut(QKeySequence(QStringLiteral("Ctrl+Z")));
    m_pActUndo->setEnabled(false);  // 20260322 ZJH Phase 2 实现，当前禁用

    // 20260322 ZJH 重做 Ctrl+Y
    m_pActRedo = pEditMenu->addAction(QStringLiteral("重做(&R)"));
    m_pActRedo->setShortcut(QKeySequence(QStringLiteral("Ctrl+Y")));
    m_pActRedo->setEnabled(false);  // 20260322 ZJH Phase 2 实现，当前禁用

    // ===== 视图菜单 =====
    QMenu* pViewMenu = pMenuBar->addMenu(QStringLiteral("视图(&V)"));

    // 20260322 ZJH 全屏切换 F11
    m_pActFullscreen = pViewMenu->addAction(QStringLiteral("全屏(&F)"));
    m_pActFullscreen->setShortcut(QKeySequence(Qt::Key_F11));
    connect(m_pActFullscreen, &QAction::triggered, this, [this]() {
        // 20260322 ZJH 切换全屏/正常状态
        if (isFullScreen()) {
            showNormal();  // 20260322 ZJH 退出全屏
        } else {
            showFullScreen();  // 20260322 ZJH 进入全屏
        }
    });

    // 20260322 ZJH 切换主题
    m_pActTheme = pViewMenu->addAction(QStringLiteral("切换主题(&T)"));
    connect(m_pActTheme, &QAction::triggered, this, &MainWindow::onMenuToggleTheme);

    // 20260322 ZJH 分割线
    pViewMenu->addSeparator();

    // 20260322 ZJH 设置
    QAction* pActSettings = pViewMenu->addAction(QStringLiteral("设置(&S)..."));
    connect(pActSettings, &QAction::triggered, this, &MainWindow::onMenuSettings);

    // ===== 帮助菜单 =====
    QMenu* pHelpMenu = pMenuBar->addMenu(QStringLiteral("帮助(&H)"));

    // 20260322 ZJH 快捷键参考 F1
    m_pActHelp = pHelpMenu->addAction(QStringLiteral("快捷键参考(&K)"));
    m_pActHelp->setShortcut(QKeySequence(Qt::Key_F1));
    connect(m_pActHelp, &QAction::triggered, this, [this]() {
        // 20260322 ZJH 显示 ShortcutHelpOverlay 快捷键参考面板
        if (m_pShortcutOverlay) {
            m_pShortcutOverlay->showOverlay();
        }
    });

    // 20260322 ZJH 关于
    m_pActAbout = pHelpMenu->addAction(QStringLiteral("关于(&A)"));
    connect(m_pActAbout, &QAction::triggered, this, &MainWindow::onMenuAbout);
}

// ===== 页面管理 =====

// 20260324 ZJH 创建全部页面，具体页面按索引分别实例化，其余为 BasePage 占位
void MainWindow::createPlaceholderPages()
{
    // 20260324 ZJH 按索引实例化各具体页面，并保存到 m_arrPages 和类型指针
    // 页面 0: ProjectPage
    m_pProjectPage = new ProjectPage(m_pPageStack);
    m_arrPages[0] = m_pProjectPage;
    m_pPageStack->addWidget(m_pProjectPage);

    // 20260324 ZJH 页面 1: GalleryPage
    m_pGalleryPage = new GalleryPage(m_pPageStack);
    m_arrPages[1] = m_pGalleryPage;
    m_pPageStack->addWidget(m_pGalleryPage);

    // 20260324 ZJH 页面 2: ImagePage
    m_pImagePage = new ImagePage(m_pPageStack);
    m_arrPages[2] = m_pImagePage;
    m_pPageStack->addWidget(m_pImagePage);

    // 20260324 ZJH 页面 3: InspectionPage
    m_pInspectionPage = new InspectionPage(m_pPageStack);
    m_arrPages[3] = m_pInspectionPage;
    m_pPageStack->addWidget(m_pInspectionPage);

    // 20260324 ZJH 页面 4: SplitPage
    m_pSplitPage = new SplitPage(m_pPageStack);
    m_arrPages[4] = m_pSplitPage;
    m_pPageStack->addWidget(m_pSplitPage);

    // 20260324 ZJH 页面 5: TrainingPage
    m_pTrainingPage = new TrainingPage(m_pPageStack);
    m_arrPages[5] = m_pTrainingPage;
    m_pPageStack->addWidget(m_pTrainingPage);

    // 20260324 ZJH 页面 6: EvaluationPage
    m_pEvaluationPage = new EvaluationPage(m_pPageStack);
    m_arrPages[6] = m_pEvaluationPage;
    m_pPageStack->addWidget(m_pEvaluationPage);

    // 20260324 ZJH 页面 7: ExportPage
    m_pExportPage = new ExportPage(m_pPageStack);
    m_arrPages[7] = m_pExportPage;
    m_pPageStack->addWidget(m_pExportPage);

    // 20260324 ZJH 统一循环连接项目生命周期信号到所有页面（消除重复信号连接样板代码）
    // projectCreated/projectOpened → onProjectLoaded, projectClosed → onProjectClosed
    for (int i = 0; i < s_nPageCount; ++i) {
        BasePage* pPage = m_arrPages[i];  // 20260324 ZJH 取当前页面基类指针
        if (!pPage) {
            continue;  // 20260324 ZJH 空指针保护
        }
        // 20260324 ZJH 项目新建时通知页面加载
        connect(Application::instance(), &Application::projectCreated,
                pPage, &BasePage::onProjectLoaded);
        // 20260324 ZJH 项目打开时通知页面加载
        connect(Application::instance(), &Application::projectOpened,
                pPage, &BasePage::onProjectLoaded);
        // 20260324 ZJH 项目关闭时通知页面清理
        connect(Application::instance(), &Application::projectClosed,
                pPage, &BasePage::onProjectClosed);
    }

    // 20260324 ZJH 项目加载后连接数据集变更信号到脏标志 + 窗口标题更新
    // 当项目数据集发生任何修改时，自动标记项目为脏并刷新标题栏显示 " *"
    auto fnConnectDirty = [this](Project* pProject) {
        if (!pProject || !pProject->dataset()) {
            return;  // 20260324 ZJH 空指针保护
        }
        // 20260324 ZJH 数据集内容变更 → 标记脏 → 刷新标题
        connect(pProject->dataset(), &ImageDataset::dataChanged, this, [this, pProject]() {
            pProject->setDirty(true);   // 20260324 ZJH 数据变更，标记为脏
            updateWindowTitle();         // 20260324 ZJH 刷新标题显示脏标志
        });
        // 20260324 ZJH 标签列表变更 → 标记脏 → 刷新标题
        connect(pProject->dataset(), &ImageDataset::labelsChanged, this, [this, pProject]() {
            pProject->setDirty(true);   // 20260324 ZJH 标签变更，标记为脏
            updateWindowTitle();         // 20260324 ZJH 刷新标题显示脏标志
        });
        // 20260324 ZJH 数据集拆分变更 → 标记脏 → 刷新标题
        connect(pProject->dataset(), &ImageDataset::splitChanged, this, [this, pProject]() {
            pProject->setDirty(true);   // 20260324 ZJH 拆分变更，标记为脏
            updateWindowTitle();         // 20260324 ZJH 刷新标题显示脏标志
        });
    };
    // 20260324 ZJH 项目新建时绑定脏标志信号
    connect(Application::instance(), &Application::projectCreated, this, fnConnectDirty);
    // 20260324 ZJH 项目打开时绑定脏标志信号
    connect(Application::instance(), &Application::projectOpened,  this, fnConnectDirty);
    // 20260324 ZJH 项目保存后刷新标题（移除脏标志 " *"）
    connect(Application::instance(), &Application::projectSaved,   this, &MainWindow::updateWindowTitle);

    // 20260322 ZJH 默认显示第一个页面（项目页）
    m_pPageStack->setCurrentIndex(0);
}

// 20260322 ZJH 切换到指定索引的页面
// 流程：当前页 onLeave → 切换 → 新页 onEnter → 淡入动画 → 更新导航栏和标题
void MainWindow::switchToPage(int nIndex)
{
    // 20260324 ZJH 边界检查，索引必须在 0 ~ s_nPageCount-1 范围内
    if (nIndex < 0 || nIndex >= s_nPageCount) {
        return;  // 20260322 ZJH 无效索引，忽略
    }

    // 20260322 ZJH 与当前页面相同，无需切换
    if (nIndex == m_nCurrentPage) {
        return;
    }

    // 20260322 ZJH 1. 调用当前页面的 onLeave() 生命周期回调
    if (m_arrPages[m_nCurrentPage]) {
        m_arrPages[m_nCurrentPage]->onLeave();
    }

    // 20260322 ZJH 2. 切换 QStackedWidget 的当前索引
    m_pPageStack->setCurrentIndex(nIndex);

    // 20260322 ZJH 3. 调用新页面的 onEnter() 生命周期回调
    if (m_arrPages[nIndex]) {
        m_arrPages[nIndex]->onEnter();
    }

    // 20260322 ZJH 4. 播放淡入动画（opacity 0 → 1，200ms）
    if (m_pFadeAnim) {
        m_pFadeAnim->stop();       // 20260322 ZJH 停止可能正在进行的动画
        m_pFadeAnim->start();      // 20260322 ZJH 重新开始淡入动画
    }

    // 20260322 ZJH 5. 更新 NavigationBar 选中状态（不触发 pageChanged 信号）
    m_pNavBar->setCurrentIndex(nIndex);

    // 20260322 ZJH 6. 记录当前页面索引
    m_nCurrentPage = nIndex;

    // 20260322 ZJH 7. 更新窗口标题
    updateWindowTitle();
}

// ===== 菜单槽函数 =====

// 20260322 ZJH 新建项目 — 委托给 ProjectPage 处理
void MainWindow::onMenuNewProject()
{
    // 20260322 ZJH 先切换到项目页
    switchToPage(0);
    // 20260322 ZJH 调用 ProjectPage 的新建项目逻辑
    if (m_pProjectPage) {
        m_pProjectPage->triggerNewProject();
    }
}

// 20260322 ZJH 打开项目 — 委托给 ProjectPage 处理
void MainWindow::onMenuOpenProject()
{
    // 20260322 ZJH 先切换到项目页
    switchToPage(0);
    // 20260322 ZJH 调用 ProjectPage 的打开项目逻辑
    if (m_pProjectPage) {
        m_pProjectPage->triggerOpenProject();
    }
}

// 20260322 ZJH 保存项目 — 调用 ProjectManager 保存当前项目
void MainWindow::onMenuSaveProject()
{
    // 20260322 ZJH 检查是否有项目可保存
    Project* pProject = Application::instance()->currentProject();
    if (!pProject) {
        QMessageBox::information(this,
            QStringLiteral("保存项目"),
            QStringLiteral("当前没有打开的项目。"));
        return;  // 20260322 ZJH 无项目可保存
    }

    // 20260322 ZJH 构建项目文件路径
    QString strProjFile = pProject->path() + "/" + pProject->name() + ".dfproj";

    // 20260322 ZJH 使用 ProjectSerializer 直接保存
    bool bOk = ProjectSerializer::save(pProject, strProjFile);
    if (bOk) {
        pProject->setDirty(false);  // 20260324 ZJH 保存成功，重置脏标志
        updateWindowTitle();  // 20260324 ZJH 刷新窗口标题（移除脏标志 " *"）
        // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
        Application::instance()->notifyProjectSaved();
        QMessageBox::information(this,
            QStringLiteral("保存成功"),
            QStringLiteral("项目已保存到:\n%1").arg(strProjFile));
    } else {
        QMessageBox::warning(this,
            QStringLiteral("保存失败"),
            QStringLiteral("无法保存项目文件。"));
    }
}

// 20260322 ZJH 关闭项目 — 关闭当前项目并切换到欢迎屏
void MainWindow::onMenuCloseProject()
{
    // 20260322 ZJH 检查是否有项目可关闭
    if (!Application::instance()->hasValidProject()) {
        QMessageBox::information(this,
            QStringLiteral("关闭项目"),
            QStringLiteral("当前没有打开的项目。"));
        return;  // 20260322 ZJH 无项目可关闭
    }

    // 20260322 ZJH 关闭项目（Application 将释放 Project 对象）
    Application::instance()->setCurrentProject(nullptr);
    // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
    Application::instance()->notifyProjectClosed();

    // 20260322 ZJH 切换到项目页（显示欢迎屏）
    switchToPage(0);
}

// 20260322 ZJH 关于对话框 — 显示版本和项目信息
void MainWindow::onMenuAbout()
{
    // 20260328 ZJH 关于对话框
    QMessageBox::about(this,
        QStringLiteral("关于 OmniMatch"),
        QStringLiteral(
            "<h2>OmniMatch Deep Learning Tool v") + QStringLiteral(OM_VERSION) + QStringLiteral("</h2>"
            "<p>纯 C++ 深度学习视觉平台</p>"
            "<hr>"
            "<p>特性：</p>"
            "<ul>"
            "<li>纯 C++ 实现的深度学习引擎（无 PyTorch/TensorFlow 依赖）</li>"
            "<li>自动微分 + 张量运算 + GPU 加速</li>"
            "<li>分类/检测/分割/异常检测全流程支持</li>"
            "<li>ONNX/TensorRT 模型导出</li>"
            "</ul>"
            "<p>&copy; 2026 OmniMatch Team</p>"
        ));
}

// 20260322 ZJH 设置 — 弹出 SettingsDialog 对话框
void MainWindow::onMenuSettings()
{
    // 20260322 ZJH 创建并显示设置对话框（模态）
    SettingsDialog dlg(this);
    dlg.exec();
}

// 20260322 ZJH 切换主题（Dark ↔ Light）
void MainWindow::onMenuToggleTheme()
{
    // 20260322 ZJH 获取当前主题，切换到对立主题
    auto* pThemeMgr = OmniMatch::ThemeManager::instance();
    if (pThemeMgr->currentTheme() == OmniMatch::ThemeManager::Theme::Dark) {
        pThemeMgr->applyTheme(OmniMatch::ThemeManager::Theme::Light);  // 20260322 ZJH 切换到亮色主题
    } else {
        pThemeMgr->applyTheme(OmniMatch::ThemeManager::Theme::Dark);   // 20260322 ZJH 切换到暗色主题
    }
}

// 20260328 ZJH 更新窗口标题，格式："OmniMatch Deep Learning Tool vX.Y.Z — [当前页面名] *"
// 当项目存在未保存修改时，标题末尾追加 " *" 脏标志
void MainWindow::updateWindowTitle()
{
    // 20260328 ZJH 构建标题字符串（使用 CMake 注入的 OM_VERSION 宏）
    QString strTitle = QStringLiteral("OmniMatch Deep Learning Tool v") + QStringLiteral(OM_VERSION);

    // 20260324 ZJH 通过 NavigationBar::pageName 获取当前页面中文名称（消除重复数组）
    const QString strPageName = OmniMatch::NavigationBar::pageName(m_nCurrentPage);
    if (!strPageName.isEmpty()) {
        strTitle += QStringLiteral(" — ") + strPageName;
    }

    // 20260324 ZJH 当项目有未保存修改时，标题末尾追加 " *" 作为脏状态指示
    Project* pProject = Application::instance()->currentProject();  // 20260324 ZJH 获取当前项目
    if (pProject && pProject->isDirty()) {
        strTitle += QStringLiteral(" *");  // 20260324 ZJH 脏标志指示符
    }

    setWindowTitle(strTitle);  // 20260322 ZJH 设置窗口标题
}

// 20260324 ZJH 窗口关闭事件拦截 — 未保存修改时提示用户选择保存/放弃/取消
void MainWindow::closeEvent(QCloseEvent* pEvent)
{
    Project* pProject = Application::instance()->currentProject();

    // 20260324 ZJH 有项目且有未保存修改时弹出确认对话框
    if (pProject && pProject->isDirty()) {
        QMessageBox::StandardButton eBtn = QMessageBox::question(
            this,
            QStringLiteral("未保存的修改"),
            QStringLiteral("项目 \"%1\" 有未保存的修改。\n\n是否在退出前保存？").arg(pProject->name()),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save  // 20260324 ZJH 默认选中"保存"
        );

        if (eBtn == QMessageBox::Save) {
            // 20260324 ZJH 用户选择保存 — 执行保存操作
            QString strProjFile = pProject->path() + "/" + pProject->name() + ".dfproj";
            bool bOk = ProjectSerializer::save(pProject, strProjFile);
            if (bOk) {
                pProject->setDirty(false);
            } else {
                // 20260324 ZJH 保存失败 — 提示用户，不关闭窗口
                QMessageBox::warning(this, QStringLiteral("保存失败"),
                    QStringLiteral("无法保存项目文件，请重试。"));
                pEvent->ignore();  // 20260324 ZJH 取消关闭
                return;
            }
        } else if (eBtn == QMessageBox::Cancel) {
            // 20260324 ZJH 用户取消 — 不关闭窗口
            pEvent->ignore();
            return;
        }
        // 20260324 ZJH Discard — 放弃修改直接关闭
    }

    pEvent->accept();  // 20260324 ZJH 允许关闭
}
