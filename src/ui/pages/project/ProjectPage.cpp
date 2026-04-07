// 20260322 ZJH ProjectPage 实现
// 欢迎屏（品牌 + 新建/打开/最近项目）与项目信息视图的双视图切换
// 集成 ProjectManager 实现项目全生命周期管理

#include "ui/pages/project/ProjectPage.h"           // 20260322 ZJH ProjectPage 类声明
#include "ui/pages/project/NewProjectDialog.h"      // 20260322 ZJH 新建项目对话框
#include "core/project/ProjectManager.h"             // 20260322 ZJH 项目管理器
#include "core/project/Project.h"                    // 20260322 ZJH Project 数据类
#include "core/project/ProjectSerializer.h"          // 20260324 ZJH 导入后自动保存
#include "core/DLTypes.h"                            // 20260322 ZJH 全局类型定义
#include "core/data/ImageDataset.h"                  // 20260322 ZJH 数据集管理
#include "core/data/LabelInfo.h"                     // 20260322 ZJH 标签信息
#include "app/Application.h"                         // 20260322 ZJH 全局事件总线

#include <QVBoxLayout>       // 20260322 ZJH 垂直布局
#include <QHBoxLayout>       // 20260322 ZJH 水平布局
#include <QLabel>            // 20260322 ZJH 文本标签
#include <QPushButton>       // 20260322 ZJH 按钮
#include <QListWidget>       // 20260322 ZJH 最近项目列表
#include <QFileDialog>       // 20260322 ZJH 文件/文件夹选择对话框
#include <QMessageBox>       // 20260322 ZJH 错误提示
#include <QDateTime>         // 20260322 ZJH 获取当前时间
#include <QScrollArea>       // 20260322 ZJH 可滚动区域（标签列表）
#include <QFrame>            // 20260322 ZJH 分割线
#include <QDebug>            // 20260322 ZJH 调试日志
#include <QFileInfo>         // 20260403 ZJH 文件路径解析（新建项目路径提取）

// 20260322 ZJH 构造函数：创建双视图布局和项目管理器
ProjectPage::ProjectPage(QWidget* pParent)
    : BasePage(pParent)               // 20260322 ZJH 初始化页面基类
    , m_pProjectManager(nullptr)      // 20260322 ZJH 项目管理器指针
    , m_pViewStack(nullptr)           // 20260322 ZJH 视图堆叠指针
    , m_pRecentList(nullptr)          // 20260322 ZJH 最近项目列表指针
    , m_pLblProjectName(nullptr)      // 20260322 ZJH 各信息标签初始化
    , m_pLblTaskType(nullptr)
    , m_pLblProjectPath(nullptr)
    , m_pLblCreateTime(nullptr)
    , m_pLblImageCount(nullptr)
    , m_pLblLabeledCount(nullptr)
    , m_pLabelListArea(nullptr)
    , m_pLblStatImages(nullptr)
    , m_pLblStatLabeled(nullptr)
    , m_pLblStatSplit(nullptr)
{
    // 20260322 ZJH 创建项目管理器（父对象设为 this，生命周期跟随页面）
    m_pProjectManager = new ProjectManager(this);

    // 20260322 ZJH 创建主布局
    QVBoxLayout* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(0, 0, 0, 0);  // 20260322 ZJH 无边距
    pMainLayout->setSpacing(0);                     // 20260322 ZJH 无间距

    // 20260322 ZJH 创建视图堆叠（欢迎屏 / 项目信息）
    m_pViewStack = new QStackedWidget(this);

    // 20260322 ZJH 视图 0：欢迎屏
    m_pViewStack->addWidget(createWelcomeView());

    // 20260322 ZJH 视图 1：项目信息
    m_pViewStack->addWidget(createProjectInfoView());

    // 20260322 ZJH 默认显示欢迎屏
    m_pViewStack->setCurrentIndex(0);

    pMainLayout->addWidget(m_pViewStack);  // 20260322 ZJH 视图堆叠占满整个页面

    // 20260324 ZJH 信号连接已移至 MainWindow::createPlaceholderPages() 中统一循环处理
    // 所有页面（包括 ProjectPage）的 projectCreated/projectOpened/projectClosed 信号
    // 均由 MainWindow 统一连接到 BasePage::onProjectLoaded/onProjectClosed
    // 此处不再重复连接，避免 onProjectLoaded/onProjectClosed 被调用两次
}

// ===== 生命周期回调 =====

// 20260322 ZJH 页面切换到前台时刷新内容
void ProjectPage::onEnter()
{
    // 20260322 ZJH 根据当前是否有项目决定显示哪个视图
    if (Application::instance()->hasValidProject()) {
        // 20260322 ZJH 有项目则显示项目信息视图
        m_pViewStack->setCurrentIndex(1);
        refreshProjectInfo();  // 20260322 ZJH 刷新项目详情
    } else {
        // 20260322 ZJH 无项目则显示欢迎屏
        m_pViewStack->setCurrentIndex(0);
        refreshRecentList();  // 20260322 ZJH 刷新最近项目列表
    }
}

// 20260324 ZJH 项目加载后切换到项目信息视图（Template Method 扩展点）
// 基类已完成 m_pProject 赋值，此处无需手动设置
void ProjectPage::onProjectLoadedImpl()
{
    m_pViewStack->setCurrentIndex(1);  // 20260322 ZJH 切换到项目信息视图
    refreshProjectInfo();               // 20260322 ZJH 刷新显示内容
}

// 20260324 ZJH 项目关闭时切换回欢迎屏（Template Method 扩展点）
// 基类将在此方法返回后清空 m_pProject
void ProjectPage::onProjectClosedImpl()
{
    m_pViewStack->setCurrentIndex(0);  // 20260322 ZJH 切换到欢迎屏
    refreshRecentList();                // 20260322 ZJH 刷新最近项目列表
}

// ===== 公共操作 =====

// 20260322 ZJH 触发新建项目（供 MainWindow 菜单调用）
void ProjectPage::triggerNewProject()
{
    onNewProject();  // 20260322 ZJH 委托给内部槽函数
}

// 20260322 ZJH 触发打开项目（供 MainWindow 菜单调用）
void ProjectPage::triggerOpenProject()
{
    onOpenProject();  // 20260322 ZJH 委托给内部槽函数
}

// ===== 槽函数 =====

// 20260403 ZJH 新建项目按钮点击（Halcon 风格对话框）
void ProjectPage::onNewProject()
{
    // 20260403 ZJH 弹出 Halcon 风格新建项目对话框
    NewProjectDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) {
        return;  // 20260403 ZJH 用户取消，不创建项目
    }

    // 20260403 ZJH 从对话框获取用户输入
    QString strName = dlg.projectName();                // 20260403 ZJH 项目名称
    om::TaskType eType = dlg.taskType();                // 20260403 ZJH 任务类型
    QString strFilePath = dlg.projectPath();            // 20260403 ZJH 项目文件路径（.omdl）
    QString strDescription = dlg.projectDescription();  // 20260403 ZJH 项目描述

    // 20260403 ZJH 从 .omdl 文件路径提取项目目录（去掉文件名）
    QFileInfo fileInfo(strFilePath);                        // 20260403 ZJH 解析文件路径信息
    QString strProjectDir = fileInfo.absolutePath() + "/" + strName;  // 20260403 ZJH 项目目录

    // 20260403 ZJH 调用 ProjectManager 创建项目（含描述）
    Project* pProject = m_pProjectManager->createProject(
        strName, eType, strProjectDir, strDescription);

    if (pProject) {
        qDebug() << "[ProjectPage] onNewProject: 项目创建成功" << strName;  // 20260403 ZJH 成功日志
    } else {
        // 20260403 ZJH 创建失败提示
        QMessageBox::warning(this,
            QStringLiteral("创建失败"),
            QStringLiteral("无法创建项目，请检查路径权限。"));
    }
}

// 20260322 ZJH 打开项目按钮点击
void ProjectPage::onOpenProject()
{
    // 20260322 ZJH 弹出文件对话框选择 .dfproj 文件
    QString strFilePath = QFileDialog::getOpenFileName(
        this,                                                // 20260322 ZJH 父窗口
        QStringLiteral("打开项目"),                           // 20260322 ZJH 对话框标题
        QString(),                                           // 20260322 ZJH 初始目录（默认）
        QStringLiteral("OmniMatch 项目 (*.dfproj)")          // 20260322 ZJH 文件过滤器
    );

    // 20260322 ZJH 用户取消
    if (strFilePath.isEmpty()) {
        return;  // 20260322 ZJH 用户未选择文件
    }

    // 20260322 ZJH 调用 ProjectManager 打开项目
    Project* pProject = m_pProjectManager->openProject(strFilePath);

    if (!pProject) {
        // 20260322 ZJH 打开失败提示
        QMessageBox::warning(this,
            QStringLiteral("打开失败"),
            QStringLiteral("无法打开项目文件，文件可能已损坏。"));
    }
}

// 20260322 ZJH 打开文件夹按钮点击
// 自动创建分类项目并导入文件夹中的图像
void ProjectPage::onOpenFolder()
{
    // 20260322 ZJH 弹出文件夹选择对话框
    QString strFolderPath = QFileDialog::getExistingDirectory(
        this,                                    // 20260322 ZJH 父窗口
        QStringLiteral("选择图像文件夹"),          // 20260322 ZJH 对话框标题
        QString(),                               // 20260322 ZJH 初始目录（默认）
        QFileDialog::ShowDirsOnly                // 20260322 ZJH 仅显示目录
    );

    // 20260322 ZJH 用户取消
    if (strFolderPath.isEmpty()) {
        return;  // 20260322 ZJH 用户未选择文件夹
    }

    // 20260322 ZJH 从文件夹名提取项目名称
    QDir dir(strFolderPath);  // 20260322 ZJH 文件夹对象
    QString strName = dir.dirName();  // 20260322 ZJH 文件夹名作为项目名

    // 20260322 ZJH 默认创建分类任务项目
    Project* pProject = m_pProjectManager->createProject(
        strName,
        om::TaskType::Classification,
        strFolderPath
    );

    if (pProject) {
        // 20260322 ZJH 导入文件夹中的图像
        pProject->dataset()->importFromFolder(strFolderPath, true);  // 20260322 ZJH 递归导入
        qDebug() << "[ProjectPage] onOpenFolder: 导入图像"
                 << pProject->dataset()->imageCount() << "张";  // 20260322 ZJH 导入日志

        // 20260324 ZJH 导入完成后自动保存项目，确保 .dfproj 包含图像条目
        QString strProjFile = pProject->path() + "/" + pProject->name() + ".dfproj";
        ProjectSerializer::save(pProject, strProjFile);
        pProject->setDirty(false);

        // 20260322 ZJH 刷新项目信息显示
        refreshProjectInfo();
    }
}

// 20260322 ZJH 最近项目列表双击打开
void ProjectPage::onRecentProjectDoubleClicked(QListWidgetItem* pItem)
{
    if (!pItem) {
        return;  // 20260322 ZJH 空项，忽略
    }

    // 20260322 ZJH 从列表项的 data 获取文件路径
    QString strFilePath = pItem->data(Qt::UserRole).toString();  // 20260322 ZJH 读取关联路径

    if (strFilePath.isEmpty()) {
        return;  // 20260322 ZJH 路径为空，忽略
    }

    // 20260322 ZJH 调用 ProjectManager 打开
    Project* pProject = m_pProjectManager->openProject(strFilePath);

    if (!pProject) {
        // 20260322 ZJH 打开失败，提示用户
        QMessageBox::warning(this,
            QStringLiteral("打开失败"),
            QStringLiteral("无法打开项目文件:\n%1").arg(strFilePath));
    }
}

// 20260322 ZJH 跳转到画廊页（导入图像）
void ProjectPage::onGoToGallery()
{
    // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
    Application::instance()->notifyNavigateToPage(om::PageIndex::Gallery);
}

// 20260322 ZJH 跳转到图像页（开始标注）
void ProjectPage::onGoToImage()
{
    // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
    Application::instance()->notifyNavigateToPage(om::PageIndex::Image);
}

// 20260322 ZJH 跳转到拆分页（查看拆分）
void ProjectPage::onGoToSplit()
{
    // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
    Application::instance()->notifyNavigateToPage(om::PageIndex::Split);
}

// ===== 视图构建 =====

// 20260322 ZJH 创建欢迎屏视图
QWidget* ProjectPage::createWelcomeView()
{
    // 20260322 ZJH 欢迎屏容器
    QWidget* pWelcome = new QWidget(this);
    pWelcome->setStyleSheet(QStringLiteral(
        "QWidget { background-color: #1a1d23; }"  // 20260322 ZJH 深色背景
    ));

    QVBoxLayout* pLayout = new QVBoxLayout(pWelcome);
    pLayout->setContentsMargins(0, 0, 0, 0);  // 20260322 ZJH 无边距
    pLayout->setSpacing(0);

    // 20260322 ZJH 上半部分弹性空间
    pLayout->addStretch(2);

    // 20260322 ZJH 1. 大标题 "OmniMatch"
    QLabel* pLblTitle = new QLabel(QStringLiteral("OmniMatch"), pWelcome);
    pLblTitle->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 居中
    pLblTitle->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: white;"            // 20260322 ZJH 白色文字
        "  font-size: 28pt;"         // 20260322 ZJH 28 号字体
        "  font-weight: bold;"       // 20260322 ZJH 粗体
        "  background: transparent;" // 20260322 ZJH 透明背景
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(pLblTitle);

    // 20260322 ZJH 2. 副标题
    QLabel* pLblSubtitle = new QLabel(
        QStringLiteral("Pure C++ Deep Learning Vision Platform"), pWelcome);
    pLblSubtitle->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 居中
    pLblSubtitle->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #64748b;"          // 20260322 ZJH 灰色文字
        "  font-size: 12pt;"         // 20260322 ZJH 12 号字体
        "  background: transparent;"
        "  border: none;"
        "  margin-top: 4px;"
        "}"
    ));
    pLayout->addWidget(pLblSubtitle);

    // 20260324 ZJH 3. 版本号（使用 CMake 注入的 OM_VERSION 宏）
    QLabel* pLblVersion = new QLabel(QStringLiteral("v") + QStringLiteral(OM_VERSION), pWelcome);
    pLblVersion->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 居中
    pLblVersion->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #475569;"          // 20260322 ZJH 暗灰色文字
        "  font-size: 10pt;"         // 20260322 ZJH 10 号字体
        "  background: transparent;"
        "  border: none;"
        "  margin-top: 2px;"
        "}"
    ));
    pLayout->addWidget(pLblVersion);

    // 20260322 ZJH 间距
    pLayout->addSpacing(32);

    // 20260322 ZJH 4. 三个大按钮（水平排列，居中）
    QHBoxLayout* pBtnLayout = new QHBoxLayout();
    pBtnLayout->setSpacing(16);  // 20260322 ZJH 按钮间距
    pBtnLayout->addStretch(1);   // 20260322 ZJH 左侧弹性空间

    // 20260322 ZJH "新建项目" 按钮
    QPushButton* pBtnNew = createStyledButton(QStringLiteral("新建项目"), "#2563eb");
    connect(pBtnNew, &QPushButton::clicked, this, &ProjectPage::onNewProject);
    pBtnLayout->addWidget(pBtnNew);

    // 20260322 ZJH "打开项目" 按钮
    QPushButton* pBtnOpen = createStyledButton(QStringLiteral("打开项目"), "#2563eb");
    connect(pBtnOpen, &QPushButton::clicked, this, &ProjectPage::onOpenProject);
    pBtnLayout->addWidget(pBtnOpen);

    // 20260322 ZJH "打开文件夹" 按钮
    QPushButton* pBtnFolder = createStyledButton(QStringLiteral("打开文件夹"), "#2563eb");
    connect(pBtnFolder, &QPushButton::clicked, this, &ProjectPage::onOpenFolder);
    pBtnLayout->addWidget(pBtnFolder);

    pBtnLayout->addStretch(1);  // 20260322 ZJH 右侧弹性空间
    pLayout->addLayout(pBtnLayout);

    // 20260322 ZJH 间距
    pLayout->addSpacing(32);

    // 20260322 ZJH 5. 最近项目列表标题
    QLabel* pLblRecent = new QLabel(QStringLiteral("最近项目"), pWelcome);
    pLblRecent->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 居中
    pLblRecent->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #94a3b8;"          // 20260322 ZJH 灰色文字
        "  font-size: 11pt;"         // 20260322 ZJH 11 号字体
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(pLblRecent);

    // 20260322 ZJH 间距
    pLayout->addSpacing(8);

    // 20260322 ZJH 6. 最近项目列表（水平居中，固定宽度）
    QHBoxLayout* pListLayout = new QHBoxLayout();
    pListLayout->addStretch(1);  // 20260322 ZJH 左侧弹性

    m_pRecentList = new QListWidget(pWelcome);
    m_pRecentList->setFixedSize(400, 200);  // 20260322 ZJH 固定大小
    m_pRecentList->setStyleSheet(QStringLiteral(
        "QListWidget {"
        "  background-color: #262a35;"   // 20260322 ZJH 暗色背景
        "  color: #e2e8f0;"              // 20260322 ZJH 浅色文字
        "  border: 1px solid #3b4252;"   // 20260322 ZJH 边框
        "  border-radius: 6px;"
        "  padding: 4px;"
        "  font-size: 10pt;"
        "}"
        "QListWidget::item {"
        "  padding: 6px 12px;"
        "  border-radius: 3px;"
        "}"
        "QListWidget::item:hover {"
        "  background-color: #2a3040;"   // 20260322 ZJH 悬停高亮
        "}"
        "QListWidget::item:selected {"
        "  background-color: #2563eb;"   // 20260322 ZJH 选中蓝色
        "}"
    ));
    // 20260322 ZJH 连接双击信号
    connect(m_pRecentList, &QListWidget::itemDoubleClicked,
            this, &ProjectPage::onRecentProjectDoubleClicked);

    pListLayout->addWidget(m_pRecentList);
    pListLayout->addStretch(1);  // 20260322 ZJH 右侧弹性
    pLayout->addLayout(pListLayout);

    // 20260322 ZJH 下半部分弹性空间
    pLayout->addStretch(3);

    return pWelcome;  // 20260322 ZJH 返回欢迎屏控件
}

// 20260322 ZJH 创建项目信息视图
QWidget* ProjectPage::createProjectInfoView()
{
    // 20260322 ZJH 项目信息容器
    QWidget* pInfoView = new QWidget(this);
    pInfoView->setStyleSheet(QStringLiteral(
        "QWidget { background-color: #1a1d23; }"  // 20260322 ZJH 深色背景
    ));

    QHBoxLayout* pMainLayout = new QHBoxLayout(pInfoView);
    pMainLayout->setContentsMargins(0, 0, 0, 0);
    pMainLayout->setSpacing(0);

    // ===== 左侧面板：项目详情 =====
    QWidget* pLeftPanel = new QWidget(pInfoView);
    pLeftPanel->setFixedWidth(300);  // 20260322 ZJH 固定宽度 300px
    pLeftPanel->setStyleSheet(QStringLiteral(
        "QWidget {"
        "  background-color: #1e2230;"  // 20260322 ZJH 稍亮的深色背景
        "  border-right: 1px solid #2a2d35;"  // 20260322 ZJH 右侧分割线
        "}"
    ));

    QVBoxLayout* pLeftLayout = new QVBoxLayout(pLeftPanel);
    pLeftLayout->setContentsMargins(20, 24, 20, 20);  // 20260322 ZJH 内边距
    pLeftLayout->setSpacing(12);  // 20260322 ZJH 控件间距

    // 20260322 ZJH 项目名称（大字）
    m_pLblProjectName = new QLabel(QStringLiteral("-"), pLeftPanel);
    m_pLblProjectName->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: white;"            // 20260322 ZJH 白色
        "  font-size: 18pt;"         // 20260322 ZJH 大号字
        "  font-weight: bold;"       // 20260322 ZJH 粗体
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    m_pLblProjectName->setWordWrap(true);  // 20260322 ZJH 长名称自动换行
    pLeftLayout->addWidget(m_pLblProjectName);

    // 20260322 ZJH 分割线
    QFrame* pLine1 = new QFrame(pLeftPanel);
    pLine1->setFrameShape(QFrame::HLine);  // 20260322 ZJH 水平线
    pLine1->setStyleSheet(QStringLiteral("background-color: #3b4252; border: none; max-height: 1px;"));
    pLeftLayout->addWidget(pLine1);

    // 20260322 ZJH 任务类型
    QLabel* pLblTaskTitle = new QLabel(QStringLiteral("任务类型"), pLeftPanel);
    pLblTaskTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(pLblTaskTitle);

    m_pLblTaskType = new QLabel(QStringLiteral("-"), pLeftPanel);
    m_pLblTaskType->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 11pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(m_pLblTaskType);

    // 20260322 ZJH 项目路径
    QLabel* pLblPathTitle = new QLabel(QStringLiteral("项目路径"), pLeftPanel);
    pLblPathTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(pLblPathTitle);

    m_pLblProjectPath = new QLabel(QStringLiteral("-"), pLeftPanel);
    m_pLblProjectPath->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 9pt; background: transparent; border: none; }"));
    m_pLblProjectPath->setWordWrap(true);  // 20260322 ZJH 长路径自动换行
    pLeftLayout->addWidget(m_pLblProjectPath);

    // 20260322 ZJH 创建时间
    QLabel* pLblTimeTitle = new QLabel(QStringLiteral("创建时间"), pLeftPanel);
    pLblTimeTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(pLblTimeTitle);

    m_pLblCreateTime = new QLabel(QStringLiteral("-"), pLeftPanel);
    m_pLblCreateTime->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 9pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(m_pLblCreateTime);

    // 20260322 ZJH 分割线
    QFrame* pLine2 = new QFrame(pLeftPanel);
    pLine2->setFrameShape(QFrame::HLine);
    pLine2->setStyleSheet(QStringLiteral("background-color: #3b4252; border: none; max-height: 1px;"));
    pLeftLayout->addWidget(pLine2);

    // 20260322 ZJH 图像数量 / 已标注数量
    m_pLblImageCount = new QLabel(QStringLiteral("图像: 0"), pLeftPanel);
    m_pLblImageCount->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 10pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(m_pLblImageCount);

    m_pLblLabeledCount = new QLabel(QStringLiteral("已标注: 0"), pLeftPanel);
    m_pLblLabeledCount->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 10pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(m_pLblLabeledCount);

    // 20260322 ZJH 分割线
    QFrame* pLine3 = new QFrame(pLeftPanel);
    pLine3->setFrameShape(QFrame::HLine);
    pLine3->setStyleSheet(QStringLiteral("background-color: #3b4252; border: none; max-height: 1px;"));
    pLeftLayout->addWidget(pLine3);

    // 20260322 ZJH 标签列表区域
    QLabel* pLblLabelsTitle = new QLabel(QStringLiteral("标签列表"), pLeftPanel);
    pLblLabelsTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; background: transparent; border: none; }"));
    pLeftLayout->addWidget(pLblLabelsTitle);

    // 20260322 ZJH 标签列表可滚动区域
    QScrollArea* pScrollArea = new QScrollArea(pLeftPanel);
    pScrollArea->setWidgetResizable(true);  // 20260322 ZJH 内容控件自动适应
    pScrollArea->setStyleSheet(QStringLiteral(
        "QScrollArea { background: transparent; border: none; }"));

    m_pLabelListArea = new QWidget(pScrollArea);
    m_pLabelListArea->setStyleSheet(QStringLiteral(
        "QWidget { background: transparent; }"));
    QVBoxLayout* pLabelLayout = new QVBoxLayout(m_pLabelListArea);
    pLabelLayout->setContentsMargins(0, 0, 0, 0);
    pLabelLayout->setSpacing(4);

    // 20260322 ZJH 初始提示：无标签
    QLabel* pNoLabels = new QLabel(QStringLiteral("暂无标签"), m_pLabelListArea);
    pNoLabels->setStyleSheet(QStringLiteral(
        "QLabel { color: #475569; font-size: 9pt; background: transparent; border: none; }"));
    pLabelLayout->addWidget(pNoLabels);
    pLabelLayout->addStretch();

    pScrollArea->setWidget(m_pLabelListArea);
    pLeftLayout->addWidget(pScrollArea, 1);  // 20260322 ZJH stretch=1，占满剩余空间

    pMainLayout->addWidget(pLeftPanel);

    // ===== 中央面板：项目概览 =====
    QWidget* pCenterPanel = new QWidget(pInfoView);
    pCenterPanel->setStyleSheet(QStringLiteral(
        "QWidget { background-color: #1a1d23; }"));  // 20260322 ZJH 深色背景

    QVBoxLayout* pCenterLayout = new QVBoxLayout(pCenterPanel);
    pCenterLayout->setContentsMargins(40, 40, 40, 40);  // 20260322 ZJH 四周大边距
    pCenterLayout->setSpacing(24);  // 20260322 ZJH 控件间距

    // 20260322 ZJH 概览标题
    QLabel* pLblOverview = new QLabel(QStringLiteral("项目概览"), pCenterPanel);
    pLblOverview->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: white;"            // 20260322 ZJH 白色
        "  font-size: 16pt;"         // 20260322 ZJH 大号字
        "  font-weight: bold;"       // 20260322 ZJH 粗体
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pCenterLayout->addWidget(pLblOverview);

    // 20260322 ZJH 三个统计卡片（水平排列）
    QHBoxLayout* pCardLayout = new QHBoxLayout();
    pCardLayout->setSpacing(16);  // 20260322 ZJH 卡片间距

    // 20260322 ZJH 图像数统计卡片
    QWidget* pCardImages = createStatCard(QStringLiteral("图像总数"), "0", "#2563eb");
    m_pLblStatImages = pCardImages->findChild<QLabel*>("statValue");  // 20260322 ZJH 通过 objectName 查找值标签
    pCardLayout->addWidget(pCardImages);

    // 20260322 ZJH 已标注统计卡片
    QWidget* pCardLabeled = createStatCard(QStringLiteral("已标注"), "0", "#16a34a");
    m_pLblStatLabeled = pCardLabeled->findChild<QLabel*>("statValue");
    pCardLayout->addWidget(pCardLabeled);

    // 20260322 ZJH 拆分状态统计卡片
    QWidget* pCardSplit = createStatCard(QStringLiteral("拆分状态"), QStringLiteral("未拆分"), "#f59e0b");
    m_pLblStatSplit = pCardSplit->findChild<QLabel*>("statValue");
    pCardLayout->addWidget(pCardSplit);

    pCenterLayout->addLayout(pCardLayout);

    // 20260322 ZJH 分割线
    QFrame* pLine4 = new QFrame(pCenterPanel);
    pLine4->setFrameShape(QFrame::HLine);
    pLine4->setStyleSheet(QStringLiteral("background-color: #2a2d35; border: none; max-height: 1px;"));
    pCenterLayout->addWidget(pLine4);

    // 20260322 ZJH 快捷操作标题
    QLabel* pLblActions = new QLabel(QStringLiteral("快捷操作"), pCenterPanel);
    pLblActions->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #94a3b8;"          // 20260322 ZJH 灰色
        "  font-size: 12pt;"         // 20260322 ZJH 12 号
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pCenterLayout->addWidget(pLblActions);

    // 20260322 ZJH 三个快捷按钮（水平排列）
    QHBoxLayout* pActionLayout = new QHBoxLayout();
    pActionLayout->setSpacing(16);  // 20260322 ZJH 按钮间距

    // 20260322 ZJH "导入图像" 按钮
    QPushButton* pBtnGallery = createStyledButton(QStringLiteral("导入图像"), "#2563eb", 140, 44);
    connect(pBtnGallery, &QPushButton::clicked, this, &ProjectPage::onGoToGallery);
    pActionLayout->addWidget(pBtnGallery);

    // 20260322 ZJH "开始标注" 按钮
    QPushButton* pBtnAnnotate = createStyledButton(QStringLiteral("开始标注"), "#16a34a", 140, 44);
    connect(pBtnAnnotate, &QPushButton::clicked, this, &ProjectPage::onGoToImage);
    pActionLayout->addWidget(pBtnAnnotate);

    // 20260322 ZJH "查看拆分" 按钮
    QPushButton* pBtnSplit = createStyledButton(QStringLiteral("查看拆分"), "#f59e0b", 140, 44);
    connect(pBtnSplit, &QPushButton::clicked, this, &ProjectPage::onGoToSplit);
    pActionLayout->addWidget(pBtnSplit);

    pActionLayout->addStretch(1);  // 20260322 ZJH 右侧弹性
    pCenterLayout->addLayout(pActionLayout);

    // 20260322 ZJH 底部弹性空间
    pCenterLayout->addStretch(1);

    pMainLayout->addWidget(pCenterPanel, 1);  // 20260322 ZJH stretch=1，占满剩余空间

    return pInfoView;  // 20260322 ZJH 返回项目信息视图
}

// ===== 辅助方法 =====

// 20260322 ZJH 刷新最近项目列表
void ProjectPage::refreshRecentList()
{
    if (!m_pRecentList) {
        return;  // 20260322 ZJH 列表控件未初始化
    }

    m_pRecentList->clear();  // 20260322 ZJH 清空现有列表

    // 20260322 ZJH 从 ProjectManager 获取最近项目路径
    QStringList vecRecent = m_pProjectManager->recentProjects();

    if (vecRecent.isEmpty()) {
        // 20260322 ZJH 无最近项目，显示提示
        QListWidgetItem* pItem = new QListWidgetItem(QStringLiteral("暂无最近项目"));
        pItem->setFlags(pItem->flags() & ~Qt::ItemIsSelectable);  // 20260322 ZJH 不可选
        m_pRecentList->addItem(pItem);
        return;  // 20260322 ZJH 提前返回
    }

    // 20260322 ZJH 逐个添加最近项目到列表
    for (const QString& strPath : vecRecent) {
        QFileInfo fi(strPath);  // 20260322 ZJH 文件信息
        // 20260322 ZJH 显示文件名（不含扩展名）和路径
        QString strDisplay = fi.completeBaseName() + QStringLiteral("  —  ") + fi.absolutePath();

        QListWidgetItem* pItem = new QListWidgetItem(strDisplay);
        pItem->setData(Qt::UserRole, strPath);  // 20260322 ZJH 关联完整路径
        pItem->setToolTip(strPath);             // 20260322 ZJH 悬停显示完整路径
        m_pRecentList->addItem(pItem);
    }
}

// 20260322 ZJH 刷新项目信息显示
void ProjectPage::refreshProjectInfo()
{
    // 20260322 ZJH 获取当前项目
    Project* pProject = Application::instance()->currentProject();
    if (!pProject) {
        return;  // 20260322 ZJH 无项目，不刷新
    }

    // 20260322 ZJH 更新左侧面板信息
    m_pLblProjectName->setText(pProject->name());  // 20260322 ZJH 项目名称
    m_pLblTaskType->setText(om::taskTypeToString(pProject->taskType()));  // 20260322 ZJH 任务类型中文名
    m_pLblProjectPath->setText(pProject->path());  // 20260322 ZJH 项目路径

    // 20260322 ZJH 创建时间（使用当前时间，项目文件中暂无存储）
    m_pLblCreateTime->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm"));

    // 20260322 ZJH 数据集统计
    const ImageDataset* pDataset = pProject->dataset();
    int nImageCount = pDataset ? pDataset->imageCount() : 0;      // 20260322 ZJH 图像数量
    int nLabeledCount = pDataset ? pDataset->labeledCount() : 0;  // 20260322 ZJH 已标注数量

    m_pLblImageCount->setText(QStringLiteral("图像: %1").arg(nImageCount));
    m_pLblLabeledCount->setText(QStringLiteral("已标注: %1").arg(nLabeledCount));

    // 20260322 ZJH 更新统计卡片
    if (m_pLblStatImages) {
        m_pLblStatImages->setText(QString::number(nImageCount));  // 20260322 ZJH 图像总数
    }
    if (m_pLblStatLabeled) {
        m_pLblStatLabeled->setText(QString::number(nLabeledCount));  // 20260322 ZJH 已标注数
    }
    if (m_pLblStatSplit) {
        // 20260322 ZJH 判断拆分状态
        int nTrain = pDataset ? pDataset->countBySplit(om::SplitType::Train) : 0;
        int nVal   = pDataset ? pDataset->countBySplit(om::SplitType::Validation) : 0;
        int nTest  = pDataset ? pDataset->countBySplit(om::SplitType::Test) : 0;
        if (nTrain > 0 || nVal > 0 || nTest > 0) {
            m_pLblStatSplit->setText(QStringLiteral("%1/%2/%3").arg(nTrain).arg(nVal).arg(nTest));
        } else {
            m_pLblStatSplit->setText(QStringLiteral("未拆分"));  // 20260322 ZJH 尚未拆分
        }
    }

    // 20260322 ZJH 更新标签列表区域
    if (m_pLabelListArea && pDataset) {
        // 20260322 ZJH 清除旧内容
        QLayout* pOldLayout = m_pLabelListArea->layout();
        if (pOldLayout) {
            // 20260322 ZJH 移除所有子控件
            QLayoutItem* pChild;
            while ((pChild = pOldLayout->takeAt(0)) != nullptr) {
                if (pChild->widget()) {
                    delete pChild->widget();  // 20260322 ZJH 删除控件
                }
                delete pChild;  // 20260322 ZJH 删除布局项
            }
        } else {
            // 20260322 ZJH 首次创建布局
            pOldLayout = new QVBoxLayout(m_pLabelListArea);
            static_cast<QVBoxLayout*>(pOldLayout)->setContentsMargins(0, 0, 0, 0);
            static_cast<QVBoxLayout*>(pOldLayout)->setSpacing(4);
        }

        // 20260322 ZJH 重新填充标签列表
        const auto& vecLabels = pDataset->labels();
        if (vecLabels.isEmpty()) {
            // 20260322 ZJH 无标签
            QLabel* pNoLabels = new QLabel(QStringLiteral("暂无标签"), m_pLabelListArea);
            pNoLabels->setStyleSheet(QStringLiteral(
                "QLabel { color: #475569; font-size: 9pt; background: transparent; border: none; }"));
            pOldLayout->addWidget(pNoLabels);
        } else {
            // 20260322 ZJH 逐个添加标签（颜色圆点 + 名称）
            for (const LabelInfo& label : vecLabels) {
                QHBoxLayout* pLabelRow = new QHBoxLayout();
                pLabelRow->setSpacing(8);  // 20260322 ZJH 圆点和名称间距

                // 20260322 ZJH 颜色圆点
                QLabel* pDot = new QLabel(m_pLabelListArea);
                pDot->setFixedSize(12, 12);  // 20260322 ZJH 12x12 像素
                pDot->setStyleSheet(QStringLiteral(
                    "QLabel {"
                    "  background-color: %1;"
                    "  border-radius: 6px;"  // 20260322 ZJH 圆形
                    "  border: none;"
                    "}"
                ).arg(label.color.name()));
                pLabelRow->addWidget(pDot);

                // 20260322 ZJH 标签名称
                QLabel* pLblName = new QLabel(label.strName, m_pLabelListArea);
                pLblName->setStyleSheet(QStringLiteral(
                    "QLabel { color: #e2e8f0; font-size: 9pt; background: transparent; border: none; }"));
                pLabelRow->addWidget(pLblName, 1);  // 20260322 ZJH stretch=1

                // 20260322 ZJH 将水平行包装到容器控件中
                QWidget* pRowWidget = new QWidget(m_pLabelListArea);
                pRowWidget->setStyleSheet("background: transparent;");
                pRowWidget->setLayout(pLabelRow);
                pOldLayout->addWidget(pRowWidget);
            }
        }

        // 20260322 ZJH 底部弹性空间
        static_cast<QVBoxLayout*>(pOldLayout)->addStretch();
    }
}

// 20260322 ZJH 创建样式化的大按钮
QPushButton* ProjectPage::createStyledButton(const QString& strText, const QString& strColor,
                                              int nWidth, int nHeight)
{
    QPushButton* pBtn = new QPushButton(strText, this);
    pBtn->setFixedSize(nWidth, nHeight);  // 20260322 ZJH 固定大小
    pBtn->setCursor(Qt::PointingHandCursor);  // 20260322 ZJH 手型光标
    pBtn->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: %1;"    // 20260322 ZJH 主色
        "  color: white;"            // 20260322 ZJH 白色文字
        "  border: none;"
        "  border-radius: 6px;"      // 20260322 ZJH 圆角
        "  font-size: 10pt;"         // 20260322 ZJH 字号
        "  font-weight: 500;"
        "}"
        "QPushButton:hover {"
        "  background-color: %1;"    // 20260322 ZJH 悬停同色（透明度由 opacity 处理）
        "  opacity: 0.9;"
        "}"
        "QPushButton:pressed {"
        "  background-color: %1;"    // 20260322 ZJH 按下同色
        "  opacity: 0.8;"
        "}"
    ).arg(strColor));

    return pBtn;  // 20260322 ZJH 返回创建的按钮
}

// 20260322 ZJH 创建统计卡片控件
QWidget* ProjectPage::createStatCard(const QString& strTitle, const QString& strValue,
                                      const QString& strColor)
{
    // 20260322 ZJH 卡片容器
    QWidget* pCard = new QWidget(this);
    pCard->setFixedHeight(100);  // 20260322 ZJH 固定高度
    pCard->setMinimumWidth(150); // 20260322 ZJH 最小宽度
    pCard->setStyleSheet(QStringLiteral(
        "QWidget {"
        "  background-color: #262a35;"     // 20260322 ZJH 暗色卡片背景
        "  border: 1px solid #3b4252;"     // 20260322 ZJH 边框
        "  border-left: 3px solid %1;"     // 20260322 ZJH 左侧彩色条
        "  border-radius: 8px;"            // 20260322 ZJH 圆角
        "}"
    ).arg(strColor));

    QVBoxLayout* pLayout = new QVBoxLayout(pCard);
    pLayout->setContentsMargins(16, 12, 16, 12);  // 20260322 ZJH 内边距
    pLayout->setSpacing(8);

    // 20260322 ZJH 卡片标题
    QLabel* pLblTitle = new QLabel(strTitle, pCard);
    pLblTitle->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #94a3b8;"          // 20260322 ZJH 灰色标题
        "  font-size: 9pt;"
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(pLblTitle);

    // 20260322 ZJH 卡片数值
    QLabel* pLblValue = new QLabel(strValue, pCard);
    pLblValue->setObjectName("statValue");  // 20260322 ZJH 设置 objectName 以便 findChild 查找
    pLblValue->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: white;"            // 20260322 ZJH 白色数值
        "  font-size: 20pt;"         // 20260322 ZJH 大号字
        "  font-weight: bold;"       // 20260322 ZJH 粗体
        "  background: transparent;"
        "  border: none;"
        "}"
    ));
    pLayout->addWidget(pLblValue);

    pLayout->addStretch();  // 20260322 ZJH 底部弹性

    return pCard;  // 20260322 ZJH 返回卡片控件
}
