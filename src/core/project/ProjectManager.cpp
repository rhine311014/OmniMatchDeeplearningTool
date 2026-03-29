// 20260322 ZJH ProjectManager 实现
// 项目创建/打开/保存/关闭全流程 + 最近项目列表管理

#include "core/project/ProjectManager.h"  // 20260322 ZJH ProjectManager 类声明
#include "core/project/Project.h"          // 20260322 ZJH Project 数据类
#include "core/project/ProjectSerializer.h"  // 20260322 ZJH 项目序列化工具
#include "core/DLTypes.h"                  // 20260322 ZJH TaskType 枚举
#include "app/Application.h"               // 20260322 ZJH 全局事件总线单例

#include <QSettings>   // 20260322 ZJH 持久化存储最近项目列表
#include <QDir>        // 20260322 ZJH 目录操作
#include <QFileInfo>   // 20260322 ZJH 文件路径信息
#include <QDebug>      // 20260322 ZJH 调试日志

#include <memory>      // 20260322 ZJH std::unique_ptr

// 20260322 ZJH 构造函数：加载最近项目列表
ProjectManager::ProjectManager(QObject* pParent)
    : QObject(pParent)  // 20260322 ZJH 初始化 QObject 基类
{
    // 20260322 ZJH 从 QSettings 读取最近项目列表
    loadRecentFromSettings();
}

// ===== 项目操作 =====

// 20260322 ZJH 创建新项目
// 流程：1. 创建 Project 对象 → 2. 设置元数据 → 3. 创建项目目录 → 4. 保存 .dfproj 文件 → 5. 通知全局
Project* ProjectManager::createProject(const QString& strName, om::TaskType eType, const QString& strPath)
{
    // 20260322 ZJH 创建新的 Project 实例
    auto pProject = std::make_unique<Project>();

    // 20260322 ZJH 设置项目元数据
    pProject->setName(strName);          // 20260322 ZJH 项目名称
    pProject->setTaskType(eType);        // 20260322 ZJH 任务类型
    pProject->setPath(strPath);          // 20260322 ZJH 项目路径

    // 20260322 ZJH 确保项目目录存在
    QDir dir(strPath);  // 20260322 ZJH 目标目录
    if (!dir.exists()) {
        // 20260322 ZJH 目录不存在则创建（含所有父目录）
        dir.mkpath(".");
    }

    // 20260324 ZJH 创建项目内 images/ 子目录，用于存放导入的图像文件
    QDir imagesDir(strPath + QStringLiteral("/images"));  // 20260324 ZJH images 子目录
    if (!imagesDir.exists()) {
        imagesDir.mkpath(QStringLiteral("."));  // 20260324 ZJH 确保 images/ 目录存在
    }

    // 20260324 ZJH 设置数据集的项目路径，使后续导入操作自动复制图像到项目目录
    pProject->dataset()->setProjectPath(strPath);

    // 20260322 ZJH 构建 .dfproj 文件路径
    QString strProjFile = strPath + "/" + strName + ".dfproj";  // 20260322 ZJH 项目文件完整路径

    // 20260322 ZJH 序列化项目到磁盘
    bool bSaveOk = ProjectSerializer::save(pProject.get(), strProjFile);  // 20260322 ZJH 调用序列化器保存
    if (!bSaveOk) {
        qDebug() << "[ProjectManager] createProject: 保存失败" << strProjFile;  // 20260322 ZJH 保存失败日志
        // 20260324 ZJH 保存失败时返回空指针，避免使用未持久化的项目
        return nullptr;  // 20260324 ZJH 保存失败，拒绝创建项目
    }

    // 20260322 ZJH 添加到最近项目列表
    addToRecent(strProjFile);

    // 20260322 ZJH 获取裸指针用于信号参数（所有权即将转移）
    Project* pRawProject = pProject.get();

    // 20260322 ZJH 将项目所有权转交给 Application 单例
    Application::instance()->setCurrentProject(std::move(pProject));

    // 20260322 ZJH 发射本地信号
    emit projectCreated(pRawProject);

    // 20260324 ZJH 通过 notify 方法发射全局事件总线信号，避免外部直接 emit
    Application::instance()->notifyProjectCreated(pRawProject);

    qDebug() << "[ProjectManager] createProject: 成功" << strName << "at" << strPath;  // 20260322 ZJH 成功日志

    return pRawProject;  // 20260322 ZJH 返回项目裸指针（所有权已由 Application 持有）
}

// 20260322 ZJH 打开已有项目
// 流程：1. 反序列化 .dfproj 文件 → 2. 转交给 Application → 3. 通知全局
Project* ProjectManager::openProject(const QString& strFilePath)
{
    // 20260322 ZJH 检查文件是否存在
    QFileInfo fi(strFilePath);  // 20260322 ZJH 文件信息
    if (!fi.exists() || !fi.isFile()) {
        qDebug() << "[ProjectManager] openProject: 文件不存在" << strFilePath;  // 20260322 ZJH 文件不存在日志
        return nullptr;  // 20260322 ZJH 文件不存在，返回空
    }

    // 20260322 ZJH 反序列化项目文件
    auto pProject = ProjectSerializer::load(strFilePath);
    if (!pProject) {
        qDebug() << "[ProjectManager] openProject: 反序列化失败" << strFilePath;  // 20260322 ZJH 反序列化失败日志
        return nullptr;  // 20260322 ZJH 反序列化失败，返回空
    }

    // 20260325 ZJH 始终用 .dfproj 文件所在目录作为项目路径
    // 旧逻辑仅在 path 为空时覆盖，但 JSON 中存的旧绝对路径可能已过时
    // （用户移动了项目文件夹、换了磁盘盘符、或在不同电脑打开）
    // 用实际文件位置覆盖，确保相对路径重定位始终基于正确的项目根目录
    pProject->setPath(fi.absolutePath());

    // 20260324 ZJH 设置数据集的项目路径，使后续导入操作自动复制图像到项目目录
    pProject->dataset()->setProjectPath(pProject->path());

    // 20260322 ZJH 添加到最近项目列表
    addToRecent(strFilePath);

    // 20260322 ZJH 获取裸指针
    Project* pRawProject = pProject.get();

    // 20260322 ZJH 转交所有权给 Application
    Application::instance()->setCurrentProject(std::move(pProject));

    // 20260322 ZJH 发射本地信号
    emit projectOpened(pRawProject);

    // 20260324 ZJH 通过 notify 方法发射全局事件总线信号，避免外部直接 emit
    Application::instance()->notifyProjectOpened(pRawProject);

    qDebug() << "[ProjectManager] openProject: 成功" << strFilePath;  // 20260322 ZJH 成功日志

    return pRawProject;  // 20260322 ZJH 返回项目裸指针
}

// 20260322 ZJH 保存项目到磁盘
// 使用项目路径 + 项目名 + ".dfproj" 构建保存路径
bool ProjectManager::saveProject(Project* pProject)
{
    // 20260322 ZJH 空指针检查
    if (!pProject) {
        qDebug() << "[ProjectManager] saveProject: 项目指针为空";  // 20260322 ZJH 空指针日志
        return false;  // 20260322 ZJH 无法保存空项目
    }

    // 20260322 ZJH 构建 .dfproj 文件路径
    QString strProjFile = pProject->path() + "/" + pProject->name() + ".dfproj";  // 20260322 ZJH 完整路径

    // 20260322 ZJH 调用序列化器保存
    bool bOk = ProjectSerializer::save(pProject, strProjFile);  // 20260322 ZJH 保存到磁盘

    if (bOk) {
        // 20260322 ZJH 保存成功，发射信号
        emit projectSaved();
        // 20260324 ZJH 通过 notify 方法发射全局信号，避免外部直接 emit
        Application::instance()->notifyProjectSaved();

        qDebug() << "[ProjectManager] saveProject: 成功" << strProjFile;  // 20260322 ZJH 成功日志
    } else {
        qDebug() << "[ProjectManager] saveProject: 失败" << strProjFile;  // 20260322 ZJH 失败日志
    }

    return bOk;  // 20260322 ZJH 返回保存结果
}

// 20260322 ZJH 另存为：将项目保存到指定路径
bool ProjectManager::saveProjectAs(Project* pProject, const QString& strFilePath)
{
    // 20260322 ZJH 空指针检查
    if (!pProject) {
        qDebug() << "[ProjectManager] saveProjectAs: 项目指针为空";  // 20260322 ZJH 空指针日志
        return false;  // 20260322 ZJH 无法保存空项目
    }

    // 20260322 ZJH 调用序列化器保存到指定路径
    bool bOk = ProjectSerializer::save(pProject, strFilePath);  // 20260322 ZJH 保存到指定路径

    if (bOk) {
        // 20260322 ZJH 保存成功后更新项目路径
        QFileInfo fi(strFilePath);  // 20260322 ZJH 文件信息
        pProject->setPath(fi.absolutePath());  // 20260322 ZJH 更新项目路径为新文件所在目录

        // 20260322 ZJH 添加到最近项目列表
        addToRecent(strFilePath);

        // 20260322 ZJH 发射保存成功信号
        emit projectSaved();
        // 20260324 ZJH 通过 notify 方法发射全局信号，避免外部直接 emit
        Application::instance()->notifyProjectSaved();

        qDebug() << "[ProjectManager] saveProjectAs: 成功" << strFilePath;  // 20260322 ZJH 成功日志
    } else {
        qDebug() << "[ProjectManager] saveProjectAs: 失败" << strFilePath;  // 20260322 ZJH 失败日志
    }

    return bOk;  // 20260322 ZJH 返回保存结果
}

// 20260322 ZJH 关闭当前项目
// 将 Application 的当前项目置空，发射关闭信号
void ProjectManager::closeProject()
{
    // 20260323 ZJH 先发射关闭信号让所有页面断开 dataset 连接，再销毁项目
    emit projectClosed();
    // 20260324 ZJH 通过 notify 方法发射全局信号，避免外部直接 emit
    Application::instance()->notifyProjectClosed();

    // 20260323 ZJH 最后销毁项目（所有页面已在信号处理中清理了 dataset 指针）
    Application::instance()->setCurrentProject(nullptr);

    qDebug() << "[ProjectManager] closeProject: 项目已关闭";  // 20260322 ZJH 关闭日志
}

// ===== 最近项目管理 =====

// 20260322 ZJH 获取最近项目路径列表
QStringList ProjectManager::recentProjects() const
{
    return m_vecRecentPaths;  // 20260322 ZJH 返回最近项目列表副本
}

// 20260322 ZJH 添加项目路径到最近列表顶部
// 如果路径已存在则移到顶部，超出 MAX_RECENT 时移除末尾
void ProjectManager::addToRecent(const QString& strPath)
{
    // 20260322 ZJH 如果路径已在列表中，先移除（避免重复）
    m_vecRecentPaths.removeAll(strPath);

    // 20260322 ZJH 插入到列表头部（最新在前）
    m_vecRecentPaths.prepend(strPath);

    // 20260322 ZJH 限制列表长度不超过 MAX_RECENT
    while (m_vecRecentPaths.size() > MAX_RECENT) {
        m_vecRecentPaths.removeLast();  // 20260322 ZJH 移除最旧的项目路径
    }

    // 20260322 ZJH 持久化到 QSettings
    saveRecentToSettings();
}

// 20260322 ZJH 清空最近项目列表
void ProjectManager::clearRecent()
{
    m_vecRecentPaths.clear();  // 20260322 ZJH 清空内存列表
    saveRecentToSettings();    // 20260322 ZJH 同步到 QSettings
}

// ===== QSettings 持久化 =====

// 20260322 ZJH 从 QSettings 加载最近项目列表
void ProjectManager::loadRecentFromSettings()
{
    QSettings settings;  // 20260322 ZJH 使用默认 QSettings（基于 QApplication 的 orgName/appName）
    // 20260322 ZJH 读取 "recentProjects" 键，默认为空列表
    m_vecRecentPaths = settings.value("recentProjects").toStringList();
}

// 20260322 ZJH 将最近项目列表保存到 QSettings
void ProjectManager::saveRecentToSettings()
{
    QSettings settings;  // 20260322 ZJH 使用默认 QSettings
    // 20260322 ZJH 写入 "recentProjects" 键
    settings.setValue("recentProjects", m_vecRecentPaths);
}
