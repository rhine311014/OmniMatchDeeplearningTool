// 20260322 ZJH ProjectManager — 项目管理器
// 提供项目的创建、打开、保存、关闭全生命周期管理
// 维护最近项目列表（QSettings 持久化）
// 通过信号通知 Application 全局事件总线
#pragma once

#include <QObject>
#include <QStringList>

// 20260322 ZJH 前向声明，避免头文件循环依赖
class Project;

// 20260322 ZJH 任务类型前向声明
namespace om {
enum class TaskType : uint8_t;
}

// 20260322 ZJH 项目管理器类
// 职责：
//   1. 创建/打开/保存/关闭项目
//   2. 序列化/反序列化项目文件（.dfproj）
//   3. 管理最近打开的项目列表
class ProjectManager : public QObject
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父 QObject
    explicit ProjectManager(QObject* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~ProjectManager() override = default;

    // ===== 项目操作 =====

    // 20260403 ZJH 创建新项目
    // 参数: strName - 项目名称
    //       eType   - 任务类型
    //       strPath - 项目存储路径（文件夹）
    //       strDescription - 项目描述（可选，默认为空）
    // 返回: 新创建的 Project 指针（所有权转交给 Application）
    Project* createProject(const QString& strName, om::TaskType eType, const QString& strPath,
                           const QString& strDescription = QString());

    // 20260322 ZJH 打开已有项目
    // 参数: strFilePath - .dfproj 文件完整路径
    // 返回: 加载的 Project 指针（所有权转交给 Application），失败返回 nullptr
    Project* openProject(const QString& strFilePath);

    // 20260322 ZJH 保存项目到磁盘
    // 参数: pProject - 要保存的项目指针
    // 返回: true 保存成功，false 保存失败
    bool saveProject(Project* pProject);

    // 20260322 ZJH 另存为：将项目保存到指定路径
    // 参数: pProject    - 要保存的项目指针
    //       strFilePath - 新的 .dfproj 文件路径
    // 返回: true 保存成功，false 保存失败
    bool saveProjectAs(Project* pProject, const QString& strFilePath);

    // 20260322 ZJH 关闭当前项目
    // 将 Application 的当前项目设为 nullptr，发射 projectClosed 信号
    void closeProject();

    // ===== 最近项目 =====

    // 20260322 ZJH 获取最近打开的项目路径列表
    // 返回: 项目路径列表（最新在前，最多 MAX_RECENT 条）
    QStringList recentProjects() const;

    // 20260322 ZJH 添加项目路径到最近列表
    // 参数: strPath - 项目 .dfproj 文件路径
    void addToRecent(const QString& strPath);

    // 20260322 ZJH 清空最近项目列表
    void clearRecent();

signals:
    // 20260322 ZJH 项目创建完成信号
    void projectCreated(Project* pProject);

    // 20260322 ZJH 项目打开完成信号
    void projectOpened(Project* pProject);

    // 20260322 ZJH 项目保存完成信号
    void projectSaved();

    // 20260322 ZJH 项目关闭完成信号
    void projectClosed();

private:
    // 20260322 ZJH 从 QSettings 加载最近项目列表到 m_vecRecentPaths
    void loadRecentFromSettings();

    // 20260322 ZJH 将 m_vecRecentPaths 保存到 QSettings
    void saveRecentToSettings();

    // 20260322 ZJH 最近打开的项目路径列表
    QStringList m_vecRecentPaths;

    // 20260322 ZJH 最近项目列表最大容量
    static constexpr int MAX_RECENT = 10;
};
