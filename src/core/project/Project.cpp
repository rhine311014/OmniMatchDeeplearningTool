// 20260322 ZJH Project 实现
// 项目元数据访问器和数据集管理

#include "core/project/Project.h"

// 20260322 ZJH 默认构造函数
// 创建空项目并初始化内部数据集实例
Project::Project()
    : m_eTaskType(om::TaskType::Classification)  // 20260322 ZJH 默认任务类型为分类
    , m_eState(om::ProjectState::Created)         // 20260322 ZJH 默认状态为已创建
    , m_pDataset(std::make_unique<ImageDataset>()) // 20260322 ZJH 创建数据集实例
{
}

// 20260322 ZJH 析构函数
// unique_ptr 自动销毁 ImageDataset
Project::~Project() = default;

// ===== 元数据访问器 =====

// 20260322 ZJH 获取项目名称
QString Project::name() const
{
    return m_strName;  // 20260322 ZJH 返回项目名称
}

// 20260322 ZJH 设置项目名称
void Project::setName(const QString& strName)
{
    m_strName = strName;  // 20260322 ZJH 更新项目名称
}

// 20260322 ZJH 获取项目路径
QString Project::path() const
{
    return m_strPath;  // 20260322 ZJH 返回项目根目录路径
}

// 20260322 ZJH 设置项目路径
void Project::setPath(const QString& strPath)
{
    m_strPath = strPath;  // 20260322 ZJH 更新项目根目录路径
}

// 20260322 ZJH 获取项目任务类型
om::TaskType Project::taskType() const
{
    return m_eTaskType;  // 20260322 ZJH 返回当前任务类型
}

// 20260322 ZJH 设置项目任务类型
void Project::setTaskType(om::TaskType eType)
{
    m_eTaskType = eType;  // 20260322 ZJH 更新任务类型
}

// 20260403 ZJH 获取项目描述
QString Project::description() const
{
    return m_strDescription;  // 20260403 ZJH 返回项目描述
}

// 20260403 ZJH 设置项目描述
void Project::setDescription(const QString& strDesc)
{
    m_strDescription = strDesc;  // 20260403 ZJH 更新项目描述
}

// 20260322 ZJH 获取项目工作流状态
om::ProjectState Project::state() const
{
    return m_eState;  // 20260322 ZJH 返回当前项目状态
}

// 20260322 ZJH 设置项目工作流状态
void Project::setState(om::ProjectState eState)
{
    m_eState = eState;  // 20260322 ZJH 更新项目状态
}

// ===== 训练配置持久化 =====

// 20260324 ZJH 获取训练配置（只读引用）
const TrainingConfig& Project::trainingConfig() const
{
    return m_trainingConfig;  // 20260324 ZJH 返回训练配置常量引用
}

// 20260324 ZJH 设置训练配置（训练开始时保存到项目）
void Project::setTrainingConfig(const TrainingConfig& config)
{
    m_trainingConfig = config;  // 20260324 ZJH 拷贝训练配置
    setDirty(true);  // 20260324 ZJH 训练配置变更，标记为脏
}

// ===== 训练历史记录 =====

// 20260324 ZJH 添加一条 Epoch 训练记录
void Project::addEpochRecord(const EpochRecord& record)
{
    m_vecTrainingHistory.append(record);  // 20260324 ZJH 追加到历史列表
    setDirty(true);  // 20260324 ZJH 训练历史变更，标记为脏
}

// 20260324 ZJH 清空训练历史记录
void Project::clearTrainingHistory()
{
    m_vecTrainingHistory.clear();  // 20260324 ZJH 清空列表
}

// 20260324 ZJH 获取全部训练历史记录
const QVector<EpochRecord>& Project::trainingHistory() const
{
    return m_vecTrainingHistory;  // 20260324 ZJH 返回训练历史常量引用
}

// 20260324 ZJH 获取最佳模型检查点路径
QString Project::bestModelPath() const
{
    return m_strBestModelPath;  // 20260324 ZJH 返回最佳模型路径
}

// 20260324 ZJH 设置最佳模型检查点路径
void Project::setBestModelPath(const QString& strPath)
{
    m_strBestModelPath = strPath;  // 20260324 ZJH 更新最佳模型路径
    setDirty(true);  // 20260324 ZJH 最佳模型路径变更，标记为脏
}

// ===== 评估结果持久化 =====

// 20260324 ZJH 获取最近一次评估结果（只读引用）
const EvaluationResult& Project::evaluationResult() const
{
    return m_lastEvalResult;  // 20260324 ZJH 返回评估结果常量引用
}

// 20260324 ZJH 检查是否已有评估结果
bool Project::hasEvaluationResult() const
{
    return m_bHasEvalResult;  // 20260324 ZJH 返回评估结果存在标志
}

// 20260324 ZJH 设置评估结果（评估完成后调用）
void Project::setEvaluationResult(const EvaluationResult& result)
{
    m_lastEvalResult = result;  // 20260324 ZJH 拷贝评估结果
    m_bHasEvalResult = true;    // 20260324 ZJH 标记已有评估结果
    setDirty(true);             // 20260324 ZJH 评估结果变更，标记为脏
}

// ===== 脏标志 =====

// 20260324 ZJH 查询项目是否有未保存的修改
bool Project::isDirty() const
{
    return m_bDirty;  // 20260324 ZJH 返回脏标志
}

// 20260324 ZJH 设置脏标志
void Project::setDirty(bool bDirty)
{
    m_bDirty = bDirty;  // 20260324 ZJH 更新脏标志
}

// ===== 数据集访问器 =====

// 20260322 ZJH 获取数据集指针（可修改版）
ImageDataset* Project::dataset()
{
    return m_pDataset.get();  // 20260322 ZJH 从 unique_ptr 获取裸指针
}

// 20260322 ZJH 获取数据集指针（只读版）
const ImageDataset* Project::dataset() const
{
    return m_pDataset.get();  // 20260322 ZJH 从 unique_ptr 获取常量裸指针
}
