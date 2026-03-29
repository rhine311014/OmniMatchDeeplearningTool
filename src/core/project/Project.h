// 20260322 ZJH Project — 项目数据管理类
// 持有项目元数据（名称/路径/任务类型/状态）和数据集对象
// 每个项目对应一个 ImageDataset 实例，管理图像和标签
#pragma once

#include <QString>
#include <QVector>
#include <memory>

#include "core/DLTypes.h"
#include "core/data/ImageDataset.h"
#include "core/training/TrainingConfig.h"       // 20260324 ZJH 训练配置结构体（持久化到 .dfproj）
#include "core/evaluation/EvaluationResult.h"   // 20260324 ZJH 评估结果结构体（持久化到 .dfproj）

// 20260324 ZJH 单个 Epoch 训练记录结构体
// 记录每轮训练的核心指标，用于训练历史持久化和回顾
struct EpochRecord
{
    int    nEpoch       = 0;    // 20260324 ZJH 轮次编号（1-based）
    double dTrainLoss   = 0.0;  // 20260324 ZJH 该轮训练集损失
    double dValLoss     = 0.0;  // 20260324 ZJH 该轮验证集损失
    double dValAccuracy = 0.0;  // 20260324 ZJH 该轮验证集准确率/主评估指标
};

// 20260322 ZJH 项目数据管理类
// 代表一个 OmniMatch 项目的完整状态
class Project
{
public:
    // 20260322 ZJH 默认构造函数：创建空项目并初始化数据集
    Project();

    // 20260322 ZJH 析构函数
    ~Project();

    // ===== 元数据访问 =====

    // 20260322 ZJH 获取项目名称
    // 返回: 项目名称字符串
    QString name() const;

    // 20260322 ZJH 设置项目名称
    // 参数: strName - 新的项目名称
    void setName(const QString& strName);

    // 20260322 ZJH 获取项目路径（项目根目录的绝对路径）
    // 返回: 项目路径字符串
    QString path() const;

    // 20260322 ZJH 设置项目路径
    // 参数: strPath - 项目根目录的绝对路径
    void setPath(const QString& strPath);

    // 20260322 ZJH 获取项目任务类型
    // 返回: 任务类型枚举值
    om::TaskType taskType() const;

    // 20260322 ZJH 设置项目任务类型
    // 参数: eType - 新的任务类型
    void setTaskType(om::TaskType eType);

    // 20260322 ZJH 获取项目工作流状态
    // 返回: 项目状态枚举值
    om::ProjectState state() const;

    // 20260322 ZJH 设置项目工作流状态
    // 参数: eState - 新的项目状态
    void setState(om::ProjectState eState);

    // ===== 训练配置持久化 =====

    // 20260324 ZJH 获取训练配置（只读）
    // 返回: 当前保存的训练配置引用
    const TrainingConfig& trainingConfig() const;

    // 20260324 ZJH 设置训练配置（训练开始时由 TrainingPage 调用）
    // 参数: config - 从 UI 收集的训练配置
    void setTrainingConfig(const TrainingConfig& config);

    // ===== 训练历史记录 =====

    // 20260324 ZJH 添加一条 Epoch 训练记录到历史列表
    // 参数: record - 单轮训练指标记录
    void addEpochRecord(const EpochRecord& record);

    // 20260324 ZJH 清空训练历史记录（新一轮训练开始前调用）
    void clearTrainingHistory();

    // 20260324 ZJH 获取全部训练历史记录（只读）
    // 返回: 训练历史记录列表引用
    const QVector<EpochRecord>& trainingHistory() const;

    // 20260324 ZJH 获取最佳模型检查点路径
    // 返回: 最佳模型文件的绝对路径（训练未完成时为空）
    QString bestModelPath() const;

    // 20260324 ZJH 设置最佳模型检查点路径（训练完成后由 TrainingPage 调用）
    // 参数: strPath - 最佳模型文件的绝对路径
    void setBestModelPath(const QString& strPath);

    // ===== 评估结果持久化 =====

    // 20260324 ZJH 获取最近一次评估结果（只读）
    // 返回: 评估结果结构体的常量引用
    const EvaluationResult& evaluationResult() const;

    // 20260324 ZJH 检查项目是否已有评估结果
    // 返回: true 表示存在评估结果，false 表示尚未评估
    bool hasEvaluationResult() const;

    // 20260324 ZJH 设置评估结果（评估完成后由 EvaluationPage 调用）
    // 参数: result - 本次评估的完整结果
    void setEvaluationResult(const EvaluationResult& result);

    // ===== 脏标志（自动保存） =====

    // 20260324 ZJH 查询项目是否有未保存的修改
    // 返回: true 表示有未保存的修改
    bool isDirty() const;

    // 20260324 ZJH 设置脏标志（数据变更时调用 setDirty(true)，保存后调用 setDirty(false)）
    // 参数: bDirty - 脏标志值，默认为 true
    void setDirty(bool bDirty = true);

    // ===== 数据集访问 =====

    // 20260322 ZJH 获取项目关联的数据集（可修改版）
    // 返回: 指向 ImageDataset 的指针（生命周期由 Project 管理）
    ImageDataset* dataset();

    // 20260322 ZJH 获取项目关联的数据集（只读版）
    // 返回: 指向 ImageDataset 的常量指针
    const ImageDataset* dataset() const;

private:
    QString m_strName;  // 20260322 ZJH 项目名称
    QString m_strPath;  // 20260322 ZJH 项目根目录绝对路径

    // 20260322 ZJH 项目任务类型（默认为分类）
    om::TaskType m_eTaskType = om::TaskType::Classification;

    // 20260322 ZJH 项目工作流状态（默认为已创建）
    om::ProjectState m_eState = om::ProjectState::Created;

    // 20260324 ZJH 训练配置（持久化到 .dfproj，程序重启后恢复上次训练参数）
    TrainingConfig m_trainingConfig;

    // 20260324 ZJH 训练历史记录（每个 Epoch 一条，持久化到 .dfproj）
    QVector<EpochRecord> m_vecTrainingHistory;

    // 20260324 ZJH 最佳模型检查点文件路径（训练完成后保存）
    QString m_strBestModelPath;

    // 20260324 ZJH 最近一次评估结果（持久化到 .dfproj）
    EvaluationResult m_lastEvalResult;

    // 20260324 ZJH 是否已有评估结果（false 表示尚未评估，不序列化空结果）
    bool m_bHasEvalResult = false;

    // 20260324 ZJH 脏标志（有未保存修改时为 true，保存后重置为 false）
    bool m_bDirty = false;

    // 20260322 ZJH 数据集实例（由 Project 独占所有权）
    // 使用 unique_ptr 管理生命周期，Project 析构时自动销毁
    std::unique_ptr<ImageDataset> m_pDataset;
};
