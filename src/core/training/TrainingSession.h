// 20260322 ZJH TrainingSession — 训练会话管理类
// 20260324 ZJH 通过 EngineBridge 执行真实模型训练，加载真实图像数据
// 支持开始/暂停/恢复/停止操作

#pragma once

#include <QObject>          // 20260322 ZJH QObject 基类（信号槽）
#include <QString>          // 20260322 ZJH 字符串参数
#include <QMutex>           // 20260324 ZJH 互斥锁，用于暂停等待条件的同步
#include <QWaitCondition>   // 20260324 ZJH 条件变量，替代 busy-wait 暂停循环
#include <atomic>           // 20260322 ZJH 原子变量用于线程间通信
#include <memory>           // 20260324 ZJH std::unique_ptr 管理 EngineBridge 生命周期

#include "core/training/TrainingConfig.h"  // 20260322 ZJH 训练配置结构体
#include "engine/bridge/EngineBridge.h"    // 20260324 ZJH EngineBridge 引擎桥接层（真实训练/推理）

// 20260322 ZJH 前向声明 Project 类，避免头文件循环依赖
class Project;

// 20260322 ZJH 训练会话管理类
// 持有训练配置和项目引用，在后台线程中执行训练循环
// 通过 epochCompleted/batchCompleted/trainingFinished 等信号与 UI 通信
class TrainingSession : public QObject
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父 QObject（用于 Qt 对象树内存管理）
    explicit TrainingSession(QObject* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~TrainingSession() override = default;

    // 20260322 ZJH 设置训练配置（必须在 startTraining 之前调用）
    // 参数: config - 训练配置结构体
    void setConfig(const TrainingConfig& config);

    // 20260322 ZJH 设置当前项目（用于获取数据集信息）
    // 参数: pProject - 当前项目指针（弱引用，不持有所有权）
    void setProject(Project* pProject);

    // 20260322 ZJH 查询当前是否正在训练
    // 返回: true 表示训练循环正在运行
    bool isRunning() const;

public slots:
    // 20260322 ZJH 开始训练（在工作线程中调用，阻塞直到训练完成或被停止）
    void startTraining();

    // 20260322 ZJH 请求停止训练（线程安全，设置原子标记）
    void stopTraining();

    // 20260322 ZJH 请求暂停训练（线程安全，设置原子标记）
    void pauseTraining();

    // 20260322 ZJH 请求恢复训练（线程安全，清除暂停标记）
    void resumeTraining();

signals:
    // 20260322 ZJH 单个 Epoch 完成信号
    // 参数: nEpoch - 当前轮次（1-based）; nTotal - 总轮次
    //       dTrainLoss - 训练损失; dValLoss - 验证损失; dMetric - 主评估指标
    void epochCompleted(int nEpoch, int nTotal, double dTrainLoss, double dValLoss, double dMetric);

    // 20260322 ZJH 单个 Batch 完成信号
    // 参数: nBatch - 当前批次（1-based）; nTotal - 总批次数
    void batchCompleted(int nBatch, int nTotal);

    // 20260322 ZJH 训练结束信号
    // 参数: bSuccess - 是否正常完成; strMessage - 结束消息
    void trainingFinished(bool bSuccess, const QString& strMessage);

    // 20260322 ZJH 训练日志输出信号
    // 参数: strMessage - 日志消息文本
    void trainingLog(const QString& strMessage);

    // 20260322 ZJH 进度百分比变化信号
    // 参数: nPercent - 当前进度（0~100）
    void progressChanged(int nPercent);

public:
    // 20260324 ZJH 获取训练完成后保存的模型文件路径
    // 返回: 模型保存的绝对路径（训练未完成时为空）
    QString modelSavePath() const;

    // 20260324 ZJH 获取引擎桥接层指针（只读，供外部查询模型参数量等信息）
    // 返回: EngineBridge 指针（模型未创建时为 nullptr）
    EngineBridge* engine() const;

private:
    // 20260324 ZJH 执行真实训练循环（通过 EngineBridge 调用引擎）
    void runTrainingLoop();
    void runTrainingLoopImpl();  // 20260326 ZJH 训练实际实现（由 runTrainingLoop 的 try/catch 包裹）
    void forceGpuCleanup();      // 20260326 ZJH 强制释放 GPU 内存（训练结束/异常后调用）

    // 20260324 ZJH 将 om::ModelArchitecture 枚举映射为 EngineBridge 期望的模型类型字符串
    // 参数: eArch - 模型架构枚举值
    // 返回: 引擎识别的模型类型字符串（如 "ResNet18", "MLP"）
    static std::string architectureToEngineString(om::ModelArchitecture eArch);

    TrainingConfig m_config;              // 20260322 ZJH 训练配置副本
    Project* m_pProject = nullptr;        // 20260322 ZJH 当前项目指针（弱引用）

    std::atomic<bool> m_bRunning{false};        // 20260322 ZJH 训练是否正在运行
    std::atomic<bool> m_bStopRequested{false};  // 20260322 ZJH 是否收到停止请求
    std::atomic<bool> m_bPaused{false};         // 20260322 ZJH 是否处于暂停状态

    QMutex m_pauseMutex;                // 20260324 ZJH 暂停条件变量配套互斥锁
    QWaitCondition m_pauseCondition;    // 20260324 ZJH 暂停等待条件，替代 busy-wait 轮询

    std::unique_ptr<EngineBridge> m_pEngine;  // 20260324 ZJH 引擎桥接层实例（PIMPL 模式隐藏 C++23 依赖）
    QString m_strModelSavePath;               // 20260324 ZJH 训练完成后模型保存的绝对路径
};
