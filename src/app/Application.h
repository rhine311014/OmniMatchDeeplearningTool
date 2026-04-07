// 20260322 ZJH Application — 全局事件总线 + 项目管理单例
// 20260324 ZJH 采用堆分配 + qApp 父对象模式，保证在 QApplication 析构前销毁
// 所有跨页面通信均通过此类的 signals/slots 实现解耦

#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <memory>

// 20260322 ZJH 前向声明 Project 类，避免头文件循环依赖
// Project 的完整定义在 core 层，待后续 Phase 实现
class Project;

// 20260322 ZJH OmniMatch 应用全局事件总线单例
// 职责：持有当前项目、广播跨模块信号、管理 GPU 硬件信息
class Application : public QObject
{
    Q_OBJECT

public:
    // 20260324 ZJH 获取进程内唯一的 Application 实例（堆分配 + qApp 父对象，确保析构顺序正确）
    static Application* instance();

    // 20260322 ZJH 禁用拷贝构造和拷贝赋值，确保单例语义
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // ===== 项目管理 =====

    // 20260322 ZJH 获取当前打开的项目，未打开时返回 nullptr
    Project* currentProject() const;

    // 20260322 ZJH 接管项目所有权，替换当前项目（nullptr 表示关闭项目）
    void setCurrentProject(std::unique_ptr<Project> pProject);

    // 20260322 ZJH 检查当前是否存在有效的已打开项目
    bool hasValidProject() const;

    // ===== 信号通知方法 =====
    // 20260324 ZJH 封装信号发射，禁止外部直接 emit Application 的信号
    // 外部调用这些方法代替 emit Application::instance()->xxxSignal()

    // 20260324 ZJH 通知项目已创建，内部发射 projectCreated 信号
    void notifyProjectCreated(Project* pProject);

    // 20260324 ZJH 通知项目已打开，内部发射 projectOpened 信号
    void notifyProjectOpened(Project* pProject);

    // 20260324 ZJH 通知项目已关闭，内部发射 projectClosed 信号
    void notifyProjectClosed();

    // 20260324 ZJH 通知项目已保存，内部发射 projectSaved 信号
    void notifyProjectSaved();

    // 20260324 ZJH 通知请求导航到指定页面，内部发射 requestNavigateToPage 信号
    void notifyNavigateToPage(int nPageIndex);

    // 20260324 ZJH 通知请求打开指定图像，内部发射 requestOpenImage 信号
    // 20260404 ZJH 新增 strAnnotationUuid: 可选，跳转后自动选中该标注（检查页实例双击用）
    void notifyOpenImage(const QString& strImageUuid, const QString& strAnnotationUuid = QString());

    // 20260324 ZJH 通知全局设置已变更，内部发射 globalSettingsChanged 信号
    void notifyGlobalSettingsChanged();

    // 20260324 ZJH 通知评估任务完成，内部发射 evaluationCompleted 信号
    void notifyEvaluationCompleted();

    // ===== 硬件信息 =====

    // 20260323 ZJH 在启动时调用一次，通过 QLibrary 动态加载 CUDA Driver 探测 GPU
    void initializePerformance();

    // 20260322 ZJH 获取 GPU 显卡名称（如 "NVIDIA GeForce RTX 3080"）
    // 未检测到 GPU 时返回 "No GPU detected"
    QString gpuName() const;

    // 20260322 ZJH 获取 GPU 显存大小（MB 为单位），未检测到时返回 0
    size_t gpuVramMB() const;

    // 20260322 ZJH 当前系统是否拥有可用的 NVIDIA GPU
    bool hasGpu() const;

signals:
    // ===== 项目生命周期信号 =====

    // 20260322 ZJH 新项目创建完成，各页面需初始化空白状态
    void projectCreated(Project* pProject);

    // 20260322 ZJH 已有项目打开完成，各页面需从磁盘加载数据
    void projectOpened(Project* pProject);

    // 20260322 ZJH 项目已关闭，各页面需清空显示内容
    void projectClosed();

    // 20260322 ZJH 项目已保存到磁盘
    void projectSaved();

    // ===== 数据变更信号 =====

    // 20260322 ZJH 图像批量导入完成，携带新增图像数量
    void imagesImported(int nCount);

    // 20260322 ZJH 图像被删除，携带被删除图像的路径列表
    void imagesDeleted(const QStringList& vecPaths);

    // 20260322 ZJH 指定图像的标注数据发生变更，携带图像 UUID
    void annotationChanged(const QString& strImageUuid);

    // ===== 训练信号 =====

    // 20260322 ZJH 训练会话开始，携带会话唯一 ID
    void trainingStarted(const QString& strSessionId);

    // 20260322 ZJH 训练进度更新（每个 epoch 触发）
    // nEpoch: 当前轮次（1-based）; nTotalEpochs: 总轮次
    // dTrainLoss: 训练集损失; dValLoss: 验证集损失; dMetric: 主评估指标（如 mAP/Acc）
    void trainingProgress(const QString& strSessionId,
                          int nEpoch, int nTotalEpochs,
                          double dTrainLoss, double dValLoss,
                          double dMetric);

    // 20260322 ZJH 训练会话正常结束
    void trainingCompleted(const QString& strSessionId);

    // 20260322 ZJH 训练会话异常终止，携带错误描述
    void trainingFailed(const QString& strSessionId, const QString& strError);

    // 20260322 ZJH 训练过程中的日志消息（逐行）
    void trainingLog(const QString& strSessionId, const QString& strMessage);

    // ===== 评估信号 =====

    // 20260322 ZJH 评估任务开始
    void evaluationStarted();

    // 20260322 ZJH 评估进度更新，nCurrent: 已处理数量; nTotal: 总数量
    void evaluationProgress(int nCurrent, int nTotal);

    // 20260322 ZJH 评估任务完成
    void evaluationCompleted();

    // ===== 导航信号 =====

    // 20260322 ZJH 请求 MainWindow 切换到指定页面索引（0-based）
    void requestNavigateToPage(int nPageIndex);

    // 20260322 ZJH 请求打开指定 UUID 对应的图像详情页
    // 20260404 ZJH 第二参数: 可选标注 UUID，用于跳转后自动选中该标注
    void requestOpenImage(const QString& strImageUuid, const QString& strAnnotationUuid);

    // ===== 设置信号 =====

    // 20260322 ZJH 全局设置（主题/语言/快捷键等）已变更，各页面据此刷新
    void globalSettingsChanged();

private:
    // 20260322 ZJH 私有构造函数，防止外部直接 new
    explicit Application(QObject* pParent = nullptr);
    ~Application() override;  // 20260322 ZJH 定义在 .cpp 中，确保 unique_ptr<Project> 析构时类型完整

    // 20260322 ZJH 当前打开的项目（Application 持有所有权）
    std::unique_ptr<Project> m_pCurrentProject;

    // 20260322 ZJH GPU 显卡名称，由 initializePerformance() 填充
    QString m_strGpuName;

    // 20260322 ZJH GPU 显存大小（MB），由 initializePerformance() 填充，未检测到为 0
    size_t m_nGpuVramMB = 0;

    // 20260322 ZJH 是否检测到可用的 NVIDIA GPU
    bool m_bHasGpu = false;
};
