// 20260322 ZJH TrainingPage 实现
// 三栏布局：左面板模型配置、中央损失曲线、右面板状态日志
// 通过 TrainingSession 在后台线程执行模拟训练，实时更新 UI

#include "ui/pages/training/TrainingPage.h"        // 20260322 ZJH TrainingPage 类声明
#include "ui/widgets/TrainingLossChart.h"           // 20260322 ZJH 损失曲线图表
#include "core/training/TrainingSession.h"          // 20260322 ZJH 训练会话
#include "core/training/TrainingConfig.h"           // 20260322 ZJH 训练配置
#include "core/project/Project.h"                   // 20260322 ZJH 项目数据
#include "core/project/ProjectManager.h"            // 20260324 ZJH 项目管理器（保留引用）
#include "core/project/ProjectSerializer.h"         // 20260324 ZJH 项目序列化器（训练完成后自动保存）
#include "core/data/ImageDataset.h"                 // 20260322 ZJH 数据集
#include "core/DLTypes.h"                           // 20260322 ZJH 类型定义
#include "app/Application.h"                        // 20260322 ZJH 全局事件总线
#include "engine/bridge/EngineBridge.h"             // 20260324 ZJH 引擎桥接层（autoSelectBatchSize）
#include "ui/dialogs/DataSynthesisDialog.h"         // 20260330 ZJH AI 数据合成对话框
#include <iostream>                                  // 20260330 ZJH std::cerr 调试输出
#include <algorithm>                                 // 20260402 ZJH std::max_element 用于诊断系统最佳指标查找
#include <chrono>                                    // 20260402 ZJH steady_clock 精确 epoch 计时

#include <QVBoxLayout>      // 20260322 ZJH 垂直布局
#include <QHBoxLayout>      // 20260322 ZJH 水平布局
#include <QFormLayout>      // 20260322 ZJH 表单布局（标签+控件）
#include <QScrollArea>      // 20260326 ZJH 左面板滚动区域（内容多时可滚动查看参数）
#include <QGroupBox>        // 20260322 ZJH 分组框
#include <QMessageBox>      // 20260322 ZJH 警告对话框
#include <QDateTime>        // 20260322 ZJH 时间戳
#include <QFileInfo>        // 20260324 ZJH QFileInfo 验证模型文件存在
#include <QFileDialog>      // 20260330 ZJH 预训练模型文件选择对话框
#include <QRegularExpressionValidator>  // 20260330 ZJH 模型标识输入验证

// 20260322 ZJH 通用暗色控件样式（下拉框/微调框/复选框/按钮共用）
static const QString s_strControlStyle = QStringLiteral(
    "QComboBox, QSpinBox, QDoubleSpinBox {"
    "  background-color: #1a1d24;"
    "  color: #e2e8f0;"
    "  border: 1px solid #333842;"
    "  border-radius: 4px;"
    "  padding: 4px 8px;"
    "  min-height: 22px;"
    "}"
    "QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {"
    "  border-color: #2563eb;"
    "}"
    "QComboBox::drop-down {"
    "  border: none;"
    "  width: 20px;"
    "}"
    "QComboBox QAbstractItemView {"
    "  background-color: #1a1d24;"
    "  color: #e2e8f0;"
    "  border: 1px solid #333842;"
    "  selection-background-color: #2563eb;"
    "}"
    "QCheckBox {"
    "  color: #e2e8f0;"
    "  spacing: 6px;"
    "}"
    "QCheckBox::indicator {"
    "  width: 16px;"
    "  height: 16px;"
    "  border: 1px solid #333842;"
    "  border-radius: 3px;"
    "  background: #1a1d24;"
    "}"
    "QCheckBox::indicator:checked {"
    "  background: #2563eb;"
    "  border-color: #2563eb;"
    "}"
    "QGroupBox {"
    "  color: #94a3b8;"
    "  border: 1px solid #3d4455;"
    "  border-radius: 6px;"
    "  margin-top: 10px;"
    "  padding: 14px 6px 6px 6px;"
    "  font-weight: bold;"
    "}"
    "QGroupBox::title {"
    "  subcontrol-origin: margin;"
    "  left: 10px;"
    "  padding: 0 6px;"
    "  color: #cbd5e1;"
    "}"
    "QLabel {"
    "  color: #94a3b8;"
    "}"
);

// 20260322 ZJH 构造函数
TrainingPage::TrainingPage(QWidget* pParent)
    : BasePage(pParent)                        // 20260406 ZJH 初始化页面基类
    , m_pCboFramework(nullptr)                 // 20260406 ZJH 训练框架下拉框初始为空
    , m_pCboCapability(nullptr)                // 20260406 ZJH 模型能力等级下拉框初始为空
    , m_pCboArchitecture(nullptr)              // 20260406 ZJH 模型架构下拉框初始为空
    , m_pCboDevice(nullptr)                    // 20260406 ZJH 设备类型下拉框初始为空
    , m_pCboOptimizer(nullptr)                 // 20260406 ZJH 优化器下拉框初始为空
    , m_pCboScheduler(nullptr)                 // 20260406 ZJH 调度器下拉框初始为空
    , m_pSpnLearningRate(nullptr)              // 20260406 ZJH 学习率微调框初始为空
    , m_pSpnBatchSize(nullptr)                 // 20260406 ZJH 批量大小微调框初始为空
    , m_pBtnAutoMaxBatch(nullptr)              // 20260406 ZJH 自动最大批量按钮初始为空
    , m_pSpnEpochs(nullptr)                    // 20260406 ZJH 训练轮次微调框初始为空
    , m_pSpnInputSize(nullptr)                 // 20260406 ZJH 输入尺寸微调框初始为空
    , m_pSpnPatience(nullptr)                  // 20260406 ZJH 早停耐心微调框初始为空
    , m_pChkAugmentation(nullptr)              // 20260406 ZJH 启用增强复选框初始为空
    , m_pSpnBrightness(nullptr)                // 20260406 ZJH 亮度增强微调框初始为空
    , m_pSpnFlipProb(nullptr)                  // 20260406 ZJH 水平翻转概率微调框初始为空
    , m_pSpnRotation(nullptr)                  // 20260406 ZJH 旋转角度微调框初始为空
    , m_pSpnVerticalFlipProb(nullptr)          // 20260406 ZJH 垂直翻转概率微调框初始为空
    , m_pChkAffine(nullptr)                    // 20260406 ZJH 仿射变换复选框初始为空
    , m_pSpnShearDeg(nullptr)                  // 20260406 ZJH 仿射剪切角度微调框初始为空
    , m_pSpnTranslate(nullptr)                 // 20260406 ZJH 仿射平移比例微调框初始为空
    , m_pChkRandomCrop(nullptr)                // 20260406 ZJH 随机裁剪复选框初始为空
    , m_pSpnCropScale(nullptr)                 // 20260406 ZJH 裁剪缩放比例微调框初始为空
    , m_pChkColorJitter(nullptr)               // 20260406 ZJH 颜色抖动复选框初始为空
    , m_pSpnSaturation(nullptr)                // 20260406 ZJH 饱和度抖动微调框初始为空
    , m_pSpnHue(nullptr)                       // 20260406 ZJH 色调抖动微调框初始为空
    , m_pChkGaussianNoise(nullptr)             // 20260406 ZJH 高斯噪声复选框初始为空
    , m_pSpnNoiseStd(nullptr)                  // 20260406 ZJH 噪声标准差微调框初始为空
    , m_pChkGaussianBlur(nullptr)              // 20260406 ZJH 高斯模糊复选框初始为空
    , m_pSpnBlurSigma(nullptr)                 // 20260406 ZJH 模糊 sigma 微调框初始为空
    , m_pChkRandomErasing(nullptr)             // 20260406 ZJH 随机擦除复选框初始为空
    , m_pSpnErasingProb(nullptr)               // 20260406 ZJH 擦除概率微调框初始为空
    , m_pSpnErasingRatio(nullptr)              // 20260406 ZJH 擦除面积比例微调框初始为空
    , m_pChkMixup(nullptr)                     // 20260406 ZJH Mixup 复选框初始为空
    , m_pSpnMixupAlpha(nullptr)                // 20260406 ZJH Mixup alpha 微调框初始为空
    , m_pChkCutMix(nullptr)                    // 20260406 ZJH CutMix 复选框初始为空
    , m_pSpnCutMixAlpha(nullptr)               // 20260406 ZJH CutMix alpha 微调框初始为空
    , m_pChkExportOnnx(nullptr)                // 20260406 ZJH 导出 ONNX 复选框初始为空
    , m_pLossChart(nullptr)                    // 20260406 ZJH 损失曲线图表初始为空
    , m_pProgressBar(nullptr)                  // 20260406 ZJH 进度条初始为空
    , m_pLblStatus(nullptr)                    // 20260406 ZJH 状态文字标签初始为空
    , m_pBtnStart(nullptr)                     // 20260406 ZJH 开始训练按钮初始为空
    , m_pBtnPause(nullptr)                     // 20260406 ZJH 暂停训练按钮初始为空
    , m_pBtnStop(nullptr)                      // 20260406 ZJH 停止训练按钮初始为空
    , m_pBtnResume(nullptr)                    // 20260406 ZJH 继续训练按钮初始为空
    , m_pLblTrainCount(nullptr)                // 20260406 ZJH 训练集数量标签初始为空
    , m_pLblValCount(nullptr)                  // 20260406 ZJH 验证集数量标签初始为空
    , m_pLblTestCount(nullptr)                 // 20260406 ZJH 测试集数量标签初始为空
    , m_pLblCheckImages(nullptr)               // 20260406 ZJH 前置检查-图像已导入标签初始为空
    , m_pLblCheckLabels(nullptr)               // 20260406 ZJH 前置检查-标签已分配标签初始为空
    , m_pLblCheckSplit(nullptr)                // 20260406 ZJH 前置检查-数据已拆分标签初始为空
    , m_pLblCurrentEpoch(nullptr)              // 20260406 ZJH 当前 Epoch 标签初始为空
    , m_pLblBestLoss(nullptr)                  // 20260406 ZJH 最佳损失标签初始为空
    , m_pLblTimeRemaining(nullptr)             // 20260406 ZJH 预计剩余时间标签初始为空
    , m_pLblEarlyStopCount(nullptr)            // 20260406 ZJH 早停计数标签初始为空
    , m_pTxtLog(nullptr)                       // 20260406 ZJH 日志文本框初始为空
    , m_pSession(nullptr)                      // 20260406 ZJH 训练会话对象初始为空
    , m_pWorkerThread(nullptr)                 // 20260406 ZJH 训练工作线程初始为空
{
    // 20260322 ZJH 1. 创建三栏面板
    m_pLeftPanel = createLeftPanel();       // 20260322 ZJH 左面板：配置
    QWidget* pCenter = createCenterPanel(); // 20260322 ZJH 中央面板：曲线+进度+按钮
    QWidget* pRight  = createRightPanel();  // 20260322 ZJH 右面板：状态+日志

    // 20260322 ZJH 2. 使用 BasePage 的三栏布局辅助方法
    setLeftPanelWidth(300);   // 20260322 ZJH 左面板 300px
    setRightPanelWidth(250);  // 20260322 ZJH 右面板 250px
    setupThreeColumnLayout(m_pLeftPanel, pCenter, pRight);

    // 20260322 ZJH 3. 创建训练会话和工作线程
    m_pWorkerThread = new QThread(this);
    m_pSession = new TrainingSession();  // 20260322 ZJH 不设置 parent，因为要 moveToThread
    m_pSession->moveToThread(m_pWorkerThread);  // 20260322 ZJH 移到工作线程

    // 20260322 ZJH 4. 连接训练会话信号到 UI 槽
    // 20260406 ZJH Epoch 完成信号 → UI 更新损失曲线和指标（跨线程队列连接）
    connect(m_pSession, &TrainingSession::epochCompleted,
            this, &TrainingPage::onEpochCompleted, Qt::QueuedConnection);
    // 20260406 ZJH 训练完成/失败信号 → UI 显示结果并恢复按钮状态
    connect(m_pSession, &TrainingSession::trainingFinished,
            this, &TrainingPage::onTrainingFinished, Qt::QueuedConnection);
    // 20260406 ZJH 训练日志信号 → UI 追加日志文本
    connect(m_pSession, &TrainingSession::trainingLog,
            this, &TrainingPage::onTrainingLog, Qt::QueuedConnection);
    // 20260406 ZJH 进度百分比信号 → UI 更新进度条
    connect(m_pSession, &TrainingSession::progressChanged,
            this, &TrainingPage::onProgressChanged, Qt::QueuedConnection);

    // 20260322 ZJH 5. 线程结束时清理 session 对象
    connect(m_pWorkerThread, &QThread::finished,
            m_pSession, &QObject::deleteLater);

    // 20260324 ZJH 线程结束时将 m_pSession 置空，防止后续访问悬挂指针
    connect(m_pWorkerThread, &QThread::finished, this, [this]() {
        m_pSession = nullptr;  // 20260324 ZJH session 已被 deleteLater 调度销毁
    });

    // 20260322 ZJH 6. 启动工作线程（线程空闲等待，不执行任何任务）
    m_pWorkerThread->start();

    // 20260322 ZJH 7. 连接控制按钮信号
    connect(m_pBtnStart, &QPushButton::clicked, this, &TrainingPage::onStartTraining);    // 20260406 ZJH 开始训练按钮 → 启动训练
    connect(m_pBtnPause, &QPushButton::clicked, this, &TrainingPage::onPauseTraining);  // 20260406 ZJH 暂停训练按钮 → 暂停训练
    connect(m_pBtnStop,  &QPushButton::clicked, this, &TrainingPage::onStopTraining);   // 20260406 ZJH 停止训练按钮 → 停止训练
    connect(m_pBtnResume, &QPushButton::clicked, this, &TrainingPage::onResumeTraining);  // 20260406 ZJH 继续训练按钮 → 恢复训练

    // 20260322 ZJH 8. 初始化按钮状态
    updateButtonStates();
}

// 20260322 ZJH 析构函数
TrainingPage::~TrainingPage()
{
    // 20260324 ZJH 先断开所有来自 m_pSession 的信号，防止析构期间回调已销毁的 UI 控件
    if (m_pSession) {
        disconnect(m_pSession, nullptr, this, nullptr);  // 20260324 ZJH 断开 session → this 的所有连接
    }

    // 20260322 ZJH 请求停止训练（如果正在运行）
    if (m_pSession && m_pSession->isRunning()) {
        m_pSession->stopTraining();
    }

    // 20260322 ZJH 停止工作线程
    if (m_pWorkerThread) {
        m_pWorkerThread->quit();   // 20260322 ZJH 请求线程退出事件循环
        // 20260324 ZJH 等待最多 3 秒，超时则强制终止防止析构挂起
        if (!m_pWorkerThread->wait(3000)) {
            // 20260324 ZJH 线程 3 秒内未退出，强制终止并记录警告
            qWarning("TrainingPage: Worker thread did not exit within 3s, terminating.");
            m_pWorkerThread->terminate();  // 20260324 ZJH 强制终止线程
            m_pWorkerThread->wait(1000);   // 20260324 ZJH 再等 1 秒确保终止完成
        }
    }

    // 20260324 ZJH 置空 session 指针（session 由 QThread::finished → deleteLater 自动回收）
    m_pSession = nullptr;
}

// ===== BasePage 生命周期回调 =====

// 20260322 ZJH 页面切换到前台
void TrainingPage::onEnter()
{
    // 20260322 ZJH 刷新模型架构下拉框（可能任务类型已变更）
    refreshArchitectureCombo();
    // 20260322 ZJH 刷新数据概览
    refreshDataOverview();
}

// 20260322 ZJH 页面离开前台
void TrainingPage::onLeave()
{
    // 20260322 ZJH 当前无需处理（训练可在后台继续）
}

// 20260322 ZJH 项目加载后
// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void TrainingPage::onProjectLoadedImpl()
{
    // 20260322 ZJH 刷新模型架构（根据新项目的任务类型）
    refreshArchitectureCombo();

    // 20260325 ZJH 从项目中恢复保存的训练参数到 UI 控件
    if (m_pProject) {
        restoreConfigToUI(m_pProject->trainingConfig());
    }

    // 20260322 ZJH 刷新数据概览和前置检查
    refreshDataOverview();
}

// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void TrainingPage::onProjectClosedImpl()
{
    // 20260322 ZJH 重置数据概览
    m_pLblTrainCount->setText(QStringLiteral("0"));
    m_pLblValCount->setText(QStringLiteral("0"));
    m_pLblTestCount->setText(QStringLiteral("0"));

    // 20260322 ZJH 重置前置检查
    m_pLblCheckImages->setText(QStringLiteral("\u2717 图像已导入"));  // 20260322 ZJH ✗
    m_pLblCheckLabels->setText(QStringLiteral("\u2717 标签已分配"));
    m_pLblCheckSplit->setText(QStringLiteral("\u2717 数据已拆分"));

    // 20260322 ZJH 清空模型架构下拉框
    m_pCboArchitecture->clear();
}

// ===== 控件创建 =====

// 20260322 ZJH 创建左面板
QWidget* TrainingPage::createLeftPanel()
{
    // 20260326 ZJH 用 QScrollArea 包裹左面板内容，增强选项多时可滚动查看
    // 之前直接返回 QWidget 导致内容挤在一起无法调参
    QScrollArea* pScroll = new QScrollArea();
    pScroll->setWidgetResizable(true);  // 20260326 ZJH 内容自适应宽度
    pScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);  // 20260326 ZJH 隐藏水平滚动条
    pScroll->setStyleSheet(QStringLiteral(
        "QScrollArea { background: #1e2230; border: none; }"
        "QScrollBar:vertical { background: #1e2230; width: 8px; }"
        "QScrollBar::handle:vertical { background: #3b4252; border-radius: 4px; min-height: 30px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"));

    QWidget* pContent = new QWidget();
    pContent->setStyleSheet(s_strControlStyle + QStringLiteral("QWidget { background-color: #1e2230; }"));

    QVBoxLayout* pLayout = new QVBoxLayout(pContent);
    pLayout->setContentsMargins(10, 8, 10, 8);  // 20260322 ZJH 边距
    pLayout->setSpacing(10);  // 20260325 ZJH 分组框之间间距加大，分清层次

    // ===== 模型配置分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("模型配置"), pContent);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);  // 20260322 ZJH 标签右对齐
        pForm->setSpacing(6);  // 20260322 ZJH 行间距

        // 20260322 ZJH 训练框架下拉框
        m_pCboFramework = new QComboBox();
        m_pCboFramework->addItem(QStringLiteral("NativeCpp"),  static_cast<int>(om::TrainingFramework::NativeCpp));
        m_pCboFramework->addItem(QStringLiteral("Libtorch"),   static_cast<int>(om::TrainingFramework::Libtorch));
        m_pCboFramework->addItem(QStringLiteral("Auto"),       static_cast<int>(om::TrainingFramework::Auto));
        pForm->addRow(QStringLiteral("训练框架"), m_pCboFramework);

        // 20260330 ZJH 模型能力等级下拉框（借鉴海康 VisionTrain "模型能力"概念）
        // 用户通过"轻量化/普通/高精度"选择能力等级，底层自动映射到具体架构
        m_pCboCapability = new QComboBox();
        m_pCboCapability->addItem(QStringLiteral("轻量化 — 推理速度优先"), static_cast<int>(om::ModelCapability::Lightweight));
        m_pCboCapability->addItem(QStringLiteral("普通 — 速度与精度均衡"), static_cast<int>(om::ModelCapability::Normal));
        m_pCboCapability->addItem(QStringLiteral("高精度 — 复杂场景优先"), static_cast<int>(om::ModelCapability::HighAccuracy));
        m_pCboCapability->setCurrentIndex(1);  // 20260330 ZJH 默认"普通"
        pForm->addRow(QStringLiteral("模型能力"), m_pCboCapability);

        // 20260330 ZJH 模型能力变更时，自动更新架构下拉框内容
        connect(m_pCboCapability, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &TrainingPage::refreshArchitectureCombo);

        // 20260330 ZJH 模型架构高级选项（默认折叠，保留给高级用户直接选择具体架构）
        m_pBtnToggleArch = new QPushButton(QStringLiteral("\u25b6 具体架构..."));
        m_pBtnToggleArch->setStyleSheet(QStringLiteral(
            "QPushButton { background: transparent; color: #60a5fa; border: none;"
            "  text-align: left; padding: 2px 0; font-size: 12px; }"
            "QPushButton:hover { color: #93c5fd; }"));
        pForm->addRow(QString(), m_pBtnToggleArch);

        // 20260330 ZJH 架构选择容器（可折叠）
        m_pArchContainer = new QWidget();
        QFormLayout* pArchForm = new QFormLayout(m_pArchContainer);
        pArchForm->setContentsMargins(0, 0, 0, 0);
        pArchForm->setSpacing(4);

        m_pCboArchitecture = new QComboBox();
        pArchForm->addRow(QStringLiteral("模型架构"), m_pCboArchitecture);

        m_pArchContainer->setVisible(false);  // 20260330 ZJH 默认折叠
        pForm->addRow(m_pArchContainer);

        // 20260330 ZJH 展开/收起架构选项
        connect(m_pBtnToggleArch, &QPushButton::clicked, this, [this]() {
            bool bShow = !m_pArchContainer->isVisible();
            m_pArchContainer->setVisible(bShow);
            m_pBtnToggleArch->setText(bShow
                ? QStringLiteral("\u25bc 收起架构选项")
                : QStringLiteral("\u25b6 具体架构..."));
        });

        // 20260322 ZJH 设备下拉框
        m_pCboDevice = new QComboBox();
        m_pCboDevice->addItem(QStringLiteral("CPU"),  static_cast<int>(om::DeviceType::CPU));
        m_pCboDevice->addItem(QStringLiteral("CUDA"), static_cast<int>(om::DeviceType::CUDA));
        pForm->addRow(QStringLiteral("设备"), m_pCboDevice);

        // 20260330 ZJH 异常检测训练模式（仅异常检测任务可见）
        m_pLblAnomalyMode = new QLabel(QStringLiteral("训练模式"));
        m_pCboAnomalyMode = new QComboBox();
        m_pCboAnomalyMode->addItem(QStringLiteral("极速模式 — PaDiM/PatchCore"),  static_cast<int>(om::AnomalyTrainingMode::Fast));
        m_pCboAnomalyMode->addItem(QStringLiteral("高精度模式 — EfficientAD/FastFlow"), static_cast<int>(om::AnomalyTrainingMode::HighAccuracy));
        pForm->addRow(m_pLblAnomalyMode, m_pCboAnomalyMode);
        // 20260330 ZJH 默认隐藏，仅异常检测任务显示
        m_pLblAnomalyMode->setVisible(false);
        m_pCboAnomalyMode->setVisible(false);

        // 20260322 ZJH 优化器下拉框
        m_pCboOptimizer = new QComboBox();
        m_pCboOptimizer->addItem(QStringLiteral("Adam"),  static_cast<int>(om::OptimizerType::Adam));
        m_pCboOptimizer->addItem(QStringLiteral("AdamW"), static_cast<int>(om::OptimizerType::AdamW));
        m_pCboOptimizer->addItem(QStringLiteral("SGD"),   static_cast<int>(om::OptimizerType::SGD));
        pForm->addRow(QStringLiteral("优化器"), m_pCboOptimizer);

        // 20260322 ZJH 调度器下拉框
        m_pCboScheduler = new QComboBox();
        m_pCboScheduler->addItem(QStringLiteral("CosineAnnealing"), static_cast<int>(om::SchedulerType::CosineAnnealing));
        m_pCboScheduler->addItem(QStringLiteral("StepLR"),          static_cast<int>(om::SchedulerType::StepLR));
        m_pCboScheduler->addItem(QStringLiteral("None"),            static_cast<int>(om::SchedulerType::None));
        pForm->addRow(QStringLiteral("调度器"), m_pCboScheduler);

        pLayout->addWidget(pGroup);
    }

    // ===== 超参数分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("超参数"), pContent);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);
        pForm->setSpacing(6);

        // 20260322 ZJH 学习率微调框 (0.0001~0.1, 步长 0.0001, 小数4位)
        m_pSpnLearningRate = new QDoubleSpinBox();
        m_pSpnLearningRate->setRange(0.0001, 0.1);
        m_pSpnLearningRate->setSingleStep(0.0001);
        m_pSpnLearningRate->setDecimals(4);
        m_pSpnLearningRate->setValue(0.001);
        pForm->addRow(QStringLiteral("学习率"), m_pSpnLearningRate);

        // 20260322 ZJH 批量大小微调框 (1~512)
        m_pSpnBatchSize = new QSpinBox();
        m_pSpnBatchSize->setRange(1, 512);
        m_pSpnBatchSize->setValue(16);
        // 20260324 ZJH 批量大小行：SpinBox + 自动最大内存按钮（水平布局）
        m_pBtnAutoMaxBatch = new QPushButton(QStringLiteral("自动最大"));
        m_pBtnAutoMaxBatch->setToolTip(QStringLiteral("根据可用内存自动选择最大批量大小"));
        m_pBtnAutoMaxBatch->setFixedWidth(70);
        QHBoxLayout* pBatchRow = new QHBoxLayout();  // 20260324 ZJH 水平布局容纳 SpinBox + 按钮
        pBatchRow->setContentsMargins(0, 0, 0, 0);
        pBatchRow->addWidget(m_pSpnBatchSize);
        pBatchRow->addWidget(m_pBtnAutoMaxBatch);
        pForm->addRow(QStringLiteral("批量大小"), pBatchRow);

        // 20260324 ZJH 自动最大批量大小按钮信号连接
        connect(m_pBtnAutoMaxBatch, &QPushButton::clicked, this, [this]() {
            // 20260324 ZJH 计算自动 batch size：使用输入尺寸^2 作为 inputDim
            int nInputSize = m_pSpnInputSize->value();  // 20260324 ZJH 当前输入尺寸
            int nInputDim = nInputSize * nInputSize;     // 20260324 ZJH 展平维度
            // 20260324 ZJH 粗略估算模型参数量（MLP ~100K, ResNet18 ~11M）
            int64_t nEstParams = 500000;  // 20260324 ZJH 默认 500K 参数估算
            int nNumClasses = 10;          // 20260324 ZJH 默认 10 类
            int nAutoBS = EngineBridge::autoSelectBatchSize(nInputDim, nNumClasses, nEstParams);
            m_pSpnBatchSize->setValue(nAutoBS);  // 20260324 ZJH 设置自动计算的 batch size
        });

        // 20260322 ZJH 训练轮次微调框 (1~1000)
        m_pSpnEpochs = new QSpinBox();
        m_pSpnEpochs->setRange(1, 1000);
        m_pSpnEpochs->setValue(50);
        pForm->addRow(QStringLiteral("训练轮次"), m_pSpnEpochs);

        // 20260330 ZJH 输入分辨率预设下拉框（借鉴海康 VisionTrain 分辨率预设）
        // 替代原自由 SpinBox，提供经验验证的分辨率档位
        m_pCboResolution = new QComboBox();
        // 20260330 ZJH 默认填充分类任务的预设（onEnter 时根据任务类型刷新）
        m_pCboResolution->addItem(QStringLiteral("224\u00d7224"), static_cast<int>(om::InputResolutionPreset::Res224));
        m_pCboResolution->addItem(QStringLiteral("320\u00d7320"), static_cast<int>(om::InputResolutionPreset::Res320));
        m_pCboResolution->addItem(QStringLiteral("512\u00d7512"), static_cast<int>(om::InputResolutionPreset::Res512));
        m_pCboResolution->addItem(QStringLiteral("自定义"),       static_cast<int>(om::InputResolutionPreset::Custom));
        pForm->addRow(QStringLiteral("输入分辨率"), m_pCboResolution);

        // 20260330 ZJH 自定义输入尺寸微调框（仅 Custom 时可见）
        m_pSpnInputSize = new QSpinBox();
        m_pSpnInputSize->setRange(32, 1024);
        m_pSpnInputSize->setSingleStep(32);
        m_pSpnInputSize->setValue(224);
        m_pSpnInputSize->setVisible(false);  // 20260330 ZJH 默认隐藏
        pForm->addRow(QStringLiteral("  自定义尺寸"), m_pSpnInputSize);

        // 20260330 ZJH 分辨率预设变更时：非 Custom 自动设置 inputSize，Custom 显示 SpinBox
        connect(m_pCboResolution, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
            auto ePreset = static_cast<om::InputResolutionPreset>(m_pCboResolution->currentData().toInt());
            if (ePreset == om::InputResolutionPreset::Custom) {
                m_pSpnInputSize->setVisible(true);  // 20260330 ZJH 显示自定义输入
            } else {
                m_pSpnInputSize->setVisible(false);  // 20260330 ZJH 隐藏自定义输入
                m_pSpnInputSize->setValue(om::resolutionPresetToPixels(ePreset));  // 20260330 ZJH 自动填充像素值
            }
        });

        // 20260324 ZJH 早停耐心微调框 (1~1000)，支持长训练
        m_pSpnPatience = new QSpinBox();
        m_pSpnPatience->setRange(1, 1000);  // 20260324 ZJH 最大值从 100 提升到 1000
        m_pSpnPatience->setValue(10);
        pForm->addRow(QStringLiteral("早停耐心"), m_pSpnPatience);

        pLayout->addWidget(pGroup);
    }

    // ===== 预训练模型与标识分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("模型选项"), pContent);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);
        pForm->setSpacing(6);

        // 20260330 ZJH 预训练模型路径行：编辑框 + 浏览按钮
        m_pEdtPretrained = new QLineEdit();
        m_pEdtPretrained->setPlaceholderText(QStringLiteral("无（从头训练）"));
        m_pEdtPretrained->setReadOnly(true);  // 20260330 ZJH 只读，通过浏览按钮选择
        m_pBtnBrowsePretrained = new QPushButton(QStringLiteral("..."));
        m_pBtnBrowsePretrained->setFixedWidth(36);
        m_pBtnBrowsePretrained->setToolTip(QStringLiteral("选择预训练模型文件 (.onnx / .omm / .pt)"));
        QHBoxLayout* pPretrainRow = new QHBoxLayout();
        pPretrainRow->setContentsMargins(0, 0, 0, 0);
        pPretrainRow->addWidget(m_pEdtPretrained);
        pPretrainRow->addWidget(m_pBtnBrowsePretrained);
        pForm->addRow(QStringLiteral("预训练模型"), pPretrainRow);

        // 20260330 ZJH 浏览预训练模型文件
        connect(m_pBtnBrowsePretrained, &QPushButton::clicked, this, [this]() {
            QString strFile = QFileDialog::getOpenFileName(this,
                QStringLiteral("选择预训练模型"),
                QString(),
                QStringLiteral("模型文件 (*.onnx *.omm *.pt *.pth);;所有文件 (*)"));
            if (!strFile.isEmpty()) {
                m_pEdtPretrained->setText(strFile);
            }
        });

        // 20260330 ZJH 模型标识输入框（仅允许数字/字母/下划线，借鉴海康 model_flag）
        m_pEdtModelTag = new QLineEdit();
        m_pEdtModelTag->setPlaceholderText(QStringLiteral("可选，仅数字/字母/下划线"));
        m_pEdtModelTag->setMaxLength(64);  // 20260330 ZJH 最大长度 64 字符
        m_pEdtModelTag->setValidator(new QRegularExpressionValidator(
            QRegularExpression(QStringLiteral("^[0-9a-zA-Z_]*$")), m_pEdtModelTag));
        pForm->addRow(QStringLiteral("模型标识"), m_pEdtModelTag);

        // 20260330 ZJH 少样本学习模式（在模型选项分组中）
        m_pChkFewShot = new QCheckBox(QStringLiteral("少样本学习模式 (5-10张/类)"));
        m_pChkFewShot->setChecked(false);  // 20260330 ZJH 默认关闭
        pForm->addRow(m_pChkFewShot);

        // 20260330 ZJH 每类样本数微调框（范围 1~20，默认 5）
        m_pSpnShotsPerClass = new QSpinBox();
        m_pSpnShotsPerClass->setRange(1, 20);  // 20260330 ZJH 少样本典型范围 1-20
        m_pSpnShotsPerClass->setValue(5);       // 20260330 ZJH 默认 5-shot
        m_pSpnShotsPerClass->setEnabled(false); // 20260330 ZJH 默认禁用，勾选后启用
        pForm->addRow(QStringLiteral("  每类样本数"), m_pSpnShotsPerClass);

        // 20260330 ZJH 少样本开关联动：勾选时启用 shots 微调框
        connect(m_pChkFewShot, &QCheckBox::toggled, m_pSpnShotsPerClass, &QSpinBox::setEnabled);

        pLayout->addWidget(pGroup);
    }

    // ===== 预训练权重分组 =====
    {
        // 20260330 ZJH 预训练权重选择（.omm 文件，可选加载预训练模型参数）
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("预训练权重"), pContent);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);  // 20260330 ZJH 标签右对齐
        pForm->setSpacing(6);  // 20260330 ZJH 行间距

        // 20260330 ZJH 使用预训练权重复选框
        m_pChkUsePretrained = new QCheckBox(QStringLiteral("使用预训练权重"));
        m_pChkUsePretrained->setChecked(false);  // 20260330 ZJH 默认不使用
        pForm->addRow(m_pChkUsePretrained);

        // 20260330 ZJH 预训练权重路径行：只读编辑框 + 浏览按钮
        m_pEdtPretrainedPath = new QLineEdit();
        m_pEdtPretrainedPath->setPlaceholderText(QStringLiteral("选择 .omm 预训练权重文件"));
        m_pEdtPretrainedPath->setReadOnly(true);   // 20260330 ZJH 只读，通过浏览按钮选择
        m_pEdtPretrainedPath->setEnabled(false);   // 20260330 ZJH 默认禁用
        m_pBtnBrowsePretrainedPath = new QPushButton(QStringLiteral("浏览..."));
        m_pBtnBrowsePretrainedPath->setFixedWidth(60);  // 20260330 ZJH 固定宽度
        m_pBtnBrowsePretrainedPath->setEnabled(false);  // 20260330 ZJH 默认禁用
        QHBoxLayout* pPretrainPathRow = new QHBoxLayout();
        pPretrainPathRow->setContentsMargins(0, 0, 0, 0);
        pPretrainPathRow->addWidget(m_pEdtPretrainedPath);
        pPretrainPathRow->addWidget(m_pBtnBrowsePretrainedPath);
        pForm->addRow(QStringLiteral("权重文件"), pPretrainPathRow);

        // 20260330 ZJH 复选框联动：勾选时启用路径编辑框和浏览按钮
        connect(m_pChkUsePretrained, &QCheckBox::toggled, this, [this](bool bChecked) {
            m_pEdtPretrainedPath->setEnabled(bChecked);    // 20260330 ZJH 联动路径框
            m_pBtnBrowsePretrainedPath->setEnabled(bChecked); // 20260330 ZJH 联动浏览按钮
            if (!bChecked) {
                m_pEdtPretrainedPath->clear();  // 20260330 ZJH 取消勾选时清空路径
            }
        });

        // 20260330 ZJH 浏览 .omm 预训练权重文件
        connect(m_pBtnBrowsePretrainedPath, &QPushButton::clicked, this, [this]() {
            QString strFile = QFileDialog::getOpenFileName(this,
                QStringLiteral("选择预训练权重文件"),
                QString(),
                QStringLiteral("OmniMatch 模型 (*.omm);;所有文件 (*)"));
            if (!strFile.isEmpty()) {
                m_pEdtPretrainedPath->setText(strFile);  // 20260330 ZJH 显示选中路径
            }
        });

        pLayout->addWidget(pGroup);
    }

    // ===== 模型优化分组 =====
    {
        // 20260330 ZJH 模型优化（训练后剪枝），可折叠分组
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("模型优化"), pContent);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);  // 20260330 ZJH 标签右对齐
        pForm->setSpacing(6);  // 20260330 ZJH 行间距

        // 20260330 ZJH 训练后剪枝启用复选框
        m_pChkPruning = new QCheckBox(QStringLiteral("训练后剪枝"));
        m_pChkPruning->setChecked(false);  // 20260330 ZJH 默认关闭
        pForm->addRow(m_pChkPruning);

        // 20260330 ZJH 剪枝方法下拉框
        m_pCboPruneMethod = new QComboBox();
        m_pCboPruneMethod->addItem(QStringLiteral("非结构化(Magnitude)"), 0);  // 20260330 ZJH 权重幅值剪枝
        m_pCboPruneMethod->addItem(QStringLiteral("结构化(Channel)"),     1);  // 20260330 ZJH 通道级剪枝
        m_pCboPruneMethod->setEnabled(false);  // 20260330 ZJH 默认禁用
        pForm->addRow(QStringLiteral("  剪枝方法"), m_pCboPruneMethod);

        // 20260330 ZJH 剪枝比例微调框（范围 0.1~0.8，步长 0.1，默认 0.3）
        m_pSpnPruneRatio = new QDoubleSpinBox();
        m_pSpnPruneRatio->setRange(0.1, 0.8);      // 20260330 ZJH 剪枝比例范围
        m_pSpnPruneRatio->setSingleStep(0.1);       // 20260330 ZJH 步长 0.1
        m_pSpnPruneRatio->setDecimals(1);           // 20260330 ZJH 1 位小数
        m_pSpnPruneRatio->setValue(0.3);            // 20260330 ZJH 默认 30%
        m_pSpnPruneRatio->setEnabled(false);        // 20260330 ZJH 默认禁用
        pForm->addRow(QStringLiteral("  剪枝比例"), m_pSpnPruneRatio);

        // 20260330 ZJH 剪枝开关联动：勾选时启用方法和比例控件
        connect(m_pChkPruning, &QCheckBox::toggled, this, [this](bool bChecked) {
            m_pCboPruneMethod->setEnabled(bChecked);   // 20260330 ZJH 联动剪枝方法
            m_pSpnPruneRatio->setEnabled(bChecked);    // 20260330 ZJH 联动剪枝比例
        });

        pLayout->addWidget(pGroup);
    }

    // ===== 增量训练 (Continual Learning) 分组 =====
    {
        // 20260402 ZJH [OPT-2.5] 增量训练入口 — 在已有模型基础上增量训练新类别
        // 使用 EWC 正则化（Elastic Weight Consolidation）防止灾难性遗忘
        // 对标 Halcon 25.11 的 Continual Learning 功能
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("增量训练"), pContent);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);  // 20260402 ZJH 标签右对齐
        pForm->setSpacing(6);  // 20260402 ZJH 行间距

        // 20260402 ZJH 增量训练启用复选框
        // 勾选后: (1) 自动加载现有模型权重 (2) 使用 EWC 正则化 (3) 降低学习率
        m_pChkContinualLearning = new QCheckBox(QStringLiteral("增量训练 (Continual Learning)"));
        m_pChkContinualLearning->setChecked(false);  // 20260402 ZJH 默认关闭
        m_pChkContinualLearning->setToolTip(QStringLiteral(
            "在已有模型基础上增量训练新类别，不遗忘旧类别。\n"
            "需要先加载预训练模型权重（模型选项中设置）。\n"
            "使用 EWC 正则化防止灾难性遗忘。"));
        pForm->addRow(m_pChkContinualLearning);

        // 20260402 ZJH EWC 正则化系数微调框（范围 1~100000，默认 1000）
        // λ 越大 → 旧知识保留越好，但新任务学习能力受限
        // λ 越小 → 新任务学习越快，但可能遗忘旧知识
        m_pSpnEwcLambda = new QDoubleSpinBox();
        m_pSpnEwcLambda->setRange(1.0, 100000.0);     // 20260402 ZJH 正则化系数范围
        m_pSpnEwcLambda->setSingleStep(100.0);         // 20260402 ZJH 步长 100
        m_pSpnEwcLambda->setDecimals(0);               // 20260402 ZJH 整数显示
        m_pSpnEwcLambda->setValue(1000.0);             // 20260402 ZJH 默认 1000（工业推荐值）
        m_pSpnEwcLambda->setEnabled(false);            // 20260402 ZJH 默认禁用，勾选增量训练后启用
        m_pSpnEwcLambda->setToolTip(QStringLiteral(
            "EWC 正则化系数 (λ):\n"
            "值越大 → 旧知识保留越好，新任务学习受限\n"
            "值越小 → 新任务学习越快，可能遗忘旧知识\n"
            "推荐范围: 100~10000"));
        pForm->addRow(QStringLiteral("  EWC λ 系数"), m_pSpnEwcLambda);

        // 20260402 ZJH 增量训练开关联动：勾选时启用 EWC λ 微调框
        connect(m_pChkContinualLearning, &QCheckBox::toggled, this, [this](bool bChecked) {
            m_pSpnEwcLambda->setEnabled(bChecked);  // 20260402 ZJH 联动 EWC λ 控件
        });

        pLayout->addWidget(pGroup);
    }

    // ===== 数据增强分组 =====
    {
        // 20260330 ZJH 增强策略预设下拉（借鉴海康 VisionTrain "数据增强策略"）
        m_pCboAugPreset = new QComboBox();
        m_pCboAugPreset->addItem(QStringLiteral("默认配置 — 自动最优"), static_cast<int>(om::AugmentationPreset::Default));
        m_pCboAugPreset->addItem(QStringLiteral("手动配置 — 精细调节"), static_cast<int>(om::AugmentationPreset::Manual));
        pLayout->addWidget(m_pCboAugPreset);

        // 20260322 ZJH 启用增强总开关
        m_pChkAugmentation = new QCheckBox(QStringLiteral("启用数据增强"));
        m_pChkAugmentation->setChecked(true);
        pLayout->addWidget(m_pChkAugmentation);

        // 20260325 ZJH 展开/收起增强选项按钮
        m_pBtnToggleAug = new QPushButton(QStringLiteral("▶ 增强选项..."));
        m_pBtnToggleAug->setStyleSheet(QStringLiteral(
            "QPushButton { background: transparent; color: #60a5fa; border: none;"
            "  text-align: left; padding: 2px 0; font-size: 12px; }"
            "QPushButton:hover { color: #93c5fd; }"));
        pLayout->addWidget(m_pBtnToggleAug);

        // 20260325 ZJH 可折叠容器：包含全部增强选项分组
        m_pAugContainer = new QWidget(pContent);
        QVBoxLayout* pAugLayout = new QVBoxLayout(m_pAugContainer);
        pAugLayout->setContentsMargins(0, 0, 0, 0);
        pAugLayout->setSpacing(8);

        // 20260325 ZJH 连接展开/收起信号
        connect(m_pBtnToggleAug, &QPushButton::clicked, this, [this]() {
            bool bShow = !m_pAugContainer->isVisible();
            m_pAugContainer->setVisible(bShow);
            m_pBtnToggleAug->setText(bShow
                ? QStringLiteral("▼ 收起增强选项")
                : QStringLiteral("▶ 增强选项..."));
        });

        // 20260324 ZJH --- 几何变换组 ---
        QGroupBox* pGrpGeom = new QGroupBox(QStringLiteral("几何变换"), m_pAugContainer);
        QFormLayout* pFormGeom = new QFormLayout(pGrpGeom);
        pFormGeom->setLabelAlignment(Qt::AlignRight);  // 20260324 ZJH 标签右对齐
        pFormGeom->setSpacing(6);  // 20260324 ZJH 行间距

        // 20260322 ZJH 水平翻转概率微调框 (0~1, 步长 0.05)
        m_pSpnFlipProb = new QDoubleSpinBox();
        m_pSpnFlipProb->setRange(0, 1.0);
        m_pSpnFlipProb->setSingleStep(0.05);
        m_pSpnFlipProb->setDecimals(2);
        m_pSpnFlipProb->setValue(0.5);
        pFormGeom->addRow(QStringLiteral("水平翻转概率"), m_pSpnFlipProb);

        // 20260324 ZJH 垂直翻转概率微调框 (0~1, 步长 0.05)
        m_pSpnVerticalFlipProb = new QDoubleSpinBox();
        m_pSpnVerticalFlipProb->setRange(0, 1.0);
        m_pSpnVerticalFlipProb->setSingleStep(0.05);
        m_pSpnVerticalFlipProb->setDecimals(2);
        m_pSpnVerticalFlipProb->setValue(0.0);
        pFormGeom->addRow(QStringLiteral("垂直翻转概率"), m_pSpnVerticalFlipProb);

        // 20260322 ZJH 旋转角度微调框 (0~180, 步长 5)
        m_pSpnRotation = new QDoubleSpinBox();
        m_pSpnRotation->setRange(0, 180.0);
        m_pSpnRotation->setSingleStep(5.0);
        m_pSpnRotation->setDecimals(1);
        m_pSpnRotation->setValue(15.0);
        pFormGeom->addRow(QStringLiteral("旋转角度"), m_pSpnRotation);

        // 20260324 ZJH 仿射变换复选框
        m_pChkAffine = new QCheckBox(QStringLiteral("仿射变换"));
        m_pChkAffine->setChecked(false);
        pFormGeom->addRow(m_pChkAffine);

        // 20260324 ZJH 仿射剪切角度微调框 (0~45, 步长 1)
        m_pSpnShearDeg = new QDoubleSpinBox();
        m_pSpnShearDeg->setRange(0, 45.0);
        m_pSpnShearDeg->setSingleStep(1.0);
        m_pSpnShearDeg->setDecimals(1);
        m_pSpnShearDeg->setValue(10.0);
        pFormGeom->addRow(QStringLiteral("  剪切角度"), m_pSpnShearDeg);

        // 20260324 ZJH 仿射平移比例微调框 (0~0.5, 步长 0.01)
        m_pSpnTranslate = new QDoubleSpinBox();
        m_pSpnTranslate->setRange(0, 0.5);
        m_pSpnTranslate->setSingleStep(0.01);
        m_pSpnTranslate->setDecimals(2);
        m_pSpnTranslate->setValue(0.1);
        pFormGeom->addRow(QStringLiteral("  平移比例"), m_pSpnTranslate);

        // 20260324 ZJH 随机缩放裁剪复选框
        m_pChkRandomCrop = new QCheckBox(QStringLiteral("随机缩放裁剪"));
        m_pChkRandomCrop->setChecked(true);
        pFormGeom->addRow(m_pChkRandomCrop);

        // 20260324 ZJH 裁剪最小缩放比例微调框 (0.3~1.0, 步长 0.05)
        m_pSpnCropScale = new QDoubleSpinBox();
        m_pSpnCropScale->setRange(0.3, 1.0);
        m_pSpnCropScale->setSingleStep(0.05);
        m_pSpnCropScale->setDecimals(2);
        m_pSpnCropScale->setValue(0.8);
        pFormGeom->addRow(QStringLiteral("  最小缩放"), m_pSpnCropScale);

        pAugLayout->addWidget(pGrpGeom);

        // 20260324 ZJH --- 颜色变换组 ---
        QGroupBox* pGrpColor = new QGroupBox(QStringLiteral("颜色变换"), m_pAugContainer);
        QFormLayout* pFormColor = new QFormLayout(pGrpColor);
        pFormColor->setLabelAlignment(Qt::AlignRight);  // 20260324 ZJH 标签右对齐
        pFormColor->setSpacing(6);  // 20260324 ZJH 行间距

        // 20260322 ZJH 亮度微调框 (0~1, 步长 0.05)
        m_pSpnBrightness = new QDoubleSpinBox();
        m_pSpnBrightness->setRange(0, 1.0);
        m_pSpnBrightness->setSingleStep(0.05);
        m_pSpnBrightness->setDecimals(2);
        m_pSpnBrightness->setValue(0.2);
        pFormColor->addRow(QStringLiteral("亮度"), m_pSpnBrightness);

        // 20260324 ZJH 颜色抖动复选框
        m_pChkColorJitter = new QCheckBox(QStringLiteral("颜色抖动"));
        m_pChkColorJitter->setChecked(true);
        pFormColor->addRow(m_pChkColorJitter);

        // 20260324 ZJH 饱和度抖动范围微调框 (0~1, 步长 0.05)
        m_pSpnSaturation = new QDoubleSpinBox();
        m_pSpnSaturation->setRange(0, 1.0);
        m_pSpnSaturation->setSingleStep(0.05);
        m_pSpnSaturation->setDecimals(2);
        m_pSpnSaturation->setValue(0.2);
        pFormColor->addRow(QStringLiteral("  饱和度"), m_pSpnSaturation);

        // 20260324 ZJH 色调抖动范围微调框 (0~0.5, 步长 0.01)
        m_pSpnHue = new QDoubleSpinBox();
        m_pSpnHue->setRange(0, 0.5);
        m_pSpnHue->setSingleStep(0.01);
        m_pSpnHue->setDecimals(2);
        m_pSpnHue->setValue(0.02);
        pFormColor->addRow(QStringLiteral("  色调"), m_pSpnHue);

        pAugLayout->addWidget(pGrpColor);

        // 20260330 ZJH --- HSV 色彩空间抖动组 ---
        QGroupBox* pGrpHsv = new QGroupBox(QStringLiteral("HSV 色彩空间抖动"), m_pAugContainer);
        QFormLayout* pFormHsv = new QFormLayout(pGrpHsv);
        pFormHsv->setLabelAlignment(Qt::AlignRight);  // 20260330 ZJH 标签右对齐
        pFormHsv->setSpacing(6);  // 20260330 ZJH 行间距

        // 20260330 ZJH HSV 增强启用复选框
        m_pChkHsvAugment = new QCheckBox(QStringLiteral("HSV 色彩空间抖动"));
        m_pChkHsvAugment->setChecked(false);  // 20260330 ZJH 默认关闭
        pFormHsv->addRow(m_pChkHsvAugment);

        // 20260330 ZJH 色调偏移微调框（范围 0~128，默认 5）
        m_pSpnHueShift = new QDoubleSpinBox();
        m_pSpnHueShift->setRange(0, 128.0);      // 20260330 ZJH 色调偏移范围
        m_pSpnHueShift->setSingleStep(1.0);       // 20260330 ZJH 步长 1
        m_pSpnHueShift->setDecimals(1);           // 20260330 ZJH 1 位小数
        m_pSpnHueShift->setValue(5.0);            // 20260330 ZJH 默认偏移 5
        m_pSpnHueShift->setEnabled(false);        // 20260330 ZJH 默认禁用
        pFormHsv->addRow(QStringLiteral("  色调偏移"), m_pSpnHueShift);

        // 20260330 ZJH 饱和度偏移微调框（范围 0~255，默认 5）
        m_pSpnSatShift = new QDoubleSpinBox();
        m_pSpnSatShift->setRange(0, 255.0);      // 20260330 ZJH 饱和度偏移范围
        m_pSpnSatShift->setSingleStep(1.0);       // 20260330 ZJH 步长 1
        m_pSpnSatShift->setDecimals(1);           // 20260330 ZJH 1 位小数
        m_pSpnSatShift->setValue(5.0);            // 20260330 ZJH 默认偏移 5
        m_pSpnSatShift->setEnabled(false);        // 20260330 ZJH 默认禁用
        pFormHsv->addRow(QStringLiteral("  饱和度偏移"), m_pSpnSatShift);

        // 20260330 ZJH 明度偏移微调框（范围 0~255，默认 5）
        m_pSpnValShift = new QDoubleSpinBox();
        m_pSpnValShift->setRange(0, 255.0);      // 20260330 ZJH 明度偏移范围
        m_pSpnValShift->setSingleStep(1.0);       // 20260330 ZJH 步长 1
        m_pSpnValShift->setDecimals(1);           // 20260330 ZJH 1 位小数
        m_pSpnValShift->setValue(5.0);            // 20260330 ZJH 默认偏移 5
        m_pSpnValShift->setEnabled(false);        // 20260330 ZJH 默认禁用
        pFormHsv->addRow(QStringLiteral("  明度偏移"), m_pSpnValShift);

        // 20260330 ZJH HSV 开关联动：勾选时启用三个偏移微调框
        connect(m_pChkHsvAugment, &QCheckBox::toggled, this, [this](bool bChecked) {
            m_pSpnHueShift->setEnabled(bChecked);  // 20260330 ZJH 联动色调偏移
            m_pSpnSatShift->setEnabled(bChecked);  // 20260330 ZJH 联动饱和度偏移
            m_pSpnValShift->setEnabled(bChecked);  // 20260330 ZJH 联动明度偏移
        });

        pAugLayout->addWidget(pGrpHsv);

        // 20260324 ZJH --- 噪声/遮挡组 ---
        QGroupBox* pGrpNoise = new QGroupBox(QStringLiteral("噪声 / 遮挡"), m_pAugContainer);
        QFormLayout* pFormNoise = new QFormLayout(pGrpNoise);
        pFormNoise->setLabelAlignment(Qt::AlignRight);  // 20260324 ZJH 标签右对齐
        pFormNoise->setSpacing(6);  // 20260324 ZJH 行间距

        // 20260324 ZJH 高斯噪声复选框
        m_pChkGaussianNoise = new QCheckBox(QStringLiteral("高斯噪声"));
        m_pChkGaussianNoise->setChecked(false);
        pFormNoise->addRow(m_pChkGaussianNoise);

        // 20260324 ZJH 高斯噪声标准差微调框 (0.001~0.2, 步长 0.005)
        m_pSpnNoiseStd = new QDoubleSpinBox();
        m_pSpnNoiseStd->setRange(0.001, 0.2);
        m_pSpnNoiseStd->setSingleStep(0.005);
        m_pSpnNoiseStd->setDecimals(3);
        m_pSpnNoiseStd->setValue(0.02);
        pFormNoise->addRow(QStringLiteral("  标准差"), m_pSpnNoiseStd);

        // 20260324 ZJH 高斯模糊复选框
        m_pChkGaussianBlur = new QCheckBox(QStringLiteral("高斯模糊"));
        m_pChkGaussianBlur->setChecked(false);
        pFormNoise->addRow(m_pChkGaussianBlur);

        // 20260324 ZJH 高斯模糊 sigma 微调框 (0.1~5.0, 步长 0.1)
        m_pSpnBlurSigma = new QDoubleSpinBox();
        m_pSpnBlurSigma->setRange(0.1, 5.0);
        m_pSpnBlurSigma->setSingleStep(0.1);
        m_pSpnBlurSigma->setDecimals(1);
        m_pSpnBlurSigma->setValue(1.0);
        pFormNoise->addRow(QStringLiteral("  Sigma"), m_pSpnBlurSigma);

        // 20260324 ZJH 随机擦除复选框
        m_pChkRandomErasing = new QCheckBox(QStringLiteral("随机擦除"));
        m_pChkRandomErasing->setChecked(false);
        pFormNoise->addRow(m_pChkRandomErasing);

        // 20260324 ZJH 擦除概率微调框 (0~1, 步长 0.05)
        m_pSpnErasingProb = new QDoubleSpinBox();
        m_pSpnErasingProb->setRange(0, 1.0);
        m_pSpnErasingProb->setSingleStep(0.05);
        m_pSpnErasingProb->setDecimals(2);
        m_pSpnErasingProb->setValue(0.3);
        pFormNoise->addRow(QStringLiteral("  擦除概率"), m_pSpnErasingProb);

        // 20260324 ZJH 擦除面积比例微调框 (0.01~0.5, 步长 0.01)
        m_pSpnErasingRatio = new QDoubleSpinBox();
        m_pSpnErasingRatio->setRange(0.01, 0.5);
        m_pSpnErasingRatio->setSingleStep(0.01);
        m_pSpnErasingRatio->setDecimals(2);
        m_pSpnErasingRatio->setValue(0.15);
        pFormNoise->addRow(QStringLiteral("  擦除面积"), m_pSpnErasingRatio);

        pAugLayout->addWidget(pGrpNoise);

        // 20260324 ZJH --- 高级混合组 ---
        QGroupBox* pGrpMix = new QGroupBox(QStringLiteral("高级混合"), m_pAugContainer);
        QFormLayout* pFormMix = new QFormLayout(pGrpMix);
        pFormMix->setLabelAlignment(Qt::AlignRight);  // 20260324 ZJH 标签右对齐
        pFormMix->setSpacing(6);  // 20260324 ZJH 行间距

        // 20260324 ZJH Mixup 复选框
        m_pChkMixup = new QCheckBox(QStringLiteral("Mixup"));
        m_pChkMixup->setChecked(false);
        pFormMix->addRow(m_pChkMixup);

        // 20260324 ZJH Mixup alpha 微调框 (0.01~2.0, 步长 0.05)
        m_pSpnMixupAlpha = new QDoubleSpinBox();
        m_pSpnMixupAlpha->setRange(0.01, 2.0);
        m_pSpnMixupAlpha->setSingleStep(0.05);
        m_pSpnMixupAlpha->setDecimals(2);
        m_pSpnMixupAlpha->setValue(0.2);
        pFormMix->addRow(QStringLiteral("  Alpha"), m_pSpnMixupAlpha);

        // 20260324 ZJH CutMix 复选框
        m_pChkCutMix = new QCheckBox(QStringLiteral("CutMix"));
        m_pChkCutMix->setChecked(false);
        pFormMix->addRow(m_pChkCutMix);

        // 20260324 ZJH CutMix alpha 微调框 (0.01~2.0, 步长 0.05)
        m_pSpnCutMixAlpha = new QDoubleSpinBox();
        m_pSpnCutMixAlpha->setRange(0.01, 2.0);
        m_pSpnCutMixAlpha->setSingleStep(0.05);
        m_pSpnCutMixAlpha->setDecimals(2);
        m_pSpnCutMixAlpha->setValue(1.0);
        pFormMix->addRow(QStringLiteral("  Alpha"), m_pSpnCutMixAlpha);

        pAugLayout->addWidget(pGrpMix);

        // 20260325 ZJH 将可折叠容器加入主布局，默认收起
        pLayout->addWidget(m_pAugContainer);
        m_pAugContainer->setVisible(false);

        // 20260330 ZJH 增强预设联动：默认配置隐藏手动参数区域，手动配置显示
        connect(m_pCboAugPreset, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
            auto ePreset = static_cast<om::AugmentationPreset>(m_pCboAugPreset->currentData().toInt());
            bool bManual = (ePreset == om::AugmentationPreset::Manual);
            // 20260330 ZJH 手动模式显示增强开关和展开按钮，默认模式全部隐藏
            m_pChkAugmentation->setVisible(bManual);
            m_pBtnToggleAug->setVisible(bManual);
            if (!bManual) {
                m_pAugContainer->setVisible(false);  // 20260330 ZJH 默认模式收起增强参数
                m_pBtnToggleAug->setText(QStringLiteral("\u25b6 增强选项..."));
            }
        });
        // 20260330 ZJH 初始状态：默认配置，隐藏手动参数
        m_pChkAugmentation->setVisible(false);
        m_pBtnToggleAug->setVisible(false);

        // 20260330 ZJH AI 数据合成按钮（在增强区域底部）
        m_pBtnDataSynth = new QPushButton(QStringLiteral("AI 数据合成..."));
        m_pBtnDataSynth->setToolTip(QStringLiteral("基于现有样本使用 AI 自动生成更多训练数据"));
        m_pBtnDataSynth->setStyleSheet(QStringLiteral(
            "QPushButton {"
            "  background-color: #1a1d24; color: #60a5fa; border: 1px solid #3d4455;"
            "  border-radius: 4px; padding: 6px 12px; font-size: 12px;"
            "}"
            "QPushButton:hover { border-color: #2563eb; color: #93c5fd; }"
            "QPushButton:pressed { background-color: #2563eb; color: white; }"));
        pLayout->addWidget(m_pBtnDataSynth);

        // 20260330 ZJH AI 数据合成按钮点击：打开 DataSynthesisDialog
        connect(m_pBtnDataSynth, &QPushButton::clicked, this, [this]() {
          try {
            std::cerr << "[DataSynth] button clicked" << std::endl;

            auto* pApp = Application::instance();
            if (!pApp || !pApp->currentProject()) {
                QMessageBox::warning(this,
                    QStringLiteral("AI \u6570\u636e\u5408\u6210"),
                    QStringLiteral("\u8bf7\u5148\u6253\u5f00\u6216\u521b\u5efa\u4e00\u4e2a\u9879\u76ee\u3002"));
                // "请先打开或创建一个项目。"
                return;
            }

            auto* pDataset = pApp->currentProject()->dataset();
            if (!pDataset || pDataset->imageCount() == 0) {
                QMessageBox::warning(this,
                    QStringLiteral("AI \u6570\u636e\u5408\u6210"),
                    QStringLiteral("\u6570\u636e\u96c6\u4e3a\u7a7a\uff0c"
                                   "\u8bf7\u5148\u5728\u56fe\u5e93\u9875\u5bfc\u5165\u56fe\u50cf\u3002"));
                // "数据集为空，请先在图库页导入图像。"
                return;
            }

            // 20260330 ZJH 合成用缩略图尺寸（256 足够合成，避免 1024 导致内存爆炸）
            int nSynthSize = 256;

            std::cerr << "[DataSynth] loading images at " << nSynthSize << "x" << nSynthSize << std::endl;

            std::vector<std::vector<float>> vecNormalImages;
            std::vector<std::vector<float>> vecDefectImages;

            const auto& vecImages = pDataset->images();
            for (int idx = 0; idx < static_cast<int>(vecImages.size()); ++idx) {
                const auto& entry = vecImages[idx];
                try {
                    QImage qimg(entry.strFilePath);
                    if (qimg.isNull()) continue;
                    qimg = qimg.scaled(nSynthSize, nSynthSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
                    qimg = qimg.convertToFormat(QImage::Format_RGB888);

                    int nH = qimg.height(), nW = qimg.width();
                    int nPixels = nH * nW;
                    std::vector<float> vecPx(static_cast<size_t>(3 * nPixels));
                    for (int y = 0; y < nH; ++y) {
                        const uchar* pLine = qimg.constScanLine(y);
                        for (int x = 0; x < nW; ++x) {
                            int i = y * nW + x;
                            vecPx[0 * nPixels + i] = pLine[x * 3 + 0] / 255.0f;
                            vecPx[1 * nPixels + i] = pLine[x * 3 + 1] / 255.0f;
                            vecPx[2 * nPixels + i] = pLine[x * 3 + 2] / 255.0f;
                        }
                    }
                    if (entry.nLabelId >= 0)
                        vecDefectImages.push_back(std::move(vecPx));
                    else
                        vecNormalImages.push_back(std::move(vecPx));
                } catch (...) {
                    std::cerr << "[DataSynth] skip image #" << idx << std::endl;
                }
            }

            if (vecDefectImages.empty()) vecDefectImages = vecNormalImages;
            if (vecNormalImages.empty()) vecNormalImages = vecDefectImages;

            std::cerr << "[DataSynth] loaded: normal=" << vecNormalImages.size()
                      << " defect=" << vecDefectImages.size() << std::endl;

            // 20260330 ZJH 打开数据合成对话框
            DataSynthesisDialog dlgFull(this);
            dlgFull.setSourceData(vecNormalImages, vecDefectImages, 3, nSynthSize, nSynthSize);

            if (dlgFull.exec() == QDialog::Accepted) {
                auto synthResult = dlgFull.getResult();
                // 20260330 ZJH 记录合成结果到训练日志
                onTrainingLog(QStringLiteral(
                    "[\u4fe1\u606f] AI \u6570\u636e\u5408\u6210\u5b8c\u6210: "
                    "\u751f\u6210\u4e86 %1 \u5f20\u5408\u6210\u8bad\u7ec3\u56fe\u50cf")
                    .arg(synthResult.nSynthCount));
                // "[信息] AI 数据合成完成: 生成了 N 张合成训练图像"
            }
          } catch (const std::exception& e) {
            std::cerr << "[DataSynth] EXCEPTION: " << e.what() << std::endl;
            QMessageBox::critical(this, QStringLiteral("AI 数据合成"),
                QStringLiteral("发生异常: %1").arg(QString::fromStdString(e.what())));
          } catch (...) {
            std::cerr << "[DataSynth] UNKNOWN EXCEPTION" << std::endl;
            QMessageBox::critical(this, QStringLiteral("AI 数据合成"),
                QStringLiteral("发生未知异常"));
          }
        });
    }

    // ===== ONNX 导出 =====
    {
        m_pChkExportOnnx = new QCheckBox(QStringLiteral("训练完成后导出 ONNX 模型"));
        m_pChkExportOnnx->setChecked(true);
        pLayout->addWidget(m_pChkExportOnnx);
    }

    // ===== 自动推荐按钮 =====
    {
        m_pBtnAutoRecommend = new QPushButton(QStringLiteral("⚡ 一键推荐最优参数"));
        m_pBtnAutoRecommend->setToolTip(QStringLiteral("根据任务类型、数据集大小和硬件自动推荐最优训练参数"));
        m_pBtnAutoRecommend->setStyleSheet(QStringLiteral(
            "QPushButton {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2563eb, stop:1 #7c3aed);"
            "  color: white; border: none; border-radius: 6px;"
            "  padding: 8px 12px; font-weight: bold; font-size: 12px;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);"
            "}"
            "QPushButton:pressed { background: #1d4ed8; }"));
        pLayout->addWidget(m_pBtnAutoRecommend);

        // 20260325 ZJH 自动推荐逻辑：根据任务类型/模型架构/数据集/硬件综合推荐
        connect(m_pBtnAutoRecommend, &QPushButton::clicked, this, [this]() {
            // 20260325 ZJH 获取当前选择的架构和任务类型
            auto eArch = static_cast<om::ModelArchitecture>(
                m_pCboArchitecture->currentData().toInt());
            om::TaskType eTask = om::TaskType::Classification;
            if (m_pProject) eTask = m_pProject->taskType();

            // 20260325 ZJH 获取数据集大小
            int nImageCount = 0;
            if (m_pProject && m_pProject->dataset()) {
                nImageCount = static_cast<int>(m_pProject->dataset()->images().size());
            }

            // 20260325 ZJH 检测 GPU 可用显存
            bool bHasGpu = false;
            size_t nGpuFreeMB = 0;
            // 20260325 ZJH 通过 EngineBridge 静态方法查询 GPU 可用显存
            {
                int nInputDimDummy = 3 * 224 * 224;
                int nEstBatch = EngineBridge::autoSelectBatchSize(nInputDimDummy, 10, 500000);
                // 20260325 ZJH 如果 autoSelectBatchSize 返回较大值，说明有大量可用内存（可能是 GPU）
                // 用 nEstBatch 间接判断：GPU 通常返回较大 batch
                (void)nEstBatch;
            }
            // 20260325 ZJH 简化判断：设备选择框有 CUDA 选项即视为有 GPU
            if (m_pCboDevice->findData(static_cast<int>(om::DeviceType::CUDA)) >= 0) {
                bHasGpu = true;
                nGpuFreeMB = 15000;  // 20260325 ZJH 保守估计 ~15GB（实际值需 CUDA API 查询）
            }

            // 20260325 ZJH ===== 推荐设备 =====
            if (bHasGpu) {
                int nDevIdx = m_pCboDevice->findData(static_cast<int>(om::DeviceType::CUDA));
                if (nDevIdx >= 0) m_pCboDevice->setCurrentIndex(nDevIdx);
            }

            // 20260325 ZJH ===== 根据架构类型推荐输入尺寸 =====
            int nRecInputSize = 224;  // 20260325 ZJH 默认 224（分类标准）
            bool bIsCnn = true;
            switch (eArch) {
                // 20260325 ZJH 分类模型：标准 224
                case om::ModelArchitecture::ResNet18:
                case om::ModelArchitecture::ResNet50:
                case om::ModelArchitecture::EfficientNetB0:
                case om::ModelArchitecture::MobileNetV4Small:
                case om::ModelArchitecture::MobileNetV4Medium:
                case om::ModelArchitecture::ConvNeXtTiny:
                case om::ModelArchitecture::RepVGGA0:
                    nRecInputSize = 224;
                    break;
                case om::ModelArchitecture::ViTTiny:
                    nRecInputSize = 224;  // 20260325 ZJH ViT patch=16 需要可整除
                    break;
                // 20260325 ZJH 检测模型：640 标准
                case om::ModelArchitecture::YOLOv5Nano:
                case om::ModelArchitecture::YOLOv8Nano:
                case om::ModelArchitecture::YOLOv11Nano:
                case om::ModelArchitecture::RTDETR:
                    nRecInputSize = 416;  // 20260325 ZJH 16GB GPU 用 416，更大 GPU 可 640
                    if (bHasGpu && nGpuFreeMB > 12000) nRecInputSize = 640;
                    break;
                // 20260325 ZJH 分割模型：256~512
                case om::ModelArchitecture::UNet:
                case om::ModelArchitecture::DeepLabV3Plus:
                case om::ModelArchitecture::MobileSegNet:  // 20260401 ZJH 轻量分割
                case om::ModelArchitecture::PSPNet:
                case om::ModelArchitecture::SegFormer:
                    nRecInputSize = 256;
                    if (bHasGpu && nGpuFreeMB > 12000) nRecInputSize = 512;
                    break;
                // 20260325 ZJH 实例分割：256~512
                case om::ModelArchitecture::YOLOv8InstanceSeg:
                case om::ModelArchitecture::MaskRCNN:
                    nRecInputSize = 256;
                    if (bHasGpu && nGpuFreeMB > 12000) nRecInputSize = 416;
                    break;
                // 20260325 ZJH 异常检测：256
                case om::ModelArchitecture::PaDiM:
                case om::ModelArchitecture::PatchCore:
                case om::ModelArchitecture::EfficientAD:
                case om::ModelArchitecture::FastFlow:
                    nRecInputSize = 256;
                    break;
                default:
                    nRecInputSize = 224;
                    bIsCnn = false;  // 20260325 ZJH MLP 等非 CNN
                    break;
            }
            m_pSpnInputSize->setValue(nRecInputSize);

            // 20260325 ZJH ===== 推荐批量大小（基于显存/内存和输入尺寸）=====
            int nRecBatch = 16;
            if (bHasGpu && bIsCnn) {
                // 20260325 ZJH GPU CNN: 根据显存和输入尺寸估算
                // CNN 中间激活值 + im2col 缓冲极大，系数需远大于 MLP
                // 经验公式: batch ≈ GPU_MB / (inputSize² × 0.05)
                // 例: 16GB GPU + 224×224 → 16000/(224²×0.05) = 6.4 → 4
                //     16GB GPU + 416×416 → 16000/(416²×0.05) = 1.8 → 1
                double dMemFactor = static_cast<double>(nGpuFreeMB) /
                    (static_cast<double>(nRecInputSize) * nRecInputSize * 0.05);
                nRecBatch = std::max(1, std::min(32, static_cast<int>(dMemFactor)));
                // 20260325 ZJH 向下取 2 的幂
                int nPow2 = 1;
                while (nPow2 * 2 <= nRecBatch) nPow2 *= 2;
                nRecBatch = nPow2;
            } else if (!bIsCnn) {
                nRecBatch = 64;  // 20260325 ZJH MLP 可用大 batch
            }
            // 20260325 ZJH 不超过数据集大小
            if (nImageCount > 0) {
                nRecBatch = std::min(nRecBatch, nImageCount);
            }
            m_pSpnBatchSize->setValue(nRecBatch);

            // 20260330 ZJH ===== 推荐学习率（基于数据集大小和任务类型，借鉴海康经验规则）=====
            double dRecLR = om::recommendLearningRate(nImageCount, eTask);
            m_pSpnLearningRate->setValue(dRecLR);

            // 20260330 ZJH ===== 推荐训练轮数（基于数据集大小和任务类型，借鉴海康经验规则）=====
            // 海康规则: >250张→100-200轮, <250张→适当增加, <50张→500-1000轮
            int nRecEpochs = om::recommendEpochs(nImageCount, eTask);
            m_pSpnEpochs->setValue(nRecEpochs);

            // 20260325 ZJH ===== 推荐早停耐心 =====
            m_pSpnPatience->setValue(std::max(5, nRecEpochs / 5));

            // 20260325 ZJH ===== 推荐优化器和调度器 =====
            int nOptIdx = m_pCboOptimizer->findData(static_cast<int>(om::OptimizerType::Adam));
            if (nOptIdx >= 0) m_pCboOptimizer->setCurrentIndex(nOptIdx);
            int nSchIdx = m_pCboScheduler->findData(static_cast<int>(om::SchedulerType::CosineAnnealing));
            if (nSchIdx >= 0) m_pCboScheduler->setCurrentIndex(nSchIdx);

            // 20260325 ZJH ===== 推荐数据增强 =====
            m_pChkAugmentation->setChecked(true);
            m_pSpnFlipProb->setValue(0.5);
            m_pSpnRotation->setValue(15.0);
            m_pSpnBrightness->setValue(0.2);
            m_pSpnVerticalFlipProb->setValue(0.0);
            m_pChkColorJitter->setChecked(true);
            m_pSpnSaturation->setValue(0.2);
            m_pSpnHue->setValue(0.02);
            m_pChkRandomCrop->setChecked(true);
            m_pSpnCropScale->setValue(0.8);

            // 20260325 ZJH 弹窗提示推荐结果
            QString strMsg = QStringLiteral(
                "已自动推荐最优参数:\n\n"
                "• 设备: %1\n"
                "• 输入尺寸: %2×%2\n"
                "• 批量大小: %3\n"
                "• 学习率: 0.001 (Adam)\n"
                "• 训练轮数: %4 (早停耐心 %5)\n"
                "• 调度器: CosineAnnealing\n"
                "• 数据增强: 已开启\n")
                .arg(bHasGpu ? "CUDA GPU" : "CPU")
                .arg(nRecInputSize)
                .arg(nRecBatch)
                .arg(nRecEpochs)
                .arg(std::max(5, nRecEpochs / 5));

            if (bHasGpu) {
                strMsg += QStringLiteral("\nGPU 可用显存: %1 MB").arg(nGpuFreeMB);
            }
            if (nImageCount > 0) {
                strMsg += QStringLiteral("\n数据集图像数: %1").arg(nImageCount);
            }

            QMessageBox::information(this, QStringLiteral("参数推荐"), strMsg);
        });
    }

    // 20260322 ZJH 添加弹性空间，使分组框紧凑排列在顶部
    pLayout->addStretch(1);

    pScroll->setWidget(pContent);  // 20260326 ZJH 将内容放入滚动区域
    return pScroll;  // 20260326 ZJH 返回可滚动的面板
}

// 20260322 ZJH 创建中央面板
QWidget* TrainingPage::createCenterPanel()
{
    QWidget* pPanel = new QWidget();
    pPanel->setStyleSheet(QStringLiteral("QWidget { background-color: #22262e; }"));

    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(8, 8, 8, 8);
    pLayout->setSpacing(8);

    // 20260322 ZJH 1. 损失曲线图表（占据大部分空间）
    m_pLossChart = new TrainingLossChart(pPanel);
    pLayout->addWidget(m_pLossChart, 1);  // 20260322 ZJH stretch=1 填充剩余空间

    // 20260322 ZJH 2. 进度区域
    {
        QWidget* pProgressArea = new QWidget(pPanel);
        QVBoxLayout* pProgressLayout = new QVBoxLayout(pProgressArea);
        pProgressLayout->setContentsMargins(0, 0, 0, 0);
        pProgressLayout->setSpacing(4);

        // 20260322 ZJH 进度条
        m_pProgressBar = new QProgressBar(pProgressArea);
        m_pProgressBar->setRange(0, 100);
        m_pProgressBar->setValue(0);
        m_pProgressBar->setTextVisible(true);  // 20260322 ZJH 显示百分比文字
        m_pProgressBar->setStyleSheet(QStringLiteral(
            "QProgressBar {"
            "  background-color: #1a1d24;"
            "  border: 1px solid #333842;"
            "  border-radius: 4px;"
            "  height: 20px;"
            "  text-align: center;"
            "  color: #e2e8f0;"
            "  font-size: 11px;"
            "}"
            "QProgressBar::chunk {"
            "  background-color: #2563eb;"
            "  border-radius: 3px;"
            "}"
        ));
        pProgressLayout->addWidget(m_pProgressBar);

        // 20260322 ZJH 状态文字标签
        m_pLblStatus = new QLabel(QStringLiteral("就绪 — 配置参数后点击\"开始训练\""));
        m_pLblStatus->setStyleSheet(QStringLiteral(
            "QLabel { color: #94a3b8; font-size: 12px; background: transparent; }"));
        m_pLblStatus->setAlignment(Qt::AlignCenter);
        pProgressLayout->addWidget(m_pLblStatus);

        pLayout->addWidget(pProgressArea);
    }

    // 20260322 ZJH 3. 控制按钮区域（水平排列）
    {
        QWidget* pButtonArea = new QWidget(pPanel);
        QHBoxLayout* pBtnLayout = new QHBoxLayout(pButtonArea);
        pBtnLayout->setContentsMargins(0, 0, 0, 0);
        pBtnLayout->setSpacing(8);

        // 20260322 ZJH 添加弹性空间使按钮居中
        pBtnLayout->addStretch(1);

        // 20260322 ZJH 开始训练按钮（蓝色大按钮）
        m_pBtnStart = new QPushButton(QStringLiteral("开始训练"));
        m_pBtnStart->setMinimumSize(120, 36);  // 20260322 ZJH 最小尺寸
        m_pBtnStart->setStyleSheet(QStringLiteral(
            "QPushButton {"
            "  background-color: #2563eb;"
            "  color: white;"
            "  border: none;"
            "  border-radius: 6px;"
            "  font-size: 14px;"
            "  font-weight: bold;"
            "  padding: 8px 24px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #3b82f6;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #333842;"
            "  color: #64748b;"
            "}"
        ));
        pBtnLayout->addWidget(m_pBtnStart);

        // 20260322 ZJH 暂停按钮
        m_pBtnPause = new QPushButton(QStringLiteral("暂停"));
        m_pBtnPause->setMinimumSize(80, 36);
        m_pBtnPause->setStyleSheet(QStringLiteral(
            "QPushButton {"
            "  background-color: #d97706;"
            "  color: white;"
            "  border: none;"
            "  border-radius: 6px;"
            "  font-size: 13px;"
            "  padding: 8px 16px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #f59e0b;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #333842;"
            "  color: #64748b;"
            "}"
        ));
        pBtnLayout->addWidget(m_pBtnPause);

        // 20260322 ZJH 停止按钮
        m_pBtnStop = new QPushButton(QStringLiteral("停止"));
        m_pBtnStop->setMinimumSize(80, 36);
        m_pBtnStop->setStyleSheet(QStringLiteral(
            "QPushButton {"
            "  background-color: #dc2626;"
            "  color: white;"
            "  border: none;"
            "  border-radius: 6px;"
            "  font-size: 13px;"
            "  padding: 8px 16px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #ef4444;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #333842;"
            "  color: #64748b;"
            "}"
        ));
        pBtnLayout->addWidget(m_pBtnStop);

        // 20260322 ZJH 继续训练按钮
        m_pBtnResume = new QPushButton(QStringLiteral("继续训练"));
        m_pBtnResume->setMinimumSize(100, 36);
        m_pBtnResume->setStyleSheet(QStringLiteral(
            "QPushButton {"
            "  background-color: #059669;"
            "  color: white;"
            "  border: none;"
            "  border-radius: 6px;"
            "  font-size: 13px;"
            "  padding: 8px 16px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #10b981;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #333842;"
            "  color: #64748b;"
            "}"
        ));
        pBtnLayout->addWidget(m_pBtnResume);

        // 20260322 ZJH 右侧弹性空间
        pBtnLayout->addStretch(1);

        pLayout->addWidget(pButtonArea);
    }

    return pPanel;
}

// 20260322 ZJH 创建右面板
QWidget* TrainingPage::createRightPanel()
{
    // 20260325 ZJH 直接创建内容面板（无滚动区域，等比缩放）
    QWidget* pPanel = new QWidget();
    pPanel->setStyleSheet(s_strControlStyle + QStringLiteral(
        "QWidget { background-color: #1e2230; }"));

    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(12, 8, 12, 8);
    pLayout->setSpacing(4);

    // ===== 数据概览分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("数据概览"), pPanel);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);
        pForm->setSpacing(4);

        m_pLblTrainCount = new QLabel(QStringLiteral("0"));
        m_pLblTrainCount->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("训练集"), m_pLblTrainCount);

        m_pLblValCount = new QLabel(QStringLiteral("0"));
        m_pLblValCount->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("验证集"), m_pLblValCount);

        m_pLblTestCount = new QLabel(QStringLiteral("0"));
        m_pLblTestCount->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("测试集"), m_pLblTestCount);

        pLayout->addWidget(pGroup);
    }

    // ===== 前置检查分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("前置检查"), pPanel);
        QVBoxLayout* pCheckLayout = new QVBoxLayout(pGroup);
        pCheckLayout->setSpacing(4);

        // 20260322 ZJH ✗ 表示未通过，✓ 表示通过
        m_pLblCheckImages = new QLabel(QStringLiteral("\u2717 图像已导入"));
        m_pLblCheckImages->setStyleSheet(QStringLiteral("color: #ef4444;"));  // 20260322 ZJH 红色
        pCheckLayout->addWidget(m_pLblCheckImages);

        m_pLblCheckLabels = new QLabel(QStringLiteral("\u2717 标签已分配"));
        m_pLblCheckLabels->setStyleSheet(QStringLiteral("color: #ef4444;"));
        pCheckLayout->addWidget(m_pLblCheckLabels);

        m_pLblCheckSplit = new QLabel(QStringLiteral("\u2717 数据已拆分"));
        m_pLblCheckSplit->setStyleSheet(QStringLiteral("color: #ef4444;"));
        pCheckLayout->addWidget(m_pLblCheckSplit);

        pLayout->addWidget(pGroup);
    }

    // ===== 训练状态分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("训练状态"), pPanel);
        QFormLayout* pForm = new QFormLayout(pGroup);
        pForm->setLabelAlignment(Qt::AlignRight);
        pForm->setSpacing(4);

        m_pLblCurrentEpoch = new QLabel(QStringLiteral("—"));
        m_pLblCurrentEpoch->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("当前 Epoch"), m_pLblCurrentEpoch);

        m_pLblBestLoss = new QLabel(QStringLiteral("—"));
        m_pLblBestLoss->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("最佳损失"), m_pLblBestLoss);

        m_pLblTimeRemaining = new QLabel(QStringLiteral("—"));
        m_pLblTimeRemaining->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("预计剩余"), m_pLblTimeRemaining);

        m_pLblEarlyStopCount = new QLabel(QStringLiteral("—"));
        m_pLblEarlyStopCount->setStyleSheet(QStringLiteral("color: #e2e8f0; font-weight: bold;"));
        pForm->addRow(QStringLiteral("早停计数"), m_pLblEarlyStopCount);

        pLayout->addWidget(pGroup);
    }

    // ===== 训练日志分组 =====
    {
        QGroupBox* pGroup = new QGroupBox(QStringLiteral("训练日志"), pPanel);
        QVBoxLayout* pLogLayout = new QVBoxLayout(pGroup);

        m_pTxtLog = new QTextEdit(pGroup);
        m_pTxtLog->setReadOnly(true);  // 20260322 ZJH 只读
        m_pTxtLog->setStyleSheet(QStringLiteral(
            "QTextEdit {"
            "  background-color: #13151a;"
            "  color: #94a3b8;"
            "  border: 1px solid #2a2d35;"
            "  border-radius: 4px;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 11px;"
            "  padding: 4px;"
            "}"
        ));
        pLogLayout->addWidget(m_pTxtLog);

        pLayout->addWidget(pGroup, 1);  // 20260322 ZJH stretch=1 让日志框占据剩余空间
    }

    return pPanel;  // 20260325 ZJH 直接返回内容面板（等比缩放，无滚动）
}

// ===== 训练控制槽函数 =====

// 20260322 ZJH 开始训练
void TrainingPage::onStartTraining()
{
    // 20260322 ZJH 防止重复启动
    if (m_pSession && m_pSession->isRunning()) {
        return;
    }

    // 20260324 ZJH ===== 训练前置校验（检查数据集状态，阻止无效训练启动） =====
    {
        // 20260324 ZJH 检查 0: 项目和数据集是否存在
        if (!m_pProject || !m_pProject->dataset()) {
            QMessageBox::warning(this, QStringLiteral("训练校验"),
                                 QStringLiteral("请先创建或打开一个项目。"));  // 20260324 ZJH 无项目提示
            return;  // 20260324 ZJH 无法训练，直接返回
        }
        ImageDataset* pDataset = m_pProject->dataset();  // 20260324 ZJH 获取数据集指针用于后续检查

        // 20260324 ZJH 检查 1: 数据集中是否有图像
        if (pDataset->imageCount() == 0) {
            QMessageBox::warning(this, QStringLiteral("训练校验"),
                                 QStringLiteral("数据集为空，请先导入图像。"));  // 20260324 ZJH 空数据集提示
            return;  // 20260324 ZJH 无图像，无法训练
        }

        // 20260324 ZJH 检查 2: 是否已定义标签（分类/检测/分割任务需要标签）
        if (pDataset->labels().isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("训练校验"),
                                 QStringLiteral("未定义标签，请先在图库页面添加标签。"));  // 20260324 ZJH 无标签提示
            return;  // 20260324 ZJH 无标签定义，无法训练
        }

        // 20260324 ZJH 检查 3: 是否有已标注的图像
        if (pDataset->labeledCount() == 0) {
            QMessageBox::warning(this, QStringLiteral("训练校验"),
                                 QStringLiteral("没有已标注的图像，请先标注图像。"));  // 20260324 ZJH 无标注提示
            return;  // 20260324 ZJH 无已标注图像，无法训练
        }

        // 20260324 ZJH 检查 4: 训练集是否为空
        if (pDataset->countBySplit(om::SplitType::Train) == 0) {
            QMessageBox::warning(this, QStringLiteral("训练校验"),
                                 QStringLiteral("训练集为空，请先在拆分页面执行数据集拆分。"));  // 20260324 ZJH 无训练集提示
            return;  // 20260324 ZJH 训练集为空，无法训练
        }

        // 20260324 ZJH 检查 5: 验证集是否为空（早停机制需要验证集）
        if (pDataset->countBySplit(om::SplitType::Validation) == 0) {
            QMessageBox::warning(this, QStringLiteral("训练校验"),
                                 QStringLiteral("验证集为空，请确保拆分时包含验证集。"));  // 20260324 ZJH 无验证集提示
            return;  // 20260324 ZJH 验证集为空，早停无法工作
        }
    }

    // 20260322 ZJH 清空之前的图表和日志
    m_pLossChart->clear();
    m_pTxtLog->clear();

    // 20260322 ZJH 重置状态跟踪变量
    m_dBestLoss = 1e9;
    m_nEarlyStopCount = 0;
    m_nCurrentEpoch = 0;

    // 20260402 ZJH 重置训练诊断系统状态（每次新训练开始清空历史）
    m_vecTrainLossHistory.clear();   // 20260402 ZJH 清空训练 loss 历史
    m_vecValLossHistory.clear();     // 20260402 ZJH 清空验证 loss 历史
    m_vecValMetricHistory.clear();   // 20260402 ZJH 清空验证指标历史
    m_dAvgEpochSec = 0.0;           // 20260402 ZJH 重置平均 epoch 耗时
    m_nDiagOverfitCount = 0;        // 20260402 ZJH 重置过拟合计数
    m_nDiagOscillCount = 0;         // 20260402 ZJH 重置震荡计数
    m_nDiagNoImproveCount = 0;      // 20260402 ZJH 重置无改善计数
    m_tpEpochStart = std::chrono::steady_clock::now();  // 20260402 ZJH 记录训练起始时刻

    // 20260322 ZJH 从 UI 收集配置
    TrainingConfig config = gatherConfig();
    m_nTotalEpochs = config.nEpochs;  // 20260322 ZJH 记录总轮次

    // 20260404 ZJH 检查 6: CUDA 可用性 — 有 GPU 时强制使用，无 GPU 时回退 CPU
#ifdef OM_HAS_CUDA
    // 20260404 ZJH 有 CUDA 支持时，无论用户选什么都强制用 GPU（用户明确要求"用GPU训练"）
    config.eDevice = om::DeviceType::CUDA;
    if (m_pCboDevice) {
        int nCudaIdx = m_pCboDevice->findData(static_cast<int>(om::DeviceType::CUDA));
        if (nCudaIdx >= 0) m_pCboDevice->setCurrentIndex(nCudaIdx);  // 20260404 ZJH 同步 UI
    }
#else
    // 20260324 ZJH 无 CUDA 支持时，强制 CPU
    if (config.eDevice == om::DeviceType::CUDA) {
        QMessageBox::StandardButton eBtn = QMessageBox::question(this,
            QStringLiteral("CUDA 不可用"),
            QStringLiteral("当前构建未包含 CUDA 支持。\n是否使用 CPU (SIMD+OpenMP) 继续训练？"),
            QMessageBox::Yes | QMessageBox::No);
        if (eBtn == QMessageBox::No) return;
        config.eDevice = om::DeviceType::CPU;
    }
#endif

    // 20260324 ZJH 持久化训练配置到项目（下次打开项目时恢复参数）
    if (m_pProject) {
        m_pProject->setTrainingConfig(config);  // 20260324 ZJH 保存训练配置到 Project
        m_pProject->clearTrainingHistory();      // 20260324 ZJH 清空上一轮训练历史，准备记录新一轮
    }

    // 20260322 ZJH 设置训练会话参数
    m_pSession->setConfig(config);
    m_pSession->setProject(m_pProject);

    // 20260322 ZJH 禁用左面板配置控件
    setControlsEnabled(false);

    // 20260322 ZJH 更新按钮状态
    m_pBtnStart->setEnabled(false);
    m_pBtnPause->setEnabled(true);
    m_pBtnStop->setEnabled(true);
    m_pBtnResume->setEnabled(false);

    // 20260322 ZJH 启动计时器
    m_elapsedTimer.start();

    // 20260322 ZJH 更新状态标签
    m_pLblStatus->setText(QStringLiteral("训练中..."));
    m_pLblCurrentEpoch->setText(QStringLiteral("0/%1").arg(m_nTotalEpochs));
    m_pLblBestLoss->setText(QStringLiteral("—"));
    m_pLblTimeRemaining->setText(QStringLiteral("计算中..."));
    m_pLblEarlyStopCount->setText(QStringLiteral("0/%1").arg(config.nPatience));

    // 20260322 ZJH 在工作线程中启动训练（通过 QMetaObject::invokeMethod 跨线程调用）
    QMetaObject::invokeMethod(m_pSession, "startTraining", Qt::QueuedConnection);
}

// 20260322 ZJH 暂停训练
void TrainingPage::onPauseTraining()
{
    if (m_pSession) {
        m_pSession->pauseTraining();  // 20260322 ZJH 设置暂停标记

        // 20260322 ZJH 更新按钮状态
        m_pBtnPause->setEnabled(false);
        m_pBtnResume->setEnabled(true);

        // 20260322 ZJH 更新状态标签
        m_pLblStatus->setText(QStringLiteral("已暂停"));
    }
}

// 20260322 ZJH 停止训练
void TrainingPage::onStopTraining()
{
    if (m_pSession) {
        m_pSession->stopTraining();  // 20260322 ZJH 设置停止标记
    }
}

// 20260322 ZJH 继续训练（从暂停恢复）
void TrainingPage::onResumeTraining()
{
    if (m_pSession) {
        m_pSession->resumeTraining();  // 20260322 ZJH 清除暂停标记

        // 20260322 ZJH 更新按钮状态
        m_pBtnPause->setEnabled(true);
        m_pBtnResume->setEnabled(false);

        // 20260322 ZJH 更新状态标签
        m_pLblStatus->setText(QStringLiteral("训练中..."));
    }
}

// 20260322 ZJH Epoch 完成处理
void TrainingPage::onEpochCompleted(int nEpoch, int nTotal, double dTrainLoss, double dValLoss, double dMetric)
{
    // 20260322 ZJH 记录当前 epoch
    m_nCurrentEpoch = nEpoch;

    // 20260402 ZJH 计算当前 epoch 耗时（使用 steady_clock 精确计时）
    auto tpNow = std::chrono::steady_clock::now();  // 20260402 ZJH 当前时刻
    double dEpochSec = std::chrono::duration<double>(tpNow - m_tpEpochStart).count();  // 20260402 ZJH 本 epoch 耗时秒
    m_tpEpochStart = tpNow;  // 20260402 ZJH 重置为下一 epoch 的起点

    // 20260402 ZJH 更新平均 epoch 耗时（指数滑动平均，alpha=0.3 给最近值更高权重）
    if (nEpoch <= 1) {
        m_dAvgEpochSec = dEpochSec;  // 20260402 ZJH 第 1 epoch 直接赋值
    } else {
        m_dAvgEpochSec = 0.3 * dEpochSec + 0.7 * m_dAvgEpochSec;  // 20260402 ZJH 滑动平均平滑波动
    }

    // 20260322 ZJH 1. 添加数据点到曲线图
    m_pLossChart->addTrainPoint(nEpoch, dTrainLoss);
    m_pLossChart->addValPoint(nEpoch, dValLoss);

    // 20260322 ZJH 2. 更新状态文字
    m_pLblStatus->setText(QStringLiteral("Epoch %1/%2 \u2014 Train: %3 | Val: %4 | Metric: %5")
        .arg(nEpoch).arg(nTotal)
        .arg(dTrainLoss, 0, 'f', 4)
        .arg(dValLoss, 0, 'f', 4)
        .arg(dMetric, 0, 'f', 4));

    // 20260322 ZJH 3. 更新训练状态面板
    m_pLblCurrentEpoch->setText(QStringLiteral("%1/%2").arg(nEpoch).arg(nTotal));

    // 20260402 ZJH 更新进度条百分比 = epoch / max_epochs * 100
    int nPercent = static_cast<int>(static_cast<double>(nEpoch) / nTotal * 100.0);  // 20260402 ZJH 百分比
    m_pProgressBar->setValue(nPercent);  // 20260402 ZJH 同步更新进度条

    // 20260322 ZJH 跟踪最佳验证损失和早停计数
    if (dValLoss < m_dBestLoss) {
        m_dBestLoss = dValLoss;    // 20260322 ZJH 更新最佳
        m_nEarlyStopCount = 0;      // 20260322 ZJH 重置早停计数
    } else {
        m_nEarlyStopCount++;  // 20260322 ZJH 早停计数 +1
    }

    m_pLblBestLoss->setText(QString::number(m_dBestLoss, 'f', 4));
    m_pLblEarlyStopCount->setText(QStringLiteral("%1/%2").arg(m_nEarlyStopCount).arg(m_pSpnPatience->value()));

    // 20260324 ZJH 5. 持久化 Epoch 记录到项目训练历史
    if (m_pProject) {
        EpochRecord record;  // 20260324 ZJH 创建 Epoch 记录
        record.nEpoch       = nEpoch;       // 20260324 ZJH 轮次编号
        record.dTrainLoss   = dTrainLoss;   // 20260324 ZJH 训练损失
        record.dValLoss     = dValLoss;     // 20260324 ZJH 验证损失
        record.dValAccuracy = dMetric;      // 20260324 ZJH 验证指标（准确率/mAP等）
        m_pProject->addEpochRecord(record);  // 20260324 ZJH 追加到训练历史
    }

    // 20260402 ZJH 6. 估算剩余时间（使用滑动平均更平滑）
    if (nEpoch > 0) {
        int nRemainingEpochs = nTotal - nEpoch;  // 20260402 ZJH 剩余 epoch 数
        QString strRemaining = formatRemainingTime(nRemainingEpochs);  // 20260402 ZJH 格式化剩余时间
        m_pLblTimeRemaining->setText(strRemaining);  // 20260402 ZJH 更新 UI 显示
    }

    // 20260402 ZJH 7. 运行训练质量诊断（记录历史并检测异常模式）
    runTrainingDiagnostics(nEpoch, nTotal,
                           static_cast<float>(dTrainLoss),
                           static_cast<float>(dValLoss),
                           static_cast<float>(dMetric));
}

// 20260322 ZJH 训练完成处理
void TrainingPage::onTrainingFinished(bool bSuccess, const QString& strMessage)
{
    // 20260322 ZJH 重新启用左面板
    setControlsEnabled(true);

    // 20260322 ZJH 更新按钮状态
    m_pBtnStart->setEnabled(true);
    m_pBtnPause->setEnabled(false);
    m_pBtnStop->setEnabled(false);
    m_pBtnResume->setEnabled(false);

    // 20260322 ZJH 更新状态标签
    if (bSuccess) {
        m_pLblStatus->setText(QStringLiteral("训练完成 \u2014 %1").arg(strMessage));
        m_pLblTimeRemaining->setText(QStringLiteral("已完成"));
    } else {
        m_pLblStatus->setText(QStringLiteral("训练中断 \u2014 %1").arg(strMessage));
        m_pLblTimeRemaining->setText(QStringLiteral("已中断"));
    }

    // 20260324 ZJH 持久化最佳模型路径和自动保存项目
    if (m_pProject) {
        // 20260324 ZJH 从 TrainingSession 获取真实保存的模型路径（而非硬编码路径）
        if (bSuccess && m_pSession) {
            QString strModelPath = m_pSession->modelSavePath();  // 20260324 ZJH 获取引擎实际保存的模型路径
            // 20260324 ZJH 验证模型文件确实存在于磁盘
            if (!strModelPath.isEmpty() && QFileInfo::exists(strModelPath)) {
                m_pProject->setBestModelPath(strModelPath);  // 20260324 ZJH 保存验证过的路径到项目
                // 20260324 ZJH 训练成功且模型文件存在，更新项目状态为 ModelTrained
                m_pProject->setState(om::ProjectState::ModelTrained);
            } else if (!strModelPath.isEmpty()) {
                // 20260324 ZJH 路径非空但文件不存在，记录警告
                qWarning("[TrainingPage] 模型路径不存在: %s", qPrintable(strModelPath));
            }
        }

        // 20260324 ZJH 训练结束后自动保存项目（包含训练配置、历史记录、模型路径）
        QString strProjFile = m_pProject->path() + QStringLiteral("/") + m_pProject->name() + QStringLiteral(".dfproj");  // 20260324 ZJH 项目文件路径
        bool bSaveOk = ProjectSerializer::save(m_pProject, strProjFile);  // 20260324 ZJH 序列化保存
        if (bSaveOk) {
            m_pProject->setDirty(false);  // 20260324 ZJH 保存成功，重置脏标志
            qDebug() << "[TrainingPage] onTrainingFinished: 项目自动保存成功" << strProjFile;  // 20260324 ZJH 保存成功日志
            Application::instance()->notifyProjectSaved();  // 20260324 ZJH 通知全局保存事件
        } else {
            qDebug() << "[TrainingPage] onTrainingFinished: 项目自动保存失败" << strProjFile;  // 20260324 ZJH 保存失败日志
        }
    }
}

// 20260322 ZJH 训练日志接收
void TrainingPage::onTrainingLog(const QString& strMessage)
{
    // 20260322 ZJH 追加日志到文本框
    m_pTxtLog->append(strMessage);

    // 20260322 ZJH 滚动到底部
    QTextCursor cursor = m_pTxtLog->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_pTxtLog->setTextCursor(cursor);
    m_pTxtLog->ensureCursorVisible();
}

// 20260322 ZJH 进度百分比更新
void TrainingPage::onProgressChanged(int nPercent)
{
    m_pProgressBar->setValue(nPercent);  // 20260322 ZJH 更新进度条
}

// 20260322 ZJH 刷新模型架构下拉框
// 20260330 ZJH 重构：根据任务类型 + 模型能力等级过滤架构，同时刷新分辨率预设和异常检测模式
void TrainingPage::refreshArchitectureCombo()
{
    // 20260322 ZJH 保存当前选择的架构索引
    int nPrevIndex = m_pCboArchitecture->currentIndex();

    // 20260322 ZJH 清空并重新填充
    m_pCboArchitecture->clear();

    // 20260322 ZJH 获取当前任务类型
    om::TaskType eTask = om::TaskType::Classification;  // 20260322 ZJH 默认分类
    if (m_pProject) {
        eTask = m_pProject->taskType();  // 20260322 ZJH 从项目获取任务类型
    }

    // 20260330 ZJH 获取当前选择的模型能力等级
    om::ModelCapability eCapability = om::ModelCapability::Normal;
    if (m_pCboCapability) {
        eCapability = static_cast<om::ModelCapability>(m_pCboCapability->currentData().toInt());
    }

    // 20260330 ZJH 根据任务类型 + 能力等级获取推荐架构列表
    QVector<om::ModelArchitecture> vecArchs = om::architecturesForCapability(eTask, eCapability);

    // 20260322 ZJH 填充下拉框
    for (const auto& eArch : vecArchs) {
        m_pCboArchitecture->addItem(
            om::modelArchitectureToString(eArch),  // 20260322 ZJH 显示名称
            static_cast<int>(eArch));               // 20260322 ZJH 用户数据存储枚举值
    }

    // 20260322 ZJH 尝试恢复之前的选择，否则选择默认架构
    if (nPrevIndex >= 0 && nPrevIndex < m_pCboArchitecture->count()) {
        m_pCboArchitecture->setCurrentIndex(nPrevIndex);
    } else if (m_pCboArchitecture->count() > 0) {
        m_pCboArchitecture->setCurrentIndex(0);  // 20260322 ZJH 选择第一个（默认）
    }

    // 20260330 ZJH ===== 刷新输入分辨率预设下拉框 =====
    if (m_pCboResolution) {
        int nPrevResIdx = m_pCboResolution->currentIndex();
        m_pCboResolution->blockSignals(true);  // 20260330 ZJH 防止刷新过程中触发联动信号
        m_pCboResolution->clear();
        QVector<om::InputResolutionPreset> vecPresets = om::resolutionPresetsForTask(eTask);
        for (const auto& ePreset : vecPresets) {
            m_pCboResolution->addItem(
                om::resolutionPresetToString(ePreset),
                static_cast<int>(ePreset));
        }
        // 20260330 ZJH 恢复之前的选择
        if (nPrevResIdx >= 0 && nPrevResIdx < m_pCboResolution->count()) {
            m_pCboResolution->setCurrentIndex(nPrevResIdx);
        } else if (m_pCboResolution->count() > 0) {
            m_pCboResolution->setCurrentIndex(0);
        }
        m_pCboResolution->blockSignals(false);
        // 20260330 ZJH 手动触发一次联动更新 inputSize
        auto ePreset = static_cast<om::InputResolutionPreset>(m_pCboResolution->currentData().toInt());
        if (ePreset != om::InputResolutionPreset::Custom) {
            m_pSpnInputSize->setValue(om::resolutionPresetToPixels(ePreset));
            m_pSpnInputSize->setVisible(false);
        }
    }

    // 20260330 ZJH ===== 异常检测训练模式显隐 =====
    bool bIsAnomaly = (eTask == om::TaskType::AnomalyDetection);
    if (m_pCboAnomalyMode) {
        m_pCboAnomalyMode->setVisible(bIsAnomaly);
        m_pLblAnomalyMode->setVisible(bIsAnomaly);
    }
}

// 20260322 ZJH 从 UI 控件收集训练配置
TrainingConfig TrainingPage::gatherConfig() const
{
    TrainingConfig config;

    // 20260322 ZJH 模型配置
    config.eFramework    = static_cast<om::TrainingFramework>(m_pCboFramework->currentData().toInt());
    // 20260330 ZJH 模型能力等级
    config.eModelCapability = static_cast<om::ModelCapability>(m_pCboCapability->currentData().toInt());
    config.eArchitecture = static_cast<om::ModelArchitecture>(m_pCboArchitecture->currentData().toInt());
    config.eDevice       = static_cast<om::DeviceType>(m_pCboDevice->currentData().toInt());
    // 20260330 ZJH 异常检测训练模式
    if (m_pCboAnomalyMode && m_pCboAnomalyMode->isVisible()) {
        config.eAnomalyMode = static_cast<om::AnomalyTrainingMode>(m_pCboAnomalyMode->currentData().toInt());
    }
    config.eOptimizer    = static_cast<om::OptimizerType>(m_pCboOptimizer->currentData().toInt());
    config.eScheduler    = static_cast<om::SchedulerType>(m_pCboScheduler->currentData().toInt());

    // 20260322 ZJH 超参数
    config.dLearningRate = m_pSpnLearningRate->value();
    config.nBatchSize    = m_pSpnBatchSize->value();
    config.nEpochs       = m_pSpnEpochs->value();
    // 20260330 ZJH 分辨率预设
    config.eResolutionPreset = static_cast<om::InputResolutionPreset>(m_pCboResolution->currentData().toInt());
    config.nInputSize    = m_pSpnInputSize->value();
    config.nPatience     = m_pSpnPatience->value();

    // 20260330 ZJH 预训练模型与标识
    config.strPretrainedModelPath = m_pEdtPretrained->text();
    config.strModelTag = m_pEdtModelTag->text();

    // 20260330 ZJH 数据增强预设
    config.eAugPreset = static_cast<om::AugmentationPreset>(m_pCboAugPreset->currentData().toInt());

    // 20260322 ZJH 数据增强（基础）
    config.bAugmentation  = m_pChkAugmentation->isChecked();  // 20260322 ZJH 总开关
    config.dAugBrightness = m_pSpnBrightness->value();        // 20260322 ZJH 亮度
    config.dAugFlipProb   = m_pSpnFlipProb->value();          // 20260322 ZJH 水平翻转概率
    config.dAugRotation   = m_pSpnRotation->value();          // 20260322 ZJH 旋转角度

    // 20260324 ZJH 数据增强（扩展 — 几何变换）
    config.dAugVerticalFlipProb = m_pSpnVerticalFlipProb->value();  // 20260324 ZJH 垂直翻转概率
    config.bAugAffine           = m_pChkAffine->isChecked();        // 20260324 ZJH 仿射变换开关
    config.dAugShearDeg         = m_pSpnShearDeg->value();          // 20260324 ZJH 仿射剪切角度
    config.dAugTranslate        = m_pSpnTranslate->value();         // 20260324 ZJH 仿射平移比例
    config.bAugRandomCrop       = m_pChkRandomCrop->isChecked();    // 20260324 ZJH 随机缩放裁剪开关
    config.dAugCropScale        = m_pSpnCropScale->value();         // 20260324 ZJH 最小缩放比例

    // 20260324 ZJH 数据增强（扩展 — 颜色变换）
    config.bAugColorJitter = m_pChkColorJitter->isChecked();  // 20260324 ZJH 颜色抖动开关
    config.dAugSaturation  = m_pSpnSaturation->value();       // 20260324 ZJH 饱和度抖动范围
    config.dAugHue         = m_pSpnHue->value();              // 20260324 ZJH 色调抖动范围

    // 20260324 ZJH 数据增强（扩展 — 噪声/遮挡）
    config.bAugGaussianNoise   = m_pChkGaussianNoise->isChecked();   // 20260324 ZJH 高斯噪声开关
    config.dAugGaussianNoiseStd = m_pSpnNoiseStd->value();           // 20260324 ZJH 噪声标准差
    config.bAugGaussianBlur    = m_pChkGaussianBlur->isChecked();    // 20260324 ZJH 高斯模糊开关
    config.dAugBlurSigma       = m_pSpnBlurSigma->value();           // 20260324 ZJH 模糊 sigma
    config.bAugRandomErasing   = m_pChkRandomErasing->isChecked();   // 20260324 ZJH 随机擦除开关
    config.dAugErasingProb     = m_pSpnErasingProb->value();         // 20260324 ZJH 擦除概率
    config.dAugErasingRatio    = m_pSpnErasingRatio->value();        // 20260324 ZJH 擦除面积比例

    // 20260324 ZJH 数据增强（扩展 — 高级混合）
    config.bAugMixup      = m_pChkMixup->isChecked();       // 20260324 ZJH Mixup 开关
    config.dAugMixupAlpha = m_pSpnMixupAlpha->value();      // 20260324 ZJH Mixup alpha
    config.bAugCutMix      = m_pChkCutMix->isChecked();     // 20260324 ZJH CutMix 开关
    config.dAugCutMixAlpha = m_pSpnCutMixAlpha->value();    // 20260324 ZJH CutMix alpha

    // 20260330 ZJH 预训练权重（新增控件）
    if (m_pChkUsePretrained && m_pChkUsePretrained->isChecked() && m_pEdtPretrainedPath) {
        config.strPretrainedModelPath = m_pEdtPretrainedPath->text();  // 20260330 ZJH 覆盖预训练路径
    }

    // 20260330 ZJH HSV 色彩空间抖动
    config.bAugHsvAugment = m_pChkHsvAugment ? m_pChkHsvAugment->isChecked() : false;
    config.dAugHueShift   = m_pSpnHueShift   ? m_pSpnHueShift->value()       : 5.0;
    config.dAugSatShift   = m_pSpnSatShift   ? m_pSpnSatShift->value()       : 5.0;
    config.dAugValShift   = m_pSpnValShift   ? m_pSpnValShift->value()       : 5.0;

    // 20260330 ZJH 少样本学习
    config.bFewShot       = m_pChkFewShot      ? m_pChkFewShot->isChecked()      : false;
    config.nShotsPerClass = m_pSpnShotsPerClass ? m_pSpnShotsPerClass->value()   : 5;

    // 20260330 ZJH 模型优化（训练后剪枝）
    config.bPruning      = m_pChkPruning     ? m_pChkPruning->isChecked()          : false;
    config.nPruneMethod  = m_pCboPruneMethod ? m_pCboPruneMethod->currentData().toInt() : 0;
    config.dPruneRatio   = m_pSpnPruneRatio  ? m_pSpnPruneRatio->value()           : 0.3;

    // 20260402 ZJH [OPT-2.5] 增量训练 (Continual Learning)
    config.bContinualLearning = m_pChkContinualLearning ? m_pChkContinualLearning->isChecked() : false;
    config.dEwcLambda = m_pSpnEwcLambda ? m_pSpnEwcLambda->value() : 1000.0;

    // 20260322 ZJH ONNX 导出
    config.bExportOnnx = m_pChkExportOnnx->isChecked();

    return config;
}

// 20260325 ZJH 将 TrainingConfig 回填到 UI 控件（gatherConfig 的逆操作）
// 项目加载时调用，恢复用户上次保存的训练参数
void TrainingPage::restoreConfigToUI(const TrainingConfig& config)
{
    // 20260325 ZJH 模型配置 — 通过 findData 匹配 enum 值选中对应项
    int nIdx = m_pCboFramework->findData(static_cast<int>(config.eFramework));
    if (nIdx >= 0) m_pCboFramework->setCurrentIndex(nIdx);  // 20260325 ZJH 训练框架

    // 20260330 ZJH 模型能力等级
    nIdx = m_pCboCapability->findData(static_cast<int>(config.eModelCapability));
    if (nIdx >= 0) m_pCboCapability->setCurrentIndex(nIdx);

    // 20260330 ZJH 刷新架构列表（基于能力等级），然后恢复架构选择
    refreshArchitectureCombo();
    nIdx = m_pCboArchitecture->findData(static_cast<int>(config.eArchitecture));
    if (nIdx >= 0) m_pCboArchitecture->setCurrentIndex(nIdx);  // 20260325 ZJH 模型架构

    nIdx = m_pCboDevice->findData(static_cast<int>(config.eDevice));
    if (nIdx >= 0) m_pCboDevice->setCurrentIndex(nIdx);  // 20260325 ZJH 设备类型

    // 20260330 ZJH 异常检测训练模式
    if (m_pCboAnomalyMode) {
        nIdx = m_pCboAnomalyMode->findData(static_cast<int>(config.eAnomalyMode));
        if (nIdx >= 0) m_pCboAnomalyMode->setCurrentIndex(nIdx);
    }

    nIdx = m_pCboOptimizer->findData(static_cast<int>(config.eOptimizer));
    if (nIdx >= 0) m_pCboOptimizer->setCurrentIndex(nIdx);  // 20260325 ZJH 优化器

    nIdx = m_pCboScheduler->findData(static_cast<int>(config.eScheduler));
    if (nIdx >= 0) m_pCboScheduler->setCurrentIndex(nIdx);  // 20260325 ZJH 调度器

    // 20260325 ZJH 超参数
    m_pSpnLearningRate->setValue(config.dLearningRate);  // 20260325 ZJH 学习率
    m_pSpnBatchSize->setValue(config.nBatchSize);        // 20260325 ZJH 批量大小
    m_pSpnEpochs->setValue(config.nEpochs);              // 20260325 ZJH 训练轮数

    // 20260330 ZJH 分辨率预设
    nIdx = m_pCboResolution->findData(static_cast<int>(config.eResolutionPreset));
    if (nIdx >= 0) m_pCboResolution->setCurrentIndex(nIdx);
    m_pSpnInputSize->setValue(config.nInputSize);        // 20260325 ZJH 输入尺寸

    m_pSpnPatience->setValue(config.nPatience);          // 20260325 ZJH 早停耐心值

    // 20260330 ZJH 预训练模型与标识
    m_pEdtPretrained->setText(config.strPretrainedModelPath);
    m_pEdtModelTag->setText(config.strModelTag);

    // 20260330 ZJH 数据增强预设
    nIdx = m_pCboAugPreset->findData(static_cast<int>(config.eAugPreset));
    if (nIdx >= 0) m_pCboAugPreset->setCurrentIndex(nIdx);

    // 20260325 ZJH 数据增强（基础）
    m_pChkAugmentation->setChecked(config.bAugmentation);  // 20260325 ZJH 总开关
    m_pSpnBrightness->setValue(config.dAugBrightness);     // 20260325 ZJH 亮度
    m_pSpnFlipProb->setValue(config.dAugFlipProb);         // 20260325 ZJH 水平翻转概率
    m_pSpnRotation->setValue(config.dAugRotation);         // 20260325 ZJH 旋转角度

    // 20260325 ZJH 数据增强（几何变换）
    m_pSpnVerticalFlipProb->setValue(config.dAugVerticalFlipProb);  // 20260325 ZJH 垂直翻转概率
    m_pChkAffine->setChecked(config.bAugAffine);                   // 20260325 ZJH 仿射变换开关
    m_pSpnShearDeg->setValue(config.dAugShearDeg);                 // 20260325 ZJH 仿射剪切角度
    m_pSpnTranslate->setValue(config.dAugTranslate);               // 20260325 ZJH 仿射平移比例
    m_pChkRandomCrop->setChecked(config.bAugRandomCrop);           // 20260325 ZJH 随机裁剪开关
    m_pSpnCropScale->setValue(config.dAugCropScale);               // 20260325 ZJH 最小缩放比例

    // 20260325 ZJH 数据增强（颜色变换）
    m_pChkColorJitter->setChecked(config.bAugColorJitter);  // 20260325 ZJH 颜色抖动开关
    m_pSpnSaturation->setValue(config.dAugSaturation);      // 20260325 ZJH 饱和度抖动范围
    m_pSpnHue->setValue(config.dAugHue);                    // 20260325 ZJH 色调抖动范围

    // 20260325 ZJH 数据增强（噪声/遮挡）
    m_pChkGaussianNoise->setChecked(config.bAugGaussianNoise);  // 20260325 ZJH 高斯噪声开关
    m_pSpnNoiseStd->setValue(config.dAugGaussianNoiseStd);      // 20260325 ZJH 噪声标准差
    m_pChkGaussianBlur->setChecked(config.bAugGaussianBlur);    // 20260325 ZJH 高斯模糊开关
    m_pSpnBlurSigma->setValue(config.dAugBlurSigma);            // 20260325 ZJH 模糊 sigma
    m_pChkRandomErasing->setChecked(config.bAugRandomErasing);  // 20260325 ZJH 随机擦除开关
    m_pSpnErasingProb->setValue(config.dAugErasingProb);        // 20260325 ZJH 擦除概率
    m_pSpnErasingRatio->setValue(config.dAugErasingRatio);      // 20260325 ZJH 擦除面积比例

    // 20260325 ZJH 数据增强（高级混合）
    m_pChkMixup->setChecked(config.bAugMixup);            // 20260325 ZJH Mixup 开关
    m_pSpnMixupAlpha->setValue(config.dAugMixupAlpha);    // 20260325 ZJH Mixup alpha
    m_pChkCutMix->setChecked(config.bAugCutMix);          // 20260325 ZJH CutMix 开关
    m_pSpnCutMixAlpha->setValue(config.dAugCutMixAlpha);  // 20260325 ZJH CutMix alpha

    // 20260330 ZJH 预训练权重（新增控件）
    if (m_pChkUsePretrained && m_pEdtPretrainedPath) {
        bool bHasPretrained = !config.strPretrainedModelPath.isEmpty();
        m_pChkUsePretrained->setChecked(bHasPretrained);  // 20260330 ZJH 有路径则自动勾选
        m_pEdtPretrainedPath->setText(config.strPretrainedModelPath);
    }

    // 20260330 ZJH HSV 色彩空间抖动
    if (m_pChkHsvAugment) m_pChkHsvAugment->setChecked(config.bAugHsvAugment);
    if (m_pSpnHueShift)   m_pSpnHueShift->setValue(config.dAugHueShift);
    if (m_pSpnSatShift)   m_pSpnSatShift->setValue(config.dAugSatShift);
    if (m_pSpnValShift)   m_pSpnValShift->setValue(config.dAugValShift);

    // 20260330 ZJH 少样本学习
    if (m_pChkFewShot)      m_pChkFewShot->setChecked(config.bFewShot);
    if (m_pSpnShotsPerClass) m_pSpnShotsPerClass->setValue(config.nShotsPerClass);

    // 20260330 ZJH 模型优化（训练后剪枝）
    if (m_pChkPruning)     m_pChkPruning->setChecked(config.bPruning);
    if (m_pCboPruneMethod) {
        nIdx = m_pCboPruneMethod->findData(config.nPruneMethod);
        if (nIdx >= 0) m_pCboPruneMethod->setCurrentIndex(nIdx);
    }
    if (m_pSpnPruneRatio)  m_pSpnPruneRatio->setValue(config.dPruneRatio);

    // 20260402 ZJH [OPT-2.5] 增量训练 (Continual Learning)
    if (m_pChkContinualLearning) m_pChkContinualLearning->setChecked(config.bContinualLearning);
    if (m_pSpnEwcLambda) m_pSpnEwcLambda->setValue(config.dEwcLambda);

    // 20260325 ZJH ONNX 导出
    m_pChkExportOnnx->setChecked(config.bExportOnnx);  // 20260325 ZJH 自动导出 ONNX
}

// 20260322 ZJH 设置左面板控件启用/禁用
void TrainingPage::setControlsEnabled(bool bEnabled)
{
    // 20260322 ZJH 模型配置控件
    m_pCboFramework->setEnabled(bEnabled);
    m_pCboCapability->setEnabled(bEnabled);     // 20260330 ZJH 模型能力等级
    m_pCboArchitecture->setEnabled(bEnabled);
    m_pCboDevice->setEnabled(bEnabled);
    if (m_pCboAnomalyMode) m_pCboAnomalyMode->setEnabled(bEnabled);  // 20260330 ZJH 异常检测模式
    m_pCboOptimizer->setEnabled(bEnabled);
    m_pCboScheduler->setEnabled(bEnabled);

    // 20260322 ZJH 超参数控件
    m_pSpnLearningRate->setEnabled(bEnabled);
    m_pSpnBatchSize->setEnabled(bEnabled);
    m_pBtnAutoMaxBatch->setEnabled(bEnabled);  // 20260324 ZJH 自动最大批量按钮联动
    m_pSpnEpochs->setEnabled(bEnabled);
    if (m_pCboResolution) m_pCboResolution->setEnabled(bEnabled);  // 20260330 ZJH 分辨率预设
    m_pSpnInputSize->setEnabled(bEnabled);
    m_pSpnPatience->setEnabled(bEnabled);

    // 20260330 ZJH 预训练模型与标识
    if (m_pEdtPretrained) m_pEdtPretrained->setEnabled(bEnabled);
    if (m_pBtnBrowsePretrained) m_pBtnBrowsePretrained->setEnabled(bEnabled);
    if (m_pEdtModelTag) m_pEdtModelTag->setEnabled(bEnabled);

    // 20260330 ZJH 数据增强预设
    if (m_pCboAugPreset) m_pCboAugPreset->setEnabled(bEnabled);

    // 20260322 ZJH 数据增强控件（基础）
    m_pChkAugmentation->setEnabled(bEnabled);
    m_pSpnBrightness->setEnabled(bEnabled);
    m_pSpnFlipProb->setEnabled(bEnabled);
    m_pSpnRotation->setEnabled(bEnabled);

    // 20260324 ZJH 数据增强控件（扩展 — 几何变换）
    m_pSpnVerticalFlipProb->setEnabled(bEnabled);
    m_pChkAffine->setEnabled(bEnabled);
    m_pSpnShearDeg->setEnabled(bEnabled);
    m_pSpnTranslate->setEnabled(bEnabled);
    m_pChkRandomCrop->setEnabled(bEnabled);
    m_pSpnCropScale->setEnabled(bEnabled);

    // 20260324 ZJH 数据增强控件（扩展 — 颜色变换）
    m_pChkColorJitter->setEnabled(bEnabled);
    m_pSpnSaturation->setEnabled(bEnabled);
    m_pSpnHue->setEnabled(bEnabled);

    // 20260324 ZJH 数据增强控件（扩展 — 噪声/遮挡）
    m_pChkGaussianNoise->setEnabled(bEnabled);
    m_pSpnNoiseStd->setEnabled(bEnabled);
    m_pChkGaussianBlur->setEnabled(bEnabled);
    m_pSpnBlurSigma->setEnabled(bEnabled);
    m_pChkRandomErasing->setEnabled(bEnabled);
    m_pSpnErasingProb->setEnabled(bEnabled);
    m_pSpnErasingRatio->setEnabled(bEnabled);

    // 20260324 ZJH 数据增强控件（扩展 — 高级混合）
    m_pChkMixup->setEnabled(bEnabled);
    m_pSpnMixupAlpha->setEnabled(bEnabled);
    m_pChkCutMix->setEnabled(bEnabled);
    m_pSpnCutMixAlpha->setEnabled(bEnabled);

    // 20260330 ZJH 预训练权重控件
    if (m_pChkUsePretrained)       m_pChkUsePretrained->setEnabled(bEnabled);
    if (m_pEdtPretrainedPath)      m_pEdtPretrainedPath->setEnabled(bEnabled && m_pChkUsePretrained->isChecked());
    if (m_pBtnBrowsePretrainedPath) m_pBtnBrowsePretrainedPath->setEnabled(bEnabled && m_pChkUsePretrained->isChecked());

    // 20260330 ZJH HSV 色彩空间抖动控件
    if (m_pChkHsvAugment) m_pChkHsvAugment->setEnabled(bEnabled);
    if (m_pSpnHueShift)   m_pSpnHueShift->setEnabled(bEnabled && m_pChkHsvAugment->isChecked());
    if (m_pSpnSatShift)   m_pSpnSatShift->setEnabled(bEnabled && m_pChkHsvAugment->isChecked());
    if (m_pSpnValShift)   m_pSpnValShift->setEnabled(bEnabled && m_pChkHsvAugment->isChecked());

    // 20260330 ZJH 少样本学习控件
    if (m_pChkFewShot)      m_pChkFewShot->setEnabled(bEnabled);
    if (m_pSpnShotsPerClass) m_pSpnShotsPerClass->setEnabled(bEnabled && m_pChkFewShot->isChecked());

    // 20260330 ZJH 模型优化（剪枝）控件
    if (m_pChkPruning)     m_pChkPruning->setEnabled(bEnabled);
    if (m_pCboPruneMethod) m_pCboPruneMethod->setEnabled(bEnabled && m_pChkPruning->isChecked());
    if (m_pSpnPruneRatio)  m_pSpnPruneRatio->setEnabled(bEnabled && m_pChkPruning->isChecked());

    // 20260330 ZJH AI 数据合成按钮
    if (m_pBtnDataSynth) m_pBtnDataSynth->setEnabled(bEnabled);

    // 20260322 ZJH ONNX 导出
    m_pChkExportOnnx->setEnabled(bEnabled);
}

// 20260322 ZJH 刷新数据概览和前置检查
void TrainingPage::refreshDataOverview()
{
    if (!m_pProject || !m_pProject->dataset()) {
        // 20260322 ZJH 无项目或无数据集，显示默认值
        m_pLblTrainCount->setText(QStringLiteral("0"));
        m_pLblValCount->setText(QStringLiteral("0"));
        m_pLblTestCount->setText(QStringLiteral("0"));

        m_pLblCheckImages->setText(QStringLiteral("\u2717 图像已导入"));
        m_pLblCheckImages->setStyleSheet(QStringLiteral("color: #ef4444;"));
        m_pLblCheckLabels->setText(QStringLiteral("\u2717 标签已分配"));
        m_pLblCheckLabels->setStyleSheet(QStringLiteral("color: #ef4444;"));
        m_pLblCheckSplit->setText(QStringLiteral("\u2717 数据已拆分"));
        m_pLblCheckSplit->setStyleSheet(QStringLiteral("color: #ef4444;"));
        return;
    }

    ImageDataset* pDataset = m_pProject->dataset();

    // 20260322 ZJH 更新数据集数量
    int nTrainCount = pDataset->countBySplit(om::SplitType::Train);
    int nValCount   = pDataset->countBySplit(om::SplitType::Validation);
    int nTestCount  = pDataset->countBySplit(om::SplitType::Test);

    m_pLblTrainCount->setText(QString::number(nTrainCount));
    m_pLblValCount->setText(QString::number(nValCount));
    m_pLblTestCount->setText(QString::number(nTestCount));

    // 20260322 ZJH 前置检查 1: 图像是否已导入
    bool bHasImages = (pDataset->imageCount() > 0);
    if (bHasImages) {
        m_pLblCheckImages->setText(QStringLiteral("\u2713 图像已导入 (%1 张)").arg(pDataset->imageCount()));
        m_pLblCheckImages->setStyleSheet(QStringLiteral("color: #22c55e;"));  // 20260322 ZJH 绿色
    } else {
        m_pLblCheckImages->setText(QStringLiteral("\u2717 图像已导入"));
        m_pLblCheckImages->setStyleSheet(QStringLiteral("color: #ef4444;"));  // 20260322 ZJH 红色
    }

    // 20260405 ZJH [修复] 前置检查 2: 标签是否已分配（显示 已标注/总数 比例）
    int nLabeled = pDataset->labeledCount();  // 20260405 ZJH 已标注图像数
    int nTotal = pDataset->imageCount();      // 20260405 ZJH 总图像数
    if (nLabeled > 0) {
        if (nLabeled < nTotal) {
            // 20260405 ZJH 部分标注: 黄色警告，显示 已标注/总数
            m_pLblCheckLabels->setText(QStringLiteral("\u2713 标签已分配 (%1/%2 张)").arg(nLabeled).arg(nTotal));
            m_pLblCheckLabels->setStyleSheet(QStringLiteral("color: #eab308;"));  // 20260405 ZJH 黄色（部分标注）
        } else {
            // 20260405 ZJH 全部标注: 绿色
            m_pLblCheckLabels->setText(QStringLiteral("\u2713 标签已分配 (%1 张)").arg(nLabeled));
            m_pLblCheckLabels->setStyleSheet(QStringLiteral("color: #22c55e;"));
        }
    } else {
        m_pLblCheckLabels->setText(QStringLiteral("\u2717 标签已分配"));
        m_pLblCheckLabels->setStyleSheet(QStringLiteral("color: #ef4444;"));
    }

    // 20260322 ZJH 前置检查 3: 数据是否已拆分
    bool bHasSplit = (nTrainCount > 0);
    if (bHasSplit) {
        m_pLblCheckSplit->setText(QStringLiteral("\u2713 数据已拆分"));
        m_pLblCheckSplit->setStyleSheet(QStringLiteral("color: #22c55e;"));
    } else {
        m_pLblCheckSplit->setText(QStringLiteral("\u2717 数据已拆分"));
        m_pLblCheckSplit->setStyleSheet(QStringLiteral("color: #ef4444;"));
    }
}

// 20260322 ZJH 更新按钮启用状态（根据训练是否在运行）
void TrainingPage::updateButtonStates()
{
    bool bRunning = (m_pSession && m_pSession->isRunning());

    m_pBtnStart->setEnabled(!bRunning);   // 20260322 ZJH 运行时禁用开始
    m_pBtnPause->setEnabled(bRunning);    // 20260322 ZJH 运行时启用暂停
    m_pBtnStop->setEnabled(bRunning);     // 20260322 ZJH 运行时启用停止
    m_pBtnResume->setEnabled(false);      // 20260322 ZJH 默认禁用继续（暂停时才启用）
}

// 20260402 ZJH 格式化剩余时间为人类可读字符串
// 使用滑动平均的 m_dAvgEpochSec 计算，比简单除法更平滑
QString TrainingPage::formatRemainingTime(int nRemainingEpochs) const
{
    // 20260402 ZJH 计算剩余总秒数
    double dRemainingSec = m_dAvgEpochSec * nRemainingEpochs;  // 20260402 ZJH 剩余秒 = 平均每epoch秒 * 剩余epoch数

    // 20260402 ZJH 分解为时分秒
    int nTotalSec = static_cast<int>(dRemainingSec);  // 20260402 ZJH 取整数秒
    int nHours   = nTotalSec / 3600;     // 20260402 ZJH 小时部分
    int nMinutes = (nTotalSec % 3600) / 60;  // 20260402 ZJH 分钟部分
    int nSeconds = nTotalSec % 60;       // 20260402 ZJH 秒部分

    // 20260402 ZJH 根据时间量级选择显示格式
    if (nHours > 0) {
        // 20260402 ZJH 超过 1 小时: "Xh Xm Xs"
        return QStringLiteral("预估剩余: %1h %2m %3s").arg(nHours).arg(nMinutes).arg(nSeconds);
    } else if (nMinutes > 0) {
        // 20260402 ZJH 超过 1 分钟: "X分X秒"
        return QStringLiteral("预估剩余: %1分%2秒").arg(nMinutes).arg(nSeconds);
    } else {
        // 20260402 ZJH 不到 1 分钟: "X秒"
        return QStringLiteral("预估剩余: %1秒").arg(nSeconds);
    }
}

// 20260402 ZJH 训练质量诊断系统
// 每个 epoch 结束后分析 loss/metric 走势，自动检测常见训练异常并给出建议
void TrainingPage::runTrainingDiagnostics(int nEpoch, int nTotalEpochs,
                                           float fTrainLoss, float fValLoss, float fMetric)
{
    // 20260402 ZJH 1. 记录历史数据（追加到向量末尾）
    m_vecTrainLossHistory.push_back(fTrainLoss);  // 20260402 ZJH 追加训练 loss
    m_vecValLossHistory.push_back(fValLoss);      // 20260402 ZJH 追加验证 loss
    m_vecValMetricHistory.push_back(fMetric);     // 20260402 ZJH 追加验证指标

    size_t nHistSize = m_vecTrainLossHistory.size();  // 20260402 ZJH 当前历史长度

    // 20260402 ZJH 至少需要 2 个 epoch 才能进行趋势分析
    if (nHistSize < 2) {
        return;  // 20260402 ZJH 数据不足，跳过诊断
    }

    // 20260402 ZJH 2. 过拟合检测: train_loss 下降但 val_loss 上升连续 3+ epoch
    {
        float fPrevTrain = m_vecTrainLossHistory[nHistSize - 2];  // 20260402 ZJH 上一 epoch 训练 loss
        float fPrevVal   = m_vecValLossHistory[nHistSize - 2];    // 20260402 ZJH 上一 epoch 验证 loss
        bool bTrainDecreased = (fTrainLoss < fPrevTrain);  // 20260402 ZJH 训练 loss 是否下降
        bool bValIncreased   = (fValLoss > fPrevVal);      // 20260402 ZJH 验证 loss 是否上升

        if (bTrainDecreased && bValIncreased) {
            m_nDiagOverfitCount++;  // 20260402 ZJH 过拟合趋势持续 +1
        } else {
            m_nDiagOverfitCount = 0;  // 20260402 ZJH 趋势中断，重置计数
        }

        // 20260402 ZJH 连续 3 epoch 过拟合时发出警告（仅在第 3 次触发时输出，避免刷屏）
        if (m_nDiagOverfitCount == 3) {
            onTrainingLog(QStringLiteral("[诊断] 检测到过拟合 (连续 %1 epoch train_loss↓ val_loss↑) "
                                        "— 建议: 启用数据增强 / 减少模型复杂度 / 使用 Dropout / 增加训练数据")
                          .arg(m_nDiagOverfitCount));
        }
    }

    // 20260402 ZJH 3. 欠拟合检测: 前 5 epoch 平均 loss 下降幅度 < 5%
    if (nHistSize == 5) {
        // 20260402 ZJH 仅在第 5 个 epoch 时触发一次检查
        float fFirstLoss = m_vecTrainLossHistory[0];   // 20260402 ZJH 第 1 epoch 的 loss
        float fFifthLoss = m_vecTrainLossHistory[4];   // 20260402 ZJH 第 5 epoch 的 loss

        // 20260402 ZJH 避免除零: 第一个 loss 为 0 时跳过检查
        if (fFirstLoss > 1e-8f) {
            float fDeclineRate = (fFirstLoss - fFifthLoss) / fFirstLoss;  // 20260402 ZJH 下降比例
            if (fDeclineRate < 0.05f) {
                // 20260402 ZJH 下降不足 5%，可能欠拟合或学习率过小
                onTrainingLog(QStringLiteral("[诊断] 训练欠拟合 (前 5 epoch loss 下降仅 %1%) "
                                            "— 建议: 增大学习率 / 增加模型容量 / 检查数据标注质量")
                              .arg(static_cast<double>(fDeclineRate * 100.0f), 0, 'f', 1));
            }
        }
    }

    // 20260402 ZJH 4. 学习率过大检测: loss 连续 3+ epoch 正负交替震荡
    if (nHistSize >= 3) {
        // 20260402 ZJH 获取最近 3 个 epoch 的 train_loss 差值符号
        float fDiff1 = m_vecTrainLossHistory[nHistSize - 2] - m_vecTrainLossHistory[nHistSize - 3];  // 20260402 ZJH 倒数第2 - 倒数第3
        float fDiff2 = m_vecTrainLossHistory[nHistSize - 1] - m_vecTrainLossHistory[nHistSize - 2];  // 20260402 ZJH 倒数第1 - 倒数第2

        // 20260402 ZJH 正负交替: 两个差值符号相反
        bool bOscillating = (fDiff1 * fDiff2 < 0.0f);  // 20260402 ZJH 乘积为负表示符号相反

        if (bOscillating) {
            m_nDiagOscillCount++;  // 20260402 ZJH 震荡计数 +1
        } else {
            m_nDiagOscillCount = 0;  // 20260402 ZJH 不震荡，重置
        }

        // 20260402 ZJH 连续 3 次震荡时警告（仅触发一次）
        if (m_nDiagOscillCount == 3) {
            double dCurrentLR = m_pSpnLearningRate->value();  // 20260402 ZJH 当前学习率
            onTrainingLog(QStringLiteral("[诊断] 学习率过大导致损失震荡 (连续 %1 epoch 正负交替) "
                                        "— 建议: 降低 LR 至 %2 (当前值的 1/10)")
                          .arg(m_nDiagOscillCount)
                          .arg(dCurrentLR / 10.0, 0, 'e', 1));
        }
    }

    // 20260402 ZJH 5. 梯度消失/停滞检测: 验证指标连续 10 epoch 无改善
    {
        // 20260402 ZJH 与最近的最佳指标比较
        float fBestMetric = *std::max_element(m_vecValMetricHistory.begin(), m_vecValMetricHistory.end());  // 20260402 ZJH 历史最佳指标

        // 20260402 ZJH 检查当前指标是否有改善（容差 0.001 避免浮点精度问题）
        if (fMetric >= fBestMetric - 0.001f) {
            m_nDiagNoImproveCount = 0;  // 20260402 ZJH 有改善，重置计数
        } else {
            m_nDiagNoImproveCount++;  // 20260402 ZJH 无改善，计数 +1
        }

        // 20260402 ZJH 连续 10 epoch 无改善时警告
        if (m_nDiagNoImproveCount == 10) {
            onTrainingLog(QStringLiteral("[诊断] 验证指标连续 10 epoch 无改善 (梯度消失/训练停滞) "
                                        "— 建议: 检查学习率调度器 / 使用残差连接 / 尝试不同优化器"));
        }
    }

    // 20260402 ZJH 6. 正面反馈: 训练进行顺利时给予积极提示
    // 条件: 每 10 epoch 检查一次，验证 loss 持续下降且验证指标持续上升
    if (nEpoch > 0 && nEpoch % 10 == 0 && nHistSize >= 10) {
        // 20260402 ZJH 检查最近 5 epoch 验证 loss 是否整体下降
        bool bValLossDecreasing = true;  // 20260402 ZJH 假设下降中
        for (size_t i = nHistSize - 4; i < nHistSize; ++i) {
            if (m_vecValLossHistory[i] > m_vecValLossHistory[i - 1] + 0.001f) {
                bValLossDecreasing = false;  // 20260402 ZJH 发现上升，打破假设
                break;
            }
        }

        // 20260402 ZJH 如果验证 loss 持续下降且无过拟合/震荡问题
        if (bValLossDecreasing && m_nDiagOverfitCount == 0 && m_nDiagOscillCount == 0) {
            onTrainingLog(QStringLiteral("[诊断] 训练收敛良好，验证指标稳步提升 (Epoch %1/%2)")
                          .arg(nEpoch).arg(nTotalEpochs));
        }
    }
}
