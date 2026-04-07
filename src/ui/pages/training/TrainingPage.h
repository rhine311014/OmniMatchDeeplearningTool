// 20260322 ZJH TrainingPage — 模型训练页面
// BasePage 子类，三栏布局：
//   左面板(300px): 模型配置 + 超参数 + 数据增强 + 导出选项
//   中央面板: 损失曲线图 + 进度条 + 控制按钮
//   右面板(250px): 数据概览 + 前置检查 + 训练状态 + 日志

#pragma once

#include "ui/pages/BasePage.h"              // 20260322 ZJH 页面基类
#include "core/training/TrainingConfig.h"   // 20260322 ZJH 训练配置结构体

#include <QComboBox>         // 20260322 ZJH 下拉选择框
#include <QSpinBox>          // 20260322 ZJH 整数微调框
#include <QDoubleSpinBox>    // 20260322 ZJH 浮点数微调框
#include <QCheckBox>         // 20260322 ZJH 复选框
#include <QPushButton>       // 20260322 ZJH 按钮
#include <QLineEdit>         // 20260330 ZJH 预训练模型路径+模型标识输入框
#include <QLabel>            // 20260322 ZJH 文本标签
#include <QProgressBar>      // 20260322 ZJH 进度条
#include <QTextEdit>         // 20260322 ZJH 日志显示文本框
#include <QThread>           // 20260322 ZJH 训练工作线程
#include <QElapsedTimer>     // 20260322 ZJH 计时器用于估算剩余时间
#include <chrono>            // 20260402 ZJH steady_clock 精确 epoch 计时
#include <vector>            // 20260402 ZJH 训练诊断历史数据容器

// 20260322 ZJH 前向声明
class TrainingLossChart;
class TrainingSession;

// 20260322 ZJH 模型训练页面
// 提供完整的训练配置 UI、实时损失曲线、训练控制和日志显示
class TrainingPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，创建三栏布局并初始化全部控件
    // 参数: pParent - 父控件指针
    explicit TrainingPage(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数，停止训练线程
    ~TrainingPage() override;

    // ===== BasePage 生命周期回调 =====

    // 20260322 ZJH 页面切换到前台时调用
    void onEnter() override;

    // 20260322 ZJH 页面离开前台时调用
    void onLeave() override;

    // 20260324 ZJH 项目加载后调用，刷新前置检查和数据概览（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时调用，清空显示（Template Method 扩展点）
    void onProjectClosedImpl() override;

private slots:
    // 20260322 ZJH 开始训练按钮点击
    void onStartTraining();

    // 20260322 ZJH 暂停训练按钮点击
    void onPauseTraining();

    // 20260322 ZJH 停止训练按钮点击
    void onStopTraining();

    // 20260322 ZJH 继续训练按钮点击（从暂停恢复）
    void onResumeTraining();

    // 20260322 ZJH Epoch 完成处理
    // 参数: nEpoch - 当前轮次; nTotal - 总轮次
    //       dTrainLoss - 训练损失; dValLoss - 验证损失; dMetric - 评估指标
    void onEpochCompleted(int nEpoch, int nTotal, double dTrainLoss, double dValLoss, double dMetric);

    // 20260322 ZJH 训练完成处理
    // 参数: bSuccess - 是否成功; strMessage - 完成消息
    void onTrainingFinished(bool bSuccess, const QString& strMessage);

    // 20260322 ZJH 训练日志接收
    // 参数: strMessage - 日志消息
    void onTrainingLog(const QString& strMessage);

    // 20260322 ZJH 进度百分比更新
    // 参数: nPercent - 进度值 0~100
    void onProgressChanged(int nPercent);

    // 20260322 ZJH 任务类型变化时更新模型架构下拉框
    void refreshArchitectureCombo();

private:
    // ===== UI 创建辅助方法 =====

    // 20260322 ZJH 创建左面板（模型配置 + 超参数 + 数据增强）
    QWidget* createLeftPanel();

    // 20260322 ZJH 创建中央面板（损失曲线 + 进度 + 控制按钮）
    QWidget* createCenterPanel();

    // 20260322 ZJH 创建右面板（数据概览 + 前置检查 + 训练状态 + 日志）
    QWidget* createRightPanel();

    // 20260322 ZJH 从 UI 控件读取当前配置到 TrainingConfig 结构体
    TrainingConfig gatherConfig() const;

    // 20260325 ZJH 将 TrainingConfig 回填到 UI 控件（项目加载时恢复保存的参数）
    void restoreConfigToUI(const TrainingConfig& config);

    // 20260322 ZJH 设置控件启用/禁用状态（训练期间禁用左面板）
    void setControlsEnabled(bool bEnabled);

    // 20260322 ZJH 刷新右面板的数据概览和前置检查
    void refreshDataOverview();

    // 20260322 ZJH 更新训练控制按钮的启用状态
    void updateButtonStates();

    // 20260322 ZJH 创建分组框辅助方法
    // 参数: strTitle - 分组框标题
    // 返回: 包含垂直布局的 QWidget 指针
    QWidget* createGroupBox(const QString& strTitle);

    // ===== 左面板控件 =====

    // 20260322 ZJH 模型配置分组
    QComboBox* m_pCboFramework;      // 20260322 ZJH 训练框架下拉框
    QComboBox* m_pCboCapability;     // 20260330 ZJH 模型能力等级下拉框（轻量化/普通/高精度）
    QComboBox* m_pCboArchitecture;   // 20260322 ZJH 模型架构下拉框（高级选项）
    QPushButton* m_pBtnToggleArch = nullptr;  // 20260330 ZJH 展开/收起模型架构选项按钮
    QWidget*  m_pArchContainer = nullptr;     // 20260330 ZJH 架构选择容器（默认折叠）
    QComboBox* m_pCboDevice;         // 20260322 ZJH 设备类型下拉框
    QComboBox* m_pCboAnomalyMode = nullptr;   // 20260330 ZJH 异常检测训练模式下拉框（极速/高精度）
    QLabel*    m_pLblAnomalyMode = nullptr;   // 20260330 ZJH 异常检测训练模式标签
    QComboBox* m_pCboOptimizer;      // 20260322 ZJH 优化器下拉框
    QComboBox* m_pCboScheduler;      // 20260322 ZJH 调度器下拉框

    // 20260322 ZJH 超参数分组
    QDoubleSpinBox* m_pSpnLearningRate;  // 20260322 ZJH 学习率微调框
    QSpinBox*       m_pSpnBatchSize;     // 20260322 ZJH 批量大小微调框
    QPushButton*    m_pBtnAutoMaxBatch;  // 20260324 ZJH 自动最大批量大小按钮
    QPushButton*    m_pBtnAutoRecommend = nullptr;  // 20260325 ZJH 自动推荐最优训练参数按钮
    QSpinBox*       m_pSpnEpochs;        // 20260322 ZJH 训练轮次微调框
    QComboBox*      m_pCboResolution = nullptr;  // 20260330 ZJH 输入分辨率预设下拉框
    QSpinBox*       m_pSpnInputSize;     // 20260322 ZJH 输入尺寸微调框（Custom时显示）
    QSpinBox*       m_pSpnPatience;      // 20260322 ZJH 早停耐心微调框

    // 20260330 ZJH 预训练模型与标识
    QLineEdit*   m_pEdtPretrained = nullptr;      // 20260330 ZJH 预训练模型路径编辑框
    QPushButton* m_pBtnBrowsePretrained = nullptr; // 20260330 ZJH 浏览预训练模型按钮
    QLineEdit*   m_pEdtModelTag = nullptr;         // 20260330 ZJH 模型标识输入框

    // 20260322 ZJH 数据增强分组
    QComboBox*      m_pCboAugPreset = nullptr;    // 20260330 ZJH 增强策略预设下拉框（默认/手动）
    QCheckBox*      m_pChkAugmentation;  // 20260322 ZJH 启用增强复选框
    QPushButton*    m_pBtnToggleAug = nullptr;    // 20260325 ZJH 展开/收起增强选项按钮
    QWidget*        m_pAugContainer = nullptr;    // 20260325 ZJH 增强选项容器（可折叠）
    QDoubleSpinBox* m_pSpnBrightness;    // 20260322 ZJH 亮度增强微调框
    QDoubleSpinBox* m_pSpnFlipProb;      // 20260322 ZJH 水平翻转概率微调框
    QDoubleSpinBox* m_pSpnRotation;      // 20260322 ZJH 旋转角度微调框

    // 20260324 ZJH 数据增强分组（扩展 — 几何变换）
    QDoubleSpinBox* m_pSpnVerticalFlipProb;  // 20260324 ZJH 垂直翻转概率微调框
    QCheckBox*      m_pChkAffine;            // 20260324 ZJH 仿射变换启用复选框
    QDoubleSpinBox* m_pSpnShearDeg;          // 20260324 ZJH 仿射剪切角度微调框
    QDoubleSpinBox* m_pSpnTranslate;         // 20260324 ZJH 仿射平移比例微调框
    QCheckBox*      m_pChkRandomCrop;        // 20260324 ZJH 随机缩放裁剪启用复选框
    QDoubleSpinBox* m_pSpnCropScale;         // 20260324 ZJH 裁剪最小缩放比例微调框

    // 20260324 ZJH 数据增强分组（扩展 — 颜色变换）
    QCheckBox*      m_pChkColorJitter;   // 20260324 ZJH 颜色抖动启用复选框
    QDoubleSpinBox* m_pSpnSaturation;    // 20260324 ZJH 饱和度抖动范围微调框
    QDoubleSpinBox* m_pSpnHue;           // 20260324 ZJH 色调抖动范围微调框

    // 20260324 ZJH 数据增强分组（扩展 — 噪声/遮挡）
    QCheckBox*      m_pChkGaussianNoise;  // 20260324 ZJH 高斯噪声启用复选框
    QDoubleSpinBox* m_pSpnNoiseStd;       // 20260324 ZJH 噪声标准差微调框
    QCheckBox*      m_pChkGaussianBlur;   // 20260324 ZJH 高斯模糊启用复选框
    QDoubleSpinBox* m_pSpnBlurSigma;      // 20260324 ZJH 模糊 sigma 微调框
    QCheckBox*      m_pChkRandomErasing;  // 20260324 ZJH 随机擦除启用复选框
    QDoubleSpinBox* m_pSpnErasingProb;    // 20260324 ZJH 擦除概率微调框
    QDoubleSpinBox* m_pSpnErasingRatio;   // 20260324 ZJH 擦除面积比例微调框

    // 20260324 ZJH 数据增强分组（扩展 — 高级混合）
    QCheckBox*      m_pChkMixup;       // 20260324 ZJH Mixup 启用复选框
    QDoubleSpinBox* m_pSpnMixupAlpha;  // 20260324 ZJH Mixup alpha 微调框
    QCheckBox*      m_pChkCutMix;      // 20260324 ZJH CutMix 启用复选框
    QDoubleSpinBox* m_pSpnCutMixAlpha; // 20260324 ZJH CutMix alpha 微调框

    // 20260330 ZJH 预训练权重启用开关（新增：控制是否使用预训练权重）
    QCheckBox*   m_pChkUsePretrained = nullptr;    // 20260330 ZJH 使用预训练权重复选框
    QLineEdit*   m_pEdtPretrainedPath = nullptr;   // 20260330 ZJH 预训练权重路径显示（只读）
    QPushButton* m_pBtnBrowsePretrainedPath = nullptr; // 20260330 ZJH 浏览预训练权重文件按钮

    // 20260330 ZJH HSV 色彩空间抖动增强
    QCheckBox*      m_pChkHsvAugment = nullptr;   // 20260330 ZJH HSV 增强启用复选框
    QDoubleSpinBox* m_pSpnHueShift = nullptr;     // 20260330 ZJH 色调偏移微调框
    QDoubleSpinBox* m_pSpnSatShift = nullptr;     // 20260330 ZJH 饱和度偏移微调框
    QDoubleSpinBox* m_pSpnValShift = nullptr;     // 20260330 ZJH 明度偏移微调框

    // 20260330 ZJH 少样本学习
    QCheckBox* m_pChkFewShot = nullptr;           // 20260330 ZJH 少样本学习模式复选框
    QSpinBox*  m_pSpnShotsPerClass = nullptr;     // 20260330 ZJH 每类样本数微调框

    // 20260330 ZJH 模型优化（训练后剪枝）
    QCheckBox*      m_pChkPruning = nullptr;      // 20260330 ZJH 训练后剪枝启用复选框
    QComboBox*      m_pCboPruneMethod = nullptr;  // 20260330 ZJH 剪枝方法下拉框
    QDoubleSpinBox* m_pSpnPruneRatio = nullptr;   // 20260330 ZJH 剪枝比例微调框

    // 20260402 ZJH [OPT-2.5] 增量训练 (Continual Learning)
    QCheckBox*      m_pChkContinualLearning = nullptr;  // 20260402 ZJH 增量训练启用复选框
    QDoubleSpinBox* m_pSpnEwcLambda = nullptr;          // 20260402 ZJH EWC 正则化系数微调框

    // 20260330 ZJH AI 数据合成
    QPushButton* m_pBtnDataSynth = nullptr;       // 20260330 ZJH AI 数据合成按钮

    // 20260322 ZJH ONNX 导出
    QCheckBox* m_pChkExportOnnx;  // 20260322 ZJH 导出 ONNX 复选框

    // ===== 中央面板控件 =====

    TrainingLossChart* m_pLossChart;   // 20260322 ZJH 损失曲线图表
    QProgressBar*      m_pProgressBar; // 20260322 ZJH 进度条
    QLabel*            m_pLblStatus;   // 20260322 ZJH 状态文字标签

    // 20260322 ZJH 控制按钮
    QPushButton* m_pBtnStart;    // 20260322 ZJH 开始训练
    QPushButton* m_pBtnPause;    // 20260322 ZJH 暂停训练
    QPushButton* m_pBtnStop;     // 20260322 ZJH 停止训练
    QPushButton* m_pBtnResume;   // 20260322 ZJH 继续训练

    // ===== 右面板控件 =====

    // 20260322 ZJH 数据概览
    QLabel* m_pLblTrainCount;    // 20260322 ZJH 训练集数量
    QLabel* m_pLblValCount;      // 20260322 ZJH 验证集数量
    QLabel* m_pLblTestCount;     // 20260322 ZJH 测试集数量

    // 20260322 ZJH 前置检查
    QLabel* m_pLblCheckImages;   // 20260322 ZJH 图像已导入检查
    QLabel* m_pLblCheckLabels;   // 20260322 ZJH 标签已分配检查
    QLabel* m_pLblCheckSplit;    // 20260322 ZJH 数据已拆分检查

    // 20260322 ZJH 训练状态
    QLabel* m_pLblCurrentEpoch;  // 20260322 ZJH 当前 Epoch
    QLabel* m_pLblBestLoss;      // 20260322 ZJH 最佳损失
    QLabel* m_pLblTimeRemaining; // 20260322 ZJH 预计剩余时间
    QLabel* m_pLblEarlyStopCount;  // 20260322 ZJH 早停计数

    // 20260322 ZJH 训练日志
    QTextEdit* m_pTxtLog;  // 20260322 ZJH 只读日志文本框

    // ===== 训练会话管理 =====

    TrainingSession* m_pSession;       // 20260322 ZJH 训练会话对象（moveToThread）
    QThread*         m_pWorkerThread;  // 20260322 ZJH 训练工作线程

    // 20260322 ZJH 训练状态跟踪
    double m_dBestLoss = 1e9;          // 20260322 ZJH 历史最佳验证损失
    int    m_nEarlyStopCount = 0;      // 20260322 ZJH 当前早停计数
    QElapsedTimer m_elapsedTimer;      // 20260322 ZJH 训练计时器
    int    m_nCurrentEpoch = 0;        // 20260322 ZJH 当前 Epoch
    int    m_nTotalEpochs = 0;         // 20260322 ZJH 总 Epoch 数

    // 20260322 ZJH 左面板容器（训练中禁用）
    QWidget* m_pLeftPanel = nullptr;

    // 20260402 ZJH ===== 训练诊断系统 =====
    std::vector<float> m_vecTrainLossHistory;   // 20260402 ZJH 历史训练 loss（按 epoch 顺序）
    std::vector<float> m_vecValLossHistory;     // 20260402 ZJH 历史验证 loss（按 epoch 顺序）
    std::vector<float> m_vecValMetricHistory;   // 20260402 ZJH 历史验证指标（按 epoch 顺序）
    std::chrono::steady_clock::time_point m_tpEpochStart;  // 20260402 ZJH epoch 计时起点
    double m_dAvgEpochSec = 0.0;               // 20260402 ZJH 平均每 epoch 秒数（滑动平均）
    int m_nDiagOverfitCount = 0;               // 20260402 ZJH 过拟合连续计数（train↓val↑）
    int m_nDiagOscillCount = 0;                // 20260402 ZJH 损失震荡连续计数（正负交替）
    int m_nDiagNoImproveCount = 0;             // 20260402 ZJH 验证指标无改善连续计数

    // 20260402 ZJH 运行训练诊断分析（每 epoch 结束后调用）
    // 参数: nEpoch - 当前轮次; nTotalEpochs - 总轮次
    //       fTrainLoss - 当前训练损失; fValLoss - 当前验证损失; fMetric - 当前验证指标
    void runTrainingDiagnostics(int nEpoch, int nTotalEpochs, float fTrainLoss, float fValLoss, float fMetric);

    // 20260402 ZJH 格式化剩余时间为人类可读字符串
    // 参数: nRemainingEpochs - 剩余 epoch 数
    // 返回: "XX分XX秒" 格式的字符串
    QString formatRemainingTime(int nRemainingEpochs) const;
};
