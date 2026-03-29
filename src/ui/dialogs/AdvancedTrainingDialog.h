// 20260323 ZJH AdvancedTrainingDialog — 高级训练配置对话框
// 提供详细的训练超参数调整、调度器配置、正则化设置
// 用于需要精细控制训练过程的高级用户
#pragma once

#include <QDialog>            // 20260323 ZJH 对话框基类
#include <QTabWidget>         // 20260323 ZJH 标签页容器
#include <QDoubleSpinBox>     // 20260323 ZJH 浮点微调框
#include <QSpinBox>           // 20260323 ZJH 整数微调框
#include <QComboBox>          // 20260323 ZJH 下拉选择框
#include <QCheckBox>          // 20260323 ZJH 复选框
#include <QDialogButtonBox>   // 20260323 ZJH 标准按钮组

// 20260323 ZJH 高级训练配置结构体
struct AdvancedTrainingConfig
{
    // ===== 优化器 =====
    double dWeightDecay = 0.0001;      // 20260323 ZJH 权重衰减
    double dMomentum = 0.9;            // 20260323 ZJH 动量（SGD）
    double dBeta1 = 0.9;              // 20260323 ZJH Adam beta1
    double dBeta2 = 0.999;            // 20260323 ZJH Adam beta2
    double dEpsilon = 1e-8;           // 20260323 ZJH Adam epsilon
    bool bAmsGrad = false;            // 20260323 ZJH 是否启用 AMSGrad
    double dGradClipNorm = 0.0;       // 20260323 ZJH 梯度裁剪范数（0=不裁剪）

    // ===== 调度器 =====
    double dWarmupRatio = 0.1;        // 20260323 ZJH 预热比例
    double dMinLr = 1e-6;             // 20260323 ZJH 最小学习率
    int nStepSize = 10;               // 20260323 ZJH StepLR 步长
    double dStepGamma = 0.1;          // 20260323 ZJH StepLR 衰减因子

    // ===== 正则化 =====
    double dDropout = 0.0;            // 20260323 ZJH Dropout 率
    double dLabelSmoothing = 0.0;     // 20260323 ZJH 标签平滑
    bool bMixup = false;              // 20260323 ZJH Mixup 增强
    double dMixupAlpha = 0.2;         // 20260323 ZJH Mixup alpha

    // ===== 训练策略 =====
    int nAccumulationSteps = 1;       // 20260323 ZJH 梯度累积步数
    bool bFP16 = false;               // 20260323 ZJH 混合精度训练
    int nNumWorkers = 4;              // 20260323 ZJH 数据加载线程数
    bool bPinMemory = true;           // 20260323 ZJH 锁页内存
};

// 20260323 ZJH 高级训练配置对话框
class AdvancedTrainingDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit AdvancedTrainingDialog(QWidget* pParent = nullptr);

    // 20260323 ZJH 构造函数（带初始配置）
    explicit AdvancedTrainingDialog(const AdvancedTrainingConfig& config, QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~AdvancedTrainingDialog() override = default;

    // 20260323 ZJH 获取配置
    AdvancedTrainingConfig config() const;

private:
    // 20260323 ZJH 创建优化器标签页
    QWidget* createOptimizerTab();

    // 20260323 ZJH 创建调度器标签页
    QWidget* createSchedulerTab();

    // 20260323 ZJH 创建正则化标签页
    QWidget* createRegularizationTab();

    // 20260323 ZJH 创建训练策略标签页
    QWidget* createStrategyTab();

    // 20260323 ZJH 从配置填充控件
    void loadConfig(const AdvancedTrainingConfig& config);

    // ===== 优化器控件 =====
    QDoubleSpinBox* m_pSpnWeightDecay;
    QDoubleSpinBox* m_pSpnMomentum;
    QDoubleSpinBox* m_pSpnBeta1;
    QDoubleSpinBox* m_pSpnBeta2;
    QDoubleSpinBox* m_pSpnEpsilon;
    QCheckBox* m_pChkAmsGrad;
    QDoubleSpinBox* m_pSpnGradClip;

    // ===== 调度器控件 =====
    QDoubleSpinBox* m_pSpnWarmupRatio;
    QDoubleSpinBox* m_pSpnMinLr;
    QSpinBox* m_pSpnStepSize;
    QDoubleSpinBox* m_pSpnStepGamma;

    // ===== 正则化控件 =====
    QDoubleSpinBox* m_pSpnDropout;
    QDoubleSpinBox* m_pSpnLabelSmooth;
    QCheckBox* m_pChkMixup;
    QDoubleSpinBox* m_pSpnMixupAlpha;

    // ===== 策略控件 =====
    QSpinBox* m_pSpnAccumSteps;
    QCheckBox* m_pChkFP16;
    QSpinBox* m_pSpnNumWorkers;
    QCheckBox* m_pChkPinMemory;
};
