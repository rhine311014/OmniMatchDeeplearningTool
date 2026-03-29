// 20260323 ZJH AdvancedTrainingDialog — 高级训练配置对话框实现

#include "ui/dialogs/AdvancedTrainingDialog.h"

#include <QVBoxLayout>    // 20260323 ZJH 垂直布局
#include <QFormLayout>    // 20260323 ZJH 表单布局
#include <QLabel>         // 20260323 ZJH 标签

// 20260323 ZJH 构造函数
AdvancedTrainingDialog::AdvancedTrainingDialog(QWidget* pParent)
    : AdvancedTrainingDialog(AdvancedTrainingConfig(), pParent)
{
}

// 20260323 ZJH 构造函数（带初始配置）
AdvancedTrainingDialog::AdvancedTrainingDialog(const AdvancedTrainingConfig& config, QWidget* pParent)
    : QDialog(pParent)
{
    setWindowTitle("高级训练配置");  // 20260324 ZJH 窗口标题汉化
    setMinimumSize(500, 450);

    // 20260323 ZJH 暗色主题
    setStyleSheet("QDialog { background: #1e2028; color: #e2e8f0; }"
                  "QTabWidget::pane { border: 1px solid #334155; background: #1e2028; }"
                  "QTabBar::tab { background: #22262e; color: #94a3b8; padding: 8px 16px; border: 1px solid #334155; }"
                  "QTabBar::tab:selected { background: #2563eb; color: white; }"
                  "QDoubleSpinBox, QSpinBox { background: #22262e; color: #e2e8f0; border: 1px solid #334155; padding: 4px; }"
                  "QCheckBox { color: #e2e8f0; }");

    QVBoxLayout* pMainLayout = new QVBoxLayout(this);

    // 20260323 ZJH 标签页容器
    QTabWidget* pTabs = new QTabWidget;
    pTabs->addTab(createOptimizerTab(), "优化器");        // 20260324 ZJH 标签页标题汉化
    pTabs->addTab(createSchedulerTab(), "调度器");       // 20260324 ZJH 标签页标题汉化
    pTabs->addTab(createRegularizationTab(), "正则化");  // 20260324 ZJH 标签页标题汉化
    pTabs->addTab(createStrategyTab(), "策略");           // 20260324 ZJH 标签页标题汉化
    pMainLayout->addWidget(pTabs);

    // 20260323 ZJH 按钮组
    QDialogButtonBox* pBtnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(pBtnBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(pBtnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    pMainLayout->addWidget(pBtnBox);

    // 20260323 ZJH 从配置填充控件
    loadConfig(config);
}

// 20260323 ZJH 创建优化器标签页
QWidget* AdvancedTrainingDialog::createOptimizerTab()
{
    QWidget* pTab = new QWidget;
    QFormLayout* pForm = new QFormLayout(pTab);

    m_pSpnWeightDecay = new QDoubleSpinBox;
    m_pSpnWeightDecay->setRange(0.0, 0.1);
    m_pSpnWeightDecay->setDecimals(6);
    m_pSpnWeightDecay->setSingleStep(0.0001);
    pForm->addRow("权重衰减 (Weight Decay):", m_pSpnWeightDecay);  // 20260324 ZJH 汉化

    m_pSpnMomentum = new QDoubleSpinBox;
    m_pSpnMomentum->setRange(0.0, 0.999);
    m_pSpnMomentum->setDecimals(3);
    pForm->addRow("动量 (SGD):", m_pSpnMomentum);  // 20260324 ZJH 汉化

    m_pSpnBeta1 = new QDoubleSpinBox;
    m_pSpnBeta1->setRange(0.0, 0.999);
    m_pSpnBeta1->setDecimals(3);
    pForm->addRow("Beta1 (Adam):", m_pSpnBeta1);  // 20260324 ZJH 技术术语保留英文

    m_pSpnBeta2 = new QDoubleSpinBox;
    m_pSpnBeta2->setRange(0.0, 0.9999);
    m_pSpnBeta2->setDecimals(4);
    pForm->addRow("Beta2 (Adam):", m_pSpnBeta2);  // 20260324 ZJH 技术术语保留英文

    m_pSpnEpsilon = new QDoubleSpinBox;
    m_pSpnEpsilon->setRange(1e-10, 1e-4);
    m_pSpnEpsilon->setDecimals(10);
    m_pSpnEpsilon->setSingleStep(1e-9);
    pForm->addRow("Epsilon:", m_pSpnEpsilon);

    m_pChkAmsGrad = new QCheckBox("启用 AMSGrad");  // 20260324 ZJH 汉化
    pForm->addRow("", m_pChkAmsGrad);

    m_pSpnGradClip = new QDoubleSpinBox;
    m_pSpnGradClip->setRange(0.0, 100.0);
    m_pSpnGradClip->setDecimals(2);
    m_pSpnGradClip->setSpecialValueText("禁用");  // 20260324 ZJH 汉化
    pForm->addRow("梯度裁剪范数:", m_pSpnGradClip);  // 20260324 ZJH 汉化

    return pTab;
}

// 20260323 ZJH 创建调度器标签页
QWidget* AdvancedTrainingDialog::createSchedulerTab()
{
    QWidget* pTab = new QWidget;
    QFormLayout* pForm = new QFormLayout(pTab);

    m_pSpnWarmupRatio = new QDoubleSpinBox;
    m_pSpnWarmupRatio->setRange(0.0, 0.5);
    m_pSpnWarmupRatio->setDecimals(3);
    m_pSpnWarmupRatio->setSingleStep(0.01);
    pForm->addRow("预热比例 (Warmup Ratio):", m_pSpnWarmupRatio);  // 20260324 ZJH 汉化

    m_pSpnMinLr = new QDoubleSpinBox;
    m_pSpnMinLr->setRange(0.0, 0.01);
    m_pSpnMinLr->setDecimals(7);
    m_pSpnMinLr->setSingleStep(0.000001);
    pForm->addRow("最小学习率:", m_pSpnMinLr);  // 20260324 ZJH 汉化

    m_pSpnStepSize = new QSpinBox;
    m_pSpnStepSize->setRange(1, 100);
    pForm->addRow("步长 (StepLR):", m_pSpnStepSize);  // 20260324 ZJH 汉化

    m_pSpnStepGamma = new QDoubleSpinBox;
    m_pSpnStepGamma->setRange(0.01, 1.0);
    m_pSpnStepGamma->setDecimals(3);
    pForm->addRow("Gamma (StepLR):", m_pSpnStepGamma);

    return pTab;
}

// 20260323 ZJH 创建正则化标签页
QWidget* AdvancedTrainingDialog::createRegularizationTab()
{
    QWidget* pTab = new QWidget;
    QFormLayout* pForm = new QFormLayout(pTab);

    m_pSpnDropout = new QDoubleSpinBox;
    m_pSpnDropout->setRange(0.0, 0.9);
    m_pSpnDropout->setDecimals(2);
    m_pSpnDropout->setSingleStep(0.05);
    pForm->addRow("Dropout 率:", m_pSpnDropout);  // 20260324 ZJH 汉化

    m_pSpnLabelSmooth = new QDoubleSpinBox;
    m_pSpnLabelSmooth->setRange(0.0, 0.3);
    m_pSpnLabelSmooth->setDecimals(2);
    m_pSpnLabelSmooth->setSingleStep(0.01);
    pForm->addRow("标签平滑:", m_pSpnLabelSmooth);  // 20260324 ZJH 汉化

    m_pChkMixup = new QCheckBox("启用 Mixup");  // 20260324 ZJH 汉化
    pForm->addRow("", m_pChkMixup);

    m_pSpnMixupAlpha = new QDoubleSpinBox;
    m_pSpnMixupAlpha->setRange(0.01, 2.0);
    m_pSpnMixupAlpha->setDecimals(2);
    pForm->addRow("Mixup Alpha:", m_pSpnMixupAlpha);

    // 20260323 ZJH Mixup alpha 仅在启用时可编辑
    connect(m_pChkMixup, &QCheckBox::toggled, m_pSpnMixupAlpha, &QDoubleSpinBox::setEnabled);

    return pTab;
}

// 20260323 ZJH 创建训练策略标签页
QWidget* AdvancedTrainingDialog::createStrategyTab()
{
    QWidget* pTab = new QWidget;
    QFormLayout* pForm = new QFormLayout(pTab);

    m_pSpnAccumSteps = new QSpinBox;
    m_pSpnAccumSteps->setRange(1, 64);
    pForm->addRow("梯度累积步数:", m_pSpnAccumSteps);  // 20260324 ZJH 汉化

    m_pChkFP16 = new QCheckBox("启用 FP16 混合精度");  // 20260324 ZJH 汉化
    pForm->addRow("", m_pChkFP16);

    m_pSpnNumWorkers = new QSpinBox;
    m_pSpnNumWorkers->setRange(0, 16);
    pForm->addRow("数据加载线程数:", m_pSpnNumWorkers);  // 20260324 ZJH 汉化

    m_pChkPinMemory = new QCheckBox("锁页内存 (Pin Memory)");  // 20260324 ZJH 汉化
    pForm->addRow("", m_pChkPinMemory);

    return pTab;
}

// 20260323 ZJH 从配置填充控件
void AdvancedTrainingDialog::loadConfig(const AdvancedTrainingConfig& config)
{
    m_pSpnWeightDecay->setValue(config.dWeightDecay);
    m_pSpnMomentum->setValue(config.dMomentum);
    m_pSpnBeta1->setValue(config.dBeta1);
    m_pSpnBeta2->setValue(config.dBeta2);
    m_pSpnEpsilon->setValue(config.dEpsilon);
    m_pChkAmsGrad->setChecked(config.bAmsGrad);
    m_pSpnGradClip->setValue(config.dGradClipNorm);

    m_pSpnWarmupRatio->setValue(config.dWarmupRatio);
    m_pSpnMinLr->setValue(config.dMinLr);
    m_pSpnStepSize->setValue(config.nStepSize);
    m_pSpnStepGamma->setValue(config.dStepGamma);

    m_pSpnDropout->setValue(config.dDropout);
    m_pSpnLabelSmooth->setValue(config.dLabelSmoothing);
    m_pChkMixup->setChecked(config.bMixup);
    m_pSpnMixupAlpha->setValue(config.dMixupAlpha);
    m_pSpnMixupAlpha->setEnabled(config.bMixup);

    m_pSpnAccumSteps->setValue(config.nAccumulationSteps);
    m_pChkFP16->setChecked(config.bFP16);
    m_pSpnNumWorkers->setValue(config.nNumWorkers);
    m_pChkPinMemory->setChecked(config.bPinMemory);
}

// 20260323 ZJH 获取当前配置
AdvancedTrainingConfig AdvancedTrainingDialog::config() const
{
    AdvancedTrainingConfig cfg;
    cfg.dWeightDecay = m_pSpnWeightDecay->value();
    cfg.dMomentum = m_pSpnMomentum->value();
    cfg.dBeta1 = m_pSpnBeta1->value();
    cfg.dBeta2 = m_pSpnBeta2->value();
    cfg.dEpsilon = m_pSpnEpsilon->value();
    cfg.bAmsGrad = m_pChkAmsGrad->isChecked();
    cfg.dGradClipNorm = m_pSpnGradClip->value();

    cfg.dWarmupRatio = m_pSpnWarmupRatio->value();
    cfg.dMinLr = m_pSpnMinLr->value();
    cfg.nStepSize = m_pSpnStepSize->value();
    cfg.dStepGamma = m_pSpnStepGamma->value();

    cfg.dDropout = m_pSpnDropout->value();
    cfg.dLabelSmoothing = m_pSpnLabelSmooth->value();
    cfg.bMixup = m_pChkMixup->isChecked();
    cfg.dMixupAlpha = m_pSpnMixupAlpha->value();

    cfg.nAccumulationSteps = m_pSpnAccumSteps->value();
    cfg.bFP16 = m_pChkFP16->isChecked();
    cfg.nNumWorkers = m_pSpnNumWorkers->value();
    cfg.bPinMemory = m_pChkPinMemory->isChecked();

    return cfg;
}
