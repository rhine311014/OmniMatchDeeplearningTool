// 20260330 ZJH AI 数据合成对话框 — 对标 Keyence AI 图像生成
// 功能: 从少量缺陷样本自动生成大量合成训练数据
// 三种策略: CopyPaste缺陷合成 / 几何+光度增强 / GAN生成
// 通过 EngineBridge::synthesizeData() 调用引擎层 om.engine.data_synthesis 模块
#pragma once

#include <QDialog>           // 20260330 ZJH 对话框基类
#include <QWidget>           // 20260330 ZJH QWidget 父类
#include <QComboBox>         // 20260330 ZJH 策略下拉选择
#include <QSpinBox>          // 20260330 ZJH 整数微调框
#include <QDoubleSpinBox>    // 20260330 ZJH 浮点数微调框
#include <QCheckBox>         // 20260330 ZJH 布尔选项复选框
#include <QPushButton>       // 20260330 ZJH 操作按钮
#include <QProgressBar>      // 20260330 ZJH 合成进度条
#include <QLabel>            // 20260330 ZJH 文本标签
#include <QGroupBox>         // 20260330 ZJH 参数分组框
#include <QVBoxLayout>       // 20260330 ZJH 垂直布局（createXxxGroup 参数）
#include <QTextEdit>         // 20260330 ZJH 日志输出区域
#include <QThread>           // 20260330 ZJH 后台合成工作线程
#include <QImage>            // 20260330 ZJH 预览图像
#include <vector>            // 20260330 ZJH std::vector 图像数据容器
#include <string>            // 20260330 ZJH std::string

// 20260330 ZJH 前向声明引擎桥接结构体
struct SynthesisParams;
struct SynthesisResult;

// 20260330 ZJH AI 数据合成对话框
// 对标 Keyence AI 图像生成功能，从少量缺陷样本自动生成大量合成训练数据
// 三种策略: CopyPaste 缺陷粘贴 / 几何+光度增强 / GAN 生成 / 自动选择
// UI 三栏布局: 左=策略+参数 | 中=预览 | 右=进度+日志
class DataSynthesisDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260330 ZJH 构造函数，创建三栏布局并初始化全部控件
    // 参数: pParent - 父控件指针（通常为 TrainingPage）
    explicit DataSynthesisDialog(QWidget* pParent = nullptr);

    // 20260330 ZJH 析构函数，停止后台工作线程
    ~DataSynthesisDialog() override;

    // 20260330 ZJH 设置源数据（从项目传入）
    // vecNormalImages: 正常样本列表，每张 [C*H*W] 展平的 CHW float [0,1]
    // vecDefectImages: 缺陷样本列表，每张 [C*H*W] 展平的 CHW float [0,1]
    // nC: 通道数（通常为 3）
    // nH: 图像高度
    // nW: 图像宽度
    void setSourceData(
        const std::vector<std::vector<float>>& vecNormalImages,
        const std::vector<std::vector<float>>& vecDefectImages,
        int nC, int nH, int nW);

    // 20260330 ZJH 合成结果数据结构（UI 层使用，与 EngineBridge::SynthesisResult 对应）
    struct SynthResult {
        std::vector<std::vector<float>> vecImages;  // 20260330 ZJH 合成的图像列表 [C*H*W]
        int nOrigCount = 0;    // 20260330 ZJH 原始缺陷样本数
        int nSynthCount = 0;   // 20260330 ZJH 合成生成数
    };

    // 20260330 ZJH 获取合成结果（对话框 Accepted 后调用）
    // 返回: 合成结果结构体
    SynthResult getResult() const;

signals:
    // 20260330 ZJH 合成进度信号（跨线程安全）
    // nPercent: 进度百分比 [0, 100]
    // strStatus: 当前状态描述文本
    void synthesisProgress(int nPercent, const QString& strStatus);

    // 20260330 ZJH 合成完成信号（跨线程安全）
    // bSuccess: 是否成功
    // strMessage: 完成/错误消息
    void synthesisFinished(bool bSuccess, const QString& strMessage);

private slots:
    // 20260330 ZJH 开始合成按钮点击 — 启动后台线程执行合成
    void onStartSynthesis();

    // 20260330 ZJH 预览按钮点击 — 用当前参数生成1张预览
    void onPreview();

    // 20260330 ZJH 策略下拉变更 — 切换显示对应的参数面板
    // nIndex: 策略索引 (0=CopyPaste, 1=增强, 2=GAN, 3=自动)
    void onStrategyChanged(int nIndex);

    // 20260330 ZJH 进度更新槽函数（从信号接收，更新 UI）
    void onProgressUpdated(int nPercent, const QString& strStatus);

    // 20260330 ZJH 合成完成槽函数（从信号接收，关闭对话框）
    void onSynthesisComplete(bool bSuccess, const QString& strMessage);

private:
    // ===== UI 创建辅助方法 =====

    // 20260330 ZJH 创建完整 UI 布局（三栏: 左参数 | 中预览 | 右日志）
    void setupUI();

    // 20260330 ZJH 创建左栏策略选择和参数分组
    QWidget* createLeftPanel();

    // 20260330 ZJH 创建中栏预览区域（前后对比）
    QWidget* createCenterPanel();

    // 20260330 ZJH 创建右栏进度和日志区域
    QWidget* createRightPanel();

    // 20260330 ZJH 创建 CopyPaste 策略参数分组框
    void createCopyPasteGroup(QVBoxLayout* pLayout);

    // 20260330 ZJH 创建几何+光度增强参数分组框
    void createAugmentGroup(QVBoxLayout* pLayout);

    // 20260330 ZJH 创建 GAN 合成参数分组框
    void createGanGroup(QVBoxLayout* pLayout);

    // 20260330 ZJH 将 CHW float [0,1] 图像转为 QImage 用于预览显示
    // vecImage: [C*H*W] 展平的浮点图像数据
    // nC: 通道数
    // nH: 图像高度
    // nW: 图像宽度
    // 返回: 转换后的 QImage（RGB888 或 Grayscale8）
    QImage floatImageToQImage(const std::vector<float>& vecImage,
                              int nC, int nH, int nW) const;

    // 20260330 ZJH 向日志区追加一行文本
    // strMessage: 日志消息文本
    void appendLog(const QString& strMessage);

    // ===== 左面板控件 =====

    QComboBox*      m_pCboStrategy = nullptr;
    QSpinBox*       m_pSpnTargetCount = nullptr;

    QGroupBox*      m_pGrpCopyPaste = nullptr;
    QDoubleSpinBox* m_pSpnScaleMin = nullptr;
    QDoubleSpinBox* m_pSpnScaleMax = nullptr;
    QDoubleSpinBox* m_pSpnRotateRange = nullptr;
    QDoubleSpinBox* m_pSpnBlendAlpha = nullptr;
    QCheckBox*      m_pChkPoisson = nullptr;
    QCheckBox*      m_pChkRandomPos = nullptr;

    QGroupBox*      m_pGrpAugment = nullptr;
    QSpinBox*       m_pSpnVariants = nullptr;
    QCheckBox*      m_pChkElastic = nullptr;
    QCheckBox*      m_pChkPerspective = nullptr;
    QCheckBox*      m_pChkColorTransfer = nullptr;

    QGroupBox*      m_pGrpGAN = nullptr;
    QSpinBox*       m_pSpnGanEpochs = nullptr;
    QLabel*         m_pLblGanStatus = nullptr;

    QLabel*         m_pLblPreviewBefore = nullptr;
    QLabel*         m_pLblPreviewAfter = nullptr;
    QLabel*         m_pLblPreviewInfo = nullptr;

    QProgressBar*   m_pBarProgress = nullptr;
    QLabel*         m_pLblProgress = nullptr;
    QPushButton*    m_pBtnStart = nullptr;
    QPushButton*    m_pBtnPreview = nullptr;
    QPushButton*    m_pBtnCancel = nullptr;
    QTextEdit*      m_pTxtLog = nullptr;

    // ===== 数据 =====

    std::vector<std::vector<float>> m_vecNormalImages;   // 20260330 ZJH 正常样本列表
    std::vector<std::vector<float>> m_vecDefectImages;   // 20260330 ZJH 缺陷样本列表
    int m_nC = 3;    // 20260330 ZJH 通道数
    int m_nH = 0;    // 20260330 ZJH 图像高度
    int m_nW = 0;    // 20260330 ZJH 图像宽度
    SynthResult m_result;       // 20260330 ZJH 合成结果
    bool m_bRunning = false;    // 20260330 ZJH 是否正在合成
};
