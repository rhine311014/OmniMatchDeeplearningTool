// 20260322 ZJH ExportPage — 模型导出页面
// BasePage 子类，三栏布局：
//   左面板(280px): 导出配置 + 输出目录 + 操作 + 前置检查
//   中央面板: 进度 + 结果卡片 + 格式兼容性表 + 导出历史
//   右面板(220px): 模型信息 + 日志

#pragma once

#include "ui/pages/BasePage.h"  // 20260322 ZJH 页面基类

#include <QComboBox>         // 20260322 ZJH 下拉选择框
#include <QLineEdit>         // 20260322 ZJH 文本输入框
#include <QCheckBox>         // 20260322 ZJH 复选框
#include <QSpinBox>          // 20260322 ZJH 整数微调框
#include <QPushButton>       // 20260322 ZJH 按钮
#include <QLabel>            // 20260322 ZJH 文本标签
#include <QProgressBar>      // 20260322 ZJH 进度条
#include <QTextEdit>         // 20260322 ZJH 日志显示文本框
#include <QTableWidget>      // 20260322 ZJH 表格
#include <QFrame>            // 20260322 ZJH 卡片容器
#include <QTimer>            // 20260322 ZJH 模拟导出计时器
#include <QThread>           // 20260402 ZJH TensorRT 异步构建线程
#include <QProgressBar>      // 20260402 ZJH TensorRT 构建进度条

// 20260322 ZJH 模型导出页面
// 提供模型格式导出配置、导出执行、结果展示和导出历史
class ExportPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，创建三栏布局并初始化全部控件
    // 参数: pParent - 父控件指针
    explicit ExportPage(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~ExportPage() override = default;

    // ===== BasePage 生命周期回调 =====

    // 20260322 ZJH 页面切换到前台时调用
    void onEnter() override;

    // 20260322 ZJH 页面离开前台时调用
    void onLeave() override;

    // 20260324 ZJH 项目加载后调用，刷新模型信息和前置检查（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时调用，清空显示（Template Method 扩展点）
    void onProjectClosedImpl() override;

private slots:
    // 20260322 ZJH 开始导出按钮点击
    void onStartExport();

    // 20260322 ZJH 打开输出目录
    void onOpenOutputDir();

    // 20260322 ZJH 浏览输出目录按钮点击
    void onBrowseOutputDir();

    // 20260322 ZJH 动态批量复选框切换
    // 参数: bChecked - 是否勾选
    void onDynamicBatchToggled(bool bChecked);

    // 20260322 ZJH 模拟导出进度更新（定时器触发）
    void onExportSimTick();

    // 20260402 ZJH 一键 TensorRT 优化按钮点击
    // 流程: 检查 ONNX 文件 → 异步构建 TensorRT → 显示进度 → 完成后显示延迟对比
    void onOptimizeTensorRT();

private:
    // ===== UI 创建辅助方法 =====

    // 20260322 ZJH 创建左面板（导出配置 + 操作 + 前置检查）
    QWidget* createLeftPanel();

    // 20260322 ZJH 创建中央面板（进度 + 结果 + 兼容性表 + 历史表）
    QWidget* createCenterPanel();

    // 20260322 ZJH 创建右面板（模型信息 + 日志）
    QWidget* createRightPanel();

    // 20260322 ZJH 填充格式兼容性表
    void populateCompatibilityTable();

    // 20260322 ZJH 添加导出历史记录
    // 参数: strFormat - 导出格式; strPath - 文件路径; strSize - 文件大小; dTimeS - 耗时秒
    void addHistoryEntry(const QString& strFormat, const QString& strPath,
                         const QString& strSize, double dTimeS);

    // 20260322 ZJH 刷新前置检查标签
    void refreshPreChecks();

    // 20260322 ZJH 刷新模型信息
    void refreshModelInfo();

    // 20260322 ZJH 追加日志消息
    // 参数: strMsg - 日志文本
    void appendLog(const QString& strMsg);

    // ===== 左面板控件 =====

    QComboBox* m_pCboFormat;       // 20260330 ZJH 导出格式下拉框（ONNX/TensorRT/OpenVINO/OMM）
    QComboBox* m_pCboBackend;      // 20260330 ZJH 推理后端下拉框（ONNX Runtime/TensorRT/OpenVINO/原生引擎）
    QCheckBox* m_pChkEncrypt;      // 20260330 ZJH 模型加密复选框
    QLineEdit* m_pEdtPassword;     // 20260330 ZJH 加密密码输入框（最大 24 字符，密码回显模式）
    QLineEdit* m_pEdtModelName;    // 20260322 ZJH 模型名称输入框
    QComboBox* m_pCboPrecision;    // 20260322 ZJH 精度下拉框（FP32/FP16/INT8）
    QCheckBox* m_pChkDynBatch;     // 20260322 ZJH 动态批量复选框
    QSpinBox*  m_pSpnBatchMin;     // 20260322 ZJH 最小批量大小
    QSpinBox*  m_pSpnBatchMax;     // 20260322 ZJH 最大批量大小

    QLineEdit*   m_pEdtOutputDir;  // 20260322 ZJH 输出目录路径
    QPushButton* m_pBtnBrowse;     // 20260322 ZJH 浏览按钮

    QPushButton* m_pBtnStartExport;  // 20260322 ZJH 开始导出按钮
    QPushButton* m_pBtnOpenDir;      // 20260322 ZJH 打开输出目录按钮

    // 20260322 ZJH 前置检查标签（2 项）
    QLabel* m_pLblCheckTrained;  // 20260322 ZJH 模型已训练
    QLabel* m_pLblCheckModel;    // 20260322 ZJH 模型文件存在

    // ===== 中央面板控件 =====

    QProgressBar* m_pProgressBar;  // 20260322 ZJH 进度条
    QLabel*       m_pLblStatus;    // 20260322 ZJH 状态文字
    QLabel*       m_pLblPhase;     // 20260322 ZJH 阶段显示

    // 20260322 ZJH 结果卡片
    QFrame* m_pCardResult;          // 20260322 ZJH 结果卡片容器
    QLabel* m_pLblResultPath;       // 20260322 ZJH 模型路径
    QLabel* m_pLblResultSize;       // 20260322 ZJH 模型大小
    QLabel* m_pLblResultTime;       // 20260322 ZJH 导出耗时

    QTableWidget* m_pTblCompat;    // 20260322 ZJH 格式兼容性表
    QTableWidget* m_pTblHistory;   // 20260322 ZJH 导出历史表

    // ===== 右面板控件 =====

    // 20260322 ZJH 模型信息
    QLabel* m_pLblArch;        // 20260322 ZJH 模型架构
    QLabel* m_pLblParams;      // 20260322 ZJH 参数量
    QLabel* m_pLblFLOPs;       // 20260322 ZJH FLOPs
    QLabel* m_pLblMemory;      // 20260322 ZJH 内存占用

    QTextEdit* m_pTxtLog;      // 20260322 ZJH 日志文本框

    // ===== 导出状态 =====

    QTimer* m_pSimTimer;       // 20260322 ZJH 模拟导出定时器
    int     m_nSimProgress;    // 20260322 ZJH 模拟导出当前进度
    int     m_nSimTotal;       // 20260322 ZJH 模拟导出总数

    // 20260402 ZJH ===== TensorRT 一键优化 =====
    QPushButton* m_pBtnOptimizeTRT = nullptr;   // 20260402 ZJH TensorRT 优化按钮
    QComboBox*   m_pCboTRTPrecision = nullptr;  // 20260402 ZJH TRT 精度选择（FP16/INT8/FP32）
    QLabel*      m_pLblTRTStatus = nullptr;     // 20260402 ZJH TRT 优化状态标签
    QProgressBar* m_pTRTProgressBar = nullptr;  // 20260402 ZJH TRT 构建进度条

    // 20260402 ZJH [OPT-3.9] 一键部署包导出
    QPushButton* m_pBtnExportDeployPkg = nullptr;  // 20260402 ZJH 导出部署包按钮

private slots:
    // 20260402 ZJH [OPT-3.9] 一键导出部署包
    // 打包内容: 模型文件(.onnx) + inference_config.json + README.txt
    void onExportDeployPackage();
};
