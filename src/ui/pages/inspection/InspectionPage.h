// 20260322 ZJH InspectionPage — 数据检查页面
// BasePage 子类，三栏布局：左侧过滤/统计面板、中央缩略图网格、右侧预览/快速标注面板
// 让用户快速浏览已标注图像，按标签/拆分/状态过滤，双击跳转标注页
// 20260324 ZJH 新增推理测试功能：加载模型、导入测试图像、单张/批量推理、结果可视化
#pragma once

#include "ui/pages/BasePage.h"       // 20260322 ZJH 页面基类

#include <QListView>                 // 20260322 ZJH 缩略图列表视图
#include <QStandardItemModel>        // 20260322 ZJH 标准条目模型
#include <QSortFilterProxyModel>     // 20260322 ZJH 排序/过滤代理模型
#include <QLabel>                    // 20260322 ZJH 文本标签
#include <QComboBox>                 // 20260322 ZJH 下拉选择框
#include <QPushButton>               // 20260322 ZJH 按钮
#include <QVBoxLayout>               // 20260322 ZJH 垂直布局
#include <QGroupBox>                 // 20260322 ZJH 分组框
#include <QProgressBar>              // 20260324 ZJH 批量推理进度条
#include <QStackedWidget>            // 20260324 ZJH 检查/推理模式切换
#include <QSpinBox>                  // 20260324 ZJH 模型参数输入
#include <QDoubleSpinBox>            // 20260330 ZJH 后处理参数浮点微调
#include <QCheckBox>                 // 20260330 ZJH 推理增强开关控件
#include <QSlider>                   // 20260401 ZJH mask 透明度滑块
#include <QVector>                   // 20260324 ZJH 推理结果向量
#include <memory>                    // 20260324 ZJH unique_ptr

// 20260322 ZJH 前向声明
class ImageDataset;
class ThumbnailDelegate;
class EngineBridge;                  // 20260324 ZJH 推理引擎桥接层
class ZoomableGraphicsView;          // 20260324 ZJH 可缩放图像查看器
class GradCAMOverlay;                // 20260324 ZJH 缺陷热力图叠加控件

// 20260322 ZJH 检查页过滤代理模型
// 支持按标签/拆分/标注状态过滤
class InspectionFilterProxy : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    explicit InspectionFilterProxy(QObject* pParent = nullptr);

    // 20260322 ZJH 设置标签过滤（-1 = 全部，其他为标签 ID）
    void setLabelFilter(int nLabelId);

    // 20260322 ZJH 设置拆分过滤（-1 = 全部，0=训练, 1=验证, 2=测试, 3=未分配）
    void setSplitFilter(int nSplitType);

    // 20260322 ZJH 设置状态过滤（0=全部, 1=已标注, 2=未标注）
    void setStatusFilter(int nStatus);

protected:
    // 20260322 ZJH 过滤行：同时满足标签+拆分+状态条件才显示
    bool filterAcceptsRow(int nSourceRow, const QModelIndex& sourceParent) const override;

private:
    int m_nLabelFilter = -1;    // 20260322 ZJH 标签过滤值（-1 = 全部）
    int m_nSplitFilter = -1;    // 20260322 ZJH 拆分过滤值（-1 = 全部）
    int m_nStatusFilter = 0;    // 20260322 ZJH 状态过滤值（0 = 全部）
};

// 20260330 ZJH 缺陷区域度量（对标 MVTec dl_anomaly_postprocessing）
// 每个连通域提取面积、质心、外接矩形、概率统计、圆度等几何/统计特征
struct DefectRegion
{
    int nArea = 0;              // 20260330 ZJH 像素面积（连通域像素总数）
    QPointF ptCentroid;         // 20260330 ZJH 质心坐标（区域内所有像素 x/y 均值）
    QRect rectBounding;         // 20260330 ZJH 外接矩形（min/max x/y 构成的 AABB）
    float fMeanProb = 0.0f;     // 20260330 ZJH 区域内平均异常概率
    float fMaxProb = 0.0f;      // 20260330 ZJH 区域内最大异常概率
    float fCircularity = 0.0f;  // 20260330 ZJH 圆度 = 4*pi*area / perimeter²（1.0 为完美圆形）
};

// 20260330 ZJH 检测框结构体（UI 层使用，由 EngineBridge 的 DetectionBox 转换而来）
// 目标检测任务中每个预测目标对应一个 InferDetection
struct InferDetection
{
    QRectF rect;        // 20260330 ZJH 边界框（图像坐标系，左上角+宽高）
    int nClassId;       // 20260330 ZJH 类别 ID
    float fScore;       // 20260330 ZJH 置信度分数 [0, 1]
    QString strLabel;   // 20260330 ZJH 类别名称（从项目标签列表映射）
};

// 20260324 ZJH 单张推理结果结构体
// 存储每张测试图像的推理输出（预测类别、置信度、耗时等）
struct InferResult
{
    QString strImagePath;          // 20260324 ZJH 图像文件路径
    int nPredictedClass = -1;      // 20260324 ZJH 预测类别 ID
    float fConfidence = 0.0f;      // 20260324 ZJH 最高类别置信度
    QVector<float> vecProbs;       // 20260324 ZJH 各类别概率向量
    double dLatencyMs = 0.0;       // 20260324 ZJH 推理耗时（毫秒）
    bool bIsDefect = false;        // 20260324 ZJH 是否为缺陷（异常检测任务用）
    QImage imgDefectMap;           // 20260325 ZJH 二值化缺陷图（白=异物, 黑=背景）
    QVector<DefectRegion> vecDefectRegions;  // 20260330 ZJH 各缺陷连通域的度量信息
    QVector<InferDetection> vecDetections;   // 20260330 ZJH 检测框列表（目标检测任务用）
};

// 20260322 ZJH 数据检查页面
// 20260324 ZJH 双模式：检查模式（原有缩略图网格）+ 推理模式（模型推理测试）
class InspectionPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化三栏布局和所有子控件
    explicit InspectionPage(QWidget* pParent = nullptr);

    // 20260324 ZJH 析构函数（需要定义以正确析构 unique_ptr<EngineBridge>）
    ~InspectionPage() override;

    // ===== 生命周期回调 =====

    // 20260322 ZJH 页面进入前台时刷新显示
    void onEnter() override;

    // 20260322 ZJH 页面离开前台
    void onLeave() override;

    // 20260324 ZJH 项目加载后绑定数据集信号并刷新（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时清空所有内容（Template Method 扩展点）
    void onProjectClosedImpl() override;

protected:
    // 20260324 ZJH 键盘事件：左右方向键导航测试图像
    void keyPressEvent(QKeyEvent* pEvent) override;

    // 20260404 ZJH 事件过滤器：处理左侧标签统计行的点击事件
    bool eventFilter(QObject* pWatched, QEvent* pEvent) override;

private slots:
    // 20260322 ZJH 标签过滤下拉框变化
    void onLabelFilterChanged(int nIndex);

    // 20260322 ZJH 拆分过滤下拉框变化
    void onSplitFilterChanged(int nIndex);

    // 20260322 ZJH 状态过滤下拉框变化
    void onStatusFilterChanged(int nIndex);

    // 20260322 ZJH 缩略图选中变化 → 更新右面板预览
    void onSelectionChanged();

    // 20260322 ZJH 缩略图双击 → 跳转标注页
    void onItemDoubleClicked(const QModelIndex& index);

    // 20260322 ZJH 快速标注按钮点击
    void onQuickLabel();

    // 20260322 ZJH 数据集内容变化时刷新模型
    void onDatasetChanged();

    // 20260322 ZJH 标签列表变化时刷新过滤面板
    void onLabelsChanged();

    // ===== 推理模式槽函数 =====

    // 20260324 ZJH 中央面板模式切换（0=检查模式, 1=推理模式）
    void onModeChanged(int nIndex);

    // 20260330 ZJH 加载模型文件（.omm，兼容旧 .dfm）
    void onLoadModel();

    // 20260324 ZJH 卸载已加载的模型
    void onUnloadModel();

    // 20260324 ZJH 导入测试图像（文件选择器）
    void onImportTestImages();

    // 20260324 ZJH 导入测试文件夹（文件夹选择器）
    void onImportTestFolder();

    // 20260324 ZJH 清除测试图像列表
    void onClearTestImages();

    // 20260324 ZJH 对当前选中图像运行单张推理
    void onRunInference();

    // 20260324 ZJH 对所有测试图像运行批量推理
    void onBatchInference();

    // 20260324 ZJH 导航到上一张测试图像
    void onPrevTestImage();

    // 20260324 ZJH 导航到下一张测试图像
    void onNextTestImage();

    // 20260324 ZJH 测试图像列表选中变化
    void onTestImageSelectionChanged();

private:
    // 20260322 ZJH 创建左侧面板（过滤 + 统计）
    QWidget* createLeftPanel();

    // 20260322 ZJH 创建中央面板（缩略图网格）
    QWidget* createCenterPanel();

    // 20260322 ZJH 创建右侧面板（预览 + 信息 + 快速标注）
    QWidget* createRightPanel();

    // 20260324 ZJH 创建推理模式的中央面板（图像查看器 + 结果叠加）
    QWidget* createInferenceCenterPanel();

    // 20260324 ZJH 创建推理模式的右侧控制面板
    QWidget* createInferenceRightPanel();

    // 20260322 ZJH 刷新缩略图模型（从数据集重建全部条目）
    void refreshModel();

    // 20260322 ZJH 更新统计标签显示
    void updateStatistics();

    // 20260322 ZJH 刷新标签过滤下拉框内容
    void refreshLabelCombo();

    // 20260322 ZJH 更新右面板预览信息
    void updatePreview(const QModelIndex& index);

    // 20260322 ZJH 清空右面板预览信息
    void clearPreview();

    // ===== 推理辅助方法 =====

    // 20260324 ZJH 更新模型状态标签（显示加载状态、架构、类别数等）
    void updateModelStatus();

    // 20260324 ZJH 更新推理按钮的启用状态（依据模型+图像条件）
    void updateInferenceButtons();

    // 20260324 ZJH 对指定图像执行推理并返回结果
    // 参数: strImagePath - 图像文件路径
    // 返回: 推理结果结构体
    InferResult runInferenceOnImage(const QString& strImagePath);

    // 20260324 ZJH 显示推理结果到右面板和图像查看器
    // 参数: result - 推理结果
    void displayInferResult(const InferResult& result);

    // 20260324 ZJH 显示批量推理汇总结果
    void displayBatchSummary();

    // 20260324 ZJH 导航到指定索引的测试图像
    // 参数: nIndex - 测试图像列表中的索引
    void navigateToTestImage(int nIndex);

    // 20260324 ZJH 获取当前项目的类别名称列表（从数据集标签获取）
    // 返回: 类别名称列表（按 labelId 排序）
    QStringList classNameList() const;

    // ===== 左面板组件 =====

    QComboBox* m_pCmbLabelFilter;       // 20260322 ZJH "标签" 过滤下拉框
    QComboBox* m_pCmbSplitFilter;       // 20260322 ZJH "拆分" 过滤下拉框
    QComboBox* m_pCmbStatusFilter;      // 20260322 ZJH "状态" 过滤下拉框

    QLabel* m_pLblFilteredCount;        // 20260322 ZJH 过滤后图像数标签
    QLabel* m_pLblLabelStats;           // 20260322 ZJH 各标签数量列表标签（备用/兼容）
    QVBoxLayout* m_pLabelStatsLayout;   // 20260402 ZJH 标签统计动态容器（对标 Halcon 彩色标签行）

    // ===== 中央面板组件（检查模式） =====

    QListView*              m_pListView;       // 20260322 ZJH 缩略图列表视图
    QStandardItemModel*     m_pModel;          // 20260322 ZJH 底层数据模型
    InspectionFilterProxy*  m_pProxyModel;     // 20260322 ZJH 过滤排序代理模型
    ThumbnailDelegate*      m_pDelegate;       // 20260322 ZJH 缩略图绘制代理

    // ===== 右面板组件（检查模式） =====

    QLabel*      m_pPreviewLabel;        // 20260322 ZJH 选中图像的预览大图
    QLabel*      m_pLblFileName;         // 20260322 ZJH 文件名
    QLabel*      m_pLblDimensions;       // 20260322 ZJH 尺寸
    QLabel*      m_pLblLabelName;        // 20260322 ZJH 标签名
    QLabel*      m_pLblSplitType;        // 20260322 ZJH 拆分类型
    QPushButton* m_pBtnQuickLabel;       // 20260322 ZJH "快速标注" 按钮

    // ===== 模式切换组件 =====

    QComboBox*       m_pCmbMode;               // 20260324 ZJH 模式切换下拉框（检查/推理）
    QStackedWidget*  m_pCenterStack;           // 20260324 ZJH 中央面板栈（检查/推理切换）
    QStackedWidget*  m_pRightStack;            // 20260324 ZJH 右面板栈（检查/推理切换）

    // ===== 推理模式 — 中央面板组件 =====

    ZoomableGraphicsView*  m_pInferView;       // 20260324 ZJH 推理图像查看器
    GradCAMOverlay*        m_pGradCAMOverlay;  // 20260324 ZJH 缺陷热力图叠加控件（预览用）
    QLabel*                m_pLblOverlayResult; // 20260324 ZJH 叠加在图像上的预测标签

    // ===== 推理模式 — 右面板组件 =====

    // 20260324 ZJH 模型加载分组
    QPushButton*  m_pBtnLoadModel;             // 20260324 ZJH "加载模型" 按钮
    QPushButton*  m_pBtnUnloadModel;           // 20260324 ZJH "卸载模型" 按钮
    QLabel*       m_pLblModelStatus;           // 20260324 ZJH 模型状态标签

    // 20260324 ZJH 模型配置（无项目时手动指定）
    QComboBox*    m_pCmbModelArch;             // 20260324 ZJH 模型架构选择
    QSpinBox*     m_pSpnInputSize;             // 20260324 ZJH 输入图像尺寸
    QSpinBox*     m_pSpnNumClasses;            // 20260324 ZJH 类别数量

    // 20260324 ZJH 测试图像分组
    QPushButton*  m_pBtnImportImages;          // 20260324 ZJH "导入测试图像" 按钮
    QPushButton*  m_pBtnImportFolder;          // 20260324 ZJH "导入测试文件夹" 按钮
    QPushButton*  m_pBtnClearImages;           // 20260324 ZJH "清除图像" 按钮
    QLabel*       m_pLblTestImageCount;        // 20260324 ZJH 测试图像数量标签
    QListView*    m_pTestImageList;            // 20260324 ZJH 测试图像文件名列表
    QStandardItemModel* m_pTestImageModel;     // 20260324 ZJH 测试图像列表数据模型

    // 20260324 ZJH 推理控制分组
    QPushButton*  m_pBtnRunInference;          // 20260324 ZJH "运行推理" 按钮（蓝色大按钮）
    QPushButton*  m_pBtnBatchInference;        // 20260324 ZJH "批量推理全部" 按钮
    QProgressBar* m_pProgressBar;              // 20260324 ZJH 批量推理进度条

    // 20260324 ZJH 图像导航分组
    QPushButton*  m_pBtnPrevImage;             // 20260324 ZJH "上一张" 按钮
    QPushButton*  m_pBtnNextImage;             // 20260324 ZJH "下一张" 按钮
    QLabel*       m_pLblImageIndex;            // 20260324 ZJH 当前图像索引标签（如 "3 / 50"）

    // 20260324 ZJH 推理结果分组
    QLabel*       m_pLblPredClass;             // 20260324 ZJH "预测类别: OK / NG"
    QLabel*       m_pLblConfidence;            // 20260324 ZJH "置信度: 98.7%"
    QLabel*       m_pLblLatency;               // 20260324 ZJH "推理耗时: 12.3ms"
    QLabel*       m_pLblClassProbs;            // 20260324 ZJH 各类别概率列表

    // ===== 后处理参数控件（借鉴 MVTec classification_threshold / defect_area_min）=====

    QDoubleSpinBox* m_pSpnConfThreshold = nullptr;  // 20260330 ZJH 缺陷置信度阈值（0.01~0.99）
    QSpinBox*       m_pSpnMinArea = nullptr;         // 20260330 ZJH 最小缺陷面积（像素，0~10000）
    QDoubleSpinBox* m_pSpnProbFloor = nullptr;       // 20260330 ZJH 缺陷图概率下限截断（0.01~0.5）

    // 20260330 ZJH 批量推理批次大小（每批送入引擎的图像张数）
    QSpinBox*       m_pSpnInferBatchSize = nullptr;  // 20260330 ZJH 批量推理张数选择 1~32

    // ===== 推理增强控件（TTA + 滑动窗口）=====

    QCheckBox*      m_pChkEnableTTA = nullptr;       // 20260330 ZJH 测试时增强开关
    QComboBox*      m_pCboTTAMode = nullptr;         // 20260330 ZJH TTA 模式选择（翻转/旋转/多尺度）
    QCheckBox*      m_pChkSlidingWindow = nullptr;   // 20260330 ZJH 滑动窗口大图检测开关
    QSpinBox*       m_pSpnWindowSize = nullptr;      // 20260330 ZJH 滑动窗口尺寸 320~1024
    QDoubleSpinBox* m_pSpnOverlap = nullptr;         // 20260330 ZJH 滑动窗口重叠率 0.0~0.5

    // ===== 检测余裕趋势控件 =====

    QProgressBar*   m_pBarMargin = nullptr;          // 20260330 ZJH 检测余裕趋势进度条（0~100%）
    QLabel*         m_pLblMarginValue = nullptr;     // 20260330 ZJH 检测余裕当前数值标签

    // ===== 分割可视化控件（Phase 1 — 对标海康级可视化）=====

    QSlider*        m_pSldMaskAlpha = nullptr;       // 20260401 ZJH mask 叠加透明度滑块（0~255）
    QLabel*         m_pLblMaskAlphaVal = nullptr;    // 20260401 ZJH 透明度数值标签
    QLabel*         m_pLblDefectMetrics = nullptr;   // 20260401 ZJH 缺陷连通域度量信息面板
    int             m_nMaskAlpha = 120;              // 20260401 ZJH 当前 mask 透明度值

    // ===== GradCAM 可视化开关 =====

    QCheckBox*      m_pChkShowGradCAM = nullptr;     // 20260330 ZJH 注意力热力图显示开关

    // ===== 推理引擎数据 =====

    std::shared_ptr<EngineBridge> m_pInferEngine;  // 20260325 ZJH 推理引擎（shared_ptr 支持后台线程创建后转移）
    bool m_bModelLoaded = false;                    // 20260324 ZJH 模型是否已加载
    QString m_strModelPath;                         // 20260324 ZJH 当前加载的模型文件路径
    int m_nModelInputSize = 224;                    // 20260324 ZJH 模型输入尺寸
    int m_nModelNumClasses = 0;                     // 20260324 ZJH 模型类别数量
    QString m_strModelArchName;                     // 20260324 ZJH 模型架构名称

    // 20260324 ZJH 测试图像列表
    QStringList m_vecTestImagePaths;                // 20260324 ZJH 测试图像文件路径列表
    int m_nCurrentTestIndex = -1;                   // 20260324 ZJH 当前选中的测试图像索引

    // 20260324 ZJH 推理结果缓存
    QVector<InferResult> m_vecResults;              // 20260324 ZJH 所有测试图像的推理结果
    bool m_bBatchCompleted = false;                 // 20260324 ZJH 批量推理是否已完成

    // ===== 数据层引用 =====

    ImageDataset* m_pDataset = nullptr;  // 20260322 ZJH 当前项目的数据集（弱引用）
};
