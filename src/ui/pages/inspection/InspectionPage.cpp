// 20260322 ZJH InspectionPage 实现
// 三栏布局数据检查页面：过滤/统计 + 缩略图网格 + 预览/快速标注
// 20260324 ZJH 新增推理测试功能：加载模型、导入测试图像、单张/批量推理、结果可视化

#include "ui/pages/inspection/InspectionPage.h"  // 20260322 ZJH 类声明
#include "ui/widgets/ThumbnailDelegate.h"         // 20260322 ZJH 缩略图绘制代理
#include "ui/widgets/ZoomableGraphicsView.h"      // 20260324 ZJH 可缩放图像查看器
#include "ui/widgets/GradCAMOverlay.h"            // 20260324 ZJH 缺陷热力图叠加控件
#include "engine/bridge/EngineBridge.h"           // 20260324 ZJH 推理引擎桥接层
#include "core/data/ImageDataset.h"               // 20260322 ZJH 数据集管理
#include "core/data/ImageEntry.h"                 // 20260322 ZJH 图像条目
#include "core/data/LabelInfo.h"                  // 20260322 ZJH 标签信息
#include "core/DLTypes.h"                         // 20260322 ZJH SplitType 枚举
#include "core/project/Project.h"                 // 20260322 ZJH 项目数据
#include "app/Application.h"                      // 20260322 ZJH 全局事件总线

#include <QHBoxLayout>          // 20260322 ZJH 水平布局
#include <QVBoxLayout>          // 20260322 ZJH 垂直布局
#include <QGroupBox>            // 20260322 ZJH 分组框
#include <QScrollArea>          // 20260322 ZJH 滚动区域
#include <QStandardItem>        // 20260322 ZJH 标准条目
#include <QPixmap>              // 20260322 ZJH 位图
#include <QFileInfo>            // 20260322 ZJH 文件信息
#include <QMessageBox>          // 20260322 ZJH 消息对话框
#include <QFileDialog>          // 20260324 ZJH 文件/文件夹选择对话框
#include <QDir>                 // 20260324 ZJH 目录遍历
#include <QKeyEvent>            // 20260324 ZJH 键盘事件
#include <QMouseEvent>          // 20260404 ZJH 鼠标事件（标签行点击过滤）
#include <QApplication>         // 20260324 ZJH processEvents 保持 UI 响应
#include <QImage>               // 20260324 ZJH 图像加载和预处理
#include <QtConcurrent>         // 20260325 ZJH 后台线程加载模型
#include <QFutureWatcher>       // 20260325 ZJH 监听后台任务完成
#include <QProgressDialog>      // 20260325 ZJH 加载进度对话框
#include <QPainter>             // 20260330 ZJH 检测框绘制
#include <QFormLayout>          // 20260330 ZJH 后处理参数表单
#include <QDoubleSpinBox>      // 20260330 ZJH 后处理参数浮点微调
#include <chrono>               // 20260324 ZJH 推理计时
#include <vector>               // 20260324 ZJH 推理输入数据
#include <algorithm>            // 20260329 ZJH nth_element 百分位阈值
#include <queue>                // 20260329 ZJH BFS 连通域面积过滤
#include <iostream>             // 20260326 ZJH std::cerr 异常日志输出
#include <cmath>               // 20260330 ZJH M_PI 圆度计算、sqrt 等数学函数
#ifndef M_PI
#define M_PI 3.14159265358979323846  // 20260330 ZJH MSVC 默认不定义 M_PI
#endif

// ===================================================================
// InspectionFilterProxy 实现
// ===================================================================

// 20260322 ZJH 构造函数
InspectionFilterProxy::InspectionFilterProxy(QObject* pParent)
    : QSortFilterProxyModel(pParent)
{
}

// 20260322 ZJH 设置标签过滤
void InspectionFilterProxy::setLabelFilter(int nLabelId)
{
    m_nLabelFilter = nLabelId;  // 20260322 ZJH 保存标签过滤值
    beginFilterChange(); endFilterChange();  // 20260323 ZJH Qt6 推荐替代 invalidateFilter
}

// 20260322 ZJH 设置拆分过滤
void InspectionFilterProxy::setSplitFilter(int nSplitType)
{
    m_nSplitFilter = nSplitType;  // 20260322 ZJH 保存拆分过滤值
    beginFilterChange(); endFilterChange();  // 20260323 ZJH Qt6 推荐替代 invalidateFilter
}

// 20260322 ZJH 设置状态过滤
void InspectionFilterProxy::setStatusFilter(int nStatus)
{
    m_nStatusFilter = nStatus;  // 20260322 ZJH 保存状态过滤值
    beginFilterChange(); endFilterChange();  // 20260323 ZJH Qt6 推荐替代 invalidateFilter
}

// 20260322 ZJH 过滤行逻辑：标签 AND 拆分 AND 状态
bool InspectionFilterProxy::filterAcceptsRow(int nSourceRow,
                                              const QModelIndex& sourceParent) const
{
    QModelIndex idx = sourceModel()->index(nSourceRow, 0, sourceParent);

    // 20260322 ZJH 1. 标签过滤（-1 = 全部，不过滤）
    if (m_nLabelFilter != -1) {
        int nLabelId = idx.data(LabelIdRole).toInt();  // 20260322 ZJH 获取标签 ID
        if (nLabelId != m_nLabelFilter) {
            return false;  // 20260322 ZJH 标签不匹配，过滤掉
        }
    }

    // 20260322 ZJH 2. 拆分过滤（-1 = 全部，不过滤）
    if (m_nSplitFilter != -1) {
        int nSplitType = idx.data(SplitTypeRole).toInt();  // 20260322 ZJH 获取拆分类型
        if (nSplitType != m_nSplitFilter) {
            return false;  // 20260322 ZJH 拆分类型不匹配，过滤掉
        }
    }

    // 20260322 ZJH 3. 状态过滤（0=全部, 1=已标注, 2=未标注）
    if (m_nStatusFilter != 0) {
        int nLabelId = idx.data(LabelIdRole).toInt();  // 20260322 ZJH 获取标签 ID
        bool bLabeled = (nLabelId >= 0);                // 20260322 ZJH 标签 ID >= 0 表示已标注
        if (m_nStatusFilter == 1 && !bLabeled) {
            return false;  // 20260322 ZJH 要求已标注，但图像未标注
        }
        if (m_nStatusFilter == 2 && bLabeled) {
            return false;  // 20260322 ZJH 要求未标注，但图像已标注
        }
    }

    return true;  // 20260322 ZJH 通过所有过滤条件
}

// ===================================================================
// InspectionPage 实现
// ===================================================================

// 20260322 ZJH 构造函数，初始化三栏布局
// 20260324 ZJH 新增推理模式组件初始化
InspectionPage::InspectionPage(QWidget* pParent)
    : BasePage(pParent)                        // 20260406 ZJH 初始化页面基类
    , m_pCmbLabelFilter(nullptr)               // 20260406 ZJH 标签过滤下拉框初始为空
    , m_pCmbSplitFilter(nullptr)               // 20260406 ZJH 拆分过滤下拉框初始为空
    , m_pCmbStatusFilter(nullptr)              // 20260406 ZJH 状态过滤下拉框初始为空
    , m_pLblFilteredCount(nullptr)             // 20260406 ZJH 过滤后图像数标签初始为空
    , m_pLblLabelStats(nullptr)                // 20260406 ZJH 标签统计标签初始为空
    , m_pLabelStatsLayout(nullptr)             // 20260402 ZJH 标签统计动态布局初始为空
    , m_pListView(nullptr)                     // 20260406 ZJH 缩略图列表视图初始为空
    , m_pModel(nullptr)                        // 20260406 ZJH 底层数据模型初始为空
    , m_pProxyModel(nullptr)                   // 20260406 ZJH 过滤排序代理模型初始为空
    , m_pDelegate(nullptr)                     // 20260406 ZJH 缩略图绘制代理初始为空
    , m_pPreviewLabel(nullptr)                 // 20260406 ZJH 预览大图标签初始为空
    , m_pLblFileName(nullptr)                  // 20260406 ZJH 文件名标签初始为空
    , m_pLblDimensions(nullptr)                // 20260406 ZJH 尺寸标签初始为空
    , m_pLblLabelName(nullptr)                 // 20260406 ZJH 标签名标签初始为空
    , m_pLblSplitType(nullptr)                 // 20260406 ZJH 拆分类型标签初始为空
    , m_pBtnQuickLabel(nullptr)                // 20260406 ZJH 快速标注按钮初始为空
    , m_pCmbMode(nullptr)                      // 20260406 ZJH 模式切换下拉框初始为空
    , m_pCenterStack(nullptr)                  // 20260406 ZJH 中央面板栈初始为空
    , m_pRightStack(nullptr)                   // 20260406 ZJH 右面板栈初始为空
    , m_pInferView(nullptr)                    // 20260406 ZJH 推理图像查看器初始为空
    , m_pGradCAMOverlay(nullptr)               // 20260406 ZJH 缺陷热力图叠加控件初始为空
    , m_pLblOverlayResult(nullptr)             // 20260406 ZJH 叠加预测标签初始为空
    , m_pBtnLoadModel(nullptr)                 // 20260406 ZJH 加载模型按钮初始为空
    , m_pBtnUnloadModel(nullptr)               // 20260406 ZJH 卸载模型按钮初始为空
    , m_pLblModelStatus(nullptr)               // 20260406 ZJH 模型状态标签初始为空
    , m_pCmbModelArch(nullptr)                 // 20260406 ZJH 模型架构选择初始为空
    , m_pSpnInputSize(nullptr)                 // 20260406 ZJH 输入尺寸微调框初始为空
    , m_pSpnNumClasses(nullptr)                // 20260406 ZJH 类别数量微调框初始为空
    , m_pBtnImportImages(nullptr)              // 20260406 ZJH 导入测试图像按钮初始为空
    , m_pBtnImportFolder(nullptr)              // 20260406 ZJH 导入测试文件夹按钮初始为空
    , m_pBtnClearImages(nullptr)               // 20260406 ZJH 清除图像按钮初始为空
    , m_pLblTestImageCount(nullptr)            // 20260406 ZJH 测试图像数量标签初始为空
    , m_pTestImageList(nullptr)                // 20260406 ZJH 测试图像列表初始为空
    , m_pTestImageModel(nullptr)               // 20260406 ZJH 测试图像数据模型初始为空
    , m_pBtnRunInference(nullptr)              // 20260406 ZJH 运行推理按钮初始为空
    , m_pBtnBatchInference(nullptr)            // 20260406 ZJH 批量推理按钮初始为空
    , m_pProgressBar(nullptr)                  // 20260406 ZJH 批量推理进度条初始为空
    , m_pBtnPrevImage(nullptr)                 // 20260406 ZJH 上一张按钮初始为空
    , m_pBtnNextImage(nullptr)                 // 20260406 ZJH 下一张按钮初始为空
    , m_pLblImageIndex(nullptr)                // 20260406 ZJH 图像索引标签初始为空
    , m_pLblPredClass(nullptr)                 // 20260406 ZJH 预测类别标签初始为空
    , m_pLblConfidence(nullptr)                // 20260406 ZJH 置信度标签初始为空
    , m_pLblLatency(nullptr)                   // 20260406 ZJH 推理耗时标签初始为空
    , m_pLblClassProbs(nullptr)                // 20260406 ZJH 类别概率标签初始为空
    , m_pDataset(nullptr)                      // 20260406 ZJH 数据集弱引用初始为空
{
    // 20260322 ZJH 创建左侧面板（检查模式的过滤+统计面板，两种模式共用）
    QWidget* pLeftPanel = createLeftPanel();

    // 20260324 ZJH 创建中央面板栈（检查模式 + 推理模式）
    m_pCenterStack = new QStackedWidget(this);
    QWidget* pInspectCenter = createCenterPanel();      // 20260322 ZJH 检查模式缩略图网格
    QWidget* pInferCenter   = createInferenceCenterPanel();  // 20260324 ZJH 推理模式图像查看器
    m_pCenterStack->addWidget(pInspectCenter);  // 20260324 ZJH 索引 0 = 检查模式
    m_pCenterStack->addWidget(pInferCenter);    // 20260324 ZJH 索引 1 = 推理模式

    // 20260324 ZJH 在中央面板顶部添加模式切换下拉框
    QWidget* pCenterWrapper = new QWidget(this);
    QVBoxLayout* pCenterWrapLayout = new QVBoxLayout(pCenterWrapper);
    pCenterWrapLayout->setContentsMargins(0, 0, 0, 0);
    pCenterWrapLayout->setSpacing(0);

    // 20260324 ZJH 模式切换工具栏
    QWidget* pModeBar = new QWidget(pCenterWrapper);
    pModeBar->setFixedHeight(36);
    pModeBar->setStyleSheet(QStringLiteral(
        "QWidget { background-color: #161b22; border-bottom: 1px solid #2a2d35; }"));
    QHBoxLayout* pModeBarLayout = new QHBoxLayout(pModeBar);
    pModeBarLayout->setContentsMargins(8, 4, 8, 4);

    // 20260324 ZJH 模式切换下拉框
    QLabel* pLblModeTitle = new QLabel(QStringLiteral("模式:"), pModeBar);
    pLblModeTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 12px; font-weight: bold; border: none; background: transparent; }"));
    pModeBarLayout->addWidget(pLblModeTitle);

    m_pCmbMode = new QComboBox(pModeBar);
    m_pCmbMode->addItem(QStringLiteral("数据检查"));  // 20260324 ZJH 索引 0 = 检查模式
    m_pCmbMode->addItem(QStringLiteral("推理测试"));  // 20260324 ZJH 索引 1 = 推理模式
    m_pCmbMode->setStyleSheet(QStringLiteral(
        "QComboBox {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 2px 8px;"
        "  min-width: 120px;"
        "}"
        "QComboBox::drop-down { border: none; width: 20px; }"
        "QComboBox QAbstractItemView {"
        "  background-color: #1e2230; color: #e2e8f0;"
        "  selection-background-color: #2563eb; border: 1px solid #2a2d35;"
        "}"
    ));
    pModeBarLayout->addWidget(m_pCmbMode);
    pModeBarLayout->addStretch(1);  // 20260324 ZJH 右侧弹性空间

    // 20260324 ZJH 连接模式切换信号
    connect(m_pCmbMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InspectionPage::onModeChanged);

    pCenterWrapLayout->addWidget(pModeBar);
    pCenterWrapLayout->addWidget(m_pCenterStack, 1);

    // 20260324 ZJH 创建右面板栈（检查模式 + 推理模式）
    m_pRightStack = new QStackedWidget(this);
    QWidget* pInspectRight = createRightPanel();              // 20260322 ZJH 检查模式预览面板
    QWidget* pInferRight   = createInferenceRightPanel();     // 20260324 ZJH 推理模式控制面板
    m_pRightStack->addWidget(pInspectRight);  // 20260324 ZJH 索引 0 = 检查模式
    m_pRightStack->addWidget(pInferRight);    // 20260324 ZJH 索引 1 = 推理模式

    // 20260322 ZJH 设置三栏布局
    setLeftPanelWidth(280);
    setRightPanelWidth(280);  // 20260324 ZJH 推理模式需要更宽的右面板
    setupThreeColumnLayout(pLeftPanel, pCenterWrapper, m_pRightStack);
}

// 20260324 ZJH 析构函数（需要显式定义以正确析构 unique_ptr<EngineBridge>）
InspectionPage::~InspectionPage() = default;

// 20260322 ZJH 页面进入前台时刷新显示
void InspectionPage::onEnter()
{
    // 20260322 ZJH 刷新统计和模型（可能在其他页面修改了数据）
    if (m_pDataset) {
        refreshModel();
        updateStatistics();
    }

    // 20260324 ZJH 推理模式下更新按钮状态
    updateInferenceButtons();
}

// 20260322 ZJH 页面离开前台（无特殊操作）
void InspectionPage::onLeave()
{
}

// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void InspectionPage::onProjectLoadedImpl()
{
    if (!m_pProject) {
        return;  // 20260322 ZJH 无效项目
    }

    // 20260322 ZJH 获取数据集引用
    m_pDataset = m_pProject->dataset();

    if (m_pDataset) {
        // 20260322 ZJH 绑定数据集变更信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::dataChanged,
                this, &InspectionPage::onDatasetChanged);

        // 20260322 ZJH 绑定图像添加信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::imagesAdded,
                this, &InspectionPage::onDatasetChanged);

        // 20260322 ZJH 绑定图像移除信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::imagesRemoved,
                this, &InspectionPage::onDatasetChanged);

        // 20260322 ZJH 绑定标签变更信号 → 刷新过滤面板
        connect(m_pDataset, &ImageDataset::labelsChanged,
                this, &InspectionPage::onLabelsChanged);

        // 20260322 ZJH 绑定拆分变更信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::splitChanged,
                this, &InspectionPage::onDatasetChanged);
    }

    // 20260322 ZJH 刷新标签过滤下拉框、缩略图模型和统计
    refreshLabelCombo();

    // 20260404 ZJH 检查页默认只显示已标注图像（过滤掉未标注原图）
    // 对标 Halcon DL Tool 检查页: 默认展示已标注数据，让用户聚焦于标注质量审核
    if (m_pCmbStatusFilter) {
        m_pCmbStatusFilter->setCurrentIndex(1);  // 20260404 ZJH 索引 1 = "已标注"
    }

    refreshModel();
    updateStatistics();

    // 20260324 ZJH 项目加载后自动填充模型配置（从 TrainingConfig 读取）
    if (m_pProject) {
        const TrainingConfig& config = m_pProject->trainingConfig();
        // 20260324 ZJH 设置输入尺寸
        if (m_pSpnInputSize) {
            m_pSpnInputSize->setValue(config.nInputSize);
        }
        // 20260329 ZJH 设置类别数（分割任务 = 标签数 + 1 背景，与训练一致）
        if (m_pSpnNumClasses && m_pDataset) {
            int nLabelCount = m_pDataset->labels().size();
            if (nLabelCount > 0) {
                // 20260329 ZJH 分割任务(语义/实例)需要 +1 背景类，与 TrainingSession 一致
                auto eTask = m_pProject->taskType();
                bool bIsSeg = (eTask == om::TaskType::SemanticSegmentation
                            || eTask == om::TaskType::InstanceSegmentation);
                int nNumClasses = bIsSeg ? (nLabelCount + 1) : nLabelCount;
                if (nNumClasses < 2) nNumClasses = 2;
                m_pSpnNumClasses->setValue(nNumClasses);
            }
        }
        // 20260329 ZJH 设置模型架构选择（含诊断日志）
        if (m_pCmbModelArch) {
            QString strArchName = om::modelArchitectureToString(config.eArchitecture);
            std::cerr << "[DIAG-ARCH] eArchitecture=" << static_cast<int>(config.eArchitecture)
                      << " name=\"" << strArchName.toStdString() << "\""
                      << " comboCount=" << m_pCmbModelArch->count() << std::endl;
            int nIdx = m_pCmbModelArch->findText(strArchName);
            std::cerr << "[DIAG-ARCH] findText idx=" << nIdx << std::endl;
            if (nIdx >= 0) {
                m_pCmbModelArch->setCurrentIndex(nIdx);
            }
        }

        // 20260329 ZJH ===== 自动加载最佳模型（训练完成后无需手动选择）=====
        QString strBestModel = m_pProject->bestModelPath();
        if (!strBestModel.isEmpty() && QFile::exists(strBestModel) && !m_bModelLoaded) {
            // 20260329 ZJH 用项目配置中的正确参数创建模型并加载权重
            auto pEngine = std::make_shared<EngineBridge>();
            QString strModelType = om::modelArchitectureToString(config.eArchitecture);
            strModelType.remove(QChar('-'));
            strModelType.remove(QChar(' '));
            int nInputSize = config.nInputSize;
            auto eTask = m_pProject->taskType();
            bool bIsSeg = (eTask == om::TaskType::SemanticSegmentation
                        || eTask == om::TaskType::InstanceSegmentation);
            int nLabelCount = m_pDataset ? m_pDataset->labels().size() : 2;
            int nNumClasses = bIsSeg ? (nLabelCount + 1) : nLabelCount;
            if (nNumClasses < 2) nNumClasses = 2;

            std::string stdType = strModelType.toStdString();
            std::string stdPath = strBestModel.toStdString();

            std::cerr << "[AUTO-LOAD] type=" << stdType
                      << " inputSize=" << nInputSize
                      << " nNumClasses=" << nNumClasses
                      << " isSeg=" << bIsSeg
                      << " path=" << stdPath << std::endl;

            // 20260330 ZJH try-catch 保护自动加载，防止模型创建/加载崩溃导致整个应用闪退
            try {
                bool bCreateOk = pEngine->createModel(stdType, nInputSize, nNumClasses);
                std::cerr << "[AUTO-LOAD] createModel=" << bCreateOk
                          << " params=" << pEngine->totalParameters() << std::endl;

                if (bCreateOk) {
                    bool bLoadOk = pEngine->loadModel(stdPath);
                    std::cerr << "[AUTO-LOAD] loadModel=" << bLoadOk << std::endl;
                    if (bLoadOk) {
                        m_pInferEngine = pEngine;
                        m_bModelLoaded = true;
                        if (m_pLblModelStatus) m_pLblModelStatus->setText(
                            QStringLiteral("✓ 已自动加载: %1").arg(QFileInfo(strBestModel).fileName()));
                        if (m_pBtnUnloadModel) m_pBtnUnloadModel->setEnabled(true);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[AUTO-LOAD] EXCEPTION: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[AUTO-LOAD] UNKNOWN EXCEPTION — skipping model load" << std::endl;
            }
        }
    }
}

// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void InspectionPage::onProjectClosedImpl()
{
    // 20260322 ZJH 断开数据集信号
    if (m_pDataset) {
        disconnect(m_pDataset, nullptr, this, nullptr);
    }
    m_pDataset = nullptr;

    // 20260322 ZJH 清空模型
    if (m_pModel) {
        m_pModel->clear();
    }

    // 20260322 ZJH 重置过滤下拉框
    refreshLabelCombo();
    updateStatistics();
    clearPreview();
}

// 20260324 ZJH 键盘事件：左右方向键导航测试图像
void InspectionPage::keyPressEvent(QKeyEvent* pEvent)
{
    // 20260324 ZJH 仅在推理模式下处理方向键
    if (m_pCmbMode && m_pCmbMode->currentIndex() == 1) {
        if (pEvent->key() == Qt::Key_Left) {
            onPrevTestImage();  // 20260324 ZJH 左键 → 上一张
            return;
        } else if (pEvent->key() == Qt::Key_Right) {
            onNextTestImage();  // 20260324 ZJH 右键 → 下一张
            return;
        }
    }
    BasePage::keyPressEvent(pEvent);  // 20260324 ZJH 其他键交给基类处理
}

// 20260404 ZJH 事件过滤器：处理左侧标签统计行的鼠标点击
// 点击标签行 → 标签过滤下拉框切换到对应标签，缩略图网格自动过滤
// 点击 "未标注的图像" 行 → 状态过滤切到"未标注"，标签过滤切回"全部"
// 再次点击同一标签行 → 取消过滤，恢复"全部"
bool InspectionPage::eventFilter(QObject* pWatched, QEvent* pEvent)
{
    // 20260404 ZJH 仅处理鼠标左键释放事件
    if (pEvent->type() == QEvent::MouseButtonRelease) {
        QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(pEvent);
        if (pMouseEvent->button() == Qt::LeftButton) {
            // 20260404 ZJH 从 widget 的 property 读取标签 ID
            QVariant varLabelId = pWatched->property("labelFilterId");
            if (varLabelId.isValid()) {
                int nLabelId = varLabelId.toInt();

                if (nLabelId == -2) {
                    // 20260404 ZJH 点击 "未标注的图像" 行
                    // 状态过滤设为"未标注"(索引 2)，标签过滤设为"全部"(索引 0)
                    if (m_pCmbStatusFilter) {
                        // 20260404 ZJH 再次点击时取消过滤（toggle 行为）
                        if (m_pCmbStatusFilter->currentIndex() == 2) {
                            m_pCmbStatusFilter->setCurrentIndex(0);  // 20260404 ZJH 恢复全部
                        } else {
                            m_pCmbStatusFilter->setCurrentIndex(2);  // 20260404 ZJH 切到未标注
                        }
                    }
                    if (m_pCmbLabelFilter) {
                        m_pCmbLabelFilter->setCurrentIndex(0);  // 20260404 ZJH 标签过滤回全部
                    }
                } else {
                    // 20260404 ZJH 点击具体标签行 → 在标签下拉框中找到并选中
                    if (m_pCmbLabelFilter) {
                        // 20260404 ZJH 查找下拉框中该标签 ID 的索引
                        int nTargetIndex = -1;
                        for (int i = 0; i < m_pCmbLabelFilter->count(); ++i) {
                            if (m_pCmbLabelFilter->itemData(i).toInt() == nLabelId) {
                                nTargetIndex = i;
                                break;
                            }
                        }
                        if (nTargetIndex >= 0) {
                            // 20260404 ZJH 再次点击同一标签 → toggle 回全部
                            if (m_pCmbLabelFilter->currentIndex() == nTargetIndex) {
                                m_pCmbLabelFilter->setCurrentIndex(0);  // 20260404 ZJH 恢复全部
                            } else {
                                m_pCmbLabelFilter->setCurrentIndex(nTargetIndex);  // 20260404 ZJH 过滤到该标签
                            }
                        }
                    }
                    // 20260404 ZJH 状态过滤恢复为"全部"（因为标签过滤已限定）
                    if (m_pCmbStatusFilter) {
                        m_pCmbStatusFilter->setCurrentIndex(0);
                    }
                }

                return true;  // 20260404 ZJH 事件已处理
            }
        }
    }

    return BasePage::eventFilter(pWatched, pEvent);  // 20260404 ZJH 其他事件交给基类
}

// ===== 检查模式槽函数 =====

// 20260322 ZJH 标签过滤下拉框变化
void InspectionPage::onLabelFilterChanged(int nIndex)
{
    if (!m_pProxyModel || !m_pCmbLabelFilter) {
        return;
    }

    // 20260322 ZJH 获取下拉框中存储的标签 ID 数据（-1 = 全部）
    int nLabelId = m_pCmbLabelFilter->itemData(nIndex).toInt();
    m_pProxyModel->setLabelFilter(nLabelId);  // 20260322 ZJH 应用过滤

    // 20260322 ZJH 更新过滤后的图像数量统计
    updateStatistics();
}

// 20260322 ZJH 拆分过滤下拉框变化
void InspectionPage::onSplitFilterChanged(int nIndex)
{
    if (!m_pProxyModel || !m_pCmbSplitFilter) {
        return;
    }

    // 20260322 ZJH 获取下拉框中存储的拆分类型数据（-1 = 全部）
    int nSplitType = m_pCmbSplitFilter->itemData(nIndex).toInt();
    m_pProxyModel->setSplitFilter(nSplitType);  // 20260322 ZJH 应用过滤

    // 20260322 ZJH 更新过滤后的图像数量统计
    updateStatistics();
}

// 20260322 ZJH 状态过滤下拉框变化
void InspectionPage::onStatusFilterChanged(int nIndex)
{
    if (!m_pProxyModel) {
        return;
    }

    // 20260322 ZJH 索引直接对应状态值：0=全部, 1=已标注, 2=未标注
    m_pProxyModel->setStatusFilter(nIndex);  // 20260322 ZJH 应用过滤

    // 20260322 ZJH 更新过滤后的图像数量统计
    updateStatistics();
}

// 20260322 ZJH 缩略图选中变化 → 更新右面板预览
void InspectionPage::onSelectionChanged()
{
    if (!m_pListView) {
        return;
    }

    // 20260322 ZJH 获取当前选中的索引列表
    QModelIndexList vecSelected = m_pListView->selectionModel()->selectedIndexes();
    if (vecSelected.isEmpty()) {
        clearPreview();  // 20260322 ZJH 无选中时清空预览
        return;
    }

    // 20260322 ZJH 取第一个选中的索引来更新预览
    QModelIndex proxyIdx = vecSelected.first();
    QModelIndex srcIdx = m_pProxyModel->mapToSource(proxyIdx);
    updatePreview(srcIdx);  // 20260322 ZJH 更新预览面板
}

// 20260322 ZJH 双击跳转标注页
void InspectionPage::onItemDoubleClicked(const QModelIndex& index)
{
    if (!index.isValid()) {
        return;
    }

    // 20260322 ZJH 从代理模型映射回源模型获取 UUID
    QModelIndex srcIdx = m_pProxyModel->mapToSource(index);
    QString strUuid = srcIdx.data(UuidRole).toString();

    if (!strUuid.isEmpty()) {
        // 20260404 ZJH 获取标注 UUID（实例视图下，双击跳转后在图像页自动选中该标注）
        QString strAnnotationUuid;
        int nAnnotationIdx = srcIdx.data(AnnotationIdxRole).toInt();
        int nImageIdx = srcIdx.data(ImageIndexRole).toInt();
        if (nAnnotationIdx >= 0 && m_pDataset) {
            const QVector<ImageEntry>& vecImages = m_pDataset->images();
            if (nImageIdx >= 0 && nImageIdx < vecImages.size()) {
                const auto& vecAnns = vecImages[nImageIdx].vecAnnotations;
                if (nAnnotationIdx < vecAnns.size()) {
                    strAnnotationUuid = vecAnns[nAnnotationIdx].strUuid;  // 20260404 ZJH 标注 UUID
                }
            }
        }

        // 20260404 ZJH 通过 notify 发射信号，携带标注 UUID
        Application::instance()->notifyOpenImage(strUuid, strAnnotationUuid);
        // 20260324 ZJH 同时请求切换到图像标注页（PageIndex::Image = 2）
        Application::instance()->notifyNavigateToPage(2);
    }
}

// 20260322 ZJH 快速标注按钮点击
void InspectionPage::onQuickLabel()
{
    if (!m_pDataset || !m_pListView || !m_pCmbLabelFilter) {
        return;
    }

    // 20260322 ZJH 获取当前标签过滤下拉框中选中的标签 ID
    int nTargetLabelId = m_pCmbLabelFilter->currentData().toInt();
    if (nTargetLabelId < 0) {
        QMessageBox::information(this,
            QStringLiteral("快速标注"),
            QStringLiteral("请先在左侧过滤面板选择一个具体的标签。"));
        return;  // 20260322 ZJH "全部" 时不能标注
    }

    // 20260322 ZJH 获取选中的图像索引列表
    QModelIndexList vecSelected = m_pListView->selectionModel()->selectedIndexes();
    if (vecSelected.isEmpty()) {
        QMessageBox::information(this,
            QStringLiteral("快速标注"),
            QStringLiteral("请先选中要标注的图像。"));
        return;
    }

    // 20260322 ZJH 批量分配标签
    int nCount = 0;  // 20260322 ZJH 计数成功标注的图像数
    for (const QModelIndex& proxyIdx : vecSelected) {
        QModelIndex srcIdx = m_pProxyModel->mapToSource(proxyIdx);
        QString strUuid = srcIdx.data(UuidRole).toString();
        if (!strUuid.isEmpty()) {
            ImageEntry* pEntry = m_pDataset->findImage(strUuid);
            if (pEntry) {
                pEntry->nLabelId = nTargetLabelId;  // 20260322 ZJH 分配标签 ID
                ++nCount;
            }
        }
    }

    if (nCount > 0) {
        // 20260322 ZJH 通知数据集变更
        emit m_pDataset->dataChanged();
    }
}

// 20260322 ZJH 数据集变化时刷新
void InspectionPage::onDatasetChanged()
{
    refreshModel();
    updateStatistics();
}

// 20260322 ZJH 标签列表变化时刷新过滤面板
void InspectionPage::onLabelsChanged()
{
    refreshLabelCombo();
    refreshModel();
    updateStatistics();
}

// ===== 推理模式槽函数 =====

// 20260324 ZJH 中央面板模式切换（0=检查模式, 1=推理模式）
void InspectionPage::onModeChanged(int nIndex)
{
    if (m_pCenterStack) {
        m_pCenterStack->setCurrentIndex(nIndex);  // 20260324 ZJH 切换中央面板
    }
    if (m_pRightStack) {
        m_pRightStack->setCurrentIndex(nIndex);  // 20260324 ZJH 切换右面板
    }
    // 20260324 ZJH 推理模式下更新按钮状态
    if (nIndex == 1) {
        updateInferenceButtons();
    }
}

// 20260330 ZJH 加载模型文件（.omm，兼容旧 .dfm）
// 20260325 ZJH 重写为异步加载：后台线程创建模型+加载权重，避免 UI 卡死
void InspectionPage::onLoadModel()
{
    // 20260324 ZJH 打开文件选择对话框
    QString strPath = QFileDialog::getOpenFileName(this,
        QStringLiteral("选择模型文件"), QString(),
        QStringLiteral("OmniMatch 模型 (*.omm);;旧格式模型 (*.dfm);;所有文件 (*)"));
    if (strPath.isEmpty()) {
        return;  // 20260324 ZJH 用户取消选择
    }

    // 20260324 ZJH 获取模型配置参数
    QString strArchName = m_pCmbModelArch ? m_pCmbModelArch->currentText() : QStringLiteral("ResNet-18");
    int nInputSize = m_pSpnInputSize ? m_pSpnInputSize->value() : 224;
    int nNumClasses = m_pSpnNumClasses ? m_pSpnNumClasses->value() : 2;

    // 20260324 ZJH 将显示名称转换为内部模型类型字符串
    QString strModelType = strArchName;
    strModelType.remove(QChar('-'));  // 20260324 ZJH "ResNet-18" → "ResNet18"
    strModelType.remove(QChar(' '));  // 20260324 ZJH 去除空格

    // 20260325 ZJH 显示模态进度对话框，防止加载大模型时 UI 无响应
    QProgressDialog* pProgress = new QProgressDialog(
        QStringLiteral("正在加载模型，请稍候..."), QString(), 0, 0, this);
    pProgress->setWindowTitle(QStringLiteral("加载模型"));
    pProgress->setWindowModality(Qt::WindowModal);
    pProgress->setMinimumDuration(0);  // 20260325 ZJH 立即显示
    pProgress->setCancelButton(nullptr);  // 20260325 ZJH 不可取消（模型创建不可中断）
    pProgress->show();
    QApplication::processEvents();  // 20260325 ZJH 强制刷新确保对话框立即显示

    // 20260325 ZJH 准备后台线程参数（值拷贝，避免跨线程引用局部变量）
    auto pEngine = std::make_shared<EngineBridge>();
    std::string stdModelType = strModelType.toStdString();
    std::string stdPath = strPath.toStdString();

    // 20260325 ZJH 后台线程执行模型创建 + 权重加载
    // 20260326 ZJH 添加 try/catch 防止异常逃逸到 QFuture 导致 UI 线程 unhandled exception 闪退
    // 返回值: 0=成功, 1=架构创建失败, 2=权重加载失败, 3=未知异常
    QFutureWatcher<int>* pWatcher = new QFutureWatcher<int>(this);
    QFuture<int> future = QtConcurrent::run(
        [pEngine, stdModelType, nInputSize, nNumClasses, stdPath]() -> int {
            try {
                if (!pEngine->createModel(stdModelType, nInputSize, nNumClasses)) {
                    return 1;  // 20260325 ZJH 架构创建失败
                }
                if (!pEngine->loadModel(stdPath)) {
                    return 2;  // 20260325 ZJH 权重加载失败
                }
                return 0;  // 20260325 ZJH 全部成功
            } catch (const std::exception& e) {
                // 20260326 ZJH 捕获模型构造函数中可能的 bad_alloc、runtime_error 等异常
                std::cerr << "[InspectionPage] model load exception: " << e.what() << std::endl;
                return 3;  // 20260326 ZJH 异常导致失败
            } catch (...) {
                // 20260326 ZJH 捕获其他未知异常，防止逃逸
                std::cerr << "[InspectionPage] model load unknown exception" << std::endl;
                return 3;  // 20260326 ZJH 未知异常
            }
        });

    // 20260325 ZJH 后台任务完成后回到 UI 线程处理结果
    connect(pWatcher, &QFutureWatcher<int>::finished, this,
        [this, pWatcher, pProgress, pEngine,
         strArchName, strPath, nInputSize, nNumClasses]() {
        // 20260325 ZJH 关闭进度对话框
        pProgress->close();
        pProgress->deleteLater();

        // 20260326 ZJH try/catch 防止 QFuture::result() 重新抛出存储的异常导致闪退
        int nResult = 3;  // 20260326 ZJH 默认异常
        try {
            nResult = pWatcher->result();
        } catch (const std::exception& e) {
            std::cerr << "[InspectionPage] result exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[InspectionPage] result unknown exception" << std::endl;
        }
        pWatcher->deleteLater();

        if (nResult == 1) {
            QMessageBox::warning(this,
                QStringLiteral("加载失败"),
                QStringLiteral("无法创建模型架构 \"%1\"。\n请确认架构、输入尺寸和类别数设置正确。")
                    .arg(strArchName));
            return;
        }
        if (nResult == 2) {
            QMessageBox::warning(this,
                QStringLiteral("加载失败"),
                QStringLiteral("无法加载模型权重文件:\n%1\n\n请确认文件格式正确且与所选架构匹配。")
                    .arg(strPath));
            return;
        }
        if (nResult == 3) {
            // 20260326 ZJH 捕获到异常时显示友好提示
            QMessageBox::critical(this,
                QStringLiteral("加载异常"),
                QStringLiteral("加载模型时发生异常。\n可能原因：内存不足、文件损坏或模型格式不兼容。"));
            return;
        }

        // 20260325 ZJH 加载成功：直接转移 shared_ptr 所有权
        m_pInferEngine = pEngine;

        // 20260324 ZJH 记录模型信息
        m_bModelLoaded = true;              // 20260406 ZJH 标记模型已成功加载
        m_strModelPath = strPath;            // 20260406 ZJH 保存模型文件路径
        m_nModelInputSize = nInputSize;      // 20260406 ZJH 保存模型输入尺寸
        m_nModelNumClasses = nNumClasses;    // 20260406 ZJH 保存模型类别数量
        m_strModelArchName = strArchName;    // 20260406 ZJH 保存模型架构名称

        // 20260324 ZJH 清空之前的推理结果
        m_vecResults.clear();
        m_bBatchCompleted = false;

        // 20260324 ZJH 更新界面状态
        updateModelStatus();
        updateInferenceButtons();
    });

    pWatcher->setFuture(future);
}

// 20260324 ZJH 卸载已加载的模型
void InspectionPage::onUnloadModel()
{
    // 20260324 ZJH 释放引擎资源
    m_pInferEngine.reset();            // 20260406 ZJH 释放推理引擎 shared_ptr
    m_bModelLoaded = false;            // 20260406 ZJH 标记模型未加载
    m_strModelPath.clear();            // 20260406 ZJH 清空模型文件路径
    m_strModelArchName.clear();        // 20260406 ZJH 清空模型架构名称
    m_nModelNumClasses = 0;            // 20260406 ZJH 重置类别数量

    // 20260324 ZJH 清空推理结果
    m_vecResults.clear();
    m_bBatchCompleted = false;

    // 20260324 ZJH 更新界面状态
    updateModelStatus();
    updateInferenceButtons();

    // 20260324 ZJH 清空推理结果显示
    if (m_pLblPredClass)  m_pLblPredClass->setText(QStringLiteral("--"));
    if (m_pLblConfidence) m_pLblConfidence->setText(QStringLiteral("--"));
    if (m_pLblLatency)    m_pLblLatency->setText(QStringLiteral("--"));
    if (m_pLblClassProbs) m_pLblClassProbs->setText(QStringLiteral(""));
}

// 20260324 ZJH 导入测试图像（文件选择器）
void InspectionPage::onImportTestImages()
{
    // 20260324 ZJH 打开多文件选择对话框
    QStringList vecFiles = QFileDialog::getOpenFileNames(this,
        QStringLiteral("选择测试图像"), QString(),
        QStringLiteral("图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"));
    if (vecFiles.isEmpty()) {
        return;  // 20260324 ZJH 用户取消选择
    }

    // 20260324 ZJH 添加到测试图像列表（去重）
    for (const QString& strFile : vecFiles) {
        if (!m_vecTestImagePaths.contains(strFile)) {
            m_vecTestImagePaths.append(strFile);
        }
    }

    // 20260324 ZJH 刷新测试图像列表模型
    if (m_pTestImageModel) {
        m_pTestImageModel->clear();
        for (const QString& strPath : m_vecTestImagePaths) {
            QStandardItem* pItem = new QStandardItem(QFileInfo(strPath).fileName());
            pItem->setData(strPath, Qt::UserRole);  // 20260324 ZJH 存储完整路径
            pItem->setForeground(QColor("#e2e8f0"));  // 20260324 ZJH 浅色文字
            m_pTestImageModel->appendRow(pItem);
        }
    }

    // 20260324 ZJH 更新测试图像计数标签
    if (m_pLblTestImageCount) {
        m_pLblTestImageCount->setText(QStringLiteral("共 %1 张测试图像")
            .arg(m_vecTestImagePaths.size()));
    }

    // 20260324 ZJH 清空之前的推理结果（图像列表已变化）
    m_vecResults.clear();
    m_bBatchCompleted = false;

    // 20260324 ZJH 更新按钮状态
    updateInferenceButtons();
}

// 20260324 ZJH 导入测试文件夹（文件夹选择器）
void InspectionPage::onImportTestFolder()
{
    // 20260324 ZJH 打开文件夹选择对话框
    QString strDir = QFileDialog::getExistingDirectory(this,
        QStringLiteral("选择测试图像文件夹"), QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (strDir.isEmpty()) {
        return;  // 20260324 ZJH 用户取消选择
    }

    // 20260324 ZJH 遍历文件夹中的图像文件
    QDir dir(strDir);
    QStringList vecNameFilters = {
        QStringLiteral("*.png"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
        QStringLiteral("*.bmp"), QStringLiteral("*.tif"), QStringLiteral("*.tiff")
    };
    QFileInfoList vecEntries = dir.entryInfoList(vecNameFilters,
        QDir::Files | QDir::NoDotAndDotDot, QDir::Name);

    // 20260324 ZJH 添加到测试图像列表（去重）
    int nAddedCount = 0;  // 20260324 ZJH 记录新增图像数
    for (const QFileInfo& info : vecEntries) {
        QString strPath = info.absoluteFilePath();
        if (!m_vecTestImagePaths.contains(strPath)) {
            m_vecTestImagePaths.append(strPath);
            ++nAddedCount;
        }
    }

    // 20260324 ZJH 刷新测试图像列表模型
    if (m_pTestImageModel) {
        m_pTestImageModel->clear();
        for (const QString& strPath : m_vecTestImagePaths) {
            QStandardItem* pItem = new QStandardItem(QFileInfo(strPath).fileName());
            pItem->setData(strPath, Qt::UserRole);  // 20260324 ZJH 存储完整路径
            pItem->setForeground(QColor("#e2e8f0"));  // 20260324 ZJH 浅色文字
            m_pTestImageModel->appendRow(pItem);
        }
    }

    // 20260324 ZJH 更新测试图像计数标签
    if (m_pLblTestImageCount) {
        m_pLblTestImageCount->setText(QStringLiteral("共 %1 张测试图像")
            .arg(m_vecTestImagePaths.size()));
    }

    // 20260324 ZJH 清空之前的推理结果
    m_vecResults.clear();
    m_bBatchCompleted = false;

    // 20260324 ZJH 更新按钮状态
    updateInferenceButtons();

    // 20260324 ZJH 提示用户导入结果
    if (nAddedCount == 0 && vecEntries.isEmpty()) {
        QMessageBox::information(this,
            QStringLiteral("导入测试文件夹"),
            QStringLiteral("所选文件夹中未找到图像文件。"));
    }
}

// 20260324 ZJH 清除测试图像列表
void InspectionPage::onClearTestImages()
{
    m_vecTestImagePaths.clear();
    m_nCurrentTestIndex = -1;
    m_vecResults.clear();
    m_bBatchCompleted = false;

    // 20260324 ZJH 清空列表模型
    if (m_pTestImageModel) {
        m_pTestImageModel->clear();
    }

    // 20260324 ZJH 更新计数标签
    if (m_pLblTestImageCount) {
        m_pLblTestImageCount->setText(QStringLiteral("共 0 张测试图像"));
    }

    // 20260324 ZJH 清空图像查看器
    if (m_pInferView) {
        m_pInferView->clearImage();
    }

    // 20260324 ZJH 清空结果显示标签，恢复到初始占位状态
    if (m_pLblPredClass)  m_pLblPredClass->setText(QStringLiteral("--"));    // 20260406 ZJH 预测类别重置
    if (m_pLblConfidence) m_pLblConfidence->setText(QStringLiteral("--"));   // 20260406 ZJH 置信度重置
    if (m_pLblLatency)    m_pLblLatency->setText(QStringLiteral("--"));      // 20260406 ZJH 推理耗时重置
    if (m_pLblClassProbs) m_pLblClassProbs->setText(QStringLiteral(""));     // 20260406 ZJH 类别概率清空
    if (m_pLblImageIndex) m_pLblImageIndex->setText(QStringLiteral("-- / --"));  // 20260406 ZJH 图像索引重置
    if (m_pLblOverlayResult) m_pLblOverlayResult->clear();

    // 20260324 ZJH 更新按钮状态
    updateInferenceButtons();
}

// 20260324 ZJH 对当前选中图像运行单张推理
void InspectionPage::onRunInference()
{
    if (!m_bModelLoaded || m_nCurrentTestIndex < 0 ||
        m_nCurrentTestIndex >= m_vecTestImagePaths.size()) {
        return;
    }

    // 20260324 ZJH 获取当前图像路径
    QString strImagePath = m_vecTestImagePaths[m_nCurrentTestIndex];

    // 20260324 ZJH 执行推理
    InferResult result = runInferenceOnImage(strImagePath);

    // 20260324 ZJH 存储结果到缓存（替换已有结果或新增）
    if (m_nCurrentTestIndex < m_vecResults.size()) {
        m_vecResults[m_nCurrentTestIndex] = result;
    } else {
        // 20260324 ZJH 扩展结果向量到当前索引
        m_vecResults.resize(m_nCurrentTestIndex + 1);
        m_vecResults[m_nCurrentTestIndex] = result;
    }

    // 20260324 ZJH 显示推理结果
    displayInferResult(result);

    // 20260324 ZJH 更新测试图像列表中的颜色标记
    if (m_pTestImageModel && m_nCurrentTestIndex < m_pTestImageModel->rowCount()) {
        QStandardItem* pItem = m_pTestImageModel->item(m_nCurrentTestIndex);
        if (pItem) {
            // 20260324 ZJH 绿色=OK（置信度高且非缺陷），红色=NG（缺陷/低置信度）
            if (result.bIsDefect || result.fConfidence < 0.5f) {
                pItem->setForeground(QColor("#ef4444"));  // 20260324 ZJH 红色 = NG
            } else {
                pItem->setForeground(QColor("#22c55e"));  // 20260324 ZJH 绿色 = OK
            }
        }
    }
}

// 20260324 ZJH 对所有测试图像运行批量推理
void InspectionPage::onBatchInference()
{
    if (!m_bModelLoaded || m_vecTestImagePaths.isEmpty()) {
        return;
    }

    // 20260324 ZJH 设置进度条范围
    if (m_pProgressBar) {
        m_pProgressBar->setRange(0, m_vecTestImagePaths.size());
        m_pProgressBar->setValue(0);
        m_pProgressBar->setVisible(true);
    }

    // 20260324 ZJH 清空之前的结果
    m_vecResults.clear();
    m_vecResults.resize(m_vecTestImagePaths.size());

    // 20260324 ZJH 禁用推理按钮防止重复点击
    if (m_pBtnRunInference)   m_pBtnRunInference->setEnabled(false);
    if (m_pBtnBatchInference) m_pBtnBatchInference->setEnabled(false);

    // 20260324 ZJH 逐张推理
    int nOkCount = 0;    // 20260324 ZJH OK 计数
    int nNgCount = 0;    // 20260324 ZJH NG 计数
    double dTotalMs = 0;  // 20260324 ZJH 总推理耗时

    for (int i = 0; i < m_vecTestImagePaths.size(); ++i) {
        // 20260324 ZJH 执行推理
        InferResult result = runInferenceOnImage(m_vecTestImagePaths[i]);
        m_vecResults[i] = result;

        // 20260324 ZJH 累计统计
        dTotalMs += result.dLatencyMs;
        if (result.bIsDefect || result.fConfidence < 0.5f) {
            ++nNgCount;
        } else {
            ++nOkCount;
        }

        // 20260324 ZJH 更新测试图像列表颜色
        if (m_pTestImageModel && i < m_pTestImageModel->rowCount()) {
            QStandardItem* pItem = m_pTestImageModel->item(i);
            if (pItem) {
                if (result.bIsDefect || result.fConfidence < 0.5f) {
                    pItem->setForeground(QColor("#ef4444"));  // 20260324 ZJH 红色
                } else {
                    pItem->setForeground(QColor("#22c55e"));  // 20260324 ZJH 绿色
                }
            }
        }

        // 20260324 ZJH 更新进度条
        if (m_pProgressBar) {
            m_pProgressBar->setValue(i + 1);
        }

        // 20260324 ZJH 处理事件保持 UI 响应
        QApplication::processEvents();
    }

    // 20260324 ZJH 标记批量推理完成
    m_bBatchCompleted = true;

    // 20260324 ZJH 恢复按钮状态
    updateInferenceButtons();

    // 20260324 ZJH 显示批量推理汇总
    displayBatchSummary();

    // 20260324 ZJH 导航到第一张图像显示结果
    if (!m_vecTestImagePaths.isEmpty()) {
        navigateToTestImage(0);
    }
}

// 20260324 ZJH 导航到上一张测试图像
void InspectionPage::onPrevTestImage()
{
    if (m_vecTestImagePaths.isEmpty()) {
        return;
    }

    // 20260324 ZJH 计算上一张索引（循环到末尾）
    int nNewIndex = m_nCurrentTestIndex - 1;
    if (nNewIndex < 0) {
        nNewIndex = m_vecTestImagePaths.size() - 1;
    }
    navigateToTestImage(nNewIndex);
}

// 20260324 ZJH 导航到下一张测试图像
void InspectionPage::onNextTestImage()
{
    if (m_vecTestImagePaths.isEmpty()) {
        return;
    }

    // 20260324 ZJH 计算下一张索引（循环到开头）
    int nNewIndex = m_nCurrentTestIndex + 1;
    if (nNewIndex >= m_vecTestImagePaths.size()) {
        nNewIndex = 0;
    }
    navigateToTestImage(nNewIndex);
}

// 20260324 ZJH 测试图像列表选中变化
void InspectionPage::onTestImageSelectionChanged()
{
    if (!m_pTestImageList) {
        return;
    }

    // 20260324 ZJH 获取选中的索引
    QModelIndexList vecSelected = m_pTestImageList->selectionModel()->selectedIndexes();
    if (vecSelected.isEmpty()) {
        return;
    }

    // 20260324 ZJH 导航到选中的图像
    int nIndex = vecSelected.first().row();
    navigateToTestImage(nIndex);
}

// ===== 私有方法 — 面板创建 =====

// 20260322 ZJH 创建左侧面板
QWidget* InspectionPage::createLeftPanel()
{
    QWidget* pPanel = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(8, 8, 8, 8);
    pLayout->setSpacing(8);

    // 20260322 ZJH 通用样式表（分组框和标签）
    QString strGroupStyle = QStringLiteral(
        "QGroupBox {"
        "  font-weight: bold;"
        "  color: #94a3b8;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  margin-top: 12px;"
        "  padding-top: 16px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px;"
        "}"
    );

    // 20260322 ZJH 下拉框/标签通用样式
    QString strComboStyle = QStringLiteral(
        "QComboBox {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 4px 8px;"
        "  min-height: 24px;"
        "}"
        "QComboBox::drop-down {"
        "  border: none;"
        "  width: 20px;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  selection-background-color: #2563eb;"
        "  border: 1px solid #2a2d35;"
        "}"
    );

    // ===== 过滤分组 =====

    // 20260322 ZJH 创建 "过滤" 分组框
    QGroupBox* pFilterGroup = new QGroupBox(QStringLiteral("过滤"), pPanel);
    pFilterGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pFilterLayout = new QVBoxLayout(pFilterGroup);
    pFilterLayout->setSpacing(6);

    // 20260322 ZJH "标签" 过滤下拉框
    QLabel* pLblLabel = new QLabel(QStringLiteral("标签:"), pFilterGroup);
    pLblLabel->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 11px; border: none; }"));
    pFilterLayout->addWidget(pLblLabel);

    m_pCmbLabelFilter = new QComboBox(pFilterGroup);
    m_pCmbLabelFilter->setStyleSheet(strComboStyle);
    m_pCmbLabelFilter->addItem(QStringLiteral("全部"), -1);  // 20260322 ZJH 默认全部
    pFilterLayout->addWidget(m_pCmbLabelFilter);

    // 20260322 ZJH 连接标签过滤变化信号
    connect(m_pCmbLabelFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InspectionPage::onLabelFilterChanged);

    // 20260322 ZJH "拆分" 过滤下拉框
    QLabel* pLblSplit = new QLabel(QStringLiteral("拆分:"), pFilterGroup);
    pLblSplit->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 11px; border: none; }"));
    pFilterLayout->addWidget(pLblSplit);

    m_pCmbSplitFilter = new QComboBox(pFilterGroup);
    m_pCmbSplitFilter->setStyleSheet(strComboStyle);
    m_pCmbSplitFilter->addItem(QStringLiteral("全部"),  -1);
    m_pCmbSplitFilter->addItem(QStringLiteral("训练"),  static_cast<int>(om::SplitType::Train));
    m_pCmbSplitFilter->addItem(QStringLiteral("验证"),  static_cast<int>(om::SplitType::Validation));
    m_pCmbSplitFilter->addItem(QStringLiteral("测试"),  static_cast<int>(om::SplitType::Test));
    m_pCmbSplitFilter->addItem(QStringLiteral("未分配"), static_cast<int>(om::SplitType::Unassigned));
    pFilterLayout->addWidget(m_pCmbSplitFilter);

    // 20260322 ZJH 连接拆分过滤变化信号
    connect(m_pCmbSplitFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InspectionPage::onSplitFilterChanged);

    // 20260322 ZJH "状态" 过滤下拉框
    QLabel* pLblStatus = new QLabel(QStringLiteral("状态:"), pFilterGroup);
    pLblStatus->setStyleSheet(QStringLiteral("QLabel { color: #94a3b8; font-size: 11px; border: none; }"));
    pFilterLayout->addWidget(pLblStatus);

    m_pCmbStatusFilter = new QComboBox(pFilterGroup);
    m_pCmbStatusFilter->setStyleSheet(strComboStyle);
    m_pCmbStatusFilter->addItem(QStringLiteral("全部"));
    m_pCmbStatusFilter->addItem(QStringLiteral("已标注"));
    m_pCmbStatusFilter->addItem(QStringLiteral("未标注"));
    pFilterLayout->addWidget(m_pCmbStatusFilter);

    // 20260322 ZJH 连接状态过滤变化信号
    connect(m_pCmbStatusFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InspectionPage::onStatusFilterChanged);

    pLayout->addWidget(pFilterGroup);

    // ===== 统计分组 =====

    // 20260322 ZJH 创建 "统计" 分组框
    QGroupBox* pStatsGroup = new QGroupBox(QStringLiteral("统计"), pPanel);
    pStatsGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pStatsLayout = new QVBoxLayout(pStatsGroup);
    pStatsLayout->setSpacing(4);

    // 20260322 ZJH 过滤后的图像数标签
    m_pLblFilteredCount = new QLabel(QStringLiteral("当前显示: 0 张图像"), pStatsGroup);
    m_pLblFilteredCount->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 13px; font-weight: bold; border: none; }"));
    pStatsLayout->addWidget(m_pLblFilteredCount);

    // 20260402 ZJH 标签统计动态容器（对标 Halcon DL Tool 彩色标签行）
    // 每个标签行: [彩色ID方块] [标签名] [右对齐计数]
    // 替代原来的纯文本 QLabel，支持动态颜色和分任务单位
    m_pLblLabelStats = nullptr;  // 20260402 ZJH 不再使用旧 QLabel
    QWidget* pLabelStatsContainer = new QWidget(pStatsGroup);  // 20260402 ZJH 容器 widget
    m_pLabelStatsLayout = new QVBoxLayout(pLabelStatsContainer);  // 20260402 ZJH 动态布局
    m_pLabelStatsLayout->setContentsMargins(0, 0, 0, 0);  // 20260402 ZJH 无边距
    m_pLabelStatsLayout->setSpacing(3);  // 20260402 ZJH 行间距 3px（紧凑）
    pStatsLayout->addWidget(pLabelStatsContainer);

    pLayout->addWidget(pStatsGroup);

    // 20260322 ZJH 弹性空间填充底部
    pLayout->addStretch(1);

    return pPanel;
}

// 20260322 ZJH 创建中央面板（检查模式缩略图网格）
QWidget* InspectionPage::createCenterPanel()
{
    QWidget* pPanel = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    // 20260322 ZJH 创建底层数据模型
    m_pModel = new QStandardItemModel(this);

    // 20260322 ZJH 创建过滤代理模型
    m_pProxyModel = new InspectionFilterProxy(this);
    m_pProxyModel->setSourceModel(m_pModel);

    // 20260322 ZJH 创建缩略图绘制代理（复用 ThumbnailDelegate）
    m_pDelegate = new ThumbnailDelegate(this);
    m_pDelegate->setThumbnailSize(140);  // 20260322 ZJH 默认缩略图大小 140px

    // 20260322 ZJH 创建 QListView（图标模式）
    m_pListView = new QListView(pPanel);
    m_pListView->setViewMode(QListView::IconMode);          // 20260322 ZJH 图标模式
    m_pListView->setResizeMode(QListView::Adjust);          // 20260322 ZJH 自适应布局
    m_pListView->setSelectionMode(QListView::ExtendedSelection);  // 20260322 ZJH 多选模式
    m_pListView->setMovement(QListView::Static);            // 20260322 ZJH 禁止拖动排列
    m_pListView->setUniformItemSizes(true);                 // 20260322 ZJH 统一尺寸优化
    m_pListView->setSpacing(4);                             // 20260322 ZJH 项间距
    m_pListView->setModel(m_pProxyModel);                   // 20260322 ZJH 设置代理模型
    m_pListView->setItemDelegate(m_pDelegate);              // 20260322 ZJH 设置绘制代理

    // 20260322 ZJH 设置网格大小
    int nGridW = 140 + 12;        // 20260322 ZJH 缩略图宽度 + 左右边距
    int nGridH = 140 + 12 + 20;   // 20260322 ZJH 缩略图高度 + 边距 + 文字高度
    m_pListView->setGridSize(QSize(nGridW, nGridH));

    // 20260322 ZJH 暗色背景样式
    m_pListView->setStyleSheet(QStringLiteral(
        "QListView {"
        "  background-color: #0d1117;"
        "  border: none;"
        "}"
    ));

    // 20260322 ZJH 连接选中变化信号
    connect(m_pListView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &InspectionPage::onSelectionChanged);

    // 20260322 ZJH 连接双击信号
    connect(m_pListView, &QListView::doubleClicked,
            this, &InspectionPage::onItemDoubleClicked);

    pLayout->addWidget(m_pListView, 1);  // 20260322 ZJH 填满中央区域

    return pPanel;
}

// 20260322 ZJH 创建右侧面板（检查模式预览+信息+快速标注）
QWidget* InspectionPage::createRightPanel()
{
    QWidget* pPanel = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(8, 8, 8, 8);
    pLayout->setSpacing(8);

    // 20260322 ZJH 通用标签样式
    QString strLabelStyle = QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 11px; border: none; }");
    QString strValueStyle = QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 12px; border: none; }");

    // 20260322 ZJH 分组框样式
    QString strGroupStyle = QStringLiteral(
        "QGroupBox {"
        "  font-weight: bold;"
        "  color: #94a3b8;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  margin-top: 12px;"
        "  padding-top: 16px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px;"
        "}"
    );

    // ===== 预览大图 =====

    // 20260322 ZJH 选中图像的预览大图
    m_pPreviewLabel = new QLabel(pPanel);
    m_pPreviewLabel->setFixedSize(200, 200);  // 20260322 ZJH 固定 200x200 预览区域
    m_pPreviewLabel->setAlignment(Qt::AlignCenter);
    m_pPreviewLabel->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  background-color: #1a1d24;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  color: #475569;"
        "  font-size: 11px;"
        "}"
    ));
    m_pPreviewLabel->setText(QStringLiteral("选中图像\n预览"));
    pLayout->addWidget(m_pPreviewLabel, 0, Qt::AlignCenter);

    // ===== 图像信息分组 =====

    QGroupBox* pInfoGroup = new QGroupBox(QStringLiteral("图像信息"), pPanel);
    pInfoGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pInfoLayout = new QVBoxLayout(pInfoGroup);
    pInfoLayout->setSpacing(4);

    // 20260322 ZJH 文件名
    QLabel* pLblFileTitle = new QLabel(QStringLiteral("文件名:"), pInfoGroup);
    pLblFileTitle->setStyleSheet(strLabelStyle);
    pInfoLayout->addWidget(pLblFileTitle);

    m_pLblFileName = new QLabel(QStringLiteral("--"), pInfoGroup);
    m_pLblFileName->setStyleSheet(strValueStyle);
    m_pLblFileName->setWordWrap(true);  // 20260322 ZJH 长文件名自动换行
    pInfoLayout->addWidget(m_pLblFileName);

    // 20260322 ZJH 尺寸
    QLabel* pLblDimTitle = new QLabel(QStringLiteral("尺寸:"), pInfoGroup);
    pLblDimTitle->setStyleSheet(strLabelStyle);
    pInfoLayout->addWidget(pLblDimTitle);

    m_pLblDimensions = new QLabel(QStringLiteral("--"), pInfoGroup);
    m_pLblDimensions->setStyleSheet(strValueStyle);
    pInfoLayout->addWidget(m_pLblDimensions);

    // 20260322 ZJH 标签
    QLabel* pLblLabelTitle = new QLabel(QStringLiteral("标签:"), pInfoGroup);
    pLblLabelTitle->setStyleSheet(strLabelStyle);
    pInfoLayout->addWidget(pLblLabelTitle);

    m_pLblLabelName = new QLabel(QStringLiteral("--"), pInfoGroup);
    m_pLblLabelName->setStyleSheet(strValueStyle);
    pInfoLayout->addWidget(m_pLblLabelName);

    // 20260322 ZJH 拆分
    QLabel* pLblSplitTitle = new QLabel(QStringLiteral("拆分:"), pInfoGroup);
    pLblSplitTitle->setStyleSheet(strLabelStyle);
    pInfoLayout->addWidget(pLblSplitTitle);

    m_pLblSplitType = new QLabel(QStringLiteral("--"), pInfoGroup);
    m_pLblSplitType->setStyleSheet(strValueStyle);
    pInfoLayout->addWidget(m_pLblSplitType);

    pLayout->addWidget(pInfoGroup);

    // ===== 快速标注按钮 =====

    m_pBtnQuickLabel = new QPushButton(QStringLiteral("快速标注"), pPanel);
    m_pBtnQuickLabel->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #2563eb;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 8px 16px;"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3b82f6;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #1d4ed8;"
        "}"
    ));
    m_pBtnQuickLabel->setToolTip(QStringLiteral("一键为选中图像分配当前标签过滤中选择的标签"));
    pLayout->addWidget(m_pBtnQuickLabel);

    // 20260322 ZJH 连接快速标注按钮信号
    connect(m_pBtnQuickLabel, &QPushButton::clicked,
            this, &InspectionPage::onQuickLabel);

    // 20260322 ZJH 弹性空间填充底部
    pLayout->addStretch(1);

    return pPanel;
}

// 20260324 ZJH 创建推理模式的中央面板（图像查看器 + 结果叠加）
QWidget* InspectionPage::createInferenceCenterPanel()
{
    QWidget* pPanel = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    // 20260324 ZJH 创建可缩放图像查看器
    m_pInferView = new ZoomableGraphicsView(pPanel);
    m_pInferView->setStyleSheet(QStringLiteral(
        "QGraphicsView { background-color: #0d1117; border: none; }"));

    // 20260324 ZJH 创建叠加在图像上的预测结果标签（左上角半透明）
    m_pLblOverlayResult = new QLabel(m_pInferView);
    m_pLblOverlayResult->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  background-color: rgba(0, 0, 0, 180);"
        "  color: #e2e8f0;"
        "  font-size: 14px;"
        "  font-weight: bold;"
        "  padding: 8px 12px;"
        "  border-radius: 6px;"
        "  border: none;"
        "}"
    ));
    m_pLblOverlayResult->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    m_pLblOverlayResult->move(12, 12);  // 20260324 ZJH 左上角偏移
    m_pLblOverlayResult->setVisible(false);

    pLayout->addWidget(m_pInferView, 1);  // 20260324 ZJH 填满中央区域

    return pPanel;
}

// 20260324 ZJH 创建推理模式的右侧控制面板
QWidget* InspectionPage::createInferenceRightPanel()
{
    QWidget* pPanel = new QWidget(this);

    // 20260324 ZJH 使用 QScrollArea 包装，以防内容超出面板高度
    QScrollArea* pScrollArea = new QScrollArea(pPanel);
    pScrollArea->setWidgetResizable(true);
    pScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    pScrollArea->setStyleSheet(QStringLiteral(
        "QScrollArea { border: none; background-color: transparent; }"
        "QScrollBar:vertical {"
        "  background-color: #161b22; width: 8px; border: none;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background-color: #2a2d35; border-radius: 4px; min-height: 20px;"
        "}"
    ));

    QWidget* pContent = new QWidget();
    QVBoxLayout* pLayout = new QVBoxLayout(pContent);
    pLayout->setContentsMargins(8, 8, 8, 8);
    pLayout->setSpacing(8);

    // 20260324 ZJH 通用样式定义
    QString strGroupStyle = QStringLiteral(
        "QGroupBox {"
        "  font-weight: bold;"
        "  color: #94a3b8;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  margin-top: 12px;"
        "  padding-top: 16px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px;"
        "}"
    );

    QString strLabelStyle = QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 11px; border: none; }");
    QString strValueStyle = QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 12px; border: none; }");

    QString strBtnStyle = QStringLiteral(
        "QPushButton {"
        "  background-color: #21262d;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 6px 12px;"
        "  font-size: 11px;"
        "}"
        "QPushButton:hover { background-color: #30363d; }"
        "QPushButton:pressed { background-color: #1a1d24; }"
        "QPushButton:disabled { color: #484f58; background-color: #161b22; border-color: #21262d; }"
    );

    QString strSpinStyle = QStringLiteral(
        "QSpinBox {"
        "  background-color: #1e2230;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 2px 6px;"
        "  min-height: 22px;"
        "}"
    );

    QString strComboStyle = QStringLiteral(
        "QComboBox {"
        "  background-color: #1e2230; color: #e2e8f0;"
        "  border: 1px solid #2a2d35; border-radius: 4px;"
        "  padding: 2px 8px; min-height: 22px;"
        "}"
        "QComboBox::drop-down { border: none; width: 20px; }"
        "QComboBox QAbstractItemView {"
        "  background-color: #1e2230; color: #e2e8f0;"
        "  selection-background-color: #2563eb; border: 1px solid #2a2d35;"
        "}"
    );

    // ===== 1. 模型加载分组 =====

    QGroupBox* pModelGroup = new QGroupBox(QStringLiteral("模型加载"), pContent);
    pModelGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pModelLayout = new QVBoxLayout(pModelGroup);
    pModelLayout->setSpacing(6);

    // 20260324 ZJH 模型架构选择
    QLabel* pLblArch = new QLabel(QStringLiteral("模型架构:"), pModelGroup);
    pLblArch->setStyleSheet(strLabelStyle);
    pModelLayout->addWidget(pLblArch);

    m_pCmbModelArch = new QComboBox(pModelGroup);
    m_pCmbModelArch->setStyleSheet(strComboStyle);
    // 20260325 ZJH 全部已实现的模型架构（与 EngineBridge::createModel 同步）
    // 分类
    m_pCmbModelArch->addItem(QStringLiteral("ResNet-18"));
    m_pCmbModelArch->addItem(QStringLiteral("ResNet-50"));
    m_pCmbModelArch->addItem(QStringLiteral("MobileNetV4-Small"));
    m_pCmbModelArch->addItem(QStringLiteral("ViT-Tiny"));
    m_pCmbModelArch->addItem(QStringLiteral("MLP"));
    // 目标检测
    m_pCmbModelArch->addItem(QStringLiteral("YOLOv5-Nano"));
    m_pCmbModelArch->addItem(QStringLiteral("YOLOv8-Nano"));
    // 语义分割
    m_pCmbModelArch->addItem(QStringLiteral("U-Net"));
    m_pCmbModelArch->addItem(QStringLiteral("DeepLabV3+"));
    m_pCmbModelArch->addItem(QStringLiteral("MobileSegNet"));     // 20260401 ZJH 轻量级分割（~1.75M 参数）
    // 异常检测
    m_pCmbModelArch->addItem(QStringLiteral("EfficientAD"));
    // 实例分割
    m_pCmbModelArch->addItem(QStringLiteral("YOLOv8-Seg"));
    m_pCmbModelArch->addItem(QStringLiteral("Mask R-CNN"));
    pModelLayout->addWidget(m_pCmbModelArch);

    // 20260324 ZJH 输入尺寸
    QLabel* pLblInputSize = new QLabel(QStringLiteral("输入尺寸:"), pModelGroup);
    pLblInputSize->setStyleSheet(strLabelStyle);
    pModelLayout->addWidget(pLblInputSize);

    m_pSpnInputSize = new QSpinBox(pModelGroup);
    m_pSpnInputSize->setStyleSheet(strSpinStyle);
    m_pSpnInputSize->setRange(32, 1024);    // 20260324 ZJH 范围 32~1024
    m_pSpnInputSize->setSingleStep(32);      // 20260324 ZJH 步长 32
    m_pSpnInputSize->setValue(224);           // 20260324 ZJH 默认 224
    m_pSpnInputSize->setSuffix(QStringLiteral(" px"));
    pModelLayout->addWidget(m_pSpnInputSize);

    // 20260324 ZJH 类别数量
    QLabel* pLblNumClasses = new QLabel(QStringLiteral("类别数量:"), pModelGroup);
    pLblNumClasses->setStyleSheet(strLabelStyle);
    pModelLayout->addWidget(pLblNumClasses);

    m_pSpnNumClasses = new QSpinBox(pModelGroup);
    m_pSpnNumClasses->setStyleSheet(strSpinStyle);
    m_pSpnNumClasses->setRange(2, 1000);     // 20260324 ZJH 范围 2~1000
    m_pSpnNumClasses->setValue(2);            // 20260324 ZJH 默认 2 类（OK/NG）
    pModelLayout->addWidget(m_pSpnNumClasses);

    // 20260324 ZJH 加载/卸载按钮行
    QHBoxLayout* pModelBtnLayout = new QHBoxLayout();
    pModelBtnLayout->setSpacing(6);

    m_pBtnLoadModel = new QPushButton(QStringLiteral("加载模型"), pModelGroup);
    m_pBtnLoadModel->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #238636;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 6px 12px;"
        "  font-weight: bold;"
        "  font-size: 11px;"
        "}"
        "QPushButton:hover { background-color: #2ea043; }"
        "QPushButton:pressed { background-color: #1a7f37; }"
    ));
    pModelBtnLayout->addWidget(m_pBtnLoadModel);

    m_pBtnUnloadModel = new QPushButton(QStringLiteral("卸载模型"), pModelGroup);
    m_pBtnUnloadModel->setStyleSheet(strBtnStyle);
    m_pBtnUnloadModel->setEnabled(false);  // 20260324 ZJH 初始禁用
    pModelBtnLayout->addWidget(m_pBtnUnloadModel);

    pModelLayout->addLayout(pModelBtnLayout);

    // 20260324 ZJH 模型状态标签
    m_pLblModelStatus = new QLabel(QStringLiteral("未加载"), pModelGroup);
    m_pLblModelStatus->setStyleSheet(QStringLiteral(
        "QLabel { color: #f97316; font-size: 11px; font-weight: bold; border: none; }"));
    m_pLblModelStatus->setWordWrap(true);
    pModelLayout->addWidget(m_pLblModelStatus);

    // 20260324 ZJH 连接模型按钮信号
    connect(m_pBtnLoadModel, &QPushButton::clicked,
            this, &InspectionPage::onLoadModel);
    connect(m_pBtnUnloadModel, &QPushButton::clicked,
            this, &InspectionPage::onUnloadModel);

    pLayout->addWidget(pModelGroup);

    // ===== 2. 测试图像分组 =====

    QGroupBox* pTestGroup = new QGroupBox(QStringLiteral("测试图像"), pContent);
    pTestGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pTestLayout = new QVBoxLayout(pTestGroup);
    pTestLayout->setSpacing(6);

    // 20260324 ZJH 导入按钮行
    QHBoxLayout* pImportLayout = new QHBoxLayout();
    pImportLayout->setSpacing(4);

    m_pBtnImportImages = new QPushButton(QStringLiteral("导入图像"), pTestGroup);
    m_pBtnImportImages->setStyleSheet(strBtnStyle);
    m_pBtnImportImages->setToolTip(QStringLiteral("选择一张或多张图像文件"));
    pImportLayout->addWidget(m_pBtnImportImages);

    m_pBtnImportFolder = new QPushButton(QStringLiteral("导入文件夹"), pTestGroup);
    m_pBtnImportFolder->setStyleSheet(strBtnStyle);
    m_pBtnImportFolder->setToolTip(QStringLiteral("导入文件夹中所有图像"));
    pImportLayout->addWidget(m_pBtnImportFolder);

    pTestLayout->addLayout(pImportLayout);

    // 20260324 ZJH 清除按钮和计数标签
    QHBoxLayout* pClearLayout = new QHBoxLayout();
    pClearLayout->setSpacing(4);

    m_pBtnClearImages = new QPushButton(QStringLiteral("清除全部"), pTestGroup);
    m_pBtnClearImages->setStyleSheet(strBtnStyle);
    pClearLayout->addWidget(m_pBtnClearImages);

    m_pLblTestImageCount = new QLabel(QStringLiteral("共 0 张测试图像"), pTestGroup);
    m_pLblTestImageCount->setStyleSheet(strValueStyle);
    pClearLayout->addWidget(m_pLblTestImageCount, 1);

    pTestLayout->addLayout(pClearLayout);

    // 20260324 ZJH 测试图像文件名列表
    m_pTestImageModel = new QStandardItemModel(this);
    m_pTestImageList = new QListView(pTestGroup);
    m_pTestImageList->setModel(m_pTestImageModel);
    m_pTestImageList->setMaximumHeight(120);  // 20260324 ZJH 限制列表高度
    m_pTestImageList->setSelectionMode(QListView::SingleSelection);
    m_pTestImageList->setStyleSheet(QStringLiteral(
        "QListView {"
        "  background-color: #0d1117;"
        "  color: #e2e8f0;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  font-size: 11px;"
        "}"
        "QListView::item { padding: 2px 4px; }"
        "QListView::item:selected { background-color: #1f6feb; }"
    ));
    pTestLayout->addWidget(m_pTestImageList);

    // 20260324 ZJH 连接测试图像按钮信号
    connect(m_pBtnImportImages, &QPushButton::clicked,
            this, &InspectionPage::onImportTestImages);
    connect(m_pBtnImportFolder, &QPushButton::clicked,
            this, &InspectionPage::onImportTestFolder);
    connect(m_pBtnClearImages, &QPushButton::clicked,
            this, &InspectionPage::onClearTestImages);
    connect(m_pTestImageList->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &InspectionPage::onTestImageSelectionChanged);

    pLayout->addWidget(pTestGroup);

    // ===== 3. 后处理参数分组（借鉴 MVTec threshold + min_area + Hikvision MinScore）=====

    QGroupBox* pPostGroup = new QGroupBox(QStringLiteral("后处理参数"), pContent);
    pPostGroup->setStyleSheet(strGroupStyle);
    QFormLayout* pPostForm = new QFormLayout(pPostGroup);
    pPostForm->setSpacing(4);
    pPostForm->setContentsMargins(8, 12, 8, 8);

    // 20260330 ZJH 缺陷置信度阈值（MVTec: classification_threshold, 海康: MinScore）
    m_pSpnConfThreshold = new QDoubleSpinBox(pPostGroup);
    m_pSpnConfThreshold->setStyleSheet(strComboStyle);
    m_pSpnConfThreshold->setRange(0.01, 0.99);
    m_pSpnConfThreshold->setSingleStep(0.05);
    m_pSpnConfThreshold->setValue(0.50);  // 20260330 ZJH 默认 50% 置信度
    m_pSpnConfThreshold->setDecimals(2);
    m_pSpnConfThreshold->setToolTip(QStringLiteral("缺陷像素占比超过此阈值判定为缺陷图像（对标 MVTec classification_threshold）"));
    pPostForm->addRow(QStringLiteral("置信阈值:"), m_pSpnConfThreshold);

    // 20260330 ZJH 最小缺陷面积（MVTec: defect_area_min, 过滤噪点）
    m_pSpnMinArea = new QSpinBox(pPostGroup);
    m_pSpnMinArea->setStyleSheet(strComboStyle);
    m_pSpnMinArea->setRange(0, 10000);
    m_pSpnMinArea->setSingleStep(10);
    m_pSpnMinArea->setValue(0);  // 20260330 ZJH 默认不过滤
    m_pSpnMinArea->setSuffix(QStringLiteral(" px"));
    m_pSpnMinArea->setToolTip(QStringLiteral("缺陷区域面积小于此值将被过滤（对标 MVTec defect_area_min）"));
    pPostForm->addRow(QStringLiteral("最小面积:"), m_pSpnMinArea);

    // 20260330 ZJH 缺陷图概率下限（低于此值截为黑色，控制缺陷图显示灵敏度）
    m_pSpnProbFloor = new QDoubleSpinBox(pPostGroup);
    m_pSpnProbFloor->setStyleSheet(strComboStyle);
    m_pSpnProbFloor->setRange(0.01, 0.50);
    m_pSpnProbFloor->setSingleStep(0.01);
    m_pSpnProbFloor->setValue(0.05);  // 20260330 ZJH 默认 5%
    m_pSpnProbFloor->setDecimals(2);
    m_pSpnProbFloor->setToolTip(QStringLiteral("缺陷图中概率低于此值的像素显示为黑色（灵敏度控制）"));
    pPostForm->addRow(QStringLiteral("概率下限:"), m_pSpnProbFloor);

    // 20260330 ZJH 批量推理批次大小（每批送入引擎的图像张数，对标 MVTec batch_size）
    m_pSpnInferBatchSize = new QSpinBox(pPostGroup);
    m_pSpnInferBatchSize->setStyleSheet(strComboStyle);
    m_pSpnInferBatchSize->setRange(1, 32);       // 20260330 ZJH 范围 1~32
    m_pSpnInferBatchSize->setSingleStep(1);      // 20260330 ZJH 步长 1
    m_pSpnInferBatchSize->setValue(1);            // 20260330 ZJH 默认单张推理
    m_pSpnInferBatchSize->setSuffix(QStringLiteral(" 张/批次"));
    m_pSpnInferBatchSize->setToolTip(QStringLiteral("每批送入推理引擎的图像张数，增大可提高 GPU 利用率"));
    pPostForm->addRow(QStringLiteral("批量推理:"), m_pSpnInferBatchSize);

    pLayout->addWidget(pPostGroup);

    // ===== 4. 推理控制分组 =====

    QGroupBox* pInferGroup = new QGroupBox(QStringLiteral("推理控制"), pContent);
    pInferGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pInferLayout = new QVBoxLayout(pInferGroup);
    pInferLayout->setSpacing(6);

    // 20260324 ZJH "运行推理" 大蓝色按钮
    m_pBtnRunInference = new QPushButton(QStringLiteral("运行推理"), pInferGroup);
    m_pBtnRunInference->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #2563eb;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 10px 16px;"
        "  font-weight: bold;"
        "  font-size: 13px;"
        "}"
        "QPushButton:hover { background-color: #3b82f6; }"
        "QPushButton:pressed { background-color: #1d4ed8; }"
        "QPushButton:disabled { background-color: #1e3a5f; color: #6b7280; }"
    ));
    m_pBtnRunInference->setEnabled(false);  // 20260324 ZJH 初始禁用
    m_pBtnRunInference->setToolTip(QStringLiteral("对当前选中的测试图像运行推理"));
    pInferLayout->addWidget(m_pBtnRunInference);

    // 20260324 ZJH "批量推理全部" 按钮
    m_pBtnBatchInference = new QPushButton(QStringLiteral("批量推理全部"), pInferGroup);
    m_pBtnBatchInference->setStyleSheet(strBtnStyle);
    m_pBtnBatchInference->setEnabled(false);  // 20260324 ZJH 初始禁用
    m_pBtnBatchInference->setToolTip(QStringLiteral("对所有测试图像逐张推理"));
    pInferLayout->addWidget(m_pBtnBatchInference);

    // 20260324 ZJH 进度条（初始隐藏）
    m_pProgressBar = new QProgressBar(pInferGroup);
    m_pProgressBar->setStyleSheet(QStringLiteral(
        "QProgressBar {"
        "  background-color: #21262d;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  text-align: center;"
        "  color: #e2e8f0;"
        "  font-size: 10px;"
        "  min-height: 18px;"
        "  max-height: 18px;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: #2563eb;"
        "  border-radius: 3px;"
        "}"
    ));
    m_pProgressBar->setVisible(false);
    pInferLayout->addWidget(m_pProgressBar);

    // 20260324 ZJH 连接推理按钮信号
    connect(m_pBtnRunInference, &QPushButton::clicked,
            this, &InspectionPage::onRunInference);
    connect(m_pBtnBatchInference, &QPushButton::clicked,
            this, &InspectionPage::onBatchInference);

    pLayout->addWidget(pInferGroup);

    // ===== 4. 图像导航分组 =====

    QGroupBox* pNavGroup = new QGroupBox(QStringLiteral("图像导航"), pContent);
    pNavGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pNavLayout = new QVBoxLayout(pNavGroup);
    pNavLayout->setSpacing(6);

    // 20260324 ZJH 上一张/下一张按钮行
    QHBoxLayout* pNavBtnLayout = new QHBoxLayout();
    pNavBtnLayout->setSpacing(6);

    m_pBtnPrevImage = new QPushButton(QStringLiteral("<< 上一张"), pNavGroup);
    m_pBtnPrevImage->setStyleSheet(strBtnStyle);
    m_pBtnPrevImage->setEnabled(false);
    pNavBtnLayout->addWidget(m_pBtnPrevImage);

    m_pBtnNextImage = new QPushButton(QStringLiteral("下一张 >>"), pNavGroup);
    m_pBtnNextImage->setStyleSheet(strBtnStyle);
    m_pBtnNextImage->setEnabled(false);
    pNavBtnLayout->addWidget(m_pBtnNextImage);

    pNavLayout->addLayout(pNavBtnLayout);

    // 20260324 ZJH 当前图像索引标签
    m_pLblImageIndex = new QLabel(QStringLiteral("-- / --"), pNavGroup);
    m_pLblImageIndex->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 12px; font-weight: bold; border: none; }"));
    m_pLblImageIndex->setAlignment(Qt::AlignCenter);
    pNavLayout->addWidget(m_pLblImageIndex);

    // 20260324 ZJH 连接导航按钮信号
    connect(m_pBtnPrevImage, &QPushButton::clicked,
            this, &InspectionPage::onPrevTestImage);
    connect(m_pBtnNextImage, &QPushButton::clicked,
            this, &InspectionPage::onNextTestImage);

    pLayout->addWidget(pNavGroup);

    // ===== 5. 推理结果分组 =====

    QGroupBox* pResultGroup = new QGroupBox(QStringLiteral("推理结果"), pContent);
    pResultGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pResultLayout = new QVBoxLayout(pResultGroup);
    pResultLayout->setSpacing(4);

    // 20260324 ZJH 预测类别
    QLabel* pLblPredTitle = new QLabel(QStringLiteral("预测类别:"), pResultGroup);
    pLblPredTitle->setStyleSheet(strLabelStyle);
    pResultLayout->addWidget(pLblPredTitle);

    m_pLblPredClass = new QLabel(QStringLiteral("--"), pResultGroup);
    m_pLblPredClass->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 16px; font-weight: bold; border: none; }"));
    pResultLayout->addWidget(m_pLblPredClass);

    // 20260324 ZJH 置信度
    QLabel* pLblConfTitle = new QLabel(QStringLiteral("置信度:"), pResultGroup);
    pLblConfTitle->setStyleSheet(strLabelStyle);
    pResultLayout->addWidget(pLblConfTitle);

    m_pLblConfidence = new QLabel(QStringLiteral("--"), pResultGroup);
    m_pLblConfidence->setStyleSheet(strValueStyle);
    pResultLayout->addWidget(m_pLblConfidence);

    // 20260324 ZJH 推理耗时
    QLabel* pLblLatTitle = new QLabel(QStringLiteral("推理耗时:"), pResultGroup);
    pLblLatTitle->setStyleSheet(strLabelStyle);
    pResultLayout->addWidget(pLblLatTitle);

    m_pLblLatency = new QLabel(QStringLiteral("--"), pResultGroup);
    m_pLblLatency->setStyleSheet(strValueStyle);
    pResultLayout->addWidget(m_pLblLatency);

    // 20260324 ZJH 各类别概率
    QLabel* pLblProbTitle = new QLabel(QStringLiteral("类别概率:"), pResultGroup);
    pLblProbTitle->setStyleSheet(strLabelStyle);
    pResultLayout->addWidget(pLblProbTitle);

    m_pLblClassProbs = new QLabel(QStringLiteral(""), pResultGroup);
    m_pLblClassProbs->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 11px; border: none; font-family: monospace; }"));
    m_pLblClassProbs->setWordWrap(true);
    m_pLblClassProbs->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    pResultLayout->addWidget(m_pLblClassProbs);

    pLayout->addWidget(pResultGroup);

    // ===== 5.5 分割可视化分组（Phase 1 新增 — mask透明度 + 缺陷度量）=====

    QGroupBox* pSegVisGroup = new QGroupBox(QStringLiteral("分割可视化"), pContent);
    pSegVisGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pSegVisLayout = new QVBoxLayout(pSegVisGroup);
    pSegVisLayout->setSpacing(6);

    // 20260401 ZJH Mask 透明度滑块（0~255，实时调节 mask overlay 不透明度）
    QLabel* pLblAlphaTitle = new QLabel(QStringLiteral("Mask 透明度:"), pSegVisGroup);
    pLblAlphaTitle->setStyleSheet(strLabelStyle);
    pSegVisLayout->addWidget(pLblAlphaTitle);

    QHBoxLayout* pAlphaRow = new QHBoxLayout();
    pAlphaRow->setSpacing(6);

    m_pSldMaskAlpha = new QSlider(Qt::Horizontal, pSegVisGroup);
    m_pSldMaskAlpha->setRange(0, 255);    // 20260401 ZJH 范围 0（完全透明）~255（完全不透明）
    m_pSldMaskAlpha->setValue(120);        // 20260401 ZJH 默认 120（约 47%）
    m_pSldMaskAlpha->setStyleSheet(QStringLiteral(
        "QSlider::groove:horizontal {"
        "  background: #21262d; height: 6px; border-radius: 3px;"
        "}"
        "QSlider::handle:horizontal {"
        "  background: #2563eb; width: 14px; height: 14px;"
        "  margin: -4px 0; border-radius: 7px;"
        "}"
        "QSlider::sub-page:horizontal {"
        "  background: #2563eb; border-radius: 3px;"
        "}"
    ));
    m_pSldMaskAlpha->setToolTip(QStringLiteral("调节分割 mask 叠加层的不透明度（0=完全透明, 255=完全不透明）"));
    pAlphaRow->addWidget(m_pSldMaskAlpha, 1);

    m_pLblMaskAlphaVal = new QLabel(QStringLiteral("120"), pSegVisGroup);
    m_pLblMaskAlphaVal->setStyleSheet(strValueStyle);
    m_pLblMaskAlphaVal->setFixedWidth(32);
    m_pLblMaskAlphaVal->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    pAlphaRow->addWidget(m_pLblMaskAlphaVal);

    pSegVisLayout->addLayout(pAlphaRow);

    // 20260401 ZJH 滑块值变化 → 更新透明度 + 数值标签 + 触发重绘当前结果
    connect(m_pSldMaskAlpha, &QSlider::valueChanged, this, [this](int nValue) {
        m_nMaskAlpha = nValue;  // 20260401 ZJH 保存新透明度值
        if (m_pLblMaskAlphaVal) {
            m_pLblMaskAlphaVal->setText(QString::number(nValue));  // 20260401 ZJH 更新数值标签
        }
        // 20260401 ZJH 如果有当前推理结果，重新渲染以应用新透明度
        if (m_nCurrentTestIndex >= 0 && m_nCurrentTestIndex < m_vecResults.size()
            && !m_vecResults[m_nCurrentTestIndex].imgDefectMap.isNull()) {
            displayInferResult(m_vecResults[m_nCurrentTestIndex]);
        }
    });

    // 20260401 ZJH 缺陷连通域度量信息面板
    QLabel* pLblMetricsTitle = new QLabel(QStringLiteral("缺陷区域度量:"), pSegVisGroup);
    pLblMetricsTitle->setStyleSheet(strLabelStyle);
    pSegVisLayout->addWidget(pLblMetricsTitle);

    m_pLblDefectMetrics = new QLabel(QStringLiteral("--"), pSegVisGroup);
    m_pLblDefectMetrics->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 10px; border: 1px solid #2a2d35;"
        "  border-radius: 4px; padding: 6px; background-color: #0d1117;"
        "  font-family: 'Consolas', 'Courier New', monospace; }"));
    m_pLblDefectMetrics->setWordWrap(true);
    m_pLblDefectMetrics->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    m_pLblDefectMetrics->setMinimumHeight(60);
    m_pLblDefectMetrics->setVisible(false);  // 20260401 ZJH 初始隐藏，推理后显示
    pSegVisLayout->addWidget(m_pLblDefectMetrics);

    pLayout->addWidget(pSegVisGroup);

    // ===== 6. 推理增强分组（TTA + 滑动窗口 + GradCAM）=====

    QGroupBox* pAugGroup = new QGroupBox(QStringLiteral("推理增强"), pContent);
    pAugGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pAugLayout = new QVBoxLayout(pAugGroup);
    pAugLayout->setSpacing(6);

    // 20260330 ZJH 测试时增强（TTA）开关
    m_pChkEnableTTA = new QCheckBox(QStringLiteral("测试时增强 (TTA)"), pAugGroup);
    m_pChkEnableTTA->setStyleSheet(QStringLiteral(
        "QCheckBox { color: #e2e8f0; font-size: 11px; spacing: 6px; }"
        "QCheckBox::indicator { width: 14px; height: 14px; }"
        "QCheckBox::indicator:unchecked {"
        "  border: 1px solid #2a2d35; border-radius: 3px; background-color: #1e2230;"
        "}"
        "QCheckBox::indicator:checked {"
        "  border: 1px solid #2563eb; border-radius: 3px; background-color: #2563eb;"
        "}"
    ));
    m_pChkEnableTTA->setToolTip(QStringLiteral("启用后对同一图像进行多次变换推理并融合结果，提升精度但增加耗时"));
    pAugLayout->addWidget(m_pChkEnableTTA);

    // 20260330 ZJH TTA 模式选择
    m_pCboTTAMode = new QComboBox(pAugGroup);
    m_pCboTTAMode->setStyleSheet(strComboStyle);
    m_pCboTTAMode->addItem(QStringLiteral("翻转"));                     // 20260330 ZJH 水平+垂直翻转（4次推理）
    m_pCboTTAMode->addItem(QStringLiteral("翻转+旋转"));               // 20260330 ZJH 翻转+90°旋转（8次推理）
    m_pCboTTAMode->addItem(QStringLiteral("翻转+旋转+多尺度"));       // 20260330 ZJH 翻转+旋转+多尺度裁剪（16+次推理）
    m_pCboTTAMode->setEnabled(false);  // 20260330 ZJH 初始禁用，TTA 开启后激活
    m_pCboTTAMode->setToolTip(QStringLiteral("选择 TTA 变换组合：翻转(4x)、翻转+旋转(8x)、全模式(16+x)"));
    pAugLayout->addWidget(m_pCboTTAMode);

    // 20260330 ZJH TTA 开关联动：勾选时启用模式下拉框，取消时禁用
    connect(m_pChkEnableTTA, &QCheckBox::toggled,
            m_pCboTTAMode, &QComboBox::setEnabled);

    // 20260330 ZJH 滑动窗口大图检测开关（SOD: Small Object Detection on large images）
    m_pChkSlidingWindow = new QCheckBox(QStringLiteral("滑动窗口 (大图检测)"), pAugGroup);
    m_pChkSlidingWindow->setStyleSheet(m_pChkEnableTTA->styleSheet());  // 20260330 ZJH 复用相同复选框样式
    m_pChkSlidingWindow->setToolTip(QStringLiteral("对超大分辨率图像按窗口切片推理后合并结果，适用于高分辨率全景检测"));
    pAugLayout->addWidget(m_pChkSlidingWindow);

    // 20260330 ZJH 滑动窗口参数行：窗口尺寸 + 重叠率
    QHBoxLayout* pSlidingParamLayout = new QHBoxLayout();
    pSlidingParamLayout->setSpacing(6);

    // 20260330 ZJH 滑动窗口尺寸
    m_pSpnWindowSize = new QSpinBox(pAugGroup);
    m_pSpnWindowSize->setStyleSheet(strSpinStyle);
    m_pSpnWindowSize->setRange(320, 1024);   // 20260330 ZJH 范围 320~1024
    m_pSpnWindowSize->setSingleStep(32);     // 20260330 ZJH 步长 32（对齐模型输入粒度）
    m_pSpnWindowSize->setValue(640);         // 20260330 ZJH 默认 640（YOLOv5/v8 常用尺寸）
    m_pSpnWindowSize->setSuffix(QStringLiteral(" px"));
    m_pSpnWindowSize->setToolTip(QStringLiteral("滑动窗口边长（像素），建议与模型输入尺寸一致"));
    m_pSpnWindowSize->setEnabled(false);     // 20260330 ZJH 初始禁用，滑动窗口开启后激活
    pSlidingParamLayout->addWidget(m_pSpnWindowSize, 1);

    // 20260330 ZJH 滑动窗口重叠率
    m_pSpnOverlap = new QDoubleSpinBox(pAugGroup);
    m_pSpnOverlap->setStyleSheet(strComboStyle);
    m_pSpnOverlap->setRange(0.0, 0.5);      // 20260330 ZJH 范围 0%~50%
    m_pSpnOverlap->setSingleStep(0.05);      // 20260330 ZJH 步长 5%
    m_pSpnOverlap->setValue(0.25);           // 20260330 ZJH 默认 25% 重叠
    m_pSpnOverlap->setDecimals(2);
    m_pSpnOverlap->setPrefix(QStringLiteral("重叠 "));
    m_pSpnOverlap->setToolTip(QStringLiteral("相邻窗口重叠比例，越大则边界检测越完整但耗时越长"));
    m_pSpnOverlap->setEnabled(false);        // 20260330 ZJH 初始禁用，滑动窗口开启后激活
    pSlidingParamLayout->addWidget(m_pSpnOverlap, 1);

    pAugLayout->addLayout(pSlidingParamLayout);

    // 20260330 ZJH 滑动窗口开关联动：勾选时启用窗口尺寸和重叠率，取消时禁用
    connect(m_pChkSlidingWindow, &QCheckBox::toggled, m_pSpnWindowSize, &QSpinBox::setEnabled);
    connect(m_pChkSlidingWindow, &QCheckBox::toggled, m_pSpnOverlap, &QDoubleSpinBox::setEnabled);

    // 20260330 ZJH GradCAM 注意力热力图可视化开关
    m_pChkShowGradCAM = new QCheckBox(QStringLiteral("显示注意力热力图 (GradCAM)"), pAugGroup);
    m_pChkShowGradCAM->setStyleSheet(m_pChkEnableTTA->styleSheet());  // 20260330 ZJH 复用相同复选框样式
    m_pChkShowGradCAM->setToolTip(QStringLiteral("推理时生成 GradCAM 热力图叠加到图像上，可视化模型关注区域"));
    pAugLayout->addWidget(m_pChkShowGradCAM);

    pLayout->addWidget(pAugGroup);

    // ===== 7. 检测余裕趋势分组 =====

    QGroupBox* pMarginGroup = new QGroupBox(QStringLiteral("检测余裕趋势"), pContent);
    pMarginGroup->setStyleSheet(strGroupStyle);

    QVBoxLayout* pMarginLayout = new QVBoxLayout(pMarginGroup);
    pMarginLayout->setSpacing(6);

    // 20260330 ZJH 检测余裕趋势标签说明
    QLabel* pLblMarginTitle = new QLabel(QStringLiteral("检测余裕趋势"), pMarginGroup);
    pLblMarginTitle->setStyleSheet(strLabelStyle);
    pMarginLayout->addWidget(pLblMarginTitle);

    // 20260330 ZJH 余裕指示进度条（0~100%，用作可视化指示器而非进度条）
    m_pBarMargin = new QProgressBar(pMarginGroup);
    m_pBarMargin->setRange(0, 100);   // 20260330 ZJH 范围 0%~100%
    m_pBarMargin->setValue(0);         // 20260330 ZJH 初始值 0%
    m_pBarMargin->setTextVisible(true);
    m_pBarMargin->setFormat(QStringLiteral("%p%"));
    m_pBarMargin->setStyleSheet(QStringLiteral(
        "QProgressBar {"
        "  background-color: #21262d;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  text-align: center;"
        "  color: #e2e8f0;"
        "  font-size: 10px;"
        "  min-height: 18px;"
        "  max-height: 18px;"
        "}"
        "QProgressBar::chunk {"
        "  border-radius: 3px;"
        "  background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        "    stop:0 #ef4444, stop:0.5 #f59e0b, stop:1 #22c55e);"
        "}"
    ));
    m_pBarMargin->setToolTip(QStringLiteral("检测余裕 = 最高置信度与阈值之差，余裕越大越稳定"));
    pMarginLayout->addWidget(m_pBarMargin);

    // 20260330 ZJH 余裕数值标签（显示当前余裕百分比）
    m_pLblMarginValue = new QLabel(QStringLiteral("余裕: --"), pMarginGroup);
    m_pLblMarginValue->setStyleSheet(strValueStyle);
    m_pLblMarginValue->setAlignment(Qt::AlignCenter);
    pMarginLayout->addWidget(m_pLblMarginValue);

    pLayout->addWidget(pMarginGroup);

    // 20260324 ZJH 弹性空间填充底部
    pLayout->addStretch(1);

    // 20260324 ZJH 设置滚动区域内容
    pScrollArea->setWidget(pContent);

    // 20260324 ZJH 外层布局包装 QScrollArea
    QVBoxLayout* pOuterLayout = new QVBoxLayout(pPanel);
    pOuterLayout->setContentsMargins(0, 0, 0, 0);
    pOuterLayout->addWidget(pScrollArea, 1);

    return pPanel;
}

// ===== 私有方法 — 检查模式 =====

// 20260322 ZJH 刷新缩略图模型
void InspectionPage::refreshModel()
{
    if (!m_pModel) {
        return;
    }

    m_pModel->clear();  // 20260322 ZJH 清空现有数据

    if (!m_pDataset) {
        return;  // 20260322 ZJH 无数据集
    }

    // 20260402 ZJH ===== 实例视图（对标 Halcon DL Tool 检查页）=====
    // 核心行为: 遍历所有图像的所有标注实例，每个 Annotation 生成一个独立缩略图
    // 一张含 3 个缺陷标注的图 → 3 个独立缩略图（每个显示裁剪后的缺陷区域）
    // 无标注的图像 → 作为整图显示（未标注项，标签 ID = -1）
    // 排序: 按标签 ID 分组，同类缺陷连续显示（Halcon 行为）

    // 20260404 ZJH [重构] 实例视图: 只要数据集中有标注就启用（对标 Halcon DL Tool 检查页）
    // Halcon 检查页无论任务类型，都将每个标注实例裁剪显示 + 彩色轮廓叠加
    // 此前仅限检测/实例分割，现扩展到所有有 vecAnnotations 的场景
    bool bInstanceView = false;  // 20260404 ZJH 默认图像视图

    const QVector<ImageEntry>& vecImages = m_pDataset->images();

    // 20260404 ZJH 扫描数据集: 只要任意图像有标注就启用实例视图
    for (const ImageEntry& entry : vecImages) {
        if (!entry.vecAnnotations.isEmpty()) {
            bInstanceView = true;
            break;  // 20260404 ZJH 找到一个就够了
        }
    }

    if (bInstanceView) {
        // 20260402 ZJH ===== 实例视图: 每个标注实例一个缩略图 =====
        // 收集所有实例，按 labelId 排序后一次性添加（Halcon 风格分组）
        struct InstanceInfo {
            int nImageIndex;      // 20260402 ZJH 源图像索引
            int nAnnotationIndex; // 20260402 ZJH 标注在 vecAnnotations 中的索引
            int nLabelId;         // 20260402 ZJH 标签 ID
            QRectF rectCrop;      // 20260402 ZJH 裁剪矩形（原图像素坐标）
        };
        QVector<InstanceInfo> vecInstances;  // 20260402 ZJH 所有实例列表

        for (int nImgIdx = 0; nImgIdx < vecImages.size(); ++nImgIdx) {
            const ImageEntry& entry = vecImages[nImgIdx];
            if (entry.vecAnnotations.isEmpty()) {
                // 20260402 ZJH 无标注的图像 → 整图作为一个 "未标注" 实例
                InstanceInfo info;
                info.nImageIndex = nImgIdx;
                info.nAnnotationIndex = -1;  // 20260402 ZJH -1 表示无标注（整图）
                info.nLabelId = entry.nLabelId;  // 20260402 ZJH 图像级标签（可能 -1）
                info.rectCrop = QRectF();  // 20260402 ZJH 空矩形 = 整图
                vecInstances.append(info);
            } else {
                // 20260402 ZJH 有标注 → 每个标注生成一个实例
                for (int nAnnIdx = 0; nAnnIdx < entry.vecAnnotations.size(); ++nAnnIdx) {
                    const auto& annotation = entry.vecAnnotations[nAnnIdx];
                    if (annotation.rectBounds.isEmpty()) continue;  // 20260402 ZJH 跳过无效 BBox
                    InstanceInfo info;
                    info.nImageIndex = nImgIdx;
                    info.nAnnotationIndex = nAnnIdx;
                    info.nLabelId = annotation.nLabelId;
                    info.rectCrop = annotation.rectBounds;  // 20260402 ZJH 标注包围框
                    vecInstances.append(info);
                }
            }
        }

        // 20260402 ZJH 按标签 ID 排序（同类缺陷连续显示，对标 Halcon）
        std::sort(vecInstances.begin(), vecInstances.end(),
                  [](const InstanceInfo& a, const InstanceInfo& b) {
                      if (a.nLabelId != b.nLabelId) return a.nLabelId < b.nLabelId;
                      return a.nImageIndex < b.nImageIndex;
                  });

        // 20260402 ZJH 生成 QStandardItem
        for (const InstanceInfo& inst : vecInstances) {
            const ImageEntry& entry = vecImages[inst.nImageIndex];
            QStandardItem* pItem = new QStandardItem();

            pItem->setData(entry.strFilePath,  FilePathRole);     // 20260402 ZJH 源图像路径
            pItem->setData(entry.strUuid,      UuidRole);         // 20260402 ZJH 源图像 UUID
            pItem->setData(inst.nLabelId,      LabelIdRole);      // 20260402 ZJH 标注标签 ID
            pItem->setData(entry.fileName(),   FileNameRole);     // 20260402 ZJH 源文件名
            pItem->setData(static_cast<int>(entry.eSplit), SplitTypeRole);

            // 20260402 ZJH 实例视图专用数据
            pItem->setData(QVariant::fromValue(inst.rectCrop), CropRectRole);  // 20260402 ZJH 裁剪矩形
            pItem->setData(inst.nAnnotationIndex, AnnotationIdxRole);  // 20260402 ZJH 标注索引
            pItem->setData(inst.nImageIndex, ImageIndexRole);          // 20260402 ZJH 源图像索引

            // 20260402 ZJH 查找标签颜色和名称
            if (inst.nLabelId >= 0) {
                const LabelInfo* pLabel = m_pDataset->findLabel(inst.nLabelId);
                if (pLabel) {
                    pItem->setData(pLabel->color,   LabelColorRole);
                    pItem->setData(pLabel->strName, LabelNameRole);
                }
            }

            // 20260404 ZJH 存储标注轮廓多边形（对标 Halcon 检查页彩色轮廓叠加）
            // 用于 ThumbnailDelegate 在裁剪缩略图上绘制标注边界
            if (inst.nAnnotationIndex >= 0 && inst.nAnnotationIndex < entry.vecAnnotations.size()) {
                const auto& ann = entry.vecAnnotations[inst.nAnnotationIndex];
                QPolygonF polyContour;  // 20260404 ZJH 标注轮廓（原图坐标系）
                if (ann.eType == AnnotationType::Polygon && !ann.polygon.isEmpty()) {
                    // 20260404 ZJH Polygon 标注: 直接使用多边形顶点
                    polyContour = ann.polygon;
                } else {
                    // 20260404 ZJH Rect / RotatedRect / 其他: 用 rectBounds 构造矩形多边形
                    polyContour << ann.rectBounds.topLeft()
                                << ann.rectBounds.topRight()
                                << ann.rectBounds.bottomRight()
                                << ann.rectBounds.bottomLeft();
                }
                pItem->setData(QVariant::fromValue(polyContour), AnnotationPolyRole);
            }

            m_pModel->appendRow(pItem);
        }
    } else {
        // 20260402 ZJH ===== 图像视图（分类/异常检测等，原有行为）=====
        for (const ImageEntry& entry : vecImages) {
            QStandardItem* pItem = new QStandardItem();
            pItem->setData(entry.strFilePath,  FilePathRole);
            pItem->setData(entry.strUuid,      UuidRole);
            pItem->setData(entry.nLabelId,     LabelIdRole);
            pItem->setData(entry.fileName(),   FileNameRole);
            pItem->setData(static_cast<int>(entry.eSplit), SplitTypeRole);

            if (entry.nLabelId >= 0) {
                const LabelInfo* pLabel = m_pDataset->findLabel(entry.nLabelId);
                if (pLabel) {
                    pItem->setData(pLabel->color,   LabelColorRole);
                    pItem->setData(pLabel->strName, LabelNameRole);
                }
            }
            m_pModel->appendRow(pItem);
        }
    }
}

// 20260322 ZJH 更新统计标签显示
void InspectionPage::updateStatistics()
{
    if (!m_pLblFilteredCount || !m_pLabelStatsLayout) {
        return;  // 20260402 ZJH 控件未初始化
    }

    // 20260402 ZJH 显示过滤后的数量（实例视图显示"个实例"，图像视图显示"张图像"）
    int nFilteredCount = m_pProxyModel ? m_pProxyModel->rowCount() : 0;
    // 20260404 ZJH 判断当前是否实例视图（与 refreshModel 保持一致）
    bool bIsInstanceView = false;
    if (m_pDataset) {
        const QVector<ImageEntry>& vecImgs = m_pDataset->images();
        for (const ImageEntry& e : vecImgs) {
            if (!e.vecAnnotations.isEmpty()) {
                bIsInstanceView = true;
                break;
            }
        }
    }
    QString strUnit = bIsInstanceView ? QStringLiteral("个标注") : QStringLiteral("张图像");
    m_pLblFilteredCount->setText(QStringLiteral("当前显示: %1 %2").arg(nFilteredCount).arg(strUnit));

    // 20260402 ZJH 清除旧的标签行 widgets
    while (m_pLabelStatsLayout->count() > 0) {
        QLayoutItem* pItem = m_pLabelStatsLayout->takeAt(0);  // 20260402 ZJH 取出首项
        if (pItem->widget()) {
            pItem->widget()->deleteLater();  // 20260402 ZJH 安全删除 widget
        }
        delete pItem;  // 20260402 ZJH 删除 layout item
    }

    if (!m_pDataset) {
        return;  // 20260402 ZJH 无数据集
    }

    // 20260402 ZJH 判断任务类型 — 决定计数单位
    // 分类任务: 一张图一个标签 → 显示 "N 张"
    // 分割/检测/实例分割: 一张图多个标注 → 显示纯数字（与 Halcon 一致）
    bool bShowUnit = true;  // 20260402 ZJH 默认显示"张"
    if (m_pProject) {
        auto eTask = m_pProject->taskType();
        // 20260402 ZJH 分割/检测/实例分割任务不加单位（标注实例数）
        if (eTask == om::TaskType::SemanticSegmentation ||
            eTask == om::TaskType::ObjectDetection ||
            eTask == om::TaskType::InstanceSegmentation) {
            bShowUnit = false;  // 20260402 ZJH 按标注实例计数，不加"张"
        }
    }

    // 20260402 ZJH 获取标签分布（已经按标注实例/图像智能计数）
    QMap<int, int> mapDist = m_pDataset->labelDistribution();
    const QVector<LabelInfo>& vecLabels = m_pDataset->labels();

    // 20260402 ZJH 未标注图像数
    int nUnlabeled = m_pDataset->unlabeledCount();

    // 20260404 ZJH 标签行通用悬停样式（可点击行）
    // 对标 Halcon DL Tool: 点击标签行 → 过滤到只显示该标签的图像
    QString strRowStyleNormal = QStringLiteral(
        "QWidget { border-radius: 3px; padding: 1px 2px; }"
        "QWidget:hover { background-color: #2a2d35; }");

    // 20260402 ZJH 先添加 "未标注的图像" 行（灰色，对标 Halcon 的 "0" 行）
    {
        QWidget* pRow = new QWidget();  // 20260402 ZJH 行容器
        pRow->setCursor(Qt::PointingHandCursor);  // 20260404 ZJH 手形光标提示可点击
        pRow->setStyleSheet(strRowStyleNormal);  // 20260404 ZJH 悬停高亮
        QHBoxLayout* pRowLayout = new QHBoxLayout(pRow);  // 20260402 ZJH 水平布局
        pRowLayout->setContentsMargins(0, 1, 0, 1);  // 20260402 ZJH 紧凑边距
        pRowLayout->setSpacing(6);  // 20260402 ZJH 元素间距

        // 20260402 ZJH 灰色 ID 方块（对标 Halcon 的 "0" 标签）
        QLabel* pIdBox = new QLabel(QStringLiteral("0"));  // 20260402 ZJH ID 数字
        pIdBox->setFixedSize(20, 18);  // 20260402 ZJH 固定尺寸
        pIdBox->setAlignment(Qt::AlignCenter);  // 20260402 ZJH 居中
        pIdBox->setStyleSheet(QStringLiteral(
            "QLabel { background-color: #4a5568; color: #e2e8f0; font-size: 11px; "
            "font-weight: bold; border-radius: 3px; border: none; }"));
        pRowLayout->addWidget(pIdBox);  // 20260402 ZJH 添加 ID 方块

        // 20260402 ZJH 标签名 "未标注的图像"
        QLabel* pName = new QLabel(QStringLiteral("未标注的图像"));
        pName->setStyleSheet(QStringLiteral(
            "QLabel { color: #94a3b8; font-size: 12px; border: none; background: transparent; }"));
        pRowLayout->addWidget(pName, 1);  // 20260402 ZJH stretch=1 占据中间空间

        // 20260402 ZJH 右对齐计数
        QLabel* pCount = new QLabel(QString::number(nUnlabeled));
        pCount->setStyleSheet(QStringLiteral(
            "QLabel { color: #94a3b8; font-size: 12px; border: none; background: transparent; }"));
        pCount->setAlignment(Qt::AlignRight | Qt::AlignVCenter);  // 20260402 ZJH 右对齐
        pRowLayout->addWidget(pCount);  // 20260402 ZJH 添加计数

        // 20260404 ZJH 点击未标注行: 状态过滤切换到"未标注"，标签过滤切回"全部"
        pRow->installEventFilter(this);
        pRow->setProperty("labelFilterId", -2);  // 20260404 ZJH -2 = 特殊值: 未标注行

        m_pLabelStatsLayout->addWidget(pRow);  // 20260402 ZJH 添加到动态容器
    }

    // 20260402 ZJH 逐标签添加行: [彩色ID] [名称] [计数]
    for (const LabelInfo& label : vecLabels) {
        int nCount = mapDist.value(label.nId, 0);  // 20260402 ZJH 标注实例数

        QWidget* pRow = new QWidget();  // 20260402 ZJH 行容器
        pRow->setCursor(Qt::PointingHandCursor);  // 20260404 ZJH 手形光标提示可点击
        pRow->setStyleSheet(strRowStyleNormal);  // 20260404 ZJH 悬停高亮
        QHBoxLayout* pRowLayout = new QHBoxLayout(pRow);  // 20260402 ZJH 水平布局
        pRowLayout->setContentsMargins(0, 1, 0, 1);
        pRowLayout->setSpacing(6);

        // 20260402 ZJH 彩色 ID 方块（使用标签自身颜色，对标 Halcon 红/蓝/黄方块）
        QLabel* pIdBox = new QLabel(QString::number(label.nId));  // 20260402 ZJH 标签 ID
        pIdBox->setFixedSize(20, 18);
        pIdBox->setAlignment(Qt::AlignCenter);
        // 20260402 ZJH 根据标签颜色亮度选择文字色（深色背景白字，浅色背景黑字）
        int nLuminance = (label.color.red() * 299 + label.color.green() * 587 + label.color.blue() * 114) / 1000;
        QString strTextColor = (nLuminance > 128) ? QStringLiteral("#1a1a2e") : QStringLiteral("#ffffff");
        pIdBox->setStyleSheet(QStringLiteral(
            "QLabel { background-color: %1; color: %2; font-size: 11px; "
            "font-weight: bold; border-radius: 3px; border: none; }")
            .arg(label.color.name(), strTextColor));
        pRowLayout->addWidget(pIdBox);

        // 20260402 ZJH 标签名称
        QLabel* pName = new QLabel(label.strName);
        pName->setStyleSheet(QStringLiteral(
            "QLabel { color: #e2e8f0; font-size: 12px; border: none; background: transparent; }"));
        pRowLayout->addWidget(pName, 1);  // 20260402 ZJH stretch=1

        // 20260402 ZJH 右对齐计数
        QLabel* pCount = new QLabel(QString::number(nCount));
        pCount->setStyleSheet(QStringLiteral(
            "QLabel { color: #e2e8f0; font-size: 12px; border: none; background: transparent; }"));
        pCount->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        pRowLayout->addWidget(pCount);

        // 20260404 ZJH 点击标签行: 标签过滤切换到该标签，状态过滤切回"全部"
        pRow->installEventFilter(this);
        pRow->setProperty("labelFilterId", label.nId);  // 20260404 ZJH 存储标签 ID

        m_pLabelStatsLayout->addWidget(pRow);  // 20260402 ZJH 添加到容器
    }
}

// 20260322 ZJH 刷新标签过滤下拉框内容
void InspectionPage::refreshLabelCombo()
{
    if (!m_pCmbLabelFilter) {
        return;
    }

    // 20260322 ZJH 阻止信号发射，避免刷新过程中触发过滤
    m_pCmbLabelFilter->blockSignals(true);
    m_pCmbLabelFilter->clear();

    // 20260402 ZJH [修改] "全部"改为"未标注"，对标 Halcon 的标签过滤面板
    // Halcon 检查页的标签过滤: "若要显示实例的子集，请选择一个或多个标签类别"
    // 不需要"全部"选项，默认显示所有，选择具体标签时过滤
    m_pCmbLabelFilter->addItem(QStringLiteral("全部"), -1);  // 20260402 ZJH 保留全部（显示所有标注+未标注）

    if (m_pDataset) {
        // 20260402 ZJH 添加各标签选项
        const QVector<LabelInfo>& vecLabels = m_pDataset->labels();
        for (const LabelInfo& label : vecLabels) {
            m_pCmbLabelFilter->addItem(label.strName, label.nId);
        }
    }

    m_pCmbLabelFilter->blockSignals(false);  // 20260322 ZJH 恢复信号
}

// 20260322 ZJH 更新右面板预览信息
void InspectionPage::updatePreview(const QModelIndex& index)
{
    if (!index.isValid()) {
        clearPreview();
        return;
    }

    // 20260322 ZJH 获取文件路径
    QString strFilePath = index.data(FilePathRole).toString();
    QString strFileName = index.data(FileNameRole).toString();
    int nLabelId = index.data(LabelIdRole).toInt();
    int nSplitType = index.data(SplitTypeRole).toInt();

    // 20260322 ZJH 加载预览图片
    QPixmap pixmap(strFilePath);
    if (!pixmap.isNull()) {
        // 20260322 ZJH 缩放适应预览区域
        QPixmap scaled = pixmap.scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        m_pPreviewLabel->setPixmap(scaled);
    } else {
        m_pPreviewLabel->setText(QStringLiteral("无法加载图像"));
    }

    // 20260322 ZJH 更新文件名
    m_pLblFileName->setText(strFileName);

    // 20260322 ZJH 更新尺寸信息（从数据集中查找）
    if (m_pDataset) {
        QString strUuid = index.data(UuidRole).toString();
        const ImageEntry* pEntry = m_pDataset->findImage(strUuid);
        if (pEntry) {
            m_pLblDimensions->setText(QStringLiteral("%1 x %2")
                .arg(pEntry->nWidth).arg(pEntry->nHeight));
        } else {
            m_pLblDimensions->setText(QStringLiteral("--"));
        }
    }

    // 20260322 ZJH 更新标签名
    if (nLabelId >= 0) {
        QString strLabelName = index.data(LabelNameRole).toString();
        if (strLabelName.isEmpty()) {
            strLabelName = QStringLiteral("标签 %1").arg(nLabelId);
        }
        m_pLblLabelName->setText(strLabelName);
        // 20260322 ZJH 已标注图像用绿色文字
        m_pLblLabelName->setStyleSheet(QStringLiteral(
            "QLabel { color: #22c55e; font-size: 12px; font-weight: bold; border: none; }"));
    } else {
        m_pLblLabelName->setText(QStringLiteral("未标注"));
        // 20260322 ZJH 未标注图像用灰色文字
        m_pLblLabelName->setStyleSheet(QStringLiteral(
            "QLabel { color: #64748b; font-size: 12px; border: none; }"));
    }

    // 20260322 ZJH 更新拆分类型
    static const QStringList s_vecSplitNames = {
        QStringLiteral("训练"), QStringLiteral("验证"),
        QStringLiteral("测试"), QStringLiteral("未分配")
    };
    if (nSplitType >= 0 && nSplitType < s_vecSplitNames.size()) {
        m_pLblSplitType->setText(s_vecSplitNames[nSplitType]);
    } else {
        m_pLblSplitType->setText(QStringLiteral("未分配"));
    }
}

// 20260322 ZJH 清空右面板预览信息
void InspectionPage::clearPreview()
{
    if (m_pPreviewLabel) {
        m_pPreviewLabel->clear();
        m_pPreviewLabel->setText(QStringLiteral("选中图像\n预览"));
    }
    if (m_pLblFileName) {
        m_pLblFileName->setText(QStringLiteral("--"));
    }
    if (m_pLblDimensions) {
        m_pLblDimensions->setText(QStringLiteral("--"));
    }
    if (m_pLblLabelName) {
        m_pLblLabelName->setText(QStringLiteral("--"));
        m_pLblLabelName->setStyleSheet(QStringLiteral(
            "QLabel { color: #e2e8f0; font-size: 12px; border: none; }"));
    }
    if (m_pLblSplitType) {
        m_pLblSplitType->setText(QStringLiteral("--"));
    }
}

// ===== 私有方法 — 推理模式 =====

// 20260324 ZJH 更新模型状态标签
void InspectionPage::updateModelStatus()
{
    if (!m_pLblModelStatus) {
        return;
    }

    if (m_bModelLoaded) {
        // 20260324 ZJH 显示已加载的模型信息
        QString strStatus = QStringLiteral("已加载: %1\n(%2类, %3x%3)")
            .arg(m_strModelArchName)
            .arg(m_nModelNumClasses)
            .arg(m_nModelInputSize);

        // 20260324 ZJH 显示参数总数
        if (m_pInferEngine) {
            int64_t nParams = m_pInferEngine->totalParameters();
            if (nParams > 1000000) {
                strStatus += QStringLiteral("\n参数量: %1M").arg(nParams / 1000000.0, 0, 'f', 1);
            } else if (nParams > 1000) {
                strStatus += QStringLiteral("\n参数量: %1K").arg(nParams / 1000.0, 0, 'f', 1);
            } else {
                strStatus += QStringLiteral("\n参数量: %1").arg(nParams);
            }
        }

        m_pLblModelStatus->setText(strStatus);
        m_pLblModelStatus->setStyleSheet(QStringLiteral(
            "QLabel { color: #22c55e; font-size: 11px; font-weight: bold; border: none; }"));

        // 20260324 ZJH 模型已加载 → 禁用加载按钮，启用卸载按钮，禁用配置控件
        if (m_pBtnLoadModel)   m_pBtnLoadModel->setEnabled(false);    // 20260406 ZJH 禁止重复加载
        if (m_pBtnUnloadModel) m_pBtnUnloadModel->setEnabled(true);   // 20260406 ZJH 允许卸载
        if (m_pCmbModelArch)   m_pCmbModelArch->setEnabled(false);    // 20260406 ZJH 锁定架构选择
        if (m_pSpnInputSize)   m_pSpnInputSize->setEnabled(false);    // 20260406 ZJH 锁定输入尺寸
        if (m_pSpnNumClasses)  m_pSpnNumClasses->setEnabled(false);   // 20260406 ZJH 锁定类别数
    } else {
        // 20260324 ZJH 显示未加载状态
        m_pLblModelStatus->setText(QStringLiteral("未加载"));
        m_pLblModelStatus->setStyleSheet(QStringLiteral(
            "QLabel { color: #f97316; font-size: 11px; font-weight: bold; border: none; }"));

        // 20260324 ZJH 模型未加载 → 启用加载按钮，禁用卸载按钮，允许修改配置
        if (m_pBtnLoadModel)   m_pBtnLoadModel->setEnabled(true);     // 20260406 ZJH 允许加载模型
        if (m_pBtnUnloadModel) m_pBtnUnloadModel->setEnabled(false);  // 20260406 ZJH 无模型可卸载
        if (m_pCmbModelArch)   m_pCmbModelArch->setEnabled(true);     // 20260406 ZJH 允许选择架构
        if (m_pSpnInputSize)   m_pSpnInputSize->setEnabled(true);     // 20260406 ZJH 允许调整输入尺寸
        if (m_pSpnNumClasses)  m_pSpnNumClasses->setEnabled(true);    // 20260406 ZJH 允许调整类别数
    }
}

// 20260324 ZJH 更新推理按钮的启用状态
void InspectionPage::updateInferenceButtons()
{
    // 20260324 ZJH "运行推理" 按钮需要模型已加载且有选中的测试图像
    bool bCanInfer = m_bModelLoaded && m_nCurrentTestIndex >= 0
                     && m_nCurrentTestIndex < m_vecTestImagePaths.size();
    if (m_pBtnRunInference) {
        m_pBtnRunInference->setEnabled(bCanInfer);
    }

    // 20260324 ZJH "批量推理全部" 按钮需要模型已加载且有测试图像
    bool bCanBatch = m_bModelLoaded && !m_vecTestImagePaths.isEmpty();
    if (m_pBtnBatchInference) {
        m_pBtnBatchInference->setEnabled(bCanBatch);
    }

    // 20260324 ZJH 导航按钮需要有测试图像
    bool bHasImages = !m_vecTestImagePaths.isEmpty();
    if (m_pBtnPrevImage) m_pBtnPrevImage->setEnabled(bHasImages);
    if (m_pBtnNextImage) m_pBtnNextImage->setEnabled(bHasImages);
}

// 20260324 ZJH 对指定图像执行推理并返回结果
// 20260326 ZJH 添加 try/catch 保护，防止 forward() 异常导致应用闪退
InferResult InspectionPage::runInferenceOnImage(const QString& strImagePath)
{
    InferResult result;
    result.strImagePath = strImagePath;

    // 20260324 ZJH 安全检查：引擎是否可用
    if (!m_pInferEngine || !m_bModelLoaded) {
        return result;
    }

    // 20260324 ZJH 加载图像
    QImage img(strImagePath);
    if (img.isNull()) {
        return result;  // 20260324 ZJH 加载失败
    }

    // 20260326 ZJH 整个推理流程用 try/catch 保护
    try {

    // 20260324 ZJH 预处理：缩放到模型输入尺寸
    QImage imgResized = img.scaled(m_nModelInputSize, m_nModelInputSize,
                                    Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    // 20260324 ZJH 转换为 RGB888 格式（确保3通道）
    imgResized = imgResized.convertToFormat(QImage::Format_RGB888);

    // 20260324 ZJH 转换为浮点向量 [C, H, W]，归一化到 [0, 1]
    int nPixelCount = m_nModelInputSize * m_nModelInputSize;
    std::vector<float> vecData(3 * nPixelCount);
    const uchar* pPixels = imgResized.constBits();
    int nBytesPerLine = imgResized.bytesPerLine();

    for (int y = 0; y < m_nModelInputSize; ++y) {
        const uchar* pRow = pPixels + y * nBytesPerLine;
        for (int x = 0; x < m_nModelInputSize; ++x) {
            int nPixIdx = y * m_nModelInputSize + x;
            int nSrcIdx = x * 3;
            // 20260324 ZJH CHW 顺序：R 通道 → G 通道 → B 通道
            vecData[0 * nPixelCount + nPixIdx] = pRow[nSrcIdx + 0] / 255.0f;  // R
            vecData[1 * nPixelCount + nPixIdx] = pRow[nSrcIdx + 1] / 255.0f;  // G
            vecData[2 * nPixelCount + nPixIdx] = pRow[nSrcIdx + 2] / 255.0f;  // B
        }
    }

    // 20260324 ZJH 运行推理并计时
    // 20260402 ZJH TTA 推理集成: 根据 TTA 开关选择普通推理或增强推理
    auto tStart = std::chrono::high_resolution_clock::now();
    BridgeInferResult bridgeResult;
    bool bTTAEnabled = (m_pChkEnableTTA && m_pChkEnableTTA->isChecked());  // 20260402 ZJH TTA 开关状态
    if (bTTAEnabled) {
        // 20260402 ZJH TTA 推理: 多增强变换 → 分别推理 → 融合结果
        // 解析 TTA 模式配置（翻转/旋转/多尺度）
        bool bHFlip = true, bVFlip = false, bRotate90 = false, bMultiScale = false;
        if (m_pCboTTAMode) {
            int nMode = m_pCboTTAMode->currentIndex();  // 20260402 ZJH 0=翻转, 1=翻转+旋转, 2=全部
            bHFlip = true;                                // 20260402 ZJH 水平翻转始终开启
            bVFlip = (nMode >= 1);                        // 20260402 ZJH 模式1+: 垂直翻转
            bRotate90 = (nMode >= 1);                     // 20260402 ZJH 模式1+: 90度旋转
            bMultiScale = (nMode >= 2);                   // 20260402 ZJH 模式2: 多尺度
        }
        bridgeResult = m_pInferEngine->inferWithTTA(vecData, 3, m_nModelInputSize, m_nModelInputSize,
                                                     bHFlip, bVFlip, bRotate90, bMultiScale);
    } else {
        bridgeResult = m_pInferEngine->infer(vecData);  // 20260324 ZJH 标准推理
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    double dMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    // 20260324 ZJH 填充结果结构体
    result.nPredictedClass = bridgeResult.nPredictedClass;
    result.fConfidence = bridgeResult.fConfidence;
    result.dLatencyMs = dMs;

    // 20260324 ZJH 复制各类别概率
    result.vecProbs.resize(static_cast<int>(bridgeResult.vecProbs.size()));
    for (int i = 0; i < result.vecProbs.size(); ++i) {
        result.vecProbs[i] = bridgeResult.vecProbs[static_cast<size_t>(i)];
    }

    // 20260330 ZJH 缺陷判定：用 UI 可调阈值替代硬编码（对标 MVTec classification_threshold / 海康 MinScore）
    {
        float fConfThresh = m_pSpnConfThreshold ? static_cast<float>(m_pSpnConfThreshold->value()) : 0.50f;
        result.bIsDefect = (result.nPredictedClass != 0) && (result.fConfidence >= fConfThresh);
    }

    // 20260325 ZJH 从异常热力图生成二值化缺陷图（黑底白斑点）
    if (!bridgeResult.vecAnomalyMap.empty() && bridgeResult.nMapW > 0 && bridgeResult.nMapH > 0) {
        int nMapW = bridgeResult.nMapW;
        int nMapH = bridgeResult.nMapH;
        size_t nTotal = static_cast<size_t>(nMapW * nMapH);

        // 20260329 ZJH ===== 概率直映射 + 低概率截断 =====
        // P(defect) 直接映射灰度，不做动态拉伸（防止正常图被放大成缺陷）
        // P < 0.05 截断为黑（正常区域），P >= 0.05 线性映射到 0-255
        const float* pMap = bridgeResult.vecAnomalyMap.data();
        // 20260330 ZJH 从 UI 读取后处理参数（替代硬编码，对标 MVTec/海康可调参数）
        float fFloor = m_pSpnProbFloor ? static_cast<float>(m_pSpnProbFloor->value()) : 0.05f;
        int nMinAreaPx = m_pSpnMinArea ? m_pSpnMinArea->value() : 0;

        // 20260330 ZJH ===== 高斯模糊平滑异常图（消除 CNN 输出的棋盘格伪影）=====
        // 3x3 Gaussian kernel: [1,2,1; 2,4,2; 1,2,1] / 16
        // 不依赖 OpenCV，手写 separable 卷积（先水平后垂直，等价于 2D 高斯）
        // 输入: pMap (原始浮点概率图)，输出: vecSmoothed (平滑后概率图)
        std::vector<float> vecSmoothed(nTotal);  // 20260330 ZJH 平滑后的概率缓冲区
        {
            // 20260330 ZJH 1D 高斯核 [1, 2, 1] / 4（separable 分解）
            // 水平方向卷积结果暂存
            std::vector<float> vecHoriz(nTotal);  // 20260330 ZJH 水平卷积中间结果
            for (int y = 0; y < nMapH; ++y) {
                for (int x = 0; x < nMapW; ++x) {
                    // 20260330 ZJH 边界处理：clamp 到 [0, nMapW-1]，避免越界
                    int nLeft  = (x > 0) ? (x - 1) : 0;               // 20260330 ZJH 左邻像素（边界 clamp）
                    int nRight = (x < nMapW - 1) ? (x + 1) : (nMapW - 1);  // 20260330 ZJH 右邻像素（边界 clamp）
                    // 20260330 ZJH [1, 2, 1] / 4 水平卷积
                    vecHoriz[y * nMapW + x] = (pMap[y * nMapW + nLeft]
                                             + 2.0f * pMap[y * nMapW + x]
                                             + pMap[y * nMapW + nRight]) * 0.25f;
                }
            }
            // 20260330 ZJH 垂直方向卷积（对水平结果再做 [1, 2, 1] / 4）
            for (int y = 0; y < nMapH; ++y) {
                // 20260330 ZJH 上下邻行索引，边界 clamp
                int nUp   = (y > 0) ? (y - 1) : 0;
                int nDown = (y < nMapH - 1) ? (y + 1) : (nMapH - 1);
                for (int x = 0; x < nMapW; ++x) {
                    // 20260330 ZJH [1, 2, 1] / 4 垂直卷积，完成 3x3 高斯
                    vecSmoothed[y * nMapW + x] = (vecHoriz[nUp * nMapW + x]
                                                + 2.0f * vecHoriz[y * nMapW + x]
                                                + vecHoriz[nDown * nMapW + x]) * 0.25f;
                }
            }
        }

        // 20260330 ZJH ===== Otsu 自适应阈值计算（替代固定 fFloor）=====
        // 构建 256-bin 直方图，最大化类间方差（Otsu 1979）
        // 当 fFloor <= 0 时启用 Otsu 自动阈值；fFloor > 0 则沿用用户指定的固定阈值
        if (fFloor <= 0.0f) {
            // 20260330 ZJH 构建 256 级直方图（将 [0,1] 概率映射到 [0,255] bin）
            std::vector<int> vecHist(256, 0);  // 20260330 ZJH 直方图计数数组
            for (size_t i = 0; i < nTotal; ++i) {
                // 20260330 ZJH 概率 clamp 到 [0,1] 后映射到 bin 索引
                int nBin = static_cast<int>(std::min(std::max(vecSmoothed[i], 0.0f), 1.0f) * 255.0f);
                nBin = std::min(nBin, 255);  // 20260330 ZJH 防止浮点精度溢出
                vecHist[nBin]++;
            }
            // 20260330 ZJH Otsu 算法：遍历所有阈值 t，找到使类间方差 σ²_B 最大的 t
            float fBestSigma = -1.0f;  // 20260330 ZJH 当前最大类间方差
            int nBestThresh = 0;        // 20260330 ZJH 最优阈值 bin
            float fTotalPixels = static_cast<float>(nTotal);  // 20260330 ZJH 总像素数
            // 20260330 ZJH 计算全局加权灰度总和（用于快速递推类间方差）
            float fSumAll = 0.0f;  // 20260330 ZJH Σ(i * hist[i])
            for (int i = 0; i < 256; ++i) {
                fSumAll += static_cast<float>(i) * static_cast<float>(vecHist[i]);
            }
            float fSumBg = 0.0f;   // 20260330 ZJH 背景类累积加权和
            float fWeightBg = 0.0f; // 20260330 ZJH 背景类累积像素数
            for (int t = 0; t < 256; ++t) {
                fWeightBg += static_cast<float>(vecHist[t]);  // 20260330 ZJH 背景像素累加
                if (fWeightBg < 1.0f) continue;  // 20260330 ZJH 背景类为空，跳过
                float fWeightFg = fTotalPixels - fWeightBg;  // 20260330 ZJH 前景像素数
                if (fWeightFg < 1.0f) break;  // 20260330 ZJH 前景类为空，后续全空，提前终止
                fSumBg += static_cast<float>(t) * static_cast<float>(vecHist[t]);  // 20260330 ZJH 背景加权和递推
                float fMeanBg = fSumBg / fWeightBg;           // 20260330 ZJH 背景类均值
                float fMeanFg = (fSumAll - fSumBg) / fWeightFg;  // 20260330 ZJH 前景类均值
                // 20260330 ZJH 类间方差 σ²_B = w_bg * w_fg * (μ_bg - μ_fg)²
                float fSigma = fWeightBg * fWeightFg * (fMeanBg - fMeanFg) * (fMeanBg - fMeanFg);
                if (fSigma > fBestSigma) {
                    fBestSigma = fSigma;  // 20260330 ZJH 更新最优方差
                    nBestThresh = t;       // 20260330 ZJH 更新最优阈值
                }
            }
            // 20260330 ZJH 将 bin 阈值转回 [0,1] 概率值
            fFloor = static_cast<float>(nBestThresh) / 255.0f;
            // 20260330 ZJH Otsu 阈值下限保护：防止全黑图得到过低阈值
            if (fFloor < 0.02f) fFloor = 0.02f;
        }

        // 20260330 ZJH ===== 概率→灰度映射（使用平滑后的概率图）=====
        QImage imgSmall(nMapW, nMapH, QImage::Format_Grayscale8);
        for (int y = 0; y < nMapH; ++y) {
            uchar* pRow = imgSmall.scanLine(y);
            for (int x = 0; x < nMapW; ++x) {
                float fProb = vecSmoothed[y * nMapW + x];  // 20260330 ZJH 使用高斯平滑后的概率
                // 20260329 ZJH 低于 floor 截黑，高于 floor 线性映射到 0-255
                int nVal = (fProb > fFloor)
                    ? static_cast<int>((fProb - fFloor) / (1.0f - fFloor) * 255.0f)
                    : 0;
                pRow[x] = static_cast<uchar>(std::min(nVal, 255));
            }
        }

        // 20260330 ZJH ===== 连通域分析 + 面积过滤 + 度量提取 =====
        // BFS flood-fill 标记连通域，提取面积/质心/外接矩形/概率统计/圆度
        // 小于 nMinAreaPx 的区域清零（噪点过滤），存活区域存入 vecDefectRegions
        {
            // 20260330 ZJH 连通域标签图（0 = 未标记）
            std::vector<int> vecLabel(nTotal, 0);
            int nNextLabel = 1;  // 20260330 ZJH 下一个可用标签编号
            // 20260330 ZJH 存储所有连通域的度量和像素列表
            struct RegionData {
                int nArea = 0;            // 20260330 ZJH 像素面积
                double dSumX = 0.0;       // 20260330 ZJH x 坐标累加（用于质心计算）
                double dSumY = 0.0;       // 20260330 ZJH y 坐标累加
                int nMinX = INT_MAX;      // 20260330 ZJH 外接矩形左边界
                int nMaxX = 0;            // 20260330 ZJH 外接矩形右边界
                int nMinY = INT_MAX;      // 20260330 ZJH 外接矩形上边界
                int nMaxY = 0;            // 20260330 ZJH 外接矩形下边界
                double dSumProb = 0.0;    // 20260330 ZJH 概率累加（用于均值计算）
                float fMaxProb = 0.0f;    // 20260330 ZJH 区域内最大概率
                std::vector<std::pair<int,int>> vecPixels;  // 20260330 ZJH 区域内所有像素坐标
            };
            std::vector<RegionData> vecRegions;  // 20260330 ZJH 所有连通域数据

            for (int y = 0; y < nMapH; ++y) {
                for (int x = 0; x < nMapW; ++x) {
                    // 20260330 ZJH 跳过已标记像素和黑色像素
                    if (imgSmall.scanLine(y)[x] == 0 || vecLabel[y * nMapW + x] != 0) continue;

                    // 20260330 ZJH BFS flood fill 标记当前连通域
                    RegionData region;  // 20260330 ZJH 当前连通域度量累加器
                    std::vector<std::pair<int,int>> vecStack;  // 20260330 ZJH BFS 工作栈
                    vecStack.push_back({x, y});
                    vecLabel[y * nMapW + x] = nNextLabel;  // 20260330 ZJH 标记起始像素
                    while (!vecStack.empty()) {
                        auto [cx, cy] = vecStack.back(); vecStack.pop_back();  // 20260330 ZJH 弹出当前像素
                        // 20260330 ZJH 累加度量
                        region.vecPixels.push_back({cx, cy});
                        region.nArea++;
                        region.dSumX += cx;  // 20260330 ZJH 质心 x 累加
                        region.dSumY += cy;  // 20260330 ZJH 质心 y 累加
                        // 20260330 ZJH 更新外接矩形边界
                        if (cx < region.nMinX) region.nMinX = cx;
                        if (cx > region.nMaxX) region.nMaxX = cx;
                        if (cy < region.nMinY) region.nMinY = cy;
                        if (cy > region.nMaxY) region.nMaxY = cy;
                        // 20260330 ZJH 概率统计（从平滑后概率图读取，而非灰度图）
                        float fProb = vecSmoothed[cy * nMapW + cx];
                        region.dSumProb += fProb;
                        if (fProb > region.fMaxProb) region.fMaxProb = fProb;
                        // 20260330 ZJH 四邻域扩展
                        const int dx[] = {-1, 1, 0, 0};
                        const int dy[] = {0, 0, -1, 1};
                        for (int d = 0; d < 4; ++d) {
                            int nx = cx + dx[d], ny = cy + dy[d];
                            // 20260330 ZJH 边界检查 + 未标记 + 非黑
                            if (nx >= 0 && nx < nMapW && ny >= 0 && ny < nMapH
                                && vecLabel[ny * nMapW + nx] == 0
                                && imgSmall.scanLine(ny)[nx] > 0) {
                                vecLabel[ny * nMapW + nx] = nNextLabel;  // 20260330 ZJH 标记邻居
                                vecStack.push_back({nx, ny});
                            }
                        }
                    }
                    vecRegions.push_back(std::move(region));  // 20260330 ZJH 存储当前连通域
                    ++nNextLabel;
                }
            }

            // 20260330 ZJH 遍历所有连通域：面积过滤 + 度量提取
            result.vecDefectRegions.clear();
            for (auto& region : vecRegions) {
                // 20260330 ZJH 面积不足 → 清零该区域像素（噪点过滤）
                if (nMinAreaPx > 0 && region.nArea < nMinAreaPx) {
                    for (auto [rx, ry] : region.vecPixels) {
                        imgSmall.scanLine(ry)[rx] = 0;  // 20260330 ZJH 直接写 scanLine，比 setPixelColor 快
                    }
                    continue;  // 20260330 ZJH 跳过该区域，不加入结果
                }

                // 20260330 ZJH ===== 周长计算：统计边界像素数 =====
                // 边界像素 = 至少有一个四邻域像素不属于同一连通域的像素
                int nPerimeter = 0;  // 20260330 ZJH 边界像素计数
                for (auto [px, py] : region.vecPixels) {
                    // 20260330 ZJH 检查四邻域是否有非本区域像素
                    bool bOnBoundary = false;
                    const int dx[] = {-1, 1, 0, 0};
                    const int dy[] = {0, 0, -1, 1};
                    for (int d = 0; d < 4; ++d) {
                        int nx = px + dx[d], ny = py + dy[d];
                        // 20260330 ZJH 越界视为边界，邻居灰度为 0 也视为边界
                        if (nx < 0 || nx >= nMapW || ny < 0 || ny >= nMapH
                            || imgSmall.scanLine(ny)[nx] == 0) {
                            bOnBoundary = true;
                            break;  // 20260330 ZJH 只需发现一个非区域邻居即可
                        }
                    }
                    if (bOnBoundary) ++nPerimeter;
                }

                // 20260330 ZJH 构建 DefectRegion 度量结构体
                DefectRegion defect;
                defect.nArea = region.nArea;
                // 20260330 ZJH 质心 = 累加坐标 / 面积
                defect.ptCentroid = QPointF(region.dSumX / region.nArea,
                                            region.dSumY / region.nArea);
                // 20260330 ZJH 外接矩形（QRect 用 top-left + size 构造）
                defect.rectBounding = QRect(region.nMinX, region.nMinY,
                                            region.nMaxX - region.nMinX + 1,
                                            region.nMaxY - region.nMinY + 1);
                // 20260330 ZJH 平均异常概率 = 概率总和 / 面积
                defect.fMeanProb = static_cast<float>(region.dSumProb / region.nArea);
                defect.fMaxProb = region.fMaxProb;
                // 20260330 ZJH 圆度 = 4*π*面积 / 周长²（周长为 0 时圆度设为 0）
                if (nPerimeter > 0) {
                    defect.fCircularity = static_cast<float>(
                        4.0 * M_PI * region.nArea / (static_cast<double>(nPerimeter) * nPerimeter));
                } else {
                    defect.fCircularity = 0.0f;  // 20260330 ZJH 单像素区域周长为 0，圆度无意义
                }
                result.vecDefectRegions.append(defect);  // 20260330 ZJH 加入结果向量
            }
        }

        // 20260401 ZJH ===== 生成逐像素类别 ID 图（分割 mask overlay 使用）=====
        // Phase 1 的 mask overlay 期望像素值 = 类别 ID（0=背景, 1=异物, 2=划痕, ...）
        // 使用 EngineBridge 返回的真实 argmax 结果（跨通道比较后的类别分配）
        if (!bridgeResult.vecArgmaxMap.empty()
            && bridgeResult.nMapW > 0 && bridgeResult.nMapH > 0) {
            // 20260401 ZJH 从 argmax 类别向量构建 QImage 类别图
            QImage imgClassMap(nMapW, nMapH, QImage::Format_Grayscale8);
            for (int y = 0; y < nMapH; ++y) {
                uchar* pRow = imgClassMap.scanLine(y);
                for (int x = 0; x < nMapW; ++x) {
                    pRow[x] = bridgeResult.vecArgmaxMap[static_cast<size_t>(y * nMapW + x)];
                }
            }
            // 20260401 ZJH 上采样类别图到原始图像尺寸（最近邻插值保持类别边界锐利）
            result.imgDefectMap = imgClassMap.scaled(
                img.width(), img.height(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
        } else {
            // 20260401 ZJH 回退：概率图（非分割模型，如异常检测）
            result.imgDefectMap = imgSmall.scaled(
                img.width(), img.height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        }
    }

    // 20260330 ZJH 将 EngineBridge 返回的检测框 (om::DetectionBox) 转换为 UI 层的 InferDetection
    // 坐标从模型输入尺寸缩放到原始图像尺寸
    if (!bridgeResult.vecDetections.empty()) {
        QStringList vecClassNames = classNameList();  // 20260330 ZJH 获取类别名称列表
        float fScaleX = static_cast<float>(img.width())  / static_cast<float>(m_nModelInputSize);  // 20260330 ZJH 水平缩放比
        float fScaleY = static_cast<float>(img.height()) / static_cast<float>(m_nModelInputSize);  // 20260330 ZJH 垂直缩放比
        for (const auto& det : bridgeResult.vecDetections) {
            InferDetection inferDet;
            // 20260330 ZJH 将 (x1,y1,x2,y2) 缩放到原图坐标并转为 QRectF
            inferDet.rect = QRectF(
                static_cast<qreal>(det.fX1 * fScaleX),
                static_cast<qreal>(det.fY1 * fScaleY),
                static_cast<qreal>((det.fX2 - det.fX1) * fScaleX),
                static_cast<qreal>((det.fY2 - det.fY1) * fScaleY));
            inferDet.nClassId = det.nClassId;  // 20260330 ZJH 类别 ID
            inferDet.fScore   = det.fScore;    // 20260330 ZJH 置信度
            // 20260330 ZJH 映射类别名称（越界时回退为 "类别 N"）
            if (det.nClassId >= 0 && det.nClassId < vecClassNames.size()) {
                inferDet.strLabel = vecClassNames[det.nClassId];
            } else {
                inferDet.strLabel = QStringLiteral("类别 %1").arg(det.nClassId);
            }
            result.vecDetections.append(inferDet);  // 20260330 ZJH 追加到结果
        }
    }

    } catch (const std::exception& e) {
        // 20260326 ZJH 捕获推理过程中的任何异常（tensor 运算、内存分配等）
        std::cerr << "[InspectionPage] inference exception: " << e.what() << std::endl;
    } catch (...) {
        // 20260326 ZJH 捕获未知异常，防止闪退
        std::cerr << "[InspectionPage] inference unknown exception" << std::endl;
    }

    return result;
}

// 20260324 ZJH 显示推理结果到右面板和图像查看器
void InspectionPage::displayInferResult(const InferResult& result)
{
    // 20260324 ZJH 获取类别名称列表
    QStringList vecClassNames = classNameList();

    // 20260324 ZJH 获取预测类别名称
    QString strClassName;
    if (result.nPredictedClass >= 0 && result.nPredictedClass < vecClassNames.size()) {
        strClassName = vecClassNames[result.nPredictedClass];
    } else if (result.nPredictedClass >= 0) {
        strClassName = QStringLiteral("类别 %1").arg(result.nPredictedClass);
    } else {
        strClassName = QStringLiteral("未知");
    }

    // 20260324 ZJH 更新预测类别标签（OK=绿色，NG=红色）
    if (m_pLblPredClass) {
        m_pLblPredClass->setText(strClassName);
        if (result.bIsDefect) {
            m_pLblPredClass->setStyleSheet(QStringLiteral(
                "QLabel { color: #ef4444; font-size: 16px; font-weight: bold; border: none; }"));
        } else {
            m_pLblPredClass->setStyleSheet(QStringLiteral(
                "QLabel { color: #22c55e; font-size: 16px; font-weight: bold; border: none; }"));
        }
    }

    // 20260324 ZJH 更新置信度标签
    if (m_pLblConfidence) {
        m_pLblConfidence->setText(QStringLiteral("%1%").arg(
            static_cast<double>(result.fConfidence) * 100.0, 0, 'f', 1));
    }

    // 20260324 ZJH 更新推理耗时标签
    if (m_pLblLatency) {
        m_pLblLatency->setText(QStringLiteral("%1 ms").arg(result.dLatencyMs, 0, 'f', 1));
    }

    // 20260324 ZJH 更新各类别概率列表
    if (m_pLblClassProbs) {
        QString strProbs;
        for (int i = 0; i < result.vecProbs.size(); ++i) {
            QString strName;
            if (i < vecClassNames.size()) {
                strName = vecClassNames[i];
            } else {
                strName = QStringLiteral("类别 %1").arg(i);
            }
            float fProb = result.vecProbs[i];
            // 20260324 ZJH 用条形图风格显示概率
            int nBarLen = static_cast<int>(fProb * 20.0f);
            QString strBar = QString(nBarLen, QChar(0x2588)) +   // 实心块
                             QString(20 - nBarLen, QChar(0x2591)); // 浅色块
            strProbs += QStringLiteral("%1: %2 %3%\n")
                .arg(strName, -8)
                .arg(strBar)
                .arg(static_cast<double>(fProb) * 100.0, 5, 'f', 1);
        }
        m_pLblClassProbs->setText(strProbs);
    }

    // 20260406 ZJH ===== Halcon 风格缺陷轮廓叠加（只画轮廓线，不填充区域）=====
    // 对标 Halcon dev_display: 原图 + 缺陷区域彩色轮廓线 + 类别图例
    // 不使用半透明填充（紫色覆盖整张图），只在缺陷边界画 2px 轮廓线
    if (!result.imgDefectMap.isNull() && m_pInferView) {
        QImage displayImage(result.strImagePath);
        if (!displayImage.isNull()) {
            displayImage = displayImage.convertToFormat(QImage::Format_ARGB32);

            // 20260406 ZJH 将缺陷类别图缩放到原图尺寸
            QImage defectScaled = result.imgDefectMap.scaled(
                displayImage.width(), displayImage.height(),
                Qt::IgnoreAspectRatio, Qt::FastTransformation)
                .convertToFormat(QImage::Format_Grayscale8);

            // 20260406 ZJH 类别颜色表（Halcon 风格高饱和色）
            struct ClassColor { quint8 r, g, b; };
            const ClassColor arrColors[] = {
                {  0,   0,   0},    // 20260406 ZJH 0=背景（不绘制）
                {  0, 255,   0},    // 20260406 ZJH 1=缺陷类1，绿色（Halcon 默认）
                {255,   0,   0},    // 20260406 ZJH 2=缺陷类2，红色
                {255, 255,   0},    // 20260406 ZJH 3=缺陷类3，黄色
                {  0, 255, 255},    // 20260406 ZJH 4=备用，青色
                {255, 128,   0},    // 20260406 ZJH 5=备用，橙色
                {128,   0, 255},    // 20260406 ZJH 6=备用，紫色
                {255, 255, 255},    // 20260406 ZJH 7=备用，白色
            };
            constexpr int nMaxColors = sizeof(arrColors) / sizeof(arrColors[0]);

            int nW = displayImage.width();
            int nH = displayImage.height();

            // 20260406 ZJH ===== 步骤1: Halcon 风格缺陷区域叠加（填充 + 轮廓线）=====
            // 缺陷区域: 半透明填充（alpha=100）让缺陷可见但不遮挡纹理
            // 边界像素: 不透明轮廓线（alpha=255, 2px）清晰勾勒边界
            QImage overlay(nW, nH, QImage::Format_ARGB32);
            overlay.fill(Qt::transparent);

            for (int y = 1; y < nH - 1; ++y) {
                const uchar* pPrevLine = defectScaled.constScanLine(y - 1);
                const uchar* pCurrLine = defectScaled.constScanLine(y);
                const uchar* pNextLine = defectScaled.constScanLine(y + 1);
                QRgb* pDstLine = reinterpret_cast<QRgb*>(overlay.scanLine(y));

                for (int x = 1; x < nW - 1; ++x) {
                    int nClass = pCurrLine[x];
                    if (nClass == 0) continue;  // 20260406 ZJH 背景跳过
                    int nIdx = (nClass < nMaxColors) ? nClass : 1;

                    // 20260406 ZJH 4-邻域边界检测
                    bool bIsBorder = (pPrevLine[x] != nClass) ||
                                     (pNextLine[x] != nClass) ||
                                     (pCurrLine[x - 1] != nClass) ||
                                     (pCurrLine[x + 1] != nClass);

                    if (bIsBorder) {
                        // 20260406 ZJH 边界: 不透明轮廓（Halcon 风格醒目边界）
                        pDstLine[x] = qRgba(arrColors[nIdx].r, arrColors[nIdx].g, arrColors[nIdx].b, 255);
                    } else {
                        // 20260406 ZJH 内部: 半透明填充（让小缺陷也清晰可见）
                        pDstLine[x] = qRgba(arrColors[nIdx].r, arrColors[nIdx].g, arrColors[nIdx].b, 100);
                    }
                }
            }

            // 20260406 ZJH 叠加到原图
            QPainter painter(&displayImage);
            painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
            painter.drawImage(0, 0, overlay);

            // 20260406 ZJH ===== 步骤2: 类别图例（左上角，Halcon 风格）=====
            QStringList vecNames = classNameList();
            QSet<int> setActiveClasses;
            for (int y = 0; y < nH; y += 4) {
                const uchar* pLine = defectScaled.constScanLine(y);
                for (int x = 0; x < nW; x += 4) {
                    if (pLine[x] > 0 && pLine[x] < nMaxColors) setActiveClasses.insert(pLine[x]);
                }
            }

            if (!setActiveClasses.isEmpty()) {
                int nLegendX = 12;
                int nLegendY = 12;  // 20260406 ZJH 左上角
                int nBoxSize = 14;

                QRect rectBg(nLegendX - 4, nLegendY - 4,
                             180, static_cast<int>(setActiveClasses.size()) * 22 + 8);
                painter.setPen(Qt::NoPen);
                painter.setBrush(QColor(0, 0, 0, 180));
                painter.drawRoundedRect(rectBg, 4, 4);

                QList<int> vecSorted = setActiveClasses.values();
                std::sort(vecSorted.begin(), vecSorted.end());
                painter.setFont(QFont(QStringLiteral("Microsoft YaHei"), 9));

                for (int nClassId : vecSorted) {
                    int nIdx = (nClassId < nMaxColors) ? nClassId : 1;
                    painter.setPen(Qt::NoPen);
                    painter.setBrush(QColor(arrColors[nIdx].r, arrColors[nIdx].g, arrColors[nIdx].b));
                    painter.drawRect(nLegendX, nLegendY, nBoxSize, nBoxSize);

                    QString strName = (nClassId - 1 >= 0 && nClassId - 1 < vecNames.size())
                        ? vecNames[nClassId - 1]
                        : QStringLiteral("类别 %1").arg(nClassId);
                    painter.setPen(Qt::white);
                    painter.drawText(nLegendX + nBoxSize + 6, nLegendY + nBoxSize - 2, strName);
                    nLegendY += 22;
                }
            }

            painter.end();
            m_pInferView->setImage(displayImage);
        } else {
            m_pInferView->setImage(result.imgDefectMap);
        }
    }

    // 20260401 ZJH ===== 缺陷连通域度量面板更新 =====
    // 将 DefectRegion 结构体中的面积、质心、外接矩、圆度等信息格式化显示到右面板
    if (m_pLblDefectMetrics) {
        if (!result.vecDefectRegions.isEmpty()) {
            QString strMetrics;
            int nRegionIdx = 0;
            for (const DefectRegion& region : result.vecDefectRegions) {
                ++nRegionIdx;
                strMetrics += QStringLiteral(
                    "--- 缺陷 #%1 ---\n"
                    "  面积: %2 px\n"
                    "  质心: (%3, %4)\n"
                    "  外接矩: [%5,%6 %7x%8]\n"
                    "  圆度: %9\n"
                    "  平均概率: %10%\n"
                    "  最大概率: %11%\n")
                    .arg(nRegionIdx)
                    .arg(region.nArea)
                    .arg(region.ptCentroid.x(), 0, 'f', 1)
                    .arg(region.ptCentroid.y(), 0, 'f', 1)
                    .arg(region.rectBounding.x())
                    .arg(region.rectBounding.y())
                    .arg(region.rectBounding.width())
                    .arg(region.rectBounding.height())
                    .arg(static_cast<double>(region.fCircularity), 0, 'f', 3)
                    .arg(static_cast<double>(region.fMeanProb) * 100.0, 0, 'f', 1)
                    .arg(static_cast<double>(region.fMaxProb) * 100.0, 0, 'f', 1);
            }
            m_pLblDefectMetrics->setText(strMetrics);
            m_pLblDefectMetrics->setVisible(true);
        } else {
            m_pLblDefectMetrics->setText(QStringLiteral("无缺陷区域"));
            m_pLblDefectMetrics->setVisible(true);
        }
    }

    // 20260330 ZJH 绘制检测框（目标检测任务：在原始图像上叠加彩色边界框 + 标签文本）
    if (!result.vecDetections.isEmpty() && m_pInferView) {
        // 20260330 ZJH 加载原始图像作为绘制底图（缺陷图为黑白，检测框需叠加在彩色原图上）
        QImage displayImage(result.strImagePath);
        if (!displayImage.isNull()) {
            // 20260330 ZJH 确保格式支持 QPainter 绘制（转为 ARGB32）
            displayImage = displayImage.convertToFormat(QImage::Format_ARGB32);
            QPainter painter(&displayImage);
            painter.setRenderHint(QPainter::Antialiasing);  // 20260330 ZJH 抗锯齿

            for (const auto& det : result.vecDetections) {
                // 20260330 ZJH 按类别 ID 分配颜色（HSV 色环均匀分布，饱和度 200，亮度 255）
                QColor color = QColor::fromHsv((det.nClassId * 60) % 360, 200, 255);

                // 20260330 ZJH 绘制边界框矩形（2px 彩色边框 + 半透明填充）
                QPen pen(color, 2);
                painter.setPen(pen);
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40));  // 20260330 ZJH alpha=40 半透明填充
                painter.drawRect(det.rect);

                // 20260330 ZJH 组装标签文本："类别名 置信度%"
                QString strText = QStringLiteral("%1 %2%").arg(det.strLabel)
                    .arg(static_cast<double>(det.fScore) * 100.0, 0, 'f', 1);

                // 20260330 ZJH 标签背景色块（框顶部上方 18px 高的色带）
                QRectF rectLabel(det.rect.left(), det.rect.top() - 18.0,
                                 strText.length() * 8.0 + 8.0, 18.0);
                painter.setPen(Qt::NoPen);
                painter.setBrush(color);
                painter.drawRect(rectLabel);

                // 20260330 ZJH 绘制白色标签文字
                painter.setPen(Qt::white);
                painter.drawText(static_cast<int>(det.rect.left()) + 4,
                                 static_cast<int>(det.rect.top()) - 4, strText);
            }
            painter.end();  // 20260330 ZJH 结束绘制

            // 20260330 ZJH 将带检测框的图像显示到查看器
            m_pInferView->setImage(displayImage);
        }
    }

    // 20260324 ZJH 更新图像查看器上的叠加标签
    if (m_pLblOverlayResult) {
        QString strOverlay = QStringLiteral("%1 | %2%  | %3ms")
            .arg(strClassName)
            .arg(static_cast<double>(result.fConfidence) * 100.0, 0, 'f', 1)
            .arg(result.dLatencyMs, 0, 'f', 1);
        m_pLblOverlayResult->setText(strOverlay);

        // 20260324 ZJH 根据结果设置叠加标签颜色
        if (result.bIsDefect) {
            m_pLblOverlayResult->setStyleSheet(QStringLiteral(
                "QLabel {"
                "  background-color: rgba(239, 68, 68, 200);"
                "  color: white;"
                "  font-size: 14px; font-weight: bold;"
                "  padding: 8px 12px; border-radius: 6px; border: none;"
                "}"));
        } else {
            m_pLblOverlayResult->setStyleSheet(QStringLiteral(
                "QLabel {"
                "  background-color: rgba(34, 197, 94, 200);"
                "  color: white;"
                "  font-size: 14px; font-weight: bold;"
                "  padding: 8px 12px; border-radius: 6px; border: none;"
                "}"));
        }
        m_pLblOverlayResult->adjustSize();
        m_pLblOverlayResult->setVisible(true);
    }
}

// 20260324 ZJH 显示批量推理汇总结果
void InspectionPage::displayBatchSummary()
{
    if (m_vecResults.isEmpty()) {
        return;
    }

    // 20260324 ZJH 统计 OK/NG 数量和平均耗时
    int nOkCount = 0;
    int nNgCount = 0;
    double dTotalMs = 0.0;

    for (const InferResult& result : m_vecResults) {
        if (result.bIsDefect) {
            ++nNgCount;
        } else {
            ++nOkCount;
        }
        dTotalMs += result.dLatencyMs;
    }

    double dAvgMs = dTotalMs / m_vecResults.size();

    // 20260324 ZJH 弹出汇总对话框
    QString strSummary = QStringLiteral(
        "批量推理完成！\n\n"
        "总数量:   %1 张\n"
        "OK (正常):  %2 张\n"
        "NG (缺陷):  %3 张\n\n"
        "平均耗时: %4 ms/张\n"
        "总耗时:   %5 ms")
        .arg(m_vecResults.size())
        .arg(nOkCount)
        .arg(nNgCount)
        .arg(dAvgMs, 0, 'f', 1)
        .arg(dTotalMs, 0, 'f', 1);

    QMessageBox::information(this,
        QStringLiteral("批量推理结果"), strSummary);
}

// 20260324 ZJH 导航到指定索引的测试图像
void InspectionPage::navigateToTestImage(int nIndex)
{
    if (nIndex < 0 || nIndex >= m_vecTestImagePaths.size()) {
        return;
    }

    m_nCurrentTestIndex = nIndex;

    // 20260324 ZJH 更新索引标签
    if (m_pLblImageIndex) {
        m_pLblImageIndex->setText(QStringLiteral("%1 / %2")
            .arg(nIndex + 1).arg(m_vecTestImagePaths.size()));
    }

    // 20260324 ZJH 加载并显示图像
    QString strImagePath = m_vecTestImagePaths[nIndex];
    QImage img(strImagePath);
    if (!img.isNull()) {
        m_pInferView->setImage(img);
        m_pInferView->fitInView();
    }

    // 20260324 ZJH 同步选中测试图像列表中的对应项
    if (m_pTestImageList && m_pTestImageModel) {
        QModelIndex modelIdx = m_pTestImageModel->index(nIndex, 0);
        m_pTestImageList->blockSignals(true);  // 20260324 ZJH 防止选中触发循环
        m_pTestImageList->setCurrentIndex(modelIdx);
        m_pTestImageList->blockSignals(false);
    }

    // 20260324 ZJH 如果有缓存的推理结果，显示之
    if (nIndex < m_vecResults.size() && m_vecResults[nIndex].nPredictedClass >= 0) {
        displayInferResult(m_vecResults[nIndex]);
    } else {
        // 20260324 ZJH 清空推理结果显示
        if (m_pLblPredClass) {
            m_pLblPredClass->setText(QStringLiteral("--"));
            m_pLblPredClass->setStyleSheet(QStringLiteral(
                "QLabel { color: #e2e8f0; font-size: 16px; font-weight: bold; border: none; }"));
        }
        if (m_pLblConfidence) m_pLblConfidence->setText(QStringLiteral("--"));
        if (m_pLblLatency)    m_pLblLatency->setText(QStringLiteral("--"));
        if (m_pLblClassProbs) m_pLblClassProbs->setText(QStringLiteral(""));
        if (m_pLblOverlayResult) m_pLblOverlayResult->setVisible(false);
    }

    // 20260324 ZJH 更新推理按钮状态
    updateInferenceButtons();
}

// 20260324 ZJH 获取当前项目的类别名称列表
QStringList InspectionPage::classNameList() const
{
    QStringList vecNames;

    // 20260324 ZJH 优先从项目数据集获取标签名称
    if (m_pDataset) {
        const QVector<LabelInfo>& vecLabels = m_pDataset->labels();
        for (const LabelInfo& label : vecLabels) {
            vecNames.append(label.strName);
        }
    }

    // 20260324 ZJH 如果没有项目或标签为空，生成默认名称
    if (vecNames.isEmpty() && m_nModelNumClasses > 0) {
        // 20260324 ZJH 二分类默认 OK/NG，多分类用类别编号
        if (m_nModelNumClasses == 2) {
            vecNames << QStringLiteral("OK") << QStringLiteral("NG");
        } else {
            for (int i = 0; i < m_nModelNumClasses; ++i) {
                vecNames << QStringLiteral("类别 %1").arg(i);
            }
        }
    }

    return vecNames;
}
