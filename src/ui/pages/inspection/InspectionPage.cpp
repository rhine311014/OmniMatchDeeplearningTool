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
#include <QApplication>         // 20260324 ZJH processEvents 保持 UI 响应
#include <QImage>               // 20260324 ZJH 图像加载和预处理
#include <QtConcurrent>         // 20260325 ZJH 后台线程加载模型
#include <QFutureWatcher>       // 20260325 ZJH 监听后台任务完成
#include <QProgressDialog>      // 20260325 ZJH 加载进度对话框
#include <chrono>               // 20260324 ZJH 推理计时
#include <vector>               // 20260324 ZJH 推理输入数据
#include <algorithm>            // 20260329 ZJH nth_element 百分位阈值
#include <queue>                // 20260329 ZJH BFS 连通域面积过滤
#include <iostream>             // 20260326 ZJH std::cerr 异常日志输出

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
    : BasePage(pParent)
    , m_pCmbLabelFilter(nullptr)
    , m_pCmbSplitFilter(nullptr)
    , m_pCmbStatusFilter(nullptr)
    , m_pLblFilteredCount(nullptr)
    , m_pLblLabelStats(nullptr)
    , m_pListView(nullptr)
    , m_pModel(nullptr)
    , m_pProxyModel(nullptr)
    , m_pDelegate(nullptr)
    , m_pPreviewLabel(nullptr)
    , m_pLblFileName(nullptr)
    , m_pLblDimensions(nullptr)
    , m_pLblLabelName(nullptr)
    , m_pLblSplitType(nullptr)
    , m_pBtnQuickLabel(nullptr)
    , m_pCmbMode(nullptr)
    , m_pCenterStack(nullptr)
    , m_pRightStack(nullptr)
    , m_pInferView(nullptr)
    , m_pGradCAMOverlay(nullptr)
    , m_pLblOverlayResult(nullptr)
    , m_pBtnLoadModel(nullptr)
    , m_pBtnUnloadModel(nullptr)
    , m_pLblModelStatus(nullptr)
    , m_pCmbModelArch(nullptr)
    , m_pSpnInputSize(nullptr)
    , m_pSpnNumClasses(nullptr)
    , m_pBtnImportImages(nullptr)
    , m_pBtnImportFolder(nullptr)
    , m_pBtnClearImages(nullptr)
    , m_pLblTestImageCount(nullptr)
    , m_pTestImageList(nullptr)
    , m_pTestImageModel(nullptr)
    , m_pBtnRunInference(nullptr)
    , m_pBtnBatchInference(nullptr)
    , m_pProgressBar(nullptr)
    , m_pBtnPrevImage(nullptr)
    , m_pBtnNextImage(nullptr)
    , m_pLblImageIndex(nullptr)
    , m_pLblPredClass(nullptr)
    , m_pLblConfidence(nullptr)
    , m_pLblLatency(nullptr)
    , m_pLblClassProbs(nullptr)
    , m_pDataset(nullptr)
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
    refreshModel();
    updateStatistics();

    // 20260324 ZJH 项目加载后自动填充模型配置（从 TrainingConfig 读取）
    if (m_pProject) {
        const TrainingConfig& config = m_pProject->trainingConfig();
        // 20260324 ZJH 设置输入尺寸
        if (m_pSpnInputSize) {
            m_pSpnInputSize->setValue(config.nInputSize);
        }
        // 20260324 ZJH 设置类别数（从数据集标签数量推断）
        if (m_pSpnNumClasses && m_pDataset) {
            int nLabelCount = m_pDataset->labels().size();
            if (nLabelCount > 0) {
                m_pSpnNumClasses->setValue(nLabelCount);
            }
        }
        // 20260324 ZJH 设置模型架构选择
        if (m_pCmbModelArch) {
            QString strArchName = om::modelArchitectureToString(config.eArchitecture);
            int nIdx = m_pCmbModelArch->findText(strArchName);
            if (nIdx >= 0) {
                m_pCmbModelArch->setCurrentIndex(nIdx);
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
        // 20260324 ZJH 通过 notify 方法发射信号，避免外部直接 emit
        Application::instance()->notifyOpenImage(strUuid);
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

// 20260324 ZJH 加载模型文件（.dfm）
// 20260325 ZJH 重写为异步加载：后台线程创建模型+加载权重，避免 UI 卡死
void InspectionPage::onLoadModel()
{
    // 20260324 ZJH 打开文件选择对话框
    QString strPath = QFileDialog::getOpenFileName(this,
        QStringLiteral("选择模型文件"), QString(),
        QStringLiteral("OmniMatch 模型 (*.dfm);;所有文件 (*)"));
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
        m_bModelLoaded = true;
        m_strModelPath = strPath;
        m_nModelInputSize = nInputSize;
        m_nModelNumClasses = nNumClasses;
        m_strModelArchName = strArchName;

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
    m_pInferEngine.reset();
    m_bModelLoaded = false;
    m_strModelPath.clear();
    m_strModelArchName.clear();
    m_nModelNumClasses = 0;

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

    // 20260324 ZJH 清空结果显示
    if (m_pLblPredClass)  m_pLblPredClass->setText(QStringLiteral("--"));
    if (m_pLblConfidence) m_pLblConfidence->setText(QStringLiteral("--"));
    if (m_pLblLatency)    m_pLblLatency->setText(QStringLiteral("--"));
    if (m_pLblClassProbs) m_pLblClassProbs->setText(QStringLiteral(""));
    if (m_pLblImageIndex) m_pLblImageIndex->setText(QStringLiteral("-- / --"));
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

    // 20260322 ZJH 各标签数量列表
    m_pLblLabelStats = new QLabel(pStatsGroup);
    m_pLblLabelStats->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 11px; border: none; }"));
    m_pLblLabelStats->setWordWrap(true);  // 20260322 ZJH 自动换行
    m_pLblLabelStats->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    pStatsLayout->addWidget(m_pLblLabelStats);

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

    // ===== 3. 推理控制分组 =====

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

    // 20260322 ZJH 遍历数据集中的所有图像条目
    const QVector<ImageEntry>& vecImages = m_pDataset->images();
    for (const ImageEntry& entry : vecImages) {
        QStandardItem* pItem = new QStandardItem();

        // 20260322 ZJH 设置自定义数据角色
        pItem->setData(entry.strFilePath,  FilePathRole);    // 20260322 ZJH 文件路径
        pItem->setData(entry.strUuid,      UuidRole);        // 20260322 ZJH UUID
        pItem->setData(entry.nLabelId,     LabelIdRole);     // 20260322 ZJH 标签 ID
        pItem->setData(entry.fileName(),   FileNameRole);    // 20260322 ZJH 文件名
        pItem->setData(static_cast<int>(entry.eSplit), SplitTypeRole);  // 20260322 ZJH 拆分类型

        // 20260322 ZJH 查找标签颜色和名称
        if (entry.nLabelId >= 0) {
            const LabelInfo* pLabel = m_pDataset->findLabel(entry.nLabelId);
            if (pLabel) {
                pItem->setData(pLabel->color,   LabelColorRole);  // 20260322 ZJH 标签颜色
                pItem->setData(pLabel->strName, LabelNameRole);   // 20260322 ZJH 标签名称
            }
        }

        m_pModel->appendRow(pItem);  // 20260322 ZJH 添加到模型
    }
}

// 20260322 ZJH 更新统计标签显示
void InspectionPage::updateStatistics()
{
    if (!m_pLblFilteredCount || !m_pLblLabelStats) {
        return;
    }

    // 20260322 ZJH 显示过滤后的图像数量
    int nFilteredCount = m_pProxyModel ? m_pProxyModel->rowCount() : 0;
    m_pLblFilteredCount->setText(QStringLiteral("当前显示: %1 张图像").arg(nFilteredCount));

    // 20260322 ZJH 构建各标签数量统计文本
    QString strStats;
    if (m_pDataset) {
        // 20260322 ZJH 获取标签分布
        QMap<int, int> mapDist = m_pDataset->labelDistribution();
        const QVector<LabelInfo>& vecLabels = m_pDataset->labels();

        for (const LabelInfo& label : vecLabels) {
            int nCount = mapDist.value(label.nId, 0);  // 20260322 ZJH 获取该标签的图像数
            strStats += QStringLiteral("%1: %2 张\n")
                .arg(label.strName)
                .arg(nCount);
        }

        // 20260322 ZJH 未标注图像数
        int nUnlabeled = m_pDataset->unlabeledCount();
        strStats += QStringLiteral("未标注: %1 张").arg(nUnlabeled);
    }

    m_pLblLabelStats->setText(strStats);
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

    // 20260322 ZJH 添加 "全部" 选项
    m_pCmbLabelFilter->addItem(QStringLiteral("全部"), -1);

    if (m_pDataset) {
        // 20260322 ZJH 添加各标签选项
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

        // 20260324 ZJH 更新按钮状态
        if (m_pBtnLoadModel)   m_pBtnLoadModel->setEnabled(false);
        if (m_pBtnUnloadModel) m_pBtnUnloadModel->setEnabled(true);
        if (m_pCmbModelArch)   m_pCmbModelArch->setEnabled(false);
        if (m_pSpnInputSize)   m_pSpnInputSize->setEnabled(false);
        if (m_pSpnNumClasses)  m_pSpnNumClasses->setEnabled(false);
    } else {
        // 20260324 ZJH 显示未加载状态
        m_pLblModelStatus->setText(QStringLiteral("未加载"));
        m_pLblModelStatus->setStyleSheet(QStringLiteral(
            "QLabel { color: #f97316; font-size: 11px; font-weight: bold; border: none; }"));

        // 20260324 ZJH 更新按钮状态
        if (m_pBtnLoadModel)   m_pBtnLoadModel->setEnabled(true);
        if (m_pBtnUnloadModel) m_pBtnUnloadModel->setEnabled(false);
        if (m_pCmbModelArch)   m_pCmbModelArch->setEnabled(true);
        if (m_pSpnInputSize)   m_pSpnInputSize->setEnabled(true);
        if (m_pSpnNumClasses)  m_pSpnNumClasses->setEnabled(true);
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
    auto tStart = std::chrono::high_resolution_clock::now();
    BridgeInferResult bridgeResult = m_pInferEngine->infer(vecData);
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

    // 20260324 ZJH 判断是否为缺陷（简单策略：类别0=OK，其他=NG，或置信度低于50%）
    result.bIsDefect = (result.nPredictedClass != 0);

    // 20260325 ZJH 从异常热力图生成二值化缺陷图（黑底白斑点）
    if (!bridgeResult.vecAnomalyMap.empty() && bridgeResult.nMapW > 0 && bridgeResult.nMapH > 0) {
        int nMapW = bridgeResult.nMapW;
        int nMapH = bridgeResult.nMapH;
        size_t nTotal = static_cast<size_t>(nMapW * nMapH);

        // 20260329 ZJH ===== 自适应百分位阈值：仅 top 2% 像素标为缺陷 =====
        // vecAnomalyMap 已经是 [0,1] 范围的 P(defect)
        const float* pMap = bridgeResult.vecAnomalyMap.data();

        // 20260329 ZJH 取第 98 百分位值作为二值化阈值（nth_element O(n) 部分排序）
        std::vector<float> vecSorted(pMap, pMap + nTotal);  // 20260329 ZJH 副本，不改原数据
        size_t n98 = static_cast<size_t>(nTotal * 0.98);  // 20260329 ZJH 第98百分位索引
        if (n98 >= nTotal) n98 = nTotal - 1;  // 20260329 ZJH 防越界
        std::nth_element(vecSorted.begin(), vecSorted.begin() + static_cast<ptrdiff_t>(n98), vecSorted.end());
        float fThresh = vecSorted[n98];  // 20260329 ZJH P98 阈值
        // 20260329 ZJH 防止全图无缺陷时 P98 极小导致大面积误标：下限 0.1
        fThresh = std::max(fThresh, 0.1f);

        // 20260329 ZJH 二值化: P > P98 → 白色(255=缺陷)，否则 → 黑色(0=正常)
        QImage imgSmall(nMapW, nMapH, QImage::Format_Grayscale8);
        for (int y = 0; y < nMapH; ++y) {
            uchar* pRow = imgSmall.scanLine(y);
            for (int x = 0; x < nMapW; ++x) {
                float fProb = pMap[y * nMapW + x];  // 20260329 ZJH P(defect) 概率值
                pRow[x] = (fProb > fThresh) ? 255 : 0;  // 20260329 ZJH 自适应阈值二值化
            }
        }

        // 20260328 ZJH ===== 形态学开运算：去除孤立白色噪点 =====
        {
            // 20260328 ZJH Step 1: 腐蚀白色缺陷区域（3×3 最小值滤波 — 缩小白斑，去噪）
            QImage imgEroded(nMapW, nMapH, QImage::Format_Grayscale8);
            imgEroded.fill(0);  // 20260328 ZJH 默认黑色（正常）
            for (int y = 1; y < nMapH - 1; ++y) {
                uchar* pOut = imgEroded.scanLine(y);
                for (int x = 1; x < nMapW - 1; ++x) {
                    // 20260328 ZJH 3×3 邻域最小值：只要有黑色邻居就变黑，缩小白色缺陷区域
                    uchar nMin = 255;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const uchar* pSrc = imgSmall.constScanLine(y + dy);
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (pSrc[x + dx] < nMin) nMin = pSrc[x + dx];
                        }
                    }
                    pOut[x] = nMin;
                }
            }

            // 20260328 ZJH Step 2: 膨胀白色缺陷区域（3×3 最大值滤波 — 恢复缺陷大小）
            QImage imgDilated(nMapW, nMapH, QImage::Format_Grayscale8);
            imgDilated.fill(0);  // 20260328 ZJH 默认黑色（正常）
            for (int y = 1; y < nMapH - 1; ++y) {
                uchar* pOut = imgDilated.scanLine(y);
                for (int x = 1; x < nMapW - 1; ++x) {
                    // 20260328 ZJH 3×3 邻域最大值：只要有白色邻居就变白，恢复缺陷区域
                    uchar nMax = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const uchar* pSrc = imgEroded.constScanLine(y + dy);
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (pSrc[x + dx] > nMax) nMax = pSrc[x + dx];
                        }
                    }
                    pOut[x] = nMax;
                }
            }
            imgSmall = imgDilated;  // 20260328 ZJH 替换为形态学开运算后的图像
        }

        // 20260329 ZJH ===== 连通域面积过滤：去除过小的白色噪斑 =====
        // 等效于 Halcon Connection() + SelectShape("area", "and", minArea, 99999)
        if (nTotal > 64) {  // 20260329 ZJH 图像极小（如8×8=64）时跳过，避免误过滤
            // 20260329 ZJH BFS flood-fill 标记连通域（4-连通）
            std::vector<int> vecLabel(nTotal, 0);  // 20260329 ZJH 每像素的连通域ID，0=背景
            int nLabelCount = 0;  // 20260329 ZJH 连通域总数
            std::vector<int> vecArea;  // 20260329 ZJH 每个连通域的面积（索引=labelID-1）
            std::queue<int> queFlood;  // 20260329 ZJH BFS 队列（存线性索引 y*W+x）

            for (int i = 0; i < static_cast<int>(nTotal); ++i) {
                // 20260329 ZJH 跳过已标记像素和黑色背景像素
                if (vecLabel[i] != 0) continue;
                const uchar* pPixels = imgSmall.constBits();
                if (pPixels[i] == 0) continue;

                // 20260329 ZJH 发现新连通域，BFS 扩展
                ++nLabelCount;
                int nArea = 0;  // 20260329 ZJH 当前连通域面积
                vecLabel[i] = nLabelCount;
                queFlood.push(i);

                while (!queFlood.empty()) {
                    int nIdx = queFlood.front();
                    queFlood.pop();
                    ++nArea;

                    int cx = nIdx % nMapW;  // 20260329 ZJH 当前像素列坐标
                    int cy = nIdx / nMapW;  // 20260329 ZJH 当前像素行坐标

                    // 20260329 ZJH 4-连通邻域：上下左右
                    const int arrDx[4] = {-1, 1,  0, 0};
                    const int arrDy[4] = { 0, 0, -1, 1};
                    for (int d = 0; d < 4; ++d) {
                        int nx = cx + arrDx[d];  // 20260329 ZJH 邻居列坐标
                        int ny = cy + arrDy[d];  // 20260329 ZJH 邻居行坐标
                        if (nx < 0 || nx >= nMapW || ny < 0 || ny >= nMapH) continue;  // 20260329 ZJH 越界跳过
                        int nNeighIdx = ny * nMapW + nx;  // 20260329 ZJH 邻居线性索引
                        if (vecLabel[nNeighIdx] != 0) continue;  // 20260329 ZJH 已标记跳过
                        if (pPixels[nNeighIdx] == 0) continue;  // 20260329 ZJH 黑色背景跳过
                        vecLabel[nNeighIdx] = nLabelCount;  // 20260329 ZJH 标记为同一连通域
                        queFlood.push(nNeighIdx);
                    }
                }
                vecArea.push_back(nArea);  // 20260329 ZJH 记录该连通域面积
            }

            // 20260329 ZJH 面积阈值: 小于总像素 0.05% 的区域视为噪点清零
            int nMinArea = std::max(1, static_cast<int>(nTotal * 0.0005));
            for (int i = 0; i < static_cast<int>(nTotal); ++i) {
                int nLbl = vecLabel[i];  // 20260329 ZJH 当前像素的连通域ID
                if (nLbl <= 0) continue;  // 20260329 ZJH 背景跳过
                // 20260329 ZJH 面积不足阈值 → 清为黑色（正常）
                if (vecArea[static_cast<size_t>(nLbl - 1)] < nMinArea) {
                    imgSmall.scanLine(i / nMapW)[i % nMapW] = 0;
                }
            }
        }

        // 20260325 ZJH 上采样到原始图像尺寸（最近邻插值保持锐利边缘）
        result.imgDefectMap = imgSmall.scaled(
            img.width(), img.height(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
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

    // 20260325 ZJH 显示二值化缺陷图（纯黑白：黑底 + 白色异物斑点）
    if (!result.imgDefectMap.isNull() && m_pInferView) {
        m_pInferView->setImage(result.imgDefectMap);
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
