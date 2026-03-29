// 20260322 ZJH GalleryPage 实现
// 三栏布局图库浏览页面：统计/过滤 + 缩略图网格 + 拖放导入

#include "ui/pages/gallery/GalleryPage.h"       // 20260322 ZJH 类声明
#include "ui/widgets/ThumbnailDelegate.h"        // 20260322 ZJH 缩略图绘制代理
#include "core/data/ImageDataset.h"              // 20260322 ZJH 数据集管理
#include "core/data/ImageEntry.h"                // 20260322 ZJH 图像条目
#include "core/data/LabelInfo.h"                 // 20260322 ZJH 标签信息
#include "core/DLTypes.h"                        // 20260322 ZJH SplitType 枚举
#include "core/project/Project.h"                // 20260322 ZJH 项目数据
#include "core/project/ProjectSerializer.h"      // 20260324 ZJH 导入后自动保存
#include "app/Application.h"                     // 20260322 ZJH 全局事件总线
#include "ui/dialogs/LabelManagementDialog.h"    // 20260322 ZJH 标签管理对话框

#include <QHBoxLayout>          // 20260322 ZJH 水平布局
#include <QVBoxLayout>          // 20260322 ZJH 垂直布局
#include <QGroupBox>            // 20260322 ZJH 分组框
#include <QFileDialog>          // 20260322 ZJH 文件/文件夹选择对话框
#include <QMessageBox>          // 20260322 ZJH 消息对话框
#include <QMimeData>            // 20260322 ZJH 拖放数据
#include <QUrl>                 // 20260322 ZJH URL（拖放文件路径）
#include <QFileInfo>            // 20260322 ZJH 文件信息
#include <QDir>                 // 20260322 ZJH 目录操作
#include <QDragEnterEvent>      // 20260322 ZJH 拖放进入事件
#include <QDragMoveEvent>       // 20260322 ZJH 拖放移动事件
#include <QDropEvent>           // 20260322 ZJH 拖放释放事件
#include <QScrollBar>           // 20260322 ZJH 滚动条
#include <QStandardItem>        // 20260322 ZJH 标准条目
#include <QPixmap>              // 20260322 ZJH 位图（创建标签色块图标）
#include <QPainter>             // 20260322 ZJH 绘图器（绘制色块图标）
#include <QFrame>               // 20260322 ZJH 分割线控件

// ===================================================================
// GalleryFilterProxy 实现
// ===================================================================

// 20260322 ZJH 构造函数
GalleryFilterProxy::GalleryFilterProxy(QObject* pParent)
    : QSortFilterProxyModel(pParent)
{
    // 20260322 ZJH 默认不区分大小写搜索
    setFilterCaseSensitivity(Qt::CaseInsensitive);
}

// 20260322 ZJH 设置搜索关键词
void GalleryFilterProxy::setSearchText(const QString& strText)
{
    m_strSearchText = strText.trimmed();  // 20260322 ZJH 去除首尾空白
    beginFilterChange(); endFilterChange();                    // 20260322 ZJH 触发重新过滤
}

// 20260322 ZJH 设置标签过滤集合
void GalleryFilterProxy::setLabelFilter(const QSet<int>& setLabelIds)
{
    m_setLabelFilter = setLabelIds;
    beginFilterChange(); endFilterChange();
}

// 20260322 ZJH 设置拆分过滤集合
void GalleryFilterProxy::setSplitFilter(const QSet<int>& setSplitTypes)
{
    m_setSplitFilter = setSplitTypes;
    beginFilterChange(); endFilterChange();
}

// 20260322 ZJH 清除所有过滤条件
void GalleryFilterProxy::clearFilters()
{
    m_strSearchText.clear();
    m_setLabelFilter.clear();
    m_setSplitFilter.clear();
    beginFilterChange(); endFilterChange();
}

// 20260322 ZJH 过滤行逻辑：搜索关键词 AND 标签过滤 AND 拆分过滤
bool GalleryFilterProxy::filterAcceptsRow(int nSourceRow,
                                          const QModelIndex& sourceParent) const
{
    QModelIndex idx = sourceModel()->index(nSourceRow, 0, sourceParent);

    // 20260322 ZJH 1. 搜索关键词过滤（匹配文件名）
    if (!m_strSearchText.isEmpty()) {
        QString strFileName = idx.data(FileNameRole).toString();  // 20260322 ZJH 获取文件名
        if (!strFileName.contains(m_strSearchText, Qt::CaseInsensitive)) {
            return false;  // 20260322 ZJH 文件名不包含搜索词，过滤掉
        }
    }

    // 20260322 ZJH 2. 标签过滤（如果有设置过滤集合）
    if (!m_setLabelFilter.isEmpty()) {
        int nLabelId = idx.data(LabelIdRole).toInt();  // 20260322 ZJH 获取标签 ID
        if (!m_setLabelFilter.contains(nLabelId)) {
            return false;  // 20260322 ZJH 标签不在过滤集合中，过滤掉
        }
    }

    // 20260322 ZJH 3. 拆分过滤（如果有设置过滤集合）
    if (!m_setSplitFilter.isEmpty()) {
        int nSplitType = idx.data(SplitTypeRole).toInt();  // 20260322 ZJH 获取拆分类型
        if (!m_setSplitFilter.contains(nSplitType)) {
            return false;  // 20260322 ZJH 拆分类型不在过滤集合中，过滤掉
        }
    }

    return true;  // 20260322 ZJH 通过所有过滤条件
}

// 20260322 ZJH 自定义排序比较
bool GalleryFilterProxy::lessThan(const QModelIndex& left,
                                  const QModelIndex& right) const
{
    // 20260322 ZJH 根据排序角色选择比较方式
    int nSortRole = sortRole();

    if (nSortRole == FileNameRole) {
        // 20260322 ZJH 按文件名字母排序
        return left.data(FileNameRole).toString().toLower()
             < right.data(FileNameRole).toString().toLower();
    } else if (nSortRole == LabelIdRole) {
        // 20260322 ZJH 按标签 ID 排序
        return left.data(LabelIdRole).toInt() < right.data(LabelIdRole).toInt();
    } else if (nSortRole == SplitTypeRole) {
        // 20260322 ZJH 按拆分类型排序
        return left.data(SplitTypeRole).toInt() < right.data(SplitTypeRole).toInt();
    }

    // 20260322 ZJH 默认按文件名排序
    return left.data(FileNameRole).toString().toLower()
         < right.data(FileNameRole).toString().toLower();
}

// ===================================================================
// GalleryPage 实现
// ===================================================================

// 20260322 ZJH 构造函数，初始化三栏布局
GalleryPage::GalleryPage(QWidget* pParent)
    : BasePage(pParent)
    , m_pStatsGroup(nullptr)
    , m_pLblTotalImages(nullptr)
    , m_pLblLabeledCount(nullptr)
    , m_pLblUnlabeledCount(nullptr)
    , m_pLblTrainCount(nullptr)
    , m_pLblValCount(nullptr)
    , m_pLblTestCount(nullptr)
    , m_pLabelFilterGroup(nullptr)
    , m_pLabelFilterLayout(nullptr)
    , m_pChkAllLabels(nullptr)
    , m_pSplitFilterGroup(nullptr)
    , m_pChkSplitAll(nullptr)
    , m_pChkSplitTrain(nullptr)
    , m_pChkSplitVal(nullptr)
    , m_pChkSplitTest(nullptr)
    , m_pBtnManageLabels(nullptr)
    , m_pBtnImportImages(nullptr)
    , m_pBtnImportFolder(nullptr)
    , m_pBtnDeleteSelected(nullptr)
    , m_pCmbSort(nullptr)
    , m_pEdtSearch(nullptr)
    , m_pSliderThumbSize(nullptr)
    , m_pListView(nullptr)
    , m_pModel(nullptr)
    , m_pProxyModel(nullptr)
    , m_pDelegate(nullptr)
    , m_pDataset(nullptr)
{
    // 20260322 ZJH 启用拖放接受
    setAcceptDrops(true);

    // 20260324 ZJH 初始化防抖定时器（50ms SingleShot）
    // 合并 dataChanged + imagesAdded + imagesRemoved + splitChanged 等快速连续信号
    m_pRefreshDebounce = new QTimer(this);
    m_pRefreshDebounce->setSingleShot(true);  // 20260324 ZJH 单次触发模式
    m_pRefreshDebounce->setInterval(50);      // 20260324 ZJH 50ms 防抖间隔
    connect(m_pRefreshDebounce, &QTimer::timeout, this, [this]() {
        // 20260324 ZJH 定时器到期时执行一次刷新（合并多次信号）
        refreshModel();
        updateStatistics();
    });

    // 20260322 ZJH 创建左侧面板和中央面板
    QWidget* pLeftPanel   = createLeftPanel();    // 20260322 ZJH 统计+过滤面板
    QWidget* pCenterPanel = createCenterPanel();  // 20260322 ZJH 工具栏+缩略图网格

    // 20260322 ZJH 设置三栏布局（左面板 280px，无右面板）
    setLeftPanelWidth(280);
    setupThreeColumnLayout(pLeftPanel, pCenterPanel, nullptr);
}

// 20260322 ZJH 页面进入前台时刷新显示
void GalleryPage::onEnter()
{
    // 20260322 ZJH 刷新统计和模型（可能在其他页面修改了数据）
    if (m_pDataset) {
        updateStatistics();
    }
}

// 20260322 ZJH 页面离开前台（无特殊操作）
void GalleryPage::onLeave()
{
}

// 20260322 ZJH 项目加载后绑定数据集信号并刷新
// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void GalleryPage::onProjectLoadedImpl()
{
    if (!m_pProject) {
        return;  // 20260322 ZJH 无效项目
    }

    // 20260322 ZJH 获取数据集引用
    m_pDataset = m_pProject->dataset();

    if (m_pDataset) {
        // 20260322 ZJH 绑定数据集变更信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::dataChanged,
                this, &GalleryPage::onDatasetChanged);

        // 20260322 ZJH 绑定图像添加信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::imagesAdded,
                this, &GalleryPage::onDatasetChanged);

        // 20260322 ZJH 绑定图像移除信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::imagesRemoved,
                this, &GalleryPage::onDatasetChanged);

        // 20260322 ZJH 绑定标签变更信号 → 刷新过滤面板
        connect(m_pDataset, &ImageDataset::labelsChanged,
                this, &GalleryPage::onLabelsChanged);

        // 20260322 ZJH 绑定拆分变更信号 → 刷新模型
        connect(m_pDataset, &ImageDataset::splitChanged,
                this, &GalleryPage::onDatasetChanged);
    }

    // 20260322 ZJH 刷新缩略图模型和统计
    refreshModel();
    updateStatistics();
    rebuildLabelFilters();
}

// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void GalleryPage::onProjectClosedImpl()
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

    // 20260322 ZJH 重置统计
    updateStatistics();
    rebuildLabelFilters();
}

// ===== 拖放事件 =====

// 20260322 ZJH 拖放进入事件
void GalleryPage::dragEnterEvent(QDragEnterEvent* pEvent)
{
    // 20260322 ZJH 检查是否包含 URL（文件/文件夹路径）
    if (pEvent->mimeData()->hasUrls()) {
        pEvent->acceptProposedAction();  // 20260322 ZJH 接受拖入
    }
}

// 20260322 ZJH 拖放移动事件
void GalleryPage::dragMoveEvent(QDragMoveEvent* pEvent)
{
    if (pEvent->mimeData()->hasUrls()) {
        pEvent->acceptProposedAction();  // 20260322 ZJH 持续接受
    }
}

// 20260322 ZJH 拖放释放事件：处理文件/文件夹导入
void GalleryPage::dropEvent(QDropEvent* pEvent)
{
    if (!m_pDataset) {
        return;  // 20260322 ZJH 没有项目/数据集，忽略
    }

    const QMimeData* pMime = pEvent->mimeData();
    if (!pMime->hasUrls()) {
        return;  // 20260322 ZJH 无 URL 数据
    }

    QStringList vecImageFiles;    // 20260322 ZJH 收集图像文件路径
    QStringList vecFolders;       // 20260322 ZJH 收集文件夹路径

    // 20260322 ZJH 支持的图像格式后缀集合
    static const QSet<QString> s_setSupportedExts = {
        "png", "jpg", "jpeg", "bmp", "tif", "tiff"
    };

    // 20260322 ZJH 遍历拖入的所有 URL
    const QList<QUrl> vecUrls = pMime->urls();
    for (const QUrl& url : vecUrls) {
        QString strPath = url.toLocalFile();  // 20260322 ZJH 转换为本地路径
        QFileInfo fi(strPath);

        if (fi.isDir()) {
            // 20260322 ZJH 是文件夹，加入文件夹列表
            vecFolders.append(strPath);
        } else if (fi.isFile()) {
            // 20260322 ZJH 是文件，检查后缀
            QString strExt = fi.suffix().toLower();
            if (s_setSupportedExts.contains(strExt)) {
                vecImageFiles.append(strPath);
            }
        }
    }

    // 20260322 ZJH 先导入文件夹（带子目录标签自动创建）
    for (const QString& strFolder : vecFolders) {
        m_pDataset->importFromFolder(strFolder, true);
    }

    // 20260322 ZJH 再导入散落的图像文件
    if (!vecImageFiles.isEmpty()) {
        m_pDataset->importFiles(vecImageFiles);
    }

    // 20260324 ZJH 拖放导入完成后自动保存项目
    autoSaveProject();

    pEvent->acceptProposedAction();  // 20260322 ZJH 标记事件已处理
}

// ===== 槽函数 =====

// 20260322 ZJH 导入图像文件
void GalleryPage::onImportImages()
{
    if (!m_pDataset) {
        QMessageBox::information(this,
            QStringLiteral("导入图像"),
            QStringLiteral("请先创建或打开一个项目。"));
        return;
    }

    // 20260322 ZJH 打开文件选择对话框（多选）
    QStringList vecFiles = QFileDialog::getOpenFileNames(
        this,
        QStringLiteral("选择图像文件"),
        QString(),
        QStringLiteral("图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)")
    );

    if (!vecFiles.isEmpty()) {
        m_pDataset->importFiles(vecFiles);  // 20260322 ZJH 导入文件列表
        // 20260324 ZJH 导入完成后自动保存项目
        autoSaveProject();
    }
}

// 20260322 ZJH 导入文件夹
void GalleryPage::onImportFolder()
{
    if (!m_pDataset) {
        QMessageBox::information(this,
            QStringLiteral("导入文件夹"),
            QStringLiteral("请先创建或打开一个项目。"));
        return;
    }

    // 20260322 ZJH 打开文件夹选择对话框
    QString strFolder = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("选择图像文件夹"),
        QString(),
        QFileDialog::ShowDirsOnly
    );

    if (!strFolder.isEmpty()) {
        m_pDataset->importFromFolder(strFolder, true);  // 20260322 ZJH 递归导入
        // 20260324 ZJH 导入完成后自动保存项目
        autoSaveProject();
    }
}

// 20260322 ZJH 删除选中的图像
void GalleryPage::onDeleteSelected()
{
    if (!m_pDataset || !m_pListView) {
        return;
    }

    // 20260322 ZJH 获取选中索引列表
    QModelIndexList vecSelected = m_pListView->selectionModel()->selectedIndexes();
    if (vecSelected.isEmpty()) {
        QMessageBox::information(this,
            QStringLiteral("删除图像"),
            QStringLiteral("请先选中要删除的图像。"));
        return;
    }

    // 20260322 ZJH 确认删除
    int nRet = QMessageBox::question(this,
        QStringLiteral("确认删除"),
        QStringLiteral("确定要从数据集中移除 %1 张图像吗？\n（不会删除磁盘上的文件）")
            .arg(vecSelected.size()),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (nRet != QMessageBox::Yes) {
        return;  // 20260322 ZJH 用户取消
    }

    // 20260322 ZJH 收集要移除的 UUID 列表
    QVector<QString> vecUuids;
    for (const QModelIndex& proxyIdx : vecSelected) {
        // 20260322 ZJH 从代理模型映射回源模型
        QModelIndex srcIdx = m_pProxyModel->mapToSource(proxyIdx);
        QString strUuid = srcIdx.data(UuidRole).toString();
        if (!strUuid.isEmpty()) {
            vecUuids.append(strUuid);
        }
    }

    // 20260322 ZJH 批量移除
    if (!vecUuids.isEmpty()) {
        m_pDataset->removeImages(vecUuids);
    }
}

// 20260322 ZJH 搜索框文本变化时实时过滤
void GalleryPage::onSearchTextChanged(const QString& strText)
{
    if (m_pProxyModel) {
        m_pProxyModel->setSearchText(strText);
    }
}

// 20260322 ZJH 排序方式变化
void GalleryPage::onSortChanged(int nIndex)
{
    if (!m_pProxyModel) {
        return;
    }

    // 20260322 ZJH 排序映射：0=名称, 1=日期(按名称代替), 2=标签, 3=拆分
    switch (nIndex) {
    case 0:  // 20260322 ZJH 按名称排序
        m_pProxyModel->setSortRole(FileNameRole);
        break;
    case 1:  // 20260322 ZJH 按日期排序（暂用文件名代替）
        m_pProxyModel->setSortRole(FilePathRole);
        break;
    case 2:  // 20260322 ZJH 按标签排序
        m_pProxyModel->setSortRole(LabelIdRole);
        break;
    case 3:  // 20260322 ZJH 按拆分排序
        m_pProxyModel->setSortRole(SplitTypeRole);
        break;
    default:
        m_pProxyModel->setSortRole(FileNameRole);
        break;
    }

    m_pProxyModel->sort(0, Qt::AscendingOrder);  // 20260322 ZJH 触发排序
}

// 20260322 ZJH 缩略图大小滑块值变化
void GalleryPage::onThumbnailSizeChanged(int nSize)
{
    if (m_pDelegate) {
        m_pDelegate->setThumbnailSize(nSize);  // 20260322 ZJH 更新代理的缩略图大小
    }

    if (m_pListView) {
        // 20260322 ZJH 更新 QListView 的网格大小
        int nGridW = nSize + 12;  // 20260322 ZJH 加上左右内边距
        int nGridH = nSize + 12 + 20;  // 20260322 ZJH 加上边距和文字行高
        m_pListView->setGridSize(QSize(nGridW, nGridH));

        // 20260322 ZJH 强制刷新视图
        m_pListView->viewport()->update();

        // 20260323 ZJH 触发布局重算（不直接 emit layoutChanged，避免跳过 layoutAboutToBeChanged）
        m_pListView->doItemsLayout();
    }
}

// 20260322 ZJH 双击跳转标注页
void GalleryPage::onItemDoubleClicked(const QModelIndex& index)
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

// 20260324 ZJH 数据集变化时通过防抖定时器延迟刷新
// dataChanged + imagesAdded + imagesRemoved + splitChanged 可能在同一操作中连续触发
// 50ms 防抖将多次信号合并为一次 refreshModel + updateStatistics
void GalleryPage::onDatasetChanged()
{
    m_pRefreshDebounce->start();  // 20260324 ZJH 重启防抖定时器（合并连续信号）
}

// 20260324 ZJH 标签列表变化时刷新过滤面板（通过防抖定时器合并模型刷新）
void GalleryPage::onLabelsChanged()
{
    rebuildLabelFilters();  // 20260324 ZJH 立即重建过滤面板（轻量操作，无需防抖）
    m_pRefreshDebounce->start();  // 20260324 ZJH 重启防抖定时器，与 onDatasetChanged 合并
}

// 20260322 ZJH 标签过滤复选框状态变化
void GalleryPage::onLabelFilterChanged()
{
    if (!m_pProxyModel) {
        return;
    }

    // 20260322 ZJH 检查 "全部" 复选框
    if (m_pChkAllLabels && m_pChkAllLabels->isChecked()) {
        // 20260322 ZJH "全部" 选中时清除标签过滤（显示所有）
        m_pProxyModel->setLabelFilter(QSet<int>());
        // 20260322 ZJH 同步勾选所有标签复选框
        for (QCheckBox* pChk : m_vecLabelCheckboxes) {
            pChk->blockSignals(true);   // 20260322 ZJH 阻止信号递归
            pChk->setChecked(true);
            pChk->blockSignals(false);
        }
        return;
    }

    // 20260322 ZJH 收集选中的标签 ID
    QSet<int> setSelected;
    for (QCheckBox* pChk : m_vecLabelCheckboxes) {
        if (pChk->isChecked()) {
            int nLabelId = pChk->property("labelId").toInt();
            setSelected.insert(nLabelId);
        }
    }

    // 20260322 ZJH 如果没有选中任何标签，也添加 -1 表示包含未标注图像
    if (setSelected.isEmpty()) {
        // 20260322 ZJH 空集合意味着不过滤，显示全部
        m_pProxyModel->setLabelFilter(QSet<int>());
    } else {
        m_pProxyModel->setLabelFilter(setSelected);
    }
}

// 20260322 ZJH 拆分过滤复选框状态变化
void GalleryPage::onSplitFilterChanged()
{
    if (!m_pProxyModel) {
        return;
    }

    // 20260322 ZJH 检查 "全部" 复选框
    if (m_pChkSplitAll && m_pChkSplitAll->isChecked()) {
        m_pProxyModel->setSplitFilter(QSet<int>());
        // 20260322 ZJH 同步勾选所有拆分复选框
        m_pChkSplitTrain->blockSignals(true);
        m_pChkSplitVal->blockSignals(true);
        m_pChkSplitTest->blockSignals(true);
        m_pChkSplitTrain->setChecked(true);
        m_pChkSplitVal->setChecked(true);
        m_pChkSplitTest->setChecked(true);
        m_pChkSplitTrain->blockSignals(false);
        m_pChkSplitVal->blockSignals(false);
        m_pChkSplitTest->blockSignals(false);
        return;
    }

    // 20260322 ZJH 收集选中的拆分类型
    QSet<int> setSelected;
    if (m_pChkSplitTrain->isChecked()) {
        setSelected.insert(static_cast<int>(om::SplitType::Train));
    }
    if (m_pChkSplitVal->isChecked()) {
        setSelected.insert(static_cast<int>(om::SplitType::Validation));
    }
    if (m_pChkSplitTest->isChecked()) {
        setSelected.insert(static_cast<int>(om::SplitType::Test));
    }

    if (setSelected.isEmpty()) {
        // 20260322 ZJH 空集合 = 不过滤
        m_pProxyModel->setSplitFilter(QSet<int>());
    } else {
        // 20260322 ZJH 同时包含未分配类型，否则未分配的图像会被隐藏
        setSelected.insert(static_cast<int>(om::SplitType::Unassigned));
        m_pProxyModel->setSplitFilter(setSelected);
    }
}

// ===== 私有方法 =====

// 20260322 ZJH 创建左侧面板
QWidget* GalleryPage::createLeftPanel()
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

    QString strLabelStyle = QStringLiteral(
        "QLabel {"
        "  color: #cbd5e1;"
        "  font-size: 12px;"
        "  border: none;"
        "  background: transparent;"
        "}"
    );

    QString strCheckBoxStyle = QStringLiteral(
        "QCheckBox {"
        "  color: #cbd5e1;"
        "  font-size: 12px;"
        "  spacing: 6px;"
        "}"
    );

    // ===== 统计分组 =====
    m_pStatsGroup = new QGroupBox(QStringLiteral("统计"), pPanel);
    m_pStatsGroup->setStyleSheet(strGroupStyle);
    QVBoxLayout* pStatsLayout = new QVBoxLayout(m_pStatsGroup);
    pStatsLayout->setSpacing(4);

    // 20260322 ZJH 创建统计标签
    m_pLblTotalImages    = new QLabel(QStringLiteral("总图像数: 0"), m_pStatsGroup);
    m_pLblLabeledCount   = new QLabel(QStringLiteral("已标注: 0"), m_pStatsGroup);
    m_pLblUnlabeledCount = new QLabel(QStringLiteral("未标注: 0"), m_pStatsGroup);
    m_pLblTrainCount     = new QLabel(QStringLiteral("训练集: 0"), m_pStatsGroup);
    m_pLblValCount       = new QLabel(QStringLiteral("验证集: 0"), m_pStatsGroup);
    m_pLblTestCount      = new QLabel(QStringLiteral("测试集: 0"), m_pStatsGroup);

    // 20260322 ZJH 应用样式
    for (QLabel* pLbl : {m_pLblTotalImages, m_pLblLabeledCount, m_pLblUnlabeledCount,
                          m_pLblTrainCount, m_pLblValCount, m_pLblTestCount}) {
        pLbl->setStyleSheet(strLabelStyle);
    }

    pStatsLayout->addWidget(m_pLblTotalImages);
    pStatsLayout->addWidget(m_pLblLabeledCount);
    pStatsLayout->addWidget(m_pLblUnlabeledCount);

    // 20260322 ZJH 添加分割线
    QFrame* pLine1 = new QFrame(m_pStatsGroup);
    pLine1->setFrameShape(QFrame::HLine);
    pLine1->setStyleSheet(QStringLiteral("color: #2a2d35;"));
    pStatsLayout->addWidget(pLine1);

    pStatsLayout->addWidget(m_pLblTrainCount);
    pStatsLayout->addWidget(m_pLblValCount);
    pStatsLayout->addWidget(m_pLblTestCount);

    pLayout->addWidget(m_pStatsGroup);

    // ===== 标签过滤分组 =====
    m_pLabelFilterGroup = new QGroupBox(QStringLiteral("标签过滤"), pPanel);
    m_pLabelFilterGroup->setStyleSheet(strGroupStyle);
    m_pLabelFilterLayout = new QVBoxLayout(m_pLabelFilterGroup);
    m_pLabelFilterLayout->setSpacing(4);

    // 20260322 ZJH "全部" 复选框
    m_pChkAllLabels = new QCheckBox(QStringLiteral("全部"), m_pLabelFilterGroup);
    m_pChkAllLabels->setChecked(true);
    m_pChkAllLabels->setStyleSheet(strCheckBoxStyle);
    m_pLabelFilterLayout->addWidget(m_pChkAllLabels);
    connect(m_pChkAllLabels, &QCheckBox::toggled,
            this, &GalleryPage::onLabelFilterChanged);

    pLayout->addWidget(m_pLabelFilterGroup);

    // ===== 拆分过滤分组 =====
    m_pSplitFilterGroup = new QGroupBox(QStringLiteral("拆分过滤"), pPanel);
    m_pSplitFilterGroup->setStyleSheet(strGroupStyle);
    QVBoxLayout* pSplitLayout = new QVBoxLayout(m_pSplitFilterGroup);
    pSplitLayout->setSpacing(4);

    m_pChkSplitAll   = new QCheckBox(QStringLiteral("全部"), m_pSplitFilterGroup);
    m_pChkSplitTrain = new QCheckBox(QStringLiteral("训练"), m_pSplitFilterGroup);
    m_pChkSplitVal   = new QCheckBox(QStringLiteral("验证"), m_pSplitFilterGroup);
    m_pChkSplitTest  = new QCheckBox(QStringLiteral("测试"), m_pSplitFilterGroup);

    m_pChkSplitAll->setChecked(true);
    m_pChkSplitTrain->setChecked(true);
    m_pChkSplitVal->setChecked(true);
    m_pChkSplitTest->setChecked(true);

    for (QCheckBox* pChk : {m_pChkSplitAll, m_pChkSplitTrain, m_pChkSplitVal, m_pChkSplitTest}) {
        pChk->setStyleSheet(strCheckBoxStyle);
    }

    pSplitLayout->addWidget(m_pChkSplitAll);
    pSplitLayout->addWidget(m_pChkSplitTrain);
    pSplitLayout->addWidget(m_pChkSplitVal);
    pSplitLayout->addWidget(m_pChkSplitTest);

    // 20260322 ZJH 连接拆分过滤信号
    connect(m_pChkSplitAll,   &QCheckBox::toggled, this, &GalleryPage::onSplitFilterChanged);
    connect(m_pChkSplitTrain, &QCheckBox::toggled, this, &GalleryPage::onSplitFilterChanged);
    connect(m_pChkSplitVal,   &QCheckBox::toggled, this, &GalleryPage::onSplitFilterChanged);
    connect(m_pChkSplitTest,  &QCheckBox::toggled, this, &GalleryPage::onSplitFilterChanged);

    pLayout->addWidget(m_pSplitFilterGroup);

    // ===== 标签管理按钮 =====
    m_pBtnManageLabels = new QPushButton(QStringLiteral("标签管理"), pPanel);
    m_pBtnManageLabels->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #2a2d35;"
        "  color: #cbd5e1;"
        "  border: 1px solid #3b3f4a;"
        "  border-radius: 4px;"
        "  padding: 6px 12px;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3b3f4a;"
        "}"
    ));
    connect(m_pBtnManageLabels, &QPushButton::clicked, this, [this]() {
        if (!m_pDataset) {
            QMessageBox::information(this, QStringLiteral("标签管理"),
                QStringLiteral("请先创建或打开一个项目。"));
            return;
        }
        LabelManagementDialog dlg(m_pDataset, this);
        dlg.exec();
        onLabelsChanged();  // 20260322 ZJH 刷新标签过滤面板
        refreshModel();     // 20260322 ZJH 刷新缩略图（标签可能已变更）
    });
    pLayout->addWidget(m_pBtnManageLabels);

    // 20260322 ZJH 弹性间距，将内容推到顶部
    pLayout->addStretch(1);

    return pPanel;
}

// 20260322 ZJH 创建中央面板
QWidget* GalleryPage::createCenterPanel()
{
    QWidget* pPanel = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    // ===== 工具栏 =====
    QWidget* pToolbar = new QWidget(pPanel);
    pToolbar->setFixedHeight(40);
    pToolbar->setStyleSheet(QStringLiteral(
        "QWidget {"
        "  background-color: #1a1d23;"
        "  border-bottom: 1px solid #2a2d35;"
        "}"
    ));
    QHBoxLayout* pToolLayout = new QHBoxLayout(pToolbar);
    pToolLayout->setContentsMargins(8, 4, 8, 4);
    pToolLayout->setSpacing(6);

    // 20260322 ZJH 按钮通用样式
    QString strBtnStyle = QStringLiteral(
        "QPushButton {"
        "  background-color: #2563eb;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 4px 12px;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3b82f6;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #1d4ed8;"
        "}"
    );

    QString strBtnDangerStyle = QStringLiteral(
        "QPushButton {"
        "  background-color: #7f1d1d;"
        "  color: #fca5a5;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 4px 12px;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #991b1b;"
        "}"
    );

    // 20260322 ZJH "导入图像" 按钮
    m_pBtnImportImages = new QPushButton(QStringLiteral("导入图像"), pToolbar);
    m_pBtnImportImages->setStyleSheet(strBtnStyle);
    connect(m_pBtnImportImages, &QPushButton::clicked, this, &GalleryPage::onImportImages);
    pToolLayout->addWidget(m_pBtnImportImages);

    // 20260322 ZJH "导入文件夹" 按钮
    m_pBtnImportFolder = new QPushButton(QStringLiteral("导入文件夹"), pToolbar);
    m_pBtnImportFolder->setStyleSheet(strBtnStyle);
    connect(m_pBtnImportFolder, &QPushButton::clicked, this, &GalleryPage::onImportFolder);
    pToolLayout->addWidget(m_pBtnImportFolder);

    // 20260322 ZJH "删除选中" 按钮
    m_pBtnDeleteSelected = new QPushButton(QStringLiteral("删除选中"), pToolbar);
    m_pBtnDeleteSelected->setStyleSheet(strBtnDangerStyle);
    connect(m_pBtnDeleteSelected, &QPushButton::clicked, this, &GalleryPage::onDeleteSelected);
    pToolLayout->addWidget(m_pBtnDeleteSelected);

    // 20260322 ZJH 分隔符
    QFrame* pSep = new QFrame(pToolbar);
    pSep->setFrameShape(QFrame::VLine);
    pSep->setStyleSheet(QStringLiteral("color: #2a2d35;"));
    pToolLayout->addWidget(pSep);

    // 20260322 ZJH 排序下拉框
    m_pCmbSort = new QComboBox(pToolbar);
    m_pCmbSort->addItems({
        QStringLiteral("名称"),
        QStringLiteral("日期"),
        QStringLiteral("标签"),
        QStringLiteral("拆分")
    });
    m_pCmbSort->setStyleSheet(QStringLiteral(
        "QComboBox {"
        "  background-color: #1e2230;"
        "  color: #cbd5e1;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 2px 8px;"
        "  font-size: 12px;"
        "  min-width: 60px;"
        "}"
    ));
    connect(m_pCmbSort, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GalleryPage::onSortChanged);
    pToolLayout->addWidget(m_pCmbSort);

    // 20260322 ZJH 搜索框
    m_pEdtSearch = new QLineEdit(pToolbar);
    m_pEdtSearch->setPlaceholderText(QStringLiteral("搜索..."));
    m_pEdtSearch->setClearButtonEnabled(true);
    m_pEdtSearch->setMaximumWidth(200);
    m_pEdtSearch->setStyleSheet(QStringLiteral(
        "QLineEdit {"
        "  background-color: #1e2230;"
        "  color: #cbd5e1;"
        "  border: 1px solid #2a2d35;"
        "  border-radius: 4px;"
        "  padding: 4px 8px;"
        "  font-size: 12px;"
        "}"
        "QLineEdit:focus {"
        "  border-color: #2563eb;"
        "}"
    ));
    connect(m_pEdtSearch, &QLineEdit::textChanged,
            this, &GalleryPage::onSearchTextChanged);
    pToolLayout->addWidget(m_pEdtSearch);

    // 20260322 ZJH 弹性空间
    pToolLayout->addStretch(1);

    // 20260322 ZJH 缩略图大小标签
    QLabel* pLblThumbSize = new QLabel(QStringLiteral("大小:"), pToolbar);
    pLblThumbSize->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 12px; border: none; background: transparent; }"));
    pToolLayout->addWidget(pLblThumbSize);

    // 20260322 ZJH 缩略图大小滑块 (80~320px)
    m_pSliderThumbSize = new QSlider(Qt::Horizontal, pToolbar);
    m_pSliderThumbSize->setRange(80, 320);
    m_pSliderThumbSize->setValue(160);
    m_pSliderThumbSize->setFixedWidth(120);
    m_pSliderThumbSize->setStyleSheet(QStringLiteral(
        "QSlider::groove:horizontal {"
        "  background: #2a2d35;"
        "  height: 4px;"
        "  border-radius: 2px;"
        "}"
        "QSlider::handle:horizontal {"
        "  background: #2563eb;"
        "  width: 12px;"
        "  height: 12px;"
        "  margin: -4px 0;"
        "  border-radius: 6px;"
        "}"
    ));
    connect(m_pSliderThumbSize, &QSlider::valueChanged,
            this, &GalleryPage::onThumbnailSizeChanged);
    pToolLayout->addWidget(m_pSliderThumbSize);

    pLayout->addWidget(pToolbar);

    // ===== 缩略图网格（QListView + IconMode） =====

    // 20260322 ZJH 创建底层数据模型
    m_pModel = new QStandardItemModel(this);

    // 20260322 ZJH 创建过滤排序代理模型
    m_pProxyModel = new GalleryFilterProxy(this);
    m_pProxyModel->setSourceModel(m_pModel);
    m_pProxyModel->setSortRole(FileNameRole);

    // 20260322 ZJH 创建缩略图绘制代理
    m_pDelegate = new ThumbnailDelegate(this);

    // 20260322 ZJH 创建 QListView
    m_pListView = new QListView(pPanel);
    m_pListView->setModel(m_pProxyModel);               // 20260322 ZJH 使用代理模型
    m_pListView->setItemDelegate(m_pDelegate);           // 20260322 ZJH 使用自定义代理
    m_pListView->setViewMode(QListView::IconMode);       // 20260322 ZJH 图标模式（网格排列）
    m_pListView->setResizeMode(QListView::Adjust);       // 20260322 ZJH 自动调整网格布局
    m_pListView->setWrapping(true);                      // 20260322 ZJH 允许换行
    m_pListView->setFlow(QListView::LeftToRight);        // 20260322 ZJH 从左到右排列
    m_pListView->setMovement(QListView::Static);         // 20260322 ZJH 禁止拖拽移动项
    m_pListView->setSelectionMode(QAbstractItemView::ExtendedSelection);  // 20260322 ZJH 支持 Ctrl/Shift 多选
    m_pListView->setUniformItemSizes(true);              // 20260322 ZJH 统一项大小（提升性能）
    m_pListView->setSpacing(4);                          // 20260322 ZJH 项间距 4px

    // 20260322 ZJH 设置初始网格大小
    int nInitSize = m_pDelegate->thumbnailSize();
    m_pListView->setGridSize(QSize(nInitSize + 12, nInitSize + 12 + 20));

    // 20260322 ZJH 设置视图样式
    m_pListView->setStyleSheet(QStringLiteral(
        "QListView {"
        "  background-color: #13151a;"
        "  border: none;"
        "  outline: none;"
        "}"
        "QListView::item {"
        "  border: none;"
        "}"
        "QListView::item:selected {"
        "  background: transparent;"
        "}"
        "QListView::item:hover {"
        "  background: transparent;"
        "}"
    ));

    // 20260322 ZJH 连接双击信号
    connect(m_pListView, &QListView::doubleClicked,
            this, &GalleryPage::onItemDoubleClicked);

    pLayout->addWidget(m_pListView, 1);  // 20260322 ZJH stretch=1，占满剩余空间

    return pPanel;
}

// 20260322 ZJH 刷新缩略图模型（从数据集重建全部条目）
void GalleryPage::refreshModel()
{
    if (!m_pModel) {
        return;
    }

    // 20260322 ZJH 清空模型
    m_pModel->clear();

    if (!m_pDataset) {
        return;  // 20260322 ZJH 无数据集
    }

    // 20260322 ZJH 遍历数据集中的每张图像，创建 QStandardItem
    const QVector<ImageEntry>& vecImages = m_pDataset->images();
    for (const ImageEntry& entry : vecImages) {
        QStandardItem* pItem = new QStandardItem();

        // 20260322 ZJH 设置自定义数据角色
        pItem->setData(entry.strFilePath, FilePathRole);
        pItem->setData(entry.strUuid, UuidRole);
        pItem->setData(entry.nLabelId, LabelIdRole);
        pItem->setData(entry.fileName(), FileNameRole);
        pItem->setData(static_cast<int>(entry.eSplit), SplitTypeRole);

        // 20260324 ZJH 设置缩略图磁盘缓存目录路径（ThumbnailDelegate 通过 ThumbDirRole 读取）
        // 用于磁盘级缩略图缓存，应用重启后无需从原图重新生成
        if (m_pDataset) {
            pItem->setData(m_pDataset->thumbnailCachePath(), ThumbDirRole);
        }

        // 20260322 ZJH 设置标签颜色和名称
        if (entry.nLabelId >= 0) {
            const LabelInfo* pLabel = m_pDataset->findLabel(entry.nLabelId);
            if (pLabel) {
                pItem->setData(pLabel->color, LabelColorRole);
                pItem->setData(pLabel->strName, LabelNameRole);
            }
        }

        // 20260322 ZJH 设置显示文本（用于默认排序）
        pItem->setText(entry.fileName());

        // 20260322 ZJH 禁止编辑
        pItem->setEditable(false);

        m_pModel->appendRow(pItem);
    }

    // 20260322 ZJH 触发排序
    if (m_pProxyModel && m_pCmbSort) {
        onSortChanged(m_pCmbSort->currentIndex());
    }
}

// 20260322 ZJH 更新统计标签
void GalleryPage::updateStatistics()
{
    if (!m_pDataset) {
        // 20260322 ZJH 无数据集，显示全零
        m_pLblTotalImages->setText(QStringLiteral("总图像数: 0"));
        m_pLblLabeledCount->setText(QStringLiteral("已标注: 0"));
        m_pLblUnlabeledCount->setText(QStringLiteral("未标注: 0"));
        m_pLblTrainCount->setText(QStringLiteral("训练集: 0"));
        m_pLblValCount->setText(QStringLiteral("验证集: 0"));
        m_pLblTestCount->setText(QStringLiteral("测试集: 0"));
        return;
    }

    // 20260322 ZJH 从数据集获取统计数据
    int nTotal    = m_pDataset->imageCount();
    int nLabeled  = m_pDataset->labeledCount();
    int nUnlabeled = nTotal - nLabeled;
    int nTrain    = m_pDataset->countBySplit(om::SplitType::Train);
    int nVal      = m_pDataset->countBySplit(om::SplitType::Validation);
    int nTest     = m_pDataset->countBySplit(om::SplitType::Test);

    // 20260322 ZJH 更新标签文本
    m_pLblTotalImages->setText(QStringLiteral("总图像数: %1").arg(nTotal));
    m_pLblLabeledCount->setText(QStringLiteral("已标注: %1").arg(nLabeled));
    m_pLblUnlabeledCount->setText(QStringLiteral("未标注: %1").arg(nUnlabeled));
    m_pLblTrainCount->setText(QStringLiteral("训练集: %1").arg(nTrain));
    m_pLblValCount->setText(QStringLiteral("验证集: %1").arg(nVal));
    m_pLblTestCount->setText(QStringLiteral("测试集: %1").arg(nTest));
}

// 20260322 ZJH 重建标签过滤面板
void GalleryPage::rebuildLabelFilters()
{
    // 20260322 ZJH 移除已有的标签复选框（保留 "全部" 复选框）
    for (QCheckBox* pChk : m_vecLabelCheckboxes) {
        m_pLabelFilterLayout->removeWidget(pChk);
        delete pChk;
    }
    m_vecLabelCheckboxes.clear();

    if (!m_pDataset) {
        return;  // 20260322 ZJH 无数据集
    }

    QString strCheckBoxStyle = QStringLiteral(
        "QCheckBox {"
        "  color: #cbd5e1;"
        "  font-size: 12px;"
        "  spacing: 6px;"
        "}"
    );

    // 20260322 ZJH 为每个标签创建一个复选框
    const QVector<LabelInfo>& vecLabels = m_pDataset->labels();
    for (const LabelInfo& label : vecLabels) {
        QCheckBox* pChk = new QCheckBox(label.strName, m_pLabelFilterGroup);
        pChk->setChecked(true);
        pChk->setProperty("labelId", label.nId);
        pChk->setStyleSheet(strCheckBoxStyle);

        // 20260322 ZJH 为复选框创建带颜色的图标
        QPixmap pixIcon(12, 12);
        pixIcon.fill(Qt::transparent);
        QPainter painter(&pixIcon);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(Qt::NoPen);
        painter.setBrush(label.color);
        painter.drawRoundedRect(0, 0, 12, 12, 2, 2);
        painter.end();
        pChk->setIcon(QIcon(pixIcon));

        connect(pChk, &QCheckBox::toggled, this, &GalleryPage::onLabelFilterChanged);
        m_pLabelFilterLayout->addWidget(pChk);
        m_vecLabelCheckboxes.append(pChk);
    }

    // 20260322 ZJH 添加 "未标注" 复选框
    QCheckBox* pChkUnlabeled = new QCheckBox(QStringLiteral("未标注"), m_pLabelFilterGroup);
    pChkUnlabeled->setChecked(true);
    pChkUnlabeled->setProperty("labelId", -1);  // 20260322 ZJH -1 表示未标注
    pChkUnlabeled->setStyleSheet(strCheckBoxStyle);
    connect(pChkUnlabeled, &QCheckBox::toggled, this, &GalleryPage::onLabelFilterChanged);
    m_pLabelFilterLayout->addWidget(pChkUnlabeled);
    m_vecLabelCheckboxes.append(pChkUnlabeled);
}

// 20260324 ZJH 辅助方法：导入图像后自动保存项目到 .dfproj
// 确保退出应用或意外关闭时已导入的图像数据不丢失
void GalleryPage::autoSaveProject()
{
    if (!m_pProject) {
        return;
    }
    QString strProjFile = m_pProject->path() + "/" + m_pProject->name() + ".dfproj";
    if (ProjectSerializer::save(m_pProject, strProjFile)) {
        m_pProject->setDirty(false);  // 20260324 ZJH 保存成功，重置脏标志
    }
}
