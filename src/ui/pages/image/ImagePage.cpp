// 20260322 ZJH ImagePage 实现
// 图像标注页面：三栏布局，图像查看/缩放/标注绘制/撤销重做/导航

#include "ui/pages/image/ImagePage.h"               // 20260322 ZJH 类声明
#include "ui/widgets/ZoomableGraphicsView.h"         // 20260322 ZJH 可缩放图像查看器
#include "ui/widgets/AnnotationController.h"         // 20260322 ZJH 标注控制器
#include "ui/widgets/AnnotationGraphicsItem.h"       // 20260322 ZJH 标注图形项
#include "core/data/ImageDataset.h"                  // 20260322 ZJH 数据集
#include "core/data/ImageEntry.h"                    // 20260322 ZJH 图像条目
#include "core/data/LabelInfo.h"                     // 20260322 ZJH 标签信息
#include "core/project/Project.h"                    // 20260322 ZJH 项目
#include "app/Application.h"                         // 20260322 ZJH 全局事件总线
#include "ui/dialogs/LabelManagementDialog.h"        // 20260322 ZJH 标签管理对话框

#include <QImageReader>    // 20260323 ZJH 受控图像加载（支持尺寸限制）
#include <QtConcurrent>   // 20260324 ZJH 异步图像加载（避免 UI 线程阻塞）
#include <QVBoxLayout>     // 20260322 ZJH 垂直布局
#include <QHBoxLayout>     // 20260322 ZJH 水平布局
#include <QScrollArea>     // 20260322 ZJH 滚动区域
#include <QGraphicsScene>  // 20260322 ZJH 图形场景
#include <QFileInfo>       // 20260322 ZJH 文件信息
#include <QImage>          // 20260322 ZJH 图像加载
#include <QKeyEvent>       // 20260322 ZJH 键盘事件
#include <QMouseEvent>     // 20260322 ZJH 鼠标事件（eventFilter 中使用）
#include <QShortcut>       // 20260322 ZJH 快捷键
#include <QFrame>          // 20260322 ZJH 分隔线

// 20260322 ZJH 统一样式表常量
static const QString s_strGroupTitleStyle = QStringLiteral(
    "QLabel { color: #94a3b8; font-size: 11px; font-weight: bold; "
    "padding: 4px 0px; }");

static const QString s_strToolBtnStyle = QStringLiteral(
    "QPushButton { background-color: #1e2230; color: #e2e8f0; border: 1px solid #2a2d35; "
    "border-radius: 4px; padding: 6px 8px; font-size: 12px; min-width: 50px; }"
    "QPushButton:checked { background-color: #2563eb; border-color: #2563eb; }"
    "QPushButton:hover { background-color: #2a3040; }");

static const QString s_strSmallBtnStyle = QStringLiteral(
    "QPushButton { background-color: #1e2230; color: #e2e8f0; border: 1px solid #2a2d35; "
    "border-radius: 3px; padding: 4px 10px; font-size: 11px; }"
    "QPushButton:hover { background-color: #2a3040; }"
    "QPushButton:pressed { background-color: #2563eb; }");

static const QString s_strNavBtnStyle = QStringLiteral(
    "QPushButton { background-color: #1e2230; color: #e2e8f0; border: 1px solid #2a2d35; "
    "border-radius: 3px; padding: 4px 8px; font-size: 13px; min-width: 30px; }"
    "QPushButton:hover { background-color: #2a3040; }"
    "QPushButton:pressed { background-color: #2563eb; }");

static const QString s_strInfoLabelStyle = QStringLiteral(
    "QLabel { color: #94a3b8; font-size: 11px; padding: 1px 0px; }");

static const QString s_strValueLabelStyle = QStringLiteral(
    "QLabel { color: #e2e8f0; font-size: 11px; padding: 1px 0px; }");

static const QString s_strPanelStyle = QStringLiteral(
    "background-color: #13151a; border: none;");

// 20260322 ZJH 构造函数，创建三栏布局
ImagePage::ImagePage(QWidget* pParent)
    : BasePage(pParent)
{
    // 20260322 ZJH 设置左右面板宽度
    setLeftPanelWidth(250);
    setRightPanelWidth(220);

    // 20260322 ZJH 创建三栏布局
    QWidget* pLeft = createLeftPanel();      // 20260322 ZJH 左侧面板
    QWidget* pCenter = createCenterPanel();  // 20260322 ZJH 中央面板
    QWidget* pRight = createRightPanel();    // 20260322 ZJH 右侧面板

    setupThreeColumnLayout(pLeft, pCenter, pRight);

    // 20260324 ZJH 创建异步图像加载监视器
    m_pImageWatcher = new QFutureWatcher<QImage>(this);  // 20260324 ZJH 生命周期由父对象管理
    // 20260324 ZJH 异步加载完成时触发回调，在主线程中处理结果
    connect(m_pImageWatcher, &QFutureWatcher<QImage>::finished,
            this, &ImagePage::onAsyncImageLoaded);

    // 20260322 ZJH 连接标注控制器信号
    connect(m_pAnnotCtrl, &AnnotationController::annotationCountChanged,
            this, &ImagePage::onAnnotationCountChanged);
    connect(m_pAnnotCtrl, &AnnotationController::annotationChanged,
            this, [this]() {
                refreshAnnotationList();  // 20260322 ZJH 标注变更时刷新列表
            });

    // 20260324 ZJH 列表选中标注时，视图居中到该标注位置
    connect(m_pAnnotCtrl, &AnnotationController::requestCenterOnItem,
            this, [this](AnnotationGraphicsItem* pItem) {
                if (pItem && m_pGraphicsView) {
                    // 20260324 ZJH 确保标注在视图中可见并居中
                    m_pGraphicsView->centerOn(pItem);
                }
            });

    // 20260322 ZJH 连接 Application 的 requestOpenImage 信号
    connect(Application::instance(), &Application::requestOpenImage,
            this, &ImagePage::loadImage);

    // 20260322 ZJH 设置快捷键
    // Ctrl+Z 撤销
    QShortcut* pUndoShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+Z")), this);
    connect(pUndoShortcut, &QShortcut::activated, m_pAnnotCtrl, &AnnotationController::undo);

    // 20260322 ZJH Ctrl+Y 重做
    QShortcut* pRedoShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+Y")), this);
    connect(pRedoShortcut, &QShortcut::activated, m_pAnnotCtrl, &AnnotationController::redo);

    // 20260322 ZJH Delete 删除选中标注
    QShortcut* pDeleteShortcut = new QShortcut(QKeySequence(Qt::Key_Delete), this);
    connect(pDeleteShortcut, &QShortcut::activated,
            m_pAnnotCtrl, &AnnotationController::deleteSelectedAnnotation);

    // 20260322 ZJH 工具快捷键
    // 20260406 ZJH V 键切换到选择工具
    QShortcut* pSelectKey = new QShortcut(QKeySequence(Qt::Key_V), this);
    connect(pSelectKey, &QShortcut::activated, this, [this]() {
        m_pBtnSelect->setChecked(true);  // 20260406 ZJH 选中选择工具按钮
        onToolChanged(0);                // 20260406 ZJH 通知工具切换为选择模式
    });

    // 20260406 ZJH B 键切换到矩形标注工具
    QShortcut* pRectKey = new QShortcut(QKeySequence(Qt::Key_B), this);
    connect(pRectKey, &QShortcut::activated, this, [this]() {
        m_pBtnRect->setChecked(true);    // 20260406 ZJH 选中矩形工具按钮
        onToolChanged(1);                // 20260406 ZJH 通知工具切换为矩形模式
    });

    // 20260406 ZJH P 键切换到多边形标注工具
    QShortcut* pPolygonKey = new QShortcut(QKeySequence(Qt::Key_P), this);
    connect(pPolygonKey, &QShortcut::activated, this, [this]() {
        m_pBtnPolygon->setChecked(true);  // 20260406 ZJH 选中多边形工具按钮
        onToolChanged(2);                  // 20260406 ZJH 通知工具切换为多边形模式
    });

    // 20260406 ZJH D 键切换到画笔标注工具
    QShortcut* pBrushKey = new QShortcut(QKeySequence(Qt::Key_D), this);
    connect(pBrushKey, &QShortcut::activated, this, [this]() {
        m_pBtnBrush->setChecked(true);    // 20260406 ZJH 选中画笔工具按钮
        onToolChanged(3);                  // 20260406 ZJH 通知工具切换为画笔模式
    });

    // 20260330 ZJH Ctrl+C 复制选中标注到剪贴板
    QShortcut* pCopyShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+C")), this);
    connect(pCopyShortcut, &QShortcut::activated, this, [this]() {
        m_pAnnotCtrl->copySelectedAnnotation();  // 20260330 ZJH 复制选中标注
    });

    // 20260330 ZJH Ctrl+V 粘贴标注（从剪贴板粘贴到当前图像，偏移 20px 避免重叠）
    QShortcut* pPasteShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+V")), this);
    connect(pPasteShortcut, &QShortcut::activated, this, [this]() {
        m_pAnnotCtrl->pasteAnnotation(QPointF(20.0, 20.0));  // 20260330 ZJH 粘贴并偏移
    });
}

// ===== 事件过滤器 =====

// 20260322 ZJH 拦截 viewport 鼠标事件，转发给 AnnotationController
bool ImagePage::eventFilter(QObject* pWatched, QEvent* pEvent)
{
    // 20260322 ZJH 仅处理 viewport 的鼠标事件
    if (pWatched == m_pGraphicsView->viewport()) {
        // 20260322 ZJH 鼠标按下
        if (pEvent->type() == QEvent::MouseButtonPress) {
            QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(pEvent);
            // 20260322 ZJH 中键留给 ZoomableGraphicsView 处理平移
            if (pMouseEvent->button() != Qt::MiddleButton) {
                QPointF ptScene = m_pGraphicsView->mapToScene(pMouseEvent->position().toPoint());
                if (m_pAnnotCtrl->handleMousePress(ptScene, pMouseEvent->button(),
                                                     pMouseEvent->modifiers())) {
                    return true;  // 20260322 ZJH 事件已被标注控制器处理
                }
            }
        }
        // 20260322 ZJH 鼠标移动
        else if (pEvent->type() == QEvent::MouseMove) {
            QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(pEvent);
            QPointF ptScene = m_pGraphicsView->mapToScene(pMouseEvent->position().toPoint());
            m_pAnnotCtrl->handleMouseMove(ptScene, pMouseEvent->buttons());
            // 20260322 ZJH 不拦截 — 让 ZoomableGraphicsView 也能更新鼠标坐标
        }
        // 20260322 ZJH 鼠标释放
        else if (pEvent->type() == QEvent::MouseButtonRelease) {
            QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(pEvent);
            if (pMouseEvent->button() != Qt::MiddleButton) {
                QPointF ptScene = m_pGraphicsView->mapToScene(pMouseEvent->position().toPoint());
                m_pAnnotCtrl->handleMouseRelease(ptScene, pMouseEvent->button());
            }
        }
        // 20260322 ZJH 鼠标双击
        else if (pEvent->type() == QEvent::MouseButtonDblClick) {
            QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(pEvent);
            QPointF ptScene = m_pGraphicsView->mapToScene(pMouseEvent->position().toPoint());
            if (m_pAnnotCtrl->handleMouseDoubleClick(ptScene, pMouseEvent->button())) {
                return true;  // 20260322 ZJH 事件已处理
            }
        }
    }

    // 20260322 ZJH 其他事件交给基类处理
    return BasePage::eventFilter(pWatched, pEvent);
}

// ===== 生命周期回调 =====

// 20260322 ZJH 进入页面
void ImagePage::onEnter()
{
    BasePage::onEnter();
    if (m_pDataset) {
        refreshLabelCombo();  // 20260322 ZJH 刷新标签下拉框
        // 20260322 ZJH 自动加载当前图像（首次进入时加载第一张）
        if (m_nCurrentIndex < 0 && m_pDataset->imageCount() > 0) {
            navigateToImage(0);  // 20260322 ZJH 导航到第一张图像
        } else if (m_nCurrentIndex >= 0 && m_nCurrentIndex < m_pDataset->imageCount()) {
            navigateToImage(m_nCurrentIndex);  // 20260322 ZJH 重新加载当前图像
        }
    }
}

// 20260322 ZJH 离开页面
void ImagePage::onLeave()
{
    BasePage::onLeave();
}

// 20260324 ZJH 项目加载扩展点（Template Method），基类已完成 m_pProject 赋值
void ImagePage::onProjectLoadedImpl()
{
    if (m_pProject) {
        m_pDataset = m_pProject->dataset();  // 20260322 ZJH 绑定数据集
        refreshLabelCombo();               // 20260322 ZJH 刷新标签下拉框
        updateNavLabel();                  // 20260322 ZJH 更新导航文字
    }
}

// 20260324 ZJH 项目关闭扩展点（Template Method），基类将在返回后清空 m_pProject
void ImagePage::onProjectClosedImpl()
{
    // 20260324 ZJH 取消正在进行的异步图像加载，防止回调访问已清空的数据集
    if (m_pImageWatcher->isRunning()) {
        m_pImageWatcher->cancel();
        m_pImageWatcher->waitForFinished();
    }
    m_nPendingIndex = -1;        // 20260324 ZJH 重置待加载索引
    m_strPendingUuid.clear();    // 20260324 ZJH 重置待加载 UUID

    // 20260322 ZJH 解绑标注控制器
    m_pAnnotCtrl->unbindImage();
    // 20260322 ZJH 清空图像查看器
    m_pGraphicsView->clearImage();
    // 20260322 ZJH 清空数据集
    m_pDataset = nullptr;
    m_nCurrentIndex = -1;
    // 20260322 ZJH 清空 UI 所有显示内容
    m_pLabelCombo->clear();                                    // 20260406 ZJH 清空标签下拉框
    m_pAnnotationList->clear();                                // 20260406 ZJH 清空标注列表
    m_pNavLabel->setText(QStringLiteral("0 / 0"));             // 20260406 ZJH 导航索引归零
    m_pThumbnailLabel->clear();                                // 20260406 ZJH 清空缩略图预览
    m_pFileNameLabel->setText(QStringLiteral("-"));             // 20260406 ZJH 文件名重置
    m_pFileSizeLabel->setText(QStringLiteral("-"));             // 20260406 ZJH 文件尺寸重置
    m_pFileBytesLabel->setText(QStringLiteral("-"));            // 20260406 ZJH 文件大小重置
    m_pFileDepthLabel->setText(QStringLiteral("-"));            // 20260406 ZJH 色深重置
    m_pMousePosLabel->setText(QStringLiteral("-"));             // 20260406 ZJH 鼠标坐标重置
    m_pPixelValueLabel->setText(QStringLiteral("-"));           // 20260406 ZJH 像素值重置
}

// ===== 图像加载与导航 =====

// 20260322 ZJH 加载指定 UUID 的图像
void ImagePage::loadImage(const QString& strUuid, const QString& strAnnotationUuid)
{
    if (!m_pDataset) {
        return;  // 20260322 ZJH 无数据集
    }

    // 20260404 ZJH 记录待选中的标注 UUID（加载完成后在 onAsyncImageLoaded 中选中）
    m_strPendingAnnotationUuid = strAnnotationUuid;

    // 20260322 ZJH 在数据集中查找图像索引
    const QVector<ImageEntry>& vecImages = m_pDataset->images();
    for (int i = 0; i < vecImages.size(); ++i) {
        if (vecImages[i].strUuid == strUuid) {
            navigateToImage(i);  // 20260322 ZJH 按索引导航
            return;
        }
    }
}

// 20260324 ZJH 按索引导航到图像（异步加载，不阻塞 UI 线程）
// 原实现在 UI 线程同步调用 QImageReader::read()，大图（如 4000x3000 JPEG）会阻塞 100-500ms
// 现改为 QtConcurrent::run 在后台线程加载，完成后在主线程回调 onAsyncImageLoaded
void ImagePage::navigateToImage(int nIndex)
{
    if (!m_pDataset) {
        return;  // 20260322 ZJH 无数据集
    }

    const QVector<ImageEntry>& vecImages = m_pDataset->images();

    // 20260322 ZJH 检查索引范围
    if (nIndex < 0 || nIndex >= vecImages.size()) {
        return;  // 20260322 ZJH 索引越界
    }

    // 20260322 ZJH 记录当前索引
    m_nCurrentIndex = nIndex;

    // 20260322 ZJH 获取图像条目
    const ImageEntry& entry = vecImages[nIndex];

    // 20260324 ZJH 记录待加载图像的上下文信息，用于异步回调中判断结果是否过期
    m_nPendingIndex     = nIndex;              // 20260324 ZJH 当前请求的索引
    m_strPendingUuid    = entry.strUuid;       // 20260324 ZJH 当前请求的图像 UUID
    m_strPendingFilePath = entry.strFilePath;  // 20260324 ZJH 当前请求的文件路径

    // 20260324 ZJH 取消之前可能正在进行的异步加载（如果用户快速连续点击导航）
    // cancel() + waitForFinished() 确保旧任务完成后再提交新任务，避免竞态
    if (m_pImageWatcher->isRunning()) {
        m_pImageWatcher->cancel();       // 20260324 ZJH 请求取消（QFuture 不保证立即停止，但标记取消状态）
        m_pImageWatcher->waitForFinished();  // 20260324 ZJH 等待旧任务完成，防止信号重入
    }

    // 20260324 ZJH 清空当前图像显示，给用户视觉反馈表示正在加载
    m_pGraphicsView->clearImage();

    // 20260322 ZJH 立即更新导航标签（不依赖图像加载结果）
    updateNavLabel();

    // 20260322 ZJH 立即更新右侧面板文件基本信息（不依赖图像像素数据）
    QFileInfo fileInfo(entry.strFilePath);
    m_pFileNameLabel->setText(fileInfo.fileName());  // 20260322 ZJH 文件名

    // 20260322 ZJH 文件大小
    qint64 nBytes = fileInfo.size();
    if (nBytes > 1024 * 1024) {
        m_pFileBytesLabel->setText(QStringLiteral("%1 MB").arg(nBytes / (1024.0 * 1024.0), 0, 'f', 2));
    } else if (nBytes > 1024) {
        m_pFileBytesLabel->setText(QStringLiteral("%1 KB").arg(nBytes / 1024.0, 0, 'f', 1));
    } else {
        m_pFileBytesLabel->setText(QStringLiteral("%1 B").arg(nBytes));
    }

    // 20260324 ZJH 图像尺寸和色深需等加载完成后更新（显示加载中占位文字）
    m_pFileSizeLabel->setText(QStringLiteral("..."));
    m_pFileDepthLabel->setText(QStringLiteral("..."));

    // 20260324 ZJH 如果 ImageEntry 的宽高尚未读取（延迟元数据，nWidth==0），先显示占位
    // 如果已有缓存元数据（nWidth>0），可立即显示尺寸
    if (entry.nWidth > 0 && entry.nHeight > 0) {
        m_pFileSizeLabel->setText(QStringLiteral("%1 x %2").arg(entry.nWidth).arg(entry.nHeight));
    }

    // 20260324 ZJH 将图像加载任务提交到后台线程（QtConcurrent::run）
    // 捕获文件路径值拷贝，lambda 中使用 QImageReader 解码图像
    QString strFilePath = entry.strFilePath;  // 20260324 ZJH 值拷贝避免引用悬挂
    auto future = QtConcurrent::run([strFilePath]() -> QImage {
        // 20260324 ZJH 后台线程：加载图像文件
        QImageReader reader(strFilePath);  // 20260324 ZJH 创建图像读取器
        // 20260323 ZJH 超过 8000x8000 (64MP) 自动缩放，防止 OOM
        QSize imgSize = reader.size();
        if (imgSize.width() > 8000 || imgSize.height() > 8000) {
            reader.setScaledSize(imgSize.scaled(8000, 8000, Qt::KeepAspectRatio));
        }
        return reader.read();  // 20260324 ZJH 解码图像（耗时操作，在后台线程完成）
    });

    // 20260324 ZJH 绑定 future 到 watcher，加载完成后自动触发 finished 信号
    m_pImageWatcher->setFuture(future);
}

// 20260324 ZJH 异步图像加载完成回调（在主线程执行）
// 检查结果是否过期（用户可能在加载期间已切换到其他图像），仅处理最新请求的结果
void ImagePage::onAsyncImageLoaded()
{
    // 20260324 ZJH 获取异步加载的结果图像
    QImage image = m_pImageWatcher->result();

    // 20260324 ZJH 检查加载结果是否已过期（用户在加载期间切换了图像）
    // 通过比较 m_nCurrentIndex 与 m_nPendingIndex 判断
    if (m_nCurrentIndex != m_nPendingIndex) {
        return;  // 20260324 ZJH 用户已切换到另一张图像，丢弃过期结果
    }

    // 20260324 ZJH 检查图像是否加载成功
    if (image.isNull()) {
        m_pFileSizeLabel->setText(QStringLiteral("加载失败"));  // 20260324 ZJH 显示错误信息
        m_pFileDepthLabel->setText(QStringLiteral("-"));
        return;  // 20260324 ZJH 图像加载失败
    }

    // 20260322 ZJH 显示图像
    m_pGraphicsView->setImage(image);

    // 20260322 ZJH 绑定标注控制器到当前图像
    ImageEntry* pMutableEntry = m_pDataset ? m_pDataset->findImage(m_strPendingUuid) : nullptr;
    if (pMutableEntry) {
        m_pAnnotCtrl->bindImage(pMutableEntry, m_pDataset);

        // 20260324 ZJH 延迟元数据回填：如果 ImageEntry 的宽高尚未读取（导入时跳过了），
        // 则在首次显示时从已加载的 QImage 回填元数据并缓存
        if (pMutableEntry->nWidth == 0 || pMutableEntry->nHeight == 0) {
            pMutableEntry->nWidth    = image.width();     // 20260324 ZJH 回填图像宽度
            pMutableEntry->nHeight   = image.height();    // 20260324 ZJH 回填图像高度
            pMutableEntry->nChannels = ImageDataset::channelsFromFormat(image.format());  // 20260324 ZJH 回填通道数
        }
    }

    // 20260322 ZJH 刷新标注列表
    refreshAnnotationList();

    // 20260404 ZJH 如果有待选中的标注 UUID（来自检查页双击跳转），自动选中并缩放定位
    if (!m_strPendingAnnotationUuid.isEmpty() && m_pAnnotCtrl) {
        // 20260404 ZJH 先选中标注（会触发 centerOn + 闪烁高亮）
        m_pAnnotCtrl->selectAnnotationByUuid(m_strPendingAnnotationUuid);

        // 20260404 ZJH 缩放视图到标注区域，让用户一眼看到标注位置
        // 在全图15%缩放下标注太小看不见，需要 fitInView 到标注包围框
        if (m_pGraphicsView && pMutableEntry) {
            for (const Annotation& ann : pMutableEntry->vecAnnotations) {
                if (ann.strUuid == m_strPendingAnnotationUuid && !ann.rectBounds.isEmpty()) {
                    // 20260404 ZJH 标注区域加 50% padding，提供上下文（不要太紧）
                    QRectF rcTarget = ann.rectBounds;
                    double dPadW = rcTarget.width() * 0.5;   // 20260404 ZJH 50% 水平 padding
                    double dPadH = rcTarget.height() * 0.5;  // 20260404 ZJH 50% 垂直 padding
                    rcTarget.adjust(-dPadW, -dPadH, dPadW, dPadH);
                    // 20260404 ZJH 使用 ZoomableGraphicsView::fitInView(QRectF) 缩放到标注区域
                    // 正确同步 m_dZoomFactor 内部状态，确保滚轮缩放/百分比显示正常
                    m_pGraphicsView->fitInView(rcTarget);
                    break;
                }
            }
        }

        m_strPendingAnnotationUuid.clear();  // 20260404 ZJH 只选中一次，清除
    }

    // 20260322 ZJH 更新右侧面板图像尺寸信息
    m_pFileSizeLabel->setText(QStringLiteral("%1 x %2").arg(image.width()).arg(image.height()));

    // 20260322 ZJH 色深
    m_pFileDepthLabel->setText(QStringLiteral("%1 bit").arg(image.depth()));

    // 20260322 ZJH 更新缩略图预览
    QPixmap thumbPixmap = QPixmap::fromImage(image).scaled(
        180, 120, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    m_pThumbnailLabel->setPixmap(thumbPixmap);
}

// 20260322 ZJH 上一张
void ImagePage::onPrevImage()
{
    if (m_nCurrentIndex > 0) {
        navigateToImage(m_nCurrentIndex - 1);
    }
}

// 20260322 ZJH 下一张
void ImagePage::onNextImage()
{
    if (m_pDataset && m_nCurrentIndex < m_pDataset->imageCount() - 1) {
        navigateToImage(m_nCurrentIndex + 1);
    }
}

// 20260322 ZJH 适应视图
void ImagePage::onFitView()
{
    m_pGraphicsView->fitInView();
}

// 20260322 ZJH 1:1 实际大小
void ImagePage::onActualSize()
{
    m_pGraphicsView->zoomToActualSize();
}

// 20260322 ZJH 缩放变化回调
void ImagePage::onZoomChanged(double dPercent)
{
    // 20260322 ZJH 更新中央面板缩放标签
    m_pZoomLabel->setText(QStringLiteral("%1%").arg(static_cast<int>(dPercent)));

    // 20260322 ZJH 更新右侧面板缩放滑块（不触发信号）
    m_pZoomSlider->blockSignals(true);
    m_pZoomSlider->setValue(static_cast<int>(dPercent));
    m_pZoomSlider->blockSignals(false);

    // 20260322 ZJH 更新右侧面板缩放数值
    m_pZoomValueLabel->setText(QStringLiteral("%1%").arg(static_cast<int>(dPercent)));
}

// 20260322 ZJH 鼠标位置变化
void ImagePage::onMousePositionChanged(const QPointF& ptScene, const QPoint& ptImage)
{
    Q_UNUSED(ptScene);
    m_pMousePosLabel->setText(QStringLiteral("(%1, %2)").arg(ptImage.x()).arg(ptImage.y()));
}

// 20260322 ZJH 像素值变化
void ImagePage::onPixelValueChanged(const QPoint& ptImage, int nGray, QRgb rgbValue)
{
    Q_UNUSED(ptImage);
    // 20260322 ZJH 显示 RGB 值和灰度值
    m_pPixelValueLabel->setText(QStringLiteral("R:%1 G:%2 B:%3 Gray:%4")
        .arg(qRed(rgbValue)).arg(qGreen(rgbValue)).arg(qBlue(rgbValue)).arg(nGray));
}

// 20260322 ZJH 工具切换回调
void ImagePage::onToolChanged(int nId)
{
    // 20260322 ZJH 将按钮 ID 映射到工具类型
    AnnotationTool eTool = static_cast<AnnotationTool>(nId);
    m_pAnnotCtrl->setCurrentTool(eTool);

    // 20260322 ZJH 笔刷大小控件仅在画笔工具时显示
    m_pBrushGroup->setVisible(nId == 3);
}

// 20260322 ZJH 标签选择变化
void ImagePage::onLabelComboChanged(int nIndex)
{
    if (!m_pDataset || nIndex < 0) {
        return;
    }

    // 20260322 ZJH 获取标签信息
    const QVector<LabelInfo>& vecLabels = m_pDataset->labels();
    if (nIndex < vecLabels.size()) {
        const LabelInfo& label = vecLabels[nIndex];
        m_pAnnotCtrl->setCurrentLabel(label.nId, label.strName, label.color);
    }
}

// 20260322 ZJH 标注数量变化
void ImagePage::onAnnotationCountChanged(int nCount)
{
    Q_UNUSED(nCount);
    refreshAnnotationList();  // 20260322 ZJH 刷新标注列表显示
}

// 20260324 ZJH 删除选中标注 — 同时支持场景选择和列表选择
void ImagePage::onDeleteAnnotation()
{
    // 20260324 ZJH 优先检查场景中是否有选中的标注
    if (m_pAnnotCtrl->hasSceneSelection()) {
        m_pAnnotCtrl->deleteSelectedAnnotation();
        return;
    }

    // 20260324 ZJH 场景无选中 → 检查标注列表中是否有选中项
    QListWidgetItem* pCurrentItem = m_pAnnotationList->currentItem();
    if (pCurrentItem) {
        QString strUuid = pCurrentItem->data(Qt::UserRole).toString();
        if (!strUuid.isEmpty()) {
            m_pAnnotCtrl->deleteAnnotationByUuid(strUuid);
        }
    }
}

// 20260322 ZJH 显示/隐藏标注
void ImagePage::onToggleAnnotationVisibility(bool bVisible)
{
    // 20260322 ZJH 遍历场景中所有标注图形项，设置可见性
    QList<QGraphicsItem*> vecItems = m_pGraphicsView->scene()->items();
    for (QGraphicsItem* pItem : vecItems) {
        AnnotationGraphicsItem* pAnnoItem = dynamic_cast<AnnotationGraphicsItem*>(pItem);
        if (pAnnoItem) {
            pAnnoItem->setVisible(bVisible);
        }
    }
}

// 20260322 ZJH 缩放滑块变化
void ImagePage::onZoomSliderChanged(int nValue)
{
    m_pGraphicsView->setZoomPercent(static_cast<double>(nValue));
}

// 20260322 ZJH 笔刷大小变化
void ImagePage::onBrushSizeChanged(int nValue)
{
    m_pAnnotCtrl->setBrushRadius(nValue);
    m_pBrushSizeLabel->setText(QStringLiteral("%1").arg(nValue));
}

// ===== 面板创建 =====

// 20260322 ZJH 创建左侧面板
QWidget* ImagePage::createLeftPanel()
{
    QWidget* pPanel = new QWidget(this);
    pPanel->setStyleSheet(s_strPanelStyle);

    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(8, 8, 8, 8);
    pLayout->setSpacing(4);

    // ===== 标签分组 =====
    {
        QVBoxLayout* pLabelLayout = new QVBoxLayout();
        pLabelLayout->setSpacing(4);

        // 20260322 ZJH 标签下拉框
        m_pLabelCombo = new QComboBox(pPanel);
        m_pLabelCombo->setStyleSheet(QStringLiteral(
            "QComboBox { background-color: #1e2230; color: #e2e8f0; border: 1px solid #2a2d35; "
            "border-radius: 3px; padding: 4px 8px; font-size: 12px; }"
            "QComboBox::drop-down { border: none; width: 20px; }"
            "QComboBox QAbstractItemView { background-color: #1e2230; color: #e2e8f0; "
            "border: 1px solid #2a2d35; selection-background-color: #2563eb; }"));
        connect(m_pLabelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &ImagePage::onLabelComboChanged);
        pLabelLayout->addWidget(m_pLabelCombo);

        // 20260322 ZJH 按钮行
        QHBoxLayout* pBtnRow = new QHBoxLayout();
        m_pBtnAssignLabel = new QPushButton(QStringLiteral("分配标签"), pPanel);
        m_pBtnAssignLabel->setStyleSheet(s_strSmallBtnStyle);
        pBtnRow->addWidget(m_pBtnAssignLabel);

        m_pBtnManageLabels = new QPushButton(QStringLiteral("管理标签"), pPanel);
        m_pBtnManageLabels->setStyleSheet(s_strSmallBtnStyle);
        pBtnRow->addWidget(m_pBtnManageLabels);
        // 20260322 ZJH 连接标签管理对话框
        connect(m_pBtnManageLabels, &QPushButton::clicked, this, [this]() {
            if (!m_pDataset) return;
            LabelManagementDialog dlg(m_pDataset, this);
            dlg.exec();
            refreshLabelCombo();  // 20260322 ZJH 刷新标签下拉
        });
        pLabelLayout->addLayout(pBtnRow);

        pLayout->addWidget(createGroupBox(QStringLiteral("标签"), pLabelLayout));
    }

    // ===== 工具分组 =====
    {
        QVBoxLayout* pToolLayout = new QVBoxLayout();
        pToolLayout->setSpacing(4);

        // 20260322 ZJH 工具按钮组（互斥）
        m_pToolGroup = new QButtonGroup(this);
        m_pToolGroup->setExclusive(true);

        QHBoxLayout* pToolBtnRow = new QHBoxLayout();

        // 20260322 ZJH 选择工具 (V)
        m_pBtnSelect = new QPushButton(QStringLiteral("选择(V)"), pPanel);
        m_pBtnSelect->setCheckable(true);
        m_pBtnSelect->setChecked(true);  // 20260322 ZJH 默认选中
        m_pBtnSelect->setStyleSheet(s_strToolBtnStyle);
        m_pToolGroup->addButton(m_pBtnSelect, 0);
        pToolBtnRow->addWidget(m_pBtnSelect);

        // 20260322 ZJH 矩形工具 (B)
        m_pBtnRect = new QPushButton(QStringLiteral("矩形(B)"), pPanel);
        m_pBtnRect->setCheckable(true);
        m_pBtnRect->setStyleSheet(s_strToolBtnStyle);
        m_pToolGroup->addButton(m_pBtnRect, 1);
        pToolBtnRow->addWidget(m_pBtnRect);

        pToolLayout->addLayout(pToolBtnRow);

        QHBoxLayout* pToolBtnRow2 = new QHBoxLayout();

        // 20260322 ZJH 多边形工具 (P)
        m_pBtnPolygon = new QPushButton(QStringLiteral("多边形(P)"), pPanel);
        m_pBtnPolygon->setCheckable(true);
        m_pBtnPolygon->setStyleSheet(s_strToolBtnStyle);
        m_pToolGroup->addButton(m_pBtnPolygon, 2);
        pToolBtnRow2->addWidget(m_pBtnPolygon);

        // 20260322 ZJH 画笔工具 (D)
        m_pBtnBrush = new QPushButton(QStringLiteral("画笔(D)"), pPanel);
        m_pBtnBrush->setCheckable(true);
        m_pBtnBrush->setStyleSheet(s_strToolBtnStyle);
        m_pToolGroup->addButton(m_pBtnBrush, 3);
        pToolBtnRow2->addWidget(m_pBtnBrush);

        pToolLayout->addLayout(pToolBtnRow2);

        // 20260322 ZJH 连接工具按钮组信号
        connect(m_pToolGroup, &QButtonGroup::idClicked,
                this, &ImagePage::onToolChanged);

        // 20260322 ZJH 笔刷大小控件组（仅画笔工具可见）
        m_pBrushGroup = new QWidget(pPanel);
        QHBoxLayout* pBrushLayout = new QHBoxLayout(m_pBrushGroup);
        pBrushLayout->setContentsMargins(0, 0, 0, 0);
        QLabel* pBrushLabel = new QLabel(QStringLiteral("笔刷:"), m_pBrushGroup);
        pBrushLabel->setStyleSheet(s_strInfoLabelStyle);
        pBrushLayout->addWidget(pBrushLabel);

        m_pBrushSlider = new QSlider(Qt::Horizontal, m_pBrushGroup);
        m_pBrushSlider->setRange(1, 50);
        m_pBrushSlider->setValue(10);
        m_pBrushSlider->setStyleSheet(QStringLiteral(
            "QSlider::groove:horizontal { background: #2a2d35; height: 4px; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #2563eb; width: 14px; height: 14px; "
            "margin: -5px 0; border-radius: 7px; }"));
        connect(m_pBrushSlider, &QSlider::valueChanged, this, &ImagePage::onBrushSizeChanged);
        pBrushLayout->addWidget(m_pBrushSlider);

        m_pBrushSizeLabel = new QLabel(QStringLiteral("10"), m_pBrushGroup);
        m_pBrushSizeLabel->setStyleSheet(s_strValueLabelStyle);
        m_pBrushSizeLabel->setFixedWidth(24);
        pBrushLayout->addWidget(m_pBrushSizeLabel);

        m_pBrushGroup->setVisible(false);  // 20260322 ZJH 默认隐藏
        pToolLayout->addWidget(m_pBrushGroup);

        pLayout->addWidget(createGroupBox(QStringLiteral("工具"), pToolLayout));
    }

    // ===== 标注列表分组 =====
    {
        QVBoxLayout* pAnnoLayout = new QVBoxLayout();
        pAnnoLayout->setSpacing(4);

        // 20260322 ZJH 标注列表
        m_pAnnotationList = new QListWidget(pPanel);
        m_pAnnotationList->setStyleSheet(QStringLiteral(
            "QListWidget { background-color: #1e2230; color: #e2e8f0; border: 1px solid #2a2d35; "
            "border-radius: 3px; font-size: 11px; }"
            "QListWidget::item { padding: 3px 6px; }"
            "QListWidget::item:selected { background-color: #2563eb; }"));
        m_pAnnotationList->setMaximumHeight(200);
        pAnnoLayout->addWidget(m_pAnnotationList);

        // 20260324 ZJH 列表选中 → 同步场景中对应标注的选中状态
        connect(m_pAnnotationList, &QListWidget::currentItemChanged,
                this, [this](QListWidgetItem* pCurrent, QListWidgetItem* /*pPrevious*/) {
            if (!pCurrent || !m_pAnnotCtrl) {
                return;
            }
            // 20260324 ZJH 从列表项取出标注 UUID
            QString strUuid = pCurrent->data(Qt::UserRole).toString();
            if (!strUuid.isEmpty()) {
                // 20260324 ZJH 在场景中选中该标注
                m_pAnnotCtrl->selectAnnotationByUuid(strUuid);
            }
        });

        // 20260322 ZJH 删除按钮
        m_pBtnDeleteAnnotation = new QPushButton(QStringLiteral("删除标注"), pPanel);
        m_pBtnDeleteAnnotation->setStyleSheet(s_strSmallBtnStyle);
        connect(m_pBtnDeleteAnnotation, &QPushButton::clicked,
                this, &ImagePage::onDeleteAnnotation);
        pAnnoLayout->addWidget(m_pBtnDeleteAnnotation);

        // 20260322 ZJH 显示/隐藏标注复选框
        m_pChkShowAnnotations = new QCheckBox(QStringLiteral("显示标注"), pPanel);
        m_pChkShowAnnotations->setChecked(true);
        m_pChkShowAnnotations->setStyleSheet(QStringLiteral(
            "QCheckBox { color: #e2e8f0; font-size: 11px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"));
        connect(m_pChkShowAnnotations, &QCheckBox::toggled,
                this, &ImagePage::onToggleAnnotationVisibility);
        pAnnoLayout->addWidget(m_pChkShowAnnotations);

        pLayout->addWidget(createGroupBox(QStringLiteral("标注列表"), pAnnoLayout));
    }

    // 20260322 ZJH 弹性空间填充底部
    pLayout->addStretch(1);

    return pPanel;
}

// 20260322 ZJH 创建中央面板
QWidget* ImagePage::createCenterPanel()
{
    QWidget* pPanel = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    // ===== 导航栏 =====
    {
        QWidget* pNavBar = new QWidget(pPanel);
        pNavBar->setFixedHeight(36);
        pNavBar->setStyleSheet(QStringLiteral(
            "background-color: #1a1d23; border-bottom: 1px solid #2a2d35;"));

        QHBoxLayout* pNavLayout = new QHBoxLayout(pNavBar);
        pNavLayout->setContentsMargins(8, 2, 8, 2);
        pNavLayout->setSpacing(6);

        // 20260322 ZJH 上一张按钮
        m_pBtnPrev = new QPushButton(QStringLiteral("<"), pNavBar);
        m_pBtnPrev->setStyleSheet(s_strNavBtnStyle);
        connect(m_pBtnPrev, &QPushButton::clicked, this, &ImagePage::onPrevImage);
        pNavLayout->addWidget(m_pBtnPrev);

        // 20260322 ZJH 导航索引标签
        m_pNavLabel = new QLabel(QStringLiteral("0 / 0"), pNavBar);
        m_pNavLabel->setStyleSheet(QStringLiteral(
            "color: #e2e8f0; font-size: 12px; padding: 0 4px;"));
        pNavLayout->addWidget(m_pNavLabel);

        // 20260322 ZJH 下一张按钮
        m_pBtnNext = new QPushButton(QStringLiteral(">"), pNavBar);
        m_pBtnNext->setStyleSheet(s_strNavBtnStyle);
        connect(m_pBtnNext, &QPushButton::clicked, this, &ImagePage::onNextImage);
        pNavLayout->addWidget(m_pBtnNext);

        // 20260322 ZJH 弹性空间
        pNavLayout->addStretch(1);

        // 20260322 ZJH 适应按钮
        m_pBtnFit = new QPushButton(QStringLiteral("适应"), pNavBar);
        m_pBtnFit->setStyleSheet(s_strNavBtnStyle);
        connect(m_pBtnFit, &QPushButton::clicked, this, &ImagePage::onFitView);
        pNavLayout->addWidget(m_pBtnFit);

        // 20260322 ZJH 1:1 按钮
        m_pBtnActual = new QPushButton(QStringLiteral("1:1"), pNavBar);
        m_pBtnActual->setStyleSheet(s_strNavBtnStyle);
        connect(m_pBtnActual, &QPushButton::clicked, this, &ImagePage::onActualSize);
        pNavLayout->addWidget(m_pBtnActual);

        // 20260322 ZJH 缩放百分比标签
        m_pZoomLabel = new QLabel(QStringLiteral("100%"), pNavBar);
        m_pZoomLabel->setStyleSheet(QStringLiteral(
            "color: #94a3b8; font-size: 11px; padding: 0 4px; min-width: 40px;"));
        pNavLayout->addWidget(m_pZoomLabel);

        pLayout->addWidget(pNavBar);
    }

    // ===== 图像查看器 =====
    {
        m_pGraphicsView = new ZoomableGraphicsView(pPanel);

        // 20260322 ZJH 创建标注控制器，绑定到同一个场景
        m_pAnnotCtrl = new AnnotationController(m_pGraphicsView->scene(), this);

        // 20260322 ZJH 连接缩放信号
        connect(m_pGraphicsView, &ZoomableGraphicsView::zoomChanged,
                this, &ImagePage::onZoomChanged);

        // 20260322 ZJH 连接鼠标信号
        connect(m_pGraphicsView, &ZoomableGraphicsView::mousePositionChanged,
                this, &ImagePage::onMousePositionChanged);
        connect(m_pGraphicsView, &ZoomableGraphicsView::pixelValueChanged,
                this, &ImagePage::onPixelValueChanged);

        // 20260322 ZJH 重写鼠标事件以支持标注工具
        // 使用 installEventFilter 机制转发鼠标事件到 AnnotationController
        m_pGraphicsView->viewport()->installEventFilter(this);

        pLayout->addWidget(m_pGraphicsView, 1);  // 20260322 ZJH stretch=1 占满剩余空间
    }

    return pPanel;
}

// 20260322 ZJH 创建右侧面板
QWidget* ImagePage::createRightPanel()
{
    QWidget* pPanel = new QWidget(this);
    pPanel->setStyleSheet(s_strPanelStyle);

    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pLayout->setContentsMargins(8, 8, 8, 8);
    pLayout->setSpacing(4);

    // ===== 预览分组 =====
    {
        QVBoxLayout* pPreviewLayout = new QVBoxLayout();
        m_pThumbnailLabel = new QLabel(pPanel);
        m_pThumbnailLabel->setFixedSize(200, 140);
        m_pThumbnailLabel->setAlignment(Qt::AlignCenter);
        m_pThumbnailLabel->setStyleSheet(QStringLiteral(
            "QLabel { background-color: #1e2230; border: 1px solid #2a2d35; border-radius: 3px; }"));
        pPreviewLayout->addWidget(m_pThumbnailLabel);
        pLayout->addWidget(createGroupBox(QStringLiteral("预览"), pPreviewLayout));
    }

    // ===== 缩放分组 =====
    {
        QVBoxLayout* pZoomLayout = new QVBoxLayout();

        m_pZoomSlider = new QSlider(Qt::Horizontal, pPanel);
        m_pZoomSlider->setRange(1, 500);
        m_pZoomSlider->setValue(100);
        m_pZoomSlider->setStyleSheet(QStringLiteral(
            "QSlider::groove:horizontal { background: #2a2d35; height: 4px; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #2563eb; width: 14px; height: 14px; "
            "margin: -5px 0; border-radius: 7px; }"));
        connect(m_pZoomSlider, &QSlider::valueChanged, this, &ImagePage::onZoomSliderChanged);
        pZoomLayout->addWidget(m_pZoomSlider);

        m_pZoomValueLabel = new QLabel(QStringLiteral("100%"), pPanel);
        m_pZoomValueLabel->setStyleSheet(s_strValueLabelStyle);
        m_pZoomValueLabel->setAlignment(Qt::AlignCenter);
        pZoomLayout->addWidget(m_pZoomValueLabel);

        pLayout->addWidget(createGroupBox(QStringLiteral("缩放"), pZoomLayout));
    }

    // ===== 文件信息分组 =====
    {
        QVBoxLayout* pInfoLayout = new QVBoxLayout();
        pInfoLayout->setSpacing(2);

        // 20260322 ZJH 文件名
        QHBoxLayout* pNameRow = new QHBoxLayout();
        QLabel* pNameTitle = new QLabel(QStringLiteral("文件名:"), pPanel);
        pNameTitle->setStyleSheet(s_strInfoLabelStyle);
        pNameRow->addWidget(pNameTitle);
        m_pFileNameLabel = new QLabel(QStringLiteral("-"), pPanel);
        m_pFileNameLabel->setStyleSheet(s_strValueLabelStyle);
        m_pFileNameLabel->setWordWrap(true);
        pNameRow->addWidget(m_pFileNameLabel, 1);
        pInfoLayout->addLayout(pNameRow);

        // 20260322 ZJH 尺寸
        QHBoxLayout* pSizeRow = new QHBoxLayout();
        QLabel* pSizeTitle = new QLabel(QStringLiteral("尺寸:"), pPanel);
        pSizeTitle->setStyleSheet(s_strInfoLabelStyle);
        pSizeRow->addWidget(pSizeTitle);
        m_pFileSizeLabel = new QLabel(QStringLiteral("-"), pPanel);
        m_pFileSizeLabel->setStyleSheet(s_strValueLabelStyle);
        pSizeRow->addWidget(m_pFileSizeLabel, 1);
        pInfoLayout->addLayout(pSizeRow);

        // 20260322 ZJH 大小
        QHBoxLayout* pBytesRow = new QHBoxLayout();
        QLabel* pBytesTitle = new QLabel(QStringLiteral("大小:"), pPanel);
        pBytesTitle->setStyleSheet(s_strInfoLabelStyle);
        pBytesRow->addWidget(pBytesTitle);
        m_pFileBytesLabel = new QLabel(QStringLiteral("-"), pPanel);
        m_pFileBytesLabel->setStyleSheet(s_strValueLabelStyle);
        pBytesRow->addWidget(m_pFileBytesLabel, 1);
        pInfoLayout->addLayout(pBytesRow);

        // 20260322 ZJH 色深
        QHBoxLayout* pDepthRow = new QHBoxLayout();
        QLabel* pDepthTitle = new QLabel(QStringLiteral("色深:"), pPanel);
        pDepthTitle->setStyleSheet(s_strInfoLabelStyle);
        pDepthRow->addWidget(pDepthTitle);
        m_pFileDepthLabel = new QLabel(QStringLiteral("-"), pPanel);
        m_pFileDepthLabel->setStyleSheet(s_strValueLabelStyle);
        pDepthRow->addWidget(m_pFileDepthLabel, 1);
        pInfoLayout->addLayout(pDepthRow);

        pLayout->addWidget(createGroupBox(QStringLiteral("文件信息"), pInfoLayout));
    }

    // ===== 鼠标信息分组 =====
    {
        QVBoxLayout* pMouseLayout = new QVBoxLayout();
        pMouseLayout->setSpacing(2);

        // 20260322 ZJH 坐标
        QHBoxLayout* pPosRow = new QHBoxLayout();
        QLabel* pPosTitle = new QLabel(QStringLiteral("坐标:"), pPanel);
        pPosTitle->setStyleSheet(s_strInfoLabelStyle);
        pPosRow->addWidget(pPosTitle);
        m_pMousePosLabel = new QLabel(QStringLiteral("-"), pPanel);
        m_pMousePosLabel->setStyleSheet(s_strValueLabelStyle);
        pPosRow->addWidget(m_pMousePosLabel, 1);
        pMouseLayout->addLayout(pPosRow);

        // 20260322 ZJH 像素值
        QHBoxLayout* pPixelRow = new QHBoxLayout();
        QLabel* pPixelTitle = new QLabel(QStringLiteral("像素:"), pPanel);
        pPixelTitle->setStyleSheet(s_strInfoLabelStyle);
        pPixelRow->addWidget(pPixelTitle);
        m_pPixelValueLabel = new QLabel(QStringLiteral("-"), pPanel);
        m_pPixelValueLabel->setStyleSheet(s_strValueLabelStyle);
        m_pPixelValueLabel->setWordWrap(true);
        pPixelRow->addWidget(m_pPixelValueLabel, 1);
        pMouseLayout->addLayout(pPixelRow);

        pLayout->addWidget(createGroupBox(QStringLiteral("鼠标"), pMouseLayout));
    }

    // 20260322 ZJH 弹性空间
    pLayout->addStretch(1);

    return pPanel;
}

// ===== 辅助方法 =====

// 20260322 ZJH 刷新标签下拉框
void ImagePage::refreshLabelCombo()
{
    m_pLabelCombo->blockSignals(true);   // 20260322 ZJH 阻止信号避免递归
    m_pLabelCombo->clear();              // 20260322 ZJH 清空

    if (m_pDataset) {
        const QVector<LabelInfo>& vecLabels = m_pDataset->labels();
        for (const LabelInfo& label : vecLabels) {
            // 20260322 ZJH 添加带颜色图标的标签项
            QPixmap pixColor(12, 12);
            pixColor.fill(label.color);
            m_pLabelCombo->addItem(QIcon(pixColor), label.strName);
        }
    }

    m_pLabelCombo->blockSignals(false);  // 20260322 ZJH 恢复信号

    // 20260322 ZJH 如果有标签，选中第一个
    if (m_pLabelCombo->count() > 0) {
        m_pLabelCombo->setCurrentIndex(0);
        onLabelComboChanged(0);
    }
}

// 20260322 ZJH 刷新标注列表
void ImagePage::refreshAnnotationList()
{
    m_pAnnotationList->clear();  // 20260322 ZJH 清空列表

    if (!m_pDataset || m_nCurrentIndex < 0) {
        return;
    }

    const QVector<ImageEntry>& vecImages = m_pDataset->images();
    if (m_nCurrentIndex >= vecImages.size()) {
        return;
    }

    const ImageEntry& entry = vecImages[m_nCurrentIndex];

    // 20260322 ZJH 遍历标注列表，添加到 QListWidget
    for (int i = 0; i < entry.vecAnnotations.size(); ++i) {
        const Annotation& anno = entry.vecAnnotations[i];

        // 20260322 ZJH 构建标注描述文字
        QString strType;
        if (anno.eType == AnnotationType::Rect) {
            strType = QStringLiteral("矩形");
        } else if (anno.eType == AnnotationType::Polygon) {
            strType = QStringLiteral("多边形");
        } else {
            strType = QStringLiteral("其他");
        }

        // 20260322 ZJH 查找标签名称
        QString strLabel = QStringLiteral("未标注");
        if (anno.nLabelId >= 0 && m_pDataset) {
            const LabelInfo* pLabel = m_pDataset->findLabel(anno.nLabelId);
            if (pLabel) {
                strLabel = pLabel->strName;
            }
        }

        QString strText = QStringLiteral("#%1 %2 [%3]").arg(i + 1).arg(strType).arg(strLabel);
        QListWidgetItem* pItem = new QListWidgetItem(strText);
        pItem->setData(Qt::UserRole, anno.strUuid);  // 20260324 ZJH 存储标注 UUID 用于选择同步和删除
        m_pAnnotationList->addItem(pItem);
    }
}

// 20260322 ZJH 更新导航栏索引文字
void ImagePage::updateNavLabel()
{
    if (!m_pDataset || m_pDataset->imageCount() == 0) {
        m_pNavLabel->setText(QStringLiteral("0 / 0"));
        return;
    }

    // 20260322 ZJH 显示 "当前索引+1 / 总数"（1-based 对用户友好）
    m_pNavLabel->setText(QStringLiteral("%1 / %2")
        .arg(m_nCurrentIndex + 1)
        .arg(m_pDataset->imageCount()));
}

// 20260322 ZJH 创建分组框容器（统一暗色风格）
QWidget* ImagePage::createGroupBox(const QString& strTitle, QLayout* pLayout)
{
    QWidget* pGroup = new QWidget(this);
    QVBoxLayout* pGroupLayout = new QVBoxLayout(pGroup);
    pGroupLayout->setContentsMargins(0, 0, 0, 8);
    pGroupLayout->setSpacing(4);

    // 20260322 ZJH 标题标签
    QLabel* pTitleLabel = new QLabel(strTitle, pGroup);
    pTitleLabel->setStyleSheet(s_strGroupTitleStyle);
    pGroupLayout->addWidget(pTitleLabel);

    // 20260322 ZJH 分隔线
    QFrame* pLine = new QFrame(pGroup);
    pLine->setFrameShape(QFrame::HLine);
    pLine->setStyleSheet(QStringLiteral("background-color: #2a2d35; max-height: 1px;"));
    pGroupLayout->addWidget(pLine);

    // 20260322 ZJH 内容布局
    QWidget* pContent = new QWidget(pGroup);
    pContent->setLayout(pLayout);
    pGroupLayout->addWidget(pContent);

    return pGroup;
}
