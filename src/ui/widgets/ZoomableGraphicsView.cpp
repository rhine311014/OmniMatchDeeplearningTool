// 20260322 ZJH ZoomableGraphicsView 实现
// 可缩放平移图像查看器：滚轮缩放、拖拽平移、像素追踪、棋盘格背景

#include "ui/widgets/ZoomableGraphicsView.h"  // 20260322 ZJH 类声明

#include <QWheelEvent>      // 20260322 ZJH 滚轮事件
#include <QMouseEvent>      // 20260322 ZJH 鼠标事件
#include <QResizeEvent>     // 20260322 ZJH 窗口大小变化事件
#include <QScrollBar>       // 20260322 ZJH 滚动条控制（平移用）
#include <QPainter>         // 20260322 ZJH 棋盘格背景绘制
#include <QtMath>           // 20260322 ZJH qBound 等数学工具
#include <QPixmap>          // 20260322 ZJH QPixmap 操作

// 20260322 ZJH 构造函数，初始化场景、渲染设置和棋盘格背景
ZoomableGraphicsView::ZoomableGraphicsView(QWidget* pParent)
    : QGraphicsView(pParent)            // 20260322 ZJH 初始化基类
    , m_pScene(nullptr)                 // 20260322 ZJH 场景指针初始化
    , m_pPixmapItem(nullptr)            // 20260322 ZJH 图像项指针初始化
    , m_dZoomFactor(1.0)                // 20260322 ZJH 初始缩放 100%
    , m_dMinZoom(0.01)                  // 20260322 ZJH 最小缩放 1%
    , m_dMaxZoom(50.0)                  // 20260322 ZJH 最大缩放 5000%
    , m_dZoomStep(1.15)                 // 20260322 ZJH 每次滚轮缩放 1.15 倍
    , m_bPanning(false)                 // 20260322 ZJH 初始状态非平移
    , m_bFitOnResize(true)              // 20260322 ZJH 默认窗口变化时自动适应
{
    // 20260322 ZJH 创建图形场景，ZoomableGraphicsView 作为父对象管理其生命周期
    m_pScene = new QGraphicsScene(this);
    setScene(m_pScene);  // 20260322 ZJH 关联场景到视图

    // 20260322 ZJH 创建图像显示项并添加到场景
    m_pPixmapItem = new QGraphicsPixmapItem();
    m_pScene->addItem(m_pPixmapItem);  // 20260322 ZJH 图像项添加到场景

    // 20260322 ZJH 设置渲染质量：平滑像素缩放
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    // 20260322 ZJH 抗锯齿
    setRenderHint(QPainter::Antialiasing, true);

    // 20260322 ZJH 设置视图属性
    setTransformationAnchor(QGraphicsView::NoAnchor);      // 20260322 ZJH 手动控制缩放锚点
    setResizeAnchor(QGraphicsView::NoAnchor);               // 20260322 ZJH 窗口大小变化时不自动调整锚点
    setDragMode(QGraphicsView::NoDrag);                     // 20260322 ZJH 拖拽模式由自定义逻辑处理
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);   // 20260322 ZJH 隐藏水平滚动条
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);     // 20260322 ZJH 隐藏垂直滚动条
    setMouseTracking(true);                                  // 20260322 ZJH 启用鼠标追踪（不按键也触发 mouseMoveEvent）

    // 20260322 ZJH 设置棋盘格背景（用于透明图像查看）
    setBackgroundBrush(createCheckerboardBrush());

    // 20260322 ZJH 设置视图背景色为深灰
    setStyleSheet(QStringLiteral(
        "QGraphicsView {"
        "  border: none;"
        "  background-color: #2a2d35;"
        "}"
    ));
}

// 20260322 ZJH 设置显示图像（QImage 版本）
void ZoomableGraphicsView::setImage(const QImage& image)
{
    // 20260322 ZJH 保存原始图像用于像素值查询
    m_currentImage = image;

    // 20260322 ZJH 将 QImage 转为 QPixmap 并设置到图像项
    m_pPixmapItem->setPixmap(QPixmap::fromImage(image));

    // 20260322 ZJH 更新场景矩形以匹配图像大小
    m_pScene->setSceneRect(m_pPixmapItem->boundingRect());

    // 20260322 ZJH 加载图像后自动适应视图
    fitInView();
}

// 20260322 ZJH 设置显示图像（QPixmap 版本）
void ZoomableGraphicsView::setImage(const QPixmap& pixmap)
{
    // 20260322 ZJH 将 QPixmap 转为 QImage 保存（用于像素值查询）
    m_currentImage = pixmap.toImage();

    // 20260322 ZJH 设置到图像项
    m_pPixmapItem->setPixmap(pixmap);

    // 20260322 ZJH 更新场景矩形
    m_pScene->setSceneRect(m_pPixmapItem->boundingRect());

    // 20260322 ZJH 加载图像后自动适应视图
    fitInView();
}

// 20260322 ZJH 清除当前图像
void ZoomableGraphicsView::clearImage()
{
    // 20260322 ZJH 设置空 QPixmap 清除图像显示
    m_pPixmapItem->setPixmap(QPixmap());
    // 20260322 ZJH 清空原始图像缓存
    m_currentImage = QImage();
    // 20260322 ZJH 重置缩放因子
    m_dZoomFactor = 1.0;
    // 20260322 ZJH 重置变换矩阵
    resetTransform();
}

// 20260322 ZJH 等比缩放图像以适应视图
void ZoomableGraphicsView::fitInView()
{
    // 20260322 ZJH 检查是否有图像可显示
    if (m_currentImage.isNull()) {
        return;  // 20260322 ZJH 无图像，直接返回
    }

    // 20260322 ZJH 重置变换矩阵到单位矩阵
    resetTransform();

    // 20260322 ZJH 获取图像尺寸和视图可见区域尺寸
    QRectF rectImage = m_pPixmapItem->boundingRect();  // 20260322 ZJH 图像边界矩形
    QRectF rectView = viewport()->rect();               // 20260322 ZJH 视图可见区域矩形

    // 20260322 ZJH 计算等比缩放因子（取较小值以确保完全显示）
    double dScaleX = rectView.width() / rectImage.width();    // 20260322 ZJH 水平缩放比
    double dScaleY = rectView.height() / rectImage.height();  // 20260322 ZJH 垂直缩放比
    m_dZoomFactor = qMin(dScaleX, dScaleY);                   // 20260322 ZJH 取较小值等比缩放

    // 20260322 ZJH 限制缩放范围
    m_dZoomFactor = qBound(m_dMinZoom, m_dZoomFactor, m_dMaxZoom);

    // 20260322 ZJH 应用缩放变换
    setTransform(QTransform::fromScale(m_dZoomFactor, m_dZoomFactor));

    // 20260322 ZJH 居中显示图像
    centerOn(m_pPixmapItem);

    // 20260322 ZJH 标记为自适应模式
    m_bFitOnResize = true;

    // 20260322 ZJH 发射缩放变化信号
    emit zoomChanged(m_dZoomFactor * 100.0);
}

// 20260404 ZJH 缩放到指定矩形区域（检查页双击跳转定位标注用）
// 正确同步 m_dZoomFactor 内部状态，确保后续滚轮缩放、百分比显示正常
void ZoomableGraphicsView::fitInView(const QRectF& rect)
{
    if (rect.isEmpty()) {
        return;  // 20260404 ZJH 空矩形，忽略
    }

    // 20260404 ZJH 重置变换矩阵
    resetTransform();

    // 20260404 ZJH 计算目标矩形填满视图所需的缩放因子
    QRectF rectView = viewport()->rect();  // 20260404 ZJH 视图可见区域
    double dScaleX = rectView.width() / rect.width();    // 20260404 ZJH 水平缩放比
    double dScaleY = rectView.height() / rect.height();  // 20260404 ZJH 垂直缩放比
    m_dZoomFactor = qMin(dScaleX, dScaleY);              // 20260404 ZJH 等比缩放取较小值

    // 20260404 ZJH 限制缩放范围
    m_dZoomFactor = qBound(m_dMinZoom, m_dZoomFactor, m_dMaxZoom);

    // 20260404 ZJH 应用缩放变换
    setTransform(QTransform::fromScale(m_dZoomFactor, m_dZoomFactor));

    // 20260404 ZJH 居中到目标矩形中心
    centerOn(rect.center());

    // 20260404 ZJH 退出自适应模式（用户已定位到特定区域）
    m_bFitOnResize = false;

    // 20260404 ZJH 发射缩放变化信号
    emit zoomChanged(m_dZoomFactor * 100.0);
}

// 20260322 ZJH 缩放到 1:1 实际像素大小
void ZoomableGraphicsView::zoomToActualSize()
{
    // 20260322 ZJH 重置变换矩阵
    resetTransform();

    // 20260322 ZJH 设置缩放因子为 1.0（100%）
    m_dZoomFactor = 1.0;

    // 20260322 ZJH 应用 1:1 缩放
    setTransform(QTransform::fromScale(1.0, 1.0));

    // 20260322 ZJH 居中显示
    centerOn(m_pPixmapItem);

    // 20260322 ZJH 退出自适应模式
    m_bFitOnResize = false;

    // 20260322 ZJH 发射缩放变化信号
    emit zoomChanged(100.0);
}

// 20260322 ZJH 获取当前缩放百分比
double ZoomableGraphicsView::zoomPercent() const
{
    return m_dZoomFactor * 100.0;  // 20260322 ZJH 缩放因子转百分比
}

// 20260322 ZJH 设置缩放百分比
void ZoomableGraphicsView::setZoomPercent(double dPercent)
{
    // 20260322 ZJH 将百分比转为缩放因子
    double dNewFactor = dPercent / 100.0;
    // 20260322 ZJH 限制缩放范围
    dNewFactor = qBound(m_dMinZoom, dNewFactor, m_dMaxZoom);

    // 20260322 ZJH 重置变换并应用新的缩放
    resetTransform();
    m_dZoomFactor = dNewFactor;
    setTransform(QTransform::fromScale(m_dZoomFactor, m_dZoomFactor));

    // 20260322 ZJH 居中显示
    centerOn(m_pPixmapItem);

    // 20260322 ZJH 退出自适应模式
    m_bFitOnResize = false;

    // 20260322 ZJH 发射缩放变化信号
    emit zoomChanged(dPercent);
}

// 20260322 ZJH 滚轮事件处理：以鼠标位置为中心进行缩放
void ZoomableGraphicsView::wheelEvent(QWheelEvent* pEvent)
{
    // 20260322 ZJH 获取鼠标在场景中的位置（缩放中心点）
    QPointF ptScenePos = mapToScene(pEvent->position().toPoint());

    // 20260322 ZJH 根据滚轮方向确定缩放方向
    double dFactor = 1.0;
    if (pEvent->angleDelta().y() > 0) {
        dFactor = m_dZoomStep;   // 20260322 ZJH 向上滚动 → 放大
    } else {
        dFactor = 1.0 / m_dZoomStep;  // 20260322 ZJH 向下滚动 → 缩小
    }

    // 20260322 ZJH 执行缩放操作
    performZoom(dFactor, ptScenePos);

    // 20260322 ZJH 消费事件，阻止传递给父控件
    pEvent->accept();
}

// 20260322 ZJH 鼠标按下事件：开始平移操作
void ZoomableGraphicsView::mousePressEvent(QMouseEvent* pEvent)
{
    // 20260322 ZJH 中键按下 → 开始平移
    if (pEvent->button() == Qt::MiddleButton) {
        m_bPanning = true;                    // 20260322 ZJH 进入平移模式
        m_ptPanStart = pEvent->position().toPoint();          // 20260322 ZJH 记录起始位置
        setCursor(Qt::ClosedHandCursor);       // 20260322 ZJH 切换鼠标为抓手光标
        pEvent->accept();                      // 20260322 ZJH 消费事件
        return;                                // 20260322 ZJH 不传递给基类
    }

    // 20260322 ZJH 其他按键交给基类处理（可能被上层标注控制器拦截）
    QGraphicsView::mousePressEvent(pEvent);
}

// 20260322 ZJH 鼠标移动事件：处理平移和像素坐标追踪
void ZoomableGraphicsView::mouseMoveEvent(QMouseEvent* pEvent)
{
    // 20260322 ZJH 如果正在平移模式，更新视图位置
    if (m_bPanning) {
        // 20260322 ZJH 计算鼠标移动的增量
        QPoint ptDelta = pEvent->position().toPoint() - m_ptPanStart;  // 20260322 ZJH 移动量
        m_ptPanStart = pEvent->position().toPoint();                     // 20260322 ZJH 更新起始位置

        // 20260322 ZJH 通过滚动条移动视图（注意方向取反）
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - ptDelta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - ptDelta.y());

        pEvent->accept();  // 20260322 ZJH 消费事件
        return;            // 20260322 ZJH 平移时不做像素追踪
    }

    // 20260322 ZJH 更新鼠标位置信息（像素坐标和像素值）
    updateMouseInfo(pEvent->position().toPoint());

    // 20260322 ZJH 传递给基类
    QGraphicsView::mouseMoveEvent(pEvent);
}

// 20260322 ZJH 鼠标释放事件：结束平移
void ZoomableGraphicsView::mouseReleaseEvent(QMouseEvent* pEvent)
{
    // 20260322 ZJH 中键释放 → 结束平移
    if (pEvent->button() == Qt::MiddleButton && m_bPanning) {
        m_bPanning = false;              // 20260322 ZJH 退出平移模式
        setCursor(Qt::ArrowCursor);       // 20260322 ZJH 恢复默认光标
        pEvent->accept();                 // 20260322 ZJH 消费事件
        return;
    }

    // 20260322 ZJH 其他按键交给基类
    QGraphicsView::mouseReleaseEvent(pEvent);
}

// 20260322 ZJH 窗口大小变化事件：自适应缩放
void ZoomableGraphicsView::resizeEvent(QResizeEvent* pEvent)
{
    // 20260322 ZJH 调用基类处理
    QGraphicsView::resizeEvent(pEvent);

    // 20260322 ZJH 如果处于自适应模式且有图像，重新适应
    if (m_bFitOnResize && !m_currentImage.isNull()) {
        fitInView();  // 20260322 ZJH 重新计算缩放以适应新窗口大小
    }
}

// 20260322 ZJH 执行缩放操作（以指定场景点为中心）
void ZoomableGraphicsView::performZoom(double dFactor, const QPointF& ptSceneCenter)
{
    // 20260322 ZJH 计算新的缩放因子
    double dNewZoom = m_dZoomFactor * dFactor;

    // 20260322 ZJH 限制缩放范围
    dNewZoom = qBound(m_dMinZoom, dNewZoom, m_dMaxZoom);

    // 20260322 ZJH 如果缩放因子未变化（已到边界），直接返回
    if (qFuzzyCompare(dNewZoom, m_dZoomFactor)) {
        return;
    }

    // 20260322 ZJH 以鼠标场景坐标为中心缩放（正确算法）
    // 1. 记录鼠标在视图中的像素位置（缩放前）
    QPointF ptViewPos = mapFromScene(ptSceneCenter);

    // 2. 应用新缩放
    m_dZoomFactor = dNewZoom;
    resetTransform();
    setTransform(QTransform::fromScale(m_dZoomFactor, m_dZoomFactor));

    // 3. 缩放后鼠标场景坐标映射到了新的视图位置，计算视图像素偏移
    QPointF ptNewViewPos = mapFromScene(ptSceneCenter);
    QPointF ptViewDelta = ptNewViewPos - ptViewPos;  // 20260322 ZJH 视图像素偏移

    // 4. 通过滚动条补偿，让鼠标指向的场景点保持在视图同一位置
    horizontalScrollBar()->setValue(
        horizontalScrollBar()->value() + static_cast<int>(ptViewDelta.x()));
    verticalScrollBar()->setValue(
        verticalScrollBar()->value() + static_cast<int>(ptViewDelta.y()));

    // 20260322 ZJH 退出自适应模式（用户手动缩放后不再自动适应）
    m_bFitOnResize = false;

    // 20260322 ZJH 发射缩放变化信号
    emit zoomChanged(m_dZoomFactor * 100.0);
}

// 20260322 ZJH 根据视图坐标更新鼠标信息
void ZoomableGraphicsView::updateMouseInfo(const QPoint& ptViewPos)
{
    // 20260322 ZJH 检查是否有图像
    if (m_currentImage.isNull()) {
        return;  // 20260322 ZJH 无图像时不追踪
    }

    // 20260322 ZJH 视图坐标 → 场景坐标
    QPointF ptScene = mapToScene(ptViewPos);

    // 20260322 ZJH 场景坐标 → 图像像素坐标（取整）
    QPoint ptImage(static_cast<int>(ptScene.x()), static_cast<int>(ptScene.y()));

    // 20260322 ZJH 发射鼠标位置变化信号
    emit mousePositionChanged(ptScene, ptImage);

    // 20260322 ZJH 检查像素坐标是否在图像范围内
    if (ptImage.x() >= 0 && ptImage.x() < m_currentImage.width() &&
        ptImage.y() >= 0 && ptImage.y() < m_currentImage.height()) {
        // 20260322 ZJH 获取像素的 ARGB 值
        QRgb rgbValue = m_currentImage.pixel(ptImage);
        // 20260322 ZJH 计算灰度值（标准加权公式）
        int nGray = qGray(rgbValue);
        // 20260322 ZJH 发射像素值信号
        emit pixelValueChanged(ptImage, nGray, rgbValue);
    }
}

// 20260322 ZJH 生成棋盘格背景画刷
// 用于显示透明图像时提供可视化参考
QBrush ZoomableGraphicsView::createCheckerboardBrush() const
{
    // 20260322 ZJH 棋盘格单元大小（像素）
    const int nCellSize = 16;

    // 20260322 ZJH 创建 2x2 单元的棋盘格位图
    QPixmap pixChecker(nCellSize * 2, nCellSize * 2);

    // 20260322 ZJH 使用 QPainter 绘制棋盘格图案
    QPainter painter(&pixChecker);
    // 20260322 ZJH 深灰色单元
    painter.fillRect(0, 0, nCellSize * 2, nCellSize * 2, QColor(50, 50, 50));
    // 20260322 ZJH 浅灰色交替单元
    painter.fillRect(0, 0, nCellSize, nCellSize, QColor(60, 60, 60));
    painter.fillRect(nCellSize, nCellSize, nCellSize, nCellSize, QColor(60, 60, 60));
    painter.end();

    // 20260322 ZJH 返回以棋盘格位图为纹理的画刷
    return QBrush(pixChecker);
}
