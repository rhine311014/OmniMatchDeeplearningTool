// 20260322 ZJH ZoomableGraphicsView — 可缩放平移的图像查看器
// QGraphicsView 子类，支持滚轮缩放、中键/左键拖拽平移、
// 像素坐标追踪、棋盘格背景（透明图像友好）
#pragma once

#include <QGraphicsView>        // 20260322 ZJH 图形视图基类
#include <QGraphicsScene>       // 20260322 ZJH 场景管理图形项
#include <QGraphicsPixmapItem>  // 20260322 ZJH 图像显示项
#include <QImage>               // 20260322 ZJH 像素级图像操作
#include <QPixmap>              // 20260322 ZJH 显示用位图
#include <QPoint>               // 20260322 ZJH 整数坐标点
#include <QPointF>              // 20260322 ZJH 浮点坐标点

// 20260322 ZJH 可缩放平移图像查看器
// 功能：
//   1. 滚轮缩放（以鼠标位置为中心，步进 1.15x）
//   2. 中键/左键拖拽平移
//   3. Shift+鼠标移动报告像素坐标和 RGB 值
//   4. fitInView 等比缩放适应视图
//   5. 深灰色棋盘格背景（透明图像可见）
class ZoomableGraphicsView : public QGraphicsView
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化场景和默认设置
    // 参数: pParent - 父控件指针
    explicit ZoomableGraphicsView(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数（默认，子控件由 Qt 对象树管理）
    ~ZoomableGraphicsView() override = default;

    // 20260322 ZJH 设置显示图像（QImage 版本）
    // 参数: image - 要显示的图像
    void setImage(const QImage& image);

    // 20260322 ZJH 设置显示图像（QPixmap 版本）
    // 参数: pixmap - 要显示的位图
    void setImage(const QPixmap& pixmap);

    // 20260322 ZJH 清除当前显示的图像
    void clearImage();

    // 20260322 ZJH 等比缩放图像以适应视图大小
    void fitInView();

    // 20260404 ZJH 等比缩放到指定矩形区域（用于检查页双击跳转定位标注）
    // 参数: rect - 目标矩形（场景坐标）
    void fitInView(const QRectF& rect);

    // 20260322 ZJH 缩放到 1:1 实际像素大小
    void zoomToActualSize();

    // 20260322 ZJH 获取当前缩放百分比
    // 返回: 缩放百分比（100.0 = 100%）
    double zoomPercent() const;

    // 20260322 ZJH 设置缩放百分比
    // 参数: dPercent - 目标缩放百分比（如 125.0 表示 125%）
    void setZoomPercent(double dPercent);

signals:
    // 20260322 ZJH 缩放比例变化信号
    // 参数: dPercent - 新的缩放百分比
    void zoomChanged(double dPercent);

    // 20260322 ZJH 鼠标位置变化信号（场景坐标 + 图像像素坐标）
    // 参数: ptScene - 场景坐标; ptImage - 图像像素坐标
    void mousePositionChanged(const QPointF& ptScene, const QPoint& ptImage);

    // 20260322 ZJH 鼠标所在像素值信号
    // 参数: ptImage - 图像像素坐标; nGray - 灰度值; rgbValue - ARGB 值
    void pixelValueChanged(const QPoint& ptImage, int nGray, QRgb rgbValue);

protected:
    // 20260322 ZJH 滚轮事件：以鼠标位置为中心缩放
    void wheelEvent(QWheelEvent* pEvent) override;

    // 20260322 ZJH 鼠标按下事件：开始平移（中键或左键无标注工具时）
    void mousePressEvent(QMouseEvent* pEvent) override;

    // 20260322 ZJH 鼠标移动事件：处理平移和像素坐标追踪
    void mouseMoveEvent(QMouseEvent* pEvent) override;

    // 20260322 ZJH 鼠标释放事件：结束平移
    void mouseReleaseEvent(QMouseEvent* pEvent) override;

    // 20260322 ZJH 窗口大小变化事件：自适应缩放
    void resizeEvent(QResizeEvent* pEvent) override;

private:
    // 20260322 ZJH 执行缩放操作
    // 参数: dFactor - 缩放因子; ptCenter - 缩放中心点（场景坐标）
    void performZoom(double dFactor, const QPointF& ptCenter);

    // 20260322 ZJH 根据视图坐标更新鼠标信息并发射信号
    // 参数: ptViewPos - 视图坐标
    void updateMouseInfo(const QPoint& ptViewPos);

    // 20260322 ZJH 生成棋盘格背景画刷
    // 返回: 棋盘格 QBrush
    QBrush createCheckerboardBrush() const;

    QGraphicsScene* m_pScene;           // 20260322 ZJH 图形场景
    QGraphicsPixmapItem* m_pPixmapItem; // 20260322 ZJH 图像显示项
    QImage m_currentImage;              // 20260322 ZJH 当前加载的原始图像（用于像素查询）
    double m_dZoomFactor = 1.0;         // 20260322 ZJH 当前缩放因子（1.0 = 100%）
    double m_dMinZoom = 0.01;           // 20260322 ZJH 最小缩放因子（1%）
    double m_dMaxZoom = 50.0;           // 20260322 ZJH 最大缩放因子（5000%）
    double m_dZoomStep = 1.15;          // 20260322 ZJH 滚轮缩放步进倍率
    bool m_bPanning = false;            // 20260322 ZJH 是否正在拖拽平移
    QPoint m_ptPanStart;                // 20260322 ZJH 平移起始的视图坐标
    bool m_bFitOnResize = true;         // 20260322 ZJH 窗口大小变化时是否自动适应
};
