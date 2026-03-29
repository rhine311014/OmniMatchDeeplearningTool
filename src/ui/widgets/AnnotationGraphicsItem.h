// 20260322 ZJH AnnotationGraphicsItem — 标注图形项
// QGraphicsItem 子类，在图像上绘制标注（矩形/多边形）
// 支持选中高亮、缩放手柄、拖拽移动、标签文字显示
#pragma once

#include <QGraphicsItem>     // 20260322 ZJH 图形项基类
#include <QPainter>          // 20260322 ZJH 绘制
#include <QRectF>            // 20260322 ZJH 矩形区域
#include <QPolygonF>         // 20260322 ZJH 多边形顶点
#include <QColor>            // 20260322 ZJH 标签颜色
#include <QString>           // 20260322 ZJH 字符串
#include <QVector>           // 20260322 ZJH 手柄位置列表

#include "core/data/Annotation.h"  // 20260322 ZJH AnnotationType 枚举

// 20260322 ZJH 标注图形项
// 功能：
//   1. 支持 Rect/Polygon 两种标注类型绘制
//   2. 绘制边框 + 半透明填充 + 标签文字
//   3. 选中时显示 8 个缩放手柄
//   4. 可拖拽移动（通过 itemChange 通知外部）
//   5. 存储关联的 Annotation UUID
class AnnotationGraphicsItem : public QGraphicsItem
{
public:
    // 20260322 ZJH 自定义图形项类型标识（用于 qgraphicsitem_cast）
    enum { Type = UserType + 100 };

    // 20260322 ZJH 构造函数
    // 参数: strUuid - 关联的 Annotation UUID
    //       eType - 标注类型（Rect/Polygon）
    //       pParent - 父图形项（通常为 nullptr）
    explicit AnnotationGraphicsItem(const QString& strUuid,
                                     AnnotationType eType,
                                     QGraphicsItem* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~AnnotationGraphicsItem() override = default;

    // 20260322 ZJH 返回自定义类型标识
    int type() const override;

    // ===== 数据访问 =====

    // 20260322 ZJH 获取关联的 Annotation UUID
    QString uuid() const;

    // 20260322 ZJH 获取标注类型
    AnnotationType annotationType() const;

    // ===== 几何设置 =====

    // 20260322 ZJH 设置矩形标注的边界矩形（场景坐标）
    // 参数: rect - 边界矩形
    void setAnnotationRect(const QRectF& rect);

    // 20260322 ZJH 获取矩形标注的边界矩形
    QRectF annotationRect() const;

    // 20260322 ZJH 设置多边形标注的顶点列表（场景坐标）
    // 参数: polygon - 多边形顶点列表
    void setAnnotationPolygon(const QPolygonF& polygon);

    // 20260322 ZJH 获取多边形标注的顶点列表
    QPolygonF annotationPolygon() const;

    // ===== 外观设置 =====

    // 20260322 ZJH 设置标签颜色（用于边框和填充）
    // 参数: color - 标签颜色
    void setLabelColor(const QColor& color);

    // 20260322 ZJH 设置标签名称文字（显示在标注左上角）
    // 参数: strName - 标签名称
    void setLabelName(const QString& strName);

    // 20260322 ZJH 设置标签 ID
    // 参数: nId - 标签 ID
    void setLabelId(int nId);

    // 20260322 ZJH 获取标签 ID
    int labelId() const;

    // ===== QGraphicsItem 重写 =====

    // 20260322 ZJH 返回图形项的边界矩形（场景坐标）
    QRectF boundingRect() const override;

    // 20260322 ZJH 返回图形项的精确形状（用于碰撞检测和选择）
    QPainterPath shape() const override;

    // 20260322 ZJH 绘制标注图形
    void paint(QPainter* pPainter,
               const QStyleOptionGraphicsItem* pOption,
               QWidget* pWidget) override;

protected:
    // 20260322 ZJH 图形项变化通知（处理位置移动）
    QVariant itemChange(GraphicsItemChange eChange, const QVariant& value) override;

private:
    // 20260322 ZJH 计算 8 个缩放手柄的位置（矩形标注）
    QVector<QRectF> handleRects() const;

    // 20260322 ZJH 绘制矩形标注
    void paintRect(QPainter* pPainter);

    // 20260322 ZJH 绘制多边形标注
    void paintPolygon(QPainter* pPainter);

    // 20260322 ZJH 绘制标签名称文字
    void paintLabel(QPainter* pPainter, const QPointF& ptTopLeft);

    // 20260322 ZJH 绘制选中手柄
    void paintHandles(QPainter* pPainter);

    QString m_strUuid;            // 20260322 ZJH 关联的 Annotation UUID
    AnnotationType m_eType;       // 20260322 ZJH 标注类型
    QRectF m_rectAnnotation;      // 20260322 ZJH 矩形标注的边界矩形（本地坐标）
    QPolygonF m_polygon;          // 20260322 ZJH 多边形标注的顶点列表（本地坐标）
    QColor m_labelColor;          // 20260322 ZJH 标签颜色
    QString m_strLabelName;       // 20260322 ZJH 标签名称
    int m_nLabelId = -1;          // 20260322 ZJH 标签 ID

    // 20260322 ZJH 缩放手柄大小（像素）
    static constexpr double s_dHandleSize = 8.0;
};
