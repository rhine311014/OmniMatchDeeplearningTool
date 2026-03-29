// 20260322 ZJH AnnotationGraphicsItem 实现
// 标注图形项绘制：矩形/多边形边框、半透明填充、标签文字、选中手柄

#include "ui/widgets/AnnotationGraphicsItem.h"  // 20260322 ZJH 类声明

#include <QStyleOptionGraphicsItem>  // 20260322 ZJH 绘制选项
#include <QFont>                     // 20260322 ZJH 标签文字字体
#include <QFontMetrics>              // 20260322 ZJH 文字尺寸测量
#include <QPen>                      // 20260322 ZJH 边框画笔

// 20260322 ZJH 构造函数，初始化标注参数并设置图形项标志
AnnotationGraphicsItem::AnnotationGraphicsItem(const QString& strUuid,
                                                 AnnotationType eType,
                                                 QGraphicsItem* pParent)
    : QGraphicsItem(pParent)        // 20260322 ZJH 初始化基类
    , m_strUuid(strUuid)            // 20260322 ZJH 保存 UUID
    , m_eType(eType)                // 20260322 ZJH 保存标注类型
    , m_labelColor(Qt::green)       // 20260322 ZJH 默认标签颜色为绿色
    , m_nLabelId(-1)                // 20260322 ZJH 初始无标签
{
    // 20260322 ZJH 设置图形项可选中
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    // 20260322 ZJH 设置图形项可移动
    setFlag(QGraphicsItem::ItemIsMovable, true);
    // 20260322 ZJH 启用位置变化通知（itemChange 回调）
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
}

// 20260322 ZJH 返回自定义类型标识
int AnnotationGraphicsItem::type() const
{
    return Type;  // 20260322 ZJH UserType + 100
}

// 20260322 ZJH 获取关联的 UUID
QString AnnotationGraphicsItem::uuid() const
{
    return m_strUuid;
}

// 20260322 ZJH 获取标注类型
AnnotationType AnnotationGraphicsItem::annotationType() const
{
    return m_eType;
}

// 20260322 ZJH 设置矩形标注的边界矩形
void AnnotationGraphicsItem::setAnnotationRect(const QRectF& rect)
{
    // 20260322 ZJH 通知场景边界即将变化
    prepareGeometryChange();
    // 20260322 ZJH 将场景坐标矩形存储为本地坐标（通过 pos() 偏移）
    m_rectAnnotation = rect;
    // 20260322 ZJH 设置图形项的位置为矩形左上角
    setPos(rect.topLeft());
    // 20260322 ZJH 本地矩形原点归零
    m_rectAnnotation.moveTopLeft(QPointF(0, 0));
}

// 20260322 ZJH 获取矩形标注的边界矩形（场景坐标）
QRectF AnnotationGraphicsItem::annotationRect() const
{
    // 20260322 ZJH 本地坐标 + 位置偏移 = 场景坐标
    return m_rectAnnotation.translated(pos());
}

// 20260322 ZJH 设置多边形标注的顶点列表
void AnnotationGraphicsItem::setAnnotationPolygon(const QPolygonF& polygon)
{
    // 20260322 ZJH 通知场景边界即将变化
    prepareGeometryChange();

    // 20260322 ZJH 计算多边形的外接矩形
    QRectF rectBounds = polygon.boundingRect();

    // 20260322 ZJH 设置图形项位置为外接矩形左上角
    setPos(rectBounds.topLeft());

    // 20260322 ZJH 将多边形顶点转为本地坐标（减去位置偏移）
    m_polygon.clear();
    for (const QPointF& pt : polygon) {
        m_polygon.append(pt - rectBounds.topLeft());
    }

    // 20260322 ZJH 同时更新本地边界矩形
    m_rectAnnotation = QRectF(QPointF(0, 0), rectBounds.size());
}

// 20260322 ZJH 获取多边形标注的顶点列表（场景坐标）
QPolygonF AnnotationGraphicsItem::annotationPolygon() const
{
    // 20260322 ZJH 本地坐标 + 位置偏移 = 场景坐标
    QPolygonF scenePolygon;
    for (const QPointF& pt : m_polygon) {
        scenePolygon.append(pt + pos());
    }
    return scenePolygon;
}

// 20260322 ZJH 设置标签颜色
void AnnotationGraphicsItem::setLabelColor(const QColor& color)
{
    m_labelColor = color;  // 20260322 ZJH 记录颜色
    update();              // 20260322 ZJH 触发重绘
}

// 20260322 ZJH 设置标签名称
void AnnotationGraphicsItem::setLabelName(const QString& strName)
{
    m_strLabelName = strName;  // 20260322 ZJH 记录名称
    update();                  // 20260322 ZJH 触发重绘
}

// 20260322 ZJH 设置标签 ID
void AnnotationGraphicsItem::setLabelId(int nId)
{
    m_nLabelId = nId;  // 20260322 ZJH 记录 ID
}

// 20260322 ZJH 获取标签 ID
int AnnotationGraphicsItem::labelId() const
{
    return m_nLabelId;
}

// 20260322 ZJH 返回图形项的边界矩形
QRectF AnnotationGraphicsItem::boundingRect() const
{
    // 20260322 ZJH 扩展边界以包含手柄和标签文字
    double dPadding = s_dHandleSize + 2.0;  // 20260322 ZJH 手柄大小 + 额外边距
    return m_rectAnnotation.adjusted(-dPadding, -dPadding - 20, dPadding, dPadding);
}

// 20260322 ZJH 返回精确形状（用于鼠标点击检测）
QPainterPath AnnotationGraphicsItem::shape() const
{
    QPainterPath path;
    if (m_eType == AnnotationType::Rect) {
        // 20260322 ZJH 矩形标注：使用矩形路径
        path.addRect(m_rectAnnotation);
    } else if (m_eType == AnnotationType::Polygon) {
        // 20260322 ZJH 多边形标注：使用多边形路径
        if (!m_polygon.isEmpty()) {
            path.addPolygon(m_polygon);
            path.closeSubpath();  // 20260322 ZJH 闭合路径
        }
    }
    return path;
}

// 20260322 ZJH 绘制标注图形
void AnnotationGraphicsItem::paint(QPainter* pPainter,
                                    const QStyleOptionGraphicsItem* pOption,
                                    QWidget* pWidget)
{
    Q_UNUSED(pOption);   // 20260322 ZJH 未使用参数
    Q_UNUSED(pWidget);   // 20260322 ZJH 未使用参数

    pPainter->save();  // 20260323 ZJH 保存画笔状态，防止修改泄漏到后续 item

    // 20260322 ZJH 根据标注类型分发绘制
    if (m_eType == AnnotationType::Rect) {
        paintRect(pPainter);      // 20260322 ZJH 绘制矩形标注
    } else if (m_eType == AnnotationType::Polygon) {
        paintPolygon(pPainter);   // 20260322 ZJH 绘制多边形标注
    }

    // 20260322 ZJH 如果选中，绘制缩放手柄
    if (isSelected()) {
        paintHandles(pPainter);   // 20260322 ZJH 绘制 8 个手柄
    }

    pPainter->restore();  // 20260323 ZJH 恢复画笔状态
}

// 20260322 ZJH 图形项变化通知
QVariant AnnotationGraphicsItem::itemChange(GraphicsItemChange eChange, const QVariant& value)
{
    // 20260322 ZJH 位置变化时可在此处理（如限制移动范围）
    if (eChange == ItemPositionHasChanged) {
        // 20260322 ZJH 位置已变化，后续由 AnnotationController 处理同步
    }
    return QGraphicsItem::itemChange(eChange, value);
}

// 20260322 ZJH 计算 8 个缩放手柄的位置
QVector<QRectF> AnnotationGraphicsItem::handleRects() const
{
    QVector<QRectF> vecHandles;
    double dHalf = s_dHandleSize / 2.0;  // 20260322 ZJH 手柄半尺寸

    QRectF r = m_rectAnnotation;  // 20260322 ZJH 标注矩形

    // 20260322 ZJH 8 个手柄位置：四角 + 四边中点
    // 左上角
    vecHandles.append(QRectF(r.left() - dHalf, r.top() - dHalf, s_dHandleSize, s_dHandleSize));
    // 上中
    vecHandles.append(QRectF(r.center().x() - dHalf, r.top() - dHalf, s_dHandleSize, s_dHandleSize));
    // 右上角
    vecHandles.append(QRectF(r.right() - dHalf, r.top() - dHalf, s_dHandleSize, s_dHandleSize));
    // 右中
    vecHandles.append(QRectF(r.right() - dHalf, r.center().y() - dHalf, s_dHandleSize, s_dHandleSize));
    // 右下角
    vecHandles.append(QRectF(r.right() - dHalf, r.bottom() - dHalf, s_dHandleSize, s_dHandleSize));
    // 下中
    vecHandles.append(QRectF(r.center().x() - dHalf, r.bottom() - dHalf, s_dHandleSize, s_dHandleSize));
    // 左下角
    vecHandles.append(QRectF(r.left() - dHalf, r.bottom() - dHalf, s_dHandleSize, s_dHandleSize));
    // 左中
    vecHandles.append(QRectF(r.left() - dHalf, r.center().y() - dHalf, s_dHandleSize, s_dHandleSize));

    return vecHandles;
}

// 20260322 ZJH 绘制矩形标注
void AnnotationGraphicsItem::paintRect(QPainter* pPainter)
{
    // 20260322 ZJH 设置半透明填充
    QColor fillColor = m_labelColor;
    fillColor.setAlpha(40);  // 20260322 ZJH 填充透明度 40/255 ≈ 16%
    pPainter->setBrush(QBrush(fillColor));

    // 20260324 ZJH 设置边框画笔：宽度 0 = cosmetic pen，Qt 渲染为恒定 1 像素（不随缩放变化）
    QPen pen(m_labelColor, 0);  // 20260324 ZJH cosmetic pen，1px 恒定宽度
    if (isSelected()) {
        pen.setStyle(Qt::SolidLine);  // 20260324 ZJH 选中时保持实线，宽度仍为 1px cosmetic
    }
    pPainter->setPen(pen);

    // 20260322 ZJH 绘制矩形
    pPainter->drawRect(m_rectAnnotation);

    // 20260322 ZJH 绘制标签名称（矩形左上角偏上方）
    paintLabel(pPainter, m_rectAnnotation.topLeft());
}

// 20260322 ZJH 绘制多边形标注
void AnnotationGraphicsItem::paintPolygon(QPainter* pPainter)
{
    // 20260322 ZJH 检查多边形是否有顶点
    if (m_polygon.isEmpty()) {
        return;  // 20260322 ZJH 无顶点，不绘制
    }

    // 20260322 ZJH 设置半透明填充
    QColor fillColor = m_labelColor;
    fillColor.setAlpha(40);  // 20260322 ZJH 填充透明度 16%
    pPainter->setBrush(QBrush(fillColor));

    // 20260324 ZJH 设置边框画笔：宽度 0 = cosmetic pen，1px 恒定宽度
    QPen pen(m_labelColor, 0);  // 20260324 ZJH cosmetic pen，不随缩放变化
    if (isSelected()) {
        pen.setStyle(Qt::SolidLine);  // 20260324 ZJH 选中时保持实线
    }
    pPainter->setPen(pen);

    // 20260322 ZJH 绘制多边形
    pPainter->drawPolygon(m_polygon);

    // 20260322 ZJH 绘制标签名称（多边形外接矩形左上角）
    paintLabel(pPainter, m_polygon.boundingRect().topLeft());
}

// 20260322 ZJH 绘制标签名称文字
void AnnotationGraphicsItem::paintLabel(QPainter* pPainter, const QPointF& ptTopLeft)
{
    // 20260322 ZJH 检查标签名称是否为空
    if (m_strLabelName.isEmpty()) {
        return;  // 20260322 ZJH 无名称不绘制
    }

    // 20260322 ZJH 设置标签文字字体
    QFont font;
    font.setPixelSize(12);        // 20260322 ZJH 12px 字号
    font.setBold(true);           // 20260322 ZJH 加粗
    pPainter->setFont(font);

    // 20260322 ZJH 计算文字尺寸
    QFontMetrics fm(font);
    int nTextWidth = fm.horizontalAdvance(m_strLabelName) + 8;  // 20260322 ZJH 文字宽度 + 左右 4px 边距
    int nTextHeight = fm.height() + 4;                            // 20260322 ZJH 文字高度 + 上下 2px 边距

    // 20260322 ZJH 标签背景矩形（在标注上方）
    QRectF rectLabelBg(ptTopLeft.x(), ptTopLeft.y() - nTextHeight - 2,
                       nTextWidth, nTextHeight);

    // 20260322 ZJH 绘制标签背景
    pPainter->setPen(Qt::NoPen);
    pPainter->setBrush(m_labelColor);
    pPainter->drawRoundedRect(rectLabelBg, 2, 2);  // 20260322 ZJH 圆角 2px

    // 20260322 ZJH 绘制标签文字（白色）
    pPainter->setPen(Qt::white);
    pPainter->drawText(rectLabelBg, Qt::AlignCenter, m_strLabelName);
}

// 20260322 ZJH 绘制选中手柄
void AnnotationGraphicsItem::paintHandles(QPainter* pPainter)
{
    // 20260322 ZJH 手柄只在矩形标注或有外接矩形时绘制
    QVector<QRectF> vecHandles = handleRects();

    // 20260322 ZJH 设置手柄样式
    pPainter->setPen(QPen(Qt::white, 1.0));          // 20260322 ZJH 白色边框
    pPainter->setBrush(QBrush(m_labelColor));         // 20260322 ZJH 标签颜色填充

    // 20260322 ZJH 逐个绘制手柄
    for (const QRectF& rectHandle : vecHandles) {
        pPainter->drawRect(rectHandle);  // 20260322 ZJH 绘制方形手柄
    }
}
