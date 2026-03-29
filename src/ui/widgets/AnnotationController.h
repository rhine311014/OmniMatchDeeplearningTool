// 20260322 ZJH AnnotationController — 标注控制器
// 管理标注工具状态、鼠标交互、标注图形项和撤销/重做堆栈
// 将鼠标事件转化为标注操作（创建/选择/移动/删除）
#pragma once

#include <QObject>          // 20260322 ZJH 信号槽基类
#include <QPointF>          // 20260322 ZJH 浮点坐标点
#include <QRectF>           // 20260322 ZJH 矩形区域
#include <QPolygonF>        // 20260322 ZJH 多边形
#include <QColor>           // 20260322 ZJH 颜色
#include <QString>          // 20260322 ZJH 字符串
#include <QMap>             // 20260322 ZJH UUID → 图形项映射
#include <QUndoStack>       // 20260322 ZJH 撤销/重做堆栈

#include "core/data/Annotation.h"  // 20260322 ZJH Annotation 结构体和类型枚举

// 20260322 ZJH 前向声明
class QGraphicsScene;
struct ImageEntry;
class ImageDataset;
class AnnotationGraphicsItem;

// 20260322 ZJH 标注工具枚举
// 定义当前鼠标交互模式
enum class AnnotationTool {
    Select  = 0,  // 20260322 ZJH 选择工具：点击选中/拖拽移动标注
    Rect    = 1,  // 20260322 ZJH 矩形工具：拖拽绘制矩形标注
    Polygon = 2,  // 20260322 ZJH 多边形工具：逐点绘制多边形标注
    Brush   = 3   // 20260322 ZJH 画笔工具：创建圆形标注（简化版）
};

// 20260322 ZJH 标注控制器
// 职责：
//   1. 管理当前工具和标签状态
//   2. 处理鼠标事件并转化为标注操作
//   3. 管理 QGraphicsScene 中的标注图形项
//   4. 通过 QUndoStack 支持撤销/重做
//   5. 同步标注数据到 ImageEntry
class AnnotationController : public QObject
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pScene - 图形场景（标注图形项添加到此场景）
    //       pParent - 父 QObject
    explicit AnnotationController(QGraphicsScene* pScene, QObject* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~AnnotationController() override;

    // ===== 工具和标签管理 =====

    // 20260322 ZJH 设置当前标注工具
    // 参数: eTool - 目标工具类型
    void setCurrentTool(AnnotationTool eTool);

    // 20260322 ZJH 获取当前标注工具
    AnnotationTool currentTool() const;

    // 20260322 ZJH 设置当前标签信息（后续创建的标注使用此标签）
    // 参数: nLabelId - 标签 ID
    //       strName - 标签名称
    //       color - 标签颜色
    void setCurrentLabel(int nLabelId, const QString& strName, const QColor& color);

    // ===== 数据绑定 =====

    // 20260322 ZJH 绑定图像条目和数据集（加载标注时需要两者）
    // 参数: pEntry - 当前图像条目指针; pDataset - 数据集指针
    void bindImage(ImageEntry* pEntry, ImageDataset* pDataset);

    // 20260322 ZJH 解除图像绑定（清空标注显示和中间状态）
    void unbindImage();

    // ===== 标注操作 =====

    // 20260322 ZJH 从绑定的 ImageEntry 加载标注到场景
    void loadAnnotationsFromEntry();

    // 20260322 ZJH 清除场景中的所有标注图形项
    void clearAllItems();

    // 20260322 ZJH 删除当前选中的标注（场景选择）
    void deleteSelectedAnnotation();

    // 20260324 ZJH 按 UUID 删除指定标注（从标注列表触发）
    void deleteAnnotationByUuid(const QString& strUuid);

    // 20260324 ZJH 按 UUID 选中场景中的标注（列表→场景同步）
    void selectAnnotationByUuid(const QString& strUuid);

    // 20260324 ZJH 检查场景中是否有选中的标注图形项
    bool hasSceneSelection() const;

    // 20260322 ZJH 撤销上一步操作
    void undo();

    // 20260322 ZJH 重做上一步被撤销的操作
    void redo();

    // 20260322 ZJH 设置画笔半径（仅画笔工具有效）
    // 参数: nRadius - 画笔半径（像素）
    void setBrushRadius(int nRadius);

    // ===== 鼠标事件处理 =====
    // 由 ZoomableGraphicsView 或 ImagePage 调用
    // 返回 true 表示事件已处理，调用者不应再传递

    // 20260322 ZJH 处理鼠标按下事件
    bool handleMousePress(const QPointF& ptScene, Qt::MouseButton btn, Qt::KeyboardModifiers mods);

    // 20260322 ZJH 处理鼠标移动事件
    bool handleMouseMove(const QPointF& ptScene, Qt::MouseButtons btns);

    // 20260322 ZJH 处理鼠标释放事件
    bool handleMouseRelease(const QPointF& ptScene, Qt::MouseButton btn);

    // 20260322 ZJH 处理鼠标双击事件
    bool handleMouseDoubleClick(const QPointF& ptScene, Qt::MouseButton btn);

    // ===== 供 UndoCommand 直接操作的方法 =====
    // 这些方法不创建 UndoCommand，直接执行底层操作

    // 20260322 ZJH 直接添加标注（用于 redo/undo 恢复）
    void addAnnotationDirect(const Annotation& annotation);

    // 20260322 ZJH 直接删除标注（用于 redo/undo 恢复）
    void removeAnnotationDirect(const QString& strUuid);

    // 20260322 ZJH 直接移动标注（用于 redo/undo 恢复）
    void moveAnnotationDirect(const QString& strUuid, const QRectF& rect, const QPolygonF& polygon);

    // 20260322 ZJH 获取撤销堆栈指针（供外部连接 canUndo/canRedo 信号）
    QUndoStack* undoStack() const;

signals:
    // 20260322 ZJH 标注内容发生变更信号
    void annotationChanged();

    // 20260322 ZJH 选中的标注变化信号
    // 参数: strUuid - 选中的标注 UUID（空表示取消选中）
    void selectionChanged(const QString& strUuid);

    // 20260322 ZJH 标注数量变化信号
    // 参数: nCount - 当前标注总数
    void annotationCountChanged(int nCount);

    // 20260324 ZJH 请求视图居中显示指定标注信号
    // 参数: pItem - 需要居中显示的标注图形项指针
    void requestCenterOnItem(AnnotationGraphicsItem* pItem);

private:
    // 20260322 ZJH 创建标注图形项并添加到场景
    // 参数: annotation - 标注数据
    // 返回: 创建的图形项指针
    AnnotationGraphicsItem* createGraphicsItem(const Annotation& annotation);

    // 20260322 ZJH 将标注数据同步到 ImageEntry
    void syncToEntry();

    // 20260322 ZJH 查找指定标签的颜色和名称
    void findLabelInfo(int nLabelId, QColor& color, QString& strName) const;

    // 20260322 ZJH 完成矩形绘制（创建标注并推入 UndoStack）
    void finishRectDrawing();

    // 20260322 ZJH 完成多边形绘制
    void finishPolygonDrawing();

    // 20260322 ZJH 完成画笔标注
    void finishBrushAnnotation(const QPointF& ptCenter);

    QGraphicsScene* m_pScene;                  // 20260322 ZJH 图形场景指针
    ImageEntry* m_pEntry = nullptr;             // 20260322 ZJH 当前绑定的图像条目
    ImageDataset* m_pDataset = nullptr;         // 20260322 ZJH 当前数据集指针
    QUndoStack* m_pUndoStack;                   // 20260322 ZJH 撤销/重做堆栈

    // ===== 工具状态 =====
    AnnotationTool m_eTool = AnnotationTool::Select;  // 20260322 ZJH 当前工具
    int m_nCurrentLabelId = -1;                         // 20260322 ZJH 当前标签 ID
    QString m_strCurrentLabelName;                      // 20260322 ZJH 当前标签名称
    QColor m_currentLabelColor = Qt::green;             // 20260322 ZJH 当前标签颜色
    int m_nBrushRadius = 10;                            // 20260322 ZJH 画笔半径

    // ===== 绘制中间状态 =====
    bool m_bDrawing = false;                   // 20260322 ZJH 是否正在绘制标注
    QPointF m_ptDrawStart;                     // 20260322 ZJH 矩形绘制起点
    QPolygonF m_drawingPolygon;                // 20260322 ZJH 多边形绘制中的顶点列表
    AnnotationGraphicsItem* m_pDrawingItem = nullptr;  // 20260322 ZJH 绘制中的临时图形项

    // ===== 移动状态 =====
    bool m_bMoving = false;                    // 20260322 ZJH 是否正在移动标注
    QRectF m_moveStartRect;                    // 20260322 ZJH 移动前的矩形
    QPolygonF m_moveStartPolygon;              // 20260322 ZJH 移动前的多边形
    QString m_strMovingUuid;                   // 20260322 ZJH 正在移动的标注 UUID

    // ===== 图形项映射 =====
    QMap<QString, AnnotationGraphicsItem*> m_mapItems;  // 20260322 ZJH UUID → 图形项映射
};
