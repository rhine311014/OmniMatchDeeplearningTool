// 20260322 ZJH AnnotationController 实现
// 标注控制器：工具状态管理、鼠标事件分发、标注 CRUD、撤销/重做

#include "ui/widgets/AnnotationController.h"       // 20260322 ZJH 类声明
#include "ui/widgets/AnnotationGraphicsItem.h"     // 20260322 ZJH 标注图形项
#include "ui/widgets/AnnotationCommands.h"         // 20260322 ZJH 撤销命令

#include "core/data/ImageEntry.h"    // 20260322 ZJH 图像条目
#include "core/data/ImageDataset.h"  // 20260322 ZJH 数据集管理
#include "core/data/LabelInfo.h"     // 20260322 ZJH 标签信息

#include <QGraphicsScene>  // 20260322 ZJH 图形场景
#include <QUuid>           // 20260322 ZJH UUID 生成
#include <QtMath>          // 20260322 ZJH qSqrt 等数学函数
#include <QTimer>          // 20260324 ZJH 定时器，用于标注闪烁高亮效果

// 20260322 ZJH 构造函数
AnnotationController::AnnotationController(QGraphicsScene* pScene, QObject* pParent)
    : QObject(pParent)                  // 20260322 ZJH 初始化基类
    , m_pScene(pScene)                  // 20260322 ZJH 保存场景指针
    , m_pEntry(nullptr)                 // 20260322 ZJH 初始无绑定图像
    , m_pDataset(nullptr)               // 20260322 ZJH 初始无数据集
    , m_pUndoStack(new QUndoStack(this))  // 20260322 ZJH 创建撤销堆栈
    , m_eTool(AnnotationTool::Select)   // 20260322 ZJH 默认选择工具
    , m_nCurrentLabelId(-1)             // 20260322 ZJH 初始无标签
    , m_nBrushRadius(10)                // 20260322 ZJH 默认画笔半径 10px
    , m_bDrawing(false)                 // 20260322 ZJH 初始非绘制状态
    , m_pDrawingItem(nullptr)           // 20260322 ZJH 初始无绘制项
    , m_bMoving(false)                  // 20260322 ZJH 初始非移动状态
{
}

// 20260322 ZJH 析构函数
AnnotationController::~AnnotationController()
{
    // 20260322 ZJH Qt 对象树自动管理 m_pUndoStack 的析构
    // 图形项由 QGraphicsScene 管理
}

// ===== 工具和标签管理 =====

// 20260322 ZJH 设置当前工具
void AnnotationController::setCurrentTool(AnnotationTool eTool)
{
    // 20260322 ZJH 如果正在绘制中切换工具，取消当前绘制
    if (m_bDrawing && m_pDrawingItem) {
        m_pScene->removeItem(m_pDrawingItem);  // 20260322 ZJH 从场景移除临时项
        delete m_pDrawingItem;                  // 20260322 ZJH 释放内存
        m_pDrawingItem = nullptr;
        m_bDrawing = false;
        m_drawingPolygon.clear();
    }
    m_eTool = eTool;  // 20260322 ZJH 记录新工具
}

// 20260322 ZJH 获取当前工具
AnnotationTool AnnotationController::currentTool() const
{
    return m_eTool;
}

// 20260322 ZJH 设置当前标签信息
void AnnotationController::setCurrentLabel(int nLabelId, const QString& strName, const QColor& color)
{
    m_nCurrentLabelId = nLabelId;          // 20260322 ZJH 记录标签 ID
    m_strCurrentLabelName = strName;       // 20260322 ZJH 记录标签名称
    m_currentLabelColor = color;           // 20260322 ZJH 记录标签颜色
}

// ===== 数据绑定 =====

// 20260322 ZJH 绑定图像条目和数据集
void AnnotationController::bindImage(ImageEntry* pEntry, ImageDataset* pDataset)
{
    // 20260322 ZJH 先解绑之前的图像
    unbindImage();

    m_pEntry = pEntry;      // 20260322 ZJH 绑定图像条目
    m_pDataset = pDataset;  // 20260322 ZJH 绑定数据集

    // 20260322 ZJH 加载标注到场景
    if (m_pEntry) {
        loadAnnotationsFromEntry();
    }
}

// 20260322 ZJH 解除图像绑定
void AnnotationController::unbindImage()
{
    // 20260322 ZJH 清除所有图形项
    clearAllItems();
    // 20260322 ZJH 清空撤销堆栈
    m_pUndoStack->clear();
    // 20260322 ZJH 重置绘制状态
    m_bDrawing = false;
    m_pDrawingItem = nullptr;
    m_drawingPolygon.clear();
    m_bMoving = false;
    // 20260322 ZJH 清空绑定
    m_pEntry = nullptr;
    m_pDataset = nullptr;
}

// ===== 标注操作 =====

// 20260322 ZJH 从 ImageEntry 加载标注到场景
void AnnotationController::loadAnnotationsFromEntry()
{
    // 20260322 ZJH 先清除现有图形项
    clearAllItems();

    // 20260322 ZJH 检查绑定是否有效
    if (!m_pEntry) {
        return;  // 20260322 ZJH 无绑定图像
    }

    // 20260322 ZJH 遍历图像的标注列表，为每个标注创建图形项
    for (const Annotation& anno : m_pEntry->vecAnnotations) {
        createGraphicsItem(anno);
    }

    // 20260322 ZJH 发射标注数量变化信号
    emit annotationCountChanged(m_mapItems.size());
}

// 20260322 ZJH 清除所有标注图形项
void AnnotationController::clearAllItems()
{
    // 20260322 ZJH 遍历映射表，从场景移除并删除所有图形项
    for (auto it = m_mapItems.begin(); it != m_mapItems.end(); ++it) {
        m_pScene->removeItem(it.value());  // 20260322 ZJH 从场景移除
        delete it.value();                  // 20260322 ZJH 释放内存
    }
    m_mapItems.clear();  // 20260322 ZJH 清空映射表
}

// 20260322 ZJH 删除当前选中的标注
void AnnotationController::deleteSelectedAnnotation()
{
    // 20260322 ZJH 获取场景中选中的图形项
    QList<QGraphicsItem*> vecSelected = m_pScene->selectedItems();
    if (vecSelected.isEmpty()) {
        return;  // 20260322 ZJH 无选中项
    }

    // 20260324 ZJH 先收集所有要删除的 UUID，避免循环中 push→redo→delete 导致
    // vecSelected 中的指针悬挂（removeAnnotationDirect 会 delete 图形项）
    QVector<QString> vecUuidsToDelete;
    vecUuidsToDelete.reserve(vecSelected.size());

    for (QGraphicsItem* pItem : vecSelected) {
        AnnotationGraphicsItem* pAnnoItem = dynamic_cast<AnnotationGraphicsItem*>(pItem);
        if (pAnnoItem) {
            vecUuidsToDelete.append(pAnnoItem->uuid());
        }
    }

    // 20260324 ZJH 逐个删除已收集的标注
    for (const QString& strUuid : vecUuidsToDelete) {
        // 20260324 ZJH 从 ImageEntry 中查找对应标注数据（用于撤销恢复）
        Annotation annoCopy;
        bool bFound = false;  // 20260324 ZJH 标记是否找到匹配标注

        if (m_pEntry) {
            for (const Annotation& anno : m_pEntry->vecAnnotations) {
                if (anno.strUuid == strUuid) {
                    annoCopy = anno;  // 20260322 ZJH 拷贝标注数据用于撤销
                    bFound = true;
                    break;
                }
            }
        }

        // 20260324 ZJH 如果在 m_pEntry 中未找到，使用图形项的 UUID 构建最小副本
        // 确保 removeAnnotationDirect 能通过 UUID 找到并删除场景中的图形项
        if (!bFound) {
            annoCopy.strUuid = strUuid;
        }

        // 20260324 ZJH 推入撤销堆栈（push 会立即调用 redo()，由命令的 redo 执行 removeAnnotationDirect）
        auto* pCmd = new DeleteAnnotationCommand(this, annoCopy);
        m_pUndoStack->push(pCmd);
    }

    // 20260324 ZJH 信号已由 removeAnnotationDirect() 在每次 redo 中发射
    // 此处不再重复发射，避免 N+1 次信号触发
}

// 20260324 ZJH 按 UUID 删除指定标注（从标注列表 UI 触发，不依赖场景选择状态）
void AnnotationController::deleteAnnotationByUuid(const QString& strUuid)
{
    if (strUuid.isEmpty()) {
        return;
    }

    // 20260324 ZJH 从 ImageEntry 中查找标注数据（用于撤销恢复）
    Annotation annoCopy;
    bool bFound = false;

    if (m_pEntry) {
        for (const Annotation& anno : m_pEntry->vecAnnotations) {
            if (anno.strUuid == strUuid) {
                annoCopy = anno;
                bFound = true;
                break;
            }
        }
    }

    // 20260324 ZJH 未找到时使用传入的 UUID 确保场景项能被删除
    if (!bFound) {
        annoCopy.strUuid = strUuid;
    }

    // 20260324 ZJH 推入撤销堆栈（push → redo → removeAnnotationDirect）
    auto* pCmd = new DeleteAnnotationCommand(this, annoCopy);
    m_pUndoStack->push(pCmd);
}

// 20260324 ZJH 按 UUID 选中场景中的标注（列表→场景同步）
// 选中后居中视图到标注位置，并闪烁高亮提示用户
void AnnotationController::selectAnnotationByUuid(const QString& strUuid)
{
    // 20260324 ZJH 先清除场景中所有选择
    m_pScene->clearSelection();

    // 20260324 ZJH 在映射表中查找对应图形项并选中
    if (m_mapItems.contains(strUuid)) {
        AnnotationGraphicsItem* pItem = m_mapItems[strUuid];
        pItem->setSelected(true);  // 20260324 ZJH 选中该标注
        emit selectionChanged(strUuid);  // 20260324 ZJH 通知选择变更

        // 20260324 ZJH 请求视图居中到该标注
        emit requestCenterOnItem(pItem);

        // 20260324 ZJH 闪烁高亮效果：临时降低不透明度 300ms 后恢复
        // QGraphicsItem 不继承 QObject，singleShot context 必须用 QObject*（this）
        qreal dOriginalOpacity = pItem->opacity();  // 20260324 ZJH 保存原始不透明度
        pItem->setOpacity(0.3);  // 20260324 ZJH 降低不透明度制造闪烁第一帧
        // 20260324 ZJH 150ms 后恢复到完全不透明（闪烁中间帧）
        QTimer::singleShot(150, this, [pItem]() {
            pItem->setOpacity(1.0);  // 20260324 ZJH 完全不透明，高亮
        });
        // 20260324 ZJH 300ms 后恢复到原始不透明度（闪烁结束）
        QTimer::singleShot(300, this, [pItem, dOriginalOpacity]() {
            pItem->setOpacity(dOriginalOpacity);  // 20260324 ZJH 恢复原始状态
        });
    }
}

// 20260324 ZJH 检查场景中是否有选中的标注图形项
bool AnnotationController::hasSceneSelection() const
{
    return !m_pScene->selectedItems().isEmpty();
}

// 20260322 ZJH 撤销
void AnnotationController::undo()
{
    m_pUndoStack->undo();  // 20260322 ZJH 执行撤销
}

// 20260322 ZJH 重做
void AnnotationController::redo()
{
    m_pUndoStack->redo();  // 20260322 ZJH 执行重做
}

// 20260322 ZJH 设置画笔半径
void AnnotationController::setBrushRadius(int nRadius)
{
    m_nBrushRadius = qBound(1, nRadius, 200);  // 20260322 ZJH 限制范围 1~200
}

// ===== 鼠标事件处理 =====

// 20260322 ZJH 处理鼠标按下事件
bool AnnotationController::handleMousePress(const QPointF& ptScene, Qt::MouseButton btn,
                                              Qt::KeyboardModifiers mods)
{
    Q_UNUSED(mods);  // 20260322 ZJH 修饰键暂未使用

    // 20260322 ZJH 只处理左键
    if (btn != Qt::LeftButton) {
        return false;
    }

    // 20260322 ZJH 根据当前工具分发
    switch (m_eTool) {
    case AnnotationTool::Select: {
        // 20260322 ZJH 选择工具：查找点击位置的标注
        QGraphicsItem* pHitItem = m_pScene->itemAt(ptScene, QTransform());
        AnnotationGraphicsItem* pAnnoItem = dynamic_cast<AnnotationGraphicsItem*>(pHitItem);

        if (pAnnoItem) {
            // 20260322 ZJH 选中标注
            m_pScene->clearSelection();
            pAnnoItem->setSelected(true);
            emit selectionChanged(pAnnoItem->uuid());

            // 20260322 ZJH 准备拖拽移动
            m_bMoving = true;
            m_strMovingUuid = pAnnoItem->uuid();
            m_moveStartRect = pAnnoItem->annotationRect();
            m_moveStartPolygon = pAnnoItem->annotationPolygon();
        } else {
            // 20260322 ZJH 点击空白区域，取消选中
            m_pScene->clearSelection();
            emit selectionChanged(QString());
        }
        return true;
    }
    case AnnotationTool::Rect: {
        // 20260322 ZJH 矩形工具：开始绘制
        m_bDrawing = true;
        m_ptDrawStart = ptScene;  // 20260322 ZJH 记录起点

        // 20260322 ZJH 创建临时图形项（0 尺寸矩形）
        Annotation tempAnno(AnnotationType::Rect);
        tempAnno.nLabelId = m_nCurrentLabelId;
        m_pDrawingItem = new AnnotationGraphicsItem(tempAnno.strUuid, AnnotationType::Rect);
        m_pDrawingItem->setAnnotationRect(QRectF(ptScene, QSizeF(0, 0)));
        m_pDrawingItem->setLabelColor(m_currentLabelColor);
        m_pDrawingItem->setLabelName(m_strCurrentLabelName);
        m_pDrawingItem->setLabelId(m_nCurrentLabelId);
        m_pDrawingItem->setFlag(QGraphicsItem::ItemIsSelectable, false);  // 20260322 ZJH 绘制中不可选
        m_pDrawingItem->setFlag(QGraphicsItem::ItemIsMovable, false);     // 20260322 ZJH 绘制中不可移动
        m_pScene->addItem(m_pDrawingItem);
        return true;
    }
    case AnnotationTool::Polygon: {
        // 20260322 ZJH 多边形工具：添加顶点
        if (!m_bDrawing) {
            // 20260322 ZJH 首次点击，开始绘制
            m_bDrawing = true;
            m_drawingPolygon.clear();
            m_drawingPolygon.append(ptScene);

            // 20260322 ZJH 创建临时图形项
            Annotation tempAnno(AnnotationType::Polygon);
            tempAnno.nLabelId = m_nCurrentLabelId;
            m_pDrawingItem = new AnnotationGraphicsItem(tempAnno.strUuid, AnnotationType::Polygon);
            m_pDrawingItem->setAnnotationPolygon(m_drawingPolygon);
            m_pDrawingItem->setLabelColor(m_currentLabelColor);
            m_pDrawingItem->setLabelName(m_strCurrentLabelName);
            m_pDrawingItem->setLabelId(m_nCurrentLabelId);
            m_pDrawingItem->setFlag(QGraphicsItem::ItemIsSelectable, false);
            m_pDrawingItem->setFlag(QGraphicsItem::ItemIsMovable, false);
            m_pScene->addItem(m_pDrawingItem);
        } else {
            // 20260322 ZJH 后续点击：添加顶点
            m_drawingPolygon.append(ptScene);
            m_pDrawingItem->setAnnotationPolygon(m_drawingPolygon);
        }
        return true;
    }
    case AnnotationTool::Brush: {
        // 20260322 ZJH 画笔工具：在点击位置创建圆形标注（简化为矩形）
        finishBrushAnnotation(ptScene);
        return true;
    }
    }

    return false;
}

// 20260322 ZJH 处理鼠标移动事件
bool AnnotationController::handleMouseMove(const QPointF& ptScene, Qt::MouseButtons btns)
{
    Q_UNUSED(btns);  // 20260322 ZJH 按钮状态暂未使用

    // 20260322 ZJH 矩形工具绘制中：更新矩形大小
    if (m_eTool == AnnotationTool::Rect && m_bDrawing && m_pDrawingItem) {
        // 20260322 ZJH 计算新矩形（标准化，确保宽高为正）
        QRectF rect = QRectF(m_ptDrawStart, ptScene).normalized();
        m_pDrawingItem->setAnnotationRect(rect);
        return true;
    }

    // 20260322 ZJH 多边形工具绘制中：橡皮筋效果（跟随鼠标）
    if (m_eTool == AnnotationTool::Polygon && m_bDrawing && m_pDrawingItem) {
        // 20260322 ZJH 临时添加当前鼠标位置作为预览点
        QPolygonF previewPolygon = m_drawingPolygon;
        previewPolygon.append(ptScene);
        m_pDrawingItem->setAnnotationPolygon(previewPolygon);
        return true;
    }

    return false;
}

// 20260322 ZJH 处理鼠标释放事件
bool AnnotationController::handleMouseRelease(const QPointF& ptScene, Qt::MouseButton btn)
{
    Q_UNUSED(ptScene);  // 20260322 ZJH 释放位置在移动完成时已知

    // 20260322 ZJH 只处理左键
    if (btn != Qt::LeftButton) {
        return false;
    }

    // 20260322 ZJH 选择工具：完成移动，记录移动命令
    if (m_eTool == AnnotationTool::Select && m_bMoving) {
        m_bMoving = false;

        // 20260322 ZJH 检查标注是否实际移动了
        if (m_mapItems.contains(m_strMovingUuid)) {
            AnnotationGraphicsItem* pItem = m_mapItems[m_strMovingUuid];
            QRectF newRect = pItem->annotationRect();
            QPolygonF newPolygon = pItem->annotationPolygon();

            // 20260322 ZJH 如果位置变化了，创建移动命令
            if (newRect != m_moveStartRect) {
                // 20260324 ZJH push 会立即调用 redo()，由命令的 redo 执行 moveAnnotationDirect 并同步数据
                auto* pCmd = new MoveAnnotationCommand(
                    this, m_strMovingUuid,
                    m_moveStartRect, newRect,
                    m_moveStartPolygon, newPolygon);
                m_pUndoStack->push(pCmd);
                // 20260324 ZJH annotationChanged 已由 moveAnnotationDirect() 在 redo 中发射
            }
        }
        return true;
    }

    // 20260322 ZJH 矩形工具：完成绘制
    if (m_eTool == AnnotationTool::Rect && m_bDrawing) {
        finishRectDrawing();
        return true;
    }

    return false;
}

// 20260322 ZJH 处理鼠标双击事件
bool AnnotationController::handleMouseDoubleClick(const QPointF& ptScene, Qt::MouseButton btn)
{
    Q_UNUSED(ptScene);
    Q_UNUSED(btn);

    // 20260322 ZJH 多边形工具：双击闭合多边形
    if (m_eTool == AnnotationTool::Polygon && m_bDrawing) {
        finishPolygonDrawing();
        return true;
    }

    return false;
}

// ===== 供 UndoCommand 直接操作的方法 =====

// 20260322 ZJH 直接添加标注（不创建 UndoCommand）
void AnnotationController::addAnnotationDirect(const Annotation& annotation)
{
    // 20260322 ZJH 创建图形项并添加到场景
    createGraphicsItem(annotation);

    // 20260322 ZJH 同步到 ImageEntry
    if (m_pEntry) {
        m_pEntry->vecAnnotations.append(annotation);
    }

    // 20260322 ZJH 发射信号
    emit annotationChanged();
    emit annotationCountChanged(m_mapItems.size());
}

// 20260322 ZJH 直接删除标注（不创建 UndoCommand）
void AnnotationController::removeAnnotationDirect(const QString& strUuid)
{
    // 20260322 ZJH 从映射表中查找图形项
    if (m_mapItems.contains(strUuid)) {
        AnnotationGraphicsItem* pItem = m_mapItems[strUuid];
        m_pScene->removeItem(pItem);   // 20260322 ZJH 从场景移除
        delete pItem;                   // 20260322 ZJH 释放内存
        m_mapItems.remove(strUuid);     // 20260322 ZJH 从映射表移除
    }

    // 20260322 ZJH 从 ImageEntry 中移除标注
    if (m_pEntry) {
        for (int i = 0; i < m_pEntry->vecAnnotations.size(); ++i) {
            if (m_pEntry->vecAnnotations[i].strUuid == strUuid) {
                m_pEntry->vecAnnotations.removeAt(i);
                break;
            }
        }
    }

    // 20260322 ZJH 发射信号
    emit annotationChanged();
    emit annotationCountChanged(m_mapItems.size());
}

// 20260322 ZJH 直接移动标注（不创建 UndoCommand）
void AnnotationController::moveAnnotationDirect(const QString& strUuid,
                                                  const QRectF& rect,
                                                  const QPolygonF& polygon)
{
    // 20260322 ZJH 查找图形项
    if (m_mapItems.contains(strUuid)) {
        AnnotationGraphicsItem* pItem = m_mapItems[strUuid];
        if (pItem->annotationType() == AnnotationType::Rect) {
            pItem->setAnnotationRect(rect);  // 20260322 ZJH 更新矩形位置
        } else if (pItem->annotationType() == AnnotationType::Polygon) {
            pItem->setAnnotationPolygon(polygon);  // 20260322 ZJH 更新多边形位置
        }
    }

    // 20260322 ZJH 同步到 ImageEntry
    syncToEntry();
    emit annotationChanged();
}

// 20260322 ZJH 获取撤销堆栈
QUndoStack* AnnotationController::undoStack() const
{
    return m_pUndoStack;
}

// ===== 私有方法 =====

// 20260322 ZJH 创建标注图形项并添加到场景
AnnotationGraphicsItem* AnnotationController::createGraphicsItem(const Annotation& annotation)
{
    // 20260322 ZJH 创建图形项
    auto* pItem = new AnnotationGraphicsItem(annotation.strUuid, annotation.eType);

    // 20260322 ZJH 根据标注类型设置几何数据
    if (annotation.eType == AnnotationType::Rect) {
        pItem->setAnnotationRect(annotation.rectBounds);
    } else if (annotation.eType == AnnotationType::RotatedRect) {
        // 20260402 ZJH 旋转矩形: 设置轴对齐包围框 + 旋转角度
        // QGraphicsItem::setRotation 围绕 transformOriginPoint 旋转
        pItem->setAnnotationRect(annotation.rectBounds);
        QPointF ptCenter = annotation.rectBounds.center();  // 20260402 ZJH 旋转中心 = 矩形中心
        pItem->setTransformOriginPoint(ptCenter - annotation.rectBounds.topLeft());
        pItem->setRotation(static_cast<double>(annotation.fAngle));  // 20260402 ZJH 应用旋转角度
    } else if (annotation.eType == AnnotationType::Polygon) {
        pItem->setAnnotationPolygon(annotation.polygon);
    }

    // 20260322 ZJH 查找标签颜色和名称
    QColor labelColor = m_currentLabelColor;
    QString strLabelName = m_strCurrentLabelName;
    if (annotation.nLabelId >= 0) {
        findLabelInfo(annotation.nLabelId, labelColor, strLabelName);
    }

    pItem->setLabelColor(labelColor);
    pItem->setLabelName(strLabelName);
    pItem->setLabelId(annotation.nLabelId);

    // 20260322 ZJH 添加到场景和映射表
    m_pScene->addItem(pItem);
    m_mapItems[annotation.strUuid] = pItem;

    return pItem;
}

// 20260322 ZJH 将场景中的标注数据同步到 ImageEntry
void AnnotationController::syncToEntry()
{
    if (!m_pEntry) {
        return;  // 20260322 ZJH 无绑定图像
    }

    // 20260322 ZJH 遍历映射表，更新每个标注的几何数据
    for (auto it = m_mapItems.begin(); it != m_mapItems.end(); ++it) {
        AnnotationGraphicsItem* pItem = it.value();
        QString strUuid = it.key();

        // 20260322 ZJH 在 ImageEntry 中查找对应标注
        for (Annotation& anno : m_pEntry->vecAnnotations) {
            if (anno.strUuid == strUuid) {
                // 20260322 ZJH 同步几何数据
                if (anno.eType == AnnotationType::Rect) {
                    anno.rectBounds = pItem->annotationRect();
                } else if (anno.eType == AnnotationType::Polygon) {
                    anno.polygon = pItem->annotationPolygon();
                    anno.rectBounds = anno.polygon.boundingRect();
                }
                break;
            }
        }
    }
}

// 20260322 ZJH 查找标签信息
void AnnotationController::findLabelInfo(int nLabelId, QColor& color, QString& strName) const
{
    if (!m_pDataset) {
        return;  // 20260322 ZJH 无数据集
    }

    // 20260322 ZJH 在数据集标签列表中查找
    for (const LabelInfo& label : m_pDataset->labels()) {
        if (label.nId == nLabelId) {
            color = label.color;
            strName = label.strName;
            return;
        }
    }
}

// 20260322 ZJH 完成矩形绘制
void AnnotationController::finishRectDrawing()
{
    m_bDrawing = false;  // 20260322 ZJH 结束绘制状态

    if (!m_pDrawingItem) {
        return;  // 20260322 ZJH 无绘制项
    }

    // 20260322 ZJH 获取最终矩形
    QRectF rect = m_pDrawingItem->annotationRect();

    // 20260322 ZJH 检查矩形尺寸是否有效（最小 3x3 像素）
    if (rect.width() < 3.0 || rect.height() < 3.0) {
        // 20260322 ZJH 太小，丢弃
        m_pScene->removeItem(m_pDrawingItem);
        delete m_pDrawingItem;
        m_pDrawingItem = nullptr;
        return;
    }

    // 20260322 ZJH 从场景移除临时图形项（将由 addAnnotationDirect 创建正式项）
    m_pScene->removeItem(m_pDrawingItem);
    delete m_pDrawingItem;
    m_pDrawingItem = nullptr;

    // 20260322 ZJH 创建正式标注数据
    Annotation anno(AnnotationType::Rect);
    anno.nLabelId = m_nCurrentLabelId;
    anno.rectBounds = rect;

    // 20260324 ZJH 推入撤销堆栈（push 会立即调用 redo()，由命令的 redo 执行 addAnnotationDirect）
    auto* pCmd = new AddAnnotationCommand(this, anno);
    m_pUndoStack->push(pCmd);
}

// 20260322 ZJH 完成多边形绘制
void AnnotationController::finishPolygonDrawing()
{
    m_bDrawing = false;  // 20260322 ZJH 结束绘制状态

    if (!m_pDrawingItem) {
        return;
    }

    // 20260322 ZJH 检查多边形顶点数是否足够（至少 3 个点）
    if (m_drawingPolygon.size() < 3) {
        m_pScene->removeItem(m_pDrawingItem);
        delete m_pDrawingItem;
        m_pDrawingItem = nullptr;
        m_drawingPolygon.clear();
        return;
    }

    // 20260322 ZJH 从场景移除临时项
    m_pScene->removeItem(m_pDrawingItem);
    delete m_pDrawingItem;
    m_pDrawingItem = nullptr;

    // 20260322 ZJH 创建正式标注数据
    Annotation anno(AnnotationType::Polygon);
    anno.nLabelId = m_nCurrentLabelId;
    anno.polygon = m_drawingPolygon;
    anno.rectBounds = m_drawingPolygon.boundingRect();

    // 20260322 ZJH 清空绘制中的多边形
    m_drawingPolygon.clear();

    // 20260324 ZJH 推入撤销堆栈（push 会立即调用 redo()，由命令的 redo 执行 addAnnotationDirect）
    auto* pCmd = new AddAnnotationCommand(this, anno);
    m_pUndoStack->push(pCmd);
}

// 20260322 ZJH 完成画笔标注（简化版：创建圆形/矩形标注）
void AnnotationController::finishBrushAnnotation(const QPointF& ptCenter)
{
    // 20260322 ZJH 以画笔中心和半径创建一个矩形标注
    QRectF rect(ptCenter.x() - m_nBrushRadius,
                ptCenter.y() - m_nBrushRadius,
                m_nBrushRadius * 2,
                m_nBrushRadius * 2);

    // 20260322 ZJH 创建标注数据
    Annotation anno(AnnotationType::Rect);
    anno.nLabelId = m_nCurrentLabelId;
    anno.rectBounds = rect;

    // 20260324 ZJH 推入撤销堆栈（push 会立即调用 redo()，由命令的 redo 执行 addAnnotationDirect）
    auto* pCmd = new AddAnnotationCommand(this, anno);
    m_pUndoStack->push(pCmd);
}

// ===== 复制/粘贴标注 (Ctrl+C / Ctrl+V) =====

// 20260330 ZJH 复制当前选中的标注到内部剪贴板
bool AnnotationController::copySelectedAnnotation()
{
    // 20260330 ZJH 查找场景中的选中项
    QList<QGraphicsItem*> vecSelected = m_pScene->selectedItems();
    if (vecSelected.isEmpty()) {
        m_bClipboardValid = false;  // 20260330 ZJH 无选中项，清空剪贴板
        return false;
    }

    // 20260330 ZJH 取第一个选中的标注图形项
    AnnotationGraphicsItem* pSelected = dynamic_cast<AnnotationGraphicsItem*>(vecSelected.first());
    if (!pSelected) {
        return false;  // 20260330 ZJH 选中项不是标注图形项
    }

    // 20260330 ZJH 通过 UUID 在 ImageEntry 中查找对应标注数据
    QString strUuid = pSelected->uuid();
    if (!m_pEntry) {
        return false;  // 20260330 ZJH 无绑定图像
    }

    for (const Annotation& anno : m_pEntry->vecAnnotations) {
        if (anno.strUuid == strUuid) {
            // 20260330 ZJH 深拷贝标注数据到剪贴板
            m_clipboardAnnotation = anno;
            m_bClipboardValid = true;  // 20260330 ZJH 标记剪贴板有效
            return true;
        }
    }

    return false;  // 20260330 ZJH 未找到对应标注
}

// 20260330 ZJH 粘贴剪贴板中的标注到当前图像
bool AnnotationController::pasteAnnotation(const QPointF& ptOffset)
{
    if (!m_bClipboardValid) {
        return false;  // 20260330 ZJH 剪贴板为空
    }

    if (!m_pEntry) {
        return false;  // 20260330 ZJH 无绑定图像
    }

    // 20260330 ZJH 创建新标注（基于剪贴板数据，生成新 UUID）
    Annotation newAnno(m_clipboardAnnotation.eType);
    newAnno.nLabelId = m_clipboardAnnotation.nLabelId;        // 20260330 ZJH 保持原标签
    newAnno.strText = m_clipboardAnnotation.strText;          // 20260330 ZJH 保持原文本
    newAnno.eSeverity = m_clipboardAnnotation.eSeverity;      // 20260330 ZJH 保持原严重度

    // 20260330 ZJH 根据标注类型，对坐标施加偏移
    if (m_clipboardAnnotation.eType == AnnotationType::Rect ||
        m_clipboardAnnotation.eType == AnnotationType::Mask) {
        // 20260330 ZJH 矩形/掩码: 平移外接矩形
        newAnno.rectBounds = m_clipboardAnnotation.rectBounds.translated(ptOffset.x(), ptOffset.y());
    } else if (m_clipboardAnnotation.eType == AnnotationType::Polygon) {
        // 20260330 ZJH 多边形: 平移所有顶点
        QPolygonF translatedPoly = m_clipboardAnnotation.polygon.translated(ptOffset.x(), ptOffset.y());
        newAnno.polygon = translatedPoly;
        newAnno.rectBounds = translatedPoly.boundingRect();
    } else if (m_clipboardAnnotation.eType == AnnotationType::TextArea) {
        // 20260330 ZJH 文字区域: 平移外接矩形
        newAnno.rectBounds = m_clipboardAnnotation.rectBounds.translated(ptOffset.x(), ptOffset.y());
    }

    // 20260330 ZJH 通过撤销堆栈添加标注（支持 Ctrl+Z 撤销粘贴）
    auto* pCmd = new AddAnnotationCommand(this, newAnno);
    m_pUndoStack->push(pCmd);

    return true;
}

// 20260330 ZJH 检查剪贴板是否有内容
bool AnnotationController::hasClipboardAnnotation() const
{
    return m_bClipboardValid;  // 20260330 ZJH 返回剪贴板有效标志
}
