// 20260322 ZJH AnnotationCommands — 标注操作的撤销/重做命令
// 基于 QUndoCommand，支持添加/删除/移动标注的撤销重做
#pragma once

#include <QUndoCommand>   // 20260322 ZJH 撤销命令基类
#include <QString>        // 20260322 ZJH 字符串
#include <QRectF>         // 20260322 ZJH 矩形区域
#include <QPolygonF>      // 20260322 ZJH 多边形

#include "core/data/Annotation.h"  // 20260322 ZJH Annotation 结构体

// 20260322 ZJH 前向声明
class AnnotationController;

// 20260322 ZJH 添加标注命令
// undo: 从控制器中删除该标注
// redo: 重新添加该标注
class AddAnnotationCommand : public QUndoCommand
{
public:
    // 20260322 ZJH 构造函数
    // 参数: pController - 标注控制器指针（用于执行添加/删除操作）
    //       annotation - 要添加的标注数据
    //       pParent - 父命令（用于命令组合，通常为 nullptr）
    AddAnnotationCommand(AnnotationController* pController,
                         const Annotation& annotation,
                         QUndoCommand* pParent = nullptr);

    // 20260322 ZJH 撤销：删除已添加的标注
    void undo() override;

    // 20260322 ZJH 重做：重新添加标注
    void redo() override;

private:
    AnnotationController* m_pController;  // 20260322 ZJH 标注控制器指针（弱引用）
    Annotation m_annotation;               // 20260322 ZJH 标注数据副本
};

// 20260322 ZJH 删除标注命令
// undo: 重新添加已删除的标注
// redo: 从控制器中删除该标注
class DeleteAnnotationCommand : public QUndoCommand
{
public:
    // 20260322 ZJH 构造函数
    // 参数: pController - 标注控制器指针
    //       annotation - 要删除的标注数据（保存完整副本用于撤销恢复）
    //       pParent - 父命令
    DeleteAnnotationCommand(AnnotationController* pController,
                            const Annotation& annotation,
                            QUndoCommand* pParent = nullptr);

    // 20260322 ZJH 撤销：恢复已删除的标注
    void undo() override;

    // 20260322 ZJH 重做：再次删除标注
    void redo() override;

private:
    AnnotationController* m_pController;  // 20260322 ZJH 标注控制器指针（弱引用）
    Annotation m_annotation;               // 20260322 ZJH 标注数据副本（用于撤销恢复）
};

// 20260322 ZJH 移动标注命令
// 存储移动前后的位置，undo/redo 切换位置
class MoveAnnotationCommand : public QUndoCommand
{
public:
    // 20260322 ZJH 构造函数
    // 参数: pController - 标注控制器指针
    //       strUuid - 被移动标注的 UUID
    //       oldRect - 移动前的边界矩形
    //       newRect - 移动后的边界矩形
    //       oldPolygon - 移动前的多边形（多边形类型时使用）
    //       newPolygon - 移动后的多边形
    //       pParent - 父命令
    MoveAnnotationCommand(AnnotationController* pController,
                          const QString& strUuid,
                          const QRectF& oldRect,
                          const QRectF& newRect,
                          const QPolygonF& oldPolygon,
                          const QPolygonF& newPolygon,
                          QUndoCommand* pParent = nullptr);

    // 20260322 ZJH 撤销：恢复到移动前的位置
    void undo() override;

    // 20260322 ZJH 重做：移动到新位置
    void redo() override;

private:
    AnnotationController* m_pController;  // 20260322 ZJH 标注控制器指针
    QString m_strUuid;                     // 20260322 ZJH 标注 UUID
    QRectF m_oldRect;                      // 20260322 ZJH 移动前边界矩形
    QRectF m_newRect;                      // 20260322 ZJH 移动后边界矩形
    QPolygonF m_oldPolygon;                // 20260322 ZJH 移动前多边形
    QPolygonF m_newPolygon;                // 20260322 ZJH 移动后多边形
};
