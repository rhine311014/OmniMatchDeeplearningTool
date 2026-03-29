// 20260322 ZJH AnnotationCommands 实现
// 标注操作的撤销/重做命令：添加/删除/移动

#include "ui/widgets/AnnotationCommands.h"      // 20260322 ZJH 命令类声明
#include "ui/widgets/AnnotationController.h"    // 20260322 ZJH 标注控制器（执行实际操作）

// ===== AddAnnotationCommand =====

// 20260322 ZJH 添加标注命令构造函数
AddAnnotationCommand::AddAnnotationCommand(AnnotationController* pController,
                                             const Annotation& annotation,
                                             QUndoCommand* pParent)
    : QUndoCommand(pParent)           // 20260322 ZJH 初始化基类
    , m_pController(pController)      // 20260322 ZJH 记录控制器指针
    , m_annotation(annotation)        // 20260322 ZJH 拷贝标注数据
{
    // 20260322 ZJH 设置命令文字描述（用于撤销菜单显示）
    setText(QStringLiteral("添加标注"));
}

// 20260322 ZJH 撤销：删除已添加的标注
void AddAnnotationCommand::undo()
{
    // 20260322 ZJH 通过控制器删除标注（不推入 UndoStack，直接操作）
    m_pController->removeAnnotationDirect(m_annotation.strUuid);
}

// 20260324 ZJH 重做：添加标注（push 时首次调用即执行，后续 redo 也正常执行）
void AddAnnotationCommand::redo()
{
    // 20260324 ZJH 通过控制器添加标注
    m_pController->addAnnotationDirect(m_annotation);
}

// ===== DeleteAnnotationCommand =====

// 20260322 ZJH 删除标注命令构造函数
DeleteAnnotationCommand::DeleteAnnotationCommand(AnnotationController* pController,
                                                   const Annotation& annotation,
                                                   QUndoCommand* pParent)
    : QUndoCommand(pParent)           // 20260322 ZJH 初始化基类
    , m_pController(pController)      // 20260322 ZJH 记录控制器指针
    , m_annotation(annotation)        // 20260322 ZJH 保存标注完整副本
{
    setText(QStringLiteral("删除标注"));
}

// 20260322 ZJH 撤销：恢复已删除的标注
void DeleteAnnotationCommand::undo()
{
    // 20260322 ZJH 通过控制器重新添加标注
    m_pController->addAnnotationDirect(m_annotation);
}

// 20260324 ZJH 重做：删除标注（push 时首次调用即执行，后续 redo 也正常执行）
void DeleteAnnotationCommand::redo()
{
    // 20260324 ZJH 通过控制器删除标注
    m_pController->removeAnnotationDirect(m_annotation.strUuid);
}

// ===== MoveAnnotationCommand =====

// 20260322 ZJH 移动标注命令构造函数
MoveAnnotationCommand::MoveAnnotationCommand(AnnotationController* pController,
                                               const QString& strUuid,
                                               const QRectF& oldRect,
                                               const QRectF& newRect,
                                               const QPolygonF& oldPolygon,
                                               const QPolygonF& newPolygon,
                                               QUndoCommand* pParent)
    : QUndoCommand(pParent)
    , m_pController(pController)
    , m_strUuid(strUuid)
    , m_oldRect(oldRect)
    , m_newRect(newRect)
    , m_oldPolygon(oldPolygon)
    , m_newPolygon(newPolygon)
{
    setText(QStringLiteral("移动标注"));
}

// 20260322 ZJH 撤销：恢复到移动前位置
void MoveAnnotationCommand::undo()
{
    // 20260322 ZJH 将标注移回原位
    m_pController->moveAnnotationDirect(m_strUuid, m_oldRect, m_oldPolygon);
}

// 20260324 ZJH 重做：移动到新位置（push 时首次调用即执行，后续 redo 也正常执行）
void MoveAnnotationCommand::redo()
{
    // 20260324 ZJH 将标注移到新位置
    m_pController->moveAnnotationDirect(m_strUuid, m_newRect, m_newPolygon);
}
