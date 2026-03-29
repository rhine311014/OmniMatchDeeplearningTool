// 20260322 ZJH Annotation 构造函数实现
// 自动生成 UUID 确保每个标注全局唯一

#include "core/data/Annotation.h"

// 20260322 ZJH 默认构造函数
// 自动生成 UUID，类型默认为 Rect
Annotation::Annotation()
    : strUuid(QUuid::createUuid().toString(QUuid::WithoutBraces))  // 20260322 ZJH 生成不含花括号的 UUID 字符串
    , eType(AnnotationType::Rect)  // 20260322 ZJH 默认标注类型为矩形框
    , nLabelId(-1)                 // 20260322 ZJH 未关联标签
{
}

// 20260322 ZJH 带类型参数的构造函数
// 自动生成 UUID 并设置指定的标注类型
// 参数: eAnnotType - 要创建的标注类型
Annotation::Annotation(AnnotationType eAnnotType)
    : strUuid(QUuid::createUuid().toString(QUuid::WithoutBraces))  // 20260322 ZJH 生成唯一 UUID
    , eType(eAnnotType)  // 20260322 ZJH 使用传入的标注类型
    , nLabelId(-1)       // 20260322 ZJH 未关联标签
{
}
