// 20260322 ZJH Annotation — 标注数据结构
// 表示一张图像上的单个标注对象（矩形框/多边形/掩码/文字区域）
// 每个标注有唯一 UUID，关联一个标签 ID
#pragma once

#include <QString>
#include <QRectF>
#include <QPolygonF>
#include <QUuid>
#include <cstdint>

// 20260322 ZJH 标注类型枚举
// 定义支持的四种标注几何形状
enum class AnnotationType : uint8_t {
    Rect        = 0,  // 20260322 ZJH 矩形框标注（目标检测 - 轴对齐矩形）
    Polygon     = 1,  // 20260322 ZJH 多边形标注（实例分割/精确轮廓用）
    Mask        = 2,  // 20260322 ZJH 像素级掩码标注（语义分割用）
    TextArea    = 3,  // 20260322 ZJH 文字区域标注（OCR 用，含文本内容）
    RotatedRect = 4   // 20260402 ZJH 旋转矩形标注（对标 Halcon "对象检测 - 自由矩形"）
                      //   5 参数: 中心(cx,cy) + 宽高(w,h) + 角度(angle)
                      //   rectBounds 存储外接轴对齐矩形，fAngle 存储旋转角度
};

// 20260330 ZJH 缺陷严重度枚举（对标海康 VisionTrain 3 级缺陷分级体系）
// 用于工业质检场景中按严重程度分类缺陷，影响训练权重和推理报告
enum class DefectSeverity : uint8_t {
    None   = 0,  // 20260330 ZJH 未指定严重度（默认值，正常样本或未评级）
    Low    = 1,  // 20260330 ZJH 低严重度（轻微瑕疵，可能不影响产品功能）
    Medium = 2,  // 20260330 ZJH 中严重度（明显缺陷，需要审查）
    High   = 3,  // 20260330 ZJH 高严重度（严重缺陷，必须剔除）
    Ignore = 4   // 20260330 ZJH 忽略区域（标注但不参与训练/评估，对标海康"忽略"标记）
};

// 20260322 ZJH 单个标注数据结构
// 存储一个标注的完整信息：类型、几何数据、关联标签、唯一标识
struct Annotation {
    QString strUuid;            // 20260322 ZJH 标注唯一标识（UUID 字符串，不含花括号）
    AnnotationType eType = AnnotationType::Rect;  // 20260324 ZJH 标注类型（矩形/多边形/掩码/文字区域），默认矩形框提供防御性初始化
    int nLabelId = -1;          // 20260322 ZJH 关联的标签 ID（-1 表示未关联）
    QRectF rectBounds;          // 20260322 ZJH 边界矩形（Rect 类型直接使用；其他类型为外接矩形）
    QPolygonF polygon;          // 20260322 ZJH 多边形顶点序列（Polygon 类型使用）
    QString strText;            // 20260322 ZJH 文字内容（TextArea 类型使用，存储 OCR 识别文本）
    DefectSeverity eSeverity = DefectSeverity::None;  // 20260330 ZJH 缺陷严重度（None/Low/Medium/High/Ignore）
    // 20260402 ZJH 旋转角度（RotatedRect 类型使用，单位: 度，默认 0 = 轴对齐）
    // 正值 = 逆时针旋转，范围 [-180, 180]
    // 对于非 RotatedRect 类型此值始终为 0，向后兼容
    float fAngle = 0.0f;
    // 20260322 ZJH Mask 类型的像素数据较大，后续按需添加（如 QImage 或 RLE 编码）

    // 20260322 ZJH 默认构造函数：自动生成 UUID
    Annotation();

    // 20260322 ZJH 带类型参数的构造函数：自动生成 UUID 并设置标注类型
    // 参数: eAnnotType - 标注类型
    explicit Annotation(AnnotationType eAnnotType);
};
