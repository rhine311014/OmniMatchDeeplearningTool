// 20260322 ZJH LabelInfo — 标签信息数据结构
// 表示数据集中一个标注类别（如 "OK"、"缺陷A"、"螺丝" 等）
// 每个标签有唯一 ID、名称、显示颜色和可见性控制
#pragma once

#include <QString>
#include <QColor>
#include <QVector>

// 20260322 ZJH 标签信息结构体
// 存储单个标注类别的所有属性
struct LabelInfo {
    int nId = -1;            // 20260322 ZJH 标签唯一标识 ID（-1 表示未分配）
    QString strName;         // 20260322 ZJH 标签名称（如 "OK"、"NG"、"scratch" 等）
    QColor color;            // 20260322 ZJH 标签在 UI 上的显示颜色（标注框/掩码颜色）
    bool bVisible = true;    // 20260322 ZJH 该标签的标注是否在 UI 上可见（隐藏时不绘制）
};

// 20260322 ZJH 返回 20 个预设标签颜色列表
// 颜色选取自工业视觉常用配色方案，确保在暗色背景下可辨识
// 颜色依次分配给新建标签（循环使用）
QVector<QColor> defaultLabelColors();
