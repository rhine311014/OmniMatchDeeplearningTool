// 20260322 ZJH ImageEntry 辅助函数实现
// 提供文件名提取和标注状态判断

#include "core/data/ImageEntry.h"

#include <QFileInfo>
#include <QUuid>

// 20260322 ZJH 默认构造函数：自动生成 UUID
ImageEntry::ImageEntry()
    : strUuid(QUuid::createUuid().toString(QUuid::WithoutBraces))  // 20260322 ZJH 生成不含花括号的 UUID 字符串
{
}

// 20260322 ZJH 从绝对路径中提取文件名（含扩展名）
// 使用 QFileInfo 确保跨平台路径解析正确
QString ImageEntry::fileName() const
{
    // 20260322 ZJH QFileInfo::fileName() 返回路径最后一段（文件名 + 扩展名）
    return QFileInfo(strFilePath).fileName();
}

// 20260322 ZJH 判断图像是否已完成标注
// 两种标注模式：
//   1. 图像级标签（分类/异常检测）：nLabelId >= 0 表示已标注
//   2. 对象级标注（检测/分割）：标注列表非空表示已标注
bool ImageEntry::isLabeled() const
{
    // 20260322 ZJH 任一模式满足即视为已标注
    if (nLabelId >= 0) {
        return true;  // 20260322 ZJH 已分配图像级标签
    }
    if (!vecAnnotations.isEmpty()) {
        return true;  // 20260322 ZJH 存在对象级标注
    }
    return false;  // 20260322 ZJH 两种模式均未标注
}
