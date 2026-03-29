// 20260322 ZJH ImageEntry — 数据集中单张图像的元数据
// 存储图像路径、尺寸、通道、文件大小、标签、拆分类型和标注列表
// 每张图像有唯一 UUID
#pragma once

#include <QString>
#include <QVector>
#include <cstdint>

#include "core/DLTypes.h"
#include "core/data/Annotation.h"

// 20260322 ZJH 单张图像条目数据结构
// 代表数据集中一张图像的完整元信息
struct ImageEntry {
    QString strUuid;                    // 20260322 ZJH 图像唯一标识（UUID 字符串）
    QString strFilePath;                // 20260322 ZJH 图像文件的绝对路径
    QString strRelativePath;            // 20260322 ZJH 相对于项目根目录的路径
    int nWidth = 0;                     // 20260322 ZJH 图像宽度（像素）
    int nHeight = 0;                    // 20260322 ZJH 图像高度（像素）
    int nChannels = 0;                  // 20260322 ZJH 图像通道数（1=灰度, 3=RGB, 4=RGBA）
    qint64 nFileSize = 0;              // 20260322 ZJH 文件大小（字节）

    int nLabelId = -1;                  // 20260322 ZJH 图像级标签 ID（分类/异常检测用，-1 表示未标注）
    om::SplitType eSplit = om::SplitType::Unassigned;  // 20260322 ZJH 数据集拆分类型（默认未分配）

    QVector<Annotation> vecAnnotations; // 20260322 ZJH 标注列表（目标检测/分割任务用）

    // 20260322 ZJH 默认构造函数：自动生成 UUID
    ImageEntry();

    // 20260322 ZJH 返回文件名（不含路径）
    // 例如 "E:/data/img001.png" 返回 "img001.png"
    QString fileName() const;

    // 20260322 ZJH 判断图像是否已标注
    // 分类/异常检测：nLabelId >= 0 即为已标注
    // 检测/分割：标注列表非空即为已标注
    bool isLabeled() const;
};
