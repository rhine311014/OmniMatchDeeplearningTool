// 20260322 ZJH DLTypes 辅助函数实现
// 提供任务类型/模型架构的字符串转换、任务-模型映射等全局辅助逻辑

#include "core/DLTypes.h"

namespace om {

// 20260322 ZJH 任务类型转换为中文显示名称
// 用于 UI 上的任务类型下拉框、标签等用户可见文本
QString taskTypeToString(TaskType eType)
{
    // 20260322 ZJH 根据枚举值返回对应的中文名称
    switch (eType) {
    case TaskType::AnomalyDetection:       return QStringLiteral("异常检测");        // 20260322 ZJH 异常检测任务
    case TaskType::Classification:          return QStringLiteral("图像分类");        // 20260322 ZJH 图像分类任务
    case TaskType::ObjectDetection:         return QStringLiteral("目标检测");        // 20260322 ZJH 目标检测任务
    case TaskType::SemanticSegmentation:    return QStringLiteral("语义分割");        // 20260322 ZJH 语义分割任务
    case TaskType::InstanceSegmentation:    return QStringLiteral("实例分割");        // 20260322 ZJH 实例分割任务
    case TaskType::DeepOCR:                 return QStringLiteral("深度OCR");         // 20260322 ZJH 深度 OCR 任务
    case TaskType::ZeroShotDefectDetection: return QStringLiteral("零样本缺陷检测");  // 20260322 ZJH 零样本缺陷检测
    case TaskType::ZeroShotObjectDetection: return QStringLiteral("零样本目标检测");  // 20260322 ZJH 零样本目标检测
    }
    return QStringLiteral("未知");  // 20260322 ZJH 防御性返回，防止未处理的枚举值
}

// 20260322 ZJH 模型架构转换为英文显示名称
// 用于 UI 上的模型选择下拉框、训练日志等
QString modelArchitectureToString(ModelArchitecture eArch)
{
    // 20260322 ZJH 根据枚举值返回对应的模型名称
    switch (eArch) {
    // 20260322 ZJH 分类模型
    case ModelArchitecture::ResNet18:          return QStringLiteral("ResNet-18");            // 20260322 ZJH 残差网络 18 层
    case ModelArchitecture::ResNet50:          return QStringLiteral("ResNet-50");            // 20260322 ZJH 残差网络 50 层
    case ModelArchitecture::EfficientNetB0:    return QStringLiteral("EfficientNet-B0");      // 20260322 ZJH 高效网络 B0
    case ModelArchitecture::MobileNetV4Small:  return QStringLiteral("MobileNetV4-Small");    // 20260322 ZJH 移动网络 V4 小
    case ModelArchitecture::MobileNetV4Medium: return QStringLiteral("MobileNetV4-Medium");   // 20260322 ZJH 移动网络 V4 中
    case ModelArchitecture::ViTTiny:           return QStringLiteral("ViT-Tiny");             // 20260322 ZJH 视觉 Transformer 微型
    case ModelArchitecture::ConvNeXtTiny:      return QStringLiteral("ConvNeXt-Tiny");        // 20260322 ZJH ConvNeXt 微型
    case ModelArchitecture::RepVGGA0:          return QStringLiteral("RepVGG-A0");            // 20260322 ZJH RepVGG A0

    // 20260322 ZJH 异常检测模型
    case ModelArchitecture::PaDiM:       return QStringLiteral("PaDiM");        // 20260322 ZJH 补丁分布建模
    case ModelArchitecture::PatchCore:   return QStringLiteral("PatchCore");    // 20260322 ZJH 核心补丁异常检测
    case ModelArchitecture::EfficientAD: return QStringLiteral("EfficientAD");  // 20260322 ZJH 高效异常检测
    case ModelArchitecture::FastFlow:    return QStringLiteral("FastFlow");     // 20260322 ZJH 快速流异常检测

    // 20260322 ZJH 目标检测模型
    case ModelArchitecture::YOLOv5Nano:  return QStringLiteral("YOLOv5-Nano");   // 20260322 ZJH YOLOv5 极小版
    case ModelArchitecture::YOLOv8Nano:  return QStringLiteral("YOLOv8-Nano");   // 20260322 ZJH YOLOv8 极小版
    case ModelArchitecture::YOLOv11Nano: return QStringLiteral("YOLOv11-Nano");  // 20260322 ZJH YOLOv11 极小版
    case ModelArchitecture::RTDETR:      return QStringLiteral("RT-DETR");       // 20260322 ZJH 实时 DETR 检测

    // 20260322 ZJH 语义分割模型
    case ModelArchitecture::UNet:          return QStringLiteral("U-Net");           // 20260322 ZJH U 型分割网络
    case ModelArchitecture::DeepLabV3Plus: return QStringLiteral("DeepLabV3+");      // 20260322 ZJH DeepLab V3 Plus
    case ModelArchitecture::PSPNet:        return QStringLiteral("PSPNet");          // 20260322 ZJH 金字塔池化分割
    case ModelArchitecture::SegFormer:     return QStringLiteral("SegFormer");       // 20260322 ZJH Transformer 分割

    // 20260322 ZJH 实例分割模型
    case ModelArchitecture::YOLOv8InstanceSeg: return QStringLiteral("YOLOv8-Seg");    // 20260322 ZJH YOLOv8 实例分割
    case ModelArchitecture::MaskRCNN:          return QStringLiteral("Mask R-CNN");     // 20260322 ZJH Mask RCNN

    // 20260322 ZJH OCR 模型
    case ModelArchitecture::PaddleOCRv4: return QStringLiteral("PaddleOCR v4");  // 20260322 ZJH PaddleOCR 第 4 版
    case ModelArchitecture::PPOCR:       return QStringLiteral("PP-OCR");        // 20260322 ZJH PP-OCR 轻量版

    // 20260322 ZJH 零样本缺陷检测模型
    case ModelArchitecture::WinCLIP:     return QStringLiteral("WinCLIP");       // 20260322 ZJH 窗口 CLIP 缺陷检测
    case ModelArchitecture::AnomalyCLIP: return QStringLiteral("AnomalyCLIP");   // 20260322 ZJH 异常 CLIP

    // 20260322 ZJH 零样本目标检测模型
    case ModelArchitecture::GroundingDINO: return QStringLiteral("Grounding DINO");  // 20260322 ZJH 接地 DINO
    case ModelArchitecture::YOLOWorld:     return QStringLiteral("YOLO-World");      // 20260322 ZJH YOLO World
    }
    return QStringLiteral("Unknown");  // 20260322 ZJH 防御性返回
}

// 20260322 ZJH 获取指定任务类型支持的全部模型架构列表
// 根据任务类型返回对应编号段的模型枚举值向量
QVector<ModelArchitecture> architecturesForTask(TaskType eType)
{
    // 20260322 ZJH 根据任务类型返回该任务可用的模型列表
    switch (eType) {
    case TaskType::Classification:
        // 20260322 ZJH 分类任务：8 种模型架构
        return {
            ModelArchitecture::ResNet18,
            ModelArchitecture::ResNet50,
            ModelArchitecture::EfficientNetB0,
            ModelArchitecture::MobileNetV4Small,
            ModelArchitecture::MobileNetV4Medium,
            ModelArchitecture::ViTTiny,
            ModelArchitecture::ConvNeXtTiny,
            ModelArchitecture::RepVGGA0
        };

    case TaskType::AnomalyDetection:
        // 20260322 ZJH 异常检测任务：4 种模型架构
        return {
            ModelArchitecture::PaDiM,
            ModelArchitecture::PatchCore,
            ModelArchitecture::EfficientAD,
            ModelArchitecture::FastFlow
        };

    case TaskType::ObjectDetection:
        // 20260322 ZJH 目标检测任务：4 种模型架构
        return {
            ModelArchitecture::YOLOv5Nano,
            ModelArchitecture::YOLOv8Nano,
            ModelArchitecture::YOLOv11Nano,
            ModelArchitecture::RTDETR
        };

    case TaskType::SemanticSegmentation:
        // 20260322 ZJH 语义分割任务：4 种模型架构
        return {
            ModelArchitecture::UNet,
            ModelArchitecture::DeepLabV3Plus,
            ModelArchitecture::PSPNet,
            ModelArchitecture::SegFormer
        };

    case TaskType::InstanceSegmentation:
        // 20260322 ZJH 实例分割任务
        // 20260326 ZJH 新增语义分割和异常检测模型，用户可在同一项目中切换，无需重新标注
        return {
            ModelArchitecture::YOLOv8InstanceSeg,
            ModelArchitecture::MaskRCNN,
            ModelArchitecture::EfficientAD,
            ModelArchitecture::UNet,
            ModelArchitecture::DeepLabV3Plus
        };

    case TaskType::DeepOCR:
        // 20260322 ZJH OCR 任务：2 种模型架构
        return {
            ModelArchitecture::PaddleOCRv4,
            ModelArchitecture::PPOCR
        };

    case TaskType::ZeroShotDefectDetection:
        // 20260322 ZJH 零样本缺陷检测任务：2 种模型架构
        return {
            ModelArchitecture::WinCLIP,
            ModelArchitecture::AnomalyCLIP
        };

    case TaskType::ZeroShotObjectDetection:
        // 20260322 ZJH 零样本目标检测任务：2 种模型架构
        return {
            ModelArchitecture::GroundingDINO,
            ModelArchitecture::YOLOWorld
        };
    }

    return {};  // 20260322 ZJH 防御性返回空列表
}

// 20260322 ZJH 获取指定任务类型的默认推荐模型架构
// 返回该任务模型列表中的第一个（最常用/最稳定的选择）
ModelArchitecture defaultArchitectureForTask(TaskType eType)
{
    // 20260322 ZJH 获取该任务的模型列表
    QVector<ModelArchitecture> vecArchitectures = architecturesForTask(eType);

    // 20260322 ZJH 如果列表非空，返回第一个（默认推荐）
    if (!vecArchitectures.isEmpty()) {
        return vecArchitectures.first();  // 20260322 ZJH 返回列表首个模型作为默认值
    }

    // 20260322 ZJH 防御性返回：列表为空时默认返回 ResNet-18
    return ModelArchitecture::ResNet18;
}

// 20260322 ZJH 判断任务类型是否为零样本任务
// 零样本任务无需训练数据标注，直接使用预训练模型推理
bool isZeroShotTask(TaskType eType)
{
    // 20260322 ZJH 仅零样本缺陷检测和零样本目标检测是零样本任务
    return (eType == TaskType::ZeroShotDefectDetection ||
            eType == TaskType::ZeroShotObjectDetection);
}

// 20260322 ZJH 判断任务类型是否需要训练
// 零样本任务不需要训练，其他任务都需要
bool taskRequiresTraining(TaskType eType)
{
    // 20260322 ZJH 非零样本任务都需要训练
    return !isZeroShotTask(eType);
}

}  // namespace om
