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
    case TaskType::ImageRetrieval:          return QStringLiteral("图像检索");        // 20260402 ZJH DL 图像检索
    case TaskType::UnsupervisedSegmentation:return QStringLiteral("无监督分割");      // 20260402 ZJH DL 无监督分割
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
    case ModelArchitecture::GCAD:        return QStringLiteral("GCAD");         // 20260402 ZJH 全局上下文异常检测

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
    case ModelArchitecture::MobileSegNet:  return QStringLiteral("MobileSegNet");    // 20260401 ZJH 轻量级工业分割

    // 20260322 ZJH 实例分割模型
    case ModelArchitecture::YOLOv8InstanceSeg: return QStringLiteral("YOLOv8-Seg");    // 20260322 ZJH YOLOv8 实例分割
    case ModelArchitecture::MaskRCNN:          return QStringLiteral("Mask R-CNN");     // 20260322 ZJH Mask RCNN

    // 20260322 ZJH OCR 模型
    case ModelArchitecture::PaddleOCRv4: return QStringLiteral("PaddleOCR v4");  // 20260322 ZJH PaddleOCR 第 4 版
    case ModelArchitecture::PPOCR:       return QStringLiteral("PP-OCR");        // 20260322 ZJH PP-OCR 轻量版
    case ModelArchitecture::DBNet:       return QStringLiteral("DBNet");         // 20260402 ZJH 可微分二值化文本检测
    case ModelArchitecture::DBNetCRNN:   return QStringLiteral("DBNet+CRNN");   // 20260402 ZJH 端到端 OCR

    // 20260322 ZJH 零样本缺陷检测模型
    case ModelArchitecture::WinCLIP:     return QStringLiteral("WinCLIP");       // 20260322 ZJH 窗口 CLIP 缺陷检测
    case ModelArchitecture::AnomalyCLIP: return QStringLiteral("AnomalyCLIP");   // 20260322 ZJH 异常 CLIP

    // 20260322 ZJH 零样本目标检测模型
    case ModelArchitecture::GroundingDINO: return QStringLiteral("Grounding DINO");  // 20260322 ZJH 接地 DINO
    case ModelArchitecture::YOLOWorld:     return QStringLiteral("YOLO-World");      // 20260322 ZJH YOLO World
    case ModelArchitecture::EdgeUNet:      return QStringLiteral("EdgeUNet");        // 20260402 ZJH DL 边缘提取
    case ModelArchitecture::Dinomaly:      return QStringLiteral("Dinomaly");        // 20260402 ZJH CVPR 2025 SOTA AD
    case ModelArchitecture::SAM2UNet:      return QStringLiteral("SAM2-UNet");      // 20260402 ZJH SAM2 编码器分割
    case ModelArchitecture::ImageRetrievalNet: return QStringLiteral("ImageRetrieval"); // 20260402 ZJH 图像检索网络
    case ModelArchitecture::MobileSAM:     return QStringLiteral("MobileSAM");      // 20260402 ZJH 无监督分割
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
            ModelArchitecture::FastFlow,
            ModelArchitecture::GCAD,          // 20260402 ZJH 全局上下文异常检测（布局+纹理）
            ModelArchitecture::Dinomaly       // 20260402 ZJH CVPR 2025 SOTA (99.6% AUROC)
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
        // 20260322 ZJH 语义分割任务：5 种模型架构
        // 20260401 ZJH 新增 MobileSegNet 轻量级分割（推荐小数据集+快速推理场景）
        return {
            ModelArchitecture::UNet,
            ModelArchitecture::DeepLabV3Plus,
            ModelArchitecture::MobileSegNet,
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
            ModelArchitecture::DeepLabV3Plus,
            ModelArchitecture::MobileSegNet
        };

    case TaskType::DeepOCR:
        // 20260322 ZJH OCR 任务：2 种模型架构
        return {
            ModelArchitecture::PaddleOCRv4,
            ModelArchitecture::PPOCR,
            ModelArchitecture::DBNet,         // 20260402 ZJH 文本检测（仅检测框）
            ModelArchitecture::DBNetCRNN      // 20260402 ZJH 端到端 OCR（检测+识别）
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

    case TaskType::ImageRetrieval:
        // 20260402 ZJH DL 图像检索（对标海康 VisionMaster）
        return {
            ModelArchitecture::ImageRetrievalNet
        };

    case TaskType::UnsupervisedSegmentation:
        // 20260402 ZJH DL 无监督分割（对标海康 VisionMaster）
        return {
            ModelArchitecture::MobileSAM
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

// 20260330 ZJH 根据任务类型和模型能力等级返回推荐的模型架构列表
// 将海康 VisionTrain 的"模型能力"概念映射到 OmniMatch 的架构枚举
// 轻量化→移动端/轻量模型, 普通→基础模型, 高精度→深层/Transformer模型
QVector<ModelArchitecture> architecturesForCapability(TaskType eType, ModelCapability eCapability)
{
    // 20260330 ZJH 根据任务类型和能力等级组合查找对应的模型列表
    switch (eType) {

    case TaskType::Classification:
        // 20260330 ZJH 分类任务的能力-架构映射
        switch (eCapability) {
        case ModelCapability::Lightweight:
            // 20260330 ZJH 轻量化：MobileNet 系列和 RepVGG（推理速度快，适合嵌入式）
            return { ModelArchitecture::MobileNetV4Small, ModelArchitecture::MobileNetV4Medium, ModelArchitecture::RepVGGA0 };
        case ModelCapability::Normal:
            // 20260330 ZJH 普通：ResNet-18 和 EfficientNet-B0（精度与速度均衡）
            return { ModelArchitecture::ResNet18, ModelArchitecture::EfficientNetB0 };
        case ModelCapability::HighAccuracy:
            // 20260330 ZJH 高精度：ResNet-50、ConvNeXt、ViT（深层网络，复杂场景）
            return { ModelArchitecture::ResNet50, ModelArchitecture::ConvNeXtTiny, ModelArchitecture::ViTTiny };
        }
        break;

    case TaskType::AnomalyDetection:
        // 20260330 ZJH 异常检测任务的能力-架构映射
        switch (eCapability) {
        case ModelCapability::Lightweight:
            // 20260330 ZJH 轻量化：PaDiM（无需训练迭代，特征提取快）
            return { ModelArchitecture::PaDiM };
        case ModelCapability::Normal:
            // 20260330 ZJH 普通：PatchCore（核心补丁记忆库，精度较高）
            return { ModelArchitecture::PaDiM, ModelArchitecture::PatchCore };
        case ModelCapability::HighAccuracy:
            // 20260330 ZJH 高精度：EfficientAD 和 FastFlow（深度模型，检出小缺陷）
            return { ModelArchitecture::EfficientAD, ModelArchitecture::FastFlow, ModelArchitecture::GCAD };
        }
        break;

    case TaskType::ObjectDetection:
        // 20260330 ZJH 目标检测任务的能力-架构映射
        switch (eCapability) {
        case ModelCapability::Lightweight:
            // 20260330 ZJH 轻量化：YOLOv5-Nano（最小最快）
            return { ModelArchitecture::YOLOv5Nano };
        case ModelCapability::Normal:
            // 20260330 ZJH 普通：YOLOv8-Nano（新一代均衡检测）
            return { ModelArchitecture::YOLOv8Nano, ModelArchitecture::YOLOv5Nano };
        case ModelCapability::HighAccuracy:
            // 20260330 ZJH 高精度：YOLOv11 和 RT-DETR（最新架构，Transformer 加持）
            return { ModelArchitecture::YOLOv11Nano, ModelArchitecture::RTDETR };
        }
        break;

    case TaskType::SemanticSegmentation:
        // 20260330 ZJH 语义分割任务的能力-架构映射
        switch (eCapability) {
        case ModelCapability::Lightweight:
            // 20260330 ZJH 轻量化：U-Net（结构简单，推理快）
            return { ModelArchitecture::UNet };
        case ModelCapability::Normal:
            // 20260330 ZJH 普通：U-Net 和 PSPNet（经典分割网络）
            return { ModelArchitecture::UNet, ModelArchitecture::PSPNet };
        case ModelCapability::HighAccuracy:
            // 20260330 ZJH 高精度：DeepLabV3+ 和 SegFormer（空洞卷积 + Transformer）
            return { ModelArchitecture::DeepLabV3Plus, ModelArchitecture::SegFormer };
        }
        break;

    case TaskType::InstanceSegmentation:
        // 20260330 ZJH 实例分割任务的能力-架构映射
        switch (eCapability) {
        case ModelCapability::Lightweight:
            // 20260330 ZJH 轻量化：YOLOv8 实例分割（单阶段，速度快）
            return { ModelArchitecture::YOLOv8InstanceSeg };
        case ModelCapability::Normal:
            // 20260330 ZJH 普通：YOLOv8-Seg + U-Net（覆盖不同精度需求）
            return { ModelArchitecture::YOLOv8InstanceSeg, ModelArchitecture::UNet };
        case ModelCapability::HighAccuracy:
            // 20260330 ZJH 高精度：Mask R-CNN + DeepLabV3+（两阶段+空洞卷积）
            return { ModelArchitecture::MaskRCNN, ModelArchitecture::DeepLabV3Plus };
        }
        break;

    case TaskType::DeepOCR:
        // 20260330 ZJH OCR 任务：模型较少，能力等级映射简化
        switch (eCapability) {
        case ModelCapability::Lightweight:
            return { ModelArchitecture::PPOCR };       // 20260330 ZJH PP-OCR 轻量版
        case ModelCapability::Normal:
            return { ModelArchitecture::PPOCR, ModelArchitecture::PaddleOCRv4, ModelArchitecture::DBNetCRNN };  // 20260402 ZJH 新增 DBNet+CRNN
        case ModelCapability::HighAccuracy:
            return { ModelArchitecture::PaddleOCRv4, ModelArchitecture::DBNetCRNN }; // 20260402 ZJH PaddleOCR v4 + DBNet+CRNN
        }
        break;

    case TaskType::ZeroShotDefectDetection:
        // 20260330 ZJH 零样本缺陷检测：不区分能力等级，返回全部
        return { ModelArchitecture::WinCLIP, ModelArchitecture::AnomalyCLIP };

    case TaskType::ZeroShotObjectDetection:
        // 20260330 ZJH 零样本目标检测：不区分能力等级，返回全部
        return { ModelArchitecture::GroundingDINO, ModelArchitecture::YOLOWorld };
    }

    return {};  // 20260330 ZJH 防御性返回空列表
}

// 20260330 ZJH 获取指定任务类型适用的分辨率预设列表
// 不同任务的合理分辨率范围不同，避免用户选择不合适的值
QVector<InputResolutionPreset> resolutionPresetsForTask(TaskType eType)
{
    switch (eType) {
    case TaskType::Classification:
        // 20260330 ZJH 分类任务：224 为经典值，最高到 512 即可
        return {
            InputResolutionPreset::Res224,
            InputResolutionPreset::Res320,
            InputResolutionPreset::Res512,
            InputResolutionPreset::Custom
        };

    case TaskType::AnomalyDetection:
        // 20260330 ZJH 异常检测：256 起步（补丁特征提取需要足够分辨率）
        return {
            InputResolutionPreset::Res224,
            InputResolutionPreset::Res320,
            InputResolutionPreset::Res512,
            InputResolutionPreset::Custom
        };

    case TaskType::ObjectDetection:
        // 20260330 ZJH 目标检测：320~1024，小目标需要大分辨率
        return {
            InputResolutionPreset::Res320,
            InputResolutionPreset::Res416,
            InputResolutionPreset::Res512,
            InputResolutionPreset::Res640,
            InputResolutionPreset::Res800,
            InputResolutionPreset::Res1024,
            InputResolutionPreset::Custom
        };

    case TaskType::SemanticSegmentation:
    case TaskType::InstanceSegmentation:
        // 20260331 ZJH 分割任务：256~1024，像素级标注需要较高分辨率
        // 256 为 DeepLabV3+ 在 16GB 显存下的推荐值
        return {
            InputResolutionPreset::Res256,
            InputResolutionPreset::Res320,
            InputResolutionPreset::Res512,
            InputResolutionPreset::Res640,
            InputResolutionPreset::Res800,
            InputResolutionPreset::Res1024,
            InputResolutionPreset::Custom
        };

    case TaskType::DeepOCR:
        // 20260330 ZJH OCR 任务：320~640，文字识别不需要超大分辨率
        return {
            InputResolutionPreset::Res320,
            InputResolutionPreset::Res512,
            InputResolutionPreset::Res640,
            InputResolutionPreset::Custom
        };

    case TaskType::ZeroShotDefectDetection:
    case TaskType::ZeroShotObjectDetection:
        // 20260330 ZJH 零样本任务：预训练模型有固定输入，提供常用尺寸
        return {
            InputResolutionPreset::Res224,
            InputResolutionPreset::Res512,
            InputResolutionPreset::Custom
        };
    }

    return { InputResolutionPreset::Res512, InputResolutionPreset::Custom };  // 20260330 ZJH 防御性默认
}

// 20260330 ZJH 分辨率预设转换为实际像素值
int resolutionPresetToPixels(InputResolutionPreset ePreset)
{
    switch (ePreset) {
    case InputResolutionPreset::Res224:  return 224;   // 20260330 ZJH 224×224
    case InputResolutionPreset::Res256:  return 256;   // 20260331 ZJH 256×256
    case InputResolutionPreset::Res320:  return 320;   // 20260330 ZJH 320×320
    case InputResolutionPreset::Res416:  return 416;   // 20260330 ZJH 416×416
    case InputResolutionPreset::Res512:  return 512;   // 20260330 ZJH 512×512
    case InputResolutionPreset::Res640:  return 640;   // 20260330 ZJH 640×640
    case InputResolutionPreset::Res800:  return 800;   // 20260330 ZJH 800×800
    case InputResolutionPreset::Res1024: return 1024;  // 20260330 ZJH 1024×1024
    case InputResolutionPreset::Custom:  return 0;     // 20260330 ZJH 自定义，需用户输入
    }
    return 512;  // 20260330 ZJH 防御性默认 512
}

// 20260330 ZJH 分辨率预设转换为显示字符串
QString resolutionPresetToString(InputResolutionPreset ePreset)
{
    switch (ePreset) {
    case InputResolutionPreset::Res224:  return QStringLiteral("224\u00d7224");    // 20260330 ZJH ×
    case InputResolutionPreset::Res256:  return QStringLiteral("256\u00d7256");    // 20260331 ZJH
    case InputResolutionPreset::Res320:  return QStringLiteral("320\u00d7320");
    case InputResolutionPreset::Res416:  return QStringLiteral("416\u00d7416");
    case InputResolutionPreset::Res512:  return QStringLiteral("512\u00d7512");
    case InputResolutionPreset::Res640:  return QStringLiteral("640\u00d7640");
    case InputResolutionPreset::Res800:  return QStringLiteral("800\u00d7800");
    case InputResolutionPreset::Res1024: return QStringLiteral("1024\u00d71024");
    case InputResolutionPreset::Custom:  return QStringLiteral("自定义");
    }
    return QStringLiteral("512\u00d7512");  // 20260330 ZJH 防御性默认
}

// 20260330 ZJH 模型能力等级转换为中文显示名称
QString modelCapabilityToString(ModelCapability eCapability)
{
    switch (eCapability) {
    case ModelCapability::Lightweight:  return QStringLiteral("轻量化");   // 20260330 ZJH 推理速度优先
    case ModelCapability::Normal:       return QStringLiteral("普通");     // 20260330 ZJH 均衡模式
    case ModelCapability::HighAccuracy: return QStringLiteral("高精度");   // 20260330 ZJH 精度优先
    }
    return QStringLiteral("普通");  // 20260330 ZJH 防御性默认
}

// 20260330 ZJH 根据数据集大小智能推荐训练轮次
// 借鉴海康 VisionTrain 经验规则:
//   >250 张: 100~200 轮 (数据充足, 不需太多轮)
//   50~250 张: 200~500 轮 (数据量中等, 适当增加)
//   <50 张: 500~1000 轮 (数据稀少, 需要更多轮次充分学习)
int recommendEpochs(int nImageCount, TaskType eType)
{
    // 20260330 ZJH 异常检测任务通常需要较少轮次（特征提取类模型收敛快）
    if (eType == TaskType::AnomalyDetection) {
        if (nImageCount > 200) return 30;   // 20260330 ZJH 数据充足
        if (nImageCount > 50)  return 50;   // 20260330 ZJH 数据中等
        return 100;                          // 20260330 ZJH 数据稀少
    }

    // 20260330 ZJH OCR 任务需要较多轮次（字符特征复杂）
    if (eType == TaskType::DeepOCR) {
        if (nImageCount > 500) return 100;
        if (nImageCount > 100) return 200;
        return 500;
    }

    // 20260407 ZJH [优化] 对齐 Halcon/MVTec DL Tool 行业标准
    // 语义分割: Halcon 推荐 60-100 epoch，配合 patience=15 早停
    // 分类: MVTec DL Tool 默认 50-100 epoch
    // 旧值偏高（500-800），过多 epoch 导致过拟合且浪费训练时间
    if (eType == TaskType::SemanticSegmentation || eType == TaskType::InstanceSegmentation) {
        if (nImageCount > 200) return 60;    // 20260407 ZJH Halcon 默认分割 60 轮
        if (nImageCount > 50)  return 80;    // 20260407 ZJH 中等: 80 轮 + 早停
        if (nImageCount > 20)  return 100;   // 20260407 ZJH 小数据: 100 轮 + 早停
        return 150;                           // 20260407 ZJH 极小: 150 轮（早停兜底）
    }
    // 20260407 ZJH 分类/检测通用规则
    if (nImageCount > 500) return 60;        // 20260407 ZJH 大数据集: 60 轮
    if (nImageCount > 200) return 80;        // 20260407 ZJH 中大: 80 轮
    if (nImageCount > 100) return 100;       // 20260407 ZJH 中等: 100 轮
    if (nImageCount > 50)  return 120;       // 20260407 ZJH 小: 120 轮
    if (nImageCount > 20)  return 150;       // 20260407 ZJH 极小: 150 轮
    return 200;                               // 20260407 ZJH 微量: 200 轮
}

// 20260330 ZJH 根据数据集大小智能推荐学习率
// 数据集越小，学习率应越保守，避免过拟合
double recommendLearningRate(int nImageCount, TaskType eType)
{
    // 20260330 ZJH 异常检测任务通常使用较小学习率
    if (eType == TaskType::AnomalyDetection) {
        return 0.0001;  // 20260330 ZJH 异常检测统一 0.0001
    }

    // 20260407 ZJH [优化] 对齐 Halcon/Anomalib 推荐 LR
    // Halcon 分割默认 0.001 (Adam)，小数据集降至 0.0001
    // Anomalib EfficientAD 使用 0.0001
    if (eType == TaskType::SemanticSegmentation || eType == TaskType::InstanceSegmentation) {
        if (nImageCount > 100) return 0.001;  // 20260407 ZJH 分割标准 LR
        if (nImageCount > 30)  return 0.0005; // 20260407 ZJH 小数据集保守
        return 0.0003;                         // 20260407 ZJH 极小数据集
    }
    // 20260407 ZJH 分类/检测通用
    if (nImageCount > 500) return 0.001;
    if (nImageCount > 100) return 0.0005;
    return 0.0003;
}

}  // namespace om
