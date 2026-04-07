// 20260322 ZJH OmniMatch 全局深度学习类型定义
// 包含所有模块共用的枚举、常量和辅助函数声明
// 覆盖任务类型、后端、设备、精度、模型架构、优化器、调度器、项目状态等
#pragma once

#include <QString>
#include <QColor>
#include <QVector>
#include <cstdint>

namespace om {

// 20260322 ZJH 深度学习任务类型枚举（8 种）
// 定义 OmniMatch 工具支持的全部视觉任务类型
enum class TaskType : uint8_t {
    AnomalyDetection     = 0,  // 20260322 ZJH 异常检测（无监督/半监督，仅需正常样本训练）
    Classification       = 1,  // 20260322 ZJH 图像分类（多类别单标签分类）
    ObjectDetection      = 2,  // 20260322 ZJH 目标检测（Bounding Box 级别定位 + 分类）
    SemanticSegmentation = 3,  // 20260322 ZJH 语义分割（像素级分类，不区分实例）
    InstanceSegmentation = 4,  // 20260322 ZJH 实例分割（像素级分类 + 实例区分）
    DeepOCR              = 5,  // 20260322 ZJH 深度OCR（文本检测 + 文字识别）
    ZeroShotDefectDetection = 6,  // 20260322 ZJH 零样本缺陷检测（CLIP 文本引导，无需训练）
    ZeroShotObjectDetection = 7,  // 20260322 ZJH 零样本目标检测（开放词汇检测，文本描述驱动）
    ImageRetrieval          = 8,  // 20260402 ZJH DL 图像检索（特征嵌入+余弦相似度搜索，对标海康）
    UnsupervisedSegmentation= 9   // 20260402 ZJH DL 无监督分割（类 SAM 自动分区，无需标注）
};

// 20260322 ZJH 推理后端类型枚举（9 种）
// 定义 OmniMatch 支持的全部推理/训练后端
enum class BackendType : uint8_t {
    OnnxRuntime = 0,  // 20260322 ZJH ONNX Runtime 推理后端
    TensorRT    = 1,  // 20260322 ZJH TensorRT 高性能推理后端
    OpenCVDNN   = 2,  // 20260322 ZJH OpenCV DNN 推理后端
    HalconDL    = 3,  // 20260322 ZJH Halcon 深度学习后端
    OpenVINO    = 4,  // 20260322 ZJH Intel OpenVINO 推理后端
    NCNN        = 5,  // 20260322 ZJH NCNN 轻量级推理后端
    MNN         = 6,  // 20260322 ZJH MNN 移动端推理后端
    Libtorch    = 7,  // 20260322 ZJH Libtorch（PyTorch C++ 前端）
    NativeCpp   = 8   // 20260322 ZJH 纯 C++ 内置引擎（OmniMatch 自研）
};

// 20260322 ZJH 设备类型枚举
// 控制模型运行在 CPU 还是 CUDA GPU 上
enum class DeviceType : uint8_t {
    CPU  = 0,  // 20260322 ZJH CPU 推理/训练
    CUDA = 1   // 20260322 ZJH NVIDIA CUDA GPU 推理/训练
};

// 20260402 ZJH 归一化层类型枚举
// 控制模型中 Conv 后接的归一化方式
// BatchNorm 在 batch < 8 时统计量不稳定，GroupNorm 不依赖 batch 维度
enum class NormLayerType : uint8_t {
    BatchNorm  = 0,  // 20260402 ZJH 批归一化（默认，batch ≥ 8 时效果最佳）
    GroupNorm  = 1,  // 20260402 ZJH 组归一化（batch < 8 时替代 BN，channels 分组独立归一化）
    Auto       = 2   // 20260402 ZJH 自动选择（batch < 8 → GroupNorm，否则 → BatchNorm）
};

// 20260322 ZJH 计算精度类型枚举
// 控制推理时的数值精度（影响速度和准确性的权衡）
enum class PrecisionType : uint8_t {
    FP32 = 0,  // 20260322 ZJH 32 位浮点（最高精度，最慢）
    FP16 = 1,  // 20260322 ZJH 16 位浮点（半精度加速，精度损失极小）
    INT8 = 2   // 20260322 ZJH 8 位整型量化（最快，需校准数据）
};

// 20260322 ZJH 数据集拆分类型枚举
// 标记每张图像属于哪个数据集子集
enum class SplitType : uint8_t {
    Train      = 0,  // 20260322 ZJH 训练集
    Validation = 1,  // 20260322 ZJH 验证集
    Test       = 2,  // 20260322 ZJH 测试集
    Unassigned = 3   // 20260322 ZJH 尚未分配（新导入的图像默认状态）
};

// 20260322 ZJH 训练框架类型枚举
// 指定训练时使用的底层框架
enum class TrainingFramework : uint8_t {
    Libtorch  = 0,  // 20260322 ZJH Libtorch 框架训练
    HalconDL  = 1,  // 20260322 ZJH Halcon 深度学习框架训练
    Auto      = 2,  // 20260322 ZJH 自动选择最优框架
    NativeCpp = 3   // 20260322 ZJH OmniMatch 纯 C++ 内置引擎训练
};

// 20260322 ZJH 模型架构枚举（简化版）
// 按任务类型分段编号，每种任务包含若干代表性模型
enum class ModelArchitecture : uint8_t {
    // 20260322 ZJH 分类模型（编号 0-9）
    ResNet18         = 0,   // 20260322 ZJH ResNet-18 基础分类网络
    ResNet50         = 1,   // 20260322 ZJH ResNet-50 深层分类网络
    EfficientNetB0   = 2,   // 20260322 ZJH EfficientNet-B0 高效分类网络
    MobileNetV4Small = 3,   // 20260322 ZJH MobileNetV4-Small 移动端分类
    MobileNetV4Medium= 4,   // 20260322 ZJH MobileNetV4-Medium 中等规模移动端
    ViTTiny          = 5,   // 20260322 ZJH Vision Transformer Tiny 分类
    ConvNeXtTiny     = 6,   // 20260322 ZJH ConvNeXt-Tiny 现代 CNN 分类
    RepVGGA0         = 7,   // 20260322 ZJH RepVGG-A0 重参数化分类

    // 20260322 ZJH 异常检测模型（编号 10-14）
    PaDiM       = 10,  // 20260322 ZJH PaDiM 基于补丁分布的异常检测
    PatchCore   = 11,  // 20260322 ZJH PatchCore 核心补丁记忆库异常检测
    EfficientAD = 12,  // 20260322 ZJH EfficientAD 高效异常检测
    FastFlow    = 13,  // 20260322 ZJH FastFlow 归一化流异常检测
    GCAD        = 14,  // 20260402 ZJH Global Context AD 全局上下文异常检测（布局+纹理双分支）

    // 20260322 ZJH 目标检测模型（编号 20-24）
    YOLOv5Nano  = 20,  // 20260322 ZJH YOLOv5-Nano 轻量检测
    YOLOv8Nano  = 21,  // 20260322 ZJH YOLOv8-Nano 轻量检测
    YOLOv11Nano = 22,  // 20260322 ZJH YOLOv11-Nano 最新轻量检测
    RTDETR      = 23,  // 20260322 ZJH RT-DETR 实时 Transformer 检测

    // 20260322 ZJH 语义分割模型（编号 30-34）
    UNet         = 30,  // 20260322 ZJH U-Net 经典分割网络
    DeepLabV3Plus= 31,  // 20260322 ZJH DeepLabV3+ 空洞卷积分割
    PSPNet       = 32,  // 20260322 ZJH PSPNet 金字塔池化分割
    SegFormer    = 33,  // 20260322 ZJH SegFormer Transformer 分割
    MobileSegNet = 34,  // 20260401 ZJH MobileSegNet 轻量级工业分割（~1.75M参数）

    // 20260322 ZJH 实例分割模型（编号 40-44）
    YOLOv8InstanceSeg = 40,  // 20260322 ZJH YOLOv8 实例分割
    MaskRCNN          = 41,  // 20260322 ZJH Mask R-CNN 实例分割

    // 20260322 ZJH OCR 模型（编号 50-54）
    PaddleOCRv4 = 50,  // 20260322 ZJH PaddleOCR v4 文字识别
    PPOCR       = 51,  // 20260322 ZJH PP-OCR 轻量文字识别
    DBNet       = 52,  // 20260402 ZJH DBNet 可微分二值化文本检测
    DBNetCRNN   = 53,  // 20260402 ZJH DBNet+CRNN 端到端 OCR 流水线（检测+识别）

    // 20260322 ZJH 零样本缺陷检测模型（编号 60-64）
    WinCLIP     = 60,  // 20260322 ZJH WinCLIP 零样本缺陷检测
    AnomalyCLIP = 61,  // 20260322 ZJH AnomalyCLIP 零样本异常检测

    // 20260322 ZJH 零样本目标检测模型（编号 70-74）
    GroundingDINO = 70,  // 20260322 ZJH Grounding DINO 零样本目标检测
    YOLOWorld     = 71,  // 20260322 ZJH YOLO-World 零样本目标检测

    // 20260402 ZJH DL 边缘提取模型（编号 80-84）— 对标 Halcon Edge Extraction
    EdgeUNet      = 80,  // 20260402 ZJH 轻量 U-Net 边缘提取（~0.5M 参数）

    // 20260402 ZJH DL 图像检索（编号 82-84）— 对标海康 VisionMaster
    ImageRetrievalNet = 82,  // 20260402 ZJH 特征嵌入网络（CNN backbone + 全局池化 + L2 归一化）

    // 20260402 ZJH CVPR 2025 / 前沿架构（编号 90-99）
    Dinomaly      = 90,  // 20260402 ZJH Dinomaly (CVPR 2025 SOTA, 99.6% AUROC MVTec AD)
    SAM2UNet      = 91,  // 20260402 ZJH SAM2-UNet (Hiera 编码器 + U-Net 解码器, IoU +20%)
    MobileSAM     = 92   // 20260402 ZJH MobileSAM 无监督分割（轻量 SAM，自动分区无需标注）
};

// 20260330 ZJH 模型能力等级枚举（借鉴海康 VisionTrain "模型能力" 概念）
// 将底层模型架构抽象为用户可理解的能力等级，简化选型决策
enum class ModelCapability : uint8_t {
    Lightweight  = 0,  // 20260330 ZJH 轻量化（推理速度优先，适合边缘/嵌入式部署）
    Normal       = 1,  // 20260330 ZJH 普通（速度与精度均衡，推荐默认选择）
    HighAccuracy = 2   // 20260330 ZJH 高精度（精度优先，适合复杂场景、类别多的任务）
};

// 20260330 ZJH 输入分辨率预设枚举（借鉴海康 VisionTrain 分辨率下拉预设）
// 提供经验验证的分辨率档位，避免用户选择无效尺寸
enum class InputResolutionPreset : uint8_t {
    Res224  = 0,   // 20260330 ZJH 224×224（分类默认，轻量场景）
    Res256  = 8,   // 20260331 ZJH 256×256（分割轻量，16GB 显存 DeepLabV3+ 推荐）
    Res320  = 1,   // 20260330 ZJH 320×320（小目标基础分辨率）
    Res416  = 2,   // 20260330 ZJH 416×416（检测经典分辨率）
    Res512  = 3,   // 20260330 ZJH 512×512（中等分辨率，精度与速度均衡）
    Res640  = 4,   // 20260330 ZJH 640×640（检测/分割高精度）
    Res800  = 5,   // 20260330 ZJH 800×800（大目标高精度场景）
    Res1024 = 6,   // 20260330 ZJH 1024×1024（超高精度，显存需求大）
    Custom  = 7    // 20260330 ZJH 自定义（用户手动输入任意尺寸）
};

// 20260330 ZJH 数据增强预设枚举（借鉴海康 VisionTrain "数据增强策略"）
// 默认配置使用经验最优参数，手动配置允许用户精细调节
enum class AugmentationPreset : uint8_t {
    Default = 0,  // 20260330 ZJH 默认配置（使用任务类型对应的最优增强策略）
    Manual  = 1   // 20260330 ZJH 手动配置（用户自定义全部增强参数）
};

// 20260330 ZJH 异常检测训练模式枚举（借鉴海康 VisionTrain 103 类型 "极速/高精度"）
// 仅在异常检测任务下可见，控制训练策略和模型复杂度
enum class AnomalyTrainingMode : uint8_t {
    Fast         = 0,  // 20260330 ZJH 极速模式（PaDiM/PatchCore，训练快，需样本对齐）
    HighAccuracy = 1   // 20260330 ZJH 高精度模式（EfficientAD/FastFlow，训练慢，检出小缺陷）
};

// 20260322 ZJH 优化器类型枚举
// 控制训练时参数更新策略
enum class OptimizerType : uint8_t {
    Adam  = 0,  // 20260322 ZJH Adam 自适应矩估计优化器
    AdamW = 1,  // 20260322 ZJH AdamW 权重衰减解耦的 Adam 变体
    SGD   = 2,  // 20260322 ZJH 随机梯度下降（带动量）
    Lion  = 3,  // 20260322 ZJH Lion 符号优化器（Google 2023）
    LAMB  = 4   // 20260322 ZJH LAMB 大批量训练优化器
};

// 20260322 ZJH 学习率调度器类型枚举
// 控制训练过程中学习率的衰减策略
enum class SchedulerType : uint8_t {
    CosineAnnealing  = 0,  // 20260322 ZJH 余弦退火（平滑衰减）
    StepLR           = 1,  // 20260322 ZJH 阶梯式衰减（固定步长降低）
    ReduceOnPlateau  = 2,  // 20260322 ZJH 自适应衰减（指标不再提升时降低）
    None             = 3   // 20260322 ZJH 不使用调度器（固定学习率）
};

// 20260322 ZJH 项目工作流状态枚举
// 标记项目当前处于哪个工作阶段（决定哪些页面可用）
enum class ProjectState : uint8_t {
    Created        = 0,  // 20260322 ZJH 项目已创建，尚未导入数据
    DataImported   = 1,  // 20260322 ZJH 数据已导入（图像已加载）
    DataLabeled    = 2,  // 20260322 ZJH 数据已标注（标签已分配）
    DataSplit      = 3,  // 20260322 ZJH 数据已拆分（训练/验证/测试集已分配）
    ModelTrained   = 4,  // 20260322 ZJH 模型已训练（至少一个训练会话完成）
    ModelEvaluated = 5   // 20260322 ZJH 模型已评估（评估报告已生成）
};

// 20260322 ZJH 页面索引常量命名空间
// 定义导航栏 8 个页签的固定索引值
namespace PageIndex {
    constexpr int Project    = 0;  // 20260322 ZJH 项目管理页
    constexpr int Gallery    = 1;  // 20260322 ZJH 图库浏览页
    constexpr int Image      = 2;  // 20260322 ZJH 图像标注页
    constexpr int Inspection = 3;  // 20260322 ZJH 数据检查页
    constexpr int Split      = 4;  // 20260322 ZJH 数据拆分页
    constexpr int Training   = 5;  // 20260322 ZJH 训练配置页
    constexpr int Evaluation = 6;  // 20260322 ZJH 评估分析页
    constexpr int Export     = 7;  // 20260322 ZJH 模型导出页
    constexpr int Count      = 8;  // 20260322 ZJH 页面总数（用于数组大小）
}

// ===== 辅助函数声明 =====

// 20260322 ZJH 任务类型转换为中文显示名称
// 参数: eType - 任务类型枚举值
// 返回: 对应的中文字符串（如 "异常检测"、"图像分类" 等）
QString taskTypeToString(TaskType eType);

// 20260322 ZJH 模型架构转换为英文显示名称
// 参数: eArch - 模型架构枚举值
// 返回: 对应的模型名称字符串（如 "ResNet-18"、"YOLOv8-Nano" 等）
QString modelArchitectureToString(ModelArchitecture eArch);

// 20260322 ZJH 获取指定任务类型支持的全部模型架构列表
// 参数: eType - 任务类型枚举值
// 返回: 该任务可用的模型架构列表（有序）
QVector<ModelArchitecture> architecturesForTask(TaskType eType);

// 20260322 ZJH 获取指定任务类型的默认（推荐）模型架构
// 参数: eType - 任务类型枚举值
// 返回: 默认模型架构（通常为该任务列表中第一个）
ModelArchitecture defaultArchitectureForTask(TaskType eType);

// 20260322 ZJH 判断任务类型是否为零样本任务（不需要训练数据标注）
// 参数: eType - 任务类型枚举值
// 返回: true 表示零样本任务，false 表示常规任务
bool isZeroShotTask(TaskType eType);

// 20260322 ZJH 判断任务类型是否需要训练（零样本任务不需要）
// 参数: eType - 任务类型枚举值
// 返回: true 表示需要训练，false 表示不需要
bool taskRequiresTraining(TaskType eType);

// 20260330 ZJH 根据任务类型和模型能力等级返回推荐的模型架构列表
// 将"模型能力"抽象映射到具体架构，隐藏底层复杂度
// 参数: eType - 任务类型; eCapability - 模型能力等级
// 返回: 该组合下推荐的模型架构列表（有序，第一个为默认推荐）
QVector<ModelArchitecture> architecturesForCapability(TaskType eType, ModelCapability eCapability);

// 20260330 ZJH 获取指定任务类型适用的分辨率预设列表
// 不同任务有不同的合理分辨率范围（如分类偏小，分割偏大）
// 参数: eType - 任务类型
// 返回: 该任务适用的分辨率预设列表
QVector<InputResolutionPreset> resolutionPresetsForTask(TaskType eType);

// 20260330 ZJH 分辨率预设转换为实际像素值
// 参数: ePreset - 分辨率预设枚举值
// 返回: 对应的像素尺寸（正方形边长），Custom 返回 0 表示需用户输入
int resolutionPresetToPixels(InputResolutionPreset ePreset);

// 20260330 ZJH 分辨率预设转换为显示字符串（如 "512×512"）
// 参数: ePreset - 分辨率预设枚举值
// 返回: 用户可读的分辨率描述
QString resolutionPresetToString(InputResolutionPreset ePreset);

// 20260330 ZJH 模型能力等级转换为中文显示名称
// 参数: eCapability - 模型能力枚举值
// 返回: 对应的中文名称（如 "轻量化"、"普通"、"高精度"）
QString modelCapabilityToString(ModelCapability eCapability);

// 20260330 ZJH 根据数据集大小智能推荐训练轮次（借鉴海康 VisionTrain 经验规则）
// 参数: nImageCount - 训练集图片数量; eType - 任务类型
// 返回: 推荐的 epoch 数
int recommendEpochs(int nImageCount, TaskType eType);

// 20260330 ZJH 根据数据集大小智能推荐学习率
// 参数: nImageCount - 训练集图片数量; eType - 任务类型
// 返回: 推荐的初始学习率
double recommendLearningRate(int nImageCount, TaskType eType);

}  // namespace om
