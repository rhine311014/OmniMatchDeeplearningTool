// 20260322 ZJH TrainingConfig — 训练配置结构体
// 定义模型训练所需的全部超参数和配置选项
// 包含框架选择、模型架构、设备、优化器、调度器、学习率、批量大小、数据增强等

#pragma once

#include "core/DLTypes.h"  // 20260322 ZJH 全局类型定义（枚举类型）

// 20260322 ZJH 训练配置结构体
// 持有一次训练会话所需的全部配置参数
// UI 层填充后传递给 TrainingSession 启动训练
struct TrainingConfig
{
    // ===== 框架与模型 =====

    // 20260322 ZJH 训练框架选择（NativeCpp / Libtorch / Auto）
    om::TrainingFramework eFramework = om::TrainingFramework::NativeCpp;

    // 20260330 ZJH 模型能力等级（轻量化/普通/高精度，借鉴海康 VisionTrain "模型能力"）
    // 用户通过此项选择能力等级，底层自动映射到具体架构
    om::ModelCapability eModelCapability = om::ModelCapability::Normal;

    // 20260322 ZJH 模型架构（默认 ResNet-18，UI 根据任务类型和能力等级动态填充）
    om::ModelArchitecture eArchitecture = om::ModelArchitecture::ResNet18;

    // 20260322 ZJH 训练设备（CPU / CUDA）
    om::DeviceType eDevice = om::DeviceType::CPU;

    // 20260330 ZJH 异常检测训练模式（仅异常检测任务有效）
    // 极速模式使用 PaDiM/PatchCore，高精度模式使用 EfficientAD/FastFlow
    om::AnomalyTrainingMode eAnomalyMode = om::AnomalyTrainingMode::Fast;

    // ===== 优化器与调度器 =====

    // 20260322 ZJH 优化器类型（Adam / AdamW / SGD）
    om::OptimizerType eOptimizer = om::OptimizerType::Adam;

    // 20260322 ZJH 学习率调度器类型（CosineAnnealing / StepLR / None）
    om::SchedulerType eScheduler = om::SchedulerType::CosineAnnealing;

    // ===== 超参数 =====

    // 20260322 ZJH 初始学习率（范围 0.0001~0.1）
    double dLearningRate = 0.001;

    // 20260322 ZJH 每批次样本数量（范围 1~512）
    int nBatchSize = 16;

    // 20260322 ZJH 训练总轮次（范围 1~1000）
    int nEpochs = 50;

    // 20260330 ZJH 输入分辨率预设（借鉴海康 VisionTrain 分辨率下拉，替代原自由输入）
    om::InputResolutionPreset eResolutionPreset = om::InputResolutionPreset::Res224;

    // 20260322 ZJH 输入图像尺寸（正方形，范围 32~1024，步长 32）
    // 20260330 ZJH 当 eResolutionPreset != Custom 时，此值由预设自动填充
    int nInputSize = 224;

    // 20260324 ZJH 早停耐心值（验证损失连续 N 轮不下降则停止，范围 1~1000）
    int nPatience = 10;

    // 20260401 ZJH Crop 尺寸（从高分辨率原图裁剪的 patch 大小，海康默认 720）
    // Patch 模式下先 crop nCropSize×nCropSize 保留缺陷细节，再 resize 到 nInputSize 送入模型
    // 0 表示不使用独立 crop（兼容旧逻辑：直接用 nInputSize 作为 patch 尺寸）
    int nCropSize = 720;

    // ===== 迁移学习（骨干冻结 + 分层学习率） =====

    // 20260331 ZJH 骨干网络冻结轮数（前 N 个 epoch 只训练 head/decoder，不更新编码器权重）
    // 防止高学习率"洗掉"预训练的 ImageNet 权重，0 表示不冻结
    int nFreezeEpochs = 5;

    // 20260331 ZJH 骨干网络学习率倍率（解冻后骨干 LR = 基础 LR × 此值）
    // 典型值 0.1，使骨干微调幅度远小于 head，保留预训练特征
    double dBackboneLrMultiplier = 0.1;

    // ===== 预训练模型与标识 =====

    // 20260330 ZJH 预训练模型文件路径（空字符串表示不使用预训练权重）
    // 支持 .onnx / .omm / .pt 格式
    QString strPretrainedModelPath;

    // 20260330 ZJH 模型标识字符串（用户自定义，仅允许数字/字母/下划线）
    // 训练完成后写入模型文件元数据，方便追溯和管理
    QString strModelTag;

    // ===== 数据增强 =====

    // 20260330 ZJH 数据增强预设模式（默认配置/手动配置）
    // 默认配置根据任务类型自动应用最优增强策略，手动模式允许精细调节
    om::AugmentationPreset eAugPreset = om::AugmentationPreset::Default;

    // 20260322 ZJH 是否启用数据增强
    bool bAugmentation = true;

    // 20260322 ZJH 亮度增强幅度（范围 0~1）
    double dAugBrightness = 0.2;

    // 20260322 ZJH 对比度增强幅度（范围 0~1）
    double dAugContrast = 0.2;

    // 20260322 ZJH 随机水平翻转概率（范围 0~1）
    double dAugFlipProb = 0.5;

    // 20260322 ZJH 随机旋转角度范围（度，范围 0~180）
    double dAugRotation = 15.0;

    // 20260324 ZJH 垂直翻转概率（范围 0~1，默认关闭）
    double dAugVerticalFlipProb = 0.0;

    // 20260324 ZJH 高斯噪声
    bool bAugGaussianNoise = false;      // 20260324 ZJH 是否启用高斯噪声
    double dAugGaussianNoiseStd = 0.02;  // 20260324 ZJH 噪声标准差

    // 20260324 ZJH 颜色抖动
    bool bAugColorJitter = true;         // 20260324 ZJH 是否启用颜色抖动
    double dAugSaturation = 0.2;         // 20260324 ZJH 饱和度抖动范围
    double dAugHue = 0.02;               // 20260324 ZJH 色调抖动范围

    // 20260324 ZJH 随机擦除（Cutout/Random Erasing）
    bool bAugRandomErasing = false;      // 20260324 ZJH 是否启用随机擦除
    double dAugErasingProb = 0.3;        // 20260324 ZJH 擦除概率
    double dAugErasingRatio = 0.15;      // 20260324 ZJH 最大擦除面积比例

    // 20260324 ZJH 随机缩放裁剪（RandomResizedCrop）
    bool bAugRandomCrop = true;          // 20260324 ZJH 是否启用随机缩放裁剪
    double dAugCropScale = 0.8;          // 20260324 ZJH 最小缩放比例

    // 20260324 ZJH 高斯模糊
    bool bAugGaussianBlur = false;       // 20260324 ZJH 是否启用高斯模糊
    double dAugBlurSigma = 1.0;          // 20260324 ZJH 最大模糊 sigma

    // 20260324 ZJH 仿射变换
    bool bAugAffine = false;             // 20260324 ZJH 是否启用仿射变换
    double dAugShearDeg = 10.0;          // 20260324 ZJH 最大剪切角度（度）
    double dAugTranslate = 0.1;          // 20260324 ZJH 最大平移比例

    // 20260324 ZJH Mixup 混合增强
    bool bAugMixup = false;              // 20260324 ZJH 是否启用 Mixup
    double dAugMixupAlpha = 0.2;         // 20260324 ZJH Mixup alpha 参数

    // 20260324 ZJH CutMix 混合增强
    bool bAugCutMix = false;             // 20260324 ZJH 是否启用 CutMix
    double dAugCutMixAlpha = 1.0;        // 20260324 ZJH CutMix alpha 参数

    // ===== HSV 色彩空间抖动 =====

    // 20260330 ZJH 是否启用 HSV 色彩空间增强（与 RGB 颜色抖动互补）
    bool bAugHsvAugment = false;
    // 20260330 ZJH 色调偏移量（范围 0~128）
    double dAugHueShift = 5.0;
    // 20260330 ZJH 饱和度偏移量（范围 0~255）
    double dAugSatShift = 5.0;
    // 20260330 ZJH 明度偏移量（范围 0~255）
    double dAugValShift = 5.0;

    // ===== 少样本学习 =====

    // 20260330 ZJH 是否启用少样本学习模式（5-10张/类即可训练）
    bool bFewShot = false;
    // 20260330 ZJH 每类样本数（范围 1~20，少样本模式下使用）
    int nShotsPerClass = 5;

    // ===== 训练引擎强化（Phase 3）=====

    // 20260401 ZJH 梯度累积步数（1=不累积, 2~8 模拟大 batch）
    // 等效 batch_size = nBatchSize × nGradAccumSteps，用于小显存场景
    int nGradAccumSteps = 1;

    // 20260402 ZJH 归一化层类型（Auto = 根据 batch size 自动选择）
    // batch < 8 时 BN 的 running_mean/var 统计量来自极少样本，几乎是随机数
    // GroupNorm 将 channels 分成 groups，每组内独立归一化，不依赖 batch 维度
    om::NormLayerType eNormLayerType = om::NormLayerType::Auto;

    // 20260402 ZJH GroupNorm 分组数（典型值 32，channels 必须被 nGroupNormGroups 整除）
    int nGroupNormGroups = 32;

    // 20260401 ZJH EMA 权重平均衰减系数（0=关闭, 0.999~0.9999=启用）
    // 推理时用 EMA 权重替代原始权重，平滑更新，通常提升 1~2% 精度
    double dEmaDecay = 0.0;

    // ===== 模型优化（训练后剪枝） =====

    // 20260330 ZJH 是否启用训练后模型剪枝
    bool bPruning = false;
    // 20260330 ZJH 剪枝方法索引（0=非结构化Magnitude, 1=结构化Channel）
    int nPruneMethod = 0;
    // 20260330 ZJH 剪枝比例（范围 0.1~0.8，表示移除权重的比例）
    double dPruneRatio = 0.3;

    // ===== 增量训练 (Continual Learning) =====

    // 20260402 ZJH [OPT-2.5] 是否启用增量训练模式
    // 勾选后: (1) 自动加载现有模型权重作为起点 (2) 使用 EWC 正则化防止灾难性遗忘
    // 适用场景: 生产线新增缺陷类型，不想从头重新训练整个模型
    bool bContinualLearning = false;

    // 20260402 ZJH EWC 正则化系数 λ（Elastic Weight Consolidation）
    // λ 越大，对旧任务参数的惩罚越强，旧知识保留越好，但新任务学习能力受限
    // 典型值 100~10000，工业场景推荐 1000
    double dEwcLambda = 1000.0;

    // ===== 导出 =====

    // 20260322 ZJH 训练完成后是否自动导出 ONNX 模型
    bool bExportOnnx = true;
};
