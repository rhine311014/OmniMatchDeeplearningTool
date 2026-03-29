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

    // 20260322 ZJH 模型架构（默认 ResNet-18，UI 根据任务类型动态填充）
    om::ModelArchitecture eArchitecture = om::ModelArchitecture::ResNet18;

    // 20260322 ZJH 训练设备（CPU / CUDA）
    om::DeviceType eDevice = om::DeviceType::CPU;

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

    // 20260322 ZJH 输入图像尺寸（正方形，范围 32~1024，步长 32）
    int nInputSize = 224;

    // 20260324 ZJH 早停耐心值（验证损失连续 N 轮不下降则停止，范围 1~1000）
    int nPatience = 10;

    // ===== 数据增强 =====

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

    // ===== 导出 =====

    // 20260322 ZJH 训练完成后是否自动导出 ONNX 模型
    bool bExportOnnx = true;
};
