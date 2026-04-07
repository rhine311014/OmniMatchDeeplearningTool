# OmniMatchDeeplearningTool
 — 纯 C++ 全流程深度学习视觉平台

> 从零构建的纯 C++ 深度学习引擎，不依赖 Python/PyTorch/TensorFlow，核心计算引擎完全自研。

## 快速开始

### 环境要求
- Windows 11
- Visual Studio 2026 (MSVC 14.50+, C++23)
- CMake 4.2+
- vcpkg
- Qt 6.10.1

### 构建

```powershell
# 在 VS Developer PowerShell (amd64) 中执行
Enter-VsDevShell -Arch amd64
cmake --preset qt6-release
cmake --build build/qt6-release --target deepforge_app
```

### 运行

**GUI 应用（推荐）：**
```bash
build/qt6-release/bin/deepforge_app.exe
```

**命令行训练：**
```bash
# MLP 训练（默认，使用合成数据）
build/qt6-release/bin/deepforge_train.exe

# ResNet-18 训练
build/qt6-release/bin/deepforge_train.exe --model resnet18 --epochs 5

# 自定义参数
build/qt6-release/bin/deepforge_train.exe --model mlp --epochs 20 --lr 0.001 --batch-size 128
```

### MNIST 数据（可选）
将 MNIST IDX 文件放在 `data/mnist/` 目录下：
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

无 MNIST 数据时自动使用合成数据集。

## 已实现功能

### 核心引擎（38 个 C++23 模块）
- **Tensor 系统**: Storage+View 分离、零拷贝 reshape/transpose/slice
- **AutoGrad**: 动态计算图、拓扑排序反向传播、链式法则
- **CPUBackend + SIMD**: matmul、conv2d、batchnorm、pooling、relu、softmax 等
- **模型**: MLP、ResNet-18/50、MobileNetV4、U-Net、ViT、CRNN、GAN
- **异常检测**: EfficientAD、PatchCore、零样本异常检测/目标检测
- **训练基础设施**: 学习率调度器、数据流水线、检查点、知识蒸馏、FP16 混合精度、并行训练
- **推理**: ONNX 导出、TensorRT 加速

### GUI 应用 (Qt6 Widgets)
- **8 个工作页面**: 项目 / 图库 / 图像标注 / 检查 / 拆分 / 训练 / 评估 / 导出
- **15 个自绘控件**: 缩略图网格 / 可缩放视图 / 标注系统(矩形+多边形+画笔+撤销重做) / 训练损失曲线 / 混淆矩阵热力图 / ROC/PR 曲线 / 饼图 / 直方图 / GradCAM 叠加 / 资源监控 / 模型复杂度
- **5 个对话框**: 设置 / 标签管理 / 增强预览 / 高级训练配置 / 帮助
- **评估后端**: 指标计算器(P/R/F1/AUC) / HTML 报告生成 / CSV 导出 / 模型导出器
- **引擎桥接**: EngineBridge 封装 MLP/ResNet18 训练+推理+序列化 (PIMPL 隔离 C++23 模块)
- **国际化**: 95 条中英文翻译，DF_TR() 宏

### 平台层
- Logger (spdlog)、Config (JSON)、FileSystem、MemoryPool、ThreadPool、Database (SQLite)

## 架构

```
Layer 5 — GUI (Qt6 Widgets + 自绘控件)
Layer 4 — Bridge (PIMPL 封装 C++23→传统 C++)
Layer 3 — Engine (Tensor, AutoGrad, Module, Optimizer, Loss, Networks — 38 个 .ixx 模块)
Layer 2 — HAL (CPUBackend + SIMD 加速)
Layer 1 — Platform (Logger, Config, FileSystem, Memory, Thread, Database)
```

## 测试

```powershell
cd build/qt6-debug
ctest --output-on-failure
```

88 个测试覆盖: Tensor、AutoGrad、NN 模块、Conv2d、ResNet-18、序列化。

## 技术栈
- C++23 模块化 (.ixx)
- MSVC 14.50 / CMake 4.2 / Ninja
- Qt 6.10.1 (Widgets / Concurrent / Svg)
- spdlog / nlohmann-json / SQLite3 / GTest
- vcpkg 包管理
