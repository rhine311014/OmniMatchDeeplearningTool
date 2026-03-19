# DeepForge — 纯 C++ 全流程深度学习视觉平台

> 从零构建的纯 C++ 深度学习引擎，不依赖 Python/PyTorch/TensorFlow，核心计算引擎完全自研。

## 快速开始

### 环境要求
- Windows 11
- Visual Studio 2026 (MSVC 14.50+, C++23)
- CMake 4.2+
- vcpkg

### 构建

```powershell
# 在 VS Developer PowerShell 中执行
cmake --preset windows-release
cmake --build build/windows-release
```

### 运行

**GUI 应用（推荐）：**
```bash
build/windows-release/bin/deepforge_app.exe
```

**命令行训练：**
```bash
# MLP 训练（默认，使用合成数据）
build/windows-release/bin/deepforge_train.exe

# ResNet-18 训练
build/windows-release/bin/deepforge_train.exe --model resnet18 --epochs 5

# 自定义参数
build/windows-release/bin/deepforge_train.exe --model mlp --epochs 20 --lr 0.001 --batch-size 128
```

### MNIST 数据（可选）
将 MNIST IDX 文件放在 `data/mnist/` 目录下：
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

无 MNIST 数据时自动使用合成数据集。

## 已实现功能

### 核心引擎
- **Tensor 系统**: Storage+View 分离、零拷贝 reshape/transpose/slice
- **AutoGrad**: 动态计算图、拓扑排序反向传播、链式法则
- **CPUBackend**: matmul、conv2d、batchnorm、pooling、relu、softmax 等

### 神经网络模块
- **层**: Linear, Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Dropout, Flatten, ReLU, Softmax
- **网络**: MLP, ResNet-18 (BasicBlock)
- **优化器**: SGD (with momentum), Adam
- **损失**: CrossEntropyLoss, MSELoss
- **序列化**: .dfm 二进制格式（带 CRC32 校验）

### GUI 应用 (SDL3 + ImGui)
- 训练工作台: 模型选择、超参配置、实时 Loss/Accuracy 曲线
- 推理工作台: 加载模型、图像推理、置信度图表
- 数据管理器: 数据集信息、类别分布
- 模型仓库: .dfm 文件管理

### 平台层
- Logger (spdlog)、Config (JSON)、FileSystem、MemoryPool、ThreadPool、Database (SQLite)

## 架构

```
Layer 5 — UI (SDL3 + ImGui + ImPlot)
Layer 4 — Business (Training, Inference, Data)
Layer 3 — Engine (Tensor, AutoGrad, Module, Optimizer, Loss, Networks)
Layer 2 — HAL (CPUBackend)
Layer 1 — Platform (Logger, Config, FileSystem, Memory, Thread, Database)
```

## 测试

```powershell
cd build/windows-release
ctest --output-on-failure
```

88 个测试覆盖: Tensor、AutoGrad、NN 模块、Conv2d、ResNet-18、序列化。

## 技术栈
- C++23 模块化 (.ixx)
- MSVC 14.50 / CMake 4.2
- SDL3 + Dear ImGui (Docking) + ImPlot
- spdlog / nlohmann-json / SQLite3 / stb_image / GTest
