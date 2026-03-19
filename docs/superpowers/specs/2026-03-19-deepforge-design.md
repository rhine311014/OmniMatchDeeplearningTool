# DeepForge — 纯 C++ 全流程深度学习视觉平台设计文档

> **日期**: 2026-03-19
> **状态**: 已批准
> **作者**: ZJH + Claude

---

## 1. 项目概述

### 1.1 定位

DeepForge 是一个从零构建的纯 C++ 全流程深度学习视觉平台，覆盖**数据标注 → 模型训练 → 推理部署**完整链路。不依赖 Python 运行时，不使用 LibTorch/TensorFlow 等框架，核心计算引擎完全自研。

### 1.2 核心需求

| 维度 | 决策 |
|------|------|
| 定位 | 全流程通用 DL 平台（标注→训练→推理） |
| Python 依赖 | 彻底无 Python，完全自研训练引擎 |
| 任务类型 | 分割 > 检测 > 分类 > 异常检测 > OCR > 实例分割 |
| 网络架构 | L1-L6 全覆盖（CNN→ResNet→U-Net→YOLO→GAN→Transformer） |
| GPU 加速 | CUDA + OpenCL 双平台 |
| 界面 | SDL3 + ImGui + OpenGL（同 Co-creation 方案） |
| 语言 | 纯 C++23 模块化，无 Rust |
| 标注 | 智能标注（含 SAM 式半自动辅助） |
| 模型格式 | 自研 .dfm + ONNX 双向 + TensorRT 引擎 |
| 数据管理 | SQLite + 版本管理 + 增强管线 + 自动划分 |
| 目标平台 | Windows + Linux + 嵌入式 (Jetson) |

### 1.3 架构选型

**方案 A 为主体 + 方案 B 训练分离思想**：

- 分层单体架构，模块边界靠 C++23 `module` 隔离
- 训练用独立线程池（`std::jthread` + 原子状态机），避免阻塞 UI
- 后期可平滑演进为训练服务进程

---

## 2. 五层架构

```
Layer 5 ─ UI 层 (SDL3 + ImGui Docking)
  ├── 标注工作台 (AnnotationWorkbench)
  ├── 训练监控台 (TrainingDashboard)
  ├── 推理工作台 (InferenceWorkbench)
  ├── 数据管理器 (DataManager)
  └── 模型仓库   (ModelRepository)

Layer 4 ─ 业务层
  ├── 标注引擎   (AnnotationEngine)    — 多边形/框/画笔/SAM辅助
  ├── 训练调度器 (TrainingScheduler)   — 任务队列 + 线程池
  ├── 推理引擎   (InferenceEngine)     — 自研/ONNX/TensorRT 三路
  ├── 模型管理器 (ModelManager)        — 格式转换/版本/导入导出
  └── 数据管线   (DataPipeline)        — 增强/划分/版本/加载

Layer 3 ─ 核心计算引擎
  ├── Tensor      — 多维张量 (CPU/GPU 统一接口)
  ├── AutoGrad    — 自动微分 (反向传播计算图)
  ├── Operators   — 算子库 (Conv/BN/ReLU/Pool/Attention...)
  ├── Optimizer   — 优化器 (SGD/Adam/AdamW)
  ├── Loss        — 损失函数 (CE/BCE/Dice/Focal)
  ├── Networks    — 预定义网络 (ResNet/U-Net/YOLO/ViT/AE)
  └── Serializer  — 模型序列化 (自研格式 + ONNX 读写)

Layer 2 ─ 硬件抽象层 (HAL)
  ├── CUDABackend   — CUDA kernel
  ├── OpenCLBackend — OpenCL kernel
  ├── CPUBackend    — SIMD (AVX2/NEON) + OpenMP
  └── DeviceManager — 设备枚举/选择/显存管理

Layer 1 ─ 平台层
  ├── MemoryPool    — 内存/显存池化分配
  ├── ThreadPool    — std::jthread 线程池
  ├── Database      — SQLite RAII 封装
  ├── FileSystem    — 跨平台文件操作
  ├── Logger        — spdlog 多 sink 日志
  └── Config        — JSON 配置管理
```

每层只依赖下层，禁止跨层调用和反向依赖。

---

## 3. 核心计算引擎

### 3.1 Tensor 系统

多维张量，统一 CPU/GPU 内存管理，支持自动微分和多数据类型。

**数据类型系统**：

```cpp
// 从一开始支持多数据类型，避免后期混合精度训练时大规模重构
export enum class DataType {
    Float32,   // 默认训练精度
    Float16,   // 混合精度训练 / 推理优化
    Int32,     // 索引/标签
    Int64,     // 大规模索引
    UInt8      // 图像原始数据
};
```

```cpp
export class Tensor {
public:
    // 工厂方法（支持 DataType 参数，默认 Float32）
    static Tensor zeros(std::vector<int> shape, DataType dtype = DataType::Float32,
                        DeviceType device = DeviceType::CPU);
    static Tensor ones(std::vector<int> shape, DataType dtype = DataType::Float32,
                       DeviceType device = DeviceType::CPU);
    static Tensor randn(std::vector<int> shape, DataType dtype = DataType::Float32,
                        DeviceType device = DeviceType::CPU);
    static Tensor fromData(void* pData, std::vector<int> shape, DataType dtype,
                           DeviceType device);

    // 类型转换
    Tensor to(DataType dtype) const;
    Tensor to(DeviceType device) const;
    Tensor cuda() const;
    Tensor cpu() const;
    Tensor half() const;       // Float32 → Float16
    Tensor toFloat() const;    // → Float32

    // 属性查询
    DataType dtype() const;
    DeviceType device() const;
    int deviceId() const;      // 预留多 GPU 支持

    // 自动微分
    void setRequiresGrad(bool bRequires);
    Tensor grad() const;
    void backward();

    // 运算（挂入计算图）
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor reshape(std::vector<int> shape) const;
    Tensor transpose(int nDim0, int nDim1) const;
    Tensor slice(int nDim, int nStart, int nEnd) const;

private:
    std::shared_ptr<TensorStorage> m_pStorage;  // 实际内存
    std::vector<int> m_vecShape;
    std::vector<int> m_vecStrides;
    int m_nOffset = 0;
    DataType m_dtype = DataType::Float32;
    bool m_bRequiresGrad = false;
    std::shared_ptr<GradFunction> m_pGradFn;
};
```

**TensorStorage** 持有原始内存指针、设备类型和设备 ID（预留多 GPU），引用计数管理生命周期。视图操作（reshape/slice/transpose）共享 Storage，仅改变 shape/strides/offset。

### 3.2 AutoGrad 自动微分引擎

动态计算图，前向传播时自动建图，反向传播时拓扑排序逆序求梯度。

```
前向: z = x.matmul(w) + b   → 每步创建 GradFunction 节点，形成 DAG
反向: loss.backward()
  → 拓扑排序计算图
  → 逆序遍历每个 GradFunction
  → 调用 backward() 计算梯度
  → 梯度累积到叶子节点
```

```cpp
export class GradFunction {
public:
    virtual ~GradFunction() = default;
    virtual std::vector<Tensor> backward(const Tensor& gradOutput) = 0;
    std::vector<std::pair<std::shared_ptr<GradFunction>, int>> m_vecInputs;
};
```

每个算子必须实现对应的 GradFunction 子类（如 MatMulBackward、Conv2dBackward 等）。

### 3.3 Module 基类

所有网络层和网络的基类，类似 PyTorch `nn.Module`：

```cpp
export class Module {
public:
    virtual ~Module() = default;

    // 前向传播（子类必须实现）
    virtual Tensor forward(const Tensor& input) = 0;

    // 多输入前向（可选，如 skip connection）
    virtual Tensor forward(const std::vector<Tensor>& vecInputs);

    // 参数管理
    std::vector<Tensor*> parameters();              // 所有可训练参数（递归）
    std::vector<std::pair<std::string, Tensor*>> namedParameters();

    // 子模块管理
    void registerModule(const std::string& strName, std::shared_ptr<Module> pModule);
    std::vector<std::shared_ptr<Module>> children();

    // 训练/评估模式切换
    void train(bool bMode = true);   // 影响 Dropout / BatchNorm 行为
    void eval();
    bool isTraining() const;

    // 设备转移（递归）
    void to(DeviceType device);
    void to(DataType dtype);

    // 梯度控制
    void zeroGrad();                  // 清零所有参数梯度
    void requiresGrad(bool bRequires);

    // 序列化
    void saveState(const std::string& strPath);
    void loadState(const std::string& strPath);

protected:
    // 子类通过此方法注册可训练参数
    void registerParameter(const std::string& strName, Tensor& param);

private:
    bool m_bTraining = true;
    std::vector<std::pair<std::string, Tensor>> m_vecParameters;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> m_vecSubModules;
};

// Sequential 容器 — 按顺序执行子模块
export class Sequential : public Module {
public:
    template<typename... Modules>
    Sequential(Modules&&... modules);
    void add(std::shared_ptr<Module> pModule);
    Tensor forward(const Tensor& input) override;
};
```

### 3.4 算子库（分批实现）

**第一批（MVP — 支持分类）**：
Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Linear, Dropout, Softmax, Flatten

**第二批（分割 + 检测）**：
ConvTranspose2d, Upsample, Sigmoid, Concat, Split, LeakyReLU, SiLU/Swish, GroupNorm, LayerNorm

**第三批（Transformer + 异常检测）**：
MultiHeadAttention, Embedding, GELU, InstanceNorm, AdaptiveAvgPool2d

每个算子实现：
- `forward()` — CPU / CUDA / OpenCL 三后端
- `backward()` — 梯度计算，同样三后端

每个算子实现时同步提供 `XxxBackward : GradFunction` 子类。第三批算子中 Transformer 相关算子（MultiHeadAttention、Embedding）的 OpenCL 实现标记为可选（OpenCL 对 Transformer 场景性能收益有限），CUDA 和 CPU 为必须。

### 3.5 优化器

| 优化器 | 优先级 | 说明 |
|--------|--------|------|
| SGD + Momentum | P0 | 基础优化器 |
| Adam | P0 | 最常用 |
| AdamW | P1 | 权重衰减解耦 |
| CosineAnnealing LR | P1 | 学习率调度 |
| WarmupLR | P1 | 预热 |

### 3.6 损失函数

| 损失函数 | 用途 |
|----------|------|
| CrossEntropyLoss | 分类 |
| BCEWithLogitsLoss | 二分类/多标签 |
| DiceLoss | 分割 |
| FocalLoss | 类别不平衡 |
| MSELoss | 回归/异常检测 |
| CombinedLoss | Dice + CE 加权组合 |

### 3.7 评估指标 (Metrics)

```cpp
export class Metric {
public:
    virtual ~Metric() = default;
    virtual void update(const Tensor& prediction, const Tensor& target) = 0;
    virtual float compute() = 0;   // 计算最终指标值
    virtual void reset() = 0;      // 重置累积状态

    std::string name() const;
};

// 各任务指标
export class Accuracy : public Metric { /* 分类准确率 */ };
export class MeanIoU : public Metric { /* 分割 mIoU */ };
export class MeanAP : public Metric { /* 检测 mAP@0.5 / mAP@0.5:0.95 */ };
export class F1Score : public Metric { /* F1 = 2*P*R/(P+R) */ };
export class AUC : public Metric { /* 异常检测 ROC-AUC */ };
export class DiceCoefficient : public Metric { /* 分割 Dice */ };
```

训练循环每个 epoch 结束后调用 `compute()` 获取指标，传入 `TrainingStatus::fMetric`。

### 3.8 预定义网络

| 网络 | 任务 | Phase |
|------|------|-------|
| MLP | 验证引擎 | Phase 1 |
| ResNet-18/34/50 | 分类 backbone | Phase 2 |
| U-Net | 语义分割 | Phase 3 |
| YOLOv5/v8 | 目标检测 | Phase 3 |
| ViT | 通用视觉 | Phase 5 |
| AutoEncoder | 异常检测 | Phase 5 |
| GAN | 异常检测 | Phase 5 |
| CRNN | OCR | Phase 5 |
| Mask R-CNN | 实例分割 | Phase 5 |

### 3.9 模型序列化

**自研格式 (.dfm — DeepForge Model)**：

```
文件结构:
  [Magic: "DFM\0"] [Version: uint32]
  [MetaData: JSON]  — 网络架构描述、输入输出形状、任务类型
  [NumLayers: uint32]
  [Layer0: name_len + name + param_count + float32[]]
  [Layer1: ...]
  ...
  [Checksum: CRC32]
```

**ONNX 互操作**：
- 导入：解析 ONNX protobuf → 构建自研 Network 对象 → 加载权重
- 导出：遍历 Network → 构建 ONNX Graph → 序列化 protobuf

**TensorRT 加载**：
- 读取 `.engine` 文件 → TensorRT API 反序列化 → 执行推理

---

## 4. 硬件抽象层 (HAL)

### 4.1 统一接口

```cpp
export class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // 内存管理
    virtual void* allocate(size_t nBytes) = 0;
    virtual void deallocate(void* pPtr) = 0;
    virtual void memcpy(void* pDst, const void* pSrc, size_t nBytes,
                        MemcpyKind kind) = 0;

    // 核心算子
    virtual void conv2d(const float* pInput, const float* pWeight,
                        const float* pBias, float* pOutput,
                        const Conv2dParams& params) = 0;
    virtual void batchNorm(...) = 0;
    virtual void matmul(...) = 0;
    virtual void relu(...) = 0;
    virtual void maxPool2d(...) = 0;
    // ... 每个算子一个虚函数
};
```

### 4.2 三后端实现

| 后端 | 技术 | 适用平台 |
|------|------|---------|
| CUDABackend | CUDA kernel (.cu) | NVIDIA GPU (Windows/Linux/Jetson) |
| OpenCLBackend | OpenCL kernel (.cl) | AMD/Intel GPU, 部分 NVIDIA |
| CPUBackend | AVX2/NEON SIMD + OpenMP | 全平台通用 |

### 4.3 DeviceManager

```cpp
export class DeviceManager {
public:
    static DeviceManager& instance();
    std::vector<DeviceInfo> enumerateDevices();   // 枚举所有可用设备
    ComputeBackend* getBackend(DeviceType type);  // 获取指定后端
    ComputeBackend* getBestBackend();             // 自动选择最优后端
    size_t getAvailableMemory(DeviceType type);   // 查询可用显存/内存
};
```

优先级：CUDA > OpenCL > CPU，运行时自动检测并选择。

**多 GPU 预留设计**：DeviceManager 通过 `deviceId` 支持多设备寻址。当前阶段仅使用单设备（`deviceId=0`），多 GPU 数据并行（Phase 6）的架构思路：
- 每个 GPU 持有模型副本 + 不同 batch 数据
- 反向传播后通过 NCCL (CUDA) 或自研 AllReduce 同步梯度
- Tensor 的 `deviceId()` 方法和 DeviceManager 的多设备枚举已为此预留

---

## 5. 业务层

### 5.1 标注引擎

**标注模式**：
- 矩形框 (BBox) — 目标检测
- 多边形 (Polygon) — 语义分割，点击顶点自动闭合
- 画笔/橡皮擦 (Brush) — 像素级标注，可调笔刷大小
- 图像级标签 (ImageLabel) — 分类
- SAM 辅助 (SAMAssist) — 点击前景/背景点或框选区域，自动生成 mask

**存储格式**：SQLite `annotations` 表

```sql
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    label_id INTEGER REFERENCES labels(id),
    type TEXT,           -- 'bbox' / 'polygon' / 'mask' / 'class'
    data TEXT,           -- JSON: {"points": [...]} / {"bbox": [x,y,w,h]} / {"rle": "..."}
    created_at DATETIME,
    updated_at DATETIME
);
```

### 5.2 训练调度器

**训练配置**：

```cpp
export struct TrainingConfig {
    // 网络配置
    std::string strNetworkType;         // "resnet18" / "unet" / "yolov8" / ...
    int nNumClasses;                    // 输出类别数
    std::vector<int> vecInputShape;     // 输入尺寸 {C, H, W}

    // 数据配置
    std::string strDatasetPath;         // 数据集路径
    float fTrainRatio = 0.8f;           // 训练集比例
    float fValRatio = 0.1f;             // 验证集比例
    int nBatchSize = 16;
    int nNumWorkers = 4;                // DataLoader 线程数

    // 训练超参
    int nEpochs = 100;
    float fLearningRate = 1e-3f;
    std::string strOptimizer = "adam";  // "sgd" / "adam" / "adamw"
    float fWeightDecay = 1e-4f;
    std::string strScheduler = "cosine"; // "step" / "cosine" / "warmup_cosine"
    int nWarmupEpochs = 5;

    // 损失函数
    std::string strLossType = "cross_entropy"; // "dice" / "focal" / "combined"

    // 数据增强
    bool bEnableAugmentation = true;
    std::vector<std::string> vecAugmentations;  // ["flip", "rotate", "color_jitter", ...]

    // 设备配置
    DeviceType deviceType = DeviceType::CUDA;
    int nDeviceId = 0;                  // GPU 设备 ID（预留多 GPU）

    // 预训练权重
    std::string strPretrainedPath;      // 可选: .dfm 或 .onnx 预训练权重路径

    // 检查点
    std::string strCheckpointDir;       // 检查点保存目录
    int nSaveEveryNEpochs = 10;         // 每 N 个 epoch 保存
    bool bEarlyStopping = true;
    int nPatience = 15;                 // 早停耐心值
};
```

```cpp
export class TrainingScheduler {
public:
    std::string submitJob(TrainingConfig config);  // 提交任务，返回 ID
    void pauseJob(const std::string& strJobId);
    void resumeJob(const std::string& strJobId);
    void cancelJob(const std::string& strJobId);
    TrainingStatus getStatus(const std::string& strJobId) const;

    struct TrainingStatus {
        int nCurrentEpoch;
        int nTotalEpochs;
        float fTrainLoss;
        float fValLoss;
        float fMetric;       // mIoU / mAP / accuracy
        float fLearningRate;
        float fEta;          // 预计剩余时间(秒)
        bool bIsRunning;
    };

private:
    ThreadPool m_threadPool{2};  // 最多 2 个并行训练任务
    std::unordered_map<std::string, std::shared_ptr<TrainingJob>> m_mapJobs;
};
```

训练在独立线程中运行，通过原子变量和条件变量与 UI 通信。

### 5.3 推理引擎

三路推理后端，统一接口：

```
InferenceEngine
  ├── NativeInference   — 自研 Tensor 引擎加载 .dfm 格式
  ├── ONNXInference     — 解析 ONNX protobuf → 映射到自研算子
  └── TensorRTInference — 加载 .engine 文件，调用 TensorRT C++ API
```

自动选择策略：TensorRT > Native(CUDA) > Native(OpenCL) > CPU

### 5.4 数据管线

**Dataset 与 DataLoader 接口**：

```cpp
// Dataset 基类 — 抽象数据集访问
export class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;                        // 数据集大小
    virtual std::pair<Tensor, Tensor> getItem(size_t nIndex) = 0;  // 返回 (input, target)
};

// 具体实现
export class ImageSegmentationDataset : public Dataset { /* 图像 + mask */ };
export class ImageDetectionDataset : public Dataset { /* 图像 + bbox + class */ };
export class ImageClassificationDataset : public Dataset { /* 图像 + label */ };

// DataLoader — 多线程批量加载
export class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> pDataset, int nBatchSize,
               bool bShuffle = true, int nNumWorkers = 4,
               int nPrefetchCount = 2);

    // 迭代器接口
    class Iterator {
    public:
        std::pair<Tensor, Tensor> operator*();   // 返回 (batchInput, batchTarget)
        Iterator& operator++();
        bool operator!=(const Iterator& other);
    };
    Iterator begin();
    Iterator end();

private:
    std::shared_ptr<Dataset> m_pDataset;
    int m_nBatchSize;
    bool m_bShuffle;
    ThreadPool m_workerPool;      // 多线程预加载
    std::queue<std::pair<Tensor, Tensor>> m_queuePrefetch;  // 预取队列
};
```

DataLoader 在后台线程中预加载和预增强下一批数据，主线程训练当前 batch 时下一 batch 已就绪。

**数据增强操作**：
- 几何：RandomFlip, RandomRotate, RandomCrop, RandomScale, Affine
- 颜色：ColorJitter, Normalize, GaussianNoise, GaussianBlur
- 高级：Mosaic, MixUp, CutMix, CutOut
- 分割专用：标签同步变换（图像和 mask 一起变换）

**数据集管理**：
- SQLite 管理图像路径、标注、元数据
- 自动划分 train/val/test（可配置比例）
- 数据集版本快照（基于文件哈希）
- DataLoader 多线程预加载 + 预增强

---

## 6. UI 层

### 6.1 技术栈

| 组件 | 技术 |
|------|------|
| 窗口 | SDL3 |
| UI | Dear ImGui (Docking 实验版) |
| 渲染主力 | OpenGL 4.6 |
| 渲染回退 | SDL3 Renderer |
| 中文字体 | 微软雅黑 + Font Awesome 6 |
| 图表 | ImPlot |
| 配色 | 工业深色主题 |

### 6.2 布局

```
┌─────────────────────────────────────────────────────────────┐
│  DeepForge v0.1.0                              [_][□][×]   │
├─────────────────────────────────────────────────────────────┤
│ [标注] [训练] [推理] [数据] [模型]              GPU: 62%   │
├───────────────────────────────────┬─────────────────────────┤
│                                   │  属性面板               │
│      主工作区                     │  ├── 标注模式: 标签列表 │
│      (图像/训练曲线/推理结果)     │  ├── 训练模式: 超参配置 │
│                                   │  ├── 推理模式: 结果详情 │
│                                   │  └── 数据模式: 统计图表 │
│                                   │                         │
├───────────────────────────────────┴─────────────────────────┤
│ 状态栏: 训练进度 | 设备状态 | 内存占用 | FPS               │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 五个工作台

| 工作台 | 功能 |
|--------|------|
| AnnotationWorkbench | 图像浏览、标注工具栏、标签管理、SAM 辅助 |
| TrainingDashboard | 超参配置、Loss/Metric 曲线、训练控制、日志 |
| InferenceWorkbench | 模型加载、图像推理、结果叠加显示、批量推理 |
| DataManager | 数据集导入、增强配置、划分、版本管理、统计 |
| ModelRepository | 模型列表、版本、格式转换、导入导出、对比 |

---

## 7. 项目结构

```
DeepForge/
├── src/
│   ├── platform/                    ← Layer 1
│   │   ├── df.platform.memory.ixx
│   │   ├── df.platform.thread_pool.ixx
│   │   ├── df.platform.database.ixx
│   │   ├── df.platform.filesystem.ixx
│   │   ├── df.platform.logger.ixx
│   │   └── df.platform.config.ixx
│   ├── hal/                         ← Layer 2
│   │   ├── df.hal.device.ixx
│   │   ├── df.hal.cuda_backend.ixx
│   │   ├── df.hal.opencl_backend.ixx
│   │   └── df.hal.cpu_backend.ixx
│   ├── engine/                      ← Layer 3
│   │   ├── df.engine.tensor.ixx
│   │   ├── df.engine.autograd.ixx
│   │   ├── df.engine.operators.ixx
│   │   ├── df.engine.optimizer.ixx
│   │   ├── df.engine.loss.ixx
│   │   ├── df.engine.networks.ixx
│   │   └── df.engine.serializer.ixx
│   ├── business/                    ← Layer 4
│   │   ├── df.biz.annotation.ixx
│   │   ├── df.biz.training.ixx
│   │   ├── df.biz.inference.ixx
│   │   ├── df.biz.model_manager.ixx
│   │   └── df.biz.data_pipeline.ixx
│   ├── ui/                          ← Layer 5
│   │   ├── df.ui.app.ixx
│   │   ├── df.ui.annotation_workbench.ixx
│   │   ├── df.ui.training_dashboard.ixx
│   │   ├── df.ui.inference_workbench.ixx
│   │   ├── df.ui.data_manager.ixx
│   │   └── df.ui.model_repository.ixx
│   ├── cuda/                        ← CUDA kernel
│   │   ├── conv2d.cu
│   │   ├── batchnorm.cu
│   │   ├── activation.cu
│   │   ├── pooling.cu
│   │   ├── matmul.cu
│   │   └── attention.cu
│   ├── opencl/                      ← OpenCL kernel
│   │   ├── conv2d.cl
│   │   ├── batchnorm.cl
│   │   ├── activation.cl
│   │   ├── pooling.cl
│   │   ├── matmul.cl
│   │   └── attention.cl              ← Transformer 算子（可选，标记为实验性）
│   └── main.cpp
├── include/                         ← 公共类型定义
│   └── df_types.h
├── resources/
│   ├── fonts/
│   │   ├── msyh.ttc
│   │   ├── fa-solid-900.ttf
│   │   └── IconsFontAwesome6.h
│   └── icons/
├── config/
│   └── default_config.json
├── data/                            ← 运行时数据目录
│   ├── datasets/
│   ├── models/
│   └── deepforge.db
├── tests/
│   ├── test_tensor.cpp
│   ├── test_autograd.cpp
│   ├── test_operators.cpp
│   └── test_networks.cpp
├── third_party/
│   ├── imgui/
│   ├── implot/
│   ├── glad/
│   ├── stb/
│   └── json/
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── DEVLOG.md
└── README.md
```

---

## 8. 第三方依赖

| 库 | 版本 | 用途 | 集成方式 |
|---|---|---|---|
| SDL3 | 3.4+ | 窗口/事件/GPU初始化 | vcpkg |
| Dear ImGui | 1.92+ Docking | 即时模式 UI | 源码编入 |
| ImPlot | latest | ImGui 图表扩展 | 源码编入 |
| glad | GL 4.6 | OpenGL 加载 | 源码编入 |
| spdlog | 1.x | 日志 | vcpkg |
| nlohmann-json | 3.x | JSON 配置/ONNX | 单头文件 |
| SQLite3 | 3.x | 数据库 | 源码编入 |
| stb_image/write | latest | 图像读写 | 单头文件 |
| protobuf | 3.x | ONNX 格式序列化/反序列化 | vcpkg |
| onnx (proto定义) | 1.x | onnx.proto / onnx.proto3 定义文件 | 源码编入 |
| Google Test | 1.x | 单元测试 | vcpkg |

**系统级依赖**（不打包）：
- CUDA Toolkit 12.x
- OpenCL SDK (Khronos / vendor)
- TensorRT (可选，仅推理)

---

## 9. 错误处理策略

### 9.1 总体原则

- 使用 `std::expected<T, Error>` (C++23) 作为主要错误传播机制，避免异常开销
- GPU 异步操作错误通过 `cudaGetLastError()` / `clGetEventInfo()` 同步检查
- 仅在不可恢复场景（程序初始化失败、内存耗尽）抛出 `std::runtime_error`

### 9.2 错误类型

```cpp
export enum class ErrorCode {
    Success = 0,
    // 张量错误
    ShapeMismatch,        // 形状不匹配
    DTypeMismatch,        // 数据类型不匹配
    DeviceMismatch,       // 设备不匹配
    // GPU 错误
    CudaError,            // CUDA 调用失败
    OpenCLError,          // OpenCL 调用失败
    OutOfMemory,          // 显存/内存不足
    // IO 错误
    FileNotFound,         // 文件不存在
    InvalidFormat,        // 文件格式错误
    SerializationError,   // 序列化/反序列化失败
    // 训练错误
    NaNDetected,          // 训练中出现 NaN
    GradientExplosion,    // 梯度爆炸
};

export struct Error {
    ErrorCode code;
    std::string strMessage;
    std::string strFile;     // 发生位置
    int nLine;
};

// 使用方式
template<typename T>
using Result = std::expected<T, Error>;
```

### 9.3 GPU 显存 OOM 处理

1. 训练前预估显存需求（基于 batch_size + 模型参数量 + 激活缓存）
2. OOM 时自动尝试：缩小 batch_size → 清理缓存 → 切换到 CPU 回退
3. 通过 MemoryPool 监控显存水位，超过 90% 阈值时预警

---

## 10. 构建系统

### 10.1 CMake 配置要点

```cmake
cmake_minimum_required(VERSION 3.28)  # C++23 模块支持所需最低版本
project(DeepForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++23 模块化支持
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491ade7")
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)

# CUDA 条件编译
option(DF_ENABLE_CUDA "Enable CUDA backend" ON)
if(DF_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
endif()

# OpenCL 条件编译
option(DF_ENABLE_OPENCL "Enable OpenCL backend" ON)
if(DF_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
endif()

# TensorRT 可选
option(DF_ENABLE_TENSORRT "Enable TensorRT inference" OFF)
if(DF_ENABLE_TENSORRT)
    find_package(TensorRT REQUIRED)
endif()

# vcpkg 依赖
find_package(SDL3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(unofficial-sqlite3 REQUIRED)
find_package(GTest REQUIRED)
find_package(protobuf REQUIRED)
```

### 10.2 OpenCL kernel 编译

OpenCL kernel (`.cl`) 文件在构建时通过 `xxd` 或自定义 CMake 脚本嵌入为 C 字符串常量，运行时通过 `clCreateProgramWithSource()` 编译。无需用户安装 OpenCL 编译器。

### 10.3 跨平台编译预设

| 预设 | 平台 | 编译器 | 说明 |
|------|------|--------|------|
| `windows-debug` | Windows | MSVC 17.10+ | 开发调试 |
| `windows-release` | Windows | MSVC 17.10+ | 发布 |
| `linux-debug` | Linux | GCC 14+ / Clang 18+ | 开发调试 |
| `linux-release` | Linux | GCC 14+ / Clang 18+ | 发布 |
| `jetson-release` | Jetson (aarch64) | GCC 14+ | 嵌入式部署 |

---

## 11. 预训练权重策略

### 11.1 权重来源

纯自研引擎从零训练大型网络（ResNet-50、ViT）收敛慢且成本高。预训练权重策略：

1. **ONNX 导入转换**：从 PyTorch 导出的 ONNX 模型中提取权重 → 映射到自研 Module 参数 → 保存为 .dfm 格式。这是一次性操作，可由开发者预先完成。
2. **权重包分发**：将常用 backbone（ResNet-18/34/50、MobileNet）的预训练权重转换为 .dfm 格式，随软件分发或提供下载。
3. **迁移学习**：加载预训练 backbone 权重 → 冻结 backbone → 只训练 head → 逐步解冻微调。

### 11.2 权重映射

```cpp
// ONNX 权重名 → 自研 Module 参数名 的映射规则
// 示例: "features.0.weight" → Module("features")->child(0)->parameter("weight")
export class WeightMapper {
public:
    void loadFromONNX(Module& module, const std::string& strOnnxPath);
    void loadFromDFM(Module& module, const std::string& strDfmPath);
};
```

---

## 12. 开发路线图

### Phase 1 — 地基（~4-6 周）

- 平台层：MemoryPool, ThreadPool, Logger, Database, Config, FileSystem
- Tensor：CPU 版本 + 基础运算（加减乘除、matmul、reshape、transpose、slice）
- AutoGrad：动态计算图 + 反向传播
- CPUBackend：朴素 C++ 实现（先正确后优化）
- **验证**：手写 2 层 MLP 在 MNIST 上收敛

### Phase 2 — 卷积核心（~4-6 周）

- 第一批算子：Conv2d, BN, ReLU, MaxPool2d, AvgPool2d, Linear, Dropout, Softmax, Flatten
- CUDA Backend：conv2d, matmul, batchnorm kernel
- 优化器：SGD + Adam
- 损失函数：CrossEntropy
- ResNet-18 网络定义
- 模型序列化 .dfm 格式
- **验证**：ResNet-18 CIFAR-10 准确率 > 90%

### Phase 3 — 分割与检测（~6-8 周）

- 第二批算子：ConvTranspose2d, Upsample, Sigmoid, Concat, Split, LeakyReLU, SiLU, GroupNorm, LayerNorm
- U-Net 网络定义 + 训练
- YOLO 检测头 + 训练
- OpenCL Backend
- ONNX 导入/导出
- 数据增强管线
- **验证**：U-Net VOC mIoU > 60%, YOLO VOC mAP > 50%（COCO mAP 目标推迟到 Phase 5 优化后评估）

### Phase 4 — UI 平台（~6-8 周）

- SDL3 + ImGui 框架搭建（双渲染后端）
- 标注工作台：矩形/多边形/画笔标注
- 训练监控台：Loss 曲线、指标、进度、控制
- 推理工作台：模型加载、结果叠加显示
- 数据管理器：导入、划分、增强配置
- 模型仓库：列表、版本、格式转换

### Phase 5 — 高级功能（~8-10 周）

- 第三批算子：MultiHeadAttention, Embedding, GELU, InstanceNorm, AdaptiveAvgPool2d
- ViT / Transformer 网络
- AutoEncoder / GAN 异常检测
- SAM 辅助标注（通过 ONNX 推理路径加载外部预训练 SAM encoder，非自研训练）
- TensorRT 引擎集成
- CRNN OCR 网络
- Mask R-CNN 实例分割
- Jetson 嵌入式适配

### Phase 6 — 打磨优化（持续）

- CUDA kernel 性能优化（tiling / shared memory / tensor core）
- 混合精度训练 (FP16)
- 数据集版本管理完善
- 多 GPU 数据并行
- SIMD 优化 CPU 后端 (AVX2 / NEON)
- 完整文档与测试覆盖

---

## 13. 关键设计决策记录

| # | 决策 | 理由 |
|---|------|------|
| D1 | 纯 C++23，无 Python | 用户硬性要求，目标完全自主可控 |
| D2 | 自研 AutoGrad 而非静态图 | 动态图调试方便，灵活性高，与 PyTorch 思路一致 |
| D3 | HAL 抽象层统一三后端 | 一套算子接口，三种实现，设备切换透明 |
| D4 | 训练用线程池而非独立进程 | 避免 IPC 复杂度，后期可演进为进程分离 |
| D5 | Phase 1 先 CPU 再 GPU | CPU 实现作为正确性参考，GPU kernel 可对比验证 |
| D6 | ImGui 而非自研 UI 框架 | ImGui 极轻量（源码编入），开发效率远超自研控件系统 |
| D7 | .dfm 自研格式 + ONNX 互通 | 自研格式性能最优，ONNX 保证生态兼容 |
| D8 | C++23 模块化 (.ixx) | 编译隔离、依赖清晰，与 Co-creation 一致 |

---

## 14. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 自研 AutoGrad 正确性 | 训练不收敛 | 与 PyTorch 逐算子对比梯度，数值梯度检查 |
| CUDA kernel 性能 | 训练慢于 PyTorch | 先保证正确，后期逐步优化热点 kernel |
| OpenCL 生态碎片化 | AMD/Intel 行为差异 | 严格遵循 OpenCL 规范，多设备测试 |
| C++23 模块编译器支持 | MSVC/GCC/Clang 差异 | 主力 MSVC，Linux 用最新 GCC/Clang |
| 项目规模过大 | 开发周期失控 | 严格按 Phase 交付，每阶段有明确验证指标 |
| 工期估计偏乐观 | 实际开发超时 | 路线图为理想估时，实际按 1.5-2x 系数调整；单人开发建议先完成 Phase 1-3 核心引擎 |
| 预训练权重缺失 | 从零训练收敛慢 | 通过 ONNX 导入 PyTorch 预训练权重，一次性转换为 .dfm 格式分发 |
