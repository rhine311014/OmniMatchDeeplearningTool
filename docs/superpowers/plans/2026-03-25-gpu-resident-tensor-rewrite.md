# GPU-Resident Tensor 系统重写实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写 Tensor 系统使训练数据/权重/梯度全部驻留 GPU 内存，消除 CPU↔GPU 数据乒乓，达到 Halcon 级 90%+ GPU 利用率。

**Architecture:** TensorStorage 扩展为 CPU/CUDA 双后端；Tensor 添加 `.to(device)` 迁移；创建 CUDABackend 镜像 CPUBackend 接口；tensor_ops 按设备类型自动调度；EngineBridge::train() 一次性上传数据到 GPU 后全程 GPU 计算。

**Tech Stack:** C++23 Modules (.ixx), CUDA 12.8, MSVC 14.50, AVX2/OpenMP (CPU fallback), CMake + Ninja

**关键约束:**
- 所有修改必须 `#ifdef DF_HAS_CUDA` 保护，无 CUDA 时 CPU 路径完全不受影响
- Tensor 按值传递语义不变（shared_ptr<TensorStorage> 共享）
- AutoGrad 系统不需要修改（type-erased，设备无关）
- Module/Linear/Conv2d 等上层模块不需要修改（只操作 Tensor 接口）

---

## 阶段概述

| 阶段 | 内容 | 影响范围 |
|------|------|----------|
| Phase 1 | TensorStorage GPU 感知 + Tensor.to(device) | tensor_storage.ixx, tensor.ixx |
| Phase 2 | CUDABackend 全算子实现 | 新建 df.hal.cuda_backend.ixx |
| Phase 3 | tensor_ops 设备调度 | tensor_ops.ixx |
| Phase 4 | EngineBridge GPU-resident 训练 | EngineBridge.cpp |
| Phase 5 | 验证 + 性能调优 | 全链路测试 |

---

## Phase 1: TensorStorage GPU 感知 + Tensor 设备迁移

### Task 1.1: TensorStorage 支持 CUDA 内存分配

**Files:**
- Modify: `src/engine/df.engine.tensor_storage.ixx`

- [ ] **Step 1: 添加 CUDA 内存分配路径**

在构造函数中根据 DeviceType 分支：
```cpp
// 20260325 ZJH GPU 感知构造函数
TensorStorage(size_t nBytes, DeviceType eDevice = DeviceType::CPU, int nDeviceId = 0)
    : m_nBytes(nBytes), m_deviceType(eDevice), m_nDeviceId(nDeviceId)
{
    if (eDevice == DeviceType::CUDA) {
#ifdef DF_HAS_CUDA
        dfCudaMalloc(&m_pData, nBytes);  // GPU 分配
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        m_pData = _aligned_malloc(nBytes, 64);  // CPU 64 字节对齐
    }
}
```

- [ ] **Step 2: 修改析构函数根据设备类型释放**

```cpp
~TensorStorage() {
    if (m_pData) {
        if (m_deviceType == DeviceType::CUDA) {
#ifdef DF_HAS_CUDA
            dfCudaFree(m_pData);
#endif
        } else {
            _aligned_free(m_pData);  // MSVC
        }
        m_pData = nullptr;
    }
}
```

- [ ] **Step 3: 添加设备间拷贝方法**

```cpp
// 20260325 ZJH 创建指定设备上的副本
std::shared_ptr<TensorStorage> copyTo(DeviceType eTargetDevice, int nTargetDeviceId = 0) const {
    auto pNew = std::make_shared<TensorStorage>(m_nBytes, eTargetDevice, nTargetDeviceId);

    if (m_deviceType == DeviceType::CPU && eTargetDevice == DeviceType::CUDA) {
        dfCudaCopyH2D(pNew->mutableData(), m_pData, m_nBytes);  // CPU → GPU
    } else if (m_deviceType == DeviceType::CUDA && eTargetDevice == DeviceType::CPU) {
        dfCudaCopyD2H(pNew->mutableData(), m_pData, m_nBytes);  // GPU → CPU
    } else if (m_deviceType == DeviceType::CUDA && eTargetDevice == DeviceType::CUDA) {
        dfCudaCopyD2D(pNew->mutableData(), m_pData, m_nBytes);  // GPU → GPU
    } else {
        std::memcpy(pNew->mutableData(), m_pData, m_nBytes);    // CPU → CPU
    }
    return pNew;
}
```

- [ ] **Step 4: 添加 extern "C" CUDA 函数声明（在 module global fragment）**

- [ ] **Step 5: 编译验证**

---

### Task 1.2: Tensor 添加 `.to(device)` 和设备查询

**Files:**
- Modify: `src/engine/df.engine.tensor.ixx`

- [ ] **Step 1: 修改 device() 返回实际设备类型**

```cpp
DeviceType device() const {
    return m_pStorage ? m_pStorage->deviceType() : DeviceType::CPU;
}
int deviceId() const {
    return m_pStorage ? m_pStorage->deviceId() : 0;
}
bool isCuda() const { return device() == DeviceType::CUDA; }
bool isCpu() const { return device() == DeviceType::CPU; }
```

- [ ] **Step 2: 添加 to(device) 设备迁移方法**

```cpp
// 20260325 ZJH 将 Tensor 迁移到指定设备（返回新 Tensor，共享 shape/strides）
Tensor to(DeviceType eDevice, int nDeviceId = 0) const {
    if (device() == eDevice && deviceId() == nDeviceId) {
        return *this;  // 已在目标设备，零拷贝返回
    }
    // 非连续张量先做 contiguous
    Tensor src = isContiguous() ? *this : contiguous();
    auto pNewStorage = src.m_pStorage->copyTo(eDevice, nDeviceId);
    Tensor result;
    result.m_pStorage = pNewStorage;
    result.m_vecShape = src.m_vecShape;
    result.m_vecStrides = src.m_vecStrides;
    result.m_nOffset = 0;  // 新 storage，offset 重置
    result.m_bRequiresGrad = src.m_bRequiresGrad;
    return result;
}
Tensor cuda(int nDeviceId = 0) const { return to(DeviceType::CUDA, nDeviceId); }
Tensor cpu() const { return to(DeviceType::CPU); }
```

- [ ] **Step 3: 修改工厂方法支持设备参数**

```cpp
static Tensor zeros(const std::vector<int>& shape, DeviceType eDevice = DeviceType::CPU);
static Tensor ones(const std::vector<int>& shape, DeviceType eDevice = DeviceType::CPU);
static Tensor randn(const std::vector<int>& shape, DeviceType eDevice = DeviceType::CPU);
static Tensor fromData(const float* pSrc, const std::vector<int>& shape,
                       DeviceType eDevice = DeviceType::CPU);
```

对于 GPU：先在 CPU 创建再 `.to(CUDA)`（简单可靠）。

- [ ] **Step 4: 编译验证**

---

## Phase 2: CUDABackend 全算子镜像

### Task 2.1: 创建 CUDABackend 模块

**Files:**
- Create: `src/hal/df.hal.cuda_backend.ixx`
- Modify: `CMakeLists.txt`（添加到 df_hal 源文件列表）

- [ ] **Step 1: 创建模块骨架**

```cpp
module;
#ifdef DF_HAS_CUDA
extern "C" {
    // 所有 CUDA kernel C 绑定声明
    int dfCudaMatmul(...);
    int dfCudaAdd(...);
    // ...
}
#endif

export module df.hal.cuda_backend;

export namespace df {

class CUDABackend {
public:
    // 与 CPUBackend 镜像的接口
    static void matmul(const float* pA, const float* pB, float* pC, int nM, int nK, int nN);
    static void add(const float* pA, const float* pB, float* pOut, size_t nCount);
    static void mul(const float* pA, const float* pB, float* pOut, size_t nCount);
    static void relu(const float* pIn, float* pOut, size_t nCount);
    static void softmax(const float* pIn, float* pOut, int nBatch, int nClasses);
    static void conv2d(...);
    static void batchNorm2d(...);
    static void maxPool2d(...);
    static void avgPool2d(...);
    // 反向传播
    static void reluBackward(...);
    static void conv2dBackwardInput(...);
    static void conv2dBackwardWeight(...);
    static void batchNorm2dBackward(...);
    // 优化器 step（直接在 GPU 上更新权重）
    static void adamStep(float* pParam, const float* pGrad, float* pM, float* pV,
                         int nCount, float fLr, float fBeta1, float fBeta2, float fEps, int nStep);
    static void sgdStep(float* pParam, const float* pGrad, float* pVelocity,
                        int nCount, float fLr, float fMomentum);
    // 辅助
    static void fillZeros(float* pData, size_t nCount);
    static void fillOnes(float* pData, size_t nCount);
    static void copy(const float* pSrc, float* pDst, size_t nCount);
};

} // namespace df
```

- [ ] **Step 2: 实现每个方法调用对应 CUDA kernel C 绑定**

所有方法直接调用 `dfCudaXxx()` 函数——指针已经在 GPU 上，无需传输。

- [ ] **Step 3: 编译验证**

---

### Task 2.2: 扩展 CUDA Kernels

**Files:**
- Modify: `src/cuda/cuda_kernels.cu`
- Modify: `src/cuda/cuda_kernels.cuh`

当前 cuda_kernels.cu 已有：matmul, add, sub, mul, relu, sigmoid, softmax, batchnorm, im2col, transpose, reduction

需要新增：
- [ ] **Step 1: 添加缺失的前向核**
  - leakyRelu, gelu, silu, tanh
  - avgPool2d, maxPool2d (带 indices 保存)
  - adaptiveAvgPool2d
  - conv2d (使用现有 im2col + matmul)
  - addBias (广播加法)
  - dropout (随机 mask)
  - layerNorm
  - fillZeros, fillOnes, fillRandn (cuRAND)

- [ ] **Step 2: 添加反向传播核**
  - reluBackward, sigmoidBackward, geluBackward, siluBackward
  - conv2dBackwardInput (col2im), conv2dBackwardWeight
  - maxPool2dBackward (使用 indices)
  - avgPool2dBackward
  - batchNorm2dBackward (已有), layerNormBackward
  - softmaxCrossEntropyBackward
  - addBackward (identity), mulBackward

- [ ] **Step 3: 添加优化器核**
  - adamStep kernel（直接在 GPU 上执行 Adam 参数更新）
  - sgdStep kernel（直接在 GPU 上执行 SGD+momentum 更新）
  这是关键——避免 GPU→CPU→GPU 的参数更新循环

- [ ] **Step 4: 添加辅助核**
  - dfCudaCopy (D2D)
  - dfCudaFill (memset)
  - dfCudaRandn (cuRAND)
  - dfCudaArgmax

- [ ] **Step 5: 更新 cuda_kernels.cuh 声明**

- [ ] **Step 6: 编译验证**

---

## Phase 3: tensor_ops 设备自动调度

### Task 3.1: 添加设备调度宏/函数

**Files:**
- Modify: `src/engine/df.engine.tensor_ops.ixx`

- [ ] **Step 1: 添加 import cuda_backend**

```cpp
import df.hal.cpu_backend;
#ifdef DF_HAS_CUDA
import df.hal.cuda_backend;
#endif
```

- [ ] **Step 2: 创建调度辅助**

```cpp
// 20260325 ZJH 设备一致性检查 + 后端选择
inline bool isCudaTensor(const Tensor& t) {
    return t.device() == DeviceType::CUDA;
}

// 确保两个张量在同一设备上
inline void checkSameDevice(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::runtime_error("Tensor device mismatch: " +
            std::to_string(static_cast<int>(a.device())) + " vs " +
            std::to_string(static_cast<int>(b.device())));
    }
}
```

- [ ] **Step 3: 修改每个 tensor op 添加设备分支**

以 tensorMatmul 为例：
```cpp
Tensor tensorMatmul(const Tensor& a, const Tensor& b) {
    checkSameDevice(a, b);
    // ... shape 计算 ...

    Tensor result = Tensor::zeros({nM, nN}, a.device());

    if (isCudaTensor(a)) {
#ifdef DF_HAS_CUDA
        CUDABackend::matmul(a.floatDataPtr(), b.floatDataPtr(),
                            result.mutableFloatDataPtr(), nM, nK, nN);
#endif
    } else {
        CPUBackend::matmul(a.floatDataPtr(), b.floatDataPtr(),
                           result.mutableFloatDataPtr(), nM, nK, nN);
    }

    // AutoGrad 部分不变
    return result;
}
```

对每个 op（约 40 个）做同样修改。模式统一：
1. checkSameDevice
2. 输出张量在输入设备上创建
3. if (isCudaTensor) → CUDABackend::xxx else → CPUBackend::xxx
4. AutoGrad 代码不变

- [ ] **Step 4: 编译验证**

---

### Task 3.2: 修改优化器直接在 GPU 上更新参数

**Files:**
- Modify: `src/engine/df.engine.optimizer.ixx`

- [ ] **Step 1: Adam::step() 添加 GPU 路径**

```cpp
void Adam::step() {
    m_nStep++;
    for (size_t i = 0; i < m_vecParams.size(); ++i) {
        Tensor* pParam = m_vecParams[i];
        Tensor grad = tensorGetGrad(*pParam);
        if (grad.numel() == 0) continue;

        if (pParam->isCuda()) {
#ifdef DF_HAS_CUDA
            // 全部在 GPU 上完成——权重、梯度、m、v 都在 GPU
            CUDABackend::adamStep(
                pParam->mutableFloatDataPtr(), grad.floatDataPtr(),
                m_vecM[i].mutableFloatDataPtr(), m_vecV[i].mutableFloatDataPtr(),
                pParam->numel(), m_fLr, m_fBeta1, m_fBeta2, m_fEps, m_nStep);
#endif
        } else {
            // 原有 CPU 路径
            // ...
        }
    }
}
```

- [ ] **Step 2: SGD::step() 同样添加 GPU 路径**

- [ ] **Step 3: 确保 m_vecM / m_vecV / m_vecVelocities 在参数同一设备上创建**

在 step() 首次调用时检查：如果参数在 GPU 上但 m/v 在 CPU 上，自动迁移。

- [ ] **Step 4: 编译验证**

---

## Phase 4: EngineBridge GPU-Resident 训练

### Task 4.1: 训练数据一次性上传 GPU

**Files:**
- Modify: `src/engine/bridge/EngineBridge.cpp`

- [ ] **Step 1: train() 中添加数据上传逻辑**

```cpp
bool EngineBridge::train(const BridgeTrainParams& params, ...) {
    // ... 现有初始化 ...

    // 20260325 ZJH GPU-Resident 训练：一次性上传全部数据到 GPU
    bool bCuda = params.bUseCuda;
#ifdef DF_HAS_CUDA
    if (bCuda) {
        dfCudaInit(0);

        // 将模型参数迁移到 GPU
        auto vecParams = m_pImpl->pModel->parameters();
        for (auto* pParam : vecParams) {
            *pParam = pParam->cuda();  // 权重 → GPU
        }

        // 将训练数据打包为 GPU Tensor
        // 全部训练数据一次 H2D 传输
        Tensor tAllTrainData = Tensor::fromData(vecTrainData.data(),
            {nTrainCount, nInputDim}).cuda();
        Tensor tAllValData = Tensor::fromData(vecValData.data(),
            {nValCount, nInputDim}).cuda();
        // 标签也上传
        // ...

        logCb("[INFO] All data uploaded to GPU (" +
              std::to_string(totalGpuBytes / 1048576) + " MB)");
    }
#endif

    // 训练循环中：从 GPU tensor 切片 batch，无需再 H2D
    for (int nEpoch = 1; nEpoch <= nEpochs; ++nEpoch) {
        for (int nBatch = 0; nBatch < nBatches; ++nBatch) {
            // GPU: 直接从 GPU 大 tensor 切片（零拷贝 view）
            // CPU: 原有双缓冲逻辑
            Tensor tInput, tLabels;
            if (bCuda) {
                tInput = tensorSlice(tAllTrainData, nStart, nEnd);  // GPU 上的 view
                tLabels = tensorSlice(tAllTrainLabels, nStart, nEnd);
            } else {
                // 原有 CPU 双缓冲
            }

            auto tOutput = m_pImpl->pModel->forward(tInput);  // 全 GPU
            auto tLoss = criterion.forward(tOutput, tLabels);  // 全 GPU
            m_pImpl->pModel->zeroGrad();
            df::tensorBackward(tLoss);  // 反向传播全 GPU
            optimizer->step();          // 参数更新全 GPU
        }
    }

    // 训练完成后将模型参数迁移回 CPU（用于序列化保存）
    if (bCuda) {
        for (auto* pParam : vecParams) {
            *pParam = pParam->cpu();
        }
    }
}
```

- [ ] **Step 2: 移除 CPUBackend 中的 GPU workspace 逻辑**

Phase 4 完成后，GPU 计算不再经过 CPUBackend::matmul → H2D → kernel → D2H 路径。
而是 CUDABackend::matmul 直接操作 GPU 指针。
CPUBackend 恢复为纯 CPU 后端。

- [ ] **Step 3: 编译验证**

---

### Task 4.2: GPU 内存管理优化

**Files:**
- Modify: `src/engine/df.engine.tensor_storage.ixx`
- Modify: `src/cuda/cuda_kernels.cu`

- [ ] **Step 1: TensorStorage 使用 GPU 内存池**

GPU 分配改为通过 `dfCudaMalloc`（已实现内存池）。

- [ ] **Step 2: 添加 GPU 内存统计**

```cpp
// 在 cuda_kernels 中添加
size_t dfCudaGetPoolUsedBytes();   // 已用
size_t dfCudaGetPoolTotalBytes();  // 池总大小
```

在训练日志中输出 GPU 内存使用量。

- [ ] **Step 3: 编译验证**

---

## Phase 5: 验证 + 性能调优

### Task 5.1: 端到端测试

- [ ] **Step 1: CPU 回归测试** — 不开 CUDA 编译，确认所有现有功能正常
- [ ] **Step 2: GPU 单元测试** — Tensor::cuda()/cpu() 往返，数据一致性
- [ ] **Step 3: GPU 训练测试** — 100 张图 + ResNet18 + 10 epochs + CUDA
- [ ] **Step 4: 对比 CPU vs GPU 训练速度和精度**
- [ ] **Step 5: 监控 GPU 利用率（nvidia-smi）确认 >80%**

### Task 5.2: 性能调优

- [ ] **Step 1: 使用 CUDA events 精确计时每个 op**
- [ ] **Step 2: 识别剩余的 H2D/D2H 传输瓶颈**
- [ ] **Step 3: 对大 batch 使用 cudaMemcpyAsync + 双流重叠**
- [ ] **Step 4: 确认 GPU 显存占满到 90%+**

---

## 预期性能目标

| 指标 | 当前 | Phase 4 后 |
|------|------|-----------|
| GPU 利用率 | 18% | **85-95%** |
| GPU 显存使用 | 1.6 GB | **12-14 GB** (RTX 3080 Ti 16GB) |
| CPU 利用率（CUDA 模式） | 100% | **10-20%**（仅数据预处理） |
| H2D 传输次数/epoch | N×batches | **1**（训练开始时一次性上传） |
| 训练吞吐量 | ~50 img/s | **500-2000 img/s** |

---

## 文件变更清单

| 文件 | 操作 | Phase |
|------|------|-------|
| `src/engine/df.engine.tensor_storage.ixx` | 修改（CUDA 分配/释放/拷贝） | 1 |
| `src/engine/df.engine.tensor.ixx` | 修改（.to()/.cuda()/.cpu()、工厂方法 device 参数） | 1 |
| `src/hal/df.hal.cuda_backend.ixx` | **新建**（镜像 CPUBackend 全接口） | 2 |
| `src/cuda/cuda_kernels.cu` | 修改（新增 ~30 个 kernel） | 2 |
| `src/cuda/cuda_kernels.cuh` | 修改（新增声明） | 2 |
| `src/engine/df.engine.tensor_ops.ixx` | 修改（设备调度分支） | 3 |
| `src/engine/df.engine.optimizer.ixx` | 修改（GPU Adam/SGD step） | 3 |
| `src/engine/bridge/EngineBridge.cpp` | 修改（GPU-resident 训练循环） | 4 |
| `src/hal/df.hal.cpu_backend.ixx` | 修改（移除 GPU workspace） | 4 |
| `CMakeLists.txt` | 修改（添加 cuda_backend.ixx） | 2 |
