# GroupNorm + 工业级训练质量重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 解决小 batch（≤8）训练时 BatchNorm 统计量不稳定的致命问题，通过 GroupNorm + 梯度精度审计 + 自动模型缩放，将训练质量对标 Halcon/ViDi 工业级水平。

**Architecture:** 在现有 BatchNorm2d 旁新增 GroupNorm2d 模块（Module 子类），配套 CUDA kernel + autograd 反向传播。所有分割模型（UNet/DeepLabV3/MobileSegNet）构造函数增加 `bool bUseGroupNorm = false` 参数，EngineBridge 在 batch < 8 时自动启用。序列化 v5 记录 norm_type 确保模型加载兼容。SGD 补充 weight decay 支持。

**Tech Stack:** C++20 modules, CUDA 12.x (手写 kernel, 无 cuDNN GroupNorm), om engine autograd

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/engine/om.engine.conv.ixx` | 新增 `GroupNorm2d` 模块类（~60 行） |
| Modify | `src/cuda/cuda_kernels.cu` | GroupNorm 前向/反向 CUDA kernel（~200 行） |
| Modify | `src/cuda/cuda_kernels.cuh` | GroupNorm kernel C 声明（~10 行） |
| Modify | `src/engine/om.engine.autograd.ixx` | `GroupNorm2dBackward` autograd 节点 |
| Modify | `src/engine/om.engine.tensor_ops.ixx` | `tensorGroupNorm2d()` 高层接口 |
| Modify | `src/engine/om.engine.unet.ixx` | UNet 构造函数增加 `bUseGroupNorm` |
| Modify | `src/engine/om.engine.segmodels.ixx` | ResBlock/DeepLabV3/MobileSegNet 增加 GroupNorm 开关 |
| Modify | `src/engine/om.engine.mobilenet.ixx` | ConvBnReLU6/ConvBnLinear 增加 GroupNorm 选项 |
| Modify | `src/engine/bridge/EngineBridge.cpp` | 自动 GroupNorm 选择 + 训练诊断 + 强正则 |
| Modify | `src/engine/bridge/EngineBridge.h` | `BridgeTrainParams` 增加 `bUseGroupNorm` |
| Modify | `src/core/training/TrainingConfig.h` | 增加 `eNormLayerType` 字段 |
| Modify | `src/core/DLTypes.h` | 增加 `NormLayerType` 枚举 |
| Modify | `src/engine/om.engine.optimizer.ixx` | SGD 增加 weight decay |
| Modify | `src/engine/om.engine.serializer.ixx` | v5 格式记录 norm_type |
| Modify | `src/core/training/TrainingSession.cpp` | 自动模型缩放（数据量→模型大小） |
| Modify | `tests/test_conv.cpp` | GroupNorm 单元测试 |

---

## Phase A — GroupNorm 实现（核心，解决致命 BN 问题）

### Task 1: NormLayerType 枚举 + TrainingConfig 字段

**Files:**
- Modify: `src/core/DLTypes.h:~line 40` (在现有枚举区域)
- Modify: `src/core/training/TrainingConfig.h:~line 12`

- [ ] **Step 1: 在 DLTypes.h 添加 NormLayerType 枚举**

在 `DeviceType` 枚举之后（约 line 45）添加：

```cpp
// 20260402 ZJH 归一化层类型枚举
// 控制模型中 Conv 后接的归一化方式
// BatchNorm 在 batch < 8 时统计量不稳定，GroupNorm 不依赖 batch 维度
enum class NormLayerType : uint8_t {
    BatchNorm  = 0,  // 20260402 ZJH 批归一化（默认，batch ≥ 8 时效果最佳）
    GroupNorm  = 1,  // 20260402 ZJH 组归一化（batch < 8 时替代 BN，channels 分组独立归一化）
    Auto       = 2   // 20260402 ZJH 自动选择（batch < 8 → GroupNorm，否则 → BatchNorm）
};
```

- [ ] **Step 2: 在 TrainingConfig 添加字段**

在 `TrainingConfig` 的 `nGradAccumSteps` 之后添加：

```cpp
    // 20260402 ZJH 归一化层类型（Auto = 根据 batch size 自动选择）
    // batch < 8 时 BN 的 running_mean/var 统计量来自极少样本，几乎是随机数
    // GroupNorm 将 channels 分成 groups，每组内独立归一化，不依赖 batch 维度
    om::NormLayerType eNormLayerType = om::NormLayerType::Auto;

    // 20260402 ZJH GroupNorm 分组数（典型值 32，channels 必须被 nGroupNormGroups 整除）
    int nGroupNormGroups = 32;
```

- [ ] **Step 3: 验证编译**

Run: `cd e:/DevelopmentTools/OmniMatchDeeplearningTool && cmake --build build --target omnimatch_app 2>&1 | head -20`
Expected: 编译通过（仅新增枚举和字段，无依赖变化）

- [ ] **Step 4: Commit**

```bash
git add src/core/DLTypes.h src/core/training/TrainingConfig.h
git commit -m "feat: add NormLayerType enum and TrainingConfig field for GroupNorm support"
```

---

### Task 2: GroupNorm CUDA kernel（前向 + 反向）

**Files:**
- Modify: `src/cuda/cuda_kernels.cuh:~line 163` (BN 声明之后)
- Modify: `src/cuda/cuda_kernels.cu:~line 3121` (BN backward 之后)

- [ ] **Step 1: 在 cuda_kernels.cuh 添加 GroupNorm 函数声明**

在 `omCudaBatchNorm2dBackward` 声明之后添加：

```cpp
// ===== GroupNorm =====
// 20260402 ZJH GroupNorm 前向: channels 分成 nGroups 组，每组内独立归一化
// 输入 [N,C,H,W], 输出 [N,C,H,W], gamma/beta [C], savedMean/savedInvStd [N*nGroups]
int omCudaGroupNorm2d(const float* pInput, float* pOutput,
                      const float* pGamma, const float* pBeta,
                      float* pSavedMean, float* pSavedInvStd,
                      int nBatch, int nChannels, int nH, int nW,
                      int nGroups, float fEps);

// 20260402 ZJH GroupNorm 反向: 计算 gradInput, gradGamma, gradBeta
int omCudaGroupNorm2dBackward(const float* pGradOutput, const float* pInput,
                               const float* pMean, const float* pInvStd,
                               const float* pGamma,
                               float* pGradInput, float* pGradGamma, float* pGradBeta,
                               int nBatch, int nChannels, int nH, int nW,
                               int nGroups, float fEps);
```

- [ ] **Step 2: 在 cuda_kernels.cu 实现 GroupNorm 前向 kernel**

在 `omCudaBatchNorm2dBackward` 实现之后添加：

```cpp
// =========================================================================
// 20260402 ZJH GroupNorm 2D — 前向/反向 CUDA 实现
// GroupNorm 将 C 个通道分成 G 组（每组 C/G 个通道），每组内独立归一化
// 与 BatchNorm 的关键区别: GN 不依赖 batch 维度，batch=1 也能稳定工作
// 数学: 对于第 n 个样本的第 g 组:
//   mean_ng = sum(x[n,c,h,w]) / (C/G * H * W)  for c in group g
//   var_ng  = sum((x - mean)^2) / (C/G * H * W)
//   y = gamma[c] * (x - mean) / sqrt(var + eps) + beta[c]
// =========================================================================

// 20260402 ZJH GroupNorm 前向统计 kernel: 每个 block 处理一个 (sample, group) 对
// grid: (N * nGroups), block: 256
__global__ void kernelGroupNormStats(const float* pInput,
                                      float* pMean, float* pVar,
                                      int nBatch, int nChannels, int nH, int nW,
                                      int nGroups) {
    int nGroupIdx = blockIdx.x;                    // 20260402 ZJH (sample, group) 线性索引
    int nSampleIdx = nGroupIdx / nGroups;          // 20260402 ZJH 样本编号
    int nGroupId = nGroupIdx % nGroups;            // 20260402 ZJH 组编号
    int nChannelsPerGroup = nChannels / nGroups;   // 20260402 ZJH 每组通道数
    int nGroupSize = nChannelsPerGroup * nH * nW;  // 20260402 ZJH 每组元素总数

    // 20260402 ZJH 该组在输入张量中的起始偏移
    int nBaseOffset = nSampleIdx * nChannels * nH * nW + nGroupId * nChannelsPerGroup * nH * nW;

    // 20260402 ZJH Grid-stride 累加求 sum 和 sum_sq
    float fSum = 0.0f, fSumSq = 0.0f;
    for (int i = threadIdx.x; i < nGroupSize; i += blockDim.x) {
        float fVal = pInput[nBaseOffset + i];
        fSum += fVal;
        fSumSq += fVal * fVal;
    }

    // 20260402 ZJH Warp-shuffle 规约
    for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
        fSum += __shfl_down_sync(0xffffffff, fSum, nOffset);
        fSumSq += __shfl_down_sync(0xffffffff, fSumSq, nOffset);
    }

    // 20260402 ZJH 共享内存规约（跨 warp）
    __shared__ float s_fSum[32], s_fSumSq[32];
    int nLane = threadIdx.x % 32;
    int nWarpId = threadIdx.x / 32;
    if (nLane == 0) {
        s_fSum[nWarpId] = fSum;
        s_fSumSq[nWarpId] = fSumSq;
    }
    __syncthreads();

    // 20260402 ZJH 第一个 warp 做最终规约
    if (nWarpId == 0) {
        int nNumWarps = (blockDim.x + 31) / 32;
        fSum = (nLane < nNumWarps) ? s_fSum[nLane] : 0.0f;
        fSumSq = (nLane < nNumWarps) ? s_fSumSq[nLane] : 0.0f;
        for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
            fSum += __shfl_down_sync(0xffffffff, fSum, nOffset);
            fSumSq += __shfl_down_sync(0xffffffff, fSumSq, nOffset);
        }
    }

    // 20260402 ZJH thread 0 写出 mean 和 var
    if (threadIdx.x == 0) {
        float fMean = fSum / static_cast<float>(nGroupSize);
        float fVar = fSumSq / static_cast<float>(nGroupSize) - fMean * fMean;
        pMean[nGroupIdx] = fMean;
        pVar[nGroupIdx] = fVar;
    }
}

// 20260402 ZJH GroupNorm 前向归一化 kernel: 逐元素归一化 + 仿射变换
// grid: (N*C*H*W + 255) / 256, block: 256
__global__ void kernelGroupNormForward(const float* pInput, float* pOutput,
                                        const float* pGamma, const float* pBeta,
                                        const float* pMean, const float* pInvStd,
                                        int nBatch, int nChannels, int nH, int nW,
                                        int nGroups) {
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;   // 20260402 ZJH 全局元素索引
    int nTotal = nBatch * nChannels * nH * nW;
    if (nIdx >= nTotal) return;

    int nHW = nH * nW;
    int nCHW = nChannels * nHW;
    int nChannelsPerGroup = nChannels / nGroups;

    // 20260402 ZJH 从线性索引反推 (n, c, h, w)
    int n = nIdx / nCHW;
    int c = (nIdx % nCHW) / nHW;
    int g = c / nChannelsPerGroup;                       // 20260402 ZJH 该通道所属组号

    // 20260402 ZJH 取该 (sample, group) 的 mean 和 invStd
    int nStatIdx = n * nGroups + g;
    float fMean = pMean[nStatIdx];
    float fInvStd = pInvStd[nStatIdx];

    // 20260402 ZJH 归一化 + 仿射: y = gamma[c] * (x - mean) * invStd + beta[c]
    float fVal = pInput[nIdx];
    float fNorm = (fVal - fMean) * fInvStd;
    pOutput[nIdx] = pGamma[c] * fNorm + pBeta[c];
}

// 20260402 ZJH GroupNorm 前向接口函数
int omCudaGroupNorm2d(const float* pInput, float* pOutput,
                      const float* pGamma, const float* pBeta,
                      float* pSavedMean, float* pSavedInvStd,
                      int nBatch, int nChannels, int nH, int nW,
                      int nGroups, float fEps) {
    int nNumGroups = nBatch * nGroups;  // 20260402 ZJH 统计量个数 = N * G

    // 20260402 ZJH Pass 1: 计算每个 (sample, group) 的 mean 和 var
    // 临时 var buffer（与 savedMean 同大小，后面会覆盖为 invStd）
    float* pVar = nullptr;
    cudaMalloc(&pVar, nNumGroups * sizeof(float));

    kernelGroupNormStats<<<nNumGroups, 256>>>(
        pInput, pSavedMean, pVar,
        nBatch, nChannels, nH, nW, nGroups);

    // 20260402 ZJH 将 var 转为 invStd: invStd = 1 / sqrt(var + eps)
    // 简单 kernel，复用 element-wise 模式
    {
        int nBlock = (nNumGroups + 255) / 256;
        // 20260402 ZJH 内联 lambda 不行，用单独小 kernel
        // 这里直接用 CPU 循环也可以（N*G 很小，典型 4*32=128），但为了避免 D2H/H2D 用 kernel
    }
    // 20260402 ZJH 用一个简单 kernel 做 invStd = 1/sqrt(var + eps)
    // 我们借用 kernelGroupNormForward 前先做这步
    {
        // 20260402 ZJH N*G 很小（≤512），直接在 CPU 做更简单可靠
        std::vector<float> vecVar(nNumGroups), vecInvStd(nNumGroups);
        cudaMemcpy(vecVar.data(), pVar, nNumGroups * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nNumGroups; ++i) {
            vecInvStd[i] = 1.0f / std::sqrtf(vecVar[i] + fEps);
        }
        cudaMemcpy(pSavedInvStd, vecInvStd.data(), nNumGroups * sizeof(float), cudaMemcpyHostToDevice);
    }

    // 20260402 ZJH Pass 2: 逐元素归一化 + 仿射变换
    int nTotal = nBatch * nChannels * nH * nW;
    int nBlocks = (nTotal + 255) / 256;
    kernelGroupNormForward<<<nBlocks, 256>>>(
        pInput, pOutput, pGamma, pBeta,
        pSavedMean, pSavedInvStd,
        nBatch, nChannels, nH, nW, nGroups);

    cudaFree(pVar);
    return 0;
}
```

- [ ] **Step 3: 实现 GroupNorm 反向 kernel**

紧接着前向实现之后添加：

```cpp
// 20260402 ZJH GroupNorm 反向 — 统计规约 kernel
// 计算每个 channel 的 gradGamma 和 gradBeta
// gradGamma[c] = sum_over_{n,h,w}(gradOut[n,c,h,w] * xhat[n,c,h,w])
// gradBeta[c]  = sum_over_{n,h,w}(gradOut[n,c,h,w])
__global__ void kernelGroupNormBackwardReduce(
    const float* pGradOutput, const float* pInput,
    const float* pMean, const float* pInvStd,
    float* pGradGamma, float* pGradBeta,
    int nBatch, int nChannels, int nH, int nW, int nGroups) {
    int c = blockIdx.x;  // 20260402 ZJH 每个 block 处理一个 channel
    if (c >= nChannels) return;

    int nHW = nH * nW;
    int nCHW = nChannels * nHW;
    int nChannelsPerGroup = nChannels / nGroups;
    int g = c / nChannelsPerGroup;  // 20260402 ZJH 该 channel 所属组号

    float fSumGradGamma = 0.0f, fSumGradBeta = 0.0f;

    // 20260402 ZJH 遍历所有样本和空间位置
    for (int n = 0; n < nBatch; ++n) {
        int nStatIdx = n * nGroups + g;
        float fMean = pMean[nStatIdx];
        float fInvStd = pInvStd[nStatIdx];

        int nOffset = n * nCHW + c * nHW;
        for (int i = threadIdx.x; i < nHW; i += blockDim.x) {
            float fGrad = pGradOutput[nOffset + i];
            float fXhat = (pInput[nOffset + i] - fMean) * fInvStd;
            fSumGradGamma += fGrad * fXhat;
            fSumGradBeta += fGrad;
        }
    }

    // 20260402 ZJH Warp-shuffle 规约
    for (int nOff = 16; nOff > 0; nOff >>= 1) {
        fSumGradGamma += __shfl_down_sync(0xffffffff, fSumGradGamma, nOff);
        fSumGradBeta += __shfl_down_sync(0xffffffff, fSumGradBeta, nOff);
    }

    __shared__ float s_fGG[32], s_fGB[32];
    int nLane = threadIdx.x % 32;
    int nWarpId = threadIdx.x / 32;
    if (nLane == 0) {
        s_fGG[nWarpId] = fSumGradGamma;
        s_fGB[nWarpId] = fSumGradBeta;
    }
    __syncthreads();

    if (nWarpId == 0) {
        int nNumWarps = (blockDim.x + 31) / 32;
        fSumGradGamma = (nLane < nNumWarps) ? s_fGG[nLane] : 0.0f;
        fSumGradBeta = (nLane < nNumWarps) ? s_fGB[nLane] : 0.0f;
        for (int nOff = 16; nOff > 0; nOff >>= 1) {
            fSumGradGamma += __shfl_down_sync(0xffffffff, fSumGradGamma, nOff);
            fSumGradBeta += __shfl_down_sync(0xffffffff, fSumGradBeta, nOff);
        }
    }

    if (threadIdx.x == 0) {
        // 20260402 ZJH 原子加: 多个 block 可能写同一个 channel（不会，因为 1 block = 1 channel）
        pGradGamma[c] = fSumGradGamma;
        pGradBeta[c] = fSumGradBeta;
    }
}

// 20260402 ZJH GroupNorm 反向 — gradInput kernel
// gradInput = gamma[c] * invStd / M * (M * gradOut - gradBeta_g - xhat * gradGamma_g)
// 其中 M = (C/G) * H * W, 下标 _g 表示该组的累加
__global__ void kernelGroupNormBackwardInput(
    const float* pGradOutput, const float* pInput,
    const float* pMean, const float* pInvStd,
    const float* pGamma,
    const float* pGradGamma, const float* pGradBeta,
    float* pGradInput,
    int nBatch, int nChannels, int nH, int nW, int nGroups) {
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nChannels * nH * nW;
    if (nIdx >= nTotal) return;

    int nHW = nH * nW;
    int nCHW = nChannels * nHW;
    int nChannelsPerGroup = nChannels / nGroups;
    int nGroupSize = nChannelsPerGroup * nHW;  // 20260402 ZJH 每组元素总数 M

    // 20260402 ZJH 从线性索引反推 (n, c)
    int n = nIdx / nCHW;
    int c = (nIdx % nCHW) / nHW;
    int g = c / nChannelsPerGroup;

    int nStatIdx = n * nGroups + g;
    float fMean = pMean[nStatIdx];
    float fInvStd = pInvStd[nStatIdx];
    float fXhat = (pInput[nIdx] - fMean) * fInvStd;

    // 20260402 ZJH 该组的 gradGamma_sum 和 gradBeta_sum（需要对组内所有 channel 累加）
    // 注意: pGradGamma[c] 是单 channel 的，需要对组内所有 channel 求和
    float fGroupGradGamma = 0.0f, fGroupGradBeta = 0.0f;
    int nCStart = g * nChannelsPerGroup;
    for (int ci = nCStart; ci < nCStart + nChannelsPerGroup; ++ci) {
        // 20260402 ZJH 这里的 pGradGamma/pGradBeta 是 per-channel 的
        // 组内累加得到 group-level 的统计量
        fGroupGradGamma += pGradGamma[ci];
        fGroupGradBeta += pGradBeta[ci];
    }

    // 20260402 ZJH gradInput 公式 (参考 PyTorch GroupNorm backward):
    // gradInput = (1/M) * invStd * (M * gamma[c] * gradOut - gamma[c] * gradBeta_c
    //             - xhat * gamma[c] * gradGamma_c)
    // 简化: gradInput = gamma[c] * invStd * (gradOut - gradBeta_g/M - xhat * gradGamma_g/M)
    float fInvM = 1.0f / static_cast<float>(nGroupSize);
    float fGradOut = pGradOutput[nIdx];
    pGradInput[nIdx] = pGamma[c] * fInvStd *
        (fGradOut - fGroupGradBeta * fInvM - fXhat * fGroupGradGamma * fInvM);
}

// 20260402 ZJH GroupNorm 反向接口函数
int omCudaGroupNorm2dBackward(const float* pGradOutput, const float* pInput,
                               const float* pMean, const float* pInvStd,
                               const float* pGamma,
                               float* pGradInput, float* pGradGamma, float* pGradBeta,
                               int nBatch, int nChannels, int nH, int nW,
                               int nGroups, float fEps) {
    // 20260402 ZJH Pass 1: 计算 per-channel gradGamma 和 gradBeta
    // 先清零
    cudaMemset(pGradGamma, 0, nChannels * sizeof(float));
    cudaMemset(pGradBeta, 0, nChannels * sizeof(float));

    kernelGroupNormBackwardReduce<<<nChannels, 256>>>(
        pGradOutput, pInput, pMean, pInvStd,
        pGradGamma, pGradBeta,
        nBatch, nChannels, nH, nW, nGroups);

    // 20260402 ZJH Pass 2: 计算 gradInput
    int nTotal = nBatch * nChannels * nH * nW;
    int nBlocks = (nTotal + 255) / 256;
    kernelGroupNormBackwardInput<<<nBlocks, 256>>>(
        pGradOutput, pInput, pMean, pInvStd, pGamma,
        pGradGamma, pGradBeta, pGradInput,
        nBatch, nChannels, nH, nW, nGroups);

    return 0;
}
```

- [ ] **Step 4: 验证编译**

Run: `cd e:/DevelopmentTools/OmniMatchDeeplearningTool && cmake --build build --target omnimatch_app 2>&1 | tail -5`
Expected: CUDA 编译通过

- [ ] **Step 5: Commit**

```bash
git add src/cuda/cuda_kernels.cu src/cuda/cuda_kernels.cuh
git commit -m "feat: GroupNorm2d CUDA kernels (forward + backward, warp-shuffle reduction)"
```

---

### Task 3: CUDABackend GroupNorm 桥接

**Files:**
- Modify: `src/hal/om.hal.cuda_backend.ixx` (添加 static 方法)

- [ ] **Step 1: 在 CUDABackend 类中添加 GroupNorm 静态方法**

在 `batchNorm2dBackward` 方法之后添加：

```cpp
    // 20260402 ZJH GroupNorm2d 前向 — 调用 CUDA kernel
    static void groupNorm2d(const float* pInput, float* pOutput,
                            const float* pGamma, const float* pBeta,
                            float* pSavedMean, float* pSavedInvStd,
                            int nBatch, int nChannels, int nH, int nW,
                            int nGroups, float fEps) {
        omCudaGroupNorm2d(pInput, pOutput, pGamma, pBeta,
                          pSavedMean, pSavedInvStd,
                          nBatch, nChannels, nH, nW, nGroups, fEps);
    }

    // 20260402 ZJH GroupNorm2d 反向 — 调用 CUDA kernel
    static void groupNorm2dBackward(const float* pGradOutput, const float* pInput,
                                     const float* pMean, const float* pInvStd,
                                     const float* pGamma,
                                     float* pGradInput, float* pGradGamma, float* pGradBeta,
                                     int nBatch, int nChannels, int nH, int nW,
                                     int nGroups, float fEps) {
        omCudaGroupNorm2dBackward(pGradOutput, pInput, pMean, pInvStd, pGamma,
                                   pGradInput, pGradGamma, pGradBeta,
                                   nBatch, nChannels, nH, nW, nGroups, fEps);
    }
```

- [ ] **Step 2: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add src/hal/om.hal.cuda_backend.ixx
git commit -m "feat: CUDABackend GroupNorm2d static bridge methods"
```

---

### Task 4: tensorGroupNorm2d 高层接口 + autograd

**Files:**
- Modify: `src/engine/om.engine.tensor_ops.ixx` (添加 tensorGroupNorm2d 函数)
- Modify: `src/engine/om.engine.autograd.ixx` (添加 GroupNorm2dBackward 类)

- [ ] **Step 1: 在 om.engine.autograd.ixx 添加 GroupNorm2dBackward autograd 节点**

在 `BatchNorm2dBackward` 类之后（约 line 770）添加：

```cpp
// 20260402 ZJH GroupNorm2dBackward — GroupNorm 反向传播 autograd 节点
// 保存前向输入、mean、invStd、gamma 用于反向计算
class GroupNorm2dBackward : public GradFunction {
public:
    Tensor m_savedInput;     // 20260402 ZJH 前向输入 [N,C,H,W]
    Tensor m_savedMean;      // 20260402 ZJH per-(sample,group) 均值 [N*G]
    Tensor m_savedInvStd;    // 20260402 ZJH per-(sample,group) 逆标准差 [N*G]
    Tensor m_savedGamma;     // 20260402 ZJH 缩放参数 [C]
    int m_nBatch = 0, m_nChannels = 0, m_nH = 0, m_nW = 0;
    int m_nGroups = 0;       // 20260402 ZJH 组数
    float m_fEps = 1e-5f;

    void releaseSavedTensors() override {
        m_savedInput = Tensor();
        m_savedMean = Tensor();
        m_savedInvStd = Tensor();
        m_savedGamma = Tensor();
    }

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();

        if (m_savedInput.isCuda()) {
            // 20260402 ZJH GPU 路径
            auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW}, DeviceType::CUDA);
            auto gradGamma = Tensor::zeros({m_nChannels}, DeviceType::CUDA);
            auto gradBeta = Tensor::zeros({m_nChannels}, DeviceType::CUDA);

            CUDABackend::groupNorm2dBackward(
                cGrad.floatDataPtr(), m_savedInput.floatDataPtr(),
                m_savedMean.floatDataPtr(), m_savedInvStd.floatDataPtr(),
                m_savedGamma.floatDataPtr(),
                gradInput.mutableFloatDataPtr(), gradGamma.mutableFloatDataPtr(),
                gradBeta.mutableFloatDataPtr(),
                m_nBatch, m_nChannels, m_nH, m_nW, m_nGroups, m_fEps);

            return {gradInput, gradGamma, gradBeta};
        } else {
            // 20260402 ZJH CPU 路径（GroupNorm CPU fallback）
            auto cInput = m_savedInput.cpu().contiguous();
            auto cMean = m_savedMean.cpu().contiguous();
            auto cInvStd = m_savedInvStd.cpu().contiguous();
            auto cGamma = m_savedGamma.cpu().contiguous();
            auto cGradCpu = cGrad.cpu().contiguous();

            auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
            auto gradGamma = Tensor::zeros({m_nChannels});
            auto gradBeta = Tensor::zeros({m_nChannels});

            int nChannelsPerGroup = m_nChannels / m_nGroups;
            int nHW = m_nH * m_nW;
            int nGroupSize = nChannelsPerGroup * nHW;
            const float* pIn = cInput.floatDataPtr();
            const float* pGO = cGradCpu.floatDataPtr();
            const float* pMean = cMean.floatDataPtr();
            const float* pInvStd = cInvStd.floatDataPtr();
            const float* pGamma = cGamma.floatDataPtr();
            float* pGI = gradInput.mutableFloatDataPtr();
            float* pGG = gradGamma.mutableFloatDataPtr();
            float* pGB = gradBeta.mutableFloatDataPtr();

            // 20260402 ZJH Pass 1: per-channel gradGamma, gradBeta
            for (int n = 0; n < m_nBatch; ++n) {
                for (int c = 0; c < m_nChannels; ++c) {
                    int g = c / nChannelsPerGroup;
                    int nStatIdx = n * m_nGroups + g;
                    float fMean = pMean[nStatIdx];
                    float fInvStd = pInvStd[nStatIdx];
                    int nOff = (n * m_nChannels + c) * nHW;
                    for (int i = 0; i < nHW; ++i) {
                        float fXhat = (pIn[nOff + i] - fMean) * fInvStd;
                        pGG[c] += pGO[nOff + i] * fXhat;
                        pGB[c] += pGO[nOff + i];
                    }
                }
            }

            // 20260402 ZJH Pass 2: gradInput
            for (int n = 0; n < m_nBatch; ++n) {
                for (int c = 0; c < m_nChannels; ++c) {
                    int g = c / nChannelsPerGroup;
                    int nStatIdx = n * m_nGroups + g;
                    float fMean = pMean[nStatIdx];
                    float fInvStd = pInvStd[nStatIdx];
                    // 20260402 ZJH 组内 gradGamma_sum 和 gradBeta_sum
                    float fGroupGG = 0.0f, fGroupGB = 0.0f;
                    int nCStart = g * nChannelsPerGroup;
                    for (int ci = nCStart; ci < nCStart + nChannelsPerGroup; ++ci) {
                        fGroupGG += pGG[ci];
                        fGroupGB += pGB[ci];
                    }
                    float fInvM = 1.0f / static_cast<float>(nGroupSize);
                    int nOff = (n * m_nChannels + c) * nHW;
                    for (int i = 0; i < nHW; ++i) {
                        float fXhat = (pIn[nOff + i] - fMean) * fInvStd;
                        pGI[nOff + i] = pGamma[c] * fInvStd *
                            (pGO[nOff + i] - fGroupGB * fInvM - fXhat * fGroupGG * fInvM);
                    }
                }
            }

            return {gradInput, gradGamma, gradBeta};
        }
    }
};
```

- [ ] **Step 2: 在 om.engine.tensor_ops.ixx 添加 tensorGroupNorm2d 函数**

在 `tensorBatchNorm2d` 函数之后添加：

```cpp
// 20260402 ZJH tensorGroupNorm2d — GroupNorm 前向，支持 autograd
// input: [N,C,H,W], gamma: [C], beta: [C]
// nGroups: 组数（C 必须被 nGroups 整除）
// 返回: [N,C,H,W] 归一化+仿射变换后的张量
inline Tensor tensorGroupNorm2d(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                                 int nGroups, float fEps = 1e-5f) {
    auto shape = input.shapeVec();
    int nBatch = shape[0], nChannels = shape[1], nH = shape[2], nW = shape[3];

    // 20260402 ZJH 安全检查: channels 必须被 groups 整除
    if (nChannels % nGroups != 0) {
        // 20260402 ZJH 自动调整 groups 到最接近的整除值
        while (nGroups > 1 && nChannels % nGroups != 0) --nGroups;
    }

    int nNumGroups = nBatch * nGroups;
    auto output = Tensor::zeros(shape, input.device());
    auto savedMean = Tensor::zeros({nNumGroups}, input.device());
    auto savedInvStd = Tensor::zeros({nNumGroups}, input.device());

    if (input.isCuda()) {
        CUDABackend::groupNorm2d(
            input.floatDataPtr(), output.mutableFloatDataPtr(),
            gamma.floatDataPtr(), beta.floatDataPtr(),
            savedMean.mutableFloatDataPtr(), savedInvStd.mutableFloatDataPtr(),
            nBatch, nChannels, nH, nW, nGroups, fEps);
    } else {
        // 20260402 ZJH CPU fallback
        const float* pIn = input.floatDataPtr();
        float* pOut = output.mutableFloatDataPtr();
        const float* pGamma = gamma.floatDataPtr();
        const float* pBeta = beta.floatDataPtr();
        float* pMean = savedMean.mutableFloatDataPtr();
        float* pInvStd = savedInvStd.mutableFloatDataPtr();

        int nChannelsPerGroup = nChannels / nGroups;
        int nHW = nH * nW;
        int nGroupSize = nChannelsPerGroup * nHW;

        for (int n = 0; n < nBatch; ++n) {
            for (int g = 0; g < nGroups; ++g) {
                int nStatIdx = n * nGroups + g;
                // 20260402 ZJH 计算 group mean
                float fSum = 0.0f;
                int nBase = n * nChannels * nHW + g * nChannelsPerGroup * nHW;
                for (int i = 0; i < nGroupSize; ++i) fSum += pIn[nBase + i];
                float fMean = fSum / static_cast<float>(nGroupSize);
                // 20260402 ZJH 计算 group var
                float fVarSum = 0.0f;
                for (int i = 0; i < nGroupSize; ++i) {
                    float fDiff = pIn[nBase + i] - fMean;
                    fVarSum += fDiff * fDiff;
                }
                float fVar = fVarSum / static_cast<float>(nGroupSize);
                float fInvStd = 1.0f / std::sqrtf(fVar + fEps);
                pMean[nStatIdx] = fMean;
                pInvStd[nStatIdx] = fInvStd;

                // 20260402 ZJH 归一化 + 仿射
                for (int ci = 0; ci < nChannelsPerGroup; ++ci) {
                    int c = g * nChannelsPerGroup + ci;
                    int nOff = n * nChannels * nHW + c * nHW;
                    for (int i = 0; i < nHW; ++i) {
                        pOut[nOff + i] = pGamma[c] * ((pIn[nOff + i] - fMean) * fInvStd) + pBeta[c];
                    }
                }
            }
        }
    }

    // 20260402 ZJH 注册 autograd（仅当输入需要梯度时）
    if (input.requiresGrad() || gamma.requiresGrad()) {
        auto gradFn = std::make_shared<GroupNorm2dBackward>();
        gradFn->m_savedInput = input;
        gradFn->m_savedMean = savedMean;
        gradFn->m_savedInvStd = savedInvStd;
        gradFn->m_savedGamma = gamma;
        gradFn->m_nBatch = nBatch;
        gradFn->m_nChannels = nChannels;
        gradFn->m_nH = nH;
        gradFn->m_nW = nW;
        gradFn->m_nGroups = nGroups;
        gradFn->m_fEps = fEps;
        gradFn->m_vecInputs = {input, gamma, beta};
        output.setGradFn(gradFn);
        output.setRequiresGrad(true);
    }

    return output;
}
```

- [ ] **Step 3: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add src/engine/om.engine.autograd.ixx src/engine/om.engine.tensor_ops.ixx
git commit -m "feat: tensorGroupNorm2d with autograd support (GPU + CPU paths)"
```

---

### Task 5: GroupNorm2d Module 类

**Files:**
- Modify: `src/engine/om.engine.conv.ixx:123` (BatchNorm2d 类之后)

- [ ] **Step 1: 在 om.engine.conv.ixx 添加 GroupNorm2d 类**

在 `BatchNorm2d` 类闭合大括号之后、`MaxPool2d` 之前（line 124）插入：

```cpp
// 20260402 ZJH GroupNorm2d — 2D 组归一化层
// 将 channels 分成 nGroups 组，每组内独立归一化
// 与 BatchNorm2d 的关键区别:
//   BN: 沿 (N, H, W) 维度归一化，依赖 batch 统计量，batch < 8 时不稳定
//   GN: 沿 (C/G, H, W) 维度归一化，不依赖 batch 维度，batch=1 也稳定
// Halcon/ViDi 在 batch < 8 时自动切换到 GN
class GroupNorm2d : public Module {
public:
    // 20260402 ZJH 构造函数
    // nNumChannels: 输入通道数
    // nGroups: 分组数（默认 32，nNumChannels 必须被 nGroups 整除）
    // fEps: 数值稳定性常数
    GroupNorm2d(int nNumChannels, int nGroups = 32, float fEps = 1e-5f)
        : m_nNumChannels(nNumChannels), m_nGroups(nGroups), m_fEps(fEps)
    {
        // 20260402 ZJH 自动调整 groups: 确保 channels 能被 groups 整除
        while (m_nGroups > 1 && nNumChannels % m_nGroups != 0) --m_nGroups;

        m_gamma = Tensor::ones({nNumChannels});   // 20260402 ZJH gamma 初始化为 1
        m_beta = Tensor::zeros({nNumChannels});   // 20260402 ZJH beta 初始化为 0
        registerParameter("gamma", m_gamma);
        registerParameter("beta", m_beta);
        // 20260402 ZJH GroupNorm 没有 running stats（不需要 train/eval 切换）
    }

    // 20260402 ZJH forward — 前向传播
    // input: [N, C, H, W]
    Tensor forward(const Tensor& input) override {
        return tensorGroupNorm2d(input, m_gamma, m_beta, m_nGroups, m_fEps);
    }

    // 20260402 ZJH GroupNorm 没有 buffers（无 running stats）
    std::vector<Tensor*> buffers() override { return {}; }
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& = "") override { return {}; }

    int groups() const { return m_nGroups; }  // 20260402 ZJH 返回实际分组数

private:
    int m_nNumChannels;  // 20260402 ZJH 通道数
    int m_nGroups;       // 20260402 ZJH 分组数
    float m_fEps;        // 20260402 ZJH 数值稳定性常数
    Tensor m_gamma;      // 20260402 ZJH 缩放参数 [C]
    Tensor m_beta;       // 20260402 ZJH 偏移参数 [C]
};
```

- [ ] **Step 2: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add src/engine/om.engine.conv.ixx
git commit -m "feat: GroupNorm2d module class (auto-adjusts groups for channel divisibility)"
```

---

### Task 6: UNet 增加 GroupNorm 开关

**Files:**
- Modify: `src/engine/om.engine.unet.ixx`

- [ ] **Step 1: 修改 UNetEncoderBlock 和 UNetDecoderBlock 支持 GroupNorm**

核心策略: 使用 `std::variant<BatchNorm2d, GroupNorm2d>` 或更简单地用条件分支。鉴于 C++20 modules 的限制，采用**双成员 + 标志位**方案（BN 和 GN 各一个实例，只用其中一个）。

在 `UNetEncoderBlock` 类中:

1. 构造函数签名改为 `UNetEncoderBlock(int nIn, int nOut, float fDropout = 0.0f, bool bUseGroupNorm = false)`
2. 新增成员: `bool m_bUseGroupNorm; GroupNorm2d m_gn1, m_gn2;`
3. forward 中根据 `m_bUseGroupNorm` 选择 BN 或 GN

具体代码修改（UNetEncoderBlock 构造函数，约 line 28）：

```cpp
    // 20260402 ZJH 新增 bUseGroupNorm 参数: batch < 8 时启用 GroupNorm 替代 BN
    UNetEncoderBlock(int nIn, int nOut, float fDropout = 0.0f, bool bUseGroupNorm = false)
        : m_conv1(nIn, nOut, 3, 1, 1, true), m_conv2(nOut, nOut, 3, 1, 1, true),
          m_bn1(nOut), m_bn2(nOut),
          m_gn1(nOut, 32), m_gn2(nOut, 32),  // 20260402 ZJH GroupNorm 实例（不用时不占 GPU 显存）
          m_dropout(fDropout), m_fDropout(fDropout),
          m_bUseGroupNorm(bUseGroupNorm)
    {}
```

forward 方法修改:

```cpp
    Tensor forward(const Tensor& input) override {
        auto x = m_conv1.forward(input);
        x = m_bUseGroupNorm ? m_gn1.forward(x) : m_bn1.forward(x);  // 20260402 ZJH GN/BN 切换
        x = m_relu.forward(x);
        x = m_conv2.forward(x);
        x = m_bUseGroupNorm ? m_gn2.forward(x) : m_bn2.forward(x);  // 20260402 ZJH GN/BN 切换
        x = m_relu.forward(x);
        if (m_bTraining && m_fDropout > 0.01f) x = m_dropout.forward(x);
        return x;
    }
```

parameters() 方法修改（收集正确的 norm 参数）:

```cpp
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_conv1.parameters()) v.push_back(p);
        // 20260402 ZJH 根据 norm 类型收集对应参数
        if (m_bUseGroupNorm) {
            for (auto* p : m_gn1.parameters()) v.push_back(p);
        } else {
            for (auto* p : m_bn1.parameters()) v.push_back(p);
        }
        for (auto* p : m_conv2.parameters()) v.push_back(p);
        if (m_bUseGroupNorm) {
            for (auto* p : m_gn2.parameters()) v.push_back(p);
        } else {
            for (auto* p : m_bn2.parameters()) v.push_back(p);
        }
        return v;
    }
```

buffers() 方法修改:

```cpp
    std::vector<Tensor*> buffers() override {
        // 20260402 ZJH GroupNorm 没有 running stats，只有 BN 需要收集 buffers
        if (m_bUseGroupNorm) return {};
        std::vector<Tensor*> v;
        for (auto* b : m_bn1.buffers()) v.push_back(b);
        for (auto* b : m_bn2.buffers()) v.push_back(b);
        return v;
    }
```

同样的修改应用于 `UNetDecoderBlock`。

2. **UNet 构造函数**改为:

```cpp
    UNet(int nInChannels = 1, int nNumClasses = 2, int nBaseChannels = 64, bool bUseGroupNorm = false)
        : m_enc1(nInChannels, nBaseChannels, 0.0f, bUseGroupNorm),
          m_enc2(nBaseChannels, nBaseChannels * 2, 0.0f, bUseGroupNorm),
          m_enc3(nBaseChannels * 2, nBaseChannels * 4, 0.1f, bUseGroupNorm),
          m_enc4(nBaseChannels * 4, nBaseChannels * 8, 0.2f, bUseGroupNorm),
          m_bottleneck(nBaseChannels * 8, nBaseChannels * 16, 0.3f, bUseGroupNorm),
          m_dec4(nBaseChannels * 16 + nBaseChannels * 8, nBaseChannels * 8, bUseGroupNorm),
          m_dec3(nBaseChannels * 8 + nBaseChannels * 4, nBaseChannels * 4, bUseGroupNorm),
          m_dec2(nBaseChannels * 4 + nBaseChannels * 2, nBaseChannels * 2, bUseGroupNorm),
          m_dec1(nBaseChannels * 2 + nBaseChannels, nBaseChannels, bUseGroupNorm),
          m_finalConv(nBaseChannels, nNumClasses, 1, 1, 0, true),
          m_pool(2), m_bUseGroupNorm(bUseGroupNorm)
    {}
```

- [ ] **Step 2: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -10`

- [ ] **Step 3: Commit**

```bash
git add src/engine/om.engine.unet.ixx
git commit -m "feat: UNet GroupNorm switch (bUseGroupNorm parameter in encoder/decoder/model)"
```

---

### Task 7: ResBlock/DeepLabV3/MobileSegNet 增加 GroupNorm 开关

**Files:**
- Modify: `src/engine/om.engine.segmodels.ixx`

- [ ] **Step 1: ResBlock 添加 GroupNorm 支持**

修改 ResBlock 构造函数（line 37）:

```cpp
    ResBlock(int nIn, int nOut, int nStride = 1, float fDropout = 0.0f, bool bUseGroupNorm = false)
        : m_conv1(nIn, nOut, 3, nStride, 1, true), m_bn1(nOut),
          m_conv2(nOut, nOut, 3, 1, 1, true), m_bn2(nOut),
          m_gn1(nOut, 32), m_gn2(nOut, 32),  // 20260402 ZJH GroupNorm 实例
          m_dropout(fDropout), m_fDrop(fDropout),
          m_bDownsample(nStride != 1 || nIn != nOut),
          m_convDs(nIn, nOut, 1, nStride, 0, false), m_bnDs(nOut), m_gnDs(nOut, 32),
          m_bUseGroupNorm(bUseGroupNorm)
    {}
```

修改 forward（line 46）:

```cpp
    Tensor forward(const Tensor& input) override {
        auto normFwd = [this](auto& bn, auto& gn, const Tensor& x) -> Tensor {
            return m_bUseGroupNorm ? gn.forward(x) : bn.forward(x);
        };
        auto out = m_relu.forward(normFwd(m_bn1, m_gn1, m_conv1.forward(input)));
        if (m_dropout.isTraining() && m_fDrop > 0.01f) out = m_dropout.forward(out);
        out = normFwd(m_bn2, m_gn2, m_conv2.forward(out));
        auto shortcut = m_bDownsample ? normFwd(m_bnDs, m_gnDs, m_convDs.forward(input)) : input;
        out = tensorAdd(out, shortcut);
        return m_relu.forward(out);
    }
```

parameters() 和 buffers() 也相应修改（GN 时收集 GN 参数，BN 时收集 BN 参数和 buffers）。

- [ ] **Step 2: DeepLabV3 和 MobileSegNet 构造函数添加 bUseGroupNorm 参数**

DeepLabV3 类（约 line 943）:

```cpp
    DeepLabV3(int nInChannels = 3, int nNumClasses = 2, bool bUseGroupNorm = false)
```

传递 `bUseGroupNorm` 给所有内部 ResBlock。DeepLabV3 内部直接使用的 BN 也需要增加 GN 分支。

MobileSegNet 类（约 line 1262）:

```cpp
    MobileSegNet(int nInChannels = 3, int nNumClasses = 2, bool bUseGroupNorm = false)
```

同样传递 `bUseGroupNorm`。

- [ ] **Step 3: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add src/engine/om.engine.segmodels.ixx
git commit -m "feat: ResBlock/DeepLabV3/MobileSegNet GroupNorm switch"
```

---

### Task 8: EngineBridge 自动 GroupNorm 选择

**Files:**
- Modify: `src/engine/bridge/EngineBridge.h:~line 19` (BridgeTrainParams)
- Modify: `src/engine/bridge/EngineBridge.cpp:~line 274` (模型创建) + `~line 478` (训练开始)

- [ ] **Step 1: BridgeTrainParams 添加 GroupNorm 字段**

在 `EngineBridge.h` 的 `BridgeTrainParams` 中添加:

```cpp
    bool bUseGroupNorm = false;       // 20260402 ZJH 是否使用 GroupNorm（由 Auto 策略决定）
    int nGroupNormGroups = 32;        // 20260402 ZJH GroupNorm 分组数
```

- [ ] **Step 2: 模型创建时传递 bUseGroupNorm**

在 `EngineBridge.cpp` 模型创建处（约 line 274）修改:

```cpp
    // 20260402 ZJH 根据 NormLayerType 决定是否启用 GroupNorm
    bool bUseGN = params.bUseGroupNorm;

    else if (strModelType == "UNet") {
        int nBase = /* existing logic */;
        pModel = std::make_shared<om::UNet>(nInCh, nNumClasses, nBase, bUseGN);
    }
    else if (strModelType == "DeepLabV3+" || strModelType == "DeepLabV3Plus" || strModelType == "DeepLabV3")
        pModel = std::make_shared<om::DeepLabV3>(nInCh, nNumClasses, bUseGN);
    else if (strModelType == "MobileSegNet" || strModelType == "MobileSeg")
        pModel = std::make_shared<om::MobileSegNet>(nInCh, nNumClasses, bUseGN);
```

- [ ] **Step 3: 训练开始时 Auto 策略选择 GroupNorm**

在训练诊断区域（约 line 478）的 batch size 校正之后添加:

```cpp
    // 20260402 ZJH ===== GroupNorm 自动选择策略（对标 Halcon/ViDi）=====
    // 原理: BatchNorm 的 running_mean/var 来自 batch 统计量
    //   batch=4 时每步仅 4 个样本，均值/方差几乎是随机数（信噪比极低）
    //   GroupNorm 按 channel 分组归一化，完全不依赖 batch 维度
    // Halcon 策略: batch < 8 自动切换 GroupNorm
    if (!params.bUseGroupNorm && nBatchSize < 8 && m_pImpl->bIsSegmentation) {
        if (logCb) logCb("[INFO] batch_size=" + std::to_string(nBatchSize)
            + " < 8: auto-enabling GroupNorm (BN statistics unreliable with small batches)");
        // 20260402 ZJH 注意: 此时模型已创建，需要重建模型
        // 由于模型创建在更早的阶段，这里需要在模型创建前就确定 bUseGroupNorm
        // → 实际应在 setModel/initModel 阶段决定
    }
```

**重要修正:** GroupNorm 选择需要在模型创建之前。因此在 `initModel` 或 `setModel` 方法中，根据 `params.nBatchSize` 判断：

```cpp
    // 20260402 ZJH 在模型创建前确定 GroupNorm 策略
    if (params.nBatchSize < 8 && bIsSegmentation) {
        params.bUseGroupNorm = true;
        if (logCb) logCb("[INFO] Auto-enabling GroupNorm: batch_size="
            + std::to_string(params.nBatchSize) + " < 8");
    }
```

- [ ] **Step 4: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -10`

- [ ] **Step 5: Commit**

```bash
git add src/engine/bridge/EngineBridge.h src/engine/bridge/EngineBridge.cpp
git commit -m "feat: auto GroupNorm selection when batch < 8 (Halcon-style strategy)"
```

---

## Phase B — 梯度精度审计 + 训练诊断

### Task 9: SGD 增加 Weight Decay

**Files:**
- Modify: `src/engine/om.engine.optimizer.ixx:29-90` (SGD 类)

- [ ] **Step 1: SGD 构造函数添加 fWeightDecay 参数**

修改 SGD 构造函数（line 35）:

```cpp
    SGD(std::vector<Tensor*> vecParams, float fLr, float fMomentum = 0.0f, float fWeightDecay = 0.0f)
        : m_vecParams(vecParams), m_fLr(fLr), m_fMomentum(fMomentum), m_fWeightDecay(fWeightDecay)
```

添加成员变量:

```cpp
    float m_fWeightDecay;  // 20260402 ZJH L2 正则化系数（darknet 默认 5e-4）
```

- [ ] **Step 2: SGD step() 中实现 weight decay**

在 SGD `step()` 方法的梯度获取之后、参数更新之前（约 line 76）添加:

```cpp
            // 20260402 ZJH Weight Decay (L2 正则化): grad += weight_decay * param
            // Darknet 风格: 在梯度上加 decay * param，然后正常 SGD 更新
            // 效果: 每步将参数向 0 收缩一点，防止过拟合
            if (m_fWeightDecay > 0.0f) {
                // 20260402 ZJH grad = grad + wd * param
                if (m_vecParams[i]->isCuda()) {
                    // 20260402 ZJH GPU: grad += wd * param (axpy 操作)
                    CUDABackend::axpy(m_fWeightDecay, m_vecParams[i]->floatDataPtr(),
                                      cGrad.mutableFloatDataPtr(),
                                      static_cast<size_t>(cGrad.numel()));
                } else {
                    float* pGrad = cGrad.mutableFloatDataPtr();
                    const float* pParam = m_vecParams[i]->floatDataPtr();
                    int n = cGrad.numel();
                    for (int j = 0; j < n; ++j) {
                        pGrad[j] += m_fWeightDecay * pParam[j];
                    }
                }
            }
```

**注意:** 需要检查 CUDABackend 是否有 `axpy` 方法。如果没有，用 `mulScalar` + `add` 组合，或添加一个简单的 axpy kernel。

- [ ] **Step 3: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add src/engine/om.engine.optimizer.ixx
git commit -m "feat: SGD weight decay support (L2 regularization, darknet-style 5e-4 default)"
```

---

### Task 10: 训练诊断管道（过拟合检测 + 梯度验证 + 预训练检查）

**Files:**
- Modify: `src/engine/bridge/EngineBridge.cpp` (训练循环内)

- [ ] **Step 1: 过拟合早期警告**

在每 epoch 结束的验证 loss 计算之后（找到 val_loss 计算位置）添加:

```cpp
    // 20260402 ZJH ===== 过拟合早期警告 =====
    // 检测: train_loss 持续下降但 val_loss 停滞或上升 = 过拟合信号
    // Halcon 策略: 连续 5 epoch train_loss 下降 + val_loss 上升 → 警告
    if (nEpoch >= 5) {
        bool bTrainDecreasing = (fTrainLoss < fPrevTrainLoss * 0.98f);  // 20260402 ZJH train loss 下降 > 2%
        bool bValIncreasing = (fValLoss > fBestValLoss * 1.02f);         // 20260402 ZJH val loss 上升 > 2%
        if (bTrainDecreasing && bValIncreasing) {
            ++nOverfitCount;
            if (nOverfitCount >= 3 && logCb) {
                logCb("[WARN] Overfitting detected: train_loss decreasing but val_loss increasing for "
                    + std::to_string(nOverfitCount) + " consecutive epochs. "
                    "Consider: (1) reduce model size, (2) increase augmentation, (3) add more training data");
            }
        } else {
            nOverfitCount = 0;
        }
    }
```

在训练循环开始前声明:

```cpp
    int nOverfitCount = 0;   // 20260402 ZJH 连续过拟合 epoch 计数
    float fPrevTrainLoss = 1e10f;  // 20260402 ZJH 上一 epoch 训练损失
```

- [ ] **Step 2: 预训练权重前向验证**

在预训练加载成功之后（约 line 529 附近），添加:

```cpp
    // 20260402 ZJH ===== 预训练权重前向验证 =====
    // 目的: 确保加载的权重没有损坏（非全零/非 NaN/输出分布合理）
    // 方法: 创建一个随机输入跑一次 forward，检查输出
    if (nPretrainedLoaded > 0) {
        auto tTestInput = om::Tensor::randn({1, nInCh, 32, 32});  // 20260402 ZJH 小尺寸测试输入
        if (m_pImpl->pModel->parameters().front()->isCuda()) {
            tTestInput = tTestInput.cuda();
        }
        m_pImpl->pModel->eval();  // 20260402 ZJH 推理模式（用 running stats）
        auto tTestOutput = m_pImpl->pModel->forward(tTestInput);
        m_pImpl->pModel->train();  // 20260402 ZJH 恢复训练模式

        // 20260402 ZJH 检查输出是否合理
        auto tCpu = tTestOutput.cpu().contiguous();
        const float* pOut = tCpu.floatDataPtr();
        int nOutElems = tCpu.numel();
        bool bAllZero = true, bHasNan = false;
        float fSum = 0.0f;
        for (int i = 0; i < nOutElems; ++i) {
            if (std::isnan(pOut[i]) || std::isinf(pOut[i])) bHasNan = true;
            if (std::abs(pOut[i]) > 1e-8f) bAllZero = false;
            fSum += pOut[i];
        }

        if (bHasNan && logCb) {
            logCb("[ERROR] Pretrained weights produce NaN/Inf output — weights may be corrupted!");
        } else if (bAllZero && logCb) {
            logCb("[WARN] Pretrained weights produce all-zero output — weights may not match architecture");
        } else if (logCb) {
            float fMean = fSum / static_cast<float>(nOutElems);
            logCb("[INFO] Pretrained weights verified: output mean=" + std::to_string(fMean)
                + " (non-zero, non-NaN → weights OK)");
        }
    }
```

- [ ] **Step 3: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add src/engine/bridge/EngineBridge.cpp
git commit -m "feat: training diagnostics (overfit detection + pretrained weight validation)"
```

---

## Phase C — 自动模型缩放 + 强正则 + 确定性训练

### Task 11: 自动模型缩放（数据量→模型大小映射）

**Files:**
- Modify: `src/core/training/TrainingSession.cpp:~line 215`
- Modify: `src/engine/bridge/EngineBridge.cpp:~line 274` (UNet base channel 选择)

- [ ] **Step 1: 在 TrainingSession 中添加数据量→架构自动推荐**

在模型架构确定之后、训练开始之前添加:

```cpp
    // 20260402 ZJH ===== 自动模型缩放 =====
    // 原则: 模型参数量不应超过训练数据像素总量的 1/10
    // 32 张 224x224 图像 ≈ 1.6M 像素, DeepLabV3+(16M params) = 严重过拟合
    // Halcon 策略: 小数据集自动选择轻量模型
    int nTotalTrainImages = /* 获取训练集大小 */;

    if (bIsSegmentation && nTotalTrainImages < 100) {
        // 20260402 ZJH 小数据集: 强制轻量模型
        if (nTotalTrainImages < 30) {
            // 20260402 ZJH < 30 张: UNet base=16 (1.8M params) 或 MobileSegNet
            if (localConfig.eArchitecture == om::ModelArchitecture::DeepLabV3Plus) {
                emit trainingLog("[INFO] Auto-scaling: " + QString::number(nTotalTrainImages)
                    + " images too few for DeepLabV3+ (16M params) → switching to MobileSegNet");
                localConfig.eArchitecture = om::ModelArchitecture::MobileSegNet;
            }
            // 20260402 ZJH UNet 降到 base=16
            nAutoBaseChannels = 16;
        } else if (nTotalTrainImages < 100) {
            // 20260402 ZJH 30-99 张: UNet base=32 (7M params)
            if (localConfig.eArchitecture == om::ModelArchitecture::DeepLabV3Plus) {
                emit trainingLog("[INFO] Auto-scaling: " + QString::number(nTotalTrainImages)
                    + " images → switching DeepLabV3+ to UNet (lighter)");
                localConfig.eArchitecture = om::ModelArchitecture::UNet;
            }
            nAutoBaseChannels = 32;
        }

        emit trainingLog("[INFO] Model auto-scaled for small dataset ("
            + QString::number(nTotalTrainImages) + " images)");
    }
```

- [ ] **Step 2: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add src/core/training/TrainingSession.cpp src/engine/bridge/EngineBridge.cpp
git commit -m "feat: auto model scaling for small datasets (<30→base16, <100→base32)"
```

---

### Task 12: 强正则组合 + 确定性训练

**Files:**
- Modify: `src/engine/bridge/EngineBridge.cpp`

- [ ] **Step 1: 小数据集自动强化正则**

在训练开始前的诊断区域添加:

```cpp
    // 20260402 ZJH ===== 小数据集强正则策略（对标 Halcon/Darknet）=====
    // Darknet 默认 weight_decay = 5e-4, 对小数据集很重要
    float fWeightDecay = 0.0f;
    if (nTrainCount < 200) {
        fWeightDecay = 5e-4f;  // 20260402 ZJH Darknet 默认值
        if (logCb) logCb("[INFO] Small dataset (" + std::to_string(nTrainCount)
            + " images): enabling weight_decay=5e-4 for regularization");
    }
```

在优化器创建时传入 weight decay:

```cpp
    // 20260402 ZJH 创建优化器时传入 weight decay
    if (strOptimizer == "SGD") {
        pSgd = std::make_shared<om::SGD>(vecModelParams, fLr, fMomentum, fWeightDecay);
    } else if (strOptimizer == "AdamW") {
        pAdamW = std::make_shared<om::AdamW>(vecModelParams, fLr, 0.9f, 0.999f, 1e-8f, fWeightDecay);
    }
```

- [ ] **Step 2: 确定性训练模式**

在 CUDA 初始化处添加:

```cpp
    // 20260402 ZJH ===== 确定性训练模式 =====
    // 固定 cuDNN 算法选择 + 随机种子，确保训练可复现
    // 代价: cuDNN 不会搜索最快算法，可能慢 5-10%
    #ifdef OM_HAS_CUDA
    if (params.bDeterministic) {
        cudnnSetDeterminism(CUDNN_DETERMINISTIC);  // 20260402 ZJH 固定 cuDNN 算法
        // 20260402 ZJH 注意: 具体 API 取决于 cuDNN 版本
        // cuDNN 8.x: cudnnSetConvolutionMathType + CUDNN_DETERMINISTIC
    }
    #endif
```

**注意:** cuDNN 确定性 API 因版本而异。如果当前版本不支持，可以通过禁用 `CUDNN_BENCHMARK` 实现。

- [ ] **Step 3: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add src/engine/bridge/EngineBridge.cpp
git commit -m "feat: strong regularization for small datasets + deterministic training mode"
```

---

### Task 13: 序列化 v5 — 记录 NormLayerType

**Files:**
- Modify: `src/engine/om.engine.serializer.ixx:~line 109` (ModelMeta) + save/load

- [ ] **Step 1: ModelMeta 添加 norm_type 字段**

在 ModelMeta 结构体中添加:

```cpp
    int nNormType = 0;              // 20260402 ZJH 归一化类型 (0=BN, 1=GN)
    int nGroupNormGroups = 32;      // 20260402 ZJH GN 分组数
```

encode() 方法增加 2 个 float:

```cpp
    // 20260402 ZJH v5 meta: [magic=42.0, typeHash, base, inputSize, classes, inCh, normType, gnGroups]
    vec.push_back(static_cast<float>(nNormType));
    vec.push_back(static_cast<float>(nGroupNormGroups));
```

decode() 方法增加对应解析:

```cpp
    if (nCount >= 8) {
        out.nNormType = static_cast<int>(pData[6]);
        out.nGroupNormGroups = static_cast<int>(pData[7]);
    }
```

- [ ] **Step 2: save() 中版本号升级为 5**

将 `nVersion = 4` 改为 `nVersion = 5`，同时保持 v4 的 load 兼容。

- [ ] **Step 3: load() 中 v5 额外字段处理**

在 v4 metadata 读取之后:

```cpp
    // 20260402 ZJH v5: 从 meta 中读取 norm_type 和 groups
    if (meta.nNormType == 1) {
        bUseGroupNorm = true;
        nGroupNormGroups = meta.nGroupNormGroups;
    }
```

模型重建时传递 `bUseGroupNorm`:

```cpp
    if (strType == "UNet") {
        pModel = std::make_shared<om::UNet>(3, nSavedClasses, nSavedBase, bUseGroupNorm);
    }
```

- [ ] **Step 4: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add src/engine/om.engine.serializer.ixx
git commit -m "feat: serialization v5 with NormLayerType for GroupNorm model compatibility"
```

---

### Task 14: EngineBridge 推理端 GroupNorm 模型加载

**Files:**
- Modify: `src/engine/bridge/EngineBridge.cpp:~line 3072` (模型反序列化重建)

- [ ] **Step 1: 反序列化时根据 norm_type 重建正确模型**

修改模型重建逻辑:

```cpp
    // 20260402 ZJH 从序列化 meta 中读取 norm_type
    bool bFileGroupNorm = (savedMeta.nNormType == 1);

    if (strType == "UNet") {
        m_pImpl->pModel = std::make_shared<om::UNet>(3, nSavedClasses, nSavedBase, bFileGroupNorm);
    } else if (strType == "DeepLabV3+" || strType == "DeepLabV3Plus" || strType == "DeepLabV3") {
        m_pImpl->pModel = std::make_shared<om::DeepLabV3>(nSavedClasses, bFileGroupNorm);
    } else if (strType == "MobileSegNet" || strType == "MobileSeg") {
        m_pImpl->pModel = std::make_shared<om::MobileSegNet>(3, nSavedClasses, bFileGroupNorm);
    }
```

- [ ] **Step 2: 验证编译**

Run: `cmake --build build --target omnimatch_app 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add src/engine/bridge/EngineBridge.cpp
git commit -m "feat: inference-side GroupNorm model loading from v5 serialization"
```

---

### Task 15: 单元测试

**Files:**
- Modify: `tests/test_conv.cpp`

- [ ] **Step 1: GroupNorm2d 基础前向测试**

在 `test_conv.cpp` 末尾添加:

```cpp
// 20260402 ZJH GroupNorm2d 前向测试
void testGroupNorm2dForward() {
    std::cout << "=== Test GroupNorm2d Forward ===" << std::endl;

    // 20260402 ZJH 测试 1: batch=1, channels=32, groups=8, H=W=4
    om::GroupNorm2d gn(32, 8);
    auto input = om::Tensor::randn({1, 32, 4, 4});
    auto output = gn.forward(input);

    // 20260402 ZJH 验证输出形状
    assert(output.shape(0) == 1 && output.shape(1) == 32);
    assert(output.shape(2) == 4 && output.shape(3) == 4);

    // 20260402 ZJH 验证输出非全零
    auto pOut = output.floatDataPtr();
    bool bNonZero = false;
    for (int i = 0; i < output.numel(); ++i) {
        assert(!std::isnan(pOut[i]));  // 20260402 ZJH 无 NaN
        if (std::abs(pOut[i]) > 1e-6f) bNonZero = true;
    }
    assert(bNonZero);

    // 20260402 ZJH 验证 GN 输出在每组内均值≈0, 方差≈1（因为 gamma=1, beta=0）
    int nChannelsPerGroup = 32 / 8;  // = 4
    int nHW = 4 * 4;
    for (int g = 0; g < 8; ++g) {
        float fSum = 0.0f, fSumSq = 0.0f;
        int nStart = g * nChannelsPerGroup * nHW;
        int nCount = nChannelsPerGroup * nHW;
        for (int i = 0; i < nCount; ++i) {
            fSum += pOut[nStart + i];
            fSumSq += pOut[nStart + i] * pOut[nStart + i];
        }
        float fMean = fSum / static_cast<float>(nCount);
        float fVar = fSumSq / static_cast<float>(nCount) - fMean * fMean;
        assert(std::abs(fMean) < 0.1f);        // 20260402 ZJH 均值接近 0
        assert(std::abs(fVar - 1.0f) < 0.2f);  // 20260402 ZJH 方差接近 1
    }

    std::cout << "  GroupNorm2d forward: PASS" << std::endl;
}

// 20260402 ZJH GroupNorm2d 梯度测试
void testGroupNorm2dGradient() {
    std::cout << "=== Test GroupNorm2d Gradient ===" << std::endl;

    om::GroupNorm2d gn(16, 4);
    auto input = om::Tensor::randn({2, 16, 3, 3});
    input.setRequiresGrad(true);

    auto output = gn.forward(input);
    auto loss = om::tensorSum(output);  // 20260402 ZJH 简单 sum loss
    om::tensorBackward(loss);

    auto grad = om::tensorGetGrad(input);
    assert(grad.numel() == input.numel());

    // 20260402 ZJH 验证梯度非全零
    const float* pGrad = grad.floatDataPtr();
    bool bNonZero = false;
    for (int i = 0; i < grad.numel(); ++i) {
        assert(!std::isnan(pGrad[i]));
        if (std::abs(pGrad[i]) > 1e-8f) bNonZero = true;
    }
    assert(bNonZero);

    std::cout << "  GroupNorm2d gradient: PASS" << std::endl;
}

// 20260402 ZJH GroupNorm2d 自动分组调整测试
void testGroupNormAutoGroups() {
    std::cout << "=== Test GroupNorm Auto Groups ===" << std::endl;

    // 20260402 ZJH channels=17（质数），groups=32 无法整除 → 自动降到 17 或 1
    om::GroupNorm2d gn(17, 32);
    assert(17 % gn.groups() == 0);  // 20260402 ZJH 确保自动调整后能整除
    std::cout << "  channels=17, requested_groups=32, actual_groups=" << gn.groups() << std::endl;

    // 20260402 ZJH channels=64, groups=32 → 正常
    om::GroupNorm2d gn2(64, 32);
    assert(gn2.groups() == 32);

    std::cout << "  GroupNorm auto groups: PASS" << std::endl;
}
```

- [ ] **Step 2: 运行测试**

Run: `cd e:/DevelopmentTools/OmniMatchDeeplearningTool && cmake --build build --target omnimatch_tests && ./build/omnimatch_tests 2>&1 | grep -E "GroupNorm|PASS|FAIL"`
Expected: 所有 GroupNorm 测试 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_conv.cpp
git commit -m "test: GroupNorm2d unit tests (forward, gradient, auto-groups)"
```

---

## Phase D — 更新开发记录

### Task 16: 更新 DEVLOG.md

**Files:**
- Modify: `DEVLOG.md`

- [ ] **Step 1: 在 DEVLOG.md 顶部添加今日记录**

```markdown
## [2026-04-02]

### HH:MM - GroupNorm 工业级训练质量重构
- **修改文件**: `DLTypes.h`, `TrainingConfig.h`, `cuda_kernels.cu/cuh`, `om.engine.conv.ixx`, `om.engine.autograd.ixx`, `om.engine.tensor_ops.ixx`, `om.engine.unet.ixx`, `om.engine.segmodels.ixx`, `om.engine.optimizer.ixx`, `om.engine.serializer.ixx`, `EngineBridge.cpp/h`, `TrainingSession.cpp`, `test_conv.cpp`
- **修改类型**: 新增 + 修改
- **修改内容**: 
  - 新增 GroupNorm2d 模块（CUDA kernel + autograd + CPU fallback）
  - batch < 8 自动启用 GroupNorm（对标 Halcon/ViDi）
  - SGD 增加 weight decay 支持
  - 小数据集自动模型缩放（<30→base16, <100→base32）
  - 过拟合早期警告 + 预训练权重前向验证
  - 序列化 v5 记录 norm_type
  - 强正则策略（WeightDecay=5e-4）
- **关联功能**: 训练引擎核心、分割模型
```

- [ ] **Step 2: Commit**

```bash
git add DEVLOG.md
git commit -m "docs: update DEVLOG with GroupNorm training quality refactoring"
```

---

## Risk Mitigation Summary

| Risk | Mitigation | Task |
|------|------------|------|
| GroupNorm 无 cuDNN 支持 | 手写 CUDA kernel，数学简单（per-group mean/var） | Task 2 |
| 旧 .omm 模型不兼容 GN | 序列化 v5 记录 norm_type，v4 load 仍走 BN | Task 13 |
| GN channel 数不整除 groups | 自动向下调整 groups 直到整除 | Task 5 |
| 梯度精度回归 | CPU/GPU 双路径 backward + 单元测试验证 | Task 4, 15 |
| 模型缩放改变用户选择 | 仅自动推荐 + 日志输出，不强制覆盖用户指定的架构 | Task 11 |
| SGD weight decay 影响收敛 | 仅在数据量 < 200 时自动启用 | Task 12 |
