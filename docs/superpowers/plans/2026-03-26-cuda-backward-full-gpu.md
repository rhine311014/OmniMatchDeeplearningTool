# CUDA Backward 全 GPU 化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 autograd backward 中所有 CPU 回退（D2H→CPU 计算→H2D），实现训练全程 GPU 驻留，最大化 GPU 利用率

**Architecture:** 为 7 个 CPU-fallback backward 函数添加 CUDA kernel 和 CUDABackend 包装，在 autograd.ixx 中用 `#ifdef DF_HAS_CUDA` 分支直接调用 GPU kernel

**Tech Stack:** CUDA 12.8, C++23 Modules, 现有 cuda_kernels.cu + df.hal.cuda_backend.ixx 框架

**现状:** 59 个 CUDA kernel 已实现，17/24 backward 已有 GPU 路径。需补齐 7 个 CPU 回退:
1. AdaptiveAvgPool2dBackwardFn — ResNet/所有 CNN 必经
2. MaxPool2dBackwardFn — ResNet/EfficientAD/UNet 必经
3. AddBiasBackward — 每个带偏置的 Conv2d 必经
4. UpsampleBilinearBackwardFn — UNet 必经
5. ConcatChannelsBackwardFn — UNet 必经
6. LayerNormBackwardFn — ViT 必经
7. BCEWithLogitsBackwardFn — 分割损失必经

---

### Task 1: AdaptiveAvgPool2d Backward CUDA kernel

**Files:**
- Modify: `src/cuda/cuda_kernels.cu` — 添加 kernel + C 绑定
- Modify: `src/hal/df.hal.cuda_backend.ixx` — 添加 CUDABackend 包装
- Modify: `src/engine/df.engine.autograd.ixx` — AdaptiveAvgPool2dBackwardFn 添加 GPU 分支

- [ ] **Step 1: cuda_kernels.cu 添加 kernel**

```cuda
__global__ void kernelAdaptiveAvgPool2dBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nChannels * nH * nW;
    if (nIdx >= nTotal) return;
    int w = nIdx % nW; int h = (nIdx / nW) % nH;
    int c = (nIdx / (nH * nW)) % nChannels;
    int n = nIdx / (nChannels * nH * nW);
    // 计算该输入像素贡献到哪些输出像素
    float fGrad = 0.0f;
    for (int oh = 0; oh < nOutH; ++oh) {
        int hStart = oh * nH / nOutH, hEnd = (oh + 1) * nH / nOutH;
        if (h < hStart || h >= hEnd) continue;
        for (int ow = 0; ow < nOutW; ++ow) {
            int wStart = ow * nW / nOutW, wEnd = (ow + 1) * nW / nOutW;
            if (w < wStart || w >= wEnd) continue;
            float fPoolSize = (float)(hEnd - hStart) * (wEnd - wStart);
            int nOutIdx = ((n * nChannels + c) * nOutH + oh) * nOutW + ow;
            fGrad += pGradOut[nOutIdx] / fPoolSize;
        }
    }
    pGradIn[nIdx] = fGrad;
}
```

加 `extern "C"` 绑定 `dfCudaAdaptiveAvgPool2dBackward()`。

- [ ] **Step 2: CUDABackend 添加包装方法**
- [ ] **Step 3: autograd.ixx AdaptiveAvgPool2dBackwardFn 添加 GPU 分支**

替换 D2H→CPU→H2D 为直接 `CUDABackend::adaptiveAvgPool2dBackward()`。

- [ ] **Step 4: 编译验证**

---

### Task 2: MaxPool2d Backward CUDA kernel（已有 kernel，接入 autograd）

**现状:** `dfCudaMaxPool2dBackward` kernel 已存在，`CUDABackend::maxPool2dBackward` 已存在，但 autograd 的 `MaxPool2dBackwardFn` 仍 CPU 回退。

**Files:**
- Modify: `src/engine/df.engine.autograd.ixx` — MaxPool2dBackwardFn 添加 GPU 分支
- Modify: `src/engine/df.engine.tensor_ops.ixx` — tensorMaxPool2d 前向保存 GPU indices

- [ ] **Step 1: tensorMaxPool2d 前向在 GPU 时保存 int indices 到 CUDA 内存**

当前前向在 GPU 上执行 maxpool，但 indices 保存为 float 在 CPU 上。改为 GPU int 存储。

- [ ] **Step 2: MaxPool2dBackwardFn 添加 GPU 分支**

直接调用 `CUDABackend::maxPool2dBackward()` 传入 GPU indices。

- [ ] **Step 3: 编译验证**

---

### Task 3: AddBias Backward CUDA kernel

**Files:**
- Modify: `src/cuda/cuda_kernels.cu` — 添加 bias gradient reduction kernel
- Modify: `src/hal/df.hal.cuda_backend.ixx` — 添加 CUDABackend::addBiasBackward
- Modify: `src/engine/df.engine.autograd.ixx` — AddBiasBackward 添加 GPU 分支

- [ ] **Step 1: cuda_kernels.cu 添加 kernel**

```cuda
// bias grad = sum over (N, H, W) for each channel
__global__ void kernelAddBiasBackward(
    const float* pGradOut, float* pGradBias,
    int nN, int nC, int nHW)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    float fSum = 0.0f;
    for (int n = 0; n < nN; ++n)
        for (int hw = 0; hw < nHW; ++hw)
            fSum += pGradOut[(n * nC + c) * nHW + hw];
    pGradBias[c] = fSum;
}
```

- [ ] **Step 2: CUDABackend 包装 + autograd GPU 分支**
- [ ] **Step 3: 编译验证**

---

### Task 4: UpsampleBilinear Backward CUDA kernel

**Files:**
- Modify: `src/cuda/cuda_kernels.cu`
- Modify: `src/hal/df.hal.cuda_backend.ixx`
- Modify: `src/engine/df.engine.autograd.ixx`

- [ ] **Step 1: 添加 CUDA kernel + C 绑定 + CUDABackend 包装**
- [ ] **Step 2: autograd UpsampleBilinearBackwardFn 添加 GPU 分支**
- [ ] **Step 3: 编译验证**

---

### Task 5: ConcatChannels Backward CUDA kernel

**Files:**
- Modify: `src/cuda/cuda_kernels.cu`
- Modify: `src/hal/df.hal.cuda_backend.ixx`
- Modify: `src/engine/df.engine.autograd.ixx`

- [ ] **Step 1: 添加 CUDA slice kernel（按通道维拆分梯度）**
- [ ] **Step 2: autograd ConcatChannelsBackwardFn 添加 GPU 分支**
- [ ] **Step 3: 编译验证**

---

### Task 6: LayerNorm Backward CUDA kernel

**Files:**
- Modify: `src/cuda/cuda_kernels.cu`
- Modify: `src/hal/df.hal.cuda_backend.ixx`
- Modify: `src/engine/df.engine.autograd.ixx`

- [ ] **Step 1: 添加 CUDA LayerNorm backward kernel**
- [ ] **Step 2: autograd LayerNormBackwardFn 添加 GPU 分支**
- [ ] **Step 3: 编译验证**

---

### Task 7: BCEWithLogits Backward CUDA kernel

**Files:**
- Modify: `src/cuda/cuda_kernels.cu`
- Modify: `src/hal/df.hal.cuda_backend.ixx`
- Modify: `src/engine/df.engine.autograd.ixx`

- [ ] **Step 1: 添加 CUDA BCE backward kernel**

```cuda
// grad = sigmoid(logit) - target (与 softmax CE backward 类似)
__global__ void kernelBCEWithLogitsBackward(
    const float* pLogits, const float* pTargets, float* pGradLogits,
    int nCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCount) return;
    float sig = 1.0f / (1.0f + expf(-pLogits[i]));
    pGradLogits[i] = (sig - pTargets[i]) / (float)nCount;
}
```

- [ ] **Step 2: CUDABackend 包装 + autograd GPU 分支**
- [ ] **Step 3: 编译验证**

---

### Task 8: 消除残留 D2H — BatchNorm2dBackward 参数梯度 GPU 化

**现状:** BatchNorm2dBackward 已有 GPU kernel 但 gamma/beta 梯度仍 D2H 到 CPU 计算。

**Files:**
- Modify: `src/engine/df.engine.autograd.ixx` — 移除 saved tensors 的 `.cpu()` 转换

- [ ] **Step 1: BatchNorm2dBackward 完全 GPU 化**
- [ ] **Step 2: 编译验证**

---

### Task 9: 全量编译 + 性能验证

- [ ] **Step 1: 全量编译（CUDA ON）**
- [ ] **Step 2: 启动应用，训练 ResNet-18 with CUDA，观察 GPU 利用率**
- [ ] **Step 3: 训练 UNet with CUDA，确认无 CPU fallback**
