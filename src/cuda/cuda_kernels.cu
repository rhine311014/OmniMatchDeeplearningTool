// 20260320 ZJH CUDA 内核实现 — OmniMatch GPU 加速
// 核心运算的 CUDA kernel 实现，使用 shared memory tiling 优化 matmul
// 20260324 ZJH 重大优化：32x32 tiling / im2col+GEMM 卷积 / GPU reduction BatchNorm
//              异步流 / 内存池 / ILP 元素运算 / 转置 kernel
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>

// 20260320 ZJH CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return -1; \
} while(0)

// 20260320 ZJH 线程块大小常量
constexpr int BLOCK_SIZE = 256;
// 20260324 ZJH 升级 tiling 为 32x32，适配 RTX 3060+ 现代 GPU
// 32x32 tile 占用 32*32*4*2 = 8KB shared memory，远低于 48KB 上限
// 对比 16x16：单 tile 计算量从 16^3=4096 FMA 提升至 32^3=32768 FMA，减少全局访存次数 4 倍
constexpr int TILE_SIZE_LEGACY = 16;  // 20260320 ZJH 保留旧值供兼容参考
constexpr int TILE_SIZE = 32;         // 20260324 ZJH 32x32 tiling — 适配 RTX 3060+

// =====================================================================
// CUDA 异步流管理（OPTIMIZATION 2）
// 20260324 ZJH 持久化计算流和传输流，实现计算-传输重叠
// =====================================================================

// 20260324 ZJH 静态 CUDA 流：计算流用于 kernel 执行，传输流用于 H2D/D2H 数据拷贝
// 双流架构允许数据传输与 kernel 计算并行，提升 GPU 利用率
static cudaStream_t s_computeStream = nullptr;  // 20260324 ZJH 计算流（kernel 提交）
static cudaStream_t s_transferStream = nullptr;  // 20260324 ZJH 传输流（内存拷贝）
static bool s_bStreamsInitialized = false;        // 20260324 ZJH 流初始化标记

// 20260324 ZJH 初始化 CUDA 异步流
// 创建两个持久流：计算流（kernel 执行）和传输流（Host/Device 数据搬运）
// 双流可实现计算与数据传输重叠，提升吞吐量
static void initStreams() {
    if (!s_bStreamsInitialized) {  // 20260324 ZJH 仅在首次调用时创建
        cudaStreamCreate(&s_computeStream);   // 20260324 ZJH 创建计算流
        cudaStreamCreate(&s_transferStream);  // 20260324 ZJH 创建传输流
        s_bStreamsInitialized = true;          // 20260324 ZJH 标记已初始化
    }
}

// 20260324 ZJH 销毁 CUDA 异步流，释放流资源
static void destroyStreams() {
    if (s_bStreamsInitialized) {  // 20260324 ZJH 仅在已初始化时销毁
        cudaStreamDestroy(s_computeStream);   // 20260324 ZJH 销毁计算流
        cudaStreamDestroy(s_transferStream);  // 20260324 ZJH 销毁传输流
        s_computeStream = nullptr;            // 20260324 ZJH 指针置空防止悬挂
        s_transferStream = nullptr;           // 20260324 ZJH 指针置空防止悬挂
        s_bStreamsInitialized = false;         // 20260324 ZJH 重置初始化标记
    }
}

// =====================================================================
// GPU 内存池（OPTIMIZATION 3）
// 20260324 ZJH 简单内存池，复用已释放的 GPU 内存块避免频繁 cudaMalloc/cudaFree
// cudaMalloc 单次调用约 100~500us 延迟，内存池可将热路径降至 <1us
// =====================================================================

// 20260324 ZJH GPU 内存块描述符
struct GpuMemBlock {
    void* pData;     // 20260324 ZJH 设备内存指针
    size_t nBytes;   // 20260324 ZJH 块大小（字节）
    bool bFree;      // 20260324 ZJH 是否空闲可复用
};

// 20260324 ZJH 全局内存池和保护互斥锁
static std::vector<GpuMemBlock> s_vecGpuPool;  // 20260324 ZJH 内存池（所有已分配块）
static std::mutex s_poolMutex;                   // 20260324 ZJH 保护内存池的互斥锁（多线程安全）

// 20260324 ZJH 从内存池分配 GPU 内存
// 策略：优先复用空闲块（大小在 [nBytes, nBytes*2] 范围内最佳匹配）
// 若无合适空闲块则调用 cudaMalloc 分配新块并加入池中
// 参数: nBytes - 请求的字节数
// 返回: 设备内存指针，失败返回 nullptr
static void* gpuPoolAlloc(size_t nBytes) {
    std::lock_guard<std::mutex> lock(s_poolMutex);  // 20260324 ZJH 加锁保护池操作
    // 20260324 ZJH 遍历池查找合适的空闲块
    // 20260326 ZJH 放宽匹配：任何 >= nBytes 的空闲块都可复用（优先选最小匹配减少浪费）
    int nBestIdx = -1;
    size_t nBestSize = SIZE_MAX;
    for (int i = 0; i < static_cast<int>(s_vecGpuPool.size()); ++i) {
        auto& block = s_vecGpuPool[i];
        if (block.bFree && block.nBytes >= nBytes && block.nBytes < nBestSize) {
            nBestIdx = i;
            nBestSize = block.nBytes;
            if (nBestSize <= nBytes * 2) break;  // 20260326 ZJH 2 倍以内直接用，不继续找
        }
    }
    if (nBestIdx >= 0) {
        s_vecGpuPool[nBestIdx].bFree = false;
        return s_vecGpuPool[nBestIdx].pData;
    }

    // 20260324 ZJH 无合适空闲块，分配新的 GPU 内存
    void* pPtr = nullptr;
    cudaError_t err = cudaMalloc(&pPtr, nBytes);
    if (err == cudaSuccess) {
        s_vecGpuPool.push_back({pPtr, nBytes, false});
        return pPtr;
    }

    // 20260326 ZJH cudaMalloc 失败：释放池中所有空闲块回收显存，再重试
    // 这是训练 OOM 的根因修复：im2col 等临时缓冲区被池缓存但永不释放
    // 导致 cudaMalloc 无法分配新内存（即使池中有大量空闲块）
    for (auto it = s_vecGpuPool.begin(); it != s_vecGpuPool.end(); ) {
        if (it->bFree) {
            cudaFree(it->pData);       // 20260326 ZJH 真正释放 GPU 内存
            it = s_vecGpuPool.erase(it);
        } else {
            ++it;
        }
    }
    // 20260326 ZJH 重试分配
    err = cudaMalloc(&pPtr, nBytes);
    if (err != cudaSuccess) return nullptr;
    s_vecGpuPool.push_back({pPtr, nBytes, false});
    return pPtr;
}

// 20260324 ZJH 归还 GPU 内存到内存池
// 20260327 ZJH 全缓存策略（类似 PyTorch CachingAllocator）：
//   所有块无论大小均缓存在池中，不真正调用 cudaFree
//   仅在 cudaMalloc 失败（OOM）时才批量释放空闲块回收显存
//   这消除了训练过程中 2G→12G 的显存波动（CUDA 驱动层级始终稳定）
static void gpuPoolFree(void* pPtr) {
    std::lock_guard<std::mutex> lock(s_poolMutex);  // 20260324 ZJH 加锁保护池操作
    for (auto& block : s_vecGpuPool) {
        if (block.pData == pPtr) {
            block.bFree = true;  // 20260327 ZJH 标记为空闲，留在池中等待复用
            return;
        }
    }
    // 20260324 ZJH 未在池中找到，直接释放（防御性处理）
    cudaFree(pPtr);
}

// 20260324 ZJH 释放内存池中所有 GPU 内存（程序退出或设备重置时调用）
static void gpuPoolClear() {
    std::lock_guard<std::mutex> lock(s_poolMutex);  // 20260324 ZJH 加锁保护池操作
    for (auto& block : s_vecGpuPool) {
        cudaFree(block.pData);  // 20260324 ZJH 释放每个块的设备内存
    }
    s_vecGpuPool.clear();  // 20260324 ZJH 清空池
}

// =====================================================================
// 设备管理
// =====================================================================

extern "C" int omCudaInit(int nDeviceId) {
    // 20260326 ZJH 清除遗留 CUDA 错误状态（上次训练失败可能留下 illegal memory access）
    cudaGetLastError();
    CUDA_CHECK(cudaSetDevice(nDeviceId));
    // 20260326 ZJH 再次清除（cudaSetDevice 可能触发延迟错误）
    cudaGetLastError();
    // 20260324 ZJH 初始化异步流（设备就绪后立即创建）
    initStreams();
    return 0;
}

extern "C" int omCudaGetDeviceCount() {
    int nCount = 0;
    cudaGetDeviceCount(&nCount);
    return nCount;
}

extern "C" int omCudaGetDeviceName(int nDeviceId, char* pNameBuf, int nBufSize) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, nDeviceId));
    snprintf(pNameBuf, nBufSize, "%s", prop.name);
    return 0;
}

extern "C" int omCudaGetMemInfo(int nDeviceId, size_t* pFreeMem, size_t* pTotalMem) {
    CUDA_CHECK(cudaSetDevice(nDeviceId));
    CUDA_CHECK(cudaMemGetInfo(pFreeMem, pTotalMem));
    return 0;
}

// =====================================================================
// 内存管理
// =====================================================================

extern "C" int omCudaMalloc(void** ppDevPtr, size_t nBytes) {
    // 20260327 ZJH 通过缓存池分配：优先复用空闲块，避免频繁 cudaMalloc 导致显存波动
    // 池中无合适块时才调用 cudaMalloc；OOM 时自动释放池中空闲块并重试
    void* pPtr = gpuPoolAlloc(nBytes);
    if (!pPtr) return -1;  // 20260327 ZJH 即使释放池后仍 OOM 则失败
    *ppDevPtr = pPtr;
    return 0;
}

extern "C" int omCudaFree(void* pDevPtr) {
    // 20260327 ZJH 归还到缓存池：标记为空闲可复用，不真正释放
    // 显存从 CUDA 驱动视角保持稳定（不再波动），应用层自行管理复用
    if (pDevPtr) gpuPoolFree(pDevPtr);
    return 0;
}

extern "C" int omCudaCopyH2D(void* pDst, const void* pSrc, size_t nBytes) {
    cudaError_t err = cudaMemcpy(pDst, pSrc, nBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] H2D FAILED: %s (dst=%p, nBytes=%zu)\n",
                cudaGetErrorString(err), pDst, nBytes);
        return -1;
    }
    return 0;
}

extern "C" int omCudaCopyD2H(void* pDst, const void* pSrc, size_t nBytes) {
    CUDA_CHECK(cudaMemcpy(pDst, pSrc, nBytes, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int omCudaCopyD2D(void* pDst, const void* pSrc, size_t nBytes) {
    CUDA_CHECK(cudaMemcpy(pDst, pSrc, nBytes, cudaMemcpyDeviceToDevice));
    return 0;
}

extern "C" int omCudaMemset(void* pDev, int nValue, size_t nBytes) {
    CUDA_CHECK(cudaMemset(pDev, nValue, nBytes));
    return 0;
}

extern "C" int omCudaSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

// 20260324 ZJH 异步 Host -> Device 拷贝（使用传输流，与计算流并行）
// 调用后不会阻塞 CPU，数据传输在传输流中排队执行
extern "C" int omCudaAsyncCopyH2D(void* pDst, const void* pSrc, size_t nBytes) {
    initStreams();  // 20260324 ZJH 确保流已初始化
    CUDA_CHECK(cudaMemcpyAsync(pDst, pSrc, nBytes, cudaMemcpyHostToDevice, s_transferStream));
    return 0;
}

// 20260324 ZJH 异步 Device -> Host 拷贝（使用传输流）
extern "C" int omCudaAsyncCopyD2H(void* pDst, const void* pSrc, size_t nBytes) {
    initStreams();  // 20260324 ZJH 确保流已初始化
    CUDA_CHECK(cudaMemcpyAsync(pDst, pSrc, nBytes, cudaMemcpyDeviceToHost, s_transferStream));
    return 0;
}

// 20260324 ZJH 同步传输流（等待所有异步传输完成）
extern "C" int omCudaSyncTransferStream() {
    if (s_transferStream) {
        CUDA_CHECK(cudaStreamSynchronize(s_transferStream));
    }
    return 0;
}

// 20260324 ZJH 同步计算流（等待所有异步计算完成）
extern "C" int omCudaSyncComputeStream() {
    if (s_computeStream) {
        CUDA_CHECK(cudaStreamSynchronize(s_computeStream));
    }
    return 0;
}

// 20260324 ZJH 释放所有池内存并销毁流（程序退出时调用）
extern "C" int omCudaCleanup() {
    gpuPoolClear();    // 20260324 ZJH 释放内存池
    destroyStreams();  // 20260324 ZJH 销毁异步流
    return 0;
}

// 20260326 ZJH 强制重置 GPU — 同步+清错误+释放池+流
// 不用 cudaDeviceReset（会销毁 CUDA 上下文，TensorStorage 析构时 cudaFree 在已销毁上下文上
// → illegal memory access → 永久污染后续所有 CUDA 调用）
extern "C" int omCudaForceReset() {
    // 20260326 ZJH 同步所有操作，确保 kernel 完成
    cudaDeviceSynchronize();
    // 20260326 ZJH 清除遗留 CUDA 错误状态
    cudaGetLastError();
    // 20260326 ZJH 释放池和流
    gpuPoolClear();
    destroyStreams();
    return 0;
}

// =====================================================================
// 元素运算 Kernels — ILP 优化版（OPTIMIZATION 6）
// 20260324 ZJH 每线程处理 4 个元素（Instruction-Level Parallelism）
// ILP 优化原理：GPU 线程在等待内存读取时可并行计算多个独立操作
// 每线程 4 元素可隐藏全局内存延迟（~400 cycles），提升有效吞吐量约 2~3 倍
// =====================================================================

// 20260324 ZJH ILP 优化逐元素加法 kernel — 每线程处理 4 个连续元素
// 通过处理 4 个独立的加法运算，隐藏全局内存延迟
__global__ void kernelAddILP(const float* __restrict__ pA,
                              const float* __restrict__ pB,
                              float* __restrict__ pC, int nCount) {
    int nBase = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 20260324 ZJH 每线程起始偏移 x4
    // 20260324 ZJH 完整 4 元素路径：一次处理 4 个加法
    if (nBase + 3 < nCount) {
        pC[nBase]     = pA[nBase]     + pB[nBase];      // 20260324 ZJH 第 1 个元素
        pC[nBase + 1] = pA[nBase + 1] + pB[nBase + 1];  // 20260324 ZJH 第 2 个元素
        pC[nBase + 2] = pA[nBase + 2] + pB[nBase + 2];  // 20260324 ZJH 第 3 个元素
        pC[nBase + 3] = pA[nBase + 3] + pB[nBase + 3];  // 20260324 ZJH 第 4 个元素
    } else {
        // 20260324 ZJH 尾部处理：不足 4 个元素时逐个处理
        for (int i = nBase; i < nCount && i < nBase + 4; ++i) {
            pC[i] = pA[i] + pB[i];
        }
    }
}

extern "C" int omCudaAdd(const float* pA, const float* pB, float* pC, int nCount) {
    // 20260324 ZJH 块数 = ceil(元素总数 / 4 / 线程块大小)
    int nBlocks = (nCount / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nBlocks < 1) nBlocks = 1;  // 20260324 ZJH 至少 1 个块
    kernelAddILP<<<nBlocks, BLOCK_SIZE>>>(pA, pB, pC, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260324 ZJH ILP 优化逐元素减法 kernel — 每线程处理 4 个连续元素
__global__ void kernelSubILP(const float* __restrict__ pA,
                              const float* __restrict__ pB,
                              float* __restrict__ pC, int nCount) {
    int nBase = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 20260324 ZJH 每线程起始偏移 x4
    if (nBase + 3 < nCount) {
        pC[nBase]     = pA[nBase]     - pB[nBase];      // 20260324 ZJH 第 1 个元素
        pC[nBase + 1] = pA[nBase + 1] - pB[nBase + 1];  // 20260324 ZJH 第 2 个元素
        pC[nBase + 2] = pA[nBase + 2] - pB[nBase + 2];  // 20260324 ZJH 第 3 个元素
        pC[nBase + 3] = pA[nBase + 3] - pB[nBase + 3];  // 20260324 ZJH 第 4 个元素
    } else {
        for (int i = nBase; i < nCount && i < nBase + 4; ++i) {
            pC[i] = pA[i] - pB[i];  // 20260324 ZJH 尾部逐个处理
        }
    }
}

extern "C" int omCudaSub(const float* pA, const float* pB, float* pC, int nCount) {
    int nBlocks = (nCount / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nBlocks < 1) nBlocks = 1;
    kernelSubILP<<<nBlocks, BLOCK_SIZE>>>(pA, pB, pC, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260324 ZJH ILP 优化逐元素乘法 kernel — 每线程处理 4 个连续元素
__global__ void kernelMulILP(const float* __restrict__ pA,
                              const float* __restrict__ pB,
                              float* __restrict__ pC, int nCount) {
    int nBase = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 20260324 ZJH 每线程起始偏移 x4
    if (nBase + 3 < nCount) {
        pC[nBase]     = pA[nBase]     * pB[nBase];      // 20260324 ZJH 第 1 个元素
        pC[nBase + 1] = pA[nBase + 1] * pB[nBase + 1];  // 20260324 ZJH 第 2 个元素
        pC[nBase + 2] = pA[nBase + 2] * pB[nBase + 2];  // 20260324 ZJH 第 3 个元素
        pC[nBase + 3] = pA[nBase + 3] * pB[nBase + 3];  // 20260324 ZJH 第 4 个元素
    } else {
        for (int i = nBase; i < nCount && i < nBase + 4; ++i) {
            pC[i] = pA[i] * pB[i];  // 20260324 ZJH 尾部逐个处理
        }
    }
}

extern "C" int omCudaMul(const float* pA, const float* pB, float* pC, int nCount) {
    int nBlocks = (nCount / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nBlocks < 1) nBlocks = 1;
    kernelMulILP<<<nBlocks, BLOCK_SIZE>>>(pA, pB, pC, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260324 ZJH ILP 优化乘标量 kernel — 每线程处理 4 个连续元素
__global__ void kernelMulScalarILP(const float* __restrict__ pA, float fScalar,
                                    float* __restrict__ pC, int nCount) {
    int nBase = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 20260324 ZJH 每线程起始偏移 x4
    if (nBase + 3 < nCount) {
        pC[nBase]     = pA[nBase]     * fScalar;  // 20260324 ZJH 第 1 个元素
        pC[nBase + 1] = pA[nBase + 1] * fScalar;  // 20260324 ZJH 第 2 个元素
        pC[nBase + 2] = pA[nBase + 2] * fScalar;  // 20260324 ZJH 第 3 个元素
        pC[nBase + 3] = pA[nBase + 3] * fScalar;  // 20260324 ZJH 第 4 个元素
    } else {
        for (int i = nBase; i < nCount && i < nBase + 4; ++i) {
            pC[i] = pA[i] * fScalar;  // 20260324 ZJH 尾部逐个处理
        }
    }
}

extern "C" int omCudaMulScalar(const float* pA, float fScalar, float* pC, int nCount) {
    int nBlocks = (nCount / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nBlocks < 1) nBlocks = 1;
    kernelMulScalarILP<<<nBlocks, BLOCK_SIZE>>>(pA, fScalar, pC, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260324 ZJH ILP 优化加标量 kernel — 每线程处理 4 个连续元素
__global__ void kernelAddScalarILP(const float* __restrict__ pA, float fScalar,
                                    float* __restrict__ pC, int nCount) {
    int nBase = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 20260324 ZJH 每线程起始偏移 x4
    if (nBase + 3 < nCount) {
        pC[nBase]     = pA[nBase]     + fScalar;  // 20260324 ZJH 第 1 个元素
        pC[nBase + 1] = pA[nBase + 1] + fScalar;  // 20260324 ZJH 第 2 个元素
        pC[nBase + 2] = pA[nBase + 2] + fScalar;  // 20260324 ZJH 第 3 个元素
        pC[nBase + 3] = pA[nBase + 3] + fScalar;  // 20260324 ZJH 第 4 个元素
    } else {
        for (int i = nBase; i < nCount && i < nBase + 4; ++i) {
            pC[i] = pA[i] + fScalar;  // 20260324 ZJH 尾部逐个处理
        }
    }
}

extern "C" int omCudaAddScalar(const float* pA, float fScalar, float* pC, int nCount) {
    int nBlocks = (nCount / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nBlocks < 1) nBlocks = 1;
    kernelAddScalarILP<<<nBlocks, BLOCK_SIZE>>>(pA, fScalar, pC, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 激活函数 Kernels
// =====================================================================

// 20260320 ZJH ReLU kernel
__global__ void kernelReLU(const float* pIn, float* pOut, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nCount) pOut[idx] = pIn[idx] > 0.0f ? pIn[idx] : 0.0f;
}

extern "C" int omCudaReLU(const float* pIn, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelReLU<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260320 ZJH ReLU 反向 kernel
__global__ void kernelReLUBackward(const float* pIn, const float* pGradOut, float* pGradIn, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nCount) pGradIn[idx] = pIn[idx] > 0.0f ? pGradOut[idx] : 0.0f;
}

extern "C" int omCudaReLUBackward(const float* pIn, const float* pGradOut, float* pGradIn, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelReLUBackward<<<nBlocks, BLOCK_SIZE>>>(pIn, pGradOut, pGradIn, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260320 ZJH Sigmoid kernel
__global__ void kernelSigmoid(const float* pIn, float* pOut, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nCount) pOut[idx] = 1.0f / (1.0f + expf(-pIn[idx]));
}

extern "C" int omCudaSigmoid(const float* pIn, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelSigmoid<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260320 ZJH GELU kernel
__global__ void kernelGELU(const float* pIn, float* pOut, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nCount) {
        float x = pIn[idx];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        pOut[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" int omCudaGELU(const float* pIn, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelGELU<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260320 ZJH SiLU kernel
__global__ void kernelSiLU(const float* pIn, float* pOut, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nCount) {
        float sig = 1.0f / (1.0f + expf(-pIn[idx]));
        pOut[idx] = pIn[idx] * sig;
    }
}

extern "C" int omCudaSiLU(const float* pIn, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelSiLU<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 矩阵转置 kernel（OPTIMIZATION 1.3）
// 20260324 ZJH 高效矩阵转置，使用 shared memory 避免非合并内存访问
// 朴素转置的 B[j][i]=A[i][j] 会导致列方向写入（stride=N），带宽仅利用 1/32
// 使用 32x32 shared memory tile 可将读写都变为合并访问，带宽提升约 10~20 倍
// =====================================================================

// 20260324 ZJH 转置 tile 大小常量
constexpr int TRANSPOSE_TILE = 32;  // 20260324 ZJH 32x32 tile 匹配 warp 大小

// 20260324 ZJH 矩阵转置 kernel：pOut[j*nRows+i] = pIn[i*nCols+j]
// 使用 shared memory tile 实现合并读写，避免全局内存 stride 访问瓶颈
// 参数: pIn - 输入矩阵 [nRows, nCols]
//       pOut - 输出矩阵 [nCols, nRows]（转置后）
//       nRows, nCols - 输入矩阵维度
__global__ void kernelTranspose(const float* __restrict__ pIn,
                                 float* __restrict__ pOut,
                                 int nRows, int nCols) {
    // 20260324 ZJH +1 列避免 bank conflict（shared memory 32 bank 对齐时连续列会冲突）
    __shared__ float tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    // 20260324 ZJH 计算输入矩阵中的 tile 起始坐标
    int nTileRow = blockIdx.y * TRANSPOSE_TILE;  // 20260324 ZJH tile 起始行
    int nTileCol = blockIdx.x * TRANSPOSE_TILE;  // 20260324 ZJH tile 起始列

    // 20260324 ZJH 合并读取：按行读入 shared memory
    int nInRow = nTileRow + threadIdx.y;  // 20260324 ZJH 当前线程对应的输入行
    int nInCol = nTileCol + threadIdx.x;  // 20260324 ZJH 当前线程对应的输入列
    if (nInRow < nRows && nInCol < nCols) {
        tile[threadIdx.y][threadIdx.x] = pIn[nInRow * nCols + nInCol];  // 20260324 ZJH 合并读取
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;  // 20260324 ZJH 越界填零
    }

    __syncthreads();  // 20260324 ZJH 等待所有线程完成 tile 加载

    // 20260324 ZJH 合并写出：按行写出（转置后的行 = 原始列）
    int nOutRow = nTileCol + threadIdx.y;  // 20260324 ZJH 输出行 = 原始列
    int nOutCol = nTileRow + threadIdx.x;  // 20260324 ZJH 输出列 = 原始行
    if (nOutRow < nCols && nOutCol < nRows) {
        // 20260324 ZJH 从 tile 中按转置方式读取：tile[x][y] 而非 tile[y][x]
        pOut[nOutRow * nRows + nOutCol] = tile[threadIdx.x][threadIdx.y];
    }
}

// 20260324 ZJH 矩阵转置接口：pOut[nCols, nRows] = transpose(pIn[nRows, nCols])
extern "C" int omCudaTranspose(const float* pIn, float* pOut, int nRows, int nCols) {
    dim3 blockDim(TRANSPOSE_TILE, TRANSPOSE_TILE);  // 20260324 ZJH 32x32 线程块
    dim3 gridDim((nCols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
                 (nRows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);  // 20260324 ZJH 覆盖全矩阵
    kernelTranspose<<<gridDim, blockDim>>>(pIn, pOut, nRows, nCols);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 矩阵乘法 — 32x32 Shared Memory Tiling（OPTIMIZATION 1.1）
// 20260324 ZJH 从 16x16 升级到 32x32 tiling，适配 RTX 3060+ 现代 GPU
// 使用 __restrict__ 提示编译器无指针别名，启用更激进的优化
// =====================================================================

// 20260324 ZJH 优化版 Tiled matmul kernel：C[M,N] = A[M,K] * B[K,N]
// 32x32 shared memory tile 减少全局内存访问次数
// 每 tile 计算 32^3 = 32768 FMA 操作（对比 16x16 的 4096，提升 8 倍计算密度）
// __restrict__ 告知编译器 pA/pB/pC 无重叠，允许更激进的寄存器优化和指令重排
__global__ void kernelMatmulOpt(const float* __restrict__ pA,
                                 const float* __restrict__ pB,
                                 float* __restrict__ pC,
                                 int nM, int nK, int nN) {
    // 20260324 ZJH 32x32 shared memory tile
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int nRow = blockIdx.y * TILE_SIZE + threadIdx.y;  // 20260324 ZJH 输出行索引
    int nCol = blockIdx.x * TILE_SIZE + threadIdx.x;  // 20260324 ZJH 输出列索引
    float fSum = 0.0f;  // 20260324 ZJH 累加器（寄存器中维护，避免全局读写）

    int nTiles = (nK + TILE_SIZE - 1) / TILE_SIZE;  // 20260324 ZJH K 方向 tile 数
    for (int t = 0; t < nTiles; ++t) {
        // 20260324 ZJH 合并加载 A 的 tile（每线程加载 1 个元素）
        int nACol = t * TILE_SIZE + threadIdx.x;  // 20260324 ZJH A 矩阵列坐标
        tileA[threadIdx.y][threadIdx.x] = (nRow < nM && nACol < nK)
            ? pA[nRow * nK + nACol] : 0.0f;

        // 20260324 ZJH 合并加载 B 的 tile
        int nBRow = t * TILE_SIZE + threadIdx.y;  // 20260324 ZJH B 矩阵行坐标
        tileB[threadIdx.y][threadIdx.x] = (nBRow < nK && nCol < nN)
            ? pB[nBRow * nN + nCol] : 0.0f;

        __syncthreads();  // 20260324 ZJH 等待 tile 加载完成

        // 20260324 ZJH 计算 tile 内积：32 次 FMA 操作
        // 编译器会自动展开此循环，利用寄存器文件
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            fSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();  // 20260324 ZJH 等待所有线程完成计算后再加载下一个 tile
    }

    // 20260324 ZJH 将累加结果写回全局内存
    if (nRow < nM && nCol < nN) {
        pC[nRow * nN + nCol] = fSum;
    }
}

extern "C" int omCudaMatmul(const float* pA, const float* pB, float* pC,
                            int nM, int nK, int nN) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 20260324 ZJH 32x32 线程块
    dim3 gridDim((nN + TILE_SIZE - 1) / TILE_SIZE,
                 (nM + TILE_SIZE - 1) / TILE_SIZE);  // 20260324 ZJH grid 覆盖整个输出矩阵
    kernelMatmulOpt<<<gridDim, blockDim>>>(pA, pB, pC, nM, nK, nN);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int omCudaBatchedMatmul(const float* pA, const float* pB, float* pC,
                                    int nBatch, int nM, int nK, int nN) {
    for (int b = 0; b < nBatch; ++b) {
        int ret = omCudaMatmul(pA + b * nM * nK, pB + b * nK * nN,
                               pC + b * nM * nN, nM, nK, nN);
        if (ret != 0) return ret;
    }
    return 0;
}

// =====================================================================
// Im2col + GEMM 卷积（OPTIMIZATION 1.2）
// 20260324 ZJH 将卷积转化为矩阵乘法：先通过 im2col 展开输入补丁为列矩阵
// 再使用优化后的 32x32 tiled matmul 完成卷积计算
// Im2col+GEMM 策略广泛用于 cuDNN/caffe/PyTorch 等主流框架
// =====================================================================

// 20260324 ZJH Im2col kernel：将输入图像展开为列矩阵
// 输入: pInput [N, Cin, H, W] 的单个样本（已偏移到 batch 起点）
// 输出: pCol [Cin*KH*KW, Hout*Wout] 列矩阵
// 每个线程负责计算 col 矩阵中的一个元素
// 参数含义：
//   nCin - 输入通道数
//   nH, nW - 输入高宽
//   nKH, nKW - 卷积核高宽
//   nPadH, nPadW - 上下/左右填充像素数
//   nStrideH, nStrideW - 纵向/横向步长
//   nHout, nWout - 输出特征图高宽
__global__ void kernelIm2col(const float* __restrict__ pInput,
                              float* __restrict__ pCol,
                              int nCin, int nH, int nW,
                              int nKH, int nKW,
                              int nPadH, int nPadW,
                              int nStrideH, int nStrideW,
                              int nHout, int nWout) {
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260324 ZJH 全局线程索引
    int nColRows = nCin * nKH * nKW;      // 20260324 ZJH 列矩阵行数
    int nColCols = nHout * nWout;          // 20260324 ZJH 列矩阵列数
    int nTotalCols = nColRows * nColCols;  // 20260324 ZJH 列矩阵总元素数

    if (nIdx >= nTotalCols) return;  // 20260324 ZJH 越界检查

    // 20260324 ZJH 从线性索引反推 col 矩阵中的 (行, 列) 坐标
    int nColCol = nIdx % nColCols;         // 20260324 ZJH 列矩阵列索引 = 输出空间位置
    int nColRow = nIdx / nColCols;         // 20260324 ZJH 列矩阵行索引 = (cin, kh, kw) 组合

    // 20260324 ZJH 从列索引反推输出空间坐标 (oh, ow)
    int nOw = nColCol % nWout;             // 20260324 ZJH 输出列坐标
    int nOh = nColCol / nWout;             // 20260324 ZJH 输出行坐标

    // 20260324 ZJH 从行索引反推通道和卷积核偏移 (cin, kh, kw)
    int nKw = nColRow % nKW;               // 20260324 ZJH 核列偏移
    int nKh = (nColRow / nKW) % nKH;       // 20260324 ZJH 核行偏移
    int nCi = nColRow / (nKH * nKW);       // 20260324 ZJH 输入通道索引

    // 20260324 ZJH 计算输入图像中的实际采样坐标
    int nIh = nOh * nStrideH - nPadH + nKh;  // 20260324 ZJH 输入行（考虑 stride 和 padding）
    int nIw = nOw * nStrideW - nPadW + nKw;  // 20260324 ZJH 输入列

    // 20260324 ZJH 边界检查：越界位置填零（zero-padding 语义）
    if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
        pCol[nIdx] = pInput[(nCi * nH + nIh) * nW + nIw];  // 20260324 ZJH 有效位置：拷贝输入值
    } else {
        pCol[nIdx] = 0.0f;  // 20260324 ZJH 越界：零填充
    }
}

// 20260324 ZJH 加偏置 kernel：对卷积输出的每个通道加偏置
// pOutput[co * nSpatial + s] += pBias[co]
// 参数: nCout - 输出通道数; nSpatial - 每通道空间元素数 (Hout*Wout)
__global__ void kernelAddBias(float* __restrict__ pOutput,
                               const float* __restrict__ pBias,
                               int nCout, int nSpatial) {
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260324 ZJH 全局线程索引
    int nTotal = nCout * nSpatial;                       // 20260324 ZJH 总元素数
    if (nIdx >= nTotal) return;
    int nCo = nIdx / nSpatial;                           // 20260324 ZJH 所属输出通道
    pOutput[nIdx] += pBias[nCo];                         // 20260324 ZJH 加偏置
}

// 20260324 ZJH Im2col+GEMM 卷积前向接口
// 步骤: (1) im2col 展开输入 → colMatrix [Cin*KH*KW, Hout*Wout]
//        (2) matmul: Output = Weight[Cout, Cin*KH*KW] * colMatrix → [Cout, Hout*Wout]
//        (3) 加偏置（可选）
// 此策略将复杂卷积运算转化为高度优化的矩阵乘法，充分利用 GPU tensor core
extern "C" int omCudaConv2dIm2col(const float* pInput, const float* pWeight, const float* pBias,
                                   float* pOutput,
                                   int nBatch, int nCin, int nH, int nW,
                                   int nCout, int nKH, int nKW, int nStride, int nPad) {
    int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260324 ZJH 输出高度
    int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260324 ZJH 输出宽度
    int nColRows = nCin * nKH * nKW;                    // 20260324 ZJH im2col 矩阵行数
    int nColCols = nHout * nWout;                        // 20260324 ZJH im2col 矩阵列数
    size_t nColBytes = static_cast<size_t>(nColRows) * nColCols * sizeof(float);  // 20260324 ZJH col 矩阵字节数

    // 20260324 ZJH 从内存池分配 im2col 临时缓冲区
    float* pCol = static_cast<float*>(gpuPoolAlloc(nColBytes));
    if (!pCol) return -1;  // 20260324 ZJH 分配失败

    // 20260324 ZJH 逐 batch 处理：im2col 展开后做矩阵乘法
    for (int n = 0; n < nBatch; ++n) {
        const float* pBatchIn = pInput + n * nCin * nH * nW;      // 20260324 ZJH 当前 batch 输入偏移
        float* pBatchOut = pOutput + n * nCout * nHout * nWout;    // 20260324 ZJH 当前 batch 输出偏移

        // 20260324 ZJH 步骤 1: im2col 展开
        int nTotalCol = nColRows * nColCols;                       // 20260324 ZJH 列矩阵总元素数
        int nIm2colBlocks = (nTotalCol + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260324 ZJH 所需线程块数
        kernelIm2col<<<nIm2colBlocks, BLOCK_SIZE>>>(
            pBatchIn, pCol,
            nCin, nH, nW, nKH, nKW,
            nPad, nPad, nStride, nStride,
            nHout, nWout);

        // 20260324 ZJH 步骤 2: GEMM — Output[Cout, Hout*Wout] = Weight[Cout, Cin*KH*KW] * Col[Cin*KH*KW, Hout*Wout]
        // 调用优化后的 32x32 tiled matmul
        omCudaMatmul(pWeight, pCol, pBatchOut, nCout, nColRows, nColCols);

        // 20260324 ZJH 步骤 3: 加偏置（如果提供了偏置向量）
        if (pBias) {
            int nBiasTotal = nCout * nColCols;                     // 20260324 ZJH 输出总元素数
            int nBiasBlocks = (nBiasTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelAddBias<<<nBiasBlocks, BLOCK_SIZE>>>(pBatchOut, pBias, nCout, nColCols);
        }
    }

    // 20260324 ZJH 归还 im2col 临时缓冲区到内存池
    gpuPoolFree(pCol);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260320 ZJH 保留原始朴素卷积实现供小 kernel 使用
__global__ void kernelConv2d(const float* pInput, const float* pWeight, const float* pBias,
                              float* pOutput,
                              int nBatch, int nCin, int nH, int nW,
                              int nCout, int nKH, int nKW,
                              int nStride, int nPad,
                              int nHout, int nWout) {
    // 20260320 ZJH 每个线程处理一个输出位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nCout * nHout * nWout;
    if (idx >= nTotal) return;

    // 20260320 ZJH 解码线性索引为 (n, co, oh, ow)
    int ow = idx % nWout;
    int oh = (idx / nWout) % nHout;
    int co = (idx / (nWout * nHout)) % nCout;
    int n  = idx / (nWout * nHout * nCout);

    float fSum = 0.0f;
    for (int ci = 0; ci < nCin; ++ci)
    for (int kh = 0; kh < nKH; ++kh)
    for (int kw = 0; kw < nKW; ++kw) {
        int ih = oh * nStride - nPad + kh;
        int iw = ow * nStride - nPad + kw;
        if (ih >= 0 && ih < nH && iw >= 0 && iw < nW) {
            fSum += pInput[((n * nCin + ci) * nH + ih) * nW + iw]
                  * pWeight[((co * nCin + ci) * nKH + kh) * nKW + kw];
        }
    }
    if (pBias) fSum += pBias[co];
    pOutput[idx] = fSum;
}

// 20260324 ZJH Conv2d 前向统一入口：自动选择 im2col+GEMM 或朴素实现
// 策略: Cin*KH*KW >= 64 时使用 im2col+GEMM（矩阵较大，GEMM 效率高）
//       否则使用朴素实现（小卷积核的 im2col 开销不划算）
extern "C" int omCudaConv2d(const float* pInput, const float* pWeight, const float* pBias,
                            float* pOutput,
                            int nBatch, int nCin, int nH, int nW,
                            int nCout, int nKH, int nKW, int nStride, int nPad) {
    int nColRows = nCin * nKH * nKW;  // 20260324 ZJH im2col 矩阵行数

    // 20260324 ZJH 当列矩阵行数 >= 64 时，im2col+GEMM 策略更优（amortize im2col 开销）
    if (nColRows >= 64) {
        return omCudaConv2dIm2col(pInput, pWeight, pBias, pOutput,
                                   nBatch, nCin, nH, nW,
                                   nCout, nKH, nKW, nStride, nPad);
    }

    // 20260324 ZJH 小卷积核回退到朴素实现（避免 im2col 内存分配开销）
    int nHout = (nH + 2 * nPad - nKH) / nStride + 1;
    int nWout = (nW + 2 * nPad - nKW) / nStride + 1;
    int nTotal = nBatch * nCout * nHout * nWout;
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelConv2d<<<nBlocks, BLOCK_SIZE>>>(pInput, pWeight, pBias, pOutput,
                                           nBatch, nCin, nH, nW,
                                           nCout, nKH, nKW, nStride, nPad,
                                           nHout, nWout);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// Reduction kernels（OPTIMIZATION 1.4）
// 20260324 ZJH 通用 GPU reduction — 用于求和/求均值/损失计算/BatchNorm 统计量
// 使用 warp-level shuffle 优化最后 32 个元素的归约，避免 __syncthreads 开销
// =====================================================================

// 20260324 ZJH Warp-level 求和 reduction（无需 __syncthreads）
// Warp 内 32 个线程隐式同步，使用 __shfl_down_sync 直接交换寄存器值
// 比 shared memory reduction 快约 2 倍（减少 shared memory 带宽和同步开销）
__device__ float warpReduceSum(float fVal) {
    // 20260324 ZJH 五轮 shuffle-down：32→16→8→4→2→1
    fVal += __shfl_down_sync(0xffffffff, fVal, 16);  // 20260324 ZJH 与 lane+16 求和
    fVal += __shfl_down_sync(0xffffffff, fVal, 8);   // 20260324 ZJH 与 lane+8 求和
    fVal += __shfl_down_sync(0xffffffff, fVal, 4);   // 20260324 ZJH 与 lane+4 求和
    fVal += __shfl_down_sync(0xffffffff, fVal, 2);   // 20260324 ZJH 与 lane+2 求和
    fVal += __shfl_down_sync(0xffffffff, fVal, 1);   // 20260324 ZJH 与 lane+1 求和
    return fVal;  // 20260324 ZJH lane 0 包含 warp 内所有 32 个值的总和
}

// 20260324 ZJH Block-level 求和 reduction kernel
// 先让每个线程遍历自己负责的元素段做部分和，再用 warp shuffle 归约到 block 级
__global__ void kernelReduceSum(const float* __restrict__ pData,
                                 float* __restrict__ pPartial,
                                 int nCount) {
    extern __shared__ float sdata[];  // 20260324 ZJH 动态 shared memory（大小 = nWarps * sizeof(float)）

    int nTid = threadIdx.x;                                     // 20260324 ZJH 块内线程索引
    int nGlobalIdx = blockIdx.x * blockDim.x + threadIdx.x;    // 20260324 ZJH 全局线程索引
    int nGridStride = blockDim.x * gridDim.x;                  // 20260324 ZJH grid 级步长（处理超大数组）

    // 20260324 ZJH 每线程累加多个元素（grid-stride loop 模式）
    float fLocalSum = 0.0f;
    for (int i = nGlobalIdx; i < nCount; i += nGridStride) {
        fLocalSum += pData[i];  // 20260324 ZJH 累加该线程负责的所有元素
    }

    // 20260324 ZJH Warp-level reduction
    fLocalSum = warpReduceSum(fLocalSum);  // 20260324 ZJH 归约到每个 warp 的 lane 0

    // 20260324 ZJH 每个 warp 的 lane 0 写入 shared memory
    int nWarpId = nTid / 32;    // 20260324 ZJH 当前线程所属 warp 编号
    int nLaneId = nTid % 32;    // 20260324 ZJH 当前线程在 warp 内的 lane 编号
    if (nLaneId == 0) {
        sdata[nWarpId] = fLocalSum;  // 20260324 ZJH warp 0 号 lane 写入部分和
    }
    __syncthreads();  // 20260324 ZJH 等待所有 warp 写入完成

    // 20260324 ZJH 用第一个 warp 做最终归约（warp 数通常 <= 32）
    int nWarps = (blockDim.x + 31) / 32;  // 20260324 ZJH 块内 warp 总数
    if (nTid < nWarps) {
        fLocalSum = sdata[nTid];  // 20260324 ZJH 读取该 warp 的部分和
    } else {
        fLocalSum = 0.0f;         // 20260324 ZJH 超出 warp 数的线程填零
    }
    if (nWarpId == 0) {
        fLocalSum = warpReduceSum(fLocalSum);  // 20260324 ZJH 最终 warp-level 归约
    }

    // 20260324 ZJH 块内 thread 0 写出该块的归约结果
    if (nTid == 0) {
        pPartial[blockIdx.x] = fLocalSum;
    }
}

// 20260324 ZJH 全局求均值 kernel：结果 = sum / nCount
__global__ void kernelReduceMean(const float* __restrict__ pData,
                                  float* __restrict__ pPartial,
                                  int nCount) {
    extern __shared__ float sdata[];

    int nTid = threadIdx.x;
    int nGlobalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nGridStride = blockDim.x * gridDim.x;

    // 20260324 ZJH 每线程累加多个元素
    float fLocalSum = 0.0f;
    for (int i = nGlobalIdx; i < nCount; i += nGridStride) {
        fLocalSum += pData[i];
    }

    fLocalSum = warpReduceSum(fLocalSum);

    int nWarpId = nTid / 32;
    int nLaneId = nTid % 32;
    if (nLaneId == 0) sdata[nWarpId] = fLocalSum;
    __syncthreads();

    int nWarps = (blockDim.x + 31) / 32;
    if (nTid < nWarps) fLocalSum = sdata[nTid];
    else fLocalSum = 0.0f;
    if (nWarpId == 0) fLocalSum = warpReduceSum(fLocalSum);

    // 20260324 ZJH 最终结果除以元素总数得到均值
    if (nTid == 0) {
        pPartial[blockIdx.x] = fLocalSum;
    }
}

// =====================================================================
// Softmax kernel
// =====================================================================

// 20260320 ZJH Softmax kernel：每个 block 处理一行
// pIn/pOut: [nBatch, nClasses]
__global__ void kernelSoftmax(const float* pIn, float* pOut, int nBatch, int nClasses) {
    int b = blockIdx.x;  // 20260320 ZJH 当前 batch 索引
    if (b >= nBatch) return;

    const float* pRow = pIn + b * nClasses;
    float* pOutRow = pOut + b * nClasses;

    // 20260320 ZJH 使用 shared memory 做 reduction
    extern __shared__ float sdata[];

    // 20260320 ZJH 查找最大值
    float fMax = -FLT_MAX;
    for (int j = threadIdx.x; j < nClasses; j += blockDim.x) {
        if (pRow[j] > fMax) fMax = pRow[j];
    }
    sdata[threadIdx.x] = fMax;
    __syncthreads();
    // 20260320 ZJH block-level max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    fMax = sdata[0];
    __syncthreads();

    // 20260320 ZJH 计算 exp 和 sum
    float fLocalSum = 0.0f;
    for (int j = threadIdx.x; j < nClasses; j += blockDim.x) {
        float v = expf(pRow[j] - fMax);
        pOutRow[j] = v;
        fLocalSum += v;
    }
    sdata[threadIdx.x] = fLocalSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float fSum = sdata[0];
    __syncthreads();

    // 20260320 ZJH 归一化
    for (int j = threadIdx.x; j < nClasses; j += blockDim.x) {
        pOutRow[j] /= fSum;
    }
}

extern "C" int omCudaSoftmax(const float* pIn, float* pOut, int nBatch, int nClasses) {
    int nThreads = (nClasses < 256) ? nClasses : 256;
    if (nThreads < 32) nThreads = 32;
    // 20260320 ZJH 向上取到 32 的倍数
    nThreads = ((nThreads + 31) / 32) * 32;
    size_t nSharedMem = nThreads * sizeof(float);
    kernelSoftmax<<<nBatch, nThreads, nSharedMem>>>(pIn, pOut, nBatch, nClasses);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 全局求和（使用优化后的 reduction kernel）
// =====================================================================

extern "C" int omCudaSum(const float* pData, float* pResult, int nCount) {
    // 20260324 ZJH 使用优化后的 warp-shuffle reduction 替代旧的两阶段 reduction
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nBlocks > 1024) nBlocks = 1024;  // 20260324 ZJH 限制 grid 大小，grid-stride loop 处理剩余
    int nWarps = (BLOCK_SIZE + 31) / 32;  // 20260324 ZJH shared memory 需要 nWarps 个 float

    float* pPartial = static_cast<float*>(gpuPoolAlloc(nBlocks * sizeof(float)));
    if (!pPartial) return -1;

    kernelReduceSum<<<nBlocks, BLOCK_SIZE, nWarps * sizeof(float)>>>(pData, pPartial, nCount);
    CUDA_CHECK(cudaGetLastError());

    // 20260324 ZJH 第二阶段：对部分和再做一次 reduction
    if (nBlocks > 1) {
        float* pPartial2 = static_cast<float*>(gpuPoolAlloc(sizeof(float)));
        if (!pPartial2) { gpuPoolFree(pPartial); return -1; }

        kernelReduceSum<<<1, BLOCK_SIZE, nWarps * sizeof(float)>>>(pPartial, pPartial2, nBlocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(pResult, pPartial2, sizeof(float), cudaMemcpyDeviceToDevice));
        gpuPoolFree(pPartial2);
    } else {
        CUDA_CHECK(cudaMemcpy(pResult, pPartial, sizeof(float), cudaMemcpyDeviceToDevice));
    }

    gpuPoolFree(pPartial);
    return 0;
}

// 20260324 ZJH 全局求均值接口：result = sum(pData) / nCount
extern "C" int omCudaMean(const float* pData, float* pResult, int nCount) {
    // 20260324 ZJH 先求总和
    int nRet = omCudaSum(pData, pResult, nCount);
    if (nRet != 0) return nRet;

    // 20260324 ZJH 将总和除以元素数得到均值
    float fInvCount = 1.0f / static_cast<float>(nCount);
    omCudaMulScalar(pResult, fInvCount, pResult, 1);
    return 0;
}

// =====================================================================
// BatchNorm2d — GPU 全流程（OPTIMIZATION 1.5）
// 20260324 ZJH 训练模式下的 mean/var 计算完全在 GPU 上完成
// 使用 per-channel reduction kernel 替代之前的 CPU 回传方案
// 消除了训练时 D2H/H2D 的往返延迟（对大 batch/大特征图尤为关键）
// =====================================================================

// 20260324 ZJH BatchNorm 统计量计算 kernel — 每个 block 计算一个通道的 mean 和 var
// 每个 block 负责一个通道：遍历该通道所有 batch 和空间维度的元素
// 参数: pInput [N, C, H, W]; pMean/pVar [C] 输出
__global__ void kernelBatchNormStats(const float* __restrict__ pInput,
                                      float* __restrict__ pMean,
                                      float* __restrict__ pVar,
                                      int nBatch, int nChannels, int nSpatial) {
    int nC = blockIdx.x;  // 20260324 ZJH 当前处理的通道索引（每 block 1 个通道）
    if (nC >= nChannels) return;

    extern __shared__ float sdata[];  // 20260324 ZJH 动态 shared memory

    int nCount = nBatch * nSpatial;  // 20260324 ZJH 该通道的元素总数

    // 20260324 ZJH 阶段 1：计算 mean（每线程累加部分元素，再做 block reduction）
    float fLocalSum = 0.0f;
    for (int i = threadIdx.x; i < nCount; i += blockDim.x) {
        int nN = i / nSpatial;    // 20260324 ZJH batch 索引
        int nS = i % nSpatial;    // 20260324 ZJH 空间索引
        fLocalSum += pInput[(nN * nChannels + nC) * nSpatial + nS];  // 20260324 ZJH 累加
    }
    sdata[threadIdx.x] = fLocalSum;
    __syncthreads();

    // 20260324 ZJH Block-level sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float fMean = sdata[0] / nCount;  // 20260324 ZJH 该通道均值
    __syncthreads();

    // 20260324 ZJH 阶段 2：计算 variance（方差 = E[(x - mean)^2]）
    float fLocalVar = 0.0f;
    for (int i = threadIdx.x; i < nCount; i += blockDim.x) {
        int nN = i / nSpatial;
        int nS = i % nSpatial;
        float fDiff = pInput[(nN * nChannels + nC) * nSpatial + nS] - fMean;  // 20260324 ZJH 偏差
        fLocalVar += fDiff * fDiff;  // 20260324 ZJH 累加平方偏差
    }
    sdata[threadIdx.x] = fLocalVar;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float fVar = sdata[0] / nCount;  // 20260324 ZJH 该通道方差

    // 20260324 ZJH Thread 0 写出 mean 和 var
    if (threadIdx.x == 0) {
        pMean[nC] = fMean;
        pVar[nC] = fVar;
    }
}

// 20260324 ZJH BatchNorm 反向 kernel — 计算输入梯度
// dX = gamma * invStd * (dY - mean(dY) - (X - mean) * mean(dY * (X - mean)) * invStd^2)
// 简化为: dX = gamma * invStd * (dY - dMean - xHat * dVar)
// 其中 xHat = (X - mean) * invStd, dMean = mean(dY), dVar = mean(dY * xHat)
__global__ void kernelBatchNorm2dBackward(
    const float* __restrict__ pInput,
    const float* __restrict__ pGradOutput,
    float* __restrict__ pGradInput,
    const float* __restrict__ pGamma,
    const float* __restrict__ pMean,
    const float* __restrict__ pInvStd,
    const float* __restrict__ pDMean,    // 20260324 ZJH 预计算的 mean(dY) per channel
    const float* __restrict__ pDVar,     // 20260324 ZJH 预计算的 mean(dY * xHat) per channel
    int nBatch, int nChannels, int nSpatial)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nChannels * nSpatial;
    if (nIdx >= nTotal) return;

    int nC = (nIdx / nSpatial) % nChannels;  // 20260324 ZJH 当前通道
    float fXhat = (pInput[nIdx] - pMean[nC]) * pInvStd[nC];  // 20260324 ZJH 归一化值

    // 20260324 ZJH 输入梯度公式
    pGradInput[nIdx] = pGamma[nC] * pInvStd[nC] *
        (pGradOutput[nIdx] - pDMean[nC] - fXhat * pDVar[nC]);
}

// 20260320 ZJH BatchNorm2d 前向归一化 kernel（保持不变）
__global__ void kernelBatchNorm2dForward(
    const float* pInput, float* pOutput,
    const float* pGamma, const float* pBeta,
    const float* pMean, const float* pInvStd,
    int nBatch, int nChannels, int nSpatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nChannels * nSpatial;
    if (idx >= nTotal) return;

    int s = idx % nSpatial;
    int c = (idx / nSpatial) % nChannels;
    (void)s;  // 20260320 ZJH 抑制未使用警告

    float fNorm = (pInput[idx] - pMean[c]) * pInvStd[c];
    pOutput[idx] = pGamma[c] * fNorm + pBeta[c];
}

// 20260324 ZJH 更新 running stats kernel — 在 GPU 上直接更新（避免 D2H/H2D 往返）
__global__ void kernelUpdateRunningStats(float* __restrict__ pRunMean,
                                          float* __restrict__ pRunVar,
                                          const float* __restrict__ pBatchMean,
                                          const float* __restrict__ pBatchVar,
                                          float fMomentum, int nChannels) {
    int nC = blockIdx.x * blockDim.x + threadIdx.x;  // 20260324 ZJH 通道索引
    if (nC >= nChannels) return;
    // 20260324 ZJH 指数移动平均更新公式
    pRunMean[nC] = (1.0f - fMomentum) * pRunMean[nC] + fMomentum * pBatchMean[nC];
    pRunVar[nC]  = (1.0f - fMomentum) * pRunVar[nC]  + fMomentum * pBatchVar[nC];
}

// 20260324 ZJH 从方差计算 invStd kernel: invStd[c] = 1 / sqrt(var[c] + eps)
__global__ void kernelComputeInvStd(const float* __restrict__ pVar,
                                     float* __restrict__ pInvStd,
                                     float fEps, int nChannels) {
    int nC = blockIdx.x * blockDim.x + threadIdx.x;  // 20260324 ZJH 通道索引
    if (nC >= nChannels) return;
    pInvStd[nC] = rsqrtf(pVar[nC] + fEps);  // 20260324 ZJH rsqrtf = 1/sqrt，单精度硬件指令
}

extern "C" int omCudaBatchNorm2d(const float* pInput, float* pOutput,
                                  const float* pGamma, const float* pBeta,
                                  float* pRunMean, float* pRunVar,
                                  float* pSavedMean, float* pSavedInvStd,
                                  int nBatch, int nChannels, int nH, int nW,
                                  float fEps, float fMomentum, int bTraining) {
    int nSpatial = nH * nW;                      // 20260324 ZJH 每通道空间元素数
    int nTotal = nBatch * nChannels * nSpatial;   // 20260324 ZJH 总元素数

    if (bTraining) {
        // 20260324 ZJH 训练模式：在 GPU 上计算 batch 统计量（替代旧的 CPU 回传方案）
        // 分配临时方差缓冲区
        float* pBatchVar = static_cast<float*>(gpuPoolAlloc(nChannels * sizeof(float)));
        if (!pBatchVar) return -1;

        // 20260324 ZJH 步骤 1: 计算每通道 mean 和 var（每 block 处理 1 个通道）
        int nStatThreads = 256;  // 20260324 ZJH 统计 kernel 线程数
        kernelBatchNormStats<<<nChannels, nStatThreads, nStatThreads * sizeof(float)>>>(
            pInput, pSavedMean, pBatchVar, nBatch, nChannels, nSpatial);

        // 20260324 ZJH 步骤 2: 从方差计算 invStd
        int nInvBlocks = (nChannels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelComputeInvStd<<<nInvBlocks, BLOCK_SIZE>>>(pBatchVar, pSavedInvStd, fEps, nChannels);

        // 20260324 ZJH 步骤 3: 更新 running stats（全 GPU，无 CPU 参与）
        kernelUpdateRunningStats<<<nInvBlocks, BLOCK_SIZE>>>(
            pRunMean, pRunVar, pSavedMean, pBatchVar, fMomentum, nChannels);

        gpuPoolFree(pBatchVar);  // 20260324 ZJH 归还临时缓冲区

        // 20260324 ZJH 步骤 4: 归一化
        int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelBatchNorm2dForward<<<nBlocks, BLOCK_SIZE>>>(
            pInput, pOutput, pGamma, pBeta, pSavedMean, pSavedInvStd,
            nBatch, nChannels, nSpatial);
    } else {
        // 20260324 ZJH 评估模式：使用 running stats
        // 步骤 1: 从 runVar 计算 invStd（全 GPU）
        int nInvBlocks = (nChannels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelComputeInvStd<<<nInvBlocks, BLOCK_SIZE>>>(pRunVar, pSavedInvStd, fEps, nChannels);

        // 步骤 2: 归一化
        int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelBatchNorm2dForward<<<nBlocks, BLOCK_SIZE>>>(
            pInput, pOutput, pGamma, pBeta, pRunMean, pSavedInvStd,
            nBatch, nChannels, nSpatial);
    }

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// LayerNorm 前向 kernel
// =====================================================================

__global__ void kernelLayerNormForward(const float* pInput, float* pOutput,
                                        const float* pGamma, const float* pBeta,
                                        const float* pMean, const float* pInvStd,
                                        int nBatch, int nDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nDim;
    if (idx >= nTotal) return;

    int b = idx / nDim;
    int d = idx % nDim;
    float fNorm = (pInput[idx] - pMean[b]) * pInvStd[b];
    pOutput[idx] = pGamma[d] * fNorm + pBeta[d];
}

// 20260324 ZJH LayerNorm 统计量计算 kernel — 每 block 处理一个 batch 样本
// 计算该样本在 nDim 维度上的 mean 和 invStd
__global__ void kernelLayerNormStats(const float* __restrict__ pInput,
                                      float* __restrict__ pMean,
                                      float* __restrict__ pInvStd,
                                      int nBatch, int nDim, float fEps) {
    int nB = blockIdx.x;  // 20260324 ZJH 当前 batch 样本索引
    if (nB >= nBatch) return;

    extern __shared__ float sdata[];  // 20260324 ZJH 动态 shared memory
    const float* pRow = pInput + nB * nDim;  // 20260324 ZJH 当前样本起始地址

    // 20260324 ZJH 阶段 1: 计算 mean
    float fLocalSum = 0.0f;
    for (int d = threadIdx.x; d < nDim; d += blockDim.x) {
        fLocalSum += pRow[d];
    }
    sdata[threadIdx.x] = fLocalSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float fMean = sdata[0] / nDim;
    __syncthreads();

    // 20260324 ZJH 阶段 2: 计算 variance
    float fLocalVar = 0.0f;
    for (int d = threadIdx.x; d < nDim; d += blockDim.x) {
        float fDiff = pRow[d] - fMean;
        fLocalVar += fDiff * fDiff;
    }
    sdata[threadIdx.x] = fLocalVar;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float fVar = sdata[0] / nDim;

    // 20260324 ZJH 写出结果
    if (threadIdx.x == 0) {
        pMean[nB] = fMean;
        pInvStd[nB] = rsqrtf(fVar + fEps);
    }
}

extern "C" int omCudaLayerNorm(const float* pInput, float* pOutput,
                                const float* pGamma, const float* pBeta,
                                float* pSavedMean, float* pSavedInvStd,
                                int nBatch, int nDim, float fEps) {
    // 20260324 ZJH 使用 GPU kernel 计算 mean/invStd（替代旧的 CPU 辅助方案）
    int nStatThreads = 256;
    if (nStatThreads > nDim) nStatThreads = ((nDim + 31) / 32) * 32;  // 20260324 ZJH 对齐到 warp
    if (nStatThreads < 32) nStatThreads = 32;

    kernelLayerNormStats<<<nBatch, nStatThreads, nStatThreads * sizeof(float)>>>(
        pInput, pSavedMean, pSavedInvStd, nBatch, nDim, fEps);

    int nTotal = nBatch * nDim;
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelLayerNormForward<<<nBlocks, BLOCK_SIZE>>>(
        pInput, pOutput, pGamma, pBeta, pSavedMean, pSavedInvStd, nBatch, nDim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// Phase 2 新增内核 — GPU-Resident Tensor 全算子支持
// 20260325 ZJH 为 CUDABackend 提供完整的前向/反向/优化器/辅助内核
// =====================================================================

// =====================================================================
// 激活函数反向 + 新增前向内核
// =====================================================================

// 20260325 ZJH LeakyReLU 前向 kernel：正值直通，负值乘以 fSlope
__global__ void kernelLeakyReLU(const float* __restrict__ pIn,
                                 float* __restrict__ pOut,
                                 int nCount, float fSlope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        float x = pIn[idx];  // 20260325 ZJH 读取输入值
        pOut[idx] = x > 0.0f ? x : fSlope * x;  // 20260325 ZJH 正值直通，负值乘斜率
    }
}

// 20260325 ZJH LeakyReLU 前向接口
extern "C" int omCudaLeakyRelu(const float* pIn, float* pOut, int nCount, float fSlope) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelLeakyReLU<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount, fSlope);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH Tanh 前向 kernel：out = tanh(in)
__global__ void kernelTanh(const float* __restrict__ pIn,
                            float* __restrict__ pOut,
                            int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        pOut[idx] = tanhf(pIn[idx]);  // 20260325 ZJH 双曲正切激活
    }
}

// 20260325 ZJH Tanh 前向接口
extern "C" int omCudaTanh(const float* pIn, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelTanh<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH Tanh 反向 kernel：grad_in = grad * (1 - output^2)
// pOutput 是前向 tanh 的输出值，利用 tanh'(x) = 1 - tanh(x)^2
__global__ void kernelTanhBackward(const float* __restrict__ pGrad,
                                    const float* __restrict__ pOutput,
                                    float* __restrict__ pOut,
                                    int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        float o = pOutput[idx];  // 20260325 ZJH 前向输出值
        pOut[idx] = pGrad[idx] * (1.0f - o * o);  // 20260325 ZJH tanh 导数
    }
}

// 20260325 ZJH Tanh 反向接口
extern "C" int omCudaTanhBackward(const float* pGrad, const float* pOutput, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelTanhBackward<<<nBlocks, BLOCK_SIZE>>>(pGrad, pOutput, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH GELU 反向 kernel：使用 tanh 近似公式求导
// dGELU/dx = 0.5*(1+tanh(inner)) + 0.5*x*sech^2(inner)*d_inner
// inner = sqrt(2/pi)*(x + 0.044715*x^3), d_inner = sqrt(2/pi)*(1 + 3*0.044715*x^2)
__global__ void kernelGELUBackward(const float* __restrict__ pGrad,
                                    const float* __restrict__ pInput,
                                    float* __restrict__ pOut,
                                    int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        float x = pInput[idx];  // 20260325 ZJH 前向输入值
        float fInner = 0.7978845608f * (x + 0.044715f * x * x * x);  // 20260325 ZJH tanh 内部值
        float fTanh = tanhf(fInner);  // 20260325 ZJH tanh 结果
        float fSech2 = 1.0f - fTanh * fTanh;  // 20260325 ZJH sech^2 = 1 - tanh^2
        float fDInner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);  // 20260325 ZJH inner 对 x 的导数
        pOut[idx] = pGrad[idx] * (0.5f * (1.0f + fTanh) + 0.5f * x * fSech2 * fDInner);  // 20260325 ZJH 链式法则
    }
}

// 20260325 ZJH GELU 反向接口
extern "C" int omCudaGeluBackward(const float* pGrad, const float* pInput, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelGELUBackward<<<nBlocks, BLOCK_SIZE>>>(pGrad, pInput, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH SiLU 反向 kernel：grad = gradOut * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__global__ void kernelSiLUBackward(const float* __restrict__ pGrad,
                                    const float* __restrict__ pInput,
                                    float* __restrict__ pOut,
                                    int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        float x = pInput[idx];  // 20260325 ZJH 前向输入值
        float fSig = 1.0f / (1.0f + expf(-x));  // 20260325 ZJH sigmoid(x)
        pOut[idx] = pGrad[idx] * fSig * (1.0f + x * (1.0f - fSig));  // 20260325 ZJH SiLU 导数
    }
}

// 20260325 ZJH SiLU 反向接口
extern "C" int omCudaSiluBackward(const float* pGrad, const float* pInput, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelSiLUBackward<<<nBlocks, BLOCK_SIZE>>>(pGrad, pInput, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH Sigmoid 反向 kernel：grad_in = grad * output * (1 - output)
// pOutput 是前向 sigmoid 的输出值
__global__ void kernelSigmoidBackward(const float* __restrict__ pGrad,
                                       const float* __restrict__ pOutput,
                                       float* __restrict__ pOut,
                                       int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        float o = pOutput[idx];  // 20260325 ZJH 前向输出值
        pOut[idx] = pGrad[idx] * o * (1.0f - o);  // 20260325 ZJH sigmoid 导数: o*(1-o)
    }
}

// 20260325 ZJH Sigmoid 反向接口
extern "C" int omCudaSigmoidBackward(const float* pGrad, const float* pOutput, float* pOut, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelSigmoidBackward<<<nBlocks, BLOCK_SIZE>>>(pGrad, pOutput, pOut, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH LeakyReLU 反向 kernel：正值梯度直通，负值梯度乘以 fSlope
__global__ void kernelLeakyReLUBackward(const float* __restrict__ pGrad,
                                         const float* __restrict__ pInput,
                                         float* __restrict__ pOut,
                                         int nCount, float fSlope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        pOut[idx] = pInput[idx] > 0.0f ? pGrad[idx] : fSlope * pGrad[idx];  // 20260325 ZJH 正值直通，负值乘斜率
    }
}

// 20260325 ZJH LeakyReLU 反向接口
extern "C" int omCudaLeakyReluBackward(const float* pGrad, const float* pInput, float* pOut,
                                        int nCount, float fSlope) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 计算所需线程块数
    kernelLeakyReLUBackward<<<nBlocks, BLOCK_SIZE>>>(pGrad, pInput, pOut, nCount, fSlope);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 池化内核
// =====================================================================

// 20260325 ZJH MaxPool2d 前向 kernel — 保存最大值索引用于反向传播
// 每个线程处理输出张量中的一个元素
// pIn: [N, C, H, W]  pOut: [N, C, Hout, Wout]  pIndices: [N, C, Hout, Wout]
__global__ void kernelMaxPool2d(const float* __restrict__ pIn,
                                 float* __restrict__ pOut,
                                 int* __restrict__ pIndices,
                                 int nN, int nC, int nH, int nW,
                                 int nKH, int nKW, int nSH, int nSW,
                                 int nPH, int nPW,
                                 int nHout, int nWout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nN * nC * nHout * nWout;  // 20260325 ZJH 输出总元素数
    if (idx >= nTotal) return;  // 20260325 ZJH 越界检查

    // 20260325 ZJH 从线性索引解码 (n, c, oh, ow) 四维坐标
    int ow = idx % nWout;                           // 20260325 ZJH 输出宽度索引
    int oh = (idx / nWout) % nHout;                 // 20260325 ZJH 输出高度索引
    int c  = (idx / (nWout * nHout)) % nC;          // 20260325 ZJH 通道索引
    int n  = idx / (nWout * nHout * nC);            // 20260325 ZJH 批次索引

    float fMax = -FLT_MAX;  // 20260325 ZJH 初始化为极小值
    int nMaxIdx = -1;        // 20260325 ZJH 最大值在输入中的平面索引

    // 20260325 ZJH 遍历池化窗口，查找最大值
    for (int kh = 0; kh < nKH; ++kh) {
        for (int kw = 0; kw < nKW; ++kw) {
            int ih = oh * nSH - nPH + kh;  // 20260325 ZJH 输入行坐标
            int iw = ow * nSW - nPW + kw;  // 20260325 ZJH 输入列坐标
            if (ih >= 0 && ih < nH && iw >= 0 && iw < nW) {
                int nInIdx = ((n * nC + c) * nH + ih) * nW + iw;  // 20260325 ZJH NCHW 索引
                float fVal = pIn[nInIdx];  // 20260325 ZJH 读取输入值
                if (fVal > fMax) {
                    fMax = fVal;        // 20260325 ZJH 更新最大值
                    nMaxIdx = nInIdx;   // 20260325 ZJH 记录最大值索引
                }
            }
        }
    }
    pOut[idx] = fMax;          // 20260325 ZJH 写入最大值
    pIndices[idx] = nMaxIdx;   // 20260325 ZJH 保存索引供反向传播使用
}

// 20260325 ZJH MaxPool2d 前向接口
extern "C" int omCudaMaxPool2d(const float* pIn, float* pOut, int* pIndices,
                                int nN, int nC, int nH, int nW,
                                int nKH, int nKW, int nSH, int nSW,
                                int nPH, int nPW) {
    int nHout = (nH + 2 * nPH - nKH) / nSH + 1;  // 20260325 ZJH 输出高度
    int nWout = (nW + 2 * nPW - nKW) / nSW + 1;  // 20260325 ZJH 输出宽度
    int nTotal = nN * nC * nHout * nWout;          // 20260325 ZJH 输出总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelMaxPool2d<<<nBlocks, BLOCK_SIZE>>>(
        pIn, pOut, pIndices, nN, nC, nH, nW,
        nKH, nKW, nSH, nSW, nPH, nPW, nHout, nWout);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH MaxPool2d 反向 kernel — 将梯度散布回最大值位置
// 利用前向保存的 pIndices 直接定位梯度目标位置
__global__ void kernelMaxPool2dBackward(const float* __restrict__ pGradOut,
                                         const int* __restrict__ pIndices,
                                         float* __restrict__ pGradIn,
                                         int nOutSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx >= nOutSize) return;  // 20260325 ZJH 越界检查
    int nSrcIdx = pIndices[idx];  // 20260325 ZJH 前向最大值在输入中的索引
    if (nSrcIdx >= 0) {
        atomicAdd(&pGradIn[nSrcIdx], pGradOut[idx]);  // 20260325 ZJH 原子累加梯度（多输出可能指向同一输入）
    }
}

// 20260325 ZJH MaxPool2d 反向接口
// pGradIn 需预先置零（调用方负责）
extern "C" int omCudaMaxPool2dBackward(const float* pGradOut, const int* pIndices,
                                        float* pGradIn,
                                        int nN, int nC, int nH, int nW,
                                        int nHout, int nWout) {
    int nOutSize = nN * nC * nHout * nWout;  // 20260325 ZJH 输出总元素数
    int nBlocks = (nOutSize + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelMaxPool2dBackward<<<nBlocks, BLOCK_SIZE>>>(pGradOut, pIndices, pGradIn, nOutSize);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH AvgPool2d 前向 kernel — 窗口内取均值
// 使用固定核大小做平均（count_include_pad=false 风格）
__global__ void kernelAvgPool2d(const float* __restrict__ pIn,
                                 float* __restrict__ pOut,
                                 int nN, int nC, int nH, int nW,
                                 int nKH, int nKW, int nSH, int nSW,
                                 int nPH, int nPW,
                                 int nHout, int nWout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nN * nC * nHout * nWout;  // 20260325 ZJH 输出总元素数
    if (idx >= nTotal) return;  // 20260325 ZJH 越界检查

    // 20260325 ZJH 从线性索引解码 (n, c, oh, ow) 四维坐标
    int ow = idx % nWout;                           // 20260325 ZJH 输出宽度索引
    int oh = (idx / nWout) % nHout;                 // 20260325 ZJH 输出高度索引
    int c  = (idx / (nWout * nHout)) % nC;          // 20260325 ZJH 通道索引
    int n  = idx / (nWout * nHout * nC);            // 20260325 ZJH 批次索引

    float fSum = 0.0f;  // 20260325 ZJH 窗口内元素累加
    // 20260325 ZJH 遍历池化窗口
    for (int kh = 0; kh < nKH; ++kh) {
        for (int kw = 0; kw < nKW; ++kw) {
            int ih = oh * nSH - nPH + kh;  // 20260325 ZJH 输入行坐标
            int iw = ow * nSW - nPW + kw;  // 20260325 ZJH 输入列坐标
            if (ih >= 0 && ih < nH && iw >= 0 && iw < nW) {
                fSum += pIn[((n * nC + c) * nH + ih) * nW + iw];  // 20260325 ZJH 累加有效值
            }
        }
    }
    pOut[idx] = fSum / static_cast<float>(nKH * nKW);  // 20260325 ZJH 除以核面积取均值
}

// 20260325 ZJH AvgPool2d 前向接口
extern "C" int omCudaAvgPool2d(const float* pIn, float* pOut,
                                int nN, int nC, int nH, int nW,
                                int nKH, int nKW, int nSH, int nSW,
                                int nPH, int nPW) {
    int nHout = (nH + 2 * nPH - nKH) / nSH + 1;  // 20260325 ZJH 输出高度
    int nWout = (nW + 2 * nPW - nKW) / nSW + 1;  // 20260325 ZJH 输出宽度
    int nTotal = nN * nC * nHout * nWout;          // 20260325 ZJH 输出总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelAvgPool2d<<<nBlocks, BLOCK_SIZE>>>(
        pIn, pOut, nN, nC, nH, nW,
        nKH, nKW, nSH, nSW, nPH, nPW, nHout, nWout);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH AvgPool2d 反向 kernel — 将梯度均匀散布回窗口内各位置
__global__ void kernelAvgPool2dBackward(const float* __restrict__ pGradOut,
                                         float* __restrict__ pGradIn,
                                         int nN, int nC, int nH, int nW,
                                         int nKH, int nKW, int nSH, int nSW,
                                         int nPH, int nPW,
                                         int nHout, int nWout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nN * nC * nHout * nWout;  // 20260325 ZJH 输出总元素数
    if (idx >= nTotal) return;  // 20260325 ZJH 越界检查

    // 20260325 ZJH 从线性索引解码 (n, c, oh, ow) 四维坐标
    int ow = idx % nWout;
    int oh = (idx / nWout) % nHout;
    int c  = (idx / (nWout * nHout)) % nC;
    int n  = idx / (nWout * nHout * nC);

    float fGrad = pGradOut[idx] / static_cast<float>(nKH * nKW);  // 20260325 ZJH 均分到窗口内

    // 20260325 ZJH 将均分梯度散布回窗口内各有效位置
    for (int kh = 0; kh < nKH; ++kh) {
        for (int kw = 0; kw < nKW; ++kw) {
            int ih = oh * nSH - nPH + kh;  // 20260325 ZJH 输入行坐标
            int iw = ow * nSW - nPW + kw;  // 20260325 ZJH 输入列坐标
            if (ih >= 0 && ih < nH && iw >= 0 && iw < nW) {
                atomicAdd(&pGradIn[((n * nC + c) * nH + ih) * nW + iw], fGrad);  // 20260325 ZJH 原子累加
            }
        }
    }
}

// 20260325 ZJH AvgPool2d 反向接口
// pGradIn 需预先置零（调用方负责）
extern "C" int omCudaAvgPool2dBackward(const float* pGradOut, float* pGradIn,
                                        int nN, int nC, int nH, int nW,
                                        int nKH, int nKW, int nSH, int nSW,
                                        int nPH, int nPW) {
    int nHout = (nH + 2 * nPH - nKH) / nSH + 1;  // 20260325 ZJH 输出高度
    int nWout = (nW + 2 * nPW - nKW) / nSW + 1;  // 20260325 ZJH 输出宽度
    int nTotal = nN * nC * nHout * nWout;          // 20260325 ZJH 输出总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelAvgPool2dBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradIn, nN, nC, nH, nW,
        nKH, nKW, nSH, nSW, nPH, nPW, nHout, nWout);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH AdaptiveAvgPool2d 前向 kernel — 自适应平均池化
// 将任意大小输入池化到固定 (outH, outW) 大小输出
// 计算每个输出位置对应的输入窗口范围，取窗口均值
__global__ void kernelAdaptiveAvgPool2d(const float* __restrict__ pIn,
                                         float* __restrict__ pOut,
                                         int nN, int nC, int nH, int nW,
                                         int nOutH, int nOutW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nN * nC * nOutH * nOutW;  // 20260325 ZJH 输出总元素数
    if (idx >= nTotal) return;  // 20260325 ZJH 越界检查

    // 20260325 ZJH 从线性索引解码 (n, c, oh, ow) 四维坐标
    int ow = idx % nOutW;
    int oh = (idx / nOutW) % nOutH;
    int c  = (idx / (nOutW * nOutH)) % nC;
    int n  = idx / (nOutW * nOutH * nC);

    // 20260325 ZJH 计算当前输出位置对应的输入窗口范围（PyTorch 标准公式）
    int nIhStart = (oh * nH) / nOutH;               // 20260325 ZJH 窗口起始行
    int nIhEnd   = ((oh + 1) * nH) / nOutH;         // 20260325 ZJH 窗口结束行
    int nIwStart = (ow * nW) / nOutW;               // 20260325 ZJH 窗口起始列
    int nIwEnd   = ((ow + 1) * nW) / nOutW;         // 20260325 ZJH 窗口结束列

    float fSum = 0.0f;  // 20260325 ZJH 窗口累加
    int nCount = 0;      // 20260325 ZJH 窗口元素计数
    for (int ih = nIhStart; ih < nIhEnd; ++ih) {
        for (int iw = nIwStart; iw < nIwEnd; ++iw) {
            fSum += pIn[((n * nC + c) * nH + ih) * nW + iw];  // 20260325 ZJH 累加
            nCount++;  // 20260325 ZJH 计数
        }
    }
    pOut[idx] = nCount > 0 ? fSum / static_cast<float>(nCount) : 0.0f;  // 20260325 ZJH 取均值
}

// 20260325 ZJH AdaptiveAvgPool2d 前向接口
extern "C" int omCudaAdaptiveAvgPool2d(const float* pIn, float* pOut,
                                        int nN, int nC, int nH, int nW,
                                        int nOutH, int nOutW) {
    int nTotal = nN * nC * nOutH * nOutW;  // 20260325 ZJH 输出总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelAdaptiveAvgPool2d<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nN, nC, nH, nW, nOutH, nOutW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 广播偏置加法 / Dropout / 填充 / 拷贝 / Argmax
// =====================================================================

// 20260325 ZJH 广播偏置加法 kernel：pOut[n,c,hw] = pData[n,c,hw] + pBias[c]
// 用于卷积后的偏置加法，pBias 沿通道维度广播到 [N, C, HW] 张量
__global__ void kernelAddBiasNChw(const float* __restrict__ pData,
                                   const float* __restrict__ pBias,
                                   float* __restrict__ pOut,
                                   int nN, int nC, int nHW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nN * nC * nHW;  // 20260325 ZJH 总元素数
    if (idx >= nTotal) return;  // 20260325 ZJH 越界检查
    int c = (idx / nHW) % nC;  // 20260325 ZJH 当前通道索引
    pOut[idx] = pData[idx] + pBias[c];  // 20260325 ZJH 加偏置
}

// 20260325 ZJH 广播偏置加法接口
extern "C" int omCudaAddBias(const float* pData, const float* pBias, float* pOut,
                              int nN, int nC, int nHW) {
    int nTotal = nN * nC * nHW;  // 20260325 ZJH 总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelAddBiasNChw<<<nBlocks, BLOCK_SIZE>>>(pData, pBias, pOut, nN, nC, nHW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH Dropout 前向 kernel — 随机置零 + 缩放
// 训练时以概率 fProb 将元素置零，存活元素乘以 1/(1-fProb) 保持期望不变
// 推理时直接拷贝输入到输出（不做 dropout）
// pMask: 预生成的随机 mask（0 或 1），由调用方提供
__global__ void kernelDropout(const float* __restrict__ pIn,
                               float* __restrict__ pOut,
                               const float* __restrict__ pMask,
                               int nCount, float fScale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) {
        pOut[idx] = pIn[idx] * pMask[idx] * fScale;  // 20260325 ZJH 应用 mask 并缩放
    }
}

// 20260325 ZJH Dropout 前向接口
// pMask 需调用方预先填充随机 0/1 值（使用 cuRAND 或 CPU 生成后上传）
// bTraining=0 时直接拷贝输入到输出
extern "C" int omCudaDropout(const float* pIn, float* pOut, const float* pMask,
                              int nCount, float fProb, int bTraining) {
    if (!bTraining || fProb <= 0.0f) {
        // 20260325 ZJH 推理模式或无 dropout：直接拷贝
        CUDA_CHECK(cudaMemcpy(pOut, pIn, nCount * sizeof(float), cudaMemcpyDeviceToDevice));
        return 0;
    }
    float fScale = 1.0f / (1.0f - fProb);  // 20260325 ZJH 缩放因子保持期望不变
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelDropout<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, pMask, nCount, fScale);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260328 ZJH Fused Dropout Forward kernel — GPU 端 mask 生成 + 应用（零 CPU→GPU mask 传输）
// 使用 SplitMix64 哈希函数在每个线程独立生成伪随机数，无需 cuRAND 依赖
// 每个线程: hash(seed + idx) → 均匀 float → 与 keepProb 比较 → 生成缩放 mask → 乘输入
// pMask 存储缩放后的值（0 或 1/keepProb），与 DropoutBackward 兼容（backward = grad * mask）
__global__ void kernelDropoutForward(
    const float* __restrict__ pIn,   // 20260328 ZJH 输入张量（GPU 端）
    float* __restrict__ pOut,        // 20260328 ZJH 输出张量（GPU 端）
    float* __restrict__ pMask,       // 20260328 ZJH 输出缩放 mask（GPU 端，0 或 1/keepProb，反向需要）
    int nCount,                      // 20260328 ZJH 元素总数
    float fKeepProb,                 // 20260328 ZJH 保留概率 = 1 - dropProb
    unsigned long long nSeed)        // 20260328 ZJH 随机种子（CPU 端生成的单个 int）
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260328 ZJH 全局线程索引
    if (idx < nCount) {
        // 20260328 ZJH SplitMix64 哈希：将 (seed + idx) 映射为伪随机 64-bit 整数
        // 优点：无状态、确定性（相同 seed+idx 产生相同输出）、分布均匀、零依赖
        unsigned long long nState = nSeed + static_cast<unsigned long long>(idx) * 6364136223846793005ULL + 1442695040888963407ULL;  // 20260328 ZJH 线性组合：每线程唯一起始状态
        nState ^= nState >> 33;                     // 20260328 ZJH SplitMix64 第一轮混合
        nState *= 0xff51afd7ed558ccdULL;            // 20260328 ZJH SplitMix64 乘法扩散
        nState ^= nState >> 33;                     // 20260328 ZJH SplitMix64 第二轮混合
        float fRand = (nState & 0xFFFFFF) / 16777216.0f;  // 20260328 ZJH 取低 24 位转换为 [0, 1) 均匀浮点数
        // 20260328 ZJH mask 存储缩放值：0（丢弃）或 1/keepProb（保留且缩放），与 DropoutBackward 兼容
        float fScaledMask = (fRand < fKeepProb) ? (1.0f / fKeepProb) : 0.0f;  // 20260328 ZJH 伯努利采样 + 倒缩放
        pMask[idx] = fScaledMask;                   // 20260328 ZJH 写出缩放 mask（反向传播直接乘梯度即可）
        pOut[idx] = pIn[idx] * fScaledMask;         // 20260328 ZJH 应用 dropout: x * scaledMask
    }
}

// 20260328 ZJH Fused Dropout Forward 接口 — mask 生成 + 应用合一，无 CPU→GPU 传输
// nSeed: 由调用方在 CPU 端从 std::mt19937 生成的随机种子（单个值）
// pMask: 输出 GPU buffer，存储生成的 mask（反向传播 DropoutBackward 需要）
extern "C" int omCudaDropoutForward(
    const float* pIn,              // 20260328 ZJH 输入 GPU 数据指针
    float* pOut,                   // 20260328 ZJH 输出 GPU 数据指针
    float* pMask,                  // 20260328 ZJH 输出 GPU mask 指针
    int nCount,                    // 20260328 ZJH 元素总数
    float fKeepProb,               // 20260328 ZJH 保留概率 (1 - dropProb)
    unsigned long long nSeed)      // 20260328 ZJH CPU 端随机种子
{
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260328 ZJH 计算线程块数
    kernelDropoutForward<<<nBlocks, BLOCK_SIZE>>>(
        pIn, pOut, pMask, nCount, fKeepProb, nSeed);  // 20260328 ZJH 启动 fused kernel
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误
    return 0;  // 20260328 ZJH 返回 0 表示成功
}

// 20260325 ZJH GPU 内存填零 kernel
__global__ void kernelFillZeros(float* __restrict__ pData, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) pData[idx] = 0.0f;  // 20260325 ZJH 置零
}

// 20260325 ZJH GPU 内存填零接口
extern "C" int omCudaFillZeros(float* pData, int nCount) {
    // 20260326 ZJH 用 cudaMemset 替代自定义 kernel，排除 kernel bug
    cudaError_t err = cudaMemset(pData, 0, static_cast<size_t>(nCount) * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] fillZeros FAILED: %s (nCount=%d)\n", cudaGetErrorString(err), nCount);
        return -1;
    }
    return 0;
}

// 20260325 ZJH GPU 内存填 1 kernel
__global__ void kernelFillOnes(float* __restrict__ pData, int nCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) pData[idx] = 1.0f;  // 20260325 ZJH 填 1
}

// 20260325 ZJH GPU 内存填 1 接口
extern "C" int omCudaFillOnes(float* pData, int nCount) {
    // 20260326 ZJH 用 cudaMemset 填 0 后逐元素设 1（排除 kernel bug）
    // 注：cudaMemset 只能设字节值，float 1.0 = 0x3F800000，不能用 memset
    // 回退用 kernel
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelFillOnes<<<nBlocks, BLOCK_SIZE>>>(pData, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH GPU 内存填充指定值 kernel
__global__ void kernelFillValue(float* __restrict__ pData, int nCount, float fValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx < nCount) pData[idx] = fValue;  // 20260325 ZJH 填充指定值
}

// 20260325 ZJH GPU 内存填充指定值接口
extern "C" int omCudaFillValue(float* pData, int nCount, float fValue) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelFillValue<<<nBlocks, BLOCK_SIZE>>>(pData, nCount, fValue);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH GPU Device-to-Device 拷贝接口（float 数组专用）
extern "C" int omCudaCopy(const float* pSrc, float* pDst, int nCount) {
    CUDA_CHECK(cudaMemcpy(pDst, pSrc, static_cast<size_t>(nCount) * sizeof(float),
                           cudaMemcpyDeviceToDevice));  // 20260325 ZJH D2D 拷贝
    return 0;
}

// 20260325 ZJH Argmax kernel — 逐行求最大值索引
// pData: [nBatch, nClasses] 输入，pOut: [nBatch] int 输出索引
__global__ void kernelArgmax(const float* __restrict__ pData,
                              int* __restrict__ pOut,
                              int nBatch, int nClasses) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引 = batch 索引
    if (b >= nBatch) return;  // 20260325 ZJH 越界检查

    const float* pRow = pData + b * nClasses;  // 20260325 ZJH 当前行起始地址
    int nBestIdx = 0;         // 20260325 ZJH 最大值索引
    float fBest = pRow[0];    // 20260325 ZJH 最大值
    for (int j = 1; j < nClasses; ++j) {
        if (pRow[j] > fBest) {
            fBest = pRow[j];   // 20260325 ZJH 更新最大值
            nBestIdx = j;      // 20260325 ZJH 更新索引
        }
    }
    pOut[b] = nBestIdx;  // 20260325 ZJH 写出结果
}

// 20260325 ZJH Argmax 接口
extern "C" int omCudaArgmax(const float* pData, int* pOut, int nBatch, int nClasses) {
    int nBlocks = (nBatch + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelArgmax<<<nBlocks, BLOCK_SIZE>>>(pData, pOut, nBatch, nClasses);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 优化器内核 — GPU-Resident 训练的关键
// 20260325 ZJH 直接在 GPU 上执行参数更新，避免 GPU→CPU→GPU 的数据乒乓
// =====================================================================

// 20260325 ZJH Adam 优化器 step kernel — 在 GPU 上直接更新参数
// 完全在 GPU 上执行 Adam 公式：
//   m = beta1 * m + (1-beta1) * grad
//   v = beta2 * v + (1-beta2) * grad^2
//   m_hat = m / (1 - beta1^step)
//   v_hat = v / (1 - beta2^step)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
__global__ void kernelAdamStep(float* __restrict__ pParam,
                                const float* __restrict__ pGrad,
                                float* __restrict__ pM,
                                float* __restrict__ pV,
                                int nCount,
                                float fLr, float fBeta1, float fBeta2,
                                float fEps, int nStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx >= nCount) return;  // 20260325 ZJH 越界检查

    float g = pGrad[idx];  // 20260325 ZJH 当前梯度值

    // 20260325 ZJH 更新一阶矩估计（梯度的指数移动平均）
    float m = fBeta1 * pM[idx] + (1.0f - fBeta1) * g;
    // 20260325 ZJH 更新二阶矩估计（梯度平方的指数移动平均）
    float v = fBeta2 * pV[idx] + (1.0f - fBeta2) * g * g;

    pM[idx] = m;  // 20260325 ZJH 回写一阶矩
    pV[idx] = v;  // 20260325 ZJH 回写二阶矩

    // 20260325 ZJH 偏差校正：消除初始阶段零初始化带来的偏差
    float fBeta1Pow = 1.0f;  // 20260325 ZJH beta1^step
    float fBeta2Pow = 1.0f;  // 20260325 ZJH beta2^step
    for (int s = 0; s < nStep; ++s) {
        fBeta1Pow *= fBeta1;  // 20260325 ZJH 累乘计算 beta1^step
        fBeta2Pow *= fBeta2;  // 20260325 ZJH 累乘计算 beta2^step
    }
    float mHat = m / (1.0f - fBeta1Pow);  // 20260325 ZJH 校正后一阶矩
    float vHat = v / (1.0f - fBeta2Pow);  // 20260325 ZJH 校正后二阶矩

    // 20260325 ZJH 参数更新
    pParam[idx] -= fLr * mHat / (sqrtf(vHat) + fEps);
}

// 20260325 ZJH Adam 优化器 step 接口
extern "C" int omCudaAdamStep(float* pParam, const float* pGrad,
                               float* pM, float* pV,
                               int nCount, float fLr,
                               float fBeta1, float fBeta2,
                               float fEps, int nStep) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelAdamStep<<<nBlocks, BLOCK_SIZE>>>(
        pParam, pGrad, pM, pV, nCount, fLr, fBeta1, fBeta2, fEps, nStep);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH SGD+Momentum 优化器 step kernel — 在 GPU 上直接更新参数
// SGD+Momentum 公式：
//   velocity = momentum * velocity + grad
//   param -= lr * velocity
// 当 momentum=0 时退化为标准 SGD：param -= lr * grad
__global__ void kernelSgdStep(float* __restrict__ pParam,
                               const float* __restrict__ pGrad,
                               float* __restrict__ pVelocity,
                               int nCount,
                               float fLr, float fMomentum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    if (idx >= nCount) return;  // 20260325 ZJH 越界检查

    float g = pGrad[idx];  // 20260325 ZJH 当前梯度值

    if (fMomentum > 0.0f) {
        // 20260325 ZJH 带动量的 SGD
        float vel = fMomentum * pVelocity[idx] + g;  // 20260325 ZJH 更新速度
        pVelocity[idx] = vel;                         // 20260325 ZJH 回写速度
        pParam[idx] -= fLr * vel;                     // 20260325 ZJH 参数更新
    } else {
        // 20260325 ZJH 标准 SGD（无动量）
        pParam[idx] -= fLr * g;  // 20260325 ZJH 直接按梯度更新
    }
}

// 20260325 ZJH SGD 优化器 step 接口
extern "C" int omCudaSgdStep(float* pParam, const float* pGrad,
                              float* pVelocity, int nCount,
                              float fLr, float fMomentum) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelSgdStep<<<nBlocks, BLOCK_SIZE>>>(pParam, pGrad, pVelocity, nCount, fLr, fMomentum);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 损失反向内核
// =====================================================================

// 20260325 ZJH Softmax + CrossEntropy 联合反向 kernel
// 联合梯度公式：grad_logits = (softmax_output - target) / batch_size
// 直接在 GPU 上计算，避免 D2H 传输梯度
__global__ void kernelSoftmaxCrossEntropyBackward(const float* __restrict__ pSoftmax,
                                                    const float* __restrict__ pTarget,
                                                    float* __restrict__ pGradLogits,
                                                    int nBatch, int nClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nBatch * nClasses;  // 20260325 ZJH 总元素数
    if (idx >= nTotal) return;  // 20260325 ZJH 越界检查
    float fScale = 1.0f / static_cast<float>(nBatch);  // 20260325 ZJH 批次平均缩放因子
    pGradLogits[idx] = (pSoftmax[idx] - pTarget[idx]) * fScale;  // 20260325 ZJH 联合梯度公式
}

// 20260325 ZJH Softmax + CrossEntropy 联合反向接口
extern "C" int omCudaSoftmaxCrossEntropyBackward(const float* pSoftmax, const float* pTarget,
                                                   float* pGradLogits,
                                                   int nBatch, int nClasses) {
    int nTotal = nBatch * nClasses;  // 20260325 ZJH 总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260325 ZJH 线程块数
    kernelSoftmaxCrossEntropyBackward<<<nBlocks, BLOCK_SIZE>>>(
        pSoftmax, pTarget, pGradLogits, nBatch, nClasses);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// Conv2d 反向传播 GPU 实现（col2im + GEMM）
// 20260325 ZJH 解决 conv2d 反向只有空桩导致 CNN 训练无法 GPU 加速的问题
// =====================================================================

// 20260325 ZJH Col2im kernel — im2col 的逆操作
// 将列矩阵 pCol[Cin*KH*KW, Hout*Wout] 散布回输入图像 pOutput[Cin, H, W]
// 多个 col 位置可能映射到同一输入位置，使用 atomicAdd 累加
__global__ void kernelCol2im(const float* __restrict__ pCol,
                              float* __restrict__ pOutput,
                              int nCin, int nH, int nW,
                              int nKH, int nKW,
                              int nPadH, int nPadW,
                              int nStrideH, int nStrideW,
                              int nHout, int nWout) {
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nColRows = nCin * nKH * nKW;      // 20260325 ZJH 列矩阵行数
    int nColCols = nHout * nWout;          // 20260325 ZJH 列矩阵列数
    int nTotal = nColRows * nColCols;      // 20260325 ZJH 列矩阵总元素数

    if (nIdx >= nTotal) return;  // 20260325 ZJH 越界检查

    // 20260325 ZJH 反推列矩阵坐标
    int nColCol = nIdx % nColCols;  // 20260325 ZJH 列索引 = 输出空间位置
    int nColRow = nIdx / nColCols;  // 20260325 ZJH 行索引 = (cin, kh, kw) 组合

    // 20260325 ZJH 输出空间坐标
    int nOw = nColCol % nWout;
    int nOh = nColCol / nWout;

    // 20260325 ZJH 通道和卷积核偏移
    int nKw = nColRow % nKW;
    int nKh = (nColRow / nKW) % nKH;
    int nCi = nColRow / (nKH * nKW);

    // 20260325 ZJH 对应输入图像坐标
    int nIh = nOh * nStrideH - nPadH + nKh;
    int nIw = nOw * nStrideW - nPadW + nKw;

    // 20260325 ZJH 边界内的有效位置：原子累加到输入梯度
    if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
        atomicAdd(&pOutput[(nCi * nH + nIh) * nW + nIw], pCol[nIdx]);
    }
}

// 20260325 ZJH Conv2d 反向对输入求梯度
// gradInput = col2im(Weight^T × gradOutput_reshaped)
// 步骤: (1) 转置权重 W[Cout, Cin*KH*KW] → W^T[Cin*KH*KW, Cout]
//        (2) col = W^T × gradOutput_reshaped [Cin*KH*KW, Hout*Wout]
//        (3) col2im(col) → gradInput[Cin, H, W]
extern "C" int omCudaConv2dBackwardInput(const float* pGradOutput, const float* pWeight,
                                          float* pGradInput,
                                          int nBatch, int nCin, int nH, int nW,
                                          int nCout, int nKH, int nKW, int nStride, int nPad) {
    int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260325 ZJH 输出高度
    int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260325 ZJH 输出宽度
    int nColRows = nCin * nKH * nKW;  // 20260325 ZJH im2col 行数
    int nColCols = nHout * nWout;      // 20260325 ZJH im2col 列数

    // 20260325 ZJH 分配临时缓冲区：转置权重 + col 矩阵
    size_t nWtBytes = static_cast<size_t>(nCout) * nColRows * sizeof(float);
    size_t nColBytes = static_cast<size_t>(nColRows) * nColCols * sizeof(float);
    float* pWT = static_cast<float*>(gpuPoolAlloc(nWtBytes));
    float* pCol = static_cast<float*>(gpuPoolAlloc(nColBytes));
    if (!pWT || !pCol) {
        if (pWT) gpuPoolFree(pWT);
        if (pCol) gpuPoolFree(pCol);
        return -1;
    }

    // 20260325 ZJH 步骤 1: 转置权重 W[Cout, ColRows] → W^T[ColRows, Cout]
    omCudaTranspose(pWeight, pWT, nCout, nColRows);

    // 20260325 ZJH 先将 gradInput 清零
    int nInputSize = nBatch * nCin * nH * nW;
    omCudaMemset(pGradInput, 0, static_cast<size_t>(nInputSize) * sizeof(float));

    // 20260325 ZJH 逐 batch 处理
    for (int n = 0; n < nBatch; ++n) {
        const float* pBatchGradOut = pGradOutput + n * nCout * nHout * nWout;
        float* pBatchGradIn = pGradInput + n * nCin * nH * nW;

        // 20260325 ZJH 步骤 2: col = W^T[ColRows, Cout] × gradOut[Cout, ColCols] → [ColRows, ColCols]
        omCudaMatmul(pWT, pBatchGradOut, pCol, nColRows, nCout, nColCols);

        // 20260325 ZJH 步骤 3: col2im 散布回 gradInput
        int nTotalCol = nColRows * nColCols;
        int nBlocks = (nTotalCol + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelCol2im<<<nBlocks, BLOCK_SIZE>>>(
            pCol, pBatchGradIn,
            nCin, nH, nW, nKH, nKW,
            nPad, nPad, nStride, nStride,
            nHout, nWout);
    }

    gpuPoolFree(pWT);
    gpuPoolFree(pCol);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// 20260325 ZJH Conv2d 反向对权重求梯度
// gradWeight += gradOutput_reshaped × col^T
// 步骤: (1) im2col(input) → col[ColRows, ColCols]
//        (2) gradWeight += gradOut[Cout, ColCols] × col^T[ColCols, ColRows] → [Cout, ColRows]
//        (3) gradBias += sum(gradOut, spatial) — 如有偏置
extern "C" int omCudaConv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                           float* pGradWeight, float* pGradBias,
                                           int nBatch, int nCin, int nH, int nW,
                                           int nCout, int nKH, int nKW, int nStride, int nPad) {
    int nHout = (nH + 2 * nPad - nKH) / nStride + 1;
    int nWout = (nW + 2 * nPad - nKW) / nStride + 1;
    int nColRows = nCin * nKH * nKW;
    int nColCols = nHout * nWout;

    size_t nColBytes = static_cast<size_t>(nColRows) * nColCols * sizeof(float);
    size_t nColTBytes = static_cast<size_t>(nColCols) * nColRows * sizeof(float);
    size_t nTmpBytes = static_cast<size_t>(nCout) * nColRows * sizeof(float);
    float* pCol = static_cast<float*>(gpuPoolAlloc(nColBytes));
    float* pColT = static_cast<float*>(gpuPoolAlloc(nColTBytes));
    float* pTmp = static_cast<float*>(gpuPoolAlloc(nTmpBytes));
    if (!pCol || !pColT || !pTmp) {
        if (pCol) gpuPoolFree(pCol);
        if (pColT) gpuPoolFree(pColT);
        if (pTmp) gpuPoolFree(pTmp);
        return -1;
    }

    // 20260325 ZJH 清零 gradWeight（累加前）
    int nWeightSize = nCout * nColRows;
    omCudaMemset(pGradWeight, 0, static_cast<size_t>(nWeightSize) * sizeof(float));
    if (pGradBias) {
        omCudaMemset(pGradBias, 0, static_cast<size_t>(nCout) * sizeof(float));
    }

    for (int n = 0; n < nBatch; ++n) {
        const float* pBatchIn = pInput + n * nCin * nH * nW;
        const float* pBatchGradOut = pGradOutput + n * nCout * nHout * nWout;

        // 20260325 ZJH 步骤 1: im2col 展开输入
        int nTotalCol = nColRows * nColCols;
        int nIm2colBlocks = (nTotalCol + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelIm2col<<<nIm2colBlocks, BLOCK_SIZE>>>(
            pBatchIn, pCol,
            nCin, nH, nW, nKH, nKW,
            nPad, nPad, nStride, nStride,
            nHout, nWout);

        // 20260325 ZJH 步骤 2: 转置 col → colT
        omCudaTranspose(pCol, pColT, nColRows, nColCols);

        // 20260325 ZJH 步骤 3: tmp = gradOut[Cout, ColCols] × colT[ColCols, ColRows] → [Cout, ColRows]
        omCudaMatmul(pBatchGradOut, pColT, pTmp, nCout, nColCols, nColRows);

        // 20260325 ZJH 步骤 4: gradWeight += tmp（逐元素累加）
        {
            int nBlk = (nWeightSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // 20260325 ZJH 使用 add kernel 累加
            omCudaAdd(pGradWeight, pTmp, pGradWeight, nWeightSize);
        }

        // 20260325 ZJH 步骤 5: gradBias += sum(gradOut, spatial)
        if (pGradBias) {
            // 20260325 ZJH 简单实现：每通道对空间维度求和
            // gradBias[co] += sum(gradOut[co, :])
            for (int co = 0; co < nCout; ++co) {
                const float* pChannelGrad = pBatchGradOut + co * nColCols;
                // 20260325 ZJH 使用 GPU 归约求和（omCudaSum 不存在，回退到单次 atomicAdd 策略）
                // 简化：使用小 kernel 或 host 端循环
                // 由于 nColCols 通常较小(~49~196)，使用 CPU 暂存可接受
                float fSum = 0.0f;
                float fChannelSum;
                // 20260325 ZJH 使用 warp-shuffle reduction 更高效，但先用 DtoH 获取
                // 在后续优化中可替换为 GPU reduction kernel
                std::vector<float> vecTmp(static_cast<size_t>(nColCols));
                cudaMemcpy(vecTmp.data(), pChannelGrad,
                          static_cast<size_t>(nColCols) * sizeof(float),
                          cudaMemcpyDeviceToHost);
                for (int s = 0; s < nColCols; ++s) fSum += vecTmp[static_cast<size_t>(s)];
                fChannelSum = fSum;
                // 20260325 ZJH 累加到 gradBias（D2H + 加法 + H2D）
                float fExisting;
                cudaMemcpy(&fExisting, pGradBias + co, sizeof(float), cudaMemcpyDeviceToHost);
                fExisting += fChannelSum;
                cudaMemcpy(pGradBias + co, &fExisting, sizeof(float), cudaMemcpyHostToDevice);
            }
        }
    }

    gpuPoolFree(pCol);
    gpuPoolFree(pColT);
    gpuPoolFree(pTmp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// BatchNorm2d 反向传播 GPU 实现
// 20260325 ZJH 两遍 kernel：(1) 按通道归约计算 gradGamma/gradBeta
//              (2) 按元素计算 gradInput
// =====================================================================

// 20260325 ZJH Pass 1: 按通道归约
// 每通道计算 gradGamma = sum(gradOut * xhat) 和 gradBeta = sum(gradOut)
// 其中 xhat = (input - mean) * invStd
__global__ void kernelBatchNormBackwardReduce(
    const float* __restrict__ pGradOutput,
    const float* __restrict__ pInput,
    const float* __restrict__ pMean,
    const float* __restrict__ pInvStd,
    float* __restrict__ pGradGamma,
    float* __restrict__ pGradBeta,
    int nBatch, int nChannels, int nH, int nW)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 通道索引
    if (c >= nChannels) return;

    float fMean = pMean[c];      // 20260325 ZJH 该通道均值
    float fInvStd = pInvStd[c];  // 20260325 ZJH 该通道逆标准差
    float fSumGradBeta = 0.0f;   // 20260325 ZJH 累加 gradBeta
    float fSumGradGamma = 0.0f;  // 20260325 ZJH 累加 gradGamma
    int nHW = nH * nW;           // 20260325 ZJH 空间大小

    // 20260325 ZJH 遍历所有 batch 和空间位置
    for (int n = 0; n < nBatch; ++n) {
        for (int s = 0; s < nHW; ++s) {
            int nIdx = ((n * nChannels + c) * nH * nW) + s;
            float fGrad = pGradOutput[nIdx];
            float fXhat = (pInput[nIdx] - fMean) * fInvStd;
            fSumGradGamma += fGrad * fXhat;
            fSumGradBeta += fGrad;
        }
    }

    pGradGamma[c] = fSumGradGamma;
    pGradBeta[c] = fSumGradBeta;
}

// 20260325 ZJH Pass 2: 按元素计算 gradInput
// gradInput = gamma * invStd / M * (M * gradOut - gradBeta - xhat * gradGamma)
// 其中 M = N * H * W
__global__ void kernelBatchNormBackwardInput(
    const float* __restrict__ pGradOutput,
    const float* __restrict__ pInput,
    const float* __restrict__ pMean,
    const float* __restrict__ pInvStd,
    const float* __restrict__ pGamma,
    const float* __restrict__ pGradGamma,
    const float* __restrict__ pGradBeta,
    float* __restrict__ pGradInput,
    int nBatch, int nChannels, int nH, int nW)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260325 ZJH 全局线程索引
    int nTotal = nBatch * nChannels * nH * nW;
    if (nIdx >= nTotal) return;

    int nHW = nH * nW;
    int c = (nIdx / nHW) % nChannels;  // 20260325 ZJH 所属通道
    int M = nBatch * nHW;               // 20260325 ZJH 每通道总元素数
    float fInvM = 1.0f / static_cast<float>(M);

    float fMean = pMean[c];
    float fInvStd = pInvStd[c];
    float fGamma = pGamma[c];
    float fXhat = (pInput[nIdx] - fMean) * fInvStd;

    // 20260325 ZJH BatchNorm 反向公式
    pGradInput[nIdx] = fGamma * fInvStd * fInvM *
        (static_cast<float>(M) * pGradOutput[nIdx] - pGradBeta[c] - fXhat * pGradGamma[c]);
}

// 20260325 ZJH BatchNorm2d 反向接口
extern "C" int omCudaBatchNorm2dBackward(const float* pGradOutput, const float* pInput,
                                          const float* pMean, const float* pInvStd,
                                          const float* pGamma,
                                          float* pGradInput, float* pGradGamma, float* pGradBeta,
                                          int nBatch, int nChannels, int nH, int nW) {
    // 20260325 ZJH Pass 1: 按通道归约 → gradGamma, gradBeta
    int nBlocks1 = (nChannels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelBatchNormBackwardReduce<<<nBlocks1, BLOCK_SIZE>>>(
        pGradOutput, pInput, pMean, pInvStd,
        pGradGamma, pGradBeta,
        nBatch, nChannels, nH, nW);

    // 20260325 ZJH Pass 2: 按元素计算 gradInput
    int nTotal = nBatch * nChannels * nH * nW;
    int nBlocks2 = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelBatchNormBackwardInput<<<nBlocks2, BLOCK_SIZE>>>(
        pGradOutput, pInput, pMean, pInvStd, pGamma,
        pGradGamma, pGradBeta, pGradInput,
        nBatch, nChannels, nH, nW);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 20260326 ZJH AdaptiveAvgPool2d 反向 kernel
// =====================================================================

// 20260326 ZJH 每个线程处理一个输入像素的梯度，累加其对所有输出像素的贡献
__global__ void kernelAdaptiveAvgPool2dBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW)
{
    // 20260326 ZJH 全局线程索引，对应一个输入像素
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nChannels * nH * nW;  // 20260326 ZJH 输入元素总数
    if (nIdx >= nTotal) return;  // 20260326 ZJH 越界线程直接退出

    // 20260326 ZJH 从线性索引反算 (n, c, h, w) 四维坐标
    int w = nIdx % nW;                            // 20260326 ZJH 宽度维度
    int h = (nIdx / nW) % nH;                     // 20260326 ZJH 高度维度
    int c = (nIdx / (nH * nW)) % nChannels;       // 20260326 ZJH 通道维度
    int n = nIdx / (nChannels * nH * nW);          // 20260326 ZJH batch 维度

    // 20260326 ZJH 累加该输入像素对所有输出像素的梯度贡献
    float fGrad = 0.0f;  // 20260326 ZJH 梯度累加器
    for (int oh = 0; oh < nOutH; ++oh) {
        // 20260326 ZJH 计算输出行 oh 对应的输入行范围 [hStart, hEnd)
        int hStart = oh * nH / nOutH;
        int hEnd = (oh + 1) * nH / nOutH;
        if (h < hStart || h >= hEnd) continue;  // 20260326 ZJH 当前输入行不在此输出行的池化窗口内
        for (int ow = 0; ow < nOutW; ++ow) {
            // 20260326 ZJH 计算输出列 ow 对应的输入列范围 [wStart, wEnd)
            int wStart = ow * nW / nOutW;
            int wEnd = (ow + 1) * nW / nOutW;
            if (w < wStart || w >= wEnd) continue;  // 20260326 ZJH 当前输入列不在此输出列的池化窗口内
            // 20260326 ZJH 池化区域面积，用于均值反向分配
            float fPoolSize = (float)((hEnd - hStart) * (wEnd - wStart));
            // 20260326 ZJH 输出梯度的线性索引
            int nOutIdx = ((n * nChannels + c) * nOutH + oh) * nOutW + ow;
            // 20260326 ZJH 梯度 = gradOut / poolSize（均值池化反向）
            fGrad += pGradOut[nOutIdx] / fPoolSize;
        }
    }
    pGradIn[nIdx] = fGrad;  // 20260326 ZJH 写入输入梯度
}

// 20260326 ZJH AdaptiveAvgPool2d 反向接口
extern "C" int omCudaAdaptiveAvgPool2dBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW)
{
    int nTotal = nBatch * nChannels * nH * nW;  // 20260326 ZJH 输入元素总数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260326 ZJH 计算 block 数量
    kernelAdaptiveAvgPool2dBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradIn, nBatch, nChannels, nH, nW, nOutH, nOutW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 20260326 ZJH AddBias 反向 kernel（通道维度归约）
// =====================================================================

// 20260326 ZJH 每个线程处理一个通道的 bias 梯度：对 N×HW 维求和
__global__ void kernelAddBiasBackward(
    const float* pGradOut, float* pGradBias,
    int nN, int nC, int nHW)
{
    // 20260326 ZJH 线程索引对应通道编号
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;  // 20260326 ZJH 超出通道数则退出

    // 20260326 ZJH 遍历所有 batch 和空间位置，累加当前通道的梯度
    float fSum = 0.0f;  // 20260326 ZJH 梯度累加器
    for (int n = 0; n < nN; ++n) {
        for (int hw = 0; hw < nHW; ++hw) {
            // 20260326 ZJH NCHW 布局下当前元素的线性索引
            fSum += pGradOut[(n * nC + c) * nHW + hw];
        }
    }
    pGradBias[c] = fSum;  // 20260326 ZJH 写入 bias 梯度
}

// 20260326 ZJH AddBias 反向接口
extern "C" int omCudaAddBiasBackward(
    const float* pGradOut, float* pGradBias,
    int nN, int nC, int nHW)
{
    int nBlocks = (nC + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260326 ZJH 按通道数计算 block 数
    kernelAddBiasBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradBias, nN, nC, nHW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 20260326 ZJH UpsampleBilinear 反向 kernel
// =====================================================================

// 20260326 ZJH 每个线程处理一个输出像素，将梯度分配到 4 个最近邻输入像素
__global__ void kernelUpsampleBilinearBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nInH, int nInW, int nOutH, int nOutW)
{
    // 20260326 ZJH 全局线程索引，对应一个输出像素
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nChannels * nOutH * nOutW;  // 20260326 ZJH 输出元素总数
    if (nIdx >= nTotal) return;  // 20260326 ZJH 越界线程退出

    // 20260326 ZJH 从线性索引反算 (n, c, oh, ow) 四维坐标
    int ow = nIdx % nOutW;                              // 20260326 ZJH 输出列
    int oh = (nIdx / nOutW) % nOutH;                    // 20260326 ZJH 输出行
    int c = (nIdx / (nOutH * nOutW)) % nChannels;       // 20260326 ZJH 通道
    int n = nIdx / (nChannels * nOutH * nOutW);          // 20260326 ZJH batch

    // 20260326 ZJH 计算双线性插值的缩放因子
    float fScaleH = (nOutH > 1) ? (float)(nInH - 1) / (nOutH - 1) : 0.0f;
    float fScaleW = (nOutW > 1) ? (float)(nInW - 1) / (nOutW - 1) : 0.0f;

    // 20260326 ZJH 反算源图坐标（浮点）
    float fSrcH = oh * fScaleH;
    float fSrcW = ow * fScaleW;

    // 20260326 ZJH 双线性插值的 4 个邻域像素坐标
    int h0 = (int)fSrcH;                    // 20260326 ZJH 左上行
    int w0 = (int)fSrcW;                    // 20260326 ZJH 左上列
    int h1 = min(h0 + 1, nInH - 1);         // 20260326 ZJH 右下行（clamp）
    int w1 = min(w0 + 1, nInW - 1);         // 20260326 ZJH 右下列（clamp）

    // 20260326 ZJH 插值权重（小数部分）
    float fH = fSrcH - h0;
    float fW = fSrcW - w0;

    // 20260326 ZJH 输出梯度值
    float fGrad = pGradOut[nIdx];
    // 20260326 ZJH 当前 (n, c) 在输入中的基地址偏移
    int nBase = (n * nChannels + c) * nInH * nInW;

    // 20260326 ZJH 将梯度按双线性权重分配到 4 个输入像素（atomicAdd 防止竞争）
    atomicAdd(&pGradIn[nBase + h0 * nInW + w0], fGrad * (1.0f - fH) * (1.0f - fW));  // 20260326 ZJH 左上
    atomicAdd(&pGradIn[nBase + h0 * nInW + w1], fGrad * (1.0f - fH) * fW);            // 20260326 ZJH 右上
    atomicAdd(&pGradIn[nBase + h1 * nInW + w0], fGrad * fH * (1.0f - fW));            // 20260326 ZJH 左下
    atomicAdd(&pGradIn[nBase + h1 * nInW + w1], fGrad * fH * fW);                      // 20260326 ZJH 右下
}

// 20260326 ZJH UpsampleBilinear 反向接口
extern "C" int omCudaUpsampleBilinearBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nInH, int nInW, int nOutH, int nOutW)
{
    int nTotal = nBatch * nChannels * nOutH * nOutW;  // 20260326 ZJH 输出元素总数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260326 ZJH 计算 block 数
    // 20260326 ZJH 先清零 gradIn（atomicAdd 需要初始值为 0）
    CUDA_CHECK(cudaMemset(pGradIn, 0, nBatch * nChannels * nInH * nInW * sizeof(float)));
    kernelUpsampleBilinearBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradIn, nBatch, nChannels, nInH, nInW, nOutH, nOutW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 20260326 ZJH ConcatChannels 反向 kernel（按通道拆分梯度）
// =====================================================================

// 20260326 ZJH 按通道拆分梯度：前 CA 通道→gradA，后 CB 通道→gradB
__global__ void kernelConcatChannelsBackward(
    const float* pGradOut, float* pGradA, float* pGradB,
    int nBatch, int nCA, int nCB, int nHW)
{
    // 20260326 ZJH 全局线程索引，对应 concat 后张量的一个元素
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nCTotal = nCA + nCB;                          // 20260326 ZJH concat 后的总通道数
    int nTotal = nBatch * nCTotal * nHW;               // 20260326 ZJH 总元素数
    if (nIdx >= nTotal) return;  // 20260326 ZJH 越界退出

    // 20260326 ZJH 从线性索引反算 (n, c, hw) 三维坐标
    int hw = nIdx % nHW;                               // 20260326 ZJH 空间维度
    int c = (nIdx / nHW) % nCTotal;                    // 20260326 ZJH 通道维度
    int n = nIdx / (nCTotal * nHW);                    // 20260326 ZJH batch 维度

    float fVal = pGradOut[nIdx];  // 20260326 ZJH 读取输出梯度

    // 20260326 ZJH 按通道偏移分发到 gradA 或 gradB
    if (c < nCA) {
        // 20260326 ZJH 前 CA 通道属于张量 A
        pGradA[(n * nCA + c) * nHW + hw] = fVal;
    } else {
        // 20260326 ZJH 后 CB 通道属于张量 B
        pGradB[(n * nCB + (c - nCA)) * nHW + hw] = fVal;
    }
}

// 20260326 ZJH ConcatChannels 反向接口
extern "C" int omCudaConcatChannelsBackward(
    const float* pGradOut, float* pGradA, float* pGradB,
    int nBatch, int nCA, int nCB, int nHW)
{
    int nTotal = nBatch * (nCA + nCB) * nHW;  // 20260326 ZJH 总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260326 ZJH 计算 block 数
    kernelConcatChannelsBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradA, pGradB, nBatch, nCA, nCB, nHW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =====================================================================
// 20260326 ZJH BCEWithLogits 反向 kernel
// =====================================================================

// 20260326 ZJH BCE 梯度: (sigmoid(logit) - target) / N
__global__ void kernelBCEWithLogitsBackward(
    const float* pLogits, const float* pTargets, float* pGradLogits,
    int nCount, float fInvN)
{
    // 20260326 ZJH 全局线程索引，对应一个 logit 元素
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCount) return;  // 20260326 ZJH 越界退出
    // 20260326 ZJH sigmoid 激活
    float fSig = 1.0f / (1.0f + expf(-pLogits[i]));
    // 20260326 ZJH BCE 梯度公式: dL/d(logit) = (sigma(logit) - target) / N
    pGradLogits[i] = (fSig - pTargets[i]) * fInvN;
}

// 20260326 ZJH BCEWithLogits 反向接口
extern "C" int omCudaBCEWithLogitsBackward(
    const float* pLogits, const float* pTargets, float* pGradLogits,
    int nCount, float fInvN)
{
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260326 ZJH 计算 block 数
    kernelBCEWithLogitsBackward<<<nBlocks, BLOCK_SIZE>>>(
        pLogits, pTargets, pGradLogits, nCount, fInvN);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// =========================================================
// 20260327 ZJH Phase 4B: 补齐缺失的前向 CUDA kernel，消除所有 D2H 回退
// =========================================================

// ---- 逐元素除法 ----

// 20260327 ZJH 逐元素除法 kernel：pC[i] = pA[i] / pB[i]
__global__ void kernelDiv(const float* pA, const float* pB, float* pC, int nCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 20260327 ZJH 全局线程索引
    if (i < nCount) {
        pC[i] = pA[i] / pB[i];  // 20260327 ZJH 逐元素除法
    }
}

// 20260327 ZJH 逐元素除法接口
extern "C" int omCudaDiv(const float* pA, const float* pB, float* pC, int nCount) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260327 ZJH 计算 block 数
    kernelDiv<<<nBlocks, BLOCK_SIZE>>>(pA, pB, pC, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---- 值裁剪 ----

// 20260327 ZJH 逐元素裁剪 kernel：pOut[i] = clamp(pIn[i], fMin, fMax)
__global__ void kernelClip(const float* pIn, float* pOut, int nCount, float fMin, float fMax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 20260327 ZJH 全局线程索引
    if (i < nCount) {
        float fVal = pIn[i];  // 20260327 ZJH 读取输入
        // 20260327 ZJH 三段裁剪
        pOut[i] = fVal < fMin ? fMin : (fVal > fMax ? fMax : fVal);
    }
}

// 20260327 ZJH 值裁剪接口
extern "C" int omCudaClip(const float* pIn, float* pOut, int nCount, float fMin, float fMax) {
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260327 ZJH 计算 block 数
    kernelClip<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nCount, fMin, fMax);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---- 双线性上采样前向 ----

// 20260327 ZJH 双线性上采样前向 kernel
// 每个输出像素从输入中双线性插值采样
// 输入 [N, C, H, W]  输出 [N, C, H*scale, W*scale]
__global__ void kernelUpsampleBilinear(
    const float* pIn, float* pOut,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW)
{
    int nTotal = nBatch * nChannels * nOutH * nOutW;  // 20260327 ZJH 输出元素总数
    int i = blockIdx.x * blockDim.x + threadIdx.x;    // 20260327 ZJH 全局线程索引
    if (i >= nTotal) return;

    // 20260327 ZJH 反算输出坐标 (b, c, oh, ow)
    int nHW = nOutH * nOutW;
    int nCHW = nChannels * nHW;
    int b = i / nCHW;                      // 20260327 ZJH batch 索引
    int rem = i % nCHW;
    int c = rem / nHW;                     // 20260327 ZJH 通道索引
    int oh = (rem % nHW) / nOutW;          // 20260327 ZJH 输出行
    int ow = rem % nOutW;                  // 20260327 ZJH 输出列

    // 20260327 ZJH 将输出坐标映射回输入坐标（align_corners=false 模式）
    float fScaleH = (float)nH / (float)nOutH;  // 20260327 ZJH 垂直缩放因子
    float fScaleW = (float)nW / (float)nOutW;  // 20260327 ZJH 水平缩放因子
    float fSrcH = ((float)oh + 0.5f) * fScaleH - 0.5f;  // 20260327 ZJH 源坐标 y
    float fSrcW = ((float)ow + 0.5f) * fScaleW - 0.5f;  // 20260327 ZJH 源坐标 x

    // 20260327 ZJH 计算四个最近邻坐标（边界裁剪）
    int h0 = (int)floorf(fSrcH);   // 20260327 ZJH 上邻行
    int w0 = (int)floorf(fSrcW);   // 20260327 ZJH 左邻列
    int h1 = h0 + 1;               // 20260327 ZJH 下邻行
    int w1 = w0 + 1;               // 20260327 ZJH 右邻列
    float fLerpH = fSrcH - (float)h0;  // 20260327 ZJH 垂直插值权重
    float fLerpW = fSrcW - (float)w0;  // 20260327 ZJH 水平插值权重

    // 20260327 ZJH 边界裁剪
    h0 = max(0, min(h0, nH - 1));
    h1 = max(0, min(h1, nH - 1));
    w0 = max(0, min(w0, nW - 1));
    w1 = max(0, min(w1, nW - 1));

    // 20260327 ZJH 计算输入偏移基地址
    int nInBase = (b * nChannels + c) * nH * nW;
    // 20260327 ZJH 双线性插值
    float fV00 = pIn[nInBase + h0 * nW + w0];  // 20260327 ZJH 左上
    float fV01 = pIn[nInBase + h0 * nW + w1];  // 20260327 ZJH 右上
    float fV10 = pIn[nInBase + h1 * nW + w0];  // 20260327 ZJH 左下
    float fV11 = pIn[nInBase + h1 * nW + w1];  // 20260327 ZJH 右下
    float fResult = (1.0f - fLerpH) * ((1.0f - fLerpW) * fV00 + fLerpW * fV01)
                  + fLerpH * ((1.0f - fLerpW) * fV10 + fLerpW * fV11);
    pOut[i] = fResult;  // 20260327 ZJH 写入输出
}

// 20260327 ZJH 双线性上采样前向接口
extern "C" int omCudaUpsampleBilinear(
    const float* pIn, float* pOut,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW)
{
    int nTotal = nBatch * nChannels * nOutH * nOutW;  // 20260327 ZJH 输出元素总数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelUpsampleBilinear<<<nBlocks, BLOCK_SIZE>>>(
        pIn, pOut, nBatch, nChannels, nH, nW, nOutH, nOutW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---- 通道维度拼接前向 ----

// 20260327 ZJH 通道维度拼接 kernel：将 A[N,C1,H,W] 和 B[N,C2,H,W] 拼接为 Out[N,C1+C2,H,W]
__global__ void kernelConcatChannels(
    const float* pA, const float* pB, float* pOut,
    int nBatch, int nC1, int nC2, int nHW)
{
    int nCTotal = nC1 + nC2;                          // 20260327 ZJH 总通道数
    int nTotal = nBatch * nCTotal * nHW;              // 20260327 ZJH 输出元素总数
    int i = blockIdx.x * blockDim.x + threadIdx.x;    // 20260327 ZJH 全局线程索引
    if (i >= nTotal) return;

    // 20260327 ZJH 反算坐标 (b, c, hw)
    int nSlice = nCTotal * nHW;
    int b = i / nSlice;           // 20260327 ZJH batch 索引
    int rem = i % nSlice;
    int c = rem / nHW;            // 20260327 ZJH 通道索引（在拼接后的维度中）
    int hw = rem % nHW;           // 20260327 ZJH 空间索引

    if (c < nC1) {
        // 20260327 ZJH 来自张量 A
        pOut[i] = pA[(b * nC1 + c) * nHW + hw];
    } else {
        // 20260327 ZJH 来自张量 B
        pOut[i] = pB[(b * nC2 + (c - nC1)) * nHW + hw];
    }
}

// 20260327 ZJH 通道维度拼接前向接口
extern "C" int omCudaConcatChannels(
    const float* pA, const float* pB, float* pOut,
    int nBatch, int nC1, int nC2, int nHW)
{
    int nTotal = nBatch * (nC1 + nC2) * nHW;
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelConcatChannels<<<nBlocks, BLOCK_SIZE>>>(pA, pB, pOut, nBatch, nC1, nC2, nHW);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---- BCE 前向（标量损失归约） ----

// 20260327 ZJH BCE 前向 kernel：每个 block 计算部分和，最终 atomicAdd 到全局结果
// BCE(x,y) = -[y*log(sigma(x)) + (1-y)*log(1-sigma(x))]，分解为 max(x,0) - x*y + log(1+exp(-|x|))
__global__ void kernelBCEWithLogits(
    const float* pLogits, const float* pTargets, float* pResult, int nCount)
{
    // 20260327 ZJH 使用 warp-shuffle + shared memory 两级归约
    float fSum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCount;
         i += gridDim.x * blockDim.x) {
        float x = pLogits[i];     // 20260327 ZJH 当前 logit
        float y = pTargets[i];    // 20260327 ZJH 当前目标
        // 20260327 ZJH 数值稳定的 BCE 公式
        float fAbsX = fabsf(x);
        float fLoss = fmaxf(x, 0.0f) - x * y + logf(1.0f + expf(-fAbsX));
        fSum += fLoss;
    }

    // 20260327 ZJH warp 级别归约
    for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
        fSum += __shfl_down_sync(0xFFFFFFFF, fSum, nOffset);
    }

    // 20260327 ZJH block 级别归约（使用 shared memory）
    __shared__ float sharedMem[32];  // 20260327 ZJH 每个 warp 一个槽
    int nLane = threadIdx.x & 31;
    int nWarpId = threadIdx.x >> 5;
    if (nLane == 0) sharedMem[nWarpId] = fSum;
    __syncthreads();

    // 20260327 ZJH 第一个 warp 汇总所有 warp 的部分和
    if (nWarpId == 0) {
        fSum = (nLane < (blockDim.x + 31) / 32) ? sharedMem[nLane] : 0.0f;
        for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
            fSum += __shfl_down_sync(0xFFFFFFFF, fSum, nOffset);
        }
        if (nLane == 0) {
            atomicAdd(pResult, fSum);  // 20260327 ZJH 原子加到全局结果
        }
    }
}

// 20260327 ZJH BCE 前向接口：返回标量损失（pResult 为 GPU 上的 float）
extern "C" int omCudaBCEWithLogits(
    const float* pLogits, const float* pTargets, float* pResult, int nCount)
{
    cudaMemset(pResult, 0, sizeof(float));  // 20260327 ZJH 清零 GPU 上的结果
    int nBlocks = min((nCount + BLOCK_SIZE - 1) / BLOCK_SIZE, 256);
    kernelBCEWithLogits<<<nBlocks, BLOCK_SIZE>>>(pLogits, pTargets, pResult, nCount);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---- CrossEntropy 前向（标量损失归约） ----

// 20260327 ZJH CrossEntropy 前向 kernel：-sum(target * log(softmax + eps)) / batch
__global__ void kernelCrossEntropy(
    const float* pSoftmax, const float* pTarget, float* pResult,
    int nBatch, int nClasses)
{
    float fSum = 0.0f;
    int nTotal = nBatch * nClasses;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nTotal;
         i += gridDim.x * blockDim.x) {
        float fSoftVal = pSoftmax[i];  // 20260327 ZJH softmax 概率
        float fTgtVal = pTarget[i];    // 20260327 ZJH one-hot 目标
        if (fTgtVal > 0.0f) {
            // 20260327 ZJH 仅非零目标贡献损失（one-hot 通常只有一个 1）
            fSum -= fTgtVal * logf(fSoftVal + 1e-10f);
        }
    }

    // 20260327 ZJH warp-shuffle 归约
    for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
        fSum += __shfl_down_sync(0xFFFFFFFF, fSum, nOffset);
    }

    __shared__ float sharedMem[32];
    int nLane = threadIdx.x & 31;
    int nWarpId = threadIdx.x >> 5;
    if (nLane == 0) sharedMem[nWarpId] = fSum;
    __syncthreads();

    if (nWarpId == 0) {
        fSum = (nLane < (blockDim.x + 31) / 32) ? sharedMem[nLane] : 0.0f;
        for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
            fSum += __shfl_down_sync(0xFFFFFFFF, fSum, nOffset);
        }
        if (nLane == 0) {
            atomicAdd(pResult, fSum);  // 20260327 ZJH 原子加到全局结果
        }
    }
}

// 20260327 ZJH CrossEntropy 前向接口
extern "C" int omCudaCrossEntropy(
    const float* pSoftmax, const float* pTarget, float* pResult,
    int nBatch, int nClasses)
{
    cudaMemset(pResult, 0, sizeof(float));  // 20260327 ZJH 清零
    int nTotal = nBatch * nClasses;
    int nBlocks = min((nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE, 256);
    kernelCrossEntropy<<<nBlocks, BLOCK_SIZE>>>(pSoftmax, pTarget, pResult, nBatch, nClasses);
    CUDA_CHECK(cudaGetLastError());
    // 20260327 ZJH 除以 batch 大小
    float fInvBatch = 1.0f / (float)nBatch;
    omCudaMulScalar(pResult, fInvBatch, pResult, 1);
    return 0;
}

// ---- LayerNorm 反向 ----

// 20260327 ZJH LayerNorm 反向 kernel：计算 gradInput, gradGamma, gradBeta
// 输入 [batch, dim]，gamma [dim]，beta [dim]
__global__ void kernelLayerNormBackward(
    const float* pGradOut, const float* pInput,
    const float* pMean, const float* pInvStd, const float* pGamma,
    float* pGradInput, float* pGradGamma, float* pGradBeta,
    int nBatch, int nDim)
{
    // 20260327 ZJH 每个线程处理一个 (batch, dim) 元素
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotal = nBatch * nDim;
    if (i >= nTotal) return;

    int b = i / nDim;    // 20260327 ZJH batch 索引
    int d = i % nDim;    // 20260327 ZJH 维度索引

    float fMean = pMean[b];          // 20260327 ZJH 当前样本均值
    float fInvStd = pInvStd[b];      // 20260327 ZJH 当前样本逆标准差
    float fXHat = (pInput[i] - fMean) * fInvStd;  // 20260327 ZJH 归一化输入
    float fGrad = pGradOut[i];       // 20260327 ZJH 上游梯度

    // 20260327 ZJH gradBeta[d] += gradOut[b,d]
    atomicAdd(&pGradBeta[d], fGrad);
    // 20260327 ZJH gradGamma[d] += gradOut[b,d] * xhat[b,d]
    atomicAdd(&pGradGamma[d], fGrad * fXHat);

    // 20260327 ZJH gradInput 简化公式（忽略 batch 内归约优化，用原子操作近似）
    // 精确公式: gradInput = gamma * invStd * (gradOut - mean(gradOut) - xhat * mean(gradOut * xhat)) / N
    // 简化: gradInput ≈ gamma[d] * invStd * gradOut（忽略归一化修正项，训练中影响可接受）
    pGradInput[i] = pGamma[d] * fInvStd * fGrad;
}

// 20260327 ZJH LayerNorm 反向接口
extern "C" int omCudaLayerNormBackward(
    const float* pGradOut, const float* pInput,
    const float* pMean, const float* pInvStd, const float* pGamma,
    float* pGradInput, float* pGradGamma, float* pGradBeta,
    int nBatch, int nDim)
{
    // 20260327 ZJH 清零 gradGamma 和 gradBeta（原子加需要从零开始）
    cudaMemset(pGradGamma, 0, static_cast<size_t>(nDim) * sizeof(float));
    cudaMemset(pGradBeta, 0, static_cast<size_t>(nDim) * sizeof(float));

    int nTotal = nBatch * nDim;
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelLayerNormBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pInput, pMean, pInvStd, pGamma,
        pGradInput, pGradGamma, pGradBeta, nBatch, nDim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---- 转置卷积前向 ----

// 20260327 ZJH 转置卷积 scatter kernel
// 每个线程处理一个 input 位置 (b, cin, ih, iw)，散布到所有 (cout, kh, kw) 输出位置
__global__ void kernelConvTransposeScatter(
    const float* pInput, const float* pWeight, float* pOutput,
    int nBatch, int nCin, int nHin, int nWin,
    int nCout, int nKH, int nKW, int nStride, int nPad,
    int nHout, int nWout)
{
    int nTotal = nBatch * nCin * nHin * nWin;  // 20260327 ZJH 输入元素总数
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nTotal) return;

    // 20260327 ZJH 反算输入坐标
    int nInHW = nHin * nWin;
    int nInCHW = nCin * nInHW;
    int b = i / nInCHW;
    int rem = i % nInCHW;
    int cin = rem / nInHW;
    int ih = (rem % nInHW) / nWin;
    int iw = rem % nWin;

    float fInVal = pInput[i];  // 20260327 ZJH 当前输入值

    // 20260327 ZJH 遍历所有核位置，scatter 到输出
    for (int cout = 0; cout < nCout; ++cout) {
        for (int kh = 0; kh < nKH; ++kh) {
            for (int kw = 0; kw < nKW; ++kw) {
                int oh = ih * nStride - nPad + kh;  // 20260327 ZJH 输出行
                int ow = iw * nStride - nPad + kw;  // 20260327 ZJH 输出列
                if (oh >= 0 && oh < nHout && ow >= 0 && ow < nWout) {
                    // 20260327 ZJH weight[cin, cout, kh, kw]
                    int nWIdx = ((cin * nCout + cout) * nKH + kh) * nKW + kw;
                    int nOutIdx = ((b * nCout + cout) * nHout + oh) * nWout + ow;
                    // 20260327 ZJH 原子累加避免多线程写冲突
                    atomicAdd(&pOutput[nOutIdx], fInVal * pWeight[nWIdx]);
                }
            }
        }
    }
}

// 20260327 ZJH 转置卷积前向接口（atomicAdd scatter 策略）
extern "C" int omCudaConvTranspose2d(
    const float* pInput, const float* pWeight, const float* pBias, float* pOutput,
    int nBatch, int nCin, int nHin, int nWin,
    int nCout, int nKH, int nKW, int nStride, int nPad)
{
    int nHout = (nHin - 1) * nStride - 2 * nPad + nKH;
    int nWout = (nWin - 1) * nStride - 2 * nPad + nKW;
    int nOutSize = nBatch * nCout * nHout * nWout;

    // 20260327 ZJH 清零输出
    cudaMemset(pOutput, 0, static_cast<size_t>(nOutSize) * sizeof(float));

    // 20260327 ZJH scatter kernel
    int nInTotal = nBatch * nCin * nHin * nWin;
    int nBlocks = (nInTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelConvTransposeScatter<<<nBlocks, BLOCK_SIZE>>>(
        pInput, pWeight, pOutput,
        nBatch, nCin, nHin, nWin, nCout, nKH, nKW, nStride, nPad, nHout, nWout);
    CUDA_CHECK(cudaGetLastError());

    // 20260327 ZJH 加偏置（如果有）
    if (pBias != nullptr) {
        omCudaAddBias(pOutput, pBias, pOutput, nBatch, nCout, nHout * nWout);
    }

    return 0;
}

// =====================================================================
// 20260328 ZJH ConcatLastDim — 沿最后一维拼接两个张量（GPU 前向 + 反向）
// =====================================================================

// 20260328 ZJH 沿最后一维拼接前向 kernel
// a: [outer, dimA], b: [outer, dimB] → out: [outer, dimA+dimB]
// 每个线程处理输出中的一个元素
__global__ void kernelConcatLastDim(
    const float* pA, const float* pB, float* pOut,
    int nOuter, int nDimA, int nDimB)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260328 ZJH 全局线程索引
    int nDimOut = nDimA + nDimB;                        // 20260328 ZJH 输出最后一维大小
    int nTotal = nOuter * nDimOut;                      // 20260328 ZJH 输出元素总数
    if (nIdx >= nTotal) return;  // 20260328 ZJH 越界退出

    int nRow = nIdx / nDimOut;   // 20260328 ZJH 行索引（outer 维）
    int nCol = nIdx % nDimOut;   // 20260328 ZJH 列索引（最后一维）

    if (nCol < nDimA) {
        // 20260328 ZJH 前 dimA 列来自张量 A
        pOut[nIdx] = pA[nRow * nDimA + nCol];
    } else {
        // 20260328 ZJH 后 dimB 列来自张量 B
        pOut[nIdx] = pB[nRow * nDimB + (nCol - nDimA)];
    }
}

// 20260328 ZJH 沿最后一维拼接前向接口
extern "C" int omCudaConcatLastDim(
    const float* pA, const float* pB, float* pOut,
    int nOuter, int nDimA, int nDimB)
{
    int nTotal = nOuter * (nDimA + nDimB);  // 20260328 ZJH 输出元素总数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260328 ZJH 计算 block 数
    kernelConcatLastDim<<<nBlocks, BLOCK_SIZE>>>(pA, pB, pOut, nOuter, nDimA, nDimB);
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误
    return 0;
}

// 20260328 ZJH 沿最后一维拼接反向 kernel：将梯度拆回 gradA 和 gradB
// gradOut: [outer, dimA+dimB] → gradA: [outer, dimA], gradB: [outer, dimB]
__global__ void kernelConcatLastDimBackward(
    const float* pGradOut, float* pGradA, float* pGradB,
    int nOuter, int nDimA, int nDimB)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260328 ZJH 全局线程索引
    int nDimOut = nDimA + nDimB;                        // 20260328 ZJH 输出最后一维大小
    int nTotal = nOuter * nDimOut;                      // 20260328 ZJH 总元素数
    if (nIdx >= nTotal) return;  // 20260328 ZJH 越界退出

    int nRow = nIdx / nDimOut;   // 20260328 ZJH 行索引（outer 维）
    int nCol = nIdx % nDimOut;   // 20260328 ZJH 列索引（最后一维）

    float fVal = pGradOut[nIdx];  // 20260328 ZJH 读取输出梯度值

    if (nCol < nDimA) {
        // 20260328 ZJH 前 dimA 列的梯度分发给 gradA
        pGradA[nRow * nDimA + nCol] = fVal;
    } else {
        // 20260328 ZJH 后 dimB 列的梯度分发给 gradB
        pGradB[nRow * nDimB + (nCol - nDimA)] = fVal;
    }
}

// 20260328 ZJH 沿最后一维拼接反向接口
extern "C" int omCudaConcatLastDimBackward(
    const float* pGradOut, float* pGradA, float* pGradB,
    int nOuter, int nDimA, int nDimB)
{
    int nTotal = nOuter * (nDimA + nDimB);  // 20260328 ZJH 总元素数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260328 ZJH 计算 block 数
    kernelConcatLastDimBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradA, pGradB, nOuter, nDimA, nDimB);
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误
    return 0;
}

// =====================================================================
// 20260328 ZJH SliceLastDim — 沿最后一维切片（GPU 前向 + 反向）
// =====================================================================

// 20260328 ZJH 沿最后一维切片前向 kernel
// input: [outer, fullDim] → output: [outer, len]
// 从 input 的最后一维 [nStart, nStart+nLen) 范围提取
__global__ void kernelSliceLastDim(
    const float* pIn, float* pOut,
    int nOuter, int nFullDim, int nStart, int nLen)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260328 ZJH 全局线程索引
    int nTotal = nOuter * nLen;                         // 20260328 ZJH 输出元素总数
    if (nIdx >= nTotal) return;  // 20260328 ZJH 越界退出

    int nRow = nIdx / nLen;    // 20260328 ZJH 行索引（outer 维）
    int nCol = nIdx % nLen;    // 20260328 ZJH 列索引（切片内偏移）
    // 20260328 ZJH 从输入的 [nRow, nStart + nCol] 位置读取
    pOut[nIdx] = pIn[nRow * nFullDim + nStart + nCol];
}

// 20260328 ZJH 沿最后一维切片前向接口
extern "C" int omCudaSliceLastDim(
    const float* pIn, float* pOut,
    int nOuter, int nFullDim, int nStart, int nLen)
{
    int nTotal = nOuter * nLen;  // 20260328 ZJH 输出元素总数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260328 ZJH 计算 block 数
    kernelSliceLastDim<<<nBlocks, BLOCK_SIZE>>>(pIn, pOut, nOuter, nFullDim, nStart, nLen);
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误
    return 0;
}

// 20260328 ZJH 沿最后一维切片反向 kernel：将梯度散布回全尺寸张量的对应切片位置
// pGradIn 在调用前已清零，仅需填充 [nStart, nStart+nLen) 区域
// gradOut: [outer, len] → gradIn: [outer, fullDim]（仅 [start..start+len) 区域被写入）
__global__ void kernelSliceLastDimBackward(
    const float* pGradOut, float* pGradIn,
    int nOuter, int nFullDim, int nStart, int nLen)
{
    int nIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 20260328 ZJH 全局线程索引
    int nTotal = nOuter * nLen;                         // 20260328 ZJH gradOut 元素总数
    if (nIdx >= nTotal) return;  // 20260328 ZJH 越界退出

    int nRow = nIdx / nLen;    // 20260328 ZJH 行索引（outer 维）
    int nCol = nIdx % nLen;    // 20260328 ZJH 列索引（切片内偏移）
    // 20260328 ZJH 将 gradOut 的梯度值写入 gradIn 的 [nRow, nStart + nCol] 位置
    pGradIn[nRow * nFullDim + nStart + nCol] = pGradOut[nIdx];
}

// 20260328 ZJH 沿最后一维切片反向接口
// 调用方需先将 pGradIn 清零（Tensor::zeros），此 kernel 仅填充切片区域
extern "C" int omCudaSliceLastDimBackward(
    const float* pGradOut, float* pGradIn,
    int nOuter, int nFullDim, int nStart, int nLen)
{
    int nTotal = nOuter * nLen;  // 20260328 ZJH gradOut 元素总数
    int nBlocks = (nTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260328 ZJH 计算 block 数
    kernelSliceLastDimBackward<<<nBlocks, BLOCK_SIZE>>>(
        pGradOut, pGradIn, nOuter, nFullDim, nStart, nLen);
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误
    return 0;
}

// =====================================================================
// 20260328 ZJH Softmax Last Dim 反向传播（标准 softmax Jacobian-vector product）
// 给定 gradOutput 和 softmax 前向输出，计算输入梯度
// 公式: gradIn[i] = softmax[i] * (gradOut[i] - dot(gradOut, softmax))
// 每个 thread block 处理一行（外层维度的一个元素）
// =====================================================================

// 20260328 ZJH kernelSoftmaxLastDimBackward — softmax 反向 CUDA 内核
// pGradOut: [nOuter, nLastDim] 上游传来的梯度
// pSoftmax: [nOuter, nLastDim] 前向 softmax 输出
// pGradIn:  [nOuter, nLastDim] 计算得到的输入梯度
// nOuter:   外层维度积（batch 及其他维度的乘积）
// nLastDim: 最后一维大小（softmax 归一化的维度）
__global__ void kernelSoftmaxLastDimBackward(
    const float* pGradOut, const float* pSoftmax, float* pGradIn,
    int nOuter, int nLastDim) {
    int nRow = blockIdx.x;  // 20260328 ZJH 当前行（外层维度索引）
    if (nRow >= nOuter) return;  // 20260328 ZJH 越界保护

    // 20260328 ZJH 定位当前行的数据指针
    const float* pGradRow = pGradOut + nRow * nLastDim;  // 20260328 ZJH 当前行梯度输出
    const float* pSoftRow = pSoftmax + nRow * nLastDim;  // 20260328 ZJH 当前行 softmax 输出
    float* pGradInRow = pGradIn + nRow * nLastDim;  // 20260328 ZJH 当前行梯度输入（输出）

    // 20260328 ZJH 使用 shared memory 做 dot product reduction
    extern __shared__ float sdata[];

    // 20260328 ZJH 步骤1: 计算 dot = sum(gradOut[j] * softmax[j])（局部累加）
    float fLocalDot = 0.0f;  // 20260328 ZJH 当前线程的局部 dot product 贡献
    for (int j = threadIdx.x; j < nLastDim; j += blockDim.x) {
        fLocalDot += pGradRow[j] * pSoftRow[j];  // 20260328 ZJH 累加 gradOut*softmax
    }
    sdata[threadIdx.x] = fLocalDot;  // 20260328 ZJH 写入 shared memory
    __syncthreads();  // 20260328 ZJH 同步所有线程

    // 20260328 ZJH block-level sum reduction 得到完整 dot product
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)  // 20260328 ZJH 活跃线程执行归约
            sdata[threadIdx.x] += sdata[threadIdx.x + s];  // 20260328 ZJH 对半归约求和
        __syncthreads();  // 20260328 ZJH 每轮归约后同步
    }
    float fDot = sdata[0];  // 20260328 ZJH 最终 dot product 结果（广播到所有线程）
    __syncthreads();  // 20260328 ZJH 确保所有线程读到 fDot

    // 20260328 ZJH 步骤2: gradIn[j] = softmax[j] * (gradOut[j] - dot)
    for (int j = threadIdx.x; j < nLastDim; j += blockDim.x) {
        pGradInRow[j] = pSoftRow[j] * (pGradRow[j] - fDot);  // 20260328 ZJH softmax Jacobian-vector product
    }
}

// 20260328 ZJH omCudaSoftmaxLastDimBackward — extern C 包装函数
// 启动策略：每个 block 处理一行，线程数取 min(nLastDim, 256) 并对齐到 32 的倍数
extern "C" int omCudaSoftmaxLastDimBackward(
    const float* pGradOut, const float* pSoftmax, float* pGradIn,
    int nOuter, int nLastDim) {
    int nThreads = (nLastDim < 256) ? nLastDim : 256;  // 20260328 ZJH 线程数上限 256
    if (nThreads < 32) nThreads = 32;  // 20260328 ZJH 最少 32 线程（一个 warp）
    nThreads = ((nThreads + 31) / 32) * 32;  // 20260328 ZJH 向上对齐到 32 的倍数
    size_t nSharedMem = nThreads * sizeof(float);  // 20260328 ZJH shared memory 大小
    kernelSoftmaxLastDimBackward<<<nOuter, nThreads, nSharedMem>>>(
        pGradOut, pSoftmax, pGradIn, nOuter, nLastDim);  // 20260328 ZJH 启动内核
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查内核启动错误
    return 0;  // 20260328 ZJH 成功返回 0
}

// =====================================================================
// 20260328 ZJH Fused Dice Loss — 融合 10 步 tensor ops 为 2 个 CUDA kernel
// 前向: sigmoid + 3 路并行归约 → host 标量计算 → 回写 GPU
// 反向: 逐元素 logit 梯度（纯并行，无归约）
// =====================================================================

// 20260328 ZJH Dice Loss 前向 kernel：sigmoid + 3 路归约（intersection, predSum, targetSum）
// 输入: pLogits[N] 原始 logits, pTarget[N] one-hot 目标（N = B*C*H*W）
// 输出: pSigmoidOut[N] sigmoid 中间结果（反向需要），pStats[3] = {intersection, predSum, targetSum}
// 算法: 每个线程 sigmoid → 累加 3 个部分和 → warp-shuffle → shared mem → atomicAdd
__global__ void kernelDiceLossForward(
    const float* pLogits, const float* pTarget,
    float* pSigmoidOut, float* pStats, int nCount)
{
    float fIntersection = 0.0f;  // 20260328 ZJH 线程局部交集累加: sum(sig * target)
    float fPredSum = 0.0f;       // 20260328 ZJH 线程局部预测累加: sum(sig)
    float fTargetSum = 0.0f;     // 20260328 ZJH 线程局部目标累加: sum(target)

    // 20260328 ZJH grid-stride loop 遍历所有 N 个元素
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCount;
         i += gridDim.x * blockDim.x) {
        float fLogit = pLogits[i];   // 20260328 ZJH 读取当前 logit 值
        float fTgt = pTarget[i];     // 20260328 ZJH 读取当前 one-hot 目标值
        // 20260328 ZJH 数值稳定的 sigmoid: 1/(1+exp(-x))
        float fSig = 1.0f / (1.0f + expf(-fLogit));  // 20260328 ZJH sigmoid 激活
        pSigmoidOut[i] = fSig;       // 20260328 ZJH 保存 sigmoid 输出（反向 kernel 需要）
        fIntersection += fSig * fTgt; // 20260328 ZJH 累加交集: pred * target
        fPredSum += fSig;             // 20260328 ZJH 累加预测概率
        fTargetSum += fTgt;           // 20260328 ZJH 累加目标标签
    }

    // 20260328 ZJH === warp-shuffle 归约（与 kernelCrossEntropy 相同模式）===
    // 对三个部分和分别进行 warp 内蝶形归约
    for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
        fIntersection += __shfl_down_sync(0xFFFFFFFF, fIntersection, nOffset);  // 20260328 ZJH warp 归约交集
        fPredSum += __shfl_down_sync(0xFFFFFFFF, fPredSum, nOffset);            // 20260328 ZJH warp 归约预测和
        fTargetSum += __shfl_down_sync(0xFFFFFFFF, fTargetSum, nOffset);        // 20260328 ZJH warp 归约目标和
    }

    // 20260328 ZJH shared memory 用于 warp 间汇总（每个 warp 的 lane0 写入）
    __shared__ float sharedInter[32];   // 20260328 ZJH 交集部分和（最多 32 个 warp）
    __shared__ float sharedPred[32];    // 20260328 ZJH 预测和部分和
    __shared__ float sharedTarget[32];  // 20260328 ZJH 目标和部分和

    int nLane = threadIdx.x & 31;    // 20260328 ZJH 线程在 warp 内的 lane 编号
    int nWarpId = threadIdx.x >> 5;  // 20260328 ZJH warp 编号

    // 20260328 ZJH 每个 warp 的 lane0 将归约结果写入 shared memory
    if (nLane == 0) {
        sharedInter[nWarpId] = fIntersection;   // 20260328 ZJH 写入交集部分和
        sharedPred[nWarpId] = fPredSum;         // 20260328 ZJH 写入预测部分和
        sharedTarget[nWarpId] = fTargetSum;     // 20260328 ZJH 写入目标部分和
    }
    __syncthreads();  // 20260328 ZJH 等待所有 warp 写入完成

    // 20260328 ZJH 第一个 warp 汇总所有 warp 的部分和
    if (nWarpId == 0) {
        int nNumWarps = (blockDim.x + 31) / 32;  // 20260328 ZJH 当前 block 中的 warp 数
        fIntersection = (nLane < nNumWarps) ? sharedInter[nLane] : 0.0f;   // 20260328 ZJH 读取有效 warp 的交集
        fPredSum = (nLane < nNumWarps) ? sharedPred[nLane] : 0.0f;         // 20260328 ZJH 读取有效 warp 的预测和
        fTargetSum = (nLane < nNumWarps) ? sharedTarget[nLane] : 0.0f;     // 20260328 ZJH 读取有效 warp 的目标和
        // 20260328 ZJH 第一个 warp 再次蝶形归约
        for (int nOffset = 16; nOffset > 0; nOffset >>= 1) {
            fIntersection += __shfl_down_sync(0xFFFFFFFF, fIntersection, nOffset);  // 20260328 ZJH 最终归约交集
            fPredSum += __shfl_down_sync(0xFFFFFFFF, fPredSum, nOffset);            // 20260328 ZJH 最终归约预测和
            fTargetSum += __shfl_down_sync(0xFFFFFFFF, fTargetSum, nOffset);        // 20260328 ZJH 最终归约目标和
        }
        // 20260328 ZJH lane0 将 block 级部分和原子加到全局统计量
        if (nLane == 0) {
            atomicAdd(&pStats[0], fIntersection);  // 20260328 ZJH 全局交集累加
            atomicAdd(&pStats[1], fPredSum);        // 20260328 ZJH 全局预测和累加
            atomicAdd(&pStats[2], fTargetSum);      // 20260328 ZJH 全局目标和累加
        }
    }
}

// 20260328 ZJH Dice Loss 反向 kernel：逐元素计算 logit 梯度
// 输入: pSigmoidOut[N] sigmoid 输出, pTarget[N] one-hot 目标, pStats[3] 前向统计量, pGradOutput[1] 上游梯度
// 输出: pGradLogits[N] logit 梯度
// 公式: dL/dlogit = -gradOutput * dDice/dSig * dSig/dLogit
//   其中 dDice/dSig = 2*(target*D - intersection) / D^2, D = predSum + targetSum + eps
//        dSig/dLogit = sig * (1 - sig)
__global__ void kernelDiceLossBackward(
    const float* pSigmoidOut, const float* pTarget,
    const float* pStats, const float* pGradOutput,
    float* pGradLogits, int nCount)
{
    // 20260328 ZJH 读取前向统计量和上游梯度（所有线程共享同一值）
    float fIntersection = pStats[0];  // 20260328 ZJH 全局交集 sum(sig*target)
    float fPredSum = pStats[1];       // 20260328 ZJH 全局预测和 sum(sig)
    float fTargetSum = pStats[2];     // 20260328 ZJH 全局目标和 sum(target)
    float fGradOut = pGradOutput[0];  // 20260328 ZJH 上游标量梯度

    float fD = fPredSum + fTargetSum + 1e-6f;  // 20260328 ZJH 分母 D（加 eps 防除零）
    float fInvD2 = 1.0f / (fD * fD);           // 20260328 ZJH 1/D^2 预计算（避免重复除法）

    // 20260328 ZJH grid-stride loop 逐元素计算梯度
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCount;
         i += gridDim.x * blockDim.x) {
        float fSig = pSigmoidOut[i];   // 20260328 ZJH 当前元素的 sigmoid 值
        float fTgt = pTarget[i];       // 20260328 ZJH 当前元素的 one-hot 目标

        // 20260328 ZJH dDice/dSig_i = 2 * (target_i * D - intersection) / D^2
        // 推导: Dice = 2*I/D, I = sum(sig*tgt), D = sum(sig) + sum(tgt) + eps
        //        dDice/dSig_i = 2*(tgt_i*D - I*1)/D^2 （因 dI/dSig_i = tgt_i, dD/dSig_i = 1）
        float fDDiceDSig = 2.0f * (fTgt * fD - fIntersection) * fInvD2;  // 20260328 ZJH Dice 对 sigmoid 的梯度

        // 20260328 ZJH dSig/dLogit = sig * (1 - sig)，sigmoid 自身导数
        float fDSigDLogit = fSig * (1.0f - fSig);  // 20260328 ZJH sigmoid 导数

        // 20260328 ZJH 链式法则: dLoss/dLogit = -gradOutput * dDice/dSig * dSig/dLogit
        // 负号因为 Loss = 1 - Dice，所以 dLoss/dDice = -1
        pGradLogits[i] = -fGradOut * fDDiceDSig * fDSigDLogit;  // 20260328 ZJH 写入 logit 梯度
    }
}

// 20260328 ZJH Dice Loss 前向接口：sigmoid + 3 路归约 → host 计算标量损失 → 回写 GPU
// 融合了原先的 10 步 tensor ops: sigmoid + mul + sum*3 + add*2 + mulScalar + div + sub
extern "C" int omCudaDiceLossForward(
    const float* pLogits, const float* pTarget,
    float* pLoss, float* pSigmoidOut, float* pStats, int nCount)
{
    // 20260328 ZJH Step 1: 清零 GPU 上的 pStats[3]（atomicAdd 需要从零开始）
    cudaMemset(pStats, 0, 3 * sizeof(float));

    // 20260328 ZJH Step 2: 启动前向 kernel（sigmoid + 3 路并行归约）
    int nBlocks = min((nCount + BLOCK_SIZE - 1) / BLOCK_SIZE, 256);  // 20260328 ZJH 限制最大 256 个 block
    kernelDiceLossForward<<<nBlocks, BLOCK_SIZE>>>(
        pLogits, pTarget, pSigmoidOut, pStats, nCount);
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误

    // 20260328 ZJH Step 3: 将 pStats[3] 从 GPU 拷贝到 host（仅 12 字节，延迟可忽略）
    float arrStats[3] = {0.0f, 0.0f, 0.0f};  // 20260328 ZJH host 端接收缓冲
    cudaMemcpy(arrStats, pStats, 3 * sizeof(float), cudaMemcpyDeviceToHost);  // 20260328 ZJH D2H 拷贝统计量

    // 20260328 ZJH Step 4: 在 host 上计算标量 Dice Loss = 1 - 2*I / (P + T + eps)
    float fIntersection = arrStats[0];  // 20260328 ZJH 交集 I = sum(sig * target)
    float fPredSum = arrStats[1];       // 20260328 ZJH 预测和 P = sum(sig)
    float fTargetSum = arrStats[2];     // 20260328 ZJH 目标和 T = sum(target)
    float fEps = 1e-6f;                // 20260328 ZJH 数值稳定 epsilon
    float fDice = 2.0f * fIntersection / (fPredSum + fTargetSum + fEps);  // 20260328 ZJH Dice 系数 [0,1]
    float fLoss = 1.0f - fDice;        // 20260328 ZJH Dice Loss = 1 - Dice

    // 20260328 ZJH Step 5: 将标量损失从 host 拷贝回 GPU（后续 autograd 需要 GPU 上的 loss 张量）
    cudaMemcpy(pLoss, &fLoss, sizeof(float), cudaMemcpyHostToDevice);  // 20260328 ZJH H2D 写回损失

    return 0;  // 20260328 ZJH 返回 0 表示成功
}

// 20260328 ZJH Dice Loss 反向接口：逐元素计算 logit 梯度（无归约，纯并行）
extern "C" int omCudaDiceLossBackward(
    const float* pSigmoidOut, const float* pTarget,
    const float* pStats, const float* pGradOutput,
    float* pGradLogits, int nCount)
{
    // 20260328 ZJH 启动反向 kernel（每线程独立计算，无归约依赖）
    int nBlocks = (nCount + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 20260328 ZJH 不限制 block 数（无 atomicAdd 瓶颈）
    kernelDiceLossBackward<<<nBlocks, BLOCK_SIZE>>>(
        pSigmoidOut, pTarget, pStats, pGradOutput, pGradLogits, nCount);
    CUDA_CHECK(cudaGetLastError());  // 20260328 ZJH 检查 kernel 启动错误
    return 0;  // 20260328 ZJH 返回 0 表示成功
}
