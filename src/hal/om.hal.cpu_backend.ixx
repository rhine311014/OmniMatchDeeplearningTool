// 20260319 ZJH CPUBackend — 朴素 C++ 计算内核
// Phase 1B: Float32 基础运算（先正确后优化）
// 20260324 ZJH 性能优化：集成 SIMD(AVX2) + OpenMP 多线程 + im2col/GEMM 卷积 + 内存池
module;

#include <cstddef>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>
#include <omp.h>       // 20260324 ZJH OpenMP 多线程并行化支持
#include <cstring>     // 20260324 ZJH memset 内存操作
// 20260325 ZJH [Phase 4] 移除 std::atomic，GPU 加速不再通过 CPUBackend 调度
// CPUBackend 恢复为纯 CPU 后端（SIMD + OpenMP），CUDA 通过 CUDABackend 独立模块处理

// 20260324 ZJH Windows 平台对齐内存分配函数
#ifdef _MSC_VER
#include <malloc.h>    // 20260324 ZJH _aligned_malloc / _aligned_free
#endif

// 20260325 ZJH [Phase 4] CUDA 函数前向声明已移除
// CPUBackend 不再直接调度 CUDA 操作。GPU 计算通过 CUDABackend 模块（om.hal.cuda_backend）独立处理
// tensor_ops 层根据 Tensor 设备类型自动选择 CPUBackend 或 CUDABackend

export module om.hal.cpu_backend;

// 20260324 ZJH 导入 SIMD 模块，获取 AVX2 加速内核
import om.hal.simd;

// 20260325 ZJH [Phase 4] GPU 加速标志和持久工作区已移除
// GPU 计算现在通过 CUDABackend 模块直接操作 GPU 指针，无需 CPUBackend 中转
// setGpuAcceleration/isGpuAccelerationEnabled/resetGpuWorkspace 保留为空操作（向后兼容）

export namespace om {

// 20260325 ZJH [Phase 4] 设置全局 GPU 加速开关（空操作，保留向后兼容）
// GPU 训练现在通过 CUDABackend 模块直接处理，CPUBackend 不再参与 GPU 调度
// 参数: bEnable — 已忽略（无实际效果）
void setGpuAcceleration(bool /*bEnable*/) {
    // 20260325 ZJH 空操作：Phase 4 后 GPU 计算通过 CUDABackend 独立调度
}

// 20260325 ZJH [Phase 4] 查询全局 GPU 加速是否启用（始终返回 false，保留向后兼容）
// 返回: false（CPUBackend 不再调度 GPU）
bool isGpuAccelerationEnabled() {
    return false;  // 20260325 ZJH Phase 4 后 CPUBackend 不参与 GPU 调度，始终返回 false
}

// 20260325 ZJH [Phase 4] 释放 GPU 持久工作区（空操作，保留向后兼容）
// GPU 内存管理已迁移到 TensorStorage + CUDABackend，此函数不再需要
void resetGpuWorkspace() {
    // 20260325 ZJH 空操作：持久工作区已移除
}

} // namespace om (GPU 加速接口 — Phase 4 后均为空操作)

export namespace om {

class CPUBackend {
public:
    // ===== 填充 =====
    // 20260319 ZJH 将 pData 所有元素置零
    static void fillZeros(float* pData, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pData[i] = 0.0f;
    }

    // 20260319 ZJH 将 pData 所有元素置 1.0f
    static void fillOnes(float* pData, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pData[i] = 1.0f;
    }

    // 20260319 ZJH 将 pData 所有元素填充为指定标量值
    static void fillValue(float* pData, size_t nCount, float fValue) {
        for (size_t i = 0; i < nCount; ++i) pData[i] = fValue;
    }

    // 20260319 ZJH 用标准正态分布（均值=0，标准差=1）填充 pData
    // 使用 thread_local 生成器保证多线程安全
    static void fillRandn(float* pData, size_t nCount) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < nCount; ++i) pData[i] = dist(gen);
    }

    // ===== 内存池 =====

    // 20260324 ZJH 线程本地空闲链表内存池，减少训练循环中的频繁堆分配开销
    // 每个线程独立维护最多 64 个已释放的对齐内存块，优先复用大小匹配的块
    static thread_local inline std::vector<std::pair<size_t, float*>> s_vecFreeList;

    // 20260324 ZJH 从内存池分配 64 字节对齐内存（适配 AVX2 的 256 位 = 32 字节要求，64 字节为缓存行大小）
    // 优先查找空闲链表中大小合适的块（nBytes <= 块大小 <= 2*nBytes），避免重新分配
    // 找不到合适块时才调用系统对齐分配
    // 参数: nBytes — 需要分配的字节数
    // 返回: 64 字节对齐的 float 指针
    static float* poolAllocAligned(size_t nBytes) {
        // 20260324 ZJH 遍历空闲链表查找大小匹配的已释放块
        for (auto it = s_vecFreeList.begin(); it != s_vecFreeList.end(); ++it) {
            // 20260324 ZJH 块大小在 [nBytes, 2*nBytes] 范围内视为匹配
            if (it->first >= nBytes && it->first <= nBytes * 2) {
                float* pPtr = it->second;  // 20260324 ZJH 取出匹配块指针
                s_vecFreeList.erase(it);   // 20260324 ZJH 从空闲链表移除
                return pPtr;               // 20260324 ZJH 返回复用的内存块
            }
        }
        // 20260324 ZJH 空闲链表中无合适块，调用系统对齐分配
#ifdef _MSC_VER
        float* pPtr = static_cast<float*>(_aligned_malloc(nBytes, 64));  // 20260324 ZJH MSVC 对齐分配
#else
        void* pRaw = nullptr;  // 20260324 ZJH POSIX 对齐分配输出指针
        posix_memalign(&pRaw, 64, nBytes);  // 20260324 ZJH GCC/Clang 对齐分配
        float* pPtr = static_cast<float*>(pRaw);
#endif
        // 20260326 ZJH 内存分配失败检查：_aligned_malloc 返回 nullptr 时抛出 C++ 异常
        // 此前缺少检查，224×224 大图训练时 im2col 缓冲区分配失败后解引用 nullptr
        // → SEH 访问违规 → abort()（C++ try/catch 无法捕获 SEH）
        if (!pPtr) {
            throw std::bad_alloc();
        }
        return pPtr;
    }

    // 20260324 ZJH 归还内存到池中或释放到系统
    // 空闲链表未满（< 64 个块）时缓存该块供后续复用，否则直接释放
    // 参数: pPtr — 待释放的对齐内存指针，nBytes — 该块的大小（字节）
    static void poolFreeAligned(float* pPtr, size_t nBytes) {
        if (!pPtr) return;  // 20260324 ZJH 空指针安全检查
        if (s_vecFreeList.size() < 64) {
            // 20260324 ZJH 空闲链表未满，缓存该块
            s_vecFreeList.push_back({nBytes, pPtr});
        } else {
            // 20260324 ZJH 空闲链表已满，直接释放
#ifdef _MSC_VER
            _aligned_free(pPtr);   // 20260324 ZJH MSVC 对齐释放
#else
            free(pPtr);            // 20260324 ZJH POSIX 对齐释放
#endif
        }
    }

    // ===== 元素运算 =====

    // 20260319 ZJH 逐元素加法：pOut[i] = pA[i] + pB[i]
    // 20260324 ZJH 优化：优先使用 AVX2 SIMD 加速（8 路并行），大数组启用 OpenMP 多线程
    static void add(const float* pA, const float* pB, float* pOut, size_t nCount) {
        // 20260324 ZJH 优先 AVX2 加速路径
        if (SIMDBackend::isAVX2Supported()) {
            SIMDBackend::addAVX2(pA, pB, pOut, nCount);
            return;  // 20260324 ZJH SIMD 完成，直接返回
        }
        // 20260324 ZJH 标量回退路径：大数组启用 OpenMP 并行
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) pOut[i] = pA[i] + pB[i];
        } else {
            for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] + pB[i];
        }
    }

    // 20260319 ZJH 逐元素减法：pOut[i] = pA[i] - pB[i]
    // 20260324 ZJH 优化：SIMD 模块无 sub 函数，仅添加 OpenMP 并行
    static void sub(const float* pA, const float* pB, float* pOut, size_t nCount) {
        if (nCount > 1024) {
            // 20260324 ZJH 大数组启用 OpenMP 并行
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) pOut[i] = pA[i] - pB[i];
        } else {
            for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] - pB[i];
        }
    }

    // 20260319 ZJH 逐元素乘法：pOut[i] = pA[i] * pB[i]
    // 20260324 ZJH 优化：优先使用 AVX2 SIMD 加速
    static void mul(const float* pA, const float* pB, float* pOut, size_t nCount) {
        // 20260324 ZJH 优先 AVX2 加速路径
        if (SIMDBackend::isAVX2Supported()) {
            SIMDBackend::mulAVX2(pA, pB, pOut, nCount);
            return;  // 20260324 ZJH SIMD 完成，直接返回
        }
        // 20260324 ZJH 标量回退路径：大数组启用 OpenMP 并行
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) pOut[i] = pA[i] * pB[i];
        } else {
            for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] * pB[i];
        }
    }

    // 20260319 ZJH 逐元素除法：pOut[i] = pA[i] / pB[i]
    // 20260324 ZJH 优化：添加 OpenMP 并行（SIMD 模块无 div 函数）
    static void div(const float* pA, const float* pB, float* pOut, size_t nCount) {
        if (nCount > 1024) {
            // 20260324 ZJH 大数组启用 OpenMP 并行
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) pOut[i] = pA[i] / pB[i];
        } else {
            for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] / pB[i];
        }
    }

    // 20260319 ZJH 加标量：pOut[i] = pA[i] + fScalar
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void addScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) pOut[i] = pA[i] + fScalar;
        } else {
            for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] + fScalar;
        }
    }

    // 20260319 ZJH 乘标量：pOut[i] = pA[i] * fScalar
    // 20260324 ZJH 优化：优先使用 AVX2 标量乘法加速
    static void mulScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
        // 20260324 ZJH 优先 AVX2 加速路径
        if (SIMDBackend::isAVX2Supported()) {
            SIMDBackend::mulScalarAVX2(pA, fScalar, pOut, nCount);
            return;  // 20260324 ZJH SIMD 完成，直接返回
        }
        // 20260324 ZJH 标量回退路径：大数组启用 OpenMP 并行
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) pOut[i] = pA[i] * fScalar;
        } else {
            for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] * fScalar;
        }
    }

    // ===== 矩阵乘法 =====
    // 20260319 ZJH A[M,K] * B[K,N] -> C[M,N], row-major 行主序
    // 20260325 ZJH [Phase 4] 纯 CPU 矩阵乘法（CUDA 调度路径已移除）
    // GPU 矩阵乘法现在通过 CUDABackend::matmul 直接操作 GPU 指针
    // CPU 分三级加速策略：
    //   Level 1: AVX2 + OpenMP（CPU 最优路径，M>=4 时按 4 行分块并行）
    //   Level 2: AVX2 单线程（小矩阵直接调用 SIMD）
    //   Level 3: 标量 + OpenMP 回退
    static void matmul(const float* pA, const float* pB, float* pC, int nM, int nK, int nN) {
        // 20260324 ZJH 优先使用 AVX2 + OpenMP 加速矩阵乘法
        if (SIMDBackend::isAVX2Supported()) {
            if (nM >= 4) {
                // 20260324 ZJH 初始化输出矩阵 C 为全零
                std::memset(pC, 0, static_cast<size_t>(nM) * nN * sizeof(float));
                // 20260324 ZJH 按 4 行为一个分块单位，多线程并行处理各行块
                // 每个线程调用 SIMDBackend::matmulAVX2 处理一个 4 行 × nN 列的子矩阵
                // schedule(dynamic, 1) 保证负载均衡（每个块计算量可能不同）
                #pragma omp parallel for schedule(dynamic, 1)
                for (int nBlock = 0; nBlock < nM; nBlock += 4) {
                    int nBlockRows = std::min(4, nM - nBlock);  // 20260324 ZJH 当前块的实际行数（末尾可能不足 4 行）
                    // 20260324 ZJH 调用 AVX2 矩阵乘法处理子矩阵 A[nBlock:nBlock+nBlockRows, :] * B
                    SIMDBackend::matmulAVX2(pA + nBlock * nK, pB, pC + nBlock * nN, nBlockRows, nK, nN);
                }
                return;  // 20260324 ZJH AVX2 + OpenMP 完成
            }
            // 20260324 ZJH M < 4 的小矩阵：直接使用 AVX2 单线程（开线程池的开销不值得）
            SIMDBackend::matmulAVX2(pA, pB, pC, nM, nK, nN);
            return;  // 20260324 ZJH AVX2 完成
        }
        // 20260324 ZJH 标量回退路径 + OpenMP 并行
        // 20260319 ZJH 初始化输出矩阵 C 为全零
        std::memset(pC, 0, static_cast<size_t>(nM) * nN * sizeof(float));
        // 20260319 ZJH 三重循环执行矩阵乘法，i-k-j 顺序提升 pB 访问局部性
        // 20260324 ZJH OpenMP 外层行并行：每一行独立计算，无数据竞争
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nM; ++i) {
            for (int k = 0; k < nK; ++k) {
                float fA_ik = pA[i * nK + k];  // 20260319 ZJH 缓存 A[i][k]，内层循环复用
                for (int j = 0; j < nN; ++j) {
                    pC[i * nN + j] += fA_ik * pB[k * nN + j];
                }
            }
        }
    }

    // ===== 归约 =====
    // 20260319 ZJH 求所有元素之和
    static float sum(const float* pData, size_t nCount) {
        float fSum = 0.0f;
        for (size_t i = 0; i < nCount; ++i) fSum += pData[i];
        return fSum;
    }

    // 20260319 ZJH 求所有元素的最大值，nCount==0 时返回 0
    static float max(const float* pData, size_t nCount) {
        if (nCount == 0) return 0.0f;
        float fMax = pData[0];
        for (size_t i = 1; i < nCount; ++i) if (pData[i] > fMax) fMax = pData[i];
        return fMax;
    }

    // 20260319 ZJH 求所有元素的最小值，nCount==0 时返回 0
    static float min(const float* pData, size_t nCount) {
        if (nCount == 0) return 0.0f;
        float fMin = pData[0];
        for (size_t i = 1; i < nCount; ++i) if (pData[i] < fMin) fMin = pData[i];
        return fMin;
    }

    // ===== 激活函数 =====

    // 20260319 ZJH ReLU 前向：out[i] = max(0, in[i])
    // 将负值置零，正值直通，实现修正线性单元激活
    // 20260324 ZJH 优化：优先使用 AVX2 SIMD 加速（_mm256_max_ps 8 路并行）
    static void relu(const float* pIn, float* pOut, size_t nCount) {
        // 20260324 ZJH 优先 AVX2 加速路径
        if (SIMDBackend::isAVX2Supported()) {
            SIMDBackend::reluAVX2(pIn, pOut, nCount);
            return;  // 20260324 ZJH SIMD 完成，直接返回
        }
        // 20260324 ZJH 标量回退路径：大数组启用 OpenMP 并行
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i)
                pOut[i] = pIn[i] > 0.0f ? pIn[i] : 0.0f;
        } else {
            for (size_t i = 0; i < nCount; ++i)
                pOut[i] = pIn[i] > 0.0f ? pIn[i] : 0.0f;  // 20260319 ZJH 正值直通，负值置零
        }
    }

    // 20260319 ZJH ReLU 反向：grad_in[i] = grad_out[i] * (in[i] > 0 ? 1 : 0)
    // 前向时输入大于零的位置梯度直通，否则梯度为零
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void reluBackward(const float* pIn, const float* pGradOut, float* pGradIn, size_t nCount) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i)
                pGradIn[i] = pIn[i] > 0.0f ? pGradOut[i] : 0.0f;
        } else {
            for (size_t i = 0; i < nCount; ++i)
                pGradIn[i] = pIn[i] > 0.0f ? pGradOut[i] : 0.0f;  // 20260319 ZJH 正值梯度直通，负值梯度截断
        }
    }

    // ===== Softmax / CrossEntropy =====

    // 20260319 ZJH Softmax 前向：沿最后一维对 [nBatch, nClasses] 做 softmax
    // 每行减最大值保证数值稳定性，再 exp + 归一化
    static void softmax(const float* pIn, float* pOut, int nBatch, int nClasses) {
        for (int b = 0; b < nBatch; ++b) {
            const float* pRow = pIn + b * nClasses;  // 20260319 ZJH 当前行输入指针
            float* pOutRow = pOut + b * nClasses;  // 20260319 ZJH 当前行输出指针
            // 20260319 ZJH 查找当前行最大值，用于减去保证数值稳定
            float fMax = pRow[0];
            for (int j = 1; j < nClasses; ++j)
                if (pRow[j] > fMax) fMax = pRow[j];
            // 20260319 ZJH 对每个元素计算 exp(x - max) 并累加求和
            float fSum = 0.0f;
            for (int j = 0; j < nClasses; ++j) {
                pOutRow[j] = std::exp(pRow[j] - fMax);  // 20260319 ZJH 减最大值后取指数
                fSum += pOutRow[j];  // 20260319 ZJH 累加指数值
            }
            // 20260319 ZJH 归一化：除以指数之和，使概率总和为 1
            for (int j = 0; j < nClasses; ++j)
                pOutRow[j] /= fSum;
        }
    }

    // 20260319 ZJH 交叉熵损失：-sum(target * log(pred)) / batch
    // target 为 one-hot 编码 [nBatch, nClasses]，pred 为 softmax 输出
    // 返回批次平均交叉熵损失值
    static float crossEntropy(const float* pPred, const float* pTarget, int nBatch, int nClasses) {
        float fLoss = 0.0f;  // 20260319 ZJH 累积损失
        for (int b = 0; b < nBatch; ++b) {
            for (int j = 0; j < nClasses; ++j) {
                // 20260319 ZJH 仅对 target > 0.5 的位置（one-hot 中为 1 的类别）计算损失
                if (pTarget[b * nClasses + j] > 0.5f) {
                    float fP = pPred[b * nClasses + j];  // 20260319 ZJH 预测概率
                    if (fP < 1e-7f) fP = 1e-7f;  // 20260319 ZJH 钳位防止 log(0)
                    fLoss -= std::log(fP);  // 20260319 ZJH 累加负对数概率
                }
            }
        }
        return fLoss / static_cast<float>(nBatch);  // 20260319 ZJH 返回批次平均损失
    }

    // 20260319 ZJH Softmax + 交叉熵联合反向：grad = (softmax_output - target) / batch
    // 联合计算避免分别求 softmax 和 CE 的梯度，数值更稳定且计算更简单
    // 20260324 ZJH 优化：大批次启用 OpenMP 并行
    static void crossEntropySoftmaxBackward(const float* pSoftmax, const float* pTarget,
                                             float* pGradInput, int nBatch, int nClasses) {
        float fScale = 1.0f / static_cast<float>(nBatch);  // 20260319 ZJH 批次平均缩放因子
        int nTotal = nBatch * nClasses;  // 20260324 ZJH 总元素数
        if (nTotal > 1024) {
            // 20260324 ZJH 大数组启用 OpenMP 并行
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nTotal; ++i) {
                pGradInput[i] = (pSoftmax[i] - pTarget[i]) * fScale;
            }
        } else {
            for (int i = 0; i < nTotal; ++i) {
                // 20260319 ZJH 联合梯度公式：(softmax - one_hot) / batch_size
                pGradInput[i] = (pSoftmax[i] - pTarget[i]) * fScale;
            }
        }
    }

    // ===== Argmax =====

    // 20260319 ZJH 逐行 argmax：返回每行最大值的索引
    // pData: [nBatch, nClasses] 输入，pOut: [nBatch] 输出索引数组
    static void argmax(const float* pData, int* pOut, int nBatch, int nClasses) {
        for (int b = 0; b < nBatch; ++b) {
            int nBestIdx = 0;  // 20260319 ZJH 当前行最大值索引
            float fBest = pData[b * nClasses];  // 20260319 ZJH 当前行最大值
            for (int j = 1; j < nClasses; ++j) {
                if (pData[b * nClasses + j] > fBest) {
                    fBest = pData[b * nClasses + j];  // 20260319 ZJH 更新最大值
                    nBestIdx = j;  // 20260319 ZJH 更新最大值索引
                }
            }
            pOut[b] = nBestIdx;  // 20260319 ZJH 记录当前行的 argmax 结果
        }
    }

    // ===== 广播加法 =====

    // 20260319 ZJH 行广播加法：matOut[b, j] = matA[b, j] + vecBias[j]
    // 用于全连接层的偏置加法：matA 形状 [nBatch, nCols]，vecBias 形状 [nCols]
    static void addBias(const float* pA, const float* pBias, float* pOut, int nBatch, int nCols) {
        for (int b = 0; b < nBatch; ++b) {
            for (int j = 0; j < nCols; ++j) {
                // 20260319 ZJH 每行的每个元素加上对应列的偏置
                pOut[b * nCols + j] = pA[b * nCols + j] + pBias[j];
            }
        }
    }

    // ===== 数据拷贝 =====
    // 20260319 ZJH 连续内存拷贝：pSrc -> pDst，共 nCount 个 float
    static void copy(const float* pSrc, float* pDst, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pDst[i] = pSrc[i];
    }

    // ===== Dilated Conv2d =====

    // 20260320 ZJH 膨胀卷积前向：支持 dilation（空洞率）参数
    // 膨胀卷积在核元素之间插入空洞，有效感受野 = KH + (KH-1)*(dilation-1)
    // pInput: [N, Cin, H, W]  pWeight: [Cout, Cin/groups, KH, KW]  pBias: [Cout]
    // 20260324 ZJH 优化：OpenMP 在 batch × group × output_channel 三层循环上并行
    static void dilatedConv2d(const float* pInput, const float* pWeight, const float* pBias,
                               float* pOutput,
                               int nBatch, int nCin, int nH, int nW,
                               int nCout, int nKH, int nKW,
                               int nStride, int nPad, int nDilation, int nGroups = 1) {
        int nEffKH = nKH + (nKH - 1) * (nDilation - 1);  // 20260320 ZJH 有效核高度
        int nEffKW = nKW + (nKW - 1) * (nDilation - 1);  // 20260320 ZJH 有效核宽度
        int nHout = (nH + 2 * nPad - nEffKH) / nStride + 1;
        int nWout = (nW + 2 * nPad - nEffKW) / nStride + 1;
        int nOutSize = nBatch * nCout * nHout * nWout;
        std::memset(pOutput, 0, static_cast<size_t>(nOutSize) * sizeof(float));  // 20260324 ZJH 用 memset 代替循环清零

        int nCinPerGroup = nCin / nGroups;    // 20260320 ZJH 每组输入通道数
        int nCoutPerGroup = nCout / nGroups;  // 20260320 ZJH 每组输出通道数

        // 20260324 ZJH 展平三层嵌套循环为单层，便于 OpenMP collapse 并行
        int nTotalTasks = nBatch * nGroups * nCoutPerGroup;  // 20260324 ZJH 总任务数

        #pragma omp parallel for schedule(dynamic, 1)
        for (int nTask = 0; nTask < nTotalTasks; ++nTask) {
            // 20260324 ZJH 从展平索引恢复 batch/group/output_channel 三维索引
            int n = nTask / (nGroups * nCoutPerGroup);        // 20260324 ZJH 批次索引
            int nRem = nTask % (nGroups * nCoutPerGroup);     // 20260324 ZJH 余数
            int g = nRem / nCoutPerGroup;                     // 20260324 ZJH 组索引
            int co = nRem % nCoutPerGroup;                    // 20260324 ZJH 组内输出通道索引
            int nAbsCo = g * nCoutPerGroup + co;              // 20260320 ZJH 绝对输出通道索引

            for (int oh = 0; oh < nHout; ++oh)
            for (int ow = 0; ow < nWout; ++ow) {
                float fSum = 0.0f;
                for (int ci = 0; ci < nCinPerGroup; ++ci) {
                    int nAbsCi = g * nCinPerGroup + ci;  // 20260320 ZJH 绝对输入通道索引
                    for (int kh = 0; kh < nKH; ++kh)
                    for (int kw = 0; kw < nKW; ++kw) {
                        // 20260320 ZJH 膨胀：核元素间隔 dilation
                        int nIh = oh * nStride - nPad + kh * nDilation;
                        int nIw = ow * nStride - nPad + kw * nDilation;
                        if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                            int nInIdx = ((n * nCin + nAbsCi) * nH + nIh) * nW + nIw;
                            int nWIdx = ((nAbsCo * nCinPerGroup + ci) * nKH + kh) * nKW + kw;
                            fSum += pInput[nInIdx] * pWeight[nWIdx];
                        }
                    }
                }
                if (pBias) fSum += pBias[nAbsCo];
                pOutput[((n * nCout + nAbsCo) * nHout + oh) * nWout + ow] = fSum;
            }
        }
    }

    // ===== GlobalAvgPool2d =====

    // 20260320 ZJH 全局平均池化：[N, C, H, W] -> [N, C, 1, 1]
    // 20260324 ZJH 优化：OpenMP 在 batch × channel 上并行
    static void globalAvgPool2d(const float* pInput, float* pOutput,
                                 int nBatch, int nChannels, int nH, int nW) {
        int nSpatial = nH * nW;
        float fInvSpatial = 1.0f / static_cast<float>(nSpatial);
        int nTotal = nBatch * nChannels;  // 20260324 ZJH 总任务数
        #pragma omp parallel for schedule(static)
        for (int nIdx = 0; nIdx < nTotal; ++nIdx) {
            // 20260324 ZJH 展平 batch × channel 循环为单层并行
            float fSum = 0.0f;
            for (int s = 0; s < nSpatial; ++s)
                fSum += pInput[nIdx * nSpatial + s];
            pOutput[nIdx] = fSum * fInvSpatial;
        }
    }

    // ===== DepthwiseConv2d =====

    // 20260320 ZJH 深度可分离卷积前向：每个通道独立卷积（groups=channels）
    static void depthwiseConv2d(const float* pInput, const float* pWeight, const float* pBias,
                                 float* pOutput,
                                 int nBatch, int nChannels, int nH, int nW,
                                 int nKH, int nKW, int nStride, int nPad) {
        // 20260320 ZJH 深度卷积 = 分组卷积（groups=channels, CoutPerGroup=1）
        dilatedConv2d(pInput, pWeight, pBias, pOutput,
                      nBatch, nChannels, nH, nW,
                      nChannels, nKH, nKW, nStride, nPad, 1, nChannels);
    }

    // ===== im2col 辅助函数 =====

    // 20260324 ZJH im2col：将输入张量的感受野区域展开为列矩阵
    // 这是 conv2d 的 GEMM 加速核心——将分散的卷积核覆盖区域重排为连续的列
    // 使 conv2d 可以转化为单次矩阵乘法，充分利用 AVX2/OpenMP 优化的 matmul
    // pInput: 单个样本的输入 [Cin, H, W]
    // pCol: 输出列矩阵 [Cin*KH*KW, Hout*Wout]
    // padding 区域自动填充为 0
    static void im2col(const float* pInput, int nCin, int nH, int nW,
                       int nKH, int nKW, int nPadH, int nPadW,
                       int nStrideH, int nStrideW, float* pCol) {
        int nHout = (nH + 2 * nPadH - nKH) / nStrideH + 1;  // 20260324 ZJH 输出高度
        int nWout = (nW + 2 * nPadW - nKW) / nStrideW + 1;  // 20260324 ZJH 输出宽度
        int nColCols = nHout * nWout;  // 20260324 ZJH 列矩阵的列数（每列对应一个输出位置）

        // 20260324 ZJH 三层循环遍历所有核通道和核位置
        // 外层按 Cin*KH*KW 组织行，内层按 Hout*Wout 组织列
        for (int c = 0; c < nCin; ++c) {           // 20260324 ZJH 遍历输入通道
            for (int kh = 0; kh < nKH; ++kh) {     // 20260324 ZJH 遍历核高度
                for (int kw = 0; kw < nKW; ++kw) { // 20260324 ZJH 遍历核宽度
                    // 20260324 ZJH 当前行在列矩阵中的行索引
                    int nRow = (c * nKH + kh) * nKW + kw;
                    for (int oh = 0; oh < nHout; ++oh) {   // 20260324 ZJH 遍历输出行
                        for (int ow = 0; ow < nWout; ++ow) { // 20260324 ZJH 遍历输出列
                            // 20260324 ZJH 计算输入图像中的实际坐标
                            int nIh = oh * nStrideH - nPadH + kh;  // 20260324 ZJH 输入行坐标
                            int nIw = ow * nStrideW - nPadW + kw;  // 20260324 ZJH 输入列坐标
                            int nCol = oh * nWout + ow;  // 20260324 ZJH 列矩阵中的列索引
                            // 20260324 ZJH 边界检查：padding 区域填零
                            if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                pCol[nRow * nColCols + nCol] = pInput[(c * nH + nIh) * nW + nIw];
                            } else {
                                pCol[nRow * nColCols + nCol] = 0.0f;  // 20260324 ZJH 零填充
                            }
                        }
                    }
                }
            }
        }
    }

    // 20260324 ZJH col2im：im2col 的逆操作，将列矩阵的梯度散布回输入形状
    // 用于 conv2d 反向传播中的 gradInput 计算
    // pCol: 梯度列矩阵 [Cin*KH*KW, Hout*Wout]
    // pInput: 输出累加到的输入梯度 [Cin, H, W]（需预先置零）
    static void col2im(const float* pCol, int nCin, int nH, int nW,
                       int nKH, int nKW, int nPadH, int nPadW,
                       int nStrideH, int nStrideW, float* pInput) {
        int nHout = (nH + 2 * nPadH - nKH) / nStrideH + 1;  // 20260324 ZJH 输出高度
        int nWout = (nW + 2 * nPadW - nKW) / nStrideW + 1;  // 20260324 ZJH 输出宽度
        int nColCols = nHout * nWout;  // 20260324 ZJH 列矩阵的列数

        // 20260324 ZJH 遍历列矩阵每个元素，将值累加回输入对应位置
        for (int c = 0; c < nCin; ++c) {
            for (int kh = 0; kh < nKH; ++kh) {
                for (int kw = 0; kw < nKW; ++kw) {
                    int nRow = (c * nKH + kh) * nKW + kw;  // 20260324 ZJH 列矩阵行索引
                    for (int oh = 0; oh < nHout; ++oh) {
                        for (int ow = 0; ow < nWout; ++ow) {
                            int nIh = oh * nStrideH - nPadH + kh;  // 20260324 ZJH 输入行坐标
                            int nIw = ow * nStrideW - nPadW + kw;  // 20260324 ZJH 输入列坐标
                            int nCol = oh * nWout + ow;  // 20260324 ZJH 列矩阵列索引
                            // 20260324 ZJH 仅对有效输入位置累加梯度
                            if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                pInput[(c * nH + nIh) * nW + nIw] += pCol[nRow * nColCols + nCol];
                            }
                        }
                    }
                }
            }
        }
    }

    // ===== Conv2d =====

    // 20260319 ZJH Conv2d 前向：NCHW 格式
    // 20260324 ZJH 优化：im2col + GEMM 方法替代朴素 7 重循环直接卷积
    // 核心思路：将分散的卷积窗口数据通过 im2col 重排为连续的列矩阵
    // 然后将卷积运算转化为 weight[Cout, Cin*KH*KW] × col[Cin*KH*KW, Hout*Wout]
    // 这样可以充分利用已优化的 matmul（AVX2 + OpenMP），获得数量级的性能提升
    // pInput: [N, Cin, H, W]  pWeight: [Cout, Cin, KH, KW]  pBias: [Cout] 或 nullptr
    // pOutput: [N, Cout, Hout, Wout]
    // Hout = (H + 2*pad - KH) / stride + 1, Wout = (W + 2*pad - KW) / stride + 1
    static void conv2d(const float* pInput, const float* pWeight, const float* pBias,
                       float* pOutput,
                       int nBatch, int nCin, int nH, int nW,
                       int nCout, int nKH, int nKW,
                       int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        int nColRows = nCin * nKH * nKW;   // 20260324 ZJH im2col 列矩阵行数
        int nColCols = nHout * nWout;       // 20260324 ZJH im2col 列矩阵列数

        // 20260324 ZJH 预分配 im2col 缓冲区（使用内存池对齐分配，跨 batch 复用）
        size_t nColBytes = static_cast<size_t>(nColRows) * nColCols * sizeof(float);
        float* pCol = poolAllocAligned(nColBytes);  // 20260324 ZJH 从内存池分配

        for (int n = 0; n < nBatch; ++n) {  // 20260319 ZJH 遍历批次
            // 20260324 ZJH Step 1: im2col — 将当前 batch 的输入展开为列矩阵
            im2col(pInput + n * nCin * nH * nW, nCin, nH, nW,
                   nKH, nKW, nPad, nPad, nStride, nStride, pCol);

            // 20260324 ZJH Step 2: GEMM — weight[Cout, Cin*KH*KW] × col[Cin*KH*KW, Hout*Wout]
            // 输出直接写入 pOutput 对应 batch 的位置
            matmul(pWeight, pCol, pOutput + n * nCout * nColCols,
                   nCout, nColRows, nColCols);

            // 20260324 ZJH Step 3: 加偏置（如果有）
            if (pBias) {
                // 20260324 ZJH 每个输出通道加上对应的偏置值
                for (int c = 0; c < nCout; ++c) {
                    float fBias = pBias[c];  // 20260324 ZJH 当前通道偏置
                    float* pOutChannel = pOutput + n * nCout * nColCols + c * nColCols;
                    for (int hw = 0; hw < nColCols; ++hw) {
                        pOutChannel[hw] += fBias;  // 20260324 ZJH 逐元素加偏置
                    }
                }
            }
        }

        // 20260324 ZJH 归还 im2col 缓冲区到内存池
        poolFreeAligned(pCol, nColBytes);
    }

    // 20260319 ZJH Conv2d 反向（对输入求梯度）：gradInput = 转置卷积
    // 20260324 ZJH 优化：weight^T × gradOutput → col，再 col2im 还原为 gradInput
    // 原理：前向 output = weight × col，反向对 col 求梯度 = weight^T × gradOutput
    // 再通过 col2im 将 col 梯度散布回输入空间
    // pGradOutput: [N, Cout, Hout, Wout]  pWeight: [Cout, Cin, KH, KW]
    // pGradInput: [N, Cin, H, W]（需预先置零）
    static void conv2dBackwardInput(const float* pGradOutput, const float* pWeight,
                                     float* pGradInput,
                                     int nBatch, int nCin, int nH, int nW,
                                     int nCout, int nKH, int nKW,
                                     int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        int nColRows = nCin * nKH * nKW;   // 20260324 ZJH 列矩阵行数
        int nColCols = nHout * nWout;       // 20260324 ZJH 列矩阵列数

        // 20260319 ZJH 初始化 gradInput 为零
        int nInputSize = nBatch * nCin * nH * nW;
        std::memset(pGradInput, 0, static_cast<size_t>(nInputSize) * sizeof(float));

        // 20260324 ZJH 转置权重矩阵：weight[Cout, Cin*KH*KW] → weightT[Cin*KH*KW, Cout]
        size_t nWeightTBytes = static_cast<size_t>(nColRows) * nCout * sizeof(float);
        float* pWeightT = poolAllocAligned(nWeightTBytes);
        transpose2d(pWeight, pWeightT, nCout, nColRows);  // 20260324 ZJH 转置权重

        // 20260324 ZJH 分配 col 缓冲区
        size_t nColBytes = static_cast<size_t>(nColRows) * nColCols * sizeof(float);
        float* pCol = poolAllocAligned(nColBytes);

        for (int n = 0; n < nBatch; ++n) {
            // 20260324 ZJH Step 1: col_grad = weight^T × gradOutput
            // weight^T[Cin*KH*KW, Cout] × gradOutput[Cout, Hout*Wout] → col[Cin*KH*KW, Hout*Wout]
            matmul(pWeightT, pGradOutput + n * nCout * nColCols, pCol,
                   nColRows, nCout, nColCols);

            // 20260324 ZJH Step 2: col2im — 将列矩阵梯度散布回输入空间
            col2im(pCol, nCin, nH, nW, nKH, nKW, nPad, nPad, nStride, nStride,
                   pGradInput + n * nCin * nH * nW);
        }

        // 20260324 ZJH 归还缓冲区到内存池
        poolFreeAligned(pCol, nColBytes);
        poolFreeAligned(pWeightT, nWeightTBytes);
    }

    // 20260319 ZJH Conv2d 反向（对权重和偏置求梯度）
    // 20260324 ZJH 优化：gradOutput × col^T → gradWeight（im2col + GEMM 方法）
    // 原理：前向 output = weight × col，反向对 weight 求梯度 = gradOutput × col^T
    // pInput: [N, Cin, H, W]  pGradOutput: [N, Cout, Hout, Wout]
    // pGradWeight: [Cout, Cin, KH, KW]  pGradBias: [Cout] 或 nullptr
    static void conv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                      float* pGradWeight, float* pGradBias,
                                      int nBatch, int nCin, int nH, int nW,
                                      int nCout, int nKH, int nKW,
                                      int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        int nColRows = nCin * nKH * nKW;   // 20260324 ZJH 列矩阵行数
        int nColCols = nHout * nWout;       // 20260324 ZJH 列矩阵列数

        // 20260319 ZJH 初始化 gradWeight 为零
        int nWeightSize = nCout * nCin * nKH * nKW;
        std::memset(pGradWeight, 0, static_cast<size_t>(nWeightSize) * sizeof(float));
        // 20260319 ZJH 初始化 gradBias 为零
        if (pGradBias) {
            std::memset(pGradBias, 0, static_cast<size_t>(nCout) * sizeof(float));
        }

        // 20260324 ZJH 分配 im2col 和 col^T 缓冲区
        size_t nColBytes = static_cast<size_t>(nColRows) * nColCols * sizeof(float);
        float* pCol = poolAllocAligned(nColBytes);       // 20260324 ZJH im2col 缓冲区
        float* pColT = poolAllocAligned(nColBytes);      // 20260324 ZJH col 转置缓冲区

        // 20260324 ZJH 临时权重梯度缓冲区（每 batch 累加到 pGradWeight）
        size_t nGradWBytes = static_cast<size_t>(nWeightSize) * sizeof(float);
        float* pTempGradW = poolAllocAligned(nGradWBytes);

        for (int n = 0; n < nBatch; ++n) {
            // 20260324 ZJH Step 1: im2col — 将输入展开为列矩阵
            im2col(pInput + n * nCin * nH * nW, nCin, nH, nW,
                   nKH, nKW, nPad, nPad, nStride, nStride, pCol);

            // 20260324 ZJH Step 2: 转置 col — col[Cin*KH*KW, Hout*Wout] → colT[Hout*Wout, Cin*KH*KW]
            transpose2d(pCol, pColT, nColRows, nColCols);

            // 20260324 ZJH Step 3: gradWeight_batch = gradOutput × col^T
            // gradOutput[Cout, Hout*Wout] × colT[Hout*Wout, Cin*KH*KW] → tempGradW[Cout, Cin*KH*KW]
            matmul(pGradOutput + n * nCout * nColCols, pColT, pTempGradW,
                   nCout, nColCols, nColRows);

            // 20260324 ZJH Step 4: 累加到总 gradWeight（多 batch 的梯度求和）
            for (int i = 0; i < nWeightSize; ++i) {
                pGradWeight[i] += pTempGradW[i];
            }

            // 20260319 ZJH 偏置梯度：所有位置的梯度之和
            if (pGradBias) {
                const float* pGradOut = pGradOutput + n * nCout * nColCols;
                for (int co = 0; co < nCout; ++co) {
                    float fBiasGrad = 0.0f;  // 20260324 ZJH 当前通道偏置梯度
                    for (int hw = 0; hw < nColCols; ++hw) {
                        fBiasGrad += pGradOut[co * nColCols + hw];
                    }
                    pGradBias[co] += fBiasGrad;  // 20260324 ZJH 累加偏置梯度
                }
            }
        }

        // 20260324 ZJH 归还缓冲区到内存池
        poolFreeAligned(pTempGradW, nGradWBytes);
        poolFreeAligned(pColT, nColBytes);
        poolFreeAligned(pCol, nColBytes);
    }

    // ===== BatchNorm2d =====

    // 20260319 ZJH BatchNorm2d 前向：NCHW 格式
    // 训练时：计算当前 batch 的均值和方差，更新 running 统计量，归一化后缩放偏移
    // 评估时：使用 running 统计量进行归一化
    // pSavedMean/pSavedInvStd: 保存训练时的均值和逆标准差，用于反向传播
    // 20260324 ZJH 优化：通道级 OpenMP 并行（每个通道的统计量计算相互独立）
    static void batchNorm2d(const float* pInput, float* pOutput,
                             const float* pGamma, const float* pBeta,
                             float* pRunMean, float* pRunVar,
                             float* pSavedMean, float* pSavedInvStd,
                             int nBatch, int nChannels, int nH, int nW,
                             float fEps, float fMomentum, bool bTraining) {
        int nSpatial = nH * nW;  // 20260319 ZJH 空间维度大小
        int nCount = nBatch * nSpatial;  // 20260319 ZJH 每通道的元素总数

        // 20260324 ZJH OpenMP 在通道维度上并行，每个通道独立计算均值/方差/归一化
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nChannels; ++c) {  // 20260319 ZJH 遍历每个通道
            float fMean = 0.0f;  // 20260319 ZJH 通道均值
            float fVar = 0.0f;   // 20260319 ZJH 通道方差

            if (bTraining) {
                // 20260319 ZJH 训练模式：从当前 batch 计算均值
                for (int n = 0; n < nBatch; ++n) {
                    for (int s = 0; s < nSpatial; ++s) {
                        int nIdx = (n * nChannels + c) * nSpatial + s;  // 20260319 ZJH NCHW 索引
                        fMean += pInput[nIdx];
                    }
                }
                fMean /= static_cast<float>(nCount);  // 20260319 ZJH 求均值

                // 20260319 ZJH 计算方差
                for (int n = 0; n < nBatch; ++n) {
                    for (int s = 0; s < nSpatial; ++s) {
                        int nIdx = (n * nChannels + c) * nSpatial + s;
                        float fDiff = pInput[nIdx] - fMean;
                        fVar += fDiff * fDiff;
                    }
                }
                fVar /= static_cast<float>(nCount);  // 20260319 ZJH 求方差

                // 20260319 ZJH 保存均值和逆标准差，用于反向传播
                pSavedMean[c] = fMean;
                pSavedInvStd[c] = 1.0f / std::sqrt(fVar + fEps);

                // 20260319 ZJH 更新 running 统计量（指数移动平均）
                pRunMean[c] = (1.0f - fMomentum) * pRunMean[c] + fMomentum * fMean;
                pRunVar[c] = (1.0f - fMomentum) * pRunVar[c] + fMomentum * fVar;
            } else {
                // 20260319 ZJH 评估模式：使用 running 统计量
                fMean = pRunMean[c];
                fVar = pRunVar[c];
                // 20260319 ZJH 仍然保存用于一致性
                pSavedMean[c] = fMean;
                pSavedInvStd[c] = 1.0f / std::sqrt(fVar + fEps);
            }

            // 20260319 ZJH 归一化 + 缩放 + 偏移：y = gamma * (x - mean) / sqrt(var + eps) + beta
            float fInvStd = 1.0f / std::sqrt(fVar + fEps);  // 20260319 ZJH 逆标准差
            for (int n = 0; n < nBatch; ++n) {
                for (int s = 0; s < nSpatial; ++s) {
                    int nIdx = (n * nChannels + c) * nSpatial + s;
                    float fNorm = (pInput[nIdx] - fMean) * fInvStd;  // 20260319 ZJH 归一化
                    pOutput[nIdx] = pGamma[c] * fNorm + pBeta[c];  // 20260319 ZJH 缩放 + 偏移
                }
            }
        }
    }

    // 20260319 ZJH BatchNorm2d 反向传播
    // 计算 gradInput, gradGamma, gradBeta
    // 20260324 ZJH 优化：通道级 OpenMP 并行
    static void batchNorm2dBackward(const float* pGradOutput, const float* pInput,
                                     const float* pSavedMean, const float* pSavedInvStd,
                                     const float* pGamma,
                                     float* pGradInput, float* pGradGamma, float* pGradBeta,
                                     int nBatch, int nChannels, int nH, int nW) {
        int nSpatial = nH * nW;  // 20260319 ZJH 空间维度大小
        int nCount = nBatch * nSpatial;  // 20260319 ZJH 每通道的元素总数
        float fInvCount = 1.0f / static_cast<float>(nCount);  // 20260319 ZJH 元素数倒数

        // 20260324 ZJH OpenMP 在通道维度上并行，每个通道独立计算梯度
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nChannels; ++c) {  // 20260319 ZJH 遍历每个通道
            float fMean = pSavedMean[c];        // 20260319 ZJH 前向保存的均值
            float fInvStd = pSavedInvStd[c];    // 20260319 ZJH 前向保存的逆标准差
            float fGamma = pGamma[c];            // 20260319 ZJH 缩放参数

            // 20260319 ZJH 第一步：计算 gradGamma 和 gradBeta
            float fGradGamma = 0.0f;
            float fGradBeta = 0.0f;
            for (int n = 0; n < nBatch; ++n) {
                for (int s = 0; s < nSpatial; ++s) {
                    int nIdx = (n * nChannels + c) * nSpatial + s;
                    float fXhat = (pInput[nIdx] - fMean) * fInvStd;  // 20260319 ZJH 归一化值
                    fGradGamma += pGradOutput[nIdx] * fXhat;  // 20260319 ZJH gradGamma = sum(dL/dy * xhat)
                    fGradBeta += pGradOutput[nIdx];  // 20260319 ZJH gradBeta = sum(dL/dy)
                }
            }
            pGradGamma[c] = fGradGamma;
            pGradBeta[c] = fGradBeta;

            // 20260319 ZJH 第二步：计算 gradInput
            // gradInput = gamma * invStd * (gradOutput - mean(gradOutput) - xhat * mean(gradOutput * xhat))
            // 简化为：gradInput = gamma * invStd / N * (N * dL/dy - sum(dL/dy) - xhat * sum(dL/dy * xhat))
            for (int n = 0; n < nBatch; ++n) {
                for (int s = 0; s < nSpatial; ++s) {
                    int nIdx = (n * nChannels + c) * nSpatial + s;
                    float fXhat = (pInput[nIdx] - fMean) * fInvStd;
                    // 20260319 ZJH BN 反向公式
                    pGradInput[nIdx] = fGamma * fInvStd * fInvCount *
                        (static_cast<float>(nCount) * pGradOutput[nIdx] - fGradBeta - fXhat * fGradGamma);
                }
            }
        }
    }

    // ===== MaxPool2d =====

    // 20260319 ZJH MaxPool2d 前向：保存最大值索引用于反向传播
    // pInput: [N, C, H, W]  pOutput: [N, C, Hout, Wout]  pIndices: [N, C, Hout, Wout]
    // 20260324 ZJH 优化：OpenMP 在 batch × channel 上并行
    static void maxPool2d(const float* pInput, float* pOutput, int* pIndices,
                           int nBatch, int nChannels, int nH, int nW,
                           int nKH, int nKW, int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        int nTotal = nBatch * nChannels;  // 20260324 ZJH batch × channel 总任务数

        // 20260324 ZJH 展平 batch × channel 循环为单层并行
        #pragma omp parallel for schedule(static)
        for (int nIdx = 0; nIdx < nTotal; ++nIdx) {
            int n = nIdx / nChannels;   // 20260324 ZJH 批次索引
            int c = nIdx % nChannels;   // 20260324 ZJH 通道索引
            for (int oh = 0; oh < nHout; ++oh) {
                for (int ow = 0; ow < nWout; ++ow) {
                    int nOutIdx = ((n * nChannels + c) * nHout + oh) * nWout + ow;
                    float fMax = -1e30f;  // 20260319 ZJH 初始化为极小值
                    int nMaxIdx = -1;      // 20260319 ZJH 最大值在输入中的平面索引
                    for (int kh = 0; kh < nKH; ++kh) {
                        for (int kw = 0; kw < nKW; ++kw) {
                            int nIh = oh * nStride - nPad + kh;
                            int nIw = ow * nStride - nPad + kw;
                            if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                int nInIdx = ((n * nChannels + c) * nH + nIh) * nW + nIw;
                                if (pInput[nInIdx] > fMax) {
                                    fMax = pInput[nInIdx];  // 20260319 ZJH 更新最大值
                                    nMaxIdx = nInIdx;  // 20260319 ZJH 记录最大值索引
                                }
                            }
                        }
                    }
                    pOutput[nOutIdx] = fMax;      // 20260319 ZJH 写入最大值
                    pIndices[nOutIdx] = nMaxIdx;  // 20260319 ZJH 保存索引
                }
            }
        }
    }

    // 20260319 ZJH MaxPool2d 反向：将梯度散布回最大值位置
    // pGradOutput: [N, C, Hout, Wout]  pIndices: [N, C, Hout, Wout]
    // pGradInput: [N, C, H, W]（需预先置零）
    static void maxPool2dBackward(const float* pGradOutput, const int* pIndices,
                                   float* pGradInput,
                                   int nBatch, int nChannels, int nHout, int nWout,
                                   int nH, int nW) {
        // 20260319 ZJH 初始化 gradInput 为零
        int nInputSize = nBatch * nChannels * nH * nW;
        for (int i = 0; i < nInputSize; ++i) pGradInput[i] = 0.0f;
        // 20260319 ZJH 遍历每个输出位置，将梯度写到对应的最大值输入位置
        int nOutSize = nBatch * nChannels * nHout * nWout;
        for (int i = 0; i < nOutSize; ++i) {
            if (pIndices[i] >= 0) {
                pGradInput[pIndices[i]] += pGradOutput[i];  // 20260319 ZJH 梯度累加到最大值位置
            }
        }
    }

    // ===== AvgPool2d =====

    // 20260319 ZJH AvgPool2d 前向：窗口内取均值
    // pInput: [N, C, H, W]  pOutput: [N, C, Hout, Wout]
    // 20260324 ZJH 优化：OpenMP 在 batch × channel 上并行
    static void avgPool2d(const float* pInput, float* pOutput,
                           int nBatch, int nChannels, int nH, int nW,
                           int nKH, int nKW, int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        int nTotal = nBatch * nChannels;  // 20260324 ZJH batch × channel 总任务数

        // 20260324 ZJH 展平 batch × channel 循环为单层并行
        #pragma omp parallel for schedule(static)
        for (int nIdx = 0; nIdx < nTotal; ++nIdx) {
            int n = nIdx / nChannels;   // 20260324 ZJH 批次索引
            int c = nIdx % nChannels;   // 20260324 ZJH 通道索引
            for (int oh = 0; oh < nHout; ++oh) {
                for (int ow = 0; ow < nWout; ++ow) {
                    float fSum = 0.0f;  // 20260319 ZJH 窗口内元素之和
                    int nValidCount = 0;  // 20260319 ZJH 有效元素计数（不含 padding）
                    for (int kh = 0; kh < nKH; ++kh) {
                        for (int kw = 0; kw < nKW; ++kw) {
                            int nIh = oh * nStride - nPad + kh;
                            int nIw = ow * nStride - nPad + kw;
                            if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                int nInIdx = ((n * nChannels + c) * nH + nIh) * nW + nIw;
                                fSum += pInput[nInIdx];
                                nValidCount++;
                            }
                        }
                    }
                    int nOutIdx = ((n * nChannels + c) * nHout + oh) * nWout + ow;
                    // 20260319 ZJH 使用固定核大小做平均（PyTorch 默认 count_include_pad=true 风格，无 pad 时等价）
                    pOutput[nOutIdx] = fSum / static_cast<float>(nKH * nKW);
                }
            }
        }
    }

    // 20260319 ZJH AvgPool2d 反向：将梯度均匀散布回窗口内各位置
    static void avgPool2dBackward(const float* pGradOutput, float* pGradInput,
                                   int nBatch, int nChannels, int nH, int nW,
                                   int nHout, int nWout,
                                   int nKH, int nKW, int nStride, int nPad) {
        // 20260319 ZJH 初始化 gradInput 为零
        int nInputSize = nBatch * nChannels * nH * nW;
        for (int i = 0; i < nInputSize; ++i) pGradInput[i] = 0.0f;

        float fScale = 1.0f / static_cast<float>(nKH * nKW);  // 20260319 ZJH 均值的缩放因子

        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannels; ++c) {
                for (int oh = 0; oh < nHout; ++oh) {
                    for (int ow = 0; ow < nWout; ++ow) {
                        int nOutIdx = ((n * nChannels + c) * nHout + oh) * nWout + ow;
                        float fGrad = pGradOutput[nOutIdx] * fScale;  // 20260319 ZJH 均分到窗口内
                        for (int kh = 0; kh < nKH; ++kh) {
                            for (int kw = 0; kw < nKW; ++kw) {
                                int nIh = oh * nStride - nPad + kh;
                                int nIw = ow * nStride - nPad + kw;
                                if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                    int nInIdx = ((n * nChannels + c) * nH + nIh) * nW + nIw;
                                    pGradInput[nInIdx] += fGrad;  // 20260319 ZJH 梯度累加
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ===== ConvTranspose2d =====

    // 20260320 ZJH ConvTranspose2d 前向：转置卷积（反卷积），用于上采样
    // pInput: [N, Cin, Hin, Win]  pWeight: [Cin, Cout, KH, KW]  pBias: [Cout] 或 nullptr
    // pOutput: [N, Cout, Hout, Wout]，其中 Hout = (Hin-1)*stride - 2*pad + KH
    static void convTranspose2d(const float* pInput, const float* pWeight, const float* pBias,
                                 float* pOutput,
                                 int nBatch, int nCin, int nHin, int nWin,
                                 int nCout, int nKH, int nKW,
                                 int nStride, int nPad) {
        int nHout = (nHin - 1) * nStride - 2 * nPad + nKH;  // 20260320 ZJH 输出高度
        int nWout = (nWin - 1) * nStride - 2 * nPad + nKW;  // 20260320 ZJH 输出宽度
        int nOutSize = nBatch * nCout * nHout * nWout;  // 20260320 ZJH 输出总元素数
        // 20260320 ZJH 初始化输出为零
        for (int i = 0; i < nOutSize; ++i) pOutput[i] = 0.0f;

        // 20260320 ZJH 遍历输入每个位置，将其值散布到输出对应位置（转置卷积核心逻辑）
        for (int n = 0; n < nBatch; ++n)       // 20260320 ZJH 遍历批次
        for (int ci = 0; ci < nCin; ++ci)      // 20260320 ZJH 遍历输入通道
        for (int ih = 0; ih < nHin; ++ih)      // 20260320 ZJH 遍历输入行
        for (int iw = 0; iw < nWin; ++iw) {   // 20260320 ZJH 遍历输入列
            float fVal = pInput[((n * nCin + ci) * nHin + ih) * nWin + iw];  // 20260320 ZJH 输入值
            for (int co = 0; co < nCout; ++co)      // 20260320 ZJH 遍历输出通道
            for (int kh = 0; kh < nKH; ++kh)        // 20260320 ZJH 遍历核高度
            for (int kw = 0; kw < nKW; ++kw) {      // 20260320 ZJH 遍历核宽度
                int oh = ih * nStride - nPad + kh;   // 20260320 ZJH 输出行坐标
                int ow = iw * nStride - nPad + kw;   // 20260320 ZJH 输出列坐标
                // 20260320 ZJH 边界检查
                if (oh >= 0 && oh < nHout && ow >= 0 && ow < nWout) {
                    // 20260320 ZJH 累加输入值 * 权重到输出位置
                    pOutput[((n * nCout + co) * nHout + oh) * nWout + ow] +=
                        fVal * pWeight[((ci * nCout + co) * nKH + kh) * nKW + kw];
                }
            }
        }
        // 20260320 ZJH 加偏置
        if (pBias) {
            for (int n = 0; n < nBatch; ++n)
            for (int co = 0; co < nCout; ++co)
            for (int oh = 0; oh < nHout; ++oh)
            for (int ow = 0; ow < nWout; ++ow)
                pOutput[((n * nCout + co) * nHout + oh) * nWout + ow] += pBias[co];
        }
    }

    // ===== Upsample (bilinear) =====

    // 20260320 ZJH 双线性上采样：[N,C,H,W] -> [N,C,H*scale,W*scale]
    // 使用半像素中心对齐（align_corners=false 风格）
    static void upsampleBilinear(const float* pInput, float* pOutput,
                                  int nBatch, int nChannels, int nH, int nW, int nScale) {
        int nHout = nH * nScale;  // 20260320 ZJH 输出高度
        int nWout = nW * nScale;  // 20260320 ZJH 输出宽度
        for (int n = 0; n < nBatch; ++n)
        for (int c = 0; c < nChannels; ++c)
        for (int oh = 0; oh < nHout; ++oh)
        for (int ow = 0; ow < nWout; ++ow) {
            // 20260320 ZJH 计算源坐标（半像素中心对齐）
            float fSrcH = (oh + 0.5f) / nScale - 0.5f;
            float fSrcW = (ow + 0.5f) / nScale - 0.5f;
            int h0 = static_cast<int>(std::floor(fSrcH));  // 20260320 ZJH 左上角行
            int w0 = static_cast<int>(std::floor(fSrcW));  // 20260320 ZJH 左上角列
            int h1 = h0 + 1;  // 20260320 ZJH 右下角行
            int w1 = w0 + 1;  // 20260320 ZJH 右下角列
            float fH = fSrcH - h0;  // 20260320 ZJH 行方向插值系数
            float fW = fSrcW - w0;  // 20260320 ZJH 列方向插值系数

            // 20260320 ZJH 取像素值的 lambda，边界时做 clamp
            auto getPixel = [&](int h, int w) -> float {
                h = std::max(0, std::min(h, nH - 1));  // 20260320 ZJH 行 clamp
                w = std::max(0, std::min(w, nW - 1));  // 20260320 ZJH 列 clamp
                return pInput[((n * nChannels + c) * nH + h) * nW + w];
            };

            // 20260320 ZJH 双线性插值公式
            pOutput[((n * nChannels + c) * nHout + oh) * nWout + ow] =
                (1 - fH) * (1 - fW) * getPixel(h0, w0) + (1 - fH) * fW * getPixel(h0, w1) +
                fH * (1 - fW) * getPixel(h1, w0) + fH * fW * getPixel(h1, w1);
        }
    }

    // 20260320 ZJH 双线性上采样反向：将 gradOutput [N,C,Hout,Wout] 的梯度散布回 gradInput [N,C,H,W]
    static void upsampleBilinearBackward(const float* pGradOutput, float* pGradInput,
                                          int nBatch, int nChannels, int nH, int nW, int nScale) {
        int nHout = nH * nScale;  // 20260320 ZJH 输出高度
        int nWout = nW * nScale;  // 20260320 ZJH 输出宽度
        // 20260320 ZJH 初始化 gradInput 为零
        int nInputSize = nBatch * nChannels * nH * nW;
        for (int i = 0; i < nInputSize; ++i) pGradInput[i] = 0.0f;

        for (int n = 0; n < nBatch; ++n)
        for (int c = 0; c < nChannels; ++c)
        for (int oh = 0; oh < nHout; ++oh)
        for (int ow = 0; ow < nWout; ++ow) {
            float fGrad = pGradOutput[((n * nChannels + c) * nHout + oh) * nWout + ow];
            float fSrcH = (oh + 0.5f) / nScale - 0.5f;
            float fSrcW = (ow + 0.5f) / nScale - 0.5f;
            int h0 = static_cast<int>(std::floor(fSrcH));
            int w0 = static_cast<int>(std::floor(fSrcW));
            int h1 = h0 + 1;
            int w1 = w0 + 1;
            float fH = fSrcH - h0;
            float fW = fSrcW - w0;

            // 20260320 ZJH 辅助 lambda：安全累加梯度到输入位置
            auto addGrad = [&](int h, int w, float fWeight) {
                h = std::max(0, std::min(h, nH - 1));
                w = std::max(0, std::min(w, nW - 1));
                pGradInput[((n * nChannels + c) * nH + h) * nW + w] += fGrad * fWeight;
            };

            // 20260320 ZJH 按双线性插值权重分配梯度
            addGrad(h0, w0, (1 - fH) * (1 - fW));
            addGrad(h0, w1, (1 - fH) * fW);
            addGrad(h1, w0, fH * (1 - fW));
            addGrad(h1, w1, fH * fW);
        }
    }

    // ===== Sigmoid =====

    // 20260320 ZJH Sigmoid 前向：out[i] = 1 / (1 + exp(-in[i]))
    // 20260324 ZJH 优化：优先使用 AVX2 SIMD 加速（多项式近似，最大误差 < 0.002）
    static void sigmoid(const float* pIn, float* pOut, size_t nCount) {
        // 20260324 ZJH 优先 AVX2 加速路径（使用快速多项式近似）
        if (SIMDBackend::isAVX2Supported()) {
            SIMDBackend::sigmoidAVX2(pIn, pOut, nCount);
            return;  // 20260324 ZJH SIMD 完成，直接返回
        }
        // 20260324 ZJH 标量回退路径：大数组启用 OpenMP 并行
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i)
                pOut[i] = 1.0f / (1.0f + std::exp(-pIn[i]));
        } else {
            for (size_t i = 0; i < nCount; ++i)
                pOut[i] = 1.0f / (1.0f + std::exp(-pIn[i]));  // 20260320 ZJH S 型激活函数
        }
    }

    // 20260320 ZJH Sigmoid 反向：grad_in[i] = grad_out[i] * out[i] * (1 - out[i])
    // pOutput 是前向的输出值（非输入值）
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void sigmoidBackward(const float* pOutput, const float* pGradOut, float* pGradIn, size_t nCount) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i)
                pGradIn[i] = pGradOut[i] * pOutput[i] * (1.0f - pOutput[i]);
        } else {
            for (size_t i = 0; i < nCount; ++i)
                pGradIn[i] = pGradOut[i] * pOutput[i] * (1.0f - pOutput[i]);  // 20260320 ZJH sigmoid 导数
        }
    }

    // ===== LeakyReLU =====

    // 20260320 ZJH LeakyReLU 前向：正值直通，负值乘以斜率 fSlope
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void leakyRelu(const float* pIn, float* pOut, size_t nCount, float fSlope = 0.01f) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i)
                pOut[i] = pIn[i] > 0 ? pIn[i] : fSlope * pIn[i];
        } else {
            for (size_t i = 0; i < nCount; ++i)
                pOut[i] = pIn[i] > 0 ? pIn[i] : fSlope * pIn[i];  // 20260320 ZJH 负值保留小梯度
        }
    }

    // 20260320 ZJH LeakyReLU 反向：正值梯度直通，负值梯度乘以斜率
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void leakyReluBackward(const float* pIn, const float* pGradOut, float* pGradIn,
                                   size_t nCount, float fSlope = 0.01f) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i)
                pGradIn[i] = pIn[i] > 0 ? pGradOut[i] : fSlope * pGradOut[i];
        } else {
            for (size_t i = 0; i < nCount; ++i)
                pGradIn[i] = pIn[i] > 0 ? pGradOut[i] : fSlope * pGradOut[i];  // 20260320 ZJH 负值区域斜率为 fSlope
        }
    }

    // ===== Concat along channel dim =====

    // 20260320 ZJH 沿通道维度拼接两个张量：[N,C1,H,W] + [N,C2,H,W] -> [N,C1+C2,H,W]
    static void concatChannels(const float* pA, const float* pB, float* pOut,
                                int nBatch, int nC1, int nC2, int nH, int nW) {
        int nSpatial = nH * nW;  // 20260320 ZJH 空间维度大小
        for (int n = 0; n < nBatch; ++n) {
            // 20260320 ZJH 拷贝张量 A 的通道数据
            for (int c = 0; c < nC1; ++c) {
                int nSrcIdx = (n * nC1 + c) * nSpatial;  // 20260320 ZJH A 的源索引
                int nDstIdx = (n * (nC1 + nC2) + c) * nSpatial;  // 20260320 ZJH 输出的目标索引
                for (int s = 0; s < nSpatial; ++s)
                    pOut[nDstIdx + s] = pA[nSrcIdx + s];
            }
            // 20260320 ZJH 拷贝张量 B 的通道数据
            for (int c = 0; c < nC2; ++c) {
                int nSrcIdx = (n * nC2 + c) * nSpatial;  // 20260320 ZJH B 的源索引
                int nDstIdx = (n * (nC1 + nC2) + nC1 + c) * nSpatial;  // 20260320 ZJH 输出的目标索引
                for (int s = 0; s < nSpatial; ++s)
                    pOut[nDstIdx + s] = pB[nSrcIdx + s];
            }
        }
    }

    // 20260320 ZJH 沿通道维度拼接反向：将 gradOutput [N,C1+C2,H,W] 拆分为 gradA [N,C1,H,W] 和 gradB [N,C2,H,W]
    static void concatChannelsBackward(const float* pGradOut, float* pGradA, float* pGradB,
                                        int nBatch, int nC1, int nC2, int nH, int nW) {
        int nSpatial = nH * nW;  // 20260320 ZJH 空间维度大小
        for (int n = 0; n < nBatch; ++n) {
            // 20260320 ZJH 拆分前 C1 个通道的梯度给 A
            for (int c = 0; c < nC1; ++c) {
                int nSrcIdx = (n * (nC1 + nC2) + c) * nSpatial;
                int nDstIdx = (n * nC1 + c) * nSpatial;
                for (int s = 0; s < nSpatial; ++s)
                    pGradA[nDstIdx + s] = pGradOut[nSrcIdx + s];
            }
            // 20260320 ZJH 拆分后 C2 个通道的梯度给 B
            for (int c = 0; c < nC2; ++c) {
                int nSrcIdx = (n * (nC1 + nC2) + nC1 + c) * nSpatial;
                int nDstIdx = (n * nC2 + c) * nSpatial;
                for (int s = 0; s < nSpatial; ++s)
                    pGradB[nDstIdx + s] = pGradOut[nSrcIdx + s];
            }
        }
    }

    // ===== DiceLoss =====

    // 20260320 ZJH Dice 损失：1 - 2*sum(p*t) / (sum(p) + sum(t) + eps)
    // 用于语义分割任务，p 为预测概率，t 为目标标签
    static float diceLoss(const float* pPred, const float* pTarget, int nCount) {
        float fIntersection = 0.0f;  // 20260320 ZJH 交集：sum(p*t)
        float fPredSum = 0.0f;       // 20260320 ZJH 预测总和：sum(p)
        float fTargetSum = 0.0f;     // 20260320 ZJH 目标总和：sum(t)
        for (int i = 0; i < nCount; ++i) {
            fIntersection += pPred[i] * pTarget[i];  // 20260320 ZJH 累加交集
            fPredSum += pPred[i];                     // 20260320 ZJH 累加预测
            fTargetSum += pTarget[i];                 // 20260320 ZJH 累加目标
        }
        float fEps = 1e-6f;  // 20260320 ZJH 数值稳定性常数
        // 20260320 ZJH Dice 系数 = 2 * intersection / (pred + target)，损失 = 1 - dice
        return 1.0f - 2.0f * fIntersection / (fPredSum + fTargetSum + fEps);
    }

    // 20260320 ZJH BCEWithLogits 前向：二元交叉熵 + sigmoid（数值稳定版本）
    // loss = mean( max(x,0) - x*t + log(1 + exp(-|x|)) )
    static float bceWithLogits(const float* pLogits, const float* pTarget, int nCount) {
        float fLoss = 0.0f;  // 20260320 ZJH 累积损失
        for (int i = 0; i < nCount; ++i) {
            float x = pLogits[i];   // 20260320 ZJH 当前 logit
            float t = pTarget[i];   // 20260320 ZJH 当前目标
            // 20260320 ZJH 数值稳定公式：max(x,0) - x*t + log(1+exp(-|x|))
            float fMaxX = x > 0 ? x : 0;
            fLoss += fMaxX - x * t + std::log(1.0f + std::exp(-std::abs(x)));
        }
        return fLoss / static_cast<float>(nCount);  // 20260320 ZJH 返回均值损失
    }

    // 20260320 ZJH BCEWithLogits 反向：grad = (sigmoid(x) - t) / count
    static void bceWithLogitsBackward(const float* pLogits, const float* pTarget,
                                       float* pGradInput, int nCount) {
        float fScale = 1.0f / static_cast<float>(nCount);  // 20260320 ZJH 均值缩放因子
        for (int i = 0; i < nCount; ++i) {
            float fSigmoid = 1.0f / (1.0f + std::exp(-pLogits[i]));  // 20260320 ZJH sigmoid(x)
            pGradInput[i] = (fSigmoid - pTarget[i]) * fScale;  // 20260320 ZJH 梯度公式
        }
    }

    // ===== GELU =====

    // 20260320 ZJH GELU 前向：高斯误差线性单元
    // GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Transformer/ViT 中标准激活函数，比 ReLU 更平滑
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void gelu(const float* pIn, float* pOut, size_t nCount) {
        const float fSqrt2OverPi = 0.7978845608f;  // 20260320 ZJH sqrt(2/π) 常数
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) {
                float x = pIn[i];
                float fInner = fSqrt2OverPi * (x + 0.044715f * x * x * x);
                pOut[i] = 0.5f * x * (1.0f + std::tanh(fInner));
            }
        } else {
            for (size_t i = 0; i < nCount; ++i) {
                float x = pIn[i];  // 20260320 ZJH 当前输入值
                // 20260320 ZJH tanh 近似公式
                float fInner = fSqrt2OverPi * (x + 0.044715f * x * x * x);
                pOut[i] = 0.5f * x * (1.0f + std::tanh(fInner));  // 20260320 ZJH GELU 输出
            }
        }
    }

    // 20260320 ZJH GELU 反向：dGELU/dx 近似
    // grad = gradOut * (0.5*(1+tanh(inner)) + 0.5*x*sech^2(inner)*sqrt(2/π)*(1+3*0.044715*x^2))
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void geluBackward(const float* pIn, const float* pGradOut, float* pGradIn, size_t nCount) {
        const float fSqrt2OverPi = 0.7978845608f;  // 20260320 ZJH sqrt(2/π) 常数
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) {
                float x = pIn[i];
                float fInner = fSqrt2OverPi * (x + 0.044715f * x * x * x);
                float fTanh = std::tanh(fInner);
                float fSech2 = 1.0f - fTanh * fTanh;
                float fDInner = fSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);
                pGradIn[i] = pGradOut[i] * (0.5f * (1.0f + fTanh) + 0.5f * x * fSech2 * fDInner);
            }
        } else {
            for (size_t i = 0; i < nCount; ++i) {
                float x = pIn[i];  // 20260320 ZJH 前向输入值
                float fInner = fSqrt2OverPi * (x + 0.044715f * x * x * x);  // 20260320 ZJH tanh 内部值
                float fTanh = std::tanh(fInner);  // 20260320 ZJH tanh 结果
                float fSech2 = 1.0f - fTanh * fTanh;  // 20260320 ZJH sech^2 = 1 - tanh^2
                float fDInner = fSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);  // 20260320 ZJH inner 对 x 的导数
                // 20260320 ZJH 链式法则
                pGradIn[i] = pGradOut[i] * (0.5f * (1.0f + fTanh) + 0.5f * x * fSech2 * fDInner);
            }
        }
    }

    // ===== SiLU (Swish) =====

    // 20260320 ZJH SiLU 前向：x * sigmoid(x)，也称为 Swish 激活
    // SiLU(x) = x / (1 + exp(-x))
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void silu(const float* pIn, float* pOut, size_t nCount) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) {
                float fSig = 1.0f / (1.0f + std::exp(-pIn[i]));
                pOut[i] = pIn[i] * fSig;
            }
        } else {
            for (size_t i = 0; i < nCount; ++i) {
                float fSig = 1.0f / (1.0f + std::exp(-pIn[i]));  // 20260320 ZJH sigmoid(x)
                pOut[i] = pIn[i] * fSig;  // 20260320 ZJH x * sigmoid(x)
            }
        }
    }

    // 20260320 ZJH SiLU 反向：grad = gradOut * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
    //                       = gradOut * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // 20260324 ZJH 优化：大数组启用 OpenMP 并行
    static void siluBackward(const float* pIn, const float* pGradOut, float* pGradIn, size_t nCount) {
        if (nCount > 1024) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(nCount); ++i) {
                float fSig = 1.0f / (1.0f + std::exp(-pIn[i]));
                pGradIn[i] = pGradOut[i] * fSig * (1.0f + pIn[i] * (1.0f - fSig));
            }
        } else {
            for (size_t i = 0; i < nCount; ++i) {
                float fSig = 1.0f / (1.0f + std::exp(-pIn[i]));  // 20260320 ZJH sigmoid(x)
                // 20260320 ZJH SiLU 导数公式
                pGradIn[i] = pGradOut[i] * fSig * (1.0f + pIn[i] * (1.0f - fSig));
            }
        }
    }

    // ===== LayerNorm =====

    // 20260320 ZJH LayerNorm 前向：沿最后一维归一化
    // pInput: [batch, dim]  pGamma/pBeta: [dim]  pOutput: [batch, dim]
    // pSavedMean/pSavedInvStd: [batch]，保存每个样本的均值和逆标准差用于反向
    static void layerNorm(const float* pInput, float* pOutput,
                          const float* pGamma, const float* pBeta,
                          float* pSavedMean, float* pSavedInvStd,
                          int nBatch, int nDim, float fEps) {
        for (int b = 0; b < nBatch; ++b) {  // 20260320 ZJH 遍历每个样本
            const float* pRow = pInput + b * nDim;  // 20260320 ZJH 当前样本输入
            float* pOutRow = pOutput + b * nDim;     // 20260320 ZJH 当前样本输出
            // 20260320 ZJH 计算均值
            float fMean = 0.0f;
            for (int d = 0; d < nDim; ++d) fMean += pRow[d];
            fMean /= static_cast<float>(nDim);
            // 20260320 ZJH 计算方差
            float fVar = 0.0f;
            for (int d = 0; d < nDim; ++d) {
                float fDiff = pRow[d] - fMean;
                fVar += fDiff * fDiff;
            }
            fVar /= static_cast<float>(nDim);
            float fInvStd = 1.0f / std::sqrt(fVar + fEps);  // 20260320 ZJH 逆标准差
            // 20260320 ZJH 保存用于反向传播
            pSavedMean[b] = fMean;
            pSavedInvStd[b] = fInvStd;
            // 20260320 ZJH 归一化 + 缩放 + 偏移
            for (int d = 0; d < nDim; ++d) {
                float fNorm = (pRow[d] - fMean) * fInvStd;
                pOutRow[d] = pGamma[d] * fNorm + pBeta[d];
            }
        }
    }

    // 20260320 ZJH LayerNorm 反向传播
    // 计算 gradInput [batch,dim], gradGamma [dim], gradBeta [dim]
    static void layerNormBackward(const float* pGradOutput, const float* pInput,
                                   const float* pSavedMean, const float* pSavedInvStd,
                                   const float* pGamma,
                                   float* pGradInput, float* pGradGamma, float* pGradBeta,
                                   int nBatch, int nDim) {
        // 20260320 ZJH 初始化 gradGamma 和 gradBeta
        for (int d = 0; d < nDim; ++d) { pGradGamma[d] = 0.0f; pGradBeta[d] = 0.0f; }

        for (int b = 0; b < nBatch; ++b) {  // 20260320 ZJH 遍历每个样本
            float fMean = pSavedMean[b];
            float fInvStd = pSavedInvStd[b];
            const float* pGradRow = pGradOutput + b * nDim;
            const float* pInRow = pInput + b * nDim;
            float* pGradInRow = pGradInput + b * nDim;

            // 20260320 ZJH 累加 gradGamma 和 gradBeta
            float fDLdGamma = 0.0f;
            float fDLdBeta = 0.0f;
            for (int d = 0; d < nDim; ++d) {
                float fXhat = (pInRow[d] - fMean) * fInvStd;
                pGradGamma[d] += pGradRow[d] * fXhat;
                pGradBeta[d] += pGradRow[d];
                fDLdGamma += pGradRow[d] * pGamma[d] * fXhat;
                fDLdBeta += pGradRow[d] * pGamma[d];
            }

            // 20260320 ZJH 计算 gradInput
            float fInvDim = 1.0f / static_cast<float>(nDim);
            for (int d = 0; d < nDim; ++d) {
                float fXhat = (pInRow[d] - fMean) * fInvStd;
                pGradInRow[d] = pGamma[d] * fInvStd * fInvDim *
                    (static_cast<float>(nDim) * pGradRow[d] - fDLdBeta - fXhat * fDLdGamma);
            }
        }
    }

    // ===== AdaptiveAvgPool2d =====

    // 20260320 ZJH AdaptiveAvgPool2d 前向：自适应平均池化到目标尺寸
    // pInput: [N, C, H, W]  pOutput: [N, C, outH, outW]
    // 每个输出位置覆盖的输入区域按 floor 划分
    static void adaptiveAvgPool2d(const float* pInput, float* pOutput,
                                   int nBatch, int nChannels, int nH, int nW,
                                   int nOutH, int nOutW) {
        for (int n = 0; n < nBatch; ++n)
        for (int c = 0; c < nChannels; ++c)
        for (int oh = 0; oh < nOutH; ++oh)
        for (int ow = 0; ow < nOutW; ++ow) {
            // 20260320 ZJH 计算输入区域的起止坐标（PyTorch 风格 floor 划分）
            int nHStart = oh * nH / nOutH;
            int nHEnd = (oh + 1) * nH / nOutH;
            int nWStart = ow * nW / nOutW;
            int nWEnd = (ow + 1) * nW / nOutW;
            float fSum = 0.0f;
            int nCount = 0;
            for (int ih = nHStart; ih < nHEnd; ++ih)
            for (int iw = nWStart; iw < nWEnd; ++iw) {
                fSum += pInput[((n * nChannels + c) * nH + ih) * nW + iw];
                nCount++;
            }
            pOutput[((n * nChannels + c) * nOutH + oh) * nOutW + ow] = fSum / static_cast<float>(nCount);
        }
    }

    // 20260320 ZJH AdaptiveAvgPool2d 反向：将梯度均匀散布回输入区域
    static void adaptiveAvgPool2dBackward(const float* pGradOutput, float* pGradInput,
                                           int nBatch, int nChannels, int nH, int nW,
                                           int nOutH, int nOutW) {
        int nInputSize = nBatch * nChannels * nH * nW;
        for (int i = 0; i < nInputSize; ++i) pGradInput[i] = 0.0f;  // 20260320 ZJH 初始化

        for (int n = 0; n < nBatch; ++n)
        for (int c = 0; c < nChannels; ++c)
        for (int oh = 0; oh < nOutH; ++oh)
        for (int ow = 0; ow < nOutW; ++ow) {
            int nHStart = oh * nH / nOutH;
            int nHEnd = (oh + 1) * nH / nOutH;
            int nWStart = ow * nW / nOutW;
            int nWEnd = (ow + 1) * nW / nOutW;
            int nCount = (nHEnd - nHStart) * (nWEnd - nWStart);
            float fGrad = pGradOutput[((n * nChannels + c) * nOutH + oh) * nOutW + ow]
                          / static_cast<float>(nCount);
            for (int ih = nHStart; ih < nHEnd; ++ih)
            for (int iw = nWStart; iw < nWEnd; ++iw) {
                pGradInput[((n * nChannels + c) * nH + ih) * nW + iw] += fGrad;
            }
        }
    }

    // ===== Batched MatMul =====

    // 20260320 ZJH 批量矩阵乘法：A[batch, M, K] * B[batch, K, N] -> C[batch, M, N]
    // 用于 Transformer 注意力的 Q*K^T 和 attn*V 计算
    // 20260324 ZJH 优化：OpenMP 在 batch 维度上并行（每个 batch 的 matmul 独立且已含 SIMD）
    static void batchedMatmul(const float* pA, const float* pB, float* pC,
                               int nBatch, int nM, int nK, int nN) {
        int nAStride = nM * nK;  // 20260320 ZJH A 每个 batch 的步长
        int nBStride = nK * nN;  // 20260320 ZJH B 每个 batch 的步长
        int nCStride = nM * nN;  // 20260320 ZJH C 每个 batch 的步长
        // 20260324 ZJH 当 batch 较多时在 batch 级并行
        // 注意：matmul 内部已有 OpenMP 并行，这里仅在 batch > 1 且矩阵较小时有意义
        // 对于大矩阵，matmul 内部的 OpenMP 已充分利用核心
        #pragma omp parallel for schedule(dynamic, 1) if(nBatch > 1 && nM * nK < 4096)
        for (int b = 0; b < nBatch; ++b) {
            matmul(pA + b * nAStride, pB + b * nBStride, pC + b * nCStride, nM, nK, nN);
        }
    }

    // 20260320 ZJH 矩阵转置：A[rows, cols] -> B[cols, rows]
    static void transpose2d(const float* pA, float* pB, int nRows, int nCols) {
        for (int r = 0; r < nRows; ++r)
        for (int c = 0; c < nCols; ++c)
            pB[c * nRows + r] = pA[r * nCols + c];
    }

    // ===== Phase 5B: tanh / clip 内核 =====

    // 20260321 ZJH tanh 前向：out[i] = tanh(in[i])
    // LSTM 门控和 cell candidate 的核心激活函数
    static void tanhForward(const float* pIn, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = std::tanh(pIn[i]);  // 20260321 ZJH 逐元素 tanh
        }
    }

    // 20260321 ZJH tanh 反向：grad_in = grad_out * (1 - out^2)
    // 需要保存前向输出 out，导数直接由输出值计算
    static void tanhBackward(const float* pOutput, const float* pGradOut,
                              float* pGradIn, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            float fO = pOutput[i];  // 20260321 ZJH 前向输出
            pGradIn[i] = pGradOut[i] * (1.0f - fO * fO);  // 20260321 ZJH tanh 导数
        }
    }

    // 20260321 ZJH clip 前向：out[i] = clamp(in[i], fMin, fMax)
    // CTC loss 中对概率值做 log-safe clip 使用
    static void clipForward(const float* pIn, float* pOut, size_t nCount,
                             float fMin, float fMax) {
        for (size_t i = 0; i < nCount; ++i) {
            float fVal = pIn[i];
            if (fVal < fMin) fVal = fMin;
            if (fVal > fMax) fVal = fMax;
            pOut[i] = fVal;
        }
    }

    // ===== Softmax Last Dim 反向 =====

    // 20260328 ZJH softmaxLastDimBackward — softmax 沿最后一维的反向传播（CPU 版）
    // 标准 softmax Jacobian-vector product 公式:
    //   gradIn[row][j] = softmax[row][j] * (gradOut[row][j] - dot(gradOut[row], softmax[row]))
    // pGradOut: [nOuter, nLastDim] 上游传来的梯度
    // pSoftmax: [nOuter, nLastDim] 前向 softmax 输出
    // pGradIn:  [nOuter, nLastDim] 计算得到的输入梯度（输出）
    // nOuter:   外层维度积
    // nLastDim: 最后一维大小（softmax 归一化的维度）
    static void softmaxLastDimBackward(const float* pGradOut, const float* pSoftmax,
                                        float* pGradIn, int nOuter, int nLastDim) {
        for (int b = 0; b < nOuter; ++b) {
            const float* pGradRow = pGradOut + b * nLastDim;  // 20260328 ZJH 当前行梯度
            const float* pSoftRow = pSoftmax + b * nLastDim;  // 20260328 ZJH 当前行 softmax
            float* pGradInRow = pGradIn + b * nLastDim;  // 20260328 ZJH 当前行输出
            // 20260328 ZJH 计算 dot = sum(gradOut[j] * softmax[j])
            float fDot = 0.0f;  // 20260328 ZJH 初始化 dot product 为零
            for (int j = 0; j < nLastDim; ++j) {
                fDot += pGradRow[j] * pSoftRow[j];  // 20260328 ZJH 逐元素累加
            }
            // 20260328 ZJH 计算 gradIn[j] = softmax[j] * (gradOut[j] - dot)
            for (int j = 0; j < nLastDim; ++j) {
                pGradInRow[j] = pSoftRow[j] * (pGradRow[j] - fDot);  // 20260328 ZJH Jacobian-vector product
            }
        }
    }

    // ===== 基于 strides 的非连续数据提取 =====

    // 20260319 ZJH 基于 strides 的非连续数据提取到连续缓冲区
    // 用于 slice/transpose 等产生非连续视图后的数据收集
    static void stridedCopy(const float* pSrc, float* pDst,
                            const std::vector<int>& vecShape,
                            const std::vector<int>& vecStrides,
                            int nOffset) {
        int nNDim = static_cast<int>(vecShape.size());  // 20260319 ZJH 张量维度数
        if (nNDim == 0) return;  // 20260319 ZJH 标量张量无需拷贝

        // 20260319 ZJH 计算总元素个数（各维度大小之积）
        int nTotal = 1;
        for (int d = 0; d < nNDim; ++d) nTotal *= vecShape[d];

        std::vector<int> vecIdx(nNDim, 0);  // 20260319 ZJH 多维索引计数器，初始全零
        for (int i = 0; i < nTotal; ++i) {
            // 20260319 ZJH 根据当前多维索引和 strides 计算源内存偏移
            int nSrcIdx = nOffset;
            for (int d = 0; d < nNDim; ++d) nSrcIdx += vecIdx[d] * vecStrides[d];
            pDst[i] = pSrc[nSrcIdx];  // 20260319 ZJH 将非连续源元素写入连续目标

            // 20260319 ZJH 更新多维索引：从最低维进位
            for (int d = nNDim - 1; d >= 0; --d) {
                vecIdx[d]++;
                if (vecIdx[d] < vecShape[d]) break;  // 20260319 ZJH 未进位则退出
                vecIdx[d] = 0;  // 20260319 ZJH 进位：当前维归零，继续向高维进位
            }
        }
    }
};

}  // namespace om
