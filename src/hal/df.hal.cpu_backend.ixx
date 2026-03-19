// 20260319 ZJH CPUBackend — 朴素 C++ 计算内核
// Phase 1B: Float32 基础运算（先正确后优化）
module;

#include <cstddef>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

export module df.hal.cpu_backend;

export namespace df {

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

    // ===== 元素运算 =====
    // 20260319 ZJH 逐元素加法：pOut[i] = pA[i] + pB[i]
    static void add(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] + pB[i];
    }

    // 20260319 ZJH 逐元素减法：pOut[i] = pA[i] - pB[i]
    static void sub(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] - pB[i];
    }

    // 20260319 ZJH 逐元素乘法：pOut[i] = pA[i] * pB[i]
    static void mul(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] * pB[i];
    }

    // 20260319 ZJH 逐元素除法：pOut[i] = pA[i] / pB[i]
    static void div(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] / pB[i];
    }

    // 20260319 ZJH 加标量：pOut[i] = pA[i] + fScalar
    static void addScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] + fScalar;
    }

    // 20260319 ZJH 乘标量：pOut[i] = pA[i] * fScalar
    static void mulScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pOut[i] = pA[i] * fScalar;
    }

    // ===== 矩阵乘法 =====
    // 20260319 ZJH A[M,K] * B[K,N] -> C[M,N], row-major 行主序
    // 使用 i-k-j 循环顺序提升 B 的缓存局部性
    static void matmul(const float* pA, const float* pB, float* pC, int nM, int nK, int nN) {
        // 20260319 ZJH 初始化输出矩阵 C 为全零
        for (int i = 0; i < nM * nN; ++i) pC[i] = 0.0f;
        // 20260319 ZJH 三重循环执行矩阵乘法，i-k-j 顺序提升 pB 访问局部性
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

    // ===== 数据拷贝 =====
    // 20260319 ZJH 连续内存拷贝：pSrc -> pDst，共 nCount 个 float
    static void copy(const float* pSrc, float* pDst, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) pDst[i] = pSrc[i];
    }

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

}  // namespace df
