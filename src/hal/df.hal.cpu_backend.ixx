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

    // ===== 激活函数 =====

    // 20260319 ZJH ReLU 前向：out[i] = max(0, in[i])
    // 将负值置零，正值直通，实现修正线性单元激活
    static void relu(const float* pIn, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i)
            pOut[i] = pIn[i] > 0.0f ? pIn[i] : 0.0f;  // 20260319 ZJH 正值直通，负值置零
    }

    // 20260319 ZJH ReLU 反向：grad_in[i] = grad_out[i] * (in[i] > 0 ? 1 : 0)
    // 前向时输入大于零的位置梯度直通，否则梯度为零
    static void reluBackward(const float* pIn, const float* pGradOut, float* pGradIn, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i)
            pGradIn[i] = pIn[i] > 0.0f ? pGradOut[i] : 0.0f;  // 20260319 ZJH 正值梯度直通，负值梯度截断
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
    static void crossEntropySoftmaxBackward(const float* pSoftmax, const float* pTarget,
                                             float* pGradInput, int nBatch, int nClasses) {
        float fScale = 1.0f / static_cast<float>(nBatch);  // 20260319 ZJH 批次平均缩放因子
        for (int i = 0; i < nBatch * nClasses; ++i) {
            // 20260319 ZJH 联合梯度公式：(softmax - one_hot) / batch_size
            pGradInput[i] = (pSoftmax[i] - pTarget[i]) * fScale;
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

    // ===== Conv2d =====

    // 20260319 ZJH Conv2d 前向：NCHW 格式，朴素直接卷积（不使用 im2col）
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
        int nOutSize = nBatch * nCout * nHout * nWout;  // 20260319 ZJH 输出总元素数
        // 20260319 ZJH 初始化输出为零
        for (int i = 0; i < nOutSize; ++i) pOutput[i] = 0.0f;

        // 20260319 ZJH 六重循环实现直接卷积
        for (int n = 0; n < nBatch; ++n) {  // 20260319 ZJH 遍历批次
            for (int co = 0; co < nCout; ++co) {  // 20260319 ZJH 遍历输出通道
                for (int oh = 0; oh < nHout; ++oh) {  // 20260319 ZJH 遍历输出行
                    for (int ow = 0; ow < nWout; ++ow) {  // 20260319 ZJH 遍历输出列
                        float fSum = 0.0f;  // 20260319 ZJH 当前输出位置的累加值
                        // 20260319 ZJH 遍历卷积核
                        for (int ci = 0; ci < nCin; ++ci) {  // 20260319 ZJH 遍历输入通道
                            for (int kh = 0; kh < nKH; ++kh) {  // 20260319 ZJH 遍历核高度
                                for (int kw = 0; kw < nKW; ++kw) {  // 20260319 ZJH 遍历核宽度
                                    int nIh = oh * nStride - nPad + kh;  // 20260319 ZJH 输入行坐标
                                    int nIw = ow * nStride - nPad + kw;  // 20260319 ZJH 输入列坐标
                                    // 20260319 ZJH 边界检查：padding 区域跳过（等效零填充）
                                    if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                        // 20260319 ZJH 输入索引: [n, ci, ih, iw]
                                        int nInIdx = ((n * nCin + ci) * nH + nIh) * nW + nIw;
                                        // 20260319 ZJH 权重索引: [co, ci, kh, kw]
                                        int nWIdx = ((co * nCin + ci) * nKH + kh) * nKW + kw;
                                        fSum += pInput[nInIdx] * pWeight[nWIdx];  // 20260319 ZJH 累加乘积
                                    }
                                }
                            }
                        }
                        // 20260319 ZJH 加偏置（如果有）
                        if (pBias) fSum += pBias[co];
                        // 20260319 ZJH 写入输出: [n, co, oh, ow]
                        int nOutIdx = ((n * nCout + co) * nHout + oh) * nWout + ow;
                        pOutput[nOutIdx] = fSum;
                    }
                }
            }
        }
    }

    // 20260319 ZJH Conv2d 反向（对输入求梯度）：gradInput = 转置卷积
    // pGradOutput: [N, Cout, Hout, Wout]  pWeight: [Cout, Cin, KH, KW]
    // pGradInput: [N, Cin, H, W]（需预先置零）
    static void conv2dBackwardInput(const float* pGradOutput, const float* pWeight,
                                     float* pGradInput,
                                     int nBatch, int nCin, int nH, int nW,
                                     int nCout, int nKH, int nKW,
                                     int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        // 20260319 ZJH 初始化 gradInput 为零
        int nInputSize = nBatch * nCin * nH * nW;
        for (int i = 0; i < nInputSize; ++i) pGradInput[i] = 0.0f;

        // 20260319 ZJH 遍历每个输出位置，将梯度散布回输入位置
        for (int n = 0; n < nBatch; ++n) {
            for (int co = 0; co < nCout; ++co) {
                for (int oh = 0; oh < nHout; ++oh) {
                    for (int ow = 0; ow < nWout; ++ow) {
                        // 20260319 ZJH gradOutput 的索引: [n, co, oh, ow]
                        int nGradOutIdx = ((n * nCout + co) * nHout + oh) * nWout + ow;
                        float fGrad = pGradOutput[nGradOutIdx];  // 20260319 ZJH 当前输出梯度
                        for (int ci = 0; ci < nCin; ++ci) {
                            for (int kh = 0; kh < nKH; ++kh) {
                                for (int kw = 0; kw < nKW; ++kw) {
                                    int nIh = oh * nStride - nPad + kh;  // 20260319 ZJH 输入行坐标
                                    int nIw = ow * nStride - nPad + kw;  // 20260319 ZJH 输入列坐标
                                    if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                        int nInIdx = ((n * nCin + ci) * nH + nIh) * nW + nIw;
                                        int nWIdx = ((co * nCin + ci) * nKH + kh) * nKW + kw;
                                        // 20260319 ZJH 梯度散布：gradInput += gradOutput * weight
                                        pGradInput[nInIdx] += fGrad * pWeight[nWIdx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 20260319 ZJH Conv2d 反向（对权重和偏置求梯度）
    // pInput: [N, Cin, H, W]  pGradOutput: [N, Cout, Hout, Wout]
    // pGradWeight: [Cout, Cin, KH, KW]  pGradBias: [Cout] 或 nullptr
    static void conv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                      float* pGradWeight, float* pGradBias,
                                      int nBatch, int nCin, int nH, int nW,
                                      int nCout, int nKH, int nKW,
                                      int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度
        // 20260319 ZJH 初始化 gradWeight 为零
        int nWeightSize = nCout * nCin * nKH * nKW;
        for (int i = 0; i < nWeightSize; ++i) pGradWeight[i] = 0.0f;
        // 20260319 ZJH 初始化 gradBias 为零
        if (pGradBias) {
            for (int i = 0; i < nCout; ++i) pGradBias[i] = 0.0f;
        }

        // 20260319 ZJH 遍历每个输出位置计算权重梯度
        for (int n = 0; n < nBatch; ++n) {
            for (int co = 0; co < nCout; ++co) {
                for (int oh = 0; oh < nHout; ++oh) {
                    for (int ow = 0; ow < nWout; ++ow) {
                        int nGradOutIdx = ((n * nCout + co) * nHout + oh) * nWout + ow;
                        float fGrad = pGradOutput[nGradOutIdx];  // 20260319 ZJH 当前输出梯度
                        // 20260319 ZJH 偏置梯度：所有位置的梯度之和
                        if (pGradBias) pGradBias[co] += fGrad;
                        for (int ci = 0; ci < nCin; ++ci) {
                            for (int kh = 0; kh < nKH; ++kh) {
                                for (int kw = 0; kw < nKW; ++kw) {
                                    int nIh = oh * nStride - nPad + kh;
                                    int nIw = ow * nStride - nPad + kw;
                                    if (nIh >= 0 && nIh < nH && nIw >= 0 && nIw < nW) {
                                        int nInIdx = ((n * nCin + ci) * nH + nIh) * nW + nIw;
                                        int nWIdx = ((co * nCin + ci) * nKH + kh) * nKW + kw;
                                        // 20260319 ZJH gradWeight += input * gradOutput
                                        pGradWeight[nWIdx] += pInput[nInIdx] * fGrad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ===== BatchNorm2d =====

    // 20260319 ZJH BatchNorm2d 前向：NCHW 格式
    // 训练时：计算当前 batch 的均值和方差，更新 running 统计量，归一化后缩放偏移
    // 评估时：使用 running 统计量进行归一化
    // pSavedMean/pSavedInvStd: 保存训练时的均值和逆标准差，用于反向传播
    static void batchNorm2d(const float* pInput, float* pOutput,
                             const float* pGamma, const float* pBeta,
                             float* pRunMean, float* pRunVar,
                             float* pSavedMean, float* pSavedInvStd,
                             int nBatch, int nChannels, int nH, int nW,
                             float fEps, float fMomentum, bool bTraining) {
        int nSpatial = nH * nW;  // 20260319 ZJH 空间维度大小
        int nCount = nBatch * nSpatial;  // 20260319 ZJH 每通道的元素总数

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
    static void batchNorm2dBackward(const float* pGradOutput, const float* pInput,
                                     const float* pSavedMean, const float* pSavedInvStd,
                                     const float* pGamma,
                                     float* pGradInput, float* pGradGamma, float* pGradBeta,
                                     int nBatch, int nChannels, int nH, int nW) {
        int nSpatial = nH * nW;  // 20260319 ZJH 空间维度大小
        int nCount = nBatch * nSpatial;  // 20260319 ZJH 每通道的元素总数
        float fInvCount = 1.0f / static_cast<float>(nCount);  // 20260319 ZJH 元素数倒数

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
    static void maxPool2d(const float* pInput, float* pOutput, int* pIndices,
                           int nBatch, int nChannels, int nH, int nW,
                           int nKH, int nKW, int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度

        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannels; ++c) {
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
    static void avgPool2d(const float* pInput, float* pOutput,
                           int nBatch, int nChannels, int nH, int nW,
                           int nKH, int nKW, int nStride, int nPad) {
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度

        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannels; ++c) {
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
    static void sigmoid(const float* pIn, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i)
            pOut[i] = 1.0f / (1.0f + std::exp(-pIn[i]));  // 20260320 ZJH S 型激活函数
    }

    // 20260320 ZJH Sigmoid 反向：grad_in[i] = grad_out[i] * out[i] * (1 - out[i])
    // pOutput 是前向的输出值（非输入值）
    static void sigmoidBackward(const float* pOutput, const float* pGradOut, float* pGradIn, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i)
            pGradIn[i] = pGradOut[i] * pOutput[i] * (1.0f - pOutput[i]);  // 20260320 ZJH sigmoid 导数
    }

    // ===== LeakyReLU =====

    // 20260320 ZJH LeakyReLU 前向：正值直通，负值乘以斜率 fSlope
    static void leakyRelu(const float* pIn, float* pOut, size_t nCount, float fSlope = 0.01f) {
        for (size_t i = 0; i < nCount; ++i)
            pOut[i] = pIn[i] > 0 ? pIn[i] : fSlope * pIn[i];  // 20260320 ZJH 负值保留小梯度
    }

    // 20260320 ZJH LeakyReLU 反向：正值梯度直通，负值梯度乘以斜率
    static void leakyReluBackward(const float* pIn, const float* pGradOut, float* pGradIn,
                                   size_t nCount, float fSlope = 0.01f) {
        for (size_t i = 0; i < nCount; ++i)
            pGradIn[i] = pIn[i] > 0 ? pGradOut[i] : fSlope * pGradOut[i];  // 20260320 ZJH 负值区域斜率为 fSlope
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

}  // namespace df
