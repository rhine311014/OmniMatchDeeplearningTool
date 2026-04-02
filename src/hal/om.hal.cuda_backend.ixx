// 20260325 ZJH CUDABackend — GPU 计算后端（镜像 CPUBackend 接口）
// Phase 2: GPU-Resident Tensor 系统的核心后端
// 所有传入的指针均为 GPU 设备内存指针，无需 H2D/D2H 传输
// 每个方法直接调用对应的 omCudaXxx C 绑定函数
module;

#include <cstddef>  // 20260325 ZJH size_t 定义

// 20260325 ZJH CUDA 函数前向声明（C 链接），在 module global fragment 中声明
// 避免在 module purview 中 #include CUDA 头文件
// 仅在 OM_HAS_CUDA 宏定义时启用（CMake 传入）
#ifdef OM_HAS_CUDA
extern "C" {
    // ===== 元素运算 =====
    // 20260325 ZJH 逐元素加法
    int omCudaAdd(const float* pA, const float* pB, float* pC, int nCount);
    // 20260325 ZJH 逐元素减法
    int omCudaSub(const float* pA, const float* pB, float* pC, int nCount);
    // 20260325 ZJH 逐元素乘法
    int omCudaMul(const float* pA, const float* pB, float* pC, int nCount);
    // 20260325 ZJH 加标量
    int omCudaAddScalar(const float* pA, float fScalar, float* pC, int nCount);
    // 20260325 ZJH 乘标量
    int omCudaMulScalar(const float* pA, float fScalar, float* pC, int nCount);

    // ===== 矩阵运算 =====
    // 20260325 ZJH 矩阵乘法（32x32 tiled）
    int omCudaMatmul(const float* pA, const float* pB, float* pC, int nM, int nK, int nN);
    // 20260325 ZJH 批量矩阵乘法
    int omCudaBatchedMatmul(const float* pA, const float* pB, float* pC,
                            int nBatch, int nM, int nK, int nN);
    // 20260325 ZJH 矩阵转置
    int omCudaTranspose(const float* pIn, float* pOut, int nRows, int nCols);

    // ===== 激活函数 =====
    // 20260325 ZJH ReLU 前向
    int omCudaReLU(const float* pIn, float* pOut, int nCount);
    // 20260325 ZJH ReLU 反向
    int omCudaReLUBackward(const float* pIn, const float* pGradOut, float* pGradIn, int nCount);
    // 20260325 ZJH Sigmoid 前向
    int omCudaSigmoid(const float* pIn, float* pOut, int nCount);
    // 20260325 ZJH Sigmoid 反向
    int omCudaSigmoidBackward(const float* pGrad, const float* pOutput, float* pOut, int nCount);
    // 20260325 ZJH GELU 前向
    int omCudaGELU(const float* pIn, float* pOut, int nCount);
    // 20260325 ZJH GELU 反向
    int omCudaGeluBackward(const float* pGrad, const float* pInput, float* pOut, int nCount);
    // 20260325 ZJH SiLU 前向
    int omCudaSiLU(const float* pIn, float* pOut, int nCount);
    // 20260325 ZJH SiLU 反向
    int omCudaSiluBackward(const float* pGrad, const float* pInput, float* pOut, int nCount);
    // 20260325 ZJH LeakyReLU 前向
    int omCudaLeakyRelu(const float* pIn, float* pOut, int nCount, float fSlope);
    // 20260325 ZJH LeakyReLU 反向
    int omCudaLeakyReluBackward(const float* pGrad, const float* pInput, float* pOut,
                                 int nCount, float fSlope);
    // 20260325 ZJH Tanh 前向
    int omCudaTanh(const float* pIn, float* pOut, int nCount);
    // 20260325 ZJH Tanh 反向
    int omCudaTanhBackward(const float* pGrad, const float* pOutput, float* pOut, int nCount);

    // ===== 卷积 =====
    // 20260325 ZJH Conv2d 前向（自动选择 im2col+GEMM 或朴素实现）
    int omCudaConv2d(const float* pInput, const float* pWeight, const float* pBias,
                     float* pOutput,
                     int nBatch, int nCin, int nH, int nW,
                     int nCout, int nKH, int nKW, int nStride, int nPad);
    // 20260331 ZJH 膨胀卷积 GPU 前向（im2col+GEMM，支持 dilation 和 groups）
    int omCudaDilatedConv2d(const float* pInput, const float* pWeight, const float* pBias,
                             float* pOutput,
                             int nBatch, int nCin, int nH, int nW,
                             int nCout, int nKH, int nKW,
                             int nStride, int nPad, int nDilation, int nGroups);
    // 20260331 ZJH 膨胀卷积反向权重梯度
    int omCudaDilatedConv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                           float* pGradWeight, float* pGradBias,
                                           int nBatch, int nCin, int nH, int nW,
                                           int nCout, int nKH, int nKW,
                                           int nStride, int nPad, int nDilation);
    // 20260325 ZJH Conv2d 反向对输入求梯度（col2im + GEMM）
    int omCudaConv2dBackwardInput(const float* pGradOutput, const float* pWeight,
                                   float* pGradInput,
                                   int nBatch, int nCin, int nH, int nW,
                                   int nCout, int nKH, int nKW, int nStride, int nPad);
    // 20260325 ZJH Conv2d 反向对权重求梯度（im2col + GEMM）
    int omCudaConv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                    float* pGradWeight, float* pGradBias,
                                    int nBatch, int nCin, int nH, int nW,
                                    int nCout, int nKH, int nKW, int nStride, int nPad);
    // 20260330 ZJH BatchNorm2d 反向（GPU 两遍 kernel），新增 fEps 参数
    int omCudaBatchNorm2dBackward(const float* pGradOutput, const float* pInput,
                                   const float* pMean, const float* pInvStd,
                                   const float* pGamma,
                                   float* pGradInput, float* pGradGamma, float* pGradBeta,
                                   int nBatch, int nChannels, int nH, int nW,
                                   float fEps = 1e-5f);

    // 20260402 ZJH GroupNorm2d 前向声明
    int omCudaGroupNorm2d(const float* pInput, float* pOutput,
                          const float* pGamma, const float* pBeta,
                          float* pSavedMean, float* pSavedInvStd,
                          int nBatch, int nChannels, int nH, int nW,
                          int nGroups, float fEps);

    // 20260402 ZJH GroupNorm2d 反向声明
    int omCudaGroupNorm2dBackward(const float* pGradOutput, const float* pInput,
                                   const float* pMean, const float* pInvStd,
                                   const float* pGamma,
                                   float* pGradInput, float* pGradGamma, float* pGradBeta,
                                   int nBatch, int nChannels, int nH, int nW,
                                   int nGroups, float fEps);

    // ===== 池化 =====
    // 20260325 ZJH MaxPool2d 前向
    int omCudaMaxPool2d(const float* pIn, float* pOut, int* pIndices,
                         int nN, int nC, int nH, int nW,
                         int nKH, int nKW, int nSH, int nSW, int nPH, int nPW);
    // 20260325 ZJH MaxPool2d 反向
    int omCudaMaxPool2dBackward(const float* pGradOut, const int* pIndices,
                                 float* pGradIn,
                                 int nN, int nC, int nH, int nW, int nHout, int nWout);
    // 20260325 ZJH AvgPool2d 前向
    int omCudaAvgPool2d(const float* pIn, float* pOut,
                         int nN, int nC, int nH, int nW,
                         int nKH, int nKW, int nSH, int nSW, int nPH, int nPW);
    // 20260325 ZJH AvgPool2d 反向
    int omCudaAvgPool2dBackward(const float* pGradOut, float* pGradIn,
                                 int nN, int nC, int nH, int nW,
                                 int nKH, int nKW, int nSH, int nSW, int nPH, int nPW);
    // 20260325 ZJH AdaptiveAvgPool2d 前向
    int omCudaAdaptiveAvgPool2d(const float* pIn, float* pOut,
                                 int nN, int nC, int nH, int nW, int nOutH, int nOutW);

    // ===== 归一化 =====
    // 20260325 ZJH BatchNorm2d 前向
    int omCudaBatchNorm2d(const float* pInput, float* pOutput,
                          const float* pGamma, const float* pBeta,
                          float* pRunMean, float* pRunVar,
                          float* pSavedMean, float* pSavedInvStd,
                          int nBatch, int nChannels, int nH, int nW,
                          float fEps, float fMomentum, int bTraining);
    // 20260325 ZJH LayerNorm 前向
    int omCudaLayerNorm(const float* pInput, float* pOutput,
                        const float* pGamma, const float* pBeta,
                        float* pSavedMean, float* pSavedInvStd,
                        int nBatch, int nDim, float fEps);

    // ===== Softmax / 损失 =====
    // 20260325 ZJH Softmax 前向
    int omCudaSoftmax(const float* pIn, float* pOut, int nBatch, int nClasses);
    // 20260325 ZJH Softmax + CrossEntropy 联合反向
    int omCudaSoftmaxCrossEntropyBackward(const float* pSoftmax, const float* pTarget,
                                           float* pGradLogits, int nBatch, int nClasses);

    // ===== 优化器 =====
    // 20260325 ZJH Adam step（GPU 上直接更新参数）
    int omCudaAdamStep(float* pParam, const float* pGrad, float* pM, float* pV,
                        int nCount, float fLr, float fBeta1, float fBeta2,
                        float fEps, int nStep);
    // 20260325 ZJH SGD step（GPU 上直接更新参数）
    int omCudaSgdStep(float* pParam, const float* pGrad, float* pVelocity,
                       int nCount, float fLr, float fMomentum);

    // ===== 辅助 =====
    // 20260325 ZJH GPU 内存填零
    int omCudaFillZeros(float* pData, int nCount);
    // 20260325 ZJH GPU 内存填 1
    int omCudaFillOnes(float* pData, int nCount);
    // 20260325 ZJH GPU 内存填充指定值
    int omCudaFillValue(float* pData, int nCount, float fValue);
    // 20260325 ZJH GPU D2D 拷贝
    int omCudaCopy(const float* pSrc, float* pDst, int nCount);
    // 20260325 ZJH 广播偏置加法
    int omCudaAddBias(const float* pData, const float* pBias, float* pOut,
                       int nN, int nC, int nHW);
    // 20260325 ZJH Dropout（旧版：需预生成 mask）
    int omCudaDropout(const float* pIn, float* pOut, const float* pMask,
                       int nCount, float fProb, int bTraining);
    // 20260328 ZJH Fused Dropout Forward（新版：GPU 端 SplitMix64 mask 生成 + 应用，零传输）
    int omCudaDropoutForward(const float* pIn, float* pOut, float* pMask,
                              int nCount, float fKeepProb, unsigned long long nSeed);
    // 20260325 ZJH Argmax
    int omCudaArgmax(const float* pData, int* pOut, int nBatch, int nClasses);
    // 20260325 ZJH 归约
    int omCudaSum(const float* pData, float* pResult, int nCount);
    int omCudaMean(const float* pData, float* pResult, int nCount);
    // 20260325 ZJH GPU 内存清零（字节级）
    int omCudaMemset(void* pDev, int nValue, size_t nBytes);
    // 20260326 ZJH AdaptiveAvgPool2d 反向传播
    int omCudaAdaptiveAvgPool2dBackward(const float* pGradOut, float* pGradIn,
        int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW);
    // 20260326 ZJH AddBias 反向传播（通道维 reduction）
    int omCudaAddBiasBackward(const float* pGradOut, float* pGradBias,
        int nN, int nC, int nHW);
    // 20260326 ZJH UpsampleBilinear 反向传播
    int omCudaUpsampleBilinearBackward(const float* pGradOut, float* pGradIn,
        int nBatch, int nChannels, int nInH, int nInW, int nOutH, int nOutW);
    // 20260326 ZJH ConcatChannels 反向传播（按通道拆分梯度）
    int omCudaConcatChannelsBackward(const float* pGradOut, float* pGradA, float* pGradB,
        int nBatch, int nCA, int nCB, int nHW);
    // 20260326 ZJH BCEWithLogits 反向传播
    int omCudaBCEWithLogitsBackward(const float* pLogits, const float* pTargets,
        float* pGradLogits, int nCount, float fInvN);

    // ===== 20260327 ZJH Phase 4B: 补齐缺失的前向 kernel =====
    // 20260327 ZJH 逐元素除法
    int omCudaDiv(const float* pA, const float* pB, float* pC, int nCount);
    // 20260327 ZJH 值裁剪
    int omCudaClip(const float* pIn, float* pOut, int nCount, float fMin, float fMax);
    // 20260327 ZJH 双线性上采样前向
    int omCudaUpsampleBilinear(const float* pIn, float* pOut,
        int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW);
    // 20260327 ZJH 通道维度拼接前向
    int omCudaConcatChannels(const float* pA, const float* pB, float* pOut,
        int nBatch, int nC1, int nC2, int nHW);
    // 20260327 ZJH BCE 前向（GPU 归约）
    int omCudaBCEWithLogits(const float* pLogits, const float* pTargets,
        float* pResult, int nCount);
    // 20260327 ZJH CrossEntropy 前向（GPU 归约）
    int omCudaCrossEntropy(const float* pSoftmax, const float* pTarget,
        float* pResult, int nBatch, int nClasses);
    // 20260327 ZJH LayerNorm 反向
    int omCudaLayerNormBackward(const float* pGradOut, const float* pInput,
        const float* pMean, const float* pInvStd, const float* pGamma,
        float* pGradInput, float* pGradGamma, float* pGradBeta,
        int nBatch, int nDim);
    // 20260327 ZJH 转置卷积前向
    int omCudaConvTranspose2d(const float* pInput, const float* pWeight, const float* pBias,
        float* pOutput,
        int nBatch, int nCin, int nHin, int nWin,
        int nCout, int nKH, int nKW, int nStride, int nPad);
    // 20260328 ZJH Softmax Last Dim 反向：gradIn = softmax * (gradOut - dot(gradOut, softmax))
    int omCudaSoftmaxLastDimBackward(const float* pGradOut, const float* pSoftmax,
        float* pGradIn, int nOuter, int nLastDim);
    // 20260328 ZJH 沿最后一维拼接前向
    int omCudaConcatLastDim(const float* pA, const float* pB, float* pOut,
        int nOuter, int nDimA, int nDimB);
    // 20260328 ZJH 沿最后一维拼接反向
    int omCudaConcatLastDimBackward(const float* pGradOut, float* pGradA, float* pGradB,
        int nOuter, int nDimA, int nDimB);
    // 20260328 ZJH 沿最后一维切片前向
    int omCudaSliceLastDim(const float* pIn, float* pOut,
        int nOuter, int nFullDim, int nStart, int nLen);
    // 20260328 ZJH 沿最后一维切片反向
    int omCudaSliceLastDimBackward(const float* pGradOut, float* pGradIn,
        int nOuter, int nFullDim, int nStart, int nLen);
    // 20260328 ZJH Fused Dice Loss 前向：sigmoid + 3 路归约 → 标量损失
    int omCudaDiceLossForward(const float* pLogits, const float* pTarget,
        float* pLoss, float* pSigmoidOut, float* pStats, int nCount);
    // 20260328 ZJH Fused Dice Loss 反向：逐元素 logit 梯度
    int omCudaDiceLossBackward(const float* pSigmoidOut, const float* pTarget,
        const float* pStats, const float* pGradOutput, float* pGradLogits, int nCount);
    // 20260329 ZJH Weighted PixelCE 前向：逐像素 softmax + 加权 CE（NCHW 原生）
    int omCudaWeightedPixelCEForward(
        const float* pLogits, const float* pTarget, const float* pClassWeights,
        float* pSoftmax, float* pLoss, float* pStats,
        int nPixels, int nClasses, int nSpatial);
    // 20260329 ZJH Weighted PixelCE 反向：逐像素 logit 梯度
    int omCudaWeightedPixelCEBackward(
        const float* pSoftmax, const float* pTarget, const float* pClassWeights,
        const float* pGradOutput, const float* pStats,
        float* pGradLogits,
        int nPixels, int nClasses, int nSpatial);

    // ===== ViT Attention Kernels =====
    // 20260330 ZJH QKV split + head rearrange + Q scaling
    int omCudaQkvSplitHeads(const float* pQkv, float* pQ, float* pK, float* pV,
                             int nBatch, int nSeqLen, int nHeads, int nHeadDim,
                             float fScale);
    // 20260330 ZJH Merge heads 反向重排
    int omCudaMergeHeads(const float* pIn, float* pOut,
                          int nBatch, int nSeqLen, int nHeads, int nHeadDim);
}
#endif

export module om.hal.cuda_backend;

export namespace om {

// 20260325 ZJH CUDABackend — GPU 计算后端
// 镜像 CPUBackend 全接口，所有指针均为 GPU 设备内存
// 每个方法仅是对 omCudaXxx C 绑定函数的薄包装
class CUDABackend {
public:
    // ===== 元素运算（GPU 指针） =====

    // 20260325 ZJH 逐元素加法：pOut[i] = pA[i] + pB[i]
    static void add(const float* pA, const float* pB, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaAdd(pA, pB, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 加法内核
#endif
    }

    // 20260325 ZJH 逐元素减法：pOut[i] = pA[i] - pB[i]
    static void sub(const float* pA, const float* pB, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaSub(pA, pB, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 减法内核
#endif
    }

    // 20260325 ZJH 逐元素乘法：pOut[i] = pA[i] * pB[i]
    static void mul(const float* pA, const float* pB, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaMul(pA, pB, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 乘法内核
#endif
    }

    // 20260325 ZJH 加标量：pOut[i] = pA[i] + fScalar
    static void addScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaAddScalar(pA, fScalar, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 加标量内核
#endif
    }

    // 20260325 ZJH 乘标量：pOut[i] = pA[i] * fScalar
    static void mulScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaMulScalar(pA, fScalar, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 乘标量内核
#endif
    }

    // ===== 矩阵运算 =====

    // 20260325 ZJH 矩阵乘法：C[M,N] = A[M,K] * B[K,N]（32x32 tiled）
    static void matmul(const float* pA, const float* pB, float* pC, int nM, int nK, int nN) {
#ifdef OM_HAS_CUDA
        omCudaMatmul(pA, pB, pC, nM, nK, nN);  // 20260325 ZJH 调用 CUDA tiled matmul
#endif
    }

    // 20260325 ZJH 批量矩阵乘法：nBatch 个独立的矩阵乘法
    static void batchedMatmul(const float* pA, const float* pB, float* pC,
                              int nBatch, int nM, int nK, int nN) {
#ifdef OM_HAS_CUDA
        omCudaBatchedMatmul(pA, pB, pC, nBatch, nM, nK, nN);  // 20260325 ZJH 调用 CUDA 批量 matmul
#endif
    }

    // 20260325 ZJH 矩阵转置：pOut[nCols, nRows] = transpose(pIn[nRows, nCols])
    static void transpose(const float* pIn, float* pOut, int nRows, int nCols) {
#ifdef OM_HAS_CUDA
        omCudaTranspose(pIn, pOut, nRows, nCols);  // 20260325 ZJH 调用 CUDA 转置内核
#endif
    }

    // ===== 激活函数 =====

    // 20260325 ZJH ReLU 前向：out[i] = max(0, in[i])
    static void relu(const float* pIn, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaReLU(pIn, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA ReLU 内核
#endif
    }

    // 20260325 ZJH ReLU 反向：grad_in[i] = input[i] > 0 ? grad_out[i] : 0
    static void reluBackward(const float* pInput, const float* pGradOut, float* pGradIn, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaReLUBackward(pInput, pGradOut, pGradIn, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA ReLU 反向内核
#endif
    }

    // 20260325 ZJH Sigmoid 前向：out[i] = 1 / (1 + exp(-in[i]))
    static void sigmoid(const float* pIn, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaSigmoid(pIn, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA Sigmoid 内核
#endif
    }

    // 20260325 ZJH Sigmoid 反向：grad_in = grad * output * (1 - output)
    static void sigmoidBackward(const float* pGrad, const float* pOutput, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaSigmoidBackward(pGrad, pOutput, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA Sigmoid 反向内核
#endif
    }

    // 20260325 ZJH GELU 前向：高斯误差线性单元
    static void gelu(const float* pIn, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaGELU(pIn, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA GELU 内核
#endif
    }

    // 20260325 ZJH GELU 反向
    static void geluBackward(const float* pGrad, const float* pInput, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaGeluBackward(pGrad, pInput, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA GELU 反向内核
#endif
    }

    // 20260325 ZJH SiLU 前向：x * sigmoid(x)
    static void silu(const float* pIn, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaSiLU(pIn, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA SiLU 内核
#endif
    }

    // 20260325 ZJH SiLU 反向
    static void siluBackward(const float* pGrad, const float* pInput, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaSiluBackward(pGrad, pInput, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA SiLU 反向内核
#endif
    }

    // 20260325 ZJH LeakyReLU 前向：正值直通，负值乘以 fSlope
    static void leakyRelu(const float* pIn, float* pOut, size_t nCount, float fSlope = 0.01f) {
#ifdef OM_HAS_CUDA
        omCudaLeakyRelu(pIn, pOut, static_cast<int>(nCount), fSlope);  // 20260325 ZJH 调用 CUDA LeakyReLU 内核
#endif
    }

    // 20260325 ZJH LeakyReLU 反向
    static void leakyReluBackward(const float* pGrad, const float* pInput, float* pOut,
                                  size_t nCount, float fSlope = 0.01f) {
#ifdef OM_HAS_CUDA
        omCudaLeakyReluBackward(pGrad, pInput, pOut, static_cast<int>(nCount), fSlope);  // 20260325 ZJH 调用 CUDA LeakyReLU 反向内核
#endif
    }

    // 20260325 ZJH Tanh 前向
    static void tanhForward(const float* pIn, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaTanh(pIn, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA Tanh 内核
#endif
    }

    // 20260325 ZJH Tanh 反向
    static void tanhBackward(const float* pGrad, const float* pOutput, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaTanhBackward(pGrad, pOutput, pOut, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA Tanh 反向内核
#endif
    }

    // ===== 卷积 =====

    // 20260325 ZJH Conv2d 前向（GPU 指针，使用 im2col+GEMM 策略）
    // pInput: [N, Cin, H, W]  pWeight: [Cout, Cin, KH, KW]  pBias: [Cout] 或 nullptr
    // pOutput: [N, Cout, Hout, Wout]
    static void conv2d(const float* pInput, const float* pWeight, const float* pBias,
                       float* pOutput,
                       int nBatch, int nCin, int nH, int nW,
                       int nCout, int nKH, int nKW,
                       int nPadH, int nPadW, int nStrH, int nStrW) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH 调用 CUDA Conv2d（对称 padding 和 stride 取 H 方向值）
        omCudaConv2d(pInput, pWeight, pBias, pOutput,
                     nBatch, nCin, nH, nW, nCout, nKH, nKW, nStrH, nPadH);
#endif
    }

    // 20260331 ZJH 膨胀卷积 GPU 前向（im2col+GEMM，支持 dilation 和 groups）
    static void dilatedConv2d(const float* pInput, const float* pWeight, const float* pBias,
                               float* pOutput,
                               int nBatch, int nCin, int nH, int nW,
                               int nCout, int nKH, int nKW,
                               int nStride, int nPad, int nDilation, int nGroups) {
#ifdef OM_HAS_CUDA
        omCudaDilatedConv2d(pInput, pWeight, pBias, pOutput,
                            nBatch, nCin, nH, nW, nCout, nKH, nKW,
                            nStride, nPad, nDilation, nGroups);
#endif
    }

    // 20260331 ZJH 膨胀卷积反向权重梯度
    static void dilatedConv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                             float* pGradWeight, float* pGradBias,
                                             int nBatch, int nCin, int nH, int nW,
                                             int nCout, int nKH, int nKW,
                                             int nStride, int nPad, int nDilation) {
#ifdef OM_HAS_CUDA
        omCudaDilatedConv2dBackwardWeight(pInput, pGradOutput, pGradWeight, pGradBias,
                                           nBatch, nCin, nH, nW, nCout, nKH, nKW,
                                           nStride, nPad, nDilation);
#endif
    }

    // 20260325 ZJH Conv2d 反向（对输入求梯度）— col2im + GEMM 完整实现
    static void conv2dBackwardInput(const float* pGradOutput, const float* pWeight,
                                    float* pGradInput,
                                    int nBatch, int nCin, int nH, int nW,
                                    int nCout, int nKH, int nKW,
                                    int nPadH, int nPadW, int nStrH, int nStrW) {
#ifdef OM_HAS_CUDA
        omCudaConv2dBackwardInput(pGradOutput, pWeight, pGradInput,
                                  nBatch, nCin, nH, nW, nCout, nKH, nKW, nStrH, nPadH);
#endif
    }

    // 20260325 ZJH Conv2d 反向（对权重求梯度）— im2col + GEMM 完整实现
    static void conv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                     float* pGradWeight, float* pGradBias,
                                     int nBatch, int nCin, int nH, int nW,
                                     int nCout, int nKH, int nKW,
                                     int nPadH, int nPadW, int nStrH, int nStrW) {
#ifdef OM_HAS_CUDA
        omCudaConv2dBackwardWeight(pInput, pGradOutput, pGradWeight, pGradBias,
                                   nBatch, nCin, nH, nW, nCout, nKH, nKW, nStrH, nPadH);
#endif
    }

    // ===== 池化 =====

    // 20260325 ZJH MaxPool2d 前向：保存最大值索引用于反向传播
    static void maxPool2d(const float* pIn, float* pOut, int* pIndices,
                          int nBatch, int nChannels, int nH, int nW,
                          int nKH, int nKW, int nStride, int nPad) {
#ifdef OM_HAS_CUDA
        omCudaMaxPool2d(pIn, pOut, pIndices,
                        nBatch, nChannels, nH, nW,
                        nKH, nKW, nStride, nStride, nPad, nPad);  // 20260325 ZJH 对称 stride 和 padding
#endif
    }

    // 20260325 ZJH MaxPool2d 反向：将梯度散布回最大值位置
    static void maxPool2dBackward(const float* pGradOut, const int* pIndices,
                                  float* pGradIn,
                                  int nBatch, int nChannels, int nH, int nW,
                                  int nHout, int nWout) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH 先将 gradInput 清零
        int nInputSize = nBatch * nChannels * nH * nW;
        omCudaMemset(pGradIn, 0, static_cast<size_t>(nInputSize) * sizeof(float));
        omCudaMaxPool2dBackward(pGradOut, pIndices, pGradIn,
                                nBatch, nChannels, nH, nW, nHout, nWout);
#endif
    }

    // 20260325 ZJH AvgPool2d 前向
    static void avgPool2d(const float* pIn, float* pOut,
                          int nBatch, int nChannels, int nH, int nW,
                          int nKH, int nKW, int nStride, int nPad) {
#ifdef OM_HAS_CUDA
        omCudaAvgPool2d(pIn, pOut,
                        nBatch, nChannels, nH, nW,
                        nKH, nKW, nStride, nStride, nPad, nPad);  // 20260325 ZJH 对称 stride 和 padding
#endif
    }

    // 20260325 ZJH AvgPool2d 反向
    static void avgPool2dBackward(const float* pGradOut, float* pGradIn,
                                  int nBatch, int nChannels, int nH, int nW,
                                  int nKH, int nKW, int nStride, int nPad) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH 先将 gradInput 清零
        int nInputSize = nBatch * nChannels * nH * nW;
        omCudaMemset(pGradIn, 0, static_cast<size_t>(nInputSize) * sizeof(float));
        int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260325 ZJH 输出高度
        int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260325 ZJH 输出宽度
        (void)nHout; (void)nWout;
        omCudaAvgPool2dBackward(pGradOut, pGradIn,
                                nBatch, nChannels, nH, nW,
                                nKH, nKW, nStride, nStride, nPad, nPad);
#endif
    }

    // 20260325 ZJH AdaptiveAvgPool2d 前向：将任意大小输入池化到固定 (outH, outW)
    static void adaptiveAvgPool2d(const float* pIn, float* pOut,
                                  int nBatch, int nChannels, int nH, int nW,
                                  int nOutH, int nOutW) {
#ifdef OM_HAS_CUDA
        omCudaAdaptiveAvgPool2d(pIn, pOut, nBatch, nChannels, nH, nW, nOutH, nOutW);
#endif
    }

    // ===== 归一化 =====

    // 20260325 ZJH BatchNorm2d 前向（GPU 全流程：统计量计算 + 归一化 + running stats 更新均在 GPU）
    static void batchNorm2d(const float* pInput, const float* pGamma, const float* pBeta,
                            float* pOutput, float* pRunMean, float* pRunVar,
                            float* pSavedMean, float* pSavedInvStd,
                            int nBatch, int nChannels, int nHW, float fEps,
                            float fMomentum, bool bTraining) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH 推导 H 和 W（假设方形，对 BatchNorm 无影响因为只用 nHW）
        // BatchNorm 只关心 nSpatial = nH * nW，不关心具体的 H 和 W 分解
        int nH = 1;       // 20260325 ZJH 将空间维度全部放在 W 上
        int nW = nHW;     // 20260325 ZJH nH * nW = nHW
        omCudaBatchNorm2d(pInput, pOutput, pGamma, pBeta,
                          pRunMean, pRunVar, pSavedMean, pSavedInvStd,
                          nBatch, nChannels, nH, nW,
                          fEps, fMomentum, bTraining ? 1 : 0);
#endif
    }

    // 20260330 ZJH BatchNorm2d 反向：在 GPU 上计算 gradInput/gradGamma/gradBeta
    // 新增 fEps 参数，传递给 cuDNN 路径使用
    static void batchNorm2dBackward(const float* pGradOutput, const float* pInput,
                                    const float* pMean, const float* pInvStd,
                                    const float* pGamma,
                                    float* pGradInput, float* pGradGamma, float* pGradBeta,
                                    int nBatch, int nChannels, int nH, int nW,
                                    float fEps = 1e-5f) {
#ifdef OM_HAS_CUDA
        omCudaBatchNorm2dBackward(pGradOutput, pInput, pMean, pInvStd, pGamma,
                                  pGradInput, pGradGamma, pGradBeta,
                                  nBatch, nChannels, nH, nW, fEps);
#endif
    }

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

    // 20260325 ZJH LayerNorm 前向
    static void layerNorm(const float* pInput, float* pOutput,
                          const float* pGamma, const float* pBeta,
                          float* pSavedMean, float* pSavedInvStd,
                          int nBatch, int nDim, float fEps) {
#ifdef OM_HAS_CUDA
        omCudaLayerNorm(pInput, pOutput, pGamma, pBeta,
                        pSavedMean, pSavedInvStd, nBatch, nDim, fEps);
#endif
    }

    // ===== Softmax / 损失 =====

    // 20260325 ZJH Softmax 前向：沿最后一维归一化
    static void softmax(const float* pIn, float* pOut, int nBatch, int nClasses) {
#ifdef OM_HAS_CUDA
        omCudaSoftmax(pIn, pOut, nBatch, nClasses);  // 20260325 ZJH 调用 CUDA Softmax 内核
#endif
    }

    // 20260325 ZJH Softmax + CrossEntropy 联合反向：grad = (softmax - target) / batch
    static void crossEntropySoftmaxBackward(const float* pSoftmax, const float* pTarget,
                                            float* pGradLogits, int nBatch, int nClasses) {
#ifdef OM_HAS_CUDA
        omCudaSoftmaxCrossEntropyBackward(pSoftmax, pTarget, pGradLogits, nBatch, nClasses);
#endif
    }

    // ===== 优化器 step（GPU 上原地更新 — GPU-Resident 训练关键） =====

    // 20260325 ZJH Adam 优化器 step — 参数/梯度/动量/方差全部在 GPU 上
    // 消除 optimizer.step() 中的 GPU→CPU→GPU 数据乒乓
    static void adamStep(float* pParam, const float* pGrad, float* pM, float* pV,
                         int nCount, float fLr, float fBeta1, float fBeta2,
                         float fEps, int nStep) {
#ifdef OM_HAS_CUDA
        omCudaAdamStep(pParam, pGrad, pM, pV, nCount, fLr, fBeta1, fBeta2, fEps, nStep);
#endif
    }

    // 20260325 ZJH SGD+Momentum 优化器 step — 参数/梯度/速度全部在 GPU 上
    static void sgdStep(float* pParam, const float* pGrad, float* pVelocity,
                        int nCount, float fLr, float fMomentum) {
#ifdef OM_HAS_CUDA
        omCudaSgdStep(pParam, pGrad, pVelocity, nCount, fLr, fMomentum);
#endif
    }

    // ===== 辅助工具 =====

    // 20260325 ZJH GPU 内存填零
    static void fillZeros(float* pData, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaFillZeros(pData, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 填零内核
#endif
    }

    // 20260325 ZJH GPU 内存填 1
    static void fillOnes(float* pData, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaFillOnes(pData, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA 填 1 内核
#endif
    }

    // 20260325 ZJH GPU 内存填充指定值
    static void fillValue(float* pData, size_t nCount, float fValue) {
#ifdef OM_HAS_CUDA
        omCudaFillValue(pData, static_cast<int>(nCount), fValue);  // 20260325 ZJH 调用 CUDA 填值内核
#endif
    }

    // 20260325 ZJH GPU Device-to-Device 拷贝
    static void copy(const float* pSrc, float* pDst, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaCopy(pSrc, pDst, static_cast<int>(nCount));  // 20260325 ZJH 调用 CUDA D2D 拷贝
#endif
    }

    // 20260325 ZJH 广播偏置加法：pOut[n,c,hw] = pData[n,c,hw] + pBias[c]
    static void addBias(const float* pData, const float* pBias, float* pOut,
                        int nBatch, int nChannels, int nHW) {
#ifdef OM_HAS_CUDA
        omCudaAddBias(pData, pBias, pOut, nBatch, nChannels, nHW);  // 20260325 ZJH 调用 CUDA 偏置加法内核
#endif
    }

    // 20260325 ZJH 逐行 Argmax
    static void argmax(const float* pData, int* pOut, int nBatch, int nClasses) {
#ifdef OM_HAS_CUDA
        omCudaArgmax(pData, pOut, nBatch, nClasses);  // 20260325 ZJH 调用 CUDA Argmax 内核
#endif
    }

    // 20260328 ZJH GPU 全局求和 — pResult 为 GPU 设备指针，结果写入 GPU 无需 D2H
    static void sum(const float* pData, float* pResult, int nCount) {
#ifdef OM_HAS_CUDA
        omCudaSum(pData, pResult, nCount);  // 20260328 ZJH 两阶段 warp-shuffle reduction，结果驻留 GPU
#endif
    }

    // 20260326 ZJH AdaptiveAvgPool2d 反向传播
    static void adaptiveAvgPool2dBackward(const float* pGradOut, float* pGradIn,
        int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW) {
#ifdef OM_HAS_CUDA
        omCudaAdaptiveAvgPool2dBackward(pGradOut, pGradIn, nBatch, nChannels, nH, nW, nOutH, nOutW);
#endif
    }

    // 20260326 ZJH AddBias 反向传播
    static void addBiasBackward(const float* pGradOut, float* pGradBias,
        int nN, int nC, int nHW) {
#ifdef OM_HAS_CUDA
        omCudaAddBiasBackward(pGradOut, pGradBias, nN, nC, nHW);
#endif
    }

    // 20260326 ZJH UpsampleBilinear 反向传播
    static void upsampleBilinearBackward(const float* pGradOut, float* pGradIn,
        int nBatch, int nChannels, int nInH, int nInW, int nOutH, int nOutW) {
#ifdef OM_HAS_CUDA
        omCudaUpsampleBilinearBackward(pGradOut, pGradIn, nBatch, nChannels, nInH, nInW, nOutH, nOutW);
#endif
    }

    // 20260326 ZJH ConcatChannels 反向传播
    static void concatChannelsBackward(const float* pGradOut, float* pGradA, float* pGradB,
        int nBatch, int nCA, int nCB, int nHW) {
#ifdef OM_HAS_CUDA
        omCudaConcatChannelsBackward(pGradOut, pGradA, pGradB, nBatch, nCA, nCB, nHW);
#endif
    }

    // 20260326 ZJH BCEWithLogits 反向传播
    static void bceWithLogitsBackward(const float* pLogits, const float* pTargets,
        float* pGradLogits, int nCount, float fInvN) {
#ifdef OM_HAS_CUDA
        omCudaBCEWithLogitsBackward(pLogits, pTargets, pGradLogits, nCount, fInvN);
#endif
    }

    // ===== 20260327 ZJH Phase 4B: 补齐缺失的前向操作 =====

    // 20260327 ZJH 逐元素除法
    static void div(const float* pA, const float* pB, float* pOut, size_t nCount) {
#ifdef OM_HAS_CUDA
        omCudaDiv(pA, pB, pOut, static_cast<int>(nCount));
#endif
    }

    // 20260327 ZJH 值裁剪
    static void clip(const float* pIn, float* pOut, size_t nCount, float fMin, float fMax) {
#ifdef OM_HAS_CUDA
        omCudaClip(pIn, pOut, static_cast<int>(nCount), fMin, fMax);
#endif
    }

    // 20260327 ZJH 双线性上采样前向
    static void upsampleBilinear(const float* pIn, float* pOut,
        int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW) {
#ifdef OM_HAS_CUDA
        omCudaUpsampleBilinear(pIn, pOut, nBatch, nChannels, nH, nW, nOutH, nOutW);
#endif
    }

    // 20260327 ZJH 通道维度拼接前向
    static void concatChannels(const float* pA, const float* pB, float* pOut,
        int nBatch, int nC1, int nC2, int nHW) {
#ifdef OM_HAS_CUDA
        omCudaConcatChannels(pA, pB, pOut, nBatch, nC1, nC2, nHW);
#endif
    }

    // 20260327 ZJH BCE 前向（GPU 归约，结果写入 GPU float，调用方需 D2H 读取标量）
    static void bceWithLogitsForward(const float* pLogits, const float* pTargets,
        float* pResult, int nCount) {
#ifdef OM_HAS_CUDA
        omCudaBCEWithLogits(pLogits, pTargets, pResult, nCount);
#endif
    }

    // 20260327 ZJH CrossEntropy 前向（GPU 归约）
    static void crossEntropyForward(const float* pSoftmax, const float* pTarget,
        float* pResult, int nBatch, int nClasses) {
#ifdef OM_HAS_CUDA
        omCudaCrossEntropy(pSoftmax, pTarget, pResult, nBatch, nClasses);
#endif
    }

    // 20260327 ZJH LayerNorm 反向
    static void layerNormBackward(const float* pGradOut, const float* pInput,
        const float* pMean, const float* pInvStd, const float* pGamma,
        float* pGradInput, float* pGradGamma, float* pGradBeta,
        int nBatch, int nDim) {
#ifdef OM_HAS_CUDA
        omCudaLayerNormBackward(pGradOut, pInput, pMean, pInvStd, pGamma,
                                pGradInput, pGradGamma, pGradBeta, nBatch, nDim);
#endif
    }

    // 20260327 ZJH 转置卷积前向
    static void convTranspose2d(const float* pInput, const float* pWeight, const float* pBias,
        float* pOutput,
        int nBatch, int nCin, int nHin, int nWin,
        int nCout, int nKH, int nKW, int nStride, int nPad) {
#ifdef OM_HAS_CUDA
        omCudaConvTranspose2d(pInput, pWeight, pBias, pOutput,
                              nBatch, nCin, nHin, nWin, nCout, nKH, nKW, nStride, nPad);
#endif
    }

    // 20260328 ZJH 沿最后一维拼接前向
    static void concatLastDim(const float* pA, const float* pB, float* pOut,
        int nOuter, int nDimA, int nDimB) {
#ifdef OM_HAS_CUDA
        omCudaConcatLastDim(pA, pB, pOut, nOuter, nDimA, nDimB);
#endif
    }

    // 20260328 ZJH 沿最后一维拼接反向
    static void concatLastDimBackward(const float* pGradOut, float* pGradA, float* pGradB,
        int nOuter, int nDimA, int nDimB) {
#ifdef OM_HAS_CUDA
        omCudaConcatLastDimBackward(pGradOut, pGradA, pGradB, nOuter, nDimA, nDimB);
#endif
    }

    // 20260328 ZJH 沿最后一维切片前向
    static void sliceLastDim(const float* pIn, float* pOut,
        int nOuter, int nFullDim, int nStart, int nLen) {
#ifdef OM_HAS_CUDA
        omCudaSliceLastDim(pIn, pOut, nOuter, nFullDim, nStart, nLen);
#endif
    }

    // 20260328 ZJH 沿最后一维切片反向
    static void sliceLastDimBackward(const float* pGradOut, float* pGradIn,
        int nOuter, int nFullDim, int nStart, int nLen) {
#ifdef OM_HAS_CUDA
        omCudaSliceLastDimBackward(pGradOut, pGradIn, nOuter, nFullDim, nStart, nLen);
#endif
    }

    // 20260328 ZJH Softmax Last Dim 反向：gradIn = softmax * (gradOut - dot(gradOut, softmax))
    // pGradOut: [nOuter, nLastDim] 上游梯度
    // pSoftmax: [nOuter, nLastDim] 前向 softmax 输出
    // pGradIn:  [nOuter, nLastDim] 输出的输入梯度
    static void softmaxLastDimBackward(const float* pGradOut, const float* pSoftmax,
        float* pGradIn, int nOuter, int nLastDim) {
#ifdef OM_HAS_CUDA
        omCudaSoftmaxLastDimBackward(pGradOut, pSoftmax, pGradIn, nOuter, nLastDim);
#endif
    }

    // 20260328 ZJH Fused Dropout Forward：GPU 端 SplitMix64 mask 生成 + 应用，零 CPU→GPU mask 传输
    // pIn[N]: 输入 GPU 数据  pOut[N]: 输出 GPU 数据（缩放后）
    // pMask[N]: 输出 GPU mask（0/1，反向需要）
    // fKeepProb = 1 - dropProb  nSeed: CPU 端随机种子
    static void dropoutForward(const float* pIn, float* pOut, float* pMask,
        int nCount, float fKeepProb, unsigned long long nSeed) {
#ifdef OM_HAS_CUDA
        omCudaDropoutForward(pIn, pOut, pMask, nCount, fKeepProb, nSeed);  // 20260328 ZJH 调用 fused dropout kernel
#endif
    }

    // 20260328 ZJH Fused Dice Loss 前向：sigmoid + 3 路归约，1 kernel 替代 10 步 tensor ops
    // pLogits[N]: 原始 logits  pTarget[N]: one-hot 目标  N = B*C*H*W
    // pLoss[1]: 输出标量损失  pSigmoidOut[N]: sigmoid 中间结果（反向需要）
    // pStats[3]: {intersection, predSum, targetSum}（反向需要）
    static void diceLossForward(const float* pLogits, const float* pTarget,
        float* pLoss, float* pSigmoidOut, float* pStats, int nCount) {
#ifdef OM_HAS_CUDA
        omCudaDiceLossForward(pLogits, pTarget, pLoss, pSigmoidOut, pStats, nCount);  // 20260328 ZJH 调用 fused 前向 kernel
#endif
    }

    // 20260328 ZJH Fused Dice Loss 反向：逐元素计算 dLoss/dLogit（纯并行，无归约）
    // pSigmoidOut[N]: 前向 sigmoid 输出  pTarget[N]: one-hot 目标
    // pStats[3]: 前向统计量  pGradOutput[1]: 上游标量梯度
    // pGradLogits[N]: 输出的 logit 梯度
    static void diceLossBackward(const float* pSigmoidOut, const float* pTarget,
        const float* pStats, const float* pGradOutput, float* pGradLogits, int nCount) {
#ifdef OM_HAS_CUDA
        omCudaDiceLossBackward(pSigmoidOut, pTarget, pStats, pGradOutput, pGradLogits, nCount);  // 20260328 ZJH 调用 fused 反向 kernel
#endif
    }

    // 20260329 ZJH Weighted Pixel-wise CE 前向：分割模型 GPU 全驻留损失
    // pLogits [B,C,H,W] 模型输出, pTarget [N] int 类别 ID (N=B*H*W)
    // pClassWeights [C] 反频率权重, pSoftmax [B,C,H,W] 输出（反向需要）
    // pLoss [1] 归一化损失, pStats [2] 临时缓冲 {lossSum, weightSum}（反向需要）
    static void weightedPixelCEForward(
        const float* pLogits, const float* pTarget, const float* pClassWeights,
        float* pSoftmax, float* pLoss, float* pStats,
        int nPixels, int nClasses, int nSpatial) {
#ifdef OM_HAS_CUDA
        omCudaWeightedPixelCEForward(pLogits, pTarget, pClassWeights,
            pSoftmax, pLoss, pStats, nPixels, nClasses, nSpatial);  // 20260329 ZJH 调用 fused 前向 kernel
#endif
    }

    // 20260329 ZJH Weighted Pixel-wise CE 反向：逐像素 logit 梯度
    // pSoftmax [B,C,H,W] 前向保存, pTarget [N] 类别 ID, pClassWeights [C]
    // pGradOutput [1] 上游梯度, pStats [2] 前向统计, pGradLogits [B,C,H,W] 输出
    static void weightedPixelCEBackward(
        const float* pSoftmax, const float* pTarget, const float* pClassWeights,
        const float* pGradOutput, const float* pStats,
        float* pGradLogits,
        int nPixels, int nClasses, int nSpatial) {
#ifdef OM_HAS_CUDA
        omCudaWeightedPixelCEBackward(pSoftmax, pTarget, pClassWeights,
            pGradOutput, pStats, pGradLogits, nPixels, nClasses, nSpatial);  // 20260329 ZJH 调用 fused 反向 kernel
#endif
    }

    // ===== ViT Attention Kernels =====

    // 20260330 ZJH QKV split + head rearrange: [B*S, 3D] → Q[BH,S,d] K[BH,S,d] V[BH,S,d]
    // Q 自动乘以 fScale 消除后续缩放步骤，全 GPU 操作无 D2H
    static void qkvSplitHeads(const float* pQkv, float* pQ, float* pK, float* pV,
                               int nBatch, int nSeqLen, int nHeads, int nHeadDim,
                               float fScale) {
#ifdef OM_HAS_CUDA
        omCudaQkvSplitHeads(pQkv, pQ, pK, pV, nBatch, nSeqLen, nHeads, nHeadDim, fScale);
#endif
    }

    // 20260330 ZJH Merge heads: [BH, S, d] → [B*S, D]
    static void mergeHeads(const float* pIn, float* pOut,
                            int nBatch, int nSeqLen, int nHeads, int nHeadDim) {
#ifdef OM_HAS_CUDA
        omCudaMergeHeads(pIn, pOut, nBatch, nSeqLen, nHeads, nHeadDim);
#endif
    }
};

} // namespace om
