// 20260328 ZJH CUDA 内核头文件 — OmniMatch GPU 加速
// 提供 C 风格接口，可被 C++23 模块通过 extern "C" 调用
// 覆盖核心运算：matmul/conv2d/elementwise/softmax/batchnorm/relu 等
// 20260324 ZJH 新增：32x32 tiled matmul / im2col+GEMM 卷积 / GPU reduction /
//                     异步流 / 内存池 / 转置 / 均值归约 / 资源清理
#pragma once

#include <stddef.h>  // 20260324 ZJH size_t 定义（C 兼容）

#ifdef __cplusplus
extern "C" {
#endif

// ===== 设备管理 =====

// 20260320 ZJH 初始化 CUDA 设备，返回 0 成功，非 0 失败
// 20260324 ZJH 同时初始化异步流（计算流 + 传输流）
int dfCudaInit(int nDeviceId);

// 20260320 ZJH 获取设备数量
int dfCudaGetDeviceCount();

// 20260320 ZJH 获取设备名称（写入 pNameBuf，最大 nBufSize 字节）
int dfCudaGetDeviceName(int nDeviceId, char* pNameBuf, int nBufSize);

// 20260320 ZJH 获取设备显存信息（单位字节）
int dfCudaGetMemInfo(int nDeviceId, size_t* pFreeMem, size_t* pTotalMem);

// ===== 内存管理（使用内存池） =====

// 20260320 ZJH 分配 GPU 内存（20260324 ZJH 通过内存池分配，复用已释放块）
int dfCudaMalloc(void** ppDevPtr, size_t nBytes);

// 20260320 ZJH 释放 GPU 内存（20260324 ZJH 归还到内存池，不实际释放）
int dfCudaFree(void* pDevPtr);

// 20260320 ZJH Host -> Device 拷贝（同步）
int dfCudaCopyH2D(void* pDst, const void* pSrc, size_t nBytes);

// 20260320 ZJH Device -> Host 拷贝（同步）
int dfCudaCopyD2H(void* pDst, const void* pSrc, size_t nBytes);

// 20260320 ZJH Device -> Device 拷贝
int dfCudaCopyD2D(void* pDst, const void* pSrc, size_t nBytes);

// 20260320 ZJH 清零 GPU 内存
int dfCudaMemset(void* pDev, int nValue, size_t nBytes);

// 20260324 ZJH 异步 Host -> Device 拷贝（使用传输流，与计算流并行）
int dfCudaAsyncCopyH2D(void* pDst, const void* pSrc, size_t nBytes);

// 20260324 ZJH 异步 Device -> Host 拷贝（使用传输流）
int dfCudaAsyncCopyD2H(void* pDst, const void* pSrc, size_t nBytes);

// 20260324 ZJH 同步传输流（等待所有异步传输完成）
int dfCudaSyncTransferStream();

// 20260324 ZJH 同步计算流（等待所有异步计算完成）
int dfCudaSyncComputeStream();

// 20260324 ZJH 释放所有池内存并销毁流（程序退出时调用）
int dfCudaCleanup();

// ===== 元素运算（ILP 优化：每线程处理 4 个元素） =====

// 20260320 ZJH 逐元素加法：pC[i] = pA[i] + pB[i]
int dfCudaAdd(const float* pA, const float* pB, float* pC, int nCount);

// 20260320 ZJH 逐元素减法
int dfCudaSub(const float* pA, const float* pB, float* pC, int nCount);

// 20260320 ZJH 逐元素乘法
int dfCudaMul(const float* pA, const float* pB, float* pC, int nCount);

// 20260320 ZJH 乘标量
int dfCudaMulScalar(const float* pA, float fScalar, float* pC, int nCount);

// 20260320 ZJH 加标量
int dfCudaAddScalar(const float* pA, float fScalar, float* pC, int nCount);

// ===== 激活函数 =====

// 20260320 ZJH ReLU 前向
int dfCudaReLU(const float* pIn, float* pOut, int nCount);

// 20260320 ZJH ReLU 反向
int dfCudaReLUBackward(const float* pIn, const float* pGradOut, float* pGradIn, int nCount);

// 20260320 ZJH Sigmoid 前向
int dfCudaSigmoid(const float* pIn, float* pOut, int nCount);

// 20260320 ZJH GELU 前向
int dfCudaGELU(const float* pIn, float* pOut, int nCount);

// 20260320 ZJH SiLU 前向
int dfCudaSiLU(const float* pIn, float* pOut, int nCount);

// ===== 矩阵运算 =====

// 20260320 ZJH 矩阵乘法：C[M,N] = A[M,K] * B[K,N]
// 20260324 ZJH 升级为 32x32 tiled matmul + __restrict__ 优化
int dfCudaMatmul(const float* pA, const float* pB, float* pC, int nM, int nK, int nN);

// 20260320 ZJH 批量矩阵乘法
int dfCudaBatchedMatmul(const float* pA, const float* pB, float* pC,
                        int nBatch, int nM, int nK, int nN);

// 20260324 ZJH 矩阵转置：pOut[nCols, nRows] = transpose(pIn[nRows, nCols])
// 使用 shared memory tile 避免非合并内存访问
int dfCudaTranspose(const float* pIn, float* pOut, int nRows, int nCols);

// ===== Softmax =====

// 20260320 ZJH Softmax 前向：沿最后一维
int dfCudaSoftmax(const float* pIn, float* pOut, int nBatch, int nClasses);

// ===== 归约 =====

// 20260320 ZJH 全局求和（20260324 ZJH 升级为 warp-shuffle 优化 reduction）
int dfCudaSum(const float* pData, float* pResult, int nCount);

// 20260324 ZJH 全局求均值：pResult = sum(pData) / nCount
int dfCudaMean(const float* pData, float* pResult, int nCount);

// ===== 卷积 =====

// 20260320 ZJH Conv2d 前向
// 20260324 ZJH 自动选择 im2col+GEMM（大卷积核）或朴素实现（小卷积核）
int dfCudaConv2d(const float* pInput, const float* pWeight, const float* pBias,
                 float* pOutput,
                 int nBatch, int nCin, int nH, int nW,
                 int nCout, int nKH, int nKW, int nStride, int nPad);

// 20260324 ZJH Conv2d 前向（显式 im2col + GEMM 策略）
int dfCudaConv2dIm2col(const float* pInput, const float* pWeight, const float* pBias,
                        float* pOutput,
                        int nBatch, int nCin, int nH, int nW,
                        int nCout, int nKH, int nKW, int nStride, int nPad);

// ===== BatchNorm =====

// 20260320 ZJH BatchNorm2d 前向
// 20260324 ZJH 训练模式下 mean/var 完全在 GPU 上计算（替代旧的 CPU 回传）
int dfCudaBatchNorm2d(const float* pInput, float* pOutput,
                      const float* pGamma, const float* pBeta,
                      float* pRunMean, float* pRunVar,
                      float* pSavedMean, float* pSavedInvStd,
                      int nBatch, int nChannels, int nH, int nW,
                      float fEps, float fMomentum, int bTraining);

// ===== LayerNorm =====

// 20260320 ZJH LayerNorm 前向
// 20260324 ZJH mean/invStd 计算完全在 GPU 上完成（替代旧的 CPU 辅助）
int dfCudaLayerNorm(const float* pInput, float* pOutput,
                    const float* pGamma, const float* pBeta,
                    float* pSavedMean, float* pSavedInvStd,
                    int nBatch, int nDim, float fEps);

// ===== 同步 =====

// 20260320 ZJH 同步当前设备
int dfCudaSynchronize();

// =====================================================================
// Phase 2 新增接口 — GPU-Resident Tensor 全算子支持
// 20260325 ZJH 为 CUDABackend 提供完整的前向/反向/优化器/辅助接口
// =====================================================================

// ===== 激活函数（新增前向 + 反向） =====

// 20260325 ZJH LeakyReLU 前向：正值直通，负值乘以 fSlope
int dfCudaLeakyRelu(const float* pIn, float* pOut, int nCount, float fSlope);

// 20260325 ZJH Tanh 前向
int dfCudaTanh(const float* pIn, float* pOut, int nCount);

// 20260325 ZJH Tanh 反向：grad_in = grad * (1 - output^2)
int dfCudaTanhBackward(const float* pGrad, const float* pOutput, float* pOut, int nCount);

// 20260325 ZJH GELU 反向
int dfCudaGeluBackward(const float* pGrad, const float* pInput, float* pOut, int nCount);

// 20260325 ZJH SiLU 反向
int dfCudaSiluBackward(const float* pGrad, const float* pInput, float* pOut, int nCount);

// 20260325 ZJH Sigmoid 反向：grad_in = grad * output * (1 - output)
int dfCudaSigmoidBackward(const float* pGrad, const float* pOutput, float* pOut, int nCount);

// 20260325 ZJH LeakyReLU 反向
int dfCudaLeakyReluBackward(const float* pGrad, const float* pInput, float* pOut,
                             int nCount, float fSlope);

// ===== 池化 =====

// 20260325 ZJH MaxPool2d 前向（保存索引用于反向）
int dfCudaMaxPool2d(const float* pIn, float* pOut, int* pIndices,
                     int nN, int nC, int nH, int nW,
                     int nKH, int nKW, int nSH, int nSW,
                     int nPH, int nPW);

// 20260325 ZJH MaxPool2d 反向（使用前向保存的索引）
int dfCudaMaxPool2dBackward(const float* pGradOut, const int* pIndices,
                             float* pGradIn,
                             int nN, int nC, int nH, int nW,
                             int nHout, int nWout);

// 20260325 ZJH AvgPool2d 前向
int dfCudaAvgPool2d(const float* pIn, float* pOut,
                     int nN, int nC, int nH, int nW,
                     int nKH, int nKW, int nSH, int nSW,
                     int nPH, int nPW);

// 20260325 ZJH AvgPool2d 反向
int dfCudaAvgPool2dBackward(const float* pGradOut, float* pGradIn,
                             int nN, int nC, int nH, int nW,
                             int nKH, int nKW, int nSH, int nSW,
                             int nPH, int nPW);

// 20260325 ZJH AdaptiveAvgPool2d 前向：任意输入大小 → 固定输出大小
int dfCudaAdaptiveAvgPool2d(const float* pIn, float* pOut,
                             int nN, int nC, int nH, int nW,
                             int nOutH, int nOutW);

// ===== 广播偏置 / Dropout / 填充 / 拷贝 / Argmax =====

// 20260325 ZJH 广播偏置加法：pOut[n,c,hw] = pData[n,c,hw] + pBias[c]
int dfCudaAddBias(const float* pData, const float* pBias, float* pOut,
                   int nN, int nC, int nHW);

// 20260325 ZJH Dropout 前向：pMask 为预生成的随机 mask
int dfCudaDropout(const float* pIn, float* pOut, const float* pMask,
                   int nCount, float fProb, int bTraining);

// 20260325 ZJH GPU 内存填零
int dfCudaFillZeros(float* pData, int nCount);

// 20260325 ZJH GPU 内存填 1
int dfCudaFillOnes(float* pData, int nCount);

// 20260325 ZJH GPU 内存填充指定值
int dfCudaFillValue(float* pData, int nCount, float fValue);

// 20260325 ZJH GPU Device-to-Device 拷贝（float 数组专用）
int dfCudaCopy(const float* pSrc, float* pDst, int nCount);

// 20260325 ZJH 逐行 Argmax：返回每行最大值索引
int dfCudaArgmax(const float* pData, int* pOut, int nBatch, int nClasses);

// ===== 优化器（GPU-Resident 训练关键） =====

// 20260325 ZJH Adam 优化器 step — 直接在 GPU 上执行参数更新
int dfCudaAdamStep(float* pParam, const float* pGrad,
                    float* pM, float* pV,
                    int nCount, float fLr,
                    float fBeta1, float fBeta2,
                    float fEps, int nStep);

// 20260325 ZJH SGD+Momentum 优化器 step — 直接在 GPU 上执行参数更新
int dfCudaSgdStep(float* pParam, const float* pGrad,
                   float* pVelocity, int nCount,
                   float fLr, float fMomentum);

// ===== 损失反向 =====

// 20260325 ZJH Softmax + CrossEntropy 联合反向：grad = (softmax - target) / batch
int dfCudaSoftmaxCrossEntropyBackward(const float* pSoftmax, const float* pTarget,
                                       float* pGradLogits,
                                       int nBatch, int nClasses);

// ===== Conv2d 反向（GPU im2col+GEMM） =====

// 20260325 ZJH Conv2d 反向对输入求梯度（col2im + GEMM）
int dfCudaConv2dBackwardInput(const float* pGradOutput, const float* pWeight,
                               float* pGradInput,
                               int nBatch, int nCin, int nH, int nW,
                               int nCout, int nKH, int nKW, int nStride, int nPad);

// 20260325 ZJH Conv2d 反向对权重求梯度（im2col + GEMM）
int dfCudaConv2dBackwardWeight(const float* pInput, const float* pGradOutput,
                                float* pGradWeight, float* pGradBias,
                                int nBatch, int nCin, int nH, int nW,
                                int nCout, int nKH, int nKW, int nStride, int nPad);

// ===== BatchNorm2d 反向 =====

// 20260325 ZJH BatchNorm2d 反向：在 GPU 上计算 gradInput/gradGamma/gradBeta
int dfCudaBatchNorm2dBackward(const float* pGradOutput, const float* pInput,
                               const float* pMean, const float* pInvStd,
                               const float* pGamma,
                               float* pGradInput, float* pGradGamma, float* pGradBeta,
                               int nBatch, int nChannels, int nH, int nW);

// =====================================================================
// Phase 3 新增接口 — 全 GPU backward + Phase 4B 前向补全
// 20260326 ZJH 补全 autograd backward 全 GPU 化所需的 C 接口声明
// =====================================================================

// ===== Backward 反向传播（7 个 CPU 回退消除） =====

// 20260326 ZJH AdaptiveAvgPool2d 反向：每个输入像素累加所有覆盖它的输出池化窗口的梯度
int dfCudaAdaptiveAvgPool2dBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW);

// 20260326 ZJH AddBias 反向：对 N×HW 维度归约，得到每个通道的 bias 梯度
int dfCudaAddBiasBackward(
    const float* pGradOut, float* pGradBias,
    int nN, int nC, int nHW);

// 20260326 ZJH UpsampleBilinear 反向：将输出梯度通过双线性插值权重分配回输入
int dfCudaUpsampleBilinearBackward(
    const float* pGradOut, float* pGradIn,
    int nBatch, int nChannels, int nInH, int nInW, int nOutH, int nOutW);

// 20260326 ZJH ConcatChannels 反向：按通道分割梯度，前 nCA 通道→gradA，后 nCB 通道→gradB
int dfCudaConcatChannelsBackward(
    const float* pGradOut, float* pGradA, float* pGradB,
    int nBatch, int nCA, int nCB, int nHW);

// 20260326 ZJH BCEWithLogits 反向：grad = (sigmoid(logit) - target) * fInvN
int dfCudaBCEWithLogitsBackward(
    const float* pLogits, const float* pTargets, float* pGradLogits,
    int nCount, float fInvN);

// 20260327 ZJH LayerNorm 反向：计算 gradInput/gradGamma/gradBeta
// pGradGamma 和 pGradBeta 由 kernel 内 atomicAdd 累加（调用前需清零）
int dfCudaLayerNormBackward(
    const float* pGradOut, const float* pInput,
    const float* pMean, const float* pInvStd, const float* pGamma,
    float* pGradInput, float* pGradGamma, float* pGradBeta,
    int nBatch, int nDim);

// ===== Phase 4B 前向补全（消除 D2H 回退） =====

// 20260327 ZJH 逐元素除法：pC[i] = pA[i] / pB[i]
int dfCudaDiv(const float* pA, const float* pB, float* pC, int nCount);

// 20260327 ZJH 值裁剪：pOut[i] = clamp(pIn[i], fMin, fMax)
int dfCudaClip(const float* pIn, float* pOut, int nCount, float fMin, float fMax);

// 20260327 ZJH 双线性上采样前向：输入 [N,C,H,W] → 输出 [N,C,nOutH,nOutW]
int dfCudaUpsampleBilinear(
    const float* pIn, float* pOut,
    int nBatch, int nChannels, int nH, int nW, int nOutH, int nOutW);

// 20260327 ZJH 通道维度拼接前向：[N,C1,H,W] + [N,C2,H,W] → [N,C1+C2,H,W]
int dfCudaConcatChannels(
    const float* pA, const float* pB, float* pOut,
    int nBatch, int nC1, int nC2, int nHW);

// 20260327 ZJH BCE 前向归约：返回标量损失（pResult 为 GPU float 指针）
// 数值稳定公式：max(x,0) - x*y + log(1+exp(-|x|))
int dfCudaBCEWithLogits(
    const float* pLogits, const float* pTargets, float* pResult, int nCount);

// 20260327 ZJH CrossEntropy 前向归约：-sum(target*log(softmax+eps)) / batch
int dfCudaCrossEntropy(
    const float* pSoftmax, const float* pTarget, float* pResult,
    int nBatch, int nClasses);

// 20260327 ZJH 转置卷积前向（atomicAdd scatter 策略）
// pBias 可为 nullptr（无偏置）
int dfCudaConvTranspose2d(
    const float* pInput, const float* pWeight, const float* pBias, float* pOutput,
    int nBatch, int nCin, int nHin, int nWin,
    int nCout, int nKH, int nKW, int nStride, int nPad);

// =====================================================================
// 20260328 ZJH ConcatLastDim / SliceLastDim — 沿最后一维拼接/切片（GPU 前向 + 反向）
// =====================================================================

// 20260328 ZJH 沿最后一维拼接前向：[outer, dimA] + [outer, dimB] → [outer, dimA+dimB]
int omCudaConcatLastDim(
    const float* pA, const float* pB, float* pOut,
    int nOuter, int nDimA, int nDimB);

// 20260328 ZJH 沿最后一维拼接反向：gradOut → gradA + gradB
int omCudaConcatLastDimBackward(
    const float* pGradOut, float* pGradA, float* pGradB,
    int nOuter, int nDimA, int nDimB);

// 20260328 ZJH 沿最后一维切片前向：[outer, fullDim] → [outer, len]（从 nStart 开始取 nLen 列）
int omCudaSliceLastDim(
    const float* pIn, float* pOut,
    int nOuter, int nFullDim, int nStart, int nLen);

// 20260328 ZJH 沿最后一维切片反向：gradOut[outer, len] → gradIn[outer, fullDim]（scatter）
// 调用方需先将 pGradIn 清零
int omCudaSliceLastDimBackward(
    const float* pGradOut, float* pGradIn,
    int nOuter, int nFullDim, int nStart, int nLen);

// 20260328 ZJH Softmax Last Dim 反向：gradIn[i] = softmax[i] * (gradOut[i] - dot(gradOut, softmax))
// 标准 softmax Jacobian-vector product，每行独立计算
int omCudaSoftmaxLastDimBackward(
    const float* pGradOut, const float* pSoftmax, float* pGradIn,
    int nOuter, int nLastDim);

// =====================================================================
// 20260328 ZJH Fused Dice Loss — 融合 sigmoid + 3 路归约 + 标量计算为单 kernel
// =====================================================================

// 20260328 ZJH Fused Dropout Forward：GPU 端 mask 生成(SplitMix64) + 应用，零 CPU→GPU 传输
// pMask[N]: 输出 mask（反向传播需要）  nSeed: CPU 端随机种子
int omCudaDropoutForward(const float* pIn, float* pOut, float* pMask,
    int nCount, float fKeepProb, unsigned long long nSeed);

// 20260328 ZJH Dice Loss 前向：sigmoid + 3 路归约 → host 标量 → 回写 GPU
// pLoss[1] 输出标量损失, pSigmoidOut[N] sigmoid 中间结果, pStats[3] = {I, P, T}
int omCudaDiceLossForward(
    const float* pLogits, const float* pTarget,
    float* pLoss, float* pSigmoidOut, float* pStats, int nCount);

// 20260328 ZJH Dice Loss 反向：逐元素计算 dLoss/dLogit（纯并行，无归约）
int omCudaDiceLossBackward(
    const float* pSigmoidOut, const float* pTarget,
    const float* pStats, const float* pGradOutput,
    float* pGradLogits, int nCount);

#ifdef __cplusplus
}
#endif
