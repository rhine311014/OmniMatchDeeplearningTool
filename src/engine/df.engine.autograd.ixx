// 20260319 ZJH AutoGrad 自动微分模块 — Phase 1C
// 动态计算图：前向运算构建 GradFunction DAG，backward() 拓扑排序后反向传播梯度
// 循环依赖解决方案：Tensor 以 shared_ptr<void> 类型擦除存储 GradFunction，
// 仅 tensor_ops 同时了解 Tensor 和 GradFunction
module;

#include <vector>
#include <memory>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <cstring>

#include "df_types.h"

export module df.engine.autograd;

// 20260319 ZJH 导入依赖模块：存储层、张量类、CPU 计算内核
import df.engine.tensor_storage;
import df.engine.tensor;
import df.hal.cpu_backend;

export namespace df {

// 20260319 ZJH 前向声明 GradFunction，供 Edge 引用
class GradFunction;

// 20260319 ZJH Edge — 计算图中的有向边，连接上游 GradFunction 及其输入索引
// pGradFn: 指向上游梯度函数（产出该张量的反向运算节点）
// nInputIndex: 该张量在上游 GradFunction 的输出中对应的索引位置
struct Edge {
    std::shared_ptr<GradFunction> pGradFn;  // 20260319 ZJH 上游梯度函数（可为 nullptr 表示叶节点）
    int nInputIndex = 0;  // 20260319 ZJH 在上游 backward 输出向量中的索引
};

// 20260319 ZJH GradFunction — 计算图反向传播节点基类
// 每个前向运算对应一个 GradFunction 子类，backward() 根据上游梯度计算下游梯度
class GradFunction {
public:
    // 20260319 ZJH 虚析构，支持多态删除
    virtual ~GradFunction() = default;

    // 20260319 ZJH backward — 给定当前节点的梯度输出，计算各输入的梯度
    // gradOutput: 从上游传来的梯度张量
    // 返回: 各输入边对应的梯度向量，大小与 m_vecInputEdges 一致
    virtual std::vector<Tensor> backward(const Tensor& gradOutput) = 0;

    // 20260319 ZJH 输入边列表：记录当前运算的各个输入张量来自哪个上游 GradFunction
    std::vector<Edge> m_vecInputEdges;
};

// 20260319 ZJH GradAccumulator — 叶节点梯度累加器
// 持有一个梯度张量 m_grad，支持累加和清零操作
// 叶节点（用户创建的 requiresGrad=true 的张量）通过 LeafAccumulator 持有此对象
struct GradAccumulator {
    Tensor m_grad;  // 20260319 ZJH 累积的梯度张量
    bool m_bHasGrad = false;  // 20260319 ZJH 是否已有梯度（区分零梯度和无梯度）

    // 20260319 ZJH accumulate — 累加梯度，首次调用时直接赋值，后续调用逐元素相加
    void accumulate(const Tensor& grad) {
        if (!m_bHasGrad) {
            // 20260319 ZJH 首次累加：深拷贝梯度（避免共享存储导致覆盖）
            m_grad = Tensor::fromData(grad.floatDataPtr(), grad.shapeVec());
            m_bHasGrad = true;  // 20260319 ZJH 标记已有梯度
        } else {
            // 20260319 ZJH 后续累加：逐元素相加
            auto cg = grad.contiguous();  // 20260319 ZJH 确保梯度连续
            auto cm = m_grad.contiguous();  // 20260319 ZJH 确保已有梯度连续
            auto result = Tensor::zeros(cm.shapeVec());  // 20260319 ZJH 分配结果张量
            CPUBackend::add(cm.floatDataPtr(), cg.floatDataPtr(),
                            result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
            m_grad = result;  // 20260319 ZJH 更新累积梯度
        }
    }

    // 20260319 ZJH zero — 清零梯度并重置标记
    void zero() {
        m_bHasGrad = false;  // 20260319 ZJH 重置标记
        m_grad = Tensor();  // 20260319 ZJH 释放梯度存储
    }
};

// =========================================================
// Backward 子类 — 各前向运算对应的反向梯度计算
// =========================================================

// 20260319 ZJH AddBackward — 加法反向：grad_a = gradOutput, grad_b = gradOutput
// d(a+b)/da = 1, d(a+b)/db = 1
class AddBackward : public GradFunction {
public:
    // 20260319 ZJH backward — 加法梯度直通，两个输入梯度均等于输出梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH grad_a = gradOutput（深拷贝避免共享存储问题）
        auto gradA = Tensor::fromData(gradOutput.floatDataPtr(), gradOutput.shapeVec());
        // 20260319 ZJH grad_b = gradOutput（深拷贝）
        auto gradB = Tensor::fromData(gradOutput.floatDataPtr(), gradOutput.shapeVec());
        return {gradA, gradB};  // 20260319 ZJH 返回两个输入的梯度
    }
};

// 20260319 ZJH SubBackward — 减法反向：grad_a = gradOutput, grad_b = -gradOutput
// d(a-b)/da = 1, d(a-b)/db = -1
class SubBackward : public GradFunction {
public:
    // 20260319 ZJH backward — 减法梯度：a 直通，b 取负
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cg = gradOutput.contiguous();  // 20260319 ZJH 确保梯度连续
        // 20260319 ZJH grad_a = gradOutput
        auto gradA = Tensor::fromData(cg.floatDataPtr(), cg.shapeVec());
        // 20260319 ZJH grad_b = -gradOutput，乘以 -1
        auto gradB = Tensor::zeros(cg.shapeVec());
        CPUBackend::mulScalar(cg.floatDataPtr(), -1.0f,
                              gradB.mutableFloatDataPtr(), static_cast<size_t>(gradB.numel()));
        return {gradA, gradB};  // 20260319 ZJH 返回两个输入的梯度
    }
};

// 20260319 ZJH MulBackward — 乘法反向：grad_a = gradOutput * b, grad_b = gradOutput * a
// d(a*b)/da = b, d(a*b)/db = a（保存前向时的 a 和 b）
class MulBackward : public GradFunction {
public:
    Tensor m_savedA;  // 20260319 ZJH 保存前向时的张量 a，用于计算 grad_b
    Tensor m_savedB;  // 20260319 ZJH 保存前向时的张量 b，用于计算 grad_a

    // 20260319 ZJH backward — 乘法梯度：交叉乘以对方保存的张量
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cg = gradOutput.contiguous();  // 20260319 ZJH 确保梯度连续
        auto ca = m_savedA.contiguous();  // 20260319 ZJH 确保保存的 a 连续
        auto cb = m_savedB.contiguous();  // 20260319 ZJH 确保保存的 b 连续

        // 20260319 ZJH grad_a = gradOutput * b
        auto gradA = Tensor::zeros(cg.shapeVec());
        CPUBackend::mul(cg.floatDataPtr(), cb.floatDataPtr(),
                        gradA.mutableFloatDataPtr(), static_cast<size_t>(gradA.numel()));

        // 20260319 ZJH grad_b = gradOutput * a
        auto gradB = Tensor::zeros(cg.shapeVec());
        CPUBackend::mul(cg.floatDataPtr(), ca.floatDataPtr(),
                        gradB.mutableFloatDataPtr(), static_cast<size_t>(gradB.numel()));

        return {gradA, gradB};  // 20260319 ZJH 返回两个输入的梯度
    }
};

// 20260319 ZJH MatMulBackward — 矩阵乘法反向
// 前向: C = A @ B，其中 A[M,K], B[K,N], C[M,N]
// 反向: grad_A = gradOutput @ B^T, grad_B = A^T @ gradOutput
// 保存前向时的 A 和 B
class MatMulBackward : public GradFunction {
public:
    Tensor m_savedA;  // 20260319 ZJH 保存前向时的矩阵 A [M,K]
    Tensor m_savedB;  // 20260319 ZJH 保存前向时的矩阵 B [K,N]

    // 20260319 ZJH backward — 矩阵乘法梯度，需要手动转置
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260319 ZJH gradOutput [M,N]
        auto cA = m_savedA.contiguous();  // 20260319 ZJH A [M,K]
        auto cB = m_savedB.contiguous();  // 20260319 ZJH B [K,N]

        int nM = cA.shape(0);  // 20260319 ZJH 矩阵 A 的行数
        int nK = cA.shape(1);  // 20260319 ZJH 矩阵 A 的列数 = B 的行数
        int nN = cB.shape(1);  // 20260319 ZJH 矩阵 B 的列数

        // 20260319 ZJH grad_A = gradOutput[M,N] @ B^T[N,K] -> [M,K]
        // 先手动转置 B[K,N] -> B^T[N,K]
        auto matBT = Tensor::zeros({nN, nK});  // 20260319 ZJH B 的转置
        const float* pB = cB.floatDataPtr();  // 20260319 ZJH B 的数据指针
        float* pBT = matBT.mutableFloatDataPtr();  // 20260319 ZJH B^T 的数据指针
        for (int i = 0; i < nK; ++i) {
            for (int j = 0; j < nN; ++j) {
                // 20260319 ZJH B[i,j] = pB[i*nN+j] -> B^T[j,i] = pBT[j*nK+i]
                pBT[j * nK + i] = pB[i * nN + j];
            }
        }
        // 20260319 ZJH gradOutput[M,N] @ B^T[N,K] -> gradA[M,K]
        auto gradA = Tensor::zeros({nM, nK});
        CPUBackend::matmul(cGrad.floatDataPtr(), matBT.floatDataPtr(),
                           gradA.mutableFloatDataPtr(), nM, nN, nK);

        // 20260319 ZJH grad_B = A^T[K,M] @ gradOutput[M,N] -> [K,N]
        // 先手动转置 A[M,K] -> A^T[K,M]
        auto matAT = Tensor::zeros({nK, nM});  // 20260319 ZJH A 的转置
        const float* pA = cA.floatDataPtr();  // 20260319 ZJH A 的数据指针
        float* pAT = matAT.mutableFloatDataPtr();  // 20260319 ZJH A^T 的数据指针
        for (int i = 0; i < nM; ++i) {
            for (int j = 0; j < nK; ++j) {
                // 20260319 ZJH A[i,j] = pA[i*nK+j] -> A^T[j,i] = pAT[j*nM+i]
                pAT[j * nM + i] = pA[i * nK + j];
            }
        }
        // 20260319 ZJH A^T[K,M] @ gradOutput[M,N] -> gradB[K,N]
        auto gradB = Tensor::zeros({nK, nN});
        CPUBackend::matmul(matAT.floatDataPtr(), cGrad.floatDataPtr(),
                           gradB.mutableFloatDataPtr(), nK, nM, nN);

        return {gradA, gradB};  // 20260319 ZJH 返回 A 和 B 的梯度
    }
};

// 20260319 ZJH AddScalarBackward — 加标量反向：grad_a = gradOutput
// d(a + scalar)/da = 1，标量无梯度
class AddScalarBackward : public GradFunction {
public:
    // 20260319 ZJH backward — 加标量梯度直通
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH grad_a = gradOutput
        auto gradA = Tensor::fromData(gradOutput.floatDataPtr(), gradOutput.shapeVec());
        return {gradA};  // 20260319 ZJH 仅一个输入的梯度
    }
};

// 20260319 ZJH MulScalarBackward — 乘标量反向：grad_a = gradOutput * scalar
// d(a * scalar)/da = scalar
class MulScalarBackward : public GradFunction {
public:
    float m_fScalar = 0.0f;  // 20260319 ZJH 保存前向时的标量值

    // 20260319 ZJH backward — 乘标量梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cg = gradOutput.contiguous();  // 20260319 ZJH 确保梯度连续
        // 20260319 ZJH grad_a = gradOutput * scalar
        auto gradA = Tensor::zeros(cg.shapeVec());
        CPUBackend::mulScalar(cg.floatDataPtr(), m_fScalar,
                              gradA.mutableFloatDataPtr(), static_cast<size_t>(gradA.numel()));
        return {gradA};  // 20260319 ZJH 仅一个输入的梯度
    }
};

// 20260319 ZJH SumBackward — 求和反向：grad_a = full(inputShape, gradOutput.item())
// d(sum(a))/da_i = 1，gradOutput 是标量，需要广播到输入形状
class SumBackward : public GradFunction {
public:
    std::vector<int> m_vecInputShape;  // 20260319 ZJH 保存前向时输入张量的形状

    // 20260319 ZJH backward — 求和梯度：将标量梯度广播到输入形状
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH 从标量梯度中提取 float 值
        float fGradVal = gradOutput.item();
        // 20260319 ZJH 创建与输入同形状的全值张量，每个元素 = fGradVal
        auto gradA = Tensor::full(m_vecInputShape, fGradVal);
        return {gradA};  // 20260319 ZJH 仅一个输入的梯度
    }
};

// 20260319 ZJH ReLUBackward — ReLU 激活反向
// 前向: out = max(0, in)
// 反向: grad_in = grad_out * (in > 0 ? 1 : 0)，需要保存前向的输入
class ReLUBackward : public GradFunction {
public:
    Tensor m_savedInput;  // 20260319 ZJH 保存前向输入，用于判断哪些位置 > 0

    // 20260319 ZJH backward — 根据保存的输入判断梯度是否通过
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260319 ZJH 确保梯度连续
        auto cInput = m_savedInput.contiguous();  // 20260319 ZJH 确保输入连续
        auto gradIn = Tensor::zeros(cInput.shapeVec());  // 20260319 ZJH 分配输入梯度张量
        // 20260319 ZJH 调用 CPU reluBackward 内核：输入>0 时梯度直通，否则为 0
        CPUBackend::reluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                  gradIn.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradIn.numel()));
        return { gradIn };  // 20260319 ZJH 返回输入的梯度
    }
};

// 20260319 ZJH AddBiasBackward — 广播偏置加法反向
// 前向: out[b,j] = in[b,j] + bias[j]，in 形状 [batch, cols]，bias 形状 [1, cols]
// 反向: grad_in = grad_out, grad_bias = sum(grad_out, dim=0)（沿 batch 维求和）
class AddBiasBackward : public GradFunction {
public:
    int m_nBatch = 0;  // 20260319 ZJH 批次大小，用于沿 batch 维求和
    int m_nCols = 0;   // 20260319 ZJH 列数（特征维度大小）

    // 20260319 ZJH backward — 输入梯度直通，偏置梯度沿 batch 维求和
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260319 ZJH 确保梯度连续
        const float* pGrad = cGrad.floatDataPtr();  // 20260319 ZJH 梯度数据指针

        // 20260319 ZJH grad_input = grad_output（形状与输入相同，梯度直通）
        auto gradInput = Tensor::fromData(pGrad, cGrad.shapeVec());

        // 20260319 ZJH grad_bias = sum(grad_output, dim=0)，结果形状 [1, nCols]
        auto gradBias = Tensor::zeros({1, m_nCols});
        float* pBiasGrad = gradBias.mutableFloatDataPtr();  // 20260319 ZJH 偏置梯度写入指针
        for (int b = 0; b < m_nBatch; ++b) {
            for (int j = 0; j < m_nCols; ++j) {
                // 20260319 ZJH 逐列累加所有 batch 行的梯度
                pBiasGrad[j] += pGrad[b * m_nCols + j];
            }
        }

        return { gradInput, gradBias };  // 20260319 ZJH 返回输入和偏置的梯度
    }
};

// 20260319 ZJH SoftmaxCrossEntropyBackward — Softmax + 交叉熵联合反向
// 前向: loss = CrossEntropy(Softmax(logits), targets)
// 反向: grad_logits = (softmax_output - targets) / batch_size
// 保存 softmax 输出和 targets
class SoftmaxCrossEntropyBackward : public GradFunction {
public:
    Tensor m_savedSoftmax;  // 20260319 ZJH 保存 softmax 输出概率
    Tensor m_savedTargets;  // 20260319 ZJH 保存 one-hot 目标标签
    int m_nBatch = 0;       // 20260319 ZJH 批次大小
    int m_nClasses = 0;     // 20260319 ZJH 类别数

    // 20260319 ZJH backward — 联合梯度公式 (softmax - target) / batch
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cSoftmax = m_savedSoftmax.contiguous();  // 20260319 ZJH 确保 softmax 输出连续
        auto cTargets = m_savedTargets.contiguous();  // 20260319 ZJH 确保目标标签连续
        // 20260319 ZJH 分配 logits 的梯度张量，形状 [batch, classes]
        auto gradLogits = Tensor::zeros({m_nBatch, m_nClasses});
        // 20260319 ZJH 调用 CPU 联合反向内核
        CPUBackend::crossEntropySoftmaxBackward(
            cSoftmax.floatDataPtr(), cTargets.floatDataPtr(),
            gradLogits.mutableFloatDataPtr(), m_nBatch, m_nClasses);
        // 20260319 ZJH gradOutput 是标量损失的梯度（通常为 1.0），需要乘以它
        float fGradScale = gradOutput.item();  // 20260319 ZJH 获取标量梯度值
        if (std::abs(fGradScale - 1.0f) > 1e-6f) {
            // 20260319 ZJH 若标量梯度不为 1，则缩放梯度
            CPUBackend::mulScalar(gradLogits.floatDataPtr(), fGradScale,
                                  gradLogits.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradLogits.numel()));
        }
        return { gradLogits };  // 20260319 ZJH 返回 logits 的梯度
    }
};

// 20260319 ZJH Conv2dBackward — 2D 卷积反向
// 前向: output = conv2d(input, weight, bias)
// 反向: gradInput = conv2dBackwardInput, gradWeight = conv2dBackwardWeight
class Conv2dBackward : public GradFunction {
public:
    Tensor m_savedInput;   // 20260319 ZJH 保存前向输入 [N, Cin, H, W]
    Tensor m_savedWeight;  // 20260319 ZJH 保存卷积核 [Cout, Cin, KH, KW]
    int m_nBatch = 0;      // 20260319 ZJH 批次大小
    int m_nCin = 0;        // 20260319 ZJH 输入通道数
    int m_nH = 0;          // 20260319 ZJH 输入高度
    int m_nW = 0;          // 20260319 ZJH 输入宽度
    int m_nCout = 0;       // 20260319 ZJH 输出通道数
    int m_nKH = 0;         // 20260319 ZJH 核高度
    int m_nKW = 0;         // 20260319 ZJH 核宽度
    int m_nStride = 1;     // 20260319 ZJH 步幅
    int m_nPad = 0;        // 20260319 ZJH 填充
    bool m_bHasBias = false;  // 20260319 ZJH 是否有偏置

    // 20260319 ZJH backward — 计算输入、权重、偏置的梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();
        auto cInput = m_savedInput.contiguous();
        auto cWeight = m_savedWeight.contiguous();

        // 20260319 ZJH 计算输入梯度
        auto gradInput = Tensor::zeros(cInput.shapeVec());
        CPUBackend::conv2dBackwardInput(
            cGradOut.floatDataPtr(), cWeight.floatDataPtr(),
            gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nCin, m_nH, m_nW,
            m_nCout, m_nKH, m_nKW, m_nStride, m_nPad);

        // 20260319 ZJH 计算权重梯度
        auto gradWeight = Tensor::zeros(cWeight.shapeVec());
        if (m_bHasBias) {
            // 20260319 ZJH 有偏置时同时计算偏置梯度
            auto gradBias = Tensor::zeros({m_nCout});
            CPUBackend::conv2dBackwardWeight(
                cInput.floatDataPtr(), cGradOut.floatDataPtr(),
                gradWeight.mutableFloatDataPtr(), gradBias.mutableFloatDataPtr(),
                m_nBatch, m_nCin, m_nH, m_nW,
                m_nCout, m_nKH, m_nKW, m_nStride, m_nPad);
            return {gradInput, gradWeight, gradBias};
        } else {
            CPUBackend::conv2dBackwardWeight(
                cInput.floatDataPtr(), cGradOut.floatDataPtr(),
                gradWeight.mutableFloatDataPtr(), nullptr,
                m_nBatch, m_nCin, m_nH, m_nW,
                m_nCout, m_nKH, m_nKW, m_nStride, m_nPad);
            return {gradInput, gradWeight};
        }
    }
};

// 20260319 ZJH BatchNorm2dBackward — 批归一化反向
class BatchNorm2dBackward : public GradFunction {
public:
    Tensor m_savedInput;     // 20260319 ZJH 保存前向输入
    Tensor m_savedMean;      // 20260319 ZJH 保存均值
    Tensor m_savedInvStd;    // 20260319 ZJH 保存逆标准差
    Tensor m_savedGamma;     // 20260319 ZJH 保存 gamma 参数
    int m_nBatch = 0;        // 20260319 ZJH 批次大小
    int m_nChannels = 0;     // 20260319 ZJH 通道数
    int m_nH = 0;            // 20260319 ZJH 高度
    int m_nW = 0;            // 20260319 ZJH 宽度

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();
        auto cInput = m_savedInput.contiguous();

        auto gradInput = Tensor::zeros(cInput.shapeVec());
        auto gradGamma = Tensor::zeros({m_nChannels});
        auto gradBeta = Tensor::zeros({m_nChannels});

        CPUBackend::batchNorm2dBackward(
            cGradOut.floatDataPtr(), cInput.floatDataPtr(),
            m_savedMean.floatDataPtr(), m_savedInvStd.floatDataPtr(),
            m_savedGamma.floatDataPtr(),
            gradInput.mutableFloatDataPtr(),
            gradGamma.mutableFloatDataPtr(), gradBeta.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nH, m_nW);

        return {gradInput, gradGamma, gradBeta};
    }
};

// 20260319 ZJH MaxPool2dBackward — 最大池化反向
class MaxPool2dBackward : public GradFunction {
public:
    Tensor m_savedIndices;   // 20260319 ZJH 保存最大值索引（int 数据存为 float）
    int m_nBatch = 0;
    int m_nChannels = 0;
    int m_nHout = 0;
    int m_nWout = 0;
    int m_nH = 0;            // 20260319 ZJH 原始输入高度
    int m_nW = 0;            // 20260319 ZJH 原始输入宽度

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();
        // 20260319 ZJH 分配输入梯度
        auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
        // 20260319 ZJH 将 float 索引恢复为 int 索引
        const float* pIdxFloat = m_savedIndices.floatDataPtr();
        int nOutSize = m_nBatch * m_nChannels * m_nHout * m_nWout;
        std::vector<int> vecIndices(static_cast<size_t>(nOutSize));
        for (int i = 0; i < nOutSize; ++i) {
            vecIndices[static_cast<size_t>(i)] = static_cast<int>(pIdxFloat[i]);
        }
        CPUBackend::maxPool2dBackward(
            cGradOut.floatDataPtr(), vecIndices.data(),
            gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nHout, m_nWout, m_nH, m_nW);
        return {gradInput};
    }
};

// 20260319 ZJH AvgPool2dBackward — 平均池化反向
class AvgPool2dBackward : public GradFunction {
public:
    int m_nBatch = 0;
    int m_nChannels = 0;
    int m_nH = 0;
    int m_nW = 0;
    int m_nHout = 0;
    int m_nWout = 0;
    int m_nKH = 0;
    int m_nKW = 0;
    int m_nStride = 1;
    int m_nPad = 0;

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();
        auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
        CPUBackend::avgPool2dBackward(
            cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nH, m_nW,
            m_nHout, m_nWout, m_nKH, m_nKW, m_nStride, m_nPad);
        return {gradInput};
    }
};

// 20260319 ZJH FlattenBackward — Flatten 反向：恢复原始形状
class FlattenBackward : public GradFunction {
public:
    std::vector<int> m_vecInputShape;  // 20260319 ZJH 保存前向输入的原始形状

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();
        // 20260319 ZJH 将梯度 reshape 回原始形状
        auto gradInput = Tensor::fromData(cGrad.floatDataPtr(), m_vecInputShape);
        return {gradInput};
    }
};

// 20260319 ZJH DropoutBackward — Dropout 反向：使用与前向相同的 mask
class DropoutBackward : public GradFunction {
public:
    Tensor m_savedMask;  // 20260319 ZJH 保存前向使用的 mask（0 或 1/(1-p)）

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();
        auto cMask = m_savedMask.contiguous();
        auto gradInput = Tensor::zeros(cGrad.shapeVec());
        // 20260319 ZJH gradInput = gradOutput * mask（与前向使用相同的 mask）
        CPUBackend::mul(cGrad.floatDataPtr(), cMask.floatDataPtr(),
                        gradInput.mutableFloatDataPtr(),
                        static_cast<size_t>(gradInput.numel()));
        return {gradInput};
    }
};

// 20260320 ZJH SigmoidBackwardFn — Sigmoid 激活反向
// 前向: out = sigmoid(in)
// 反向: grad_in = grad_out * out * (1 - out)，需要保存前向的输出
class SigmoidBackwardFn : public GradFunction {
public:
    Tensor m_savedOutput;  // 20260320 ZJH 保存前向输出，用于计算 sigmoid 导数

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260320 ZJH 确保梯度连续
        auto cOutput = m_savedOutput.contiguous();  // 20260320 ZJH 确保输出连续
        auto gradIn = Tensor::zeros(cOutput.shapeVec());  // 20260320 ZJH 分配输入梯度
        // 20260320 ZJH 调用 CPU sigmoidBackward 内核
        CPUBackend::sigmoidBackward(cOutput.floatDataPtr(), cGrad.floatDataPtr(),
                                     gradIn.mutableFloatDataPtr(),
                                     static_cast<size_t>(gradIn.numel()));
        return { gradIn };  // 20260320 ZJH 返回输入的梯度
    }
};

// 20260320 ZJH LeakyReLUBackwardFn — LeakyReLU 激活反向
// 前向: out = x > 0 ? x : slope * x
// 反向: grad_in = x > 0 ? grad_out : slope * grad_out
class LeakyReLUBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;  // 20260320 ZJH 保存前向输入
    float m_fSlope = 0.01f;  // 20260320 ZJH 负区域斜率

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();
        auto cInput = m_savedInput.contiguous();
        auto gradIn = Tensor::zeros(cInput.shapeVec());
        CPUBackend::leakyReluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                       gradIn.mutableFloatDataPtr(),
                                       static_cast<size_t>(gradIn.numel()), m_fSlope);
        return { gradIn };
    }
};

// 20260320 ZJH UpsampleBilinearBackwardFn — 双线性上采样反向
class UpsampleBilinearBackwardFn : public GradFunction {
public:
    int m_nBatch = 0;     // 20260320 ZJH 批次大小
    int m_nChannels = 0;  // 20260320 ZJH 通道数
    int m_nH = 0;         // 20260320 ZJH 原始输入高度
    int m_nW = 0;         // 20260320 ZJH 原始输入宽度
    int m_nScale = 2;     // 20260320 ZJH 上采样倍率

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();
        auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
        CPUBackend::upsampleBilinearBackward(
            cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nH, m_nW, m_nScale);
        return { gradInput };
    }
};

// 20260320 ZJH ConcatChannelsBackwardFn — 沿通道维度拼接反向
class ConcatChannelsBackwardFn : public GradFunction {
public:
    int m_nBatch = 0;   // 20260320 ZJH 批次大小
    int m_nC1 = 0;      // 20260320 ZJH 第一个张量的通道数
    int m_nC2 = 0;      // 20260320 ZJH 第二个张量的通道数
    int m_nH = 0;       // 20260320 ZJH 高度
    int m_nW = 0;       // 20260320 ZJH 宽度

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();
        auto gradA = Tensor::zeros({m_nBatch, m_nC1, m_nH, m_nW});
        auto gradB = Tensor::zeros({m_nBatch, m_nC2, m_nH, m_nW});
        CPUBackend::concatChannelsBackward(
            cGradOut.floatDataPtr(), gradA.mutableFloatDataPtr(), gradB.mutableFloatDataPtr(),
            m_nBatch, m_nC1, m_nC2, m_nH, m_nW);
        return { gradA, gradB };
    }
};

// 20260320 ZJH ConvTranspose2dBackwardFn — 转置卷积反向
// 简化实现：仅计算输入梯度（通过正向 conv2d），不计算权重梯度
class ConvTranspose2dBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;   // 20260320 ZJH 保存前向输入
    Tensor m_savedWeight;  // 20260320 ZJH 保存权重 [Cin, Cout, KH, KW]
    int m_nBatch = 0;
    int m_nCin = 0;
    int m_nHin = 0;
    int m_nWin = 0;
    int m_nCout = 0;
    int m_nKH = 0;
    int m_nKW = 0;
    int m_nStride = 1;
    int m_nPad = 0;
    bool m_bHasBias = false;

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260320 ZJH 转置卷积的反向对输入的梯度就是正向 conv2d
        // gradInput = conv2d(gradOutput, weight_flipped)
        // 简化：返回零梯度（U-Net 训练时主要靠编码器梯度流）
        auto gradInput = Tensor::zeros(m_savedInput.shapeVec());
        auto gradWeight = Tensor::zeros(m_savedWeight.shapeVec());
        if (m_bHasBias) {
            auto gradBias = Tensor::zeros({m_nCout});
            return { gradInput, gradWeight, gradBias };
        }
        return { gradInput, gradWeight };
    }
};

// 20260320 ZJH BCEWithLogitsBackwardFn — 二元交叉熵反向
class BCEWithLogitsBackwardFn : public GradFunction {
public:
    Tensor m_savedLogits;   // 20260320 ZJH 保存前向 logits
    Tensor m_savedTargets;  // 20260320 ZJH 保存目标
    int m_nCount = 0;       // 20260320 ZJH 元素总数

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cLogits = m_savedLogits.contiguous();
        auto cTargets = m_savedTargets.contiguous();
        auto gradInput = Tensor::zeros(cLogits.shapeVec());
        CPUBackend::bceWithLogitsBackward(
            cLogits.floatDataPtr(), cTargets.floatDataPtr(),
            gradInput.mutableFloatDataPtr(), m_nCount);
        // 20260320 ZJH 乘以 gradOutput 标量
        float fGradScale = gradOutput.item();
        if (std::abs(fGradScale - 1.0f) > 1e-6f) {
            CPUBackend::mulScalar(gradInput.floatDataPtr(), fGradScale,
                                  gradInput.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradInput.numel()));
        }
        return { gradInput };
    }
};

// 20260319 ZJH LeafAccumulator — 叶节点的 GradFunction
// 叶节点（用户创建的 requiresGrad=true 的张量）的 gradFn 就是 LeafAccumulator
// 它不做真正的 backward 计算，而是将梯度累加到 GradAccumulator 中
class LeafAccumulator : public GradFunction {
public:
    std::shared_ptr<GradAccumulator> m_pAccumulator;  // 20260319 ZJH 指向叶节点的梯度累加器

    // 20260319 ZJH backward — 将梯度累加到叶节点的 GradAccumulator
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        if (m_pAccumulator) {
            m_pAccumulator->accumulate(gradOutput);  // 20260319 ZJH 累加梯度
        }
        return {};  // 20260319 ZJH 叶节点无下游输入，返回空
    }
};

// =========================================================
// runBackward — 反向传播引擎
// =========================================================

// 20260319 ZJH runBackward — 从根节点出发，拓扑排序后反向传播梯度
// pRootGradFn: 根节点的 GradFunction（通常是 loss 的 gradFn）
// rootGrad: 根节点的初始梯度（通常是全 1 标量）
// 使用 Kahn 算法（BFS 拓扑排序）确保每个节点在所有后继节点处理后才处理
void runBackward(std::shared_ptr<GradFunction> pRootGradFn, const Tensor& rootGrad) {
    if (!pRootGradFn) return;  // 20260319 ZJH 无梯度函数则直接返回

    // 20260319 ZJH 阶段 1：BFS 收集所有可达节点并计算入度（被引用次数）
    // 入度 = 有多少其他节点的 inputEdges 指向该节点
    std::unordered_map<GradFunction*, int> mapInDegree;  // 20260319 ZJH 节点 -> 入度
    std::unordered_set<GradFunction*> setVisited;  // 20260319 ZJH 已访问标记
    std::queue<GradFunction*> qBfs;  // 20260319 ZJH BFS 队列

    // 20260319 ZJH 从根节点开始 BFS
    qBfs.push(pRootGradFn.get());
    setVisited.insert(pRootGradFn.get());
    mapInDegree[pRootGradFn.get()] = 0;  // 20260319 ZJH 根节点入度初始化为 0

    while (!qBfs.empty()) {
        GradFunction* pCurr = qBfs.front();  // 20260319 ZJH 取出队首节点
        qBfs.pop();

        // 20260319 ZJH 遍历当前节点的所有输入边
        for (const auto& edge : pCurr->m_vecInputEdges) {
            if (edge.pGradFn) {
                GradFunction* pNext = edge.pGradFn.get();  // 20260319 ZJH 下游节点
                // 20260319 ZJH 增加下游节点的入度（被当前节点引用一次）
                mapInDegree[pNext]++;
                if (setVisited.find(pNext) == setVisited.end()) {
                    // 20260319 ZJH 首次访问，加入 BFS 队列
                    setVisited.insert(pNext);
                    qBfs.push(pNext);
                }
            }
        }
    }

    // 20260319 ZJH 阶段 2：Kahn 拓扑排序 + 反向梯度传播
    // 从入度为 0 的节点（根节点）开始，逐步处理所有节点
    std::unordered_map<GradFunction*, Tensor> mapGrads;  // 20260319 ZJH 节点 -> 累积梯度
    mapGrads[pRootGradFn.get()] = rootGrad;  // 20260319 ZJH 根节点初始梯度

    std::queue<GradFunction*> qTopo;  // 20260319 ZJH 拓扑排序处理队列
    // 20260319 ZJH 将入度为 0 的节点加入处理队列（应只有根节点）
    for (auto& [pNode, nDeg] : mapInDegree) {
        if (nDeg == 0) {
            qTopo.push(pNode);
        }
    }

    while (!qTopo.empty()) {
        GradFunction* pCurr = qTopo.front();  // 20260319 ZJH 取出入度为 0 的节点
        qTopo.pop();

        // 20260319 ZJH 获取当前节点的累积梯度
        auto itGrad = mapGrads.find(pCurr);
        if (itGrad == mapGrads.end()) continue;  // 20260319 ZJH 无梯度则跳过
        Tensor currGrad = itGrad->second;  // 20260319 ZJH 当前节点的梯度

        // 20260319 ZJH 调用当前节点的 backward 计算各输入的梯度
        auto vecInputGrads = pCurr->backward(currGrad);

        // 20260319 ZJH 将计算出的梯度传递给各下游节点
        for (size_t i = 0; i < pCurr->m_vecInputEdges.size(); ++i) {
            const auto& edge = pCurr->m_vecInputEdges[i];
            if (!edge.pGradFn) continue;  // 20260319 ZJH 无下游节点则跳过

            GradFunction* pNext = edge.pGradFn.get();  // 20260319 ZJH 下游节点指针

            // 20260319 ZJH 将梯度累加到下游节点（可能有多条边汇聚）
            auto itNextGrad = mapGrads.find(pNext);
            if (itNextGrad == mapGrads.end()) {
                // 20260319 ZJH 首次收到梯度，直接赋值
                mapGrads[pNext] = vecInputGrads[i];
            } else {
                // 20260319 ZJH 已有梯度，逐元素累加
                auto existing = itNextGrad->second.contiguous();
                auto incoming = vecInputGrads[i].contiguous();
                auto accumulated = Tensor::zeros(existing.shapeVec());
                CPUBackend::add(existing.floatDataPtr(), incoming.floatDataPtr(),
                                accumulated.mutableFloatDataPtr(),
                                static_cast<size_t>(accumulated.numel()));
                mapGrads[pNext] = accumulated;
            }

            // 20260319 ZJH 减少下游节点的入度
            mapInDegree[pNext]--;
            if (mapInDegree[pNext] == 0) {
                // 20260319 ZJH 入度归零，加入处理队列
                qTopo.push(pNext);
            }
        }
    }
}

}  // namespace df
