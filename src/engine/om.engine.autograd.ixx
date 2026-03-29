// 20260319 ZJH AutoGrad 自动微分模块 — Phase 1C
// 动态计算图：前向运算构建 GradFunction DAG，backward() 拓扑排序后反向传播梯度
// 循环依赖解决方案：Tensor 以 shared_ptr<void> 类型擦除存储 GradFunction，
// 仅 tensor_ops 同时了解 Tensor 和 GradFunction
// 20260325 ZJH GPU 调度重写：反向传播根据张量设备自动调度 CUDABackend / CPUBackend，
//              消除旧版强制 .cpu() 导致的 GPU 利用率低下问题
module;

#include <vector>
#include <memory>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <cstring>

#include "om_types.h"

export module om.engine.autograd;

// 20260319 ZJH 导入依赖模块：存储层、张量类、CPU 计算内核
import om.engine.tensor_storage;
import om.engine.tensor;
import om.hal.cpu_backend;
// 20260325 ZJH 导入 CUDA 后端（条件编译），用于反向传播 GPU 调度
#ifdef OM_HAS_CUDA
import om.hal.cuda_backend;
#endif

export namespace om {

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

    // 20260327 ZJH releaseSavedTensors — 反向传播完成后释放保存的中间张量
    // 前向传播保存的激活值（特征图、权重副本等）占大量 GPU 显存
    // 反向计算完毕后这些张量不再需要，立即释放可降低峰值显存 30-50%
    // 基类默认空实现，子类覆盖以释放各自的 m_savedXxx 成员
    virtual void releaseSavedTensors() {}

    // 20260319 ZJH 输入边列表：记录当前运算的各个输入张量来自哪个上游 GradFunction
    std::vector<Edge> m_vecInputEdges;
};

// 20260319 ZJH GradAccumulator — 叶节点梯度累加器
// 持有一个梯度张量 m_grad，支持累加和清零操作
// 叶节点（用户创建的 requiresGrad=true 的张量）通过 LeafAccumulator 持有此对象
// 20260325 ZJH GPU 调度重写：梯度累加在张量所在设备上进行，不强制迁移到 CPU
struct GradAccumulator {
    Tensor m_grad;  // 20260319 ZJH 累积的梯度张量
    bool m_bHasGrad = false;  // 20260319 ZJH 是否已有梯度（区分零梯度和无梯度）

    // 20260319 ZJH accumulate — 累加梯度，首次调用时直接赋值，后续调用逐元素相加
    // 20260325 ZJH GPU 调度重写：在张量所在设备上操作，GPU 用 CUDABackend，CPU 用 CPUBackend
    void accumulate(const Tensor& grad) {
        if (!m_bHasGrad) {
            // 20260319 ZJH 首次累加：深拷贝梯度（避免共享存储导致覆盖）
            auto cg = grad.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
            if (cg.isCuda()) {
                // 20260325 ZJH GPU 路径：在 GPU 上深拷贝
                m_grad = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
                CUDABackend::copy(cg.floatDataPtr(), m_grad.mutableFloatDataPtr(),
                                  static_cast<size_t>(cg.numel()));
            } else
#endif
            {
                // 20260325 ZJH CPU 路径：fromData 深拷贝
                m_grad = Tensor::fromData(cg.floatDataPtr(), cg.shapeVec());
            }
            m_bHasGrad = true;  // 20260319 ZJH 标记已有梯度
        } else {
            // 20260319 ZJH 后续累加：逐元素相加
            auto cg = grad.contiguous();  // 20260325 ZJH 确保连续
            auto cm = m_grad.contiguous();  // 20260325 ZJH 确保已有梯度连续
#ifdef OM_HAS_CUDA
            if (cm.isCuda()) {
                // 20260325 ZJH GPU 路径：在 GPU 上累加
                auto gpuGrad = cg.isCuda() ? cg : cg.cuda();  // 20260325 ZJH 确保 incoming 也在 GPU
                auto result = Tensor::zeros(cm.shapeVec(), DeviceType::CUDA);
                CUDABackend::add(cm.floatDataPtr(), gpuGrad.floatDataPtr(),
                                 result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
                m_grad = result;
            } else
#endif
            {
                // 20260325 ZJH CPU 路径
                auto cpuGrad = cg.isCuda() ? cg.cpu() : cg;  // 20260325 ZJH 安全回退
                auto result = Tensor::zeros(cm.shapeVec());
                CPUBackend::add(cm.floatDataPtr(), cpuGrad.floatDataPtr(),
                                result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
                m_grad = result;
            }
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
// 20260325 ZJH GPU 调度重写：每个 backward 根据 gradOutput 设备自动选择后端
// =========================================================

// 20260319 ZJH AddBackward — 加法反向：grad_a = gradOutput, grad_b = gradOutput
// d(a+b)/da = 1, d(a+b)/db = 1
class AddBackward : public GradFunction {
public:
    // 20260319 ZJH backward — 加法梯度直通，两个输入梯度均等于输出梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cg = gradOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cg.isCuda()) {
            // 20260325 ZJH GPU 路径：在 GPU 上深拷贝梯度
            auto gradA = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::copy(cg.floatDataPtr(), gradA.mutableFloatDataPtr(),
                              static_cast<size_t>(cg.numel()));
            auto gradB = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::copy(cg.floatDataPtr(), gradB.mutableFloatDataPtr(),
                              static_cast<size_t>(cg.numel()));
            return {gradA, gradB};
        }
#endif
        // 20260319 ZJH CPU 路径：grad_a = gradOutput（深拷贝避免共享存储问题）
        auto gradA = Tensor::fromData(cg.floatDataPtr(), cg.shapeVec());
        auto gradB = Tensor::fromData(cg.floatDataPtr(), cg.shapeVec());
        return {gradA, gradB};  // 20260319 ZJH 返回两个输入的梯度
    }
};

// 20260319 ZJH SubBackward — 减法反向：grad_a = gradOutput, grad_b = -gradOutput
// d(a-b)/da = 1, d(a-b)/db = -1
class SubBackward : public GradFunction {
public:
    // 20260319 ZJH backward — 减法梯度：a 直通，b 取负
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cg = gradOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cg.isCuda()) {
            // 20260325 ZJH GPU 路径
            auto gradA = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::copy(cg.floatDataPtr(), gradA.mutableFloatDataPtr(),
                              static_cast<size_t>(cg.numel()));
            auto gradB = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::mulScalar(cg.floatDataPtr(), -1.0f,
                                   gradB.mutableFloatDataPtr(), static_cast<size_t>(gradB.numel()));
            return {gradA, gradB};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradA = Tensor::fromData(cg.floatDataPtr(), cg.shapeVec());
        auto gradB = Tensor::zeros(cg.shapeVec());
        CPUBackend::mulScalar(cg.floatDataPtr(), -1.0f,
                              gradB.mutableFloatDataPtr(), static_cast<size_t>(gradB.numel()));
        return {gradA, gradB};
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
        auto cg = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto ca = m_savedA.contiguous();  // 20260325 ZJH 确保保存的张量连续
        auto cb = m_savedB.contiguous();  // 20260325 ZJH 确保保存的张量连续
#ifdef OM_HAS_CUDA
        if (cg.isCuda()) {
            // 20260325 ZJH GPU 路径：在 GPU 上计算交叉乘积
            auto gradA = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::mul(cg.floatDataPtr(), cb.floatDataPtr(),
                             gradA.mutableFloatDataPtr(), static_cast<size_t>(gradA.numel()));
            auto gradB = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::mul(cg.floatDataPtr(), ca.floatDataPtr(),
                             gradB.mutableFloatDataPtr(), static_cast<size_t>(gradB.numel()));
            return {gradA, gradB};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradA = Tensor::zeros(cg.shapeVec());
        CPUBackend::mul(cg.floatDataPtr(), cb.floatDataPtr(),
                        gradA.mutableFloatDataPtr(), static_cast<size_t>(gradA.numel()));
        auto gradB = Tensor::zeros(cg.shapeVec());
        CPUBackend::mul(cg.floatDataPtr(), ca.floatDataPtr(),
                        gradB.mutableFloatDataPtr(), static_cast<size_t>(gradB.numel()));
        return {gradA, gradB};
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
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cA = m_savedA.contiguous();  // 20260325 ZJH 确保连续
        auto cB = m_savedB.contiguous();  // 20260325 ZJH 确保连续

        int nM = cA.shape(0);  // 20260319 ZJH 矩阵 A 的行数
        int nK = cA.shape(1);  // 20260319 ZJH 矩阵 A 的列数 = B 的行数
        int nN = cB.shape(1);  // 20260319 ZJH 矩阵 B 的列数

#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：使用 CUDABackend 的 transpose + matmul
            // grad_A = gradOutput[M,N] @ B^T[N,K] -> [M,K]
            auto matBT = Tensor::zeros({nN, nK}, DeviceType::CUDA);
            CUDABackend::transpose(cB.floatDataPtr(), matBT.mutableFloatDataPtr(), nK, nN);
            auto gradA = Tensor::zeros({nM, nK}, DeviceType::CUDA);
            CUDABackend::matmul(cGrad.floatDataPtr(), matBT.floatDataPtr(),
                                gradA.mutableFloatDataPtr(), nM, nN, nK);

            // grad_B = A^T[K,M] @ gradOutput[M,N] -> [K,N]
            auto matAT = Tensor::zeros({nK, nM}, DeviceType::CUDA);
            CUDABackend::transpose(cA.floatDataPtr(), matAT.mutableFloatDataPtr(), nM, nK);
            auto gradB = Tensor::zeros({nK, nN}, DeviceType::CUDA);
            CUDABackend::matmul(matAT.floatDataPtr(), cGrad.floatDataPtr(),
                                gradB.mutableFloatDataPtr(), nK, nM, nN);

            return {gradA, gradB};
        }
#endif
        // 20260319 ZJH CPU 路径：手动转置 + CPUBackend::matmul
        // grad_A = gradOutput[M,N] @ B^T[N,K]
        auto matBT = Tensor::zeros({nN, nK});
        const float* pB = cB.floatDataPtr();
        float* pBT = matBT.mutableFloatDataPtr();
        for (int i = 0; i < nK; ++i) {
            for (int j = 0; j < nN; ++j) {
                pBT[j * nK + i] = pB[i * nN + j];
            }
        }
        auto gradA = Tensor::zeros({nM, nK});
        CPUBackend::matmul(cGrad.floatDataPtr(), matBT.floatDataPtr(),
                           gradA.mutableFloatDataPtr(), nM, nN, nK);

        // grad_B = A^T[K,M] @ gradOutput[M,N]
        auto matAT = Tensor::zeros({nK, nM});
        const float* pA = cA.floatDataPtr();
        float* pAT = matAT.mutableFloatDataPtr();
        for (int i = 0; i < nM; ++i) {
            for (int j = 0; j < nK; ++j) {
                pAT[j * nM + i] = pA[i * nK + j];
            }
        }
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
        auto cg = gradOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cg.isCuda()) {
            // 20260325 ZJH GPU 路径：在 GPU 上深拷贝
            auto gradA = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::copy(cg.floatDataPtr(), gradA.mutableFloatDataPtr(),
                              static_cast<size_t>(cg.numel()));
            return {gradA};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradA = Tensor::fromData(cg.floatDataPtr(), cg.shapeVec());
        return {gradA};
    }
};

// 20260319 ZJH MulScalarBackward — 乘标量反向：grad_a = gradOutput * scalar
// d(a * scalar)/da = scalar
class MulScalarBackward : public GradFunction {
public:
    float m_fScalar = 0.0f;  // 20260319 ZJH 保存前向时的标量值

    // 20260319 ZJH backward — 乘标量梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cg = gradOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cg.isCuda()) {
            // 20260325 ZJH GPU 路径
            auto gradA = Tensor::zeros(cg.shapeVec(), DeviceType::CUDA);
            CUDABackend::mulScalar(cg.floatDataPtr(), m_fScalar,
                                   gradA.mutableFloatDataPtr(), static_cast<size_t>(gradA.numel()));
            return {gradA};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradA = Tensor::zeros(cg.shapeVec());
        CPUBackend::mulScalar(cg.floatDataPtr(), m_fScalar,
                              gradA.mutableFloatDataPtr(), static_cast<size_t>(gradA.numel()));
        return {gradA};
    }
};

// 20260319 ZJH SumBackward — 求和反向：grad_a = full(inputShape, gradOutput.item())
// d(sum(a))/da_i = 1，gradOutput 是标量，需要广播到输入形状
class SumBackward : public GradFunction {
public:
    std::vector<int> m_vecInputShape;  // 20260319 ZJH 保存前向时输入张量的形状

    // 20260319 ZJH backward — 求和梯度：将标量梯度广播到输入形状
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH 从标量梯度中提取 float 值（item() 已有 GPU 安全保护）
        float fGradVal = gradOutput.item();
        // 20260325 ZJH 在 gradOutput 所在设备上创建广播张量
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            auto gradA = Tensor::zeros(m_vecInputShape, DeviceType::CUDA);
            CUDABackend::fillValue(gradA.mutableFloatDataPtr(),
                                   static_cast<size_t>(gradA.numel()), fGradVal);
            return {gradA};
        }
#endif
        // 20260319 ZJH CPU 路径：创建与输入同形状的全值张量
        auto gradA = Tensor::full(m_vecInputShape, fGradVal);
        return {gradA};
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
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cInput = m_savedInput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend::reluBackward(pInput, pGradOut, pGradIn, n)
            auto gradIn = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            CUDABackend::reluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                      gradIn.mutableFloatDataPtr(),
                                      static_cast<size_t>(gradIn.numel()));
            return { gradIn };
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradIn = Tensor::zeros(cInput.shapeVec());
        CPUBackend::reluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                  gradIn.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradIn.numel()));
        return { gradIn };
    }
};

// 20260319 ZJH AddBiasBackward — 广播偏置加法反向
// 前向: out[b,j] = in[b,j] + bias[j]，in 形状 [batch, cols]，bias 形状 [1, cols]
// 反向: grad_in = grad_out, grad_bias = sum(grad_out, dim=0)（沿 batch 维求和）
// 20260326 ZJH 已全 GPU 化：CUDABackend::addBiasBackward 使用 atomicAdd 归约偏置梯度
class AddBiasBackward : public GradFunction {
public:
    int m_nBatch = 0;  // 20260319 ZJH 批次大小，用于沿 batch 维求和
    int m_nCols = 0;   // 20260319 ZJH 列数（特征维度大小）

    // 20260319 ZJH backward — 输入梯度直通，偏置梯度沿 batch 维求和
    // 20260326 ZJH 全 GPU 化：CUDABackend::addBiasBackward 通道维 reduction
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            auto cGrad = gradOutput.contiguous();
            // 20260326 ZJH grad_input = grad_output（GPU 上 copy）
            auto gradInput = Tensor::zeros(cGrad.shapeVec(), DeviceType::CUDA);
            CUDABackend::copy(cGrad.floatDataPtr(), gradInput.mutableFloatDataPtr(),
                              static_cast<size_t>(cGrad.numel()));
            // 20260326 ZJH grad_bias GPU reduction
            auto gradBias = Tensor::zeros({1, m_nCols}, DeviceType::CUDA);
            CUDABackend::addBiasBackward(cGrad.floatDataPtr(), gradBias.mutableFloatDataPtr(),
                                          m_nBatch, m_nCols, 1);
            return { gradInput, gradBias };
        }
#endif
        // 20260319 ZJH CPU 路径
        auto cGrad = gradOutput.contiguous();
        const float* pGrad = cGrad.floatDataPtr();
        auto gradInput = Tensor::fromData(pGrad, cGrad.shapeVec());
        auto gradBias = Tensor::zeros({1, m_nCols});
        float* pBiasGrad = gradBias.mutableFloatDataPtr();
        for (int b = 0; b < m_nBatch; ++b) {
            for (int j = 0; j < m_nCols; ++j) {
                pBiasGrad[j] += pGrad[b * m_nCols + j];
            }
        }
        return { gradInput, gradBias };
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

    // 20260327 ZJH 释放 softmax 输出和目标
    void releaseSavedTensors() override { m_savedSoftmax = Tensor(); m_savedTargets = Tensor(); }

    // 20260319 ZJH backward — 联合梯度公式 (softmax - target) / batch
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cSoftmax = m_savedSoftmax.contiguous();  // 20260325 ZJH 确保连续
        auto cTargets = m_savedTargets.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cSoftmax.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend::crossEntropySoftmaxBackward
            auto gradLogits = Tensor::zeros({m_nBatch, m_nClasses}, DeviceType::CUDA);
            CUDABackend::crossEntropySoftmaxBackward(
                cSoftmax.floatDataPtr(), cTargets.floatDataPtr(),
                gradLogits.mutableFloatDataPtr(), m_nBatch, m_nClasses);
            // 20260319 ZJH gradOutput 是标量损失的梯度（通常为 1.0）
            float fGradScale = gradOutput.item();
            if (std::abs(fGradScale - 1.0f) > 1e-6f) {
                // 20260325 ZJH 在 GPU 上缩放
                auto scaled = Tensor::zeros({m_nBatch, m_nClasses}, DeviceType::CUDA);
                CUDABackend::mulScalar(gradLogits.floatDataPtr(), fGradScale,
                                       scaled.mutableFloatDataPtr(),
                                       static_cast<size_t>(scaled.numel()));
                return { scaled };
            }
            return { gradLogits };
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradLogits = Tensor::zeros({m_nBatch, m_nClasses});
        CPUBackend::crossEntropySoftmaxBackward(
            cSoftmax.floatDataPtr(), cTargets.floatDataPtr(),
            gradLogits.mutableFloatDataPtr(), m_nBatch, m_nClasses);
        float fGradScale = gradOutput.item();
        if (std::abs(fGradScale - 1.0f) > 1e-6f) {
            CPUBackend::mulScalar(gradLogits.floatDataPtr(), fGradScale,
                                  gradLogits.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradLogits.numel()));
        }
        return { gradLogits };
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

    // 20260327 ZJH 释放保存的输入和权重张量（Conv2d 是最大的显存消费者）
    void releaseSavedTensors() override {
        m_savedInput = Tensor();   // 20260327 ZJH 释放前向输入
        m_savedWeight = Tensor();  // 20260327 ZJH 释放卷积核副本
    }

    // 20260319 ZJH backward — 计算输入、权重、偏置的梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cInput = m_savedInput.contiguous();
        auto cWeight = m_savedWeight.contiguous();

#ifdef OM_HAS_CUDA
        if (cGradOut.isCuda()) {
            // 20260325 ZJH GPU 路径：完整 im2col+GEMM 反向
            auto gradInput = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            CUDABackend::conv2dBackwardInput(
                cGradOut.floatDataPtr(), cWeight.floatDataPtr(),
                gradInput.mutableFloatDataPtr(),
                m_nBatch, m_nCin, m_nH, m_nW,
                m_nCout, m_nKH, m_nKW, m_nPad, m_nPad, m_nStride, m_nStride);

            auto gradWeight = Tensor::zeros(cWeight.shapeVec(), DeviceType::CUDA);
            if (m_bHasBias) {
                auto gradBias = Tensor::zeros({m_nCout}, DeviceType::CUDA);
                CUDABackend::conv2dBackwardWeight(
                    cInput.floatDataPtr(), cGradOut.floatDataPtr(),
                    gradWeight.mutableFloatDataPtr(), gradBias.mutableFloatDataPtr(),
                    m_nBatch, m_nCin, m_nH, m_nW,
                    m_nCout, m_nKH, m_nKW, m_nPad, m_nPad, m_nStride, m_nStride);
                return {gradInput, gradWeight, gradBias};
            } else {
                CUDABackend::conv2dBackwardWeight(
                    cInput.floatDataPtr(), cGradOut.floatDataPtr(),
                    gradWeight.mutableFloatDataPtr(), nullptr,
                    m_nBatch, m_nCin, m_nH, m_nW,
                    m_nCout, m_nKH, m_nKW, m_nPad, m_nPad, m_nStride, m_nStride);
                return {gradInput, gradWeight};
            }
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradInput = Tensor::zeros(cInput.shapeVec());
        CPUBackend::conv2dBackwardInput(
            cGradOut.floatDataPtr(), cWeight.floatDataPtr(),
            gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nCin, m_nH, m_nW,
            m_nCout, m_nKH, m_nKW, m_nStride, m_nPad);

        auto gradWeight = Tensor::zeros(cWeight.shapeVec());
        if (m_bHasBias) {
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

    // 20260327 ZJH 释放 BN 保存的输入和统计量
    void releaseSavedTensors() override {
        m_savedInput = Tensor();
        m_savedMean = Tensor();
        m_savedInvStd = Tensor();
        m_savedGamma = Tensor();
    }

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cInput = m_savedInput.contiguous();
#ifdef OM_HAS_CUDA
        if (cGradOut.isCuda()) {
            // 20260325 ZJH GPU 路径：完整 BatchNorm 反向（两遍 kernel）
            auto gradInput = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            auto gradGamma = Tensor::zeros({m_nChannels}, DeviceType::CUDA);
            auto gradBeta = Tensor::zeros({m_nChannels}, DeviceType::CUDA);
            CUDABackend::batchNorm2dBackward(
                cGradOut.floatDataPtr(), cInput.floatDataPtr(),
                m_savedMean.floatDataPtr(), m_savedInvStd.floatDataPtr(),
                m_savedGamma.floatDataPtr(),
                gradInput.mutableFloatDataPtr(),
                gradGamma.mutableFloatDataPtr(), gradBeta.mutableFloatDataPtr(),
                m_nBatch, m_nChannels, m_nH, m_nW);
            return {gradInput, gradGamma, gradBeta};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto cpuMean = m_savedMean.isCuda() ? m_savedMean.cpu() : m_savedMean;
        auto cpuInvStd = m_savedInvStd.isCuda() ? m_savedInvStd.cpu() : m_savedInvStd;
        auto cpuGamma = m_savedGamma.isCuda() ? m_savedGamma.cpu() : m_savedGamma;

        auto gradInput = Tensor::zeros(cInput.shapeVec());
        auto gradGamma = Tensor::zeros({m_nChannels});
        auto gradBeta = Tensor::zeros({m_nChannels});

        CPUBackend::batchNorm2dBackward(
            cGradOut.floatDataPtr(), cInput.floatDataPtr(),
            cpuMean.floatDataPtr(), cpuInvStd.floatDataPtr(),
            cpuGamma.floatDataPtr(),
            gradInput.mutableFloatDataPtr(),
            gradGamma.mutableFloatDataPtr(), gradBeta.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nH, m_nW);
        return {gradInput, gradGamma, gradBeta};
    }
};

// 20260319 ZJH MaxPool2dBackward — 最大池化反向
// 20260327 ZJH 已全 GPU 化：索引以 reinterpret_cast<int*> 方式读取 float 存储的 int 数据，
//              CUDABackend::maxPool2dBackward 直接在 GPU 上完成梯度散布
class MaxPool2dBackward : public GradFunction {
public:
    Tensor m_savedIndices;   // 20260319 ZJH 保存最大值索引（int 数据存为 float）
    int m_nBatch = 0;
    int m_nChannels = 0;
    int m_nHout = 0;
    int m_nWout = 0;
    int m_nH = 0;            // 20260319 ZJH 原始输入高度
    int m_nW = 0;            // 20260319 ZJH 原始输入宽度

    // 20260327 ZJH 释放索引张量
    void releaseSavedTensors() override { m_savedIndices = Tensor(); }

    // 20260326 ZJH 全 GPU 化：CUDABackend::maxPool2dBackward 直接在 GPU 上散布梯度
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            auto cGradOut = gradOutput.contiguous();
            auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW}, DeviceType::CUDA);
            // 20260326 ZJH indices 存为 float 在 GPU 上，需 reinterpret 为 int*
            // CUDABackend::maxPool2dBackward 接受 int* indices（前向已保存为 int）
            auto cIdx = m_savedIndices.contiguous();
            CUDABackend::maxPool2dBackward(
                cGradOut.floatDataPtr(), reinterpret_cast<const int*>(cIdx.floatDataPtr()),
                gradInput.mutableFloatDataPtr(),
                m_nBatch, m_nChannels, m_nH, m_nW, m_nHout, m_nWout);
            return {gradInput};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto cGradOut = gradOutput.contiguous();
        auto cpuIndices = m_savedIndices.isCuda() ? m_savedIndices.cpu() : m_savedIndices;
        auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
        const float* pIdxFloat = cpuIndices.floatDataPtr();
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
        auto cGradOut = gradOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGradOut.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend::avgPool2dBackward
            auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW}, DeviceType::CUDA);
            CUDABackend::avgPool2dBackward(
                cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
                m_nBatch, m_nChannels, m_nH, m_nW,
                m_nKH, m_nKW, m_nStride, m_nPad);
            return {gradInput};
        }
#endif
        // 20260319 ZJH CPU 路径
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
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：在 GPU 上 reshape（深拷贝）
            auto gradInput = Tensor::zeros(m_vecInputShape, DeviceType::CUDA);
            CUDABackend::copy(cGrad.floatDataPtr(), gradInput.mutableFloatDataPtr(),
                              static_cast<size_t>(cGrad.numel()));
            return {gradInput};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradInput = Tensor::fromData(cGrad.floatDataPtr(), m_vecInputShape);
        return {gradInput};
    }
};

// 20260319 ZJH DropoutBackward — Dropout 反向：使用与前向相同的 mask
class DropoutBackward : public GradFunction {
public:
    Tensor m_savedMask;  // 20260319 ZJH 保存前向使用的 mask（0 或 1/(1-p)）

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cMask = m_savedMask.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径
            auto gradInput = Tensor::zeros(cGrad.shapeVec(), DeviceType::CUDA);
            CUDABackend::mul(cGrad.floatDataPtr(), cMask.floatDataPtr(),
                             gradInput.mutableFloatDataPtr(),
                             static_cast<size_t>(gradInput.numel()));
            return {gradInput};
        }
#endif
        // 20260319 ZJH CPU 路径
        auto gradInput = Tensor::zeros(cGrad.shapeVec());
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
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cOutput = m_savedOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：注意参数顺序 CUDABackend(pGrad, pOutput, pOut, n)
            auto gradIn = Tensor::zeros(cOutput.shapeVec(), DeviceType::CUDA);
            CUDABackend::sigmoidBackward(cGrad.floatDataPtr(), cOutput.floatDataPtr(),
                                         gradIn.mutableFloatDataPtr(),
                                         static_cast<size_t>(gradIn.numel()));
            return { gradIn };
        }
#endif
        // 20260320 ZJH CPU 路径：注意参数顺序 CPUBackend(pOutput, pGradOut, pGradIn, n)
        auto gradIn = Tensor::zeros(cOutput.shapeVec());
        CPUBackend::sigmoidBackward(cOutput.floatDataPtr(), cGrad.floatDataPtr(),
                                     gradIn.mutableFloatDataPtr(),
                                     static_cast<size_t>(gradIn.numel()));
        return { gradIn };
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
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cInput = m_savedInput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend(pGrad, pInput, pOut, n, slope)
            auto gradIn = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            CUDABackend::leakyReluBackward(cGrad.floatDataPtr(), cInput.floatDataPtr(),
                                           gradIn.mutableFloatDataPtr(),
                                           static_cast<size_t>(gradIn.numel()), m_fSlope);
            return { gradIn };
        }
#endif
        // 20260320 ZJH CPU 路径：CPUBackend(pIn, pGradOut, pGradIn, n, slope)
        auto gradIn = Tensor::zeros(cInput.shapeVec());
        CPUBackend::leakyReluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                       gradIn.mutableFloatDataPtr(),
                                       static_cast<size_t>(gradIn.numel()), m_fSlope);
        return { gradIn };
    }
};

// 20260320 ZJH UpsampleBilinearBackwardFn — 双线性上采样反向
// 20260326 ZJH 已全 GPU 化：CUDABackend::upsampleBilinearBackward 原子归约梯度至输入尺寸
class UpsampleBilinearBackwardFn : public GradFunction {
public:
    int m_nBatch = 0;     // 20260320 ZJH 批次大小
    int m_nChannels = 0;  // 20260320 ZJH 通道数
    int m_nH = 0;         // 20260320 ZJH 原始输入高度
    int m_nW = 0;         // 20260320 ZJH 原始输入宽度
    int m_nScale = 2;     // 20260320 ZJH 上采样倍率

    // 20260326 ZJH 全 GPU 化
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        int nOutH = m_nH * m_nScale;  // 20260326 ZJH 上采样后高度
        int nOutW = m_nW * m_nScale;  // 20260326 ZJH 上采样后宽度
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            auto cGradOut = gradOutput.contiguous();
            auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW}, DeviceType::CUDA);
            CUDABackend::upsampleBilinearBackward(
                cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
                m_nBatch, m_nChannels, m_nH, m_nW, nOutH, nOutW);
            return { gradInput };
        }
#endif
        auto cGradOut = gradOutput.contiguous();
        auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
        CPUBackend::upsampleBilinearBackward(
            cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nH, m_nW, m_nScale);
        return { gradInput };
    }
};

// 20260320 ZJH ConcatChannelsBackwardFn — 沿通道维度拼接反向
// 20260326 ZJH 全 GPU 化：CUDABackend::concatChannelsBackward
class ConcatChannelsBackwardFn : public GradFunction {
public:
    int m_nBatch = 0;   // 20260320 ZJH 批次大小
    int m_nC1 = 0;      // 20260320 ZJH 第一个张量的通道数
    int m_nC2 = 0;      // 20260320 ZJH 第二个张量的通道数
    int m_nH = 0;       // 20260320 ZJH 高度
    int m_nW = 0;       // 20260320 ZJH 宽度

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        int nHW = m_nH * m_nW;
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            auto cGradOut = gradOutput.contiguous();
            auto gradA = Tensor::zeros({m_nBatch, m_nC1, m_nH, m_nW}, DeviceType::CUDA);
            auto gradB = Tensor::zeros({m_nBatch, m_nC2, m_nH, m_nW}, DeviceType::CUDA);
            CUDABackend::concatChannelsBackward(
                cGradOut.floatDataPtr(), gradA.mutableFloatDataPtr(), gradB.mutableFloatDataPtr(),
                m_nBatch, m_nC1, m_nC2, nHW);
            return { gradA, gradB };
        }
#endif
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
// 简化实现：仅返回零梯度（U-Net 训练时主要靠编码器梯度流）
class ConvTranspose2dBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;   // 20260320 ZJH 保存前向输入
    Tensor m_savedWeight;  // 20260320 ZJH 保存权重 [Cin, Cout, KH, KW]
    // 20260327 ZJH 释放保存的输入和权重
    void releaseSavedTensors() override { m_savedInput = Tensor(); m_savedWeight = Tensor(); }
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
        // 20260320 ZJH 简化：返回零梯度
        // 20260325 ZJH 在原始设备上创建零张量
        DeviceType eDev = gradOutput.device();
        auto gradInput = Tensor::zeros(m_savedInput.shapeVec(), eDev);
        auto gradWeight = Tensor::zeros(m_savedWeight.shapeVec(), eDev);
        if (m_bHasBias) {
            auto gradBias = Tensor::zeros({m_nCout}, eDev);
            return { gradInput, gradWeight, gradBias };
        }
        return { gradInput, gradWeight };
    }
};

// 20260320 ZJH BCEWithLogitsBackwardFn — 二元交叉熵反向
// 20260326 ZJH 全 GPU 化：CUDABackend::bceWithLogitsBackward
class BCEWithLogitsBackwardFn : public GradFunction {
public:
    Tensor m_savedLogits;   // 20260320 ZJH 保存前向 logits
    Tensor m_savedTargets;  // 20260320 ZJH 保存目标
    int m_nCount = 0;       // 20260320 ZJH 元素总数

    // 20260327 ZJH 释放保存的 logits 和目标
    void releaseSavedTensors() override { m_savedLogits = Tensor(); m_savedTargets = Tensor(); }

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            auto cLogits = m_savedLogits.contiguous();
            auto cTargets = m_savedTargets.contiguous();
            auto gradInput = Tensor::zeros(cLogits.shapeVec(), DeviceType::CUDA);
            float fInvN = 1.0f / static_cast<float>(m_nCount);
            CUDABackend::bceWithLogitsBackward(
                cLogits.floatDataPtr(), cTargets.floatDataPtr(),
                gradInput.mutableFloatDataPtr(), m_nCount, fInvN);
            return { gradInput };
        }
#endif
        auto cLogits = (m_savedLogits.isCuda() ? m_savedLogits.cpu() : m_savedLogits).contiguous();
        auto cTargets = (m_savedTargets.isCuda() ? m_savedTargets.cpu() : m_savedTargets).contiguous();
        auto gradInput = Tensor::zeros(cLogits.shapeVec());
        CPUBackend::bceWithLogitsBackward(
            cLogits.floatDataPtr(), cTargets.floatDataPtr(),
            gradInput.mutableFloatDataPtr(), m_nCount);
        float fGradScale = gradOutput.item();
        if (std::abs(fGradScale - 1.0f) > 1e-6f) {
            CPUBackend::mulScalar(gradInput.floatDataPtr(), fGradScale,
                                  gradInput.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradInput.numel()));
        }
        return { gradInput };
    }
};

// =========================================================
// Phase 5: 新增反向函数 — GELU / SiLU / LayerNorm / AdaptiveAvgPool2d
// =========================================================

// 20260320 ZJH GELUBackwardFn — GELU 激活反向
// 保存前向输入用于计算 GELU 导数
class GELUBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;  // 20260320 ZJH 保存前向输入

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cInput = m_savedInput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend(pGrad, pInput, pOut, n)
            auto gradIn = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            CUDABackend::geluBackward(cGrad.floatDataPtr(), cInput.floatDataPtr(),
                                      gradIn.mutableFloatDataPtr(),
                                      static_cast<size_t>(gradIn.numel()));
            return { gradIn };
        }
#endif
        // 20260320 ZJH CPU 路径：CPUBackend(pIn, pGradOut, pGradIn, n)
        auto gradIn = Tensor::zeros(cInput.shapeVec());
        CPUBackend::geluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                  gradIn.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradIn.numel()));
        return { gradIn };
    }
};

// 20260320 ZJH SiLUBackwardFn — SiLU (Swish) 激活反向
class SiLUBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;  // 20260320 ZJH 保存前向输入

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cInput = m_savedInput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend(pGrad, pInput, pOut, n)
            auto gradIn = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            CUDABackend::siluBackward(cGrad.floatDataPtr(), cInput.floatDataPtr(),
                                      gradIn.mutableFloatDataPtr(),
                                      static_cast<size_t>(gradIn.numel()));
            return { gradIn };
        }
#endif
        // 20260320 ZJH CPU 路径：CPUBackend(pIn, pGradOut, pGradIn, n)
        auto gradIn = Tensor::zeros(cInput.shapeVec());
        CPUBackend::siluBackward(cInput.floatDataPtr(), cGrad.floatDataPtr(),
                                  gradIn.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradIn.numel()));
        return { gradIn };
    }
};

// 20260320 ZJH LayerNormBackwardFn — LayerNorm 反向
// 20260327 ZJH 全 GPU 化：CUDABackend::layerNormBackward 直接在 GPU 上计算
class LayerNormBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;      // 20260320 ZJH 保存前向输入
    Tensor m_savedMean;       // 20260320 ZJH 保存均值 [batch]
    Tensor m_savedInvStd;     // 20260320 ZJH 保存逆标准差 [batch]
    Tensor m_savedGamma;      // 20260320 ZJH 保存 gamma 参数 [dim]
    int m_nBatch = 0;         // 20260320 ZJH 批次大小
    int m_nDim = 0;           // 20260320 ZJH 归一化维度大小

    // 20260327 ZJH 释放 LayerNorm 保存的张量
    void releaseSavedTensors() override {
        m_savedInput = Tensor();
        m_savedMean = Tensor();
        m_savedInvStd = Tensor();
        m_savedGamma = Tensor();
    }

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            // 20260327 ZJH GPU 路径：全量在 GPU 上计算 LayerNorm 反向
            auto cGradOut = gradOutput.contiguous();
            auto cInput = m_savedInput.contiguous();
            auto gradInput = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);
            auto gradGamma = Tensor::zeros({m_nDim}, DeviceType::CUDA);
            auto gradBeta = Tensor::zeros({m_nDim}, DeviceType::CUDA);
            CUDABackend::layerNormBackward(
                cGradOut.floatDataPtr(), cInput.floatDataPtr(),
                m_savedMean.floatDataPtr(), m_savedInvStd.floatDataPtr(),
                m_savedGamma.floatDataPtr(),
                gradInput.mutableFloatDataPtr(),
                gradGamma.mutableFloatDataPtr(), gradBeta.mutableFloatDataPtr(),
                m_nBatch, m_nDim);
            return { gradInput, gradGamma, gradBeta };
        }
#endif
        // 20260320 ZJH CPU 路径
        auto cGradOut = gradOutput.contiguous();
        auto cInput = m_savedInput.contiguous();
        auto gradInput = Tensor::zeros(cInput.shapeVec());
        auto gradGamma = Tensor::zeros({m_nDim});
        auto gradBeta = Tensor::zeros({m_nDim});
        CPUBackend::layerNormBackward(
            cGradOut.floatDataPtr(), cInput.floatDataPtr(),
            m_savedMean.floatDataPtr(), m_savedInvStd.floatDataPtr(),
            m_savedGamma.floatDataPtr(),
            gradInput.mutableFloatDataPtr(),
            gradGamma.mutableFloatDataPtr(), gradBeta.mutableFloatDataPtr(),
            m_nBatch, m_nDim);
        return { gradInput, gradGamma, gradBeta };
    }
};

// 20260320 ZJH AdaptiveAvgPool2dBackwardFn — 自适应平均池化反向
// 20260326 ZJH 全 GPU 化：直接调用 CUDABackend::adaptiveAvgPool2dBackward，消除 D2H 乒乓
class AdaptiveAvgPool2dBackwardFn : public GradFunction {
public:
    int m_nBatch = 0;
    int m_nChannels = 0;
    int m_nH = 0;          // 20260320 ZJH 原始输入高度
    int m_nW = 0;          // 20260320 ZJH 原始输入宽度
    int m_nOutH = 0;       // 20260320 ZJH 目标输出高度
    int m_nOutW = 0;       // 20260320 ZJH 目标输出宽度

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            // 20260326 ZJH GPU 直调：零 D2H 开销
            auto cGradOut = gradOutput.contiguous();
            auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW}, DeviceType::CUDA);
            CUDABackend::adaptiveAvgPool2dBackward(
                cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
                m_nBatch, m_nChannels, m_nH, m_nW, m_nOutH, m_nOutW);
            return { gradInput };
        }
#endif
        // 20260320 ZJH CPU 路径
        auto cGradOut = gradOutput.contiguous();
        auto gradInput = Tensor::zeros({m_nBatch, m_nChannels, m_nH, m_nW});
        CPUBackend::adaptiveAvgPool2dBackward(
            cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
            m_nBatch, m_nChannels, m_nH, m_nW, m_nOutH, m_nOutW);
        return { gradInput };
    }
};

// 20260321 ZJH TanhBackwardFn — tanh 激活反向
// 前向: out = tanh(in)
// 反向: grad_in = grad_out * (1 - out^2)，需要保存前向输出
class TanhBackwardFn : public GradFunction {
public:
    Tensor m_savedOutput;  // 20260321 ZJH 保存前向输出，用于计算 tanh 导数

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260325 ZJH 确保连续
        auto cOutput = m_savedOutput.contiguous();  // 20260325 ZJH 确保连续
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260325 ZJH GPU 路径：CUDABackend(pGrad, pOutput, pOut, n)
            auto gradIn = Tensor::zeros(cOutput.shapeVec(), DeviceType::CUDA);
            CUDABackend::tanhBackward(cGrad.floatDataPtr(), cOutput.floatDataPtr(),
                                      gradIn.mutableFloatDataPtr(),
                                      static_cast<size_t>(gradIn.numel()));
            return { gradIn };
        }
#endif
        // 20260321 ZJH CPU 路径：CPUBackend(pOutput, pGradOut, pGradIn, n)
        auto gradIn = Tensor::zeros(cOutput.shapeVec());
        CPUBackend::tanhBackward(cOutput.floatDataPtr(), cGrad.floatDataPtr(),
                                  gradIn.mutableFloatDataPtr(),
                                  static_cast<size_t>(gradIn.numel()));
        return { gradIn };
    }
};

// 20260328 ZJH ConcatLastDimBackwardFn — 沿最后一维拼接反向
// 将 gradOutput [outer, dimA+dimB] 拆为 gradA [outer, dimA] + gradB [outer, dimB]
class ConcatLastDimBackwardFn : public GradFunction {
public:
    int m_nOuter = 0;   // 20260328 ZJH 外层元素数（最后一维之外的所有维乘积）
    int m_nDimA = 0;    // 20260328 ZJH 第一个输入的最后一维大小
    int m_nDimB = 0;    // 20260328 ZJH 第二个输入的最后一维大小
    std::vector<int> m_vecShapeA;  // 20260328 ZJH 第一个输入的完整形状
    std::vector<int> m_vecShapeB;  // 20260328 ZJH 第二个输入的完整形状

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            // 20260328 ZJH GPU 路径：CUDABackend 拆分梯度
            auto cGradOut = gradOutput.contiguous();  // 20260328 ZJH 确保连续
            auto gradA = Tensor::zeros(m_vecShapeA, DeviceType::CUDA);  // 20260328 ZJH 在 GPU 上分配 gradA
            auto gradB = Tensor::zeros(m_vecShapeB, DeviceType::CUDA);  // 20260328 ZJH 在 GPU 上分配 gradB
            CUDABackend::concatLastDimBackward(
                cGradOut.floatDataPtr(), gradA.mutableFloatDataPtr(), gradB.mutableFloatDataPtr(),
                m_nOuter, m_nDimA, m_nDimB);
            return { gradA, gradB };  // 20260328 ZJH 返回拆分后的梯度
        }
#endif
        // 20260328 ZJH CPU 路径：逐行拆分
        auto cGradOut = gradOutput.contiguous();  // 20260328 ZJH 确保连续
        auto gradA = Tensor::zeros(m_vecShapeA);  // 20260328 ZJH CPU 上分配 gradA
        auto gradB = Tensor::zeros(m_vecShapeB);  // 20260328 ZJH CPU 上分配 gradB
        const float* pGO = cGradOut.floatDataPtr();  // 20260328 ZJH gradOutput 数据指针
        float* pGA = gradA.mutableFloatDataPtr();     // 20260328 ZJH gradA 数据指针
        float* pGB = gradB.mutableFloatDataPtr();     // 20260328 ZJH gradB 数据指针
        int nDimOut = m_nDimA + m_nDimB;              // 20260328 ZJH 拼接后的最后一维大小
        for (int i = 0; i < m_nOuter; ++i) {
            // 20260328 ZJH 前 dimA 列的梯度拷给 gradA
            for (int j = 0; j < m_nDimA; ++j) {
                pGA[i * m_nDimA + j] = pGO[i * nDimOut + j];
            }
            // 20260328 ZJH 后 dimB 列的梯度拷给 gradB
            for (int j = 0; j < m_nDimB; ++j) {
                pGB[i * m_nDimB + j] = pGO[i * nDimOut + m_nDimA + j];
            }
        }
        return { gradA, gradB };  // 20260328 ZJH 返回拆分后的梯度
    }
};

// 20260328 ZJH SliceLastDimBackwardFn — 沿最后一维切片反向
// 将 gradOutput [outer, sliceLen] 散布回 gradInput [outer, fullDim] 的 [start, start+len) 区域
class SliceLastDimBackwardFn : public GradFunction {
public:
    int m_nOuter = 0;    // 20260328 ZJH 外层元素数
    int m_nFullDim = 0;  // 20260328 ZJH 原始输入最后一维大小
    int m_nStart = 0;    // 20260328 ZJH 切片起始位置
    int m_nLen = 0;      // 20260328 ZJH 切片长度
    std::vector<int> m_vecFullShape;  // 20260328 ZJH 原始输入的完整形状

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
#ifdef OM_HAS_CUDA
        if (gradOutput.isCuda()) {
            // 20260328 ZJH GPU 路径：CUDABackend scatter 梯度回全尺寸张量
            auto cGradOut = gradOutput.contiguous();  // 20260328 ZJH 确保连续
            // 20260328 ZJH 分配全零的 gradInput，仅切片区域被 kernel 写入
            auto gradInput = Tensor::zeros(m_vecFullShape, DeviceType::CUDA);
            CUDABackend::sliceLastDimBackward(
                cGradOut.floatDataPtr(), gradInput.mutableFloatDataPtr(),
                m_nOuter, m_nFullDim, m_nStart, m_nLen);
            return { gradInput };  // 20260328 ZJH 返回 scatter 后的梯度
        }
#endif
        // 20260328 ZJH CPU 路径：全零张量 + 逐行 scatter
        auto cGradOut = gradOutput.contiguous();  // 20260328 ZJH 确保连续
        auto gradInput = Tensor::zeros(m_vecFullShape);  // 20260328 ZJH CPU 上分配全零 gradInput
        const float* pGO = cGradOut.floatDataPtr();       // 20260328 ZJH gradOutput 数据指针
        float* pGI = gradInput.mutableFloatDataPtr();      // 20260328 ZJH gradInput 数据指针
        for (int i = 0; i < m_nOuter; ++i) {
            // 20260328 ZJH 将 gradOutput 的每行写入 gradInput 的 [start, start+len) 区域
            for (int j = 0; j < m_nLen; ++j) {
                pGI[i * m_nFullDim + m_nStart + j] = pGO[i * m_nLen + j];
            }
        }
        return { gradInput };  // 20260328 ZJH 返回 scatter 后的梯度
    }
};

// 20260328 ZJH SoftmaxLastDimBackwardFn — 沿最后一维 softmax 的反向传播
// 前向: out[i] = exp(in[i]) / sum(exp(in[j]))（沿最后一维归一化）
// 反向: gradIn[i] = softmax[i] * (gradOut[i] - dot(gradOut, softmax))
// 这是标准的 softmax Jacobian-vector product，需要保存前向 softmax 输出
class SoftmaxLastDimBackwardFn : public GradFunction {
public:
    Tensor m_savedOutput;  // 20260328 ZJH 保存前向 softmax 输出，反向计算需要
    int m_nLastDim = 0;    // 20260328 ZJH 最后一维大小（softmax 归一化的维度）

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260328 ZJH 确保梯度连续
        auto cOutput = m_savedOutput.contiguous();  // 20260328 ZJH 确保 softmax 输出连续
        int nOuter = cOutput.numel() / m_nLastDim;  // 20260328 ZJH 计算外层维度积
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260328 ZJH GPU 路径：调用 CUDA softmax backward kernel
            auto gradIn = Tensor::zeros(cOutput.shapeVec(), DeviceType::CUDA);
            CUDABackend::softmaxLastDimBackward(
                cGrad.floatDataPtr(), cOutput.floatDataPtr(),
                gradIn.mutableFloatDataPtr(), nOuter, m_nLastDim);
            return { gradIn };  // 20260328 ZJH 返回 GPU 上的输入梯度
        }
#endif
        // 20260328 ZJH CPU 路径：调用 CPUBackend softmax backward
        auto gradIn = Tensor::zeros(cOutput.shapeVec());
        CPUBackend::softmaxLastDimBackward(
            cGrad.floatDataPtr(), cOutput.floatDataPtr(),
            gradIn.mutableFloatDataPtr(), nOuter, m_nLastDim);
        return { gradIn };  // 20260328 ZJH 返回 CPU 上的输入梯度
    }

    // 20260328 ZJH 释放保存的 softmax 输出张量以节省显存
    void releaseSavedTensors() override {
        m_savedOutput = Tensor();  // 20260328 ZJH 释放存储
    }
};

// 20260328 ZJH ClipBackwardFn — 值裁剪的反向传播
// 前向: out[i] = clamp(in[i], fMin, fMax)
// 反向: gradIn[i] = (input[i] >= fMin && input[i] <= fMax) ? gradOut[i] : 0
// 梯度在 [fMin, fMax] 范围内直通，范围外梯度为零（硬截断）
class ClipBackwardFn : public GradFunction {
public:
    Tensor m_savedInput;  // 20260328 ZJH 保存前向输入，用于判断梯度是否通过
    float m_fMin = 0.0f;  // 20260328 ZJH 裁剪下界
    float m_fMax = 1.0f;  // 20260328 ZJH 裁剪上界

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260328 ZJH 确保梯度连续
        auto cInput = m_savedInput.contiguous();  // 20260328 ZJH 确保输入连续
        int nTotal = static_cast<int>(cInput.numel());  // 20260328 ZJH 总元素数
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260328 ZJH GPU 路径：迁移到 CPU 计算（clip backward 计算量小，不值得单独写 kernel）
            auto cpuGrad = cGrad.cpu();  // 20260328 ZJH 梯度迁回 CPU
            auto cpuInput = cInput.isCuda() ? cInput.cpu() : cInput;  // 20260328 ZJH 输入迁回 CPU
            auto gradIn = Tensor::zeros(cpuInput.shapeVec());  // 20260328 ZJH CPU 上分配结果
            const float* pGO = cpuGrad.floatDataPtr();  // 20260328 ZJH 梯度数据指针
            const float* pIn = cpuInput.floatDataPtr();  // 20260328 ZJH 输入数据指针
            float* pGI = gradIn.mutableFloatDataPtr();  // 20260328 ZJH 输出数据指针
            for (int i = 0; i < nTotal; ++i) {
                // 20260328 ZJH 输入在 [min, max] 范围内则梯度直通，否则为零
                pGI[i] = (pIn[i] >= m_fMin && pIn[i] <= m_fMax) ? pGO[i] : 0.0f;
            }
            return { gradIn.cuda() };  // 20260328 ZJH 结果迁回 GPU
        }
#endif
        // 20260328 ZJH CPU 路径：逐元素计算 clip 梯度
        auto gradIn = Tensor::zeros(cInput.shapeVec());  // 20260328 ZJH 分配结果张量
        const float* pGO = cGrad.floatDataPtr();  // 20260328 ZJH 梯度数据指针
        const float* pIn = cInput.floatDataPtr();  // 20260328 ZJH 输入数据指针
        float* pGI = gradIn.mutableFloatDataPtr();  // 20260328 ZJH 输出数据指针
        for (int i = 0; i < nTotal; ++i) {
            // 20260328 ZJH 输入在 [min, max] 范围内则梯度直通，否则为零
            pGI[i] = (pIn[i] >= m_fMin && pIn[i] <= m_fMax) ? pGO[i] : 0.0f;
        }
        return { gradIn };  // 20260328 ZJH 返回 CPU 上的输入梯度
    }

    // 20260328 ZJH 释放保存的输入张量以节省显存
    void releaseSavedTensors() override {
        m_savedInput = Tensor();  // 20260328 ZJH 释放存储
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
// 20260325 ZJH GPU 调度重写：梯度累加在张量所在设备上进行
// =========================================================

// 20260319 ZJH runBackward — 从根节点出发，拓扑排序后反向传播梯度
// pRootGradFn: 根节点的 GradFunction（通常是 loss 的 gradFn）
// rootGrad: 根节点的初始梯度（通常是全 1 标量）
// 使用 Kahn 算法（BFS 拓扑排序）确保每个节点在所有后继节点处理后才处理
void runBackward(std::shared_ptr<GradFunction> pRootGradFn, const Tensor& rootGrad) {
    if (!pRootGradFn) return;  // 20260319 ZJH 无梯度函数则直接返回

    // 20260319 ZJH 阶段 1：BFS 收集所有可达节点并计算入度（被引用次数）
    std::unordered_map<GradFunction*, int> mapInDegree;
    std::unordered_set<GradFunction*> setVisited;
    std::queue<GradFunction*> qBfs;

    qBfs.push(pRootGradFn.get());
    setVisited.insert(pRootGradFn.get());
    mapInDegree[pRootGradFn.get()] = 0;

    while (!qBfs.empty()) {
        GradFunction* pCurr = qBfs.front();
        qBfs.pop();

        for (const auto& edge : pCurr->m_vecInputEdges) {
            if (edge.pGradFn) {
                GradFunction* pNext = edge.pGradFn.get();
                mapInDegree[pNext]++;
                if (setVisited.find(pNext) == setVisited.end()) {
                    setVisited.insert(pNext);
                    qBfs.push(pNext);
                }
            }
        }
    }

    // 20260319 ZJH 阶段 2：Kahn 拓扑排序 + 反向梯度传播
    std::unordered_map<GradFunction*, Tensor> mapGrads;
    mapGrads[pRootGradFn.get()] = rootGrad;

    std::queue<GradFunction*> qTopo;
    for (auto& [pNode, nDeg] : mapInDegree) {
        if (nDeg == 0) {
            qTopo.push(pNode);
        }
    }

    while (!qTopo.empty()) {
        GradFunction* pCurr = qTopo.front();
        qTopo.pop();

        auto itGrad = mapGrads.find(pCurr);
        if (itGrad == mapGrads.end()) continue;
        Tensor currGrad = itGrad->second;

        // 20260319 ZJH 调用当前节点的 backward 计算各输入的梯度
        auto vecInputGrads = pCurr->backward(currGrad);

        // 20260327 ZJH 立即释放当前节点保存的中间张量（前向激活值等）
        // 反向计算完成后这些张量不再需要，释放可回收大量 GPU 显存
        // 效果：降低峰值显存 30-50%，batch 间显存曲线更平滑
        pCurr->releaseSavedTensors();

        // 20260327 ZJH 释放当前节点的梯度（已传播到下游，不再需要）
        mapGrads.erase(pCurr);

        // 20260319 ZJH 将计算出的梯度传递给各下游节点
        for (size_t i = 0; i < pCurr->m_vecInputEdges.size(); ++i) {
            const auto& edge = pCurr->m_vecInputEdges[i];
            if (!edge.pGradFn) continue;

            GradFunction* pNext = edge.pGradFn.get();

            // 20260319 ZJH 将梯度累加到下游节点（可能有多条边汇聚）
            // 20260325 ZJH GPU 调度重写：在张量所在设备上累加
            auto itNextGrad = mapGrads.find(pNext);
            if (itNextGrad == mapGrads.end()) {
                // 20260319 ZJH 首次收到梯度，直接赋值
                mapGrads[pNext] = vecInputGrads[i];
            } else {
                // 20260319 ZJH 已有梯度，逐元素累加
                auto existing = itNextGrad->second.contiguous();
                auto incoming = vecInputGrads[i].contiguous();
#ifdef OM_HAS_CUDA
                if (existing.isCuda()) {
                    // 20260325 ZJH GPU 路径：在 GPU 上累加
                    auto gpuIncoming = incoming.isCuda() ? incoming : incoming.cuda();
                    auto accumulated = Tensor::zeros(existing.shapeVec(), DeviceType::CUDA);
                    CUDABackend::add(existing.floatDataPtr(), gpuIncoming.floatDataPtr(),
                                     accumulated.mutableFloatDataPtr(),
                                     static_cast<size_t>(accumulated.numel()));
                    mapGrads[pNext] = accumulated;
                } else
#endif
                {
                    // 20260325 ZJH CPU 路径
                    auto cpuIncoming = incoming.isCuda() ? incoming.cpu() : incoming;
                    auto accumulated = Tensor::zeros(existing.shapeVec());
                    CPUBackend::add(existing.floatDataPtr(), cpuIncoming.floatDataPtr(),
                                    accumulated.mutableFloatDataPtr(),
                                    static_cast<size_t>(accumulated.numel()));
                    mapGrads[pNext] = accumulated;
                }
            }

            // 20260319 ZJH 减少下游节点的入度
            mapInDegree[pNext]--;
            if (mapInDegree[pNext] == 0) {
                qTopo.push(pNext);
            }
        }
    }
}

// 20260328 ZJH Transpose2dBatchedBackwardFn — 批量转置反向
// 前向: [batch, M, N] -> [batch, N, M]
// 反向: transpose 是自身的逆运算，再转置一次即可还原梯度形状
// 无需额外 CUDA kernel，直接复用 CUDABackend::transpose / CPUBackend::transpose2d
class Transpose2dBatchedBackwardFn : public GradFunction {
public:
    std::vector<int> m_vecOrigShape;  // 20260328 ZJH 前向输入的原始形状 [batch, M, N]

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // 20260328 ZJH 确保梯度连续
        // 20260328 ZJH gradOutput 形状为 [batch, N, M]，转置回 [batch, M, N]
        int nBatch = cGrad.shape(0);  // 20260328 ZJH 批次大小
        int nN = cGrad.shape(1);      // 20260328 ZJH 转置后的行数（原始列数）
        int nM = cGrad.shape(2);      // 20260328 ZJH 转置后的列数（原始行数）
        auto gradInput = Tensor::zeros(m_vecOrigShape, cGrad.device());  // 20260328 ZJH 在原设备上分配 [batch, M, N]
        int nNM = nN * nM;  // 20260328 ZJH 单个矩阵元素数
#ifdef OM_HAS_CUDA
        if (cGrad.isCuda()) {
            // 20260328 ZJH GPU 路径：逐 batch 调用 CUDABackend::transpose
            const float* pIn = cGrad.floatDataPtr();      // 20260328 ZJH 梯度输入指针
            float* pOut = gradInput.mutableFloatDataPtr(); // 20260328 ZJH 梯度输出指针
            for (int b = 0; b < nBatch; ++b) {
                CUDABackend::transpose(pIn + b * nNM, pOut + b * nNM, nN, nM);  // 20260328 ZJH [N,M] -> [M,N]
            }
            return { gradInput };  // 20260328 ZJH 返回还原形状的梯度
        }
#endif
        // 20260328 ZJH CPU 路径：逐 batch 调用 CPUBackend::transpose2d
        const float* pIn = cGrad.floatDataPtr();      // 20260328 ZJH 梯度输入指针
        float* pOut = gradInput.mutableFloatDataPtr(); // 20260328 ZJH 梯度输出指针
        for (int b = 0; b < nBatch; ++b) {
            CPUBackend::transpose2d(pIn + b * nNM, pOut + b * nNM, nN, nM);  // 20260328 ZJH [N,M] -> [M,N]
        }
        return { gradInput };  // 20260328 ZJH 返回还原形状的梯度
    }
};

// 20260328 ZJH DiceLossBackwardFn — Fused Dice Loss 反向传播
// 前向: loss = 1 - 2*sum(sig*target) / (sum(sig) + sum(target) + eps)
// 反向: dLoss/dLogit = -gradOutput * 2*(target_i*D - I)/(D^2) * sig*(1-sig)
// 保存 sigmoid 输出 + one-hot 目标 + 3 个统计量，反向直接调用 fused kernel
class DiceLossBackwardFn : public GradFunction {
public:
    Tensor m_savedSigmoid;  // 20260328 ZJH 前向 sigmoid 输出 [B,C,H,W]，用于计算 sig*(1-sig)
    Tensor m_savedTarget;   // 20260328 ZJH one-hot 目标掩码 [B,C,H,W]，用于计算 dDice/dSig
    Tensor m_savedStats;    // 20260328 ZJH {intersection, predSum, targetSum}，前向归约统计量

    // 20260328 ZJH 反向传播：根据设备类型调度 GPU/CPU 路径
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGradOut = gradOutput.contiguous();  // 20260328 ZJH 确保上游梯度连续
        auto cSigmoid = m_savedSigmoid.contiguous();  // 20260328 ZJH 确保 sigmoid 输出连续
        auto cTarget = m_savedTarget.contiguous();     // 20260328 ZJH 确保 one-hot 目标连续
        auto cStats = m_savedStats.contiguous();       // 20260328 ZJH 确保统计量连续
        int nCount = static_cast<int>(cSigmoid.numel());  // 20260328 ZJH 元素总数 N = B*C*H*W

#ifdef OM_HAS_CUDA
        if (cGradOut.isCuda()) {
            // 20260328 ZJH GPU 路径：调用 fused CUDA 反向 kernel
            auto tGradLogits = Tensor::zeros(cSigmoid.shapeVec(), DeviceType::CUDA);  // 20260328 ZJH 分配 GPU 梯度张量
            CUDABackend::diceLossBackward(
                cSigmoid.floatDataPtr(), cTarget.floatDataPtr(),
                cStats.floatDataPtr(), cGradOut.floatDataPtr(),
                tGradLogits.mutableFloatDataPtr(), nCount);  // 20260328 ZJH 1 kernel 完成全部反向计算
            return { tGradLogits };  // 20260328 ZJH 返回 logit 梯度
        }
#endif
        // 20260328 ZJH CPU 路径：手动逐元素计算（与 CUDA kernel 等价的标量实现）
        auto tGradLogits = Tensor::zeros(cSigmoid.shapeVec());  // 20260328 ZJH 分配 CPU 梯度张量
        const float* pSig = cSigmoid.floatDataPtr();     // 20260328 ZJH sigmoid 输出指针
        const float* pTgt = cTarget.floatDataPtr();       // 20260328 ZJH one-hot 目标指针
        const float* pSt = cStats.floatDataPtr();         // 20260328 ZJH 统计量指针
        float fGradOut = cGradOut.floatDataPtr()[0];       // 20260328 ZJH 上游标量梯度
        float* pGradLogits = tGradLogits.mutableFloatDataPtr();  // 20260328 ZJH 输出梯度指针

        float fIntersection = pSt[0];  // 20260328 ZJH 交集 I
        float fPredSum = pSt[1];       // 20260328 ZJH 预测和 P
        float fTargetSum = pSt[2];     // 20260328 ZJH 目标和 T
        float fD = fPredSum + fTargetSum + 1e-6f;  // 20260328 ZJH 分母 D
        float fInvD2 = 1.0f / (fD * fD);           // 20260328 ZJH 1/D^2

        // 20260328 ZJH 逐元素计算 logit 梯度
        for (int i = 0; i < nCount; ++i) {
            float fSig = pSig[i];    // 20260328 ZJH 当前 sigmoid 值
            float fTgt = pTgt[i];    // 20260328 ZJH 当前目标值
            // 20260328 ZJH dDice/dSig = 2*(target*D - intersection) / D^2
            float fDDiceDSig = 2.0f * (fTgt * fD - fIntersection) * fInvD2;
            // 20260328 ZJH dSig/dLogit = sig * (1 - sig)
            float fDSigDLogit = fSig * (1.0f - fSig);
            // 20260328 ZJH 链式法则: dLoss/dLogit = -gradOut * dDice/dSig * dSig/dLogit
            pGradLogits[i] = -fGradOut * fDDiceDSig * fDSigDLogit;
        }
        return { tGradLogits };  // 20260328 ZJH 返回 logit 梯度
    }
};

}  // namespace om
