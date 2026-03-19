// 20260319 ZJH Tensor 运算模块 — 元素运算、matmul、reshape、transpose、slice、归约、自动微分集成
module;

#include <vector>
#include <stdexcept>
#include <memory>
#include <cassert>
#include <random>
#include <cmath>
#include "df_types.h"

export module df.engine.tensor_ops;

// 20260319 ZJH 导入依赖模块：存储层、张量类、CPU 计算内核、自动微分
import df.engine.tensor_storage;
import df.engine.tensor;
import df.hal.cpu_backend;
import df.engine.autograd;

export namespace df {

// =========================================================
// AutoGrad 辅助函数
// =========================================================

// 20260319 ZJH castGradFn — 将类型擦除的 shared_ptr<void> 转换为 shared_ptr<GradFunction>
// 用于从 Tensor 中提取 GradFunction（Tensor 以 void 存储避免循环依赖）
std::shared_ptr<GradFunction> castGradFn(std::shared_ptr<void> p) {
    return std::static_pointer_cast<GradFunction>(p);  // 20260319 ZJH 静态类型转换
}

// 20260319 ZJH castGradAccum — 将类型擦除的 shared_ptr<void> 转换为 shared_ptr<GradAccumulator>
std::shared_ptr<GradAccumulator> castGradAccum(std::shared_ptr<void> p) {
    return std::static_pointer_cast<GradAccumulator>(p);  // 20260319 ZJH 静态类型转换
}

// 20260319 ZJH tensorSetRequiresGrad — 设置张量是否需要梯度，同时创建 GradAccumulator
// 必须在 tensor_ops 中实现（而非 Tensor::setRequiresGrad），因为需要创建 GradAccumulator 对象
// GradAccumulator 定义在 autograd 模块中，tensor.ixx 不可导入 autograd
void tensorSetRequiresGrad(Tensor& t, bool b) {
    t.setRequiresGrad(b);  // 20260319 ZJH 设置 requires_grad 标志
    if (b && !t.gradAccumRaw()) {
        // 20260319 ZJH 首次设置为 true 时，立即创建 GradAccumulator
        // 这样后续的拷贝也会共享同一个 GradAccumulator（shared_ptr 拷贝语义）
        auto pAccum = std::make_shared<GradAccumulator>();
        t.setGradAccumRaw(pAccum);  // 20260319 ZJH 存储为 shared_ptr<void>
    }
}

// 20260319 ZJH ensureLeafAccumulator — 确保叶节点有 LeafAccumulator 和 GradAccumulator
// 叶节点条件：requiresGrad=true 且 gradFnRaw 为 nullptr 或已是 LeafAccumulator
// 此函数在每个 op 中调用，确保叶节点参与计算图
void ensureLeafAccumulator(Tensor& t) {
    // 20260319 ZJH 仅处理需要梯度且尚无 gradFn 的叶节点
    if (t.requiresGrad() && !t.gradFnRaw()) {
        // 20260319 ZJH 获取或创建 GradAccumulator
        std::shared_ptr<GradAccumulator> pAccum;
        if (t.gradAccumRaw()) {
            // 20260319 ZJH 已有累加器（由 tensorSetRequiresGrad 创建），复用
            pAccum = castGradAccum(t.gradAccumRaw());
        } else {
            // 20260319 ZJH 兜底创建 GradAccumulator（理论上不应走到这里）
            pAccum = std::make_shared<GradAccumulator>();
            t.setGradAccumRaw(pAccum);  // 20260319 ZJH 设置到张量上
        }
        // 20260319 ZJH 创建 LeafAccumulator 作为叶节点的 GradFunction
        auto pLeaf = std::make_shared<LeafAccumulator>();
        pLeaf->m_pAccumulator = pAccum;  // 20260319 ZJH 关联到 GradAccumulator
        t.setGradFnRaw(pLeaf);  // 20260319 ZJH 设置 gradFn
    }
}

// 20260319 ZJH makeEdge — 为张量创建 Edge（用于连接计算图）
// 如果张量有 gradFn，则创建指向该 gradFn 的 Edge；否则返回空 Edge
Edge makeEdge(Tensor& t, int nIndex = 0) {
    ensureLeafAccumulator(t);  // 20260319 ZJH 确保叶节点有 LeafAccumulator
    Edge edge;
    if (t.gradFnRaw()) {
        edge.pGradFn = castGradFn(t.gradFnRaw());  // 20260319 ZJH 设置上游 gradFn
        edge.nInputIndex = nIndex;  // 20260319 ZJH 输入索引
    }
    return edge;
}

// ===== 元素运算 =====

// 20260319 ZJH 逐元素加法：result = a + b（形状必须相同）
// 先将两个张量转为连续内存，再调用 CPUBackend::add 逐元素相加
// 如果任一输入需要梯度，则创建 AddBackward 并连接计算图
Tensor tensorAdd(Tensor a, Tensor b) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保 a 为连续内存（非连续则拷贝）
    auto cb = b.contiguous();  // 20260319 ZJH 确保 b 为连续内存
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配与 a 相同形状的输出张量
    // 20260319 ZJH 调用 CPU 逐元素加法内核，numel() 个 float
    CPUBackend::add(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，创建 AddBackward 节点
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<AddBackward>();  // 20260319 ZJH 创建加法反向节点
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));  // 20260319 ZJH 连接输入 a
        pBackward->m_vecInputEdges.push_back(makeEdge(b, 0));  // 20260319 ZJH 连接输入 b
        result.setGradFnRaw(pBackward);  // 20260319 ZJH 设置结果的 gradFn
        result.setRequiresGrad(true);  // 20260319 ZJH 结果也需要梯度
    }

    return result;  // 20260319 ZJH 返回计算结果张量
}

// 20260319 ZJH 逐元素减法：result = a - b
// AutoGrad: SubBackward, grad_a = gradOutput, grad_b = -gradOutput
Tensor tensorSub(Tensor a, Tensor b) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 逐元素减法内核
    CPUBackend::sub(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，创建 SubBackward 节点
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<SubBackward>();
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(b, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH 逐元素乘法：result = a * b（Hadamard 积）
// AutoGrad: MulBackward, 保存 a 和 b, grad_a = gradOutput*b, grad_b = gradOutput*a
Tensor tensorMul(Tensor a, Tensor b) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 逐元素乘法内核
    CPUBackend::mul(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，创建 MulBackward 并保存输入
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<MulBackward>();
        pBackward->m_savedA = a;  // 20260319 ZJH 保存 a 用于反向传播
        pBackward->m_savedB = b;  // 20260319 ZJH 保存 b 用于反向传播
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(b, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH 逐元素除法：result = a / b（调用方保证 b 中无零）
// 暂不支持 AutoGrad（无 DivBackward）
Tensor tensorDiv(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 逐元素除法内核
    CPUBackend::div(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    return result;
}

// 20260319 ZJH 加标量：result = a + fScalar（广播到所有元素）
// AutoGrad: AddScalarBackward, grad_a = gradOutput
Tensor tensorAddScalar(Tensor a, float fScalar) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 加标量内核
    CPUBackend::addScalar(ca.floatDataPtr(), fScalar,
                          result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果输入需要梯度，创建 AddScalarBackward
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<AddScalarBackward>();
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH 乘标量：result = a * fScalar（广播到所有元素）
// AutoGrad: MulScalarBackward, 保存 scalar, grad_a = gradOutput * scalar
Tensor tensorMulScalar(Tensor a, float fScalar) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 乘标量内核
    CPUBackend::mulScalar(ca.floatDataPtr(), fScalar,
                          result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果输入需要梯度，创建 MulScalarBackward
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<MulScalarBackward>();
        pBackward->m_fScalar = fScalar;  // 20260319 ZJH 保存标量值
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// ===== 矩阵乘法 =====

// 20260319 ZJH 二维矩阵乘法：result = a @ b
// a 形状 [M, K]，b 形状 [K, N]，result 形状 [M, N]
// AutoGrad: MatMulBackward, 保存 A 和 B
Tensor tensorMatmul(Tensor a, Tensor b) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保 a 连续，matmul 需要行主序连续内存
    auto cb = b.contiguous();  // 20260319 ZJH 确保 b 连续
    int nM = ca.shape(0);  // 20260319 ZJH 行数（矩阵 A 的行）
    int nK = ca.shape(1);  // 20260319 ZJH 内维度（A 的列 = B 的行）
    int nN = cb.shape(1);  // 20260319 ZJH 列数（矩阵 B 的列）
    auto result = Tensor::zeros({nM, nN});  // 20260319 ZJH 分配 [M, N] 输出张量
    // 20260319 ZJH 调用 CPU matmul 内核：A[M,K]*B[K,N]->C[M,N]
    CPUBackend::matmul(ca.floatDataPtr(), cb.floatDataPtr(),
                       result.mutableFloatDataPtr(), nM, nK, nN);

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，创建 MatMulBackward
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<MatMulBackward>();
        pBackward->m_savedA = a;  // 20260319 ZJH 保存 A 用于反向传播
        pBackward->m_savedB = b;  // 20260319 ZJH 保存 B 用于反向传播
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(b, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;  // 20260319 ZJH 返回矩阵乘法结果
}

// ===== 形状变换 =====

// 20260319 ZJH reshape — 改变张量形状，不拷贝数据（若原张量已连续）
// vecNewShape: 新形状，元素总数必须与原张量一致
Tensor tensorReshape(const Tensor& t, std::vector<int> vecNewShape) {
    // 20260319 ZJH 验证新旧形状的元素总数相等
    int nNewNumel = 1;
    for (int s : vecNewShape) nNewNumel *= s;  // 20260319 ZJH 计算新形状的元素总数
    if (nNewNumel != t.numel()) {
        // 20260319 ZJH 元素数不匹配时抛出异常
        throw std::invalid_argument("tensorReshape: total element count mismatch");
    }

    if (t.isContiguous()) {
        // 20260319 ZJH 原张量连续：直接共享 Storage，仅修改 shape/strides
        std::vector<int> vecNewStrides(vecNewShape.size());
        if (!vecNewShape.empty()) {
            vecNewStrides.back() = 1;  // 20260319 ZJH 最低维步长固定为 1
            for (int d = static_cast<int>(vecNewShape.size()) - 2; d >= 0; --d) {
                vecNewStrides[static_cast<size_t>(d)] =
                    vecNewStrides[static_cast<size_t>(d + 1)]
                    * vecNewShape[static_cast<size_t>(d + 1)];
            }
        }
        return Tensor::makeView(t.storage(), vecNewShape, vecNewStrides, t.offset());
    } else {
        // 20260319 ZJH 非连续张量：先连续化再 reshape
        auto ct = t.contiguous();
        return tensorReshape(ct, vecNewShape);
    }
}

// 20260319 ZJH transpose — 交换两个维度，生成视图（不拷贝数据）
Tensor tensorTranspose(const Tensor& t, int nDim0, int nDim1) {
    auto vecShape   = t.shapeVec();
    auto vecStrides = t.stridesVec();
    std::swap(vecShape[static_cast<size_t>(nDim0)], vecShape[static_cast<size_t>(nDim1)]);
    std::swap(vecStrides[static_cast<size_t>(nDim0)], vecStrides[static_cast<size_t>(nDim1)]);
    return Tensor::makeView(t.storage(), vecShape, vecStrides, t.offset());
}

// 20260319 ZJH slice — 在指定维度上截取子区间 [nStart, nEnd)
Tensor tensorSlice(const Tensor& t, int nDim, int nStart, int nEnd) {
    auto vecShape   = t.shapeVec();
    auto vecStrides = t.stridesVec();
    int nNewOffset = t.offset() + nStart * vecStrides[static_cast<size_t>(nDim)];
    vecShape[static_cast<size_t>(nDim)] = nEnd - nStart;
    return Tensor::makeView(t.storage(), vecShape, vecStrides, nNewOffset);
}

// ===== 归约运算 =====

// 20260319 ZJH 全局求和：返回标量张量（shape={1}），支持自动微分
// 返回类型从 float 改为 Tensor，使 sum 可参与计算图
// AutoGrad: SumBackward, grad_a = full(inputShape, gradOutput.item())
Tensor tensorSum(Tensor t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存
    float fSum = CPUBackend::sum(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
    // 20260319 ZJH 创建标量张量（shape={1}）存储求和结果
    auto result = Tensor::full({1}, fSum);

    // 20260319 ZJH AutoGrad: 如果输入需要梯度，创建 SumBackward
    if (t.requiresGrad()) {
        auto pBackward = std::make_shared<SumBackward>();
        pBackward->m_vecInputShape = t.shapeVec();  // 20260319 ZJH 保存输入形状
        pBackward->m_vecInputEdges.push_back(makeEdge(t, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;  // 20260319 ZJH 返回标量张量
}

// 20260319 ZJH 全局最大值：返回张量所有元素中的最大值（不参与计算图）
float tensorMax(const Tensor& t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存
    return CPUBackend::max(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
}

// 20260319 ZJH 全局最小值：返回张量所有元素中的最小值（不参与计算图）
float tensorMin(const Tensor& t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存
    return CPUBackend::min(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
}

// =========================================================
// AutoGrad 用户接口
// =========================================================

// 20260319 ZJH tensorBackward — 对 loss 张量执行反向传播
// loss 必须是标量张量（numel()==1），从 loss 的 gradFn 出发反向传播
void tensorBackward(Tensor& loss) {
    assert(loss.numel() == 1 && "tensorBackward — loss must be scalar (numel==1)");
    auto pGradFn = castGradFn(loss.gradFnRaw());  // 20260319 ZJH 获取 loss 的 GradFunction
    if (!pGradFn) return;  // 20260319 ZJH 无计算图则直接返回
    // 20260319 ZJH 初始梯度为全 1 标量（dl/dl = 1）
    auto rootGrad = Tensor::ones({1});
    runBackward(pGradFn, rootGrad);  // 20260319 ZJH 执行反向传播
}

// 20260319 ZJH tensorGetGrad — 获取叶节点的梯度张量
// 通过 gradAccumRaw 获取 GradAccumulator，返回其中的 m_grad
Tensor tensorGetGrad(const Tensor& t) {
    auto pRaw = t.gradAccumRaw();  // 20260319 ZJH 获取类型擦除的 GradAccumulator
    if (!pRaw) return Tensor();  // 20260319 ZJH 无累加器则返回空张量
    auto pAccum = castGradAccum(pRaw);  // 20260319 ZJH 类型转换
    if (!pAccum->m_bHasGrad) return Tensor();  // 20260319 ZJH 无梯度则返回空张量
    return pAccum->m_grad;  // 20260319 ZJH 返回累积的梯度
}

// 20260319 ZJH tensorZeroGrad — 清零叶节点的梯度
void tensorZeroGrad(Tensor& t) {
    auto pRaw = t.gradAccumRaw();  // 20260319 ZJH 获取类型擦除的 GradAccumulator
    if (!pRaw) return;  // 20260319 ZJH 无累加器则直接返回
    auto pAccum = castGradAccum(pRaw);  // 20260319 ZJH 类型转换
    pAccum->zero();  // 20260319 ZJH 清零梯度
}

// =========================================================
// Phase 1D: 激活函数、损失函数、广播运算
// =========================================================

// 20260319 ZJH tensorReLU — 逐元素 ReLU 激活，支持自动微分
// 前向: out[i] = max(0, in[i])
// 反向: grad_in[i] = grad_out[i] * (in[i] > 0 ? 1 : 0)
Tensor tensorReLU(Tensor a) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU ReLU 前向内核
    CPUBackend::relu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                     static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果输入需要梯度，创建 ReLUBackward 节点
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<ReLUBackward>();  // 20260319 ZJH 创建 ReLU 反向节点
        pBackward->m_savedInput = ca;  // 20260319 ZJH 保存输入，反向时判断哪些位置 > 0
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));  // 20260319 ZJH 连接输入边
        result.setGradFnRaw(pBackward);  // 20260319 ZJH 设置结果的 gradFn
        result.setRequiresGrad(true);  // 20260319 ZJH 结果也需要梯度
    }

    return result;  // 20260319 ZJH 返回 ReLU 激活后的张量
}

// 20260319 ZJH tensorAddBias — 广播偏置加法，支持自动微分
// 输入 a 形状 [batch, cols]，bias 形状 [1, cols]
// 输出形状 [batch, cols]，每行加上 bias 的对应列值
Tensor tensorAddBias(Tensor a, Tensor bias) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保输入连续
    auto cb = bias.contiguous();  // 20260319 ZJH 确保偏置连续
    int nBatch = ca.shape(0);  // 20260319 ZJH 批次大小
    int nCols = ca.shape(1);   // 20260319 ZJH 特征维度

    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 广播加法内核：每行加偏置
    CPUBackend::addBias(ca.floatDataPtr(), cb.floatDataPtr(),
                        result.mutableFloatDataPtr(), nBatch, nCols);

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，创建 AddBiasBackward 节点
    if (a.requiresGrad() || bias.requiresGrad()) {
        auto pBackward = std::make_shared<AddBiasBackward>();  // 20260319 ZJH 创建广播加法反向节点
        pBackward->m_nBatch = nBatch;  // 20260319 ZJH 保存批次大小
        pBackward->m_nCols = nCols;    // 20260319 ZJH 保存列数
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));     // 20260319 ZJH 连接输入边
        pBackward->m_vecInputEdges.push_back(makeEdge(bias, 0));  // 20260319 ZJH 连接偏置边
        result.setGradFnRaw(pBackward);  // 20260319 ZJH 设置结果的 gradFn
        result.setRequiresGrad(true);    // 20260319 ZJH 结果也需要梯度
    }

    return result;  // 20260319 ZJH 返回广播加法结果
}

// 20260319 ZJH tensorSoftmaxCrossEntropy — Softmax + 交叉熵联合损失，支持自动微分
// logits: [batch, classes] 原始分数，targets: [batch, classes] one-hot 编码
// 返回: 标量损失张量 (shape={1})，支持反向传播
// 联合计算的好处：数值稳定性更好，反向梯度公式更简单
Tensor tensorSoftmaxCrossEntropy(Tensor logits, const Tensor& targets) {
    auto cLogits = logits.contiguous();   // 20260319 ZJH 确保 logits 连续
    auto cTargets = targets.contiguous(); // 20260319 ZJH 确保 targets 连续
    int nBatch = cLogits.shape(0);    // 20260319 ZJH 批次大小
    int nClasses = cLogits.shape(1);  // 20260319 ZJH 类别数

    // 20260319 ZJH 第一步：计算 softmax 概率
    auto softmaxOut = Tensor::zeros({nBatch, nClasses});
    CPUBackend::softmax(cLogits.floatDataPtr(), softmaxOut.mutableFloatDataPtr(),
                        nBatch, nClasses);

    // 20260319 ZJH 第二步：计算交叉熵损失
    float fLoss = CPUBackend::crossEntropy(softmaxOut.floatDataPtr(), cTargets.floatDataPtr(),
                                           nBatch, nClasses);

    // 20260319 ZJH 创建标量损失张量
    auto result = Tensor::full({1}, fLoss);

    // 20260319 ZJH AutoGrad: 如果 logits 需要梯度，创建联合反向节点
    if (logits.requiresGrad()) {
        auto pBackward = std::make_shared<SoftmaxCrossEntropyBackward>();
        pBackward->m_savedSoftmax = softmaxOut;   // 20260319 ZJH 保存 softmax 输出
        pBackward->m_savedTargets = cTargets;      // 20260319 ZJH 保存 one-hot 目标
        pBackward->m_nBatch = nBatch;              // 20260319 ZJH 保存批次大小
        pBackward->m_nClasses = nClasses;          // 20260319 ZJH 保存类别数
        pBackward->m_vecInputEdges.push_back(makeEdge(logits, 0));  // 20260319 ZJH 连接 logits 边
        result.setGradFnRaw(pBackward);  // 20260319 ZJH 设置结果的 gradFn
        result.setRequiresGrad(true);    // 20260319 ZJH 结果也需要梯度
    }

    return result;  // 20260319 ZJH 返回标量损失张量
}

// 20260319 ZJH tensorArgmax — 逐行返回最大值索引，不参与自动微分
// 输入: t 形状 [batch, classes]
// 返回: int 向量，长度为 batch，每个元素是该行最大值的列索引
std::vector<int> tensorArgmax(const Tensor& t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保输入连续
    int nBatch = ct.shape(0);      // 20260319 ZJH 批次大小
    int nClasses = ct.shape(1);    // 20260319 ZJH 类别数
    std::vector<int> vecResult(static_cast<size_t>(nBatch));  // 20260319 ZJH 分配结果向量
    // 20260319 ZJH 调用 CPU argmax 内核
    CPUBackend::argmax(ct.floatDataPtr(), vecResult.data(), nBatch, nClasses);
    return vecResult;  // 20260319 ZJH 返回各行最大值索引
}

// =========================================================
// Phase 2 Part 2: Conv2d / BatchNorm2d / Pool / Flatten / Dropout
// =========================================================

// 20260319 ZJH tensorConv2d — 2D 卷积前向，支持自动微分
// input: [N, Cin, H, W]  weight: [Cout, Cin, KH, KW]  bias: [Cout]（可为空张量）
// 返回: [N, Cout, Hout, Wout]
Tensor tensorConv2d(Tensor input, Tensor weight, Tensor bias, int nStride, int nPad) {
    auto cInput = input.contiguous();   // 20260319 ZJH 确保输入连续
    auto cWeight = weight.contiguous(); // 20260319 ZJH 确保权重连续
    int nBatch = cInput.shape(0);       // 20260319 ZJH 批次大小
    int nCin = cInput.shape(1);         // 20260319 ZJH 输入通道数
    int nH = cInput.shape(2);           // 20260319 ZJH 输入高度
    int nW = cInput.shape(3);           // 20260319 ZJH 输入宽度
    int nCout = cWeight.shape(0);       // 20260319 ZJH 输出通道数
    int nKH = cWeight.shape(2);         // 20260319 ZJH 核高度
    int nKW = cWeight.shape(3);         // 20260319 ZJH 核宽度
    int nHout = (nH + 2 * nPad - nKH) / nStride + 1;  // 20260319 ZJH 输出高度
    int nWout = (nW + 2 * nPad - nKW) / nStride + 1;  // 20260319 ZJH 输出宽度

    auto result = Tensor::zeros({nBatch, nCout, nHout, nWout});  // 20260319 ZJH 分配输出张量

    // 20260319 ZJH 判断是否有偏置
    bool bHasBias = (bias.numel() > 0);
    const float* pBias = bHasBias ? bias.contiguous().floatDataPtr() : nullptr;

    CPUBackend::conv2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pBias,
                       result.mutableFloatDataPtr(),
                       nBatch, nCin, nH, nW, nCout, nKH, nKW, nStride, nPad);

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，创建 Conv2dBackward
    if (input.requiresGrad() || weight.requiresGrad()) {
        auto pBackward = std::make_shared<Conv2dBackward>();
        pBackward->m_savedInput = cInput;
        pBackward->m_savedWeight = cWeight;
        pBackward->m_nBatch = nBatch;
        pBackward->m_nCin = nCin;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_nCout = nCout;
        pBackward->m_nKH = nKH;
        pBackward->m_nKW = nKW;
        pBackward->m_nStride = nStride;
        pBackward->m_nPad = nPad;
        pBackward->m_bHasBias = bHasBias;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));   // 20260319 ZJH 输入边
        pBackward->m_vecInputEdges.push_back(makeEdge(weight, 0));  // 20260319 ZJH 权重边
        if (bHasBias) {
            pBackward->m_vecInputEdges.push_back(makeEdge(bias, 0));  // 20260319 ZJH 偏置边
        }
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH tensorBatchNorm2d — 批归一化前向，支持自动微分
// input: [N, C, H, W]  gamma/beta: [C]  runMean/runVar: [C]
Tensor tensorBatchNorm2d(Tensor input, Tensor gamma, Tensor beta,
                          Tensor& runMean, Tensor& runVar,
                          bool bTraining, float fEps, float fMomentum) {
    auto cInput = input.contiguous();
    auto cGamma = gamma.contiguous();
    auto cBeta = beta.contiguous();
    int nBatch = cInput.shape(0);
    int nChannels = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    auto result = Tensor::zeros(cInput.shapeVec());
    auto savedMean = Tensor::zeros({nChannels});   // 20260319 ZJH 保存均值
    auto savedInvStd = Tensor::zeros({nChannels}); // 20260319 ZJH 保存逆标准差

    CPUBackend::batchNorm2d(
        cInput.floatDataPtr(), result.mutableFloatDataPtr(),
        cGamma.floatDataPtr(), cBeta.floatDataPtr(),
        runMean.mutableFloatDataPtr(), runVar.mutableFloatDataPtr(),
        savedMean.mutableFloatDataPtr(), savedInvStd.mutableFloatDataPtr(),
        nBatch, nChannels, nH, nW, fEps, fMomentum, bTraining);

    // 20260319 ZJH AutoGrad
    if (input.requiresGrad() || gamma.requiresGrad()) {
        auto pBackward = std::make_shared<BatchNorm2dBackward>();
        pBackward->m_savedInput = cInput;
        pBackward->m_savedMean = savedMean;
        pBackward->m_savedInvStd = savedInvStd;
        pBackward->m_savedGamma = cGamma;
        pBackward->m_nBatch = nBatch;
        pBackward->m_nChannels = nChannels;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(gamma, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(beta, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH tensorMaxPool2d — 最大池化前向，支持自动微分
// input: [N, C, H, W]  返回: [N, C, Hout, Wout]
Tensor tensorMaxPool2d(Tensor input, int nKernelSize, int nStride, int nPad) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nChannels = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);
    int nHout = (nH + 2 * nPad - nKernelSize) / nStride + 1;
    int nWout = (nW + 2 * nPad - nKernelSize) / nStride + 1;

    auto result = Tensor::zeros({nBatch, nChannels, nHout, nWout});
    int nOutSize = nBatch * nChannels * nHout * nWout;
    std::vector<int> vecIndices(static_cast<size_t>(nOutSize));  // 20260319 ZJH 临时 int 索引

    CPUBackend::maxPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                           vecIndices.data(),
                           nBatch, nChannels, nH, nW,
                           nKernelSize, nKernelSize, nStride, nPad);

    // 20260319 ZJH AutoGrad
    if (input.requiresGrad()) {
        // 20260319 ZJH 将 int 索引转为 float 存储到 Tensor 中（避免引入 int Tensor）
        auto savedIndices = Tensor::zeros({nBatch, nChannels, nHout, nWout});
        float* pIdxFloat = savedIndices.mutableFloatDataPtr();
        for (int i = 0; i < nOutSize; ++i) {
            pIdxFloat[i] = static_cast<float>(vecIndices[static_cast<size_t>(i)]);
        }

        auto pBackward = std::make_shared<MaxPool2dBackward>();
        pBackward->m_savedIndices = savedIndices;
        pBackward->m_nBatch = nBatch;
        pBackward->m_nChannels = nChannels;
        pBackward->m_nHout = nHout;
        pBackward->m_nWout = nWout;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH tensorAvgPool2d — 平均池化前向，支持自动微分
// input: [N, C, H, W]  返回: [N, C, Hout, Wout]
Tensor tensorAvgPool2d(Tensor input, int nKernelSize, int nStride, int nPad) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nChannels = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);
    int nHout = (nH + 2 * nPad - nKernelSize) / nStride + 1;
    int nWout = (nW + 2 * nPad - nKernelSize) / nStride + 1;

    auto result = Tensor::zeros({nBatch, nChannels, nHout, nWout});
    CPUBackend::avgPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                           nBatch, nChannels, nH, nW,
                           nKernelSize, nKernelSize, nStride, nPad);

    // 20260319 ZJH AutoGrad
    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<AvgPool2dBackward>();
        pBackward->m_nBatch = nBatch;
        pBackward->m_nChannels = nChannels;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_nHout = nHout;
        pBackward->m_nWout = nWout;
        pBackward->m_nKH = nKernelSize;
        pBackward->m_nKW = nKernelSize;
        pBackward->m_nStride = nStride;
        pBackward->m_nPad = nPad;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH tensorFlatten — 将 [N, C, H, W, ...] 展平为 [N, -1]
// startDim: 从哪个维度开始展平（默认 1，保留 batch 维）
Tensor tensorFlatten(Tensor input, int nStartDim) {
    auto cInput = input.contiguous();
    auto vecShape = cInput.shapeVec();

    // 20260319 ZJH 计算新形状
    std::vector<int> vecNewShape;
    int nFlatSize = 1;
    for (int d = 0; d < static_cast<int>(vecShape.size()); ++d) {
        if (d < nStartDim) {
            vecNewShape.push_back(vecShape[static_cast<size_t>(d)]);
        } else {
            nFlatSize *= vecShape[static_cast<size_t>(d)];
        }
    }
    vecNewShape.push_back(nFlatSize);

    // 20260319 ZJH 使用 fromData 创建新张量（保证连续）
    auto result = Tensor::fromData(cInput.floatDataPtr(), vecNewShape);

    // 20260319 ZJH AutoGrad
    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<FlattenBackward>();
        pBackward->m_vecInputShape = vecShape;  // 20260319 ZJH 保存原始形状
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260319 ZJH tensorDropout — Dropout 前向，支持自动微分
// 训练时随机将元素置零，概率为 fProb，剩余元素放大 1/(1-p)
// 评估时直接透传
Tensor tensorDropout(Tensor input, float fProb, bool bTraining) {
    auto cInput = input.contiguous();

    if (!bTraining || fProb <= 0.0f) {
        // 20260319 ZJH 评估模式或概率为 0 时直接透传
        return Tensor::fromData(cInput.floatDataPtr(), cInput.shapeVec());
    }

    int nNumel = cInput.numel();
    auto result = Tensor::zeros(cInput.shapeVec());
    auto mask = Tensor::zeros(cInput.shapeVec());  // 20260319 ZJH 生成 mask

    // 20260319 ZJH 使用 thread_local 随机数生成器
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float fScale = 1.0f / (1.0f - fProb);  // 20260319 ZJH 放大因子
    const float* pInput = cInput.floatDataPtr();
    float* pResult = result.mutableFloatDataPtr();
    float* pMask = mask.mutableFloatDataPtr();

    for (int i = 0; i < nNumel; ++i) {
        if (dist(gen) >= fProb) {
            pMask[i] = fScale;      // 20260319 ZJH 保留并放大
            pResult[i] = pInput[i] * fScale;
        } else {
            pMask[i] = 0.0f;        // 20260319 ZJH 置零
            pResult[i] = 0.0f;
        }
    }

    // 20260319 ZJH AutoGrad
    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<DropoutBackward>();
        pBackward->m_savedMask = mask;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

}  // namespace df
