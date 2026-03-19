// 20260319 ZJH Tensor 运算模块 — 元素运算、matmul、reshape、transpose、slice、归约、自动微分集成
module;

#include <vector>
#include <stdexcept>
#include <memory>
#include <cassert>
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

}  // namespace df
