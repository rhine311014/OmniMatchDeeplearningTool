// 20260319 ZJH Tensor 运算模块 — 元素运算、matmul、reshape、transpose、slice、归约、自动微分集成
module;

#include <vector>
#include <stdexcept>
#include <memory>
#include <cassert>
#include <random>
#include <cmath>
#include "om_types.h"

export module om.engine.tensor_ops;

// 20260319 ZJH 导入依赖模块：存储层、张量类、CPU 计算内核、自动微分
import om.engine.tensor_storage;
import om.engine.tensor;
import om.hal.cpu_backend;
// 20260325 ZJH Phase 3: GPU-Resident 重写，导入 CUDA 后端用于设备调度
#ifdef OM_HAS_CUDA
import om.hal.cuda_backend;
#endif
import om.engine.autograd;

export namespace om {

// =========================================================
// 20260325 ZJH Phase 3: 设备调度辅助函数
// =========================================================

// 20260325 ZJH 判断张量是否驻留在 CUDA GPU 上，用于前向/反向运算的后端选择
inline bool isCudaTensor(const Tensor& t) {
    return t.device() == DeviceType::CUDA;  // 20260325 ZJH 查询底层 TensorStorage 的设备类型
}

// 20260325 ZJH 检查两个张量是否在同一设备上，不一致时抛出异常
// 所有涉及双张量运算（add/sub/mul/matmul 等）的函数入口都必须调用此检查
inline void checkSameDevice(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        // 20260325 ZJH 设备不一致，抛出详细错误信息（含设备类型编号）
        throw std::runtime_error("Tensor device mismatch: " +
            std::to_string(static_cast<int>(a.device())) + " vs " +
            std::to_string(static_cast<int>(b.device())));
    }
}

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
// 20260325 ZJH Phase 3: 添加设备调度，CUDA 张量调用 CUDABackend
Tensor tensorAdd(Tensor a, Tensor b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260319 ZJH 确保 a 为连续内存（非连续则拷贝）
    auto cb = b.contiguous();  // 20260319 ZJH 确保 b 为连续内存
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH CUDA 路径：直接操作 GPU 指针
        CUDABackend::add(ca.floatDataPtr(), cb.floatDataPtr(),
                         result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
#endif
    } else {
        // 20260319 ZJH CPU 路径：调用 CPU 逐元素加法内核
        CPUBackend::add(ca.floatDataPtr(), cb.floatDataPtr(),
                        result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorSub(Tensor a, Tensor b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::sub(ca.floatDataPtr(), cb.floatDataPtr(),
                         result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::sub(ca.floatDataPtr(), cb.floatDataPtr(),
                        result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorMul(Tensor a, Tensor b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::mul(ca.floatDataPtr(), cb.floatDataPtr(),
                         result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::mul(ca.floatDataPtr(), cb.floatDataPtr(),
                        result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    }

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
// 20260325 ZJH Phase 3: 添加设备调度（CUDABackend 暂无 div，GPU 张量临时回退 CPU）
Tensor tensorDiv(const Tensor& a, const Tensor& b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        // 20260327 ZJH GPU 路径：CUDABackend::div 直接在 GPU 上逐元素除法
        auto ca = a.contiguous();
        auto cb = b.contiguous();
        auto result = Tensor::zeros(ca.shapeVec(), DeviceType::CUDA);
        CUDABackend::div(ca.floatDataPtr(), cb.floatDataPtr(),
                         result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
        return result;
#endif
    }
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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorAddScalar(Tensor a, float fScalar) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::addScalar(ca.floatDataPtr(), fScalar,
                               result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::addScalar(ca.floatDataPtr(), fScalar,
                              result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorMulScalar(Tensor a, float fScalar) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::mulScalar(ca.floatDataPtr(), fScalar,
                               result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::mulScalar(ca.floatDataPtr(), fScalar,
                              result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorMatmul(Tensor a, Tensor b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260319 ZJH 确保 a 连续，matmul 需要行主序连续内存
    auto cb = b.contiguous();  // 20260319 ZJH 确保 b 连续
    int nM = ca.shape(0);  // 20260319 ZJH 行数（矩阵 A 的行）
    int nK = ca.shape(1);  // 20260319 ZJH 内维度（A 的列 = B 的行）
    int nN = cb.shape(1);  // 20260319 ZJH 列数（矩阵 B 的列）
    auto result = Tensor::zeros({nM, nN}, a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::matmul(ca.floatDataPtr(), cb.floatDataPtr(),
                            result.mutableFloatDataPtr(), nM, nK, nN);
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::matmul(ca.floatDataPtr(), cb.floatDataPtr(),
                           result.mutableFloatDataPtr(), nM, nK, nN);
    }

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
// 20260328 ZJH GPU 全路径：omCudaSum 结果直接写入 GPU 张量，零 D2H（除 .item() 的 4 字节）
Tensor tensorSum(Tensor t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存

#ifdef OM_HAS_CUDA
    if (isCudaTensor(t)) {
        // 20260328 ZJH CUDA 路径：result 在 GPU 上直接承接归约结果，避免整张量 D2H
        Tensor result = Tensor::zeros({1}, DeviceType::CUDA);  // 20260328 ZJH 单元素 GPU 输出张量
        CUDABackend::sum(ct.floatDataPtr(), result.mutableFloatDataPtr(),
                         static_cast<int>(ct.numel()));  // 20260328 ZJH GPU warp-shuffle 归约
        if (t.requiresGrad()) {
            auto pBackward = std::make_shared<SumBackward>();
            pBackward->m_vecInputShape = t.shapeVec();  // 20260328 ZJH 保存输入形状供 backward 广播
            pBackward->m_vecInputEdges.push_back(makeEdge(t, 0));
            result.setGradFnRaw(pBackward);
            result.setRequiresGrad(true);
        }
        return result;  // 20260328 ZJH 返回 GPU 单元素张量（调用 .item() 时才 4 字节 D2H）
    }
#endif

    // 20260319 ZJH CPU 路径
    float fSum = CPUBackend::sum(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
    auto result = Tensor::full({1}, fSum, t.device());  // 20260325 ZJH 标量张量保留在 CPU 上

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
// 20260325 ZJH Phase 3: GPU 张量临时迁移到 CPU 做归约（返回标量，开销可接受）
float tensorMax(const Tensor& t) {
    if (isCudaTensor(t)) {
        auto cpuT = t.cpu().contiguous();  // 20260325 ZJH GPU→CPU 拷贝
        return CPUBackend::max(cpuT.floatDataPtr(), static_cast<size_t>(cpuT.numel()));
    }
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存
    return CPUBackend::max(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
}

// 20260319 ZJH 全局最小值：返回张量所有元素中的最小值（不参与计算图）
// 20260325 ZJH Phase 3: GPU 张量临时迁移到 CPU 做归约
float tensorMin(const Tensor& t) {
    if (isCudaTensor(t)) {
        auto cpuT = t.cpu().contiguous();  // 20260325 ZJH GPU→CPU 拷贝
        return CPUBackend::min(cpuT.floatDataPtr(), static_cast<size_t>(cpuT.numel()));
    }
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
    // 20260327 ZJH 全 GPU 化修复：rootGrad 必须在 loss 所在设备上创建，
    //              否则 BCE 系列损失的首个 backward 函数检测到 CPU 梯度会走 CPU 回退路径，
    //              导致整条反向传播链全部在 CPU 上执行
    auto rootGrad = Tensor::ones({1}, loss.device());
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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorReLU(Tensor a) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::relu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                          static_cast<size_t>(result.numel()));
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::relu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                         static_cast<size_t>(result.numel()));
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorAddBias(Tensor a, Tensor bias) {
    checkSameDevice(a, bias);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260319 ZJH 确保输入连续
    auto cb = bias.contiguous();  // 20260319 ZJH 确保偏置连续
    int nBatch = ca.shape(0);  // 20260319 ZJH 批次大小
    int nCols = ca.shape(1);   // 20260319 ZJH 特征维度

    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH CUDABackend::addBias 签名为 (pData, pBias, pOut, nBatch, nChannels, nHW)
        // 对 [batch, cols] 2D 张量，nChannels=nCols, nHW=1
        CUDABackend::addBias(ca.floatDataPtr(), cb.floatDataPtr(),
                             result.mutableFloatDataPtr(), nBatch, nCols, 1);
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::addBias(ca.floatDataPtr(), cb.floatDataPtr(),
                            result.mutableFloatDataPtr(), nBatch, nCols);
    }

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
// 20260325 ZJH Phase 3: 添加设备调度（softmax 和 crossEntropy 分开处理）
Tensor tensorSoftmaxCrossEntropy(Tensor logits, const Tensor& targets) {
    checkSameDevice(logits, targets);  // 20260325 ZJH 确保两个张量在同一设备上
    auto cLogits = logits.contiguous();   // 20260319 ZJH 确保 logits 连续
    auto cTargets = targets.contiguous(); // 20260319 ZJH 确保 targets 连续
    int nBatch = cLogits.shape(0);    // 20260319 ZJH 批次大小
    int nClasses = cLogits.shape(1);  // 20260319 ZJH 类别数

    // 20260325 ZJH softmax 在输入设备上计算
    auto softmaxOut = Tensor::zeros({nBatch, nClasses}, logits.device());

#ifdef OM_HAS_CUDA
    if (isCudaTensor(logits)) {
        // 20260328 ZJH GPU 全路径：result 直接承接 crossEntropyForward 的 GPU 输出，
        //              不再 D2H→H2D 往返，result 为单元素 GPU 张量
        CUDABackend::softmax(cLogits.floatDataPtr(), softmaxOut.mutableFloatDataPtr(),
                             nBatch, nClasses);  // 20260327 ZJH GPU softmax
        auto result = Tensor::zeros({1}, DeviceType::CUDA);  // 20260328 ZJH GPU 标量输出
        CUDABackend::crossEntropyForward(softmaxOut.floatDataPtr(), cTargets.floatDataPtr(),
                                          result.mutableFloatDataPtr(), nBatch, nClasses);
        // 20260328 ZJH AutoGrad: result 直接作为 GPU 张量参与计算图
        if (logits.requiresGrad()) {
            auto pBackward = std::make_shared<SoftmaxCrossEntropyBackward>();
            pBackward->m_savedSoftmax = softmaxOut;
            pBackward->m_savedTargets = cTargets;
            pBackward->m_nBatch = nBatch;
            pBackward->m_nClasses = nClasses;
            pBackward->m_vecInputEdges.push_back(makeEdge(logits, 0));
            result.setGradFnRaw(pBackward);
            result.setRequiresGrad(true);
        }
        return result;  // 20260328 ZJH 返回 GPU 标量张量，零 D2H
    }
#endif

    // 20260319 ZJH CPU 路径
    CPUBackend::softmax(cLogits.floatDataPtr(), softmaxOut.mutableFloatDataPtr(),
                        nBatch, nClasses);
    float fLoss = CPUBackend::crossEntropy(softmaxOut.floatDataPtr(), cTargets.floatDataPtr(),
                                           nBatch, nClasses);

    // 20260319 ZJH 创建标量损失张量（CPU）
    auto result = Tensor::full({1}, fLoss, logits.device());

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
// 20260325 ZJH Phase 3: GPU 张量迁移到 CPU 做 argmax（返回 CPU int 向量）
std::vector<int> tensorArgmax(const Tensor& t) {
    // 20260325 ZJH argmax 结果为 CPU int 向量，GPU 张量先迁移到 CPU
    Tensor ct;
    if (isCudaTensor(t)) {
        ct = t.cpu().contiguous();  // 20260325 ZJH GPU→CPU 拷贝后连续化
    } else {
        ct = t.contiguous();  // 20260319 ZJH 确保输入连续
    }
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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorConv2d(Tensor input, Tensor weight, Tensor bias, int nStride, int nPad) {
    checkSameDevice(input, weight);  // 20260325 ZJH 确保输入和权重在同一设备上
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

    auto result = Tensor::zeros({nBatch, nCout, nHout, nWout}, input.device());  // 20260325 ZJH 在输入设备上分配输出张量

    // 20260319 ZJH 判断是否有偏置
    bool bHasBias = (bias.numel() > 0);
    const float* pBias = bHasBias ? bias.contiguous().floatDataPtr() : nullptr;

    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH CUDA 路径：CUDABackend::conv2d 参数顺序为 (padH, padW, strH, strW)
        CUDABackend::conv2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pBias,
                            result.mutableFloatDataPtr(),
                            nBatch, nCin, nH, nW, nCout, nKH, nKW,
                            nPad, nPad, nStride, nStride);
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::conv2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pBias,
                           result.mutableFloatDataPtr(),
                           nBatch, nCin, nH, nW, nCout, nKH, nKW, nStride, nPad);
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
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

    auto result = Tensor::zeros(cInput.shapeVec(), input.device());  // 20260325 ZJH 在输入设备上分配
    auto savedMean = Tensor::zeros({nChannels}, input.device());   // 20260325 ZJH 保存均值
    auto savedInvStd = Tensor::zeros({nChannels}, input.device()); // 20260325 ZJH 保存逆标准差

    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        if (runMean.isCpu()) runMean = runMean.cuda();
        if (runVar.isCpu())  runVar = runVar.cuda();
        if (cGamma.isCpu()) cGamma = cGamma.cuda();
        if (cBeta.isCpu())  cBeta = cBeta.cuda();

        int nHW = nH * nW;
        CUDABackend::batchNorm2d(
            cInput.floatDataPtr(), cGamma.floatDataPtr(), cBeta.floatDataPtr(),
            result.mutableFloatDataPtr(),
            runMean.mutableFloatDataPtr(), runVar.mutableFloatDataPtr(),
            savedMean.mutableFloatDataPtr(), savedInvStd.mutableFloatDataPtr(),
            nBatch, nChannels, nHW, fEps, fMomentum, bTraining);
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::batchNorm2d(
            cInput.floatDataPtr(), result.mutableFloatDataPtr(),
            cGamma.floatDataPtr(), cBeta.floatDataPtr(),
            runMean.mutableFloatDataPtr(), runVar.mutableFloatDataPtr(),
            savedMean.mutableFloatDataPtr(), savedInvStd.mutableFloatDataPtr(),
            nBatch, nChannels, nH, nW, fEps, fMomentum, bTraining);
    }

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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorMaxPool2d(Tensor input, int nKernelSize, int nStride, int nPad) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nChannels = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);
    int nHout = (nH + 2 * nPad - nKernelSize) / nStride + 1;
    int nWout = (nW + 2 * nPad - nKernelSize) / nStride + 1;

    auto result = Tensor::zeros({nBatch, nChannels, nHout, nWout}, input.device());  // 20260325 ZJH 在输入设备上分配
    int nOutSize = nBatch * nChannels * nHout * nWout;  // 20260319 ZJH 输出元素总数

    // 20260327 ZJH GPU 全驻留：索引直接在 GPU 上以 int（reinterpret 为 float Tensor）存储，
    //              反向传播时无需 D2H 拷贝，MaxPool2dBackward 直接 reinterpret 回 int*
    Tensor savedIndices;  // 20260327 ZJH 索引张量，GPU 或 CPU 取决于输入设备

    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        // 20260327 ZJH GPU 路径：索引直接留在 GPU 上（int 与 float 均为 4 字节）
        savedIndices = Tensor::zeros({nBatch, nChannels, nHout, nWout}, DeviceType::CUDA);
        CUDABackend::maxPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                               reinterpret_cast<int*>(savedIndices.mutableFloatDataPtr()),
                               nBatch, nChannels, nH, nW,
                               nKernelSize, nKernelSize, nStride, nPad);
#endif
    } else {
        // 20260319 ZJH CPU 路径：先用临时 int 向量存索引，再转为 float Tensor
        std::vector<int> vecIndices(static_cast<size_t>(nOutSize));
        CPUBackend::maxPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                               vecIndices.data(),
                               nBatch, nChannels, nH, nW,
                               nKernelSize, nKernelSize, nStride, nPad);
        // 20260327 ZJH CPU 索引转为 float 存储（保持 CPU 路径兼容性）
        savedIndices = Tensor::zeros({nBatch, nChannels, nHout, nWout});
        float* pIdxFloat = savedIndices.mutableFloatDataPtr();
        for (int i = 0; i < nOutSize; ++i) {
            pIdxFloat[i] = static_cast<float>(vecIndices[static_cast<size_t>(i)]);
        }
    }

    // 20260319 ZJH AutoGrad
    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<MaxPool2dBackward>();
        pBackward->m_savedIndices = savedIndices;  // 20260327 ZJH GPU 张量直接存 GPU，零 D2H
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
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorAvgPool2d(Tensor input, int nKernelSize, int nStride, int nPad) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nChannels = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);
    int nHout = (nH + 2 * nPad - nKernelSize) / nStride + 1;
    int nWout = (nW + 2 * nPad - nKernelSize) / nStride + 1;

    auto result = Tensor::zeros({nBatch, nChannels, nHout, nWout}, input.device());  // 20260325 ZJH 在输入设备上分配
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        CUDABackend::avgPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                               nBatch, nChannels, nH, nW,
                               nKernelSize, nKernelSize, nStride, nPad);
#endif
    } else {
        // 20260319 ZJH CPU 路径
        CPUBackend::avgPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                               nBatch, nChannels, nH, nW,
                               nKernelSize, nKernelSize, nStride, nPad);
    }

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
    // 20260325 ZJH Phase 3: 已连续化的张量直接用 reshape 零拷贝视图（适用于 CPU 和 GPU）
    auto result = tensorReshape(cInput, vecNewShape);

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
// 20260325 ZJH Phase 3: 添加设备调度
// 20260328 ZJH GPU 路径优化：使用 fused kernel 在 GPU 端生成 mask + 应用，消除 CPU→GPU mask 传输
Tensor tensorDropout(Tensor input, float fProb, bool bTraining) {
    auto cInput = input.contiguous();  // 20260319 ZJH 确保输入连续

    if (!bTraining || fProb <= 0.0f) {
        // 20260319 ZJH 评估模式或概率为 0 时直接透传
        // 20260325 ZJH Phase 3: GPU 张量使用 reshape 零拷贝，避免 fromData 访问 GPU 指针
        return tensorReshape(cInput, cInput.shapeVec());
    }

    int nNumel = cInput.numel();  // 20260319 ZJH 元素总数
    auto result = Tensor::zeros(cInput.shapeVec(), input.device());  // 20260325 ZJH 在输入设备上分配输出

    // 20260319 ZJH 使用 thread_local 随机数生成器（GPU 路径仅取单个种子，CPU 路径逐元素采样）
    static thread_local std::mt19937 gen(std::random_device{}());

    // 20260328 ZJH mask 张量：GPU 路径在 GPU 上直接生成，CPU 路径在 CPU 上生成
    Tensor mask;

    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        // 20260328 ZJH GPU 路径（优化）：fused kernel 在 GPU 端用 SplitMix64 生成 mask 并乘输入
        // 仅从 CPU 传一个 64-bit 种子，消除 N*4 字节的 CPU→GPU mask 传输
        float fKeepProb = 1.0f - fProb;  // 20260328 ZJH 保留概率
        unsigned long long nSeed = gen();  // 20260328 ZJH 从 CPU 端 mt19937 取单个随机种子
        mask = Tensor::zeros(cInput.shapeVec(), DeviceType::CUDA);  // 20260328 ZJH mask 直接分配在 GPU 上
        CUDABackend::dropoutForward(
            cInput.floatDataPtr(),               // 20260328 ZJH GPU 输入指针
            result.mutableFloatDataPtr(),         // 20260328 ZJH GPU 输出指针
            mask.mutableFloatDataPtr(),           // 20260328 ZJH GPU mask 输出指针（0 或 1/keepProb）
            nNumel, fKeepProb, nSeed);            // 20260328 ZJH 元素数、保留概率、种子
#endif
    } else {
        // 20260319 ZJH CPU 路径：逐元素生成 mask 并应用（保持原有实现）
        mask = Tensor::zeros(cInput.shapeVec());  // 20260319 ZJH mask 在 CPU 上分配
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // 20260319 ZJH 均匀分布 [0,1)
        float fScale = 1.0f / (1.0f - fProb);  // 20260319 ZJH 倒缩放因子
        float* pMask = mask.mutableFloatDataPtr();  // 20260319 ZJH CPU mask 指针
        // 20260319 ZJH 在 CPU 上生成 dropout mask（0 或 1/(1-p)）
        for (int i = 0; i < nNumel; ++i) {
            if (dist(gen) >= fProb) {
                pMask[i] = fScale;      // 20260319 ZJH 保留并放大
            } else {
                pMask[i] = 0.0f;        // 20260319 ZJH 置零
            }
        }
        // 20260319 ZJH 逐元素乘 mask
        const float* pInput = cInput.floatDataPtr();   // 20260319 ZJH 输入指针
        float* pResult = result.mutableFloatDataPtr();  // 20260319 ZJH 输出指针
        for (int i = 0; i < nNumel; ++i) {
            pResult[i] = pInput[i] * pMask[i];  // 20260319 ZJH 应用 dropout mask
        }
    }

    // 20260319 ZJH AutoGrad: 注册反向节点（mask 已包含缩放因子，backward 直接乘梯度即可）
    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<DropoutBackward>();  // 20260319 ZJH 创建反向节点
        pBackward->m_savedMask = mask;  // 20260319 ZJH 保存 mask（GPU 路径已在 GPU 上，无需迁移）
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));  // 20260319 ZJH 连接输入边
        result.setGradFnRaw(pBackward);   // 20260319 ZJH 设置结果的梯度函数
        result.setRequiresGrad(true);     // 20260319 ZJH 标记结果需要梯度
    }

    return result;  // 20260328 ZJH 返回 dropout 结果
}

// =========================================================
// Phase 3: 新增激活函数和运算 — Sigmoid / LeakyReLU / Upsample / Concat / ConvTranspose2d
// =========================================================

// 20260320 ZJH tensorSigmoid — Sigmoid 激活函数，支持自动微分
// 前向: out[i] = 1 / (1 + exp(-in[i]))
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorSigmoid(Tensor a) {
    auto ca = a.contiguous();  // 20260320 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::sigmoid(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                              static_cast<size_t>(result.numel()));
#endif
    } else {
        CPUBackend::sigmoid(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                             static_cast<size_t>(result.numel()));
    }

    // 20260320 ZJH AutoGrad: 保存输出（非输入），sigmoid 反向需要输出值
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<SigmoidBackwardFn>();
        pBackward->m_savedOutput = result;  // 20260320 ZJH 保存输出用于反向
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorLeakyReLU — LeakyReLU 激活函数，支持自动微分
// 前向: out[i] = in[i] > 0 ? in[i] : slope * in[i]
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorLeakyReLU(Tensor a, float fSlope = 0.01f) {
    auto ca = a.contiguous();  // 20260320 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::leakyRelu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                                static_cast<size_t>(result.numel()), fSlope);
#endif
    } else {
        CPUBackend::leakyRelu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                               static_cast<size_t>(result.numel()), fSlope);
    }

    // 20260320 ZJH AutoGrad: 保存输入和斜率
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<LeakyReLUBackwardFn>();
        pBackward->m_savedInput = ca;  // 20260320 ZJH 保存输入
        pBackward->m_fSlope = fSlope;  // 20260320 ZJH 保存斜率
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorUpsampleBilinear — 双线性上采样，支持自动微分
// input: [N, C, H, W] -> output: [N, C, H*scale, W*scale]
Tensor tensorUpsampleBilinear(Tensor input, int nScale) {
    auto cInput = input.contiguous();  // 20260320 ZJH 确保输入连续
    int nBatch = cInput.shape(0);      // 20260320 ZJH 批次大小
    int nChannels = cInput.shape(1);   // 20260320 ZJH 通道数
    int nH = cInput.shape(2);          // 20260320 ZJH 输入高度
    int nW = cInput.shape(3);          // 20260320 ZJH 输入宽度
    int nHout = nH * nScale;           // 20260320 ZJH 输出高度
    int nWout = nW * nScale;           // 20260320 ZJH 输出宽度

    auto result = Tensor::zeros({nBatch, nChannels, nHout, nWout}, input.device());  // 20260325 ZJH 在输入设备上分配
    // 20260327 ZJH GPU 路径：直接在 GPU 上双线性插值，零 D2H
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        CUDABackend::upsampleBilinear(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                                       nBatch, nChannels, nH, nW, nHout, nWout);
#endif
    } else {
        CPUBackend::upsampleBilinear(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                                      nBatch, nChannels, nH, nW, nScale);
    }

    // 20260320 ZJH AutoGrad
    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<UpsampleBilinearBackwardFn>();
        pBackward->m_nBatch = nBatch;
        pBackward->m_nChannels = nChannels;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_nScale = nScale;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorConcatChannels — 沿通道维度拼接两个张量，支持自动微分
// a: [N, C1, H, W]  b: [N, C2, H, W]  -> output: [N, C1+C2, H, W]
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorConcatChannels(Tensor a, Tensor b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260320 ZJH 确保 a 连续
    auto cb = b.contiguous();  // 20260320 ZJH 确保 b 连续
    int nBatch = ca.shape(0);  // 20260320 ZJH 批次大小
    int nC1 = ca.shape(1);    // 20260320 ZJH a 的通道数
    int nC2 = cb.shape(1);    // 20260320 ZJH b 的通道数
    int nH = ca.shape(2);     // 20260320 ZJH 高度
    int nW = ca.shape(3);     // 20260320 ZJH 宽度

    auto result = Tensor::zeros({nBatch, nC1 + nC2, nH, nW}, a.device());  // 20260325 ZJH 在输入设备上分配
    // 20260327 ZJH GPU 路径：直接在 GPU 上拼接通道，零 D2H
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        int nHW = nH * nW;  // 20260327 ZJH 空间维度大小
        CUDABackend::concatChannels(ca.floatDataPtr(), cb.floatDataPtr(),
                                     result.mutableFloatDataPtr(),
                                     nBatch, nC1, nC2, nHW);
#endif
    } else {
        CPUBackend::concatChannels(ca.floatDataPtr(), cb.floatDataPtr(),
                                    result.mutableFloatDataPtr(),
                                    nBatch, nC1, nC2, nH, nW);
    }

    // 20260320 ZJH AutoGrad
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<ConcatChannelsBackwardFn>();
        pBackward->m_nBatch = nBatch;
        pBackward->m_nC1 = nC1;
        pBackward->m_nC2 = nC2;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(b, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorConvTranspose2d — 转置卷积前向，支持自动微分
// input: [N, Cin, Hin, Win]  weight: [Cin, Cout, KH, KW]  bias: [Cout]（可为空）
// 返回: [N, Cout, Hout, Wout]，Hout = (Hin-1)*stride - 2*pad + KH
Tensor tensorConvTranspose2d(Tensor input, Tensor weight, Tensor bias, int nStride, int nPad) {
    auto cInput = input.contiguous();    // 20260320 ZJH 确保输入连续
    auto cWeight = weight.contiguous();  // 20260320 ZJH 确保权重连续
    int nBatch = cInput.shape(0);        // 20260320 ZJH 批次大小
    int nCin = cInput.shape(1);          // 20260320 ZJH 输入通道数
    int nHin = cInput.shape(2);          // 20260320 ZJH 输入高度
    int nWin = cInput.shape(3);          // 20260320 ZJH 输入宽度
    int nCout = cWeight.shape(1);        // 20260320 ZJH 输出通道数（注意：权重形状 [Cin, Cout, KH, KW]）
    int nKH = cWeight.shape(2);          // 20260320 ZJH 核高度
    int nKW = cWeight.shape(3);          // 20260320 ZJH 核宽度
    int nHout = (nHin - 1) * nStride - 2 * nPad + nKH;  // 20260320 ZJH 输出高度
    int nWout = (nWin - 1) * nStride - 2 * nPad + nKW;  // 20260320 ZJH 输出宽度

    auto result = Tensor::zeros({nBatch, nCout, nHout, nWout}, input.device());  // 20260325 ZJH 在输入设备上分配

    bool bHasBias = (bias.numel() > 0);  // 20260320 ZJH 判断是否有偏置
    const float* pBias = bHasBias ? bias.contiguous().floatDataPtr() : nullptr;

    // 20260327 ZJH GPU 路径：CUDABackend::convTranspose2d scatter 策略
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        const float* pGpuBias = bHasBias ? bias.contiguous().floatDataPtr() : nullptr;
        CUDABackend::convTranspose2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pGpuBias,
                                      result.mutableFloatDataPtr(),
                                      nBatch, nCin, nHin, nWin, nCout, nKH, nKW, nStride, nPad);
#endif
    } else {
        CPUBackend::convTranspose2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pBias,
                                     result.mutableFloatDataPtr(),
                                     nBatch, nCin, nHin, nWin, nCout, nKH, nKW, nStride, nPad);
    }

    // 20260320 ZJH AutoGrad
    if (input.requiresGrad() || weight.requiresGrad()) {
        auto pBackward = std::make_shared<ConvTranspose2dBackwardFn>();
        pBackward->m_savedInput = cInput;
        pBackward->m_savedWeight = cWeight;
        pBackward->m_nBatch = nBatch;
        pBackward->m_nCin = nCin;
        pBackward->m_nHin = nHin;
        pBackward->m_nWin = nWin;
        pBackward->m_nCout = nCout;
        pBackward->m_nKH = nKH;
        pBackward->m_nKW = nKW;
        pBackward->m_nStride = nStride;
        pBackward->m_nPad = nPad;
        pBackward->m_bHasBias = bHasBias;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(weight, 0));
        if (bHasBias) {
            pBackward->m_vecInputEdges.push_back(makeEdge(bias, 0));
        }
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorBCEWithLogitsLoss — 二元交叉熵损失（含 sigmoid），支持自动微分
// logits: [N, ...] 原始 logits，targets: [N, ...] 二元目标（0/1）
// 返回: 标量损失张量
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorBCEWithLogitsLoss(Tensor logits, const Tensor& targets) {
    checkSameDevice(logits, targets);  // 20260325 ZJH 确保两个张量在同一设备上
    auto cLogits = logits.contiguous();
    auto cTargets = targets.contiguous();
    int nCount = cLogits.numel();  // 20260320 ZJH 元素总数

#ifdef OM_HAS_CUDA
    if (isCudaTensor(logits)) {
        // 20260328 ZJH GPU 全路径：BCE 求和 + GPU 标量除法，result 直接作为 GPU 张量返回，零 D2H
        auto result = Tensor::zeros({1}, DeviceType::CUDA);  // 20260328 ZJH GPU 单元素输出
        CUDABackend::bceWithLogitsForward(cLogits.floatDataPtr(), cTargets.floatDataPtr(),
                                           result.mutableFloatDataPtr(), nCount);
        // 20260328 ZJH GPU 原地标量乘（除以 nCount = 乘以 1/nCount），避免 D2H
        float fInvN = 1.0f / static_cast<float>(nCount);
        CUDABackend::mulScalar(result.floatDataPtr(), fInvN, result.mutableFloatDataPtr(), 1);
        if (logits.requiresGrad()) {
            auto pBackward = std::make_shared<BCEWithLogitsBackwardFn>();
            pBackward->m_savedLogits = cLogits;
            pBackward->m_savedTargets = cTargets;
            pBackward->m_nCount = nCount;
            pBackward->m_vecInputEdges.push_back(makeEdge(logits, 0));
            result.setGradFnRaw(pBackward);
            result.setRequiresGrad(true);
        }
        return result;  // 20260328 ZJH 返回 GPU 标量张量，零 D2H
    }
#endif

    // 20260319 ZJH CPU 路径
    float fLoss = CPUBackend::bceWithLogits(cLogits.floatDataPtr(), cTargets.floatDataPtr(), nCount);
    auto result = Tensor::full({1}, fLoss, logits.device());  // 20260325 ZJH 在 CPU 上创建标量损失

    // 20260320 ZJH AutoGrad
    if (logits.requiresGrad()) {
        auto pBackward = std::make_shared<BCEWithLogitsBackwardFn>();
        pBackward->m_savedLogits = cLogits;
        pBackward->m_savedTargets = cTargets;
        pBackward->m_nCount = nCount;
        pBackward->m_vecInputEdges.push_back(makeEdge(logits, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// =========================================================
// Phase 5: GELU / SiLU / LayerNorm / AdaptiveAvgPool2d / BatchedMatmul
// =========================================================

// 20260320 ZJH tensorGELU — GELU 激活函数，支持自动微分
// GELU(x) ≈ 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x^3)))
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorGELU(Tensor a) {
    auto ca = a.contiguous();  // 20260320 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::gelu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                          static_cast<size_t>(result.numel()));
#endif
    } else {
        CPUBackend::gelu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                         static_cast<size_t>(result.numel()));
    }

    // 20260320 ZJH AutoGrad
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<GELUBackwardFn>();
        pBackward->m_savedInput = ca;  // 20260320 ZJH 保存输入
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }
    return result;
}

// 20260320 ZJH tensorSiLU — SiLU (Swish) 激活函数，支持自动微分
// SiLU(x) = x * sigmoid(x)
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorSiLU(Tensor a) {
    auto ca = a.contiguous();  // 20260320 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::silu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                          static_cast<size_t>(result.numel()));
#endif
    } else {
        CPUBackend::silu(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                         static_cast<size_t>(result.numel()));
    }

    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<SiLUBackwardFn>();
        pBackward->m_savedInput = ca;
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }
    return result;
}

// 20260320 ZJH tensorLayerNorm — LayerNorm 前向，支持自动微分
// input: [batch, dim]  gamma/beta: [dim]
// 返回: [batch, dim] 归一化后的张量
Tensor tensorLayerNorm(Tensor input, Tensor gamma, Tensor beta, float fEps = 1e-5f) {
    auto cInput = input.contiguous();
    auto cGamma = gamma.contiguous();
    auto cBeta = beta.contiguous();
    int nBatch = cInput.shape(0);  // 20260320 ZJH 批次大小
    int nDim = cInput.shape(1);    // 20260320 ZJH 归一化维度

    auto result = Tensor::zeros(cInput.shapeVec(), input.device());  // 20260325 ZJH 在输入设备上分配
    auto savedMean = Tensor::zeros({nBatch}, input.device());     // 20260325 ZJH 保存均值
    auto savedInvStd = Tensor::zeros({nBatch}, input.device());   // 20260325 ZJH 保存逆标准差

    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        CUDABackend::layerNorm(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                                cGamma.floatDataPtr(), cBeta.floatDataPtr(),
                                savedMean.mutableFloatDataPtr(), savedInvStd.mutableFloatDataPtr(),
                                nBatch, nDim, fEps);
#endif
    } else {
        CPUBackend::layerNorm(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                               cGamma.floatDataPtr(), cBeta.floatDataPtr(),
                               savedMean.mutableFloatDataPtr(), savedInvStd.mutableFloatDataPtr(),
                               nBatch, nDim, fEps);
    }

    // 20260320 ZJH AutoGrad
    if (input.requiresGrad() || gamma.requiresGrad()) {
        auto pBackward = std::make_shared<LayerNormBackwardFn>();
        pBackward->m_savedInput = cInput;
        pBackward->m_savedMean = savedMean;
        pBackward->m_savedInvStd = savedInvStd;
        pBackward->m_savedGamma = cGamma;
        pBackward->m_nBatch = nBatch;
        pBackward->m_nDim = nDim;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(gamma, 0));
        pBackward->m_vecInputEdges.push_back(makeEdge(beta, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorAdaptiveAvgPool2d — 自适应平均池化，支持自动微分
// input: [N, C, H, W] -> output: [N, C, outH, outW]
Tensor tensorAdaptiveAvgPool2d(Tensor input, int nOutH, int nOutW) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nChannels = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    auto result = Tensor::zeros({nBatch, nChannels, nOutH, nOutW}, input.device());  // 20260325 ZJH 在输入设备上分配
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(input)) {
#ifdef OM_HAS_CUDA
        CUDABackend::adaptiveAvgPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                                        nBatch, nChannels, nH, nW, nOutH, nOutW);
#endif
    } else {
        CPUBackend::adaptiveAvgPool2d(cInput.floatDataPtr(), result.mutableFloatDataPtr(),
                                       nBatch, nChannels, nH, nW, nOutH, nOutW);
    }

    if (input.requiresGrad()) {
        auto pBackward = std::make_shared<AdaptiveAvgPool2dBackwardFn>();
        pBackward->m_nBatch = nBatch;
        pBackward->m_nChannels = nChannels;
        pBackward->m_nH = nH;
        pBackward->m_nW = nW;
        pBackward->m_nOutH = nOutH;
        pBackward->m_nOutW = nOutW;
        pBackward->m_vecInputEdges.push_back(makeEdge(input, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260320 ZJH tensorBatchedMatmul — 批量矩阵乘法（不含自动微分，用于注意力推理）
// A: [batch, M, K]  B: [batch, K, N]  -> C: [batch, M, N]
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorBatchedMatmul(const Tensor& a, const Tensor& b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();
    auto cb = b.contiguous();
    int nBatch = ca.shape(0);
    int nM = ca.shape(1);
    int nK = ca.shape(2);
    int nN = cb.shape(2);

    auto result = Tensor::zeros({nBatch, nM, nN}, a.device());  // 20260325 ZJH 在输入设备上分配
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::batchedMatmul(ca.floatDataPtr(), cb.floatDataPtr(),
                                    result.mutableFloatDataPtr(), nBatch, nM, nK, nN);
#endif
    } else {
        CPUBackend::batchedMatmul(ca.floatDataPtr(), cb.floatDataPtr(),
                                   result.mutableFloatDataPtr(), nBatch, nM, nK, nN);
    }
    return result;
}

// 20260320 ZJH tensorTranspose2dBatched — 批量转置最后两维
// input: [batch, M, N] -> output: [batch, N, M]
// 20260325 ZJH Phase 3: 添加设备调度
// 20260328 ZJH 添加 autograd 注册：transpose 的反向就是再次转置
Tensor tensorTranspose2dBatched(const Tensor& t) {
    auto ct = t.contiguous();  // 20260320 ZJH 确保输入内存连续
    int nBatch = ct.shape(0);  // 20260320 ZJH 批次大小
    int nM = ct.shape(1);      // 20260320 ZJH 原始行数
    int nN = ct.shape(2);      // 20260320 ZJH 原始列数

    auto result = Tensor::zeros({nBatch, nN, nM}, t.device());  // 20260325 ZJH 在输入设备上分配
    if (isCudaTensor(t)) {
#ifdef OM_HAS_CUDA
        // 20260325 ZJH GPU 路径：逐 batch 调用 CUDABackend::transpose
        const float* pIn = ct.floatDataPtr();       // 20260325 ZJH 输入 GPU 指针
        float* pOut = result.mutableFloatDataPtr();  // 20260325 ZJH 输出 GPU 指针
        int nMN = nM * nN;  // 20260325 ZJH 单矩阵元素数
        for (int b = 0; b < nBatch; ++b) {
            CUDABackend::transpose(pIn + b * nMN, pOut + b * nMN, nM, nN);  // 20260325 ZJH [M,N]->[N,M]
        }
#endif
    } else {
        // 20260319 ZJH CPU 路径
        const float* pIn = ct.floatDataPtr();       // 20260319 ZJH 输入 CPU 指针
        float* pOut = result.mutableFloatDataPtr();  // 20260319 ZJH 输出 CPU 指针
        int nMN = nM * nN;  // 20260319 ZJH 单矩阵元素数
        for (int b = 0; b < nBatch; ++b) {
            CPUBackend::transpose2d(pIn + b * nMN, pOut + b * nMN, nM, nN);  // 20260319 ZJH [M,N]->[N,M]
        }
    }

    // 20260328 ZJH AutoGrad: 注册批量转置反向节点（转置是自身的逆运算）
    if (t.requiresGrad()) {
        auto pBackward = std::make_shared<Transpose2dBatchedBackwardFn>();  // 20260328 ZJH 创建反向节点
        pBackward->m_vecOrigShape = ct.shapeVec();  // 20260328 ZJH 保存原始形状 [batch, M, N]
        pBackward->m_vecInputEdges.push_back(makeEdge(ct, 0));  // 20260328 ZJH 连接输入边
        result.setGradFnRaw(pBackward);   // 20260328 ZJH 设置结果的梯度函数
        result.setRequiresGrad(true);     // 20260328 ZJH 标记结果需要梯度
    }

    return result;  // 20260328 ZJH 返回转置结果 [batch, N, M]
}

// 20260320 ZJH tensorSoftmaxLastDim — 沿最后一维做 softmax（用于注意力权重）
// input: [*, seq_len]  -> output: [*, seq_len]
// 20260325 ZJH Phase 3: 添加设备调度
// 20260328 ZJH 添加 autograd 注册：保存 softmax 输出用于反向传播
Tensor tensorSoftmaxLastDim(const Tensor& t) {
    auto ct = t.contiguous();
    auto vecShape = ct.shapeVec();
    int nLastDim = vecShape.back();  // 20260320 ZJH 最后一维大小
    int nOuter = ct.numel() / nLastDim;  // 20260320 ZJH 外层维度积

    auto result = Tensor::zeros(vecShape, t.device());  // 20260325 ZJH 在输入设备上分配
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(t)) {
#ifdef OM_HAS_CUDA
        CUDABackend::softmax(ct.floatDataPtr(), result.mutableFloatDataPtr(), nOuter, nLastDim);
#endif
    } else {
        // 20260320 ZJH CPU 路径
        CPUBackend::softmax(ct.floatDataPtr(), result.mutableFloatDataPtr(), nOuter, nLastDim);
    }

    // 20260328 ZJH AutoGrad: 注册 softmax 反向传播节点
    if (t.requiresGrad()) {
        auto pBackward = std::make_shared<SoftmaxLastDimBackwardFn>();  // 20260328 ZJH 创建反向节点
        pBackward->m_savedOutput = result;  // 20260328 ZJH 保存 softmax 输出用于反向
        pBackward->m_nLastDim = nLastDim;  // 20260328 ZJH 保存最后一维大小
        // 20260328 ZJH 注意：makeEdge 需要非 const 引用，使用 ct（已 contiguous 的输入）
        pBackward->m_vecInputEdges.push_back(makeEdge(ct, 0));  // 20260328 ZJH 连接输入边
        result.setGradFnRaw(pBackward);  // 20260328 ZJH 设置结果的梯度函数
        result.setRequiresGrad(true);  // 20260328 ZJH 标记结果需要梯度
    }

    return result;
}

// =========================================================
// Phase 5B: LSTM / CRNN / 实例分割所需运算
// =========================================================

// 20260321 ZJH tensorTanh — tanh 激活函数，支持自动微分
// 前向: out[i] = tanh(in[i])，LSTM cell candidate 和门控的核心激活
// 20260325 ZJH Phase 3: 添加设备调度
Tensor tensorTanh(Tensor a) {
    auto ca = a.contiguous();  // 20260321 ZJH 确保输入连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260325 ZJH 在输入设备上分配输出张量
    // 20260325 ZJH 根据设备类型选择后端
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::tanhForward(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                                  static_cast<size_t>(result.numel()));
#endif
    } else {
        CPUBackend::tanhForward(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                                 static_cast<size_t>(result.numel()));
    }

    // 20260321 ZJH AutoGrad: 保存输出，tanh 反向需要输出值
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<TanhBackwardFn>();
        pBackward->m_savedOutput = result;  // 20260321 ZJH 保存输出用于反向
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));
        result.setGradFnRaw(pBackward);
        result.setRequiresGrad(true);
    }

    return result;
}

// 20260321 ZJH tensorClip — 值裁剪（CTC loss 中防止 log(0)）
// 20260325 ZJH Phase 3: 添加设备调度
// 20260328 ZJH 添加 autograd 注册：保存输入和裁剪边界用于反向传播
Tensor tensorClip(const Tensor& a, float fMin, float fMax) {
    auto ca = a.contiguous();  // 20260327 ZJH 确保连续
    auto result = Tensor::zeros(ca.shapeVec(), a.device());  // 20260327 ZJH 在输入设备上分配
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        // 20260327 ZJH GPU 路径：直接在 GPU 上裁剪
        CUDABackend::clip(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                          static_cast<size_t>(result.numel()), fMin, fMax);
#endif
    } else {
        CPUBackend::clipForward(ca.floatDataPtr(), result.mutableFloatDataPtr(),
                                 static_cast<size_t>(result.numel()), fMin, fMax);
    }

    // 20260328 ZJH AutoGrad: 注册 clip 反向传播节点
    if (a.requiresGrad()) {
        auto pBackward = std::make_shared<ClipBackwardFn>();  // 20260328 ZJH 创建反向节点
        pBackward->m_savedInput = ca;  // 20260328 ZJH 保存输入用于判断梯度是否通过
        pBackward->m_fMin = fMin;  // 20260328 ZJH 保存裁剪下界
        pBackward->m_fMax = fMax;  // 20260328 ZJH 保存裁剪上界
        pBackward->m_vecInputEdges.push_back(makeEdge(ca, 0));  // 20260328 ZJH 连接输入边
        result.setGradFnRaw(pBackward);  // 20260328 ZJH 设置结果的梯度函数
        result.setRequiresGrad(true);  // 20260328 ZJH 标记结果需要梯度
    }

    return result;
}

// 20260321 ZJH tensorSliceLastDim — 沿最后一维切片 [start, end)
// 用于 LSTM 将 [batch, 4*hidden] 拆为 4 个 [batch, hidden] 的门向量
// 20260325 ZJH Phase 3: 添加设备调度
// 20260328 ZJH GPU 直接切片（零 D2H）+ autograd 反向传播
Tensor tensorSliceLastDim(Tensor t, int nStart, int nEnd) {
    auto ct = t.contiguous();  // 20260328 ZJH 确保输入连续
    auto vecShape = ct.shapeVec();  // 20260321 ZJH 获取形状
    int nNDim = static_cast<int>(vecShape.size());  // 20260321 ZJH 维度数
    int nLastDim = vecShape[nNDim - 1];  // 20260321 ZJH 最后一维大小
    int nSliceLen = nEnd - nStart;  // 20260321 ZJH 切片长度
    int nOuter = ct.numel() / nLastDim;  // 20260321 ZJH 外层元素数

    // 20260321 ZJH 构建输出形状
    auto vecOutShape = vecShape;
    vecOutShape[nNDim - 1] = nSliceLen;

    // 20260328 ZJH 在输入设备上分配输出张量
    auto result = Tensor::zeros(vecOutShape, t.device());

    // 20260328 ZJH GPU 路径：CUDABackend 直接在 GPU 上切片，零 D2H
    if (isCudaTensor(t)) {
#ifdef OM_HAS_CUDA
        CUDABackend::sliceLastDim(ct.floatDataPtr(), result.mutableFloatDataPtr(),
                                   nOuter, nLastDim, nStart, nSliceLen);
#endif
    } else {
        // 20260321 ZJH CPU 路径：逐行切片拷贝
        const float* pSrc = ct.floatDataPtr();        // 20260321 ZJH 源数据指针
        float* pDst = result.mutableFloatDataPtr();    // 20260321 ZJH 目标数据指针
        for (int i = 0; i < nOuter; ++i) {
            for (int j = 0; j < nSliceLen; ++j) {
                pDst[i * nSliceLen + j] = pSrc[i * nLastDim + nStart + j];
            }
        }
    }

    // 20260328 ZJH AutoGrad：注册 SliceLastDimBackwardFn
    if (t.requiresGrad()) {
        auto pBackward = std::make_shared<SliceLastDimBackwardFn>();  // 20260328 ZJH 创建反向函数
        pBackward->m_nOuter = nOuter;        // 20260328 ZJH 保存外层元素数
        pBackward->m_nFullDim = nLastDim;    // 20260328 ZJH 保存原始最后一维大小
        pBackward->m_nStart = nStart;        // 20260328 ZJH 保存切片起始位置
        pBackward->m_nLen = nSliceLen;       // 20260328 ZJH 保存切片长度
        pBackward->m_vecFullShape = vecShape;  // 20260328 ZJH 保存原始完整形状（用于反向分配 gradInput）
        pBackward->m_vecInputEdges.push_back(makeEdge(t, 0));  // 20260328 ZJH 连接输入的梯度边
        result.setGradFnRaw(pBackward);    // 20260328 ZJH 设置反向函数
        result.setRequiresGrad(true);      // 20260328 ZJH 标记需要梯度
    }

    return result;
}

// 20260321 ZJH tensorConcatLastDim — 沿最后一维拼接两个张量
// 用于双向 LSTM 将正向和反向隐藏状态拼接
// 20260325 ZJH Phase 3: 添加设备调度
// 20260328 ZJH GPU 直接拼接（零 D2H）+ autograd 反向传播
Tensor tensorConcatLastDim(Tensor a, Tensor b) {
    checkSameDevice(a, b);  // 20260325 ZJH 确保两个张量在同一设备上
    auto ca = a.contiguous();  // 20260328 ZJH 确保 a 连续
    auto cb = b.contiguous();  // 20260328 ZJH 确保 b 连续
    auto vecShapeA = ca.shapeVec();  // 20260321 ZJH A 的形状
    auto vecShapeB = cb.shapeVec();  // 20260321 ZJH B 的形状
    int nNDim = static_cast<int>(vecShapeA.size());  // 20260321 ZJH 维度数
    int nDimA = vecShapeA[nNDim - 1];  // 20260321 ZJH A 最后一维大小
    int nDimB = vecShapeB[nNDim - 1];  // 20260321 ZJH B 最后一维大小
    int nOuter = ca.numel() / nDimA;   // 20260321 ZJH 外层元素数

    // 20260321 ZJH 构建输出形状
    auto vecOutShape = vecShapeA;
    vecOutShape[nNDim - 1] = nDimA + nDimB;

    // 20260328 ZJH 在输入设备上分配输出张量
    auto result = Tensor::zeros(vecOutShape, a.device());

    // 20260328 ZJH GPU 路径：CUDABackend 直接在 GPU 上拼接，零 D2H
    if (isCudaTensor(a)) {
#ifdef OM_HAS_CUDA
        CUDABackend::concatLastDim(ca.floatDataPtr(), cb.floatDataPtr(),
                                    result.mutableFloatDataPtr(),
                                    nOuter, nDimA, nDimB);
#endif
    } else {
        // 20260321 ZJH CPU 路径：逐行拼接
        const float* pA = ca.floatDataPtr();   // 20260321 ZJH A 数据指针
        const float* pB = cb.floatDataPtr();   // 20260321 ZJH B 数据指针
        float* pOut = result.mutableFloatDataPtr();  // 20260321 ZJH 输出数据指针
        int nOutDim = nDimA + nDimB;  // 20260321 ZJH 输出最后一维大小
        for (int i = 0; i < nOuter; ++i) {
            // 20260321 ZJH 拷贝 A 的最后一维数据
            for (int j = 0; j < nDimA; ++j) {
                pOut[i * nOutDim + j] = pA[i * nDimA + j];
            }
            // 20260321 ZJH 拷贝 B 的最后一维数据
            for (int j = 0; j < nDimB; ++j) {
                pOut[i * nOutDim + nDimA + j] = pB[i * nDimB + j];
            }
        }
    }

    // 20260328 ZJH AutoGrad：注册 ConcatLastDimBackwardFn
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<ConcatLastDimBackwardFn>();  // 20260328 ZJH 创建反向函数
        pBackward->m_nOuter = nOuter;   // 20260328 ZJH 保存外层元素数
        pBackward->m_nDimA = nDimA;     // 20260328 ZJH 保存 A 最后一维大小
        pBackward->m_nDimB = nDimB;     // 20260328 ZJH 保存 B 最后一维大小
        pBackward->m_vecShapeA = vecShapeA;  // 20260328 ZJH 保存 A 形状（用于反向分配 gradA）
        pBackward->m_vecShapeB = vecShapeB;  // 20260328 ZJH 保存 B 形状（用于反向分配 gradB）
        pBackward->m_vecInputEdges.push_back(makeEdge(a, 0));  // 20260328 ZJH 连接 a 的梯度边
        pBackward->m_vecInputEdges.push_back(makeEdge(b, 0));  // 20260328 ZJH 连接 b 的梯度边
        result.setGradFnRaw(pBackward);    // 20260328 ZJH 设置反向函数
        result.setRequiresGrad(true);      // 20260328 ZJH 标记需要梯度
    }

    return result;
}

// 20260328 ZJH tensorDiceLoss — Fused Dice Loss：sigmoid + 归约 + 标量计算，单 kernel 替代 10 步 ops
// 输入: logits [B,C,H,W] 模型原始输出, target [B,C,H,W] one-hot 掩码
// 输出: 标量 loss {1} = 1 - 2*sum(sig*target) / (sum(sig) + sum(target) + eps)
// GPU 路径: CUDABackend::diceLossForward (1 fused kernel)
// CPU 路径: 内联计算 sigmoid + 3 路累加 + 标量公式
// AutoGrad: 注册 DiceLossBackwardFn，反向时调用 fused backward kernel
Tensor tensorDiceLoss(Tensor logits, const Tensor& target) {
    auto cLogits = logits.contiguous();  // 20260328 ZJH 确保 logits 连续
    auto cTarget = target.contiguous();  // 20260328 ZJH 确保 target 连续
    int nCount = static_cast<int>(cLogits.numel());  // 20260328 ZJH 元素总数 N = B*C*H*W

    // 20260328 ZJH 输出: 标量损失 {1}, sigmoid 中间结果 [B,C,H,W], 统计量 {3}
    Tensor tLoss;           // 20260328 ZJH 标量损失张量
    Tensor tSigmoidOut;     // 20260328 ZJH sigmoid 中间结果（反向需要）
    Tensor tStats;          // 20260328 ZJH {intersection, predSum, targetSum}

    if (isCudaTensor(logits)) {
#ifdef OM_HAS_CUDA
        // 20260328 ZJH GPU 路径：所有 tensor 在 GPU 分配，调用 fused kernel
        tLoss = Tensor::zeros({1}, DeviceType::CUDA);          // 20260328 ZJH GPU 标量损失
        tSigmoidOut = Tensor::zeros(cLogits.shapeVec(), DeviceType::CUDA);  // 20260328 ZJH GPU sigmoid 缓存
        tStats = Tensor::zeros({3}, DeviceType::CUDA);         // 20260328 ZJH GPU 统计量
        CUDABackend::diceLossForward(
            cLogits.floatDataPtr(), cTarget.floatDataPtr(),
            tLoss.mutableFloatDataPtr(), tSigmoidOut.mutableFloatDataPtr(),
            tStats.mutableFloatDataPtr(), nCount);  // 20260328 ZJH 1 fused kernel 完成前向
#endif
    } else {
        // 20260328 ZJH CPU 路径：内联计算（sigmoid + 3 路累加 + 标量公式）
        tSigmoidOut = Tensor::zeros(cLogits.shapeVec());  // 20260328 ZJH CPU sigmoid 缓存
        const float* pLogits = cLogits.floatDataPtr();     // 20260328 ZJH logits 只读指针
        const float* pTarget = cTarget.floatDataPtr();     // 20260328 ZJH target 只读指针
        float* pSigOut = tSigmoidOut.mutableFloatDataPtr();  // 20260328 ZJH sigmoid 可写指针

        float fIntersection = 0.0f;  // 20260328 ZJH 交集 I = sum(sig * target)
        float fPredSum = 0.0f;       // 20260328 ZJH 预测和 P = sum(sig)
        float fTargetSum = 0.0f;     // 20260328 ZJH 目标和 T = sum(target)

        // 20260328 ZJH 逐元素: sigmoid + 累加 3 个统计量
        for (int i = 0; i < nCount; ++i) {
            float fSig = 1.0f / (1.0f + std::exp(-pLogits[i]));  // 20260328 ZJH sigmoid
            pSigOut[i] = fSig;               // 20260328 ZJH 保存 sigmoid
            fIntersection += fSig * pTarget[i];  // 20260328 ZJH 累加交集
            fPredSum += fSig;                // 20260328 ZJH 累加预测
            fTargetSum += pTarget[i];        // 20260328 ZJH 累加目标
        }

        // 20260328 ZJH 计算标量 Dice Loss
        float fEps = 1e-6f;  // 20260328 ZJH 数值稳定 epsilon
        float fDice = 2.0f * fIntersection / (fPredSum + fTargetSum + fEps);  // 20260328 ZJH Dice 系数
        float fLossVal = 1.0f - fDice;  // 20260328 ZJH Dice Loss

        tLoss = Tensor::full({1}, fLossVal);  // 20260328 ZJH 创建标量损失张量

        // 20260328 ZJH 保存统计量（CPU 反向需要）
        tStats = Tensor::zeros({3});  // 20260328 ZJH CPU 统计量
        float* pStats = tStats.mutableFloatDataPtr();  // 20260328 ZJH 统计量可写指针
        pStats[0] = fIntersection;  // 20260328 ZJH 交集
        pStats[1] = fPredSum;       // 20260328 ZJH 预测和
        pStats[2] = fTargetSum;     // 20260328 ZJH 目标和
    }

    // 20260328 ZJH AutoGrad: 注册 DiceLossBackwardFn
    if (logits.requiresGrad()) {
        auto pBackward = std::make_shared<DiceLossBackwardFn>();  // 20260328 ZJH 创建反向函数
        pBackward->m_savedSigmoid = tSigmoidOut;  // 20260328 ZJH 保存 sigmoid 输出
        pBackward->m_savedTarget = cTarget;        // 20260328 ZJH 保存 one-hot 目标
        pBackward->m_savedStats = tStats;          // 20260328 ZJH 保存统计量
        pBackward->m_vecInputEdges.push_back(makeEdge(logits, 0));  // 20260328 ZJH 连接 logits 的梯度边
        tLoss.setGradFnRaw(pBackward);   // 20260328 ZJH 设置反向函数
        tLoss.setRequiresGrad(true);     // 20260328 ZJH 标记需要梯度
    }

    return tLoss;  // 20260328 ZJH 返回标量 Dice Loss
}

}  // namespace om
