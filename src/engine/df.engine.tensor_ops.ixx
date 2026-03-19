// 20260319 ZJH Tensor 运算模块 — 元素运算、matmul、reshape、transpose、slice、归约
module;

#include <vector>
#include <stdexcept>
#include <memory>
#include <cassert>
#include "df_types.h"

export module df.engine.tensor_ops;

// 20260319 ZJH 导入依赖模块：存储层、张量类、CPU 计算内核
import df.engine.tensor_storage;
import df.engine.tensor;
import df.hal.cpu_backend;

export namespace df {

// ===== 元素运算 =====

// 20260319 ZJH 逐元素加法：result = a + b（形状必须相同）
// 先将两个张量转为连续内存，再调用 CPUBackend::add 逐元素相加
Tensor tensorAdd(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保 a 为连续内存（非连续则拷贝）
    auto cb = b.contiguous();  // 20260319 ZJH 确保 b 为连续内存
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配与 a 相同形状的输出张量
    // 20260319 ZJH 调用 CPU 逐元素加法内核，numel() 个 float
    CPUBackend::add(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    return result;  // 20260319 ZJH 返回计算结果张量
}

// 20260319 ZJH 逐元素减法：result = a - b
Tensor tensorSub(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 逐元素减法内核
    CPUBackend::sub(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    return result;
}

// 20260319 ZJH 逐元素乘法：result = a * b（Hadamard 积）
Tensor tensorMul(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto cb = b.contiguous();  // 20260319 ZJH 连续化 b
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 逐元素乘法内核
    CPUBackend::mul(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    return result;
}

// 20260319 ZJH 逐元素除法：result = a / b（调用方保证 b 中无零）
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
Tensor tensorAddScalar(const Tensor& a, float fScalar) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 加标量内核
    CPUBackend::addScalar(ca.floatDataPtr(), fScalar,
                          result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    return result;
}

// 20260319 ZJH 乘标量：result = a * fScalar（广播到所有元素）
Tensor tensorMulScalar(const Tensor& a, float fScalar) {
    auto ca = a.contiguous();  // 20260319 ZJH 连续化 a
    auto result = Tensor::zeros(ca.shapeVec());  // 20260319 ZJH 分配输出张量
    // 20260319 ZJH 调用 CPU 乘标量内核
    CPUBackend::mulScalar(ca.floatDataPtr(), fScalar,
                          result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));
    return result;
}

// ===== 矩阵乘法 =====

// 20260319 ZJH 二维矩阵乘法：result = a @ b
// a 形状 [M, K]，b 形状 [K, N]，result 形状 [M, N]
// 调用方须保证 a.ndim()==2, b.ndim()==2, a.shape(1)==b.shape(0)
Tensor tensorMatmul(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();  // 20260319 ZJH 确保 a 连续，matmul 需要行主序连续内存
    auto cb = b.contiguous();  // 20260319 ZJH 确保 b 连续
    int nM = ca.shape(0);  // 20260319 ZJH 行数（矩阵 A 的行）
    int nK = ca.shape(1);  // 20260319 ZJH 内维度（A 的列 = B 的行）
    int nN = cb.shape(1);  // 20260319 ZJH 列数（矩阵 B 的列）
    auto result = Tensor::zeros({nM, nN});  // 20260319 ZJH 分配 [M, N] 输出张量
    // 20260319 ZJH 调用 CPU matmul 内核：A[M,K]*B[K,N]->C[M,N]
    CPUBackend::matmul(ca.floatDataPtr(), cb.floatDataPtr(),
                       result.mutableFloatDataPtr(), nM, nK, nN);
    return result;  // 20260319 ZJH 返回矩阵乘法结果
}

// ===== 形状变换 =====

// 20260319 ZJH reshape — 改变张量形状，不拷贝数据（若原张量已连续）
// vecNewShape: 新形状，元素总数必须与原张量一致
// 若原张量已连续则共享 Storage 创建视图；否则先连续化再递归
Tensor tensorReshape(const Tensor& t, std::vector<int> vecNewShape) {
    // 20260319 ZJH 验证新旧形状的元素总数相等
    int nNewNumel = 1;
    for (int s : vecNewShape) nNewNumel *= s;  // 20260319 ZJH 计算新形状的元素总数
    if (nNewNumel != t.numel()) {
        // 20260319 ZJH 元素数不匹配时抛出异常，防止数据越界或丢失
        throw std::invalid_argument("tensorReshape: total element count mismatch");
    }

    if (t.isContiguous()) {
        // 20260319 ZJH 原张量连续：直接共享 Storage，仅修改 shape/strides，零拷贝
        std::vector<int> vecNewStrides(vecNewShape.size());  // 20260319 ZJH 新步长向量
        if (!vecNewShape.empty()) {
            vecNewStrides.back() = 1;  // 20260319 ZJH 最低维步长固定为 1（行主序）
            // 20260319 ZJH 从倒数第二维向高维逐一计算步长
            for (int d = static_cast<int>(vecNewShape.size()) - 2; d >= 0; --d) {
                vecNewStrides[static_cast<size_t>(d)] =
                    vecNewStrides[static_cast<size_t>(d + 1)]
                    * vecNewShape[static_cast<size_t>(d + 1)];
            }
        }
        // 20260319 ZJH 调用 makeView 创建共享存储的视图张量，offset 不变
        return Tensor::makeView(t.storage(), vecNewShape, vecNewStrides, t.offset());
    } else {
        // 20260319 ZJH 非连续张量：先连续化（数据拷贝），再对连续副本执行 reshape
        auto ct = t.contiguous();
        return tensorReshape(ct, vecNewShape);  // 20260319 ZJH 递归调用，连续化后必然走零拷贝路径
    }
}

// 20260319 ZJH transpose — 交换两个维度，生成视图（不拷贝数据）
// nDim0, nDim1: 要交换的两个维度索引
// 通过交换 shape 和 strides 中对应维度的值实现视图转置
Tensor tensorTranspose(const Tensor& t, int nDim0, int nDim1) {
    auto vecShape   = t.shapeVec();    // 20260319 ZJH 拷贝原形状向量（将被修改）
    auto vecStrides = t.stridesVec();  // 20260319 ZJH 拷贝原步长向量（将被修改）
    // 20260319 ZJH 交换 shape 中两个维度的大小
    std::swap(vecShape[static_cast<size_t>(nDim0)], vecShape[static_cast<size_t>(nDim1)]);
    // 20260319 ZJH 交换 strides 中对应维度的步长，实现逻辑转置
    std::swap(vecStrides[static_cast<size_t>(nDim0)], vecStrides[static_cast<size_t>(nDim1)]);
    // 20260319 ZJH 创建共享 Storage 的视图，offset 不变，形状和步长已转置
    return Tensor::makeView(t.storage(), vecShape, vecStrides, t.offset());
}

// 20260319 ZJH slice — 在指定维度上截取子区间 [nStart, nEnd)，生成视图（不拷贝数据）
// nDim: 要切片的维度；nStart: 起始索引（含）；nEnd: 结束索引（不含）
// 通过调整 offset 和对应维度的 shape 实现零拷贝切片
Tensor tensorSlice(const Tensor& t, int nDim, int nStart, int nEnd) {
    auto vecShape   = t.shapeVec();    // 20260319 ZJH 拷贝原形状向量
    auto vecStrides = t.stridesVec();  // 20260319 ZJH 拷贝原步长向量（切片后步长不变）
    // 20260319 ZJH 计算新 offset：原 offset 加上 nStart 乘以该维度步长
    int nNewOffset = t.offset() + nStart * vecStrides[static_cast<size_t>(nDim)];
    // 20260319 ZJH 修改切片维度的大小为区间长度 (nEnd - nStart)
    vecShape[static_cast<size_t>(nDim)] = nEnd - nStart;
    // 20260319 ZJH 创建共享 Storage 的视图，新 offset 指向切片起始元素
    return Tensor::makeView(t.storage(), vecShape, vecStrides, nNewOffset);
}

// ===== 归约运算 =====

// 20260319 ZJH 全局求和：返回张量所有元素之和（单个 float）
float tensorSum(const Tensor& t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存，CPUBackend::sum 需要线性数组
    return CPUBackend::sum(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
}

// 20260319 ZJH 全局最大值：返回张量所有元素中的最大值
float tensorMax(const Tensor& t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存
    return CPUBackend::max(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
}

// 20260319 ZJH 全局最小值：返回张量所有元素中的最小值
float tensorMin(const Tensor& t) {
    auto ct = t.contiguous();  // 20260319 ZJH 确保连续内存
    return CPUBackend::min(ct.floatDataPtr(), static_cast<size_t>(ct.numel()));
}

}  // namespace df
