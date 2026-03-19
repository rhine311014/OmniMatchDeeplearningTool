// 20260319 ZJH Tensor 类实现 — Phase 1B-T4
// 采用 Storage+View 分离设计：shared_ptr<TensorStorage> 持有原始内存，
// Tensor 自身持有 shape / strides / offset，支持零拷贝视图操作
module;

#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <cstddef>
#include <cassert>
#include <cstring>

#include "df_types.h"

export module df.engine.tensor;
import df.engine.tensor_storage;
import df.hal.cpu_backend;

export namespace df {

// 20260319 ZJH Tensor — 多维浮点张量，当前仅支持 Float32 + CPU
class Tensor {
public:
    // =========================================================
    // 构造 / 析构
    // =========================================================

    // 20260319 ZJH 默认构造 — 生成空张量（无形状、无存储）
    Tensor() = default;

    // 20260319 ZJH 拷贝构造 — 共享同一 Storage（视图语义）
    Tensor(const Tensor&) = default;

    // 20260319 ZJH 移动构造
    Tensor(Tensor&&) noexcept = default;

    // 20260319 ZJH 拷贝赋值 — 共享 Storage
    Tensor& operator=(const Tensor&) = default;

    // 20260319 ZJH 移动赋值
    Tensor& operator=(Tensor&&) noexcept = default;

    ~Tensor() = default;

    // =========================================================
    // 工厂方法（静态）
    // =========================================================

    // 20260319 ZJH zeros — 创建指定形状的全零张量，Float32 + CPU
    static Tensor zeros(const std::vector<int>& vecShape) {
        Tensor t;
        // 20260319 ZJH 分配连续存储并计算 strides/offset
        t.initContiguous(vecShape);
        // 20260319 ZJH 调用 CPUBackend 将所有元素置零
        CPUBackend::fillZeros(t.mutableFloatDataPtr(), t.numel());
        return t;
    }

    // 20260319 ZJH ones — 创建指定形状的全 1 张量
    static Tensor ones(const std::vector<int>& vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        // 20260319 ZJH 调用 CPUBackend 将所有元素置 1.0f
        CPUBackend::fillOnes(t.mutableFloatDataPtr(), t.numel());
        return t;
    }

    // 20260319 ZJH full — 创建指定形状并填充为标量值 fValue 的张量
    static Tensor full(const std::vector<int>& vecShape, float fValue) {
        Tensor t;
        t.initContiguous(vecShape);
        // 20260319 ZJH 调用 CPUBackend 将所有元素填充为指定标量
        CPUBackend::fillValue(t.mutableFloatDataPtr(), t.numel(), fValue);
        return t;
    }

    // 20260319 ZJH randn — 创建指定形状并用标准正态分布填充的张量
    static Tensor randn(const std::vector<int>& vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        // 20260319 ZJH 调用 CPUBackend 以 N(0,1) 随机填充
        CPUBackend::fillRandn(t.mutableFloatDataPtr(), t.numel());
        return t;
    }

    // 20260319 ZJH fromData — 从外部 float 数组构造张量（深拷贝）
    // pSrcData: 源数据指针；vecShape: 目标形状，元素总数必须匹配
    static Tensor fromData(const float* pSrcData, const std::vector<int>& vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        // 20260319 ZJH 计算字节数并调用 CPUBackend::copy 拷贝数据
        size_t nCount = static_cast<size_t>(t.numel());
        CPUBackend::copy(pSrcData, t.mutableFloatDataPtr(), nCount);
        return t;
    }

    // 20260319 ZJH makeView — 供 tensor_ops 使用的视图创建接口
    // pStorage: 共享存储；vecShape / vecStrides: 视图形状和步长；nOffset: 起始偏移（元素单位）
    static Tensor makeView(std::shared_ptr<TensorStorage> pStorage,
                           const std::vector<int>& vecShape,
                           const std::vector<int>& vecStrides,
                           int nOffset) {
        Tensor t;
        t.m_pStorage  = std::move(pStorage);  // 20260319 ZJH 共享原始存储，不拷贝数据
        t.m_vecShape   = vecShape;             // 20260319 ZJH 视图的形状
        t.m_vecStrides = vecStrides;           // 20260319 ZJH 视图的步长（元素单位）
        t.m_nOffset    = nOffset;              // 20260319 ZJH 视图在存储中的起始偏移
        return t;
    }

    // =========================================================
    // 属性访问
    // =========================================================

    // 20260319 ZJH 返回张量维度数
    int ndim() const {
        return static_cast<int>(m_vecShape.size());
    }

    // 20260319 ZJH 返回第 nDim 维的大小，越界时抛出异常
    int shape(int nDim) const {
        if (nDim < 0 || nDim >= ndim()) {
            throw std::out_of_range("Tensor::shape — dim out of range");
        }
        return m_vecShape[static_cast<size_t>(nDim)];
    }

    // 20260319 ZJH 返回完整形状向量（只读副本）
    const std::vector<int>& shapeVec() const {
        return m_vecShape;
    }

    // 20260319 ZJH 返回第 nDim 维的步长（元素单位），越界时抛出异常
    int stride(int nDim) const {
        if (nDim < 0 || nDim >= ndim()) {
            throw std::out_of_range("Tensor::stride — dim out of range");
        }
        return m_vecStrides[static_cast<size_t>(nDim)];
    }

    // 20260319 ZJH 返回完整步长向量（只读副本）
    const std::vector<int>& stridesVec() const {
        return m_vecStrides;
    }

    // 20260319 ZJH 返回张量元素总数（各维度大小之积）
    int numel() const {
        if (m_vecShape.empty()) return 0;  // 20260319 ZJH 空张量元素数为 0
        int nTotal = 1;
        for (int s : m_vecShape) nTotal *= s;  // 20260319 ZJH 逐维相乘求乘积
        return nTotal;
    }

    // 20260319 ZJH 返回数据类型（当前固定 Float32）
    DataType dtype() const {
        return DataType::Float32;
    }

    // 20260319 ZJH 返回设备类型（当前固定 CPU）
    DeviceType device() const {
        return DeviceType::CPU;
    }

    // 20260319 ZJH 判断张量是否连续（行主序 C 连续）
    // 最低维步长必须为 1，其余维步长必须满足 stride[d] = stride[d+1]*shape[d+1]
    bool isContiguous() const {
        int nDims = ndim();
        if (nDims == 0) return true;  // 20260319 ZJH 标量视为连续

        // 20260319 ZJH 检查最低维步长是否为 1
        if (m_vecStrides[static_cast<size_t>(nDims - 1)] != 1) return false;

        // 20260319 ZJH 从倒数第二维开始向高维逐一验证步长连续性
        for (int d = nDims - 2; d >= 0; --d) {
            int nExpected = m_vecStrides[static_cast<size_t>(d + 1)]
                          * m_vecShape[static_cast<size_t>(d + 1)];
            if (m_vecStrides[static_cast<size_t>(d)] != nExpected) return false;
        }
        return true;  // 20260319 ZJH 所有维度步长均满足连续条件
    }

    // =========================================================
    // 数据访问
    // =========================================================

    // 20260319 ZJH 返回底层 float 数组常量指针（含 offset），供只读访问
    const float* floatDataPtr() const {
        if (!m_pStorage) {
            throw std::runtime_error("Tensor::floatDataPtr — empty tensor");
        }
        // 20260319 ZJH 从存储起始地址加上元素偏移量得到实际指针
        const float* pBase = static_cast<const float*>(m_pStorage->data());
        return pBase + m_nOffset;  // 20260319 ZJH m_nOffset 单位为元素数
    }

    // 20260319 ZJH 返回底层 float 数组可写指针（含 offset），供写入访问
    float* mutableFloatDataPtr() {
        if (!m_pStorage) {
            throw std::runtime_error("Tensor::mutableFloatDataPtr — empty tensor");
        }
        float* pBase = static_cast<float*>(m_pStorage->mutableData());
        return pBase + m_nOffset;  // 20260319 ZJH m_nOffset 单位为元素数
    }

    // 20260319 ZJH 多维索引随机访问（只读）
    // vecIndices: 每个维度的索引，长度必须等于 ndim()
    float at(const std::vector<int>& vecIndices) const {
        // 20260319 ZJH 验证索引维度数与张量维度数一致
        if (static_cast<int>(vecIndices.size()) != ndim()) {
            throw std::invalid_argument("Tensor::at — indices size mismatch");
        }
        // 20260319 ZJH 计算线性偏移：offset + sum(indices[d] * strides[d])
        int nLinear = m_nOffset;
        for (int d = 0; d < ndim(); ++d) {
            int nIdx = vecIndices[static_cast<size_t>(d)];
            // 20260319 ZJH 边界检查
            if (nIdx < 0 || nIdx >= m_vecShape[static_cast<size_t>(d)]) {
                throw std::out_of_range("Tensor::at — index out of range");
            }
            nLinear += nIdx * m_vecStrides[static_cast<size_t>(d)];
        }
        // 20260319 ZJH 从存储中读取对应位置的 float 值
        const float* pBase = static_cast<const float*>(m_pStorage->data());
        return pBase[nLinear];
    }

    // 20260319 ZJH 多维索引随机写入
    // vecIndices: 每个维度的索引；fValue: 要写入的值
    void setAt(const std::vector<int>& vecIndices, float fValue) {
        // 20260319 ZJH 验证索引维度数与张量维度数一致
        if (static_cast<int>(vecIndices.size()) != ndim()) {
            throw std::invalid_argument("Tensor::setAt — indices size mismatch");
        }
        // 20260319 ZJH 计算线性偏移
        int nLinear = m_nOffset;
        for (int d = 0; d < ndim(); ++d) {
            int nIdx = vecIndices[static_cast<size_t>(d)];
            if (nIdx < 0 || nIdx >= m_vecShape[static_cast<size_t>(d)]) {
                throw std::out_of_range("Tensor::setAt — index out of range");
            }
            nLinear += nIdx * m_vecStrides[static_cast<size_t>(d)];
        }
        // 20260319 ZJH 写入目标位置
        float* pBase = static_cast<float*>(m_pStorage->mutableData());
        pBase[nLinear] = fValue;
    }

    // =========================================================
    // 连续化
    // =========================================================

    // 20260319 ZJH 若当前张量连续则返回自身，否则将数据拷贝到新的连续张量
    Tensor contiguous() const {
        if (isContiguous()) {
            return *this;  // 20260319 ZJH 已连续，直接返回当前视图
        }
        // 20260319 ZJH 创建新的连续存储
        Tensor result;
        result.initContiguous(m_vecShape);

        // 20260319 ZJH 将步长和形状转换为 stridedCopy 所需的 vector<int>
        // m_vecStrides 已是 vector<int>，可直接传入
        const float* pSrcBase = static_cast<const float*>(m_pStorage->data());
        float* pDstBase       = result.mutableFloatDataPtr();

        // 20260319 ZJH 调用 CPUBackend::stridedCopy 按非连续步长提取数据
        CPUBackend::stridedCopy(pSrcBase, pDstBase,
                                m_vecShape,    // 20260319 ZJH 各维度大小
                                m_vecStrides,  // 20260319 ZJH 源张量步长（元素单位）
                                m_nOffset);    // 20260319 ZJH 源张量起始偏移
        return result;
    }

    // =========================================================
    // 内部访问器（供 tensor_ops 使用）
    // =========================================================

    // 20260319 ZJH 返回共享存储指针（供 tensor_ops 构造视图）
    std::shared_ptr<TensorStorage> storage() const {
        return m_pStorage;
    }

    // 20260319 ZJH 返回元素单位偏移（供 tensor_ops 读取视图起始位置）
    int offset() const {
        return m_nOffset;
    }

    // =========================================================
    // AutoGrad 支持（类型擦除，避免循环依赖）
    // =========================================================

    // 20260319 ZJH 设置是否需要梯度，若为 true 则懒创建 GradAccumulator
    // GradAccumulator 以 shared_ptr<void> 类型擦除存储，避免 tensor.ixx 导入 autograd.ixx
    void setRequiresGrad(bool b) {
        m_bRequiresGrad = b;  // 20260319 ZJH 记录是否需要梯度
        // 20260319 ZJH 注意: GradAccumulator 的实际创建由 tensor_ops 中的 ensureLeafAccumulator 完成
    }

    // 20260319 ZJH 返回当前张量是否需要梯度
    bool requiresGrad() const {
        return m_bRequiresGrad;
    }

    // 20260319 ZJH 设置类型擦除的 GradFunction（由 tensor_ops 在前向运算后调用）
    void setGradFnRaw(std::shared_ptr<void> p) {
        m_pGradFn = std::move(p);  // 20260319 ZJH 存储 GradFunction 的 shared_ptr<void>
    }

    // 20260319 ZJH 获取类型擦除的 GradFunction
    std::shared_ptr<void> gradFnRaw() const {
        return m_pGradFn;  // 20260319 ZJH 返回 shared_ptr<void>，由调用方 static_pointer_cast
    }

    // 20260319 ZJH 设置类型擦除的 GradAccumulator（叶节点的梯度累加器）
    void setGradAccumRaw(std::shared_ptr<void> p) {
        m_pGradAccumulator = std::move(p);  // 20260319 ZJH 存储 GradAccumulator 的 shared_ptr<void>
    }

    // 20260319 ZJH 获取类型擦除的 GradAccumulator
    std::shared_ptr<void> gradAccumRaw() const {
        return m_pGradAccumulator;  // 20260319 ZJH 返回 shared_ptr<void>
    }

    // 20260319 ZJH item — 从标量张量（numel()==1）中提取单个 float 值
    // 用于 SumBackward 等需要读取标量梯度值的场景
    float item() const {
        assert(numel() == 1 && "Tensor::item — only scalar tensors (numel==1) supported");
        return floatDataPtr()[0];  // 20260319 ZJH 返回第一个（也是唯一一个）元素
    }

private:
    // =========================================================
    // 私有成员
    // =========================================================

    // 20260319 ZJH 共享存储：多个视图 Tensor 可指向同一 TensorStorage
    std::shared_ptr<TensorStorage> m_pStorage;

    // 20260319 ZJH 张量形状：每个维度的元素个数，如 {2, 3, 4}
    std::vector<int> m_vecShape;

    // 20260319 ZJH 步长向量（元素单位）：stride[d] 表示第 d 维相邻元素在内存中的间距
    std::vector<int> m_vecStrides;

    // 20260319 ZJH 存储起始偏移（元素单位）：视图操作后可能不从 0 开始
    int m_nOffset = 0;

    // 20260319 ZJH 是否需要梯度（用户设置的叶节点标记）
    bool m_bRequiresGrad = false;

    // 20260319 ZJH 类型擦除的 GradFunction（实际类型为 shared_ptr<GradFunction>）
    // 非叶节点由前向运算自动设置；叶节点由 ensureLeafAccumulator 设置为 LeafAccumulator
    std::shared_ptr<void> m_pGradFn;

    // 20260319 ZJH 类型擦除的 GradAccumulator（实际类型为 shared_ptr<GradAccumulator>）
    // 仅叶节点持有，用于累积反向传播的梯度
    std::shared_ptr<void> m_pGradAccumulator;

    // =========================================================
    // 私有辅助
    // =========================================================

    // 20260319 ZJH 初始化连续存储：计算行主序步长并分配内存
    // vecShape: 目标形状；为空时创建零元素张量
    void initContiguous(const std::vector<int>& vecShape) {
        m_vecShape = vecShape;   // 20260319 ZJH 记录形状
        m_nOffset  = 0;          // 20260319 ZJH 连续张量从 offset=0 开始

        int nDims = static_cast<int>(vecShape.size());
        m_vecStrides.resize(static_cast<size_t>(nDims));  // 20260319 ZJH 分配步长数组

        if (nDims == 0) {
            // 20260319 ZJH 标量张量：无形状，分配 1 个元素的存储
            m_pStorage = std::make_shared<TensorStorage>(sizeof(float));
            return;
        }

        // 20260319 ZJH 从最低维开始倒序计算行主序步长
        // 最低维步长 = 1，stride[d] = stride[d+1] * shape[d+1]
        m_vecStrides[static_cast<size_t>(nDims - 1)] = 1;
        for (int d = nDims - 2; d >= 0; --d) {
            m_vecStrides[static_cast<size_t>(d)] =
                m_vecStrides[static_cast<size_t>(d + 1)]
                * vecShape[static_cast<size_t>(d + 1)];
        }

        // 20260319 ZJH 计算总元素数并分配对应字节的对齐存储
        size_t nTotal = static_cast<size_t>(numel());
        m_pStorage = std::make_shared<TensorStorage>(nTotal * sizeof(float));
    }
};

}  // namespace df
