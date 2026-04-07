// 20260330 ZJH 分布式训练模块 — 数据并行 + 梯度累积
// DataParallel: 将 mini-batch 按 batch 维度分片到多 GPU，各 GPU 独立前向/反向，梯度全规约后更新
//   单 GPU 模式 (nWorldSize=1): 直接在模型上 forward/backward，无通信开销
//   多 GPU 模式: API 完整但实际多卡通信需 NCCL 支持（此实现为概念框架）
// GradientAccumulator: 当 GPU 显存不足以容纳大 batch 时，分 N 步累积梯度后统一更新
//   完全功能实现，不依赖多 GPU
module;

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

export module om.engine.distributed;

// 20260330 ZJH 导入依赖模块：张量类、张量运算、模块基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {

// 20260330 ZJH 分布式训练配置
// nWorldSize: 参与训练的 GPU 数量（1=单卡模式，>1=多卡数据并行）
// nLocalRank: 当前进程对应的 GPU 编号（0-indexed）
// bSyncBatchNorm: 是否跨 GPU 同步 BatchNorm 的均值/方差统计量
//   开启后精度更高（全局 batch 统计），但通信开销增大
struct DistributedConfig {
    int nWorldSize = 1;           // 20260330 ZJH GPU 数量
    int nLocalRank = 0;           // 20260330 ZJH 当前 GPU 编号（0-indexed）
    bool bSyncBatchNorm = true;   // 20260330 ZJH 同步 BN（跨 GPU 聚合均值/方差）
};

// 20260330 ZJH DataParallel — 数据并行包装器
// 核心思想：
//   1. 将输入 mini-batch 沿 batch 维度均匀分片到各 GPU
//   2. 每个 GPU 用相同的模型副本独立执行 forward
//   3. 各 GPU 独立 backward 计算本地梯度
//   4. AllReduce 聚合所有 GPU 的梯度（求和后除以 worldSize）
//   5. 用同步后的梯度更新参数（所有 GPU 保持同步）
//
// 当前实现：
//   nWorldSize=1 → 单卡模式，直接调用模型的 forward/backward
//   nWorldSize>1 → 打印警告，回退到单卡（实际多卡需 NCCL）
class DataParallel {
public:
    // 20260330 ZJH 构造函数
    // pModel: 待并行化的模型（shared_ptr，所有 GPU 共享同一份参数）
    // config: 分布式配置（GPU 数量、当前 rank 等）
    DataParallel(std::shared_ptr<Module> pModel, const DistributedConfig& config)
        : m_pModel(pModel), m_config(config) {
        // 20260330 ZJH 参数校验
        if (!m_pModel) {
            throw std::runtime_error("DataParallel — model pointer is null");
        }
        if (m_config.nWorldSize < 1) {
            throw std::runtime_error("DataParallel — nWorldSize must be >= 1");
        }
        if (m_config.nLocalRank < 0 || m_config.nLocalRank >= m_config.nWorldSize) {
            throw std::runtime_error("DataParallel — nLocalRank out of range [0, nWorldSize)");
        }
        // 20260330 ZJH 多卡模式下输出警告（当前实现不含 NCCL 通信）
        if (m_config.nWorldSize > 1) {
            std::cerr << "[DataParallel] WARNING: nWorldSize=" << m_config.nWorldSize
                      << " but multi-GPU (NCCL) not available in pure C++ build. "
                      << "Falling back to single-GPU mode on rank " << m_config.nLocalRank
                      << "." << std::endl;
        }
    }

    // 20260330 ZJH forward — 前向传播
    // 单卡模式: 直接调用模型的 forward
    // 多卡模式 (概念): 按 batch 维度切分 input，各 GPU forward，拼接输出
    // input: 输入张量 [N, C, H, W] 或 [N, D]，第 0 维为 batch
    // 返回: 前向输出张量
    Tensor forward(const Tensor& input) {
        // 20260330 ZJH 单卡模式：直接前向
        if (m_config.nWorldSize <= 1) {
            return m_pModel->forward(input);  // 20260330 ZJH 无分片，直接计算
        }

        // 20260330 ZJH 多卡模式（概念实现）：按 batch 维度切分
        // 计算当前 rank 负责的 batch 范围
        int nBatchSize = input.shape(0);  // 20260330 ZJH 总 batch 大小
        int nChunkSize = nBatchSize / m_config.nWorldSize;  // 20260330 ZJH 每卡分得的 batch 大小
        int nRemainder = nBatchSize % m_config.nWorldSize;  // 20260330 ZJH 余数分配给前几个 rank

        // 20260330 ZJH 计算当前 rank 的起止索引
        // 前 nRemainder 个 rank 各多分 1 个样本
        int nStart = 0;  // 20260330 ZJH 当前 rank 的起始 batch 索引
        for (int r = 0; r < m_config.nLocalRank; ++r) {
            nStart += nChunkSize + (r < nRemainder ? 1 : 0);  // 20260330 ZJH 累加前面各 rank 的 chunk 大小
        }
        int nLocalChunk = nChunkSize + (m_config.nLocalRank < nRemainder ? 1 : 0);  // 20260330 ZJH 当前 rank 的 chunk 大小
        int nEnd = nStart + nLocalChunk;  // 20260330 ZJH 当前 rank 的结束索引（不含）

        // 20260330 ZJH 沿 batch 维度切片
        Tensor localInput = tensorSlice(input, 0, nStart, nEnd);  // 20260330 ZJH 取当前 rank 的子 batch

        // 20260330 ZJH 本地前向传播
        Tensor localOutput = m_pModel->forward(localInput);  // 20260330 ZJH 单 GPU forward

        // 20260330 ZJH 注意：实际多卡需要 AllGather 收集各 rank 的 localOutput 后拼接
        // 当前实现仅返回本 rank 的输出（单卡回退行为）
        return localOutput;  // 20260330 ZJH 返回本 rank 输出
    }

    // 20260330 ZJH backward — 反向传播
    // 对给定的标量损失执行反向传播
    // 单卡模式: 直接调用 autograd backward
    // 多卡模式 (概念): 各 GPU 独立 backward 后调用 synchronizeGradients
    // loss: 标量损失张量（shape={1}）
    void backward(const Tensor& loss) {
        // 20260330 ZJH 直接执行反向传播（autograd 自动追踪计算图）
        Tensor mutableLoss = loss;  // 20260330 ZJH tensorBackward 需要非 const 引用
        tensorBackward(mutableLoss);  // 20260330 ZJH 调用 autograd 全局反向传播

        // 20260330 ZJH 多卡模式下需要同步梯度
        if (m_config.nWorldSize > 1) {
            synchronizeGradients();  // 20260330 ZJH AllReduce 梯度同步
        }
    }

    // 20260330 ZJH synchronizeGradients — 跨 GPU 梯度同步 (AllReduce)
    // 算法: 将各 GPU 上相同参数的梯度求和，再除以 nWorldSize
    // 效果: 等价于在完整 mini-batch 上计算梯度
    // 当前实现: 单卡模式下为空操作；多卡需 NCCL 的 AllReduce
    void synchronizeGradients() {
        // 20260330 ZJH 单卡模式：无需同步
        if (m_config.nWorldSize <= 1) {
            return;
        }

        // 20260330 ZJH 多卡模式（概念框架）
        // 实际实现需要:
        //   1. 遍历 m_pModel->parameters()，获取每个参数的 .grad()
        //   2. 调用 NCCL ncclAllReduce(grad_data, grad_data, count, ncclFloat, ncclSum, comm, stream)
        //   3. 对每个梯度除以 nWorldSize（或在 AllReduce 时使用 ncclAvg）
        //
        // 伪代码:
        //   for (auto& [name, paramPtr] : m_pModel->namedParameters()) {
        //       Tensor grad = paramPtr->grad();
        //       ncclAllReduce(grad.data(), grad.data(), grad.numel(), ncclFloat, ncclSum, m_comm, m_stream);
        //       grad = tensorDiv(grad, Tensor::full({1}, static_cast<float>(m_config.nWorldSize)));
        //   }

        // 20260330 ZJH 当前实现：获取参数并模拟单卡规约（梯度除以 1，无实际效果）
        auto vecParams = m_pModel->parameters();  // 20260330 ZJH 获取所有可训练参数
        // 20260330 ZJH 遍历参数进行梯度归一化（多卡时需要除以 nWorldSize）
        for (auto* pParam : vecParams) {
            if (pParam->requiresGrad() && pParam->gradAccumRaw()) {
                // 20260330 ZJH 参数有梯度，在真正的多卡实现中此处执行 AllReduce
                // 当前回退模式下不做额外操作（单卡梯度已是正确的）
                (void)pParam;  // 20260330 ZJH 抑制未使用变量警告
            }
        }
    }

    // 20260330 ZJH module — 获取底层模型的引用
    // 用于访问模型参数、保存/加载等操作
    Module& module() {
        return *m_pModel;  // 20260330 ZJH 解引用返回模型引用
    }

    // 20260330 ZJH module — const 版本
    const Module& module() const {
        return *m_pModel;  // 20260330 ZJH 解引用返回模型常引用
    }

    // 20260330 ZJH getConfig — 获取分布式配置
    const DistributedConfig& getConfig() const {
        return m_config;  // 20260330 ZJH 返回配置引用
    }

    // 20260330 ZJH isMasterRank — 判断当前是否为主 rank (rank 0)
    // 主 rank 通常负责日志打印、模型保存等
    bool isMasterRank() const {
        return m_config.nLocalRank == 0;  // 20260330 ZJH rank 0 为主
    }

private:
    std::shared_ptr<Module> m_pModel;  // 20260330 ZJH 模型指针（所有 GPU 共享）
    DistributedConfig m_config;        // 20260330 ZJH 分布式训练配置
};

// 20260330 ZJH GradientAccumulator — 梯度累积器
// 当 GPU 显存不足以容纳大 batch 时，将逻辑 batch 拆分为多个物理 mini-batch
// 每步累积梯度（不更新参数），累积 N 步后统一执行 optimizer.step()
// 关键: 每步的损失需除以累积步数 N，使得梯度量级与直接使用大 batch 等价
//
// 使用示例:
//   GradientAccumulator accum(4);  // 逻辑 batch = 4 x 物理 batch
//   for (auto& batch : dataLoader) {
//       Tensor loss = model.forward(batch);
//       Tensor scaledLoss = accum.scaleLoss(loss);  // loss / 4
//       tensorBackward(scaledLoss);
//       accum.step();
//       if (accum.shouldStep()) {
//           optimizer.step();
//           optimizer.zeroGrad();
//           accum.reset();
//       }
//   }
class GradientAccumulator {
public:
    // 20260330 ZJH 构造函数
    // nAccumulationSteps: 梯度累积步数（逻辑 batch 被拆分为 N 个物理 mini-batch）
    //   必须 >= 1，默认 4（即 4 步累积一次更新）
    explicit GradientAccumulator(int nAccumulationSteps = 4)
        : m_nAccumSteps(nAccumulationSteps), m_nCurrentStep(0) {
        // 20260330 ZJH 参数校验：累积步数必须为正整数
        if (m_nAccumSteps < 1) {
            throw std::runtime_error("GradientAccumulator — nAccumulationSteps must be >= 1");
        }
    }

    // 20260330 ZJH shouldStep — 判断是否应该执行优化器更新
    // 当累积计数器达到 m_nAccumSteps 时返回 true
    // 调用时机: 每次 backward + step() 之后检查
    // 返回: true = 应执行 optimizer.step() + zeroGrad + reset
    bool shouldStep() const {
        return m_nCurrentStep >= m_nAccumSteps;  // 20260330 ZJH 达到累积步数
    }

    // 20260330 ZJH scaleLoss — 缩放损失值
    // 将损失除以累积步数 N，使得 N 步累积的梯度等价于一次大 batch 的梯度
    // 数学证明: ∇L_total = (1/N) · Σ ∇L_i，等价于 Σ ∇(L_i / N)
    // loss: 原始损失张量（标量 shape={1}）
    // 返回: 缩放后的损失 = loss / nAccumSteps
    Tensor scaleLoss(const Tensor& loss) {
        // 20260330 ZJH 累积步数为 1 时无需缩放（避免不必要的除法运算）
        if (m_nAccumSteps == 1) {
            return loss;  // 20260330 ZJH 直接返回原始损失
        }
        // 20260330 ZJH 构造缩放因子张量 (1/N)，与损失逐元素相乘
        float fScale = 1.0f / static_cast<float>(m_nAccumSteps);  // 20260330 ZJH 缩放因子
        Tensor scaleFactor = Tensor::full({1}, fScale, loss.device());  // 20260330 ZJH 标量张量
        return tensorMul(loss, scaleFactor);  // 20260330 ZJH loss * (1/N)
    }

    // 20260330 ZJH step — 递增累积计数器
    // 每次 backward 之后调用，记录已累积的梯度步数
    void step() {
        ++m_nCurrentStep;  // 20260330 ZJH 计数器 +1
    }

    // 20260330 ZJH reset — 重置累积计数器为 0
    // 在 optimizer.step() + zeroGrad 之后调用，开始下一轮累积
    void reset() {
        m_nCurrentStep = 0;  // 20260330 ZJH 清零
    }

    // 20260330 ZJH getAccumulationSteps — 获取累积步数配置
    int getAccumulationSteps() const {
        return m_nAccumSteps;  // 20260330 ZJH 返回 N
    }

    // 20260330 ZJH getCurrentStep — 获取当前累积到第几步
    int getCurrentStep() const {
        return m_nCurrentStep;  // 20260330 ZJH 返回当前计数
    }

    // 20260330 ZJH getProgress — 获取当前累积进度 [0.0, 1.0]
    // 用于 UI 显示或日志打印
    float getProgress() const {
        return static_cast<float>(m_nCurrentStep) / static_cast<float>(m_nAccumSteps);
    }

private:
    int m_nAccumSteps;     // 20260330 ZJH 梯度累积步数 N（构造后不可变）
    int m_nCurrentStep;    // 20260330 ZJH 当前已累积的步数（0 ~ N）
};

}  // namespace om
