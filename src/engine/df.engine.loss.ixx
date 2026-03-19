// 20260319 ZJH 损失函数模块 — Phase 2 Part 1
// 实现 CrossEntropyLoss 和 MSELoss 两种损失函数
module;

export module df.engine.loss;

// 20260319 ZJH 导入依赖模块：张量类、张量运算、模块基类
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;

export namespace df {

// 20260319 ZJH CrossEntropyLoss — Softmax + 交叉熵联合损失
// 联合计算确保数值稳定性，梯度公式简单：(softmax - target) / batch
class CrossEntropyLoss {
public:
    // 20260319 ZJH forward — 计算 softmax 交叉熵损失
    // logits: [batch, classes] 原始未归一化分数
    // targets: [batch, classes] one-hot 编码标签
    // 返回: 标量损失张量（shape={1}），支持反向传播
    Tensor forward(const Tensor& logits, const Tensor& targets) {
        return tensorSoftmaxCrossEntropy(logits, targets);  // 20260319 ZJH 调用联合前向（含 autograd）
    }
};

// 20260319 ZJH MSELoss — 均方误差损失
// loss = sum((predictions - targets)^2)
class MSELoss {
public:
    // 20260319 ZJH forward — 计算均方误差损失
    // predictions: [batch, dim] 模型预测值
    // targets: [batch, dim] 目标值
    // 返回: 标量损失张量（shape={1}），支持反向传播
    Tensor forward(const Tensor& predictions, const Tensor& targets) {
        auto diff = tensorSub(predictions, targets);  // 20260319 ZJH 差值：pred - target
        auto sq = tensorMul(diff, diff);  // 20260319 ZJH 逐元素平方
        return tensorSum(sq);  // 20260319 ZJH 全局求和返回标量
    }
};

}  // namespace df
