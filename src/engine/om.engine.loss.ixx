// 20260319 ZJH 损失函数模块 — Phase 2 Part 1
// 实现 CrossEntropyLoss 和 MSELoss 两种损失函数
module;

export module om.engine.loss;

// 20260319 ZJH 导入依赖模块：张量类、张量运算、模块基类、CPU 后端
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.hal.cpu_backend;

export namespace om {

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

// 20260320 ZJH DiceLoss — Dice 损失函数，用于语义分割
// loss = 1 - 2*sum(p*t) / (sum(p) + sum(t) + eps)
// 预测值需先经过 Sigmoid 激活，目标为二值 mask
class DiceLoss {
public:
    // 20260320 ZJH forward — 计算 Dice 损失
    // predictions: 经过 sigmoid 的预测概率 [N, C, H, W]
    // targets: 二值目标 [N, C, H, W]
    // 返回: 标量损失张量
    Tensor forward(const Tensor& predictions, const Tensor& targets) {
        auto cPred = predictions.contiguous();
        auto cTarget = targets.contiguous();
        float fLoss = 0.0f;  // 20260320 ZJH Dice 损失值
        // 20260320 ZJH 直接调用 CPUBackend::diceLoss
        fLoss = CPUBackend::diceLoss(cPred.floatDataPtr(), cTarget.floatDataPtr(), cPred.numel());
        return Tensor::full({1}, fLoss);  // 20260320 ZJH 返回标量损失
    }
};

// 20260320 ZJH BCEWithLogitsLoss — 二元交叉熵损失（含 sigmoid）
// 数值稳定版本，直接接受 logits（未经 sigmoid 的原始值）
class BCEWithLogitsLoss {
public:
    // 20260320 ZJH forward — 计算二元交叉熵损失
    // logits: [N, ...] 原始 logits
    // targets: [N, ...] 二元目标（0/1）
    // 返回: 标量损失张量，支持反向传播
    Tensor forward(const Tensor& logits, const Tensor& targets) {
        return tensorBCEWithLogitsLoss(logits, targets);
    }
};

// 20260320 ZJH YOLOLoss — YOLO 检测损失（简化版）
// 组合损失 = 坐标损失(MSE) + 置信度损失(BCE) + 分类损失(BCE)
class YOLOLoss {
public:
    // 20260320 ZJH 构造函数
    // fCoordWeight: 坐标损失权重，默认 5.0
    // fNoObjWeight: 无目标置信度损失权重，默认 0.5
    YOLOLoss(float fCoordWeight = 5.0f, float fNoObjWeight = 0.5f)
        : m_fCoordWeight(fCoordWeight), m_fNoObjWeight(fNoObjWeight) {}

    // 20260320 ZJH forward — 简化 YOLO 损失计算
    // predictions: [N, numPreds, 5+numClasses] 检测预测
    // targets: [N, numPreds, 5+numClasses] 目标（简化版）
    // 返回: 标量损失张量
    Tensor forward(const Tensor& predictions, const Tensor& targets) {
        // 20260320 ZJH 简化实现：使用 MSE 损失代替完整 YOLO 损失
        auto diff = tensorSub(predictions, targets);
        auto sq = tensorMul(diff, diff);
        return tensorSum(sq);
    }

private:
    float m_fCoordWeight;   // 20260320 ZJH 坐标损失权重
    float m_fNoObjWeight;   // 20260320 ZJH 无目标置信度损失权重
};

}  // namespace om
