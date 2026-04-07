// 20260319 ZJH 激活函数模块 — Phase 2 Part 1
// 实现 ReLU 等激活函数的 Module 封装
module;  // 20260406 ZJH 全局模块片段声明，允许在模块单元中使用传统 #include

export module om.engine.activations;  // 20260406 ZJH 导出模块声明：om.engine.activations 激活函数模块

// 20260319 ZJH 导入依赖模块：张量类、张量运算、模块基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {  // 20260406 ZJH 导出命名空间 om，所有深度学习引擎组件统一归属此命名空间

// 20260319 ZJH ReLU — 修正线性单元激活模块
// 前向: out = max(0, input)，无可训练参数
class ReLU : public Module {
public:
    // 20260319 ZJH forward — 对输入逐元素执行 ReLU 激活
    // input: 任意形状的输入张量
    // 返回: 与输入同形状的激活输出张量
    Tensor forward(const Tensor& input) override {
        return tensorReLU(input);  // 20260319 ZJH 调用 tensor_ops 的 ReLU 运算（支持自动微分）
    }
};

// 20260320 ZJH Sigmoid — S 型激活模块
// 前向: out = 1 / (1 + exp(-input))，输出范围 [0, 1]，无可训练参数
class Sigmoid : public Module {
public:
    // 20260320 ZJH forward — 对输入逐元素执行 Sigmoid 激活
    Tensor forward(const Tensor& input) override {
        return tensorSigmoid(input);  // 20260320 ZJH 调用 tensor_ops 的 Sigmoid 运算
    }
};

// 20260320 ZJH LeakyReLU — 带泄漏的修正线性单元激活模块
// 前向: out = x > 0 ? x : slope * x，负区域保留小梯度
class LeakyReLU : public Module {
public:
    // 20260320 ZJH 构造函数
    // fSlope: 负区域斜率，默认 0.01
    LeakyReLU(float fSlope = 0.01f) : m_fSlope(fSlope) {}

    // 20260320 ZJH forward — 对输入逐元素执行 LeakyReLU 激活
    Tensor forward(const Tensor& input) override {
        return tensorLeakyReLU(input, m_fSlope);  // 20260320 ZJH 调用 tensor_ops 的 LeakyReLU 运算
    }

private:
    float m_fSlope;  // 20260320 ZJH 负区域斜率
};

// 20260320 ZJH GELU — 高斯误差线性单元激活模块
// Transformer/ViT 中标准激活函数，比 ReLU 更平滑
class GELU : public Module {
public:
    // 20260406 ZJH forward — 对输入逐元素执行 GELU 激活
    // input: 任意形状的输入张量
    // 返回: 与输入同形状的 GELU 激活输出张量
    Tensor forward(const Tensor& input) override {
        return tensorGELU(input);  // 20260320 ZJH 调用 tensor_ops 的 GELU 运算
    }
};

// 20260320 ZJH SiLU — SiLU (Swish) 激活模块
// SiLU(x) = x * sigmoid(x)，YOLOv5/v8 中常用
class SiLU : public Module {
public:
    // 20260406 ZJH forward — 对输入逐元素执行 SiLU (Swish) 激活
    // input: 任意形状的输入张量
    // 返回: 与输入同形状的 SiLU 激活输出张量
    Tensor forward(const Tensor& input) override {
        return tensorSiLU(input);  // 20260320 ZJH 调用 tensor_ops 的 SiLU 运算
    }
};

// 20260321 ZJH Tanh — 双曲正切激活模块
// LSTM 核心激活函数，输出范围 [-1, 1]
class Tanh : public Module {
public:
    // 20260406 ZJH forward — 对输入逐元素执行 Tanh 激活
    // input: 任意形状的输入张量
    // 返回: 与输入同形状的 Tanh 激活输出张量，值域 [-1, 1]
    Tensor forward(const Tensor& input) override {
        return tensorTanh(input);  // 20260321 ZJH 调用 tensor_ops 的 Tanh 运算
    }
};

}  // namespace om  // 20260406 ZJH 结束 om 命名空间
