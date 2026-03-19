// 20260319 ZJH 激活函数模块 — Phase 2 Part 1
// 实现 ReLU 等激活函数的 Module 封装
module;

export module df.engine.activations;

// 20260319 ZJH 导入依赖模块：张量类、张量运算、模块基类
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;

export namespace df {

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

}  // namespace df
