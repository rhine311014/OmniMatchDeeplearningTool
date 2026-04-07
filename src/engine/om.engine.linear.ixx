// 20260319 ZJH Linear 全连接层模块 — Phase 2 Part 1
// 实现 y = x @ W + b 线性变换，支持 Kaiming 初始化和可选偏置
module;  // 20260406 ZJH 全局模块片段声明，允许在模块单元中使用传统 #include

#include <vector>   // 20260406 ZJH 标准库动态数组容器（本模块未直接使用，保留供扩展）
#include <string>   // 20260406 ZJH 标准库字符串类（本模块未直接使用，保留供扩展）
#include <cmath>    // 20260406 ZJH 数学函数库，提供 std::sqrt 用于 Kaiming 初始化缩放因子计算

export module om.engine.linear;  // 20260406 ZJH 导出模块声明：om.engine.linear 全连接层模块

// 20260319 ZJH 导入依赖模块：张量类、张量运算、模块基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {  // 20260406 ZJH 导出命名空间 om，所有深度学习引擎组件统一归属此命名空间

// 20260319 ZJH Linear — 全连接层：y = x @ W + b
// W 形状 [nInFeatures, nOutFeatures]，b 形状 [1, nOutFeatures]
// 采用 Kaiming 初始化：W ~ N(0, sqrt(2/fan_in))
class Linear : public Module {
public:
    // 20260319 ZJH 构造函数：初始化权重和偏置
    // nInFeatures: 输入特征维度
    // nOutFeatures: 输出特征维度
    // bBias: 是否使用偏置，默认 true
    Linear(int nInFeatures, int nOutFeatures, bool bBias = true)
        : m_nInFeatures(nInFeatures), m_nOutFeatures(nOutFeatures), m_bUseBias(bBias)
    {
        // 20260319 ZJH Kaiming 初始化：W ~ N(0, sqrt(2/fan_in))
        m_weight = Tensor::randn({nInFeatures, nOutFeatures});  // 20260319 ZJH 标准正态随机权重
        float fScale = std::sqrt(2.0f / static_cast<float>(nInFeatures));  // 20260319 ZJH 缩放因子
        float* pW = m_weight.mutableFloatDataPtr();  // 20260319 ZJH 权重可写指针
        // 20260319 ZJH 逐元素乘以缩放因子完成 Kaiming 初始化
        for (int i = 0; i < m_weight.numel(); ++i) {
            pW[i] *= fScale;  // 20260319 ZJH 缩放权重
        }
        registerParameter("weight", m_weight);  // 20260319 ZJH 注册权重参数（自动开启梯度）

        // 20260319 ZJH 可选偏置初始化为零
        if (bBias) {
            m_bias = Tensor::zeros({1, nOutFeatures});  // 20260319 ZJH 零偏置 [1, out]
            registerParameter("bias", m_bias);  // 20260319 ZJH 注册偏置参数
        }
    }

    // 20260319 ZJH forward — 前向传播：y = x @ W + b
    // input: [batch, nInFeatures] 输入张量
    // 返回: [batch, nOutFeatures] 输出张量
    Tensor forward(const Tensor& input) override {
        // 20260319 ZJH 矩阵乘法：[batch, in] @ [in, out] -> [batch, out]
        auto result = tensorMatmul(input, m_weight);
        // 20260319 ZJH 若启用偏置，广播加法：[batch, out] + [1, out] -> [batch, out]
        if (m_bUseBias) {
            result = tensorAddBias(result, m_bias);  // 20260319 ZJH 加偏置
        }
        return result;  // 20260319 ZJH 返回线性变换结果
    }

private:
    int m_nInFeatures;   // 20260319 ZJH 输入特征维度
    int m_nOutFeatures;  // 20260319 ZJH 输出特征维度
    bool m_bUseBias;     // 20260319 ZJH 是否使用偏置
    Tensor m_weight;     // 20260319 ZJH 权重张量 [in, out]
    Tensor m_bias;       // 20260319 ZJH 偏置张量 [1, out]（无偏置时为空张量）
};

}  // namespace om  // 20260406 ZJH 结束 om 命名空间
