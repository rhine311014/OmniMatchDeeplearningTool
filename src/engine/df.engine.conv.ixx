// 20260319 ZJH 卷积/池化/正则化模块 — Phase 2 Part 2
// Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Dropout, Flatten, Softmax 模块封装
module;

#include <vector>
#include <string>
#include <cmath>

export module df.engine.conv;

// 20260319 ZJH 导入依赖模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;

export namespace df {

// 20260319 ZJH Conv2d — 2D 卷积层
// 权重形状: [Cout, Cin, KH, KW]，偏置形状: [Cout]
// 前向: output = conv2d(input, weight, bias, stride, padding)
class Conv2d : public Module {
public:
    // 20260319 ZJH 构造函数
    // nInChannels: 输入通道数
    // nOutChannels: 输出通道数
    // nKernelSize: 卷积核大小（正方形）
    // nStride: 步幅，默认 1
    // nPadding: 填充，默认 0
    // bBias: 是否使用偏置，默认 true
    Conv2d(int nInChannels, int nOutChannels, int nKernelSize,
           int nStride = 1, int nPadding = 0, bool bBias = true)
        : m_nInChannels(nInChannels), m_nOutChannels(nOutChannels),
          m_nKernelSize(nKernelSize), m_nStride(nStride), m_nPadding(nPadding),
          m_bUseBias(bBias)
    {
        // 20260319 ZJH Kaiming 初始化：W ~ N(0, sqrt(2 / (Cin * KH * KW)))
        m_weight = Tensor::randn({nOutChannels, nInChannels, nKernelSize, nKernelSize});
        float fFanIn = static_cast<float>(nInChannels * nKernelSize * nKernelSize);
        float fScale = std::sqrt(2.0f / fFanIn);  // 20260319 ZJH Kaiming 缩放因子
        float* pW = m_weight.mutableFloatDataPtr();
        for (int i = 0; i < m_weight.numel(); ++i) {
            pW[i] *= fScale;  // 20260319 ZJH 缩放权重
        }
        registerParameter("weight", m_weight);  // 20260319 ZJH 注册权重参数

        if (bBias) {
            m_bias = Tensor::zeros({nOutChannels});  // 20260319 ZJH 零初始化偏置
            registerParameter("bias", m_bias);       // 20260319 ZJH 注册偏置参数
        }
    }

    // 20260319 ZJH forward — 前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, Hout, Wout]
    Tensor forward(const Tensor& input) override {
        return tensorConv2d(input, m_weight, m_bias, m_nStride, m_nPadding);
    }

private:
    int m_nInChannels;   // 20260319 ZJH 输入通道数
    int m_nOutChannels;  // 20260319 ZJH 输出通道数
    int m_nKernelSize;   // 20260319 ZJH 卷积核大小
    int m_nStride;       // 20260319 ZJH 步幅
    int m_nPadding;      // 20260319 ZJH 填充
    bool m_bUseBias;     // 20260319 ZJH 是否使用偏置
    Tensor m_weight;     // 20260319 ZJH 卷积核 [Cout, Cin, KH, KW]
    Tensor m_bias;       // 20260319 ZJH 偏置 [Cout]
};

// 20260319 ZJH BatchNorm2d — 2D 批归一化层
// 维护 gamma, beta 可训练参数和 running_mean, running_var 非训练状态
class BatchNorm2d : public Module {
public:
    // 20260319 ZJH 构造函数
    // nNumFeatures: 通道数
    // fEps: 数值稳定性常数，默认 1e-5
    // fMomentum: running 统计量动量，默认 0.1
    BatchNorm2d(int nNumFeatures, float fEps = 1e-5f, float fMomentum = 0.1f)
        : m_nNumFeatures(nNumFeatures), m_fEps(fEps), m_fMomentum(fMomentum)
    {
        m_gamma = Tensor::ones({nNumFeatures});     // 20260319 ZJH gamma 初始化为 1
        m_beta = Tensor::zeros({nNumFeatures});     // 20260319 ZJH beta 初始化为 0
        m_runMean = Tensor::zeros({nNumFeatures});  // 20260319 ZJH running mean 初始化为 0
        m_runVar = Tensor::ones({nNumFeatures});    // 20260319 ZJH running var 初始化为 1
        registerParameter("gamma", m_gamma);  // 20260319 ZJH 注册 gamma 参数
        registerParameter("beta", m_beta);    // 20260319 ZJH 注册 beta 参数
        // 20260319 ZJH running_mean 和 running_var 不注册为参数（不被优化器更新）
    }

    // 20260319 ZJH forward — 前向传播
    // input: [N, C, H, W]
    Tensor forward(const Tensor& input) override {
        return tensorBatchNorm2d(input, m_gamma, m_beta,
                                  m_runMean, m_runVar,
                                  m_bTraining, m_fEps, m_fMomentum);
    }

    // 20260319 ZJH 获取 running 统计量（用于序列化）
    Tensor& runningMean() { return m_runMean; }
    Tensor& runningVar() { return m_runVar; }

private:
    int m_nNumFeatures;   // 20260319 ZJH 通道数
    float m_fEps;          // 20260319 ZJH 数值稳定性常数
    float m_fMomentum;     // 20260319 ZJH running 统计量动量
    Tensor m_gamma;        // 20260319 ZJH 缩放参数 [C]
    Tensor m_beta;         // 20260319 ZJH 偏移参数 [C]
    Tensor m_runMean;      // 20260319 ZJH 运行均值 [C]（非训练参数）
    Tensor m_runVar;       // 20260319 ZJH 运行方差 [C]（非训练参数）
};

// 20260319 ZJH MaxPool2d — 2D 最大池化层
class MaxPool2d : public Module {
public:
    // 20260319 ZJH 构造函数
    // nKernelSize: 池化窗口大小
    // nStride: 步幅，默认与 kernelSize 相同
    // nPadding: 填充，默认 0
    MaxPool2d(int nKernelSize, int nStride = -1, int nPadding = 0)
        : m_nKernelSize(nKernelSize),
          m_nStride(nStride > 0 ? nStride : nKernelSize),  // 20260319 ZJH 默认步幅等于核大小
          m_nPadding(nPadding) {}

    Tensor forward(const Tensor& input) override {
        return tensorMaxPool2d(input, m_nKernelSize, m_nStride, m_nPadding);
    }

private:
    int m_nKernelSize;  // 20260319 ZJH 池化窗口大小
    int m_nStride;      // 20260319 ZJH 步幅
    int m_nPadding;     // 20260319 ZJH 填充
};

// 20260319 ZJH AvgPool2d — 2D 平均池化层
class AvgPool2d : public Module {
public:
    AvgPool2d(int nKernelSize, int nStride = -1, int nPadding = 0)
        : m_nKernelSize(nKernelSize),
          m_nStride(nStride > 0 ? nStride : nKernelSize),
          m_nPadding(nPadding) {}

    Tensor forward(const Tensor& input) override {
        return tensorAvgPool2d(input, m_nKernelSize, m_nStride, m_nPadding);
    }

private:
    int m_nKernelSize;
    int m_nStride;
    int m_nPadding;
};

// 20260319 ZJH Dropout — 随机失活层
// 训练时以概率 p 将元素置零，评估时直接透传
class Dropout : public Module {
public:
    // 20260319 ZJH 构造函数
    // fProb: 失活概率，默认 0.5
    Dropout(float fProb = 0.5f) : m_fProb(fProb) {}

    Tensor forward(const Tensor& input) override {
        return tensorDropout(input, m_fProb, m_bTraining);  // 20260319 ZJH 根据训练模式决定行为
    }

private:
    float m_fProb;  // 20260319 ZJH 失活概率
};

// 20260319 ZJH Flatten — 展平层
// 将 [N, C, H, W, ...] 展平为 [N, C*H*W*...]
class Flatten : public Module {
public:
    // 20260319 ZJH 构造函数
    // nStartDim: 从哪个维度开始展平，默认 1（保留 batch 维）
    Flatten(int nStartDim = 1) : m_nStartDim(nStartDim) {}

    Tensor forward(const Tensor& input) override {
        return tensorFlatten(input, m_nStartDim);
    }

private:
    int m_nStartDim;  // 20260319 ZJH 开始展平的维度
};

// 20260319 ZJH Softmax — Softmax 激活模块
// 沿最后一维做 softmax 归一化（用于推理，训练时通常与 CrossEntropy 联合使用）
class Softmax : public Module {
public:
    Tensor forward(const Tensor& input) override {
        auto cInput = input.contiguous();
        // 20260319 ZJH 假设输入为 [batch, classes] 二维
        int nBatch = cInput.shape(0);
        int nClasses = cInput.shape(1);
        auto result = Tensor::zeros(cInput.shapeVec());
        // 20260319 ZJH 直接调用 softmax（不参与自动微分，推理用）
        // 使用 CPUBackend 中已有的 softmax 内核
        const float* pIn = cInput.floatDataPtr();
        float* pOut = result.mutableFloatDataPtr();
        for (int b = 0; b < nBatch; ++b) {
            const float* pRow = pIn + b * nClasses;
            float* pOutRow = pOut + b * nClasses;
            float fMax = pRow[0];
            for (int j = 1; j < nClasses; ++j)
                if (pRow[j] > fMax) fMax = pRow[j];
            float fSum = 0.0f;
            for (int j = 0; j < nClasses; ++j) {
                pOutRow[j] = std::exp(pRow[j] - fMax);
                fSum += pOutRow[j];
            }
            for (int j = 0; j < nClasses; ++j)
                pOutRow[j] /= fSum;
        }
        return result;
    }
};

// 20260320 ZJH ConvTranspose2d — 2D 转置卷积层（反卷积）
// 权重形状: [Cin, Cout, KH, KW]（注意与 Conv2d 相反），偏置形状: [Cout]
// 前向: output = convTranspose2d(input, weight, bias, stride, padding)
class ConvTranspose2d : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数
    // nOutChannels: 输出通道数
    // nKernelSize: 卷积核大小（正方形）
    // nStride: 步幅，默认 1
    // nPadding: 填充，默认 0
    // bBias: 是否使用偏置，默认 true
    ConvTranspose2d(int nInChannels, int nOutChannels, int nKernelSize,
                    int nStride = 1, int nPadding = 0, bool bBias = true)
        : m_nInChannels(nInChannels), m_nOutChannels(nOutChannels),
          m_nKernelSize(nKernelSize), m_nStride(nStride), m_nPadding(nPadding),
          m_bUseBias(bBias)
    {
        // 20260320 ZJH Kaiming 初始化：权重形状 [Cin, Cout, KH, KW]
        m_weight = Tensor::randn({nInChannels, nOutChannels, nKernelSize, nKernelSize});
        float fFanIn = static_cast<float>(nInChannels * nKernelSize * nKernelSize);
        float fScale = std::sqrt(2.0f / fFanIn);  // 20260320 ZJH Kaiming 缩放因子
        float* pW = m_weight.mutableFloatDataPtr();
        for (int i = 0; i < m_weight.numel(); ++i) {
            pW[i] *= fScale;
        }
        registerParameter("weight", m_weight);

        if (bBias) {
            m_bias = Tensor::zeros({nOutChannels});
            registerParameter("bias", m_bias);
        }
    }

    // 20260320 ZJH forward — 转置卷积前向传播
    // input: [N, Cin, Hin, Win]
    // 返回: [N, Cout, Hout, Wout]，Hout = (Hin-1)*stride - 2*pad + KH
    Tensor forward(const Tensor& input) override {
        return tensorConvTranspose2d(input, m_weight, m_bias, m_nStride, m_nPadding);
    }

private:
    int m_nInChannels;   // 20260320 ZJH 输入通道数
    int m_nOutChannels;  // 20260320 ZJH 输出通道数
    int m_nKernelSize;   // 20260320 ZJH 卷积核大小
    int m_nStride;       // 20260320 ZJH 步幅
    int m_nPadding;      // 20260320 ZJH 填充
    bool m_bUseBias;     // 20260320 ZJH 是否使用偏置
    Tensor m_weight;     // 20260320 ZJH 权重 [Cin, Cout, KH, KW]
    Tensor m_bias;       // 20260320 ZJH 偏置 [Cout]
};

// 20260320 ZJH Upsample — 双线性上采样层
// 将 [N, C, H, W] 上采样为 [N, C, H*scale, W*scale]
class Upsample : public Module {
public:
    // 20260320 ZJH 构造函数
    // nScale: 上采样倍率，默认 2
    Upsample(int nScale = 2) : m_nScale(nScale) {}

    // 20260320 ZJH forward — 双线性上采样前向传播
    Tensor forward(const Tensor& input) override {
        return tensorUpsampleBilinear(input, m_nScale);
    }

private:
    int m_nScale;  // 20260320 ZJH 上采样倍率
};

}  // namespace df
