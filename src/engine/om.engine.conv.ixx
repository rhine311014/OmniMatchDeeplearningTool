// 20260319 ZJH 卷积/池化/正则化模块 — Phase 2 Part 2
// Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Dropout, Flatten, Softmax 模块封装
module;

#include <vector>
#include <string>
#include <cmath>
#include <random>

export module om.engine.conv;

// 20260319 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.hal.cpu_backend;
import om.hal.cuda_backend;
import om.engine.autograd;

export namespace om {

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
    // 20260330 ZJH 构造函数增加 nGroups 参数，支持分组/深度可分离卷积
    // nGroups: 分组数（1=标准卷积, nInChannels=深度可分离卷积）
    Conv2d(int nInChannels, int nOutChannels, int nKernelSize,
           int nStride = 1, int nPadding = 0, bool bBias = true, int nGroups = 1)
        : m_nInChannels(nInChannels), m_nOutChannels(nOutChannels),
          m_nKernelSize(nKernelSize), m_nStride(nStride), m_nPadding(nPadding),
          m_bUseBias(bBias), m_nGroups(nGroups)
    {
        // 20260330 ZJH 分组卷积权重形状: [Cout, Cin/Groups, KH, KW]
        int nCinPerGroup = nInChannels / nGroups;  // 20260330 ZJH 每组输入通道数
        m_weight = Tensor::randn({nOutChannels, nCinPerGroup, nKernelSize, nKernelSize});
        // 20260330 ZJH Kaiming 初始化基于每组输入通道数（非总通道数）
        float fFanIn = static_cast<float>(nCinPerGroup * nKernelSize * nKernelSize);
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
        // 20260330 ZJH 传入 groups 参数支持分组卷积
        return tensorConv2d(input, m_weight, m_bias, m_nStride, m_nPadding, m_nGroups);
    }

private:
    int m_nInChannels;   // 20260319 ZJH 输入通道数
    int m_nOutChannels;  // 20260319 ZJH 输出通道数
    int m_nKernelSize;   // 20260319 ZJH 卷积核大小
    int m_nStride;       // 20260319 ZJH 步幅
    int m_nPadding;      // 20260319 ZJH 填充
    bool m_bUseBias;     // 20260319 ZJH 是否使用偏置
    int m_nGroups;       // 20260330 ZJH 分组数（1=标准, Cin=深度可分离）
    Tensor m_weight;     // 20260330 ZJH 卷积核 [Cout, Cin/Groups, KH, KW]
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
        // 20260326 ZJH running stats 注册为 buffer（独立于 parameters，不影响优化器）
        // GPU 训练时通过 buffers() 收集并迁移到 GPU
        registerBuffer("running_mean", m_runMean);
        registerBuffer("running_var", m_runVar);
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

// 20260402 ZJH GroupNorm2d — 2D 组归一化层
// 将 channels 分成 nGroups 组，每组内独立归一化
// 与 BatchNorm2d 的关键区别:
//   BN: 沿 (N, H, W) 维度归一化，依赖 batch 统计量，batch < 8 时不稳定
//   GN: 沿 (C/G, H, W) 维度归一化，不依赖 batch 维度，batch=1 也稳定
// Halcon/ViDi 在 batch < 8 时自动切换到 GN
class GroupNorm2d : public Module {
public:
    // 20260402 ZJH 构造函数
    // nNumChannels: 输入通道数
    // nGroups: 分组数（默认 32，nNumChannels 必须被 nGroups 整除）
    // fEps: 数值稳定性常数
    GroupNorm2d(int nNumChannels, int nGroups = 32, float fEps = 1e-5f)
        : m_nNumChannels(nNumChannels), m_nGroups(nGroups), m_fEps(fEps)
    {
        // 20260402 ZJH 自动调整 groups: 确保 channels 能被 groups 整除
        while (m_nGroups > 1 && nNumChannels % m_nGroups != 0) --m_nGroups;

        m_gamma = Tensor::ones({nNumChannels});   // 20260402 ZJH gamma 初始化为 1
        m_beta = Tensor::zeros({nNumChannels});   // 20260402 ZJH beta 初始化为 0
        registerParameter("gamma", m_gamma);
        registerParameter("beta", m_beta);
        // 20260402 ZJH GroupNorm 没有 running stats（不需要 train/eval 切换）
    }

    // 20260402 ZJH forward — 前向传播
    // input: [N, C, H, W]
    Tensor forward(const Tensor& input) override {
        return tensorGroupNorm2d(input, m_gamma, m_beta, m_nGroups, m_fEps);
    }

    // 20260402 ZJH GroupNorm 没有 buffers（无 running stats）
    std::vector<Tensor*> buffers() override { return {}; }
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& = "") override { return {}; }

    int groups() const { return m_nGroups; }  // 20260402 ZJH 返回实际分组数

private:
    int m_nNumChannels;  // 20260402 ZJH 通道数
    int m_nGroups;       // 20260402 ZJH 分组数
    float m_fEps;        // 20260402 ZJH 数值稳定性常数
    Tensor m_gamma;      // 20260402 ZJH 缩放参数 [C]
    Tensor m_beta;       // 20260402 ZJH 偏移参数 [C]
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
            float fSum = 0.0f;  // 20260330 ZJH softmax 分母累加器
            for (int j = 0; j < nClasses; ++j) {
                pOutRow[j] = std::exp(pRow[j] - fMax);
                fSum += pOutRow[j];
            }
            if (fSum < 1e-30f) fSum = 1e-30f;  // 20260330 ZJH 防止除零（全 -inf 输入时 fSum 可能为 0）
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

// 20260320 ZJH DilatedConv2d — 膨胀卷积（空洞卷积）模块
// DeepLabV3 ASPP 的核心组件，通过 dilation 参数扩大感受野而不增加参数量
class DilatedConv2d : public Module {
public:
    // 20260320 ZJH 构造函数
    // nDilation: 膨胀率（1=标准卷积, 6/12/18=ASPP 典型值）
    // nGroups: 分组数（1=标准, nInChannels=深度可分离）
    DilatedConv2d(int nInChannels, int nOutChannels, int nKernelSize,
                  int nStride = 1, int nPadding = 0, int nDilation = 1,
                  int nGroups = 1, bool bBias = true)
        : m_nInChannels(nInChannels), m_nOutChannels(nOutChannels),
          m_nKernelSize(nKernelSize), m_nStride(nStride), m_nPadding(nPadding),
          m_nDilation(nDilation), m_nGroups(nGroups), m_bUseBias(bBias)
    {
        int nCinPerGroup = nInChannels / nGroups;
        m_weight = Tensor::randn({nOutChannels, nCinPerGroup, nKernelSize, nKernelSize});
        float fFanIn = static_cast<float>(nCinPerGroup * nKernelSize * nKernelSize);
        float fScale = std::sqrt(2.0f / fFanIn);
        float* pW = m_weight.mutableFloatDataPtr();
        for (int i = 0; i < m_weight.numel(); ++i) pW[i] *= fScale;
        registerParameter("weight", m_weight);
        if (bBias) {
            m_bias = Tensor::zeros({nOutChannels});
            registerParameter("bias", m_bias);
        }
    }

    Tensor forward(const Tensor& input) override {
        auto cInput = input.contiguous();
        auto cWeight = m_weight.contiguous();
        int nBatch = cInput.shape(0);
        int nH = cInput.shape(2);
        int nW = cInput.shape(3);
        int nEffKH = m_nKernelSize + (m_nKernelSize - 1) * (m_nDilation - 1);
        int nEffKW = nEffKH;
        int nHout = (nH + 2 * m_nPadding - nEffKH) / m_nStride + 1;
        int nWout = (nW + 2 * m_nPadding - nEffKW) / m_nStride + 1;

        // 20260331 ZJH GPU 路径: 原生 CUDA 膨胀 im2col+GEMM + autograd（零 D2H）
        if (isCudaTensor(input)) {
            auto result = Tensor::zeros({nBatch, m_nOutChannels, nHout, nWout}, DeviceType::CUDA);
            bool bHasBias = (m_bias.numel() > 0);
            const float* pBias = bHasBias ? m_bias.contiguous().floatDataPtr() : nullptr;
            CUDABackend::dilatedConv2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pBias,
                                        result.mutableFloatDataPtr(),
                                        nBatch, m_nInChannels, nH, nW,
                                        m_nOutChannels, m_nKernelSize, m_nKernelSize,
                                        m_nStride, m_nPadding, m_nDilation, m_nGroups);
            // 20260331 ZJH 注册 autograd backward（让 ASPP 膨胀卷积权重可被优化器更新）
            if (input.requiresGrad() || m_weight.requiresGrad()) {
                auto pBackward = std::make_shared<DilatedConv2dBackward>();
                pBackward->m_savedInput = cInput;
                pBackward->m_savedWeight = cWeight;
                pBackward->m_nBatch = nBatch;
                pBackward->m_nCin = m_nInChannels;
                pBackward->m_nH = nH;   pBackward->m_nW = nW;
                pBackward->m_nCout = m_nOutChannels;
                pBackward->m_nKH = m_nKernelSize;  pBackward->m_nKW = m_nKernelSize;
                pBackward->m_nStride = m_nStride;
                pBackward->m_nPad = m_nPadding;
                pBackward->m_nDilation = m_nDilation;
                pBackward->m_bHasBias = bHasBias;
                Tensor tInputRef = input;  // 20260331 ZJH 非 const 拷贝
                pBackward->m_vecInputEdges.push_back(makeEdge(tInputRef, 0));
                pBackward->m_vecInputEdges.push_back(makeEdge(m_weight, 0));
                if (bHasBias) pBackward->m_vecInputEdges.push_back(makeEdge(m_bias, 0));
                result.setGradFnRaw(pBackward);
                result.setRequiresGrad(true);
            }
            return result;
        }
        // 20260331 ZJH CPU 路径: CPUBackend 膨胀卷积
        auto result = Tensor::zeros({nBatch, m_nOutChannels, nHout, nWout});
        bool bHasBias = (m_bias.numel() > 0);
        const float* pBias = bHasBias ? m_bias.contiguous().floatDataPtr() : nullptr;

        CPUBackend::dilatedConv2d(cInput.floatDataPtr(), cWeight.floatDataPtr(), pBias,
                                   result.mutableFloatDataPtr(),
                                   nBatch, m_nInChannels, nH, nW,
                                   m_nOutChannels, m_nKernelSize, m_nKernelSize,
                                   m_nStride, m_nPadding, m_nDilation, m_nGroups);
        return result;
    }

    // 20260330 ZJH 显式重写 parameters() 避免 Module 基类默认实现在 C++23 模块边界下的指针失效问题
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        v.push_back(&m_weight);
        if (m_bUseBias && m_bias.numel() > 0) v.push_back(&m_bias);
        return v;
    }

private:
    int m_nInChannels, m_nOutChannels, m_nKernelSize;
    int m_nStride, m_nPadding, m_nDilation, m_nGroups;
    bool m_bUseBias;
    Tensor m_weight, m_bias;
};

// 20260320 ZJH Dropout2d — 空间 Dropout（整个通道置零）
// 训练时以概率 p 将整个特征图通道置零，评估时直接透传
class Dropout2d : public Module {
public:
    Dropout2d(float fProb = 0.5f) : m_fProb(fProb) {}

    // 20260329 ZJH Dropout2d forward — 用 tensorMul 保持 autograd 链
    // 旧实现: Tensor::zeros + 手动拷贝 → 返回的 tensor 无 gradFn → 梯度断链
    // 新实现: 生成 mask → tensorMul(input, mask) → MulBackward 自动注册 → 梯度正常回传
    Tensor forward(const Tensor& input) override {
        if (!m_bTraining || m_fProb <= 0.0f) {
            // 20260325 ZJH 评估模式直接透传（零拷贝视图）
            return tensorReshape(input.contiguous(), input.shapeVec());
        }
        // 20260329 ZJH 生成通道级 mask [B, C, H, W]（CPU 上生成，GPU 时上传）
        int nBatch = input.shape(0);
        int nChannels = input.shape(1);
        int nSpatial = input.numel() / (nBatch * nChannels);
        float fScale = 1.0f / (1.0f - m_fProb);  // 20260329 ZJH 逆概率缩放

        auto tMask = Tensor::zeros(input.shapeVec());  // 20260329 ZJH CPU 上构建 mask
        float* pMask = tMask.mutableFloatDataPtr();

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannels; ++c) {
                // 20260329 ZJH 每通道随机: 保留→fScale, 丢弃→0
                float fVal = (dist(gen) >= m_fProb) ? fScale : 0.0f;
                int nOffset = (n * nChannels + c) * nSpatial;
                for (int s = 0; s < nSpatial; ++s) pMask[nOffset + s] = fVal;
            }
        }

        // 20260329 ZJH GPU 时上传 mask
        if (input.isCuda()) tMask = tMask.cuda();

        // 20260329 ZJH tensorMul 自带 autograd: backward 时 gradInput = gradOutput * mask
        return tensorMul(input, tMask);
    }

private:
    float m_fProb;
};

// 20260320 ZJH LayerNorm — 层归一化模块
// 沿最后一维归一化，Transformer/ViT 中标准归一化层
// 与 BatchNorm 不同：BN 沿 batch 和空间维归一化，LN 沿特征维归一化
class LayerNorm : public Module {
public:
    // 20260320 ZJH 构造函数
    // nDim: 归一化的维度大小（最后一维）
    // fEps: 数值稳定性常数，默认 1e-5
    LayerNorm(int nDim, float fEps = 1e-5f)
        : m_nDim(nDim), m_fEps(fEps)
    {
        m_gamma = Tensor::ones({nDim});   // 20260320 ZJH gamma 初始化为 1
        m_beta = Tensor::zeros({nDim});   // 20260320 ZJH beta 初始化为 0
        registerParameter("gamma", m_gamma);
        registerParameter("beta", m_beta);
    }

    Tensor forward(const Tensor& input) override {
        return tensorLayerNorm(input, m_gamma, m_beta, m_fEps);
    }

private:
    int m_nDim;      // 20260320 ZJH 归一化维度大小
    float m_fEps;    // 20260320 ZJH 数值稳定性常数
    Tensor m_gamma;  // 20260320 ZJH 缩放参数 [dim]
    Tensor m_beta;   // 20260320 ZJH 偏移参数 [dim]
};

// 20260320 ZJH AdaptiveAvgPool2d — 自适应平均池化模块
// 将 [N, C, H, W] 池化到 [N, C, outH, outW]
class AdaptiveAvgPool2d : public Module {
public:
    // 20260320 ZJH 构造函数
    // nOutH: 目标输出高度
    // nOutW: 目标输出宽度
    AdaptiveAvgPool2d(int nOutH, int nOutW)
        : m_nOutH(nOutH), m_nOutW(nOutW) {}

    Tensor forward(const Tensor& input) override {
        return tensorAdaptiveAvgPool2d(input, m_nOutH, m_nOutW);
    }

private:
    int m_nOutH;  // 20260320 ZJH 目标输出高度
    int m_nOutW;  // 20260320 ZJH 目标输出宽度
};

}  // namespace om
