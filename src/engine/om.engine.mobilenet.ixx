// 20260322 ZJH MobileNetV4-Small 分类网络模块
// 实现倒残差块 (InvertedResidual) + ReLU6 + MobileNetV4-Small 分类网络
// MobileNetV4 核心思想: 倒残差结构 expand→depthwise→project + 线性瓶颈 + 残差连接
// Small 变体: 轻量级分类网络，适合边缘部署和移动端推理
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>

export module om.engine.mobilenet;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;

export namespace om {

// 20260322 ZJH ReLU6 — ReLU6 激活模块
// 前向: out = min(max(x, 0), 6)，MobileNet 系列标准激活函数
// 限制输出范围在 [0, 6]，防止量化精度损失，适合低精度推理
class ReLU6 : public Module {
public:
    // 20260322 ZJH forward — 对输入逐元素执行 ReLU6 激活
    // input: 任意形状的输入张量
    // 返回: 与输入同形状的激活输出张量，值域 [0, 6]
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 先执行 ReLU 将负值截为 0
        auto out = tensorReLU(input);
        // 20260322 ZJH 再用 tensorClip 将大于 6 的值截为 6
        out = tensorClip(out, 0.0f, 6.0f);
        return out;  // 20260322 ZJH 返回 [0, 6] 范围内的激活输出
    }
};

// 20260322 ZJH ConvBnReLU6 — Conv2d + BatchNorm2d + ReLU6 组合模块
// MobileNet 中最基础的构建单元，几乎所有卷积层后都跟 BN + ReLU6
// 20260330 ZJH 新增 nGroups 参数支持深度可分离卷积
class ConvBnReLU6 : public Module {
public:
    // 20260330 ZJH 构造函数增加 nGroups 参数
    // nGroups: 分组数（1=标准卷积, nInChannels=深度可分离卷积）
    ConvBnReLU6(int nInChannels, int nOutChannels, int nKernelSize,
                int nStride = 1, int nPadding = 0, int nGroups = 1)
        : m_conv(nInChannels, nOutChannels, nKernelSize, nStride, nPadding, false, nGroups),  // 20260330 ZJH 传入 groups
          m_bn(nOutChannels)  // 20260322 ZJH 批归一化
    {}

    // 20260322 ZJH forward — Conv → BN → ReLU6 前向传播
    Tensor forward(const Tensor& input) override {
        auto out = m_conv.forward(input);    // 20260322 ZJH 卷积
        out = m_bn.forward(out);             // 20260322 ZJH 批归一化
        out = m_relu6.forward(out);          // 20260322 ZJH ReLU6 激活
        return out;  // 20260322 ZJH 返回卷积+归一化+激活后的特征图
    }

    // 20260322 ZJH 重写 parameters() 收集卷积和 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto vecConvP = m_conv.parameters();  // 20260322 ZJH 卷积参数
        vecResult.insert(vecResult.end(), vecConvP.begin(), vecConvP.end());
        auto vecBnP = m_bn.parameters();  // 20260322 ZJH BN 参数
        vecResult.insert(vecResult.end(), vecBnP.begin(), vecBnP.end());
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters() 收集命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto vecConvP = m_conv.namedParameters(makeP("conv"));
        vecResult.insert(vecResult.end(), vecConvP.begin(), vecConvP.end());
        auto vecBnP = m_bn.namedParameters(makeP("bn"));
        vecResult.insert(vecResult.end(), vecBnP.begin(), vecBnP.end());
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        return m_bn.buffers();  // 20260328 ZJH bn running_mean/running_var
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::string strBnPrefix = strPrefix.empty() ? "bn" : strPrefix + ".bn";
        return m_bn.namedBuffers(strBnPrefix);  // 20260328 ZJH bn 缓冲区
    }

    // 20260322 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv.train(bMode);
        m_bn.train(bMode);
    }

private:
    Conv2d m_conv;        // 20260322 ZJH 卷积层
    BatchNorm2d m_bn;     // 20260322 ZJH 批归一化层
    ReLU6 m_relu6;        // 20260322 ZJH ReLU6 激活（无状态，共用）
};

// 20260322 ZJH ConvBnLinear — Conv2d + BatchNorm2d 组合模块（无激活函数）
// 用于 InvertedResidual 的 project 阶段，线性瓶颈不加激活
class ConvBnLinear : public Module {
public:
    // 20260330 ZJH 构造函数增加 nGroups（保持接口一致）
    ConvBnLinear(int nInChannels, int nOutChannels, int nKernelSize,
                 int nStride = 1, int nPadding = 0, int nGroups = 1)
        : m_conv(nInChannels, nOutChannels, nKernelSize, nStride, nPadding, false, nGroups),
          m_bn(nOutChannels)
    {}

    // 20260322 ZJH forward — Conv → BN 前向传播（无激活）
    Tensor forward(const Tensor& input) override {
        auto out = m_conv.forward(input);    // 20260322 ZJH 卷积
        out = m_bn.forward(out);             // 20260322 ZJH 批归一化
        return out;  // 20260322 ZJH 返回线性投影结果（不经过激活函数）
    }

    // 20260322 ZJH 重写 parameters() 收集卷积和 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto vecConvP = m_conv.parameters();
        vecResult.insert(vecResult.end(), vecConvP.begin(), vecConvP.end());
        auto vecBnP = m_bn.parameters();
        vecResult.insert(vecResult.end(), vecBnP.begin(), vecBnP.end());
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters() 收集命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto vecConvP = m_conv.namedParameters(makeP("conv"));
        vecResult.insert(vecResult.end(), vecConvP.begin(), vecConvP.end());
        auto vecBnP = m_bn.namedParameters(makeP("bn"));
        vecResult.insert(vecResult.end(), vecBnP.begin(), vecBnP.end());
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        return m_bn.buffers();  // 20260328 ZJH bn running_mean/running_var
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::string strBnPrefix = strPrefix.empty() ? "bn" : strPrefix + ".bn";
        return m_bn.namedBuffers(strBnPrefix);  // 20260328 ZJH bn 缓冲区
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv.train(bMode);
        m_bn.train(bMode);
    }

private:
    Conv2d m_conv;        // 20260322 ZJH 卷积层
    BatchNorm2d m_bn;     // 20260322 ZJH 批归一化层
};

// 20260322 ZJH InvertedResidual — MobileNetV2/V4 倒残差块
// 结构: expand Conv1x1+BN+ReLU6 → depthwise Conv3x3+BN+ReLU6 → project Conv1x1+BN（线性）
// 残差连接: 仅当 stride=1 且 inChannels=outChannels 时启用
// 注意: depthwise 卷积使用普通 Conv2d groups=1 近似（tensor_ops 未暴露 depthwiseConv2d）
class InvertedResidual : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入通道数
    // nOutChannels: 输出通道数
    // nStride: depthwise 卷积步幅（1 或 2）
    // nExpandRatio: 扩展比率，expand 后通道数 = nInChannels * nExpandRatio
    InvertedResidual(int nInChannels, int nOutChannels, int nStride, int nExpandRatio)
        : m_nInChannels(nInChannels),
          m_nOutChannels(nOutChannels),
          m_nStride(nStride),
          m_nExpandRatio(nExpandRatio),
          m_bUseResidual(nStride == 1 && nInChannels == nOutChannels)  // 20260322 ZJH 残差连接条件
    {
        int nHiddenChannels = nInChannels * nExpandRatio;  // 20260322 ZJH 扩展后的隐藏通道数

        // 20260322 ZJH expand 阶段: 1x1 卷积扩展通道（仅当扩展比率 > 1 时使用）
        if (nExpandRatio != 1) {
            m_pExpand = std::make_unique<ConvBnReLU6>(nInChannels, nHiddenChannels, 1, 1, 0);
        }

        // 20260330 ZJH depthwise 阶段: 3x3 深度可分离卷积（groups=nHiddenChannels）
        // 每个通道独立卷积，参数量 = nHiddenChannels × 1 × 3 × 3（而非 nHidden² × 3 × 3）
        m_pDepthwise = std::make_unique<ConvBnReLU6>(nHiddenChannels, nHiddenChannels, 3, nStride, 1, nHiddenChannels);

        // 20260322 ZJH project 阶段: 1x1 卷积降维（线性瓶颈，无激活函数）
        m_pProject = std::make_unique<ConvBnLinear>(nHiddenChannels, nOutChannels, 1, 1, 0);
    }

    // 20260322 ZJH forward — 倒残差块前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, Hout, Wout]
    Tensor forward(const Tensor& input) override {
        Tensor out = input;  // 20260322 ZJH 初始化中间结果

        // 20260322 ZJH expand: 1x1 卷积扩展通道（扩展比率 > 1 时）
        if (m_pExpand) {
            out = m_pExpand->forward(out);
        }

        // 20260322 ZJH depthwise: 3x3 卷积提取空间特征
        out = m_pDepthwise->forward(out);

        // 20260322 ZJH project: 1x1 卷积降维（线性瓶颈）
        out = m_pProject->forward(out);

        // 20260322 ZJH 残差连接: 仅当 stride=1 且输入输出通道相同时
        if (m_bUseResidual) {
            out = tensorAdd(out, input);  // 20260322 ZJH 残差加法
        }

        return out;  // 20260322 ZJH 返回倒残差块输出
    }

    // 20260322 ZJH 重写 parameters() 收集所有子模块参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        if (m_pExpand)    appendVec(m_pExpand->parameters());      // 20260322 ZJH expand 参数
        appendVec(m_pDepthwise->parameters());                      // 20260322 ZJH depthwise 参数
        appendVec(m_pProject->parameters());                        // 20260322 ZJH project 参数
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        if (m_pExpand)    appendVec(m_pExpand->namedParameters(makeP("expand")));
        appendVec(m_pDepthwise->namedParameters(makeP("depthwise")));
        appendVec(m_pProject->namedParameters(makeP("project")));
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集子模块 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        if (m_pExpand)    appendVec(m_pExpand->buffers());     // 20260328 ZJH expand 内部 BN 缓冲区
        appendVec(m_pDepthwise->buffers());                     // 20260328 ZJH depthwise 内部 BN 缓冲区
        appendVec(m_pProject->buffers());                       // 20260328 ZJH project 内部 BN 缓冲区
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集子模块 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        if (m_pExpand)    appendVec(m_pExpand->namedBuffers(makeP("expand")));
        appendVec(m_pDepthwise->namedBuffers(makeP("depthwise")));
        appendVec(m_pProject->namedBuffers(makeP("project")));
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        if (m_pExpand)    m_pExpand->train(bMode);
        m_pDepthwise->train(bMode);
        m_pProject->train(bMode);
    }

private:
    int m_nInChannels;      // 20260322 ZJH 输入通道数
    int m_nOutChannels;     // 20260322 ZJH 输出通道数
    int m_nStride;          // 20260322 ZJH depthwise 步幅
    int m_nExpandRatio;     // 20260322 ZJH 扩展比率
    bool m_bUseResidual;    // 20260322 ZJH 是否使用残差连接

    // 20260322 ZJH 使用 unique_ptr 避免移动后指针失效
    std::unique_ptr<ConvBnReLU6> m_pExpand;       // 20260322 ZJH expand 1x1 卷积（可选）
    std::unique_ptr<ConvBnReLU6> m_pDepthwise;    // 20260322 ZJH depthwise 3x3 卷积
    std::unique_ptr<ConvBnLinear> m_pProject;     // 20260322 ZJH project 1x1 线性卷积
};

// 20260322 ZJH MobileNetV4Small — MobileNetV4-Small 分类网络
// 网络结构 (Small):
//   stem(3→16) → 6 个 InvertedResidual 阶段 → head(160→1280→nClasses)
//   阶段配置 [expand_ratio, out_channels, num_blocks, stride]:
//     Stage 1: [1, 16, 1, 1]  → 16→16
//     Stage 2: [6, 24, 2, 2]  → 16→24, stride=2
//     Stage 3: [6, 32, 3, 2]  → 24→32, stride=2
//     Stage 4: [6, 64, 4, 2]  → 32→64, stride=2
//     Stage 5: [6, 96, 3, 1]  → 64→96
//     Stage 6: [6, 160, 3, 2] → 96→160, stride=2
//   head: Conv1x1(160→1280) + AdaptiveAvgPool2d(1,1) + Linear(1280, nClasses)
class MobileNetV4Small : public Module {
public:
    // 20260322 ZJH 构造函数
    // nNumClasses: 分类类别数，默认 10
    // nInChannels: 输入图像通道数，默认 3（RGB）
    MobileNetV4Small(int nNumClasses = 10, int nInChannels = 3)
        : m_nNumClasses(nNumClasses),
          m_stem(nInChannels, 16, 3, 2, 1),  // 20260322 ZJH stem: Conv3x3 stride=2，输入通道→16
          m_headConv(160, 1280, 1, 1, 0),    // 20260322 ZJH head Conv1x1: 160→1280
          m_adaptivePool(1, 1),               // 20260322 ZJH 自适应平均池化到 1x1
          m_fc(1280, nNumClasses)              // 20260322 ZJH 全连接分类器: 1280 → nClasses
    {
        // 20260322 ZJH 构建 6 个 InvertedResidual 阶段
        // 阶段配置: {expand_ratio, out_channels, num_blocks, stride_first_block}
        // Stage 1: 16→16, expand=1, 1 block, stride=1
        m_vecStages.reserve(6);  // 20260322 ZJH 预分配 6 个阶段

        // 20260322 ZJH Stage 1: 1 个块，16→16，expand=1，stride=1
        {
            std::vector<std::unique_ptr<InvertedResidual>> vecBlocks;
            vecBlocks.push_back(std::make_unique<InvertedResidual>(16, 16, 1, 1));
            m_vecStages.push_back(std::move(vecBlocks));
        }

        // 20260322 ZJH Stage 2: 2 个块，16→24，expand=6，第一块 stride=2
        {
            std::vector<std::unique_ptr<InvertedResidual>> vecBlocks;
            vecBlocks.push_back(std::make_unique<InvertedResidual>(16, 24, 2, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(24, 24, 1, 6));
            m_vecStages.push_back(std::move(vecBlocks));
        }

        // 20260322 ZJH Stage 3: 3 个块，24→32，expand=6，第一块 stride=2
        {
            std::vector<std::unique_ptr<InvertedResidual>> vecBlocks;
            vecBlocks.push_back(std::make_unique<InvertedResidual>(24, 32, 2, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(32, 32, 1, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(32, 32, 1, 6));
            m_vecStages.push_back(std::move(vecBlocks));
        }

        // 20260322 ZJH Stage 4: 4 个块，32→64，expand=6，第一块 stride=2
        {
            std::vector<std::unique_ptr<InvertedResidual>> vecBlocks;
            vecBlocks.push_back(std::make_unique<InvertedResidual>(32, 64, 2, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(64, 64, 1, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(64, 64, 1, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(64, 64, 1, 6));
            m_vecStages.push_back(std::move(vecBlocks));
        }

        // 20260322 ZJH Stage 5: 3 个块，64→96，expand=6，stride=1（不下采样）
        {
            std::vector<std::unique_ptr<InvertedResidual>> vecBlocks;
            vecBlocks.push_back(std::make_unique<InvertedResidual>(64, 96, 1, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(96, 96, 1, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(96, 96, 1, 6));
            m_vecStages.push_back(std::move(vecBlocks));
        }

        // 20260322 ZJH Stage 6: 3 个块，96→160，expand=6，第一块 stride=2
        {
            std::vector<std::unique_ptr<InvertedResidual>> vecBlocks;
            vecBlocks.push_back(std::make_unique<InvertedResidual>(96, 160, 2, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(160, 160, 1, 6));
            vecBlocks.push_back(std::make_unique<InvertedResidual>(160, 160, 1, 6));
            m_vecStages.push_back(std::move(vecBlocks));
        }
    }

    // 20260322 ZJH forward — MobileNetV4-Small 前向传播
    // input: [N, 3, H, W]（RGB 图像）
    // 返回: [N, nNumClasses]（分类 logits）
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH stem: Conv3x3 stride=2 + BN + ReLU6
        auto x = m_stem.forward(input);  // 20260322 ZJH [N,3,H,W] → [N,16,H/2,W/2]

        // 20260322 ZJH 6 个 InvertedResidual 阶段
        for (auto& vecBlocks : m_vecStages) {
            for (auto& pBlock : vecBlocks) {
                x = pBlock->forward(x);  // 20260322 ZJH 逐块前向传播
            }
        }

        // 20260322 ZJH head: Conv1x1(160→1280) + BN + ReLU6
        x = m_headConv.forward(x);  // 20260322 ZJH [N,160,H',W'] → [N,1280,H',W']

        // 20260322 ZJH 自适应平均池化: [N,1280,H',W'] → [N,1280,1,1]
        x = m_adaptivePool.forward(x);

        // 20260322 ZJH 展平: [N,1280,1,1] → [N,1280]
        x = m_flatten.forward(x);

        // 20260322 ZJH 全连接分类器: [N,1280] → [N,nClasses]
        x = m_fc.forward(x);

        return x;  // 20260322 ZJH 返回分类 logits
    }

    // 20260322 ZJH 重写 parameters() 收集所有子模块参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        // 20260322 ZJH stem 参数
        appendVec(m_stem.parameters());

        // 20260322 ZJH 各阶段 InvertedResidual 参数
        for (auto& vecBlocks : m_vecStages) {
            for (auto& pBlock : vecBlocks) {
                appendVec(pBlock->parameters());
            }
        }

        // 20260322 ZJH head 参数
        appendVec(m_headConv.parameters());
        appendVec(m_fc.parameters());

        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        // 20260322 ZJH stem
        appendVec(m_stem.namedParameters(makeP("stem")));

        // 20260322 ZJH 各阶段
        for (size_t s = 0; s < m_vecStages.size(); ++s) {
            for (size_t b = 0; b < m_vecStages[s].size(); ++b) {
                std::string strName = "stage" + std::to_string(s) + ".block" + std::to_string(b);
                appendVec(m_vecStages[s][b]->namedParameters(makeP(strName)));
            }
        }

        // 20260322 ZJH head
        appendVec(m_headConv.namedParameters(makeP("head_conv")));
        appendVec(m_fc.namedParameters(makeP("fc")));

        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        // 20260328 ZJH stem 内部 BN 缓冲区
        appendVec(m_stem.buffers());
        // 20260328 ZJH 各阶段 InvertedResidual 内部 BN 缓冲区
        for (auto& vecBlocks : m_vecStages) {
            for (auto& pBlock : vecBlocks) {
                appendVec(pBlock->buffers());
            }
        }
        // 20260328 ZJH head 内部 BN 缓冲区
        appendVec(m_headConv.buffers());
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集所有 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        // 20260328 ZJH stem
        appendVec(m_stem.namedBuffers(makeP("stem")));
        // 20260328 ZJH 各阶段
        for (size_t s = 0; s < m_vecStages.size(); ++s) {
            for (size_t b = 0; b < m_vecStages[s].size(); ++b) {
                std::string strName = "stage" + std::to_string(s) + ".block" + std::to_string(b);
                appendVec(m_vecStages[s][b]->namedBuffers(makeP(strName)));
            }
        }
        // 20260328 ZJH head
        appendVec(m_headConv.namedBuffers(makeP("head_conv")));
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_stem.train(bMode);
        for (auto& vecBlocks : m_vecStages) {
            for (auto& pBlock : vecBlocks) {
                pBlock->train(bMode);
            }
        }
        m_headConv.train(bMode);
        m_fc.train(bMode);
    }

private:
    int m_nNumClasses;  // 20260322 ZJH 分类类别数

    ConvBnReLU6 m_stem;       // 20260322 ZJH stem: Conv3x3 stride=2, 3→16
    ConvBnReLU6 m_headConv;   // 20260322 ZJH head: Conv1x1, 160→1280

    // 20260322 ZJH 6 个阶段，每阶段包含若干个 InvertedResidual 块
    // 使用 unique_ptr 避免 vector realloc 导致指针失效
    std::vector<std::vector<std::unique_ptr<InvertedResidual>>> m_vecStages;

    AdaptiveAvgPool2d m_adaptivePool;  // 20260322 ZJH 自适应平均池化到 1x1
    Flatten m_flatten;                  // 20260322 ZJH 展平层
    Linear m_fc;                        // 20260322 ZJH 全连接分类器
};

}  // namespace om
