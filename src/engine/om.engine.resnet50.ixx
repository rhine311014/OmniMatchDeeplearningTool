// 20260322 ZJH ResNet50 模块 — Bottleneck 残差块 + ResNet50 深度分类网络
// Bottleneck: 1x1 降维 → 3x3 卷积 → 1x1 升维 + 残差连接
// ResNet50: 4 个残差层 [3, 4, 6, 3] 个 Bottleneck，通道 64→256→512→1024→2048
// 适合 ImageNet 等大规模分类任务
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>

export module om.engine.resnet50;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;

export namespace om {

// 20260322 ZJH Bottleneck — ResNet-50/101/152 瓶颈残差块
// 结构: conv1(1x1, 降维) → bn1 → relu → conv2(3x3) → bn2 → relu → conv3(1x1, 升维) → bn3 → + shortcut → relu
// 扩展因子 expansion=4: 输出通道 = nPlanes * 4
// 如果输入输出维度不同（stride != 1 或通道数变化），使用 1x1 卷积做 downsample
class Bottleneck : public Module {
public:
    // 20260322 ZJH 扩展因子常量
    static constexpr int s_nExpansion = 4;

    // 20260322 ZJH 构造函数
    // nInChannels: 输入通道数
    // nPlanes: 中间瓶颈通道数（输出通道 = nPlanes * 4）
    // nStride: 第二层 3x3 卷积的步幅（用于空间下采样），默认 1
    Bottleneck(int nInChannels, int nPlanes, int nStride = 1)
        : m_nPlanes(nPlanes),
          m_conv1(nInChannels, nPlanes, 1, 1, 0, false),          // 20260322 ZJH 1x1 降维: Cin → nPlanes
          m_bn1(nPlanes),                                           // 20260322 ZJH 第一层 BN
          m_conv2(nPlanes, nPlanes, 3, nStride, 1, false),         // 20260322 ZJH 3x3 卷积: nPlanes → nPlanes
          m_bn2(nPlanes),                                           // 20260322 ZJH 第二层 BN
          m_conv3(nPlanes, nPlanes * s_nExpansion, 1, 1, 0, false), // 20260322 ZJH 1x1 升维: nPlanes → nPlanes*4
          m_bn3(nPlanes * s_nExpansion)                              // 20260322 ZJH 第三层 BN
    {
        int nOutChannels = nPlanes * s_nExpansion;  // 20260322 ZJH 输出通道数
        // 20260322 ZJH 如果步幅不为 1 或输入输出通道不匹配，需要下采样跳跃连接
        if (nStride != 1 || nInChannels != nOutChannels) {
            m_pDownsampleConv = std::make_unique<Conv2d>(nInChannels, nOutChannels, 1, nStride, 0, false);
            m_pDownsampleBn = std::make_unique<BatchNorm2d>(nOutChannels);
        }
    }

    // 20260322 ZJH forward — Bottleneck 前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, nPlanes*4, Hout, Wout]
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 主路径: 1x1降维 → BN → ReLU → 3x3 → BN → ReLU → 1x1升维 → BN
        auto out = m_conv1.forward(input);   // 20260322 ZJH 1x1 降维
        out = m_bn1.forward(out);            // 20260322 ZJH BN
        out = m_relu.forward(out);           // 20260322 ZJH ReLU

        out = m_conv2.forward(out);          // 20260322 ZJH 3x3 卷积
        out = m_bn2.forward(out);            // 20260322 ZJH BN
        out = m_relu.forward(out);           // 20260322 ZJH ReLU

        out = m_conv3.forward(out);          // 20260322 ZJH 1x1 升维
        out = m_bn3.forward(out);            // 20260322 ZJH BN

        // 20260322 ZJH 跳跃连接
        Tensor shortcut;
        if (m_pDownsampleConv) {
            shortcut = m_pDownsampleConv->forward(input);    // 20260322 ZJH 1x1 下采样
            shortcut = m_pDownsampleBn->forward(shortcut);   // 20260322 ZJH BN
        } else {
            shortcut = input;  // 20260322 ZJH 维度匹配时直接使用输入
        }

        // 20260322 ZJH 残差加法
        out = tensorAdd(out, shortcut);
        // 20260322 ZJH 最终 ReLU
        out = m_relu.forward(out);

        return out;  // 20260322 ZJH 返回 Bottleneck 输出
    }

    // 20260322 ZJH 重写 parameters() 收集所有子层参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        appendVec(m_conv1.parameters());  // 20260322 ZJH conv1 参数
        appendVec(m_bn1.parameters());    // 20260322 ZJH bn1 参数
        appendVec(m_conv2.parameters());  // 20260322 ZJH conv2 参数
        appendVec(m_bn2.parameters());    // 20260322 ZJH bn2 参数
        appendVec(m_conv3.parameters());  // 20260322 ZJH conv3 参数
        appendVec(m_bn3.parameters());    // 20260322 ZJH bn3 参数

        if (m_pDownsampleConv) {
            appendVec(m_pDownsampleConv->parameters());  // 20260322 ZJH downsample conv 参数
            appendVec(m_pDownsampleBn->parameters());    // 20260322 ZJH downsample bn 参数
        }

        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };

        appendParams("conv1", m_conv1);
        appendParams("bn1", m_bn1);
        appendParams("conv2", m_conv2);
        appendParams("bn2", m_bn2);
        appendParams("conv3", m_conv3);
        appendParams("bn3", m_bn3);

        if (m_pDownsampleConv) {
            appendParams("downsample_conv", *m_pDownsampleConv);
            appendParams("downsample_bn", *m_pDownsampleBn);
        }

        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        appendVec(m_bn1.buffers());  // 20260328 ZJH bn1 running_mean/running_var
        appendVec(m_bn2.buffers());  // 20260328 ZJH bn2 running_mean/running_var
        appendVec(m_bn3.buffers());  // 20260328 ZJH bn3 running_mean/running_var
        if (m_pDownsampleBn) {
            appendVec(m_pDownsampleBn->buffers());  // 20260328 ZJH downsample BN 缓冲区
        }
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("bn1", m_bn1);
        appendBufs("bn2", m_bn2);
        appendBufs("bn3", m_bn3);
        if (m_pDownsampleBn) {
            appendBufs("downsample_bn", *m_pDownsampleBn);
        }
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);  m_bn1.train(bMode);
        m_conv2.train(bMode);  m_bn2.train(bMode);
        m_conv3.train(bMode);  m_bn3.train(bMode);
        if (m_pDownsampleConv) {
            m_pDownsampleConv->train(bMode);
            m_pDownsampleBn->train(bMode);
        }
    }

private:
    int m_nPlanes;  // 20260322 ZJH 中间瓶颈通道数

    Conv2d m_conv1;      // 20260322 ZJH 1x1 降维卷积
    BatchNorm2d m_bn1;   // 20260322 ZJH 第一层 BN
    Conv2d m_conv2;      // 20260322 ZJH 3x3 卷积
    BatchNorm2d m_bn2;   // 20260322 ZJH 第二层 BN
    Conv2d m_conv3;      // 20260322 ZJH 1x1 升维卷积
    BatchNorm2d m_bn3;   // 20260322 ZJH 第三层 BN
    ReLU m_relu;         // 20260322 ZJH ReLU 激活（无状态，共用）

    std::unique_ptr<Conv2d> m_pDownsampleConv;     // 20260322 ZJH 下采样 1x1 卷积（可选）
    std::unique_ptr<BatchNorm2d> m_pDownsampleBn;  // 20260322 ZJH 下采样 BN（可选）
};

// 20260322 ZJH ResNet50 — 50 层残差网络
// 结构: conv1(7x7, stride=2) → bn1 → relu → maxpool(3x3, stride=2)
//       → layer1[3 Bottleneck] → layer2[4 Bottleneck] → layer3[6 Bottleneck] → layer4[3 Bottleneck]
//       → AdaptiveAvgPool2d(1,1) → Flatten → Linear(2048, nClasses)
// 通道变化: 64 → 256 → 512 → 1024 → 2048
// 针对小图像（MNIST/CIFAR）优化: conv1 使用 3x3 stride=1 pad=1，跳过 maxpool
class ResNet50 : public Module {
public:
    // 20260322 ZJH 构造函数
    // nNumClasses: 分类类别数，默认 10
    // 20260326 ZJH nInChannels 默认改为 3（RGB），修复推理时 3 通道输入与 1 通道权重不匹配导致崩溃
    ResNet50(int nNumClasses = 10, int nInChannels = 3)
        : m_conv1(nInChannels, 64, 3, 1, 1, false),  // 20260322 ZJH 小图像: 3x3 stride=1 pad=1
          m_bn1(64),                                    // 20260322 ZJH 初始 BN
          m_adaptivePool(1, 1),                         // 20260322 ZJH 自适应平均池化到 1x1
          m_fc(2048, nNumClasses)                       // 20260322 ZJH 全连接: 2048 → nClasses
    {
        // 20260322 ZJH 构建 layer1: 3 个 Bottleneck(64→256)
        // 第一块: 64→64(planes), 输出 64*4=256
        // 后续块: 256→64(planes), 输出 64*4=256
        m_vecLayer1.reserve(3);
        m_vecLayer1.push_back(std::make_unique<Bottleneck>(64, 64, 1));    // 20260322 ZJH block0: 64→256
        m_vecLayer1.push_back(std::make_unique<Bottleneck>(256, 64, 1));   // 20260322 ZJH block1: 256→256
        m_vecLayer1.push_back(std::make_unique<Bottleneck>(256, 64, 1));   // 20260322 ZJH block2: 256→256

        // 20260322 ZJH 构建 layer2: 4 个 Bottleneck(256→512), 第一块 stride=2
        m_vecLayer2.reserve(4);
        m_vecLayer2.push_back(std::make_unique<Bottleneck>(256, 128, 2));  // 20260322 ZJH block0: 256→512, 下采样
        m_vecLayer2.push_back(std::make_unique<Bottleneck>(512, 128, 1));  // 20260322 ZJH block1: 512→512
        m_vecLayer2.push_back(std::make_unique<Bottleneck>(512, 128, 1));  // 20260322 ZJH block2: 512→512
        m_vecLayer2.push_back(std::make_unique<Bottleneck>(512, 128, 1));  // 20260322 ZJH block3: 512→512

        // 20260322 ZJH 构建 layer3: 6 个 Bottleneck(512→1024), 第一块 stride=2
        m_vecLayer3.reserve(6);
        m_vecLayer3.push_back(std::make_unique<Bottleneck>(512, 256, 2));   // 20260322 ZJH block0: 512→1024, 下采样
        m_vecLayer3.push_back(std::make_unique<Bottleneck>(1024, 256, 1));  // 20260322 ZJH block1: 1024→1024
        m_vecLayer3.push_back(std::make_unique<Bottleneck>(1024, 256, 1));  // 20260322 ZJH block2: 1024→1024
        m_vecLayer3.push_back(std::make_unique<Bottleneck>(1024, 256, 1));  // 20260322 ZJH block3: 1024→1024
        m_vecLayer3.push_back(std::make_unique<Bottleneck>(1024, 256, 1));  // 20260322 ZJH block4: 1024→1024
        m_vecLayer3.push_back(std::make_unique<Bottleneck>(1024, 256, 1));  // 20260322 ZJH block5: 1024→1024

        // 20260322 ZJH 构建 layer4: 3 个 Bottleneck(1024→2048), 第一块 stride=2
        m_vecLayer4.reserve(3);
        m_vecLayer4.push_back(std::make_unique<Bottleneck>(1024, 512, 2));  // 20260322 ZJH block0: 1024→2048, 下采样
        m_vecLayer4.push_back(std::make_unique<Bottleneck>(2048, 512, 1));  // 20260322 ZJH block1: 2048→2048
        m_vecLayer4.push_back(std::make_unique<Bottleneck>(2048, 512, 1));  // 20260322 ZJH block2: 2048→2048
    }

    // 20260322 ZJH forward — ResNet50 前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, nNumClasses]
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 初始层: Conv3x3 → BN → ReLU
        auto x = m_conv1.forward(input);  // 20260322 ZJH [N,Cin,H,W] → [N,64,H,W]
        x = m_bn1.forward(x);
        x = m_relu.forward(x);
        // 20260322 ZJH 小图像不使用 maxpool

        // 20260322 ZJH layer1: [N,64,H,W] → [N,256,H,W]
        for (auto& pBlock : m_vecLayer1) {
            x = pBlock->forward(x);
        }

        // 20260322 ZJH layer2: [N,256,H,W] → [N,512,H/2,W/2]
        for (auto& pBlock : m_vecLayer2) {
            x = pBlock->forward(x);
        }

        // 20260322 ZJH layer3: [N,512,H/2,W/2] → [N,1024,H/4,W/4]
        for (auto& pBlock : m_vecLayer3) {
            x = pBlock->forward(x);
        }

        // 20260322 ZJH layer4: [N,1024,H/4,W/4] → [N,2048,H/8,W/8]
        for (auto& pBlock : m_vecLayer4) {
            x = pBlock->forward(x);
        }

        // 20260322 ZJH 自适应平均池化: [N,2048,*,*] → [N,2048,1,1]
        x = m_adaptivePool.forward(x);
        // 20260322 ZJH 展平: [N,2048,1,1] → [N,2048]
        x = m_flatten.forward(x);
        // 20260322 ZJH 全连接: [N,2048] → [N,nClasses]
        x = m_fc.forward(x);

        return x;  // 20260322 ZJH 返回分类 logits
    }

    // 20260322 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        appendVec(m_conv1.parameters());
        appendVec(m_bn1.parameters());

        for (auto& pBlock : m_vecLayer1) appendVec(pBlock->parameters());
        for (auto& pBlock : m_vecLayer2) appendVec(pBlock->parameters());
        for (auto& pBlock : m_vecLayer3) appendVec(pBlock->parameters());
        for (auto& pBlock : m_vecLayer4) appendVec(pBlock->parameters());

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

        appendVec(m_conv1.namedParameters(makeP("conv1")));
        appendVec(m_bn1.namedParameters(makeP("bn1")));

        for (size_t i = 0; i < m_vecLayer1.size(); ++i)
            appendVec(m_vecLayer1[i]->namedParameters(makeP("layer1." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer2.size(); ++i)
            appendVec(m_vecLayer2[i]->namedParameters(makeP("layer2." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer3.size(); ++i)
            appendVec(m_vecLayer3[i]->namedParameters(makeP("layer3." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer4.size(); ++i)
            appendVec(m_vecLayer4[i]->namedParameters(makeP("layer4." + std::to_string(i))));

        appendVec(m_fc.namedParameters(makeP("fc")));

        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        appendVec(m_bn1.buffers());  // 20260328 ZJH 初始 BN 缓冲区
        for (auto& pBlock : m_vecLayer1) appendVec(pBlock->buffers());  // 20260328 ZJH layer1 BN 缓冲区
        for (auto& pBlock : m_vecLayer2) appendVec(pBlock->buffers());  // 20260328 ZJH layer2 BN 缓冲区
        for (auto& pBlock : m_vecLayer3) appendVec(pBlock->buffers());  // 20260328 ZJH layer3 BN 缓冲区
        for (auto& pBlock : m_vecLayer4) appendVec(pBlock->buffers());  // 20260328 ZJH layer4 BN 缓冲区
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
        appendVec(m_bn1.namedBuffers(makeP("bn1")));  // 20260328 ZJH 初始 BN
        for (size_t i = 0; i < m_vecLayer1.size(); ++i)
            appendVec(m_vecLayer1[i]->namedBuffers(makeP("layer1." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer2.size(); ++i)
            appendVec(m_vecLayer2[i]->namedBuffers(makeP("layer2." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer3.size(); ++i)
            appendVec(m_vecLayer3[i]->namedBuffers(makeP("layer3." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer4.size(); ++i)
            appendVec(m_vecLayer4[i]->namedBuffers(makeP("layer4." + std::to_string(i))));
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);
        m_bn1.train(bMode);
        for (auto& pBlock : m_vecLayer1) pBlock->train(bMode);
        for (auto& pBlock : m_vecLayer2) pBlock->train(bMode);
        for (auto& pBlock : m_vecLayer3) pBlock->train(bMode);
        for (auto& pBlock : m_vecLayer4) pBlock->train(bMode);
        m_fc.train(bMode);
    }

private:
    Conv2d m_conv1;       // 20260322 ZJH 初始 3x3 卷积
    BatchNorm2d m_bn1;    // 20260322 ZJH 初始 BN
    ReLU m_relu;          // 20260322 ZJH ReLU 激活（无状态）

    // 20260322 ZJH 使用 unique_ptr 存储 Bottleneck，避免 vector realloc 指针失效
    std::vector<std::unique_ptr<Bottleneck>> m_vecLayer1;  // 20260322 ZJH layer1: 64→256, 3 blocks
    std::vector<std::unique_ptr<Bottleneck>> m_vecLayer2;  // 20260322 ZJH layer2: 256→512, 4 blocks
    std::vector<std::unique_ptr<Bottleneck>> m_vecLayer3;  // 20260322 ZJH layer3: 512→1024, 6 blocks
    std::vector<std::unique_ptr<Bottleneck>> m_vecLayer4;  // 20260322 ZJH layer4: 1024→2048, 3 blocks

    AdaptiveAvgPool2d m_adaptivePool;  // 20260322 ZJH 自适应平均池化
    Flatten m_flatten;                  // 20260322 ZJH 展平层
    Linear m_fc;                        // 20260322 ZJH 全连接分类器
};

}  // namespace om
