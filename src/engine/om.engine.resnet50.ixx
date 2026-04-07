// 20260322 ZJH ResNet50 模块 — Bottleneck 残差块 + ResNet50 深度分类网络
// Bottleneck: 1x1 降维 → 3x3 卷积 → 1x1 升维 + 残差连接
// ResNet50: 4 个残差层 [3, 4, 6, 3] 个 Bottleneck，通道 64→256→512→1024→2048
// 适合 ImageNet 等大规模分类任务
module;

#include <vector>   // 20260406 ZJH std::vector 用于参数列表和层列表
#include <string>   // 20260406 ZJH std::string 用于模块命名
#include <memory>   // 20260406 ZJH std::unique_ptr 用于子模块管理
#include <cmath>    // 20260406 ZJH 数学函数（预留）

export module om.engine.resnet50;  // 20260406 ZJH 导出 ResNet50 模块接口

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;       // 20260406 ZJH Tensor 数据结构
import om.engine.tensor_ops;   // 20260406 ZJH tensorAdd 等张量运算
import om.engine.module;       // 20260406 ZJH Module 基类、BatchNorm2d
import om.engine.conv;         // 20260406 ZJH Conv2d、AdaptiveAvgPool2d
import om.engine.linear;       // 20260406 ZJH Linear 全连接层、Flatten
import om.engine.activations;  // 20260406 ZJH ReLU 激活函数

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

        return vecResult;  // 20260406 ZJH 返回所有参数
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;  // 20260406 ZJH 命名参数收集容器
        // 20260406 ZJH lambda: 为子层参数添加层级前缀
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };

        appendParams("conv1", m_conv1);  // 20260406 ZJH 1x1 降维卷积参数
        appendParams("bn1", m_bn1);      // 20260406 ZJH 第一层 BN 参数
        appendParams("conv2", m_conv2);  // 20260406 ZJH 3x3 卷积参数
        appendParams("bn2", m_bn2);      // 20260406 ZJH 第二层 BN 参数
        appendParams("conv3", m_conv3);  // 20260406 ZJH 1x1 升维卷积参数
        appendParams("bn3", m_bn3);      // 20260406 ZJH 第三层 BN 参数

        if (m_pDownsampleConv) {
            appendParams("downsample_conv", *m_pDownsampleConv);  // 20260406 ZJH 下采样卷积参数
            appendParams("downsample_bn", *m_pDownsampleBn);      // 20260406 ZJH 下采样 BN 参数
        }

        return vecResult;  // 20260406 ZJH 返回所有命名参数
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

    // 20260322 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) override {
        m_bTraining = bMode;  // 20260406 ZJH 设置自身训练标志
        m_conv1.train(bMode);  m_bn1.train(bMode);  // 20260406 ZJH 传播到 conv1/bn1
        m_conv2.train(bMode);  m_bn2.train(bMode);  // 20260406 ZJH 传播到 conv2/bn2
        m_conv3.train(bMode);  m_bn3.train(bMode);  // 20260406 ZJH 传播到 conv3/bn3
        if (m_pDownsampleConv) {
            m_pDownsampleConv->train(bMode);  // 20260406 ZJH 传播到下采样卷积
            m_pDownsampleBn->train(bMode);    // 20260406 ZJH 传播到下采样 BN
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

    // 20260322 ZJH 重写 parameters() 收集所有子层可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;  // 20260406 ZJH 参数收集容器
        // 20260406 ZJH lambda: 将子模块参数追加到结果列表
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        appendVec(m_conv1.parameters());  // 20260406 ZJH 初始卷积参数
        appendVec(m_bn1.parameters());    // 20260406 ZJH 初始 BN 参数

        // 20260406 ZJH 遍历四个残差层收集所有 Bottleneck 参数
        for (auto& pBlock : m_vecLayer1) appendVec(pBlock->parameters());
        for (auto& pBlock : m_vecLayer2) appendVec(pBlock->parameters());
        for (auto& pBlock : m_vecLayer3) appendVec(pBlock->parameters());
        for (auto& pBlock : m_vecLayer4) appendVec(pBlock->parameters());

        appendVec(m_fc.parameters());  // 20260406 ZJH 全连接分类器参数

        return vecResult;  // 20260406 ZJH 返回所有可训练参数
    }

    // 20260322 ZJH 重写 namedParameters() 收集所有命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;  // 20260406 ZJH 命名参数收集容器
        // 20260406 ZJH lambda: 构建带前缀的参数名
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        // 20260406 ZJH lambda: 追加命名参数列表
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        appendVec(m_conv1.namedParameters(makeP("conv1")));  // 20260406 ZJH 初始卷积
        appendVec(m_bn1.namedParameters(makeP("bn1")));      // 20260406 ZJH 初始 BN

        // 20260406 ZJH 遍历四个残差层，为每个 Bottleneck 添加 layer*.i 前缀
        for (size_t i = 0; i < m_vecLayer1.size(); ++i)
            appendVec(m_vecLayer1[i]->namedParameters(makeP("layer1." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer2.size(); ++i)
            appendVec(m_vecLayer2[i]->namedParameters(makeP("layer2." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer3.size(); ++i)
            appendVec(m_vecLayer3[i]->namedParameters(makeP("layer3." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer4.size(); ++i)
            appendVec(m_vecLayer4[i]->namedParameters(makeP("layer4." + std::to_string(i))));

        appendVec(m_fc.namedParameters(makeP("fc")));  // 20260406 ZJH 全连接分类器

        return vecResult;  // 20260406 ZJH 返回所有命名参数
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

    // 20260322 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) override {
        m_bTraining = bMode;  // 20260406 ZJH 设置自身训练标志
        m_conv1.train(bMode);  // 20260406 ZJH 传播到初始卷积
        m_bn1.train(bMode);    // 20260406 ZJH 传播到初始 BN
        // 20260406 ZJH 传播到四个残差层的所有 Bottleneck
        for (auto& pBlock : m_vecLayer1) pBlock->train(bMode);
        for (auto& pBlock : m_vecLayer2) pBlock->train(bMode);
        for (auto& pBlock : m_vecLayer3) pBlock->train(bMode);
        for (auto& pBlock : m_vecLayer4) pBlock->train(bMode);
        m_fc.train(bMode);  // 20260406 ZJH 传播到全连接层
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

// =============================================================================
// 20260402 ZJH BlurPool — 抗锯齿下采样（Anti-aliased Pooling, Zhang 2019）
// 在 stride>1 下采样前先做 3x3 高斯低通滤波，消除混叠伪影
// ETFA 2025 论文证明: PatchCore + BlurPool → 生产线运动模糊鲁棒性 +2-3%
// 用法: 替换 MaxPool(stride=2) 为 MaxPool(stride=1) + BlurPool(stride=2)
// =============================================================================
class BlurPool : public Module {
public:
    // 20260402 ZJH 构造函数
    // nChannels: 通道数; nStride: 下采样步幅
    BlurPool(int nChannels, int nStride = 2)
        : m_nChannels(nChannels), m_nStride(nStride)  // 20260406 ZJH 保存通道数和下采样步幅
    {
        // 20260402 ZJH 3x3 高斯核（归一化）: [1,2,1; 2,4,2; 1,2,1] / 16
        m_kernel = Tensor::zeros({1, 1, 3, 3});  // 20260406 ZJH 创建 3x3 核张量
        float* pK = m_kernel.mutableFloatDataPtr();  // 20260406 ZJH 获取核数据指针
        // 20260406 ZJH 按行填充高斯核权重（总和为 1.0，已归一化）
        pK[0]=1/16.f; pK[1]=2/16.f; pK[2]=1/16.f;  // 20260406 ZJH 第 0 行: [1, 2, 1] / 16
        pK[3]=2/16.f; pK[4]=4/16.f; pK[5]=2/16.f;  // 20260406 ZJH 第 1 行: [2, 4, 2] / 16
        pK[6]=1/16.f; pK[7]=2/16.f; pK[8]=1/16.f;  // 20260406 ZJH 第 2 行: [1, 2, 1] / 16
    }

    // 20260402 ZJH forward — 高斯模糊 + stride 下采样
    // 20260406 ZJH input: [N, C, H, W] 输入特征图
    // 20260406 ZJH 返回: [N, C, H/stride, W/stride] 抗锯齿下采样结果
    Tensor forward(const Tensor& input) override {
        auto cIn = input.contiguous();  // 20260406 ZJH 确保内存连续
        int nN = cIn.shape(0), nC = cIn.shape(1), nH = cIn.shape(2), nW = cIn.shape(3);  // 20260406 ZJH 获取输入维度
        int nOutH = (nH + 1) / m_nStride;  // 20260402 ZJH 输出高度
        int nOutW = (nW + 1) / m_nStride;  // 20260402 ZJH 输出宽度
        auto result = Tensor::zeros({nN, nC, nOutH, nOutW});  // 20260406 ZJH 创建输出张量
        float* pO = result.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pI = cIn.floatDataPtr();  // 20260406 ZJH 输入数据指针
        const float* pK = m_kernel.floatDataPtr();  // 20260406 ZJH 高斯核数据指针

        // 20260402 ZJH 逐通道 3x3 高斯卷积 + stride 采样
        // 20260406 ZJH 四重循环: batch → 通道 → 输出高度 → 输出宽度
        for (int n = 0; n < nN; ++n) {
            for (int c = 0; c < nC; ++c) {
                for (int oh = 0; oh < nOutH; ++oh) {
                    for (int ow = 0; ow < nOutW; ++ow) {
                        int nCenterH = oh * m_nStride;  // 20260406 ZJH 输入中心行坐标
                        int nCenterW = ow * m_nStride;  // 20260406 ZJH 输入中心列坐标
                        float fSum = 0.0f;  // 20260406 ZJH 加权求和累加器
                        // 20260406 ZJH 3x3 核窗口遍历（kh, kw 从 -1 到 1）
                        for (int kh = -1; kh <= 1; ++kh) {
                            for (int kw = -1; kw <= 1; ++kw) {
                                int nh = nCenterH + kh, nw = nCenterW + kw;  // 20260406 ZJH 输入采样坐标
                                // 20260406 ZJH 边界检查：超出范围的像素用零填充（隐式 zero-padding）
                                if (nh >= 0 && nh < nH && nw >= 0 && nw < nW) {
                                    fSum += pI[((n*nC+c)*nH+nh)*nW+nw] * pK[(kh+1)*3+(kw+1)];  // 20260406 ZJH 输入×核权重
                                }
                            }
                        }
                        pO[((n*nC+c)*nOutH+oh)*nOutW+ow] = fSum;  // 20260406 ZJH 写入输出
                    }
                }
            }
        }
        return result;  // 20260406 ZJH 返回抗锯齿下采样结果
    }

    std::vector<Tensor*> parameters() override { return {}; }  // 20260402 ZJH 无可学习参数
    void train(bool bMode = true) override { m_bTraining = bMode; }  // 20260406 ZJH 设置训练模式标志

private:
    int m_nChannels;   // 20260406 ZJH 通道数
    int m_nStride;     // 20260406 ZJH 下采样步幅
    Tensor m_kernel;  // 20260402 ZJH 3x3 高斯核
};

}  // namespace om
