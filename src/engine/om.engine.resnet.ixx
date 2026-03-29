// 20260319 ZJH ResNet 模块 — Phase 2 Part 3
// 实现 BasicBlock（ResNet-18/34 基本残差块）和 ResNet18 网络
// BasicBlock: 两层 3x3 卷积 + 跳跃连接（skip connection）
// ResNet18: conv1 -> [layer1, layer2, layer3, layer4] -> avgpool -> fc
// 针对小图像（MNIST 28x28, CIFAR-10 32x32）优化：
//   conv1 使用 3x3 stride 1 pad 1（而非 ImageNet 的 7x7 stride 2）
//   跳过初始 MaxPool2d
module;

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <cmath>

export module om.engine.resnet;

// 20260319 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;

export namespace om {

// 20260319 ZJH BasicBlock — ResNet-18/34 基本残差块
// 结构: conv1(3x3) -> bn1 -> relu -> conv2(3x3) -> bn2 -> + shortcut -> relu
// 如果输入输出维度不同（stride != 1 或通道数变化），使用 1x1 卷积做 downsample
class BasicBlock : public Module {
public:
    // 20260319 ZJH 构造函数
    // nInChannels: 输入通道数
    // nOutChannels: 输出通道数
    // nStride: 第一层卷积的步幅（用于空间下采样），默认 1
    BasicBlock(int nInChannels, int nOutChannels, int nStride = 1)
        : m_conv1(nInChannels, nOutChannels, 3, nStride, 1, false),    // 20260319 ZJH 3x3 卷积，stride 可变，pad=1，无偏置
          m_bn1(nOutChannels),                                          // 20260319 ZJH 第一层 BN
          m_conv2(nOutChannels, nOutChannels, 3, 1, 1, false),         // 20260319 ZJH 3x3 卷积，stride=1，pad=1，无偏置
          m_bn2(nOutChannels)                                           // 20260319 ZJH 第二层 BN
    {
        // 20260319 ZJH 如果步幅不为 1 或通道数变化，需要下采样跳跃连接
        // 使用 unique_ptr 避免 move-assign 导致 registerParameter 指针失效
        if (nStride != 1 || nInChannels != nOutChannels) {
            m_pDownsampleConv = std::make_unique<Conv2d>(nInChannels, nOutChannels, 1, nStride, 0, false);  // 20260319 ZJH 1x1 卷积下采样
            m_pDownsampleBn = std::make_unique<BatchNorm2d>(nOutChannels);  // 20260319 ZJH 下采样后的 BN
        }
    }

    // 20260319 ZJH forward — 残差块前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, Hout, Wout]
    // 计算路径: output = relu(bn2(conv2(relu(bn1(conv1(input))))) + shortcut(input))
    Tensor forward(const Tensor& input) override {
        // 20260319 ZJH 主路径: conv1 -> bn1 -> relu -> conv2 -> bn2
        auto out = m_conv1.forward(input);   // 20260319 ZJH 第一层卷积: [N,Cin,H,W] -> [N,Cout,H',W']
        out = m_bn1.forward(out);             // 20260319 ZJH 第一层批归一化
        out = m_relu.forward(out);            // 20260319 ZJH ReLU 激活
        out = m_conv2.forward(out);           // 20260319 ZJH 第二层卷积: [N,Cout,H',W'] -> [N,Cout,H',W']
        out = m_bn2.forward(out);             // 20260319 ZJH 第二层批归一化

        // 20260319 ZJH 跳跃连接（shortcut）
        Tensor shortcut;  // 20260319 ZJH 跳跃连接张量
        if (m_pDownsampleConv) {
            // 20260319 ZJH 维度不匹配时通过 1x1 卷积 + BN 下采样
            shortcut = m_pDownsampleConv->forward(input);  // 20260319 ZJH 1x1 卷积调整通道和空间维度
            shortcut = m_pDownsampleBn->forward(shortcut);  // 20260319 ZJH BN 归一化
        } else {
            // 20260319 ZJH 维度匹配时直接使用输入作为跳跃连接
            shortcut = input;
        }

        // 20260319 ZJH 残差加法: out = out + shortcut（通过 tensorAdd 支持自动微分）
        out = tensorAdd(out, shortcut);
        // 20260319 ZJH 最终 ReLU 激活
        out = m_relu.forward(out);

        return out;  // 20260319 ZJH 返回残差块输出
    }

    // 20260319 ZJH 重写 parameters() 收集所有子层参数
    // BasicBlock 不使用 registerModule，手动收集各子层参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;  // 20260319 ZJH 参数收集容器

        // 20260319 ZJH 收集主路径参数
        auto vecConv1Params = m_conv1.parameters();  // 20260319 ZJH conv1 参数
        vecResult.insert(vecResult.end(), vecConv1Params.begin(), vecConv1Params.end());
        auto vecBn1Params = m_bn1.parameters();  // 20260319 ZJH bn1 参数
        vecResult.insert(vecResult.end(), vecBn1Params.begin(), vecBn1Params.end());
        auto vecConv2Params = m_conv2.parameters();  // 20260319 ZJH conv2 参数
        vecResult.insert(vecResult.end(), vecConv2Params.begin(), vecConv2Params.end());
        auto vecBn2Params = m_bn2.parameters();  // 20260319 ZJH bn2 参数
        vecResult.insert(vecResult.end(), vecBn2Params.begin(), vecBn2Params.end());

        // 20260319 ZJH 收集下采样路径参数（仅在维度不匹配时存在）
        if (m_pDownsampleConv) {
            auto vecDsConvParams = m_pDownsampleConv->parameters();  // 20260319 ZJH downsample conv 参数
            vecResult.insert(vecResult.end(), vecDsConvParams.begin(), vecDsConvParams.end());
            auto vecDsBnParams = m_pDownsampleBn->parameters();  // 20260319 ZJH downsample bn 参数
            vecResult.insert(vecResult.end(), vecDsBnParams.begin(), vecDsBnParams.end());
        }

        return vecResult;  // 20260319 ZJH 返回所有参数
    }

    // 20260319 ZJH 重写 namedParameters() 收集所有子层命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;

        // 20260319 ZJH 辅助 lambda：为子层参数添加前缀
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };

        appendParams("conv1", m_conv1);   // 20260319 ZJH conv1 参数
        appendParams("bn1", m_bn1);       // 20260319 ZJH bn1 参数
        appendParams("conv2", m_conv2);   // 20260319 ZJH conv2 参数
        appendParams("bn2", m_bn2);       // 20260319 ZJH bn2 参数

        if (m_pDownsampleConv) {
            appendParams("downsample_conv", *m_pDownsampleConv);  // 20260319 ZJH downsample conv 参数
            appendParams("downsample_bn", *m_pDownsampleBn);      // 20260319 ZJH downsample bn 参数
        }

        return vecResult;
    }

    // 20260327 ZJH 重写 namedBuffers() 收集所有子层命名缓冲区（BN running stats）
    // BasicBlock 不使用 registerModule，手动收集各子层缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;

        // 20260327 ZJH 辅助 lambda：为子层缓冲区添加前缀
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };

        appendBufs("bn1", m_bn1);   // 20260327 ZJH bn1 running_mean/running_var
        appendBufs("bn2", m_bn2);   // 20260327 ZJH bn2 running_mean/running_var

        if (m_pDownsampleBn) {
            appendBufs("downsample_bn", *m_pDownsampleBn);  // 20260327 ZJH downsample bn buffers
        }

        return vecResult;
    }

    // 20260319 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) override {
        m_bTraining = bMode;           // 20260319 ZJH 设置自身训练标志
        m_conv1.train(bMode);           // 20260319 ZJH 传播到 conv1
        m_bn1.train(bMode);             // 20260319 ZJH 传播到 bn1
        m_conv2.train(bMode);           // 20260319 ZJH 传播到 conv2
        m_bn2.train(bMode);             // 20260319 ZJH 传播到 bn2
        if (m_pDownsampleConv) {
            m_pDownsampleConv->train(bMode);  // 20260319 ZJH 传播到 downsample conv
            m_pDownsampleBn->train(bMode);    // 20260319 ZJH 传播到 downsample bn
        }
    }

private:
    Conv2d m_conv1;        // 20260319 ZJH 第一层 3x3 卷积
    Conv2d m_conv2;        // 20260319 ZJH 第二层 3x3 卷积
    BatchNorm2d m_bn1;     // 20260319 ZJH 第一层批归一化
    BatchNorm2d m_bn2;     // 20260319 ZJH 第二层批归一化
    ReLU m_relu;           // 20260319 ZJH ReLU 激活函数（共用实例，无状态）

    // 20260319 ZJH 下采样路径使用 unique_ptr 避免 move 导致 registerParameter 指针失效
    std::unique_ptr<Conv2d> m_pDownsampleConv;        // 20260319 ZJH 下采样 1x1 卷积（可选）
    std::unique_ptr<BatchNorm2d> m_pDownsampleBn;     // 20260319 ZJH 下采样批归一化（可选）
};

// 20260319 ZJH ResNet18 — 18 层残差网络
// 针对小图像（MNIST 28x28）优化的架构：
//   conv1: 3x3 stride 1 pad 1（保持空间分辨率）
//   不使用 MaxPool2d（小图像不需要初始下采样）
//   4 个残差层：64->64, 64->128(stride 2), 128->256(stride 2), 256->512(stride 2)
//   全局平均池化 -> Flatten -> 全连接分类器
// 28x28 输入的空间尺寸变化：28 -> 28 -> 14 -> 7 -> 4
class ResNet18 : public Module {
public:
    // 20260319 ZJH 构造函数
    // 20260326 ZJH 新增 nInChannels 参数，默认 3（RGB），兼容灰度(1)和彩色(3)输入
    // nNumClasses: 分类类别数，默认 10
    // nInChannels: 输入图像通道数，默认 3（RGB）
    ResNet18(int nNumClasses = 10, int nInChannels = 3)
    {
        m_conv1 = std::make_shared<Conv2d>(nInChannels, 64, 3, 1, 1, false);
        m_bn1 = std::make_shared<BatchNorm2d>(64);
        m_maxpool = std::make_shared<MaxPool2d>(3, 2, 1);
        m_avgpool = std::make_shared<AdaptiveAvgPool2d>(1, 1);
        m_fc = std::make_shared<Linear>(512, nNumClasses);
        m_relu = std::make_shared<ReLU>();
        m_flatten = std::make_shared<Flatten>();

        registerModule("conv1", m_conv1);
        registerModule("bn1", m_bn1);
        registerModule("maxpool", m_maxpool);
        registerModule("avgpool", m_avgpool);
        registerModule("fc", m_fc);
        registerModule("relu", m_relu);
        registerModule("flatten", m_flatten);

        // 20260319 ZJH 构建四个残差层
        // 每个 layer 使用 registerModule 注册，名称为 layer1.0, layer1.1 等
        auto addLayer = [&](const std::string& strName, std::vector<std::shared_ptr<BasicBlock>>& vec, int inC, int outC, int stride) {
            vec.reserve(2);
            for (int i = 0; i < 2; ++i) {
                int s = (i == 0) ? stride : 1;
                int c = (i == 0) ? inC : outC;
                auto block = std::make_shared<BasicBlock>(c, outC, s);
                vec.push_back(block);
                registerModule(strName + "." + std::to_string(i), block);
            }
        };

        addLayer("layer1", m_vecLayer1, 64, 64, 1);
        addLayer("layer2", m_vecLayer2, 64, 128, 2);
        addLayer("layer3", m_vecLayer3, 128, 256, 2);
        addLayer("layer4", m_vecLayer4, 256, 512, 2);
    }

    // 20260319 ZJH forward — ResNet18 前向传播
    Tensor forward(const Tensor& input) override {
        auto x = m_conv1->forward(input);
        x = m_bn1->forward(x);
        x = m_relu->forward(x);
        if (input.shape(2) > 32) {
            x = m_maxpool->forward(x);
        }

        for (auto& block : m_vecLayer1) x = block->forward(x);
        for (auto& block : m_vecLayer2) x = block->forward(x);
        for (auto& block : m_vecLayer3) x = block->forward(x);
        for (auto& block : m_vecLayer4) x = block->forward(x);

        x = m_avgpool->forward(x);
        x = m_flatten->forward(x);
        x = m_fc->forward(x);
        return x;
    }

    // 20260327 ZJH 移除手动重写的 parameters/namedParameters/namedBuffers/train，
    // 直接继承 Module 的递归版本，利用已注册的 shared_ptr 自动处理。

private:
    std::shared_ptr<Conv2d> m_conv1;
    std::shared_ptr<BatchNorm2d> m_bn1;
    std::shared_ptr<ReLU> m_relu;
    std::shared_ptr<MaxPool2d> m_maxpool;

    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer1;
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer2;
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer3;
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer4;

    std::shared_ptr<AdaptiveAvgPool2d> m_avgpool;
    std::shared_ptr<Flatten> m_flatten;
    std::shared_ptr<Linear> m_fc;
};

}  // namespace om
