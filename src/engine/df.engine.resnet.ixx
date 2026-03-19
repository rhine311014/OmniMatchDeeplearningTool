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

export module df.engine.resnet;

// 20260319 ZJH 导入依赖模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;
import df.engine.conv;
import df.engine.linear;
import df.engine.activations;

export namespace df {

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
    std::vector<Tensor*> parameters() {
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
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") {
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

    // 20260319 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) {
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
    // nNumClasses: 分类类别数，默认 10（MNIST/CIFAR-10）
    ResNet18(int nNumClasses = 10)
        : m_conv1(1, 64, 3, 1, 1, false),  // 20260319 ZJH 初始 3x3 卷积: [N,1,28,28] -> [N,64,28,28]
          m_bn1(64),                         // 20260319 ZJH 初始 BN
          m_maxpool(3, 2, 1),                // 20260319 ZJH MaxPool（小图像不使用，但保留实例）
          m_avgpool(4, 1, 0),                // 20260319 ZJH 全局平均池化: 4x4 -> 1x1
          m_fc(512, nNumClasses)             // 20260319 ZJH 全连接分类器: 512 -> nNumClasses
    {
        // 20260319 ZJH 预分配向量容量，防止 emplace_back 时重新分配导致指针失效
        // 注意：BasicBlock 内部使用 registerParameter 保存成员变量指针，
        // 如果 vector 重新分配内存，移动后的对象中指针会失效
        m_vecLayer1.reserve(2);  // 20260319 ZJH layer1 预分配 2 个元素
        m_vecLayer2.reserve(2);  // 20260319 ZJH layer2 预分配 2 个元素
        m_vecLayer3.reserve(2);  // 20260319 ZJH layer3 预分配 2 个元素
        m_vecLayer4.reserve(2);  // 20260319 ZJH layer4 预分配 2 个元素

        // 20260319 ZJH 构建 layer1: 2 个 BasicBlock(64, 64, stride=1)
        m_vecLayer1.emplace_back(64, 64, 1);   // 20260319 ZJH block1: 64->64, 保持分辨率
        m_vecLayer1.emplace_back(64, 64, 1);   // 20260319 ZJH block2: 64->64, 保持分辨率

        // 20260319 ZJH 构建 layer2: 2 个 BasicBlock(64->128, stride=2 for first block)
        m_vecLayer2.emplace_back(64, 128, 2);  // 20260319 ZJH block1: 64->128, 空间下采样 28->14
        m_vecLayer2.emplace_back(128, 128, 1); // 20260319 ZJH block2: 128->128, 保持分辨率

        // 20260319 ZJH 构建 layer3: 2 个 BasicBlock(128->256, stride=2 for first block)
        m_vecLayer3.emplace_back(128, 256, 2); // 20260319 ZJH block1: 128->256, 空间下采样 14->7
        m_vecLayer3.emplace_back(256, 256, 1); // 20260319 ZJH block2: 256->256, 保持分辨率

        // 20260319 ZJH 构建 layer4: 2 个 BasicBlock(256->512, stride=2 for first block)
        m_vecLayer4.emplace_back(256, 512, 2); // 20260319 ZJH block1: 256->512, 空间下采样 7->4
        m_vecLayer4.emplace_back(512, 512, 1); // 20260319 ZJH block2: 512->512, 保持分辨率
    }

    // 20260319 ZJH forward — ResNet18 前向传播
    // input: [N, 1, 28, 28]（灰度图）
    // 返回: [N, nNumClasses]
    Tensor forward(const Tensor& input) override {
        // 20260319 ZJH 初始卷积 + BN + ReLU: [N,1,28,28] -> [N,64,28,28]
        auto x = m_conv1.forward(input);  // 20260319 ZJH 3x3 卷积
        x = m_bn1.forward(x);             // 20260319 ZJH 批归一化
        x = m_relu.forward(x);            // 20260319 ZJH ReLU 激活
        // 20260319 ZJH 注意：小图像不使用 maxpool（跳过 m_maxpool）

        // 20260319 ZJH layer1: [N,64,28,28] -> [N,64,28,28]
        for (auto& block : m_vecLayer1) {
            x = block.forward(x);  // 20260319 ZJH 逐块前向
        }

        // 20260319 ZJH layer2: [N,64,28,28] -> [N,128,14,14]
        for (auto& block : m_vecLayer2) {
            x = block.forward(x);  // 20260319 ZJH 逐块前向
        }

        // 20260319 ZJH layer3: [N,128,14,14] -> [N,256,7,7]
        for (auto& block : m_vecLayer3) {
            x = block.forward(x);  // 20260319 ZJH 逐块前向
        }

        // 20260319 ZJH layer4: [N,256,7,7] -> [N,512,4,4]
        for (auto& block : m_vecLayer4) {
            x = block.forward(x);  // 20260319 ZJH 逐块前向
        }

        // 20260319 ZJH 全局平均池化: [N,512,4,4] -> [N,512,1,1]
        x = m_avgpool.forward(x);
        // 20260319 ZJH 展平: [N,512,1,1] -> [N,512]
        x = m_flatten.forward(x);
        // 20260319 ZJH 全连接分类器: [N,512] -> [N,nNumClasses]
        x = m_fc.forward(x);

        return x;  // 20260319 ZJH 返回分类 logits
    }

    // 20260319 ZJH 重写 parameters() 收集所有子层参数
    // 注意：Module::parameters() 不是虚函数，必须在具体类型上调用
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> vecResult;  // 20260319 ZJH 参数收集容器

        // 20260319 ZJH 辅助 lambda：合并参数向量到结果
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        // 20260319 ZJH 收集初始层参数（直接在具体类型上调用）
        appendVec(m_conv1.parameters());  // 20260319 ZJH conv1 权重
        appendVec(m_bn1.parameters());    // 20260319 ZJH bn1 gamma/beta

        // 20260319 ZJH 收集四个残差层的参数（BasicBlock::parameters()）
        for (auto& block : m_vecLayer1) appendVec(block.parameters());  // 20260319 ZJH layer1 参数
        for (auto& block : m_vecLayer2) appendVec(block.parameters());  // 20260319 ZJH layer2 参数
        for (auto& block : m_vecLayer3) appendVec(block.parameters());  // 20260319 ZJH layer3 参数
        for (auto& block : m_vecLayer4) appendVec(block.parameters());  // 20260319 ZJH layer4 参数

        // 20260319 ZJH 收集分类器参数
        appendVec(m_fc.parameters());  // 20260319 ZJH fc 权重和偏置

        return vecResult;  // 20260319 ZJH 返回所有参数
    }

    // 20260319 ZJH 重写 namedParameters() 收集所有子层命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") {
        std::vector<std::pair<std::string, Tensor*>> vecResult;

        // 20260319 ZJH 辅助 lambda：合并命名参数向量到结果
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };

        // 20260319 ZJH 辅助 lambda：生成带前缀的全名
        auto makePrefix = [&](const std::string& strSubPrefix) -> std::string {
            return strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
        };

        // 20260319 ZJH 初始层（直接在具体类型上调用）
        appendVec(m_conv1.namedParameters(makePrefix("conv1")));
        appendVec(m_bn1.namedParameters(makePrefix("bn1")));

        // 20260319 ZJH 四个残差层（BasicBlock::namedParameters()）
        for (size_t i = 0; i < m_vecLayer1.size(); ++i)
            appendVec(m_vecLayer1[i].namedParameters(makePrefix("layer1." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer2.size(); ++i)
            appendVec(m_vecLayer2[i].namedParameters(makePrefix("layer2." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer3.size(); ++i)
            appendVec(m_vecLayer3[i].namedParameters(makePrefix("layer3." + std::to_string(i))));
        for (size_t i = 0; i < m_vecLayer4.size(); ++i)
            appendVec(m_vecLayer4[i].namedParameters(makePrefix("layer4." + std::to_string(i))));

        // 20260319 ZJH 分类器
        appendVec(m_fc.namedParameters(makePrefix("fc")));

        return vecResult;
    }

    // 20260319 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) {
        m_bTraining = bMode;  // 20260319 ZJH 设置自身训练标志
        m_conv1.train(bMode);
        m_bn1.train(bMode);
        for (auto& block : m_vecLayer1) block.train(bMode);
        for (auto& block : m_vecLayer2) block.train(bMode);
        for (auto& block : m_vecLayer3) block.train(bMode);
        for (auto& block : m_vecLayer4) block.train(bMode);
        m_fc.train(bMode);
    }

private:
    Conv2d m_conv1;           // 20260319 ZJH 初始 3x3 卷积
    BatchNorm2d m_bn1;        // 20260319 ZJH 初始批归一化
    ReLU m_relu;              // 20260319 ZJH ReLU 激活（无状态，共用）
    MaxPool2d m_maxpool;      // 20260319 ZJH 最大池化（小图像不使用）

    std::vector<BasicBlock> m_vecLayer1;  // 20260319 ZJH 第一残差层: 64->64
    std::vector<BasicBlock> m_vecLayer2;  // 20260319 ZJH 第二残差层: 64->128 (stride 2)
    std::vector<BasicBlock> m_vecLayer3;  // 20260319 ZJH 第三残差层: 128->256 (stride 2)
    std::vector<BasicBlock> m_vecLayer4;  // 20260319 ZJH 第四残差层: 256->512 (stride 2)

    AvgPool2d m_avgpool;      // 20260319 ZJH 全局平均池化
    Flatten m_flatten;        // 20260319 ZJH 展平层
    Linear m_fc;              // 20260319 ZJH 全连接分类器
};

}  // namespace df
