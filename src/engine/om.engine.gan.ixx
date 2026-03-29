// 20260320 ZJH GAN 异常检测模块 — Phase 5
// 实现 DCGAN 风格的生成器和判别器，用于异常检测
// 训练策略：仅使用正常样本训练 GAN，异常检测时比较重建误差
module;

#include <vector>
#include <cmath>

export module om.engine.gan;

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.engine.linear;

export namespace om {

// 20260320 ZJH Generator — DCGAN 风格生成器
// 从潜在空间 [N, latentDim] 生成图像 [N, C, H, W]
// 结构：FC -> reshape -> ConvTranspose2d x3 -> Sigmoid
class Generator : public Module {
public:
    // 20260320 ZJH 构造函数
    // nLatentDim: 潜在空间维度，默认 64
    // nOutChannels: 输出通道数，默认 1（灰度）
    // nImgSize: 输出图像尺寸，默认 28
    Generator(int nLatentDim = 64, int nOutChannels = 1, int nImgSize = 28)
        : m_nLatentDim(nLatentDim), m_nOutChannels(nOutChannels), m_nImgSize(nImgSize),
          m_fc(nLatentDim, 128 * 7 * 7, true),
          m_deconv1(128, 64, 4, 2, 1, true),    // 7x7 -> 14x14
          m_deconv2(64, nOutChannels, 4, 2, 1, true)  // 14x14 -> 28x28
    {}

    // 20260320 ZJH forward — 从噪声生成图像
    // input: [N, latentDim]
    // 返回: [N, C, H, W]
    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);
        // 20260320 ZJH FC -> reshape
        auto h = m_fc.forward(input);  // [N, 128*7*7]
        h = m_leakyRelu.forward(h);
        // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝（替代 fromData 解引用指针）
        auto h4d = tensorReshape(h.contiguous(), {nBatch, 128, 7, 7});
        // 20260320 ZJH ConvTranspose2d 上采样
        h4d = m_deconv1.forward(h4d);  // [N, 64, 14, 14]
        h4d = m_leakyRelu.forward(h4d);
        h4d = m_deconv2.forward(h4d);  // [N, C, 28, 28]
        // 20260320 ZJH Sigmoid 输出到 [0,1]
        h4d = m_sigmoid.forward(h4d);
        return h4d;
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_fc.parameters()) vec.push_back(p);
        for (auto* p : m_deconv1.parameters()) vec.push_back(p);
        for (auto* p : m_deconv2.parameters()) vec.push_back(p);
        return vec;
    }

private:
    int m_nLatentDim;
    int m_nOutChannels;
    int m_nImgSize;
    Linear m_fc;
    ConvTranspose2d m_deconv1;
    ConvTranspose2d m_deconv2;
    LeakyReLU m_leakyRelu{0.2f};
    Sigmoid m_sigmoid;
};

// 20260320 ZJH Discriminator — DCGAN 风格判别器
// 输入图像 [N, C, H, W]，输出真/假概率 [N, 1]
// 结构：Conv2d x2 -> Flatten -> FC -> Sigmoid
class Discriminator : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数，默认 1
    // nImgSize: 输入图像尺寸，默认 28
    Discriminator(int nInChannels = 1, int nImgSize = 28)
        : m_nInChannels(nInChannels), m_nImgSize(nImgSize),
          m_conv1(nInChannels, 32, 4, 2, 1, true),  // 28x28 -> 14x14
          m_conv2(32, 64, 4, 2, 1, true),            // 14x14 -> 7x7
          m_fc(64 * 7 * 7, 1, true)
    {}

    // 20260320 ZJH forward — 判别图像真/假
    // input: [N, C, H, W]
    // 返回: [N, 1] 概率值
    Tensor forward(const Tensor& input) override {
        auto h = m_conv1.forward(input);    // [N, 32, 14, 14]
        h = m_leakyRelu.forward(h);
        h = m_conv2.forward(h);             // [N, 64, 7, 7]
        h = m_leakyRelu.forward(h);
        h = tensorFlatten(h, 1);            // [N, 64*7*7]
        h = m_fc.forward(h);               // [N, 1]
        h = m_sigmoid.forward(h);          // [N, 1] 概率
        return h;
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_conv1.parameters()) vec.push_back(p);
        for (auto* p : m_conv2.parameters()) vec.push_back(p);
        for (auto* p : m_fc.parameters()) vec.push_back(p);
        return vec;
    }

private:
    int m_nInChannels;
    int m_nImgSize;
    Conv2d m_conv1;
    Conv2d m_conv2;
    Linear m_fc;
    LeakyReLU m_leakyRelu{0.2f};
    Sigmoid m_sigmoid;
};

// 20260320 ZJH AnomalyGAN — 基于 GAN 的异常检测模型
// 训练：仅使用正常样本训练 Generator + Discriminator
// 推理：对测试图像计算重建误差和判别分数，超过阈值判定为异常
class AnomalyGAN : public Module {
public:
    // 20260320 ZJH 构造函数
    // nLatentDim: 潜在空间维度
    // nChannels: 图像通道数
    // nImgSize: 图像尺寸
    AnomalyGAN(int nLatentDim = 64, int nChannels = 1, int nImgSize = 28)
        : m_nLatentDim(nLatentDim), m_nChannels(nChannels),
          m_generator(nLatentDim, nChannels, nImgSize),
          m_discriminator(nChannels, nImgSize)
    {}

    // 20260320 ZJH forward — 生成器前向（接受噪声输入）
    Tensor forward(const Tensor& noise) override {
        return m_generator.forward(noise);
    }

    // 20260320 ZJH discriminate — 判别器前向
    Tensor discriminate(const Tensor& image) {
        return m_discriminator.forward(image);
    }

    // 20260320 ZJH 获取生成器/判别器参数
    std::vector<Tensor*> generatorParameters() { return m_generator.parameters(); }
    std::vector<Tensor*> discriminatorParameters() { return m_discriminator.parameters(); }

    std::vector<Tensor*> parameters() override {
        auto vec = m_generator.parameters();
        auto dVec = m_discriminator.parameters();
        vec.insert(vec.end(), dVec.begin(), dVec.end());
        return vec;
    }

    // 20260320 ZJH 获取潜在空间维度
    int latentDim() const { return m_nLatentDim; }

    // 20260320 ZJH 计算异常分数：判别器输出越低，异常可能性越大
    // 返回 1 - D(x)，正常样本接近 0，异常样本接近 1
    // 20260325 ZJH GPU 安全修复：使用 item() 读取标量值，内部已有 GPU D2H 保护
    float anomalyScore(const Tensor& image) {
        auto dScore = m_discriminator.forward(image);
        return 1.0f - dScore.item();  // 20260325 ZJH item() 自动处理 GPU→CPU D2H 传输
    }

private:
    int m_nLatentDim;
    int m_nChannels;
    Generator m_generator;
    Discriminator m_discriminator;
};

}  // namespace om
