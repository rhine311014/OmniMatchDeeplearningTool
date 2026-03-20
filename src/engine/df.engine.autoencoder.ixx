// 20260320 ZJH 卷积自编码器（异常检测）模块 — Phase 3
// 编码器: Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> 瓶颈
// 解码器: ConvTranspose -> ConvTranspose -> ConvTranspose -> 重建
// 异常分数 = MSE(输入, 重建输出)
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>

export module df.engine.autoencoder;

// 20260320 ZJH 导入依赖模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;
import df.engine.conv;
import df.engine.activations;

export namespace df {

// 20260320 ZJH ConvAutoEncoder — 卷积自编码器，用于异常检测
// 编码器: 通过卷积+池化将输入压缩到低维瓶颈表示
// 解码器: 通过转置卷积恢复原始分辨率
// 异常分数 = MSE(input, reconstruction)，正常样本重建误差小，异常样本重建误差大
// 输入: [N, nInChannels, 28, 28]（适配 MNIST 大小）
// 输出: [N, nInChannels, 28, 28]（重建图像）
class ConvAutoEncoder : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数，默认 1（灰度图）
    // nLatentDim: 瓶颈层通道数，默认 64
    ConvAutoEncoder(int nInChannels = 1, int nLatentDim = 64)
        : m_nInChannels(nInChannels),
          // 20260320 ZJH 编码器卷积层
          m_encConv1(nInChannels, 32, 3, 1, 1, true),   // 20260320 ZJH [N,1,28,28] -> [N,32,28,28]
          m_encConv2(32, 64, 3, 1, 1, true),             // 20260320 ZJH [N,32,14,14] -> [N,64,14,14]
          m_encConv3(64, nLatentDim, 3, 1, 1, true),     // 20260320 ZJH [N,64,7,7] -> [N,latent,7,7]
          // 20260320 ZJH 编码器 BN
          m_encBn1(32),
          m_encBn2(64),
          m_encBn3(nLatentDim),
          // 20260320 ZJH 池化
          m_pool(2),                                      // 20260320 ZJH 2x2 最大池化
          // 20260320 ZJH 解码器转置卷积层
          // ConvTranspose2d 权重形状: [Cin, Cout, KH, KW]
          // [N,latent,7,7] -> [N,64,14,14]: Hout = (7-1)*2 - 2*0 + 2 = 14
          m_decConv1(nLatentDim, 64, 2, 2, 0, true),
          // [N,64,14,14] -> [N,32,28,28]: Hout = (14-1)*2 - 2*0 + 2 = 28
          m_decConv2(64, 32, 2, 2, 0, true),
          // [N,32,28,28] -> [N,nIn,28,28]: 1x1 卷积调整通道数
          m_decConv3(32, nInChannels, 3, 1, 1, true),
          // 20260320 ZJH 解码器 BN
          m_decBn1(64),
          m_decBn2(32)
    {}

    // 20260320 ZJH forward — 自编码器前向传播（编码+解码）
    // input: [N, nInChannels, 28, 28]
    // 返回: [N, nInChannels, 28, 28]（重建图像）
    Tensor forward(const Tensor& input) override {
        auto latent = encode(input);   // 20260320 ZJH 编码
        auto output = decode(latent);  // 20260320 ZJH 解码
        return output;
    }

    // 20260320 ZJH encode — 编码器前向
    // input: [N, nInChannels, 28, 28]
    // 返回: [N, nLatentDim, 7, 7]（瓶颈表示）
    Tensor encode(const Tensor& input) {
        // 20260320 ZJH 编码器第 1 层: [N,1,28,28] -> [N,32,28,28] -> pool -> [N,32,14,14]
        auto x = m_encConv1.forward(input);
        x = m_encBn1.forward(x);
        x = m_relu.forward(x);
        x = m_pool.forward(x);

        // 20260320 ZJH 编码器第 2 层: [N,32,14,14] -> [N,64,14,14] -> pool -> [N,64,7,7]
        x = m_encConv2.forward(x);
        x = m_encBn2.forward(x);
        x = m_relu.forward(x);
        x = m_pool.forward(x);

        // 20260320 ZJH 编码器第 3 层: [N,64,7,7] -> [N,latent,7,7]
        x = m_encConv3.forward(x);
        x = m_encBn3.forward(x);
        x = m_relu.forward(x);

        return x;  // 20260320 ZJH 返回瓶颈表示
    }

    // 20260320 ZJH decode — 解码器前向
    // latent: [N, nLatentDim, 7, 7]
    // 返回: [N, nInChannels, 28, 28]（重建图像）
    Tensor decode(const Tensor& latent) {
        // 20260320 ZJH 解码器第 1 层: [N,latent,7,7] -> [N,64,14,14]
        auto x = m_decConv1.forward(latent);
        x = m_decBn1.forward(x);
        x = m_relu.forward(x);

        // 20260320 ZJH 解码器第 2 层: [N,64,14,14] -> [N,32,28,28]
        x = m_decConv2.forward(x);
        x = m_decBn2.forward(x);
        x = m_relu.forward(x);

        // 20260320 ZJH 解码器第 3 层: [N,32,28,28] -> [N,nIn,28,28]
        x = m_decConv3.forward(x);
        // 20260320 ZJH 输出层使用 Sigmoid 将值映射到 [0,1]
        x = m_sigmoid.forward(x);

        return x;  // 20260320 ZJH 返回重建图像
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        // 20260320 ZJH 编码器参数
        append(m_encConv1.parameters());
        append(m_encBn1.parameters());
        append(m_encConv2.parameters());
        append(m_encBn2.parameters());
        append(m_encConv3.parameters());
        append(m_encBn3.parameters());
        // 20260320 ZJH 解码器参数
        append(m_decConv1.parameters());
        append(m_decBn1.parameters());
        append(m_decConv2.parameters());
        append(m_decBn2.parameters());
        append(m_decConv3.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_encConv1.train(bMode); m_encBn1.train(bMode);
        m_encConv2.train(bMode); m_encBn2.train(bMode);
        m_encConv3.train(bMode); m_encBn3.train(bMode);
        m_decConv1.train(bMode); m_decBn1.train(bMode);
        m_decConv2.train(bMode); m_decBn2.train(bMode);
        m_decConv3.train(bMode);
    }

private:
    int m_nInChannels;  // 20260320 ZJH 输入通道数

    // 20260320 ZJH 编码器
    Conv2d m_encConv1, m_encConv2, m_encConv3;        // 20260320 ZJH 编码器卷积
    BatchNorm2d m_encBn1, m_encBn2, m_encBn3;         // 20260320 ZJH 编码器 BN
    MaxPool2d m_pool;                                   // 20260320 ZJH 池化层
    ReLU m_relu;                                        // 20260320 ZJH ReLU 激活

    // 20260320 ZJH 解码器
    ConvTranspose2d m_decConv1, m_decConv2;            // 20260320 ZJH 解码器转置卷积（上采样）
    Conv2d m_decConv3;                                  // 20260320 ZJH 解码器最终卷积
    BatchNorm2d m_decBn1, m_decBn2;                    // 20260320 ZJH 解码器 BN
    Sigmoid m_sigmoid;                                  // 20260320 ZJH 输出激活 [0,1]
};

}  // namespace df
