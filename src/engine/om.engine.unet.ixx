// 20260320 ZJH U-Net 语义分割网络模块 — Phase 3
// 实现编码器-解码器结构，带跳跃连接（skip connections）
// 编码器: 1->64->128->256->512
// 瓶颈层: 512->1024
// 解码器: 1024->512->256->128->64
// 最终: 64->nClasses (1x1 conv)
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>

export module om.engine.unet;

// 20260320 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;

export namespace om {

// 20260320 ZJH UNetEncoderBlock — U-Net 编码器块
// 结构: Conv2d(3x3, pad=1) -> BN -> ReLU -> Conv2d(3x3, pad=1) -> BN -> ReLU
// 保持空间维度不变（padding=1, stride=1, kernel=3）
class UNetEncoderBlock : public Module {
public:
    // 20260320 ZJH 构造函数
    // nIn: 输入通道数
    // nOut: 输出通道数
    // 20260320 ZJH fDropout: 空间 Dropout 概率（深层使用更高值）
    // 20260402 ZJH bUseGroupNorm: 是否使用 GroupNorm 替代 BatchNorm
    UNetEncoderBlock(int nIn, int nOut, float fDropout = 0.0f, bool bUseGroupNorm = false)
        : m_conv1(nIn, nOut, 3, 1, 1, true),
          m_conv2(nOut, nOut, 3, 1, 1, true),
          m_bn1(nOut), m_bn2(nOut),
          m_gn1(nOut), m_gn2(nOut),  // 20260402 ZJH GroupNorm 实例（nGroups 自动调整）
          m_dropout(fDropout),  // 20260329 ZJH 传入正确概率（旧代码用 0.0 初始化，dropout 永远不生效）
          m_fDropout(fDropout),
          m_bUseGroupNorm(bUseGroupNorm)  // 20260402 ZJH 记录归一化模式
    {}

    // 20260320 ZJH forward — 编码器块前向传播
    // input: [N, nIn, H, W]
    // 返回: [N, nOut, H, W]（空间维度不变）
    Tensor forward(const Tensor& input) override {
        auto x = m_conv1.forward(input);
        x = m_bUseGroupNorm ? m_gn1.forward(x) : m_bn1.forward(x);  // 20260402 ZJH 根据标志选择归一化层
        x = m_relu.forward(x);
        x = m_conv2.forward(x);
        x = m_bUseGroupNorm ? m_gn2.forward(x) : m_bn2.forward(x);  // 20260402 ZJH 根据标志选择归一化层
        x = m_relu.forward(x);
        // 20260320 ZJH Dropout2d 正则化（仅训练模式生效）
        if (m_bTraining && m_fDropout > 0.01f) x = m_dropout.forward(x);
        return x;
    }

    // 20260320 ZJH 重写 parameters() 手动收集子层参数
    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_conv1.parameters());
        append(m_bUseGroupNorm ? m_gn1.parameters() : m_bn1.parameters());  // 20260402 ZJH 选择归一化层参数
        append(m_conv2.parameters());
        append(m_bUseGroupNorm ? m_gn2.parameters() : m_bn2.parameters());  // 20260402 ZJH 选择归一化层参数
        return vecResult;
    }

    // 20260326 ZJH 重写 buffers() 收集 BN running stats
    // 20260402 ZJH GroupNorm 无 running stats，返回空
    std::vector<Tensor*> buffers() override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_bn1.buffers());
        append(m_bn2.buffers());
        return vecResult;
    }

    // 20260320 ZJH 重写 train() 递归设置训练模式
    // 20260402 ZJH 传播到 GN 或 BN
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);
        if (m_bUseGroupNorm) {  // 20260402 ZJH 训练模式传播到当前使用的归一化层
            m_gn1.train(bMode); m_gn2.train(bMode);
        } else {
            m_bn1.train(bMode); m_bn2.train(bMode);
        }
        m_dropout.train(bMode);
    }

private:
    Conv2d m_conv1;
    Conv2d m_conv2;
    BatchNorm2d m_bn1;      // 20260320 ZJH 第一层 BatchNorm
    BatchNorm2d m_bn2;      // 20260320 ZJH 第二层 BatchNorm
    GroupNorm2d m_gn1;      // 20260402 ZJH 第一层 GroupNorm（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gn2;      // 20260402 ZJH 第二层 GroupNorm（bUseGroupNorm=true 时使用）
    ReLU m_relu;
    Dropout2d m_dropout;  // 20260329 ZJH 构造函数初始化列表中由 fDropout 初始化
    float m_fDropout = 0.0f;
    bool m_bUseGroupNorm = false;  // 20260402 ZJH 归一化模式标志
};

// 20260320 ZJH UNetDecoderBlock — U-Net 解码器块
// 结构: Upsample(2x) -> Concat(skip) -> Conv(3x3,pad=1) -> BN -> ReLU -> Conv(3x3,pad=1) -> BN -> ReLU
// 输入通道数 = nIn（上采样后）+ nIn（skip 连接）= 2*nIn -> 输出 nOut
class UNetDecoderBlock : public Module {
public:
    // 20260320 ZJH 构造函数
    // nIn: 来自上一层的通道数（上采样后与 skip 拼接前的通道数）
    // nOut: 输出通道数
    // 20260402 ZJH bUseGroupNorm: 是否使用 GroupNorm 替代 BatchNorm
    UNetDecoderBlock(int nIn, int nOut, bool bUseGroupNorm = false)
        : m_upsample(2),                            // 20260320 ZJH 2 倍双线性上采样
          m_conv1(nIn, nOut, 3, 1, 1, true),        // 20260320 ZJH 第一层卷积：拼接后的通道数 -> nOut
          m_conv2(nOut, nOut, 3, 1, 1, true),       // 20260320 ZJH 第二层卷积
          m_bn1(nOut),                               // 20260320 ZJH 第一层 BN
          m_bn2(nOut),                               // 20260320 ZJH 第二层 BN
          m_gn1(nOut),                               // 20260402 ZJH 第一层 GN（bUseGroupNorm=true 时使用）
          m_gn2(nOut),                               // 20260402 ZJH 第二层 GN（bUseGroupNorm=true 时使用）
          m_bUseGroupNorm(bUseGroupNorm)             // 20260402 ZJH 记录归一化模式
    {}

    // 20260320 ZJH forwardWithSkip — 带 skip 连接的解码器前向
    // input: [N, Cin, H, W] 来自更深层的特征
    // skip: [N, Cskip, 2*H, 2*W] 来自编码器对应层的特征
    // 返回: [N, nOut, 2*H, 2*W]
    Tensor forwardWithSkip(const Tensor& input, const Tensor& skip) {
        auto x = m_upsample.forward(input);  // 20260320 ZJH 上采样: [N,C,H,W] -> [N,C,2H,2W]
        x = tensorConcatChannels(x, skip);   // 20260320 ZJH 拼接 skip 连接: [N,C+Cskip,2H,2W]
        x = m_conv1.forward(x);              // 20260320 ZJH 第一层卷积
        x = m_bUseGroupNorm ? m_gn1.forward(x) : m_bn1.forward(x);  // 20260402 ZJH 根据标志选择归一化层
        x = m_relu.forward(x);               // 20260320 ZJH ReLU 激活
        x = m_conv2.forward(x);              // 20260320 ZJH 第二层卷积
        x = m_bUseGroupNorm ? m_gn2.forward(x) : m_bn2.forward(x);  // 20260402 ZJH 根据标志选择归一化层
        x = m_relu.forward(x);               // 20260320 ZJH ReLU 激活
        return x;
    }

    // 20260320 ZJH forward — 标准接口（不使用 skip 连接，仅上采样+卷积）
    Tensor forward(const Tensor& input) override {
        auto x = m_upsample.forward(input);
        x = m_conv1.forward(x);
        x = m_bUseGroupNorm ? m_gn1.forward(x) : m_bn1.forward(x);  // 20260402 ZJH 根据标志选择归一化层
        x = m_relu.forward(x);
        x = m_conv2.forward(x);
        x = m_bUseGroupNorm ? m_gn2.forward(x) : m_bn2.forward(x);  // 20260402 ZJH 根据标志选择归一化层
        x = m_relu.forward(x);
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_conv1.parameters());
        append(m_bUseGroupNorm ? m_gn1.parameters() : m_bn1.parameters());  // 20260402 ZJH 选择归一化层参数
        append(m_conv2.parameters());
        append(m_bUseGroupNorm ? m_gn2.parameters() : m_bn2.parameters());  // 20260402 ZJH 选择归一化层参数
        return vecResult;
    }

    // 20260326 ZJH 重写 buffers() 收集 BN running stats
    // 20260402 ZJH GroupNorm 无 running stats，返回空
    std::vector<Tensor*> buffers() override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_bn1.buffers());
        append(m_bn2.buffers());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    // 20260402 ZJH 传播到 GN 或 BN
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);
        if (m_bUseGroupNorm) {  // 20260402 ZJH 训练模式传播到当前使用的归一化层
            m_gn1.train(bMode); m_gn2.train(bMode);
        } else {
            m_bn1.train(bMode); m_bn2.train(bMode);
        }
    }

private:
    Upsample m_upsample;   // 20260320 ZJH 2 倍双线性上采样
    Conv2d m_conv1;         // 20260320 ZJH 第一层卷积
    Conv2d m_conv2;         // 20260320 ZJH 第二层卷积
    BatchNorm2d m_bn1;      // 20260320 ZJH 第一层 BN
    BatchNorm2d m_bn2;      // 20260320 ZJH 第二层 BN
    GroupNorm2d m_gn1;      // 20260402 ZJH 第一层 GroupNorm（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gn2;      // 20260402 ZJH 第二层 GroupNorm（bUseGroupNorm=true 时使用）
    ReLU m_relu;            // 20260320 ZJH ReLU 激活
    bool m_bUseGroupNorm = false;  // 20260402 ZJH 归一化模式标志
};

// 20260320 ZJH UNet — U-Net 语义分割网络
// 编码器: nInChannels->64->128->256->512
// 瓶颈层: 512->1024
// 解码器: 1024+512->512, 512+256->256, 256+128->128, 128+64->64
// 最终: 64->nNumClasses (1x1 卷积)
// 输入 [N, nInChannels, 64, 64] -> 输出 [N, nNumClasses, 64, 64]
class UNet : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数，默认 1（灰度图）
    // nNumClasses: 输出类别数，默认 2
    // 20260329 ZJH nBaseChannels: 基础通道数，控制模型大小
    //   64（默认）= 标准 UNet（28M params），适合大数据集
    //   32 = 轻量 UNet（7M params），适合 <100 张图，速度 ~4x
    //   16 = 极轻量 UNet（1.8M params），适合 <30 张图，速度 ~16x
    // 20260402 ZJH bUseGroupNorm: 是否在所有编码器/解码器/瓶颈层使用 GroupNorm 替代 BatchNorm
    UNet(int nInChannels = 1, int nNumClasses = 2, int nBaseChannels = 64, bool bUseGroupNorm = false)
        : m_enc1(nInChannels, nBaseChannels, 0.0f, bUseGroupNorm),  // 20260320 ZJH 浅层不 Dropout
          m_enc2(nBaseChannels, nBaseChannels * 2, 0.0f, bUseGroupNorm),
          m_enc3(nBaseChannels * 2, nBaseChannels * 4, 0.1f, bUseGroupNorm),          // 20260320 ZJH 深层渐增 Dropout
          m_enc4(nBaseChannels * 4, nBaseChannels * 8, 0.2f, bUseGroupNorm),
          m_bottleneck(nBaseChannels * 8, nBaseChannels * 16, 0.3f, bUseGroupNorm),   // 20260320 ZJH 瓶颈层最高 Dropout
          m_dec4(nBaseChannels * 16 + nBaseChannels * 8, nBaseChannels * 8, bUseGroupNorm),
          m_dec3(nBaseChannels * 8 + nBaseChannels * 4, nBaseChannels * 4, bUseGroupNorm),
          m_dec2(nBaseChannels * 4 + nBaseChannels * 2, nBaseChannels * 2, bUseGroupNorm),
          m_dec1(nBaseChannels * 2 + nBaseChannels, nBaseChannels, bUseGroupNorm),
          m_finalConv(nBaseChannels, nNumClasses, 1, 1, 0, true),  // 20260320 ZJH 1x1 卷积
          m_pool(2),                   // 20260320 ZJH 2x2 最大池化，stride=2
          m_bUseGroupNorm(bUseGroupNorm)  // 20260402 ZJH 记录归一化模式
    {}

    // 20260320 ZJH forward — U-Net 前向传播
    // input: [N, nInChannels, H, W]（H, W 应为 16 的倍数）
    // 返回: [N, nNumClasses, H, W]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH 编码器路径（存储 skip 连接）
        auto skip1 = m_enc1.forward(input);          // 20260320 ZJH [N,64,H,W]
        auto p1 = m_pool.forward(skip1);             // 20260320 ZJH [N,64,H/2,W/2]

        auto skip2 = m_enc2.forward(p1);             // 20260320 ZJH [N,128,H/2,W/2]
        auto p2 = m_pool.forward(skip2);             // 20260320 ZJH [N,128,H/4,W/4]

        auto skip3 = m_enc3.forward(p2);             // 20260320 ZJH [N,256,H/4,W/4]
        auto p3 = m_pool.forward(skip3);             // 20260320 ZJH [N,256,H/8,W/8]

        auto skip4 = m_enc4.forward(p3);             // 20260320 ZJH [N,512,H/8,W/8]
        auto p4 = m_pool.forward(skip4);             // 20260320 ZJH [N,512,H/16,W/16]

        // 20260320 ZJH 瓶颈层
        auto bottleneck = m_bottleneck.forward(p4);  // 20260320 ZJH [N,1024,H/16,W/16]

        // 20260320 ZJH 解码器路径（使用 skip 连接）
        auto d4 = m_dec4.forwardWithSkip(bottleneck, skip4);  // 20260320 ZJH [N,512,H/8,W/8]
        auto d3 = m_dec3.forwardWithSkip(d4, skip3);          // 20260320 ZJH [N,256,H/4,W/4]
        auto d2 = m_dec2.forwardWithSkip(d3, skip2);          // 20260320 ZJH [N,128,H/2,W/2]
        auto d1 = m_dec1.forwardWithSkip(d2, skip1);          // 20260320 ZJH [N,64,H,W]

        // 20260320 ZJH 最终 1x1 卷积分类
        auto output = m_finalConv.forward(d1);                // 20260320 ZJH [N,nClasses,H,W]

        return output;
    }

    // 20260320 ZJH 重写 parameters() 收集所有子层参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        // 20260320 ZJH 编码器参数
        append(m_enc1.parameters());
        append(m_enc2.parameters());
        append(m_enc3.parameters());
        append(m_enc4.parameters());
        // 20260320 ZJH 瓶颈层参数
        append(m_bottleneck.parameters());
        // 20260320 ZJH 解码器参数
        append(m_dec4.parameters());
        append(m_dec3.parameters());
        append(m_dec2.parameters());
        append(m_dec1.parameters());
        // 20260320 ZJH 最终分类层参数
        append(m_finalConv.parameters());
        return vecResult;
    }

    // 20260326 ZJH 重写 buffers() 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_enc1.buffers());  append(m_enc2.buffers());
        append(m_enc3.buffers());  append(m_enc4.buffers());
        append(m_bottleneck.buffers());
        append(m_dec4.buffers()); append(m_dec3.buffers());
        append(m_dec2.buffers()); append(m_dec1.buffers());
        return vecResult;
    }

    // 20260320 ZJH 重写 train() 递归设置训练模式
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_enc1.train(bMode);
        m_enc2.train(bMode);
        m_enc3.train(bMode);
        m_enc4.train(bMode);
        m_bottleneck.train(bMode);
        m_dec4.train(bMode);
        m_dec3.train(bMode);
        m_dec2.train(bMode);
        m_dec1.train(bMode);
        m_finalConv.train(bMode);
    }

private:
    UNetEncoderBlock m_enc1, m_enc2, m_enc3, m_enc4;  // 20260320 ZJH 编码器
    UNetEncoderBlock m_bottleneck;                      // 20260320 ZJH 瓶颈层
    UNetDecoderBlock m_dec4, m_dec3, m_dec2, m_dec1;   // 20260320 ZJH 解码器
    Conv2d m_finalConv;                                 // 20260320 ZJH 1x1 分类卷积
    MaxPool2d m_pool;                                   // 20260320 ZJH 最大池化
    bool m_bUseGroupNorm = false;                       // 20260402 ZJH 归一化模式标志
};

}  // namespace om
