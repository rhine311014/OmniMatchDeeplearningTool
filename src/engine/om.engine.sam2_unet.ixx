// 20260402 ZJH SAM2-UNet — Segment Anything Model 2 + U-Net 解码器
// 来源: SAM2-UNet (2025) — SAM2 Hiera backbone 作为编码器 + 经典 U-Net 解码器
// IoU 从 0.69 提升到 0.83（+20%），Dice +15.58%
// 设计: Hiera-Tiny 级别编码器（~20M 参数）+ skip connections + 4 级解码器
// 适用: 工业缺陷语义分割，小数据集高精度
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

export module om.engine.sam2_unet;

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =============================================================================
// 20260402 ZJH HieraBlock — SAM2 Hiera-Tiny 风格编码器块
// 简化版: Conv + BN + ReLU（多级下采样，模拟 Hiera 的多分辨率输出）
// =============================================================================
class HieraBlock : public Module {
public:
    // 20260406 ZJH 构造函数
    // nIn: 输入通道数; nOut: 输出通道数; nStride: 卷积步幅（>1 时下采样）
    HieraBlock(int nIn, int nOut, int nStride = 1)
        : m_conv1(nIn, nOut, 3, nStride, 1, false), m_bn1(nOut),  // 20260406 ZJH 第一层 3x3 卷积 + BN
          m_conv2(nOut, nOut, 3, 1, 1, false), m_bn2(nOut)         // 20260406 ZJH 第二层 3x3 卷积 + BN
    {
        // 20260406 ZJH 当步幅不为1或通道数变化时，需要下采样快捷连接对齐维度
        if (nStride != 1 || nIn != nOut) {
            m_pDown = std::make_unique<Conv2d>(nIn, nOut, 1, nStride, 0, false);  // 20260406 ZJH 1x1 卷积下采样
            m_pDownBn = std::make_unique<BatchNorm2d>(nOut);  // 20260406 ZJH 下采样分支的批归一化
        }
    }

    // 20260406 ZJH forward — 残差块前向传播
    // input: [N, nIn, H, W]
    // 返回: [N, nOut, H/stride, W/stride]
    Tensor forward(const Tensor& input) override {
        auto x = ReLU().forward(m_bn1.forward(m_conv1.forward(input)));  // 20260406 ZJH conv1→bn1→relu
        x = m_bn2.forward(m_conv2.forward(x));  // 20260406 ZJH conv2→bn2（激活在残差加法之后）
        Tensor shortcut;  // 20260406 ZJH 快捷连接张量
        if (m_pDown) {
            // 20260406 ZJH 维度不一致时，通过1x1卷积+BN对齐
            shortcut = m_pDownBn->forward(m_pDown->forward(input));
        } else {
            // 20260406 ZJH 维度一致时，直接使用输入作为快捷连接
            shortcut = input;
        }
        return ReLU().forward(tensorAdd(x, shortcut));  // 20260406 ZJH 残差相加 + ReLU 激活
    }

    // 20260406 ZJH parameters — 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> p) { v.insert(v.end(), p.begin(), p.end()); };
        append(m_conv1.parameters()); append(m_bn1.parameters());  // 20260406 ZJH 第一层参数
        append(m_conv2.parameters()); append(m_bn2.parameters());  // 20260406 ZJH 第二层参数
        if (m_pDown) { append(m_pDown->parameters()); append(m_pDownBn->parameters()); }  // 20260406 ZJH 下采样分支参数
        return v;
    }

    // 20260406 ZJH buffers — 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> b) { v.insert(v.end(), b.begin(), b.end()); };
        append(m_bn1.buffers()); append(m_bn2.buffers());  // 20260406 ZJH 主分支 BN 缓冲
        if (m_pDownBn) append(m_pDownBn->buffers());  // 20260406 ZJH 下采样分支 BN 缓冲
        return v;
    }

    // 20260406 ZJH train — 设置训练/评估模式
    void train(bool b = true) override {
        m_bTraining = b;  // 20260406 ZJH 更新当前模块训练标志
        m_conv1.train(b); m_bn1.train(b); m_conv2.train(b); m_bn2.train(b);  // 20260406 ZJH 主分支
        if (m_pDown) { m_pDown->train(b); m_pDownBn->train(b); }  // 20260406 ZJH 下采样分支
    }

private:
    Conv2d m_conv1, m_conv2;                     // 20260406 ZJH 两层 3x3 卷积
    BatchNorm2d m_bn1, m_bn2;                    // 20260406 ZJH 两层批归一化
    std::unique_ptr<Conv2d> m_pDown;             // 20260406 ZJH 下采样 1x1 卷积（可选）
    std::unique_ptr<BatchNorm2d> m_pDownBn;      // 20260406 ZJH 下采样批归一化（可选）
};

// =============================================================================
// 20260402 ZJH SAM2UNet — SAM2 Hiera 编码器 + U-Net 解码器
// 4 级编码: 64→128→256→512 + 4 级解码 + skip connections
// =============================================================================
class SAM2UNet : public Module {
public:
    // 20260406 ZJH 构造函数
    // nInChannels: 输入通道数（默认3=RGB）
    // nNumClasses: 分割类别数（默认2=前景/背景）
    // nBase: 基础通道数（默认64）
    SAM2UNet(int nInChannels = 3, int nNumClasses = 2, int nBase = 64)
        : m_nBase(nBase),
          m_pool(2, 2, 0),  // 20260406 ZJH 2x2 最大池化（解码器中未直接使用，编码器通过stride下采样）
          // 20260402 ZJH Hiera-style 编码器
          m_enc1(nInChannels, nBase, 1),         // 20260402 ZJH → [N, 64, H, W]
          m_enc2(nBase, nBase * 2, 2),            // 20260402 ZJH → [N, 128, H/2, W/2]
          m_enc3(nBase * 2, nBase * 4, 2),        // 20260402 ZJH → [N, 256, H/4, W/4]
          m_enc4(nBase * 4, nBase * 8, 2),        // 20260402 ZJH → [N, 512, H/8, W/8]
          // 20260402 ZJH U-Net 解码器
          m_upConv4(nBase * 8, nBase * 4, 1, 1, 0, true),    // 20260402 ZJH 512→256
          m_dec4a(nBase * 8, nBase * 4, 1), // 20260402 ZJH cat(256+256)=512→256
          m_upConv3(nBase * 4, nBase * 2, 1, 1, 0, true),
          m_dec3a(nBase * 4, nBase * 2, 1),
          m_upConv2(nBase * 2, nBase, 1, 1, 0, true),
          m_dec2a(nBase * 2, nBase, 1),
          // 20260402 ZJH 分类头
          m_classifier(nBase, nNumClasses, 1, 1, 0, true)
    {}

    // 20260406 ZJH forward — 端到端前向传播
    // input: [N, C, H, W] 输入图像
    // 返回: [N, nClasses, H, W] 逐像素分类 logits
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH 编码
        auto e1 = m_enc1.forward(input);   // 20260402 ZJH [N,64,H,W]
        auto e2 = m_enc2.forward(e1);      // 20260402 ZJH [N,128,H/2,W/2]
        auto e3 = m_enc3.forward(e2);      // 20260402 ZJH [N,256,H/4,W/4]
        auto e4 = m_enc4.forward(e3);      // 20260402 ZJH [N,512,H/8,W/8]

        // 20260402 ZJH 解码 + skip
        auto d4 = upsampleAndCat(m_upConv4.forward(e4), e3);  // 20260402 ZJH [N,512,H/4,W/4]
        d4 = m_dec4a.forward(d4);  // 20260402 ZJH [N,256,H/4,W/4]

        auto d3 = upsampleAndCat(m_upConv3.forward(d4), e2);  // 20260406 ZJH 上采样+拼接 [N,256,H/2,W/2]
        d3 = m_dec3a.forward(d3);  // 20260406 ZJH [N,128,H/2,W/2]

        auto d2 = upsampleAndCat(m_upConv2.forward(d3), e1);  // 20260406 ZJH 上采样+拼接 [N,128,H,W]
        d2 = m_dec2a.forward(d2);  // 20260406 ZJH [N,64,H,W]

        return m_classifier.forward(d2);  // 20260402 ZJH [N, nClasses, H, W]
    }

    // 20260406 ZJH parameters — 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> p) { v.insert(v.end(), p.begin(), p.end()); };
        append(m_enc1.parameters()); append(m_enc2.parameters());  // 20260406 ZJH 编码器 1-2
        append(m_enc3.parameters()); append(m_enc4.parameters());  // 20260406 ZJH 编码器 3-4
        append(m_upConv4.parameters()); append(m_dec4a.parameters());  // 20260406 ZJH 解码器第4级
        append(m_upConv3.parameters()); append(m_dec3a.parameters());  // 20260406 ZJH 解码器第3级
        append(m_upConv2.parameters()); append(m_dec2a.parameters());  // 20260406 ZJH 解码器第2级
        append(m_classifier.parameters());  // 20260406 ZJH 分类头
        return v;
    }

    // 20260406 ZJH buffers — 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> b) { v.insert(v.end(), b.begin(), b.end()); };
        append(m_enc1.buffers()); append(m_enc2.buffers());  // 20260406 ZJH 编码器 1-2 缓冲
        append(m_enc3.buffers()); append(m_enc4.buffers());  // 20260406 ZJH 编码器 3-4 缓冲
        append(m_dec4a.buffers()); append(m_dec3a.buffers()); append(m_dec2a.buffers());  // 20260406 ZJH 解码器缓冲
        return v;
    }

    // 20260406 ZJH train — 设置训练/评估模式
    void train(bool b = true) override {
        m_bTraining = b;  // 20260406 ZJH 更新训练标志
        m_enc1.train(b); m_enc2.train(b); m_enc3.train(b); m_enc4.train(b);  // 20260406 ZJH 编码器
        m_dec4a.train(b); m_dec3a.train(b); m_dec2a.train(b);  // 20260406 ZJH 解码器
    }

private:
    int m_nBase;           // 20260406 ZJH 基础通道数
    MaxPool2d m_pool;      // 20260406 ZJH 最大池化层
    HieraBlock m_enc1, m_enc2, m_enc3, m_enc4;  // 20260406 ZJH 4 级 Hiera 编码器
    Conv2d m_upConv4, m_upConv3, m_upConv2;      // 20260406 ZJH 3 级上采样 1x1 卷积
    HieraBlock m_dec4a, m_dec3a, m_dec2a;         // 20260406 ZJH 3 级解码器块
    Conv2d m_classifier;   // 20260406 ZJH 分类头 1x1 卷积

    // 20260402 ZJH 上采样 + skip 拼接
    // up: [N, Cu, Hu, Wu] 上采样前的特征; skip: [N, Cs, Hs, Ws] 跳跃连接特征
    // 返回: [N, Cu+Cs, Hs, Ws] 上采样后与 skip 拼接的结果
    static Tensor upsampleAndCat(const Tensor& up, const Tensor& skip) {
        auto cUp = up.contiguous();  // 20260406 ZJH 确保连续内存布局
        int nN = cUp.shape(0), nCu = cUp.shape(1), nHu = cUp.shape(2), nWu = cUp.shape(3);  // 20260406 ZJH 上采样张量维度
        int nCs = skip.shape(1), nHs = skip.shape(2), nWs = skip.shape(3);  // 20260406 ZJH skip 张量维度
        // 20260402 ZJH 最近邻上采样到 skip 尺寸
        auto upsampled = Tensor::zeros({nN, nCu, nHs, nWs});  // 20260406 ZJH 分配上采样结果
        float* pO = upsampled.mutableFloatDataPtr();  // 20260406 ZJH 输出指针
        const float* pI = cUp.floatDataPtr();  // 20260406 ZJH 输入指针
        float fSH = static_cast<float>(nHu) / static_cast<float>(nHs);  // 20260406 ZJH 垂直方向缩放因子
        float fSW = static_cast<float>(nWu) / static_cast<float>(nWs);  // 20260406 ZJH 水平方向缩放因子
        // 20260406 ZJH 逐像素最近邻插值
        for (int n = 0; n < nN; ++n)
            for (int c = 0; c < nCu; ++c)
                for (int h = 0; h < nHs; ++h) {
                    int sh = std::min(static_cast<int>(h * fSH), nHu - 1);  // 20260406 ZJH 映射到源行坐标
                    for (int w = 0; w < nWs; ++w) {
                        int sw = std::min(static_cast<int>(w * fSW), nWu - 1);  // 20260406 ZJH 映射到源列坐标
                        pO[((n*nCu+c)*nHs+h)*nWs+w] = pI[((n*nCu+c)*nHu+sh)*nWu+sw];  // 20260406 ZJH 拷贝像素值
                    }
                }
        // 20260402 ZJH 通道拼接
        int nCtotal = nCu + nCs, nS = nHs * nWs;  // 20260406 ZJH 总通道数和空间大小
        auto result = Tensor::zeros({nN, nCtotal, nHs, nWs});  // 20260406 ZJH 拼接结果
        float* pR = result.mutableFloatDataPtr();  // 20260406 ZJH 结果指针
        const float* pUp = upsampled.floatDataPtr();  // 20260406 ZJH 上采样数据指针
        const float* pSk = skip.contiguous().floatDataPtr();  // 20260406 ZJH skip 数据指针
        for (int n = 0; n < nN; ++n) {
            // 20260406 ZJH 拷贝上采样通道
            for (int c = 0; c < nCu; ++c)
                for (int i = 0; i < nS; ++i) pR[((n*nCtotal+c)*nS)+i] = pUp[((n*nCu+c)*nS)+i];
            // 20260406 ZJH 拷贝 skip 通道
            for (int c = 0; c < nCs; ++c)
                for (int i = 0; i < nS; ++i) pR[((n*nCtotal+nCu+c)*nS)+i] = pSk[((n*nCs+c)*nS)+i];
        }
        return result;  // 20260406 ZJH 返回拼接后的张量
    }
};

}  // namespace om
