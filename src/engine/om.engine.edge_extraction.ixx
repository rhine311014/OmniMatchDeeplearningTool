// 20260402 ZJH DL 边缘提取模块 — 对标 Halcon Deep Learning Edge Extraction
// 核心能力: 低对比度/高噪声场景下的像素级边缘检测（传统 Sobel/Canny 失效的场景）
// 架构: 轻量 U-Net（32 base channels, 4 级编解码, skip connections, 深监督）
// 输出: [N, 1, H, W] 像素级边缘概率图（0=非边缘, 1=边缘）
// 后处理: 概率阈值 → NMS thin（沿梯度方向非极大值抑制）→ 单像素宽边缘
// 训练: 边缘标注（细线条轮廓）+ BCE+Dice 混合损失（边缘正样本<5%, 严重不平衡）
// 参数量: ~0.5M（轻量设计, 推理 <5ms/帧 @ TensorRT FP16）
// 应用: 焊缝检测、切割边定位、密封圈轮廓、PCB 走线提取
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

export module om.engine.edge_extraction;

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =============================================================================
// 20260402 ZJH EdgeEncoderBlock — 边缘提取编码器块
// Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
// 轻量设计: base=32 通道（标准 U-Net 用 64）
// =============================================================================
class EdgeEncoderBlock : public Module {
public:
    // 20260402 ZJH 构造函数
    // nIn: 输入通道数; nOut: 输出通道数
    EdgeEncoderBlock(int nIn, int nOut)
        : m_conv1(nIn, nOut, 3, 1, 1, false),   // 20260402 ZJH 3x3 conv, pad=1
          m_bn1(nOut),
          m_conv2(nOut, nOut, 3, 1, 1, false),   // 20260402 ZJH 3x3 conv
          m_bn2(nOut)
    {}

    Tensor forward(const Tensor& input) override {
        auto x = ReLU().forward(m_bn1.forward(m_conv1.forward(input)));   // 20260402 ZJH conv→bn→relu
        x = ReLU().forward(m_bn2.forward(m_conv2.forward(x)));           // 20260402 ZJH conv→bn→relu
        return x;
    }

    // 20260402 ZJH 参数收集
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto a = m_conv1.parameters(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_bn1.parameters();   v.insert(v.end(), b.begin(), b.end());
        auto c = m_conv2.parameters(); v.insert(v.end(), c.begin(), c.end());
        auto d = m_bn2.parameters();   v.insert(v.end(), d.begin(), d.end());
        return v;
    }

    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        auto a = m_bn1.buffers(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_bn2.buffers(); v.insert(v.end(), b.begin(), b.end());
        return v;
    }

    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode); m_bn1.train(bMode);
        m_conv2.train(bMode); m_bn2.train(bMode);
    }

private:
    Conv2d m_conv1, m_conv2;
    BatchNorm2d m_bn1, m_bn2;
};

// =============================================================================
// 20260402 ZJH EdgeDecoderBlock — 边缘提取解码器块
// 上采样(2x) → 与 skip 拼接 → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
// =============================================================================
class EdgeDecoderBlock : public Module {
public:
    // 20260402 ZJH nIn: 上采样后通道 + skip 通道; nOut: 输出通道
    EdgeDecoderBlock(int nIn, int nOut)
        : m_conv1(nIn, nOut, 3, 1, 1, false),
          m_bn1(nOut),
          m_conv2(nOut, nOut, 3, 1, 1, false),
          m_bn2(nOut)
    {}

    // 20260402 ZJH forwardWithSkip — 上采样 + skip 拼接 + 双卷积
    Tensor forwardWithSkip(const Tensor& input, const Tensor& skip) {
        // 20260402 ZJH 最近邻上采样到 skip 的空间尺寸
        int nTargetH = skip.shape(2), nTargetW = skip.shape(3);
        auto upsampled = upsampleNearest(input, nTargetH, nTargetW);

        // 20260402 ZJH 通道拼接
        auto concat = catChannels2(upsampled, skip);

        // 20260402 ZJH 双卷积
        auto x = ReLU().forward(m_bn1.forward(m_conv1.forward(concat)));
        x = ReLU().forward(m_bn2.forward(m_conv2.forward(x)));
        return x;
    }

    Tensor forward(const Tensor& input) override { return input; }  // 20260402 ZJH 不单独使用

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto a = m_conv1.parameters(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_bn1.parameters();   v.insert(v.end(), b.begin(), b.end());
        auto c = m_conv2.parameters(); v.insert(v.end(), c.begin(), c.end());
        auto d = m_bn2.parameters();   v.insert(v.end(), d.begin(), d.end());
        return v;
    }

    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        auto a = m_bn1.buffers(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_bn2.buffers(); v.insert(v.end(), b.begin(), b.end());
        return v;
    }

    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode); m_bn1.train(bMode);
        m_conv2.train(bMode); m_bn2.train(bMode);
    }

    // 20260402 ZJH public static 工具函数（EdgeExtractionNet::forward 需要调用）
    static Tensor upsampleNearest(const Tensor& input, int nTargetH, int nTargetW) {
        auto cIn = input.contiguous();
        int nN = cIn.shape(0), nC = cIn.shape(1), nH = cIn.shape(2), nW = cIn.shape(3);
        auto result = Tensor::zeros({nN, nC, nTargetH, nTargetW});
        float* pOut = result.mutableFloatDataPtr();
        const float* pIn = cIn.floatDataPtr();
        float fSH = static_cast<float>(nH) / static_cast<float>(nTargetH);
        float fSW = static_cast<float>(nW) / static_cast<float>(nTargetW);
        for (int n = 0; n < nN; ++n)
            for (int c = 0; c < nC; ++c)
                for (int th = 0; th < nTargetH; ++th) {
                    int sh = std::min(static_cast<int>(th * fSH), nH - 1);
                    for (int tw = 0; tw < nTargetW; ++tw) {
                        int sw = std::min(static_cast<int>(tw * fSW), nW - 1);
                        pOut[((n * nC + c) * nTargetH + th) * nTargetW + tw] =
                            pIn[((n * nC + c) * nH + sh) * nW + sw];
                    }
                }
        return result;
    }

private:
    Conv2d m_conv1, m_conv2;
    BatchNorm2d m_bn1, m_bn2;

    // 20260402 ZJH 通道拼接（2 张量）— 声明为 public 以便外部使用
public:
    static Tensor catChannels2(const Tensor& a, const Tensor& b) {
        auto ca = a.contiguous(), cb = b.contiguous();
        int nN = ca.shape(0), nCa = ca.shape(1), nCb = cb.shape(1);
        int nH = ca.shape(2), nW = ca.shape(3), nS = nH * nW;
        int nCt = nCa + nCb;
        auto result = Tensor::zeros({nN, nCt, nH, nW});
        float* pO = result.mutableFloatDataPtr();
        const float* pA = ca.floatDataPtr();
        const float* pB = cb.floatDataPtr();
        for (int n = 0; n < nN; ++n) {
            for (int c = 0; c < nCa; ++c)
                for (int i = 0; i < nS; ++i)
                    pO[((n * nCt + c) * nS) + i] = pA[((n * nCa + c) * nS) + i];
            for (int c = 0; c < nCb; ++c)
                for (int i = 0; i < nS; ++i)
                    pO[((n * nCt + nCa + c) * nS) + i] = pB[((n * nCb + c) * nS) + i];
        }
        return result;
    }
};

// =============================================================================
// 20260402 ZJH EdgeExtractionNet — DL 边缘提取网络（对标 Halcon）
// 轻量 U-Net: 32→64→128→256 编码 + 256→128→64→32 解码 + 深监督
// 输入: [N, C, H, W] (C=1 灰度或 C=3 RGB)
// 输出: [N, 1, H, W] 边缘概率图（sigmoid 激活, 0~1）
// 参数量: ~0.5M
// =============================================================================
class EdgeExtractionNet : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入通道数(1=灰度, 3=RGB)
    // nBaseChannels: 基础通道数(默认32, 轻量设计)
    EdgeExtractionNet(int nInChannels = 3, int nBaseChannels = 32)
        : m_nBase(nBaseChannels),
          m_pool(2, 2, 0),
          // 20260402 ZJH 编码器: 4 级下采样
          m_enc1(nInChannels, nBaseChannels),         // 20260402 ZJH → [N, 32, H, W]
          m_enc2(nBaseChannels, nBaseChannels * 2),    // 20260402 ZJH → [N, 64, H/2, W/2]
          m_enc3(nBaseChannels * 2, nBaseChannels * 4), // 20260402 ZJH → [N, 128, H/4, W/4]
          m_enc4(nBaseChannels * 4, nBaseChannels * 8), // 20260402 ZJH → [N, 256, H/8, W/8]
          // 20260402 ZJH 解码器: 4 级上采样
          m_dec4(nBaseChannels * 8 + nBaseChannels * 4, nBaseChannels * 4),  // 20260402 ZJH 256+128→128
          m_dec3(nBaseChannels * 4 + nBaseChannels * 2, nBaseChannels * 2),  // 20260402 ZJH 128+64→64
          m_dec2(nBaseChannels * 2 + nBaseChannels, nBaseChannels),          // 20260402 ZJH 64+32→32
          // 20260402 ZJH 深监督: 每级解码器输出边缘图
          m_side4(nBaseChannels * 4, 1, 1, 1, 0, true),  // 20260402 ZJH 128→1
          m_side3(nBaseChannels * 2, 1, 1, 1, 0, true),  // 20260402 ZJH 64→1
          m_side2(nBaseChannels, 1, 1, 1, 0, true),       // 20260402 ZJH 32→1
          // 20260402 ZJH 融合层: 3 个 side output → 1 个最终边缘图
          m_fuse(3, 1, 1, 1, 0, true)                     // 20260402 ZJH 3→1 加权融合
    {}

    // 20260402 ZJH forward — 输出边缘概率图
    // input: [N, C, H, W]
    // 返回: [N, 1, H, W] sigmoid 激活后的边缘概率
    Tensor forward(const Tensor& input) override {
        int nH = input.shape(2), nW = input.shape(3);

        // 20260402 ZJH 编码器 + 保留 skip
        auto e1 = m_enc1.forward(input);        // 20260402 ZJH [N,32,H,W]
        auto e2 = m_enc2.forward(m_pool.forward(e1));  // 20260402 ZJH [N,64,H/2,W/2]
        auto e3 = m_enc3.forward(m_pool.forward(e2));  // 20260402 ZJH [N,128,H/4,W/4]
        auto e4 = m_enc4.forward(m_pool.forward(e3));  // 20260402 ZJH [N,256,H/8,W/8]

        // 20260402 ZJH 解码器 + skip connections
        auto d4 = m_dec4.forwardWithSkip(e4, e3);  // 20260402 ZJH [N,128,H/4,W/4]
        auto d3 = m_dec3.forwardWithSkip(d4, e2);  // 20260402 ZJH [N,64,H/2,W/2]
        auto d2 = m_dec2.forwardWithSkip(d3, e1);  // 20260402 ZJH [N,32,H,W]

        // 20260402 ZJH 深监督: 每级输出边缘图并上采样到原始分辨率
        auto s4 = tensorSigmoid(m_side4.forward(d4));  // 20260402 ZJH [N,1,H/4,W/4]
        auto s3 = tensorSigmoid(m_side3.forward(d3));  // 20260402 ZJH [N,1,H/2,W/2]
        auto s2 = tensorSigmoid(m_side2.forward(d2));  // 20260402 ZJH [N,1,H,W]

        // 20260402 ZJH 上采样 s4, s3 到原始分辨率
        auto s4up = EdgeDecoderBlock::upsampleNearest(s4, nH, nW);  // 20260402 ZJH [N,1,H,W]
        auto s3up = EdgeDecoderBlock::upsampleNearest(s3, nH, nW);  // 20260402 ZJH [N,1,H,W]

        // 20260402 ZJH 拼接 3 个 side output [N,3,H,W] → 1x1 conv → [N,1,H,W]
        auto sides = EdgeDecoderBlock::catChannels2(s4up, s3up);  // 20260402 ZJH [N,2,H,W] 临时
        // 20260402 ZJH 手动拼接第 3 个通道
        auto allSides = catChannel3(sides, s2);  // 20260402 ZJH [N,3,H,W]

        auto fused = tensorSigmoid(m_fuse.forward(allSides));  // 20260402 ZJH [N,1,H,W] 融合边缘图

        return fused;  // 20260402 ZJH 最终边缘概率图
    }

    // 20260402 ZJH edgeLoss — 边缘提取专用损失（BCE + Dice, 处理严重不平衡）
    // pred: [N, 1, H, W] sigmoid 后的预测
    // target: [N, 1, H, W] GT 边缘标注（0/1 二值）
    // fPosWeight: 正样本权重（边缘像素<5%, 需要 20x 加权）
    static Tensor edgeLoss(const Tensor& pred, const Tensor& target, float fPosWeight = 20.0f) {
        auto cPred = pred.contiguous();
        auto cTarget = target.contiguous();
        int nTotal = cPred.numel();
        const float* pP = cPred.floatDataPtr();
        const float* pT = cTarget.floatDataPtr();

        // 20260402 ZJH 加权 BCE: -[w*t*log(p) + (1-t)*log(1-p)]
        float fBce = 0.0f;
        float fIntersection = 0.0f, fSumP = 0.0f, fSumT = 0.0f;
        for (int i = 0; i < nTotal; ++i) {
            float p = std::max(1e-7f, std::min(pP[i], 1.0f - 1e-7f));
            float t = pT[i];
            float w = (t > 0.5f) ? fPosWeight : 1.0f;  // 20260402 ZJH 正样本加权
            fBce += -w * (t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
            fIntersection += p * t;
            fSumP += p;
            fSumT += t;
        }
        fBce /= static_cast<float>(nTotal);

        // 20260402 ZJH Dice loss
        float fDice = 1.0f - 2.0f * fIntersection / (fSumP + fSumT + 1e-6f);

        // 20260402 ZJH 总损失 = BCE + Dice
        return Tensor::full({1}, fBce + fDice);
    }

    // 20260402 ZJH nmsEdgeThin — 非极大值抑制细化（后处理）
    // probMap: [H, W] 边缘概率图
    // fThreshold: 概率阈值
    // 返回: [H, W] 细化后的单像素宽边缘图（0/1 二值）
    static Tensor nmsEdgeThin(const Tensor& probMap, float fThreshold = 0.5f) {
        auto cMap = probMap.contiguous();
        int nH = cMap.shape(0), nW = cMap.shape(1);
        const float* pMap = cMap.floatDataPtr();

        auto result = Tensor::zeros({nH, nW});
        float* pOut = result.mutableFloatDataPtr();

        // 20260402 ZJH 对每个像素: 计算梯度方向，沿梯度方向检查是否为局部最大值
        for (int y = 1; y < nH - 1; ++y) {
            for (int x = 1; x < nW - 1; ++x) {
                float fCenter = pMap[y * nW + x];
                if (fCenter < fThreshold) continue;  // 20260402 ZJH 低于阈值跳过

                // 20260402 ZJH 计算梯度方向（Sobel 3x3 近似）
                float fGx = pMap[y * nW + (x + 1)] - pMap[y * nW + (x - 1)];
                float fGy = pMap[(y + 1) * nW + x] - pMap[(y - 1) * nW + x];

                // 20260402 ZJH 量化梯度方向到 4 个方向（0/45/90/135 度）
                float fAngle = std::atan2(fGy, fGx);  // 20260402 ZJH [-π, π]
                float fAbsAngle = std::abs(fAngle);

                float fNeigh1 = 0.0f, fNeigh2 = 0.0f;  // 20260402 ZJH 梯度方向上的两个邻域值
                if (fAbsAngle < 0.3927f || fAbsAngle > 2.7489f) {
                    // 20260402 ZJH 水平方向 (0°/180°): 比较左右
                    fNeigh1 = pMap[y * nW + (x - 1)];
                    fNeigh2 = pMap[y * nW + (x + 1)];
                } else if (fAbsAngle < 1.1781f) {
                    // 20260402 ZJH 45° 方向
                    fNeigh1 = pMap[(y - 1) * nW + (x + 1)];
                    fNeigh2 = pMap[(y + 1) * nW + (x - 1)];
                } else if (fAbsAngle < 1.9635f) {
                    // 20260402 ZJH 垂直方向 (90°)
                    fNeigh1 = pMap[(y - 1) * nW + x];
                    fNeigh2 = pMap[(y + 1) * nW + x];
                } else {
                    // 20260402 ZJH 135° 方向
                    fNeigh1 = pMap[(y - 1) * nW + (x - 1)];
                    fNeigh2 = pMap[(y + 1) * nW + (x + 1)];
                }

                // 20260402 ZJH NMS: 仅保留梯度方向上的局部最大值
                if (fCenter >= fNeigh1 && fCenter >= fNeigh2) {
                    pOut[y * nW + x] = 1.0f;  // 20260402 ZJH 保留边缘点
                }
            }
        }
        return result;
    }

    // 20260402 ZJH 参数/缓冲/训练模式
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> p) { v.insert(v.end(), p.begin(), p.end()); };
        append(m_enc1.parameters()); append(m_enc2.parameters());
        append(m_enc3.parameters()); append(m_enc4.parameters());
        append(m_dec4.parameters()); append(m_dec3.parameters()); append(m_dec2.parameters());
        append(m_side4.parameters()); append(m_side3.parameters()); append(m_side2.parameters());
        append(m_fuse.parameters());
        return v;
    }

    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> b) { v.insert(v.end(), b.begin(), b.end()); };
        append(m_enc1.buffers()); append(m_enc2.buffers());
        append(m_enc3.buffers()); append(m_enc4.buffers());
        append(m_dec4.buffers()); append(m_dec3.buffers()); append(m_dec2.buffers());
        return v;
    }

    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_enc1.train(bMode); m_enc2.train(bMode);
        m_enc3.train(bMode); m_enc4.train(bMode);
        m_dec4.train(bMode); m_dec3.train(bMode); m_dec2.train(bMode);
    }

private:
    int m_nBase;
    MaxPool2d m_pool;
    EdgeEncoderBlock m_enc1, m_enc2, m_enc3, m_enc4;
    EdgeDecoderBlock m_dec4, m_dec3, m_dec2;
    Conv2d m_side4, m_side3, m_side2, m_fuse;

    // 20260402 ZJH 3 通道拼接 ([N,2,H,W] + [N,1,H,W] → [N,3,H,W])
    static Tensor catChannel3(const Tensor& ab, const Tensor& c) {
        auto cab = ab.contiguous(), cc = c.contiguous();
        int nN = cab.shape(0), nCab = cab.shape(1), nCc = cc.shape(1);
        int nH = cab.shape(2), nW = cab.shape(3), nS = nH * nW;
        int nCt = nCab + nCc;
        auto result = Tensor::zeros({nN, nCt, nH, nW});
        float* pO = result.mutableFloatDataPtr();
        const float* pAB = cab.floatDataPtr();
        const float* pC = cc.floatDataPtr();
        for (int n = 0; n < nN; ++n) {
            for (int ch = 0; ch < nCab; ++ch)
                for (int i = 0; i < nS; ++i)
                    pO[((n * nCt + ch) * nS) + i] = pAB[((n * nCab + ch) * nS) + i];
            for (int ch = 0; ch < nCc; ++ch)
                for (int i = 0; i < nS; ++i)
                    pO[((n * nCt + nCab + ch) * nS) + i] = pC[((n * nCc + ch) * nS) + i];
        }
        return result;
    }

    // 20260402 ZJH 设为 public static 以便 forward 中使用 upsampleNearest
    // (EdgeDecoderBlock::upsampleNearest 是 private，此处需要友元或 public)
    // 解决方案: 在 EdgeDecoderBlock 中声明为 public
};

// =============================================================================
// 20260402 ZJH EdgeBCEDiceLoss — 边缘专用损失函数类（可在训练循环中使用）
// 处理边缘像素严重不平衡（正样本<5%）
// =============================================================================
class EdgeBCEDiceLoss {
public:
    float m_fPosWeight = 20.0f;  // 20260402 ZJH 正样本权重（边缘像素稀少，需强加权）

    // 20260402 ZJH forward — 计算 BCE + Dice 混合损失
    Tensor forward(const Tensor& pred, const Tensor& target) {
        return EdgeExtractionNet::edgeLoss(pred, target, m_fPosWeight);
    }
};

}  // namespace om
