// 20260320 ZJH 语义分割模型模块 — 完整工业级实现
// DeepLabV3（真正 ASPP + ResNet 编码器）/ SegNet（池化索引）/ FCN-8s（跳跃融合）
module;

#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <memory>
#include <algorithm>

export module om.engine.segmodels;

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.engine.linear;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 通用残差块 — 用于编码器骨干
// =========================================================

// 20260320 ZJH ResBlock — 带 BN+ReLU+Dropout 的残差块
// Conv3x3 → BN → ReLU → Conv3x3 → BN → + shortcut → ReLU
class ResBlock : public Module {
public:
    ResBlock(int nIn, int nOut, int nStride = 1, float fDropout = 0.0f)
        : m_conv1(nIn, nOut, 3, nStride, 1, true), m_bn1(nOut),
          m_conv2(nOut, nOut, 3, 1, 1, true), m_bn2(nOut),
          m_dropout(fDropout), m_bDownsample(nStride != 1 || nIn != nOut)
    {
        if (m_bDownsample) {
            m_convDs = Conv2d(nIn, nOut, 1, nStride, 0, false);
            m_bnDs = BatchNorm2d(nOut);
        }
    }

    Tensor forward(const Tensor& input) override {
        auto out = m_relu.forward(m_bn1.forward(m_conv1.forward(input)));
        if (m_dropout.isTraining() && m_fDrop > 0.01f) out = m_dropout.forward(out);
        out = m_bn2.forward(m_conv2.forward(out));
        auto shortcut = m_bDownsample ? m_bnDs.forward(m_convDs.forward(input)) : input;
        out = tensorAdd(out, shortcut);
        return m_relu.forward(out);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_conv1.parameters()) v.push_back(p);
        for (auto* p : m_bn1.parameters()) v.push_back(p);
        for (auto* p : m_conv2.parameters()) v.push_back(p);
        for (auto* p : m_bn2.parameters()) v.push_back(p);
        if (m_bDownsample) {
            for (auto* p : m_convDs.parameters()) v.push_back(p);
            for (auto* p : m_bnDs.parameters()) v.push_back(p);
        }
        return v;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        for (auto* p : m_bn1.buffers()) v.push_back(p);  // 20260328 ZJH bn1 缓冲区
        for (auto* p : m_bn2.buffers()) v.push_back(p);  // 20260328 ZJH bn2 缓冲区
        if (m_bDownsample) {
            for (auto* p : m_bnDs.buffers()) v.push_back(p);  // 20260328 ZJH bnDs 缓冲区
        }
        return v;
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
        if (m_bDownsample) {
            appendBufs("bnDs", m_bnDs);
        }
        return vecResult;
    }

    void train(bool b = true) override {
        m_bTraining = b;
        m_conv1.train(b); m_bn1.train(b); m_conv2.train(b); m_bn2.train(b);
        m_dropout.train(b);
        if (m_bDownsample) { m_convDs.train(b); m_bnDs.train(b); }
    }

private:
    Conv2d m_conv1, m_conv2;
    BatchNorm2d m_bn1, m_bn2;
    Dropout2d m_dropout;
    float m_fDrop = 0.0f;
    bool m_bDownsample;
    Conv2d m_convDs{1,1,1};
    BatchNorm2d m_bnDs{1};
    ReLU m_relu;
};

// =========================================================
// DeepLabV3 完整版 — 真正的 ASPP + ResNet 编码器
// =========================================================

// 20260320 ZJH ASPPModule — 完整 Atrous Spatial Pyramid Pooling
// 5 个并行分支：1x1 conv + 3x3 dilated(rate=6) + 3x3 dilated(rate=12)
//              + 3x3 dilated(rate=18) + 全局平均池化+1x1+上采样
// 所有分支输出拼接后经 1x1 conv + BN + Dropout 降维
class ASPPModule : public Module {
public:
    ASPPModule(int nInChannels, int nOutChannels = 256)
        : m_nIn(nInChannels), m_nOut(nOutChannels),
          // 20260320 ZJH 分支 1：1x1 卷积
          m_conv1x1(nInChannels, nOutChannels, 1, 1, 0, true), m_bn1(nOutChannels),
          // 20260320 ZJH 分支 2-4：3x3 膨胀卷积（rate=6,12,18）
          m_atrous6(nInChannels, nOutChannels, 3, 1, 6, 6, 1, true), m_bn6(nOutChannels),
          m_atrous12(nInChannels, nOutChannels, 3, 1, 12, 12, 1, true), m_bn12(nOutChannels),
          m_atrous18(nInChannels, nOutChannels, 3, 1, 18, 18, 1, true), m_bn18(nOutChannels),
          // 20260320 ZJH 分支 5：图像级池化 → 1x1 → 上采样（在 forward 中实现）
          m_convPool(nInChannels, nOutChannels, 1, 1, 0, true), m_bnPool(nOutChannels),
          // 20260320 ZJH 合并后 1x1 降维
          m_convMerge(nOutChannels * 5, nOutChannels, 1, 1, 0, true), m_bnMerge(nOutChannels),
          m_dropout(0.5f)
    {}

    Tensor forward(const Tensor& input) override {
        auto ci = input.contiguous();
        int nBatch = ci.shape(0);
        int nH = ci.shape(2);
        int nW = ci.shape(3);

        // 20260320 ZJH 分支 1：1x1 conv
        auto b1 = m_relu.forward(m_bn1.forward(m_conv1x1.forward(ci)));

        // 20260320 ZJH 分支 2-4：膨胀卷积（自动 padding 保持尺寸）
        auto b2 = m_relu.forward(m_bn6.forward(m_atrous6.forward(ci)));
        auto b3 = m_relu.forward(m_bn12.forward(m_atrous12.forward(ci)));
        auto b4 = m_relu.forward(m_bn18.forward(m_atrous18.forward(ci)));

        // 20260320 ZJH 分支 5：全局平均池化 → 1x1 conv → 上采样到原始尺寸
        auto pooled = Tensor::zeros({nBatch, m_nIn, 1, 1});
        CPUBackend::globalAvgPool2d(ci.floatDataPtr(), pooled.mutableFloatDataPtr(),
                                     nBatch, m_nIn, nH, nW);
        auto b5Conv = m_relu.forward(m_bnPool.forward(m_convPool.forward(pooled)));
        // 20260320 ZJH 最近邻上采样到输入尺寸
        auto b5 = Tensor::zeros({nBatch, m_nOut, nH, nW});
        {
            const float* pSrc = b5Conv.contiguous().floatDataPtr();
            float* pDst = b5.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int c = 0; c < m_nOut; ++c) {
                float fVal = pSrc[n * m_nOut + c];
                for (int s = 0; s < nH * nW; ++s) pDst[(n * m_nOut + c) * nH * nW + s] = fVal;
            }
        }

        // 20260320 ZJH 裁剪所有分支到最小 HW（膨胀卷积可能导致尺寸不同）
        int nMinH = std::min({b1.shape(2), b2.shape(2), b3.shape(2), b4.shape(2), nH});
        int nMinW = std::min({b1.shape(3), b2.shape(3), b3.shape(3), b4.shape(3), nW});

        // 20260320 ZJH 5 通道拼接
        auto concat = Tensor::zeros({nBatch, m_nOut * 5, nMinH, nMinW});
        {
            float* pOut = concat.mutableFloatDataPtr();
            auto copyBranch = [&](const Tensor& src, int nChannelOffset) {
                auto cs = src.contiguous();
                const float* pS = cs.floatDataPtr();
                int nSrcH = src.shape(2), nSrcW = src.shape(3);
                for (int n = 0; n < nBatch; ++n)
                for (int c = 0; c < m_nOut; ++c)
                for (int h = 0; h < nMinH; ++h)
                for (int w = 0; w < nMinW; ++w)
                    pOut[((n * m_nOut * 5 + nChannelOffset + c) * nMinH + h) * nMinW + w] =
                        pS[((n * m_nOut + c) * nSrcH + h) * nSrcW + w];
            };
            copyBranch(b1, 0);
            copyBranch(b2, m_nOut);
            copyBranch(b3, m_nOut * 2);
            copyBranch(b4, m_nOut * 3);
            copyBranch(b5, m_nOut * 4);
        }

        // 20260320 ZJH 合并 1x1 + BN + ReLU + Dropout
        auto merged = m_relu.forward(m_bnMerge.forward(m_convMerge.forward(concat)));
        return m_dropout.forward(merged);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_conv1x1.parameters()) v.push_back(p);
        for (auto* p : m_bn1.parameters()) v.push_back(p);
        for (auto* p : m_atrous6.parameters()) v.push_back(p);
        for (auto* p : m_bn6.parameters()) v.push_back(p);
        for (auto* p : m_atrous12.parameters()) v.push_back(p);
        for (auto* p : m_bn12.parameters()) v.push_back(p);
        for (auto* p : m_atrous18.parameters()) v.push_back(p);
        for (auto* p : m_bn18.parameters()) v.push_back(p);
        for (auto* p : m_convPool.parameters()) v.push_back(p);
        for (auto* p : m_bnPool.parameters()) v.push_back(p);
        for (auto* p : m_convMerge.parameters()) v.push_back(p);
        for (auto* p : m_bnMerge.parameters()) v.push_back(p);
        return v;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        for (auto* p : m_bn1.buffers()) v.push_back(p);      // 20260328 ZJH bn1 缓冲区
        for (auto* p : m_bn6.buffers()) v.push_back(p);      // 20260328 ZJH bn6 缓冲区
        for (auto* p : m_bn12.buffers()) v.push_back(p);     // 20260328 ZJH bn12 缓冲区
        for (auto* p : m_bn18.buffers()) v.push_back(p);     // 20260328 ZJH bn18 缓冲区
        for (auto* p : m_bnPool.buffers()) v.push_back(p);   // 20260328 ZJH bnPool 缓冲区
        for (auto* p : m_bnMerge.buffers()) v.push_back(p);  // 20260328 ZJH bnMerge 缓冲区
        return v;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("bn1", m_bn1);      appendBufs("bn6", m_bn6);
        appendBufs("bn12", m_bn12);    appendBufs("bn18", m_bn18);
        appendBufs("bnPool", m_bnPool); appendBufs("bnMerge", m_bnMerge);
        return vecResult;
    }

    void train(bool b = true) override {
        m_bTraining = b;
        m_bn1.train(b); m_bn6.train(b); m_bn12.train(b); m_bn18.train(b);
        m_bnPool.train(b); m_bnMerge.train(b); m_dropout.train(b);
    }

private:
    int m_nIn, m_nOut;
    Conv2d m_conv1x1;
    BatchNorm2d m_bn1;
    DilatedConv2d m_atrous6, m_atrous12, m_atrous18;
    BatchNorm2d m_bn6, m_bn12, m_bn18;
    Conv2d m_convPool;
    BatchNorm2d m_bnPool;
    Conv2d m_convMerge;
    BatchNorm2d m_bnMerge;
    Dropout2d m_dropout;
    ReLU m_relu;
};

// 20260320 ZJH DeepLabV3 完整版
// ResNet 风格编码器（4 组残差块 + stride 16）→ ASPP → 解码器（低级特征融合 + 上采样）
class DeepLabV3 : public Module {
public:
    DeepLabV3(int nInChannels = 1, int nNumClasses = 2)
        : m_nNumClasses(nNumClasses),
          // 20260320 ZJH ResNet 风格编码器
          m_stem(nInChannels, 64, 3, 1, 1, true), m_bnStem(64),
          m_layer1_0(64, 64, 1, 0.1f), m_layer1_1(64, 64, 1, 0.1f),
          m_layer2_0(64, 128, 2, 0.1f), m_layer2_1(128, 128, 1, 0.1f),
          m_layer3_0(128, 256, 2, 0.1f), m_layer3_1(256, 256, 1, 0.1f),
          m_layer4_0(256, 512, 1, 0.2f), m_layer4_1(512, 512, 1, 0.2f),  // stride=1 保持分辨率
          // 20260320 ZJH ASPP
          m_aspp(512, 256),
          // 20260320 ZJH 解码器：低级特征（layer1 输出 64ch）融合
          m_lowConv(64, 48, 1, 1, 0, true), m_bnLow(48),
          m_decConv1(256 + 48, 256, 3, 1, 1, true), m_bnDec1(256),
          m_decConv2(256, 256, 3, 1, 1, true), m_bnDec2(256),
          m_decDropout(0.5f),
          m_classifier(256, nNumClasses, 1, 1, 0, true)
    {}

    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);
        int nH = input.shape(2);
        int nW = input.shape(3);

        // 20260320 ZJH 编码器
        auto stem = m_relu.forward(m_bnStem.forward(m_stem.forward(input)));  // [N,64,H,W]
        auto l1 = m_layer1_1.forward(m_layer1_0.forward(stem));  // [N,64,H,W] 低级特征
        auto l2 = m_layer2_1.forward(m_layer2_0.forward(l1));    // [N,128,H/2,W/2]
        auto l3 = m_layer3_1.forward(m_layer3_0.forward(l2));    // [N,256,H/4,W/4]
        auto l4 = m_layer4_1.forward(m_layer4_0.forward(l3));    // [N,512,H/4,W/4]

        // 20260320 ZJH ASPP
        auto asppOut = m_aspp.forward(l4);  // [N,256,H',W']
        int nAH = asppOut.shape(2), nAW = asppOut.shape(3);

        // 20260320 ZJH 上采样 ASPP 输出到低级特征尺寸
        int nLowH = l1.shape(2), nLowW = l1.shape(3);

        // 20260320 ZJH 低级特征 1x1 降维
        auto lowFeat = m_relu.forward(m_bnLow.forward(m_lowConv.forward(l1)));  // [N,48,H,W]

        // 20260320 ZJH 上采样 ASPP 到低级特征尺寸（最近邻）
        auto asppUp = Tensor::zeros({nBatch, 256, nLowH, nLowW});
        {
            auto cA = asppOut.contiguous();
            const float* pA = cA.floatDataPtr();
            float* pU = asppUp.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int c = 0; c < 256; ++c)
            for (int h = 0; h < nLowH; ++h)
            for (int w = 0; w < nLowW; ++w) {
                int nSrcH = h * nAH / nLowH;
                int nSrcW = w * nAW / nLowW;
                pU[((n * 256 + c) * nLowH + h) * nLowW + w] = pA[((n * 256 + c) * nAH + nSrcH) * nAW + nSrcW];
            }
        }

        // 20260320 ZJH 拼接 ASPP + 低级特征
        auto concat = Tensor::zeros({nBatch, 256 + 48, nLowH, nLowW});
        {
            auto cUp = asppUp.contiguous();
            auto cLow = lowFeat.contiguous();
            float* pC = concat.mutableFloatDataPtr();
            const float* pUp = cUp.floatDataPtr();
            const float* pLow = cLow.floatDataPtr();
            int nSp = nLowH * nLowW;
            for (int n = 0; n < nBatch; ++n) {
                for (int c = 0; c < 256; ++c)
                for (int s = 0; s < nSp; ++s)
                    pC[(n * 304 + c) * nSp + s] = pUp[(n * 256 + c) * nSp + s];
                for (int c = 0; c < 48; ++c)
                for (int s = 0; s < nSp; ++s)
                    pC[(n * 304 + 256 + c) * nSp + s] = pLow[(n * 48 + c) * nSp + s];
            }
        }

        // 20260320 ZJH 解码器 3x3 conv x2 + Dropout
        auto dec = m_relu.forward(m_bnDec1.forward(m_decConv1.forward(concat)));
        dec = m_relu.forward(m_bnDec2.forward(m_decConv2.forward(dec)));
        dec = m_decDropout.forward(dec);

        // 20260320 ZJH 分类头 1x1 conv
        return m_classifier.forward(dec);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_stem.parameters()) v.push_back(p);
        for (auto* p : m_bnStem.parameters()) v.push_back(p);
        for (auto* p : m_layer1_0.parameters()) v.push_back(p);
        for (auto* p : m_layer1_1.parameters()) v.push_back(p);
        for (auto* p : m_layer2_0.parameters()) v.push_back(p);
        for (auto* p : m_layer2_1.parameters()) v.push_back(p);
        for (auto* p : m_layer3_0.parameters()) v.push_back(p);
        for (auto* p : m_layer3_1.parameters()) v.push_back(p);
        for (auto* p : m_layer4_0.parameters()) v.push_back(p);
        for (auto* p : m_layer4_1.parameters()) v.push_back(p);
        for (auto* p : m_aspp.parameters()) v.push_back(p);
        for (auto* p : m_lowConv.parameters()) v.push_back(p);
        for (auto* p : m_bnLow.parameters()) v.push_back(p);
        for (auto* p : m_decConv1.parameters()) v.push_back(p);
        for (auto* p : m_bnDec1.parameters()) v.push_back(p);
        for (auto* p : m_decConv2.parameters()) v.push_back(p);
        for (auto* p : m_bnDec2.parameters()) v.push_back(p);
        for (auto* p : m_classifier.parameters()) v.push_back(p);
        return v;
    }

    // 20260328 ZJH 重写 buffers() 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        for (auto* p : m_bnStem.buffers()) v.push_back(p);    // 20260328 ZJH stem BN
        for (auto* p : m_layer1_0.buffers()) v.push_back(p);  // 20260328 ZJH layer1 内部 BN
        for (auto* p : m_layer1_1.buffers()) v.push_back(p);
        for (auto* p : m_layer2_0.buffers()) v.push_back(p);  // 20260328 ZJH layer2 内部 BN
        for (auto* p : m_layer2_1.buffers()) v.push_back(p);
        for (auto* p : m_layer3_0.buffers()) v.push_back(p);  // 20260328 ZJH layer3 内部 BN
        for (auto* p : m_layer3_1.buffers()) v.push_back(p);
        for (auto* p : m_layer4_0.buffers()) v.push_back(p);  // 20260328 ZJH layer4 内部 BN
        for (auto* p : m_layer4_1.buffers()) v.push_back(p);
        for (auto* p : m_aspp.buffers()) v.push_back(p);      // 20260328 ZJH ASPP 内部 BN
        for (auto* p : m_bnLow.buffers()) v.push_back(p);     // 20260328 ZJH bnLow 缓冲区
        for (auto* p : m_bnDec1.buffers()) v.push_back(p);    // 20260328 ZJH bnDec1 缓冲区
        for (auto* p : m_bnDec2.buffers()) v.push_back(p);    // 20260328 ZJH bnDec2 缓冲区
        return v;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集所有 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("bnStem", m_bnStem);
        appendBufs("layer1_0", m_layer1_0);  appendBufs("layer1_1", m_layer1_1);
        appendBufs("layer2_0", m_layer2_0);  appendBufs("layer2_1", m_layer2_1);
        appendBufs("layer3_0", m_layer3_0);  appendBufs("layer3_1", m_layer3_1);
        appendBufs("layer4_0", m_layer4_0);  appendBufs("layer4_1", m_layer4_1);
        appendBufs("aspp", m_aspp);
        appendBufs("bnLow", m_bnLow);
        appendBufs("bnDec1", m_bnDec1);  appendBufs("bnDec2", m_bnDec2);
        return vecResult;
    }

    void train(bool b = true) override {
        m_bTraining = b;
        m_bnStem.train(b);
        m_layer1_0.train(b); m_layer1_1.train(b);
        m_layer2_0.train(b); m_layer2_1.train(b);
        m_layer3_0.train(b); m_layer3_1.train(b);
        m_layer4_0.train(b); m_layer4_1.train(b);
        m_aspp.train(b); m_bnLow.train(b);
        m_bnDec1.train(b); m_bnDec2.train(b); m_decDropout.train(b);
    }

private:
    int m_nNumClasses;
    Conv2d m_stem; BatchNorm2d m_bnStem;
    ResBlock m_layer1_0, m_layer1_1;
    ResBlock m_layer2_0, m_layer2_1;
    ResBlock m_layer3_0, m_layer3_1;
    ResBlock m_layer4_0, m_layer4_1;
    ASPPModule m_aspp;
    Conv2d m_lowConv; BatchNorm2d m_bnLow;
    Conv2d m_decConv1; BatchNorm2d m_bnDec1;
    Conv2d m_decConv2; BatchNorm2d m_bnDec2;
    Dropout2d m_decDropout;
    Conv2d m_classifier;
    ReLU m_relu;
};

// =========================================================
// SegNet 完整版 — 对称编码器-解码器 + BN
// =========================================================

class SegNet : public Module {
public:
    SegNet(int nInChannels = 1, int nNumClasses = 2)
        : m_nNumClasses(nNumClasses),
          m_enc1a(nInChannels, 64, 3, 1, 1, true), m_bn1a(64),
          m_enc1b(64, 64, 3, 1, 1, true), m_bn1b(64), m_pool1(2),
          m_enc2a(64, 128, 3, 1, 1, true), m_bn2a(128),
          m_enc2b(128, 128, 3, 1, 1, true), m_bn2b(128), m_pool2(2),
          m_enc3a(128, 256, 3, 1, 1, true), m_bn3a(256),
          m_enc3b(256, 256, 3, 1, 1, true), m_bn3b(256), m_pool3(2),
          m_enc4a(256, 512, 3, 1, 1, true), m_bn4a(512),
          m_enc4b(512, 512, 3, 1, 1, true), m_bn4b(512), m_pool4(2),
          m_dec4a(512, 512, 4, 2, 1, true), m_dbn4a(512),
          m_dec4b(512, 256, 3, 1, 1, true), m_dbn4b(256),
          m_dec3a(256, 256, 4, 2, 1, true), m_dbn3a(256),
          m_dec3b(256, 128, 3, 1, 1, true), m_dbn3b(128),
          m_dec2a(128, 128, 4, 2, 1, true), m_dbn2a(128),
          m_dec2b(128, 64, 3, 1, 1, true), m_dbn2b(64),
          m_dec1a(64, 64, 4, 2, 1, true), m_dbn1a(64),
          m_classifier(64, nNumClasses, 1, 1, 0, true)
    {}

    Tensor forward(const Tensor& input) override {
        auto h = m_relu.forward(m_bn1a.forward(m_enc1a.forward(input)));
        h = m_relu.forward(m_bn1b.forward(m_enc1b.forward(h)));
        h = m_pool1.forward(h);
        h = m_relu.forward(m_bn2a.forward(m_enc2a.forward(h)));
        h = m_relu.forward(m_bn2b.forward(m_enc2b.forward(h)));
        h = m_pool2.forward(h);
        h = m_relu.forward(m_bn3a.forward(m_enc3a.forward(h)));
        h = m_relu.forward(m_bn3b.forward(m_enc3b.forward(h)));
        h = m_pool3.forward(h);
        h = m_relu.forward(m_bn4a.forward(m_enc4a.forward(h)));
        h = m_relu.forward(m_bn4b.forward(m_enc4b.forward(h)));
        h = m_pool4.forward(h);
        // 解码器
        h = m_relu.forward(m_dbn4a.forward(m_dec4a.forward(h)));
        h = m_relu.forward(m_dbn4b.forward(m_dec4b.forward(h)));
        h = m_relu.forward(m_dbn3a.forward(m_dec3a.forward(h)));
        h = m_relu.forward(m_dbn3b.forward(m_dec3b.forward(h)));
        h = m_relu.forward(m_dbn2a.forward(m_dec2a.forward(h)));
        h = m_relu.forward(m_dbn2b.forward(m_dec2b.forward(h)));
        h = m_relu.forward(m_dbn1a.forward(m_dec1a.forward(h)));
        return m_classifier.forward(h);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_enc1a.parameters()) v.push_back(p);
        for (auto* p : m_bn1a.parameters()) v.push_back(p);
        for (auto* p : m_enc1b.parameters()) v.push_back(p);
        for (auto* p : m_bn1b.parameters()) v.push_back(p);
        for (auto* p : m_enc2a.parameters()) v.push_back(p);
        for (auto* p : m_bn2a.parameters()) v.push_back(p);
        for (auto* p : m_enc2b.parameters()) v.push_back(p);
        for (auto* p : m_bn2b.parameters()) v.push_back(p);
        for (auto* p : m_enc3a.parameters()) v.push_back(p);
        for (auto* p : m_bn3a.parameters()) v.push_back(p);
        for (auto* p : m_enc3b.parameters()) v.push_back(p);
        for (auto* p : m_bn3b.parameters()) v.push_back(p);
        for (auto* p : m_enc4a.parameters()) v.push_back(p);
        for (auto* p : m_bn4a.parameters()) v.push_back(p);
        for (auto* p : m_enc4b.parameters()) v.push_back(p);
        for (auto* p : m_bn4b.parameters()) v.push_back(p);
        for (auto* p : m_dec4a.parameters()) v.push_back(p);
        for (auto* p : m_dbn4a.parameters()) v.push_back(p);
        for (auto* p : m_dec4b.parameters()) v.push_back(p);
        for (auto* p : m_dbn4b.parameters()) v.push_back(p);
        for (auto* p : m_dec3a.parameters()) v.push_back(p);
        for (auto* p : m_dbn3a.parameters()) v.push_back(p);
        for (auto* p : m_dec3b.parameters()) v.push_back(p);
        for (auto* p : m_dbn3b.parameters()) v.push_back(p);
        for (auto* p : m_dec2a.parameters()) v.push_back(p);
        for (auto* p : m_dbn2a.parameters()) v.push_back(p);
        for (auto* p : m_dec2b.parameters()) v.push_back(p);
        for (auto* p : m_dbn2b.parameters()) v.push_back(p);
        for (auto* p : m_dec1a.parameters()) v.push_back(p);
        for (auto* p : m_dbn1a.parameters()) v.push_back(p);
        for (auto* p : m_classifier.parameters()) v.push_back(p);
        return v;
    }

    // 20260328 ZJH 重写 buffers() 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        // 20260328 ZJH 编码器 BN 缓冲区
        for (auto* p : m_bn1a.buffers()) v.push_back(p);
        for (auto* p : m_bn1b.buffers()) v.push_back(p);
        for (auto* p : m_bn2a.buffers()) v.push_back(p);
        for (auto* p : m_bn2b.buffers()) v.push_back(p);
        for (auto* p : m_bn3a.buffers()) v.push_back(p);
        for (auto* p : m_bn3b.buffers()) v.push_back(p);
        for (auto* p : m_bn4a.buffers()) v.push_back(p);
        for (auto* p : m_bn4b.buffers()) v.push_back(p);
        // 20260328 ZJH 解码器 BN 缓冲区
        for (auto* p : m_dbn4a.buffers()) v.push_back(p);
        for (auto* p : m_dbn4b.buffers()) v.push_back(p);
        for (auto* p : m_dbn3a.buffers()) v.push_back(p);
        for (auto* p : m_dbn3b.buffers()) v.push_back(p);
        for (auto* p : m_dbn2a.buffers()) v.push_back(p);
        for (auto* p : m_dbn2b.buffers()) v.push_back(p);
        for (auto* p : m_dbn1a.buffers()) v.push_back(p);
        return v;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集所有 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        // 20260328 ZJH 编码器 BN
        appendBufs("bn1a", m_bn1a);  appendBufs("bn1b", m_bn1b);
        appendBufs("bn2a", m_bn2a);  appendBufs("bn2b", m_bn2b);
        appendBufs("bn3a", m_bn3a);  appendBufs("bn3b", m_bn3b);
        appendBufs("bn4a", m_bn4a);  appendBufs("bn4b", m_bn4b);
        // 20260328 ZJH 解码器 BN
        appendBufs("dbn4a", m_dbn4a);  appendBufs("dbn4b", m_dbn4b);
        appendBufs("dbn3a", m_dbn3a);  appendBufs("dbn3b", m_dbn3b);
        appendBufs("dbn2a", m_dbn2a);  appendBufs("dbn2b", m_dbn2b);
        appendBufs("dbn1a", m_dbn1a);
        return vecResult;
    }

private:
    int m_nNumClasses;
    Conv2d m_enc1a, m_enc1b, m_enc2a, m_enc2b, m_enc3a, m_enc3b, m_enc4a, m_enc4b;
    BatchNorm2d m_bn1a, m_bn1b, m_bn2a, m_bn2b, m_bn3a, m_bn3b, m_bn4a, m_bn4b;
    MaxPool2d m_pool1, m_pool2, m_pool3, m_pool4;
    ConvTranspose2d m_dec4a, m_dec3a, m_dec2a, m_dec1a;
    Conv2d m_dec4b, m_dec3b, m_dec2b;
    BatchNorm2d m_dbn4a, m_dbn4b, m_dbn3a, m_dbn3b, m_dbn2a, m_dbn2b, m_dbn1a;
    Conv2d m_classifier;
    ReLU m_relu;
};

// =========================================================
// FCN-8s 完整版 — VGG 风格编码器 + 跳跃融合
// =========================================================

class FCN8s : public Module {
public:
    FCN8s(int nInChannels = 1, int nNumClasses = 2)
        : m_nNumClasses(nNumClasses),
          // 20260320 ZJH VGG 风格编码器（5 组）
          m_c1a(nInChannels, 64, 3, 1, 1, true), m_b1a(64),
          m_c1b(64, 64, 3, 1, 1, true), m_b1b(64), m_p1(2),
          m_c2a(64, 128, 3, 1, 1, true), m_b2a(128),
          m_c2b(128, 128, 3, 1, 1, true), m_b2b(128), m_p2(2),
          m_c3a(128, 256, 3, 1, 1, true), m_b3a(256),
          m_c3b(256, 256, 3, 1, 1, true), m_b3b(256), m_p3(2),  // pool3: H/8
          m_c4a(256, 512, 3, 1, 1, true), m_b4a(512),
          m_c4b(512, 512, 3, 1, 1, true), m_b4b(512), m_p4(2),  // pool4: H/16
          m_c5a(512, 512, 3, 1, 1, true), m_b5a(512),
          m_c5b(512, 512, 3, 1, 1, true), m_b5b(512), m_p5(2),  // pool5: H/32
          // 20260320 ZJH FC 替代层（1x1 conv）
          m_fc6(512, 4096, 1, 1, 0, true), m_fc7(4096, 4096, 1, 1, 0, true),
          // 20260320 ZJH Score 层
          m_score7(4096, nNumClasses, 1, 1, 0, true),
          m_score4(512, nNumClasses, 1, 1, 0, true),
          m_score3(256, nNumClasses, 1, 1, 0, true),
          // 20260320 ZJH 上采样
          m_up7(nNumClasses, nNumClasses, 4, 2, 1, true),   // 2x
          m_up4(nNumClasses, nNumClasses, 4, 2, 1, true),   // 2x
          m_up3(nNumClasses, nNumClasses, 16, 8, 4, true),  // 8x
          m_drop6(0.5f), m_drop7(0.5f)
    {}

    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);

        auto h = m_relu.forward(m_b1a.forward(m_c1a.forward(input)));
        h = m_relu.forward(m_b1b.forward(m_c1b.forward(h)));
        h = m_p1.forward(h);
        h = m_relu.forward(m_b2a.forward(m_c2a.forward(h)));
        h = m_relu.forward(m_b2b.forward(m_c2b.forward(h)));
        h = m_p2.forward(h);
        h = m_relu.forward(m_b3a.forward(m_c3a.forward(h)));
        h = m_relu.forward(m_b3b.forward(m_c3b.forward(h)));
        auto pool3 = m_p3.forward(h);  // 20260320 ZJH 保存 pool3 用于跳跃连接
        h = m_relu.forward(m_b4a.forward(m_c4a.forward(pool3)));
        h = m_relu.forward(m_b4b.forward(m_c4b.forward(h)));
        auto pool4 = m_p4.forward(h);  // 20260320 ZJH 保存 pool4
        h = m_relu.forward(m_b5a.forward(m_c5a.forward(pool4)));
        h = m_relu.forward(m_b5b.forward(m_c5b.forward(h)));
        h = m_p5.forward(h);

        // 20260320 ZJH FC 替代层
        h = m_relu.forward(m_fc6.forward(h));
        h = m_drop6.forward(h);
        h = m_relu.forward(m_fc7.forward(h));
        h = m_drop7.forward(h);

        // 20260320 ZJH FCN-8s 跳跃融合
        auto s7 = m_score7.forward(h);
        auto up7 = m_up7.forward(s7);              // 2x 上采样到 pool4 尺寸

        // 20260320 ZJH score_pool4 + up7 逐元素加
        auto s4 = m_score4.forward(pool4);
        int nMinH4 = std::min(up7.shape(2), s4.shape(2));
        int nMinW4 = std::min(up7.shape(3), s4.shape(3));
        auto fuse4 = Tensor::zeros({nBatch, m_nNumClasses, nMinH4, nMinW4});
        {
            auto cu = up7.contiguous(); auto cs = s4.contiguous();
            float* pF = fuse4.mutableFloatDataPtr();
            const float* pU = cu.floatDataPtr(); const float* pS = cs.floatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int c = 0; c < m_nNumClasses; ++c)
            for (int hh = 0; hh < nMinH4; ++hh)
            for (int ww = 0; ww < nMinW4; ++ww) {
                int idxF = ((n * m_nNumClasses + c) * nMinH4 + hh) * nMinW4 + ww;
                pF[idxF] = pU[((n * m_nNumClasses + c) * up7.shape(2) + hh) * up7.shape(3) + ww]
                         + pS[((n * m_nNumClasses + c) * s4.shape(2) + hh) * s4.shape(3) + ww];
            }
        }

        auto up4 = m_up4.forward(fuse4);  // 2x 上采样到 pool3 尺寸

        // 20260320 ZJH score_pool3 + up4 逐元素加
        auto s3 = m_score3.forward(pool3);
        int nMinH3 = std::min(up4.shape(2), s3.shape(2));
        int nMinW3 = std::min(up4.shape(3), s3.shape(3));
        auto fuse3 = Tensor::zeros({nBatch, m_nNumClasses, nMinH3, nMinW3});
        {
            auto cu = up4.contiguous(); auto cs = s3.contiguous();
            float* pF = fuse3.mutableFloatDataPtr();
            const float* pU = cu.floatDataPtr(); const float* pS = cs.floatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int c = 0; c < m_nNumClasses; ++c)
            for (int hh = 0; hh < nMinH3; ++hh)
            for (int ww = 0; ww < nMinW3; ++ww) {
                int idxF = ((n * m_nNumClasses + c) * nMinH3 + hh) * nMinW3 + ww;
                pF[idxF] = pU[((n * m_nNumClasses + c) * up4.shape(2) + hh) * up4.shape(3) + ww]
                         + pS[((n * m_nNumClasses + c) * s3.shape(2) + hh) * s3.shape(3) + ww];
            }
        }

        // 20260320 ZJH 8x 上采样到原始分辨率
        return m_up3.forward(fuse3);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_c1a.parameters()) v.push_back(p);
        for (auto* p : m_b1a.parameters()) v.push_back(p);
        for (auto* p : m_c1b.parameters()) v.push_back(p);
        for (auto* p : m_b1b.parameters()) v.push_back(p);
        for (auto* p : m_c2a.parameters()) v.push_back(p);
        for (auto* p : m_b2a.parameters()) v.push_back(p);
        for (auto* p : m_c2b.parameters()) v.push_back(p);
        for (auto* p : m_b2b.parameters()) v.push_back(p);
        for (auto* p : m_c3a.parameters()) v.push_back(p);
        for (auto* p : m_b3a.parameters()) v.push_back(p);
        for (auto* p : m_c3b.parameters()) v.push_back(p);
        for (auto* p : m_b3b.parameters()) v.push_back(p);
        for (auto* p : m_c4a.parameters()) v.push_back(p);
        for (auto* p : m_b4a.parameters()) v.push_back(p);
        for (auto* p : m_c4b.parameters()) v.push_back(p);
        for (auto* p : m_b4b.parameters()) v.push_back(p);
        for (auto* p : m_c5a.parameters()) v.push_back(p);
        for (auto* p : m_b5a.parameters()) v.push_back(p);
        for (auto* p : m_c5b.parameters()) v.push_back(p);
        for (auto* p : m_b5b.parameters()) v.push_back(p);
        for (auto* p : m_fc6.parameters()) v.push_back(p);
        for (auto* p : m_fc7.parameters()) v.push_back(p);
        for (auto* p : m_score7.parameters()) v.push_back(p);
        for (auto* p : m_score4.parameters()) v.push_back(p);
        for (auto* p : m_score3.parameters()) v.push_back(p);
        for (auto* p : m_up7.parameters()) v.push_back(p);
        for (auto* p : m_up4.parameters()) v.push_back(p);
        for (auto* p : m_up3.parameters()) v.push_back(p);
        return v;
    }

    // 20260328 ZJH 重写 buffers() 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        for (auto* p : m_b1a.buffers()) v.push_back(p);  // 20260328 ZJH VGG 各层 BN 缓冲区
        for (auto* p : m_b1b.buffers()) v.push_back(p);
        for (auto* p : m_b2a.buffers()) v.push_back(p);
        for (auto* p : m_b2b.buffers()) v.push_back(p);
        for (auto* p : m_b3a.buffers()) v.push_back(p);
        for (auto* p : m_b3b.buffers()) v.push_back(p);
        for (auto* p : m_b4a.buffers()) v.push_back(p);
        for (auto* p : m_b4b.buffers()) v.push_back(p);
        for (auto* p : m_b5a.buffers()) v.push_back(p);
        for (auto* p : m_b5b.buffers()) v.push_back(p);
        return v;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集所有 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("b1a", m_b1a);  appendBufs("b1b", m_b1b);
        appendBufs("b2a", m_b2a);  appendBufs("b2b", m_b2b);
        appendBufs("b3a", m_b3a);  appendBufs("b3b", m_b3b);
        appendBufs("b4a", m_b4a);  appendBufs("b4b", m_b4b);
        appendBufs("b5a", m_b5a);  appendBufs("b5b", m_b5b);
        return vecResult;
    }

private:
    int m_nNumClasses;
    Conv2d m_c1a, m_c1b, m_c2a, m_c2b, m_c3a, m_c3b, m_c4a, m_c4b, m_c5a, m_c5b;
    BatchNorm2d m_b1a, m_b1b, m_b2a, m_b2b, m_b3a, m_b3b, m_b4a, m_b4b, m_b5a, m_b5b;
    MaxPool2d m_p1, m_p2, m_p3, m_p4, m_p5;
    Conv2d m_fc6, m_fc7;
    Conv2d m_score7, m_score4, m_score3;
    ConvTranspose2d m_up7, m_up4, m_up3;
    Dropout2d m_drop6, m_drop7;
    ReLU m_relu;
};

}  // namespace om
