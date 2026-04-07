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
import om.engine.mobilenet;
import om.hal.cpu_backend;
import om.hal.cuda_backend;

export namespace om {

// =========================================================
// 通用残差块 — 用于编码器骨干
// =========================================================

// 20260320 ZJH ResBlock — 带 BN+ReLU+Dropout 的残差块
// Conv3x3 → BN → ReLU → Conv3x3 → BN → + shortcut → ReLU
class ResBlock : public Module {
public:
    // 20260331 ZJH 修复: m_convDs/m_bnDs 必须在初始化列表中构造正确尺寸
    // 不能在构造函数体中 operator= 覆盖，因为 registerParameter 存的是 &m_weight
    // move assignment 不更新内部自指针 → 导致 parameters() 返回悬空指针（BAD PARAM）
    // 20260402 ZJH 新增 bUseGroupNorm 参数，支持 GN/BN 双模式切换
    ResBlock(int nIn, int nOut, int nStride = 1, float fDropout = 0.0f, bool bUseGroupNorm = false)
        : m_conv1(nIn, nOut, 3, nStride, 1, true), m_bn1(nOut),
          m_conv2(nOut, nOut, 3, 1, 1, true), m_bn2(nOut),
          m_dropout(fDropout), m_fDrop(fDropout),
          m_bDownsample(nStride != 1 || nIn != nOut),
          m_convDs(nIn, nOut, 1, nStride, 0, false), m_bnDs(nOut),
          m_gn1(nOut), m_gn2(nOut), m_gnDs(nOut),  // 20260402 ZJH GroupNorm 实例（nGroups 自动调整）
          m_bUseGroupNorm(bUseGroupNorm)            // 20260402 ZJH 记录归一化模式
    {
    }

    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH 根据 m_bUseGroupNorm 选择归一化层（GN 或 BN）
        // 不使用三元运算符（BatchNorm2d 和 GroupNorm2d 类型不同，无法隐式转换）
        auto conv1Out = m_conv1.forward(input);  // 20260402 ZJH 第一层卷积
        auto norm1Out = m_bUseGroupNorm ? m_gn1.forward(conv1Out) : m_bn1.forward(conv1Out);
        auto out = m_relu.forward(norm1Out);
        if (m_dropout.isTraining() && m_fDrop > 0.01f) out = m_dropout.forward(out);
        auto conv2Out = m_conv2.forward(out);  // 20260402 ZJH 第二层卷积
        out = m_bUseGroupNorm ? m_gn2.forward(conv2Out) : m_bn2.forward(conv2Out);
        Tensor shortcut;  // 20260402 ZJH 残差分支
        if (m_bDownsample) {
            auto dsConvOut = m_convDs.forward(input);
            shortcut = m_bUseGroupNorm ? m_gnDs.forward(dsConvOut) : m_bnDs.forward(dsConvOut);
        } else {
            shortcut = input;
        }
        out = tensorAdd(out, shortcut);
        return m_relu.forward(out);
    }

    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        // 20260402 ZJH 辅助 lambda: 追加子模块参数
        auto append = [&](std::vector<Tensor*> ps) { for (auto* p : ps) v.push_back(p); };
        append(m_conv1.parameters());
        append(m_bUseGroupNorm ? m_gn1.parameters() : m_bn1.parameters());  // 20260402 ZJH 选择归一化层参数
        append(m_conv2.parameters());
        append(m_bUseGroupNorm ? m_gn2.parameters() : m_bn2.parameters());  // 20260402 ZJH 选择归一化层参数
        if (m_bDownsample) {
            append(m_convDs.parameters());
            append(m_bUseGroupNorm ? m_gnDs.parameters() : m_bnDs.parameters());  // 20260402 ZJH 下采样归一化
        }
        return v;
    }

    // 20260331 ZJH 重写 namedParameters() 收集所有子层命名参数
    // 20260402 ZJH 支持 GN/BN 双模式（命名使用 gn1/gn2/gnDs 或 bn1/bn2/bnDs）
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        // 20260331 ZJH 辅助 lambda：为子层参数添加前缀
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };
        appendParams("conv1", m_conv1);  // 20260331 ZJH 第一层 3x3 卷积参数
        // 20260402 ZJH 根据归一化模式选择 GN 或 BN 命名参数
        if (m_bUseGroupNorm) appendParams("gn1", m_gn1);
        else                 appendParams("bn1", m_bn1);
        appendParams("conv2", m_conv2);  // 20260331 ZJH 第二层 3x3 卷积参数
        if (m_bUseGroupNorm) appendParams("gn2", m_gn2);
        else                 appendParams("bn2", m_bn2);
        if (m_bDownsample) {
            appendParams("convDs", m_convDs);  // 20260331 ZJH 下采样 1x1 卷积参数
            if (m_bUseGroupNorm) appendParams("gnDs", m_gnDs);
            else                 appendParams("bnDs", m_bnDs);
        }
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    // 20260402 ZJH GN 没有 running stats，直接返回空
    std::vector<Tensor*> buffers() override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
        std::vector<Tensor*> v;
        for (auto* p : m_bn1.buffers()) v.push_back(p);  // 20260328 ZJH bn1 缓冲区
        for (auto* p : m_bn2.buffers()) v.push_back(p);  // 20260328 ZJH bn2 缓冲区
        if (m_bDownsample) {
            for (auto* p : m_bnDs.buffers()) v.push_back(p);  // 20260328 ZJH bnDs 缓冲区
        }
        return v;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    // 20260402 ZJH GN 模式下返回空（GN 无 running stats）
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
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
        m_conv1.train(b); m_conv2.train(b);
        m_dropout.train(b);
        // 20260402 ZJH 训练模式传播到当前使用的归一化层
        if (m_bUseGroupNorm) {
            m_gn1.train(b); m_gn2.train(b);
            if (m_bDownsample) m_gnDs.train(b);
        } else {
            m_bn1.train(b); m_bn2.train(b);
            if (m_bDownsample) m_bnDs.train(b);
        }
    }

private:
    Conv2d m_conv1, m_conv2;  // 20260406 ZJH 两层 3x3 卷积
    BatchNorm2d m_bn1, m_bn2;  // 20260406 ZJH 两层 BatchNorm
    Dropout2d m_dropout;  // 20260406 ZJH 空间 Dropout（仅训练模式生效）
    float m_fDrop = 0.0f;  // 20260406 ZJH Dropout 概率
    bool m_bDownsample;  // 20260406 ZJH 是否需要下采样捷径（stride!=1 或通道数变化）
    Conv2d m_convDs;       // 20260331 ZJH 在初始化列表中构造（不再使用默认初始化器）
    BatchNorm2d m_bnDs;    // 20260331 ZJH 在初始化列表中构造
    GroupNorm2d m_gn1;     // 20260402 ZJH 第一层 GroupNorm（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gn2;     // 20260402 ZJH 第二层 GroupNorm（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gnDs;    // 20260402 ZJH 下采样 GroupNorm（bUseGroupNorm=true 时使用）
    bool m_bUseGroupNorm = false;  // 20260402 ZJH 归一化模式标志
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
    // 20260402 ZJH 新增 bUseGroupNorm 参数，支持 GN/BN 双模式切换
    ASPPModule(int nInChannels, int nOutChannels = 256, bool bUseGroupNorm = false)
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
          m_dropout(0.5f),
          // 20260402 ZJH GroupNorm 实例（nGroups 自动调整）
          m_gn1(nOutChannels), m_gn6(nOutChannels), m_gn12(nOutChannels), m_gn18(nOutChannels),
          m_gnPool(nOutChannels), m_gnMerge(nOutChannels),
          m_bUseGroupNorm(bUseGroupNorm)  // 20260402 ZJH 记录归一化模式
    {}

    Tensor forward(const Tensor& input) override {
        auto ci = input.contiguous();
        int nBatch = ci.shape(0);
        int nH = ci.shape(2);
        int nW = ci.shape(3);

        // 20260402 ZJH 辅助 lambda: 选择 GN 或 BN 归一化层
        auto norm = [&](auto& gn, auto& bn) -> Module& { return m_bUseGroupNorm ? static_cast<Module&>(gn) : static_cast<Module&>(bn); };

        // 20260320 ZJH 分支 1：1x1 conv
        auto b1 = m_relu.forward(norm(m_gn1, m_bn1).forward(m_conv1x1.forward(ci)));

        // 20260320 ZJH 分支 2-4：膨胀卷积（自动 padding 保持尺寸）
        auto b2 = m_relu.forward(norm(m_gn6, m_bn6).forward(m_atrous6.forward(ci)));
        auto b3 = m_relu.forward(norm(m_gn12, m_bn12).forward(m_atrous12.forward(ci)));
        auto b4 = m_relu.forward(norm(m_gn18, m_bn18).forward(m_atrous18.forward(ci)));

        // 20260331 ZJH 分支 5：全局平均池化 → 1x1 conv → 双线性上采样到原始尺寸
        // GPU: avgPool2d(kernel=H, stride=H) → [B,C,1,1]，零 D2H
        // CPU: CPUBackend::globalAvgPool2d
        Tensor pooled;
        if (isCudaTensor(ci)) {
            pooled = Tensor::zeros({nBatch, m_nIn, 1, 1}, DeviceType::CUDA);
            CUDABackend::avgPool2d(ci.floatDataPtr(), pooled.mutableFloatDataPtr(),
                                    nBatch, m_nIn, nH, nW, nH, nW, nH, 0);
        } else {
            pooled = Tensor::zeros({nBatch, m_nIn, 1, 1});
            CPUBackend::globalAvgPool2d(ci.floatDataPtr(), pooled.mutableFloatDataPtr(),
                                         nBatch, m_nIn, nH, nW);
        }
        auto b5Conv = m_relu.forward(norm(m_gnPool, m_bnPool).forward(m_convPool.forward(pooled)));
        // 20260331 ZJH 双线性上采样 [B,C,1,1] → [B,C,nH,nW]（设备无关）
        auto b5 = tensorUpsampleBilinear(b5Conv, nH);

        // 20260331 ZJH 裁剪所有分支到最小 HW（膨胀卷积可能导致尺寸不同）
        int nMinH = std::min({b1.shape(2), b2.shape(2), b3.shape(2), b4.shape(2), b5.shape(2)});
        int nMinW = std::min({b1.shape(3), b2.shape(3), b3.shape(3), b4.shape(3), b5.shape(3)});

        // 20260331 ZJH 裁剪辅助 lambda: 若 HW 不等于 nMinH×nMinW 则 slice 到目标尺寸
        auto cropToMin = [&](Tensor t) -> Tensor {
            if (t.shape(2) != nMinH) t = tensorSlice(t, 2, 0, nMinH);
            if (t.shape(3) != nMinW) t = tensorSlice(t, 3, 0, nMinW);
            return t;
        };

        // 20260331 ZJH 5 分支通道拼接（设备无关，支持 autograd）
        auto concat = tensorConcatChannels(cropToMin(b1), cropToMin(b2));
        concat = tensorConcatChannels(concat, cropToMin(b3));
        concat = tensorConcatChannels(concat, cropToMin(b4));
        concat = tensorConcatChannels(concat, cropToMin(b5));

        // 20260320 ZJH 合并 1x1 + BN/GN + ReLU + Dropout
        auto merged = m_relu.forward(norm(m_gnMerge, m_bnMerge).forward(m_convMerge.forward(concat)));
        return m_dropout.forward(merged);
    }

    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> ps) { for (auto* p : ps) v.push_back(p); };
        append(m_conv1x1.parameters());
        append(m_bUseGroupNorm ? m_gn1.parameters() : m_bn1.parameters());
        append(m_atrous6.parameters());
        append(m_bUseGroupNorm ? m_gn6.parameters() : m_bn6.parameters());
        append(m_atrous12.parameters());
        append(m_bUseGroupNorm ? m_gn12.parameters() : m_bn12.parameters());
        append(m_atrous18.parameters());
        append(m_bUseGroupNorm ? m_gn18.parameters() : m_bn18.parameters());
        append(m_convPool.parameters());
        append(m_bUseGroupNorm ? m_gnPool.parameters() : m_bnPool.parameters());
        append(m_convMerge.parameters());
        append(m_bUseGroupNorm ? m_gnMerge.parameters() : m_bnMerge.parameters());
        return v;
    }

    // 20260331 ZJH 重写 namedParameters() 收集所有子层命名参数
    // 20260402 ZJH 支持 GN/BN 双模式
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };
        appendParams("conv1x1", m_conv1x1);    // 20260331 ZJH 1x1 分支
        if (m_bUseGroupNorm) appendParams("gn1", m_gn1);
        else                 appendParams("bn1", m_bn1);
        appendParams("atrous6", m_atrous6);     // 20260331 ZJH 膨胀率 6 分支
        if (m_bUseGroupNorm) appendParams("gn6", m_gn6);
        else                 appendParams("bn6", m_bn6);
        appendParams("atrous12", m_atrous12);   // 20260331 ZJH 膨胀率 12 分支
        if (m_bUseGroupNorm) appendParams("gn12", m_gn12);
        else                 appendParams("bn12", m_bn12);
        appendParams("atrous18", m_atrous18);   // 20260331 ZJH 膨胀率 18 分支
        if (m_bUseGroupNorm) appendParams("gn18", m_gn18);
        else                 appendParams("bn18", m_bn18);
        appendParams("convPool", m_convPool);   // 20260331 ZJH 全局池化分支
        if (m_bUseGroupNorm) appendParams("gnPool", m_gnPool);
        else                 appendParams("bnPool", m_bnPool);
        appendParams("convMerge", m_convMerge); // 20260331 ZJH 合并 1x1
        if (m_bUseGroupNorm) appendParams("gnMerge", m_gnMerge);
        else                 appendParams("bnMerge", m_bnMerge);
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    // 20260402 ZJH GN 没有 running stats，直接返回空
    std::vector<Tensor*> buffers() override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
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
    // 20260402 ZJH GN 模式下返回空
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
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
        m_dropout.train(b);
        // 20260402 ZJH 训练模式传播到当前使用的归一化层
        if (m_bUseGroupNorm) {
            m_gn1.train(b); m_gn6.train(b); m_gn12.train(b); m_gn18.train(b);
            m_gnPool.train(b); m_gnMerge.train(b);
        } else {
            m_bn1.train(b); m_bn6.train(b); m_bn12.train(b); m_bn18.train(b);
            m_bnPool.train(b); m_bnMerge.train(b);
        }
    }

private:
    int m_nIn, m_nOut;  // 20260406 ZJH 输入/输出通道数
    Conv2d m_conv1x1;  // 20260406 ZJH 分支1: 1x1 卷积
    BatchNorm2d m_bn1;  // 20260406 ZJH 分支1 BN
    DilatedConv2d m_atrous6, m_atrous12, m_atrous18;  // 20260406 ZJH 分支2-4: 膨胀卷积 (rate=6/12/18)
    BatchNorm2d m_bn6, m_bn12, m_bn18;  // 20260406 ZJH 分支2-4 BN
    Conv2d m_convPool;  // 20260406 ZJH 分支5: 全局池化后 1x1 卷积
    BatchNorm2d m_bnPool;  // 20260406 ZJH 分支5 BN
    Conv2d m_convMerge;  // 20260406 ZJH 合并 1x1 降维 (5*nOut → nOut)
    BatchNorm2d m_bnMerge;  // 20260406 ZJH 合并 BN
    Dropout2d m_dropout;  // 20260406 ZJH Dropout (p=0.5)
    // 20260402 ZJH GroupNorm 实例（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gn1, m_gn6, m_gn12, m_gn18;
    GroupNorm2d m_gnPool, m_gnMerge;
    bool m_bUseGroupNorm = false;  // 20260402 ZJH 归一化模式标志
    ReLU m_relu;
};

// 20260320 ZJH DeepLabV3 完整版
// ResNet 风格编码器（4 组残差块 + stride 16）→ ASPP → 解码器（低级特征融合 + 上采样）
class DeepLabV3 : public Module {
public:
    // 20260402 ZJH 新增 bUseGroupNorm 参数，支持 GN/BN 双模式切换
    DeepLabV3(int nInChannels = 1, int nNumClasses = 2, bool bUseGroupNorm = false)
        : m_nNumClasses(nNumClasses),
          // 20260320 ZJH ResNet 风格编码器
          m_stem(nInChannels, 64, 3, 1, 1, true), m_bnStem(64),
          // 20260402 ZJH 传递 bUseGroupNorm 给 ResBlock
          m_layer1_0(64, 64, 1, 0.1f, bUseGroupNorm), m_layer1_1(64, 64, 1, 0.1f, bUseGroupNorm),
          m_layer2_0(64, 128, 2, 0.1f, bUseGroupNorm), m_layer2_1(128, 128, 1, 0.1f, bUseGroupNorm),
          m_layer3_0(128, 256, 2, 0.1f, bUseGroupNorm), m_layer3_1(256, 256, 1, 0.1f, bUseGroupNorm),
          m_layer4_0(256, 512, 1, 0.2f, bUseGroupNorm), m_layer4_1(512, 512, 1, 0.2f, bUseGroupNorm),
          // 20260320 ZJH ASPP（20260402 ZJH 传递 bUseGroupNorm）
          m_aspp(512, 256, bUseGroupNorm),
          // 20260331 ZJH 解码器：低级特征改用 layer2 输出（128ch, H/2 分辨率）
          m_lowConv(128, 48, 1, 1, 0, true), m_bnLow(48),
          m_decConv1(256 + 48, 256, 3, 1, 1, true), m_bnDec1(256),
          m_decConv2(256, 256, 3, 1, 1, true), m_bnDec2(256),
          m_decDropout(0.5f),
          m_classifier(256, nNumClasses, 1, 1, 0, true),
          // 20260402 ZJH GroupNorm 实例（解码器直属 BN 的对应 GN）
          m_gnStem(64), m_gnLow(48), m_gnDec1(256), m_gnDec2(256),
          m_bUseGroupNorm(bUseGroupNorm)  // 20260402 ZJH 记录归一化模式
    {}

    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH 编码器
        // 20260402 ZJH stem 归一化根据 m_bUseGroupNorm 选择 GN 或 BN
        auto stemConv = m_stem.forward(input);  // 20260402 ZJH stem 卷积
        auto stemNorm = m_bUseGroupNorm ? m_gnStem.forward(stemConv) : m_bnStem.forward(stemConv);
        auto stem = m_relu.forward(stemNorm);  // [N,64,H,W]
        auto l1 = m_layer1_1.forward(m_layer1_0.forward(stem));  // [N,64,H,W] 低级特征
        auto l2 = m_layer2_1.forward(m_layer2_0.forward(l1));    // [N,128,H/2,W/2]
        auto l3 = m_layer3_1.forward(m_layer3_0.forward(l2));    // [N,256,H/4,W/4]
        auto l4 = m_layer4_1.forward(m_layer4_0.forward(l3));    // [N,512,H/4,W/4]

        // 20260320 ZJH ASPP
        auto asppOut = m_aspp.forward(l4);  // [N,256,H',W']
        int nAH = asppOut.shape(2);  // 20260331 ZJH ASPP 输出空间高度

        // 20260331 ZJH 上采样 ASPP 输出到低级特征尺寸（使用 l2 而非 l1）
        // l2 在 H/2 分辨率，decoder im2col 缓冲减少 4 倍（684MB→171MB）
        int nLowH = l2.shape(2);  // 20260331 ZJH 低级特征空间高度（H/2）

        // 20260331 ZJH 低级特征 1x1 降维（l2: 128ch → 48ch）
        // 20260402 ZJH 根据 m_bUseGroupNorm 选择 GN 或 BN
        auto lowConvOut = m_lowConv.forward(l2);  // 20260402 ZJH 低级特征 1x1 降维
        auto lowNorm = m_bUseGroupNorm ? m_gnLow.forward(lowConvOut) : m_bnLow.forward(lowConvOut);
        auto lowFeat = m_relu.forward(lowNorm);  // [N,48,H/2,W/2]

        // 20260331 ZJH 双线性上采样 ASPP 到低级特征尺寸（设备无关，支持 autograd）
        // ASPP 输出 [B,256,H/4,W/4]，低级特征 [B,48,H,W]，倍率 = nLowH / nAH
        int nUpsampleScale = std::max(1, nLowH / std::max(1, nAH));  // 20260331 ZJH 安全除法
        auto asppUp = tensorUpsampleBilinear(asppOut, nUpsampleScale);

        // 20260331 ZJH 拼接 ASPP + 低级特征（设备无关，支持 autograd）
        auto concat = tensorConcatChannels(asppUp, lowFeat);

        // 20260320 ZJH 解码器 3x3 conv x2 + Dropout
        // 20260402 ZJH 解码器归一化根据 m_bUseGroupNorm 选择 GN 或 BN
        auto decConv1Out = m_decConv1.forward(concat);  // 20260402 ZJH 解码器 conv1
        auto decNorm1 = m_bUseGroupNorm ? m_gnDec1.forward(decConv1Out) : m_bnDec1.forward(decConv1Out);
        auto dec = m_relu.forward(decNorm1);
        auto decConv2Out = m_decConv2.forward(dec);  // 20260402 ZJH 解码器 conv2
        auto decNorm2 = m_bUseGroupNorm ? m_gnDec2.forward(decConv2Out) : m_bnDec2.forward(decConv2Out);
        dec = m_relu.forward(decNorm2);
        dec = m_decDropout.forward(dec);

        // 20260320 ZJH 分类头 1x1 conv
        auto logits = m_classifier.forward(dec);  // [N, nClasses, H/2, W/2]

        // 20260331 ZJH 上采样到输入分辨率（低级特征在 H/2，需要 2x 上采样恢复）
        int nInH = input.shape(2);
        int nOutH = logits.shape(2);
        if (nOutH < nInH) {
            int nScale = nInH / nOutH;  // 20260331 ZJH 通常为 2
            logits = tensorUpsampleBilinear(logits, nScale);
        }
        return logits;
    }

    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> ps) { for (auto* p : ps) v.push_back(p); };
        append(m_stem.parameters());
        append(m_bUseGroupNorm ? m_gnStem.parameters() : m_bnStem.parameters());  // 20260402 ZJH stem 归一化
        // 20260402 ZJH ResBlock 内部已处理 GN/BN 切换
        append(m_layer1_0.parameters()); append(m_layer1_1.parameters());
        append(m_layer2_0.parameters()); append(m_layer2_1.parameters());
        append(m_layer3_0.parameters()); append(m_layer3_1.parameters());
        append(m_layer4_0.parameters()); append(m_layer4_1.parameters());
        append(m_aspp.parameters());  // 20260402 ZJH ASPP 内部已处理 GN/BN 切换
        append(m_lowConv.parameters());
        append(m_bUseGroupNorm ? m_gnLow.parameters() : m_bnLow.parameters());    // 20260402 ZJH 低级特征归一化
        append(m_decConv1.parameters());
        append(m_bUseGroupNorm ? m_gnDec1.parameters() : m_bnDec1.parameters());  // 20260402 ZJH 解码器1归一化
        append(m_decConv2.parameters());
        append(m_bUseGroupNorm ? m_gnDec2.parameters() : m_bnDec2.parameters());  // 20260402 ZJH 解码器2归一化
        append(m_classifier.parameters());
        return v;
    }

    // 20260331 ZJH 重写 namedParameters() — 完整层级命名
    // 20260402 ZJH 支持 GN/BN 双模式
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };
        // 20260331 ZJH 编码器 stem
        appendParams("stem", m_stem);
        // 20260402 ZJH stem 归一化命名
        if (m_bUseGroupNorm) appendParams("gnStem", m_gnStem);
        else                 appendParams("bnStem", m_bnStem);
        // 20260331 ZJH 编码器 layer1-4（ResBlock 内部已处理 GN/BN 命名）
        appendParams("layer1_0", m_layer1_0);  appendParams("layer1_1", m_layer1_1);
        appendParams("layer2_0", m_layer2_0);  appendParams("layer2_1", m_layer2_1);
        appendParams("layer3_0", m_layer3_0);  appendParams("layer3_1", m_layer3_1);
        appendParams("layer4_0", m_layer4_0);  appendParams("layer4_1", m_layer4_1);
        // 20260331 ZJH ASPP 模块（内部已处理 GN/BN 命名）
        appendParams("aspp", m_aspp);
        // 20260331 ZJH 解码器
        appendParams("lowConv", m_lowConv);
        if (m_bUseGroupNorm) appendParams("gnLow", m_gnLow);
        else                 appendParams("bnLow", m_bnLow);
        appendParams("decConv1", m_decConv1);
        if (m_bUseGroupNorm) appendParams("gnDec1", m_gnDec1);
        else                 appendParams("bnDec1", m_bnDec1);
        appendParams("decConv2", m_decConv2);
        if (m_bUseGroupNorm) appendParams("gnDec2", m_gnDec2);
        else                 appendParams("bnDec2", m_bnDec2);
        appendParams("classifier", m_classifier);  // 20260331 ZJH 分类头
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集所有 BN running stats
    // 20260402 ZJH GN 模式下直属 BN 无缓冲区，但 ResBlock/ASPP 子模块内部处理
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        // 20260402 ZJH 直属归一化层的缓冲区（GN 模式下无缓冲区）
        if (!m_bUseGroupNorm) {
            for (auto* p : m_bnStem.buffers()) v.push_back(p);
        }
        // 20260402 ZJH ResBlock/ASPP 子模块内部已处理 GN/BN 缓冲区
        for (auto* p : m_layer1_0.buffers()) v.push_back(p);
        for (auto* p : m_layer1_1.buffers()) v.push_back(p);
        for (auto* p : m_layer2_0.buffers()) v.push_back(p);
        for (auto* p : m_layer2_1.buffers()) v.push_back(p);
        for (auto* p : m_layer3_0.buffers()) v.push_back(p);
        for (auto* p : m_layer3_1.buffers()) v.push_back(p);
        for (auto* p : m_layer4_0.buffers()) v.push_back(p);
        for (auto* p : m_layer4_1.buffers()) v.push_back(p);
        for (auto* p : m_aspp.buffers()) v.push_back(p);
        if (!m_bUseGroupNorm) {
            for (auto* p : m_bnLow.buffers()) v.push_back(p);
            for (auto* p : m_bnDec1.buffers()) v.push_back(p);
            for (auto* p : m_bnDec2.buffers()) v.push_back(p);
        }
        return v;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集所有 BN 命名缓冲区
    // 20260402 ZJH 支持 GN/BN 双模式
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        if (!m_bUseGroupNorm) appendBufs("bnStem", m_bnStem);
        appendBufs("layer1_0", m_layer1_0);  appendBufs("layer1_1", m_layer1_1);
        appendBufs("layer2_0", m_layer2_0);  appendBufs("layer2_1", m_layer2_1);
        appendBufs("layer3_0", m_layer3_0);  appendBufs("layer3_1", m_layer3_1);
        appendBufs("layer4_0", m_layer4_0);  appendBufs("layer4_1", m_layer4_1);
        appendBufs("aspp", m_aspp);
        if (!m_bUseGroupNorm) {
            appendBufs("bnLow", m_bnLow);
            appendBufs("bnDec1", m_bnDec1);  appendBufs("bnDec2", m_bnDec2);
        }
        return vecResult;
    }

    void train(bool b = true) override {
        m_bTraining = b;
        // 20260402 ZJH 训练模式传播到当前使用的归一化层
        if (m_bUseGroupNorm) m_gnStem.train(b);
        else                 m_bnStem.train(b);
        m_layer1_0.train(b); m_layer1_1.train(b);
        m_layer2_0.train(b); m_layer2_1.train(b);
        m_layer3_0.train(b); m_layer3_1.train(b);
        m_layer4_0.train(b); m_layer4_1.train(b);
        m_aspp.train(b);
        if (m_bUseGroupNorm) {
            m_gnLow.train(b); m_gnDec1.train(b); m_gnDec2.train(b);
        } else {
            m_bnLow.train(b); m_bnDec1.train(b); m_bnDec2.train(b);
        }
        m_decDropout.train(b);
    }

private:
    int m_nNumClasses;  // 20260406 ZJH 输出分割类别数
    // 20260406 ZJH ResNet 风格编码器
    Conv2d m_stem; BatchNorm2d m_bnStem;  // 20260406 ZJH Stem 3x3 卷积 + BN
    ResBlock m_layer1_0, m_layer1_1;  // 20260406 ZJH 编码器层1: 64→64 (stride=1)
    ResBlock m_layer2_0, m_layer2_1;  // 20260406 ZJH 编码器层2: 64→128 (stride=2)
    ResBlock m_layer3_0, m_layer3_1;  // 20260406 ZJH 编码器层3: 128→256 (stride=2)
    ResBlock m_layer4_0, m_layer4_1;  // 20260406 ZJH 编码器层4: 256→512 (stride=1, 保持 H/4)
    ASPPModule m_aspp;  // 20260406 ZJH ASPP 空洞空间金字塔池化模块
    // 20260406 ZJH 解码器: 低级特征融合 + 3x3 conv + 分类头
    Conv2d m_lowConv; BatchNorm2d m_bnLow;  // 20260406 ZJH 低级特征 1x1 降维 (128→48)
    Conv2d m_decConv1; BatchNorm2d m_bnDec1;  // 20260406 ZJH 解码器 conv1 (304→256)
    Conv2d m_decConv2; BatchNorm2d m_bnDec2;  // 20260406 ZJH 解码器 conv2 (256→256)
    Dropout2d m_decDropout;  // 20260406 ZJH 解码器 Dropout (p=0.5)
    Conv2d m_classifier;  // 20260406 ZJH 分类头 1x1 Conv (256→nClasses)
    // 20260402 ZJH GroupNorm 实例（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gnStem;   // 20260402 ZJH stem GN (64ch)
    GroupNorm2d m_gnLow;    // 20260402 ZJH 低级特征 GN (48ch)
    GroupNorm2d m_gnDec1;   // 20260402 ZJH 解码器1 GN (256ch)
    GroupNorm2d m_gnDec2;   // 20260402 ZJH 解码器2 GN (256ch)
    bool m_bUseGroupNorm = false;  // 20260402 ZJH 归一化模式标志
    ReLU m_relu;
};

// =========================================================
// SegNet 完整版 — 对称编码器-解码器 + BN
// =========================================================

// 20260406 ZJH SegNet — 对称编码器-解码器分割网络
// 编码器: 4 组 Conv-BN-ReLU + MaxPool 逐步下采样
// 解码器: 4 组 ConvTranspose(上采样) + Conv-BN-ReLU 逐步恢复分辨率
// 最终: 1x1 Conv 分类头输出逐像素类别 logits
// 输入: [N, nInChannels, H, W] -> 输出: [N, nNumClasses, H, W]
class SegNet : public Module {
public:
    // 20260406 ZJH 构造函数
    // nInChannels: 输入通道数（默认 1 灰度图）
    // nNumClasses: 输出类别数（默认 2，含背景）
    SegNet(int nInChannels = 1, int nNumClasses = 2)
        : m_nNumClasses(nNumClasses),
          // 20260406 ZJH 编码器组1: nIn→64, 保持分辨率后 2x 下采样
          m_enc1a(nInChannels, 64, 3, 1, 1, true), m_bn1a(64),
          m_enc1b(64, 64, 3, 1, 1, true), m_bn1b(64), m_pool1(2),
          // 20260406 ZJH 编码器组2: 64→128, 2x 下采样
          m_enc2a(64, 128, 3, 1, 1, true), m_bn2a(128),
          m_enc2b(128, 128, 3, 1, 1, true), m_bn2b(128), m_pool2(2),
          // 20260406 ZJH 编码器组3: 128→256, 2x 下采样
          m_enc3a(128, 256, 3, 1, 1, true), m_bn3a(256),
          m_enc3b(256, 256, 3, 1, 1, true), m_bn3b(256), m_pool3(2),
          // 20260406 ZJH 编码器组4: 256→512, 2x 下采样（总下采样 16 倍）
          m_enc4a(256, 512, 3, 1, 1, true), m_bn4a(512),
          m_enc4b(512, 512, 3, 1, 1, true), m_bn4b(512), m_pool4(2),
          // 20260406 ZJH 解码器组4: ConvTranspose 2x 上采样 + Conv 512→256
          m_dec4a(512, 512, 4, 2, 1, true), m_dbn4a(512),
          m_dec4b(512, 256, 3, 1, 1, true), m_dbn4b(256),
          // 20260406 ZJH 解码器组3: 2x 上采样 + Conv 256→128
          m_dec3a(256, 256, 4, 2, 1, true), m_dbn3a(256),
          m_dec3b(256, 128, 3, 1, 1, true), m_dbn3b(128),
          // 20260406 ZJH 解码器组2: 2x 上采样 + Conv 128→64
          m_dec2a(128, 128, 4, 2, 1, true), m_dbn2a(128),
          m_dec2b(128, 64, 3, 1, 1, true), m_dbn2b(64),
          // 20260406 ZJH 解码器组1: 2x 上采样恢复到原始分辨率
          m_dec1a(64, 64, 4, 2, 1, true), m_dbn1a(64),
          // 20260406 ZJH 分类头: 1x1 Conv 输出 nClasses 通道
          m_classifier(64, nNumClasses, 1, 1, 0, true)
    {}

    // 20260406 ZJH forward — SegNet 前向传播
    // input: [N, nInChannels, H, W]（H, W 应为 16 的倍数）
    // 返回: [N, nNumClasses, H, W] 逐像素分类 logits
    Tensor forward(const Tensor& input) override {
        // 20260406 ZJH 编码器组1: [N,Cin,H,W] → [N,64,H/2,W/2]
        auto h = m_relu.forward(m_bn1a.forward(m_enc1a.forward(input)));
        h = m_relu.forward(m_bn1b.forward(m_enc1b.forward(h)));
        h = m_pool1.forward(h);  // 20260406 ZJH 2x 下采样
        // 20260406 ZJH 编码器组2: [N,64,H/2,W/2] → [N,128,H/4,W/4]
        h = m_relu.forward(m_bn2a.forward(m_enc2a.forward(h)));
        h = m_relu.forward(m_bn2b.forward(m_enc2b.forward(h)));
        h = m_pool2.forward(h);  // 20260406 ZJH 2x 下采样
        // 20260406 ZJH 编码器组3: [N,128,H/4,W/4] → [N,256,H/8,W/8]
        h = m_relu.forward(m_bn3a.forward(m_enc3a.forward(h)));
        h = m_relu.forward(m_bn3b.forward(m_enc3b.forward(h)));
        h = m_pool3.forward(h);  // 20260406 ZJH 2x 下采样
        // 20260406 ZJH 编码器组4: [N,256,H/8,W/8] → [N,512,H/16,W/16]
        h = m_relu.forward(m_bn4a.forward(m_enc4a.forward(h)));
        h = m_relu.forward(m_bn4b.forward(m_enc4b.forward(h)));
        h = m_pool4.forward(h);  // 20260406 ZJH 2x 下采样
        // 20260406 ZJH 解码器组4: [N,512,H/16,W/16] → [N,256,H/8,W/8]
        h = m_relu.forward(m_dbn4a.forward(m_dec4a.forward(h)));  // 20260406 ZJH ConvTranspose 2x 上采样
        h = m_relu.forward(m_dbn4b.forward(m_dec4b.forward(h)));  // 20260406 ZJH Conv 512→256
        // 20260406 ZJH 解码器组3: [N,256,H/8,W/8] → [N,128,H/4,W/4]
        h = m_relu.forward(m_dbn3a.forward(m_dec3a.forward(h)));
        h = m_relu.forward(m_dbn3b.forward(m_dec3b.forward(h)));
        // 20260406 ZJH 解码器组2: [N,128,H/4,W/4] → [N,64,H/2,W/2]
        h = m_relu.forward(m_dbn2a.forward(m_dec2a.forward(h)));
        h = m_relu.forward(m_dbn2b.forward(m_dec2b.forward(h)));
        // 20260406 ZJH 解码器组1: [N,64,H/2,W/2] → [N,64,H,W]
        h = m_relu.forward(m_dbn1a.forward(m_dec1a.forward(h)));
        // 20260406 ZJH 分类头: [N,64,H,W] → [N,nClasses,H,W]
        return m_classifier.forward(h);  // 20260406 ZJH 返回逐像素分类 logits
    }

    // 20260406 ZJH 重写 parameters() 收集编码器+解码器+分类头所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        // 20260406 ZJH 编码器组1-4 卷积 + BN 参数
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
        // 20260406 ZJH 解码器组4-1 反卷积/卷积 + BN 参数
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
        // 20260406 ZJH 分类头参数
        for (auto* p : m_classifier.parameters()) v.push_back(p);
        return v;  // 20260406 ZJH 返回所有可训练参数指针
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
    int m_nNumClasses;  // 20260406 ZJH 输出分割类别数
    // 20260406 ZJH 编码器: 4 组 Conv-BN-ReLU + MaxPool
    Conv2d m_enc1a, m_enc1b, m_enc2a, m_enc2b, m_enc3a, m_enc3b, m_enc4a, m_enc4b;  // 20260406 ZJH 编码器卷积层
    BatchNorm2d m_bn1a, m_bn1b, m_bn2a, m_bn2b, m_bn3a, m_bn3b, m_bn4a, m_bn4b;  // 20260406 ZJH 编码器 BN 层
    MaxPool2d m_pool1, m_pool2, m_pool3, m_pool4;  // 20260406 ZJH 各组的 2x2 最大池化
    // 20260406 ZJH 解码器: ConvTranspose(上采样) + Conv-BN-ReLU
    ConvTranspose2d m_dec4a, m_dec3a, m_dec2a, m_dec1a;  // 20260406 ZJH 反卷积上采样层（4组各 1 个）
    Conv2d m_dec4b, m_dec3b, m_dec2b;  // 20260406 ZJH 解码器 3x3 卷积层（组4/3/2 各 1 个）
    BatchNorm2d m_dbn4a, m_dbn4b, m_dbn3a, m_dbn3b, m_dbn2a, m_dbn2b, m_dbn1a;  // 20260406 ZJH 解码器 BN 层
    Conv2d m_classifier;  // 20260406 ZJH 1x1 分类头卷积
    ReLU m_relu;  // 20260406 ZJH 共用 ReLU 激活
};

// =========================================================
// FCN-8s 完整版 — VGG 风格编码器 + 跳跃融合
// =========================================================

// 20260406 ZJH FCN-8s — 全卷积网络（VGG 编码器 + 跳跃融合）
// 编码器: VGG-16 风格 5 组 Conv-BN-ReLU + Pool
// FC 替代: 1x1 Conv 替代全连接层
// 跳跃融合: score_7(2x) + pool4 → (2x) + pool3 → (8x) 恢复原始分辨率
// 输入: [N, nInChannels, H, W] → 输出: [N, nNumClasses, H, W]
class FCN8s : public Module {
public:
    // 20260406 ZJH 构造函数
    // nInChannels: 输入通道数（默认 1 灰度图）
    // nNumClasses: 输出类别数（默认 2，含背景）
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

    // 20260406 ZJH forward — FCN-8s 前向传播
    // input: [N, nInChannels, H, W]
    // 返回: [N, nNumClasses, H, W] 逐像素分类 logits
    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);  // 20260406 ZJH 批次大小

        // 20260406 ZJH VGG 编码器组1: [N,Cin,H,W] → [N,64,H/2,W/2]
        auto h = m_relu.forward(m_b1a.forward(m_c1a.forward(input)));
        h = m_relu.forward(m_b1b.forward(m_c1b.forward(h)));
        h = m_p1.forward(h);  // 20260406 ZJH pool1: 2x 下采样
        // 20260406 ZJH VGG 编码器组2: [N,64,H/2,W/2] → [N,128,H/4,W/4]
        h = m_relu.forward(m_b2a.forward(m_c2a.forward(h)));
        h = m_relu.forward(m_b2b.forward(m_c2b.forward(h)));
        h = m_p2.forward(h);  // 20260406 ZJH pool2: 2x 下采样
        // 20260406 ZJH VGG 编码器组3: [N,128,H/4,W/4] → [N,256,H/8,W/8]
        h = m_relu.forward(m_b3a.forward(m_c3a.forward(h)));
        h = m_relu.forward(m_b3b.forward(m_c3b.forward(h)));
        auto pool3 = m_p3.forward(h);  // 20260320 ZJH 保存 pool3 用于跳跃连接
        // 20260406 ZJH VGG 编码器组4: [N,256,H/8,W/8] → [N,512,H/16,W/16]
        h = m_relu.forward(m_b4a.forward(m_c4a.forward(pool3)));
        h = m_relu.forward(m_b4b.forward(m_c4b.forward(h)));
        auto pool4 = m_p4.forward(h);  // 20260320 ZJH 保存 pool4
        // 20260406 ZJH VGG 编码器组5: [N,512,H/16,W/16] → [N,512,H/32,W/32]
        h = m_relu.forward(m_b5a.forward(m_c5a.forward(pool4)));
        h = m_relu.forward(m_b5b.forward(m_c5b.forward(h)));
        h = m_p5.forward(h);  // 20260406 ZJH pool5: 2x 下采样

        // 20260320 ZJH FC 替代层
        h = m_relu.forward(m_fc6.forward(h));
        h = m_drop6.forward(h);
        h = m_relu.forward(m_fc7.forward(h));
        h = m_drop7.forward(h);

        // 20260320 ZJH FCN-8s 跳跃融合
        auto s7 = m_score7.forward(h);
        auto up7 = m_up7.forward(s7);              // 2x 上采样到 pool4 尺寸

        // 20260320 ZJH score_pool4 + up7 逐元素加（第一次跳跃融合）
        auto s4 = m_score4.forward(pool4);  // 20260406 ZJH pool4 投影到 nClasses 通道
        int nMinH4 = std::min(up7.shape(2), s4.shape(2));  // 20260406 ZJH 取最小高度（防止尺寸不匹配）
        int nMinW4 = std::min(up7.shape(3), s4.shape(3));  // 20260406 ZJH 取最小宽度
        auto fuse4 = Tensor::zeros({nBatch, m_nNumClasses, nMinH4, nMinW4});  // 20260406 ZJH 融合结果张量
        {
            auto cu = up7.contiguous(); auto cs = s4.contiguous();  // 20260406 ZJH 确保连续存储
            float* pF = fuse4.mutableFloatDataPtr();  // 20260406 ZJH 融合结果指针
            const float* pU = cu.floatDataPtr(); const float* pS = cs.floatDataPtr();  // 20260406 ZJH 上采样/score 数据指针
            // 20260406 ZJH 逐像素逐通道相加: fuse4 = up7 + score_pool4
            for (int n = 0; n < nBatch; ++n)
            for (int c = 0; c < m_nNumClasses; ++c)
            for (int hh = 0; hh < nMinH4; ++hh)
            for (int ww = 0; ww < nMinW4; ++ww) {
                int idxF = ((n * m_nNumClasses + c) * nMinH4 + hh) * nMinW4 + ww;  // 20260406 ZJH 融合张量线性索引
                pF[idxF] = pU[((n * m_nNumClasses + c) * up7.shape(2) + hh) * up7.shape(3) + ww]  // 20260406 ZJH up7 值
                         + pS[((n * m_nNumClasses + c) * s4.shape(2) + hh) * s4.shape(3) + ww];  // 20260406 ZJH score_pool4 值
            }
        }

        auto up4 = m_up4.forward(fuse4);  // 20260406 ZJH 2x 上采样到 pool3 尺寸

        // 20260320 ZJH score_pool3 + up4 逐元素加（第二次跳跃融合）
        auto s3 = m_score3.forward(pool3);  // 20260406 ZJH pool3 投影到 nClasses 通道
        int nMinH3 = std::min(up4.shape(2), s3.shape(2));  // 20260406 ZJH 取最小高度
        int nMinW3 = std::min(up4.shape(3), s3.shape(3));  // 20260406 ZJH 取最小宽度
        auto fuse3 = Tensor::zeros({nBatch, m_nNumClasses, nMinH3, nMinW3});  // 20260406 ZJH 融合结果张量
        {
            auto cu = up4.contiguous(); auto cs = s3.contiguous();  // 20260406 ZJH 确保连续存储
            float* pF = fuse3.mutableFloatDataPtr();  // 20260406 ZJH 融合结果指针
            const float* pU = cu.floatDataPtr(); const float* pS = cs.floatDataPtr();  // 20260406 ZJH 上采样/score 数据指针
            // 20260406 ZJH 逐像素逐通道相加: fuse3 = up4 + score_pool3
            for (int n = 0; n < nBatch; ++n)
            for (int c = 0; c < m_nNumClasses; ++c)
            for (int hh = 0; hh < nMinH3; ++hh)
            for (int ww = 0; ww < nMinW3; ++ww) {
                int idxF = ((n * m_nNumClasses + c) * nMinH3 + hh) * nMinW3 + ww;  // 20260406 ZJH 融合张量线性索引
                pF[idxF] = pU[((n * m_nNumClasses + c) * up4.shape(2) + hh) * up4.shape(3) + ww]  // 20260406 ZJH up4 值
                         + pS[((n * m_nNumClasses + c) * s3.shape(2) + hh) * s3.shape(3) + ww];  // 20260406 ZJH score_pool3 值
            }
        }

        // 20260320 ZJH 8x 上采样到原始分辨率
        return m_up3.forward(fuse3);
    }

    // 20260406 ZJH 重写 parameters() 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        // 20260406 ZJH VGG 编码器组1-5 卷积 + BN 参数
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
        // 20260406 ZJH FC 替代层参数
        for (auto* p : m_fc6.parameters()) v.push_back(p);
        for (auto* p : m_fc7.parameters()) v.push_back(p);
        // 20260406 ZJH Score 层 + 上采样层参数
        for (auto* p : m_score7.parameters()) v.push_back(p);
        for (auto* p : m_score4.parameters()) v.push_back(p);
        for (auto* p : m_score3.parameters()) v.push_back(p);
        for (auto* p : m_up7.parameters()) v.push_back(p);
        for (auto* p : m_up4.parameters()) v.push_back(p);
        for (auto* p : m_up3.parameters()) v.push_back(p);
        return v;  // 20260406 ZJH 返回所有可训练参数指针
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
    int m_nNumClasses;  // 20260406 ZJH 输出分割类别数
    // 20260406 ZJH VGG 风格编码器: 5 组 Conv-BN-ReLU + Pool
    Conv2d m_c1a, m_c1b, m_c2a, m_c2b, m_c3a, m_c3b, m_c4a, m_c4b, m_c5a, m_c5b;  // 20260406 ZJH VGG 卷积层
    BatchNorm2d m_b1a, m_b1b, m_b2a, m_b2b, m_b3a, m_b3b, m_b4a, m_b4b, m_b5a, m_b5b;  // 20260406 ZJH VGG BN 层
    MaxPool2d m_p1, m_p2, m_p3, m_p4, m_p5;  // 20260406 ZJH 各组的 2x2 最大池化
    // 20260406 ZJH FC 替代层: 1x1 Conv 替代原始 VGG 的全连接层
    Conv2d m_fc6, m_fc7;  // 20260406 ZJH fc6: 512→4096, fc7: 4096→4096
    // 20260406 ZJH Score 层: 将特征投影到 nClasses 通道
    Conv2d m_score7, m_score4, m_score3;  // 20260406 ZJH 分别对 fc7/pool4/pool3 的 score 投影
    // 20260406 ZJH 上采样层: ConvTranspose 逐步恢复分辨率
    ConvTranspose2d m_up7, m_up4, m_up3;  // 20260406 ZJH up7: 2x, up4: 2x, up3: 8x 上采样
    Dropout2d m_drop6, m_drop7;  // 20260406 ZJH fc6/fc7 后的 Dropout（p=0.5）
    ReLU m_relu;  // 20260406 ZJH 共用 ReLU 激活
};

// =========================================================
// MobileSegNet — 轻量级工业分割网络（对标海康 ASI_SEG 4.7MB）
// MobileNetV4-Small 编码器 + ASPP-Lite(3分支) + 轻量解码器
// 总参数量 ~1.75M ≈ 7MB FP32 / 1.75MB INT8
// =========================================================

// 20260401 ZJH ASPPLite — 3 分支轻量 ASPP 模块
// 比标准 ASPPModule(5分支) 少 40% 参数量，适合轻量骨干网络
// 分支1: 1x1 conv, 分支2: 3x3 dilation=6, 分支3: 全局平均池化
class ASPPLite : public Module {
public:
    // 20260401 ZJH 构造函数
    // nInChannels: 输入通道数（来自编码器最后一层）
    // nMidChannels: 每分支输出通道数（3分支拼接后 = nMidChannels * 3）
    // nOutChannels: 合并后输出通道数
    // 20260402 ZJH 新增 bUseGroupNorm 参数，支持 GN/BN 双模式切换
    ASPPLite(int nInChannels, int nMidChannels = 48, int nOutChannels = 96, bool bUseGroupNorm = false)
        : m_nIn(nInChannels), m_nMid(nMidChannels), m_nOut(nOutChannels),
          // 20260401 ZJH 分支1: 1x1 标准卷积（降维）
          m_conv1x1(nInChannels, nMidChannels, 1, 1, 0, true), m_bn1(nMidChannels),
          // 20260401 ZJH 分支2: 3x3 膨胀卷积 dilation=6（中等感受野）
          m_atrous6(nInChannels, nMidChannels, 3, 1, 6, 6, 1, true), m_bn6(nMidChannels),
          // 20260401 ZJH 分支3: 全局平均池化 → 1x1 conv（全局上下文）
          m_convPool(nInChannels, nMidChannels, 1, 1, 0, true), m_bnPool(nMidChannels),
          // 20260401 ZJH 合并: 3分支拼接后 1x1 降维
          m_convMerge(nMidChannels * 3, nOutChannels, 1, 1, 0, true), m_bnMerge(nOutChannels),
          m_dropout(0.3f),  // 20260401 ZJH 轻量网络用较低 dropout
          // 20260402 ZJH GroupNorm 实例（nGroups 自动调整）
          m_gn1(nMidChannels), m_gn6(nMidChannels),
          m_gnPool(nMidChannels), m_gnMerge(nOutChannels),
          m_bUseGroupNorm(bUseGroupNorm)  // 20260402 ZJH 记录归一化模式
    {}

    Tensor forward(const Tensor& input) override {
        auto ci = input.contiguous();
        int nBatch = ci.shape(0);  // 20260401 ZJH 批大小
        int nH = ci.shape(2);      // 20260401 ZJH 空间高度
        int nW = ci.shape(3);      // 20260401 ZJH 空间宽度

        // 20260402 ZJH 辅助 lambda: 选择 GN 或 BN 归一化层
        auto norm = [&](auto& gn, auto& bn) -> Module& { return m_bUseGroupNorm ? static_cast<Module&>(gn) : static_cast<Module&>(bn); };

        // 20260401 ZJH 分支1: 1x1 conv + BN/GN + ReLU（局部特征）
        auto b1 = m_relu.forward(norm(m_gn1, m_bn1).forward(m_conv1x1.forward(ci)));

        // 20260401 ZJH 分支2: 3x3 dilation=6 + BN/GN + ReLU（中等感受野）
        auto b2 = m_relu.forward(norm(m_gn6, m_bn6).forward(m_atrous6.forward(ci)));

        // 20260401 ZJH 分支3: 全局平均池化 → 1x1 conv → 上采样到输入尺寸
        Tensor pooled;
        if (isCudaTensor(ci)) {
            pooled = Tensor::zeros({nBatch, m_nIn, 1, 1}, DeviceType::CUDA);
            CUDABackend::avgPool2d(ci.floatDataPtr(), pooled.mutableFloatDataPtr(),
                                    nBatch, m_nIn, nH, nW, nH, nW, nH, 0);
        } else {
            pooled = Tensor::zeros({nBatch, m_nIn, 1, 1});
            CPUBackend::globalAvgPool2d(ci.floatDataPtr(), pooled.mutableFloatDataPtr(),
                                         nBatch, m_nIn, nH, nW);
        }
        auto b3Conv = m_relu.forward(norm(m_gnPool, m_bnPool).forward(m_convPool.forward(pooled)));
        auto b3 = tensorUpsampleBilinear(b3Conv, nH);  // 20260401 ZJH [B,C,1,1] → [B,C,H,W]

        // 20260401 ZJH 裁剪到最小 HW（膨胀卷积可能导致尺寸略有差异）
        int nMinH = std::min({b1.shape(2), b2.shape(2), b3.shape(2)});
        int nMinW = std::min({b1.shape(3), b2.shape(3), b3.shape(3)});
        auto cropToMin = [&](Tensor t) -> Tensor {
            if (t.shape(2) != nMinH) t = tensorSlice(t, 2, 0, nMinH);
            if (t.shape(3) != nMinW) t = tensorSlice(t, 3, 0, nMinW);
            return t;
        };

        // 20260401 ZJH 3 分支通道拼接 [B, nMid*3, H, W]
        auto concat = tensorConcatChannels(cropToMin(b1), cropToMin(b2));
        concat = tensorConcatChannels(concat, cropToMin(b3));

        // 20260401 ZJH 合并 1x1 + BN/GN + ReLU + Dropout
        auto merged = m_relu.forward(norm(m_gnMerge, m_bnMerge).forward(m_convMerge.forward(concat)));
        return m_dropout.forward(merged);
    }

    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> ps) { for (auto* p : ps) v.push_back(p); };
        append(m_conv1x1.parameters());
        append(m_bUseGroupNorm ? m_gn1.parameters() : m_bn1.parameters());
        append(m_atrous6.parameters());
        append(m_bUseGroupNorm ? m_gn6.parameters() : m_bn6.parameters());
        append(m_convPool.parameters());
        append(m_bUseGroupNorm ? m_gnPool.parameters() : m_bnPool.parameters());
        append(m_convMerge.parameters());
        append(m_bUseGroupNorm ? m_gnMerge.parameters() : m_bnMerge.parameters());
        return v;
    }

    // 20260401 ZJH 重写 namedParameters()
    // 20260402 ZJH 支持 GN/BN 双模式
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };
        appendParams("conv1x1", m_conv1x1);    // 20260401 ZJH 1x1 分支
        if (m_bUseGroupNorm) appendParams("gn1", m_gn1);
        else                 appendParams("bn1", m_bn1);
        appendParams("atrous6", m_atrous6);     // 20260401 ZJH 膨胀率 6 分支
        if (m_bUseGroupNorm) appendParams("gn6", m_gn6);
        else                 appendParams("bn6", m_bn6);
        appendParams("convPool", m_convPool);   // 20260401 ZJH 全局池化分支
        if (m_bUseGroupNorm) appendParams("gnPool", m_gnPool);
        else                 appendParams("bnPool", m_bnPool);
        appendParams("convMerge", m_convMerge); // 20260401 ZJH 合并 1x1
        if (m_bUseGroupNorm) appendParams("gnMerge", m_gnMerge);
        else                 appendParams("bnMerge", m_bnMerge);
        return vecResult;
    }

    // 20260401 ZJH 收集 BN running stats
    // 20260402 ZJH GN 没有 running stats，直接返回空
    std::vector<Tensor*> buffers() override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
        std::vector<Tensor*> v;
        for (auto* p : m_bn1.buffers()) v.push_back(p);
        for (auto* p : m_bn6.buffers()) v.push_back(p);
        for (auto* p : m_bnPool.buffers()) v.push_back(p);
        for (auto* p : m_bnMerge.buffers()) v.push_back(p);
        return v;
    }

    // 20260401 ZJH 收集 BN 命名缓冲区
    // 20260402 ZJH GN 模式下返回空
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        if (m_bUseGroupNorm) return {};  // 20260402 ZJH GN 没有 running stats
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("bn1", m_bn1);
        appendBufs("bn6", m_bn6);
        appendBufs("bnPool", m_bnPool);
        appendBufs("bnMerge", m_bnMerge);
        return vecResult;
    }

    void train(bool b = true) override {
        m_bTraining = b;
        m_dropout.train(b);
        // 20260402 ZJH 训练模式传播到当前使用的归一化层
        if (m_bUseGroupNorm) {
            m_gn1.train(b); m_gn6.train(b);
            m_gnPool.train(b); m_gnMerge.train(b);
        } else {
            m_bn1.train(b); m_bn6.train(b);
            m_bnPool.train(b); m_bnMerge.train(b);
        }
    }

private:
    int m_nIn, m_nMid, m_nOut;  // 20260406 ZJH 输入/中间/输出通道数
    Conv2d m_conv1x1;  BatchNorm2d m_bn1;              // 20260401 ZJH 分支1: 1x1
    DilatedConv2d m_atrous6;  BatchNorm2d m_bn6;       // 20260401 ZJH 分支2: dilation=6
    Conv2d m_convPool;  BatchNorm2d m_bnPool;           // 20260401 ZJH 分支3: GAP
    Conv2d m_convMerge;  BatchNorm2d m_bnMerge;         // 20260401 ZJH 合并层
    Dropout2d m_dropout;
    // 20260402 ZJH GroupNorm 实例（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gn1, m_gn6;        // 20260402 ZJH 分支1/2 GN
    GroupNorm2d m_gnPool, m_gnMerge; // 20260402 ZJH 池化分支/合并层 GN
    bool m_bUseGroupNorm = false;    // 20260402 ZJH 归一化模式标志
    ReLU m_relu;
};

// =========================================================
// SEBlock — Squeeze-and-Excitation 通道注意力（超越海康 ASI_SEG 的独有模块）
// 海康 ASI_SEG 是纯 CNN 无注意力机制
// SE 让网络学会"关注哪些通道更重要"，对缺陷检测尤其有效
// 参数开销 < 5%，精度提升 2~3%
// =========================================================

// 20260401 ZJH SEBlock — Squeeze-and-Excitation 通道注意力模块
// 结构: GlobalAvgPool → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → 通道加权
// 参数量: 2 × C × C/r（reduction=4 时仅增加 ~3% 参数）
class SEBlock : public Module {
public:
    // 20260401 ZJH 构造函数
    // nChannels: 输入/输出通道数
    // nReduction: 压缩比（默认 4，即中间层通道数 = C/4）
    SEBlock(int nChannels, int nReduction = 4)
        : m_nChannels(nChannels),
          m_nMid(std::max(1, nChannels / nReduction)),
          // 20260401 ZJH FC1: 压缩（C → C/r），用 1x1 Conv 实现（等效于逐像素 FC）
          m_fc1(nChannels, std::max(1, nChannels / nReduction), 1, 1, 0, true),
          // 20260401 ZJH FC2: 恢复（C/r → C）
          m_fc2(std::max(1, nChannels / nReduction), nChannels, 1, 1, 0, true)
    {}

    // 20260401 ZJH 前向传播: input × sigmoid(FC2(ReLU(FC1(GAP(input)))))
    Tensor forward(const Tensor& input) override {
        int nB = input.shape(0);    // 20260401 ZJH 批大小
        int nC = input.shape(1);    // 20260401 ZJH 通道数
        int nH = input.shape(2);    // 20260401 ZJH 空间高度
        int nW = input.shape(3);    // 20260401 ZJH 空间宽度

        // 20260401 ZJH Squeeze: 全局平均池化 [B,C,H,W] → [B,C,1,1]
        auto pooled = tensorAdaptiveAvgPool2d(input, 1, 1);  // 20260401 ZJH 自适应平均池化到 1x1

        // 20260401 ZJH Excitation: FC1 → ReLU → FC2 → Sigmoid
        auto se = m_relu.forward(m_fc1.forward(pooled));    // 20260401 ZJH [B, C/r, 1, 1]
        se = tensorSigmoid(m_fc2.forward(se));              // 20260401 ZJH [B, C, 1, 1] 通道权重

        // 20260401 ZJH Scale: 通道加权 input × se（广播乘法 [B,C,H,W] × [B,C,1,1]）
        // 使用 tensorUpsampleBilinear 将 [B,C,1,1] 扩展到 [B,C,H,W] 再逐元素乘
        auto seExpanded = tensorUpsampleBilinear(se, nH);   // 20260401 ZJH [B,C,1,1] → [B,C,H,H]
        return tensorMul(input, seExpanded);                 // 20260401 ZJH 通道重标定
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        for (auto* p : m_fc1.parameters()) v.push_back(p);
        for (auto* p : m_fc2.parameters()) v.push_back(p);
        return v;
    }

    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };
        appendParams("fc1", m_fc1);
        appendParams("fc2", m_fc2);
        return vecResult;
    }

    std::vector<Tensor*> buffers() override { return {}; }
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& = "") override { return {}; }

    void train(bool b = true) override { m_bTraining = b; }

private:
    int m_nChannels;     // 20260401 ZJH 输入通道数
    int m_nMid;          // 20260401 ZJH 中间层通道数（C/reduction）
    Conv2d m_fc1;        // 20260401 ZJH 压缩层（1x1 conv 等效 FC）
    Conv2d m_fc2;        // 20260401 ZJH 恢复层（1x1 conv 等效 FC）
    ReLU m_relu;
};

// 20260401 ZJH MobileSegNet — 轻量级分割网络主体
// 编码器: MobileNetV4-Small 风格（InvertedResidual 块，到 Stage5 截止）
// 中间层: ASPPLite（3 分支）
// 解码器: 低级特征融合 + 双线性上采样到原图
// 特点: 参数量 ~1.75M，推理速度约为 DeepLabV3 的 3-5 倍
class MobileSegNet : public Module {
public:
    // 20260401 ZJH 构造函数
    // nInChannels: 输入图像通道数（1=灰度, 3=RGB）
    // nNumClasses: 分割类别数（含背景）
    // 20260402 ZJH 新增 bUseGroupNorm 参数，支持 GN/BN 双模式切换
    MobileSegNet(int nInChannels = 1, int nNumClasses = 2, bool bUseGroupNorm = false)
        : m_nNumClasses(nNumClasses),
          // ===== 编码器: MobileNet 风格倒残差块 =====
          // 20260401 ZJH Stem: 标准 3x3 conv（非深度可分离）降维到 16ch
          m_stem(nInChannels, 16, 3, 2, 1, true), m_bnStem(16),
          // 20260401 ZJH Stage1: IR(16→16, stride=1, expand=1) x1 — 保持分辨率
          m_stage1_0(16, 16, 1, 1),
          // 20260401 ZJH Stage2: IR(16→24, stride=2, expand=6) + IR(24→24, stride=1, expand=6)
          // ★ Stage2 输出 = 低级特征（24ch, H/4）
          m_stage2_0(16, 24, 2, 6), m_stage2_1(24, 24, 1, 6),
          // 20260401 ZJH Stage3: IR(24→32, stride=2, expand=6) x3
          m_stage3_0(24, 32, 2, 6), m_stage3_1(32, 32, 1, 6), m_stage3_2(32, 32, 1, 6),
          // 20260401 ZJH Stage4: IR(32→64, stride=2, expand=6) x4
          m_stage4_0(32, 64, 2, 6), m_stage4_1(64, 64, 1, 6),
          m_stage4_2(64, 64, 1, 6), m_stage4_3(64, 64, 1, 6),
          // 20260401 ZJH Stage5: IR(64→96, stride=1, expand=6) x3
          // ★ Stage5 输出 = 高级特征（96ch, H/16）
          m_stage5_0(64, 96, 1, 6), m_stage5_1(96, 96, 1, 6), m_stage5_2(96, 96, 1, 6),
          // ===== SE 通道注意力（超越海康 ASI_SEG 的独有模块）=====
          // 20260401 ZJH 在 Stage3/4/5 末尾插入 SE 注意力，让网络聚焦缺陷相关通道
          m_se3(32, 4), m_se4(64, 4), m_se5(96, 4),
          // ===== ASPP-Lite: 3 分支轻量空洞空间金字塔 =====
          // 20260402 ZJH 传递 bUseGroupNorm 给 ASPPLite
          m_aspp(96, 48, 96, bUseGroupNorm),
          // ===== 解码器: 低级特征融合 + 上采样 =====
          // 20260401 ZJH 低级特征 1x1 降维（Stage2: 24ch → 16ch）
          m_lowConv(24, 16, 1, 1, 0, true), m_bnLow(16),
          // 20260401 ZJH 解码器 3x3 卷积（concat 后 96+16=112 → 64）
          m_decConv1(96 + 16, 64, 3, 1, 1, true), m_bnDec1(64),
          // 20260401 ZJH 解码器第二层 3x3 卷积（精炼特征）
          m_decConv2(64, 64, 3, 1, 1, true), m_bnDec2(64),
          m_decDropout(0.1f),  // 20260401 ZJH 轻量网络用低 dropout
          // 20260401 ZJH 分类头 1x1 conv（64 → nClasses）
          m_classifier(64, nNumClasses, 1, 1, 0, true),
          // 20260402 ZJH GroupNorm 实例（解码器直属 BN 的对应 GN）
          m_gnStem(16), m_gnLow(16), m_gnDec1(64), m_gnDec2(64),
          m_bUseGroupNorm(bUseGroupNorm)  // 20260402 ZJH 记录归一化模式
    {}

    // 20260401 ZJH 前向传播
    // 输入: [N, Cin, H, W]
    // 输出: [N, nClasses, H, W] — 与输入同分辨率的逐像素分类 logits
    Tensor forward(const Tensor& input) override {
        // 20260401 ZJH ===== 编码器 =====
        // Stem: Conv3x3(stride=2) + BN/GN + ReLU → [N, 16, H/2, W/2]
        // 20260402 ZJH stem 归一化根据 m_bUseGroupNorm 选择 GN 或 BN
        auto mStemConv = m_stem.forward(input);  // 20260402 ZJH stem 卷积
        auto mStemNorm = m_bUseGroupNorm ? m_gnStem.forward(mStemConv) : m_bnStem.forward(mStemConv);
        auto stem = m_relu.forward(mStemNorm);

        // 20260401 ZJH Stage1: [N, 16, H/2, W/2]
        auto s1 = m_stage1_0.forward(stem);

        // 20260401 ZJH Stage2: [N, 24, H/4, W/4] ★ 低级特征
        auto s2 = m_stage2_1.forward(m_stage2_0.forward(s1));

        // 20260401 ZJH Stage3: [N, 32, H/8, W/8] + SE 通道注意力
        auto s3 = m_se3.forward(m_stage3_2.forward(m_stage3_1.forward(m_stage3_0.forward(s2))));

        // 20260401 ZJH Stage4: [N, 64, H/16, W/16] + SE 通道注意力
        auto s4 = m_se4.forward(m_stage4_3.forward(m_stage4_2.forward(m_stage4_1.forward(m_stage4_0.forward(s3)))));

        // 20260401 ZJH Stage5: [N, 96, H/16, W/16] ★ 高级特征 + SE 通道注意力
        auto s5 = m_se5.forward(m_stage5_2.forward(m_stage5_1.forward(m_stage5_0.forward(s4))));

        // 20260401 ZJH ===== ASPP-Lite =====
        auto asppOut = m_aspp.forward(s5);  // 20260401 ZJH [N, 96, H/16, W/16]

        // 20260401 ZJH ===== 解码器 =====
        // 低级特征 1x1 降维: 24ch → 16ch
        // 20260402 ZJH 根据 m_bUseGroupNorm 选择 GN 或 BN
        auto mLowConvOut = m_lowConv.forward(s2);  // 20260402 ZJH 低级特征降维
        auto mLowNorm = m_bUseGroupNorm ? m_gnLow.forward(mLowConvOut) : m_bnLow.forward(mLowConvOut);
        auto lowFeat = m_relu.forward(mLowNorm);  // [N, 16, H/4, W/4]

        // 20260401 ZJH 上采样 ASPP 输出到低级特征尺寸（H/16 → H/4, 即 4x）
        int nLowH = lowFeat.shape(2);  // 20260401 ZJH 低级特征高度
        int nAsppH = asppOut.shape(2);  // 20260401 ZJH ASPP 输出高度
        int nUpsampleScale = std::max(1, nLowH / std::max(1, nAsppH));
        auto asppUp = tensorUpsampleBilinear(asppOut, nUpsampleScale);

        // 20260401 ZJH 通道拼接: ASPP(96ch) + lowFeat(16ch) = 112ch
        auto concat = tensorConcatChannels(asppUp, lowFeat);

        // 20260401 ZJH 解码器 3x3 conv x2
        // 20260402 ZJH 解码器归一化根据 m_bUseGroupNorm 选择 GN 或 BN
        auto mDecConv1Out = m_decConv1.forward(concat);  // 20260402 ZJH 解码器 conv1
        auto mDecNorm1 = m_bUseGroupNorm ? m_gnDec1.forward(mDecConv1Out) : m_bnDec1.forward(mDecConv1Out);
        auto dec = m_relu.forward(mDecNorm1);  // 112→64
        auto mDecConv2Out = m_decConv2.forward(dec);  // 20260402 ZJH 解码器 conv2
        auto mDecNorm2 = m_bUseGroupNorm ? m_gnDec2.forward(mDecConv2Out) : m_bnDec2.forward(mDecConv2Out);
        dec = m_relu.forward(mDecNorm2);  // 64→64
        dec = m_decDropout.forward(dec);

        // 20260401 ZJH 分类头 1x1
        auto logits = m_classifier.forward(dec);  // [N, nClasses, H/4, W/4]

        // 20260401 ZJH 上采样到输入分辨率（H/4 → H, 即 4x）
        int nInH = input.shape(2);
        int nOutH = logits.shape(2);
        if (nOutH < nInH) {
            int nScale = nInH / nOutH;
            logits = tensorUpsampleBilinear(logits, nScale);
        }
        return logits;  // 20260401 ZJH [N, nClasses, H, W]
    }

    // 20260401 ZJH 收集所有可训练参数
    // 20260402 ZJH 根据 m_bUseGroupNorm 收集 GN 或 BN 的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto append = [&](std::vector<Tensor*> ps) { for (auto* p : ps) v.push_back(p); };
        // 20260401 ZJH 编码器参数
        append(m_stem.parameters());
        append(m_bUseGroupNorm ? m_gnStem.parameters() : m_bnStem.parameters());  // 20260402 ZJH stem 归一化
        append(m_stage1_0.parameters());
        append(m_stage2_0.parameters()); append(m_stage2_1.parameters());
        append(m_stage3_0.parameters()); append(m_stage3_1.parameters()); append(m_stage3_2.parameters());
        append(m_stage4_0.parameters()); append(m_stage4_1.parameters());
        append(m_stage4_2.parameters()); append(m_stage4_3.parameters());
        append(m_stage5_0.parameters()); append(m_stage5_1.parameters()); append(m_stage5_2.parameters());
        // 20260401 ZJH SE 注意力参数
        append(m_se3.parameters()); append(m_se4.parameters()); append(m_se5.parameters());
        // 20260401 ZJH ASPP-Lite 参数（内部已处理 GN/BN 切换）
        append(m_aspp.parameters());
        // 20260401 ZJH 解码器参数
        append(m_lowConv.parameters());
        append(m_bUseGroupNorm ? m_gnLow.parameters() : m_bnLow.parameters());    // 20260402 ZJH 低级特征归一化
        append(m_decConv1.parameters());
        append(m_bUseGroupNorm ? m_gnDec1.parameters() : m_bnDec1.parameters());  // 20260402 ZJH 解码器1归一化
        append(m_decConv2.parameters());
        append(m_bUseGroupNorm ? m_gnDec2.parameters() : m_bnDec2.parameters());  // 20260402 ZJH 解码器2归一化
        append(m_classifier.parameters());
        return v;
    }

    // 20260401 ZJH 命名参数收集（用于序列化和迁移学习）
    // 20260402 ZJH 支持 GN/BN 双模式
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendParams = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecParams = mod.namedParameters(strFullPrefix);
            vecResult.insert(vecResult.end(), vecParams.begin(), vecParams.end());
        };
        // 20260401 ZJH 编码器
        appendParams("stem", m_stem);
        // 20260402 ZJH stem 归一化命名
        if (m_bUseGroupNorm) appendParams("gnStem", m_gnStem);
        else                 appendParams("bnStem", m_bnStem);
        appendParams("stage1_0", m_stage1_0);
        appendParams("stage2_0", m_stage2_0);  appendParams("stage2_1", m_stage2_1);
        appendParams("stage3_0", m_stage3_0);  appendParams("stage3_1", m_stage3_1);
        appendParams("stage3_2", m_stage3_2);
        appendParams("stage4_0", m_stage4_0);  appendParams("stage4_1", m_stage4_1);
        appendParams("stage4_2", m_stage4_2);  appendParams("stage4_3", m_stage4_3);
        appendParams("stage5_0", m_stage5_0);  appendParams("stage5_1", m_stage5_1);
        appendParams("stage5_2", m_stage5_2);
        // 20260401 ZJH SE 注意力
        appendParams("se3", m_se3);  appendParams("se4", m_se4);  appendParams("se5", m_se5);
        // 20260401 ZJH ASPP-Lite（内部已处理 GN/BN 命名）
        appendParams("aspp", m_aspp);
        // 20260401 ZJH 解码器
        appendParams("lowConv", m_lowConv);
        if (m_bUseGroupNorm) appendParams("gnLow", m_gnLow);
        else                 appendParams("bnLow", m_bnLow);
        appendParams("decConv1", m_decConv1);
        if (m_bUseGroupNorm) appendParams("gnDec1", m_gnDec1);
        else                 appendParams("bnDec1", m_bnDec1);
        appendParams("decConv2", m_decConv2);
        if (m_bUseGroupNorm) appendParams("gnDec2", m_gnDec2);
        else                 appendParams("bnDec2", m_bnDec2);
        appendParams("classifier", m_classifier);
        return vecResult;
    }

    // 20260401 ZJH 收集所有 BN running stats（推理时需要）
    // 20260402 ZJH GN 模式下直属 BN 无缓冲区，InvertedResidual 子模块内部不受影响
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        // 20260402 ZJH 直属归一化层的缓冲区（GN 模式下无缓冲区）
        if (!m_bUseGroupNorm) {
            for (auto* p : m_bnStem.buffers()) v.push_back(p);
        }
        // 20260402 ZJH InvertedResidual 子模块内部 BN（不受 bUseGroupNorm 控制）
        for (auto* p : m_stage1_0.buffers()) v.push_back(p);
        for (auto* p : m_stage2_0.buffers()) v.push_back(p);
        for (auto* p : m_stage2_1.buffers()) v.push_back(p);
        for (auto* p : m_stage3_0.buffers()) v.push_back(p);
        for (auto* p : m_stage3_1.buffers()) v.push_back(p);
        for (auto* p : m_stage3_2.buffers()) v.push_back(p);
        for (auto* p : m_stage4_0.buffers()) v.push_back(p);
        for (auto* p : m_stage4_1.buffers()) v.push_back(p);
        for (auto* p : m_stage4_2.buffers()) v.push_back(p);
        for (auto* p : m_stage4_3.buffers()) v.push_back(p);
        for (auto* p : m_stage5_0.buffers()) v.push_back(p);
        for (auto* p : m_stage5_1.buffers()) v.push_back(p);
        for (auto* p : m_stage5_2.buffers()) v.push_back(p);
        for (auto* p : m_aspp.buffers()) v.push_back(p);  // 20260402 ZJH ASPPLite 内部已处理 GN/BN
        if (!m_bUseGroupNorm) {
            for (auto* p : m_bnLow.buffers()) v.push_back(p);
            for (auto* p : m_bnDec1.buffers()) v.push_back(p);
            for (auto* p : m_bnDec2.buffers()) v.push_back(p);
        }
        return v;
    }

    // 20260401 ZJH 命名缓冲区收集（用于序列化）
    // 20260402 ZJH 支持 GN/BN 双模式
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        if (!m_bUseGroupNorm) appendBufs("bnStem", m_bnStem);
        appendBufs("stage1_0", m_stage1_0);
        appendBufs("stage2_0", m_stage2_0);  appendBufs("stage2_1", m_stage2_1);
        appendBufs("stage3_0", m_stage3_0);  appendBufs("stage3_1", m_stage3_1);
        appendBufs("stage3_2", m_stage3_2);
        appendBufs("stage4_0", m_stage4_0);  appendBufs("stage4_1", m_stage4_1);
        appendBufs("stage4_2", m_stage4_2);  appendBufs("stage4_3", m_stage4_3);
        appendBufs("stage5_0", m_stage5_0);  appendBufs("stage5_1", m_stage5_1);
        appendBufs("stage5_2", m_stage5_2);
        appendBufs("aspp", m_aspp);
        if (!m_bUseGroupNorm) {
            appendBufs("bnLow", m_bnLow);
            appendBufs("bnDec1", m_bnDec1);  appendBufs("bnDec2", m_bnDec2);
        }
        return vecResult;
    }

    // 20260401 ZJH 递归设置训练/推理模式
    void train(bool b = true) override {
        m_bTraining = b;
        // 20260402 ZJH 训练模式传播到当前使用的归一化层
        if (m_bUseGroupNorm) m_gnStem.train(b);
        else                 m_bnStem.train(b);
        m_stage1_0.train(b);
        m_stage2_0.train(b); m_stage2_1.train(b);
        m_stage3_0.train(b); m_stage3_1.train(b); m_stage3_2.train(b);
        m_stage4_0.train(b); m_stage4_1.train(b); m_stage4_2.train(b); m_stage4_3.train(b);
        m_stage5_0.train(b); m_stage5_1.train(b); m_stage5_2.train(b);
        m_aspp.train(b);
        if (m_bUseGroupNorm) {
            m_gnLow.train(b); m_gnDec1.train(b); m_gnDec2.train(b);
        } else {
            m_bnLow.train(b); m_bnDec1.train(b); m_bnDec2.train(b);
        }
        m_decDropout.train(b);
    }

private:
    int m_nNumClasses;  // 20260406 ZJH 输出分割类别数

    // 20260401 ZJH ===== 编码器: MobileNet 风格 =====
    Conv2d m_stem;  BatchNorm2d m_bnStem;                          // 20260401 ZJH Stem 3x3
    InvertedResidual m_stage1_0;                                    // 20260401 ZJH Stage1 (16→16)
    InvertedResidual m_stage2_0, m_stage2_1;                        // 20260401 ZJH Stage2 (16→24) ★low
    InvertedResidual m_stage3_0, m_stage3_1, m_stage3_2;            // 20260401 ZJH Stage3 (24→32)
    InvertedResidual m_stage4_0, m_stage4_1, m_stage4_2, m_stage4_3;  // 20260401 ZJH Stage4 (32→64)
    InvertedResidual m_stage5_0, m_stage5_1, m_stage5_2;            // 20260401 ZJH Stage5 (64→96) ★high

    // 20260401 ZJH ===== SE 通道注意力（超越海康）=====
    SEBlock m_se3;  // 20260401 ZJH Stage3 后 SE (32ch, reduction=4)
    SEBlock m_se4;  // 20260401 ZJH Stage4 后 SE (64ch, reduction=4)
    SEBlock m_se5;  // 20260401 ZJH Stage5 后 SE (96ch, reduction=4)

    // 20260401 ZJH ===== ASPP-Lite =====
    ASPPLite m_aspp;

    // 20260401 ZJH ===== 解码器 =====
    Conv2d m_lowConv;  BatchNorm2d m_bnLow;            // 20260401 ZJH 低级特征降维 24→16
    Conv2d m_decConv1;  BatchNorm2d m_bnDec1;          // 20260401 ZJH 解码器 conv1 (112→64)
    Conv2d m_decConv2;  BatchNorm2d m_bnDec2;          // 20260401 ZJH 解码器 conv2 (64→64)
    Dropout2d m_decDropout;
    Conv2d m_classifier;                                // 20260401 ZJH 分类头 (64→nClasses)
    // 20260402 ZJH GroupNorm 实例（bUseGroupNorm=true 时使用）
    GroupNorm2d m_gnStem;   // 20260402 ZJH stem GN (16ch)
    GroupNorm2d m_gnLow;    // 20260402 ZJH 低级特征 GN (16ch)
    GroupNorm2d m_gnDec1;   // 20260402 ZJH 解码器1 GN (64ch)
    GroupNorm2d m_gnDec2;   // 20260402 ZJH 解码器2 GN (64ch)
    bool m_bUseGroupNorm = false;  // 20260402 ZJH 归一化模式标志
    ReLU m_relu;
};

}  // namespace om
