// 20260402 ZJH ConvNeXt-Tiny 模块 — 现代化纯 CNN 架构 (Liu et al., 2022)
// 精度对标 Swin Transformer，ImageNet Top-1: 82.1%（远超 ResNet50 76.1%）
// 核心: DW-Conv7x7 → LayerNorm → 1x1Conv → GELU → 1x1Conv
// 4 Stage: [3, 3, 9, 3] 层，通道 [96, 192, 384, 768]
module;

#include <vector>       // 20260402 ZJH std::vector 用于形状描述和参数列表
#include <string>       // 20260402 ZJH std::string 用于模块命名
#include <cmath>        // 20260402 ZJH std::sqrt 用于 LayerNorm
#include <memory>       // 20260402 ZJH std::shared_ptr 用于子模块注册
#include <algorithm>    // 20260402 ZJH std::max 用于 clamp
#include <numeric>      // 20260402 ZJH std::accumulate（预留）

export module om.engine.convnext;

// 20260402 ZJH 导入依赖模块
import om.engine.tensor;       // 20260402 ZJH Tensor 数据结构
import om.engine.tensor_ops;   // 20260402 ZJH tensorAdd/tensorMul/tensorMulScalar 等运算
import om.engine.module;       // 20260402 ZJH Module 基类
import om.engine.conv;         // 20260402 ZJH Conv2d 卷积层
import om.engine.activations;  // 20260402 ZJH GELU 激活函数
import om.engine.linear;       // 20260402 ZJH Linear 全连接层（1x1Conv 等价实现）

export namespace om {

// =====================================================================
// 20260402 ZJH LayerNormChannel — 沿通道维度的 Layer Normalization
// ConvNeXt 的核心创新之一: 使用 LN 替代 BN
// 输入: [B, C, H, W]，沿 C 维度归一化（每个空间位置独立）
// 与标准 LayerNorm 的区别: 作用于 [C] 维度而非 [H,W] 维度
// =====================================================================
class LayerNormChannel : public Module {
public:
    // 20260402 ZJH 构造函数
    // 参数: nChannels - 通道数（归一化维度）
    //       fEps - 数值稳定性常数（防止除零）
    LayerNormChannel(int nChannels, float fEps = 1e-6f)
        : m_nChannels(nChannels)  // 20260402 ZJH 保存通道数
        , m_fEps(fEps)            // 20260402 ZJH 保存 epsilon
    {
        // 20260402 ZJH 可学习的缩放参数 gamma，初始化为全 1
        m_weight = Tensor::ones({nChannels});
        // 20260402 ZJH 可学习的偏移参数 beta，初始化为全 0
        m_bias = Tensor::zeros({nChannels});
        // 20260402 ZJH 注册为可训练参数
        registerParameter("weight", m_weight);
        registerParameter("bias", m_bias);
    }

    // 20260402 ZJH 前向传播: 沿通道维度归一化
    // 输入: [B, C, H, W]
    // 输出: [B, C, H, W]（归一化后）
    Tensor forward(const Tensor& input) override {
        int nB = input.shape(0);  // 20260402 ZJH batch 大小
        int nC = input.shape(1);  // 20260402 ZJH 通道数
        int nH = input.shape(2);  // 20260402 ZJH 高度
        int nW = input.shape(3);  // 20260402 ZJH 宽度
        int nSpatial = nH * nW;   // 20260402 ZJH 空间像素数

        // 20260402 ZJH 获取输入数据指针
        const float* pIn = input.floatDataPtr();
        // 20260402 ZJH 创建输出张量
        auto output = Tensor::zeros({nB, nC, nH, nW});
        float* pOut = output.mutableFloatDataPtr();
        // 20260402 ZJH 获取 gamma/beta 参数指针
        const float* pW = m_weight.floatDataPtr();
        const float* pBias = m_bias.floatDataPtr();

        // 20260402 ZJH 遍历每个 batch 和每个空间位置
        for (int b = 0; b < nB; ++b) {
            for (int s = 0; s < nSpatial; ++s) {
                // 20260402 ZJH 计算沿通道维度的均值
                float fMean = 0.0f;
                for (int c = 0; c < nC; ++c) {
                    // 20260402 ZJH 索引: b*C*H*W + c*H*W + s
                    fMean += pIn[b * nC * nSpatial + c * nSpatial + s];
                }
                fMean /= static_cast<float>(nC);  // 20260402 ZJH 通道均值

                // 20260402 ZJH 计算沿通道维度的方差
                float fVar = 0.0f;
                for (int c = 0; c < nC; ++c) {
                    float fDiff = pIn[b * nC * nSpatial + c * nSpatial + s] - fMean;
                    fVar += fDiff * fDiff;
                }
                fVar /= static_cast<float>(nC);  // 20260402 ZJH 通道方差

                // 20260402 ZJH 归一化 + 仿射变换: y = gamma * (x - mean) / sqrt(var + eps) + beta
                float fInvStd = 1.0f / std::sqrt(fVar + m_fEps);  // 20260402 ZJH 标准差倒数
                for (int c = 0; c < nC; ++c) {
                    int nIdx = b * nC * nSpatial + c * nSpatial + s;  // 20260402 ZJH 线性索引
                    float fNorm = (pIn[nIdx] - fMean) * fInvStd;      // 20260402 ZJH 归一化值
                    pOut[nIdx] = pW[c] * fNorm + pBias[c];             // 20260402 ZJH 仿射变换
                }
            }
        }
        return output;  // 20260402 ZJH 返回归一化后的张量
    }

private:
    int m_nChannels;    // 20260402 ZJH 通道数（归一化维度大小）
    float m_fEps;       // 20260402 ZJH 数值稳定性 epsilon
    Tensor m_weight;    // 20260402 ZJH gamma 缩放参数 [C]
    Tensor m_bias;      // 20260402 ZJH beta 偏移参数 [C]
};

// =====================================================================
// 20260402 ZJH ConvNeXtBlock — ConvNeXt 核心构建块
// 架构: DW-Conv7x7 → LayerNorm → 1x1Conv(4x expand) → GELU → 1x1Conv
// 设计哲学: 用现代化技巧（大核、LN、GELU）提升传统 CNN
// 参考: "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
// =====================================================================
class ConvNeXtBlock : public Module {
public:
    // 20260402 ZJH 构造函数
    // 参数: nDim - 输入/输出通道数
    //       nExpandRatio - 中间层扩展倍率（默认 4，即 MLP 隐藏层 = 4 * nDim）
    ConvNeXtBlock(int nDim, int nExpandRatio = 4)
        : m_nDim(nDim)                                       // 20260402 ZJH 保存通道数
        , m_dwConv(nDim, nDim, 7, 1, 3, true, nDim)         // 20260402 ZJH 7x7 深度可分离卷积（groups=nDim）
        , m_norm(nDim)                                        // 20260402 ZJH LayerNorm 归一化
        , m_pwConv1(nDim, nDim * nExpandRatio, 1, 1, 0, true) // 20260402 ZJH 1x1 逐点卷积（通道扩展 4x）
        , m_pwConv2(nDim * nExpandRatio, nDim, 1, 1, 0, true) // 20260402 ZJH 1x1 逐点卷积（通道恢复）
    {
        // 20260402 ZJH 注册子模块以便 parameters() 能遍历到所有权重
        registerModule("dwconv", std::make_shared<Conv2d>(m_dwConv));
        registerModule("norm", std::make_shared<LayerNormChannel>(m_norm));
        registerModule("pwconv1", std::make_shared<Conv2d>(m_pwConv1));
        registerModule("pwconv2", std::make_shared<Conv2d>(m_pwConv2));

        // 20260402 ZJH Layer Scale 参数（初始化为 1e-6，稳定深层网络训练）
        // ConvNeXt 论文: 小初始值防止深层残差分支过早主导
        m_gamma = Tensor::ones({nDim});
        float* pGamma = m_gamma.mutableFloatDataPtr();
        for (int i = 0; i < nDim; ++i) {
            pGamma[i] = 1e-6f;  // 20260402 ZJH 初始化为极小值
        }
        registerParameter("gamma", m_gamma);  // 20260402 ZJH 注册为可训练参数
    }

    // 20260402 ZJH 前向传播
    // 输入: [B, C, H, W]
    // 输出: [B, C, H, W]（残差连接后）
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH 保存残差（identity shortcut）
        auto residual = input;

        // 20260402 ZJH Step 1: 7x7 深度可分离卷积（大感受野，计算量低）
        auto x = m_dwConv.forward(input);

        // 20260402 ZJH Step 2: LayerNorm（沿通道维度归一化）
        x = m_norm.forward(x);

        // 20260402 ZJH Step 3: 1x1 逐点卷积（通道扩展 4x，即 inverted bottleneck）
        x = m_pwConv1.forward(x);

        // 20260402 ZJH Step 4: GELU 激活（ConvNeXt 论文推荐，比 ReLU 平滑）
        x = m_gelu.forward(x);

        // 20260402 ZJH Step 5: 1x1 逐点卷积（通道恢复到原始维度）
        x = m_pwConv2.forward(x);

        // 20260402 ZJH Step 6: Layer Scale（逐通道乘以可学习系数 gamma）
        {
            int nB = x.shape(0);       // 20260402 ZJH batch 大小
            int nC = x.shape(1);       // 20260402 ZJH 通道数
            int nH = x.shape(2);       // 20260402 ZJH 高度
            int nW = x.shape(3);       // 20260402 ZJH 宽度
            int nSpatial = nH * nW;    // 20260402 ZJH 空间像素数
            const float* pGamma = m_gamma.floatDataPtr();     // 20260402 ZJH gamma 参数指针
            float* pX = x.mutableFloatDataPtr();               // 20260402 ZJH 数据指针
            // 20260402 ZJH 逐通道乘以 gamma: x[b,c,h,w] *= gamma[c]
            for (int b = 0; b < nB; ++b) {
                for (int c = 0; c < nC; ++c) {
                    float fG = pGamma[c];  // 20260402 ZJH 当前通道的 gamma 值
                    int nBase = b * nC * nSpatial + c * nSpatial;  // 20260402 ZJH 起始偏移
                    for (int s = 0; s < nSpatial; ++s) {
                        pX[nBase + s] *= fG;  // 20260402 ZJH 逐元素乘以 gamma
                    }
                }
            }
        }

        // 20260402 ZJH Step 7: 残差连接（x = x + residual）
        return tensorAdd(x, residual);
    }

private:
    int m_nDim;                // 20260402 ZJH 通道数
    Conv2d m_dwConv;           // 20260402 ZJH 7x7 深度可分离卷积
    LayerNormChannel m_norm;   // 20260402 ZJH LayerNorm
    Conv2d m_pwConv1;          // 20260402 ZJH 1x1 逐点卷积（扩展）
    Conv2d m_pwConv2;          // 20260402 ZJH 1x1 逐点卷积（恢复）
    GELU m_gelu;               // 20260402 ZJH GELU 激活函数
    Tensor m_gamma;            // 20260402 ZJH Layer Scale 参数 [C]
};

// =====================================================================
// 20260402 ZJH ConvNeXtDownsample — Stage 间下采样层
// 使用 LayerNorm + 2x2 Conv stride=2 替代传统的 MaxPool
// 更平滑的空间下采样，减少信息损失
// =====================================================================
class ConvNeXtDownsample : public Module {
public:
    // 20260402 ZJH 构造函数
    // 参数: nInChannels - 输入通道数
    //       nOutChannels - 输出通道数
    ConvNeXtDownsample(int nInChannels, int nOutChannels)
        : m_norm(nInChannels)                               // 20260402 ZJH 下采样前 LayerNorm
        , m_conv(nInChannels, nOutChannels, 2, 2, 0, true)  // 20260402 ZJH 2x2 Conv stride=2（空间减半）
    {
        // 20260402 ZJH 注册子模块
        registerModule("norm", std::make_shared<LayerNormChannel>(m_norm));
        registerModule("conv", std::make_shared<Conv2d>(m_conv));
    }

    // 20260402 ZJH 前向传播: LN → 2x2 Conv stride=2
    Tensor forward(const Tensor& input) override {
        auto x = m_norm.forward(input);   // 20260402 ZJH LayerNorm 归一化
        x = m_conv.forward(x);            // 20260402 ZJH 2x2 Conv 下采样
        return x;                          // 20260402 ZJH 返回下采样结果
    }

private:
    LayerNormChannel m_norm;  // 20260402 ZJH LayerNorm
    Conv2d m_conv;            // 20260402 ZJH 2x2 Conv stride=2
};

// =====================================================================
// 20260402 ZJH ConvNeXtTiny — ConvNeXt-Tiny 完整模型
// 架构配置:
//   Stem: 4x4 Conv stride=4 + LayerNorm（16x 下采样到 56x56 from 224x224）
//   Stage 1: 3 blocks, 96 channels
//   Stage 2: 3 blocks, 192 channels（2x 下采样）
//   Stage 3: 9 blocks, 384 channels（2x 下采样）
//   Stage 4: 3 blocks, 768 channels（2x 下采样）
//   Head: GlobalAvgPool + LayerNorm + Linear(768 → nNumClasses)
// 总参数量: ~28.6M（vs ResNet50 ~25.6M）
// ImageNet Top-1: 82.1%
// =====================================================================
class ConvNeXtTiny : public Module {
public:
    // 20260402 ZJH 构造函数
    // 参数: nNumClasses - 分类类别数
    //       nInChannels - 输入通道数（默认 3 = RGB）
    ConvNeXtTiny(int nNumClasses = 1000, int nInChannels = 3)
        : m_nNumClasses(nNumClasses)
        // 20260402 ZJH Stem: 4x4 Conv stride=4（等效 ViT 的 patch embedding，16x 下采样）
        , m_stemConv(nInChannels, 96, 4, 4, 0, true)
        , m_stemNorm(96)
        // 20260402 ZJH 三个 Stage 间下采样层（Stage 1→2, 2→3, 3→4）
        , m_down1(96, 192)    // 20260402 ZJH 96 → 192 通道，空间减半
        , m_down2(192, 384)   // 20260402 ZJH 192 → 384 通道，空间减半
        , m_down3(384, 768)   // 20260402 ZJH 384 → 768 通道，空间减半
        // 20260402 ZJH 最终分类头: LayerNorm + Linear
        , m_headNorm(768)                   // 20260402 ZJH 全局平均池化后的 LayerNorm
        , m_headLinear(768, nNumClasses)    // 20260402 ZJH 分类全连接层
    {
        // 20260402 ZJH 注册 Stem 子模块
        registerModule("stem_conv", std::make_shared<Conv2d>(m_stemConv));
        registerModule("stem_norm", std::make_shared<LayerNormChannel>(m_stemNorm));

        // 20260402 ZJH 创建 Stage 1: 3 个 ConvNeXtBlock，96 通道
        for (int i = 0; i < 3; ++i) {
            auto pBlock = std::make_shared<ConvNeXtBlock>(96);
            m_vecStage1.push_back(pBlock);
            registerModule("stage1_block" + std::to_string(i), pBlock);
        }

        // 20260402 ZJH 注册下采样层 1→2
        registerModule("down1", std::make_shared<ConvNeXtDownsample>(m_down1));

        // 20260402 ZJH 创建 Stage 2: 3 个 ConvNeXtBlock，192 通道
        for (int i = 0; i < 3; ++i) {
            auto pBlock = std::make_shared<ConvNeXtBlock>(192);
            m_vecStage2.push_back(pBlock);
            registerModule("stage2_block" + std::to_string(i), pBlock);
        }

        // 20260402 ZJH 注册下采样层 2→3
        registerModule("down2", std::make_shared<ConvNeXtDownsample>(m_down2));

        // 20260402 ZJH 创建 Stage 3: 9 个 ConvNeXtBlock，384 通道（最深 stage）
        for (int i = 0; i < 9; ++i) {
            auto pBlock = std::make_shared<ConvNeXtBlock>(384);
            m_vecStage3.push_back(pBlock);
            registerModule("stage3_block" + std::to_string(i), pBlock);
        }

        // 20260402 ZJH 注册下采样层 3→4
        registerModule("down3", std::make_shared<ConvNeXtDownsample>(m_down3));

        // 20260402 ZJH 创建 Stage 4: 3 个 ConvNeXtBlock，768 通道
        for (int i = 0; i < 3; ++i) {
            auto pBlock = std::make_shared<ConvNeXtBlock>(768);
            m_vecStage4.push_back(pBlock);
            registerModule("stage4_block" + std::to_string(i), pBlock);
        }

        // 20260402 ZJH 注册分类头子模块
        registerModule("head_norm", std::make_shared<LayerNormChannel>(m_headNorm));
        registerModule("head_linear", std::make_shared<Linear>(m_headLinear));
    }

    // 20260402 ZJH 前向传播
    // 输入: [B, C, H, W]（典型 [B, 3, 224, 224]）
    // 输出: [B, nNumClasses]（分类 logits）
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH Stem: 4x4 Conv stride=4 → LN
        // 224x224 → 56x56，通道 3 → 96
        auto x = m_stemConv.forward(input);
        x = m_stemNorm.forward(x);

        // 20260402 ZJH Stage 1: 3 blocks @ 96ch, 56x56
        for (auto& pBlock : m_vecStage1) {
            x = pBlock->forward(x);
        }

        // 20260402 ZJH Downsample 1: 56x56 → 28x28, 96 → 192
        x = m_down1.forward(x);

        // 20260402 ZJH Stage 2: 3 blocks @ 192ch, 28x28
        for (auto& pBlock : m_vecStage2) {
            x = pBlock->forward(x);
        }

        // 20260402 ZJH Downsample 2: 28x28 → 14x14, 192 → 384
        x = m_down2.forward(x);

        // 20260402 ZJH Stage 3: 9 blocks @ 384ch, 14x14（最深 stage，提取高层语义特征）
        for (auto& pBlock : m_vecStage3) {
            x = pBlock->forward(x);
        }

        // 20260402 ZJH Downsample 3: 14x14 → 7x7, 384 → 768
        x = m_down3.forward(x);

        // 20260402 ZJH Stage 4: 3 blocks @ 768ch, 7x7
        for (auto& pBlock : m_vecStage4) {
            x = pBlock->forward(x);
        }

        // 20260402 ZJH Global Average Pooling: [B, 768, 7, 7] → [B, 768, 1, 1] → [B, 768]
        {
            int nB = x.shape(0);        // 20260402 ZJH batch 大小
            int nC = x.shape(1);        // 20260402 ZJH 通道数（768）
            int nH = x.shape(2);        // 20260402 ZJH 空间高度
            int nW = x.shape(3);        // 20260402 ZJH 空间宽度
            int nSpatial = nH * nW;     // 20260402 ZJH 空间像素数
            auto pooled = Tensor::zeros({nB, nC, 1, 1});  // 20260402 ZJH 池化结果
            const float* pX = x.floatDataPtr();
            float* pPool = pooled.mutableFloatDataPtr();
            // 20260402 ZJH 遍历每个 batch 和通道，计算空间平均值
            for (int b = 0; b < nB; ++b) {
                for (int c = 0; c < nC; ++c) {
                    float fSum = 0.0f;  // 20260402 ZJH 空间累加
                    int nBase = b * nC * nSpatial + c * nSpatial;  // 20260402 ZJH 起始偏移
                    for (int s = 0; s < nSpatial; ++s) {
                        fSum += pX[nBase + s];  // 20260402 ZJH 累加所有空间位置
                    }
                    pPool[b * nC + c] = fSum / static_cast<float>(nSpatial);  // 20260402 ZJH 平均值
                }
            }
            // 20260402 ZJH reshape 为 [B, 768]
            x = tensorReshape(pooled, {nB, nC});
        }

        // 20260402 ZJH Head LayerNorm（作用于 [B, 768]，沿最后维度归一化）
        // 将 [B, 768] reshape 为 [B, 768, 1, 1] 以复用 LayerNormChannel
        {
            int nB = x.shape(0);   // 20260402 ZJH batch 大小
            int nC = x.shape(1);   // 20260402 ZJH 通道数（768）
            x = tensorReshape(x, {nB, nC, 1, 1});  // 20260402 ZJH 升维以适配 LayerNormChannel
            x = m_headNorm.forward(x);               // 20260402 ZJH LayerNorm
            x = tensorReshape(x, {nB, nC});          // 20260402 ZJH 恢复 [B, 768]
        }

        // 20260402 ZJH 分类全连接层: [B, 768] → [B, nNumClasses]
        x = m_headLinear.forward(x);

        return x;  // 20260402 ZJH 返回分类 logits（未经 softmax）
    }

    // 20260402 ZJH buffers 和 namedBuffers override
    // ConvNeXt 使用 LayerNorm（无 running stats），仅返回空列表
    std::vector<Tensor*> buffers() override { return {}; }
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& /*strPrefix*/ = "") override { return {}; }

private:
    int m_nNumClasses;    // 20260402 ZJH 分类类别数

    // 20260402 ZJH Stem（patchify 下采样）
    Conv2d m_stemConv;              // 20260402 ZJH 4x4 Conv stride=4
    LayerNormChannel m_stemNorm;    // 20260402 ZJH Stem 后 LayerNorm

    // 20260402 ZJH 4 个 Stage 的 Block 列表
    std::vector<std::shared_ptr<ConvNeXtBlock>> m_vecStage1;  // 20260402 ZJH Stage 1: 3 blocks @ 96ch
    std::vector<std::shared_ptr<ConvNeXtBlock>> m_vecStage2;  // 20260402 ZJH Stage 2: 3 blocks @ 192ch
    std::vector<std::shared_ptr<ConvNeXtBlock>> m_vecStage3;  // 20260402 ZJH Stage 3: 9 blocks @ 384ch
    std::vector<std::shared_ptr<ConvNeXtBlock>> m_vecStage4;  // 20260402 ZJH Stage 4: 3 blocks @ 768ch

    // 20260402 ZJH Stage 间下采样层
    ConvNeXtDownsample m_down1;  // 20260402 ZJH 96 → 192
    ConvNeXtDownsample m_down2;  // 20260402 ZJH 192 → 384
    ConvNeXtDownsample m_down3;  // 20260402 ZJH 384 → 768

    // 20260402 ZJH 分类头
    LayerNormChannel m_headNorm;    // 20260402 ZJH 最终 LayerNorm
    Linear m_headLinear;            // 20260402 ZJH 分类全连接层
};

}  // namespace om
