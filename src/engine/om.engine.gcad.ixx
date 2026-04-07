// 20260402 ZJH Global Context Anomaly Detection (GCAD) 模块
// 对标 MVTec Halcon 22.05+ 的 Global Context AD 能力
// 核心创新: 双分支架构同时检测局部纹理异常和全局布局异常
//   - 局部分支 (Local Branch): Teacher-Student patch 特征对比，检测划痕/污渍等纹理缺陷
//   - 全局分支 (Global Branch): ViT 编码器 + 高斯分布建模，检测缺件/错位等布局异常
//   - 融合层 (Fusion): 双分支异常分数加权融合，输出统一的异常分数 + 热力图
// 训练: 仅需正常样本（无监督），20-100 张即可
// 推理: 同时输出 (1) 全局异常分数 (2) 逐像素热力图 (3) 布局异常标志
module;

// 20260406 ZJH 标准库头文件包含
#include <vector>    // 20260406 ZJH 动态数组容器
#include <string>    // 20260406 ZJH 字符串类型
#include <memory>    // 20260406 ZJH 智能指针 (shared_ptr)
#include <cmath>     // 20260406 ZJH 数学函数 (sqrt, exp)
#include <algorithm> // 20260406 ZJH 算法函数 (std::max, std::min)
#include <numeric>   // 20260406 ZJH 数值算法

export module om.engine.gcad;

// 20260402 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =============================================================================
// 20260402 ZJH LocalFeatureExtractor — 局部特征提取器
// 4 层 CNN backbone，提取 patch 级别特征
// 输出: [N, 256, H/8, W/8] 局部特征图
// 与 EfficientAD 共用相同的特征空间（便于迁移学习）
// =============================================================================
class LocalFeatureExtractor : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入图像通道数（默认 3 = RGB）
    LocalFeatureExtractor(int nInChannels = 3)
        : m_conv1(nInChannels, 64, 3, 1, 1, false),   // 20260402 ZJH 层1: Cin→64
          m_bn1(64),
          m_pool1(2, 2, 0),                             // 20260402 ZJH stride-2 下采样
          m_conv2(64, 128, 3, 1, 1, false),             // 20260402 ZJH 层2: 64→128
          m_bn2(128),
          m_pool2(2, 2, 0),
          m_conv3(128, 256, 3, 1, 1, false),            // 20260402 ZJH 层3: 128→256
          m_bn3(256),
          m_pool3(2, 2, 0),
          m_conv4(256, 256, 3, 1, 1, false),            // 20260402 ZJH 层4: 256→256（保持通道数）
          m_bn4(256)
    {
        // 20260402 ZJH 注册所有子模块
        // 20260402 ZJH 值成员，通过基类 EfficientADBackbone 风格的 parameters() 手动管理
    }

    // 20260402 ZJH forward — 提取局部特征
    // input: [N, C, H, W] 输入图像
    // 返回: [N, 256, H/8, W/8] 局部特征图
    Tensor forward(const Tensor& input) override {
        auto x = m_pool1.forward(ReLU().forward(m_bn1.forward(m_conv1.forward(input))));   // 20260402 ZJH /2
        x = m_pool2.forward(ReLU().forward(m_bn2.forward(m_conv2.forward(x))));             // 20260402 ZJH /4
        x = m_pool3.forward(ReLU().forward(m_bn3.forward(m_conv3.forward(x))));             // 20260402 ZJH /8
        x = ReLU().forward(m_bn4.forward(m_conv4.forward(x)));                              // 20260402 ZJH 不池化，保持 /8
        return x;  // 20260402 ZJH [N, 256, H/8, W/8]
    }

private:
    Conv2d m_conv1, m_conv2, m_conv3, m_conv4;  // 20260402 ZJH 卷积层
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;     // 20260402 ZJH 归一化层
    MaxPool2d m_pool1, m_pool2, m_pool3;          // 20260402 ZJH 池化层
};

// =============================================================================
// 20260402 ZJH GlobalContextEncoder — 全局上下文编码器
// 轻量 ViT 架构，编码全图空间关系
// 输入: [N, 256, H/8, W/8] 局部特征图（从 LocalFeatureExtractor 输出）
// 输出: [N, nEmbedDim] 全局上下文向量
// 设计: 将特征图 flatten 为 patch 序列 → Transformer 编码 → CLS token 作为全局表示
// =============================================================================
class GlobalContextEncoder : public Module {
public:
    // 20260402 ZJH 构造函数
    // nFeatureDim: 输入特征维度（来自 LocalFeatureExtractor 的通道数，默认 256）
    // nEmbedDim: Transformer 嵌入维度（默认 128，轻量设计）
    // nNumHeads: 多头注意力头数（默认 4）
    // nNumLayers: Transformer 层数（默认 3，轻量设计防过拟合）
    GlobalContextEncoder(int nFeatureDim = 256, int nEmbedDim = 128,
                         int nNumHeads = 4, int nNumLayers = 3)
        : m_nFeatureDim(nFeatureDim), m_nEmbedDim(nEmbedDim),
          m_nNumHeads(nNumHeads), m_nNumLayers(nNumLayers),
          m_projInput(nFeatureDim, nEmbedDim, true),    // 20260402 ZJH 特征投影层
          m_normFinal(nEmbedDim, nEmbedDim, true),     // 20260402 ZJH 用 Linear 模拟 LayerNorm affine
          m_globalPool(nEmbedDim, nEmbedDim, true)     // 20260402 ZJH 全局池化后投影
    {
        // 20260402 ZJH CLS token — 可学习的全局表示向量
        m_clsToken = Tensor::randn({1, 1, nEmbedDim});  // 20260402 ZJH [1, 1, D]
        float* pCls = m_clsToken.mutableFloatDataPtr();  // 20260402 ZJH 缩放初始化
        for (int i = 0; i < nEmbedDim; ++i) pCls[i] *= 0.02f;
        registerParameter("cls_token", m_clsToken);  // 20260402 ZJH 注册为可学习参数

        // 20260402 ZJH 位置编码（最大支持 1024 个 patch + 1 个 CLS）
        m_posEmbed = Tensor::randn({1, 1025, nEmbedDim});  // 20260402 ZJH [1, maxSeq, D]
        float* pPos = m_posEmbed.mutableFloatDataPtr();
        for (int i = 0; i < 1025 * nEmbedDim; ++i) pPos[i] *= 0.02f;
        registerParameter("pos_embed", m_posEmbed);

        // 20260402 ZJH Transformer 自注意力层（简化版: Q*K^T/sqrt(d) → softmax → V）
        // 每层: MultiHeadAttention + FFN + 残差 + LayerNorm
        for (int i = 0; i < nNumLayers; ++i) {
            auto pQKV = std::make_shared<Linear>(nEmbedDim, nEmbedDim * 3, true);  // 20260402 ZJH QKV 投影
            auto pOut = std::make_shared<Linear>(nEmbedDim, nEmbedDim, true);       // 20260402 ZJH 输出投影
            auto pFfn1 = std::make_shared<Linear>(nEmbedDim, nEmbedDim * 4, true);  // 20260402 ZJH FFN 第1层
            auto pFfn2 = std::make_shared<Linear>(nEmbedDim * 4, nEmbedDim, true);  // 20260402 ZJH FFN 第2层
            m_vecQKV.push_back(pQKV);
            m_vecOutProj.push_back(pOut);
            m_vecFfn1.push_back(pFfn1);
            m_vecFfn2.push_back(pFfn2);
            registerModule("qkv_" + std::to_string(i), pQKV);
            registerModule("out_" + std::to_string(i), pOut);
            registerModule("ffn1_" + std::to_string(i), pFfn1);
            registerModule("ffn2_" + std::to_string(i), pFfn2);
        }

        // 20260402 ZJH m_projInput/m_normFinal/m_globalPool 是值成员，不用 registerModule
    }

    // 20260402 ZJH forward — 编码全局上下文
    // featureMap: [N, 256, fH, fW] 局部特征图
    // 返回: [N, nEmbedDim] 全局上下文向量
    Tensor forward(const Tensor& featureMap) override {
        int nBatch = featureMap.shape(0);   // 20260402 ZJH batch size
        int nC = featureMap.shape(1);       // 20260402 ZJH 特征通道数
        int nFH = featureMap.shape(2);      // 20260402 ZJH 特征图高
        int nFW = featureMap.shape(3);      // 20260402 ZJH 特征图宽
        int nSeqLen = nFH * nFW;            // 20260402 ZJH patch 数量

        // 20260402 ZJH Step 1: reshape [N, C, fH, fW] → [N, nSeqLen, C]
        auto flatFeature = tensorReshape(featureMap, {nBatch, nC, nSeqLen});  // 20260402 ZJH [N, C, S]
        flatFeature = tensorTranspose(flatFeature, 1, 2);  // 20260402 ZJH [N, S, C]

        // 20260402 ZJH Step 2: 线性投影到嵌入维度 [N, S, C] → [N, S, D]
        auto embedded = applyLinearBatched(flatFeature, m_projInput, nBatch, nSeqLen);

        // 20260402 ZJH Step 3: 拼接 CLS token [N, S+1, D]
        // 将 m_clsToken [1,1,D] 扩展到 [N,1,D] 再拼接
        auto clsExpanded = expandCls(m_clsToken, nBatch);  // 20260402 ZJH [N, 1, D]
        // 20260402 ZJH 手动拼接 CLS + embedded: [N,1,D] + [N,S,D] → [N,S+1,D]
        auto withCls = catDim1(clsExpanded, embedded, nBatch, 1, nSeqLen, m_nEmbedDim);
        int nTotalSeq = nSeqLen + 1;  // 20260402 ZJH 总序列长度

        // 20260402 ZJH Step 4: 添加位置编码（截断到实际序列长度）
        auto posSlice = tensorSlice(m_posEmbed, 1, 0, nTotalSeq);  // 20260402 ZJH [1, S+1, D]
        withCls = tensorAdd(withCls, posSlice);  // 20260402 ZJH 广播加法

        // 20260402 ZJH Step 5: Transformer 编码层
        auto x = withCls;  // 20260402 ZJH 当前表示
        for (int layer = 0; layer < m_nNumLayers; ++layer) {
            x = transformerBlock(x, layer, nBatch, nTotalSeq);  // 20260402 ZJH 一层 Transformer
        }

        // 20260402 ZJH Step 6: LayerNorm + 提取 CLS token 作为全局表示
        // CLS token 位于 dim=1 的第 0 个位置
        x = applyLayerNorm(x, m_normFinal, nBatch, nTotalSeq);  // 20260402 ZJH LayerNorm

        // 20260402 ZJH 提取 CLS: x[:, 0, :] → [N, D]
        auto clsOut = tensorSlice(x, 1, 0, 1);  // 20260402 ZJH [N, 1, D]
        clsOut = tensorReshape(clsOut, {nBatch, m_nEmbedDim});  // 20260402 ZJH [N, D]

        // 20260402 ZJH Step 7: 最终全局投影
        clsOut = m_globalPool.forward(clsOut);  // 20260402 ZJH [N, D]

        return clsOut;  // 20260402 ZJH 全局上下文向量
    }

private:
    int m_nFeatureDim;   // 20260402 ZJH 输入特征维度
    int m_nEmbedDim;     // 20260402 ZJH 嵌入维度
    int m_nNumHeads;     // 20260402 ZJH 注意力头数
    int m_nNumLayers;    // 20260402 ZJH Transformer 层数

    Linear m_projInput;   // 20260402 ZJH 输入投影层
    Linear m_normFinal;   // 20260402 ZJH 用 Linear 模拟 LayerNorm (简化)
    Linear m_globalPool;  // 20260402 ZJH 全局池化投影

    Tensor m_clsToken;    // 20260402 ZJH CLS token [1, 1, D]
    Tensor m_posEmbed;    // 20260402 ZJH 位置编码 [1, maxSeq, D]

    // 20260402 ZJH Transformer 层权重
    std::vector<std::shared_ptr<Linear>> m_vecQKV;      // 20260402 ZJH QKV 投影
    std::vector<std::shared_ptr<Linear>> m_vecOutProj;   // 20260402 ZJH 输出投影
    std::vector<std::shared_ptr<Linear>> m_vecFfn1;      // 20260402 ZJH FFN 第1层
    std::vector<std::shared_ptr<Linear>> m_vecFfn2;      // 20260402 ZJH FFN 第2层

    // 20260402 ZJH expandCls — 将 CLS token 从 [1,1,D] 扩展到 [N,1,D]
    // 20260402 ZJH catDim1 — 沿 dim=1 拼接两个 3D 张量
    // a: [N, Sa, D], b: [N, Sb, D] → [N, Sa+Sb, D]
    static Tensor catDim1(const Tensor& a, const Tensor& b,
                          int nBatch, int nSa, int nSb, int nD) {
        int nStotal = nSa + nSb;  // 20260406 ZJH 拼接后的序列长度
        auto result = Tensor::zeros({nBatch, nStotal, nD});  // 20260406 ZJH 分配输出张量
        float* pOut = result.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pA = a.contiguous().floatDataPtr();  // 20260406 ZJH a 的数据指针
        const float* pB = b.contiguous().floatDataPtr();  // 20260406 ZJH b 的数据指针
        // 20260406 ZJH 逐 batch 拷贝：先 a 的 Sa 个 token，再 b 的 Sb 个 token
        for (int n = 0; n < nBatch; ++n) {
            // 20260406 ZJH 拷贝 a 的部分 [n, 0..Sa-1, :]
            for (int s = 0; s < nSa; ++s)
                for (int d = 0; d < nD; ++d)
                    pOut[(n * nStotal + s) * nD + d] = pA[(n * nSa + s) * nD + d];
            // 20260406 ZJH 拷贝 b 的部分 [n, Sa..Sa+Sb-1, :]
            for (int s = 0; s < nSb; ++s)
                for (int d = 0; d < nD; ++d)
                    pOut[(n * nStotal + nSa + s) * nD + d] = pB[(n * nSb + s) * nD + d];
        }
        return result;  // 20260406 ZJH 返回 [N, Sa+Sb, D] 拼接结果
    }

    // 20260406 ZJH expandCls — 将 CLS token 从 [1,1,D] 广播扩展到 [N,1,D]
    // cls: [1, 1, D] 共享 CLS token
    // nBatch: 目标 batch 大小 N
    // 返回: [N, 1, D] 每个 batch 拥有相同的 CLS token 副本
    Tensor expandCls(const Tensor& cls, int nBatch) {
        auto result = Tensor::zeros({nBatch, 1, m_nEmbedDim});  // 20260402 ZJH 分配输出
        float* pResult = result.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pCls = cls.contiguous().floatDataPtr();  // 20260406 ZJH CLS token 数据指针
        // 20260406 ZJH 将 CLS token 复制到每个 batch 位置
        for (int b = 0; b < nBatch; ++b) {
            for (int d = 0; d < m_nEmbedDim; ++d) {
                pResult[b * m_nEmbedDim + d] = pCls[d];  // 20260402 ZJH 复制到每个 batch
            }
        }
        return result;  // 20260406 ZJH 返回 [N, 1, D] 扩展后的 CLS
    }

    // 20260402 ZJH applyLinearBatched — 对 [N, S, Cin] 批量应用 Linear 得到 [N, S, Cout]
    Tensor applyLinearBatched(const Tensor& input, Linear& linear,
                              int nBatch, int nSeqLen) {
        int nCin = input.shape(2);   // 20260402 ZJH 输入维度
        // 20260402 ZJH reshape [N, S, Cin] → [N*S, Cin] → Linear → [N*S, Cout] → [N, S, Cout]
        auto flat = tensorReshape(input, {nBatch * nSeqLen, nCin});
        auto out = linear.forward(flat);
        int nCout = out.shape(1);  // 20260402 ZJH 输出维度
        return tensorReshape(out, {nBatch, nSeqLen, nCout});
    }

    // 20260402 ZJH applyLayerNorm — 简化 LayerNorm（逐 token 归一化）
    // 使用 Linear 作为可学习 affine 变换（非标准 LN，但在此场景有效）
    Tensor applyLayerNorm(const Tensor& input, Linear& norm,
                          int nBatch, int nSeqLen) {
        // 20260402 ZJH 对每个 token 做 L2 归一化然后过 linear
        auto cInput = input.contiguous();  // 20260406 ZJH 确保连续内存
        int nDim = cInput.shape(2);  // 20260406 ZJH 嵌入维度 D
        auto result = Tensor::zeros(cInput.shapeVec());  // 20260406 ZJH 分配归一化输出
        float* pResult = result.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pInput = cInput.floatDataPtr();  // 20260406 ZJH 输入数据指针

        for (int b = 0; b < nBatch; ++b) {
            for (int s = 0; s < nSeqLen; ++s) {  // 20260406 ZJH 逐 token 归一化
                int nOff = (b * nSeqLen + s) * nDim;  // 20260406 ZJH 当前 token 在连续内存中的起始偏移
                // 20260402 ZJH 计算均值
                float fMean = 0.0f;
                for (int d = 0; d < nDim; ++d) fMean += pInput[nOff + d];
                fMean /= static_cast<float>(nDim);
                // 20260402 ZJH 计算方差
                float fVar = 0.0f;
                for (int d = 0; d < nDim; ++d) {
                    float fDiff = pInput[nOff + d] - fMean;
                    fVar += fDiff * fDiff;
                }
                fVar /= static_cast<float>(nDim);  // 20260406 ZJH 方差 = 平方差和 / D
                float fInvStd = 1.0f / std::sqrt(fVar + 1e-5f);  // 20260406 ZJH 标准差的倒数（加 eps 防除零）
                // 20260402 ZJH 归一化
                for (int d = 0; d < nDim; ++d) {
                    pResult[nOff + d] = (pInput[nOff + d] - fMean) * fInvStd;
                }
            }
        }
        return result;  // 20260406 ZJH 返回归一化后的张量（与输入形状相同）
    }

    // 20260402 ZJH transformerBlock — 单层 Transformer: MHSA + FFN + 残差
    // 20260406 ZJH input: [N, S, D] 当前层输入
    // 20260406 ZJH nLayer: 当前层索引（用于选择对应的权重矩阵）
    // 20260406 ZJH 返回: [N, S, D] 经过自注意力和前馈后的输出
    Tensor transformerBlock(const Tensor& input, int nLayer,
                            int nBatch, int nSeqLen) {
        // 20260402 ZJH 1. Multi-Head Self-Attention
        auto normed = applyLayerNorm(input, m_normFinal, nBatch, nSeqLen);  // 20260402 ZJH Pre-LN
        auto attnOut = multiHeadAttention(normed, nLayer, nBatch, nSeqLen);
        auto x = tensorAdd(input, attnOut);  // 20260402 ZJH 残差连接

        // 20260402 ZJH 2. Feed-Forward Network
        auto normed2 = applyLayerNorm(x, m_normFinal, nBatch, nSeqLen);  // 20260402 ZJH Pre-LN
        auto ffnOut = feedForward(normed2, nLayer, nBatch, nSeqLen);
        x = tensorAdd(x, ffnOut);  // 20260402 ZJH 残差连接

        return x;
    }

    // 20260402 ZJH multiHeadAttention — 多头自注意力
    Tensor multiHeadAttention(const Tensor& input, int nLayer,
                               int nBatch, int nSeqLen) {
        // 20260402 ZJH QKV 投影: [N, S, D] → [N, S, 3D]
        auto qkv = applyLinearBatched(input, *m_vecQKV[nLayer], nBatch, nSeqLen);
        int nD = m_nEmbedDim;
        int nHeadDim = nD / m_nNumHeads;  // 20260402 ZJH 每头维度
        float fScale = 1.0f / std::sqrt(static_cast<float>(nHeadDim));  // 20260402 ZJH 缩放因子

        auto cQKV = qkv.contiguous();
        const float* pQKV = cQKV.floatDataPtr();

        // 20260402 ZJH 计算注意力 + 加权求和
        auto result = Tensor::zeros({nBatch, nSeqLen, nD});
        float* pResult = result.mutableFloatDataPtr();

        // 20260402 ZJH 逐 batch、逐 head 计算
        for (int b = 0; b < nBatch; ++b) {
            for (int h = 0; h < m_nNumHeads; ++h) {
                int nHeadOff = h * nHeadDim;  // 20260402 ZJH 当前 head 在 D 维中的偏移

                // 20260402 ZJH 计算 attention scores: Q * K^T / sqrt(d)
                std::vector<float> vecScores(nSeqLen * nSeqLen, 0.0f);  // 20260402 ZJH [S, S]
                for (int qi = 0; qi < nSeqLen; ++qi) {
                    for (int ki = 0; ki < nSeqLen; ++ki) {
                        float fDot = 0.0f;
                        for (int d = 0; d < nHeadDim; ++d) {
                            int nQIdx = (b * nSeqLen + qi) * (3 * nD) + nHeadOff + d;          // 20260402 ZJH Q
                            int nKIdx = (b * nSeqLen + ki) * (3 * nD) + nD + nHeadOff + d;     // 20260402 ZJH K
                            fDot += pQKV[nQIdx] * pQKV[nKIdx];
                        }
                        vecScores[qi * nSeqLen + ki] = fDot * fScale;  // 20260402 ZJH 缩放后的注意力分数
                    }
                }

                // 20260402 ZJH softmax（逐行）
                for (int qi = 0; qi < nSeqLen; ++qi) {
                    float fMax = vecScores[qi * nSeqLen];
                    for (int ki = 1; ki < nSeqLen; ++ki) {
                        fMax = std::max(fMax, vecScores[qi * nSeqLen + ki]);
                    }
                    float fSum = 0.0f;
                    for (int ki = 0; ki < nSeqLen; ++ki) {
                        vecScores[qi * nSeqLen + ki] = std::exp(vecScores[qi * nSeqLen + ki] - fMax);
                        fSum += vecScores[qi * nSeqLen + ki];
                    }
                    float fInv = 1.0f / (fSum + 1e-10f);
                    for (int ki = 0; ki < nSeqLen; ++ki) {
                        vecScores[qi * nSeqLen + ki] *= fInv;
                    }
                }

                // 20260402 ZJH 加权求和 V
                for (int qi = 0; qi < nSeqLen; ++qi) {
                    for (int d = 0; d < nHeadDim; ++d) {
                        float fVal = 0.0f;
                        for (int ki = 0; ki < nSeqLen; ++ki) {
                            int nVIdx = (b * nSeqLen + ki) * (3 * nD) + 2 * nD + nHeadOff + d;  // 20260402 ZJH V
                            fVal += vecScores[qi * nSeqLen + ki] * pQKV[nVIdx];
                        }
                        pResult[(b * nSeqLen + qi) * nD + nHeadOff + d] = fVal;  // 20260402 ZJH 写入结果
                    }
                }
            }
        }

        // 20260402 ZJH 输出投影
        return applyLinearBatched(result, *m_vecOutProj[nLayer], nBatch, nSeqLen);
    }

    // 20260402 ZJH feedForward — FFN: Linear(D→4D) → GELU → Linear(4D→D)
    Tensor feedForward(const Tensor& input, int nLayer,
                       int nBatch, int nSeqLen) {
        auto h = applyLinearBatched(input, *m_vecFfn1[nLayer], nBatch, nSeqLen);  // 20260402 ZJH D→4D
        // 20260402 ZJH GELU 激活（近似: x * sigmoid(1.702 * x)）
        auto cH = h.contiguous();
        int nTotal = cH.numel();
        auto activated = Tensor::zeros(cH.shapeVec());
        float* pOut = activated.mutableFloatDataPtr();
        const float* pIn = cH.floatDataPtr();
        for (int i = 0; i < nTotal; ++i) {
            float fX = pIn[i];
            float fSigmoid = 1.0f / (1.0f + std::exp(-1.702f * fX));  // 20260402 ZJH sigmoid(1.702x)
            pOut[i] = fX * fSigmoid;  // 20260402 ZJH GELU ≈ x * σ(1.702x)
        }
        return applyLinearBatched(activated, *m_vecFfn2[nLayer], nBatch, nSeqLen);  // 20260402 ZJH 4D→D
    }
};

// =============================================================================
// 20260402 ZJH GaussianDistribution — 多元高斯分布建模（对角协方差）
// 用于全局上下文向量的正常分布建模
// 训练时: 收集正常样本的全局向量，计算均值和方差
// 推理时: 用马氏距离衡量新样本偏离正常分布的程度
// =============================================================================
class GaussianDistribution {
public:
    // 20260402 ZJH fit — 从正常样本的全局上下文向量拟合高斯分布
    // vecFeatures: N 个 [D] 维特征向量
    // 使用 Welford 在线算法，数值稳定
    void fit(const std::vector<std::vector<float>>& vecFeatures) {
        if (vecFeatures.empty()) return;  // 20260402 ZJH 无数据则跳过
        int nDim = static_cast<int>(vecFeatures[0].size());  // 20260402 ZJH 特征维度
        int nN = static_cast<int>(vecFeatures.size());        // 20260402 ZJH 样本数

        m_vecMean.assign(nDim, 0.0f);  // 20260402 ZJH 初始化均值
        m_vecVar.assign(nDim, 0.0f);   // 20260402 ZJH 初始化方差

        // 20260402 ZJH Welford 在线算法计算均值和方差
        std::vector<float> vecM2(nDim, 0.0f);  // 20260402 ZJH 二阶矩累加器
        for (int i = 0; i < nN; ++i) {
            float fDelta = 0.0f;  // 20260402 ZJH 差值
            for (int d = 0; d < nDim; ++d) {
                fDelta = vecFeatures[i][d] - m_vecMean[d];  // 20260402 ZJH x - mean_old
                m_vecMean[d] += fDelta / static_cast<float>(i + 1);  // 20260402 ZJH 更新均值
                float fDelta2 = vecFeatures[i][d] - m_vecMean[d];  // 20260402 ZJH x - mean_new
                vecM2[d] += fDelta * fDelta2;  // 20260402 ZJH 累加 (x-old)(x-new)
            }
        }

        // 20260402 ZJH 计算方差（加 epsilon 防除零）
        for (int d = 0; d < nDim; ++d) {
            m_vecVar[d] = (nN > 1) ? (vecM2[d] / static_cast<float>(nN - 1)) : 1.0f;
            m_vecVar[d] = std::max(m_vecVar[d], 1e-6f);  // 20260402 ZJH 下限裁剪
        }
        m_bFitted = true;  // 20260402 ZJH 标记已拟合
    }

    // 20260402 ZJH mahalanobisDistance — 计算马氏距离（对角协方差版本）
    // vecFeature: [D] 维待测特征向量
    // 返回: 马氏距离（越大越异常）
    float mahalanobisDistance(const std::vector<float>& vecFeature) const {
        if (!m_bFitted || vecFeature.size() != m_vecMean.size()) {
            return 0.0f;  // 20260402 ZJH 未拟合或维度不匹配
        }
        float fDist = 0.0f;  // 20260402 ZJH 距离累加
        int nDim = static_cast<int>(m_vecMean.size());
        for (int d = 0; d < nDim; ++d) {
            float fDiff = vecFeature[d] - m_vecMean[d];  // 20260402 ZJH 偏差
            fDist += (fDiff * fDiff) / m_vecVar[d];       // 20260402 ZJH (x-μ)²/σ² 对角马氏
        }
        return std::sqrt(fDist);  // 20260402 ZJH 返回马氏距离
    }

    // 20260402 ZJH isFitted — 是否已拟合
    bool isFitted() const { return m_bFitted; }

    // 20260402 ZJH mean/var 访问器（用于序列化）
    const std::vector<float>& mean() const { return m_vecMean; }
    const std::vector<float>& variance() const { return m_vecVar; }

    // 20260402 ZJH 设置均值和方差（用于反序列化/加载）
    void setMeanVar(const std::vector<float>& vecMean, const std::vector<float>& vecVar) {
        m_vecMean = vecMean;
        m_vecVar = vecVar;
        m_bFitted = true;
    }

private:
    std::vector<float> m_vecMean;  // 20260402 ZJH 均值向量 [D]
    std::vector<float> m_vecVar;   // 20260402 ZJH 方差向量 [D]（对角协方差）
    bool m_bFitted = false;        // 20260402 ZJH 是否已拟合
};

// =============================================================================
// 20260402 ZJH GCAD — Global Context Anomaly Detection 完整模型
// 双分支架构:
//   局部分支: Teacher-Student 知识蒸馏（检测纹理异常）
//   全局分支: ViT 编码器 + 高斯建模（检测布局异常）
// 训练模式:
//   Phase 1: 用正常样本训练 Teacher（或冻结预训练 backbone）
//   Phase 2: 冻结 Teacher，训练 Student 模仿 Teacher 输出
//   Phase 3: 收集正常样本的全局向量，拟合高斯分布
// 推理模式:
//   输入图像 → 双分支并行 → 局部分数 + 全局分数 → 融合
// =============================================================================

// 20260402 ZJH GCADResult — GCAD 推理结果结构
struct GCADResult {
    float fGlobalScore;                  // 20260402 ZJH 全局异常分数（马氏距离）
    float fLocalScore;                   // 20260402 ZJH 局部异常分数（最大像素异常值）
    float fFusedScore;                   // 20260402 ZJH 融合异常分数
    bool bIsAnomaly;                     // 20260402 ZJH 是否为异常
    bool bIsLayoutAnomaly;               // 20260402 ZJH 是否为布局异常（GCAD 独有！）
    std::vector<float> vecAnomalyMap;    // 20260402 ZJH 逐像素热力图 [H/8, W/8]
    int nMapH;                           // 20260402 ZJH 热力图高度
    int nMapW;                           // 20260402 ZJH 热力图宽度
};

class GCAD : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入图像通道数（默认 3 = RGB）
    // nEmbedDim: 全局编码器嵌入维度（默认 128）
    GCAD(int nInChannels = 3, int nEmbedDim = 128)
        : m_nInChannels(nInChannels), m_nEmbedDim(nEmbedDim),
          m_teacher(nInChannels),     // 20260402 ZJH 教师网络（冻结）
          m_student(nInChannels),     // 20260402 ZJH 学生网络（训练）
          m_globalEncoder(256, nEmbedDim, 4, 3)  // 20260402 ZJH 全局编码器
    {
        // 20260402 ZJH 子网络为值成员（非 shared_ptr），不使用 registerModule
        // 参数收集和 train/eval 切换通过手动覆盖 parameters()/train() 实现
        // （与 EfficientADBackbone 的做法一致）
    }

    // 20260402 ZJH forward — 前向推理（训练和推理共用）
    // input: [N, C, H, W] 输入图像
    // 返回: [N, 1, H/8, W/8] 局部异常热力图（Teacher-Student MSE 差异）
    // 全局异常分数通过 predictGlobal() 单独获取
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH 教师网络提取特征（推理时冻结）
        auto teacherFeatures = m_teacher.forward(input);  // 20260402 ZJH [N, 256, H/8, W/8]
        // 20260402 ZJH 学生网络提取特征
        auto studentFeatures = m_student.forward(input);  // 20260402 ZJH [N, 256, H/8, W/8]

        // 20260402 ZJH 局部异常图 = MSE(teacher - student) 逐通道平均
        auto diff = tensorSub(teacherFeatures, studentFeatures);  // 20260402 ZJH [N, 256, H/8, W/8]
        auto sq = tensorMul(diff, diff);  // 20260402 ZJH 逐元素平方

        // 20260402 ZJH 沿通道维度求均值得到单通道热力图
        auto cSq = sq.contiguous();
        int nBatch = cSq.shape(0);
        int nC = cSq.shape(1);
        int nFH = cSq.shape(2);
        int nFW = cSq.shape(3);
        auto anomalyMap = Tensor::zeros({nBatch, 1, nFH, nFW});  // 20260402 ZJH [N, 1, fH, fW]
        float* pMap = anomalyMap.mutableFloatDataPtr();
        const float* pSq = cSq.floatDataPtr();
        float fInvC = 1.0f / static_cast<float>(nC);  // 20260402 ZJH 通道均值系数
        for (int b = 0; b < nBatch; ++b) {
            for (int h = 0; h < nFH; ++h) {
                for (int w = 0; w < nFW; ++w) {
                    float fSum = 0.0f;
                    for (int c = 0; c < nC; ++c) {
                        fSum += pSq[((b * nC + c) * nFH + h) * nFW + w];
                    }
                    pMap[(b * nFH + h) * nFW + w] = fSum * fInvC;  // 20260402 ZJH 通道均值
                }
            }
        }

        return anomalyMap;  // 20260402 ZJH [N, 1, H/8, W/8] 局部异常热力图
    }

    // 20260402 ZJH predictGlobal — 提取全局上下文向量
    // input: [N, C, H, W] 输入图像
    // 返回: [N, nEmbedDim] 全局上下文向量
    Tensor predictGlobal(const Tensor& input) {
        auto teacherFeatures = m_teacher.forward(input);  // 20260402 ZJH [N, 256, H/8, W/8]
        return m_globalEncoder.forward(teacherFeatures);   // 20260402 ZJH [N, D]
    }

    // 20260402 ZJH predict — 完整推理：局部 + 全局融合
    // input: [1, C, H, W] 单张图像
    // 返回: GCADResult 结构
    GCADResult predict(const Tensor& input) {
        GCADResult result;  // 20260402 ZJH 推理结果

        // 20260402 ZJH 1. 局部异常热力图
        auto anomalyMap = forward(input);  // 20260402 ZJH [1, 1, fH, fW]
        auto cMap = anomalyMap.contiguous();
        int nFH = cMap.shape(2);  // 20260402 ZJH 热力图高
        int nFW = cMap.shape(3);  // 20260402 ZJH 热力图宽
        result.nMapH = nFH;
        result.nMapW = nFW;

        const float* pMap = cMap.floatDataPtr();
        int nSpatial = nFH * nFW;
        result.vecAnomalyMap.resize(nSpatial);
        float fMaxLocal = 0.0f;  // 20260402 ZJH 局部最大异常值
        for (int i = 0; i < nSpatial; ++i) {
            result.vecAnomalyMap[i] = pMap[i];
            fMaxLocal = std::max(fMaxLocal, pMap[i]);
        }
        result.fLocalScore = fMaxLocal;  // 20260402 ZJH 局部异常分数 = 最大像素值

        // 20260402 ZJH 2. 全局异常分数
        auto globalVec = predictGlobal(input);  // 20260402 ZJH [1, D]
        auto cGlobal = globalVec.contiguous();
        const float* pGlobal = cGlobal.floatDataPtr();
        int nDim = cGlobal.shape(1);

        std::vector<float> vecGlobalFeature(nDim);
        for (int d = 0; d < nDim; ++d) {
            vecGlobalFeature[d] = pGlobal[d];
        }

        result.fGlobalScore = m_globalDist.mahalanobisDistance(vecGlobalFeature);  // 20260402 ZJH 马氏距离

        // 20260402 ZJH 3. 融合分数: 加权平均（全局权重更高，因为布局异常更难检测）
        result.fFusedScore = 0.4f * result.fLocalScore + 0.6f * result.fGlobalScore;

        // 20260402 ZJH 4. 异常判定（使用校准阈值）
        result.bIsAnomaly = (result.fFusedScore > m_fAnomalyThreshold);
        result.bIsLayoutAnomaly = (result.fGlobalScore > m_fLayoutThreshold);  // 20260402 ZJH GCAD 独有

        return result;
    }

    // 20260402 ZJH fitGlobalDistribution — 用正常样本拟合全局分布
    // 在训练完成后调用，收集所有正常样本的全局向量
    void fitGlobalDistribution(const std::vector<std::vector<float>>& vecNormalFeatures) {
        m_globalDist.fit(vecNormalFeatures);  // 20260402 ZJH 拟合高斯分布

        // 20260402 ZJH 自适应阈值: mean + 3*sigma (3-sigma 规则)
        if (!vecNormalFeatures.empty()) {
            std::vector<float> vecDists;  // 20260402 ZJH 正常样本的马氏距离
            vecDists.reserve(vecNormalFeatures.size());
            for (const auto& feat : vecNormalFeatures) {
                vecDists.push_back(m_globalDist.mahalanobisDistance(feat));
            }
            // 20260402 ZJH 计算正常分布的均值和标准差
            float fMean = 0.0f;
            for (float d : vecDists) fMean += d;
            fMean /= static_cast<float>(vecDists.size());
            float fStd = 0.0f;
            for (float d : vecDists) fStd += (d - fMean) * (d - fMean);
            fStd = std::sqrt(fStd / static_cast<float>(vecDists.size()));
            m_fLayoutThreshold = fMean + 3.0f * fStd;  // 20260402 ZJH 3-sigma 阈值
        }
    }

    // 20260402 ZJH calibrateThreshold — 校准异常检测阈值
    // vecNormalScores: 正常样本的融合异常分数列表
    void calibrateThreshold(const std::vector<float>& vecNormalScores) {
        if (vecNormalScores.empty()) return;  // 20260406 ZJH 空列表不校准
        float fMean = 0.0f;  // 20260406 ZJH 均值累加器
        for (float s : vecNormalScores) fMean += s;  // 20260406 ZJH 累加所有分数
        fMean /= static_cast<float>(vecNormalScores.size());  // 20260406 ZJH 求均值
        float fStd = 0.0f;  // 20260406 ZJH 方差累加器
        for (float s : vecNormalScores) fStd += (s - fMean) * (s - fMean);  // 20260406 ZJH 累加平方差
        fStd = std::sqrt(fStd / static_cast<float>(vecNormalScores.size()));  // 20260406 ZJH 标准差
        m_fAnomalyThreshold = fMean + 3.0f * fStd;  // 20260402 ZJH 3-sigma 阈值
    }

    // 20260402 ZJH 访问器
    float anomalyThreshold() const { return m_fAnomalyThreshold; }  // 20260406 ZJH 获取融合异常阈值
    float layoutThreshold() const { return m_fLayoutThreshold; }  // 20260406 ZJH 获取布局异常阈值
    void setAnomalyThreshold(float fThreshold) { m_fAnomalyThreshold = fThreshold; }  // 20260406 ZJH 设置融合异常阈值
    void setLayoutThreshold(float fThreshold) { m_fLayoutThreshold = fThreshold; }  // 20260406 ZJH 设置布局异常阈值

    // 20260402 ZJH 获取全局分布（用于序列化保存）
    const GaussianDistribution& globalDistribution() const { return m_globalDist; }  // 20260406 ZJH const 引用
    GaussianDistribution& globalDistribution() { return m_globalDist; }  // 20260406 ZJH 可变引用（用于设置参数）

    // 20260402 ZJH 获取子网络引用（用于冻结 teacher 参数等）
    LocalFeatureExtractor& teacher() { return m_teacher; }  // 20260406 ZJH 获取教师网络引用
    LocalFeatureExtractor& student() { return m_student; }  // 20260406 ZJH 获取学生网络引用
    GlobalContextEncoder& globalEncoder() { return m_globalEncoder; }  // 20260406 ZJH 获取全局编码器引用

    // 20260402 ZJH 重写 parameters() — 收集 teacher + student + globalEncoder 所有参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto a = m_teacher.parameters(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_student.parameters(); v.insert(v.end(), b.begin(), b.end());
        auto c = m_globalEncoder.parameters(); v.insert(v.end(), c.begin(), c.end());
        return v;
    }

    // 20260402 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> v;
        auto makeP = [&](const std::string& s) { return strPrefix.empty() ? s : strPrefix + "." + s; };
        auto a = m_teacher.namedParameters(makeP("teacher")); v.insert(v.end(), a.begin(), a.end());
        auto b = m_student.namedParameters(makeP("student")); v.insert(v.end(), b.begin(), b.end());
        auto c = m_globalEncoder.namedParameters(makeP("global_encoder")); v.insert(v.end(), c.begin(), c.end());
        return v;
    }

    // 20260402 ZJH 重写 buffers()
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;
        auto a = m_teacher.buffers(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_student.buffers(); v.insert(v.end(), b.begin(), b.end());
        auto c = m_globalEncoder.buffers(); v.insert(v.end(), c.begin(), c.end());
        return v;
    }

    // 20260402 ZJH 重写 namedBuffers()
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> v;
        auto makeP = [&](const std::string& s) { return strPrefix.empty() ? s : strPrefix + "." + s; };
        auto a = m_teacher.namedBuffers(makeP("teacher")); v.insert(v.end(), a.begin(), a.end());
        auto b = m_student.namedBuffers(makeP("student")); v.insert(v.end(), b.begin(), b.end());
        auto c = m_globalEncoder.namedBuffers(makeP("global_encoder")); v.insert(v.end(), c.begin(), c.end());
        return v;
    }

    // 20260402 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_teacher.train(bMode);
        m_student.train(bMode);
        m_globalEncoder.train(bMode);
    }

private:
    int m_nInChannels;  // 20260402 ZJH 输入通道数
    int m_nEmbedDim;    // 20260402 ZJH 全局嵌入维度

    // 20260402 ZJH 双分支网络
    LocalFeatureExtractor m_teacher;       // 20260402 ZJH 教师网络（训练后冻结）
    LocalFeatureExtractor m_student;       // 20260402 ZJH 学生网络（训练目标）
    GlobalContextEncoder m_globalEncoder;  // 20260402 ZJH 全局上下文编码器

    // 20260402 ZJH 全局分布模型
    GaussianDistribution m_globalDist;  // 20260402 ZJH 正常样本的全局特征分布

    // 20260402 ZJH 阈值参数
    float m_fAnomalyThreshold = 0.5f;  // 20260402 ZJH 融合异常阈值
    float m_fLayoutThreshold = 3.0f;   // 20260402 ZJH 布局异常阈值（马氏距离）
};

}  // 20260406 ZJH namespace om 结束
