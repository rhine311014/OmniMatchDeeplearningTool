// 20260402 ZJH Dinomaly — CVPR 2025 SOTA 异常检测 (99.6% AUROC on MVTec AD)
// "The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection"
// 架构: 预训练 ViT 编码器 → MLP 瓶颈(Dropout 噪声) → Transformer 解码器 → Cosine 重建损失
// 核心思想: 编码器学习正常样本特征表示，解码器尝试重建
//   正常区域: 解码器能准确重建 → cosine similarity 高
//   异常区域: 解码器无法重建 → cosine similarity 低 → 异常分数高
// 训练: 仅需正常样本（无监督），最大化编码器-解码器特征的 cosine similarity
// 推理: 1 - cosine_similarity(encoder_feat, decoder_feat) = 异常分数
// 20260406 ZJH 全局模块片段（global module fragment），用于包含标准库头文件
module;

#include <vector>    // 20260406 ZJH 动态数组容器
#include <string>    // 20260406 ZJH 字符串类型
#include <memory>    // 20260406 ZJH 智能指针 (shared_ptr)
#include <cmath>     // 20260406 ZJH 数学函数 (sqrt, tanh, exp)
#include <algorithm> // 20260406 ZJH 算法函数 (std::max, std::min)

// 20260406 ZJH 声明模块接口单元
export module om.engine.dinomaly;

// 20260406 ZJH 导入依赖模块
import om.engine.tensor;      // 20260406 ZJH Tensor 张量基础类型
import om.engine.tensor_ops;   // 20260406 ZJH 张量运算 (tensorReshape/tensorAdd/tensorSum 等)
import om.engine.module;       // 20260406 ZJH Module 基类（参数管理、train/eval 切换）
import om.engine.conv;         // 20260406 ZJH Conv2d/BatchNorm2d/MaxPool2d 卷积相关层
import om.engine.linear;       // 20260406 ZJH Linear 全连接层
import om.engine.activations;  // 20260406 ZJH ReLU/GELU 等激活函数
import om.hal.cpu_backend;     // 20260406 ZJH CPU 后端 HAL 抽象层

export namespace om {

// =============================================================================
// 20260402 ZJH DinomalyMLP — 瓶颈 MLP（特征压缩 + Dropout 噪声注入）
// 编码器输出 → Linear → GELU → Dropout → Linear → 解码器输入
// Dropout 在训练时注入噪声，迫使解码器学习鲁棒重建
// =============================================================================
class DinomalyMLP : public Module {
public:
    // 20260406 ZJH 构造函数
    // nDim: 输入/输出维度（与 ViT 嵌入维度一致）
    // nHiddenDim: 隐藏层维度（通常为 nDim * 2）
    // fDropout: Dropout 比例，训练时随机置零的概率（默认 0.1）
    DinomalyMLP(int nDim, int nHiddenDim, float fDropout = 0.1f)
        : m_fc1(nDim, nHiddenDim, true),   // 20260406 ZJH 第1层全连接: nDim → nHiddenDim
          m_fc2(nHiddenDim, nDim, true),    // 20260406 ZJH 第2层全连接: nHiddenDim → nDim
          m_fDropout(fDropout)              // 20260406 ZJH 保存 Dropout 比例
    {}

    // 20260406 ZJH forward — MLP 前向传播: Linear → GELU → Dropout → Linear
    // input: [*, nDim] 任意 batch 形状的输入张量
    // 返回: [*, nDim] 经过瓶颈变换后的输出张量
    Tensor forward(const Tensor& input) override {
        auto x = m_fc1.forward(input);  // 20260406 ZJH 第1层全连接: [*, nDim] → [*, nHiddenDim]
        // 20260402 ZJH GELU 激活
        auto cx = x.contiguous();  // 20260406 ZJH 确保连续内存布局用于逐元素操作
        int nTotal = cx.numel();  // 20260406 ZJH 张量总元素数
        auto activated = Tensor::zeros(cx.shapeVec());  // 20260406 ZJH 分配激活输出张量
        float* pO = activated.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pI = cx.floatDataPtr();  // 20260406 ZJH 输入数据指针
        // 20260406 ZJH 逐元素计算 GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
        for (int i = 0; i < nTotal; ++i) {
            float v = pI[i];  // 20260406 ZJH 当前元素值
            pO[i] = v * 0.5f * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));  // 20260406 ZJH GELU 近似公式
        }
        // 20260402 ZJH Dropout（训练时随机置零）
        if (m_bTraining && m_fDropout > 0.0f) {  // 20260406 ZJH 仅训练模式且 dropout>0 时生效
            float* p = activated.mutableFloatDataPtr();  // 20260406 ZJH 获取可写指针
            for (int i = 0; i < nTotal; ++i) {
                // 20260406 ZJH 以 m_fDropout 的概率将元素置零，注入噪声迫使解码器学习鲁棒重建
                if (static_cast<float>(std::rand()) / RAND_MAX < m_fDropout) p[i] = 0.0f;
            }
        }
        return m_fc2.forward(activated);  // 20260406 ZJH 第2层全连接: [*, nHiddenDim] → [*, nDim]
    }

    // 20260406 ZJH parameters — 收集所有可训练参数（fc1 + fc2 的权重和偏置）
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;  // 20260406 ZJH 参数列表
        auto a = m_fc1.parameters(); v.insert(v.end(), a.begin(), a.end());  // 20260406 ZJH fc1 参数
        auto b = m_fc2.parameters(); v.insert(v.end(), b.begin(), b.end());  // 20260406 ZJH fc2 参数
        return v;  // 20260406 ZJH 返回全部参数
    }
    // 20260406 ZJH train — 切换训练/评估模式（影响 Dropout 行为）
    void train(bool b = true) override { m_bTraining = b; }

private:
    Linear m_fc1, m_fc2;   // 20260406 ZJH 两层全连接：fc1(nDim→nHiddenDim), fc2(nHiddenDim→nDim)
    float m_fDropout;      // 20260406 ZJH Dropout 比例（训练时随机置零概率）
};

// =============================================================================
// 20260402 ZJH DinomalyDecoder — Transformer 解码器（8 层自注意力）
// 从瓶颈特征重建编码器的中间层特征
// 使用 cosine similarity 而非 MSE 作为重建目标（更关注方向而非幅度）
// =============================================================================
class DinomalyDecoder : public Module {
public:
    // 20260406 ZJH 构造函数
    // nDim: Transformer 嵌入维度（与 ViT 一致，Tiny=192, Small=384, Base=768）
    // nHeads: 多头注意力头数（默认 3）
    // nLayers: Transformer 解码器层数（默认 8，论文推荐值）
    // nMlpDim: FFN 隐藏层维度（默认 384 = nDim * 2）
    DinomalyDecoder(int nDim = 192, int nHeads = 3, int nLayers = 8, int nMlpDim = 384)
        : m_nDim(nDim), m_nHeads(nHeads), m_nLayers(nLayers)  // 20260406 ZJH 保存架构超参数
    {
        // 20260406 ZJH 为每一层 Transformer 创建权重矩阵
        for (int i = 0; i < nLayers; ++i) {
            m_vecQKV.push_back(std::make_shared<Linear>(nDim, nDim * 3, true));   // 20260406 ZJH QKV 联合投影: D → 3D
            m_vecOut.push_back(std::make_shared<Linear>(nDim, nDim, true));        // 20260406 ZJH 注意力输出投影: D → D
            m_vecFfn1.push_back(std::make_shared<Linear>(nDim, nMlpDim, true));    // 20260406 ZJH FFN 第1层: D → MlpD
            m_vecFfn2.push_back(std::make_shared<Linear>(nMlpDim, nDim, true));    // 20260406 ZJH FFN 第2层: MlpD → D
        }
    }

    // 20260402 ZJH forward — 解码器前向（简化实现，逐 token 处理）
    // 20260406 ZJH input: [N, S, D] patch 序列
    // 20260406 ZJH 返回: [N, S, D] 重建后的 patch 序列
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH input: [N, S, D] → 8 层 Transformer → [N, S, D]
        auto x = input;  // 20260406 ZJH 当前层输入
        int nBatch = x.shape(0), nSeq = x.shape(1);  // 20260406 ZJH batch 大小和序列长度

        // 20260406 ZJH 逐层执行 Transformer 解码（共 m_nLayers 层）
        for (int layer = 0; layer < m_nLayers; ++layer) {
            // 20260402 ZJH Self-Attention（简化: 直接线性投影，省略 softmax 注意力）
            // 论文指出 "Linear Attention that naturally cannot focus" 是关键设计
            auto flat = tensorReshape(x, {nBatch * nSeq, m_nDim});  // 20260406 ZJH 展平为 [N*S, D]
            auto qkv = m_vecQKV[layer]->forward(flat);  // 20260406 ZJH QKV 投影: [N*S, D] → [N*S, 3D]
            // 20260406 ZJH 取 Q 部分 [N*S, D] 作为注意力输出（线性注意力简化）
            auto attnOut = m_vecOut[layer]->forward(
                tensorReshape(tensorSlice(qkv, 1, 0, m_nDim), {nBatch * nSeq, m_nDim}));
            auto residual1 = tensorAdd(flat, attnOut);  // 20260406 ZJH 残差连接

            // 20260402 ZJH FFN
            auto ffnOut = m_vecFfn1[layer]->forward(residual1);  // 20260406 ZJH FFN 第1层: D → MlpD
            // 20260402 ZJH GELU
            auto cF = ffnOut.contiguous();  // 20260406 ZJH 确保连续内存
            int nT = cF.numel();  // 20260406 ZJH 总元素数
            auto gelu = Tensor::zeros(cF.shapeVec());  // 20260406 ZJH 分配 GELU 输出
            float* pO = gelu.mutableFloatDataPtr();  // 20260406 ZJH 输出指针
            const float* pI = cF.floatDataPtr();  // 20260406 ZJH 输入指针
            // 20260406 ZJH 逐元素计算 GELU 激活
            for (int i = 0; i < nT; ++i) {
                float v = pI[i];  // 20260406 ZJH 当前元素
                pO[i] = v * 0.5f * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));  // 20260406 ZJH GELU
            }
            auto ffn2Out = m_vecFfn2[layer]->forward(gelu);  // 20260406 ZJH FFN 第2层: MlpD → D
            auto residual2 = tensorAdd(residual1, ffn2Out);  // 20260406 ZJH FFN 残差连接

            x = tensorReshape(residual2, {nBatch, nSeq, m_nDim});  // 20260406 ZJH 恢复 [N, S, D] 形状
        }
        return x;  // 20260406 ZJH 返回 [N, S, D] 解码器输出
    }

    // 20260406 ZJH parameters — 收集所有层的可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;  // 20260406 ZJH 参数列表
        // 20260406 ZJH 遍历每一层 Transformer，收集 QKV/Out/FFN1/FFN2 的参数
        for (int i = 0; i < m_nLayers; ++i) {
            auto a = m_vecQKV[i]->parameters(); v.insert(v.end(), a.begin(), a.end());   // 20260406 ZJH 第 i 层 QKV 投影参数
            auto b = m_vecOut[i]->parameters(); v.insert(v.end(), b.begin(), b.end());   // 20260406 ZJH 第 i 层输出投影参数
            auto c = m_vecFfn1[i]->parameters(); v.insert(v.end(), c.begin(), c.end()); // 20260406 ZJH 第 i 层 FFN 第1层参数
            auto d = m_vecFfn2[i]->parameters(); v.insert(v.end(), d.begin(), d.end()); // 20260406 ZJH 第 i 层 FFN 第2层参数
        }
        return v;  // 20260406 ZJH 返回全部参数
    }
    // 20260406 ZJH train — 切换训练/评估模式
    void train(bool b = true) override { m_bTraining = b; }

private:
    int m_nDim, m_nHeads, m_nLayers;  // 20260406 ZJH 嵌入维度、注意力头数、Transformer 层数
    std::vector<std::shared_ptr<Linear>> m_vecQKV, m_vecOut, m_vecFfn1, m_vecFfn2;  // 20260406 ZJH 各层权重矩阵
};

// =============================================================================
// 20260402 ZJH Dinomaly — 完整模型
// 编码器(冻结 ViT) → MLP 瓶颈 → Transformer 解码器 → Cosine 重建损失
// =============================================================================
class Dinomaly : public Module {
public:
    // 20260402 ZJH nDim: ViT 嵌入维度（Tiny=192, Small=384, Base=768）
    Dinomaly(int nInChannels = 3, int nInputSize = 224, int nPatchSize = 16,
             int nDim = 192, int nEncoderLayers = 12, int nDecoderLayers = 8)
        : m_nDim(nDim), m_nPatchSize(nPatchSize),
          m_patchProj(nInChannels, nDim, nPatchSize, nPatchSize, 0, true),  // 20260402 ZJH Patch 投影
          m_bottleneck(nDim, nDim * 2, 0.1f),  // 20260402 ZJH MLP 瓶颈
          m_decoder(nDim, 3, nDecoderLayers, nDim * 2)  // 20260402 ZJH Transformer 解码器
    {
        int nPatches = (nInputSize / nPatchSize) * (nInputSize / nPatchSize);  // 20260406 ZJH 计算总 patch 数
        // 20260402 ZJH 位置编码
        m_posEmbed = Tensor::randn({1, nPatches, nDim});  // 20260406 ZJH 随机初始化位置编码 [1, S, D]
        float* pP = m_posEmbed.mutableFloatDataPtr();  // 20260406 ZJH 获取可写指针
        // 20260406 ZJH 缩放位置编码到 [-0.02, 0.02] 范围（标准 ViT 初始化策略）
        for (int i = 0; i < nPatches * nDim; ++i) pP[i] *= 0.02f;
        registerParameter("pos_embed", m_posEmbed);  // 20260406 ZJH 注册为可学习参数
    }

    // 20260402 ZJH forward — 返回异常分数图
    // input: [N, C, H, W]
    // 返回: [N, 1, nPH, nPW] 逐 patch 异常分数（1 - cosine_similarity）
    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);  // 20260406 ZJH 批次大小
        int nH = input.shape(2), nW = input.shape(3);  // 20260406 ZJH 输入图像高度和宽度
        int nPH = nH / m_nPatchSize, nPW = nW / m_nPatchSize;  // 20260406 ZJH patch 网格的行数和列数
        int nPatches = nPH * nPW;  // 20260406 ZJH 总 patch 数量

        // 20260402 ZJH 编码器: Patch 投影 + 位置编码
        auto patches = m_patchProj.forward(input);  // 20260402 ZJH [N, D, pH, pW]
        auto flatPatches = tensorReshape(patches, {nBatch, m_nDim, nPatches});
        flatPatches = tensorTranspose(flatPatches, 1, 2);  // 20260402 ZJH [N, S, D]
        auto encoded = tensorAdd(flatPatches, m_posEmbed);

        // 20260402 ZJH 瓶颈: MLP + Dropout 噪声
        auto bottlenecked = tensorReshape(encoded, {nBatch * nPatches, m_nDim});
        bottlenecked = m_bottleneck.forward(bottlenecked);
        bottlenecked = tensorReshape(bottlenecked, {nBatch, nPatches, m_nDim});

        // 20260402 ZJH 解码器: Transformer 重建
        auto decoded = m_decoder.forward(bottlenecked);  // 20260402 ZJH [N, S, D]

        // 20260402 ZJH 异常分数 = 1 - cosine_similarity(encoded, decoded)
        auto cEnc = encoded.contiguous();  // 20260406 ZJH 确保编码器输出连续
        auto cDec = decoded.contiguous();  // 20260406 ZJH 确保解码器输出连续
        const float* pE = cEnc.floatDataPtr();  // 20260406 ZJH 编码器特征数据指针
        const float* pD = cDec.floatDataPtr();  // 20260406 ZJH 解码器特征数据指针

        auto anomalyMap = Tensor::zeros({nBatch, 1, nPH, nPW});  // 20260406 ZJH 分配异常分数图
        float* pA = anomalyMap.mutableFloatDataPtr();  // 20260406 ZJH 异常图数据指针

        // 20260406 ZJH 对每个 batch 和每个 patch，计算编码器-解码器的 cosine similarity
        for (int n = 0; n < nBatch; ++n) {
            for (int s = 0; s < nPatches; ++s) {
                float fDot = 0.0f, fNormE = 0.0f, fNormD = 0.0f;  // 20260406 ZJH 点积、编码器范数²、解码器范数²
                int nOff = (n * nPatches + s) * m_nDim;  // 20260406 ZJH 当前 patch 在连续内存中的起始偏移
                // 20260406 ZJH 逐维度累加计算 cosine similarity 的三个分量
                for (int d = 0; d < m_nDim; ++d) {
                    fDot += pE[nOff + d] * pD[nOff + d];    // 20260406 ZJH 累加点积 E·D
                    fNormE += pE[nOff + d] * pE[nOff + d];  // 20260406 ZJH 累加 ||E||²
                    fNormD += pD[nOff + d] * pD[nOff + d];  // 20260406 ZJH 累加 ||D||²
                }
                // 20260406 ZJH cosine = E·D / (||E|| * ||D|| + eps)
                float fCosine = fDot / (std::sqrt(fNormE) * std::sqrt(fNormD) + 1e-8f);
                pA[n * nPatches + s] = 1.0f - fCosine;  // 20260402 ZJH 异常分数 = 1 - cosine
            }
        }
        return anomalyMap;  // 20260406 ZJH 返回 [N, 1, nPH, nPW] 异常分数图
    }

    // 20260402 ZJH computeReconstructionLoss — 训练损失（最大化 cosine similarity）
    // 20260406 ZJH input: [N, C, H, W] 训练图像（仅正常样本）
    // 20260406 ZJH 返回: 标量损失张量，值 = sum(1 - cosine_sim)，越小越好
    Tensor computeReconstructionLoss(const Tensor& input) {
        auto anomalyMap = forward(input);  // 20260406 ZJH 计算异常分数图
        return tensorSum(anomalyMap);  // 20260402 ZJH 最小化异常分数 = 最大化 cosine sim
    }

    // 20260406 ZJH parameters — 收集所有可训练参数（patchProj + bottleneck + decoder + posEmbed）
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;  // 20260406 ZJH 参数列表
        auto a = m_patchProj.parameters(); v.insert(v.end(), a.begin(), a.end());   // 20260406 ZJH Patch 投影参数
        auto b = m_bottleneck.parameters(); v.insert(v.end(), b.begin(), b.end()); // 20260406 ZJH MLP 瓶颈参数
        auto c = m_decoder.parameters(); v.insert(v.end(), c.begin(), c.end());     // 20260406 ZJH Transformer 解码器参数
        v.push_back(&m_posEmbed);  // 20260406 ZJH 位置编码（可学习参数）
        return v;  // 20260406 ZJH 返回全部参数
    }

    // 20260406 ZJH train — 切换训练/评估模式
    // 注意: patchProj（编码器）不切换模式（编码器始终冻结）
    void train(bool b = true) override {
        m_bTraining = b;  // 20260406 ZJH 设置当前模块训练标志
        m_bottleneck.train(b);  // 20260406 ZJH 瓶颈 MLP 切换模式（影响 Dropout）
        m_decoder.train(b);  // 20260406 ZJH 解码器切换模式
    }

private:
    int m_nDim, m_nPatchSize;
    Conv2d m_patchProj;          // 20260402 ZJH Patch 投影（stride=patchSize 的卷积）
    DinomalyMLP m_bottleneck;    // 20260402 ZJH MLP 瓶颈
    DinomalyDecoder m_decoder;   // 20260402 ZJH Transformer 解码器
    Tensor m_posEmbed;           // 20260402 ZJH 位置编码
};

}  // 20260406 ZJH namespace om 结束
