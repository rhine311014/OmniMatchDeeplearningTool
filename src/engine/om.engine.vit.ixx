// 20260320 ZJH Vision Transformer (ViT) 模块 — Phase 5
// 实现 ViT-Tiny：PatchEmbedding + TransformerEncoder + ClassificationHead
// 用于图像分类任务，支持任意输入尺寸（需为 patch_size 整除）
module;

#include <vector>
#include <string>
#include <cmath>

export module om.engine.vit;

// 20260320 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.engine.linear;

export namespace om {

// 20260320 ZJH PatchEmbedding — 将图像分割为 patch 并嵌入到 D 维向量
// 输入: [N, C, H, W] -> 输出: [N, numPatches+1, embedDim]
class PatchEmbedding : public Module {
public:
    PatchEmbedding(int nImgSize, int nPatchSize, int nInChannels, int nEmbedDim)
        : m_nImgSize(nImgSize), m_nPatchSize(nPatchSize),
          m_nInChannels(nInChannels), m_nEmbedDim(nEmbedDim),
          m_proj(nInChannels, nEmbedDim, nPatchSize, nPatchSize, 0, true)
    {
        m_nNumPatches = (nImgSize / nPatchSize) * (nImgSize / nPatchSize);

        // 20260320 ZJH CLS token 和位置编码
        m_clsToken = Tensor::randn({1, 1, nEmbedDim});
        m_posEmbed = Tensor::randn({1, m_nNumPatches + 1, nEmbedDim});
        float fScale = 0.02f;
        float* pCls = m_clsToken.mutableFloatDataPtr();
        for (int i = 0; i < nEmbedDim; ++i) pCls[i] *= fScale;
        float* pPos = m_posEmbed.mutableFloatDataPtr();
        for (int i = 0; i < (m_nNumPatches + 1) * nEmbedDim; ++i) pPos[i] *= fScale;
        registerParameter("cls_token", m_clsToken);
        registerParameter("pos_embed", m_posEmbed);
    }

    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);
        // 20260320 ZJH Conv2d 投影
        auto projected = m_proj.forward(input);
        int nGridH = projected.shape(2);
        int nGridW = projected.shape(3);
        int nNumPatches = nGridH * nGridW;

        // 20260320 ZJH reshape 到 [N, numPatches, embedDim]（转置 [N,D,P] -> [N,P,D]）
        auto cFlat = tensorFlatten(projected, 2).contiguous();
        auto tokens = Tensor::zeros({nBatch, nNumPatches, m_nEmbedDim});
        {
            const float* pSrc = cFlat.floatDataPtr();
            float* pDst = tokens.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int p = 0; p < nNumPatches; ++p)
            for (int d = 0; d < m_nEmbedDim; ++d)
                pDst[(n * nNumPatches + p) * m_nEmbedDim + d] =
                    pSrc[(n * m_nEmbedDim + d) * nNumPatches + p];
        }

        // 20260320 ZJH 拼接 CLS token + 加位置编码
        auto result = Tensor::zeros({nBatch, nNumPatches + 1, m_nEmbedDim});
        {
            const float* pCls = m_clsToken.floatDataPtr();
            const float* pTokens = tokens.floatDataPtr();
            const float* pPos = m_posEmbed.floatDataPtr();
            float* pOut = result.mutableFloatDataPtr();
            int nSeqLen = nNumPatches + 1;
            for (int n = 0; n < nBatch; ++n) {
                for (int d = 0; d < m_nEmbedDim; ++d)
                    pOut[n * nSeqLen * m_nEmbedDim + d] = pCls[d] + pPos[d];
                for (int p = 0; p < nNumPatches; ++p)
                    for (int d = 0; d < m_nEmbedDim; ++d)
                        pOut[(n * nSeqLen + 1 + p) * m_nEmbedDim + d] =
                            pTokens[(n * nNumPatches + p) * m_nEmbedDim + d]
                            + pPos[(1 + p) * m_nEmbedDim + d];
            }
        }
        return result;
    }

    // 20260320 ZJH 手动收集参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_proj.parameters()) vec.push_back(p);
        for (auto* p : Module::parameters()) vec.push_back(p);  // cls_token, pos_embed
        return vec;
    }

private:
    int m_nImgSize, m_nPatchSize, m_nInChannels, m_nEmbedDim, m_nNumPatches;
    Conv2d m_proj;
    Tensor m_clsToken;
    Tensor m_posEmbed;
};

// 20260320 ZJH MultiHeadAttention — 多头注意力模块
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int nEmbedDim, int nHeads)
        : m_nEmbedDim(nEmbedDim), m_nHeads(nHeads),
          m_nHeadDim(nEmbedDim / nHeads),
          m_qkv(nEmbedDim, 3 * nEmbedDim, true),
          m_outProj(nEmbedDim, nEmbedDim, true)
    {
        m_fScale = 1.0f / std::sqrt(static_cast<float>(m_nHeadDim));
    }

    Tensor forward(const Tensor& input) override {
        auto cInput = input.contiguous();
        int nBatch = cInput.shape(0);
        int nSeqLen = cInput.shape(1);
        int nDim = cInput.shape(2);

        // 20260320 ZJH QKV 投影
        // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
        auto flat = tensorReshape(cInput, {nBatch * nSeqLen, nDim});
        auto qkv = m_qkv.forward(flat).contiguous();
        // 20260325 ZJH QKV 拆分需要逐元素访问指针，GPU 张量先迁移到 CPU
        bool bWasCuda = qkv.isCuda();
        auto cpuQkv = bWasCuda ? qkv.cpu() : qkv;
        const float* pQKV = cpuQkv.floatDataPtr();

        // 20260320 ZJH 拆分 Q, K, V 按头重排（CPU 上操作）
        int nBH = nBatch * m_nHeads;
        auto Q = Tensor::zeros({nBH, nSeqLen, m_nHeadDim});
        auto K = Tensor::zeros({nBH, nSeqLen, m_nHeadDim});
        auto V = Tensor::zeros({nBH, nSeqLen, m_nHeadDim});
        float* pQ = Q.mutableFloatDataPtr();
        float* pK = K.mutableFloatDataPtr();
        float* pV = V.mutableFloatDataPtr();

        for (int n = 0; n < nBatch; ++n)
        for (int s = 0; s < nSeqLen; ++s) {
            const float* pRow = pQKV + (n * nSeqLen + s) * 3 * nDim;
            for (int h = 0; h < m_nHeads; ++h)
            for (int d = 0; d < m_nHeadDim; ++d) {
                int nOff = h * m_nHeadDim + d;
                int nBHIdx = n * m_nHeads + h;
                pQ[(nBHIdx * nSeqLen + s) * m_nHeadDim + d] = pRow[nOff];
                pK[(nBHIdx * nSeqLen + s) * m_nHeadDim + d] = pRow[nDim + nOff];
                pV[(nBHIdx * nSeqLen + s) * m_nHeadDim + d] = pRow[2 * nDim + nOff];
            }
        }

        // 20260325 ZJH 若原始在 GPU 上，将 Q/K/V 迁回 GPU 后续 matmul 在 GPU 执行
        if (bWasCuda) {
            Q = Q.cuda(); K = K.cuda(); V = V.cuda();
        }

        // 20260320 ZJH Scaled Dot-Product Attention
        auto kT = tensorTranspose2dBatched(K);
        auto scores = tensorBatchedMatmul(Q, kT);
        // 20260325 ZJH 缩放操作需要指针访问，迁移到 CPU 后再迁回
        {
            auto cpuScores = bWasCuda ? scores.cpu() : scores;
            float* pS = cpuScores.mutableFloatDataPtr();
            for (int i = 0; i < cpuScores.numel(); ++i) pS[i] *= m_fScale;
            scores = bWasCuda ? cpuScores.cuda() : cpuScores;
        }
        auto attnW = tensorSoftmaxLastDim(scores);
        auto context = tensorBatchedMatmul(attnW, V);

        // 20260320 ZJH 重排回 [N, seqLen, embedDim]（需要指针访问，在 CPU 上操作）
        auto cpuContext = bWasCuda ? context.cpu() : context;
        auto outFlat = Tensor::zeros({nBatch * nSeqLen, nDim});
        {
            const float* pCtx = cpuContext.floatDataPtr();
            float* pOut = outFlat.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int s = 0; s < nSeqLen; ++s)
            for (int h = 0; h < m_nHeads; ++h)
            for (int d = 0; d < m_nHeadDim; ++d)
                pOut[(n * nSeqLen + s) * nDim + h * m_nHeadDim + d] =
                    pCtx[((n * m_nHeads + h) * nSeqLen + s) * m_nHeadDim + d];
        }
        // 20260325 ZJH 若原始在 GPU 上，将 outFlat 迁回 GPU
        if (bWasCuda) outFlat = outFlat.cuda();

        auto projected = m_outProj.forward(outFlat);
        // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
        return tensorReshape(projected.contiguous(), {nBatch, nSeqLen, nDim});
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_qkv.parameters()) vec.push_back(p);
        for (auto* p : m_outProj.parameters()) vec.push_back(p);
        return vec;
    }

private:
    int m_nEmbedDim, m_nHeads, m_nHeadDim;
    float m_fScale;
    Linear m_qkv;
    Linear m_outProj;
};

// 20260320 ZJH TransformerBlock — Pre-norm Transformer 编码器块
class TransformerBlock : public Module {
public:
    TransformerBlock(int nEmbedDim, int nHeads, int nMlpDim)
        : m_nEmbedDim(nEmbedDim),
          m_norm1(nEmbedDim), m_attn(nEmbedDim, nHeads),
          m_norm2(nEmbedDim),
          m_fc1(nEmbedDim, nMlpDim, true), m_fc2(nMlpDim, nEmbedDim, true)
    {}

    Tensor forward(const Tensor& input) override {
        auto cInput = input.contiguous();
        int nBatch = cInput.shape(0);
        int nSeqLen = cInput.shape(1);

        // 20260320 ZJH Pre-norm + Attention + Residual
        auto n1 = applyLN(cInput, m_norm1, nBatch, nSeqLen);
        auto attnOut = m_attn.forward(n1);
        auto afterAttn = addResidual(cInput, attnOut, nBatch, nSeqLen);

        // 20260320 ZJH Pre-norm + MLP + Residual
        auto n2 = applyLN(afterAttn, m_norm2, nBatch, nSeqLen);
        auto mlpOut = applyMLP(n2, nBatch, nSeqLen);
        auto result = addResidual(afterAttn, mlpOut, nBatch, nSeqLen);

        return result;
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_norm1.parameters()) vec.push_back(p);
        for (auto* p : m_attn.parameters()) vec.push_back(p);
        for (auto* p : m_norm2.parameters()) vec.push_back(p);
        for (auto* p : m_fc1.parameters()) vec.push_back(p);
        for (auto* p : m_fc2.parameters()) vec.push_back(p);
        return vec;
    }

private:
    // 20260320 ZJH 对 [N, seqLen, dim] 应用 LayerNorm
    // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
    Tensor applyLN(const Tensor& input, LayerNorm& ln, int nBatch, int nSeqLen) {
        auto ci = input.contiguous();
        auto flat = tensorReshape(ci, {nBatch * nSeqLen, m_nEmbedDim});
        auto normed = ln.forward(flat);
        return tensorReshape(normed.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});
    }

    // 20260320 ZJH MLP: fc1 -> GELU -> fc2
    // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
    Tensor applyMLP(const Tensor& input, int nBatch, int nSeqLen) {
        auto ci = input.contiguous();
        auto flat = tensorReshape(ci, {nBatch * nSeqLen, m_nEmbedDim});
        auto h = m_fc1.forward(flat);
        h = m_gelu.forward(h);
        auto out = m_fc2.forward(h);
        return tensorReshape(out.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});
    }

    // 20260320 ZJH 残差加法（3D 张量逐元素加）
    // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
    Tensor addResidual(const Tensor& a, const Tensor& b, int nBatch, int nSeqLen) {
        auto ca = a.contiguous();
        auto cb = b.contiguous();
        auto flatA = tensorReshape(ca, {nBatch * nSeqLen, m_nEmbedDim});
        auto flatB = tensorReshape(cb, {nBatch * nSeqLen, m_nEmbedDim});
        auto sum = tensorAdd(flatA, flatB);
        return tensorReshape(sum.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});
    }

    int m_nEmbedDim;
    LayerNorm m_norm1;
    MultiHeadAttention m_attn;
    LayerNorm m_norm2;
    Linear m_fc1;
    Linear m_fc2;
    GELU m_gelu;
};

// 20260320 ZJH ViT — Vision Transformer 完整网络
// 输入: [N, C, H, W] -> 输出: [N, numClasses]
class ViT : public Module {
public:
    ViT(int nImgSize = 32, int nPatchSize = 4, int nInChannels = 1,
        int nNumClasses = 10, int nEmbedDim = 192, int nDepth = 6,
        int nHeads = 3, int nMlpDim = 384)
        : m_nNumClasses(nNumClasses), m_nEmbedDim(nEmbedDim), m_nDepth(nDepth),
          m_patchEmbed(nImgSize, nPatchSize, nInChannels, nEmbedDim),
          m_normFinal(nEmbedDim),
          m_head(nEmbedDim, nNumClasses, true)
    {
        for (int i = 0; i < nDepth; ++i)
            m_vecBlocks.emplace_back(nEmbedDim, nHeads, nMlpDim);
    }

    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);
        auto tokens = m_patchEmbed.forward(input);
        int nSeqLen = tokens.shape(1);

        // 20260320 ZJH Transformer Encoder
        auto x = tokens;
        for (int i = 0; i < m_nDepth; ++i)
            x = m_vecBlocks[static_cast<size_t>(i)].forward(x);

        // 20260320 ZJH 提取 CLS token
        auto cx = x.contiguous();
        auto cls = Tensor::zeros({nBatch, m_nEmbedDim});
        {
            const float* pX = cx.floatDataPtr();
            float* pC = cls.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n)
            for (int d = 0; d < m_nEmbedDim; ++d)
                pC[n * m_nEmbedDim + d] = pX[n * nSeqLen * m_nEmbedDim + d];
        }

        // 20260320 ZJH Final Norm + Classification Head
        auto normed = m_normFinal.forward(cls);
        return m_head.forward(normed);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_patchEmbed.parameters()) vec.push_back(p);
        for (auto& block : m_vecBlocks)
            for (auto* p : block.parameters()) vec.push_back(p);
        for (auto* p : m_normFinal.parameters()) vec.push_back(p);
        for (auto* p : m_head.parameters()) vec.push_back(p);
        return vec;
    }

private:
    int m_nNumClasses, m_nEmbedDim, m_nDepth;
    PatchEmbedding m_patchEmbed;
    std::vector<TransformerBlock> m_vecBlocks;
    LayerNorm m_normFinal;
    Linear m_head;
};

}  // namespace om
