// 20260320 ZJH Vision Transformer (ViT) 模块 — Phase 5
// 实现 ViT-Tiny：PatchEmbedding + TransformerEncoder + ClassificationHead
// 用于图像分类任务，支持任意输入尺寸（需为 patch_size 整除）
module;

#include <vector>   // 20260406 ZJH std::vector 用于参数列表和 TransformerBlock 列表
#include <string>   // 20260406 ZJH std::string 用于模块命名
#include <cmath>    // 20260406 ZJH std::sqrt 用于注意力缩放因子计算

export module om.engine.vit;  // 20260406 ZJH 导出 ViT 模块接口

// 20260320 ZJH 导入依赖模块
import om.engine.tensor;       // 20260406 ZJH Tensor 数据结构
import om.engine.tensor_ops;   // 20260406 ZJH tensorReshape/tensorAdd/tensorSlice 等运算
import om.engine.module;       // 20260406 ZJH Module 基类、LayerNorm
import om.engine.conv;         // 20260406 ZJH Conv2d 卷积层（用于 patch 投影）
import om.engine.activations;  // 20260406 ZJH GELU 激活函数
import om.engine.linear;       // 20260406 ZJH Linear 全连接层
import om.hal.cuda_backend;  // 20260330 ZJH QKV split/merge GPU kernel

export namespace om {

// 20260320 ZJH PatchEmbedding — 将图像分割为 patch 并嵌入到 D 维向量
// 输入: [N, C, H, W] -> 输出: [N, numPatches+1, embedDim]
class PatchEmbedding : public Module {
public:
    // 20260406 ZJH 构造函数
    // nImgSize: 输入图像尺寸（正方形边长，如 32/224）
    // nPatchSize: patch 大小（如 4 表示 4x4 像素为一个 patch）
    // nInChannels: 输入图像通道数（1=灰度, 3=RGB）
    // nEmbedDim: 嵌入向量维度（patch 投影后的特征维度）
    PatchEmbedding(int nImgSize, int nPatchSize, int nInChannels, int nEmbedDim)
        : m_nImgSize(nImgSize), m_nPatchSize(nPatchSize),  // 20260406 ZJH 保存图像尺寸和 patch 尺寸
          m_nInChannels(nInChannels), m_nEmbedDim(nEmbedDim),  // 20260406 ZJH 保存通道数和嵌入维度
          m_proj(nInChannels, nEmbedDim, nPatchSize, nPatchSize, 0, true)  // 20260406 ZJH 投影卷积: kernel=patchSize, stride=patchSize, 无 padding, 有偏置
    {
        // 20260406 ZJH 计算 patch 总数: (imgSize/patchSize)²
        m_nNumPatches = (nImgSize / nPatchSize) * (nImgSize / nPatchSize);

        // 20260320 ZJH CLS token 和位置编码
        m_clsToken = Tensor::randn({1, 1, nEmbedDim});  // 20260406 ZJH 可学习的分类 token [1, 1, D]
        m_posEmbed = Tensor::randn({1, m_nNumPatches + 1, nEmbedDim});  // 20260406 ZJH 可学习的位置编码 [1, P+1, D]（含 CLS 位置）
        float fScale = 0.02f;  // 20260406 ZJH 缩放因子，防止初始化值过大
        float* pCls = m_clsToken.mutableFloatDataPtr();  // 20260406 ZJH CLS token 数据指针
        for (int i = 0; i < nEmbedDim; ++i) pCls[i] *= fScale;  // 20260406 ZJH 缩放 CLS token 初始值
        float* pPos = m_posEmbed.mutableFloatDataPtr();  // 20260406 ZJH 位置编码数据指针
        for (int i = 0; i < (m_nNumPatches + 1) * nEmbedDim; ++i) pPos[i] *= fScale;  // 20260406 ZJH 缩放位置编码初始值
        registerParameter("cls_token", m_clsToken);  // 20260406 ZJH 注册为可训练参数
        registerParameter("pos_embed", m_posEmbed);  // 20260406 ZJH 注册为可训练参数
    }

    // 20260406 ZJH forward — PatchEmbedding 前向传播
    // input: [N, C, H, W] 输入图像
    // 返回: [N, numPatches+1, embedDim] 包含 CLS token 的 patch 嵌入序列
    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);  // 20260406 ZJH batch 大小
        // 20260320 ZJH Conv2d 投影
        auto projected = m_proj.forward(input);  // 20260406 ZJH [N, C, H, W] → [N, D, gridH, gridW]
        int nGridH = projected.shape(2);  // 20260406 ZJH patch 网格高度
        int nGridW = projected.shape(3);  // 20260406 ZJH patch 网格宽度
        int nNumPatches = nGridH * nGridW;  // 20260406 ZJH 实际 patch 数量

        // 20260320 ZJH reshape 到 [N, numPatches, embedDim]（转置 [N,D,P] -> [N,P,D]）
        // 20260330 ZJH 使用 tensor ops 替代原始指针循环，支持 CUDA 设备
        auto cFlat = tensorFlatten(projected, 2);  // 20260330 ZJH [N, D, P]
        auto tokens = tensorTranspose(cFlat, 1, 2).contiguous();  // 20260330 ZJH [N, P, D]，GPU 安全

        // 20260320 ZJH 拼接 CLS token + 加位置编码
        // 20260330 ZJH 使用 tensor ops 替代原始指针循环，支持 CUDA 设备
        // 20260330 ZJH 扩展 CLS token 到 [N, 1, D]
        auto clsExpanded = tensorReshape(m_clsToken, {1, 1, m_nEmbedDim});  // 20260330 ZJH [1, 1, D]
        // 20260330 ZJH 广播 CLS + 位置编码 cls 部分
        auto posSliceCls = tensorSlice(m_posEmbed, 0, 0, 1);  // 20260330 ZJH [1, D] -> CLS 位置编码
        auto posSliceClsR = tensorReshape(posSliceCls, {1, 1, m_nEmbedDim});  // 20260330 ZJH [1, 1, D]
        auto clsWithPos = tensorAdd(clsExpanded, posSliceClsR);  // 20260330 ZJH [1, 1, D]
        // 20260330 ZJH patch tokens 加位置编码
        auto posSlicePatches = tensorSlice(m_posEmbed, 0, 1, nNumPatches + 1);  // 20260330 ZJH [P, D]
        auto posSlicePatchesR = tensorReshape(posSlicePatches, {1, nNumPatches, m_nEmbedDim});  // 20260330 ZJH [1, P, D]
        auto tokensWithPos = tensorAdd(tokens, posSlicePatchesR);  // 20260330 ZJH 广播加法 [N, P, D]

        // 20260330 ZJH 拼接 [N,1,D] + [N,P,D] => [N, P+1, D]
        // 20260330 ZJH 在 CPU 上完成拼接，然后一次性搬到 GPU（如需要）
        auto cpuClsWithPos = clsWithPos.contiguous();  // 20260330 ZJH [1, 1, D] CLS+位置编码
        auto cpuTok = tokensWithPos.contiguous();  // 20260330 ZJH [N, P, D]
        if (cpuClsWithPos.isCuda()) cpuClsWithPos = cpuClsWithPos.cpu();  // 20260330 ZJH CUDA→CPU
        if (cpuTok.isCuda()) cpuTok = cpuTok.cpu();  // 20260330 ZJH CUDA→CPU
        auto cpuRes = Tensor::zeros({nBatch, nNumPatches + 1, m_nEmbedDim});  // 20260330 ZJH CPU 结果张量
        {
            float* pOut = cpuRes.mutableFloatDataPtr();  // 20260330 ZJH 结果数据指针
            const float* pC = cpuClsWithPos.floatDataPtr();  // 20260330 ZJH CLS token 数据指针
            const float* pT = cpuTok.floatDataPtr();  // 20260330 ZJH patch tokens 数据指针
            int nSeqLen = nNumPatches + 1;  // 20260330 ZJH 序列长度（含 CLS）
            for (int n = 0; n < nBatch; ++n) {
                // 20260330 ZJH 拷贝 CLS token（同一份广播到每个 batch）
                std::memcpy(pOut + n * nSeqLen * m_nEmbedDim,
                            pC,
                            static_cast<size_t>(m_nEmbedDim) * sizeof(float));
                // 20260330 ZJH 拷贝 patch tokens
                std::memcpy(pOut + (n * nSeqLen + 1) * m_nEmbedDim,
                            pT + n * nNumPatches * m_nEmbedDim,
                            static_cast<size_t>(nNumPatches * m_nEmbedDim) * sizeof(float));
            }
        }
        // 20260330 ZJH 如果输入在 CUDA 上，把结果搬到 GPU
        auto result = input.isCuda() ? cpuRes.cuda() : cpuRes;  // 20260406 ZJH 根据输入设备决定输出设备
        return result;  // 20260406 ZJH 返回 [N, P+1, D] 的 patch 嵌入序列
    }

    // 20260320 ZJH 手动收集参数
    // 20260406 ZJH 返回投影卷积参数 + CLS token + 位置编码
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;  // 20260406 ZJH 参数收集容器
        for (auto* p : m_proj.parameters()) vec.push_back(p);  // 20260406 ZJH 投影卷积的权重和偏置
        for (auto* p : Module::parameters()) vec.push_back(p);  // cls_token, pos_embed
        return vec;  // 20260406 ZJH 返回所有可训练参数
    }

private:
    int m_nImgSize;       // 20260406 ZJH 输入图像尺寸（正方形边长）
    int m_nPatchSize;     // 20260406 ZJH patch 大小
    int m_nInChannels;    // 20260406 ZJH 输入通道数
    int m_nEmbedDim;      // 20260406 ZJH 嵌入维度
    int m_nNumPatches;    // 20260406 ZJH patch 总数
    Conv2d m_proj;        // 20260406 ZJH 投影卷积层（将 patch 投影到嵌入空间）
    Tensor m_clsToken;    // 20260406 ZJH 可学习的分类 token [1, 1, D]
    Tensor m_posEmbed;    // 20260406 ZJH 可学习的位置编码 [1, P+1, D]
};

// 20260320 ZJH MultiHeadAttention — 多头注意力模块
// 20260406 ZJH 将输入序列通过 QKV 投影后分头计算 Scaled Dot-Product Attention
class MultiHeadAttention : public Module {
public:
    // 20260406 ZJH 构造函数
    // nEmbedDim: 嵌入维度（必须能被 nHeads 整除）
    // nHeads: 注意力头数
    MultiHeadAttention(int nEmbedDim, int nHeads)
        : m_nEmbedDim(nEmbedDim), m_nHeads(nHeads),  // 20260406 ZJH 保存嵌入维度和头数
          m_nHeadDim(nEmbedDim / nHeads),  // 20260406 ZJH 每个头的维度 d_k = D / H
          m_qkv(nEmbedDim, 3 * nEmbedDim, true),  // 20260406 ZJH QKV 联合投影: D → 3D（含偏置）
          m_outProj(nEmbedDim, nEmbedDim, true)  // 20260406 ZJH 输出投影: D → D（含偏置）
    {
        m_fScale = 1.0f / std::sqrt(static_cast<float>(m_nHeadDim));  // 20260406 ZJH 注意力缩放因子 1/sqrt(d_k)
    }

    // 20260330 ZJH forward — 全 GPU 重写，消除所有 D2H/H2D 拷贝
    // 原实现: 3 次 .cpu() + 3 次 .cuda()（每个 TransformerBlock 6 次 PCIe 传输）
    // 新实现: QKV split/scale/merge 全部由 CUDA kernel 完成，零 D2H
    // 20260406 ZJH input: [B, S, D] 输入序列
    // 20260406 ZJH 返回: [B, S, D] 注意力输出
    Tensor forward(const Tensor& input) override {
        auto cInput = input.contiguous();  // 20260406 ZJH 确保内存连续
        int nBatch = cInput.shape(0);  // 20260406 ZJH batch 大小
        int nSeqLen = cInput.shape(1);  // 20260406 ZJH 序列长度（含 CLS token）
        int nDim = cInput.shape(2);  // 20260406 ZJH 嵌入维度

        // 20260330 ZJH QKV 投影: [B*S, D] → [B*S, 3D]
        auto flat = tensorReshape(cInput, {nBatch * nSeqLen, nDim});  // 20260406 ZJH 展平 batch 和序列维度以适配 Linear
        auto qkv = m_qkv.forward(flat).contiguous();  // 20260406 ZJH 联合投影得到 Q/K/V

        int nBH = nBatch * m_nHeads;  // 20260330 ZJH batch × heads 总数

        // 20260406 ZJH 判断数据是否在 GPU 上，选择对应计算路径
        if (qkv.isCuda()) {
#ifdef OM_HAS_CUDA
            // 20260330 ZJH GPU 路径：单次 kernel 完成 QKV 拆分 + head 重排 + Q 缩放
            auto Q = Tensor::zeros({nBH, nSeqLen, m_nHeadDim}, DeviceType::CUDA);  // 20260406 ZJH Query 张量 [BH, S, d_k]
            auto K = Tensor::zeros({nBH, nSeqLen, m_nHeadDim}, DeviceType::CUDA);  // 20260406 ZJH Key 张量 [BH, S, d_k]
            auto V = Tensor::zeros({nBH, nSeqLen, m_nHeadDim}, DeviceType::CUDA);  // 20260406 ZJH Value 张量 [BH, S, d_k]
            // 20260406 ZJH 调用 CUDA kernel 拆分 QKV 并重排头维度，Q 同时乘以 scale
            CUDABackend::qkvSplitHeads(qkv.floatDataPtr(),
                                        Q.mutableFloatDataPtr(), K.mutableFloatDataPtr(), V.mutableFloatDataPtr(),
                                        nBatch, nSeqLen, m_nHeads, m_nHeadDim, m_fScale);

            // 20260330 ZJH Scaled Dot-Product Attention（Q 已含 scale，无需额外缩放）
            auto kT = tensorTranspose2dBatched(K);  // 20260406 ZJH K 转置: [BH, S, d_k] → [BH, d_k, S]
            auto scores = tensorBatchedMatmul(Q, kT);  // 20260406 ZJH 注意力分数: Q @ K^T → [BH, S, S]
            auto attnW = tensorSoftmaxLastDim(scores);  // 20260406 ZJH Softmax 归一化注意力权重
            auto context = tensorBatchedMatmul(attnW, V).contiguous();  // 20260406 ZJH 加权求和: attn @ V → [BH, S, d_k]

            // 20260330 ZJH Merge heads: [BH, S, d] → [B*S, D]，单次 kernel
            auto outFlat = Tensor::zeros({nBatch * nSeqLen, nDim}, DeviceType::CUDA);  // 20260406 ZJH 合并后结果
            // 20260406 ZJH 调用 CUDA kernel 将多头输出拼接回原始维度
            CUDABackend::mergeHeads(context.floatDataPtr(), outFlat.mutableFloatDataPtr(),
                                     nBatch, nSeqLen, m_nHeads, m_nHeadDim);

            auto projected = m_outProj.forward(outFlat);  // 20260406 ZJH 输出投影: [B*S, D] → [B*S, D]
            return tensorReshape(projected.contiguous(), {nBatch, nSeqLen, nDim});  // 20260406 ZJH 恢复 [B, S, D] 形状
#endif
        }

        // 20260330 ZJH CPU 路径：保留原有指针操作（无 D2H 问题）
        const float* pQKV = qkv.floatDataPtr();  // 20260406 ZJH QKV 联合数据指针
        auto Q = Tensor::zeros({nBH, nSeqLen, m_nHeadDim});  // 20260406 ZJH CPU Query 张量
        auto K = Tensor::zeros({nBH, nSeqLen, m_nHeadDim});  // 20260406 ZJH CPU Key 张量
        auto V = Tensor::zeros({nBH, nSeqLen, m_nHeadDim});  // 20260406 ZJH CPU Value 张量
        float* pQ = Q.mutableFloatDataPtr();  // 20260406 ZJH Q 数据指针
        float* pK = K.mutableFloatDataPtr();  // 20260406 ZJH K 数据指针
        float* pV = V.mutableFloatDataPtr();  // 20260406 ZJH V 数据指针

        // 20260406 ZJH 遍历 batch×seq，拆分 QKV 并重排为多头格式
        for (int n = 0; n < nBatch; ++n)
        for (int s = 0; s < nSeqLen; ++s) {
            const float* pRow = pQKV + (n * nSeqLen + s) * 3 * nDim;  // 20260406 ZJH 当前位置的 QKV 行
            // 20260406 ZJH 遍历每个头和头内维度
            for (int h = 0; h < m_nHeads; ++h)
            for (int d = 0; d < m_nHeadDim; ++d) {
                int nOff = h * m_nHeadDim + d;  // 20260406 ZJH 在 D 维度中的偏移
                int nBHIdx = n * m_nHeads + h;  // 20260406 ZJH batch-head 联合索引
                // 20260330 ZJH Q 乘以 scale 与 GPU 路径一致
                pQ[(nBHIdx * nSeqLen + s) * m_nHeadDim + d] = pRow[nOff] * m_fScale;  // 20260406 ZJH Q 取前 D 维并缩放
                pK[(nBHIdx * nSeqLen + s) * m_nHeadDim + d] = pRow[nDim + nOff];  // 20260406 ZJH K 取中间 D 维
                pV[(nBHIdx * nSeqLen + s) * m_nHeadDim + d] = pRow[2 * nDim + nOff];  // 20260406 ZJH V 取后 D 维
            }
        }

        // 20260330 ZJH Attention 计算（Q 已含 scale）
        auto kT = tensorTranspose2dBatched(K);  // 20260406 ZJH K 转置: [BH, S, d_k] → [BH, d_k, S]
        auto scores = tensorBatchedMatmul(Q, kT);  // 20260406 ZJH 注意力分数: [BH, S, S]
        auto attnW = tensorSoftmaxLastDim(scores);  // 20260406 ZJH Softmax 归一化
        auto context = tensorBatchedMatmul(attnW, V);  // 20260406 ZJH 加权求和: [BH, S, d_k]

        // 20260330 ZJH CPU merge heads
        auto cpuContext = context.contiguous();  // 20260406 ZJH 确保内存连续
        auto outFlat = Tensor::zeros({nBatch * nSeqLen, nDim});  // 20260406 ZJH 合并后的输出 [B*S, D]
        {
            const float* pCtx = cpuContext.floatDataPtr();  // 20260406 ZJH context 数据指针
            float* pOut = outFlat.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
            // 20260406 ZJH 将多头 [BH, S, d_k] 重排为 [B*S, D]（头维度拼接回嵌入维度）
            for (int n = 0; n < nBatch; ++n)
            for (int s = 0; s < nSeqLen; ++s)
            for (int h = 0; h < m_nHeads; ++h)
            for (int d = 0; d < m_nHeadDim; ++d)
                pOut[(n * nSeqLen + s) * nDim + h * m_nHeadDim + d] =
                    pCtx[((n * m_nHeads + h) * nSeqLen + s) * m_nHeadDim + d];
        }

        auto projected = m_outProj.forward(outFlat);  // 20260406 ZJH 输出投影: [B*S, D] → [B*S, D]
        return tensorReshape(projected.contiguous(), {nBatch, nSeqLen, nDim});  // 20260406 ZJH 恢复 [B, S, D] 形状
    }

    // 20260406 ZJH 重写 parameters() 收集 QKV 投影和输出投影的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;  // 20260406 ZJH 参数收集容器
        for (auto* p : m_qkv.parameters()) vec.push_back(p);  // 20260406 ZJH QKV 投影的权重和偏置
        for (auto* p : m_outProj.parameters()) vec.push_back(p);  // 20260406 ZJH 输出投影的权重和偏置
        return vec;  // 20260406 ZJH 返回所有可训练参数
    }

private:
    int m_nEmbedDim;  // 20260406 ZJH 嵌入维度 D
    int m_nHeads;     // 20260406 ZJH 注意力头数 H
    int m_nHeadDim;   // 20260406 ZJH 每个头的维度 d_k = D / H
    float m_fScale;   // 20260406 ZJH 注意力缩放因子 1/sqrt(d_k)
    Linear m_qkv;     // 20260406 ZJH QKV 联合投影层 D → 3D
    Linear m_outProj;  // 20260406 ZJH 输出投影层 D → D
};

// 20260320 ZJH TransformerBlock — Pre-norm Transformer 编码器块
// 20260406 ZJH 结构: LN → Attention → Residual → LN → MLP → Residual
class TransformerBlock : public Module {
public:
    // 20260406 ZJH 构造函数
    // nEmbedDim: 嵌入维度
    // nHeads: 注意力头数
    // nMlpDim: MLP 隐藏层维度（通常为 2*embedDim）
    TransformerBlock(int nEmbedDim, int nHeads, int nMlpDim)
        : m_nEmbedDim(nEmbedDim),  // 20260406 ZJH 保存嵌入维度
          m_norm1(nEmbedDim), m_attn(nEmbedDim, nHeads),  // 20260406 ZJH 第一个 LayerNorm + 多头注意力
          m_norm2(nEmbedDim),  // 20260406 ZJH 第二个 LayerNorm
          m_fc1(nEmbedDim, nMlpDim, true), m_fc2(nMlpDim, nEmbedDim, true)  // 20260406 ZJH MLP 两层全连接（含偏置）
    {}

    // 20260406 ZJH forward — TransformerBlock 前向传播
    // input: [N, S, D] 输入序列
    // 返回: [N, S, D] 经过注意力和 MLP 处理后的序列
    Tensor forward(const Tensor& input) override {
        auto cInput = input.contiguous();  // 20260406 ZJH 确保内存连续
        int nBatch = cInput.shape(0);  // 20260406 ZJH batch 大小
        int nSeqLen = cInput.shape(1);  // 20260406 ZJH 序列长度

        // 20260320 ZJH Pre-norm + Attention + Residual
        auto n1 = applyLN(cInput, m_norm1, nBatch, nSeqLen);  // 20260406 ZJH 第一个 LayerNorm
        auto attnOut = m_attn.forward(n1);  // 20260406 ZJH 多头注意力
        auto afterAttn = addResidual(cInput, attnOut, nBatch, nSeqLen);  // 20260406 ZJH 残差连接

        // 20260320 ZJH Pre-norm + MLP + Residual
        auto n2 = applyLN(afterAttn, m_norm2, nBatch, nSeqLen);  // 20260406 ZJH 第二个 LayerNorm
        auto mlpOut = applyMLP(n2, nBatch, nSeqLen);  // 20260406 ZJH MLP 前馈网络
        auto result = addResidual(afterAttn, mlpOut, nBatch, nSeqLen);  // 20260406 ZJH 残差连接

        return result;  // 20260406 ZJH 返回 Transformer 块输出
    }

    // 20260406 ZJH 重写 parameters() 收集所有子层参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;  // 20260406 ZJH 参数收集容器
        for (auto* p : m_norm1.parameters()) vec.push_back(p);  // 20260406 ZJH LN1 参数（gamma, beta）
        for (auto* p : m_attn.parameters()) vec.push_back(p);  // 20260406 ZJH 多头注意力参数（QKV + outProj）
        for (auto* p : m_norm2.parameters()) vec.push_back(p);  // 20260406 ZJH LN2 参数（gamma, beta）
        for (auto* p : m_fc1.parameters()) vec.push_back(p);  // 20260406 ZJH MLP fc1 参数
        for (auto* p : m_fc2.parameters()) vec.push_back(p);  // 20260406 ZJH MLP fc2 参数
        return vec;  // 20260406 ZJH 返回所有可训练参数
    }

private:
    // 20260320 ZJH 对 [N, seqLen, dim] 应用 LayerNorm
    // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
    // 20260406 ZJH 将 3D 张量展平为 2D 后应用 LayerNorm，再恢复 3D 形状
    Tensor applyLN(const Tensor& input, LayerNorm& ln, int nBatch, int nSeqLen) {
        auto ci = input.contiguous();  // 20260406 ZJH 确保内存连续
        auto flat = tensorReshape(ci, {nBatch * nSeqLen, m_nEmbedDim});  // 20260406 ZJH [N, S, D] → [N*S, D]
        auto normed = ln.forward(flat);  // 20260406 ZJH 逐行 LayerNorm
        return tensorReshape(normed.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});  // 20260406 ZJH 恢复 [N, S, D]
    }

    // 20260320 ZJH MLP: fc1 -> GELU -> fc2
    // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
    // 20260406 ZJH 两层全连接前馈网络，中间用 GELU 激活
    Tensor applyMLP(const Tensor& input, int nBatch, int nSeqLen) {
        auto ci = input.contiguous();  // 20260406 ZJH 确保内存连续
        auto flat = tensorReshape(ci, {nBatch * nSeqLen, m_nEmbedDim});  // 20260406 ZJH [N, S, D] → [N*S, D]
        auto h = m_fc1.forward(flat);  // 20260406 ZJH 第一层全连接: D → MlpDim
        h = m_gelu.forward(h);  // 20260406 ZJH GELU 激活
        auto out = m_fc2.forward(h);  // 20260406 ZJH 第二层全连接: MlpDim → D
        return tensorReshape(out.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});  // 20260406 ZJH 恢复 [N, S, D]
    }

    // 20260320 ZJH 残差加法（3D 张量逐元素加）
    // 20260325 ZJH GPU 安全修复：使用 tensorReshape 零拷贝替代 fromData
    // 20260406 ZJH 将两个 3D 张量展平后逐元素相加，再恢复形状
    Tensor addResidual(const Tensor& a, const Tensor& b, int nBatch, int nSeqLen) {
        auto ca = a.contiguous();  // 20260406 ZJH 确保 a 内存连续
        auto cb = b.contiguous();  // 20260406 ZJH 确保 b 内存连续
        auto flatA = tensorReshape(ca, {nBatch * nSeqLen, m_nEmbedDim});  // 20260406 ZJH 展平 a
        auto flatB = tensorReshape(cb, {nBatch * nSeqLen, m_nEmbedDim});  // 20260406 ZJH 展平 b
        auto sum = tensorAdd(flatA, flatB);  // 20260406 ZJH 逐元素加法
        return tensorReshape(sum.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});  // 20260406 ZJH 恢复 [N, S, D]
    }

    int m_nEmbedDim;              // 20260406 ZJH 嵌入维度
    LayerNorm m_norm1;            // 20260406 ZJH 第一个 LayerNorm（注意力前）
    MultiHeadAttention m_attn;    // 20260406 ZJH 多头注意力模块
    LayerNorm m_norm2;            // 20260406 ZJH 第二个 LayerNorm（MLP 前）
    Linear m_fc1;                 // 20260406 ZJH MLP 第一层全连接 D → MlpDim
    Linear m_fc2;                 // 20260406 ZJH MLP 第二层全连接 MlpDim → D
    GELU m_gelu;                  // 20260406 ZJH GELU 激活函数（无状态）
};

// 20260320 ZJH ViT — Vision Transformer 完整网络
// 输入: [N, C, H, W] -> 输出: [N, numClasses]
// 20260406 ZJH 架构: PatchEmbedding → N 个 TransformerBlock → LayerNorm → Linear 分类头
class ViT : public Module {
public:
    // 20260406 ZJH 构造函数
    // nImgSize: 输入图像尺寸（正方形边长，默认 32）
    // nPatchSize: patch 大小（默认 4）
    // nInChannels: 输入通道数（默认 1 灰度）
    // nNumClasses: 分类类别数（默认 10）
    // nEmbedDim: 嵌入维度（默认 192，ViT-Tiny）
    // nDepth: Transformer 块数量（默认 6）
    // nHeads: 注意力头数（默认 3）
    // nMlpDim: MLP 隐藏层维度（默认 384）
    ViT(int nImgSize = 32, int nPatchSize = 4, int nInChannels = 1,
        int nNumClasses = 10, int nEmbedDim = 192, int nDepth = 6,
        int nHeads = 3, int nMlpDim = 384)
        : m_nNumClasses(nNumClasses), m_nEmbedDim(nEmbedDim), m_nDepth(nDepth),  // 20260406 ZJH 保存网络超参数
          m_patchEmbed(nImgSize, nPatchSize, nInChannels, nEmbedDim),  // 20260406 ZJH Patch 嵌入层
          m_normFinal(nEmbedDim),  // 20260406 ZJH 最终 LayerNorm
          m_head(nEmbedDim, nNumClasses, true)  // 20260406 ZJH 分类头全连接层（含偏置）
    {
        // 20260406 ZJH 构建 nDepth 个 Transformer 编码器块
        for (int i = 0; i < nDepth; ++i)
            m_vecBlocks.emplace_back(nEmbedDim, nHeads, nMlpDim);  // 20260406 ZJH 原地构造 TransformerBlock
    }

    // 20260406 ZJH forward — ViT 前向传播
    // input: [N, C, H, W] 输入图像
    // 返回: [N, nNumClasses] 分类 logits
    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);  // 20260406 ZJH batch 大小
        auto tokens = m_patchEmbed.forward(input);  // 20260406 ZJH 图像 → patch 嵌入序列 [N, P+1, D]
        int nSeqLen = tokens.shape(1);  // 20260406 ZJH 序列长度（含 CLS token）

        // 20260320 ZJH Transformer Encoder
        auto x = tokens;  // 20260406 ZJH 初始化编码器输入
        // 20260406 ZJH 逐层通过 Transformer 编码器块
        for (int i = 0; i < m_nDepth; ++i)
            x = m_vecBlocks[static_cast<size_t>(i)].forward(x);  // 20260406 ZJH 第 i 个 Transformer 块

        // 20260320 ZJH 提取 CLS token
        // 20260330 ZJH 使用 tensorSlice + tensorReshape 替代原始指针循环，支持 CUDA 设备
        auto clsSlice = tensorSlice(x, 1, 0, 1);  // 20260330 ZJH [N, 1, D] — 取序列第 0 个位置（CLS）
        auto cls = tensorReshape(clsSlice.contiguous(), {nBatch, m_nEmbedDim});  // 20260330 ZJH [N, D]

        // 20260320 ZJH Final Norm + Classification Head
        auto normed = m_normFinal.forward(cls);  // 20260406 ZJH 最终 LayerNorm: [N, D] → [N, D]
        return m_head.forward(normed);  // 20260406 ZJH 分类头: [N, D] → [N, nClasses]
    }

    // 20260406 ZJH 重写 parameters() 收集所有子模块参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;  // 20260406 ZJH 参数收集容器
        for (auto* p : m_patchEmbed.parameters()) vec.push_back(p);  // 20260406 ZJH PatchEmbedding 参数
        // 20260406 ZJH 遍历所有 TransformerBlock 的参数
        for (auto& block : m_vecBlocks)
            for (auto* p : block.parameters()) vec.push_back(p);
        for (auto* p : m_normFinal.parameters()) vec.push_back(p);  // 20260406 ZJH 最终 LayerNorm 参数
        for (auto* p : m_head.parameters()) vec.push_back(p);  // 20260406 ZJH 分类头参数
        return vec;  // 20260406 ZJH 返回所有可训练参数
    }

private:
    int m_nNumClasses;    // 20260406 ZJH 分类类别数
    int m_nEmbedDim;      // 20260406 ZJH 嵌入维度
    int m_nDepth;         // 20260406 ZJH Transformer 块数量
    PatchEmbedding m_patchEmbed;              // 20260406 ZJH Patch 嵌入层
    std::vector<TransformerBlock> m_vecBlocks;  // 20260406 ZJH Transformer 编码器块列表
    LayerNorm m_normFinal;                    // 20260406 ZJH 最终 LayerNorm
    Linear m_head;                            // 20260406 ZJH 分类头全连接层
};

}  // namespace om
