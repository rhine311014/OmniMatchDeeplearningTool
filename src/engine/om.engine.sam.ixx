// 20260330 ZJH Segment Anything Model (SAM) 模块 — 轻量级 MobileSAM 风格实现
// 用于交互式分割辅助标注，支持点提示和边界框提示
// 架构: SAMImageEncoder(ViT-Tiny) + SAMPromptEncoder + SAMMaskDecoder
// 参考: Meta SAM / MobileSAM / HikRobot CNNSAMPromptSegmentCpp
module;

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>

export module om.engine.sam;

// 20260330 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.engine.linear;
import om.engine.vit;  // 20260330 ZJH 复用 MultiHeadAttention, TransformerBlock

export namespace om {

// =========================================================================
// 20260330 ZJH SAMImageEncoder — 轻量级 ViT-Tiny 图像编码器
// 输入: [1, 3, imageSize, imageSize] → 输出: [1, 256, gridSize, gridSize]
// 流程: PatchEmbed(16x16) → 4个 TransformerBlock → Reshape → Neck(2x Conv2d)
// 这是 SAM 的 heavy 部分，每张图像只需执行一次
// =========================================================================
class SAMImageEncoder : public Module {
public:
    // 20260330 ZJH 构造函数
    // nImageSize: 输入图像尺寸（1024 或 512），必须为 patchSize 整除
    // nEmbedDim: Transformer 嵌入维度（ViT-Tiny = 192）
    // nOutDim: 输出特征维度（SAM 标准 = 256）
    // nDepth: Transformer 编码器深度（4 = 轻量级）
    // nHeads: 多头注意力头数（3 = ViT-Tiny 标准）
    SAMImageEncoder(int nImageSize = 1024, int nEmbedDim = 192, int nOutDim = 256,
                    int nDepth = 4, int nHeads = 3)
        : m_nImageSize(nImageSize),
          m_nEmbedDim(nEmbedDim),
          m_nOutDim(nOutDim),
          m_nDepth(nDepth),
          m_nPatchSize(16),
          m_nGridSize(nImageSize / 16),  // 20260330 ZJH 1024/16=64, 512/16=32
          m_patchEmbed(nImageSize, 16, 3, nEmbedDim),  // 20260330 ZJH 16x16 patch → embedDim 维
          m_normPost(nEmbedDim),  // 20260330 ZJH Transformer 输出后的 LayerNorm
          m_neck1(nEmbedDim, nOutDim, 1, 1, 0, false),  // 20260330 ZJH 1x1 Conv 投影到 outDim
          m_neckBn1(nOutDim),  // 20260330 ZJH 第一层 Neck 的 BatchNorm
          m_neck2(nOutDim, nOutDim, 3, 1, 1, false),  // 20260330 ZJH 3x3 Conv 细化特征
          m_neckBn2(nOutDim)  // 20260330 ZJH 第二层 Neck 的 BatchNorm
    {
        // 20260330 ZJH 创建 nDepth 个 TransformerBlock
        // MLP 隐藏层维度 = 4 * embedDim（ViT 标准比例）
        int nMlpDim = 4 * nEmbedDim;  // 20260330 ZJH MLP 隐藏维度 768（4 × 192）
        for (int i = 0; i < nDepth; ++i) {
            m_vecBlocks.emplace_back(nEmbedDim, nHeads, nMlpDim);
        }
    }

    // 20260330 ZJH forward — 图像编码前向传播
    // input: [N, 3, imageSize, imageSize] 归一化后的输入图像
    // 返回: [N, outDim, gridSize, gridSize] 图像嵌入特征
    Tensor forward(const Tensor& input) override {
        int nBatch = input.shape(0);  // 20260330 ZJH batch 大小（通常为 1）

        // 20260330 ZJH Step 1: Patch Embedding — [N, 3, H, W] → [N, numPatches+1, embedDim]
        auto tokens = m_patchEmbed.forward(input);
        int nSeqLen = tokens.shape(1);  // 20260330 ZJH 序列长度 = numPatches + 1 (含 CLS token)

        // 20260330 ZJH Step 2: Transformer 编码 — 4 层 TransformerBlock
        auto x = tokens;  // 20260330 ZJH 工作张量
        for (int i = 0; i < m_nDepth; ++i) {
            x = m_vecBlocks[static_cast<size_t>(i)].forward(x);
        }

        // 20260330 ZJH Step 3: 后归一化 — 对每个 token 做 LayerNorm
        auto cx = x.contiguous();
        auto flat = tensorReshape(cx, {nBatch * nSeqLen, m_nEmbedDim});
        auto normed = m_normPost.forward(flat);
        auto normed3d = tensorReshape(normed.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});

        // 20260330 ZJH Step 4: 去掉 CLS token，保留 patch tokens → [N, numPatches, embedDim]
        int nNumPatches = m_nGridSize * m_nGridSize;  // 20260330 ZJH 64*64=4096 或 32*32=1024
        auto patchTokens = Tensor::zeros({nBatch, nNumPatches, m_nEmbedDim});
        {
            const float* pSrc = normed3d.contiguous().floatDataPtr();
            float* pDst = patchTokens.mutableFloatDataPtr();
            // 20260330 ZJH 跳过第 0 个 token (CLS)，拷贝后续 numPatches 个 token
            for (int n = 0; n < nBatch; ++n) {
                const float* pBatchSrc = pSrc + n * nSeqLen * m_nEmbedDim + m_nEmbedDim;  // 20260330 ZJH +embedDim 跳过 CLS
                float* pBatchDst = pDst + n * nNumPatches * m_nEmbedDim;
                for (int i = 0; i < nNumPatches * m_nEmbedDim; ++i) {
                    pBatchDst[i] = pBatchSrc[i];  // 20260330 ZJH 逐元素拷贝 patch tokens
                }
            }
        }

        // 20260330 ZJH Step 5: Reshape 到 2D 空间格式 — [N, numPatches, D] → [N, D, gridH, gridW]
        // 需要转置: [N, P, D] → [N, D, P] → reshape [N, D, gridH, gridW]
        auto spatialFeatures = Tensor::zeros({nBatch, m_nEmbedDim, m_nGridSize, m_nGridSize});
        {
            const float* pSrc = patchTokens.floatDataPtr();
            float* pDst = spatialFeatures.mutableFloatDataPtr();
            // 20260330 ZJH 转置 [N, P, D] → [N, D, gridH, gridW]
            for (int n = 0; n < nBatch; ++n) {
                for (int p = 0; p < nNumPatches; ++p) {
                    for (int d = 0; d < m_nEmbedDim; ++d) {
                        // 20260330 ZJH 源索引: (n, p, d)，目标索引: (n, d, p) 再映射到 (n, d, h, w)
                        pDst[(n * m_nEmbedDim + d) * nNumPatches + p] =
                            pSrc[(n * nNumPatches + p) * m_nEmbedDim + d];
                    }
                }
            }
        }

        // 20260330 ZJH Step 6: Neck — 将 embedDim 通道投影到 outDim 通道
        // Conv1x1(embedDim→outDim) + BN + Conv3x3(outDim→outDim) + BN
        auto neck1Out = m_neck1.forward(spatialFeatures);   // 20260330 ZJH [N, outDim, G, G]
        auto neckBn1Out = m_neckBn1.forward(neck1Out);      // 20260330 ZJH 批归一化
        auto neck2Out = m_neck2.forward(neckBn1Out);         // 20260330 ZJH [N, outDim, G, G]
        auto neckBn2Out = m_neckBn2.forward(neck2Out);       // 20260330 ZJH 批归一化
        return neckBn2Out;  // 20260330 ZJH 返回 [N, 256, gridSize, gridSize]
    }

    // 20260330 ZJH parameters — 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_patchEmbed.parameters()) vec.push_back(p);  // 20260330 ZJH Patch Embedding 参数
        for (auto& block : m_vecBlocks) {
            for (auto* p : block.parameters()) vec.push_back(p);  // 20260330 ZJH Transformer 参数
        }
        for (auto* p : m_normPost.parameters()) vec.push_back(p);  // 20260330 ZJH 后归一化参数
        for (auto* p : m_neck1.parameters()) vec.push_back(p);     // 20260330 ZJH Neck Conv1 参数
        for (auto* p : m_neckBn1.parameters()) vec.push_back(p);   // 20260330 ZJH Neck BN1 参数
        for (auto* p : m_neck2.parameters()) vec.push_back(p);     // 20260330 ZJH Neck Conv2 参数
        for (auto* p : m_neckBn2.parameters()) vec.push_back(p);   // 20260330 ZJH Neck BN2 参数
        return vec;
    }

    // 20260330 ZJH buffers — 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_neckBn1.buffers()) vec.push_back(p);  // 20260330 ZJH BN1 running stats
        for (auto* p : m_neckBn2.buffers()) vec.push_back(p);  // 20260330 ZJH BN2 running stats
        return vec;
    }

    // 20260330 ZJH 获取 grid 尺寸（外部需要用于提示坐标归一化）
    int gridSize() const { return m_nGridSize; }
    int imageSize() const { return m_nImageSize; }

private:
    int m_nImageSize;    // 20260330 ZJH 输入图像尺寸（1024 或 512）
    int m_nEmbedDim;     // 20260330 ZJH Transformer 嵌入维度
    int m_nOutDim;       // 20260330 ZJH 输出特征维度（256）
    int m_nDepth;        // 20260330 ZJH Transformer 深度
    int m_nPatchSize;    // 20260330 ZJH Patch 大小（固定 16）
    int m_nGridSize;     // 20260330 ZJH 特征图空间尺寸（imageSize/patchSize）

    PatchEmbedding m_patchEmbed;             // 20260330 ZJH Patch 嵌入层
    std::vector<TransformerBlock> m_vecBlocks;  // 20260330 ZJH Transformer 编码器块
    LayerNorm m_normPost;                    // 20260330 ZJH 后归一化层
    Conv2d m_neck1;                          // 20260330 ZJH Neck 第一层 1x1 卷积
    BatchNorm2d m_neckBn1;                   // 20260330 ZJH Neck 第一层 BN
    Conv2d m_neck2;                          // 20260330 ZJH Neck 第二层 3x3 卷积
    BatchNorm2d m_neckBn2;                   // 20260330 ZJH Neck 第二层 BN
};


// =========================================================================
// 20260330 ZJH SAMPromptEncoder — 提示编码器
// 将点提示 (x, y, label) 和边界框提示编码为 [nPrompts, embedDim] 的提示 token
// 使用学习的位置编码 + 类型嵌入（前景/背景/框角点）
// =========================================================================
class SAMPromptEncoder : public Module {
public:
    // 20260330 ZJH 构造函数
    // nEmbedDim: 嵌入维度（与图像编码器输出一致 = 256）
    // nImageSize: 输入图像尺寸（用于坐标归一化）
    // nNumPointTypes: 点类型数量（0=背景, 1=前景, 2=左上角, 3=右下角）
    SAMPromptEncoder(int nEmbedDim = 256, int nImageSize = 1024, int nNumPointTypes = 4)
        : m_nEmbedDim(nEmbedDim),
          m_nImageSize(nImageSize),
          m_nNumPointTypes(nNumPointTypes),
          m_coordProj1(2, 128, true),   // 20260330 ZJH 坐标投影第一层: (x,y) → 128 维
          m_coordProj2(128, nEmbedDim, true)  // 20260330 ZJH 坐标投影第二层: 128 → embedDim
    {
        // 20260330 ZJH 为每种点类型创建一个类型嵌入向量
        // 类型 0: 背景点, 1: 前景点, 2: 边界框左上角, 3: 边界框右下角
        m_typeEmbedding = Tensor::randn({nNumPointTypes, nEmbedDim});
        float fScale = 0.02f;  // 20260330 ZJH 小缩放因子初始化
        float* pEmb = m_typeEmbedding.mutableFloatDataPtr();
        for (int i = 0; i < nNumPointTypes * nEmbedDim; ++i) {
            pEmb[i] *= fScale;  // 20260330 ZJH 缩放类型嵌入
        }
        registerParameter("type_embedding", m_typeEmbedding);

        // 20260330 ZJH 不属于任何提示的 "无提示" token（用于掩码解码器的额外输入）
        m_notAPointEmbed = Tensor::zeros({1, nEmbedDim});
        registerParameter("not_a_point_embed", m_notAPointEmbed);
    }

    // 20260330 ZJH forward — 未使用（SAMPromptEncoder 通过专用方法编码）
    Tensor forward(const Tensor& /*input*/) override {
        return Tensor::zeros({1, m_nEmbedDim});  // 20260330 ZJH 占位实现
    }

    // 20260330 ZJH encodePoints — 编码点提示
    // points: [N, 3] 张量，每行 (x, y, label)，label: 0=背景, 1=前景
    // 返回: [N, embedDim] 提示 token
    Tensor encodePoints(const Tensor& points) {
        auto cPoints = points.contiguous();
        int nNumPoints = cPoints.shape(0);  // 20260330 ZJH 点数量
        const float* pPts = cPoints.floatDataPtr();

        // 20260330 ZJH Step 1: 提取归一化坐标 — 将像素坐标归一化到 [0, 1]
        auto coords = Tensor::zeros({nNumPoints, 2});
        float* pCoords = coords.mutableFloatDataPtr();
        std::vector<int> vecLabels(nNumPoints);  // 20260330 ZJH 存储每个点的类型标签
        for (int i = 0; i < nNumPoints; ++i) {
            float fX = pPts[i * 3 + 0];      // 20260330 ZJH 像素 x 坐标
            float fY = pPts[i * 3 + 1];      // 20260330 ZJH 像素 y 坐标
            int nLabel = static_cast<int>(pPts[i * 3 + 2]);  // 20260330 ZJH 点类型标签
            // 20260330 ZJH 归一化到 [0, 1] 范围（相对于图像尺寸）
            pCoords[i * 2 + 0] = fX / static_cast<float>(m_nImageSize);
            pCoords[i * 2 + 1] = fY / static_cast<float>(m_nImageSize);
            vecLabels[static_cast<size_t>(i)] = nLabel;  // 20260330 ZJH 存储标签
        }

        // 20260330 ZJH Step 2: 坐标投影 — 2D 坐标 → embedDim 维嵌入
        // 使用简单的 2 层 MLP: Linear(2→128) + ReLU + Linear(128→embedDim)
        auto h = m_coordProj1.forward(coords);  // 20260330 ZJH [N, 128]
        h = m_relu.forward(h);                    // 20260330 ZJH ReLU 激活
        auto coordEmbed = m_coordProj2.forward(h);  // 20260330 ZJH [N, embedDim]

        // 20260330 ZJH Step 3: 加上类型嵌入 — 区分前景/背景/框角点
        auto result = Tensor::zeros({nNumPoints, m_nEmbedDim});
        {
            const float* pCoordEmb = coordEmbed.contiguous().floatDataPtr();
            const float* pTypeEmb = m_typeEmbedding.contiguous().floatDataPtr();
            float* pOut = result.mutableFloatDataPtr();
            for (int i = 0; i < nNumPoints; ++i) {
                int nType = vecLabels[static_cast<size_t>(i)];  // 20260330 ZJH 当前点的类型
                // 20260330 ZJH 限制类型索引范围防止越界
                nType = std::max(0, std::min(nType, m_nNumPointTypes - 1));
                for (int d = 0; d < m_nEmbedDim; ++d) {
                    // 20260330 ZJH 坐标嵌入 + 类型嵌入 = 最终提示 token
                    pOut[i * m_nEmbedDim + d] =
                        pCoordEmb[i * m_nEmbedDim + d] + pTypeEmb[nType * m_nEmbedDim + d];
                }
            }
        }
        return result;  // 20260330 ZJH 返回 [N, embedDim] 提示 token
    }

    // 20260330 ZJH encodeBox — 编码边界框提示
    // 将 (x1, y1, x2, y2) 编码为 2 个角点提示 token
    // fX1, fY1: 左上角像素坐标; fX2, fY2: 右下角像素坐标
    // 返回: [2, embedDim] 两个角点的提示 token
    Tensor encodeBox(float fX1, float fY1, float fX2, float fY2) {
        // 20260330 ZJH 将边界框表示为 2 个特殊点: 左上角(type=2) + 右下角(type=3)
        auto boxPoints = Tensor::zeros({2, 3});
        float* pData = boxPoints.mutableFloatDataPtr();
        pData[0] = fX1;  pData[1] = fY1;  pData[2] = 2.0f;  // 20260330 ZJH 左上角, type=2
        pData[3] = fX2;  pData[4] = fY2;  pData[5] = 3.0f;  // 20260330 ZJH 右下角, type=3
        return encodePoints(boxPoints);  // 20260330 ZJH 复用点编码逻辑
    }

    // 20260330 ZJH 获取 "无提示" 嵌入（用于解码器输入填充）
    const Tensor& notAPointEmbed() const { return m_notAPointEmbed; }

    // 20260330 ZJH parameters — 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_coordProj1.parameters()) vec.push_back(p);   // 20260330 ZJH 坐标投影层1
        for (auto* p : m_coordProj2.parameters()) vec.push_back(p);   // 20260330 ZJH 坐标投影层2
        for (auto* p : Module::parameters()) vec.push_back(p);        // 20260330 ZJH type_embedding, not_a_point_embed
        return vec;
    }

private:
    int m_nEmbedDim;         // 20260330 ZJH 嵌入维度（256）
    int m_nImageSize;        // 20260330 ZJH 图像尺寸（用于坐标归一化）
    int m_nNumPointTypes;    // 20260330 ZJH 点类型数量（4）

    Linear m_coordProj1;     // 20260330 ZJH 坐标投影第一层 (2→128)
    Linear m_coordProj2;     // 20260330 ZJH 坐标投影第二层 (128→embedDim)
    ReLU m_relu;             // 20260330 ZJH 坐标投影中间激活

    Tensor m_typeEmbedding;    // 20260330 ZJH 点类型嵌入 [numTypes, embedDim]
    Tensor m_notAPointEmbed;   // 20260330 ZJH "无提示" 嵌入 [1, embedDim]
};


// =========================================================================
// 20260330 ZJH SAMMaskDecoder — 轻量级掩码解码器
// 输入: 图像嵌入 [1, 256, G, G] + 提示 token [N, 256]
// 输出: 掩码 [1, 1, 256, 256] + IoU 分数
// 使用交叉注意力融合提示信息与图像特征，然后上采样预测掩码
// =========================================================================
class SAMMaskDecoder : public Module {
public:
    // 20260330 ZJH 构造函数
    // nEmbedDim: 嵌入维度（256）
    // nNumHeads: 交叉注意力头数（2 = 轻量级）
    // nMaskOutSize: 输出掩码尺寸（256）
    SAMMaskDecoder(int nEmbedDim = 256, int nNumHeads = 2, int nMaskOutSize = 256)
        : m_nEmbedDim(nEmbedDim),
          m_nNumHeads(nNumHeads),
          m_nMaskOutSize(nMaskOutSize),
          // 20260330 ZJH 交叉注意力: 提示→图像方向（query=提示, key/value=图像特征）
          m_crossAttn1(nEmbedDim, nNumHeads),
          // 20260330 ZJH 交叉注意力: 图像→提示方向（query=图像, key/value=更新后的提示）
          m_crossAttn2(nEmbedDim, nNumHeads),
          // 20260330 ZJH 提示自注意力（更新提示 token 内部关系）
          m_selfAttn(nEmbedDim, nNumHeads),
          m_norm1(nEmbedDim),  // 20260330 ZJH 交叉注意力1后的归一化
          m_norm2(nEmbedDim),  // 20260330 ZJH 交叉注意力2后的归一化
          m_norm3(nEmbedDim),  // 20260330 ZJH 自注意力后的归一化
          // 20260330 ZJH 掩码预测 MLP: embedDim → embedDim/2 → embedDim/4
          m_maskMlp1(nEmbedDim, nEmbedDim / 2, true),
          m_maskMlp2(nEmbedDim / 2, nEmbedDim / 4, true),
          m_maskMlp3(nEmbedDim / 4, nEmbedDim / 4, true),
          // 20260330 ZJH IoU 预测头: embedDim → 128 → 1
          m_iouMlp1(nEmbedDim, 128, true),
          m_iouMlp2(128, 1, true),
          // 20260330 ZJH 上采样层: 特征图从 gridSize → 2*gridSize → maskOutSize
          m_upConv1(nEmbedDim, nEmbedDim / 4, 2, 2, 0, true),  // 20260330 ZJH 转置卷积 2x 上采样
          m_upBn1(nEmbedDim / 4),  // 20260330 ZJH 上采样后的 BN
          m_upConv2(nEmbedDim / 4, nEmbedDim / 8, 2, 2, 0, true),  // 20260330 ZJH 转置卷积 2x 上采样
          m_upBn2(nEmbedDim / 8)  // 20260330 ZJH 上采样后的 BN
    {
        // 20260330 ZJH 输出 token — 解码器额外输入，用于聚合全局信息生成掩码
        m_outputToken = Tensor::randn({1, nEmbedDim});
        float fScale = 0.02f;  // 20260330 ZJH 小缩放因子
        float* pTok = m_outputToken.mutableFloatDataPtr();
        for (int i = 0; i < nEmbedDim; ++i) pTok[i] *= fScale;
        registerParameter("output_token", m_outputToken);

        // 20260330 ZJH IoU token — 用于预测 IoU 分数
        m_iouToken = Tensor::randn({1, nEmbedDim});
        float* pIou = m_iouToken.mutableFloatDataPtr();
        for (int i = 0; i < nEmbedDim; ++i) pIou[i] *= fScale;
        registerParameter("iou_token", m_iouToken);
    }

    // 20260330 ZJH forward — 未使用（通过 decode 方法调用）
    Tensor forward(const Tensor& /*input*/) override {
        return Tensor::zeros({1});  // 20260330 ZJH 占位实现
    }

    // 20260330 ZJH SAMDecoderOutput — 解码器输出结构
    struct SAMDecoderOutput {
        Tensor mask;       // 20260330 ZJH [1, 1, maskOutSize, maskOutSize] 掩码 logits
        float fIoUScore;   // 20260330 ZJH 预测的 IoU 分数（sigmoid 后）
    };

    // 20260330 ZJH decode — 掩码解码前向传播
    // imageEmbedding: [1, embedDim, gridH, gridW] 图像嵌入
    // promptTokens: [nPrompts, embedDim] 提示 token
    // 返回: SAMDecoderOutput（掩码 + IoU 分数）
    SAMDecoderOutput decode(const Tensor& imageEmbedding, const Tensor& promptTokens) {
        auto cImgEmb = imageEmbedding.contiguous();
        auto cPrompt = promptTokens.contiguous();
        int nBatch = cImgEmb.shape(0);          // 20260330 ZJH batch（通常 1）
        int nEmbDim = cImgEmb.shape(1);         // 20260330 ZJH 嵌入维度（256）
        int nGridH = cImgEmb.shape(2);          // 20260330 ZJH 特征图高度
        int nGridW = cImgEmb.shape(3);          // 20260330 ZJH 特征图宽度
        int nGridSize = nGridH * nGridW;        // 20260330 ZJH 空间 token 数量
        int nNumPrompts = cPrompt.shape(0);     // 20260330 ZJH 提示 token 数量

        // 20260330 ZJH Step 1: 将图像嵌入展平为 token 序列 [1, gridSize, embedDim]
        auto imgTokens = Tensor::zeros({nBatch, nGridSize, m_nEmbedDim});
        {
            const float* pImg = cImgEmb.floatDataPtr();
            float* pOut = imgTokens.mutableFloatDataPtr();
            // 20260330 ZJH 转置: [N, D, H, W] → [N, H*W, D]
            for (int n = 0; n < nBatch; ++n) {
                for (int hw = 0; hw < nGridSize; ++hw) {
                    for (int d = 0; d < m_nEmbedDim; ++d) {
                        pOut[(n * nGridSize + hw) * m_nEmbedDim + d] =
                            pImg[(n * m_nEmbedDim + d) * nGridSize + hw];
                    }
                }
            }
        }

        // 20260330 ZJH Step 2: 构造解码器输入序列
        // 拼接: [outputToken, iouToken, promptTokens] → [1, 2+nPrompts, embedDim]
        int nDecoderSeqLen = 2 + nNumPrompts;  // 20260330 ZJH outputToken + iouToken + 提示
        auto decoderTokens = Tensor::zeros({nBatch, nDecoderSeqLen, m_nEmbedDim});
        {
            const float* pOutTok = m_outputToken.contiguous().floatDataPtr();
            const float* pIouTok = m_iouToken.contiguous().floatDataPtr();
            const float* pPrompt = cPrompt.floatDataPtr();
            float* pDec = decoderTokens.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n) {
                // 20260330 ZJH 第 0 个: outputToken
                for (int d = 0; d < m_nEmbedDim; ++d)
                    pDec[(n * nDecoderSeqLen + 0) * m_nEmbedDim + d] = pOutTok[d];
                // 20260330 ZJH 第 1 个: iouToken
                for (int d = 0; d < m_nEmbedDim; ++d)
                    pDec[(n * nDecoderSeqLen + 1) * m_nEmbedDim + d] = pIouTok[d];
                // 20260330 ZJH 后续: promptTokens
                for (int p = 0; p < nNumPrompts; ++p)
                    for (int d = 0; d < m_nEmbedDim; ++d)
                        pDec[(n * nDecoderSeqLen + 2 + p) * m_nEmbedDim + d] =
                            pPrompt[p * m_nEmbedDim + d];
            }
        }

        // 20260330 ZJH Step 3: 提示自注意力 — 更新解码器 token 内部关系
        auto selfAttnOut = m_selfAttn.forward(decoderTokens);  // 20260330 ZJH [1, 2+N, D]
        auto afterSelfNorm = applyLayerNorm3D(selfAttnOut, m_norm3, nBatch, nDecoderSeqLen);

        // 20260330 ZJH Step 4: 交叉注意力1 — 提示查询图像特征
        // Query=解码器tokens, Key/Value=图像tokens
        // 简化实现: 拼接为 [1, decoderLen+gridSize, D]，MHA 后取前 decoderLen 个
        auto crossIn1 = concatTokens(afterSelfNorm, imgTokens, nBatch, nDecoderSeqLen, nGridSize);
        auto crossOut1 = m_crossAttn1.forward(crossIn1);  // 20260330 ZJH [1, decoderLen+gridSize, D]
        // 20260330 ZJH 提取解码器部分 + 残差连接
        auto decoderAfterCross1 = extractAndResidual(crossOut1, afterSelfNorm, nBatch, nDecoderSeqLen);
        auto normDecoder1 = applyLayerNorm3D(decoderAfterCross1, m_norm1, nBatch, nDecoderSeqLen);

        // 20260330 ZJH Step 5: 交叉注意力2 — 图像查询更新后的提示
        auto crossIn2 = concatTokens(imgTokens, normDecoder1, nBatch, nGridSize, nDecoderSeqLen);
        auto crossOut2 = m_crossAttn2.forward(crossIn2);  // 20260330 ZJH [1, gridSize+decoderLen, D]
        // 20260330 ZJH 提取图像部分 + 残差连接
        auto imgAfterCross = extractAndResidual(crossOut2, imgTokens, nBatch, nGridSize);
        auto normImg = applyLayerNorm3D(imgAfterCross, m_norm2, nBatch, nGridSize);

        // 20260330 ZJH Step 6: 上采样图像特征 — [1, D, G, G] → [1, D/4, 2G, 2G] → [1, D/8, 4G, 4G]
        // 先 reshape 回空间格式
        auto imgSpatial = Tensor::zeros({nBatch, m_nEmbedDim, nGridH, nGridW});
        {
            const float* pSrc = normImg.contiguous().floatDataPtr();
            float* pDst = imgSpatial.mutableFloatDataPtr();
            // 20260330 ZJH 转置: [N, HW, D] → [N, D, H, W]
            for (int n = 0; n < nBatch; ++n)
                for (int hw = 0; hw < nGridSize; ++hw)
                    for (int d = 0; d < m_nEmbedDim; ++d)
                        pDst[(n * m_nEmbedDim + d) * nGridSize + hw] =
                            pSrc[(n * nGridSize + hw) * m_nEmbedDim + d];
        }

        // 20260330 ZJH 转置卷积上采样: [1, 256, G, G] → [1, 64, 2G, 2G] → [1, 32, 4G, 4G]
        auto up1 = m_upConv1.forward(imgSpatial);   // 20260330 ZJH [1, 64, 2G, 2G]
        auto upBn1 = m_upBn1.forward(up1);           // 20260330 ZJH BN
        auto upRelu1 = m_relu.forward(upBn1);        // 20260330 ZJH ReLU
        auto up2 = m_upConv2.forward(upRelu1);       // 20260330 ZJH [1, 32, 4G, 4G]
        auto upBn2 = m_upBn2.forward(up2);            // 20260330 ZJH BN
        auto upRelu2 = m_relu.forward(upBn2);         // 20260330 ZJH ReLU 激活

        int nUpH = upRelu2.shape(2);  // 20260330 ZJH 上采样后的空间尺寸（4*G）
        int nUpW = upRelu2.shape(3);
        int nUpChannels = upRelu2.shape(1);  // 20260330 ZJH 上采样后的通道数（embedDim/8=32）
        int nUpSpatial = nUpH * nUpW;         // 20260330 ZJH 上采样后的空间大小

        // 20260330 ZJH Step 7: 提取 outputToken → MLP → 掩码权重 [1, D/4]
        auto outputTokenResult = extractToken(normDecoder1, 0, nBatch, nDecoderSeqLen);  // 20260330 ZJH [1, D]
        auto maskW1 = m_maskMlp1.forward(outputTokenResult);  // 20260330 ZJH [1, D/2]
        maskW1 = m_relu.forward(maskW1);
        auto maskW2 = m_maskMlp2.forward(maskW1);  // 20260330 ZJH [1, D/4]
        maskW2 = m_relu.forward(maskW2);
        auto maskWeights = m_maskMlp3.forward(maskW2);  // 20260330 ZJH [1, D/4] — 与上采样通道数匹配

        // 20260330 ZJH Step 8: 掩码预测 — 上采样特征与掩码权重做点积
        // upFeatures: [1, upChannels, upH, upW], maskWeights: [1, upChannels]
        // 对每个空间位置: mask(h,w) = sum_c(upFeatures(c,h,w) * maskWeights(c))
        auto maskLogits = Tensor::zeros({nBatch, 1, nUpH, nUpW});
        {
            auto cUp = upRelu2.contiguous();
            auto cMW = maskWeights.contiguous();
            const float* pUp = cUp.floatDataPtr();
            const float* pMW = cMW.floatDataPtr();
            float* pMask = maskLogits.mutableFloatDataPtr();
            for (int n = 0; n < nBatch; ++n) {
                for (int hw = 0; hw < nUpSpatial; ++hw) {
                    float fSum = 0.0f;  // 20260330 ZJH 通道维度点积累加
                    for (int c = 0; c < nUpChannels; ++c) {
                        // 20260330 ZJH upFeatures 布局: [N, C, H, W] → (n*C + c)*HW + hw
                        fSum += pUp[(n * nUpChannels + c) * nUpSpatial + hw] * pMW[n * nUpChannels + c];
                    }
                    pMask[n * nUpSpatial + hw] = fSum;  // 20260330 ZJH 写入掩码 logit
                }
            }
        }

        // 20260330 ZJH Step 9: 如果掩码尺寸不是目标尺寸，需要双线性上采样
        Tensor finalMask = maskLogits;
        if (nUpH != m_nMaskOutSize || nUpW != m_nMaskOutSize) {
            // 20260330 ZJH 计算上采样倍率（向上取整）
            int nScale = (m_nMaskOutSize + nUpH - 1) / nUpH;
            if (nScale > 1) {
                finalMask = tensorUpsampleBilinear(maskLogits, nScale);  // 20260330 ZJH 双线性上采样
            }
            // 20260330 ZJH 裁剪到目标尺寸（如果上采样后略大）
            if (finalMask.shape(2) != m_nMaskOutSize || finalMask.shape(3) != m_nMaskOutSize) {
                auto cropped = Tensor::zeros({nBatch, 1, m_nMaskOutSize, m_nMaskOutSize});
                const float* pSrc = finalMask.contiguous().floatDataPtr();
                float* pDst = cropped.mutableFloatDataPtr();
                int nSrcW = finalMask.shape(3);
                for (int n = 0; n < nBatch; ++n)
                    for (int h = 0; h < m_nMaskOutSize; ++h)
                        for (int w = 0; w < m_nMaskOutSize; ++w)
                            pDst[(n * m_nMaskOutSize + h) * m_nMaskOutSize + w] =
                                pSrc[(n * finalMask.shape(2) + h) * nSrcW + w];
                finalMask = cropped;
            }
        }

        // 20260330 ZJH Step 10: IoU 预测 — 从 iouToken 预测掩码质量分数
        auto iouTokenResult = extractToken(normDecoder1, 1, nBatch, nDecoderSeqLen);  // 20260330 ZJH [1, D]
        auto iouH = m_iouMlp1.forward(iouTokenResult);  // 20260330 ZJH [1, 128]
        iouH = m_relu.forward(iouH);
        auto iouLogit = m_iouMlp2.forward(iouH);  // 20260330 ZJH [1, 1]

        // 20260330 ZJH Sigmoid → IoU 分数 [0, 1]
        float fIoURaw = iouLogit.contiguous().floatDataPtr()[0];
        float fIoUScore = 1.0f / (1.0f + std::exp(-fIoURaw));  // 20260330 ZJH sigmoid

        SAMDecoderOutput output;
        output.mask = finalMask;           // 20260330 ZJH [1, 1, maskOutSize, maskOutSize] logits
        output.fIoUScore = fIoUScore;      // 20260330 ZJH 预测 IoU [0, 1]
        return output;
    }

    // 20260330 ZJH parameters — 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_crossAttn1.parameters()) vec.push_back(p);   // 20260330 ZJH 交叉注意力1
        for (auto* p : m_crossAttn2.parameters()) vec.push_back(p);   // 20260330 ZJH 交叉注意力2
        for (auto* p : m_selfAttn.parameters()) vec.push_back(p);     // 20260330 ZJH 自注意力
        for (auto* p : m_norm1.parameters()) vec.push_back(p);        // 20260330 ZJH LN1
        for (auto* p : m_norm2.parameters()) vec.push_back(p);        // 20260330 ZJH LN2
        for (auto* p : m_norm3.parameters()) vec.push_back(p);        // 20260330 ZJH LN3
        for (auto* p : m_maskMlp1.parameters()) vec.push_back(p);     // 20260330 ZJH 掩码 MLP1
        for (auto* p : m_maskMlp2.parameters()) vec.push_back(p);     // 20260330 ZJH 掩码 MLP2
        for (auto* p : m_maskMlp3.parameters()) vec.push_back(p);     // 20260330 ZJH 掩码 MLP3
        for (auto* p : m_iouMlp1.parameters()) vec.push_back(p);      // 20260330 ZJH IoU MLP1
        for (auto* p : m_iouMlp2.parameters()) vec.push_back(p);      // 20260330 ZJH IoU MLP2
        for (auto* p : m_upConv1.parameters()) vec.push_back(p);      // 20260330 ZJH 上采样 Conv1
        for (auto* p : m_upBn1.parameters()) vec.push_back(p);        // 20260330 ZJH 上采样 BN1
        for (auto* p : m_upConv2.parameters()) vec.push_back(p);      // 20260330 ZJH 上采样 Conv2
        for (auto* p : m_upBn2.parameters()) vec.push_back(p);        // 20260330 ZJH 上采样 BN2
        for (auto* p : Module::parameters()) vec.push_back(p);        // 20260330 ZJH output_token, iou_token
        return vec;
    }

    // 20260330 ZJH buffers — 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_upBn1.buffers()) vec.push_back(p);  // 20260330 ZJH BN1 running stats
        for (auto* p : m_upBn2.buffers()) vec.push_back(p);  // 20260330 ZJH BN2 running stats
        return vec;
    }

private:
    // 20260330 ZJH applyLayerNorm3D — 对 [N, seqLen, dim] 应用 LayerNorm
    Tensor applyLayerNorm3D(const Tensor& input, LayerNorm& ln, int nBatch, int nSeqLen) {
        auto ci = input.contiguous();
        auto flat = tensorReshape(ci, {nBatch * nSeqLen, m_nEmbedDim});
        auto normed = ln.forward(flat);
        return tensorReshape(normed.contiguous(), {nBatch, nSeqLen, m_nEmbedDim});
    }

    // 20260330 ZJH concatTokens — 拼接两组 token 序列
    // a: [N, seqA, D], b: [N, seqB, D] → [N, seqA+seqB, D]
    Tensor concatTokens(const Tensor& a, const Tensor& b,
                        int nBatch, int nSeqA, int nSeqB) {
        auto ca = a.contiguous();
        auto cb = b.contiguous();
        int nTotalSeq = nSeqA + nSeqB;  // 20260330 ZJH 拼接后的序列长度
        auto result = Tensor::zeros({nBatch, nTotalSeq, m_nEmbedDim});
        const float* pA = ca.floatDataPtr();
        const float* pB = cb.floatDataPtr();
        float* pOut = result.mutableFloatDataPtr();
        for (int n = 0; n < nBatch; ++n) {
            // 20260330 ZJH 拷贝序列 A
            for (int s = 0; s < nSeqA; ++s)
                for (int d = 0; d < m_nEmbedDim; ++d)
                    pOut[(n * nTotalSeq + s) * m_nEmbedDim + d] =
                        pA[(n * nSeqA + s) * m_nEmbedDim + d];
            // 20260330 ZJH 拷贝序列 B
            for (int s = 0; s < nSeqB; ++s)
                for (int d = 0; d < m_nEmbedDim; ++d)
                    pOut[(n * nTotalSeq + nSeqA + s) * m_nEmbedDim + d] =
                        pB[(n * nSeqB + s) * m_nEmbedDim + d];
        }
        return result;
    }

    // 20260330 ZJH extractAndResidual — 从拼接的注意力输出中提取前 seqLen 个 token + 残差连接
    // crossOut: [N, totalSeq, D], residual: [N, seqLen, D] → [N, seqLen, D]
    Tensor extractAndResidual(const Tensor& crossOut, const Tensor& residual,
                              int nBatch, int nSeqLen) {
        auto cCross = crossOut.contiguous();
        auto cRes = residual.contiguous();
        int nTotalSeq = crossOut.shape(1);  // 20260330 ZJH 拼接后总序列长度
        auto result = Tensor::zeros({nBatch, nSeqLen, m_nEmbedDim});
        const float* pCross = cCross.floatDataPtr();
        const float* pRes = cRes.floatDataPtr();
        float* pOut = result.mutableFloatDataPtr();
        for (int n = 0; n < nBatch; ++n)
            for (int s = 0; s < nSeqLen; ++s)
                for (int d = 0; d < m_nEmbedDim; ++d)
                    // 20260330 ZJH 注意力输出（前 seqLen 个）+ 残差
                    pOut[(n * nSeqLen + s) * m_nEmbedDim + d] =
                        pCross[(n * nTotalSeq + s) * m_nEmbedDim + d] +
                        pRes[(n * nSeqLen + s) * m_nEmbedDim + d];
        return result;
    }

    // 20260330 ZJH extractToken — 从 [N, seqLen, D] 中提取第 idx 个 token → [N, D]
    Tensor extractToken(const Tensor& tokens, int nIdx, int nBatch, int nSeqLen) {
        auto ct = tokens.contiguous();
        auto result = Tensor::zeros({nBatch, m_nEmbedDim});
        const float* pSrc = ct.floatDataPtr();
        float* pDst = result.mutableFloatDataPtr();
        for (int n = 0; n < nBatch; ++n)
            for (int d = 0; d < m_nEmbedDim; ++d)
                pDst[n * m_nEmbedDim + d] = pSrc[(n * nSeqLen + nIdx) * m_nEmbedDim + d];
        return result;
    }

    int m_nEmbedDim;      // 20260330 ZJH 嵌入维度（256）
    int m_nNumHeads;       // 20260330 ZJH 注意力头数（2）
    int m_nMaskOutSize;    // 20260330 ZJH 输出掩码尺寸（256）

    MultiHeadAttention m_crossAttn1;  // 20260330 ZJH 提示→图像交叉注意力
    MultiHeadAttention m_crossAttn2;  // 20260330 ZJH 图像→提示交叉注意力
    MultiHeadAttention m_selfAttn;    // 20260330 ZJH 提示自注意力
    LayerNorm m_norm1;                // 20260330 ZJH 归一化层1
    LayerNorm m_norm2;                // 20260330 ZJH 归一化层2
    LayerNorm m_norm3;                // 20260330 ZJH 归一化层3

    Linear m_maskMlp1;     // 20260330 ZJH 掩码 MLP 第一层
    Linear m_maskMlp2;     // 20260330 ZJH 掩码 MLP 第二层
    Linear m_maskMlp3;     // 20260330 ZJH 掩码 MLP 第三层
    Linear m_iouMlp1;      // 20260330 ZJH IoU 预测 MLP 第一层
    Linear m_iouMlp2;      // 20260330 ZJH IoU 预测 MLP 第二层

    ConvTranspose2d m_upConv1;  // 20260330 ZJH 转置卷积上采样层1
    BatchNorm2d m_upBn1;        // 20260330 ZJH 上采样 BN1
    ConvTranspose2d m_upConv2;  // 20260330 ZJH 转置卷积上采样层2
    BatchNorm2d m_upBn2;        // 20260330 ZJH 上采样 BN2
    ReLU m_relu;                // 20260330 ZJH ReLU 激活

    Tensor m_outputToken;  // 20260330 ZJH 输出 token [1, embedDim]
    Tensor m_iouToken;     // 20260330 ZJH IoU token [1, embedDim]
};


// =========================================================================
// 20260330 ZJH SAMResult — SAM 预测结果结构
// =========================================================================
struct SAMResult {
    Tensor mask;       // 20260330 ZJH [1, 1, 256, 256] 掩码 logits（>0 为前景）
    float fIoUScore;   // 20260330 ZJH 预测的 IoU 分数 [0, 1]
};


// =========================================================================
// 20260330 ZJH SAM — Segment Anything Model 主模型
// 组合 SAMImageEncoder + SAMPromptEncoder + SAMMaskDecoder
// 支持点提示和边界框提示的交互式分割
// =========================================================================
class SAM : public Module {
public:
    // 20260330 ZJH 构造函数
    // nImageSize: 输入图像尺寸（1024=标准, 512=快速处理）
    // nEmbedDim: 图像/掩码嵌入维度（256 = SAM 标准）
    // nViTEmbedDim: ViT 内部嵌入维度（192 = ViT-Tiny）
    // nViTDepth: ViT Transformer 深度（4 = 轻量级）
    // nViTHeads: ViT 注意力头数（3 = ViT-Tiny）
    // nDecoderHeads: 解码器注意力头数（2 = 轻量级）
    SAM(int nImageSize = 1024, int nEmbedDim = 256, int nViTEmbedDim = 192,
        int nViTDepth = 4, int nViTHeads = 3, int nDecoderHeads = 2)
        : m_nImageSize(nImageSize),
          m_nEmbedDim(nEmbedDim),
          m_encoder(nImageSize, nViTEmbedDim, nEmbedDim, nViTDepth, nViTHeads),
          m_promptEncoder(nEmbedDim, nImageSize),
          m_decoder(nEmbedDim, nDecoderHeads, 256)  // 20260330 ZJH 输出掩码固定 256x256
    {}

    // 20260330 ZJH forward — 未使用（SAM 通过 encodeImage + predict 两阶段调用）
    Tensor forward(const Tensor& /*input*/) override {
        return Tensor::zeros({1});  // 20260330 ZJH 占位实现
    }

    // 20260330 ZJH encodeImage — 编码图像（heavy 操作，每张图只需调用一次）
    // image: [1, 3, imageSize, imageSize] 归一化后的输入图像
    // 返回: [1, 256, gridSize, gridSize] 图像嵌入特征
    Tensor encodeImage(const Tensor& image) {
        return m_encoder.forward(image);  // 20260330 ZJH 委托给图像编码器
    }

    // 20260330 ZJH predict — 从点提示生成掩码（轻量操作，可多次调用）
    // imageEmbedding: encodeImage() 的输出 [1, 256, G, G]
    // points: [N, 3] 每行 (x, y, label)，label: 0=背景, 1=前景
    // 返回: SAMResult（掩码 + IoU 分数）
    SAMResult predict(const Tensor& imageEmbedding, const Tensor& points) {
        // 20260330 ZJH Step 1: 编码点提示 → [N, 256] 提示 token
        auto promptTokens = m_promptEncoder.encodePoints(points);
        // 20260330 ZJH Step 2: 解码器生成掩码
        auto decoderOut = m_decoder.decode(imageEmbedding, promptTokens);
        // 20260330 ZJH 封装为 SAMResult
        SAMResult result;
        result.mask = decoderOut.mask;
        result.fIoUScore = decoderOut.fIoUScore;
        return result;
    }

    // 20260330 ZJH predictFromBox — 从边界框提示生成掩码
    // imageEmbedding: encodeImage() 的输出 [1, 256, G, G]
    // fX1, fY1: 左上角像素坐标
    // fX2, fY2: 右下角像素坐标
    // 返回: SAMResult（掩码 + IoU 分数）
    SAMResult predictFromBox(const Tensor& imageEmbedding,
                             float fX1, float fY1, float fX2, float fY2) {
        // 20260330 ZJH Step 1: 编码边界框提示 → [2, 256] 两个角点 token
        auto promptTokens = m_promptEncoder.encodeBox(fX1, fY1, fX2, fY2);
        // 20260330 ZJH Step 2: 解码器生成掩码
        auto decoderOut = m_decoder.decode(imageEmbedding, promptTokens);
        // 20260330 ZJH 封装为 SAMResult
        SAMResult result;
        result.mask = decoderOut.mask;
        result.fIoUScore = decoderOut.fIoUScore;
        return result;
    }

    // 20260330 ZJH predictWithPointsAndBox — 混合提示：点 + 边界框
    // imageEmbedding: encodeImage() 的输出
    // points: [N, 3] 点提示
    // fX1, fY1, fX2, fY2: 边界框坐标
    // 返回: SAMResult（掩码 + IoU 分数）
    SAMResult predictWithPointsAndBox(const Tensor& imageEmbedding,
                                      const Tensor& points,
                                      float fX1, float fY1, float fX2, float fY2) {
        auto cPoints = points.contiguous();
        int nNumPoints = cPoints.shape(0);  // 20260330 ZJH 点提示数量

        // 20260330 ZJH 合并点提示 + 框角点为一个提示张量 [N+2, 3]
        auto combined = Tensor::zeros({nNumPoints + 2, 3});
        float* pComb = combined.mutableFloatDataPtr();

        // 20260330 ZJH 拷贝点提示
        const float* pPts = cPoints.floatDataPtr();
        for (int i = 0; i < nNumPoints * 3; ++i) {
            pComb[i] = pPts[i];
        }

        // 20260330 ZJH 添加框角点（type=2 左上, type=3 右下）
        pComb[nNumPoints * 3 + 0] = fX1;  pComb[nNumPoints * 3 + 1] = fY1;  pComb[nNumPoints * 3 + 2] = 2.0f;
        pComb[nNumPoints * 3 + 3] = fX2;  pComb[nNumPoints * 3 + 4] = fY2;  pComb[nNumPoints * 3 + 5] = 3.0f;

        // 20260330 ZJH 编码合并后的提示 → [N+2, 256]
        auto promptTokens = m_promptEncoder.encodePoints(combined);
        auto decoderOut = m_decoder.decode(imageEmbedding, promptTokens);
        SAMResult result;
        result.mask = decoderOut.mask;
        result.fIoUScore = decoderOut.fIoUScore;
        return result;
    }

    // 20260330 ZJH parameters — 收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_encoder.parameters()) vec.push_back(p);         // 20260330 ZJH 图像编码器
        for (auto* p : m_promptEncoder.parameters()) vec.push_back(p);   // 20260330 ZJH 提示编码器
        for (auto* p : m_decoder.parameters()) vec.push_back(p);         // 20260330 ZJH 掩码解码器
        return vec;
    }

    // 20260330 ZJH buffers — 收集所有 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vec;
        for (auto* p : m_encoder.buffers()) vec.push_back(p);   // 20260330 ZJH 编码器 BN stats
        for (auto* p : m_decoder.buffers()) vec.push_back(p);   // 20260330 ZJH 解码器 BN stats
        return vec;
    }

    // 20260330 ZJH 访问器 — 获取子模块引用（用于单独加载权重或调试）
    SAMImageEncoder& encoder() { return m_encoder; }
    SAMPromptEncoder& promptEncoder() { return m_promptEncoder; }
    SAMMaskDecoder& decoder() { return m_decoder; }

    // 20260330 ZJH 获取配置信息
    int imageSize() const { return m_nImageSize; }
    int embedDim() const { return m_nEmbedDim; }

private:
    int m_nImageSize;               // 20260330 ZJH 输入图像尺寸
    int m_nEmbedDim;                // 20260330 ZJH 嵌入维度

    SAMImageEncoder m_encoder;      // 20260330 ZJH 图像编码器
    SAMPromptEncoder m_promptEncoder;  // 20260330 ZJH 提示编码器
    SAMMaskDecoder m_decoder;       // 20260330 ZJH 掩码解码器
};


// =========================================================================
// 20260330 ZJH 工具函数 — SAM 后处理
// =========================================================================

// 20260330 ZJH upsampleMask — 将 256x256 掩码上采样到原图尺寸
// mask: [1, 1, 256, 256] 掩码 logits
// nOrigH: 原始图像高度（像素）
// nOrigW: 原始图像宽度（像素）
// 返回: [1, 1, nOrigH, nOrigW] 上采样后的掩码 logits
Tensor upsampleMask(const Tensor& mask, int nOrigH, int nOrigW) {
    auto cMask = mask.contiguous();
    int nMaskH = cMask.shape(2);  // 20260330 ZJH 原始掩码高度（256）
    int nMaskW = cMask.shape(3);  // 20260330 ZJH 原始掩码宽度（256）

    // 20260330 ZJH 双线性插值上采样到目标尺寸
    auto result = Tensor::zeros({1, 1, nOrigH, nOrigW});
    const float* pSrc = cMask.floatDataPtr();
    float* pDst = result.mutableFloatDataPtr();

    // 20260330 ZJH 缩放因子（目标→源映射）
    float fScaleH = static_cast<float>(nMaskH) / static_cast<float>(nOrigH);
    float fScaleW = static_cast<float>(nMaskW) / static_cast<float>(nOrigW);

    for (int h = 0; h < nOrigH; ++h) {
        for (int w = 0; w < nOrigW; ++w) {
            // 20260330 ZJH 目标像素映射到源掩码坐标
            float fSrcH = (static_cast<float>(h) + 0.5f) * fScaleH - 0.5f;
            float fSrcW = (static_cast<float>(w) + 0.5f) * fScaleW - 0.5f;

            // 20260330 ZJH 双线性插值四个邻居
            int nH0 = static_cast<int>(std::floor(fSrcH));
            int nW0 = static_cast<int>(std::floor(fSrcW));
            int nH1 = nH0 + 1;
            int nW1 = nW0 + 1;
            float fDh = fSrcH - static_cast<float>(nH0);  // 20260330 ZJH 垂直方向小数部分
            float fDw = fSrcW - static_cast<float>(nW0);  // 20260330 ZJH 水平方向小数部分

            // 20260330 ZJH 边界裁剪（clamp 到有效范围）
            nH0 = std::max(0, std::min(nH0, nMaskH - 1));
            nH1 = std::max(0, std::min(nH1, nMaskH - 1));
            nW0 = std::max(0, std::min(nW0, nMaskW - 1));
            nW1 = std::max(0, std::min(nW1, nMaskW - 1));

            // 20260330 ZJH 读取四个邻居值
            float fV00 = pSrc[nH0 * nMaskW + nW0];  // 20260330 ZJH 左上
            float fV01 = pSrc[nH0 * nMaskW + nW1];  // 20260330 ZJH 右上
            float fV10 = pSrc[nH1 * nMaskW + nW0];  // 20260330 ZJH 左下
            float fV11 = pSrc[nH1 * nMaskW + nW1];  // 20260330 ZJH 右下

            // 20260330 ZJH 双线性插值计算
            float fVal = fV00 * (1.0f - fDh) * (1.0f - fDw)
                       + fV01 * (1.0f - fDh) * fDw
                       + fV10 * fDh * (1.0f - fDw)
                       + fV11 * fDh * fDw;

            pDst[h * nOrigW + w] = fVal;  // 20260330 ZJH 写入目标像素
        }
    }
    return result;  // 20260330 ZJH 返回上采样后的掩码
}

// 20260330 ZJH binarizeMask — 将掩码 logits 二值化为 uint8 掩码
// mask: [1, 1, H, W] 掩码 logits（sigmoid 之前）
// fThreshold: 二值化阈值（默认 0.0，对应 sigmoid=0.5）
// 返回: H*W 的 uint8 向量，255=前景, 0=背景
std::vector<uint8_t> binarizeMask(const Tensor& mask, float fThreshold = 0.0f) {
    auto cMask = mask.contiguous();
    int nH = cMask.shape(2);   // 20260330 ZJH 掩码高度
    int nW = cMask.shape(3);   // 20260330 ZJH 掩码宽度
    int nTotal = nH * nW;      // 20260330 ZJH 总像素数
    const float* pData = cMask.floatDataPtr();

    std::vector<uint8_t> vecBinary(static_cast<size_t>(nTotal));
    for (int i = 0; i < nTotal; ++i) {
        // 20260330 ZJH logit > threshold 为前景（255），否则为背景（0）
        vecBinary[static_cast<size_t>(i)] = (pData[i] > fThreshold) ? 255 : 0;
    }
    return vecBinary;  // 20260330 ZJH 返回二值掩码
}

// 20260330 ZJH preprocessImageForSAM — SAM 图像预处理辅助函数
// 将任意尺寸的 [1, 3, H, W] 图像 resize + padding 到 [1, 3, targetSize, targetSize]
// 使用长边缩放 + 零填充（SAM 标准预处理流程）
// srcImage: [1, 3, srcH, srcW] 原始图像（已归一化到 [0, 1]）
// nTargetSize: 目标尺寸（1024 或 512）
// 返回: [1, 3, targetSize, targetSize] 预处理后的图像
Tensor preprocessImageForSAM(const Tensor& srcImage, int nTargetSize = 1024) {
    auto cSrc = srcImage.contiguous();
    int nSrcH = cSrc.shape(2);  // 20260330 ZJH 原始高度
    int nSrcW = cSrc.shape(3);  // 20260330 ZJH 原始宽度

    // 20260330 ZJH 计算缩放比例（长边缩放到 targetSize）
    float fScale = static_cast<float>(nTargetSize) /
                   static_cast<float>(std::max(nSrcH, nSrcW));
    int nNewH = static_cast<int>(std::round(static_cast<float>(nSrcH) * fScale));
    int nNewW = static_cast<int>(std::round(static_cast<float>(nSrcW) * fScale));
    // 20260330 ZJH 确保不超过目标尺寸
    nNewH = std::min(nNewH, nTargetSize);
    nNewW = std::min(nNewW, nTargetSize);

    // 20260330 ZJH 创建目标张量（零填充）
    auto result = Tensor::zeros({1, 3, nTargetSize, nTargetSize});
    const float* pSrc = cSrc.floatDataPtr();
    float* pDst = result.mutableFloatDataPtr();

    // 20260330 ZJH 双线性插值缩放 + 左上对齐填充
    float fScaleH = static_cast<float>(nSrcH) / static_cast<float>(nNewH);
    float fScaleW = static_cast<float>(nSrcW) / static_cast<float>(nNewW);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < nNewH; ++h) {
            for (int w = 0; w < nNewW; ++w) {
                // 20260330 ZJH 映射到原始坐标
                float fSrcH = (static_cast<float>(h) + 0.5f) * fScaleH - 0.5f;
                float fSrcW = (static_cast<float>(w) + 0.5f) * fScaleW - 0.5f;
                int nH0 = static_cast<int>(std::floor(fSrcH));
                int nW0 = static_cast<int>(std::floor(fSrcW));
                int nH1 = nH0 + 1;
                int nW1 = nW0 + 1;
                float fDh = fSrcH - static_cast<float>(nH0);
                float fDw = fSrcW - static_cast<float>(nW0);
                nH0 = std::max(0, std::min(nH0, nSrcH - 1));
                nH1 = std::max(0, std::min(nH1, nSrcH - 1));
                nW0 = std::max(0, std::min(nW0, nSrcW - 1));
                nW1 = std::max(0, std::min(nW1, nSrcW - 1));

                // 20260330 ZJH 读取源图像四邻域像素
                float fV00 = pSrc[c * nSrcH * nSrcW + nH0 * nSrcW + nW0];
                float fV01 = pSrc[c * nSrcH * nSrcW + nH0 * nSrcW + nW1];
                float fV10 = pSrc[c * nSrcH * nSrcW + nH1 * nSrcW + nW0];
                float fV11 = pSrc[c * nSrcH * nSrcW + nH1 * nSrcW + nW1];
                float fVal = fV00 * (1 - fDh) * (1 - fDw) + fV01 * (1 - fDh) * fDw
                           + fV10 * fDh * (1 - fDw) + fV11 * fDh * fDw;

                // 20260330 ZJH 写入目标（左上角对齐，右/下方零填充）
                pDst[c * nTargetSize * nTargetSize + h * nTargetSize + w] = fVal;
            }
        }
    }
    return result;  // 20260330 ZJH 返回预处理后的图像
}

}  // namespace om
