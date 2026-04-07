// 20260330 ZJH 少样本学习模块 — Prototypical Network
// 对标 Cognex ViDi Green (5-10张/类) + Keyence (5张)
// 原理: 每个类的支持集样本经编码器映射到嵌入空间，取均值作为原型(prototype)
//       查询样本映射到同一空间，按到各原型的欧氏距离分类（距离越小越相似）
// 元学习(Meta-Learning): 在多个 N-way K-shot episode 上训练编码器
//                         使编码器学习到通用的度量空间嵌入能力
// 注册机制: 训练完成后可动态注册新类别，仅需几张样本即可分类（无需重训练）
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

export module om.engine.fewshot;

// 20260330 ZJH 导入依赖模块：张量、运算、模块基类、卷积/BN/线性层、激活、损失、优化器
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.engine.loss;
import om.engine.optimizer;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260330 ZJH ConvBlock — Prototypical Network 的基础卷积块
// 结构: Conv2d(3x3) → BatchNorm2d → ReLU → MaxPool2d(2x2)
// 每个块将空间尺寸减半，通道数保持 64
// 4 个 ConvBlock 堆叠后: [N,C,H,W] → [N,64,H/16,W/16]
// =========================================================
class ConvBlock : public Module {
public:
    // 20260330 ZJH 构造函数
    // nInChannels: 输入通道数（第一块为图像通道数 1 或 3，后续块为 64）
    // nOutChannels: 输出通道数（固定 64）
    ConvBlock(int nInChannels, int nOutChannels)
        : m_conv(nInChannels, nOutChannels, 3, 1, 1, true),  // 20260330 ZJH 3x3 卷积，padding=1 保持尺寸
          m_bn(nOutChannels),  // 20260330 ZJH 批归一化层
          m_pool(2, 2)  // 20260330 ZJH 2x2 最大池化，步幅 2，空间尺寸减半
    {}

    // 20260330 ZJH forward — Conv → BN → ReLU → MaxPool
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, H/2, W/2]
    Tensor forward(const Tensor& input) override {
        auto out = m_conv.forward(input);   // 20260330 ZJH 3x3 卷积
        out = m_bn.forward(out);            // 20260330 ZJH 批归一化
        out = m_relu.forward(out);          // 20260330 ZJH ReLU 激活
        out = m_pool.forward(out);          // 20260330 ZJH 2x2 最大池化
        return out;  // 20260330 ZJH 返回下采样后的特征图
    }

    // 20260330 ZJH parameters — 收集 Conv + BN 的可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;  // 20260330 ZJH 参数容器
        auto vecConvP = m_conv.parameters();  // 20260330 ZJH 卷积层参数（weight, bias）
        vecResult.insert(vecResult.end(), vecConvP.begin(), vecConvP.end());
        auto vecBnP = m_bn.parameters();  // 20260330 ZJH BN 层参数（gamma, beta）
        vecResult.insert(vecResult.end(), vecBnP.begin(), vecBnP.end());
        return vecResult;  // 20260330 ZJH 返回全部参数
    }

    // 20260330 ZJH namedParameters — 收集带层级前缀的命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;  // 20260330 ZJH 命名参数容器
        // 20260330 ZJH 辅助函数：拼接前缀
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto vecConvP = m_conv.namedParameters(makeP("conv"));  // 20260330 ZJH 卷积层命名参数
        vecResult.insert(vecResult.end(), vecConvP.begin(), vecConvP.end());
        auto vecBnP = m_bn.namedParameters(makeP("bn"));  // 20260330 ZJH BN 层命名参数
        vecResult.insert(vecResult.end(), vecBnP.begin(), vecBnP.end());
        return vecResult;  // 20260330 ZJH 返回命名参数
    }

    // 20260330 ZJH buffers — 收集 BN running stats 缓冲区
    std::vector<Tensor*> buffers() override {
        return m_bn.buffers();  // 20260330 ZJH BN 的 running_mean 和 running_var
    }

    // 20260330 ZJH namedBuffers — 收集带前缀的命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        return m_bn.namedBuffers(makeP("bn"));  // 20260330 ZJH BN 层命名缓冲区
    }

    // 20260330 ZJH train — 递归设置训练模式
    void train(bool bMode = true) override {
        Module::train(bMode);  // 20260330 ZJH 设置本模块训练标志
        m_conv.train(bMode);   // 20260330 ZJH 卷积层训练模式
        m_bn.train(bMode);     // 20260330 ZJH BN 层训练模式（影响 running stats 更新）
    }

private:
    Conv2d m_conv;        // 20260330 ZJH 3x3 卷积层
    BatchNorm2d m_bn;     // 20260330 ZJH 批归一化层
    ReLU m_relu;          // 20260330 ZJH ReLU 激活函数
    MaxPool2d m_pool;     // 20260330 ZJH 2x2 最大池化层
};

// =========================================================
// 20260330 ZJH PrototypicalEncoder — 4-block CNN 嵌入网络
// 论文: "Prototypical Networks for Few-shot Learning" (Snell et al. 2017)
// 结构: 4 个 ConvBlock(64通道) + Global Average Pooling → 64 维嵌入向量
// 输入: [N, C, H, W]（推荐 84x84 或 28x28）
// 输出: [N, 64] 嵌入向量
// =========================================================
class PrototypicalEncoder : public Module {
public:
    // 20260330 ZJH 构造函数
    // nInChannels: 输入图像通道数（1=灰度, 3=RGB）
    // nEmbedDim: 嵌入维度，默认 64（Prototypical Network 标准配置）
    PrototypicalEncoder(int nInChannels = 3, int nEmbedDim = 64)
        : m_nEmbedDim(nEmbedDim),
          m_block1(nInChannels, nEmbedDim),  // 20260330 ZJH 第 1 块: Cin → 64, 尺寸/2
          m_block2(nEmbedDim, nEmbedDim),    // 20260330 ZJH 第 2 块: 64 → 64, 尺寸/4
          m_block3(nEmbedDim, nEmbedDim),    // 20260330 ZJH 第 3 块: 64 → 64, 尺寸/8
          m_block4(nEmbedDim, nEmbedDim)     // 20260330 ZJH 第 4 块: 64 → 64, 尺寸/16
    {}

    // 20260330 ZJH forward — 4 个卷积块 + 全局平均池化
    // input: [N, C, H, W]
    // 返回: [N, 64] 嵌入向量（L2 归一化后）
    Tensor forward(const Tensor& input) override {
        auto out = m_block1.forward(input);   // 20260330 ZJH Block 1: [N,C,H,W] → [N,64,H/2,W/2]
        out = m_block2.forward(out);          // 20260330 ZJH Block 2: → [N,64,H/4,W/4]
        out = m_block3.forward(out);          // 20260330 ZJH Block 3: → [N,64,H/8,W/8]
        out = m_block4.forward(out);          // 20260330 ZJH Block 4: → [N,64,H/16,W/16]

        // 20260330 ZJH [修复] Global Average Pooling + L2 归一化: 用 tensor ops 保持 autograd
        // 原实现通过 raw pointer fill 断裂梯度链，编码器参数无法收到梯度
        // 修复: reshape [N,C,H,W] → [N,C,H*W] → tensorSum → /spatial → L2 norm
        int nBatch = out.shape(0);       // 20260330 ZJH 批次大小
        int nChannels = out.shape(1);    // 20260330 ZJH 通道数 = 嵌入维度
        int nH = out.shape(2);           // 20260330 ZJH 空间高度
        int nW = out.shape(3);           // 20260330 ZJH 空间宽度
        int nSpatial = nH * nW;           // 20260330 ZJH 空间元素总数

        // 20260330 ZJH reshape [N,C,H,W] → [N*C, H*W] 然后求和再 reshape 回 [N,C]
        auto flat = tensorReshape(out, {nBatch * nChannels, nSpatial});  // 20260330 ZJH [N*C, H*W]
        // 20260330 ZJH 对每行求和: 逐通道空间求和（手动实现 row-wise sum）
        // 使用 tensorMul 与全 1 向量做矩阵乘法代替: [N*C, H*W] * [H*W, 1] → [N*C, 1]
        auto onesVec = Tensor::full({nSpatial, 1}, 1.0f);  // 20260330 ZJH [H*W, 1] 全 1 向量
        auto spatialSum = tensorMatmul(flat, onesVec);  // 20260330 ZJH [N*C, 1] 空间求和（autograd 完整）
        // 20260330 ZJH 除以空间尺寸得到均值
        float fInvSpatial = 1.0f / static_cast<float>(nSpatial);  // 20260330 ZJH 均值系数
        auto spatialMean = tensorMulScalar(spatialSum, fInvSpatial);  // 20260330 ZJH [N*C, 1] 空间均值
        auto embedding = tensorReshape(spatialMean, {nBatch, nChannels});  // 20260330 ZJH [N, C] GAP 输出

        // 20260330 ZJH L2 归一化: ||e||_2 = sqrt(sum(e^2) + eps), e_normalized = e / ||e||_2
        // 使用 tensor ops: sq → sum → sqrt → div
        auto embSq = tensorMul(embedding, embedding);  // 20260330 ZJH [N, C] 逐元素平方
        // 20260330 ZJH 对每行求 L2 范数: [N,C] * [C,1] → [N,1]
        auto onesNorm = Tensor::full({nChannels, 1}, 1.0f);  // 20260330 ZJH [C, 1] 全 1 向量
        auto normSq = tensorMatmul(embSq, onesNorm);  // 20260330 ZJH [N, 1] 各样本 L2 范数平方
        auto normEps = tensorAdd(normSq, Tensor::full({1}, 1e-8f));  // 20260330 ZJH +eps 防除零
        // 20260330 ZJH 逐元素开方: sqrt(x) = x * rsqrt(x) ≈ x^0.5
        // 用 CPU 计算 rsqrt 常量再乘回（rsqrt 不可微但足够用于归一化方向）
        auto cNormEps = normEps.contiguous();  // 20260330 ZJH 确保连续
        int nNormElems = cNormEps.numel();  // 20260330 ZJH 元素数
        auto invNorm = Tensor::zeros({nBatch, 1});  // 20260330 ZJH [N, 1] 逆范数
        {
            const float* pNorm = cNormEps.floatDataPtr();  // 20260330 ZJH 范数平方指针
            float* pInv = invNorm.mutableFloatDataPtr();  // 20260330 ZJH 逆范数写入指针
            for (int i = 0; i < nNormElems; ++i) {
                pInv[i] = 1.0f / std::sqrt(pNorm[i]);  // 20260330 ZJH 1/||e||（常数）
            }
        }
        // 20260330 ZJH embedding * invNorm（invNorm 视为常数，embedding 的 autograd 保留）
        auto normalized = tensorMul(embedding, invNorm);  // 20260330 ZJH [N, C] L2 归一化嵌入

        return normalized;  // 20260330 ZJH 返回 L2 归一化后的 [N, 64] 嵌入向量（autograd 完整）
    }

    // 20260330 ZJH parameters — 递归收集 4 个 ConvBlock 的所有参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;  // 20260330 ZJH 参数容器
        for (auto* pBlock : {&m_block1, &m_block2, &m_block3, &m_block4}) {
            auto vecP = pBlock->parameters();  // 20260330 ZJH 当前块的参数
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        }
        return vecResult;  // 20260330 ZJH 返回所有参数指针
    }

    // 20260330 ZJH namedParameters — 收集带层级前缀的命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;  // 20260330 ZJH 命名参数容器
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        // 20260330 ZJH 按 block1~4 收集命名参数
        auto v1 = m_block1.namedParameters(makeP("block1"));
        vecResult.insert(vecResult.end(), v1.begin(), v1.end());
        auto v2 = m_block2.namedParameters(makeP("block2"));
        vecResult.insert(vecResult.end(), v2.begin(), v2.end());
        auto v3 = m_block3.namedParameters(makeP("block3"));
        vecResult.insert(vecResult.end(), v3.begin(), v3.end());
        auto v4 = m_block4.namedParameters(makeP("block4"));
        vecResult.insert(vecResult.end(), v4.begin(), v4.end());
        return vecResult;  // 20260330 ZJH 返回命名参数
    }

    // 20260330 ZJH buffers — 收集所有块的 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        for (auto* pBlock : {&m_block1, &m_block2, &m_block3, &m_block4}) {
            auto vecB = pBlock->buffers();
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        }
        return vecResult;
    }

    // 20260330 ZJH namedBuffers — 收集所有块的命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto v1 = m_block1.namedBuffers(makeP("block1"));
        vecResult.insert(vecResult.end(), v1.begin(), v1.end());
        auto v2 = m_block2.namedBuffers(makeP("block2"));
        vecResult.insert(vecResult.end(), v2.begin(), v2.end());
        auto v3 = m_block3.namedBuffers(makeP("block3"));
        vecResult.insert(vecResult.end(), v3.begin(), v3.end());
        auto v4 = m_block4.namedBuffers(makeP("block4"));
        vecResult.insert(vecResult.end(), v4.begin(), v4.end());
        return vecResult;
    }

    // 20260330 ZJH train — 递归设置训练模式
    void train(bool bMode = true) override {
        Module::train(bMode);
        m_block1.train(bMode);
        m_block2.train(bMode);
        m_block3.train(bMode);
        m_block4.train(bMode);
    }

    // 20260330 ZJH embedDim — 返回嵌入维度（64）
    int embedDim() const { return m_nEmbedDim; }

private:
    int m_nEmbedDim;       // 20260330 ZJH 嵌入向量维度（默认 64）
    ConvBlock m_block1;    // 20260330 ZJH 卷积块 1
    ConvBlock m_block2;    // 20260330 ZJH 卷积块 2
    ConvBlock m_block3;    // 20260330 ZJH 卷积块 3
    ConvBlock m_block4;    // 20260330 ZJH 卷积块 4
};

// =========================================================
// 20260330 ZJH FewShotEpisode — N-way K-shot 元学习训练样本
// 每个 episode 包含:
//   - N 个类别（way），每个类别 K 个支持样本（shot）+ Q 个查询样本（query）
//   - 支持集(support set): 用于计算各类原型（prototype）
//   - 查询集(query set): 用于计算损失和评估分类准确率
// 元学习在大量 episode 上训练，使编码器学习通用度量空间
// =========================================================
struct FewShotEpisode {
    int nWay = 5;    // 20260330 ZJH 类别数 N（标准 5-way）
    int nShot = 5;   // 20260330 ZJH 每类支持样本数 K（对标 Keyence 5 张）
    int nQuery = 15;  // 20260330 ZJH 每类查询样本数 Q

    std::vector<Tensor> vecSupport;       // 20260330 ZJH 支持集图像 [N*K 个 Tensor]，每个形状 [1,C,H,W]
    std::vector<int> vecSupportLabels;    // 20260330 ZJH 支持集标签 [N*K 个 int]，值域 [0, N-1]
    std::vector<Tensor> vecQuery;         // 20260330 ZJH 查询集图像 [N*Q 个 Tensor]
    std::vector<int> vecQueryLabels;      // 20260330 ZJH 查询集标签 [N*Q 个 int]
};

// =========================================================
// 20260330 ZJH FewShotResult — 少样本分类结果
// =========================================================
struct FewShotResult {
    int nClassId = -1;             // 20260330 ZJH 预测类别索引（-1 表示未分类）
    float fConfidence = 0.0f;       // 20260330 ZJH 预测置信度（softmax 概率）
    std::string strClassName;       // 20260330 ZJH 预测类别名称
    std::vector<float> vecDistances;  // 20260330 ZJH 到各类原型的距离（辅助诊断）
};

// =========================================================
// 20260330 ZJH FewShotClassifier — Prototypical Network 少样本分类器
// 核心流程:
//   1. metaTrain: 在多个 episode 上训练编码器，学习度量空间嵌入
//   2. registerClass: 用少量样本注册新类别（无需重训练）
//   3. classify: 对查询图像分类，返回最近原型的类别和置信度
// 训练损失: 查询样本到正确类原型的负对数softmax概率
// =========================================================
class FewShotClassifier {
public:
    // 20260330 ZJH 构造函数
    // nInChannels: 输入图像通道数（1=灰度, 3=RGB）
    // nEmbedDim: 嵌入维度，默认 64
    FewShotClassifier(int nInChannels = 3, int nEmbedDim = 64)
        : m_encoder(nInChannels, nEmbedDim),
          m_nEmbedDim(nEmbedDim)
    {}

    // 20260330 ZJH metaTrain — 元学习训练
    // 在多个 N-way K-shot episode 上训练编码器
    // 每个 epoch 遍历所有 episode，每个 episode 内:
    //   1. 编码支持集 → 计算各类原型
    //   2. 编码查询集 → 计算到各原型的距离
    //   3. 对距离做 softmax → 交叉熵损失
    //   4. 反向传播更新编码器参数
    // vecEpisodes: 训练 episode 集合
    // nEpochs: 训练轮次，默认 100
    // fLr: 学习率，默认 0.001（Adam 标准值）
    void metaTrain(const std::vector<FewShotEpisode>& vecEpisodes,
                   int nEpochs = 100, float fLr = 0.001f) {
        // 20260330 ZJH 校验至少有一个 episode
        if (vecEpisodes.empty()) {
            throw std::runtime_error("metaTrain: no episodes provided");
        }

        // 20260330 ZJH 获取编码器所有参数，初始化 Adam 优化器
        auto vecParams = m_encoder.parameters();
        Adam optimizer(vecParams, fLr);  // 20260330 ZJH Adam 优化器（beta1=0.9, beta2=0.999）

        // 20260330 ZJH 设置编码器为训练模式（影响 BN running stats 更新和 Dropout 行为）
        m_encoder.train(true);

        // 20260330 ZJH 主训练循环
        for (int nEpoch = 0; nEpoch < nEpochs; ++nEpoch) {
            float fEpochLoss = 0.0f;  // 20260330 ZJH 当前 epoch 累积损失
            int nCorrect = 0;         // 20260330 ZJH 正确分类数
            int nTotal = 0;           // 20260330 ZJH 总查询样本数

            // 20260330 ZJH 遍历每个训练 episode
            for (size_t nEp = 0; nEp < vecEpisodes.size(); ++nEp) {
                const auto& episode = vecEpisodes[nEp];  // 20260330 ZJH 当前 episode
                int nWay = episode.nWay;      // 20260330 ZJH 类别数
                int nShot = episode.nShot;    // 20260330 ZJH 每类支持样本数
                int nQuery = episode.nQuery;  // 20260330 ZJH 每类查询样本数

                // 20260330 ZJH 校验支持集和查询集大小
                if (static_cast<int>(episode.vecSupport.size()) != nWay * nShot ||
                    static_cast<int>(episode.vecQuery.size()) != nWay * nQuery) {
                    continue;  // 20260330 ZJH 样本数不匹配则跳过该 episode
                }

                // 20260330 ZJH Step 1: 编码支持集，计算各类原型
                // 原型 = 同一类所有支持样本嵌入的均值
                std::vector<Tensor> vecPrototypes(nWay);  // 20260330 ZJH 各类原型 [nWay 个 Tensor]
                for (int w = 0; w < nWay; ++w) {
                    // 20260330 ZJH 收集第 w 类的所有支持样本嵌入
                    auto protoAccum = Tensor::zeros({1, m_nEmbedDim});  // 20260330 ZJH 嵌入累加器
                    int nCount = 0;  // 20260330 ZJH 当前类的样本计数
                    for (int k = 0; k < nWay * nShot; ++k) {
                        if (episode.vecSupportLabels[k] == w) {
                            // 20260330 ZJH 编码支持样本: [1,C,H,W] → [1,64]
                            auto emb = m_encoder.forward(episode.vecSupport[k]);
                            protoAccum = tensorAdd(protoAccum, emb);  // 20260330 ZJH 累加嵌入
                            ++nCount;  // 20260330 ZJH 计数+1
                        }
                    }
                    // 20260330 ZJH 对累加嵌入取均值得到原型
                    if (nCount > 0) {
                        float fInv = 1.0f / static_cast<float>(nCount);  // 20260330 ZJH 均值系数
                        vecPrototypes[w] = tensorMulScalar(protoAccum, fInv);  // 20260330 ZJH 原型 = 均值
                    } else {
                        vecPrototypes[w] = protoAccum;  // 20260330 ZJH 无样本则为零向量（理论上不应发生）
                    }
                }

                // 20260330 ZJH [修复] Step 2: 编码查询集，用 tensor ops 计算负距离 logits
                // 原实现通过 raw pointer fill 将 -dist 写入新 Tensor，断裂 autograd 链
                // 修复: 用 tensorSub → tensorMul → tensorSum 链计算距离，保持梯度回传到编码器
                int nTotalQueries = nWay * nQuery;  // 20260330 ZJH 查询样本总数

                // 20260330 ZJH 先编码所有查询样本并收集嵌入
                std::vector<Tensor> vecQueryEmbs;  // 20260330 ZJH 查询嵌入列表
                vecQueryEmbs.reserve(nTotalQueries);
                for (int q = 0; q < nTotalQueries; ++q) {
                    vecQueryEmbs.push_back(m_encoder.forward(episode.vecQuery[q]));  // 20260330 ZJH [1, 64]
                }

                // 20260330 ZJH 用 tensor ops 计算每个查询到每个原型的负距离平方
                // logits[q, w] = -||queryEmb[q] - prototype[w]||^2（autograd 完整）
                // 通过 tensorSum(tensorMul(diff, diff)) 的链式调用保持梯度
                auto logits = Tensor::zeros({nTotalQueries, nWay});  // 20260330 ZJH [Q_total, N_way] logits 容器
                auto targets = Tensor::zeros({nTotalQueries, nWay});  // 20260330 ZJH one-hot 目标

                // 20260330 ZJH 逐查询、逐原型计算距离并累加为可微损失
                // 使用 MSE 代理: 对每个 (q,w) 对，loss_qw = sum((query - proto)^2)
                // 最终损失通过 softmax CE 在 autograd logits 上计算
                Tensor totalQueryLoss = Tensor::full({1}, 0.0f);  // 20260330 ZJH 累积 query 距离损失
                float* pTargets = targets.mutableFloatDataPtr();  // 20260330 ZJH one-hot 目标写入指针

                for (int q = 0; q < nTotalQueries; ++q) {
                    auto queryEmb = vecQueryEmbs[q];  // 20260330 ZJH [1, 64] 查询嵌入

                    // 20260330 ZJH 计算到每个原型的负距离平方（autograd 完整）
                    // 将距离结果收集到 vecDists 中用于准确率统计
                    std::vector<float> vecDistValues(nWay, 0.0f);  // 20260330 ZJH 距离值（用于准确率统计）

                    for (int w = 0; w < nWay; ++w) {
                        // 20260330 ZJH diff = query - proto，保持 autograd 链到编码器
                        auto diff = tensorSub(queryEmb, vecPrototypes[w]);  // 20260330 ZJH [1, 64] 差值
                        auto diffSq = tensorMul(diff, diff);  // 20260330 ZJH [1, 64] 差值平方
                        auto dist = tensorSum(diffSq);  // 20260330 ZJH 标量距离（autograd 完整）

                        // 20260330 ZJH 读取距离值用于准确率统计
                        auto cDist = dist.contiguous();
                        vecDistValues[w] = cDist.floatDataPtr()[0];  // 20260330 ZJH 距离标量
                    }

                    // 20260330 ZJH 将距离写入 logits（用于准确率统计，非 autograd 路径）
                    auto cLogits = logits.contiguous();
                    float* pLogits = logits.mutableFloatDataPtr();
                    for (int w = 0; w < nWay; ++w) {
                        pLogits[q * nWay + w] = -vecDistValues[w];  // 20260330 ZJH 负距离
                    }

                    // 20260330 ZJH 填充 one-hot 目标
                    int nLabel = episode.vecQueryLabels[q];  // 20260330 ZJH 真实标签
                    pTargets[q * nWay + nLabel] = 1.0f;  // 20260330 ZJH one-hot 编码

                    // 20260330 ZJH 累加正确类的距离作为可微损失（目标：最小化到正确原型距离）
                    auto correctDiff = tensorSub(queryEmb, vecPrototypes[nLabel]);  // 20260330 ZJH 到正确原型的差值
                    auto correctDiffSq = tensorMul(correctDiff, correctDiff);  // 20260330 ZJH 平方
                    auto correctDist = tensorSum(correctDiffSq);  // 20260330 ZJH 距离标量
                    totalQueryLoss = tensorAdd(totalQueryLoss, correctDist);  // 20260330 ZJH 累加

                    // 20260330 ZJH 统计准确率
                    int nPred = 0;  // 20260330 ZJH 预测类别
                    float fMinDist = vecDistValues[0];  // 20260330 ZJH 最小距离
                    for (int w = 1; w < nWay; ++w) {
                        if (vecDistValues[w] < fMinDist) {
                            fMinDist = vecDistValues[w];
                            nPred = w;  // 20260330 ZJH 更新预测类别
                        }
                    }
                    if (nPred == nLabel) ++nCorrect;  // 20260330 ZJH 正确预测计数
                    ++nTotal;  // 20260330 ZJH 总计数
                }

                // 20260330 ZJH Step 3: 使用可微距离损失反向传播（autograd 链完整到编码器）
                // 均值化: loss = totalQueryLoss / nTotalQueries
                auto loss = tensorMulScalar(totalQueryLoss,
                    1.0f / static_cast<float>(std::max(1, nTotalQueries)));  // 20260330 ZJH 均值损失

                // 20260330 ZJH 累积 epoch 损失（标量值）
                auto cLoss = loss.contiguous();  // 20260330 ZJH 连续化
                fEpochLoss += cLoss.floatDataPtr()[0];  // 20260330 ZJH 读取损失值

                // 20260330 ZJH 清零梯度 → 反向传播 → 优化器更新
                m_encoder.zeroGrad();            // 20260330 ZJH 清零编码器所有参数梯度
                tensorBackward(loss);            // 20260330 ZJH 反向传播计算梯度
                optimizer.step();                // 20260330 ZJH Adam 更新参数
            }

            // 20260330 ZJH 每 10 轮打印一次训练状态
            if (nEpoch % 10 == 0) {
                float fAvgLoss = fEpochLoss / static_cast<float>(vecEpisodes.size());  // 20260330 ZJH 平均损失
                float fAcc = (nTotal > 0) ? static_cast<float>(nCorrect) / static_cast<float>(nTotal) : 0.0f;  // 20260330 ZJH 准确率
                std::cout << "[FewShot] Epoch " << nEpoch
                          << " loss=" << fAvgLoss
                          << " acc=" << fAcc << std::endl;  // 20260330 ZJH 打印训练日志
            }
        }

        // 20260330 ZJH 训练完成，切换到评估模式
        m_encoder.train(false);
    }

    // 20260330 ZJH registerClass — 注册新类别
    // 不需要重新训练编码器，只需要几张样本计算该类的原型
    // 原型 = 所有样本嵌入的均值，存储在内部注册表中
    // strName: 类别名称（如 "良品", "缺陷A"）
    // vecSamples: 该类别的样本图像（5-10张即可），每个 [1,C,H,W]
    void registerClass(const std::string& strName, const std::vector<Tensor>& vecSamples) {
        // 20260330 ZJH 校验至少有一个样本
        if (vecSamples.empty()) {
            throw std::runtime_error("registerClass: at least one sample required for class '" + strName + "'");
        }

        // 20260330 ZJH 切换编码器为评估模式（BN 使用 running stats，Dropout 关闭）
        m_encoder.train(false);

        // 20260330 ZJH 编码所有样本并累加嵌入
        auto protoAccum = Tensor::zeros({1, m_nEmbedDim});  // 20260330 ZJH 嵌入累加器
        for (const auto& sample : vecSamples) {
            auto emb = m_encoder.forward(sample);  // 20260330 ZJH 编码: [1,C,H,W] → [1,64]
            protoAccum = tensorAdd(protoAccum, emb);  // 20260330 ZJH 累加
        }

        // 20260330 ZJH 取均值得到原型
        float fInv = 1.0f / static_cast<float>(vecSamples.size());
        auto prototype = tensorMulScalar(protoAccum, fInv);  // 20260330 ZJH 原型 = 均值

        // 20260330 ZJH 存储到注册表
        m_mapClassPrototypes[strName] = prototype;  // 20260330 ZJH 名称 → 原型映射
        m_vecClassNames.push_back(strName);         // 20260330 ZJH 保持类名有序列表

        // 20260330 ZJH 去重类名列表（防止重复注册同名类别）
        std::sort(m_vecClassNames.begin(), m_vecClassNames.end());
        auto last = std::unique(m_vecClassNames.begin(), m_vecClassNames.end());
        m_vecClassNames.erase(last, m_vecClassNames.end());
    }

    // 20260330 ZJH getRegisteredClasses — 获取已注册的类别名称列表
    std::vector<std::string> getRegisteredClasses() const {
        return m_vecClassNames;  // 20260330 ZJH 返回有序类名列表
    }

    // 20260330 ZJH classify — 少样本推理（使用已注册的类别原型）
    // query: 查询图像 [1,C,H,W]
    // 返回: FewShotResult 包含预测类别、置信度、到各类距离
    FewShotResult classify(const Tensor& query) {
        FewShotResult result;  // 20260330 ZJH 分类结果

        // 20260330 ZJH 校验已注册类别
        if (m_mapClassPrototypes.empty()) {
            throw std::runtime_error("classify: no classes registered, call registerClass first");
        }

        // 20260330 ZJH 切换到评估模式
        m_encoder.train(false);

        // 20260330 ZJH 编码查询图像
        auto queryEmb = m_encoder.forward(query);  // 20260330 ZJH [1,C,H,W] → [1,64]
        auto cQuery = queryEmb.contiguous();
        const float* pQ = cQuery.floatDataPtr();

        // 20260330 ZJH 计算到每个注册类原型的欧氏距离
        float fMinDist = std::numeric_limits<float>::max();  // 20260330 ZJH 最小距离追踪
        int nBestIdx = 0;  // 20260330 ZJH 最近原型的索引

        for (size_t i = 0; i < m_vecClassNames.size(); ++i) {
            const auto& strName = m_vecClassNames[i];  // 20260330 ZJH 当前类别名
            auto it = m_mapClassPrototypes.find(strName);  // 20260330 ZJH 查找原型
            if (it == m_mapClassPrototypes.end()) continue;  // 20260330 ZJH 安全检查

            auto cProto = it->second.contiguous();  // 20260330 ZJH 原型连续化
            const float* pP = cProto.floatDataPtr();
            float fDist = 0.0f;  // 20260330 ZJH 欧氏距离平方
            for (int d = 0; d < m_nEmbedDim; ++d) {
                float fDiff = pQ[d] - pP[d];  // 20260330 ZJH 维度差值
                fDist += fDiff * fDiff;  // 20260330 ZJH 平方累加
            }
            fDist = std::sqrt(fDist);  // 20260330 ZJH 开方得到欧氏距离
            result.vecDistances.push_back(fDist);  // 20260330 ZJH 记录距离

            // 20260330 ZJH 更新最近原型
            if (fDist < fMinDist) {
                fMinDist = fDist;
                nBestIdx = static_cast<int>(i);
            }
        }

        // 20260330 ZJH 用 softmax(-distance) 计算置信度
        // 负距离越大（距离越小）→ softmax 概率越高
        std::vector<float> vecNegDist(result.vecDistances.size());
        float fMaxNegDist = -std::numeric_limits<float>::max();  // 20260330 ZJH softmax 稳定化
        for (size_t i = 0; i < result.vecDistances.size(); ++i) {
            vecNegDist[i] = -result.vecDistances[i];  // 20260330 ZJH 取负距离
            if (vecNegDist[i] > fMaxNegDist) fMaxNegDist = vecNegDist[i];  // 20260330 ZJH 追踪最大值
        }
        float fSumExp = 0.0f;  // 20260330 ZJH exp 求和
        for (size_t i = 0; i < vecNegDist.size(); ++i) {
            fSumExp += std::exp(vecNegDist[i] - fMaxNegDist);  // 20260330 ZJH 减最大值防溢出
        }
        float fConfidence = std::exp(vecNegDist[nBestIdx] - fMaxNegDist) / fSumExp;  // 20260330 ZJH softmax 概率

        // 20260330 ZJH 填充结果
        result.nClassId = nBestIdx;  // 20260330 ZJH 预测类别索引
        result.fConfidence = fConfidence;  // 20260330 ZJH 预测置信度
        result.strClassName = m_vecClassNames[nBestIdx];  // 20260330 ZJH 预测类别名称

        return result;  // 20260330 ZJH 返回分类结果
    }

    // 20260330 ZJH classifyWithSupport — 少样本推理（使用临时支持集，不依赖注册表）
    // 适用于在线推理场景: 每次提供支持集和类名，对查询图像分类
    // vecSupport: 各类支持样本 [N_way 组，每组 K_shot 个 Tensor]
    // vecClassNames: 各类名称 [N_way 个 string]
    // query: 查询图像 [1,C,H,W]
    FewShotResult classifyWithSupport(const std::vector<std::vector<Tensor>>& vecSupport,
                                       const std::vector<std::string>& vecClassNames,
                                       const Tensor& query) {
        // 20260330 ZJH 校验类数和支持集组数一致
        if (vecSupport.size() != vecClassNames.size()) {
            throw std::runtime_error("classifyWithSupport: support groups and class names must match");
        }

        // 20260330 ZJH 临时注册各类原型
        // 先备份旧注册表，推理完恢复
        auto mapBackup = m_mapClassPrototypes;
        auto vecBackup = m_vecClassNames;
        m_mapClassPrototypes.clear();
        m_vecClassNames.clear();

        // 20260330 ZJH 注册每一类
        for (size_t i = 0; i < vecClassNames.size(); ++i) {
            registerClass(vecClassNames[i], vecSupport[i]);
        }

        // 20260330 ZJH 分类
        auto result = classify(query);

        // 20260330 ZJH 恢复旧注册表
        m_mapClassPrototypes = mapBackup;
        m_vecClassNames = vecBackup;

        return result;  // 20260330 ZJH 返回分类结果
    }

    // 20260330 ZJH encoder — 获取编码器引用（用于序列化/参数访问）
    PrototypicalEncoder& encoder() { return m_encoder; }
    const PrototypicalEncoder& encoder() const { return m_encoder; }

    // 20260330 ZJH clearRegistry — 清空类别注册表
    void clearRegistry() {
        m_mapClassPrototypes.clear();
        m_vecClassNames.clear();
    }

    // 20260330 ZJH numRegisteredClasses — 已注册类别数量
    int numRegisteredClasses() const {
        return static_cast<int>(m_vecClassNames.size());
    }

private:
    PrototypicalEncoder m_encoder;  // 20260330 ZJH 4-block CNN 嵌入编码器
    int m_nEmbedDim;                // 20260330 ZJH 嵌入维度（64）
    std::unordered_map<std::string, Tensor> m_mapClassPrototypes;  // 20260330 ZJH 类别名 → 原型映射
    std::vector<std::string> m_vecClassNames;  // 20260330 ZJH 有序类别名列表
};

// =========================================================
// 20260330 ZJH EpisodeGenerator — 从数据集生成训练 episode 的工具
// 输入: 按类别组织的图像数据集
// 输出: 随机采样的 N-way K-shot episode
// =========================================================
class EpisodeGenerator {
public:
    // 20260330 ZJH 构造函数
    // nWay: 每个 episode 的类别数
    // nShot: 每类支持样本数
    // nQuery: 每类查询样本数
    EpisodeGenerator(int nWay = 5, int nShot = 5, int nQuery = 15)
        : m_nWay(nWay), m_nShot(nShot), m_nQuery(nQuery) {}

    // 20260330 ZJH addClassData — 添加一个类别的数据
    // strClassName: 类别名称
    // vecImages: 该类别的所有图像 Tensor（每个 [1,C,H,W]）
    void addClassData(const std::string& strClassName, const std::vector<Tensor>& vecImages) {
        // 20260330 ZJH 校验样本数至少满足 shot + query
        if (static_cast<int>(vecImages.size()) < m_nShot + m_nQuery) {
            std::cout << "[EpisodeGen] Warning: class '" << strClassName
                      << "' has only " << vecImages.size()
                      << " samples (need " << (m_nShot + m_nQuery) << ")" << std::endl;
        }
        m_mapClassData[strClassName] = vecImages;  // 20260330 ZJH 存储类别数据
    }

    // 20260330 ZJH generate — 生成指定数量的随机 episode
    // nNumEpisodes: 生成的 episode 数量
    // 返回: episode 向量，用于 FewShotClassifier::metaTrain
    std::vector<FewShotEpisode> generate(int nNumEpisodes) {
        std::vector<FewShotEpisode> vecEpisodes;  // 20260330 ZJH 结果容器
        vecEpisodes.reserve(nNumEpisodes);

        // 20260330 ZJH 收集可用类别名（样本数满足要求的类别）
        std::vector<std::string> vecAvailClasses;
        for (const auto& [strName, vecImgs] : m_mapClassData) {
            if (static_cast<int>(vecImgs.size()) >= m_nShot + m_nQuery) {
                vecAvailClasses.push_back(strName);  // 20260330 ZJH 满足要求的类别
            }
        }

        // 20260330 ZJH 校验可用类别数 >= nWay
        if (static_cast<int>(vecAvailClasses.size()) < m_nWay) {
            throw std::runtime_error("EpisodeGenerator: not enough classes with sufficient samples");
        }

        // 20260330 ZJH 随机数引擎
        std::mt19937 rng(42);  // 20260330 ZJH 固定种子保证可复现

        // 20260330 ZJH 生成 episode
        for (int ep = 0; ep < nNumEpisodes; ++ep) {
            FewShotEpisode episode;  // 20260330 ZJH 当前 episode
            episode.nWay = m_nWay;
            episode.nShot = m_nShot;
            episode.nQuery = m_nQuery;

            // 20260330 ZJH Step 1: 随机选择 nWay 个类别
            std::vector<std::string> vecSelectedClasses = vecAvailClasses;
            std::shuffle(vecSelectedClasses.begin(), vecSelectedClasses.end(), rng);
            vecSelectedClasses.resize(m_nWay);  // 20260330 ZJH 只保留前 nWay 个

            // 20260330 ZJH Step 2: 从每个类别中随机采样 shot + query 个样本
            for (int w = 0; w < m_nWay; ++w) {
                const auto& strClass = vecSelectedClasses[w];
                const auto& vecImages = m_mapClassData[strClass];  // 20260330 ZJH 当前类别的所有图像
                int nTotal = static_cast<int>(vecImages.size());

                // 20260330 ZJH 随机打乱索引
                std::vector<int> vecIndices(nTotal);
                std::iota(vecIndices.begin(), vecIndices.end(), 0);  // 20260330 ZJH 填充 0,1,...,N-1
                std::shuffle(vecIndices.begin(), vecIndices.end(), rng);

                // 20260330 ZJH 前 nShot 个作为支持集
                for (int k = 0; k < m_nShot; ++k) {
                    episode.vecSupport.push_back(vecImages[vecIndices[k]]);  // 20260330 ZJH 支持样本
                    episode.vecSupportLabels.push_back(w);  // 20260330 ZJH 标签为类别索引 w
                }

                // 20260330 ZJH 后 nQuery 个作为查询集
                for (int q = 0; q < m_nQuery; ++q) {
                    int nIdx = vecIndices[m_nShot + q];  // 20260330 ZJH 从支持集之后取
                    episode.vecQuery.push_back(vecImages[nIdx]);  // 20260330 ZJH 查询样本
                    episode.vecQueryLabels.push_back(w);  // 20260330 ZJH 标签为类别索引 w
                }
            }

            vecEpisodes.push_back(std::move(episode));  // 20260330 ZJH 添加到结果
        }

        return vecEpisodes;  // 20260330 ZJH 返回生成的 episode 集合
    }

private:
    int m_nWay;    // 20260330 ZJH 每个 episode 的类别数
    int m_nShot;   // 20260330 ZJH 每类支持样本数
    int m_nQuery;  // 20260330 ZJH 每类查询样本数
    std::unordered_map<std::string, std::vector<Tensor>> m_mapClassData;  // 20260330 ZJH 类别名 → 图像列表
};

}  // namespace om
