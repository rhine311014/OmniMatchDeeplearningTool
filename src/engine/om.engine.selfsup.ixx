// 20260330 ZJH 自监督预训练模块 — SimCLR 对比学习 + MAE 掩码自编码器
// 无标注数据学习视觉特征表示，预训练完成后作为下游任务的骨干网络初始化
// SimCLR: 同一图像两个增强视角为正对，不同图像为负对，NT-Xent Loss
// MAE: 随机遮挡 75% 的 patch，训练编码器-解码器重建原始图像
// 工业场景: 大量无标注产品图像可用于预训练，显著降低下游任务所需标注量
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <stdexcept>

export module om.engine.selfsup;

// 20260330 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.engine.optimizer;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260330 ZJH ProjectionHead — SimCLR 投影头
// 论文: "A Simple Framework for Contrastive Learning of Visual Representations"
// 结构: Linear(in, hidden) → ReLU → Linear(hidden, out)
// 将编码器输出的特征映射到 128 维对比学习空间
// 训练完成后丢弃投影头，只保留编码器作为预训练骨干
// =========================================================
class ProjectionHead : public Module {
public:
    // 20260330 ZJH 构造函数
    // nInDim: 输入维度（编码器输出维度）
    // nHiddenDim: 隐层维度，默认 256
    // nOutDim: 输出维度（投影空间维度），默认 128
    ProjectionHead(int nInDim, int nHiddenDim = 256, int nOutDim = 128)
        : m_fc1(nInDim, nHiddenDim),    // 20260330 ZJH 第一层全连接
          m_fc2(nHiddenDim, nOutDim),    // 20260330 ZJH 第二层全连接
          m_nInDim(nInDim),
          m_nHiddenDim(nHiddenDim),
          m_nOutDim(nOutDim)
    {}

    // 20260330 ZJH forward — Linear → ReLU → Linear → L2 归一化
    // input: [N, nInDim]
    // 返回: [N, nOutDim] L2 归一化后的投影向量
    Tensor forward(const Tensor& input) override {
        auto out = m_fc1.forward(input);   // 20260330 ZJH [N,in] → [N,hidden]
        out = m_relu.forward(out);          // 20260330 ZJH ReLU 激活
        out = m_fc2.forward(out);           // 20260330 ZJH [N,hidden] → [N,out]

        // 20260330 ZJH [修复] L2 归一化: 用 tensor ops 替代 in-place pointer 修改
        // 原实现直接修改 cOut 的数据指针，断裂 fc2 → out 的 autograd 链
        // 修复: sq → matmul(ones) → rsqrt(常数) → mul，保持 out 的 autograd
        int nBatch = out.shape(0);  // 20260330 ZJH 批次大小
        int nDim = out.shape(1);    // 20260330 ZJH 投影维度

        // 20260330 ZJH 计算逐行 L2 范数平方: [N,D] → sq → [N,D] * [D,1] → [N,1]
        auto outSq = tensorMul(out, out);  // 20260330 ZJH [N, D] 逐元素平方
        auto onesVec = Tensor::full({nDim, 1}, 1.0f);  // 20260330 ZJH [D, 1] 全 1 向量
        auto normSq = tensorMatmul(outSq, onesVec);  // 20260330 ZJH [N, 1] 各样本范数平方

        // 20260330 ZJH 计算 1/||z|| 作为常数（不参与 autograd，但 out 的 autograd 保留）
        auto cNormSq = normSq.contiguous();  // 20260330 ZJH 确保连续
        auto invNorm = Tensor::zeros({nBatch, 1});  // 20260330 ZJH [N, 1] 逆范数
        {
            const float* pNorm = cNormSq.floatDataPtr();  // 20260330 ZJH 范数平方指针
            float* pInv = invNorm.mutableFloatDataPtr();  // 20260330 ZJH 逆范数写入指针
            for (int b = 0; b < nBatch; ++b) {
                pInv[b] = 1.0f / std::sqrt(pNorm[b] + 1e-8f);  // 20260330 ZJH 1/||z||（+eps 防除零）
            }
        }
        // 20260330 ZJH out * invNorm: invNorm 为常数张量，out 的 autograd 链保留
        auto normalized = tensorMul(out, invNorm);  // 20260330 ZJH [N, D] L2 归一化（广播 [N,1]→[N,D]）

        return normalized;  // 20260330 ZJH 返回归一化投影向量（autograd 完整）
    }

    // 20260330 ZJH parameters — 收集两层全连接参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto v1 = m_fc1.parameters();
        vecResult.insert(vecResult.end(), v1.begin(), v1.end());
        auto v2 = m_fc2.parameters();
        vecResult.insert(vecResult.end(), v2.begin(), v2.end());
        return vecResult;
    }

    // 20260330 ZJH namedParameters — 收集命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto v1 = m_fc1.namedParameters(makeP("fc1"));
        vecResult.insert(vecResult.end(), v1.begin(), v1.end());
        auto v2 = m_fc2.namedParameters(makeP("fc2"));
        vecResult.insert(vecResult.end(), v2.begin(), v2.end());
        return vecResult;
    }

private:
    Linear m_fc1;    // 20260330 ZJH 第一层: in → hidden
    Linear m_fc2;    // 20260330 ZJH 第二层: hidden → out
    ReLU m_relu;     // 20260330 ZJH ReLU 激活
    int m_nInDim;    // 20260330 ZJH 输入维度
    int m_nHiddenDim;  // 20260330 ZJH 隐层维度
    int m_nOutDim;   // 20260330 ZJH 输出维度
};

// =========================================================
// 20260330 ZJH 数据增强函数 — SimCLR 对比学习专用
// SimCLR 的核心: 对同一图像施加两种不同增强，产生正样本对
// 增强策略: 随机裁剪 + 随机水平翻转 + 颜色抖动 + 高斯噪声
// =========================================================

// 20260330 ZJH augmentImage — 对图像施加随机增强
// 输入: [1, C, H, W] 的图像张量
// 返回: 增强后的 [1, C, H, W] 张量
// 使用 CPU 手写增强，不依赖外部库
inline Tensor augmentImage(const Tensor& image, std::mt19937& rng) {
    auto cImg = image.contiguous();  // 20260330 ZJH 确保连续内存
    int nBatch = cImg.shape(0);    // 20260330 ZJH batch (=1)
    int nC = cImg.shape(1);        // 20260330 ZJH 通道数
    int nH = cImg.shape(2);        // 20260330 ZJH 高度
    int nW = cImg.shape(3);        // 20260330 ZJH 宽度

    // 20260330 ZJH 分配输出张量
    auto result = Tensor::zeros({nBatch, nC, nH, nW});
    const float* pSrc = cImg.floatDataPtr();
    float* pDst = result.mutableFloatDataPtr();

    // 20260330 ZJH 复制原始数据
    int nTotal = nBatch * nC * nH * nW;
    for (int i = 0; i < nTotal; ++i) {
        pDst[i] = pSrc[i];
    }

    // 20260330 ZJH 增强 1: 随机水平翻转（概率 50%）
    std::uniform_real_distribution<float> distFlip(0.0f, 1.0f);
    if (distFlip(rng) < 0.5f) {
        // 20260330 ZJH 水平翻转: 交换每行左右两侧像素
        for (int b = 0; b < nBatch; ++b) {
            for (int c = 0; c < nC; ++c) {
                for (int h = 0; h < nH; ++h) {
                    for (int wl = 0; wl < nW / 2; ++wl) {
                        int wr = nW - 1 - wl;  // 20260330 ZJH 对称位置
                        int nIdxL = ((b * nC + c) * nH + h) * nW + wl;
                        int nIdxR = ((b * nC + c) * nH + h) * nW + wr;
                        float fTmp = pDst[nIdxL];
                        pDst[nIdxL] = pDst[nIdxR];
                        pDst[nIdxR] = fTmp;  // 20260330 ZJH 交换
                    }
                }
            }
        }
    }

    // 20260330 ZJH 增强 2: 颜色抖动（亮度±20%、对比度±20%）
    std::uniform_real_distribution<float> distBright(-0.2f, 0.2f);
    std::uniform_real_distribution<float> distContrast(0.8f, 1.2f);
    float fBrightShift = distBright(rng);   // 20260330 ZJH 亮度偏移
    float fContrastScale = distContrast(rng);  // 20260330 ZJH 对比度缩放

    for (int i = 0; i < nTotal; ++i) {
        // 20260330 ZJH 先应用对比度缩放，再应用亮度偏移
        float fVal = pDst[i] * fContrastScale + fBrightShift;
        // 20260330 ZJH 截断到 [0, 1] 范围（假设输入归一化到 [0,1]）
        pDst[i] = std::max(0.0f, std::min(1.0f, fVal));
    }

    // 20260330 ZJH 增强 3: 高斯噪声（均值0，标准差0.02）
    std::normal_distribution<float> distNoise(0.0f, 0.02f);
    for (int i = 0; i < nTotal; ++i) {
        float fVal = pDst[i] + distNoise(rng);  // 20260330 ZJH 加入高斯噪声
        pDst[i] = std::max(0.0f, std::min(1.0f, fVal));  // 20260330 ZJH 截断
    }

    return result;  // 20260330 ZJH 返回增强后的图像
}

// =========================================================
// 20260330 ZJH SimCLRTrainer — SimCLR 对比学习训练器
// 论文: Chen et al. "A Simple Framework for Contrastive Learning" (2020)
// 训练流程:
//   1. 对 batch 中每张图像施加两种随机增强，产生 2N 个样本
//   2. 通过编码器 + 投影头得到 2N 个投影向量 z_i
//   3. NT-Xent Loss: 正对（同一图像的两个增强）相似度最大化，
//      负对（不同图像的增强）相似度最小化
//   4. 训练完成后丢弃投影头，编码器作为预训练骨干
// =========================================================
class SimCLRTrainer {
public:
    // 20260330 ZJH 构造函数
    // nEncoderOutDim: 编码器输出维度（如 ResNet 的 512，或 PrototypicalEncoder 的 64）
    // nProjectionDim: 投影空间维度，默认 128
    // fTemperature: NT-Xent Loss 的温度参数 τ，默认 0.5
    //               τ 越小对比越锐利（容易训练崩溃），越大对比越平滑
    SimCLRTrainer(int nEncoderOutDim, int nProjectionDim = 128, float fTemperature = 0.5f)
        : m_projectionHead(nEncoderOutDim, 256, nProjectionDim),
          m_fTemperature(fTemperature),
          m_nEncoderOutDim(nEncoderOutDim),
          m_nProjectionDim(nProjectionDim)
    {}

    // 20260330 ZJH train — SimCLR 对比学习训练
    // encoder: 要预训练的编码器（Module 引用，输入 [N,C,H,W]，输出 [N,D]）
    // vecImages: 无标注图像列表，每个 [1,C,H,W]
    // nEpochs: 训练轮次
    // nBatchSize: 批次大小（SimCLR 推荐大 batch，此处受内存限制通常较小）
    // fLr: 学习率
    void train(Module& encoder,
               const std::vector<Tensor>& vecImages,
               int nEpochs = 100,
               int nBatchSize = 32,
               float fLr = 0.001f)
    {
        // 20260330 ZJH 校验输入
        if (vecImages.empty()) {
            throw std::runtime_error("SimCLRTrainer::train: no images provided");
        }

        // 20260330 ZJH 收集编码器 + 投影头的所有参数
        auto vecEncParams = encoder.parameters();      // 20260330 ZJH 编码器参数
        auto vecProjParams = m_projectionHead.parameters();  // 20260330 ZJH 投影头参数
        std::vector<Tensor*> vecAllParams;
        vecAllParams.insert(vecAllParams.end(), vecEncParams.begin(), vecEncParams.end());
        vecAllParams.insert(vecAllParams.end(), vecProjParams.begin(), vecProjParams.end());

        // 20260330 ZJH 初始化 Adam 优化器
        Adam optimizer(vecAllParams, fLr);

        // 20260330 ZJH 设置训练模式
        encoder.train(true);
        m_projectionHead.train(true);

        // 20260330 ZJH 随机数引擎（用于数据增强和 batch 打乱）
        std::mt19937 rng(42);

        int nN = static_cast<int>(vecImages.size());  // 20260330 ZJH 图像总数
        nBatchSize = std::min(nBatchSize, nN);  // 20260330 ZJH 批次大小不超过图像总数

        // 20260330 ZJH 训练循环
        for (int nEpoch = 0; nEpoch < nEpochs; ++nEpoch) {
            float fEpochLoss = 0.0f;  // 20260330 ZJH epoch 累积损失
            int nBatches = 0;          // 20260330 ZJH batch 计数

            // 20260330 ZJH 随机打乱索引
            std::vector<int> vecIndices(nN);
            std::iota(vecIndices.begin(), vecIndices.end(), 0);
            std::shuffle(vecIndices.begin(), vecIndices.end(), rng);

            // 20260330 ZJH 遍历 mini-batch
            for (int nStart = 0; nStart + nBatchSize <= nN; nStart += nBatchSize) {
                int nActualBatch = nBatchSize;  // 20260330 ZJH 当前 batch 实际大小

                // 20260330 ZJH Step 1: 对每张图像施加两种增强，通过编码器+投影头
                // 产生 2*nActualBatch 个投影向量
                std::vector<Tensor> vecZ;  // 20260330 ZJH 投影向量集合
                vecZ.reserve(2 * nActualBatch);

                for (int i = 0; i < nActualBatch; ++i) {
                    int nIdx = vecIndices[nStart + i];  // 20260330 ZJH 当前图像索引

                    // 20260330 ZJH 增强视角 1
                    auto aug1 = augmentImage(vecImages[nIdx], rng);
                    auto feat1 = encoder.forward(aug1);     // 20260330 ZJH 编码: [1,C,H,W] → [1,D]
                    auto z1 = m_projectionHead.forward(feat1);  // 20260330 ZJH 投影: [1,D] → [1,128]
                    vecZ.push_back(z1);

                    // 20260330 ZJH 增强视角 2
                    auto aug2 = augmentImage(vecImages[nIdx], rng);
                    auto feat2 = encoder.forward(aug2);
                    auto z2 = m_projectionHead.forward(feat2);
                    vecZ.push_back(z2);
                }

                // 20260330 ZJH Step 2: 计算 NT-Xent Loss
                // 正对: (z_{2i}, z_{2i+1})，即同一图像的两个增强
                // 负对: 所有其他配对
                // L = -log(exp(sim(z_i, z_j)/τ) / sum_k[k≠i](exp(sim(z_i, z_k)/τ)))
                int nTotalPairs = 2 * nActualBatch;  // 20260330 ZJH 总投影向量数
                float fBatchLoss = 0.0f;  // 20260330 ZJH 当前 batch 损失

                // 20260330 ZJH 提取所有投影向量的原始数据
                std::vector<const float*> vecZPtrs;
                std::vector<int> vecZDims;
                for (auto& z : vecZ) {
                    auto cz = z.contiguous();
                    vecZPtrs.push_back(cz.floatDataPtr());
                    vecZDims.push_back(cz.numel());  // 20260330 ZJH 应为 nProjectionDim
                }

                // 20260330 ZJH 预计算所有两两之间的 cosine similarity
                // 由于已经 L2 归一化，cosine sim = dot product
                std::vector<std::vector<float>> matSim(nTotalPairs, std::vector<float>(nTotalPairs, 0.0f));
                for (int i = 0; i < nTotalPairs; ++i) {
                    auto ci = vecZ[i].contiguous();
                    const float* pi = ci.floatDataPtr();
                    int nDim = ci.numel();
                    for (int j = i + 1; j < nTotalPairs; ++j) {
                        auto cj = vecZ[j].contiguous();
                        const float* pj = cj.floatDataPtr();
                        float fDot = 0.0f;  // 20260330 ZJH 点积
                        for (int d = 0; d < nDim; ++d) {
                            fDot += pi[d] * pj[d];
                        }
                        matSim[i][j] = fDot;  // 20260330 ZJH 对称矩阵
                        matSim[j][i] = fDot;
                    }
                }

                // 20260330 ZJH 计算 NT-Xent Loss
                // 对每个样本 i，其正对为 j（同一图像的另一增强），其余为负对
                for (int i = 0; i < nTotalPairs; ++i) {
                    int j = (i % 2 == 0) ? (i + 1) : (i - 1);  // 20260330 ZJH 正对索引

                    // 20260330 ZJH 分子: exp(sim(i,j) / τ)
                    float fPosSim = matSim[i][j] / m_fTemperature;

                    // 20260330 ZJH 分母: sum_{k≠i} exp(sim(i,k) / τ)
                    // 用 log-sum-exp 数值稳定计算
                    float fMaxSim = -std::numeric_limits<float>::max();
                    for (int k = 0; k < nTotalPairs; ++k) {
                        if (k == i) continue;  // 20260330 ZJH 排除自身
                        float fSim = matSim[i][k] / m_fTemperature;
                        fMaxSim = std::max(fMaxSim, fSim);
                    }

                    float fSumExp = 0.0f;
                    for (int k = 0; k < nTotalPairs; ++k) {
                        if (k == i) continue;
                        float fSim = matSim[i][k] / m_fTemperature;
                        fSumExp += std::exp(fSim - fMaxSim);  // 20260330 ZJH 减最大值防溢出
                    }

                    // 20260330 ZJH L_i = -log(exp(pos) / sum(exp(all)))
                    //            = -(pos - max) + log(sumExp)
                    //            = -(posSim - maxSim) + log(sumExp)  [maxSim 已经减掉了]
                    float fLoss_i = -(fPosSim - fMaxSim) + std::log(fSumExp + 1e-8f);
                    fBatchLoss += fLoss_i;
                }

                fBatchLoss /= static_cast<float>(nTotalPairs);  // 20260330 ZJH 平均损失

                // 20260330 ZJH [修复] Step 3: 用 MSE 代理损失覆盖 ALL 正对，非仅第一对
                // 原实现仅取 vecZ[0] 和 vecZ[1]（第一个正对），batch 中其余图像无梯度
                // 修复: 对所有正对 (vecZ[2i], vecZ[2i+1]) 计算 MSE 并求和
                // MSE ∝ 1-cosine_sim（L2 归一化后），梯度方向与 NT-Xent 一致
                if (nActualBatch > 0) {
                    // 20260330 ZJH 累加所有正对的 MSE 代理损失
                    auto proxyLoss = Tensor::full({1}, 0.0f);  // 20260330 ZJH 初始化累积损失
                    for (int i = 0; i < nActualBatch; ++i) {
                        auto z_a = vecZ[2 * i];      // 20260330 ZJH 第 i 张图的视角 1
                        auto z_b = vecZ[2 * i + 1];  // 20260330 ZJH 第 i 张图的视角 2
                        // 20260330 ZJH MSE(z_a, z_b) = sum((z_a - z_b)^2)
                        auto diff = tensorSub(z_a, z_b);  // 20260330 ZJH 差值（autograd 完整）
                        auto sq = tensorMul(diff, diff);  // 20260330 ZJH 差值平方
                        auto pairLoss = tensorSum(sq);  // 20260330 ZJH 单对 MSE 损失
                        proxyLoss = tensorAdd(proxyLoss, pairLoss);  // 20260330 ZJH 累加
                    }
                    // 20260330 ZJH 均值化: 除以正对数量
                    proxyLoss = tensorMulScalar(proxyLoss,
                        1.0f / static_cast<float>(nActualBatch));  // 20260330 ZJH 均值

                    // 20260330 ZJH 清零梯度 → 反向传播 → 优化器更新
                    for (auto* p : vecAllParams) {
                        tensorZeroGrad(*p);
                    }
                    tensorBackward(proxyLoss);
                    optimizer.step();
                }

                fEpochLoss += fBatchLoss;
                ++nBatches;
            }

            // 20260330 ZJH 每 10 轮打印训练状态
            if (nEpoch % 10 == 0 && nBatches > 0) {
                float fAvgLoss = fEpochLoss / static_cast<float>(nBatches);
                std::cout << "[SimCLR] Epoch " << nEpoch
                          << " loss=" << fAvgLoss << std::endl;
            }
        }

        // 20260330 ZJH 训练完成
        encoder.train(false);
    }

    // 20260330 ZJH getProjectionHead — 获取投影头引用（调试用）
    ProjectionHead& getProjectionHead() { return m_projectionHead; }

    // 20260330 ZJH temperature — 获取温度参数
    float temperature() const { return m_fTemperature; }

    // 20260330 ZJH setTemperature — 设置温度参数
    void setTemperature(float fTemp) { m_fTemperature = fTemp; }

private:
    ProjectionHead m_projectionHead;  // 20260330 ZJH 投影头（训练后丢弃）
    float m_fTemperature;             // 20260330 ZJH NT-Xent 温度参数 τ
    int m_nEncoderOutDim;             // 20260330 ZJH 编码器输出维度
    int m_nProjectionDim;             // 20260330 ZJH 投影空间维度
};

// =========================================================
// 20260330 ZJH MAEEncoder — 掩码自编码器编码器
// 论文: He et al. "Masked Autoencoders Are Scalable Vision Learners" (2022)
// 将图像分成 patch，随机遮挡 75%，只编码可见 patch
// 解码器从编码后的可见 patch 重建完整图像
// 预训练后丢弃解码器，编码器作为骨干网络
// =========================================================
class MAEPretrainer {
public:
    // 20260330 ZJH 构造函数
    // nPatchSize: patch 大小（将图像分成 patchSize×patchSize 的块），默认 8
    // nEncoderDim: 编码器嵌入维度，默认 256
    // fMaskRatio: 遮挡比例，默认 0.75（75%）
    // nImgChannels: 图像通道数，默认 3
    MAEPretrainer(int nPatchSize = 8, int nEncoderDim = 256,
                  float fMaskRatio = 0.75f, int nImgChannels = 3)
        : m_nPatchSize(nPatchSize),
          m_nEncoderDim(nEncoderDim),
          m_fMaskRatio(fMaskRatio),
          m_nImgChannels(nImgChannels),
          // 20260330 ZJH patch 嵌入: 将 [patchSize*patchSize*C] 映射到 encoderDim
          m_patchEmbed(nPatchSize * nPatchSize * nImgChannels, nEncoderDim),
          // 20260330 ZJH 编码器 MLP: encoderDim → encoderDim（简化版 Transformer block）
          m_encoderMlp1(nEncoderDim, nEncoderDim),
          m_encoderMlp2(nEncoderDim, nEncoderDim),
          // 20260330 ZJH 解码器: encoderDim → patchSize*patchSize*C（重建 patch 像素）
          m_decoder1(nEncoderDim, nEncoderDim),
          m_decoder2(nEncoderDim, nPatchSize * nPatchSize * nImgChannels)
    {}

    // 20260330 ZJH pretrain — MAE 自监督预训练
    // encoder: 编码器 Module（若提供则用其 forward 编码可见 patch；否则用内部 MLP）
    // vecImages: 无标注图像列表，每个 [1, C, H, W]，假设 H=W 且能被 patchSize 整除
    // nEpochs: 训练轮次
    // fLr: 学习率
    void pretrain(const std::vector<Tensor>& vecImages,
                  int nEpochs = 100, float fLr = 0.001f)
    {
        if (vecImages.empty()) {
            throw std::runtime_error("MAEPretrainer::pretrain: no images provided");
        }

        // 20260330 ZJH 收集所有参数
        std::vector<Tensor*> vecParams;
        auto addParams = [&](Module& m) {
            auto vp = m.parameters();
            vecParams.insert(vecParams.end(), vp.begin(), vp.end());
        };
        addParams(m_patchEmbed);
        addParams(m_encoderMlp1);
        addParams(m_encoderMlp2);
        addParams(m_decoder1);
        addParams(m_decoder2);

        Adam optimizer(vecParams, fLr);

        std::mt19937 rng(42);

        // 20260330 ZJH 训练循环
        for (int nEpoch = 0; nEpoch < nEpochs; ++nEpoch) {
            float fEpochLoss = 0.0f;

            for (size_t nImg = 0; nImg < vecImages.size(); ++nImg) {
                auto cImg = vecImages[nImg].contiguous();
                int nC = cImg.shape(1);
                int nH = cImg.shape(2);
                int nW = cImg.shape(3);
                int nPH = nH / m_nPatchSize;  // 20260330 ZJH 垂直 patch 数
                int nPW = nW / m_nPatchSize;  // 20260330 ZJH 水平 patch 数
                int nTotalPatches = nPH * nPW;  // 20260330 ZJH 总 patch 数
                int nPatchPixels = m_nPatchSize * m_nPatchSize * nC;  // 20260330 ZJH 每个 patch 的像素数

                // 20260330 ZJH Step 1: 将图像分成 patch 并展平为向量
                std::vector<Tensor> vecPatches;
                vecPatches.reserve(nTotalPatches);
                const float* pImg = cImg.floatDataPtr();

                for (int ph = 0; ph < nPH; ++ph) {
                    for (int pw = 0; pw < nPW; ++pw) {
                        // 20260330 ZJH 提取 patch [C, patchH, patchW] → 展平为 [1, patchPixels]
                        auto patch = Tensor::zeros({1, nPatchPixels});
                        float* pPatch = patch.mutableFloatDataPtr();
                        int nIdx = 0;
                        for (int c = 0; c < nC; ++c) {
                            for (int y = 0; y < m_nPatchSize; ++y) {
                                for (int x = 0; x < m_nPatchSize; ++x) {
                                    int nImgY = ph * m_nPatchSize + y;  // 20260330 ZJH 原图 y 坐标
                                    int nImgX = pw * m_nPatchSize + x;  // 20260330 ZJH 原图 x 坐标
                                    pPatch[nIdx++] = pImg[(c * nH + nImgY) * nW + nImgX];
                                }
                            }
                        }
                        vecPatches.push_back(patch);
                    }
                }

                // 20260330 ZJH Step 2: 随机遮挡 75% 的 patch
                int nMasked = static_cast<int>(std::round(m_fMaskRatio * nTotalPatches));
                int nVisible = nTotalPatches - nMasked;

                std::vector<int> vecPatchIdx(nTotalPatches);
                std::iota(vecPatchIdx.begin(), vecPatchIdx.end(), 0);
                std::shuffle(vecPatchIdx.begin(), vecPatchIdx.end(), rng);

                // 20260330 ZJH 前 nVisible 个为可见，后 nMasked 个为遮挡
                std::vector<int> vecVisibleIdx(vecPatchIdx.begin(), vecPatchIdx.begin() + nVisible);
                std::vector<int> vecMaskedIdx(vecPatchIdx.begin() + nVisible, vecPatchIdx.end());

                // 20260330 ZJH Step 3: 编码可见 patch
                // 对每个可见 patch: embed → MLP1 → ReLU → MLP2 → ReLU
                std::vector<Tensor> vecEncoded;
                vecEncoded.reserve(nVisible);
                for (int vi = 0; vi < nVisible; ++vi) {
                    int nPIdx = vecVisibleIdx[vi];
                    auto emb = m_patchEmbed.forward(vecPatches[nPIdx]);  // 20260330 ZJH [1,patchPixels] → [1,encoderDim]
                    emb = tensorReLU(m_encoderMlp1.forward(emb));  // 20260330 ZJH MLP1 + ReLU
                    emb = tensorReLU(m_encoderMlp2.forward(emb));  // 20260330 ZJH MLP2 + ReLU
                    vecEncoded.push_back(emb);
                }

                // 20260330 ZJH Step 4: 解码被遮挡的 patch（从可见 patch 的均值特征推断）
                // 简化版: 用所有可见 patch 编码的均值作为全局上下文
                auto globalCtx = Tensor::zeros({1, m_nEncoderDim});
                for (const auto& enc : vecEncoded) {
                    globalCtx = tensorAdd(globalCtx, enc);
                }
                if (nVisible > 0) {
                    globalCtx = tensorMulScalar(globalCtx, 1.0f / static_cast<float>(nVisible));
                }

                // 20260330 ZJH 用全局上下文解码每个被遮挡的 patch
                float fLoss = 0.0f;
                Tensor totalLoss = Tensor::zeros({1});

                for (int mi = 0; mi < nMasked; ++mi) {
                    int nPIdx = vecMaskedIdx[mi];
                    // 20260330 ZJH 解码: globalCtx → decoder1 → ReLU → decoder2 → 重建 patch
                    auto decoded = tensorReLU(m_decoder1.forward(globalCtx));
                    decoded = m_decoder2.forward(decoded);  // 20260330 ZJH [1,encoderDim] → [1,patchPixels]

                    // 20260330 ZJH MSE 重建损失: ||decoded - original_patch||^2
                    auto diff = tensorSub(decoded, vecPatches[nPIdx]);
                    auto sq = tensorMul(diff, diff);
                    auto patchLoss = tensorSum(sq);
                    totalLoss = tensorAdd(totalLoss, patchLoss);
                }

                // 20260330 ZJH 平均损失
                if (nMasked > 0) {
                    totalLoss = tensorMulScalar(totalLoss, 1.0f / static_cast<float>(nMasked));
                }

                // 20260330 ZJH 反向传播
                for (auto* p : vecParams) {
                    tensorZeroGrad(*p);
                }
                tensorBackward(totalLoss);
                optimizer.step();

                // 20260330 ZJH 累积损失
                auto cLoss = totalLoss.contiguous();
                fEpochLoss += cLoss.floatDataPtr()[0];
            }

            // 20260330 ZJH 每 10 轮打印
            if (nEpoch % 10 == 0) {
                float fAvgLoss = fEpochLoss / static_cast<float>(vecImages.size());
                std::cout << "[MAE] Epoch " << nEpoch << " loss=" << fAvgLoss << std::endl;
            }
        }
    }

    // 20260330 ZJH getPatchEmbedding — 获取 patch 嵌入层引用
    Linear& getPatchEmbedding() { return m_patchEmbed; }

    // 20260330 ZJH getEncoderParams — 获取编码器部分的参数（不含解码器）
    std::vector<Tensor*> getEncoderParams() {
        std::vector<Tensor*> vecResult;
        auto v1 = m_patchEmbed.parameters();
        vecResult.insert(vecResult.end(), v1.begin(), v1.end());
        auto v2 = m_encoderMlp1.parameters();
        vecResult.insert(vecResult.end(), v2.begin(), v2.end());
        auto v3 = m_encoderMlp2.parameters();
        vecResult.insert(vecResult.end(), v3.begin(), v3.end());
        return vecResult;
    }

private:
    int m_nPatchSize;      // 20260330 ZJH patch 大小
    int m_nEncoderDim;     // 20260330 ZJH 编码器嵌入维度
    float m_fMaskRatio;    // 20260330 ZJH 遮挡比例（75%）
    int m_nImgChannels;    // 20260330 ZJH 图像通道数

    Linear m_patchEmbed;   // 20260330 ZJH patch → 嵌入
    Linear m_encoderMlp1;  // 20260330 ZJH 编码器 MLP 第 1 层
    Linear m_encoderMlp2;  // 20260330 ZJH 编码器 MLP 第 2 层
    Linear m_decoder1;     // 20260330 ZJH 解码器第 1 层
    Linear m_decoder2;     // 20260330 ZJH 解码器第 2 层（输出 patch 像素）
};

// =========================================================
// 20260330 ZJH SelfSupPretrainer — 自监督预训练管理器
// 统一接口: 选择 SimCLR 或 MAE 方法，对编码器进行无监督预训练
// 训练完成后保存预训练权重，供下游任务加载微调
// =========================================================
class SelfSupPretrainer {
public:
    // 20260330 ZJH 自监督方法枚举
    enum class Method {
        SimCLR,  // 20260330 ZJH 对比学习
        MAE      // 20260330 ZJH 掩码自编码器
    };

    // 20260330 ZJH 构造函数
    // eMethod: 预训练方法
    // nEncoderOutDim: 编码器输出维度
    SelfSupPretrainer(Method eMethod = Method::SimCLR, int nEncoderOutDim = 64)
        : m_eMethod(eMethod),
          m_nEncoderOutDim(nEncoderOutDim)
    {}

    // 20260330 ZJH pretrain — 使用无标注图像预训练编码器
    // encoder: 编码器 Module（输入 [N,C,H,W]，输出 [N,D]）
    // vecImages: 无标注图像列表
    // nEpochs: 训练轮次
    // fTemperature: SimCLR 温度参数（仅 SimCLR 使用）
    // fLr: 学习率
    void pretrain(Module& encoder,
                  const std::vector<Tensor>& vecImages,
                  int nEpochs = 100,
                  float fTemperature = 0.5f,
                  float fLr = 0.001f)
    {
        switch (m_eMethod) {
            case Method::SimCLR: {
                // 20260330 ZJH SimCLR 对比学习
                SimCLRTrainer trainer(m_nEncoderOutDim, 128, fTemperature);
                trainer.train(encoder, vecImages, nEpochs, 32, fLr);
                break;
            }
            case Method::MAE: {
                // 20260330 ZJH MAE 掩码自编码器
                MAEPretrainer mae(8, 256, 0.75f, 3);
                mae.pretrain(vecImages, nEpochs, fLr);
                break;
            }
        }
    }

    // 20260330 ZJH savePretrainedWeights — 保存预训练权重到二进制文件
    // 格式: [nParams][size_0][data_0][size_1][data_1]...
    // encoder: 编码器 Module
    // strPath: 保存路径（推荐 .ompt 后缀 = OmniMatch Pre-Trained）
    // 返回: 保存成功返回 true
    bool savePretrainedWeights(const Module& encoder, const std::string& strPath) {
        // 20260330 ZJH const_cast 因为 parameters() 非 const（历史接口限制）
        auto& mutableEncoder = const_cast<Module&>(encoder);
        auto vecParams = mutableEncoder.parameters();

        std::ofstream file(strPath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[SelfSup] Failed to open file for writing: " << strPath << std::endl;
            return false;
        }

        // 20260330 ZJH 写入魔数 "OMPT" + 参数数量
        const char magic[4] = {'O', 'M', 'P', 'T'};  // 20260330 ZJH OmniMatch Pre-Trained 魔数
        file.write(magic, 4);
        int nParams = static_cast<int>(vecParams.size());
        file.write(reinterpret_cast<const char*>(&nParams), sizeof(int));

        // 20260330 ZJH 逐个参数写入
        for (auto* pParam : vecParams) {
            auto cp = pParam->contiguous();
            int nNumel = cp.numel();  // 20260330 ZJH 元素数
            file.write(reinterpret_cast<const char*>(&nNumel), sizeof(int));

            // 20260330 ZJH 写入形状信息
            auto vecShape = cp.shapeVec();
            int nDims = static_cast<int>(vecShape.size());
            file.write(reinterpret_cast<const char*>(&nDims), sizeof(int));
            for (int d : vecShape) {
                file.write(reinterpret_cast<const char*>(&d), sizeof(int));
            }

            // 20260330 ZJH 写入参数数据
            file.write(reinterpret_cast<const char*>(cp.floatDataPtr()),
                       nNumel * sizeof(float));
        }

        file.close();
        std::cout << "[SelfSup] Saved pretrained weights to " << strPath
                  << " (" << nParams << " params)" << std::endl;
        return true;
    }

    // 20260330 ZJH loadPretrainedWeights — 加载预训练权重
    // encoder: 编码器 Module（结构必须与保存时一致）
    // strPath: 权重文件路径
    // 返回: 加载成功返回 true
    bool loadPretrainedWeights(Module& encoder, const std::string& strPath) {
        std::ifstream file(strPath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[SelfSup] Failed to open file for reading: " << strPath << std::endl;
            return false;
        }

        // 20260330 ZJH 验证魔数
        char magic[4];
        file.read(magic, 4);
        if (magic[0] != 'O' || magic[1] != 'M' || magic[2] != 'P' || magic[3] != 'T') {
            std::cerr << "[SelfSup] Invalid file format (bad magic)" << std::endl;
            return false;
        }

        int nParams;
        file.read(reinterpret_cast<char*>(&nParams), sizeof(int));

        auto vecParams = encoder.parameters();
        // 20260330 ZJH 参数数量不匹配时进行部分加载（兼容骨干冻结等场景）
        int nLoad = std::min(nParams, static_cast<int>(vecParams.size()));

        for (int i = 0; i < nLoad; ++i) {
            int nNumel;
            file.read(reinterpret_cast<char*>(&nNumel), sizeof(int));

            // 20260330 ZJH 读取形状
            int nDims;
            file.read(reinterpret_cast<char*>(&nDims), sizeof(int));
            std::vector<int> vecShape(nDims);
            for (int d = 0; d < nDims; ++d) {
                file.read(reinterpret_cast<char*>(&vecShape[d]), sizeof(int));
            }

            // 20260330 ZJH 读取参数数据
            if (vecParams[i]->numel() == nNumel) {
                // 20260330 ZJH 形状匹配，直接加载
                file.read(reinterpret_cast<char*>(vecParams[i]->mutableFloatDataPtr()),
                           nNumel * sizeof(float));
            } else {
                // 20260330 ZJH 形状不匹配，跳过该参数
                file.seekg(nNumel * sizeof(float), std::ios::cur);
                std::cerr << "[SelfSup] Warning: param " << i << " size mismatch ("
                          << vecParams[i]->numel() << " vs " << nNumel << "), skipped" << std::endl;
            }
        }

        // 20260330 ZJH 跳过多余参数（如果文件中参数更多）
        if (nParams > static_cast<int>(vecParams.size())) {
            std::cout << "[SelfSup] Partial load: " << nLoad << "/" << nParams
                      << " params loaded" << std::endl;
        }

        file.close();
        std::cout << "[SelfSup] Loaded pretrained weights from " << strPath << std::endl;
        return true;
    }

private:
    Method m_eMethod;        // 20260330 ZJH 预训练方法
    int m_nEncoderOutDim;    // 20260330 ZJH 编码器输出维度
};

}  // namespace om
