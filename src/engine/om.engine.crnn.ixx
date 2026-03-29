// 20260321 ZJH CRNN OCR 引擎模块 — Phase 5B
// 实现 LSTM 单元 + 双向 LSTM + CTC 解码 + CRNN 网络
// 架构: CNN 特征提取 → BiLSTM 序列建模 → FC 分类头 → CTC 解码
// 用于工业场景中的序列号/条码/铭牌文字识别
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <limits>

export module om.engine.crnn;

// 20260321 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 辅助函数（需在 CRNN 之前定义）
// =========================================================

// 20260321 ZJH tensorMaxPool2dAsym — 支持非对称核/步幅的最大池化
// 用于 CRNN 中 2x1 池化（竖直方向缩小，水平方向保持）
Tensor tensorMaxPool2dAsym(Tensor input, int nKH, int nKW, int nSH, int nSW) {
    auto ci = input.contiguous();
    auto vecShape = ci.shapeVec();
    int nBatch = vecShape[0];
    int nChannels = vecShape[1];
    int nH = vecShape[2];
    int nW = vecShape[3];

    int nOutH = (nH - nKH) / nSH + 1;  // 20260321 ZJH 输出高度
    int nOutW = (nW - nKW) / nSW + 1;  // 20260321 ZJH 输出宽度

    auto result = Tensor::zeros({nBatch, nChannels, nOutH, nOutW});
    const float* pIn = ci.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260321 ZJH 非对称最大池化
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nChannels; ++c) {
            for (int oh = 0; oh < nOutH; ++oh) {
                for (int ow = 0; ow < nOutW; ++ow) {
                    float fMax = -std::numeric_limits<float>::max();
                    for (int kh = 0; kh < nKH; ++kh) {
                        for (int kw = 0; kw < nKW; ++kw) {
                            int nIH = oh * nSH + kh;
                            int nIW = ow * nSW + kw;
                            float fVal = pIn[b * nChannels * nH * nW + c * nH * nW + nIH * nW + nIW];
                            fMax = std::max(fMax, fVal);
                        }
                    }
                    pOut[b * nChannels * nOutH * nOutW + c * nOutH * nOutW + oh * nOutW + ow] = fMax;
                }
            }
        }
    }

    return result;
}

// =========================================================
// LSTMCell — 单步 LSTM 计算单元
// =========================================================

// 20260321 ZJH LSTMCell — 实现单个时间步的 LSTM 计算
// 四个门: 遗忘门(f)、输入门(i)、候选值(g)、输出门(o)
// 公式:
//   [f, i, g, o] = [sigma, sigma, tanh, sigma](W_ih @ x + W_hh @ h + b)
//   c_new = f * c + i * g
//   h_new = o * tanh(c_new)
class LSTMCell : public Module {
public:
    // 20260321 ZJH 构造函数
    // nInputSize: 输入特征维度
    // nHiddenSize: 隐藏状态维度
    LSTMCell(int nInputSize, int nHiddenSize)
        : m_nInputSize(nInputSize), m_nHiddenSize(nHiddenSize)
    {
        // 20260321 ZJH 输入→门权重 W_ih: [input_size, 4*hidden_size]
        // 4 个门合并到一个矩阵中，提高计算效率
        m_weightIH = Tensor::randn({nInputSize, 4 * nHiddenSize});
        float fScale = std::sqrt(1.0f / static_cast<float>(nInputSize));
        float* pWih = m_weightIH.mutableFloatDataPtr();
        for (int i = 0; i < m_weightIH.numel(); ++i) {
            pWih[i] *= fScale;  // 20260321 ZJH 缩放初始化
        }
        registerParameter("weight_ih", m_weightIH);

        // 20260321 ZJH 隐藏→门权重 W_hh: [hidden_size, 4*hidden_size]
        m_weightHH = Tensor::randn({nHiddenSize, 4 * nHiddenSize});
        float fScaleH = std::sqrt(1.0f / static_cast<float>(nHiddenSize));
        float* pWhh = m_weightHH.mutableFloatDataPtr();
        for (int i = 0; i < m_weightHH.numel(); ++i) {
            pWhh[i] *= fScaleH;  // 20260321 ZJH 缩放初始化
        }
        registerParameter("weight_hh", m_weightHH);

        // 20260321 ZJH 偏置 b: [1, 4*hidden_size]，遗忘门偏置初始化为 1（有助于长期记忆）
        m_bias = Tensor::zeros({1, 4 * nHiddenSize});
        float* pB = m_bias.mutableFloatDataPtr();
        // 20260321 ZJH 遗忘门偏置设为 1.0（位于前 hidden_size 个元素）
        for (int i = 0; i < nHiddenSize; ++i) {
            pB[i] = 1.0f;  // 20260321 ZJH 鼓励初期不遗忘
        }
        registerParameter("bias", m_bias);
    }

    // 20260321 ZJH forward — 不使用此接口，LSTM 需要 (h, c) 状态
    Tensor forward(const Tensor& input) override {
        // 20260321 ZJH LSTM 使用 step() 接口，此处仅返回输入
        return input;
    }

    // 20260321 ZJH step — 单步 LSTM 前向
    // input: [batch, input_size] 当前时间步输入
    // h: [batch, hidden_size] 上一步隐藏状态
    // c: [batch, hidden_size] 上一步细胞状态
    // 返回: {h_new, c_new}
    struct LSTMState {
        Tensor h;  // 20260321 ZJH 新隐藏状态 [batch, hidden]
        Tensor c;  // 20260321 ZJH 新细胞状态 [batch, hidden]
    };

    LSTMState step(const Tensor& input, const Tensor& h, const Tensor& c) {
        // 20260321 ZJH gates = x @ W_ih + h @ W_hh + b
        // gates: [batch, 4*hidden_size]
        auto xProj = tensorMatmul(input, m_weightIH);  // 20260321 ZJH [batch, 4H]
        auto hProj = tensorMatmul(h, m_weightHH);      // 20260321 ZJH [batch, 4H]
        auto gates = tensorAdd(tensorAdd(xProj, hProj), m_bias);  // 20260321 ZJH [batch, 4H]

        int nH = m_nHiddenSize;  // 20260321 ZJH 简化命名

        // 20260321 ZJH 切分为 4 个门: [batch, H] 每个
        auto fGate = tensorSigmoid(tensorSliceLastDim(gates, 0, nH));        // 遗忘门
        auto iGate = tensorSigmoid(tensorSliceLastDim(gates, nH, 2 * nH));   // 输入门
        auto gGate = tensorTanh(tensorSliceLastDim(gates, 2 * nH, 3 * nH));  // 候选值
        auto oGate = tensorSigmoid(tensorSliceLastDim(gates, 3 * nH, 4 * nH)); // 输出门

        // 20260321 ZJH c_new = f * c + i * g（逐元素）
        auto cNew = tensorAdd(tensorMul(fGate, c), tensorMul(iGate, gGate));

        // 20260321 ZJH h_new = o * tanh(c_new)
        auto hNew = tensorMul(oGate, tensorTanh(cNew));

        return { hNew, cNew };
    }

    int hiddenSize() const { return m_nHiddenSize; }
    int inputSize() const { return m_nInputSize; }

private:
    int m_nInputSize;     // 20260321 ZJH 输入特征维度
    int m_nHiddenSize;    // 20260321 ZJH 隐藏状态维度
    Tensor m_weightIH;    // 20260321 ZJH 输入→门权重 [input, 4*hidden]
    Tensor m_weightHH;    // 20260321 ZJH 隐藏→门权重 [hidden, 4*hidden]
    Tensor m_bias;        // 20260321 ZJH 偏置 [1, 4*hidden]
};

// =========================================================
// BiLSTM — 双向 LSTM
// =========================================================

// 20260321 ZJH BiLSTM — 双向 LSTM 层
// 正向 LSTM 从左到右处理序列，反向 LSTM 从右到左
// 输出拼接: [h_forward; h_backward] 得到 2*hidden_size 维表示
class BiLSTM : public Module {
public:
    // 20260321 ZJH 构造函数
    // nInputSize: 输入特征维度
    // nHiddenSize: 每个方向的隐藏维度（输出维度 = 2 * nHiddenSize）
    BiLSTM(int nInputSize, int nHiddenSize)
        : m_nHiddenSize(nHiddenSize)
    {
        // 20260321 ZJH 正向和反向各一个 LSTMCell
        m_pForward = std::make_shared<LSTMCell>(nInputSize, nHiddenSize);
        m_pBackward = std::make_shared<LSTMCell>(nInputSize, nHiddenSize);
        registerModule("forward_lstm", m_pForward);
        registerModule("backward_lstm", m_pBackward);
    }

    // 20260321 ZJH forward — 双向 LSTM 前向
    // input: [batch, seq_len, input_size] 输入序列
    // 返回: [batch, seq_len, 2*hidden_size] 双向拼接输出
    Tensor forward(const Tensor& input) override {
        auto ci = input.contiguous();
        auto vecShape = ci.shapeVec();
        int nBatch = vecShape[0];       // 20260321 ZJH 批次大小
        int nSeqLen = vecShape[1];      // 20260321 ZJH 序列长度
        int nInputDim = vecShape[2];    // 20260321 ZJH 输入维度

        int nH = m_nHiddenSize;  // 20260321 ZJH 隐藏维度

        // 20260321 ZJH 初始化隐藏状态和细胞状态为零
        Tensor hFwd = Tensor::zeros({nBatch, nH});
        Tensor cFwd = Tensor::zeros({nBatch, nH});
        Tensor hBwd = Tensor::zeros({nBatch, nH});
        Tensor cBwd = Tensor::zeros({nBatch, nH});

        // 20260321 ZJH 提取每个时间步的输入切片 [batch, input_dim]
        std::vector<Tensor> vecInputSlices(nSeqLen);
        const float* pIn = ci.floatDataPtr();
        for (int t = 0; t < nSeqLen; ++t) {
            auto slice = Tensor::zeros({nBatch, nInputDim});
            float* pSlice = slice.mutableFloatDataPtr();
            // 20260321 ZJH 拷贝第 t 个时间步数据
            for (int b = 0; b < nBatch; ++b) {
                for (int d = 0; d < nInputDim; ++d) {
                    pSlice[b * nInputDim + d] = pIn[b * nSeqLen * nInputDim + t * nInputDim + d];
                }
            }
            vecInputSlices[t] = slice;
        }

        // 20260321 ZJH 正向 LSTM：从左到右
        std::vector<Tensor> vecFwdOutputs(nSeqLen);
        for (int t = 0; t < nSeqLen; ++t) {
            auto state = m_pForward->step(vecInputSlices[t], hFwd, cFwd);
            hFwd = state.h;  // 20260321 ZJH 更新正向隐藏状态
            cFwd = state.c;  // 20260321 ZJH 更新正向细胞状态
            vecFwdOutputs[t] = hFwd;  // 20260321 ZJH 保存正向输出
        }

        // 20260321 ZJH 反向 LSTM：从右到左
        std::vector<Tensor> vecBwdOutputs(nSeqLen);
        for (int t = nSeqLen - 1; t >= 0; --t) {
            auto state = m_pBackward->step(vecInputSlices[t], hBwd, cBwd);
            hBwd = state.h;  // 20260321 ZJH 更新反向隐藏状态
            cBwd = state.c;  // 20260321 ZJH 更新反向细胞状态
            vecBwdOutputs[t] = hBwd;  // 20260321 ZJH 保存反向输出
        }

        // 20260321 ZJH 拼接正向和反向输出: [batch, 2*hidden]
        // 然后组装为 [batch, seq_len, 2*hidden]
        auto result = Tensor::zeros({nBatch, nSeqLen, 2 * nH});
        float* pOut = result.mutableFloatDataPtr();
        for (int t = 0; t < nSeqLen; ++t) {
            // 20260321 ZJH 拼接第 t 步的正向和反向输出
            auto catted = tensorConcatLastDim(vecFwdOutputs[t], vecBwdOutputs[t]);
            const float* pCat = catted.contiguous().floatDataPtr();
            // 20260321 ZJH 写入结果的第 t 时间步
            for (int b = 0; b < nBatch; ++b) {
                for (int d = 0; d < 2 * nH; ++d) {
                    pOut[b * nSeqLen * 2 * nH + t * 2 * nH + d] = pCat[b * 2 * nH + d];
                }
            }
        }

        return result;
    }

    int outputSize() const { return 2 * m_nHiddenSize; }

private:
    int m_nHiddenSize;  // 20260321 ZJH 每个方向的隐藏维度
    std::shared_ptr<LSTMCell> m_pForward;   // 20260321 ZJH 正向 LSTM
    std::shared_ptr<LSTMCell> m_pBackward;  // 20260321 ZJH 反向 LSTM
};

// =========================================================
// CTCDecoder — CTC 贪心解码器
// =========================================================

// 20260321 ZJH CTCDecoder — CTC (Connectionist Temporal Classification) 解码
// 贪心解码: 每个时间步取 argmax，合并连续重复字符，去除空白符
class CTCDecoder {
public:
    // 20260321 ZJH 构造函数
    // nBlankIndex: 空白符在字符集中的索引（通常为 0 或最后一个）
    CTCDecoder(int nBlankIndex = 0) : m_nBlankIndex(nBlankIndex) {}

    // 20260321 ZJH greedyDecode — 贪心解码单个样本
    // logits: [seq_len, num_classes] 每个时间步的类别分数
    // 返回: 解码后的字符索引序列（去重+去空白）
    std::vector<int> greedyDecode(const Tensor& logits) {
        auto cl = logits.contiguous();
        auto vecShape = cl.shapeVec();
        int nSeqLen = vecShape[0];     // 20260321 ZJH 序列长度
        int nClasses = vecShape[1];    // 20260321 ZJH 类别数
        const float* pData = cl.floatDataPtr();

        std::vector<int> vecRaw;  // 20260321 ZJH 每步 argmax 结果
        vecRaw.reserve(nSeqLen);

        // 20260321 ZJH 每个时间步取 argmax
        for (int t = 0; t < nSeqLen; ++t) {
            int nBestIdx = 0;
            float fBestVal = pData[t * nClasses];
            for (int c = 1; c < nClasses; ++c) {
                float fVal = pData[t * nClasses + c];
                if (fVal > fBestVal) {
                    fBestVal = fVal;
                    nBestIdx = c;
                }
            }
            vecRaw.push_back(nBestIdx);
        }

        // 20260321 ZJH CTC 合并规则：去除连续重复 + 去除空白符
        std::vector<int> vecDecoded;
        int nPrev = -1;  // 20260321 ZJH 前一个非空白字符
        for (int idx : vecRaw) {
            if (idx != m_nBlankIndex && idx != nPrev) {
                vecDecoded.push_back(idx);  // 20260321 ZJH 新字符
            }
            nPrev = idx;  // 20260321 ZJH 更新前一个字符
        }

        return vecDecoded;
    }

    // 20260321 ZJH batchGreedyDecode — 批量贪心解码
    // logits: [batch, seq_len, num_classes]
    // 返回: 每个样本的解码结果
    std::vector<std::vector<int>> batchGreedyDecode(const Tensor& logits) {
        auto cl = logits.contiguous();
        auto vecShape = cl.shapeVec();
        int nBatch = vecShape[0];
        int nSeqLen = vecShape[1];
        int nClasses = vecShape[2];

        std::vector<std::vector<int>> vecResults;
        vecResults.reserve(nBatch);

        // 20260321 ZJH 逐样本解码
        for (int b = 0; b < nBatch; ++b) {
            // 20260321 ZJH 提取第 b 个样本的 logits [seq_len, classes]
            auto sampleLogits = Tensor::zeros({nSeqLen, nClasses});
            float* pDst = sampleLogits.mutableFloatDataPtr();
            const float* pSrc = cl.floatDataPtr() + b * nSeqLen * nClasses;
            for (int i = 0; i < nSeqLen * nClasses; ++i) {
                pDst[i] = pSrc[i];
            }
            vecResults.push_back(greedyDecode(sampleLogits));
        }

        return vecResults;
    }

private:
    int m_nBlankIndex;  // 20260321 ZJH 空白符索引
};

// =========================================================
// CTCLoss — CTC 损失函数（简化版前向-后向算法）
// =========================================================

// 20260321 ZJH CTCLoss — 简化 CTC 损失
// 使用标准 CTC 前向-后向算法计算对数似然损失
// 空白符标签默认为 index 0
class CTCLoss {
public:
    // 20260321 ZJH 构造函数
    // nBlankIndex: 空白符索引
    CTCLoss(int nBlankIndex = 0) : m_nBlankIndex(nBlankIndex) {}

    // 20260321 ZJH forward — 计算 CTC 损失（简化版）
    // logits: [batch, seq_len, num_classes] — 网络输出（logits，非概率）
    // targets: 每个样本的目标标签序列
    // 返回: 标量损失张量
    // 20260325 ZJH GPU 安全修复：CTC loss 内部全部用 CPU 指针操作，GPU 张量先迁移到 CPU
    Tensor forward(const Tensor& logits, const std::vector<std::vector<int>>& vecTargets) {
        auto cl = (logits.isCuda() ? logits.cpu() : logits).contiguous();
        auto vecShape = cl.shapeVec();
        int nBatch = vecShape[0];
        int nT = vecShape[1];        // 20260321 ZJH 时间步数（序列长度）
        int nC = vecShape[2];        // 20260321 ZJH 类别数

        float fTotalLoss = 0.0f;

        // 20260321 ZJH 逐样本计算 CTC loss
        for (int b = 0; b < nBatch; ++b) {
            const auto& vecTarget = vecTargets[b];  // 20260321 ZJH 当前样本的目标序列
            int nL = static_cast<int>(vecTarget.size());  // 20260321 ZJH 目标长度

            // 20260321 ZJH 构建扩展标签序列：插入空白符
            // 例如 [1,2,3] -> [0,1,0,2,0,3,0]，长度 = 2*L + 1
            int nS = 2 * nL + 1;
            std::vector<int> vecExtended(nS);
            for (int i = 0; i < nS; ++i) {
                vecExtended[i] = (i % 2 == 0) ? m_nBlankIndex : vecTarget[i / 2];
            }

            // 20260321 ZJH 计算 log softmax：log_prob[t][c] = log(softmax(logits[t]))
            std::vector<std::vector<float>> vecLogProb(nT, std::vector<float>(nC));
            const float* pLogits = cl.floatDataPtr() + b * nT * nC;
            for (int t = 0; t < nT; ++t) {
                // 20260321 ZJH log-sum-exp 技巧
                float fMax = pLogits[t * nC];
                for (int c = 1; c < nC; ++c) {
                    fMax = std::max(fMax, pLogits[t * nC + c]);
                }
                float fSumExp = 0.0f;
                for (int c = 0; c < nC; ++c) {
                    fSumExp += std::exp(pLogits[t * nC + c] - fMax);
                }
                float fLogSumExp = fMax + std::log(fSumExp + 1e-10f);
                for (int c = 0; c < nC; ++c) {
                    vecLogProb[t][c] = pLogits[t * nC + c] - fLogSumExp;
                }
            }

            // 20260321 ZJH CTC 前向算法（对数域）
            // alpha[t][s] = 在时间步 t 处于扩展标签位置 s 的对数前向概率
            const float fNegInf = -1e30f;
            std::vector<std::vector<float>> vecAlpha(nT, std::vector<float>(nS, fNegInf));

            // 20260321 ZJH 初始化: t=0 只能在前两个位置（空白 或 第一个字符）
            vecAlpha[0][0] = vecLogProb[0][vecExtended[0]];
            if (nS > 1) {
                vecAlpha[0][1] = vecLogProb[0][vecExtended[1]];
            }

            // 20260321 ZJH 前向递推
            for (int t = 1; t < nT; ++t) {
                for (int s = 0; s < nS; ++s) {
                    // 20260321 ZJH 从同一位置转移
                    float fLogSum = vecAlpha[t - 1][s];

                    // 20260321 ZJH 从前一个位置转移
                    if (s > 0) {
                        fLogSum = logSumExp(fLogSum, vecAlpha[t - 1][s - 1]);
                    }

                    // 20260321 ZJH 跳过空白符转移（前提：当前非空白且与 s-2 不同）
                    if (s > 1 && vecExtended[s] != m_nBlankIndex &&
                        vecExtended[s] != vecExtended[s - 2]) {
                        fLogSum = logSumExp(fLogSum, vecAlpha[t - 1][s - 2]);
                    }

                    // 20260321 ZJH 加上当前时间步的发射概率
                    vecAlpha[t][s] = fLogSum + vecLogProb[t][vecExtended[s]];
                }
            }

            // 20260321 ZJH 损失 = -log P(target | input)
            // 最终概率 = alpha[T-1][S-1] + alpha[T-1][S-2]
            float fLogProb = vecAlpha[nT - 1][nS - 1];
            if (nS > 1) {
                fLogProb = logSumExp(fLogProb, vecAlpha[nT - 1][nS - 2]);
            }

            fTotalLoss += -fLogProb;  // 20260321 ZJH 累加负对数似然
        }

        // 20260321 ZJH 平均损失
        fTotalLoss /= static_cast<float>(nBatch);

        auto result = Tensor::zeros({1});
        result.mutableFloatDataPtr()[0] = fTotalLoss;
        return result;
    }

private:
    int m_nBlankIndex;  // 20260321 ZJH 空白符索引

    // 20260321 ZJH logSumExp — 对数域安全加法
    // log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    static float logSumExp(float a, float b) {
        if (a == -1e30f) return b;
        if (b == -1e30f) return a;
        float fMax = std::max(a, b);
        return fMax + std::log(std::exp(a - fMax) + std::exp(b - fMax));
    }
};

// =========================================================
// CRNN — 完整 OCR 网络
// =========================================================

// 20260321 ZJH CRNN — CNN + RNN + CTC 端到端 OCR 网络
// 输入: 灰度图像 [batch, 1, 32, W]（高度固定 32，宽度可变）
// 输出: [batch, seq_len, num_classes] 每个时间步的类别 logits
// 其中 seq_len = W / 4（经过两次 stride=2 池化后的宽度）
class CRNN : public Module {
public:
    // 20260321 ZJH 构造函数
    // nNumClasses: 字符类别数（含空白符）
    // nHiddenSize: LSTM 隐藏维度
    // nImgHeight: 输入图像高度（默认 32）
    CRNN(int nNumClasses, int nHiddenSize = 128, int nImgHeight = 32)
        : m_nNumClasses(nNumClasses), m_nHiddenSize(nHiddenSize), m_nImgHeight(nImgHeight)
    {
        // 20260321 ZJH CNN 特征提取器（VGG 风格）
        // Conv 层逐步增加通道数，Pool 层缩小空间尺寸
        // 输入: [B, 1, 32, W]
        m_pConv1 = std::make_shared<Conv2d>(1, 64, 3, 1, 1);    // -> [B, 64, 32, W]
        m_pConv2 = std::make_shared<Conv2d>(64, 128, 3, 1, 1);   // -> [B, 128, 16, W/2] (after pool)
        m_pConv3 = std::make_shared<Conv2d>(128, 256, 3, 1, 1);  // -> [B, 256, 8, W/4] (after pool)
        m_pConv4 = std::make_shared<Conv2d>(256, 256, 3, 1, 1);  // -> [B, 256, 8, W/4]
        m_pConv5 = std::make_shared<Conv2d>(256, 512, 3, 1, 1);  // -> [B, 512, 4, W/4]
        m_pConv6 = std::make_shared<Conv2d>(512, 512, 3, 1, 1);  // -> [B, 512, 4, W/4]

        // 20260321 ZJH BatchNorm 层加速收敛
        m_pBn1 = std::make_shared<BatchNorm2d>(64);
        m_pBn2 = std::make_shared<BatchNorm2d>(128);
        m_pBn3 = std::make_shared<BatchNorm2d>(256);
        m_pBn4 = std::make_shared<BatchNorm2d>(256);
        m_pBn5 = std::make_shared<BatchNorm2d>(512);
        m_pBn6 = std::make_shared<BatchNorm2d>(512);

        // 20260321 ZJH 经过 CNN 后，特征图高度 = imgHeight / 8 = 4
        // 列方向特征维度 = 512 * 4 = 2048，映射到 LSTM 输入维度
        int nCnnOutputDim = 512 * (nImgHeight / 8);
        m_pMapToSeq = std::make_shared<Linear>(nCnnOutputDim, nHiddenSize);

        // 20260321 ZJH 双向 LSTM（2 层堆叠）
        m_pBiLSTM1 = std::make_shared<BiLSTM>(nHiddenSize, nHiddenSize);
        m_pBiLSTM2 = std::make_shared<BiLSTM>(2 * nHiddenSize, nHiddenSize);

        // 20260321 ZJH 分类头: 2*hidden -> num_classes
        m_pClassifier = std::make_shared<Linear>(2 * nHiddenSize, nNumClasses);

        // 20260321 ZJH 注册所有子模块用于参数管理
        registerModule("conv1", m_pConv1);
        registerModule("bn1", m_pBn1);
        registerModule("conv2", m_pConv2);
        registerModule("bn2", m_pBn2);
        registerModule("conv3", m_pConv3);
        registerModule("bn3", m_pBn3);
        registerModule("conv4", m_pConv4);
        registerModule("bn4", m_pBn4);
        registerModule("conv5", m_pConv5);
        registerModule("bn5", m_pBn5);
        registerModule("conv6", m_pConv6);
        registerModule("bn6", m_pBn6);
        registerModule("map_to_seq", m_pMapToSeq);
        registerModule("bilstm1", m_pBiLSTM1);
        registerModule("bilstm2", m_pBiLSTM2);
        registerModule("classifier", m_pClassifier);
    }

    // 20260321 ZJH forward — CRNN 前向传播
    // input: [batch, 1, 32, W] 灰度图像
    // 返回: [batch, seq_len, num_classes] 每个时间步的 logits
    Tensor forward(const Tensor& input) override {
        auto ci = input.contiguous();
        auto vecShape = ci.shapeVec();
        int nBatch = vecShape[0];
        int nW = vecShape[3];  // 20260321 ZJH 原始宽度

        // 20260321 ZJH CNN 特征提取
        auto x = tensorReLU(m_pBn1->forward(m_pConv1->forward(ci)));
        x = tensorMaxPool2d(x, 2, 2, 0);  // 20260321 ZJH [B, 64, 16, W/2]

        x = tensorReLU(m_pBn2->forward(m_pConv2->forward(x)));
        x = tensorMaxPool2d(x, 2, 2, 0);  // 20260321 ZJH [B, 128, 8, W/4]

        x = tensorReLU(m_pBn3->forward(m_pConv3->forward(x)));
        x = tensorReLU(m_pBn4->forward(m_pConv4->forward(x)));
        x = tensorMaxPool2dAsym(x, 2, 1, 2, 1);  // 20260321 ZJH [B, 256, 4, W/4] — 竖直方向池化

        x = tensorReLU(m_pBn5->forward(m_pConv5->forward(x)));
        x = tensorReLU(m_pBn6->forward(m_pConv6->forward(x)));

        // 20260321 ZJH 此时 x: [B, 512, H', W'] 其中 H'=imgH/8, W'=W/4
        auto vecFeatShape = x.contiguous().shapeVec();
        int nFeatH = vecFeatShape[2];
        int nFeatW = vecFeatShape[3];
        int nChannels = vecFeatShape[1];

        // 20260321 ZJH 将 CNN 特征图转换为序列
        // 每一列作为一个时间步: [B, C*H, W'] -> [B, W', C*H]
        auto xCont = x.contiguous();
        int nColDim = nChannels * nFeatH;  // 20260321 ZJH 每列的特征维度
        auto seqInput = Tensor::zeros({nBatch, nFeatW, nColDim});
        const float* pFeat = xCont.floatDataPtr();
        float* pSeq = seqInput.mutableFloatDataPtr();

        // 20260321 ZJH 重排: [B, C, H, W] -> [B, W, C*H]
        for (int b = 0; b < nBatch; ++b) {
            for (int w = 0; w < nFeatW; ++w) {
                for (int c = 0; c < nChannels; ++c) {
                    for (int h = 0; h < nFeatH; ++h) {
                        int nSrcIdx = b * nChannels * nFeatH * nFeatW + c * nFeatH * nFeatW + h * nFeatW + w;
                        int nDstIdx = b * nFeatW * nColDim + w * nColDim + c * nFeatH + h;
                        pSeq[nDstIdx] = pFeat[nSrcIdx];
                    }
                }
            }
        }

        // 20260321 ZJH 映射到 LSTM 输入维度: [B, W', colDim] -> [B, W', hidden]
        // 逐时间步应用 Linear
        auto mappedSeq = Tensor::zeros({nBatch, nFeatW, m_nHiddenSize});
        float* pMapped = mappedSeq.mutableFloatDataPtr();
        for (int t = 0; t < nFeatW; ++t) {
            // 20260321 ZJH 提取第 t 步 [B, colDim]
            auto stepInput = Tensor::zeros({nBatch, nColDim});
            float* pStep = stepInput.mutableFloatDataPtr();
            for (int b = 0; b < nBatch; ++b) {
                for (int d = 0; d < nColDim; ++d) {
                    pStep[b * nColDim + d] = pSeq[b * nFeatW * nColDim + t * nColDim + d];
                }
            }

            auto stepOut = m_pMapToSeq->forward(stepInput);  // 20260321 ZJH [B, hidden]
            const float* pStepOut = stepOut.contiguous().floatDataPtr();
            for (int b = 0; b < nBatch; ++b) {
                for (int d = 0; d < m_nHiddenSize; ++d) {
                    pMapped[b * nFeatW * m_nHiddenSize + t * m_nHiddenSize + d] =
                        pStepOut[b * m_nHiddenSize + d];
                }
            }
        }

        // 20260321 ZJH 双向 LSTM 序列建模
        auto lstmOut = m_pBiLSTM1->forward(mappedSeq);  // [B, W', 2*hidden]
        lstmOut = m_pBiLSTM2->forward(lstmOut);          // [B, W', 2*hidden]

        // 20260321 ZJH 分类头: 逐时间步 [B, 2*hidden] -> [B, num_classes]
        auto lstmCont = lstmOut.contiguous();
        int nSeqLen = nFeatW;
        int nLstmDim = 2 * m_nHiddenSize;
        auto output = Tensor::zeros({nBatch, nSeqLen, m_nNumClasses});
        float* pOutput = output.mutableFloatDataPtr();

        for (int t = 0; t < nSeqLen; ++t) {
            auto stepIn = Tensor::zeros({nBatch, nLstmDim});
            float* pStepIn = stepIn.mutableFloatDataPtr();
            const float* pLstm = lstmCont.floatDataPtr();
            for (int b = 0; b < nBatch; ++b) {
                for (int d = 0; d < nLstmDim; ++d) {
                    pStepIn[b * nLstmDim + d] = pLstm[b * nSeqLen * nLstmDim + t * nLstmDim + d];
                }
            }

            auto stepOut = m_pClassifier->forward(stepIn);  // 20260321 ZJH [B, num_classes]
            const float* pOut = stepOut.contiguous().floatDataPtr();
            for (int b = 0; b < nBatch; ++b) {
                for (int c = 0; c < m_nNumClasses; ++c) {
                    pOutput[b * nSeqLen * m_nNumClasses + t * m_nNumClasses + c] =
                        pOut[b * m_nNumClasses + c];
                }
            }
        }

        return output;
    }

    int numClasses() const { return m_nNumClasses; }
    int hiddenSize() const { return m_nHiddenSize; }

private:
    int m_nNumClasses;   // 20260321 ZJH 字符类别数（含空白符）
    int m_nHiddenSize;   // 20260321 ZJH LSTM 隐藏维度
    int m_nImgHeight;    // 20260321 ZJH 输入图像高度

    // 20260321 ZJH CNN 特征提取层
    std::shared_ptr<Conv2d> m_pConv1, m_pConv2, m_pConv3, m_pConv4, m_pConv5, m_pConv6;
    std::shared_ptr<BatchNorm2d> m_pBn1, m_pBn2, m_pBn3, m_pBn4, m_pBn5, m_pBn6;

    // 20260321 ZJH 序列映射层
    std::shared_ptr<Linear> m_pMapToSeq;

    // 20260321 ZJH 双层双向 LSTM
    std::shared_ptr<BiLSTM> m_pBiLSTM1, m_pBiLSTM2;

    // 20260321 ZJH 分类头
    std::shared_ptr<Linear> m_pClassifier;
};

}  // namespace om
