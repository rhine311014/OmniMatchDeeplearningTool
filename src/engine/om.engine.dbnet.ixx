// 20260402 ZJH DBNet — Differentiable Binarization Network 文本检测模块
// 对标 Halcon Deep OCR 文本检测 + Cognex ViDi Blue Read 定位能力
// 架构: ResNet18 backbone → FPN 特征金字塔 → DB Head（概率图 + 阈值图）
// 核心创新: 可微分二值化 B = sigmoid(k * (P - T))
//   P: 概率图（哪里有文字）
//   T: 阈值图（每个位置的最优阈值，可学习）
//   k: 放大因子（默认 50，使 sigmoid 接近阶跃函数）
// 推理后处理: 概率图二值化 → 连通域分析 → 最小外接矩形 → 文本框输出
// 训练: 监督学习，需文本框标注（复用现有 BBox 标注类型）
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>

export module om.engine.dbnet;

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
// 20260402 ZJH TextBox — 文本检测结果结构
// 表示一个检测到的文本区域
// =============================================================================
struct TextBox {
    float fX1, fY1;      // 20260402 ZJH 左上角坐标（归一化 [0,1]）
    float fX2, fY2;      // 20260402 ZJH 右下角坐标（归一化 [0,1]）
    float fScore;         // 20260402 ZJH 检测置信度 [0, 1]
    float fAngle;         // 20260402 ZJH 旋转角度（度，用于倾斜文本）
};

// =============================================================================
// 20260402 ZJH FPNBlock — Feature Pyramid Network 单层上采样融合块
// 将高层语义特征上采样后与低层空间特征融合
// =============================================================================
class FPNBlock : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入通道数（高层特征）
    // nOutChannels: 输出通道数（融合后统一通道）
    FPNBlock(int nInChannels, int nOutChannels)
        : m_conv1x1(nInChannels, nOutChannels, 1, 1, 0, true),  // 20260402 ZJH 1x1 通道对齐
          m_conv3x3(nOutChannels, nOutChannels, 3, 1, 1, false), // 20260402 ZJH 3x3 平滑
          m_bn(nOutChannels)
    {
        // 20260402 ZJH 值成员，通过 parameters() 手动管理
    }

    // 20260402 ZJH forward — 通道对齐 + 平滑
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, H, W]
    Tensor forward(const Tensor& input) override {
        auto x = m_conv1x1.forward(input);   // 20260402 ZJH 通道对齐
        x = m_conv3x3.forward(x);            // 20260402 ZJH 平滑
        x = m_bn.forward(x);                 // 20260402 ZJH 归一化
        x = ReLU().forward(x);                   // 20260402 ZJH 激活
        return x;
    }

private:
    Conv2d m_conv1x1;   // 20260402 ZJH 1x1 卷积
    Conv2d m_conv3x3;   // 20260402 ZJH 3x3 卷积
    BatchNorm2d m_bn;    // 20260402 ZJH 批归一化
};

// =============================================================================
// 20260402 ZJH DBHead — Differentiable Binarization Head
// 输入 FPN 融合特征 → 输出概率图 P 和阈值图 T
// 可微分二值化: B = sigmoid(k * (P - T))
// =============================================================================
class DBHead : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入特征通道数（FPN 输出）
    // fK: 放大因子（控制 sigmoid 陡峭程度，默认 50）
    DBHead(int nInChannels, float fK = 50.0f)
        : m_fK(fK),
          m_convProb1(nInChannels, nInChannels / 4, 3, 1, 1, false),  // 20260402 ZJH 概率分支 conv1
          m_bnProb1(nInChannels / 4),
          m_convProb2(nInChannels / 4, 1, 1, 1, 0, true),             // 20260402 ZJH 概率分支输出（单通道）
          m_convThresh1(nInChannels, nInChannels / 4, 3, 1, 1, false), // 20260402 ZJH 阈值分支 conv1
          m_bnThresh1(nInChannels / 4),
          m_convThresh2(nInChannels / 4, 1, 1, 1, 0, true)            // 20260402 ZJH 阈值分支输出（单通道）
    {
        // 20260402 ZJH 值成员，通过 parameters() 手动管理
    }

    // 20260402 ZJH forward — 输出概率图（训练时还输出阈值图用于 DB loss）
    // features: [N, C, H, W] FPN 融合特征
    // 返回: [N, 2, H, W]  — channel 0 = 概率图 P, channel 1 = 阈值图 T
    //   或训练模式下 [N, 3, H, W] — channel 2 = 可微分二值图 B
    Tensor forward(const Tensor& features) override {
        int nBatch = features.shape(0);
        int nH = features.shape(2);
        int nW = features.shape(3);

        // 20260402 ZJH 概率分支: conv → bn → relu → conv → sigmoid → P
        auto probFeat = ReLU().forward(m_bnProb1.forward(m_convProb1.forward(features)));  // 20260402 ZJH [N, C/4, H, W]
        auto probLogit = m_convProb2.forward(probFeat);  // 20260402 ZJH [N, 1, H, W] raw logit

        // 20260402 ZJH 阈值分支: conv → bn → relu → conv → sigmoid → T
        auto threshFeat = ReLU().forward(m_bnThresh1.forward(m_convThresh1.forward(features)));
        auto threshLogit = m_convThresh2.forward(threshFeat);  // 20260402 ZJH [N, 1, H, W] raw logit

        // 20260402 ZJH sigmoid 激活得到 P 和 T
        auto probMap = tensorSigmoid(probLogit);     // 20260402 ZJH P ∈ [0, 1]
        auto threshMap = tensorSigmoid(threshLogit); // 20260402 ZJH T ∈ [0, 1]

        // 20260402 ZJH 可微分二值化: B = sigmoid(k * (P - T))
        auto diff = tensorSub(probMap, threshMap);  // 20260402 ZJH P - T
        auto scaled = tensorMulScalar(diff, m_fK);   // 20260402 ZJH k * (P - T)
        auto binaryMap = tensorSigmoid(scaled);      // 20260402 ZJH B = sigmoid(k*(P-T))

        // 20260402 ZJH 拼接输出 [N, 3, H, W]: P, T, B
        auto output = Tensor::zeros({nBatch, 3, nH, nW});
        float* pOut = output.mutableFloatDataPtr();
        const float* pP = probMap.contiguous().floatDataPtr();
        const float* pT = threshMap.contiguous().floatDataPtr();
        const float* pB = binaryMap.contiguous().floatDataPtr();
        int nSpatial = nH * nW;

        for (int b = 0; b < nBatch; ++b) {
            for (int i = 0; i < nSpatial; ++i) {
                pOut[(b * 3 + 0) * nSpatial + i] = pP[b * nSpatial + i];  // 20260402 ZJH P
                pOut[(b * 3 + 1) * nSpatial + i] = pT[b * nSpatial + i];  // 20260402 ZJH T
                pOut[(b * 3 + 2) * nSpatial + i] = pB[b * nSpatial + i];  // 20260402 ZJH B
            }
        }

        return output;  // 20260402 ZJH [N, 3, H, W]
    }

private:
    float m_fK;  // 20260402 ZJH 放大因子

    // 20260402 ZJH 概率分支
    Conv2d m_convProb1;
    BatchNorm2d m_bnProb1;
    Conv2d m_convProb2;

    // 20260402 ZJH 阈值分支
    Conv2d m_convThresh1;
    BatchNorm2d m_bnThresh1;
    Conv2d m_convThresh2;
};

// =============================================================================
// 20260402 ZJH 辅助函数（必须在 DBNet 之前定义，DBNet::forward 中调用）
// =============================================================================

// 20260402 ZJH tensorUpsampleNN — 最近邻上采样
// input: [N, C, H, W] → [N, C, targetH, targetW]
inline Tensor tensorUpsampleNN(const Tensor& input, int nTargetH, int nTargetW) {
    auto cInput = input.contiguous();
    int nN = cInput.shape(0), nC = cInput.shape(1);
    int nH = cInput.shape(2), nW = cInput.shape(3);
    auto result = Tensor::zeros({nN, nC, nTargetH, nTargetW});
    float* pOut = result.mutableFloatDataPtr();
    const float* pIn = cInput.floatDataPtr();
    float fScaleH = static_cast<float>(nH) / static_cast<float>(nTargetH);
    float fScaleW = static_cast<float>(nW) / static_cast<float>(nTargetW);
    for (int n = 0; n < nN; ++n) {
        for (int c = 0; c < nC; ++c) {
            for (int th = 0; th < nTargetH; ++th) {
                int nSrcH = std::min(static_cast<int>(th * fScaleH), nH - 1);
                for (int tw = 0; tw < nTargetW; ++tw) {
                    int nSrcW = std::min(static_cast<int>(tw * fScaleW), nW - 1);
                    pOut[((n * nC + c) * nTargetH + th) * nTargetW + tw] =
                        pIn[((n * nC + c) * nH + nSrcH) * nW + nSrcW];
                }
            }
        }
    }
    return result;
}

// 20260402 ZJH tensorCatChannels4 — 沿通道维度拼接 4 个张量
inline Tensor tensorCatChannels4(const Tensor& a, const Tensor& b,
                                  const Tensor& c, const Tensor& d) {
    auto ca = a.contiguous(), cb = b.contiguous();
    auto cc = c.contiguous(), cd = d.contiguous();
    int nN = ca.shape(0);
    int nCa = ca.shape(1), nCb = cb.shape(1), nCc = cc.shape(1), nCd = cd.shape(1);
    int nH = ca.shape(2), nW = ca.shape(3);
    int nCtotal = nCa + nCb + nCc + nCd;
    int nSpatial = nH * nW;
    auto result = Tensor::zeros({nN, nCtotal, nH, nW});
    float* pOut = result.mutableFloatDataPtr();
    const float* pA = ca.floatDataPtr();
    const float* pB = cb.floatDataPtr();
    const float* pC = cc.floatDataPtr();
    const float* pD = cd.floatDataPtr();
    for (int n = 0; n < nN; ++n) {
        int nOff = 0;
        for (int ch = 0; ch < nCa; ++ch)
            for (int i = 0; i < nSpatial; ++i)
                pOut[((n * nCtotal + nOff + ch) * nSpatial) + i] = pA[((n * nCa + ch) * nSpatial) + i];
        nOff += nCa;
        for (int ch = 0; ch < nCb; ++ch)
            for (int i = 0; i < nSpatial; ++i)
                pOut[((n * nCtotal + nOff + ch) * nSpatial) + i] = pB[((n * nCb + ch) * nSpatial) + i];
        nOff += nCb;
        for (int ch = 0; ch < nCc; ++ch)
            for (int i = 0; i < nSpatial; ++i)
                pOut[((n * nCtotal + nOff + ch) * nSpatial) + i] = pC[((n * nCc + ch) * nSpatial) + i];
        nOff += nCc;
        for (int ch = 0; ch < nCd; ++ch)
            for (int i = 0; i < nSpatial; ++i)
                pOut[((n * nCtotal + nOff + ch) * nSpatial) + i] = pD[((n * nCd + ch) * nSpatial) + i];
    }
    return result;
}

// =============================================================================
// 20260402 ZJH DBNet — 完整 Differentiable Binarization Network
// ResNet18 backbone → 4 级 FPN → DB Head → 文本检测
// =============================================================================
class DBNet : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入图像通道数（默认 3 = RGB）
    // nFpnChannels: FPN 统一通道数（默认 64，轻量设计）
    DBNet(int nInChannels = 3, int nFpnChannels = 64)
        : m_nFpnChannels(nFpnChannels),
          // 20260402 ZJH ResNet18 风格 backbone（4 阶段）
          m_convStem(nInChannels, 64, 7, 2, 3, false),  // 20260402 ZJH stem: /2
          m_bnStem(64),
          m_poolStem(3, 2, 1),                            // 20260402 ZJH stem pool: /4
          // 20260402 ZJH Stage 1: 64→64, /4
          m_convS1a(64, 64, 3, 1, 1, false), m_bnS1a(64),
          m_convS1b(64, 64, 3, 1, 1, false), m_bnS1b(64),
          // 20260402 ZJH Stage 2: 64→128, /8
          m_convS2a(64, 128, 3, 2, 1, false), m_bnS2a(128),
          m_convS2b(128, 128, 3, 1, 1, false), m_bnS2b(128),
          m_convS2down(64, 128, 1, 2, 0, false), m_bnS2down(128),  // 20260402 ZJH shortcut
          // 20260402 ZJH Stage 3: 128→256, /16
          m_convS3a(128, 256, 3, 2, 1, false), m_bnS3a(256),
          m_convS3b(256, 256, 3, 1, 1, false), m_bnS3b(256),
          m_convS3down(128, 256, 1, 2, 0, false), m_bnS3down(256),
          // 20260402 ZJH Stage 4: 256→512, /32
          m_convS4a(256, 512, 3, 2, 1, false), m_bnS4a(512),
          m_convS4b(512, 512, 3, 1, 1, false), m_bnS4b(512),
          m_convS4down(256, 512, 1, 2, 0, false), m_bnS4down(512),
          // 20260402 ZJH FPN 侧向连接
          m_fpn4(512, nFpnChannels),
          m_fpn3(256, nFpnChannels),
          m_fpn2(128, nFpnChannels),
          m_fpn1(64, nFpnChannels),
          // 20260402 ZJH 融合后平滑
          m_convFuse(nFpnChannels * 4, nFpnChannels, 3, 1, 1, false),
          m_bnFuse(nFpnChannels),
          // 20260402 ZJH DB Head
          m_dbHead(nFpnChannels)
    {
        // 20260402 ZJH 注册所有子模块
        // 20260402 ZJH 所有子模块为值成员，不使用 registerModule
        // 参数收集通过基类 Module::parameters() 的默认行为或手动覆盖处理
    }

    // 20260402 ZJH forward — 端到端前向传播
    // input: [N, C, H, W] 输入图像（建议 H=W=640 或 H=W=320）
    // 返回: [N, 3, H/4, W/4] — 概率图 P + 阈值图 T + 二值图 B
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH Backbone: 4 阶段特征提取
        // Stem: /4
        auto stem = ReLU().forward(m_bnStem.forward(m_convStem.forward(input)));  // 20260402 ZJH [N,64,H/2,W/2]
        stem = m_poolStem.forward(stem);  // 20260402 ZJH [N,64,H/4,W/4]

        // 20260402 ZJH Stage 1: /4（无下采样）
        auto s1 = ReLU().forward(m_bnS1a.forward(m_convS1a.forward(stem)));  // 20260402 ZJH [N,64,H/4,W/4]
        s1 = m_bnS1b.forward(m_convS1b.forward(s1));                     // 20260402 ZJH [N,64,H/4,W/4]
        s1 = ReLU().forward(tensorAdd(s1, stem));                             // 20260402 ZJH 残差连接

        // 20260402 ZJH Stage 2: /8
        auto s2 = ReLU().forward(m_bnS2a.forward(m_convS2a.forward(s1)));    // 20260402 ZJH [N,128,H/8,W/8]
        s2 = m_bnS2b.forward(m_convS2b.forward(s2));                     // 20260402 ZJH [N,128,H/8,W/8]
        auto s2short = m_bnS2down.forward(m_convS2down.forward(s1));      // 20260402 ZJH shortcut
        s2 = ReLU().forward(tensorAdd(s2, s2short));                          // 20260402 ZJH 残差

        // 20260402 ZJH Stage 3: /16
        auto s3 = ReLU().forward(m_bnS3a.forward(m_convS3a.forward(s2)));
        s3 = m_bnS3b.forward(m_convS3b.forward(s3));
        auto s3short = m_bnS3down.forward(m_convS3down.forward(s2));
        s3 = ReLU().forward(tensorAdd(s3, s3short));

        // 20260402 ZJH Stage 4: /32
        auto s4 = ReLU().forward(m_bnS4a.forward(m_convS4a.forward(s3)));
        s4 = m_bnS4b.forward(m_convS4b.forward(s4));
        auto s4short = m_bnS4down.forward(m_convS4down.forward(s3));
        s4 = ReLU().forward(tensorAdd(s4, s4short));

        // 20260402 ZJH FPN: 自顶向下特征金字塔
        auto f4 = m_fpn4.forward(s4);  // 20260402 ZJH [N, fpnC, H/32, W/32]
        auto f3 = m_fpn3.forward(s3);  // 20260402 ZJH [N, fpnC, H/16, W/16]
        auto f2 = m_fpn2.forward(s2);  // 20260402 ZJH [N, fpnC, H/8, W/8]
        auto f1 = m_fpn1.forward(s1);  // 20260402 ZJH [N, fpnC, H/4, W/4]

        // 20260402 ZJH 上采样到统一尺寸 (H/4, W/4) 并拼接
        int nTargetH = f1.shape(2);  // 20260402 ZJH H/4
        int nTargetW = f1.shape(3);  // 20260402 ZJH W/4
        auto f4up = tensorUpsampleNN(f4, nTargetH, nTargetW);  // 20260402 ZJH 8x 上采样
        auto f3up = tensorUpsampleNN(f3, nTargetH, nTargetW);  // 20260402 ZJH 4x 上采样
        auto f2up = tensorUpsampleNN(f2, nTargetH, nTargetW);  // 20260402 ZJH 2x 上采样

        // 20260402 ZJH 拼接 [N, fpnC*4, H/4, W/4]
        auto fused = tensorCatChannels4(f1, f2up, f3up, f4up);  // 20260402 ZJH 4 级拼接

        // 20260402 ZJH 融合卷积
        fused = ReLU().forward(m_bnFuse.forward(m_convFuse.forward(fused)));  // 20260402 ZJH [N, fpnC, H/4, W/4]

        // 20260402 ZJH DB Head: 输出概率图 + 阈值图 + 二值图
        return m_dbHead.forward(fused);  // 20260402 ZJH [N, 3, H/4, W/4]
    }

    // 20260402 ZJH detectTextBoxes — 推理后处理：从概率图提取文本框
    // probMap: [H, W] 单张概率图（forward 输出的 channel 0）
    // fThreshold: 二值化阈值（默认 0.3）
    // fMinScore: 最低检测分数过滤（默认 0.5）
    // 返回: 检测到的文本框列表
    static std::vector<TextBox> detectTextBoxes(const Tensor& probMap,
                                                 float fThreshold = 0.3f,
                                                 float fMinScore = 0.5f) {
        auto cMap = probMap.contiguous();
        auto vecShape = cMap.shapeVec();
        int nH = vecShape[0];  // 20260402 ZJH 概率图高
        int nW = vecShape[1];  // 20260402 ZJH 概率图宽
        const float* pMap = cMap.floatDataPtr();

        // 20260402 ZJH Step 1: 二值化
        std::vector<uint8_t> vecBinary(nH * nW, 0);  // 20260402 ZJH 二值图
        for (int i = 0; i < nH * nW; ++i) {
            vecBinary[i] = (pMap[i] > fThreshold) ? 1 : 0;  // 20260402 ZJH 阈值二值化
        }

        // 20260402 ZJH Step 2: 连通域分析 (simple flood fill)
        std::vector<int> vecLabels(nH * nW, 0);  // 20260402 ZJH 连通域标签
        int nNextLabel = 1;  // 20260402 ZJH 下一个标签 ID
        std::vector<TextBox> vecResults;  // 20260402 ZJH 结果

        for (int r = 0; r < nH; ++r) {
            for (int c = 0; c < nW; ++c) {
                int nIdx = r * nW + c;
                if (vecBinary[nIdx] == 0 || vecLabels[nIdx] != 0) continue;

                // 20260402 ZJH Flood fill 找连通域
                int nLabel = nNextLabel++;  // 20260402 ZJH 分配新标签
                std::vector<std::pair<int, int>> vecStack;  // 20260402 ZJH BFS 栈
                vecStack.push_back({r, c});
                vecLabels[nIdx] = nLabel;

                int nMinR = r, nMaxR = r, nMinC = c, nMaxC = c;  // 20260402 ZJH 包围框
                float fScoreSum = 0.0f;  // 20260402 ZJH 区域内概率和
                int nPixelCount = 0;      // 20260402 ZJH 区域像素数

                while (!vecStack.empty()) {
                    auto [cr, cc] = vecStack.back();
                    vecStack.pop_back();
                    int cIdx = cr * nW + cc;

                    nMinR = std::min(nMinR, cr);  nMaxR = std::max(nMaxR, cr);
                    nMinC = std::min(nMinC, cc);  nMaxC = std::max(nMaxC, cc);
                    fScoreSum += pMap[cIdx];
                    nPixelCount++;

                    // 20260402 ZJH 4-邻域扩展
                    int arrDr[] = {-1, 1, 0, 0};
                    int arrDc[] = {0, 0, -1, 1};
                    for (int d = 0; d < 4; ++d) {
                        int nr = cr + arrDr[d], nc = cc + arrDc[d];
                        if (nr >= 0 && nr < nH && nc >= 0 && nc < nW) {
                            int nNIdx = nr * nW + nc;
                            if (vecBinary[nNIdx] == 1 && vecLabels[nNIdx] == 0) {
                                vecLabels[nNIdx] = nLabel;
                                vecStack.push_back({nr, nc});
                            }
                        }
                    }
                }

                // 20260402 ZJH 构造文本框（过滤小区域和低分数）
                float fAvgScore = fScoreSum / static_cast<float>(nPixelCount);
                if (fAvgScore >= fMinScore && nPixelCount >= 10) {  // 20260402 ZJH 最少 10 像素
                    TextBox box;
                    box.fX1 = static_cast<float>(nMinC) / static_cast<float>(nW);
                    box.fY1 = static_cast<float>(nMinR) / static_cast<float>(nH);
                    box.fX2 = static_cast<float>(nMaxC + 1) / static_cast<float>(nW);
                    box.fY2 = static_cast<float>(nMaxR + 1) / static_cast<float>(nH);
                    box.fScore = fAvgScore;
                    box.fAngle = 0.0f;  // 20260402 ZJH 简化版不计算旋转
                    vecResults.push_back(box);
                }
            }
        }

        return vecResults;  // 20260402 ZJH 返回检测到的文本框
    }

private:
    int m_nFpnChannels;  // 20260402 ZJH FPN 通道数

    // 20260402 ZJH Backbone: ResNet18 风格
    Conv2d m_convStem;  BatchNorm2d m_bnStem;  MaxPool2d m_poolStem;
    Conv2d m_convS1a, m_convS1b;  BatchNorm2d m_bnS1a, m_bnS1b;
    Conv2d m_convS2a, m_convS2b, m_convS2down;  BatchNorm2d m_bnS2a, m_bnS2b, m_bnS2down;
    Conv2d m_convS3a, m_convS3b, m_convS3down;  BatchNorm2d m_bnS3a, m_bnS3b, m_bnS3down;
    Conv2d m_convS4a, m_convS4b, m_convS4down;  BatchNorm2d m_bnS4a, m_bnS4b, m_bnS4down;

    // 20260402 ZJH FPN 侧向连接
    FPNBlock m_fpn4, m_fpn3, m_fpn2, m_fpn1;

    // 20260402 ZJH 融合层
    Conv2d m_convFuse;  BatchNorm2d m_bnFuse;

    // 20260402 ZJH DB Head
    DBHead m_dbHead;
};

// =============================================================================
// 20260402 ZJH DBLoss — DBNet 训练损失函数
// 三项组合: L_prob(BCE) + L_thresh(L1) + L_binary(BCE)
// L_prob: 概率图的 BCE 损失（主要监督信号）
// L_thresh: 阈值图的 L1 回归损失（约束阈值在合理范围）
// L_binary: 可微分二值图的 BCE 损失（辅助监督）
// =============================================================================
class DBLoss {
public:
    float m_fProbWeight = 1.0f;    // 20260402 ZJH 概率图损失权重
    float m_fThreshWeight = 5.0f;  // 20260402 ZJH 阈值图损失权重（需较大权重以稳定训练）
    float m_fBinaryWeight = 1.0f;  // 20260402 ZJH 二值图损失权重

    // 20260402 ZJH forward — 计算 DB 总损失
    // output: [N, 3, H, W] — DBNet 输出 (P, T, B)
    // target: [N, 1, H, W] — GT 二值 mask（文本区域=1, 背景=0）
    // threshTarget: [N, 1, H, W] — GT 阈值图（边界附近高，内部低，可选）
    // 返回: 标量损失
    Tensor forward(const Tensor& output, const Tensor& target, const Tensor& threshTarget) {
        int nBatch = output.shape(0);
        int nH = output.shape(2);
        int nW = output.shape(3);
        int nSpatial = nH * nW;

        auto cOutput = output.contiguous();
        auto cTarget = target.contiguous();
        auto cThreshTarget = threshTarget.contiguous();
        const float* pOut = cOutput.floatDataPtr();
        const float* pTarget = cTarget.floatDataPtr();
        const float* pThreshT = cThreshTarget.floatDataPtr();

        float fProbLoss = 0.0f;    // 20260402 ZJH 概率图 BCE
        float fThreshLoss = 0.0f;  // 20260402 ZJH 阈值图 L1
        float fBinaryLoss = 0.0f;  // 20260402 ZJH 二值图 BCE
        int nTotal = nBatch * nSpatial;  // 20260402 ZJH 总像素数

        for (int b = 0; b < nBatch; ++b) {
            for (int i = 0; i < nSpatial; ++i) {
                float fP = pOut[(b * 3 + 0) * nSpatial + i];  // 20260402 ZJH 概率
                float fT = pOut[(b * 3 + 1) * nSpatial + i];  // 20260402 ZJH 阈值
                float fB = pOut[(b * 3 + 2) * nSpatial + i];  // 20260402 ZJH 二值
                float fGT = pTarget[b * nSpatial + i];         // 20260402 ZJH GT

                // 20260402 ZJH 概率图 BCE: -[y*log(p) + (1-y)*log(1-p)]
                float fPClamp = std::max(1e-7f, std::min(fP, 1.0f - 1e-7f));
                fProbLoss += -(fGT * std::log(fPClamp) + (1.0f - fGT) * std::log(1.0f - fPClamp));

                // 20260402 ZJH 阈值图 L1
                float fTGT = pThreshT[b * nSpatial + i];
                fThreshLoss += std::abs(fT - fTGT);

                // 20260402 ZJH 二值图 BCE
                float fBClamp = std::max(1e-7f, std::min(fB, 1.0f - 1e-7f));
                fBinaryLoss += -(fGT * std::log(fBClamp) + (1.0f - fGT) * std::log(1.0f - fBClamp));
            }
        }

        // 20260402 ZJH 平均并加权
        float fInvTotal = 1.0f / static_cast<float>(std::max(nTotal, 1));
        float fTotalLoss = m_fProbWeight * fProbLoss * fInvTotal +
                           m_fThreshWeight * fThreshLoss * fInvTotal +
                           m_fBinaryWeight * fBinaryLoss * fInvTotal;

        return Tensor::full({1}, fTotalLoss);  // 20260402 ZJH 标量损失
    }
};

}  // namespace om
