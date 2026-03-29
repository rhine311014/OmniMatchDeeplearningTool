// 20260321 ZJH Phase 5B 测试 — Tanh/LSTM/BiLSTM/CTCDecoder/CTCLoss/CRNN/ROIAlign/InstanceSeg
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.engine.linear;
import om.engine.crnn;
import om.engine.instance_seg;
import om.hal.cpu_backend;

using namespace om;

// ===== Tanh 测试 =====

TEST(Phase5B, TanhForward) {
    // 20260321 ZJH 测试 tanh 激活函数基本前向
    auto input = Tensor::fromData(std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}.data(), {1, 5});
    auto output = tensorTanh(input);
    const float* pOut = output.floatDataPtr();

    // 20260321 ZJH tanh(0) = 0
    EXPECT_NEAR(pOut[2], 0.0f, 1e-5f);
    // 20260321 ZJH tanh(1) ≈ 0.7616
    EXPECT_NEAR(pOut[3], std::tanh(1.0f), 1e-4f);
    // 20260321 ZJH tanh(-1) ≈ -0.7616
    EXPECT_NEAR(pOut[1], std::tanh(-1.0f), 1e-4f);
    // 20260321 ZJH tanh 输出范围 [-1, 1]
    for (int i = 0; i < 5; ++i) {
        EXPECT_GE(pOut[i], -1.0f);
        EXPECT_LE(pOut[i], 1.0f);
    }
}

TEST(Phase5B, TanhModule) {
    // 20260321 ZJH 测试 Tanh 模块封装
    Tanh tanhMod;
    auto input = Tensor::randn({2, 8});
    auto output = tanhMod.forward(input);
    EXPECT_EQ(output.shape(0), 2);
    EXPECT_EQ(output.shape(1), 8);
}

TEST(Phase5B, TanhGradient) {
    // 20260321 ZJH 测试 tanh 自动微分
    auto input = Tensor::fromData(std::vector<float>{0.5f, -0.3f, 1.0f}.data(), {1, 3});
    tensorSetRequiresGrad(input, true);

    auto output = tensorTanh(input);
    auto loss = tensorSum(output);
    tensorBackward(loss);

    auto grad = tensorGetGrad(input);
    const float* pGrad = grad.floatDataPtr();
    // 20260321 ZJH tanh'(x) = 1 - tanh(x)^2
    const float* pIn = input.floatDataPtr();
    for (int i = 0; i < 3; ++i) {
        float fTanh = std::tanh(pIn[i]);
        float fExpectedGrad = 1.0f - fTanh * fTanh;
        EXPECT_NEAR(pGrad[i], fExpectedGrad, 1e-4f);
    }
}

// ===== tensorSliceLastDim 测试 =====

TEST(Phase5B, SliceLastDim) {
    // 20260321 ZJH 测试最后一维切片
    auto input = Tensor::fromData(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}.data(), {2, 4});
    auto slice = tensorSliceLastDim(input, 1, 3);  // 取 [1, 3)
    EXPECT_EQ(slice.shape(0), 2);
    EXPECT_EQ(slice.shape(1), 2);
    const float* p = slice.floatDataPtr();
    EXPECT_FLOAT_EQ(p[0], 2.0f);  // 第 0 行的 [1]
    EXPECT_FLOAT_EQ(p[1], 3.0f);  // 第 0 行的 [2]
    EXPECT_FLOAT_EQ(p[2], 6.0f);  // 第 1 行的 [1]
    EXPECT_FLOAT_EQ(p[3], 7.0f);  // 第 1 行的 [2]
}

// ===== tensorConcatLastDim 测试 =====

TEST(Phase5B, ConcatLastDim) {
    // 20260321 ZJH 测试最后一维拼接
    auto a = Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = Tensor::fromData(std::vector<float>{5, 6, 7, 8, 9, 10}.data(), {2, 3});
    auto result = tensorConcatLastDim(a, b);
    EXPECT_EQ(result.shape(0), 2);
    EXPECT_EQ(result.shape(1), 5);
    const float* p = result.floatDataPtr();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 5.0f);
    EXPECT_FLOAT_EQ(p[3], 6.0f);
    EXPECT_FLOAT_EQ(p[4], 7.0f);
}

// ===== LSTMCell 测试 =====

TEST(Phase5B, LSTMCellStep) {
    // 20260321 ZJH 测试 LSTM 单步前向
    int nInputSize = 4;
    int nHiddenSize = 8;
    int nBatch = 2;

    LSTMCell cell(nInputSize, nHiddenSize);

    auto input = Tensor::randn({nBatch, nInputSize});
    auto h = Tensor::zeros({nBatch, nHiddenSize});
    auto c = Tensor::zeros({nBatch, nHiddenSize});

    auto state = cell.step(input, h, c);

    // 20260321 ZJH 检查输出形状
    EXPECT_EQ(state.h.shape(0), nBatch);
    EXPECT_EQ(state.h.shape(1), nHiddenSize);
    EXPECT_EQ(state.c.shape(0), nBatch);
    EXPECT_EQ(state.c.shape(1), nHiddenSize);

    // 20260321 ZJH h 应在 [-1, 1] 范围（tanh 输出），放宽到 [-1.01, 1.01] 防止浮点精度
    const float* pH = state.h.floatDataPtr();
    for (int i = 0; i < nBatch * nHiddenSize; ++i) {
        EXPECT_GE(pH[i], -1.01f);
        EXPECT_LE(pH[i], 1.01f);
    }
}

TEST(Phase5B, LSTMCellSequence) {
    // 20260321 ZJH 测试 LSTM 多步序列处理
    int nInputSize = 4;
    int nHiddenSize = 8;
    int nBatch = 1;
    int nSeqLen = 5;

    LSTMCell cell(nInputSize, nHiddenSize);
    auto h = Tensor::zeros({nBatch, nHiddenSize});
    auto c = Tensor::zeros({nBatch, nHiddenSize});

    // 20260321 ZJH 模拟 5 步序列处理
    for (int t = 0; t < nSeqLen; ++t) {
        auto input = Tensor::randn({nBatch, nInputSize});
        auto state = cell.step(input, h, c);
        h = state.h;
        c = state.c;
    }

    // 20260321 ZJH 经过多步后，隐藏状态应非零
    const float* pH = h.floatDataPtr();
    float fSumAbs = 0.0f;
    for (int i = 0; i < nHiddenSize; ++i) {
        fSumAbs += std::abs(pH[i]);
    }
    EXPECT_GT(fSumAbs, 0.01f);
}

// ===== BiLSTM 测试 =====

TEST(Phase5B, BiLSTMForward) {
    // 20260321 ZJH 测试双向 LSTM 前向传播
    int nInputSize = 4;
    int nHiddenSize = 8;
    int nBatch = 2;
    int nSeqLen = 3;

    BiLSTM bilstm(nInputSize, nHiddenSize);

    auto input = Tensor::randn({nBatch, nSeqLen, nInputSize});
    auto output = bilstm.forward(input);

    // 20260321 ZJH 输出应为 [batch, seq_len, 2*hidden]
    EXPECT_EQ(output.shape(0), nBatch);
    EXPECT_EQ(output.shape(1), nSeqLen);
    EXPECT_EQ(output.shape(2), 2 * nHiddenSize);
}

// ===== CTC 解码器测试 =====

TEST(Phase5B, CTCGreedyDecode) {
    // 20260321 ZJH 测试 CTC 贪心解码
    CTCDecoder decoder(0);  // blank index = 0

    // 20260321 ZJH 模拟 logits: 5 时间步，4 类（0=blank, 1-3=字符）
    // 设计的序列: blank, 1, 1, 2, blank -> 应解码为 [1, 2]
    std::vector<float> vecLogits = {
        10.0f, -1.0f, -1.0f, -1.0f,  // t0: blank
        -1.0f, 10.0f, -1.0f, -1.0f,  // t1: class 1
        -1.0f, 10.0f, -1.0f, -1.0f,  // t2: class 1 (重复)
        -1.0f, -1.0f, 10.0f, -1.0f,  // t3: class 2
        10.0f, -1.0f, -1.0f, -1.0f,  // t4: blank
    };
    auto logits = Tensor::fromData(vecLogits.data(), {5, 4});
    auto decoded = decoder.greedyDecode(logits);

    // 20260321 ZJH 连续重复去重后应为 [1, 2]
    ASSERT_EQ(decoded.size(), 2u);
    EXPECT_EQ(decoded[0], 1);
    EXPECT_EQ(decoded[1], 2);
}

TEST(Phase5B, CTCGreedyDecodeAllBlank) {
    // 20260321 ZJH 测试全空白输入
    CTCDecoder decoder(0);
    std::vector<float> vecLogits = {
        10.0f, -1.0f, -1.0f,
        10.0f, -1.0f, -1.0f,
        10.0f, -1.0f, -1.0f,
    };
    auto logits = Tensor::fromData(vecLogits.data(), {3, 3});
    auto decoded = decoder.greedyDecode(logits);
    EXPECT_TRUE(decoded.empty());
}

// ===== CTC Loss 测试 =====

TEST(Phase5B, CTCLossBasic) {
    // 20260321 ZJH 测试 CTC 损失基本功能
    CTCLoss ctcLoss(0);

    // 20260321 ZJH batch=1, seq_len=5, num_classes=4
    auto logits = Tensor::randn({1, 5, 4});
    std::vector<std::vector<int>> vecTargets = {{1, 2}};  // 目标序列

    auto loss = ctcLoss.forward(logits, vecTargets);
    EXPECT_EQ(loss.numel(), 1);
    // 20260321 ZJH 损失应为正值
    EXPECT_GT(loss.floatDataPtr()[0], 0.0f);
}

TEST(Phase5B, CTCLossCorrectPrediction) {
    // 20260321 ZJH 测试当 logits 正确时损失应较低
    CTCLoss ctcLoss(0);

    // 20260321 ZJH 设计一个 logits 使得正确目标的概率很高
    std::vector<float> vecLogits = {
        10.0f, -5.0f, -5.0f,  // t0: blank (高概率)
        -5.0f, 10.0f, -5.0f,  // t1: class 1 (高概率)
        10.0f, -5.0f, -5.0f,  // t2: blank
        -5.0f, -5.0f, 10.0f,  // t3: class 2 (高概率)
        10.0f, -5.0f, -5.0f,  // t4: blank
    };
    auto correctLogits = Tensor::fromData(vecLogits.data(), {1, 5, 3});

    // 20260321 ZJH 随机 logits（不太确定）
    auto randomLogits = Tensor::randn({1, 5, 3});

    std::vector<std::vector<int>> vecTargets = {{1, 2}};

    auto correctLoss = ctcLoss.forward(correctLogits, vecTargets);
    auto randomLoss = ctcLoss.forward(randomLogits, vecTargets);

    // 20260321 ZJH 正确预测的损失应低于随机预测
    EXPECT_LT(correctLoss.floatDataPtr()[0], randomLoss.floatDataPtr()[0]);
}

// ===== CRNN 网络测试 =====

TEST(Phase5B, CRNNForwardShape) {
    // 20260321 ZJH 测试 CRNN 前向传播输出形状
    int nNumClasses = 37;  // 26 字母 + 10 数字 + 1 空白
    int nHidden = 32;      // 小隐藏维度（测试用）
    int nBatch = 1;
    int nWidth = 64;       // 输入宽度（必须能被 4 整除）

    CRNN crnn(nNumClasses, nHidden, 32);

    // 20260321 ZJH 输入: [1, 1, 32, 64] 灰度图
    auto input = Tensor::randn({nBatch, 1, 32, nWidth});
    auto output = crnn.forward(input);

    // 20260321 ZJH 输出: [1, 16, 37] — seq_len = 64/4 = 16
    EXPECT_EQ(output.shape(0), nBatch);
    EXPECT_EQ(output.shape(1), nWidth / 4);  // 经过两次 stride=2 池化
    EXPECT_EQ(output.shape(2), nNumClasses);
}

TEST(Phase5B, CRNNParameters) {
    // 20260321 ZJH 测试 CRNN 参数数量
    CRNN crnn(37, 32, 32);
    auto params = crnn.parameters();
    // 20260321 ZJH 应有大量参数（CNN + LSTM + FC）
    EXPECT_GT(params.size(), 20u);

    // 20260321 ZJH 所有参数应为非空
    for (auto* p : params) {
        EXPECT_GT(p->numel(), 0);
    }
}

// ===== ROI Align 测试 =====

TEST(Phase5B, ROIAlignBasic) {
    // 20260321 ZJH 测试 ROI Align 基本功能
    ROIAlign roiAlign(7, 7, 1.0f, 2);  // 7x7 输出，无缩放

    // 20260321 ZJH 创建简单特征图 [1, 1, 16, 16]
    auto features = Tensor::randn({1, 1, 16, 16});

    // 20260321 ZJH 2 个 ROI
    std::vector<float> vecRois = {
        2.0f, 2.0f, 10.0f, 10.0f,  // ROI 1
        4.0f, 4.0f, 12.0f, 12.0f,  // ROI 2
    };
    auto rois = Tensor::fromData(vecRois.data(), {2, 4});

    auto output = roiAlign.forward(features, rois);

    // 20260321 ZJH 输出形状: [2, 1, 7, 7]
    EXPECT_EQ(output.shape(0), 2);
    EXPECT_EQ(output.shape(1), 1);
    EXPECT_EQ(output.shape(2), 7);
    EXPECT_EQ(output.shape(3), 7);
}

TEST(Phase5B, ROIAlignSpatialScale) {
    // 20260321 ZJH 测试带空间缩放的 ROI Align
    ROIAlign roiAlign(3, 3, 0.5f, 2);  // 0.5x 缩放

    auto features = Tensor::randn({1, 2, 8, 8});

    std::vector<float> vecRois = {
        0.0f, 0.0f, 16.0f, 16.0f,  // 原图坐标，缩放后 [0,0,8,8]
    };
    auto rois = Tensor::fromData(vecRois.data(), {1, 4});

    auto output = roiAlign.forward(features, rois);
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 2);
    EXPECT_EQ(output.shape(2), 3);
    EXPECT_EQ(output.shape(3), 3);
}

// ===== ProtoNet 测试 =====

TEST(Phase5B, ProtoNetForward) {
    // 20260321 ZJH 测试 ProtoNet 前向
    int nNumProtos = 16;
    ProtoNet protoNet(64, nNumProtos);

    auto input = Tensor::randn({1, 64, 16, 16});
    auto protos = protoNet.forward(input);

    // 20260321 ZJH 输出: [1, 16, 16, 16]
    EXPECT_EQ(protos.shape(0), 1);
    EXPECT_EQ(protos.shape(1), nNumProtos);
    EXPECT_EQ(protos.shape(2), 16);
    EXPECT_EQ(protos.shape(3), 16);
}

// ===== InstanceHead 测试 =====

TEST(Phase5B, InstanceHeadForward) {
    // 20260321 ZJH 测试 InstanceHead 前向
    int nClasses = 3;
    int nProtos = 16;
    int nAnchors = 3;
    InstanceHead head(64, nClasses, nProtos, nAnchors);

    auto input = Tensor::randn({1, 64, 8, 8});
    auto output = head.forward(input);

    // 20260321 ZJH 输出通道 = A*C + A*4 + A*K = 3*3 + 3*4 + 3*16 = 9+12+48 = 69
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), nAnchors * nClasses + nAnchors * 4 + nAnchors * nProtos);
    EXPECT_EQ(output.shape(2), 8);
    EXPECT_EQ(output.shape(3), 8);
}

// ===== SimpleInstanceSeg 测试 =====

TEST(Phase5B, SimpleInstanceSegForward) {
    // 20260321 ZJH 测试简化实例分割网络前向
    SimpleInstanceSeg model(1, 3, 16);  // 灰度输入，3 类，16 prototypes

    auto input = Tensor::randn({1, 1, 32, 32});
    auto output = model.forward(input);

    // 20260321 ZJH 输出应为有效张量
    EXPECT_GT(output.numel(), 0);
}

TEST(Phase5B, SimpleInstanceSegParameters) {
    // 20260321 ZJH 测试实例分割网络参数
    SimpleInstanceSeg model(1, 3, 16);
    auto params = model.parameters();
    EXPECT_GT(params.size(), 10u);
}

// ===== assembleMasks 测试 =====

TEST(Phase5B, AssembleMasks) {
    // 20260321 ZJH 测试 mask 组装
    int nK = 4;   // 4 个 prototype
    int nH = 8;
    int nW = 8;
    int nN = 2;   // 2 个实例

    auto prototypes = Tensor::randn({nK, nH, nW});
    auto coeffs = Tensor::randn({nN, nK});

    auto masks = assembleMasks(prototypes, coeffs);

    // 20260321 ZJH 输出: [2, 8, 8]
    EXPECT_EQ(masks.shape(0), nN);
    EXPECT_EQ(masks.shape(1), nH);
    EXPECT_EQ(masks.shape(2), nW);

    // 20260321 ZJH mask 值应在 [0, 1]（sigmoid 后）
    const float* pMask = masks.floatDataPtr();
    for (int i = 0; i < nN * nH * nW; ++i) {
        EXPECT_GE(pMask[i], 0.0f);
        EXPECT_LE(pMask[i], 1.0f);
    }
}

// ===== NMS 测试 =====

TEST(Phase5B, InstanceNMS) {
    // 20260321 ZJH 测试实例级 NMS
    std::vector<InstanceResult> vecResults;

    // 20260321 ZJH 两个高度重叠的检测（同类）
    vecResults.push_back({10, 10, 50, 50, 0, 0.9f, {}});
    vecResults.push_back({12, 12, 52, 52, 0, 0.7f, {}});
    // 20260321 ZJH 一个不重叠的检测
    vecResults.push_back({100, 100, 150, 150, 0, 0.8f, {}});
    // 20260321 ZJH 不同类别的检测
    vecResults.push_back({10, 10, 50, 50, 1, 0.85f, {}});

    auto kept = instanceNMS(vecResults, 0.5f);

    // 20260321 ZJH 应保留 3 个（高分的重叠框 + 不重叠框 + 不同类别框）
    EXPECT_EQ(kept.size(), 3u);
}

// ===== Mask IoU 测试 =====

TEST(Phase5B, MaskIoU) {
    // 20260321 ZJH 测试 mask IoU 计算
    std::vector<float> vecMaskA = {1.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vecMaskB = {1.0f, 0.0f, 1.0f, 0.0f};

    float fIoU = computeMaskIoU(vecMaskA, vecMaskB, 0.5f);
    // 20260321 ZJH 交集 = 1, 并集 = 3, IoU = 1/3
    EXPECT_NEAR(fIoU, 1.0f / 3.0f, 1e-5f);
}

TEST(Phase5B, MaskIoUPerfect) {
    // 20260321 ZJH 测试完全重合的 IoU
    std::vector<float> vecMask = {1.0f, 1.0f, 1.0f, 0.0f};
    float fIoU = computeMaskIoU(vecMask, vecMask, 0.5f);
    EXPECT_NEAR(fIoU, 1.0f, 1e-5f);
}
