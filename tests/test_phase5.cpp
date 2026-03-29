// 20260320 ZJH Phase 5 测试 — GELU/SiLU/LayerNorm/AdaptiveAvgPool2d/ViT/DataPipeline/ONNX
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.engine.linear;
import om.engine.vit;
import om.engine.data_pipeline;
import om.engine.gan;
import om.engine.optimizer;
import om.engine.scheduler;
import om.engine.metrics;
import om.engine.checkpoint;
import om.engine.parallel;
import om.engine.distillation;
import om.engine.onnx;
import om.hal.cpu_backend;

using namespace om;

// ===== GELU 测试 =====

TEST(Phase5, GELUForward) {
    // 20260320 ZJH 测试 GELU 激活函数基本前向
    auto input = Tensor::fromData(std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}.data(), {1, 5});
    auto output = tensorGELU(input);
    const float* pOut = output.floatDataPtr();

    // 20260320 ZJH GELU(0) = 0
    EXPECT_NEAR(pOut[2], 0.0f, 1e-4f);
    // 20260320 ZJH GELU(x) > 0 for x > 0
    EXPECT_GT(pOut[3], 0.0f);
    EXPECT_GT(pOut[4], 0.0f);
    // 20260320 ZJH GELU(x) < 0 for x < 0 (略微负值)
    EXPECT_LT(pOut[0], 0.0f);
}

TEST(Phase5, GELUModule) {
    // 20260320 ZJH 测试 GELU 模块封装
    GELU gelu;
    auto input = Tensor::randn({2, 8});
    auto output = gelu.forward(input);
    EXPECT_EQ(output.shape(0), 2);
    EXPECT_EQ(output.shape(1), 8);
}

// ===== SiLU 测试 =====

TEST(Phase5, SiLUForward) {
    // 20260320 ZJH 测试 SiLU 激活函数
    auto input = Tensor::fromData(std::vector<float>{-2.0f, 0.0f, 2.0f}.data(), {1, 3});
    auto output = tensorSiLU(input);
    const float* pOut = output.floatDataPtr();

    // 20260320 ZJH SiLU(0) = 0 * sigmoid(0) = 0
    EXPECT_NEAR(pOut[1], 0.0f, 1e-4f);
    // 20260320 ZJH SiLU(2) ≈ 2 * 0.8808 ≈ 1.7616
    EXPECT_NEAR(pOut[2], 2.0f * (1.0f / (1.0f + std::exp(-2.0f))), 1e-3f);
}

// ===== LayerNorm 测试 =====

TEST(Phase5, LayerNormForward) {
    // 20260320 ZJH 测试 LayerNorm 基本前向
    int nDim = 4;
    LayerNorm ln(nDim);
    auto input = Tensor::fromData(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}.data(), {2, 4});
    auto output = ln.forward(input);

    EXPECT_EQ(output.shape(0), 2);
    EXPECT_EQ(output.shape(1), 4);

    // 20260320 ZJH 每行归一化后均值应近似 0
    const float* pOut = output.floatDataPtr();
    float fSum0 = pOut[0] + pOut[1] + pOut[2] + pOut[3];
    EXPECT_NEAR(fSum0, 0.0f, 0.1f);
}

TEST(Phase5, LayerNormModule) {
    // 20260320 ZJH 测试 LayerNorm 有可训练参数
    LayerNorm ln(64);
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 2u);  // gamma 和 beta
}

// ===== AdaptiveAvgPool2d 测试 =====

TEST(Phase5, AdaptiveAvgPool2dForward) {
    // 20260320 ZJH 测试自适应平均池化
    auto input = Tensor::ones({1, 2, 8, 8});  // 全 1 张量
    auto output = tensorAdaptiveAvgPool2d(input, 1, 1);  // 全局平均池化

    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 2);
    EXPECT_EQ(output.shape(2), 1);
    EXPECT_EQ(output.shape(3), 1);

    // 20260320 ZJH 全 1 张量的全局平均 = 1.0
    EXPECT_NEAR(output.floatDataPtr()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(output.floatDataPtr()[1], 1.0f, 1e-5f);
}

TEST(Phase5, AdaptiveAvgPool2dModule) {
    // 20260320 ZJH 测试模块封装
    AdaptiveAvgPool2d pool(2, 2);
    auto input = Tensor::randn({1, 3, 16, 16});
    auto output = pool.forward(input);
    EXPECT_EQ(output.shape(2), 2);
    EXPECT_EQ(output.shape(3), 2);
}

// ===== ViT 测试 =====

TEST(Phase5, PatchEmbeddingForward) {
    // 20260320 ZJH 测试 PatchEmbedding 前向
    int nImgSize = 8;
    int nPatchSize = 4;
    int nInChannels = 1;
    int nEmbedDim = 16;

    PatchEmbedding pe(nImgSize, nPatchSize, nInChannels, nEmbedDim);
    auto input = Tensor::randn({1, nInChannels, nImgSize, nImgSize});
    auto output = pe.forward(input);

    int nNumPatches = (nImgSize / nPatchSize) * (nImgSize / nPatchSize);  // 4
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), nNumPatches + 1);  // patches + CLS token
    EXPECT_EQ(output.shape(2), nEmbedDim);
}

TEST(Phase5, MultiHeadAttentionForward) {
    // 20260320 ZJH 测试多头注意力前向
    int nEmbedDim = 16;
    int nHeads = 2;
    int nSeqLen = 5;
    int nBatch = 1;

    MultiHeadAttention mha(nEmbedDim, nHeads);
    auto input = Tensor::randn({nBatch, nSeqLen, nEmbedDim});
    auto output = mha.forward(input);

    EXPECT_EQ(output.shape(0), nBatch);
    EXPECT_EQ(output.shape(1), nSeqLen);
    EXPECT_EQ(output.shape(2), nEmbedDim);
}

TEST(Phase5, TransformerBlockForward) {
    // 20260320 ZJH 测试 Transformer 块前向
    int nEmbedDim = 16;
    int nHeads = 2;
    int nMlpDim = 32;
    int nSeqLen = 5;
    int nBatch = 1;

    TransformerBlock block(nEmbedDim, nHeads, nMlpDim);
    auto input = Tensor::randn({nBatch, nSeqLen, nEmbedDim});
    auto output = block.forward(input);

    EXPECT_EQ(output.shape(0), nBatch);
    EXPECT_EQ(output.shape(1), nSeqLen);
    EXPECT_EQ(output.shape(2), nEmbedDim);
}

TEST(Phase5, ViTForward) {
    // 20260320 ZJH 测试完整 ViT 前向传播
    // ViT-Tiny: 小尺寸配置，适合快速测试
    int nImgSize = 8;
    int nPatchSize = 4;
    int nInChannels = 1;
    int nNumClasses = 10;
    int nEmbedDim = 16;
    int nDepth = 2;
    int nHeads = 2;
    int nMlpDim = 32;

    ViT vit(nImgSize, nPatchSize, nInChannels, nNumClasses, nEmbedDim, nDepth, nHeads, nMlpDim);
    auto input = Tensor::randn({1, nInChannels, nImgSize, nImgSize});
    auto output = vit.forward(input);

    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), nNumClasses);
}

TEST(Phase5, ViTParameters) {
    // 20260320 ZJH 测试 ViT 参数数量合理
    ViT vit(8, 4, 1, 10, 16, 2, 2, 32);
    auto params = vit.parameters();
    // 20260320 ZJH 应有大量参数（patch embed + blocks + norm + head）
    EXPECT_GT(params.size(), 10u);

    // 20260320 ZJH 计算总参数量
    int nTotalParams = 0;
    for (auto* p : params) nTotalParams += p->numel();
    EXPECT_GT(nTotalParams, 1000);  // 至少上千个参数
}

// ===== 批量矩阵乘法测试 =====

TEST(Phase5, BatchedMatmulForward) {
    // 20260320 ZJH 测试批量矩阵乘法
    int nBatch = 2;
    int nM = 3;
    int nK = 4;
    int nN = 5;

    auto A = Tensor::ones({nBatch, nM, nK});
    auto B = Tensor::ones({nBatch, nK, nN});
    auto C = tensorBatchedMatmul(A, B);

    EXPECT_EQ(C.shape(0), nBatch);
    EXPECT_EQ(C.shape(1), nM);
    EXPECT_EQ(C.shape(2), nN);

    // 20260320 ZJH 全 1 矩阵相乘结果每个元素 = K = 4
    EXPECT_NEAR(C.floatDataPtr()[0], 4.0f, 1e-5f);
}

// ===== 数据管线测试 =====

TEST(Phase5, RawImageLoadBMP) {
    // 20260320 ZJH 测试 BMP 加载（不存在的文件应返回无效图像）
    auto img = loadBMP("nonexistent.bmp");
    EXPECT_FALSE(img.isValid());
}

TEST(Phase5, ResizeImage) {
    // 20260320 ZJH 测试图像缩放
    std::vector<float> vecSrc(1 * 4 * 4, 1.0f);  // [1, 4, 4] 全 1
    auto vecDst = resizeImage(vecSrc, 1, 4, 4, 2, 2);
    EXPECT_EQ(vecDst.size(), 1u * 2 * 2);
    EXPECT_NEAR(vecDst[0], 1.0f, 1e-5f);
}

TEST(Phase5, AugmentImageFlip) {
    // 20260320 ZJH 测试数据增强：确保不崩溃
    std::vector<float> vecData(1 * 8 * 8, 0.5f);
    AugmentConfig config;
    config.bRandomHFlip = true;
    config.bNormalize = false;
    augmentImage(vecData, 1, 8, 8, config);
    EXPECT_EQ(vecData.size(), 64u);
}

TEST(Phase5, DatasetSplitRatios) {
    // 20260320 ZJH 测试数据集划分
    auto split = splitDataset(100, 0.8f, 0.1f);
    EXPECT_EQ(split.vecTrainIndices.size(), 80u);
    EXPECT_EQ(split.vecValIndices.size(), 10u);
    EXPECT_EQ(split.vecTestIndices.size(), 10u);
}

// ===== ONNX 导出测试 =====

TEST(Phase5, OnnxExportText) {
    // 20260320 ZJH 测试 ONNX 文本导出
    // 创建简单的 MLP 模型
    Sequential model;
    model.add(std::make_unique<Linear>(4, 8, true));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(8, 2, true));

    std::string strPath = "test_export.onnx.txt";
    bool bSuccess = OnnxExporter::exportToText(strPath, model, {1, 4}, 2);
    EXPECT_TRUE(bSuccess);

    // 20260320 ZJH 验证文件已创建
    EXPECT_TRUE(std::filesystem::exists(strPath));

    // 20260320 ZJH 清理
    std::filesystem::remove(strPath);
}

TEST(Phase5, OnnxExportBinary) {
    // 20260320 ZJH 测试 ONNX 二进制导出
    Sequential model;
    model.add(std::make_unique<Linear>(4, 8, true));
    model.add(std::make_unique<Linear>(8, 2, true));

    std::string strPath = "test_export.dfonnx";
    bool bSuccess = OnnxExporter::exportBinary(strPath, model, {1, 4}, 2);
    EXPECT_TRUE(bSuccess);

    // 20260320 ZJH 验证文件已创建且有内容
    EXPECT_TRUE(std::filesystem::exists(strPath));
    auto nFileSize = std::filesystem::file_size(strPath);
    EXPECT_GT(nFileSize, 100u);  // 至少有一些数据

    std::filesystem::remove(strPath);
}

// ===== Softmax 测试 =====

TEST(Phase5, SoftmaxLastDim) {
    // 20260320 ZJH 测试沿最后一维 Softmax
    auto input = Tensor::fromData(std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}.data(), {2, 3});
    auto output = tensorSoftmaxLastDim(input);

    const float* pOut = output.floatDataPtr();
    // 20260320 ZJH 每行之和应为 1
    float fSum0 = pOut[0] + pOut[1] + pOut[2];
    float fSum1 = pOut[3] + pOut[4] + pOut[5];
    EXPECT_NEAR(fSum0, 1.0f, 1e-5f);
    EXPECT_NEAR(fSum1, 1.0f, 1e-5f);
}

// ===== GAN 测试 =====

TEST(Phase5, GeneratorForward) {
    // 20260320 ZJH 测试 GAN 生成器前向
    Generator gen(32, 1, 28);
    auto noise = Tensor::randn({1, 32});
    auto output = gen.forward(noise);
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 1);
    EXPECT_EQ(output.shape(2), 28);
    EXPECT_EQ(output.shape(3), 28);
}

TEST(Phase5, DiscriminatorForward) {
    // 20260320 ZJH 测试 GAN 判别器前向
    Discriminator disc(1, 28);
    auto image = Tensor::randn({1, 1, 28, 28});
    auto output = disc.forward(image);
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 1);
    // 20260320 ZJH 输出应在 [0, 1] 之间（经过 sigmoid）
    EXPECT_GE(output.floatDataPtr()[0], 0.0f);
    EXPECT_LE(output.floatDataPtr()[0], 1.0f);
}

TEST(Phase5, AnomalyGANParameters) {
    // 20260320 ZJH 测试 AnomalyGAN 参数收集
    AnomalyGAN gan(32, 1, 28);
    auto genParams = gan.generatorParameters();
    auto discParams = gan.discriminatorParameters();
    auto allParams = gan.parameters();
    EXPECT_GT(genParams.size(), 0u);
    EXPECT_GT(discParams.size(), 0u);
    EXPECT_EQ(allParams.size(), genParams.size() + discParams.size());
}

// ===== AdamW 测试 =====

TEST(Phase5, AdamWStep) {
    // 20260320 ZJH 测试 AdamW 优化器
    auto w = Tensor::ones({2, 3});
    tensorSetRequiresGrad(w, true);
    auto loss = tensorSum(w);
    tensorBackward(loss);

    std::vector<Tensor*> params = {&w};
    AdamW opt(params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
    opt.step();

    // 20260320 ZJH 权重应已更新（不再全为 1）
    const float* pW = w.floatDataPtr();
    bool bChanged = false;
    for (int i = 0; i < 6; ++i) {
        if (std::abs(pW[i] - 1.0f) > 1e-5f) { bChanged = true; break; }
    }
    EXPECT_TRUE(bChanged);
}

// ===== 学习率调度器测试 =====

TEST(Phase5, CosineAnnealingLR) {
    CosineAnnealingLR sched(0.1f, 100, 1e-6f);
    float fStart = sched.step(0);
    float fMid = sched.step(50);
    float fEnd = sched.step(100);
    // 20260320 ZJH 开始时接近 base_lr，中间下降，结束时接近 eta_min
    EXPECT_NEAR(fStart, 0.1f, 0.001f);
    EXPECT_LT(fMid, fStart);
    EXPECT_NEAR(fEnd, 1e-6f, 0.001f);
}

TEST(Phase5, WarmupCosineAnnealingLR) {
    WarmupCosineAnnealingLR sched(0.1f, 10, 100, 1e-6f);
    float fFirst = sched.step(0);
    float fWarmupEnd = sched.step(9);
    // 20260320 ZJH 预热阶段 lr 逐渐增加
    EXPECT_LT(fFirst, fWarmupEnd);
    EXPECT_NEAR(fWarmupEnd, 0.1f, 0.01f);
}

TEST(Phase5, StepLR) {
    StepLR sched(0.1f, 30, 0.1f);
    EXPECT_NEAR(sched.step(0), 0.1f, 1e-5f);
    EXPECT_NEAR(sched.step(29), 0.1f, 1e-5f);
    EXPECT_NEAR(sched.step(30), 0.01f, 1e-5f);
    EXPECT_NEAR(sched.step(60), 0.001f, 1e-5f);
}

// ===== Metrics 测试 =====

TEST(Phase5, ConfusionMatrixAccuracy) {
    // 20260320 ZJH 测试混淆矩阵准确率计算
    ConfusionMatrix cm(3);
    // 20260320 ZJH 完美预测
    cm.update(0, 0); cm.update(1, 1); cm.update(2, 2);
    cm.update(0, 0); cm.update(1, 1);
    EXPECT_NEAR(cm.accuracy(), 1.0f, 1e-5f);
    // 20260320 ZJH 加一个错误预测
    cm.update(0, 1);
    EXPECT_NEAR(cm.accuracy(), 5.0f / 6.0f, 1e-4f);
}

TEST(Phase5, ConfusionMatrixPrecisionRecallF1) {
    ConfusionMatrix cm(2);
    // TP=3, FP=1, FN=2, TN=4
    cm.batchUpdate({0,0,0,0,0,1,1,1,1,1}, {0,0,0,1,1,0,1,1,1,1});
    auto prec = cm.precision();
    auto rec = cm.recall();
    auto f1 = cm.f1Score();
    // 类 0: TP=3, FP=1(类1被预测为0), FN=2(类0被预测为1)
    EXPECT_NEAR(prec[0], 3.0f / 4.0f, 1e-4f);
    EXPECT_NEAR(rec[0], 3.0f / 5.0f, 1e-4f);
    EXPECT_GT(f1[0], 0.0f);
}

TEST(Phase5, EarlyStoppingTrigger) {
    EarlyStopping es(3, 0.0f);
    EXPECT_FALSE(es.step(1.0f));  // 改善
    EXPECT_FALSE(es.step(0.9f));  // 改善
    EXPECT_FALSE(es.step(0.95f)); // 未改善 counter=1
    EXPECT_FALSE(es.step(0.95f)); // 未改善 counter=2
    EXPECT_TRUE(es.step(0.95f));  // 未改善 counter=3 = patience -> 触发
}

TEST(Phase5, EarlyStoppingReset) {
    EarlyStopping es(3, 0.0f);
    es.step(1.0f); es.step(1.1f); es.step(1.2f);
    EXPECT_EQ(es.counter(), 2);
    es.step(0.5f);  // 改善，重置计数
    EXPECT_EQ(es.counter(), 0);
}

TEST(Phase5, ComputeIoU) {
    DetectionBox a{0, 0, 10, 10, 0, 1.0f};
    DetectionBox b{5, 5, 15, 15, 0, 1.0f};
    float fIoU = computeIoU(a, b);
    // 交集 = 5*5=25, 并集 = 100+100-25=175
    EXPECT_NEAR(fIoU, 25.0f / 175.0f, 1e-4f);
}

TEST(Phase5, ComputeROCAUC) {
    // 20260320 ZJH 完美分离的情况 AUC=1.0
    std::vector<float> scores = {0.9f, 0.8f, 0.2f, 0.1f};
    std::vector<int> labels = {1, 1, 0, 0};
    float fAUC = computeROCAUC(scores, labels);
    EXPECT_NEAR(fAUC, 1.0f, 0.01f);
}

TEST(Phase5, FindOptimalThreshold) {
    std::vector<float> scores = {0.9f, 0.7f, 0.3f, 0.1f};
    std::vector<int> labels = {1, 1, 0, 0};
    float fThresh = findOptimalThreshold(scores, labels);
    EXPECT_GT(fThresh, 0.2f);
    EXPECT_LT(fThresh, 0.8f);
}

// ===== Checkpoint 测试 =====

TEST(Phase5, CheckpointSaveLoad) {
    // 20260320 ZJH 测试检查点保存和恢复
    Sequential model;
    model.add(std::make_unique<Linear>(4, 8, true));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(8, 2, true));

    // 20260320 ZJH 修改一个参数值使其非默认
    auto params = model.parameters();
    if (!params.empty()) {
        params[0]->mutableFloatDataPtr()[0] = 42.0f;
    }

    // 20260320 ZJH 保存检查点
    CheckpointManager mgr("test_ckpt_dir", 5, true);
    auto path = mgr.saveCheckpoint(model, "test_save", 10, 0.5f, 95.0f);
    EXPECT_FALSE(path.empty());
    EXPECT_TRUE(std::filesystem::exists(path));

    // 20260320 ZJH 创建新模型并恢复
    Sequential model2;
    model2.add(std::make_unique<Linear>(4, 8, true));
    model2.add(std::make_unique<ReLU>());
    model2.add(std::make_unique<Linear>(8, 2, true));

    auto info = CheckpointManager::loadCheckpoint(model2, path);
    EXPECT_EQ(info.nEpoch, 10);
    EXPECT_NEAR(info.fLoss, 0.5f, 1e-5f);
    EXPECT_NEAR(info.fMetric, 95.0f, 1e-5f);

    // 20260320 ZJH 验证参数已恢复
    auto params2 = model2.parameters();
    EXPECT_NEAR(params2[0]->floatDataPtr()[0], 42.0f, 1e-5f);

    // 20260320 ZJH 清理
    std::filesystem::remove_all("test_ckpt_dir");
}

TEST(Phase5, CheckpointManagerOnEpochEnd) {
    Sequential model;
    model.add(std::make_unique<Linear>(4, 2, true));

    CheckpointManager mgr("test_ckpt_dir2", 5, true);

    // 20260320 ZJH 第一个 epoch 应保存 best
    auto s1 = mgr.onEpochEnd(model, 0, 1.0f, 50.0f);
    EXPECT_FALSE(s1.empty());

    // 20260320 ZJH 更差的不应保存 best
    auto s2 = mgr.onEpochEnd(model, 1, 1.5f, 45.0f);
    EXPECT_TRUE(s2.empty());

    // 20260320 ZJH 更好的应保存 best
    auto s3 = mgr.onEpochEnd(model, 2, 0.5f, 70.0f);
    EXPECT_FALSE(s3.empty());

    // 20260320 ZJH 第 5 个 epoch 应定期保存
    mgr.onEpochEnd(model, 3, 0.8f, 60.0f);
    mgr.onEpochEnd(model, 4, 0.7f, 65.0f);  // epoch 5 (index 4)

    // 20260320 ZJH 清理
    std::filesystem::remove_all("test_ckpt_dir2");
}

// ===== 并行推理测试 =====

TEST(Phase5, InferenceThreadPoolBatchInfer) {
    // 20260320 ZJH 测试并行推理线程池
    Sequential model;
    model.add(std::make_unique<Linear>(4, 8, true));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(8, 2, true));
    model.eval();

    InferenceThreadPool pool(2);
    std::vector<Tensor> inputs;
    for (int i = 0; i < 8; ++i) inputs.push_back(Tensor::randn({1, 4}));

    auto result = pool.batchInfer(&model, inputs);
    EXPECT_EQ(result.nNumImages, 8);
    EXPECT_EQ(result.nNumThreads, 2);
    EXPECT_EQ((int)result.vecOutputs.size(), 8);
    EXPECT_GT(result.fTotalTimeMs, 0.0f);
    // 20260320 ZJH 验证每个输出形状正确
    for (auto& out : result.vecOutputs) {
        EXPECT_EQ(out.shape(0), 1);
        EXPECT_EQ(out.shape(1), 2);
    }
}

TEST(Phase5, InferenceTimerBenchmark) {
    float fMs = InferenceTimer::benchmarkMs([]() {
        auto t = Tensor::randn({1, 100});
        auto r = Tensor::randn({1, 100});
        tensorAdd(t, r);
    }, 5);
    EXPECT_GE(fMs, 0.0f);
}

TEST(Phase5, ModelOptimizerAnalyze) {
    Sequential model;
    model.add(std::make_unique<Linear>(784, 128, true));
    model.add(std::make_unique<Linear>(128, 10, true));
    auto report = ModelOptimizer::analyze(model);
    EXPECT_GT(report.nTotalParams, 100000);
    EXPECT_GT(report.fEstMemoryMB, 0.0f);
    EXPECT_GT(report.nRecommendedThreads, 0);
    EXPECT_GT(report.nRecommendedBatchSize, 0);
}

// ===== 知识蒸馏测试 =====

TEST(Phase5, KLDivLoss) {
    KLDivLoss kl;
    auto student = Tensor::fromData(std::vector<float>{1,2,3,4, 2,1,3,4}.data(), {2, 4});
    auto teacher = Tensor::fromData(std::vector<float>{1,2,3,4, 2,1,3,4}.data(), {2, 4});
    auto loss = kl.forward(student, teacher, 4.0f);
    // 20260320 ZJH 相同分布 KL=0
    EXPECT_NEAR(loss.item(), 0.0f, 0.01f);
}

TEST(Phase5, KLDivLossDifferent) {
    KLDivLoss kl;
    auto student = Tensor::fromData(std::vector<float>{5,0,0,0}.data(), {1, 4});
    auto teacher = Tensor::fromData(std::vector<float>{0,0,0,5}.data(), {1, 4});
    auto loss = kl.forward(student, teacher, 4.0f);
    // 20260320 ZJH 不同分布 KL>0
    EXPECT_GT(loss.item(), 0.0f);
}

TEST(Phase5, ModelPruner) {
    Sequential model;
    model.add(std::make_unique<Linear>(10, 5, true));
    float fBefore = ModelPruner::sparsity(model);
    ModelPruner::prune(model, 0.5f);
    float fAfter = ModelPruner::sparsity(model);
    EXPECT_GT(fAfter, fBefore);
    EXPECT_GE(fAfter, 0.4f);
}

TEST(Phase5, DistillationManagerComputeLoss) {
    DistillConfig cfg;
    cfg.fTemperature = 4.0f;
    cfg.fAlphaSoft = 0.5f;
    cfg.fAlphaHard = 0.5f;
    DistillationManager mgr(cfg);
    auto sLogits = Tensor::randn({2, 4});
    auto tLogits = Tensor::randn({2, 4});
    auto hardLoss = Tensor::full({1}, 1.5f);
    float fTotal = mgr.computeLoss(sLogits, tLogits, hardLoss, 0);
    EXPECT_GT(fTotal, 0.0f);
    EXPECT_GT(mgr.lastHardLoss(), 0.0f);
}

TEST(Phase5, CompressionRatio) {
    Sequential big;
    big.add(std::make_unique<Linear>(100, 50, true));
    big.add(std::make_unique<Linear>(50, 10, true));
    Sequential small;
    small.add(std::make_unique<Linear>(100, 10, true));
    float fRatio = DistillationManager::compressionRatio(big, small);
    EXPECT_GT(fRatio, 1.0f);
}

TEST(Phase5, QuantizationHelper) {
    auto input = Tensor::fromData(std::vector<float>{-1.0f, 0.0f, 0.5f, 1.0f}.data(), {1, 4});
    auto [scale, zp] = QuantizationHelper::calibrate(input);
    EXPECT_GT(scale, 0.0f);
    auto quantized = QuantizationHelper::fakeQuantize(input, scale, zp);
    EXPECT_EQ(quantized.numel(), 4);
    // 20260320 ZJH 量化后值应接近原始值
    const float* pO = input.floatDataPtr();
    const float* pQ = quantized.floatDataPtr();
    // 20260320 ZJH INT8 量化精度有限，容差放宽
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(pO[i], pQ[i], scale * 1.5f);
}
