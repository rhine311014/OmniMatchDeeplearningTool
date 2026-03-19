// 20260319 ZJH DeepForge Phase 1D — MNIST MLP 训练主程序
// 两层全连接网络：784 -> 128 (ReLU) -> 10 (Softmax + CrossEntropy)
// SGD 优化器，batch_size=64, lr=0.01, epochs=10

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.mnist;
import df.hal.cpu_backend;

int main() {
    // 20260319 ZJH 训练超参数定义
    const int nEpochs = 10;           // 20260319 ZJH 训练轮数
    const int nBatchSize = 64;        // 20260319 ZJH 批次大小
    const float fLearningRate = 0.01f; // 20260319 ZJH 学习率
    const int nInputDim = 784;        // 20260319 ZJH 输入维度（28x28 像素展平）
    const int nHiddenDim = 128;       // 20260319 ZJH 隐藏层维度
    const int nOutputDim = 10;        // 20260319 ZJH 输出维度（10 个数字类别）

    std::printf("========================================\n");
    std::printf("  DeepForge Phase 1D - MNIST MLP Train\n");
    std::printf("========================================\n");
    std::printf("Epochs: %d\n", nEpochs);
    std::printf("Batch size: %d\n", nBatchSize);
    std::printf("Learning rate: %.4f\n", static_cast<double>(fLearningRate));
    std::printf("Architecture: %d -> %d (ReLU) -> %d (Softmax+CE)\n", nInputDim, nHiddenDim, nOutputDim);
    std::printf("\n");

    // ===== 第一步：加载 MNIST 数据 =====
    // 20260319 ZJH MNIST 数据文件的相对路径
    std::string strTrainImagesPath = "data/mnist/train-images-idx3-ubyte";
    std::string strTrainLabelsPath = "data/mnist/train-labels-idx1-ubyte";
    std::string strTestImagesPath  = "data/mnist/t10k-images-idx3-ubyte";
    std::string strTestLabelsPath  = "data/mnist/t10k-labels-idx1-ubyte";

    df::MnistDataset trainData;  // 20260319 ZJH 训练集
    df::MnistDataset testData;   // 20260319 ZJH 测试集
    bool bUseSyntheticData = false;  // 20260319 ZJH 是否使用合成数据

    try {
        std::printf("Loading MNIST training data...\n");
        trainData = df::loadMnist(strTrainImagesPath, strTrainLabelsPath);
        std::printf("Loading MNIST test data...\n");
        testData = df::loadMnist(strTestImagesPath, strTestLabelsPath);
    } catch (const std::runtime_error& e) {
        // 20260319 ZJH 数据文件缺失时切换到合成数据模式
        std::printf("\n[WARNING] %s\n", e.what());
        std::printf("MNIST data not found. Switching to SYNTHETIC DATA mode.\n");
        std::printf("(Place MNIST files in data/mnist/ for real training)\n\n");
        bUseSyntheticData = true;

        // 20260319 ZJH 生成合成分类数据：10 个类别，每类特征向量中对应位置较高
        // 训练集 1000 样本，测试集 200 样本
        int nTrainSynth = 1000;
        int nTestSynth = 200;

        // 20260319 ZJH 生成合成数据的 lambda
        auto generateSynthetic = [&](int nSamples) -> df::MnistDataset {
            df::MnistDataset ds;
            ds.m_nSamples = nSamples;
            // 20260319 ZJH 初始化图像和标签张量
            ds.m_images = df::Tensor::zeros({nSamples, nInputDim});
            ds.m_labels = df::Tensor::zeros({nSamples, nOutputDim});
            float* pImages = ds.m_images.mutableFloatDataPtr();
            float* pLabels = ds.m_labels.mutableFloatDataPtr();

            // 20260319 ZJH 简单合成规则：类别 c 的样本在维度 [c*78, (c+1)*78) 有较高值
            for (int i = 0; i < nSamples; ++i) {
                int nClass = i % nOutputDim;  // 20260319 ZJH 循环分配类别
                // 20260319 ZJH 基底噪声 0.1
                for (int j = 0; j < nInputDim; ++j) {
                    pImages[i * nInputDim + j] = 0.1f;
                }
                // 20260319 ZJH 类别特征区域设为 0.9
                int nStart = nClass * (nInputDim / nOutputDim);
                int nEnd = (nClass + 1) * (nInputDim / nOutputDim);
                if (nEnd > nInputDim) nEnd = nInputDim;
                for (int j = nStart; j < nEnd; ++j) {
                    pImages[i * nInputDim + j] = 0.9f;
                }
                // 20260319 ZJH one-hot 标签
                pLabels[i * nOutputDim + nClass] = 1.0f;
            }
            return ds;
        };

        trainData = generateSynthetic(nTrainSynth);
        testData = generateSynthetic(nTestSynth);
    }

    std::printf("Training samples: %d\n", trainData.m_nSamples);
    std::printf("Test samples: %d\n", testData.m_nSamples);
    std::printf("\n");

    // ===== 第二步：初始化网络权重 =====
    // 20260319 ZJH W1: [784, 128]，小随机初始化（randn * 0.01）
    auto matW1 = df::Tensor::randn({nInputDim, nHiddenDim});
    // 20260319 ZJH 缩放 W1 为小值：逐元素乘 0.01
    {
        float* pW1 = matW1.mutableFloatDataPtr();  // 20260319 ZJH W1 可写指针
        for (int i = 0; i < matW1.numel(); ++i) {
            pW1[i] *= 0.01f;  // 20260319 ZJH 缩小初始权重，防止训练初期梯度爆炸
        }
    }
    df::tensorSetRequiresGrad(matW1, true);  // 20260319 ZJH W1 需要梯度

    // 20260319 ZJH b1: [1, 128]，零初始化
    auto matB1 = df::Tensor::zeros({1, nHiddenDim});
    df::tensorSetRequiresGrad(matB1, true);  // 20260319 ZJH b1 需要梯度

    // 20260319 ZJH W2: [128, 10]，小随机初始化（randn * 0.01）
    auto matW2 = df::Tensor::randn({nHiddenDim, nOutputDim});
    {
        float* pW2 = matW2.mutableFloatDataPtr();  // 20260319 ZJH W2 可写指针
        for (int i = 0; i < matW2.numel(); ++i) {
            pW2[i] *= 0.01f;  // 20260319 ZJH 缩小初始权重
        }
    }
    df::tensorSetRequiresGrad(matW2, true);  // 20260319 ZJH W2 需要梯度

    // 20260319 ZJH b2: [1, 10]，零初始化
    auto matB2 = df::Tensor::zeros({1, nOutputDim});
    df::tensorSetRequiresGrad(matB2, true);  // 20260319 ZJH b2 需要梯度

    std::printf("Model initialized:\n");
    std::printf("  W1: [%d, %d]\n", nInputDim, nHiddenDim);
    std::printf("  b1: [1, %d]\n", nHiddenDim);
    std::printf("  W2: [%d, %d]\n", nHiddenDim, nOutputDim);
    std::printf("  b2: [1, %d]\n", nOutputDim);
    std::printf("\n");

    // ===== 第三步：训练循环 =====
    // 20260319 ZJH 计算每轮的批次数量
    int nNumBatches = trainData.m_nSamples / nBatchSize;  // 20260319 ZJH 整除，丢弃最后不足一个 batch 的数据

    for (int nEpoch = 0; nEpoch < nEpochs; ++nEpoch) {
        auto timeStart = std::chrono::steady_clock::now();  // 20260319 ZJH 计时开始

        float fEpochLoss = 0.0f;   // 20260319 ZJH 当前轮累积损失
        int nCorrect = 0;           // 20260319 ZJH 当前轮正确预测数

        for (int nBatch = 0; nBatch < nNumBatches; ++nBatch) {
            // 20260319 ZJH 提取当前批次的图像和标签
            int nStart = nBatch * nBatchSize;  // 20260319 ZJH 当前批次起始索引
            // 20260319 ZJH 用 slice 截取当前批次数据
            auto batchImages = df::tensorSlice(trainData.m_images, 0, nStart, nStart + nBatchSize);  // [batch, 784]
            auto batchLabels = df::tensorSlice(trainData.m_labels, 0, nStart, nStart + nBatchSize);  // [batch, 10]
            // 20260319 ZJH 将切片视图连续化，确保后续 matmul 正确
            batchImages = batchImages.contiguous();
            batchLabels = batchLabels.contiguous();

            // ===== 前向传播 =====
            // 20260319 ZJH 第一层：h = ReLU(x @ W1 + b1)
            auto matZ1 = df::tensorMatmul(batchImages, matW1);  // [batch, 128] = [batch, 784] @ [784, 128]
            auto matH1 = df::tensorAddBias(matZ1, matB1);       // [batch, 128] + [1, 128] 广播加偏置
            auto matA1 = df::tensorReLU(matH1);                 // [batch, 128] ReLU 激活

            // 20260319 ZJH 第二层：logits = h @ W2 + b2
            auto matZ2 = df::tensorMatmul(matA1, matW2);    // [batch, 10] = [batch, 128] @ [128, 10]
            auto matLogits = df::tensorAddBias(matZ2, matB2); // [batch, 10] + [1, 10] 广播加偏置

            // 20260319 ZJH 计算损失：softmax + cross-entropy
            auto loss = df::tensorSoftmaxCrossEntropy(matLogits, batchLabels);
            fEpochLoss += loss.item();  // 20260319 ZJH 累加批次损失

            // 20260319 ZJH 计算训练准确率
            auto vecPredicted = df::tensorArgmax(matLogits);  // 20260319 ZJH 预测类别索引
            auto vecActual = df::tensorArgmax(batchLabels);   // 20260319 ZJH 实际类别索引
            for (int i = 0; i < nBatchSize; ++i) {
                if (vecPredicted[static_cast<size_t>(i)] == vecActual[static_cast<size_t>(i)]) {
                    nCorrect++;  // 20260319 ZJH 预测正确则计数加 1
                }
            }

            // ===== 反向传播 =====
            // 20260319 ZJH 清零所有参数的梯度（防止梯度累积）
            df::tensorZeroGrad(matW1);
            df::tensorZeroGrad(matB1);
            df::tensorZeroGrad(matW2);
            df::tensorZeroGrad(matB2);

            // 20260319 ZJH 执行反向传播，计算所有参数的梯度
            df::tensorBackward(loss);

            // ===== SGD 参数更新 =====
            // 20260319 ZJH param -= lr * grad（简单 SGD，无动量）
            // 20260319 ZJH 更新 W1
            {
                auto gradW1 = df::tensorGetGrad(matW1);  // 20260319 ZJH 获取 W1 的梯度
                if (gradW1.numel() > 0) {
                    float* pW1 = matW1.mutableFloatDataPtr();           // 20260319 ZJH W1 可写指针
                    const float* pGrad = gradW1.floatDataPtr();         // 20260319 ZJH 梯度只读指针
                    for (int i = 0; i < matW1.numel(); ++i) {
                        pW1[i] -= fLearningRate * pGrad[i];  // 20260319 ZJH SGD 更新规则
                    }
                }
            }
            // 20260319 ZJH 更新 b1
            {
                auto gradB1 = df::tensorGetGrad(matB1);
                if (gradB1.numel() > 0) {
                    float* pB1 = matB1.mutableFloatDataPtr();
                    const float* pGrad = gradB1.floatDataPtr();
                    for (int i = 0; i < matB1.numel(); ++i) {
                        pB1[i] -= fLearningRate * pGrad[i];
                    }
                }
            }
            // 20260319 ZJH 更新 W2
            {
                auto gradW2 = df::tensorGetGrad(matW2);
                if (gradW2.numel() > 0) {
                    float* pW2 = matW2.mutableFloatDataPtr();
                    const float* pGrad = gradW2.floatDataPtr();
                    for (int i = 0; i < matW2.numel(); ++i) {
                        pW2[i] -= fLearningRate * pGrad[i];
                    }
                }
            }
            // 20260319 ZJH 更新 b2
            {
                auto gradB2 = df::tensorGetGrad(matB2);
                if (gradB2.numel() > 0) {
                    float* pB2 = matB2.mutableFloatDataPtr();
                    const float* pGrad = gradB2.floatDataPtr();
                    for (int i = 0; i < matB2.numel(); ++i) {
                        pB2[i] -= fLearningRate * pGrad[i];
                    }
                }
            }
        }

        // 20260319 ZJH 计算当前轮的平均损失和训练准确率
        float fAvgLoss = fEpochLoss / static_cast<float>(nNumBatches);
        float fTrainAcc = 100.0f * static_cast<float>(nCorrect) / static_cast<float>(nNumBatches * nBatchSize);

        // ===== 测试集评估 =====
        // 20260319 ZJH 在测试集上计算准确率（不需要梯度）
        int nTestCorrect = 0;  // 20260319 ZJH 测试集正确预测数
        int nTestBatches = testData.m_nSamples / nBatchSize;  // 20260319 ZJH 测试批次数
        for (int nBatch = 0; nBatch < nTestBatches; ++nBatch) {
            int nStart = nBatch * nBatchSize;
            auto batchImages = df::tensorSlice(testData.m_images, 0, nStart, nStart + nBatchSize);
            auto batchLabels = df::tensorSlice(testData.m_labels, 0, nStart, nStart + nBatchSize);
            batchImages = batchImages.contiguous();
            batchLabels = batchLabels.contiguous();

            // 20260319 ZJH 前向传播（不创建计算图）
            // 使用不带 requiresGrad 的中间张量来避免构建计算图
            auto cW1 = df::Tensor::fromData(matW1.floatDataPtr(), matW1.shapeVec());  // 20260319 ZJH W1 的不需要梯度的副本
            auto cB1 = df::Tensor::fromData(matB1.floatDataPtr(), matB1.shapeVec());  // 20260319 ZJH b1 的副本
            auto cW2 = df::Tensor::fromData(matW2.floatDataPtr(), matW2.shapeVec());  // 20260319 ZJH W2 的副本
            auto cB2 = df::Tensor::fromData(matB2.floatDataPtr(), matB2.shapeVec());  // 20260319 ZJH b2 的副本

            // 20260319 ZJH 第一层推理
            auto matZ1 = df::tensorMatmul(batchImages, cW1);
            auto matH1 = df::tensorAddBias(matZ1, cB1);
            auto matA1 = df::tensorReLU(matH1);

            // 20260319 ZJH 第二层推理
            auto matZ2 = df::tensorMatmul(matA1, cW2);
            auto matLogits = df::tensorAddBias(matZ2, cB2);

            // 20260319 ZJH 计算准确率
            auto vecPredicted = df::tensorArgmax(matLogits);
            auto vecActual = df::tensorArgmax(batchLabels);
            for (int i = 0; i < nBatchSize; ++i) {
                if (vecPredicted[static_cast<size_t>(i)] == vecActual[static_cast<size_t>(i)]) {
                    nTestCorrect++;
                }
            }
        }
        float fTestAcc = 100.0f * static_cast<float>(nTestCorrect) / static_cast<float>(nTestBatches * nBatchSize);

        auto timeEnd = std::chrono::steady_clock::now();  // 20260319 ZJH 计时结束
        auto nDuration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();

        // 20260319 ZJH 输出当前轮的训练信息
        std::printf("Epoch %d/%d | Loss: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%% | Time: %lldms\n",
                    nEpoch + 1, nEpochs,
                    static_cast<double>(fAvgLoss),
                    static_cast<double>(fTrainAcc),
                    static_cast<double>(fTestAcc),
                    static_cast<long long>(nDuration));
    }

    std::printf("\nTraining complete!\n");

    return 0;  // 20260319 ZJH 正常退出
}
