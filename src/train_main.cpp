// 20260319 ZJH OmniMatch Phase 2 — MNIST 训练主程序（MLP / ResNet18 双模式）
// 命令行参数:
//   --model mlp|resnet18    模型类型，默认 mlp
//   --epochs N              训练轮数，默认 10
//   --lr RATE               学习率，默认 mlp=0.01 / resnet18=0.001
//   --batch-size N          批次大小，默认 64

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <memory>
#include <cstring>

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.linear;
import om.engine.activations;
import om.engine.conv;
import om.engine.resnet;
import om.engine.optimizer;
import om.engine.loss;
import om.engine.mnist;
import om.hal.cpu_backend;

// 20260319 ZJH printUsage — 输出命令行用法说明
static void printUsage(const char* pProgramName) {
    std::printf("Usage: %s [--model mlp|resnet18] [--epochs N] [--lr RATE] [--batch-size N]\n\n", pProgramName);
    std::printf("Options:\n");
    std::printf("  --model      Model type: mlp (default) or resnet18\n");
    std::printf("  --epochs     Number of training epochs (default: 10)\n");
    std::printf("  --lr         Learning rate (default: 0.01 for mlp, 0.001 for resnet18)\n");
    std::printf("  --batch-size Batch size (default: 64)\n");
    std::printf("  --help       Show this help message\n");
}

int main(int argc, char* argv[]) {
    // 20260319 ZJH 默认训练超参数
    std::string strModel = "mlp";       // 20260319 ZJH 模型类型
    int nEpochs = 10;                   // 20260319 ZJH 训练轮数
    float fLearningRate = -1.0f;        // 20260319 ZJH 学习率，-1 表示使用模型默认值
    int nBatchSize = 64;                // 20260319 ZJH 批次大小
    const int nInputDim = 784;          // 20260319 ZJH 输入维度（28x28 像素展平）
    const int nHiddenDim = 128;         // 20260319 ZJH MLP 隐藏层维度
    const int nOutputDim = 10;          // 20260319 ZJH 输出维度（10 个数字类别）

    // 20260319 ZJH 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            strModel = argv[++i];  // 20260319 ZJH 读取模型类型
        } else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            nEpochs = std::atoi(argv[++i]);  // 20260319 ZJH 读取训练轮数
        } else if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            fLearningRate = static_cast<float>(std::atof(argv[++i]));  // 20260319 ZJH 读取学习率
        } else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            nBatchSize = std::atoi(argv[++i]);  // 20260319 ZJH 读取批次大小
        } else if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);  // 20260319 ZJH 输出用法后退出
            return 0;
        } else {
            std::printf("Unknown argument: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;  // 20260319 ZJH 未知参数时返回错误码
        }
    }

    // 20260319 ZJH 验证模型类型
    bool bUseResNet = (strModel == "resnet18");  // 20260319 ZJH 是否使用 ResNet18
    if (strModel != "mlp" && strModel != "resnet18") {
        std::printf("Error: unknown model type '%s'. Use 'mlp' or 'resnet18'.\n", strModel.c_str());
        return 1;  // 20260319 ZJH 无效模型类型
    }

    // 20260319 ZJH 设置默认学习率（基于模型类型）
    if (fLearningRate < 0.0f) {
        fLearningRate = bUseResNet ? 0.001f : 0.01f;  // 20260319 ZJH ResNet 用 0.001，MLP 用 0.01
    }

    // 20260319 ZJH 输出训练配置信息
    std::printf("========================================\n");
    std::printf("  OmniMatch Phase 2 - MNIST Train\n");
    if (bUseResNet) {
        std::printf("  Model: ResNet-18 (CNN)\n");
    } else {
        std::printf("  Model: MLP (784->128->10)\n");
    }
    std::printf("========================================\n");
    std::printf("Epochs: %d\n", nEpochs);
    std::printf("Batch size: %d\n", nBatchSize);
    std::printf("Learning rate: %.4f\n", static_cast<double>(fLearningRate));
    if (bUseResNet) {
        std::printf("Optimizer: Adam\n");
        std::printf("Architecture: Conv3x3(1,64) -> [BasicBlock x8] -> AvgPool -> FC(512,10)\n");
    } else {
        std::printf("Optimizer: SGD\n");
        std::printf("Architecture: %d -> %d (ReLU) -> %d (Softmax+CE)\n", nInputDim, nHiddenDim, nOutputDim);
    }
    std::printf("\n");

    // ===== 第一步：加载 MNIST 数据 =====
    // 20260319 ZJH MNIST 数据文件的相对路径
    std::string strTrainImagesPath = "data/mnist/train-images-idx3-ubyte";
    std::string strTrainLabelsPath = "data/mnist/train-labels-idx1-ubyte";
    std::string strTestImagesPath  = "data/mnist/t10k-images-idx3-ubyte";
    std::string strTestLabelsPath  = "data/mnist/t10k-labels-idx1-ubyte";

    om::MnistDataset trainData;  // 20260319 ZJH 训练集
    om::MnistDataset testData;   // 20260319 ZJH 测试集
    bool bUseSyntheticData = false;  // 20260319 ZJH 是否使用合成数据

    try {
        std::printf("Loading MNIST training data...\n");
        trainData = om::loadMnist(strTrainImagesPath, strTrainLabelsPath);
        std::printf("Loading MNIST test data...\n");
        testData = om::loadMnist(strTestImagesPath, strTestLabelsPath);
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
        auto generateSynthetic = [&](int nSamples) -> om::MnistDataset {
            om::MnistDataset ds;
            ds.m_nSamples = nSamples;
            // 20260319 ZJH 初始化图像和标签张量
            ds.m_images = om::Tensor::zeros({nSamples, nInputDim});
            ds.m_labels = om::Tensor::zeros({nSamples, nOutputDim});
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

    // ===== 第二步：构建模型 =====
    // 20260319 ZJH 根据模型类型选择不同的网络架构
    std::shared_ptr<om::Module> pModel;  // 20260319 ZJH 通用模型指针

    // 20260319 ZJH ResNet18 模型
    std::shared_ptr<om::ResNet18> pResNet;
    // 20260319 ZJH MLP 模型
    std::shared_ptr<om::Sequential> pMLP;

    if (bUseResNet) {
        // 20260319 ZJH ResNet18: 输入 [N,1,28,28] -> 输出 [N,10]
        pResNet = std::make_shared<om::ResNet18>(nOutputDim);
        pModel = pResNet;
    } else {
        // 20260319 ZJH MLP: Sequential(Linear(784,128), ReLU, Linear(128,10))
        pMLP = std::make_shared<om::Sequential>();
        pMLP->add(std::make_shared<om::Linear>(nInputDim, nHiddenDim));   // 20260319 ZJH 第一层：784->128
        pMLP->add(std::make_shared<om::ReLU>());                           // 20260319 ZJH ReLU 激活
        pMLP->add(std::make_shared<om::Linear>(nHiddenDim, nOutputDim));  // 20260319 ZJH 第二层：128->10
        pModel = pMLP;
    }

    // 20260319 ZJH 获取模型参数
    auto vecParams = pModel->parameters();
    int nTotalParams = 0;  // 20260319 ZJH 总参数量
    for (auto* pParam : vecParams) {
        nTotalParams += pParam->numel();  // 20260319 ZJH 累加各参数元素数
    }
    std::printf("Model initialized:\n");
    std::printf("  Parameters: %d tensors, %d total values\n",
                static_cast<int>(vecParams.size()), nTotalParams);
    std::printf("\n");

    // ===== 第三步：创建优化器和损失函数 =====
    // 20260319 ZJH 根据模型类型选择优化器
    // Adam 用于 ResNet（CNN 训练更稳定），SGD 用于 MLP
    std::unique_ptr<om::Adam> pAdamOpt;    // 20260319 ZJH Adam 优化器（ResNet 使用）
    std::unique_ptr<om::SGD> pSgdOpt;      // 20260319 ZJH SGD 优化器（MLP 使用）

    if (bUseResNet) {
        pAdamOpt = std::make_unique<om::Adam>(vecParams, fLearningRate);  // 20260319 ZJH 创建 Adam
    } else {
        pSgdOpt = std::make_unique<om::SGD>(vecParams, fLearningRate);    // 20260319 ZJH 创建 SGD
    }

    // 20260319 ZJH 创建交叉熵损失函数
    om::CrossEntropyLoss criterion;

    // ===== 第四步：训练循环 =====
    // 20260319 ZJH 计算每轮的批次数量
    int nNumBatches = trainData.m_nSamples / nBatchSize;  // 20260319 ZJH 整除，丢弃不足一个 batch 的数据

    for (int nEpoch = 0; nEpoch < nEpochs; ++nEpoch) {
        auto timeStart = std::chrono::steady_clock::now();  // 20260319 ZJH 计时开始

        float fEpochLoss = 0.0f;   // 20260319 ZJH 当前轮累积损失
        int nCorrect = 0;           // 20260319 ZJH 当前轮正确预测数

        for (int nBatch = 0; nBatch < nNumBatches; ++nBatch) {
            // 20260319 ZJH 提取当前批次的图像和标签
            int nStart = nBatch * nBatchSize;  // 20260319 ZJH 当前批次起始索引
            // 20260319 ZJH 用 slice 截取当前批次数据
            auto batchImages = om::tensorSlice(trainData.m_images, 0, nStart, nStart + nBatchSize);
            auto batchLabels = om::tensorSlice(trainData.m_labels, 0, nStart, nStart + nBatchSize);
            // 20260319 ZJH 将切片视图连续化，确保后续运算正确
            batchImages = batchImages.contiguous();
            batchLabels = batchLabels.contiguous();

            // 20260319 ZJH 如果是 ResNet，将 [batch, 784] reshape 为 [batch, 1, 28, 28]
            if (bUseResNet) {
                batchImages = om::tensorReshape(batchImages, {nBatchSize, 1, 28, 28});
            }

            // ===== 前向传播 =====
            // 20260319 ZJH 使用模型的 forward 方法
            auto matLogits = pModel->forward(batchImages);  // 20260319 ZJH [batch, 10]

            // 20260319 ZJH 计算损失
            auto loss = criterion.forward(matLogits, batchLabels);  // 20260319 ZJH 标量损失
            fEpochLoss += loss.item();  // 20260319 ZJH 累加批次损失

            // 20260319 ZJH 计算训练准确率
            auto vecPredicted = om::tensorArgmax(matLogits);  // 20260319 ZJH 预测类别索引
            auto vecActual = om::tensorArgmax(batchLabels);   // 20260319 ZJH 实际类别索引
            for (int i = 0; i < nBatchSize; ++i) {
                if (vecPredicted[static_cast<size_t>(i)] == vecActual[static_cast<size_t>(i)]) {
                    nCorrect++;  // 20260319 ZJH 预测正确则计数加 1
                }
            }

            // ===== 反向传播 + 参数更新 =====
            // 20260319 ZJH 清零梯度 -> 反向传播 -> 参数更新
            if (bUseResNet) {
                pAdamOpt->zeroGrad();       // 20260319 ZJH Adam 清零梯度
                om::tensorBackward(loss);    // 20260319 ZJH 执行反向传播
                pAdamOpt->step();            // 20260319 ZJH Adam 参数更新
            } else {
                pSgdOpt->zeroGrad();        // 20260319 ZJH SGD 清零梯度
                om::tensorBackward(loss);    // 20260319 ZJH 执行反向传播
                pSgdOpt->step();             // 20260319 ZJH SGD 参数更新
            }

            // 20260319 ZJH 每 5 个 batch 打印一次进度（ResNet 很慢，提供实时反馈）
            if (bUseResNet && (nBatch + 1) % 5 == 0) {
                float fBatchAvgLoss = fEpochLoss / static_cast<float>(nBatch + 1);
                std::printf("  Epoch %d/%d, Batch %d/%d, Avg Loss: %.4f\n",
                            nEpoch + 1, nEpochs, nBatch + 1, nNumBatches,
                            static_cast<double>(fBatchAvgLoss));
            }
        }

        // 20260319 ZJH 计算当前轮的平均损失和训练准确率
        float fAvgLoss = fEpochLoss / static_cast<float>(nNumBatches);
        float fTrainAcc = 100.0f * static_cast<float>(nCorrect) / static_cast<float>(nNumBatches * nBatchSize);

        // ===== 测试集评估 =====
        // 20260319 ZJH 切换模型到评估模式
        pModel->eval();
        int nTestCorrect = 0;  // 20260319 ZJH 测试集正确预测数
        int nTestBatches = testData.m_nSamples / nBatchSize;  // 20260319 ZJH 测试批次数
        for (int nBatch = 0; nBatch < nTestBatches; ++nBatch) {
            int nStart = nBatch * nBatchSize;
            auto batchImages = om::tensorSlice(testData.m_images, 0, nStart, nStart + nBatchSize);
            auto batchLabels = om::tensorSlice(testData.m_labels, 0, nStart, nStart + nBatchSize);
            batchImages = batchImages.contiguous();
            batchLabels = batchLabels.contiguous();

            // 20260319 ZJH 如果是 ResNet，reshape 为 [batch, 1, 28, 28]
            if (bUseResNet) {
                batchImages = om::tensorReshape(batchImages, {nBatchSize, 1, 28, 28});
            }

            // 20260319 ZJH 前向推理
            auto matLogits = pModel->forward(batchImages);

            // 20260319 ZJH 计算准确率
            auto vecPredicted = om::tensorArgmax(matLogits);
            auto vecActual = om::tensorArgmax(batchLabels);
            for (int i = 0; i < nBatchSize; ++i) {
                if (vecPredicted[static_cast<size_t>(i)] == vecActual[static_cast<size_t>(i)]) {
                    nTestCorrect++;
                }
            }
        }
        float fTestAcc = 100.0f * static_cast<float>(nTestCorrect) / static_cast<float>(nTestBatches * nBatchSize);

        // 20260319 ZJH 恢复训练模式
        pModel->train();

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
