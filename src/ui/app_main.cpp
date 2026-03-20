// 20260320 ZJH DeepForge Phase 4 — SDL3 + ImGui 桌面 GUI 应用程序
// 主界面：4 个工作台（训练/推理/数据管理/模型仓库）+ 状态栏 + 启动闪屏
// 使用 ImGui Docking 布局，ImPlot 实时绘制损失/准确率曲线
// 训练在独立 std::jthread 中执行，原子变量 + 互斥锁保证线程安全
// 20260320 ZJH 全部 UI 文本汉化为简体中文，启动时显示闪屏动画

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <implot.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <functional>

// 20260319 ZJH 导入 DeepForge 引擎模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;
import df.engine.linear;
import df.engine.activations;
import df.engine.conv;
import df.engine.resnet;
import df.engine.optimizer;
import df.engine.loss;
import df.engine.mnist;
import df.engine.serializer;
import df.hal.cpu_backend;

// ============================================================================
// 20260319 ZJH 训练状态结构体 — 在 UI 线程和训练线程之间共享
// ============================================================================
struct TrainingState {
    // 20260319 ZJH 原子标志，线程安全的读写
    std::atomic<bool> bRunning{false};        // 20260319 ZJH 训练是否正在运行
    std::atomic<bool> bPaused{false};         // 20260319 ZJH 训练是否暂停
    std::atomic<bool> bStopRequested{false};  // 20260319 ZJH 是否请求停止训练
    std::atomic<bool> bCompleted{false};      // 20260319 ZJH 训练是否完成
    std::atomic<int> nCurrentEpoch{0};        // 20260319 ZJH 当前训练轮数
    std::atomic<int> nTotalEpochs{10};        // 20260319 ZJH 总训练轮数
    std::atomic<int> nCurrentBatch{0};        // 20260319 ZJH 当前批次
    std::atomic<int> nTotalBatches{0};        // 20260319 ZJH 每轮总批次数
    std::atomic<float> fCurrentLoss{0.0f};    // 20260319 ZJH 当前批次损失
    std::atomic<float> fTrainAcc{0.0f};       // 20260319 ZJH 当前轮训练准确率
    std::atomic<float> fTestAcc{0.0f};        // 20260319 ZJH 当前轮测试准确率

    // 20260319 ZJH 互斥锁保护的历史数据，UI 线程读取时需加锁
    std::mutex mutex;
    std::vector<float> vecLossHistory;        // 20260319 ZJH 每轮平均损失历史
    std::vector<float> vecTrainAccHistory;    // 20260319 ZJH 每轮训练准确率历史
    std::vector<float> vecTestAccHistory;     // 20260319 ZJH 每轮测试准确率历史
    std::string strLog;                       // 20260319 ZJH 训练日志文本
    std::string strSavedModelPath;            // 20260319 ZJH 保存的模型路径

    // 20260319 ZJH 重置所有状态，准备新一轮训练
    void reset() {
        bRunning.store(false);
        bPaused.store(false);
        bStopRequested.store(false);
        bCompleted.store(false);
        nCurrentEpoch.store(0);
        nCurrentBatch.store(0);
        nTotalBatches.store(0);
        fCurrentLoss.store(0.0f);
        fTrainAcc.store(0.0f);
        fTestAcc.store(0.0f);
        std::lock_guard<std::mutex> lock(mutex);
        vecLossHistory.clear();
        vecTrainAccHistory.clear();
        vecTestAccHistory.clear();
        strLog.clear();
        strSavedModelPath.clear();
    }

    // 20260319 ZJH 追加一行日志（线程安全）
    void appendLog(const std::string& strLine) {
        std::lock_guard<std::mutex> lock(mutex);
        strLog += strLine + "\n";
    }
};

// ============================================================================
// 20260319 ZJH 推理状态结构体
// ============================================================================
struct InferenceState {
    bool bModelLoaded = false;                // 20260319 ZJH 是否已加载模型
    std::string strModelPath;                 // 20260319 ZJH 模型文件路径
    std::string strModelType;                 // 20260319 ZJH 模型类型（mlp / resnet18）
    std::string strImagePath;                 // 20260319 ZJH 待推理图像路径
    bool bHasResult = false;                  // 20260319 ZJH 是否有推理结果
    int nPredictedClass = -1;                 // 20260319 ZJH 预测类别
    float arrConfidence[10] = {};             // 20260319 ZJH 各类别置信度
    std::string strResultLog;                 // 20260319 ZJH 推理结果日志
    SDL_Texture* pImageTexture = nullptr;     // 20260319 ZJH 图像 SDL 纹理（用于 ImGui 显示）
    int nImageWidth = 0;                      // 20260319 ZJH 图像宽度
    int nImageHeight = 0;                     // 20260319 ZJH 图像高度

    // 20260319 ZJH 模型指针
    std::shared_ptr<df::Module> pModel;
};

// ============================================================================
// 20260319 ZJH 模型信息结构体（用于模型仓库）
// ============================================================================
struct ModelInfo {
    std::string strFilePath;                  // 20260319 ZJH 模型文件完整路径
    std::string strFileName;                  // 20260319 ZJH 模型文件名
    std::uintmax_t nFileSize = 0;             // 20260319 ZJH 文件大小（字节）
    std::string strLastModified;              // 20260319 ZJH 最后修改时间
};

// ============================================================================
// 20260319 ZJH 全局应用状态
// ============================================================================
struct AppState {
    int nActiveTab = 0;                       // 20260319 ZJH 当前激活的工作台标签页（0=训练，1=推理，2=数据，3=模型）
    TrainingState trainState;                 // 20260319 ZJH 训练状态
    InferenceState inferState;                // 20260319 ZJH 推理状态

    // 20260319 ZJH 训练超参数（UI 可调）
    int nSelectedModel = 0;                   // 20260319 ZJH 选中的模型索引（0=MLP, 1=ResNet-18）
    int nEpochs = 10;                         // 20260319 ZJH 训练轮数
    int nBatchSize = 64;                      // 20260319 ZJH 批次大小
    float fLearningRate = 0.01f;              // 20260319 ZJH 学习率
    int nSelectedOptimizer = 0;               // 20260319 ZJH 选中的优化器（0=SGD, 1=Adam）

    // 20260319 ZJH 训练线程
    std::unique_ptr<std::jthread> pTrainThread;

    // 20260319 ZJH 模型仓库
    std::vector<ModelInfo> vecModels;         // 20260319 ZJH 模型列表
    bool bModelsScanned = false;              // 20260319 ZJH 是否已扫描模型目录

    // 20260319 ZJH 数据集信息
    bool bMnistAvailable = false;             // 20260319 ZJH MNIST 数据是否可用
    int nMnistTrainSamples = 0;              // 20260319 ZJH 训练集样本数
    int nMnistTestSamples = 0;               // 20260319 ZJH 测试集样本数
    bool bDataChecked = false;                // 20260319 ZJH 是否已检查数据
};

// ============================================================================
// 20260319 ZJH 辅助函数前向声明
// ============================================================================
static void setupImGuiStyle();
static void drawMenuBar(AppState& state);
static void drawTrainingWorkbench(AppState& state);
static void drawInferenceWorkbench(AppState& state, SDL_Renderer* pRenderer);
static void drawDataManager(AppState& state);
static void drawModelRepository(AppState& state);
static void drawStatusBar(AppState& state);
static void startTraining(AppState& state);
static void scanModels(AppState& state);
static void checkDatasets(AppState& state);

// ============================================================================
// 20260319 ZJH 训练线程函数 — 在独立线程中运行完整训练循环
// ============================================================================
static void trainingThreadFunc(AppState& state) {
    auto& ts = state.trainState;  // 20260319 ZJH 训练状态引用
    ts.bRunning.store(true);
    ts.bCompleted.store(false);
    ts.appendLog("========================================");
    ts.appendLog("  DeepForge 训练开始");
    ts.appendLog("========================================");

    // 20260319 ZJH 读取 UI 超参数
    bool bUseResNet = (state.nSelectedModel == 1);  // 20260319 ZJH 是否使用 ResNet
    int nEpochs = state.nEpochs;                    // 20260319 ZJH 训练轮数
    int nBatchSize = state.nBatchSize;               // 20260319 ZJH 批次大小
    float fLearningRate = state.fLearningRate;       // 20260319 ZJH 学习率
    bool bUseAdam = (state.nSelectedOptimizer == 1); // 20260319 ZJH 是否使用 Adam
    const int nInputDim = 784;                       // 20260319 ZJH 输入维度（28x28）
    const int nHiddenDim = 128;                      // 20260319 ZJH MLP 隐藏层维度
    const int nOutputDim = 10;                       // 20260319 ZJH 输出维度（10 个类别）

    ts.nTotalEpochs.store(nEpochs);

    // 20260319 ZJH 配置信息写入日志
    if (bUseResNet) {
        ts.appendLog("  模型: ResNet-18 卷积神经网络");
    } else {
        ts.appendLog("  模型: MLP 全连接网络 (784→128→10)");
    }
    {
        char arrBuf[256];
        std::snprintf(arrBuf, sizeof(arrBuf), "  训练轮数: %d, 批次大小: %d, 学习率: %.4f, 优化器: %s",
                      nEpochs, nBatchSize, static_cast<double>(fLearningRate),
                      bUseAdam ? "Adam" : "SGD");
        ts.appendLog(arrBuf);
    }
    ts.appendLog("");

    // ===== 第一步：加载数据 =====
    // 20260319 ZJH MNIST 数据文件路径
    std::string strTrainImagesPath = "data/mnist/train-images-idx3-ubyte";
    std::string strTrainLabelsPath = "data/mnist/train-labels-idx1-ubyte";
    std::string strTestImagesPath  = "data/mnist/t10k-images-idx3-ubyte";
    std::string strTestLabelsPath  = "data/mnist/t10k-labels-idx1-ubyte";

    df::MnistDataset trainData;  // 20260319 ZJH 训练集
    df::MnistDataset testData;   // 20260319 ZJH 测试集

    try {
        ts.appendLog("正在加载 MNIST 训练数据...");
        trainData = df::loadMnist(strTrainImagesPath, strTrainLabelsPath);
        ts.appendLog("正在加载 MNIST 测试数据...");
        testData = df::loadMnist(strTestImagesPath, strTestLabelsPath);
        {
            char arrBuf[128];
            std::snprintf(arrBuf, sizeof(arrBuf), "已加载: 训练=%d, 测试=%d 样本",
                          trainData.m_nSamples, testData.m_nSamples);
            ts.appendLog(arrBuf);
        }
    } catch (const std::runtime_error&) {
        // 20260319 ZJH 数据文件缺失时切换到合成数据模式
        ts.appendLog("[警告] MNIST 数据不可用，使用合成数据");
        int nTrainSynth = 1000;   // 20260319 ZJH 合成训练样本数
        int nTestSynth = 200;     // 20260319 ZJH 合成测试样本数

        // 20260319 ZJH 生成合成分类数据的 lambda
        auto generateSynthetic = [&](int nSamples) -> df::MnistDataset {
            df::MnistDataset ds;
            ds.m_nSamples = nSamples;
            ds.m_images = df::Tensor::zeros({nSamples, nInputDim});
            ds.m_labels = df::Tensor::zeros({nSamples, nOutputDim});
            float* pImages = ds.m_images.mutableFloatDataPtr();
            float* pLabels = ds.m_labels.mutableFloatDataPtr();

            // 20260319 ZJH 类别 c 的样本在维度 [c*78, (c+1)*78) 有较高值
            for (int i = 0; i < nSamples; ++i) {
                int nClass = i % nOutputDim;
                for (int j = 0; j < nInputDim; ++j) {
                    pImages[i * nInputDim + j] = 0.1f;  // 20260319 ZJH 基底噪声
                }
                int nStart = nClass * (nInputDim / nOutputDim);
                int nEnd = (nClass + 1) * (nInputDim / nOutputDim);
                if (nEnd > nInputDim) nEnd = nInputDim;
                for (int j = nStart; j < nEnd; ++j) {
                    pImages[i * nInputDim + j] = 0.9f;  // 20260319 ZJH 类别特征区域
                }
                pLabels[i * nOutputDim + nClass] = 1.0f;  // 20260319 ZJH one-hot 标签
            }
            return ds;
        };

        trainData = generateSynthetic(nTrainSynth);
        testData = generateSynthetic(nTestSynth);
        {
            char arrBuf[128];
            std::snprintf(arrBuf, sizeof(arrBuf), "合成数据: 训练=%d, 测试=%d 样本",
                          trainData.m_nSamples, testData.m_nSamples);
            ts.appendLog(arrBuf);
        }
    }
    ts.appendLog("");

    // 20260319 ZJH 检查是否请求停止
    if (ts.bStopRequested.load()) {
        ts.appendLog("训练已被用户停止。");
        ts.bRunning.store(false);
        return;
    }

    // ===== 第二步：构建模型 =====
    // 20260319 ZJH 根据模型类型创建网络
    std::shared_ptr<df::Module> pModel;
    std::shared_ptr<df::ResNet18> pResNet;
    std::shared_ptr<df::Sequential> pMLP;

    if (bUseResNet) {
        ts.appendLog("正在构建 ResNet-18 模型...");
        pResNet = std::make_shared<df::ResNet18>(nOutputDim);
        pModel = pResNet;
    } else {
        ts.appendLog("正在构建 MLP 模型...");
        pMLP = std::make_shared<df::Sequential>();
        pMLP->add(std::make_shared<df::Linear>(nInputDim, nHiddenDim));
        pMLP->add(std::make_shared<df::ReLU>());
        pMLP->add(std::make_shared<df::Linear>(nHiddenDim, nOutputDim));
        pModel = pMLP;
    }

    // 20260319 ZJH 获取模型参数并输出信息
    auto vecParams = pModel->parameters();
    int nTotalParams = 0;
    for (auto* pParam : vecParams) {
        nTotalParams += pParam->numel();
    }
    {
        char arrBuf[128];
        std::snprintf(arrBuf, sizeof(arrBuf), "模型: %d 个张量, %d 个参数",
                      static_cast<int>(vecParams.size()), nTotalParams);
        ts.appendLog(arrBuf);
    }
    ts.appendLog("");

    // ===== 第三步：创建优化器和损失函数 =====
    std::unique_ptr<df::Adam> pAdamOpt;
    std::unique_ptr<df::SGD> pSgdOpt;

    if (bUseAdam) {
        pAdamOpt = std::make_unique<df::Adam>(vecParams, fLearningRate);
        ts.appendLog("优化器: Adam");
    } else {
        pSgdOpt = std::make_unique<df::SGD>(vecParams, fLearningRate);
        ts.appendLog("优化器: SGD");
    }

    df::CrossEntropyLoss criterion;  // 20260319 ZJH 交叉熵损失函数
    ts.appendLog("损失函数: CrossEntropyLoss");
    ts.appendLog("");

    // ===== 第四步：训练循环 =====
    int nNumBatches = trainData.m_nSamples / nBatchSize;  // 20260319 ZJH 整除批次数
    ts.nTotalBatches.store(nNumBatches);

    for (int nEpoch = 0; nEpoch < nEpochs; ++nEpoch) {
        // 20260319 ZJH 检查停止请求
        if (ts.bStopRequested.load()) {
            ts.appendLog("训练已被用户停止。");
            break;
        }

        // 20260319 ZJH 暂停等待
        while (ts.bPaused.load() && !ts.bStopRequested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (ts.bStopRequested.load()) {
            ts.appendLog("训练已被用户停止。");
            break;
        }

        ts.nCurrentEpoch.store(nEpoch + 1);
        auto timeStart = std::chrono::steady_clock::now();

        float fEpochLoss = 0.0f;  // 20260319 ZJH 累积损失
        int nCorrect = 0;          // 20260319 ZJH 正确预测计数

        for (int nBatch = 0; nBatch < nNumBatches; ++nBatch) {
            // 20260319 ZJH 检查停止请求
            if (ts.bStopRequested.load()) break;

            // 20260319 ZJH 暂停等待
            while (ts.bPaused.load() && !ts.bStopRequested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (ts.bStopRequested.load()) break;

            ts.nCurrentBatch.store(nBatch + 1);

            // 20260319 ZJH 提取当前批次数据
            int nStart = nBatch * nBatchSize;
            auto batchImages = df::tensorSlice(trainData.m_images, 0, nStart, nStart + nBatchSize);
            auto batchLabels = df::tensorSlice(trainData.m_labels, 0, nStart, nStart + nBatchSize);
            batchImages = batchImages.contiguous();
            batchLabels = batchLabels.contiguous();

            // 20260319 ZJH ResNet 需要 NCHW 格式
            if (bUseResNet) {
                batchImages = df::tensorReshape(batchImages, {nBatchSize, 1, 28, 28});
            }

            // 20260319 ZJH 前向传播
            auto matLogits = pModel->forward(batchImages);
            auto loss = criterion.forward(matLogits, batchLabels);
            fEpochLoss += loss.item();

            // 20260319 ZJH 更新实时损失
            ts.fCurrentLoss.store(loss.item());

            // 20260319 ZJH 计算训练准确率
            auto vecPredicted = df::tensorArgmax(matLogits);
            auto vecActual = df::tensorArgmax(batchLabels);
            for (int i = 0; i < nBatchSize; ++i) {
                if (vecPredicted[static_cast<size_t>(i)] == vecActual[static_cast<size_t>(i)]) {
                    nCorrect++;
                }
            }

            // 20260319 ZJH 反向传播 + 参数更新
            if (bUseAdam) {
                pAdamOpt->zeroGrad();
                df::tensorBackward(loss);
                pAdamOpt->step();
            } else {
                pSgdOpt->zeroGrad();
                df::tensorBackward(loss);
                pSgdOpt->step();
            }
        }

        if (ts.bStopRequested.load()) break;

        // 20260319 ZJH 计算当前轮的平均损失和训练准确率
        float fAvgLoss = fEpochLoss / static_cast<float>(nNumBatches);
        float fTrainAcc = 100.0f * static_cast<float>(nCorrect) / static_cast<float>(nNumBatches * nBatchSize);

        // ===== 测试集评估 =====
        pModel->eval();
        int nTestCorrect = 0;
        int nTestBatches = testData.m_nSamples / nBatchSize;
        for (int nBatch = 0; nBatch < nTestBatches; ++nBatch) {
            int nStartTest = nBatch * nBatchSize;
            auto batchImages = df::tensorSlice(testData.m_images, 0, nStartTest, nStartTest + nBatchSize);
            auto batchLabels = df::tensorSlice(testData.m_labels, 0, nStartTest, nStartTest + nBatchSize);
            batchImages = batchImages.contiguous();
            batchLabels = batchLabels.contiguous();

            if (bUseResNet) {
                batchImages = df::tensorReshape(batchImages, {nBatchSize, 1, 28, 28});
            }

            auto matLogits = pModel->forward(batchImages);
            auto vecPredicted = df::tensorArgmax(matLogits);
            auto vecActual = df::tensorArgmax(batchLabels);
            for (int i = 0; i < nBatchSize; ++i) {
                if (vecPredicted[static_cast<size_t>(i)] == vecActual[static_cast<size_t>(i)]) {
                    nTestCorrect++;
                }
            }
        }
        float fTestAcc = 100.0f * static_cast<float>(nTestCorrect) / static_cast<float>(nTestBatches * nBatchSize);
        pModel->train();

        // 20260319 ZJH 更新原子变量
        ts.fTrainAcc.store(fTrainAcc);
        ts.fTestAcc.store(fTestAcc);

        auto timeEnd = std::chrono::steady_clock::now();
        auto nDuration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();

        // 20260319 ZJH 更新历史数据（加锁）
        {
            std::lock_guard<std::mutex> lock(ts.mutex);
            ts.vecLossHistory.push_back(fAvgLoss);
            ts.vecTrainAccHistory.push_back(fTrainAcc);
            ts.vecTestAccHistory.push_back(fTestAcc);
        }

        // 20260319 ZJH 写入日志
        {
            char arrBuf[256];
            std::snprintf(arrBuf, sizeof(arrBuf),
                          "轮次 %d/%d | 损失: %.4f | 训练: %.2f%% | 测试: %.2f%% | %lldms",
                          nEpoch + 1, nEpochs,
                          static_cast<double>(fAvgLoss),
                          static_cast<double>(fTrainAcc),
                          static_cast<double>(fTestAcc),
                          static_cast<long long>(nDuration));
            ts.appendLog(arrBuf);
        }
    }

    // ===== 第五步：保存模型 =====
    if (!ts.bStopRequested.load()) {
        try {
            // 20260319 ZJH 确保模型保存目录存在
            std::filesystem::create_directories("data/models");
            std::string strModelName = bUseResNet ? "resnet18" : "mlp";
            // 20260319 ZJH 使用时间戳命名模型文件
            auto now = std::chrono::system_clock::now();
            auto nTime = std::chrono::system_clock::to_time_t(now);
            char arrTimeBuf[64];
            struct tm tmLocal;
            localtime_s(&tmLocal, &nTime);  // 20260319 ZJH MSVC 安全版本
            std::strftime(arrTimeBuf, sizeof(arrTimeBuf), "%Y%m%d_%H%M%S", &tmLocal);
            std::string strSavePath = "data/models/" + strModelName + "_" + arrTimeBuf + ".dfm";

            df::ModelSerializer::save(*pModel, strSavePath);
            ts.appendLog("");
            ts.appendLog("模型已保存：" + strSavePath);
            {
                std::lock_guard<std::mutex> lock(ts.mutex);
                ts.strSavedModelPath = strSavePath;
            }
        } catch (const std::exception& e) {
            ts.appendLog(std::string("模型保存错误：") + e.what());
        }
    }

    ts.appendLog("");
    ts.appendLog("训练完成！");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260319 ZJH 设置工业深色主题
// ============================================================================
static void setupImGuiStyle() {
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;       // 20260319 ZJH 窗口圆角半径
    style.FrameRounding = 2.0f;        // 20260319 ZJH 控件边框圆角
    style.GrabRounding = 2.0f;         // 20260319 ZJH 滑块手柄圆角
    style.TabRounding = 3.0f;          // 20260319 ZJH 标签页圆角
    style.WindowPadding = ImVec2(8, 8);  // 20260319 ZJH 窗口内边距
    style.FramePadding = ImVec2(5, 4);   // 20260319 ZJH 控件内边距
    style.ItemSpacing = ImVec2(8, 5);    // 20260319 ZJH 控件间距

    ImVec4* colors = style.Colors;
    // 20260319 ZJH 背景色：深灰色系
    colors[ImGuiCol_WindowBg]        = ImVec4(0.12f, 0.12f, 0.14f, 1.0f);
    colors[ImGuiCol_ChildBg]         = ImVec4(0.10f, 0.10f, 0.12f, 1.0f);
    colors[ImGuiCol_PopupBg]         = ImVec4(0.14f, 0.14f, 0.16f, 0.96f);
    // 20260319 ZJH 边框色：微亮边框
    colors[ImGuiCol_Border]          = ImVec4(0.28f, 0.28f, 0.32f, 1.0f);
    // 20260319 ZJH 控件色：蓝色主色调
    colors[ImGuiCol_FrameBg]         = ImVec4(0.16f, 0.16f, 0.20f, 1.0f);
    colors[ImGuiCol_FrameBgHovered]  = ImVec4(0.22f, 0.22f, 0.28f, 1.0f);
    colors[ImGuiCol_FrameBgActive]   = ImVec4(0.26f, 0.26f, 0.34f, 1.0f);
    // 20260319 ZJH 标题栏
    colors[ImGuiCol_TitleBg]         = ImVec4(0.08f, 0.08f, 0.10f, 1.0f);
    colors[ImGuiCol_TitleBgActive]   = ImVec4(0.12f, 0.14f, 0.20f, 1.0f);
    // 20260319 ZJH 标签页
    colors[ImGuiCol_Tab]             = ImVec4(0.14f, 0.14f, 0.18f, 1.0f);
    colors[ImGuiCol_TabHovered]      = ImVec4(0.24f, 0.36f, 0.58f, 1.0f);
    colors[ImGuiCol_TabSelected]     = ImVec4(0.20f, 0.30f, 0.50f, 1.0f);
    // 20260319 ZJH 按钮
    colors[ImGuiCol_Button]          = ImVec4(0.20f, 0.30f, 0.48f, 1.0f);
    colors[ImGuiCol_ButtonHovered]   = ImVec4(0.26f, 0.38f, 0.58f, 1.0f);
    colors[ImGuiCol_ButtonActive]    = ImVec4(0.30f, 0.42f, 0.64f, 1.0f);
    // 20260319 ZJH 头部（表头等）
    colors[ImGuiCol_Header]          = ImVec4(0.20f, 0.28f, 0.44f, 1.0f);
    colors[ImGuiCol_HeaderHovered]   = ImVec4(0.26f, 0.36f, 0.54f, 1.0f);
    colors[ImGuiCol_HeaderActive]    = ImVec4(0.30f, 0.40f, 0.60f, 1.0f);
    // 20260319 ZJH 分隔线
    colors[ImGuiCol_Separator]       = ImVec4(0.28f, 0.28f, 0.32f, 1.0f);
    // 20260319 ZJH 文本
    colors[ImGuiCol_Text]            = ImVec4(0.90f, 0.90f, 0.92f, 1.0f);
    colors[ImGuiCol_TextDisabled]    = ImVec4(0.50f, 0.50f, 0.54f, 1.0f);
    // 20260319 ZJH 进度条
    colors[ImGuiCol_PlotHistogram]   = ImVec4(0.28f, 0.56f, 0.90f, 1.0f);
    // 20260319 ZJH 选中色
    colors[ImGuiCol_CheckMark]       = ImVec4(0.40f, 0.70f, 1.0f, 1.0f);
    colors[ImGuiCol_SliderGrab]      = ImVec4(0.30f, 0.50f, 0.80f, 1.0f);
    colors[ImGuiCol_SliderGrabActive]= ImVec4(0.36f, 0.58f, 0.90f, 1.0f);
}

// ============================================================================
// 20260319 ZJH 启动训练
// ============================================================================
static void startTraining(AppState& state) {
    // 20260319 ZJH 如果已有训练线程在运行，不重复启动
    if (state.trainState.bRunning.load()) return;

    // 20260319 ZJH 等待旧线程结束并清理
    if (state.pTrainThread) {
        state.pTrainThread->request_stop();
        state.pTrainThread->join();
        state.pTrainThread.reset();
    }

    // 20260319 ZJH 重置训练状态
    state.trainState.reset();

    // 20260319 ZJH 启动训练线程
    state.pTrainThread = std::make_unique<std::jthread>([&state](std::stop_token) {
        trainingThreadFunc(state);
    });
}

// ============================================================================
// 20260319 ZJH 扫描模型目录
// ============================================================================
static void scanModels(AppState& state) {
    state.vecModels.clear();
    std::string strModelDir = "data/models";

    // 20260319 ZJH 确保目录存在
    if (!std::filesystem::exists(strModelDir)) {
        std::filesystem::create_directories(strModelDir);
        return;
    }

    // 20260319 ZJH 遍历目录中的 .dfm 文件
    for (auto& entry : std::filesystem::directory_iterator(strModelDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dfm") {
            ModelInfo info;
            info.strFilePath = entry.path().string();
            info.strFileName = entry.path().filename().string();
            info.nFileSize = entry.file_size();

            // 20260319 ZJH 格式化最后修改时间
            auto ftime = entry.last_write_time();
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            char arrTimeBuf[64];
            auto nTimeT = std::chrono::system_clock::to_time_t(sctp);
            struct tm tmLocal;
            localtime_s(&tmLocal, &nTimeT);  // 20260319 ZJH MSVC 安全版本
            std::strftime(arrTimeBuf, sizeof(arrTimeBuf), "%Y-%m-%d %H:%M:%S", &tmLocal);
            info.strLastModified = arrTimeBuf;

            state.vecModels.push_back(info);
        }
    }

    // 20260319 ZJH 按文件名排序（最新的在前）
    std::sort(state.vecModels.begin(), state.vecModels.end(),
              [](const ModelInfo& a, const ModelInfo& b) {
                  return a.strFileName > b.strFileName;  // 20260319 ZJH 降序排列
              });

    state.bModelsScanned = true;
}

// ============================================================================
// 20260319 ZJH 检查可用数据集
// ============================================================================
static void checkDatasets(AppState& state) {
    // 20260319 ZJH 检查 MNIST 数据文件是否存在
    bool bTrainImages = std::filesystem::exists("data/mnist/train-images-idx3-ubyte");
    bool bTrainLabels = std::filesystem::exists("data/mnist/train-labels-idx1-ubyte");
    bool bTestImages  = std::filesystem::exists("data/mnist/t10k-images-idx3-ubyte");
    bool bTestLabels  = std::filesystem::exists("data/mnist/t10k-labels-idx1-ubyte");

    state.bMnistAvailable = bTrainImages && bTrainLabels && bTestImages && bTestLabels;

    if (state.bMnistAvailable) {
        // 20260319 ZJH 读取样本数量（从 IDX 文件头）
        auto readSampleCount = [](const std::string& strPath) -> int {
            std::ifstream ifs(strPath, std::ios::binary);
            if (!ifs.is_open()) return 0;
            // 20260319 ZJH IDX 格式：前 4 字节魔数，接下来 4 字节为样本数（大端序）
            unsigned char arrHeader[8];
            ifs.read(reinterpret_cast<char*>(arrHeader), 8);
            return (arrHeader[4] << 24) | (arrHeader[5] << 16) | (arrHeader[6] << 8) | arrHeader[7];
        };
        state.nMnistTrainSamples = readSampleCount("data/mnist/train-images-idx3-ubyte");
        state.nMnistTestSamples = readSampleCount("data/mnist/t10k-images-idx3-ubyte");
    } else {
        state.nMnistTrainSamples = 0;
        state.nMnistTestSamples = 0;
    }

    state.bDataChecked = true;
}

// ============================================================================
// 20260319 ZJH 绘制训练工作台
// ============================================================================
static void drawTrainingWorkbench(AppState& state) {
    auto& ts = state.trainState;

    // 20260319 ZJH 左侧：控制面板
    ImGui::BeginChild("TrainControls", ImVec2(320, 0), ImGuiChildFlags_Borders);
    {
        // 20260320 ZJH 训练配置标题
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "训练配置");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 模型选择下拉框
        const char* arrModels[] = {"MLP 全连接网络 (784→128→10)", "ResNet-18 卷积神经网络"};
        bool bDisabled = ts.bRunning.load();  // 20260319 ZJH 训练中禁用配置修改
        if (bDisabled) ImGui::BeginDisabled();
        ImGui::Combo("模型", &state.nSelectedModel, arrModels, 2);

        // 20260320 ZJH 优化器选择
        const char* arrOptimizers[] = {"SGD", "Adam"};
        ImGui::Combo("优化器", &state.nSelectedOptimizer, arrOptimizers, 2);

        ImGui::Spacing();
        // 20260320 ZJH 超参数标题
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "超参数");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 训练轮数滑块
        ImGui::SliderInt("训练轮数", &state.nEpochs, 1, 100);
        // 20260320 ZJH 批次大小滑块
        ImGui::SliderInt("批次大小", &state.nBatchSize, 8, 256);
        // 20260320 ZJH 学习率输入
        ImGui::InputFloat("学习率", &state.fLearningRate, 0.001f, 0.01f, "%.4f");
        // 20260319 ZJH 限制学习率范围
        if (state.fLearningRate < 0.0001f) state.fLearningRate = 0.0001f;
        if (state.fLearningRate > 1.0f) state.fLearningRate = 1.0f;

        if (bDisabled) ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260319 ZJH 训练控制按钮
        float fButtonWidth = 90.0f;

        if (!ts.bRunning.load()) {
            // 20260320 ZJH 训练未运行时显示 开始训练 按钮
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.65f, 0.20f, 1.0f));
            if (ImGui::Button("开始训练", ImVec2(fButtonWidth, 30))) {
                startTraining(state);
            }
            ImGui::PopStyleColor(2);
        } else {
            // 20260320 ZJH 训练运行中显示 暂停/恢复 和 停止 按钮
            if (ts.bPaused.load()) {
                // 20260320 ZJH 暂停中，显示 恢复
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.45f, 0.10f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.55f, 0.15f, 1.0f));
                if (ImGui::Button("恢复", ImVec2(fButtonWidth, 30))) {
                    ts.bPaused.store(false);
                }
                ImGui::PopStyleColor(2);
            } else {
                // 20260320 ZJH 运行中，显示 暂停
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.45f, 0.10f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.55f, 0.15f, 1.0f));
                if (ImGui::Button("暂停", ImVec2(fButtonWidth, 30))) {
                    ts.bPaused.store(true);
                }
                ImGui::PopStyleColor(2);
            }

            ImGui::SameLine();

            // 20260320 ZJH 停止 按钮
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.65f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f, 0.20f, 0.20f, 1.0f));
            if (ImGui::Button("停止", ImVec2(fButtonWidth, 30))) {
                ts.bStopRequested.store(true);
                ts.bPaused.store(false);  // 20260319 ZJH 取消暂停以让线程响应停止
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 训练进度信息
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "训练进度");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 当前轮次进度
        int nCurEpoch = ts.nCurrentEpoch.load();
        int nTotalEpochs = ts.nTotalEpochs.load();
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "轮次: %d / %d", nCurEpoch, nTotalEpochs);
            ImGui::Text("%s", arrBuf);
        }

        // 20260319 ZJH 进度条
        float fProgress = (nTotalEpochs > 0) ?
            static_cast<float>(nCurEpoch) / static_cast<float>(nTotalEpochs) : 0.0f;
        ImGui::ProgressBar(fProgress, ImVec2(-1, 0));

        // 20260319 ZJH 批次进度
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "批次: %d / %d",
                          ts.nCurrentBatch.load(), ts.nTotalBatches.load());
            ImGui::Text("%s", arrBuf);
        }

        ImGui::Spacing();
        // 20260319 ZJH 当前指标
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "损失: %.4f", static_cast<double>(ts.fCurrentLoss.load()));
            ImGui::Text("%s", arrBuf);
        }
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "训练准确率: %.2f%%", static_cast<double>(ts.fTrainAcc.load()));
            ImGui::Text("%s", arrBuf);
        }
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "测试准确率: %.2f%%", static_cast<double>(ts.fTestAcc.load()));
            ImGui::Text("%s", arrBuf);
        }

        // 20260319 ZJH 训练状态指示
        ImGui::Spacing();
        if (ts.bRunning.load()) {
            if (ts.bPaused.load()) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "状态: 已暂停");
            } else {
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "状态: 训练中...");
            }
        } else if (ts.bCompleted.load()) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "状态: 已完成");
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "状态: 空闲");
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // 20260319 ZJH 右侧：图表和日志
    ImGui::BeginChild("TrainCharts", ImVec2(0, 0));
    {
        // 20260319 ZJH 上半部分：损失和准确率图表
        float fChartHeight = ImGui::GetContentRegionAvail().y * 0.55f;

        ImGui::BeginChild("Charts", ImVec2(0, fChartHeight));
        {
            float fHalfWidth = (ImGui::GetContentRegionAvail().x - 8.0f) * 0.5f;

            // 20260319 ZJH 损失曲线图
            ImGui::BeginChild("LossChart", ImVec2(fHalfWidth, 0));
            {
                std::lock_guard<std::mutex> lock(ts.mutex);
                // 20260320 ZJH 损失曲线图
                if (ImPlot::BeginPlot("损失曲线", ImVec2(-1, -1))) {
                    ImPlot::SetupAxes("轮次", "损失");
                    if (!ts.vecLossHistory.empty()) {
                        // 20260320 ZJH 绘制损失折线
                        ImPlot::PlotLine("训练损失",
                                         ts.vecLossHistory.data(),
                                         static_cast<int>(ts.vecLossHistory.size()));
                    }
                    ImPlot::EndPlot();
                }
            }
            ImGui::EndChild();

            ImGui::SameLine();

            // 20260319 ZJH 准确率曲线图
            ImGui::BeginChild("AccChart", ImVec2(0, 0));
            {
                std::lock_guard<std::mutex> lock(ts.mutex);
                // 20260320 ZJH 准确率曲线图
                if (ImPlot::BeginPlot("准确率曲线", ImVec2(-1, -1))) {
                    ImPlot::SetupAxes("轮次", "准确率 (%)");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 105, ImPlotCond_Always);
                    if (!ts.vecTrainAccHistory.empty()) {
                        // 20260320 ZJH 绘制训练准确率
                        ImPlot::PlotLine("训练准确率",
                                         ts.vecTrainAccHistory.data(),
                                         static_cast<int>(ts.vecTrainAccHistory.size()));
                    }
                    if (!ts.vecTestAccHistory.empty()) {
                        // 20260320 ZJH 绘制测试准确率
                        ImPlot::PlotLine("测试准确率",
                                         ts.vecTestAccHistory.data(),
                                         static_cast<int>(ts.vecTestAccHistory.size()));
                    }
                    ImPlot::EndPlot();
                }
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();

        // 20260320 ZJH 下半部分：训练日志
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "训练日志");
        ImGui::Separator();
        ImGui::BeginChild("TrainLog", ImVec2(0, 0), ImGuiChildFlags_Borders);
        {
            std::lock_guard<std::mutex> lock(ts.mutex);
            // 20260319 ZJH 使用 TextUnformatted 高效显示大段文本
            ImGui::TextUnformatted(ts.strLog.c_str());
            // 20260319 ZJH 自动滚动到底部（当有新内容时）
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) {
                ImGui::SetScrollHereY(1.0f);
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260319 ZJH 绘制推理工作台
// ============================================================================
static void drawInferenceWorkbench(AppState& state, SDL_Renderer* pRenderer) {
    auto& is = state.inferState;

    // 20260319 ZJH 左侧：控制面板
    ImGui::BeginChild("InferControls", ImVec2(320, 0), ImGuiChildFlags_Borders);
    {
        // 20260320 ZJH 模型加载区域
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "模型加载");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 模型路径输入
        static char arrModelPath[512] = "";
        ImGui::InputText("模型路径", arrModelPath, sizeof(arrModelPath));

        // 20260320 ZJH 模型类型选择（加载时需要匹配模型结构）
        static int nInferModelType = 0;
        const char* arrModelTypes[] = {"MLP", "ResNet-18"};
        ImGui::Combo("模型类型", &nInferModelType, arrModelTypes, 2);

        // 20260320 ZJH 加载模型按钮
        if (ImGui::Button("加载模型", ImVec2(-1, 30))) {
            std::string strPath(arrModelPath);
            if (!strPath.empty() && std::filesystem::exists(strPath)) {
                try {
                    // 20260319 ZJH 根据类型创建模型并加载权重
                    if (nInferModelType == 1) {
                        auto pResNet = std::make_shared<df::ResNet18>(10);
                        df::ModelSerializer::load(*pResNet, strPath);
                        pResNet->eval();
                        is.pModel = pResNet;
                        is.strModelType = "ResNet-18";
                    } else {
                        auto pMLP = std::make_shared<df::Sequential>();
                        pMLP->add(std::make_shared<df::Linear>(784, 128));
                        pMLP->add(std::make_shared<df::ReLU>());
                        pMLP->add(std::make_shared<df::Linear>(128, 10));
                        df::ModelSerializer::load(*pMLP, strPath);
                        pMLP->eval();
                        is.pModel = pMLP;
                        is.strModelType = "MLP";
                    }
                    is.bModelLoaded = true;
                    is.strModelPath = strPath;
                    is.strResultLog = "模型已加载: " + strPath;
                } catch (const std::exception& e) {
                    is.bModelLoaded = false;
                    is.strResultLog = std::string("加载错误: ") + e.what();
                }
            } else {
                is.strResultLog = "路径无效或文件不存在。";
            }
        }

        // 20260320 ZJH 显示模型加载状态
        if (is.bModelLoaded) {
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "模型: 已加载 (%s)", is.strModelType.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "未加载模型");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 图像输入区域
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "图像输入");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 图像路径输入
        static char arrImagePath[512] = "";
        ImGui::InputText("图像路径", arrImagePath, sizeof(arrImagePath));

        // 20260320 ZJH 加载图像按钮
        if (ImGui::Button("加载图像", ImVec2(-1, 30))) {
            std::string strImgPath(arrImagePath);
            if (!strImgPath.empty() && std::filesystem::exists(strImgPath)) {
                // 20260319 ZJH 使用 stb_image 加载图像
                int nWidth = 0, nHeight = 0, nChannels = 0;
                unsigned char* pPixels = stbi_load(strImgPath.c_str(), &nWidth, &nHeight, &nChannels, 4);
                if (pPixels) {
                    // 20260319 ZJH 释放旧纹理
                    if (is.pImageTexture) {
                        SDL_DestroyTexture(is.pImageTexture);
                        is.pImageTexture = nullptr;
                    }
                    // 20260319 ZJH 创建 SDL 纹理
                    is.pImageTexture = SDL_CreateTexture(pRenderer,
                                                         SDL_PIXELFORMAT_RGBA32,
                                                         SDL_TEXTUREACCESS_STATIC,
                                                         nWidth, nHeight);
                    if (is.pImageTexture) {
                        SDL_UpdateTexture(is.pImageTexture, nullptr, pPixels, nWidth * 4);
                        is.nImageWidth = nWidth;
                        is.nImageHeight = nHeight;
                        is.strImagePath = strImgPath;
                        is.strResultLog = "图像已加载: " + std::to_string(nWidth) + "x" + std::to_string(nHeight);
                    }
                    stbi_image_free(pPixels);
                } else {
                    is.strResultLog = "图像加载失败。";
                }
            } else {
                is.strResultLog = "图像路径无效。";
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260319 ZJH 运行推理按钮
        bool bCanInfer = is.bModelLoaded;
        if (!bCanInfer) ImGui::BeginDisabled();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.65f, 0.20f, 1.0f));
        if (ImGui::Button("开始推理", ImVec2(-1, 35))) {
            // 20260319 ZJH 执行推理
            try {
                // 20260319 ZJH 如果有图像，转为 28x28 灰度后推理
                // 如果没有图像，用合成数据演示
                df::Tensor inputTensor = df::Tensor::zeros({1, 784});
                float* pInput = inputTensor.mutableFloatDataPtr();

                if (is.pImageTexture && !is.strImagePath.empty()) {
                    // 20260319 ZJH 重新加载图像为灰度 28x28
                    int nW = 0, nH = 0, nC = 0;
                    unsigned char* pGray = stbi_load(is.strImagePath.c_str(), &nW, &nH, &nC, 1);
                    if (pGray) {
                        // 20260319 ZJH 简单最近邻缩放到 28x28
                        for (int y = 0; y < 28; ++y) {
                            for (int x = 0; x < 28; ++x) {
                                int nSrcX = x * nW / 28;
                                int nSrcY = y * nH / 28;
                                pInput[y * 28 + x] = static_cast<float>(pGray[nSrcY * nW + nSrcX]) / 255.0f;
                            }
                        }
                        stbi_image_free(pGray);
                    }
                } else {
                    // 20260319 ZJH 无图像时使用合成输入（类别 3 的特征）
                    for (int j = 0; j < 784; ++j) pInput[j] = 0.1f;
                    for (int j = 3 * 78; j < 4 * 78; ++j) pInput[j] = 0.9f;
                }

                // 20260319 ZJH ResNet 需要 NCHW 格式
                if (is.strModelType == "ResNet-18") {
                    inputTensor = df::tensorReshape(inputTensor, {1, 1, 28, 28});
                }

                // 20260319 ZJH 前向传播
                auto output = is.pModel->forward(inputTensor);
                output = output.contiguous();

                // 20260319 ZJH 计算 softmax 置信度
                const float* pOut = output.floatDataPtr();
                float fMaxVal = pOut[0];
                for (int i = 1; i < 10; ++i) {
                    if (pOut[i] > fMaxVal) fMaxVal = pOut[i];
                }
                float fSum = 0.0f;
                for (int i = 0; i < 10; ++i) {
                    is.arrConfidence[i] = std::exp(pOut[i] - fMaxVal);
                    fSum += is.arrConfidence[i];
                }
                for (int i = 0; i < 10; ++i) {
                    is.arrConfidence[i] /= fSum;
                }

                // 20260319 ZJH 找到最大置信度的类别
                is.nPredictedClass = 0;
                for (int i = 1; i < 10; ++i) {
                    if (is.arrConfidence[i] > is.arrConfidence[is.nPredictedClass]) {
                        is.nPredictedClass = i;
                    }
                }

                is.bHasResult = true;
                {
                    char arrBuf[128];
                    std::snprintf(arrBuf, sizeof(arrBuf), "预测类别: %d (%.2f%% 置信度)",
                                  is.nPredictedClass,
                                  static_cast<double>(is.arrConfidence[is.nPredictedClass] * 100.0f));
                    is.strResultLog = arrBuf;
                }
            } catch (const std::exception& e) {
                is.strResultLog = std::string("推理错误: ") + e.what();
                is.bHasResult = false;
            }
        }
        ImGui::PopStyleColor(2);

        if (!bCanInfer) ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260319 ZJH 结果日志
        ImGui::TextWrapped("%s", is.strResultLog.c_str());
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // 20260319 ZJH 右侧：图像显示和结果
    ImGui::BeginChild("InferResults", ImVec2(0, 0));
    {
        // 20260319 ZJH 图像预览区域
        // 20260320 ZJH 图像预览区域
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "图像预览");
        ImGui::Separator();

        float fPreviewHeight = ImGui::GetContentRegionAvail().y * 0.5f;
        ImGui::BeginChild("ImagePreview", ImVec2(0, fPreviewHeight), ImGuiChildFlags_Borders);
        {
            if (is.pImageTexture) {
                // 20260319 ZJH 计算等比缩放显示尺寸
                float fAvailW = ImGui::GetContentRegionAvail().x;
                float fAvailH = ImGui::GetContentRegionAvail().y;
                float fScale = std::min(fAvailW / static_cast<float>(is.nImageWidth),
                                        fAvailH / static_cast<float>(is.nImageHeight));
                float fDispW = static_cast<float>(is.nImageWidth) * fScale;
                float fDispH = static_cast<float>(is.nImageHeight) * fScale;

                // 20260319 ZJH 居中显示
                float fOffsetX = (fAvailW - fDispW) * 0.5f;
                float fOffsetY = (fAvailH - fDispH) * 0.5f;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + fOffsetX);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + fOffsetY);
                ImGui::Image(static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(is.pImageTexture)),
                             ImVec2(fDispW, fDispH));
            } else {
                // 20260319 ZJH 无图像时显示占位文本
                float fAvailH = ImGui::GetContentRegionAvail().y;
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + fAvailH * 0.4f);
                ImGui::TextDisabled("    未加载图像。请在上方输入文件路径。");
            }
        }
        ImGui::EndChild();

        // 20260319 ZJH 推理结果
        // 20260320 ZJH 推理结果区域
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "推理结果");
        ImGui::Separator();

        ImGui::BeginChild("InferResultPanel", ImVec2(0, 0), ImGuiChildFlags_Borders);
        {
            if (is.bHasResult) {
                // 20260319 ZJH 显示预测类别
                ImGui::Text("预测类别: %d", is.nPredictedClass);
                ImGui::Text("置信度: %.2f%%",
                            static_cast<double>(is.arrConfidence[is.nPredictedClass] * 100.0f));
                ImGui::Spacing();

                // 20260319 ZJH 置信度条形图
                // 20260320 ZJH 置信度分布条形图
                if (ImPlot::BeginPlot("置信度分布", ImVec2(-1, -1))) {
                    ImPlot::SetupAxes("类别", "置信度");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.05, ImPlotCond_Always);

                    // 20260319 ZJH 绘制条形图
                    double arrPositions[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
                    double arrValues[10];
                    for (int i = 0; i < 10; ++i) {
                        arrValues[i] = static_cast<double>(is.arrConfidence[i]);
                    }
                    ImPlot::PlotBars("置信度", arrPositions, arrValues, 10, 0.6);
                    ImPlot::EndPlot();
                }
            } else {
                float fAvailH = ImGui::GetContentRegionAvail().y;
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + fAvailH * 0.4f);
                ImGui::TextDisabled("    执行推理后查看结果。");
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260319 ZJH 绘制数据管理工作台
// ============================================================================
static void drawDataManager(AppState& state) {
    // 20260319 ZJH 首次进入时检查数据集
    if (!state.bDataChecked) {
        checkDatasets(state);
    }

    ImGui::BeginChild("DataLeft", ImVec2(350, 0), ImGuiChildFlags_Borders);
    {
        // 20260320 ZJH 可用数据集标题
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "可用数据集");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 刷新按钮
        if (ImGui::Button("刷新", ImVec2(-1, 25))) {
            checkDatasets(state);
        }

        ImGui::Spacing();

        // 20260319 ZJH MNIST 数据集信息
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.20f, 0.28f, 0.44f, 1.0f));
        // 20260320 ZJH MNIST 数据集信息
        if (ImGui::CollapsingHeader("MNIST 数据集", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (state.bMnistAvailable) {
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "  状态: 已就绪");
                ImGui::Text("  训练样本: %d", state.nMnistTrainSamples);
                ImGui::Text("  测试样本: %d", state.nMnistTestSamples);
                ImGui::Text("  类别: 10 (数字 0-9)");
                ImGui::Text("  图像尺寸: 28x28 灰度");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "  状态: 未找到");
                ImGui::TextWrapped("  请将 MNIST IDX 文件放入 data/mnist/ 目录:");
                ImGui::TextDisabled("    train-images-idx3-ubyte");
                ImGui::TextDisabled("    train-labels-idx1-ubyte");
                ImGui::TextDisabled("    t10k-images-idx3-ubyte");
                ImGui::TextDisabled("    t10k-labels-idx1-ubyte");
            }
        }
        ImGui::PopStyleColor();

        ImGui::Spacing();

        // 20260319 ZJH 合成数据集（总是可用）
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.20f, 0.28f, 0.44f, 1.0f));
        // 20260320 ZJH 合成数据集
        if (ImGui::CollapsingHeader("合成数据", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "  状态: 始终可用");
            ImGui::Text("  训练样本: 1000");
            ImGui::Text("  测试样本: 200");
            ImGui::Text("  类别: 10");
            ImGui::TextWrapped("  当 MNIST 数据不可用时自动生成。");
        }
        ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260319 ZJH 导入数据集按钮（占位功能）
        // 20260320 ZJH 导入数据集按钮（占位功能）
        if (ImGui::Button("导入数据集...", ImVec2(-1, 30))) {
            // 20260320 ZJH 未来实现自定义数据集导入
        }
        ImGui::TextDisabled("自定义数据集导入功能即将推出。");
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // 20260319 ZJH 右侧：数据集统计和预览
    ImGui::BeginChild("DataRight", ImVec2(0, 0));
    {
        // 20260320 ZJH 数据集统计标题
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "数据集统计");
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 类别分布图表
        if (ImPlot::BeginPlot("类别分布 (MNIST)", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("类别", "数量");

            if (state.bMnistAvailable) {
                // 20260319 ZJH MNIST 各类别约 6000 个训练样本（均匀分布）
                double arrClasses[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
                double arrCounts[10];
                double fAvgPerClass = static_cast<double>(state.nMnistTrainSamples) / 10.0;
                for (int i = 0; i < 10; ++i) {
                    arrCounts[i] = fAvgPerClass;  // 20260319 ZJH 近似均匀分布
                }
                ImPlot::PlotBars("训练样本", arrClasses, arrCounts, 10, 0.6);
            } else {
                // 20260319 ZJH 合成数据：每类 100 个
                double arrClasses[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
                double arrCounts[10] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
                ImPlot::PlotBars("合成样本", arrClasses, arrCounts, 10, 0.6);
            }
            ImPlot::EndPlot();
        }

        ImGui::Spacing();

        // 20260319 ZJH 数据集详细信息表格
        // 20260320 ZJH 数据集信息汇总表格
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "数据集信息");
        ImGui::Separator();

        if (ImGui::BeginTable("DatasetTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("数据集");
            ImGui::TableSetupColumn("样本数");
            ImGui::TableSetupColumn("类别数");
            ImGui::TableSetupColumn("状态");
            ImGui::TableHeadersRow();

            // 20260319 ZJH MNIST 行
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("MNIST (训练)");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", state.bMnistAvailable ? state.nMnistTrainSamples : 0);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("10");
            ImGui::TableSetColumnIndex(3);
            if (state.bMnistAvailable) {
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "已就绪");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "未找到");
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("MNIST (测试)");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", state.bMnistAvailable ? state.nMnistTestSamples : 0);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("10");
            ImGui::TableSetColumnIndex(3);
            if (state.bMnistAvailable) {
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "已就绪");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "未找到");
            }

            // 20260320 ZJH 合成数据行
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("合成数据 (训练)");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("1000");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("10");
            ImGui::TableSetColumnIndex(3);
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "已就绪");

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("合成数据 (测试)");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("200");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("10");
            ImGui::TableSetColumnIndex(3);
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "已就绪");

            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260319 ZJH 绘制模型仓库工作台
// ============================================================================
static void drawModelRepository(AppState& state) {
    // 20260319 ZJH 首次进入时扫描模型目录
    if (!state.bModelsScanned) {
        scanModels(state);
    }

    // 20260319 ZJH 刷新按钮
    // 20260320 ZJH 刷新模型列表按钮
    if (ImGui::Button("刷新")) {
        scanModels(state);
    }

    ImGui::SameLine();
    ImGui::Text("data/models/ 目录中的模型: %d", static_cast<int>(state.vecModels.size()));

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260319 ZJH 模型列表表格
    if (ImGui::BeginTable("ModelTable", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetContentRegionAvail().y - 40))) {
        ImGui::TableSetupColumn("文件名", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("大小", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("修改时间", ImGuiTableColumnFlags_WidthFixed, 180);
        ImGui::TableSetupColumn("操作", ImGuiTableColumnFlags_WidthFixed, 160);
        ImGui::TableSetupColumn("类型", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableHeadersRow();

        // 20260319 ZJH 用于标记待删除的模型索引
        int nDeleteIndex = -1;

        for (int i = 0; i < static_cast<int>(state.vecModels.size()); ++i) {
            auto& model = state.vecModels[static_cast<size_t>(i)];

            ImGui::TableNextRow();

            // 20260319 ZJH 文件名
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", model.strFileName.c_str());

            // 20260319 ZJH 文件大小
            ImGui::TableSetColumnIndex(1);
            if (model.nFileSize < 1024) {
                ImGui::Text("%llu B", static_cast<unsigned long long>(model.nFileSize));
            } else if (model.nFileSize < 1024 * 1024) {
                ImGui::Text("%.1f KB", static_cast<double>(model.nFileSize) / 1024.0);
            } else {
                ImGui::Text("%.1f MB", static_cast<double>(model.nFileSize) / (1024.0 * 1024.0));
            }

            // 20260319 ZJH 最后修改时间
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", model.strLastModified.c_str());

            // 20260319 ZJH 操作按钮
            ImGui::TableSetColumnIndex(3);
            ImGui::PushID(i);  // 20260319 ZJH 为每行按钮设置唯一 ID

            // 20260319 ZJH 使用模型路径加载到推理工作台
            if (ImGui::SmallButton("加载")) {
                // 20260319 ZJH 切换到推理标签并设置路径
                state.nActiveTab = 1;
            }
            ImGui::SameLine();

            // 20260319 ZJH 删除按钮
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.15f, 0.15f, 1.0f));
            if (ImGui::SmallButton("删除")) {
                nDeleteIndex = i;
            }
            ImGui::PopStyleColor();
            ImGui::PopID();

            // 20260319 ZJH 模型类型推断
            ImGui::TableSetColumnIndex(4);
            if (model.strFileName.find("mlp") != std::string::npos) {
                ImGui::Text("MLP");
            } else if (model.strFileName.find("resnet") != std::string::npos) {
                ImGui::Text("ResNet-18");
            } else {
                ImGui::Text("未知");
            }
        }

        ImGui::EndTable();

        // 20260319 ZJH 执行删除操作（在表格绘制完成后）
        if (nDeleteIndex >= 0 && nDeleteIndex < static_cast<int>(state.vecModels.size())) {
            auto& model = state.vecModels[static_cast<size_t>(nDeleteIndex)];
            try {
                std::filesystem::remove(model.strFilePath);
            } catch (...) {}
            state.vecModels.erase(state.vecModels.begin() + nDeleteIndex);
        }
    }

    // 20260319 ZJH 空列表提示
    if (state.vecModels.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("暂无模型。请先训练模型。");
    }
}

// ============================================================================
// 20260319 ZJH 绘制状态栏
// ============================================================================
static void drawStatusBar(AppState& state) {
    auto& ts = state.trainState;

    ImGuiViewport* pViewport = ImGui::GetMainViewport();
    float fBarHeight = 28.0f;

    // 20260319 ZJH 设置状态栏位置（窗口底部）
    ImGui::SetNextWindowPos(ImVec2(pViewport->WorkPos.x, pViewport->WorkPos.y + pViewport->WorkSize.y - fBarHeight));
    ImGui::SetNextWindowSize(ImVec2(pViewport->WorkSize.x, fBarHeight));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.10f, 1.0f));

    ImGui::Begin("##StatusBar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav);

    // 20260319 ZJH 状态文本
    // 20260320 ZJH 状态栏中文文本
    if (ts.bRunning.load()) {
        if (ts.bPaused.load()) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "状态: 训练已暂停");
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "状态: 训练中...");
        }
    } else {
        ImGui::Text("状态: 就绪");
    }

    ImGui::SameLine(200);
    ImGui::Text("设备: CPU");

    // 20260320 ZJH 版本信息
    ImGui::SameLine(350);
    ImGui::Text("DeepForge 深度学习平台 v0.1.0");

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// ============================================================================
// 20260319 ZJH 主函数 — SDL3 + ImGui 应用程序入口
// ============================================================================
int main(int, char**) {
    // ===== 第一步：初始化 SDL3 =====
    // 20260319 ZJH 初始化 SDL 视频子系统
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    // 20260319 ZJH 创建主窗口（1280x720，可调整大小）
    // 20260320 ZJH 创建主窗口，标题为中文
    SDL_Window* pWindow = SDL_CreateWindow("DeepForge 深度学习平台 v0.1.0",
                                            1280, 720,
                                            SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);
    if (!pWindow) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // 20260319 ZJH 创建 SDL 渲染器
    SDL_Renderer* pRenderer = SDL_CreateRenderer(pWindow, nullptr);
    if (!pRenderer) {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(pWindow);
        SDL_Quit();
        return 1;
    }

    // ===== 第二步：初始化 ImGui =====
    // 20260319 ZJH 创建 ImGui 上下文
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    // 20260319 ZJH 配置 ImGui IO
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;  // 20260319 ZJH 启用 Docking

    // 20260319 ZJH 设置深色工业主题
    ImGui::StyleColorsDark();
    setupImGuiStyle();

    // 20260319 ZJH 尝试加载中文字体（微软雅黑）
    // 如果找不到字体文件，使用 ImGui 默认字体（无法显示中文，但不会崩溃）
    const char* arrFontPaths[] = {
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc"
    };

    bool bFontLoaded = false;
    for (const char* pFontPath : arrFontPaths) {
        if (std::filesystem::exists(pFontPath)) {
            // 20260319 ZJH 加载字体，包含常用中文字符范围
            ImFontConfig fontConfig;
            fontConfig.MergeMode = false;
            io.Fonts->AddFontFromFileTTF(pFontPath, 16.0f, &fontConfig,
                                          io.Fonts->GetGlyphRangesChineseFull());
            bFontLoaded = true;
            break;
        }
    }

    if (!bFontLoaded) {
        // 20260319 ZJH 使用默认字体
        io.Fonts->AddFontDefault();
    }

    // 20260319 ZJH 初始化 SDL3 后端
    ImGui_ImplSDL3_InitForSDLRenderer(pWindow, pRenderer);
    ImGui_ImplSDLRenderer3_Init(pRenderer);

    // ===== 第三步：闪屏动画 =====
    // 20260320 ZJH 启动闪屏，显示 2.5 秒后进入主界面
    {
        auto splashStart = std::chrono::steady_clock::now();  // 20260320 ZJH 闪屏开始时间
        const float fSplashDuration = 2.5f;  // 20260320 ZJH 闪屏持续时间（秒）
        bool bSplashRunning = true;  // 20260320 ZJH 闪屏循环标志

        while (bSplashRunning) {
            // 20260320 ZJH 处理 SDL 事件（允许关闭窗口）
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL3_ProcessEvent(&event);
                if (event.type == SDL_EVENT_QUIT) {
                    bSplashRunning = false;  // 20260320 ZJH 用户关闭窗口时退出闪屏
                    break;
                }
                if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                    event.window.windowID == SDL_GetWindowID(pWindow)) {
                    bSplashRunning = false;
                    break;
                }
            }
            if (!bSplashRunning) break;

            // 20260320 ZJH 计算已过时间和淡入透明度
            auto now = std::chrono::steady_clock::now();
            float fElapsed = std::chrono::duration<float>(now - splashStart).count();
            if (fElapsed > fSplashDuration) break;  // 20260320 ZJH 闪屏时间到，进入主界面

            float fAlpha = std::min(fElapsed / 0.8f, 1.0f);  // 20260320 ZJH 0.8 秒淡入效果

            // 20260320 ZJH 开始 ImGui 新帧
            ImGui_ImplSDLRenderer3_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();

            // 20260320 ZJH 全屏无边框窗口作为闪屏背景
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.08f, 1.0f));
            ImGui::Begin("##Splash", nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoDocking |
                         ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoCollapse);

            auto* pDrawList = ImGui::GetWindowDrawList();  // 20260320 ZJH 获取绘制列表
            float fCenterX = io.DisplaySize.x * 0.5f;  // 20260320 ZJH 屏幕中心 X
            float fCenterY = io.DisplaySize.y * 0.5f;  // 20260320 ZJH 屏幕中心 Y

            // 20260320 ZJH 绘制背景渐变（从深色到稍浅的深色）
            ImU32 nTopColor = IM_COL32(15, 15, 20, static_cast<int>(255 * fAlpha));
            ImU32 nBotColor = IM_COL32(30, 30, 45, static_cast<int>(255 * fAlpha));
            pDrawList->AddRectFilledMultiColor(
                ImVec2(0, 0), io.DisplaySize,
                nTopColor, nTopColor, nBotColor, nBotColor);

            // 20260320 ZJH 绘制 "DF" Logo 方块（蓝色圆角矩形 + 白色文字）
            float fLogoSize = 80.0f;  // 20260320 ZJH Logo 方块尺寸
            float fLogoX = fCenterX - fLogoSize * 0.5f;  // 20260320 ZJH Logo 左上角 X
            float fLogoY = fCenterY - 120.0f;  // 20260320 ZJH Logo 左上角 Y
            pDrawList->AddRectFilled(
                ImVec2(fLogoX, fLogoY),
                ImVec2(fLogoX + fLogoSize, fLogoY + fLogoSize),
                IM_COL32(50, 120, 220, static_cast<int>(240 * fAlpha)),
                12.0f);  // 20260320 ZJH 12px 圆角
            // 20260320 ZJH 在 Logo 方块中绘制 "DF" 文字
            ImFont* pFont = ImGui::GetFont();
            float fDfFontSize = 42.0f;
            const char* strDf = "DF";
            ImVec2 dfTextSize = pFont->CalcTextSizeA(fDfFontSize, FLT_MAX, 0.0f, strDf);
            pDrawList->AddText(pFont, fDfFontSize,
                ImVec2(fLogoX + (fLogoSize - dfTextSize.x) * 0.5f,
                       fLogoY + (fLogoSize - dfTextSize.y) * 0.5f),
                IM_COL32(255, 255, 255, static_cast<int>(255 * fAlpha)),
                strDf);

            // 20260320 ZJH 绘制标题 "DeepForge"
            float fTitleFontSize = 36.0f;
            const char* strTitle = "DeepForge";
            ImVec2 titleSize = pFont->CalcTextSizeA(fTitleFontSize, FLT_MAX, 0.0f, strTitle);
            pDrawList->AddText(pFont, fTitleFontSize,
                ImVec2(fCenterX - titleSize.x * 0.5f, fLogoY + fLogoSize + 20.0f),
                IM_COL32(230, 230, 240, static_cast<int>(255 * fAlpha)),
                strTitle);

            // 20260320 ZJH 绘制副标题（中文）
            float fSubFontSize = 18.0f;
            const char* strSubtitle = "纯 C++ 全流程深度学习视觉平台";
            ImVec2 subSize = pFont->CalcTextSizeA(fSubFontSize, FLT_MAX, 0.0f, strSubtitle);
            pDrawList->AddText(pFont, fSubFontSize,
                ImVec2(fCenterX - subSize.x * 0.5f, fLogoY + fLogoSize + 64.0f),
                IM_COL32(160, 180, 220, static_cast<int>(220 * fAlpha)),
                strSubtitle);

            // 20260320 ZJH 绘制版本号
            float fVerFontSize = 14.0f;
            const char* strVersion = "v0.1.0";
            ImVec2 verSize = pFont->CalcTextSizeA(fVerFontSize, FLT_MAX, 0.0f, strVersion);
            pDrawList->AddText(pFont, fVerFontSize,
                ImVec2(fCenterX - verSize.x * 0.5f, fLogoY + fLogoSize + 90.0f),
                IM_COL32(120, 140, 170, static_cast<int>(200 * fAlpha)),
                strVersion);

            // 20260320 ZJH 绘制旋转加载点动画
            int nDotCount = 8;  // 20260320 ZJH 点的数量
            float fDotPhase = fElapsed * 3.0f;  // 20260320 ZJH 旋转速度
            float fDotRadius = 20.0f;  // 20260320 ZJH 旋转半径
            float fDotCenterY = fLogoY + fLogoSize + 130.0f;  // 20260320 ZJH 点阵中心 Y
            for (int i = 0; i < nDotCount; ++i) {
                float fAngle = fDotPhase + static_cast<float>(i) * (2.0f * 3.14159265f / static_cast<float>(nDotCount));
                float fDotX = fCenterX + std::cos(fAngle) * fDotRadius;
                float fDotY = fDotCenterY + std::sin(fAngle) * fDotRadius;
                // 20260320 ZJH 每个点的透明度随相位变化，产生追逐效果
                float fDotAlpha = (std::sin(fAngle - fDotPhase) + 1.0f) * 0.5f;
                float fDotSize = 3.0f + fDotAlpha * 2.0f;  // 20260320 ZJH 点大小随透明度变化
                pDrawList->AddCircleFilled(ImVec2(fDotX, fDotY), fDotSize,
                    IM_COL32(100, 160, 255, static_cast<int>(fDotAlpha * 255.0f * fAlpha)));
            }

            // 20260320 ZJH 绘制 "正在初始化..." 提示文字
            float fInitFontSize = 14.0f;
            const char* strInit = "正在初始化...";
            ImVec2 initSize = pFont->CalcTextSizeA(fInitFontSize, FLT_MAX, 0.0f, strInit);
            pDrawList->AddText(pFont, fInitFontSize,
                ImVec2(fCenterX - initSize.x * 0.5f, fDotCenterY + fDotRadius + 20.0f),
                IM_COL32(140, 150, 180, static_cast<int>(180 * fAlpha)),
                strInit);

            // 20260320 ZJH 绘制版权信息
            float fCopyFontSize = 12.0f;
            const char* strCopy = "© 2026 ZJH";
            ImVec2 copySize = pFont->CalcTextSizeA(fCopyFontSize, FLT_MAX, 0.0f, strCopy);
            pDrawList->AddText(pFont, fCopyFontSize,
                ImVec2(fCenterX - copySize.x * 0.5f, io.DisplaySize.y - 40.0f),
                IM_COL32(100, 100, 120, static_cast<int>(150 * fAlpha)),
                strCopy);

            ImGui::End();
            ImGui::PopStyleColor();
            ImGui::PopStyleVar();

            // 20260320 ZJH 渲染闪屏帧
            ImGui::Render();
            SDL_SetRenderDrawColor(pRenderer, 15, 15, 20, 255);
            SDL_RenderClear(pRenderer);
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer);
            SDL_RenderPresent(pRenderer);
        }

        // 20260320 ZJH 如果用户在闪屏阶段关闭了窗口，直接退出
        if (!bSplashRunning) {
            ImGui_ImplSDLRenderer3_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImPlot::DestroyContext();
            ImGui::DestroyContext();
            SDL_DestroyRenderer(pRenderer);
            SDL_DestroyWindow(pWindow);
            SDL_Quit();
            return 0;
        }
    }

    // ===== 第四步：应用状态初始化 =====
    AppState appState;

    // ===== 第五步：主循环 =====
    bool bRunning = true;
    while (bRunning) {
        // 20260319 ZJH 处理 SDL 事件
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                bRunning = false;  // 20260319 ZJH 收到退出事件时结束主循环
            }
            // 20260319 ZJH 窗口关闭事件
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                event.window.windowID == SDL_GetWindowID(pWindow)) {
                bRunning = false;
            }
        }

        // 20260319 ZJH 开始 ImGui 新帧
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // 20260319 ZJH 创建全屏 DockSpace
        ImGuiViewport* pViewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(pViewport->WorkPos);
        ImGui::SetNextWindowSize(ImVec2(pViewport->WorkSize.x, pViewport->WorkSize.y - 28.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("##MainDock", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::PopStyleVar();

        // 20260319 ZJH 顶部工具栏（标签页切换 + GPU 信息）
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
        {
            // 20260320 ZJH 工作台标签页按钮（中文）
            const char* arrTabs[] = {"训练", "推理", "数据管理", "模型仓库"};
            for (int i = 0; i < 4; ++i) {
                if (i > 0) ImGui::SameLine();
                bool bSelected = (appState.nActiveTab == i);
                if (bSelected) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.30f, 0.50f, 1.0f));
                }
                if (ImGui::Button(arrTabs[i], ImVec2(0, 0))) {
                    appState.nActiveTab = i;
                }
                if (bSelected) {
                    ImGui::PopStyleColor();
                }
            }

            // 20260319 ZJH 右侧显示设备信息
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80);
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "设备: CPU");
        }
        ImGui::PopStyleVar();

        ImGui::Separator();

        // 20260319 ZJH 主工作区内容（根据选中标签页绘制对应工作台）
        ImGui::BeginChild("WorkspaceContent", ImVec2(0, 0));
        {
            switch (appState.nActiveTab) {
                case 0:
                    drawTrainingWorkbench(appState);
                    break;
                case 1:
                    drawInferenceWorkbench(appState, pRenderer);
                    break;
                case 2:
                    drawDataManager(appState);
                    break;
                case 3:
                    drawModelRepository(appState);
                    break;
            }
        }
        ImGui::EndChild();

        ImGui::End();  // 20260319 ZJH 结束 MainDock 窗口

        // 20260319 ZJH 绘制状态栏
        drawStatusBar(appState);

        // 20260319 ZJH 渲染
        ImGui::Render();
        SDL_SetRenderDrawColor(pRenderer, 30, 30, 30, 255);
        SDL_RenderClear(pRenderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer);
        SDL_RenderPresent(pRenderer);
    }

    // ===== 第六步：清理 =====
    // 20260319 ZJH 停止训练线程
    if (appState.pTrainThread) {
        appState.trainState.bStopRequested.store(true);
        appState.trainState.bPaused.store(false);
        appState.pTrainThread->request_stop();
        appState.pTrainThread->join();
        appState.pTrainThread.reset();
    }

    // 20260319 ZJH 释放推理图像纹理
    if (appState.inferState.pImageTexture) {
        SDL_DestroyTexture(appState.inferState.pImageTexture);
    }

    // 20260319 ZJH 关闭 ImGui 后端
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    // 20260319 ZJH 关闭 SDL
    SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    SDL_Quit();

    return 0;  // 20260319 ZJH 正常退出
}
