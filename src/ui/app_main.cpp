// 20260320 ZJH DeepForge Phase 4 — SDL3 + ImGui 桌面 GUI 应用程序
// MVTec Deep Learning Tool 风格 UI：左侧任务导航 + 步骤标签页 + 右侧属性面板
// GPU 动态检测（通过 LoadLibrary 加载 nvcuda.dll），状态栏显示 GPU 信息
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
#include <array>

// 20260320 ZJH Windows 头文件用于动态加载 CUDA 驱动库
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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
// 20260320 ZJH GPU 设备信息结构体
// ============================================================================
struct GpuDeviceInfo {
    std::string strName;           // 20260320 ZJH GPU 名称
    size_t nTotalMemoryMB = 0;     // 20260320 ZJH 显存大小 (MB)
    int nComputeCapMajor = 0;      // 20260320 ZJH 计算能力主版本
    int nComputeCapMinor = 0;      // 20260320 ZJH 计算能力次版本
};

// ============================================================================
// 20260320 ZJH 检测可用 GPU 设备（通过动态加载 nvcuda.dll）
// ============================================================================
static std::vector<GpuDeviceInfo> detectGpuDevices() {
    std::vector<GpuDeviceInfo> vecDevices;  // 20260320 ZJH 存储检测到的 GPU 设备列表

#ifdef _WIN32
    // 20260320 ZJH 尝试动态加载 CUDA 驱动库
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    if (!hCuda) return vecDevices;  // 20260320 ZJH CUDA 驱动不可用，返回空列表

    // 20260320 ZJH 定义 CUDA Driver API 函数指针类型
    typedef int (*CuInit_t)(unsigned int);
    typedef int (*CuDeviceGetCount_t)(int*);
    typedef int (*CuDeviceGetName_t)(char*, int, int);
    typedef int (*CuDeviceTotalMem_t)(size_t*, int);
    typedef int (*CuDeviceGetAttribute_t)(int*, int, int);

    // 20260320 ZJH 获取各 API 函数指针
    auto pfnCuInit = (CuInit_t)GetProcAddress(hCuda, "cuInit");
    auto pfnCuDeviceGetCount = (CuDeviceGetCount_t)GetProcAddress(hCuda, "cuDeviceGetCount");
    auto pfnCuDeviceGetName = (CuDeviceGetName_t)GetProcAddress(hCuda, "cuDeviceGetName");
    auto pfnCuDeviceTotalMem = (CuDeviceTotalMem_t)GetProcAddress(hCuda, "cuDeviceTotalMem_v2");
    auto pfnCuDeviceGetAttribute = (CuDeviceGetAttribute_t)GetProcAddress(hCuda, "cuDeviceGetAttribute");

    // 20260320 ZJH 必须有 cuInit 和 cuDeviceGetCount 才能继续
    if (!pfnCuInit || !pfnCuDeviceGetCount) {
        FreeLibrary(hCuda);
        return vecDevices;
    }

    // 20260320 ZJH 初始化 CUDA 驱动
    if (pfnCuInit(0) != 0) {
        FreeLibrary(hCuda);  // 20260320 ZJH 初始化失败
        return vecDevices;
    }

    // 20260320 ZJH 查询 GPU 设备数量
    int nDeviceCount = 0;
    pfnCuDeviceGetCount(&nDeviceCount);

    // 20260320 ZJH 遍历每个 GPU 设备，获取详细信息
    for (int i = 0; i < nDeviceCount; ++i) {
        GpuDeviceInfo info;  // 20260320 ZJH 当前设备信息

        // 20260320 ZJH 获取设备名称
        char arrName[256] = {};
        if (pfnCuDeviceGetName) pfnCuDeviceGetName(arrName, 256, i);
        info.strName = arrName;

        // 20260320 ZJH 获取设备总显存
        size_t nTotalMem = 0;
        if (pfnCuDeviceTotalMem) pfnCuDeviceTotalMem(&nTotalMem, i);
        info.nTotalMemoryMB = nTotalMem / (1024 * 1024);  // 20260320 ZJH 转换为 MB

        // 20260320 ZJH 获取计算能力（属性 75 = 主版本, 76 = 次版本）
        if (pfnCuDeviceGetAttribute) {
            int nMajor = 0, nMinor = 0;
            pfnCuDeviceGetAttribute(&nMajor, 75, i);  // 20260320 ZJH CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            pfnCuDeviceGetAttribute(&nMinor, 76, i);  // 20260320 ZJH CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            info.nComputeCapMajor = nMajor;
            info.nComputeCapMinor = nMinor;
        }

        vecDevices.push_back(info);  // 20260320 ZJH 添加到设备列表
    }

    FreeLibrary(hCuda);  // 20260320 ZJH 释放 CUDA 驱动库
#endif

    return vecDevices;  // 20260320 ZJH 返回检测到的 GPU 设备列表
}

// ============================================================================
// 20260320 ZJH 查询 GPU 当前显存使用情况（动态加载 nvcuda.dll）
// 返回 {已用MB, 总计MB}，失败返回 {0, 0}
// ============================================================================
static std::pair<size_t, size_t> queryGpuMemoryUsage() {
#ifdef _WIN32
    // 20260320 ZJH 动态加载 CUDA 驱动
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    if (!hCuda) return {0, 0};

    // 20260320 ZJH 获取所需函数指针
    typedef int (*CuInit_t)(unsigned int);
    typedef int (*CuCtxGetCurrent_t)(void**);
    typedef int (*CuDeviceGet_t)(int*, int);
    typedef int (*CuCtxCreate_t)(void**, unsigned int, int);
    typedef int (*CuCtxDestroy_t)(void*);
    typedef int (*CuMemGetInfo_t)(size_t*, size_t*);

    auto pfnCuInit = (CuInit_t)GetProcAddress(hCuda, "cuInit");
    auto pfnCuMemGetInfo = (CuMemGetInfo_t)GetProcAddress(hCuda, "cuMemGetInfo_v2");

    if (!pfnCuInit || !pfnCuMemGetInfo) {
        FreeLibrary(hCuda);
        return {0, 0};
    }

    if (pfnCuInit(0) != 0) {
        FreeLibrary(hCuda);
        return {0, 0};
    }

    // 20260320 ZJH 尝试查询显存，需要有活跃的 CUDA 上下文
    size_t nFree = 0, nTotal = 0;
    // 20260320 ZJH 如果没有活跃上下文，cuMemGetInfo 会返回错误
    int nResult = pfnCuMemGetInfo(&nFree, &nTotal);
    FreeLibrary(hCuda);

    if (nResult == 0 && nTotal > 0) {
        size_t nUsedMB = (nTotal - nFree) / (1024 * 1024);
        size_t nTotalMB = nTotal / (1024 * 1024);
        return {nUsedMB, nTotalMB};
    }
#endif
    return {0, 0};  // 20260320 ZJH 无法查询
}

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
    std::atomic<float> fCurrentLR{0.01f};     // 20260320 ZJH 当前学习率

    // 20260319 ZJH 互斥锁保护的历史数据，UI 线程读取时需加锁
    std::mutex mutex;
    std::vector<float> vecLossHistory;        // 20260319 ZJH 每轮平均损失历史
    std::vector<float> vecTrainAccHistory;    // 20260319 ZJH 每轮训练准确率历史
    std::vector<float> vecTestAccHistory;     // 20260319 ZJH 每轮测试准确率历史
    std::string strLog;                       // 20260319 ZJH 训练日志文本
    std::string strSavedModelPath;            // 20260319 ZJH 保存的模型路径

    // 20260320 ZJH 混淆矩阵数据（10x10，训练完成后填充）
    std::array<std::array<int, 10>, 10> arrConfusionMatrix{};  // 20260320 ZJH [实际][预测]
    bool bHasConfusionMatrix = false;         // 20260320 ZJH 是否已计算混淆矩阵

    // 20260320 ZJH 每批次耗时（用于 ETA 估算）
    std::atomic<float> fAvgBatchTimeMs{0.0f};  // 20260320 ZJH 平均每批次耗时（毫秒）

    // 20260320 ZJH 最佳验证损失
    float fBestValLoss = 999.0f;              // 20260320 ZJH 训练过程中最小的验证损失
    float fTotalTrainingTimeSec = 0.0f;       // 20260320 ZJH 总训练耗时（秒）

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
        fCurrentLR.store(0.01f);
        fAvgBatchTimeMs.store(0.0f);
        std::lock_guard<std::mutex> lock(mutex);
        vecLossHistory.clear();
        vecTrainAccHistory.clear();
        vecTestAccHistory.clear();
        strLog.clear();
        strSavedModelPath.clear();
        arrConfusionMatrix = {};
        bHasConfusionMatrix = false;
        fBestValLoss = 999.0f;
        fTotalTrainingTimeSec = 0.0f;
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
// 20260320 ZJH 全局应用状态（MVTec 风格 UI）
// ============================================================================
struct AppState {
    // 20260320 ZJH 任务类型索引：0=分类, 1=检测, 2=分割, 3=异常检测
    int nActiveTask = 0;
    // 20260320 ZJH 当前步骤索引：0=数据, 1=配置, 2=训练, 3=评估
    int nActiveStep = 0;

    TrainingState trainState;                 // 20260319 ZJH 训练状态
    InferenceState inferState;                // 20260319 ZJH 推理状态

    // 20260319 ZJH 训练超参数（UI 可调）
    int nSelectedModel = 0;                   // 20260320 ZJH 选中的模型索引（0=MLP, 1=ResNet-18, 2=ResNet-34）
    int nEpochs = 10;                         // 20260319 ZJH 训练轮数
    int nBatchSize = 64;                      // 20260319 ZJH 批次大小
    float fLearningRate = 0.01f;              // 20260319 ZJH 学习率
    int nSelectedOptimizer = 0;               // 20260320 ZJH 选中的优化器（0=SGD, 1=Adam, 2=AdamW）
    int nLRSchedule = 0;                      // 20260320 ZJH 学习率策略（0=固定, 1=余弦退火, 2=预热）
    int nSelectedDataset = 0;                 // 20260320 ZJH 数据集选择（0=MNIST, 1=合成数据, 2=自定义）
    float fTrainSplit = 0.8f;                 // 20260320 ZJH 训练集比例
    float fValSplit = 0.1f;                   // 20260320 ZJH 验证集比例
    float fTestSplit = 0.1f;                  // 20260320 ZJH 测试集比例

    // 20260320 ZJH 数据增强选项
    bool bAugFlip = false;                    // 20260320 ZJH 翻转
    bool bAugRotate = false;                  // 20260320 ZJH 旋转
    bool bAugColorJitter = false;             // 20260320 ZJH 色彩抖动

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

    // 20260320 ZJH GPU 相关
    std::vector<GpuDeviceInfo> vecGpuDevices; // 20260320 ZJH 检测到的 GPU 列表
    bool bGpuDetected = false;                // 20260320 ZJH 是否检测到 GPU
    int nSelectedDevice = 0;                  // 20260320 ZJH 选中的设备（0=CPU, 1+=GPU）
    bool bShowDevicePopup = false;            // 20260320 ZJH 是否显示设备选择弹窗
    size_t nGpuMemUsedMB = 0;                // 20260320 ZJH GPU 已用显存 (MB)
    size_t nGpuMemTotalMB = 0;               // 20260320 ZJH GPU 总显存 (MB)
    float fGpuMemQueryTimer = 0.0f;          // 20260320 ZJH 显存查询计时器
};

// ============================================================================
// 20260319 ZJH 辅助函数前向声明
// ============================================================================
static void setupImGuiStyle();
static void drawStatusBar(AppState& state);
static void startTraining(AppState& state);
static void scanModels(AppState& state);
static void checkDatasets(AppState& state);

// ============================================================================
// 20260320 ZJH 颜色常量定义（蓝色主色调 #3366CC）
// ============================================================================
static const ImVec4 s_colAccent       = ImVec4(0.20f, 0.40f, 0.80f, 1.0f);   // 20260320 ZJH 主色调蓝
static const ImVec4 s_colAccentLight  = ImVec4(0.40f, 0.60f, 1.00f, 1.0f);   // 20260320 ZJH 浅蓝高亮
static const ImVec4 s_colAccentDark   = ImVec4(0.15f, 0.30f, 0.60f, 1.0f);   // 20260320 ZJH 深蓝
static const ImVec4 s_colGreen        = ImVec4(0.20f, 0.85f, 0.30f, 1.0f);   // 20260320 ZJH 绿色状态
static const ImVec4 s_colOrange       = ImVec4(1.00f, 0.65f, 0.10f, 1.0f);   // 20260320 ZJH 橙色警告
static const ImVec4 s_colRed          = ImVec4(0.90f, 0.25f, 0.25f, 1.0f);   // 20260320 ZJH 红色错误
static const ImVec4 s_colGray         = ImVec4(0.55f, 0.55f, 0.60f, 1.0f);   // 20260320 ZJH 灰色文本
static const ImVec4 s_colSectionTitle = ImVec4(0.40f, 0.70f, 1.00f, 1.0f);   // 20260320 ZJH 区段标题色

// ============================================================================
// 20260320 ZJH 辅助：绘制区段标题（蓝色文字 + 分隔线）
// ============================================================================
static void drawSectionTitle(const char* strTitle) {
    ImGui::Spacing();
    ImGui::TextColored(s_colSectionTitle, "%s", strTitle);  // 20260320 ZJH 蓝色标题文字
    ImGui::Separator();
    ImGui::Spacing();
}

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

    // 20260320 ZJH 记录训练开始时间（用于总耗时统计）
    auto timeTrainStart = std::chrono::steady_clock::now();

    // 20260319 ZJH 读取 UI 超参数
    bool bUseResNet = (state.nSelectedModel >= 1);  // 20260320 ZJH 使用 ResNet 系列
    int nEpochs = state.nEpochs;                    // 20260319 ZJH 训练轮数
    int nBatchSize = state.nBatchSize;               // 20260319 ZJH 批次大小
    float fLearningRate = state.fLearningRate;       // 20260319 ZJH 学习率
    bool bUseAdam = (state.nSelectedOptimizer >= 1); // 20260320 ZJH 使用 Adam 或 AdamW
    const int nInputDim = 784;                       // 20260319 ZJH 输入维度（28x28）
    const int nHiddenDim = 128;                      // 20260319 ZJH MLP 隐藏层维度
    const int nOutputDim = 10;                       // 20260319 ZJH 输出维度（10 个类别）

    ts.nTotalEpochs.store(nEpochs);
    ts.fCurrentLR.store(fLearningRate);

    // 20260319 ZJH 配置信息写入日志
    const char* arrModelNames[] = {"MLP (784->128->10)", "ResNet-18", "ResNet-34"};
    ts.appendLog(std::string("  模型: ") + arrModelNames[state.nSelectedModel]);
    {
        char arrBuf[256];
        const char* arrOptNames[] = {"SGD", "Adam", "AdamW"};
        std::snprintf(arrBuf, sizeof(arrBuf), "  训练轮数: %d, 批次大小: %d, 学习率: %.4f, 优化器: %s",
                      nEpochs, nBatchSize, static_cast<double>(fLearningRate),
                      arrOptNames[state.nSelectedOptimizer]);
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

            // 20260320 ZJH 记录批次开始时间（用于 ETA 估算）
            auto timeBatchStart = std::chrono::steady_clock::now();

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

            // 20260320 ZJH 更新平均每批次耗时
            auto timeBatchEnd = std::chrono::steady_clock::now();
            float fBatchMs = std::chrono::duration<float, std::milli>(timeBatchEnd - timeBatchStart).count();
            // 20260320 ZJH 指数移动平均平滑
            float fPrev = ts.fAvgBatchTimeMs.load();
            if (fPrev < 0.001f) {
                ts.fAvgBatchTimeMs.store(fBatchMs);  // 20260320 ZJH 第一次直接赋值
            } else {
                ts.fAvgBatchTimeMs.store(fPrev * 0.9f + fBatchMs * 0.1f);  // 20260320 ZJH EMA 平滑
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

        // 20260320 ZJH 最后一轮时计算混淆矩阵
        bool bLastEpoch = (nEpoch == nEpochs - 1);
        if (bLastEpoch) {
            std::lock_guard<std::mutex> lock(ts.mutex);
            ts.arrConfusionMatrix = {};  // 20260320 ZJH 清空混淆矩阵
        }

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
                // 20260320 ZJH 最后一轮填充混淆矩阵
                if (bLastEpoch) {
                    int nActual = vecActual[static_cast<size_t>(i)];
                    int nPred = vecPredicted[static_cast<size_t>(i)];
                    if (nActual >= 0 && nActual < 10 && nPred >= 0 && nPred < 10) {
                        std::lock_guard<std::mutex> lock(ts.mutex);
                        ts.arrConfusionMatrix[static_cast<size_t>(nActual)][static_cast<size_t>(nPred)]++;
                    }
                }
            }
        }
        float fTestAcc = 100.0f * static_cast<float>(nTestCorrect) / static_cast<float>(nTestBatches * nBatchSize);
        pModel->train();

        if (bLastEpoch) {
            std::lock_guard<std::mutex> lock(ts.mutex);
            ts.bHasConfusionMatrix = true;  // 20260320 ZJH 标记混淆矩阵已计算
        }

        // 20260319 ZJH 更新原子变量
        ts.fTrainAcc.store(fTrainAcc);
        ts.fTestAcc.store(fTestAcc);

        // 20260320 ZJH 更新最佳验证损失
        {
            std::lock_guard<std::mutex> lock(ts.mutex);
            if (fAvgLoss < ts.fBestValLoss) {
                ts.fBestValLoss = fAvgLoss;
            }
        }

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

    // 20260320 ZJH 计算总训练耗时
    auto timeTrainEnd = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(ts.mutex);
        ts.fTotalTrainingTimeSec = std::chrono::duration<float>(timeTrainEnd - timeTrainStart).count();
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
// 20260319 ZJH 设置工业深色主题（蓝色主色调 #3366CC）
// ============================================================================
static void setupImGuiStyle() {
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;       // 20260319 ZJH 窗口圆角半径
    style.FrameRounding = 3.0f;        // 20260320 ZJH 控件边框圆角加大
    style.GrabRounding = 3.0f;         // 20260320 ZJH 滑块手柄圆角
    style.TabRounding = 4.0f;          // 20260320 ZJH 标签页圆角
    style.WindowPadding = ImVec2(8, 8);  // 20260319 ZJH 窗口内边距
    style.FramePadding = ImVec2(6, 4);   // 20260320 ZJH 控件内边距
    style.ItemSpacing = ImVec2(8, 5);    // 20260319 ZJH 控件间距
    style.ChildRounding = 4.0f;          // 20260320 ZJH 子窗口圆角

    ImVec4* colors = style.Colors;
    // 20260319 ZJH 背景色：深灰色系
    colors[ImGuiCol_WindowBg]        = ImVec4(0.11f, 0.11f, 0.13f, 1.0f);
    colors[ImGuiCol_ChildBg]         = ImVec4(0.09f, 0.09f, 0.11f, 1.0f);
    colors[ImGuiCol_PopupBg]         = ImVec4(0.13f, 0.13f, 0.16f, 0.96f);
    // 20260319 ZJH 边框色：微亮边框
    colors[ImGuiCol_Border]          = ImVec4(0.25f, 0.25f, 0.30f, 1.0f);
    // 20260319 ZJH 控件色：蓝色主色调
    colors[ImGuiCol_FrameBg]         = ImVec4(0.15f, 0.15f, 0.19f, 1.0f);
    colors[ImGuiCol_FrameBgHovered]  = ImVec4(0.20f, 0.22f, 0.28f, 1.0f);
    colors[ImGuiCol_FrameBgActive]   = ImVec4(0.24f, 0.26f, 0.34f, 1.0f);
    // 20260319 ZJH 标题栏
    colors[ImGuiCol_TitleBg]         = ImVec4(0.07f, 0.07f, 0.09f, 1.0f);
    colors[ImGuiCol_TitleBgActive]   = ImVec4(0.10f, 0.12f, 0.18f, 1.0f);
    // 20260319 ZJH 标签页
    colors[ImGuiCol_Tab]             = ImVec4(0.13f, 0.13f, 0.17f, 1.0f);
    colors[ImGuiCol_TabHovered]      = ImVec4(0.22f, 0.34f, 0.56f, 1.0f);
    colors[ImGuiCol_TabSelected]     = ImVec4(0.18f, 0.28f, 0.48f, 1.0f);
    // 20260320 ZJH 按钮色：蓝色主色调 #3366CC
    colors[ImGuiCol_Button]          = ImVec4(0.18f, 0.30f, 0.52f, 1.0f);
    colors[ImGuiCol_ButtonHovered]   = ImVec4(0.24f, 0.38f, 0.62f, 1.0f);
    colors[ImGuiCol_ButtonActive]    = ImVec4(0.28f, 0.42f, 0.68f, 1.0f);
    // 20260319 ZJH 头部（表头等）
    colors[ImGuiCol_Header]          = ImVec4(0.18f, 0.26f, 0.42f, 1.0f);
    colors[ImGuiCol_HeaderHovered]   = ImVec4(0.24f, 0.34f, 0.52f, 1.0f);
    colors[ImGuiCol_HeaderActive]    = ImVec4(0.28f, 0.38f, 0.58f, 1.0f);
    // 20260319 ZJH 分隔线
    colors[ImGuiCol_Separator]       = ImVec4(0.25f, 0.25f, 0.30f, 1.0f);
    // 20260319 ZJH 文本
    colors[ImGuiCol_Text]            = ImVec4(0.88f, 0.88f, 0.92f, 1.0f);
    colors[ImGuiCol_TextDisabled]    = ImVec4(0.48f, 0.48f, 0.52f, 1.0f);
    // 20260319 ZJH 进度条
    colors[ImGuiCol_PlotHistogram]   = ImVec4(0.26f, 0.52f, 0.88f, 1.0f);
    // 20260319 ZJH 选中色
    colors[ImGuiCol_CheckMark]       = ImVec4(0.40f, 0.70f, 1.0f, 1.0f);
    colors[ImGuiCol_SliderGrab]      = ImVec4(0.28f, 0.48f, 0.78f, 1.0f);
    colors[ImGuiCol_SliderGrabActive]= ImVec4(0.34f, 0.56f, 0.88f, 1.0f);
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
// 20260320 ZJH 绘制左侧任务导航面板
// ============================================================================
static void drawTaskSidebar(AppState& state) {
    // 20260320 ZJH 4 个任务类型的名称和描述
    struct TaskDef {
        const char* strIcon;     // 20260320 ZJH 图标文字
        const char* strName;     // 20260320 ZJH 任务名称
        const char* strDesc;     // 20260320 ZJH 任务描述
        bool bAvailable;         // 20260320 ZJH 是否可用
    };
    TaskDef arrTasks[] = {
        {"[C]", "图像分类",  "MLP / ResNet",       true},
        {"[D]", "目标检测",  "YOLOv5",             false},
        {"[S]", "语义分割",  "U-Net",              false},
        {"[A]", "异常检测",  "AutoEncoder",        false},
    };

    drawSectionTitle("任务类型");

    // 20260320 ZJH 绘制每个任务按钮
    for (int i = 0; i < 4; ++i) {
        bool bSelected = (state.nActiveTask == i);  // 20260320 ZJH 当前选中状态
        bool bAvail = arrTasks[i].bAvailable;        // 20260320 ZJH 是否可用

        // 20260320 ZJH 选中项使用蓝色高亮背景
        if (bSelected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.32f, 0.56f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.38f, 0.62f, 1.0f));
        } else if (!bAvail) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.14f, 0.14f, 0.16f, 1.0f));
        }

        // 20260320 ZJH 任务按钮：图标 + 名称
        char arrLabel[128];
        std::snprintf(arrLabel, sizeof(arrLabel), "%s %s", arrTasks[i].strIcon, arrTasks[i].strName);
        if (ImGui::Button(arrLabel, ImVec2(-1, 40))) {
            state.nActiveTask = i;  // 20260320 ZJH 切换任务
            if (bAvail) {
                state.nActiveStep = 0;  // 20260320 ZJH 切换到第一步
            }
        }

        if (bSelected || !bAvail) {
            ImGui::PopStyleColor(2);
        }

        // 20260320 ZJH 任务描述小文字
        ImGui::TextDisabled("    %s", arrTasks[i].strDesc);

        // 20260320 ZJH 不可用任务标记
        if (!bAvail) {
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 10);
            ImGui::TextColored(s_colOrange, "*");
        }

        ImGui::Spacing();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH 设备信息区域
    drawSectionTitle("设备信息");

    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        // 20260320 ZJH 显示已选设备
        if (state.nSelectedDevice == 0) {
            ImGui::TextColored(s_colOrange, "  CPU");
        } else {
            int nIdx = state.nSelectedDevice - 1;
            if (nIdx < static_cast<int>(state.vecGpuDevices.size())) {
                ImGui::TextColored(s_colGreen, "  GPU %d", nIdx);
                ImGui::TextWrapped("  %s", state.vecGpuDevices[static_cast<size_t>(nIdx)].strName.c_str());
            }
        }
    } else {
        ImGui::TextColored(s_colOrange, "  CPU (无 GPU)");
    }
}

// ============================================================================
// 20260320 ZJH 绘制步骤 1：数据
// ============================================================================
static void drawStepData(AppState& state) {
    // 20260320 ZJH 首次进入时检查数据集
    if (!state.bDataChecked) {
        checkDatasets(state);
    }

    // 20260320 ZJH 数据集选择
    drawSectionTitle("数据集选择");

    const char* arrDatasets[] = {"MNIST (手写数字)", "合成数据 (自动生成)", "自定义 (文件夹导入)"};
    ImGui::Combo("数据集", &state.nSelectedDataset, arrDatasets, 3);

    ImGui::Spacing();

    // 20260320 ZJH 显示选中数据集状态
    if (state.nSelectedDataset == 0) {
        // 20260320 ZJH MNIST 数据集
        if (state.bMnistAvailable) {
            ImGui::TextColored(s_colGreen, "MNIST 数据已就绪");
            ImGui::Text("  训练样本: %d", state.nMnistTrainSamples);
            ImGui::Text("  测试样本: %d", state.nMnistTestSamples);
            ImGui::Text("  类别: 10 (数字 0-9)");
            ImGui::Text("  图像尺寸: 28x28 灰度");
        } else {
            ImGui::TextColored(s_colRed, "MNIST 数据未找到");
            ImGui::TextWrapped("请将 MNIST IDX 文件放入 data/mnist/ 目录");
            ImGui::TextDisabled("  train-images-idx3-ubyte");
            ImGui::TextDisabled("  train-labels-idx1-ubyte");
            ImGui::TextDisabled("  t10k-images-idx3-ubyte");
            ImGui::TextDisabled("  t10k-labels-idx1-ubyte");
        }
    } else if (state.nSelectedDataset == 1) {
        // 20260320 ZJH 合成数据
        ImGui::TextColored(s_colGreen, "合成数据始终可用");
        ImGui::Text("  训练样本: 1000");
        ImGui::Text("  测试样本: 200");
        ImGui::Text("  类别: 10");
    } else {
        // 20260320 ZJH 自定义数据集（占位）
        ImGui::TextColored(s_colOrange, "自定义数据集导入功能即将推出");
        ImGui::TextDisabled("  支持按文件夹结构自动分类");
    }

    if (ImGui::Button("刷新数据检查", ImVec2(160, 28))) {
        checkDatasets(state);  // 20260320 ZJH 重新检查数据集
    }

    ImGui::Spacing();

    // 20260320 ZJH 数据集划分比例
    drawSectionTitle("数据集划分");

    ImGui::SliderFloat("训练集比例", &state.fTrainSplit, 0.5f, 0.95f, "%.2f");
    ImGui::SliderFloat("验证集比例", &state.fValSplit, 0.0f, 0.3f, "%.2f");
    // 20260320 ZJH 自动计算测试集比例
    state.fTestSplit = 1.0f - state.fTrainSplit - state.fValSplit;
    if (state.fTestSplit < 0.0f) {
        state.fValSplit = 1.0f - state.fTrainSplit;
        state.fTestSplit = 0.0f;
    }
    ImGui::Text("测试集比例: %.2f", static_cast<double>(state.fTestSplit));

    ImGui::Spacing();

    // 20260320 ZJH 类别分布统计图
    drawSectionTitle("类别分布");

    if (ImPlot::BeginPlot("##ClassDist", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("类别", "数量");

        if (state.bMnistAvailable && state.nSelectedDataset == 0) {
            // 20260320 ZJH MNIST 近似均匀分布
            double arrClasses[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            double arrCounts[10];
            double fAvgPerClass = static_cast<double>(state.nMnistTrainSamples) / 10.0;
            for (int i = 0; i < 10; ++i) {
                arrCounts[i] = fAvgPerClass;
            }
            ImPlot::PlotBars("训练样本", arrClasses, arrCounts, 10, 0.6);
        } else {
            // 20260320 ZJH 合成数据：每类 100 个
            double arrClasses[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            double arrCounts[10] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
            ImPlot::PlotBars("合成样本", arrClasses, arrCounts, 10, 0.6);
        }
        ImPlot::EndPlot();
    }
}

// ============================================================================
// 20260320 ZJH 绘制步骤 2：配置
// ============================================================================
static void drawStepConfig(AppState& state) {
    bool bDisabled = state.trainState.bRunning.load();  // 20260320 ZJH 训练中禁用修改
    if (bDisabled) ImGui::BeginDisabled();

    // 20260320 ZJH 模型架构
    drawSectionTitle("模型架构");

    const char* arrModels[] = {"MLP 全连接网络 (784->128->10)", "ResNet-18 卷积神经网络", "ResNet-34 卷积神经网络 (即将推出)"};
    ImGui::Combo("骨干网络", &state.nSelectedModel, arrModels, 3);
    // 20260320 ZJH ResNet-34 尚未实现，限制选择
    if (state.nSelectedModel == 2) state.nSelectedModel = 1;

    ImGui::Spacing();

    // 20260320 ZJH 模型描述
    if (state.nSelectedModel == 0) {
        ImGui::TextWrapped("MLP: 3 层全连接网络，适合低分辨率分类。参数量约 ~101K。");
    } else {
        ImGui::TextWrapped("ResNet-18: 18 层残差卷积网络，具有跳跃连接。参数量约 ~11.2M。");
    }

    ImGui::Spacing();

    // 20260320 ZJH 基本参数
    drawSectionTitle("基本参数");

    ImGui::SliderInt("训练轮数", &state.nEpochs, 1, 100);
    ImGui::SliderInt("批次大小", &state.nBatchSize, 8, 256);
    ImGui::InputFloat("学习率", &state.fLearningRate, 0.001f, 0.01f, "%.4f");
    if (state.fLearningRate < 0.0001f) state.fLearningRate = 0.0001f;
    if (state.fLearningRate > 1.0f) state.fLearningRate = 1.0f;

    ImGui::Spacing();

    // 20260320 ZJH 优化器
    drawSectionTitle("优化器");

    const char* arrOptimizers[] = {"SGD (动量)", "Adam", "AdamW (即将推出)"};
    ImGui::Combo("优化器", &state.nSelectedOptimizer, arrOptimizers, 3);
    if (state.nSelectedOptimizer == 2) state.nSelectedOptimizer = 1;  // 20260320 ZJH AdamW 尚未实现

    ImGui::Spacing();

    // 20260320 ZJH 学习率策略
    drawSectionTitle("学习率策略");

    const char* arrLRSchedules[] = {"固定", "余弦退火 (即将推出)", "预热 (即将推出)"};
    ImGui::Combo("策略", &state.nLRSchedule, arrLRSchedules, 3);
    if (state.nLRSchedule > 0) state.nLRSchedule = 0;  // 20260320 ZJH 仅固定可用

    ImGui::Spacing();

    // 20260320 ZJH 数据增强
    drawSectionTitle("数据增强");

    ImGui::Checkbox("水平翻转", &state.bAugFlip);
    ImGui::Checkbox("随机旋转", &state.bAugRotate);
    ImGui::Checkbox("色彩抖动", &state.bAugColorJitter);
    ImGui::TextDisabled("数据增强功能即将推出");

    ImGui::Spacing();

    // 20260320 ZJH 设备选择
    drawSectionTitle("设备选择");

    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        // 20260320 ZJH 构建设备选项列表
        std::vector<std::string> vecDeviceNames;
        vecDeviceNames.push_back("CPU");
        for (size_t i = 0; i < state.vecGpuDevices.size(); ++i) {
            char arrBuf[256];
            std::snprintf(arrBuf, sizeof(arrBuf), "GPU %d: %s (%zuGB)",
                          static_cast<int>(i),
                          state.vecGpuDevices[i].strName.c_str(),
                          state.vecGpuDevices[i].nTotalMemoryMB / 1024);
            vecDeviceNames.push_back(arrBuf);
        }
        // 20260320 ZJH 使用下拉框选择设备
        if (ImGui::BeginCombo("计算设备", vecDeviceNames[static_cast<size_t>(state.nSelectedDevice)].c_str())) {
            for (int i = 0; i < static_cast<int>(vecDeviceNames.size()); ++i) {
                bool bSel = (state.nSelectedDevice == i);
                if (ImGui::Selectable(vecDeviceNames[static_cast<size_t>(i)].c_str(), bSel)) {
                    state.nSelectedDevice = i;
                }
            }
            ImGui::EndCombo();
        }
        if (state.nSelectedDevice > 0) {
            ImGui::TextDisabled("GPU 训练功能即将推出，当前仅支持 CPU");
        }
    } else {
        ImGui::Text("计算设备: CPU");
        ImGui::TextDisabled("未检测到 GPU");
    }

    ImGui::Spacing();

    // 20260320 ZJH 预估信息
    drawSectionTitle("预估信息");

    int nTotalParams = (state.nSelectedModel == 0) ? 101770 : 11170000;
    float fEstMemMB = static_cast<float>(nTotalParams) * 4.0f / (1024.0f * 1024.0f) * 3.0f;  // 20260320 ZJH 参数 + 梯度 + 优化器
    ImGui::Text("参数量: %s", (state.nSelectedModel == 0) ? "~101K" : "~11.2M");
    ImGui::Text("预估显存: %.1f MB", static_cast<double>(fEstMemMB));

    int nSamples = state.bMnistAvailable ? state.nMnistTrainSamples : 1000;
    int nBatches = nSamples / state.nBatchSize;
    float fEstTimeSec = static_cast<float>(nBatches * state.nEpochs) * (state.nSelectedModel == 0 ? 0.005f : 0.1f);
    if (fEstTimeSec < 60.0f) {
        ImGui::Text("预估时间: %.0f 秒", static_cast<double>(fEstTimeSec));
    } else {
        ImGui::Text("预估时间: %.1f 分钟", static_cast<double>(fEstTimeSec) / 60.0);
    }

    if (bDisabled) ImGui::EndDisabled();
}

// ============================================================================
// 20260320 ZJH 绘制步骤 3：训练
// ============================================================================
static void drawStepTrain(AppState& state) {
    auto& ts = state.trainState;

    // 20260320 ZJH 训练控制按钮区域
    {
        float fAvailW = ImGui::GetContentRegionAvail().x;

        if (!ts.bRunning.load()) {
            // 20260320 ZJH 训练未运行：大号蓝色 [开始训练] 按钮居中
            float fBtnW = 200.0f;
            float fBtnH = 45.0f;
            ImGui::SetCursorPosX((fAvailW - fBtnW) * 0.5f);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.40f, 0.78f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.48f, 0.88f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.52f, 0.92f, 1.0f));
            if (ImGui::Button("开始训练", ImVec2(fBtnW, fBtnH))) {
                startTraining(state);
            }
            ImGui::PopStyleColor(3);
        } else {
            // 20260320 ZJH 训练中：显示暂停/恢复和停止按钮
            float fBtnW = 120.0f;
            ImGui::SetCursorPosX((fAvailW - fBtnW * 2 - 10) * 0.5f);

            if (ts.bPaused.load()) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.45f, 0.10f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.55f, 0.15f, 1.0f));
                if (ImGui::Button("恢复训练", ImVec2(fBtnW, 35))) {
                    ts.bPaused.store(false);
                }
                ImGui::PopStyleColor(2);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.45f, 0.10f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.55f, 0.15f, 1.0f));
                if (ImGui::Button("暂停", ImVec2(fBtnW, 35))) {
                    ts.bPaused.store(true);
                }
                ImGui::PopStyleColor(2);
            }

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.18f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.22f, 0.22f, 1.0f));
            if (ImGui::Button("停止", ImVec2(fBtnW, 35))) {
                ts.bStopRequested.store(true);
                ts.bPaused.store(false);
            }
            ImGui::PopStyleColor(2);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH 训练进度信息
    {
        int nCurEpoch = ts.nCurrentEpoch.load();
        int nTotEpochs = ts.nTotalEpochs.load();
        int nCurBatch = ts.nCurrentBatch.load();
        int nTotBatches = ts.nTotalBatches.load();

        // 20260320 ZJH 轮次进度条
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "轮次: %d / %d", nCurEpoch, nTotEpochs);
            ImGui::Text("%s", arrBuf);
        }
        float fEpochProg = (nTotEpochs > 0) ? static_cast<float>(nCurEpoch) / static_cast<float>(nTotEpochs) : 0.0f;
        ImGui::ProgressBar(fEpochProg, ImVec2(-1, 0));

        // 20260320 ZJH 批次进度条
        {
            char arrBuf[64];
            std::snprintf(arrBuf, sizeof(arrBuf), "批次: %d / %d", nCurBatch, nTotBatches);
            ImGui::Text("%s", arrBuf);
        }
        float fBatchProg = (nTotBatches > 0) ? static_cast<float>(nCurBatch) / static_cast<float>(nTotBatches) : 0.0f;
        ImGui::ProgressBar(fBatchProg, ImVec2(-1, 0));

        // 20260320 ZJH 预计剩余时间 (ETA)
        float fAvgBatchMs = ts.fAvgBatchTimeMs.load();
        if (fAvgBatchMs > 0.001f && ts.bRunning.load()) {
            // 20260320 ZJH 计算剩余批次数
            int nRemainingBatchesThisEpoch = nTotBatches - nCurBatch;
            int nRemainingEpochs = nTotEpochs - nCurEpoch;
            int nTotalRemainingBatches = nRemainingBatchesThisEpoch + nRemainingEpochs * nTotBatches;
            float fEtaSec = static_cast<float>(nTotalRemainingBatches) * fAvgBatchMs / 1000.0f;
            if (fEtaSec < 60.0f) {
                ImGui::Text("预计剩余时间: %.0f 秒", static_cast<double>(fEtaSec));
            } else if (fEtaSec < 3600.0f) {
                ImGui::Text("预计剩余时间: %.1f 分钟", static_cast<double>(fEtaSec) / 60.0);
            } else {
                ImGui::Text("预计剩余时间: %.1f 小时", static_cast<double>(fEtaSec) / 3600.0);
            }
        }

        // 20260320 ZJH 当前学习率
        ImGui::Text("当前学习率: %.6f", static_cast<double>(ts.fCurrentLR.load()));

        // 20260320 ZJH 训练状态指示
        if (ts.bRunning.load()) {
            if (ts.bPaused.load()) {
                ImGui::TextColored(s_colOrange, "状态: 已暂停");
            } else {
                ImGui::TextColored(s_colGreen, "状态: 训练中...");
            }
        } else if (ts.bCompleted.load()) {
            ImGui::TextColored(s_colAccentLight, "状态: 已完成");
        } else {
            ImGui::TextColored(s_colGray, "状态: 空闲");
        }
    }

    ImGui::Spacing();

    // 20260320 ZJH 两个并排图表：损失 + 准确率
    float fChartHeight = ImGui::GetContentRegionAvail().y * 0.45f;
    if (fChartHeight < 150.0f) fChartHeight = 150.0f;

    ImGui::BeginChild("TrainCharts", ImVec2(0, fChartHeight));
    {
        float fHalfWidth = (ImGui::GetContentRegionAvail().x - 8.0f) * 0.5f;

        // 20260320 ZJH 损失曲线图
        ImGui::BeginChild("LossChart", ImVec2(fHalfWidth, 0));
        {
            std::lock_guard<std::mutex> lock(ts.mutex);
            if (ImPlot::BeginPlot("损失曲线", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("轮次", "损失");
                if (!ts.vecLossHistory.empty()) {
                    ImPlot::PlotLine("训练损失",
                                     ts.vecLossHistory.data(),
                                     static_cast<int>(ts.vecLossHistory.size()));
                }
                ImPlot::EndPlot();
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // 20260320 ZJH 准确率曲线图
        ImGui::BeginChild("AccChart", ImVec2(0, 0));
        {
            std::lock_guard<std::mutex> lock(ts.mutex);
            if (ImPlot::BeginPlot("准确率曲线", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("轮次", "准确率 (%)");
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 105, ImPlotCond_Always);
                if (!ts.vecTrainAccHistory.empty()) {
                    ImPlot::PlotLine("训练准确率",
                                     ts.vecTrainAccHistory.data(),
                                     static_cast<int>(ts.vecTrainAccHistory.size()));
                }
                if (!ts.vecTestAccHistory.empty()) {
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

    // 20260320 ZJH 训练日志
    drawSectionTitle("训练日志");
    ImGui::BeginChild("TrainLog", ImVec2(0, 0), ImGuiChildFlags_Borders);
    {
        std::lock_guard<std::mutex> lock(ts.mutex);
        ImGui::TextUnformatted(ts.strLog.c_str());
        // 20260319 ZJH 自动滚动到底部
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) {
            ImGui::SetScrollHereY(1.0f);
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260320 ZJH 绘制步骤 4：评估
// ============================================================================
static void drawStepEvaluate(AppState& state) {
    auto& ts = state.trainState;

    if (!ts.bCompleted.load()) {
        // 20260320 ZJH 训练未完成时显示提示
        ImGui::Spacing();
        ImGui::Spacing();
        float fAvailW = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(fAvailW * 0.3f);
        ImGui::TextColored(s_colGray, "请先完成训练以查看评估结果");
        return;
    }

    // 20260320 ZJH 模型性能概览
    drawSectionTitle("模型性能概览");

    // 20260320 ZJH 大字显示测试准确率
    {
        float fTestAcc = ts.fTestAcc.load();
        char arrBuf[64];
        std::snprintf(arrBuf, sizeof(arrBuf), "%.2f%%", static_cast<double>(fTestAcc));

        ImGui::PushFont(ImGui::GetFont());
        auto* pDrawList = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float fFontSize = 32.0f;
        pDrawList->AddText(ImGui::GetFont(), fFontSize, pos,
                          IM_COL32(100, 200, 255, 255), arrBuf);
        ImGui::Dummy(ImVec2(0, fFontSize + 4));
        ImGui::PopFont();
    }
    ImGui::Text("测试准确率");

    ImGui::Spacing();

    // 20260320 ZJH 训练统计表
    {
        std::lock_guard<std::mutex> lock(ts.mutex);
        ImGui::Text("训练轮数: %d", ts.nTotalEpochs.load());
        ImGui::Text("总耗时: %.1f 秒", static_cast<double>(ts.fTotalTrainingTimeSec));
        ImGui::Text("最佳验证损失: %.4f", static_cast<double>(ts.fBestValLoss));
        ImGui::Text("最终训练准确率: %.2f%%", static_cast<double>(ts.fTrainAcc.load()));
        if (!ts.strSavedModelPath.empty()) {
            ImGui::Text("模型路径: %s", ts.strSavedModelPath.c_str());
        }
    }

    ImGui::Spacing();

    // 20260320 ZJH 混淆矩阵
    drawSectionTitle("混淆矩阵");

    {
        std::lock_guard<std::mutex> lock(ts.mutex);
        if (ts.bHasConfusionMatrix) {
            // 20260320 ZJH 将混淆矩阵转为 double 数组供 ImPlot 使用
            // ImPlot PlotHeatmap 需要一维数组，行优先
            static double arrHeatmapData[100];
            double fMaxVal = 0.0;
            for (int r = 0; r < 10; ++r) {
                for (int c = 0; c < 10; ++c) {
                    double v = static_cast<double>(ts.arrConfusionMatrix[static_cast<size_t>(r)][static_cast<size_t>(c)]);
                    arrHeatmapData[r * 10 + c] = v;
                    if (v > fMaxVal) fMaxVal = v;
                }
            }

            // 20260320 ZJH 使用 ImPlot 绘制热力图
            static const char* arrLabels[] = {"0","1","2","3","4","5","6","7","8","9"};
            if (ImPlot::BeginPlot("##ConfMatrix", ImVec2(350, 350))) {
                ImPlot::SetupAxes("预测类别", "实际类别");
                ImPlot::SetupAxisTicks(ImAxis_X1, 0, 9, 10, arrLabels);
                ImPlot::SetupAxisTicks(ImAxis_Y1, 0, 9, 10, arrLabels);
                ImPlot::PlotHeatmap("##heatmap", arrHeatmapData, 10, 10,
                                    0, fMaxVal, "%g",
                                    ImPlotPoint(0, 0), ImPlotPoint(10, 10));
                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextDisabled("混淆矩阵数据不可用");
        }
    }

    ImGui::Spacing();

    // 20260320 ZJH 分类报告表格
    drawSectionTitle("分类报告");

    {
        std::lock_guard<std::mutex> lock(ts.mutex);
        if (ts.bHasConfusionMatrix) {
            if (ImGui::BeginTable("ClassReport", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("类别");
                ImGui::TableSetupColumn("精确率 (P)");
                ImGui::TableSetupColumn("召回率 (R)");
                ImGui::TableSetupColumn("F1 分数");
                ImGui::TableHeadersRow();

                for (int c = 0; c < 10; ++c) {
                    // 20260320 ZJH 计算该类别的 TP, FP, FN
                    int nTP = ts.arrConfusionMatrix[static_cast<size_t>(c)][static_cast<size_t>(c)];
                    int nFP = 0, nFN = 0;
                    for (int i = 0; i < 10; ++i) {
                        if (i != c) {
                            nFP += ts.arrConfusionMatrix[static_cast<size_t>(i)][static_cast<size_t>(c)];  // 20260320 ZJH 其他类被预测为 c
                            nFN += ts.arrConfusionMatrix[static_cast<size_t>(c)][static_cast<size_t>(i)];  // 20260320 ZJH c 被预测为其他类
                        }
                    }
                    float fPrecision = (nTP + nFP > 0) ? static_cast<float>(nTP) / static_cast<float>(nTP + nFP) : 0.0f;
                    float fRecall = (nTP + nFN > 0) ? static_cast<float>(nTP) / static_cast<float>(nTP + nFN) : 0.0f;
                    float fF1 = (fPrecision + fRecall > 0.0f) ? 2.0f * fPrecision * fRecall / (fPrecision + fRecall) : 0.0f;

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%d", c);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.3f", static_cast<double>(fPrecision));
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.3f", static_cast<double>(fRecall));
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.3f", static_cast<double>(fF1));
                }

                ImGui::EndTable();
            }
        } else {
            ImGui::TextDisabled("分类报告数据不可用");
        }
    }

    ImGui::Spacing();

    // 20260320 ZJH 导出按钮
    drawSectionTitle("模型导出");

    if (ImGui::Button("导出模型 (.dfm)", ImVec2(180, 32))) {
        // 20260320 ZJH 模型已在训练完成时自动保存
    }
    ImGui::SameLine();
    if (ImGui::Button("导出 ONNX (即将推出)", ImVec2(200, 32))) {
        // 20260320 ZJH 占位
    }
}

// ============================================================================
// 20260320 ZJH 绘制占位任务页面（检测/分割/异常检测即将推出）
// ============================================================================
static void drawComingSoonTask(const char* strTaskName, const char* strDescription) {
    ImGui::Spacing();
    ImGui::Spacing();

    float fAvailW = ImGui::GetContentRegionAvail().x;
    float fAvailH = ImGui::GetContentRegionAvail().y;
    float fCardW = 400.0f;
    float fCardH = 250.0f;

    // 20260320 ZJH 居中绘制卡片
    ImGui::SetCursorPosX((fAvailW - fCardW) * 0.5f);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + fAvailH * 0.15f);

    ImGui::BeginChild("##ComingSoon", ImVec2(fCardW, fCardH), ImGuiChildFlags_Borders);
    {
        ImGui::Spacing();
        ImGui::Spacing();

        // 20260320 ZJH 任务名称（大字）
        auto* pDrawList = ImGui::GetWindowDrawList();
        ImVec2 posStart = ImGui::GetCursorScreenPos();
        float fTitleSize = 28.0f;
        ImVec2 titleSize = ImGui::GetFont()->CalcTextSizeA(fTitleSize, FLT_MAX, 0.0f, strTaskName);
        float fTitleX = posStart.x + (fCardW - titleSize.x) * 0.5f;
        pDrawList->AddText(ImGui::GetFont(), fTitleSize,
                          ImVec2(fTitleX, posStart.y),
                          IM_COL32(180, 200, 240, 255), strTaskName);
        ImGui::Dummy(ImVec2(0, fTitleSize + 10));

        // 20260320 ZJH "即将推出" 标签
        ImGui::Spacing();
        float fBadgeW = 120.0f;
        ImGui::SetCursorPosX((fCardW - fBadgeW) * 0.5f);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.60f, 0.40f, 0.10f, 1.0f));
        ImGui::Button("即将推出", ImVec2(fBadgeW, 28));
        ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::Spacing();

        // 20260320 ZJH 描述文字
        ImGui::SetCursorPosX(20);
        ImGui::PushTextWrapPos(fCardW - 20);
        ImGui::TextColored(s_colGray, "%s", strDescription);
        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260320 ZJH 绘制右侧属性面板（根据当前步骤显示上下文信息）
// ============================================================================
static void drawPropertiesPanel(AppState& state) {
    auto& ts = state.trainState;

    drawSectionTitle("属性");

    // 20260320 ZJH 根据当前步骤显示不同内容
    switch (state.nActiveStep) {
        case 0: {
            // 20260320 ZJH 步骤 1：数据集详情
            ImGui::TextColored(s_colAccentLight, "数据集信息");
            ImGui::Separator();
            ImGui::Spacing();

            const char* arrDatasetNames[] = {"MNIST", "合成数据", "自定义"};
            ImGui::Text("当前: %s", arrDatasetNames[state.nSelectedDataset]);

            if (state.nSelectedDataset == 0 && state.bMnistAvailable) {
                ImGui::Text("训练: %d", state.nMnistTrainSamples);
                ImGui::Text("测试: %d", state.nMnistTestSamples);
                ImGui::Text("类别: 10");
                ImGui::Text("尺寸: 28x28");
                ImGui::Text("通道: 灰度");
            } else if (state.nSelectedDataset == 1) {
                ImGui::Text("训练: 1000");
                ImGui::Text("测试: 200");
                ImGui::Text("类别: 10");
            }
            break;
        }
        case 1: {
            // 20260320 ZJH 步骤 2：模型架构信息
            ImGui::TextColored(s_colAccentLight, "模型信息");
            ImGui::Separator();
            ImGui::Spacing();

            if (state.nSelectedModel == 0) {
                ImGui::TextWrapped("MLP 全连接网络");
                ImGui::Spacing();
                ImGui::TextDisabled("  输入: 784");
                ImGui::TextDisabled("  隐藏: 128 + ReLU");
                ImGui::TextDisabled("  输出: 10");
                ImGui::Spacing();
                ImGui::Text("参数: ~101K");
            } else {
                ImGui::TextWrapped("ResNet-18");
                ImGui::Spacing();
                ImGui::TextDisabled("  Conv 3x3 -> 64");
                ImGui::TextDisabled("  Layer1: 64x2");
                ImGui::TextDisabled("  Layer2: 128x2");
                ImGui::TextDisabled("  Layer3: 256x2");
                ImGui::TextDisabled("  Layer4: 512x2");
                ImGui::TextDisabled("  AvgPool -> FC 10");
                ImGui::Spacing();
                ImGui::Text("参数: ~11.2M");
            }
            break;
        }
        case 2: {
            // 20260320 ZJH 步骤 3：实时训练统计
            ImGui::TextColored(s_colAccentLight, "实时统计");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("损失: %.4f", static_cast<double>(ts.fCurrentLoss.load()));
            ImGui::Text("训练准确率: %.2f%%", static_cast<double>(ts.fTrainAcc.load()));
            ImGui::Text("测试准确率: %.2f%%", static_cast<double>(ts.fTestAcc.load()));
            ImGui::Text("学习率: %.6f", static_cast<double>(ts.fCurrentLR.load()));

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // 20260320 ZJH 轮次/批次
            ImGui::Text("轮次: %d/%d", ts.nCurrentEpoch.load(), ts.nTotalEpochs.load());
            ImGui::Text("批次: %d/%d", ts.nCurrentBatch.load(), ts.nTotalBatches.load());

            // 20260320 ZJH ETA
            float fAvgMs = ts.fAvgBatchTimeMs.load();
            if (fAvgMs > 0.001f) {
                ImGui::Text("批次耗时: %.1f ms", static_cast<double>(fAvgMs));
            }

            break;
        }
        case 3: {
            // 20260320 ZJH 步骤 4：评估摘要
            ImGui::TextColored(s_colAccentLight, "评估摘要");
            ImGui::Separator();
            ImGui::Spacing();

            if (ts.bCompleted.load()) {
                ImGui::Text("测试准确率: %.2f%%", static_cast<double>(ts.fTestAcc.load()));
                ImGui::Text("训练准确率: %.2f%%", static_cast<double>(ts.fTrainAcc.load()));
                {
                    std::lock_guard<std::mutex> lock(ts.mutex);
                    ImGui::Text("最佳损失: %.4f", static_cast<double>(ts.fBestValLoss));
                    ImGui::Text("耗时: %.1fs", static_cast<double>(ts.fTotalTrainingTimeSec));
                }
            } else {
                ImGui::TextDisabled("训练未完成");
            }
            break;
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH GPU 信息（始终显示）
    ImGui::TextColored(s_colAccentLight, "GPU 信息");
    ImGui::Separator();
    ImGui::Spacing();

    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        for (size_t i = 0; i < state.vecGpuDevices.size(); ++i) {
            auto& gpu = state.vecGpuDevices[i];
            ImGui::Text("GPU %d:", static_cast<int>(i));
            ImGui::TextWrapped("  %s", gpu.strName.c_str());
            ImGui::Text("  显存: %zu MB", gpu.nTotalMemoryMB);
            ImGui::Text("  算力: %d.%d", gpu.nComputeCapMajor, gpu.nComputeCapMinor);
        }
        // 20260320 ZJH 显存使用（如果有数据）
        if (state.nGpuMemTotalMB > 0) {
            ImGui::Spacing();
            ImGui::Text("显存使用: %zu/%zu MB", state.nGpuMemUsedMB, state.nGpuMemTotalMB);
        }
    } else {
        ImGui::TextDisabled("无 NVIDIA GPU");
    }
}

// ============================================================================
// 20260320 ZJH 绘制状态栏（底部）
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
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.07f, 0.07f, 0.09f, 1.0f));

    ImGui::Begin("##StatusBar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav);

    // 20260320 ZJH 状态文本
    if (ts.bRunning.load()) {
        if (ts.bPaused.load()) {
            ImGui::TextColored(s_colOrange, "状态: 训练已暂停");
        } else {
            char arrBuf[128];
            std::snprintf(arrBuf, sizeof(arrBuf), "状态: 训练中 (轮次 %d/%d, 批次 %d/%d)",
                          ts.nCurrentEpoch.load(), ts.nTotalEpochs.load(),
                          ts.nCurrentBatch.load(), ts.nTotalBatches.load());
            ImGui::TextColored(s_colGreen, "%s", arrBuf);
        }
    } else if (ts.bCompleted.load()) {
        ImGui::TextColored(s_colAccentLight, "状态: 训练已完成");
    } else {
        ImGui::Text("状态: 就绪");
    }

    // 20260320 ZJH GPU 信息
    ImGui::SameLine(400);
    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        auto& gpu = state.vecGpuDevices[0];
        char arrBuf[256];
        if (state.nGpuMemTotalMB > 0) {
            std::snprintf(arrBuf, sizeof(arrBuf), "GPU: %s %zuGB | 显存: %zu/%zu MB",
                          gpu.strName.c_str(), gpu.nTotalMemoryMB / 1024,
                          state.nGpuMemUsedMB, state.nGpuMemTotalMB);
        } else {
            std::snprintf(arrBuf, sizeof(arrBuf), "GPU: %s %zuGB",
                          gpu.strName.c_str(), gpu.nTotalMemoryMB / 1024);
        }
        ImGui::TextColored(s_colGreen, "%s", arrBuf);
    } else {
        ImGui::TextColored(s_colOrange, "设备: CPU");
    }

    // 20260320 ZJH 版本信息
    float fVerWidth = ImGui::CalcTextSize("DeepForge v0.1.0").x;
    ImGui::SameLine(ImGui::GetWindowWidth() - fVerWidth - 15);
    ImGui::Text("DeepForge v0.1.0");

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

    // 20260320 ZJH 创建主窗口（1280x720，可调整大小，最大化）
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
    const char* arrFontPaths[] = {
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc"
    };

    bool bFontLoaded = false;
    for (const char* pFontPath : arrFontPaths) {
        if (std::filesystem::exists(pFontPath)) {
            ImFontConfig fontConfig;
            fontConfig.MergeMode = false;
            io.Fonts->AddFontFromFileTTF(pFontPath, 16.0f, &fontConfig,
                                          io.Fonts->GetGlyphRangesChineseFull());
            bFontLoaded = true;
            break;
        }
    }

    if (!bFontLoaded) {
        io.Fonts->AddFontDefault();
    }

    // 20260319 ZJH 初始化 SDL3 后端
    ImGui_ImplSDL3_InitForSDLRenderer(pWindow, pRenderer);
    ImGui_ImplSDLRenderer3_Init(pRenderer);

    // ===== 第三步：闪屏动画 =====
    // 20260320 ZJH 启动闪屏，显示 2.5 秒后进入主界面
    {
        auto splashStart = std::chrono::steady_clock::now();
        const float fSplashDuration = 2.5f;
        bool bSplashRunning = true;

        while (bSplashRunning) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL3_ProcessEvent(&event);
                if (event.type == SDL_EVENT_QUIT) {
                    bSplashRunning = false;
                    break;
                }
                if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                    event.window.windowID == SDL_GetWindowID(pWindow)) {
                    bSplashRunning = false;
                    break;
                }
            }
            if (!bSplashRunning) break;

            float fElapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - splashStart).count();
            if (fElapsed > fSplashDuration) break;

            float fAlpha = std::min(fElapsed / 0.8f, 1.0f);

            ImGui_ImplSDLRenderer3_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.08f, 1.0f));
            ImGui::Begin("##Splash", nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoDocking |
                         ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoCollapse);

            auto* pDrawList = ImGui::GetWindowDrawList();
            float fCenterX = io.DisplaySize.x * 0.5f;
            float fCenterY = io.DisplaySize.y * 0.5f;

            // 20260320 ZJH 背景渐变
            ImU32 nTopColor = IM_COL32(15, 15, 20, static_cast<int>(255 * fAlpha));
            ImU32 nBotColor = IM_COL32(30, 30, 45, static_cast<int>(255 * fAlpha));
            pDrawList->AddRectFilledMultiColor(
                ImVec2(0, 0), io.DisplaySize,
                nTopColor, nTopColor, nBotColor, nBotColor);

            // 20260320 ZJH DF Logo
            float fLogoSize = 80.0f;
            float fLogoX = fCenterX - fLogoSize * 0.5f;
            float fLogoY = fCenterY - 120.0f;
            pDrawList->AddRectFilled(
                ImVec2(fLogoX, fLogoY),
                ImVec2(fLogoX + fLogoSize, fLogoY + fLogoSize),
                IM_COL32(50, 120, 220, static_cast<int>(240 * fAlpha)),
                12.0f);
            ImFont* pFont = ImGui::GetFont();
            float fDfFontSize = 42.0f;
            const char* strDf = "DF";
            ImVec2 dfTextSize = pFont->CalcTextSizeA(fDfFontSize, FLT_MAX, 0.0f, strDf);
            pDrawList->AddText(pFont, fDfFontSize,
                ImVec2(fLogoX + (fLogoSize - dfTextSize.x) * 0.5f,
                       fLogoY + (fLogoSize - dfTextSize.y) * 0.5f),
                IM_COL32(255, 255, 255, static_cast<int>(255 * fAlpha)),
                strDf);

            // 20260320 ZJH 标题
            float fTitleFontSize = 36.0f;
            const char* strTitle = "DeepForge";
            ImVec2 titleSize = pFont->CalcTextSizeA(fTitleFontSize, FLT_MAX, 0.0f, strTitle);
            pDrawList->AddText(pFont, fTitleFontSize,
                ImVec2(fCenterX - titleSize.x * 0.5f, fLogoY + fLogoSize + 20.0f),
                IM_COL32(230, 230, 240, static_cast<int>(255 * fAlpha)),
                strTitle);

            // 20260320 ZJH 副标题
            float fSubFontSize = 18.0f;
            const char* strSubtitle = "纯 C++ 全流程深度学习视觉平台";
            ImVec2 subSize = pFont->CalcTextSizeA(fSubFontSize, FLT_MAX, 0.0f, strSubtitle);
            pDrawList->AddText(pFont, fSubFontSize,
                ImVec2(fCenterX - subSize.x * 0.5f, fLogoY + fLogoSize + 64.0f),
                IM_COL32(160, 180, 220, static_cast<int>(220 * fAlpha)),
                strSubtitle);

            // 20260320 ZJH 版本号
            float fVerFontSize = 14.0f;
            const char* strVersion = "v0.1.0";
            ImVec2 verSize = pFont->CalcTextSizeA(fVerFontSize, FLT_MAX, 0.0f, strVersion);
            pDrawList->AddText(pFont, fVerFontSize,
                ImVec2(fCenterX - verSize.x * 0.5f, fLogoY + fLogoSize + 90.0f),
                IM_COL32(120, 140, 170, static_cast<int>(200 * fAlpha)),
                strVersion);

            // 20260320 ZJH 旋转加载动画
            int nDotCount = 8;
            float fDotPhase = fElapsed * 3.0f;
            float fDotRadius = 20.0f;
            float fDotCenterY = fLogoY + fLogoSize + 130.0f;
            for (int i = 0; i < nDotCount; ++i) {
                float fAngle = fDotPhase + static_cast<float>(i) * (2.0f * 3.14159265f / static_cast<float>(nDotCount));
                float fDotX = fCenterX + std::cos(fAngle) * fDotRadius;
                float fDotY = fDotCenterY + std::sin(fAngle) * fDotRadius;
                float fDotAlpha = (std::sin(fAngle - fDotPhase) + 1.0f) * 0.5f;
                float fDotSize = 3.0f + fDotAlpha * 2.0f;
                pDrawList->AddCircleFilled(ImVec2(fDotX, fDotY), fDotSize,
                    IM_COL32(100, 160, 255, static_cast<int>(fDotAlpha * 255.0f * fAlpha)));
            }

            // 20260320 ZJH 初始化提示
            float fInitFontSize = 14.0f;
            const char* strInit = "正在初始化...";
            ImVec2 initSize = pFont->CalcTextSizeA(fInitFontSize, FLT_MAX, 0.0f, strInit);
            pDrawList->AddText(pFont, fInitFontSize,
                ImVec2(fCenterX - initSize.x * 0.5f, fDotCenterY + fDotRadius + 20.0f),
                IM_COL32(140, 150, 180, static_cast<int>(180 * fAlpha)),
                strInit);

            // 20260320 ZJH 版权信息
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

            ImGui::Render();
            SDL_SetRenderDrawColor(pRenderer, 15, 15, 20, 255);
            SDL_RenderClear(pRenderer);
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer);
            SDL_RenderPresent(pRenderer);
        }

        // 20260320 ZJH 如果用户在闪屏阶段关闭窗口
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

    // 20260320 ZJH 启动时检测 GPU
    appState.vecGpuDevices = detectGpuDevices();
    appState.bGpuDetected = !appState.vecGpuDevices.empty();

    // ===== 第五步：主循环 =====
    bool bRunning = true;
    while (bRunning) {
        // 20260319 ZJH 处理 SDL 事件
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                bRunning = false;
            }
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                event.window.windowID == SDL_GetWindowID(pWindow)) {
                bRunning = false;
            }
        }

        // 20260320 ZJH 定期查询 GPU 显存使用情况（每 5 秒一次）
        if (appState.bGpuDetected) {
            appState.fGpuMemQueryTimer += io.DeltaTime;
            if (appState.fGpuMemQueryTimer > 5.0f) {
                appState.fGpuMemQueryTimer = 0.0f;
                auto [nUsed, nTotal] = queryGpuMemoryUsage();
                appState.nGpuMemUsedMB = nUsed;
                appState.nGpuMemTotalMB = nTotal;
            }
        }

        // 20260319 ZJH 开始 ImGui 新帧
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // 20260320 ZJH 创建全屏主窗口（扣除状态栏高度）
        ImGuiViewport* pViewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(pViewport->WorkPos);
        ImGui::SetNextWindowSize(ImVec2(pViewport->WorkSize.x, pViewport->WorkSize.y - 28.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("##MainDock", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::PopStyleVar();

        // ============================================================
        // 20260320 ZJH 顶部标题栏：应用名称 + GPU 信息
        // ============================================================
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
        {
            // 20260320 ZJH 左侧应用标题
            ImGui::TextColored(s_colAccentLight, "DeepForge 深度学习平台 v0.1.0");

            // 20260320 ZJH 右侧 GPU 信息（可点击）
            float fGpuTextWidth = 300.0f;
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - fGpuTextWidth + 12);

            if (appState.bGpuDetected && !appState.vecGpuDevices.empty()) {
                auto& gpu = appState.vecGpuDevices[0];
                char arrGpuLabel[256];
                std::snprintf(arrGpuLabel, sizeof(arrGpuLabel), "GPU: %s (%zuGB)",
                              gpu.strName.c_str(), gpu.nTotalMemoryMB / 1024);

                // 20260320 ZJH 绿色圆点 + GPU 信息
                auto* pDrawList = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                pDrawList->AddCircleFilled(ImVec2(pos.x + 4, pos.y + 10), 5.0f,
                                          IM_COL32(50, 220, 80, 255));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 14);

                if (ImGui::SmallButton(arrGpuLabel)) {
                    appState.bShowDevicePopup = true;  // 20260320 ZJH 打开设备选择弹窗
                }
            } else {
                // 20260320 ZJH 橙色圆点 + CPU
                auto* pDrawList = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                pDrawList->AddCircleFilled(ImVec2(pos.x + 4, pos.y + 10), 5.0f,
                                          IM_COL32(255, 165, 30, 255));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 14);

                if (ImGui::SmallButton("设备: CPU")) {
                    appState.bShowDevicePopup = true;
                }
            }
        }
        ImGui::PopStyleVar();

        ImGui::Separator();

        // 20260320 ZJH 设备选择弹窗
        if (appState.bShowDevicePopup) {
            ImGui::OpenPopup("设备选择");
            appState.bShowDevicePopup = false;
        }
        if (ImGui::BeginPopup("设备选择")) {
            ImGui::TextColored(s_colAccentLight, "可用计算设备");
            ImGui::Separator();
            ImGui::Spacing();

            // 20260320 ZJH CPU 选项
            bool bCpuSel = (appState.nSelectedDevice == 0);
            if (ImGui::Selectable("CPU", bCpuSel)) {
                appState.nSelectedDevice = 0;
            }

            // 20260320 ZJH GPU 选项
            for (int i = 0; i < static_cast<int>(appState.vecGpuDevices.size()); ++i) {
                auto& gpu = appState.vecGpuDevices[static_cast<size_t>(i)];
                char arrBuf[256];
                std::snprintf(arrBuf, sizeof(arrBuf), "GPU %d: %s (%zuMB, 算力 %d.%d)",
                              i, gpu.strName.c_str(), gpu.nTotalMemoryMB,
                              gpu.nComputeCapMajor, gpu.nComputeCapMinor);
                bool bGpuSel = (appState.nSelectedDevice == i + 1);
                if (ImGui::Selectable(arrBuf, bGpuSel)) {
                    appState.nSelectedDevice = i + 1;
                }
            }

            if (appState.vecGpuDevices.empty()) {
                ImGui::TextDisabled("未检测到 NVIDIA GPU");
            }

            ImGui::Spacing();
            ImGui::TextDisabled("GPU 训练功能即将推出");
            ImGui::EndPopup();
        }

        // ============================================================
        // 20260320 ZJH 三栏布局：左侧任务导航 | 中间主内容 | 右侧属性面板
        // ============================================================
        float fSidebarWidth = 160.0f;     // 20260320 ZJH 左侧导航宽度
        float fPropertiesWidth = 200.0f;  // 20260320 ZJH 右侧属性面板宽度
        float fContentHeight = ImGui::GetContentRegionAvail().y;

        // 20260320 ZJH 左侧任务导航栏
        ImGui::BeginChild("##Sidebar", ImVec2(fSidebarWidth, fContentHeight), ImGuiChildFlags_Borders);
        drawTaskSidebar(appState);
        ImGui::EndChild();

        ImGui::SameLine();

        // 20260320 ZJH 中间主内容区域
        float fMainWidth = ImGui::GetContentRegionAvail().x - fPropertiesWidth - 4;
        ImGui::BeginChild("##MainContent", ImVec2(fMainWidth, fContentHeight), ImGuiChildFlags_Borders);
        {
            // 20260320 ZJH 根据任务类型显示不同内容
            if (appState.nActiveTask == 0) {
                // 20260320 ZJH 图像分类 — 完整功能
                // 步骤标签页
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(16, 6));
                {
                    const char* arrSteps[] = {"1.数据", "2.配置", "3.训练", "4.评估"};
                    for (int i = 0; i < 4; ++i) {
                        if (i > 0) ImGui::SameLine();
                        bool bStepSel = (appState.nActiveStep == i);
                        if (bStepSel) {
                            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.35f, 0.60f, 1.0f));
                        }
                        if (ImGui::Button(arrSteps[i], ImVec2(0, 0))) {
                            appState.nActiveStep = i;
                        }
                        if (bStepSel) {
                            ImGui::PopStyleColor();
                        }
                    }
                }
                ImGui::PopStyleVar();

                ImGui::Separator();

                // 20260320 ZJH 步骤内容
                ImGui::BeginChild("##StepContent", ImVec2(0, 0));
                {
                    switch (appState.nActiveStep) {
                        case 0: drawStepData(appState);     break;
                        case 1: drawStepConfig(appState);   break;
                        case 2: drawStepTrain(appState);    break;
                        case 3: drawStepEvaluate(appState); break;
                    }
                }
                ImGui::EndChild();

            } else if (appState.nActiveTask == 1) {
                drawComingSoonTask("目标检测",
                    "基于 YOLOv5 的实时目标检测。支持自定义数据集训练、多类别检测、NMS 后处理。"
                    "预计支持 COCO 格式标注导入和模型导出。");
            } else if (appState.nActiveTask == 2) {
                drawComingSoonTask("语义分割",
                    "基于 U-Net 的像素级语义分割。支持二值分割和多类分割。"
                    "适用于缺陷检测、医学图像分析等工业场景。");
            } else if (appState.nActiveTask == 3) {
                drawComingSoonTask("异常检测",
                    "基于 AutoEncoder 的无监督异常检测。仅需正常样本训练，"
                    "通过重建误差检测异常区域。适用于表面缺陷检测、质量控制。");
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // 20260320 ZJH 右侧属性面板
        ImGui::BeginChild("##Properties", ImVec2(0, fContentHeight), ImGuiChildFlags_Borders);
        {
            if (appState.nActiveTask == 0) {
                drawPropertiesPanel(appState);
            } else {
                drawSectionTitle("属性");
                ImGui::TextDisabled("选择可用任务");
                ImGui::TextDisabled("查看详细信息");
            }
        }
        ImGui::EndChild();

        ImGui::End();  // 20260320 ZJH 结束 MainDock

        // 20260320 ZJH 绘制状态栏
        drawStatusBar(appState);

        // 20260320 ZJH 渲染
        ImGui::Render();
        SDL_SetRenderDrawColor(pRenderer, 25, 25, 28, 255);
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

    return 0;  // 20260320 ZJH 正常退出
}
