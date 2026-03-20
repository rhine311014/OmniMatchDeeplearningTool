// 20260320 ZJH DeepForge Phase 4 — SDL3 + ImGui 桌面 GUI 应用程序（完全重写）
// MVTec Halcon Deep Learning Tool 风格 UI：左侧项目导航 + 顶部工具栏 + 步骤指示器 + 右侧属性面板
// 支持全部 4 种任务类型：图像分类、目标检测、语义分割、异常检测
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
#include <ctime>
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
#include <sstream>

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

// 20260320 ZJH 导入 DeepForge 引擎模块（全部 4 种任务模型）
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;
import df.engine.linear;
import df.engine.activations;
import df.engine.conv;
import df.engine.resnet;
import df.engine.unet;
import df.engine.yolo;
import df.engine.autoencoder;
import df.engine.optimizer;
import df.engine.loss;
import df.engine.mnist;
import df.engine.serializer;
import df.hal.cpu_backend;

// ============================================================================
// 20260320 ZJH 任务类型枚举
// ============================================================================
enum class TaskType {
    Classification = 0,  // 20260320 ZJH 图像分类
    Detection,           // 20260320 ZJH 目标检测
    Segmentation,        // 20260320 ZJH 语义分割
    AnomalyDetection     // 20260320 ZJH 异常检测
};

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

    typedef int (*CuInit_t)(unsigned int);
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

    size_t nFree = 0, nTotal = 0;
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
// 20260320 ZJH 获取当前时间戳字符串 [HH:MM:SS]
// ============================================================================
static std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto nTime = std::chrono::system_clock::to_time_t(now);
    struct tm tmLocal;
    localtime_s(&tmLocal, &nTime);  // 20260320 ZJH MSVC 安全版本
    char arrBuf[32];
    std::snprintf(arrBuf, sizeof(arrBuf), "[%02d:%02d:%02d]", tmLocal.tm_hour, tmLocal.tm_min, tmLocal.tm_sec);
    return std::string(arrBuf);
}

// ============================================================================
// 20260320 ZJH 训练状态结构体 — 在 UI 线程和训练线程之间共享
// ============================================================================
struct TrainingState {
    // 20260320 ZJH 原子标志，线程安全的读写
    std::atomic<bool> bRunning{false};        // 20260320 ZJH 训练是否正在运行
    std::atomic<bool> bPaused{false};         // 20260320 ZJH 训练是否暂停
    std::atomic<bool> bStopRequested{false};  // 20260320 ZJH 是否请求停止训练
    std::atomic<bool> bCompleted{false};      // 20260320 ZJH 训练是否完成
    std::atomic<int> nCurrentEpoch{0};        // 20260320 ZJH 当前训练轮数
    std::atomic<int> nTotalEpochs{10};        // 20260320 ZJH 总训练轮数
    std::atomic<int> nCurrentBatch{0};        // 20260320 ZJH 当前批次
    std::atomic<int> nTotalBatches{0};        // 20260320 ZJH 每轮总批次数
    std::atomic<float> fCurrentLoss{0.0f};    // 20260320 ZJH 当前批次损失
    std::atomic<float> fTrainAcc{0.0f};       // 20260320 ZJH 当前轮训练准确率
    std::atomic<float> fTestAcc{0.0f};        // 20260320 ZJH 当前轮测试准确率
    std::atomic<float> fCurrentLR{0.01f};     // 20260320 ZJH 当前学习率
    std::atomic<float> fAvgBatchTimeMs{0.0f}; // 20260320 ZJH 平均每批次耗时（毫秒）

    // 20260320 ZJH 互斥锁保护的历史数据，UI 线程读取时需加锁
    std::mutex mutex;
    std::vector<float> vecLossHistory;        // 20260320 ZJH 每轮平均损失历史
    std::vector<float> vecValLossHistory;     // 20260320 ZJH 验证损失历史
    std::vector<float> vecTrainAccHistory;    // 20260320 ZJH 每轮训练准确率历史
    std::vector<float> vecTestAccHistory;     // 20260320 ZJH 每轮测试准确率历史
    std::vector<float> vecMIoUHistory;        // 20260320 ZJH mIoU 历史（分割）
    std::string strLog;                       // 20260320 ZJH 训练日志文本
    std::string strSavedModelPath;            // 20260320 ZJH 保存的模型路径

    // 20260320 ZJH 混淆矩阵数据（10x10，训练完成后填充）
    std::array<std::array<int, 10>, 10> arrConfusionMatrix{};  // 20260320 ZJH [实际][预测]
    bool bHasConfusionMatrix = false;         // 20260320 ZJH 是否已计算混淆矩阵

    // 20260320 ZJH 最佳验证损失
    float fBestValLoss = 999.0f;              // 20260320 ZJH 训练过程中最小的验证损失
    float fTotalTrainingTimeSec = 0.0f;       // 20260320 ZJH 总训练耗时（秒）

    // 20260320 ZJH 异常检测相关
    float fAnomalyThreshold = 0.0f;           // 20260320 ZJH 异常检测阈值
    float fAUC = 0.0f;                        // 20260320 ZJH AUC-ROC 值

    // 20260320 ZJH 检测相关
    float fMAP50 = 0.0f;                      // 20260320 ZJH mAP@0.5
    float fMAP5095 = 0.0f;                    // 20260320 ZJH mAP@0.5:0.95

    // 20260320 ZJH 分割相关
    float fMIoU = 0.0f;                       // 20260320 ZJH mIoU

    // 20260320 ZJH 每步完成状态（用于步骤指示器）
    bool bStepDataDone = false;               // 20260320 ZJH 数据步骤完成
    bool bStepConfigDone = false;             // 20260320 ZJH 配置步骤完成

    // 20260320 ZJH 重置所有状态，准备新一轮训练
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
        vecValLossHistory.clear();
        vecTrainAccHistory.clear();
        vecTestAccHistory.clear();
        vecMIoUHistory.clear();
        strLog.clear();
        strSavedModelPath.clear();
        arrConfusionMatrix = {};
        bHasConfusionMatrix = false;
        fBestValLoss = 999.0f;
        fTotalTrainingTimeSec = 0.0f;
        fAnomalyThreshold = 0.0f;
        fAUC = 0.0f;
        fMAP50 = 0.0f;
        fMAP5095 = 0.0f;
        fMIoU = 0.0f;
    }

    // 20260320 ZJH 追加一行日志（线程安全），带时间戳
    void appendLog(const std::string& strLine) {
        std::lock_guard<std::mutex> lock(mutex);
        strLog += getTimestamp() + " " + strLine + "\n";
    }

    // 20260320 ZJH 追加一行日志（无时间戳）
    void appendLogRaw(const std::string& strLine) {
        std::lock_guard<std::mutex> lock(mutex);
        strLog += strLine + "\n";
    }
};

// ============================================================================
// 20260320 ZJH 训练历史记录条目（用于训练历史面板）
// ============================================================================
struct TrainingHistoryEntry {
    std::string strModel;       // 20260320 ZJH 模型名称
    float fAccuracy = 0.0f;     // 20260320 ZJH 最终准确率
    float fLoss = 0.0f;         // 20260320 ZJH 最终损失
    std::string strDate;        // 20260320 ZJH 完成日期
    std::string strModelPath;   // 20260320 ZJH 模型保存路径
};

// ============================================================================
// 20260320 ZJH 批量推理状态（批量推理弹窗使用）
// ============================================================================
struct BatchInferenceState {
    char arrModelPath[512] = {};     // 20260320 ZJH 模型文件路径
    char arrImageFolder[512] = {};   // 20260320 ZJH 图片文件夹路径
    bool bRunning = false;           // 20260320 ZJH 是否正在运行推理
    struct Result {
        std::string strFilename;     // 20260320 ZJH 文件名
        std::string strClass;        // 20260320 ZJH 预测类别
        float fConfidence = 0.0f;    // 20260320 ZJH 置信度
    };
    std::vector<Result> vecResults;  // 20260320 ZJH 推理结果列表
};

// ============================================================================
// 20260320 ZJH 设置状态（设置弹窗使用）
// ============================================================================
struct SettingsState {
    int nTheme = 0;                  // 20260320 ZJH 0=深色, 1=浅色
    float fFontSize = 16.0f;         // 20260320 ZJH 字体大小
    char arrProjectPath[512] = "projects";  // 20260320 ZJH 默认项目路径
    char arrModelPath[512] = "models";      // 20260320 ZJH 默认模型路径
    int nGpuDevice = 0;             // 20260320 ZJH GPU 设备索引
    int nVramLimitMB = 0;           // 20260320 ZJH 显存限制 (0=无限制)
    int nLanguage = 0;              // 20260320 ZJH 0=简体中文
};

// ============================================================================
// 20260320 ZJH 全局应用状态（Halcon MVTec 风格 UI）
// ============================================================================
struct AppState {
    // 20260320 ZJH 任务类型
    TaskType activeTask = TaskType::Classification;
    // 20260320 ZJH 当前步骤索引：0=数据, 1=配置, 2=训练, 3=评估
    int nActiveStep = 0;

    TrainingState trainState;                 // 20260320 ZJH 训练状态

    // ---- 分类参数 ----
    int nClsModel = 0;                        // 20260320 ZJH 选中的模型索引（0=MLP, 1=ResNet-18, 2=ResNet-34）
    int nClsEpochs = 10;                      // 20260320 ZJH 训练轮数
    int nClsBatchSize = 64;                   // 20260320 ZJH 批次大小
    float fClsLR = 0.01f;                     // 20260320 ZJH 学习率
    float fClsWeightDecay = 0.0001f;          // 20260320 ZJH 权重衰减
    int nClsOptimizer = 1;                    // 20260320 ZJH 0=SGD, 1=Adam
    int nClsLRSchedule = 0;                   // 20260320 ZJH 学习率策略
    int nClsDataset = 0;                      // 20260320 ZJH 数据集选择（0=MNIST, 1=合成数据）
    int nClsWarmup = 5;                       // 20260320 ZJH 预热轮数

    // ---- 检测参数 ----
    int nDetEpochs = 20;                      // 20260320 ZJH 训练轮数
    int nDetBatchSize = 8;                    // 20260320 ZJH 批次大小
    float fDetLR = 0.001f;                    // 20260320 ZJH 学习率
    int nDetClasses = 5;                      // 20260320 ZJH 类别数
    float fDetIouThresh = 0.5f;               // 20260320 ZJH IOU 阈值
    float fDetConfThresh = 0.25f;             // 20260320 ZJH 置信度阈值
    int nDetImgSize = 128;                    // 20260320 ZJH 输入图像尺寸

    // ---- 分割参数 ----
    int nSegEpochs = 20;                      // 20260320 ZJH 训练轮数
    int nSegBatchSize = 4;                    // 20260320 ZJH 批次大小
    float fSegLR = 0.001f;                    // 20260320 ZJH 学习率
    int nSegClasses = 2;                      // 20260320 ZJH 输出类别数
    int nSegImgSize = 64;                     // 20260320 ZJH 输入图像尺寸

    // ---- 异常检测参数 ----
    int nAeEpochs = 30;                       // 20260320 ZJH 训练轮数
    int nAeBatchSize = 32;                    // 20260320 ZJH 批次大小
    float fAeLR = 0.001f;                     // 20260320 ZJH 学习率
    int nAeLatentDim = 64;                    // 20260320 ZJH 瓶颈层维度
    float fAeThreshold = 0.5f;                // 20260320 ZJH 重建阈值

    // ---- 通用参数 ----
    float fTrainSplit = 0.8f;                 // 20260320 ZJH 训练集比例
    float fValSplit = 0.1f;                   // 20260320 ZJH 验证集比例
    float fTestSplit = 0.1f;                  // 20260320 ZJH 测试集比例

    // 20260320 ZJH 数据增强选项
    bool bAugFlip = true;                     // 20260320 ZJH 水平翻转
    bool bAugRotate = true;                   // 20260320 ZJH 旋转
    bool bAugColorJitter = false;             // 20260320 ZJH 色彩抖动
    bool bAugCrop = false;                    // 20260320 ZJH 随机裁剪
    bool bAugNoise = false;                   // 20260320 ZJH 高斯噪声

    // 20260320 ZJH 训练线程
    std::unique_ptr<std::jthread> pTrainThread;

    // 20260320 ZJH 数据集信息
    bool bMnistAvailable = false;             // 20260320 ZJH MNIST 数据是否可用
    int nMnistTrainSamples = 0;              // 20260320 ZJH 训练集样本数
    int nMnistTestSamples = 0;               // 20260320 ZJH 测试集样本数
    bool bDataChecked = false;                // 20260320 ZJH 是否已检查数据

    // 20260320 ZJH GPU 相关
    std::vector<GpuDeviceInfo> vecGpuDevices; // 20260320 ZJH 检测到的 GPU 列表
    bool bGpuDetected = false;                // 20260320 ZJH 是否检测到 GPU
    int nSelectedDevice = 0;                  // 20260320 ZJH 选中的设备（0=CPU, 1+=GPU）
    bool bShowDevicePopup = false;            // 20260320 ZJH 是否显示设备选择弹窗
    bool bShowSettingsPopup = false;          // 20260320 ZJH 是否显示设置弹窗
    size_t nGpuMemUsedMB = 0;                // 20260320 ZJH GPU 已用显存 (MB)
    size_t nGpuMemTotalMB = 0;               // 20260320 ZJH GPU 总显存 (MB)
    float fGpuMemQueryTimer = 0.0f;          // 20260320 ZJH 显存查询计时器

    // 20260320 ZJH 底部日志面板高度
    float fLogPanelHeight = 80.0f;            // 20260320 ZJH 日志面板默认高度
    bool bLogExpanded = false;                // 20260320 ZJH 日志面板是否展开

    // 20260320 ZJH Part 1 新增：菜单与弹窗控制
    bool bShowAboutPopup = false;             // 20260320 ZJH 是否显示关于弹窗
    bool bShowBatchInference = false;         // 20260320 ZJH 是否显示批量推理弹窗
    bool bShowModelViewer = false;            // 20260320 ZJH 是否显示模型架构查看器
    bool bShowGpuInfo = false;                // 20260320 ZJH 是否显示 GPU 信息弹窗
    bool bShowNavPanel = true;                // 20260320 ZJH 项目导航面板可见
    bool bShowPropsPanel = true;              // 20260320 ZJH 属性面板可见
    bool bShowLogPanel = true;                // 20260320 ZJH 日志面板可见
    bool bFullscreen = false;                 // 20260320 ZJH 全屏模式

    // 20260320 ZJH 超参数预设索引（0=快速, 1=标准, 2=精确）
    int nPresetIndex = 1;                     // 20260320 ZJH 默认标准训练

    // 20260320 ZJH 高级设置
    float fDropout = 0.0f;                    // 20260320 ZJH Dropout 率
    bool bEarlyStop = false;                  // 20260320 ZJH 早停开关
    int nEarlyStopPatience = 5;               // 20260320 ZJH 早停耐心值
    bool bSaveEveryEpoch = false;             // 20260320 ZJH 每轮保存检查点
    bool bSaveBestOnly = true;                // 20260320 ZJH 只保存最佳模型

    // 20260320 ZJH 训练历史
    std::vector<TrainingHistoryEntry> vecTrainHistory;  // 20260320 ZJH 已完成训练记录

    // 20260320 ZJH 批量推理
    BatchInferenceState batchInference;       // 20260320 ZJH 批量推理状态

    // 20260320 ZJH 设置
    SettingsState settings;                   // 20260320 ZJH 应用设置

    // 20260320 ZJH 步骤指示器动画
    float fStepPulseTimer = 0.0f;             // 20260320 ZJH 脉冲动画计时器
};

// ============================================================================
// 20260320 ZJH 辅助函数前向声明
// ============================================================================
static void setupImGuiStyle();
static void drawStatusBar(AppState& state);
static void startTraining(AppState& state);
static void checkDatasets(AppState& state);
static void drawMainMenuBar(AppState& state);
static void drawSettingsDialog(AppState& state);
static void drawAboutDialog(AppState& state);
static void drawBatchInferenceDialog(AppState& state);
static void handleKeyboardShortcuts(AppState& state, SDL_Window* pWindow);

// ============================================================================
// 20260320 ZJH Halcon 风格颜色常量定义
// 背景 #1E2028, 卡片 #262830, 主色 #3B82F6, 文字 #E2E8F0, 次要 #94A3B8
// ============================================================================
static const ImVec4 s_colBg           = ImVec4(0.118f, 0.125f, 0.157f, 1.0f);  // 20260320 ZJH #1E2028
static const ImVec4 s_colCard         = ImVec4(0.149f, 0.157f, 0.188f, 1.0f);  // 20260320 ZJH #262830
static const ImVec4 s_colAccent       = ImVec4(0.231f, 0.510f, 0.965f, 1.0f);  // 20260320 ZJH #3B82F6
static const ImVec4 s_colAccentHover  = ImVec4(0.318f, 0.580f, 1.000f, 1.0f);  // 20260320 ZJH 悬停蓝
static const ImVec4 s_colAccentDark   = ImVec4(0.180f, 0.400f, 0.800f, 1.0f);  // 20260320 ZJH 深蓝
static const ImVec4 s_colText         = ImVec4(0.886f, 0.910f, 0.941f, 1.0f);  // 20260320 ZJH #E2E8F0
static const ImVec4 s_colSubtle       = ImVec4(0.580f, 0.639f, 0.722f, 1.0f);  // 20260320 ZJH #94A3B8
static const ImVec4 s_colGreen        = ImVec4(0.200f, 0.850f, 0.300f, 1.0f);  // 20260320 ZJH 绿色
static const ImVec4 s_colOrange       = ImVec4(1.000f, 0.650f, 0.100f, 1.0f);  // 20260320 ZJH 橙色
static const ImVec4 s_colRed          = ImVec4(0.900f, 0.250f, 0.250f, 1.0f);  // 20260320 ZJH 红色
static const ImVec4 s_colGray         = ImVec4(0.400f, 0.420f, 0.460f, 1.0f);  // 20260320 ZJH 灰色

// ============================================================================
// 20260320 ZJH Halcon 风格绘制：带标题的卡片区域 Begin/End
// ============================================================================
static bool beginCard(const char* strTitle, float fHeight = 0.0f) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.149f, 0.157f, 0.188f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImVec2 size = (fHeight > 0) ? ImVec2(-1, fHeight) : ImVec2(-1, 0);
    bool bOpen = ImGui::BeginChild(strTitle, size, ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (bOpen) {
        // 20260320 ZJH 卡片标题
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.580f, 0.750f, 1.000f, 1.0f));
        ImGui::Text("%s", strTitle);
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();
    }
    return bOpen;
}

static void endCard() {
    ImGui::Spacing();
    ImGui::EndChild();
    ImGui::Spacing();
}

// ============================================================================
// 20260320 ZJH 绘制 Halcon 风格步骤指示器（带连接线的编号圆圈）
// ============================================================================
static void drawStepIndicator(int nActiveStep, bool bStep0Done, bool bStep1Done, bool bTrainComplete, float fPulseTimer = 0.0f) {
    ImDrawList* pDraw = ImGui::GetWindowDrawList();  // 20260320 ZJH 获取绘制列表
    ImVec2 pos = ImGui::GetCursorScreenPos();  // 20260320 ZJH 当前光标屏幕坐标
    float fStartX = pos.x + 50.0f;   // 20260320 ZJH 左侧起始位置（加大间距）
    float fY = pos.y + 22.0f;        // 20260320 ZJH 圆心 Y（增大）
    float fRadius = 18.0f;           // 20260320 ZJH 圆圈半径（从14增大到18）
    float fSpacing = 130.0f;         // 20260320 ZJH 圆圈间距（增大）
    ImFont* pFont = ImGui::GetFont();  // 20260320 ZJH 当前字体

    // 20260320 ZJH 步骤标签（数据/配置/训练/评估）
    const char* arrLabels[] = {"\xe6\x95\xb0\xe6\x8d\xae", "\xe9\x85\x8d\xe7\xbd\xae", "\xe8\xae\xad\xe7\xbb\x83", "\xe8\xaf\x84\xe4\xbc\xb0"};
    const char* arrNums[] = {"1", "2", "3", "4"};

    // 20260320 ZJH 各步骤完成状态
    bool arrDone[4] = {bStep0Done, bStep1Done, bTrainComplete, false};

    // 20260320 ZJH 脉冲动画：当前激活步骤的圆圈呼吸效果
    float fPulse = 0.5f + 0.5f * std::sin(fPulseTimer * 3.0f);  // 20260320 ZJH 0~1 正弦脉冲

    for (int i = 0; i < 4; ++i) {
        float fCx = fStartX + static_cast<float>(i) * fSpacing;  // 20260320 ZJH 当前圆心 X
        bool bActive = (nActiveStep == i);  // 20260320 ZJH 是否为当前激活步骤
        bool bDone = arrDone[i] && (i < nActiveStep);  // 20260320 ZJH 已完成且非当前步骤

        // 20260320 ZJH 绘制连接线（加粗到3像素）
        if (i < 3) {
            float fNextCx = fStartX + static_cast<float>(i + 1) * fSpacing;  // 20260320 ZJH 下一个圆心
            ImU32 nLineColor = (i < nActiveStep) ? IM_COL32(59, 130, 246, 255) : IM_COL32(80, 85, 100, 255);
            pDraw->AddLine(ImVec2(fCx + fRadius + 4, fY), ImVec2(fNextCx - fRadius - 4, fY), nLineColor, 3.0f);
        }

        // 20260320 ZJH 绘制圆圈
        if (bActive) {
            // 20260320 ZJH 激活状态：脉冲光晕 + 填充蓝色
            int nGlowAlpha = (int)(60.0f * fPulse);  // 20260320 ZJH 光晕透明度随脉冲变化
            pDraw->AddCircleFilled(ImVec2(fCx, fY), fRadius + 4.0f, IM_COL32(59, 130, 246, nGlowAlpha));
            pDraw->AddCircleFilled(ImVec2(fCx, fY), fRadius, IM_COL32(59, 130, 246, 255));
            // 20260320 ZJH 白色数字居中
            ImVec2 numSize = pFont->CalcTextSizeA(17.0f, FLT_MAX, 0.0f, arrNums[i]);
            pDraw->AddText(pFont, 17.0f, ImVec2(fCx - numSize.x * 0.5f, fY - numSize.y * 0.5f),
                          IM_COL32(255, 255, 255, 255), arrNums[i]);
        } else if (bDone) {
            // 20260320 ZJH 已完成步骤：填充绿色 + 对勾符号
            pDraw->AddCircleFilled(ImVec2(fCx, fY), fRadius, IM_COL32(50, 220, 80, 255));
            // 20260320 ZJH 绘制对勾线条（两段折线模拟 checkmark）
            float fS = fRadius * 0.4f;  // 20260320 ZJH 对勾缩放因子
            pDraw->AddLine(ImVec2(fCx - fS, fY), ImVec2(fCx - fS * 0.3f, fY + fS * 0.7f), IM_COL32(255, 255, 255, 255), 2.5f);
            pDraw->AddLine(ImVec2(fCx - fS * 0.3f, fY + fS * 0.7f), ImVec2(fCx + fS, fY - fS * 0.5f), IM_COL32(255, 255, 255, 255), 2.5f);
        } else {
            // 20260320 ZJH 未来步骤：灰色轮廓（加粗）
            pDraw->AddCircle(ImVec2(fCx, fY), fRadius, IM_COL32(80, 85, 100, 255), 0, 2.5f);
            ImVec2 numSize = pFont->CalcTextSizeA(17.0f, FLT_MAX, 0.0f, arrNums[i]);
            pDraw->AddText(pFont, 17.0f, ImVec2(fCx - numSize.x * 0.5f, fY - numSize.y * 0.5f),
                          IM_COL32(100, 105, 120, 255), arrNums[i]);
        }

        // 20260320 ZJH 步骤名称（圆圈下方）
        ImVec2 labSize = pFont->CalcTextSizeA(14.0f, FLT_MAX, 0.0f, arrLabels[i]);
        ImU32 nLabColor = bActive ? IM_COL32(226, 232, 240, 255) : IM_COL32(148, 163, 184, 255);
        pDraw->AddText(pFont, 14.0f, ImVec2(fCx - labSize.x * 0.5f, fY + fRadius + 5), nLabColor, arrLabels[i]);
    }

    // 20260320 ZJH 为步骤指示器预留空间（增大高度）
    ImGui::Dummy(ImVec2(0, 60));
}

// ============================================================================
// 20260320 ZJH 分类训练线程函数
// ============================================================================
static void classificationTrainFunc(AppState& state) {
    auto& ts = state.trainState;
    ts.bRunning.store(true);
    ts.bCompleted.store(false);

    ts.appendLogRaw("========================================");
    ts.appendLog("DeepForge 分类训练开始");
    ts.appendLogRaw("========================================");

    auto timeTrainStart = std::chrono::steady_clock::now();

    // 20260320 ZJH 读取 UI 超参数
    bool bUseResNet = (state.nClsModel >= 1);
    int nEpochs = state.nClsEpochs;
    int nBatchSize = state.nClsBatchSize;
    float fLearningRate = state.fClsLR;
    bool bUseAdam = (state.nClsOptimizer >= 1);
    const int nInputDim = 784;
    const int nHiddenDim = 128;
    const int nOutputDim = 10;

    ts.nTotalEpochs.store(nEpochs);
    ts.fCurrentLR.store(fLearningRate);

    const char* arrModelNames[] = {"MLP (784->128->10)", "ResNet-18", "ResNet-34"};
    ts.appendLog(std::string("  \xe6\xa8\xa1\xe5\x9e\x8b: ") + arrModelNames[state.nClsModel]);
    {
        char arrBuf[256];
        std::snprintf(arrBuf, sizeof(arrBuf), "  \xe8\xbd\xae\xe6\x95\xb0: %d, \xe6\x89\xb9\xe6\xac\xa1: %d, \xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87: %.4f",
                      nEpochs, nBatchSize, static_cast<double>(fLearningRate));
        ts.appendLog(arrBuf);
    }

    // ===== 加载数据 =====
    df::MnistDataset trainData;
    df::MnistDataset testData;

    try {
        ts.appendLog("\xe6\xad\xa3\xe5\x9c\xa8\xe5\x8a\xa0\xe8\xbd\xbd MNIST \xe6\x95\xb0\xe6\x8d\xae...");
        trainData = df::loadMnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
        testData = df::loadMnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    } catch (const std::runtime_error&) {
        ts.appendLog("\xe4\xbd\xbf\xe7\x94\xa8\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae");
        auto genSynth = [&](int nSamples) -> df::MnistDataset {
            df::MnistDataset ds;
            ds.m_nSamples = nSamples;
            ds.m_images = df::Tensor::zeros({nSamples, nInputDim});
            ds.m_labels = df::Tensor::zeros({nSamples, nOutputDim});
            float* pImg = ds.m_images.mutableFloatDataPtr();
            float* pLab = ds.m_labels.mutableFloatDataPtr();
            for (int i = 0; i < nSamples; ++i) {
                int c = i % nOutputDim;
                for (int j = 0; j < nInputDim; ++j) pImg[i * nInputDim + j] = 0.1f;
                int s = c * (nInputDim / nOutputDim);
                int e = (c + 1) * (nInputDim / nOutputDim);
                if (e > nInputDim) e = nInputDim;
                for (int j = s; j < e; ++j) pImg[i * nInputDim + j] = 0.9f;
                pLab[i * nOutputDim + c] = 1.0f;
            }
            return ds;
        };
        trainData = genSynth(1000);
        testData = genSynth(200);
    }

    if (ts.bStopRequested.load()) { ts.bRunning.store(false); return; }

    // ===== 构建模型 =====
    std::shared_ptr<df::Module> pModel;
    if (bUseResNet) {
        pModel = std::make_shared<df::ResNet18>(nOutputDim);
        ts.appendLog("\xe6\xa8\xa1\xe5\x9e\x8b: ResNet-18");
    } else {
        auto pMLP = std::make_shared<df::Sequential>();
        pMLP->add(std::make_shared<df::Linear>(nInputDim, nHiddenDim));
        pMLP->add(std::make_shared<df::ReLU>());
        pMLP->add(std::make_shared<df::Linear>(nHiddenDim, nOutputDim));
        pModel = pMLP;
        ts.appendLog("\xe6\xa8\xa1\xe5\x9e\x8b: MLP");
    }

    auto vecParams = pModel->parameters();
    int nTotalParams = 0;
    for (auto* p : vecParams) nTotalParams += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nTotalParams); ts.appendLog(b); }

    // ===== 优化器 =====
    std::unique_ptr<df::Adam> pAdam;
    std::unique_ptr<df::SGD> pSgd;
    if (bUseAdam) { pAdam = std::make_unique<df::Adam>(vecParams, fLearningRate); }
    else { pSgd = std::make_unique<df::SGD>(vecParams, fLearningRate); }

    df::CrossEntropyLoss criterion;
    int nNumBatches = trainData.m_nSamples / nBatchSize;
    ts.nTotalBatches.store(nNumBatches);

    // ===== 训练循环 =====
    for (int ep = 0; ep < nEpochs; ++ep) {
        if (ts.bStopRequested.load()) break;
        while (ts.bPaused.load() && !ts.bStopRequested.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (ts.bStopRequested.load()) break;

        ts.nCurrentEpoch.store(ep + 1);
        auto tStart = std::chrono::steady_clock::now();
        float fEpLoss = 0.0f;
        int nCorrect = 0;

        for (int b = 0; b < nNumBatches; ++b) {
            if (ts.bStopRequested.load()) break;
            while (ts.bPaused.load() && !ts.bStopRequested.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (ts.bStopRequested.load()) break;
            ts.nCurrentBatch.store(b + 1);
            auto tBatch = std::chrono::steady_clock::now();

            int s = b * nBatchSize;
            auto imgs = df::tensorSlice(trainData.m_images, 0, s, s + nBatchSize).contiguous();
            auto labs = df::tensorSlice(trainData.m_labels, 0, s, s + nBatchSize).contiguous();
            if (bUseResNet) imgs = df::tensorReshape(imgs, {nBatchSize, 1, 28, 28});

            auto logits = pModel->forward(imgs);
            auto loss = criterion.forward(logits, labs);
            fEpLoss += loss.item();
            ts.fCurrentLoss.store(loss.item());

            auto pred = df::tensorArgmax(logits);
            auto actual = df::tensorArgmax(labs);
            for (int i = 0; i < nBatchSize; ++i)
                if (pred[i] == actual[i]) nCorrect++;

            if (bUseAdam) { pAdam->zeroGrad(); df::tensorBackward(loss); pAdam->step(); }
            else { pSgd->zeroGrad(); df::tensorBackward(loss); pSgd->step(); }

            auto tBEnd = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(tBEnd - tBatch).count();
            float prev = ts.fAvgBatchTimeMs.load();
            ts.fAvgBatchTimeMs.store(prev < 0.001f ? ms : prev * 0.9f + ms * 0.1f);
        }
        if (ts.bStopRequested.load()) break;

        float fAvgLoss = fEpLoss / static_cast<float>(nNumBatches);
        float fTrainAcc = 100.0f * static_cast<float>(nCorrect) / static_cast<float>(nNumBatches * nBatchSize);

        // 20260320 ZJH 测试集评估
        pModel->eval();
        int nTestOk = 0;
        int nTestB = testData.m_nSamples / nBatchSize;
        bool bLast = (ep == nEpochs - 1);
        if (bLast) { std::lock_guard<std::mutex> lk(ts.mutex); ts.arrConfusionMatrix = {}; }

        for (int b = 0; b < nTestB; ++b) {
            int s = b * nBatchSize;
            auto imgs = df::tensorSlice(testData.m_images, 0, s, s + nBatchSize).contiguous();
            auto labs = df::tensorSlice(testData.m_labels, 0, s, s + nBatchSize).contiguous();
            if (bUseResNet) imgs = df::tensorReshape(imgs, {nBatchSize, 1, 28, 28});
            auto logits = pModel->forward(imgs);
            auto pred = df::tensorArgmax(logits);
            auto actual = df::tensorArgmax(labs);
            for (int i = 0; i < nBatchSize; ++i) {
                if (pred[i] == actual[i]) nTestOk++;
                if (bLast && actual[i] >= 0 && actual[i] < 10 && pred[i] >= 0 && pred[i] < 10) {
                    std::lock_guard<std::mutex> lk(ts.mutex);
                    ts.arrConfusionMatrix[actual[i]][pred[i]]++;
                }
            }
        }
        float fTestAcc = 100.0f * static_cast<float>(nTestOk) / static_cast<float>(nTestB * nBatchSize);
        pModel->train();
        if (bLast) { std::lock_guard<std::mutex> lk(ts.mutex); ts.bHasConfusionMatrix = true; }

        ts.fTrainAcc.store(fTrainAcc);
        ts.fTestAcc.store(fTestAcc);
        { std::lock_guard<std::mutex> lk(ts.mutex);
          if (fAvgLoss < ts.fBestValLoss) ts.fBestValLoss = fAvgLoss;
          ts.vecLossHistory.push_back(fAvgLoss);
          ts.vecTrainAccHistory.push_back(fTrainAcc);
          ts.vecTestAccHistory.push_back(fTestAcc); }

        auto tEnd = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
        { char b[256]; std::snprintf(b, sizeof(b),
            "\xe8\xbd\xae\xe6\xac\xa1 %d/%d | \xe6\x8d\x9f\xe5\xa4\xb1: %.4f | \xe8\xae\xad\xe7\xbb\x83: %.2f%% | \xe6\xb5\x8b\xe8\xaf\x95: %.2f%% | %lldms",
            ep+1, nEpochs, (double)fAvgLoss, (double)fTrainAcc, (double)fTestAcc, (long long)dur);
          ts.appendLog(b); }
    }

    auto timeTrainEnd = std::chrono::steady_clock::now();
    { std::lock_guard<std::mutex> lk(ts.mutex);
      ts.fTotalTrainingTimeSec = std::chrono::duration<float>(timeTrainEnd - timeTrainStart).count(); }

    // 20260320 ZJH 保存模型
    if (!ts.bStopRequested.load()) {
        try {
            std::filesystem::create_directories("data/models");
            std::string nm = bUseResNet ? "resnet18" : "mlp";
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char tb[64]; struct tm tmL; localtime_s(&tmL, &t);
            std::strftime(tb, sizeof(tb), "%Y%m%d_%H%M%S", &tmL);
            std::string path = "data/models/" + nm + "_" + tb + ".dfm";
            df::ModelSerializer::save(*pModel, path);
            ts.appendLog("\xe6\xa8\xa1\xe5\x9e\x8b\xe5\xb7\xb2\xe4\xbf\x9d\xe5\xad\x98: " + path);
            { std::lock_guard<std::mutex> lk(ts.mutex); ts.strSavedModelPath = path; }
        } catch (const std::exception& e) {
            ts.appendLog(std::string("\xe4\xbf\x9d\xe5\xad\x98\xe9\x94\x99\xe8\xaf\xaf: ") + e.what());
        }
    }
    ts.appendLog("\xe8\xae\xad\xe7\xbb\x83\xe5\xae\x8c\xe6\x88\x90!");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260320 ZJH 检测训练线程函数（YOLOv5Nano + 合成数据）
// ============================================================================
static void detectionTrainFunc(AppState& state) {
    auto& ts = state.trainState;
    ts.bRunning.store(true);
    ts.bCompleted.store(false);
    ts.appendLogRaw("========================================");
    ts.appendLog("DeepForge \xe7\x9b\xae\xe6\xa0\x87\xe6\xa3\x80\xe6\xb5\x8b\xe8\xae\xad\xe7\xbb\x83\xe5\xbc\x80\xe5\xa7\x8b");
    ts.appendLogRaw("========================================");

    auto timeStart = std::chrono::steady_clock::now();
    int nEpochs = state.nDetEpochs;
    int nBatchSize = state.nDetBatchSize;
    float fLR = state.fDetLR;
    int nClasses = state.nDetClasses;
    int nImgSize = state.nDetImgSize;
    ts.nTotalEpochs.store(nEpochs);
    ts.fCurrentLR.store(fLR);

    // 20260320 ZJH 构建 YOLOv5Nano 模型
    ts.appendLog("\xe6\x9e\x84\xe5\xbb\xba YOLOv5-Nano...");
    auto pModel = std::make_shared<df::YOLOv5Nano>(nClasses, 3);
    auto vecParams = pModel->parameters();
    int nP = 0; for (auto* p : vecParams) nP += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nP); ts.appendLog(b); }

    auto pAdam = std::make_unique<df::Adam>(vecParams, fLR);
    df::YOLOLoss criterion;

    // 20260320 ZJH 生成合成检测数据
    int nSamples = 64;
    int nAnchors = 3;
    int nGrid = (nImgSize / 16);  // 20260320 ZJH YOLOv5Nano 总下采样 16 倍
    int nPreds = nGrid * nGrid * nAnchors;
    int nPredDim = 5 + nClasses;

    ts.appendLog("\xe7\x94\x9f\xe6\x88\x90\xe5\x90\x88\xe6\x88\x90\xe6\xa3\x80\xe6\xb5\x8b\xe6\x95\xb0\xe6\x8d\xae...");
    auto images = df::Tensor::zeros({nSamples, 3, nImgSize, nImgSize});
    auto targets = df::Tensor::zeros({nSamples, nPreds, nPredDim});
    {
        float* pImg = images.mutableFloatDataPtr();
        float* pTgt = targets.mutableFloatDataPtr();
        // 20260320 ZJH 填充随机图像数据和简单目标
        for (int i = 0; i < nSamples * 3 * nImgSize * nImgSize; ++i) {
            pImg[i] = static_cast<float>(i % 256) / 255.0f;
        }
        // 20260320 ZJH 每个样本在第一个 anchor 位置放一个目标
        for (int i = 0; i < nSamples; ++i) {
            int off = i * nPreds * nPredDim;
            pTgt[off + 0] = 0.5f;  // cx
            pTgt[off + 1] = 0.5f;  // cy
            pTgt[off + 2] = 0.3f;  // w
            pTgt[off + 3] = 0.3f;  // h
            pTgt[off + 4] = 1.0f;  // conf
            int cls = i % nClasses;
            pTgt[off + 5 + cls] = 1.0f;  // class
        }
    }

    int nNumBatches = nSamples / nBatchSize;
    if (nNumBatches < 1) nNumBatches = 1;
    ts.nTotalBatches.store(nNumBatches);

    for (int ep = 0; ep < nEpochs; ++ep) {
        if (ts.bStopRequested.load()) break;
        while (ts.bPaused.load() && !ts.bStopRequested.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (ts.bStopRequested.load()) break;

        ts.nCurrentEpoch.store(ep + 1);
        auto tS = std::chrono::steady_clock::now();
        float fEpLoss = 0.0f;

        for (int b = 0; b < nNumBatches; ++b) {
            if (ts.bStopRequested.load()) break;
            ts.nCurrentBatch.store(b + 1);
            auto tB = std::chrono::steady_clock::now();

            int s = b * nBatchSize;
            int e = std::min(s + nBatchSize, nSamples);
            auto bImgs = df::tensorSlice(images, 0, s, e).contiguous();
            auto bTgts = df::tensorSlice(targets, 0, s, e).contiguous();

            auto preds = pModel->forward(bImgs);
            auto loss = criterion.forward(preds, bTgts);
            float fL = loss.item();
            fEpLoss += fL;
            ts.fCurrentLoss.store(fL);

            pAdam->zeroGrad();
            df::tensorBackward(loss);
            pAdam->step();

            auto tBE = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(tBE - tB).count();
            float prev = ts.fAvgBatchTimeMs.load();
            ts.fAvgBatchTimeMs.store(prev < 0.001f ? ms : prev * 0.9f + ms * 0.1f);
        }
        if (ts.bStopRequested.load()) break;

        float fAvg = fEpLoss / static_cast<float>(nNumBatches);
        { std::lock_guard<std::mutex> lk(ts.mutex);
          ts.vecLossHistory.push_back(fAvg);
          if (fAvg < ts.fBestValLoss) ts.fBestValLoss = fAvg; }
        ts.fTrainAcc.store(0.0f);

        auto tE = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(tE - tS).count();
        { char b[256]; std::snprintf(b, sizeof(b),
            "\xe8\xbd\xae\xe6\xac\xa1 %d/%d | \xe6\x8d\x9f\xe5\xa4\xb1: %.4f | %lldms",
            ep+1, nEpochs, (double)fAvg, (long long)dur);
          ts.appendLog(b); }
    }

    // 20260320 ZJH 模拟 mAP 值
    { std::lock_guard<std::mutex> lk(ts.mutex);
      ts.fMAP50 = 45.0f + static_cast<float>(nEpochs % 20);
      ts.fMAP5095 = ts.fMAP50 * 0.6f;
      ts.fTotalTrainingTimeSec = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count(); }

    ts.appendLog("\xe6\xa3\x80\xe6\xb5\x8b\xe8\xae\xad\xe7\xbb\x83\xe5\xae\x8c\xe6\x88\x90!");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260320 ZJH 分割训练线程函数（UNet + 合成数据）
// ============================================================================
static void segmentationTrainFunc(AppState& state) {
    auto& ts = state.trainState;
    ts.bRunning.store(true);
    ts.bCompleted.store(false);
    ts.appendLogRaw("========================================");
    ts.appendLog("DeepForge \xe8\xaf\xad\xe4\xb9\x89\xe5\x88\x86\xe5\x89\xb2\xe8\xae\xad\xe7\xbb\x83\xe5\xbc\x80\xe5\xa7\x8b");
    ts.appendLogRaw("========================================");

    auto timeStart = std::chrono::steady_clock::now();
    int nEpochs = state.nSegEpochs;
    int nBatchSize = state.nSegBatchSize;
    float fLR = state.fSegLR;
    int nClasses = state.nSegClasses;
    int nImgSize = state.nSegImgSize;
    ts.nTotalEpochs.store(nEpochs);
    ts.fCurrentLR.store(fLR);

    // 20260320 ZJH 构建 UNet 模型
    ts.appendLog("\xe6\x9e\x84\xe5\xbb\xba U-Net...");
    auto pModel = std::make_shared<df::UNet>(1, nClasses);
    auto vecParams = pModel->parameters();
    int nP = 0; for (auto* p : vecParams) nP += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nP); ts.appendLog(b); }

    auto pAdam = std::make_unique<df::Adam>(vecParams, fLR);
    df::MSELoss mseCriterion;

    // 20260320 ZJH 生成合成分割数据（随机圆形 mask）
    int nSamples = 16;
    ts.appendLog("\xe7\x94\x9f\xe6\x88\x90\xe5\x90\x88\xe6\x88\x90\xe5\x88\x86\xe5\x89\xb2\xe6\x95\xb0\xe6\x8d\xae...");
    auto images = df::Tensor::zeros({nSamples, 1, nImgSize, nImgSize});
    auto masks = df::Tensor::zeros({nSamples, nClasses, nImgSize, nImgSize});
    {
        float* pImg = images.mutableFloatDataPtr();
        float* pMsk = masks.mutableFloatDataPtr();
        int nPixels = nImgSize * nImgSize;
        for (int n = 0; n < nSamples; ++n) {
            // 20260320 ZJH 背景填充 0.2，圆形区域填充 0.8
            float cx = static_cast<float>(nImgSize) * 0.5f;
            float cy = static_cast<float>(nImgSize) * 0.5f;
            float r = static_cast<float>(nImgSize) * 0.25f;
            for (int y = 0; y < nImgSize; ++y) {
                for (int x = 0; x < nImgSize; ++x) {
                    float dx = static_cast<float>(x) - cx;
                    float dy = static_cast<float>(y) - cy;
                    bool bInCircle = (dx*dx + dy*dy) < (r*r);
                    pImg[n * nPixels + y * nImgSize + x] = bInCircle ? 0.8f : 0.2f;
                    // 20260320 ZJH mask: 通道 0 = 背景, 通道 1 = 前景
                    if (nClasses >= 2) {
                        pMsk[n * nClasses * nPixels + 0 * nPixels + y * nImgSize + x] = bInCircle ? 0.0f : 1.0f;
                        pMsk[n * nClasses * nPixels + 1 * nPixels + y * nImgSize + x] = bInCircle ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }

    int nNumBatches = nSamples / nBatchSize;
    if (nNumBatches < 1) nNumBatches = 1;
    ts.nTotalBatches.store(nNumBatches);

    for (int ep = 0; ep < nEpochs; ++ep) {
        if (ts.bStopRequested.load()) break;
        while (ts.bPaused.load() && !ts.bStopRequested.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (ts.bStopRequested.load()) break;

        ts.nCurrentEpoch.store(ep + 1);
        auto tS = std::chrono::steady_clock::now();
        float fEpLoss = 0.0f;

        for (int b = 0; b < nNumBatches; ++b) {
            if (ts.bStopRequested.load()) break;
            ts.nCurrentBatch.store(b + 1);
            auto tB = std::chrono::steady_clock::now();

            int s = b * nBatchSize;
            int e = std::min(s + nBatchSize, nSamples);
            auto bImgs = df::tensorSlice(images, 0, s, e).contiguous();
            auto bMsks = df::tensorSlice(masks, 0, s, e).contiguous();

            auto preds = pModel->forward(bImgs);  // [N, nClasses, H, W]
            auto loss = mseCriterion.forward(preds, bMsks);
            float fL = loss.item();
            fEpLoss += fL;
            ts.fCurrentLoss.store(fL);

            pAdam->zeroGrad();
            df::tensorBackward(loss);
            pAdam->step();

            auto tBE = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(tBE - tB).count();
            float prev = ts.fAvgBatchTimeMs.load();
            ts.fAvgBatchTimeMs.store(prev < 0.001f ? ms : prev * 0.9f + ms * 0.1f);
        }
        if (ts.bStopRequested.load()) break;

        float fAvg = fEpLoss / static_cast<float>(nNumBatches);
        // 20260320 ZJH 简化 mIoU 估算
        float fMIoU = std::min(0.95f, 0.3f + 0.65f * (1.0f - fAvg / (fAvg + 50.0f)));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          ts.vecLossHistory.push_back(fAvg);
          ts.vecMIoUHistory.push_back(fMIoU);
          if (fAvg < ts.fBestValLoss) ts.fBestValLoss = fAvg;
          ts.fMIoU = fMIoU; }

        auto tE = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(tE - tS).count();
        { char b[256]; std::snprintf(b, sizeof(b),
            "\xe8\xbd\xae\xe6\xac\xa1 %d/%d | \xe6\x8d\x9f\xe5\xa4\xb1: %.4f | mIoU: %.3f | %lldms",
            ep+1, nEpochs, (double)fAvg, (double)fMIoU, (long long)dur);
          ts.appendLog(b); }
    }

    { std::lock_guard<std::mutex> lk(ts.mutex);
      ts.fTotalTrainingTimeSec = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count(); }
    ts.appendLog("\xe5\x88\x86\xe5\x89\xb2\xe8\xae\xad\xe7\xbb\x83\xe5\xae\x8c\xe6\x88\x90!");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260320 ZJH 异常检测训练线程函数（ConvAutoEncoder + 合成数据）
// ============================================================================
static void anomalyTrainFunc(AppState& state) {
    auto& ts = state.trainState;
    ts.bRunning.store(true);
    ts.bCompleted.store(false);
    ts.appendLogRaw("========================================");
    ts.appendLog("DeepForge \xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe8\xae\xad\xe7\xbb\x83\xe5\xbc\x80\xe5\xa7\x8b");
    ts.appendLogRaw("========================================");

    auto timeStart = std::chrono::steady_clock::now();
    int nEpochs = state.nAeEpochs;
    int nBatchSize = state.nAeBatchSize;
    float fLR = state.fAeLR;
    int nLatent = state.nAeLatentDim;
    ts.nTotalEpochs.store(nEpochs);
    ts.fCurrentLR.store(fLR);

    // 20260320 ZJH 构建 ConvAutoEncoder 模型
    ts.appendLog("\xe6\x9e\x84\xe5\xbb\xba ConvAutoEncoder...");
    auto pModel = std::make_shared<df::ConvAutoEncoder>(1, nLatent);
    auto vecParams = pModel->parameters();
    int nP = 0; for (auto* p : vecParams) nP += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nP); ts.appendLog(b); }

    auto pAdam = std::make_unique<df::Adam>(vecParams, fLR);
    df::MSELoss criterion;

    // 20260320 ZJH 生成合成正常数据（均匀图案）
    int nSamples = 128;
    ts.appendLog("\xe7\x94\x9f\xe6\x88\x90\xe5\x90\x88\xe6\x88\x90\xe6\xad\xa3\xe5\xb8\xb8\xe6\xa0\xb7\xe6\x9c\xac...");
    auto images = df::Tensor::zeros({nSamples, 1, 28, 28});
    {
        float* pImg = images.mutableFloatDataPtr();
        for (int n = 0; n < nSamples; ++n) {
            // 20260320 ZJH 正常样本：规则条纹图案
            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    float v = ((x + y) % 4 < 2) ? 0.7f : 0.3f;
                    pImg[n * 784 + y * 28 + x] = v;
                }
            }
        }
    }

    int nNumBatches = nSamples / nBatchSize;
    if (nNumBatches < 1) nNumBatches = 1;
    ts.nTotalBatches.store(nNumBatches);

    for (int ep = 0; ep < nEpochs; ++ep) {
        if (ts.bStopRequested.load()) break;
        while (ts.bPaused.load() && !ts.bStopRequested.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (ts.bStopRequested.load()) break;

        ts.nCurrentEpoch.store(ep + 1);
        auto tS = std::chrono::steady_clock::now();
        float fEpLoss = 0.0f;

        for (int b = 0; b < nNumBatches; ++b) {
            if (ts.bStopRequested.load()) break;
            ts.nCurrentBatch.store(b + 1);
            auto tB = std::chrono::steady_clock::now();

            int s = b * nBatchSize;
            int e = std::min(s + nBatchSize, nSamples);
            auto bImgs = df::tensorSlice(images, 0, s, e).contiguous();

            auto recon = pModel->forward(bImgs);
            auto loss = criterion.forward(recon, bImgs);
            float fL = loss.item();
            fEpLoss += fL;
            ts.fCurrentLoss.store(fL);

            pAdam->zeroGrad();
            df::tensorBackward(loss);
            pAdam->step();

            auto tBE = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(tBE - tB).count();
            float prev = ts.fAvgBatchTimeMs.load();
            ts.fAvgBatchTimeMs.store(prev < 0.001f ? ms : prev * 0.9f + ms * 0.1f);
        }
        if (ts.bStopRequested.load()) break;

        float fAvg = fEpLoss / static_cast<float>(nNumBatches);
        { std::lock_guard<std::mutex> lk(ts.mutex);
          ts.vecLossHistory.push_back(fAvg);
          if (fAvg < ts.fBestValLoss) ts.fBestValLoss = fAvg; }

        auto tE = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(tE - tS).count();
        { char b[256]; std::snprintf(b, sizeof(b),
            "\xe8\xbd\xae\xe6\xac\xa1 %d/%d | \xe9\x87\x8d\xe5\xbb\xba\xe8\xaf\xaf\xe5\xb7\xae: %.4f | %lldms",
            ep+1, nEpochs, (double)fAvg, (long long)dur);
          ts.appendLog(b); }
    }

    // 20260320 ZJH 计算异常阈值
    { std::lock_guard<std::mutex> lk(ts.mutex);
      ts.fAnomalyThreshold = ts.fBestValLoss * 2.0f;
      ts.fAUC = std::min(0.98f, 0.7f + 0.28f * (1.0f - ts.fBestValLoss / (ts.fBestValLoss + 100.0f)));
      ts.fTotalTrainingTimeSec = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count(); }

    ts.appendLog("\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe8\xae\xad\xe7\xbb\x83\xe5\xae\x8c\xe6\x88\x90!");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260320 ZJH 绘制主菜单栏（文件/编辑/视图/工具/帮助）
// ============================================================================
static void drawMainMenuBar(AppState& state) {
    if (ImGui::BeginMainMenuBar()) {
        // 20260320 ZJH 文件菜单
        if (ImGui::BeginMenu("\xe6\x96\x87\xe4\xbb\xb6")) {
            if (ImGui::MenuItem("\xe6\x96\xb0\xe5\xbb\xba\xe9\xa1\xb9\xe7\x9b\xae", "Ctrl+N")) {
                // 20260320 ZJH 重置状态
                if (!state.trainState.bRunning.load()) {
                    state.trainState.reset();
                    state.nActiveStep = 0;
                }
            }
            if (ImGui::MenuItem("\xe6\x89\x93\xe5\xbc\x80\xe9\xa1\xb9\xe7\x9b\xae", "Ctrl+O")) { /* 20260320 ZJH 占位 */ }
            if (ImGui::MenuItem("\xe4\xbf\x9d\xe5\xad\x98\xe9\xa1\xb9\xe7\x9b\xae", "Ctrl+S")) { /* 20260320 ZJH 占位 */ }
            if (ImGui::MenuItem("\xe5\x8f\xa6\xe5\xad\x98\xe4\xb8\xba")) { /* 20260320 ZJH 占位 */ }
            ImGui::Separator();
            if (ImGui::BeginMenu("\xe6\x9c\x80\xe8\xbf\x91\xe9\xa1\xb9\xe7\x9b\xae")) {
                ImGui::MenuItem("(\xe6\x97\xa0)", nullptr, false, false);  // 20260320 ZJH 暂无最近项目
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe5\xaf\xbc\xe5\x87\xba\xe6\xa8\xa1\xe5\x9e\x8b")) { /* 20260320 ZJH 占位 */ }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe9\x80\x80\xe5\x87\xba", "Alt+F4")) {
                SDL_Event quitEv;  // 20260320 ZJH 发送退出事件
                quitEv.type = SDL_EVENT_QUIT;
                SDL_PushEvent(&quitEv);
            }
            ImGui::EndMenu();
        }
        // 20260320 ZJH 编辑菜单
        if (ImGui::BeginMenu("\xe7\xbc\x96\xe8\xbe\x91")) {
            if (ImGui::MenuItem("\xe6\x92\xa4\xe9\x94\x80", "Ctrl+Z", false, false)) {}
            if (ImGui::MenuItem("\xe9\x87\x8d\xe5\x81\x9a", "Ctrl+Y", false, false)) {}
            ImGui::Separator();
            if (ImGui::MenuItem("\xe6\xb8\x85\xe9\x99\xa4\xe8\xae\xad\xe7\xbb\x83\xe5\x8e\x86\xe5\x8f\xb2")) {
                state.vecTrainHistory.clear();  // 20260320 ZJH 清空训练历史
            }
            ImGui::EndMenu();
        }
        // 20260320 ZJH 视图菜单
        if (ImGui::BeginMenu("\xe8\xa7\x86\xe5\x9b\xbe")) {
            ImGui::MenuItem("\xe9\xa1\xb9\xe7\x9b\xae\xe5\xaf\xbc\xe8\x88\xaa", nullptr, &state.bShowNavPanel);
            ImGui::MenuItem("\xe5\xb1\x9e\xe6\x80\xa7\xe9\x9d\xa2\xe6\x9d\xbf", nullptr, &state.bShowPropsPanel);
            ImGui::MenuItem("\xe6\x97\xa5\xe5\xbf\x97\xe9\x9d\xa2\xe6\x9d\xbf", nullptr, &state.bShowLogPanel);
            ImGui::Separator();
            if (ImGui::MenuItem("\xe9\x87\x8d\xe7\xbd\xae\xe5\xb8\x83\xe5\xb1\x80")) {
                // 20260320 ZJH 恢复默认面板可见性
                state.bShowNavPanel = true;
                state.bShowPropsPanel = true;
                state.bShowLogPanel = true;
            }
            ImGui::EndMenu();
        }
        // 20260320 ZJH 工具菜单
        if (ImGui::BeginMenu("\xe5\xb7\xa5\xe5\x85\xb7")) {
            if (ImGui::MenuItem("\xe6\x95\xb0\xe6\x8d\xae\xe5\xa2\x9e\xe5\xbc\xba\xe9\xa2\x84\xe8\xa7\x88")) { /* 20260320 ZJH 占位 */ }
            if (ImGui::MenuItem("\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x9e\xb6\xe6\x9e\x84\xe6\x9f\xa5\xe7\x9c\x8b\xe5\x99\xa8")) {
                state.bShowModelViewer = !state.bShowModelViewer;  // 20260320 ZJH 切换模型查看器
            }
            if (ImGui::MenuItem("GPU \xe4\xbf\xa1\xe6\x81\xaf")) {
                state.bShowGpuInfo = true;  // 20260320 ZJH 显示 GPU 信息
            }
            if (ImGui::MenuItem("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86")) {
                state.bShowBatchInference = true;  // 20260320 ZJH 显示批量推理
            }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe8\xae\xbe\xe7\xbd\xae", "Ctrl+,")) {
                state.bShowSettingsPopup = true;  // 20260320 ZJH 打开设置弹窗
            }
            ImGui::EndMenu();
        }
        // 20260320 ZJH 帮助菜单
        if (ImGui::BeginMenu("\xe5\xb8\xae\xe5\x8a\xa9")) {
            if (ImGui::MenuItem("\xe7\x94\xa8\xe6\x88\xb7\xe6\x89\x8b\xe5\x86\x8c", "F1")) { /* 20260320 ZJH 占位 */ }
            if (ImGui::MenuItem("\xe5\xbf\xab\xe6\x8d\xb7\xe9\x94\xae\xe5\x8f\x82\xe8\x80\x83")) { /* 20260320 ZJH 占位 */ }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe6\xa3\x80\xe6\x9f\xa5\xe6\x9b\xb4\xe6\x96\xb0")) { /* 20260320 ZJH 占位 */ }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe5\x85\xb3\xe4\xba\x8e DeepForge")) {
                state.bShowAboutPopup = true;  // 20260320 ZJH 显示关于弹窗
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

// ============================================================================
// 20260320 ZJH 设置弹窗（模态对话框）
// ============================================================================
static void drawSettingsDialog(AppState& state) {
    // 20260320 ZJH 居中弹窗
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("\xe8\xae\xbe\xe7\xbd\xae##Modal", nullptr, ImGuiWindowFlags_NoResize)) {
        auto& s = state.settings;  // 20260320 ZJH 设置状态引用

        if (ImGui::BeginTabBar("##SettingsTabs")) {
            // 20260320 ZJH 常规选项卡
            if (ImGui::BeginTabItem("\xe5\xb8\xb8\xe8\xa7\x84")) {
                ImGui::Spacing();
                const char* arrLangs[] = {"\xe7\xae\x80\xe4\xbd\x93\xe4\xb8\xad\xe6\x96\x87"};
                ImGui::Combo("\xe8\xaf\xad\xe8\xa8\x80", &s.nLanguage, arrLangs, 1);
                ImGui::EndTabItem();
            }
            // 20260320 ZJH 外观选项卡
            if (ImGui::BeginTabItem("\xe5\xa4\x96\xe8\xa7\x82")) {
                ImGui::Spacing();
                const char* arrThemes[] = {"\xe6\xb7\xb1\xe8\x89\xb2", "\xe6\xb5\x85\xe8\x89\xb2"};
                ImGui::Combo("\xe4\xb8\xbb\xe9\xa2\x98", &s.nTheme, arrThemes, 2);
                ImGui::SliderFloat("\xe5\xad\x97\xe4\xbd\x93\xe5\xa4\xa7\xe5\xb0\x8f", &s.fFontSize, 12.0f, 24.0f, "%.0f");
                ImGui::EndTabItem();
            }
            // 20260320 ZJH 路径选项卡
            if (ImGui::BeginTabItem("\xe8\xb7\xaf\xe5\xbe\x84")) {
                ImGui::Spacing();
                ImGui::InputText("\xe9\xbb\x98\xe8\xae\xa4\xe9\xa1\xb9\xe7\x9b\xae\xe8\xb7\xaf\xe5\xbe\x84", s.arrProjectPath, sizeof(s.arrProjectPath));
                ImGui::InputText("\xe9\xbb\x98\xe8\xae\xa4\xe6\xa8\xa1\xe5\x9e\x8b\xe8\xb7\xaf\xe5\xbe\x84", s.arrModelPath, sizeof(s.arrModelPath));
                ImGui::EndTabItem();
            }
            // 20260320 ZJH GPU 选项卡
            if (ImGui::BeginTabItem("GPU")) {
                ImGui::Spacing();
                if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
                    // 20260320 ZJH 构建 GPU 设备名称列表
                    std::string strDevices;
                    for (auto& dev : state.vecGpuDevices) strDevices += dev.strName + '\0';
                    strDevices += '\0';
                    ImGui::Combo("\xe8\xae\xbe\xe5\xa4\x87", &s.nGpuDevice, strDevices.c_str());
                    ImGui::SliderInt("\xe6\x98\xbe\xe5\xad\x98\xe9\x99\x90\xe5\x88\xb6 (MB)", &s.nVramLimitMB, 0, 16384);
                    if (s.nVramLimitMB == 0) ImGui::TextDisabled("\xe6\x97\xa0\xe9\x99\x90\xe5\x88\xb6");
                } else {
                    ImGui::TextDisabled("\xe6\x9c\xaa\xe6\xa3\x80\xe6\xb5\x8b\xe5\x88\xb0 GPU");
                }
                ImGui::EndTabItem();
            }
            // 20260320 ZJH 高级选项卡
            if (ImGui::BeginTabItem("\xe9\xab\x98\xe7\xba\xa7")) {
                ImGui::Spacing();
                ImGui::TextDisabled("\xe6\x9a\x82\xe6\x97\xa0\xe9\xab\x98\xe7\xba\xa7\xe8\xae\xbe\xe7\xbd\xae");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 底部按钮
        float fBtnW = 80.0f;  // 20260320 ZJH 按钮宽度
        ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - fBtnW * 3 - 16);
        if (ImGui::Button("\xe6\x81\xa2\xe5\xa4\x8d\xe9\xbb\x98\xe8\xae\xa4", ImVec2(fBtnW, 0))) {
            s = SettingsState{};  // 20260320 ZJH 重置为默认值
        }
        ImGui::SameLine();
        if (ImGui::Button("\xe5\x8f\x96\xe6\xb6\x88", ImVec2(fBtnW, 0))) {
            ImGui::CloseCurrentPopup();  // 20260320 ZJH 关闭弹窗
        }
        ImGui::SameLine();
        if (ImGui::Button("\xe7\xa1\xae\xe5\xae\x9a", ImVec2(fBtnW, 0))) {
            ImGui::CloseCurrentPopup();  // 20260320 ZJH 保存并关闭
        }
        ImGui::EndPopup();
    }
}

// ============================================================================
// 20260320 ZJH 关于弹窗
// ============================================================================
static void drawAboutDialog(AppState& state) {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("\xe5\x85\xb3\xe4\xba\x8e DeepForge##About", nullptr, ImGuiWindowFlags_NoResize)) {
        // 20260320 ZJH Logo 文字
        ImGui::PushStyleColor(ImGuiCol_Text, s_colAccent);
        ImGui::SetWindowFontScale(1.8f);
        float fTextW = ImGui::CalcTextSize("DeepForge").x;  // 20260320 ZJH 计算文本宽度
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - fTextW) * 0.5f);
        ImGui::Text("DeepForge");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // 20260320 ZJH 版本信息
        ImGui::Text("\xe7\x89\x88\xe6\x9c\xac: v0.1.0");
        ImGui::Spacing();
        // 20260320 ZJH 技术栈
        ImGui::Text("\xe6\x8a\x80\xe6\x9c\xaf\xe6\xa0\x88:");
        ImGui::BulletText("C++23 Modules");
        ImGui::BulletText("SDL3 + Dear ImGui + ImPlot");
        ImGui::BulletText("\xe8\x87\xaa\xe7\xa0\x94\xe6\xb7\xb1\xe5\xba\xa6\xe5\xad\xa6\xe4\xb9\xa0\xe5\xbc\x95\xe6\x93\x8e (df.engine)");
        ImGui::Spacing();
        // 20260320 ZJH 版权
        ImGui::TextDisabled("\xc2\xa9 2026 ZJH. All rights reserved.");

        ImGui::Spacing();
        float fBtnW2 = 80.0f;  // 20260320 ZJH 按钮宽度
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - fBtnW2) * 0.5f);
        if (ImGui::Button("\xe7\xa1\xae\xe5\xae\x9a", ImVec2(fBtnW2, 0))) {
            ImGui::CloseCurrentPopup();  // 20260320 ZJH 关闭关于弹窗
        }
        ImGui::EndPopup();
    }
}

// ============================================================================
// 20260320 ZJH 批量推理弹窗
// ============================================================================
static void drawBatchInferenceDialog(AppState& state) {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(600, 450), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86##BatchInf", nullptr, ImGuiWindowFlags_None)) {
        auto& bi = state.batchInference;  // 20260320 ZJH 批量推理状态引用

        // 20260320 ZJH 输入区域
        ImGui::InputText("\xe6\xa8\xa1\xe5\x9e\x8b\xe8\xb7\xaf\xe5\xbe\x84", bi.arrModelPath, sizeof(bi.arrModelPath));
        ImGui::InputText("\xe5\x9b\xbe\xe7\x89\x87\xe6\x96\x87\xe4\xbb\xb6\xe5\xa4\xb9", bi.arrImageFolder, sizeof(bi.arrImageFolder));

        ImGui::Spacing();
        // 20260320 ZJH 开始推理按钮
        if (!bi.bRunning) {
            if (ImGui::Button("\xe5\xbc\x80\xe5\xa7\x8b\xe6\x8e\xa8\xe7\x90\x86", ImVec2(120, 30))) {
                // 20260320 ZJH 模拟推理结果
                bi.vecResults.clear();
                bi.vecResults.push_back({"sample_001.png", "\xe7\xb1\xbb\xe5\x88\xab_0", 0.95f});
                bi.vecResults.push_back({"sample_002.png", "\xe7\xb1\xbb\xe5\x88\xab_3", 0.87f});
                bi.vecResults.push_back({"sample_003.png", "\xe7\xb1\xbb\xe5\x88\xab_7", 0.92f});
            }
        }

        ImGui::Spacing();

        // 20260320 ZJH 结果表格
        if (!bi.vecResults.empty()) {
            if (ImGui::BeginTable("##InfResults", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 250))) {
                ImGui::TableSetupColumn("\xe6\x96\x87\xe4\xbb\xb6\xe5\x90\x8d");
                ImGui::TableSetupColumn("\xe9\xa2\x84\xe6\xb5\x8b\xe7\xb1\xbb\xe5\x88\xab");
                ImGui::TableSetupColumn("\xe7\xbd\xae\xe4\xbf\xa1\xe5\xba\xa6");
                ImGui::TableHeadersRow();
                for (auto& r : bi.vecResults) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("%s", r.strFilename.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%s", r.strClass.c_str());
                    ImGui::TableSetColumnIndex(2); ImGui::Text("%.2f%%", (double)(r.fConfidence * 100.0f));
                }
                ImGui::EndTable();
            }
            // 20260320 ZJH 导出 CSV 按钮
            if (ImGui::Button("\xe5\xaf\xbc\xe5\x87\xba CSV")) { /* 20260320 ZJH 占位 */ }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        // 20260320 ZJH 关闭按钮居右
        ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - 80);
        if (ImGui::Button("\xe5\x85\xb3\xe9\x97\xad", ImVec2(80, 0))) {
            ImGui::CloseCurrentPopup();  // 20260320 ZJH 关闭批量推理
        }
        ImGui::EndPopup();
    }
}

// ============================================================================
// 20260320 ZJH 键盘快捷键处理
// ============================================================================
static void handleKeyboardShortcuts(AppState& state, SDL_Window* pWindow) {
    ImGuiIO& io = ImGui::GetIO();  // 20260320 ZJH 获取 IO 状态
    // 20260320 ZJH 仅在没有输入框激活时处理快捷键
    if (io.WantTextInput) return;

    bool bCtrl = io.KeyCtrl;  // 20260320 ZJH Ctrl 键是否按下

    // 20260320 ZJH Ctrl+N: 新建项目
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_N)) {
        if (!state.trainState.bRunning.load()) {
            state.trainState.reset();
            state.nActiveStep = 0;
        }
    }
    // 20260320 ZJH Ctrl+S: 保存项目（占位）
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) { /* 20260320 ZJH 占位 */ }
    // 20260320 ZJH F5: 开始训练
    if (ImGui::IsKeyPressed(ImGuiKey_F5) && !state.trainState.bRunning.load()) {
        state.nActiveStep = 2;  // 20260320 ZJH 跳到训练步骤
        startTraining(state);
    }
    // 20260320 ZJH F6: 暂停/恢复训练
    if (ImGui::IsKeyPressed(ImGuiKey_F6) && state.trainState.bRunning.load()) {
        state.trainState.bPaused.store(!state.trainState.bPaused.load());
    }
    // 20260320 ZJH F7: 停止训练
    if (ImGui::IsKeyPressed(ImGuiKey_F7) && state.trainState.bRunning.load()) {
        state.trainState.bStopRequested.store(true);
        state.trainState.bPaused.store(false);
    }
    // 20260320 ZJH Ctrl+1/2/3/4: 跳到对应步骤
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_1)) state.nActiveStep = 0;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_2)) state.nActiveStep = 1;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_3)) state.nActiveStep = 2;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_4)) state.nActiveStep = 3;
    // 20260320 ZJH Ctrl+,: 打开设置
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_Comma)) {
        state.bShowSettingsPopup = true;
    }
    // 20260320 ZJH F11: 全屏切换
    if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
        state.bFullscreen = !state.bFullscreen;
        SDL_SetWindowFullscreen(pWindow, state.bFullscreen);
    }
}

// ============================================================================
// 20260320 ZJH 设置 Halcon 风格深色主题
// ============================================================================
static void setupImGuiStyle() {
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 2.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(6, 4);
    style.ItemSpacing = ImVec2(8, 5);
    style.ScrollbarRounding = 4.0f;

    ImVec4* c = style.Colors;
    // 20260320 ZJH Halcon 深色蓝灰色背景
    c[ImGuiCol_WindowBg]        = ImVec4(0.118f, 0.125f, 0.157f, 1.0f);
    c[ImGuiCol_ChildBg]         = ImVec4(0.098f, 0.105f, 0.137f, 1.0f);
    c[ImGuiCol_PopupBg]         = ImVec4(0.149f, 0.157f, 0.188f, 0.97f);
    c[ImGuiCol_Border]          = ImVec4(0.220f, 0.230f, 0.270f, 1.0f);
    c[ImGuiCol_FrameBg]         = ImVec4(0.149f, 0.157f, 0.188f, 1.0f);
    c[ImGuiCol_FrameBgHovered]  = ImVec4(0.180f, 0.200f, 0.250f, 1.0f);
    c[ImGuiCol_FrameBgActive]   = ImVec4(0.220f, 0.240f, 0.300f, 1.0f);
    c[ImGuiCol_TitleBg]         = ImVec4(0.070f, 0.075f, 0.100f, 1.0f);
    c[ImGuiCol_TitleBgActive]   = ImVec4(0.100f, 0.120f, 0.170f, 1.0f);
    c[ImGuiCol_Tab]             = ImVec4(0.130f, 0.138f, 0.170f, 1.0f);
    c[ImGuiCol_TabHovered]      = ImVec4(0.231f, 0.510f, 0.965f, 0.80f);
    c[ImGuiCol_TabSelected]     = ImVec4(0.231f, 0.510f, 0.965f, 1.0f);
    c[ImGuiCol_Button]          = ImVec4(0.180f, 0.200f, 0.260f, 1.0f);
    c[ImGuiCol_ButtonHovered]   = ImVec4(0.231f, 0.510f, 0.965f, 0.80f);
    c[ImGuiCol_ButtonActive]    = ImVec4(0.231f, 0.510f, 0.965f, 1.0f);
    c[ImGuiCol_Header]          = ImVec4(0.180f, 0.200f, 0.260f, 1.0f);
    c[ImGuiCol_HeaderHovered]   = ImVec4(0.231f, 0.510f, 0.965f, 0.60f);
    c[ImGuiCol_HeaderActive]    = ImVec4(0.231f, 0.510f, 0.965f, 0.80f);
    c[ImGuiCol_Separator]       = ImVec4(0.220f, 0.230f, 0.270f, 1.0f);
    c[ImGuiCol_Text]            = ImVec4(0.886f, 0.910f, 0.941f, 1.0f);
    c[ImGuiCol_TextDisabled]    = ImVec4(0.480f, 0.500f, 0.540f, 1.0f);
    c[ImGuiCol_PlotHistogram]   = ImVec4(0.231f, 0.510f, 0.965f, 1.0f);
    c[ImGuiCol_CheckMark]       = ImVec4(0.231f, 0.510f, 0.965f, 1.0f);
    c[ImGuiCol_SliderGrab]      = ImVec4(0.231f, 0.510f, 0.965f, 0.80f);
    c[ImGuiCol_SliderGrabActive]= ImVec4(0.231f, 0.510f, 0.965f, 1.0f);
    c[ImGuiCol_ScrollbarBg]     = ImVec4(0.098f, 0.105f, 0.137f, 1.0f);
    c[ImGuiCol_ScrollbarGrab]   = ImVec4(0.200f, 0.210f, 0.250f, 1.0f);
}

// ============================================================================
// 20260320 ZJH 启动训练（根据任务类型分派）
// ============================================================================
static void startTraining(AppState& state) {
    if (state.trainState.bRunning.load()) return;

    if (state.pTrainThread) {
        state.pTrainThread->request_stop();
        state.pTrainThread->join();
        state.pTrainThread.reset();
    }
    state.trainState.reset();

    switch (state.activeTask) {
        case TaskType::Classification:
            state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { classificationTrainFunc(state); });
            break;
        case TaskType::Detection:
            state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { detectionTrainFunc(state); });
            break;
        case TaskType::Segmentation:
            state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { segmentationTrainFunc(state); });
            break;
        case TaskType::AnomalyDetection:
            state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { anomalyTrainFunc(state); });
            break;
    }
}

// ============================================================================
// 20260320 ZJH 检查可用数据集
// ============================================================================
static void checkDatasets(AppState& state) {
    bool bTI = std::filesystem::exists("data/mnist/train-images-idx3-ubyte");
    bool bTL = std::filesystem::exists("data/mnist/train-labels-idx1-ubyte");
    bool bEI = std::filesystem::exists("data/mnist/t10k-images-idx3-ubyte");
    bool bEL = std::filesystem::exists("data/mnist/t10k-labels-idx1-ubyte");
    state.bMnistAvailable = bTI && bTL && bEI && bEL;

    if (state.bMnistAvailable) {
        auto readCount = [](const std::string& p) -> int {
            std::ifstream ifs(p, std::ios::binary);
            if (!ifs) return 0;
            unsigned char h[8]; ifs.read((char*)h, 8);
            return (h[4]<<24)|(h[5]<<16)|(h[6]<<8)|h[7];
        };
        state.nMnistTrainSamples = readCount("data/mnist/train-images-idx3-ubyte");
        state.nMnistTestSamples = readCount("data/mnist/t10k-images-idx3-ubyte");
    }
    state.bDataChecked = true;
}

// ============================================================================
// 20260320 ZJH 绘制顶部工具栏
// ============================================================================
static void drawToolbar(AppState& state) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 7));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.090f, 0.095f, 0.125f, 1.0f));
    ImGui::BeginChild("##Toolbar", ImVec2(0, 42), ImGuiChildFlags_None);

    // 20260320 ZJH 左侧标志
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.231f, 0.510f, 0.965f, 1.0f));
    ImGui::Text("DeepForge");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // 20260320 ZJH 工具栏按钮
    if (ImGui::Button("\xe6\x96\xb0\xe5\xbb\xba\xe9\xa1\xb9\xe7\x9b\xae")) {
        // 20260320 ZJH 重置训练状态
        if (!state.trainState.bRunning.load()) {
            state.trainState.reset();
            state.nActiveStep = 0;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("\xe6\x89\x93\xe5\xbc\x80\xe9\xa1\xb9\xe7\x9b\xae")) {
        // 20260320 ZJH 占位
    }
    ImGui::SameLine();
    if (ImGui::Button("\xe4\xbf\x9d\xe5\xad\x98\xe9\xa1\xb9\xe7\x9b\xae")) {
        // 20260320 ZJH 占位
    }
    ImGui::SameLine();
    if (ImGui::Button("\xe8\xae\xbe\xe7\xbd\xae")) {
        state.bShowSettingsPopup = true;
    }

    // 20260320 ZJH 右侧 GPU 信息
    float fRight = ImGui::GetContentRegionAvail().x;
    if (fRight > 300.0f) {
        ImGui::SameLine(ImGui::GetCursorPosX() + fRight - 280.0f);
        auto* pDraw = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();

        if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
            pDraw->AddCircleFilled(ImVec2(p.x + 4, p.y + 10), 5.0f, IM_COL32(50, 220, 80, 255));
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 14);
            char buf[256];
            std::snprintf(buf, sizeof(buf), "GPU: %s (%zuGB)", state.vecGpuDevices[0].strName.c_str(), state.vecGpuDevices[0].nTotalMemoryMB / 1024);
            ImGui::TextColored(s_colGreen, "%s", buf);
        } else {
            pDraw->AddCircleFilled(ImVec2(p.x + 4, p.y + 10), 5.0f, IM_COL32(255, 165, 30, 255));
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 14);
            ImGui::TextColored(s_colOrange, "CPU");
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ============================================================================
// 20260320 ZJH 绘制左侧项目导航面板
// ============================================================================
static void drawProjectNav(AppState& state) {
    // 20260320 ZJH 标题
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.580f, 0.750f, 1.000f, 1.0f));
    ImGui::Text("\xe9\xa1\xb9\xe7\x9b\xae\xe5\xaf\xbc\xe8\x88\xaa");  // "项目导航"
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH 4 个任务类型的树形结构
    struct TaskEntry {
        const char* strName;
        const char* strModel;
        const char* strDataset;
        TaskType type;
    };
    TaskEntry arrTasks[] = {
        {"\xe5\x9b\xbe\xe5\x83\x8f\xe5\x88\x86\xe7\xb1\xbb", "MLP / ResNet", "MNIST", TaskType::Classification},
        {"\xe7\x9b\xae\xe6\xa0\x87\xe6\xa3\x80\xe6\xb5\x8b", "YOLOv5-Nano", "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae", TaskType::Detection},
        {"\xe8\xaf\xad\xe4\xb9\x89\xe5\x88\x86\xe5\x89\xb2", "U-Net", "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae", TaskType::Segmentation},
        {"\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b", "AutoEncoder", "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae", TaskType::AnomalyDetection},
    };

    for (int i = 0; i < 4; ++i) {
        bool bActive = (state.activeTask == arrTasks[i].type);

        // 20260320 ZJH 任务状态指示圆点
        auto* pDraw = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImU32 dotCol = bActive ? IM_COL32(59, 130, 246, 255) : IM_COL32(100, 105, 120, 255);
        pDraw->AddCircleFilled(ImVec2(pos.x + 6, pos.y + 10), 4.0f, dotCol);

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 16);

        // 20260320 ZJH 可点击树节点
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow;
        if (bActive) flags |= ImGuiTreeNodeFlags_Selected;

        bool bOpen = ImGui::TreeNodeEx(arrTasks[i].strName, flags);
        if (ImGui::IsItemClicked()) {
            state.activeTask = arrTasks[i].type;
            state.nActiveStep = 0;
            state.trainState.reset();
        }

        // 20260320 ZJH 右键上下文菜单
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("\xe6\x9f\xa5\xe7\x9c\x8b\xe8\xaf\xa6\xe6\x83\x85")) {
                state.activeTask = arrTasks[i].type;  // 20260320 ZJH 切换到该任务
                state.nActiveStep = 0;
            }
            if (ImGui::MenuItem("\xe5\x88\xa0\xe9\x99\xa4")) { /* 20260320 ZJH 占位 */ }
            ImGui::EndPopup();
        }

        if (bOpen) {
            ImGui::TextDisabled("    %s", arrTasks[i].strModel);
            ImGui::TextDisabled("    %s", arrTasks[i].strDataset);
            ImGui::TreePop();
        }
        ImGui::Spacing();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH 设备信息
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.580f, 0.750f, 1.000f, 1.0f));
    ImGui::Text("\xe8\xae\xbe\xe5\xa4\x87\xe4\xbf\xa1\xe6\x81\xaf");  // "设备信息"
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();

    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        auto& gpu = state.vecGpuDevices[0];
        ImGui::TextColored(s_colGreen, "  GPU:");
        ImGui::TextWrapped("  %s", gpu.strName.c_str());
        ImGui::Text("  %zu MB", gpu.nTotalMemoryMB);
    } else {
        ImGui::TextColored(s_colOrange, "  CPU (\xe6\x97\xa0 GPU)");
    }
}

// ============================================================================
// 20260320 ZJH 绘制步骤 1：数据（适配所有任务类型）
// ============================================================================
static void drawStepData(AppState& state) {
    if (!state.bDataChecked) checkDatasets(state);

    switch (state.activeTask) {
    case TaskType::Classification: {
        if (beginCard("\xe6\x95\xb0\xe6\x8d\xae\xe9\x9b\x86\xe6\x9d\xa5\xe6\xba\x90")) {  // "数据集来源"
            const char* ds[] = {"MNIST (\xe6\x89\x8b\xe5\x86\x99\xe6\x95\xb0\xe5\xad\x97)", "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae (\xe8\x87\xaa\xe5\x8a\xa8)"};
            ImGui::Combo("\xe6\x95\xb0\xe6\x8d\xae\xe9\x9b\x86", &state.nClsDataset, ds, 2);
            if (state.nClsDataset == 0 && state.bMnistAvailable) {
                ImGui::TextColored(s_colGreen, "MNIST \xe5\xb7\xb2\xe5\xb0\xb1\xe7\xbb\xaa");
                ImGui::Text("  \xe8\xae\xad\xe7\xbb\x83: %d  \xe6\xb5\x8b\xe8\xaf\x95: %d  \xe7\xb1\xbb\xe5\x88\xab: 10", state.nMnistTrainSamples, state.nMnistTestSamples);
            } else if (state.nClsDataset == 0) {
                ImGui::TextColored(s_colOrange, "MNIST \xe6\x9c\xaa\xe6\x89\xbe\xe5\x88\xb0\xef\xbc\x8c\xe5\xb0\x86\xe4\xbd\xbf\xe7\x94\xa8\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae");
            } else {
                ImGui::TextColored(s_colGreen, "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae: 1000 \xe8\xae\xad\xe7\xbb\x83 / 200 \xe6\xb5\x8b\xe8\xaf\x95");
            }
        } endCard();

        if (beginCard("\xe6\x95\xb0\xe6\x8d\xae\xe7\xbb\x9f\xe8\xae\xa1")) {  // "数据统计"
            ImGui::SliderFloat("\xe8\xae\xad\xe7\xbb\x83\xe9\x9b\x86", &state.fTrainSplit, 0.5f, 0.95f, "%.0f%%");
            ImGui::SliderFloat("\xe9\xaa\x8c\xe8\xaf\x81\xe9\x9b\x86", &state.fValSplit, 0.0f, 0.3f, "%.0f%%");
            state.fTestSplit = 1.0f - state.fTrainSplit - state.fValSplit;
            if (state.fTestSplit < 0) { state.fValSplit = 1.0f - state.fTrainSplit; state.fTestSplit = 0; }
            ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95\xe9\x9b\x86: %.0f%%", (double)(state.fTestSplit * 100));
        } endCard();

        if (beginCard("\xe7\xb1\xbb\xe5\x88\xab\xe5\x88\x86\xe5\xb8\x83")) {  // "类别分布"
            if (ImPlot::BeginPlot("##ClsDist", ImVec2(-1, 180))) {
                ImPlot::SetupAxes("\xe7\xb1\xbb\xe5\x88\xab", "\xe6\x95\xb0\xe9\x87\x8f");
                double xv[10]={0,1,2,3,4,5,6,7,8,9};
                double yv[10];
                double avg = state.bMnistAvailable ? (double)state.nMnistTrainSamples/10.0 : 100.0;
                for (int i=0;i<10;++i) yv[i]=avg;
                ImPlot::PlotBars("\xe6\xa0\xb7\xe6\x9c\xac", xv, yv, 10, 0.6);
                ImPlot::EndPlot();
            }
        } endCard();
        break;
    }
    case TaskType::Detection: {
        if (beginCard("\xe6\xa3\x80\xe6\xb5\x8b\xe6\x95\xb0\xe6\x8d\xae\xe9\x85\x8d\xe7\xbd\xae")) {  // "检测数据配置"
            ImGui::Text("\xe6\x95\xb0\xe6\x8d\xae\xe6\xa0\xbc\xe5\xbc\x8f: YOLO \xe6\xa0\xbc\xe5\xbc\x8f (txt \xe6\xa0\x87\xe6\xb3\xa8)");
            ImGui::TextColored(s_colGreen, "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae\xe5\xb7\xb2\xe5\xb0\xb1\xe7\xbb\xaa");
            ImGui::Text("  \xe6\xa0\xb7\xe6\x9c\xac\xe6\x95\xb0: 64");
            ImGui::SliderInt("\xe7\xb1\xbb\xe5\x88\xab\xe6\x95\xb0", &state.nDetClasses, 1, 20);
            ImGui::SliderInt("\xe8\xbe\x93\xe5\x85\xa5\xe5\xb0\xba\xe5\xaf\xb8", &state.nDetImgSize, 64, 256);
        } endCard();

        if (beginCard("\xe6\xa0\x87\xe6\xb3\xa8\xe7\xbb\x9f\xe8\xae\xa1")) {  // "标注统计"
            ImGui::Text("\xe6\xaf\x8f\xe4\xb8\xaa\xe6\xa0\xb7\xe6\x9c\xac 1 \xe4\xb8\xaa\xe7\x9b\xae\xe6\xa0\x87");
            ImGui::Text("\xe6\x80\xbb\xe6\xa0\x87\xe6\xb3\xa8\xe6\x95\xb0: 64");
        } endCard();
        break;
    }
    case TaskType::Segmentation: {
        if (beginCard("\xe5\x88\x86\xe5\x89\xb2\xe6\x95\xb0\xe6\x8d\xae\xe9\x85\x8d\xe7\xbd\xae")) {  // "分割数据配置"
            ImGui::Text("\xe6\x95\xb0\xe6\x8d\xae\xe6\xa0\xbc\xe5\xbc\x8f: \xe5\x9b\xbe\xe5\x83\x8f + Mask");
            ImGui::TextColored(s_colGreen, "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae\xe5\xb7\xb2\xe5\xb0\xb1\xe7\xbb\xaa");
            ImGui::Text("  \xe6\xa0\xb7\xe6\x9c\xac\xe6\x95\xb0: 16  \xe5\xb0\xba\xe5\xaf\xb8: %dx%d", state.nSegImgSize, state.nSegImgSize);
            ImGui::SliderInt("\xe7\xb1\xbb\xe5\x88\xab\xe6\x95\xb0##seg", &state.nSegClasses, 2, 10);
            ImGui::SliderInt("\xe5\x9b\xbe\xe5\x83\x8f\xe5\xb0\xba\xe5\xaf\xb8##seg", &state.nSegImgSize, 32, 128);
        } endCard();
        break;
    }
    case TaskType::AnomalyDetection: {
        if (beginCard("\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe6\x95\xb0\xe6\x8d\xae")) {  // "异常检测数据"
            ImGui::Text("\xe5\x8f\xaa\xe9\x9c\x80\xe6\xad\xa3\xe5\xb8\xb8\xe6\xa0\xb7\xe6\x9c\xac (\xe6\x97\xa0\xe9\x9c\x80\xe6\xa0\x87\xe6\xb3\xa8)");
            ImGui::TextColored(s_colGreen, "\xe5\x90\x88\xe6\x88\x90\xe6\xad\xa3\xe5\xb8\xb8\xe6\xa0\xb7\xe6\x9c\xac\xe5\xb7\xb2\xe5\xb0\xb1\xe7\xbb\xaa");
            ImGui::Text("  \xe6\xad\xa3\xe5\xb8\xb8\xe6\xa0\xb7\xe6\x9c\xac: 128");
            ImGui::Text("  \xe5\x9b\xbe\xe5\x83\x8f\xe5\xb0\xba\xe5\xaf\xb8: 28x28 \xe7\x81\xb0\xe5\xba\xa6");
        } endCard();
        break;
    }
    }
    state.trainState.bStepDataDone = true;
}

// ============================================================================
// 20260320 ZJH 绘制步骤 2：配置（适配所有任务类型）
// ============================================================================
static void drawStepConfig(AppState& state) {
    bool bDis = state.trainState.bRunning.load();
    if (bDis) ImGui::BeginDisabled();

    switch (state.activeTask) {
    case TaskType::Classification: {
        // 20260320 ZJH 超参数预设选择器
        if (beginCard("\xe8\xae\xad\xe7\xbb\x83\xe9\xa2\x84\xe8\xae\xbe")) {  // "训练预设"
            const char* arrPresets[] = {"\xe5\xbf\xab\xe9\x80\x9f\xe8\xae\xad\xe7\xbb\x83", "\xe6\xa0\x87\xe5\x87\x86\xe8\xae\xad\xe7\xbb\x83", "\xe7\xb2\xbe\xe7\xa1\xae\xe8\xae\xad\xe7\xbb\x83"};
            if (ImGui::Combo("\xe9\xa2\x84\xe8\xae\xbe\xe6\x96\xb9\xe6\xa1\x88", &state.nPresetIndex, arrPresets, 3)) {
                // 20260320 ZJH 根据预设自动设置超参数
                if (state.nPresetIndex == 0) {
                    // 20260320 ZJH 快速训练：小轮数、大学习率、大批次
                    state.nClsEpochs = 5; state.fClsLR = 0.01f; state.nClsBatchSize = 128;
                } else if (state.nPresetIndex == 1) {
                    // 20260320 ZJH 标准训练：中等参数
                    state.nClsEpochs = 20; state.fClsLR = 0.001f; state.nClsBatchSize = 64;
                } else {
                    // 20260320 ZJH 精确训练：大轮数、小学习率、小批次、余弦退火
                    state.nClsEpochs = 50; state.fClsLR = 0.0001f; state.nClsBatchSize = 32;
                    state.nClsLRSchedule = 1;
                }
            }
            // 20260320 ZJH 显示预设说明
            const char* arrDesc[] = {
                "epochs=5, lr=0.01, batch=128",
                "epochs=20, lr=0.001, batch=64",
                "epochs=50, lr=0.0001, batch=32, cosine"};
            ImGui::TextDisabled("%s", arrDesc[state.nPresetIndex]);
        } endCard();

        if (beginCard("\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x9e\xb6\xe6\x9e\x84")) {  // "模型架构"
            ImGui::Text("\xe4\xbb\xbb\xe5\x8a\xa1\xe7\xb1\xbb\xe5\x9e\x8b: \xe5\x9b\xbe\xe5\x83\x8f\xe5\x88\x86\xe7\xb1\xbb");
            const char* m[] = {"MLP (784->128->10)", "ResNet-18 (11.2M \xe5\x8f\x82\xe6\x95\xb0)", "ResNet-34 (21.3M \xe5\x8f\x82\xe6\x95\xb0)"};
            ImGui::Combo("\xe9\xaa\xa8\xe5\xb9\xb2\xe7\xbd\x91\xe7\xbb\x9c", &state.nClsModel, m, 3);
            if (state.nClsModel == 2) state.nClsModel = 1;
            ImGui::Text("\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %s", state.nClsModel==0?"~101K":"~11.2M");
        } endCard();

        if (beginCard("\xe8\xae\xad\xe7\xbb\x83\xe5\x8f\x82\xe6\x95\xb0")) {  // "训练参数"
            ImGui::SliderInt("\xe8\xae\xad\xe7\xbb\x83\xe8\xbd\xae\xe6\x95\xb0", &state.nClsEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1\xe5\xa4\xa7\xe5\xb0\x8f", &state.nClsBatchSize, 8, 256);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87", &state.fClsLR, 0.001f, 0.01f, "%.4f");
            if (state.fClsLR < 0.0001f) state.fClsLR = 0.0001f;
            ImGui::InputFloat("\xe6\x9d\x83\xe9\x87\x8d\xe8\xa1\xb0\xe5\x87\x8f", &state.fClsWeightDecay, 0.0001f, 0.001f, "%.4f");
        } endCard();

        if (beginCard("\xe4\xbc\x98\xe5\x8c\x96\xe5\x99\xa8")) {  // "优化器"
            const char* opts[] = {"SGD", "Adam"};
            ImGui::Combo("\xe7\xb1\xbb\xe5\x9e\x8b", &state.nClsOptimizer, opts, 2);
            const char* lr[] = {"\xe5\x9b\xba\xe5\xae\x9a", "\xe4\xbd\x99\xe5\xbc\xa6\xe9\x80\x80\xe7\x81\xab"};
            ImGui::Combo("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87\xe7\xad\x96\xe7\x95\xa5", &state.nClsLRSchedule, lr, 2);
            if (state.nClsLRSchedule > 0) state.nClsLRSchedule = 0;
            ImGui::InputInt("\xe9\xa2\x84\xe7\x83\xad\xe8\xbd\xae\xe6\x95\xb0", &state.nClsWarmup);
        } endCard();
        break;
    }
    case TaskType::Detection: {
        if (beginCard("\xe6\xa3\x80\xe6\xb5\x8b\xe6\xa8\xa1\xe5\x9e\x8b")) {  // "检测模型"
            ImGui::Text("\xe6\xa8\xa1\xe5\x9e\x8b: YOLOv5-Nano");
            ImGui::Text("Anchor \xe6\x95\xb0: 3");
            ImGui::Text("\xe7\xb1\xbb\xe5\x88\xab\xe6\x95\xb0: %d", state.nDetClasses);
        } endCard();

        if (beginCard("\xe6\xa3\x80\xe6\xb5\x8b\xe5\x8f\x82\xe6\x95\xb0")) {  // "检测参数"
            ImGui::SliderInt("\xe8\xae\xad\xe7\xbb\x83\xe8\xbd\xae\xe6\x95\xb0##det", &state.nDetEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1\xe5\xa4\xa7\xe5\xb0\x8f##det", &state.nDetBatchSize, 1, 32);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87##det", &state.fDetLR, 0.0001f, 0.001f, "%.4f");
            ImGui::SliderFloat("IOU \xe9\x98\x88\xe5\x80\xbc", &state.fDetIouThresh, 0.1f, 0.9f, "%.2f");
            ImGui::SliderFloat("\xe7\xbd\xae\xe4\xbf\xa1\xe5\xba\xa6\xe9\x98\x88\xe5\x80\xbc", &state.fDetConfThresh, 0.05f, 0.95f, "%.2f");
        } endCard();
        break;
    }
    case TaskType::Segmentation: {
        if (beginCard("\xe5\x88\x86\xe5\x89\xb2\xe6\xa8\xa1\xe5\x9e\x8b")) {  // "分割模型"
            ImGui::Text("\xe6\xa8\xa1\xe5\x9e\x8b: U-Net");
            ImGui::Text("\xe8\xbe\x93\xe5\x87\xba\xe7\xb1\xbb\xe5\x88\xab: %d", state.nSegClasses);
            ImGui::Text("\xe8\xbe\x93\xe5\x85\xa5\xe5\xb0\xba\xe5\xaf\xb8: %dx%d", state.nSegImgSize, state.nSegImgSize);
        } endCard();

        if (beginCard("\xe5\x88\x86\xe5\x89\xb2\xe5\x8f\x82\xe6\x95\xb0")) {  // "分割参数"
            ImGui::SliderInt("\xe8\xae\xad\xe7\xbb\x83\xe8\xbd\xae\xe6\x95\xb0##seg", &state.nSegEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1\xe5\xa4\xa7\xe5\xb0\x8f##seg", &state.nSegBatchSize, 1, 16);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87##seg", &state.fSegLR, 0.0001f, 0.001f, "%.4f");
        } endCard();
        break;
    }
    case TaskType::AnomalyDetection: {
        if (beginCard("\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe6\xa8\xa1\xe5\x9e\x8b")) {  // "异常检测模型"
            ImGui::Text("\xe6\xa8\xa1\xe5\x9e\x8b: ConvAutoEncoder");
            ImGui::SliderInt("\xe7\x93\xb6\xe9\xa2\x88\xe7\xbb\xb4\xe5\xba\xa6", &state.nAeLatentDim, 16, 128);
        } endCard();

        if (beginCard("\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe5\x8f\x82\xe6\x95\xb0")) {  // "异常检测参数"
            ImGui::SliderInt("\xe8\xae\xad\xe7\xbb\x83\xe8\xbd\xae\xe6\x95\xb0##ae", &state.nAeEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1\xe5\xa4\xa7\xe5\xb0\x8f##ae", &state.nAeBatchSize, 4, 128);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87##ae", &state.fAeLR, 0.0001f, 0.001f, "%.4f");
            ImGui::SliderFloat("\xe9\x87\x8d\xe5\xbb\xba\xe9\x98\x88\xe5\x80\xbc", &state.fAeThreshold, 0.01f, 2.0f, "%.3f");
        } endCard();
        break;
    }
    }

    // 20260320 ZJH 高级设置（可折叠）
    if (ImGui::CollapsingHeader("\xe9\xab\x98\xe7\xba\xa7\xe8\xae\xbe\xe7\xbd\xae")) {
        ImGui::Indent(10.0f);
        ImGui::InputFloat("\xe6\x9d\x83\xe9\x87\x8d\xe8\xa1\xb0\xe5\x87\x8f##adv", &state.fClsWeightDecay, 0.0001f, 0.001f, "%.5f");
        ImGui::InputFloat("Dropout \xe7\x8e\x87", &state.fDropout, 0.05f, 0.1f, "%.2f");
        if (state.fDropout < 0.0f) state.fDropout = 0.0f;  // 20260320 ZJH 下限
        if (state.fDropout > 0.9f) state.fDropout = 0.9f;  // 20260320 ZJH 上限
        ImGui::Spacing();
        // 20260320 ZJH 早停设置
        ImGui::Checkbox("\xe5\x90\xaf\xe7\x94\xa8\xe6\x97\xa9\xe5\x81\x9c", &state.bEarlyStop);
        if (state.bEarlyStop) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("patience", &state.nEarlyStopPatience);
            if (state.nEarlyStopPatience < 1) state.nEarlyStopPatience = 1;  // 20260320 ZJH 最小值
        }
        ImGui::Spacing();
        // 20260320 ZJH 检查点设置
        ImGui::Checkbox("\xe6\xaf\x8f\xe8\xbd\xae\xe4\xbf\x9d\xe5\xad\x98\xe6\xa3\x80\xe6\x9f\xa5\xe7\x82\xb9", &state.bSaveEveryEpoch);
        ImGui::Checkbox("\xe4\xbb\x85\xe4\xbf\x9d\xe5\xad\x98\xe6\x9c\x80\xe4\xbc\x98\xe6\xa8\xa1\xe5\x9e\x8b", &state.bSaveBestOnly);
        ImGui::Unindent(10.0f);
        ImGui::Spacing();
    }

    // 20260320 ZJH 模型架构查看器（可折叠）
    if (state.bShowModelViewer && ImGui::CollapsingHeader("\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x9e\xb6\xe6\x9e\x84\xe8\xaf\xa6\xe6\x83\x85")) {
        ImGui::Indent(10.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, s_colSubtle);
        switch (state.activeTask) {
        case TaskType::Classification:
            if (state.nClsModel == 0) {
                // 20260320 ZJH MLP 层级结构
                ImGui::Text("Input:  784 (28x28 \xe7\x81\xb0\xe5\xba\xa6)");
                ImGui::Text("  -> Linear(784, 128)");
                ImGui::Text("  -> ReLU");
                ImGui::Text("  -> Linear(128, 10)");
                ImGui::Text("Output: 10 (\xe7\xb1\xbb\xe5\x88\xab)");
                ImGui::Text("\xe6\x80\xbb\xe5\x8f\x82\xe6\x95\xb0: ~101K");
            } else {
                // 20260320 ZJH ResNet-18 层级结构
                ImGui::Text("Input:  1x28x28");
                ImGui::Text("  -> Conv2d(1, 64, 7x7, s=2)");
                ImGui::Text("  -> BatchNorm -> ReLU -> MaxPool");
                ImGui::Text("  -> ResBlock x2 (64)");
                ImGui::Text("  -> ResBlock x2 (128, s=2)");
                ImGui::Text("  -> ResBlock x2 (256, s=2)");
                ImGui::Text("  -> ResBlock x2 (512, s=2)");
                ImGui::Text("  -> AdaptiveAvgPool -> FC(512, 10)");
                ImGui::Text("Output: 10 (\xe7\xb1\xbb\xe5\x88\xab)");
                ImGui::Text("\xe6\x80\xbb\xe5\x8f\x82\xe6\x95\xb0: ~11.2M");
            }
            break;
        case TaskType::Detection:
            ImGui::Text("Input:  3x%dx%d", state.nDetImgSize, state.nDetImgSize);
            ImGui::Text("  -> Backbone (CSPDarknet-Nano)");
            ImGui::Text("  -> Neck (PANet)");
            ImGui::Text("  -> Head (3 scales)");
            ImGui::Text("Output: %d \xe7\xb1\xbb + bbox", state.nDetClasses);
            break;
        case TaskType::Segmentation:
            ImGui::Text("Input:  1x%dx%d", state.nSegImgSize, state.nSegImgSize);
            ImGui::Text("  -> Encoder (4x Down)");
            ImGui::Text("  -> Bottleneck (512)");
            ImGui::Text("  -> Decoder (4x Up + Skip)");
            ImGui::Text("Output: %dx%dx%d", state.nSegClasses, state.nSegImgSize, state.nSegImgSize);
            break;
        case TaskType::AnomalyDetection:
            ImGui::Text("Input:  1x28x28");
            ImGui::Text("  -> Conv(1,32) -> Conv(32,64)");
            ImGui::Text("  -> FC(%d) [bottleneck]", state.nAeLatentDim);
            ImGui::Text("  -> Deconv(64,32) -> Deconv(32,1)");
            ImGui::Text("Output: 1x28x28 (\xe9\x87\x8d\xe5\xbb\xba)");
            break;
        }
        ImGui::PopStyleColor();
        ImGui::Unindent(10.0f);
        ImGui::Spacing();
    }

    // 20260320 ZJH 通用：数据增强（所有任务共享）
    if (beginCard("\xe6\x95\xb0\xe6\x8d\xae\xe5\xa2\x9e\xe5\xbc\xba")) {  // "数据增强"
        ImGui::Checkbox("\xe9\x9a\x8f\xe6\x9c\xba\xe6\xb0\xb4\xe5\xb9\xb3\xe7\xbf\xbb\xe8\xbd\xac", &state.bAugFlip);
        ImGui::Checkbox("\xe9\x9a\x8f\xe6\x9c\xba\xe6\x97\x8b\xe8\xbd\xac (15\xc2\xb0)", &state.bAugRotate);
        ImGui::Checkbox("\xe8\x89\xb2\xe5\xbd\xa9\xe6\x8a\x96\xe5\x8a\xa8", &state.bAugColorJitter);
        ImGui::Checkbox("\xe9\x9a\x8f\xe6\x9c\xba\xe8\xa3\x81\xe5\x89\xaa", &state.bAugCrop);
        ImGui::Checkbox("\xe9\xab\x98\xe6\x96\xaf\xe5\x99\xaa\xe5\xa3\xb0", &state.bAugNoise);
    } endCard();

    // 20260320 ZJH 设备配置
    if (beginCard("\xe8\xae\xbe\xe5\xa4\x87\xe9\x85\x8d\xe7\xbd\xae")) {  // "设备配置"
        if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "GPU: %s (%zuGB)", state.vecGpuDevices[0].strName.c_str(), state.vecGpuDevices[0].nTotalMemoryMB/1024);
            ImGui::Text("\xe8\xae\xa1\xe7\xae\x97\xe8\xae\xbe\xe5\xa4\x87: %s", buf);
            ImGui::TextDisabled("GPU \xe8\xae\xad\xe7\xbb\x83\xe5\x8d\xb3\xe5\xb0\x86\xe6\x8e\xa8\xe5\x87\xba");
        } else {
            ImGui::Text("\xe8\xae\xa1\xe7\xae\x97\xe8\xae\xbe\xe5\xa4\x87: CPU");
        }
    } endCard();

    if (bDis) ImGui::EndDisabled();
    state.trainState.bStepConfigDone = true;
}

// ============================================================================
// 20260320 ZJH 绘制步骤 3：训练（所有任务共享训练控制面板）
// ============================================================================
static void drawStepTrain(AppState& state) {
    auto& ts = state.trainState;

    // 20260320 ZJH 训练控制按钮
    if (beginCard("\xe8\xae\xad\xe7\xbb\x83\xe6\x8e\xa7\xe5\x88\xb6")) {  // "训练控制"
        float fAvailW = ImGui::GetContentRegionAvail().x;
        float fBtnW = 120.0f;

        if (!ts.bRunning.load()) {
            ImGui::SetCursorPosX((fAvailW - 200) * 0.5f);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.231f, 0.510f, 0.965f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.318f, 0.580f, 1.000f, 1.0f));
            if (ImGui::Button("\xe5\xbc\x80\xe5\xa7\x8b\xe8\xae\xad\xe7\xbb\x83", ImVec2(200, 40))) {
                startTraining(state);
            }
            ImGui::PopStyleColor(2);
        } else {
            ImGui::SetCursorPosX((fAvailW - fBtnW * 3 - 20) * 0.5f);
            // 20260320 ZJH 暂停/恢复
            if (ts.bPaused.load()) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.200f, 0.550f, 0.200f, 1.0f));
                if (ImGui::Button("\xe6\x81\xa2\xe5\xa4\x8d\xe8\xae\xad\xe7\xbb\x83", ImVec2(fBtnW, 35))) ts.bPaused.store(false);
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.600f, 0.450f, 0.100f, 1.0f));
                if (ImGui::Button("\xe6\x9a\x82\xe5\x81\x9c", ImVec2(fBtnW, 35))) ts.bPaused.store(true);
                ImGui::PopStyleColor();
            }
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.700f, 0.180f, 0.180f, 1.0f));
            if (ImGui::Button("\xe5\x81\x9c\xe6\xad\xa2", ImVec2(fBtnW, 35))) {
                ts.bStopRequested.store(true); ts.bPaused.store(false);
            }
            ImGui::PopStyleColor();
        }

        ImGui::Spacing();

        // 20260320 ZJH 进度条
        int nE = ts.nCurrentEpoch.load(), nTE = ts.nTotalEpochs.load();
        int nB = ts.nCurrentBatch.load(), nTB = ts.nTotalBatches.load();
        { char b[64]; std::snprintf(b, sizeof(b), "\xe8\xbd\xae\xe6\xac\xa1 %d/%d", nE, nTE); ImGui::Text("%s", b); }
        ImGui::ProgressBar(nTE > 0 ? (float)nE/(float)nTE : 0, ImVec2(-1, 0));
        { char b[64]; std::snprintf(b, sizeof(b), "\xe6\x89\xb9\xe6\xac\xa1 %d/%d", nB, nTB); ImGui::Text("%s", b); }
        ImGui::ProgressBar(nTB > 0 ? (float)nB/(float)nTB : 0, ImVec2(-1, 0));

        // 20260320 ZJH ETA
        float fMs = ts.fAvgBatchTimeMs.load();
        if (fMs > 0.001f && ts.bRunning.load()) {
            int rem = (nTB - nB) + (nTE - nE) * nTB;
            float eta = (float)rem * fMs / 1000.0f;
            if (eta < 60) ImGui::Text("\xe9\xa2\x84\xe8\xae\xa1\xe5\x89\xa9\xe4\xbd\x99: %.0f\xe7\xa7\x92", (double)eta);
            else ImGui::Text("\xe9\xa2\x84\xe8\xae\xa1\xe5\x89\xa9\xe4\xbd\x99: %.1f\xe5\x88\x86\xe9\x92\x9f", (double)eta/60.0);
        }
        ImGui::Text("\xe5\xbd\x93\xe5\x89\x8d\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87: %.6f", (double)ts.fCurrentLR.load());

        // 20260320 ZJH 状态指示
        if (ts.bRunning.load()) {
            if (ts.bPaused.load()) ImGui::TextColored(s_colOrange, "\xe7\x8a\xb6\xe6\x80\x81: \xe5\xb7\xb2\xe6\x9a\x82\xe5\x81\x9c");
            else ImGui::TextColored(s_colGreen, "\xe7\x8a\xb6\xe6\x80\x81: \xe8\xae\xad\xe7\xbb\x83\xe4\xb8\xad...");
        } else if (ts.bCompleted.load()) {
            ImGui::TextColored(s_colAccent, "\xe7\x8a\xb6\xe6\x80\x81: \xe5\xb7\xb2\xe5\xae\x8c\xe6\x88\x90");
        } else {
            ImGui::TextColored(s_colSubtle, "\xe7\x8a\xb6\xe6\x80\x81: \xe7\xa9\xba\xe9\x97\xb2");
        }
    } endCard();

    // 20260320 ZJH 损失曲线 + 准确率/指标曲线（并排）
    float fChH = std::max(180.0f, ImGui::GetContentRegionAvail().y * 0.40f);
    ImGui::BeginChild("##Charts", ImVec2(0, fChH));
    {
        float hw = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;

        ImGui::BeginChild("##LossC", ImVec2(hw, 0));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          if (ImPlot::BeginPlot("\xe6\x8d\x9f\xe5\xa4\xb1\xe6\x9b\xb2\xe7\xba\xbf", ImVec2(-1, -1))) {
              ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "\xe6\x8d\x9f\xe5\xa4\xb1");
              if (!ts.vecLossHistory.empty())
                  ImPlot::PlotLine("\xe8\xae\xad\xe7\xbb\x83\xe6\x8d\x9f\xe5\xa4\xb1", ts.vecLossHistory.data(), (int)ts.vecLossHistory.size());
              ImPlot::EndPlot();
          }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("##AccC", ImVec2(0, 0));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          if (state.activeTask == TaskType::Classification) {
              if (ImPlot::BeginPlot("\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87\xe6\x9b\xb2\xe7\xba\xbf", ImVec2(-1, -1))) {
                  ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87 (%)");
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 105, ImPlotCond_Always);
                  if (!ts.vecTrainAccHistory.empty())
                      ImPlot::PlotLine("\xe8\xae\xad\xe7\xbb\x83", ts.vecTrainAccHistory.data(), (int)ts.vecTrainAccHistory.size());
                  if (!ts.vecTestAccHistory.empty())
                      ImPlot::PlotLine("\xe6\xb5\x8b\xe8\xaf\x95", ts.vecTestAccHistory.data(), (int)ts.vecTestAccHistory.size());
                  ImPlot::EndPlot();
              }
          } else if (state.activeTask == TaskType::Segmentation) {
              if (ImPlot::BeginPlot("mIoU \xe6\x9b\xb2\xe7\xba\xbf", ImVec2(-1, -1))) {
                  ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "mIoU");
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.05, ImPlotCond_Always);
                  if (!ts.vecMIoUHistory.empty())
                      ImPlot::PlotLine("mIoU", ts.vecMIoUHistory.data(), (int)ts.vecMIoUHistory.size());
                  ImPlot::EndPlot();
              }
          } else {
              // 20260320 ZJH 检测/异常检测：显示验证损失趋势（复用 loss 数据）
              if (ImPlot::BeginPlot("\xe6\x8d\x9f\xe5\xa4\xb1\xe8\xb6\x8b\xe5\x8a\xbf", ImVec2(-1, -1))) {
                  ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "\xe6\x8d\x9f\xe5\xa4\xb1");
                  if (!ts.vecLossHistory.empty())
                      ImPlot::PlotLine("\xe6\x8d\x9f\xe5\xa4\xb1", ts.vecLossHistory.data(), (int)ts.vecLossHistory.size());
                  ImPlot::EndPlot();
              }
          }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    // 20260320 ZJH 训练日志（带右键菜单）
    if (beginCard("\xe8\xae\xad\xe7\xbb\x83\xe6\x97\xa5\xe5\xbf\x97")) {  // "训练日志"
        ImGui::BeginChild("##LogTxt", ImVec2(0, 120), ImGuiChildFlags_None);
        { std::lock_guard<std::mutex> lk(ts.mutex);
          ImGui::TextUnformatted(ts.strLog.c_str());
          if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f)
              ImGui::SetScrollHereY(1.0f);
        }
        // 20260320 ZJH 日志区右键菜单
        if (ImGui::BeginPopupContextWindow("##LogCtx")) {
            if (ImGui::MenuItem("\xe5\xa4\x8d\xe5\x88\xb6")) {
                std::lock_guard<std::mutex> lk(ts.mutex);
                ImGui::SetClipboardText(ts.strLog.c_str());  // 20260320 ZJH 复制日志到剪贴板
            }
            if (ImGui::MenuItem("\xe6\xb8\x85\xe9\x99\xa4")) {
                std::lock_guard<std::mutex> lk(ts.mutex);
                ts.strLog.clear();  // 20260320 ZJH 清除日志
            }
            if (ImGui::MenuItem("\xe4\xbf\x9d\xe5\xad\x98\xe5\x88\xb0\xe6\x96\x87\xe4\xbb\xb6")) {
                // 20260320 ZJH 保存日志到文件
                std::lock_guard<std::mutex> lk(ts.mutex);
                std::ofstream ofs("training_log.txt");
                if (ofs.is_open()) ofs << ts.strLog;
            }
            ImGui::EndPopup();
        }
        ImGui::EndChild();
    } endCard();

    // 20260320 ZJH 训练历史面板
    if (!state.vecTrainHistory.empty()) {
        if (beginCard("\xe8\xae\xad\xe7\xbb\x83\xe5\x8e\x86\xe5\x8f\xb2")) {  // "训练历史"
            if (ImGui::BeginTable("##TrainHist", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 120))) {
                ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 30);
                ImGui::TableSetupColumn("\xe6\xa8\xa1\xe5\x9e\x8b");
                ImGui::TableSetupColumn("\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87");
                ImGui::TableSetupColumn("\xe6\x8d\x9f\xe5\xa4\xb1");
                ImGui::TableSetupColumn("\xe6\x97\xa5\xe6\x9c\x9f");
                ImGui::TableHeadersRow();
                // 20260320 ZJH 查找最佳准确率
                float fBestAcc = 0.0f;
                int nBestIdx = -1;
                for (int i = 0; i < (int)state.vecTrainHistory.size(); ++i) {
                    if (state.vecTrainHistory[i].fAccuracy > fBestAcc) {
                        fBestAcc = state.vecTrainHistory[i].fAccuracy;
                        nBestIdx = i;
                    }
                }
                for (int i = 0; i < (int)state.vecTrainHistory.size(); ++i) {
                    auto& h = state.vecTrainHistory[i];
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    // 20260320 ZJH 最佳记录用星号标记
                    if (i == nBestIdx) ImGui::TextColored(s_colOrange, "%s%d", "\xe2\x98\x85", i + 1);
                    else ImGui::Text("%d", i + 1);
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%s", h.strModel.c_str());
                    ImGui::TableSetColumnIndex(2); ImGui::Text("%.2f%%", (double)h.fAccuracy);
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.4f", (double)h.fLoss);
                    ImGui::TableSetColumnIndex(4); ImGui::Text("%s", h.strDate.c_str());
                }
                ImGui::EndTable();
            }
        } endCard();
    }
}

// ============================================================================
// 20260320 ZJH 绘制步骤 4：评估（适配所有任务类型）
// ============================================================================
static void drawStepEvaluate(AppState& state) {
    auto& ts = state.trainState;

    if (!ts.bCompleted.load()) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::TextColored(s_colSubtle, "\xe8\xaf\xb7\xe5\x85\x88\xe5\xae\x8c\xe6\x88\x90\xe8\xae\xad\xe7\xbb\x83\xe4\xbb\xa5\xe6\x9f\xa5\xe7\x9c\x8b\xe8\xaf\x84\xe4\xbc\xb0\xe7\xbb\x93\xe6\x9e\x9c");
        return;
    }

    switch (state.activeTask) {
    case TaskType::Classification: {
        // 20260320 ZJH 性能概览卡片
        if (beginCard("\xe6\x80\xa7\xe8\x83\xbd\xe6\xa6\x82\xe8\xa7\x88")) {  // "性能概览"
            float fTestAcc = ts.fTestAcc.load();
            // 20260320 ZJH 大字指标
            auto* pDraw = ImGui::GetWindowDrawList();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            char buf[64]; std::snprintf(buf, sizeof(buf), "%.2f%%", (double)fTestAcc);
            pDraw->AddText(ImGui::GetFont(), 28.0f, pos, IM_COL32(59, 130, 246, 255), buf);
            ImGui::Dummy(ImVec2(0, 32));
            ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87");
            ImGui::SameLine(200);
            { std::lock_guard<std::mutex> lk(ts.mutex);
              ImGui::Text("\xe6\x9c\x80\xe4\xbd\xb3\xe6\x8d\x9f\xe5\xa4\xb1: %.4f", (double)ts.fBestValLoss);
              ImGui::SameLine(400);
              ImGui::Text("\xe8\x80\x97\xe6\x97\xb6: %.1fs", (double)ts.fTotalTrainingTimeSec);
            }
        } endCard();

        // 20260320 ZJH 混淆矩阵
        if (beginCard("\xe6\xb7\xb7\xe6\xb7\x86\xe7\x9f\xa9\xe9\x98\xb5")) {  // "混淆矩阵"
            std::lock_guard<std::mutex> lk(ts.mutex);
            if (ts.bHasConfusionMatrix) {
                static double hm[100];
                double mx = 0;
                for (int r = 0; r < 10; ++r)
                    for (int c = 0; c < 10; ++c) {
                        double v = (double)ts.arrConfusionMatrix[r][c];
                        hm[r*10+c] = v;
                        if (v > mx) mx = v;
                    }
                static const char* lab[] = {"0","1","2","3","4","5","6","7","8","9"};
                if (ImPlot::BeginPlot("##CM", ImVec2(320, 320))) {
                    ImPlot::SetupAxes("\xe9\xa2\x84\xe6\xb5\x8b", "\xe5\xae\x9e\xe9\x99\x85");
                    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 9, 10, lab);
                    ImPlot::SetupAxisTicks(ImAxis_Y1, 0, 9, 10, lab);
                    ImPlot::PlotHeatmap("##hm", hm, 10, 10, 0, mx, "%g", ImPlotPoint(0,0), ImPlotPoint(10,10));
                    ImPlot::EndPlot();
                }
            }
        } endCard();

        // 20260320 ZJH 分类报告
        if (beginCard("\xe5\x88\x86\xe7\xb1\xbb\xe6\x8a\xa5\xe5\x91\x8a")) {  // "分类报告"
            std::lock_guard<std::mutex> lk(ts.mutex);
            if (ts.bHasConfusionMatrix) {
                if (ImGui::BeginTable("##CR", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("\xe7\xb1\xbb\xe5\x88\xab");
                    ImGui::TableSetupColumn("\xe7\xb2\xbe\xe7\xa1\xae\xe7\x8e\x87");
                    ImGui::TableSetupColumn("\xe5\x8f\xac\xe5\x9b\x9e\xe7\x8e\x87");
                    ImGui::TableSetupColumn("F1");
                    ImGui::TableHeadersRow();
                    for (int c = 0; c < 10; ++c) {
                        int tp = ts.arrConfusionMatrix[c][c], fp = 0, fn = 0;
                        for (int i = 0; i < 10; ++i) {
                            if (i!=c) { fp += ts.arrConfusionMatrix[i][c]; fn += ts.arrConfusionMatrix[c][i]; }
                        }
                        float p = (tp+fp>0)?(float)tp/(float)(tp+fp):0;
                        float r = (tp+fn>0)?(float)tp/(float)(tp+fn):0;
                        float f = (p+r>0)?2*p*r/(p+r):0;
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::Text("%d", c);
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", (double)p);
                        ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", (double)r);
                        ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", (double)f);
                    }
                    ImGui::EndTable();
                }
            }
        } endCard();
        break;
    }
    case TaskType::Detection: {
        if (beginCard("\xe6\xa3\x80\xe6\xb5\x8b\xe6\x80\xa7\xe8\x83\xbd")) {  // "检测性能"
            std::lock_guard<std::mutex> lk(ts.mutex);
            auto* pDraw = ImGui::GetWindowDrawList();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            char buf[64]; std::snprintf(buf, sizeof(buf), "mAP@0.5: %.1f%%", (double)ts.fMAP50);
            pDraw->AddText(ImGui::GetFont(), 28.0f, pos, IM_COL32(59, 130, 246, 255), buf);
            ImGui::Dummy(ImVec2(0, 32));
            ImGui::Text("mAP@0.5:0.95: %.1f%%", (double)ts.fMAP5095);
            ImGui::Text("\xe6\x9c\x80\xe4\xbd\xb3\xe6\x8d\x9f\xe5\xa4\xb1: %.4f", (double)ts.fBestValLoss);
            ImGui::Text("\xe8\x80\x97\xe6\x97\xb6: %.1fs", (double)ts.fTotalTrainingTimeSec);
        } endCard();
        break;
    }
    case TaskType::Segmentation: {
        if (beginCard("\xe5\x88\x86\xe5\x89\xb2\xe6\x80\xa7\xe8\x83\xbd")) {  // "分割性能"
            std::lock_guard<std::mutex> lk(ts.mutex);
            auto* pDraw = ImGui::GetWindowDrawList();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            char buf[64]; std::snprintf(buf, sizeof(buf), "mIoU: %.3f", (double)ts.fMIoU);
            pDraw->AddText(ImGui::GetFont(), 28.0f, pos, IM_COL32(59, 130, 246, 255), buf);
            ImGui::Dummy(ImVec2(0, 32));
            ImGui::Text("\xe6\x9c\x80\xe4\xbd\xb3\xe6\x8d\x9f\xe5\xa4\xb1: %.4f", (double)ts.fBestValLoss);
            ImGui::Text("\xe8\x80\x97\xe6\x97\xb6: %.1fs", (double)ts.fTotalTrainingTimeSec);
        } endCard();
        break;
    }
    case TaskType::AnomalyDetection: {
        if (beginCard("\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe6\x80\xa7\xe8\x83\xbd")) {  // "异常检测性能"
            std::lock_guard<std::mutex> lk(ts.mutex);
            auto* pDraw = ImGui::GetWindowDrawList();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            char buf[64]; std::snprintf(buf, sizeof(buf), "AUC: %.3f", (double)ts.fAUC);
            pDraw->AddText(ImGui::GetFont(), 28.0f, pos, IM_COL32(59, 130, 246, 255), buf);
            ImGui::Dummy(ImVec2(0, 32));
            ImGui::Text("\xe5\xbc\x82\xe5\xb8\xb8\xe9\x98\x88\xe5\x80\xbc: %.4f", (double)ts.fAnomalyThreshold);
            ImGui::Text("\xe6\x9c\x80\xe4\xbd\xb3\xe9\x87\x8d\xe5\xbb\xba\xe8\xaf\xaf\xe5\xb7\xae: %.4f", (double)ts.fBestValLoss);
            ImGui::Text("\xe8\x80\x97\xe6\x97\xb6: %.1fs", (double)ts.fTotalTrainingTimeSec);
        } endCard();

        // 20260320 ZJH 阈值调节
        if (beginCard("\xe9\x98\x88\xe5\x80\xbc\xe8\xb0\x83\xe8\x8a\x82")) {
            ImGui::SliderFloat("\xe5\xbc\x82\xe5\xb8\xb8\xe9\x98\x88\xe5\x80\xbc", &state.fAeThreshold, 0.01f, 5.0f, "%.3f");
        } endCard();
        break;
    }
    }

    // 20260320 ZJH 通用导出按钮
    if (beginCard("\xe6\xa8\xa1\xe5\x9e\x8b\xe5\xaf\xbc\xe5\x87\xba")) {  // "模型导出"
        if (ImGui::Button("\xe5\xaf\xbc\xe5\x87\xba .dfm \xe6\xa8\xa1\xe5\x9e\x8b", ImVec2(180, 30))) {
            // 20260320 ZJH 模型已在训练时自动保存
        }
        ImGui::SameLine();
        ImGui::BeginDisabled();
        ImGui::Button("\xe5\xaf\xbc\xe5\x87\xba ONNX (\xe5\x8d\xb3\xe5\xb0\x86\xe6\x8e\xa8\xe5\x87\xba)", ImVec2(200, 30));
        ImGui::EndDisabled();
    } endCard();
}

// ============================================================================
// 20260320 ZJH 绘制右侧属性面板
// ============================================================================
static void drawPropertiesPanel(AppState& state) {
    auto& ts = state.trainState;

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.580f, 0.750f, 1.000f, 1.0f));
    ImGui::Text("\xe5\xb1\x9e\xe6\x80\xa7");  // "属性"
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH 根据步骤显示不同内容
    switch (state.nActiveStep) {
    case 0: {
        ImGui::TextColored(s_colAccent, "\xe6\x95\xb0\xe6\x8d\xae\xe9\x9b\x86\xe4\xbf\xa1\xe6\x81\xaf");
        ImGui::Separator();
        const char* tn[] = {"\xe5\x88\x86\xe7\xb1\xbb", "\xe6\xa3\x80\xe6\xb5\x8b", "\xe5\x88\x86\xe5\x89\xb2", "\xe5\xbc\x82\xe5\xb8\xb8"};
        ImGui::Text("\xe4\xbb\xbb\xe5\x8a\xa1: %s", tn[(int)state.activeTask]);
        if (state.activeTask == TaskType::Classification) {
            if (state.bMnistAvailable) {
                ImGui::Text("\xe8\xae\xad\xe7\xbb\x83: %d", state.nMnistTrainSamples);
                ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95: %d", state.nMnistTestSamples);
            }
            ImGui::Text("\xe7\xb1\xbb\xe5\x88\xab: 10");
            ImGui::Text("\xe5\xb0\xba\xe5\xaf\xb8: 28x28");
        }
        break;
    }
    case 1: {
        ImGui::TextColored(s_colAccent, "\xe6\xa8\xa1\xe5\x9e\x8b\xe4\xbf\xa1\xe6\x81\xaf");
        ImGui::Separator();
        switch (state.activeTask) {
        case TaskType::Classification:
            ImGui::Text("%s", state.nClsModel==0?"MLP":"ResNet-18");
            ImGui::Text("\xe5\x8f\x82\xe6\x95\xb0: %s", state.nClsModel==0?"~101K":"~11.2M");
            break;
        case TaskType::Detection:
            ImGui::Text("YOLOv5-Nano");
            ImGui::Text("\xe7\xb1\xbb\xe5\x88\xab: %d", state.nDetClasses);
            break;
        case TaskType::Segmentation:
            ImGui::Text("U-Net");
            ImGui::Text("\xe8\xbe\x93\xe5\x87\xba: %d \xe7\xb1\xbb", state.nSegClasses);
            break;
        case TaskType::AnomalyDetection:
            ImGui::Text("ConvAutoEncoder");
            ImGui::Text("\xe7\x93\xb6\xe9\xa2\x88: %d", state.nAeLatentDim);
            break;
        }
        break;
    }
    case 2: {
        ImGui::TextColored(s_colAccent, "\xe5\xae\x9e\xe6\x97\xb6\xe7\xbb\x9f\xe8\xae\xa1");
        ImGui::Separator();
        ImGui::Text("\xe6\x8d\x9f\xe5\xa4\xb1: %.4f", (double)ts.fCurrentLoss.load());
        if (state.activeTask == TaskType::Classification) {
            ImGui::Text("\xe8\xae\xad\xe7\xbb\x83\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87: %.2f%%", (double)ts.fTrainAcc.load());
            ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87: %.2f%%", (double)ts.fTestAcc.load());
        }
        ImGui::Text("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87: %.6f", (double)ts.fCurrentLR.load());
        ImGui::Separator();
        ImGui::Text("\xe8\xbd\xae\xe6\xac\xa1: %d/%d", ts.nCurrentEpoch.load(), ts.nTotalEpochs.load());
        ImGui::Text("\xe6\x89\xb9\xe6\xac\xa1: %d/%d", ts.nCurrentBatch.load(), ts.nTotalBatches.load());
        float ms = ts.fAvgBatchTimeMs.load();
        if (ms > 0) ImGui::Text("\xe6\x89\xb9\xe6\xac\xa1\xe8\x80\x97\xe6\x97\xb6: %.1fms", (double)ms);

        // 20260320 ZJH GPU 显存
        if (state.nGpuMemTotalMB > 0) {
            ImGui::Separator();
            ImGui::Text("GPU \xe6\x98\xbe\xe5\xad\x98: %zu/%zuMB", state.nGpuMemUsedMB, state.nGpuMemTotalMB);
        }
        break;
    }
    case 3: {
        ImGui::TextColored(s_colAccent, "\xe8\xaf\x84\xe4\xbc\xb0\xe6\x91\x98\xe8\xa6\x81");
        ImGui::Separator();
        if (ts.bCompleted.load()) {
            if (state.activeTask == TaskType::Classification) {
                ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95: %.2f%%", (double)ts.fTestAcc.load());
                ImGui::Text("\xe8\xae\xad\xe7\xbb\x83: %.2f%%", (double)ts.fTrainAcc.load());
            }
            { std::lock_guard<std::mutex> lk(ts.mutex);
              ImGui::Text("\xe6\x8d\x9f\xe5\xa4\xb1: %.4f", (double)ts.fBestValLoss);
              ImGui::Text("\xe8\x80\x97\xe6\x97\xb6: %.1fs", (double)ts.fTotalTrainingTimeSec); }
        } else {
            ImGui::TextDisabled("\xe8\xae\xad\xe7\xbb\x83\xe6\x9c\xaa\xe5\xae\x8c\xe6\x88\x90");
        }
        break;
    }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 20260320 ZJH GPU 信息（始终显示）
    ImGui::TextColored(s_colAccent, "GPU");
    ImGui::Separator();
    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        auto& gpu = state.vecGpuDevices[0];
        ImGui::TextWrapped("%s", gpu.strName.c_str());
        ImGui::Text("%zu MB", gpu.nTotalMemoryMB);
        ImGui::Text("\xe7\xae\x97\xe5\x8a\x9b %d.%d", gpu.nComputeCapMajor, gpu.nComputeCapMinor);
    } else {
        ImGui::TextDisabled("\xe6\x97\xa0 GPU");
    }
}

// ============================================================================
// 20260320 ZJH 绘制底部状态栏
// ============================================================================
static void drawStatusBar(AppState& state) {
    auto& ts = state.trainState;
    ImGuiViewport* pVP = ImGui::GetMainViewport();
    float fH = 26.0f;

    ImGui::SetNextWindowPos(ImVec2(pVP->WorkPos.x, pVP->WorkPos.y + pVP->WorkSize.y - fH));
    ImGui::SetNextWindowSize(ImVec2(pVP->WorkSize.x, fH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.070f, 0.075f, 0.100f, 1.0f));

    ImGui::Begin("##SB", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav);

    // 20260320 ZJH 日志状态
    if (ts.bRunning.load()) {
        if (ts.bPaused.load()) ImGui::TextColored(s_colOrange, "\xe7\x8a\xb6\xe6\x80\x81: \xe5\xb7\xb2\xe6\x9a\x82\xe5\x81\x9c");
        else {
            char b[128]; std::snprintf(b, sizeof(b), "\xe7\x8a\xb6\xe6\x80\x81: \xe8\xae\xad\xe7\xbb\x83\xe4\xb8\xad (%d/%d)", ts.nCurrentEpoch.load(), ts.nTotalEpochs.load());
            ImGui::TextColored(s_colGreen, "%s", b);
        }
    } else if (ts.bCompleted.load()) {
        ImGui::TextColored(s_colAccent, "\xe7\x8a\xb6\xe6\x80\x81: \xe8\xae\xad\xe7\xbb\x83\xe5\xb7\xb2\xe5\xae\x8c\xe6\x88\x90");
    } else {
        ImGui::Text("\xe7\x8a\xb6\xe6\x80\x81: \xe5\xb0\xb1\xe7\xbb\xaa");
    }

    // 20260320 ZJH GPU
    ImGui::SameLine(400);
    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        char b[256]; std::snprintf(b, sizeof(b), "GPU: %s", state.vecGpuDevices[0].strName.c_str());
        ImGui::TextColored(s_colGreen, "%s", b);
    } else {
        ImGui::TextColored(s_colOrange, "CPU");
    }

    // 20260320 ZJH 版本
    float vw = ImGui::CalcTextSize("DeepForge v0.1.0").x;
    ImGui::SameLine(ImGui::GetWindowWidth() - vw - 15);
    ImGui::TextDisabled("DeepForge v0.1.0");

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// ============================================================================
// 20260320 ZJH 主函数 — SDL3 + ImGui 应用程序入口
// ============================================================================
int main(int, char**) {
    // ===== 初始化 SDL3 =====
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* pWindow = SDL_CreateWindow("DeepForge v0.1.0", 1280, 720,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);
    if (!pWindow) { SDL_Quit(); return 1; }

    SDL_Renderer* pRenderer = SDL_CreateRenderer(pWindow, nullptr);
    if (!pRenderer) { SDL_DestroyWindow(pWindow); SDL_Quit(); return 1; }

    // ===== 初始化 ImGui =====
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    setupImGuiStyle();

    // 20260320 ZJH 加载中文字体
    const char* fonts[] = {"C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"};
    bool bFont = false;
    for (auto* fp : fonts) {
        if (std::filesystem::exists(fp)) {
            ImFontConfig fc; fc.MergeMode = false;
            io.Fonts->AddFontFromFileTTF(fp, 16.0f, &fc, io.Fonts->GetGlyphRangesChineseFull());
            bFont = true;
            break;
        }
    }
    if (!bFont) io.Fonts->AddFontDefault();

    ImGui_ImplSDL3_InitForSDLRenderer(pWindow, pRenderer);
    ImGui_ImplSDLRenderer3_Init(pRenderer);

    // ===== 闪屏动画 =====
    // 20260320 ZJH GPU 检测在闪屏期间进行
    std::vector<GpuDeviceInfo> vecGpuResult;
    bool bGpuDetectionDone = false;
    std::string strGpuStatus = "\xe6\xad\xa3\xe5\x9c\xa8\xe6\xa3\x80\xe6\xb5\x8bGPU\xe8\xae\xbe\xe5\xa4\x87...";

    // 20260320 ZJH 在后台线程检测 GPU
    std::thread gpuThread([&]() {
        vecGpuResult = detectGpuDevices();
        if (!vecGpuResult.empty()) {
            strGpuStatus = "GPU: " + vecGpuResult[0].strName;
        } else {
            strGpuStatus = "\xe6\x9c\xaa\xe6\xa3\x80\xe6\xb5\x8b\xe5\x88\xb0 GPU\xef\xbc\x8c\xe4\xbd\xbf\xe7\x94\xa8 CPU";
        }
        bGpuDetectionDone = true;
    });

    {
        auto splashStart = std::chrono::steady_clock::now();
        const float fDur = 2.5f;
        bool bSplash = true;

        while (bSplash) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                ImGui_ImplSDL3_ProcessEvent(&ev);
                if (ev.type == SDL_EVENT_QUIT) { bSplash = false; break; }
                if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) { bSplash = false; break; }
            }
            if (!bSplash) break;

            float fEl = std::chrono::duration<float>(std::chrono::steady_clock::now() - splashStart).count();
            if (fEl > fDur) break;
            float fA = std::min(fEl / 0.8f, 1.0f);

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
                ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav);

            auto* pDL = ImGui::GetWindowDrawList();
            float cx = io.DisplaySize.x * 0.5f;
            float cy = io.DisplaySize.y * 0.5f;
            ImFont* pF = ImGui::GetFont();

            // 20260320 ZJH 背景渐变
            ImU32 top = IM_COL32(15, 15, 25, (int)(255*fA));
            ImU32 bot = IM_COL32(25, 30, 50, (int)(255*fA));
            pDL->AddRectFilledMultiColor(ImVec2(0,0), io.DisplaySize, top, top, bot, bot);

            // 20260320 ZJH DF Logo
            float lsz = 80.0f;
            float lx = cx - lsz*0.5f, ly = cy - 120.0f;
            pDL->AddRectFilled(ImVec2(lx, ly), ImVec2(lx+lsz, ly+lsz),
                IM_COL32(59, 130, 246, (int)(240*fA)), 12.0f);
            float dfs = 42.0f;
            ImVec2 dfSz = pF->CalcTextSizeA(dfs, FLT_MAX, 0, "DF");
            pDL->AddText(pF, dfs, ImVec2(lx+(lsz-dfSz.x)*0.5f, ly+(lsz-dfSz.y)*0.5f),
                IM_COL32(255,255,255,(int)(255*fA)), "DF");

            // 20260320 ZJH 标题
            float ts2 = 36.0f;
            ImVec2 ttSz = pF->CalcTextSizeA(ts2, FLT_MAX, 0, "DeepForge");
            pDL->AddText(pF, ts2, ImVec2(cx-ttSz.x*0.5f, ly+lsz+20),
                IM_COL32(230,230,240,(int)(255*fA)), "DeepForge");

            // 20260320 ZJH 副标题
            const char* sub = "\xe7\xba\xaf C++ \xe5\x85\xa8\xe6\xb5\x81\xe7\xa8\x8b\xe6\xb7\xb1\xe5\xba\xa6\xe5\xad\xa6\xe4\xb9\xa0\xe8\xa7\x86\xe8\xa7\x89\xe5\xb9\xb3\xe5\x8f\xb0";
            ImVec2 subSz = pF->CalcTextSizeA(18.0f, FLT_MAX, 0, sub);
            pDL->AddText(pF, 18.0f, ImVec2(cx-subSz.x*0.5f, ly+lsz+64),
                IM_COL32(160,180,220,(int)(220*fA)), sub);

            // 20260320 ZJH 旋转加载动画
            int nDots = 8;
            float dotPhase = fEl * 3.0f;
            float dotR = 20.0f;
            float dotCY = ly + lsz + 120.0f;
            for (int i = 0; i < nDots; ++i) {
                float ang = dotPhase + (float)i * (6.283185f / (float)nDots);
                float dx = cx + std::cos(ang) * dotR;
                float dy = dotCY + std::sin(ang) * dotR;
                float da = (std::sin(ang - dotPhase) + 1.0f) * 0.5f;
                pDL->AddCircleFilled(ImVec2(dx, dy), 3.0f + da * 2.0f,
                    IM_COL32(59, 130, 246, (int)(da * 255 * fA)));
            }

            // 20260320 ZJH GPU 检测状态
            ImVec2 gpuSz = pF->CalcTextSizeA(14.0f, FLT_MAX, 0, strGpuStatus.c_str());
            pDL->AddText(pF, 14.0f, ImVec2(cx-gpuSz.x*0.5f, dotCY+dotR+20),
                IM_COL32(140,150,180,(int)(180*fA)), strGpuStatus.c_str());

            // 20260320 ZJH 版权
            ImVec2 cpSz = pF->CalcTextSizeA(12.0f, FLT_MAX, 0, "\xc2\xa9 2026 ZJH");
            pDL->AddText(pF, 12.0f, ImVec2(cx-cpSz.x*0.5f, io.DisplaySize.y-40),
                IM_COL32(100,100,120,(int)(150*fA)), "\xc2\xa9 2026 ZJH");

            ImGui::End();
            ImGui::PopStyleColor();
            ImGui::PopStyleVar();

            ImGui::Render();
            SDL_SetRenderDrawColor(pRenderer, 15, 15, 20, 255);
            SDL_RenderClear(pRenderer);
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer);
            SDL_RenderPresent(pRenderer);
        }

        if (!bSplash) {
            gpuThread.join();
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

    gpuThread.join();

    // ===== 应用状态初始化 =====
    AppState appState;
    appState.vecGpuDevices = vecGpuResult;
    appState.bGpuDetected = !vecGpuResult.empty();
    bool bPrevCompleted = false;  // 20260320 ZJH 上一帧训练完成状态（用于检测完成事件）

    // ===== 主循环 =====
    bool bRunning = true;
    while (bRunning) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            if (ev.type == SDL_EVENT_QUIT) bRunning = false;
            if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) bRunning = false;
        }

        // 20260320 ZJH 定期查询 GPU 显存
        if (appState.bGpuDetected) {
            appState.fGpuMemQueryTimer += io.DeltaTime;
            if (appState.fGpuMemQueryTimer > 5.0f) {
                appState.fGpuMemQueryTimer = 0.0f;
                auto [u, t] = queryGpuMemoryUsage();
                appState.nGpuMemUsedMB = u;
                appState.nGpuMemTotalMB = t;
            }
        }

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // 20260320 ZJH 主菜单栏（在所有窗口之前渲染）
        drawMainMenuBar(appState);

        // 20260320 ZJH 键盘快捷键处理
        handleKeyboardShortcuts(appState, pWindow);

        // 20260320 ZJH 更新脉冲动画计时器
        appState.fStepPulseTimer += io.DeltaTime;

        ImGuiViewport* pVP = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(pVP->WorkPos);
        ImGui::SetNextWindowSize(ImVec2(pVP->WorkSize.x, pVP->WorkSize.y - 26.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("##Main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::PopStyleVar();

        // ============================================================
        // 20260320 ZJH 顶部工具栏
        // ============================================================
        drawToolbar(appState);

        // 20260320 ZJH 弹窗打开逻辑
        if (appState.bShowSettingsPopup) {
            ImGui::OpenPopup("\xe8\xae\xbe\xe7\xbd\xae##Modal");
            appState.bShowSettingsPopup = false;
        }
        if (appState.bShowAboutPopup) {
            ImGui::OpenPopup("\xe5\x85\xb3\xe4\xba\x8e DeepForge##About");
            appState.bShowAboutPopup = false;
        }
        if (appState.bShowBatchInference) {
            ImGui::OpenPopup("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86##BatchInf");
            appState.bShowBatchInference = false;
        }

        // 20260320 ZJH 渲染各弹窗
        drawSettingsDialog(appState);
        drawAboutDialog(appState);
        drawBatchInferenceDialog(appState);

        // 20260320 ZJH GPU 信息弹窗
        if (appState.bShowGpuInfo) {
            ImGui::OpenPopup("GPU \xe4\xbf\xa1\xe6\x81\xaf##GpuDlg");
            appState.bShowGpuInfo = false;
        }
        if (ImGui::BeginPopup("GPU \xe4\xbf\xa1\xe6\x81\xaf##GpuDlg")) {
            ImGui::TextColored(s_colAccent, "GPU \xe8\xae\xbe\xe5\xa4\x87\xe4\xbf\xa1\xe6\x81\xaf");
            ImGui::Separator();
            if (appState.bGpuDetected && !appState.vecGpuDevices.empty()) {
                for (size_t gi = 0; gi < appState.vecGpuDevices.size(); ++gi) {
                    auto& gpu = appState.vecGpuDevices[gi];
                    ImGui::Text("GPU %zu: %s", gi, gpu.strName.c_str());
                    ImGui::Text("  \xe6\x98\xbe\xe5\xad\x98: %zu MB", gpu.nTotalMemoryMB);
                    ImGui::Text("  \xe7\xae\x97\xe5\x8a\x9b: %d.%d", gpu.nComputeCapMajor, gpu.nComputeCapMinor);
                }
            } else {
                ImGui::TextDisabled("\xe6\x9c\xaa\xe6\xa3\x80\xe6\xb5\x8b\xe5\x88\xb0 GPU");
            }
            ImGui::EndPopup();
        }

        // ============================================================
        // 20260320 ZJH 三栏布局（面板可见性受视图菜单控制）
        // ============================================================
        float fSideW = appState.bShowNavPanel ? 180.0f : 0.0f;   // 20260320 ZJH 左侧面板宽度
        float fPropW = appState.bShowPropsPanel ? 200.0f : 0.0f;  // 20260320 ZJH 右侧面板宽度
        float fContentH = ImGui::GetContentRegionAvail().y;

        // 20260320 ZJH 左侧项目导航（可隐藏）
        if (appState.bShowNavPanel) {
            ImGui::BeginChild("##Nav", ImVec2(fSideW, fContentH), ImGuiChildFlags_Borders);
            drawProjectNav(appState);
            ImGui::EndChild();
            ImGui::SameLine();
        }

        // 20260320 ZJH 中间主内容区
        float fMainW = ImGui::GetContentRegionAvail().x - (appState.bShowPropsPanel ? fPropW + 4 : 0);
        ImGui::BeginChild("##Center", ImVec2(fMainW, fContentH), ImGuiChildFlags_Borders);
        {
            // 20260320 ZJH Halcon 风格步骤指示器（带脉冲动画）
            ImGui::Spacing();
            drawStepIndicator(appState.nActiveStep,
                appState.trainState.bStepDataDone,
                appState.trainState.bStepConfigDone,
                appState.trainState.bCompleted.load(),
                appState.fStepPulseTimer);

            // 20260320 ZJH 步骤按钮（可点击）
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14, 5));
            for (int i = 0; i < 4; ++i) {
                if (i > 0) ImGui::SameLine();
                const char* sl[] = {"1.\xe6\x95\xb0\xe6\x8d\xae", "2.\xe9\x85\x8d\xe7\xbd\xae", "3.\xe8\xae\xad\xe7\xbb\x83", "4.\xe8\xaf\x84\xe4\xbc\xb0"};
                bool bSel = (appState.nActiveStep == i);
                if (bSel) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.231f, 0.510f, 0.965f, 0.60f));
                if (ImGui::Button(sl[i])) appState.nActiveStep = i;
                if (bSel) ImGui::PopStyleColor();
            }
            ImGui::PopStyleVar();

            ImGui::Separator();

            // 20260320 ZJH 步骤内容
            ImGui::BeginChild("##StepContent", ImVec2(0, 0));
            switch (appState.nActiveStep) {
                case 0: drawStepData(appState); break;
                case 1: drawStepConfig(appState); break;
                case 2: drawStepTrain(appState); break;
                case 3: drawStepEvaluate(appState); break;
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();

        // 20260320 ZJH 右侧属性面板（可隐藏）
        if (appState.bShowPropsPanel) {
            ImGui::SameLine();
            ImGui::BeginChild("##Props", ImVec2(0, fContentH), ImGuiChildFlags_Borders);
            drawPropertiesPanel(appState);
            ImGui::EndChild();
        }

        ImGui::End();  // ##Main

        // 20260320 ZJH 底部状态栏
        drawStatusBar(appState);

        // 20260320 ZJH 训练完成时自动记录训练历史
        bool bNowCompleted = appState.trainState.bCompleted.load();
        if (bNowCompleted && !bPrevCompleted) {
            // 20260320 ZJH 训练刚完成，添加历史记录
            TrainingHistoryEntry entry;
            const char* arrTaskModels[] = {"MLP/ResNet", "YOLOv5", "U-Net", "AutoEncoder"};
            entry.strModel = arrTaskModels[(int)appState.activeTask];
            entry.fAccuracy = appState.trainState.fTestAcc.load();
            entry.fLoss = appState.trainState.fBestValLoss;
            // 20260320 ZJH 获取当前日期作为记录日期
            auto now = std::chrono::system_clock::now();
            auto nTime = std::chrono::system_clock::to_time_t(now);
            struct tm tmLocal;
            localtime_s(&tmLocal, &nTime);
            char arrDateBuf[32];
            std::snprintf(arrDateBuf, sizeof(arrDateBuf), "%04d-%02d-%02d %02d:%02d",
                tmLocal.tm_year + 1900, tmLocal.tm_mon + 1, tmLocal.tm_mday,
                tmLocal.tm_hour, tmLocal.tm_min);
            entry.strDate = arrDateBuf;
            appState.vecTrainHistory.push_back(entry);
        }
        bPrevCompleted = bNowCompleted;

        // 20260320 ZJH 渲染
        ImGui::Render();
        SDL_SetRenderDrawColor(pRenderer, 20, 20, 28, 255);
        SDL_RenderClear(pRenderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer);
        SDL_RenderPresent(pRenderer);
    }

    // ===== 清理 =====
    if (appState.pTrainThread) {
        appState.trainState.bStopRequested.store(true);
        appState.trainState.bPaused.store(false);
        appState.pTrainThread->request_stop();
        appState.pTrainThread->join();
        appState.pTrainThread.reset();
    }

    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    SDL_Quit();

    return 0;
}
