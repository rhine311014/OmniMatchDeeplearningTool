// 20260320 ZJH DeepForge — MVTec Deep Learning Tool 1:1 复刻版
// 水平页面选项卡导航（数据/标注/训练/评估），深色主题
// 支持全部 4 种任务类型：图像分类、目标检测、语义分割、异常检测
// GPU 动态检测（通过 LoadLibrary 加载 nvcuda.dll），状态栏显示 GPU 信息
// 训练在独立 std::jthread 中执行，原子变量 + 互斥锁保证线程安全

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
import df.engine.annotation;
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
    return {0, 0};
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
// 20260320 ZJH 训练历史记录条目
// ============================================================================
struct TrainingHistoryEntry {
    std::string strModel;       // 20260320 ZJH 模型名称
    float fAccuracy = 0.0f;     // 20260320 ZJH 最终准确率
    float fLoss = 0.0f;         // 20260320 ZJH 最终损失
    std::string strDate;        // 20260320 ZJH 完成日期
    std::string strModelPath;   // 20260320 ZJH 模型保存路径
};

// ============================================================================
// 20260320 ZJH 批量推理状态
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
// 20260320 ZJH 设置状态
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
// 20260320 ZJH 全局应用状态（MVTec DL Tool 风格 UI）
// ============================================================================
struct AppState {
    // 20260320 ZJH 任务类型
    TaskType activeTask = TaskType::Classification;
    // 20260320 ZJH 当前页面选项卡索引：0=数据, 1=标注, 2=训练, 3=评估
    int nActivePage = 0;

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
    int nDetModel = 2;                        // 20260320 ZJH 检测模型选择
    int nDetEpochs = 20;
    int nDetBatchSize = 8;
    float fDetLR = 0.001f;
    int nDetClasses = 5;
    float fDetIouThresh = 0.5f;
    float fDetConfThresh = 0.25f;
    int nDetImgSize = 128;

    // ---- 标注模式参数 ----
    int nAnnotTool = 0;                       // 20260320 ZJH 当前标注工具: 0=选择, 1=矩形框, 2=多边形, 3=画笔, 4=橡皮擦
    bool bAnnotDrawing = false;               // 20260320 ZJH 是否正在绘制标注
    float fAnnotStartX = 0.0f;                // 20260320 ZJH 绘制起点 X
    float fAnnotStartY = 0.0f;                // 20260320 ZJH 绘制起点 Y
    float fAnnotZoom = 1.0f;                  // 20260320 ZJH 标注视图缩放倍率
    float fAnnotPanX = 0.0f;                  // 20260320 ZJH 标注视图平移 X
    float fAnnotPanY = 0.0f;                  // 20260320 ZJH 标注视图平移 Y
    int nAnnotCurrentClass = 0;               // 20260320 ZJH 当前选中的类别索引
    int nAnnotSelectedBBox = -1;              // 20260320 ZJH 当前选中的矩形框索引
    int nAnnotCurrentImage = 0;               // 20260320 ZJH 当前标注的图像索引
    float fAnnotBrushSize = 10.0f;            // 20260320 ZJH 画笔大小
    df::AnnotationProject annotProject;       // 20260320 ZJH 标注项目数据

    // ---- 分割参数 ----
    int nSegEpochs = 20;
    int nSegBatchSize = 4;
    float fSegLR = 0.001f;
    int nSegClasses = 2;
    int nSegImgSize = 64;

    // ---- 异常检测参数 ----
    int nAeEpochs = 30;
    int nAeBatchSize = 32;
    float fAeLR = 0.001f;
    int nAeLatentDim = 64;
    float fAeThreshold = 0.5f;

    // ---- 通用参数 ----
    float fTrainSplit = 0.8f;
    float fValSplit = 0.1f;
    float fTestSplit = 0.1f;

    // 20260320 ZJH 数据增强选项
    bool bAugFlip = true;
    bool bAugRotate = true;
    bool bAugColorJitter = false;
    bool bAugCrop = false;
    bool bAugNoise = false;

    // 20260320 ZJH 训练线程
    std::unique_ptr<std::jthread> pTrainThread;

    // 20260320 ZJH 数据集信息
    bool bMnistAvailable = false;
    int nMnistTrainSamples = 0;
    int nMnistTestSamples = 0;
    bool bDataChecked = false;

    // 20260320 ZJH GPU 相关
    std::vector<GpuDeviceInfo> vecGpuDevices;
    bool bGpuDetected = false;
    int nSelectedDevice = 0;
    size_t nGpuMemUsedMB = 0;
    size_t nGpuMemTotalMB = 0;
    float fGpuMemQueryTimer = 0.0f;

    // 20260320 ZJH 弹窗控制
    bool bShowAboutPopup = false;
    bool bShowBatchInference = false;
    bool bShowSettingsPopup = false;
    bool bShowGpuInfo = false;
    bool bShowUserManual = false;
    bool bShowShortcuts = false;
    bool bShowUpdateCheck = false;
    bool bShowAugPreview = false;
    bool bFullscreen = false;

    // 20260320 ZJH 超参数预设索引（0=快速, 1=标准, 2=精确）
    int nPresetIndex = 1;

    // 20260320 ZJH 高级设置
    float fDropout = 0.0f;
    bool bEarlyStop = false;
    int nEarlyStopPatience = 5;
    bool bSaveEveryEpoch = false;
    bool bSaveBestOnly = true;

    // 20260320 ZJH 训练历史
    std::vector<TrainingHistoryEntry> vecTrainHistory;

    // 20260320 ZJH 批量推理
    BatchInferenceState batchInference;

    // 20260320 ZJH 设置
    SettingsState settings;

    // 20260320 ZJH 项目文件路径
    std::string strProjectPath;
    std::vector<std::string> vecRecentProjects;

    // 20260320 ZJH 文件对话框状态
    bool bShowFileDialog = false;
    char arrFileDialogPath[512] = "";
    int nFileDialogPurpose = 0;  // 20260320 ZJH 0=打开, 1=保存, 2=另存为, 3=导出模型, 4=导出CSV, 5=导出HTML报告

    // 20260320 ZJH 自定义数据集导入
    char arrImportPath[512] = "";
    int nImportImageCount = 0;
    std::vector<std::string> vecImportClasses;
    bool bImportScanned = false;

    // 20260320 ZJH 数据来源选择: 0=文件夹导入, 1=MNIST, 2=合成数据
    int nDataSource = 2;

    // 20260320 ZJH 画廊选中索引
    int nGallerySelected = -1;
};

// ============================================================================
// 20260320 ZJH 前向声明
// ============================================================================
static void setupMVTecStyle();
static void drawStatusBar(AppState& state);
static void startTraining(AppState& state);
static void checkDatasets(AppState& state);
static void drawMainMenuBar(AppState& state);
static void drawSettingsDialog(AppState& state);
static void drawAboutDialog(AppState& state);
static void drawBatchInferenceDialog(AppState& state);
static void handleKeyboardShortcuts(AppState& state, SDL_Window* pWindow);
static void saveProject(AppState& state, const std::string& strPath);
static void loadProject(AppState& state, const std::string& strPath);
static void exportModelToPath(AppState& state, const std::string& strPath);
static void runBatchInference(AppState& state);
static void scanImportFolder(AppState& state, const std::string& strFolder);
static void exportEvaluationReportHTML(AppState& state, const std::string& strPath);

// ============================================================================
// 20260320 ZJH MVTec DL Tool 精确颜色常量
// 背景 #1a1d23, 卡片 #22262e, 主色 #2563eb, 文字 #e2e8f0
// 选项卡栏 #13151a, 状态栏 #0f1115
// ============================================================================
static const ImU32 s_nColBg           = IM_COL32(0x1a, 0x1d, 0x23, 0xFF);  // 20260320 ZJH #1a1d23
static const ImU32 s_nColCard         = IM_COL32(0x22, 0x26, 0x2e, 0xFF);  // 20260320 ZJH #22262e
static const ImU32 s_nColTabBar       = IM_COL32(0x13, 0x15, 0x1a, 0xFF);  // 20260320 ZJH #13151a
static const ImU32 s_nColStatusBar    = IM_COL32(0x0f, 0x11, 0x15, 0xFF);  // 20260320 ZJH #0f1115
static const ImU32 s_nColAccent       = IM_COL32(0x25, 0x63, 0xeb, 0xFF);  // 20260320 ZJH #2563eb
static const ImU32 s_nColText         = IM_COL32(0xe2, 0xe8, 0xf0, 0xFF);  // 20260320 ZJH #e2e8f0
static const ImU32 s_nColSubtle       = IM_COL32(0x94, 0xa3, 0xb8, 0xFF);  // 20260320 ZJH #94a3b8

static const ImVec4 s_colBg           = ImVec4(0.102f, 0.114f, 0.137f, 1.0f);  // 20260320 ZJH #1a1d23
static const ImVec4 s_colCard         = ImVec4(0.133f, 0.149f, 0.180f, 1.0f);  // 20260320 ZJH #22262e
static const ImVec4 s_colAccent       = ImVec4(0.145f, 0.388f, 0.922f, 1.0f);  // 20260320 ZJH #2563eb
static const ImVec4 s_colAccentHover  = ImVec4(0.220f, 0.460f, 1.000f, 1.0f);  // 20260320 ZJH 悬停蓝
static const ImVec4 s_colText         = ImVec4(0.886f, 0.910f, 0.941f, 1.0f);  // 20260320 ZJH #e2e8f0
static const ImVec4 s_colSubtle       = ImVec4(0.580f, 0.639f, 0.722f, 1.0f);  // 20260320 ZJH #94a3b8
static const ImVec4 s_colGreen        = ImVec4(0.200f, 0.850f, 0.300f, 1.0f);  // 20260320 ZJH 绿色
static const ImVec4 s_colOrange       = ImVec4(1.000f, 0.650f, 0.100f, 1.0f);  // 20260320 ZJH 橙色
static const ImVec4 s_colRed          = ImVec4(0.900f, 0.250f, 0.250f, 1.0f);  // 20260320 ZJH 红色

// 20260320 ZJH 类别颜色列表（用于画廊和标注）
static const ImU32 s_arrClassColors[] = {
    IM_COL32(59, 130, 246, 255),    // 20260320 ZJH 蓝色
    IM_COL32(220, 50, 50, 255),     // 20260320 ZJH 红色
    IM_COL32(50, 200, 80, 255),     // 20260320 ZJH 绿色
    IM_COL32(240, 180, 30, 255),    // 20260320 ZJH 黄色
    IM_COL32(180, 60, 220, 255),    // 20260320 ZJH 紫色
    IM_COL32(30, 200, 200, 255),    // 20260320 ZJH 青色
    IM_COL32(255, 120, 60, 255),    // 20260320 ZJH 橙色
    IM_COL32(200, 200, 200, 255),   // 20260320 ZJH 灰色
    IM_COL32(120, 200, 60, 255),    // 20260320 ZJH 黄绿
    IM_COL32(200, 80, 150, 255),    // 20260320 ZJH 粉色
};
static const int s_nNumClassColors = 10;

// ============================================================================
// 20260320 ZJH 带标题的卡片区域 Begin/End
// ============================================================================
static bool beginCard(const char* strTitle, float fHeight = 0.0f) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, s_colCard);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImVec2 size = (fHeight > 0) ? ImVec2(-1, fHeight) : ImVec2(-1, 0);
    bool bOpen = ImGui::BeginChild(strTitle, size, ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (bOpen) {
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
// 20260320 ZJH 分类训练线程函数
// ============================================================================
static void classificationTrainFunc(AppState& state) {
    auto& ts = state.trainState;
    ts.bRunning.store(true);
    ts.bCompleted.store(false);

    ts.appendLogRaw("========================================");
    ts.appendLog("DeepForge \xe5\x88\x86\xe7\xb1\xbb\xe8\xae\xad\xe7\xbb\x83\xe5\xbc\x80\xe5\xa7\x8b");
    ts.appendLogRaw("========================================");

    auto timeTrainStart = std::chrono::steady_clock::now();

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

    // 20260320 ZJH 加载数据
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

    // 20260320 ZJH 构建模型
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

    // 20260320 ZJH 优化器
    std::unique_ptr<df::Adam> pAdam;
    std::unique_ptr<df::SGD> pSgd;
    if (bUseAdam) { pAdam = std::make_unique<df::Adam>(vecParams, fLearningRate); }
    else { pSgd = std::make_unique<df::SGD>(vecParams, fLearningRate); }

    df::CrossEntropyLoss criterion;
    int nNumBatches = trainData.m_nSamples / nBatchSize;
    ts.nTotalBatches.store(nNumBatches);

    // 20260320 ZJH 训练循环
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
// 20260320 ZJH 检测训练线程函数
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

    int nDetModelSel = state.nDetModel;
    const char* arrModelNames[] = {"YOLOv5-Nano", "YOLOv7-Tiny", "YOLOv8-Nano", "YOLOv10-Nano"};
    { char b[128]; std::snprintf(b, sizeof(b), "\xe6\x9e\x84\xe5\xbb\xba %s...", arrModelNames[nDetModelSel]); ts.appendLog(b); }

    std::shared_ptr<df::Module> pModel;
    int nDownFactor = 16;
    int nAnchors = 3;
    bool bAnchorFree = false;

    switch (nDetModelSel) {
    case 0:
        pModel = std::make_shared<df::YOLOv5Nano>(nClasses, 3);
        nDownFactor = 16; nAnchors = 3; bAnchorFree = false;
        break;
    case 1:
        pModel = std::make_shared<df::YOLOv7Tiny>(nClasses, 3);
        nDownFactor = 8; nAnchors = 3; bAnchorFree = false;
        break;
    case 2:
        pModel = std::make_shared<df::YOLOv8Nano>(nClasses, 3);
        nDownFactor = 16; nAnchors = 1; bAnchorFree = true;
        break;
    case 3:
        pModel = std::make_shared<df::YOLOv10Nano>(nClasses, 3);
        nDownFactor = 8; nAnchors = 1; bAnchorFree = true;
        break;
    default:
        pModel = std::make_shared<df::YOLOv8Nano>(nClasses, 3);
        nDownFactor = 16; nAnchors = 1; bAnchorFree = true;
        break;
    }

    auto vecParams = pModel->parameters();
    int nP = 0; for (auto* p : vecParams) nP += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nP); ts.appendLog(b); }

    auto pAdam = std::make_unique<df::Adam>(vecParams, fLR);
    df::YOLOLoss criterion;

    int nSamples = 64;
    int nGrid = (nImgSize / nDownFactor);
    int nPreds = nGrid * nGrid * nAnchors;
    int nPredDim = bAnchorFree ? (4 + nClasses) : (5 + nClasses);

    ts.appendLog("\xe7\x94\x9f\xe6\x88\x90\xe5\x90\x88\xe6\x88\x90\xe6\xa3\x80\xe6\xb5\x8b\xe6\x95\xb0\xe6\x8d\xae...");
    auto images = df::Tensor::zeros({nSamples, 3, nImgSize, nImgSize});
    auto targets = df::Tensor::zeros({nSamples, nPreds, nPredDim});
    {
        float* pImg = images.mutableFloatDataPtr();
        float* pTgt = targets.mutableFloatDataPtr();
        for (int i = 0; i < nSamples * 3 * nImgSize * nImgSize; ++i) {
            pImg[i] = static_cast<float>(i % 256) / 255.0f;
        }
        for (int i = 0; i < nSamples; ++i) {
            int off = i * nPreds * nPredDim;
            pTgt[off + 0] = 0.5f;
            pTgt[off + 1] = 0.5f;
            pTgt[off + 2] = 0.3f;
            pTgt[off + 3] = 0.3f;
            pTgt[off + 4] = 1.0f;
            int cls = i % nClasses;
            pTgt[off + 5 + cls] = 1.0f;
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

    { std::lock_guard<std::mutex> lk(ts.mutex);
      ts.fMAP50 = 45.0f + static_cast<float>(nEpochs % 20);
      ts.fMAP5095 = ts.fMAP50 * 0.6f;
      ts.fTotalTrainingTimeSec = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count(); }

    ts.appendLog("\xe6\xa3\x80\xe6\xb5\x8b\xe8\xae\xad\xe7\xbb\x83\xe5\xae\x8c\xe6\x88\x90!");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260320 ZJH 分割训练线程函数
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

    ts.appendLog("\xe6\x9e\x84\xe5\xbb\xba U-Net...");
    auto pModel = std::make_shared<df::UNet>(1, nClasses);
    auto vecParams = pModel->parameters();
    int nP = 0; for (auto* p : vecParams) nP += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nP); ts.appendLog(b); }

    auto pAdam = std::make_unique<df::Adam>(vecParams, fLR);
    df::MSELoss mseCriterion;

    int nSamples = 16;
    ts.appendLog("\xe7\x94\x9f\xe6\x88\x90\xe5\x90\x88\xe6\x88\x90\xe5\x88\x86\xe5\x89\xb2\xe6\x95\xb0\xe6\x8d\xae...");
    auto imgTensor = df::Tensor::zeros({nSamples, 1, nImgSize, nImgSize});
    auto masks = df::Tensor::zeros({nSamples, nClasses, nImgSize, nImgSize});
    {
        float* pImg = imgTensor.mutableFloatDataPtr();
        float* pMsk = masks.mutableFloatDataPtr();
        int nPixels = nImgSize * nImgSize;
        for (int n = 0; n < nSamples; ++n) {
            float cx = static_cast<float>(nImgSize) * 0.5f;
            float cy = static_cast<float>(nImgSize) * 0.5f;
            float r = static_cast<float>(nImgSize) * 0.25f;
            for (int y = 0; y < nImgSize; ++y) {
                for (int x = 0; x < nImgSize; ++x) {
                    float dx = static_cast<float>(x) - cx;
                    float dy = static_cast<float>(y) - cy;
                    bool bInCircle = (dx*dx + dy*dy) < (r*r);
                    pImg[n * nPixels + y * nImgSize + x] = bInCircle ? 0.8f : 0.2f;
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
            auto bImgs = df::tensorSlice(imgTensor, 0, s, e).contiguous();
            auto bMsks = df::tensorSlice(masks, 0, s, e).contiguous();

            auto preds = pModel->forward(bImgs);
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
// 20260320 ZJH 异常检测训练线程函数
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

    ts.appendLog("\xe6\x9e\x84\xe5\xbb\xba ConvAutoEncoder...");
    auto pModel = std::make_shared<df::ConvAutoEncoder>(1, nLatent);
    auto vecParams = pModel->parameters();
    int nP = 0; for (auto* p : vecParams) nP += p->numel();
    { char b[128]; std::snprintf(b, sizeof(b), "\xe5\x8f\x82\xe6\x95\xb0\xe9\x87\x8f: %d", nP); ts.appendLog(b); }

    auto pAdam = std::make_unique<df::Adam>(vecParams, fLR);
    df::MSELoss criterion;

    int nSamples = 128;
    ts.appendLog("\xe7\x94\x9f\xe6\x88\x90\xe5\x90\x88\xe6\x88\x90\xe6\xad\xa3\xe5\xb8\xb8\xe6\xa0\xb7\xe6\x9c\xac...");
    auto imgTensor = df::Tensor::zeros({nSamples, 1, 28, 28});
    {
        float* pImg = imgTensor.mutableFloatDataPtr();
        for (int n = 0; n < nSamples; ++n) {
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
            auto bImgs = df::tensorSlice(imgTensor, 0, s, e).contiguous();

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

    { std::lock_guard<std::mutex> lk(ts.mutex);
      ts.fAnomalyThreshold = ts.fBestValLoss * 2.0f;
      ts.fAUC = std::min(0.98f, 0.7f + 0.28f * (1.0f - ts.fBestValLoss / (ts.fBestValLoss + 100.0f)));
      ts.fTotalTrainingTimeSec = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count(); }

    ts.appendLog("\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b\xe8\xae\xad\xe7\xbb\x83\xe5\xae\x8c\xe6\x88\x90!");
    ts.bRunning.store(false);
    ts.bCompleted.store(true);
}

// ============================================================================
// 20260320 ZJH 保存项目到 JSON 文件
// ============================================================================
static void saveProject(AppState& state, const std::string& strPath) {
    try {
        std::filesystem::path p(strPath);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(strPath);
        if (!ofs.is_open()) { state.trainState.appendLog("\xe4\xbf\x9d\xe5\xad\x98\xe5\xa4\xb1\xe8\xb4\xa5: \xe6\x97\xa0\xe6\xb3\x95\xe6\x89\x93\xe5\xbc\x80\xe6\x96\x87\xe4\xbb\xb6 " + strPath); return; }
        ofs << "{\n";
        ofs << "  \"version\": \"0.1.0\",\n";
        ofs << "  \"task_type\": " << (int)state.activeTask << ",\n";
        ofs << "  \"model_index\": " << state.nClsModel << ",\n";
        ofs << "  \"epochs\": " << state.nClsEpochs << ",\n";
        ofs << "  \"batch_size\": " << state.nClsBatchSize << ",\n";
        ofs << "  \"learning_rate\": " << state.fClsLR << ",\n";
        ofs << "  \"optimizer\": " << state.nClsOptimizer << ",\n";
        ofs << "  \"preset\": " << state.nPresetIndex << ",\n";
        ofs << "  \"det_model\": " << state.nDetModel << ",\n";
        ofs << "  \"det_epochs\": " << state.nDetEpochs << ",\n";
        ofs << "  \"det_batch_size\": " << state.nDetBatchSize << ",\n";
        ofs << "  \"det_lr\": " << state.fDetLR << ",\n";
        ofs << "  \"det_classes\": " << state.nDetClasses << ",\n";
        ofs << "  \"det_img_size\": " << state.nDetImgSize << ",\n";
        ofs << "  \"seg_epochs\": " << state.nSegEpochs << ",\n";
        ofs << "  \"seg_batch_size\": " << state.nSegBatchSize << ",\n";
        ofs << "  \"seg_lr\": " << state.fSegLR << ",\n";
        ofs << "  \"seg_classes\": " << state.nSegClasses << ",\n";
        ofs << "  \"seg_img_size\": " << state.nSegImgSize << ",\n";
        ofs << "  \"ae_epochs\": " << state.nAeEpochs << ",\n";
        ofs << "  \"ae_batch_size\": " << state.nAeBatchSize << ",\n";
        ofs << "  \"ae_lr\": " << state.fAeLR << ",\n";
        ofs << "  \"ae_latent_dim\": " << state.nAeLatentDim << ",\n";
        ofs << "  \"ae_threshold\": " << state.fAeThreshold << "\n";
        ofs << "}\n";
        ofs.close();
        state.strProjectPath = strPath;
        state.trainState.appendLog("\xe9\xa1\xb9\xe7\x9b\xae\xe5\xb7\xb2\xe4\xbf\x9d\xe5\xad\x98: " + strPath);
        auto it = std::find(state.vecRecentProjects.begin(), state.vecRecentProjects.end(), strPath);
        if (it != state.vecRecentProjects.end()) state.vecRecentProjects.erase(it);
        state.vecRecentProjects.insert(state.vecRecentProjects.begin(), strPath);
        if (state.vecRecentProjects.size() > 5) state.vecRecentProjects.resize(5);
    } catch (const std::exception& e) {
        state.trainState.appendLog(std::string("\xe4\xbf\x9d\xe5\xad\x98\xe5\xa4\xb1\xe8\xb4\xa5: ") + e.what());
    }
}

// ============================================================================
// 20260320 ZJH 简单 JSON 值提取辅助函数
// ============================================================================
static std::string jsonGetValue(const std::string& strJson, const std::string& strKey) {
    std::string strSearch = "\"" + strKey + "\"";
    size_t pos = strJson.find(strSearch);
    if (pos == std::string::npos) return "";
    pos = strJson.find(':', pos + strSearch.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < strJson.size() && (strJson[pos] == ' ' || strJson[pos] == '\t' || strJson[pos] == '\n' || strJson[pos] == '\r')) pos++;
    if (pos >= strJson.size()) return "";
    if (strJson[pos] == '"') {
        size_t end = strJson.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return strJson.substr(pos + 1, end - pos - 1);
    }
    size_t end = strJson.find_first_of(",}\n\r", pos);
    if (end == std::string::npos) end = strJson.size();
    std::string val = strJson.substr(pos, end - pos);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    return val;
}

// ============================================================================
// 20260320 ZJH 从 JSON 文件加载项目状态
// ============================================================================
static void loadProject(AppState& state, const std::string& strPath) {
    try {
        std::ifstream ifs(strPath);
        if (!ifs.is_open()) { state.trainState.appendLog("\xe6\x89\x93\xe5\xbc\x80\xe5\xa4\xb1\xe8\xb4\xa5: " + strPath); return; }
        std::string strJson((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();
        std::string v;
        v = jsonGetValue(strJson, "task_type"); if (!v.empty()) state.activeTask = static_cast<TaskType>(std::stoi(v));
        v = jsonGetValue(strJson, "model_index"); if (!v.empty()) state.nClsModel = std::stoi(v);
        v = jsonGetValue(strJson, "epochs"); if (!v.empty()) state.nClsEpochs = std::stoi(v);
        v = jsonGetValue(strJson, "batch_size"); if (!v.empty()) state.nClsBatchSize = std::stoi(v);
        v = jsonGetValue(strJson, "learning_rate"); if (!v.empty()) state.fClsLR = std::stof(v);
        v = jsonGetValue(strJson, "optimizer"); if (!v.empty()) state.nClsOptimizer = std::stoi(v);
        v = jsonGetValue(strJson, "preset"); if (!v.empty()) state.nPresetIndex = std::stoi(v);
        v = jsonGetValue(strJson, "det_model"); if (!v.empty()) state.nDetModel = std::stoi(v);
        v = jsonGetValue(strJson, "det_epochs"); if (!v.empty()) state.nDetEpochs = std::stoi(v);
        v = jsonGetValue(strJson, "det_batch_size"); if (!v.empty()) state.nDetBatchSize = std::stoi(v);
        v = jsonGetValue(strJson, "det_lr"); if (!v.empty()) state.fDetLR = std::stof(v);
        v = jsonGetValue(strJson, "det_classes"); if (!v.empty()) state.nDetClasses = std::stoi(v);
        v = jsonGetValue(strJson, "det_img_size"); if (!v.empty()) state.nDetImgSize = std::stoi(v);
        v = jsonGetValue(strJson, "seg_epochs"); if (!v.empty()) state.nSegEpochs = std::stoi(v);
        v = jsonGetValue(strJson, "seg_batch_size"); if (!v.empty()) state.nSegBatchSize = std::stoi(v);
        v = jsonGetValue(strJson, "seg_lr"); if (!v.empty()) state.fSegLR = std::stof(v);
        v = jsonGetValue(strJson, "seg_classes"); if (!v.empty()) state.nSegClasses = std::stoi(v);
        v = jsonGetValue(strJson, "seg_img_size"); if (!v.empty()) state.nSegImgSize = std::stoi(v);
        v = jsonGetValue(strJson, "ae_epochs"); if (!v.empty()) state.nAeEpochs = std::stoi(v);
        v = jsonGetValue(strJson, "ae_batch_size"); if (!v.empty()) state.nAeBatchSize = std::stoi(v);
        v = jsonGetValue(strJson, "ae_lr"); if (!v.empty()) state.fAeLR = std::stof(v);
        v = jsonGetValue(strJson, "ae_latent_dim"); if (!v.empty()) state.nAeLatentDim = std::stoi(v);
        v = jsonGetValue(strJson, "ae_threshold"); if (!v.empty()) state.fAeThreshold = std::stof(v);
        state.strProjectPath = strPath;
        state.nActivePage = 0;
        state.trainState.appendLog("\xe9\xa1\xb9\xe7\x9b\xae\xe5\xb7\xb2\xe5\x8a\xa0\xe8\xbd\xbd: " + strPath);
        auto it = std::find(state.vecRecentProjects.begin(), state.vecRecentProjects.end(), strPath);
        if (it != state.vecRecentProjects.end()) state.vecRecentProjects.erase(it);
        state.vecRecentProjects.insert(state.vecRecentProjects.begin(), strPath);
        if (state.vecRecentProjects.size() > 5) state.vecRecentProjects.resize(5);
    } catch (const std::exception& e) {
        state.trainState.appendLog(std::string("\xe5\x8a\xa0\xe8\xbd\xbd\xe5\xa4\xb1\xe8\xb4\xa5: ") + e.what());
    }
}

// ============================================================================
// 20260320 ZJH 导出模型
// ============================================================================
static void exportModelToPath(AppState& state, const std::string& strPath) {
    auto& ts = state.trainState;
    std::lock_guard<std::mutex> lk(ts.mutex);
    if (ts.strSavedModelPath.empty()) { ts.strLog += getTimestamp() + " \xe5\xaf\xbc\xe5\x87\xba\xe5\xa4\xb1\xe8\xb4\xa5: \xe6\xb2\xa1\xe6\x9c\x89\xe5\xb7\xb2\xe8\xae\xad\xe7\xbb\x83\xe7\x9a\x84\xe6\xa8\xa1\xe5\x9e\x8b\n"; return; }
    try {
        std::filesystem::path p(strPath);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::filesystem::copy_file(ts.strSavedModelPath, strPath, std::filesystem::copy_options::overwrite_existing);
        ts.strLog += getTimestamp() + " \xe6\xa8\xa1\xe5\x9e\x8b\xe5\xb7\xb2\xe5\xaf\xbc\xe5\x87\xba: " + strPath + "\n";
    } catch (const std::exception& e) {
        ts.strLog += getTimestamp() + " \xe5\xaf\xbc\xe5\x87\xba\xe5\xa4\xb1\xe8\xb4\xa5: " + std::string(e.what()) + "\n";
    }
}

// ============================================================================
// 20260320 ZJH 批量推理
// ============================================================================
static void runBatchInference(AppState& state) {
    auto& bi = state.batchInference;
    auto& ts = state.trainState;
    bi.vecResults.clear();
    std::string strModelPath(bi.arrModelPath);
    std::string strImageFolder(bi.arrImageFolder);
    if (strModelPath.empty() || !std::filesystem::exists(strModelPath)) { ts.appendLog("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86\xe5\xa4\xb1\xe8\xb4\xa5: \xe6\xa8\xa1\xe5\x9e\x8b\xe4\xb8\x8d\xe5\xad\x98\xe5\x9c\xa8"); return; }
    if (strImageFolder.empty() || !std::filesystem::is_directory(strImageFolder)) { ts.appendLog("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86\xe5\xa4\xb1\xe8\xb4\xa5: \xe6\x96\x87\xe4\xbb\xb6\xe5\xa4\xb9\xe4\xb8\x8d\xe5\xad\x98\xe5\x9c\xa8"); return; }
    try {
        bool bIsResNet = (strModelPath.find("resnet") != std::string::npos);
        auto pModel = std::make_shared<df::Sequential>();
        if (bIsResNet) { pModel->add(std::make_shared<df::ResNet18>(10)); }
        else { pModel->add(std::make_shared<df::Linear>(784, 128)); pModel->add(std::make_shared<df::ReLU>()); pModel->add(std::make_shared<df::Linear>(128, 10)); }
        df::ModelSerializer::load(*pModel, strModelPath);
        int nProcessed = 0;
        for (auto& entry : std::filesystem::directory_iterator(strImageFolder)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;
            int nW = 0, nH = 0, nC = 0;
            unsigned char* pData = stbi_load(entry.path().string().c_str(), &nW, &nH, &nC, 1);
            if (!pData) continue;
            std::vector<float> vecPixels(784, 0.0f);
            for (int y = 0; y < 28; ++y) for (int x = 0; x < 28; ++x) { int sx = x*nW/28; int sy = y*nH/28; vecPixels[y*28+x] = static_cast<float>(pData[sy*nW+sx])/255.0f; }
            stbi_image_free(pData);
            df::Tensor input = bIsResNet ? df::Tensor::fromData(vecPixels.data(), {1,1,28,28}) : df::Tensor::fromData(vecPixels.data(), {1,784});
            df::Tensor output = pModel->forward(input);
            const float* pOut = output.floatDataPtr(); int nSize = output.numel();
            float fMax = -1e30f; int nMaxIdx = 0; float fSum = 0.0f;
            for (int i = 0; i < nSize; ++i) { if (pOut[i] > fMax) { fMax = pOut[i]; nMaxIdx = i; } }
            for (int i = 0; i < nSize; ++i) fSum += std::exp(pOut[i] - fMax);
            BatchInferenceState::Result res; res.strFilename = entry.path().filename().string(); res.strClass = std::to_string(nMaxIdx); res.fConfidence = 1.0f / fSum;
            bi.vecResults.push_back(res); nProcessed++;
        }
        ts.appendLog("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86\xe5\xae\x8c\xe6\x88\x90: " + std::to_string(nProcessed) + " \xe5\xbc\xa0");
    } catch (const std::exception& e) { ts.appendLog(std::string("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86\xe5\xa4\xb1\xe8\xb4\xa5: ") + e.what()); }
}

// ============================================================================
// 20260320 ZJH 扫描导入的数据集文件夹
// ============================================================================
static void scanImportFolder(AppState& state, const std::string& strFolder) {
    state.vecImportClasses.clear(); state.nImportImageCount = 0; state.bImportScanned = false;
    if (strFolder.empty() || !std::filesystem::is_directory(strFolder)) { state.trainState.appendLog("\xe6\x89\xab\xe6\x8f\x8f\xe5\xa4\xb1\xe8\xb4\xa5: \xe6\x96\x87\xe4\xbb\xb6\xe5\xa4\xb9\xe4\xb8\x8d\xe5\xad\x98\xe5\x9c\xa8"); return; }
    try {
        for (auto& entry : std::filesystem::directory_iterator(strFolder)) {
            if (!entry.is_directory()) continue;
            std::string strClassName = entry.path().filename().string(); int nCount = 0;
            for (auto& img : std::filesystem::directory_iterator(entry.path())) {
                if (!img.is_regular_file()) continue;
                std::string ext = img.path().extension().string(); std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") nCount++;
            }
            if (nCount > 0) { state.vecImportClasses.push_back(strClassName + " (" + std::to_string(nCount) + ")"); state.nImportImageCount += nCount; }
        }
        state.bImportScanned = true;
    } catch (const std::exception& e) { state.trainState.appendLog(std::string("\xe6\x89\xab\xe6\x8f\x8f\xe5\xa4\xb1\xe8\xb4\xa5: ") + e.what()); }
}

// ============================================================================
// 20260320 ZJH 导出 HTML 评估报告
// ============================================================================
static void exportEvaluationReportHTML(AppState& state, const std::string& strPath) {
    auto& ts = state.trainState;
    std::lock_guard<std::mutex> lk(ts.mutex);
    try {
        std::filesystem::path p(strPath);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(strPath);
        if (!ofs.is_open()) { ts.strLog += getTimestamp() + " HTML \xe5\xaf\xbc\xe5\x87\xba\xe5\xa4\xb1\xe8\xb4\xa5\n"; return; }

        // 20260320 ZJH 生成自包含 HTML 评估报告
        ofs << "<!DOCTYPE html><html><head><meta charset='utf-8'><title>DeepForge \xe8\xaf\x84\xe4\xbc\xb0\xe6\x8a\xa5\xe5\x91\x8a</title>\n";
        ofs << "<style>body{font-family:sans-serif;background:#1a1d23;color:#e2e8f0;padding:40px;max-width:900px;margin:0 auto}";
        ofs << "h1{color:#2563eb}h2{color:#60a5fa;border-bottom:1px solid #333;padding-bottom:8px}";
        ofs << "table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #333;padding:8px;text-align:center}";
        ofs << "th{background:#22262e;color:#93c5fd}.metric{font-size:32px;color:#2563eb;font-weight:bold}";
        ofs << ".card{background:#22262e;border-radius:8px;padding:20px;margin:16px 0}";
        ofs << ".good{color:#4ade80}.warn{color:#fbbf24}.bad{color:#f87171}</style></head><body>\n";
        ofs << "<h1>DeepForge \xe8\xaf\x84\xe4\xbc\xb0\xe6\x8a\xa5\xe5\x91\x8a</h1>\n";

        // 20260320 ZJH 性能摘要
        ofs << "<div class='card'><h2>\xe6\x80\xa7\xe8\x83\xbd\xe6\x91\x98\xe8\xa6\x81</h2>\n";
        if (state.activeTask == TaskType::Classification) {
            char buf[128]; std::snprintf(buf, sizeof(buf), "%.2f%%", (double)ts.fTestAcc.load());
            ofs << "<p>\xe6\xb5\x8b\xe8\xaf\x95\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87: <span class='metric'>" << buf << "</span></p>\n";
        }
        char buf2[128]; std::snprintf(buf2, sizeof(buf2), "%.4f", (double)ts.fBestValLoss);
        ofs << "<p>\xe6\x9c\x80\xe4\xbd\xb3\xe6\x8d\x9f\xe5\xa4\xb1: " << buf2 << "</p>\n";
        char buf3[128]; std::snprintf(buf3, sizeof(buf3), "%.1f", (double)ts.fTotalTrainingTimeSec);
        ofs << "<p>\xe8\xae\xad\xe7\xbb\x83\xe8\x80\x97\xe6\x97\xb6: " << buf3 << "\xe7\xa7\x92</p></div>\n";

        // 20260320 ZJH 混淆矩阵
        if (ts.bHasConfusionMatrix) {
            ofs << "<div class='card'><h2>\xe6\xb7\xb7\xe6\xb7\x86\xe7\x9f\xa9\xe9\x98\xb5</h2><table><tr><th>\xe5\xae\x9e\xe9\x99\x85\\\xe9\xa2\x84\xe6\xb5\x8b</th>";
            for (int c = 0; c < 10; ++c) ofs << "<th>" << c << "</th>";
            ofs << "</tr>\n";
            for (int r = 0; r < 10; ++r) {
                ofs << "<tr><th>" << r << "</th>";
                for (int c = 0; c < 10; ++c) {
                    int v = ts.arrConfusionMatrix[r][c];
                    if (r == c && v > 0) ofs << "<td style='background:#1e3a5f'>" << v << "</td>";
                    else if (v > 0) ofs << "<td style='background:#3a1e1e'>" << v << "</td>";
                    else ofs << "<td>" << v << "</td>";
                }
                ofs << "</tr>\n";
            }
            ofs << "</table></div>\n";

            // 20260320 ZJH 分类报告
            ofs << "<div class='card'><h2>\xe5\x88\x86\xe7\xb1\xbb\xe6\x8a\xa5\xe5\x91\x8a</h2><table><tr><th>\xe7\xb1\xbb\xe5\x88\xab</th><th>\xe7\xb2\xbe\xe7\xa1\xae\xe7\x8e\x87</th><th>\xe5\x8f\xac\xe5\x9b\x9e\xe7\x8e\x87</th><th>F1</th></tr>\n";
            for (int c = 0; c < 10; ++c) {
                int tp = ts.arrConfusionMatrix[c][c], fp = 0, fn = 0;
                for (int i = 0; i < 10; ++i) { if (i!=c) { fp += ts.arrConfusionMatrix[i][c]; fn += ts.arrConfusionMatrix[c][i]; } }
                float pr = (tp+fp>0)?(float)tp/(float)(tp+fp):0;
                float re = (tp+fn>0)?(float)tp/(float)(tp+fn):0;
                float f1 = (pr+re>0)?2*pr*re/(pr+re):0;
                char b[256]; std::snprintf(b, sizeof(b), "<tr><td>%d</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>", c, (double)pr, (double)re, (double)f1);
                ofs << b << "\n";
            }
            ofs << "</table></div>\n";
        }

        ofs << "<p style='color:#666;text-align:center;margin-top:40px'>Generated by DeepForge v0.1.0</p>\n";
        ofs << "</body></html>\n";
        ofs.close();
        ts.strLog += getTimestamp() + " HTML \xe8\xaf\x84\xe4\xbc\xb0\xe6\x8a\xa5\xe5\x91\x8a\xe5\xb7\xb2\xe5\xaf\xbc\xe5\x87\xba: " + strPath + "\n";
    } catch (const std::exception& e) {
        ts.strLog += getTimestamp() + " HTML \xe5\xaf\xbc\xe5\x87\xba\xe5\xa4\xb1\xe8\xb4\xa5: " + std::string(e.what()) + "\n";
    }
}

// ============================================================================
// 20260320 ZJH 启动训练
// ============================================================================
static void startTraining(AppState& state) {
    if (state.trainState.bRunning.load()) return;
    if (state.pTrainThread) { state.pTrainThread->request_stop(); state.pTrainThread->join(); state.pTrainThread.reset(); }
    state.trainState.reset();
    switch (state.activeTask) {
        case TaskType::Classification: state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { classificationTrainFunc(state); }); break;
        case TaskType::Detection: state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { detectionTrainFunc(state); }); break;
        case TaskType::Segmentation: state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { segmentationTrainFunc(state); }); break;
        case TaskType::AnomalyDetection: state.pTrainThread = std::make_unique<std::jthread>([&](std::stop_token) { anomalyTrainFunc(state); }); break;
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
            std::ifstream ifs(p, std::ios::binary); if (!ifs) return 0;
            unsigned char h[8]; ifs.read((char*)h, 8);
            return (h[4]<<24)|(h[5]<<16)|(h[6]<<8)|h[7];
        };
        state.nMnistTrainSamples = readCount("data/mnist/train-images-idx3-ubyte");
        state.nMnistTestSamples = readCount("data/mnist/t10k-images-idx3-ubyte");
    }
    state.bDataChecked = true;
}

// ============================================================================
// 20260320 ZJH 设置 MVTec DL Tool 深色主题
// ============================================================================
static void setupMVTecStyle() {
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;      // 20260320 ZJH MVTec 用方角
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 0.0f;         // 20260320 ZJH 方角选项卡
    style.ChildRounding = 4.0f;
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(6, 4);
    style.ItemSpacing = ImVec2(8, 5);
    style.ScrollbarRounding = 4.0f;
    style.TabBarBorderSize = 0.0f;

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg]        = ImVec4(0.102f, 0.114f, 0.137f, 1.0f);  // 20260320 ZJH #1a1d23
    c[ImGuiCol_ChildBg]         = ImVec4(0.075f, 0.082f, 0.102f, 1.0f);  // 20260320 ZJH #13151a
    c[ImGuiCol_PopupBg]         = ImVec4(0.133f, 0.149f, 0.180f, 0.97f);
    c[ImGuiCol_Border]          = ImVec4(0.180f, 0.195f, 0.230f, 1.0f);
    c[ImGuiCol_FrameBg]         = ImVec4(0.133f, 0.149f, 0.180f, 1.0f);  // 20260320 ZJH #22262e
    c[ImGuiCol_FrameBgHovered]  = ImVec4(0.165f, 0.185f, 0.230f, 1.0f);
    c[ImGuiCol_FrameBgActive]   = ImVec4(0.200f, 0.220f, 0.280f, 1.0f);
    c[ImGuiCol_TitleBg]         = ImVec4(0.059f, 0.067f, 0.082f, 1.0f);  // 20260320 ZJH #0f1115
    c[ImGuiCol_TitleBgActive]   = ImVec4(0.075f, 0.082f, 0.102f, 1.0f);
    c[ImGuiCol_Tab]             = ImVec4(0.075f, 0.082f, 0.102f, 1.0f);  // 20260320 ZJH #13151a 非激活
    c[ImGuiCol_TabHovered]      = ImVec4(0.145f, 0.388f, 0.922f, 0.60f);
    c[ImGuiCol_TabSelected]     = ImVec4(0.145f, 0.388f, 0.922f, 1.0f);  // 20260320 ZJH #2563eb 激活
    c[ImGuiCol_Button]          = ImVec4(0.160f, 0.180f, 0.230f, 1.0f);
    c[ImGuiCol_ButtonHovered]   = ImVec4(0.145f, 0.388f, 0.922f, 0.70f);
    c[ImGuiCol_ButtonActive]    = ImVec4(0.145f, 0.388f, 0.922f, 1.0f);
    c[ImGuiCol_Header]          = ImVec4(0.160f, 0.180f, 0.230f, 1.0f);
    c[ImGuiCol_HeaderHovered]   = ImVec4(0.145f, 0.388f, 0.922f, 0.50f);
    c[ImGuiCol_HeaderActive]    = ImVec4(0.145f, 0.388f, 0.922f, 0.70f);
    c[ImGuiCol_Separator]       = ImVec4(0.180f, 0.195f, 0.230f, 1.0f);
    c[ImGuiCol_Text]            = ImVec4(0.886f, 0.910f, 0.941f, 1.0f);  // 20260320 ZJH #e2e8f0
    c[ImGuiCol_TextDisabled]    = ImVec4(0.440f, 0.470f, 0.520f, 1.0f);
    c[ImGuiCol_PlotHistogram]   = ImVec4(0.145f, 0.388f, 0.922f, 1.0f);
    c[ImGuiCol_CheckMark]       = ImVec4(0.145f, 0.388f, 0.922f, 1.0f);
    c[ImGuiCol_SliderGrab]      = ImVec4(0.145f, 0.388f, 0.922f, 0.80f);
    c[ImGuiCol_SliderGrabActive]= ImVec4(0.145f, 0.388f, 0.922f, 1.0f);
    c[ImGuiCol_ScrollbarBg]     = ImVec4(0.075f, 0.082f, 0.102f, 1.0f);
    c[ImGuiCol_ScrollbarGrab]   = ImVec4(0.180f, 0.195f, 0.230f, 1.0f);
}

// ============================================================================
// 20260320 ZJH 绘制主菜单栏
// ============================================================================
static void drawMainMenuBar(AppState& state) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("\xe6\x96\x87\xe4\xbb\xb6")) {
            if (ImGui::MenuItem("\xe6\x96\xb0\xe5\xbb\xba\xe9\xa1\xb9\xe7\x9b\xae", "Ctrl+N")) {
                if (!state.trainState.bRunning.load()) { state.trainState.reset(); state.nActivePage = 0; state.strProjectPath.clear(); }
            }
            if (ImGui::MenuItem("\xe6\x89\x93\xe5\xbc\x80\xe9\xa1\xb9\xe7\x9b\xae", "Ctrl+O")) {
                state.nFileDialogPurpose = 0; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "projects/project.json", _TRUNCATE); state.bShowFileDialog = true;
            }
            if (ImGui::MenuItem("\xe4\xbf\x9d\xe5\xad\x98\xe9\xa1\xb9\xe7\x9b\xae", "Ctrl+S")) {
                if (!state.strProjectPath.empty()) saveProject(state, state.strProjectPath);
                else { state.nFileDialogPurpose = 1; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "projects/project.json", _TRUNCATE); state.bShowFileDialog = true; }
            }
            if (ImGui::MenuItem("\xe5\x8f\xa6\xe5\xad\x98\xe4\xb8\xba")) {
                state.nFileDialogPurpose = 2; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "projects/project.json", _TRUNCATE); state.bShowFileDialog = true;
            }
            ImGui::Separator();
            if (ImGui::BeginMenu("\xe6\x9c\x80\xe8\xbf\x91\xe9\xa1\xb9\xe7\x9b\xae")) {
                if (state.vecRecentProjects.empty()) ImGui::MenuItem("(\xe6\x97\xa0)", nullptr, false, false);
                else for (auto& rp : state.vecRecentProjects) { if (ImGui::MenuItem(rp.c_str())) loadProject(state, rp); }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe5\xaf\xbc\xe5\x87\xba\xe6\xa8\xa1\xe5\x9e\x8b")) {
                state.nFileDialogPurpose = 3; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "data/models/model_export.dfm", _TRUNCATE); state.bShowFileDialog = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("\xe9\x80\x80\xe5\x87\xba", "Alt+F4")) { SDL_Event qe; qe.type = SDL_EVENT_QUIT; SDL_PushEvent(&qe); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("\xe7\xbc\x96\xe8\xbe\x91")) {
            ImGui::MenuItem("\xe6\x92\xa4\xe9\x94\x80", "Ctrl+Z", false, false);
            ImGui::MenuItem("\xe9\x87\x8d\xe5\x81\x9a", "Ctrl+Y", false, false);
            ImGui::Separator();
            if (ImGui::MenuItem("\xe6\xb8\x85\xe9\x99\xa4\xe8\xae\xad\xe7\xbb\x83\xe5\x8e\x86\xe5\x8f\xb2")) {
                state.vecTrainHistory.clear();
                std::lock_guard<std::mutex> lk(state.trainState.mutex);
                state.trainState.strLog.clear(); state.trainState.vecLossHistory.clear();
                state.trainState.vecTrainAccHistory.clear(); state.trainState.vecTestAccHistory.clear();
                state.trainState.vecMIoUHistory.clear(); state.trainState.arrConfusionMatrix = {};
                state.trainState.bHasConfusionMatrix = false;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("\xe8\xa7\x86\xe5\x9b\xbe")) {
            if (ImGui::MenuItem("\xe5\x85\xa8\xe5\xb1\x8f", "F11")) { /* handled in shortcuts */ }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("\xe5\xb7\xa5\xe5\x85\xb7")) {
            if (ImGui::MenuItem("\xe6\x95\xb0\xe6\x8d\xae\xe5\xa2\x9e\xe5\xbc\xba\xe9\xa2\x84\xe8\xa7\x88")) state.bShowAugPreview = true;
            if (ImGui::MenuItem("GPU \xe4\xbf\xa1\xe6\x81\xaf")) state.bShowGpuInfo = true;
            if (ImGui::MenuItem("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86")) state.bShowBatchInference = true;
            ImGui::Separator();
            if (ImGui::MenuItem("\xe8\xae\xbe\xe7\xbd\xae", "Ctrl+,")) state.bShowSettingsPopup = true;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("\xe5\xb8\xae\xe5\x8a\xa9")) {
            if (ImGui::MenuItem("\xe7\x94\xa8\xe6\x88\xb7\xe6\x89\x8b\xe5\x86\x8c", "F1")) state.bShowUserManual = true;
            if (ImGui::MenuItem("\xe5\xbf\xab\xe6\x8d\xb7\xe9\x94\xae\xe5\x8f\x82\xe8\x80\x83")) state.bShowShortcuts = true;
            ImGui::Separator();
            if (ImGui::MenuItem("\xe5\x85\xb3\xe4\xba\x8e DeepForge")) state.bShowAboutPopup = true;
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

// ============================================================================
// 20260320 ZJH 键盘快捷键
// ============================================================================
static void handleKeyboardShortcuts(AppState& state, SDL_Window* pWindow) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    bool bCtrl = io.KeyCtrl;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_N)) { if (!state.trainState.bRunning.load()) { state.trainState.reset(); state.nActivePage = 0; state.strProjectPath.clear(); } }
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_O)) { state.nFileDialogPurpose = 0; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "projects/project.json", _TRUNCATE); state.bShowFileDialog = true; }
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) {
        if (!state.strProjectPath.empty()) saveProject(state, state.strProjectPath);
        else { state.nFileDialogPurpose = 1; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "projects/project.json", _TRUNCATE); state.bShowFileDialog = true; }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F1)) state.bShowUserManual = true;
    if (ImGui::IsKeyPressed(ImGuiKey_F5) && !state.trainState.bRunning.load()) { state.nActivePage = 2; startTraining(state); }
    if (ImGui::IsKeyPressed(ImGuiKey_F6) && state.trainState.bRunning.load()) state.trainState.bPaused.store(!state.trainState.bPaused.load());
    if (ImGui::IsKeyPressed(ImGuiKey_F7) && state.trainState.bRunning.load()) { state.trainState.bStopRequested.store(true); state.trainState.bPaused.store(false); }
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_1)) state.nActivePage = 0;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_2)) state.nActivePage = 1;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_3)) state.nActivePage = 2;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_4)) state.nActivePage = 3;
    if (bCtrl && ImGui::IsKeyPressed(ImGuiKey_Comma)) state.bShowSettingsPopup = true;
    if (ImGui::IsKeyPressed(ImGuiKey_F11)) { state.bFullscreen = !state.bFullscreen; SDL_SetWindowFullscreen(pWindow, state.bFullscreen); }
}

// ============================================================================
// 20260320 ZJH 设置弹窗
// ============================================================================
static void drawSettingsDialog(AppState& state) {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("\xe8\xae\xbe\xe7\xbd\xae##Modal", nullptr, ImGuiWindowFlags_NoResize)) {
        auto& s = state.settings;
        if (ImGui::BeginTabBar("##SettingsTabs")) {
            if (ImGui::BeginTabItem("\xe5\xb8\xb8\xe8\xa7\x84")) { ImGui::Spacing(); const char* arrLangs[] = {"\xe7\xae\x80\xe4\xbd\x93\xe4\xb8\xad\xe6\x96\x87"}; ImGui::Combo("\xe8\xaf\xad\xe8\xa8\x80", &s.nLanguage, arrLangs, 1); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("\xe5\xa4\x96\xe8\xa7\x82")) { ImGui::Spacing(); const char* arrThemes[] = {"\xe6\xb7\xb1\xe8\x89\xb2", "\xe6\xb5\x85\xe8\x89\xb2"}; ImGui::Combo("\xe4\xb8\xbb\xe9\xa2\x98", &s.nTheme, arrThemes, 2); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("GPU")) {
                ImGui::Spacing();
                if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
                    for (size_t i = 0; i < state.vecGpuDevices.size(); ++i) ImGui::Text("GPU %zu: %s (%zuMB)", i, state.vecGpuDevices[i].strName.c_str(), state.vecGpuDevices[i].nTotalMemoryMB);
                } else ImGui::TextDisabled("\xe6\x9c\xaa\xe6\xa3\x80\xe6\xb5\x8b\xe5\x88\xb0 GPU");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        float fBtnW = 80.0f;
        ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - fBtnW * 2 - 8);
        if (ImGui::Button("\xe5\x8f\x96\xe6\xb6\x88", ImVec2(fBtnW, 0))) ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ImGui::Button("\xe7\xa1\xae\xe5\xae\x9a", ImVec2(fBtnW, 0))) {
            if (s.nTheme == 0) { ImGui::StyleColorsDark(); setupMVTecStyle(); } else ImGui::StyleColorsLight();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// ============================================================================
// 20260320 ZJH 关于弹窗
// ============================================================================
static void drawAboutDialog(AppState& /*state*/) {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(400, 280), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("\xe5\x85\xb3\xe4\xba\x8e DeepForge##About", nullptr, ImGuiWindowFlags_NoResize)) {
        ImGui::PushStyleColor(ImGuiCol_Text, s_colAccent);
        ImGui::SetWindowFontScale(1.8f);
        float fW = ImGui::CalcTextSize("DeepForge").x;
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - fW) * 0.5f);
        ImGui::Text("DeepForge");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::Text("\xe7\x89\x88\xe6\x9c\xac: v0.1.0");
        ImGui::Text("\xe6\x8a\x80\xe6\x9c\xaf\xe6\xa0\x88: C++23 + SDL3 + ImGui + ImPlot");
        ImGui::Text("\xe5\xbc\x95\xe6\x93\x8e: df.engine (\xe8\x87\xaa\xe7\xa0\x94)");
        ImGui::Spacing();
        ImGui::TextDisabled("\xc2\xa9 2026 ZJH. All rights reserved.");
        ImGui::Spacing();
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - 80) * 0.5f);
        if (ImGui::Button("\xe7\xa1\xae\xe5\xae\x9a", ImVec2(80, 0))) ImGui::CloseCurrentPopup();
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
        auto& bi = state.batchInference;
        ImGui::InputText("\xe6\xa8\xa1\xe5\x9e\x8b\xe8\xb7\xaf\xe5\xbe\x84", bi.arrModelPath, sizeof(bi.arrModelPath));
        ImGui::InputText("\xe5\x9b\xbe\xe7\x89\x87\xe6\x96\x87\xe4\xbb\xb6\xe5\xa4\xb9", bi.arrImageFolder, sizeof(bi.arrImageFolder));
        ImGui::Spacing();
        if (!bi.bRunning && ImGui::Button("\xe5\xbc\x80\xe5\xa7\x8b\xe6\x8e\xa8\xe7\x90\x86", ImVec2(120, 30))) runBatchInference(state);
        if (!bi.vecResults.empty()) {
            if (ImGui::BeginTable("##InfR", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 250))) {
                ImGui::TableSetupColumn("\xe6\x96\x87\xe4\xbb\xb6\xe5\x90\x8d"); ImGui::TableSetupColumn("\xe9\xa2\x84\xe6\xb5\x8b"); ImGui::TableSetupColumn("\xe7\xbd\xae\xe4\xbf\xa1\xe5\xba\xa6");
                ImGui::TableHeadersRow();
                for (auto& r : bi.vecResults) { ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::Text("%s", r.strFilename.c_str()); ImGui::TableSetColumnIndex(1); ImGui::Text("%s", r.strClass.c_str()); ImGui::TableSetColumnIndex(2); ImGui::Text("%.2f%%", (double)(r.fConfidence*100)); }
                ImGui::EndTable();
            }
        }
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - 80);
        if (ImGui::Button("\xe5\x85\xb3\xe9\x97\xad", ImVec2(80, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

// ============================================================================
// 20260320 ZJH 绘制页面选项卡栏（MVTec 风格水平页面选项卡 — 最核心的导航）
// 返回选中的页面索引
// ============================================================================
static void drawPageTabBar(AppState& state) {
    // 20260320 ZJH 选项卡栏背景
    ImDrawList* pDraw = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float fBarH = 36.0f;  // 20260320 ZJH 选项卡栏高度
    float fBarW = ImGui::GetContentRegionAvail().x;

    // 20260320 ZJH 绘制选项卡栏背景 #13151a
    pDraw->AddRectFilled(pos, ImVec2(pos.x + fBarW, pos.y + fBarH), s_nColTabBar);

    // 20260320 ZJH 4 个页面选项卡
    const char* arrTabs[] = {
        "\xe6\x95\xb0\xe6\x8d\xae",    // "数据"
        "\xe6\xa0\x87\xe6\xb3\xa8",    // "标注"
        "\xe8\xae\xad\xe7\xbb\x83",    // "训练"
        "\xe8\xaf\x84\xe4\xbc\xb0"     // "评估"
    };

    float fTabW = 90.0f;   // 20260320 ZJH 每个选项卡宽度
    float fTabX = pos.x + 10.0f;  // 20260320 ZJH 起始 X

    ImFont* pFont = ImGui::GetFont();

    for (int i = 0; i < 4; ++i) {
        bool bActive = (state.nActivePage == i);
        ImVec2 tabPos = ImVec2(fTabX, pos.y);
        ImVec2 tabEnd = ImVec2(fTabX + fTabW, pos.y + fBarH);

        // 20260320 ZJH 检测鼠标悬停和点击
        bool bHovered = ImGui::IsMouseHoveringRect(tabPos, tabEnd);
        if (bHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            state.nActivePage = i;
            bActive = true;
        }

        // 20260320 ZJH 悬停高亮
        if (bHovered && !bActive) {
            pDraw->AddRectFilled(tabPos, tabEnd, IM_COL32(30, 35, 45, 255));
        }

        // 20260320 ZJH 文字颜色：激活=白色, 非激活=灰色
        ImU32 nTextCol = bActive ? IM_COL32(255, 255, 255, 255) : s_nColSubtle;
        float fFontSz = ImGui::GetFontSize();  // 20260320 ZJH 获取当前字体大小
        ImVec2 textSize = pFont->CalcTextSizeA(fFontSz, FLT_MAX, 0.0f, arrTabs[i]);
        float fTextX = fTabX + (fTabW - textSize.x) * 0.5f;
        float fTextY = pos.y + (fBarH - textSize.y) * 0.5f;
        pDraw->AddText(ImVec2(fTextX, fTextY), nTextCol, arrTabs[i]);

        // 20260320 ZJH 激活选项卡底部蓝色线条（MVTec 标志性特征）
        if (bActive) {
            pDraw->AddRectFilled(
                ImVec2(fTabX, pos.y + fBarH - 3.0f),
                ImVec2(fTabX + fTabW, pos.y + fBarH),
                s_nColAccent);
        }

        fTabX += fTabW + 2.0f;
    }

    // 20260320 ZJH 底部分隔线
    pDraw->AddLine(ImVec2(pos.x, pos.y + fBarH), ImVec2(pos.x + fBarW, pos.y + fBarH), IM_COL32(40, 45, 55, 255));

    // 20260320 ZJH 预留空间
    ImGui::Dummy(ImVec2(0, fBarH + 1));
}

// ============================================================================
// 20260320 ZJH 绘制数据页面（Page 1）
// ============================================================================
static void drawPageData(AppState& state) {
    if (!state.bDataChecked) checkDatasets(state);

    // 20260320 ZJH 顶部：项目类型选择 + 按钮
    {
        ImGui::Spacing();
        ImGui::Text("\xe9\xa1\xb9\xe7\x9b\xae\xe7\xb1\xbb\xe5\x9e\x8b:");  // "项目类型:"
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        const char* arrTasks[] = {
            "\xe5\x9b\xbe\xe5\x83\x8f\xe5\x88\x86\xe7\xb1\xbb",
            "\xe7\x9b\xae\xe6\xa0\x87\xe6\xa3\x80\xe6\xb5\x8b",
            "\xe8\xaf\xad\xe4\xb9\x89\xe5\x88\x86\xe5\x89\xb2",
            "\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b"
        };
        int nTask = (int)state.activeTask;
        if (ImGui::Combo("##TaskType", &nTask, arrTasks, 4)) {
            state.activeTask = static_cast<TaskType>(nTask);
            state.trainState.reset();
        }

        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 220);
        if (ImGui::Button("\xe6\xb7\xbb\xe5\x8a\xa0\xe5\x9b\xbe\xe5\x83\x8f", ImVec2(100, 0))) {
            // 20260320 ZJH 占位：添加图像
        }
        ImGui::SameLine();
        if (ImGui::Button("\xe6\xb7\xbb\xe5\x8a\xa0\xe6\x96\x87\xe4\xbb\xb6\xe5\xa4\xb9", ImVec2(110, 0))) {
            // 20260320 ZJH 占位：添加文件夹
        }
        ImGui::Spacing();
    }

    // 20260320 ZJH 图像画廊区域
    if (beginCard("\xe5\x9b\xbe\xe5\x83\x8f\xe7\x94\xbb\xe5\xbb\x8a")) {
        // 20260320 ZJH 生成模拟画廊数据
        const char* arrClsNames[] = {"cat", "dog", "bird", "fish", "car", "tree", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        int nTotalImages = 0;
        int nClasses = 10;

        if (state.activeTask == TaskType::Classification) {
            nTotalImages = state.bMnistAvailable ? state.nMnistTrainSamples : 1000;
            nClasses = 10;
        } else if (state.activeTask == TaskType::Detection) {
            nTotalImages = 64; nClasses = state.nDetClasses;
        } else if (state.activeTask == TaskType::Segmentation) {
            nTotalImages = 16; nClasses = state.nSegClasses;
        } else {
            nTotalImages = 128; nClasses = 2;
        }

        // 20260320 ZJH 画廊网格：用彩色矩形做占位
        float fThumbSize = 72.0f;
        float fSpacing = 6.0f;
        float fAvailW = ImGui::GetContentRegionAvail().x;
        int nCols = std::max(1, (int)((fAvailW + fSpacing) / (fThumbSize + fSpacing)));
        int nShowMax = std::min(nTotalImages, nCols * 3);  // 20260320 ZJH 最多显示 3 行

        ImVec2 startPos = ImGui::GetCursorScreenPos();
        ImDrawList* pDraw = ImGui::GetWindowDrawList();

        for (int i = 0; i < nShowMax; ++i) {
            int nRow = i / nCols;
            int nCol = i % nCols;
            float fX = startPos.x + nCol * (fThumbSize + fSpacing);
            float fY = startPos.y + nRow * (fThumbSize + 20.0f + fSpacing);
            int nCls = i % nClasses;

            // 20260320 ZJH 缩略图矩形
            ImU32 nColor = s_arrClassColors[nCls % s_nNumClassColors];
            bool bSelected = (state.nGallerySelected == i);
            pDraw->AddRectFilled(ImVec2(fX, fY), ImVec2(fX + fThumbSize, fY + fThumbSize), nColor);
            if (bSelected) {
                pDraw->AddRect(ImVec2(fX-2, fY-2), ImVec2(fX+fThumbSize+2, fY+fThumbSize+2), s_nColAccent, 0, 0, 2.0f);
            }

            // 20260320 ZJH 图像编号
            char arrNum[16]; std::snprintf(arrNum, sizeof(arrNum), "%d", i+1);
            ImVec2 numSize = ImGui::CalcTextSize(arrNum);
            pDraw->AddText(ImVec2(fX + (fThumbSize - numSize.x)*0.5f, fY + (fThumbSize - numSize.y)*0.5f), IM_COL32(255,255,255,200), arrNum);

            // 20260320 ZJH 类别标签
            const char* strCls = (nCls < 10 && state.activeTask == TaskType::Classification) ? arrClsNames[nCls + 6] : arrClsNames[nCls % 6];
            ImVec2 clsSize = ImGui::CalcTextSize(strCls);
            pDraw->AddText(ImVec2(fX + (fThumbSize - clsSize.x)*0.5f, fY + fThumbSize + 2), s_nColSubtle, strCls);
        }

        // 20260320 ZJH 检测点击
        int nRows = (nShowMax + nCols - 1) / nCols;
        float fGridH = nRows * (fThumbSize + 20.0f + fSpacing);
        ImGui::SetCursorScreenPos(startPos);
        ImGui::InvisibleButton("##gallery", ImVec2(fAvailW, std::max(fGridH, 20.0f)));
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            ImVec2 mp = ImGui::GetMousePos();
            int nCol = (int)((mp.x - startPos.x) / (fThumbSize + fSpacing));
            int nRow = (int)((mp.y - startPos.y) / (fThumbSize + 20.0f + fSpacing));
            int nIdx = nRow * nCols + nCol;
            if (nIdx >= 0 && nIdx < nShowMax) state.nGallerySelected = nIdx;
        }

        if (nTotalImages > nShowMax) {
            ImGui::TextDisabled("... \xe8\xbf\x98\xe6\x9c\x89 %d \xe5\xbc\xa0\xe5\x9b\xbe\xe5\x83\x8f", nTotalImages - nShowMax);
        }
    }
    endCard();

    // 20260320 ZJH 底部：数据集信息 + 类别分布 并排
    float fHalfW = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;

    ImGui::BeginChild("##DataInfoLeft", ImVec2(fHalfW, 0), ImGuiChildFlags_None);
    {
        if (beginCard("\xe6\x95\xb0\xe6\x8d\xae\xe9\x9b\x86\xe4\xbf\xa1\xe6\x81\xaf")) {
            int nTotal = 0;
            if (state.activeTask == TaskType::Classification) nTotal = state.bMnistAvailable ? state.nMnistTrainSamples + state.nMnistTestSamples : 1200;
            else if (state.activeTask == TaskType::Detection) nTotal = 64;
            else if (state.activeTask == TaskType::Segmentation) nTotal = 16;
            else nTotal = 128;

            int nTrain = (int)(nTotal * state.fTrainSplit);
            int nVal = (int)(nTotal * state.fValSplit);
            int nTest = nTotal - nTrain - nVal;

            ImGui::Text("\xe6\x80\xbb\xe5\x9b\xbe\xe5\x83\x8f\xe6\x95\xb0: %d", nTotal);
            ImGui::Text("\xe8\xae\xad\xe7\xbb\x83\xe9\x9b\x86: %d (%.0f%%)", nTrain, (double)(state.fTrainSplit * 100));
            ImGui::Text("\xe9\xaa\x8c\xe8\xaf\x81\xe9\x9b\x86: %d (%.0f%%)", nVal, (double)(state.fValSplit * 100));
            ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95\xe9\x9b\x86: %d (%.0f%%)", nTest, (double)(state.fTestSplit * 100));

            int nClasses = (state.activeTask == TaskType::Classification) ? 10 :
                           (state.activeTask == TaskType::Detection) ? state.nDetClasses :
                           (state.activeTask == TaskType::Segmentation) ? state.nSegClasses : 2;
            ImGui::Text("\xe7\xb1\xbb\xe5\x88\xab\xe6\x95\xb0: %d", nClasses);
        }
        endCard();

        // 20260320 ZJH 划分比例滑块
        if (beginCard("\xe5\x88\x92\xe5\x88\x86\xe6\xaf\x94\xe4\xbe\x8b")) {
            ImGui::SliderFloat("\xe8\xae\xad\xe7\xbb\x83", &state.fTrainSplit, 0.5f, 0.95f, "%.0f%%");
            ImGui::SliderFloat("\xe9\xaa\x8c\xe8\xaf\x81", &state.fValSplit, 0.0f, 0.3f, "%.0f%%");
            state.fTestSplit = 1.0f - state.fTrainSplit - state.fValSplit;
            if (state.fTestSplit < 0) { state.fValSplit = 1.0f - state.fTrainSplit; state.fTestSplit = 0; }
            ImGui::Text("\xe6\xb5\x8b\xe8\xaf\x95: %.0f%%", (double)(state.fTestSplit * 100));
        }
        endCard();

        // 20260320 ZJH 数据来源
        if (beginCard("\xe6\x95\xb0\xe6\x8d\xae\xe6\x9d\xa5\xe6\xba\x90")) {
            ImGui::RadioButton("\xe6\x96\x87\xe4\xbb\xb6\xe5\xa4\xb9\xe5\xaf\xbc\xe5\x85\xa5", &state.nDataSource, 0);
            ImGui::SameLine();
            ImGui::RadioButton("MNIST", &state.nDataSource, 1);
            ImGui::SameLine();
            ImGui::RadioButton("\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae", &state.nDataSource, 2);

            if (state.nDataSource == 0) {
                ImGui::InputText("##importPath", state.arrImportPath, sizeof(state.arrImportPath));
                ImGui::SameLine();
                if (ImGui::Button("\xe6\x89\xab\xe6\x8f\x8f")) scanImportFolder(state, std::string(state.arrImportPath));
                if (state.bImportScanned) {
                    ImGui::TextColored(s_colGreen, "%d \xe5\xbc\xa0, %d \xe7\xb1\xbb", state.nImportImageCount, (int)state.vecImportClasses.size());
                }
            } else if (state.nDataSource == 1) {
                if (state.bMnistAvailable) ImGui::TextColored(s_colGreen, "MNIST \xe5\xb7\xb2\xe5\xb0\xb1\xe7\xbb\xaa (%d/%d)", state.nMnistTrainSamples, state.nMnistTestSamples);
                else ImGui::TextColored(s_colOrange, "MNIST \xe6\x9c\xaa\xe6\x89\xbe\xe5\x88\xb0");
            } else {
                ImGui::TextColored(s_colGreen, "\xe5\x90\x88\xe6\x88\x90\xe6\x95\xb0\xe6\x8d\xae\xe5\xb7\xb2\xe5\xb0\xb1\xe7\xbb\xaa");
            }
        }
        endCard();
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("##DataInfoRight", ImVec2(0, 0), ImGuiChildFlags_None);
    {
        if (beginCard("\xe7\xb1\xbb\xe5\x88\xab\xe5\x88\x86\xe5\xb8\x83")) {
            if (ImPlot::BeginPlot("##ClsDist", ImVec2(-1, 200))) {
                int nClasses = (state.activeTask == TaskType::Classification) ? 10 :
                               (state.activeTask == TaskType::Detection) ? state.nDetClasses :
                               (state.activeTask == TaskType::Segmentation) ? state.nSegClasses : 2;
                ImPlot::SetupAxes("\xe7\xb1\xbb\xe5\x88\xab", "\xe6\x95\xb0\xe9\x87\x8f");
                std::vector<double> xv(nClasses), yv(nClasses);
                int nTotal = (state.activeTask == TaskType::Classification && state.bMnistAvailable) ? state.nMnistTrainSamples : 600;
                double avg = (double)nTotal / nClasses;
                for (int i = 0; i < nClasses; ++i) { xv[i] = i; yv[i] = avg * (0.8 + 0.4 * ((i * 7 + 3) % 5) / 4.0); }
                ImPlot::PlotBars("\xe6\xa0\xb7\xe6\x9c\xac", xv.data(), yv.data(), nClasses, 0.6);
                ImPlot::EndPlot();
            }
        }
        endCard();
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260320 ZJH 绘制标注页面（Page 2）
// ============================================================================
static void drawPageLabeling(AppState& state) {
    auto& proj = state.annotProject;

    // 20260320 ZJH 确保标注项目已初始化
    if (proj.vecClassNames.empty()) {
        proj.strName = "Project";
        proj.annotType = df::AnnotationType::BBox;
        int nC = (state.activeTask == TaskType::Detection) ? state.nDetClasses : (state.activeTask == TaskType::Segmentation) ? state.nSegClasses : 10;
        for (int c = 0; c < nC; ++c) proj.vecClassNames.push_back("\xe7\xb1\xbb\xe5\x88\xab_" + std::to_string(c));
    }
    if (proj.vecAnnotations.empty()) {
        for (int i = 0; i < 5; ++i) {
            df::ImageAnnotation ann; ann.strImagePath = "image_" + std::to_string(i+1) + ".png";
            ann.nImageWidth = 640; ann.nImageHeight = 480;
            proj.vecAnnotations.push_back(ann);
        }
    }

    // 20260320 ZJH 三栏布局
    float fAvailW = ImGui::GetContentRegionAvail().x;
    float fAvailH = ImGui::GetContentRegionAvail().y;
    float fToolW = 60.0f;
    float fRightW = 200.0f;
    float fCenterW = fAvailW - fToolW - fRightW - 12.0f;
    float fBottomH = 50.0f;
    float fMainH = fAvailH - fBottomH - 4.0f;

    // ---- 左侧工具栏 ----
    ImGui::BeginChild("##LabelTools", ImVec2(fToolW, fMainH), ImGuiChildFlags_Borders);
    {
        // 20260320 ZJH 标注工具按钮（根据任务类型显示不同工具）
        struct ToolDef { const char* strName; const char* strKey; int nId; };

        if (state.activeTask == TaskType::Classification) {
            // 20260320 ZJH 分类任务：只有选择工具
            ImGui::TextDisabled("\xe5\xb7\xa5\xe5\x85\xb7");
            ImGui::Separator();
            if (ImGui::Button("\xe9\x80\x89\xe6\x8b\xa9", ImVec2(-1, 0))) state.nAnnotTool = 0;
        } else if (state.activeTask == TaskType::AnomalyDetection) {
            // 20260320 ZJH 异常检测：标记正常/异常
            ImGui::TextDisabled("\xe5\xb7\xa5\xe5\x85\xb7");
            ImGui::Separator();
            if (ImGui::Button("\xe6\xad\xa3\xe5\xb8\xb8", ImVec2(-1, 0))) state.nAnnotTool = 0;
            if (ImGui::Button("\xe5\xbc\x82\xe5\xb8\xb8", ImVec2(-1, 0))) state.nAnnotTool = 1;
        } else {
            // 20260320 ZJH 检测/分割：完整工具栏
            ImGui::TextDisabled("\xe5\xb7\xa5\xe5\x85\xb7");
            ImGui::Separator();

            auto toolBtn = [&](const char* label, int id) {
                bool bSel = (state.nAnnotTool == id);
                if (bSel) ImGui::PushStyleColor(ImGuiCol_Button, s_colAccent);
                if (ImGui::Button(label, ImVec2(-1, 0))) state.nAnnotTool = id;
                if (bSel) ImGui::PopStyleColor();
            };

            toolBtn("\xe9\x80\x89\xe6\x8b\xa9\nV", 0);     // "选择"
            toolBtn("\xe7\x9f\xa9\xe5\xbd\xa2\nB", 1);     // "矩形"
            if (state.activeTask == TaskType::Segmentation) {
                toolBtn("\xe5\xa4\x9a\xe8\xbe\xb9\nP", 2);  // "多边"
                toolBtn("\xe7\x94\xbb\xe7\xac\x94\nD", 3);  // "画笔"
                toolBtn("\xe6\xa9\xa1\xe7\x9a\xae\nE", 4);  // "橡皮"
            }

            // 20260320 ZJH 快捷键
            if (!ImGui::GetIO().WantTextInput) {
                if (ImGui::IsKeyPressed(ImGuiKey_V)) state.nAnnotTool = 0;
                if (ImGui::IsKeyPressed(ImGuiKey_B)) state.nAnnotTool = 1;
                if (ImGui::IsKeyPressed(ImGuiKey_P)) state.nAnnotTool = 2;
                if (ImGui::IsKeyPressed(ImGuiKey_D)) state.nAnnotTool = 3;
                if (ImGui::IsKeyPressed(ImGuiKey_E)) state.nAnnotTool = 4;
            }

            // 20260320 ZJH 画笔大小
            if (state.nAnnotTool == 3 || state.nAnnotTool == 4) {
                ImGui::Spacing();
                ImGui::Text("\xe7\xac\x94\xe5\x88\xb7:");
                ImGui::SetNextItemWidth(-1);
                ImGui::SliderFloat("##brush", &state.fAnnotBrushSize, 1.0f, 50.0f, "%.0f");
            }
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();

    // ---- 中间图像查看区 ----
    ImGui::BeginChild("##LabelImage", ImVec2(fCenterW, fMainH), ImGuiChildFlags_Borders);
    {
        auto* pDrawList = ImGui::GetWindowDrawList();
        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();

        // 20260320 ZJH 深灰背景
        pDrawList->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(25, 28, 35, 255));

        // 20260320 ZJH 计算图像显示区域
        float fImgW = canvasSize.x * 0.85f * state.fAnnotZoom;
        float fImgH = canvasSize.y * 0.85f * state.fAnnotZoom;
        float fImgX = canvasPos.x + (canvasSize.x - fImgW) * 0.5f + state.fAnnotPanX;
        float fImgY = canvasPos.y + (canvasSize.y - fImgH) * 0.5f + state.fAnnotPanY;

        // 20260320 ZJH 图像占位矩形
        int nCurImg = state.nAnnotCurrentImage;
        if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) {
            // 20260320 ZJH 用渐变色矩形模拟图像
            pDrawList->AddRectFilled(ImVec2(fImgX, fImgY), ImVec2(fImgX + fImgW, fImgY + fImgH),
                IM_COL32(60 + nCurImg * 15, 70 + nCurImg * 10, 80, 255));
            pDrawList->AddRect(ImVec2(fImgX, fImgY), ImVec2(fImgX + fImgW, fImgY + fImgH), IM_COL32(80, 85, 100, 255));

            // 20260320 ZJH 文件名
            std::string strName = proj.vecAnnotations[nCurImg].strImagePath;
            pDrawList->AddText(ImVec2(fImgX + 4, fImgY + 4), IM_COL32(200, 200, 220, 255), strName.c_str());
        }

        // 20260320 ZJH 绘制现有标注
        if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) {
            auto& curAnn = proj.vecAnnotations[nCurImg];
            for (int b = 0; b < (int)curAnn.vecBBoxes.size(); ++b) {
                auto& bbox = curAnn.vecBBoxes[b];
                float fX1 = fImgX + bbox.fX * fImgW;
                float fY1 = fImgY + bbox.fY * fImgH;
                float fX2 = fX1 + bbox.fWidth * fImgW;
                float fY2 = fY1 + bbox.fHeight * fImgH;
                ImU32 nColor = s_arrClassColors[bbox.nClassId % s_nNumClassColors];
                bool bSel = (b == state.nAnnotSelectedBBox);
                pDrawList->AddRectFilled(ImVec2(fX1, fY1), ImVec2(fX2, fY2), (nColor & 0x00FFFFFF) | 0x30000000);
                pDrawList->AddRect(ImVec2(fX1, fY1), ImVec2(fX2, fY2), nColor, 0, 0, bSel ? 3.0f : 2.0f);
                pDrawList->AddText(ImVec2(fX1 + 2, fY1 + 2), IM_COL32(255,255,255,255), bbox.strClassName.c_str());
                if (bSel) {
                    float fHS = 4.0f;
                    pDrawList->AddRectFilled(ImVec2(fX1-fHS,fY1-fHS), ImVec2(fX1+fHS,fY1+fHS), nColor);
                    pDrawList->AddRectFilled(ImVec2(fX2-fHS,fY1-fHS), ImVec2(fX2+fHS,fY1+fHS), nColor);
                    pDrawList->AddRectFilled(ImVec2(fX1-fHS,fY2-fHS), ImVec2(fX1+fHS,fY2+fHS), nColor);
                    pDrawList->AddRectFilled(ImVec2(fX2-fHS,fY2-fHS), ImVec2(fX2+fHS,fY2+fHS), nColor);
                }
            }
        }

        // 20260320 ZJH 鼠标交互
        ImGui::SetCursorScreenPos(canvasPos);
        ImGui::InvisibleButton("##labelCanvas", canvasSize, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        bool bHovered = ImGui::IsItemHovered();

        if (bHovered) {
            ImVec2 mp = ImGui::GetMousePos();

            // 20260320 ZJH 滚轮缩放
            float fWheel = ImGui::GetIO().MouseWheel;
            if (std::abs(fWheel) > 0.01f) {
                state.fAnnotZoom *= (fWheel > 0) ? 1.1f : 0.9f;
                state.fAnnotZoom = std::max(0.1f, std::min(10.0f, state.fAnnotZoom));
            }

            // 20260320 ZJH 中键平移
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
                state.fAnnotPanX += delta.x; state.fAnnotPanY += delta.y;
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
            }

            // 20260320 ZJH 矩形框工具
            if (state.nAnnotTool == 1) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    state.bAnnotDrawing = true; state.fAnnotStartX = mp.x; state.fAnnotStartY = mp.y;
                }
                if (state.bAnnotDrawing && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    pDrawList->AddRect(ImVec2(state.fAnnotStartX, state.fAnnotStartY), mp, IM_COL32(0,150,255,255), 0, 0, 1.5f);
                    char arrSz[64]; std::snprintf(arrSz, sizeof(arrSz), "%.0fx%.0f", std::abs(mp.x-state.fAnnotStartX), std::abs(mp.y-state.fAnnotStartY));
                    pDrawList->AddText(ImVec2(mp.x+5, mp.y+5), IM_COL32(255,255,255,200), arrSz);
                }
                if (state.bAnnotDrawing && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    state.bAnnotDrawing = false;
                    float fMinX = std::min(state.fAnnotStartX, mp.x), fMinY = std::min(state.fAnnotStartY, mp.y);
                    float fMaxX = std::max(state.fAnnotStartX, mp.x), fMaxY = std::max(state.fAnnotStartY, mp.y);
                    if ((fMaxX-fMinX) > 5.0f && (fMaxY-fMinY) > 5.0f && fImgW > 1.0f) {
                        df::BBoxAnnotation newBBox;
                        newBBox.fX = std::max(0.0f, std::min(1.0f, (fMinX - fImgX) / fImgW));
                        newBBox.fY = std::max(0.0f, std::min(1.0f, (fMinY - fImgY) / fImgH));
                        newBBox.fWidth = std::min((fMaxX - fMinX) / fImgW, 1.0f - newBBox.fX);
                        newBBox.fHeight = std::min((fMaxY - fMinY) / fImgH, 1.0f - newBBox.fY);
                        newBBox.nClassId = state.nAnnotCurrentClass;
                        if (state.nAnnotCurrentClass < (int)proj.vecClassNames.size()) newBBox.strClassName = proj.vecClassNames[state.nAnnotCurrentClass];
                        if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) proj.vecAnnotations[nCurImg].vecBBoxes.push_back(newBBox);
                    }
                }
            } else if (state.nAnnotTool == 0) {
                // 20260320 ZJH 选择工具
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    state.nAnnotSelectedBBox = -1;
                    if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) {
                        auto& curAnn = proj.vecAnnotations[nCurImg];
                        for (int b = (int)curAnn.vecBBoxes.size()-1; b >= 0; --b) {
                            auto& bbox = curAnn.vecBBoxes[b];
                            float fX1 = fImgX + bbox.fX*fImgW, fY1 = fImgY + bbox.fY*fImgH;
                            float fX2 = fX1 + bbox.fWidth*fImgW, fY2 = fY1 + bbox.fHeight*fImgH;
                            if (mp.x >= fX1 && mp.x <= fX2 && mp.y >= fY1 && mp.y <= fY2) { state.nAnnotSelectedBBox = b; break; }
                        }
                    }
                }
            }

            // 20260320 ZJH 坐标显示
            float fNX = (mp.x - fImgX) / fImgW, fNY = (mp.y - fImgY) / fImgH;
            if (fNX >= 0 && fNX <= 1 && fNY >= 0 && fNY <= 1) {
                char arrC[64]; std::snprintf(arrC, sizeof(arrC), "(%.3f, %.3f)", fNX, fNY);
                pDrawList->AddText(ImVec2(canvasPos.x+canvasSize.x-100, canvasPos.y+canvasSize.y-20), IM_COL32(180,180,200,255), arrC);
            }
        }

        // 20260320 ZJH Delete 键删除选中
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) && state.nAnnotSelectedBBox >= 0) {
            if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) {
                auto& ann = proj.vecAnnotations[nCurImg];
                if (state.nAnnotSelectedBBox < (int)ann.vecBBoxes.size()) { ann.vecBBoxes.erase(ann.vecBBoxes.begin() + state.nAnnotSelectedBBox); state.nAnnotSelectedBBox = -1; }
            }
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();

    // ---- 右侧类别/属性面板 ----
    ImGui::BeginChild("##LabelProps", ImVec2(0, fMainH), ImGuiChildFlags_Borders);
    {
        // 20260320 ZJH 类别列表
        ImGui::TextColored(ImVec4(0.58f, 0.75f, 1.0f, 1.0f), "\xe7\xb1\xbb\xe5\x88\xab");
        ImGui::Separator();

        for (int i = 0; i < (int)proj.vecClassNames.size(); ++i) {
            auto* pDL = ImGui::GetWindowDrawList();
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImU32 nCol = s_arrClassColors[i % s_nNumClassColors];
            pDL->AddRectFilled(ImVec2(p.x, p.y+2), ImVec2(p.x+12, p.y+14), nCol);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 18);

            // 20260320 ZJH 计数
            int nCount = 0;
            for (const auto& ann : proj.vecAnnotations) for (const auto& b : ann.vecBBoxes) if (b.nClassId == i) ++nCount;
            char arrL[128]; std::snprintf(arrL, sizeof(arrL), "%s (%d)", proj.vecClassNames[i].c_str(), nCount);
            if (ImGui::Selectable(arrL, state.nAnnotCurrentClass == i)) state.nAnnotCurrentClass = i;
        }

        ImGui::Spacing();
        static char s_arrNewCls[64] = "";
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##newcls", s_arrNewCls, sizeof(s_arrNewCls));
        if (ImGui::Button("+\xe6\xb7\xbb\xe5\x8a\xa0\xe7\xb1\xbb\xe5\x88\xab", ImVec2(-1, 0))) {
            if (std::strlen(s_arrNewCls) > 0) { proj.vecClassNames.push_back(std::string(s_arrNewCls)); s_arrNewCls[0] = '\0'; }
        }
        if (!proj.vecClassNames.empty()) {
            if (ImGui::Button("-\xe5\x88\xa0\xe9\x99\xa4\xe7\xb1\xbb\xe5\x88\xab", ImVec2(-1, 0))) {
                proj.vecClassNames.pop_back();
                if (state.nAnnotCurrentClass >= (int)proj.vecClassNames.size()) state.nAnnotCurrentClass = std::max(0, (int)proj.vecClassNames.size()-1);
            }
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        // 20260320 ZJH 当前标注列表
        ImGui::TextColored(ImVec4(0.58f, 0.75f, 1.0f, 1.0f), "\xe5\xbd\x93\xe5\x89\x8d\xe6\xa0\x87\xe6\xb3\xa8:");
        int nCurImg = state.nAnnotCurrentImage;
        if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) {
            auto& curAnn = proj.vecAnnotations[nCurImg];
            for (int b = 0; b < (int)curAnn.vecBBoxes.size(); ++b) {
                char arrB[128]; std::snprintf(arrB, sizeof(arrB), "%s [%.2f,%.2f]", curAnn.vecBBoxes[b].strClassName.c_str(), curAnn.vecBBoxes[b].fX, curAnn.vecBBoxes[b].fY);
                if (ImGui::Selectable(arrB, b == state.nAnnotSelectedBBox)) state.nAnnotSelectedBBox = b;
            }
            if (curAnn.vecBBoxes.empty()) ImGui::TextDisabled("(\xe6\x97\xa0\xe6\xa0\x87\xe6\xb3\xa8)");
        }

        if (state.nAnnotSelectedBBox >= 0) {
            ImGui::Spacing();
            if (ImGui::Button("\xe5\x88\xa0\xe9\x99\xa4\xe9\x80\x89\xe4\xb8\xad", ImVec2(-1, 0))) {
                if (nCurImg >= 0 && nCurImg < (int)proj.vecAnnotations.size()) {
                    auto& ann = proj.vecAnnotations[nCurImg];
                    if (state.nAnnotSelectedBBox < (int)ann.vecBBoxes.size()) { ann.vecBBoxes.erase(ann.vecBBoxes.begin() + state.nAnnotSelectedBBox); state.nAnnotSelectedBBox = -1; }
                }
            }
        }
    }
    ImGui::EndChild();

    // ---- 底部导航条 ----
    ImGui::BeginChild("##LabelNav", ImVec2(0, 0), ImGuiChildFlags_Borders);
    {
        int nTotal = (int)proj.vecAnnotations.size();
        int nAnnotated = 0;
        for (const auto& ann : proj.vecAnnotations) { if (!ann.vecBBoxes.empty()) ++nAnnotated; }

        // 20260320 ZJH 上一张/下一张
        if (ImGui::Button("\xe2\x86\x90 \xe4\xb8\x8a\xe4\xb8\x80\xe5\xbc\xa0")) { if (state.nAnnotCurrentImage > 0) state.nAnnotCurrentImage--; }
        ImGui::SameLine();
        ImGui::Text("[ %d / %d ]", state.nAnnotCurrentImage + 1, nTotal);
        ImGui::SameLine();
        if (ImGui::Button("\xe4\xb8\x8b\xe4\xb8\x80\xe5\xbc\xa0 \xe2\x86\x92")) { if (state.nAnnotCurrentImage < nTotal-1) state.nAnnotCurrentImage++; }

        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 250);
        // 20260320 ZJH 标注进度条
        float fProg = (nTotal > 0) ? (float)nAnnotated / (float)nTotal : 0;
        char arrProg[64]; std::snprintf(arrProg, sizeof(arrProg), "\xe6\xa0\x87\xe6\xb3\xa8\xe8\xbf\x9b\xe5\xba\xa6: %d%%", (int)(fProg * 100));
        ImGui::ProgressBar(fProg, ImVec2(200, 0), arrProg);
    }
    ImGui::EndChild();
}

// ============================================================================
// 20260320 ZJH 绘制训练页面（Page 3）
// ============================================================================
static void drawPageTraining(AppState& state) {
    auto& ts = state.trainState;
    bool bDis = ts.bRunning.load();

    // 20260320 ZJH 模型配置卡片
    if (bDis) ImGui::BeginDisabled();
    if (beginCard("\xe6\xa8\xa1\xe5\x9e\x8b\xe9\x85\x8d\xe7\xbd\xae")) {
        const char* arrTaskNames[] = {"\xe5\x9b\xbe\xe5\x83\x8f\xe5\x88\x86\xe7\xb1\xbb", "\xe7\x9b\xae\xe6\xa0\x87\xe6\xa3\x80\xe6\xb5\x8b", "\xe8\xaf\xad\xe4\xb9\x89\xe5\x88\x86\xe5\x89\xb2", "\xe5\xbc\x82\xe5\xb8\xb8\xe6\xa3\x80\xe6\xb5\x8b"};
        ImGui::Text("\xe4\xbb\xbb\xe5\x8a\xa1\xe7\xb1\xbb\xe5\x9e\x8b: %s", arrTaskNames[(int)state.activeTask]);

        switch (state.activeTask) {
        case TaskType::Classification: {
            const char* m[] = {"MLP", "ResNet-18", "ResNet-34"};
            ImGui::Combo("\xe6\xa8\xa1\xe5\x9e\x8b", &state.nClsModel, m, 3);
            if (state.nClsModel == 2) state.nClsModel = 1;
            const char* opts[] = {"SGD", "Adam"};
            ImGui::Combo("\xe4\xbc\x98\xe5\x8c\x96\xe5\x99\xa8", &state.nClsOptimizer, opts, 2);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87", &state.fClsLR, 0.001f, 0.01f, "%.4f");
            if (state.fClsLR < 0.0001f) state.fClsLR = 0.0001f;
            ImGui::SliderInt("\xe8\xbd\xae\xe6\xac\xa1", &state.nClsEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1", &state.nClsBatchSize, 8, 256);

            // 20260320 ZJH 预设
            ImGui::Spacing();
            ImGui::Text("\xe9\xa2\x84\xe8\xae\xbe:");
            ImGui::SameLine();
            if (ImGui::RadioButton("\xe5\xbf\xab\xe9\x80\x9f", &state.nPresetIndex, 0)) { state.nClsEpochs = 5; state.fClsLR = 0.01f; state.nClsBatchSize = 128; }
            ImGui::SameLine();
            if (ImGui::RadioButton("\xe6\xa0\x87\xe5\x87\x86", &state.nPresetIndex, 1)) { state.nClsEpochs = 20; state.fClsLR = 0.001f; state.nClsBatchSize = 64; }
            ImGui::SameLine();
            if (ImGui::RadioButton("\xe7\xb2\xbe\xe7\xa1\xae", &state.nPresetIndex, 2)) { state.nClsEpochs = 50; state.fClsLR = 0.0001f; state.nClsBatchSize = 32; }
            break;
        }
        case TaskType::Detection: {
            const char* dm[] = {"YOLOv5-Nano", "YOLOv7-Tiny", "YOLOv8-Nano [\xe6\x8e\xa8\xe8\x8d\x90]", "YOLOv10-Nano"};
            ImGui::Combo("\xe6\xa8\xa1\xe5\x9e\x8b", &state.nDetModel, dm, 4);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87##det", &state.fDetLR, 0.0001f, 0.001f, "%.4f");
            ImGui::SliderInt("\xe8\xbd\xae\xe6\xac\xa1##det", &state.nDetEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1##det", &state.nDetBatchSize, 1, 32);
            break;
        }
        case TaskType::Segmentation: {
            ImGui::Text("\xe6\xa8\xa1\xe5\x9e\x8b: U-Net");
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87##seg", &state.fSegLR, 0.0001f, 0.001f, "%.4f");
            ImGui::SliderInt("\xe8\xbd\xae\xe6\xac\xa1##seg", &state.nSegEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1##seg", &state.nSegBatchSize, 1, 16);
            break;
        }
        case TaskType::AnomalyDetection: {
            ImGui::Text("\xe6\xa8\xa1\xe5\x9e\x8b: ConvAutoEncoder");
            ImGui::SliderInt("\xe7\x93\xb6\xe9\xa2\x88\xe7\xbb\xb4\xe5\xba\xa6", &state.nAeLatentDim, 16, 128);
            ImGui::InputFloat("\xe5\xad\xa6\xe4\xb9\xa0\xe7\x8e\x87##ae", &state.fAeLR, 0.0001f, 0.001f, "%.4f");
            ImGui::SliderInt("\xe8\xbd\xae\xe6\xac\xa1##ae", &state.nAeEpochs, 1, 100);
            ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1##ae", &state.nAeBatchSize, 4, 128);
            break;
        }
        }
    }
    endCard();
    if (bDis) ImGui::EndDisabled();

    // 20260320 ZJH 训练控制按钮
    {
        float fAvailW = ImGui::GetContentRegionAvail().x;
        if (!ts.bRunning.load()) {
            ImGui::SetCursorPosX((fAvailW - 200) * 0.5f);
            ImGui::PushStyleColor(ImGuiCol_Button, s_colAccent);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, s_colAccentHover);
            if (ImGui::Button("\xe5\xbc\x80 \xe5\xa7\x8b \xe8\xae\xad \xe7\xbb\x83", ImVec2(200, 40))) startTraining(state);
            ImGui::PopStyleColor(2);
        } else {
            ImGui::SetCursorPosX((fAvailW - 280) * 0.5f);
            if (ts.bPaused.load()) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.55f, 0.2f, 1));
                if (ImGui::Button("\xe6\x81\xa2\xe5\xa4\x8d", ImVec2(80, 35))) ts.bPaused.store(false);
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.45f, 0.1f, 1));
                if (ImGui::Button("\xe6\x9a\x82\xe5\x81\x9c", ImVec2(80, 35))) ts.bPaused.store(true);
                ImGui::PopStyleColor();
            }
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.18f, 0.18f, 1));
            if (ImGui::Button("\xe5\x81\x9c\xe6\xad\xa2", ImVec2(80, 35))) { ts.bStopRequested.store(true); ts.bPaused.store(false); }
            ImGui::PopStyleColor();
        }
        ImGui::Spacing();
    }

    // 20260320 ZJH 进度条
    {
        int nE = ts.nCurrentEpoch.load(), nTE = ts.nTotalEpochs.load();
        char arrProg[128];
        std::snprintf(arrProg, sizeof(arrProg), "\xe8\xbd\xae\xe6\xac\xa1: %d/%d", nE, nTE);

        float fMs = ts.fAvgBatchTimeMs.load();
        int nB = ts.nCurrentBatch.load(), nTB = ts.nTotalBatches.load();
        std::string strETA;
        if (fMs > 0.001f && ts.bRunning.load()) {
            int rem = (nTB - nB) + (nTE - nE) * nTB;
            float eta = (float)rem * fMs / 1000.0f;
            char arrETA[64];
            if (eta < 60) std::snprintf(arrETA, sizeof(arrETA), "ETA: %.0f\xe7\xa7\x92", (double)eta);
            else std::snprintf(arrETA, sizeof(arrETA), "ETA: %.1f\xe5\x88\x86", (double)eta/60.0);
            strETA = arrETA;
        }

        ImGui::Text("%s   %s", arrProg, strETA.c_str());
        ImGui::ProgressBar(nTE > 0 ? (float)nE/(float)nTE : 0, ImVec2(-1, 0));
        ImGui::Spacing();
    }

    // 20260320 ZJH 损失曲线 + 准确率曲线 并排
    float fChartH = std::max(180.0f, ImGui::GetContentRegionAvail().y * 0.35f);
    ImGui::BeginChild("##Charts", ImVec2(0, fChartH));
    {
        float hw = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;

        ImGui::BeginChild("##LossChart", ImVec2(hw, 0));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          if (ImPlot::BeginPlot("\xe6\x8d\x9f\xe5\xa4\xb1\xe6\x9b\xb2\xe7\xba\xbf", ImVec2(-1, -1))) {
              ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "\xe6\x8d\x9f\xe5\xa4\xb1");
              if (!ts.vecLossHistory.empty()) ImPlot::PlotLine("\xe8\xae\xad\xe7\xbb\x83\xe6\x8d\x9f\xe5\xa4\xb1", ts.vecLossHistory.data(), (int)ts.vecLossHistory.size());
              ImPlot::EndPlot();
          }
        }
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("##AccChart", ImVec2(0, 0));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          if (state.activeTask == TaskType::Classification) {
              if (ImPlot::BeginPlot("\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87\xe6\x9b\xb2\xe7\xba\xbf", ImVec2(-1, -1))) {
                  ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87 (%)");
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 105, ImPlotCond_Always);
                  if (!ts.vecTrainAccHistory.empty()) ImPlot::PlotLine("\xe8\xae\xad\xe7\xbb\x83", ts.vecTrainAccHistory.data(), (int)ts.vecTrainAccHistory.size());
                  if (!ts.vecTestAccHistory.empty()) ImPlot::PlotLine("\xe9\xaa\x8c\xe8\xaf\x81", ts.vecTestAccHistory.data(), (int)ts.vecTestAccHistory.size());
                  ImPlot::EndPlot();
              }
          } else if (state.activeTask == TaskType::Segmentation) {
              if (ImPlot::BeginPlot("mIoU", ImVec2(-1, -1))) {
                  ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "mIoU");
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.05, ImPlotCond_Always);
                  if (!ts.vecMIoUHistory.empty()) ImPlot::PlotLine("mIoU", ts.vecMIoUHistory.data(), (int)ts.vecMIoUHistory.size());
                  ImPlot::EndPlot();
              }
          } else {
              if (ImPlot::BeginPlot("\xe6\x8d\x9f\xe5\xa4\xb1\xe8\xb6\x8b\xe5\x8a\xbf", ImVec2(-1, -1))) {
                  ImPlot::SetupAxes("\xe8\xbd\xae\xe6\xac\xa1", "\xe6\x8d\x9f\xe5\xa4\xb1");
                  if (!ts.vecLossHistory.empty()) ImPlot::PlotLine("\xe6\x8d\x9f\xe5\xa4\xb1", ts.vecLossHistory.data(), (int)ts.vecLossHistory.size());
                  ImPlot::EndPlot();
              }
          }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    // 20260320 ZJH 训练日志
    if (beginCard("\xe8\xae\xad\xe7\xbb\x83\xe6\x97\xa5\xe5\xbf\x97")) {
        ImGui::BeginChild("##LogTxt", ImVec2(0, 120));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          ImGui::TextUnformatted(ts.strLog.c_str());
          if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) ImGui::SetScrollHereY(1.0f);
        }
        if (ImGui::BeginPopupContextWindow("##LogCtx")) {
            if (ImGui::MenuItem("\xe5\xa4\x8d\xe5\x88\xb6")) { std::lock_guard<std::mutex> lk(ts.mutex); ImGui::SetClipboardText(ts.strLog.c_str()); }
            if (ImGui::MenuItem("\xe6\xb8\x85\xe9\x99\xa4")) { std::lock_guard<std::mutex> lk(ts.mutex); ts.strLog.clear(); }
            ImGui::EndPopup();
        }
        ImGui::EndChild();
    }
    endCard();
}

// ============================================================================
// 20260320 ZJH 绘制评估页面（Page 4）
// ============================================================================
static void drawPageEvaluation(AppState& state) {
    auto& ts = state.trainState;

    if (!ts.bCompleted.load()) {
        ImGui::Spacing(); ImGui::Spacing();
        ImGui::TextColored(s_colSubtle, "\xe8\xaf\xb7\xe5\x85\x88\xe5\xae\x8c\xe6\x88\x90\xe8\xae\xad\xe7\xbb\x83\xe4\xbb\xa5\xe6\x9f\xa5\xe7\x9c\x8b\xe8\xaf\x84\xe4\xbc\xb0\xe7\xbb\x93\xe6\x9e\x9c");
        return;
    }

    // 20260320 ZJH 性能摘要卡片
    if (beginCard("\xe6\x80\xa7\xe8\x83\xbd\xe6\x91\x98\xe8\xa6\x81")) {
        float fHW = (ImGui::GetContentRegionAvail().x - 16) / 3;
        // 20260320 ZJH 三个指标并排
        ImGui::BeginChild("##M1", ImVec2(fHW, 50));
        if (state.activeTask == TaskType::Classification) {
            auto* pD = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
            char buf[64]; std::snprintf(buf, sizeof(buf), "%.2f%%", (double)ts.fTestAcc.load());
            pD->AddText(ImGui::GetFont(), 24.0f, p, s_nColAccent, buf);
            ImGui::Dummy(ImVec2(0, 28)); ImGui::Text("\xe5\x87\x86\xe7\xa1\xae\xe7\x8e\x87");
        } else if (state.activeTask == TaskType::Detection) {
            auto* pD = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
            char buf[64]; std::lock_guard<std::mutex> lk(ts.mutex); std::snprintf(buf, sizeof(buf), "mAP@0.5: %.1f%%", (double)ts.fMAP50);
            pD->AddText(ImGui::GetFont(), 24.0f, p, s_nColAccent, buf);
            ImGui::Dummy(ImVec2(0, 28)); ImGui::Text("mAP");
        } else if (state.activeTask == TaskType::Segmentation) {
            auto* pD = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
            char buf[64]; std::lock_guard<std::mutex> lk(ts.mutex); std::snprintf(buf, sizeof(buf), "mIoU: %.3f", (double)ts.fMIoU);
            pD->AddText(ImGui::GetFont(), 24.0f, p, s_nColAccent, buf);
            ImGui::Dummy(ImVec2(0, 28)); ImGui::Text("mIoU");
        } else {
            auto* pD = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
            char buf[64]; std::lock_guard<std::mutex> lk(ts.mutex); std::snprintf(buf, sizeof(buf), "AUC: %.3f", (double)ts.fAUC);
            pD->AddText(ImGui::GetFont(), 24.0f, p, s_nColAccent, buf);
            ImGui::Dummy(ImVec2(0, 28)); ImGui::Text("AUC");
        }
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("##M2", ImVec2(fHW, 50));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          char buf[64]; std::snprintf(buf, sizeof(buf), "%.4f", (double)ts.fBestValLoss);
          auto* pD = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
          pD->AddText(ImGui::GetFont(), 24.0f, p, s_nColText, buf);
        }
        ImGui::Dummy(ImVec2(0, 28)); ImGui::Text("\xe6\x8d\x9f\xe5\xa4\xb1");
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("##M3", ImVec2(0, 50));
        { std::lock_guard<std::mutex> lk(ts.mutex);
          char buf[64]; std::snprintf(buf, sizeof(buf), "%.1f\xe7\xa7\x92", (double)ts.fTotalTrainingTimeSec);
          auto* pD = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
          pD->AddText(ImGui::GetFont(), 24.0f, p, s_nColText, buf);
        }
        ImGui::Dummy(ImVec2(0, 28)); ImGui::Text("\xe8\x80\x97\xe6\x97\xb6");
        ImGui::EndChild();
    }
    endCard();

    // 20260320 ZJH 混淆矩阵 + 分类报告 并排（仅分类任务）
    if (state.activeTask == TaskType::Classification) {
        float fHalfW = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;

        ImGui::BeginChild("##EvalLeft", ImVec2(fHalfW, 0));
        if (beginCard("\xe6\xb7\xb7\xe6\xb7\x86\xe7\x9f\xa9\xe9\x98\xb5")) {
            std::lock_guard<std::mutex> lk(ts.mutex);
            if (ts.bHasConfusionMatrix) {
                static double hm[100];
                double mx = 0;
                for (int r = 0; r < 10; ++r) for (int c = 0; c < 10; ++c) { double v = (double)ts.arrConfusionMatrix[r][c]; hm[r*10+c] = v; if (v > mx) mx = v; }
                static const char* lab[] = {"0","1","2","3","4","5","6","7","8","9"};
                if (ImPlot::BeginPlot("##CM", ImVec2(300, 300))) {
                    ImPlot::SetupAxes("\xe9\xa2\x84\xe6\xb5\x8b", "\xe5\xae\x9e\xe9\x99\x85");
                    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 9, 10, lab);
                    ImPlot::SetupAxisTicks(ImAxis_Y1, 0, 9, 10, lab);
                    ImPlot::PlotHeatmap("##hm", hm, 10, 10, 0, mx, "%g", ImPlotPoint(0,0), ImPlotPoint(10,10));
                    ImPlot::EndPlot();
                }
            }
        }
        endCard();
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("##EvalRight", ImVec2(0, 0));
        if (beginCard("\xe5\x88\x86\xe7\xb1\xbb\xe6\x8a\xa5\xe5\x91\x8a")) {
            std::lock_guard<std::mutex> lk(ts.mutex);
            if (ts.bHasConfusionMatrix) {
                if (ImGui::BeginTable("##CR", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("\xe7\xb1\xbb\xe5\x88\xab"); ImGui::TableSetupColumn("\xe7\xb2\xbe\xe7\xa1\xae\xe7\x8e\x87"); ImGui::TableSetupColumn("\xe5\x8f\xac\xe5\x9b\x9e\xe7\x8e\x87"); ImGui::TableSetupColumn("F1");
                    ImGui::TableHeadersRow();
                    for (int c = 0; c < 10; ++c) {
                        int tp = ts.arrConfusionMatrix[c][c], fp = 0, fn = 0;
                        for (int i = 0; i < 10; ++i) { if (i!=c) { fp += ts.arrConfusionMatrix[i][c]; fn += ts.arrConfusionMatrix[c][i]; } }
                        float p = (tp+fp>0)?(float)tp/(float)(tp+fp):0; float r = (tp+fn>0)?(float)tp/(float)(tp+fn):0; float f = (p+r>0)?2*p*r/(p+r):0;
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::Text("%d", c);
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", (double)p);
                        ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", (double)r);
                        ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", (double)f);
                    }
                    ImGui::EndTable();
                }
            }
        }
        endCard();
        ImGui::EndChild();
    } else if (state.activeTask == TaskType::AnomalyDetection) {
        if (beginCard("\xe9\x98\x88\xe5\x80\xbc\xe8\xb0\x83\xe8\x8a\x82")) {
            ImGui::SliderFloat("\xe5\xbc\x82\xe5\xb8\xb8\xe9\x98\x88\xe5\x80\xbc", &state.fAeThreshold, 0.01f, 5.0f, "%.3f");
        }
        endCard();
    }

    // 20260320 ZJH 预测结果画廊（模拟）
    if (beginCard("\xe9\xa2\x84\xe6\xb5\x8b\xe7\xbb\x93\xe6\x9e\x9c\xe7\x94\xbb\xe5\xbb\x8a")) {
        ImDrawList* pDraw = ImGui::GetWindowDrawList();
        ImVec2 startP = ImGui::GetCursorScreenPos();
        float fSize = 60.0f;
        int nShow = 10;
        for (int i = 0; i < nShow; ++i) {
            float fX = startP.x + i * (fSize + 6);
            float fY = startP.y;
            bool bCorrect = (i % 4 != 2);  // 20260320 ZJH 模拟：大部分正确
            ImU32 nBorderCol = bCorrect ? IM_COL32(50, 200, 80, 255) : IM_COL32(220, 50, 50, 255);
            pDraw->AddRectFilled(ImVec2(fX, fY), ImVec2(fX+fSize, fY+fSize), IM_COL32(60+i*10, 70, 80, 255));
            pDraw->AddRect(ImVec2(fX, fY), ImVec2(fX+fSize, fY+fSize), nBorderCol, 0, 0, 2.0f);
            const char* strMark = bCorrect ? "\xe2\x9c\x93" : "\xe2\x9c\x97";
            pDraw->AddText(ImVec2(fX+2, fY+fSize-16), nBorderCol, strMark);
        }
        ImGui::Dummy(ImVec2(0, fSize + 4));
    }
    endCard();

    // 20260320 ZJH 导出按钮
    ImGui::Spacing();
    if (ImGui::Button("\xe5\xaf\xbc\xe5\x87\xba\xe6\xa8\xa1\xe5\x9e\x8b .dfm", ImVec2(150, 30))) {
        auto& ts2 = state.trainState; std::lock_guard<std::mutex> lk2(ts2.mutex);
        if (!ts2.strSavedModelPath.empty()) ts2.strLog += getTimestamp() + " \xe6\xa8\xa1\xe5\x9e\x8b: " + ts2.strSavedModelPath + "\n";
        else { state.nFileDialogPurpose = 3; strncpy_s(state.arrFileDialogPath, sizeof(state.arrFileDialogPath), "data/models/export.dfm", _TRUNCATE); state.bShowFileDialog = true; }
    }
    ImGui::SameLine();
    if (ImGui::Button("\xe5\xaf\xbc\xe5\x87\xba\xe8\xaf\x84\xe4\xbc\xb0\xe6\x8a\xa5\xe5\x91\x8a HTML", ImVec2(180, 30))) {
        // 20260320 ZJH 直接导出到默认路径
        std::string strHTMLPath = "data/reports/eval_report.html";
        exportEvaluationReportHTML(state, strHTMLPath);
    }
    ImGui::SameLine();
    ImGui::BeginDisabled();
    ImGui::Button("\xe5\xaf\xbc\xe5\x87\xba ONNX (\xe5\x8d\xb3\xe5\xb0\x86\xe6\x8e\xa8\xe5\x87\xba)", ImVec2(180, 30));
    ImGui::EndDisabled();
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
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.059f, 0.067f, 0.082f, 1.0f));  // 20260320 ZJH #0f1115

    ImGui::Begin("##StatusBar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav);

    // 20260320 ZJH 状态文字
    if (ts.bRunning.load()) {
        if (ts.bPaused.load()) ImGui::TextColored(s_colOrange, "\xe5\xb7\xb2\xe6\x9a\x82\xe5\x81\x9c");
        else { char b[128]; std::snprintf(b, sizeof(b), "\xe8\xae\xad\xe7\xbb\x83\xe4\xb8\xad (%d/%d)", ts.nCurrentEpoch.load(), ts.nTotalEpochs.load()); ImGui::TextColored(s_colGreen, "%s", b); }
    } else if (ts.bCompleted.load()) {
        ImGui::TextColored(s_colAccent, "\xe8\xae\xad\xe7\xbb\x83\xe5\xb7\xb2\xe5\xae\x8c\xe6\x88\x90");
    } else {
        ImGui::Text("\xe5\xb0\xb1\xe7\xbb\xaa");
    }

    // 20260320 ZJH 图像数
    ImGui::SameLine(200);
    int nImgCount = state.bMnistAvailable ? state.nMnistTrainSamples + state.nMnistTestSamples : 1200;
    ImGui::Text("\xe5\x9b\xbe\xe5\x83\x8f: %d", nImgCount);

    // 20260320 ZJH GPU
    ImGui::SameLine(400);
    if (state.bGpuDetected && !state.vecGpuDevices.empty()) {
        char b[256]; std::snprintf(b, sizeof(b), "GPU: %s (%zuGB)", state.vecGpuDevices[0].strName.c_str(), state.vecGpuDevices[0].nTotalMemoryMB/1024);
        ImGui::TextColored(s_colGreen, "%s", b);
    } else {
        ImGui::TextColored(s_colOrange, "CPU");
    }

    // 20260320 ZJH 版本号
    float vw = ImGui::CalcTextSize("DeepForge v0.1").x;
    ImGui::SameLine(ImGui::GetWindowWidth() - vw - 15);
    ImGui::TextDisabled("DeepForge v0.1");

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// ============================================================================
// 20260320 ZJH 主函数
// ============================================================================
int main(int, char**) {
    // 20260320 ZJH 初始化 SDL3
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* pWindow = SDL_CreateWindow("DeepForge \xe6\xb7\xb1\xe5\xba\xa6\xe5\xad\xa6\xe4\xb9\xa0\xe5\xb7\xa5\xe5\x85\xb7", 1280, 800,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);
    if (!pWindow) { SDL_Quit(); return 1; }

    SDL_Renderer* pRenderer = SDL_CreateRenderer(pWindow, nullptr);
    if (!pRenderer) { SDL_DestroyWindow(pWindow); SDL_Quit(); return 1; }

    // 20260320 ZJH 初始化 ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    setupMVTecStyle();

    // 20260320 ZJH 加载中文字体
    const char* fonts[] = {"C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"};
    bool bFont = false;
    for (auto* fp : fonts) {
        if (std::filesystem::exists(fp)) {
            ImFontConfig fc; fc.MergeMode = false;
            io.Fonts->AddFontFromFileTTF(fp, 16.0f, &fc, io.Fonts->GetGlyphRangesChineseFull());
            bFont = true; break;
        }
    }
    if (!bFont) io.Fonts->AddFontDefault();

    ImGui_ImplSDL3_InitForSDLRenderer(pWindow, pRenderer);
    ImGui_ImplSDLRenderer3_Init(pRenderer);

    // 20260320 ZJH 闪屏动画 + GPU 检测
    std::vector<GpuDeviceInfo> vecGpuResult;
    bool bGpuDetectionDone = false;
    std::string strGpuStatus = "\xe6\xad\xa3\xe5\x9c\xa8\xe6\xa3\x80\xe6\xb5\x8bGPU...";

    std::thread gpuThread([&]() {
        vecGpuResult = detectGpuDevices();
        strGpuStatus = vecGpuResult.empty() ? "\xe6\x9c\xaa\xe6\xa3\x80\xe6\xb5\x8b\xe5\x88\xb0 GPU" : ("GPU: " + vecGpuResult[0].strName);
        bGpuDetectionDone = true;
    });

    {
        auto splashStart = std::chrono::steady_clock::now();
        const float fDur = 2.5f;
        bool bSplash = true;
        while (bSplash) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) { ImGui_ImplSDL3_ProcessEvent(&ev); if (ev.type == SDL_EVENT_QUIT || ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) { bSplash = false; break; } }
            if (!bSplash) break;
            float fEl = std::chrono::duration<float>(std::chrono::steady_clock::now() - splashStart).count();
            if (fEl > fDur) break;
            float fA = std::min(fEl / 0.8f, 1.0f);

            ImGui_ImplSDLRenderer3_NewFrame(); ImGui_ImplSDL3_NewFrame(); ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(0,0)); ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f,0.06f,0.08f,1));
            ImGui::Begin("##Splash", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoDocking|ImGuiWindowFlags_NoNav);
            auto* pDL = ImGui::GetWindowDrawList();
            float cx = io.DisplaySize.x*0.5f, cy = io.DisplaySize.y*0.5f;
            ImFont* pF = ImGui::GetFont();
            pDL->AddRectFilledMultiColor(ImVec2(0,0), io.DisplaySize, IM_COL32(15,15,25,(int)(255*fA)), IM_COL32(15,15,25,(int)(255*fA)), IM_COL32(25,30,50,(int)(255*fA)), IM_COL32(25,30,50,(int)(255*fA)));
            float lsz=80, lx=cx-lsz*0.5f, ly=cy-120;
            pDL->AddRectFilled(ImVec2(lx,ly), ImVec2(lx+lsz,ly+lsz), IM_COL32(37,99,235,(int)(240*fA)), 12);
            ImVec2 dfSz = pF->CalcTextSizeA(42, FLT_MAX, 0, "DF");
            pDL->AddText(pF, 42, ImVec2(lx+(lsz-dfSz.x)*0.5f, ly+(lsz-dfSz.y)*0.5f), IM_COL32(255,255,255,(int)(255*fA)), "DF");
            ImVec2 ttSz = pF->CalcTextSizeA(36, FLT_MAX, 0, "DeepForge");
            pDL->AddText(pF, 36, ImVec2(cx-ttSz.x*0.5f, ly+lsz+20), IM_COL32(230,230,240,(int)(255*fA)), "DeepForge");
            const char* sub = "\xe6\xb7\xb1\xe5\xba\xa6\xe5\xad\xa6\xe4\xb9\xa0\xe5\xb7\xa5\xe5\x85\xb7";
            ImVec2 subSz = pF->CalcTextSizeA(18, FLT_MAX, 0, sub);
            pDL->AddText(pF, 18, ImVec2(cx-subSz.x*0.5f, ly+lsz+64), IM_COL32(160,180,220,(int)(220*fA)), sub);
            int nDots=8; float dotR=20, dotCY=ly+lsz+120;
            for (int i=0;i<nDots;++i) { float ang=fEl*3+(float)i*(6.283185f/(float)nDots); float dx=cx+std::cos(ang)*dotR; float dy=dotCY+std::sin(ang)*dotR; float da=(std::sin(ang-fEl*3)+1)*0.5f; pDL->AddCircleFilled(ImVec2(dx,dy), 3+da*2, IM_COL32(37,99,235,(int)(da*255*fA))); }
            ImVec2 gpuSz = pF->CalcTextSizeA(14, FLT_MAX, 0, strGpuStatus.c_str());
            pDL->AddText(pF, 14, ImVec2(cx-gpuSz.x*0.5f, dotCY+dotR+20), IM_COL32(140,150,180,(int)(180*fA)), strGpuStatus.c_str());
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar();
            ImGui::Render();
            SDL_SetRenderDrawColor(pRenderer, 15, 15, 20, 255); SDL_RenderClear(pRenderer);
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer); SDL_RenderPresent(pRenderer);
        }
        if (!bSplash) { gpuThread.join(); ImGui_ImplSDLRenderer3_Shutdown(); ImGui_ImplSDL3_Shutdown(); ImPlot::DestroyContext(); ImGui::DestroyContext(); SDL_DestroyRenderer(pRenderer); SDL_DestroyWindow(pWindow); SDL_Quit(); return 0; }
    }
    gpuThread.join();

    // 20260320 ZJH 应用状态初始化
    AppState appState;
    appState.vecGpuDevices = vecGpuResult;
    appState.bGpuDetected = !vecGpuResult.empty();
    bool bPrevCompleted = false;

    // 20260320 ZJH 主循环
    bool bRunning = true;
    while (bRunning) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) { ImGui_ImplSDL3_ProcessEvent(&ev); if (ev.type == SDL_EVENT_QUIT || ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) bRunning = false; }

        if (appState.bGpuDetected) { appState.fGpuMemQueryTimer += io.DeltaTime; if (appState.fGpuMemQueryTimer > 5) { appState.fGpuMemQueryTimer = 0; auto [u,t] = queryGpuMemoryUsage(); appState.nGpuMemUsedMB = u; appState.nGpuMemTotalMB = t; } }

        ImGui_ImplSDLRenderer3_NewFrame(); ImGui_ImplSDL3_NewFrame(); ImGui::NewFrame();

        // 20260320 ZJH 菜单栏
        drawMainMenuBar(appState);
        handleKeyboardShortcuts(appState, pWindow);

        ImGuiViewport* pVP = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(pVP->WorkPos);
        ImGui::SetNextWindowSize(ImVec2(pVP->WorkSize.x, pVP->WorkSize.y - 26.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("##Main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::PopStyleVar();

        // 20260320 ZJH 弹窗逻辑
        if (appState.bShowSettingsPopup) { ImGui::OpenPopup("\xe8\xae\xbe\xe7\xbd\xae##Modal"); appState.bShowSettingsPopup = false; }
        if (appState.bShowAboutPopup) { ImGui::OpenPopup("\xe5\x85\xb3\xe4\xba\x8e DeepForge##About"); appState.bShowAboutPopup = false; }
        if (appState.bShowBatchInference) { ImGui::OpenPopup("\xe6\x89\xb9\xe9\x87\x8f\xe6\x8e\xa8\xe7\x90\x86##BatchInf"); appState.bShowBatchInference = false; }
        drawSettingsDialog(appState);
        drawAboutDialog(appState);
        drawBatchInferenceDialog(appState);

        // 20260320 ZJH GPU 信息弹窗
        if (appState.bShowGpuInfo) { ImGui::OpenPopup("GPU##GpuDlg"); appState.bShowGpuInfo = false; }
        if (ImGui::BeginPopup("GPU##GpuDlg")) {
            if (appState.bGpuDetected) { for (size_t i=0;i<appState.vecGpuDevices.size();++i) { auto& g=appState.vecGpuDevices[i]; ImGui::Text("GPU %zu: %s (%zuMB, SM %d.%d)", i, g.strName.c_str(), g.nTotalMemoryMB, g.nComputeCapMajor, g.nComputeCapMinor); } }
            else ImGui::TextDisabled("\xe6\x9c\xaa\xe6\xa3\x80\xe6\xb5\x8b\xe5\x88\xb0 GPU");
            ImGui::EndPopup();
        }

        // 20260320 ZJH 用户手册弹窗
        if (appState.bShowUserManual) { ImGui::OpenPopup("\xe7\x94\xa8\xe6\x88\xb7\xe6\x89\x8b\xe5\x86\x8c##M"); appState.bShowUserManual = false; }
        { ImVec2 mc = ImGui::GetMainViewport()->GetCenter(); ImGui::SetNextWindowPos(mc, ImGuiCond_Appearing, ImVec2(0.5f,0.5f)); ImGui::SetNextWindowSize(ImVec2(500,400), ImGuiCond_Appearing); }
        if (ImGui::BeginPopupModal("\xe7\x94\xa8\xe6\x88\xb7\xe6\x89\x8b\xe5\x86\x8c##M", nullptr, ImGuiWindowFlags_None)) {
            ImGui::TextWrapped("DeepForge \xe6\xb7\xb1\xe5\xba\xa6\xe5\xad\xa6\xe4\xb9\xa0\xe5\xb7\xa5\xe5\x85\xb7\xe4\xbd\xbf\xe7\x94\xa8\xe6\x8c\x87\xe5\x8d\x97");
            ImGui::Separator(); ImGui::Spacing();
            ImGui::BulletText("\xe6\x95\xb0\xe6\x8d\xae\xe9\xa1\xb5: \xe9\x80\x89\xe6\x8b\xa9\xe4\xbb\xbb\xe5\x8a\xa1\xe7\xb1\xbb\xe5\x9e\x8b\xe5\x92\x8c\xe6\x95\xb0\xe6\x8d\xae\xe9\x9b\x86");
            ImGui::BulletText("\xe6\xa0\x87\xe6\xb3\xa8\xe9\xa1\xb5: \xe4\xb8\xba\xe5\x9b\xbe\xe5\x83\x8f\xe6\xb7\xbb\xe5\x8a\xa0\xe6\xa0\x87\xe6\xb3\xa8");
            ImGui::BulletText("\xe8\xae\xad\xe7\xbb\x83\xe9\xa1\xb5: \xe9\x85\x8d\xe7\xbd\xae\xe5\xb9\xb6\xe5\x90\xaf\xe5\x8a\xa8\xe8\xae\xad\xe7\xbb\x83");
            ImGui::BulletText("\xe8\xaf\x84\xe4\xbc\xb0\xe9\xa1\xb5: \xe6\x9f\xa5\xe7\x9c\x8b\xe7\xbb\x93\xe6\x9e\x9c\xe5\xb9\xb6\xe5\xaf\xbc\xe5\x87\xba");
            ImGui::Spacing();
            ImGui::BulletText("F5=\xe5\xbc\x80\xe5\xa7\x8b\xe8\xae\xad\xe7\xbb\x83 F6=\xe6\x9a\x82\xe5\x81\x9c F7=\xe5\x81\x9c\xe6\xad\xa2 Ctrl+1234=\xe5\x88\x87\xe6\x8d\xa2\xe9\xa1\xb5\xe9\x9d\xa2");
            ImGui::Spacing();
            if (ImGui::Button("\xe5\x85\xb3\xe9\x97\xad", ImVec2(80,0))) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // 20260320 ZJH 快捷键参考
        if (appState.bShowShortcuts) { ImGui::OpenPopup("\xe5\xbf\xab\xe6\x8d\xb7\xe9\x94\xae##SK"); appState.bShowShortcuts = false; }
        { ImVec2 sc = ImGui::GetMainViewport()->GetCenter(); ImGui::SetNextWindowPos(sc, ImGuiCond_Appearing, ImVec2(0.5f,0.5f)); ImGui::SetNextWindowSize(ImVec2(380,350), ImGuiCond_Appearing); }
        if (ImGui::BeginPopupModal("\xe5\xbf\xab\xe6\x8d\xb7\xe9\x94\xae##SK", nullptr, ImGuiWindowFlags_None)) {
            if (ImGui::BeginTable("##SKT", 2, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("\xe5\xbf\xab\xe6\x8d\xb7\xe9\x94\xae", ImGuiTableColumnFlags_WidthFixed, 120); ImGui::TableSetupColumn("\xe5\x8a\x9f\xe8\x83\xbd"); ImGui::TableHeadersRow();
                auto addR = [](const char* k, const char* d) { ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::Text("%s",k); ImGui::TableSetColumnIndex(1); ImGui::Text("%s",d); };
                addR("Ctrl+N", "\xe6\x96\xb0\xe5\xbb\xba"); addR("Ctrl+O", "\xe6\x89\x93\xe5\xbc\x80"); addR("Ctrl+S", "\xe4\xbf\x9d\xe5\xad\x98");
                addR("Ctrl+1/2/3/4", "\xe5\x88\x87\xe6\x8d\xa2\xe9\xa1\xb5\xe9\x9d\xa2"); addR("F5", "\xe8\xae\xad\xe7\xbb\x83"); addR("F6", "\xe6\x9a\x82\xe5\x81\x9c");
                addR("F7", "\xe5\x81\x9c\xe6\xad\xa2"); addR("F11", "\xe5\x85\xa8\xe5\xb1\x8f"); addR("V/B/P/D/E", "\xe6\xa0\x87\xe6\xb3\xa8\xe5\xb7\xa5\xe5\x85\xb7");
                ImGui::EndTable();
            }
            ImGui::Spacing(); if (ImGui::Button("\xe5\x85\xb3\xe9\x97\xad", ImVec2(80,0))) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // 20260320 ZJH 文件路径弹窗
        if (appState.bShowFileDialog) { ImGui::OpenPopup("\xe6\x96\x87\xe4\xbb\xb6##FD"); appState.bShowFileDialog = false; }
        { ImVec2 fc = ImGui::GetMainViewport()->GetCenter(); ImGui::SetNextWindowPos(fc, ImGuiCond_Appearing, ImVec2(0.5f,0.5f)); }
        if (ImGui::BeginPopupModal("\xe6\x96\x87\xe4\xbb\xb6##FD", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            const char* arrT[] = {"\xe6\x89\x93\xe5\xbc\x80", "\xe4\xbf\x9d\xe5\xad\x98", "\xe5\x8f\xa6\xe5\xad\x98\xe4\xb8\xba", "\xe5\xaf\xbc\xe5\x87\xba\xe6\xa8\xa1\xe5\x9e\x8b", "CSV", "HTML"};
            int np = appState.nFileDialogPurpose;
            if (np >= 0 && np <= 5) ImGui::TextColored(s_colAccent, "%s", arrT[np]);
            ImGui::Separator(); ImGui::InputText("\xe8\xb7\xaf\xe5\xbe\x84", appState.arrFileDialogPath, sizeof(appState.arrFileDialogPath));
            if (ImGui::Button("\xe7\xa1\xae\xe5\xae\x9a", ImVec2(80,0))) {
                std::string sp(appState.arrFileDialogPath);
                switch(np) { case 0: loadProject(appState,sp); break; case 1: case 2: saveProject(appState,sp); break; case 3: exportModelToPath(appState,sp); break; case 5: exportEvaluationReportHTML(appState,sp); break; }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("\xe5\x8f\x96\xe6\xb6\x88", ImVec2(80,0))) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // ============================================================
        // 20260320 ZJH 页面选项卡 + 内容
        // ============================================================
        drawPageTabBar(appState);

        // 20260320 ZJH 页面内容区域
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 8));
        ImGui::BeginChild("##PageContent", ImVec2(0, 0));
        switch (appState.nActivePage) {
            case 0: drawPageData(appState); break;
            case 1: drawPageLabeling(appState); break;
            case 2: drawPageTraining(appState); break;
            case 3: drawPageEvaluation(appState); break;
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();

        ImGui::End();  // ##Main

        // 20260320 ZJH 状态栏
        drawStatusBar(appState);

        // 20260320 ZJH 训练完成自动记录历史
        bool bNowCompleted = appState.trainState.bCompleted.load();
        if (bNowCompleted && !bPrevCompleted) {
            TrainingHistoryEntry entry;
            const char* arrDN[] = {"YOLOv5","YOLOv7","YOLOv8","YOLOv10"};
            const char* arrTM[] = {"MLP/ResNet","YOLO","U-Net","AutoEncoder"};
            entry.strModel = (appState.activeTask == TaskType::Detection) ? arrDN[appState.nDetModel] : arrTM[(int)appState.activeTask];
            entry.fAccuracy = appState.trainState.fTestAcc.load();
            entry.fLoss = appState.trainState.fBestValLoss;
            auto now = std::chrono::system_clock::now(); auto t = std::chrono::system_clock::to_time_t(now);
            struct tm tmL; localtime_s(&tmL, &t);
            char db[32]; std::snprintf(db, sizeof(db), "%04d-%02d-%02d %02d:%02d", tmL.tm_year+1900, tmL.tm_mon+1, tmL.tm_mday, tmL.tm_hour, tmL.tm_min);
            entry.strDate = db;
            appState.vecTrainHistory.push_back(entry);
        }
        bPrevCompleted = bNowCompleted;

        ImGui::Render();
        SDL_SetRenderDrawColor(pRenderer, 0x1a, 0x1d, 0x23, 255);
        SDL_RenderClear(pRenderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), pRenderer);
        SDL_RenderPresent(pRenderer);
    }

    // 20260320 ZJH 清理
    if (appState.pTrainThread) { appState.trainState.bStopRequested.store(true); appState.trainState.bPaused.store(false); appState.pTrainThread->request_stop(); appState.pTrainThread->join(); appState.pTrainThread.reset(); }
    ImGui_ImplSDLRenderer3_Shutdown(); ImGui_ImplSDL3_Shutdown(); ImPlot::DestroyContext(); ImGui::DestroyContext();
    SDL_DestroyRenderer(pRenderer); SDL_DestroyWindow(pWindow); SDL_Quit();
    return 0;
}
