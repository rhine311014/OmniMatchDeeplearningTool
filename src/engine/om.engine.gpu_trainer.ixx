// 20260321 ZJH GPU 训练性能优化器模块
// 自动检测 GPU → VRAM 感知 auto batch size → CUDA stream 异步流水线
// 目标: GPU 利用率 ≥90%, 显存利用率 ≥90%
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

export module om.engine.gpu_trainer;

export namespace om {

// =========================================================
// GPU 设备信息查询
// =========================================================

// 20260321 ZJH GpuInfo — GPU 设备详细信息
struct GpuInfo {
    std::string strName;               // 20260321 ZJH GPU 名称
    size_t nTotalVramBytes = 0;        // 20260321 ZJH 总显存（字节）
    size_t nFreeVramBytes = 0;         // 20260321 ZJH 可用显存（字节）
    int nComputeCapMajor = 0;          // 20260321 ZJH 计算能力主版本
    int nComputeCapMinor = 0;          // 20260321 ZJH 计算能力次版本
    int nMultiProcessorCount = 0;      // 20260321 ZJH SM 数量
    int nMaxThreadsPerSM = 0;          // 20260321 ZJH 每个 SM 最大线程数
    int nMaxBlocksPerSM = 0;           // 20260321 ZJH 每个 SM 最大块数
    size_t nSharedMemPerBlock = 0;     // 20260321 ZJH 每块共享内存（字节）
    int nWarpSize = 32;                // 20260321 ZJH warp 大小
    float fClockRateMHz = 0.0f;        // 20260321 ZJH 核心频率 MHz
    float fMemClockRateMHz = 0.0f;     // 20260321 ZJH 显存频率 MHz
    int nMemBusWidth = 0;              // 20260321 ZJH 显存总线位宽
    bool bTensorCoreSupport = false;   // 20260321 ZJH 是否支持 Tensor Core（Volta+）
    bool bFP16Support = false;         // 20260321 ZJH 是否支持 FP16（Pascal+）
    bool bAvailable = false;           // 20260321 ZJH 是否可用
};

// 20260321 ZJH 通过动态加载 nvcuda.dll 获取 GPU 信息
// 不依赖编译时 CUDA SDK，运行时检测
GpuInfo queryGpuInfo(int nDeviceId = 0) {
    GpuInfo info;

#ifdef _WIN32
    // 20260321 ZJH 动态加载 nvcuda.dll
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    if (!hCuda) return info;  // 20260321 ZJH 无 NVIDIA 驱动

    // 20260321 ZJH cuInit
    typedef int (*cuInit_t)(unsigned int);
    auto pfnInit = (cuInit_t)GetProcAddress(hCuda, "cuInit");
    if (!pfnInit || pfnInit(0) != 0) { FreeLibrary(hCuda); return info; }

    // 20260321 ZJH cuDeviceGetCount
    typedef int (*cuDeviceGetCount_t)(int*);
    auto pfnCount = (cuDeviceGetCount_t)GetProcAddress(hCuda, "cuDeviceGetCount");
    int nCount = 0;
    if (!pfnCount || pfnCount(&nCount) != 0 || nDeviceId >= nCount) { FreeLibrary(hCuda); return info; }

    // 20260321 ZJH cuDeviceGet
    typedef int (*cuDeviceGet_t)(int*, int);
    auto pfnGet = (cuDeviceGet_t)GetProcAddress(hCuda, "cuDeviceGet");
    int nDev = 0;
    if (!pfnGet || pfnGet(&nDev, nDeviceId) != 0) { FreeLibrary(hCuda); return info; }

    // 20260321 ZJH cuDeviceGetName
    typedef int (*cuDeviceGetName_t)(char*, int, int);
    auto pfnName = (cuDeviceGetName_t)GetProcAddress(hCuda, "cuDeviceGetName");
    char arrName[256] = {};
    if (pfnName) pfnName(arrName, sizeof(arrName), nDev);
    info.strName = arrName;

    // 20260321 ZJH cuDeviceTotalMem
    typedef int (*cuDeviceTotalMem_t)(size_t*, int);
    auto pfnMem = (cuDeviceTotalMem_t)GetProcAddress(hCuda, "cuDeviceTotalMem_v2");
    if (pfnMem) pfnMem(&info.nTotalVramBytes, nDev);

    // 20260321 ZJH cuDeviceGetAttribute 获取详细属性
    typedef int (*cuDeviceGetAttribute_t)(int*, int, int);
    auto pfnAttr = (cuDeviceGetAttribute_t)GetProcAddress(hCuda, "cuDeviceGetAttribute");
    if (pfnAttr) {
        int nVal = 0;
        pfnAttr(&nVal, 75, nDev);  info.nComputeCapMajor = nVal;       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        pfnAttr(&nVal, 76, nDev);  info.nComputeCapMinor = nVal;       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        pfnAttr(&nVal, 16, nDev);  info.nMultiProcessorCount = nVal;   // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        pfnAttr(&nVal, 39, nDev);  info.nMaxThreadsPerSM = nVal;       // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
        pfnAttr(&nVal, 106, nDev); info.nMaxBlocksPerSM = nVal;        // CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR
        pfnAttr(&nVal, 8, nDev);   info.nSharedMemPerBlock = static_cast<size_t>(nVal); // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
        pfnAttr(&nVal, 10, nDev);  info.nWarpSize = nVal;              // CU_DEVICE_ATTRIBUTE_WARP_SIZE
        pfnAttr(&nVal, 13, nDev);  info.fClockRateMHz = nVal / 1000.0f; // CU_DEVICE_ATTRIBUTE_CLOCK_RATE (kHz)
        pfnAttr(&nVal, 36, nDev);  info.fMemClockRateMHz = nVal / 1000.0f; // CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE (kHz)
        pfnAttr(&nVal, 37, nDev);  info.nMemBusWidth = nVal;           // CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
    }

    // 20260321 ZJH 根据计算能力判断特性支持
    info.bFP16Support = (info.nComputeCapMajor >= 6);          // Pascal (6.0)+
    info.bTensorCoreSupport = (info.nComputeCapMajor >= 7);    // Volta (7.0)+
    info.bAvailable = true;
    info.nFreeVramBytes = info.nTotalVramBytes;  // 20260321 ZJH 初始近似（cuMemGetInfo 需要 context）

    FreeLibrary(hCuda);
#endif

    return info;
}

// =========================================================
// GPU 训练性能配置器
// =========================================================

// 20260321 ZJH GpuTrainingConfig — GPU 训练优化配置
// 根据 GPU 硬件参数自动计算最优训练参数
struct GpuTrainingConfig {
    // 20260321 ZJH 输入参数
    int nModelParamCount = 0;          // 模型参数量
    int nInputChannels = 1;            // 输入通道数
    int nInputHeight = 28;             // 输入高度
    int nInputWidth = 28;              // 输入宽度
    int nNumClasses = 10;              // 类别数
    float fTargetVramUsage = 0.90f;    // 目标显存利用率（默认 90%）
    float fTargetGpuUsage = 0.90f;     // 目标 GPU 利用率（默认 90%）

    // 20260321 ZJH 输出参数（由 optimize() 计算）
    int nOptimalBatchSize = 64;        // 最优 batch size
    int nNumStreams = 2;               // CUDA stream 数量（流水线并行）
    int nPrefetchBatches = 2;          // 数据预取批次数
    bool bUseFP16 = false;             // 是否使用 FP16 混合精度
    bool bUseTensorCore = false;       // 是否使用 Tensor Core
    bool bUsePinnedMemory = true;      // 是否使用锁页内存
    bool bUseAsyncTransfer = true;     // 是否使用异步数据传输
    int nDataLoaderThreads = 4;        // 数据加载线程数
    size_t nEstimatedVramUsageMB = 0;  // 预估显存使用量 MB
    size_t nAvailableVramMB = 0;       // 可用显存 MB
    float fEstimatedGpuUtil = 0.0f;    // 预估 GPU 利用率
    std::string strOptimizationLog;    // 优化过程日志
};

// 20260321 ZJH GpuPerformanceOptimizer — GPU 训练性能优化器
// 自动分析 GPU 硬件能力，计算最优训练参数
class GpuPerformanceOptimizer {
public:
    // 20260321 ZJH 构造函数
    // nDeviceId: GPU 设备索引
    GpuPerformanceOptimizer(int nDeviceId = 0) : m_nDeviceId(nDeviceId) {
        m_gpuInfo = queryGpuInfo(nDeviceId);
    }

    // 20260321 ZJH optimize — 计算最优训练配置
    // 根据模型大小和 GPU 硬件自动计算 batch size、stream 数、FP16 等参数
    GpuTrainingConfig optimize(const GpuTrainingConfig& inputCfg) {
        GpuTrainingConfig cfg = inputCfg;
        char buf[512];

        if (!m_gpuInfo.bAvailable) {
            cfg.strOptimizationLog = "GPU \xe4\xb8\x8d\xe5\x8f\xaf\xe7\x94\xa8\xef\xbc\x8c\xe4\xbd\xbf\xe7\x94\xa8 CPU \xe8\xae\xad\xe7\xbb\x83\n";
            cfg.nOptimalBatchSize = 32;
            return cfg;
        }

        size_t nVramBytes = m_gpuInfo.nTotalVramBytes;
        size_t nVramMB = nVramBytes / (1024 * 1024);
        cfg.nAvailableVramMB = nVramMB;

        std::snprintf(buf, sizeof(buf), "GPU: %s  VRAM: %zuMB  SM: %d  CC: %d.%d\n",
                      m_gpuInfo.strName.c_str(), nVramMB,
                      m_gpuInfo.nMultiProcessorCount,
                      m_gpuInfo.nComputeCapMajor, m_gpuInfo.nComputeCapMinor);
        cfg.strOptimizationLog += buf;

        // 20260321 ZJH 1. FP16/Tensor Core 决策
        cfg.bUseFP16 = m_gpuInfo.bFP16Support;
        cfg.bUseTensorCore = m_gpuInfo.bTensorCoreSupport;
        if (cfg.bUseFP16) {
            cfg.strOptimizationLog += "FP16 \xe6\xb7\xb7\xe5\x90\x88\xe7\xb2\xbe\xe5\xba\xa6: \xe5\xbc\x80\xe5\x90\xaf (2x \xe5\x90\x9e\xe5\x90\x90\xe9\x87\x8f\xe6\x8f\x90\xe5\x8d\x87)\n";
        }
        if (cfg.bUseTensorCore) {
            cfg.strOptimizationLog += "Tensor Core: \xe5\xbc\x80\xe5\x90\xaf (matmul \xe5\x8a\xa0\xe9\x80\x9f)\n";
        }

        // 20260321 ZJH 2. 计算模型内存占用
        // 参数 + 梯度 + 优化器状态（Adam = 2x） + 激活缓存
        int nParamBytes = cfg.nModelParamCount * 4;  // FP32
        if (cfg.bUseFP16) nParamBytes = cfg.nModelParamCount * 2;  // FP16 参数
        int nGradBytes = cfg.nModelParamCount * 4;     // 梯度始终 FP32
        int nOptBytes = cfg.nModelParamCount * 4 * 2;  // Adam: m + v
        size_t nModelMemBytes = static_cast<size_t>(nParamBytes + nGradBytes + nOptBytes);

        std::snprintf(buf, sizeof(buf), "\xe6\xa8\xa1\xe5\x9e\x8b\xe5\x86\x85\xe5\xad\x98: %.1fMB (\xe5\x8f\x82\xe6\x95\xb0+\xe6\xa2\xaf\xe5\xba\xa6+\xe4\xbc\x98\xe5\x8c\x96\xe5\x99\xa8)\n",
                      nModelMemBytes / (1024.0 * 1024.0));
        cfg.strOptimizationLog += buf;

        // 20260321 ZJH 3. 计算单张图像的激活内存占用（近似）
        // 粗略估计: 输入 + 各层激活 ≈ 输入大小 × 模型深度系数
        size_t nInputBytes = static_cast<size_t>(cfg.nInputChannels * cfg.nInputHeight * cfg.nInputWidth * 4);
        float fActivationMultiplier = 8.0f;  // 20260321 ZJH 典型 CNN 激活占输入的 8 倍
        size_t nPerSampleActivation = static_cast<size_t>(nInputBytes * fActivationMultiplier);

        // 20260321 ZJH 4. 计算目标显存 = 总显存 × 目标利用率
        size_t nTargetVram = static_cast<size_t>(nVramBytes * cfg.fTargetVramUsage);
        // 20260321 ZJH 留出 200MB 给系统/驱动
        if (nTargetVram > 200 * 1024 * 1024) {
            nTargetVram -= 200 * 1024 * 1024;
        }

        // 20260321 ZJH 5. 计算最优 batch size
        // batch_size = (目标显存 - 模型内存) / 每样本激活内存
        size_t nAvailForBatch = (nTargetVram > nModelMemBytes) ? (nTargetVram - nModelMemBytes) : 0;
        int nMaxBatch = static_cast<int>(nAvailForBatch / std::max(nPerSampleActivation, (size_t)1));
        nMaxBatch = std::max(1, nMaxBatch);

        // 20260321 ZJH 对齐到 8 的倍数（Tensor Core 友好）
        if (cfg.bUseTensorCore && nMaxBatch >= 8) {
            nMaxBatch = (nMaxBatch / 8) * 8;
        }

        // 20260321 ZJH 上限保护
        nMaxBatch = std::min(nMaxBatch, 1024);
        cfg.nOptimalBatchSize = nMaxBatch;

        // 20260321 ZJH 计算预估显存使用
        cfg.nEstimatedVramUsageMB = (nModelMemBytes + static_cast<size_t>(nMaxBatch) * nPerSampleActivation) / (1024 * 1024);

        std::snprintf(buf, sizeof(buf), "\xe6\x9c\x80\xe4\xbc\x98 batch size: %d (\xe6\x98\xbe\xe5\xad\x98\xe5\x88\xa9\xe7\x94\xa8: %zuMB / %zuMB = %.0f%%)\n",
                      nMaxBatch, cfg.nEstimatedVramUsageMB, nVramMB,
                      100.0 * cfg.nEstimatedVramUsageMB / nVramMB);
        cfg.strOptimizationLog += buf;

        // 20260321 ZJH 6. CUDA stream 数量（流水线并行提高 GPU 利用率）
        // 2 个 stream: 一个做计算，一个做数据传输 → 隐藏传输延迟
        cfg.nNumStreams = 2;
        if (m_gpuInfo.nMultiProcessorCount >= 40) {
            cfg.nNumStreams = 4;  // 20260321 ZJH 大 GPU（如 3090/4090）用 4 stream
        }

        // 20260321 ZJH 7. 数据预取
        cfg.nPrefetchBatches = cfg.nNumStreams;
        cfg.bUsePinnedMemory = true;
        cfg.bUseAsyncTransfer = true;

        // 20260321 ZJH 8. 数据加载线程数 = min(CPU核心数/2, 8)
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        cfg.nDataLoaderThreads = std::min(static_cast<int>(si.dwNumberOfProcessors) / 2, 8);
#else
        cfg.nDataLoaderThreads = 4;
#endif
        cfg.nDataLoaderThreads = std::max(cfg.nDataLoaderThreads, 2);

        // 20260321 ZJH 9. GPU 利用率预估
        // 大 batch + 流水线 + FP16 → 高利用率
        float fBatchUtil = std::min(1.0f, nMaxBatch / 64.0f);  // batch 越大利用率越高
        float fStreamUtil = 0.85f + 0.05f * cfg.nNumStreams;     // 多 stream 提升
        float fFP16Boost = cfg.bUseFP16 ? 1.1f : 1.0f;
        cfg.fEstimatedGpuUtil = std::min(0.98f, fBatchUtil * fStreamUtil * fFP16Boost);

        std::snprintf(buf, sizeof(buf), "\xe9\xa2\x84\xe4\xbc\xb0 GPU \xe5\x88\xa9\xe7\x94\xa8\xe7\x8e\x87: %.0f%%  Stream: %d  \xe9\xa2\x84\xe5\x8f\x96: %d \xe6\x89\xb9\n",
                      cfg.fEstimatedGpuUtil * 100, cfg.nNumStreams, cfg.nPrefetchBatches);
        cfg.strOptimizationLog += buf;

        std::snprintf(buf, sizeof(buf), "\xe9\x94\x81\xe9\xa1\xb5\xe5\x86\x85\xe5\xad\x98: %s  \xe5\xbc\x82\xe6\xad\xa5\xe4\xbc\xa0\xe8\xbe\x93: %s  \xe5\x8a\xa0\xe8\xbd\xbd\xe7\xba\xbf\xe7\xa8\x8b: %d\n",
                      cfg.bUsePinnedMemory ? "\xe5\xbc\x80" : "\xe5\x85\xb3",
                      cfg.bUseAsyncTransfer ? "\xe5\xbc\x80" : "\xe5\x85\xb3",
                      cfg.nDataLoaderThreads);
        cfg.strOptimizationLog += buf;

        return cfg;
    }

    // 20260321 ZJH getGpuInfo — 获取 GPU 硬件信息
    const GpuInfo& getGpuInfo() const { return m_gpuInfo; }

    // 20260321 ZJH isGpuAvailable — GPU 是否可用
    bool isGpuAvailable() const { return m_gpuInfo.bAvailable; }

    // 20260321 ZJH getVramMB — 总显存 MB
    size_t getVramMB() const { return m_gpuInfo.nTotalVramBytes / (1024 * 1024); }

    // 20260321 ZJH generatePerformanceReport — 生成性能优化报告
    std::string generatePerformanceReport(const GpuTrainingConfig& cfg) {
        std::string strReport;
        char buf[512];

        strReport += "========================================\n";
        strReport += "    OmniMatch GPU \xe6\x80\xa7\xe8\x83\xbd\xe4\xbc\x98\xe5\x8c\x96\xe6\x8a\xa5\xe5\x91\x8a\n";
        strReport += "========================================\n\n";

        std::snprintf(buf, sizeof(buf), "GPU: %s\n", m_gpuInfo.strName.c_str());
        strReport += buf;
        std::snprintf(buf, sizeof(buf), "VRAM: %zu MB (%.1f GB)\n",
                      m_gpuInfo.nTotalVramBytes / (1024 * 1024),
                      m_gpuInfo.nTotalVramBytes / (1024.0 * 1024 * 1024));
        strReport += buf;
        std::snprintf(buf, sizeof(buf), "SM: %d  CC: %d.%d  Core: %.0f MHz  Mem: %.0f MHz x %dbit\n",
                      m_gpuInfo.nMultiProcessorCount,
                      m_gpuInfo.nComputeCapMajor, m_gpuInfo.nComputeCapMinor,
                      (double)m_gpuInfo.fClockRateMHz, (double)m_gpuInfo.fMemClockRateMHz,
                      m_gpuInfo.nMemBusWidth);
        strReport += buf;

        strReport += "\n--- \xe4\xbc\x98\xe5\x8c\x96\xe5\x86\xb3\xe7\xad\x96 ---\n";
        std::snprintf(buf, sizeof(buf), "Batch Size: %d (\xe5\xa1\xab\xe5\x85\x85 %.0f%% \xe6\x98\xbe\xe5\xad\x98)\n",
                      cfg.nOptimalBatchSize, 100.0 * cfg.nEstimatedVramUsageMB / cfg.nAvailableVramMB);
        strReport += buf;
        std::snprintf(buf, sizeof(buf), "FP16: %s  Tensor Core: %s\n",
                      cfg.bUseFP16 ? "\xe5\xbc\x80\xe5\x90\xaf" : "\xe5\x85\xb3\xe9\x97\xad",
                      cfg.bUseTensorCore ? "\xe5\xbc\x80\xe5\x90\xaf" : "\xe5\x85\xb3\xe9\x97\xad");
        strReport += buf;
        std::snprintf(buf, sizeof(buf), "CUDA Streams: %d  \xe6\x95\xb0\xe6\x8d\xae\xe9\xa2\x84\xe5\x8f\x96: %d \xe6\x89\xb9\n",
                      cfg.nNumStreams, cfg.nPrefetchBatches);
        strReport += buf;
        std::snprintf(buf, sizeof(buf), "\xe9\xa2\x84\xe4\xbc\xb0 GPU \xe5\x88\xa9\xe7\x94\xa8\xe7\x8e\x87: %.0f%%\n",
                      (double)(cfg.fEstimatedGpuUtil * 100));
        strReport += buf;
        std::snprintf(buf, sizeof(buf), "\xe9\xa2\x84\xe4\xbc\xb0\xe6\x98\xbe\xe5\xad\x98\xe5\x88\xa9\xe7\x94\xa8\xe7\x8e\x87: %.0f%%\n",
                      100.0 * cfg.nEstimatedVramUsageMB / std::max(cfg.nAvailableVramMB, (size_t)1));
        strReport += buf;

        return strReport;
    }

private:
    int m_nDeviceId;   // 20260321 ZJH GPU 设备索引
    GpuInfo m_gpuInfo; // 20260321 ZJH 设备信息缓存
};

}  // namespace om
