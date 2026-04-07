// 20260321 ZJH TensorRT 推理引擎集成模块
// 20260330 ZJH 增强: TrtBuildConfig 精度模式 + INT8 校准器 + buildTrtEngine() 转换流水线 + TrtInference 推理包装器
// 条件编译：OM_HAS_TENSORRT 定义时使用真实 TensorRT API
// 未定义时提供 stub 实现（空方法 + 错误信息），确保无 TensorRT SDK 也能编译
module;

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <functional>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <cassert>

#ifdef OM_HAS_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

export module om.engine.tensorrt;

export namespace om {

// =========================================================
// TensorRTConfig — TensorRT 推理配置（保留向后兼容）
// =========================================================

// 20260321 ZJH TensorRTConfig — 存储 TensorRT 引擎构建和推理所需的全部配置
struct TensorRTConfig {
    std::string strOnnxPath;                   // 20260321 ZJH ONNX 模型文件路径
    std::string strEnginePath;                 // 20260321 ZJH TensorRT engine 序列化缓存路径
    int nMaxBatchSize = 1;                     // 20260321 ZJH 最大批次大小
    bool bFP16 = true;                         // 20260321 ZJH 是否启用 FP16 半精度推理（需 GPU 支持）
    bool bINT8 = false;                        // 20260321 ZJH 是否启用 INT8 量化推理（需校准数据）
    size_t nWorkspaceSize = 1ULL << 30;        // 20260321 ZJH 工作空间大小，默认 1GB
    int nDLACore = -1;                         // 20260321 ZJH DLA 深度学习加速器核心编号（-1 表示不使用）
};

// =========================================================
// TrtBuildConfig — 新版转换流水线配置（对标 TensorRT trtexec）
// =========================================================

// 20260330 ZJH TrtBuildConfig — 用于 buildTrtEngine() 高级转换配置
// 支持 FP32/FP16/INT8 三种精度模式，INT8 需要校准图像目录
struct TrtBuildConfig {
    // 20260330 ZJH 精度枚举：FP32 全精度、FP16 半精度（2x 加速）、INT8 量化（4x 加速）
    enum Precision { FP32, FP16, INT8 } ePrecision = FP16;
    int nMaxBatchSize = 1;                          // 20260330 ZJH 最大批次大小，动态 batch 的上界
    size_t nMaxWorkspaceBytes = 1ULL << 30;         // 20260330 ZJH TRT 内部算法选择的工作空间上限，默认 1GB
    bool bStrictTypes = false;                      // 20260330 ZJH 是否强制使用指定精度（禁止 TRT 回退到更高精度）
    std::string strCalibrationDataDir;              // 20260330 ZJH INT8 校准图像目录路径（仅 INT8 模式需要）
    int nCalibBatchSize = 8;                        // 20260330 ZJH INT8 校准时每批图像数量
    int nCalibMaxBatches = 50;                      // 20260330 ZJH INT8 校准最大批次数（限制校准时间）
    int nDLACore = -1;                              // 20260330 ZJH DLA 加速器核心编号（-1 不使用，Jetson 专用）
    bool bGpuFallback = true;                       // 20260330 ZJH DLA 不支持的层是否回退到 GPU
    std::vector<std::string> vecDynamicInputNames;  // 20260330 ZJH 动态输入名称列表（空则使用模型默认）
    int nOptBatchSize = 1;                          // 20260330 ZJH 优化 profile 的最佳批次大小
    bool bVerboseLog = false;                       // 20260330 ZJH 是否输出详细构建日志（kINFO 级别）
};

// =========================================================
// 条件编译：OM_HAS_TENSORRT 真实实现
// =========================================================

#ifdef OM_HAS_TENSORRT

// 20260321 ZJH TRTLogger — TensorRT 内部日志回调
// 实现 nvinfer1::ILogger 接口，将 TensorRT 内部消息转发到标准错误输出
class TRTLogger : public nvinfer1::ILogger {
public:
    // 20260330 ZJH 构造函数：可指定最低输出级别
    // minSeverity: 最低输出级别，低于此级别的日志将被忽略
    explicit TRTLogger(Severity minSeverity = Severity::kWARNING)
        : m_minSeverity(minSeverity) {}

    // 20260321 ZJH log — 接收 TensorRT 内部日志
    // severity: 日志级别（kERROR, kWARNING, kINFO, kVERBOSE）
    // msg: 日志消息内容
    void log(Severity severity, const char* msg) noexcept override {
        // 20260330 ZJH 根据配置的最低级别过滤日志
        if (severity <= m_minSeverity) {
            // 20260330 ZJH 根据级别添加不同前缀，方便排查问题
            const char* pPrefix = "[TensorRT] ";  // 20260330 ZJH 默认前缀
            switch (severity) {
                case Severity::kINTERNAL_ERROR: pPrefix = "[TRT INTERNAL_ERROR] "; break;
                case Severity::kERROR:          pPrefix = "[TRT ERROR] "; break;
                case Severity::kWARNING:        pPrefix = "[TRT WARNING] "; break;
                case Severity::kINFO:           pPrefix = "[TRT INFO] "; break;
                case Severity::kVERBOSE:        pPrefix = "[TRT VERBOSE] "; break;
            }
            std::cerr << pPrefix << msg << std::endl;  // 20260321 ZJH 输出到标准错误流
        }
    }

    // 20260330 ZJH setMinSeverity — 运行时调整日志级别
    void setMinSeverity(Severity severity) { m_minSeverity = severity; }

private:
    Severity m_minSeverity;  // 20260330 ZJH 最低输出级别
};

// 20260321 ZJH TRTDeleter — TensorRT 对象自定义删除器
// 用于 std::unique_ptr，在释放时调用 destroy() 而非 delete
struct TRTDeleter {
    // 20260321 ZJH 调用 TensorRT 对象的 destroy 方法释放资源
    template <typename T>
    void operator()(T* pObj) const {
        if (pObj) {
            pObj->destroy();  // 20260321 ZJH TensorRT 对象必须通过 destroy() 释放
        }
    }
};

// =========================================================
// INT8EntropyCalibrator — INT8 量化校准器（MinMax/Entropy 策略）
// =========================================================

// 20260330 ZJH INT8EntropyCalibrator — 从图像文件夹读取校准数据
// 实现 nvinfer1::IInt8EntropyCalibrator2 接口
// 校准流程: 遍历校准图片 → 预处理为模型输入格式 → 送入 TRT 统计激活值分布 → 生成 INT8 量化参数
class INT8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    // 20260330 ZJH 构造函数
    // strCalibDir: 校准图像目录路径
    // nBatchSize: 每批校准图像数量
    // nInputH: 模型输入高度
    // nInputW: 模型输入宽度
    // nInputC: 模型输入通道数（通常为 3）
    // nMaxBatches: 最大校准批次数
    INT8EntropyCalibrator(const std::string& strCalibDir,
                          int nBatchSize, int nInputH, int nInputW, int nInputC,
                          int nMaxBatches)
        : m_nBatchSize(nBatchSize)
        , m_nInputH(nInputH)
        , m_nInputW(nInputW)
        , m_nInputC(nInputC)
        , m_nMaxBatches(nMaxBatches)
        , m_nCurrentBatch(0)
        , m_pDeviceBuf(nullptr)
    {
        // 20260330 ZJH 计算单张图像的元素数和单批的字节数
        m_nImageSize = static_cast<size_t>(nInputC) * nInputH * nInputW;
        size_t nBatchBytes = static_cast<size_t>(nBatchSize) * m_nImageSize * sizeof(float);

        // 20260330 ZJH 在 GPU 上分配校准数据缓冲区
        cudaMalloc(&m_pDeviceBuf, nBatchBytes);

        // 20260330 ZJH 收集校准目录下的图像文件（jpg/png/bmp）
        if (std::filesystem::exists(strCalibDir) && std::filesystem::is_directory(strCalibDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(strCalibDir)) {
                if (!entry.is_regular_file()) continue;  // 20260330 ZJH 跳过非文件项
                std::string strExt = entry.path().extension().string();
                // 20260330 ZJH 转小写比较扩展名
                std::transform(strExt.begin(), strExt.end(), strExt.begin(), ::tolower);
                if (strExt == ".jpg" || strExt == ".jpeg" || strExt == ".png" || strExt == ".bmp") {
                    m_vecImagePaths.push_back(entry.path().string());  // 20260330 ZJH 记录有效图像路径
                }
            }
            // 20260330 ZJH 排序确保校准结果可复现
            std::sort(m_vecImagePaths.begin(), m_vecImagePaths.end());
        }

        // 20260330 ZJH 预分配 CPU 侧批次缓冲区
        m_vecHostBuf.resize(static_cast<size_t>(nBatchSize) * m_nImageSize, 0.0f);

        // 20260330 ZJH 尝试读取已有的校准缓存文件
        m_strCacheFile = strCalibDir + "/calibration_cache.bin";
        std::ifstream cacheFile(m_strCacheFile, std::ios::binary | std::ios::ate);
        if (cacheFile.is_open()) {
            size_t nCacheSize = cacheFile.tellg();  // 20260330 ZJH 获取缓存文件大小
            cacheFile.seekg(0, std::ios::beg);
            m_vecCalibCache.resize(nCacheSize);
            cacheFile.read(reinterpret_cast<char*>(m_vecCalibCache.data()), nCacheSize);
            cacheFile.close();
            std::cerr << "[TensorRT] Loaded calibration cache: " << nCacheSize << " bytes" << std::endl;
        }
    }

    // 20260330 ZJH 析构函数：释放 GPU 校准缓冲区
    ~INT8EntropyCalibrator() override {
        if (m_pDeviceBuf) {
            cudaFree(m_pDeviceBuf);  // 20260330 ZJH 释放 CUDA 内存
            m_pDeviceBuf = nullptr;
        }
    }

    // 20260330 ZJH getBatchSize — 返回校准批次大小
    int getBatchSize() const noexcept override {
        return m_nBatchSize;
    }

    // 20260330 ZJH getBatch — 获取下一批校准数据
    // bindings: 输出绑定指针数组（需填入 GPU 指针）
    // names: 输入名称数组（TRT 传入）
    // nbBindings: 绑定数量
    // 返回: 成功获取返回 true，无更多数据返回 false
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        (void)names;       // 20260330 ZJH 抑制未使用参数警告
        (void)nbBindings;  // 20260330 ZJH 抑制未使用参数警告

        // 20260330 ZJH 检查是否超过最大批次数或图像已用完
        if (m_nCurrentBatch >= m_nMaxBatches) return false;
        size_t nStartIdx = static_cast<size_t>(m_nCurrentBatch) * m_nBatchSize;
        if (nStartIdx >= m_vecImagePaths.size()) return false;  // 20260330 ZJH 所有图像已处理完

        // 20260330 ZJH 逐张读取并预处理图像到 CPU 缓冲区
        // 注意: 此处使用简化的二进制读取方式（假设 raw float 数据）
        // 生产环境应集成 OpenCV 读取真实图像并做 resize + normalize
        std::fill(m_vecHostBuf.begin(), m_vecHostBuf.end(), 0.0f);  // 20260330 ZJH 清零缓冲区

        for (int i = 0; i < m_nBatchSize; ++i) {
            size_t nImgIdx = nStartIdx + i;  // 20260330 ZJH 当前图像在列表中的索引
            if (nImgIdx >= m_vecImagePaths.size()) break;  // 20260330 ZJH 不足一个 batch 时用零填充

            // 20260330 ZJH 读取图像文件的原始字节
            // 实际使用时应替换为 OpenCV imread + resize + CHW 转换 + /255.0 归一化
            std::ifstream imgFile(m_vecImagePaths[nImgIdx], std::ios::binary | std::ios::ate);
            if (!imgFile.is_open()) continue;  // 20260330 ZJH 跳过无法打开的文件

            size_t nFileSize = imgFile.tellg();  // 20260330 ZJH 获取文件大小
            imgFile.seekg(0, std::ios::beg);

            // 20260330 ZJH 读取最多 m_nImageSize 个 float 的数据量
            size_t nReadFloats = std::min(nFileSize / sizeof(float), m_nImageSize);
            size_t nOffset = static_cast<size_t>(i) * m_nImageSize;  // 20260330 ZJH 在批次缓冲区中的偏移
            imgFile.read(reinterpret_cast<char*>(m_vecHostBuf.data() + nOffset),
                        nReadFloats * sizeof(float));
            imgFile.close();
        }

        // 20260330 ZJH 将 CPU 缓冲区拷贝到 GPU
        size_t nBatchBytes = static_cast<size_t>(m_nBatchSize) * m_nImageSize * sizeof(float);
        cudaMemcpy(m_pDeviceBuf, m_vecHostBuf.data(), nBatchBytes, cudaMemcpyHostToDevice);

        bindings[0] = m_pDeviceBuf;  // 20260330 ZJH 设置输入绑定为 GPU 缓冲区指针
        ++m_nCurrentBatch;  // 20260330 ZJH 推进批次计数器

        std::cerr << "[TensorRT] Calibration batch " << m_nCurrentBatch
                  << "/" << m_nMaxBatches << std::endl;
        return true;
    }

    // 20260330 ZJH readCalibrationCache — 读取校准缓存
    // 返回已缓存的校准数据指针，TRT 使用缓存可跳过重新校准
    const void* readCalibrationCache(size_t& length) noexcept override {
        if (m_vecCalibCache.empty()) {
            length = 0;
            return nullptr;  // 20260330 ZJH 无缓存，TRT 将执行完整校准
        }
        length = m_vecCalibCache.size();
        return m_vecCalibCache.data();  // 20260330 ZJH 返回缓存数据
    }

    // 20260330 ZJH writeCalibrationCache — 保存校准缓存到文件
    // 校准完成后 TRT 回调此方法，将量化参数持久化
    void writeCalibrationCache(const void* pCache, size_t nLength) noexcept override {
        // 20260330 ZJH 写入校准缓存文件，下次可跳过重新校准
        std::ofstream cacheFile(m_strCacheFile, std::ios::binary);
        if (cacheFile.is_open()) {
            cacheFile.write(static_cast<const char*>(pCache), nLength);
            cacheFile.close();
            std::cerr << "[TensorRT] Saved calibration cache: " << nLength << " bytes" << std::endl;
        }
    }

private:
    int m_nBatchSize;                          // 20260330 ZJH 每批校准图像数量
    int m_nInputH, m_nInputW, m_nInputC;       // 20260330 ZJH 模型输入尺寸
    size_t m_nImageSize;                       // 20260330 ZJH 单张图像元素数
    int m_nMaxBatches;                         // 20260330 ZJH 最大校准批次数
    int m_nCurrentBatch;                       // 20260330 ZJH 当前批次索引
    void* m_pDeviceBuf;                        // 20260330 ZJH GPU 校准数据缓冲区
    std::vector<float> m_vecHostBuf;           // 20260330 ZJH CPU 侧批次缓冲区
    std::vector<std::string> m_vecImagePaths;  // 20260330 ZJH 校准图像路径列表
    std::string m_strCacheFile;                // 20260330 ZJH 校准缓存文件路径
    std::vector<uint8_t> m_vecCalibCache;      // 20260330 ZJH 已加载的校准缓存数据
};

// 20260321 ZJH TensorRTEngine — TensorRT 推理引擎（真实实现）
// 支持从 ONNX 模型构建 engine、序列化/反序列化、执行推理
class TensorRTEngine {
public:
    // 20260321 ZJH 构造函数，初始化日志器和就绪标志
    TensorRTEngine() : m_bReady(false) {}

    // 20260321 ZJH 析构函数，释放 CUDA 资源
    ~TensorRTEngine() {
        // 20260321 ZJH 释放 GPU 显存缓冲区
        for (void* pBuf : m_vecBuffers) {
            if (pBuf) {
                cudaFree(pBuf);  // 20260321 ZJH 释放 CUDA 分配的 GPU 内存
            }
        }
    }

    // 20260321 ZJH build — 从 ONNX 模型文件构建 TensorRT 引擎
    // config: 包含 ONNX 路径、精度模式、工作空间等配置
    // 返回: 构建成功返回 true，失败返回 false
    bool build(const TensorRTConfig& config) {
        // 20260321 ZJH 创建 TensorRT builder 实例
        auto pBuilder = std::unique_ptr<nvinfer1::IBuilder, TRTDeleter>(
            nvinfer1::createInferBuilder(m_logger));
        if (!pBuilder) return false;  // 20260321 ZJH builder 创建失败

        // 20260321 ZJH 创建网络定义（显式批次模式）
        const auto nExplicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto pNetwork = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDeleter>(
            pBuilder->createNetworkV2(nExplicitBatch));
        if (!pNetwork) return false;  // 20260321 ZJH 网络定义创建失败

        // 20260321 ZJH 创建 ONNX 解析器，将 ONNX 模型解析到网络定义中
        auto pParser = std::unique_ptr<nvonnxparser::IParser, TRTDeleter>(
            nvonnxparser::createParser(*pNetwork, m_logger));
        if (!pParser) return false;  // 20260321 ZJH 解析器创建失败

        // 20260321 ZJH 解析 ONNX 文件
        if (!pParser->parseFromFile(config.strOnnxPath.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "[TensorRT] Failed to parse ONNX: " << config.strOnnxPath << std::endl;
            return false;  // 20260321 ZJH ONNX 解析失败
        }

        // 20260321 ZJH 创建构建配置
        auto pConfig = std::unique_ptr<nvinfer1::IBuilderConfig, TRTDeleter>(
            pBuilder->createBuilderConfig());
        if (!pConfig) return false;  // 20260321 ZJH 配置创建失败

        // 20260321 ZJH 设置最大工作空间内存
        pConfig->setMaxWorkspaceSize(config.nWorkspaceSize);

        // 20260321 ZJH 根据配置启用 FP16 模式（半精度推理，速度提升约 2 倍）
        if (config.bFP16 && pBuilder->platformHasFastFp16()) {
            pConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        // 20260321 ZJH 根据配置启用 INT8 模式（需要校准数据，速度提升约 4 倍）
        if (config.bINT8 && pBuilder->platformHasFastInt8()) {
            pConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
            // 20260321 ZJH 注意：INT8 需要提供校准器（IInt8Calibrator），此处未实现
        }

        // 20260321 ZJH 配置 DLA 深度学习加速器（Jetson 平台专用）
        if (config.nDLACore >= 0) {
            if (pBuilder->getNbDLACores() > config.nDLACore) {
                pConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
                pConfig->setDLACore(config.nDLACore);
                pConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);  // 20260321 ZJH DLA 不支持的层回退到 GPU
            }
        }

        // 20260321 ZJH 设置最大批次大小
        pBuilder->setMaxBatchSize(config.nMaxBatchSize);

        // 20260321 ZJH 构建优化后的 TensorRT 引擎（耗时操作，可能需要数分钟）
        m_pEngine.reset(pBuilder->buildEngineWithConfig(*pNetwork, *pConfig));
        if (!m_pEngine) {
            std::cerr << "[TensorRT] Failed to build engine" << std::endl;
            return false;  // 20260321 ZJH 引擎构建失败
        }

        // 20260321 ZJH 创建执行上下文（用于推理）
        m_pContext.reset(m_pEngine->createExecutionContext());
        if (!m_pContext) return false;  // 20260321 ZJH 上下文创建失败

        // 20260321 ZJH 分配输入/输出 GPU 缓冲区
        if (!allocateBuffers()) return false;

        // 20260321 ZJH 自动保存引擎缓存（加速下次加载）
        if (!config.strEnginePath.empty()) {
            saveEngine(config.strEnginePath);
        }

        m_bReady = true;  // 20260321 ZJH 标记引擎就绪
        return true;
    }

    // 20260321 ZJH loadEngine — 从序列化文件加载已构建的 TensorRT 引擎
    // strPath: 序列化 engine 文件路径
    // 返回: 加载成功返回 true
    bool loadEngine(const std::string& strPath) {
        // 20260321 ZJH 读取序列化文件到内存
        std::ifstream file(strPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "[TensorRT] Cannot open engine file: " << strPath << std::endl;
            return false;  // 20260321 ZJH 文件打开失败
        }

        size_t nSize = file.tellg();  // 20260321 ZJH 获取文件大小
        file.seekg(0, std::ios::beg);  // 20260321 ZJH 回到文件起始位置
        std::vector<char> vecEngineData(nSize);  // 20260321 ZJH 分配读取缓冲区
        file.read(vecEngineData.data(), nSize);  // 20260321 ZJH 一次性读取全部数据
        file.close();

        // 20260321 ZJH 创建 TensorRT 运行时
        auto pRuntime = std::unique_ptr<nvinfer1::IRuntime, TRTDeleter>(
            nvinfer1::createInferRuntime(m_logger));
        if (!pRuntime) return false;  // 20260321 ZJH 运行时创建失败

        // 20260321 ZJH 从序列化数据反序列化引擎
        m_pEngine.reset(pRuntime->deserializeCudaEngine(
            vecEngineData.data(), nSize, nullptr));
        if (!m_pEngine) {
            std::cerr << "[TensorRT] Failed to deserialize engine" << std::endl;
            return false;  // 20260321 ZJH 反序列化失败
        }

        // 20260321 ZJH 创建执行上下文
        m_pContext.reset(m_pEngine->createExecutionContext());
        if (!m_pContext) return false;  // 20260321 ZJH 上下文创建失败

        // 20260321 ZJH 分配 GPU 缓冲区
        if (!allocateBuffers()) return false;

        m_bReady = true;  // 20260321 ZJH 标记引擎就绪
        return true;
    }

    // 20260321 ZJH saveEngine — 将当前引擎序列化保存到文件
    // strPath: 输出文件路径（.engine 或 .trt）
    // 返回: 保存成功返回 true
    bool saveEngine(const std::string& strPath) {
        if (!m_pEngine) return false;  // 20260321 ZJH 引擎未就绪

        // 20260321 ZJH 序列化引擎为字节流
        auto pSerialized = std::unique_ptr<nvinfer1::IHostMemory, TRTDeleter>(
            m_pEngine->serialize());
        if (!pSerialized) return false;  // 20260321 ZJH 序列化失败

        // 20260321 ZJH 写入文件
        std::ofstream file(strPath, std::ios::binary);
        if (!file.is_open()) return false;  // 20260321 ZJH 文件打开失败
        file.write(static_cast<const char*>(pSerialized->data()), pSerialized->size());
        file.close();
        return true;
    }

    // 20260321 ZJH infer — 执行推理
    // pInput: 输入数据指针（float 数组，CPU 内存）
    // nBatchSize: 当前批次大小（不超过 nMaxBatchSize）
    // 返回: 推理输出结果（float 向量）
    std::vector<float> infer(const float* pInput, int nBatchSize) {
        if (!m_bReady || !m_pContext) return {};  // 20260321 ZJH 引擎未就绪，返回空

        // 20260321 ZJH 计算输入数据字节数
        size_t nInputBytes = static_cast<size_t>(nBatchSize) * m_nInputSize * sizeof(float);
        // 20260321 ZJH 将输入数据从 CPU 拷贝到 GPU
        cudaMemcpy(m_vecBuffers[0], pInput, nInputBytes, cudaMemcpyHostToDevice);

        // 20260321 ZJH 执行推理（异步入队后同步等待）
        m_pContext->enqueueV2(m_vecBuffers.data(), nullptr, nullptr);
        cudaStreamSynchronize(nullptr);  // 20260321 ZJH 等待推理完成

        // 20260321 ZJH 计算输出数据元素数
        size_t nOutputElements = static_cast<size_t>(nBatchSize) * m_nOutputSize;
        std::vector<float> vecOutput(nOutputElements);  // 20260321 ZJH 分配输出缓冲区
        size_t nOutputBytes = nOutputElements * sizeof(float);
        // 20260321 ZJH 将输出数据从 GPU 拷贝回 CPU
        cudaMemcpy(vecOutput.data(), m_vecBuffers[1], nOutputBytes, cudaMemcpyDeviceToHost);

        return vecOutput;  // 20260321 ZJH 返回推理结果
    }

    // 20260321 ZJH getInputNames — 获取模型所有输入张量的名称
    // 返回: 输入名称列表
    std::vector<std::string> getInputNames() {
        std::vector<std::string> vecNames;  // 20260321 ZJH 存储输入名称
        if (!m_pEngine) return vecNames;  // 20260321 ZJH 引擎未就绪

        // 20260321 ZJH 遍历所有绑定，筛选输入绑定
        for (int i = 0; i < m_pEngine->getNbBindings(); ++i) {
            if (m_pEngine->bindingIsInput(i)) {
                vecNames.push_back(m_pEngine->getBindingName(i));  // 20260321 ZJH 记录输入名称
            }
        }
        return vecNames;
    }

    // 20260321 ZJH getInputShapes — 获取模型所有输入张量的形状
    // 返回: 每个输入的维度列表（如 [[1, 3, 224, 224]]）
    std::vector<std::vector<int>> getInputShapes() {
        std::vector<std::vector<int>> vecShapes;  // 20260321 ZJH 存储输入形状
        if (!m_pEngine) return vecShapes;  // 20260321 ZJH 引擎未就绪

        // 20260321 ZJH 遍历所有绑定，筛选输入绑定并提取维度
        for (int i = 0; i < m_pEngine->getNbBindings(); ++i) {
            if (m_pEngine->bindingIsInput(i)) {
                auto dims = m_pEngine->getBindingDimensions(i);  // 20260321 ZJH 获取绑定维度
                std::vector<int> vecShape;
                for (int d = 0; d < dims.nbDims; ++d) {
                    vecShape.push_back(dims.d[d]);  // 20260321 ZJH 逐维记录
                }
                vecShapes.push_back(vecShape);
            }
        }
        return vecShapes;
    }

    // 20260321 ZJH isReady — 检查引擎是否已构建并可以执行推理
    // 返回: 就绪返回 true
    bool isReady() const {
        return m_bReady;
    }

private:
    // 20260321 ZJH allocateBuffers — 为所有输入/输出绑定分配 GPU 显存
    // 返回: 分配成功返回 true
    bool allocateBuffers() {
        int nBindings = m_pEngine->getNbBindings();  // 20260321 ZJH 获取绑定总数
        m_vecBuffers.resize(nBindings, nullptr);  // 20260321 ZJH 初始化缓冲区指针数组

        for (int i = 0; i < nBindings; ++i) {
            auto dims = m_pEngine->getBindingDimensions(i);  // 20260321 ZJH 获取绑定维度
            // 20260321 ZJH 计算绑定的元素总数（各维度之积）
            size_t nVolume = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                nVolume *= static_cast<size_t>(dims.d[d] > 0 ? dims.d[d] : 1);
            }

            // 20260321 ZJH 记录输入和输出的大小（不含 batch 维度的元素数）
            if (m_pEngine->bindingIsInput(i)) {
                m_nInputSize = static_cast<int>(nVolume);  // 20260321 ZJH 记录单个输入的元素数
            } else {
                m_nOutputSize = static_cast<int>(nVolume);  // 20260321 ZJH 记录单个输出的元素数
            }

            // 20260321 ZJH 在 GPU 上分配对应大小的显存
            size_t nBytes = nVolume * sizeof(float);
            cudaMalloc(&m_vecBuffers[i], nBytes);
        }
        return true;
    }

    TRTLogger m_logger;  // 20260321 ZJH TensorRT 内部日志回调实例
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> m_pEngine;   // 20260321 ZJH TensorRT 引擎
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> m_pContext;  // 20260321 ZJH 执行上下文
    std::vector<void*> m_vecBuffers;  // 20260321 ZJH GPU 缓冲区指针数组（输入+输出）
    int m_nInputSize = 0;   // 20260321 ZJH 单个输入的元素数
    int m_nOutputSize = 0;  // 20260321 ZJH 单个输出的元素数
    bool m_bReady = false;  // 20260321 ZJH 引擎是否就绪
};

// =========================================================
// buildTrtEngine — ONNX → TensorRT .engine 一站式转换
// =========================================================

// 20260330 ZJH buildTrtEngine — 从 ONNX 模型文件构建 TensorRT 序列化引擎
// 完整流水线: 解析 ONNX → 创建网络 → 配置精度/优化 → 构建引擎 → 序列化到文件
// strOnnxPath: 输入 ONNX 模型路径（.onnx）
// strEnginePath: 输出 TensorRT 引擎路径（.engine / .trt）
// config: 构建配置（精度、批次、工作空间等）
// 返回: 构建成功返回 true
bool buildTrtEngine(const std::string& strOnnxPath,
                    const std::string& strEnginePath,
                    const TrtBuildConfig& config)
{
    // 20260330 ZJH 验证输入文件存在
    if (!std::filesystem::exists(strOnnxPath)) {
        std::cerr << "[TensorRT] ONNX file not found: " << strOnnxPath << std::endl;
        return false;
    }

    // 20260330 ZJH 根据 verbose 配置设置日志级别
    auto logLevel = config.bVerboseLog
        ? nvinfer1::ILogger::Severity::kINFO
        : nvinfer1::ILogger::Severity::kWARNING;
    TRTLogger logger(logLevel);

    std::cerr << "[TensorRT] Building engine from: " << strOnnxPath << std::endl;
    std::cerr << "[TensorRT] Precision: "
              << (config.ePrecision == TrtBuildConfig::FP32 ? "FP32" :
                  config.ePrecision == TrtBuildConfig::FP16 ? "FP16" : "INT8")
              << ", MaxBatch: " << config.nMaxBatchSize
              << ", Workspace: " << (config.nMaxWorkspaceBytes >> 20) << " MB"
              << std::endl;

    auto tStart = std::chrono::steady_clock::now();  // 20260330 ZJH 记录构建开始时间

    // ------ 步骤 1: 创建 Builder ------
    // 20260330 ZJH 创建 TensorRT builder 实例（引擎构建的入口对象）
    auto pBuilder = std::unique_ptr<nvinfer1::IBuilder, TRTDeleter>(
        nvinfer1::createInferBuilder(logger));
    if (!pBuilder) {
        std::cerr << "[TensorRT] Failed to create builder" << std::endl;
        return false;
    }

    // ------ 步骤 2: 创建网络定义 ------
    // 20260330 ZJH 使用显式批次模式（EXPLICIT_BATCH），ONNX 模型必须使用此模式
    const auto nExplicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto pNetwork = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDeleter>(
        pBuilder->createNetworkV2(nExplicitBatch));
    if (!pNetwork) {
        std::cerr << "[TensorRT] Failed to create network definition" << std::endl;
        return false;
    }

    // ------ 步骤 3: 解析 ONNX 模型 ------
    // 20260330 ZJH 创建 ONNX 解析器并解析模型文件
    auto pParser = std::unique_ptr<nvonnxparser::IParser, TRTDeleter>(
        nvonnxparser::createParser(*pNetwork, logger));
    if (!pParser) {
        std::cerr << "[TensorRT] Failed to create ONNX parser" << std::endl;
        return false;
    }

    // 20260330 ZJH 解析 ONNX 文件，获取网络结构和权重
    if (!pParser->parseFromFile(strOnnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        // 20260330 ZJH 输出所有解析错误信息
        for (int i = 0; i < pParser->getNbErrors(); ++i) {
            std::cerr << "[TensorRT] Parser error: "
                      << pParser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    std::cerr << "[TensorRT] ONNX parsed: "
              << pNetwork->getNbInputs() << " inputs, "
              << pNetwork->getNbOutputs() << " outputs, "
              << pNetwork->getNbLayers() << " layers" << std::endl;

    // ------ 步骤 4: 配置构建选项 ------
    // 20260330 ZJH 创建构建配置对象
    auto pConfig = std::unique_ptr<nvinfer1::IBuilderConfig, TRTDeleter>(
        pBuilder->createBuilderConfig());
    if (!pConfig) {
        std::cerr << "[TensorRT] Failed to create builder config" << std::endl;
        return false;
    }

    // 20260330 ZJH 设置工作空间内存上限
    pConfig->setMaxWorkspaceSize(config.nMaxWorkspaceBytes);

    // 20260330 ZJH 配置精度模式
    if (config.ePrecision == TrtBuildConfig::FP16 || config.ePrecision == TrtBuildConfig::INT8) {
        // 20260330 ZJH FP16: 检查 GPU 是否支持快速半精度运算
        if (pBuilder->platformHasFastFp16()) {
            pConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cerr << "[TensorRT] FP16 enabled" << std::endl;
        } else {
            std::cerr << "[TensorRT] WARNING: GPU does not support fast FP16, falling back to FP32" << std::endl;
        }
    }

    // 20260330 ZJH INT8 校准器（仅 INT8 模式）
    std::unique_ptr<INT8EntropyCalibrator> pCalibrator;  // 20260330 ZJH 校准器生命周期管理
    if (config.ePrecision == TrtBuildConfig::INT8) {
        if (pBuilder->platformHasFastInt8()) {
            pConfig->setFlag(nvinfer1::BuilderFlag::kINT8);

            // 20260330 ZJH 从网络第一个输入推断校准数据尺寸
            auto inputDims = pNetwork->getInput(0)->getDimensions();
            int nInputC = (inputDims.nbDims >= 2) ? inputDims.d[1] : 3;   // 20260330 ZJH 通道数
            int nInputH = (inputDims.nbDims >= 3) ? inputDims.d[2] : 224; // 20260330 ZJH 高度
            int nInputW = (inputDims.nbDims >= 4) ? inputDims.d[3] : 224; // 20260330 ZJH 宽度

            // 20260330 ZJH 创建 INT8 熵校准器
            pCalibrator = std::make_unique<INT8EntropyCalibrator>(
                config.strCalibrationDataDir,
                config.nCalibBatchSize,
                nInputH, nInputW, nInputC,
                config.nCalibMaxBatches);
            pConfig->setInt8Calibrator(pCalibrator.get());

            std::cerr << "[TensorRT] INT8 calibration enabled, data dir: "
                      << config.strCalibrationDataDir << std::endl;
        } else {
            std::cerr << "[TensorRT] WARNING: GPU does not support fast INT8, falling back to FP16/FP32" << std::endl;
        }
    }

    // 20260330 ZJH 严格精度模式（禁止 TRT 自动回退到更高精度）
    if (config.bStrictTypes) {
        pConfig->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        std::cerr << "[TensorRT] Strict type constraints enabled" << std::endl;
    }

    // 20260330 ZJH 配置 DLA 深度学习加速器（Jetson 平台）
    if (config.nDLACore >= 0) {
        if (pBuilder->getNbDLACores() > config.nDLACore) {
            pConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            pConfig->setDLACore(config.nDLACore);
            if (config.bGpuFallback) {
                pConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            }
            std::cerr << "[TensorRT] DLA core " << config.nDLACore << " enabled" << std::endl;
        } else {
            std::cerr << "[TensorRT] WARNING: DLA core " << config.nDLACore
                      << " not available (have " << pBuilder->getNbDLACores() << ")" << std::endl;
        }
    }

    // ------ 步骤 5: 配置动态输入 Profile ------
    // 20260330 ZJH 为动态 batch 维度创建优化 profile
    if (config.nMaxBatchSize > 1) {
        auto pProfile = pBuilder->createOptimizationProfile();
        for (int i = 0; i < pNetwork->getNbInputs(); ++i) {
            auto pInput = pNetwork->getInput(i);
            auto dims = pInput->getDimensions();  // 20260330 ZJH 获取输入维度

            // 20260330 ZJH 检查是否有动态维度（-1 表示动态）
            bool bHasDynamic = false;
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] == -1) { bHasDynamic = true; break; }
            }

            if (bHasDynamic) {
                // 20260330 ZJH 构建 min/opt/max 维度
                nvinfer1::Dims minDims = dims, optDims = dims, maxDims = dims;
                // 20260330 ZJH 假设第 0 维是 batch 维度
                if (dims.d[0] == -1) {
                    minDims.d[0] = 1;                            // 20260330 ZJH 最小批次 = 1
                    optDims.d[0] = config.nOptBatchSize;         // 20260330 ZJH 最优批次
                    maxDims.d[0] = config.nMaxBatchSize;         // 20260330 ZJH 最大批次
                }
                // 20260330 ZJH 其他动态维度保持原值（-1 替换为 1）
                for (int d = 1; d < dims.nbDims; ++d) {
                    if (dims.d[d] == -1) {
                        minDims.d[d] = 1;
                        optDims.d[d] = dims.d[d] > 0 ? dims.d[d] : 224;
                        maxDims.d[d] = dims.d[d] > 0 ? dims.d[d] : 2048;
                    }
                }

                pProfile->setDimensions(pInput->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
                pProfile->setDimensions(pInput->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
                pProfile->setDimensions(pInput->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);

                std::cerr << "[TensorRT] Dynamic profile for '" << pInput->getName()
                          << "': min batch=" << minDims.d[0]
                          << ", opt=" << optDims.d[0]
                          << ", max=" << maxDims.d[0] << std::endl;
            }
        }
        pConfig->addOptimizationProfile(pProfile);
    }

    // ------ 步骤 6: 构建引擎 ------
    // 20260330 ZJH 执行引擎构建（TRT 内部进行层融合、内核选择等优化，耗时较长）
    std::cerr << "[TensorRT] Building optimized engine (this may take several minutes)..." << std::endl;
    pBuilder->setMaxBatchSize(config.nMaxBatchSize);

    auto pEngine = std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter>(
        pBuilder->buildEngineWithConfig(*pNetwork, *pConfig));
    if (!pEngine) {
        std::cerr << "[TensorRT] Engine build FAILED" << std::endl;
        return false;
    }

    // ------ 步骤 7: 序列化并保存 ------
    // 20260330 ZJH 将构建好的引擎序列化为字节流并写入文件
    auto pSerialized = std::unique_ptr<nvinfer1::IHostMemory, TRTDeleter>(
        pEngine->serialize());
    if (!pSerialized) {
        std::cerr << "[TensorRT] Engine serialization FAILED" << std::endl;
        return false;
    }

    // 20260330 ZJH 确保输出目录存在
    std::filesystem::path outputPath(strEnginePath);
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    // 20260330 ZJH 写入序列化引擎到文件
    std::ofstream engineFile(strEnginePath, std::ios::binary);
    if (!engineFile.is_open()) {
        std::cerr << "[TensorRT] Cannot create engine file: " << strEnginePath << std::endl;
        return false;
    }
    engineFile.write(static_cast<const char*>(pSerialized->data()), pSerialized->size());
    engineFile.close();

    // 20260330 ZJH 输出构建摘要
    auto tEnd = std::chrono::steady_clock::now();
    double dElapsedSec = std::chrono::duration<double>(tEnd - tStart).count();
    double dSizeMB = static_cast<double>(pSerialized->size()) / (1024.0 * 1024.0);

    std::cerr << "[TensorRT] Engine built successfully!" << std::endl;
    std::cerr << "[TensorRT]   Output: " << strEnginePath << std::endl;
    std::cerr << "[TensorRT]   Size: " << dSizeMB << " MB" << std::endl;
    std::cerr << "[TensorRT]   Layers: " << pNetwork->getNbLayers()
              << " -> " << pEngine->getNbBindings() << " bindings" << std::endl;
    std::cerr << "[TensorRT]   Build time: " << dElapsedSec << " seconds" << std::endl;

    return true;
}

// =========================================================
// TrtInference — 轻量推理包装器（加载 .engine 并执行推理）
// =========================================================

// 20260330 ZJH TrtInference — 对外推理接口
// 比 TensorRTEngine 更简洁的 API，专注于 "加载 + 推理 + 释放" 三步曲
class TrtInference {
public:
    // 20260330 ZJH 构造函数
    TrtInference() = default;

    // 20260330 ZJH 析构函数：自动释放资源
    ~TrtInference() { release(); }

    // 20260330 ZJH 禁止拷贝（GPU 资源不可拷贝）
    TrtInference(const TrtInference&) = delete;
    TrtInference& operator=(const TrtInference&) = delete;

    // 20260330 ZJH 允许移动
    TrtInference(TrtInference&& other) noexcept
        : m_pEngine(std::move(other.m_pEngine))
        , m_pContext(std::move(other.m_pContext))
        , m_pStream(other.m_pStream)
        , m_vecDeviceBuffers(std::move(other.m_vecDeviceBuffers))
        , m_vecBindingSizes(std::move(other.m_vecBindingSizes))
        , m_vecBindingIsInput(std::move(other.m_vecBindingIsInput))
        , m_bReady(other.m_bReady)
    {
        other.m_pStream = nullptr;
        other.m_bReady = false;
    }

    // 20260330 ZJH loadEngine — 加载已序列化的 TensorRT 引擎文件
    // strEnginePath: .engine / .trt 文件路径
    // 返回: 加载成功返回 true
    bool loadEngine(const std::string& strEnginePath) {
        release();  // 20260330 ZJH 释放之前的引擎（如果有）

        // 20260330 ZJH 读取引擎文件到内存
        std::ifstream file(strEnginePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "[TrtInference] Cannot open: " << strEnginePath << std::endl;
            return false;
        }
        size_t nFileSize = file.tellg();
        if (nFileSize == 0) {
            std::cerr << "[TrtInference] Empty engine file: " << strEnginePath << std::endl;
            return false;
        }
        file.seekg(0, std::ios::beg);
        std::vector<char> vecData(nFileSize);
        file.read(vecData.data(), nFileSize);
        file.close();

        // 20260330 ZJH 创建 TensorRT 运行时并反序列化引擎
        auto pRuntime = std::unique_ptr<nvinfer1::IRuntime, TRTDeleter>(
            nvinfer1::createInferRuntime(m_logger));
        if (!pRuntime) {
            std::cerr << "[TrtInference] Failed to create runtime" << std::endl;
            return false;
        }

        m_pEngine.reset(pRuntime->deserializeCudaEngine(vecData.data(), nFileSize, nullptr));
        if (!m_pEngine) {
            std::cerr << "[TrtInference] Failed to deserialize engine" << std::endl;
            return false;
        }

        // 20260330 ZJH 创建执行上下文
        m_pContext.reset(m_pEngine->createExecutionContext());
        if (!m_pContext) {
            std::cerr << "[TrtInference] Failed to create execution context" << std::endl;
            return false;
        }

        // 20260330 ZJH 创建 CUDA 流（异步推理用）
        cudaStreamCreate(&m_pStream);

        // 20260330 ZJH 分配所有绑定的 GPU 缓冲区
        int nBindings = m_pEngine->getNbBindings();
        m_vecDeviceBuffers.resize(nBindings, nullptr);
        m_vecBindingSizes.resize(nBindings, 0);
        m_vecBindingIsInput.resize(nBindings, false);

        for (int i = 0; i < nBindings; ++i) {
            auto dims = m_pEngine->getBindingDimensions(i);
            // 20260330 ZJH 计算绑定元素总数
            size_t nVolume = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                nVolume *= static_cast<size_t>(dims.d[d] > 0 ? dims.d[d] : 1);
            }
            m_vecBindingSizes[i] = nVolume;
            m_vecBindingIsInput[i] = m_pEngine->bindingIsInput(i);

            // 20260330 ZJH 分配 GPU 显存
            size_t nBytes = nVolume * sizeof(float);
            cudaMalloc(&m_vecDeviceBuffers[i], nBytes);
        }

        m_bReady = true;
        std::cerr << "[TrtInference] Engine loaded: " << nBindings << " bindings" << std::endl;
        return true;
    }

    // 20260330 ZJH infer — 执行推理
    // vecInput: 输入数据（float 向量，CPU 内存，已按 NCHW 排列）
    // vecShape: 输入张量形状（如 {1, 3, 224, 224}）
    // 返回: 推理输出（float 向量），失败返回空
    std::vector<float> infer(const std::vector<float>& vecInput,
                             const std::vector<int64_t>& vecShape) {
        if (!m_bReady) {
            std::cerr << "[TrtInference] Engine not ready" << std::endl;
            return {};
        }

        // 20260330 ZJH 查找第一个输入和第一个输出绑定索引
        int nInputIdx = -1, nOutputIdx = -1;
        for (int i = 0; i < static_cast<int>(m_vecBindingIsInput.size()); ++i) {
            if (m_vecBindingIsInput[i] && nInputIdx < 0)  nInputIdx = i;
            if (!m_vecBindingIsInput[i] && nOutputIdx < 0) nOutputIdx = i;
        }
        if (nInputIdx < 0 || nOutputIdx < 0) {
            std::cerr << "[TrtInference] No input/output binding found" << std::endl;
            return {};
        }

        // 20260330 ZJH 如果输入有动态维度，设置实际输入形状
        if (!vecShape.empty()) {
            nvinfer1::Dims inputDims;
            inputDims.nbDims = static_cast<int>(vecShape.size());
            for (int d = 0; d < inputDims.nbDims; ++d) {
                inputDims.d[d] = static_cast<int>(vecShape[d]);
            }
            m_pContext->setBindingDimensions(nInputIdx, inputDims);
        }

        // 20260330 ZJH 验证输入数据大小与绑定匹配
        size_t nExpectedInput = m_vecBindingSizes[nInputIdx];
        if (vecInput.size() < nExpectedInput) {
            std::cerr << "[TrtInference] Input size mismatch: got " << vecInput.size()
                      << ", expected " << nExpectedInput << std::endl;
            return {};
        }

        // 20260330 ZJH H2D: 将输入数据从 CPU 拷贝到 GPU（异步）
        size_t nInputBytes = nExpectedInput * sizeof(float);
        cudaMemcpyAsync(m_vecDeviceBuffers[nInputIdx], vecInput.data(),
                        nInputBytes, cudaMemcpyHostToDevice, m_pStream);

        // 20260330 ZJH 执行异步推理
        m_pContext->enqueueV2(m_vecDeviceBuffers.data(), m_pStream, nullptr);

        // 20260330 ZJH D2H: 将输出数据从 GPU 拷贝回 CPU（异步）
        size_t nOutputElements = m_vecBindingSizes[nOutputIdx];
        std::vector<float> vecOutput(nOutputElements);
        size_t nOutputBytes = nOutputElements * sizeof(float);
        cudaMemcpyAsync(vecOutput.data(), m_vecDeviceBuffers[nOutputIdx],
                        nOutputBytes, cudaMemcpyDeviceToHost, m_pStream);

        // 20260330 ZJH 同步等待所有异步操作完成
        cudaStreamSynchronize(m_pStream);

        return vecOutput;
    }

    // 20260330 ZJH inferMultiOutput — 执行推理并获取所有输出绑定
    // vecInput: 输入数据
    // vecShape: 输入形状
    // 返回: 每个输出绑定的结果向量
    std::vector<std::vector<float>> inferMultiOutput(
        const std::vector<float>& vecInput,
        const std::vector<int64_t>& vecShape)
    {
        std::vector<std::vector<float>> vecOutputs;
        if (!m_bReady) return vecOutputs;

        // 20260330 ZJH 查找输入绑定
        int nInputIdx = -1;
        for (int i = 0; i < static_cast<int>(m_vecBindingIsInput.size()); ++i) {
            if (m_vecBindingIsInput[i]) { nInputIdx = i; break; }
        }
        if (nInputIdx < 0) return vecOutputs;

        // 20260330 ZJH 设置动态输入形状
        if (!vecShape.empty()) {
            nvinfer1::Dims inputDims;
            inputDims.nbDims = static_cast<int>(vecShape.size());
            for (int d = 0; d < inputDims.nbDims; ++d) {
                inputDims.d[d] = static_cast<int>(vecShape[d]);
            }
            m_pContext->setBindingDimensions(nInputIdx, inputDims);
        }

        // 20260330 ZJH H2D
        size_t nInputBytes = m_vecBindingSizes[nInputIdx] * sizeof(float);
        cudaMemcpyAsync(m_vecDeviceBuffers[nInputIdx], vecInput.data(),
                        nInputBytes, cudaMemcpyHostToDevice, m_pStream);

        // 20260330 ZJH 执行推理
        m_pContext->enqueueV2(m_vecDeviceBuffers.data(), m_pStream, nullptr);

        // 20260330 ZJH D2H: 收集所有输出绑定
        for (int i = 0; i < static_cast<int>(m_vecBindingIsInput.size()); ++i) {
            if (!m_vecBindingIsInput[i]) {
                size_t nElements = m_vecBindingSizes[i];
                std::vector<float> vecOut(nElements);
                cudaMemcpyAsync(vecOut.data(), m_vecDeviceBuffers[i],
                                nElements * sizeof(float), cudaMemcpyDeviceToHost, m_pStream);
                vecOutputs.push_back(std::move(vecOut));
            }
        }

        cudaStreamSynchronize(m_pStream);
        return vecOutputs;
    }

    // 20260330 ZJH release — 释放所有 GPU 资源
    void release() {
        // 20260330 ZJH 释放 GPU 缓冲区
        for (void* pBuf : m_vecDeviceBuffers) {
            if (pBuf) cudaFree(pBuf);
        }
        m_vecDeviceBuffers.clear();
        m_vecBindingSizes.clear();
        m_vecBindingIsInput.clear();

        // 20260330 ZJH 销毁 CUDA 流
        if (m_pStream) {
            cudaStreamDestroy(m_pStream);
            m_pStream = nullptr;
        }

        // 20260330 ZJH 释放 TRT 上下文和引擎
        m_pContext.reset();
        m_pEngine.reset();
        m_bReady = false;
    }

    // 20260330 ZJH isReady — 检查引擎是否已加载并就绪
    bool isReady() const { return m_bReady; }

    // 20260330 ZJH getBindingNames — 获取所有绑定名称及输入/输出类型
    std::vector<std::pair<std::string, bool>> getBindingNames() const {
        std::vector<std::pair<std::string, bool>> vecBindings;
        if (!m_pEngine) return vecBindings;
        for (int i = 0; i < m_pEngine->getNbBindings(); ++i) {
            vecBindings.emplace_back(
                m_pEngine->getBindingName(i),
                m_pEngine->bindingIsInput(i));
        }
        return vecBindings;
    }

    // 20260330 ZJH getBindingShape — 获取指定绑定的维度
    std::vector<int> getBindingShape(int nIndex) const {
        std::vector<int> vecShape;
        if (!m_pEngine || nIndex < 0 || nIndex >= m_pEngine->getNbBindings()) return vecShape;
        auto dims = m_pEngine->getBindingDimensions(nIndex);
        for (int d = 0; d < dims.nbDims; ++d) {
            vecShape.push_back(dims.d[d]);
        }
        return vecShape;
    }

private:
    TRTLogger m_logger;  // 20260330 ZJH 日志器
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> m_pEngine;       // 20260330 ZJH TRT 引擎
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> m_pContext; // 20260330 ZJH 执行上下文
    cudaStream_t m_pStream = nullptr;                  // 20260330 ZJH CUDA 流（异步推理）
    std::vector<void*> m_vecDeviceBuffers;             // 20260330 ZJH GPU 缓冲区指针数组
    std::vector<size_t> m_vecBindingSizes;             // 20260330 ZJH 每个绑定的元素数
    std::vector<bool> m_vecBindingIsInput;             // 20260330 ZJH 每个绑定是否为输入
    bool m_bReady = false;                             // 20260330 ZJH 引擎就绪标志
};

#else  // OM_HAS_TENSORRT 未定义

// =========================================================
// Stub 实现：无 TensorRT SDK 时的空壳类
// =========================================================

// 20260321 ZJH TensorRTEngine — Stub 实现（无 TensorRT SDK）
// 所有方法返回空/false，build/load 时输出提示信息
class TensorRTEngine {
public:
    // 20260321 ZJH 构造函数（Stub）
    TensorRTEngine() = default;

    // 20260321 ZJH build — Stub：ONNX 构建不可用，输出提示并返回 false
    bool build(const TensorRTConfig& config) {
        (void)config;  // 20260321 ZJH 抑制未使用参数警告
        std::cerr << "[TensorRT] TensorRT not available — build skipped" << std::endl;
        return false;  // 20260321 ZJH 无 TensorRT，构建失败
    }

    // 20260321 ZJH loadEngine — Stub：加载不可用，输出提示并返回 false
    bool loadEngine(const std::string& strPath) {
        (void)strPath;  // 20260321 ZJH 抑制未使用参数警告
        std::cerr << "[TensorRT] TensorRT not available — load skipped" << std::endl;
        return false;  // 20260321 ZJH 无 TensorRT，加载失败
    }

    // 20260321 ZJH saveEngine — Stub：保存不可用，返回 false
    bool saveEngine(const std::string& strPath) {
        (void)strPath;  // 20260321 ZJH 抑制未使用参数警告
        return false;  // 20260321 ZJH 无 TensorRT，保存失败
    }

    // 20260321 ZJH infer — Stub：推理不可用，返回空向量
    std::vector<float> infer(const float* pInput, int nBatchSize) {
        (void)pInput;      // 20260321 ZJH 抑制未使用参数警告
        (void)nBatchSize;  // 20260321 ZJH 抑制未使用参数警告
        return {};  // 20260321 ZJH 无 TensorRT，返回空结果
    }

    // 20260321 ZJH getInputNames — Stub：返回空列表
    std::vector<std::string> getInputNames() {
        return {};  // 20260321 ZJH 无 TensorRT，无输入信息
    }

    // 20260321 ZJH getInputShapes — Stub：返回空列表
    std::vector<std::vector<int>> getInputShapes() {
        return {};  // 20260321 ZJH 无 TensorRT，无形状信息
    }

    // 20260321 ZJH isReady — Stub：始终返回 false
    bool isReady() const {
        return false;  // 20260321 ZJH 无 TensorRT，引擎永远不就绪
    }
};

// 20260330 ZJH buildTrtEngine — Stub：TensorRT 不可用，返回 false
// strOnnxPath: ONNX 模型路径（未使用）
// strEnginePath: 输出引擎路径（未使用）
// config: 构建配置（未使用）
inline bool buildTrtEngine(const std::string& strOnnxPath,
                           const std::string& strEnginePath,
                           const TrtBuildConfig& config)
{
    (void)strOnnxPath;   // 20260330 ZJH 抑制未使用参数警告
    (void)strEnginePath; // 20260330 ZJH 抑制未使用参数警告
    (void)config;        // 20260330 ZJH 抑制未使用参数警告
    std::cerr << "[TensorRT] TensorRT SDK not available — buildTrtEngine() skipped" << std::endl;
    std::cerr << "[TensorRT] Install TensorRT and define OM_HAS_TENSORRT to enable" << std::endl;
    return false;
}

// 20260330 ZJH TrtInference — Stub 实现（无 TensorRT SDK）
class TrtInference {
public:
    TrtInference() = default;
    ~TrtInference() = default;

    // 20260330 ZJH loadEngine — Stub：加载不可用
    bool loadEngine(const std::string& strEnginePath) {
        (void)strEnginePath;
        std::cerr << "[TrtInference] TensorRT not available — load skipped" << std::endl;
        return false;
    }

    // 20260330 ZJH infer — Stub：推理不可用
    std::vector<float> infer(const std::vector<float>& vecInput,
                             const std::vector<int64_t>& vecShape) {
        (void)vecInput;
        (void)vecShape;
        return {};
    }

    // 20260330 ZJH inferMultiOutput — Stub：多输出推理不可用
    std::vector<std::vector<float>> inferMultiOutput(
        const std::vector<float>& vecInput,
        const std::vector<int64_t>& vecShape)
    {
        (void)vecInput;
        (void)vecShape;
        return {};
    }

    // 20260330 ZJH release — Stub：无资源需释放
    void release() {}

    // 20260330 ZJH isReady — Stub：始终返回 false
    bool isReady() const { return false; }

    // 20260330 ZJH getBindingNames — Stub：返回空
    std::vector<std::pair<std::string, bool>> getBindingNames() const { return {}; }

    // 20260330 ZJH getBindingShape — Stub：返回空
    std::vector<int> getBindingShape(int nIndex) const { (void)nIndex; return {}; }
};

#endif  // OM_HAS_TENSORRT

// =========================================================
// 辅助函数
// =========================================================

// 20260321 ZJH isTensorRTAvailable — 检查 TensorRT 是否可用
// 返回: 编译时定义了 OM_HAS_TENSORRT 则返回 true，否则 false
inline bool isTensorRTAvailable() {
#ifdef OM_HAS_TENSORRT
    return true;   // 20260321 ZJH TensorRT SDK 已集成
#else
    return false;  // 20260321 ZJH TensorRT SDK 未集成
#endif
}

// 20260321 ZJH getTensorRTVersion — 获取 TensorRT 版本字符串
// 返回: 如 "8.6.1"（来自 NV_TENSORRT_MAJOR/MINOR/PATCH 宏），
//       若 TensorRT 不可用则返回 "N/A"
inline std::string getTensorRTVersion() {
#ifdef OM_HAS_TENSORRT
    // 20260321 ZJH 使用 NvInfer.h 中定义的版本宏拼接版本号
    return std::to_string(NV_TENSORRT_MAJOR) + "."
         + std::to_string(NV_TENSORRT_MINOR) + "."
         + std::to_string(NV_TENSORRT_PATCH);
#else
    return "N/A";  // 20260321 ZJH TensorRT 不可用
#endif
}

// 20260330 ZJH getPrecisionString — 获取精度模式的可读字符串
// ePrecision: 精度枚举值
// 返回: "FP32" / "FP16" / "INT8"
inline std::string getPrecisionString(TrtBuildConfig::Precision ePrecision) {
    switch (ePrecision) {
        case TrtBuildConfig::FP32: return "FP32";  // 20260330 ZJH 全精度
        case TrtBuildConfig::FP16: return "FP16";  // 20260330 ZJH 半精度
        case TrtBuildConfig::INT8: return "INT8";   // 20260330 ZJH 量化
        default: return "Unknown";
    }
}

// 20260330 ZJH estimateEngineSize — 估算 TensorRT 引擎文件大小
// 根据 ONNX 模型文件大小和精度模式估算（经验公式）
// strOnnxPath: ONNX 模型路径
// ePrecision: 目标精度
// 返回: 估算的引擎文件大小（字节），文件不存在返回 0
inline size_t estimateEngineSize(const std::string& strOnnxPath,
                                  TrtBuildConfig::Precision ePrecision) {
    // 20260330 ZJH 获取 ONNX 文件大小
    std::ifstream file(strOnnxPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return 0;
    size_t nOnnxSize = file.tellg();
    file.close();

    // 20260330 ZJH 经验系数: FP32 约 1.2x, FP16 约 0.7x, INT8 约 0.4x ONNX 大小
    double dFactor = 1.2;
    if (ePrecision == TrtBuildConfig::FP16) dFactor = 0.7;
    if (ePrecision == TrtBuildConfig::INT8) dFactor = 0.4;

    return static_cast<size_t>(nOnnxSize * dFactor);
}

}  // namespace om
