// 20260321 ZJH TensorRT 推理引擎集成模块
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

#ifdef OM_HAS_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

export module om.engine.tensorrt;

export namespace om {

// =========================================================
// TensorRTConfig — TensorRT 推理配置
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
// 条件编译：OM_HAS_TENSORRT 真实实现
// =========================================================

#ifdef OM_HAS_TENSORRT

// 20260321 ZJH TRTLogger — TensorRT 内部日志回调
// 实现 nvinfer1::ILogger 接口，将 TensorRT 内部消息转发到标准错误输出
class TRTLogger : public nvinfer1::ILogger {
public:
    // 20260321 ZJH log — 接收 TensorRT 内部日志
    // severity: 日志级别（kERROR, kWARNING, kINFO, kVERBOSE）
    // msg: 日志消息内容
    void log(Severity severity, const char* msg) noexcept override {
        // 20260321 ZJH 仅输出警告及以上级别的日志，减少冗余输出
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;  // 20260321 ZJH 输出到标准错误流
        }
    }
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

}  // namespace om
