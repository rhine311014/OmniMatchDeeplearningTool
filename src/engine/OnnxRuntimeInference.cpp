// 20260330 ZJH ONNX Runtime 推理引擎实现
// 条件编译：OM_HAS_ONNXRUNTIME 宏控制是否启用真实 ORT 调用
// 未启用时所有接口返回失败/空，不影响编译和链接

#include "OnnxRuntimeInference.h"  // 20260330 ZJH 自身头文件

#include <algorithm>    // 20260330 ZJH std::copy — 复制输出数据
#include <numeric>      // 20260330 ZJH std::accumulate — 计算张量元素总数
#include <stdexcept>    // 20260330 ZJH std::runtime_error — 异常处理
#include <iostream>     // 20260330 ZJH std::cerr — 错误日志输出
#include <cassert>      // 20260330 ZJH assert — 调试断言

// 20260330 ZJH 条件包含 ONNX Runtime C++ API 头文件
// 仅在 CMake 配置 OM_ENABLE_ONNXRUNTIME=ON 且找到 onnxruntime 包时定义此宏
#ifdef OM_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>  // 20260330 ZJH ONNX Runtime C++ 高级 API
#endif

namespace om {

// ============================================================================
// PIMPL 实现结构体
// ============================================================================

struct OnnxRuntimeInference::Impl {

#ifdef OM_HAS_ONNXRUNTIME
    // ---- ORT 启用时的完整实现 ----

    // 20260330 ZJH ORT 全局环境（进程生命周期，线程安全）
    std::unique_ptr<Ort::Env> pEnv;

    // 20260330 ZJH ORT 推理会话（封装已加载的模型）
    std::unique_ptr<Ort::Session> pSession;

    // 20260330 ZJH ORT 内存分配器（用于获取输入/输出张量名称）
    Ort::AllocatorWithDefaultOptions allocator;

    // 20260330 ZJH 缓存的输入/输出名称（C 字符串指针）
    std::vector<std::string> vecInputNames;   // 20260330 ZJH 输入张量名称列表
    std::vector<std::string> vecOutputNames;  // 20260330 ZJH 输出张量名称列表

    // 20260330 ZJH 缓存的输入/输出形状
    std::vector<std::vector<int64_t>> vecInputShapes;   // 20260330 ZJH 输入形状列表
    std::vector<std::vector<int64_t>> vecOutputShapes;  // 20260330 ZJH 输出形状列表

    // 20260330 ZJH 模型是否已加载标志
    bool bLoaded = false;

    // 20260330 ZJH 最近一次错误信息
    std::string strLastError;

    // 20260330 ZJH 执行推理（核心方法，单次/批量共用）
    // vecInputData: 展平的浮点输入数据
    // vecShape: 输入张量形状
    // 返回: 所有输出张量列表
    std::vector<OnnxInferOutput> runInference(const std::vector<float>& vecInputData,
                                               const std::vector<int64_t>& vecShape) {
        // 20260330 ZJH 初始化空结果
        std::vector<OnnxInferOutput> vecResults;

        // 20260330 ZJH 检查模型是否已加载
        if (!bLoaded || !pSession) {
            strLastError = "Model not loaded";  // 20260330 ZJH 模型未加载
            return vecResults;
        }

        try {
            // 20260330 ZJH 创建 ORT 内存信息（CPU 分配，供 CreateTensor 使用）
            auto memInfo = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,      // 20260330 ZJH 使用 Arena 内存池
                OrtMemTypeDefault       // 20260330 ZJH 默认内存类型
            );

            // 20260330 ZJH 计算输入元素总数
            int64_t nInputSize = 1;  // 20260330 ZJH 初始化为 1，逐维累乘
            for (auto dim : vecShape) {
                nInputSize *= dim;  // 20260330 ZJH 累乘每个维度
            }

            // 20260330 ZJH 验证输入数据长度与形状一致
            if (static_cast<int64_t>(vecInputData.size()) != nInputSize) {
                strLastError = "Input data size mismatch: expected " +
                               std::to_string(nInputSize) + ", got " +
                               std::to_string(vecInputData.size());
                return vecResults;  // 20260330 ZJH 数据长度不匹配，返回空
            }

            // 20260330 ZJH 创建输入张量（不复制数据，直接引用外部内存）
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memInfo,                                          // 20260330 ZJH 内存信息
                const_cast<float*>(vecInputData.data()),          // 20260330 ZJH 数据指针（ORT API 要求非 const）
                static_cast<size_t>(nInputSize),                  // 20260330 ZJH 元素总数
                vecShape.data(),                                  // 20260330 ZJH 形状指针
                vecShape.size()                                   // 20260330 ZJH 维度数
            );

            // 20260330 ZJH 构建输入/输出名称的 C 字符串数组（ORT API 要求 const char*[]）
            std::vector<const char*> vecInputCStrs;   // 20260330 ZJH 输入名称 C 指针
            std::vector<const char*> vecOutputCStrs;  // 20260330 ZJH 输出名称 C 指针
            for (const auto& name : vecInputNames) {
                vecInputCStrs.push_back(name.c_str());   // 20260330 ZJH 转换为 C 字符串
            }
            for (const auto& name : vecOutputNames) {
                vecOutputCStrs.push_back(name.c_str());  // 20260330 ZJH 转换为 C 字符串
            }

            // 20260330 ZJH 执行推理（同步阻塞调用）
            auto outputTensors = pSession->Run(
                Ort::RunOptions{nullptr},                          // 20260330 ZJH 默认运行选项
                vecInputCStrs.data(),                              // 20260330 ZJH 输入名称数组
                &inputTensor,                                      // 20260330 ZJH 输入张量数组
                1,                                                 // 20260330 ZJH 输入张量数量（单输入）
                vecOutputCStrs.data(),                             // 20260330 ZJH 输出名称数组
                vecOutputCStrs.size()                              // 20260330 ZJH 输出张量数量
            );

            // 20260330 ZJH 遍历所有输出张量，提取数据和形状
            for (size_t i = 0; i < outputTensors.size(); ++i) {
                OnnxInferOutput output;  // 20260330 ZJH 当前输出结构体

                // 20260330 ZJH 设置输出名称
                if (i < vecOutputNames.size()) {
                    output.strName = vecOutputNames[i];  // 20260330 ZJH 使用缓存的名称
                }

                // 20260330 ZJH 获取输出张量的类型和形状信息
                auto typeInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
                auto outputShape = typeInfo.GetShape();     // 20260330 ZJH 输出形状
                auto nElementCount = typeInfo.GetElementCount();  // 20260330 ZJH 元素总数

                // 20260330 ZJH 保存形状信息
                output.vecShape.assign(outputShape.begin(), outputShape.end());

                // 20260330 ZJH 提取浮点数据
                const float* pOutputData = outputTensors[i].GetTensorData<float>();
                output.vecData.assign(pOutputData, pOutputData + nElementCount);

                // 20260330 ZJH 添加到结果列表
                vecResults.push_back(std::move(output));
            }

            // 20260330 ZJH 清空错误信息（推理成功）
            strLastError.clear();

        } catch (const Ort::Exception& e) {
            // 20260330 ZJH 捕获 ORT 特定异常（模型不兼容/输入形状错误等）
            strLastError = std::string("ONNX Runtime error: ") + e.what();
            std::cerr << "[OnnxRuntimeInference] " << strLastError << std::endl;
        } catch (const std::exception& e) {
            // 20260330 ZJH 捕获标准异常（内存不足等）
            strLastError = std::string("Inference error: ") + e.what();
            std::cerr << "[OnnxRuntimeInference] " << strLastError << std::endl;
        }

        return vecResults;  // 20260330 ZJH 返回推理结果（可能为空）
    }

#else
    // ---- ORT 未启用时的存根实现 ----

    // 20260330 ZJH 模型加载状态（始终 false）
    bool bLoaded = false;

    // 20260330 ZJH 最近一次错误信息
    std::string strLastError = "ONNX Runtime not available: build with OM_ENABLE_ONNXRUNTIME=ON";

#endif  // OM_HAS_ONNXRUNTIME
};

// ============================================================================
// 构造/析构/移动
// ============================================================================

// 20260330 ZJH 构造函数：创建 PIMPL 实现
OnnxRuntimeInference::OnnxRuntimeInference()
    : m_pImpl(std::make_unique<Impl>())  // 20260330 ZJH 分配实现对象
{
}

// 20260330 ZJH 析构函数：释放 PIMPL（默认析构即可，但需在 .cpp 中定义以见到 Impl 完整类型）
OnnxRuntimeInference::~OnnxRuntimeInference() = default;

// 20260330 ZJH 移动构造函数：转移 PIMPL 所有权
OnnxRuntimeInference::OnnxRuntimeInference(OnnxRuntimeInference&&) noexcept = default;

// 20260330 ZJH 移动赋值运算符：转移 PIMPL 所有权
OnnxRuntimeInference& OnnxRuntimeInference::operator=(OnnxRuntimeInference&&) noexcept = default;

// ============================================================================
// loadModel — 加载 ONNX 模型
// ============================================================================

bool OnnxRuntimeInference::loadModel(const std::string& strModelPath,
                                      const OnnxInferConfig& config) {
#ifdef OM_HAS_ONNXRUNTIME
    try {
        // 20260330 ZJH 释放旧的 Session（如果存在）
        release();

        // 20260330 ZJH 创建 ORT 全局环境
        // 日志级别 WARNING：仅输出警告和错误，不输出 verbose 信息
        m_pImpl->pEnv = std::make_unique<Ort::Env>(
            ORT_LOGGING_LEVEL_WARNING,  // 20260330 ZJH 日志级别
            "OmniMatchOrtInference"     // 20260330 ZJH 日志标签
        );

        // 20260330 ZJH 配置会话选项
        Ort::SessionOptions sessionOptions;

        // 20260330 ZJH 设置算子内并行线程数（CPU 密集型运算的并行度）
        sessionOptions.SetIntraOpNumThreads(config.nIntraOpThreads);

        // 20260330 ZJH 启用内存复用模式（减少重复分配，适合多次推理）
        if (config.bEnableMemoryPattern) {
            sessionOptions.EnableMemPattern();  // 20260330 ZJH 启用内存模式
        }

        // 20260330 ZJH 启用图优化（算子融合/常量折叠/布局转换）
        if (config.bEnableGraphOptimization) {
            sessionOptions.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL  // 20260330 ZJH 启用所有优化
            );
        } else {
            sessionOptions.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_DISABLE_ALL  // 20260330 ZJH 禁用所有优化
            );
        }

        // 20260330 ZJH 根据设备配置添加 Execution Provider
        if (config.eDevice == OrtDeviceType::CUDA) {
            // 20260330 ZJH 尝试添加 CUDA Execution Provider
            OrtCUDAProviderOptions cudaOptions;                          // 20260330 ZJH CUDA 提供者选项
            cudaOptions.device_id = config.nCudaDeviceId;                // 20260330 ZJH GPU 设备 ID
            cudaOptions.arena_extend_strategy = 0;                       // 20260330 ZJH Arena 扩展策略 (kNextPowerOfTwo)
            cudaOptions.gpu_mem_limit = SIZE_MAX;                        // 20260330 ZJH GPU 内存限制（不限制）
            cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;  // 20260330 ZJH cuDNN 卷积算法搜索（穷举搜索最优）
            cudaOptions.do_copy_in_default_stream = 1;                   // 20260330 ZJH 使用默认 CUDA 流进行数据拷贝

            // 20260330 ZJH 添加 CUDA 加速提供者（失败时自动回退到 CPU）
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            std::cerr << "[OnnxRuntimeInference] CUDA EP requested, device_id="
                      << config.nCudaDeviceId << std::endl;
        }
        // 20260330 ZJH CPU Execution Provider 始终作为 fallback 自动添加，无需显式调用

        // 20260330 ZJH 将 UTF-8 路径转换为宽字符串（Windows 平台 ORT 要求 wstring）
#ifdef _WIN32
        // 20260330 ZJH Windows: std::string -> std::wstring (MultiByteToWideChar)
        int nWideLen = MultiByteToWideChar(CP_UTF8, 0,
                                            strModelPath.c_str(),
                                            static_cast<int>(strModelPath.size()),
                                            nullptr, 0);
        std::wstring wstrModelPath(nWideLen, L'\0');  // 20260330 ZJH 预分配宽字符串
        MultiByteToWideChar(CP_UTF8, 0,
                            strModelPath.c_str(),
                            static_cast<int>(strModelPath.size()),
                            wstrModelPath.data(), nWideLen);

        // 20260330 ZJH 创建推理会话（加载模型、解析图结构、分配内存）
        m_pImpl->pSession = std::make_unique<Ort::Session>(
            *m_pImpl->pEnv,        // 20260330 ZJH 全局环境
            wstrModelPath.c_str(), // 20260330 ZJH 模型路径（宽字符串）
            sessionOptions         // 20260330 ZJH 会话选项
        );
#else
        // 20260330 ZJH Linux/macOS: 直接使用 UTF-8 路径
        m_pImpl->pSession = std::make_unique<Ort::Session>(
            *m_pImpl->pEnv,          // 20260330 ZJH 全局环境
            strModelPath.c_str(),    // 20260330 ZJH 模型路径
            sessionOptions           // 20260330 ZJH 会话选项
        );
#endif

        // 20260330 ZJH 提取输入张量信息
        size_t nNumInputs = m_pImpl->pSession->GetInputCount();  // 20260330 ZJH 输入数量
        m_pImpl->vecInputNames.clear();     // 20260330 ZJH 清空旧名称
        m_pImpl->vecInputShapes.clear();    // 20260330 ZJH 清空旧形状

        for (size_t i = 0; i < nNumInputs; ++i) {
            // 20260330 ZJH 获取输入名称（ORT 返回 AllocatedStringPtr，需转为 std::string）
            auto namePtr = m_pImpl->pSession->GetInputNameAllocated(i, m_pImpl->allocator);
            m_pImpl->vecInputNames.emplace_back(namePtr.get());  // 20260330 ZJH 缓存名称

            // 20260330 ZJH 获取输入形状信息
            auto typeInfo = m_pImpl->pSession->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();  // 20260330 ZJH 动态维度为 -1

            m_pImpl->vecInputShapes.push_back(shape);  // 20260330 ZJH 缓存形状
        }

        // 20260330 ZJH 提取输出张量信息
        size_t nNumOutputs = m_pImpl->pSession->GetOutputCount();  // 20260330 ZJH 输出数量
        m_pImpl->vecOutputNames.clear();    // 20260330 ZJH 清空旧名称
        m_pImpl->vecOutputShapes.clear();   // 20260330 ZJH 清空旧形状

        for (size_t i = 0; i < nNumOutputs; ++i) {
            // 20260330 ZJH 获取输出名称
            auto namePtr = m_pImpl->pSession->GetOutputNameAllocated(i, m_pImpl->allocator);
            m_pImpl->vecOutputNames.emplace_back(namePtr.get());  // 20260330 ZJH 缓存名称

            // 20260330 ZJH 获取输出形状信息
            auto typeInfo = m_pImpl->pSession->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();  // 20260330 ZJH 动态维度为 -1

            m_pImpl->vecOutputShapes.push_back(shape);  // 20260330 ZJH 缓存形状
        }

        // 20260330 ZJH 标记为已加载
        m_pImpl->bLoaded = true;
        m_pImpl->strLastError.clear();  // 20260330 ZJH 清空错误信息

        // 20260330 ZJH 输出加载成功日志
        std::cerr << "[OnnxRuntimeInference] Model loaded: " << strModelPath
                  << " | inputs=" << nNumInputs
                  << " | outputs=" << nNumOutputs << std::endl;

        return true;  // 20260330 ZJH 加载成功

    } catch (const Ort::Exception& e) {
        // 20260330 ZJH 捕获 ORT 异常（文件不存在/格式不支持/opset 不兼容等）
        m_pImpl->strLastError = std::string("ORT load error: ") + e.what();
        std::cerr << "[OnnxRuntimeInference] " << m_pImpl->strLastError << std::endl;
        m_pImpl->bLoaded = false;
        return false;  // 20260330 ZJH 加载失败
    } catch (const std::exception& e) {
        // 20260330 ZJH 捕获标准异常
        m_pImpl->strLastError = std::string("Load error: ") + e.what();
        std::cerr << "[OnnxRuntimeInference] " << m_pImpl->strLastError << std::endl;
        m_pImpl->bLoaded = false;
        return false;  // 20260330 ZJH 加载失败
    }

#else
    // 20260330 ZJH ORT 未启用：输出提示信息并返回失败
    (void)strModelPath;  // 20260330 ZJH 避免未使用参数警告
    (void)config;        // 20260330 ZJH 避免未使用参数警告
    m_pImpl->strLastError = "ONNX Runtime not available. "
                            "Rebuild with -DOM_ENABLE_ONNXRUNTIME=ON and install onnxruntime package. "
                            "Download from: https://github.com/microsoft/onnxruntime/releases";
    std::cerr << "[OnnxRuntimeInference] " << m_pImpl->strLastError << std::endl;
    return false;  // 20260330 ZJH ORT 不可用
#endif
}

// ============================================================================
// getModelInfo — 查询模型元信息
// ============================================================================

OnnxRuntimeInference::ModelInfo OnnxRuntimeInference::getModelInfo() const {
    ModelInfo info;  // 20260330 ZJH 初始化空信息

#ifdef OM_HAS_ONNXRUNTIME
    // 20260330 ZJH 检查模型是否已加载
    if (m_pImpl->bLoaded) {
        info.vecInputNames = m_pImpl->vecInputNames;     // 20260330 ZJH 复制输入名称
        info.vecOutputNames = m_pImpl->vecOutputNames;   // 20260330 ZJH 复制输出名称
        info.vecInputShapes = m_pImpl->vecInputShapes;   // 20260330 ZJH 复制输入形状
        info.vecOutputShapes = m_pImpl->vecOutputShapes; // 20260330 ZJH 复制输出形状
        info.nNumInputs = static_cast<int>(m_pImpl->vecInputNames.size());    // 20260330 ZJH 输入数量
        info.nNumOutputs = static_cast<int>(m_pImpl->vecOutputNames.size());  // 20260330 ZJH 输出数量
    }
#endif

    return info;  // 20260330 ZJH 返回模型信息（未加载时为空）
}

// ============================================================================
// infer — 单张图像推理
// ============================================================================

std::vector<OnnxInferOutput> OnnxRuntimeInference::infer(
    const std::vector<float>& vecInput,
    const std::vector<int64_t>& vecInputShape)
{
#ifdef OM_HAS_ONNXRUNTIME
    // 20260330 ZJH 委托给内部通用推理方法
    return m_pImpl->runInference(vecInput, vecInputShape);
#else
    // 20260330 ZJH ORT 未启用：返回空结果
    (void)vecInput;       // 20260330 ZJH 避免未使用参数警告
    (void)vecInputShape;  // 20260330 ZJH 避免未使用参数警告
    return {};
#endif
}

// ============================================================================
// inferBatch — 批量推理
// ============================================================================

std::vector<OnnxInferOutput> OnnxRuntimeInference::inferBatch(
    const std::vector<float>& vecBatchInput,
    const std::vector<int64_t>& vecBatchShape)
{
#ifdef OM_HAS_ONNXRUNTIME
    // 20260330 ZJH 批量推理与单张推理逻辑相同
    // ONNX 模型的输入本身支持 batch 维度，只需传入 [B, C, H, W] 形状
    return m_pImpl->runInference(vecBatchInput, vecBatchShape);
#else
    // 20260330 ZJH ORT 未启用：返回空结果
    (void)vecBatchInput;  // 20260330 ZJH 避免未使用参数警告
    (void)vecBatchShape;  // 20260330 ZJH 避免未使用参数警告
    return {};
#endif
}

// ============================================================================
// release — 释放模型资源
// ============================================================================

void OnnxRuntimeInference::release() {
#ifdef OM_HAS_ONNXRUNTIME
    // 20260330 ZJH 释放顺序：先 Session 后 Env（Session 持有 Env 内部引用）
    m_pImpl->pSession.reset();      // 20260330 ZJH 释放推理会话
    m_pImpl->pEnv.reset();          // 20260330 ZJH 释放全局环境
    m_pImpl->vecInputNames.clear();   // 20260330 ZJH 清空输入名称缓存
    m_pImpl->vecOutputNames.clear();  // 20260330 ZJH 清空输出名称缓存
    m_pImpl->vecInputShapes.clear();  // 20260330 ZJH 清空输入形状缓存
    m_pImpl->vecOutputShapes.clear(); // 20260330 ZJH 清空输出形状缓存
#endif
    m_pImpl->bLoaded = false;  // 20260330 ZJH 标记为未加载
}

// ============================================================================
// isLoaded — 检查模型加载状态
// ============================================================================

bool OnnxRuntimeInference::isLoaded() const {
    return m_pImpl->bLoaded;  // 20260330 ZJH 返回加载标志
}

// ============================================================================
// warmup — 预热推理引擎
// ============================================================================

bool OnnxRuntimeInference::warmup(int nRounds) {
#ifdef OM_HAS_ONNXRUNTIME
    // 20260330 ZJH 检查模型是否已加载
    if (!m_pImpl->bLoaded || m_pImpl->vecInputShapes.empty()) {
        m_pImpl->strLastError = "Cannot warmup: model not loaded";
        return false;  // 20260330 ZJH 模型未加载，无法预热
    }

    // 20260330 ZJH 构造全零输入数据用于预热
    // 将动态维度（-1）替换为 1，确保形状合法
    auto warmupShape = m_pImpl->vecInputShapes[0];  // 20260330 ZJH 取第一个输入的形状
    int64_t nTotalElements = 1;  // 20260330 ZJH 元素总数

    for (auto& dim : warmupShape) {
        if (dim <= 0) {
            dim = 1;  // 20260330 ZJH 将动态维度替换为 1
        }
        nTotalElements *= dim;  // 20260330 ZJH 累乘
    }

    // 20260330 ZJH 创建全零输入向量
    std::vector<float> vecWarmupData(static_cast<size_t>(nTotalElements), 0.0f);

    // 20260330 ZJH 执行 nRounds 轮预热推理
    for (int i = 0; i < nRounds; ++i) {
        auto results = m_pImpl->runInference(vecWarmupData, warmupShape);
        // 20260330 ZJH 检查推理是否成功
        if (results.empty() && !m_pImpl->strLastError.empty()) {
            std::cerr << "[OnnxRuntimeInference] Warmup round " << (i + 1)
                      << "/" << nRounds << " failed: " << m_pImpl->strLastError << std::endl;
            return false;  // 20260330 ZJH 预热失败
        }
    }

    std::cerr << "[OnnxRuntimeInference] Warmup completed: " << nRounds << " rounds" << std::endl;
    return true;  // 20260330 ZJH 预热成功

#else
    // 20260330 ZJH ORT 未启用：返回失败
    (void)nRounds;
    m_pImpl->strLastError = "ONNX Runtime not available";
    return false;
#endif
}

// ============================================================================
// getLastError — 获取最近一次错误信息
// ============================================================================

std::string OnnxRuntimeInference::getLastError() const {
    return m_pImpl->strLastError;  // 20260330 ZJH 返回错误信息
}

}  // namespace om
