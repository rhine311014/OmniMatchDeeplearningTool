// 20260330 ZJH OpenVINO 推理引擎实现
// 条件编译：OM_HAS_OPENVINO 宏控制是否启用真实 OpenVINO 调用
// 未启用时所有接口返回失败/空，不影响编译和链接
// API 实现与 OnnxRuntimeInference.cpp 保持一致风格

#include "OvInference.h"  // 20260330 ZJH 自身头文件

#include <algorithm>    // 20260330 ZJH std::copy — 复制输出数据
#include <numeric>      // 20260330 ZJH std::accumulate — 计算张量元素总数
#include <stdexcept>    // 20260330 ZJH std::runtime_error — 异常处理
#include <iostream>     // 20260330 ZJH std::cerr — 错误日志输出
#include <cassert>      // 20260330 ZJH assert — 调试断言

// 20260330 ZJH 条件包含 OpenVINO C++ API 头文件
// 仅在 CMake 配置 OM_ENABLE_OPENVINO=ON 且找到 openvino 包时定义此宏
#ifdef OM_HAS_OPENVINO
#include <openvino/openvino.hpp>  // 20260330 ZJH OpenVINO 统一 C++ API
#endif

namespace om {

// ============================================================================
// PIMPL 实现结构体
// ============================================================================

struct OpenVINOInference::Impl {

#ifdef OM_HAS_OPENVINO
    // ---- OpenVINO 启用时的完整实现 ----

    // 20260330 ZJH OpenVINO Core 对象（管理设备和插件加载）
    std::unique_ptr<ov::Core> pCore;

    // 20260330 ZJH 编译后的模型（已优化，绑定到特定设备）
    std::unique_ptr<ov::CompiledModel> pCompiledModel;

    // 20260330 ZJH 推理请求（封装输入/输出张量和推理调用）
    std::unique_ptr<ov::InferRequest> pInferRequest;

    // 20260330 ZJH 缓存的输入/输出名称
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
    std::vector<OvInferOutput> runInference(const std::vector<float>& vecInputData,
                                             const std::vector<int64_t>& vecShape) {
        // 20260330 ZJH 初始化空结果
        std::vector<OvInferOutput> vecResults;

        // 20260330 ZJH 检查模型是否已加载
        if (!bLoaded || !pInferRequest) {
            strLastError = "Model not loaded";  // 20260330 ZJH 模型未加载
            return vecResults;
        }

        try {
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

            // 20260330 ZJH 将 int64_t 形状转换为 ov::Shape (size_t)
            ov::Shape ovShape;
            ovShape.reserve(vecShape.size());  // 20260330 ZJH 预分配
            for (auto dim : vecShape) {
                ovShape.push_back(static_cast<size_t>(dim));  // 20260330 ZJH int64_t -> size_t
            }

            // 20260330 ZJH 创建输入张量（不复制数据，直接引用外部内存）
            ov::Tensor inputTensor(
                ov::element::f32,                                 // 20260330 ZJH 数据类型为 float32
                ovShape,                                          // 20260330 ZJH 张量形状
                const_cast<float*>(vecInputData.data())           // 20260330 ZJH 数据指针（OpenVINO 要求非 const）
            );

            // 20260330 ZJH 设置输入张量到推理请求
            pInferRequest->set_input_tensor(inputTensor);

            // 20260330 ZJH 执行同步推理
            pInferRequest->infer();

            // 20260330 ZJH 提取所有输出张量
            size_t nNumOutputs = pCompiledModel->outputs().size();  // 20260330 ZJH 输出数量
            for (size_t i = 0; i < nNumOutputs; ++i) {
                OvInferOutput output;  // 20260330 ZJH 当前输出结构体

                // 20260330 ZJH 设置输出名称
                if (i < vecOutputNames.size()) {
                    output.strName = vecOutputNames[i];  // 20260330 ZJH 使用缓存的名称
                }

                // 20260330 ZJH 获取输出张量
                ov::Tensor outputTensor = pInferRequest->get_output_tensor(i);

                // 20260330 ZJH 获取输出形状
                auto outputShape = outputTensor.get_shape();
                for (auto dim : outputShape) {
                    output.vecShape.push_back(static_cast<int64_t>(dim));  // 20260330 ZJH size_t -> int64_t
                }

                // 20260330 ZJH 获取输出元素总数
                size_t nElementCount = outputTensor.get_size();

                // 20260330 ZJH 提取浮点数据
                const float* pOutputData = outputTensor.data<float>();
                output.vecData.assign(pOutputData, pOutputData + nElementCount);

                // 20260330 ZJH 添加到结果列表
                vecResults.push_back(std::move(output));
            }

            // 20260330 ZJH 清空错误信息（推理成功）
            strLastError.clear();

        } catch (const ov::Exception& e) {
            // 20260330 ZJH 捕获 OpenVINO 特定异常（模型不兼容/输入形状错误等）
            strLastError = std::string("OpenVINO error: ") + e.what();
            std::cerr << "[OpenVINOInference] " << strLastError << std::endl;
        } catch (const std::exception& e) {
            // 20260330 ZJH 捕获标准异常（内存不足等）
            strLastError = std::string("Inference error: ") + e.what();
            std::cerr << "[OpenVINOInference] " << strLastError << std::endl;
        }

        return vecResults;  // 20260330 ZJH 返回推理结果（可能为空）
    }

#else
    // ---- OpenVINO 未启用时的存根实现 ----

    // 20260330 ZJH 模型加载状态（始终 false）
    bool bLoaded = false;

    // 20260330 ZJH 最近一次错误信息
    std::string strLastError = "OpenVINO not available: build with OM_ENABLE_OPENVINO=ON";

#endif  // OM_HAS_OPENVINO
};

// ============================================================================
// 构造/析构/移动
// ============================================================================

// 20260330 ZJH 构造函数：创建 PIMPL 实现
OpenVINOInference::OpenVINOInference()
    : m_pImpl(std::make_unique<Impl>())  // 20260330 ZJH 分配实现对象
{
}

// 20260330 ZJH 析构函数：释放 PIMPL（默认析构即可，但需在 .cpp 中定义以见到 Impl 完整类型）
OpenVINOInference::~OpenVINOInference() = default;

// 20260330 ZJH 移动构造函数：转移 PIMPL 所有权
OpenVINOInference::OpenVINOInference(OpenVINOInference&&) noexcept = default;

// 20260330 ZJH 移动赋值运算符：转移 PIMPL 所有权
OpenVINOInference& OpenVINOInference::operator=(OpenVINOInference&&) noexcept = default;

// ============================================================================
// loadModel — 加载 ONNX 或 OpenVINO IR 模型
// ============================================================================

bool OpenVINOInference::loadModel(const std::string& strModelPath,
                                   const OvInferConfig& config) {
#ifdef OM_HAS_OPENVINO
    try {
        // 20260330 ZJH 释放旧的 CompiledModel（如果存在）
        release();

        // 20260330 ZJH 创建 OpenVINO Core 对象（管理设备插件）
        m_pImpl->pCore = std::make_unique<ov::Core>();

        // 20260330 ZJH 配置模型缓存（加速二次加载，OpenVINO 将编译后模型缓存到磁盘）
        if (config.bEnableCaching && !config.strCacheDir.empty()) {
            m_pImpl->pCore->set_property(ov::cache_dir(config.strCacheDir));
            std::cerr << "[OpenVINOInference] Model cache enabled: " << config.strCacheDir << std::endl;
        }

        // 20260330 ZJH 读取模型（支持 ONNX 和 IR 格式，OpenVINO 自动识别）
        std::shared_ptr<ov::Model> pModel = m_pImpl->pCore->read_model(strModelPath);

        // 20260330 ZJH 配置推理属性
        ov::AnyMap mapProperties;  // 20260330 ZJH 属性键值映射

        // 20260330 ZJH 设置推理线程数（0=自动）
        if (config.nNumThreads > 0) {
            mapProperties[ov::inference_num_threads.name()] = config.nNumThreads;
        }

        // 20260330 ZJH 编译模型到目标设备
        // OpenVINO 在此阶段完成图优化、算子融合、内存规划
        auto compiledModel = m_pImpl->pCore->compile_model(
            pModel,                 // 20260330 ZJH 原始模型
            config.strDevice,       // 20260330 ZJH 目标设备（"CPU"/"GPU"/"AUTO"）
            mapProperties           // 20260330 ZJH 推理属性
        );

        m_pImpl->pCompiledModel = std::make_unique<ov::CompiledModel>(std::move(compiledModel));

        // 20260330 ZJH 创建推理请求（封装输入/输出缓冲区和推理调度）
        auto inferRequest = m_pImpl->pCompiledModel->create_infer_request();
        m_pImpl->pInferRequest = std::make_unique<ov::InferRequest>(std::move(inferRequest));

        // 20260330 ZJH 提取输入张量信息
        m_pImpl->vecInputNames.clear();     // 20260330 ZJH 清空旧名称
        m_pImpl->vecInputShapes.clear();    // 20260330 ZJH 清空旧形状

        for (const auto& input : pModel->inputs()) {
            // 20260330 ZJH 获取输入名称
            std::string strName = input.get_any_name();  // 20260330 ZJH any_name 优先返回友好名
            m_pImpl->vecInputNames.push_back(strName);

            // 20260330 ZJH 获取输入形状（动态维度标记为 -1）
            std::vector<int64_t> vecShape;
            auto partialShape = input.get_partial_shape();  // 20260330 ZJH 可能含动态维度
            if (partialShape.is_static()) {
                // 20260330 ZJH 静态形状：直接转换
                auto staticShape = partialShape.get_shape();
                for (auto dim : staticShape) {
                    vecShape.push_back(static_cast<int64_t>(dim));
                }
            } else {
                // 20260330 ZJH 动态形状：逐维检查
                for (size_t d = 0; d < partialShape.size(); ++d) {
                    if (partialShape[d].is_static()) {
                        vecShape.push_back(static_cast<int64_t>(partialShape[d].get_length()));
                    } else {
                        vecShape.push_back(-1);  // 20260330 ZJH 动态维度标记为 -1
                    }
                }
            }
            m_pImpl->vecInputShapes.push_back(vecShape);
        }

        // 20260330 ZJH 提取输出张量信息
        m_pImpl->vecOutputNames.clear();    // 20260330 ZJH 清空旧名称
        m_pImpl->vecOutputShapes.clear();   // 20260330 ZJH 清空旧形状

        for (const auto& output : pModel->outputs()) {
            // 20260330 ZJH 获取输出名称
            std::string strName = output.get_any_name();
            m_pImpl->vecOutputNames.push_back(strName);

            // 20260330 ZJH 获取输出形状
            std::vector<int64_t> vecShape;
            auto partialShape = output.get_partial_shape();
            if (partialShape.is_static()) {
                auto staticShape = partialShape.get_shape();
                for (auto dim : staticShape) {
                    vecShape.push_back(static_cast<int64_t>(dim));
                }
            } else {
                for (size_t d = 0; d < partialShape.size(); ++d) {
                    if (partialShape[d].is_static()) {
                        vecShape.push_back(static_cast<int64_t>(partialShape[d].get_length()));
                    } else {
                        vecShape.push_back(-1);  // 20260330 ZJH 动态维度标记
                    }
                }
            }
            m_pImpl->vecOutputShapes.push_back(vecShape);
        }

        // 20260330 ZJH 标记为已加载
        m_pImpl->bLoaded = true;
        m_pImpl->strLastError.clear();  // 20260330 ZJH 清空错误信息

        // 20260330 ZJH 输出加载成功日志
        std::cerr << "[OpenVINOInference] Model loaded: " << strModelPath
                  << " | device=" << config.strDevice
                  << " | inputs=" << m_pImpl->vecInputNames.size()
                  << " | outputs=" << m_pImpl->vecOutputNames.size() << std::endl;

        return true;  // 20260330 ZJH 加载成功

    } catch (const ov::Exception& e) {
        // 20260330 ZJH 捕获 OpenVINO 异常（文件不存在/格式不支持/设备不可用等）
        m_pImpl->strLastError = std::string("OpenVINO load error: ") + e.what();
        std::cerr << "[OpenVINOInference] " << m_pImpl->strLastError << std::endl;
        m_pImpl->bLoaded = false;
        return false;  // 20260330 ZJH 加载失败
    } catch (const std::exception& e) {
        // 20260330 ZJH 捕获标准异常
        m_pImpl->strLastError = std::string("Load error: ") + e.what();
        std::cerr << "[OpenVINOInference] " << m_pImpl->strLastError << std::endl;
        m_pImpl->bLoaded = false;
        return false;  // 20260330 ZJH 加载失败
    }

#else
    // 20260330 ZJH OpenVINO 未启用：输出提示信息并返回失败
    (void)strModelPath;  // 20260330 ZJH 避免未使用参数警告
    (void)config;        // 20260330 ZJH 避免未使用参数警告
    m_pImpl->strLastError = "OpenVINO not available. "
                            "Rebuild with -DOM_ENABLE_OPENVINO=ON and install OpenVINO toolkit. "
                            "Download from: https://github.com/openvinotoolkit/openvino/releases";
    std::cerr << "[OpenVINOInference] " << m_pImpl->strLastError << std::endl;
    return false;  // 20260330 ZJH OpenVINO 不可用
#endif
}

// ============================================================================
// getModelInfo — 查询模型元信息
// ============================================================================

OpenVINOInference::ModelInfo OpenVINOInference::getModelInfo() const {
    ModelInfo info;  // 20260330 ZJH 初始化空信息

#ifdef OM_HAS_OPENVINO
    // 20260330 ZJH 检查模型是否已加载
    if (m_pImpl->bLoaded) {
        info.vecInputNames = m_pImpl->vecInputNames;       // 20260330 ZJH 复制输入名称
        info.vecOutputNames = m_pImpl->vecOutputNames;     // 20260330 ZJH 复制输出名称
        info.vecInputShapes = m_pImpl->vecInputShapes;     // 20260330 ZJH 复制输入形状
        info.vecOutputShapes = m_pImpl->vecOutputShapes;   // 20260330 ZJH 复制输出形状
        info.nNumInputs = static_cast<int>(m_pImpl->vecInputNames.size());    // 20260330 ZJH 输入数量
        info.nNumOutputs = static_cast<int>(m_pImpl->vecOutputNames.size());  // 20260330 ZJH 输出数量
    }
#endif

    return info;  // 20260330 ZJH 返回模型信息（未加载时为空）
}

// ============================================================================
// infer — 单张图像推理
// ============================================================================

std::vector<OvInferOutput> OpenVINOInference::infer(
    const std::vector<float>& vecInput,
    const std::vector<int64_t>& vecShape)
{
#ifdef OM_HAS_OPENVINO
    // 20260330 ZJH 委托给内部通用推理方法
    return m_pImpl->runInference(vecInput, vecShape);
#else
    // 20260330 ZJH OpenVINO 未启用：返回空结果
    (void)vecInput;   // 20260330 ZJH 避免未使用参数警告
    (void)vecShape;   // 20260330 ZJH 避免未使用参数警告
    return {};
#endif
}

// ============================================================================
// inferBatch — 批量推理
// ============================================================================

std::vector<OvInferOutput> OpenVINOInference::inferBatch(
    const std::vector<float>& vecBatchInput,
    const std::vector<int64_t>& vecBatchShape)
{
#ifdef OM_HAS_OPENVINO
    // 20260330 ZJH 批量推理与单张推理逻辑相同
    // OpenVINO 模型输入本身支持 batch 维度，只需传入 [B, C, H, W] 形状
    return m_pImpl->runInference(vecBatchInput, vecBatchShape);
#else
    // 20260330 ZJH OpenVINO 未启用：返回空结果
    (void)vecBatchInput;   // 20260330 ZJH 避免未使用参数警告
    (void)vecBatchShape;   // 20260330 ZJH 避免未使用参数警告
    return {};
#endif
}

// ============================================================================
// release — 释放模型资源
// ============================================================================

void OpenVINOInference::release() {
#ifdef OM_HAS_OPENVINO
    // 20260330 ZJH 释放顺序：先 InferRequest 后 CompiledModel 后 Core
    // InferRequest 持有 CompiledModel 内部引用，必须先释放
    m_pImpl->pInferRequest.reset();     // 20260330 ZJH 释放推理请求
    m_pImpl->pCompiledModel.reset();    // 20260330 ZJH 释放编译后模型
    m_pImpl->pCore.reset();             // 20260330 ZJH 释放 Core 对象
    m_pImpl->vecInputNames.clear();     // 20260330 ZJH 清空输入名称缓存
    m_pImpl->vecOutputNames.clear();    // 20260330 ZJH 清空输出名称缓存
    m_pImpl->vecInputShapes.clear();    // 20260330 ZJH 清空输入形状缓存
    m_pImpl->vecOutputShapes.clear();   // 20260330 ZJH 清空输出形状缓存
#endif
    m_pImpl->bLoaded = false;  // 20260330 ZJH 标记为未加载
}

// ============================================================================
// isLoaded — 检查模型加载状态
// ============================================================================

bool OpenVINOInference::isLoaded() const {
    return m_pImpl->bLoaded;  // 20260330 ZJH 返回加载标志
}

// ============================================================================
// warmup — 预热推理引擎
// ============================================================================

bool OpenVINOInference::warmup(int nRounds) {
#ifdef OM_HAS_OPENVINO
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
            std::cerr << "[OpenVINOInference] Warmup round " << (i + 1)
                      << "/" << nRounds << " failed: " << m_pImpl->strLastError << std::endl;
            return false;  // 20260330 ZJH 预热失败
        }
    }

    std::cerr << "[OpenVINOInference] Warmup completed: " << nRounds << " rounds" << std::endl;
    return true;  // 20260330 ZJH 预热成功

#else
    // 20260330 ZJH OpenVINO 未启用：返回失败
    (void)nRounds;
    m_pImpl->strLastError = "OpenVINO not available";
    return false;
#endif
}

// ============================================================================
// getLastError — 获取最近一次错误信息
// ============================================================================

std::string OpenVINOInference::getLastError() const {
    return m_pImpl->strLastError;  // 20260330 ZJH 返回错误信息
}

}  // namespace om
