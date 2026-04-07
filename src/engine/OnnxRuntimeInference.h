// 20260330 ZJH ONNX Runtime 推理引擎 — 对标 MVTec 的 onnxruntime.dll 集成
// PIMPL 隔离 ONNX Runtime SDK 头文件，避免污染 C++23 module 编译
// 支持 CPU 和 CUDA 两种推理设备，自动选择最优 Execution Provider
// 处理多输出模型（分类 [N,C]、检测 [N,P,5+C]、分割 [N,C,H,W]）
#pragma once

#include <string>       // 20260330 ZJH std::string — 模型路径/张量名称
#include <vector>       // 20260330 ZJH std::vector — 张量数据/形状容器
#include <memory>       // 20260330 ZJH std::unique_ptr — PIMPL 指针
#include <cstdint>      // 20260330 ZJH int64_t — ONNX 张量形状

namespace om {

// 20260330 ZJH 推理设备类型：CPU 通用后端 / CUDA GPU 加速后端
enum class OrtDeviceType {
    CPU,    // 20260330 ZJH 使用 CPU Execution Provider（兼容所有平台）
    CUDA    // 20260330 ZJH 使用 CUDA Execution Provider（需 NVIDIA GPU + CUDA Toolkit）
};

// 20260330 ZJH ONNX 推理配置 — 控制设备选择、线程数、内存优化等
struct OnnxInferConfig {
    OrtDeviceType eDevice = OrtDeviceType::CPU;   // 20260330 ZJH 推理设备（默认 CPU）
    int nCudaDeviceId = 0;                         // 20260330 ZJH CUDA 设备 ID（多卡时选择 GPU）
    int nIntraOpThreads = 4;                       // 20260330 ZJH 算子内并行线程数（CPU 模式有效）
    bool bEnableMemoryPattern = true;              // 20260330 ZJH 启用内存复用模式（减少分配次数）
    bool bEnableGraphOptimization = true;           // 20260330 ZJH 启用图优化（算子融合/常量折叠）

    // 20260402 ZJH [OPT-3.6] 高级 ORT 优化选项
    // 对标 ONNX Runtime Performance Tuning Guide

    // 20260402 ZJH 图优化级别:
    //   0 = ORT_DISABLE_ALL: 禁用所有优化
    //   1 = ORT_ENABLE_BASIC: 基础优化（常量折叠、冗余消除）
    //   2 = ORT_ENABLE_EXTENDED: 扩展优化（算子融合: Conv+BN, MatMul+Add→Gemm）
    //   99 = ORT_ENABLE_ALL: 全部优化（包括布局优化、量化融合等）
    int nGraphOptLevel = 99;

    // 20260402 ZJH 优化后模型缓存路径（非空时保存优化后的 .onnx 到此路径）
    // 首次加载时执行全图优化并缓存，后续加载跳过优化直接读缓存
    // 效果: 模型加载时间从 2-5 秒减少到 <0.5 秒（大模型）
    std::string strOptimizedModelPath;

    // 20260402 ZJH 执行模式:
    //   0 = ORT_SEQUENTIAL: 串行执行（避免多线程开销，适合小模型）
    //   1 = ORT_PARALLEL: 并行执行（适合大模型、多分支网络）
    int nExecutionMode = 0;

    // 20260402 ZJH 启用 IO Binding（GPU 推理时预分配 GPU buffer，避免 CPU↔GPU 拷贝）
    // 效果: 小模型推理延迟减少 20-40%（消除 Host→Device 数据传输）
    bool bEnableIOBinding = false;

    // 20260402 ZJH 启用 TensorRT EP（ONNX Runtime 内置 TensorRT 加速）
    // 优先于 CUDA EP，利用 TensorRT 的算子融合和量化优化
    // 需要系统安装 TensorRT SDK
    bool bEnableTensorRTEP = false;

    // 20260402 ZJH 日志级别: 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    int nLogLevel = 2;
};

// 20260330 ZJH 推理输出 — 单个输出张量的数据和元信息
struct OnnxInferOutput {
    std::vector<float> vecData;           // 20260330 ZJH 输出张量展平浮点数据
    std::vector<int64_t> vecShape;        // 20260330 ZJH 输出张量形状（如 [1,1000] 或 [1,C,H,W]）
    std::string strName;                   // 20260330 ZJH 输出张量名称（对应 ONNX 图输出名）
};

// 20260330 ZJH OnnxRuntimeInference — ONNX Runtime 推理引擎封装
// 提供完整的模型加载→查询→推理→释放生命周期管理
// 内部使用 PIMPL 隔离 onnxruntime_cxx_api.h，不污染外部编译环境
class OnnxRuntimeInference {
public:
    // 20260330 ZJH 构造函数：初始化 PIMPL 实现指针
    OnnxRuntimeInference();

    // 20260330 ZJH 析构函数：释放 ORT Session/Env 资源
    ~OnnxRuntimeInference();

    // 20260330 ZJH 不可拷贝（ORT Session 不可拷贝）
    OnnxRuntimeInference(const OnnxRuntimeInference&) = delete;
    OnnxRuntimeInference& operator=(const OnnxRuntimeInference&) = delete;

    // 20260330 ZJH 可移动（转移 Session 所有权）
    OnnxRuntimeInference(OnnxRuntimeInference&&) noexcept;
    OnnxRuntimeInference& operator=(OnnxRuntimeInference&&) noexcept;

    // 20260330 ZJH 加载 ONNX 模型文件
    // strModelPath: .onnx 文件的绝对路径
    // config: 推理配置（设备/线程/优化选项）
    // 返回: true 表示加载成功，false 表示失败（文件不存在/格式错误/ORT 未启用）
    bool loadModel(const std::string& strModelPath, const OnnxInferConfig& config = {});

    // 20260330 ZJH 模型元信息结构体 — 输入输出名称和形状
    struct ModelInfo {
        std::vector<std::string> vecInputNames;              // 20260330 ZJH 所有输入张量名称
        std::vector<std::string> vecOutputNames;             // 20260330 ZJH 所有输出张量名称
        std::vector<std::vector<int64_t>> vecInputShapes;    // 20260330 ZJH 输入形状（动态维度为 -1）
        std::vector<std::vector<int64_t>> vecOutputShapes;   // 20260330 ZJH 输出形状（动态维度为 -1）
        int nNumInputs = 0;                                   // 20260330 ZJH 输入张量总数
        int nNumOutputs = 0;                                  // 20260330 ZJH 输出张量总数
    };

    // 20260330 ZJH 查询已加载模型的输入输出信息
    // 返回: ModelInfo 结构体（模型未加载时返回空信息）
    ModelInfo getModelInfo() const;

    // 20260330 ZJH 单张图像推理
    // vecInput: 输入数据（CHW 格式，已归一化为 [0,1] 或 ImageNet 标准化）
    // vecInputShape: 输入张量形状（如 {1, 3, 224, 224}）
    // 返回: 所有输出张量的数据和形状
    std::vector<OnnxInferOutput> infer(const std::vector<float>& vecInput,
                                        const std::vector<int64_t>& vecInputShape);

    // 20260330 ZJH 批量推理 — 多张图像打包为一个批量张量
    // vecBatchInput: 展平的批量数据 [B*C*H*W]
    // vecBatchShape: 批量张量形状（如 {4, 3, 224, 224}）
    // 返回: 所有输出张量的数据和形状（batch 维度保留在输出中）
    std::vector<OnnxInferOutput> inferBatch(const std::vector<float>& vecBatchInput,
                                             const std::vector<int64_t>& vecBatchShape);

    // 20260330 ZJH 释放模型资源（Session + Env）
    void release();

    // 20260330 ZJH 检查模型是否已加载
    // 返回: true 表示 Session 已创建且可用
    bool isLoaded() const;

    // 20260330 ZJH 预热推理引擎（消除首次推理的冷启动延迟）
    // GPU 场景下首次推理需分配显存/编译 kernel，预热可将延迟转移到初始化阶段
    // nRounds: 预热轮次（每轮执行一次全零输入推理）
    // 返回: true 表示预热成功
    bool warmup(int nRounds = 3);

    // 20260330 ZJH 获取最近一次错误信息
    // 返回: 错误描述字符串（无错误时返回空）
    std::string getLastError() const;

private:
    // 20260330 ZJH PIMPL 前向声明（实现在 .cpp 中，隔离 ORT 头文件）
    struct Impl;
    std::unique_ptr<Impl> m_pImpl;  // 20260330 ZJH 实现指针
};

}  // namespace om
