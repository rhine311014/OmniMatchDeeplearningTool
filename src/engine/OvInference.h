#pragma once
// 20260330 ZJH OpenVINO CPU 推理后端 — 对标 MVTec openvino 集成
// Intel CPU 优化推理，支持 ONNX/IR 模型格式
// PIMPL 隔离 OpenVINO SDK 头文件，避免污染 C++23 module 编译
// 处理多输出模型（分类 [N,C]、检测 [N,P,5+C]、分割 [N,C,H,W]）
// API 设计与 OnnxRuntimeInference 保持一致，便于后端切换

#include <string>       // 20260330 ZJH std::string — 模型路径/张量名称
#include <vector>       // 20260330 ZJH std::vector — 张量数据/形状容器
#include <memory>       // 20260330 ZJH std::unique_ptr — PIMPL 指针
#include <cstdint>      // 20260330 ZJH int64_t — 张量形状维度类型

namespace om {

// 20260330 ZJH OpenVINO 推理配置 — 控制设备选择、线程数、动态批量等
struct OvInferConfig {
    int nNumThreads = 0;              // 20260330 ZJH 推理线程数（0=自动，由 OpenVINO 根据 CPU 核心数决定）
    bool bEnableDynamicBatch = false;  // 20260330 ZJH 启用动态批量（允许运行时改变 batch 维度）
    std::string strDevice = "CPU";     // 20260330 ZJH 推理设备（"CPU" / "GPU" / "AUTO"）
    bool bEnableCaching = false;       // 20260330 ZJH 启用模型缓存（加速二次加载）
    std::string strCacheDir;           // 20260330 ZJH 模型缓存目录路径（bEnableCaching=true 时生效）
};

// 20260330 ZJH OpenVINO 推理输出 — 单个输出张量的数据和元信息
struct OvInferOutput {
    std::vector<float> vecData;        // 20260330 ZJH 输出张量展平浮点数据
    std::vector<int64_t> vecShape;     // 20260330 ZJH 输出张量形状（如 [1,1000] 或 [1,C,H,W]）
    std::string strName;                // 20260330 ZJH 输出张量名称（对应模型图输出名）
};

// 20260330 ZJH OpenVINOInference — OpenVINO 推理引擎封装
// 提供完整的模型加载→查询→推理→释放生命周期管理
// 内部使用 PIMPL 隔离 openvino/openvino.hpp，不污染外部编译环境
// 支持 ONNX (.onnx) 和 OpenVINO IR (.xml/.bin) 两种模型格式
class OpenVINOInference {
public:
    // 20260330 ZJH 构造函数：初始化 PIMPL 实现指针
    OpenVINOInference();

    // 20260330 ZJH 析构函数：释放 CompiledModel/Core 资源
    ~OpenVINOInference();

    // 20260330 ZJH 不可拷贝（CompiledModel 不可拷贝）
    OpenVINOInference(const OpenVINOInference&) = delete;
    OpenVINOInference& operator=(const OpenVINOInference&) = delete;

    // 20260330 ZJH 可移动（转移 CompiledModel 所有权）
    OpenVINOInference(OpenVINOInference&&) noexcept;
    OpenVINOInference& operator=(OpenVINOInference&&) noexcept;

    // 20260330 ZJH 加载模型文件（ONNX 或 OpenVINO IR 格式）
    // strModelPath: .onnx 或 .xml 文件的绝对路径
    //   - ONNX: 直接传入 .onnx 文件路径
    //   - IR:   传入 .xml 文件路径（.bin 权重文件需与 .xml 同目录同名）
    // config: 推理配置（设备/线程/缓存选项）
    // 返回: true 表示加载成功，false 表示失败（文件不存在/格式错误/OpenVINO 未启用）
    bool loadModel(const std::string& strModelPath, const OvInferConfig& config = {});

    // 20260330 ZJH 单张图像推理
    // vecInput: 输入数据（CHW 格式，已归一化为 [0,1] 或 ImageNet 标准化）
    // vecShape: 输入张量形状（如 {1, 3, 224, 224}）
    // 返回: 所有输出张量的数据和形状
    std::vector<OvInferOutput> infer(const std::vector<float>& vecInput,
                                      const std::vector<int64_t>& vecShape);

    // 20260330 ZJH 批量推理 — 多张图像打包为一个批量张量
    // vecBatchInput: 展平的批量数据 [B*C*H*W]
    // vecBatchShape: 批量张量形状（如 {4, 3, 224, 224}）
    // 返回: 所有输出张量的数据和形状（batch 维度保留在输出中）
    std::vector<OvInferOutput> inferBatch(const std::vector<float>& vecBatchInput,
                                           const std::vector<int64_t>& vecBatchShape);

    // 20260330 ZJH 检查模型是否已加载
    // 返回: true 表示 CompiledModel 已创建且可用
    bool isLoaded() const;

    // 20260330 ZJH 释放模型资源（CompiledModel + Core）
    void release();

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

    // 20260330 ZJH 预热推理引擎（消除首次推理的编译/优化延迟）
    // OpenVINO 首次推理需完成图优化和 kernel 编译，预热可将延迟转移到初始化阶段
    // nRounds: 预热轮次（每轮执行一次全零输入推理）
    // 返回: true 表示预热成功
    bool warmup(int nRounds = 3);

    // 20260330 ZJH 获取最近一次错误信息
    // 返回: 错误描述字符串（无错误时返回空）
    std::string getLastError() const;

private:
    // 20260330 ZJH PIMPL 前向声明（实现在 .cpp 中，隔离 OpenVINO 头文件）
    struct Impl;
    std::unique_ptr<Impl> m_pImpl;  // 20260330 ZJH 实现指针
};

}  // namespace om
