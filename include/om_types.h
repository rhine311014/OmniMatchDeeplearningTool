// 20260319 ZJH OmniMatch 全局公共类型定义
// 所有层共享的枚举、错误码、Result 别名等基础类型
#pragma once

#include <expected>
#include <string>
#include <cstdint>

namespace om {

// 20260319 ZJH 设备类型枚举 — 区分计算后端（CPU / CUDA / OpenCL）
enum class DeviceType {
    CPU = 0,   // CPU 后端（AVX2/NEON SIMD + OpenMP）
    CUDA,      // NVIDIA GPU（CUDA kernel）
    OpenCL     // AMD/Intel GPU（OpenCL kernel）
};

// 20260319 ZJH 张量数据类型 — 从一开始支持多类型，避免后期混合精度时大规模重构
enum class DataType {
    Float32 = 0,  // 默认训练精度
    Float16,      // 混合精度训练 / 推理优化
    Int32,        // 索引 / 标签
    Int64,        // 大规模索引
    UInt8          // 图像原始数据
};

// 20260319 ZJH 错误码枚举 — 覆盖张量、GPU、IO、训练四大类错误
enum class ErrorCode {
    Success = 0,
    // 张量错误
    ShapeMismatch,       // 形状不匹配（如矩阵乘法维度不一致）
    DTypeMismatch,       // 数据类型不匹配
    DeviceMismatch,      // 设备不匹配（如 CPU 张量与 GPU 张量运算）
    // GPU 错误
    CudaError,           // CUDA API 调用失败
    OpenCLError,         // OpenCL API 调用失败
    OutOfMemory,         // 显存或内存不足
    // IO 错误
    FileNotFound,        // 文件不存在
    InvalidFormat,       // 文件格式错误
    SerializationError,  // 序列化 / 反序列化失败
    // 训练错误
    NaNDetected,         // 训练过程中出现 NaN
    GradientExplosion,   // 梯度爆炸（梯度范数超过阈值）
    // 通用错误
    InvalidArgument,     // 参数非法
    InternalError        // 内部错误
};

// 20260319 ZJH 错误信息结构体 — 携带错误码、描述信息和发生位置
struct Error {
    ErrorCode code;         // 错误码
    std::string strMessage; // 可读错误描述
    std::string strFile;    // 发生错误的源文件
    int nLine = 0;          // 发生错误的行号
};

// 20260319 ZJH Result 别名 — 使用 C++23 std::expected 作为主要错误传播机制
// 成功时持有 T，失败时持有 Error，避免异常开销
template<typename T>
using Result = std::expected<T, Error>;

// 20260319 ZJH 便捷宏 — 在当前位置创建 Error 对象
#define OM_ERROR(code, msg) \
    om::Error{ om::ErrorCode::code, msg, __FILE__, __LINE__ }

// 20260319 ZJH 返回值类型别名 — DataType 对应的字节大小
inline size_t dataTypeSize(DataType dtype) {
    // 20260319 ZJH 根据数据类型返回单个元素占用的字节数
    switch (dtype) {
        case DataType::Float32: return 4;  // 32 位浮点
        case DataType::Float16: return 2;  // 16 位半精度浮点
        case DataType::Int32:   return 4;  // 32 位整数
        case DataType::Int64:   return 8;  // 64 位整数
        case DataType::UInt8:   return 1;  // 8 位无符号整数
        default:                return 0;  // 未知类型返回 0
    }
}

}  // namespace om
