#pragma once
// 20260330 ZJH 结构化错误码体系 — 对标海康 MVD_ErrorDefine.h
// 每个错误有唯一编码 + 描述 + 修复建议，替代 bool 返回值
// 按模块分段编码：通用(0x0000) / 模型(0x0001) / 训练(0x0002) / 推理(0x0003) / CUDA(0x0004) / 数据(0x0005)

#include <string>
#include <cstdint>

namespace om {

// 20260330 ZJH 错误码枚举（按模块分段，高16位为模块号，低16位为错误序号）
enum class ErrorCode : uint32_t {
    // ===== 通用 (0x0000xxxx) =====
    OK                       = 0x00000000,  // 20260330 ZJH 操作成功
    InternalError            = 0x00000001,  // 20260330 ZJH 内部未知错误
    InvalidArgument          = 0x00000002,  // 20260330 ZJH 参数无效
    NotImplemented           = 0x00000003,  // 20260330 ZJH 功能尚未实现
    FileNotFound             = 0x00000004,  // 20260330 ZJH 文件未找到
    PermissionDenied         = 0x00000005,  // 20260330 ZJH 权限不足

    // ===== 模型 (0x0001xxxx) =====
    ModelNotCreated          = 0x00010001,  // 20260330 ZJH 模型未创建（需先调用 create/load）
    ModelTypeMismatch        = 0x00010002,  // 20260330 ZJH 模型类型不匹配（如用分类模型做检测）
    ModelLoadFailed          = 0x00010003,  // 20260330 ZJH 模型文件加载失败（格式错误或文件损坏）
    ModelSerializationError  = 0x00010004,  // 20260330 ZJH 模型序列化/反序列化错误
    ModelArchitectureMismatch = 0x00010005, // 20260330 ZJH 模型架构不匹配（权重与网络结构不一致）
    ModelWeightShapeMismatch = 0x00010006,  // 20260330 ZJH 权重张量形状不匹配

    // ===== 训练 (0x0002xxxx) =====
    TrainingDataEmpty        = 0x00020001,  // 20260330 ZJH 训练数据为空（无图像或无标注）
    TrainingCudaOOM          = 0x00020002,  // 20260330 ZJH 训练时 CUDA 显存不足
    TrainingDimensionError   = 0x00020003,  // 20260330 ZJH 训练张量维度错误
    TrainingUserAborted      = 0x00020004,  // 20260330 ZJH 用户手动中止训练
    TrainingNaNLoss          = 0x00020005,  // 20260330 ZJH 训练出现 NaN 损失（学习率过大或数据异常）
    TrainingNoValidation     = 0x00020006,  // 20260330 ZJH 无验证集（需先在 Split 页面拆分数据）
    TrainingAugmentError     = 0x00020007,  // 20260330 ZJH 数据增强处理失败

    // ===== 推理 (0x0003xxxx) =====
    InferenceInputMismatch   = 0x00030001,  // 20260330 ZJH 推理输入尺寸/通道不匹配
    InferenceOutputEmpty     = 0x00030002,  // 20260330 ZJH 推理输出为空
    InferencePostProcessError = 0x00030003, // 20260330 ZJH 后处理解码错误

    // ===== CUDA (0x0004xxxx) =====
    CudaInitFailed           = 0x00040001,  // 20260330 ZJH CUDA 运行时初始化失败
    CudaKernelError          = 0x00040002,  // 20260330 ZJH CUDA kernel 执行错误
    CudaOutOfMemory          = 0x00040003,  // 20260330 ZJH CUDA 显存分配失败
    CudaDeviceNotFound       = 0x00040004,  // 20260330 ZJH 未检测到 CUDA 设备

    // ===== 数据 (0x0005xxxx) =====
    DataFormatError          = 0x00050001,  // 20260330 ZJH 数据格式错误（JSON/XML 解析失败）
    DataCorrupted            = 0x00050002,  // 20260330 ZJH 数据损坏（校验和不匹配）
    DataLabelMissing         = 0x00050003,  // 20260330 ZJH 标签信息缺失
    ImageLoadFailed          = 0x00050004,  // 20260330 ZJH 图像文件加载失败（格式不支持或文件损坏）
};

// 20260330 ZJH 错误信息结构体 — 携带错误码 + 描述 + 上下文 + 修复建议
struct Error {
    ErrorCode eCode = ErrorCode::OK;   // 20260330 ZJH 错误码
    std::string strMessage;            // 20260330 ZJH 人可读错误描述
    std::string strContext;            // 20260330 ZJH 函数名/调用栈上下文
    std::string strSuggestion;         // 20260330 ZJH 修复建议

    // 20260330 ZJH 判断是否成功
    bool ok() const { return eCode == ErrorCode::OK; }

    // 20260330 ZJH 显式 bool 转换，用于 if(error) 判断（true = 有错误）
    explicit operator bool() const { return !ok(); }

    // 20260330 ZJH 工厂方法 — 创建成功结果
    static Error Ok() { return {ErrorCode::OK, "", "", ""}; }

    // 20260330 ZJH 工厂方法 — 创建错误结果
    // e: 错误码
    // msg: 错误描述
    // ctx: 发生错误的函数/位置（可选）
    // sug: 修复建议（可选）
    static Error Make(ErrorCode e, const std::string& msg,
                      const std::string& ctx = "", const std::string& sug = "") {
        return {e, msg, ctx, sug};  // 20260330 ZJH 聚合初始化
    }
};

// 20260330 ZJH 错误码转可读字符串 — 覆盖所有枚举值
// 返回中英文混合描述，便于日志输出和 UI 显示
inline const char* errorCodeToString(ErrorCode e) {
    switch (e) {
        // ===== 通用 =====
        case ErrorCode::OK:
            return "OK (成功)";
        case ErrorCode::InternalError:
            return "Internal Error (内部未知错误)";
        case ErrorCode::InvalidArgument:
            return "Invalid Argument (参数无效)";
        case ErrorCode::NotImplemented:
            return "Not Implemented (功能尚未实现)";
        case ErrorCode::FileNotFound:
            return "File Not Found (文件未找到)";
        case ErrorCode::PermissionDenied:
            return "Permission Denied (权限不足)";

        // ===== 模型 =====
        case ErrorCode::ModelNotCreated:
            return "Model Not Created (模型未创建，请先加载或创建模型)";
        case ErrorCode::ModelTypeMismatch:
            return "Model Type Mismatch (模型类型不匹配，如分类模型不可用于检测)";
        case ErrorCode::ModelLoadFailed:
            return "Model Load Failed (模型加载失败，文件可能损坏或格式不支持)";
        case ErrorCode::ModelSerializationError:
            return "Model Serialization Error (模型序列化/反序列化错误)";
        case ErrorCode::ModelArchitectureMismatch:
            return "Model Architecture Mismatch (模型架构与权重文件不一致)";
        case ErrorCode::ModelWeightShapeMismatch:
            return "Model Weight Shape Mismatch (权重张量形状与层定义不匹配)";

        // ===== 训练 =====
        case ErrorCode::TrainingDataEmpty:
            return "Training Data Empty (训练数据为空，请先导入图像并标注)";
        case ErrorCode::TrainingCudaOOM:
            return "Training CUDA OOM (训练时显存不足，请减小 batch size 或图像尺寸)";
        case ErrorCode::TrainingDimensionError:
            return "Training Dimension Error (训练张量维度不匹配)";
        case ErrorCode::TrainingUserAborted:
            return "Training User Aborted (用户手动中止训练)";
        case ErrorCode::TrainingNaNLoss:
            return "Training NaN Loss (损失值为 NaN，请降低学习率或检查数据)";
        case ErrorCode::TrainingNoValidation:
            return "Training No Validation (无验证集，请在拆分页面划分数据)";
        case ErrorCode::TrainingAugmentError:
            return "Training Augment Error (数据增强处理失败)";

        // ===== 推理 =====
        case ErrorCode::InferenceInputMismatch:
            return "Inference Input Mismatch (推理输入尺寸或通道数不匹配)";
        case ErrorCode::InferenceOutputEmpty:
            return "Inference Output Empty (推理输出为空，模型可能未正确加载)";
        case ErrorCode::InferencePostProcessError:
            return "Inference PostProcess Error (后处理解码失败)";

        // ===== CUDA =====
        case ErrorCode::CudaInitFailed:
            return "CUDA Init Failed (CUDA 运行时初始化失败，请检查驱动版本)";
        case ErrorCode::CudaKernelError:
            return "CUDA Kernel Error (CUDA 核函数执行错误)";
        case ErrorCode::CudaOutOfMemory:
            return "CUDA Out of Memory (CUDA 显存不足，请释放其他 GPU 程序)";
        case ErrorCode::CudaDeviceNotFound:
            return "CUDA Device Not Found (未检测到 CUDA 设备，请检查 GPU 安装)";

        // ===== 数据 =====
        case ErrorCode::DataFormatError:
            return "Data Format Error (数据格式错误，JSON/XML 解析失败)";
        case ErrorCode::DataCorrupted:
            return "Data Corrupted (数据文件损坏，校验和不匹配)";
        case ErrorCode::DataLabelMissing:
            return "Data Label Missing (标签信息缺失，请先创建标签)";
        case ErrorCode::ImageLoadFailed:
            return "Image Load Failed (图像加载失败，格式不支持或文件损坏)";

        default:
            return "Unknown Error (未知错误)";
    }
}

// 20260330 ZJH 获取错误码对应的修复建议
// 为常见错误提供可操作的修复步骤
inline const char* errorCodeSuggestion(ErrorCode e) {
    switch (e) {
        case ErrorCode::OK:
            return "";
        case ErrorCode::InternalError:
            return "请查看日志获取详细堆栈信息，并联系开发人员";
        case ErrorCode::InvalidArgument:
            return "请检查传入参数的类型、范围和格式是否正确";
        case ErrorCode::NotImplemented:
            return "该功能尚在开发中，请等待后续版本更新";
        case ErrorCode::FileNotFound:
            return "请确认文件路径正确且文件存在";
        case ErrorCode::PermissionDenied:
            return "请以管理员权限运行或检查文件/目录访问权限";

        case ErrorCode::ModelNotCreated:
            return "请先通过 Training 或 Import 创建/加载模型";
        case ErrorCode::ModelTypeMismatch:
            return "请确保模型任务类型与当前任务一致（分类/检测/分割等）";
        case ErrorCode::ModelLoadFailed:
            return "请确认 .omm 文件未损坏，版本兼容当前软件";
        case ErrorCode::ModelSerializationError:
            return "请重新导出模型或检查磁盘空间";
        case ErrorCode::ModelArchitectureMismatch:
            return "权重文件与网络定义不一致，请使用匹配的模型文件";
        case ErrorCode::ModelWeightShapeMismatch:
            return "请检查模型输入尺寸和类别数是否与权重文件一致";

        case ErrorCode::TrainingDataEmpty:
            return "请在 Gallery 页面导入图像，并在 Image 页面完成标注";
        case ErrorCode::TrainingCudaOOM:
            return "请减小 Batch Size、降低输入分辨率，或关闭其他 GPU 程序";
        case ErrorCode::TrainingDimensionError:
            return "请检查输入图像尺寸和标注格式是否与模型要求一致";
        case ErrorCode::TrainingUserAborted:
            return "训练已被用户中止，可重新开始训练";
        case ErrorCode::TrainingNaNLoss:
            return "请将学习率降低 10 倍，检查是否存在异常标注或空图像";
        case ErrorCode::TrainingNoValidation:
            return "请在 Split 页面将数据拆分为训练集和验证集";
        case ErrorCode::TrainingAugmentError:
            return "请检查增强参数设置，确保旋转/缩放范围合理";

        case ErrorCode::InferenceInputMismatch:
            return "请确认输入图像的尺寸和通道数与模型训练时一致";
        case ErrorCode::InferenceOutputEmpty:
            return "请确认模型已正确加载且输入数据有效";
        case ErrorCode::InferencePostProcessError:
            return "请检查后处理参数（阈值、NMS 等）设置是否合理";

        case ErrorCode::CudaInitFailed:
            return "请安装兼容的 NVIDIA 驱动，确认 CUDA 版本匹配";
        case ErrorCode::CudaKernelError:
            return "请检查输入数据维度，必要时使用 CPU 后端替代";
        case ErrorCode::CudaOutOfMemory:
            return "请关闭其他 GPU 程序释放显存，或减小 Batch Size";
        case ErrorCode::CudaDeviceNotFound:
            return "请确认系统安装了 NVIDIA GPU 并已正确安装驱动";

        case ErrorCode::DataFormatError:
            return "请确认数据文件格式正确（JSON/XML 语法有效）";
        case ErrorCode::DataCorrupted:
            return "数据文件可能已损坏，请从备份恢复或重新导出";
        case ErrorCode::DataLabelMissing:
            return "请在项目设置中创建标签后再进行标注";
        case ErrorCode::ImageLoadFailed:
            return "请确认图像格式为 BMP/PNG/JPG/TIFF，且文件未损坏";

        default:
            return "请查看日志获取更多信息";
    }
}

// 20260330 ZJH 获取错误码的十六进制字符串表示
// 用于日志输出：如 "0x00020005"
inline std::string errorCodeToHex(ErrorCode e) {
    char buf[12];                            // 20260330 ZJH 缓冲区容纳 "0x" + 8位十六进制 + '\0'
    std::snprintf(buf, sizeof(buf), "0x%08X", static_cast<uint32_t>(e));
    return std::string(buf);                 // 20260330 ZJH 返回格式化后的十六进制字符串
}

// 20260330 ZJH 格式化完整错误报告
// 输出格式: "[0xXXXXXXXX] 错误描述 | 上下文 | 建议"
inline std::string formatError(const Error& err) {
    if (err.ok()) {                          // 20260330 ZJH 成功时返回简单字符串
        return "OK";
    }
    std::string strResult;                   // 20260330 ZJH 结果字符串
    strResult += "[";
    strResult += errorCodeToHex(err.eCode);  // 20260330 ZJH 十六进制错误码
    strResult += "] ";
    strResult += errorCodeToString(err.eCode);  // 20260330 ZJH 错误描述
    if (!err.strMessage.empty()) {           // 20260330 ZJH 附加详细消息
        strResult += " — ";
        strResult += err.strMessage;
    }
    if (!err.strContext.empty()) {            // 20260330 ZJH 附加上下文
        strResult += " | Context: ";
        strResult += err.strContext;
    }
    if (!err.strSuggestion.empty()) {        // 20260330 ZJH 附加修复建议
        strResult += " | Suggestion: ";
        strResult += err.strSuggestion;
    }
    return strResult;
}

}  // namespace om
