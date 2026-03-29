// 20260320 ZJH ONNX 导出模块 — Phase 5
// 将 OmniMatch 模型导出为 ONNX 格式（简化版，不依赖 protobuf）
// 使用 FlatBuffers 风格的自定义二进制写入
module;

#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <unordered_map>

export module om.engine.onnx;

import om.engine.tensor;
import om.engine.module;

export namespace om {

// 20260320 ZJH ONNX 数据类型枚举（对应 onnx.proto TensorProto.DataType）
enum class OnnxDataType : int32_t {
    FLOAT = 1,
    INT32 = 6,
    INT64 = 7
};

// 20260320 ZJH OnnxTensorInfo — ONNX 张量信息
struct OnnxTensorInfo {
    std::string strName;          // 20260320 ZJH 张量名称
    std::vector<int64_t> vecShape;  // 20260320 ZJH 形状
    std::vector<float> vecData;   // 20260320 ZJH 数据（仅初始化器使用）
    OnnxDataType dataType = OnnxDataType::FLOAT;  // 20260320 ZJH 数据类型
};

// 20260320 ZJH OnnxNode — ONNX 计算图节点
struct OnnxNode {
    std::string strOpType;                   // 20260320 ZJH 算子类型 (Conv, Relu, MatMul, etc.)
    std::vector<std::string> vecInputs;      // 20260320 ZJH 输入张量名称
    std::vector<std::string> vecOutputs;     // 20260320 ZJH 输出张量名称
    std::unordered_map<std::string, int64_t> mapIntAttrs;     // 20260320 ZJH 整数属性
    std::unordered_map<std::string, float> mapFloatAttrs;     // 20260320 ZJH 浮点属性
    std::unordered_map<std::string, std::vector<int64_t>> mapIntListAttrs;  // 20260320 ZJH 整数列表属性
};

// 20260320 ZJH OnnxGraph — ONNX 计算图
struct OnnxGraph {
    std::string strName = "omnimatch_model";  // 20260320 ZJH 图名称
    std::vector<OnnxNode> vecNodes;            // 20260320 ZJH 节点列表
    std::vector<OnnxTensorInfo> vecInputs;     // 20260320 ZJH 图输入
    std::vector<OnnxTensorInfo> vecOutputs;    // 20260320 ZJH 图输出
    std::vector<OnnxTensorInfo> vecInitializers;  // 20260320 ZJH 权重初始化器
};

// 20260320 ZJH OnnxExporter — ONNX 模型导出器
// 将 OmniMatch 模型的参数和图结构导出为 ONNX 格式
// 注意：这是简化版本，导出一个自定义的 JSON-like 文本格式
// 而非完整的 protobuf 二进制格式（需要 protobuf 库）
class OnnxExporter {
public:
    // 20260320 ZJH exportToText — 导出为 ONNX 文本描述格式
    // 包含模型元数据和所有权重参数
    // strPath: 输出文件路径（.onnx.txt）
    // model: OmniMatch 模型
    // vecInputShape: 输入张量形状（如 {1, 1, 28, 28}）
    // nNumClasses: 输出类别数
    static bool exportToText(const std::string& strPath,
                              Module& model,
                              const std::vector<int>& vecInputShape,
                              int nNumClasses) {
        std::ofstream file(strPath);
        if (!file.is_open()) return false;

        // 20260320 ZJH 写入文件头
        file << "# OmniMatch ONNX Export\n";
        file << "# Format: OmniMatch ONNX Text v1.0\n";
        file << "# IR Version: 9\n";
        file << "# Opset Version: 18\n\n";

        // 20260320 ZJH 写入输入信息
        file << "[Input]\n";
        file << "name: input\n";
        file << "type: float32\n";
        file << "shape: [";
        for (size_t i = 0; i < vecInputShape.size(); ++i) {
            if (i > 0) file << ", ";
            file << vecInputShape[i];
        }
        file << "]\n\n";

        // 20260320 ZJH 写入输出信息
        file << "[Output]\n";
        file << "name: output\n";
        file << "type: float32\n";
        file << "shape: [1, " << nNumClasses << "]\n\n";

        // 20260320 ZJH 写入模型参数
        auto vecNamedParams = model.namedParameters();
        file << "[Parameters]\n";
        file << "count: " << vecNamedParams.size() << "\n\n";

        for (const auto& [name, pTensor] : vecNamedParams) {
            auto ct = pTensor->contiguous();
            auto vecShape = ct.shapeVec();
            int nNumel = ct.numel();

            file << "  [Param: " << name << "]\n";
            file << "  shape: [";
            for (size_t i = 0; i < vecShape.size(); ++i) {
                if (i > 0) file << ", ";
                file << vecShape[i];
            }
            file << "]\n";
            file << "  numel: " << nNumel << "\n";
            file << "  dtype: float32\n";

            // 20260320 ZJH 写入参数数据（hex 格式）
            file << "  data_hex: ";
            const float* pData = ct.floatDataPtr();
            const unsigned char* pBytes = reinterpret_cast<const unsigned char*>(pData);
            for (int i = 0; i < nNumel * 4; ++i) {
                char hex[3];
                snprintf(hex, sizeof(hex), "%02x", pBytes[i]);
                file << hex;
            }
            file << "\n\n";
        }

        file << "# End of ONNX Export\n";
        return true;
    }

    // 20260320 ZJH exportBinary — 导出为 ONNX 二进制格式（简化版）
    // 自定义二进制格式：Header + Parameters
    // 这不是标准 ONNX protobuf 格式，但可以被自研推理引擎加载
    static bool exportBinary(const std::string& strPath,
                              Module& model,
                              const std::vector<int>& vecInputShape,
                              int nNumClasses) {
        std::ofstream file(strPath, std::ios::binary);
        if (!file.is_open()) return false;

        // 20260320 ZJH 文件魔数和版本
        const char magic[] = "OMONNX01";
        file.write(magic, 8);

        // 20260320 ZJH 写入输入形状
        int32_t nInputDims = static_cast<int32_t>(vecInputShape.size());
        file.write(reinterpret_cast<const char*>(&nInputDims), 4);
        for (auto dim : vecInputShape) {
            int32_t d = static_cast<int32_t>(dim);
            file.write(reinterpret_cast<const char*>(&d), 4);
        }

        // 20260320 ZJH 写入输出信息
        int32_t nOutputClasses = static_cast<int32_t>(nNumClasses);
        file.write(reinterpret_cast<const char*>(&nOutputClasses), 4);

        // 20260320 ZJH 写入参数
        auto vecNamedParams = model.namedParameters();
        int32_t nParamCount = static_cast<int32_t>(vecNamedParams.size());
        file.write(reinterpret_cast<const char*>(&nParamCount), 4);

        for (const auto& [name, pTensor] : vecNamedParams) {
            auto ct = pTensor->contiguous();
            auto vecShape = ct.shapeVec();
            int nNumel = ct.numel();

            // 20260320 ZJH 写入参数名（长度 + 字符串）
            int32_t nNameLen = static_cast<int32_t>(name.size());
            file.write(reinterpret_cast<const char*>(&nNameLen), 4);
            file.write(name.c_str(), nNameLen);

            // 20260320 ZJH 写入形状
            int32_t nDims = static_cast<int32_t>(vecShape.size());
            file.write(reinterpret_cast<const char*>(&nDims), 4);
            for (auto s : vecShape) {
                int32_t d = static_cast<int32_t>(s);
                file.write(reinterpret_cast<const char*>(&d), 4);
            }

            // 20260320 ZJH 写入数据
            int32_t nBytes = static_cast<int32_t>(nNumel * 4);
            file.write(reinterpret_cast<const char*>(&nBytes), 4);
            file.write(reinterpret_cast<const char*>(ct.floatDataPtr()), nBytes);
        }

        return true;
    }
};

}  // namespace om
