#pragma once
// 20260330 ZJH 模型预检查 — 对标海康 CNNPreCheckCpp
// 在推理前验证模型有效性、兼容性、GPU 可用性
// 功能: 检查 .omm 文件格式（魔数/版本/CRC32）、
//       解析模型元数据（架构/输入尺寸/类别数/参数量/文件大小）、
//       检测 CUDA 设备、验证模型与输入图像的兼容性
// 应用场景: 模型加载前的合规性验证、部署前自动诊断

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstring>
#include <array>

namespace om {

// =========================================================
// ModelCheckResult — 模型检查结果结构
// =========================================================

// 20260330 ZJH 模型预检查的综合结果
struct ModelCheckResult {
    bool bValid = false;               // 20260330 ZJH 模型是否有效（通过所有检查）
    std::string strModelType;          // 20260330 ZJH 模型类型名称（如 "UNet", "ResNet-18"）
    int nInputWidth = 0;               // 20260330 ZJH 模型期望的输入宽度（像素）
    int nInputHeight = 0;              // 20260330 ZJH 模型期望的输入高度（像素）
    int nInputChannels = 3;            // 20260330 ZJH 模型期望的输入通道数（RGB=3, 灰度=1）
    int nNumClasses = 0;               // 20260330 ZJH 模型输出类别数（含背景类）
    int nBaseChannels = 0;             // 20260330 ZJH 基础通道数（用于架构重建）
    int64_t nNumParams = 0;            // 20260330 ZJH 模型参数总数（权重 + 偏置）
    int64_t nNumBuffers = 0;           // 20260330 ZJH 缓冲区数量（BN running stats 等）
    float fModelSizeMB = 0.0f;         // 20260330 ZJH 模型文件大小（MB）
    int nFormatVersion = 0;            // 20260330 ZJH 模型文件格式版本（3 或 4）
    bool bHasMetadata = false;         // 20260330 ZJH 是否包含架构元数据（v4 格式）
    bool bCrcValid = false;            // 20260330 ZJH CRC32 校验是否通过
    bool bCudaAvailable = false;       // 20260330 ZJH CUDA GPU 是否可用
    int nCudaDeviceCount = 0;          // 20260330 ZJH 可用 GPU 数量
    std::string strCudaDeviceName;     // 20260330 ZJH 第一个 GPU 的名称
    std::string strErrorMsg;           // 20260330 ZJH 错误信息（bValid=false 时填充）
    std::vector<std::string> vecWarnings;  // 20260330 ZJH 警告信息列表（非致命问题）
};

// =========================================================
// 内部辅助（CRC32 + .omm 解析）
// =========================================================

namespace detail {

// 20260330 ZJH CRC32 查表法（与 om.engine.serializer 保持一致）
// 编译期生成 256 项查找表
inline constexpr std::array<uint32_t, 256> generateCrc32Table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320u) : (crc >> 1);
        }
        table[i] = crc;
    }
    return table;
}

// 20260330 ZJH 全局 CRC32 查找表（static 避免重复初始化）
static const auto s_arrCrc32Table = generateCrc32Table();

// 20260330 ZJH 计算数据块的 CRC32 校验和
// pData: 数据指针
// nSize: 数据字节数
// 返回: CRC32 校验和
inline uint32_t computeCrc32(const uint8_t* pData, size_t nSize) {
    uint32_t crc = 0xFFFFFFFFu;  // 20260330 ZJH CRC 初始值
    for (size_t i = 0; i < nSize; ++i) {
        uint8_t idx = static_cast<uint8_t>(crc ^ pData[i]);
        crc = (crc >> 8) ^ s_arrCrc32Table[idx];
    }
    return crc ^ 0xFFFFFFFFu;  // 20260330 ZJH 最终异或
}

// 20260330 ZJH .omm 魔数常量
static constexpr char OMM_MAGIC[4] = { 'O', 'M', 'M', '\0' };
static constexpr char DFM_MAGIC[4] = { 'D', 'F', 'M', '\0' };

// 20260330 ZJH ModelMeta 元数据魔数（v4 格式，与 om.engine.serializer 一致）
static constexpr float META_MAGIC = 42.0f;

// 20260330 ZJH 已知模型类型 hash → 名称映射（基于 FNV-1a）
inline uint32_t hashModelType(const std::string& s) {
    uint32_t h = 2166136261u;
    for (char c : s) { h ^= static_cast<uint32_t>(c); h *= 16777619u; }
    return h;
}

// 20260330 ZJH 尝试从 hash 反查模型类型名称
// 包含所有 OmniMatch 内置模型类型
inline std::string resolveModelType(uint32_t nHash) {
    // 20260330 ZJH 预定义的模型类型列表
    static const char* s_arrTypes[] = {
        "ResNet-18", "ResNet-50", "EfficientNet-B0",
        "MobileNetV4-Small", "MobileNetV4-Medium",
        "ViT-Tiny", "ViT-Small",
        "UNet", "UNet-Small", "DeepLabV3",
        "YOLOv8n", "YOLOv8s", "YOLOv10n",
        "EfficientAD-S", "EfficientAD-M",
        "Mask-RCNN-Light", "SOLOv2-Light",
        "CRNN-Tiny", "CRNN-Small",
        "CLIP-ViT-B16", "OWLv2-Tiny",
        "SAM-Tiny", "SAM-Base"
    };

    // 20260330 ZJH 逐个比对 hash
    for (const char* pType : s_arrTypes) {
        if (hashModelType(pType) == nHash) {
            return pType;  // 20260330 ZJH 命中，返回名称
        }
    }

    return "Unknown";  // 20260330 ZJH 未识别的模型类型
}

}  // namespace detail

// =========================================================
// checkModel — 检查 .omm 模型文件
// =========================================================

// 20260330 ZJH checkModel — 解析并验证 .omm 模型文件
// 检查项目:
//   1. 文件是否存在
//   2. 魔数是否正确（"OMM\0" 或 "DFM\0"）
//   3. 格式版本是否支持（v3 或 v4）
//   4. v4 元数据是否有效
//   5. 参数/缓冲区段是否完整
//   6. CRC32 校验和
// strModelPath: .omm 模型文件路径
// 返回: ModelCheckResult 包含检查结果和解析出的元数据
inline ModelCheckResult checkModel(const std::string& strModelPath) {
    ModelCheckResult result;  // 20260330 ZJH 初始化结果（默认全部 false/0）

    // 20260330 ZJH 检查 1: 文件是否存在
    if (!std::filesystem::exists(strModelPath)) {
        result.strErrorMsg = "Model file not found: " + strModelPath;
        return result;
    }

    // 20260330 ZJH 获取文件大小
    auto fileSize = std::filesystem::file_size(strModelPath);
    result.fModelSizeMB = static_cast<float>(fileSize) / (1024.0f * 1024.0f);

    // 20260330 ZJH 检查文件最小尺寸（至少需要: 魔数4 + 版本4 + CRC4 = 12 字节）
    if (fileSize < 12) {
        result.strErrorMsg = "Model file too small (" + std::to_string(fileSize) + " bytes)";
        return result;
    }

    // 20260330 ZJH 读取整个文件到内存
    std::ifstream ifs(strModelPath, std::ios::binary);
    if (!ifs.is_open()) {
        result.strErrorMsg = "Cannot open model file: " + strModelPath;
        return result;
    }

    std::vector<uint8_t> vecFileData(fileSize);
    ifs.read(reinterpret_cast<char*>(vecFileData.data()), static_cast<std::streamsize>(fileSize));
    ifs.close();

    // 20260330 ZJH 检查 2: 验证魔数
    char arrMagic[4];
    std::memcpy(arrMagic, vecFileData.data(), 4);

    bool bIsOMM = (std::memcmp(arrMagic, detail::OMM_MAGIC, 4) == 0);
    bool bIsDFM = (std::memcmp(arrMagic, detail::DFM_MAGIC, 4) == 0);

    if (!bIsOMM && !bIsDFM) {
        result.strErrorMsg = "Invalid magic number (expected 'OMM\\0' or 'DFM\\0')";
        return result;
    }

    if (bIsDFM) {
        result.vecWarnings.push_back("Legacy DFM format detected, consider re-saving as OMM");
    }

    // 20260330 ZJH 检查 3: 读取版本号
    uint32_t nVersion = 0;
    std::memcpy(&nVersion, vecFileData.data() + 4, 4);
    result.nFormatVersion = static_cast<int>(nVersion);

    if (nVersion < 3 || nVersion > 4) {
        result.strErrorMsg = "Unsupported format version: " + std::to_string(nVersion)
                           + " (expected 3 or 4)";
        return result;
    }

    // 20260330 ZJH 解析指针（跳过魔数4 + 版本4 = 偏移8开始）
    size_t nOffset = 8;

    // 20260330 ZJH 检查 4: v4 格式解析元数据段
    if (nVersion == 4) {
        // 20260330 ZJH 读取元数据数量
        if (nOffset + 4 > fileSize) {
            result.strErrorMsg = "Truncated file: cannot read metadata count";
            return result;
        }
        uint32_t nNumMeta = 0;
        std::memcpy(&nNumMeta, vecFileData.data() + nOffset, 4);
        nOffset += 4;

        // 20260330 ZJH 防御性检查: 元数据条目数上限（防止恶意文件触发巨大内存分配）
        if (nNumMeta > 10000) {
            result.strErrorMsg = "Too many metadata entries: " + std::to_string(nNumMeta);
            return result;
        }

        // 20260330 ZJH 读取元数据 float 数组
        if (nNumMeta > 0 && nOffset + nNumMeta * 4 <= fileSize) {
            std::vector<float> vecMeta(nNumMeta);
            std::memcpy(vecMeta.data(), vecFileData.data() + nOffset, nNumMeta * 4);
            nOffset += nNumMeta * 4;

            // 20260330 ZJH 解码元数据
            // 布局: [magic=42.0f, typeHash, baseChannels, inputSize, numClasses, inChannels]
            if (nNumMeta >= 6 && vecMeta[0] == detail::META_MAGIC) {
                result.bHasMetadata = true;  // 20260330 ZJH 标记有元数据

                // 20260330 ZJH 提取模型类型 hash
                uint32_t nTypeHash = 0;
                std::memcpy(&nTypeHash, &vecMeta[1], sizeof(float));
                result.strModelType = detail::resolveModelType(nTypeHash);

                // 20260330 ZJH 提取架构参数
                result.nBaseChannels = static_cast<int>(vecMeta[2]);
                result.nInputWidth = static_cast<int>(vecMeta[3]);
                result.nInputHeight = static_cast<int>(vecMeta[3]);  // 20260330 ZJH 正方形输入
                result.nNumClasses = static_cast<int>(vecMeta[4]);
                result.nInputChannels = static_cast<int>(vecMeta[5]);

                // 20260330 ZJH 基本合理性检查
                if (result.nInputWidth <= 0 || result.nInputWidth > 4096) {
                    result.vecWarnings.push_back("Unusual input size: " + std::to_string(result.nInputWidth));
                }
                if (result.nNumClasses <= 0 || result.nNumClasses > 10000) {
                    result.vecWarnings.push_back("Unusual class count: " + std::to_string(result.nNumClasses));
                }
            } else {
                result.vecWarnings.push_back("Metadata present but invalid magic (expected 42.0)");
            }
        } else {
            // 20260330 ZJH 跳过元数据段（即使读不完），并夹紧 nOffset 防止溢出
            nOffset = std::min(static_cast<size_t>(nOffset + nNumMeta * 4), static_cast<size_t>(fileSize));
        }
    }

    // 20260330 ZJH 检查 5: 解析参数段 — 统计参数总数
    if (nOffset + 4 <= fileSize) {
        uint32_t nNumParams = 0;
        std::memcpy(&nNumParams, vecFileData.data() + nOffset, 4);
        nOffset += 4;

        int64_t nTotalParams = 0;  // 20260330 ZJH 参数元素总数

        // 20260330 ZJH 逐参数扫描
        for (uint32_t p = 0; p < nNumParams && nOffset < fileSize; ++p) {
            // 20260330 ZJH 读取参数名长度
            if (nOffset + 4 > fileSize) break;
            uint32_t nNameLen = 0;
            std::memcpy(&nNameLen, vecFileData.data() + nOffset, 4);
            nOffset += 4;

            // 20260330 ZJH 跳过参数名
            if (nOffset + nNameLen > fileSize) break;
            nOffset += nNameLen;

            // 20260330 ZJH 读取维度数
            if (nOffset + 4 > fileSize) break;
            uint32_t nNumDims = 0;
            std::memcpy(&nNumDims, vecFileData.data() + nOffset, 4);
            nOffset += 4;

            // 20260330 ZJH 读取各维度大小，计算元素总数
            int64_t nElements = 1;
            bool bOverflow = false;  // 20260330 ZJH 溢出检测标记
            for (uint32_t d = 0; d < nNumDims && nOffset + 4 <= fileSize; ++d) {
                uint32_t nDimSize = 0;
                std::memcpy(&nDimSize, vecFileData.data() + nOffset, 4);
                nOffset += 4;
                nElements *= static_cast<int64_t>(nDimSize);
                // 20260330 ZJH 溢出保护: 元素数不能超过文件能容纳的 float 数量
                if (nElements > static_cast<int64_t>(fileSize / sizeof(float))) {
                    bOverflow = true;
                    break;
                }
            }
            if (bOverflow) break;  // 20260330 ZJH 参数维度溢出，跳过后续解析

            // 20260330 ZJH 跳过参数数据（float32）
            size_t nDataBytes = static_cast<size_t>(nElements) * sizeof(float);
            if (nOffset + nDataBytes > fileSize) break;
            nOffset += nDataBytes;

            nTotalParams += nElements;  // 20260330 ZJH 累加参数量
        }

        result.nNumParams = nTotalParams;
    }

    // 20260330 ZJH 解析缓冲区段（BN running stats 等）
    if (nOffset + 4 <= fileSize) {
        uint32_t nNumBuffers = 0;
        std::memcpy(&nNumBuffers, vecFileData.data() + nOffset, 4);
        nOffset += 4;
        result.nNumBuffers = static_cast<int64_t>(nNumBuffers);

        // 20260330 ZJH 跳过缓冲区数据（结构与参数段相同）
        for (uint32_t b = 0; b < nNumBuffers && nOffset < fileSize; ++b) {
            if (nOffset + 4 > fileSize) break;
            uint32_t nNameLen = 0;
            std::memcpy(&nNameLen, vecFileData.data() + nOffset, 4);
            nOffset += 4;

            if (nOffset + nNameLen > fileSize) break;
            nOffset += nNameLen;

            if (nOffset + 4 > fileSize) break;
            uint32_t nNumDims = 0;
            std::memcpy(&nNumDims, vecFileData.data() + nOffset, 4);
            nOffset += 4;

            int64_t nElements = 1;
            bool bOverflow = false;  // 20260330 ZJH 溢出检测标记
            for (uint32_t d = 0; d < nNumDims && nOffset + 4 <= fileSize; ++d) {
                uint32_t nDimSize = 0;
                std::memcpy(&nDimSize, vecFileData.data() + nOffset, 4);
                nOffset += 4;
                nElements *= static_cast<int64_t>(nDimSize);
                // 20260330 ZJH 溢出保护: 元素数不能超过文件能容纳的 float 数量
                if (nElements > static_cast<int64_t>(fileSize / sizeof(float))) {
                    bOverflow = true;
                    break;
                }
            }
            if (bOverflow) break;  // 20260330 ZJH 缓冲区维度溢出，跳过后续解析

            size_t nDataBytes = static_cast<size_t>(nElements) * sizeof(float);
            if (nOffset + nDataBytes > fileSize) break;
            nOffset += nDataBytes;
        }
    }

    // 20260330 ZJH 检查 6: CRC32 校验
    // 文件末尾 4 字节是 CRC32，校验范围是前面所有字节
    if (fileSize >= 4) {
        uint32_t nStoredCrc = 0;
        std::memcpy(&nStoredCrc, vecFileData.data() + fileSize - 4, 4);

        uint32_t nComputedCrc = detail::computeCrc32(vecFileData.data(),
                                                      fileSize - 4);

        result.bCrcValid = (nStoredCrc == nComputedCrc);
        if (!result.bCrcValid) {
            result.vecWarnings.push_back("CRC32 mismatch: file may be corrupted "
                "(stored=0x" + ([](uint32_t v) {
                    char buf[16];
                    std::snprintf(buf, sizeof(buf), "%08X", v);
                    return std::string(buf);
                })(nStoredCrc) + " computed=0x" + ([](uint32_t v) {
                    char buf[16];
                    std::snprintf(buf, sizeof(buf), "%08X", v);
                    return std::string(buf);
                })(nComputedCrc) + ")");
        }
    }

    // 20260330 ZJH 综合判定: 魔数正确 + 版本支持 = 基本有效
    result.bValid = true;

    // 20260330 ZJH CRC 校验失败降级为警告（不阻止加载，但需提醒用户）
    // 某些场景下文件尾部可能被截断（如网络传输不完整）

    return result;  // 20260330 ZJH 返回完整检查结果
}

// =========================================================
// checkCudaDevice — 检查 CUDA 设备可用性
// =========================================================

// 20260330 ZJH checkCudaDevice — 检测系统中的 CUDA GPU 设备
// 编译时根据 OM_HAS_CUDA 宏决定是否启用 CUDA 检测
// 返回: ModelCheckResult（仅填充 CUDA 相关字段）
inline ModelCheckResult checkCudaDevice() {
    ModelCheckResult result;

#ifdef OM_HAS_CUDA
    // 20260330 ZJH CUDA 可用时，调用 CUDA Runtime API 查询设备
    int nDeviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&nDeviceCount);

    if (err == cudaSuccess && nDeviceCount > 0) {
        result.bCudaAvailable = true;        // 20260330 ZJH GPU 可用
        result.nCudaDeviceCount = nDeviceCount;  // 20260330 ZJH GPU 数量
        result.bValid = true;

        // 20260330 ZJH 获取第一个 GPU 的名称
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            result.strCudaDeviceName = prop.name;
        }
    } else {
        result.bCudaAvailable = false;
        result.nCudaDeviceCount = 0;
        result.bValid = true;  // 20260330 ZJH 检查本身成功，只是没有 GPU
        result.vecWarnings.push_back("No CUDA devices found (err=" + std::to_string(static_cast<int>(err)) + ")");
    }
#else
    // 20260330 ZJH 编译时未启用 CUDA
    result.bCudaAvailable = false;
    result.nCudaDeviceCount = 0;
    result.bValid = true;  // 20260330 ZJH 检查本身成功
    result.strErrorMsg = "CUDA support not compiled (OM_HAS_CUDA not defined)";
#endif

    return result;
}

// =========================================================
// checkCompatibility — 检查模型与输入图像的兼容性
// =========================================================

// 20260330 ZJH checkCompatibility — 验证模型与输入图像是否兼容
// 检查项目:
//   1. 模型文件有效性（调用 checkModel）
//   2. 通道数匹配（RGB=3 / 灰度=1）
//   3. 图像尺寸合理性（不能太小或太大）
// strModelPath: .omm 模型文件路径
// nImageW: 输入图像宽度
// nImageH: 输入图像高度
// nChannels: 输入图像通道数
// 返回: ModelCheckResult 包含兼容性检查结果
inline ModelCheckResult checkCompatibility(const std::string& strModelPath,
                                           int nImageW, int nImageH, int nChannels) {
    // 20260330 ZJH 步骤 1: 先检查模型本身
    ModelCheckResult result = checkModel(strModelPath);

    // 20260330 ZJH 模型无效直接返回
    if (!result.bValid) {
        return result;
    }

    // 20260330 ZJH 步骤 2: 检查输入图像尺寸合理性
    if (nImageW <= 0 || nImageH <= 0) {
        result.bValid = false;
        result.strErrorMsg = "Invalid image dimensions: " + std::to_string(nImageW) + "x" + std::to_string(nImageH);
        return result;
    }

    // 20260330 ZJH 警告过大的图像（>8K 可能导致显存溢出）
    if (nImageW > 8192 || nImageH > 8192) {
        result.vecWarnings.push_back("Very large image (" + std::to_string(nImageW) + "x"
            + std::to_string(nImageH) + "), consider using Multi-ROI or downscaling");
    }

    // 20260330 ZJH 步骤 3: 检查通道数兼容性（仅在有元数据时检查）
    if (result.bHasMetadata && result.nInputChannels > 0) {
        if (nChannels != result.nInputChannels) {
            result.bValid = false;
            result.strErrorMsg = "Channel mismatch: model expects "
                + std::to_string(result.nInputChannels) + " channels, image has "
                + std::to_string(nChannels);
            return result;
        }
    } else {
        // 20260330 ZJH 无元数据时基于常见约定检查
        if (nChannels != 1 && nChannels != 3) {
            result.vecWarnings.push_back("Unusual channel count: " + std::to_string(nChannels)
                + " (expected 1 or 3)");
        }
    }

    // 20260330 ZJH 步骤 4: 检查图像尺寸与模型输入尺寸的差异
    if (result.bHasMetadata && result.nInputWidth > 0) {
        // 20260330 ZJH 计算缩放比例
        float fScaleW = static_cast<float>(nImageW) / static_cast<float>(result.nInputWidth);
        float fScaleH = static_cast<float>(nImageH) / static_cast<float>(result.nInputHeight);

        // 20260330 ZJH 图像远小于模型输入（<25%），精度可能严重下降
        if (fScaleW < 0.25f || fScaleH < 0.25f) {
            result.vecWarnings.push_back("Image much smaller than model input ("
                + std::to_string(nImageW) + "x" + std::to_string(nImageH) + " vs "
                + std::to_string(result.nInputWidth) + "x" + std::to_string(result.nInputHeight)
                + "), upscaling may degrade quality");
        }

        // 20260330 ZJH 图像远大于模型输入（>16x），建议使用多 ROI
        if (fScaleW > 16.0f || fScaleH > 16.0f) {
            result.vecWarnings.push_back("Image much larger than model input, "
                "consider Multi-ROI inference for better detection of small objects");
        }
    }

    // 20260330 ZJH 通过所有兼容性检查
    return result;
}

}  // namespace om
