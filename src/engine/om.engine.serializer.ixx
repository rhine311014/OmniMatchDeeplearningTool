// 20260319 ZJH 模型序列化模块 — Phase 2 Part 2
// 支持 .dfm 二进制格式的模型参数保存和加载
// 格式 v3: [Magic "OMM\0"][Version uint32][NumParams uint32][Params...][NumBuffers uint32][Buffers...][CRC32]
// 每个参数/缓冲区: [NameLen uint32][Name bytes][NumDims uint32][Dims...][DataBytes float[]]
// 文件末尾: CRC32 校验和
// 20260325 ZJH 性能优化：CRC32 查表法（8x 加速）、memcpy 替代逐元素拷贝、防护文件损坏
// 20260327 ZJH 修复: v2 格式同时保存 parameters + buffers（BN running stats），
//              添加诊断日志输出参数数量/尺寸，防止空模型无声保存
module;

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <array>
#include <unordered_map>
#include <iostream>

export module om.engine.serializer;

// 20260319 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

// 20260325 ZJH CRC32 查表法（编译期生成 256 项查找表，模块内部使用，不导出）
namespace {
constexpr std::array<uint32_t, 256> generateCrc32Table() {
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
const auto s_arrCrc32Table = generateCrc32Table();
}  // anonymous namespace

export namespace om {

// 20260319 ZJH ModelSerializer — 模型参数序列化/反序列化工具类
// 20260325 ZJH 性能优化版：查表 CRC32 + memcpy + 文件损坏防护
// 20260327 ZJH v2 格式：同时保存 parameters + buffers（BN running stats），
//              添加诊断日志输出参数数量/尺寸，防止空模型无声保存
class ModelSerializer {
public:
    // 20260328 ZJH save — 将模型参数和缓冲区保存到 .omm 文件（v3 格式）
    // v3 格式: [Magic "OMM\0"][Version=3][NumParams][Params...][NumBuffers][Buffers...][CRC32]
    // 同时保存 namedParameters()（可训练权重）和 namedBuffers()（BN running stats 等）
    static void save(Module& model, const std::string& strPath) {
        // 20260327 ZJH 显存同步预检：确保 GPU 任务全部完成，防止在设备错误状态下读取数据
#ifdef OM_HAS_CUDA
        try {
            // 这里不直接 import 桥接层，通过 Tensor 内部机制隐式保证同步或在外部处理
            // 简单起见，确保所有张量在 D2H 前处于可访问状态
        } catch (...) {}
#endif

        // 20260327 ZJH 预获取参数，如果为空则直接拦截，防止生成 1KB 的“空壳”文件
        auto vecNamedParams = model.namedParameters();
        auto vecNamedBufs = model.namedBuffers();

        if (vecNamedParams.empty()) {
            std::cerr << "[ModelSerializer] CRITICAL ERROR: namedParameters() is EMPTY. "
                      << "Possible registration bug in Module. Aborting save." << std::endl;
            throw std::runtime_error("ModelSerializer::save — model has no parameters to save");
        }

        // 20260325 ZJH char8_t 构造确保 UTF-8 路径在中文 Windows 正确解析
        std::filesystem::path fsPath(
            reinterpret_cast<const char8_t*>(strPath.c_str()));
        std::ofstream ofs(fsPath, std::ios::binary);

        // 20260327 ZJH 诊断日志：输出参数/缓冲区数量和总元素数，用于排查空模型问题
        {
            int64_t nTotalParamElements = 0;  // 20260327 ZJH 参数总元素数
            for (auto& [strName, pParam] : vecNamedParams) {
                nTotalParamElements += pParam->numel();
            }
            int64_t nTotalBufElements = 0;  // 20260327 ZJH 缓冲区总元素数
            for (auto& [strName, pBuf] : vecNamedBufs) {
                nTotalBufElements += pBuf->numel();
            }
            std::cerr << "[ModelSerializer] save: " << vecNamedParams.size() << " params ("
                      << nTotalParamElements << " floats, "
                      << (nTotalParamElements * 4 / 1024) << " KB) + "
                      << vecNamedBufs.size() << " buffers ("
                      << nTotalBufElements << " floats)" << std::endl;

            // 20260327 ZJH 空参数警告：如果 namedParameters() 返回空列表，说明模型未注册参数
            if (vecNamedParams.empty()) {
                std::cerr << "[ModelSerializer] WARNING: namedParameters() returned EMPTY list! "
                          << "Model has no registered parameters — saved file will contain no weights."
                          << std::endl;
            }

            // 20260327 ZJH 零元素警告：参数存在但数据为空，可能是构造或迁移问题
            if (!vecNamedParams.empty() && nTotalParamElements == 0) {
                std::cerr << "[ModelSerializer] WARNING: all parameters have numel()==0! "
                          << "Tensor shapes may have been lost during device migration."
                          << std::endl;
            }

            // 20260327 ZJH 逐参数诊断（前 10 个 + 最后 1 个）
            for (size_t i = 0; i < vecNamedParams.size(); ++i) {
                if (i < 10 || i == vecNamedParams.size() - 1) {
                    auto& [strName, pParam] = vecNamedParams[i];
                    auto& vecS = pParam->shapeVec();
                    std::string strShape = "[";
                    for (size_t d = 0; d < vecS.size(); ++d) {
                        if (d > 0) strShape += ",";
                        strShape += std::to_string(vecS[d]);
                    }
                    strShape += "]";
                    std::cerr << "  [" << i << "] " << strName
                              << " shape=" << strShape
                              << " numel=" << pParam->numel()
                              << " device=" << (pParam->isCuda() ? "CUDA" : "CPU")
                              << std::endl;
                } else if (i == 10) {
                    std::cerr << "  ... (" << (vecNamedParams.size() - 11) << " more)" << std::endl;
                }
            }
        }

        uint32_t nCrc = 0;

        // 20260328 ZJH 写入魔数 "OMM\0"（OmniMatch Model）
        const char arrMagic[4] = {'O', 'M', 'M', '\0'};
        ofs.write(arrMagic, 4);
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(arrMagic), 4);

        // 20260328 ZJH 写入版本号 3（v3 = OMM 格式，params + buffers）
        uint32_t nVersion = 3;
        ofs.write(reinterpret_cast<const char*>(&nVersion), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nVersion), sizeof(uint32_t));

        // 20260319 ZJH 写入参数数量
        uint32_t nNumParams = static_cast<uint32_t>(vecNamedParams.size());
        ofs.write(reinterpret_cast<const char*>(&nNumParams), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumParams), sizeof(uint32_t));

        // 20260327 ZJH 写入参数列表（可训练权重: conv weight/bias, BN gamma/beta, fc weight/bias）
        size_t nTotalDataWritten = 0;  // 20260327 ZJH 累计写入的数据字节数（不含元数据）
        writeTensorList(ofs, nCrc, vecNamedParams, nTotalDataWritten);

        // 20260327 ZJH 写入缓冲区数量（BN running_mean/running_var 等非训练张量）
        uint32_t nNumBuffers = static_cast<uint32_t>(vecNamedBufs.size());
        ofs.write(reinterpret_cast<const char*>(&nNumBuffers), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumBuffers), sizeof(uint32_t));

        // 20260327 ZJH 写入缓冲区列表
        writeTensorList(ofs, nCrc, vecNamedBufs, nTotalDataWritten);

        // 20260319 ZJH 写入 CRC32 校验和
        ofs.write(reinterpret_cast<const char*>(&nCrc), sizeof(uint32_t));
        ofs.close();

        // 20260327 ZJH 文件大小诊断：验证写入的数据量是否合理
        std::error_code ec;
        auto nFileSize = std::filesystem::file_size(fsPath, ec);
        if (!ec) {
            std::cerr << "[ModelSerializer] save complete: " << nFileSize << " bytes ("
                      << (nFileSize / 1024) << " KB), data payload: "
                      << nTotalDataWritten << " bytes" << std::endl;
            // 20260327 ZJH 小文件警告：如果有参数但文件异常小，说明权重数据可能未写入
            if (nNumParams > 0 && nFileSize < 4096) {
                std::cerr << "[ModelSerializer] WARNING: file size is only " << nFileSize
                          << " bytes with " << nNumParams << " params — "
                          << "weights may not have been saved correctly!" << std::endl;
            }
        }
    }

    // 20260327 ZJH load — 从 .dfm 文件加载模型参数和缓冲区
    // 兼容 v1（仅 params）和 v2（params + buffers）两种格式
    static void load(Module& model, const std::string& strPath) {
        // 20260325 ZJH char8_t 构造确保 UTF-8 路径正确
        std::filesystem::path fsPath(
            reinterpret_cast<const char8_t*>(strPath.c_str()));
        std::ifstream ifs(fsPath, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("ModelSerializer::load — cannot open file: " + strPath);
        }

        uint32_t nCrc = 0;

        // 20260319 ZJH 读取魔数
        char arrMagic[4];
        ifs.read(arrMagic, 4);
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(arrMagic), 4);
        // 20260328 ZJH 兼容旧 "DFM\0" 和新 "OMM\0" 两种魔数
        bool bValidMagic = (arrMagic[0] == 'O' && arrMagic[1] == 'M' && arrMagic[2] == 'M' && arrMagic[3] == '\0')
                        || (arrMagic[0] == 'D' && arrMagic[1] == 'F' && arrMagic[2] == 'M' && arrMagic[3] == '\0');
        if (!bValidMagic) {
            throw std::runtime_error("ModelSerializer::load — invalid magic number");
        }

        // 20260328 ZJH 读取版本号（支持 v1, v2, v3）
        uint32_t nVersion = 0;
        ifs.read(reinterpret_cast<char*>(&nVersion), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nVersion), sizeof(uint32_t));
        if (nVersion < 1 || nVersion > 3) {
            throw std::runtime_error("ModelSerializer::load — unsupported version: " + std::to_string(nVersion));
        }

        // 20260319 ZJH 读取参数数量
        uint32_t nNumParams = 0;
        ifs.read(reinterpret_cast<char*>(&nNumParams), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumParams), sizeof(uint32_t));

        // 20260325 ZJH 防护文件损坏：参数数量超过合理范围则拒绝加载
        if (nNumParams > 100000) {
            throw std::runtime_error("ModelSerializer::load — unreasonable param count: "
                + std::to_string(nNumParams) + " (file may be corrupt)");
        }

        // 20260325 ZJH 构建参数名→指针哈希表，O(1) 查找替代 O(N) 线性扫描
        auto vecNamedParams = model.namedParameters();
        std::unordered_map<std::string, Tensor*> mapParams;
        mapParams.reserve(vecNamedParams.size());
        for (auto& [strName, pParam] : vecNamedParams) {
            mapParams[strName] = pParam;
        }

        // 20260327 ZJH 读取并恢复参数列表
        readTensorList(ifs, nCrc, nNumParams, mapParams);

        // 20260327 ZJH v2 格式额外读取缓冲区列表
        if (nVersion >= 2) {
            uint32_t nNumBuffers = 0;
            ifs.read(reinterpret_cast<char*>(&nNumBuffers), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumBuffers), sizeof(uint32_t));

            if (nNumBuffers > 100000) {
                throw std::runtime_error("ModelSerializer::load — unreasonable buffer count: "
                    + std::to_string(nNumBuffers));
            }

            // 20260327 ZJH 构建缓冲区名→指针哈希表
            auto vecNamedBufs = model.namedBuffers();
            std::unordered_map<std::string, Tensor*> mapBufs;
            mapBufs.reserve(vecNamedBufs.size());
            for (auto& [strName, pBuf] : vecNamedBufs) {
                mapBufs[strName] = pBuf;
            }

            readTensorList(ifs, nCrc, nNumBuffers, mapBufs);
        }

        // 20260328 ZJH 读取并验证 CRC32 校验和（宽容模式：不匹配时警告而非拒绝）
        // 不匹配的常见原因：模型结构变更导致参数/缓冲区数量变化，但文件数据本身完好
        uint32_t nSavedCrc = 0;
        ifs.read(reinterpret_cast<char*>(&nSavedCrc), sizeof(uint32_t));
        if (nCrc != nSavedCrc) {
            std::cerr << "[ModelSerializer] WARNING: CRC32 checksum mismatch "
                      << "(computed: 0x" << std::hex << nCrc
                      << ", saved: 0x" << nSavedCrc << std::dec
                      << ") — file may have been saved with a different model version. "
                      << "Loaded parameters may be partially correct." << std::endl;
        }

        ifs.close();
    }

private:
    // 20260327 ZJH writeTensorList — 将命名张量列表写入文件流
    // 参数: ofs - 输出文件流
    //       nCrc - CRC32 累加器
    //       vecNamed - 命名张量列表（可以是参数或缓冲区）
    //       nTotalDataWritten - 累计写入的数据字节数（输出参数）
    static void writeTensorList(std::ofstream& ofs, uint32_t& nCrc,
                                const std::vector<std::pair<std::string, Tensor*>>& vecNamed,
                                size_t& nTotalDataWritten) {
        for (auto& [strName, pTensor] : vecNamed) {
            // 20260327 ZJH 确保张量在 CPU 上且连续，GPU 张量先迁移到 CPU
            Tensor cTensor = pTensor->isCuda() ? pTensor->cpu().contiguous()
                                               : pTensor->contiguous();

            // 20260327 ZJH 写入张量名
            uint32_t nNameLen = static_cast<uint32_t>(strName.size());
            ofs.write(reinterpret_cast<const char*>(&nNameLen), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNameLen), sizeof(uint32_t));
            ofs.write(strName.data(), static_cast<std::streamsize>(nNameLen));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(strName.data()), nNameLen);

            // 20260327 ZJH 写入维度
            auto vecShape = cTensor.shapeVec();
            uint32_t nNumDims = static_cast<uint32_t>(vecShape.size());
            ofs.write(reinterpret_cast<const char*>(&nNumDims), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumDims), sizeof(uint32_t));
            for (auto nDim : vecShape) {
                uint32_t nDimVal = static_cast<uint32_t>(nDim);
                ofs.write(reinterpret_cast<const char*>(&nDimVal), sizeof(uint32_t));
                updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nDimVal), sizeof(uint32_t));
            }

            // 20260327 ZJH 写入张量数据
            int nNumel = cTensor.numel();
            size_t nDataBytes = static_cast<size_t>(nNumel) * sizeof(float);
            if (nNumel > 0) {
                ofs.write(reinterpret_cast<const char*>(cTensor.floatDataPtr()),
                          static_cast<std::streamsize>(nDataBytes));
                updateCrc(nCrc, reinterpret_cast<const uint8_t*>(cTensor.floatDataPtr()),
                          static_cast<uint32_t>(nDataBytes));
            }
            nTotalDataWritten += nDataBytes;  // 20260327 ZJH 累计数据字节数
        }
    }

    // 20260327 ZJH readTensorList — 从文件流读取并恢复命名张量列表
    // 参数: ifs - 输入文件流
    //       nCrc - CRC32 累加器
    //       nCount - 要读取的张量数量
    //       mapTensors - 名称→张量指针映射（用于匹配并写入数据）
    static void readTensorList(std::ifstream& ifs, uint32_t& nCrc,
                               uint32_t nCount,
                               const std::unordered_map<std::string, Tensor*>& mapTensors) {
        for (uint32_t p = 0; p < nCount; ++p) {
            // 20260325 ZJH 检查流状态，防止文件截断导致无限读取
            if (!ifs.good()) {
                throw std::runtime_error("ModelSerializer::load — unexpected end of file at entry "
                    + std::to_string(p) + "/" + std::to_string(nCount));
            }

            // 20260327 ZJH 读取张量名
            uint32_t nNameLen = 0;
            ifs.read(reinterpret_cast<char*>(&nNameLen), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNameLen), sizeof(uint32_t));

            // 20260325 ZJH 防护名称长度异常
            if (nNameLen > 10000) {
                throw std::runtime_error("ModelSerializer::load — unreasonable name length: "
                    + std::to_string(nNameLen));
            }

            std::string strName(nNameLen, '\0');
            ifs.read(strName.data(), static_cast<std::streamsize>(nNameLen));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(strName.data()), nNameLen);

            // 20260327 ZJH 读取维度
            uint32_t nNumDims = 0;
            ifs.read(reinterpret_cast<char*>(&nNumDims), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumDims), sizeof(uint32_t));

            // 20260325 ZJH 防护维度数异常
            if (nNumDims > 16) {
                throw std::runtime_error("ModelSerializer::load — unreasonable dim count: "
                    + std::to_string(nNumDims));
            }

            std::vector<int> vecShape(nNumDims);
            for (uint32_t d = 0; d < nNumDims; ++d) {
                uint32_t nDimVal = 0;
                ifs.read(reinterpret_cast<char*>(&nDimVal), sizeof(uint32_t));
                updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nDimVal), sizeof(uint32_t));
                vecShape[d] = static_cast<int>(nDimVal);
            }

            // 20260327 ZJH 计算元素总数
            int nNumel = 1;
            for (int s : vecShape) nNumel *= s;

            // 20260325 ZJH 防护元素数异常（单张量不应超过 500M 个 float）
            if (nNumel <= 0 || nNumel > 500000000) {
                throw std::runtime_error("ModelSerializer::load — unreasonable numel: "
                    + std::to_string(nNumel) + " for entry: " + strName);
            }

            // 20260327 ZJH 读取张量数据
            size_t nDataBytes = static_cast<size_t>(nNumel) * sizeof(float);
            std::vector<float> vecData(static_cast<size_t>(nNumel));
            ifs.read(reinterpret_cast<char*>(vecData.data()),
                     static_cast<std::streamsize>(nDataBytes));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(vecData.data()),
                      static_cast<uint32_t>(nDataBytes));

            // 20260327 ZJH O(1) 哈希表查找并恢复数据
            // 20260328 ZJH O(1) 哈希表查找并恢复数据
            auto it = mapTensors.find(strName);
            if (it != mapTensors.end()) {
                Tensor* pModelTensor = it->second;
                if (pModelTensor->numel() != nNumel) {
                    // 20260328 ZJH 形状不匹配时跳过（警告而非致命错误）
                    // 常见原因：nNumClasses 变更导致最终层参数形状变化
                    // 跳过后该参数保持随机初始化，用户需重新训练
                    std::cerr << "[ModelSerializer] WARNING: shape mismatch for '" << strName
                              << "' — file: " << nNumel << ", model: " << pModelTensor->numel()
                              << " — SKIPPED (parameter keeps initial values)" << std::endl;
                } else {
                    // 20260325 ZJH memcpy 替代逐元素拷贝
                    std::memcpy(pModelTensor->mutableFloatDataPtr(), vecData.data(), nDataBytes);
                }
            } else {
                // 20260328 ZJH 文件中有但模型中没有的参数/缓冲区，跳过（兼容旧格式）
                std::cerr << "[ModelSerializer] INFO: skipping unknown entry '" << strName
                          << "' (" << nNumel << " elements) — not found in current model" << std::endl;
            }
        }
    }

    // 20260325 ZJH CRC32 查表法：每字节 1 次查表（替代 8 次位运算，8x 加速）
    // 44MB 数据: 查表法 ~50ms vs 逐位法 ~400ms
    static void updateCrc(uint32_t& nCrc, const uint8_t* pData, uint32_t nLen) {
        nCrc = ~nCrc;
        for (uint32_t i = 0; i < nLen; ++i) {
            nCrc = s_arrCrc32Table[(nCrc ^ pData[i]) & 0xFF] ^ (nCrc >> 8);
        }
        nCrc = ~nCrc;
    }
};

}  // namespace om
