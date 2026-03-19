// 20260319 ZJH 模型序列化模块 — Phase 2 Part 2
// 支持 .dfm 二进制格式的模型参数保存和加载
// 格式: [Magic "DFM\0"][Version uint32][NumParams uint32]
// 每个参数: [NameLen uint32][Name bytes][NumDims uint32][Dims...][DataBytes float[]]
// 文件末尾: CRC32 校验和
module;

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>

export module df.engine.serializer;

// 20260319 ZJH 导入依赖模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;

export namespace df {

// 20260319 ZJH ModelSerializer — 模型参数序列化/反序列化工具类
class ModelSerializer {
public:
    // 20260319 ZJH save — 将模型参数保存到 .dfm 文件
    // model: 要保存的模型
    // strPath: 输出文件路径
    static void save(Module& model, const std::string& strPath) {
        std::ofstream ofs(strPath, std::ios::binary);  // 20260319 ZJH 以二进制模式打开文件
        if (!ofs.is_open()) {
            throw std::runtime_error("ModelSerializer::save — cannot open file: " + strPath);
        }

        // 20260319 ZJH 获取所有命名参数
        auto vecNamedParams = model.namedParameters();
        uint32_t nCrc = 0;  // 20260319 ZJH CRC32 校验和

        // 20260319 ZJH 写入魔数 "DFM\0"（4 字节）
        const char arrMagic[4] = {'D', 'F', 'M', '\0'};
        ofs.write(arrMagic, 4);
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(arrMagic), 4);

        // 20260319 ZJH 写入版本号（uint32，当前为 1）
        uint32_t nVersion = 1;
        ofs.write(reinterpret_cast<const char*>(&nVersion), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nVersion), sizeof(uint32_t));

        // 20260319 ZJH 写入参数数量
        uint32_t nNumParams = static_cast<uint32_t>(vecNamedParams.size());
        ofs.write(reinterpret_cast<const char*>(&nNumParams), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumParams), sizeof(uint32_t));

        // 20260319 ZJH 逐个写入参数
        for (auto& [strName, pParam] : vecNamedParams) {
            auto cParam = pParam->contiguous();  // 20260319 ZJH 确保参数连续

            // 20260319 ZJH 写入参数名长度和名称
            uint32_t nNameLen = static_cast<uint32_t>(strName.size());
            ofs.write(reinterpret_cast<const char*>(&nNameLen), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNameLen), sizeof(uint32_t));
            ofs.write(strName.data(), static_cast<std::streamsize>(nNameLen));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(strName.data()), nNameLen);

            // 20260319 ZJH 写入维度数和各维度大小
            auto vecShape = cParam.shapeVec();
            uint32_t nNumDims = static_cast<uint32_t>(vecShape.size());
            ofs.write(reinterpret_cast<const char*>(&nNumDims), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumDims), sizeof(uint32_t));
            for (auto nDim : vecShape) {
                uint32_t nDimVal = static_cast<uint32_t>(nDim);
                ofs.write(reinterpret_cast<const char*>(&nDimVal), sizeof(uint32_t));
                updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nDimVal), sizeof(uint32_t));
            }

            // 20260319 ZJH 写入参数数据（float 数组）
            int nNumel = cParam.numel();
            ofs.write(reinterpret_cast<const char*>(cParam.floatDataPtr()),
                      static_cast<std::streamsize>(nNumel) * static_cast<std::streamsize>(sizeof(float)));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(cParam.floatDataPtr()),
                      static_cast<uint32_t>(nNumel * sizeof(float)));
        }

        // 20260319 ZJH 写入 CRC32 校验和
        ofs.write(reinterpret_cast<const char*>(&nCrc), sizeof(uint32_t));
        ofs.close();
    }

    // 20260319 ZJH load — 从 .dfm 文件加载模型参数
    // model: 目标模型（必须与保存时结构一致）
    // strPath: 输入文件路径
    static void load(Module& model, const std::string& strPath) {
        std::ifstream ifs(strPath, std::ios::binary);  // 20260319 ZJH 以二进制模式打开文件
        if (!ifs.is_open()) {
            throw std::runtime_error("ModelSerializer::load — cannot open file: " + strPath);
        }

        uint32_t nCrc = 0;  // 20260319 ZJH CRC32 校验和（累积计算用于验证）

        // 20260319 ZJH 读取魔数
        char arrMagic[4];
        ifs.read(arrMagic, 4);
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(arrMagic), 4);
        if (arrMagic[0] != 'D' || arrMagic[1] != 'F' || arrMagic[2] != 'M' || arrMagic[3] != '\0') {
            throw std::runtime_error("ModelSerializer::load — invalid magic number");
        }

        // 20260319 ZJH 读取版本号
        uint32_t nVersion = 0;
        ifs.read(reinterpret_cast<char*>(&nVersion), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nVersion), sizeof(uint32_t));
        if (nVersion != 1) {
            throw std::runtime_error("ModelSerializer::load — unsupported version");
        }

        // 20260319 ZJH 读取参数数量
        uint32_t nNumParams = 0;
        ifs.read(reinterpret_cast<char*>(&nNumParams), sizeof(uint32_t));
        updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumParams), sizeof(uint32_t));

        // 20260319 ZJH 获取模型当前的命名参数
        auto vecNamedParams = model.namedParameters();

        // 20260319 ZJH 逐个读取参数并写入模型
        for (uint32_t p = 0; p < nNumParams; ++p) {
            // 20260319 ZJH 读取参数名
            uint32_t nNameLen = 0;
            ifs.read(reinterpret_cast<char*>(&nNameLen), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNameLen), sizeof(uint32_t));
            std::string strName(nNameLen, '\0');
            ifs.read(strName.data(), static_cast<std::streamsize>(nNameLen));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(strName.data()), nNameLen);

            // 20260319 ZJH 读取维度数和各维度大小
            uint32_t nNumDims = 0;
            ifs.read(reinterpret_cast<char*>(&nNumDims), sizeof(uint32_t));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nNumDims), sizeof(uint32_t));
            std::vector<int> vecShape(nNumDims);
            for (uint32_t d = 0; d < nNumDims; ++d) {
                uint32_t nDimVal = 0;
                ifs.read(reinterpret_cast<char*>(&nDimVal), sizeof(uint32_t));
                updateCrc(nCrc, reinterpret_cast<const uint8_t*>(&nDimVal), sizeof(uint32_t));
                vecShape[d] = static_cast<int>(nDimVal);
            }

            // 20260319 ZJH 计算元素总数
            int nNumel = 1;
            for (int s : vecShape) nNumel *= s;

            // 20260319 ZJH 读取参数数据
            std::vector<float> vecData(static_cast<size_t>(nNumel));
            ifs.read(reinterpret_cast<char*>(vecData.data()),
                     static_cast<std::streamsize>(nNumel) * static_cast<std::streamsize>(sizeof(float)));
            updateCrc(nCrc, reinterpret_cast<const uint8_t*>(vecData.data()),
                      static_cast<uint32_t>(nNumel * sizeof(float)));

            // 20260319 ZJH 在模型参数中查找匹配的参数名并更新数据
            for (auto& [strModelName, pModelParam] : vecNamedParams) {
                if (strModelName == strName) {
                    // 20260319 ZJH 验证形状匹配
                    if (pModelParam->numel() != nNumel) {
                        throw std::runtime_error("ModelSerializer::load — shape mismatch for param: " + strName);
                    }
                    // 20260319 ZJH 将数据拷贝到模型参数中
                    float* pDst = pModelParam->mutableFloatDataPtr();
                    for (int i = 0; i < nNumel; ++i) {
                        pDst[i] = vecData[static_cast<size_t>(i)];
                    }
                    break;  // 20260319 ZJH 找到后退出内层循环
                }
            }
        }

        // 20260319 ZJH 读取并验证 CRC32 校验和
        uint32_t nSavedCrc = 0;
        ifs.read(reinterpret_cast<char*>(&nSavedCrc), sizeof(uint32_t));
        if (nCrc != nSavedCrc) {
            throw std::runtime_error("ModelSerializer::load — CRC32 checksum mismatch");
        }

        ifs.close();
    }

private:
    // 20260319 ZJH 简单 CRC32 计算（使用标准多项式 0xEDB88320）
    static void updateCrc(uint32_t& nCrc, const uint8_t* pData, uint32_t nLen) {
        nCrc = ~nCrc;  // 20260319 ZJH 取反（CRC32 初始值约定）
        for (uint32_t i = 0; i < nLen; ++i) {
            nCrc ^= pData[i];
            for (int bit = 0; bit < 8; ++bit) {
                if (nCrc & 1) {
                    nCrc = (nCrc >> 1) ^ 0xEDB88320u;  // 20260319 ZJH 标准 CRC32 多项式
                } else {
                    nCrc >>= 1;
                }
            }
        }
        nCrc = ~nCrc;  // 20260319 ZJH 取反完成
    }
};

}  // namespace df
