// 20260320 ZJH 数据管线模块 — Phase 5
// Dataset/DataLoader/数据增强/图像加载，支持分类/检测/分割任务
module;

#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <functional>
#include <array>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

export module om.engine.data_pipeline;

import om.engine.tensor;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260324 ZJH 图像加载：纯 C++ BMP 加载和基础操作
// GUI 应用中使用 Qt6 QImage 加载更多格式，此处提供引擎层独立后备方案
// =========================================================

// 20260320 ZJH RawImage — 原始图像数据结构
// 存储解码后的像素数据（float 归一化到 [0,1]）
struct RawImage {
    std::vector<float> vecData;  // 20260320 ZJH 像素数据 [C, H, W]，CHW 排列，float [0,1]
    int nWidth = 0;              // 20260320 ZJH 图像宽度
    int nHeight = 0;             // 20260320 ZJH 图像高度
    int nChannels = 0;           // 20260320 ZJH 通道数（1=灰度，3=RGB）

    // 20260320 ZJH 是否有效（已加载数据）
    bool isValid() const { return !vecData.empty() && nWidth > 0 && nHeight > 0; }
};

// 20260330 ZJH ResizeMode — 图像缩放插值模式枚举
// 支持最近邻、双线性插值、Letterbox（保持宽高比等比缩放+灰色填充）
enum class ResizeMode {
    NearestNeighbor,  // 20260330 ZJH 最近邻插值（速度最快，质量最低）
    Bilinear,         // 20260330 ZJH 双线性插值（速度与质量平衡，推荐默认）
    Letterbox          // 20260330 ZJH Letterbox 模式（保持宽高比，灰色填充）
};

// 20260330 ZJH LetterboxInfo — Letterbox 缩放后的坐标映射信息
// 用于推理后将检测框坐标从 letterbox 空间映射回原始图像坐标
struct LetterboxInfo {
    float fScale;     // 20260330 ZJH 缩放比例因子（min(targetW/srcW, targetH/srcH)）
    int nPadLeft;     // 20260330 ZJH 左侧填充像素数
    int nPadTop;      // 20260330 ZJH 顶部填充像素数
    int nNewW;        // 20260330 ZJH 等比缩放后的实际图像宽度（不含填充）
    int nNewH;        // 20260330 ZJH 等比缩放后的实际图像高度（不含填充）
};

// 20260330 ZJH NormPreset — 归一化预设枚举
// 定义常用的图像归一化方案，简化配置
enum class NormPreset {
    None,      // 20260330 ZJH 不归一化，保持原始 [0,1] 值
    ZeroOne,   // 20260330 ZJH 仅 /255 归一化（当前默认行为，输入已经是 [0,1] 则无操作）
    MeanStd,   // 20260330 ZJH 用户自定义 mean/std 逐通道归一化
    ImageNet   // 20260330 ZJH ImageNet 预训练模型标准归一化 mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
};

// 20260320 ZJH loadBMP — 简易 BMP 加载器（不依赖外部库）
// 支持 24 位无压缩 BMP
// 返回: RawImage 结构，CHW 格式，float [0,1]
RawImage loadBMP(const std::string& strPath) {
    RawImage img;
    std::ifstream file(strPath, std::ios::binary);
    if (!file.is_open()) return img;  // 20260320 ZJH 打开失败返回空

    // 20260320 ZJH 读取 BMP 文件头
    char header[54];
    file.read(header, 54);
    // 20260330 ZJH 防御性检查: 读取是否成功且达到 54 字节
    if (!file.good() || file.gcount() < 54) return img;
    if (header[0] != 'B' || header[1] != 'M') return img;  // 20260320 ZJH 验证 BMP 魔数

    // 20260330 ZJH 使用 memcpy 替代 reinterpret_cast 避免未对齐访问 UB
    int nDataOffset = 0;
    std::memcpy(&nDataOffset, &header[10], sizeof(int));  // 20260330 ZJH 像素数据偏移
    std::memcpy(&img.nWidth, &header[18], sizeof(int));   // 20260330 ZJH 宽度
    std::memcpy(&img.nHeight, &header[22], sizeof(int));  // 20260330 ZJH 高度
    short nBppShort = 0;
    std::memcpy(&nBppShort, &header[28], sizeof(short));  // 20260330 ZJH 每像素位数
    int nBpp = static_cast<int>(nBppShort);

    // 20260330 ZJH 检查压缩方式: 仅支持无压缩 BMP (compression == 0)
    int nCompression = 0;
    std::memcpy(&nCompression, &header[30], sizeof(int));
    if (nCompression != 0) return img;  // 20260330 ZJH 压缩 BMP 不支持，直接返回空

    if (nBpp != 24 && nBpp != 8) return img;  // 20260320 ZJH 仅支持 8 位灰度和 24 位 RGB

    img.nChannels = (nBpp == 24) ? 3 : 1;
    bool bFlipped = img.nHeight > 0;  // 20260320 ZJH BMP 通常底到顶存储
    if (img.nHeight < 0) img.nHeight = -img.nHeight;

    // 20260320 ZJH 每行字节数（4 字节对齐）
    int nRowBytes = ((img.nWidth * (nBpp / 8) + 3) / 4) * 4;

    // 20260330 ZJH 校验 nDataOffset 合法性: 至少 54 字节头部，不超过文件大小
    file.seekg(0, std::ios::end);
    auto nFileSize = file.tellg();  // 20260330 ZJH 获取文件总大小
    if (nDataOffset < 54 || static_cast<std::streampos>(nDataOffset) > nFileSize) return img;

    file.seekg(nDataOffset);
    std::vector<unsigned char> vecRawPixels(static_cast<size_t>(nRowBytes * img.nHeight));
    file.read(reinterpret_cast<char*>(vecRawPixels.data()),
              static_cast<std::streamsize>(vecRawPixels.size()));

    // 20260320 ZJH 转换为 CHW float [0,1]
    img.vecData.resize(static_cast<size_t>(img.nChannels * img.nHeight * img.nWidth));
    for (int y = 0; y < img.nHeight; ++y) {
        int nSrcY = bFlipped ? (img.nHeight - 1 - y) : y;  // 20260320 ZJH 处理翻转
        for (int x = 0; x < img.nWidth; ++x) {
            int nSrcIdx = nSrcY * nRowBytes + x * (nBpp / 8);
            if (img.nChannels == 3) {
                // 20260320 ZJH BMP 存 BGR，转为 RGB CHW
                float fB = vecRawPixels[static_cast<size_t>(nSrcIdx)] / 255.0f;
                float fG = vecRawPixels[static_cast<size_t>(nSrcIdx + 1)] / 255.0f;
                float fR = vecRawPixels[static_cast<size_t>(nSrcIdx + 2)] / 255.0f;
                img.vecData[static_cast<size_t>(0 * img.nHeight * img.nWidth + y * img.nWidth + x)] = fR;
                img.vecData[static_cast<size_t>(1 * img.nHeight * img.nWidth + y * img.nWidth + x)] = fG;
                img.vecData[static_cast<size_t>(2 * img.nHeight * img.nWidth + y * img.nWidth + x)] = fB;
            } else {
                img.vecData[static_cast<size_t>(y * img.nWidth + x)] =
                    vecRawPixels[static_cast<size_t>(nSrcIdx)] / 255.0f;
            }
        }
    }

    return img;
}

// =========================================================
// 20260330 ZJH 16-bit TIFF 加载器（无外部依赖）
// 工业线扫相机/X-ray 常用 16-bit 灰度 TIFF 格式
// 支持: 8-bit/16-bit 灰度, 8-bit/16-bit RGB, Little-Endian/Big-Endian
// 仅支持无压缩 (Compression=1) 的 Strip 数据格式
// =========================================================

// 20260330 ZJH TiffImage — TIFF 图像数据结构
// 存储解码后的像素数据（float 归一化到 [0,1]），HWC 排列
struct TiffImage {
    std::vector<float> vecData;  // 20260330 ZJH 像素数据 [H, W, C]，float [0,1]
    int nWidth = 0;              // 20260330 ZJH 图像宽度（像素）
    int nHeight = 0;             // 20260330 ZJH 图像高度（像素）
    int nChannels = 0;           // 20260330 ZJH 通道数（1=��度，3=RGB）
    int nBitsPerSample = 0;      // 20260330 ZJH 原始位深（8 或 16）

    // 20260330 ZJH 是否有效（已成功加载数据）
    bool isValid() const { return !vecData.empty() && nWidth > 0 && nHeight > 0; }
};

// 20260330 ZJH loadTiff — 纯 C++ 实现的 TIFF 加载器（不依赖 libtiff）
// 支持的 TIFF 子集:
//   - 字节序: Little-Endian ("II") 和 Big-Endian ("MM")
//   - 位深: 8-bit 和 16-bit
//   - 通道: 灰度 (SamplesPerPixel=1) 和 RGB (SamplesPerPixel=3)
//   - 压缩: 仅无压缩 (Compression=1)
//   - 数据布局: Strip（条带）格式
// 不支持: LZW/Deflate 压缩, Tile 布局, CMYK, Alpha 通道, Planar 配置
// strPath: TIFF 文件路径
// 返回: TiffImage 结构体，加载失败时 isValid() 返回 false
TiffImage loadTiff(const std::string& strPath) {
    TiffImage tiff;  // 20260330 ZJH 结果结构体

    // 20260330 ZJH 打开文件
    std::ifstream file(strPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[loadTiff] ERROR: cannot open file: " << strPath << std::endl;
        return tiff;  // 20260330 ZJH 打开失败返回空
    }

    // 20260330 ZJH 读取整个文件到内存（便于随机访问 IFD 和 strip 数据）
    file.seekg(0, std::ios::end);
    size_t nFileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (nFileSize < 8) {
        std::cerr << "[loadTiff] ERROR: file too small (" << nFileSize << " bytes)" << std::endl;
        return tiff;  // 20260330 ZJH 文件过小，不可能是有效 TIFF
    }

    std::vector<uint8_t> vecFile(nFileSize);  // 20260330 ZJH 文件内容缓冲区
    file.read(reinterpret_cast<char*>(vecFile.data()), static_cast<std::streamsize>(nFileSize));
    file.close();

    // 20260330 ZJH 解析 TIFF 头部: 字节序标记 (2 bytes) + 魔数 42 (2 bytes) + IFD 偏移 (4 bytes)
    // 字节序: "II" = Little-Endian (Intel), "MM" = Big-Endian (Motorola)
    bool bLittleEndian = false;  // 20260330 ZJH 字节序标志
    if (vecFile[0] == 'I' && vecFile[1] == 'I') {
        bLittleEndian = true;  // 20260330 ZJH Intel 字节序（最常见）
    } else if (vecFile[0] == 'M' && vecFile[1] == 'M') {
        bLittleEndian = false;  // 20260330 ZJH Motorola 字节序
    } else {
        std::cerr << "[loadTiff] ERROR: invalid byte order mark" << std::endl;
        return tiff;  // 20260330 ZJH 非法字节序标记
    }

    // 20260330 ZJH 字节序感知的读取辅助 lambda
    // readU16: 从缓冲区指定偏移读取 16 位无符号整数
    auto readU16 = [&](size_t nOffset) -> uint16_t {
        if (nOffset + 1 >= nFileSize) return 0;  // 20260330 ZJH 越界保护
        uint16_t val;
        if (bLittleEndian) {
            val = static_cast<uint16_t>(vecFile[nOffset])
                | (static_cast<uint16_t>(vecFile[nOffset + 1]) << 8);
        } else {
            val = (static_cast<uint16_t>(vecFile[nOffset]) << 8)
                | static_cast<uint16_t>(vecFile[nOffset + 1]);
        }
        return val;  // 20260330 ZJH 返回解析后的 16 位值
    };

    // 20260330 ZJH readU32: 从缓冲区指定偏移读取 32 位无符号整数
    auto readU32 = [&](size_t nOffset) -> uint32_t {
        if (nOffset + 3 >= nFileSize) return 0;  // 20260330 ZJH 越界保护
        uint32_t val;
        if (bLittleEndian) {
            val = static_cast<uint32_t>(vecFile[nOffset])
                | (static_cast<uint32_t>(vecFile[nOffset + 1]) << 8)
                | (static_cast<uint32_t>(vecFile[nOffset + 2]) << 16)
                | (static_cast<uint32_t>(vecFile[nOffset + 3]) << 24);
        } else {
            val = (static_cast<uint32_t>(vecFile[nOffset]) << 24)
                | (static_cast<uint32_t>(vecFile[nOffset + 1]) << 16)
                | (static_cast<uint32_t>(vecFile[nOffset + 2]) << 8)
                | static_cast<uint32_t>(vecFile[nOffset + 3]);
        }
        return val;  // 20260330 ZJH 返回解析后的 32 位值
    };

    // 20260330 ZJH 验证 TIFF 魔数 (42)
    uint16_t nMagic = readU16(2);
    if (nMagic != 42) {
        std::cerr << "[loadTiff] ERROR: invalid magic number " << nMagic << " (expected 42)" << std::endl;
        return tiff;  // 20260330 ZJH 魔数不匹配
    }

    // 20260330 ZJH 读取第一个 IFD（Image File Directory）的偏移
    uint32_t nIfdOffset = readU32(4);
    if (nIfdOffset == 0 || nIfdOffset >= nFileSize) {
        std::cerr << "[loadTiff] ERROR: invalid IFD offset " << nIfdOffset << std::endl;
        return tiff;  // 20260330 ZJH IFD 偏移无效
    }

    // 20260330 ZJH 解析 IFD 标签
    // TIFF IFD 格式: [NumEntries uint16][Entry[12 bytes] * N][NextIFD uint32]
    // 每个 Entry: [Tag uint16][Type uint16][Count uint32][Value/Offset uint32]
    uint16_t nNumEntries = readU16(nIfdOffset);
    if (nNumEntries == 0 || nNumEntries > 500) {
        std::cerr << "[loadTiff] ERROR: invalid IFD entry count " << nNumEntries << std::endl;
        return tiff;  // 20260330 ZJH 条目数不合理
    }

    // 20260330 ZJH TIFF 标签常量（Baseline TIFF 6.0 规范）
    constexpr uint16_t TAG_IMAGE_WIDTH       = 256;   // 20260330 ZJH 图像宽度
    constexpr uint16_t TAG_IMAGE_LENGTH      = 257;   // 20260330 ZJH 图像高度
    constexpr uint16_t TAG_BITS_PER_SAMPLE   = 258;   // 20260330 ZJH 每样本位数
    constexpr uint16_t TAG_COMPRESSION       = 259;   // 20260330 ZJH 压缩方式
    constexpr uint16_t TAG_SAMPLES_PER_PIXEL = 277;   // 20260330 ZJH 每像素样本数
    constexpr uint16_t TAG_ROWS_PER_STRIP    = 278;   // 20260330 ZJH 每条带行数
    constexpr uint16_t TAG_STRIP_OFFSETS     = 273;   // 20260330 ZJH 条带数据偏移
    constexpr uint16_t TAG_STRIP_BYTE_COUNTS = 279;   // 20260330 ZJH 条带数据字节数

    // 20260330 ZJH TIFF 数据类型大小（字节）
    // Type: 1=BYTE(1), 2=ASCII(1), 3=SHORT(2), 4=LONG(4), 5=RATIONAL(8)
    auto typeSize = [](uint16_t nType) -> int {
        switch (nType) {
            case 1: return 1;   // 20260330 ZJH BYTE
            case 2: return 1;   // 20260330 ZJH ASCII
            case 3: return 2;   // 20260330 ZJH SHORT
            case 4: return 4;   // 20260330 ZJH LONG
            case 5: return 8;   // 20260330 ZJH RATIONAL
            default: return 1;  // 20260330 ZJH 未知类型默认 1 字节
        }
    };

    // 20260330 ZJH 初始化要提取的关键标签值
    int nImageWidth = 0;              // 20260330 ZJH 图像宽度
    int nImageHeight = 0;             // 20260330 ZJH 图像高度
    int nBitsPerSample = 8;           // 20260330 ZJH 默认 8 位
    int nCompression = 1;             // 20260330 ZJH 默认无压缩
    int nSamplesPerPixel = 1;         // 20260330 ZJH 默认灰度
    int nRowsPerStrip = 0;            // 20260330 ZJH 每条带行数
    std::vector<uint32_t> vecStripOffsets;     // 20260330 ZJH 各条带的文件偏移
    std::vector<uint32_t> vecStripByteCounts;  // 20260330 ZJH 各条带的字节数

    // 20260330 ZJH readTagValue — 从 IFD 条目读取单个值（SHORT 或 LONG）
    // 如果值总字节数 <= 4，值直接存储在 Value/Offset 字段
    // 否则 Value/Offset 字段存储指向值数据的文件偏移
    auto readTagValue = [&](size_t nEntryOffset) -> uint32_t {
        uint16_t nType = readU16(nEntryOffset + 2);     // 20260330 ZJH 数据类型
        uint32_t nCount = readU32(nEntryOffset + 4);     // 20260330 ZJH 值的数量
        size_t nValueBytes = static_cast<size_t>(nCount) * typeSize(nType);  // 20260330 ZJH 总字节数

        if (nValueBytes <= 4) {
            // 20260330 ZJH 值内联存储在条目的 Value/Offset 字段（offset+8 处）
            if (nType == 3) {  // 20260330 ZJH SHORT 类型
                return readU16(nEntryOffset + 8);
            } else {           // 20260330 ZJH LONG 或其他类型
                return readU32(nEntryOffset + 8);
            }
        } else {
            // 20260330 ZJH 值存储在外部，Value/Offset 是文件偏移
            uint32_t nValOffset = readU32(nEntryOffset + 8);
            if (nType == 3) {
                return readU16(nValOffset);
            } else {
                return readU32(nValOffset);
            }
        }
    };

    // 20260330 ZJH readTagArray — 从 IFD 条目读取值数组（用于 StripOffsets/StripByteCounts）
    auto readTagArray = [&](size_t nEntryOffset) -> std::vector<uint32_t> {
        uint16_t nType = readU16(nEntryOffset + 2);      // 20260330 ZJH 数据类型
        uint32_t nCount = readU32(nEntryOffset + 4);      // 20260330 ZJH 值的数量
        size_t nValueBytes = static_cast<size_t>(nCount) * typeSize(nType);

        std::vector<uint32_t> vecValues;
        vecValues.reserve(nCount);

        size_t nDataOffset;  // 20260330 ZJH 值数据的起始偏移
        if (nValueBytes <= 4) {
            nDataOffset = nEntryOffset + 8;  // 20260330 ZJH 内联存储
        } else {
            nDataOffset = readU32(nEntryOffset + 8);  // 20260330 ZJH 外部存储
        }

        // 20260330 ZJH 逐个读取值
        for (uint32_t i = 0; i < nCount; ++i) {
            if (nType == 3) {  // 20260330 ZJH SHORT 类型（2 字节）
                vecValues.push_back(readU16(nDataOffset + i * 2));
            } else {           // 20260330 ZJH LONG 类型（4 字节）
                vecValues.push_back(readU32(nDataOffset + i * 4));
            }
        }
        return vecValues;  // 20260330 ZJH 返回值数组
    };

    // 20260330 ZJH 遍历所有 IFD 条目，提取关键标签
    for (uint16_t i = 0; i < nNumEntries; ++i) {
        size_t nEntryOffset = nIfdOffset + 2 + static_cast<size_t>(i) * 12;  // 20260330 ZJH 每条目 12 字节
        if (nEntryOffset + 12 > nFileSize) break;  // 20260330 ZJH 越界保护

        uint16_t nTag = readU16(nEntryOffset);  // 20260330 ZJH 读取标签 ID

        // 20260330 ZJH 根据标签 ID 提取对应的值
        switch (nTag) {
            case TAG_IMAGE_WIDTH:
                nImageWidth = static_cast<int>(readTagValue(nEntryOffset));
                break;
            case TAG_IMAGE_LENGTH:
                nImageHeight = static_cast<int>(readTagValue(nEntryOffset));
                break;
            case TAG_BITS_PER_SAMPLE:
                nBitsPerSample = static_cast<int>(readTagValue(nEntryOffset));
                break;
            case TAG_COMPRESSION:
                nCompression = static_cast<int>(readTagValue(nEntryOffset));
                break;
            case TAG_SAMPLES_PER_PIXEL:
                nSamplesPerPixel = static_cast<int>(readTagValue(nEntryOffset));
                break;
            case TAG_ROWS_PER_STRIP:
                nRowsPerStrip = static_cast<int>(readTagValue(nEntryOffset));
                break;
            case TAG_STRIP_OFFSETS:
                vecStripOffsets = readTagArray(nEntryOffset);
                break;
            case TAG_STRIP_BYTE_COUNTS:
                vecStripByteCounts = readTagArray(nEntryOffset);
                break;
            default:
                break;  // 20260330 ZJH 忽略其他标签
        }
    }

    // 20260330 ZJH 验证必需标签
    if (nImageWidth <= 0 || nImageHeight <= 0) {
        std::cerr << "[loadTiff] ERROR: invalid dimensions " << nImageWidth << "x" << nImageHeight << std::endl;
        return tiff;  // 20260330 ZJH 尺寸无效
    }

    // 20260330 ZJH 验证压缩方式（仅支持无压缩）
    if (nCompression != 1) {
        std::cerr << "[loadTiff] ERROR: unsupported compression type " << nCompression
                  << " (only uncompressed/1 supported)" << std::endl;
        return tiff;  // 20260330 ZJH 不支持压缩格式
    }

    // 20260330 ZJH 验证位深（仅支持 8 位和 16 位）
    if (nBitsPerSample != 8 && nBitsPerSample != 16) {
        std::cerr << "[loadTiff] ERROR: unsupported bits per sample " << nBitsPerSample
                  << " (only 8 and 16 supported)" << std::endl;
        return tiff;  // 20260330 ZJH 不支持的位深
    }

    // 20260330 ZJH 验证通道数（仅支持灰度和 RGB）
    if (nSamplesPerPixel != 1 && nSamplesPerPixel != 3) {
        std::cerr << "[loadTiff] ERROR: unsupported samples per pixel " << nSamplesPerPixel
                  << " (only 1 and 3 supported)" << std::endl;
        return tiff;  // 20260330 ZJH 不支持的通道数
    }

    // 20260330 ZJH 验证 strip 数据
    if (vecStripOffsets.empty()) {
        std::cerr << "[loadTiff] ERROR: no StripOffsets tag found" << std::endl;
        return tiff;  // 20260330 ZJH 缺少条带偏移信息
    }

    // 20260330 ZJH 如果没有 RowsPerStrip 标签，默认整张图为一个 strip
    if (nRowsPerStrip <= 0) {
        nRowsPerStrip = nImageHeight;
    }

    // 20260330 ZJH 计算预期的原始数据总字节数
    int nBytesPerPixel = nSamplesPerPixel * (nBitsPerSample / 8);     // 20260330 ZJH 每像素字节数
    size_t nExpectedBytes = static_cast<size_t>(nImageWidth)
                          * static_cast<size_t>(nImageHeight)
                          * static_cast<size_t>(nBytesPerPixel);       // 20260330 ZJH 预期总字节数

    // 20260330 ZJH 从 strip 数据中读取原始像素
    std::vector<uint8_t> vecRawPixels;
    vecRawPixels.reserve(nExpectedBytes);  // 20260330 ZJH 预分配内存

    for (size_t s = 0; s < vecStripOffsets.size(); ++s) {
        uint32_t nStripOff = vecStripOffsets[s];  // 20260330 ZJH 条带起始偏移

        // 20260330 ZJH 确定条带字节数
        uint32_t nStripBytes;
        if (s < vecStripByteCounts.size()) {
            nStripBytes = vecStripByteCounts[s];  // 20260330 ZJH 从标签获取
        } else {
            // 20260330 ZJH 缺少 StripByteCounts，根据 RowsPerStrip 计算
            int nStripRows = std::min(nRowsPerStrip,
                                      nImageHeight - static_cast<int>(s) * nRowsPerStrip);
            nStripBytes = static_cast<uint32_t>(nStripRows * nImageWidth * nBytesPerPixel);
        }

        // 20260330 ZJH 越界检查
        if (static_cast<size_t>(nStripOff) + nStripBytes > nFileSize) {
            std::cerr << "[loadTiff] WARNING: strip " << s << " exceeds file size, truncating" << std::endl;
            nStripBytes = static_cast<uint32_t>(nFileSize - nStripOff);  // 20260330 ZJH 截断到文件末尾
        }

        // 20260330 ZJH 将条带数据追加到原始像素缓冲区
        vecRawPixels.insert(vecRawPixels.end(),
                            vecFile.data() + nStripOff,
                            vecFile.data() + nStripOff + nStripBytes);
    }

    // 20260330 ZJH 检查实际读取的数据量是否足够
    if (vecRawPixels.size() < nExpectedBytes) {
        std::cerr << "[loadTiff] WARNING: read " << vecRawPixels.size()
                  << " bytes but expected " << nExpectedBytes << std::endl;
        // 20260330 ZJH 不足时补零，避免越界访问
        vecRawPixels.resize(nExpectedBytes, 0);
    }

    // 20260330 ZJH 填充输出结构体元数据
    tiff.nWidth = nImageWidth;
    tiff.nHeight = nImageHeight;
    tiff.nChannels = nSamplesPerPixel;
    tiff.nBitsPerSample = nBitsPerSample;

    // 20260330 ZJH 将原始像素数据转换为 float [0,1]，HWC 排列
    // 16-bit: 除以 65535.0f，8-bit: 除以 255.0f
    size_t nTotalPixels = static_cast<size_t>(nImageWidth)
                        * static_cast<size_t>(nImageHeight)
                        * static_cast<size_t>(nSamplesPerPixel);
    tiff.vecData.resize(nTotalPixels);

    if (nBitsPerSample == 16) {
        // 20260330 ZJH 16-bit 数据转换
        // 每个样本占 2 字节，字节序由 TIFF 头部的字节序标记决定
        float fScale = 1.0f / 65535.0f;  // 20260330 ZJH 归一化因子
        for (size_t i = 0; i < nTotalPixels; ++i) {
            size_t nByteOffset = i * 2;  // 20260330 ZJH 每样本 2 字节
            uint16_t nVal;
            if (bLittleEndian) {
                nVal = static_cast<uint16_t>(vecRawPixels[nByteOffset])
                     | (static_cast<uint16_t>(vecRawPixels[nByteOffset + 1]) << 8);
            } else {
                nVal = (static_cast<uint16_t>(vecRawPixels[nByteOffset]) << 8)
                     | static_cast<uint16_t>(vecRawPixels[nByteOffset + 1]);
            }
            tiff.vecData[i] = static_cast<float>(nVal) * fScale;  // 20260330 ZJH 归一化到 [0,1]
        }
    } else {
        // 20260330 ZJH 8-bit 数据转换
        float fScale = 1.0f / 255.0f;  // 20260330 ZJH 归一化因子
        for (size_t i = 0; i < nTotalPixels; ++i) {
            tiff.vecData[i] = static_cast<float>(vecRawPixels[i]) * fScale;  // 20260330 ZJH 归一化到 [0,1]
        }
    }

    std::cerr << "[loadTiff] loaded " << nImageWidth << "x" << nImageHeight
              << " " << nBitsPerSample << "-bit " << nSamplesPerPixel << "ch"
              << (bLittleEndian ? " LE" : " BE") << std::endl;

    return tiff;  // 20260330 ZJH 返回加载完成的 TIFF 图像
}

// 20260330 ZJH tiffToRawImage — 将 TiffImage (HWC) 转换为引擎标准 RawImage (CHW)
// 引擎内部统一使用 CHW 格式，loadBMP 直接输出 CHW
// 本函数提供 TIFF → 引擎的格式桥接
// tiffImg: 输入 TiffImage（HWC 排列）
// 返回: RawImage（CHW 排列，float [0,1]）
RawImage tiffToRawImage(const TiffImage& tiffImg) {
    RawImage img;  // 20260330 ZJH 结果结构体
    if (!tiffImg.isValid()) return img;  // 20260330 ZJH 无效输入直接返回

    img.nWidth = tiffImg.nWidth;
    img.nHeight = tiffImg.nHeight;
    img.nChannels = tiffImg.nChannels;

    // 20260330 ZJH HWC → CHW 转换
    size_t nH = static_cast<size_t>(tiffImg.nHeight);
    size_t nW = static_cast<size_t>(tiffImg.nWidth);
    size_t nC = static_cast<size_t>(tiffImg.nChannels);
    img.vecData.resize(nC * nH * nW);

    for (size_t c = 0; c < nC; ++c) {
        for (size_t y = 0; y < nH; ++y) {
            for (size_t x = 0; x < nW; ++x) {
                // 20260330 ZJH HWC 索引: y * W * C + x * C + c
                size_t nHwcIdx = y * nW * nC + x * nC + c;
                // 20260330 ZJH CHW 索引: c * H * W + y * W + x
                size_t nChwIdx = c * nH * nW + y * nW + x;
                img.vecData[nChwIdx] = tiffImg.vecData[nHwcIdx];
            }
        }
    }

    return img;  // 20260330 ZJH 返回 CHW 格式的 RawImage
}

// 20260330 ZJH loadImageAuto — 自动识别文件格式并加载图像
// 支持 BMP (.bmp) 和 TIFF (.tif, .tiff) 格式
// 返回引擎标准 RawImage (CHW, float [0,1])
// 扩展格式时只需在此函数添加分支
RawImage loadImageAuto(const std::string& strPath) {
    // 20260330 ZJH 提取文件扩展名（转小写）
    std::string strExt;
    size_t nDot = strPath.rfind('.');
    if (nDot != std::string::npos) {
        strExt = strPath.substr(nDot);
        // 20260330 ZJH 转小写
        for (auto& c : strExt) {
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        }
    }

    // 20260330 ZJH 根据扩展名选择加载器
    if (strExt == ".bmp") {
        return loadBMP(strPath);  // 20260330 ZJH BMP 加载
    }
    if (strExt == ".tif" || strExt == ".tiff") {
        TiffImage tiffImg = loadTiff(strPath);  // 20260330 ZJH TIFF 加载
        return tiffToRawImage(tiffImg);          // 20260330 ZJH HWC → CHW 转换
    }

    // 20260330 ZJH 尝试按 TIFF 格式加载（某些工业设备输出的文件无标准扩展名）
    // 检查文件头魔数是否为 TIFF
    std::ifstream probe(strPath, std::ios::binary);
    if (probe.is_open()) {
        char arrHeader[4] = {};
        probe.read(arrHeader, 4);
        probe.close();
        // 20260330 ZJH 检查 TIFF 魔数: "II\x2A\x00" 或 "MM\x00\x2A"
        bool bIsTiff = ((arrHeader[0] == 'I' && arrHeader[1] == 'I'
                         && arrHeader[2] == 0x2A && arrHeader[3] == 0x00)
                     || (arrHeader[0] == 'M' && arrHeader[1] == 'M'
                         && arrHeader[2] == 0x00 && arrHeader[3] == 0x2A));
        if (bIsTiff) {
            TiffImage tiffImg = loadTiff(strPath);
            return tiffToRawImage(tiffImg);
        }
        // 20260330 ZJH 检查 BMP 魔数: "BM"
        if (arrHeader[0] == 'B' && arrHeader[1] == 'M') {
            return loadBMP(strPath);
        }
    }

    // 20260330 ZJH 无法识别的格式，默认尝试 BMP
    std::cerr << "[loadImageAuto] WARNING: unknown format for '" << strPath
              << "', trying BMP loader" << std::endl;
    return loadBMP(strPath);
}

// =========================================================
// 20260406 ZJH 数据增强 — 工业级图像增强管线
// =========================================================

// 20260320 ZJH 前向声明 resizeImage（augmentImage 的随机缩放需要调用）
// 20260330 ZJH 更新签名：新增 eMode 参数支持多种插值模式，默认双线性插值
std::vector<float> resizeImage(const std::vector<float>& vecSrc,
                                int nC, int nSrcH, int nSrcW,
                                int nDstH, int nDstW,
                                ResizeMode eMode = ResizeMode::Bilinear);

// 20260320 ZJH AugmentConfig — 完整工业级数据增强配置
struct AugmentConfig {
    // ---- 几何变换 ----
    bool bRandomHFlip = true;          // 20260320 ZJH 随机水平翻转（镜像）
    bool bRandomVFlip = false;         // 20260320 ZJH 随机垂直翻转
    bool bRandomRotate90 = false;      // 20260320 ZJH 随机 90°/180°/270° 旋转
    bool bRandomRotate = false;        // 20260320 ZJH 随机任意角度旋转
    float fRotateRange = 15.0f;        // 20260320 ZJH 旋转角度范围 [-range, +range] 度
    bool bRandomScale = false;         // 20260320 ZJH 随机缩放
    float fScaleMin = 0.8f;            // 20260320 ZJH 最小缩放比
    float fScaleMax = 1.2f;            // 20260320 ZJH 最大缩放比
    bool bRandomTranslate = false;     // 20260320 ZJH 随机平移
    float fTranslateRange = 0.1f;      // 20260320 ZJH 平移比例 [0,1]
    bool bRandomShear = false;         // 20260320 ZJH 随机错切
    float fShearRange = 5.0f;          // 20260320 ZJH 错切角度范围（度）
    bool bRandomCrop = false;          // 20260320 ZJH 随机裁剪
    float fCropRatio = 0.9f;           // 20260320 ZJH 裁剪保留比例

    // ---- 颜色/灰度变换 ----
    bool bColorJitter = false;         // 20260320 ZJH 颜色抖动
    float fJitterBrightness = 0.1f;    // 20260320 ZJH 亮度变化幅度
    float fJitterContrast = 0.1f;      // 20260320 ZJH 对比度变化幅度
    float fJitterSaturation = 0.1f;    // 20260320 ZJH 饱和度变化幅度
    float fJitterHue = 0.02f;          // 20260320 ZJH 色调变化幅度
    bool bGammaCorrection = false;     // 20260320 ZJH 伽马校正
    float fGammaMin = 0.7f;            // 20260320 ZJH 最小伽马值
    float fGammaMax = 1.5f;            // 20260320 ZJH 最大伽马值
    bool bHistogramEQ = false;         // 20260320 ZJH 直方图均衡化
    bool bCLAHE = false;              // 20260320 ZJH 自适应直方图均衡（CLAHE）
    bool bInvert = false;              // 20260320 ZJH 随机反色
    float fInvertProb = 0.1f;          // 20260320 ZJH 反色概率
    bool bGrayscale = false;           // 20260320 ZJH 随机转灰度（RGB→Gray→RGB）
    float fGrayscaleProb = 0.1f;       // 20260320 ZJH 转灰度概率

    // ---- 噪声/模糊 ----
    bool bGaussianNoise = false;       // 20260320 ZJH 高斯噪声
    float fNoiseStd = 0.01f;           // 20260320 ZJH 噪声标准差
    bool bGaussianBlur = false;        // 20260320 ZJH 高斯模糊
    int nBlurKernelSize = 3;           // 20260320 ZJH 模糊核大小（奇数）
    float fBlurSigma = 1.0f;           // 20260320 ZJH 模糊 sigma
    bool bSaltPepper = false;          // 20260320 ZJH 椒盐噪声
    float fSaltPepperProb = 0.01f;     // 20260320 ZJH 椒盐概率

    // ---- HSV 空间颜色变换（对标海康 VisionTrain） ----
    bool bHsvJitter = false;           // 20260330 ZJH HSV 空间颜色抖动（独立控制 H/S/V 三通道）
    float fHueShift = 5.0f;            // 20260330 ZJH 色调偏移范围 [-fHueShift, +fHueShift]，映射到 [-127,128] 整数区间
    float fSatShift = 5.0f;            // 20260330 ZJH 饱和度偏移范围 [-fSatShift, +fSatShift]，映射到 [-255,255] 整数区间
    float fValShift = 5.0f;            // 20260330 ZJH 明度偏移范围 [-fValShift, +fValShift]，映射到 [-255,255] 整数区间

    // ---- 形态学操作（对标海康 VisionTrain） ----
    bool bMorphology = false;          // 20260330 ZJH 随机形态学操作（腐蚀或膨胀，各 50% 概率）
    int nMorphKernelSize = 3;          // 20260330 ZJH 形态学核大小（3/5/7），必须为奇数

    // ---- 画布扩展（对标海康 VisionTrain） ----
    bool bCanvasExpand = false;        // 20260330 ZJH 画布扩展（将原图随机放置在更大画布上，灰色填充）
    float fCanvasRatio = 1.05f;        // 20260330 ZJH 画布扩展比例 [1.0, 2.0]，1.0 表示不扩展

    // ---- 遮挡/混合（高级） ----
    bool bCutOut = false;              // 20260320 ZJH CutOut 随机遮挡
    int nCutOutSize = 8;               // 20260320 ZJH CutOut 遮挡区域大小
    int nCutOutCount = 1;              // 20260320 ZJH CutOut 遮挡数量
    bool bRandomErasing = false;       // 20260320 ZJH 随机擦除（Random Erasing）
    float fErasingProb = 0.5f;         // 20260320 ZJH 擦除概率
    float fErasingMinArea = 0.02f;     // 20260320 ZJH 最小擦除面积比
    float fErasingMaxArea = 0.33f;     // 20260320 ZJH 最大擦除面积比

    // ---- 20260402 ZJH [OPT-2.3] 高级增强（对标 Anomalib/YOLO/CutPaste 论文）----

    // 20260402 ZJH CutPaste — 异常检测专用自监督增强
    // 论文: CutPaste: Self-Supervised Learning for Anomaly Detection (Li et al., 2021)
    // 从同一图像随机裁剪 patch 并粘贴到另一位置，生成 "伪异常" 样本
    // 训练分类器区分正常/CutPaste 图像，间接学习正常样本分布
    bool bCutPaste = false;            // 20260402 ZJH 启用 CutPaste 增强
    float fCutPasteMinArea = 0.02f;    // 20260402 ZJH 裁剪区域最小面积比（相对图像总面积）
    float fCutPasteMaxArea = 0.15f;    // 20260402 ZJH 裁剪区域最大面积比
    float fCutPasteMinAspect = 0.3f;   // 20260402 ZJH 裁剪区域最小宽高比
    float fCutPasteMaxAspect = 3.3f;   // 20260402 ZJH 裁剪区域最大宽高比

    // 20260402 ZJH MixUp — 图像级混合增强
    // 论文: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    // image = λ × img1 + (1-λ) × img2, label = λ × label1 + (1-λ) × label2
    // λ ~ Beta(α, α)，α 越小混合越少，α=0.2 为推荐值
    // 效果: 强正则化，防过拟合，尤其对小数据集分类有效 +2-4%
    bool bMixUp = false;               // 20260402 ZJH 启用 MixUp
    float fMixUpAlpha = 0.2f;          // 20260402 ZJH Beta 分布参数 α（越大混合越强）

    // 20260402 ZJH CutMix — 区域级混合增强
    // 论文: CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    // 随机裁剪 img2 的一块区域粘贴到 img1，标签按面积比混合
    // 相比 MixUp: CutMix 保留局部结构信息，对定位类任务更友好
    bool bCutMix = false;              // 20260402 ZJH 启用 CutMix
    float fCutMixAlpha = 1.0f;         // 20260402 ZJH Beta 分布参数 α

    // 20260402 ZJH Mosaic — 4 图拼接增强（YOLO 系列标配）
    // 将 4 张训练图像拼接成一张，每张占据四分之一区域
    // 效果: 增加上下文多样性，小目标检测 mAP +3-5%
    // 适用: 目标检测/实例分割任务
    bool bMosaic = false;              // 20260402 ZJH 启用 Mosaic 4 图拼接
    float fMosaicProb = 0.5f;          // 20260402 ZJH Mosaic 触发概率

    // 20260402 ZJH Elastic Deformation — 弹性形变
    // 对标 U-Net 论文的弹性变形增强（医学图像/布料/柔性材料检测）
    // 在随机位移场上施加高斯平滑，模拟自然形变
    bool bElasticDeform = false;       // 20260402 ZJH 启用弹性形变
    float fElasticAlpha = 50.0f;       // 20260402 ZJH 形变幅度（像素），越大变形越剧烈
    float fElasticSigma = 5.0f;        // 20260402 ZJH 高斯平滑 sigma，越大越平滑

    // ---- 归一化 ----
    bool bNormalize = true;            // 20260320 ZJH 是否归一化
    float fMeanR = 0.5f;
    float fMeanG = 0.5f;
    float fMeanB = 0.5f;
    float fStdR = 0.5f;
    float fStdG = 0.5f;
    float fStdB = 0.5f;

    // 20260330 ZJH 归一化预设模式（优先于手动 fMeanR/fStdR 配置）
    // 当 eNormPreset != None 时，使用预设值覆盖手动配置
    NormPreset eNormPreset = NormPreset::None;

    // 20260330 ZJH 逐通道自定义 mean/std（仅 NormPreset::MeanStd 模式下生效）
    // RGB 三通道顺序，对于灰度图仅使用第 0 通道
    std::array<float, 3> vecMeanPerChannel = {0.5f, 0.5f, 0.5f};  // 20260330 ZJH 用户自定义逐通道均值
    std::array<float, 3> vecStdPerChannel = {0.5f, 0.5f, 0.5f};   // 20260330 ZJH 用户自定义逐通道标准差
};

// 20260330 ZJH normalizeImage — 图像归一化（支持多种预设模式）
// data: CHW 格式的图像数据，float [0,1]，就地修改
// nC: 通道数, nH: 高度, nW: 宽度
// config: 增强配置（包含归一化参数）
void normalizeImage(std::vector<float>& data, int nC, int nH, int nW,
                    const AugmentConfig& config) {
    int nSpatial = nH * nW;  // 20260330 ZJH 每通道像素数

    // 20260330 ZJH 根据预设模式确定实际使用的 mean/std 值
    std::array<float, 3> arrMean = {0.0f, 0.0f, 0.0f};  // 20260330 ZJH 各通道均值
    std::array<float, 3> arrStd = {1.0f, 1.0f, 1.0f};   // 20260330 ZJH 各通道标准差

    switch (config.eNormPreset) {
    case NormPreset::None:
        // 20260330 ZJH 不归一化，直接返回
        return;

    case NormPreset::ZeroOne:
        // 20260330 ZJH 仅 /255 归一化（输入已经是 [0,1]，此模式下无额外操作）
        return;

    case NormPreset::MeanStd:
        // 20260330 ZJH 使用用户自定义逐通道 mean/std
        arrMean = config.vecMeanPerChannel;
        arrStd = config.vecStdPerChannel;
        break;

    case NormPreset::ImageNet:
        // 20260330 ZJH ImageNet 标准归一化参数（RGB 通道顺序）
        // 来源: torchvision.transforms.Normalize
        arrMean = {0.485f, 0.456f, 0.406f};
        arrStd = {0.229f, 0.224f, 0.225f};
        break;
    }

    // 20260330 ZJH 逐通道应用归一化: out = (pixel - mean) * invStd
    // 20260330 ZJH 预计算逆标准差，乘法比除法快 3-5 倍（BP-3 性能优化）
    if (nC >= 3) {
        // 20260330 ZJH 多通道图像（RGB），逐通道独立归一化
        for (int c = 0; c < nC; ++c) {
            // 20260330 ZJH 对超过 3 通道的情况，第 3+ 通道复用第 2 通道参数
            int nParamIdx = std::min(c, 2);
            float fMean = arrMean[static_cast<size_t>(nParamIdx)];  // 20260330 ZJH 当前通道均值
            float fStd = arrStd[static_cast<size_t>(nParamIdx)];    // 20260330 ZJH 当前通道标准差
            // 20260330 ZJH 防止除零
            if (fStd < 1e-7f) fStd = 1e-7f;
            // 20260330 ZJH 预计算逆标准差，内循环用乘法替代除法
            float fInvStd = 1.0f / fStd;
            for (int i = 0; i < nSpatial; ++i) {
                data[static_cast<size_t>(c * nSpatial + i)] =
                    (data[static_cast<size_t>(c * nSpatial + i)] - fMean) * fInvStd;
            }
        }
    } else {
        // 20260330 ZJH 灰度图像，使用第 0 通道参数
        float fMean = arrMean[0];
        float fStd = arrStd[0];
        if (fStd < 1e-7f) fStd = 1e-7f;  // 20260330 ZJH 防止除零
        // 20260330 ZJH 预计算逆标准差，内循环用乘法替代除法
        float fInvStd = 1.0f / fStd;
        for (auto& v : data) {
            v = (v - fMean) * fInvStd;
        }
    }
}

// 20260320 ZJH augmentImage — 完整工业级数据增强
// data: 输入图像数据 CHW 格式，float [0,1]，inplace 修改
void augmentImage(std::vector<float>& data, int nC, int nH, int nW,
                  const AugmentConfig& config) {
    static thread_local std::mt19937 gen(std::random_device{}());  // 20260406 ZJH 线程局部随机数引擎，避免多线程竞争
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);  // 20260406 ZJH [0,1] 均匀分布，用于概率判断
    int nSpatial = nH * nW;  // 20260406 ZJH 每通道像素数（用于 CHW 索引计算）

    // ====== 20260406 ZJH 几何变换 ======

    // 20260320 ZJH 随机水平翻转（镜像）
    if (config.bRandomHFlip && dist01(gen) > 0.5f) {
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW / 2; ++x) {
                    size_t i1 = static_cast<size_t>(c * nSpatial + y * nW + x);
                    size_t i2 = static_cast<size_t>(c * nSpatial + y * nW + (nW - 1 - x));
                    std::swap(data[i1], data[i2]);
                }
    }

    // 20260320 ZJH 随机垂直翻转
    if (config.bRandomVFlip && dist01(gen) > 0.5f) {
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH / 2; ++y)
                for (int x = 0; x < nW; ++x) {
                    size_t i1 = static_cast<size_t>(c * nSpatial + y * nW + x);
                    size_t i2 = static_cast<size_t>(c * nSpatial + (nH - 1 - y) * nW + x);
                    std::swap(data[i1], data[i2]);
                }
    }

    // 20260320 ZJH 随机 90°/180°/270° 旋转
    if (config.bRandomRotate90 && nH == nW) {
        int nRot = static_cast<int>(dist01(gen) * 4.0f) % 4;  // 0/1/2/3
        if (nRot > 0) {
            std::vector<float> tmp(data.size());
            for (int r = 0; r < nRot; ++r) {
                for (int c = 0; c < nC; ++c)
                    for (int y = 0; y < nH; ++y)
                        for (int x = 0; x < nW; ++x) {
                            // 20260320 ZJH 顺时针 90°: (x,y) -> (y, W-1-x)
                            int ny = x, nx = nW - 1 - y;
                            tmp[static_cast<size_t>(c * nSpatial + ny * nW + nx)] =
                                data[static_cast<size_t>(c * nSpatial + y * nW + x)];
                        }
                data = tmp;
            }
        }
    }

    // 20260320 ZJH 随机任意角度旋转（双线性插值）
    if (config.bRandomRotate) {
        float fAngleDeg = (dist01(gen) * 2.0f - 1.0f) * config.fRotateRange;
        float fAngleRad = fAngleDeg * 3.14159265f / 180.0f;
        float fCos = std::cos(fAngleRad), fSin = std::sin(fAngleRad);
        float fCx = nW * 0.5f, fCy = nH * 0.5f;

        std::vector<float> rotated(data.size(), 0.0f);
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    // 20260320 ZJH 逆映射：从目标查源
                    float fSrcX = fCos * (x - fCx) + fSin * (y - fCy) + fCx;
                    float fSrcY = -fSin * (x - fCx) + fCos * (y - fCy) + fCy;
                    // 20260320 ZJH 双线性插值
                    int x0 = static_cast<int>(std::floor(fSrcX));
                    int y0 = static_cast<int>(std::floor(fSrcY));
                    float fx = fSrcX - x0, fy = fSrcY - y0;
                    auto getP = [&](int py, int px) -> float {
                        py = std::max(0, std::min(nH - 1, py));
                        px = std::max(0, std::min(nW - 1, px));
                        return data[static_cast<size_t>(c * nSpatial + py * nW + px)];
                    };
                    rotated[static_cast<size_t>(c * nSpatial + y * nW + x)] =
                        (1 - fy) * ((1 - fx) * getP(y0, x0) + fx * getP(y0, x0 + 1)) +
                        fy * ((1 - fx) * getP(y0 + 1, x0) + fx * getP(y0 + 1, x0 + 1));
                }
        data = rotated;
    }

    // 20260320 ZJH 随机缩放（通过 resize）
    if (config.bRandomScale) {
        float fScale = config.fScaleMin + dist01(gen) * (config.fScaleMax - config.fScaleMin);
        int nNewH = static_cast<int>(nH * fScale);
        int nNewW = static_cast<int>(nW * fScale);
        if (nNewH > 0 && nNewW > 0 && (nNewH != nH || nNewW != nW)) {
            auto scaled = resizeImage(data, nC, nH, nW, nNewH, nNewW);
            // 20260320 ZJH 中心裁剪或填充回原始尺寸
            data.assign(static_cast<size_t>(nC * nH * nW), 0.0f);
            int nOffY = (nNewH - nH) / 2, nOffX = (nNewW - nW) / 2;
            for (int c = 0; c < nC; ++c)
                for (int y = 0; y < nH; ++y)
                    for (int x = 0; x < nW; ++x) {
                        int sy = y + nOffY, sx = x + nOffX;
                        if (sy >= 0 && sy < nNewH && sx >= 0 && sx < nNewW)
                            data[static_cast<size_t>(c * nSpatial + y * nW + x)] =
                                scaled[static_cast<size_t>(c * nNewH * nNewW + sy * nNewW + sx)];
                    }
        }
    }

    // ====== 20260406 ZJH 颜色/灰度变换 ======

    // 20260320 ZJH 亮度 + 对比度抖动
    if (config.bColorJitter) {
        float fBright = (dist01(gen) * 2.0f - 1.0f) * config.fJitterBrightness;
        float fContrast = 1.0f + (dist01(gen) * 2.0f - 1.0f) * config.fJitterContrast;
        for (auto& v : data) {
            v = fContrast * (v - 0.5f) + 0.5f + fBright;
            v = std::max(0.0f, std::min(1.0f, v));
        }
    }

    // 20260320 ZJH 伽马校正：out = in^gamma
    if (config.bGammaCorrection) {
        float fGamma = config.fGammaMin + dist01(gen) * (config.fGammaMax - config.fGammaMin);
        for (auto& v : data) {
            v = std::pow(std::max(0.0f, v), fGamma);
            v = std::min(1.0f, v);
        }
    }

    // 20260320 ZJH 直方图均衡化（逐通道）
    if (config.bHistogramEQ) {
        for (int c = 0; c < nC; ++c) {
            // 20260320 ZJH 计算直方图（256 bins）
            int hist[256] = {};
            for (int i = 0; i < nSpatial; ++i) {
                int bin = static_cast<int>(data[static_cast<size_t>(c * nSpatial + i)] * 255.0f);
                bin = std::max(0, std::min(255, bin));
                hist[bin]++;
            }
            // 20260320 ZJH CDF 累积
            int cdf[256];
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i) cdf[i] = cdf[i - 1] + hist[i];
            int nCdfMin = 0;
            for (int i = 0; i < 256; ++i) { if (cdf[i] > 0) { nCdfMin = cdf[i]; break; } }
            float fDenom = static_cast<float>(nSpatial - nCdfMin);
            if (fDenom < 1.0f) fDenom = 1.0f;
            // 20260320 ZJH 映射
            for (int i = 0; i < nSpatial; ++i) {
                int bin = static_cast<int>(data[static_cast<size_t>(c * nSpatial + i)] * 255.0f);
                bin = std::max(0, std::min(255, bin));
                data[static_cast<size_t>(c * nSpatial + i)] =
                    static_cast<float>(cdf[bin] - nCdfMin) / fDenom;
            }
        }
    }

    // 20260320 ZJH 随机反色
    if (config.bInvert && dist01(gen) < config.fInvertProb) {
        for (auto& v : data) v = 1.0f - v;
    }

    // 20260320 ZJH 随机转灰度（对 RGB 图像）
    if (config.bGrayscale && nC == 3 && dist01(gen) < config.fGrayscaleProb) {
        for (int i = 0; i < nSpatial; ++i) {
            float fGray = 0.299f * data[static_cast<size_t>(i)]
                        + 0.587f * data[static_cast<size_t>(nSpatial + i)]
                        + 0.114f * data[static_cast<size_t>(2 * nSpatial + i)];
            data[static_cast<size_t>(i)] = fGray;
            data[static_cast<size_t>(nSpatial + i)] = fGray;
            data[static_cast<size_t>(2 * nSpatial + i)] = fGray;
        }
    }

    // ====== HSV 空间颜色变换（对标海康 VisionTrain） ======

    // 20260330 ZJH HSV 空间颜色抖动 — RGB→HSV→偏移H/S/V→HSV→RGB
    // 独立控制色调、饱和度、明度，比简单的亮度/对比度抖动更精细
    // 实现: 纯 C++ 手写 RGB↔HSV 转换，不依赖 OpenCV
    if (config.bHsvJitter && nC == 3) {
        std::uniform_real_distribution<float> distHue(-config.fHueShift, config.fHueShift);    // 20260330 ZJH H 偏移分布
        std::uniform_real_distribution<float> distSat(-config.fSatShift, config.fSatShift);    // 20260330 ZJH S 偏移分布
        std::uniform_real_distribution<float> distVal(-config.fValShift, config.fValShift);    // 20260330 ZJH V 偏移分布

        float fHueDelta = distHue(gen);   // 20260330 ZJH 当前帧的色调偏移量
        float fSatDelta = distSat(gen);   // 20260330 ZJH 当前帧的饱和度偏移量
        float fValDelta = distVal(gen);   // 20260330 ZJH 当前帧的明度偏移量

        for (int i = 0; i < nSpatial; ++i) {
            // 20260330 ZJH 读取 RGB 值（CHW 格式，[0,1] 范围）
            float fR = data[static_cast<size_t>(0 * nSpatial + i)];
            float fG = data[static_cast<size_t>(1 * nSpatial + i)];
            float fB = data[static_cast<size_t>(2 * nSpatial + i)];

            // 20260330 ZJH RGB → HSV 转换（H: [0,360), S: [0,1], V: [0,1]）
            float fMax = std::max({fR, fG, fB});  // 20260330 ZJH 最大通道值 = V
            float fMin = std::min({fR, fG, fB});  // 20260330 ZJH 最小通道值
            float fDelta = fMax - fMin;             // 20260330 ZJH 色差

            float fH = 0.0f;  // 20260330 ZJH 色调 [0, 360)
            float fS = 0.0f;  // 20260330 ZJH 饱和度 [0, 1]
            float fV = fMax;   // 20260330 ZJH 明度 [0, 1]

            // 20260330 ZJH 计算饱和度
            if (fMax > 1e-7f) {
                fS = fDelta / fMax;
            }

            // 20260330 ZJH 计算色调（当 fDelta 接近 0 时为灰色，H 无意义保持 0）
            if (fDelta > 1e-7f) {
                if (fMax == fR) {
                    fH = 60.0f * (fG - fB) / fDelta;              // 20260330 ZJH 红色为主色调
                    if (fH < 0.0f) fH += 360.0f;                   // 20260330 ZJH 负值修正
                } else if (fMax == fG) {
                    fH = 60.0f * (2.0f + (fB - fR) / fDelta);     // 20260330 ZJH 绿色为主色调
                } else {
                    fH = 60.0f * (4.0f + (fR - fG) / fDelta);     // 20260330 ZJH 蓝色为主色调
                }
            }

            // 20260330 ZJH 应用 HSV 偏移
            // fHueShift 对应 [-127,128] 映射到 360°空间: delta * (360/128)
            fH += fHueDelta * (360.0f / 128.0f);
            // 20260330 ZJH 色调环绕 [0, 360) — 使用 fmod 处理多圈偏移
            fH = std::fmod(fH, 360.0f);           // 20260330 ZJH 先取模到 (-360, 360)
            if (fH < 0.0f) fH += 360.0f;           // 20260330 ZJH 负值修正到 [0, 360)

            // 20260330 ZJH 饱和度偏移（fSatShift 对应 [-255,255] 归一化到 [0,1]）
            fS += fSatDelta / 255.0f;
            fS = std::max(0.0f, std::min(1.0f, fS));  // 20260330 ZJH 钳制到 [0,1]

            // 20260330 ZJH 明度偏移（fValShift 对应 [-255,255] 归一化到 [0,1]）
            fV += fValDelta / 255.0f;
            fV = std::max(0.0f, std::min(1.0f, fV));  // 20260330 ZJH 钳制到 [0,1]

            // 20260330 ZJH HSV → RGB 反转换
            float fC = fV * fS;                                      // 20260330 ZJH 色度
            float fHPrime = fH / 60.0f;                               // 20260330 ZJH 色调扇区 [0,6)
            float fX = fC * (1.0f - std::abs(std::fmod(fHPrime, 2.0f) - 1.0f));  // 20260330 ZJH 中间值
            float fM = fV - fC;                                       // 20260330 ZJH 明度偏移

            float fR2 = 0.0f, fG2 = 0.0f, fB2 = 0.0f;  // 20260330 ZJH 输出 RGB
            if (fHPrime < 1.0f)      { fR2 = fC; fG2 = fX; fB2 = 0.0f; }  // 20260330 ZJH 扇区 0
            else if (fHPrime < 2.0f) { fR2 = fX; fG2 = fC; fB2 = 0.0f; }  // 20260330 ZJH 扇区 1
            else if (fHPrime < 3.0f) { fR2 = 0.0f; fG2 = fC; fB2 = fX; }  // 20260330 ZJH 扇区 2
            else if (fHPrime < 4.0f) { fR2 = 0.0f; fG2 = fX; fB2 = fC; }  // 20260330 ZJH 扇区 3
            else if (fHPrime < 5.0f) { fR2 = fX; fG2 = 0.0f; fB2 = fC; }  // 20260330 ZJH 扇区 4
            else                     { fR2 = fC; fG2 = 0.0f; fB2 = fX; }  // 20260330 ZJH 扇区 5

            // 20260330 ZJH 加上明度偏移，写回 CHW 数据
            data[static_cast<size_t>(0 * nSpatial + i)] = std::max(0.0f, std::min(1.0f, fR2 + fM));
            data[static_cast<size_t>(1 * nSpatial + i)] = std::max(0.0f, std::min(1.0f, fG2 + fM));
            data[static_cast<size_t>(2 * nSpatial + i)] = std::max(0.0f, std::min(1.0f, fB2 + fM));
        }
    }

    // ====== 形态学操作（腐蚀/膨胀，对标海康 VisionTrain） ======

    // 20260330 ZJH 随机形态学操作 — 在灰度/逐通道上执行腐蚀或膨胀
    // 腐蚀: 局部最小值（缩小亮区域）; 膨胀: 局部最大值（扩大亮区域）
    // 50% 概率选择腐蚀，50% 概率选择膨胀
    if (config.bMorphology) {
        bool bErode = (dist01(gen) < 0.5f);  // 20260330 ZJH true=腐蚀, false=膨胀
        int nK = config.nMorphKernelSize;     // 20260330 ZJH 核大小
        if (nK % 2 == 0) nK++;                // 20260330 ZJH 确保核大小为奇数
        int nR = nK / 2;                       // 20260330 ZJH 核半径

        std::vector<float> vecMorphResult(data.size());  // 20260330 ZJH 形态学操作结果缓冲区

        for (int c = 0; c < nC; ++c) {
            for (int y = 0; y < nH; ++y) {
                for (int x = 0; x < nW; ++x) {
                    float fExtreme = bErode ? 1.0f : 0.0f;  // 20260330 ZJH 腐蚀取最小值初始为1，膨胀取最大值初始为0

                    // 20260330 ZJH 在核窗口内搜索极值
                    for (int ky = -nR; ky <= nR; ++ky) {
                        for (int kx = -nR; kx <= nR; ++kx) {
                            // 20260330 ZJH 边界钳制
                            int sy = std::max(0, std::min(nH - 1, y + ky));
                            int sx = std::max(0, std::min(nW - 1, x + kx));
                            float fVal = data[static_cast<size_t>(c * nSpatial + sy * nW + sx)];

                            if (bErode) {
                                fExtreme = std::min(fExtreme, fVal);   // 20260330 ZJH 腐蚀: 取局部最小值
                            } else {
                                fExtreme = std::max(fExtreme, fVal);   // 20260330 ZJH 膨胀: 取局部最大值
                            }
                        }
                    }

                    vecMorphResult[static_cast<size_t>(c * nSpatial + y * nW + x)] = fExtreme;
                }
            }
        }
        data = vecMorphResult;  // 20260330 ZJH 写回结果
    }

    // ====== 画布扩展（对标海康 VisionTrain Canvas Expand） ======

    // 20260330 ZJH 画布扩展 — 创建更大画布，将原图随机放置其中，灰色填充边缘
    // 模拟实际工业场景中目标在视野中位置偏移的情况
    // fCanvasRatio 控制扩展比例，1.0 表示不扩展，2.0 表示画布面积 4 倍
    if (config.bCanvasExpand && config.fCanvasRatio > 1.0f) {
        // 20260330 ZJH 计算扩展后的画布尺寸
        int nCanvasH = static_cast<int>(nH * config.fCanvasRatio);  // 20260330 ZJH 扩展后高度
        int nCanvasW = static_cast<int>(nW * config.fCanvasRatio);  // 20260330 ZJH 扩展后宽度
        int nCanvasSpatial = nCanvasH * nCanvasW;                    // 20260330 ZJH 扩展画布每通道像素数

        // 20260330 ZJH 创建灰色画布（114/255 ≈ 0.447f，与 YOLO letterbox 一致）
        const float fPadGray = 114.0f / 255.0f;
        std::vector<float> vecCanvas(static_cast<size_t>(nC * nCanvasSpatial), fPadGray);

        // 20260330 ZJH 随机选择原图在画布中的放置位置
        int nMaxOffY = nCanvasH - nH;  // 20260330 ZJH Y 方向最大偏移
        int nMaxOffX = nCanvasW - nW;  // 20260330 ZJH X 方向最大偏移
        int nOffY = (nMaxOffY > 0) ? static_cast<int>(dist01(gen) * static_cast<float>(nMaxOffY)) : 0;  // 20260330 ZJH 随机 Y 偏移
        int nOffX = (nMaxOffX > 0) ? static_cast<int>(dist01(gen) * static_cast<float>(nMaxOffX)) : 0;  // 20260330 ZJH 随机 X 偏移

        // 20260330 ZJH 将原图拷贝到画布上的随机位置
        for (int c = 0; c < nC; ++c) {
            for (int y = 0; y < nH; ++y) {
                for (int x = 0; x < nW; ++x) {
                    int nDstY = y + nOffY;  // 20260330 ZJH 画布目标行
                    int nDstX = x + nOffX;  // 20260330 ZJH 画布目标列
                    vecCanvas[static_cast<size_t>(c * nCanvasSpatial + nDstY * nCanvasW + nDstX)] =
                        data[static_cast<size_t>(c * nSpatial + y * nW + x)];
                }
            }
        }

        // 20260330 ZJH 将扩展画布缩放回原始尺寸，保持数据维度一致
        data = resizeImage(vecCanvas, nC, nCanvasH, nCanvasW, nH, nW, ResizeMode::Bilinear);
    }

    // ====== 20260406 ZJH 噪声/模糊 ======

    // 20260320 ZJH 高斯噪声
    if (config.bGaussianNoise) {
        std::normal_distribution<float> noiseDist(0.0f, config.fNoiseStd);
        for (auto& v : data) v = std::max(0.0f, std::min(1.0f, v + noiseDist(gen)));
    }

    // 20260330 ZJH 高斯模糊（可分离 1D 两趟卷积，OPT-1 性能优化）
    // 将 O(K²) 2D 卷积拆分为水平+垂直两趟 O(2K) 可分离卷积
    // 对 5x5 核约 2.5x 加速，对 3x3 核约 1.8x 加速
    if (config.bGaussianBlur) {
        int nK = config.nBlurKernelSize;
        if (nK % 2 == 0) nK++;  // 20260330 ZJH 确保核大小为奇数
        int nR = nK / 2;  // 20260330 ZJH 核半径
        float fSig = config.fBlurSigma;

        // 20260330 ZJH 生成 1D 高斯核（可分离滤波器只需一维核）
        std::vector<float> kernel1d(static_cast<size_t>(nK));
        float fKernelSum = 0.0f;
        for (int k = -nR; k <= nR; ++k) {
            float fVal = std::exp(-(k * k) / (2.0f * fSig * fSig));  // 20260330 ZJH 高斯权重
            kernel1d[static_cast<size_t>(k + nR)] = fVal;
            fKernelSum += fVal;
        }
        // 20260330 ZJH 归一化 1D 核，使权重总和为 1
        for (auto& v : kernel1d) v /= fKernelSum;

        // 20260330 ZJH 第一趟：水平方向 1D 卷积（沿行方向）
        std::vector<float> vecTemp(data.size());  // 20260330 ZJH 中间缓冲区
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    float fSum = 0.0f;
                    for (int kx = -nR; kx <= nR; ++kx) {
                        // 20260330 ZJH 边界钳制（clamp 到图像边缘）
                        int sx = std::max(0, std::min(nW - 1, x + kx));
                        fSum += data[static_cast<size_t>(c * nSpatial + y * nW + sx)]
                              * kernel1d[static_cast<size_t>(kx + nR)];
                    }
                    vecTemp[static_cast<size_t>(c * nSpatial + y * nW + x)] = fSum;
                }

        // 20260330 ZJH 第二趟：垂直方向 1D 卷积（沿列方向）
        std::vector<float> blurred(data.size());  // 20260330 ZJH 最终结果缓冲区
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    float fSum = 0.0f;
                    for (int ky = -nR; ky <= nR; ++ky) {
                        // 20260330 ZJH 边界钳制（clamp 到图像边缘）
                        int sy = std::max(0, std::min(nH - 1, y + ky));
                        fSum += vecTemp[static_cast<size_t>(c * nSpatial + sy * nW + x)]
                              * kernel1d[static_cast<size_t>(ky + nR)];
                    }
                    blurred[static_cast<size_t>(c * nSpatial + y * nW + x)] = fSum;
                }
        data = blurred;
    }

    // 20260320 ZJH 椒盐噪声
    if (config.bSaltPepper) {
        for (int i = 0; i < nC * nSpatial; ++i) {
            float r = dist01(gen);
            if (r < config.fSaltPepperProb * 0.5f) data[static_cast<size_t>(i)] = 0.0f;       // 20260406 ZJH 椒（黑色像素）
            else if (r < config.fSaltPepperProb) data[static_cast<size_t>(i)] = 1.0f;  // 20260406 ZJH 盐（白色像素）
        }
    }

    // ====== 20260406 ZJH 遮挡/擦除 ======

    // 20260320 ZJH CutOut 随机遮挡
    if (config.bCutOut) {
        for (int k = 0; k < config.nCutOutCount; ++k) {
            int nCx = static_cast<int>(dist01(gen) * nW);
            int nCy = static_cast<int>(dist01(gen) * nH);
            int nHalf = config.nCutOutSize / 2;
            for (int c = 0; c < nC; ++c)
                for (int y = std::max(0, nCy - nHalf); y < std::min(nH, nCy + nHalf); ++y)
                    for (int x = std::max(0, nCx - nHalf); x < std::min(nW, nCx + nHalf); ++x)
                        data[static_cast<size_t>(c * nSpatial + y * nW + x)] = 0.0f;
        }
    }

    // 20260320 ZJH Random Erasing（随机擦除）
    if (config.bRandomErasing && dist01(gen) < config.fErasingProb) {
        // 20260406 ZJH 在 [minArea, maxArea] 范围内随机选择擦除面积
        float fArea = nH * nW * (config.fErasingMinArea + dist01(gen) * (config.fErasingMaxArea - config.fErasingMinArea));
        float fRatio = 0.3f + dist01(gen) * 2.7f;  // 20260406 ZJH 随机宽高比 [0.3, 3.0]
        int nErH = static_cast<int>(std::sqrt(fArea / fRatio));  // 20260406 ZJH 擦除区域高度
        int nErW = static_cast<int>(std::sqrt(fArea * fRatio));  // 20260406 ZJH 擦除区域宽度
        nErH = std::min(nErH, nH); nErW = std::min(nErW, nW);  // 20260406 ZJH 钳制到图像边界
        int nY0 = static_cast<int>(dist01(gen) * (nH - nErH));  // 20260406 ZJH 随机起始行
        int nX0 = static_cast<int>(dist01(gen) * (nW - nErW));  // 20260406 ZJH 随机起始列
        std::normal_distribution<float> eraseDist(0.5f, 0.2f);  // 20260406 ZJH 擦除填充值服从正态分布 N(0.5, 0.2)
        for (int c = 0; c < nC; ++c)
            for (int y = nY0; y < nY0 + nErH; ++y)
                for (int x = nX0; x < nX0 + nErW; ++x)
                    // 20260406 ZJH 用随机噪声填充擦除区域，钳制到 [0,1]
                    data[static_cast<size_t>(c * nSpatial + y * nW + x)] = std::max(0.0f, std::min(1.0f, eraseDist(gen)));
    }

    // ====== 20260406 ZJH 归一化（最后执行） ======
    // 20260330 ZJH 优先使用 NormPreset 模式，若 eNormPreset != None 则走新路径
    if (config.eNormPreset != NormPreset::None) {
        // 20260330 ZJH 使用新的逐通道归一化函数
        normalizeImage(data, nC, nH, nW, config);
    } else if (config.bNormalize) {
        // 20260320 ZJH 兼容旧的手动 mean/std 归一化（bNormalize 标志位路径）
        // 20260330 ZJH 预计算逆标准差，乘法替代除法（BP-3 性能优化）
        if (nC == 3) {
            float fMean[3] = {config.fMeanR, config.fMeanG, config.fMeanB};
            float fStd[3] = {config.fStdR, config.fStdG, config.fStdB};
            // 20260330 ZJH 预计算各通道逆标准差
            float fInvStd[3] = {1.0f / fStd[0], 1.0f / fStd[1], 1.0f / fStd[2]};
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < nSpatial; ++i)
                    data[static_cast<size_t>(c * nSpatial + i)] =
                        (data[static_cast<size_t>(c * nSpatial + i)] - fMean[c]) * fInvStd[c];
        } else {
            // 20260330 ZJH 灰度图预计算逆标准差
            float fInvStdR = 1.0f / config.fStdR;
            for (auto& v : data) v = (v - config.fMeanR) * fInvStdR;
        }
    }
}

// =========================================================
// 20260330 ZJH 图像缩放函数（支持多种插值模式）
// =========================================================

// 20260330 ZJH resizeImage — 图像缩放（CHW 格式）
// 将 [C, srcH, srcW] 缩放到 [C, dstH, dstW]
// 支持最近邻和双线性插值两种模式（Letterbox 请使用 letterboxImage）
// vecSrc: 源图像数据 CHW 格式
// nC: 通道数, nSrcH/nSrcW: 源尺寸, nDstH/nDstW: 目标尺寸
// eMode: 插值模式（默认 Bilinear）
std::vector<float> resizeImage(const std::vector<float>& vecSrc,
                                int nC, int nSrcH, int nSrcW,
                                int nDstH, int nDstW,
                                ResizeMode eMode) {
    std::vector<float> vecDst(static_cast<size_t>(nC * nDstH * nDstW));  // 20260330 ZJH 分配目标缓冲区

    if (eMode == ResizeMode::NearestNeighbor) {
        // 20260320 ZJH 最近邻插值（原始实现，速度快但有锯齿）
        for (int c = 0; c < nC; ++c) {
            for (int dy = 0; dy < nDstH; ++dy) {
                int nSrcY = dy * nSrcH / nDstH;  // 20260320 ZJH 最近邻映射
                for (int dx = 0; dx < nDstW; ++dx) {
                    int nSrcX = dx * nSrcW / nDstW;
                    vecDst[static_cast<size_t>(c * nDstH * nDstW + dy * nDstW + dx)] =
                        vecSrc[static_cast<size_t>(c * nSrcH * nSrcW + nSrcY * nSrcW + nSrcX)];
                }
            }
        }
    } else {
        // 20260330 ZJH 双线性插值（默认模式）
        // 对每个输出像素，计算对应的源图像浮点坐标，采样4个最近邻像素加权平均
        // 使用 align_corners=false 映射策略: srcCoord = (dstCoord + 0.5) * srcSize/dstSize - 0.5
        int nSrcSpatial = nSrcH * nSrcW;  // 20260330 ZJH 源图像每通道像素数
        int nDstSpatial = nDstH * nDstW;  // 20260330 ZJH 目标图像每通道像素数

        // 20260330 ZJH 预计算 Y 方向映射表（OPT-5 性能优化）
        // Y 映射仅依赖 dy，与 dx 和通道无关，提取到外层避免重复计算
        float fScaleY = static_cast<float>(nSrcH) / static_cast<float>(nDstH);  // 20260330 ZJH Y 缩放因子
        std::vector<int> vecY0(static_cast<size_t>(nDstH));     // 20260330 ZJH 预计算上方行索引
        std::vector<int> vecY1(static_cast<size_t>(nDstH));     // 20260330 ZJH 预计算下方行索引
        std::vector<float> vecFracY(static_cast<size_t>(nDstH));  // 20260330 ZJH 预计算 Y 插值权重
        for (int dy = 0; dy < nDstH; ++dy) {
            float fSrcY = (static_cast<float>(dy) + 0.5f) * fScaleY - 0.5f;  // 20260330 ZJH 源 Y 坐标
            int nY0 = static_cast<int>(std::floor(fSrcY));  // 20260330 ZJH 上方行（未钳制）
            vecFracY[static_cast<size_t>(dy)] = fSrcY - static_cast<float>(nY0);  // 20260330 ZJH 小数部分
            vecY0[static_cast<size_t>(dy)] = std::max(0, std::min(nSrcH - 1, nY0));      // 20260330 ZJH 钳制上方行
            vecY1[static_cast<size_t>(dy)] = std::max(0, std::min(nSrcH - 1, nY0 + 1));  // 20260330 ZJH 钳制下方行
        }

        // 20260330 ZJH 预计算 X 方向映射表（同理，X 映射仅依赖 dx）
        float fScaleX = static_cast<float>(nSrcW) / static_cast<float>(nDstW);  // 20260330 ZJH X 缩放因子
        std::vector<int> vecX0(static_cast<size_t>(nDstW));     // 20260330 ZJH 预计算左列索引
        std::vector<int> vecX1(static_cast<size_t>(nDstW));     // 20260330 ZJH 预计算右列索引
        std::vector<float> vecFracX(static_cast<size_t>(nDstW));  // 20260330 ZJH 预计算 X 插值权重
        for (int dx = 0; dx < nDstW; ++dx) {
            float fSrcX = (static_cast<float>(dx) + 0.5f) * fScaleX - 0.5f;  // 20260330 ZJH 源 X 坐标
            int nX0 = static_cast<int>(std::floor(fSrcX));  // 20260330 ZJH 左列（未钳制）
            vecFracX[static_cast<size_t>(dx)] = fSrcX - static_cast<float>(nX0);  // 20260330 ZJH 小数部分
            vecX0[static_cast<size_t>(dx)] = std::max(0, std::min(nSrcW - 1, nX0));      // 20260330 ZJH 钳制左列
            vecX1[static_cast<size_t>(dx)] = std::max(0, std::min(nSrcW - 1, nX0 + 1));  // 20260330 ZJH 钳制右列
        }

        // 20260330 ZJH 逐通道双线性插值，使用预计算映射表直接查表
        for (int c = 0; c < nC; ++c) {
            for (int dy = 0; dy < nDstH; ++dy) {
                // 20260330 ZJH 从预计算表中取出 Y 映射参数（无需重复计算）
                int nY0 = vecY0[static_cast<size_t>(dy)];
                int nY1 = vecY1[static_cast<size_t>(dy)];
                float fFracY = vecFracY[static_cast<size_t>(dy)];

                for (int dx = 0; dx < nDstW; ++dx) {
                    // 20260330 ZJH 从预计算表中取出 X 映射参数（无需重复计算）
                    int nX0 = vecX0[static_cast<size_t>(dx)];
                    int nX1 = vecX1[static_cast<size_t>(dx)];
                    float fFracX = vecFracX[static_cast<size_t>(dx)];

                    // 20260330 ZJH 采样4个邻居像素值
                    float fTopLeft = vecSrc[static_cast<size_t>(c * nSrcSpatial + nY0 * nSrcW + nX0)];      // 20260330 ZJH 左上
                    float fTopRight = vecSrc[static_cast<size_t>(c * nSrcSpatial + nY0 * nSrcW + nX1)];     // 20260330 ZJH 右上
                    float fBottomLeft = vecSrc[static_cast<size_t>(c * nSrcSpatial + nY1 * nSrcW + nX0)];   // 20260330 ZJH 左下
                    float fBottomRight = vecSrc[static_cast<size_t>(c * nSrcSpatial + nY1 * nSrcW + nX1)];  // 20260330 ZJH 右下

                    // 20260330 ZJH 双线性加权平均
                    // result = (1-fy)*((1-fx)*TL + fx*TR) + fy*((1-fx)*BL + fx*BR)
                    float fTop = (1.0f - fFracX) * fTopLeft + fFracX * fTopRight;          // 20260330 ZJH 上行插值
                    float fBottom = (1.0f - fFracX) * fBottomLeft + fFracX * fBottomRight;  // 20260330 ZJH 下行插值
                    float fResult = (1.0f - fFracY) * fTop + fFracY * fBottom;              // 20260330 ZJH 最终插值结果

                    vecDst[static_cast<size_t>(c * nDstSpatial + dy * nDstW + dx)] = fResult;
                }
            }
        }
    }
    return vecDst;
}

// 20260330 ZJH letterboxImage — Letterbox 等比缩放 + 灰色填充
// 保持原始宽高比将图像缩放到目标尺寸，不足部分用灰色(114/255≈0.447)填充
// 常用于 YOLO 系列目标检测模型的预处理
// vecSrc: 源图像数据 CHW 格式, nC: 通道数
// nSrcH/nSrcW: 源尺寸, nDstH/nDstW: 目标尺寸
// 返回: {缩放后图像数据, LetterboxInfo 坐标映射信息}
std::pair<std::vector<float>, LetterboxInfo> letterboxImage(
    const std::vector<float>& vecSrc,
    int nC, int nSrcH, int nSrcW,
    int nDstH, int nDstW) {

    // 20260330 ZJH 计算等比缩放因子：取宽高比中较小的那个，保证图像完整显示
    float fScaleW = static_cast<float>(nDstW) / static_cast<float>(nSrcW);  // 20260330 ZJH 宽度缩放比
    float fScaleH = static_cast<float>(nDstH) / static_cast<float>(nSrcH);  // 20260330 ZJH 高度缩放比
    float fScale = std::min(fScaleW, fScaleH);                               // 20260330 ZJH 取较小值保持宽高比

    // 20260330 ZJH 计算等比缩放后的实际尺寸（不含填充）
    int nNewW = static_cast<int>(std::round(static_cast<float>(nSrcW) * fScale));  // 20260330 ZJH 缩放后宽度
    int nNewH = static_cast<int>(std::round(static_cast<float>(nSrcH) * fScale));  // 20260330 ZJH 缩放后高度
    // 20260330 ZJH 安全钳制，防止超过目标尺寸
    nNewW = std::min(nNewW, nDstW);
    nNewH = std::min(nNewH, nDstH);

    // 20260330 ZJH 计算居中填充的偏移量（左/上填充量）
    int nPadLeft = (nDstW - nNewW) / 2;  // 20260330 ZJH 左侧填充像素数
    int nPadTop = (nDstH - nNewH) / 2;   // 20260330 ZJH 顶部填充像素数

    // 20260330 ZJH 先使用双线性插值将源图像缩放到 nNewH x nNewW
    std::vector<float> vecResized = resizeImage(vecSrc, nC, nSrcH, nSrcW, nNewH, nNewW, ResizeMode::Bilinear);

    // 20260330 ZJH 创建目标缓冲区，初始填充灰色（114/255 ≈ 0.447f）
    // 这是 YOLO 系列模型的标准 letterbox 填充色
    const float fPadValue = 114.0f / 255.0f;  // 20260330 ZJH 填充灰色值
    std::vector<float> vecDst(static_cast<size_t>(nC * nDstH * nDstW), fPadValue);

    // 20260330 ZJH 将缩放后的图像拷贝到目标缓冲区的居中位置
    for (int c = 0; c < nC; ++c) {
        for (int y = 0; y < nNewH; ++y) {
            for (int x = 0; x < nNewW; ++x) {
                // 20260330 ZJH 目标位置 = 填充偏移 + 缩放图像坐标
                int nDstY = y + nPadTop;   // 20260330 ZJH 目标行
                int nDstX = x + nPadLeft;  // 20260330 ZJH 目标列
                vecDst[static_cast<size_t>(c * nDstH * nDstW + nDstY * nDstW + nDstX)] =
                    vecResized[static_cast<size_t>(c * nNewH * nNewW + y * nNewW + x)];
            }
        }
    }

    // 20260330 ZJH 构建坐标映射信息，供推理后坐标反变换使用
    LetterboxInfo info;
    info.fScale = fScale;      // 20260330 ZJH 缩放因子
    info.nPadLeft = nPadLeft;  // 20260330 ZJH 左填充
    info.nPadTop = nPadTop;    // 20260330 ZJH 上填充
    info.nNewW = nNewW;        // 20260330 ZJH 实际图像宽度
    info.nNewH = nNewH;        // 20260330 ZJH 实际图像高度

    return {vecDst, info};
}

// =========================================================
// 20260330 ZJH 批量级数据增强（MixUp / CutMix / Mosaic）
// 这些增强在 batch 层面操作，混合多个样本，提升模型泛化能力
// =========================================================

// 20260330 ZJH mixupBatch — MixUp 批量增强
// 将两批样本按随机权重线性混合（图像和标签均混合）
// 论文: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
// vecBatchA/vecBatchB: 两批图像数据，每个元素为一张图的 CHW float 数据
// vecLabelsA/vecLabelsB: 两批软标签，每个元素为 one-hot 或概率分布向量
// fAlpha: 混合强度控制参数，lambda 在 [0.3, 0.7] 范围内均匀采样（简化 Beta 分布）
// 返回: {混合后图像批次, 混合后标签批次}
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
mixupBatch(const std::vector<std::vector<float>>& vecBatchA,
           const std::vector<std::vector<float>>& vecLabelsA,
           const std::vector<std::vector<float>>& vecBatchB,
           const std::vector<std::vector<float>>& vecLabelsB,
           float fAlpha = 0.2f) {

    static thread_local std::mt19937 gen(std::random_device{}());  // 20260330 ZJH 线程局部随机数生成器

    // 20260330 ZJH 取两批中较小的 batch size，确保一一对应
    size_t nBatchSize = std::min(vecBatchA.size(), vecBatchB.size());

    std::vector<std::vector<float>> vecMixedImages(nBatchSize);   // 20260330 ZJH 混合后图像
    std::vector<std::vector<float>> vecMixedLabels(nBatchSize);   // 20260330 ZJH 混合后标签

    // 20260330 ZJH 根据 fAlpha 控制混合范围: lambda ∈ [0.5-fAlpha, 0.5+fAlpha]
    // fAlpha 越大混合范围越宽（更多极端混合），越小则 lambda 越集中在 0.5 附近
    float fLambdaMin = std::max(0.0f, 0.5f - fAlpha);   // 20260330 ZJH 下界钳制
    float fLambdaMax = std::min(1.0f, 0.5f + fAlpha);   // 20260330 ZJH 上界钳制
    std::uniform_real_distribution<float> distLambda(fLambdaMin, fLambdaMax);

    for (size_t n = 0; n < nBatchSize; ++n) {
        float fLambda = distLambda(gen);  // 20260330 ZJH 当前样本的混合权重

        const auto& vecImgA = vecBatchA[n];  // 20260330 ZJH 样本 A 图像
        const auto& vecImgB = vecBatchB[n];  // 20260330 ZJH 样本 B 图像

        // 20260330 ZJH 混合图像: mixed = lambda * A + (1 - lambda) * B
        size_t nImgSize = std::min(vecImgA.size(), vecImgB.size());  // 20260330 ZJH 取较小尺寸防越界
        vecMixedImages[n].resize(nImgSize);
        for (size_t i = 0; i < nImgSize; ++i) {
            vecMixedImages[n][i] = fLambda * vecImgA[i] + (1.0f - fLambda) * vecImgB[i];
        }

        // 20260330 ZJH 混合标签: mixed_label = lambda * labelA + (1 - lambda) * labelB
        const auto& vecLblA = vecLabelsA[n];  // 20260330 ZJH 样本 A 标签
        const auto& vecLblB = vecLabelsB[n];  // 20260330 ZJH 样本 B 标签
        size_t nLabelSize = std::min(vecLblA.size(), vecLblB.size());  // 20260330 ZJH 标签维度
        vecMixedLabels[n].resize(nLabelSize);
        for (size_t i = 0; i < nLabelSize; ++i) {
            vecMixedLabels[n][i] = fLambda * vecLblA[i] + (1.0f - fLambda) * vecLblB[i];
        }
    }

    return {vecMixedImages, vecMixedLabels};
}

// 20260330 ZJH cutmixBatch — CutMix 批量增强
// 在图像 A 上挖一个随机矩形区域，用图像 B 的对应区域填充
// 标签按面积比例混合（矩形面积占总面积的比例）
// 论文: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
// vecBatchA/vecBatchB: 两批图像数据 CHW 格式
// vecLabelsA/vecLabelsB: 两批软标签
// fAlpha: 控制参数（面积比在 [0.2, 0.8] 范围内采样）
// nH/nW/nC: 图像的高度、宽度、通道数（同一 batch 内所有图像尺寸相同）
// 返回: {混合后图像批次, 混合后标签批次}
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
cutmixBatch(const std::vector<std::vector<float>>& vecBatchA,
            const std::vector<std::vector<float>>& vecLabelsA,
            const std::vector<std::vector<float>>& vecBatchB,
            const std::vector<std::vector<float>>& vecLabelsB,
            int nH, int nW, int nC,
            float fAlpha = 0.2f) {

    static thread_local std::mt19937 gen(std::random_device{}());  // 20260330 ZJH 线程局部随机数生成器
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    size_t nBatchSize = std::min(vecBatchA.size(), vecBatchB.size());  // 20260330 ZJH 实际批次大小

    std::vector<std::vector<float>> vecMixedImages(nBatchSize);  // 20260330 ZJH 输出图像
    std::vector<std::vector<float>> vecMixedLabels(nBatchSize);  // 20260330 ZJH 输出标签

    // 20260330 ZJH 根据 fAlpha 控制面积比采样范围: [0.5-fAlpha, 0.5+fAlpha]
    float fAreaMin = std::max(0.0f, 0.5f - fAlpha);   // 20260330 ZJH 最小面积比
    float fAreaMax = std::min(1.0f, 0.5f + fAlpha);   // 20260330 ZJH 最大面积比
    std::uniform_real_distribution<float> distArea(fAreaMin, fAreaMax);

    for (size_t n = 0; n < nBatchSize; ++n) {
        // 20260330 ZJH 根据 fAlpha 参数采样面积比例
        float fAreaRatio = distArea(gen);

        // 20260330 ZJH 随机生成矩形宽高比 [0.5, 2.0]
        float fAspectRatio = 0.5f + dist01(gen) * 1.5f;

        // 20260330 ZJH 根据面积比和宽高比计算矩形的宽和高
        float fCutArea = static_cast<float>(nH * nW) * fAreaRatio;        // 20260330 ZJH 矩形像素面积
        int nCutW = static_cast<int>(std::sqrt(fCutArea * fAspectRatio));  // 20260330 ZJH 矩形宽度
        int nCutH = static_cast<int>(std::sqrt(fCutArea / fAspectRatio));  // 20260330 ZJH 矩形高度
        // 20260330 ZJH 钳制到图像边界内
        nCutW = std::max(1, std::min(nCutW, nW));
        nCutH = std::max(1, std::min(nCutH, nH));

        // 20260330 ZJH 随机生成矩形左上角坐标
        int nX0 = static_cast<int>(dist01(gen) * static_cast<float>(nW - nCutW));  // 20260330 ZJH 左列
        int nY0 = static_cast<int>(dist01(gen) * static_cast<float>(nH - nCutH));  // 20260330 ZJH 上行
        // 20260330 ZJH 安全钳制
        nX0 = std::max(0, std::min(nX0, nW - nCutW));
        nY0 = std::max(0, std::min(nY0, nH - nCutH));

        // 20260330 ZJH 计算实际面积比例（用于标签混合权重）
        float fActualRatio = static_cast<float>(nCutW * nCutH) / static_cast<float>(nH * nW);

        // 20260330 ZJH 以图像 A 为基底，复制一份
        vecMixedImages[n] = vecBatchA[n];

        // 20260330 ZJH 将矩形区域用图像 B 的对应像素替换
        int nSpatial = nH * nW;  // 20260330 ZJH 每通道像素数
        for (int c = 0; c < nC; ++c) {
            for (int y = nY0; y < nY0 + nCutH; ++y) {
                for (int x = nX0; x < nX0 + nCutW; ++x) {
                    size_t nIdx = static_cast<size_t>(c * nSpatial + y * nW + x);
                    // 20260330 ZJH 边界检查后替换
                    if (nIdx < vecBatchB[n].size()) {
                        vecMixedImages[n][nIdx] = vecBatchB[n][nIdx];
                    }
                }
            }
        }

        // 20260330 ZJH 混合标签: label = (1 - areaRatio) * labelA + areaRatio * labelB
        const auto& vecLblA = vecLabelsA[n];
        const auto& vecLblB = vecLabelsB[n];
        size_t nLabelSize = std::min(vecLblA.size(), vecLblB.size());
        vecMixedLabels[n].resize(nLabelSize);
        for (size_t i = 0; i < nLabelSize; ++i) {
            vecMixedLabels[n][i] = (1.0f - fActualRatio) * vecLblA[i] + fActualRatio * vecLblB[i];
        }
    }

    return {vecMixedImages, vecMixedLabels};
}

// 20260330 ZJH mosaicAugment — Mosaic 4图拼接增强
// 将4张图像拼接成一张，随机选择中心分割点
// 常用于 YOLOv4/v5 目标检测训练，大幅提升小目标检测能力
// 论文: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection", 2020
// vec4Images: 4张图像的 CHW float 数据
// vec4Labels: 4张图对应的标签（分类任务: one-hot; 检测任务: bbox 列表拼接）
// nTargetH/nTargetW: 输出图像尺寸
// nChannels: 通道数
// 返回: {拼接后图像数据 CHW, 合并后标签（4份标签拼接）}
std::pair<std::vector<float>, std::vector<float>>
mosaicAugment(const std::vector<std::vector<float>>& vec4Images,
              const std::vector<std::vector<float>>& vec4Labels,
              int nTargetH, int nTargetW, int nChannels) {

    static thread_local std::mt19937 gen(std::random_device{}());  // 20260330 ZJH 线程局部随机数生成器
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // 20260330 ZJH 输入校验：需要恰好 4 张图像
    if (vec4Images.size() < 4 || vec4Labels.size() < 4) {
        // 20260330 ZJH 不足 4 张时返回空
        return {{}, {}};
    }

    // 20260330 ZJH 随机中心分割点（在图像中心 25%~75% 范围内）
    // 避免分割点过于靠边导致某个象限几乎不可见
    int nCenterX = static_cast<int>((0.25f + dist01(gen) * 0.5f) * static_cast<float>(nTargetW));  // 20260330 ZJH X 分割坐标
    int nCenterY = static_cast<int>((0.25f + dist01(gen) * 0.5f) * static_cast<float>(nTargetH));  // 20260330 ZJH Y 分割坐标

    // 20260330 ZJH 四个象限的尺寸
    // 象限 0: 左上 [0, 0] ~ [nCenterY, nCenterX]
    // 象限 1: 右上 [0, nCenterX] ~ [nCenterY, nTargetW]
    // 象限 2: 左下 [nCenterY, 0] ~ [nTargetH, nCenterX]
    // 象限 3: 右下 [nCenterY, nCenterX] ~ [nTargetH, nTargetW]
    int arrQuadW[4] = {nCenterX, nTargetW - nCenterX, nCenterX, nTargetW - nCenterX};        // 20260330 ZJH 各象限宽度
    int arrQuadH[4] = {nCenterY, nCenterY, nTargetH - nCenterY, nTargetH - nCenterY};        // 20260330 ZJH 各象限高度
    int arrQuadOffX[4] = {0, nCenterX, 0, nCenterX};                                          // 20260330 ZJH 各象限 X 偏移
    int arrQuadOffY[4] = {0, 0, nCenterY, nCenterY};                                          // 20260330 ZJH 各象限 Y 偏移

    // 20260330 ZJH 创建输出缓冲区，初始填充灰色
    const float fPadValue = 114.0f / 255.0f;  // 20260330 ZJH 灰色填充值（与 letterbox 一致）
    std::vector<float> vecOutput(static_cast<size_t>(nChannels * nTargetH * nTargetW), fPadValue);

    // 20260330 ZJH 合并后的标签（将 4 份标签拼接）
    std::vector<float> vecMergedLabels;

    for (int q = 0; q < 4; ++q) {
        int nQW = arrQuadW[q];    // 20260330 ZJH 当前象限宽度
        int nQH = arrQuadH[q];    // 20260330 ZJH 当前象限高度
        int nQOffX = arrQuadOffX[q];  // 20260330 ZJH 当前象限 X 偏移
        int nQOffY = arrQuadOffY[q];  // 20260330 ZJH 当前象限 Y 偏移

        // 20260330 ZJH 跳过尺寸为零的象限
        if (nQW <= 0 || nQH <= 0) continue;

        const auto& vecSrcImg = vec4Images[static_cast<size_t>(q)];  // 20260330 ZJH 当前源图像

        // 20260330 ZJH 推算源图像的尺寸（从数据长度反推，假设是正方形或已知通道数）
        int nSrcPixels = static_cast<int>(vecSrcImg.size()) / nChannels;  // 20260330 ZJH 源图像总像素数
        int nSrcH = static_cast<int>(std::sqrt(static_cast<float>(nSrcPixels)));  // 20260330 ZJH 估计源高度
        int nSrcW = (nSrcH > 0) ? (nSrcPixels / nSrcH) : 0;                      // 20260330 ZJH 估计源宽度

        // 20260330 ZJH 安全检查
        if (nSrcH <= 0 || nSrcW <= 0 || vecSrcImg.size() < static_cast<size_t>(nChannels * nSrcH * nSrcW)) {
            continue;
        }

        // 20260330 ZJH 将源图像缩放到当前象限尺寸（双线性插值）
        std::vector<float> vecResized = resizeImage(vecSrcImg, nChannels, nSrcH, nSrcW, nQH, nQW, ResizeMode::Bilinear);

        // 20260330 ZJH 将缩放后图像拷贝到输出缓冲区的对应象限位置
        for (int c = 0; c < nChannels; ++c) {
            for (int y = 0; y < nQH; ++y) {
                for (int x = 0; x < nQW; ++x) {
                    int nDstY = nQOffY + y;  // 20260330 ZJH 输出缓冲区行
                    int nDstX = nQOffX + x;  // 20260330 ZJH 输出缓冲区列
                    vecOutput[static_cast<size_t>(c * nTargetH * nTargetW + nDstY * nTargetW + nDstX)] =
                        vecResized[static_cast<size_t>(c * nQH * nQW + y * nQW + x)];
                }
            }
        }

        // 20260330 ZJH 拼接标签（检测任务中，将各图的 bbox 列表直接合并）
        const auto& vecLbl = vec4Labels[static_cast<size_t>(q)];
        vecMergedLabels.insert(vecMergedLabels.end(), vecLbl.begin(), vecLbl.end());
    }

    return {vecOutput, vecMergedLabels};
}

// =========================================================
// 20260406 ZJH Dataset 基类和具体实现
// =========================================================

// 20260320 ZJH Dataset — 数据集抽象基类
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;                           // 20260320 ZJH 数据集大小
    virtual std::pair<Tensor, Tensor> getItem(size_t nIdx) = 0;  // 20260320 ZJH 获取 (input, target)
};

// 20260320 ZJH ImageClassificationDataset — 图像分类数据集
// 从文件夹结构加载：root/class_name/*.bmp|*.png|*.jpg
// 每个子文件夹名就是类别名，类别 ID 按字母排序
class ImageClassificationDataset : public Dataset {
public:
    // 20260320 ZJH 构造函数
    // strRootPath: 数据集根目录
    // nTargetH/nTargetW: 目标图像尺寸
    // nTargetC: 目标通道数（1=灰度, 3=RGB）
    // config: 数据增强配置
    ImageClassificationDataset(const std::string& strRootPath,
                                int nTargetH, int nTargetW, int nTargetC = 3,
                                AugmentConfig config = {})
        : m_strRootPath(strRootPath), m_nTargetH(nTargetH), m_nTargetW(nTargetW),
          m_nTargetC(nTargetC), m_config(config)
    {
        scanDirectory();  // 20260320 ZJH 扫描目录结构
    }

    size_t size() const override { return m_vecSamples.size(); }

    // 20260320 ZJH 获取第 nIdx 个样本
    // 返回: (image_tensor [C, H, W], label_tensor [numClasses] one-hot)
    std::pair<Tensor, Tensor> getItem(size_t nIdx) override {
        const auto& sample = m_vecSamples[nIdx];

        // 20260320 ZJH 加载图像
        RawImage img = loadBMP(sample.strPath);
        std::vector<float> vecData;
        int nC = 0, nH = 0, nW = 0;

        if (img.isValid()) {
            nC = img.nChannels;
            nH = img.nHeight;
            nW = img.nWidth;
            vecData = img.vecData;
        } else {
            // 20260320 ZJH 加载失败：返回黑色图像
            nC = m_nTargetC;
            nH = m_nTargetH;
            nW = m_nTargetW;
            vecData.resize(static_cast<size_t>(nC * nH * nW), 0.0f);
        }

        // 20260320 ZJH 缩放到目标尺寸
        if (nH != m_nTargetH || nW != m_nTargetW) {
            vecData = resizeImage(vecData, nC, nH, nW, m_nTargetH, m_nTargetW);
            nH = m_nTargetH;
            nW = m_nTargetW;
        }

        // 20260320 ZJH 通道数转换（RGB -> 灰度 或 灰度 -> RGB）
        if (nC == 3 && m_nTargetC == 1) {
            // 20260320 ZJH RGB 转灰度：0.299*R + 0.587*G + 0.114*B
            std::vector<float> vecGray(static_cast<size_t>(nH * nW));
            int nSpatial = nH * nW;
            for (int i = 0; i < nSpatial; ++i) {
                vecGray[static_cast<size_t>(i)] =
                    0.299f * vecData[static_cast<size_t>(i)] +
                    0.587f * vecData[static_cast<size_t>(nSpatial + i)] +
                    0.114f * vecData[static_cast<size_t>(2 * nSpatial + i)];
            }
            vecData = vecGray;
            nC = 1;
        } else if (nC == 1 && m_nTargetC == 3) {
            // 20260320 ZJH 灰度转 RGB：三通道复制
            int nSpatial = nH * nW;
            std::vector<float> vecRGB(static_cast<size_t>(3 * nSpatial));
            for (int i = 0; i < nSpatial; ++i) {
                vecRGB[static_cast<size_t>(i)] = vecData[static_cast<size_t>(i)];
                vecRGB[static_cast<size_t>(nSpatial + i)] = vecData[static_cast<size_t>(i)];
                vecRGB[static_cast<size_t>(2 * nSpatial + i)] = vecData[static_cast<size_t>(i)];
            }
            vecData = vecRGB;
            nC = 3;
        }

        // 20260320 ZJH 应用数据增强
        augmentImage(vecData, nC, nH, nW, m_config);

        // 20260320 ZJH 创建图像张量 [C, H, W]
        auto imgTensor = Tensor::fromData(vecData.data(), {nC, nH, nW});

        // 20260320 ZJH 创建 one-hot 标签张量 [numClasses]
        auto labelTensor = Tensor::zeros({static_cast<int>(m_vecClassNames.size())});
        labelTensor.mutableFloatDataPtr()[sample.nClassId] = 1.0f;

        return {imgTensor, labelTensor};
    }

    // 20260320 ZJH 获取类别数量
    int numClasses() const { return static_cast<int>(m_vecClassNames.size()); }

    // 20260320 ZJH 获取类别名列表
    const std::vector<std::string>& classNames() const { return m_vecClassNames; }

private:
    struct Sample {
        std::string strPath;  // 20260320 ZJH 文件路径
        int nClassId;         // 20260320 ZJH 类别 ID
    };

    // 20260320 ZJH 扫描目录结构，建立样本列表
    void scanDirectory() {
        namespace fs = std::filesystem;
        if (!fs::exists(m_strRootPath)) return;

        // 20260320 ZJH 收集子目录作为类别
        std::vector<std::string> vecDirs;
        for (const auto& entry : fs::directory_iterator(m_strRootPath)) {
            if (entry.is_directory()) {
                vecDirs.push_back(entry.path().filename().string());
            }
        }
        std::sort(vecDirs.begin(), vecDirs.end());  // 20260320 ZJH 按字母排序
        m_vecClassNames = vecDirs;

        // 20260320 ZJH 扫描每个类别目录下的图像文件
        for (int classId = 0; classId < static_cast<int>(vecDirs.size()); ++classId) {
            std::string strClassDir = m_strRootPath + "/" + vecDirs[static_cast<size_t>(classId)];
            for (const auto& entry : fs::directory_iterator(strClassDir)) {
                if (!entry.is_regular_file()) continue;
                std::string strExt = entry.path().extension().string();
                // 20260320 ZJH 支持的图像格式
                if (strExt == ".bmp" || strExt == ".BMP" ||
                    strExt == ".png" || strExt == ".PNG" ||
                    strExt == ".jpg" || strExt == ".JPG" ||
                    strExt == ".jpeg" || strExt == ".JPEG") {
                    m_vecSamples.push_back({entry.path().string(), classId});
                }
            }
        }
    }

    std::string m_strRootPath;              // 20260320 ZJH 数据集根目录
    int m_nTargetH;                         // 20260320 ZJH 目标高度
    int m_nTargetW;                         // 20260320 ZJH 目标宽度
    int m_nTargetC;                         // 20260320 ZJH 目标通道数
    AugmentConfig m_config;                 // 20260320 ZJH 数据增强配置
    std::vector<Sample> m_vecSamples;       // 20260320 ZJH 样本列表
    std::vector<std::string> m_vecClassNames;  // 20260320 ZJH 类别名列表
};

// =========================================================
// 20260406 ZJH DataLoader — 数据加载器
// =========================================================

// 20260320 ZJH DataLoader — 批量数据加载器
// 支持 shuffle、批量加载，返回 (batchInput, batchTarget)
class DataLoader {
public:
    // 20260320 ZJH 构造函数
    // pDataset: 数据集指针（不拥有所有权）
    // nBatchSize: 批量大小
    // bShuffle: 是否随机打乱
    DataLoader(Dataset* pDataset, int nBatchSize, bool bShuffle = true)
        : m_pDataset(pDataset), m_nBatchSize(nBatchSize), m_bShuffle(bShuffle)
    {
        m_nNumSamples = static_cast<int>(pDataset->size());
        m_nNumBatches = (m_nNumSamples + nBatchSize - 1) / nBatchSize;  // 20260320 ZJH 向上取整

        // 20260320 ZJH 初始化索引序列
        m_vecIndices.resize(static_cast<size_t>(m_nNumSamples));
        for (int i = 0; i < m_nNumSamples; ++i)
            m_vecIndices[static_cast<size_t>(i)] = i;
    }

    // 20260320 ZJH reset — 重置到第一个 batch，可选 shuffle
    void reset() {
        m_nCurrentBatch = 0;
        if (m_bShuffle) {
            static thread_local std::mt19937 gen(std::random_device{}());
            std::shuffle(m_vecIndices.begin(), m_vecIndices.end(), gen);
        }
    }

    // 20260320 ZJH hasNext — 是否还有剩余 batch
    bool hasNext() const { return m_nCurrentBatch < m_nNumBatches; }

    // 20260320 ZJH next — 获取下一个 batch
    // 返回: (batchInput [N, ...], batchTarget [N, ...])
    std::pair<Tensor, Tensor> next() {
        int nStart = m_nCurrentBatch * m_nBatchSize;
        int nEnd = std::min(nStart + m_nBatchSize, m_nNumSamples);
        int nActualBatch = nEnd - nStart;

        // 20260320 ZJH 获取第一个样本确定形状
        auto [firstInput, firstTarget] = m_pDataset->getItem(
            static_cast<size_t>(m_vecIndices[static_cast<size_t>(nStart)]));

        auto vecInputShape = firstInput.shapeVec();
        auto vecTargetShape = firstTarget.shapeVec();

        // 20260320 ZJH 构建 batch 形状
        std::vector<int> vecBatchInputShape = {nActualBatch};
        vecBatchInputShape.insert(vecBatchInputShape.end(), vecInputShape.begin(), vecInputShape.end());
        std::vector<int> vecBatchTargetShape = {nActualBatch};
        vecBatchTargetShape.insert(vecBatchTargetShape.end(), vecTargetShape.begin(), vecTargetShape.end());

        auto batchInput = Tensor::zeros(vecBatchInputShape);
        auto batchTarget = Tensor::zeros(vecBatchTargetShape);

        int nInputSize = firstInput.numel();   // 20260320 ZJH 每个输入的元素数
        int nTargetSize = firstTarget.numel(); // 20260320 ZJH 每个目标的元素数

        // 20260320 ZJH 填入第一个样本
        {
            float* pInput = batchInput.mutableFloatDataPtr();
            float* pTarget = batchTarget.mutableFloatDataPtr();
            const float* pFI = firstInput.floatDataPtr();
            const float* pFT = firstTarget.floatDataPtr();
            for (int j = 0; j < nInputSize; ++j) pInput[j] = pFI[j];
            for (int j = 0; j < nTargetSize; ++j) pTarget[j] = pFT[j];
        }

        // 20260320 ZJH 填入后续样本
        for (int i = 1; i < nActualBatch; ++i) {
            int nIdx = m_vecIndices[static_cast<size_t>(nStart + i)];
            auto [inp, tgt] = m_pDataset->getItem(static_cast<size_t>(nIdx));
            auto ci = inp.contiguous();
            auto ct = tgt.contiguous();
            float* pInput = batchInput.mutableFloatDataPtr() + i * nInputSize;
            float* pTarget = batchTarget.mutableFloatDataPtr() + i * nTargetSize;
            const float* pI = ci.floatDataPtr();
            const float* pT = ct.floatDataPtr();
            for (int j = 0; j < nInputSize; ++j) pInput[j] = pI[j];
            for (int j = 0; j < nTargetSize; ++j) pTarget[j] = pT[j];
        }

        m_nCurrentBatch++;
        return {batchInput, batchTarget};
    }

    // 20260320 ZJH 获取 batch 总数
    int numBatches() const { return m_nNumBatches; }

    // 20260320 ZJH 获取数据集大小
    int numSamples() const { return m_nNumSamples; }

private:
    Dataset* m_pDataset;               // 20260320 ZJH 数据集指针
    int m_nBatchSize;                  // 20260320 ZJH 批量大小
    bool m_bShuffle;                   // 20260320 ZJH 是否随机打乱
    int m_nNumSamples;                 // 20260320 ZJH 样本总数
    int m_nNumBatches;                 // 20260320 ZJH batch 总数
    int m_nCurrentBatch = 0;           // 20260320 ZJH 当前 batch 索引
    std::vector<int> m_vecIndices;     // 20260320 ZJH 打乱后的索引序列
};

// 20260320 ZJH splitDataset — 按比例划分训练/验证/测试索引
// 返回: {trainIndices, valIndices, testIndices}
struct DatasetSplit {
    std::vector<size_t> vecTrainIndices;
    std::vector<size_t> vecValIndices;
    std::vector<size_t> vecTestIndices;
};

DatasetSplit splitDataset(size_t nTotalSize, float fTrainRatio = 0.8f,
                           float fValRatio = 0.1f, unsigned int nSeed = 42) {
    DatasetSplit split;
    std::vector<size_t> vecIndices(nTotalSize);
    for (size_t i = 0; i < nTotalSize; ++i) vecIndices[i] = i;

    // 20260320 ZJH 使用固定种子打乱（可复现）
    std::mt19937 gen(nSeed);
    std::shuffle(vecIndices.begin(), vecIndices.end(), gen);

    size_t nTrain = static_cast<size_t>(nTotalSize * fTrainRatio);
    size_t nVal = static_cast<size_t>(nTotalSize * fValRatio);

    split.vecTrainIndices.assign(vecIndices.begin(), vecIndices.begin() + nTrain);
    split.vecValIndices.assign(vecIndices.begin() + nTrain, vecIndices.begin() + nTrain + nVal);
    split.vecTestIndices.assign(vecIndices.begin() + nTrain + nVal, vecIndices.end());

    return split;
}

// =========================================================
// 20260330 ZJH 智能参数自动调优 — 对标海康 VisionTrain IntelligentParam
// 根据数据集统计信息自动推荐训练超参数
// =========================================================

// 20260330 ZJH DatasetStats — 数据集统计信息结构
// 在训练前通过扫描数据集得出，作为自动调参的输入
struct DatasetStats {
    int nTotalImages = 0;              // 20260330 ZJH 数据集总图像数
    int nNumClasses = 0;               // 20260330 ZJH 类别总数
    int nMinSamplesPerClass = 0;       // 20260330 ZJH 最少类别的样本数
    int nMaxSamplesPerClass = 0;       // 20260330 ZJH 最多类别的样本数
    float fClassImbalanceRatio = 1.0f; // 20260330 ZJH 类别不平衡比（max/min），1.0 表示完全平衡
    int nMeanImageWidth = 0;           // 20260330 ZJH 图像平均宽度（像素）
    int nMeanImageHeight = 0;          // 20260330 ZJH 图像平均高度（像素）
};

// 20260330 ZJH RecommendedParams — 推荐的训练超参数结构
// 由 autoTuneParams() 根据数据集特征计算得出
struct RecommendedParams {
    float fLearningRate = 1e-3f;       // 20260330 ZJH 推荐学习率
    int nBatchSize = 16;               // 20260330 ZJH 推荐批量大小
    int nEpochs = 200;                 // 20260330 ZJH 推荐训练轮数
    int nInputSize = 224;              // 20260330 ZJH 推荐输入图像尺寸（正方形边长）
    bool bUseAugmentation = true;      // 20260330 ZJH 是否推荐启用数据增强
    float fAugProbability = 0.5f;      // 20260330 ZJH 推荐数据增强概率
    std::string strOptimizer = "Adam"; // 20260330 ZJH 推荐优化器名称
    std::string strScheduler = "CosineAnnealing";  // 20260330 ZJH 推荐学习率调度器
    std::string strReason;             // 20260330 ZJH 推荐理由（人可读的说明文本）
};

// 20260330 ZJH autoTuneParams — 智能参数自动调优
// 根据数据集统计信息和任务类型，自动推荐训练超参数
// 逻辑对标海康 VisionTrain 的智能参数策略:
//   - 小数据集 (<50): 高 epoch、强增强、低学习率，防止过拟合
//   - 中数据集 (50-250): 适中 epoch、中等增强
//   - 大数据集 (>250): 低 epoch、标准增强，数据本身已足够多样化
//   - 类别不平衡: 推荐 Focal Loss + 过采样
//   - 大分辨率图像: 推荐 patch 模式或 SOD（小目标检测）
// stats: 数据集统计信息
// strTaskType: 任务类型字符串 ("classification"/"detection"/"segmentation"/"anomaly"/"instance_seg"/"ocr")
// 返回: 推荐参数结构
RecommendedParams autoTuneParams(const DatasetStats& stats, const std::string& strTaskType) {
    RecommendedParams params;  // 20260330 ZJH 初始化为默认值
    std::string strReason;      // 20260330 ZJH 累积推荐理由

    // ====== 20260330 ZJH 根据数据集规模调整 epoch 和增强强度 ======

    if (stats.nTotalImages < 50) {
        // 20260330 ZJH 极小数据集策略（<50 张）
        // 数据极少，必须大幅延长训练并配合强增强来防止过拟合
        params.nEpochs = 800;               // 20260330 ZJH 高 epoch 数充分利用有限数据
        params.fLearningRate = 5e-4f;       // 20260330 ZJH 低学习率避免震荡，小数据集梯度噪声大
        params.bUseAugmentation = true;     // 20260330 ZJH 强制开启增强
        params.fAugProbability = 0.8f;      // 20260330 ZJH 高增强概率（80%样本被增强）
        params.nBatchSize = 4;              // 20260330 ZJH 小批量防止梯度估计过于平滑
        strReason += "Small dataset (<50 images): high epochs=800, strong augmentation(0.8), low LR=5e-4, batch=4. ";
    } else if (stats.nTotalImages < 250) {
        // 20260330 ZJH 中等数据集策略（50-250 张）
        params.nEpochs = 200;               // 20260330 ZJH 中等 epoch 数
        params.fLearningRate = 1e-3f;       // 20260330 ZJH 标准学习率
        params.bUseAugmentation = true;     // 20260330 ZJH 开启增强
        params.fAugProbability = 0.5f;      // 20260330 ZJH 中等增强概率
        params.nBatchSize = 8;              // 20260330 ZJH 中等批量
        strReason += "Medium dataset (50-250 images): epochs=200, moderate augmentation(0.5), LR=1e-3, batch=8. ";
    } else if (stats.nTotalImages < 1000) {
        // 20260330 ZJH 较大数据集策略（250-1000 张）
        params.nEpochs = 100;               // 20260330 ZJH 较少 epoch，数据量已足够
        params.fLearningRate = 1e-3f;       // 20260330 ZJH 标准学习率
        params.bUseAugmentation = true;     // 20260330 ZJH 标准增强
        params.fAugProbability = 0.3f;      // 20260330 ZJH 低增强概率
        params.nBatchSize = 16;             // 20260330 ZJH 标准批量
        strReason += "Standard dataset (250-1000 images): epochs=100, standard augmentation(0.3), LR=1e-3, batch=16. ";
    } else {
        // 20260330 ZJH 大数据集策略（>1000 张）
        params.nEpochs = 50;                // 20260330 ZJH 少量 epoch，数据多样性已充足
        params.fLearningRate = 2e-3f;       // 20260330 ZJH 稍高学习率加速收敛
        params.bUseAugmentation = true;     // 20260330 ZJH 轻度增强即可
        params.fAugProbability = 0.2f;      // 20260330 ZJH 低增强概率
        params.nBatchSize = 32;             // 20260330 ZJH 大批量提升GPU利用率
        strReason += "Large dataset (>1000 images): epochs=50, light augmentation(0.2), LR=2e-3, batch=32. ";
    }

    // ====== 20260330 ZJH 根据类别不平衡调整策略 ======

    if (stats.fClassImbalanceRatio > 5.0f) {
        // 20260330 ZJH 严重不平衡（最大类/最小类 > 5x）
        // 推荐 Focal Loss + 过采样，防止模型偏向多数类
        params.strOptimizer = "AdamW";  // 20260330 ZJH AdamW 权重衰减更适合不平衡场景
        strReason += "Severe class imbalance (ratio=" + std::to_string(static_cast<int>(stats.fClassImbalanceRatio))
                   + "x): recommend FocalLoss + oversampling, using AdamW optimizer. ";
    } else if (stats.fClassImbalanceRatio > 2.0f) {
        // 20260330 ZJH 中度不平衡（2x-5x）
        strReason += "Moderate class imbalance (ratio=" + std::to_string(static_cast<int>(stats.fClassImbalanceRatio))
                   + "x): consider weighted loss or oversampling. ";
    }

    // ====== 20260330 ZJH 根据图像分辨率调整输入尺寸和策略 ======

    int nMeanDim = (stats.nMeanImageWidth + stats.nMeanImageHeight) / 2;  // 20260330 ZJH 图像平均边长

    if (nMeanDim <= 64) {
        // 20260330 ZJH 极小图像（如 CIFAR 级别）
        params.nInputSize = 64;
        strReason += "Very small images (mean ~" + std::to_string(nMeanDim) + "px): inputSize=64. ";
    } else if (nMeanDim <= 256) {
        // 20260330 ZJH 中小图像
        params.nInputSize = 224;  // 20260330 ZJH 标准 ImageNet 尺寸
        strReason += "Standard images (mean ~" + std::to_string(nMeanDim) + "px): inputSize=224. ";
    } else if (nMeanDim <= 640) {
        // 20260330 ZJH 中大图像
        params.nInputSize = 448;
        strReason += "Medium-large images (mean ~" + std::to_string(nMeanDim) + "px): inputSize=448. ";
    } else if (nMeanDim <= 1280) {
        // 20260330 ZJH 大图像
        params.nInputSize = 640;
        strReason += "Large images (mean ~" + std::to_string(nMeanDim) + "px): inputSize=640, consider patch-based training. ";
    } else {
        // 20260330 ZJH 超大图像（工业高分辨率相机）
        params.nInputSize = 640;
        params.nBatchSize = std::max(1, params.nBatchSize / 2);  // 20260330 ZJH 减半批量节省显存
        strReason += "Very large images (mean ~" + std::to_string(nMeanDim)
                   + "px): inputSize=640, halved batchSize=" + std::to_string(params.nBatchSize)
                   + ", strongly recommend patch mode or SOD (Small Object Detection). ";
    }

    // ====== 20260330 ZJH 根据任务类型微调参数 ======

    if (strTaskType == "detection") {
        // 20260330 ZJH 目标检测任务：通常需要更大输入尺寸和更长训练
        if (params.nInputSize < 416) params.nInputSize = 416;  // 20260330 ZJH 检测模型最小 416
        params.nEpochs = static_cast<int>(params.nEpochs * 1.5f);  // 20260330 ZJH 检测任务收敛较慢
        params.strScheduler = "CosineAnnealing";
        strReason += "Detection task: inputSize>=416, epochs*1.5, CosineAnnealing scheduler. ";
    } else if (strTaskType == "segmentation" || strTaskType == "instance_seg") {
        // 20260330 ZJH 分割任务：需要大输入保留细节
        if (params.nInputSize < 256) params.nInputSize = 256;  // 20260330 ZJH 分割至少 256
        params.fLearningRate *= 0.5f;  // 20260330 ZJH 分割任务用略低学习率更稳定
        params.strScheduler = "PolyLR";  // 20260330 ZJH 多项式衰减更适合分割
        strReason += "Segmentation task: inputSize>=256, LR*0.5=" + std::to_string(params.fLearningRate)
                   + ", PolyLR scheduler. ";
    } else if (strTaskType == "anomaly") {
        // 20260330 ZJH 异常检测任务：通常只有正常样本，需强增强模拟变异
        params.bUseAugmentation = true;
        params.fAugProbability = std::max(params.fAugProbability, 0.6f);  // 20260330 ZJH 至少 60% 增强
        params.nEpochs = std::max(params.nEpochs, 300);  // 20260330 ZJH 异常检测需充分学习正常分布
        strReason += "Anomaly detection: strong augmentation(>0.6), epochs>=300. ";
    } else if (strTaskType == "ocr") {
        // 20260330 ZJH OCR 任务：通常需要较高分辨率输入
        if (params.nInputSize < 320) params.nInputSize = 320;
        strReason += "OCR task: inputSize>=320. ";
    } else {
        // 20260330 ZJH 分类任务（默认）
        params.strScheduler = "CosineAnnealing";
        strReason += "Classification task: CosineAnnealing scheduler. ";
    }

    // ====== 20260330 ZJH 优化器选择 ======

    // 20260330 ZJH 对于极小数据集，SGD+Momentum 比 Adam 更不容易过拟合
    if (stats.nTotalImages < 50 && params.strOptimizer == "Adam") {
        params.strOptimizer = "SGD";
        params.fLearningRate *= 10.0f;  // 20260330 ZJH SGD 通常需要更大学习率
        strReason += "Switched to SGD for very small dataset (better generalization). ";
    }

    params.strReason = strReason;  // 20260330 ZJH 保存推荐理由
    return params;
}

// =========================================================
// 20260402 ZJH AsyncDataLoader — 多线程异步数据预取管线
// 对标 PyTorch DataLoader(num_workers=4, prefetch_factor=2)
// 架构: N 个 worker 线程并行执行 加载→增强→归一化 → 写入环形缓冲区
// 训练线程从缓冲区取已就绪的 batch，实现 CPU 数据准备与 GPU 训练完全重叠
// 预期效果: 消除数据加载瓶颈，训练速度提升 1.5-3x
// =========================================================

class AsyncDataLoader {
public:
    // 20260402 ZJH 构造函数 — 初始化 worker 数量和预取缓冲区大小
    // nNumWorkers: 工作线程数，推荐等于 CPU 物理核数的一半（默认 4）
    // nPrefetchCount: 预取 batch 数，控制环形缓冲区深度（默认 2）
    AsyncDataLoader(int nNumWorkers = 4, int nPrefetchCount = 2)
        : m_nNumWorkers(nNumWorkers)   // 20260402 ZJH 保存 worker 线程数
        , m_nPrefetchCount(nPrefetchCount)  // 20260402 ZJH 保存预取深度
    {
        // 20260402 ZJH 参数下限保护: 至少 1 个 worker，至少 1 个预取缓冲
        if (m_nNumWorkers < 1) m_nNumWorkers = 1;
        if (m_nPrefetchCount < 1) m_nPrefetchCount = 1;
    }

    // 20260402 ZJH 析构函数 — 确保所有 worker 线程安全终止
    ~AsyncDataLoader() {
        stop();  // 20260402 ZJH 复用 stop() 逻辑，避免资源泄漏
    }

    // 20260402 ZJH configure — 配置数据源和训练参数
    // vecData: 全部训练数据，连续存储 [N * inputDim] float，CHW 格式
    // vecLabels: 全部标签 [N] int，每个样本一个整数类别标签
    // nInputDim: 单个样本的维度（C * H * W），用于计算数据偏移
    // nNumClasses: 总类别数，用于生成 one-hot 标签向量
    // nBatchSize: 每个 batch 的样本数
    void configure(const std::vector<float>& vecData,
                   const std::vector<int>& vecLabels,
                   int nInputDim, int nNumClasses, int nBatchSize)
    {
        // 20260402 ZJH 保存数据源指针（引用外部数据，避免拷贝）
        m_pData = &vecData;
        m_pLabels = &vecLabels;
        // 20260402 ZJH 保存维度参数
        m_nInputDim = nInputDim;
        m_nNumClasses = nNumClasses;
        m_nBatchSize = nBatchSize;

        // 20260402 ZJH 计算样本总数（从标签数组推导，因为每个样本对应一个标签）
        int nTotalSamples = static_cast<int>(vecLabels.size());

        // 20260402 ZJH 计算总 batch 数（向上取整，最后一个 batch 可能不满）
        m_nTotalBatches = (nTotalSamples + nBatchSize - 1) / nBatchSize;

        // 20260402 ZJH 初始化索引数组 [0, 1, 2, ..., N-1]，后续 startEpoch 中洗牌
        m_vecIndices.resize(static_cast<size_t>(nTotalSamples));
        for (int i = 0; i < nTotalSamples; ++i) {
            m_vecIndices[static_cast<size_t>(i)] = i;
        }

        // 20260402 ZJH 分配环形缓冲区
        // 缓冲区大小 = worker 数 + 预取数，保证足够的并发写入槽位
        m_nBufferSize = m_nNumWorkers + m_nPrefetchCount;
        m_vecRingBuffer.resize(static_cast<size_t>(m_nBufferSize));

        // 20260402 ZJH 为每个缓冲槽位预分配内存，避免运行时反复 alloc
        for (auto& buf : m_vecRingBuffer) {
            buf.vecInput.resize(static_cast<size_t>(nBatchSize) * nInputDim);  // 20260402 ZJH 输入: [batchSize * inputDim]
            buf.vecLabel.resize(static_cast<size_t>(nBatchSize) * nNumClasses);  // 20260402 ZJH 标签: [batchSize * numClasses] one-hot
            buf.nActualSize = 0;  // 20260402 ZJH 初始为空
            buf.bReady = false;   // 20260402 ZJH 初始未就绪
        }
    }

    // 20260402 ZJH setAugmentConfig — 设置数据增强配置
    // cfg: 增强配置结构体，包含几何变换、颜色抖动、噪声等参数
    void setAugmentConfig(const AugmentConfig& cfg) {
        m_augCfg = cfg;       // 20260402 ZJH 保存增强配置（值拷贝）
        m_bAugment = true;    // 20260402 ZJH 标记启用增强
    }

    // 20260402 ZJH setNormPreset — 设置归一化预设模式
    // ePreset: 归一化预设枚举（None/ZeroOne/MeanStd/ImageNet）
    void setNormPreset(NormPreset ePreset) {
        m_eNorm = ePreset;  // 20260402 ZJH 保存预设模式
        // 20260402 ZJH 当预设不是 None 或 ZeroOne 时，标记需要归一化
        m_bNormalize = (ePreset != NormPreset::None && ePreset != NormPreset::ZeroOne);
    }

    // 20260402 ZJH startEpoch — 开始新的训练 epoch
    // 1. 停止上一个 epoch 的残留 worker 线程
    // 2. Fisher-Yates 洗牌样本索引（保证均匀随机排列）
    // 3. 重置所有计数器和缓冲区状态
    // 4. 启动新一批 worker 线程
    // nSeed: 随机种子（可用 epoch 编号保证可复现性）
    void startEpoch(unsigned int nSeed) {
        // 20260402 ZJH 第一步: 停止旧 worker（如果有的话）
        stopWorkers();

        // 20260402 ZJH 第二步: Fisher-Yates 洗牌索引
        // Fisher-Yates 算法保证 O(N) 时间生成均匀随机排列
        {
            std::mt19937 gen(nSeed);  // 20260402 ZJH 使用固定种子的梅森旋转引擎
            int nN = static_cast<int>(m_vecIndices.size());  // 20260402 ZJH 样本总数
            for (int i = nN - 1; i > 0; --i) {
                // 20260402 ZJH 从 [0, i] 均匀随机选择交换位置
                std::uniform_int_distribution<int> dist(0, i);
                int j = dist(gen);  // 20260402 ZJH 随机目标索引
                std::swap(m_vecIndices[static_cast<size_t>(i)],
                          m_vecIndices[static_cast<size_t>(j)]);
            }
        }

        // 20260402 ZJH 第三步: 重置生产者/消费者状态
        m_nNextBatch.store(0, std::memory_order_relaxed);  // 20260402 ZJH 生产者从第 0 个 batch 开始
        m_nHead = 0;  // 20260402 ZJH 环形缓冲区写入头归零
        m_nTail = 0;  // 20260402 ZJH 环形缓冲区读取尾归零
        m_nProducedCount = 0;  // 20260402 ZJH 已写入缓冲区的 batch 计数归零
        m_nConsumedCount = 0;  // 20260402 ZJH 已消费的 batch 计数归零
        m_bStopped.store(false, std::memory_order_relaxed);  // 20260402 ZJH 清除停止标志

        // 20260402 ZJH 重置所有缓冲槽位为未就绪
        for (auto& buf : m_vecRingBuffer) {
            buf.bReady = false;
            buf.nActualSize = 0;
        }

        // 20260402 ZJH 第四步: 启动 worker 线程池
        m_vecWorkers.reserve(static_cast<size_t>(m_nNumWorkers));
        for (int i = 0; i < m_nNumWorkers; ++i) {
            // 20260402 ZJH 每个 worker 执行 workerLoop()，独立领取 batch 任务
            m_vecWorkers.emplace_back(&AsyncDataLoader::workerLoop, this);
        }
    }

    // 20260402 ZJH getNextBatch — 获取下一个已就绪的 batch（阻塞等待）
    // vecBatchInput: [out] 输出缓冲区，调用者提供，大小 [batchSize * inputDim]
    // vecBatchLabel: [out] 输出缓冲区，调用者提供，大小 [batchSize * numClasses]
    // 返回: true 表示成功获取 batch，false 表示本 epoch 所有 batch 已消费完毕
    bool getNextBatch(std::vector<float>& vecBatchInput,
                      std::vector<float>& vecBatchLabel)
    {
        // 20260402 ZJH 检查是否所有 batch 已消费完毕
        if (m_nConsumedCount >= m_nTotalBatches) {
            return false;  // 20260402 ZJH 本 epoch 结束
        }

        // 20260402 ZJH 等待 m_nTail 位置的缓冲区变为就绪状态
        {
            std::unique_lock<std::mutex> lock(m_mtxBuffer);  // 20260402 ZJH 获取缓冲区互斥锁
            // 20260402 ZJH 条件等待: 直到 tail 位置的 batch 就绪，或被提前停止
            m_cvProduced.wait(lock, [this]() {
                return m_vecRingBuffer[static_cast<size_t>(m_nTail)].bReady || m_bStopped.load(std::memory_order_relaxed);
            });

            // 20260402 ZJH 如果被提前停止且当前槽位未就绪，返回 epoch 结束
            if (m_bStopped.load(std::memory_order_relaxed) &&
                !m_vecRingBuffer[static_cast<size_t>(m_nTail)].bReady) {
                return false;  // 20260402 ZJH 提前终止
            }

            // 20260402 ZJH 从缓冲区拷贝数据到调用者的输出缓冲区
            BatchBuffer& buf = m_vecRingBuffer[static_cast<size_t>(m_nTail)];

            // 20260402 ZJH 计算实际数据大小（最后一个 batch 可能不满）
            size_t nInputBytes = static_cast<size_t>(buf.nActualSize) * m_nInputDim;
            size_t nLabelBytes = static_cast<size_t>(buf.nActualSize) * m_nNumClasses;

            // 20260402 ZJH 确保输出缓冲区大小足够
            vecBatchInput.resize(nInputBytes);
            vecBatchLabel.resize(nLabelBytes);

            // 20260402 ZJH 拷贝输入数据
            std::copy(buf.vecInput.begin(),
                      buf.vecInput.begin() + static_cast<ptrdiff_t>(nInputBytes),
                      vecBatchInput.begin());

            // 20260402 ZJH 拷贝标签数据
            std::copy(buf.vecLabel.begin(),
                      buf.vecLabel.begin() + static_cast<ptrdiff_t>(nLabelBytes),
                      vecBatchLabel.begin());

            // 20260402 ZJH 标记当前槽位为已消费（可复用）
            buf.bReady = false;

            // 20260402 ZJH 前进消费者尾指针（环形取模）
            m_nTail = (m_nTail + 1) % m_nBufferSize;
            ++m_nConsumedCount;  // 20260402 ZJH 递增已消费计数
        }

        // 20260402 ZJH 通知生产者: 有空闲槽位可写入
        m_cvConsumed.notify_one();

        return true;  // 20260402 ZJH 成功获取一个 batch
    }

    // 20260402 ZJH stop — 提前停止所有 worker 线程并清理缓冲区
    // 用于训练中途取消（用户按 Stop 按钮）或异常退出
    void stop() {
        stopWorkers();  // 20260402 ZJH 停止并回收所有 worker 线程
    }

    // 20260402 ZJH batchesReady — 查询缓冲区中已就绪的 batch 数
    // 用于性能监控和调试（判断预取是否跟得上消费速度）
    // 返回: 当前缓冲区中 bReady==true 的槽位数量
    int batchesReady() const {
        std::lock_guard<std::mutex> lock(m_mtxBuffer);  // 20260402 ZJH 加锁保证一致性读取
        int nCount = 0;  // 20260402 ZJH 就绪计数器
        for (const auto& buf : m_vecRingBuffer) {
            if (buf.bReady) ++nCount;  // 20260402 ZJH 统计就绪槽位
        }
        return nCount;
    }

private:
    // 20260402 ZJH BatchBuffer — 环形缓冲区中的单个槽位
    // 存放一个完整 batch 的预处理数据（输入 + one-hot 标签）
    struct BatchBuffer {
        std::vector<float> vecInput;   // 20260402 ZJH 输入数据 [batchSize * inputDim]，CHW 格式
        std::vector<float> vecLabel;   // 20260402 ZJH one-hot 标签 [batchSize * numClasses]
        int nActualSize = 0;           // 20260402 ZJH 实际样本数（最后 batch 可能 < batchSize）
        bool bReady = false;           // 20260402 ZJH 是否已填充完毕（生产者→消费者信号）
    };

    // ---- 20260402 ZJH 环形缓冲区 ----
    std::vector<BatchBuffer> m_vecRingBuffer;  // 20260402 ZJH 环形缓冲区数组
    int m_nHead = 0;       // 20260402 ZJH 生产者头指针（下一个写入位置）
    int m_nTail = 0;       // 20260402 ZJH 消费者尾指针（下一个读取位置）
    int m_nBufferSize = 0; // 20260402 ZJH 缓冲区总槽位数
    int m_nProducedCount = 0;  // 20260402 ZJH 已写入缓冲区的 batch 数（本 epoch）
    int m_nConsumedCount = 0;  // 20260402 ZJH 已消费的 batch 数（本 epoch）

    // ---- 20260402 ZJH 线程池与同步原语 ----
    std::vector<std::thread> m_vecWorkers;  // 20260402 ZJH worker 线程池
    std::atomic<bool> m_bStopped{false};    // 20260402 ZJH 停止标志（原子变量，无锁读写）
    mutable std::mutex m_mtxBuffer;         // 20260402 ZJH 缓冲区互斥锁（保护 ring buffer 状态）
    std::condition_variable m_cvProduced;   // 20260402 ZJH 生产者→消费者通知（batch 就绪）
    std::condition_variable m_cvConsumed;   // 20260402 ZJH 消费者→生产者通知（槽位空闲）

    // ---- 20260402 ZJH 数据源（指向外部数据，不拥有所有权） ----
    const std::vector<float>* m_pData = nullptr;   // 20260402 ZJH 训练数据指针 [N * inputDim]
    const std::vector<int>* m_pLabels = nullptr;   // 20260402 ZJH 标签指针 [N]
    std::vector<int> m_vecIndices;                 // 20260402 ZJH 洗牌后的样本索引数组
    std::atomic<int> m_nNextBatch{0};              // 20260402 ZJH 下一个待准备的 batch 编号（原子递增）
    int m_nTotalBatches = 0;                       // 20260402 ZJH 本 epoch 总 batch 数

    // ---- 20260402 ZJH 配置参数 ----
    int m_nInputDim = 0;        // 20260402 ZJH 单样本维度（C * H * W）
    int m_nNumClasses = 0;      // 20260402 ZJH 类别总数
    int m_nBatchSize = 0;       // 20260402 ZJH batch 大小
    int m_nNumWorkers = 4;      // 20260402 ZJH worker 线程数
    int m_nPrefetchCount = 2;   // 20260402 ZJH 预取深度
    AugmentConfig m_augCfg;     // 20260402 ZJH 增强配置
    NormPreset m_eNorm = NormPreset::None;  // 20260402 ZJH 归一化预设模式
    bool m_bAugment = false;    // 20260402 ZJH 是否启用增强
    bool m_bNormalize = false;  // 20260402 ZJH 是否启用归一化

    // 20260402 ZJH stopWorkers — 内部辅助：停止所有 worker 线程并等待回收
    // 设置停止标志 → 唤醒所有等待线程 → join 回收 → 清空线程池
    void stopWorkers() {
        // 20260402 ZJH 设置原子停止标志，worker 循环会检测此标志退出
        m_bStopped.store(true, std::memory_order_release);

        // 20260402 ZJH 唤醒所有可能阻塞在条件变量上的线程
        m_cvConsumed.notify_all();  // 20260402 ZJH 唤醒等待空闲槽位的 worker
        m_cvProduced.notify_all();  // 20260402 ZJH 唤醒等待数据的消费者

        // 20260402 ZJH 逐个 join 回收 worker 线程
        for (auto& th : m_vecWorkers) {
            if (th.joinable()) {
                th.join();  // 20260402 ZJH 阻塞直到 worker 线程退出
            }
        }
        m_vecWorkers.clear();  // 20260402 ZJH 清空线程池容器
    }

    // 20260402 ZJH workerLoop — worker 线程主循环
    // 每个 worker 独立运行此函数，通过原子递增 m_nNextBatch 领取 batch 任务
    // 领取到任务后：
    //   1. 等待环形缓冲区有空闲槽位
    //   2. 调用 fillOneBatch() 填充数据（加载 + 增强 + 归一化）
    //   3. 标记槽位就绪，通知消费者
    // 退出条件: 所有 batch 已领取完毕 或 m_bStopped 标志为 true
    void workerLoop() {
        while (true) {
            // 20260402 ZJH 检查停止标志
            if (m_bStopped.load(std::memory_order_acquire)) {
                return;  // 20260402 ZJH 收到停止信号，退出循环
            }

            // 20260402 ZJH 原子递增领取下一个 batch 编号
            // fetch_add 保证多个 worker 不会领取同一个 batch
            int nBatchIdx = m_nNextBatch.fetch_add(1, std::memory_order_acq_rel);

            // 20260402 ZJH 如果编号超出总 batch 数，说明本 epoch 所有 batch 已分配完毕
            if (nBatchIdx >= m_nTotalBatches) {
                return;  // 20260402 ZJH 无更多任务，worker 退出
            }

            // 20260402 ZJH 等待环形缓冲区中有空闲槽位可写入
            int nSlot = -1;  // 20260402 ZJH 分配到的缓冲区槽位索引
            {
                std::unique_lock<std::mutex> lock(m_mtxBuffer);  // 20260402 ZJH 获取缓冲区锁

                // 20260402 ZJH 条件等待: 直到有空闲槽位（head 位置未就绪）或被停止
                m_cvConsumed.wait(lock, [this]() {
                    // 20260402 ZJH 空闲条件: 已生产数 - 已消费数 < 缓冲区大小
                    return (m_nProducedCount - m_nConsumedCount < m_nBufferSize)
                           || m_bStopped.load(std::memory_order_relaxed);
                });

                // 20260402 ZJH 再次检查停止标志（可能是被 stop() 唤醒的）
                if (m_bStopped.load(std::memory_order_relaxed)) {
                    return;  // 20260402 ZJH 收到停止信号，退出
                }

                // 20260402 ZJH 分配 head 位置的槽位给当前 worker
                nSlot = m_nHead;
                // 20260402 ZJH 前进 head 指针（环形取模）
                m_nHead = (m_nHead + 1) % m_nBufferSize;
            }
            // 20260402 ZJH 注意: 解锁后在槽位上填充数据，不阻塞其他 worker

            // 20260402 ZJH 在分配到的槽位中填充 batch 数据（CPU 密集操作，无锁执行）
            BatchBuffer& buf = m_vecRingBuffer[static_cast<size_t>(nSlot)];
            fillOneBatch(buf, nBatchIdx);

            // 20260402 ZJH 标记槽位就绪，通知消费者
            {
                std::lock_guard<std::mutex> lock(m_mtxBuffer);  // 20260402 ZJH 获取锁保护状态修改
                buf.bReady = true;      // 20260402 ZJH 标记为已就绪
                ++m_nProducedCount;     // 20260402 ZJH 递增已生产计数
            }
            m_cvProduced.notify_one();  // 20260402 ZJH 通知消费者有新 batch 就绪
        }
    }

    // 20260402 ZJH fillOneBatch — 填充单个 batch 的完整数据
    // 执行流程: 索引切片 → 数据拷贝 → 数据增强(可选) → 归一化(可选) → one-hot 标签
    // buf: [out] 目标缓冲区槽位
    // nBatchIdx: batch 编号（用于计算样本索引范围）
    void fillOneBatch(BatchBuffer& buf, int nBatchIdx) {
        // 20260402 ZJH 计算当前 batch 的起始索引和实际样本数
        int nStart = nBatchIdx * m_nBatchSize;  // 20260402 ZJH 起始样本在索引数组中的位置
        int nTotalSamples = static_cast<int>(m_vecIndices.size());  // 20260402 ZJH 总样本数
        // 20260402 ZJH 实际样本数: 正常 batch 等于 batchSize，最后一个 batch 可能较少
        int nCurBatch = std::min(m_nBatchSize, nTotalSamples - nStart);

        buf.nActualSize = nCurBatch;  // 20260402 ZJH 记录实际样本数

        // 20260402 ZJH 清零标签缓冲区（one-hot 编码需要全零底板）
        std::fill(buf.vecLabel.begin(),
                  buf.vecLabel.begin() + static_cast<ptrdiff_t>(nCurBatch) * m_nNumClasses,
                  0.0f);

        // 20260402 ZJH 逐样本处理: 拷贝 + 增强 + 归一化 + one-hot
        for (int i = 0; i < nCurBatch; ++i) {
            // 20260402 ZJH 通过洗牌索引获取原始样本编号
            int nSampleIdx = m_vecIndices[static_cast<size_t>(nStart + i)];
            // 20260402 ZJH 计算源数据偏移（连续存储: sample[k] 在 vecData[k * inputDim] 处）
            int nSrcOffset = nSampleIdx * m_nInputDim;

            // 20260402 ZJH 目标数据偏移（batch 内第 i 个样本）
            size_t nDstOffset = static_cast<size_t>(i) * m_nInputDim;

            // 20260402 ZJH 边界检查: 防止源数据越界访问
            if (nSrcOffset + m_nInputDim <= static_cast<int>(m_pData->size())) {
                // 20260402 ZJH 拷贝输入数据到缓冲区
                std::copy(m_pData->begin() + nSrcOffset,
                          m_pData->begin() + nSrcOffset + m_nInputDim,
                          buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset));
            } else {
                // 20260402 ZJH 源数据不足时填零（防御性处理）
                std::fill(buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset),
                          buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset + m_nInputDim),
                          0.0f);
            }

            // 20260402 ZJH 数据增强: 对单个样本执行（在拷贝到缓冲区之后，就地修改）
            if (m_bAugment) {
                // 20260402 ZJH 构造临时 vector 视图指向缓冲区中的当前样本
                // 注意: augmentImage 需要独立的 vector，此处使用子范围拷贝+回写
                // 为避免额外分配，使用 thread_local 临时缓冲区
                thread_local std::vector<float> vecTmpAug;  // 20260402 ZJH 线程局部临时缓冲
                vecTmpAug.resize(static_cast<size_t>(m_nInputDim));

                // 20260402 ZJH 拷贝当前样本到临时缓冲区
                std::copy(buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset),
                          buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset + m_nInputDim),
                          vecTmpAug.begin());

                // 20260402 ZJH 推导图像空间维度（假设方形图像: H = W = sqrt(inputDim / C)）
                // 对于非方形或非标准维度，使用 inputDim 作为整体处理
                int nC = 3;  // 20260402 ZJH 默认 3 通道（RGB）
                int nSpatialDim = m_nInputDim / nC;  // 20260402 ZJH 每通道像素数
                int nH = static_cast<int>(std::sqrt(static_cast<float>(nSpatialDim)));  // 20260402 ZJH 估算高度
                int nW = nSpatialDim / std::max(nH, 1);  // 20260402 ZJH 估算宽度

                // 20260402 ZJH 如果维度不能整除，回退到灰度单通道假设
                if (nC * nH * nW != m_nInputDim) {
                    nC = 1;  // 20260402 ZJH 回退为灰度
                    nSpatialDim = m_nInputDim;
                    nH = static_cast<int>(std::sqrt(static_cast<float>(nSpatialDim)));
                    nW = nSpatialDim / std::max(nH, 1);
                    // 20260402 ZJH 如果仍不整除，使用 1D 处理（nH=1, nW=inputDim）
                    if (nH * nW != nSpatialDim) {
                        nH = 1;
                        nW = m_nInputDim;
                    }
                }

                // 20260402 ZJH 执行增强（就地修改 vecTmpAug）
                augmentImage(vecTmpAug, nC, nH, nW, m_augCfg);

                // 20260402 ZJH 回写增强后的数据到缓冲区
                std::copy(vecTmpAug.begin(), vecTmpAug.end(),
                          buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset));
            }

            // 20260402 ZJH 归一化: 对单个样本执行（在增强之后）
            if (m_bNormalize) {
                // 20260402 ZJH 使用线程局部临时缓冲区（与增强共享或独立分配）
                thread_local std::vector<float> vecTmpNorm;
                vecTmpNorm.resize(static_cast<size_t>(m_nInputDim));

                // 20260402 ZJH 拷贝到临时缓冲区
                std::copy(buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset),
                          buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset + m_nInputDim),
                          vecTmpNorm.begin());

                // 20260402 ZJH 推导通道/空间维度（同增强逻辑）
                int nC = 3;
                int nSpatialDim = m_nInputDim / nC;
                int nH = static_cast<int>(std::sqrt(static_cast<float>(nSpatialDim)));
                int nW = nSpatialDim / std::max(nH, 1);
                if (nC * nH * nW != m_nInputDim) {
                    nC = 1;
                    nSpatialDim = m_nInputDim;
                    nH = static_cast<int>(std::sqrt(static_cast<float>(nSpatialDim)));
                    nW = nSpatialDim / std::max(nH, 1);
                    if (nH * nW != nSpatialDim) { nH = 1; nW = m_nInputDim; }
                }

                // 20260402 ZJH 构造归一化配置（使用当前预设模式）
                AugmentConfig normCfg;
                normCfg.eNormPreset = m_eNorm;  // 20260402 ZJH 传递预设模式

                // 20260402 ZJH 执行归一化（就地修改 vecTmpNorm）
                normalizeImage(vecTmpNorm, nC, nH, nW, normCfg);

                // 20260402 ZJH 回写归一化后的数据到缓冲区
                std::copy(vecTmpNorm.begin(), vecTmpNorm.end(),
                          buf.vecInput.begin() + static_cast<ptrdiff_t>(nDstOffset));
            }

            // 20260402 ZJH 设置 one-hot 标签
            int nLabel = (*m_pLabels)[static_cast<size_t>(nSampleIdx)];
            // 20260402 ZJH 边界检查: 标签值必须在 [0, numClasses) 范围内
            if (nLabel >= 0 && nLabel < m_nNumClasses) {
                // 20260402 ZJH 在 one-hot 向量的对应位置设为 1.0f
                buf.vecLabel[static_cast<size_t>(i) * m_nNumClasses + nLabel] = 1.0f;
            }
        }
    }
};

}  // namespace om
