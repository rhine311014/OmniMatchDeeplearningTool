// 20260319 ZJH FileSystem 模块 — 跨平台文件与目录操作封装
// 基于 std::filesystem + 标准 IO，提供目录创建、文件读写、目录枚举等功能
module;

// 20260319 ZJH 全局模块片段：所有 #include 必须置于此处（export module 声明之前）
#include <filesystem>   // std::filesystem — 目录/文件系统操作
#include <fstream>      // std::ifstream / std::ofstream — 文件读写流
#include <string>       // std::string — 路径和文本内容
#include <vector>       // std::vector — 二进制缓冲区和文件列表
#include <expected>     // std::expected / std::unexpected — Result<T> 基础
#include <cstdint>      // uint8_t — 二进制数据元素类型
#include <iterator>     // std::istreambuf_iterator — 高效文件内容读取

// 20260319 ZJH 引入公共类型定义（ErrorCode / Error / Result<T> / DF_ERROR 宏）
#include "df_types.h"

export module df.platform.filesystem;

// 20260319 ZJH std::filesystem 命名空间别名，缩短内部引用
namespace fs = std::filesystem;

export namespace df {

// 20260319 ZJH FileSystem 静态工具类 — 封装常用文件系统操作，全部使用静态方法，无需实例化
// 所有可能失败的操作均返回 Result<T>，成功携带结果值，失败携带 Error（含错误码和描述）
class FileSystem {
public:
    // 20260319 ZJH 禁止构造 — 纯静态工具类，不应被实例化
    FileSystem() = delete;

    // -----------------------------------------------------------------------
    // 20260319 ZJH ensureDir — 递归创建目录（含所有中间级目录）
    // 参数: strPath — 目标目录路径（绝对或相对路径均可）
    // 返回: Result<void> — 成功返回空值；目录已存在也视为成功；失败返回 InternalError
    // -----------------------------------------------------------------------
    static Result<void> ensureDir(const std::string& strPath) {
        // 20260319 ZJH 目录已存在时直接返回成功，避免无谓的系统调用
        if (fs::exists(strPath)) {
            return {};  // 已存在，视为成功
        }

        std::error_code ec;  // 用于捕获 std::filesystem 的错误码，避免异常抛出
        // 20260319 ZJH create_directories 递归创建多级目录，等价于 mkdir -p
        // 返回 false 表示失败（ec 携带具体操作系统错误）
        bool bCreated = fs::create_directories(strPath, ec);
        if (!bCreated && ec) {
            // 20260319 ZJH ec.message() 返回操作系统级错误描述（如"拒绝访问"）
            return std::unexpected(DF_ERROR(InternalError,
                "Failed to create directory: " + strPath + " (" + ec.message() + ")"));
        }
        return {};  // 目录创建成功
    }

    // -----------------------------------------------------------------------
    // 20260319 ZJH exists — 检查路径是否存在（文件或目录均可）
    // 参数: strPath — 要检查的路径
    // 返回: true 表示路径存在，false 表示不存在或访问失败
    // -----------------------------------------------------------------------
    static bool exists(const std::string& strPath) {
        std::error_code ec;  // 使用 error_code 重载，路径无效时不抛异常
        return fs::exists(strPath, ec);  // ec 非零表示底层访问错误，此时 exists 返回 false
    }

    // -----------------------------------------------------------------------
    // 20260319 ZJH writeText — 将字符串内容写入文本文件（覆盖写）
    // 参数: strPath    — 目标文件路径
    //       strContent — 要写入的文本内容
    // 返回: Result<void> — 成功返回空值；打开/写入失败返回 InternalError
    // -----------------------------------------------------------------------
    static Result<void> writeText(const std::string& strPath, const std::string& strContent) {
        std::ofstream ofs(strPath);  // 以覆盖写模式打开文件，不存在则自动创建
        // 20260319 ZJH is_open() 失败通常因为父目录不存在或无写入权限
        if (!ofs.is_open()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Cannot open file for writing: " + strPath));
        }
        ofs << strContent;  // 将字符串内容写入文件流
        // 20260319 ZJH 检查流状态，确认写入过程未出现 IO 错误
        if (!ofs.good()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Write failed for file: " + strPath));
        }
        return {};  // 写入成功
    }

    // -----------------------------------------------------------------------
    // 20260319 ZJH readText — 将文本文件全部内容读取为字符串
    // 参数: strPath — 源文件路径
    // 返回: Result<std::string> — 成功返回文件内容字符串；文件不存在返回 FileNotFound；
    //        读取失败返回 InternalError
    // -----------------------------------------------------------------------
    static Result<std::string> readText(const std::string& strPath) {
        // 20260319 ZJH 先检查文件是否存在，给出语义更明确的错误码
        if (!fs::exists(strPath)) {
            return std::unexpected(DF_ERROR(FileNotFound,
                "File not found: " + strPath));
        }

        std::ifstream ifs(strPath);  // 以只读模式打开文件
        // 20260319 ZJH exists() 通过后仍可能因权限等原因导致 open 失败
        if (!ifs.is_open()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Cannot open file for reading: " + strPath));
        }

        // 20260319 ZJH istreambuf_iterator 直接读取底层缓冲区，比 getline 循环更高效
        // 将整个文件流内容一次性构造为 std::string
        std::string strContent(
            (std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>()
        );

        // 20260319 ZJH 检查流读取状态，确认无 IO 错误
        if (ifs.bad()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Read failed for file: " + strPath));
        }

        return strContent;  // 返回完整文件内容
    }

    // -----------------------------------------------------------------------
    // 20260319 ZJH writeBinary — 将字节数组写入二进制文件（覆盖写）
    // 参数: strPath  — 目标文件路径
    //       vecData  — 要写入的字节数组（uint8_t 元素）
    // 返回: Result<void> — 成功返回空值；打开/写入失败返回 InternalError
    // -----------------------------------------------------------------------
    static Result<void> writeBinary(const std::string& strPath,
                                     const std::vector<uint8_t>& vecData) {
        // 20260319 ZJH std::ios::binary 防止 Windows 上 \n 被转换为 \r\n，保证字节精确写入
        std::ofstream ofs(strPath, std::ios::binary);
        if (!ofs.is_open()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Cannot open file for binary writing: " + strPath));
        }

        if (!vecData.empty()) {
            // 20260319 ZJH reinterpret_cast<char*> 将 uint8_t* 转换为 write() 所需的 char*
            // vecData.size() 给出要写入的字节总数
            ofs.write(reinterpret_cast<const char*>(vecData.data()),
                      static_cast<std::streamsize>(vecData.size()));
        }

        // 20260319 ZJH 检查写入后的流状态，确认所有字节均已成功写入
        if (!ofs.good()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Binary write failed for file: " + strPath));
        }
        return {};  // 二进制写入成功
    }

    // -----------------------------------------------------------------------
    // 20260319 ZJH readBinary — 将二进制文件全部内容读取为字节数组
    // 参数: strPath — 源文件路径
    // 返回: Result<std::vector<uint8_t>> — 成功返回字节数组；文件不存在返回 FileNotFound；
    //        读取失败返回 InternalError
    // -----------------------------------------------------------------------
    static Result<std::vector<uint8_t>> readBinary(const std::string& strPath) {
        // 20260319 ZJH 先检查文件是否存在，提供明确的 FileNotFound 错误码
        if (!fs::exists(strPath)) {
            return std::unexpected(DF_ERROR(FileNotFound,
                "File not found: " + strPath));
        }

        // 20260319 ZJH std::ios::binary + std::ios::ate 组合：二进制模式打开，
        // 并将读取位置移至文件末尾以便 tellg() 获取文件大小
        std::ifstream ifs(strPath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            return std::unexpected(DF_ERROR(InternalError,
                "Cannot open file for binary reading: " + strPath));
        }

        // 20260319 ZJH tellg() 在 ate 模式下返回文件大小（字节数）
        std::streamsize nSize = ifs.tellg();
        // 20260319 ZJH 将读取位置重置到文件头，以便后续 read() 从头读取
        ifs.seekg(0, std::ios::beg);

        // 20260319 ZJH 预分配目标缓冲区，避免 push_back 的多次重分配开销
        std::vector<uint8_t> vecData(static_cast<size_t>(nSize));

        if (nSize > 0) {
            // 20260319 ZJH 一次性读取全部字节，reinterpret_cast 将 uint8_t* 转为 char*
            ifs.read(reinterpret_cast<char*>(vecData.data()), nSize);
            // 20260319 ZJH 验证实际读取字节数等于预期文件大小
            if (ifs.gcount() != nSize) {
                return std::unexpected(DF_ERROR(InternalError,
                    "Binary read incomplete for file: " + strPath));
            }
        }

        return vecData;  // 返回完整字节数组
    }

    // -----------------------------------------------------------------------
    // 20260319 ZJH listFiles — 枚举指定目录下的文件（非递归，可按扩展名过滤）
    // 参数: strDir       — 要枚举的目录路径
    //       strExtension — 扩展名过滤（含点号，如 ".txt"）；空字符串表示返回所有文件
    // 返回: Result<std::vector<std::string>> — 成功返回文件路径字符串列表（绝对路径）；
    //        目录不存在返回 FileNotFound；不是目录返回 InvalidArgument
    // -----------------------------------------------------------------------
    static Result<std::vector<std::string>> listFiles(const std::string& strDir,
                                                       const std::string& strExtension = "") {
        // 20260319 ZJH 目录不存在时返回 FileNotFound，提示调用方先调用 ensureDir
        if (!fs::exists(strDir)) {
            return std::unexpected(DF_ERROR(FileNotFound,
                "Directory not found: " + strDir));
        }
        // 20260319 ZJH 路径存在但不是目录时返回 InvalidArgument，防止对文件调用此方法
        if (!fs::is_directory(strDir)) {
            return std::unexpected(DF_ERROR(InvalidArgument,
                "Path is not a directory: " + strDir));
        }

        std::vector<std::string> vecFiles;  // 用于收集符合条件的文件路径

        // 20260319 ZJH directory_iterator 非递归枚举目录下的所有直接子项
        for (const auto& entry : fs::directory_iterator(strDir)) {
            // 20260319 ZJH 跳过子目录，只收集常规文件
            if (!entry.is_regular_file()) {
                continue;  // 非常规文件（目录、符号链接等）跳过
            }

            // 20260319 ZJH 扩展名过滤：strExtension 非空时只收集匹配的文件
            if (!strExtension.empty()) {
                // 20260319 ZJH entry.path().extension() 返回含点号的扩展名（如 ".txt"）
                if (entry.path().extension().string() != strExtension) {
                    continue;  // 扩展名不匹配，跳过此文件
                }
            }

            // 20260319 ZJH 将符合条件的文件路径转换为 string 后加入结果列表
            vecFiles.push_back(entry.path().string());
        }

        return vecFiles;  // 返回文件路径列表（可能为空，但这不是错误）
    }
};

}  // namespace df
