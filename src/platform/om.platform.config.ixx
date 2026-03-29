// 20260319 ZJH Config 模块 — JSON 配置文件加载/保存
// 基于 nlohmann-json，提供类型安全的配置键值存取
module;

// 20260319 ZJH 全局模块片段：所有 #include 必须放在此处，位于 export module 声明之前
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <expected>
#include <filesystem>

// 20260319 ZJH 引入公共类型定义（ErrorCode / Error / Result<T> / OM_ERROR 宏）
#include "om_types.h"

export module om.platform.config;

export namespace om {

// 20260319 ZJH 配置管理器 — 封装 nlohmann::json 对象，提供类型安全的读写和文件 IO
class Config {
public:
    // 20260319 ZJH 默认构造 — 创建空配置对象，m_jsonData 为空 JSON object
    Config() = default;

    // 20260319 ZJH 设置键值对
    // 参数: strKey — 配置键名（字符串）
    //       value  — 配置值，支持 string / int / double / bool 等 nlohmann-json 可序列化类型
    template<typename T>
    void set(const std::string& strKey, const T& value) {
        m_jsonData[strKey] = value;  // nlohmann-json 通过模板特化自动完成 C++ 类型到 JSON 值的映射
    }

    // 20260319 ZJH 获取配置值（无默认值版本）
    // 参数: strKey — 配置键名
    // 返回: T — 对应键的配置值
    // 注意: 键不存在时抛出 std::out_of_range；类型不匹配时抛出 nlohmann::json::type_error
    template<typename T>
    T get(const std::string& strKey) const {
        return m_jsonData.at(strKey).get<T>();  // at() 越界时抛异常，调用方需自行捕获
    }

    // 20260319 ZJH 获取配置值（带默认值版本）
    // 参数: strKey       — 配置键名
    //       defaultValue — 键不存在时返回的默认值，不触发异常
    // 返回: T — 对应键的配置值，或 defaultValue
    template<typename T>
    T get(const std::string& strKey, const T& defaultValue) const {
        // 20260319 ZJH 先检查键是否存在，避免 at() 抛出异常
        if (m_jsonData.contains(strKey)) {
            return m_jsonData.at(strKey).get<T>();  // 键存在时正常取值
        }
        return defaultValue;  // 键不存在时返回调用方提供的默认值
    }

    // 20260319 ZJH 检查配置键是否存在
    // 参数: strKey — 要查询的键名
    // 返回: true 表示键存在，false 表示不存在
    bool has(const std::string& strKey) const {
        return m_jsonData.contains(strKey);  // nlohmann-json contains() 不抛异常
    }

    // 20260319 ZJH 保存配置到 JSON 文件
    // 参数: strPath — 目标文件路径（绝对或相对路径均可）
    // 返回: Result<void> — 成功时为空值，失败时携带 IO 错误信息
    Result<void> save(const std::string& strPath) const {
        std::ofstream ofs(strPath);  // 打开输出文件流，文件不存在时自动创建
        // 20260319 ZJH 文件打开失败通常因为路径不存在或无写入权限
        if (!ofs.is_open()) {
            return std::unexpected(OM_ERROR(FileNotFound, "Cannot open file for writing: " + strPath));
        }
        ofs << m_jsonData.dump(4);  // 4 空格缩进美化输出，便于人工查阅
        return {};  // 返回成功（std::expected<void,...> 的成功态为空 {}）
    }

    // 20260319 ZJH 从 JSON 文件加载配置（静态工厂方法）
    // 参数: strPath — 配置文件路径
    // 返回: Result<Config> — 成功返回填充好的 Config 对象，失败返回带错误码的 Error
    static Result<Config> load(const std::string& strPath) {
        // 20260319 ZJH 先用 std::filesystem 检查文件是否存在，比直接 ifstream.open 更明确
        if (!std::filesystem::exists(strPath)) {
            return std::unexpected(OM_ERROR(FileNotFound, "Config file not found: " + strPath));
        }

        std::ifstream ifs(strPath);  // 以只读模式打开文件
        // 20260319 ZJH exists() 通过后仍可能因权限问题导致 open 失败
        if (!ifs.is_open()) {
            return std::unexpected(OM_ERROR(FileNotFound, "Cannot open config file: " + strPath));
        }

        Config config;  // 创建空 Config 对象，后续填充 JSON 数据
        try {
            ifs >> config.m_jsonData;  // nlohmann-json 流解析，解析失败抛 parse_error
        } catch (const nlohmann::json::parse_error& e) {
            // 20260319 ZJH JSON 格式错误（如缺少括号、非法字符等）映射到 InvalidFormat 错误码
            return std::unexpected(OM_ERROR(InvalidFormat, std::string("JSON parse error: ") + e.what()));
        }
        return config;  // 返回成功加载的 Config 对象
    }

private:
    nlohmann::json m_jsonData;  // 底层 JSON 数据存储，所有键值对均保存在此对象中
};

}  // namespace om
