// 20260319 ZJH Logger 模块 — spdlog 多 sink 日志封装
// 提供统一的日志接口，支持控制台 + 文件双 sink 输出
module;

// 20260319 ZJH 全局模块片段：所有 #include 必须放在此处，位于 export module 声明之前
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

export module om.platform.logger;

export namespace om {

// 20260319 ZJH 日志级别枚举 — 映射到 spdlog 级别，与 spdlog::level::level_enum 数值对应
enum class LogLevel {
    Trace    = 0,  // 最详细，用于开发调试
    Debug    = 1,  // 调试信息
    Info     = 2,  // 一般信息（默认级别）
    Warn     = 3,  // 警告
    Error    = 4,  // 错误
    Critical = 5,  // 严重错误
    Off      = 6   // 关闭日志
};

// 20260319 ZJH 日志管理器 — 静态方法全局访问，线程安全
// 封装 spdlog 多 sink logger，对外暴露统一接口
class Logger {
public:
    // 20260319 ZJH 初始化日志系统
    // 参数: strAppName — 应用名称，用于 logger 标识（也用于 spdlog 注册键名）
    //       strLogFile — 可选日志文件路径，空字符串则仅输出到控制台
    static void init(const std::string& strAppName,
                     const std::string& strLogFile = "")
    {
        // 20260319 ZJH 加锁保证多线程环境下初始化操作的原子性
        std::lock_guard<std::mutex> lock(s_mutex);

        // 20260319 ZJH 若同名 logger 已存在则先移除，支持测试中多次重复调用 init
        spdlog::drop(strAppName);

        // 20260319 ZJH 构造 sink 容器，收集所有输出目标
        std::vector<spdlog::sink_ptr> vecSinks;

        // 20260319 ZJH 创建多线程安全的彩色控制台 sink，输出到 stdout
        auto pConsoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        // 20260319 ZJH 控制台日志格式：时间戳 + 彩色级别 + 消息内容
        pConsoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        vecSinks.push_back(pConsoleSink);  // 将控制台 sink 加入列表

        // 20260319 ZJH 若指定了文件路径，则追加轮转文件 sink
        if (!strLogFile.empty()) {
            // 20260319 ZJH 5MB 单文件上限，保留最多 3 个历史轮转文件
            auto pFileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                strLogFile, 5 * 1024 * 1024, 3);
            // 20260319 ZJH 文件日志格式：时间戳 + 级别 + 源文件位置 + 消息
            pFileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
            vecSinks.push_back(pFileSink);  // 将文件 sink 加入列表
        }

        // 20260319 ZJH 使用 sink 列表创建具名多 sink logger 实例
        s_pLogger = std::make_shared<spdlog::logger>(
            strAppName, vecSinks.begin(), vecSinks.end());

        // 20260319 ZJH 设置最低输出级别为 debug，捕获所有调试及以上日志
        s_pLogger->set_level(spdlog::level::debug);

        // 20260319 ZJH warn 及以上级别自动 flush，避免程序崩溃时丢失关键日志
        s_pLogger->flush_on(spdlog::level::warn);

        // 20260319 ZJH 注册为 spdlog 全局默认 logger，方便全局宏访问
        spdlog::set_default_logger(s_pLogger);
    }

    // 20260319 ZJH 动态设置日志输出级别
    // 参数: level — 新的日志级别，低于此级别的日志将被过滤
    static void setLevel(LogLevel level)
    {
        // 20260319 ZJH 仅在 logger 已初始化时执行，避免空指针
        if (s_pLogger) {
            // 20260319 ZJH 将枚举值转换为 spdlog 内部级别类型并应用
            s_pLogger->set_level(
                static_cast<spdlog::level::level_enum>(static_cast<int>(level)));
        }
    }

    // 20260319 ZJH TRACE 级别日志 — 用于极细粒度的执行路径追踪
    // 注意：MSVC 模块边界不支持直接导出 format_string_t 模板参数，改用 std::string_view
    static void trace(std::string_view strMsg)
    {
        // 20260319 ZJH 防止 logger 未初始化时调用导致崩溃
        if (s_pLogger) s_pLogger->trace(strMsg);
    }

    // 20260319 ZJH DEBUG 级别日志 — 用于开发期调试信息输出
    static void debug(std::string_view strMsg)
    {
        if (s_pLogger) s_pLogger->debug(strMsg);  // 转发到 spdlog debug 输出
    }

    // 20260319 ZJH INFO 级别日志 — 记录正常运行状态信息
    static void info(std::string_view strMsg)
    {
        if (s_pLogger) s_pLogger->info(strMsg);  // 转发到 spdlog info 输出
    }

    // 20260319 ZJH WARN 级别日志 — 记录潜在问题但不影响主流程的警告
    static void warn(std::string_view strMsg)
    {
        if (s_pLogger) s_pLogger->warn(strMsg);  // 转发到 spdlog warn 输出，并自动 flush
    }

    // 20260319 ZJH ERROR 级别日志 — 记录导致功能失败的错误
    static void error(std::string_view strMsg)
    {
        if (s_pLogger) s_pLogger->error(strMsg);  // 转发到 spdlog error 输出
    }

    // 20260319 ZJH CRITICAL 级别日志 — 记录导致系统崩溃的严重错误
    static void critical(std::string_view strMsg)
    {
        if (s_pLogger) s_pLogger->critical(strMsg);  // 转发到 spdlog critical 输出
    }

private:
    // 20260319 ZJH 全局 logger 实例 — inline static 在头文件/模块中直接定义
    static inline std::shared_ptr<spdlog::logger> s_pLogger;

    // 20260319 ZJH 初始化互斥锁 — 保护 init 方法的线程安全
    static inline std::mutex s_mutex;
};

}  // namespace om
