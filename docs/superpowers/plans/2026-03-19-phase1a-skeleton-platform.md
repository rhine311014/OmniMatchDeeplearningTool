# Phase 1A: 项目骨架 + 平台层基础设施 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 DeepForge 项目骨架（CMake + vcpkg + 目录结构 + git），并实现平台层 6 个模块（Logger、Config、FileSystem、MemoryPool、ThreadPool、Database），为上层 Tensor/AutoGrad 提供基础设施。

**Architecture:** C++23 模块化 (.ixx) 分层架构。平台层是最底层（Layer 1），不依赖任何上层。每个模块一个 .ixx 文件，对外导出一个 C++23 module。使用 vcpkg 管理第三方依赖（spdlog、nlohmann-json、sqlite3、GTest）。CMake 4.2+ 的 `CXX_MODULE_SETS` 管理模块编译。

**Tech Stack:** C++23, MSVC 14.50 (VS2026), CMake 4.2.3, vcpkg, spdlog, nlohmann-json, SQLite3, Google Test

**Spec Reference:** `docs/superpowers/specs/2026-03-19-deepforge-design.md`

---

## File Map

| 文件路径 | 职责 |
|---------|------|
| `CMakeLists.txt` | 顶层 CMake 配置 |
| `CMakePresets.json` | 编译预设（windows-debug/release） |
| `vcpkg.json` | vcpkg 依赖清单 |
| `include/df_types.h` | 公共类型定义（ErrorCode、Result、DeviceType 等） |
| `src/platform/df.platform.logger.ixx` | spdlog 封装，多 sink 日志 |
| `src/platform/df.platform.config.ixx` | JSON 配置加载/保存 |
| `src/platform/df.platform.filesystem.ixx` | 跨平台文件操作封装 |
| `src/platform/df.platform.memory.ixx` | 内存池化分配器 |
| `src/platform/df.platform.thread_pool.ixx` | std::jthread 线程池 |
| `src/platform/df.platform.database.ixx` | SQLite RAII 封装 |
| `tests/test_logger.cpp` | Logger 单元测试 |
| `tests/test_config.cpp` | Config 单元测试 |
| `tests/test_filesystem.cpp` | FileSystem 单元测试 |
| `tests/test_memory.cpp` | MemoryPool 单元测试 |
| `tests/test_thread_pool.cpp` | ThreadPool 单元测试 |
| `tests/test_database.cpp` | Database 单元测试 |

---

## Task 1: 项目骨架 — CMake + vcpkg + 目录结构 + git

**Files:**
- Create: `CMakeLists.txt`
- Create: `CMakePresets.json`
- Create: `vcpkg.json`
- Create: `include/df_types.h`
- Create: `.gitignore`
- Create: 6 个 stub `.ixx` 文件（空模块占位，确保 CMake configure 通过）

- [ ] **Step 1: 初始化 git 仓库**

```bash
cd E:/DevelopmentTools/Co-creationDeepLearningTool
git init
```

- [ ] **Step 2: 创建 .gitignore**

```gitignore
# Build
build/
out/
cmake-build-*/

# IDE
.vs/
.vscode/
*.user
*.suo

# vcpkg
vcpkg_installed/

# Data
data/datasets/
data/models/
*.db

# OS
Thumbs.db
Desktop.ini
.DS_Store
```

- [ ] **Step 3: 创建目录结构**

```bash
mkdir -p src/platform src/hal src/engine src/business src/ui src/cuda src/opencl
mkdir -p include resources/fonts resources/icons config data tests third_party
```

- [ ] **Step 4: 创建 vcpkg.json**

```json
{
  "name": "deepforge",
  "version": "0.1.0",
  "description": "Pure C++ Deep Learning Vision Platform",
  "dependencies": [
    "spdlog",
    "nlohmann-json",
    "sqlite3",
    "gtest"
  ]
}
```

- [ ] **Step 5: 创建 include/df_types.h — 公共类型定义**

```cpp
// 20260319 ZJH DeepForge 全局公共类型定义
// 所有层共享的枚举、错误码、Result 别名等基础类型
#pragma once

#include <expected>
#include <string>
#include <cstdint>

namespace df {

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
#define DF_ERROR(code, msg) \
    df::Error{ df::ErrorCode::code, msg, __FILE__, __LINE__ }

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

}  // namespace df
```

- [ ] **Step 6: 创建 CMakeLists.txt**

```cmake
# 20260319 ZJH DeepForge 顶层 CMake 配置
# 纯 C++ 全流程深度学习视觉平台
cmake_minimum_required(VERSION 3.30)

project(DeepForge
    VERSION 0.1.0
    DESCRIPTION "Pure C++ Deep Learning Vision Platform"
    LANGUAGES CXX
)

# ---------- C++23 标准 ----------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- 输出目录 ----------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------- 编译选项 ----------
if(MSVC)
    add_compile_options(/W4 /utf-8 /Zc:__cplusplus)
endif()

# ---------- 条件编译选项 ----------
option(DF_ENABLE_CUDA "Enable CUDA backend" OFF)
option(DF_ENABLE_OPENCL "Enable OpenCL backend" OFF)
option(DF_ENABLE_TENSORRT "Enable TensorRT inference" OFF)
option(DF_BUILD_TESTS "Build unit tests" ON)

# ---------- CUDA（Phase 2 启用） ----------
if(DF_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---------- vcpkg 依赖 ----------
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

# ---------- 平台层静态库（Layer 1） ----------
add_library(df_platform STATIC)

target_sources(df_platform
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/platform/df.platform.logger.ixx
            src/platform/df.platform.config.ixx
            src/platform/df.platform.filesystem.ixx
            src/platform/df.platform.memory.ixx
            src/platform/df.platform.thread_pool.ixx
            src/platform/df.platform.database.ixx
)

target_include_directories(df_platform PUBLIC include)

target_link_libraries(df_platform
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        unofficial::sqlite3::sqlite3
)

# ---------- 测试 ----------
if(DF_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)

    # 辅助函数：添加测试可执行文件
    function(df_add_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE df_platform GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    df_add_test(test_logger tests/test_logger.cpp)
    df_add_test(test_config tests/test_config.cpp)
    df_add_test(test_filesystem tests/test_filesystem.cpp)
    df_add_test(test_memory tests/test_memory.cpp)
    df_add_test(test_thread_pool tests/test_thread_pool.cpp)
    df_add_test(test_database tests/test_database.cpp)
endif()
```

- [ ] **Step 7: 创建 CMakePresets.json**

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "windows-base",
      "hidden": true,
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "toolchainFile": "E:/DevelopmentTools/vcpkg/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": {
        "CMAKE_CXX_STANDARD": "23"
      }
    },
    {
      "name": "windows-debug",
      "displayName": "Windows Debug",
      "inherits": "windows-base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      }
    },
    {
      "name": "windows-release",
      "displayName": "Windows Release",
      "inherits": "windows-base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "windows-debug",
      "configurePreset": "windows-debug"
    },
    {
      "name": "windows-release",
      "configurePreset": "windows-release"
    }
  ],
  "testPresets": [
    {
      "name": "windows-debug",
      "configurePreset": "windows-debug",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

- [ ] **Step 8: 创建 stub .ixx 文件（确保 CMake configure 通过）**

每个 stub 文件只包含最小的 module 声明，后续 Task 会替换为真正实现。

`src/platform/df.platform.logger.ixx`:
```cpp
export module df.platform.logger;
```

`src/platform/df.platform.config.ixx`:
```cpp
export module df.platform.config;
```

`src/platform/df.platform.filesystem.ixx`:
```cpp
export module df.platform.filesystem;
```

`src/platform/df.platform.memory.ixx`:
```cpp
export module df.platform.memory;
```

`src/platform/df.platform.thread_pool.ixx`:
```cpp
export module df.platform.thread_pool;
```

`src/platform/df.platform.database.ixx`:
```cpp
export module df.platform.database;
```

- [ ] **Step 9: 验证 CMake configure 成功**

```bash
cmake --preset windows-debug
```

Expected: Configure 成功，无错误。所有 stub 模块已就绪，CMake 能正确识别 C++23 模块。

- [ ] **Step 10: 提交初始骨架**

```bash
git add .gitignore CMakeLists.txt CMakePresets.json vcpkg.json include/df_types.h config/default_config.json src/platform/*.ixx
git commit -m "feat: init project skeleton with CMake, vcpkg, and type definitions"
```

---

## Task 2: Logger — spdlog 多 sink 日志封装

**Files:**
- Create: `src/platform/df.platform.logger.ixx`
- Create: `tests/test_logger.cpp`

- [ ] **Step 1: 编写 Logger 测试**

```cpp
// 20260319 ZJH Logger 模块单元测试
#include <gtest/gtest.h>
import df.platform.logger;

// 20260319 ZJH 测试初始化 — Logger 应能正常初始化且不抛异常
TEST(LoggerTest, InitDoesNotThrow) {
    EXPECT_NO_THROW(df::Logger::init("test_app"));
}

// 20260319 ZJH 测试日志输出 — 各级别日志调用不应崩溃
TEST(LoggerTest, LogLevelsDoNotCrash) {
    df::Logger::init("test_app");
    EXPECT_NO_THROW(df::Logger::info("info message: {}", 42));
    EXPECT_NO_THROW(df::Logger::warn("warn message: {}", "test"));
    EXPECT_NO_THROW(df::Logger::error("error message"));
    EXPECT_NO_THROW(df::Logger::debug("debug message"));
}

// 20260319 ZJH 测试日志级别设置 — 应能动态切换日志级别
TEST(LoggerTest, SetLevel) {
    df::Logger::init("test_app");
    EXPECT_NO_THROW(df::Logger::setLevel(df::LogLevel::Warn));
    EXPECT_NO_THROW(df::Logger::setLevel(df::LogLevel::Debug));
}

// 20260319 ZJH 测试文件日志 — 初始化时指定日志文件路径
TEST(LoggerTest, InitWithFile) {
    EXPECT_NO_THROW(df::Logger::init("test_app", "test_log.txt"));
    df::Logger::info("file log test");
    // 清理测试日志文件
    std::remove("test_log.txt");
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cmake --build build/windows-debug --target test_logger
```

Expected: 编译失败 — `import df.platform.logger` 找不到模块

- [ ] **Step 3: 实现 Logger 模块**

```cpp
// 20260319 ZJH Logger 模块 — spdlog 多 sink 日志封装
// 提供统一的日志接口，支持控制台 + 文件双 sink 输出
module;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

export module df.platform.logger;

export namespace df {

// 20260319 ZJH 日志级别枚举 — 映射到 spdlog 级别
enum class LogLevel {
    Trace = 0,  // 最详细，用于开发调试
    Debug,      // 调试信息
    Info,       // 一般信息（默认级别）
    Warn,       // 警告
    Error,      // 错误
    Critical,   // 严重错误
    Off         // 关闭日志
};

// 20260319 ZJH 日志管理器 — 静态方法全局访问，线程安全
class Logger {
public:
    // 20260319 ZJH 初始化日志系统
    // 参数: strAppName — 应用名称，用于 logger 标识
    //       strLogFile — 可选日志文件路径，空则仅输出到控制台
    static void init(const std::string& strAppName,
                     const std::string& strLogFile = "") {
        std::lock_guard<std::mutex> lock(s_mutex);  // 保证初始化线程安全

        // 20260319 ZJH 如果已存在同名 logger，先移除（支持多次 init 调用，测试友好）
        spdlog::drop(strAppName);

        // 20260319 ZJH 创建 sink 列表
        std::vector<spdlog::sink_ptr> vecSinks;

        // 20260319 ZJH 控制台 sink — 彩色输出到 stdout
        auto pConsoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        pConsoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        vecSinks.push_back(pConsoleSink);

        // 20260319 ZJH 文件 sink — 可选，5MB 单文件上限，保留 3 个轮转文件
        if (!strLogFile.empty()) {
            auto pFileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                strLogFile, 5 * 1024 * 1024, 3);
            pFileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
            vecSinks.push_back(pFileSink);
        }

        // 20260319 ZJH 创建多 sink logger
        s_pLogger = std::make_shared<spdlog::logger>(
            strAppName, vecSinks.begin(), vecSinks.end());
        s_pLogger->set_level(spdlog::level::debug);  // 默认 debug 级别
        s_pLogger->flush_on(spdlog::level::warn);     // warn 及以上自动刷新

        // 20260319 ZJH 注册为全局默认 logger
        spdlog::set_default_logger(s_pLogger);
    }

    // 20260319 ZJH 设置日志级别
    static void setLevel(LogLevel level) {
        if (s_pLogger) {
            s_pLogger->set_level(static_cast<spdlog::level::level_enum>(static_cast<int>(level)));
        }
    }

    // 20260319 ZJH 各级别日志输出（变参模板转发到 spdlog）
    template<typename... Args>
    static void trace(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        if (s_pLogger) s_pLogger->trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        if (s_pLogger) s_pLogger->debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        if (s_pLogger) s_pLogger->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        if (s_pLogger) s_pLogger->warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        if (s_pLogger) s_pLogger->error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void critical(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        if (s_pLogger) s_pLogger->critical(fmt, std::forward<Args>(args)...);
    }

private:
    static inline std::shared_ptr<spdlog::logger> s_pLogger;  // 全局 logger 实例
    static inline std::mutex s_mutex;                           // 初始化互斥锁
};

}  // namespace df
```

- [ ] **Step 4: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_logger
cd build/windows-debug && ctest -R test_logger -V
```

Expected: 4 tests PASSED

- [ ] **Step 5: 提交**

```bash
git add src/platform/df.platform.logger.ixx tests/test_logger.cpp
git commit -m "feat(platform): add Logger module with spdlog multi-sink"
```

---

## Task 3: Config — JSON 配置加载/保存

**Files:**
- Create: `src/platform/df.platform.config.ixx`
- Create: `tests/test_config.cpp`

- [ ] **Step 1: 编写 Config 测试**

```cpp
// 20260319 ZJH Config 模块单元测试
#include <gtest/gtest.h>
#include <filesystem>
import df.platform.config;

namespace fs = std::filesystem;

// 20260319 ZJH 测试默认构造 — 空配置应可正常创建
TEST(ConfigTest, DefaultConstruct) {
    df::Config config;
    EXPECT_FALSE(config.has("nonexistent"));
}

// 20260319 ZJH 测试键值存取 — 支持 string / int / float / bool
TEST(ConfigTest, SetAndGet) {
    df::Config config;
    config.set("name", std::string("DeepForge"));
    config.set("version", 1);
    config.set("learning_rate", 0.001);
    config.set("enable_cuda", true);

    EXPECT_EQ(config.get<std::string>("name"), "DeepForge");
    EXPECT_EQ(config.get<int>("version"), 1);
    EXPECT_DOUBLE_EQ(config.get<double>("learning_rate"), 0.001);
    EXPECT_TRUE(config.get<bool>("enable_cuda"));
}

// 20260319 ZJH 测试默认值 — 键不存在时返回默认值
TEST(ConfigTest, GetWithDefault) {
    df::Config config;
    EXPECT_EQ(config.get<int>("missing", 42), 42);
    EXPECT_EQ(config.get<std::string>("missing", "default"), "default");
}

// 20260319 ZJH 测试文件保存和加载 — 序列化到 JSON 文件后再读回
TEST(ConfigTest, SaveAndLoad) {
    const std::string strPath = "test_config_temp.json";

    // 保存
    {
        df::Config config;
        config.set("batch_size", 16);
        config.set("model", std::string("resnet18"));
        auto result = config.save(strPath);
        EXPECT_TRUE(result.has_value());
    }

    // 加载
    {
        auto result = df::Config::load(strPath);
        ASSERT_TRUE(result.has_value());
        auto& config = result.value();
        EXPECT_EQ(config.get<int>("batch_size"), 16);
        EXPECT_EQ(config.get<std::string>("model"), "resnet18");
    }

    // 清理
    fs::remove(strPath);
}

// 20260319 ZJH 测试加载不存在的文件 — 应返回错误
TEST(ConfigTest, LoadNonExistent) {
    auto result = df::Config::load("nonexistent_file.json");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, df::ErrorCode::FileNotFound);
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cmake --build build/windows-debug --target test_config
```

Expected: 编译失败 — 模块不存在

- [ ] **Step 3: 实现 Config 模块**

```cpp
// 20260319 ZJH Config 模块 — JSON 配置文件加载/保存
// 基于 nlohmann-json，提供类型安全的配置键值存取
module;

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <expected>
#include <filesystem>

// 20260319 ZJH 引入公共类型定义
#include "df_types.h"

export module df.platform.config;

export namespace df {

// 20260319 ZJH 配置管理器 — 封装 JSON 对象，提供类型安全的读写和文件 IO
class Config {
public:
    // 20260319 ZJH 默认构造 — 创建空配置
    Config() = default;

    // 20260319 ZJH 设置键值对
    // 参数: strKey — 配置键名，value — 配置值（支持 string/int/double/bool）
    template<typename T>
    void set(const std::string& strKey, const T& value) {
        m_jsonData[strKey] = value;  // nlohmann-json 自动序列化
    }

    // 20260319 ZJH 获取配置值（无默认值版本）
    // 键不存在或类型不匹配时抛出异常
    template<typename T>
    T get(const std::string& strKey) const {
        return m_jsonData.at(strKey).get<T>();
    }

    // 20260319 ZJH 获取配置值（带默认值版本）
    // 键不存在时返回默认值，不抛异常
    template<typename T>
    T get(const std::string& strKey, const T& defaultValue) const {
        if (m_jsonData.contains(strKey)) {
            return m_jsonData.at(strKey).get<T>();
        }
        return defaultValue;  // 键不存在时返回调用方提供的默认值
    }

    // 20260319 ZJH 检查键是否存在
    bool has(const std::string& strKey) const {
        return m_jsonData.contains(strKey);
    }

    // 20260319 ZJH 保存配置到 JSON 文件
    // 返回: Result<void> — 成功或包含 IO 错误信息
    Result<void> save(const std::string& strPath) const {
        std::ofstream ofs(strPath);  // 打开输出文件流
        if (!ofs.is_open()) {
            // 文件打开失败，返回错误
            return std::unexpected(DF_ERROR(FileNotFound, "Cannot open file for writing: " + strPath));
        }
        ofs << m_jsonData.dump(4);  // 4 空格缩进美化输出
        return {};  // 返回成功（空 expected）
    }

    // 20260319 ZJH 从 JSON 文件加载配置（静态工厂方法）
    // 返回: Result<Config> — 成功返回 Config 对象，失败返回错误
    static Result<Config> load(const std::string& strPath) {
        // 20260319 ZJH 先检查文件是否存在
        if (!std::filesystem::exists(strPath)) {
            return std::unexpected(DF_ERROR(FileNotFound, "Config file not found: " + strPath));
        }

        std::ifstream ifs(strPath);  // 打开输入文件流
        if (!ifs.is_open()) {
            return std::unexpected(DF_ERROR(FileNotFound, "Cannot open config file: " + strPath));
        }

        Config config;
        try {
            ifs >> config.m_jsonData;  // nlohmann-json 解析
        } catch (const nlohmann::json::parse_error& e) {
            // JSON 解析失败
            return std::unexpected(DF_ERROR(InvalidFormat, std::string("JSON parse error: ") + e.what()));
        }
        return config;  // 返回成功
    }

private:
    nlohmann::json m_jsonData;  // 底层 JSON 数据存储
};

}  // namespace df
```

- [ ] **Step 4: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_config
cd build/windows-debug && ctest -R test_config -V
```

Expected: 5 tests PASSED

- [ ] **Step 5: 提交**

```bash
git add src/platform/df.platform.config.ixx tests/test_config.cpp
git commit -m "feat(platform): add Config module with JSON load/save"
```

---

## Task 4: FileSystem — 跨平台文件操作封装

**Files:**
- Create: `src/platform/df.platform.filesystem.ixx`
- Create: `tests/test_filesystem.cpp`

- [ ] **Step 1: 编写 FileSystem 测试**

```cpp
// 20260319 ZJH FileSystem 模块单元测试
#include <gtest/gtest.h>
#include <filesystem>
import df.platform.filesystem;

namespace fs = std::filesystem;

// 20260319 ZJH 测试确保目录存在 — 创建不存在的目录
TEST(FileSystemTest, EnsureDir) {
    const std::string strDir = "test_fs_temp_dir/sub/deep";
    auto result = df::FileSystem::ensureDir(strDir);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(fs::exists(strDir));
    fs::remove_all("test_fs_temp_dir");  // 清理
}

// 20260319 ZJH 测试读写文本文件
TEST(FileSystemTest, ReadWriteText) {
    const std::string strPath = "test_fs_text.txt";
    const std::string strContent = "Hello DeepForge!";

    auto writeResult = df::FileSystem::writeText(strPath, strContent);
    EXPECT_TRUE(writeResult.has_value());

    auto readResult = df::FileSystem::readText(strPath);
    ASSERT_TRUE(readResult.has_value());
    EXPECT_EQ(readResult.value(), strContent);

    fs::remove(strPath);  // 清理
}

// 20260319 ZJH 测试读写二进制文件
TEST(FileSystemTest, ReadWriteBinary) {
    const std::string strPath = "test_fs_bin.dat";
    std::vector<uint8_t> vecData = {0x00, 0xFF, 0x42, 0x99};

    auto writeResult = df::FileSystem::writeBinary(strPath, vecData);
    EXPECT_TRUE(writeResult.has_value());

    auto readResult = df::FileSystem::readBinary(strPath);
    ASSERT_TRUE(readResult.has_value());
    EXPECT_EQ(readResult.value(), vecData);

    fs::remove(strPath);  // 清理
}

// 20260319 ZJH 测试文件存在检查
TEST(FileSystemTest, Exists) {
    EXPECT_FALSE(df::FileSystem::exists("nonexistent_file_xyz"));
    df::FileSystem::writeText("test_fs_exists.txt", "data");
    EXPECT_TRUE(df::FileSystem::exists("test_fs_exists.txt"));
    fs::remove("test_fs_exists.txt");
}

// 20260319 ZJH 测试列举目录中指定扩展名的文件
TEST(FileSystemTest, ListFiles) {
    fs::create_directories("test_fs_list");
    df::FileSystem::writeText("test_fs_list/a.txt", "a");
    df::FileSystem::writeText("test_fs_list/b.txt", "b");
    df::FileSystem::writeText("test_fs_list/c.json", "{}");

    auto result = df::FileSystem::listFiles("test_fs_list", ".txt");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 2);

    fs::remove_all("test_fs_list");  // 清理
}

// 20260319 ZJH 测试读取不存在的文件 — 应返回错误
TEST(FileSystemTest, ReadNonExistent) {
    auto result = df::FileSystem::readText("nonexistent_file_xyz.txt");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, df::ErrorCode::FileNotFound);
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cmake --build build/windows-debug --target test_filesystem
```

Expected: 编译失败

- [ ] **Step 3: 实现 FileSystem 模块**

```cpp
// 20260319 ZJH FileSystem 模块 — 跨平台文件操作封装
// 基于 C++17 <filesystem> 和标准 IO，提供 Result 风格错误处理
module;

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <expected>
#include <cstdint>

#include "df_types.h"

export module df.platform.filesystem;

namespace fs = std::filesystem;

export namespace df {

// 20260319 ZJH 文件系统工具类 — 全部静态方法，无状态
class FileSystem {
public:
    // 20260319 ZJH 确保目录存在 — 不存在则递归创建
    static Result<void> ensureDir(const std::string& strPath) {
        std::error_code ec;
        fs::create_directories(strPath, ec);  // 递归创建目录
        if (ec) {
            return std::unexpected(DF_ERROR(InternalError, "Failed to create directory: " + ec.message()));
        }
        return {};
    }

    // 20260319 ZJH 检查路径是否存在
    static bool exists(const std::string& strPath) {
        return fs::exists(strPath);
    }

    // 20260319 ZJH 写入文本文件
    static Result<void> writeText(const std::string& strPath, const std::string& strContent) {
        std::ofstream ofs(strPath);
        if (!ofs.is_open()) {
            return std::unexpected(DF_ERROR(FileNotFound, "Cannot open file for writing: " + strPath));
        }
        ofs << strContent;
        return {};
    }

    // 20260319 ZJH 读取文本文件全部内容
    static Result<std::string> readText(const std::string& strPath) {
        if (!fs::exists(strPath)) {
            return std::unexpected(DF_ERROR(FileNotFound, "File not found: " + strPath));
        }
        std::ifstream ifs(strPath);
        if (!ifs.is_open()) {
            return std::unexpected(DF_ERROR(FileNotFound, "Cannot open file: " + strPath));
        }
        // 20260319 ZJH 使用迭代器一次性读取整个文件
        std::string strContent((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
        return strContent;
    }

    // 20260319 ZJH 写入二进制文件
    static Result<void> writeBinary(const std::string& strPath, const std::vector<uint8_t>& vecData) {
        std::ofstream ofs(strPath, std::ios::binary);
        if (!ofs.is_open()) {
            return std::unexpected(DF_ERROR(FileNotFound, "Cannot open file for writing: " + strPath));
        }
        ofs.write(reinterpret_cast<const char*>(vecData.data()), vecData.size());
        return {};
    }

    // 20260319 ZJH 读取二进制文件全部内容
    static Result<std::vector<uint8_t>> readBinary(const std::string& strPath) {
        if (!fs::exists(strPath)) {
            return std::unexpected(DF_ERROR(FileNotFound, "File not found: " + strPath));
        }
        std::ifstream ifs(strPath, std::ios::binary | std::ios::ate);  // ate 定位到末尾获取大小
        if (!ifs.is_open()) {
            return std::unexpected(DF_ERROR(FileNotFound, "Cannot open file: " + strPath));
        }
        auto nSize = ifs.tellg();  // 获取文件大小
        ifs.seekg(0);              // 回到文件开头

        std::vector<uint8_t> vecData(static_cast<size_t>(nSize));
        ifs.read(reinterpret_cast<char*>(vecData.data()), nSize);
        return vecData;
    }

    // 20260319 ZJH 列举目录中指定扩展名的文件
    // 参数: strDir — 目录路径, strExtension — 扩展名过滤（如 ".txt"），空则列举全部
    static Result<std::vector<std::string>> listFiles(
            const std::string& strDir, const std::string& strExtension = "") {
        if (!fs::exists(strDir) || !fs::is_directory(strDir)) {
            return std::unexpected(DF_ERROR(FileNotFound, "Directory not found: " + strDir));
        }

        std::vector<std::string> vecFiles;
        for (const auto& entry : fs::directory_iterator(strDir)) {
            if (!entry.is_regular_file()) continue;  // 跳过非文件项
            // 20260319 ZJH 扩展名为空则不过滤，否则只保留匹配项
            if (strExtension.empty() || entry.path().extension().string() == strExtension) {
                vecFiles.push_back(entry.path().string());
            }
        }
        return vecFiles;
    }
};

}  // namespace df
```

- [ ] **Step 4: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_filesystem
cd build/windows-debug && ctest -R test_filesystem -V
```

Expected: 6 tests PASSED

- [ ] **Step 5: 提交**

```bash
git add src/platform/df.platform.filesystem.ixx tests/test_filesystem.cpp
git commit -m "feat(platform): add FileSystem module with text/binary IO"
```

---

## Task 5: MemoryPool — 内存池化分配器

**Files:**
- Create: `src/platform/df.platform.memory.ixx`
- Create: `tests/test_memory.cpp`

- [ ] **Step 1: 编写 MemoryPool 测试**

```cpp
// 20260319 ZJH MemoryPool 模块单元测试
#include <gtest/gtest.h>
#include <cstring>
import df.platform.memory;

// 20260319 ZJH 测试基本分配与释放 — 分配的内存应非空
TEST(MemoryPoolTest, AllocateAndDeallocate) {
    df::MemoryPool pool;
    void* pPtr = pool.allocate(1024);
    ASSERT_NE(pPtr, nullptr);
    pool.deallocate(pPtr);
}

// 20260319 ZJH 测试分配的内存可写可读
TEST(MemoryPoolTest, MemoryIsUsable) {
    df::MemoryPool pool;
    auto* pData = static_cast<float*>(pool.allocate(100 * sizeof(float)));
    ASSERT_NE(pData, nullptr);
    // 写入测试数据
    for (int i = 0; i < 100; ++i) {
        pData[i] = static_cast<float>(i);
    }
    // 读取验证
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(pData[i], static_cast<float>(i));
    }
    pool.deallocate(pData);
}

// 20260319 ZJH 测试零大小分配 — 应返回 nullptr
TEST(MemoryPoolTest, ZeroSizeReturnsNull) {
    df::MemoryPool pool;
    void* pPtr = pool.allocate(0);
    EXPECT_EQ(pPtr, nullptr);
}

// 20260319 ZJH 测试多次分配释放 — 验证无内存泄漏（不崩溃即可）
TEST(MemoryPoolTest, MultipleAllocations) {
    df::MemoryPool pool;
    std::vector<void*> vecPtrs;
    for (int i = 0; i < 100; ++i) {
        vecPtrs.push_back(pool.allocate(1024));
        ASSERT_NE(vecPtrs.back(), nullptr);
    }
    for (auto* pPtr : vecPtrs) {
        pool.deallocate(pPtr);
    }
}

// 20260319 ZJH 测试内存池统计 — 已分配字节数应正确
TEST(MemoryPoolTest, Statistics) {
    df::MemoryPool pool;
    EXPECT_EQ(pool.allocatedBytes(), 0);

    void* pPtr = pool.allocate(2048);
    EXPECT_GE(pool.allocatedBytes(), 2048);  // 对齐后可能略大

    pool.deallocate(pPtr);
    EXPECT_EQ(pool.allocatedBytes(), 0);
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cmake --build build/windows-debug --target test_memory
```

Expected: 编译失败

- [ ] **Step 3: 实现 MemoryPool 模块**

```cpp
// 20260319 ZJH MemoryPool 模块 — CPU 内存池化分配器
// 当前为简单封装版本：64 字节对齐分配 + 统计追踪
// 后续 Phase 2 引入真正的空闲链表池化和 GPU 显存池化
module;

#include <cstdlib>
#include <cstddef>
#include <unordered_map>
#include <mutex>
#include <cstdint>

export module df.platform.memory;

export namespace df {

// 20260319 ZJH CPU 内存池 — 64 字节对齐分配 + 分配追踪
// Phase 1 使用简单分配 + 统计；Phase 2 扩展为空闲链表池化 + GPU 显存管理
class MemoryPool {
public:
    MemoryPool() = default;

    // 20260319 ZJH 析构时检查是否有未释放的内存
    ~MemoryPool() = default;

    // 20260319 ZJH 分配指定字节数的对齐内存
    // 参数: nBytes — 请求的字节数
    // 返回: 对齐到 64 字节边界的内存指针，失败或 nBytes==0 返回 nullptr
    void* allocate(size_t nBytes) {
        if (nBytes == 0) return nullptr;  // 零大小直接返回空

        // 20260319 ZJH 向上对齐到 64 字节边界（缓存行对齐，避免 false sharing）
        size_t nAligned = (nBytes + s_nAlignment - 1) & ~(s_nAlignment - 1);

        // 20260319 ZJH 使用平台对齐分配函数
#ifdef _MSC_VER
        void* pPtr = _aligned_malloc(nAligned, s_nAlignment);
#else
        void* pPtr = std::aligned_alloc(s_nAlignment, nAligned);
#endif
        if (pPtr) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_mapAllocations[pPtr] = nAligned;  // 记录分配大小
            m_nAllocatedBytes += nAligned;       // 累加统计
        }
        return pPtr;
    }

    // 20260319 ZJH 释放之前分配的内存
    void deallocate(void* pPtr) {
        if (!pPtr) return;  // 空指针忽略

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_mapAllocations.find(pPtr);
        if (it != m_mapAllocations.end()) {
            m_nAllocatedBytes -= it->second;  // 减去已释放的大小
            m_mapAllocations.erase(it);       // 移除记录
        }

#ifdef _MSC_VER
        _aligned_free(pPtr);
#else
        std::free(pPtr);
#endif
    }

    // 20260319 ZJH 查询当前已分配的总字节数
    size_t allocatedBytes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_nAllocatedBytes;
    }

private:
    static constexpr size_t s_nAlignment = 64;  // 缓存行对齐字节数

    mutable std::mutex m_mutex;                             // 线程安全互斥锁
    std::unordered_map<void*, size_t> m_mapAllocations;     // 分配记录表
    size_t m_nAllocatedBytes = 0;                           // 已分配字节数统计
};

}  // namespace df
```

- [ ] **Step 4: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_memory
cd build/windows-debug && ctest -R test_memory -V
```

Expected: 5 tests PASSED

- [ ] **Step 5: 提交**

```bash
git add src/platform/df.platform.memory.ixx tests/test_memory.cpp
git commit -m "feat(platform): add MemoryPool with aligned allocation and stats"
```

---

## Task 6: ThreadPool — std::jthread 线程池

**Files:**
- Create: `src/platform/df.platform.thread_pool.ixx`
- Create: `tests/test_thread_pool.cpp`

- [ ] **Step 1: 编写 ThreadPool 测试**

```cpp
// 20260319 ZJH ThreadPool 模块单元测试
#include <gtest/gtest.h>
#include <atomic>
#include <numeric>
#include <vector>
import df.platform.thread_pool;

// 20260319 ZJH 测试基本任务提交与执行
TEST(ThreadPoolTest, SubmitAndGetResult) {
    df::ThreadPool pool(2);
    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

// 20260319 ZJH 测试多任务并行执行
TEST(ThreadPoolTest, MultipleTasks) {
    df::ThreadPool pool(4);
    std::vector<std::future<int>> vecFutures;

    for (int i = 0; i < 100; ++i) {
        vecFutures.push_back(pool.submit([i]() { return i * i; }));
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vecFutures[i].get(), i * i);
    }
}

// 20260319 ZJH 测试原子计数 — 验证所有任务都被执行
TEST(ThreadPoolTest, AllTasksExecuted) {
    df::ThreadPool pool(4);
    std::atomic<int> nCounter{0};

    for (int i = 0; i < 1000; ++i) {
        pool.submit([&nCounter]() { nCounter.fetch_add(1); });
    }

    pool.waitAll();  // 等待所有任务完成
    EXPECT_EQ(nCounter.load(), 1000);
}

// 20260319 ZJH 测试线程池大小查询
TEST(ThreadPoolTest, ThreadCount) {
    df::ThreadPool pool(3);
    EXPECT_EQ(pool.threadCount(), 3);
}

// 20260319 ZJH 测试 void 返回值任务
TEST(ThreadPoolTest, VoidTask) {
    df::ThreadPool pool(2);
    bool bExecuted = false;
    auto future = pool.submit([&bExecuted]() { bExecuted = true; });
    future.get();
    EXPECT_TRUE(bExecuted);
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cmake --build build/windows-debug --target test_thread_pool
```

Expected: 编译失败

- [ ] **Step 3: 实现 ThreadPool 模块**

```cpp
// 20260319 ZJH ThreadPool 模块 — std::jthread 线程池
// 提供 submit() 提交任务并返回 future，waitAll() 等待全部完成
module;

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <vector>
#include <atomic>
#include <type_traits>

export module df.platform.thread_pool;

export namespace df {

// 20260319 ZJH 通用线程池 — 固定线程数，任务队列 + 条件变量调度
class ThreadPool {
public:
    // 20260319 ZJH 构造函数 — 启动 nThreads 个工作线程
    explicit ThreadPool(int nThreads)
        : m_bStopping(false), m_nPendingTasks(0) {
        m_vecWorkers.reserve(nThreads);
        for (int i = 0; i < nThreads; ++i) {
            m_vecWorkers.emplace_back([this](std::stop_token stopToken) {
                workerLoop(stopToken);
            });
        }
    }

    // 20260319 ZJH 析构 — 通知所有线程停止并等待退出
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_bStopping = true;
        }
        m_cvTask.notify_all();  // 唤醒所有等待中的工作线程
        // jthread 析构时自动 request_stop + join
    }

    // 20260319 ZJH 提交任务 — 返回 std::future 用于获取结果
    // 参数: func — 可调用对象, args — 参数
    template<typename F, typename... Args>
    auto submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;

        // 20260319 ZJH 将任务打包为 packaged_task 以获取 future
        auto pTask = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...)
        );

        std::future<ReturnType> future = pTask->get_future();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_nPendingTasks.fetch_add(1);  // 待完成任务数 +1
            m_queueTasks.emplace([pTask]() { (*pTask)(); });  // 入队
        }
        m_cvTask.notify_one();  // 唤醒一个工作线程

        return future;
    }

    // 20260319 ZJH 等待所有已提交任务完成
    void waitAll() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cvDone.wait(lock, [this]() {
            return m_nPendingTasks.load() == 0;  // 所有任务完成
        });
    }

    // 20260319 ZJH 获取线程数
    int threadCount() const {
        return static_cast<int>(m_vecWorkers.size());
    }

private:
    // 20260319 ZJH 工作线程循环 — 从队列取任务执行
    void workerLoop(std::stop_token stopToken) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cvTask.wait(lock, [this, &stopToken]() {
                    return m_bStopping || stopToken.stop_requested() || !m_queueTasks.empty();
                });

                // 20260319 ZJH 停止条件：收到停止信号且队列为空
                if ((m_bStopping || stopToken.stop_requested()) && m_queueTasks.empty()) {
                    return;
                }

                task = std::move(m_queueTasks.front());
                m_queueTasks.pop();
            }

            task();  // 执行任务（在锁外执行，避免阻塞其他线程）

            // 20260319 ZJH 任务完成，待完成数 -1，通知 waitAll
            if (m_nPendingTasks.fetch_sub(1) == 1) {
                m_cvDone.notify_all();
            }
        }
    }

    std::vector<std::jthread> m_vecWorkers;        // 工作线程列表
    std::queue<std::function<void()>> m_queueTasks; // 任务队列
    std::mutex m_mutex;                             // 队列互斥锁
    std::condition_variable m_cvTask;               // 新任务通知
    std::condition_variable m_cvDone;               // 任务完成通知
    std::atomic<int> m_nPendingTasks;               // 待完成任务数
    bool m_bStopping;                               // 停止标志
};

}  // namespace df
```

- [ ] **Step 4: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_thread_pool
cd build/windows-debug && ctest -R test_thread_pool -V
```

Expected: 5 tests PASSED

- [ ] **Step 5: 提交**

```bash
git add src/platform/df.platform.thread_pool.ixx tests/test_thread_pool.cpp
git commit -m "feat(platform): add ThreadPool with jthread and future-based API"
```

---

## Task 7: Database — SQLite RAII 封装

**Files:**
- Create: `src/platform/df.platform.database.ixx`
- Create: `tests/test_database.cpp`

- [ ] **Step 1: 编写 Database 测试**

```cpp
// 20260319 ZJH Database 模块单元测试
#include <gtest/gtest.h>
#include <filesystem>
import df.platform.database;

namespace fs = std::filesystem;

// 20260319 ZJH 测试创建内存数据库
TEST(DatabaseTest, OpenInMemory) {
    auto result = df::Database::open(":memory:");
    ASSERT_TRUE(result.has_value());
}

// 20260319 ZJH 测试建表和插入
TEST(DatabaseTest, CreateTableAndInsert) {
    auto result = df::Database::open(":memory:");
    ASSERT_TRUE(result.has_value());
    auto& db = result.value();

    auto execResult = db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value REAL)");
    EXPECT_TRUE(execResult.has_value());

    execResult = db.execute("INSERT INTO test (name, value) VALUES ('hello', 3.14)");
    EXPECT_TRUE(execResult.has_value());
}

// 20260319 ZJH 测试查询
TEST(DatabaseTest, Query) {
    auto result = df::Database::open(":memory:");
    ASSERT_TRUE(result.has_value());
    auto& db = result.value();

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)");
    db.execute("INSERT INTO items (name) VALUES ('alpha')");
    db.execute("INSERT INTO items (name) VALUES ('beta')");

    auto queryResult = db.query("SELECT name FROM items ORDER BY name");
    ASSERT_TRUE(queryResult.has_value());
    auto& rows = queryResult.value();
    ASSERT_EQ(rows.size(), 2);
    EXPECT_EQ(rows[0]["name"], "alpha");
    EXPECT_EQ(rows[1]["name"], "beta");
}

// 20260319 ZJH 测试文件数据库
TEST(DatabaseTest, FileDatabase) {
    const std::string strPath = "test_db_temp.db";

    {
        auto result = df::Database::open(strPath);
        ASSERT_TRUE(result.has_value());
        auto& db = result.value();
        db.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)");
        db.execute("INSERT INTO data (val) VALUES (42)");
    }  // RAII: 析构时自动关闭

    // 重新打开验证数据持久化
    {
        auto result = df::Database::open(strPath);
        ASSERT_TRUE(result.has_value());
        auto& db = result.value();
        auto queryResult = db.query("SELECT val FROM data");
        ASSERT_TRUE(queryResult.has_value());
        EXPECT_EQ(queryResult.value()[0]["val"], "42");
    }

    fs::remove(strPath);  // 清理
}

// 20260319 ZJH 测试无效 SQL — 应返回错误
TEST(DatabaseTest, InvalidSQL) {
    auto result = df::Database::open(":memory:");
    ASSERT_TRUE(result.has_value());
    auto& db = result.value();

    auto execResult = db.execute("INVALID SQL STATEMENT");
    EXPECT_FALSE(execResult.has_value());
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cmake --build build/windows-debug --target test_database
```

Expected: 编译失败

- [ ] **Step 3: 实现 Database 模块**

```cpp
// 20260319 ZJH Database 模块 — SQLite3 RAII 封装
// 提供 open/execute/query 接口，自动管理连接生命周期
module;

#include <sqlite3.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <expected>
#include <memory>

#include "df_types.h"

export module df.platform.database;

export namespace df {

// 20260319 ZJH 查询结果行类型 — 列名 → 值的映射（值统一为 string）
using Row = std::unordered_map<std::string, std::string>;

// 20260319 ZJH SQLite 数据库 RAII 封装
class Database {
public:
    // 20260319 ZJH 禁止拷贝，允许移动
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&& other) noexcept : m_pDb(other.m_pDb) {
        other.m_pDb = nullptr;
    }
    Database& operator=(Database&& other) noexcept {
        if (this != &other) {
            close();
            m_pDb = other.m_pDb;
            other.m_pDb = nullptr;
        }
        return *this;
    }

    // 20260319 ZJH 析构 — 自动关闭数据库连接
    ~Database() { close(); }

    // 20260319 ZJH 打开数据库（静态工厂方法）
    // 参数: strPath — 数据库文件路径，":memory:" 为内存数据库
    static Result<Database> open(const std::string& strPath) {
        Database db;
        int nResult = sqlite3_open(strPath.c_str(), &db.m_pDb);
        if (nResult != SQLITE_OK) {
            std::string strErr = db.m_pDb ? sqlite3_errmsg(db.m_pDb) : "Unknown error";
            return std::unexpected(DF_ERROR(InternalError, "Failed to open database: " + strErr));
        }
        // 20260319 ZJH 启用 WAL 模式 — 提高并发读写性能
        sqlite3_exec(db.m_pDb, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
        return db;
    }

    // 20260319 ZJH 执行非查询 SQL（CREATE / INSERT / UPDATE / DELETE）
    Result<void> execute(const std::string& strSQL) {
        char* pErrMsg = nullptr;
        int nResult = sqlite3_exec(m_pDb, strSQL.c_str(), nullptr, nullptr, &pErrMsg);
        if (nResult != SQLITE_OK) {
            std::string strErr = pErrMsg ? pErrMsg : "Unknown error";
            sqlite3_free(pErrMsg);  // 释放 SQLite 分配的错误消息
            return std::unexpected(DF_ERROR(InternalError, "SQL error: " + strErr));
        }
        return {};
    }

    // 20260319 ZJH 执行查询 SQL — 返回行列表
    Result<std::vector<Row>> query(const std::string& strSQL) {
        std::vector<Row> vecRows;
        char* pErrMsg = nullptr;

        // 20260319 ZJH 使用 sqlite3_exec 回调收集结果
        int nResult = sqlite3_exec(m_pDb, strSQL.c_str(),
            [](void* pUserData, int nCols, char** ppValues, char** ppNames) -> int {
                auto* pRows = static_cast<std::vector<Row>*>(pUserData);
                Row row;
                for (int i = 0; i < nCols; ++i) {
                    // 20260319 ZJH NULL 值存为空字符串
                    row[ppNames[i]] = ppValues[i] ? ppValues[i] : "";
                }
                pRows->push_back(std::move(row));
                return 0;  // 返回 0 继续遍历
            },
            &vecRows, &pErrMsg);

        if (nResult != SQLITE_OK) {
            std::string strErr = pErrMsg ? pErrMsg : "Unknown error";
            sqlite3_free(pErrMsg);
            return std::unexpected(DF_ERROR(InternalError, "Query error: " + strErr));
        }
        return vecRows;
    }

private:
    // 20260319 ZJH 私有构造 — 只能通过 open() 工厂方法创建
    Database() : m_pDb(nullptr) {}

    // 20260319 ZJH 关闭数据库连接
    void close() {
        if (m_pDb) {
            sqlite3_close(m_pDb);
            m_pDb = nullptr;
        }
    }

    sqlite3* m_pDb;  // SQLite 数据库连接句柄
};

}  // namespace df
```

- [ ] **Step 4: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_database
cd build/windows-debug && ctest -R test_database -V
```

Expected: 5 tests PASSED

- [ ] **Step 5: 提交**

```bash
git add src/platform/df.platform.database.ixx tests/test_database.cpp
git commit -m "feat(platform): add Database module with SQLite RAII wrapper"
```

---

## Task 8: 全量构建验证 + 默认配置

**Files:**
- Create: `config/default_config.json`

- [ ] **Step 1: 创建默认配置文件**

```json
{
  "app": {
    "name": "DeepForge",
    "version": "0.1.0"
  },
  "training": {
    "default_batch_size": 16,
    "default_learning_rate": 0.001,
    "default_epochs": 100,
    "default_optimizer": "adam",
    "checkpoint_dir": "data/models/checkpoints"
  },
  "inference": {
    "default_device": "cuda",
    "fallback_device": "cpu"
  },
  "data": {
    "dataset_dir": "data/datasets",
    "model_dir": "data/models",
    "database_path": "data/deepforge.db"
  },
  "ui": {
    "theme": "dark",
    "font_size": 16
  }
}
```

- [ ] **Step 2: 全量构建所有目标**

```bash
cmake --preset windows-debug
cmake --build build/windows-debug
```

Expected: 全部编译成功，无错误

- [ ] **Step 3: 运行全部测试**

```bash
cd build/windows-debug && ctest --output-on-failure
```

Expected: 所有测试通过（Logger 4 + Config 5 + FileSystem 6 + Memory 5 + ThreadPool 5 + Database 5 = 30 tests）

- [ ] **Step 4: 提交配置文件**

```bash
git add config/default_config.json
git commit -m "feat: add default configuration file"
```

---

## Summary

| Task | 模块 | 测试数 | 关键验证点 |
|------|------|--------|-----------|
| 1 | 项目骨架 | — | CMake configure 成功 |
| 2 | Logger | 4 | 多级别日志 + 文件输出 |
| 3 | Config | 5 | JSON 读写 + 类型安全 + 错误处理 |
| 4 | FileSystem | 6 | 文本/二进制 IO + 目录操作 |
| 5 | MemoryPool | 5 | 对齐分配 + 统计 + 线程安全 |
| 6 | ThreadPool | 5 | 并发任务 + future + waitAll |
| 7 | Database | 5 | CRUD + RAII + 持久化 |
| 8 | 全量验证 | 30 | 全部编译通过 + 全部测试通过 |
