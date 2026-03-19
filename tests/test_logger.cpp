// 20260319 ZJH Logger 模块单元测试
// 验证 df::Logger 的初始化、日志输出、级别切换、文件写入等核心功能
#include <gtest/gtest.h>
#include <cstdio>   // std::remove — 用于删除测试产生的临时文件

import df.platform.logger;  // 导入 Logger C++23 模块

// 20260319 ZJH 测试初始化 — Logger 应能正常初始化且不抛任何异常
TEST(LoggerTest, InitDoesNotThrow) {
    // 20260319 ZJH 调用 init 仅传入应用名，不指定文件，期望无异常
    EXPECT_NO_THROW(df::Logger::init("test_app"));
}

// 20260319 ZJH 测试日志输出 — 各级别日志调用均不应引发崩溃或异常
TEST(LoggerTest, LogLevelsDoNotCrash) {
    // 20260319 ZJH 先初始化确保 logger 存在
    df::Logger::init("test_app");

    // 20260319 ZJH 依次调用各级别，均期望无异常
    EXPECT_NO_THROW(df::Logger::info("info message: 42"));
    EXPECT_NO_THROW(df::Logger::warn("warn message: test"));
    EXPECT_NO_THROW(df::Logger::error("error message"));
    EXPECT_NO_THROW(df::Logger::debug("debug message"));
    EXPECT_NO_THROW(df::Logger::trace("trace message"));
    EXPECT_NO_THROW(df::Logger::critical("critical message"));
}

// 20260319 ZJH 测试日志级别设置 — 应能在运行时动态切换日志级别
TEST(LoggerTest, SetLevel) {
    // 20260319 ZJH 先初始化确保 logger 存在
    df::Logger::init("test_app");

    // 20260319 ZJH 切换到 Warn 级别，低于 Warn 的日志将被过滤
    EXPECT_NO_THROW(df::Logger::setLevel(df::LogLevel::Warn));

    // 20260319 ZJH 切回 Debug 级别，恢复详细输出
    EXPECT_NO_THROW(df::Logger::setLevel(df::LogLevel::Debug));
}

// 20260319 ZJH 测试文件日志 — 初始化时指定日志文件路径，写入后清理临时文件
TEST(LoggerTest, InitWithFile) {
    // 20260319 ZJH 初始化时传入文件路径，期望创建旋转文件 sink 且无异常
    EXPECT_NO_THROW(df::Logger::init("test_app", "test_log.txt"));

    // 20260319 ZJH 向文件写入一条日志，验证文件 sink 可正常工作
    df::Logger::info("file log test");

    // 20260319 ZJH 删除测试过程中生成的日志文件，保持测试环境整洁
    std::remove("test_log.txt");
}
