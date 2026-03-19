// 20260319 ZJH Config 模块单元测试
// 覆盖：默认构造、键值存取、默认值、文件保存/加载、错误路径
#include <gtest/gtest.h>
#include <filesystem>
import df.platform.config;

// 20260319 ZJH std::filesystem 别名，用于测试后的临时文件清理
namespace fs = std::filesystem;

// 20260319 ZJH 测试默认构造 — 空配置应可正常创建，且不含任何键
TEST(ConfigTest, DefaultConstruct) {
    df::Config config;  // 默认构造，m_jsonData 应为空 JSON object
    EXPECT_FALSE(config.has("nonexistent"));  // 空配置中不应存在任何键
}

// 20260319 ZJH 测试键值存取 — 支持 string / int / double / bool 四种常用类型
TEST(ConfigTest, SetAndGet) {
    df::Config config;

    // 20260319 ZJH 分别设置四种类型的配置项
    config.set("name", std::string("DeepForge"));  // 字符串类型
    config.set("version", 1);                       // 整数类型
    config.set("learning_rate", 0.001);             // 浮点类型
    config.set("enable_cuda", true);                // 布尔类型

    // 20260319 ZJH 逐一验证取回值与设置值完全一致
    EXPECT_EQ(config.get<std::string>("name"), "DeepForge");
    EXPECT_EQ(config.get<int>("version"), 1);
    EXPECT_DOUBLE_EQ(config.get<double>("learning_rate"), 0.001);
    EXPECT_TRUE(config.get<bool>("enable_cuda"));
}

// 20260319 ZJH 测试默认值 — 键不存在时应返回调用方提供的默认值，不抛异常
TEST(ConfigTest, GetWithDefault) {
    df::Config config;  // 空配置，所有键均不存在

    // 20260319 ZJH 整数类型默认值测试
    EXPECT_EQ(config.get<int>("missing", 42), 42);
    // 20260319 ZJH 字符串类型默认值测试
    EXPECT_EQ(config.get<std::string>("missing", std::string("default")), "default");
}

// 20260319 ZJH 测试文件保存和加载 — 写入后读回值应与原始值完全一致
TEST(ConfigTest, SaveAndLoad) {
    const std::string strPath = "test_config_temp.json";  // 临时文件路径

    // 20260319 ZJH 第一个作用域：创建配置并保存到文件
    {
        df::Config config;
        config.set("batch_size", 16);                       // 设置整数配置项
        config.set("model", std::string("resnet18"));       // 设置字符串配置项
        auto result = config.save(strPath);
        EXPECT_TRUE(result.has_value());  // 保存应成功
    }

    // 20260319 ZJH 第二个作用域：重新从文件加载，验证数据持久化正确
    {
        auto result = df::Config::load(strPath);
        ASSERT_TRUE(result.has_value());  // 加载应成功，ASSERT 失败则立即中止本测试
        auto& config = result.value();
        EXPECT_EQ(config.get<int>("batch_size"), 16);              // 整数值应与保存前一致
        EXPECT_EQ(config.get<std::string>("model"), "resnet18");   // 字符串值应与保存前一致
    }

    // 20260319 ZJH 清理测试产生的临时文件，避免污染工作目录
    fs::remove(strPath);
}

// 20260319 ZJH 测试加载不存在的文件 — 应返回 FileNotFound 错误，而非抛出异常
TEST(ConfigTest, LoadNonExistent) {
    auto result = df::Config::load("nonexistent_file.json");
    EXPECT_FALSE(result.has_value());  // 加载应失败
    // 20260319 ZJH 验证错误码为 FileNotFound，确保错误类型分类正确
    EXPECT_EQ(result.error().code, df::ErrorCode::FileNotFound);
}
