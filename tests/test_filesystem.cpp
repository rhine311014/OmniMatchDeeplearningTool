// 20260319 ZJH FileSystem 模块单元测试
// 覆盖：目录创建、文本读写、二进制读写、存在性检查、文件枚举、不存在文件读取错误
#include <gtest/gtest.h>
#include <filesystem>   // 测试中用于清理临时文件/目录
#include <cstdint>      // uint8_t
import om.platform.filesystem;

// 20260319 ZJH std::filesystem 别名，简化测试清理代码
namespace fs = std::filesystem;

// 20260319 ZJH 测试基础目录 — 所有临时测试文件均放在此子目录内，便于统一清理
static const std::string s_strTestRoot = "test_filesystem_temp";

// -----------------------------------------------------------------------
// 20260319 ZJH 测试 ensureDir — 递归创建多级目录，创建后应可通过 exists() 验证
// -----------------------------------------------------------------------
TEST(FileSystemTest, EnsureDir) {
    // 20260319 ZJH 构造多级嵌套路径，测试递归创建能力
    std::string strNestedDir = s_strTestRoot + "/level1/level2/level3";

    // 20260319 ZJH 调用 ensureDir，期望成功创建全部中间目录
    auto result = om::FileSystem::ensureDir(strNestedDir);
    EXPECT_TRUE(result.has_value()) << "ensureDir 应成功创建多级嵌套目录";

    // 20260319 ZJH 用 std::filesystem::exists 独立验证目录确实被创建
    EXPECT_TRUE(fs::exists(strNestedDir)) << "目录应在创建后存在于文件系统中";

    // 20260319 ZJH 再次调用 ensureDir 验证幂等性 — 目录已存在时不应返回错误
    auto result2 = om::FileSystem::ensureDir(strNestedDir);
    EXPECT_TRUE(result2.has_value()) << "目录已存在时 ensureDir 应幂等返回成功";

    // 20260319 ZJH 清理测试产生的临时目录树，避免污染工作目录
    fs::remove_all(s_strTestRoot);
}

// -----------------------------------------------------------------------
// 20260319 ZJH 测试文本读写 — 写入字符串后读回，内容应完全一致
// -----------------------------------------------------------------------
TEST(FileSystemTest, ReadWriteText) {
    // 20260319 ZJH 先确保父目录存在（测试隔离，每个测试独立创建所需目录）
    om::FileSystem::ensureDir(s_strTestRoot);

    std::string strFilePath = s_strTestRoot + "/hello.txt";  // 临时文本文件路径
    // 20260319 ZJH 包含换行和特殊字符，测试文本内容的完整性保留
    std::string strContent = "Hello, OmniMatch!\n第二行内容\n特殊字符: <>&\"";

    // 20260319 ZJH 写入文本文件
    auto writeResult = om::FileSystem::writeText(strFilePath, strContent);
    ASSERT_TRUE(writeResult.has_value()) << "writeText 应成功写入文件";

    // 20260319 ZJH 读回文本内容，并与原始字符串逐字节比较
    auto readResult = om::FileSystem::readText(strFilePath);
    ASSERT_TRUE(readResult.has_value()) << "readText 应成功读取已存在的文件";
    EXPECT_EQ(readResult.value(), strContent) << "读回内容应与写入内容完全一致";

    // 20260319 ZJH 清理临时文件
    fs::remove_all(s_strTestRoot);
}

// -----------------------------------------------------------------------
// 20260319 ZJH 测试二进制读写 — 写入字节数组后读回，内容和长度应完全一致
// -----------------------------------------------------------------------
TEST(FileSystemTest, ReadWriteBinary) {
    // 20260319 ZJH 准备父目录
    om::FileSystem::ensureDir(s_strTestRoot);

    std::string strFilePath = s_strTestRoot + "/data.bin";  // 临时二进制文件路径

    // 20260319 ZJH 构造包含各种边界值的测试字节序列（含 0x00、0xFF 等特殊字节）
    std::vector<uint8_t> vecOriginal = {
        0x00, 0x01, 0x7F, 0x80, 0xFF,  // 边界值和中间值
        0xDE, 0xAD, 0xBE, 0xEF,        // 常见魔数
        0x0A, 0x0D                      // 换行符（检验二进制模式是否正确，不被转换）
    };

    // 20260319 ZJH 写入二进制数据
    auto writeResult = om::FileSystem::writeBinary(strFilePath, vecOriginal);
    ASSERT_TRUE(writeResult.has_value()) << "writeBinary 应成功写入二进制文件";

    // 20260319 ZJH 读回二进制数据
    auto readResult = om::FileSystem::readBinary(strFilePath);
    ASSERT_TRUE(readResult.has_value()) << "readBinary 应成功读取已存在的二进制文件";

    const auto& vecRead = readResult.value();
    // 20260319 ZJH 首先验证字节数相同，再逐字节比较
    ASSERT_EQ(vecRead.size(), vecOriginal.size()) << "读回字节数应与写入字节数相同";
    EXPECT_EQ(vecRead, vecOriginal) << "读回字节序列应与写入字节序列完全一致";

    // 20260319 ZJH 清理临时文件
    fs::remove_all(s_strTestRoot);
}

// -----------------------------------------------------------------------
// 20260319 ZJH 测试 exists — 分别检查不存在的路径和已存在的文件
// -----------------------------------------------------------------------
TEST(FileSystemTest, Exists) {
    // 20260319 ZJH 不存在的路径应返回 false
    EXPECT_FALSE(om::FileSystem::exists("nonexistent_path_xyz_12345"))
        << "不存在的路径应返回 false";

    // 20260319 ZJH 创建一个真实文件，验证 exists() 返回 true
    om::FileSystem::ensureDir(s_strTestRoot);
    std::string strFilePath = s_strTestRoot + "/exist_check.txt";
    om::FileSystem::writeText(strFilePath, "exists test");

    // 20260319 ZJH 已存在的文件路径应返回 true
    EXPECT_TRUE(om::FileSystem::exists(strFilePath))
        << "已创建的文件路径应返回 true";

    // 20260319 ZJH 已存在的目录路径也应返回 true
    EXPECT_TRUE(om::FileSystem::exists(s_strTestRoot))
        << "已创建的目录路径应返回 true";

    // 20260319 ZJH 清理临时文件
    fs::remove_all(s_strTestRoot);
}

// -----------------------------------------------------------------------
// 20260319 ZJH 测试 listFiles — 创建混合扩展名文件后按 .txt 过滤，验证结果准确
// -----------------------------------------------------------------------
TEST(FileSystemTest, ListFiles) {
    // 20260319 ZJH 创建测试目录并在其中创建多个不同扩展名的文件
    std::string strListDir = s_strTestRoot + "/list_test";
    om::FileSystem::ensureDir(strListDir);

    // 20260319 ZJH 创建 3 个 .txt 文件和 2 个 .bin 文件，用于验证扩展名过滤
    om::FileSystem::writeText(strListDir + "/file1.txt", "text file 1");
    om::FileSystem::writeText(strListDir + "/file2.txt", "text file 2");
    om::FileSystem::writeText(strListDir + "/file3.txt", "text file 3");
    om::FileSystem::writeBinary(strListDir + "/data1.bin", {0x01, 0x02});
    om::FileSystem::writeBinary(strListDir + "/data2.bin", {0x03, 0x04});

    // 20260319 ZJH 按 .txt 扩展名过滤，应只返回 3 个文件
    auto resultTxt = om::FileSystem::listFiles(strListDir, ".txt");
    ASSERT_TRUE(resultTxt.has_value()) << "listFiles 应成功枚举目录";
    EXPECT_EQ(resultTxt.value().size(), 3u) << "过滤 .txt 后应有 3 个文件";

    // 20260319 ZJH 不指定扩展名（空字符串），应返回全部 5 个文件
    auto resultAll = om::FileSystem::listFiles(strListDir, "");
    ASSERT_TRUE(resultAll.has_value()) << "不带过滤的 listFiles 应成功";
    EXPECT_EQ(resultAll.value().size(), 5u) << "不过滤时应返回全部 5 个文件";

    // 20260319 ZJH 过滤不存在的扩展名，应返回空列表（而非错误）
    auto resultNone = om::FileSystem::listFiles(strListDir, ".xyz");
    ASSERT_TRUE(resultNone.has_value()) << "无匹配文件时 listFiles 应返回空列表而非错误";
    EXPECT_EQ(resultNone.value().size(), 0u) << "无匹配扩展名时列表应为空";

    // 20260319 ZJH 清理临时目录
    fs::remove_all(s_strTestRoot);
}

// -----------------------------------------------------------------------
// 20260319 ZJH 测试读取不存在的文件 — 应返回 FileNotFound 错误而非抛出异常
// -----------------------------------------------------------------------
TEST(FileSystemTest, ReadNonExistent) {
    // 20260319 ZJH 尝试读取一个确认不存在的文本文件路径
    auto textResult = om::FileSystem::readText("definitely_nonexistent_file_abc.txt");
    // 20260319 ZJH 期望返回失败（has_value() == false）
    EXPECT_FALSE(textResult.has_value()) << "读取不存在的文件应返回错误";
    // 20260319 ZJH 错误码应为 FileNotFound，确保语义正确
    EXPECT_EQ(textResult.error().code, om::ErrorCode::FileNotFound)
        << "错误码应为 FileNotFound";

    // 20260319 ZJH 同样验证 readBinary 对不存在文件的错误处理
    auto binResult = om::FileSystem::readBinary("definitely_nonexistent_file_abc.bin");
    EXPECT_FALSE(binResult.has_value()) << "读取不存在的二进制文件应返回错误";
    EXPECT_EQ(binResult.error().code, om::ErrorCode::FileNotFound)
        << "二进制读取错误码应为 FileNotFound";
}
