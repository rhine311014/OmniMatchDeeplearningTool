// 20260319 ZJH Database 模块单元测试
// 覆盖：内存数据库打开、建表插入、查询排序、文件数据库持久化、非法 SQL 错误返回

#include <gtest/gtest.h>
#include <filesystem>   // 20260319 ZJH 用于测试结束后清理临时数据库文件

// 20260319 ZJH 导入 df.platform.database 模块，获取 df::Database 和 df::Row 定义
import df.platform.database;

// =========================================================================
// 20260319 ZJH 测试 1 — 打开内存数据库
// 目的：验证 ":memory:" 路径可成功打开，open() 返回有效的 Result<Database>
// =========================================================================
TEST(DatabaseTest, OpenInMemory) {
    // 20260319 ZJH 以内存模式打开 SQLite 数据库，不产生磁盘文件
    auto result = df::Database::open(":memory:");

    // 20260319 ZJH has_value() 为 true 表示打开成功，无错误
    EXPECT_TRUE(result.has_value())
        << "Expected in-memory database to open successfully, got error: "
        << (result.has_value() ? "" : result.error().strMessage);
}

// =========================================================================
// 20260319 ZJH 测试 2 — 建表并插入数据
// 目的：验证 execute() 能成功执行 CREATE TABLE 和 INSERT 语句，均返回成功态
// =========================================================================
TEST(DatabaseTest, CreateTableAndInsert) {
    // 20260319 ZJH 打开内存数据库，ASSERT 失败则终止本用例（避免后续对空对象操作）
    auto dbResult = df::Database::open(":memory:");
    ASSERT_TRUE(dbResult.has_value()) << dbResult.error().strMessage;

    auto& db = dbResult.value();  // 取出 Database 对象的引用，避免额外拷贝（已禁拷贝，只能引用）

    // 20260319 ZJH 执行 CREATE TABLE 语句，建立测试用表
    auto createResult = db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);");
    EXPECT_TRUE(createResult.has_value())
        << "CREATE TABLE failed: " << (createResult.has_value() ? "" : createResult.error().strMessage);

    // 20260319 ZJH 执行 INSERT 语句，插入一条测试记录
    auto insertResult = db.execute(
        "INSERT INTO items (id, name) VALUES (1, 'apple');");
    EXPECT_TRUE(insertResult.has_value())
        << "INSERT failed: " << (insertResult.has_value() ? "" : insertResult.error().strMessage);
}

// =========================================================================
// 20260319 ZJH 测试 3 — 查询并验证结果
// 目的：插入 2 行数据后，SELECT ORDER BY 查询应返回正确的行数和列值
// =========================================================================
TEST(DatabaseTest, Query) {
    // 20260319 ZJH 打开内存数据库
    auto dbResult = df::Database::open(":memory:");
    ASSERT_TRUE(dbResult.has_value()) << dbResult.error().strMessage;

    auto& db = dbResult.value();  // 取 Database 引用

    // 20260319 ZJH 建表
    ASSERT_TRUE(db.execute(
        "CREATE TABLE fruits (id INTEGER PRIMARY KEY, name TEXT);").has_value());

    // 20260319 ZJH 插入第一行
    ASSERT_TRUE(db.execute(
        "INSERT INTO fruits (id, name) VALUES (1, 'banana');").has_value());

    // 20260319 ZJH 插入第二行（id 更小，ORDER BY id 应排在前面）
    ASSERT_TRUE(db.execute(
        "INSERT INTO fruits (id, name) VALUES (2, 'cherry');").has_value());

    // 20260319 ZJH 按 id 升序查询全部记录
    auto queryResult = db.query("SELECT id, name FROM fruits ORDER BY id ASC;");
    ASSERT_TRUE(queryResult.has_value())
        << "Query failed: " << (queryResult.has_value() ? "" : queryResult.error().strMessage);

    const auto& rows = queryResult.value();  // 取出结果集引用

    // 20260319 ZJH 验证返回行数为 2（与插入行数一致）
    ASSERT_EQ(rows.size(), 2u) << "Expected 2 rows, got " << rows.size();

    // 20260319 ZJH 验证第一行（id=1，name="banana"）
    EXPECT_EQ(rows[0].at("id"), "1");
    EXPECT_EQ(rows[0].at("name"), "banana");

    // 20260319 ZJH 验证第二行（id=2，name="cherry"）
    EXPECT_EQ(rows[1].at("id"), "2");
    EXPECT_EQ(rows[1].at("name"), "cherry");
}

// =========================================================================
// 20260319 ZJH 测试 4 — 文件数据库持久化
// 目的：打开文件数据库、插入数据、RAII 关闭、重新打开后数据仍存在
// =========================================================================
TEST(DatabaseTest, FileDatabase) {
    // 20260319 ZJH 使用临时目录下的测试数据库文件，避免污染工作目录
    std::string strDbPath = (std::filesystem::temp_directory_path()
        / "df_test_persistence.db").string();

    // 20260319 ZJH 若上次测试异常退出遗留文件，先删除确保干净起点
    std::filesystem::remove(strDbPath);

    // 20260319 ZJH 第一阶段：打开文件数据库、建表、插入数据
    {
        // 20260319 ZJH 在作用域内创建 Database 对象，作用域结束时自动析构（RAII）
        auto dbResult = df::Database::open(strDbPath);
        ASSERT_TRUE(dbResult.has_value()) << dbResult.error().strMessage;

        auto& db = dbResult.value();  // 取引用

        // 20260319 ZJH 建表并插入持久化测试数据
        ASSERT_TRUE(db.execute(
            "CREATE TABLE IF NOT EXISTS persist (key TEXT, value TEXT);").has_value());
        ASSERT_TRUE(db.execute(
            "INSERT INTO persist VALUES ('hello', 'world');").has_value());
        // 20260319 ZJH 作用域结束，db 析构，sqlite3_close 被调用，数据落盘
    }

    // 20260319 ZJH 第二阶段：重新打开同一文件，验证数据仍然存在（持久化有效）
    {
        auto dbResult2 = df::Database::open(strDbPath);
        ASSERT_TRUE(dbResult2.has_value()) << dbResult2.error().strMessage;

        auto& db2 = dbResult2.value();  // 取引用

        // 20260319 ZJH 查询上次写入的数据
        auto queryResult = db2.query("SELECT key, value FROM persist;");
        ASSERT_TRUE(queryResult.has_value())
            << "Query after reopen failed: " << queryResult.error().strMessage;

        const auto& rows = queryResult.value();  // 取出结果集引用

        // 20260319 ZJH 验证数据行存在且值正确
        ASSERT_EQ(rows.size(), 1u) << "Expected 1 row after reopen, got " << rows.size();
        EXPECT_EQ(rows[0].at("key"), "hello");
        EXPECT_EQ(rows[0].at("value"), "world");
    }

    // 20260319 ZJH 测试结束后清理临时文件（包括 WAL 日志文件 .wal 和共享内存文件 .shm）
    std::filesystem::remove(strDbPath);                      // 主数据库文件
    std::filesystem::remove(strDbPath + "-wal");             // WAL 日志文件
    std::filesystem::remove(strDbPath + "-shm");             // 共享内存映射文件
}

// =========================================================================
// 20260319 ZJH 测试 5 — 非法 SQL 返回错误
// 目的：执行语法错误的 SQL 语句，execute() 应返回失败态（has_value() == false）
// =========================================================================
TEST(DatabaseTest, InvalidSQL) {
    // 20260319 ZJH 打开内存数据库
    auto dbResult = df::Database::open(":memory:");
    ASSERT_TRUE(dbResult.has_value()) << dbResult.error().strMessage;

    auto& db = dbResult.value();  // 取 Database 引用

    // 20260319 ZJH 执行语法错误的 SQL（关键字拼错，SQLite 无法解析）
    auto result = db.execute("THIS IS NOT VALID SQL !!!;");

    // 20260319 ZJH 期望 execute 返回错误态（has_value() == false）
    EXPECT_FALSE(result.has_value())
        << "Expected error for invalid SQL, but got success";

    // 20260319 ZJH 确认错误信息字段非空（说明 SQLite 返回了有效的错误描述）
    if (!result.has_value()) {
        EXPECT_FALSE(result.error().strMessage.empty())
            << "Error message should not be empty for invalid SQL";
    }
}
