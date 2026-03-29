// 20260319 ZJH Database 模块 — SQLite RAII 封装
// 提供内存数据库和文件数据库的安全打开、执行和查询接口
module;

// 20260319 ZJH 全局模块片段：所有 #include 必须放在此处，位于 export module 声明之前
#include <sqlite3.h>    // SQLite C 接口，所有 sqlite3_* 函数定义在此
#include <string>       // std::string — SQL 语句及列名存储
#include <vector>       // std::vector<Row> — 查询结果集容器
#include <unordered_map>// std::unordered_map — 单行数据（列名 -> 值）
#include <expected>     // std::expected / std::unexpected — C++23 错误传播
#include <utility>      // std::move — 移动语义支持

// 20260319 ZJH 引入公共类型定义（ErrorCode / Error / Result<T> / OM_ERROR 宏）
#include "om_types.h"

export module om.platform.database;

export namespace om {

// 20260319 ZJH Row 类型别名 — 表示查询结果中的一行，列名映射到字符串值
// 使用 unordered_map 而非 map，O(1) 平均访问速度更适合读多的查询场景
using Row = std::unordered_map<std::string, std::string>;

// 20260319 ZJH Database — SQLite 数据库的 RAII 封装类
// 职责：持有 sqlite3* 句柄，析构时自动关闭；仅允许移动，禁止拷贝，防止双重关闭
class Database {
public:
    // 20260319 ZJH 禁止拷贝构造 — sqlite3* 不可共享，防止析构时双重 sqlite3_close
    Database(const Database&) = delete;

    // 20260319 ZJH 禁止拷贝赋值 — 同上，防止资源泄漏或双重关闭
    Database& operator=(const Database&) = delete;

    // 20260319 ZJH 移动构造 — 从 other 接管 sqlite3* 句柄，other 置空以防双重关闭
    // 参数: other — 被移动的 Database 对象（移动后 other.m_pDb == nullptr）
    Database(Database&& other) noexcept
        : m_pDb(other.m_pDb)  // 接管 other 的原始句柄
    {
        other.m_pDb = nullptr;  // 将 other 的句柄置空，析构时不会重复 close
    }

    // 20260319 ZJH 移动赋值 — 先释放自身已有句柄，再接管 other 的句柄
    // 参数: other — 被移动的 Database 对象
    // 返回: *this — 支持链式赋值
    Database& operator=(Database&& other) noexcept {
        // 20260319 ZJH 自赋值检查：若 this == &other 则直接返回，避免误关闭自身句柄
        if (this != &other) {
            close();              // 先关闭自身现有句柄（若非 nullptr）
            m_pDb = other.m_pDb;  // 接管 other 的句柄
            other.m_pDb = nullptr;// 将 other 置空，防止 other 析构时重复 close
        }
        return *this;  // 返回自身引用，支持链式赋值表达式
    }

    // 20260319 ZJH 析构函数 — RAII 核心：对象销毁时自动关闭数据库连接
    ~Database() {
        close();  // 若 m_pDb 非 nullptr 则调用 sqlite3_close 释放资源
    }

    // 20260319 ZJH 静态工厂方法 — 打开（或创建）SQLite 数据库
    // 参数: strPath — 数据库文件路径；":memory:" 表示内存数据库（进程结束即消失）
    // 返回: Result<Database> — 成功时持有已打开的 Database 对象，失败时携带错误信息
    static Result<Database> open(const std::string& strPath) {
        Database db;  // 创建空 Database 对象（m_pDb == nullptr）

        // 20260319 ZJH sqlite3_open 以读写模式打开数据库；文件不存在时自动创建
        // SQLITE_OK == 0 表示成功，其余值表示各类错误
        int nRet = sqlite3_open(strPath.c_str(), &db.m_pDb);
        if (nRet != SQLITE_OK) {
            // 20260319 ZJH 从 sqlite3 对象取出可读错误信息（UTF-8）
            std::string strErr = sqlite3_errmsg(db.m_pDb);
            // 20260319 ZJH 关闭可能已分配的句柄（即使 open 失败，sqlite3 也可能分配了句柄）
            sqlite3_close(db.m_pDb);
            db.m_pDb = nullptr;  // 置空，避免析构时再次 close
            return std::unexpected(OM_ERROR(InternalError,
                "sqlite3_open failed: " + strErr));
        }

        // 20260319 ZJH 开启 WAL（Write-Ahead Logging）日志模式
        // WAL 相比默认 DELETE 日志模式：读写并发更好、写入更快、崩溃恢复更安全
        // 内存数据库（":memory:"）同样接受此命令（会静默忽略 WAL 选项，无副作用）
        int nWalRet = sqlite3_exec(db.m_pDb, "PRAGMA journal_mode=WAL;",
            nullptr, nullptr, nullptr);
        if (nWalRet != SQLITE_OK) {
            // 20260319 ZJH WAL 设置失败不视为致命错误，实际生产中可只打警告日志
            // 此处为简洁起见也视为成功，继续返回已打开的数据库
            // （WAL 在部分只读文件系统上不可用，但基础功能仍正常）
        }

        return db;  // 以移动语义返回 Database 对象（RVO / NRVO 优化）
    }

    // 20260319 ZJH 执行非查询 SQL 语句（DDL / DML：CREATE TABLE / INSERT / UPDATE / DELETE）
    // 参数: strSQL — 要执行的 SQL 语句字符串
    // 返回: Result<void> — 成功时为空成功态，失败时携带 SQLite 错误信息
    Result<void> execute(const std::string& strSQL) {
        char* pErrMsg = nullptr;  // sqlite3_exec 将错误信息写入此指针（需手动 sqlite3_free）

        // 20260319 ZJH sqlite3_exec 执行一条或多条 SQL（分号分隔）
        // 第三、四参数为回调函数和用户数据，非查询语句传 nullptr 即可
        int nRet = sqlite3_exec(m_pDb, strSQL.c_str(), nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            // 20260319 ZJH 将 C 字符串错误信息复制到 std::string，再立即释放 C 字符串
            std::string strErr = pErrMsg ? pErrMsg : "unknown error";
            sqlite3_free(pErrMsg);  // 释放 sqlite3_exec 分配的错误信息内存，防止内存泄漏
            return std::unexpected(OM_ERROR(InternalError,
                "sqlite3_exec failed: " + strErr));
        }

        return {};  // 返回成功（std::expected<void,...> 成功态为空 {}）
    }

    // 20260319 ZJH 执行查询 SQL 语句并返回结果集
    // 参数: strSQL — SELECT 语句字符串
    // 返回: Result<std::vector<Row>> — 成功时持有所有行的列表，失败时携带错误信息
    //        每行为 unordered_map<列名, 字符串值>；NULL 列值映射为空字符串 ""
    Result<std::vector<Row>> query(const std::string& strSQL) {
        std::vector<Row> vecRows;  // 收集所有查询结果行

        // 20260319 ZJH 定义回调函数，sqlite3_exec 每找到一行就调用一次
        // 参数含义（sqlite3_exec 回调约定）：
        //   pData    — 用户传入的 void* 指针，此处指向 vecRows
        //   nCols    — 当前行的列数
        //   ppValues — 当前行各列的值（C 字符串数组，NULL 值的指针为 nullptr）
        //   ppNames  — 各列的列名（C 字符串数组）
        // 返回 0 表示继续，返回非 0 则中止并让 sqlite3_exec 返回 SQLITE_ABORT
        auto callback = [](void* pData, int nCols, char** ppValues, char** ppNames) -> int {
            // 20260319 ZJH 将 void* 还原为 std::vector<Row>* 以便追加行数据
            auto* pVec = static_cast<std::vector<Row>*>(pData);

            Row row;  // 构建当前行的列名->值映射
            for (int i = 0; i < nCols; ++i) {
                // 20260319 ZJH 列名取自 ppNames[i]（永远非 nullptr）
                std::string strColName = ppNames[i];
                // 20260319 ZJH ppValues[i] 为 nullptr 时表示 NULL 值，统一映射为空字符串
                std::string strValue = (ppValues[i] != nullptr) ? ppValues[i] : "";
                row[strColName] = strValue;  // 写入列名->值映射
            }
            pVec->push_back(std::move(row));  // 将当前行追加到结果集
            return 0;  // 返回 0 告知 sqlite3_exec 继续处理下一行
        };

        char* pErrMsg = nullptr;  // 错误信息指针，sqlite3_exec 分配，需手动 free

        // 20260319 ZJH 以 &vecRows 作为用户数据传入，回调函数中转换回 vector 指针使用
        int nRet = sqlite3_exec(m_pDb, strSQL.c_str(), callback, &vecRows, &pErrMsg);
        if (nRet != SQLITE_OK) {
            // 20260319 ZJH 执行失败时复制错误信息并释放 C 字符串内存
            std::string strErr = pErrMsg ? pErrMsg : "unknown error";
            sqlite3_free(pErrMsg);  // 防止内存泄漏
            return std::unexpected(OM_ERROR(InternalError,
                "sqlite3_exec (query) failed: " + strErr));
        }

        return vecRows;  // 返回已收集的所有行数据
    }

private:
    // 20260319 ZJH 私有默认构造 — 只允许通过静态工厂 open() 创建对象，保证句柄有效性
    Database() : m_pDb(nullptr) {}  // m_pDb 初始化为 nullptr，表示尚未打开数据库

    // 20260319 ZJH 内部关闭方法 — 仅在析构和移动赋值中调用，避免代码重复
    void close() {
        // 20260319 ZJH 仅当句柄非 nullptr 时才调用 close，防止重复关闭导致未定义行为
        if (m_pDb != nullptr) {
            sqlite3_close(m_pDb);  // 释放 sqlite3 连接所有资源（缓存、prepared statements 等）
            m_pDb = nullptr;        // 置空，标记已关闭，防止析构时再次调用
        }
    }

    sqlite3* m_pDb;  // 底层 SQLite 数据库连接句柄；nullptr 表示未打开或已关闭
};

}  // namespace om
