// 20260321 ZJH 数据集版本管理模块
// 基于 SQLite 的数据集快照创建/恢复/列举/删除，支持版本化管理训练数据集
module;

// 20260321 ZJH 全局模块片段：所有 #include 必须放在此处，位于 export module 声明之前
#include <sqlite3.h>        // SQLite C 接口，sqlite3_open / sqlite3_exec / sqlite3_prepare_v2 等
#include <string>           // std::string — 路径、名称、SQL 语句存储
#include <vector>           // std::vector — 图像路径列表、标签列表、类别列表
#include <numeric>          // std::accumulate — 用于拼接路径字符串计算哈希
#include <functional>       // std::hash — 路径列表的哈希计算
#include <sstream>          // std::ostringstream — 拼接字符串用于哈希输入
#include <cstring>          // std::memset — 初始化缓冲区（可选）

export module om.engine.dataset_version;

export namespace om {

// 20260321 ZJH DatasetSnapshot — 数据集快照信息结构体
// 描述某一时刻数据集的元信息，不包含实际图像数据，仅存储路径引用和统计摘要
struct DatasetSnapshot {
    int nVersionId = 0;               // 20260321 ZJH 版本号（snapshots 表主键，自增）
    std::string strName;              // 20260321 ZJH 快照名称（用户自定义，如 "baseline_v1"）
    std::string strTimestamp;         // 20260321 ZJH 创建时间（ISO 8601 格式，由 SQLite datetime('now') 生成）
    int nNumImages = 0;               // 20260321 ZJH 本次快照包含的图像总数量
    int nNumClasses = 0;              // 20260321 ZJH 本次快照包含的类别总数量
    std::string strDescription;       // 20260321 ZJH 版本描述（用户可选填写，如 "添加了100张缺陷样本"）
    std::string strHash;              // 20260321 ZJH 数据集哈希（所有图像路径拼接后的 std::hash 结果，用于快速比对版本差异）
};

// 20260321 ZJH DatasetVersionManager — 数据集版本管理器
// 职责：通过 SQLite 数据库持久化保存数据集的版本历史（快照），
//       支持创建快照、恢复到任意版本、列出历史、删除指定版本。
// 设计要点：
//   - 直接使用 sqlite3 C API，不依赖 om.platform.database，减少模块间耦合
//   - RAII 管理 sqlite3* 句柄，析构时自动 close
//   - 所有写操作使用事务（BEGIN/COMMIT/ROLLBACK）保证原子性
//   - 大批量插入使用 prepared statement + 参数绑定，性能优于逐条 sqlite3_exec
class DatasetVersionManager {
public:
    // 20260321 ZJH 构造函数 — 打开（或创建）指定路径的 SQLite 数据库文件
    // 参数: strDbPath — 数据库文件路径（如 "data/dataset_versions.db"），不存在时自动创建
    // 注意: 构造后必须调用 initialize() 创建表结构，否则后续操作会因表不存在而失败
    DatasetVersionManager(const std::string& strDbPath)
        : m_pDb(nullptr)  // 初始化句柄为 nullptr，表示尚未成功打开
        , m_strDbPath(strDbPath)  // 保存路径，用于日志或重新打开
    {
        // 20260321 ZJH sqlite3_open 以读写模式打开数据库；文件不存在时自动创建空数据库
        // 返回 SQLITE_OK(0) 表示成功，其他值表示失败
        int nRet = sqlite3_open(strDbPath.c_str(), &m_pDb);
        if (nRet != SQLITE_OK) {
            // 20260321 ZJH 打开失败时，sqlite3 可能仍分配了句柄，需要关闭并置空
            if (m_pDb != nullptr) {
                sqlite3_close(m_pDb);  // 释放可能已分配的资源
                m_pDb = nullptr;       // 置空防止析构时重复 close
            }
        }
    }

    // 20260321 ZJH 析构函数 — RAII 核心：对象销毁时自动关闭数据库连接
    // 若 m_pDb 为 nullptr（打开失败或已移动）则不执行任何操作
    ~DatasetVersionManager() {
        // 20260321 ZJH 仅当句柄非空时才关闭，防止重复 close 导致未定义行为
        if (m_pDb != nullptr) {
            sqlite3_close(m_pDb);  // 释放 sqlite3 连接所持有的所有资源
            m_pDb = nullptr;       // 置空，标记已关闭（虽然析构后不再访问，但保持良好习惯）
        }
    }

    // 20260321 ZJH 禁止拷贝 — sqlite3* 句柄不可共享，防止双重 close
    DatasetVersionManager(const DatasetVersionManager&) = delete;
    DatasetVersionManager& operator=(const DatasetVersionManager&) = delete;

    // 20260321 ZJH initialize — 创建数据库表结构（Schema）
    // 包含 3 张表：snapshots（快照元信息）、snapshot_images（快照关联的图像路径+标签）、
    //              snapshot_classes（快照关联的类别名称+索引）
    // 使用 IF NOT EXISTS 保证幂等：多次调用不会报错也不会清除已有数据
    // 返回: true — schema 创建成功；false — 数据库未打开或 SQL 执行失败
    bool initialize() {
        // 20260321 ZJH 检查数据库句柄是否有效，构造函数可能因路径无效而打开失败
        if (m_pDb == nullptr) {
            return false;  // 句柄无效，无法执行任何 SQL
        }

        // 20260321 ZJH snapshots 表 — 存储每个快照的元信息
        // id: 自增主键，即版本号（nVersionId）
        // name: 快照名称（用户指定，NOT NULL 约束）
        // timestamp: 创建时间，默认取当前 UTC 时间（datetime('now')）
        // num_images: 该快照包含的图像数量
        // num_classes: 该快照包含的类别数量
        // description: 用户可选的版本描述
        // hash: 路径列表的哈希值，用于快速判断两个版本的数据集是否相同
        const char* pSqlSnapshots =
            "CREATE TABLE IF NOT EXISTS snapshots ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  name TEXT NOT NULL,"
            "  timestamp TEXT DEFAULT (datetime('now')),"
            "  num_images INTEGER,"
            "  num_classes INTEGER,"
            "  description TEXT,"
            "  hash TEXT"
            ");";

        // 20260321 ZJH snapshot_images 表 — 存储快照关联的所有图像路径和标签
        // snapshot_id: 外键，关联 snapshots.id（级联关系由应用层管理）
        // image_path: 图像文件的绝对或相对路径
        // label_id: 图像对应的类别索引，-1 表示未标注
        const char* pSqlImages =
            "CREATE TABLE IF NOT EXISTS snapshot_images ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  snapshot_id INTEGER REFERENCES snapshots(id),"
            "  image_path TEXT NOT NULL,"
            "  label_id INTEGER DEFAULT -1"
            ");";

        // 20260321 ZJH snapshot_classes 表 — 存储快照关联的类别名称映射
        // snapshot_id: 外键，关联 snapshots.id
        // class_index: 类别索引（与 label_id 对应，0-based）
        // class_name: 类别名称（如 "cat", "dog", "defect_scratch"）
        const char* pSqlClasses =
            "CREATE TABLE IF NOT EXISTS snapshot_classes ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  snapshot_id INTEGER REFERENCES snapshots(id),"
            "  class_index INTEGER,"
            "  class_name TEXT NOT NULL"
            ");";

        // 20260321 ZJH 依次执行三条 CREATE TABLE 语句
        // 任一条失败则返回 false，由调用方决定是否重试或报错
        char* pErrMsg = nullptr;  // sqlite3_exec 会将错误信息写入此指针

        // 20260321 ZJH 创建 snapshots 表
        int nRet = sqlite3_exec(m_pDb, pSqlSnapshots, nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);  // 释放错误信息内存，防止泄漏
            return false;           // 返回失败
        }

        // 20260321 ZJH 创建 snapshot_images 表
        nRet = sqlite3_exec(m_pDb, pSqlImages, nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);  // 释放错误信息
            return false;
        }

        // 20260321 ZJH 创建 snapshot_classes 表
        nRet = sqlite3_exec(m_pDb, pSqlClasses, nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);  // 释放错误信息
            return false;
        }

        // 20260321 ZJH 为 snapshot_images 和 snapshot_classes 创建索引，加速按 snapshot_id 查询
        // 恢复快照时需要按 snapshot_id 查询所有关联图像和类别，索引可将查询从全表扫描降为 B-tree 查找
        const char* pSqlIdx1 =
            "CREATE INDEX IF NOT EXISTS idx_images_snapshot ON snapshot_images(snapshot_id);";
        const char* pSqlIdx2 =
            "CREATE INDEX IF NOT EXISTS idx_classes_snapshot ON snapshot_classes(snapshot_id);";

        // 20260321 ZJH 索引创建失败不视为致命错误（功能正常，只是查询较慢），但仍尝试创建
        sqlite3_exec(m_pDb, pSqlIdx1, nullptr, nullptr, nullptr);
        sqlite3_exec(m_pDb, pSqlIdx2, nullptr, nullptr, nullptr);

        return true;  // 三张表全部创建成功
    }

    // 20260321 ZJH createSnapshot — 创建新的数据集快照
    // 将当前数据集的全部图像路径、标签和类别名称持久化到数据库中
    // 参数:
    //   strName        — 快照名称（必填，如 "v1_baseline"）
    //   vecImagePaths  — 所有图像文件路径列表
    //   vecLabels      — 每张图像对应的类别索引（与 vecImagePaths 一一对应）
    //   vecClassNames  — 类别名称列表（索引即 class_index）
    //   strDescription — 可选的版本描述
    // 返回: 新创建的版本号（snapshots.id），失败返回 -1
    // 事务保证: 使用 BEGIN/COMMIT 事务，保证快照元信息和图像/类别数据的原子写入
    int createSnapshot(const std::string& strName,
                       const std::vector<std::string>& vecImagePaths,
                       const std::vector<int>& vecLabels,
                       const std::vector<std::string>& vecClassNames,
                       const std::string& strDescription = "") {
        // 20260321 ZJH 检查数据库连接
        if (m_pDb == nullptr) {
            return -1;  // 数据库未打开，无法创建快照
        }

        // 20260321 ZJH 计算图像路径列表的哈希值，用于快速比对版本差异
        std::string strHash = computeHash(vecImagePaths);

        // 20260321 ZJH 图像数量和类别数量
        int nNumImages = static_cast<int>(vecImagePaths.size());    // 图像总数
        int nNumClasses = static_cast<int>(vecClassNames.size());   // 类别总数

        // 20260321 ZJH 开启事务 — 保证快照元信息+图像数据+类别数据的原子写入
        // 若中途失败则回滚，避免出现只有元信息没有图像数据的不完整快照
        char* pErrMsg = nullptr;  // 错误信息指针
        int nRet = sqlite3_exec(m_pDb, "BEGIN TRANSACTION;", nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);  // 释放错误信息
            return -1;              // 事务开启失败
        }

        // ==========================================
        // 步骤 1: 插入快照元信息到 snapshots 表
        // ==========================================
        // 20260321 ZJH 使用 prepared statement 防止 SQL 注入（name/description 可能含特殊字符）
        const char* pSqlInsertSnapshot =
            "INSERT INTO snapshots (name, num_images, num_classes, description, hash) "
            "VALUES (?, ?, ?, ?, ?);";

        sqlite3_stmt* pStmt = nullptr;  // prepared statement 句柄

        // 20260321 ZJH sqlite3_prepare_v2 编译 SQL 语句为字节码
        // 参数: m_pDb — 数据库句柄
        //       pSqlInsertSnapshot — SQL 模板（? 为占位符）
        //       -1 — SQL 长度（-1 表示自动计算到 \0）
        //       &pStmt — 输出编译后的 statement 句柄
        //       nullptr — 忽略未使用的 SQL 尾部
        nRet = sqlite3_prepare_v2(m_pDb, pSqlInsertSnapshot, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);  // 回滚事务
            return -1;  // SQL 编译失败
        }

        // 20260321 ZJH 绑定参数值到占位符（索引从 1 开始）
        // SQLITE_TRANSIENT 表示 SQLite 会复制字符串内容，调用方可安全释放原始字符串
        sqlite3_bind_text(pStmt, 1, strName.c_str(), -1, SQLITE_TRANSIENT);          // ? 1 = name
        sqlite3_bind_int(pStmt, 2, nNumImages);                                       // ? 2 = num_images
        sqlite3_bind_int(pStmt, 3, nNumClasses);                                      // ? 3 = num_classes
        sqlite3_bind_text(pStmt, 4, strDescription.c_str(), -1, SQLITE_TRANSIENT);    // ? 4 = description
        sqlite3_bind_text(pStmt, 5, strHash.c_str(), -1, SQLITE_TRANSIENT);          // ? 5 = hash

        // 20260321 ZJH sqlite3_step 执行编译好的 SQL
        // INSERT 成功时返回 SQLITE_DONE，查询时返回 SQLITE_ROW
        nRet = sqlite3_step(pStmt);
        sqlite3_finalize(pStmt);  // 无论成功与否都必须 finalize 释放 statement 资源
        pStmt = nullptr;          // 置空防止后续误用

        if (nRet != SQLITE_DONE) {
            // 20260321 ZJH INSERT 失败，回滚事务并返回错误
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return -1;
        }

        // 20260321 ZJH 获取刚插入行的自增 ID（即新快照的版本号）
        // sqlite3_last_insert_rowid 返回最近一次 INSERT 的 rowid
        int nSnapshotId = static_cast<int>(sqlite3_last_insert_rowid(m_pDb));

        // ==========================================
        // 步骤 2: 批量插入图像路径和标签到 snapshot_images 表
        // ==========================================
        // 20260321 ZJH 使用 prepared statement + 循环绑定，比逐条 sqlite3_exec 快数十倍
        // 原因：sqlite3_exec 每次都要解析+编译 SQL，而 prepared statement 只编译一次
        const char* pSqlInsertImage =
            "INSERT INTO snapshot_images (snapshot_id, image_path, label_id) VALUES (?, ?, ?);";

        nRet = sqlite3_prepare_v2(m_pDb, pSqlInsertImage, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);  // 回滚
            return -1;  // SQL 编译失败
        }

        // 20260321 ZJH 遍历所有图像，逐条绑定参数并执行 INSERT
        for (int i = 0; i < nNumImages; ++i) {
            // 20260321 ZJH 重置 statement 以复用（清除上一次绑定的参数和执行状态）
            sqlite3_reset(pStmt);

            // 20260321 ZJH 绑定本次循环的参数
            sqlite3_bind_int(pStmt, 1, nSnapshotId);                                             // snapshot_id
            sqlite3_bind_text(pStmt, 2, vecImagePaths[i].c_str(), -1, SQLITE_TRANSIENT);        // image_path
            // 20260321 ZJH 若 vecLabels 有对应元素则使用，否则默认 -1（未标注）
            int nLabel = (i < static_cast<int>(vecLabels.size())) ? vecLabels[i] : -1;
            sqlite3_bind_int(pStmt, 3, nLabel);                                                  // label_id

            // 20260321 ZJH 执行 INSERT
            nRet = sqlite3_step(pStmt);
            if (nRet != SQLITE_DONE) {
                // 20260321 ZJH 某条 INSERT 失败，释放 statement 并回滚整个事务
                sqlite3_finalize(pStmt);
                sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
                return -1;
            }
        }

        sqlite3_finalize(pStmt);  // 释放图像插入的 statement
        pStmt = nullptr;

        // ==========================================
        // 步骤 3: 批量插入类别名称到 snapshot_classes 表
        // ==========================================
        const char* pSqlInsertClass =
            "INSERT INTO snapshot_classes (snapshot_id, class_index, class_name) VALUES (?, ?, ?);";

        nRet = sqlite3_prepare_v2(m_pDb, pSqlInsertClass, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);  // 回滚
            return -1;
        }

        // 20260321 ZJH 遍历所有类别名称，逐条绑定参数并执行 INSERT
        for (int i = 0; i < nNumClasses; ++i) {
            sqlite3_reset(pStmt);  // 重置 statement 以复用

            sqlite3_bind_int(pStmt, 1, nSnapshotId);                                             // snapshot_id
            sqlite3_bind_int(pStmt, 2, i);                                                       // class_index（0-based）
            sqlite3_bind_text(pStmt, 3, vecClassNames[i].c_str(), -1, SQLITE_TRANSIENT);        // class_name

            nRet = sqlite3_step(pStmt);
            if (nRet != SQLITE_DONE) {
                sqlite3_finalize(pStmt);
                sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);  // 回滚
                return -1;
            }
        }

        sqlite3_finalize(pStmt);  // 释放类别插入的 statement
        pStmt = nullptr;

        // ==========================================
        // 步骤 4: 提交事务
        // ==========================================
        // 20260321 ZJH COMMIT 将所有步骤的写入原子地持久化到磁盘
        nRet = sqlite3_exec(m_pDb, "COMMIT;", nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);
            // 20260321 ZJH COMMIT 失败时尝试回滚（极少发生，通常是磁盘满或锁冲突）
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return -1;
        }

        return nSnapshotId;  // 返回新创建的版本号
    }

    // 20260321 ZJH restoreSnapshot — 恢复到指定版本的数据集快照
    // 从数据库中读取指定版本号对应的所有图像路径、标签和类别名称
    // 参数:
    //   nVersionId    — 要恢复的版本号（snapshots.id）
    //   vecImagePaths — [输出] 恢复后的图像路径列表
    //   vecLabels     — [输出] 恢复后的标签列表（与 vecImagePaths 一一对应）
    //   vecClassNames — [输出] 恢复后的类别名称列表
    // 返回: true — 恢复成功；false — 版本不存在或数据库错误
    bool restoreSnapshot(int nVersionId,
                         std::vector<std::string>& vecImagePaths,
                         std::vector<int>& vecLabels,
                         std::vector<std::string>& vecClassNames) {
        // 20260321 ZJH 检查数据库连接
        if (m_pDb == nullptr) {
            return false;
        }

        // 20260321 ZJH 清空输出参数，确保调用方得到干净的结果
        vecImagePaths.clear();
        vecLabels.clear();
        vecClassNames.clear();

        // ==========================================
        // 步骤 1: 验证指定版本是否存在
        // ==========================================
        // 20260321 ZJH 查询 snapshots 表确认 id 存在
        const char* pSqlCheckExists = "SELECT id FROM snapshots WHERE id = ?;";
        sqlite3_stmt* pStmt = nullptr;

        int nRet = sqlite3_prepare_v2(m_pDb, pSqlCheckExists, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            return false;  // SQL 编译失败
        }

        sqlite3_bind_int(pStmt, 1, nVersionId);  // 绑定版本号
        nRet = sqlite3_step(pStmt);
        sqlite3_finalize(pStmt);  // 释放 statement
        pStmt = nullptr;

        // 20260321 ZJH SQLITE_ROW 表示找到了匹配行，SQLITE_DONE 表示没找到
        if (nRet != SQLITE_ROW) {
            return false;  // 指定版本不存在
        }

        // ==========================================
        // 步骤 2: 查询该版本的所有图像路径和标签
        // ==========================================
        // 20260321 ZJH 按 id 升序查询，保证恢复后的顺序与创建时一致
        const char* pSqlQueryImages =
            "SELECT image_path, label_id FROM snapshot_images WHERE snapshot_id = ? ORDER BY id ASC;";

        nRet = sqlite3_prepare_v2(m_pDb, pSqlQueryImages, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            return false;
        }

        sqlite3_bind_int(pStmt, 1, nVersionId);  // 绑定版本号

        // 20260321 ZJH 逐行读取查询结果
        // sqlite3_step 每次返回 SQLITE_ROW 表示还有数据行，SQLITE_DONE 表示遍历完毕
        while ((nRet = sqlite3_step(pStmt)) == SQLITE_ROW) {
            // 20260321 ZJH sqlite3_column_text 返回当前行指定列的文本值
            // 列索引从 0 开始：0 = image_path, 1 = label_id
            const unsigned char* pText = sqlite3_column_text(pStmt, 0);  // image_path 列
            // 20260321 ZJH 防止 NULL 值导致 string 构造崩溃
            std::string strPath = (pText != nullptr)
                ? std::string(reinterpret_cast<const char*>(pText))
                : std::string();
            vecImagePaths.push_back(strPath);

            // 20260321 ZJH sqlite3_column_int 返回当前行指定列的整数值
            int nLabel = sqlite3_column_int(pStmt, 1);  // label_id 列
            vecLabels.push_back(nLabel);
        }

        sqlite3_finalize(pStmt);  // 释放 statement
        pStmt = nullptr;

        // ==========================================
        // 步骤 3: 查询该版本的所有类别名称
        // ==========================================
        // 20260321 ZJH 按 class_index 升序查询，保证类别名称列表的索引顺序正确
        const char* pSqlQueryClasses =
            "SELECT class_name FROM snapshot_classes WHERE snapshot_id = ? ORDER BY class_index ASC;";

        nRet = sqlite3_prepare_v2(m_pDb, pSqlQueryClasses, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            return false;
        }

        sqlite3_bind_int(pStmt, 1, nVersionId);  // 绑定版本号

        // 20260321 ZJH 逐行读取类别名称
        while ((nRet = sqlite3_step(pStmt)) == SQLITE_ROW) {
            const unsigned char* pText = sqlite3_column_text(pStmt, 0);  // class_name 列
            std::string strClassName = (pText != nullptr)
                ? std::string(reinterpret_cast<const char*>(pText))
                : std::string();
            vecClassNames.push_back(strClassName);
        }

        sqlite3_finalize(pStmt);  // 释放 statement
        pStmt = nullptr;

        return true;  // 恢复成功
    }

    // 20260321 ZJH listSnapshots — 列出数据库中所有快照的元信息
    // 按创建时间降序排列（最新的在前），不包含图像/类别明细数据
    // 返回: DatasetSnapshot 列表；若数据库未打开或查询失败则返回空列表
    std::vector<DatasetSnapshot> listSnapshots() {
        std::vector<DatasetSnapshot> vecSnapshots;  // 结果列表

        // 20260321 ZJH 检查数据库连接
        if (m_pDb == nullptr) {
            return vecSnapshots;  // 返回空列表
        }

        // 20260321 ZJH 查询所有快照，按 id 降序（最新创建的版本排在前面）
        const char* pSqlQuery =
            "SELECT id, name, timestamp, num_images, num_classes, description, hash "
            "FROM snapshots ORDER BY id DESC;";

        sqlite3_stmt* pStmt = nullptr;
        int nRet = sqlite3_prepare_v2(m_pDb, pSqlQuery, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            return vecSnapshots;  // SQL 编译失败，返回空列表
        }

        // 20260321 ZJH 逐行读取快照元信息
        while ((nRet = sqlite3_step(pStmt)) == SQLITE_ROW) {
            DatasetSnapshot snapshot;  // 构建单个快照对象

            // 20260321 ZJH 读取各列（索引从 0 开始，对应 SELECT 中的列顺序）
            snapshot.nVersionId = sqlite3_column_int(pStmt, 0);  // id

            // 20260321 ZJH 读取文本列，防止 NULL 导致崩溃
            const unsigned char* pName = sqlite3_column_text(pStmt, 1);  // name
            snapshot.strName = (pName != nullptr)
                ? std::string(reinterpret_cast<const char*>(pName))
                : std::string();

            const unsigned char* pTimestamp = sqlite3_column_text(pStmt, 2);  // timestamp
            snapshot.strTimestamp = (pTimestamp != nullptr)
                ? std::string(reinterpret_cast<const char*>(pTimestamp))
                : std::string();

            snapshot.nNumImages = sqlite3_column_int(pStmt, 3);   // num_images
            snapshot.nNumClasses = sqlite3_column_int(pStmt, 4);  // num_classes

            const unsigned char* pDesc = sqlite3_column_text(pStmt, 5);  // description
            snapshot.strDescription = (pDesc != nullptr)
                ? std::string(reinterpret_cast<const char*>(pDesc))
                : std::string();

            const unsigned char* pHash = sqlite3_column_text(pStmt, 6);  // hash
            snapshot.strHash = (pHash != nullptr)
                ? std::string(reinterpret_cast<const char*>(pHash))
                : std::string();

            vecSnapshots.push_back(snapshot);  // 追加到结果列表
        }

        sqlite3_finalize(pStmt);  // 释放 statement
        return vecSnapshots;       // 返回所有快照（可能为空）
    }

    // 20260321 ZJH deleteSnapshot — 删除指定版本号的快照及其关联的所有图像和类别数据
    // 使用事务保证原子删除：三张表的数据要么全部删除，要么全部保留
    // 参数: nVersionId — 要删除的版本号
    // 返回: true — 删除成功；false — 版本不存在或数据库错误
    bool deleteSnapshot(int nVersionId) {
        // 20260321 ZJH 检查数据库连接
        if (m_pDb == nullptr) {
            return false;
        }

        // 20260321 ZJH 开启事务 — 保证三张表的删除操作原子执行
        char* pErrMsg = nullptr;
        int nRet = sqlite3_exec(m_pDb, "BEGIN TRANSACTION;", nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);
            return false;
        }

        // 20260321 ZJH 删除关联的图像数据（先删子表，再删主表，避免外键约束问题）
        sqlite3_stmt* pStmt = nullptr;

        // 20260321 ZJH 删除 snapshot_images 中该版本的所有行
        const char* pSqlDelImages = "DELETE FROM snapshot_images WHERE snapshot_id = ?;";
        nRet = sqlite3_prepare_v2(m_pDb, pSqlDelImages, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }
        sqlite3_bind_int(pStmt, 1, nVersionId);
        nRet = sqlite3_step(pStmt);
        sqlite3_finalize(pStmt);
        pStmt = nullptr;

        if (nRet != SQLITE_DONE) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }

        // 20260321 ZJH 删除 snapshot_classes 中该版本的所有行
        const char* pSqlDelClasses = "DELETE FROM snapshot_classes WHERE snapshot_id = ?;";
        nRet = sqlite3_prepare_v2(m_pDb, pSqlDelClasses, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }
        sqlite3_bind_int(pStmt, 1, nVersionId);
        nRet = sqlite3_step(pStmt);
        sqlite3_finalize(pStmt);
        pStmt = nullptr;

        if (nRet != SQLITE_DONE) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }

        // 20260321 ZJH 删除 snapshots 主表中的该版本记录
        const char* pSqlDelSnapshot = "DELETE FROM snapshots WHERE id = ?;";
        nRet = sqlite3_prepare_v2(m_pDb, pSqlDelSnapshot, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }
        sqlite3_bind_int(pStmt, 1, nVersionId);
        nRet = sqlite3_step(pStmt);
        sqlite3_finalize(pStmt);
        pStmt = nullptr;

        if (nRet != SQLITE_DONE) {
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }

        // 20260321 ZJH 检查是否实际删除了主表行（changes() > 0 表示确实存在该版本）
        int nChanges = sqlite3_changes(m_pDb);

        // 20260321 ZJH 提交事务
        nRet = sqlite3_exec(m_pDb, "COMMIT;", nullptr, nullptr, &pErrMsg);
        if (nRet != SQLITE_OK) {
            sqlite3_free(pErrMsg);
            sqlite3_exec(m_pDb, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }

        // 20260321 ZJH 若 changes == 0 表示该版本号不存在（DELETE 成功但未匹配任何行）
        return (nChanges > 0);
    }

    // 20260321 ZJH getLatestSnapshot — 获取最新创建的快照
    // 按 id 降序取第一条（id 最大的即最新创建的）
    // 返回: 最新的 DatasetSnapshot；若无快照则返回 nVersionId == 0 的默认对象
    DatasetSnapshot getLatestSnapshot() {
        DatasetSnapshot snapshot;  // 默认构造，nVersionId == 0 表示无效

        // 20260321 ZJH 检查数据库连接
        if (m_pDb == nullptr) {
            return snapshot;
        }

        // 20260321 ZJH LIMIT 1 只取一条，ORDER BY id DESC 保证是最新的
        const char* pSqlQuery =
            "SELECT id, name, timestamp, num_images, num_classes, description, hash "
            "FROM snapshots ORDER BY id DESC LIMIT 1;";

        sqlite3_stmt* pStmt = nullptr;
        int nRet = sqlite3_prepare_v2(m_pDb, pSqlQuery, -1, &pStmt, nullptr);
        if (nRet != SQLITE_OK) {
            return snapshot;  // SQL 编译失败，返回默认空快照
        }

        // 20260321 ZJH 尝试读取一行
        nRet = sqlite3_step(pStmt);
        if (nRet == SQLITE_ROW) {
            // 20260321 ZJH 找到最新快照，读取各列
            snapshot.nVersionId = sqlite3_column_int(pStmt, 0);

            const unsigned char* pName = sqlite3_column_text(pStmt, 1);
            snapshot.strName = (pName != nullptr)
                ? std::string(reinterpret_cast<const char*>(pName))
                : std::string();

            const unsigned char* pTimestamp = sqlite3_column_text(pStmt, 2);
            snapshot.strTimestamp = (pTimestamp != nullptr)
                ? std::string(reinterpret_cast<const char*>(pTimestamp))
                : std::string();

            snapshot.nNumImages = sqlite3_column_int(pStmt, 3);
            snapshot.nNumClasses = sqlite3_column_int(pStmt, 4);

            const unsigned char* pDesc = sqlite3_column_text(pStmt, 5);
            snapshot.strDescription = (pDesc != nullptr)
                ? std::string(reinterpret_cast<const char*>(pDesc))
                : std::string();

            const unsigned char* pHash = sqlite3_column_text(pStmt, 6);
            snapshot.strHash = (pHash != nullptr)
                ? std::string(reinterpret_cast<const char*>(pHash))
                : std::string();
        }
        // 20260321 ZJH 若 nRet == SQLITE_DONE 则表示表为空，snapshot 保持默认值（nVersionId == 0）

        sqlite3_finalize(pStmt);  // 释放 statement
        return snapshot;
    }

    // 20260321 ZJH computeHash — 计算图像路径列表的哈希值
    // 将所有路径以换行符拼接为一个长字符串，然后使用 std::hash<std::string> 计算哈希
    // 用途：快速判断两个版本的数据集内容是否相同（路径完全一致则哈希相同）
    // 局限：不检测文件内容变化（同一路径下文件被替换不会改变哈希）；
    //       路径顺序不同也会导致哈希不同（设计如此，顺序变化也视为版本变更）
    // 参数: vecPaths — 图像路径列表
    // 返回: 哈希值的十六进制字符串表示（如 "a1b2c3d4e5f67890"）
    std::string computeHash(const std::vector<std::string>& vecPaths) {
        // 20260321 ZJH 使用 ostringstream 拼接所有路径，以换行符分隔
        std::ostringstream oss;
        for (size_t i = 0; i < vecPaths.size(); ++i) {
            if (i > 0) {
                oss << '\n';  // 路径间以换行符分隔
            }
            oss << vecPaths[i];  // 追加路径字符串
        }

        // 20260321 ZJH 使用标准库的 std::hash 计算字符串哈希（平台相关，但足够用于版本比对）
        std::string strCombined = oss.str();  // 拼接后的完整字符串
        size_t nHashValue = std::hash<std::string>{}(strCombined);  // 计算哈希

        // 20260321 ZJH 将哈希值转为 16 位十六进制字符串（固定宽度，前补零）
        // 使用 ostringstream 而非 snprintf 以保持纯 C++ 风格
        std::ostringstream ossHex;
        ossHex << std::hex << nHashValue;  // 十六进制输出
        return ossHex.str();  // 返回哈希字符串
    }

private:
    sqlite3* m_pDb;              // 20260321 ZJH SQLite 数据库连接句柄；nullptr 表示未打开或已关闭
    std::string m_strDbPath;     // 20260321 ZJH 数据库文件路径（构造时传入，用于日志和调试）
};

}  // namespace om
