module;
// 20260319 ZJH 全局模块片段：引入标准库头文件，避免 C++ 模块边界的宏冲突问题
#include <cstdlib>    // malloc/free/_aligned_malloc/_aligned_free
#include <cstddef>    // size_t
#include <unordered_map>  // 用于记录每次分配地址到字节数的映射
#include <mutex>      // 保护多线程并发访问
#include <cstdint>    // uintptr_t 等整型

export module om.platform.memory;

// 20260319 ZJH 导出 df 命名空间下的 MemoryPool 类
export namespace om {

// 20260319 ZJH MemoryPool：线程安全的内存池，提供 64 字节缓存行对齐分配与统计功能
// 设计目标：
//   1. 分配的内存始终 64 字节对齐，避免伪共享（false sharing），提升缓存效率
//   2. 线程安全：所有公开方法通过 std::mutex 保护共享状态
//   3. 统计功能：通过 unordered_map 跟踪每块已分配内存的大小，支持 allocatedBytes() 查询
class MemoryPool {
public:
    // 20260319 ZJH 默认构造/析构；析构不自动释放未 deallocate 的内存（由调用方负责）
    MemoryPool() = default;
    ~MemoryPool() = default;

    // 20260319 ZJH 禁止拷贝和移动，MemoryPool 持有互斥量，语义上不可复制
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    // 20260319 ZJH 分配 nBytes 字节、64 字节对齐的内存块
    // 参数: nBytes - 请求分配的字节数
    // 返回: 指向已分配内存的指针；若 nBytes == 0 则返回 nullptr
    void* allocate(size_t nBytes) {
        // 20260319 ZJH 特殊情况：分配 0 字节时直接返回 nullptr，不进行实际分配
        if (nBytes == 0) {
            return nullptr;
        }

        // 20260319 ZJH 进行对齐分配：MSVC 使用 _aligned_malloc，其他平台使用 std::aligned_alloc
        void* pPtr = nullptr;
#ifdef _MSC_VER
        // 20260319 ZJH _aligned_malloc(size, alignment)：MSVC 专用对齐分配函数
        // 必须使用配对的 _aligned_free 释放，否则行为未定义
        pPtr = _aligned_malloc(nBytes, s_nAlignment);
#else
        // 20260319 ZJH std::aligned_alloc 要求 nBytes 必须是 alignment 的整数倍
        // 向上对齐 nBytes 到 s_nAlignment 的整数倍
        size_t nAligned = (nBytes + s_nAlignment - 1) & ~(s_nAlignment - 1);
        pPtr = std::aligned_alloc(s_nAlignment, nAligned);
#endif
        // 20260319 ZJH 若平台分配失败（如内存耗尽），直接返回 nullptr
        if (pPtr == nullptr) {
            return nullptr;
        }

        // 20260319 ZJH 加锁更新统计信息：记录 (地址 -> 字节数) 映射并累加已分配字节数
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_mapAllocations[pPtr] = nBytes;  // 记录该地址对应分配的字节数
            m_nAllocatedBytes += nBytes;       // 累加当前已分配总字节数
        }

        return pPtr;  // 返回指向已分配对齐内存的指针
    }

    // 20260319 ZJH 释放由 allocate() 返回的内存块，并更新统计信息
    // 参数: pPtr - 之前由 allocate() 返回的指针；传入 nullptr 时为无操作
    void deallocate(void* pPtr) {
        // 20260319 ZJH 传入 nullptr 时直接返回，符合 free(nullptr) 的惯例
        if (pPtr == nullptr) {
            return;
        }

        // 20260319 ZJH 加锁查找并移除该地址的分配记录，同时更新已分配字节数
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_mapAllocations.find(pPtr);  // 查找该指针的分配记录
            if (it != m_mapAllocations.end()) {
                // 20260319 ZJH 找到记录：从已分配总量中扣除对应字节数，并移除映射项
                m_nAllocatedBytes -= it->second;  // 扣除该块的字节数
                m_mapAllocations.erase(it);        // 移除该块的记录
            }
            // 20260319 ZJH 若未找到（即不是由本池分配的指针），跳过统计更新，仍继续释放
        }

        // 20260319 ZJH 调用平台对应的对齐内存释放函数
#ifdef _MSC_VER
        // 20260319 ZJH _aligned_free 与 _aligned_malloc 配对使用，释放对齐内存
        _aligned_free(pPtr);
#else
        // 20260319 ZJH std::free 可释放 std::aligned_alloc 分配的内存
        std::free(pPtr);
#endif
    }

    // 20260319 ZJH 查询当前已分配的总字节数（不包含已释放部分）
    // 返回: 当前池中所有未释放分配块的字节数之和
    size_t allocatedBytes() const {
        // 20260319 ZJH 加锁读取，保证多线程环境下读取的一致性
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_nAllocatedBytes;  // 返回当前已分配但未释放的总字节数
    }

private:
    // 20260319 ZJH 对齐粒度：64 字节 = 典型 x86-64 CPU 缓存行大小
    // 64 字节对齐可避免跨缓存行访问和多线程伪共享问题
    static constexpr size_t s_nAlignment = 64;

    // 20260319 ZJH 互斥量：保护 m_mapAllocations 和 m_nAllocatedBytes 的并发访问
    // mutable 修饰允许在 const 方法（allocatedBytes）中加锁
    mutable std::mutex m_mutex;

    // 20260319 ZJH 分配记录映射：key = 已分配内存指针，value = 该块的字节数
    // 用于 deallocate 时快速查找字节数并更新统计
    std::unordered_map<void*, size_t> m_mapAllocations;

    // 20260319 ZJH 当前已分配但未释放的总字节数，由 allocate/deallocate 维护
    size_t m_nAllocatedBytes = 0;
};

} // namespace om
