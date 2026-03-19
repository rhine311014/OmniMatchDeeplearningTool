// 20260319 ZJH MemoryPool 单元测试
// 验证：对齐分配、内存可用性、零字节分配、多次分配/释放、统计字节数追踪
#include <gtest/gtest.h>
import df.platform.memory;

// 20260319 ZJH 测试1：基本分配与释放
// 验证 allocate(1024) 返回非 nullptr 指针，deallocate 不崩溃
TEST(MemoryPool, AllocateAndDeallocate) {
    df::MemoryPool pool;  // 创建内存池实例

    // 20260319 ZJH 分配 1024 字节，期望返回有效指针
    void* pPtr = pool.allocate(1024);
    EXPECT_NE(pPtr, nullptr);  // 分配 1024 字节不应返回 nullptr

    // 20260319 ZJH 释放已分配的内存，期望不崩溃
    pool.deallocate(pPtr);
}

// 20260319 ZJH 测试2：分配的内存可正常读写
// 分配 100 个 float 的空间，逐个写入并回读验证数据完整性
TEST(MemoryPool, MemoryIsUsable) {
    df::MemoryPool pool;  // 创建内存池实例

    // 20260319 ZJH 分配足以存放 100 个 float 的内存块
    const size_t nCount = 100;  // 元素数量
    void* pRaw = pool.allocate(nCount * sizeof(float));
    ASSERT_NE(pRaw, nullptr);  // 分配必须成功，否则后续操作无意义

    // 20260319 ZJH 将原始指针转换为 float* 以便按元素访问
    float* pFloats = static_cast<float*>(pRaw);

    // 20260319 ZJH 写入：每个元素赋值为其下标的浮点数（0.0f, 1.0f, ..., 99.0f）
    for (size_t i = 0; i < nCount; ++i) {
        pFloats[i] = static_cast<float>(i);  // 写入测试值
    }

    // 20260319 ZJH 回读：逐一验证每个元素的值与写入值一致
    for (size_t i = 0; i < nCount; ++i) {
        EXPECT_FLOAT_EQ(pFloats[i], static_cast<float>(i));  // 读写一致性校验
    }

    // 20260319 ZJH 释放内存，避免统计泄漏
    pool.deallocate(pRaw);
}

// 20260319 ZJH 测试3：分配 0 字节应返回 nullptr
TEST(MemoryPool, ZeroSizeReturnsNull) {
    df::MemoryPool pool;  // 创建内存池实例

    // 20260319 ZJH 分配 0 字节，期望返回 nullptr（不触发实际分配）
    void* pPtr = pool.allocate(0);
    EXPECT_EQ(pPtr, nullptr);  // 零字节分配必须返回 nullptr
}

// 20260319 ZJH 测试4：连续 100 次分配与释放不崩溃
// 验证内存池在高频分配/释放场景下的稳定性
TEST(MemoryPool, MultipleAllocations) {
    df::MemoryPool pool;  // 创建内存池实例

    const int nIterations = 100;  // 分配/释放轮次数
    for (int i = 0; i < nIterations; ++i) {
        // 20260319 ZJH 每次分配 64 字节（恰好等于对齐粒度），验证对齐分配路径
        void* pPtr = pool.allocate(64);
        EXPECT_NE(pPtr, nullptr);  // 每次分配必须成功

        // 20260319 ZJH 立即释放，验证 deallocate 正确维护内部状态
        pool.deallocate(pPtr);
    }

    // 20260319 ZJH 所有分配已释放，统计字节数应归零
    EXPECT_EQ(pool.allocatedBytes(), 0u);
}

// 20260319 ZJH 测试5：allocatedBytes() 正确追踪已分配字节数
// 场景：初始为 0 -> 分配两块后 >= 2048 -> 全部释放后回到 0
TEST(MemoryPool, Statistics) {
    df::MemoryPool pool;  // 创建内存池实例

    // 20260319 ZJH 初始状态：未分配任何内存，统计应为 0
    EXPECT_EQ(pool.allocatedBytes(), 0u);

    // 20260319 ZJH 分配两块各 1024 字节的内存
    void* pA = pool.allocate(1024);  // 第一块：1024 字节
    void* pB = pool.allocate(1024);  // 第二块：1024 字节

    ASSERT_NE(pA, nullptr);  // 第一块分配必须成功
    ASSERT_NE(pB, nullptr);  // 第二块分配必须成功

    // 20260319 ZJH 两块均已分配，总字节数应 >= 2048（实际分配量可能有对齐补齐）
    EXPECT_GE(pool.allocatedBytes(), static_cast<size_t>(2048));

    // 20260319 ZJH 释放第一块，统计应减少 1024
    pool.deallocate(pA);
    EXPECT_GE(pool.allocatedBytes(), static_cast<size_t>(1024));  // 仍有第二块未释放

    // 20260319 ZJH 释放第二块，统计应归零
    pool.deallocate(pB);
    EXPECT_EQ(pool.allocatedBytes(), 0u);  // 全部释放后统计必须为 0
}
