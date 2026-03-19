// 20260319 ZJH ThreadPool 单元测试
// 覆盖：提交返回值任务、批量任务、任务计数正确性、threadCount、void 任务
#include <gtest/gtest.h>
#include <atomic>    // std::atomic，用于多线程安全计数
#include <vector>    // std::vector，存储 future
#include <future>    // std::future，接收 submit 结果

import df.platform.thread_pool;

// 20260319 ZJH 测试1：提交一个返回 int 的 lambda，验证 future.get() 返回正确值
TEST(ThreadPoolTest, SubmitAndGetResult) {
    // 20260319 ZJH 创建 2 线程的线程池，足够运行单个任务
    df::ThreadPool pool(2);

    // 20260319 ZJH 提交返回 42 的 lambda，立即获取 future
    auto fut = pool.submit([]() -> int {
        return 42;  // 任务体：直接返回整数值
    });

    // 20260319 ZJH fut.get() 阻塞等待任务执行完毕，并取回返回值
    int nResult = fut.get();
    // 20260319 ZJH 验证返回值与预期一致
    EXPECT_EQ(nResult, 42);
}

// 20260319 ZJH 测试2：提交 100 个计算 i*i 的任务，验证所有 future 返回正确结果
TEST(ThreadPoolTest, MultipleTasks) {
    // 20260319 ZJH 创建 4 线程的线程池，模拟多核并行
    df::ThreadPool pool(4);

    // 20260319 ZJH 存放所有 future，避免在 submit 后立即 get 导致串行
    std::vector<std::future<int>> vecFutures;
    vecFutures.reserve(100);  // 预分配，避免 push_back 时重新分配

    // 20260319 ZJH 提交 100 个任务，每个任务计算 i*i
    for (int i = 0; i < 100; ++i) {
        // 20260319 ZJH 按值捕获 i，确保每个 lambda 拥有独立的 i 副本
        vecFutures.push_back(pool.submit([i]() -> int {
            return i * i;  // 计算平方值
        }));
    }

    // 20260319 ZJH 逐一获取结果并验证：第 i 个 future 应返回 i*i
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vecFutures[i].get(), i * i);
    }
}

// 20260319 ZJH 测试3：提交 1000 个递增原子计数的任务，waitAll() 后验证计数恰好为 1000
TEST(ThreadPoolTest, AllTasksExecuted) {
    // 20260319 ZJH 创建 4 线程的线程池
    df::ThreadPool pool(4);

    // 20260319 ZJH 原子计数器，线程安全地统计任务执行次数
    std::atomic<int> nCounter{0};

    // 20260319 ZJH 提交 1000 个任务，每个任务将计数器 +1
    for (int i = 0; i < 1000; ++i) {
        // 20260319 ZJH 按引用捕获 nCounter（原子变量，引用捕获安全）
        pool.submit([&nCounter]() -> void {
            nCounter.fetch_add(1, std::memory_order_relaxed);  // 原子递增
        });
    }

    // 20260319 ZJH waitAll() 阻塞直到所有 1000 个任务均执行完毕
    pool.waitAll();

    // 20260319 ZJH 验证计数器恰好等于 1000，确保没有任务丢失或重复执行
    EXPECT_EQ(nCounter.load(), 1000);
}

// 20260319 ZJH 测试4：验证 threadCount() 返回构造时指定的线程数
TEST(ThreadPoolTest, ThreadCount) {
    // 20260319 ZJH 创建 3 线程的线程池
    df::ThreadPool pool(3);

    // 20260319 ZJH threadCount() 应返回构造时传入的线程数 3
    EXPECT_EQ(pool.threadCount(), 3);
}

// 20260319 ZJH 测试5：提交 void 返回类型的 lambda，通过 future.get() 验证执行完成
TEST(ThreadPoolTest, VoidTask) {
    // 20260319 ZJH 创建 2 线程的线程池
    df::ThreadPool pool(2);

    // 20260319 ZJH 使用原子标志跟踪 void 任务是否被执行
    std::atomic<bool> bExecuted{false};

    // 20260319 ZJH 提交 void lambda：执行后将 bExecuted 置为 true
    auto fut = pool.submit([&bExecuted]() -> void {
        bExecuted.store(true, std::memory_order_release);  // 标记已执行
    });

    // 20260319 ZJH fut.get() 对 void future 阻塞等待任务完成，不返回值
    // 若任务中有异常，get() 会重新抛出
    fut.get();

    // 20260319 ZJH 验证 void 任务确实已被执行
    EXPECT_TRUE(bExecuted.load(std::memory_order_acquire));
}
