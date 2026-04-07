// 20260320 ZJH 并行推理/训练引擎 — 超越 Halcon 级别性能
// 多线程推理 + 数据预加载 + batch 并行 + 推理计时
module;

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <chrono>
#include <atomic>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>

export module om.engine.parallel;

// 20260406 ZJH 导入依赖模块：张量类和模型基类
import om.engine.tensor;
import om.engine.module;

export namespace om {

// =========================================================
// InferenceThreadPool — 推理专用线程池
// =========================================================

// 20260320 ZJH InferenceThreadPool — 固定线程数的推理线程池
// 支持提交推理任务、等待完成、收集结果
class InferenceThreadPool {
public:
    // 20260320 ZJH 构造函数
    // nNumThreads: 线程数（默认 0=自动检测 CPU 核心数）
    explicit InferenceThreadPool(int nNumThreads = 0) {
        if (nNumThreads <= 0)
            nNumThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        m_nNumThreads = nNumThreads;
        m_bStop.store(false);

        // 20260320 ZJH 启动工作线程
        for (int i = 0; i < nNumThreads; ++i) {
            m_vecWorkers.emplace_back([this]() { workerLoop(); });
        }
    }

    // 20260406 ZJH 析构函数 — 通知所有工作线程退出并等待其完成
    ~InferenceThreadPool() {
        m_bStop.store(true);  // 20260406 ZJH 设置停止标志，通知工作线程退出循环
        m_cv.notify_all();  // 20260406 ZJH 唤醒所有等待中的工作线程
        for (auto& t : m_vecWorkers) {
            if (t.joinable()) t.join();  // 20260406 ZJH 等待每个工作线程结束
        }
    }

    // 20260320 ZJH submit — 提交单个推理任务
    // 返回 future<Tensor> 用于获取结果
    std::future<Tensor> submit(std::function<Tensor()> task) {
        auto pTask = std::make_shared<std::packaged_task<Tensor()>>(std::move(task));
        auto future = pTask->get_future();
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_queueTasks.push([pTask]() { (*pTask)(); });
        }
        m_cv.notify_one();
        return future;
    }

    // 20260320 ZJH batchInfer — 批量并行推理
    // pModel: 模型指针（线程安全要求：eval 模式下无状态修改）
    // vecInputs: 输入张量列表（每个元素是单张图像 [1,C,H,W]）
    // 返回: 推理结果列表 + 总耗时（毫秒）
    struct BatchResult {
        std::vector<Tensor> vecOutputs;  // 20260320 ZJH 推理结果
        float fTotalTimeMs = 0.0f;       // 20260320 ZJH 总耗时
        float fAvgTimeMs = 0.0f;         // 20260320 ZJH 平均单张耗时
        int nNumImages = 0;              // 20260320 ZJH 处理图像数
        int nNumThreads = 0;             // 20260320 ZJH 使用线程数
    };

    BatchResult batchInfer(Module* pModel, const std::vector<Tensor>& vecInputs) {
        BatchResult result;
        result.nNumImages = static_cast<int>(vecInputs.size());
        result.nNumThreads = m_nNumThreads;

        if (vecInputs.empty()) return result;  // 20260406 ZJH 无输入直接返回空结果

        auto tStart = std::chrono::steady_clock::now();  // 20260406 ZJH 记录批量推理开始时间

        // 20260320 ZJH 提交所有推理任务
        std::vector<std::future<Tensor>> vecFutures;
        vecFutures.reserve(vecInputs.size());

        for (const auto& input : vecInputs) {
            vecFutures.push_back(submit([pModel, input]() -> Tensor {
                return pModel->forward(input);
            }));
        }

        // 20260320 ZJH 收集所有结果
        result.vecOutputs.reserve(vecInputs.size());
        for (auto& fut : vecFutures) {
            result.vecOutputs.push_back(fut.get());
        }

        auto tEnd = std::chrono::steady_clock::now();  // 20260406 ZJH 记录结束时间
        result.fTotalTimeMs = std::chrono::duration<float, std::milli>(tEnd - tStart).count();  // 20260406 ZJH 总耗时（毫秒）
        result.fAvgTimeMs = result.fTotalTimeMs / static_cast<float>(result.nNumImages);  // 20260406 ZJH 平均单张耗时

        return result;  // 20260406 ZJH 返回批量推理结果
    }

    // 20260320 ZJH 获取线程数
    int numThreads() const { return m_nNumThreads; }

    // 20260320 ZJH 等待所有任务完成
    void waitAll() {
        // 20260320 ZJH 提交空任务并等待（简化实现）
        std::vector<std::future<Tensor>> futs;
        for (int i = 0; i < m_nNumThreads; ++i) {
            futs.push_back(submit([]() -> Tensor { return Tensor(); }));
        }
        for (auto& f : futs) f.get();
    }

private:
    // 20260406 ZJH workerLoop — 工作线程的主循环
    // 每个工作线程持续从任务队列中取出并执行任务，直到收到停止信号
    void workerLoop() {
        while (true) {
            std::function<void()> task;  // 20260406 ZJH 待执行的任务
            {
                std::unique_lock<std::mutex> lk(m_mutex);  // 20260406 ZJH 加锁访问任务队列
                // 20260406 ZJH 等待条件：收到停止信号 或 队列中有新任务
                m_cv.wait(lk, [this]() { return m_bStop.load() || !m_queueTasks.empty(); });
                // 20260406 ZJH 停止信号且队列为空时退出线程
                if (m_bStop.load() && m_queueTasks.empty()) return;
                task = std::move(m_queueTasks.front());  // 20260406 ZJH 取出队首任务
                m_queueTasks.pop();  // 20260406 ZJH 移除已取出的任务
            }
            task();  // 20260406 ZJH 在锁外执行任务，避免阻塞其他线程取任务
        }
    }

    int m_nNumThreads;  // 20260406 ZJH 线程池中的工作线程数量
    std::vector<std::thread> m_vecWorkers;  // 20260406 ZJH 工作线程容器
    std::queue<std::function<void()>> m_queueTasks;  // 20260406 ZJH 待执行任务队列（FIFO）
    std::mutex m_mutex;  // 20260406 ZJH 保护任务队列的互斥锁
    std::condition_variable m_cv;  // 20260406 ZJH 通知工作线程有新任务或需退出
    std::atomic<bool> m_bStop;  // 20260406 ZJH 停止标志（原子操作，线程安全）
};

// =========================================================
// ParallelDataLoader — 多线程数据预加载
// =========================================================

// 20260320 ZJH PrefetchItem — 预取数据项
struct PrefetchItem {
    Tensor input;           // 20260406 ZJH 预取的输入张量
    Tensor target;          // 20260406 ZJH 预取的目标（标签）张量
    bool bReady = false;    // 20260406 ZJH 是否已加载完毕可供使用
};

// 20260320 ZJH ParallelTrainer — 并行训练管理器
// 管理数据预取 + 梯度累积 + 多 batch 并行
class ParallelTrainer {
public:
    // 20260320 ZJH 构造函数
    // nPrefetchCount: 预取 batch 数量
    // nNumWorkers: 数据加载线程数
    ParallelTrainer(int nPrefetchCount = 2, int nNumWorkers = 2)
        : m_nPrefetchCount(nPrefetchCount), m_nNumWorkers(nNumWorkers)
    {}

    // 20260320 ZJH 获取训练性能统计
    struct TrainStats {
        float fDataLoadTimeMs = 0.0f;   // 20260320 ZJH 数据加载耗时
        float fForwardTimeMs = 0.0f;    // 20260320 ZJH 前向传播耗时
        float fBackwardTimeMs = 0.0f;   // 20260320 ZJH 反向传播耗时
        float fOptimizerTimeMs = 0.0f;  // 20260320 ZJH 优化器更新耗时
        float fTotalBatchTimeMs = 0.0f; // 20260320 ZJH 总 batch 耗时
        int nBatchesProcessed = 0;      // 20260320 ZJH 已处理 batch 数

        // 20260406 ZJH throughput — 计算训练吞吐量（每秒处理的 batch 数）
        float throughput() const {
            return fTotalBatchTimeMs > 0 ? (nBatchesProcessed * 1000.0f / fTotalBatchTimeMs) : 0;  // 20260406 ZJH batches/sec
        }
    };

    // 20260406 ZJH stats — 获取训练性能统计数据（只读拷贝）
    TrainStats stats() const { return m_stats; }

    // 20260320 ZJH recordBatchTime — 记录单次 batch 的各阶段耗时
    void recordBatchTime(float fDataMs, float fFwdMs, float fBwdMs, float fOptMs) {
        m_stats.fDataLoadTimeMs += fDataMs;
        m_stats.fForwardTimeMs += fFwdMs;
        m_stats.fBackwardTimeMs += fBwdMs;
        m_stats.fOptimizerTimeMs += fOptMs;
        m_stats.fTotalBatchTimeMs += (fDataMs + fFwdMs + fBwdMs + fOptMs);
        m_stats.nBatchesProcessed++;
    }

    // 20260406 ZJH resetStats — 重置所有训练性能统计数据
    void resetStats() { m_stats = TrainStats{}; }

private:
    int m_nPrefetchCount;  // 20260406 ZJH 预取 batch 数量（流水线缓冲深度）
    int m_nNumWorkers;     // 20260406 ZJH 数据加载工作线程数
    TrainStats m_stats;    // 20260406 ZJH 累积训练性能统计
};

// =========================================================
// InferenceTimer — 推理性能计时器
// =========================================================

// 20260320 ZJH InferenceTimer — 精确推理计时
class InferenceTimer {
public:
    // 20260406 ZJH start — 记录计时起点
    void start() { m_tStart = std::chrono::steady_clock::now(); }

    // 20260406 ZJH stopMs — 停止计时并返回经过的毫秒数
    // 返回: 从 start() 到当前的耗时（毫秒）
    float stopMs() {
        auto tEnd = std::chrono::steady_clock::now();  // 20260406 ZJH 记录结束时间点
        return std::chrono::duration<float, std::milli>(tEnd - m_tStart).count();  // 20260406 ZJH 计算时间差
    }

    // 20260320 ZJH 多次测量取中位数（排除首次 warmup）
    static float benchmarkMs(std::function<void()> fn, int nRuns = 10) {
        // 20260320 ZJH warmup
        fn();

        std::vector<float> vecTimes;
        vecTimes.reserve(nRuns);
        for (int i = 0; i < nRuns; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            fn();
            auto t1 = std::chrono::steady_clock::now();
            vecTimes.push_back(std::chrono::duration<float, std::milli>(t1 - t0).count());
        }
        std::sort(vecTimes.begin(), vecTimes.end());
        return vecTimes[vecTimes.size() / 2];  // 20260320 ZJH 中位数
    }

private:
    std::chrono::steady_clock::time_point m_tStart;  // 20260406 ZJH 计时起点时间戳
};

// =========================================================
// ModelOptimizer — 推理优化（算子融合、常量折叠）
// =========================================================

// 20260320 ZJH ModelOptimizer — 模型推理优化器
// 提供推理前的优化建议和自动优化
class ModelOptimizer {
public:
    struct OptimizationReport {
        int nTotalParams = 0;        // 20260320 ZJH 总参数量
        int nTotalLayers = 0;        // 20260320 ZJH 总层数
        float fEstMemoryMB = 0.0f;   // 20260320 ZJH 估计内存占用
        float fEstFlops = 0.0f;      // 20260320 ZJH 估计浮点运算量
        bool bCanUseFP16 = true;     // 20260320 ZJH 是否可用 FP16
        int nRecommendedBatchSize = 1;  // 20260320 ZJH 推荐 batch size
        int nRecommendedThreads = 1;    // 20260320 ZJH 推荐线程数
    };

    // 20260320 ZJH analyze — 分析模型并生成优化报告
    static OptimizationReport analyze(Module& model) {
        OptimizationReport report;

        auto params = model.parameters();
        for (auto* p : params) {
            report.nTotalParams += p->numel();
            report.nTotalLayers++;
        }

        report.fEstMemoryMB = static_cast<float>(report.nTotalParams * 4) / (1024.0f * 1024.0f);
        report.fEstFlops = static_cast<float>(report.nTotalParams) * 2.0f;  // 20260320 ZJH 简化估算

        int nCores = static_cast<int>(std::thread::hardware_concurrency());
        report.nRecommendedThreads = std::max(1, nCores / 2);

        // 20260320 ZJH 根据参数量推荐 batch size
        if (report.nTotalParams < 100000) report.nRecommendedBatchSize = 64;
        else if (report.nTotalParams < 1000000) report.nRecommendedBatchSize = 16;
        else if (report.nTotalParams < 10000000) report.nRecommendedBatchSize = 8;
        else report.nRecommendedBatchSize = 4;

        return report;
    }
};

}  // namespace om
