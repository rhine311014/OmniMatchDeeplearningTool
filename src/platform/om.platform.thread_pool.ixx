module;
// 20260319 ZJH 全局模块片段：引入标准库头文件
// C++ 模块边界不能直接包含带宏的标准头，必须在此全局片段中引入
#include <thread>               // std::jthread, std::stop_token
#include <mutex>                // std::mutex, std::unique_lock, std::lock_guard
#include <condition_variable>   // std::condition_variable
#include <functional>           // std::function
#include <future>               // std::future, std::packaged_task, std::promise
#include <queue>                // std::queue
#include <vector>               // std::vector
#include <atomic>               // std::atomic
#include <type_traits>          // std::invoke_result_t
#include <memory>               // std::make_shared
#include <stdexcept>            // std::runtime_error

export module om.platform.thread_pool;

// 20260319 ZJH 导出 df 命名空间，所有符号对模块使用者可见
export namespace om {

// 20260319 ZJH ThreadPool：基于 std::jthread 的固定大小线程池
// 设计要点：
//   1. 构造时创建固定数量工作线程，析构时自动通知并等待所有线程退出（jthread 自动 join）
//   2. submit() 接受任意可调用对象，返回 std::future<ReturnType> 供调用方等待结果
//   3. waitAll() 阻塞直到所有已提交任务执行完毕
//   4. 使用原子计数 m_nPendingTasks 跟踪待完成任务数，配合条件变量实现 waitAll
class ThreadPool {
public:
    // 20260319 ZJH 构造函数：创建 nThreads 个工作线程并立即开始等待任务
    // 参数: nThreads - 线程池中的工作线程数量，通常设为硬件并发数
    explicit ThreadPool(int nThreads)
        : m_nPendingTasks(0)  // 初始无待处理任务
        , m_bStopping(false)  // 初始未停止
    {
        // 20260319 ZJH 按指定数量创建工作线程，每个线程执行 workerLoop
        // std::jthread 在析构时自动请求停止并 join，无需手动管理生命周期
        for (int i = 0; i < nThreads; ++i) {
            // 20260319 ZJH 传入 stop_token 感知线程池停止信号，避免线程无限等待
            m_vecWorkers.emplace_back([this](std::stop_token stopToken) {
                workerLoop(stopToken);  // 每个线程进入工作循环
            });
        }
    }

    // 20260319 ZJH 析构函数：通知所有工作线程停止，然后等待它们退出
    // jthread 析构时自动调用 request_stop() 并 join()，此处只需设置停止标志并唤醒
    ~ThreadPool() {
        {
            // 20260319 ZJH 加锁设置停止标志，确保工作线程在检查队列时能看到最新状态
            std::lock_guard<std::mutex> lock(m_mutex);
            m_bStopping = true;  // 通知所有工作线程即将停止
        }
        // 20260319 ZJH 唤醒所有等待任务的工作线程，让它们检查停止标志并退出
        m_cvTask.notify_all();
        // 20260319 ZJH m_vecWorkers 析构时，每个 jthread 自动 join，确保所有线程退出后析构完成
    }

    // 20260319 ZJH 禁止拷贝和移动：线程池持有互斥量和线程，语义上不可复制/移动
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // 20260319 ZJH submit：向线程池提交一个任务，返回 std::future 以便调用方获取结果
    // 模板参数 F - 可调用对象类型；Args - 参数类型包
    // 返回: std::future<ReturnType>，调用方通过 .get() 阻塞等待结果或检查异常
    template<typename F, typename... Args>
    auto submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        // 20260319 ZJH 推导返回类型：利用 invoke_result_t 获取 F(Args...) 的返回类型
        using ReturnType = std::invoke_result_t<F, Args...>;

        // 20260319 ZJH 将 func 和 args 打包为 packaged_task，用 shared_ptr 管理生命周期
        // packaged_task 不可拷贝，必须通过 shared_ptr 在 lambda 捕获中共享所有权
        auto pTask = std::make_shared<std::packaged_task<ReturnType()>>(
            [func = std::forward<F>(func), ...capturedArgs = std::forward<Args>(args)]() mutable {
                // 20260319 ZJH 调用原始可调用对象并转发参数，执行实际计算
                return func(std::forward<Args>(capturedArgs)...);
            }
        );

        // 20260319 ZJH 从 packaged_task 取出 future，在入队之前获取，避免竞态
        std::future<ReturnType> futResult = pTask->get_future();

        {
            // 20260319 ZJH 加锁后将任务包装为 std::function<void()> 推入队列
            std::lock_guard<std::mutex> lock(m_mutex);
            // 20260319 ZJH 若线程池已在停止中，拒绝接受新任务并抛出异常
            if (m_bStopping) {
                throw std::runtime_error("ThreadPool is stopping, cannot submit new tasks");
            }
            // 20260319 ZJH 原子递增待处理任务计数，在任务入队前递增保证 waitAll 语义正确
            m_nPendingTasks.fetch_add(1, std::memory_order_relaxed);
            // 20260319 ZJH 将 packaged_task 包装为 void() 函数推入队列
            // 捕获 pTask 的 shared_ptr，保证任务对象在执行完成前不被销毁
            m_queueTasks.push([pTask]() {
                (*pTask)();  // 执行打包的任务，结果自动存入关联的 promise
            });
        }

        // 20260319 ZJH 通知一个等待中的工作线程有新任务可取
        m_cvTask.notify_one();

        return futResult;  // 返回 future，调用方持有此 future 等待任务结果
    }

    // 20260319 ZJH waitAll：阻塞当前线程，直到所有已提交的任务都执行完毕
    // 利用原子计数 m_nPendingTasks 为 0 作为完成条件
    void waitAll() {
        // 20260319 ZJH 使用条件变量等待，避免忙等（busy-wait）浪费 CPU
        std::unique_lock<std::mutex> lock(m_mutex);
        // 20260319 ZJH 循环检查待处理任务数，防止虚假唤醒（spurious wakeup）
        m_cvDone.wait(lock, [this]() {
            return m_nPendingTasks.load(std::memory_order_relaxed) == 0;
        });
    }

    // 20260319 ZJH threadCount：返回线程池中工作线程的数量
    // 返回: 构造时创建的线程数
    int threadCount() const {
        // 20260319 ZJH m_vecWorkers 在构造后不再修改，无需加锁即可安全读取
        return static_cast<int>(m_vecWorkers.size());
    }

private:
    // 20260319 ZJH workerLoop：工作线程主循环，持续从队列取任务并执行
    // 参数: stopToken - std::jthread 提供的停止令牌，用于感知线程停止请求
    void workerLoop(std::stop_token stopToken) {
        // 20260319 ZJH 工作线程持续循环，直到收到停止信号且任务队列为空
        while (true) {
            std::function<void()> task;  // 存放从队列中取出的单个任务

            {
                // 20260319 ZJH 加锁后等待：队列非空 或 线程池停止信号到来
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cvTask.wait(lock, [this, &stopToken]() {
                    // 20260319 ZJH 唤醒条件：队列中有任务可执行，或线程池正在停止
                    return !m_queueTasks.empty() || m_bStopping || stopToken.stop_requested();
                });

                // 20260319 ZJH 若队列为空且停止标志已设置，工作线程退出循环
                if (m_queueTasks.empty() && (m_bStopping || stopToken.stop_requested())) {
                    return;  // 退出工作循环，线程正常结束
                }

                // 20260319 ZJH 队列非空时取出队首任务，减少持锁时间（执行在锁外进行）
                if (!m_queueTasks.empty()) {
                    task = std::move(m_queueTasks.front());  // 取出任务，转移所有权
                    m_queueTasks.pop();                       // 从队列移除已取出的任务
                }
            }

            // 20260319 ZJH 在锁外执行任务，避免任务执行期间阻塞其他线程提交或取任务
            if (task) {
                task();  // 执行任务，结果通过 packaged_task 内部的 promise 传递给 future

                // 20260319 ZJH 任务执行完成后，原子递减待处理任务计数
                int nRemaining = m_nPendingTasks.fetch_sub(1, std::memory_order_acq_rel) - 1;
                // 20260319 ZJH 若计数降至 0，通知所有在 waitAll() 中等待的线程
                if (nRemaining == 0) {
                    m_cvDone.notify_all();  // 唤醒 waitAll() 中等待的调用线程
                }
            }
        }
    }

    // 20260319 ZJH 工作线程容器：使用 jthread 自动管理线程生命周期（析构时自动 join）
    std::vector<std::jthread> m_vecWorkers;

    // 20260319 ZJH 任务队列：存放待执行的 std::function<void()> 包装任务
    // std::queue 提供 FIFO 语义，保证任务按提交顺序被取出执行
    std::queue<std::function<void()>> m_queueTasks;

    // 20260319 ZJH 互斥量：保护 m_queueTasks 和 m_bStopping 的并发访问
    mutable std::mutex m_mutex;

    // 20260319 ZJH 任务条件变量：工作线程在此等待新任务到来或停止信号
    std::condition_variable m_cvTask;

    // 20260319 ZJH 完成条件变量：waitAll() 在此等待所有任务完成（m_nPendingTasks == 0）
    std::condition_variable m_cvDone;

    // 20260319 ZJH 待处理任务计数：submit() 时 +1，任务执行完毕时 -1
    // 原子类型保证无锁读写，fetch_sub 返回旧值用于判断是否降至 0
    std::atomic<int> m_nPendingTasks;

    // 20260319 ZJH 停止标志：析构时置 true，通知工作线程在队列清空后退出
    // 由 m_mutex 保护写操作，工作线程在持锁的条件变量等待中检查此标志
    bool m_bStopping;
};

} // namespace om
