# 开发记录

## [2026-03-19]

### 21:35 - Phase 2 Part 1 Module/Optimizer 系统完成，73 个测试通过
- **修改文件**: `src/engine/df.engine.module.ixx`, `src/engine/df.engine.linear.ixx`, `src/engine/df.engine.activations.ixx`, `src/engine/df.engine.optimizer.ixx`, `src/engine/df.engine.loss.ixx`, `tests/test_nn.cpp`, `src/main.cpp`, `CMakeLists.txt`
- **修改类型**: 新增/修改
- **修改内容**: 新增 5 个引擎层模块：Module 基类（参数管理/子模块注册/训练评估模式/递归参数遍历/梯度清零）+ Sequential 顺序容器；Linear 全连接层（Kaiming 初始化/可选偏置/y=x@W+b）；ReLU 激活模块；SGD 优化器（支持动量）+ Adam 优化器（一阶/二阶矩偏差校正）；CrossEntropyLoss + MSELoss 损失函数。编写 10 个 test_nn 测试用例（LinearForward/LinearWithBias/SequentialForward/SGDStep/AdamStep/LinearBackward/CrossEntropyForward/ModuleParameters/ZeroGrad/TrainEvalMode），连同原有 63 个测试共 73 个全部通过（10 个测试套件 100%，0.12s）。重构 main.cpp 使用 Module/Optimizer API：Sequential 构建网络、SGD 优化器绑定参数、CrossEntropyLoss 计算损失，训练循环简化为 forward/zeroGrad/backward/step 四步；合成数据训练正常收敛（loss 2.07->0.20，epoch 2 达 100% 准确率）
- **关联功能**: Phase 2 Part 1 / nn.Module 系统 / 优化器 / 损失函数

### 20:58 - Phase 1D MNIST MLP 训练完整实现，63 个测试通过
- **修改文件**: `src/hal/df.hal.cpu_backend.ixx`, `src/engine/df.engine.autograd.ixx`, `src/engine/df.engine.tensor_ops.ixx`, `src/engine/df.engine.mnist.ixx`, `src/main.cpp`, `CMakeLists.txt`
- **修改类型**: 新增/修改
- **修改内容**: CPUBackend 新增 7 个内核（relu/reluBackward/softmax/crossEntropy/crossEntropySoftmaxBackward/argmax/addBias）；autograd 新增 3 个 Backward 子类（ReLUBackward/AddBiasBackward/SoftmaxCrossEntropyBackward）；tensor_ops 新增 4 个运算函数（tensorReLU/tensorAddBias/tensorSoftmaxCrossEntropy/tensorArgmax）；新建 MNIST IDX 格式数据加载器模块（df.engine.mnist），支持图像归一化和标签 one-hot 编码；新建训练主程序 deepforge_train（两层 MLP：784->128->10，SGD 优化，batch_size=64, lr=0.01, epochs=10）；CMakeLists 添加 mnist 模块和 deepforge_train 可执行目标；9 个测试套件 63 个用例全部通过；训练程序在无 MNIST 数据时输出清晰的下载指引
- **关联功能**: Phase 1D / MNIST MLP 训练 / 首个可运行的深度学习训练程序

### 20:36 - Phase 1C AutoGrad 自动微分系统完成，63 个测试全部通过
- **修改文件**: `CMakeLists.txt`, `src/engine/df.engine.autograd.ixx`, `src/engine/df.engine.tensor.ixx`, `src/engine/df.engine.tensor_ops.ixx`, `tests/test_autograd.cpp`, `tests/test_tensor_ops.cpp`
- **修改类型**: 新增/修改
- **修改内容**: 实现动态计算图自动微分引擎；新增 autograd 模块导出 Edge、GradFunction 基类、GradAccumulator 梯度累加器、8 个 Backward 子类（Add/Sub/Mul/MatMul/AddScalar/MulScalar/Sum/LeafAccumulator）、runBackward 拓扑排序反向传播；Tensor 类新增 requiresGrad/gradFnRaw/gradAccumRaw/item 等 AutoGrad 支持接口（类型擦除避免循环依赖）；tensor_ops 集成 AutoGrad——所有算术运算（add/sub/mul/matmul/addScalar/mulScalar）在输入需要梯度时自动构建计算图；tensorSum 返回类型从 float 改为 Tensor（标量 shape={1}）以参与计算图；新增 tensorBackward/tensorGetGrad/tensorZeroGrad/tensorSetRequiresGrad 用户接口；修复 test_tensor_ops 中 tensorSum 调用（添加 .item()）；编写 8 个 AutoGrad 测试（AddGradient/SubGradient/MulGradient/MatMulGradient 数值梯度检查/MulScalarGradient/ChainRule/ZeroGrad/OnlyLeafHasGrad），连同原有 55 个测试共 63 个全部通过（9 个测试套件 100%）
- **关联功能**: Phase 1C / AutoGrad 自动微分

### 20:20 - Phase 1B 全量验证里程碑通过
- **修改文件**: `DEVLOG.md`
- **修改类型**: 记录
- **修改内容**: 执行 Phase 1B 全量重建 + ctest 验证，8 个测试套件 55 个用例全部通过（100%，总耗时 0.12s）。各套件结果：Phase 1A — test_logger 4/4、test_config 5/5、test_filesystem 6/6、test_memory 5/5、test_thread_pool 5/5、test_database 5/5（小计 30/30）；Phase 1B — test_tensor 11/11、test_tensor_ops 14/14（小计 25/25）；合计 55/55，0 失败
- **关联功能**: Phase 1B 验收里程碑

### 23:58 - Phase 1B-T5：TensorOps 运算模块实现与 14 个测试全部通过
- **修改文件**: `src/engine/df.engine.tensor_ops.ixx`, `tests/test_tensor_ops.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现 tensor_ops 模块，导出 df 命名空间下 12 个运算函数：逐元素加/减/乘/除（tensorAdd/Sub/Mul/Div）、标量加/乘（tensorAddScalar/MulScalar）、矩阵乘法（tensorMatmul，调用 CPUBackend::matmul A[M,K]*B[K,N]->C[M,N]）、零拷贝 reshape（连续张量共享 Storage 创建 makeView，非连续先 contiguous 再 reshape）、零拷贝 transpose（交换两维 shape/strides 创建视图）、零拷贝 slice（调整 offset 和指定维度大小创建视图）、全局归约 sum/max/min（先连续化再调用 CPUBackend 内核）；编写 14 个 GTest 单元测试（Add/Sub/Mul/Div/Matmul2D/MatmulNonSquare/Reshape/ReshapeFlatten/Transpose/TransposeThenContiguous/Slice/SliceDim1/ScalarOps/Reductions），全部通过（14/14，0ms total）
- **关联功能**: Phase 1B / 引擎层 TensorOps / Task 1B-T5

### 23:55 - Phase 1B-T4：Tensor 类实现与 11 个测试全部通过
- **修改文件**: `src/engine/df.engine.tensor.ixx`, `tests/test_tensor.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现 Tensor 类，采用 Storage+View 分离设计：shared_ptr<TensorStorage> 持有原始内存，Tensor 自身持有 shape / strides / offset，支持零拷贝视图；工厂方法 zeros/ones/full/randn/fromData（全部 Float32+CPU）；属性 ndim/shape/shapeVec/stride/stridesVec/numel/dtype/device/isContiguous；数据访问 floatDataPtr/mutableFloatDataPtr（非模板，规避 MSVC 跨模块模板问题）/ at / setAt；contiguous() 自身连续则返回自身，否则通过 CPUBackend::stridedCopy 复制到新连续张量；makeView/storage/offset 供 tensor_ops 使用；initContiguous 计算行主序步长（最低维=1，高维=低维步长×低维大小）；编写 11 个 GTest 单元测试全部通过（Zeros/Ones/Full/Randn/FromData/OneDimensional/ThreeDimensional/Strides/IsContiguous/AtAccess/ShapeVec，总耗时 1ms）
- **关联功能**: Phase 1B / 引擎层 Tensor 类 / Task 1B-T4

### 23:30 - Phase 1B T1-T3：CMake 配置 + TensorStorage + CPUBackend
- **修改文件**: `CMakeLists.txt`, `src/engine/df.engine.tensor_storage.ixx`, `src/engine/df.engine.tensor.ixx`, `src/engine/df.engine.tensor_ops.ixx`, `src/hal/df.hal.cpu_backend.ixx`, `tests/test_tensor.cpp`, `tests/test_tensor_ops.cpp`
- **修改类型**: 新增
- **修改内容**: CMakeLists.txt 新增 df_hal（Layer 2）和 df_engine（Layer 3）两个静态库及 df_add_engine_test 函数；TensorStorage 模块实现 64 字节对齐内存分配（MSVC _aligned_malloc / 其他 std::aligned_alloc），支持拷贝构造与禁止复制；CPUBackend 模块实现 float32 填充（zeros/ones/value/randn）、逐元素运算（add/sub/mul/div/addScalar/mulScalar）、matmul（i-k-j 顺序）、归约（sum/max/min）、连续拷贝与 stridedCopy；tensor/tensor_ops 存根模块及两个测试占位文件；全量构建 27/27 步骤全部成功，无错误无警告
- **关联功能**: Phase 1B / HAL 层 / 引擎层

### 19:45 - Phase 1A 全量构建验证通过
- **修改文件**: `config/default_config.json`
- **修改类型**: 新增
- **修改内容**: 创建默认配置文件（app / training / inference / data / ui 五个配置节）；执行全量 cmake --preset windows-debug + cmake --build 构建成功（无错误无警告）；执行 ctest -V 运行全部 30 个单元测试，100% 通过（Logger 4 + Config 5 + FileSystem 6 + Memory 5 + ThreadPool 5 + Database 5 = 30/30，总耗时 0.19s）
- **关联功能**: Phase 1A 验收 / Task 8

### 21:45 - Database 模块实现与测试通过
- **修改文件**: `src/platform/df.platform.database.ixx`, `tests/test_database.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现 SQLite RAII 封装模块；Database 类持有 sqlite3* 句柄，析构时自动调用 sqlite3_close；静态工厂方法 open(path) 返回 Result<Database>，支持 ":memory:" 内存数据库和磁盘文件数据库；open() 成功后立即执行 PRAGMA journal_mode=WAL 开启 WAL 日志模式（提升并发和崩溃恢复能力）；execute() 用于非查询语句（DDL/DML），返回 Result<void>；query() 用 sqlite3_exec + lambda 回调收集行数据，返回 Result<std::vector<Row>>，Row 为 unordered_map<string,string>，NULL 值映射为空字符串；移动构造/移动赋值均正确处理句柄转移和置空，禁止拷贝；编写 5 个 GTest 单元测试（OpenInMemory / CreateTableAndInsert / Query / FileDatabase / InvalidSQL），全部通过（5/5，51ms total）
- **关联功能**: 数据库平台层 / Task 7

### 21:15 - ThreadPool 模块实现与测试通过
- **修改文件**: `src/platform/df.platform.thread_pool.ixx`, `tests/test_thread_pool.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现固定大小线程池；使用 std::jthread（C++20）自动管理线程生命周期，析构时自动 request_stop + join；submit() 模板方法接受任意可调用对象，通过 std::packaged_task + shared_ptr 跨模块边界安全传递，返回 std::future<ReturnType>；waitAll() 通过 std::atomic<int> m_nPendingTasks 计数配合 std::condition_variable 实现无忙等阻塞；析构时设置 m_bStopping 标志并 notify_all，工作线程在队列清空后退出；编写 5 个 GTest 单元测试（SubmitAndGetResult / MultipleTasks / AllTasksExecuted / ThreadCount / VoidTask），全部通过（5/5，2ms total）
- **关联功能**: 线程池平台层 / Task 6

### 20:45 - MemoryPool 模块实现与测试通过
- **修改文件**: `src/platform/df.platform.memory.ixx`, `tests/test_memory.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现线程安全 64 字节缓存行对齐内存池；MSVC 使用 _aligned_malloc/_aligned_free，其他平台使用 std::aligned_alloc/std::free；通过 std::mutex + std::unordered_map<void*, size_t> 追踪每块分配的字节数，支持 allocatedBytes() 实时统计；allocate(0) 返回 nullptr；编写 5 个 GTest 单元测试（AllocateAndDeallocate / MemoryIsUsable / ZeroSizeReturnsNull / MultipleAllocations / Statistics），全部通过（5/5）
- **关联功能**: 内存平台层 / Task 5

### 20:15 - FileSystem 模块实现与测试通过
- **修改文件**: `src/platform/df.platform.filesystem.ixx`, `tests/test_filesystem.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现 std::filesystem + 标准 IO 文件系统封装模块；FileSystem 静态工具类提供 ensureDir（递归创建目录，幂等）、exists（路径存在性检查）、writeText/readText（文本文件覆盖写/全量读）、writeBinary/readBinary（二进制文件写/读，std::ios::binary 防止 CRLF 转换）、listFiles（非递归枚举，支持扩展名过滤）共 7 个方法；所有可失败操作返回 Result<T>，FileNotFound / InternalError / InvalidArgument 三种错误码按语义分配；编写 6 个 GTest 单元测试（EnsureDir / ReadWriteText / ReadWriteBinary / Exists / ListFiles / ReadNonExistent），全部通过（6/6）；C4834 nodiscard 警告为测试辅助调用中忽略返回值所致，无功能影响
- **关联功能**: 文件系统平台层 / Task 4

### 19:30 - Config 模块实现与测试通过
- **修改文件**: `src/platform/df.platform.config.ixx`, `tests/test_config.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现 nlohmann-json 配置封装模块；支持 string/int/double/bool 类型安全读写（模板 set/get/get+默认值）；has() 键存在检查；save() 保存到文件（4 空格缩进）；load() 静态工厂方法（含文件不存在和 JSON 格式错误两种错误路径）；编写 5 个 GTest 单元测试，全部通过（5/5）；nlohmann-json #include 置于全局模块片段，模板方法 MSVC 模块边界无兼容问题
- **关联功能**: 配置平台层 / Task 3

### 19:09 - Logger 模块实现与测试通过
- **修改文件**: `src/platform/df.platform.logger.ixx`, `tests/test_logger.cpp`
- **修改类型**: 修改（替换占位存根）
- **修改内容**: 实现 spdlog 多 sink 日志封装模块；控制台彩色 sink + 轮转文件 sink（5MB/3份）；线程安全初始化；LogLevel 枚举；Logger 静态类方法（init/setLevel/trace/debug/info/warn/error/critical）；编写 4 个 GTest 单元测试，全部通过（4/4）；修复 MSVC 模块边界 template 兼容问题，改用 std::string_view 参数；修复链接器 x86/x64 CRT 混用问题（Enter-VsDevShell -Arch amd64）
- **关联功能**: 日志平台层 / Task 2

### 当前时间 - 项目骨架初始化完成
- **修改文件**: `CMakeLists.txt`, `CMakePresets.json`, `vcpkg.json`, `include/df_types.h`, `src/platform/*.ixx`, `tests/*.cpp`, `.gitignore`
- **修改类型**: 新增
- **修改内容**: 初始化 DeepForge 项目骨架；创建 git 仓库、目录结构、CMake 配置（Ninja + MSVC + vcpkg toolchain）、C++23 模块占位文件、全局类型头文件、GTest 占位测试文件；CMake configure 与全量 build（38 步）均验证通过
- **关联功能**: 项目基础架构 / Task 1
