# 20260319 ZJH DeepForge 顶层 CMake 配置
# 纯 C++ 全流程深度学习视觉平台
cmake_minimum_required(VERSION 3.30)

project(DeepForge
    VERSION 0.1.0
    DESCRIPTION "Pure C++ Deep Learning Vision Platform"
    LANGUAGES CXX
)

# ---------- C++23 标准 ----------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- 输出目录 ----------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------- 编译选项 ----------
if(MSVC)
    add_compile_options(/W4 /utf-8 /Zc:__cplusplus)
endif()

# ---------- 条件编译选项 ----------
option(DF_ENABLE_CUDA "Enable CUDA backend" OFF)
option(DF_ENABLE_OPENCL "Enable OpenCL backend" OFF)
option(DF_ENABLE_TENSORRT "Enable TensorRT inference" OFF)
option(DF_BUILD_TESTS "Build unit tests" ON)

# ---------- CUDA（Phase 2 启用） ----------
if(DF_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---------- vcpkg 依赖 ----------
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

# ---------- 平台层静态库（Layer 1） ----------
add_library(df_platform STATIC)

target_sources(df_platform
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/platform/df.platform.logger.ixx
            src/platform/df.platform.config.ixx
            src/platform/df.platform.filesystem.ixx
            src/platform/df.platform.memory.ixx
            src/platform/df.platform.thread_pool.ixx
            src/platform/df.platform.database.ixx
)

target_include_directories(df_platform PUBLIC include)

target_link_libraries(df_platform
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        unofficial::sqlite3::sqlite3
)

# ---------- HAL 层静态库（Layer 2） ----------
add_library(df_hal STATIC)

target_sources(df_hal
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/hal/df.hal.cpu_backend.ixx
)

target_include_directories(df_hal PUBLIC include)
target_link_libraries(df_hal PUBLIC df_platform)

# ---------- 引擎层静态库（Layer 3） ----------
add_library(df_engine STATIC)

target_sources(df_engine
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/engine/df.engine.tensor_storage.ixx
            src/engine/df.engine.tensor.ixx
            src/engine/df.engine.autograd.ixx
            src/engine/df.engine.tensor_ops.ixx
            src/engine/df.engine.module.ixx
            src/engine/df.engine.linear.ixx
            src/engine/df.engine.activations.ixx
            src/engine/df.engine.optimizer.ixx
            src/engine/df.engine.loss.ixx
            src/engine/df.engine.conv.ixx
            src/engine/df.engine.serializer.ixx
            src/engine/df.engine.mnist.ixx
)

target_include_directories(df_engine PUBLIC include)
target_link_libraries(df_engine PUBLIC df_hal)

# ---------- 训练可执行文件 ----------
# 20260319 ZJH Phase 1D: MNIST MLP 训练程序
add_executable(deepforge_train src/main.cpp)
target_link_libraries(deepforge_train PRIVATE df_engine)

# ---------- 测试 ----------
if(DF_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)

    # 辅助函数：添加测试可执行文件
    function(df_add_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE df_platform GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    df_add_test(test_logger tests/test_logger.cpp)
    df_add_test(test_config tests/test_config.cpp)
    df_add_test(test_filesystem tests/test_filesystem.cpp)
    df_add_test(test_memory tests/test_memory.cpp)
    df_add_test(test_thread_pool tests/test_thread_pool.cpp)
    df_add_test(test_database tests/test_database.cpp)

    # 20260319 ZJH 引擎层测试需要链接 df_engine
    function(df_add_engine_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE df_engine GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    df_add_engine_test(test_tensor tests/test_tensor.cpp)
    df_add_engine_test(test_tensor_ops tests/test_tensor_ops.cpp)
    df_add_engine_test(test_autograd tests/test_autograd.cpp)
    df_add_engine_test(test_nn tests/test_nn.cpp)
    df_add_engine_test(test_conv tests/test_conv.cpp)
endif()
