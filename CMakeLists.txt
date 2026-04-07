# 20260330 ZJH OmniMatch v2.0.0 顶层 CMake 配置
# 纯 C++ 全流程深度学习视觉平台 — Qt6 重构版
# 20260330 ZJH 重构：拆分单体目标为子库 + install 规则 + windeployqt + CPack
cmake_minimum_required(VERSION 3.30)

project(OmniMatch
    VERSION 2.0.0
    DESCRIPTION "Pure C++ Deep Learning Vision Platform"
    LANGUAGES CXX
)

# ---------- C++23 标准 ----------
# 20260322 ZJH 使用 C++23 标准，关闭非标扩展，确保模块兼容性
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Qt6 自动化工具 ----------
# 20260322 ZJH 启用 Qt 元对象系统自动处理（MOC/RCC/UIC）
set(CMAKE_AUTOMOC ON)   # 20260322 ZJH 自动处理 Q_OBJECT 宏，生成 moc_*.cpp
set(CMAKE_AUTORCC ON)   # 20260322 ZJH 自动编译 .qrc 资源文件
set(CMAKE_AUTOUIC ON)   # 20260322 ZJH 自动处理 .ui 设计文件（预留）

# ---------- 输出目录 ----------
# 20260322 ZJH 统一可执行文件、动态库、静态库输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------- 编译选项 ----------
# 20260322 ZJH MSVC 专用编译选项
if(MSVC)
    # 20260325 ZJH 仅对 C/CXX 语言设置 MSVC 编译选项，避免传递给 nvcc (CUDA 语言)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>             # 20260322 ZJH 启用4级警告
        $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>          # 20260322 ZJH 源文件和执行字符集均使用 UTF-8
        $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus> # 20260322 ZJH 让 __cplusplus 宏返回正确标准版本号
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4251>         # 20260322 ZJH 屏蔽 DLL 接口导出相关警告
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4275>         # 20260322 ZJH 屏蔽非 DLL 接口基类警告
    )
    # 20260326 ZJH /EHa 启用异步异常处理，使 C++ catch(...) 也能捕获 SEH（访问违规/栈溢出）
    # 默认 /EHsc 只捕获 C++ throw，SEH 导致 abort() 闪退且无错误信息
    # 代价：轻微性能损失（编译器不能假设非 C++ 代码不抛异常），但对调试和稳定性至关重要
    string(REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    # 20260322 ZJH 禁止 Windows.h 定义 min/max 宏，避免与 std::min/max 冲突
    add_compile_definitions(NOMINMAX)
endif()

# ---------- 条件编译选项 ----------
# 20260322 ZJH 各后端加速开关，默认关闭
option(OM_ENABLE_CUDA     "Enable CUDA backend"         ON)
option(OM_ENABLE_CUBLAS   "Enable cuBLAS for GEMM acceleration (requires CUDA)" ON)
option(OM_ENABLE_CUDNN    "Enable cuDNN for Conv2d/BatchNorm acceleration (requires CUDA)" ON)
option(OM_ENABLE_OPENCL   "Enable OpenCL backend"       OFF)
option(OM_ENABLE_TENSORRT    "Enable TensorRT inference"       OFF)
option(OM_ENABLE_ONNXRUNTIME "Enable ONNX Runtime inference"  OFF)
option(OM_ENABLE_OPENVINO    "Enable OpenVINO inference"      OFF)
option(OM_BUILD_TESTS        "Build unit tests"                ON)

# ---------- CUDA（可选后端） ----------
# 20260322 ZJH CUDA 内核静态库，仅在 OM_ENABLE_CUDA=ON 时编译
if(OM_ENABLE_CUDA)
    # 20260324 ZJH 检测 CUDA 编译器可用性
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # 20260325 ZJH CUDA 12.8 尚不支持 VS 2026 (MSVC 14.50)，需要跳过版本检查
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        # 20260324 ZJH 覆盖 Pascal(60) 到 Hopper(90) 全部主流架构
        # Pascal(60/61): GTX 1060/1080; Volta(70): V100; Turing(75): RTX 2060/2080;
        # Ampere(80/86): A100/RTX 3060/3090; Ada(89): RTX 4060/4090; Hopper(90): H100
        # 20260324 ZJH 覆盖 Turing(75) 到 Ada(89) 主流架构
        # 20260402 ZJH 临时移除 SM60/70(Pascal/Volta) 和 SM90(Hopper) 减少编译时间
        #   RTX 3080 Ti = SM86，当前硬件完全覆盖
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
        find_package(CUDAToolkit REQUIRED)

        add_library(om_cuda STATIC
            src/cuda/cuda_kernels.cu
        )
        target_include_directories(om_cuda PUBLIC src/cuda)
        target_link_libraries(om_cuda PUBLIC CUDA::cudart)
        # 20260330 ZJH cuBLAS 集成：替代手写 GEMM kernel，提供 Tensor Core 加速
        # cuBLAS 是 CUDA Toolkit 自带库，find_package(CUDAToolkit) 已自动提供 CUDA::cublas
        if(OM_ENABLE_CUBLAS)
            target_link_libraries(om_cuda PUBLIC CUDA::cublas)
            target_compile_definitions(om_cuda PRIVATE OM_USE_CUBLAS)
            message(STATUS "cuBLAS enabled: GEMM operations will use cublasSgemm")
        endif()
        # 20260330 ZJH cuDNN 集成：替代手写 Conv2d/BatchNorm kernel，提供 Winograd/FFT 加速
        # cuDNN 是 CUDA Toolkit 自带库（需安装 cuDNN 组件），find_package(CUDAToolkit) 自动提供 CUDA::cudnn
        if(OM_ENABLE_CUDNN)
            # 20260330 ZJH cuDNN 可能未安装，检查 target 是否存在后再链接
            if(TARGET CUDA::cudnn)
                target_link_libraries(om_cuda PUBLIC CUDA::cudnn)
                target_compile_definitions(om_cuda PRIVATE OM_USE_CUDNN)
                message(STATUS "cuDNN enabled: Conv2d/BatchNorm will use cuDNN optimized implementations")
            else()
                message(STATUS "cuDNN not found: falling back to hand-written Conv2d/BatchNorm kernels")
            endif()
        endif()
        # 20260325 ZJH CUDA 目标属性：隔离编译标志，避免 CXX 标志传递给 nvcc
        set_target_properties(om_cuda PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION ON
        )
        # 20260325 ZJH 移除从全局继承的 CXX 标志（nvcc 不识别 /W4 /utf-8 /Zc:__cplusplus）
        # 使用 -Xcompiler 包装 MSVC 标志传给 host compiler
        target_compile_options(om_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --allow-unsupported-compiler
                -Xcompiler=/W3
                -Xcompiler=/utf-8
                -Wno-deprecated-gpu-targets
            >
        )
        # 20260402 ZJH [修复] CUDA 12.8 + MSVC 18 兼容性: cuda_bf16.hpp inline asm
        # 在 x64 下用 32 位 "r" 约束传递 64 位指针导致编译错误
        # --diag-suppress=bad_asm_operand_size 抑制该警告/错误
        # 额外定义宏减少 bf16 code path 触发
        target_compile_definitions(om_cuda PRIVATE
            __CUDA_NO_BFLOAT16_CONVERSIONS__
            __CUDA_NO_BFLOAT162_CONVERSIONS__
            __CUDA_NO_BFLOAT16_OPERATORS__
            __CUDA_NO_BFLOAT162_OPERATORS__
        )
    else()
        message(WARNING "CUDA compiler not found, OM_ENABLE_CUDA ignored")
        set(OM_ENABLE_CUDA OFF)
    endif()
endif()

# ---------- vcpkg 依赖 ----------
# 20260322 ZJH 基础工具库：日志/JSON/SQLite
find_package(spdlog           CONFIG REQUIRED)
find_package(nlohmann_json    CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

# ---------- Qt6 依赖 ----------
# 20260322 ZJH Qt6 UI 框架：Widgets 主框架、Charts 图表、Svg 矢量图、Concurrent 多线程
find_package(Qt6 REQUIRED COMPONENTS
    Widgets     # 20260322 ZJH Qt 窗口/控件核心
    Charts      # 20260322 ZJH 训练曲线/指标图表
    Svg         # 20260322 ZJH SVG 图标支持
    Concurrent  # 20260322 ZJH QtConcurrent 异步任务
    Network     # 20260330 ZJH UpdateChecker 网络请求（检查更新）
)

# ---------- OpenMP 多线程并行化 ----------
# 20260324 ZJH 为 CPU 后端内核提供 OpenMP 并行支持
# matmul/conv2d/batchnorm/pooling/激活函数等计算密集型操作均已使用 #pragma omp parallel
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found, CPU backend will run single-threaded")
endif()

# ---------- 平台层静态库（Layer 1） ----------
# 20260322 ZJH 日志/配置/文件系统/内存/线程池/数据库抽象
add_library(om_platform STATIC)

target_sources(om_platform
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/platform/om.platform.logger.ixx
            src/platform/om.platform.config.ixx
            src/platform/om.platform.filesystem.ixx
            src/platform/om.platform.memory.ixx
            src/platform/om.platform.thread_pool.ixx
            src/platform/om.platform.database.ixx
)

target_include_directories(om_platform PUBLIC include)

target_link_libraries(om_platform
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        unofficial::sqlite3::sqlite3
)

# ---------- HAL 层静态库（Layer 2） ----------
# 20260322 ZJH 硬件抽象层：CPU 后端 + SIMD 加速
add_library(om_hal STATIC)

target_sources(om_hal
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/hal/om.hal.cpu_backend.ixx
            src/hal/om.hal.simd.ixx
)

target_include_directories(om_hal PUBLIC include)
target_link_libraries(om_hal PUBLIC om_platform)

# 20260324 ZJH 链接 OpenMP 库到 HAL 层，使 CPU 后端的 #pragma omp 指令生效
if(OpenMP_CXX_FOUND)
    target_link_libraries(om_hal PUBLIC OpenMP::OpenMP_CXX)
endif()

# 20260324 ZJH CUDA 后端链接到 HAL 层（可选），使 CPUBackend::matmul 可调度到 GPU
# 20260325 ZJH 新增 CUDABackend 模块：镜像 CPUBackend 接口，所有运算在 GPU 上执行
if(OM_ENABLE_CUDA)
    target_sources(om_hal
        PUBLIC
            FILE_SET CXX_MODULES FILES
                src/hal/om.hal.cuda_backend.ixx
    )
    target_link_libraries(om_hal PUBLIC om_cuda)
    target_compile_definitions(om_hal PUBLIC OM_HAS_CUDA=1)
endif()

# =====================================================
# 引擎层（Layer 3）— 拆分为 3 个子库提升构建并行度
# 20260330 ZJH 原 om_engine 单体库(36+ .ixx)拆为:
#   om_engine_core   — 基础张量/自动微分/模块/优化器/损失/层
#   om_engine_vision — 模型架构(ResNet/ViT/UNet/YOLO/GAN等)
#   om_engine_tools  — 序列化/导出/数据管线/度量/训练工具
# 任何子库改动只触发该子库重编译+增量链接
# =====================================================

# ---------- om_engine_core: 基础张量/自动微分/模块/优化器/损失（核心模块） ----------
# 20260330 ZJH 最底层引擎库：张量存储/运算、自动微分、神经网络模块基类、
# 优化器(SGD/Adam)、损失函数、学习率调度、基础层(线性/卷积/激活)
add_library(om_engine_core STATIC)

target_sources(om_engine_core
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/engine/om.engine.tensor_storage.ixx  # 20260330 ZJH 张量内存管理
            src/engine/om.engine.tensor.ixx          # 20260330 ZJH 张量核心类
            src/engine/om.engine.tensor_ops.ixx      # 20260330 ZJH 张量运算（matmul/conv/pool等）
            src/engine/om.engine.autograd.ixx        # 20260330 ZJH 自动微分引擎
            src/engine/om.engine.module.ixx          # 20260330 ZJH Module 基类 + Sequential
            src/engine/om.engine.optimizer.ixx       # 20260330 ZJH SGD/Adam/AdamW 优化器
            src/engine/om.engine.loss.ixx            # 20260330 ZJH 损失函数（CE/Dice/Focal/CIoU等）
            src/engine/om.engine.scheduler.ixx       # 20260330 ZJH 学习率调度（Cosine/Warmup等）
            src/engine/om.engine.linear.ixx          # 20260330 ZJH 全连接层
            src/engine/om.engine.conv.ixx            # 20260330 ZJH 卷积层（Conv2d/DepthwiseSep等）
            src/engine/om.engine.activations.ixx     # 20260330 ZJH 激活函数（ReLU/GELU/SiLU等）
)

target_include_directories(om_engine_core PUBLIC include)
target_link_libraries(om_engine_core PUBLIC om_hal)

# 20260330 ZJH CUDA 后端链接到引擎核心层（可选）
if(OM_ENABLE_CUDA)
    target_link_libraries(om_engine_core PUBLIC om_cuda)
    target_compile_definitions(om_engine_core PUBLIC OM_HAS_CUDA=1)
endif()

# ---------- om_engine_vision: 模型架构（16 个模型模块） ----------
# 20260330 ZJH 所有深度学习模型架构：分类(ResNet/MobileNet/ViT)、分割(UNet/SegModels)、
# 检测(YOLO)、OCR(CRNN)、异常检测(AE/EfficientAD/PatchCore/ZeroShot)、
# 生成对抗(GAN)、实例分割、半精度支持
add_library(om_engine_vision STATIC)

target_sources(om_engine_vision
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/engine/om.engine.resnet.ixx          # 20260330 ZJH ResNet18/34 分类模型
            src/engine/om.engine.resnet50.ixx        # 20260330 ZJH ResNet50/101/152 深层模型
            src/engine/om.engine.mobilenet.ixx       # 20260330 ZJH MobileNetV2/V3 轻量模型
            src/engine/om.engine.vit.ixx             # 20260330 ZJH Vision Transformer
            src/engine/om.engine.convnext.ixx       # 20260402 ZJH ConvNeXt-Tiny 现代 CNN 分类
            src/engine/om.engine.unet.ixx            # 20260330 ZJH UNet 语义分割
            src/engine/om.engine.segmodels.ixx       # 20260330 ZJH DeepLabV3+/FPN 分割模型
            src/engine/om.engine.yolo.ixx            # 20260330 ZJH YOLOv7/v8/v10 目标检测
            src/engine/om.engine.crnn.ixx            # 20260330 ZJH CRNN OCR 文字识别
            src/engine/om.engine.autoencoder.ixx     # 20260330 ZJH 自编码器(异常检测基线)
            src/engine/om.engine.gan.ixx             # 20260330 ZJH 生成对抗网络
            src/engine/om.engine.instance_seg.ixx    # 20260330 ZJH 实例分割(Mask R-CNN等)
            src/engine/om.engine.efficientad.ixx     # 20260330 ZJH EfficientAD 异常检测
            src/engine/om.engine.gcad.ixx            # 20260402 ZJH GCAD 全局上下文异常检测
            src/engine/om.engine.dbnet.ixx           # 20260402 ZJH DBNet 可微分二值化文本检测
            src/engine/om.engine.edge_extraction.ixx # 20260402 ZJH DL 边缘提取（对标 Halcon）
            src/engine/om.engine.dinomaly.ixx        # 20260402 ZJH Dinomaly (CVPR 2025 SOTA AD)
            src/engine/om.engine.sam2_unet.ixx       # 20260402 ZJH SAM2-UNet 分割
            src/engine/om.engine.retrieval.ixx       # 20260402 ZJH DL 图像检索（特征嵌入+余弦相似度）
            src/engine/om.engine.sam.ixx             # 20260402 ZJH MobileSAM 无监督分割
            src/engine/om.engine.defect_generator.ixx # 20260402 ZJH AI 缺陷生成器（DRAEM+ / DDPM Tiny）
            src/engine/om.engine.patchcore.ixx       # 20260330 ZJH PatchCore 异常检测
            src/engine/om.engine.zeroshot_ad.ixx     # 20260330 ZJH 零样本异常检测
            src/engine/om.engine.zeroshot_det.ixx    # 20260330 ZJH 零样本目标检测
            src/engine/om.engine.fp16.ixx            # 20260330 ZJH FP16 半精度推理支持
            src/engine/om.engine.sod.ixx             # 20260330 ZJH SOD 大图滑窗检测
            src/engine/om.engine.pruning.ixx         # 20260330 ZJH 模型剪枝与量化
            src/engine/om.engine.fewshot.ixx         # 20260330 ZJH 少样本学习(Prototypical Network)
            src/engine/om.engine.active_learning.ixx # 20260330 ZJH 主动学习(不确定性/多样性采样)
            src/engine/om.engine.selfsup.ixx         # 20260330 ZJH 自监督预训练(SimCLR/MAE)
            src/engine/om.engine.inference_enhance.ixx # 20260330 ZJH 推理增强(Ensemble/TTA/多尺度/VAE/WBF)
            # om.engine.pretrained 已移至 om_engine_tools（需同时访问 serializer 和模型类）
)

target_include_directories(om_engine_vision PUBLIC include)
# 20260330 ZJH vision 依赖 core（所有模型都需要张量/Module/损失/优化器）
target_link_libraries(om_engine_vision PUBLIC om_engine_core)

# ---------- om_engine_tools: 工具模块（序列化/导出/数据管线/度量/训练工具） ----------
# 20260330 ZJH 训练/推理辅助工具：模型序列化(.omm)、ONNX/TensorRT导出、
# 数据增强管线、标注处理、数据集版本管理、检查点、知识蒸馏、评估指标、
# GPU训练器、并行训练、MNIST示例
add_library(om_engine_tools STATIC)

target_sources(om_engine_tools
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/engine/om.engine.serializer.ixx      # 20260330 ZJH 模型序列化(.omm格式)
            # 20260330 ZJH pretrained 已移至 om_engine_vision（需导入模型类型）
            src/engine/om.engine.onnx.ixx            # 20260330 ZJH ONNX 导出(手写protobuf)
            src/engine/om.engine.tensorrt.ixx        # 20260330 ZJH TensorRT 导出/推理
            src/engine/om.engine.data_pipeline.ixx   # 20260330 ZJH 数据增强管线
            src/engine/om.engine.data_synthesis.ixx  # 20260330 ZJH AI数据合成(对标Keyence)
            src/engine/om.engine.annotation.ixx      # 20260330 ZJH 标注数据处理
            src/engine/om.engine.dataset_version.ixx # 20260330 ZJH 数据集版本管理
            src/engine/om.engine.checkpoint.ixx      # 20260330 ZJH 训练检查点管理
            src/engine/om.engine.distillation.ixx    # 20260330 ZJH 知识蒸馏
            src/engine/om.engine.metrics.ixx         # 20260330 ZJH 评估指标(mIoU/mAP等)
            src/engine/om.engine.gpu_trainer.ixx     # 20260330 ZJH GPU 训练器
            src/engine/om.engine.parallel.ixx        # 20260330 ZJH 并行训练(数据并行)
            src/engine/om.engine.hypersearch.ixx     # 20260330 ZJH 超参数搜索(Grid/Random/Bayesian)
            src/engine/om.engine.distributed.ixx     # 20260330 ZJH 分布式训练(DataParallel+梯度累积)
            src/engine/om.engine.gradcam.ixx         # 20260330 ZJH GradCAM/GradCAM++ 注意力可视化
            src/engine/om.engine.mnist.ixx           # 20260330 ZJH MNIST 示例数据集
            src/engine/om.engine.advanced_learning.ixx  # 20260330 ZJH 高级学习范式(半监督/持续学习/数据质量/AB测试)
            src/engine/om.engine.production_monitor.ixx # 20260330 ZJH 生产监控(余裕跟踪/漂移检测/SPC)
            src/engine/om.engine.hybrid_judge.ixx       # 20260330 ZJH 混合判定(AI+传统+规则融合)
            src/engine/om.engine.pretrained.ixx        # 20260330 ZJH 预训练权重管理(需 serializer+模型类)
)

target_include_directories(om_engine_tools PUBLIC include)
# 20260330 ZJH tools 依赖 core（序列化/导出需要张量和Module）
# tools 不直接依赖 vision — 序列化通过 Module 基类接口操作，无需知道具体模型类型
# 20260330 ZJH tools 依赖 core + vision: pretrained 模块需要导入模型类(resnet/unet等)和序列化器
target_link_libraries(om_engine_tools PUBLIC om_engine_core om_engine_vision)

# ---------- om_engine: 聚合引擎接口库（兼容旧的链接目标） ----------
# 20260330 ZJH 保留 om_engine 作为 INTERFACE 库，聚合三个子库
# 旧代码中 target_link_libraries(... om_engine) 无需修改即可正常工作
add_library(om_engine INTERFACE)
target_link_libraries(om_engine INTERFACE
    om_engine_core
    om_engine_vision
    om_engine_tools
)

# ---------- 引擎桥接层静态库（Layer 3.5） ----------
# 20260323 ZJH 将 C++23 模块引擎封装为传统 C++ 接口
# 独立编译，避免 C++23 import 与 Qt 传统 #include 在同一翻译单元冲突
add_library(om_bridge STATIC
    src/engine/bridge/EngineBridge.h
    src/engine/bridge/EngineBridge.cpp
    src/engine/bridge/ModelRegistry.h
    src/engine/bridge/ModelRegistry.cpp
    # 20260330 ZJH ONNX Runtime 推理引擎（PIMPL 隔离 ORT SDK 头文件）
    src/engine/OnnxRuntimeInference.h
    src/engine/OnnxRuntimeInference.cpp
    # 20260330 ZJH OpenVINO 推理引擎（PIMPL 隔离 OpenVINO SDK 头文件）
    src/engine/OvInference.h
    src/engine/OvInference.cpp
)
target_include_directories(om_bridge PUBLIC src)
target_link_libraries(om_bridge PUBLIC om_engine)
# 20260404 ZJH [修复] 传递 OM_HAS_CUDA 到引擎桥接层
# EngineBridge.cpp 中的 GPU 迁移代码被 #ifdef OM_HAS_CUDA 保护，缺少定义则模型不会迁移到 GPU
if(OM_ENABLE_CUDA)
    target_compile_definitions(om_bridge PRIVATE OM_HAS_CUDA=1)
    target_link_libraries(om_bridge PUBLIC om_cuda)
endif()

# 20260330 ZJH ONNX Runtime 推理集成（可选）
# 启用条件：-DOM_ENABLE_ONNXRUNTIME=ON 且系统中已安装 onnxruntime
# 安装方式：vcpkg install onnxruntime 或从 GitHub releases 下载预编译包
# 未启用时 OnnxRuntimeInference::loadModel() 返回 false 并输出提示信息
if(OM_ENABLE_ONNXRUNTIME)
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        target_link_libraries(om_bridge PRIVATE onnxruntime::onnxruntime)
        target_compile_definitions(om_bridge PRIVATE OM_HAS_ONNXRUNTIME)
        message(STATUS "ONNX Runtime enabled: inference via onnxruntime EP (CPU/CUDA)")
    else()
        # 20260330 ZJH 尝试手动查找（用户可能通过 ONNXRUNTIME_ROOT 指定路径）
        find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            HINTS $ENV{ONNXRUNTIME_ROOT}/include
                  ${ONNXRUNTIME_ROOT}/include
        )
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            HINTS $ENV{ONNXRUNTIME_ROOT}/lib
                  ${ONNXRUNTIME_ROOT}/lib
        )
        if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
            target_include_directories(om_bridge PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
            target_link_libraries(om_bridge PRIVATE ${ONNXRUNTIME_LIBRARY})
            target_compile_definitions(om_bridge PRIVATE OM_HAS_ONNXRUNTIME)
            message(STATUS "ONNX Runtime enabled (manual): ${ONNXRUNTIME_LIBRARY}")
        else()
            message(WARNING "OM_ENABLE_ONNXRUNTIME=ON but onnxruntime not found. "
                            "Set ONNXRUNTIME_ROOT or install via vcpkg. "
                            "OnnxRuntimeInference will be a no-op stub.")
        endif()
    endif()
endif()

# 20260330 ZJH OpenVINO 推理集成（可选）
# 启用条件：-DOM_ENABLE_OPENVINO=ON 且系统中已安装 OpenVINO
# 安装方式：vcpkg install openvino 或从 Intel 官网下载 OpenVINO toolkit
# 未启用时 OpenVINOInference::loadModel() 返回 false 并输出提示信息
if(OM_ENABLE_OPENVINO)
    find_package(OpenVINO QUIET)
    if(OpenVINO_FOUND)
        target_link_libraries(om_bridge PRIVATE openvino::runtime)
        target_compile_definitions(om_bridge PRIVATE OM_HAS_OPENVINO)
        message(STATUS "OpenVINO enabled: inference via OpenVINO Runtime (CPU/GPU/AUTO)")
    else()
        # 20260330 ZJH 尝试手动查找（用户可能通过 OPENVINO_ROOT 指定路径）
        find_path(OPENVINO_INCLUDE_DIR
            NAMES openvino/openvino.hpp
            HINTS $ENV{OPENVINO_ROOT}/runtime/include
                  ${OPENVINO_ROOT}/runtime/include
                  $ENV{INTEL_OPENVINO_DIR}/runtime/include
        )
        find_library(OPENVINO_LIBRARY
            NAMES openvino
            HINTS $ENV{OPENVINO_ROOT}/runtime/lib/intel64
                  ${OPENVINO_ROOT}/runtime/lib/intel64
                  $ENV{INTEL_OPENVINO_DIR}/runtime/lib/intel64
        )
        if(OPENVINO_INCLUDE_DIR AND OPENVINO_LIBRARY)
            target_include_directories(om_bridge PRIVATE ${OPENVINO_INCLUDE_DIR})
            target_link_libraries(om_bridge PRIVATE ${OPENVINO_LIBRARY})
            target_compile_definitions(om_bridge PRIVATE OM_HAS_OPENVINO)
            message(STATUS "OpenVINO enabled (manual): ${OPENVINO_LIBRARY}")
        else()
            message(WARNING "OM_ENABLE_OPENVINO=ON but OpenVINO not found. "
                            "Set OPENVINO_ROOT or install OpenVINO toolkit. "
                            "OpenVINOInference will be a no-op stub.")
        endif()
    endif()
endif()

# ---------- 训练可执行文件 ----------
# 20260322 ZJH MNIST MLP/ResNet18 命令行训练程序（保留原有功能）
add_executable(omnimatch_train src/train_main.cpp)
target_link_libraries(omnimatch_train PRIVATE om_engine)

# =====================================================
# GUI 应用程序（Qt6）— 拆分为子库提升构建并行度
# 20260330 ZJH 原 omnimatch_app 80+ 文件单体目标拆为:
#   om_core       — 核心数据层（DLTypes/ImageDataset/Project等）
#   om_ui_widgets — 通用 UI 控件（图表/图形视图/标注等）
#   om_ui_dialogs — 对话框（设置/标签管理/增强预览等）
#   om_ui_pages   — 8 个工作流页面
#   omnimatch_app — 顶层可执行文件（仅 main + Application + MainWindow）
# =====================================================

# ---------- om_core: 核心数据层 ----------
# 20260330 ZJH 全局类型定义、数据模型、项目管理、训练会话、评估计算、国际化
add_library(om_core STATIC
    # 20260322 ZJH 全局类型定义（任务类型/推理后端/框架等枚举）
    src/core/DLTypes.h
    src/core/DLTypes.cpp
    # 20260330 ZJH 结构化错误码体系
    src/core/ErrorCode.h
    # 20260330 ZJH 通用 NMS 工具（标准/Soft/类别感知/Batched）
    src/core/NmsUtils.h
    # 20260322 ZJH 标签信息数据模型
    src/core/data/LabelInfo.h
    src/core/data/LabelInfo.cpp
    # 20260322 ZJH 标注数据模型
    src/core/data/Annotation.h
    src/core/data/Annotation.cpp
    # 20260322 ZJH 图像条目数据模型
    src/core/data/ImageEntry.h
    src/core/data/ImageEntry.cpp
    # 20260322 ZJH 数据集管理（QAbstractListModel，Q_OBJECT）
    src/core/data/ImageDataset.h
    src/core/data/ImageDataset.cpp
    # 20260322 ZJH 项目元数据 + 数据集持有
    src/core/project/Project.h
    src/core/project/Project.cpp
    # 20260322 ZJH 项目序列化器（Project <-> JSON .omdl 双向转换）
    src/core/project/ProjectSerializer.h
    src/core/project/ProjectSerializer.cpp
    # 20260322 ZJH 项目管理器（创建/打开/保存/关闭 + 最近项目列表，Q_OBJECT）
    src/core/project/ProjectManager.h
    src/core/project/ProjectManager.cpp
    # 20260322 ZJH 训练配置结构体（超参数/框架/模型架构/增强/导出配置）
    src/core/training/TrainingConfig.h
    # 20260322 ZJH 训练会话（后台线程执行训练循环，信号报告进度，Q_OBJECT）
    src/core/training/TrainingSession.h
    src/core/training/TrainingSession.cpp
    # 20260322 ZJH 评估结果数据结构（指标 + 混淆矩阵 + 性能统计）
    src/core/evaluation/EvaluationResult.h
    # 20260323 ZJH 评估指标计算器（精确率/召回率/F1/混淆矩阵/ROC/PR/AUC）
    src/core/evaluation/MetricsCalculator.h
    src/core/evaluation/MetricsCalculator.cpp
    # 20260323 ZJH 评估报告生成器（HTML 可视化报告 + CSV 数据表导出）
    src/core/evaluation/ReportGenerator.h
    src/core/evaluation/ReportGenerator.cpp
    # 20260323 ZJH 模型导出器（ONNX/TensorRT/OpenVINO/DFM 格式导出，Q_OBJECT）
    src/core/evaluation/ModelExporter.h
    src/core/evaluation/ModelExporter.cpp
    # 20260323 ZJH 国际化管理器（中英文内置翻译字典 + Qt QTranslator，Q_OBJECT）
    src/core/i18n/TranslationManager.h
    src/core/i18n/TranslationManager.cpp
    # 20260330 ZJH 版本更新检查器（GitHub Releases API，Q_OBJECT）
    src/core/UpdateChecker.h
    src/core/UpdateChecker.cpp
)

target_include_directories(om_core PUBLIC src)
# 20260330 ZJH 注入版本号宏，供 ReportGenerator 等使用
target_compile_definitions(om_core PRIVATE
    OM_VERSION="${PROJECT_VERSION}"
)
# 20260330 ZJH om_core 依赖 Qt6 基础模块（QObject/QAbstractListModel/QThread）
# 和 om_bridge（TrainingSession 通过桥接层调用引擎）
target_link_libraries(om_core PUBLIC
    om_bridge
    Qt6::Widgets
    Qt6::Concurrent
    Qt6::Network    # 20260330 ZJH UpdateChecker 需要 QNetworkAccessManager
)

# ---------- om_ui_widgets: 通用 UI 控件 ----------
# 20260330 ZJH 可复用控件：图像查看器、标注系统、图表、覆盖层、通知等
add_library(om_ui_widgets STATIC
    # 20260322 ZJH 可缩放图像查看器（滚轮缩放 + 拖拽平移 + 像素追踪，Q_OBJECT）
    src/ui/widgets/ZoomableGraphicsView.h
    src/ui/widgets/ZoomableGraphicsView.cpp
    # 20260322 ZJH 标注图形项（矩形/多边形绘制 + 选中手柄 + 拖拽移动）
    src/ui/widgets/AnnotationGraphicsItem.h
    src/ui/widgets/AnnotationGraphicsItem.cpp
    # 20260322 ZJH 标注撤销/重做命令（添加/删除/移动标注）
    src/ui/widgets/AnnotationCommands.h
    src/ui/widgets/AnnotationCommands.cpp
    # 20260322 ZJH 标注控制器（工具状态管理 + 鼠标事件分发 + 标注 CRUD，Q_OBJECT）
    src/ui/widgets/AnnotationController.h
    src/ui/widgets/AnnotationController.cpp
    # 20260322 ZJH 缩略图绘制代理（自定义绘制缩略图网格项，Q_OBJECT）
    src/ui/widgets/ThumbnailDelegate.h
    src/ui/widgets/ThumbnailDelegate.cpp
    # 20260322 ZJH 类别分布条形图控件（Qt Charts 水平条形图，Q_OBJECT）
    src/ui/widgets/ClassDistributionChart.h
    src/ui/widgets/ClassDistributionChart.cpp
    # 20260322 ZJH 训练损失曲线图表（Qt Charts 双线折线图，Q_OBJECT）
    src/ui/widgets/TrainingLossChart.h
    src/ui/widgets/TrainingLossChart.cpp
    # 20260322 ZJH 混淆矩阵热力图控件（QPainter 自绘 + 归一化 + 悬停高亮，Q_OBJECT）
    src/ui/widgets/ConfusionMatrixHeatmap.h
    src/ui/widgets/ConfusionMatrixHeatmap.cpp
    # 20260323 ZJH ROC/PR 曲线图表控件（QPainter 自绘多类别曲线 + AUC 标注，Q_OBJECT）
    src/ui/widgets/ROCPRCurveChart.h
    src/ui/widgets/ROCPRCurveChart.cpp
    # 20260323 ZJH 标签分布饼图控件（QPainter 自绘 + 悬停高亮 + 图例，Q_OBJECT）
    src/ui/widgets/LabelPieChart.h
    src/ui/widgets/LabelPieChart.cpp
    # 20260323 ZJH 置信度分布直方图（QPainter 自绘 + 正确/错误双色，Q_OBJECT）
    src/ui/widgets/ConfidenceHistogramChart.h
    src/ui/widgets/ConfidenceHistogramChart.cpp
    # 20260323 ZJH 半透明加载遮罩层（旋转动画 + 状态文字，Q_OBJECT）
    src/ui/widgets/LoadingOverlay.h
    src/ui/widgets/LoadingOverlay.cpp
    # 20260323 ZJH 弹出式通知控件（右上角弹出 + 自动淡出 + 四种类型，Q_OBJECT）
    src/ui/widgets/ToastNotification.h
    src/ui/widgets/ToastNotification.cpp
    # 20260323 ZJH 系统资源监控控件（CPU/内存/GPU/显存条形图，Q_OBJECT）
    src/ui/widgets/ResourceMonitorWidget.h
    src/ui/widgets/ResourceMonitorWidget.cpp
    # 20260323 ZJH 模型复杂度信息控件（参数量/FLOPs/内存/层数，Q_OBJECT）
    src/ui/widgets/ModelComplexityWidget.h
    src/ui/widgets/ModelComplexityWidget.cpp
    # 20260323 ZJH Grad-CAM 热力图叠加控件（原图 + 热力图混合 + 色彩映射，Q_OBJECT）
    src/ui/widgets/GradCAMOverlay.h
    src/ui/widgets/GradCAMOverlay.cpp
    # 20260322 ZJH 启动画面（品牌展示 + 渐隐动画）
    src/ui/widgets/SplashScreen.h
    src/ui/widgets/SplashScreen.cpp
    # 20260323 ZJH 快捷键参考面板（半透明遮罩 + 中央卡片 + 分组快捷键表格，Q_OBJECT）
    src/ui/widgets/ShortcutHelpOverlay.h
    src/ui/widgets/ShortcutHelpOverlay.cpp
)

target_include_directories(om_ui_widgets PUBLIC src)
# 20260330 ZJH 注入版本号宏，供 SplashScreen 等使用
target_compile_definitions(om_ui_widgets PRIVATE
    OM_VERSION="${PROJECT_VERSION}"
)
# 20260330 ZJH 控件依赖 om_core（数据模型）+ Qt6 Charts/Svg
target_link_libraries(om_ui_widgets PUBLIC
    om_core
    Qt6::Widgets
    Qt6::Charts
    Qt6::Svg
)

# ---------- om_ui_dialogs: 对话框 ----------
# 20260330 ZJH 模态对话框：新建项目/设置/标签管理/数据增强预览/高级训练配置/帮助
add_library(om_ui_dialogs STATIC
    # 20260322 ZJH 新建项目对话框（项目名称/任务类型/路径输入，Q_OBJECT）
    src/ui/pages/project/NewProjectDialog.h
    src/ui/pages/project/NewProjectDialog.cpp
    # 20260322 ZJH 设置对话框（常规/外观/GPU 三标签页，Q_OBJECT）
    src/ui/dialogs/SettingsDialog.h
    src/ui/dialogs/SettingsDialog.cpp
    # 20260322 ZJH 标签管理对话框（QTableWidget 列表 + 增删改 + 颜色选择，Q_OBJECT）
    src/ui/dialogs/LabelManagementDialog.h
    src/ui/dialogs/LabelManagementDialog.cpp
    # 20260323 ZJH 数据增强预览对话框（参数面板 + 3x3 网格预览 + 实时增强，Q_OBJECT）
    src/ui/dialogs/AugmentationPreviewDialog.h
    src/ui/dialogs/AugmentationPreviewDialog.cpp
    # 20260323 ZJH 高级训练配置对话框（优化器/调度器/正则化/策略，Q_OBJECT）
    src/ui/dialogs/AdvancedTrainingDialog.h
    src/ui/dialogs/AdvancedTrainingDialog.cpp
    # 20260323 ZJH 帮助对话框（使用指南/快捷键/关于，Q_OBJECT）
    src/ui/dialogs/HelpDialog.h
    src/ui/dialogs/HelpDialog.cpp
    # 20260330 ZJH AI 数据合成对话框（CopyPaste/增强/GAN 三策略合成，Q_OBJECT）
    src/ui/dialogs/DataSynthesisDialog.h
    src/ui/dialogs/DataSynthesisDialog.cpp
)

target_include_directories(om_ui_dialogs PUBLIC src)
# 20260330 ZJH 注入版本号宏，供 HelpDialog 等使用
target_compile_definitions(om_ui_dialogs PRIVATE
    OM_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(om_ui_dialogs PUBLIC
    om_core
    om_ui_widgets
    om_bridge
    Qt6::Widgets
    Qt6::Concurrent
)

# ---------- om_ui_pages: 8 个工作流页面 ----------
# 20260330 ZJH 每个页面对应一个工作流步骤：项目/图库/标注/检查/拆分/训练/评估/导出
add_library(om_ui_pages STATIC
    # 20260322 ZJH 页面基类（生命周期回调 + 三栏布局辅助）
    src/ui/pages/BasePage.h
    src/ui/pages/BasePage.cpp
    # 20260322 ZJH 项目管理页面（欢迎屏 + 项目信息双视图，Q_OBJECT）
    src/ui/pages/project/ProjectPage.h
    src/ui/pages/project/ProjectPage.cpp
    # 20260322 ZJH 图库浏览页面（缩略图网格 + 导入/过滤/搜索/拖放，Q_OBJECT）
    src/ui/pages/gallery/GalleryPage.h
    src/ui/pages/gallery/GalleryPage.cpp
    # 20260322 ZJH 图像标注页面（三栏布局 + 标注绘制 + 缩放控制，Q_OBJECT）
    src/ui/pages/image/ImagePage.h
    src/ui/pages/image/ImagePage.cpp
    # 20260322 ZJH 数据检查页面（缩略图网格 + 过滤/统计 + 预览/快速标注，Q_OBJECT）
    src/ui/pages/inspection/InspectionPage.h
    src/ui/pages/inspection/InspectionPage.cpp
    # 20260322 ZJH 数据集拆分页面（滑块联动 + 统计卡片 + 分布图表，Q_OBJECT）
    src/ui/pages/split/SplitPage.h
    src/ui/pages/split/SplitPage.cpp
    # 20260322 ZJH 训练配置页面（损失曲线 + 进度控制 + 日志，Q_OBJECT）
    src/ui/pages/training/TrainingPage.h
    src/ui/pages/training/TrainingPage.cpp
    # 20260322 ZJH 评估分析页面（混淆矩阵 + 指标卡片 + 详细表格，Q_OBJECT）
    src/ui/pages/evaluation/EvaluationPage.h
    src/ui/pages/evaluation/EvaluationPage.cpp
    # 20260322 ZJH 模型导出页面（格式配置 + 兼容性表 + 导出历史，Q_OBJECT）
    src/ui/pages/export/ExportPage.h
    src/ui/pages/export/ExportPage.cpp
)

target_include_directories(om_ui_pages PUBLIC src)
# 20260330 ZJH 注入版本号宏，供 ProjectPage 等使用
target_compile_definitions(om_ui_pages PRIVATE
    OM_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(om_ui_pages PUBLIC
    om_core
    om_ui_widgets
    om_ui_dialogs
    Qt6::Widgets
    Qt6::Charts
    Qt6::Concurrent
)
# 20260404 ZJH [修复] 传递 OM_HAS_CUDA 到 UI 层，否则 TrainingPage 的
# #ifndef OM_HAS_CUDA 会强制将 CUDA 设备回退到 CPU，导致 GPU 训练不可用
if(OM_ENABLE_CUDA)
    target_compile_definitions(om_ui_pages PRIVATE OM_HAS_CUDA=1)
endif()

# ---------- omnimatch_app: 顶层可执行文件 ----------
# 20260330 ZJH 仅包含入口(main)和顶层框架(Application/MainWindow/NavigationBar/StatusBar/ThemeManager)
add_executable(omnimatch_app
    src/main.cpp
    src/app/Application.h   # 20260322 ZJH 全局事件总线头文件（MOC 需感知 Q_OBJECT）
    src/app/Application.cpp # 20260322 ZJH 全局事件总线实现（GPU 探测 + 项目管理）
    # 20260322 ZJH 主题管理器：加载 QSS 并全局切换暗色/亮色主题
    src/app/ThemeManager.h
    src/app/ThemeManager.cpp
    # 20260322 ZJH 顶部导航栏：8 个工作流页签均分排列，带底部指示线
    src/app/NavigationBar.h
    src/app/NavigationBar.cpp
    # 20260322 ZJH 底部状态栏：消息 + 标注进度 + 进度条 + GPU 信息
    src/app/StatusBar.h
    src/app/StatusBar.cpp
    # 20260322 ZJH 主窗口：菜单栏 + 导航栏 + 页面堆叠 + 状态栏
    src/app/MainWindow.h
    src/app/MainWindow.cpp
    # 20260403 ZJH Windows 应用图标资源（嵌入 .ico 到 exe，资源管理器/任务栏显示图标）
    resources/app_icon.rc
)

# 20260322 ZJH 添加 src/ 目录到 include 搜索路径
# 使 #include "app/Application.h" 能正确解析到 src/app/Application.h
target_include_directories(omnimatch_app PRIVATE
    src
)

# 20260324 ZJH 注入版本号宏 OM_VERSION，消除源码中硬编码的 "2.0.0" 版本字符串
target_compile_definitions(omnimatch_app PRIVATE
    OM_VERSION="${PROJECT_VERSION}"
)

# 20260330 ZJH 链接拆分后的子库 + Qt6 模块
target_link_libraries(omnimatch_app PRIVATE
    om_bridge
    om_engine
    om_core
    om_ui_widgets
    om_ui_dialogs
    om_ui_pages
    Qt6::Widgets
    Qt6::Charts
    Qt6::Svg
    Qt6::Concurrent
    Qt6::Network    # 20260330 ZJH UpdateChecker 网络请求
)

# 20260322 ZJH 嵌入 Qt 资源文件（主题/图标/字体等）
# resources.qrc 自身已定义 prefix 和文件列表，直接传入 .qrc 路径
qt_add_resources(omnimatch_app "app_resources"
    PREFIX "/"
    BASE "${CMAKE_SOURCE_DIR}/resources"
    FILES
        resources/themes/dark_theme.qss
        resources/themes/light_theme.qss
        resources/icons/app_icon.svg
)

# 20260322 ZJH Windows 下暂保留控制台便于调试（发布版改为 TRUE）
# set_target_properties(omnimatch_app PROPERTIES
#     WIN32_EXECUTABLE TRUE
# )

# =====================================================
# 安装规则 + windeployqt + CPack 打包
# 20260330 ZJH 新增部署打包支持
# =====================================================

# ---------- install() 规则 ----------
# 20260330 ZJH 安装可执行文件到 bin/ 目录
install(TARGETS omnimatch_app RUNTIME DESTINATION bin)
install(TARGETS omnimatch_train RUNTIME DESTINATION bin)

# 20260330 ZJH 安装资源文件（主题/图标等）
install(DIRECTORY resources/ DESTINATION share/omnimatch/resources
    FILES_MATCHING
    PATTERN "*.qss"
    PATTERN "*.svg"
    PATTERN "*.png"
    PATTERN "*.ico"
)

# ---------- windeployqt 自动收集 Qt DLL ----------
# 20260330 ZJH 在 POST_BUILD 时自动运行 windeployqt，将 Qt DLL 复制到构建输出目录
# 这样 build/bin/ 里的 exe 可以直接双击运行，无需手动设置 PATH
find_program(WINDEPLOYQT_EXECUTABLE windeployqt
    HINTS "${Qt6_DIR}/../../../bin"
          "$ENV{QTDIR}/bin"
)

if(WINDEPLOYQT_EXECUTABLE)
    message(STATUS "windeployqt found: ${WINDEPLOYQT_EXECUTABLE}")

    # 20260330 ZJH omnimatch_app 的 windeployqt
    add_custom_command(TARGET omnimatch_app POST_BUILD
        COMMAND "${WINDEPLOYQT_EXECUTABLE}"
            --no-translations
            --no-compiler-runtime
            --no-system-d3d-compiler
            --no-opengl-sw
            $<TARGET_FILE:omnimatch_app>
        COMMENT "Running windeployqt for omnimatch_app..."
        VERBATIM
    )
else()
    message(WARNING "windeployqt not found - Qt DLLs will not be auto-deployed. "
                    "Set Qt6_DIR or QTDIR to enable automatic Qt deployment.")
endif()

# ---------- CPack NSIS 安装包 ----------
# 20260330 ZJH 使用 CPack + NSIS 生成 Windows 安装包
set(CPACK_GENERATOR "NSIS")
set(CPACK_PACKAGE_NAME "OmniMatch")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "OmniMatch")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OmniMatch Deep Learning Vision Platform")
set(CPACK_NSIS_DISPLAY_NAME "OmniMatch Deep Learning Tool ${PROJECT_VERSION}")
set(CPACK_NSIS_PACKAGE_NAME "OmniMatch ${PROJECT_VERSION}")
# 20260330 ZJH 安装包行为配置
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
set(CPACK_NSIS_MODIFY_PATH OFF)
set(CPACK_NSIS_MUI_FINISHPAGE_RUN "omnimatch_app.exe")
# 20260330 ZJH 安装包文件命名: OmniMatch-2.0.0-win64.exe
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-win64")
# 20260330 ZJH 默认安装路径
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
# 20260330 ZJH 开始菜单快捷方式
set(CPACK_NSIS_CREATE_ICONS_EXTRA
    "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\OmniMatch.lnk' '$INSTDIR\\\\bin\\\\omnimatch_app.exe'")
set(CPACK_NSIS_DELETE_ICONS_EXTRA
    "Delete '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\OmniMatch.lnk'")

include(CPack)

# ---------- 测试 ----------
# 20260322 ZJH 单元测试目标（可通过 OM_BUILD_TESTS=OFF 关闭）
if(OM_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)

    # 20260322 ZJH 辅助函数：添加平台层测试可执行文件
    function(df_add_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE om_platform GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    df_add_test(test_logger       tests/test_logger.cpp)
    df_add_test(test_config       tests/test_config.cpp)
    df_add_test(test_filesystem   tests/test_filesystem.cpp)
    df_add_test(test_memory       tests/test_memory.cpp)
    df_add_test(test_thread_pool  tests/test_thread_pool.cpp)
    df_add_test(test_database     tests/test_database.cpp)

    # 20260330 ZJH 辅助函数：添加引擎层测试可执行文件
    # 链接 om_engine（INTERFACE 聚合库）自动传递 core + vision + tools 三个子库
    function(df_add_engine_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE om_engine GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    df_add_engine_test(test_tensor      tests/test_tensor.cpp)
    df_add_engine_test(test_tensor_ops  tests/test_tensor_ops.cpp)
    df_add_engine_test(test_autograd    tests/test_autograd.cpp)
    df_add_engine_test(test_nn          tests/test_nn.cpp)
    df_add_engine_test(test_conv        tests/test_conv.cpp)
    df_add_engine_test(test_resnet      tests/test_resnet.cpp)
    df_add_engine_test(test_models      tests/test_models.cpp)
    df_add_engine_test(test_phase5      tests/test_phase5.cpp)
    df_add_engine_test(test_phase5b     tests/test_phase5b.cpp)
    df_add_engine_test(test_regression  tests/test_regression.cpp)  # 20260402 ZJH 万级回归测试

    # 20260330 ZJH 全面单元测试（暂时禁用 — 使用过时 API 如 TensorOps::add/Tensor::clone/backward）
    # 需要重写为当前引擎 API（tensorAdd/tensorSub 等自由函数，runBackward 而非 Tensor::backward）
    # TODO: 重写 test_comprehensive.cpp 适配当前 API
    # add_executable(test_comprehensive tests/test_comprehensive.cpp)
    # target_link_libraries(test_comprehensive PRIVATE om_engine om_bridge)
    # target_include_directories(test_comprehensive PRIVATE src)
    # add_test(NAME test_comprehensive COMMAND test_comprehensive)
endif()
