# 20260328 ZJH OmniMatch v2.0.0 顶层 CMake 配置
# 纯 C++ 全流程深度学习视觉平台 — Qt6 重构版
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
option(OM_ENABLE_OPENCL   "Enable OpenCL backend"       OFF)
option(OM_ENABLE_TENSORRT "Enable TensorRT inference"   OFF)
option(OM_BUILD_TESTS     "Build unit tests"             ON)

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
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
        find_package(CUDAToolkit REQUIRED)

        add_library(om_cuda STATIC
            src/cuda/cuda_kernels.cu
        )
        target_include_directories(om_cuda PUBLIC src/cuda)
        target_link_libraries(om_cuda PUBLIC CUDA::cudart)
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

# ---------- 引擎层静态库（Layer 3） ----------
# 20260322 ZJH 深度学习引擎：张量/自动微分/模型/训练/推理全套模块
add_library(om_engine STATIC)

target_sources(om_engine
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/engine/om.engine.tensor_storage.ixx
            src/engine/om.engine.tensor.ixx
            src/engine/om.engine.autograd.ixx
            src/engine/om.engine.tensor_ops.ixx
            src/engine/om.engine.module.ixx
            src/engine/om.engine.linear.ixx
            src/engine/om.engine.activations.ixx
            src/engine/om.engine.optimizer.ixx
            src/engine/om.engine.loss.ixx
            src/engine/om.engine.conv.ixx
            src/engine/om.engine.resnet.ixx
            src/engine/om.engine.unet.ixx
            src/engine/om.engine.yolo.ixx
            src/engine/om.engine.autoencoder.ixx
            src/engine/om.engine.annotation.ixx
            src/engine/om.engine.serializer.ixx
            src/engine/om.engine.mnist.ixx
            src/engine/om.engine.vit.ixx
            src/engine/om.engine.data_pipeline.ixx
            src/engine/om.engine.onnx.ixx
            src/engine/om.engine.scheduler.ixx
            src/engine/om.engine.gan.ixx
            src/engine/om.engine.metrics.ixx
            src/engine/om.engine.checkpoint.ixx
            src/engine/om.engine.segmodels.ixx
            src/engine/om.engine.parallel.ixx
            src/engine/om.engine.distillation.ixx
            src/engine/om.engine.zeroshot_ad.ixx
            src/engine/om.engine.zeroshot_det.ixx
            src/engine/om.engine.crnn.ixx
            src/engine/om.engine.instance_seg.ixx
            src/engine/om.engine.fp16.ixx
            src/engine/om.engine.tensorrt.ixx
            src/engine/om.engine.dataset_version.ixx
            src/engine/om.engine.gpu_trainer.ixx
            src/engine/om.engine.mobilenet.ixx
            src/engine/om.engine.resnet50.ixx
            src/engine/om.engine.efficientad.ixx
            src/engine/om.engine.patchcore.ixx
)

target_include_directories(om_engine PUBLIC include)
target_link_libraries(om_engine PUBLIC om_hal)

# 20260322 ZJH CUDA 后端链接（可选）
if(OM_ENABLE_CUDA)
    target_link_libraries(om_engine PUBLIC om_cuda)
    target_compile_definitions(om_engine PUBLIC OM_HAS_CUDA=1)
endif()

# ---------- 引擎桥接层静态库（Layer 3.5） ----------
# 20260323 ZJH 将 C++23 模块引擎封装为传统 C++ 接口
# 独立编译，避免 C++23 import 与 Qt 传统 #include 在同一翻译单元冲突
add_library(om_bridge STATIC
    src/engine/bridge/EngineBridge.h
    src/engine/bridge/EngineBridge.cpp
)
target_include_directories(om_bridge PUBLIC src)
target_link_libraries(om_bridge PUBLIC om_engine)

# ---------- 训练可执行文件 ----------
# 20260322 ZJH MNIST MLP/ResNet18 命令行训练程序（保留原有功能）
add_executable(omnimatch_train src/train_main.cpp)
target_link_libraries(omnimatch_train PRIVATE om_engine)

# ---------- GUI 应用程序（Qt6） ----------
# 20260322 ZJH OmniMatch 桌面 GUI 应用，基于 Qt6 Widgets 框架
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
    # 20260322 ZJH 页面基类：生命周期回调 + 三栏布局辅助
    src/ui/pages/BasePage.h
    src/ui/pages/BasePage.cpp
    # 20260322 ZJH 启动画面：品牌展示 + 渐隐动画
    src/ui/widgets/SplashScreen.h
    src/ui/widgets/SplashScreen.cpp
    # 20260322 ZJH Core 数据层：全局类型定义
    src/core/DLTypes.h
    src/core/DLTypes.cpp
    # 20260322 ZJH Core 数据层：标签信息
    src/core/data/LabelInfo.h
    src/core/data/LabelInfo.cpp
    # 20260322 ZJH Core 数据层：标注数据
    src/core/data/Annotation.h
    src/core/data/Annotation.cpp
    # 20260322 ZJH Core 数据层：图像条目
    src/core/data/ImageEntry.h
    src/core/data/ImageEntry.cpp
    # 20260322 ZJH Core 数据层：数据集管理（Q_OBJECT，MOC 需感知）
    src/core/data/ImageDataset.h
    src/core/data/ImageDataset.cpp
    # 20260322 ZJH 项目管理类：元数据 + 数据集持有
    src/core/project/Project.h
    src/core/project/Project.cpp
    # 20260322 ZJH 项目序列化器：Project ←→ JSON (.dfproj) 双向转换
    src/core/project/ProjectSerializer.h
    src/core/project/ProjectSerializer.cpp
    # 20260322 ZJH 项目管理器：创建/打开/保存/关闭 + 最近项目列表（Q_OBJECT）
    src/core/project/ProjectManager.h
    src/core/project/ProjectManager.cpp
    # 20260322 ZJH 新建项目对话框：项目名称/任务类型/路径输入（Q_OBJECT）
    src/ui/pages/project/NewProjectDialog.h
    src/ui/pages/project/NewProjectDialog.cpp
    # 20260322 ZJH 项目管理页面：欢迎屏 + 项目信息双视图（Q_OBJECT）
    src/ui/pages/project/ProjectPage.h
    src/ui/pages/project/ProjectPage.cpp
    # 20260322 ZJH 缩略图绘制代理：自定义绘制缩略图网格项（Q_OBJECT）
    src/ui/widgets/ThumbnailDelegate.h
    src/ui/widgets/ThumbnailDelegate.cpp
    # 20260322 ZJH 图库浏览页面：缩略图网格 + 导入/过滤/搜索/拖放（Q_OBJECT）
    src/ui/pages/gallery/GalleryPage.h
    src/ui/pages/gallery/GalleryPage.cpp
    # 20260322 ZJH 可缩放图像查看器：滚轮缩放 + 拖拽平移 + 像素追踪（Q_OBJECT）
    src/ui/widgets/ZoomableGraphicsView.h
    src/ui/widgets/ZoomableGraphicsView.cpp
    # 20260322 ZJH 标注图形项：矩形/多边形绘制 + 选中手柄 + 拖拽移动
    src/ui/widgets/AnnotationGraphicsItem.h
    src/ui/widgets/AnnotationGraphicsItem.cpp
    # 20260322 ZJH 标注撤销/重做命令：添加/删除/移动标注
    src/ui/widgets/AnnotationCommands.h
    src/ui/widgets/AnnotationCommands.cpp
    # 20260322 ZJH 标注控制器：工具状态管理 + 鼠标事件分发 + 标注 CRUD（Q_OBJECT）
    src/ui/widgets/AnnotationController.h
    src/ui/widgets/AnnotationController.cpp
    # 20260322 ZJH 图像标注页面：三栏布局 + 标注绘制 + 缩放控制（Q_OBJECT）
    src/ui/pages/image/ImagePage.h
    src/ui/pages/image/ImagePage.cpp
    # 20260322 ZJH 类别分布条形图控件：Qt Charts 水平条形图，显示各标签图像数量分布
    src/ui/widgets/ClassDistributionChart.h
    src/ui/widgets/ClassDistributionChart.cpp
    # 20260322 ZJH 数据集拆分页面：三栏布局 + 滑块联动 + 统计卡片 + 分布图表（Q_OBJECT）
    src/ui/pages/split/SplitPage.h
    src/ui/pages/split/SplitPage.cpp
    # 20260322 ZJH 训练损失曲线图表：Qt Charts 双线折线图，实时绘制训练/验证损失
    src/ui/widgets/TrainingLossChart.h
    src/ui/widgets/TrainingLossChart.cpp
    # 20260322 ZJH 训练配置结构体：超参数/框架/模型架构/增强/导出全部配置
    src/core/training/TrainingConfig.h
    # 20260322 ZJH 训练会话：后台线程执行模拟训练循环，信号报告进度（Q_OBJECT）
    src/core/training/TrainingSession.h
    src/core/training/TrainingSession.cpp
    # 20260322 ZJH 训练配置页面：三栏布局 + 损失曲线 + 进度控制 + 日志（Q_OBJECT）
    src/ui/pages/training/TrainingPage.h
    src/ui/pages/training/TrainingPage.cpp
    # 20260322 ZJH 评估结果数据结构：指标 + 混淆矩阵 + 性能统计
    src/core/evaluation/EvaluationResult.h
    # 20260322 ZJH 混淆矩阵热力图控件：QPainter 自绘，支持归一化和悬停高亮（Q_OBJECT）
    src/ui/widgets/ConfusionMatrixHeatmap.h
    src/ui/widgets/ConfusionMatrixHeatmap.cpp
    # 20260322 ZJH 评估分析页面：三栏布局 + 混淆矩阵 + 指标卡片 + 详细表格（Q_OBJECT）
    src/ui/pages/evaluation/EvaluationPage.h
    src/ui/pages/evaluation/EvaluationPage.cpp
    # 20260322 ZJH 模型导出页面：三栏布局 + 格式配置 + 兼容性表 + 导出历史（Q_OBJECT）
    src/ui/pages/export/ExportPage.h
    src/ui/pages/export/ExportPage.cpp
    # 20260322 ZJH 数据检查页面：三栏布局 + 缩略图网格 + 过滤/统计 + 预览/快速标注（Q_OBJECT）
    src/ui/pages/inspection/InspectionPage.h
    src/ui/pages/inspection/InspectionPage.cpp
    # 20260322 ZJH 设置对话框：三标签页（常规/外观/GPU）+ OK/Cancel/Apply（Q_OBJECT）
    src/ui/dialogs/SettingsDialog.h
    src/ui/dialogs/SettingsDialog.cpp
    # 20260322 ZJH 标签管理对话框：QTableWidget 列表 + 增删改 + 颜色选择（Q_OBJECT）
    src/ui/dialogs/LabelManagementDialog.h
    src/ui/dialogs/LabelManagementDialog.cpp
    # 20260322 ZJH 快捷键参考面板：半透明遮罩 + 中央卡片 + 分组快捷键表格（Q_OBJECT）
    src/ui/widgets/ShortcutHelpOverlay.h
    src/ui/widgets/ShortcutHelpOverlay.cpp
    # 20260323 ZJH 评估指标计算器：精确率/召回率/F1/混淆矩阵/ROC/PR曲线/AUC
    src/core/evaluation/MetricsCalculator.h
    src/core/evaluation/MetricsCalculator.cpp
    # 20260323 ZJH 评估报告生成器：HTML 可视化报告 + CSV 数据表导出
    src/core/evaluation/ReportGenerator.h
    src/core/evaluation/ReportGenerator.cpp
    # 20260323 ZJH 模型导出器：ONNX/TensorRT/OpenVINO/DFM 格式导出（Q_OBJECT）
    src/core/evaluation/ModelExporter.h
    src/core/evaluation/ModelExporter.cpp
    # 20260323 ZJH ROC/PR 曲线图表控件：QPainter 自绘多类别曲线+AUC标注（Q_OBJECT）
    src/ui/widgets/ROCPRCurveChart.h
    src/ui/widgets/ROCPRCurveChart.cpp
    # 20260323 ZJH 标签分布饼图控件：QPainter 自绘饼图+悬停高亮+图例（Q_OBJECT）
    src/ui/widgets/LabelPieChart.h
    src/ui/widgets/LabelPieChart.cpp
    # 20260323 ZJH 置信度分布直方图：QPainter 自绘直方图，支持正确/错误双色模式（Q_OBJECT）
    src/ui/widgets/ConfidenceHistogramChart.h
    src/ui/widgets/ConfidenceHistogramChart.cpp
    # 20260323 ZJH 半透明加载遮罩层：旋转动画+状态文字+父控件同步（Q_OBJECT）
    src/ui/widgets/LoadingOverlay.h
    src/ui/widgets/LoadingOverlay.cpp
    # 20260323 ZJH 弹出式通知控件：右上角弹出+自动淡出+四种类型（Q_OBJECT）
    src/ui/widgets/ToastNotification.h
    src/ui/widgets/ToastNotification.cpp
    # 20260323 ZJH 系统资源监控控件：CPU/内存/GPU/显存条形图（Q_OBJECT）
    src/ui/widgets/ResourceMonitorWidget.h
    src/ui/widgets/ResourceMonitorWidget.cpp
    # 20260323 ZJH 模型复杂度信息控件：参数量/FLOPs/内存/层数（Q_OBJECT）
    src/ui/widgets/ModelComplexityWidget.h
    src/ui/widgets/ModelComplexityWidget.cpp
    # 20260323 ZJH Grad-CAM 热力图叠加控件：原图+热力图混合+色彩映射（Q_OBJECT）
    src/ui/widgets/GradCAMOverlay.h
    src/ui/widgets/GradCAMOverlay.cpp
    # 20260323 ZJH 数据增强预览对话框：参数面板+3x3网格预览+实时增强（Q_OBJECT）
    src/ui/dialogs/AugmentationPreviewDialog.h
    src/ui/dialogs/AugmentationPreviewDialog.cpp
    # 20260323 ZJH 高级训练配置对话框：优化器/调度器/正则化/策略（Q_OBJECT）
    src/ui/dialogs/AdvancedTrainingDialog.h
    src/ui/dialogs/AdvancedTrainingDialog.cpp
    # 20260323 ZJH 帮助对话框：使用指南/快捷键/关于（Q_OBJECT）
    src/ui/dialogs/HelpDialog.h
    src/ui/dialogs/HelpDialog.cpp
    # 20260323 ZJH 国际化管理器：中英文内置翻译字典 + Qt QTranslator（Q_OBJECT）
    src/core/i18n/TranslationManager.h
    src/core/i18n/TranslationManager.cpp
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

# 20260322 ZJH 链接 Qt6 模块、引擎层和桥接层
target_link_libraries(omnimatch_app PRIVATE
    om_bridge
    om_engine
    Qt6::Widgets
    Qt6::Charts
    Qt6::Svg
    Qt6::Concurrent
)

# 20260322 ZJH 嵌入 Qt 资源文件（主题/图标/字体等）
# resources.qrc 自身已定义 prefix 和文件列表，直接传入 .qrc 路径
qt_add_resources(omnimatch_app "app_resources"
    PREFIX "/"
    BASE "${CMAKE_SOURCE_DIR}/resources"
    FILES
        resources/themes/dark_theme.qss
        resources/themes/light_theme.qss
)

# 20260322 ZJH Windows 下暂保留控制台便于调试（发布版改为 TRUE）
# set_target_properties(omnimatch_app PROPERTIES
#     WIN32_EXECUTABLE TRUE
# )

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

    # 20260322 ZJH 辅助函数：添加引擎层测试可执行文件
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
endif()
