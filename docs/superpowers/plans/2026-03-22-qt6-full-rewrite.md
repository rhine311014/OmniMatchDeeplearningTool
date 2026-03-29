# DeepForge Qt6 全面重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DeepForge 从 SDL3+ImGui 单文件 UI 完全重构为 Qt6 多文件架构，界面和功能与 OmniMatchDeepLearningTool 完全一致。

**Architecture:** 参照 OmniMatch 的四层架构（app 框架层 → ui 页面/控件层 → core 业务逻辑层 → engine 引擎层），保留 DeepForge 已有的 C++23 模块化自研引擎（tensor/autograd/CNN/LSTM/CUDA），仅重写 UI 和 core 层。使用 Qt6 Widgets + Charts + Svg 构建 8 页面工业视觉深度学习工具。

**Tech Stack:** Qt 6.10.1 (Widgets/Charts/Svg/Concurrent), C++23 Modules (.ixx), CMake 3.30+, vcpkg, MSVC 14.50

**Reference:** `E:\DevelopmentTools\OmniMatchDeepLearningTool` — 目标 UI 参考实现

---

## 阶段概述

| 阶段 | 内容 | 预估文件数 | 预估行数 |
|------|------|----------|---------|
| Phase 1 | Qt6 项目骨架 + CMake + 主窗口 + 导航栏 + 主题 | ~15 | ~3,000 |
| Phase 2 | 项目页 + Core 数据层（Project/ImageDataset/Annotation） | ~20 | ~6,000 |
| Phase 3 | 画廊页 + 图像标注页（ZoomableGraphicsView + AnnotationController） | ~20 | ~8,000 |
| Phase 4 | 拆分页 + 训练页（TrainingSession + LossChart） | ~15 | ~6,000 |
| Phase 5 | 评估页 + 导出页（ConfusionMatrix/ROC/Report） | ~15 | ~5,000 |
| Phase 6 | 检查页 + 对话框 + 主题QSS + 国际化 + 打磨 | ~20 | ~6,000 |

**总计:** ~105 个文件, ~34,000 行新代码

---

## Phase 1: Qt6 项目骨架

### Task 1.1: CMakeLists.txt 重写（Qt6 构建系统）

**Files:**
- Modify: `CMakeLists.txt` (完全重写)
- Create: `CMakePresets.json` (更新)

- [ ] **Step 1: 备份当前 CMakeLists.txt**
```bash
cp CMakeLists.txt CMakeLists.txt.sdl3.bak
```

- [ ] **Step 2: 重写 CMakeLists.txt**

关键变更：
- 移除 SDL3/ImGui/ImPlot/stb 依赖
- 添加 Qt6::Widgets, Qt6::Charts, Qt6::Svg, Qt6::Concurrent
- 启用 CMAKE_AUTOMOC / CMAKE_AUTORCC / CMAKE_AUTOUIC
- 保留 df_platform / df_hal / df_engine 静态库
- 新建 df_core 静态库（core/ 目录）
- 主目标改为 deepforge_app（Qt6 GUI）
- 资源文件 resources/resources.qrc

- [ ] **Step 3: 验证配置成功**
```bash
cmake --preset windows-release
```

- [ ] **Step 4: Commit**

---

### Task 1.2: 资源文件 + 主题 QSS

**Files:**
- Create: `resources/resources.qrc`
- Create: `resources/themes/dark_theme.qss`
- Create: `resources/themes/light_theme.qss`

- [ ] **Step 1: 创建 QRC 资源清单**
- [ ] **Step 2: 编写暗色主题 QSS**（参考 OmniMatch dark_theme.qss 风格：#1a1d23 背景, #2563eb 强调色, #e2e8f0 文字）
- [ ] **Step 3: 编写亮色主题 QSS**
- [ ] **Step 4: Commit**

---

### Task 1.3: 应用入口 + Application 单例

**Files:**
- Create: `src/app/Application.h`
- Create: `src/app/Application.cpp`
- Modify: `src/main.cpp` (重写为 Qt6 入口)

- [ ] **Step 1: Application 类** — 全局事件总线，项目管理，硬件检测
  - 单例 `instance()`
  - 当前项目 `currentProject()` / `setCurrentProject()`
  - 信号: projectCreated/Opened/Closed/Saved, imagesImported, trainingStarted/Progress/Completed, evaluationStarted/Progress/Completed

- [ ] **Step 2: main.cpp** — QApplication 初始化 → SplashScreen → ThemeManager → MainWindow
- [ ] **Step 3: 编译验证**
- [ ] **Step 4: Commit**

---

### Task 1.4: ThemeManager 主题管理器

**Files:**
- Create: `src/app/ThemeManager.h`
- Create: `src/app/ThemeManager.cpp`

- [ ] **Step 1: 实现** — 单例，applyTheme(Dark/Light)，加载 QSS 文件，发射 themeChanged 信号
- [ ] **Step 2: Commit**

---

### Task 1.5: NavigationBar 导航栏

**Files:**
- Create: `src/app/NavigationBar.h`
- Create: `src/app/NavigationBar.cpp`

- [ ] **Step 1: 实现** — QWidget，8 个 QPushButton 水平排列，互斥选中，底部蓝色指示线
  - 按钮文字: 项目 / 图库 / 图像 / 检查 / 拆分 / 训练 / 评估 / 导出
  - 信号: `pageChanged(int)`
  - `setPageEnabled(int, bool)` 控制页签可用性

- [ ] **Step 2: Commit**

---

### Task 1.6: StatusBar 状态栏

**Files:**
- Create: `src/app/StatusBar.h`
- Create: `src/app/StatusBar.cpp`

- [ ] **Step 1: 实现** — 左侧消息 + 右侧标注进度 + 隐藏进度条 + GPU/CPU 监控标签
- [ ] **Step 2: Commit**

---

### Task 1.7: MainWindow 主窗口

**Files:**
- Create: `src/app/MainWindow.h`
- Create: `src/app/MainWindow.cpp`

- [ ] **Step 1: 实现** — QMainWindow
  - 菜单栏: 文件(新建/打开/保存/关闭/退出) + 编辑(撤销/重做) + 视图(全屏/主题) + 工具(实时推理/自动标注) + 帮助(快捷键/关于)
  - 中央布局: NavigationBar + QStackedWidget(8页) + StatusBar
  - 快捷键: Ctrl+N/O/S/W/Q, Alt+1~8, F1/F5/F6/F7/F11
  - 页面切换淡入动画 (QPropertyAnimation + QGraphicsOpacityEffect)

- [ ] **Step 2: 创建 8 个占位页面 (BasePage 子类，显示页面名称)**
- [ ] **Step 3: 编译运行，确认 8 页面切换正常**
- [ ] **Step 4: Commit**

---

### Task 1.8: BasePage 页面基类

**Files:**
- Create: `src/ui/pages/BasePage.h`
- Create: `src/ui/pages/BasePage.cpp`

- [ ] **Step 1: 实现** — QWidget 基类
  - 三栏布局: setupThreeColumnLayout(left, center, right)
  - 生命周期: onEnter() / onLeave() / onProjectLoaded() / onProjectClosed()
  - QSplitter 分割器管理
  - 左右面板宽度设置

- [ ] **Step 2: Commit**

---

### Task 1.9: SplashScreen 启动动画

**Files:**
- Create: `src/ui/widgets/SplashScreen.h`
- Create: `src/ui/widgets/SplashScreen.cpp`

- [ ] **Step 1: 实现** — 圆形 Logo + "DeepForge" 标题 + 加载进度条 + 渐隐退出
- [ ] **Step 2: Commit**

---

**Phase 1 验收:** 运行 deepforge_app.exe 显示主窗口，8 页面切换正常，暗色主题生效，启动动画显示。

---

## Phase 2: 项目管理 + Core 数据层

### Task 2.1: DLTypes.h 核心类型定义

**Files:**
- Create: `src/core/DLTypes.h`

- [ ] **Step 1: 定义** — TaskType(8种), BackendType, DeviceType, PrecisionType, SplitType, ModelArchitecture(带 architecturesForTask()), OptimizerType, SchedulerType, ProjectState, PageIndex 常量

---

### Task 2.2: ImageEntry + LabelInfo + Annotation 数据模型

**Files:**
- Create: `src/core/data/ImageEntry.h/.cpp`
- Create: `src/core/data/LabelInfo.h/.cpp`
- Create: `src/core/data/Annotation.h/.cpp`

- [ ] **Step 1: ImageEntry** — UUID, 路径, 标签ID, 拆分类型, 尺寸, 标注列表
- [ ] **Step 2: LabelInfo** — ID, 名称, 颜色, 可见性
- [ ] **Step 3: Annotation** — UUID, 类型(Rect/Polygon/Mask/Text), 坐标数据, 标签关联

---

### Task 2.3: ImageDataset 数据集管理

**Files:**
- Create: `src/core/data/ImageDataset.h/.cpp`

- [ ] **Step 1: 实现** — 图像增删改查, 标签管理, 拆分管理, 过滤排序, 统计

---

### Task 2.4: Project + ProjectManager + ProjectSerializer

**Files:**
- Create: `src/core/project/Project.h/.cpp`
- Create: `src/core/project/ProjectManager.h/.cpp`
- Create: `src/core/project/ProjectSerializer.h/.cpp`

- [ ] **Step 1: Project** — 项目元数据 + ImageDataset + 训练配置 + 模型路径
- [ ] **Step 2: ProjectManager** — 新建/打开/保存/关闭, 最近项目列表
- [ ] **Step 3: ProjectSerializer** — JSON 序列化/反序列化

---

### Task 2.5: ProjectPage 项目页

**Files:**
- Create: `src/ui/pages/project/ProjectPage.h/.cpp`
- Create: `src/ui/pages/project/NewProjectDialog.h/.cpp`

- [ ] **Step 1: ProjectPage** — 欢迎屏(新建/打开/最近项目) ↔ 项目信息面板(元数据+统计+快捷按钮)
- [ ] **Step 2: NewProjectDialog** — 任务类型选择 + 项目名称 + 路径

---

### Task 2.6: SettingsManager 设置管理

**Files:**
- Create: `src/core/settings/SettingsManager.h/.cpp`

---

**Phase 2 验收:** 可以新建/保存/打开项目，项目页正确显示项目信息。

---

## Phase 3: 画廊页 + 图像标注页

### Task 3.1: ThumbnailDelegate 缩略图绘制代理

**Files:**
- Create: `src/ui/widgets/ThumbnailDelegate.h/.cpp`

- [ ] **Step 1: 实现** — QStyledItemDelegate, 绘制缩略图+文件名+标签色块+拆分标记

---

### Task 3.2: GalleryPage 画廊页

**Files:**
- Create: `src/ui/pages/gallery/GalleryPage.h/.cpp`

- [ ] **Step 1: 左面板** — 统计(总数/已标注/未标注/训练/验证/测试) + 标签过滤 + 拆分过滤
- [ ] **Step 2: 中央** — 工具栏(导入图像/导入文件夹/删除/排序/搜索/缩略图大小滑块) + QListView(IconMode)+ThumbnailDelegate
- [ ] **Step 3: 右面板(可选)** — ClassDistributionChart 类别分布条形图 + LabelPieChart 饼图
- [ ] **Step 4: 拖放导入** — dragEnterEvent/dropEvent 支持

---

### Task 3.3: ZoomableGraphicsView 可缩放图像视图

**Files:**
- Create: `src/ui/widgets/ZoomableGraphicsView.h/.cpp`

- [ ] **Step 1: 实现** — QGraphicsView, 1%~5000% 缩放, 滚轮缩放(以鼠标为中心), 左键平移, 像素值追踪(Shift+悬停), fitInView/zoomToActualSize
- [ ] **Step 2: 信号** — zoomChanged, mousePositionChanged, pixelValueChanged

---

### Task 3.4: AnnotationController + AnnotationGraphicsItem + AnnotationCommands

**Files:**
- Create: `src/ui/widgets/AnnotationController.h/.cpp`
- Create: `src/ui/widgets/AnnotationGraphicsItem.h/.cpp`
- Create: `src/ui/widgets/AnnotationCommands.h/.cpp`

- [ ] **Step 1: AnnotationGraphicsItem** — QGraphicsItem, 矩形/多边形/画笔/文字 4 种绘制, 选中高亮, 缩放手柄
- [ ] **Step 2: AnnotationCommands** — QUndoCommand 子类: AddAnnotationCommand, DeleteAnnotationCommand, MoveAnnotationCommand
- [ ] **Step 3: AnnotationController** — 工具状态机(Select/Rect/Polygon/Brush/Text), 鼠标事件处理, 撤销重做栈(QUndoStack)

---

### Task 3.5: ImagePage 图像标注页

**Files:**
- Create: `src/ui/pages/image/ImagePage.h/.cpp`

- [ ] **Step 1: 左面板** — 标签下拉+分配按钮 + 工具按钮组(5个) + 笔刷大小滑块 + 标注列表
- [ ] **Step 2: 中央** — 图像导航栏(上一张/下一张/计数/适应/实际大小/缩放百分比) + ZoomableGraphicsView
- [ ] **Step 3: 右面板** — 缩略图预览 + 缩放滑块 + 文件信息(名称/尺寸/大小/色深) + 鼠标坐标+像素值

---

**Phase 3 验收:** 可以导入图像，画廊缩略图显示，图像页标注(矩形/多边形/画笔)，撤销/重做正常。

---

## Phase 4: 拆分页 + 训练页

### Task 4.1: SplitPage 拆分页

**Files:**
- Create: `src/ui/pages/split/SplitPage.h/.cpp`

- [ ] **Step 1: 左面板** — 拆分名称 + 比例滑块(训练/验证，测试自动计算) + 分层采样 + 预设按钮(70/15/15, 80/10/10, 60/20/20) + 执行/重置
- [ ] **Step 2: 中央** — 概览卡片(训练/验证/测试数量) + 标签分布表格 + ClassDistributionChart

---

### Task 4.2: TrainingLossChart 损失曲线图

**Files:**
- Create: `src/ui/widgets/TrainingLossChart.h/.cpp`

- [ ] **Step 1: 实现** — QChart + QLineSeries, 训练损失(蓝) + 验证损失(橙), 实时更新, 平滑开关

---

### Task 4.3: TrainingSession 训练会话

**Files:**
- Create: `src/core/training/TrainingSession.h/.cpp`
- Create: `src/core/training/TrainingConfig.h`

- [ ] **Step 1: TrainingConfig** — 框架/架构/设备/优化器/调度器/超参数/增强参数
- [ ] **Step 2: TrainingSession** — QThread 工作线程, 信号: epochCompleted/batchCompleted/trainingFinished/trainingFailed, 暂停/恢复/停止
- [ ] **Step 3: 桥接 DeepForge 引擎** — 调用 df::Module/df::Adam/df::Tensor 进行训练

---

### Task 4.4: TrainingPage 训练页

**Files:**
- Create: `src/ui/pages/training/TrainingPage.h/.cpp`

- [ ] **Step 1: 左面板** — 框架选择 + 模型架构 + 设备 + 优化器 + 调度器 + 超参数 + 增强参数 + 高级配置
- [ ] **Step 2: 中央** — TrainingLossChart + 进度条 + 开始/停止/恢复按钮 + 状态显示(epoch/loss/ETA/早停计数)
- [ ] **Step 3: 右面板** — 数据统计 + 前置检查(✓/✗) + 日志文本框

---

**Phase 4 验收:** 拆分数据集，启动训练，Loss 曲线实时更新，暂停/停止正常。

---

## Phase 5: 评估页 + 导出页

### Task 5.1: EvaluationWorker + MetricsCalculator

**Files:**
- Create: `src/core/evaluation/EvaluationWorker.h/.cpp`
- Create: `src/core/evaluation/MetricsCalculator.h/.cpp`
- Create: `src/core/evaluation/EvaluationResult.h`

---

### Task 5.2: 评估可视化控件

**Files:**
- Create: `src/ui/widgets/ConfusionMatrixHeatmap.h/.cpp`
- Create: `src/ui/widgets/ROCPRCurveChart.h/.cpp`
- Create: `src/ui/widgets/EvaluationChartWidget.h/.cpp`

---

### Task 5.3: EvaluationPage 评估页

**Files:**
- Create: `src/ui/pages/evaluation/EvaluationPage.h/.cpp`

- [ ] **Step 1: 左面板** — 数据范围选择 + 阈值调节 + 运行/清除/导出按钮 + 前置检查
- [ ] **Step 2: 中央** — 指标卡片(3个) + 详细指标表格 + 混淆矩阵热力图 + ROC/PR 曲线
- [ ] **Step 3: 右面板** — 数据统计 + 性能指标(延迟/吞吐) + 日志

---

### Task 5.4: ExportPage + ReportGenerator

**Files:**
- Create: `src/ui/pages/export/ExportPage.h/.cpp`
- Create: `src/core/evaluation/ReportGenerator.h/.cpp`
- Create: `src/core/evaluation/ModelExporter.h/.cpp`

---

**Phase 5 验收:** 评估完成后显示混淆矩阵/ROC/指标，导出 ONNX/HTML 报告。

---

## Phase 6: 检查页 + 对话框 + 打磨

### Task 6.1: InspectionPage 检查页

**Files:**
- Create: `src/ui/pages/inspection/InspectionPage.h/.cpp`

---

### Task 6.2: 对话框集合

**Files:**
- Create: `src/ui/dialogs/SettingsDialog.h/.cpp`
- Create: `src/ui/dialogs/LabelManagementDialog.h/.cpp`
- Create: `src/ui/dialogs/AugmentationPreviewDialog.h/.cpp`
- Create: `src/ui/dialogs/AdvancedTrainingDialog.h/.cpp`
- Create: `src/ui/dialogs/HelpDialog.h/.cpp`

---

### Task 6.3: 辅助控件

**Files:**
- Create: `src/ui/widgets/LoadingOverlay.h/.cpp`
- Create: `src/ui/widgets/ToastNotification.h/.cpp`
- Create: `src/ui/widgets/ShortcutHelpOverlay.h/.cpp`
- Create: `src/ui/widgets/ResourceMonitorWidget.h/.cpp`
- Create: `src/ui/widgets/ModelComplexityWidget.h/.cpp`
- Create: `src/ui/widgets/ClassDistributionChart.h/.cpp`
- Create: `src/ui/widgets/LabelPieChart.h/.cpp`
- Create: `src/ui/widgets/ConfidenceHistogramChart.h/.cpp`
- Create: `src/ui/widgets/GradCAMOverlay.h/.cpp`

---

### Task 6.4: 国际化 (i18n)

**Files:**
- Create: `src/core/i18n/TranslationManager.h/.cpp`

---

### Task 6.5: 清理 + 移除旧 UI

- [ ] **Step 1: 删除** `src/ui/app_main.cpp` (SDL3+ImGui UI)
- [ ] **Step 2: 移除** SDL3/ImGui/ImPlot/stb vcpkg 依赖
- [ ] **Step 3: 更新** CMakePresets.json
- [ ] **Step 4: 全量编译 + 测试**

---

**Phase 6 验收:** 完整 8 页面功能，暗/亮主题切换，所有对话框可用，快捷键完整。

---

## DeepForge 引擎桥接策略

当前 DeepForge 引擎使用 C++23 模块 (.ixx)，Qt6 使用传统 .h/.cpp。桥接方式：

1. **df_engine 静态库保持不变**（.ixx 模块编译为 .lib）
2. **新建 `src/engine/bridge/` 目录**，用 .cpp 文件 `import df.engine.*` 桥接
3. **TrainingSession.cpp 通过 bridge 调用 df::Module/df::Adam/df::Tensor**
4. **NativeCppTrainingEngine 直接映射到 DeepForge 引擎**

```cpp
// src/engine/bridge/DeepForgeBridge.cpp
import df.engine.tensor;
import df.engine.module;
import df.engine.optimizer;
// ... 提供 C++ 普通函数接口给 Qt6 层调用
```

---

## 关键决策记录

| # | 决策 | 理由 |
|---|------|------|
| D1 | Qt6 Widgets 而非 QML | OmniMatch 使用 Widgets，工业应用更稳定 |
| D2 | 保留 C++23 模块引擎 | 自研引擎是核心资产，已验证稳定 |
| D3 | 参照 OmniMatch 但独立编写 | 避免直接复制，代码风格统一为 DeepForge 规范 |
| D4 | 8 页面架构 | MVTec DL Tool 标准工作流 |
| D5 | BasePage 三栏布局 | OmniMatch 验证过的 UI 模式 |
| D6 | Application 事件总线 | 解耦页面间通信 |
| D7 | QUndoStack 撤销重做 | Qt 内置，比手写栈更可靠 |
| D8 | 分 6 阶段交付 | 每阶段可独立验收 |
