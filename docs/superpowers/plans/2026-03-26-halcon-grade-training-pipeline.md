# Halcon 级训练流水线集成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已实现但未集成的数据增强、学习率调度、EfficientAD 专项训练、缺陷图后处理四个模块接入训练/推理流水线，达到海康/Halcon 基础对标水平

**Architecture:** TrainingSession 加载图像时调用 augmentImage()；EngineBridge::train() 每 epoch 调用 scheduler.step() 更新学习率；EfficientAD 检测时走专用蒸馏损失训练路径；InspectionPage 推理后加形态学后处理

**Tech Stack:** C++23 Modules, Qt6, 已有 df.engine.data_pipeline / df.engine.scheduler / df.engine.efficientad 模块

---

### Task 1: 数据增强接入训练流水线

**Files:**
- Modify: `src/core/training/TrainingSession.cpp:215-308` (图像加载循环)
- Modify: `src/engine/bridge/EngineBridge.h` (BridgeTrainParams 增强字段)

- [ ] **Step 1: BridgeTrainParams 添加增强开关**

在 `EngineBridge.h` 的 `BridgeTrainParams` 结构体末尾添加：
```cpp
bool bAugmentation = true;  // 20260326 ZJH 是否启用数据增强（默认开启）
```

- [ ] **Step 2: TrainingSession 导入增强模块并在图像加载时调用**

在 `TrainingSession.cpp` 的图像加载循环中（img.convertToFormat 之后、CHW 转换之前），对训练集图像调用增强：
- 随机水平翻转 (50%)
- 随机垂直翻转 (50%)
- 随机亮度/对比度抖动 (±15%)
- 随机旋转 (±10°)

使用 QImage 原生变换实现（不依赖 df.engine.data_pipeline 的 C++23 模块，避免跨模块编译问题）。

- [ ] **Step 3: TrainingSession 传递增强参数**

在 `TrainingSession.cpp` 构建 `BridgeTrainParams` 处设置 `params.bAugmentation = true`。

- [ ] **Step 4: 编译验证**

- [ ] **Step 5: Commit**

---

### Task 2: 学习率调度接入优化器

**Files:**
- Modify: `src/engine/df.engine.optimizer.ixx` (Adam/SGD 添加 setLearningRate)
- Modify: `src/engine/bridge/EngineBridge.cpp:417-698` (训练循环添加调度)

- [ ] **Step 1: Adam/SGD 添加动态学习率设置方法**

在 `df.engine.optimizer.ixx` 的 Adam 和 SGD 类中添加：
```cpp
void setLearningRate(float fNewLr) { m_fLr = fNewLr; }
float learningRate() const { return m_fLr; }
```

- [ ] **Step 2: EngineBridge::train() 添加 Cosine Annealing 调度**

在 epoch 循环开头（line ~417），每 epoch 计算新学习率：
```cpp
// Cosine Annealing: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / T_max))
float fCosLr = fLr * 0.01f + 0.5f * (fLr - fLr * 0.01f)
             * (1.0f + std::cos(3.14159265f * nEpoch / nEpochs));
if (pAdam) pAdam->setLearningRate(fCosLr);
else if (pSgd) pSgd->setLearningRate(fCosLr);
```

内联实现，不引入 scheduler 模块（避免跨模块依赖问题）。

- [ ] **Step 3: 日志输出当前学习率**

在 epoch 日志中添加 LR 显示。

- [ ] **Step 4: 编译验证**

- [ ] **Step 5: Commit**

---

### Task 3: EfficientAD 专用蒸馏训练路径

**Files:**
- Modify: `src/engine/bridge/EngineBridge.cpp:144-233` (createModel 标记模型类型)
- Modify: `src/engine/bridge/EngineBridge.cpp:298-720` (train 函数分支)
- Modify: `src/engine/bridge/EngineBridge.cpp:725-885` (infer 函数提取异常图)

- [ ] **Step 1: EngineSessionImpl 添加模型类型标记**

```cpp
bool bIsEfficientAD = false;  // 20260326 ZJH 是否为 EfficientAD 异常检测模型
```

在 createModel 的 EfficientAD 分支中设置 `m_pImpl->bIsEfficientAD = true`。

- [ ] **Step 2: train() 中 EfficientAD 走蒸馏损失路径**

在训练循环中检测 `bIsEfficientAD`：
- 冻结教师网络：`static_cast<df::EfficientAD*>(pModel)->freezeTeacher()`
- 优化器仅传入学生参数：`studentParameters()`
- 损失函数替换为：`computeDistillationLoss(tInput)` 替代 `CrossEntropyLoss`
- 标签不需要（异常检测是无监督/单类别训练）

- [ ] **Step 3: infer() 中 EfficientAD 提取空间异常图**

在推理函数中检测 `bIsEfficientAD`：
- 调用 `computeAnomalyScore(tIn)` 获取 [1, 1, H/8, W/8] 异常分数图
- 将异常分数图写入 `result.vecAnomalyMap`

- [ ] **Step 4: 编译验证**

- [ ] **Step 5: Commit**

---

### Task 4: 缺陷图形态学后处理

**Files:**
- Modify: `src/ui/pages/inspection/InspectionPage.cpp:2074-2101` (二值化缺陷图生成)

- [x] **Step 1: 自适应百分位阈值 P98**

实现：nth_element O(n) 取第98百分位，下限0.1防全黑。替代固定0.5。

- [x] **Step 2: 添加形态学开运算（腐蚀+膨胀）去噪**

3×3 min/max 滤波实现（已于 2026-03-28 完成）。

- [x] **Step 3: 添加面积过滤（去除过小区域）**

BFS 4-连通 flood-fill + 面积<0.05%清零。图像≤64px时跳过。

- [x] **Step 4: 编译验证**

qt6-debug preset: 26/26 targets 编译通过，omnimatch_app.exe 链接成功。

- [ ] **Step 5: Commit**

---

### Task 5: 集成测试 — 完整训练+推理流程验证

- [ ] **Step 1: 启动应用，创建项目，导入训练图像**
- [ ] **Step 2: 选择 ResNet-18 分类模型，开始训练**
  - 验证：日志显示学习率变化（Cosine Annealing）
  - 验证：不崩溃，训练正常完成
- [ ] **Step 3: 加载训练好的模型，导入测试图像，运行推理**
  - 验证：分类结果正确显示
  - 验证：不崩溃
- [ ] **Step 4: 选择 EfficientAD 异常检测模型，开始训练**
  - 验证：日志显示蒸馏损失（非交叉熵）
  - 验证：训练正常完成
- [ ] **Step 5: EfficientAD 推理**
  - 验证：二值化缺陷图正确显示异常区域
  - 验证：形态学后处理去除了噪点
- [ ] **Step 6: Commit 全部修改**
