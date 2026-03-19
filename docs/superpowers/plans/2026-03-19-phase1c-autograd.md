# Phase 1C: AutoGrad 自动微分引擎 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现动态计算图自动微分引擎，使 Tensor 运算在前向传播时自动建图，调用 `backward()` 时通过拓扑排序逆序计算梯度，为 Phase 1D 的 MLP 训练提供核心能力。

**Architecture:** 采用 PyTorch 风格的动态计算图：每个需要梯度的运算生成一个 GradFunction 节点，节点记录输入边（指向前驱 GradFunction）。`backward()` 从 loss 节点出发，拓扑排序后逆序遍历计算图，逐节点调用 `backward()` 计算梯度并累积到叶子节点的 `.grad()` 上。

**Tech Stack:** C++23 modules (.ixx), MSVC 14.50, GTest, 数值梯度检查验证正确性

**Spec Reference:** `docs/superpowers/specs/2026-03-19-deepforge-design.md` §3.2

**Dependencies:** Phase 1B (Tensor + CPUBackend) 已完成

---

## Scope

Phase 1C 实现 AutoGrad 核心机制 + MLP 所需算子的梯度：

| 实现 | 内容 |
|------|------|
| GradFunction 基类 | backward() 虚函数 + 输入边记录 |
| Tensor 扩展 | requiresGrad, grad, backward, gradFn |
| AddBackward | d/da(a+b)=1, d/db(a+b)=1 |
| SubBackward | d/da(a-b)=1, d/db(a-b)=-1 |
| MulBackward | d/da(a*b)=b, d/db(a*b)=a |
| MatMulBackward | d/dA(A@B)=grad@B^T, d/dB(A@B)=A^T@grad |
| AddScalarBackward | d/da(a+s)=1 |
| MulScalarBackward | d/da(a*s)=s |
| SumBackward | d/da(sum(a))=ones |
| 拓扑排序 + backward 引擎 | BFS/DFS 排序 + 逆序梯度传播 |
| 数值梯度检查 | 有限差分验证解析梯度 |

**不实现**：ReLU/Sigmoid/Softmax backward（Phase 2 算子层）、Div backward（MLP 不需要）。

---

## File Map

| 文件路径 | 职责 |
|---------|------|
| `src/engine/df.engine.autograd.ixx` | GradFunction 基类 + 各算子 Backward 子类 + backward 引擎 |
| `src/engine/df.engine.tensor.ixx` | **修改**：添加 m_bRequiresGrad, m_pGrad, m_pGradFn 字段和相关方法 |
| `src/engine/df.engine.tensor_ops.ixx` | **修改**：前向运算时构建计算图（创建 GradFunction 节点） |
| `tests/test_autograd.cpp` | AutoGrad 单元测试 + 数值梯度检查 |
| `CMakeLists.txt` | **修改**：添加 autograd 模块到 df_engine + 新测试 |

---

## Task 1: GradFunction 基类 + Backward 引擎 + CMake

**Files:**
- Create: `src/engine/df.engine.autograd.ixx`
- Modify: `CMakeLists.txt` — 添加 autograd 模块和测试
- Create: `tests/test_autograd.cpp` (stub)

- [ ] **Step 1: 更新 CMakeLists.txt**

在 df_engine 的 FILE_SET CXX_MODULES 中添加：
```
src/engine/df.engine.autograd.ixx
```

在测试部分添加：
```cmake
df_add_engine_test(test_autograd tests/test_autograd.cpp)
```

- [ ] **Step 2: 创建 autograd stub + test stub**

`src/engine/df.engine.autograd.ixx`:
```cpp
export module df.engine.autograd;
```

`tests/test_autograd.cpp`:
```cpp
#include <gtest/gtest.h>
TEST(AutogradStub, Placeholder) { EXPECT_TRUE(true); }
```

- [ ] **Step 3: 实现 GradFunction 基类和所有 Backward 子类**

```cpp
// 20260319 ZJH AutoGrad 模块 — 动态计算图自动微分引擎
// GradFunction 基类 + 各算子 Backward 子类 + backward 引擎
module;

#include <vector>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <stdexcept>

#include "df_types.h"

export module df.engine.autograd;

import df.engine.tensor_storage;
import df.engine.tensor;
import df.hal.cpu_backend;

export namespace df {

// 20260319 ZJH 前向声明 — GradFunction 需要引用自身的 shared_ptr
class GradFunction;

// 20260319 ZJH Edge — 计算图中的一条边，连接到前驱节点的第 nInputIndex 个输出
struct Edge {
    std::shared_ptr<GradFunction> pGradFn;  // 前驱节点（产生该输入的运算）
    int nInputIndex = 0;                     // 该输入在前驱节点输出中的索引
};

// 20260319 ZJH GradFunction 基类 — 计算图中的节点
// 每个需要梯度的运算创建一个 GradFunction 子类实例
class GradFunction {
public:
    virtual ~GradFunction() = default;

    // 20260319 ZJH 反向传播：输入 gradOutput，返回各输入的梯度
    // gradOutput: 从下游传回的梯度张量
    // 返回: 与输入数量相同的梯度向量（对应 m_vecInputEdges 的顺序）
    virtual std::vector<Tensor> backward(const Tensor& gradOutput) = 0;

    // 20260319 ZJH 输入边列表 — 记录该运算的每个输入来自哪个前驱节点
    std::vector<Edge> m_vecInputEdges;
};

// ==================== Backward 子类 ====================

// 20260319 ZJH AddBackward — a + b 的反向
// d(a+b)/da = 1, d(a+b)/db = 1
// 因此 grad_a = gradOutput, grad_b = gradOutput
class AddBackward : public GradFunction {
public:
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH 加法的两个输入梯度都等于输出梯度（链式法则乘以 1）
        return { gradOutput, gradOutput };
    }
};

// 20260319 ZJH SubBackward — a - b 的反向
// d(a-b)/da = 1, d(a-b)/db = -1
class SubBackward : public GradFunction {
public:
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH grad_a = gradOutput, grad_b = -gradOutput
        auto negGrad = Tensor::zeros(gradOutput.shapeVec());
        CPUBackend::mulScalar(gradOutput.floatDataPtr(), -1.0f,
                              negGrad.mutableFloatDataPtr(),
                              static_cast<size_t>(negGrad.numel()));
        return { gradOutput, negGrad };
    }
};

// 20260319 ZJH MulBackward — a * b（逐元素）的反向
// d(a*b)/da = b, d(a*b)/db = a
class MulBackward : public GradFunction {
public:
    Tensor m_savedA;  // 20260319 ZJH 保存前向传播时的输入 a
    Tensor m_savedB;  // 20260319 ZJH 保存前向传播时的输入 b

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH grad_a = gradOutput * b
        auto gradA = Tensor::zeros(gradOutput.shapeVec());
        auto cg = gradOutput.contiguous();
        auto cb = m_savedB.contiguous();
        CPUBackend::mul(cg.floatDataPtr(), cb.floatDataPtr(),
                        gradA.mutableFloatDataPtr(),
                        static_cast<size_t>(gradA.numel()));
        // 20260319 ZJH grad_b = gradOutput * a
        auto gradB = Tensor::zeros(gradOutput.shapeVec());
        auto ca = m_savedA.contiguous();
        CPUBackend::mul(cg.floatDataPtr(), ca.floatDataPtr(),
                        gradB.mutableFloatDataPtr(),
                        static_cast<size_t>(gradB.numel()));
        return { gradA, gradB };
    }
};

// 20260319 ZJH MatMulBackward — A @ B 的反向
// d(A@B)/dA = gradOutput @ B^T
// d(A@B)/dB = A^T @ gradOutput
class MatMulBackward : public GradFunction {
public:
    Tensor m_savedA;  // 20260319 ZJH [M, K]
    Tensor m_savedB;  // 20260319 ZJH [K, N]

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto cGrad = gradOutput.contiguous();  // [M, N]
        auto cA = m_savedA.contiguous();       // [M, K]
        auto cB = m_savedB.contiguous();       // [K, N]

        int nM = cA.shape(0);
        int nK = cA.shape(1);
        int nN = cB.shape(1);

        // 20260319 ZJH grad_A = gradOutput @ B^T  →  [M,N] @ [N,K] = [M,K]
        // 需要 B 的转置：从 [K,N] 行主序转为 [N,K] 行主序
        auto bT = Tensor::zeros({nN, nK});
        {
            const float* pB = cB.floatDataPtr();
            float* pBT = bT.mutableFloatDataPtr();
            for (int i = 0; i < nK; ++i) {
                for (int j = 0; j < nN; ++j) {
                    pBT[j * nK + i] = pB[i * nN + j];
                }
            }
        }
        auto gradA = Tensor::zeros({nM, nK});
        CPUBackend::matmul(cGrad.floatDataPtr(), bT.floatDataPtr(),
                           gradA.mutableFloatDataPtr(), nM, nN, nK);

        // 20260319 ZJH grad_B = A^T @ gradOutput  →  [K,M] @ [M,N] = [K,N]
        auto aT = Tensor::zeros({nK, nM});
        {
            const float* pA = cA.floatDataPtr();
            float* pAT = aT.mutableFloatDataPtr();
            for (int i = 0; i < nM; ++i) {
                for (int j = 0; j < nK; ++j) {
                    pAT[j * nM + i] = pA[i * nK + j];
                }
            }
        }
        auto gradB = Tensor::zeros({nK, nN});
        CPUBackend::matmul(aT.floatDataPtr(), cGrad.floatDataPtr(),
                           gradB.mutableFloatDataPtr(), nK, nM, nN);

        return { gradA, gradB };
    }
};

// 20260319 ZJH AddScalarBackward — a + scalar 的反向
// d(a+s)/da = 1 → grad_a = gradOutput
class AddScalarBackward : public GradFunction {
public:
    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        return { gradOutput };  // 标量不需要梯度，只返回 a 的梯度
    }
};

// 20260319 ZJH MulScalarBackward — a * scalar 的反向
// d(a*s)/da = s → grad_a = gradOutput * s
class MulScalarBackward : public GradFunction {
public:
    float m_fScalar = 0.0f;  // 20260319 ZJH 保存前向传播时的标量值

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        auto gradA = Tensor::zeros(gradOutput.shapeVec());
        auto cg = gradOutput.contiguous();
        CPUBackend::mulScalar(cg.floatDataPtr(), m_fScalar,
                              gradA.mutableFloatDataPtr(),
                              static_cast<size_t>(gradA.numel()));
        return { gradA };
    }
};

// 20260319 ZJH SumBackward — sum(a) 的反向
// d(sum(a))/da = ones_like(a)
class SumBackward : public GradFunction {
public:
    std::vector<int> m_vecInputShape;  // 20260319 ZJH 输入张量的形状

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // 20260319 ZJH gradOutput 是标量（单元素），需要广播到输入形状
        float fGradVal = gradOutput.at({0});
        auto gradA = Tensor::full(m_vecInputShape, fGradVal);
        return { gradA };
    }
};

// ==================== Backward 引擎 ====================

// 20260319 ZJH 拓扑排序 + 反向传播
// 从 rootGradFn 出发，BFS 收集所有节点，拓扑排序后逆序执行 backward
// 参数:
//   rootGradFn — loss 的 GradFunction
//   rootGrad — 初始梯度（通常为全 1 标量）
//   leafGradAccumulator — 回调函数，将梯度累积到叶子节点
//     参数: (GradFunction 指针, 输入索引, 梯度)
//     当遇到输入边的 pGradFn 为 nullptr 时（叶子节点），调用此回调
void runBackward(
    std::shared_ptr<GradFunction> rootGradFn,
    const Tensor& rootGrad,
    std::function<void(const Edge&, const Tensor&)> leafGradAccumulator)
{
    if (!rootGradFn) return;

    // 20260319 ZJH Step 1: 拓扑排序（Kahn 算法）
    // 收集所有节点并计算入度
    std::vector<std::shared_ptr<GradFunction>> vecOrder;
    std::unordered_map<GradFunction*, int> mapInDegree;
    std::unordered_set<GradFunction*> setVisited;

    // 20260319 ZJH BFS 收集所有可达节点
    std::vector<std::shared_ptr<GradFunction>> vecQueue;
    vecQueue.push_back(rootGradFn);
    setVisited.insert(rootGradFn.get());

    size_t nHead = 0;
    while (nHead < vecQueue.size()) {
        auto pCurrent = vecQueue[nHead++];
        mapInDegree[pCurrent.get()] = 0;  // 初始化入度

        for (const auto& edge : pCurrent->m_vecInputEdges) {
            if (edge.pGradFn && setVisited.find(edge.pGradFn.get()) == setVisited.end()) {
                setVisited.insert(edge.pGradFn.get());
                vecQueue.push_back(edge.pGradFn);
            }
        }
    }

    // 20260319 ZJH 计算入度：被多少个下游节点引用
    for (const auto& pNode : vecQueue) {
        for (const auto& edge : pNode->m_vecInputEdges) {
            if (edge.pGradFn) {
                mapInDegree[edge.pGradFn.get()]++;
            }
        }
    }

    // 20260319 ZJH Kahn 算法：从入度为 0 的节点开始
    std::vector<std::shared_ptr<GradFunction>> vecTopoQueue;
    // root 节点是唯一入度为 0 的（它没有下游消费者）
    for (const auto& pNode : vecQueue) {
        if (mapInDegree[pNode.get()] == 0) {
            vecTopoQueue.push_back(pNode);
        }
    }

    size_t nTopoHead = 0;
    while (nTopoHead < vecTopoQueue.size()) {
        auto pCurrent = vecTopoQueue[nTopoHead++];
        vecOrder.push_back(pCurrent);

        for (const auto& edge : pCurrent->m_vecInputEdges) {
            if (edge.pGradFn) {
                mapInDegree[edge.pGradFn.get()]--;
                if (mapInDegree[edge.pGradFn.get()] == 0) {
                    vecTopoQueue.push_back(edge.pGradFn);
                }
            }
        }
    }

    // 20260319 ZJH Step 2: 按拓扑序遍历，逐节点执行 backward
    // 用 map 存储每个节点的累积梯度
    std::unordered_map<GradFunction*, Tensor> mapGrads;
    mapGrads[rootGradFn.get()] = rootGrad;

    for (const auto& pNode : vecOrder) {
        auto it = mapGrads.find(pNode.get());
        if (it == mapGrads.end()) continue;  // 该节点无梯度输入，跳过

        const Tensor& nodeGrad = it->second;

        // 20260319 ZJH 调用该节点的 backward 计算各输入的梯度
        auto vecInputGrads = pNode->backward(nodeGrad);

        // 20260319 ZJH 将梯度分发给各输入边
        for (size_t i = 0; i < pNode->m_vecInputEdges.size() && i < vecInputGrads.size(); ++i) {
            const auto& edge = pNode->m_vecInputEdges[i];
            const auto& inputGrad = vecInputGrads[i];

            if (edge.pGradFn) {
                // 20260319 ZJH 非叶子：累积梯度到前驱节点
                auto jt = mapGrads.find(edge.pGradFn.get());
                if (jt == mapGrads.end()) {
                    mapGrads[edge.pGradFn.get()] = inputGrad;
                } else {
                    // 20260319 ZJH 梯度累加（多条路径汇聚到同一节点）
                    auto accumulated = Tensor::zeros(jt->second.shapeVec());
                    auto ca = jt->second.contiguous();
                    auto cb = inputGrad.contiguous();
                    CPUBackend::add(ca.floatDataPtr(), cb.floatDataPtr(),
                                    accumulated.mutableFloatDataPtr(),
                                    static_cast<size_t>(accumulated.numel()));
                    jt->second = accumulated;
                }
            } else {
                // 20260319 ZJH 叶子节点：通过回调累积梯度
                leafGradAccumulator(edge, inputGrad);
            }
        }
    }
}

}  // namespace df
```

- [ ] **Step 4: 验证编译通过**

- [ ] **Step 5: 提交**

```bash
git add CMakeLists.txt src/engine/df.engine.autograd.ixx tests/test_autograd.cpp
git commit -m "feat(engine): add AutoGrad engine with GradFunction and backward"
```

---

## Task 2: 扩展 Tensor — 添加 AutoGrad 字段

**Files:**
- Modify: `src/engine/df.engine.tensor.ixx`

- [ ] **Step 1: 给 Tensor 类添加以下成员和方法**

新增私有成员：
```cpp
bool m_bRequiresGrad = false;                      // 是否需要计算梯度
std::shared_ptr<Tensor> m_pGrad;                    // 累积梯度
std::shared_ptr<GradFunction> m_pGradFn;            // 产生此张量的运算节点
```

新增公有方法：
```cpp
// 20260319 ZJH 设置是否需要梯度
void setRequiresGrad(bool bRequires) { m_bRequiresGrad = bRequires; }

// 20260319 ZJH 是否需要梯度
bool requiresGrad() const { return m_bRequiresGrad; }

// 20260319 ZJH 获取梯度张量（只读引用）
const Tensor& grad() const {
    if (!m_pGrad) {
        static Tensor s_emptyGrad;
        return s_emptyGrad;
    }
    return *m_pGrad;
}

// 20260319 ZJH 累积梯度（内部使用）
void accumulateGrad(const Tensor& gradToAdd) {
    if (!m_pGrad) {
        m_pGrad = std::make_shared<Tensor>(Tensor::zeros(shapeVec()));
    }
    // 累加
    auto cOld = m_pGrad->contiguous();
    auto cNew = gradToAdd.contiguous();
    CPUBackend::add(cOld.floatDataPtr(), cNew.floatDataPtr(),
                    m_pGrad->mutableFloatDataPtr(),
                    static_cast<size_t>(m_pGrad->numel()));
}

// 20260319 ZJH 清零梯度
void zeroGrad() {
    if (m_pGrad) {
        CPUBackend::fillZeros(m_pGrad->mutableFloatDataPtr(), m_pGrad->numel());
    }
}

// 20260319 ZJH 设置 GradFunction（前向运算时由 tensor_ops 调用）
void setGradFn(std::shared_ptr<GradFunction> pGradFn) { m_pGradFn = pGradFn; }

// 20260319 ZJH 获取 GradFunction
std::shared_ptr<GradFunction> gradFn() const { return m_pGradFn; }

// 20260319 ZJH 反向传播 — 从此张量开始反向求梯度
// 仅对标量（numel()==1）或提供 gradOutput 时可调用
void backward(const Tensor& gradOutput);
void backward();  // 隐式 gradOutput = ones({1})
```

**重要**：Tensor 类需要 `import df.engine.autograd;` 来使用 GradFunction 和 runBackward。但 autograd 也 import tensor。这是循环依赖！

**解决方案**：`backward()` 的实现放在 `tensor_ops.ixx` 中（作为自由函数 `tensorBackward`），Tensor 类本身只存储 `shared_ptr<GradFunction>` 但不引用 autograd 模块。GradFunction 在 autograd 模块中前向声明即可。

**实际做法**：
- Tensor 类使用 `std::shared_ptr<void>` 存储 GradFunction（类型擦除避免循环依赖）
- tensor_ops 中的 `tensorBackward()` 函数负责类型转换并调用 `runBackward()`
- Tensor 上的 `.backward()` 方法不存在 — 改为用 `tensorBackward(loss)` 自由函数

这与 Phase 1B 的自由函数设计一致（tensorAdd/Sub/Mul 都是自由函数）。

修改后的字段：
```cpp
// 在 Tensor private 中
bool m_bRequiresGrad = false;
std::shared_ptr<Tensor> m_pGrad;           // 累积梯度
std::shared_ptr<void> m_pGradFn;           // 类型擦除的 GradFunction 指针
```

- [ ] **Step 2: 修改 Tensor 类**

添加上述字段和方法。`makeView` 也要传播 requiresGrad 相关信息（视图操作不改变 requiresGrad 状态）。

- [ ] **Step 3: 验证现有 55 个测试仍然通过（无回归）**

- [ ] **Step 4: 提交**

```bash
git add src/engine/df.engine.tensor.ixx
git commit -m "feat(engine): extend Tensor with AutoGrad fields (requiresGrad, grad, gradFn)"
```

---

## Task 3: 修改 tensor_ops — 前向运算构建计算图 + tensorBackward

**Files:**
- Modify: `src/engine/df.engine.tensor_ops.ixx`

- [ ] **Step 1: 修改每个运算函数**

在每个前向运算（tensorAdd/Sub/Mul/MatMul/AddScalar/MulScalar/Sum）中，如果任一输入 `requiresGrad()`，则：
1. 创建对应的 Backward 子类实例
2. 保存必要的前向数据（如 MulBackward 保存 a, b；MatMulBackward 保存 A, B）
3. 设置 `m_vecInputEdges`（指向输入张量的 gradFn）
4. 将 Backward 实例设置为输出张量的 gradFn
5. 输出张量标记为 requiresGrad

示例改造 tensorAdd：
```cpp
Tensor tensorAdd(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();
    auto cb = b.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::add(ca.floatDataPtr(), cb.floatDataPtr(),
                    result.mutableFloatDataPtr(), static_cast<size_t>(result.numel()));

    // 20260319 ZJH AutoGrad: 如果任一输入需要梯度，构建计算图
    if (a.requiresGrad() || b.requiresGrad()) {
        auto pBackward = std::make_shared<AddBackward>();
        // Edge 指向输入的 gradFn（叶子节点为 nullptr）
        pBackward->m_vecInputEdges = {
            { castGradFn(a.gradFnRaw()), 0 },
            { castGradFn(b.gradFnRaw()), 0 }
        };
        result.setRequiresGrad(true);
        result.setGradFnRaw(pBackward);
    }

    return result;
}
```

- [ ] **Step 2: 实现 tensorBackward 自由函数**

```cpp
// 20260319 ZJH 从 loss 张量开始反向传播
void tensorBackward(Tensor& loss) {
    auto pGradFn = castGradFn(loss.gradFnRaw());
    if (!pGradFn) return;

    // 20260319 ZJH 初始梯度为全 1（loss 是标量）
    Tensor rootGrad = Tensor::ones({1});

    // 20260319 ZJH 叶子节点梯度累积回调
    // 需要知道哪些张量是叶子 — 通过 Edge 的 pGradFn == nullptr 判断
    // 叶子梯度累积需要访问原始张量的 accumulateGrad()
    // 这里用全局 leaf registry（在 setRequiresGrad 时注册）
    runBackward(pGradFn, rootGrad,
        [](const Edge& edge, const Tensor& grad) {
            // 叶子节点梯度通过 leaf registry 累积
            // 具体实现见下文
        });
}
```

**叶子节点梯度累积策略**：
- 创建一个 `GradAccumulator` 共享对象（`shared_ptr<GradAccumulator>`），Tensor 持有它
- `LeafAccumulator` GradFunction 子类持有 `shared_ptr<GradAccumulator>`（安全引用，无生命周期风险）
- 每个 `setRequiresGrad(true)` 的叶子张量，在参与运算时用 `LeafAccumulator` 作为其 gradFn

```cpp
// 20260319 ZJH GradAccumulator — 叶子张量的梯度累积器
// 通过 shared_ptr 被 Tensor 和 LeafAccumulator 共同持有，生命周期安全
struct GradAccumulator {
    Tensor m_grad;       // 累积梯度
    bool m_bHasGrad = false;  // 是否已有梯度

    void accumulate(const Tensor& gradToAdd) {
        if (!m_bHasGrad) {
            m_grad = Tensor::zeros(gradToAdd.shapeVec());
            m_bHasGrad = true;
        }
        auto cOld = m_grad.contiguous();
        auto cNew = gradToAdd.contiguous();
        auto result = Tensor::zeros(m_grad.shapeVec());
        CPUBackend::add(cOld.floatDataPtr(), cNew.floatDataPtr(),
                        result.mutableFloatDataPtr(),
                        static_cast<size_t>(result.numel()));
        m_grad = result;
    }

    void zero() {
        if (m_bHasGrad) {
            CPUBackend::fillZeros(m_grad.mutableFloatDataPtr(), m_grad.numel());
        }
    }
};

class LeafAccumulator : public GradFunction {
public:
    std::shared_ptr<GradAccumulator> m_pAccumulator;  // 共享持有，生命周期安全

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        if (m_pAccumulator) {
            m_pAccumulator->accumulate(gradOutput);
        }
        return {};  // 叶子无前驱
    }
};
```

**Tensor 类修改**：m_pGrad 替换为 `shared_ptr<GradAccumulator>`，`grad()` 返回 `m_pAccumulator->m_grad`，`zeroGrad()` 调用 `m_pAccumulator->zero()`。

修改 tensor_ops 中每个运算：当输入是叶子（requiresGrad==true 但 gradFn==nullptr）时，创建 LeafAccumulator，将输入的 GradAccumulator 传入。

**`tensorSum` API 变更**：当前 `tensorSum` 返回 `float`，AutoGrad 需要返回 `Tensor`。改为：
- `tensorSum` 返回 `Tensor`（标量张量 shape={1}，带 SumBackward gradFn）
- 新增 `Tensor::item()` 方法返回标量张量的 float 值
- 旧的 `tensorMax`/`tensorMin` 保持返回 float（暂不需要 backward）

- [ ] **Step 3: 验证编译通过 + 旧测试不回归**

- [ ] **Step 4: 提交**

```bash
git add src/engine/df.engine.tensor_ops.ixx
git commit -m "feat(engine): integrate AutoGrad into tensor_ops — build compute graph on forward"
```

---

## Task 4: AutoGrad 测试 — 数值梯度验证

**Files:**
- Create: `tests/test_autograd.cpp`

- [ ] **Step 1: 编写测试**

```cpp
// 20260319 ZJH AutoGrad 单元测试 — 数值梯度检查验证解析梯度
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.autograd;

// 20260319 ZJH 辅助：数值梯度 (f(x+eps) - f(x-eps)) / (2*eps)
// 对 tensor 的每个元素逐一扰动，计算数值梯度
// func: 接受 Tensor 返回标量 Tensor 的函数
std::vector<float> numericalGradient(
    const df::Tensor& x,
    std::function<df::Tensor(const df::Tensor&)> func,
    float fEps = 1e-4f)
{
    int nNumel = x.numel();
    std::vector<float> vecGrad(nNumel);
    auto cx = x.contiguous();

    for (int i = 0; i < nNumel; ++i) {
        // x + eps
        auto xPlus = df::Tensor::fromData(cx.floatDataPtr(), cx.shapeVec());
        xPlus.mutableFloatDataPtr()[i] += fEps;
        float fPlus = df::tensorSum(func(xPlus));

        // x - eps
        auto xMinus = df::Tensor::fromData(cx.floatDataPtr(), cx.shapeVec());
        xMinus.mutableFloatDataPtr()[i] -= fEps;
        float fMinus = df::tensorSum(func(xMinus));

        vecGrad[i] = (fPlus - fMinus) / (2.0f * fEps);
    }
    return vecGrad;
}

// 20260319 ZJH 辅助：比较解析梯度和数值梯度
void checkGradient(const df::Tensor& analyticGrad,
                   const std::vector<float>& vecNumericGrad,
                   float fTol = 1e-3f) {
    auto cGrad = analyticGrad.contiguous();
    const float* pAnalytic = cGrad.floatDataPtr();
    ASSERT_EQ(analyticGrad.numel(), static_cast<int>(vecNumericGrad.size()));
    for (int i = 0; i < analyticGrad.numel(); ++i) {
        EXPECT_NEAR(pAnalytic[i], vecNumericGrad[i], fTol)
            << "Gradient mismatch at index " << i;
    }
}

// ==================== 测试用例 ====================

// 20260319 ZJH 测试加法梯度 — loss = sum(a + b)
TEST(AutogradTest, AddGradient) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = df::tensorAdd(a, b);
    auto loss = df::tensorSum(c);
    df::tensorBackward(loss);

    // d(sum(a+b))/da = 1 for all elements
    auto cGradA = a.grad().contiguous();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(cGradA.floatDataPtr()[i], 1.0f);
    }
    auto cGradB = b.grad().contiguous();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(cGradB.floatDataPtr()[i], 1.0f);
    }
}

// 20260319 ZJH 测试减法梯度 — loss = sum(a - b)
TEST(AutogradTest, SubGradient) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3}.data(), {3});
    auto b = df::Tensor::fromData(std::vector<float>{4, 5, 6}.data(), {3});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = df::tensorSub(a, b);
    auto loss = df::tensorSum(c);
    df::tensorBackward(loss);

    auto cGradA = a.grad().contiguous();
    auto cGradB = b.grad().contiguous();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(cGradA.floatDataPtr()[i], 1.0f);
        EXPECT_FLOAT_EQ(cGradB.floatDataPtr()[i], -1.0f);
    }
}

// 20260319 ZJH 测试乘法梯度 — loss = sum(a * b)
TEST(AutogradTest, MulGradient) {
    auto a = df::Tensor::fromData(std::vector<float>{2, 3}.data(), {2});
    auto b = df::Tensor::fromData(std::vector<float>{4, 5}.data(), {2});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = df::tensorMul(a, b);
    auto loss = df::tensorSum(c);
    df::tensorBackward(loss);

    // d(sum(a*b))/da = b
    EXPECT_FLOAT_EQ(a.grad().at({0}), 4.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1}), 5.0f);
    // d(sum(a*b))/db = a
    EXPECT_FLOAT_EQ(b.grad().at({0}), 2.0f);
    EXPECT_FLOAT_EQ(b.grad().at({1}), 3.0f);
}

// 20260319 ZJH 测试 matmul 梯度 — 数值梯度检查
TEST(AutogradTest, MatMulGradient) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = df::tensorMatmul(a, b);
    auto loss = df::tensorSum(c);
    df::tensorBackward(loss);

    // 数值梯度检查 a
    auto numGradA = numericalGradient(a, [&](const df::Tensor& x) {
        return df::tensorMatmul(x, b);
    });
    checkGradient(a.grad(), numGradA);

    // 数值梯度检查 b
    auto numGradB = numericalGradient(b, [&](const df::Tensor& x) {
        return df::tensorMatmul(a, x);
    });
    checkGradient(b.grad(), numGradB);
}

// 20260319 ZJH 测试标量乘法梯度 — loss = sum(a * 3.0)
TEST(AutogradTest, MulScalarGradient) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3}.data(), {3});
    a.setRequiresGrad(true);

    auto c = df::tensorMulScalar(a, 3.0f);
    auto loss = df::tensorSum(c);
    df::tensorBackward(loss);

    // d(sum(a*3))/da = 3
    auto cGrad = a.grad().contiguous();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(cGrad.floatDataPtr()[i], 3.0f);
    }
}

// 20260319 ZJH 测试链式法则 — loss = sum((a + b) * a)
TEST(AutogradTest, ChainRule) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3}.data(), {3});
    auto b = df::Tensor::fromData(std::vector<float>{4, 5, 6}.data(), {3});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = df::tensorAdd(a, b);    // c = a + b
    auto d = df::tensorMul(c, a);    // d = c * a = (a+b)*a = a^2 + ab
    auto loss = df::tensorSum(d);    // loss = sum(a^2 + ab)
    df::tensorBackward(loss);

    // d(loss)/da = 2a + b = {6, 9, 12}
    EXPECT_NEAR(a.grad().at({0}), 6.0f, 1e-3f);
    EXPECT_NEAR(a.grad().at({1}), 9.0f, 1e-3f);
    EXPECT_NEAR(a.grad().at({2}), 12.0f, 1e-3f);

    // d(loss)/db = a = {1, 2, 3}
    EXPECT_NEAR(b.grad().at({0}), 1.0f, 1e-3f);
    EXPECT_NEAR(b.grad().at({1}), 2.0f, 1e-3f);
    EXPECT_NEAR(b.grad().at({2}), 3.0f, 1e-3f);
}

// 20260319 ZJH 测试 zeroGrad — 梯度清零后可重新 backward
TEST(AutogradTest, ZeroGrad) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2}.data(), {2});
    a.setRequiresGrad(true);

    auto loss1 = df::tensorSum(df::tensorMulScalar(a, 2.0f));
    df::tensorBackward(loss1);
    EXPECT_FLOAT_EQ(a.grad().at({0}), 2.0f);

    a.zeroGrad();
    EXPECT_FLOAT_EQ(a.grad().at({0}), 0.0f);
}

// 20260319 ZJH 测试非叶子不保留梯度（叶子才有 grad）
TEST(AutogradTest, OnlyLeafHasGrad) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2}.data(), {2});
    a.setRequiresGrad(true);

    auto b = df::tensorMulScalar(a, 2.0f);  // b 是中间节点
    auto loss = df::tensorSum(b);
    df::tensorBackward(loss);

    // a 是叶子，有梯度
    EXPECT_EQ(a.grad().numel(), 2);
    // b 是中间节点，不保留梯度（grad 为空）
    EXPECT_EQ(b.grad().numel(), 0);
}
```

- [ ] **Step 2: 编译并运行测试**

Expected: 8 个测试全部通过

- [ ] **Step 3: 提交**

```bash
git add tests/test_autograd.cpp
git commit -m "test(engine): add AutoGrad tests with numerical gradient checks"
```

---

## Task 5: 全量验证

- [ ] **Step 1: 全量构建 + 运行全部测试**

Expected:
- Phase 1A: 30 tests
- Phase 1B: 25 tests (tensor 11 + tensor_ops 14)
- Phase 1C: 8 tests
- **Total: 63 tests**

- [ ] **Step 2: 更新 DEVLOG.md**

- [ ] **Step 3: 提交**

```bash
git add DEVLOG.md
git commit -m "milestone: Phase 1C complete — AutoGrad engine (63/63 tests passing)"
```

---

## Summary

| Task | 模块 | 测试数 | 关键验证点 |
|------|------|--------|-----------|
| 1 | GradFunction + Backward 引擎 | — | 编译通过 |
| 2 | Tensor AutoGrad 字段 | — | 旧测试不回归 |
| 3 | tensor_ops 计算图构建 | — | 旧测试不回归 |
| 4 | AutoGrad 测试 | 8 | Add/Sub/Mul/MatMul/Scalar/Chain/ZeroGrad/LeafOnly |
| 5 | 全量验证 | 63 | 全部通过 |

### 设计要点

1. **类型擦除避免循环依赖**：Tensor 用 `shared_ptr<void>` 存储 GradFunction，tensor_ops 中做 `static_pointer_cast<GradFunction>`
2. **LeafAccumulator**：叶子张量参与运算时自动获得一个 LeafAccumulator 作为 gradFn，backward 到达时将梯度累积到叶子的 m_pGrad
3. **自由函数 tensorBackward(loss)**：保持 Phase 1B 的自由函数风格，避免 Tensor 类直接依赖 autograd 模块
4. **数值梯度检查**：MatMul 梯度最容易出错，用有限差分 `(f(x+ε)-f(x-ε))/(2ε)` 验证解析梯度
5. **链式法则测试**：`sum((a+b)*a)` 验证梯度通过多层运算正确传播和累积
