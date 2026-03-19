# Phase 1B: Tensor + CPUBackend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Tensor 多维张量系统和 CPUBackend 计算后端，支持基础运算（工厂方法、元素运算、matmul、reshape、transpose、slice），为后续 AutoGrad（Phase 1C）和 MLP 验证（Phase 1D）提供计算基础。

**Architecture:** Tensor 采用 Storage + View 分离设计：TensorStorage 持有原始内存（引用计数），Tensor 是 Storage 之上的视图（shape/strides/offset）。视图操作（reshape/slice/transpose）共享 Storage，零拷贝。CPUBackend 提供朴素 C++ 实现的计算内核（先正确后优化）。Tensor 运算委托给 CPUBackend。暂不实现 AutoGrad 相关字段（Phase 1C 添加）。

**Tech Stack:** C++23 modules (.ixx), MSVC 14.50, CMake 4.2.3, GTest

**Spec Reference:** `docs/superpowers/specs/2026-03-19-deepforge-design.md` §3.1, §4.1-4.3

**Dependencies:** Phase 1A (df_platform) 已完成

---

## Scope Decisions

Phase 1B **只实现** CPU Float32 路径。以下内容**推迟**：

| 推迟到 | 内容 |
|--------|------|
| Phase 1C | AutoGrad 字段（m_bRequiresGrad, m_pGradFn, backward()） |
| Phase 1C | GradFunction 基类及子类 |
| Phase 2 | Float16/Int32/Int64/UInt8 数据类型实际计算 |
| Phase 2 | CUDA/OpenCL Backend |
| Phase 2 | DeviceManager 完整实现 |

Tensor 类预留 dtype/device 字段但当前仅 Float32+CPU 有实际计算路径。

---

## File Map

| 文件路径 | 职责 |
|---------|------|
| `src/engine/df.engine.tensor_storage.ixx` | TensorStorage — 引用计数原始内存持有者 |
| `src/engine/df.engine.tensor.ixx` | Tensor — 多维张量视图，工厂方法，属性查询 |
| `src/engine/df.engine.tensor_ops.ixx` | Tensor 运算方法 — 元素运算、matmul、reshape、transpose、slice |
| `src/hal/df.hal.cpu_backend.ixx` | CPUBackend — 朴素 C++ 计算内核（Float32 only） |
| `tests/test_tensor.cpp` | Tensor 创建、属性、数据访问测试 |
| `tests/test_tensor_ops.cpp` | Tensor 运算测试（加减乘除、matmul、reshape、transpose、slice） |
| `CMakeLists.txt` | 修改：添加 df_engine 库 + df_hal 库 + 新测试目标 |

**设计决策 — 为什么拆成 3 个 .ixx 而非 1 个：**
- `tensor_storage.ixx`：独立的内存管理，Tensor 和 AutoGrad 都依赖它
- `tensor.ixx`：Tensor 类定义 + 工厂方法 + 属性（不含运算逻辑，保持文件聚焦）
- `tensor_ops.ixx`：运算方法实现，依赖 CPUBackend（后续会膨胀，独立文件便于维护）

---

## Task 1: CMake 配置 — 添加 df_engine 和 df_hal 库

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 在 CMakeLists.txt 中添加 HAL 层和引擎层库 + 测试目标**

在 `df_platform` 定义之后、测试部分之前，添加：

```cmake
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
            src/engine/df.engine.tensor_ops.ixx
)

target_include_directories(df_engine PUBLIC include)
target_link_libraries(df_engine PUBLIC df_hal)
```

在测试部分，更新 `df_add_test` 函数并添加新测试（需要链接 `df_engine` 而非 `df_platform`）：

```cmake
    # 引擎层测试需要链接 df_engine
    function(df_add_engine_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE df_engine GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    df_add_engine_test(test_tensor tests/test_tensor.cpp)
    df_add_engine_test(test_tensor_ops tests/test_tensor_ops.cpp)
```

- [ ] **Step 2: 创建所有 stub 文件**

`src/hal/df.hal.cpu_backend.ixx`:
```cpp
export module df.hal.cpu_backend;
```

`src/engine/df.engine.tensor_storage.ixx`:
```cpp
export module df.engine.tensor_storage;
```

`src/engine/df.engine.tensor.ixx`:
```cpp
export module df.engine.tensor;
```

`src/engine/df.engine.tensor_ops.ixx`:
```cpp
export module df.engine.tensor_ops;
```

`tests/test_tensor.cpp`:
```cpp
#include <gtest/gtest.h>
TEST(TensorStub, Placeholder) { EXPECT_TRUE(true); }
```

`tests/test_tensor_ops.cpp`:
```cpp
#include <gtest/gtest.h>
TEST(TensorOpsStub, Placeholder) { EXPECT_TRUE(true); }
```

- [ ] **Step 3: 验证 CMake configure + build 成功**

```bash
cmake --preset windows-debug
cmake --build build/windows-debug
```

- [ ] **Step 4: 提交**

```bash
git add CMakeLists.txt src/hal/*.ixx src/engine/*.ixx tests/test_tensor.cpp tests/test_tensor_ops.cpp
git commit -m "feat: add df_hal and df_engine libraries with stubs for Tensor system"
```

---

## Task 2: TensorStorage — 引用计数内存持有者

**Files:**
- Create: `src/engine/df.engine.tensor_storage.ixx`

- [ ] **Step 1: 实现 TensorStorage**

TensorStorage 负责持有原始内存、管理分配/释放。使用 `std::shared_ptr` 自然实现引用计数。

```cpp
// 20260319 ZJH TensorStorage — 张量原始内存持有者
// 通过 shared_ptr 引用计数管理生命周期
// 视图操作（reshape/slice/transpose）共享同一 Storage 实例
module;

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "df_types.h"

export module df.engine.tensor_storage;

export namespace df {

// 20260319 ZJH TensorStorage — 持有连续内存块，支持 CPU 分配
// 通过 shared_ptr<TensorStorage> 在多个 Tensor 视图间共享
class TensorStorage {
public:
    // 20260319 ZJH 构造 — 分配 nBytes 字节的 64 字节对齐内存
    explicit TensorStorage(size_t nBytes)
        : m_nBytes(nBytes), m_pData(nullptr), m_deviceType(DeviceType::CPU), m_nDeviceId(0)
    {
        if (nBytes > 0) {
            // 20260319 ZJH 64 字节对齐分配（缓存行对齐）
#ifdef _MSC_VER
            m_pData = _aligned_malloc(nBytes, 64);
#else
            m_pData = std::aligned_alloc(64, (nBytes + 63) & ~63);
#endif
            if (!m_pData) {
                throw std::bad_alloc();  // 内存分配失败
            }
        }
    }

    // 20260319 ZJH 从外部数据构造（拷贝模式）
    // 参数: pSrcData — 源数据指针, nBytes — 数据字节数
    TensorStorage(const void* pSrcData, size_t nBytes)
        : TensorStorage(nBytes)
    {
        if (pSrcData && nBytes > 0) {
            std::memcpy(m_pData, pSrcData, nBytes);  // 拷贝源数据到新分配内存
        }
    }

    // 20260319 ZJH 禁止拷贝 — Storage 只能通过 shared_ptr 共享
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;

    // 20260319 ZJH 析构 — 释放对齐内存
    ~TensorStorage() {
        if (m_pData) {
#ifdef _MSC_VER
            _aligned_free(m_pData);
#else
            std::free(m_pData);
#endif
            m_pData = nullptr;
        }
    }

    // 20260319 ZJH 获取原始数据指针（只读）
    const void* data() const { return m_pData; }

    // 20260319 ZJH 获取原始数据指针（可写）
    void* mutableData() { return m_pData; }

    // 20260319 ZJH 获取字节数
    size_t bytes() const { return m_nBytes; }

    // 20260319 ZJH 获取设备类型
    DeviceType deviceType() const { return m_deviceType; }

    // 20260319 ZJH 获取设备 ID（预留多 GPU）
    int deviceId() const { return m_nDeviceId; }

private:
    void* m_pData;              // 原始内存指针
    size_t m_nBytes;            // 分配的字节数
    DeviceType m_deviceType;    // 设备类型（当前仅 CPU）
    int m_nDeviceId;            // 设备 ID（预留多 GPU，当前恒为 0）
};

}  // namespace df
```

注意：TensorStorage 不在 test_tensor.cpp 中单独测试，而是通过 Tensor 的测试间接覆盖。它足够简单，且直接依赖 `_aligned_malloc`/`std::aligned_alloc` 等已验证的系统调用。

- [ ] **Step 2: 验证编译通过**

```bash
cmake --build build/windows-debug --target df_engine
```

- [ ] **Step 3: 提交**

```bash
git add src/engine/df.engine.tensor_storage.ixx
git commit -m "feat(engine): add TensorStorage with aligned memory management"
```

---

## Task 3: CPUBackend — 朴素 C++ 计算内核

**Files:**
- Create: `src/hal/df.hal.cpu_backend.ixx`

- [ ] **Step 1: 实现 CPUBackend**

Phase 1B 的 CPUBackend 只需提供 Tensor 基础运算所需的内核函数。不实现完整的 ComputeBackend 虚接口（那是 Phase 2 HAL 抽象的事），而是直接提供静态函数。

```cpp
// 20260319 ZJH CPUBackend — 朴素 C++ 计算内核
// Phase 1B: 提供 Float32 基础运算（先正确后优化）
// Phase 2 扩展为 ComputeBackend 虚接口实现 + SIMD 优化
module;

#include <cstddef>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

export module df.hal.cpu_backend;

export namespace df {

// 20260319 ZJH CPU 计算后端 — 全部静态方法，朴素循环实现
class CPUBackend {
public:
    // ==================== 填充操作 ====================

    // 20260319 ZJH 填零 — 将 nCount 个 float 元素置为 0
    static void fillZeros(float* pData, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pData[i] = 0.0f;
        }
    }

    // 20260319 ZJH 填一 — 将 nCount 个 float 元素置为 1
    static void fillOnes(float* pData, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pData[i] = 1.0f;
        }
    }

    // 20260319 ZJH 填充常量值
    static void fillValue(float* pData, size_t nCount, float fValue) {
        for (size_t i = 0; i < nCount; ++i) {
            pData[i] = fValue;
        }
    }

    // 20260319 ZJH 正态分布随机填充（均值 0，标准差 1）
    static void fillRandn(float* pData, size_t nCount) {
        // 20260319 ZJH 使用 Mersenne Twister 引擎 + 标准正态分布
        static thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < nCount; ++i) {
            pData[i] = dist(gen);
        }
    }

    // ==================== 元素级运算 ====================
    // 注意：所有元素级运算假设输入已经是连续内存（contiguous）
    // 非连续张量需要先做 contiguous() 转换（在 Tensor 层处理）

    // 20260319 ZJH 逐元素加法: pOut[i] = pA[i] + pB[i]
    static void add(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] + pB[i];
        }
    }

    // 20260319 ZJH 逐元素减法: pOut[i] = pA[i] - pB[i]
    static void sub(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] - pB[i];
        }
    }

    // 20260319 ZJH 逐元素乘法（Hadamard积）: pOut[i] = pA[i] * pB[i]
    static void mul(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] * pB[i];
        }
    }

    // 20260319 ZJH 逐元素除法: pOut[i] = pA[i] / pB[i]
    static void div(const float* pA, const float* pB, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] / pB[i];
        }
    }

    // 20260319 ZJH 标量加法: pOut[i] = pA[i] + fScalar
    static void addScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] + fScalar;
        }
    }

    // 20260319 ZJH 标量乘法: pOut[i] = pA[i] * fScalar
    static void mulScalar(const float* pA, float fScalar, float* pOut, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] * fScalar;
        }
    }

    // ==================== 矩阵运算 ====================

    // 20260319 ZJH 矩阵乘法: C = A * B
    // A: [M x K], B: [K x N], C: [M x N]
    // 行主序（row-major）存储
    static void matmul(const float* pA, const float* pB, float* pC,
                       int nM, int nK, int nN) {
        // 20260319 ZJH 先清零输出矩阵
        for (int i = 0; i < nM * nN; ++i) {
            pC[i] = 0.0f;
        }
        // 20260319 ZJH 三重循环朴素矩阵乘法（ikj 顺序，对 B 的访问更连续）
        for (int i = 0; i < nM; ++i) {
            for (int k = 0; k < nK; ++k) {
                float fA_ik = pA[i * nK + k];  // 缓存 A[i][k]
                for (int j = 0; j < nN; ++j) {
                    pC[i * nN + j] += fA_ik * pB[k * nN + j];
                }
            }
        }
    }

    // ==================== 归约运算 ====================

    // 20260319 ZJH 求和
    static float sum(const float* pData, size_t nCount) {
        float fSum = 0.0f;
        for (size_t i = 0; i < nCount; ++i) {
            fSum += pData[i];
        }
        return fSum;
    }

    // 20260319 ZJH 求最大值（nCount 必须 > 0）
    static float max(const float* pData, size_t nCount) {
        if (nCount == 0) return 0.0f;  // 空输入安全返回
        float fMax = pData[0];
        for (size_t i = 1; i < nCount; ++i) {
            if (pData[i] > fMax) fMax = pData[i];
        }
        return fMax;
    }

    // 20260319 ZJH 求最小值（nCount 必须 > 0）
    static float min(const float* pData, size_t nCount) {
        if (nCount == 0) return 0.0f;  // 空输入安全返回
        float fMin = pData[0];
        for (size_t i = 1; i < nCount; ++i) {
            if (pData[i] < fMin) fMin = pData[i];
        }
        return fMin;
    }

    // ==================== 数据拷贝 ====================

    // 20260319 ZJH 连续内存拷贝
    static void copy(const float* pSrc, float* pDst, size_t nCount) {
        for (size_t i = 0; i < nCount; ++i) {
            pDst[i] = pSrc[i];
        }
    }

    // 20260319 ZJH 基于 strides 的非连续数据提取到连续缓冲区
    // 用于将非连续视图（transpose/slice结果）拷贝为连续内存
    // 参数:
    //   pSrc — 源 Storage 数据指针（完整内存块）
    //   pDst — 目标连续缓冲区
    //   vecShape — 张量形状
    //   vecStrides — 源张量步长（元素为单位，非字节）
    //   nOffset — 源张量在 Storage 中的偏移（元素为单位）
    static void stridedCopy(const float* pSrc, float* pDst,
                            const std::vector<int>& vecShape,
                            const std::vector<int>& vecStrides,
                            int nOffset) {
        int nNDim = static_cast<int>(vecShape.size());
        if (nNDim == 0) return;

        // 20260319 ZJH 计算总元素数
        int nTotal = 1;
        for (int d = 0; d < nNDim; ++d) {
            nTotal *= vecShape[d];
        }

        // 20260319 ZJH 用多维索引迭代，逐元素根据 strides 定位源地址
        std::vector<int> vecIdx(nNDim, 0);  // 当前多维索引
        for (int i = 0; i < nTotal; ++i) {
            // 20260319 ZJH 计算源偏移 = offset + sum(idx[d] * strides[d])
            int nSrcIdx = nOffset;
            for (int d = 0; d < nNDim; ++d) {
                nSrcIdx += vecIdx[d] * vecStrides[d];
            }
            pDst[i] = pSrc[nSrcIdx];

            // 20260319 ZJH 递增多维索引（最后一维优先，行主序）
            for (int d = nNDim - 1; d >= 0; --d) {
                vecIdx[d]++;
                if (vecIdx[d] < vecShape[d]) break;
                vecIdx[d] = 0;  // 进位到上一维
            }
        }
    }
};

}  // namespace df
```

- [ ] **Step 2: 验证编译通过**

```bash
cmake --build build/windows-debug --target df_hal
```

- [ ] **Step 3: 提交**

```bash
git add src/hal/df.hal.cpu_backend.ixx
git commit -m "feat(hal): add CPUBackend with naive Float32 compute kernels"
```

---

## Task 4: Tensor 类 — 创建、属性、数据访问

**Files:**
- Create: `src/engine/df.engine.tensor.ixx`
- Create: `tests/test_tensor.cpp`

- [ ] **Step 1: 编写 Tensor 测试**

```cpp
// 20260319 ZJH Tensor 创建、属性、数据访问单元测试
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
import df.engine.tensor;

// 20260319 ZJH 测试 zeros 工厂 — 创建全零张量
TEST(TensorTest, Zeros) {
    auto t = df::Tensor::zeros({2, 3});
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dtype(), df::DataType::Float32);
    EXPECT_EQ(t.device(), df::DeviceType::CPU);
    // 验证全零
    const float* pData = t.dataPtr<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(pData[i], 0.0f);
    }
}

// 20260319 ZJH 测试 ones 工厂 — 创建全一张量
TEST(TensorTest, Ones) {
    auto t = df::Tensor::ones({3, 4});
    EXPECT_EQ(t.numel(), 12);
    const float* pData = t.dataPtr<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(pData[i], 1.0f);
    }
}

// 20260319 ZJH 测试 full 工厂 — 创建指定值填充的张量
TEST(TensorTest, Full) {
    auto t = df::Tensor::full({2, 2}, 3.14f);
    const float* pData = t.dataPtr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(pData[i], 3.14f);
    }
}

// 20260319 ZJH 测试 randn 工厂 — 正态分布随机张量
TEST(TensorTest, Randn) {
    auto t = df::Tensor::randn({100});
    EXPECT_EQ(t.numel(), 100);
    // 验证不是全零（概率极低）
    const float* pData = t.dataPtr<float>();
    float fSum = 0.0f;
    for (int i = 0; i < 100; ++i) fSum += std::abs(pData[i]);
    EXPECT_GT(fSum, 0.0f);
}

// 20260319 ZJH 测试 fromData — 从现有数据创建张量（拷贝）
TEST(TensorTest, FromData) {
    std::vector<float> vecData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = df::Tensor::fromData(vecData.data(), {2, 3});
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);
    const float* pData = t.dataPtr<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(pData[i], vecData[i]);
    }
}

// 20260319 ZJH 测试 1D 张量
TEST(TensorTest, OneDimensional) {
    auto t = df::Tensor::zeros({5});
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_EQ(t.shape(0), 5);
    EXPECT_EQ(t.numel(), 5);
}

// 20260319 ZJH 测试 3D 张量
TEST(TensorTest, ThreeDimensional) {
    auto t = df::Tensor::ones({2, 3, 4});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);
    EXPECT_EQ(t.shape(2), 4);
}

// 20260319 ZJH 测试 strides 计算 — 行主序 C-contiguous
TEST(TensorTest, Strides) {
    auto t = df::Tensor::zeros({2, 3, 4});
    // 行主序 strides: [3*4, 4, 1] = [12, 4, 1]
    EXPECT_EQ(t.stride(0), 12);
    EXPECT_EQ(t.stride(1), 4);
    EXPECT_EQ(t.stride(2), 1);
}

// 20260319 ZJH 测试 isContiguous
TEST(TensorTest, IsContiguous) {
    auto t = df::Tensor::zeros({2, 3});
    EXPECT_TRUE(t.isContiguous());
}

// 20260319 ZJH 测试 at() 多维索引访问
TEST(TensorTest, AtAccess) {
    std::vector<float> vecData = {1, 2, 3, 4, 5, 6};
    auto t = df::Tensor::fromData(vecData.data(), {2, 3});
    // t[0][0]=1, t[0][1]=2, t[0][2]=3, t[1][0]=4, t[1][1]=5, t[1][2]=6
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

// 20260319 ZJH 测试 shapeVec 返回完整形状
TEST(TensorTest, ShapeVec) {
    auto t = df::Tensor::zeros({2, 3, 4});
    auto vecShape = t.shapeVec();
    ASSERT_EQ(vecShape.size(), 3);
    EXPECT_EQ(vecShape[0], 2);
    EXPECT_EQ(vecShape[1], 3);
    EXPECT_EQ(vecShape[2], 4);
}
```

- [ ] **Step 2: 实现 Tensor 模块**

```cpp
// 20260319 ZJH Tensor 类 — 多维张量视图
// Storage + View 分离设计：TensorStorage 持有内存，Tensor 是视图
module;

#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <cstddef>
#include <cassert>

#include "df_types.h"

export module df.engine.tensor;

import df.engine.tensor_storage;
import df.hal.cpu_backend;

export namespace df {

// 20260319 ZJH Tensor — 多维张量，Storage 之上的视图
// 工厂方法创建，支持多维索引访问，视图操作共享 Storage
class Tensor {
public:
    // ==================== 工厂方法 ====================

    // 20260319 ZJH 创建全零张量
    static Tensor zeros(std::vector<int> vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        CPUBackend::fillZeros(t.mutableDataPtr<float>(), t.numel());
        return t;
    }

    // 20260319 ZJH 创建全一张量
    static Tensor ones(std::vector<int> vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        CPUBackend::fillOnes(t.mutableDataPtr<float>(), t.numel());
        return t;
    }

    // 20260319 ZJH 创建指定值填充的张量
    static Tensor full(std::vector<int> vecShape, float fValue) {
        Tensor t;
        t.initContiguous(vecShape);
        CPUBackend::fillValue(t.mutableDataPtr<float>(), t.numel(), fValue);
        return t;
    }

    // 20260319 ZJH 创建正态分布随机张量
    static Tensor randn(std::vector<int> vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        CPUBackend::fillRandn(t.mutableDataPtr<float>(), t.numel());
        return t;
    }

    // 20260319 ZJH 从现有 float 数据创建张量（拷贝数据）
    static Tensor fromData(const float* pData, std::vector<int> vecShape) {
        Tensor t;
        t.initContiguous(vecShape);
        CPUBackend::copy(pData, t.mutableDataPtr<float>(), t.numel());
        return t;
    }

    // ==================== 属性查询 ====================

    // 20260319 ZJH 维度数
    int ndim() const { return static_cast<int>(m_vecShape.size()); }

    // 20260319 ZJH 第 nDim 维的大小
    int shape(int nDim) const { return m_vecShape[nDim]; }

    // 20260319 ZJH 返回完整形状向量
    std::vector<int> shapeVec() const { return m_vecShape; }

    // 20260319 ZJH 第 nDim 维的步长（元素为单位）
    int stride(int nDim) const { return m_vecStrides[nDim]; }

    // 20260319 ZJH 返回完整步长向量
    std::vector<int> stridesVec() const { return m_vecStrides; }

    // 20260319 ZJH 总元素数
    int numel() const {
        if (m_vecShape.empty()) return 0;
        int n = 1;
        for (int s : m_vecShape) n *= s;
        return n;
    }

    // 20260319 ZJH 数据类型
    DataType dtype() const { return m_dtype; }

    // 20260319 ZJH 设备类型
    DeviceType device() const { return m_deviceType; }

    // 20260319 ZJH 是否连续内存（行主序 C-contiguous）
    bool isContiguous() const {
        if (m_vecShape.empty()) return true;
        int nExpected = 1;
        // 20260319 ZJH 从最后一维到第一维检查步长是否递增
        for (int d = ndim() - 1; d >= 0; --d) {
            if (m_vecStrides[d] != nExpected) return false;
            nExpected *= m_vecShape[d];
        }
        return true;
    }

    // ==================== 数据访问 ====================

    // 20260319 ZJH 获取只读数据指针（带偏移）
    template<typename T = float>
    const T* dataPtr() const {
        auto* pBase = static_cast<const T*>(m_pStorage->data());
        return pBase + m_nOffset;
    }

    // 20260319 ZJH 获取可写数据指针（带偏移）
    template<typename T = float>
    T* mutableDataPtr() {
        auto* pBase = static_cast<T*>(m_pStorage->mutableData());
        return pBase + m_nOffset;
    }

    // 20260319 ZJH 多维索引访问单个元素值
    // 参数: vecIndices — 各维索引，如 {i, j, k}
    float at(std::vector<int> vecIndices) const {
        int nIdx = m_nOffset;
        for (int d = 0; d < ndim(); ++d) {
            nIdx += vecIndices[d] * m_vecStrides[d];
        }
        return static_cast<const float*>(m_pStorage->data())[nIdx];
    }

    // 20260319 ZJH 设置单个元素值
    void setAt(std::vector<int> vecIndices, float fValue) {
        int nIdx = m_nOffset;
        for (int d = 0; d < ndim(); ++d) {
            nIdx += vecIndices[d] * m_vecStrides[d];
        }
        static_cast<float*>(m_pStorage->mutableData())[nIdx] = fValue;
    }

    // ==================== 连续化 ====================

    // 20260319 ZJH 返回连续内存版本（如已连续则返回自身，否则拷贝）
    Tensor contiguous() const {
        if (isContiguous()) return *this;  // 已连续，共享 Storage
        // 20260319 ZJH 非连续：分配新 Storage，用 stridedCopy 拷贝数据
        Tensor result;
        result.initContiguous(m_vecShape);
        CPUBackend::stridedCopy(
            static_cast<const float*>(m_pStorage->data()),
            result.mutableDataPtr<float>(),
            m_vecShape, m_vecStrides, m_nOffset);
        return result;
    }

    // ==================== 内部构造支持（供 tensor_ops 使用） ====================

    // 20260319 ZJH 获取 Storage 指针（供 tensor_ops 模块访问）
    std::shared_ptr<TensorStorage> storage() const { return m_pStorage; }

    // 20260319 ZJH 获取偏移量
    int offset() const { return m_nOffset; }

    // 20260319 ZJH 构造视图张量（用于 reshape/transpose/slice）
    static Tensor makeView(std::shared_ptr<TensorStorage> pStorage,
                           std::vector<int> vecShape,
                           std::vector<int> vecStrides,
                           int nOffset) {
        Tensor t;
        t.m_pStorage = pStorage;
        t.m_vecShape = vecShape;
        t.m_vecStrides = vecStrides;
        t.m_nOffset = nOffset;
        return t;
    }

    // 20260319 ZJH 默认构造（空张量）
    Tensor() = default;

private:
    // 20260319 ZJH 初始化为连续内存张量
    void initContiguous(const std::vector<int>& vecShape) {
        m_vecShape = vecShape;
        m_dtype = DataType::Float32;
        m_deviceType = DeviceType::CPU;
        m_nOffset = 0;

        // 20260319 ZJH 计算行主序步长
        m_vecStrides.resize(vecShape.size());
        if (!vecShape.empty()) {
            m_vecStrides.back() = 1;
            for (int d = static_cast<int>(vecShape.size()) - 2; d >= 0; --d) {
                m_vecStrides[d] = m_vecStrides[d + 1] * vecShape[d + 1];
            }
        }

        // 20260319 ZJH 分配 Storage
        int nNumel = 1;
        for (int s : vecShape) nNumel *= s;
        m_pStorage = std::make_shared<TensorStorage>(nNumel * sizeof(float));
    }

    std::shared_ptr<TensorStorage> m_pStorage;     // 底层内存
    std::vector<int> m_vecShape;                    // 各维大小
    std::vector<int> m_vecStrides;                  // 各维步长（元素为单位）
    int m_nOffset = 0;                              // 在 Storage 中的元素偏移
    DataType m_dtype = DataType::Float32;            // 数据类型
    DeviceType m_deviceType = DeviceType::CPU;       // 设备类型
};

}  // namespace df
```

- [ ] **Step 3: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_tensor
ctest -R test_tensor -V
```

Expected: 11 tests PASSED

- [ ] **Step 4: 提交**

```bash
git add src/engine/df.engine.tensor.ixx tests/test_tensor.cpp
git commit -m "feat(engine): add Tensor class with factory methods and data access"
```

---

## Task 5: Tensor 运算 — 元素运算、matmul、reshape、transpose、slice

**Files:**
- Create: `src/engine/df.engine.tensor_ops.ixx`
- Create: `tests/test_tensor_ops.cpp`

- [ ] **Step 1: 编写 Tensor 运算测试**

```cpp
// 20260319 ZJH Tensor 运算单元测试
#include <gtest/gtest.h>
#include <cmath>
import df.engine.tensor;
import df.engine.tensor_ops;

// ==================== 元素运算 ====================

// 20260319 ZJH 测试逐元素加法
TEST(TensorOpsTest, Add) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    auto c = df::tensorAdd(a, b);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 10.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 12.0f);
}

// 20260319 ZJH 测试逐元素减法
TEST(TensorOpsTest, Sub) {
    auto a = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {4});
    auto b = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {4});
    auto c = df::tensorSub(a, b);
    EXPECT_FLOAT_EQ(c.at({0}), 4.0f);
    EXPECT_FLOAT_EQ(c.at({3}), 4.0f);
}

// 20260319 ZJH 测试逐元素乘法
TEST(TensorOpsTest, Mul) {
    auto a = df::Tensor::fromData(std::vector<float>{2, 3, 4, 5}.data(), {4});
    auto b = df::Tensor::fromData(std::vector<float>{10, 20, 30, 40}.data(), {4});
    auto c = df::tensorMul(a, b);
    EXPECT_FLOAT_EQ(c.at({0}), 20.0f);
    EXPECT_FLOAT_EQ(c.at({3}), 200.0f);
}

// 20260319 ZJH 测试逐元素除法
TEST(TensorOpsTest, Div) {
    auto a = df::Tensor::fromData(std::vector<float>{10, 20, 30, 40}.data(), {4});
    auto b = df::Tensor::fromData(std::vector<float>{2, 4, 5, 8}.data(), {4});
    auto c = df::tensorDiv(a, b);
    EXPECT_FLOAT_EQ(c.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({3}), 5.0f);
}

// ==================== 矩阵乘法 ====================

// 20260319 ZJH 测试 2D 矩阵乘法
TEST(TensorOpsTest, Matmul2D) {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    auto c = df::tensorMatmul(a, b);
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 2);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 19.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 43.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 50.0f);
}

// 20260319 ZJH 测试非方阵矩阵乘法
TEST(TensorOpsTest, MatmulNonSquare) {
    // A: [2, 3], B: [3, 2] -> C: [2, 2]
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::Tensor::fromData(std::vector<float>{7, 8, 9, 10, 11, 12}.data(), {3, 2});
    auto c = df::tensorMatmul(a, b);
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 2);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    EXPECT_FLOAT_EQ(c.at({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 154.0f);
}

// ==================== 形状变换 ====================

// 20260319 ZJH 测试 reshape — 改变形状但共享内存
TEST(TensorOpsTest, Reshape) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::tensorReshape(a, {3, 2});
    EXPECT_EQ(b.shape(0), 3);
    EXPECT_EQ(b.shape(1), 2);
    EXPECT_EQ(b.numel(), 6);
    // 数据顺序不变
    EXPECT_FLOAT_EQ(b.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at({2, 1}), 6.0f);
}

// 20260319 ZJH 测试 reshape 到 1D
TEST(TensorOpsTest, ReshapeFlatten) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::tensorReshape(a, {4});
    EXPECT_EQ(b.ndim(), 1);
    EXPECT_EQ(b.shape(0), 4);
    EXPECT_FLOAT_EQ(b.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at({3}), 4.0f);
}

// 20260319 ZJH 测试 transpose — 交换两个维度
TEST(TensorOpsTest, Transpose) {
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::tensorTranspose(a, 0, 1);
    EXPECT_EQ(b.shape(0), 3);
    EXPECT_EQ(b.shape(1), 2);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(b.at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(b.at({2, 1}), 6.0f);
    // transpose 是视图操作，不连续
    EXPECT_FALSE(b.isContiguous());
}

// 20260319 ZJH 测试 transpose 后 contiguous
TEST(TensorOpsTest, TransposeThenContiguous) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::tensorTranspose(a, 0, 1);
    auto c = b.contiguous();
    EXPECT_TRUE(c.isContiguous());
    EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 4.0f);
}

// 20260319 ZJH 测试 slice — 在指定维度切片
TEST(TensorOpsTest, Slice) {
    // t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    auto t = df::Tensor::fromData(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {3, 3});
    // slice(dim=0, start=1, end=3) -> [[4, 5, 6], [7, 8, 9]]
    auto s = df::tensorSlice(t, 0, 1, 3);
    EXPECT_EQ(s.shape(0), 2);
    EXPECT_EQ(s.shape(1), 3);
    EXPECT_FLOAT_EQ(s.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(s.at({1, 2}), 9.0f);
}

// 20260319 ZJH 测试 slice 第二维
TEST(TensorOpsTest, SliceDim1) {
    auto t = df::Tensor::fromData(
        std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    // slice(dim=1, start=0, end=2) -> [[1, 2], [4, 5]]
    auto s = df::tensorSlice(t, 1, 0, 2);
    EXPECT_EQ(s.shape(0), 2);
    EXPECT_EQ(s.shape(1), 2);
    EXPECT_FLOAT_EQ(s.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(s.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(s.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(s.at({1, 1}), 5.0f);
}

// 20260319 ZJH 测试标量运算
TEST(TensorOpsTest, ScalarOps) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3}.data(), {3});
    auto b = df::tensorAddScalar(a, 10.0f);
    EXPECT_FLOAT_EQ(b.at({0}), 11.0f);
    EXPECT_FLOAT_EQ(b.at({2}), 13.0f);

    auto c = df::tensorMulScalar(a, 3.0f);
    EXPECT_FLOAT_EQ(c.at({0}), 3.0f);
    EXPECT_FLOAT_EQ(c.at({2}), 9.0f);
}

// 20260319 ZJH 测试归约运算
TEST(TensorOpsTest, Reductions) {
    auto t = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5}.data(), {5});
    EXPECT_FLOAT_EQ(df::tensorSum(t), 15.0f);
    EXPECT_FLOAT_EQ(df::tensorMax(t), 5.0f);
    EXPECT_FLOAT_EQ(df::tensorMin(t), 1.0f);
}
```

- [ ] **Step 2: 实现 tensor_ops 模块**

```cpp
// 20260319 ZJH Tensor 运算模块 — 元素运算、matmul、reshape、transpose、slice
// 所有运算委托给 CPUBackend 执行
module;

#include <vector>
#include <stdexcept>
#include <memory>
#include <cassert>

#include "df_types.h"

export module df.engine.tensor_ops;

import df.engine.tensor_storage;
import df.engine.tensor;
import df.hal.cpu_backend;

export namespace df {

// ==================== 元素级运算 ====================
// 前置条件：形状相同，连续内存（非连续自动 contiguous()）

// 20260319 ZJH 逐元素加法
Tensor tensorAdd(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();  // 确保连续
    auto cb = b.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::add(ca.dataPtr<float>(), cb.dataPtr<float>(),
                    result.mutableDataPtr<float>(), result.numel());
    return result;
}

// 20260319 ZJH 逐元素减法
Tensor tensorSub(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();
    auto cb = b.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::sub(ca.dataPtr<float>(), cb.dataPtr<float>(),
                    result.mutableDataPtr<float>(), result.numel());
    return result;
}

// 20260319 ZJH 逐元素乘法（Hadamard积）
Tensor tensorMul(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();
    auto cb = b.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::mul(ca.dataPtr<float>(), cb.dataPtr<float>(),
                    result.mutableDataPtr<float>(), result.numel());
    return result;
}

// 20260319 ZJH 逐元素除法
Tensor tensorDiv(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();
    auto cb = b.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::div(ca.dataPtr<float>(), cb.dataPtr<float>(),
                    result.mutableDataPtr<float>(), result.numel());
    return result;
}

// 20260319 ZJH 标量加法
Tensor tensorAddScalar(const Tensor& a, float fScalar) {
    auto ca = a.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::addScalar(ca.dataPtr<float>(), fScalar,
                          result.mutableDataPtr<float>(), result.numel());
    return result;
}

// 20260319 ZJH 标量乘法
Tensor tensorMulScalar(const Tensor& a, float fScalar) {
    auto ca = a.contiguous();
    auto result = Tensor::zeros(ca.shapeVec());
    CPUBackend::mulScalar(ca.dataPtr<float>(), fScalar,
                          result.mutableDataPtr<float>(), result.numel());
    return result;
}

// ==================== 矩阵乘法 ====================

// 20260319 ZJH 2D 矩阵乘法: C = A @ B
// A: [M, K], B: [K, N] -> C: [M, N]
Tensor tensorMatmul(const Tensor& a, const Tensor& b) {
    auto ca = a.contiguous();
    auto cb = b.contiguous();

    int nM = ca.shape(0);
    int nK = ca.shape(1);
    int nN = cb.shape(1);

    auto result = Tensor::zeros({nM, nN});
    CPUBackend::matmul(ca.dataPtr<float>(), cb.dataPtr<float>(),
                       result.mutableDataPtr<float>(), nM, nK, nN);
    return result;
}

// ==================== 形状变换 ====================

// 20260319 ZJH reshape — 改变形状，连续张量零拷贝（共享 Storage）
Tensor tensorReshape(const Tensor& t, std::vector<int> vecNewShape) {
    // 20260319 ZJH 验证元素数量不变
    int nNewNumel = 1;
    for (int s : vecNewShape) nNewNumel *= s;
    if (nNewNumel != t.numel()) {
        throw std::invalid_argument("reshape: total element count mismatch");
    }

    if (t.isContiguous()) {
        // 20260319 ZJH 连续张量：直接创建新视图，共享 Storage
        std::vector<int> vecNewStrides(vecNewShape.size());
        if (!vecNewShape.empty()) {
            vecNewStrides.back() = 1;
            for (int d = static_cast<int>(vecNewShape.size()) - 2; d >= 0; --d) {
                vecNewStrides[d] = vecNewStrides[d + 1] * vecNewShape[d + 1];
            }
        }
        return Tensor::makeView(t.storage(), vecNewShape, vecNewStrides, t.offset());
    } else {
        // 20260319 ZJH 非连续张量：先拷贝为连续，再 reshape
        auto ct = t.contiguous();
        return tensorReshape(ct, vecNewShape);
    }
}

// 20260319 ZJH transpose — 交换两个维度（零拷贝视图操作）
Tensor tensorTranspose(const Tensor& t, int nDim0, int nDim1) {
    auto vecShape = t.shapeVec();
    auto vecStrides = t.stridesVec();
    // 20260319 ZJH 交换指定维度的 shape 和 strides
    std::swap(vecShape[nDim0], vecShape[nDim1]);
    std::swap(vecStrides[nDim0], vecStrides[nDim1]);
    return Tensor::makeView(t.storage(), vecShape, vecStrides, t.offset());
}

// 20260319 ZJH slice — 在指定维度上切片 [nStart, nEnd)
// 零拷贝视图操作：调整 offset 和 shape
Tensor tensorSlice(const Tensor& t, int nDim, int nStart, int nEnd) {
    auto vecShape = t.shapeVec();
    auto vecStrides = t.stridesVec();
    // 20260319 ZJH 偏移增加 start * stride（跳过前 start 个元素）
    int nNewOffset = t.offset() + nStart * vecStrides[nDim];
    // 20260319 ZJH 该维大小变为 end - start
    vecShape[nDim] = nEnd - nStart;
    return Tensor::makeView(t.storage(), vecShape, vecStrides, nNewOffset);
}

// ==================== 归约运算 ====================

// 20260319 ZJH 全局求和
float tensorSum(const Tensor& t) {
    auto ct = t.contiguous();
    return CPUBackend::sum(ct.dataPtr<float>(), ct.numel());
}

// 20260319 ZJH 全局最大值
float tensorMax(const Tensor& t) {
    auto ct = t.contiguous();
    return CPUBackend::max(ct.dataPtr<float>(), ct.numel());
}

// 20260319 ZJH 全局最小值
float tensorMin(const Tensor& t) {
    auto ct = t.contiguous();
    return CPUBackend::min(ct.dataPtr<float>(), ct.numel());
}

}  // namespace df
```

- [ ] **Step 3: 编译并运行测试**

```bash
cmake --build build/windows-debug --target test_tensor_ops
ctest -R test_tensor_ops -V
```

Expected: 16 tests PASSED

- [ ] **Step 4: 提交**

```bash
git add src/engine/df.engine.tensor_ops.ixx tests/test_tensor_ops.cpp
git commit -m "feat(engine): add Tensor operations — arithmetic, matmul, reshape, transpose, slice"
```

---

## Task 6: 全量验证 + 运行全部测试

- [ ] **Step 1: 全量构建**

```bash
cmake --preset windows-debug
cmake --build build/windows-debug
```

- [ ] **Step 2: 运行全部测试**

```bash
ctest --output-on-failure
```

Expected: 所有测试通过
- Phase 1A: Logger 4 + Config 5 + FileSystem 6 + Memory 5 + ThreadPool 5 + Database 5 = 30
- Phase 1B: Tensor 11 + TensorOps 16 = 27
- **总计: 57 tests**

- [ ] **Step 3: 更新 DEVLOG.md**

- [ ] **Step 4: 提交**

```bash
git add DEVLOG.md
git commit -m "milestone: Phase 1B complete — Tensor + CPUBackend (57/57 tests passing)"
```

---

## Summary

| Task | 模块 | 测试数 | 关键验证点 |
|------|------|--------|-----------|
| 1 | CMake 配置 | — | df_hal + df_engine 库编译成功 |
| 2 | TensorStorage | — | 间接通过 Tensor 测试覆盖 |
| 3 | CPUBackend | — | 间接通过 TensorOps 测试覆盖 |
| 4 | Tensor | 11 | 工厂方法 + 属性 + 数据访问 + strides + contiguous |
| 5 | TensorOps | 16 | 加减乘除 + matmul + reshape + transpose + slice + 归约 |
| 6 | 全量验证 | 57 | Phase 1A + 1B 全部通过 |

### 设计要点回顾

1. **Storage + View 分离**：reshape/transpose/slice 是零拷贝视图操作，共享底层内存
2. **contiguous()** 按需转换：非连续视图在参与运算前自动拷贝为连续内存
3. **自由函数 vs 成员方法**：运算使用 `tensorAdd(a, b)` 而非 `a.add(b)`，便于 Phase 1C 添加 AutoGrad 时在函数内部挂入计算图，无需修改 Tensor 类
4. **仅 Float32**：其他 dtype 预留接口，Phase 2 实现
