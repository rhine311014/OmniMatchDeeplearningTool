// 20260319 ZJH Tensor 类单元测试 — Phase 1B-T4，共 11 个测试用例
// 覆盖：工厂方法、属性访问、步长计算、连续性判断、随机访问
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

import df.engine.tensor;

// ============================================================
// Test 1 — zeros
// 20260319 ZJH 验证 zeros 工厂方法：形状、维度、元素数、类型、设备、所有值为 0
// ============================================================
TEST(TensorTest, Zeros) {
    // 20260319 ZJH 创建 2×3 的全零张量
    df::Tensor t = df::Tensor::zeros({2, 3});

    // 20260319 ZJH 验证维度数
    EXPECT_EQ(t.ndim(), 2);

    // 20260319 ZJH 验证各维度形状
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);

    // 20260319 ZJH 验证元素总数
    EXPECT_EQ(t.numel(), 6);

    // 20260319 ZJH 验证数据类型为 Float32
    EXPECT_EQ(t.dtype(), df::DataType::Float32);

    // 20260319 ZJH 验证设备类型为 CPU
    EXPECT_EQ(t.device(), df::DeviceType::CPU);

    // 20260319 ZJH 遍历所有元素，验证均为 0.0f
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(pData[i], 0.0f);
    }
}

// ============================================================
// Test 2 — ones
// 20260319 ZJH 验证 ones 工厂方法：元素数正确，所有值为 1.0f
// ============================================================
TEST(TensorTest, Ones) {
    // 20260319 ZJH 创建 3×4 的全 1 张量
    df::Tensor t = df::Tensor::ones({3, 4});

    // 20260319 ZJH 验证元素总数
    EXPECT_EQ(t.numel(), 12);

    // 20260319 ZJH 遍历所有元素，验证均为 1.0f
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(pData[i], 1.0f);
    }
}

// ============================================================
// Test 3 — full
// 20260319 ZJH 验证 full 工厂方法：所有值填充为指定标量
// ============================================================
TEST(TensorTest, Full) {
    // 20260319 ZJH 创建 2×2 的张量，填充值为 3.14
    df::Tensor t = df::Tensor::full({2, 2}, 3.14f);

    // 20260319 ZJH 验证所有元素均为 3.14f
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(pData[i], 3.14f);
    }
}

// ============================================================
// Test 4 — randn
// 20260319 ZJH 验证 randn 工厂方法：元素数正确，绝对值之和大于 0（非全零）
// ============================================================
TEST(TensorTest, Randn) {
    // 20260319 ZJH 创建长度为 100 的随机张量
    df::Tensor t = df::Tensor::randn({100});

    // 20260319 ZJH 验证元素总数
    EXPECT_EQ(t.numel(), 100);

    // 20260319 ZJH 计算所有元素绝对值之和，验证正态随机数非全零
    const float* pData = t.floatDataPtr();
    float fAbsSum = 0.0f;
    for (int i = 0; i < t.numel(); ++i) {
        fAbsSum += std::abs(pData[i]);
    }
    EXPECT_GT(fAbsSum, 0.0f);  // 20260319 ZJH 正态分布不可能全为零
}

// ============================================================
// Test 5 — fromData
// 20260319 ZJH 验证 fromData 工厂方法：形状正确，数据与源一致
// ============================================================
TEST(TensorTest, FromData) {
    // 20260319 ZJH 源数据：{1,2,3,4,5,6}，目标形状 {2,3}
    float arrSrc[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    df::Tensor t = df::Tensor::fromData(arrSrc, {2, 3});

    // 20260319 ZJH 验证形状
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);

    // 20260319 ZJH 验证数据值与源一致
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(pData[i], arrSrc[i]);
    }
}

// ============================================================
// Test 6 — OneDimensional
// 20260319 ZJH 验证一维张量：ndim=1，shape(0)=5
// ============================================================
TEST(TensorTest, OneDimensional) {
    // 20260319 ZJH 创建长度为 5 的一维全零张量
    df::Tensor t = df::Tensor::zeros({5});

    // 20260319 ZJH 验证维度数为 1
    EXPECT_EQ(t.ndim(), 1);

    // 20260319 ZJH 验证唯一维度的大小为 5
    EXPECT_EQ(t.shape(0), 5);
}

// ============================================================
// Test 7 — ThreeDimensional
// 20260319 ZJH 验证三维张量：ndim=3，numel=24
// ============================================================
TEST(TensorTest, ThreeDimensional) {
    // 20260319 ZJH 创建 2×3×4 的三维全零张量
    df::Tensor t = df::Tensor::zeros({2, 3, 4});

    // 20260319 ZJH 验证维度数为 3
    EXPECT_EQ(t.ndim(), 3);

    // 20260319 ZJH 验证元素总数 2*3*4 = 24
    EXPECT_EQ(t.numel(), 24);
}

// ============================================================
// Test 8 — Strides
// 20260319 ZJH 验证行主序步长计算：{2,3,4} 对应步长应为 {12,4,1}
// ============================================================
TEST(TensorTest, Strides) {
    // 20260319 ZJH 创建 2×3×4 的三维张量
    df::Tensor t = df::Tensor::zeros({2, 3, 4});

    // 20260319 ZJH 行主序步长：stride[2]=1, stride[1]=4, stride[0]=12
    EXPECT_EQ(t.stride(0), 12);  // 20260319 ZJH 第 0 维步长 = 3*4
    EXPECT_EQ(t.stride(1),  4);  // 20260319 ZJH 第 1 维步长 = 4
    EXPECT_EQ(t.stride(2),  1);  // 20260319 ZJH 第 2 维步长 = 1（最低维）
}

// ============================================================
// Test 9 — IsContiguous
// 20260319 ZJH 验证新建张量为连续存储
// ============================================================
TEST(TensorTest, IsContiguous) {
    // 20260319 ZJH 工厂方法创建的张量应始终是连续的
    df::Tensor t = df::Tensor::zeros({2, 3, 4});
    EXPECT_TRUE(t.isContiguous());
}

// ============================================================
// Test 10 — AtAccess
// 20260319 ZJH 验证多维索引随机访问正确性
// ============================================================
TEST(TensorTest, AtAccess) {
    // 20260319 ZJH 从 {1,2,3,4,5,6} 构建 2×3 张量
    float arrSrc[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    df::Tensor t = df::Tensor::fromData(arrSrc, {2, 3});

    // 20260319 ZJH 验证 at({0,0}) == 1.0f（第一行第一列）
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);

    // 20260319 ZJH 验证 at({0,2}) == 3.0f（第一行第三列）
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);

    // 20260319 ZJH 验证 at({1,0}) == 4.0f（第二行第一列）
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);

    // 20260319 ZJH 验证 at({1,2}) == 6.0f（第二行第三列）
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

// ============================================================
// Test 11 — ShapeVec
// 20260319 ZJH 验证 shapeVec() 返回完整形状向量
// ============================================================
TEST(TensorTest, ShapeVec) {
    // 20260319 ZJH 创建 2×3×4 的张量
    df::Tensor t = df::Tensor::zeros({2, 3, 4});

    // 20260319 ZJH 获取形状向量
    const std::vector<int>& vecShape = t.shapeVec();

    // 20260319 ZJH 验证形状向量内容为 {2, 3, 4}
    ASSERT_EQ(static_cast<int>(vecShape.size()), 3);
    EXPECT_EQ(vecShape[0], 2);
    EXPECT_EQ(vecShape[1], 3);
    EXPECT_EQ(vecShape[2], 4);
}
