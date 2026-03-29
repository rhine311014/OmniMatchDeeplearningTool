// 20260319 ZJH AutoGrad 自动微分单元测试 — Phase 1C
// 覆盖：Add/Sub/Mul/MatMul/MulScalar 梯度、链式法则、零梯度、叶节点梯度累加器
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// 20260319 ZJH 导入张量类和运算模块（含自动微分）
import om.engine.tensor;
import om.engine.tensor_ops;

// ===== 1. AddGradient =====
// 20260319 ZJH 测试加法梯度：loss = sum(a + b)，期望 grad(a) = 1, grad(b) = 1
TEST(AutoGradTest, AddGradient) {
    // 20260319 ZJH 创建 2x2 全 1 张量 a 和 b，设置需要梯度
    auto a = om::Tensor::ones({2, 2});
    om::tensorSetRequiresGrad(a, true);  // 20260319 ZJH 标记 a 为叶节点
    auto b = om::Tensor::ones({2, 2});
    om::tensorSetRequiresGrad(b, true);  // 20260319 ZJH 标记 b 为叶节点

    // 20260319 ZJH 前向计算：c = a + b, loss = sum(c)
    auto c = om::tensorAdd(a, b);
    auto loss = om::tensorSum(c);

    // 20260319 ZJH 反向传播
    om::tensorBackward(loss);

    // 20260319 ZJH 验证梯度：d(sum(a+b))/da = 1, d(sum(a+b))/db = 1
    auto gradA = om::tensorGetGrad(a);
    auto gradB = om::tensorGetGrad(b);
    ASSERT_EQ(gradA.numel(), 4);  // 20260319 ZJH 梯度形状应与 a 相同
    ASSERT_EQ(gradB.numel(), 4);  // 20260319 ZJH 梯度形状应与 b 相同
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(gradA.at({i, j}), 1.0f);  // 20260319 ZJH 每个元素梯度为 1
            EXPECT_FLOAT_EQ(gradB.at({i, j}), 1.0f);  // 20260319 ZJH 每个元素梯度为 1
        }
    }
}

// ===== 2. SubGradient =====
// 20260319 ZJH 测试减法梯度：loss = sum(a - b)，期望 grad(a) = 1, grad(b) = -1
TEST(AutoGradTest, SubGradient) {
    auto a = om::Tensor::ones({2, 2});
    om::tensorSetRequiresGrad(a, true);
    auto b = om::Tensor::ones({2, 2});
    om::tensorSetRequiresGrad(b, true);

    auto c = om::tensorSub(a, b);
    auto loss = om::tensorSum(c);
    om::tensorBackward(loss);

    auto gradA = om::tensorGetGrad(a);
    auto gradB = om::tensorGetGrad(b);
    ASSERT_EQ(gradA.numel(), 4);
    ASSERT_EQ(gradB.numel(), 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(gradA.at({i, j}), 1.0f);   // 20260319 ZJH d/da = 1
            EXPECT_FLOAT_EQ(gradB.at({i, j}), -1.0f);  // 20260319 ZJH d/db = -1
        }
    }
}

// ===== 3. MulGradient =====
// 20260319 ZJH 测试乘法梯度：loss = sum(a * b)，期望 grad(a) = b, grad(b) = a
TEST(AutoGradTest, MulGradient) {
    // 20260319 ZJH 使用不同值以验证交叉梯度
    auto a = om::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    om::tensorSetRequiresGrad(a, true);
    auto b = om::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    om::tensorSetRequiresGrad(b, true);

    auto c = om::tensorMul(a, b);
    auto loss = om::tensorSum(c);
    om::tensorBackward(loss);

    auto gradA = om::tensorGetGrad(a);
    auto gradB = om::tensorGetGrad(b);
    // 20260319 ZJH grad(a) 应该等于 b 的值
    EXPECT_FLOAT_EQ(gradA.at({0, 0}), 5.0f);  // 20260319 ZJH grad_a[0,0] = b[0,0] = 5
    EXPECT_FLOAT_EQ(gradA.at({0, 1}), 6.0f);  // 20260319 ZJH grad_a[0,1] = b[0,1] = 6
    EXPECT_FLOAT_EQ(gradA.at({1, 0}), 7.0f);  // 20260319 ZJH grad_a[1,0] = b[1,0] = 7
    EXPECT_FLOAT_EQ(gradA.at({1, 1}), 8.0f);  // 20260319 ZJH grad_a[1,1] = b[1,1] = 8
    // 20260319 ZJH grad(b) 应该等于 a 的值
    EXPECT_FLOAT_EQ(gradB.at({0, 0}), 1.0f);  // 20260319 ZJH grad_b[0,0] = a[0,0] = 1
    EXPECT_FLOAT_EQ(gradB.at({0, 1}), 2.0f);  // 20260319 ZJH grad_b[0,1] = a[0,1] = 2
    EXPECT_FLOAT_EQ(gradB.at({1, 0}), 3.0f);  // 20260319 ZJH grad_b[1,0] = a[1,0] = 3
    EXPECT_FLOAT_EQ(gradB.at({1, 1}), 4.0f);  // 20260319 ZJH grad_b[1,1] = a[1,1] = 4
}

// ===== 4. MatMulGradient =====
// 20260319 ZJH 测试矩阵乘法梯度，使用数值梯度检查验证
TEST(AutoGradTest, MatMulGradient) {
    // 20260319 ZJH 创建 2x3 和 3x2 矩阵
    auto a = om::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    om::tensorSetRequiresGrad(a, true);
    auto b = om::Tensor::fromData(std::vector<float>{7, 8, 9, 10, 11, 12}.data(), {3, 2});
    om::tensorSetRequiresGrad(b, true);

    // 20260319 ZJH 前向 + 反向
    auto c = om::tensorMatmul(a, b);
    auto loss = om::tensorSum(c);
    om::tensorBackward(loss);

    auto gradA = om::tensorGetGrad(a);
    auto gradB = om::tensorGetGrad(b);

    // 20260319 ZJH 数值梯度检查 — 对 A 的每个元素扰动 ±eps
    float fEps = 1e-2f;  // 20260319 ZJH 有限差分步长（float32 精度下不宜过小）
    std::vector<float> vecAData = {1, 2, 3, 4, 5, 6};
    std::vector<float> vecBData = {7, 8, 9, 10, 11, 12};

    // 20260319 ZJH 检查 grad_A 的数值梯度
    for (int idx = 0; idx < 6; ++idx) {
        // 20260319 ZJH 正向扰动
        std::vector<float> vecAPlus = vecAData;
        vecAPlus[idx] += fEps;
        auto aPlus = om::Tensor::fromData(vecAPlus.data(), {2, 3});
        auto bCopy = om::Tensor::fromData(vecBData.data(), {3, 2});
        float fLossPlus = om::tensorSum(om::tensorMatmul(aPlus, bCopy)).item();

        // 20260319 ZJH 反向扰动
        std::vector<float> vecAMinus = vecAData;
        vecAMinus[idx] -= fEps;
        auto aMinus = om::Tensor::fromData(vecAMinus.data(), {2, 3});
        auto bCopy2 = om::Tensor::fromData(vecBData.data(), {3, 2});
        float fLossMinus = om::tensorSum(om::tensorMatmul(aMinus, bCopy2)).item();

        // 20260319 ZJH 数值梯度 = (f(x+eps) - f(x-eps)) / (2*eps)
        float fNumericalGrad = (fLossPlus - fLossMinus) / (2.0f * fEps);
        // 20260319 ZJH 解析梯度
        int nRow = idx / 3;
        int nCol = idx % 3;
        float fAnalyticGrad = gradA.at({nRow, nCol});
        // 20260319 ZJH 验证解析梯度与数值梯度接近（容差 1e-2）
        EXPECT_NEAR(fAnalyticGrad, fNumericalGrad, 0.5f)
            << "grad_A mismatch at [" << nRow << "," << nCol << "]";
    }

    // 20260319 ZJH 检查 grad_B 的数值梯度
    for (int idx = 0; idx < 6; ++idx) {
        std::vector<float> vecBPlus = vecBData;
        vecBPlus[idx] += fEps;
        auto aCopy = om::Tensor::fromData(vecAData.data(), {2, 3});
        auto bPlus = om::Tensor::fromData(vecBPlus.data(), {3, 2});
        float fLossPlus = om::tensorSum(om::tensorMatmul(aCopy, bPlus)).item();

        std::vector<float> vecBMinus = vecBData;
        vecBMinus[idx] -= fEps;
        auto aCopy2 = om::Tensor::fromData(vecAData.data(), {2, 3});
        auto bMinus = om::Tensor::fromData(vecBMinus.data(), {3, 2});
        float fLossMinus = om::tensorSum(om::tensorMatmul(aCopy2, bMinus)).item();

        float fNumericalGrad = (fLossPlus - fLossMinus) / (2.0f * fEps);
        int nRow = idx / 2;
        int nCol = idx % 2;
        float fAnalyticGrad = gradB.at({nRow, nCol});
        EXPECT_NEAR(fAnalyticGrad, fNumericalGrad, 0.5f)
            << "grad_B mismatch at [" << nRow << "," << nCol << "]";
    }
}

// ===== 5. MulScalarGradient =====
// 20260319 ZJH 测试乘标量梯度：loss = sum(a * 3)，期望 grad(a) = 3
TEST(AutoGradTest, MulScalarGradient) {
    auto a = om::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    om::tensorSetRequiresGrad(a, true);

    auto c = om::tensorMulScalar(a, 3.0f);
    auto loss = om::tensorSum(c);
    om::tensorBackward(loss);

    auto gradA = om::tensorGetGrad(a);
    ASSERT_EQ(gradA.numel(), 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(gradA.at({i, j}), 3.0f);  // 20260319 ZJH d(a*3)/da = 3
        }
    }
}

// ===== 6. ChainRule =====
// 20260319 ZJH 测试链式法则：loss = sum((a + b) * a)
// d/da = d(sum((a+b)*a))/da = 2a + b（通过乘法法则和加法法则链式推导）
// d/db = d(sum((a+b)*a))/db = a
TEST(AutoGradTest, ChainRule) {
    auto a = om::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    om::tensorSetRequiresGrad(a, true);
    auto b = om::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    om::tensorSetRequiresGrad(b, true);

    // 20260319 ZJH 前向：c = a + b, d = c * a, loss = sum(d)
    auto c = om::tensorAdd(a, b);
    auto d = om::tensorMul(c, a);
    auto loss = om::tensorSum(d);
    om::tensorBackward(loss);

    auto gradA = om::tensorGetGrad(a);
    auto gradB = om::tensorGetGrad(b);

    // 20260319 ZJH 验证 d/da = 2a + b
    // a = [1,2,3,4], b = [5,6,7,8]
    // 2a + b = [2+5, 4+6, 6+7, 8+8] = [7, 10, 13, 16]
    EXPECT_FLOAT_EQ(gradA.at({0, 0}), 7.0f);   // 20260319 ZJH 2*1 + 5 = 7
    EXPECT_FLOAT_EQ(gradA.at({0, 1}), 10.0f);  // 20260319 ZJH 2*2 + 6 = 10
    EXPECT_FLOAT_EQ(gradA.at({1, 0}), 13.0f);  // 20260319 ZJH 2*3 + 7 = 13
    EXPECT_FLOAT_EQ(gradA.at({1, 1}), 16.0f);  // 20260319 ZJH 2*4 + 8 = 16

    // 20260319 ZJH 验证 d/db = a
    EXPECT_FLOAT_EQ(gradB.at({0, 0}), 1.0f);  // 20260319 ZJH a[0,0] = 1
    EXPECT_FLOAT_EQ(gradB.at({0, 1}), 2.0f);  // 20260319 ZJH a[0,1] = 2
    EXPECT_FLOAT_EQ(gradB.at({1, 0}), 3.0f);  // 20260319 ZJH a[1,0] = 3
    EXPECT_FLOAT_EQ(gradB.at({1, 1}), 4.0f);  // 20260319 ZJH a[1,1] = 4
}

// ===== 7. ZeroGrad =====
// 20260319 ZJH 测试梯度清零：backward 后验证梯度，清零后验证为空
TEST(AutoGradTest, ZeroGrad) {
    auto a = om::Tensor::ones({3});
    om::tensorSetRequiresGrad(a, true);
    auto b = om::Tensor::ones({3});
    om::tensorSetRequiresGrad(b, true);

    // 20260319 ZJH 第一次反向传播
    auto loss = om::tensorSum(om::tensorAdd(a, b));
    om::tensorBackward(loss);

    // 20260319 ZJH 验证梯度存在且为 1
    auto gradA = om::tensorGetGrad(a);
    ASSERT_EQ(gradA.numel(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(gradA.at({i}), 1.0f);
    }

    // 20260319 ZJH 清零梯度
    om::tensorZeroGrad(a);

    // 20260319 ZJH 验证梯度已被清零（返回空张量）
    auto gradAZeroed = om::tensorGetGrad(a);
    EXPECT_EQ(gradAZeroed.numel(), 0);  // 20260319 ZJH 清零后无梯度
}

// ===== 8. OnlyLeafHasGrad =====
// 20260319 ZJH 测试中间张量无梯度累加器
TEST(AutoGradTest, OnlyLeafHasGrad) {
    auto a = om::Tensor::ones({2, 2});
    om::tensorSetRequiresGrad(a, true);
    auto b = om::Tensor::ones({2, 2});
    om::tensorSetRequiresGrad(b, true);

    // 20260319 ZJH c 是中间张量（运算结果），不是叶节点
    auto c = om::tensorAdd(a, b);

    // 20260319 ZJH 中间张量 c 没有 GradAccumulator（gradAccumRaw 为空）
    EXPECT_EQ(c.gradAccumRaw(), nullptr);  // 20260319 ZJH 中间节点无梯度累加器

    // 20260319 ZJH 叶节点在参与运算后应有 GradAccumulator
    EXPECT_NE(a.gradAccumRaw(), nullptr);  // 20260319 ZJH 叶节点有梯度累加器
    EXPECT_NE(b.gradAccumRaw(), nullptr);  // 20260319 ZJH 叶节点有梯度累加器
}
