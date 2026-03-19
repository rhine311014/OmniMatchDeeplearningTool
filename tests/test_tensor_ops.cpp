// 20260319 ZJH Tensor 运算单元测试 — Phase 1B-T5，共 16 个测试用例
// 覆盖范围：元素运算（add/sub/mul/div）、matmul、reshape、transpose、slice、标量运算、归约
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// 20260319 ZJH 导入张量类和运算模块（C++20 模块导入）
import df.engine.tensor;
import df.engine.tensor_ops;

// ===== 元素运算 =====

// 20260319 ZJH 测试逐元素加法：[1,2,3,4] + [5,6,7,8] = [6,8,10,12]（2x2 矩阵）
TEST(TensorOpsTest, Add) {
    // 20260319 ZJH 构造 2x2 输入张量 a 和 b
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    auto c = df::tensorAdd(a, b);  // 20260319 ZJH 执行逐元素加法
    EXPECT_FLOAT_EQ(c.at({0, 0}), 6.0f);   // 20260319 ZJH 1+5=6
    EXPECT_FLOAT_EQ(c.at({0, 1}), 8.0f);   // 20260319 ZJH 2+6=8
    EXPECT_FLOAT_EQ(c.at({1, 0}), 10.0f);  // 20260319 ZJH 3+7=10
    EXPECT_FLOAT_EQ(c.at({1, 1}), 12.0f);  // 20260319 ZJH 4+8=12
}

// 20260319 ZJH 测试逐元素减法：[5,6,7,8] - [1,2,3,4] = [4,4,4,4]（1D 向量）
TEST(TensorOpsTest, Sub) {
    auto a = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {4});
    auto b = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {4});
    auto c = df::tensorSub(a, b);  // 20260319 ZJH 执行逐元素减法
    EXPECT_FLOAT_EQ(c.at({0}), 4.0f);  // 20260319 ZJH 5-1=4
    EXPECT_FLOAT_EQ(c.at({3}), 4.0f);  // 20260319 ZJH 8-4=4
}

// 20260319 ZJH 测试逐元素乘法：[2,3,4,5] * [10,20,30,40] = [20,60,120,200]
TEST(TensorOpsTest, Mul) {
    auto a = df::Tensor::fromData(std::vector<float>{2, 3, 4, 5}.data(), {4});
    auto b = df::Tensor::fromData(std::vector<float>{10, 20, 30, 40}.data(), {4});
    auto c = df::tensorMul(a, b);  // 20260319 ZJH 执行 Hadamard 积
    EXPECT_FLOAT_EQ(c.at({0}), 20.0f);   // 20260319 ZJH 2*10=20
    EXPECT_FLOAT_EQ(c.at({3}), 200.0f);  // 20260319 ZJH 5*40=200
}

// 20260319 ZJH 测试逐元素除法：[10,20,30,40] / [2,4,5,8] = [5,5,6,5]
TEST(TensorOpsTest, Div) {
    auto a = df::Tensor::fromData(std::vector<float>{10, 20, 30, 40}.data(), {4});
    auto b = df::Tensor::fromData(std::vector<float>{2, 4, 5, 8}.data(), {4});
    auto c = df::tensorDiv(a, b);  // 20260319 ZJH 执行逐元素除法
    EXPECT_FLOAT_EQ(c.at({0}), 5.0f);  // 20260319 ZJH 10/2=5
    EXPECT_FLOAT_EQ(c.at({3}), 5.0f);  // 20260319 ZJH 40/8=5
}

// ===== 矩阵乘法 =====

// 20260319 ZJH 测试方形矩阵乘法：[[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
TEST(TensorOpsTest, Matmul2D) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::Tensor::fromData(std::vector<float>{5, 6, 7, 8}.data(), {2, 2});
    auto c = df::tensorMatmul(a, b);  // 20260319 ZJH 执行 2x2 矩阵乘法
    EXPECT_EQ(c.shape(0), 2);  // 20260319 ZJH 输出形状验证 [2, 2]
    EXPECT_EQ(c.shape(1), 2);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 19.0f);  // 20260319 ZJH 1*5+2*7=19
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);  // 20260319 ZJH 1*6+2*8=22
    EXPECT_FLOAT_EQ(c.at({1, 0}), 43.0f);  // 20260319 ZJH 3*5+4*7=43
    EXPECT_FLOAT_EQ(c.at({1, 1}), 50.0f);  // 20260319 ZJH 3*6+4*8=50
}

// 20260319 ZJH 测试非方形矩阵乘法：[2,3] @ [3,2] -> [2,2]
// A = [[1,2,3],[4,5,6]]，B = [[7,8],[9,10],[11,12]]
TEST(TensorOpsTest, MatmulNonSquare) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::Tensor::fromData(std::vector<float>{7, 8, 9, 10, 11, 12}.data(), {3, 2});
    auto c = df::tensorMatmul(a, b);  // 20260319 ZJH 执行 [2,3]@[3,2] 矩阵乘法
    EXPECT_EQ(c.shape(0), 2);  // 20260319 ZJH 输出行数 = A 的行数
    EXPECT_EQ(c.shape(1), 2);  // 20260319 ZJH 输出列数 = B 的列数
    EXPECT_FLOAT_EQ(c.at({0, 0}), 58.0f);   // 20260319 ZJH 1*7+2*9+3*11=58
    EXPECT_FLOAT_EQ(c.at({0, 1}), 64.0f);   // 20260319 ZJH 1*8+2*10+3*12=64
    EXPECT_FLOAT_EQ(c.at({1, 0}), 139.0f);  // 20260319 ZJH 4*7+5*9+6*11=139
    EXPECT_FLOAT_EQ(c.at({1, 1}), 154.0f);  // 20260319 ZJH 4*8+5*10+6*12=154
}

// ===== 形状变换 =====

// 20260319 ZJH 测试 reshape：[2,3] -> [3,2]，元素顺序不变，数据共享
TEST(TensorOpsTest, Reshape) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::tensorReshape(a, {3, 2});  // 20260319 ZJH 2x3 reshape 为 3x2
    EXPECT_EQ(b.shape(0), 3);   // 20260319 ZJH 验证新形状第 0 维
    EXPECT_EQ(b.shape(1), 2);   // 20260319 ZJH 验证新形状第 1 维
    EXPECT_EQ(b.numel(), 6);    // 20260319 ZJH 元素总数不变
    EXPECT_FLOAT_EQ(b.at({0, 0}), 1.0f);  // 20260319 ZJH 第一个元素仍为 1
    EXPECT_FLOAT_EQ(b.at({2, 1}), 6.0f);  // 20260319 ZJH 最后一个元素仍为 6
}

// 20260319 ZJH 测试 flatten：[2,2] -> [4]，验证一维化操作
TEST(TensorOpsTest, ReshapeFlatten) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4}.data(), {2, 2});
    auto b = df::tensorReshape(a, {4});  // 20260319 ZJH 2x2 flatten 为 1D 向量
    EXPECT_EQ(b.ndim(), 1);     // 20260319 ZJH 维度数应变为 1
    EXPECT_EQ(b.shape(0), 4);   // 20260319 ZJH 唯一维度大小为 4
    EXPECT_FLOAT_EQ(b.at({0}), 1.0f);  // 20260319 ZJH 第一个元素为 1
    EXPECT_FLOAT_EQ(b.at({3}), 4.0f);  // 20260319 ZJH 最后一个元素为 4
}

// 20260319 ZJH 测试 transpose：[2,3] 转置为 [3,2]
// 原始: [[1,2,3],[4,5,6]] 转置后: [[1,4],[2,5],[3,6]]
TEST(TensorOpsTest, Transpose) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::tensorTranspose(a, 0, 1);  // 20260319 ZJH 交换维度 0 和 1
    EXPECT_EQ(b.shape(0), 3);  // 20260319 ZJH 转置后第 0 维 = 原第 1 维 = 3
    EXPECT_EQ(b.shape(1), 2);  // 20260319 ZJH 转置后第 1 维 = 原第 0 维 = 2
    EXPECT_FLOAT_EQ(b.at({0, 0}), 1.0f);  // 20260319 ZJH b[0,0]=a[0,0]=1
    EXPECT_FLOAT_EQ(b.at({0, 1}), 4.0f);  // 20260319 ZJH b[0,1]=a[1,0]=4
    EXPECT_FLOAT_EQ(b.at({1, 0}), 2.0f);  // 20260319 ZJH b[1,0]=a[0,1]=2
    EXPECT_FLOAT_EQ(b.at({2, 1}), 6.0f);  // 20260319 ZJH b[2,1]=a[1,2]=6
    EXPECT_FALSE(b.isContiguous());  // 20260319 ZJH 转置后视图不连续（步长不满足行主序）
}

// 20260319 ZJH 测试 transpose 后连续化：步长恢复为行主序
TEST(TensorOpsTest, TransposeThenContiguous) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto b = df::tensorTranspose(a, 0, 1);  // 20260319 ZJH 先转置
    auto c = b.contiguous();               // 20260319 ZJH 再连续化（执行 stridedCopy）
    EXPECT_TRUE(c.isContiguous());         // 20260319 ZJH 连续化后必须连续
    EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);  // 20260319 ZJH 连续化后元素值不变
    EXPECT_FLOAT_EQ(c.at({0, 1}), 4.0f);  // 20260319 ZJH b[0,1]=a[1,0]=4
}

// 20260319 ZJH 测试 slice（dim=0）：从 3x3 矩阵中截取行 [1, 3)，得到 2x3 子矩阵
TEST(TensorOpsTest, Slice) {
    auto t = df::Tensor::fromData(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {3, 3});
    auto s = df::tensorSlice(t, 0, 1, 3);  // 20260319 ZJH 沿第 0 维截取 [1,3)，即第 1、2 行
    EXPECT_EQ(s.shape(0), 2);  // 20260319 ZJH 截取 2 行（3-1=2）
    EXPECT_EQ(s.shape(1), 3);  // 20260319 ZJH 列数不变
    EXPECT_FLOAT_EQ(s.at({0, 0}), 4.0f);  // 20260319 ZJH 原矩阵第 1 行第 0 列 = 4
    EXPECT_FLOAT_EQ(s.at({1, 2}), 9.0f);  // 20260319 ZJH 原矩阵第 2 行第 2 列 = 9
}

// 20260319 ZJH 测试 slice（dim=1）：从 2x3 矩阵中截取列 [0, 2)，得到 2x2 子矩阵
TEST(TensorOpsTest, SliceDim1) {
    auto t = df::Tensor::fromData(
        std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3});
    auto s = df::tensorSlice(t, 1, 0, 2);  // 20260319 ZJH 沿第 1 维截取 [0,2)，即前 2 列
    EXPECT_EQ(s.shape(0), 2);  // 20260319 ZJH 行数不变
    EXPECT_EQ(s.shape(1), 2);  // 20260319 ZJH 截取 2 列（2-0=2）
    EXPECT_FLOAT_EQ(s.at({0, 0}), 1.0f);  // 20260319 ZJH 第 0 行第 0 列 = 1
    EXPECT_FLOAT_EQ(s.at({0, 1}), 2.0f);  // 20260319 ZJH 第 0 行第 1 列 = 2
    EXPECT_FLOAT_EQ(s.at({1, 0}), 4.0f);  // 20260319 ZJH 第 1 行第 0 列 = 4
    EXPECT_FLOAT_EQ(s.at({1, 1}), 5.0f);  // 20260319 ZJH 第 1 行第 1 列 = 5
}

// ===== 标量运算 =====

// 20260319 ZJH 测试加标量和乘标量操作
TEST(TensorOpsTest, ScalarOps) {
    auto a = df::Tensor::fromData(std::vector<float>{1, 2, 3}.data(), {3});

    // 20260319 ZJH 加标量：[1,2,3] + 10 = [11,12,13]
    auto b = df::tensorAddScalar(a, 10.0f);
    EXPECT_FLOAT_EQ(b.at({0}), 11.0f);  // 20260319 ZJH 1+10=11
    EXPECT_FLOAT_EQ(b.at({2}), 13.0f);  // 20260319 ZJH 3+10=13

    // 20260319 ZJH 乘标量：[1,2,3] * 3 = [3,6,9]
    auto c = df::tensorMulScalar(a, 3.0f);
    EXPECT_FLOAT_EQ(c.at({0}), 3.0f);  // 20260319 ZJH 1*3=3
    EXPECT_FLOAT_EQ(c.at({2}), 9.0f);  // 20260319 ZJH 3*3=9
}

// ===== 归约运算 =====

// 20260319 ZJH 测试全局 sum/max/min 归约：[1,2,3,4,5]
TEST(TensorOpsTest, Reductions) {
    auto t = df::Tensor::fromData(std::vector<float>{1, 2, 3, 4, 5}.data(), {5});
    EXPECT_FLOAT_EQ(df::tensorSum(t).item(), 15.0f);  // 20260319 ZJH 1+2+3+4+5=15，tensorSum 返回标量张量
    EXPECT_FLOAT_EQ(df::tensorMax(t), 5.0f);   // 20260319 ZJH 最大值为 5
    EXPECT_FLOAT_EQ(df::tensorMin(t), 1.0f);   // 20260319 ZJH 最小值为 1
}
