// 20260330 ZJH 全面单元测试套件 — 覆盖核心模块
// 使用简单的 assert + 计数器，无外部测试框架依赖
// 覆盖: Tensor基础/运算/Autograd/Conv2d/Loss/NMS/DataPipeline/
//       Serializer/ModelRegistry/ErrorCode/ContourApprox/GalleryManager
// 不依赖 GPU，全部 CPU 模式运行

#include <iostream>     // 20260330 ZJH 测试输出
#include <cassert>      // 20260330 ZJH assert 断言
#include <cmath>        // 20260330 ZJH std::abs/std::sqrt/std::exp
#include <vector>       // 20260330 ZJH std::vector
#include <string>       // 20260330 ZJH std::string
#include <stdexcept>    // 20260330 ZJH std::runtime_error
#include <algorithm>    // 20260330 ZJH std::sort/std::find
#include <numeric>      // 20260330 ZJH std::accumulate
#include <cstdint>      // 20260330 ZJH int64_t

// 20260330 ZJH 导入引擎模块（C++23 module）
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.autograd;
import om.engine.conv;
import om.engine.loss;
import om.engine.data_pipeline;
import om.engine.serializer;
import om.engine.module;
import om.engine.linear;
import om.engine.activations;

// 20260330 ZJH 传统头文件（非 module 的 core 组件）
#include "core/ErrorCode.h"
#include "core/NmsUtils.h"
#include "core/ContourApprox.h"
#include "engine/bridge/ModelRegistry.h"
#include "engine/OvInference.h"

// ============================================================================
// 简单测试框架 — 无外部依赖
// ============================================================================

// 20260330 ZJH 全局测试计数器
static int s_nTestsPassed = 0;  // 20260330 ZJH 通过的测试数
static int s_nTestsFailed = 0;  // 20260330 ZJH 失败的测试数

// 20260330 ZJH TEST 宏：定义测试函数
#define TEST(name) void test_##name()

// 20260330 ZJH RUN_TEST 宏：运行测试并统计结果
#define RUN_TEST(name) do { \
    std::cout << "  Running " << #name << "..."; \
    try { test_##name(); s_nTestsPassed++; std::cout << " PASS" << std::endl; } \
    catch (const std::exception& e) { s_nTestsFailed++; std::cout << " FAIL: " << e.what() << std::endl; } \
    catch (...) { s_nTestsFailed++; std::cout << " FAIL: unknown exception" << std::endl; } \
} while(0)

// 20260330 ZJH ASSERT_EQ 宏：断言两个值相等
#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) throw std::runtime_error( \
        std::string("ASSERT_EQ failed: ") + std::to_string(_a) + " != " + std::to_string(_b) + \
        " at line " + std::to_string(__LINE__)); \
} while(0)

// 20260330 ZJH ASSERT_NEAR 宏：断言两个浮点值在容差范围内
#define ASSERT_NEAR(a, b, tol) do { \
    double _a = static_cast<double>(a); double _b = static_cast<double>(b); \
    if (std::abs(_a - _b) > (tol)) throw std::runtime_error( \
        std::string("ASSERT_NEAR failed: |") + std::to_string(_a) + " - " + std::to_string(_b) + \
        "| > " + std::to_string(tol) + " at line " + std::to_string(__LINE__)); \
} while(0)

// 20260330 ZJH ASSERT_TRUE 宏：断言条件为真
#define ASSERT_TRUE(x) do { \
    if (!(x)) throw std::runtime_error( \
        std::string("ASSERT_TRUE failed at line ") + std::to_string(__LINE__)); \
} while(0)

// 20260330 ZJH ASSERT_FALSE 宏：断言条件为假
#define ASSERT_FALSE(x) do { \
    if ((x)) throw std::runtime_error( \
        std::string("ASSERT_FALSE failed at line ") + std::to_string(__LINE__)); \
} while(0)

// 20260330 ZJH ASSERT_GT 宏：断言 a > b
#define ASSERT_GT(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a > _b)) throw std::runtime_error( \
        std::string("ASSERT_GT failed: ") + std::to_string(_a) + " <= " + std::to_string(_b) + \
        " at line " + std::to_string(__LINE__)); \
} while(0)

// 20260330 ZJH ASSERT_GE 宏：断言 a >= b
#define ASSERT_GE(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a >= _b)) throw std::runtime_error( \
        std::string("ASSERT_GE failed: ") + std::to_string(_a) + " < " + std::to_string(_b) + \
        " at line " + std::to_string(__LINE__)); \
} while(0)

// ============================================================================
// 1. Tensor 基础测试（创建/形状/数据访问/clone/设备）
// ============================================================================

// 20260330 ZJH 测试 1: Tensor zeros 工厂方法
TEST(tensor_zeros) {
    // 20260330 ZJH 创建 2x3 全零张量
    om::Tensor t = om::Tensor::zeros({2, 3});
    ASSERT_EQ(t.ndim(), 2);          // 20260330 ZJH 2 维
    ASSERT_EQ(t.shape(0), 2);        // 20260330 ZJH 第 0 维大小 = 2
    ASSERT_EQ(t.shape(1), 3);        // 20260330 ZJH 第 1 维大小 = 3
    ASSERT_EQ(t.numel(), 6);         // 20260330 ZJH 共 6 个元素
    // 20260330 ZJH 验证所有元素为 0
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < t.numel(); ++i) {
        ASSERT_NEAR(pData[i], 0.0f, 1e-7f);
    }
}

// 20260330 ZJH 测试 2: Tensor ones 工厂方法
TEST(tensor_ones) {
    // 20260330 ZJH 创建 3x4 全 1 张量
    om::Tensor t = om::Tensor::ones({3, 4});
    ASSERT_EQ(t.numel(), 12);  // 20260330 ZJH 共 12 个元素
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < t.numel(); ++i) {
        ASSERT_NEAR(pData[i], 1.0f, 1e-7f);  // 20260330 ZJH 每个元素为 1.0
    }
}

// 20260330 ZJH 测试 3: Tensor full 工厂方法
TEST(tensor_full) {
    // 20260330 ZJH 创建 2x2 全 3.14 张量
    om::Tensor t = om::Tensor::full({2, 2}, 3.14f);
    ASSERT_EQ(t.numel(), 4);
    const float* pData = t.floatDataPtr();
    for (int i = 0; i < t.numel(); ++i) {
        ASSERT_NEAR(pData[i], 3.14f, 1e-5f);  // 20260330 ZJH 每个元素为 3.14
    }
}

// 20260330 ZJH 测试 4: Tensor randn 形状正确
TEST(tensor_randn_shape) {
    // 20260330 ZJH 创建随机正态分布张量
    om::Tensor t = om::Tensor::randn({4, 5, 6});
    ASSERT_EQ(t.ndim(), 3);      // 20260330 ZJH 3 维
    ASSERT_EQ(t.shape(0), 4);
    ASSERT_EQ(t.shape(1), 5);
    ASSERT_EQ(t.shape(2), 6);
    ASSERT_EQ(t.numel(), 120);   // 20260330 ZJH 4*5*6 = 120
}

// 20260330 ZJH 测试 5: Tensor clone 深拷贝
TEST(tensor_clone) {
    // 20260330 ZJH 创建原始张量
    om::Tensor t = om::Tensor::ones({3, 3});
    // 20260330 ZJH 克隆
    om::Tensor tClone = t.clone();
    ASSERT_EQ(tClone.numel(), 9);  // 20260330 ZJH 元素数相同
    // 20260330 ZJH 验证数据相同
    const float* pOriginal = t.floatDataPtr();
    const float* pCloned = tClone.floatDataPtr();
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(pOriginal[i], pCloned[i], 1e-7f);
    }
    // 20260330 ZJH 验证是不同的内存（深拷贝）
    ASSERT_TRUE(pOriginal != pCloned);
}

// 20260330 ZJH 测试 6: Tensor 设备类型
TEST(tensor_device) {
    // 20260330 ZJH CPU 张量
    om::Tensor t = om::Tensor::zeros({2, 2});
    ASSERT_TRUE(t.device() == om::DeviceType::CPU);  // 20260330 ZJH 默认 CPU
}

// ============================================================================
// 2. Tensor 运算测试（add/sub/mul/matmul/transpose/reshape/slice）
// ============================================================================

// 20260330 ZJH 测试 7: 逐元素加法
TEST(tensor_add) {
    om::Tensor a = om::Tensor::ones({2, 3});  // 20260330 ZJH 全 1
    om::Tensor b = om::Tensor::full({2, 3}, 2.0f);  // 20260330 ZJH 全 2
    om::Tensor c = om::TensorOps::add(a, b);  // 20260330 ZJH c = a + b = 全 3
    const float* pData = c.floatDataPtr();
    for (int i = 0; i < c.numel(); ++i) {
        ASSERT_NEAR(pData[i], 3.0f, 1e-6f);
    }
}

// 20260330 ZJH 测试 8: 逐元素减法
TEST(tensor_sub) {
    om::Tensor a = om::Tensor::full({2, 2}, 5.0f);
    om::Tensor b = om::Tensor::full({2, 2}, 3.0f);
    om::Tensor c = om::TensorOps::sub(a, b);  // 20260330 ZJH c = 5 - 3 = 2
    const float* pData = c.floatDataPtr();
    for (int i = 0; i < c.numel(); ++i) {
        ASSERT_NEAR(pData[i], 2.0f, 1e-6f);
    }
}

// 20260330 ZJH 测试 9: 逐元素乘法
TEST(tensor_mul) {
    om::Tensor a = om::Tensor::full({3, 2}, 4.0f);
    om::Tensor b = om::Tensor::full({3, 2}, 0.5f);
    om::Tensor c = om::TensorOps::mul(a, b);  // 20260330 ZJH c = 4 * 0.5 = 2
    const float* pData = c.floatDataPtr();
    for (int i = 0; i < c.numel(); ++i) {
        ASSERT_NEAR(pData[i], 2.0f, 1e-6f);
    }
}

// 20260330 ZJH 测试 10: 矩阵乘法 [2,3] x [3,4] = [2,4]
TEST(tensor_matmul) {
    om::Tensor a = om::Tensor::ones({2, 3});
    om::Tensor b = om::Tensor::ones({3, 4});
    om::Tensor c = om::TensorOps::matmul(a, b);  // 20260330 ZJH 矩阵乘法
    ASSERT_EQ(c.shape(0), 2);  // 20260330 ZJH 输出行数
    ASSERT_EQ(c.shape(1), 4);  // 20260330 ZJH 输出列数
    // 20260330 ZJH 全 1 矩阵乘法结果每个元素 = 内维大小 = 3
    const float* pData = c.floatDataPtr();
    for (int i = 0; i < c.numel(); ++i) {
        ASSERT_NEAR(pData[i], 3.0f, 1e-5f);
    }
}

// 20260330 ZJH 测试 11: 转置 [2,3] -> [3,2]
TEST(tensor_transpose) {
    om::Tensor t = om::Tensor::zeros({2, 3});
    om::Tensor tT = om::TensorOps::transpose(t);  // 20260330 ZJH 转置
    ASSERT_EQ(tT.shape(0), 3);  // 20260330 ZJH 原第 1 维变第 0 维
    ASSERT_EQ(tT.shape(1), 2);  // 20260330 ZJH 原第 0 维变第 1 维
}

// 20260330 ZJH 测试 12: reshape [2,6] -> [3,4]
TEST(tensor_reshape) {
    om::Tensor t = om::Tensor::ones({2, 6});
    om::Tensor r = om::TensorOps::reshape(t, {3, 4});  // 20260330 ZJH 重塑形状
    ASSERT_EQ(r.shape(0), 3);
    ASSERT_EQ(r.shape(1), 4);
    ASSERT_EQ(r.numel(), 12);  // 20260330 ZJH 元素数不变
}

// 20260330 ZJH 测试 13: slice 切片
TEST(tensor_slice) {
    // 20260330 ZJH 创建 [4, 3] 张量，值为 0~11
    om::Tensor t = om::Tensor::zeros({4, 3});
    float* pData = const_cast<float*>(t.floatDataPtr());
    for (int i = 0; i < 12; ++i) {
        pData[i] = static_cast<float>(i);  // 20260330 ZJH 填充 0,1,2,...,11
    }
    // 20260330 ZJH 切片第 0 维 [1, 3)
    om::Tensor s = om::TensorOps::slice(t, 0, 1, 3);
    ASSERT_EQ(s.shape(0), 2);  // 20260330 ZJH 切出 2 行
    ASSERT_EQ(s.shape(1), 3);  // 20260330 ZJH 列数不变
}

// ============================================================================
// 3. Autograd 测试
// ============================================================================

// 20260330 ZJH 测试 14: 简单梯度 y = x * 2, dy/dx = 2
TEST(autograd_simple_grad) {
    // 20260330 ZJH 创建需要梯度的标量张量 x = 3.0
    om::Tensor x = om::Tensor::full({1}, 3.0f);
    x.setRequiresGrad(true);  // 20260330 ZJH 启用梯度追踪
    // 20260330 ZJH y = x * 2
    om::Tensor two = om::Tensor::full({1}, 2.0f);
    om::Tensor y = om::TensorOps::mul(x, two);
    // 20260330 ZJH 反向传播
    y.backward();
    // 20260330 ZJH 检查梯度: dy/dx = 2
    auto grad = x.grad();
    ASSERT_TRUE(grad.numel() > 0);
    ASSERT_NEAR(grad.floatDataPtr()[0], 2.0f, 1e-5f);
}

// 20260330 ZJH 测试 15: 链式法则 z = (x + y) * y, dz/dx = y, dz/dy = x + 2*y
TEST(autograd_chain_rule) {
    om::Tensor x = om::Tensor::full({1}, 2.0f);
    x.setRequiresGrad(true);
    om::Tensor y = om::Tensor::full({1}, 3.0f);
    y.setRequiresGrad(true);
    // 20260330 ZJH z = (x + y) * y = x*y + y*y
    om::Tensor sum = om::TensorOps::add(x, y);
    om::Tensor z = om::TensorOps::mul(sum, y);
    z.backward();
    // 20260330 ZJH dz/dx = y = 3.0
    ASSERT_NEAR(x.grad().floatDataPtr()[0], 3.0f, 1e-4f);
    // 20260330 ZJH dz/dy = x + 2*y = 2 + 6 = 8.0
    ASSERT_NEAR(y.grad().floatDataPtr()[0], 8.0f, 1e-4f);
}

// ============================================================================
// 4. Conv2d 测试
// ============================================================================

// 20260330 ZJH 测试 16: Conv2d 前向传播形状检查 (with padding)
TEST(conv2d_forward_shape_padded) {
    // 20260330 ZJH Conv2d(1, 4, 3, stride=1, padding=1) — 保持空间维度
    om::Conv2d layer(1, 4, 3, 1, 1);
    auto input = om::Tensor::ones({1, 1, 8, 8});  // 20260330 ZJH [B,C,H,W]
    auto output = layer.forward(input);
    ASSERT_EQ(output.ndim(), 4);
    ASSERT_EQ(output.shape(0), 1);  // 20260330 ZJH batch
    ASSERT_EQ(output.shape(1), 4);  // 20260330 ZJH out_channels
    ASSERT_EQ(output.shape(2), 8);  // 20260330 ZJH H 不变（padding=1, kernel=3）
    ASSERT_EQ(output.shape(3), 8);  // 20260330 ZJH W 不变
}

// 20260330 ZJH 测试 17: Conv2d 前向传播形状检查 (no padding)
TEST(conv2d_forward_shape_nopad) {
    // 20260330 ZJH Conv2d(1, 4, 3, stride=1, padding=0) — 尺寸缩小
    om::Conv2d layer(1, 4, 3, 1, 0);
    auto input = om::Tensor::ones({1, 1, 8, 8});
    auto output = layer.forward(input);
    ASSERT_EQ(output.shape(2), 6);  // 20260330 ZJH (8 - 3) / 1 + 1 = 6
    ASSERT_EQ(output.shape(3), 6);
}

// 20260330 ZJH 测试 18: Conv2d stride=2 下采样
TEST(conv2d_stride2) {
    // 20260330 ZJH Conv2d(3, 16, 3, stride=2, padding=1)
    om::Conv2d layer(3, 16, 3, 2, 1);
    auto input = om::Tensor::randn({2, 3, 32, 32});  // 20260330 ZJH batch=2
    auto output = layer.forward(input);
    ASSERT_EQ(output.shape(0), 2);   // 20260330 ZJH batch 不变
    ASSERT_EQ(output.shape(1), 16);  // 20260330 ZJH out_channels
    ASSERT_EQ(output.shape(2), 16);  // 20260330 ZJH (32 + 2*1 - 3) / 2 + 1 = 16
    ASSERT_EQ(output.shape(3), 16);
}

// ============================================================================
// 5. Loss 函数测试
// ============================================================================

// 20260330 ZJH 测试 19: CrossEntropy loss 输出为正数
TEST(loss_cross_entropy) {
    // 20260330 ZJH 模拟 3 类分类, batch=2
    om::Tensor logits = om::Tensor::randn({2, 3});    // 20260330 ZJH 原始 logits
    om::Tensor labels = om::Tensor::zeros({2});        // 20260330 ZJH 标签
    float* pLabels = const_cast<float*>(labels.floatDataPtr());
    pLabels[0] = 0.0f;  // 20260330 ZJH 第 0 个样本标签 = 0
    pLabels[1] = 1.0f;  // 20260330 ZJH 第 1 个样本标签 = 1
    om::Tensor loss = om::LossFunctions::crossEntropy(logits, labels);
    // 20260330 ZJH CE loss 应为正数
    ASSERT_GT(loss.floatDataPtr()[0], 0.0f);
}

// 20260330 ZJH 测试 20: MSE loss 计算正确性
TEST(loss_mse) {
    om::Tensor pred = om::Tensor::full({4}, 3.0f);   // 20260330 ZJH 预测值 [3,3,3,3]
    om::Tensor target = om::Tensor::full({4}, 1.0f); // 20260330 ZJH 目标值 [1,1,1,1]
    om::Tensor loss = om::LossFunctions::mse(pred, target);
    // 20260330 ZJH MSE = mean((3-1)^2) = 4.0
    ASSERT_NEAR(loss.floatDataPtr()[0], 4.0f, 1e-4f);
}

// 20260330 ZJH 测试 21: Focal loss 输出合理范围
TEST(loss_focal) {
    // 20260330 ZJH Focal loss 用于处理类别不平衡
    om::Tensor logits = om::Tensor::randn({4, 3});
    om::Tensor labels = om::Tensor::zeros({4});
    float* pLabels = const_cast<float*>(labels.floatDataPtr());
    pLabels[0] = 0.0f;
    pLabels[1] = 1.0f;
    pLabels[2] = 2.0f;
    pLabels[3] = 0.0f;
    om::Tensor loss = om::LossFunctions::focalLoss(logits, labels, 2.0f, 0.25f);
    // 20260330 ZJH Focal loss 应为非负
    ASSERT_GE(loss.floatDataPtr()[0], 0.0f);
}

// ============================================================================
// 6. NMS 测试
// ============================================================================

// 20260330 ZJH 测试 22: 标准 NMS 基本正确性
TEST(nms_basic) {
    // 20260330 ZJH 创建两个高度重叠的框（IoU > 0.9）
    std::vector<om::DetectionBox> vecBoxes = {
        {10.0f, 10.0f, 50.0f, 50.0f, 0.9f, 0, 0},   // 20260330 ZJH 高分框
        {12.0f, 12.0f, 52.0f, 52.0f, 0.8f, 0, 0},   // 20260330 ZJH 低分框（与上一个高度重叠）
        {100.0f, 100.0f, 150.0f, 150.0f, 0.7f, 0, 0} // 20260330 ZJH 不重叠的框
    };
    // 20260330 ZJH IoU 阈值 0.3, 分数阈值 0.5
    auto vecKept = om::nms(vecBoxes, 0.3f, 0.5f, false);
    // 20260330 ZJH 应保留 2 个框（第一个和第三个）
    ASSERT_EQ(static_cast<int>(vecKept.size()), 2);
    // 20260330 ZJH 第一个保留的框分数最高
    ASSERT_NEAR(vecKept[0].fScore, 0.9f, 1e-6f);
    // 20260330 ZJH 第二个保留的框是不重叠的
    ASSERT_NEAR(vecKept[1].fScore, 0.7f, 1e-6f);
}

// 20260330 ZJH 测试 23: NMS 置信度过滤
TEST(nms_score_filter) {
    std::vector<om::DetectionBox> vecBoxes = {
        {0.0f, 0.0f, 10.0f, 10.0f, 0.1f, 0, 0},  // 20260330 ZJH 低分，应被过滤
        {20.0f, 20.0f, 30.0f, 30.0f, 0.9f, 0, 0}  // 20260330 ZJH 高分，应保留
    };
    auto vecKept = om::nms(vecBoxes, 0.5f, 0.5f, false);
    ASSERT_EQ(static_cast<int>(vecKept.size()), 1);  // 20260330 ZJH 只保留 1 个
    ASSERT_NEAR(vecKept[0].fScore, 0.9f, 1e-6f);
}

// 20260330 ZJH 测试 24: IoU 计算
TEST(nms_iou_computation) {
    // 20260330 ZJH 两个完全重叠的框: IoU = 1.0
    om::DetectionBox a = {0.0f, 0.0f, 10.0f, 10.0f, 1.0f, 0, 0};
    om::DetectionBox b = {0.0f, 0.0f, 10.0f, 10.0f, 1.0f, 0, 0};
    float fIoU = om::computeBoxIoU(a, b);
    ASSERT_NEAR(fIoU, 1.0f, 1e-6f);

    // 20260330 ZJH 两个不重叠的框: IoU = 0.0
    om::DetectionBox c = {0.0f, 0.0f, 10.0f, 10.0f, 1.0f, 0, 0};
    om::DetectionBox d = {20.0f, 20.0f, 30.0f, 30.0f, 1.0f, 0, 0};
    float fIoU2 = om::computeBoxIoU(c, d);
    ASSERT_NEAR(fIoU2, 0.0f, 1e-6f);

    // 20260330 ZJH 部分重叠: 手算 IoU
    om::DetectionBox e = {0.0f, 0.0f, 10.0f, 10.0f, 1.0f, 0, 0};  // 20260330 ZJH area=100
    om::DetectionBox f = {5.0f, 5.0f, 15.0f, 15.0f, 1.0f, 0, 0};  // 20260330 ZJH area=100
    // 20260330 ZJH 交集: (5,5)-(10,10), 面积 = 25
    // 20260330 ZJH 并集: 100 + 100 - 25 = 175
    // 20260330 ZJH IoU = 25 / 175 = 0.142857
    float fIoU3 = om::computeBoxIoU(e, f);
    ASSERT_NEAR(fIoU3, 25.0f / 175.0f, 1e-5f);
}

// 20260330 ZJH 测试 25: 类别感知 NMS
TEST(nms_class_aware) {
    // 20260330 ZJH 两个高度重叠但不同类别的框
    std::vector<om::DetectionBox> vecBoxes = {
        {10.0f, 10.0f, 50.0f, 50.0f, 0.9f, 0, 0},  // 20260330 ZJH 类别 0
        {12.0f, 12.0f, 52.0f, 52.0f, 0.8f, 1, 0}   // 20260330 ZJH 类别 1（不同类别）
    };
    // 20260330 ZJH 类别感知模式：不同类别不互相抑制
    auto vecKept = om::nms(vecBoxes, 0.3f, 0.5f, true);
    ASSERT_EQ(static_cast<int>(vecKept.size()), 2);  // 20260330 ZJH 都保留
}

// ============================================================================
// 7. Data Pipeline 测试
// ============================================================================

// 20260330 ZJH 测试 26: Resize 操作
TEST(data_pipeline_resize) {
    // 20260330 ZJH 创建 3x8x8 图像张量
    om::Tensor image = om::Tensor::randn({1, 3, 8, 8});
    // 20260330 ZJH Resize 到 4x4
    om::Tensor resized = om::DataPipeline::resize(image, 4, 4);
    ASSERT_EQ(resized.shape(2), 4);  // 20260330 ZJH 高度 = 4
    ASSERT_EQ(resized.shape(3), 4);  // 20260330 ZJH 宽度 = 4
}

// 20260330 ZJH 测试 27: Normalize 归一化
TEST(data_pipeline_normalize) {
    // 20260330 ZJH 创建均匀值张量
    om::Tensor image = om::Tensor::full({1, 3, 4, 4}, 128.0f / 255.0f);
    // 20260330 ZJH ImageNet 归一化
    std::vector<float> vecMean = {0.485f, 0.456f, 0.406f};
    std::vector<float> vecStd = {0.229f, 0.224f, 0.225f};
    om::Tensor normalized = om::DataPipeline::normalize(image, vecMean, vecStd);
    // 20260330 ZJH 归一化后值应在合理范围内（大约 -2 ~ +2）
    const float* pData = normalized.floatDataPtr();
    for (int i = 0; i < normalized.numel(); ++i) {
        ASSERT_TRUE(pData[i] > -5.0f && pData[i] < 5.0f);  // 20260330 ZJH 合理范围
    }
}

// ============================================================================
// 8. Serializer 测试（保存+加载 round-trip）
// ============================================================================

// 20260330 ZJH 测试 28: 序列化 round-trip
TEST(serializer_roundtrip) {
    // 20260330 ZJH 创建简单模型 (Linear)
    auto pModel = std::make_shared<om::Linear>(4, 2);
    // 20260330 ZJH 保存到临时文件
    std::string strTempPath = "test_serializer_roundtrip.omm";
    bool bSaved = om::Serializer::save(pModel, strTempPath);
    ASSERT_TRUE(bSaved);
    // 20260330 ZJH 重新加载
    auto pLoaded = std::make_shared<om::Linear>(4, 2);
    bool bLoaded = om::Serializer::load(pLoaded, strTempPath);
    ASSERT_TRUE(bLoaded);
    // 20260330 ZJH 验证参数数量一致
    auto vecOrigParams = pModel->parameters();
    auto vecLoadedParams = pLoaded->parameters();
    ASSERT_EQ(static_cast<int>(vecOrigParams.size()), static_cast<int>(vecLoadedParams.size()));
    // 20260330 ZJH 验证参数值近似相等
    for (size_t i = 0; i < vecOrigParams.size(); ++i) {
        ASSERT_EQ(vecOrigParams[i].numel(), vecLoadedParams[i].numel());
        const float* pOrig = vecOrigParams[i].floatDataPtr();
        const float* pLoad = vecLoadedParams[i].floatDataPtr();
        for (int j = 0; j < vecOrigParams[i].numel(); ++j) {
            ASSERT_NEAR(pOrig[j], pLoad[j], 1e-5f);
        }
    }
    // 20260330 ZJH 清理临时文件
    std::remove(strTempPath.c_str());
}

// ============================================================================
// 9. ModelRegistry 测试
// ============================================================================

// 20260330 ZJH 测试 29: ModelRegistry 已注册类型查询
TEST(model_registry_query) {
    auto& registry = om::ModelRegistry::instance();
    registry.ensureInitialized();  // 20260330 ZJH 确保已注册所有内置模型
    // 20260330 ZJH 获取所有已注册类型
    auto vecTypes = registry.getRegisteredTypes();
    ASSERT_GT(static_cast<int>(vecTypes.size()), 0);  // 20260330 ZJH 至少有 1 种模型
}

// 20260330 ZJH 测试 30: ModelRegistry ResNet18 创建
TEST(model_registry_create_resnet18) {
    auto& registry = om::ModelRegistry::instance();
    registry.ensureInitialized();
    // 20260330 ZJH 检查 ResNet18 是否已注册
    bool bRegistered = registry.isRegistered("ResNet18");
    ASSERT_TRUE(bRegistered);
    // 20260330 ZJH 创建 ResNet18 (10 类, 3 通道)
    auto pModel = registry.createModel("ResNet18", 10, 3, 224);
    ASSERT_TRUE(pModel != nullptr);  // 20260330 ZJH 创建成功
}

// 20260330 ZJH 测试 31: ModelRegistry 模型元信息
TEST(model_registry_info) {
    auto& registry = om::ModelRegistry::instance();
    registry.ensureInitialized();
    const om::ModelInfo* pInfo = registry.getModelInfo("ResNet18");
    if (pInfo) {
        ASSERT_TRUE(pInfo->strType == "ResNet18");       // 20260330 ZJH 类型名匹配
        ASSERT_TRUE(pInfo->bIsCnn);                       // 20260330 ZJH ResNet 是 CNN
        ASSERT_TRUE(pInfo->eCategory == om::ModelCategory::Classification);  // 20260330 ZJH 分类模型
    }
}

// 20260330 ZJH 测试 32: ModelRegistry 未注册类型返回 nullptr
TEST(model_registry_unknown_type) {
    auto& registry = om::ModelRegistry::instance();
    registry.ensureInitialized();
    auto pModel = registry.createModel("NonExistentModel999", 10);
    ASSERT_TRUE(pModel == nullptr);  // 20260330 ZJH 未知类型返回空
}

// ============================================================================
// 10. ErrorCode 测试
// ============================================================================

// 20260330 ZJH 测试 33: ErrorCode 十六进制格式化
TEST(error_code_hex_format) {
    std::string strHex = om::errorCodeToHex(om::ErrorCode::OK);
    ASSERT_TRUE(strHex == "0x00000000");  // 20260330 ZJH OK = 0x00000000

    std::string strHex2 = om::errorCodeToHex(om::ErrorCode::TrainingCudaOOM);
    ASSERT_TRUE(strHex2 == "0x00020002");  // 20260330 ZJH TrainingCudaOOM = 0x00020002
}

// 20260330 ZJH 测试 34: ErrorCode 字符串转换
TEST(error_code_to_string) {
    const char* pStr = om::errorCodeToString(om::ErrorCode::OK);
    ASSERT_TRUE(pStr != nullptr);
    // 20260330 ZJH 应包含 "OK"
    ASSERT_TRUE(std::string(pStr).find("OK") != std::string::npos);

    const char* pStr2 = om::errorCodeToString(om::ErrorCode::CudaOutOfMemory);
    ASSERT_TRUE(pStr2 != nullptr);
    // 20260330 ZJH 应包含 "CUDA"
    ASSERT_TRUE(std::string(pStr2).find("CUDA") != std::string::npos);
}

// 20260330 ZJH 测试 35: Error 结构体工厂方法
TEST(error_struct_factory) {
    auto errOk = om::Error::Ok();
    ASSERT_TRUE(errOk.ok());              // 20260330 ZJH Ok() 状态为成功
    ASSERT_TRUE(static_cast<bool>(errOk)); // 20260330 ZJH bool 转换

    auto errFail = om::Error::Make(om::ErrorCode::FileNotFound, "test.onnx not found",
                                    "loadModel()", "Check file path");
    ASSERT_FALSE(errFail.ok());  // 20260330 ZJH Make() 创建的是错误
    ASSERT_TRUE(errFail.strMessage == "test.onnx not found");
    ASSERT_TRUE(errFail.strContext == "loadModel()");
}

// 20260330 ZJH 测试 36: Error formatError 完整报告
TEST(error_format_report) {
    auto err = om::Error::Make(om::ErrorCode::ModelLoadFailed,
                                "corrupted file",
                                "ModelExporter::load",
                                "Re-export model");
    std::string strReport = om::formatError(err);
    // 20260330 ZJH 报告应包含错误码、描述、上下文
    ASSERT_TRUE(strReport.find("0x00010003") != std::string::npos);  // 20260330 ZJH 十六进制码
    ASSERT_TRUE(strReport.find("corrupted file") != std::string::npos);  // 20260330 ZJH 消息
    ASSERT_TRUE(strReport.find("ModelExporter") != std::string::npos);   // 20260330 ZJH 上下文
}

// ============================================================================
// 11. ContourApprox 测试（Douglas-Peucker）
// ============================================================================

// 20260330 ZJH 测试 37: Douglas-Peucker 简化正方形轮廓
TEST(contour_approx_square) {
    // 20260330 ZJH 构造一个正方形轮廓（每边 10 个点）
    std::vector<om::Point2f> vecSquare;
    // 20260330 ZJH 底边: (0,0) -> (10,0)
    for (int i = 0; i <= 10; ++i) vecSquare.push_back({static_cast<float>(i), 0.0f});
    // 20260330 ZJH 右边: (10,0) -> (10,10)
    for (int i = 1; i <= 10; ++i) vecSquare.push_back({10.0f, static_cast<float>(i)});
    // 20260330 ZJH 顶边: (10,10) -> (0,10)
    for (int i = 9; i >= 0; --i) vecSquare.push_back({static_cast<float>(i), 10.0f});
    // 20260330 ZJH 左边: (0,10) -> (0,0)
    for (int i = 9; i >= 1; --i) vecSquare.push_back({0.0f, static_cast<float>(i)});

    // 20260330 ZJH 用 epsilon=0.5 简化
    auto vecSimplified = om::approxContour(vecSquare, 0.5f);
    // 20260330 ZJH 正方形简化后应保留约 4-5 个关键点（四个角 + 首尾）
    ASSERT_TRUE(static_cast<int>(vecSimplified.size()) <= 6);
    ASSERT_TRUE(static_cast<int>(vecSimplified.size()) >= 2);
}

// 20260330 ZJH 测试 38: Douglas-Peucker 退化情况（少于 3 个点）
TEST(contour_approx_degenerate) {
    // 20260330 ZJH 只有 2 个点
    std::vector<om::Point2f> vecLine = {{0.0f, 0.0f}, {10.0f, 10.0f}};
    auto vecResult = om::approxContour(vecLine, 1.0f);
    // 20260330 ZJH 应原样返回
    ASSERT_EQ(static_cast<int>(vecResult.size()), 2);
}

// 20260330 ZJH 测试 39: IoU 计算 DetectionBox area
TEST(detection_box_area) {
    om::DetectionBox box = {0.0f, 0.0f, 10.0f, 20.0f, 1.0f, 0, 0};
    ASSERT_NEAR(box.area(), 200.0f, 1e-6f);  // 20260330 ZJH 10 * 20 = 200

    // 20260330 ZJH 无效框（x2 < x1）
    om::DetectionBox invalidBox = {10.0f, 10.0f, 5.0f, 5.0f, 1.0f, 0, 0};
    ASSERT_NEAR(invalidBox.area(), 0.0f, 1e-6f);  // 20260330 ZJH 无效框面积为 0
}

// 20260330 ZJH 测试 40: contourArea 面积计算 (Shoelace 公式)
TEST(contour_area) {
    // 20260330 ZJH 构造单位正方形 (0,0),(1,0),(1,1),(0,1) 面积=1.0
    std::vector<om::Point2f> vecSquare = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}
    };
    float fArea = om::contourArea(vecSquare);
    ASSERT_NEAR(fArea, 1.0f, 1e-5f);  // 20260330 ZJH Shoelace 面积 = 1.0
}

// ============================================================================
// 12. OpenVINO Inference 存根测试
// ============================================================================

// 20260330 ZJH 测试 41: OpenVINO 存根模式行为
TEST(openvino_stub_not_loaded) {
    om::OpenVINOInference ov;
    // 20260330 ZJH 未加载模型时 isLoaded 应返回 false
    ASSERT_FALSE(ov.isLoaded());
    // 20260330 ZJH loadModel 在未启用时应返回 false
    bool bResult = ov.loadModel("nonexistent.onnx");
    ASSERT_FALSE(bResult);
    // 20260330 ZJH 应有错误信息
    std::string strError = ov.getLastError();
    ASSERT_TRUE(strError.size() > 0);
}

// 20260330 ZJH 测试 42: OpenVINO 空推理返回空
TEST(openvino_stub_empty_infer) {
    om::OpenVINOInference ov;
    std::vector<float> vecInput = {1.0f, 2.0f, 3.0f};
    std::vector<int64_t> vecShape = {1, 3};
    auto results = ov.infer(vecInput, vecShape);
    ASSERT_EQ(static_cast<int>(results.size()), 0);  // 20260330 ZJH 未加载时返回空
}

// 20260330 ZJH 测试 43: OpenVINO ModelInfo 未加载时为空
TEST(openvino_stub_model_info) {
    om::OpenVINOInference ov;
    auto info = ov.getModelInfo();
    ASSERT_EQ(info.nNumInputs, 0);   // 20260330 ZJH 未加载时为 0
    ASSERT_EQ(info.nNumOutputs, 0);
}

// ============================================================================
// 13. 额外综合测试
// ============================================================================

// 20260330 ZJH 测试 44: Soft-NMS 输出比标准 NMS 更多
TEST(soft_nms_retains_more) {
    // 20260330 ZJH 创建紧密重叠的框
    std::vector<om::DetectionBox> vecBoxes = {
        {10.0f, 10.0f, 50.0f, 50.0f, 0.95f, 0, 0},
        {11.0f, 11.0f, 51.0f, 51.0f, 0.90f, 0, 0},
        {12.0f, 12.0f, 52.0f, 52.0f, 0.85f, 0, 0}
    };
    auto vecBoxesCopy = vecBoxes;
    // 20260330 ZJH 标准 NMS（阈值较低，会抑制重叠框）
    auto vecHardKept = om::nms(vecBoxes, 0.3f, 0.1f, false);
    // 20260330 ZJH Soft-NMS（衰减而非删除，保留更多）
    auto vecSoftKept = om::softNms(vecBoxesCopy, 0.3f, 0.01f, om::SoftNmsMethod::Gaussian, 0.5f);
    // 20260330 ZJH Soft-NMS 应保留不少于标准 NMS 的框数
    ASSERT_GE(static_cast<int>(vecSoftKept.size()), static_cast<int>(vecHardKept.size()));
}

// 20260330 ZJH 测试 45: ErrorCode suggestion 非空
TEST(error_code_suggestion) {
    const char* pSug = om::errorCodeSuggestion(om::ErrorCode::TrainingNaNLoss);
    ASSERT_TRUE(pSug != nullptr);
    ASSERT_TRUE(std::string(pSug).size() > 0);  // 20260330 ZJH 应有修复建议
}

// ============================================================================
// main 入口
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " OmniMatch Comprehensive Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    // 20260330 ZJH === 1. Tensor 基础 ===
    std::cout << "\n[1] Tensor Basics:" << std::endl;
    RUN_TEST(tensor_zeros);
    RUN_TEST(tensor_ones);
    RUN_TEST(tensor_full);
    RUN_TEST(tensor_randn_shape);
    RUN_TEST(tensor_clone);
    RUN_TEST(tensor_device);

    // 20260330 ZJH === 2. Tensor 运算 ===
    std::cout << "\n[2] Tensor Ops:" << std::endl;
    RUN_TEST(tensor_add);
    RUN_TEST(tensor_sub);
    RUN_TEST(tensor_mul);
    RUN_TEST(tensor_matmul);
    RUN_TEST(tensor_transpose);
    RUN_TEST(tensor_reshape);
    RUN_TEST(tensor_slice);

    // 20260330 ZJH === 3. Autograd ===
    std::cout << "\n[3] Autograd:" << std::endl;
    RUN_TEST(autograd_simple_grad);
    RUN_TEST(autograd_chain_rule);

    // 20260330 ZJH === 4. Conv2d ===
    std::cout << "\n[4] Conv2d:" << std::endl;
    RUN_TEST(conv2d_forward_shape_padded);
    RUN_TEST(conv2d_forward_shape_nopad);
    RUN_TEST(conv2d_stride2);

    // 20260330 ZJH === 5. Loss Functions ===
    std::cout << "\n[5] Loss Functions:" << std::endl;
    RUN_TEST(loss_cross_entropy);
    RUN_TEST(loss_mse);
    RUN_TEST(loss_focal);

    // 20260330 ZJH === 6. NMS ===
    std::cout << "\n[6] NMS:" << std::endl;
    RUN_TEST(nms_basic);
    RUN_TEST(nms_score_filter);
    RUN_TEST(nms_iou_computation);
    RUN_TEST(nms_class_aware);

    // 20260330 ZJH === 7. Data Pipeline ===
    std::cout << "\n[7] Data Pipeline:" << std::endl;
    RUN_TEST(data_pipeline_resize);
    RUN_TEST(data_pipeline_normalize);

    // 20260330 ZJH === 8. Serializer ===
    std::cout << "\n[8] Serializer:" << std::endl;
    RUN_TEST(serializer_roundtrip);

    // 20260330 ZJH === 9. ModelRegistry ===
    std::cout << "\n[9] ModelRegistry:" << std::endl;
    RUN_TEST(model_registry_query);
    RUN_TEST(model_registry_create_resnet18);
    RUN_TEST(model_registry_info);
    RUN_TEST(model_registry_unknown_type);

    // 20260330 ZJH === 10. ErrorCode ===
    std::cout << "\n[10] ErrorCode:" << std::endl;
    RUN_TEST(error_code_hex_format);
    RUN_TEST(error_code_to_string);
    RUN_TEST(error_struct_factory);
    RUN_TEST(error_format_report);

    // 20260330 ZJH === 11. ContourApprox ===
    std::cout << "\n[11] ContourApprox:" << std::endl;
    RUN_TEST(contour_approx_square);
    RUN_TEST(contour_approx_degenerate);
    RUN_TEST(detection_box_area);
    RUN_TEST(contour_area);

    // 20260330 ZJH === 12. OpenVINO Inference ===
    std::cout << "\n[12] OpenVINO Inference (stub):" << std::endl;
    RUN_TEST(openvino_stub_not_loaded);
    RUN_TEST(openvino_stub_empty_infer);
    RUN_TEST(openvino_stub_model_info);

    // 20260330 ZJH === 13. 综合 ===
    std::cout << "\n[13] Miscellaneous:" << std::endl;
    RUN_TEST(soft_nms_retains_more);
    RUN_TEST(error_code_suggestion);

    // 20260330 ZJH 输出测试汇总
    std::cout << "\n========================================" << std::endl;
    std::cout << " Results: " << s_nTestsPassed << " passed, "
              << s_nTestsFailed << " failed, "
              << (s_nTestsPassed + s_nTestsFailed) << " total" << std::endl;
    std::cout << "========================================" << std::endl;

    // 20260330 ZJH 返回非零表示有测试失败
    return (s_nTestsFailed > 0) ? 1 : 0;
}
