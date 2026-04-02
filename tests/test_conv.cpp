// 20260319 ZJH Conv/Pool/BN/Flatten/Dropout/Serializer 单元测试 — Phase 2 Part 2
// 覆盖：Conv2d/BatchNorm2d/MaxPool2d/AvgPool2d/Flatten/Dropout/Softmax 前向/反向、序列化、简单 CNN 组合
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>
#include <filesystem>
#include <cstdio>

// 20260319 ZJH 导入所有需要的模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.linear;
import om.engine.activations;
import om.engine.conv;
import om.engine.serializer;

// ===== 1. Conv2dForward =====
// 20260319 ZJH 测试 Conv2d(1,4,3,stride=1,pad=1) 在 [1,1,8,8] 上的前向传播
TEST(ConvTest, Conv2dForward) {
    // 20260319 ZJH 创建 Conv2d(1, 4, 3, stride=1, padding=1) 层
    om::Conv2d layer(1, 4, 3, 1, 1);
    // 20260319 ZJH 创建 [1, 1, 8, 8] 全 1 输入
    auto input = om::Tensor::ones({1, 1, 8, 8});
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状: pad=1 保持空间维度，所以 [1, 4, 8, 8]
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);  // 20260319 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 4);  // 20260319 ZJH 输出通道数
    EXPECT_EQ(output.shape(2), 8);  // 20260319 ZJH 输出高度
    EXPECT_EQ(output.shape(3), 8);  // 20260319 ZJH 输出宽度
}

// ===== 2. Conv2dNoPad =====
// 20260319 ZJH 测试 Conv2d(1,4,3,stride=1,pad=0) 在 [1,1,8,8] 上
TEST(ConvTest, Conv2dNoPad) {
    om::Conv2d layer(1, 4, 3, 1, 0);
    auto input = om::Tensor::ones({1, 1, 8, 8});
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状: (8 - 3) / 1 + 1 = 6, 所以 [1, 4, 6, 6]
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 4);
    EXPECT_EQ(output.shape(2), 6);
    EXPECT_EQ(output.shape(3), 6);
}

// ===== 3. BatchNorm2dForward =====
// 20260319 ZJH 测试 BatchNorm2d(4) 在 [2,4,8,8] 上
TEST(ConvTest, BatchNorm2dForward) {
    om::BatchNorm2d layer(4);
    // 20260319 ZJH 创建随机输入
    auto input = om::Tensor::randn({2, 4, 8, 8});
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状不变
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 2);
    EXPECT_EQ(output.shape(1), 4);
    EXPECT_EQ(output.shape(2), 8);
    EXPECT_EQ(output.shape(3), 8);
    // 20260319 ZJH 验证归一化后的值在合理范围内（不是 NaN 或极端值）
    for (int i = 0; i < std::min(output.numel(), 10); ++i) {
        float fVal = output.floatDataPtr()[i];
        EXPECT_FALSE(std::isnan(fVal));
        EXPECT_FALSE(std::isinf(fVal));
    }
}

// ===== 4. MaxPool2dForward =====
// 20260319 ZJH 测试 MaxPool2d(2,2) 在 [1,1,4,4] 上
TEST(ConvTest, MaxPool2dForward) {
    om::MaxPool2d layer(2, 2);
    // 20260319 ZJH 创建 [1,1,4,4] 输入并手动设置值
    auto input = om::Tensor::zeros({1, 1, 4, 4});
    // 20260319 ZJH 填充 0-15 的值
    float* pData = input.mutableFloatDataPtr();
    for (int i = 0; i < 16; ++i) pData[i] = static_cast<float>(i);
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状: (4 - 2) / 2 + 1 = 2, 所以 [1, 1, 2, 2]
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 1);
    EXPECT_EQ(output.shape(2), 2);
    EXPECT_EQ(output.shape(3), 2);
    // 20260319 ZJH 验证最大值正确
    // 输入矩阵:
    // 0  1  2  3
    // 4  5  6  7
    // 8  9  10 11
    // 12 13 14 15
    // 2x2 池化: max(0,1,4,5)=5, max(2,3,6,7)=7, max(8,9,12,13)=13, max(10,11,14,15)=15
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 7.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 13.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 15.0f);
}

// ===== 5. AvgPool2dForward =====
// 20260319 ZJH 测试 AvgPool2d(2,2) 在 [1,1,4,4] 上
TEST(ConvTest, AvgPool2dForward) {
    om::AvgPool2d layer(2, 2);
    auto input = om::Tensor::zeros({1, 1, 4, 4});
    float* pData = input.mutableFloatDataPtr();
    for (int i = 0; i < 16; ++i) pData[i] = static_cast<float>(i);
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状: [1, 1, 2, 2]
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(2), 2);
    EXPECT_EQ(output.shape(3), 2);
    // 20260319 ZJH avg(0,1,4,5)=2.5, avg(2,3,6,7)=4.5
    EXPECT_NEAR(output.at({0, 0, 0, 0}), 2.5f, 1e-5f);
    EXPECT_NEAR(output.at({0, 0, 0, 1}), 4.5f, 1e-5f);
}

// ===== 6. FlattenForward =====
// 20260319 ZJH 测试 Flatten 将 [2,3,4,4] 展平为 [2,48]
TEST(ConvTest, FlattenForward) {
    om::Flatten layer;
    auto input = om::Tensor::ones({2, 3, 4, 4});
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状
    ASSERT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.shape(0), 2);   // 20260319 ZJH 批次大小保持
    EXPECT_EQ(output.shape(1), 48);  // 20260319 ZJH 3*4*4 = 48
}

// ===== 7. DropoutForward =====
// 20260319 ZJH 测试 Dropout 形状不变，评估模式直接透传
TEST(ConvTest, DropoutForward) {
    om::Dropout layer(0.5f);
    auto input = om::Tensor::ones({4, 10});

    // 20260319 ZJH 训练模式
    layer.train(true);
    auto output = layer.forward(input);
    ASSERT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.shape(0), 4);
    EXPECT_EQ(output.shape(1), 10);

    // 20260319 ZJH 评估模式：输出应与输入相同
    layer.eval();
    auto outputEval = layer.forward(input);
    for (int i = 0; i < outputEval.numel(); ++i) {
        EXPECT_FLOAT_EQ(outputEval.floatDataPtr()[i], 1.0f);  // 20260319 ZJH 评估模式直接透传
    }
}

// ===== 8. Conv2dBackward =====
// 20260319 ZJH 测试 Conv2d 反向传播，验证权重梯度形状匹配
TEST(ConvTest, Conv2dBackward) {
    om::Conv2d layer(1, 2, 3, 1, 1);  // 20260319 ZJH Conv2d(1, 2, 3, stride=1, pad=1)
    auto input = om::Tensor::ones({1, 1, 4, 4});
    auto output = layer.forward(input);
    // 20260319 ZJH 计算损失: sum(output)
    auto loss = om::tensorSum(output);
    layer.zeroGrad();
    om::tensorBackward(loss);

    // 20260319 ZJH 验证参数梯度存在
    auto vecParams = layer.parameters();
    ASSERT_GE(vecParams.size(), static_cast<size_t>(1));

    // 20260319 ZJH 验证权重梯度形状: [2, 1, 3, 3]
    auto gradWeight = om::tensorGetGrad(*vecParams[0]);
    ASSERT_EQ(gradWeight.numel(), 2 * 1 * 3 * 3);  // 20260319 ZJH 18 个元素
    EXPECT_EQ(gradWeight.shape(0), 2);   // 20260319 ZJH Cout
    EXPECT_EQ(gradWeight.shape(1), 1);   // 20260319 ZJH Cin
    EXPECT_EQ(gradWeight.shape(2), 3);   // 20260319 ZJH KH
    EXPECT_EQ(gradWeight.shape(3), 3);   // 20260319 ZJH KW

    // 20260319 ZJH 验证偏置梯度
    if (vecParams.size() >= 2) {
        auto gradBias = om::tensorGetGrad(*vecParams[1]);
        ASSERT_EQ(gradBias.numel(), 2);  // 20260319 ZJH 偏置梯度 [2]
    }
}

// ===== 9. SerializeSaveLoad =====
// 20260319 ZJH 测试模型序列化：保存后加载到新模型，验证参数匹配
TEST(ConvTest, SerializeSaveLoad) {
    // 20260319 ZJH 构建一个简单模型
    auto pModel1 = std::make_shared<om::Sequential>();
    pModel1->add(std::make_shared<om::Linear>(4, 3));
    pModel1->add(std::make_shared<om::Linear>(3, 2));

    // 20260319 ZJH 保存模型
    std::string strPath = "test_model_serialize.omm";
    om::ModelSerializer::save(*pModel1, strPath);

    // 20260319 ZJH 创建相同结构的新模型
    auto pModel2 = std::make_shared<om::Sequential>();
    pModel2->add(std::make_shared<om::Linear>(4, 3));
    pModel2->add(std::make_shared<om::Linear>(3, 2));

    // 20260319 ZJH 加载参数
    om::ModelSerializer::load(*pModel2, strPath);

    // 20260319 ZJH 验证参数匹配
    auto vecParams1 = pModel1->parameters();
    auto vecParams2 = pModel2->parameters();
    ASSERT_EQ(vecParams1.size(), vecParams2.size());

    for (size_t i = 0; i < vecParams1.size(); ++i) {
        ASSERT_EQ(vecParams1[i]->numel(), vecParams2[i]->numel());
        auto c1 = vecParams1[i]->contiguous();
        auto c2 = vecParams2[i]->contiguous();
        for (int j = 0; j < c1.numel(); ++j) {
            EXPECT_FLOAT_EQ(c1.floatDataPtr()[j], c2.floatDataPtr()[j]);
        }
    }

    // 20260319 ZJH 清理临时文件
    std::remove(strPath.c_str());
}

// ===== 11. GroupNorm2dForward =====
// 20260402 ZJH GroupNorm2d 前向测试：验证输出形状、非 NaN、以及每组内均值≈0 方差≈1
TEST(ConvTest, GroupNorm2dForward) {
    // 20260402 ZJH 创建 GroupNorm2d(32 channels, 8 groups)
    om::GroupNorm2d gn(32, 8);
    // 20260402 ZJH 输入 [1, 32, 4, 4] 随机张量
    auto input = om::Tensor::randn({1, 32, 4, 4});
    auto output = gn.forward(input);

    // 20260402 ZJH 验证输出形状不变: [1, 32, 4, 4]
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);   // 20260402 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 32);  // 20260402 ZJH 通道数
    EXPECT_EQ(output.shape(2), 4);   // 20260402 ZJH 高度
    EXPECT_EQ(output.shape(3), 4);   // 20260402 ZJH 宽度

    // 20260402 ZJH 获取输出数据指针，验证无 NaN、非全零
    auto pOut = output.contiguous().floatDataPtr();
    bool bNonZero = false;  // 20260402 ZJH 标记是否存在非零元素
    for (int i = 0; i < output.numel(); ++i) {
        EXPECT_FALSE(std::isnan(pOut[i]));  // 20260402 ZJH 无 NaN
        if (std::abs(pOut[i]) > 1e-6f) bNonZero = true;
    }
    EXPECT_TRUE(bNonZero);  // 20260402 ZJH 输出不应全为零

    // 20260402 ZJH 验证每组内归一化统计量（gamma=1, beta=0 时均值≈0, 方差≈1）
    int nChannelsPerGroup = 32 / 8;  // 20260402 ZJH 每组 4 个通道
    int nHW = 4 * 4;                 // 20260402 ZJH 每通道空间元素数
    int nCountPerGroup = nChannelsPerGroup * nHW;  // 20260402 ZJH 每组元素总数 = 4*16 = 64
    for (int g = 0; g < 8; ++g) {
        float fSum = 0.0f;    // 20260402 ZJH 累积该组元素之和
        float fSumSq = 0.0f;  // 20260402 ZJH 累积该组元素平方之和
        int nStart = g * nCountPerGroup;  // 20260402 ZJH 该组在展平后数组中的起始偏移
        for (int i = 0; i < nCountPerGroup; ++i) {
            fSum   += pOut[nStart + i];
            fSumSq += pOut[nStart + i] * pOut[nStart + i];
        }
        float fMean = fSum / static_cast<float>(nCountPerGroup);  // 20260402 ZJH 组内均值
        // 20260402 ZJH 方差 = E[x^2] - (E[x])^2
        float fVar = fSumSq / static_cast<float>(nCountPerGroup) - fMean * fMean;
        // 20260402 ZJH 允许浮点计算误差：均值≈0（容差 0.15），方差≈1（容差 0.3）
        EXPECT_NEAR(fMean, 0.0f, 0.15f) << "Group " << g << " mean out of range";
        EXPECT_NEAR(fVar,  1.0f, 0.3f)  << "Group " << g << " var out of range";
    }
}

// ===== 12. GroupNormAutoGroups =====
// 20260402 ZJH GroupNorm2d 自动分组调整测试：当 groups 不能整除 channels 时自动降级
TEST(ConvTest, GroupNormAutoGroups) {
    // 20260402 ZJH channels=17（质数），请求 groups=32，无法整除 → 自动降级
    om::GroupNorm2d gn1(17, 32);
    // 20260402 ZJH 实际 groups 必须能整除 17
    EXPECT_EQ(17 % gn1.groups(), 0) << "groups=" << gn1.groups() << " must divide channels=17";
    // 20260402 ZJH 17 是质数，最终只能是 1 或 17
    EXPECT_TRUE(gn1.groups() == 1 || gn1.groups() == 17)
        << "Expected groups=1 or 17, got " << gn1.groups();

    // 20260402 ZJH channels=64, groups=32 → 可以整除，保持 32
    om::GroupNorm2d gn2(64, 32);
    EXPECT_EQ(gn2.groups(), 32);  // 20260402 ZJH 64 % 32 == 0，不应降级

    // 20260402 ZJH channels=3, groups=32 → 无法整除，自动降级
    om::GroupNorm2d gn3(3, 32);
    EXPECT_EQ(3 % gn3.groups(), 0) << "groups=" << gn3.groups() << " must divide channels=3";
    // 20260402 ZJH 3 的因子只有 1 和 3
    EXPECT_TRUE(gn3.groups() == 1 || gn3.groups() == 3)
        << "Expected groups=1 or 3, got " << gn3.groups();
}

// ===== 13. GroupNormParameters =====
// 20260402 ZJH GroupNorm2d 参数收集测试：验证 gamma/beta 数量及 buffers 为空
TEST(ConvTest, GroupNormParameters) {
    // 20260402 ZJH 创建 GroupNorm2d(64 channels, 32 groups)
    om::GroupNorm2d gn(64, 32);

    // 20260402 ZJH 参数应有 2 个：gamma [64] 和 beta [64]
    auto vecParams = gn.parameters();
    ASSERT_EQ(vecParams.size(), static_cast<size_t>(2));  // 20260402 ZJH gamma + beta
    EXPECT_EQ(vecParams[0]->numel(), 64);  // 20260402 ZJH gamma 形状 [64]
    EXPECT_EQ(vecParams[1]->numel(), 64);  // 20260402 ZJH beta 形状 [64]

    // 20260402 ZJH GroupNorm 无 running stats，buffers 应为空
    auto vecBufs = gn.buffers();
    EXPECT_TRUE(vecBufs.empty());  // 20260402 ZJH 无 running_mean / running_var
}

// ===== 10. SimpleCNN =====
// 20260319 ZJH 测试简单 CNN: Sequential(Conv2d, ReLU, MaxPool2d, Flatten, Linear) 前向传播形状正确
TEST(ConvTest, SimpleCNN) {
    // 20260319 ZJH 构建简单 CNN 模型
    auto pModel = std::make_shared<om::Sequential>();
    // 20260319 ZJH Conv2d(1, 4, 3, stride=1, pad=1): [N,1,8,8] -> [N,4,8,8]
    pModel->add(std::make_shared<om::Conv2d>(1, 4, 3, 1, 1));
    // 20260319 ZJH ReLU
    pModel->add(std::make_shared<om::ReLU>());
    // 20260319 ZJH MaxPool2d(2, 2): [N,4,8,8] -> [N,4,4,4]
    pModel->add(std::make_shared<om::MaxPool2d>(2, 2));
    // 20260319 ZJH Flatten: [N,4,4,4] -> [N,64]
    pModel->add(std::make_shared<om::Flatten>());
    // 20260319 ZJH Linear(64, 10): [N,64] -> [N,10]
    pModel->add(std::make_shared<om::Linear>(64, 10));

    // 20260319 ZJH 创建输入 [2, 1, 8, 8]
    auto input = om::Tensor::randn({2, 1, 8, 8});
    auto output = pModel->forward(input);

    // 20260319 ZJH 验证输出形状 [2, 10]
    ASSERT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.shape(0), 2);   // 20260319 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 10);  // 20260319 ZJH 输出类别数

    // 20260319 ZJH 验证输出不是 NaN
    for (int i = 0; i < output.numel(); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== GroupNorm2d 测试 =====

// 20260402 ZJH 测试 GroupNorm2d 前向传播: 32 通道 8 组
TEST(ConvTest, GroupNorm2dForward) {
    // 20260402 ZJH 创建 GroupNorm2d(32 channels, 8 groups)
    om::GroupNorm2d gn(32, 8);
    // 20260402 ZJH 创建 [1, 32, 4, 4] 随机输入
    auto input = om::Tensor::randn({1, 32, 4, 4});
    auto output = gn.forward(input);

    // 20260402 ZJH 验证输出形状
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);   // 20260402 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 32);  // 20260402 ZJH 通道数不变
    EXPECT_EQ(output.shape(2), 4);   // 20260402 ZJH 空间维度不变
    EXPECT_EQ(output.shape(3), 4);

    // 20260402 ZJH 验证输出非全零、无 NaN
    auto tCpu = output.cpu().contiguous();
    const float* pOut = tCpu.floatDataPtr();
    bool bNonZero = false;
    for (int i = 0; i < tCpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(pOut[i]));  // 20260402 ZJH 无 NaN
        if (std::abs(pOut[i]) > 1e-6f) bNonZero = true;
    }
    EXPECT_TRUE(bNonZero);  // 20260402 ZJH 输出不全为零

    // 20260402 ZJH 验证每组内均值≈0, 方差≈1（gamma=1, beta=0 初始化时）
    int nChannelsPerGroup = 32 / 8;  // 20260402 ZJH 每组 4 通道
    int nHW = 4 * 4;                 // 20260402 ZJH 空间维度 16
    for (int g = 0; g < 8; ++g) {
        float fSum = 0.0f, fSumSq = 0.0f;
        int nStart = g * nChannelsPerGroup * nHW;  // 20260402 ZJH 该组起始偏移
        int nCount = nChannelsPerGroup * nHW;       // 20260402 ZJH 该组元素总数
        for (int i = 0; i < nCount; ++i) {
            fSum += pOut[nStart + i];
            fSumSq += pOut[nStart + i] * pOut[nStart + i];
        }
        float fMean = fSum / static_cast<float>(nCount);
        float fVar = fSumSq / static_cast<float>(nCount) - fMean * fMean;
        EXPECT_NEAR(fMean, 0.0f, 0.15f);   // 20260402 ZJH 均值接近 0
        EXPECT_NEAR(fVar, 1.0f, 0.3f);     // 20260402 ZJH 方差接近 1
    }
}

// 20260402 ZJH 测试 GroupNorm2d 自动分组调整
TEST(ConvTest, GroupNormAutoGroups) {
    // 20260402 ZJH channels=17（质数），groups=32 无法整除 → 自动降
    om::GroupNorm2d gn(17, 32);
    EXPECT_EQ(17 % gn.groups(), 0);  // 20260402 ZJH 自动调整后能整除

    // 20260402 ZJH channels=64, groups=32 → 正常不变
    om::GroupNorm2d gn2(64, 32);
    EXPECT_EQ(gn2.groups(), 32);

    // 20260402 ZJH channels=3（RGB），groups=32 → 降到 3 或 1
    om::GroupNorm2d gn3(3, 32);
    EXPECT_EQ(3 % gn3.groups(), 0);  // 20260402 ZJH 能整除即可
}

// 20260402 ZJH 测试 GroupNorm2d 参数和 buffer 收集
TEST(ConvTest, GroupNormParameters) {
    om::GroupNorm2d gn(64, 32);

    // 20260402 ZJH 验证参数: gamma [64] + beta [64] = 2 个参数
    auto params = gn.parameters();
    EXPECT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0]->numel(), 64);  // 20260402 ZJH gamma
    EXPECT_EQ(params[1]->numel(), 64);  // 20260402 ZJH beta

    // 20260402 ZJH GroupNorm 没有 running stats → buffers 为空
    auto bufs = gn.buffers();
    EXPECT_TRUE(bufs.empty());
}
