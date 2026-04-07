// 20260319 ZJH ResNet 单元测试 — Phase 2 Part 3
// 覆盖：BasicBlock 同尺寸/下采样、ResNet18 前向/参数量/保存加载
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
import om.engine.resnet;
import om.engine.serializer;

// ===== 1. BasicBlockSameSize =====
// 20260319 ZJH 测试 BasicBlock(64,64) 在 [1,64,8,8] 上前向传播，输出形状不变
TEST(ResNetTest, BasicBlockSameSize) {
    // 20260319 ZJH 创建 BasicBlock(64, 64, stride=1)
    om::BasicBlock block(64, 64, 1);
    // 20260319 ZJH 创建 [1, 64, 8, 8] 随机输入
    auto input = om::Tensor::randn({1, 64, 8, 8});
    auto output = block.forward(input);
    // 20260319 ZJH 验证输出形状: 步幅 1、通道不变，空间维度不变
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);   // 20260319 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 64);  // 20260319 ZJH 输出通道数
    EXPECT_EQ(output.shape(2), 8);   // 20260319 ZJH 输出高度
    EXPECT_EQ(output.shape(3), 8);   // 20260319 ZJH 输出宽度
    // 20260319 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 2. BasicBlockDownsample =====
// 20260319 ZJH 测试 BasicBlock(64,128,stride=2) 在 [1,64,8,8] 上前向传播，空间下采样
TEST(ResNetTest, BasicBlockDownsample) {
    // 20260319 ZJH 创建 BasicBlock(64, 128, stride=2)
    om::BasicBlock block(64, 128, 2);
    // 20260319 ZJH 创建 [1, 64, 8, 8] 随机输入
    auto input = om::Tensor::randn({1, 64, 8, 8});
    auto output = block.forward(input);
    // 20260319 ZJH 验证输出形状: 通道 64->128，空间 8->4（stride=2）
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);    // 20260319 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 128);  // 20260319 ZJH 输出通道数
    EXPECT_EQ(output.shape(2), 4);    // 20260319 ZJH 输出高度: (8+2*1-3)/2+1 = 4
    EXPECT_EQ(output.shape(3), 4);    // 20260319 ZJH 输出宽度: 同上
    // 20260319 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 3. ResNet18Forward =====
// 20260319 ZJH 测试 ResNet18(10) 在 [1,1,28,28] 上的完整前向传播
TEST(ResNetTest, ResNet18Forward) {
    // 20260319 ZJH 创建 ResNet18，10 个类别
    om::ResNet18 model(10);
    // 20260319 ZJH 创建 [1, 1, 28, 28] 随机输入（模拟 MNIST 灰度图）
    auto input = om::Tensor::randn({1, 1, 28, 28});
    auto output = model.forward(input);
    // 20260319 ZJH 验证输出形状: [1, 10]
    ASSERT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.shape(0), 1);   // 20260319 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 10);  // 20260319 ZJH 类别数
    // 20260319 ZJH 验证输出不包含 NaN 或 Inf
    for (int i = 0; i < output.numel(); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
        EXPECT_FALSE(std::isinf(output.floatDataPtr()[i]));
    }
}

// ===== 4. ResNet18Parameters =====
// 20260319 ZJH 测试 ResNet18 参数量在合理范围内
TEST(ResNetTest, ResNet18Parameters) {
    // 20260319 ZJH 创建 ResNet18
    om::ResNet18 model(10);
    auto vecParams = model.parameters();
    // 20260319 ZJH 计算总参数量
    int nTotalParams = 0;
    for (auto* pParam : vecParams) {
        nTotalParams += pParam->numel();  // 20260319 ZJH 累加各参数元素数
    }
    // 20260319 ZJH 标准 ResNet18 约 11M 参数（ImageNet 版本）
    // 我们的版本：输入通道 1（非 3），无初始 maxpool，参数量约 11M
    // 验证参数量 > 10 万且 < 2000 万（合理范围）
    EXPECT_GT(nTotalParams, 100000);     // 20260319 ZJH 至少 10 万参数
    EXPECT_LT(nTotalParams, 20000000);   // 20260319 ZJH 不超过 2000 万参数
    // 20260319 ZJH 验证有足够多的参数张量
    EXPECT_GT(vecParams.size(), static_cast<size_t>(20));  // 20260319 ZJH 至少 20 个参数张量
    // 20260319 ZJH 输出参数信息用于调试
    std::printf("ResNet18 parameters: %d tensors, %d total values\n",
                static_cast<int>(vecParams.size()), nTotalParams);
}

// ===== 5. ResNet18SaveLoad =====
// 20260319 ZJH 测试 ResNet18 的保存和加载，验证参数完全匹配
TEST(ResNetTest, ResNet18SaveLoad) {
    // 20260319 ZJH 创建原始 ResNet18
    om::ResNet18 model1(10);

    // 20260319 ZJH 保存模型
    std::string strPath = "test_resnet18_serialize.omm";
    om::ModelSerializer::save(model1, strPath);

    // 20260319 ZJH 创建相同结构的新 ResNet18
    om::ResNet18 model2(10);

    // 20260319 ZJH 加载参数
    om::ModelSerializer::load(model2, strPath);

    // 20260319 ZJH 验证参数匹配
    auto vecParams1 = model1.parameters();
    auto vecParams2 = model2.parameters();
    ASSERT_EQ(vecParams1.size(), vecParams2.size());  // 20260319 ZJH 参数数量一致

    for (size_t i = 0; i < vecParams1.size(); ++i) {
        ASSERT_EQ(vecParams1[i]->numel(), vecParams2[i]->numel());  // 20260319 ZJH 各参数元素数一致
        auto c1 = vecParams1[i]->contiguous();
        auto c2 = vecParams2[i]->contiguous();
        for (int j = 0; j < c1.numel(); ++j) {
            EXPECT_FLOAT_EQ(c1.floatDataPtr()[j], c2.floatDataPtr()[j]);  // 20260319 ZJH 参数值完全匹配
        }
    }

    // 20260319 ZJH 清理临时文件
    std::remove(strPath.c_str());
}
