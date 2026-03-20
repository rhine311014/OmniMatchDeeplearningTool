// 20260320 ZJH 模型单元测试 — Phase 3
// 覆盖: U-Net / YOLOv5Nano / ConvAutoEncoder / ConvTranspose2d / Sigmoid / LeakyReLU / DiceLoss / ConcatChannels
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdio>

// 20260320 ZJH 导入所有需要的模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;
import df.engine.activations;
import df.engine.conv;
import df.engine.loss;
import df.engine.unet;
import df.engine.yolo;
import df.engine.autoencoder;
import df.hal.cpu_backend;

// ===== 1. UNetForward =====
// 20260320 ZJH 测试 UNet(1,2) 在 [1,1,64,64] 上前向传播，输出 [1,2,64,64]
TEST(ModelsTest, UNetForward) {
    // 20260320 ZJH 创建 UNet，1 通道输入，2 类输出
    df::UNet model(1, 2);
    // 20260320 ZJH 创建 [1, 1, 64, 64] 随机输入
    auto input = df::Tensor::randn({1, 1, 64, 64});
    auto output = model.forward(input);
    // 20260320 ZJH 验证输出形状: [1, 2, 64, 64]
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);   // 20260320 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 2);   // 20260320 ZJH 类别数
    EXPECT_EQ(output.shape(2), 64);  // 20260320 ZJH 高度保持不变
    EXPECT_EQ(output.shape(3), 64);  // 20260320 ZJH 宽度保持不变
    // 20260320 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 2. UNetParameters =====
// 20260320 ZJH 测试 UNet 参数量在合理范围内
TEST(ModelsTest, UNetParameters) {
    df::UNet model(1, 2);
    auto vecParams = model.parameters();
    int nTotalParams = 0;
    for (auto* pParam : vecParams) {
        nTotalParams += pParam->numel();
    }
    // 20260320 ZJH U-Net 参数量应在合理范围内（数百万级）
    EXPECT_GT(nTotalParams, 100000);     // 20260320 ZJH 至少 10 万参数
    EXPECT_LT(nTotalParams, 50000000);   // 20260320 ZJH 不超过 5000 万参数
    EXPECT_GT(vecParams.size(), static_cast<size_t>(30));  // 20260320 ZJH 至少 30 个参数张量
    std::printf("UNet parameters: %d tensors, %d total values\n",
                static_cast<int>(vecParams.size()), nTotalParams);
}

// ===== 3. YOLOForward =====
// 20260320 ZJH 测试 YOLOv5Nano(20) 在 [1,3,128,128] 上前向传播
TEST(ModelsTest, YOLOForward) {
    // 20260320 ZJH 创建 YOLOv5Nano，20 个类别
    df::YOLOv5Nano model(20, 3);
    // 20260320 ZJH 创建 [1, 3, 128, 128] 随机输入
    auto input = df::Tensor::randn({1, 3, 128, 128});
    auto output = model.forward(input);
    // 20260320 ZJH 验证输出形状: [1, numPredictions, 25]
    // numPredictions = (128/16)*(128/16)*3 = 8*8*3 = 192
    ASSERT_EQ(output.ndim(), 3);
    EXPECT_EQ(output.shape(0), 1);    // 20260320 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 192);  // 20260320 ZJH 8*8*3 个预测
    EXPECT_EQ(output.shape(2), 25);   // 20260320 ZJH 5 + 20 = 25
    // 20260320 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 4. AutoEncoderForward =====
// 20260320 ZJH 测试 ConvAutoEncoder(1) 在 [1,1,28,28] 上前向传播
TEST(ModelsTest, AutoEncoderForward) {
    df::ConvAutoEncoder model(1, 64);
    auto input = df::Tensor::randn({1, 1, 28, 28});
    auto output = model.forward(input);
    // 20260320 ZJH 验证输出形状: [1, 1, 28, 28]（重建与输入同形状）
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);    // 20260320 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 1);    // 20260320 ZJH 通道数
    EXPECT_EQ(output.shape(2), 28);   // 20260320 ZJH 高度
    EXPECT_EQ(output.shape(3), 28);   // 20260320 ZJH 宽度
    // 20260320 ZJH 验证输出在 [0,1] 范围内（Sigmoid 激活）
    for (int i = 0; i < output.numel(); ++i) {
        float fVal = output.floatDataPtr()[i];
        EXPECT_GE(fVal, 0.0f);  // 20260320 ZJH 输出 >= 0
        EXPECT_LE(fVal, 1.0f);  // 20260320 ZJH 输出 <= 1
    }
}

// ===== 5. AutoEncoderEncodeDecode =====
// 20260320 ZJH 测试编码和解码分别调用，验证中间形状
TEST(ModelsTest, AutoEncoderEncodeDecode) {
    df::ConvAutoEncoder model(1, 64);
    auto input = df::Tensor::randn({1, 1, 28, 28});
    // 20260320 ZJH 编码
    auto latent = model.encode(input);
    ASSERT_EQ(latent.ndim(), 4);
    EXPECT_EQ(latent.shape(0), 1);    // 20260320 ZJH 批次
    EXPECT_EQ(latent.shape(1), 64);   // 20260320 ZJH 瓶颈通道数
    EXPECT_EQ(latent.shape(2), 7);    // 20260320 ZJH 高度: 28/2/2 = 7
    EXPECT_EQ(latent.shape(3), 7);    // 20260320 ZJH 宽度: 28/2/2 = 7
    // 20260320 ZJH 解码
    auto decoded = model.decode(latent);
    ASSERT_EQ(decoded.ndim(), 4);
    EXPECT_EQ(decoded.shape(0), 1);   // 20260320 ZJH 批次
    EXPECT_EQ(decoded.shape(1), 1);   // 20260320 ZJH 通道数
    EXPECT_EQ(decoded.shape(2), 28);  // 20260320 ZJH 高度恢复
    EXPECT_EQ(decoded.shape(3), 28);  // 20260320 ZJH 宽度恢复
}

// ===== 6. ConvTranspose2dForward =====
// 20260320 ZJH 测试 ConvTranspose2d 的形状计算
TEST(ModelsTest, ConvTranspose2dForward) {
    // 20260320 ZJH ConvTranspose2d(4, 2, kernel=2, stride=2, pad=0)
    df::ConvTranspose2d deconv(4, 2, 2, 2, 0, true);
    // 20260320 ZJH 输入 [1, 4, 7, 7]
    auto input = df::Tensor::randn({1, 4, 7, 7});
    auto output = deconv.forward(input);
    // 20260320 ZJH 输出: Hout = (7-1)*2 - 2*0 + 2 = 14
    ASSERT_EQ(output.ndim(), 4);
    EXPECT_EQ(output.shape(0), 1);    // 20260320 ZJH 批次
    EXPECT_EQ(output.shape(1), 2);    // 20260320 ZJH 输出通道
    EXPECT_EQ(output.shape(2), 14);   // 20260320 ZJH 输出高度
    EXPECT_EQ(output.shape(3), 14);   // 20260320 ZJH 输出宽度
}

// ===== 7. SigmoidForward =====
// 20260320 ZJH 验证 Sigmoid 输出在 [0, 1] 范围内
TEST(ModelsTest, SigmoidForward) {
    auto input = df::Tensor::randn({2, 10});
    auto output = df::tensorSigmoid(input);
    ASSERT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.shape(0), 2);
    EXPECT_EQ(output.shape(1), 10);
    // 20260320 ZJH 验证所有输出在 [0, 1]
    for (int i = 0; i < output.numel(); ++i) {
        float fVal = output.floatDataPtr()[i];
        EXPECT_GE(fVal, 0.0f);
        EXPECT_LE(fVal, 1.0f);
    }
    // 20260320 ZJH 验证 sigmoid(0) ≈ 0.5
    auto zero = df::Tensor::zeros({1});
    auto half = df::tensorSigmoid(zero);
    EXPECT_NEAR(half.floatDataPtr()[0], 0.5f, 1e-6f);
}

// ===== 8. LeakyReLUForward =====
// 20260320 ZJH 验证 LeakyReLU 的负区域斜率
TEST(ModelsTest, LeakyReLUForward) {
    // 20260320 ZJH 创建测试数据: [-2, -1, 0, 1, 2]
    float arrData[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto input = df::Tensor::fromData(arrData, {5});
    float fSlope = 0.1f;
    auto output = df::tensorLeakyReLU(input, fSlope);
    ASSERT_EQ(output.numel(), 5);
    // 20260320 ZJH 验证输出值
    EXPECT_NEAR(output.floatDataPtr()[0], -0.2f, 1e-6f);   // -2 * 0.1
    EXPECT_NEAR(output.floatDataPtr()[1], -0.1f, 1e-6f);   // -1 * 0.1
    EXPECT_NEAR(output.floatDataPtr()[2], 0.0f, 1e-6f);    // 0
    EXPECT_NEAR(output.floatDataPtr()[3], 1.0f, 1e-6f);    // 1
    EXPECT_NEAR(output.floatDataPtr()[4], 2.0f, 1e-6f);    // 2
}

// ===== 9. DiceLossForward =====
// 20260320 ZJH 验证 Dice 损失计算
TEST(ModelsTest, DiceLossForward) {
    // 20260320 ZJH 完全匹配时 Dice 损失应接近 0
    float arrPred[] = {1.0f, 1.0f, 0.0f, 0.0f};
    float arrTarget[] = {1.0f, 1.0f, 0.0f, 0.0f};
    auto pred = df::Tensor::fromData(arrPred, {4});
    auto target = df::Tensor::fromData(arrTarget, {4});
    df::DiceLoss diceLoss;
    auto loss = diceLoss.forward(pred, target);
    // 20260320 ZJH Dice = 2*2/(2+2+eps) ≈ 1.0, loss ≈ 0.0
    EXPECT_LT(loss.item(), 0.01f);

    // 20260320 ZJH 完全不匹配时 Dice 损失应接近 1
    float arrPred2[] = {0.0f, 0.0f, 1.0f, 1.0f};
    auto pred2 = df::Tensor::fromData(arrPred2, {4});
    auto loss2 = diceLoss.forward(pred2, target);
    EXPECT_GT(loss2.item(), 0.9f);
}

// ===== 10. YOLOv8NanoForward =====
// 20260320 ZJH 测试 YOLOv8Nano(20) 在 [1,3,128,128] 上前向传播
// YOLOv8 使用解耦头（anchor-free），输出维度为 [N, H*W, 4+nClasses]
TEST(ModelsTest, YOLOv8NanoForward) {
    // 20260320 ZJH 创建 YOLOv8Nano，20 个类别
    df::YOLOv8Nano model(20, 3);
    // 20260320 ZJH 创建 [1, 3, 128, 128] 随机输入
    auto input = df::Tensor::randn({1, 3, 128, 128});
    auto output = model.forward(input);
    // 20260320 ZJH 验证输出形状: [1, (128/16)*(128/16), 4+20] = [1, 64, 24]
    ASSERT_EQ(output.ndim(), 3);
    EXPECT_EQ(output.shape(0), 1);    // 20260320 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 64);   // 20260320 ZJH 8*8 = 64 个空间位置（anchor-free）
    EXPECT_EQ(output.shape(2), 24);   // 20260320 ZJH 4 + 20 = 24（无 conf 通道）
    // 20260320 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 11. YOLOv7TinyForward =====
// 20260320 ZJH 测试 YOLOv7Tiny(20) 在 [1,3,128,128] 上前向传播
// YOLOv7-Tiny 使用 ELAN 块 + anchor-based 检测头，3 级下采样（总 /8）
TEST(ModelsTest, YOLOv7TinyForward) {
    // 20260320 ZJH 创建 YOLOv7Tiny，20 个类别
    df::YOLOv7Tiny model(20, 3);
    // 20260320 ZJH 创建 [1, 3, 128, 128] 随机输入
    auto input = df::Tensor::randn({1, 3, 128, 128});
    auto output = model.forward(input);
    // 20260320 ZJH 验证输出形状: [1, (128/8)*(128/8)*3, 5+20] = [1, 768, 25]
    ASSERT_EQ(output.ndim(), 3);
    EXPECT_EQ(output.shape(0), 1);     // 20260320 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 768);   // 20260320 ZJH 16*16*3 = 768 个预测
    EXPECT_EQ(output.shape(2), 25);    // 20260320 ZJH 5 + 20 = 25
    // 20260320 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 12. YOLOv10NanoForward =====
// 20260320 ZJH 测试 YOLOv10Nano(20) 在 [1,3,128,128] 上前向传播
// YOLOv10 使用 SCDown 解耦下采样 + C2f 块 + 解耦头，3 级下采样（总 /8）
TEST(ModelsTest, YOLOv10NanoForward) {
    // 20260320 ZJH 创建 YOLOv10Nano，20 个类别
    df::YOLOv10Nano model(20, 3);
    // 20260320 ZJH 创建 [1, 3, 128, 128] 随机输入
    auto input = df::Tensor::randn({1, 3, 128, 128});
    auto output = model.forward(input);
    // 20260320 ZJH 验证输出形状: [1, (128/8)*(128/8), 4+20] = [1, 256, 24]
    // SCDown 下采样 3 次: 128 -> 64 -> 32 -> 16, 所以 16*16 = 256
    ASSERT_EQ(output.ndim(), 3);
    EXPECT_EQ(output.shape(0), 1);     // 20260320 ZJH 批次大小
    EXPECT_EQ(output.shape(1), 256);   // 20260320 ZJH 16*16 = 256 个空间位置
    EXPECT_EQ(output.shape(2), 24);    // 20260320 ZJH 4 + 20 = 24
    // 20260320 ZJH 验证输出不包含 NaN
    for (int i = 0; i < std::min(output.numel(), 20); ++i) {
        EXPECT_FALSE(std::isnan(output.floatDataPtr()[i]));
    }
}

// ===== 13. ConcatChannels =====
// 20260320 ZJH 测试沿通道维度拼接
TEST(ModelsTest, ConcatChannels) {
    // 20260320 ZJH [1, 3, 4, 4] + [1, 5, 4, 4] -> [1, 8, 4, 4]
    auto a = df::Tensor::ones({1, 3, 4, 4});
    auto b = df::Tensor::full({1, 5, 4, 4}, 2.0f);
    auto result = df::tensorConcatChannels(a, b);
    ASSERT_EQ(result.ndim(), 4);
    EXPECT_EQ(result.shape(0), 1);  // 20260320 ZJH 批次
    EXPECT_EQ(result.shape(1), 8);  // 20260320 ZJH 3 + 5 = 8
    EXPECT_EQ(result.shape(2), 4);  // 20260320 ZJH 高度
    EXPECT_EQ(result.shape(3), 4);  // 20260320 ZJH 宽度
    // 20260320 ZJH 验证前 3 个通道值为 1.0
    EXPECT_NEAR(result.at({0, 0, 0, 0}), 1.0f, 1e-6f);
    EXPECT_NEAR(result.at({0, 2, 3, 3}), 1.0f, 1e-6f);
    // 20260320 ZJH 验证后 5 个通道值为 2.0
    EXPECT_NEAR(result.at({0, 3, 0, 0}), 2.0f, 1e-6f);
    EXPECT_NEAR(result.at({0, 7, 3, 3}), 2.0f, 1e-6f);
}
