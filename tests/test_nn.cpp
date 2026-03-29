// 20260319 ZJH nn.Module 系统单元测试 — Phase 2 Part 1
// 覆盖：Linear/Sequential/ReLU 前向、SGD/Adam 优化器、CrossEntropyLoss、参数管理、梯度清零、训练/评估模式
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>

// 20260319 ZJH 导入所有 Phase 2 新模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.linear;
import om.engine.activations;
import om.engine.optimizer;
import om.engine.loss;

// ===== 1. LinearForward =====
// 20260319 ZJH 测试 Linear 层前向传播输出形状正确性
TEST(NNTest, LinearForward) {
    // 20260319 ZJH 创建 Linear(4, 3) 全连接层
    om::Linear layer(4, 3);
    // 20260319 ZJH 创建 [2, 4] 输入张量（batch=2, features=4）
    auto input = om::Tensor::ones({2, 4});
    // 20260319 ZJH 前向传播
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状为 [2, 3]
    ASSERT_EQ(output.ndim(), 2);  // 20260319 ZJH 二维输出
    EXPECT_EQ(output.shape(0), 2);  // 20260319 ZJH 批次大小保持为 2
    EXPECT_EQ(output.shape(1), 3);  // 20260319 ZJH 输出特征维度为 3
}

// ===== 2. LinearWithBias =====
// 20260319 ZJH 测试 Linear 层偏置正确添加
TEST(NNTest, LinearWithBias) {
    // 20260319 ZJH 创建 Linear(2, 2, bias=true) 层
    om::Linear layer(2, 2, true);
    // 20260319 ZJH 创建 [1, 2] 零输入
    auto input = om::Tensor::zeros({1, 2});
    // 20260319 ZJH 前向传播：零输入 @ W + b = b（零输入乘任何权重都是零，结果只有偏置）
    auto output = layer.forward(input);
    // 20260319 ZJH 验证输出形状
    EXPECT_EQ(output.shape(0), 1);  // 20260319 ZJH 批次大小 1
    EXPECT_EQ(output.shape(1), 2);  // 20260319 ZJH 输出维度 2
    // 20260319 ZJH 偏置初始化为零，所以零输入的输出应该全为零
    EXPECT_FLOAT_EQ(output.at({0, 0}), 0.0f);  // 20260319 ZJH 结果应为 0（零偏置 + 零乘积）
    EXPECT_FLOAT_EQ(output.at({0, 1}), 0.0f);  // 20260319 ZJH 结果应为 0
}

// ===== 3. SequentialForward =====
// 20260319 ZJH 测试 Sequential 容器链式前向传播
TEST(NNTest, SequentialForward) {
    // 20260319 ZJH 构建三层网络：Linear(4,3) -> ReLU -> Linear(3,2)
    auto pModel = std::make_shared<om::Sequential>();
    pModel->add(std::make_shared<om::Linear>(4, 3));  // 20260319 ZJH 第一层：4->3
    pModel->add(std::make_shared<om::ReLU>());         // 20260319 ZJH ReLU 激活
    pModel->add(std::make_shared<om::Linear>(3, 2));  // 20260319 ZJH 第二层：3->2

    // 20260319 ZJH 创建 [5, 4] 输入（batch=5, features=4）
    auto input = om::Tensor::ones({5, 4});
    auto output = pModel->forward(input);  // 20260319 ZJH 前向传播

    // 20260319 ZJH 验证输出形状为 [5, 2]
    ASSERT_EQ(output.ndim(), 2);  // 20260319 ZJH 二维输出
    EXPECT_EQ(output.shape(0), 5);  // 20260319 ZJH 批次大小保持为 5
    EXPECT_EQ(output.shape(1), 2);  // 20260319 ZJH 输出特征维度为 2
}

// ===== 4. SGDStep =====
// 20260319 ZJH 测试 SGD 优化器单步更新
TEST(NNTest, SGDStep) {
    // 20260319 ZJH 创建一个需要梯度的参数张量，初始化为全 1
    auto param = om::Tensor::ones({2, 3});
    om::tensorSetRequiresGrad(param, true);  // 20260319 ZJH 设置需要梯度

    // 20260319 ZJH 简单前向传播：loss = sum(param * 2)
    auto doubled = om::tensorMulScalar(param, 2.0f);  // 20260319 ZJH param * 2
    auto loss = om::tensorSum(doubled);  // 20260319 ZJH sum(param * 2) = 2 * sum(param)

    // 20260319 ZJH 反向传播
    om::tensorBackward(loss);

    // 20260319 ZJH 验证梯度为 2.0（d(sum(2*param))/dparam = 2）
    auto grad = om::tensorGetGrad(param);
    ASSERT_EQ(grad.numel(), 6);  // 20260319 ZJH 梯度形状应与参数相同
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(grad.at({i, j}), 2.0f);  // 20260319 ZJH 每个元素梯度为 2
        }
    }

    // 20260319 ZJH 保存更新前的参数值
    float fParamBefore = param.at({0, 0});  // 20260319 ZJH 应为 1.0

    // 20260319 ZJH 创建 SGD 优化器，lr=0.1
    std::vector<om::Tensor*> vecParams = {&param};
    om::SGD optimizer(vecParams, 0.1f);
    optimizer.step();  // 20260319 ZJH 执行一步更新

    // 20260319 ZJH 验证参数更新：param = 1.0 - 0.1 * 2.0 = 0.8
    float fParamAfter = param.at({0, 0});
    EXPECT_NEAR(fParamAfter, 0.8f, 1e-5f);  // 20260319 ZJH 验证参数已正确更新
    EXPECT_LT(fParamAfter, fParamBefore);  // 20260319 ZJH 参数值应减小
}

// ===== 5. AdamStep =====
// 20260319 ZJH 测试 Adam 优化器单步更新
TEST(NNTest, AdamStep) {
    // 20260319 ZJH 创建参数张量，初始化为全 2
    auto param = om::Tensor::full({2, 2}, 2.0f);
    om::tensorSetRequiresGrad(param, true);  // 20260319 ZJH 设置需要梯度

    // 20260319 ZJH 简单前向 + 反向
    auto scaled = om::tensorMulScalar(param, 3.0f);  // 20260319 ZJH param * 3
    auto loss = om::tensorSum(scaled);  // 20260319 ZJH sum(param * 3)
    om::tensorBackward(loss);  // 20260319 ZJH 反向传播，grad = 3.0

    // 20260319 ZJH 保存更新前的参数值
    float fParamBefore = param.at({0, 0});  // 20260319 ZJH 应为 2.0

    // 20260319 ZJH 创建 Adam 优化器
    std::vector<om::Tensor*> vecParams = {&param};
    om::Adam optimizer(vecParams, 0.01f);  // 20260319 ZJH lr=0.01
    optimizer.step();  // 20260319 ZJH 执行一步更新

    // 20260319 ZJH 验证参数已被更新（值应减小）
    float fParamAfter = param.at({0, 0});
    EXPECT_LT(fParamAfter, fParamBefore);  // 20260319 ZJH Adam 更新后参数应减小
    EXPECT_GT(fParamAfter, 0.0f);  // 20260319 ZJH 不应降为负数（一步更新量有限）
}

// ===== 6. LinearBackward =====
// 20260319 ZJH 测试 Linear 层反向传播，验证权重梯度存在且形状正确
TEST(NNTest, LinearBackward) {
    // 20260319 ZJH 创建 Linear(3, 2) 层
    om::Linear layer(3, 2);
    // 20260319 ZJH 创建 [4, 3] 输入（batch=4, features=3）
    auto input = om::Tensor::ones({4, 3});
    // 20260319 ZJH 前向传播
    auto output = layer.forward(input);
    // 20260319 ZJH 计算损失：sum(output)
    auto loss = om::tensorSum(output);
    // 20260319 ZJH 清零梯度后反向传播
    layer.zeroGrad();
    om::tensorBackward(loss);

    // 20260319 ZJH 获取所有参数并验证梯度
    auto vecParams = layer.parameters();
    ASSERT_GE(vecParams.size(), static_cast<size_t>(1));  // 20260319 ZJH 至少有权重参数

    // 20260319 ZJH 验证权重梯度（第一个参数是 weight [3, 2]）
    auto gradWeight = om::tensorGetGrad(*vecParams[0]);
    ASSERT_EQ(gradWeight.numel(), 6);  // 20260319 ZJH 梯度元素数 = 3 * 2 = 6
    EXPECT_EQ(gradWeight.shape(0), 3);  // 20260319 ZJH 梯度行数 = 输入特征数
    EXPECT_EQ(gradWeight.shape(1), 2);  // 20260319 ZJH 梯度列数 = 输出特征数

    // 20260319 ZJH 验证偏置梯度（第二个参数是 bias [1, 2]）
    if (vecParams.size() >= 2) {
        auto gradBias = om::tensorGetGrad(*vecParams[1]);
        ASSERT_EQ(gradBias.numel(), 2);  // 20260319 ZJH 偏置梯度元素数 = 2
    }
}

// ===== 7. CrossEntropyForward =====
// 20260319 ZJH 测试 CrossEntropyLoss 前向计算，验证输出为标量
TEST(NNTest, CrossEntropyForward) {
    om::CrossEntropyLoss criterion;  // 20260319 ZJH 创建交叉熵损失

    // 20260319 ZJH 创建 logits [2, 3] 和 one-hot targets [2, 3]
    auto logits = om::Tensor::zeros({2, 3});
    // 20260319 ZJH 设置 logits 值：第一个样本偏向类别 0，第二个偏向类别 1
    logits.setAt({0, 0}, 2.0f);
    logits.setAt({0, 1}, 0.5f);
    logits.setAt({0, 2}, 0.1f);
    logits.setAt({1, 0}, 0.1f);
    logits.setAt({1, 1}, 2.5f);
    logits.setAt({1, 2}, 0.3f);

    auto targets = om::Tensor::zeros({2, 3});
    targets.setAt({0, 0}, 1.0f);  // 20260319 ZJH 第一个样本类别 0
    targets.setAt({1, 1}, 1.0f);  // 20260319 ZJH 第二个样本类别 1

    // 20260319 ZJH 前向计算损失
    auto loss = criterion.forward(logits, targets);

    // 20260319 ZJH 验证输出是标量
    EXPECT_EQ(loss.numel(), 1);  // 20260319 ZJH 标量张量元素数为 1
    EXPECT_GT(loss.item(), 0.0f);  // 20260319 ZJH 损失值应大于零
    EXPECT_LT(loss.item(), 10.0f);  // 20260319 ZJH 损失值应在合理范围内
}

// ===== 8. ModuleParameters =====
// 20260319 ZJH 测试 model.parameters() 递归返回所有参数
TEST(NNTest, ModuleParameters) {
    // 20260319 ZJH 构建模型：Linear(4,3) -> ReLU -> Linear(3,2)
    auto pModel = std::make_shared<om::Sequential>();
    pModel->add(std::make_shared<om::Linear>(4, 3));  // 20260319 ZJH weight[4,3] + bias[1,3] = 2 个参数
    pModel->add(std::make_shared<om::ReLU>());         // 20260319 ZJH 无参数
    pModel->add(std::make_shared<om::Linear>(3, 2));  // 20260319 ZJH weight[3,2] + bias[1,2] = 2 个参数

    auto vecParams = pModel->parameters();  // 20260319 ZJH 获取所有参数
    // 20260319 ZJH 总共应有 4 个参数（两层各有 weight + bias）
    EXPECT_EQ(vecParams.size(), static_cast<size_t>(4));

    // 20260319 ZJH 验证所有参数都不为空且需要梯度
    for (auto* pParam : vecParams) {
        EXPECT_GT(pParam->numel(), 0);  // 20260319 ZJH 参数不为空
        EXPECT_TRUE(pParam->requiresGrad());  // 20260319 ZJH 参数需要梯度
    }

    // 20260319 ZJH 验证参数形状：weight1[4,3], bias1[1,3], weight2[3,2], bias2[1,2]
    EXPECT_EQ(vecParams[0]->numel(), 12);  // 20260319 ZJH weight1: 4*3=12
    EXPECT_EQ(vecParams[1]->numel(), 3);   // 20260319 ZJH bias1: 1*3=3
    EXPECT_EQ(vecParams[2]->numel(), 6);   // 20260319 ZJH weight2: 3*2=6
    EXPECT_EQ(vecParams[3]->numel(), 2);   // 20260319 ZJH bias2: 1*2=2
}

// ===== 9. ZeroGrad =====
// 20260319 ZJH 测试 model.zeroGrad() 清零所有参数梯度
TEST(NNTest, ZeroGrad) {
    // 20260319 ZJH 创建 Linear(3, 2) 层
    om::Linear layer(3, 2);
    // 20260319 ZJH 前向 + 反向传播，使参数产生梯度
    auto input = om::Tensor::ones({2, 3});
    auto output = layer.forward(input);
    auto loss = om::tensorSum(output);
    om::tensorBackward(loss);

    // 20260319 ZJH 验证梯度存在
    auto vecParams = layer.parameters();
    auto gradBefore = om::tensorGetGrad(*vecParams[0]);
    ASSERT_GT(gradBefore.numel(), 0);  // 20260319 ZJH 反向传播后应有梯度

    // 20260319 ZJH 清零所有梯度
    layer.zeroGrad();

    // 20260319 ZJH 验证梯度已清零
    for (auto* pParam : vecParams) {
        auto grad = om::tensorGetGrad(*pParam);
        EXPECT_EQ(grad.numel(), 0);  // 20260319 ZJH 清零后梯度应为空（zero() 释放了存储）
    }
}

// ===== 10. TrainEvalMode =====
// 20260319 ZJH 测试训练/评估模式递归传播
TEST(NNTest, TrainEvalMode) {
    // 20260319 ZJH 构建模型
    auto pModel = std::make_shared<om::Sequential>();
    auto pLinear1 = std::make_shared<om::Linear>(4, 3);
    auto pRelu = std::make_shared<om::ReLU>();
    auto pLinear2 = std::make_shared<om::Linear>(3, 2);
    pModel->add(pLinear1);
    pModel->add(pRelu);
    pModel->add(pLinear2);

    // 20260319 ZJH 默认应为训练模式
    EXPECT_TRUE(pModel->isTraining());  // 20260319 ZJH Sequential 训练模式
    EXPECT_TRUE(pLinear1->isTraining());  // 20260319 ZJH Linear1 训练模式
    EXPECT_TRUE(pRelu->isTraining());  // 20260319 ZJH ReLU 训练模式
    EXPECT_TRUE(pLinear2->isTraining());  // 20260319 ZJH Linear2 训练模式

    // 20260319 ZJH 设置为评估模式
    pModel->eval();
    EXPECT_FALSE(pModel->isTraining());  // 20260319 ZJH Sequential 评估模式
    EXPECT_FALSE(pLinear1->isTraining());  // 20260319 ZJH Linear1 评估模式
    EXPECT_FALSE(pRelu->isTraining());  // 20260319 ZJH ReLU 评估模式
    EXPECT_FALSE(pLinear2->isTraining());  // 20260319 ZJH Linear2 评估模式

    // 20260319 ZJH 恢复为训练模式
    pModel->train();
    EXPECT_TRUE(pModel->isTraining());  // 20260319 ZJH Sequential 恢复训练模式
    EXPECT_TRUE(pLinear1->isTraining());  // 20260319 ZJH Linear1 恢复训练模式
    EXPECT_TRUE(pRelu->isTraining());  // 20260319 ZJH ReLU 恢复训练模式
    EXPECT_TRUE(pLinear2->isTraining());  // 20260319 ZJH Linear2 恢复训练模式
}
