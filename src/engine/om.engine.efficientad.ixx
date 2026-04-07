// 20260322 ZJH EfficientAD 异常检测模块
// 实现教师-学生异常检测网络 (EfficientAD)
// 核心思想: 教师网络提取正常样本特征，学生网络模仿教师输出
//           推理时教师-学生输出差异（MSE）= 异常分数图
// 教师网络: 预训练或在正常样本上训练的轻量 CNN
// 学生网络: 与教师结构相同但随机初始化，通过知识蒸馏学习正常模式
// 异常区域: 学生无法复现教师输出 → MSE 高 → 异常
// 20260402 ZJH Phase 1.2: 支持 ImageNet 预训练 ResNet18 骨干替换 4 层轻量 CNN
//   教师网络使用预训练 ResNet18（冻结），学生网络使用随机初始化 ResNet18
//   bUsePretrainedBackbone=true(默认) → ResNet18 骨干
//   bUsePretrainedBackbone=false → 保留原有 4 层 CNN（向后兼容）
module;

// 20260406 ZJH 标准库头文件包含
#include <vector>    // 20260406 ZJH 动态数组容器
#include <string>    // 20260406 ZJH 字符串类型
#include <utility>   // 20260406 ZJH std::pair, std::move 等工具
#include <string>    // 20260406 ZJH 重复包含（无副作用，保留兼容）
#include <memory>    // 20260406 ZJH 智能指针 (shared_ptr, unique_ptr)
#include <cmath>     // 20260406 ZJH 数学函数 (sqrt, exp)

export module om.engine.efficientad;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.engine.resnet;  // 20260402 ZJH 导入 ResNet18/BasicBlock 用于预训练骨干

export namespace om {

// 20260402 ZJH ResNet18Backbone — EfficientAD 专用 ResNet18 骨干网络
// 与 EfficientADBackbone 接口兼容（提供 forward / forwardFeatures / forwardMultiScale）
// 结构: conv1(3x3,s1,p1) → bn1 → relu → maxpool(3,2,1)
//        → Layer1(2xBasicBlock, 64→64)   [N, 64, H/2, W/2]
//        → Layer2(2xBasicBlock, 64→128, s2) [N, 128, H/4, W/4]
//        → Layer3(2xBasicBlock, 128→256, s2) [N, 256, H/8, W/8]
//        → Layer4(2xBasicBlock, 256→512, s2) [N, 512, H/16, W/16]
// forward 输出 [N, 256, H/16, W/16]（通过 1x1 conv 将 512→256 适配原始接口）
// forwardFeatures 输出 [N, 256, H/8, W/8]（使用 Layer3 输出）
class ResNet18Backbone : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    ResNet18Backbone(int nInChannels = 3) {
        // 20260402 ZJH Stem: conv1 → bn1 → relu → maxpool
        m_pConv1 = std::make_shared<Conv2d>(nInChannels, 64, 3, 1, 1, false);  // 20260402 ZJH 3x3 stem conv
        m_pBn1 = std::make_shared<BatchNorm2d>(64);  // 20260402 ZJH stem BN
        m_pMaxpool = std::make_shared<MaxPool2d>(3, 2, 1);  // 20260402 ZJH 3x3 stride=2 pad=1
        m_pRelu = std::make_shared<ReLU>();  // 20260402 ZJH 共用 ReLU

        // 20260402 ZJH 注册 stem 子模块
        registerModule("conv1", m_pConv1);
        registerModule("bn1", m_pBn1);
        registerModule("maxpool", m_pMaxpool);
        registerModule("relu", m_pRelu);

        // 20260402 ZJH 构建 Layer1~Layer4（与 ResNet18 完全一致）
        auto addLayer = [&](const std::string& strName,
                            std::vector<std::shared_ptr<BasicBlock>>& vecBlocks,
                            int nInC, int nOutC, int nStride) {
            vecBlocks.reserve(2);  // 20260402 ZJH 每层 2 个 BasicBlock
            for (int i = 0; i < 2; ++i) {
                int s = (i == 0) ? nStride : 1;  // 20260402 ZJH 仅第一个 block 使用指定 stride
                int c = (i == 0) ? nInC : nOutC;  // 20260402 ZJH 第一个 block 输入通道
                auto pBlock = std::make_shared<BasicBlock>(c, nOutC, s);
                vecBlocks.push_back(pBlock);
                registerModule(strName + "." + std::to_string(i), pBlock);  // 20260402 ZJH 注册子模块
            }
        };

        addLayer("layer1", m_vecLayer1, 64, 64, 1);    // 20260402 ZJH 64→64, stride=1
        addLayer("layer2", m_vecLayer2, 64, 128, 2);   // 20260402 ZJH 64→128, stride=2
        addLayer("layer3", m_vecLayer3, 128, 256, 2);  // 20260402 ZJH 128→256, stride=2
        addLayer("layer4", m_vecLayer4, 256, 512, 2);  // 20260402 ZJH 256→512, stride=2

        // 20260402 ZJH 适配层: 1x1 conv 将 Layer4 的 512ch 降维到 256ch
        // 保持与 EfficientADBackbone forward() 输出 [N, 256, H/16, W/16] 接口一致
        m_pAdaptConv = std::make_shared<Conv2d>(512, 256, 1, 1, 0, true);  // 20260402 ZJH 1x1 conv 512→256
        registerModule("adapt_conv", m_pAdaptConv);

        // 20260402 ZJH 适配层: 1x1 conv 将 Layer3 的 256ch 保持不变（可省略，但保持架构对称性）
        // forwardFeatures 直接使用 Layer3 输出（已经是 256ch），无需适配
    }

    // 20260402 ZJH forward — 骨干网络前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, 256, H/16, W/16] 特征图（与 EfficientADBackbone::forward 输出维度一致）
    Tensor forward(const Tensor& input) override {
        // 20260402 ZJH Stem
        auto x = m_pConv1->forward(input);   // 20260402 ZJH [N, 64, H, W]
        x = m_pBn1->forward(x);
        x = m_pRelu->forward(x);
        if (input.shape(2) > 32) {
            x = m_pMaxpool->forward(x);      // 20260402 ZJH [N, 64, H/2, W/2]
        }

        // 20260402 ZJH Layer1~Layer4
        for (auto& pBlock : m_vecLayer1) x = pBlock->forward(x);  // 20260402 ZJH [N, 64, H/2, W/2]
        for (auto& pBlock : m_vecLayer2) x = pBlock->forward(x);  // 20260402 ZJH [N, 128, H/4, W/4]
        for (auto& pBlock : m_vecLayer3) x = pBlock->forward(x);  // 20260402 ZJH [N, 256, H/8, W/8]
        for (auto& pBlock : m_vecLayer4) x = pBlock->forward(x);  // 20260402 ZJH [N, 512, H/16, W/16]

        // 20260402 ZJH 适配: 512→256 通道（匹配 EfficientADBackbone 的 256ch 输出）
        x = m_pAdaptConv->forward(x);  // 20260402 ZJH [N, 256, H/16, W/16]

        return x;  // 20260402 ZJH 返回 [N, 256, H/16, W/16]
    }

    // 20260402 ZJH forwardFeatures — 提取中间特征（不经过 Layer4）
    // 使用 Layer3 输出 [N, 256, H/8, W/8]（与 EfficientADBackbone::forwardFeatures 一致）
    // 用于高分辨率异常定位
    Tensor forwardFeatures(const Tensor& input) {
        // 20260402 ZJH Stem
        auto x = m_pConv1->forward(input);
        x = m_pBn1->forward(x);
        x = m_pRelu->forward(x);
        if (input.shape(2) > 32) {
            x = m_pMaxpool->forward(x);
        }

        // 20260402 ZJH Layer1~Layer3（跳过 Layer4 以获得更高空间分辨率）
        for (auto& pBlock : m_vecLayer1) x = pBlock->forward(x);  // 20260402 ZJH [N, 64, H/2, W/2]
        for (auto& pBlock : m_vecLayer2) x = pBlock->forward(x);  // 20260402 ZJH [N, 128, H/4, W/4]
        for (auto& pBlock : m_vecLayer3) x = pBlock->forward(x);  // 20260402 ZJH [N, 256, H/8, W/8]

        return x;  // 20260402 ZJH 返回 [N, 256, H/8, W/8] 高分辨率特征
    }

    // 20260402 ZJH forwardMultiScale — 提取三级多尺度特征（与 EfficientADBackbone 接口一致）
    // 返回: {feat2: [N,128,H/4,W/4], feat3: [N,256,H/8,W/8], feat4: [N,256,H/16,W/16]}
    // 注意: feat2 通道数 128 对应 Layer2 输出（与 EfficientADBackbone 的 64 不同）
    //       feat4 使用 Layer4(512ch) 经 adaptConv 降维到 256（与 EfficientADBackbone 一致）
    struct MultiScaleFeatures {
        Tensor feat2;  // 20260402 ZJH [N, 128, H/4, W/4] — Layer2 输出
        Tensor feat3;  // 20260402 ZJH [N, 256, H/8, W/8] — Layer3 输出
        Tensor feat4;  // 20260402 ZJH [N, 256, H/16, W/16] — Layer4 + adaptConv 输出
    };

    MultiScaleFeatures forwardMultiScale(const Tensor& input) {
        // 20260402 ZJH Stem
        auto x = m_pConv1->forward(input);
        x = m_pBn1->forward(x);
        x = m_pRelu->forward(x);
        if (input.shape(2) > 32) {
            x = m_pMaxpool->forward(x);
        }

        // 20260402 ZJH Layer1
        for (auto& pBlock : m_vecLayer1) x = pBlock->forward(x);

        // 20260402 ZJH Layer2 — 保留输出作为 feat2
        for (auto& pBlock : m_vecLayer2) x = pBlock->forward(x);
        auto feat2 = x;  // 20260402 ZJH [N, 128, H/4, W/4]

        // 20260402 ZJH Layer3 — 保留输出作为 feat3
        for (auto& pBlock : m_vecLayer3) x = pBlock->forward(x);
        auto feat3 = x;  // 20260402 ZJH [N, 256, H/8, W/8]

        // 20260402 ZJH Layer4 + adaptConv — 作为 feat4
        for (auto& pBlock : m_vecLayer4) x = pBlock->forward(x);
        auto feat4 = m_pAdaptConv->forward(x);  // 20260402 ZJH [N, 256, H/16, W/16]

        return {feat2, feat3, feat4};  // 20260402 ZJH 返回三级特征
    }

private:
    // 20260402 ZJH Stem 模块
    std::shared_ptr<Conv2d> m_pConv1;        // 20260402 ZJH stem conv 3→64
    std::shared_ptr<BatchNorm2d> m_pBn1;     // 20260402 ZJH stem BN
    std::shared_ptr<MaxPool2d> m_pMaxpool;   // 20260402 ZJH stem MaxPool
    std::shared_ptr<ReLU> m_pRelu;           // 20260402 ZJH ReLU（共用）

    // 20260402 ZJH ResNet18 四个残差层
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer1;  // 20260402 ZJH 64→64
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer2;  // 20260402 ZJH 64→128, stride=2
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer3;  // 20260402 ZJH 128→256, stride=2
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer4;  // 20260402 ZJH 256→512, stride=2

    // 20260402 ZJH 适配层: 512→256（匹配 EfficientADBackbone 输出维度）
    std::shared_ptr<Conv2d> m_pAdaptConv;  // 20260402 ZJH 1x1 conv 512→256
};

// 20260322 ZJH EfficientADBackbone — EfficientAD 共用骨干网络
// 4 层 Conv+BN+ReLU+MaxPool 结构，提取多尺度特征
// 通道变化: nInChannels → 32 → 64 → 128 → 256
// 空间变化: 每层 MaxPool stride=2 下采样，输出空间尺寸 = 输入 / 16
class EfficientADBackbone : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    EfficientADBackbone(int nInChannels = 3)
        : m_conv1(nInChannels, 32, 3, 1, 1, false),  // 20260322 ZJH 层1: Cin→32, 3x3, pad=1
          m_bn1(32),                                    // 20260406 ZJH 层1 BN: 32 通道
          m_pool1(2, 2, 0),                            // 20260322 ZJH MaxPool 2x2 stride=2
          m_conv2(32, 64, 3, 1, 1, false),             // 20260322 ZJH 层2: 32→64
          m_bn2(64),                                    // 20260406 ZJH 层2 BN: 64 通道
          m_pool2(2, 2, 0),                            // 20260406 ZJH 层2 MaxPool 2x2 stride=2
          m_conv3(64, 128, 3, 1, 1, false),            // 20260322 ZJH 层3: 64→128
          m_bn3(128),                                   // 20260406 ZJH 层3 BN: 128 通道
          m_pool3(2, 2, 0),                            // 20260406 ZJH 层3 MaxPool 2x2 stride=2
          m_conv4(128, 256, 3, 1, 1, false),           // 20260322 ZJH 层4: 128→256
          m_bn4(256),                                   // 20260406 ZJH 层4 BN: 256 通道
          m_pool4(2, 2, 0)                             // 20260406 ZJH 层4 MaxPool 2x2 stride=2
    {}

    // 20260322 ZJH forward — 骨干网络前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, 256, H/16, W/16] 特征图
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 层1: Conv3x3 → BN → ReLU → MaxPool
        auto x = m_conv1.forward(input);
        x = m_bn1.forward(x);
        x = m_relu.forward(x);
        x = m_pool1.forward(x);   // 20260322 ZJH 空间下采样 /2

        // 20260322 ZJH 层2: Conv3x3 → BN → ReLU → MaxPool
        x = m_conv2.forward(x);
        x = m_bn2.forward(x);
        x = m_relu.forward(x);
        x = m_pool2.forward(x);   // 20260322 ZJH 空间下采样 /4

        // 20260322 ZJH 层3: Conv3x3 → BN → ReLU → MaxPool
        x = m_conv3.forward(x);
        x = m_bn3.forward(x);
        x = m_relu.forward(x);
        x = m_pool3.forward(x);   // 20260322 ZJH 空间下采样 /8

        // 20260322 ZJH 层4: Conv3x3 → BN → ReLU → MaxPool
        x = m_conv4.forward(x);
        x = m_bn4.forward(x);
        x = m_relu.forward(x);
        x = m_pool4.forward(x);   // 20260322 ZJH 空间下采样 /16

        return x;  // 20260322 ZJH 返回 [N, 256, H/16, W/16] 特征图
    }

    // 20260322 ZJH 提取中间特征（不经过最后一层 MaxPool）
    // 用于更高分辨率的异常定位
    // 返回: [N, 256, H/8, W/8] 特征图
    Tensor forwardFeatures(const Tensor& input) {
        auto x = m_conv1.forward(input);
        x = m_bn1.forward(x);
        x = m_relu.forward(x);
        x = m_pool1.forward(x);

        x = m_conv2.forward(x);
        x = m_bn2.forward(x);
        x = m_relu.forward(x);
        x = m_pool2.forward(x);

        x = m_conv3.forward(x);
        x = m_bn3.forward(x);
        x = m_relu.forward(x);
        x = m_pool3.forward(x);

        // 20260322 ZJH 最后一层不做 MaxPool，保留更高空间分辨率
        x = m_conv4.forward(x);
        x = m_bn4.forward(x);
        x = m_relu.forward(x);

        return x;  // 20260322 ZJH 返回 [N, 256, H/8, W/8] 高分辨率特征图
    }

    // 20260402 ZJH forwardMultiScale — 提取三级多尺度特征（对标 Halcon 多尺度异常检测）
    // 返回: {feat_H/4, feat_H/8, feat_H/16} 三级特征金字塔
    // feat2: [N,  64, H/4, W/4]  — 细粒度纹理，检测小缺陷
    // feat3: [N, 128, H/8, W/8]  — 中等尺度
    // feat4: [N, 256, H/16, W/16] — 粗粒度语义，检测大面积异常
    struct MultiScaleFeatures {
        Tensor feat2;  // 20260402 ZJH [N, 64, H/4, W/4]
        Tensor feat3;  // 20260402 ZJH [N, 128, H/8, W/8]
        Tensor feat4;  // 20260402 ZJH [N, 256, H/16, W/16]
    };

    MultiScaleFeatures forwardMultiScale(const Tensor& input) {
        // 20260402 ZJH 层1: /2
        auto x1 = m_relu.forward(m_bn1.forward(m_conv1.forward(input)));
        x1 = m_pool1.forward(x1);  // 20260402 ZJH [N, 32, H/2, W/2]

        // 20260402 ZJH 层2: /4 — 保留为 feat2
        auto x2 = m_relu.forward(m_bn2.forward(m_conv2.forward(x1)));
        x2 = m_pool2.forward(x2);  // 20260402 ZJH [N, 64, H/4, W/4]

        // 20260402 ZJH 层3: /8 — 保留为 feat3
        auto x3 = m_relu.forward(m_bn3.forward(m_conv3.forward(x2)));
        x3 = m_pool3.forward(x3);  // 20260402 ZJH [N, 128, H/8, W/8]

        // 20260402 ZJH 层4: /16 — 保留为 feat4
        auto x4 = m_relu.forward(m_bn4.forward(m_conv4.forward(x3)));
        x4 = m_pool4.forward(x4);  // 20260402 ZJH [N, 256, H/16, W/16]

        return {x2, x3, x4};  // 20260402 ZJH 返回三级特征
    }

    // 20260322 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        appendVec(m_conv1.parameters());  appendVec(m_bn1.parameters());
        appendVec(m_conv2.parameters());  appendVec(m_bn2.parameters());
        appendVec(m_conv3.parameters());  appendVec(m_bn3.parameters());
        appendVec(m_conv4.parameters());  appendVec(m_bn4.parameters());
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        appendVec(m_conv1.namedParameters(makeP("conv1")));
        appendVec(m_bn1.namedParameters(makeP("bn1")));
        appendVec(m_conv2.namedParameters(makeP("conv2")));
        appendVec(m_bn2.namedParameters(makeP("bn2")));
        appendVec(m_conv3.namedParameters(makeP("conv3")));
        appendVec(m_bn3.namedParameters(makeP("bn3")));
        appendVec(m_conv4.namedParameters(makeP("conv4")));
        appendVec(m_bn4.namedParameters(makeP("bn4")));
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        appendVec(m_bn1.buffers());  // 20260328 ZJH bn1 running_mean/running_var
        appendVec(m_bn2.buffers());  // 20260328 ZJH bn2 running_mean/running_var
        appendVec(m_bn3.buffers());  // 20260328 ZJH bn3 running_mean/running_var
        appendVec(m_bn4.buffers());  // 20260328 ZJH bn4 running_mean/running_var
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("bn1", m_bn1);
        appendBufs("bn2", m_bn2);
        appendBufs("bn3", m_bn3);
        appendBufs("bn4", m_bn4);
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);  m_bn1.train(bMode);
        m_conv2.train(bMode);  m_bn2.train(bMode);
        m_conv3.train(bMode);  m_bn3.train(bMode);
        m_conv4.train(bMode);  m_bn4.train(bMode);
    }

private:
    Conv2d m_conv1, m_conv2, m_conv3, m_conv4;        // 20260322 ZJH 4 层 3x3 卷积
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;            // 20260322 ZJH 4 层 BN
    MaxPool2d m_pool1, m_pool2, m_pool3, m_pool4;      // 20260322 ZJH 4 层 MaxPool
    ReLU m_relu;                                         // 20260322 ZJH ReLU 激活（无状态，共用）
};

// 20260322 ZJH EfficientAD — 教师-学生异常检测网络
// 持有教师网络 (teacher) 和学生网络 (student) 两个相同结构的骨干
// 训练阶段:
//   1. 教师网络先在正常样本上预训练（或使用预训练权重冻结）
//   2. 学生网络通过模仿教师输出进行知识蒸馏训练
//   3. 损失函数: MSE(teacher_features, student_features)
// 推理阶段:
//   1. 同时前向计算教师和学生的特征
//   2. 异常分数 = MSE(teacher_features, student_features)
//   3. 正常区域学生能准确复现教师输出（MSE低），异常区域学生失败（MSE高）
// 20260402 ZJH Phase 1.2: 支持 ResNet18 预训练骨干
//   bUsePretrainedBackbone=true → 教师=预训练 ResNet18（冻结），学生=随机 ResNet18
//   bUsePretrainedBackbone=false → 教师/学生=原始 4 层 CNN（向后兼容）
class EfficientAD : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    // 20260402 ZJH bUsePretrainedBackbone: 是否使用 ResNet18 骨干，默认 true
    EfficientAD(int nInChannels = 3, bool bUsePretrainedBackbone = true)
        : // 20260328 ZJH 异常分数统计量（标量张量），用于自适应阈值校准
          // 存为 Tensor buffer 而非 float，利用现有序列化器自动保存/加载
          m_tScoreMean(Tensor::zeros({1})),   // 20260328 ZJH 正常样本异常分数均值
          m_tScoreStd(Tensor::zeros({1})),    // 20260328 ZJH 正常样本异常分数标准差
          m_tThreshold(Tensor::zeros({1})),   // 20260328 ZJH 异常判定阈值 (mean + k*std)
          m_bUsePretrainedBackbone(bUsePretrainedBackbone)  // 20260402 ZJH 骨干选择标志
    {
        if (m_bUsePretrainedBackbone) {
            // 20260402 ZJH 使用 ResNet18 骨干: 教师=预训练（冻结），学生=随机初始化
            m_pResNetTeacher = std::make_shared<ResNet18Backbone>(nInChannels);  // 20260402 ZJH 教师 ResNet18
            m_pResNetStudent = std::make_shared<ResNet18Backbone>(nInChannels);  // 20260402 ZJH 学生 ResNet18
        } else {
            // 20260402 ZJH 使用原始 4 层 CNN（向后兼容）
            m_pTeacher = std::make_shared<EfficientADBackbone>(nInChannels);  // 20260322 ZJH 教师网络
            m_pStudent = std::make_shared<EfficientADBackbone>(nInChannels);  // 20260322 ZJH 学生网络
        }
    }

    // 20260322 ZJH forward — 计算异常分数图
    // input: [N, Cin, H, W]
    // 返回: [N, 1, H/8, W/8] 异常分数图（逐像素 MSE）
    // 注意: forward 默认使用 forwardFeatures（高分辨率）计算异常分数
    Tensor forward(const Tensor& input) override {
        return computeAnomalyScore(input);  // 20260322 ZJH 委托给异常分数计算
    }

    // 20260322 ZJH computeAnomalyScore — 计算逐像素异常分数图
    // 对教师和学生的特征图逐元素求 MSE 差异
    // input: [N, Cin, H, W]
    // 返回: [N, 1, Hf, Wf] 异常分数图（Hf=H/8, Wf=W/8）
    Tensor computeAnomalyScore(const Tensor& input) {
        // 20260402 ZJH 根据骨干类型选择教师/学生网络
        Tensor teacherFeats, studentFeats;
        if (m_bUsePretrainedBackbone) {
            // 20260402 ZJH ResNet18 骨干: forwardFeatures 输出 [N, 256, H/8, W/8]
            teacherFeats = m_pResNetTeacher->forwardFeatures(input);  // 20260402 ZJH 教师 ResNet18 特征
            studentFeats = m_pResNetStudent->forwardFeatures(input);  // 20260402 ZJH 学生 ResNet18 特征
        } else {
            // 20260322 ZJH 教师特征（评估模式，不更新 BN 统计量）
            teacherFeats = m_pTeacher->forwardFeatures(input);  // 20260322 ZJH [N, 256, H/8, W/8]
            // 20260322 ZJH 学生特征
            studentFeats = m_pStudent->forwardFeatures(input);  // 20260322 ZJH [N, 256, H/8, W/8]
        }

        // 20260322 ZJH 计算逐元素差异: diff = teacher - student
        auto diff = tensorSub(teacherFeats, studentFeats);  // 20260322 ZJH [N, 256, H/8, W/8]

        // 20260322 ZJH 计算逐元素平方: diff^2
        auto diffSq = tensorMul(diff, diff);  // 20260322 ZJH [N, 256, H/8, W/8]

        // 20260322 ZJH 沿通道维求均值得到异常分数图
        // 手动计算通道均值: [N, 256, H, W] → [N, 1, H, W]
        auto cDiffSq = diffSq.contiguous();
        int nBatch = cDiffSq.shape(0);       // 20260322 ZJH 批次大小
        int nChannels = cDiffSq.shape(1);    // 20260322 ZJH 通道数 (256)
        int nH = cDiffSq.shape(2);           // 20260322 ZJH 特征图高度
        int nW = cDiffSq.shape(3);           // 20260322 ZJH 特征图宽度

        // 20260322 ZJH 创建输出异常分数图 [N, 1, H, W]
        auto anomalyMap = Tensor::zeros({nBatch, 1, nH, nW});
        const float* pDiffSq = cDiffSq.floatDataPtr();    // 20260322 ZJH 平方差数据指针
        float* pOut = anomalyMap.mutableFloatDataPtr();     // 20260322 ZJH 输出数据指针

        int nSpatial = nH * nW;  // 20260322 ZJH 空间元素数
        float fInvC = 1.0f / static_cast<float>(nChannels);  // 20260322 ZJH 通道均值缩放因子

        // 20260322 ZJH 对每个 batch 和空间位置，沿通道维求均值
        for (int n = 0; n < nBatch; ++n) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    float fSum = 0.0f;  // 20260322 ZJH 通道维累加
                    for (int c = 0; c < nChannels; ++c) {
                        // 20260322 ZJH 索引: [n, c, h, w] = n*C*H*W + c*H*W + h*W + w
                        int nIdx = n * nChannels * nSpatial + c * nSpatial + h * nW + w;
                        fSum += pDiffSq[nIdx];
                    }
                    // 20260322 ZJH 输出索引: [n, 0, h, w] = n*H*W + h*W + w
                    pOut[n * nSpatial + h * nW + w] = fSum * fInvC;
                }
            }
        }

        return anomalyMap;  // 20260322 ZJH 返回异常分数图 [N, 1, H/8, W/8]
    }

    // 20260402 ZJH computeAnomalyScoreMultiScale — 多尺度异常分数图（无额外参数）
    // 提取 3 级特征（H/4, H/8, H/16），各级独立计算 Teacher-Student MSE
    // 然后上采样到 H/4 后求和融合 → 分辨率从 H/8 提升到 H/4（4x 像素密度）
    // input: [N, Cin, H, W]
    // 返回: [N, 1, H/4, W/4] 多尺度融合异常分数图
    Tensor computeAnomalyScoreMultiScale(const Tensor& input) {
        // 20260402 ZJH 根据骨干类型提取教师和学生的三级特征
        // ResNet18Backbone::MultiScaleFeatures 和 EfficientADBackbone::MultiScaleFeatures
        // 结构相同（feat2, feat3, feat4），但通道数可能不同:
        //   ResNet18: feat2=[N,128,H/4,W/4], feat3=[N,256,H/8,W/8], feat4=[N,256,H/16,W/16]
        //   Legacy:   feat2=[N,64,H/4,W/4],  feat3=[N,128,H/8,W/8], feat4=[N,256,H/16,W/16]
        // MSE 差异计算要求 teacher 和 student 使用相同骨干（通道数一致）
        EfficientADBackbone::MultiScaleFeatures teacherMS, studentMS;
        if (m_bUsePretrainedBackbone) {
            // 20260402 ZJH ResNet18 骨干多尺度特征
            auto tMS = m_pResNetTeacher->forwardMultiScale(input);
            auto sMS = m_pResNetStudent->forwardMultiScale(input);
            // 20260402 ZJH 转换为 EfficientADBackbone::MultiScaleFeatures（结构兼容）
            teacherMS = {tMS.feat2, tMS.feat3, tMS.feat4};
            studentMS = {sMS.feat2, sMS.feat3, sMS.feat4};
        } else {
            teacherMS = m_pTeacher->forwardMultiScale(input);
            studentMS = m_pStudent->forwardMultiScale(input);
        }

        // 20260402 ZJH 各级 MSE 差异
        auto diff2 = tensorMul(tensorSub(teacherMS.feat2, studentMS.feat2),
                               tensorSub(teacherMS.feat2, studentMS.feat2));  // 20260402 ZJH [N,64,H/4,W/4]
        auto diff3 = tensorMul(tensorSub(teacherMS.feat3, studentMS.feat3),
                               tensorSub(teacherMS.feat3, studentMS.feat3));  // 20260402 ZJH [N,128,H/8,W/8]
        auto diff4 = tensorMul(tensorSub(teacherMS.feat4, studentMS.feat4),
                               tensorSub(teacherMS.feat4, studentMS.feat4));  // 20260402 ZJH [N,256,H/16,W/16]

        // 20260402 ZJH 各级通道均值 → [N,1,Hx,Wx]
        auto score2 = channelMean(diff2);  // 20260402 ZJH [N,1,H/4,W/4]
        auto score3 = channelMean(diff3);  // 20260402 ZJH [N,1,H/8,W/8]
        auto score4 = channelMean(diff4);  // 20260402 ZJH [N,1,H/16,W/16]

        // 20260402 ZJH 上采样到统一尺寸 H/4
        int nTargetH = score2.shape(2), nTargetW = score2.shape(3);
        auto score3up = nnUpsample(score3, nTargetH, nTargetW);  // 20260402 ZJH 最近邻上采样
        auto score4up = nnUpsample(score4, nTargetH, nTargetW);

        // 20260402 ZJH 加权融合: 高层权重更大（语义更强）
        auto fused = tensorAdd(tensorAdd(
            tensorMulScalar(score2, 0.25f),   // 20260402 ZJH 浅层 25%
            tensorMulScalar(score3up, 0.35f)), // 20260402 ZJH 中层 35%
            tensorMulScalar(score4up, 0.40f)); // 20260402 ZJH 深层 40%

        return fused;  // 20260402 ZJH [N, 1, H/4, W/4] 多尺度融合异常图
    }

    // 20260322 ZJH computeDistillationLoss — 计算知识蒸馏损失（训练学生时使用）
    // MSE(teacher_features, student_features) 全局平均
    // input: [N, Cin, H, W]
    // 返回: 标量损失张量 [1]
    Tensor computeDistillationLoss(const Tensor& input) {
        // 20260402 ZJH 根据骨干类型选择教师/学生网络计算蒸馏损失
        Tensor teacherFeats, studentFeats;
        if (m_bUsePretrainedBackbone) {
            // 20260402 ZJH ResNet18 骨干: 教师(冻结) vs 学生(可训练)
            teacherFeats = m_pResNetTeacher->forwardFeatures(input);
            studentFeats = m_pResNetStudent->forwardFeatures(input);
        } else {
            // 20260322 ZJH 教师特征（梯度不需要流过教师网络）
            teacherFeats = m_pTeacher->forwardFeatures(input);
            // 20260322 ZJH 学生特征（梯度需要流过学生网络）
            studentFeats = m_pStudent->forwardFeatures(input);
        }

        // 20260322 ZJH 计算 MSE 损失: mean((teacher - student)^2)
        auto diff = tensorSub(teacherFeats, studentFeats);
        auto diffSq = tensorMul(diff, diff);

        // 20260322 ZJH 全局求和再除以元素数得到 MSE
        auto loss = tensorSum(diffSq);  // 20260322 ZJH 求和
        float fNumel = static_cast<float>(diffSq.numel());  // 20260322 ZJH 元素总数
        loss = tensorMulScalar(loss, 1.0f / fNumel);  // 20260322 ZJH 平均化

        return loss;  // 20260322 ZJH 返回 MSE 损失标量
    }

    // 20260322 ZJH computeImageAnomalyScore — 计算整张图像的异常分数（标量）
    // 取异常分数图的最大值作为图像级异常分数
    // input: [N, Cin, H, W]
    // 返回: 异常分数（float）
    float computeImageAnomalyScore(const Tensor& input) {
        auto anomalyMap = computeAnomalyScore(input);  // 20260322 ZJH [N, 1, H/8, W/8]
        return tensorMax(anomalyMap);  // 20260322 ZJH 取最大值作为图像级异常分数
    }

    // 20260328 ZJH calibrate — 根据正常样本的异常分数分布校准阈值
    // vecScores: 所有正常训练样本的图像级异常分数（max of anomaly map）
    // fK: sigma 倍数，默认 3.0（3-sigma 规则：99.7% 正常样本在阈值以下）
    // 校准公式: threshold = mean + k * std
    void calibrate(const std::vector<float>& vecScores, float fK = 3.0f) {
        if (vecScores.empty()) return;  // 20260328 ZJH 空列表不校准

        // 20260328 ZJH 计算均值
        double dSum = 0.0;
        for (float fScore : vecScores) dSum += static_cast<double>(fScore);
        float fMean = static_cast<float>(dSum / static_cast<double>(vecScores.size()));

        // 20260328 ZJH 计算标准差
        double dVarSum = 0.0;
        for (float fScore : vecScores) {
            double dDiff = static_cast<double>(fScore) - static_cast<double>(fMean);
            dVarSum += dDiff * dDiff;
        }
        float fStd = static_cast<float>(std::sqrt(dVarSum / static_cast<double>(vecScores.size())));

        // 20260328 ZJH 防止 std=0（所有分数相同时），设置最小 std 为 mean 的 10%
        if (fStd < fMean * 0.1f) fStd = fMean * 0.1f;
        // 20260328 ZJH 极端情况：mean 也为 0，使用绝对下限
        if (fStd < 1e-6f) fStd = 1e-6f;

        // 20260328 ZJH 计算阈值: mean + k * std
        float fThreshold = fMean + fK * fStd;

        // 20260328 ZJH 写入张量 buffer（序列化器自动保存）
        m_tScoreMean.mutableFloatDataPtr()[0] = fMean;
        m_tScoreStd.mutableFloatDataPtr()[0] = fStd;
        m_tThreshold.mutableFloatDataPtr()[0] = fThreshold;

        m_bCalibrated = true;  // 20260328 ZJH 标记已校准
    }

    // 20260328 ZJH isCalibrated — 是否已完成阈值校准
    bool isCalibrated() const {
        // 20260328 ZJH threshold > 0 说明已校准（加载旧模型时 threshold=0 表示未校准）
        return m_bCalibrated || m_tThreshold.floatDataPtr()[0] > 0.0f;
    }

    // 20260328 ZJH anomalyThreshold — 获取异常判定阈值
    // 未校准时返回 fallback 值 0.5f（向后兼容旧模型）
    float anomalyThreshold() const {
        float fThreshold = m_tThreshold.floatDataPtr()[0];
        return (fThreshold > 0.0f) ? fThreshold : 0.5f;  // 20260328 ZJH fallback
    }

    // 20260402 ZJH setAnomalyThreshold — 手动设置异常判定阈值
    // 用于 F1-max 校准后覆盖 3-sigma 基线阈值
    // 参数: fThreshold - 新阈值（必须 > 0）
    void setAnomalyThreshold(float fThreshold) {
        if (fThreshold > 0.0f) {
            m_tThreshold.mutableFloatDataPtr()[0] = fThreshold;  // 20260402 ZJH 写入阈值张量
            m_bCalibrated = true;  // 20260402 ZJH 标记为已校准
        }
    }

    // 20260328 ZJH scoreMean — 获取正常样本异常分数均值
    float scoreMean() const { return m_tScoreMean.floatDataPtr()[0]; }

    // 20260328 ZJH scoreStd — 获取正常样本异常分数标准差
    float scoreStd() const { return m_tScoreStd.floatDataPtr()[0]; }

    // 20260402 ZJH isUsingPretrainedBackbone — 查询是否使用预训练骨干
    bool isUsingPretrainedBackbone() const { return m_bUsePretrainedBackbone; }

    // 20260322 ZJH getTeacher — 获取教师网络（原始 4 层 CNN）
    // 20260402 ZJH 仅在 bUsePretrainedBackbone=false 时有效
    std::shared_ptr<EfficientADBackbone> getTeacher() { return m_pTeacher; }

    // 20260322 ZJH getStudent — 获取学生网络（原始 4 层 CNN）
    // 20260402 ZJH 仅在 bUsePretrainedBackbone=false 时有效
    std::shared_ptr<EfficientADBackbone> getStudent() { return m_pStudent; }

    // 20260402 ZJH getResNetTeacher — 获取 ResNet18 教师网络
    std::shared_ptr<ResNet18Backbone> getResNetTeacher() { return m_pResNetTeacher; }

    // 20260402 ZJH getResNetStudent — 获取 ResNet18 学生网络
    std::shared_ptr<ResNet18Backbone> getResNetStudent() { return m_pResNetStudent; }

    // 20260402 ZJH getTeacherModule — 获取教师网络的 Module 引用（跨骨干类型统一接口）
    // 用于外部预训练权重加载（EngineBridge）
    Module* getTeacherModule() {
        if (m_bUsePretrainedBackbone && m_pResNetTeacher) return m_pResNetTeacher.get();
        if (m_pTeacher) return m_pTeacher.get();
        return nullptr;  // 20260402 ZJH 不应到达
    }

    // 20260402 ZJH getStudentModule — 获取学生网络的 Module 引用（跨骨干类型统一接口）
    Module* getStudentModule() {
        if (m_bUsePretrainedBackbone && m_pResNetStudent) return m_pResNetStudent.get();
        if (m_pStudent) return m_pStudent.get();
        return nullptr;
    }

    // 20260322 ZJH freezeTeacher — 冻结教师网络参数（训练学生时调用）
    // 20260402 ZJH 根据骨干类型冻结对应的教师网络
    void freezeTeacher() {
        if (m_bUsePretrainedBackbone && m_pResNetTeacher) {
            m_pResNetTeacher->eval();  // 20260402 ZJH ResNet18 教师设置为评估模式
        } else if (m_pTeacher) {
            m_pTeacher->eval();  // 20260322 ZJH 原始 CNN 教师设置为评估模式
        }
        // 20260322 ZJH 注意: 冻结梯度需要在优化器中排除教师参数
        // 实际训练时只将学生参数传给优化器即可
    }

    // 20260322 ZJH studentParameters — 仅返回学生网络的参数（用于优化器）
    // 20260402 ZJH 根据骨干类型返回对应学生网络的参数
    std::vector<Tensor*> studentParameters() {
        if (m_bUsePretrainedBackbone && m_pResNetStudent) {
            return m_pResNetStudent->parameters();  // 20260402 ZJH ResNet18 学生参数
        }
        return m_pStudent->parameters();  // 20260322 ZJH 原始 CNN 学生参数
    }

    // 20260322 ZJH 重写 parameters() — 返回教师+学生所有参数
    // 20260402 ZJH 根据骨干类型收集对应的参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        if (m_bUsePretrainedBackbone) {
            // 20260402 ZJH ResNet18 骨干: 教师 + 学生
            auto vecTeacher = m_pResNetTeacher->parameters();
            vecResult.insert(vecResult.end(), vecTeacher.begin(), vecTeacher.end());
            auto vecStudent = m_pResNetStudent->parameters();
            vecResult.insert(vecResult.end(), vecStudent.begin(), vecStudent.end());
        } else {
            // 20260322 ZJH 原始 CNN: 教师 + 学生
            auto vecTeacher = m_pTeacher->parameters();
            vecResult.insert(vecResult.end(), vecTeacher.begin(), vecTeacher.end());
            auto vecStudent = m_pStudent->parameters();
            vecResult.insert(vecResult.end(), vecStudent.begin(), vecStudent.end());
        }
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    // 20260402 ZJH 根据骨干类型收集对应的命名参数
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        if (m_bUsePretrainedBackbone) {
            appendVec(m_pResNetTeacher->namedParameters(makeP("teacher")));
            appendVec(m_pResNetStudent->namedParameters(makeP("student")));
        } else {
            appendVec(m_pTeacher->namedParameters(makeP("teacher")));
            appendVec(m_pStudent->namedParameters(makeP("student")));
        }
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集教师+学生 BN running stats + 阈值统计量
    // 20260402 ZJH 根据骨干类型收集对应的缓冲区
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        if (m_bUsePretrainedBackbone) {
            auto vecTeacher = m_pResNetTeacher->buffers();
            vecResult.insert(vecResult.end(), vecTeacher.begin(), vecTeacher.end());
            auto vecStudent = m_pResNetStudent->buffers();
            vecResult.insert(vecResult.end(), vecStudent.begin(), vecStudent.end());
        } else {
            auto vecTeacher = m_pTeacher->buffers();  // 20260328 ZJH 教师网络 BN 缓冲区
            vecResult.insert(vecResult.end(), vecTeacher.begin(), vecTeacher.end());
            auto vecStudent = m_pStudent->buffers();  // 20260328 ZJH 学生网络 BN 缓冲区
            vecResult.insert(vecResult.end(), vecStudent.begin(), vecStudent.end());
        }
        // 20260328 ZJH 异常阈值统计量（校准后序列化保存）
        vecResult.push_back(&m_tScoreMean);
        vecResult.push_back(&m_tScoreStd);
        vecResult.push_back(&m_tThreshold);
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集教师+学生 BN 命名缓冲区 + 阈值统计量
    // 20260402 ZJH 根据骨干类型收集对应的命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        if (m_bUsePretrainedBackbone) {
            appendVec(m_pResNetTeacher->namedBuffers(makeP("teacher")));
            appendVec(m_pResNetStudent->namedBuffers(makeP("student")));
        } else {
            appendVec(m_pTeacher->namedBuffers(makeP("teacher")));
            appendVec(m_pStudent->namedBuffers(makeP("student")));
        }
        // 20260328 ZJH 阈值统计量作为命名缓冲区，序列化器自动保存/加载
        vecResult.push_back({makeP("anomaly_score_mean"), &m_tScoreMean});
        vecResult.push_back({makeP("anomaly_score_std"), &m_tScoreStd});
        vecResult.push_back({makeP("anomaly_threshold"), &m_tThreshold});
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    // 20260402 ZJH 根据骨干类型传播训练模式
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        if (m_bUsePretrainedBackbone) {
            m_pResNetTeacher->train(bMode);  // 20260402 ZJH ResNet18 教师
            m_pResNetStudent->train(bMode);  // 20260402 ZJH ResNet18 学生
        } else {
            m_pTeacher->train(bMode);  // 20260322 ZJH 原始 CNN 教师
            m_pStudent->train(bMode);  // 20260322 ZJH 原始 CNN 学生
        }
    }

    // 20260402 ZJH nnUpsample — 最近邻上采样（EfficientAD 内部使用，避免前向引用 FPN 类）
    // 20260406 ZJH input: [N, C, H, W] 输入张量
    // 20260406 ZJH nTH, nTW: 目标高度和宽度
    // 20260406 ZJH 返回: [N, C, nTH, nTW] 上采样后的张量
    static Tensor nnUpsample(const Tensor& input, int nTH, int nTW) {
        auto cI = input.contiguous();  // 20260406 ZJH 确保输入内存连续
        int nN = cI.shape(0), nC = cI.shape(1), nH = cI.shape(2), nW = cI.shape(3);  // 20260406 ZJH 源张量维度
        auto r = Tensor::zeros({nN, nC, nTH, nTW});  // 20260406 ZJH 分配输出张量
        float* pO = r.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pI = cI.floatDataPtr();  // 20260406 ZJH 输入数据指针
        float fSH = static_cast<float>(nH) / static_cast<float>(nTH);  // 20260406 ZJH 高度缩放比
        float fSW = static_cast<float>(nW) / static_cast<float>(nTW);  // 20260406 ZJH 宽度缩放比
        // 20260406 ZJH 四重循环遍历: batch → channel → 目标高 → 目标宽
        for (int n = 0; n < nN; ++n)
            for (int c = 0; c < nC; ++c)
                for (int th = 0; th < nTH; ++th) {
                    int sh = std::min(static_cast<int>(th * fSH), nH - 1);  // 20260406 ZJH 映射源 y 坐标（截断到边界）
                    for (int tw = 0; tw < nTW; ++tw) {
                        int sw = std::min(static_cast<int>(tw * fSW), nW - 1);  // 20260406 ZJH 映射源 x 坐标
                        pO[((n*nC+c)*nTH+th)*nTW+tw] = pI[((n*nC+c)*nH+sh)*nW+sw];  // 20260406 ZJH 最近邻复制
                    }
                }
        return r;  // 20260406 ZJH 返回上采样结果
    }

    // 20260402 ZJH channelMean — 沿通道维求均值 [N,C,H,W] → [N,1,H,W]
    // 20260406 ZJH 用于将多通道 MSE 差异图压缩为单通道异常分数图
    static Tensor channelMean(const Tensor& t) {
        auto ct = t.contiguous();  // 20260406 ZJH 确保连续内存
        int nN = ct.shape(0), nC = ct.shape(1), nH = ct.shape(2), nW = ct.shape(3);  // 20260406 ZJH 获取各维度大小
        int nS = nH * nW;  // 20260406 ZJH 空间元素数
        auto result = Tensor::zeros({nN, 1, nH, nW});  // 20260406 ZJH 分配单通道输出
        float* pO = result.mutableFloatDataPtr();  // 20260406 ZJH 输出数据指针
        const float* pI = ct.floatDataPtr();  // 20260406 ZJH 输入数据指针
        float fInvC = 1.0f / static_cast<float>(nC);  // 20260406 ZJH 通道均值缩放因子
        // 20260406 ZJH 三重循环: batch → 高 → 宽，内层遍历通道求和后取均值
        for (int n = 0; n < nN; ++n)
            for (int h = 0; h < nH; ++h)
                for (int w = 0; w < nW; ++w) {
                    float fSum = 0.0f;  // 20260406 ZJH 通道维累加器
                    for (int c = 0; c < nC; ++c) fSum += pI[((n*nC+c)*nH+h)*nW+w];  // 20260406 ZJH 累加所有通道
                    pO[(n*nH+h)*nW+w] = fSum * fInvC;  // 20260406 ZJH 写入通道均值
                }
        return result;  // 20260406 ZJH 返回 [N, 1, H, W] 通道均值结果
    }

private:
    // 20260402 ZJH 原始 4 层 CNN 骨干（bUsePretrainedBackbone=false 时使用）
    std::shared_ptr<EfficientADBackbone> m_pTeacher;  // 20260322 ZJH 教师网络（原始 CNN）
    std::shared_ptr<EfficientADBackbone> m_pStudent;  // 20260322 ZJH 学生网络（原始 CNN）

    // 20260402 ZJH ResNet18 骨干（bUsePretrainedBackbone=true 时使用）
    std::shared_ptr<ResNet18Backbone> m_pResNetTeacher;  // 20260402 ZJH 教师 ResNet18（预训练+冻结）
    std::shared_ptr<ResNet18Backbone> m_pResNetStudent;  // 20260402 ZJH 学生 ResNet18（随机初始化）

    bool m_bUsePretrainedBackbone = true;  // 20260402 ZJH 骨干选择标志

    // 20260328 ZJH 异常阈值校准统计量（存为 Tensor buffer，序列化器自动保存/加载）
    Tensor m_tScoreMean;   // 20260328 ZJH 正常样本异常分数均值 [1]
    Tensor m_tScoreStd;    // 20260328 ZJH 正常样本异常分数标准差 [1]
    Tensor m_tThreshold;   // 20260328 ZJH 异常判定阈值 = mean + k*std [1]
    bool m_bCalibrated = false;  // 20260328 ZJH 当前会话是否已校准（非持久化）
};

// =============================================================================
// 20260402 ZJH MultiScaleAnomalyFPN — 多尺度异常检测特征金字塔融合
// 将三级特征（H/4, H/8, H/16）上采样到统一尺寸 → 通道拼接 → 1x1 conv 融合
// 输出 H/4 分辨率异常热力图（相比原始 H/8 提升 4x 像素密度）
// 对标 Halcon 金字塔特征异常检测 — 小缺陷检出率显著提升
// =============================================================================
class MultiScaleAnomalyFPN : public Module {
public:
    // 20260402 ZJH 构造函数
    // 输入: feat2(64ch) + feat3(128ch) + feat4(256ch) = 448 通道拼接
    // 输出: 1 通道异常热力图
    MultiScaleAnomalyFPN()
        : m_convFuse(448, 128, 1, 1, 0, false),  // 20260402 ZJH 1x1 通道压缩 448→128
          m_bnFuse(128),
          m_convOut(128, 1, 1, 1, 0, true)        // 20260402 ZJH 1x1 输出层 128→1
    {
        // 20260402 ZJH 子模块为值成员，通过 parameters()/train() 手动管理（非 registerModule）
    }

    // 20260402 ZJH forward — 融合三级 Teacher-Student MSE 差异图
    // diffMaps: {diff2[N,64,H/4,W/4], diff3[N,128,H/8,W/8], diff4[N,256,H/16,W/16]}
    // 返回: [N, 1, H/4, W/4] 多尺度融合异常热力图
    Tensor forward(const Tensor& /*unused*/) override { return Tensor::zeros({1}); }

    // 20260402 ZJH forwardFPN — 实际融合入口
    Tensor forwardFPN(const Tensor& diff2, const Tensor& diff3, const Tensor& diff4) {
        int nBatch = diff2.shape(0);
        int nTargetH = diff2.shape(2);  // 20260402 ZJH H/4
        int nTargetW = diff2.shape(3);  // 20260402 ZJH W/4

        // 20260402 ZJH 上采样 diff3(H/8) → H/4（最近邻插值）
        auto up3 = upsampleNearest(diff3, nTargetH, nTargetW);  // 20260402 ZJH [N,128,H/4,W/4]
        // 20260402 ZJH 上采样 diff4(H/16) → H/4
        auto up4 = upsampleNearest(diff4, nTargetH, nTargetW);  // 20260402 ZJH [N,256,H/4,W/4]

        // 20260402 ZJH 通道拼接: [N, 64+128+256, H/4, W/4] = [N, 448, H/4, W/4]
        auto fused = catChannels3(diff2, up3, up4, nBatch, nTargetH, nTargetW);

        // 20260402 ZJH 1x1 Conv → BN → ReLU → 1x1 Conv → 输出
        fused = ReLU().forward(m_bnFuse.forward(m_convFuse.forward(fused)));  // 20260402 ZJH [N,128,H/4,W/4]
        auto anomalyMap = m_convOut.forward(fused);  // 20260402 ZJH [N,1,H/4,W/4]

        // 20260402 ZJH ReLU 确保异常分数非负
        return ReLU().forward(anomalyMap);
    }

    // 20260402 ZJH parameters/train 覆盖
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;
        auto a = m_convFuse.parameters(); v.insert(v.end(), a.begin(), a.end());
        auto b = m_bnFuse.parameters();   v.insert(v.end(), b.begin(), b.end());
        auto c = m_convOut.parameters();  v.insert(v.end(), c.begin(), c.end());
        return v;
    }

    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> v;
        auto makeP = [&](const std::string& s) { return strPrefix.empty() ? s : strPrefix + "." + s; };
        auto a = m_convFuse.namedParameters(makeP("fuse_conv")); v.insert(v.end(), a.begin(), a.end());
        auto b = m_bnFuse.namedParameters(makeP("fuse_bn"));     v.insert(v.end(), b.begin(), b.end());
        auto c = m_convOut.namedParameters(makeP("out_conv"));   v.insert(v.end(), c.begin(), c.end());
        return v;
    }

    std::vector<Tensor*> buffers() override {
        return m_bnFuse.buffers();
    }
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        auto makeP = [&](const std::string& s) { return strPrefix.empty() ? s : strPrefix + "." + s; };
        return m_bnFuse.namedBuffers(makeP("fuse_bn"));
    }

    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_convFuse.train(bMode); m_bnFuse.train(bMode); m_convOut.train(bMode);
    }

private:
    Conv2d m_convFuse;     // 20260402 ZJH 1x1 融合卷积
    BatchNorm2d m_bnFuse;  // 20260402 ZJH 融合 BN
    Conv2d m_convOut;      // 20260402 ZJH 1x1 输出卷积

    // 20260402 ZJH upsampleNearest — 最近邻上采样
    static Tensor upsampleNearest(const Tensor& input, int nTargetH, int nTargetW) {
        auto cIn = input.contiguous();
        int nN = cIn.shape(0), nC = cIn.shape(1), nH = cIn.shape(2), nW = cIn.shape(3);
        auto result = Tensor::zeros({nN, nC, nTargetH, nTargetW});
        float* pOut = result.mutableFloatDataPtr();
        const float* pIn = cIn.floatDataPtr();
        float fScaleH = static_cast<float>(nH) / static_cast<float>(nTargetH);
        float fScaleW = static_cast<float>(nW) / static_cast<float>(nTargetW);
        for (int n = 0; n < nN; ++n) {
            for (int c = 0; c < nC; ++c) {
                for (int th = 0; th < nTargetH; ++th) {
                    int nSrcH = std::min(static_cast<int>(th * fScaleH), nH - 1);
                    for (int tw = 0; tw < nTargetW; ++tw) {
                        int nSrcW = std::min(static_cast<int>(tw * fScaleW), nW - 1);
                        pOut[((n * nC + c) * nTargetH + th) * nTargetW + tw] =
                            pIn[((n * nC + c) * nH + nSrcH) * nW + nSrcW];
                    }
                }
            }
        }
        return result;
    }

    // 20260402 ZJH catChannels3 — 三张量通道拼接
    static Tensor catChannels3(const Tensor& a, const Tensor& b, const Tensor& c,
                                int nBatch, int nH, int nW) {
        auto ca = a.contiguous(), cb = b.contiguous(), cc = c.contiguous();
        int nCa = ca.shape(1), nCb = cb.shape(1), nCc = cc.shape(1);
        int nCtotal = nCa + nCb + nCc;
        int nSpatial = nH * nW;
        auto result = Tensor::zeros({nBatch, nCtotal, nH, nW});
        float* pOut = result.mutableFloatDataPtr();
        const float* pA = ca.floatDataPtr();
        const float* pB = cb.floatDataPtr();
        const float* pC = cc.floatDataPtr();
        for (int n = 0; n < nBatch; ++n) {
            // 20260402 ZJH 拷贝 A 通道
            for (int ch = 0; ch < nCa; ++ch)
                for (int i = 0; i < nSpatial; ++i)
                    pOut[((n * nCtotal + ch) * nSpatial) + i] = pA[((n * nCa + ch) * nSpatial) + i];
            // 20260402 ZJH 拷贝 B 通道
            for (int ch = 0; ch < nCb; ++ch)
                for (int i = 0; i < nSpatial; ++i)
                    pOut[((n * nCtotal + nCa + ch) * nSpatial) + i] = pB[((n * nCb + ch) * nSpatial) + i];
            // 20260402 ZJH 拷贝 C 通道
            for (int ch = 0; ch < nCc; ++ch)
                for (int i = 0; i < nSpatial; ++i)
                    pOut[((n * nCtotal + nCa + nCb + ch) * nSpatial) + i] = pC[((n * nCc + ch) * nSpatial) + i];
        }
        return result;
    }
};

}  // 20260406 ZJH namespace om 结束
