// 20260320 ZJH YOLOv5-nano 简化目标检测网络模块 — Phase 3
// 单尺度检测的简化 YOLO 网络
// 骨干: Conv(stem) -> [CSPBlock + 下采样] x4
// 检测头: 卷积输出 [N, H*W*nAnchors, 5+nClasses]
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>

export module df.engine.yolo;

// 20260320 ZJH 导入依赖模块
import df.engine.tensor;
import df.engine.tensor_ops;
import df.engine.module;
import df.engine.conv;
import df.engine.activations;

export namespace df {

// 20260320 ZJH CSPBlock — Cross Stage Partial 块（简化版）
// 将输入分为两部分：一部分直通，一部分经过卷积变换，最后拼接
// 简化实现：conv1 -> bn1 -> act -> conv2 -> bn2 -> act -> 与输入相加（残差连接）
class CSPBlock : public Module {
public:
    // 20260320 ZJH 构造函数
    // nChannels: 输入和输出通道数（保持不变）
    CSPBlock(int nChannels)
        : m_conv1(nChannels, nChannels, 3, 1, 1, false),   // 20260320 ZJH 第一层 3x3 卷积
          m_conv2(nChannels, nChannels, 3, 1, 1, false),   // 20260320 ZJH 第二层 3x3 卷积
          m_bn1(nChannels),                                 // 20260320 ZJH 第一层 BN
          m_bn2(nChannels),                                 // 20260320 ZJH 第二层 BN
          m_act(0.1f)                                       // 20260320 ZJH LeakyReLU 斜率 0.1
    {}

    // 20260320 ZJH forward — CSP 块前向传播
    // input: [N, C, H, W]
    // 返回: [N, C, H, W]（残差连接，维度不变）
    Tensor forward(const Tensor& input) override {
        auto x = m_conv1.forward(input);  // 20260320 ZJH 第一层卷积
        x = m_bn1.forward(x);             // 20260320 ZJH 第一层 BN
        x = m_act.forward(x);             // 20260320 ZJH LeakyReLU
        x = m_conv2.forward(x);           // 20260320 ZJH 第二层卷积
        x = m_bn2.forward(x);             // 20260320 ZJH 第二层 BN
        x = m_act.forward(x);             // 20260320 ZJH LeakyReLU
        // 20260320 ZJH 残差连接：output = x + input
        x = tensorAdd(x, input);
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_conv1.parameters());
        append(m_bn1.parameters());
        append(m_conv2.parameters());
        append(m_bn2.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);
        m_bn1.train(bMode);
        m_conv2.train(bMode);
        m_bn2.train(bMode);
    }

private:
    Conv2d m_conv1, m_conv2;   // 20260320 ZJH 两层卷积
    BatchNorm2d m_bn1, m_bn2;  // 20260320 ZJH 两层 BN
    LeakyReLU m_act;           // 20260320 ZJH LeakyReLU 激活
};

// 20260320 ZJH YOLOHead — YOLO 检测头
// 将特征图转换为检测输出
// 输入: [N, C, H, W] 特征图
// 输出: [N, H*W*nAnchors, 5+nClasses]（cx, cy, w, h, conf, class_probs）
class YOLOHead : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数
    // nAnchors: 每个位置的 anchor 数量
    // nClasses: 类别数
    YOLOHead(int nInChannels, int nAnchors, int nClasses)
        : m_nAnchors(nAnchors), m_nClasses(nClasses),
          m_conv1(nInChannels, nInChannels, 3, 1, 1, true),   // 20260320 ZJH 3x3 特征提取
          m_conv2(nInChannels, nAnchors * (5 + nClasses), 1, 1, 0, true),  // 20260320 ZJH 1x1 预测
          m_bn1(nInChannels),                                   // 20260320 ZJH BN
          m_act(0.1f)                                           // 20260320 ZJH LeakyReLU
    {}

    // 20260320 ZJH forward — 检测头前向传播
    // input: [N, C, H, W]
    // 返回: [N, H*W*nAnchors, 5+nClasses]
    Tensor forward(const Tensor& input) override {
        auto x = m_conv1.forward(input);   // 20260320 ZJH 3x3 卷积特征提取
        x = m_bn1.forward(x);              // 20260320 ZJH BN
        x = m_act.forward(x);              // 20260320 ZJH LeakyReLU
        x = m_conv2.forward(x);            // 20260320 ZJH 1x1 预测层: [N, nAnchors*(5+nClasses), H, W]

        // 20260320 ZJH 重塑输出为 [N, H*W*nAnchors, 5+nClasses]
        int nBatch = x.shape(0);            // 20260320 ZJH 批次大小
        int nH = x.shape(2);               // 20260320 ZJH 特征图高度
        int nW = x.shape(3);               // 20260320 ZJH 特征图宽度
        int nPredPerAnchor = 5 + m_nClasses;  // 20260320 ZJH 每个 anchor 的预测数
        int nTotalPreds = nH * nW * m_nAnchors;  // 20260320 ZJH 总预测数

        // 20260320 ZJH reshape: [N, nAnchors*(5+nClasses), H, W] -> [N, nTotalPreds, 5+nClasses]
        auto cx = x.contiguous();
        auto output = Tensor::fromData(cx.floatDataPtr(), {nBatch, nTotalPreds, nPredPerAnchor});
        return output;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_conv1.parameters());
        append(m_bn1.parameters());
        append(m_conv2.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);
        m_bn1.train(bMode);
        m_conv2.train(bMode);
    }

private:
    int m_nAnchors;    // 20260320 ZJH 每个位置的 anchor 数
    int m_nClasses;    // 20260320 ZJH 类别数
    Conv2d m_conv1;    // 20260320 ZJH 3x3 特征提取卷积
    Conv2d m_conv2;    // 20260320 ZJH 1x1 预测卷积
    BatchNorm2d m_bn1; // 20260320 ZJH BN
    LeakyReLU m_act;   // 20260320 ZJH LeakyReLU
};

// 20260320 ZJH YOLOv5Nano — 简化单尺度 YOLOv5-nano 检测网络
// 骨干: stem(3x3) -> [down(3x3,stride=2) + CSP] x4
// 头: YOLOHead 输出检测结果
// 输入: [N, 3, H, W]（H, W 应为 32 的倍数）
// 输出: [N, numPredictions, 5+nClasses]
class YOLOv5Nano : public Module {
public:
    // 20260320 ZJH 构造函数
    // nNumClasses: 检测类别数，默认 20（VOC）
    // nInChannels: 输入通道数，默认 3（RGB）
    YOLOv5Nano(int nNumClasses = 20, int nInChannels = 3)
        : m_stem(nInChannels, 16, 3, 1, 1, false),   // 20260320 ZJH 初始 3x3 卷积: nIn->16
          m_stemBn(16),                                // 20260320 ZJH stem BN
          m_act(0.1f),                                 // 20260320 ZJH LeakyReLU
          m_down1(16, 32, 3, 2, 1, false),            // 20260320 ZJH 下采样 1: 16->32, stride=2
          m_bn1(32),                                   // 20260320 ZJH BN1
          m_csp1(32),                                  // 20260320 ZJH CSP 块 1
          m_down2(32, 64, 3, 2, 1, false),            // 20260320 ZJH 下采样 2: 32->64, stride=2
          m_bn2(64),                                   // 20260320 ZJH BN2
          m_csp2(64),                                  // 20260320 ZJH CSP 块 2
          m_down3(64, 128, 3, 2, 1, false),           // 20260320 ZJH 下采样 3: 64->128, stride=2
          m_bn3(128),                                  // 20260320 ZJH BN3
          m_csp3(128),                                 // 20260320 ZJH CSP 块 3
          m_down4(128, 256, 3, 2, 1, false),          // 20260320 ZJH 下采样 4: 128->256, stride=2
          m_bn4(256),                                  // 20260320 ZJH BN4
          m_csp4(256),                                 // 20260320 ZJH CSP 块 4
          m_head(256, 3, nNumClasses)                  // 20260320 ZJH 检测头: 3 个 anchor, nClasses 个类别
    {}

    // 20260320 ZJH forward — YOLOv5Nano 前向传播
    // input: [N, nInChannels, H, W]
    // 返回: [N, numPredictions, 5+nClasses]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH Stem: [N,3,H,W] -> [N,16,H,W]
        auto x = m_stem.forward(input);
        x = m_stemBn.forward(x);
        x = m_act.forward(x);

        // 20260320 ZJH Stage 1: [N,16,H,W] -> [N,32,H/2,W/2]
        x = m_down1.forward(x);
        x = m_bn1.forward(x);
        x = m_act.forward(x);
        x = m_csp1.forward(x);

        // 20260320 ZJH Stage 2: [N,32,H/2,W/2] -> [N,64,H/4,W/4]
        x = m_down2.forward(x);
        x = m_bn2.forward(x);
        x = m_act.forward(x);
        x = m_csp2.forward(x);

        // 20260320 ZJH Stage 3: [N,64,H/4,W/4] -> [N,128,H/8,W/8]
        x = m_down3.forward(x);
        x = m_bn3.forward(x);
        x = m_act.forward(x);
        x = m_csp3.forward(x);

        // 20260320 ZJH Stage 4: [N,128,H/8,W/8] -> [N,256,H/16,W/16]
        x = m_down4.forward(x);
        x = m_bn4.forward(x);
        x = m_act.forward(x);
        x = m_csp4.forward(x);

        // 20260320 ZJH 检测头: [N,256,H/16,W/16] -> [N, (H/16)*(W/16)*3, 5+nClasses]
        x = m_head.forward(x);

        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_stem.parameters());
        append(m_stemBn.parameters());
        append(m_down1.parameters());
        append(m_bn1.parameters());
        append(m_csp1.parameters());
        append(m_down2.parameters());
        append(m_bn2.parameters());
        append(m_csp2.parameters());
        append(m_down3.parameters());
        append(m_bn3.parameters());
        append(m_csp3.parameters());
        append(m_down4.parameters());
        append(m_bn4.parameters());
        append(m_csp4.parameters());
        append(m_head.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_stem.train(bMode);
        m_stemBn.train(bMode);
        m_down1.train(bMode); m_bn1.train(bMode); m_csp1.train(bMode);
        m_down2.train(bMode); m_bn2.train(bMode); m_csp2.train(bMode);
        m_down3.train(bMode); m_bn3.train(bMode); m_csp3.train(bMode);
        m_down4.train(bMode); m_bn4.train(bMode); m_csp4.train(bMode);
        m_head.train(bMode);
    }

private:
    Conv2d m_stem;        // 20260320 ZJH Stem 卷积
    BatchNorm2d m_stemBn; // 20260320 ZJH Stem BN
    LeakyReLU m_act;      // 20260320 ZJH 共用 LeakyReLU 激活

    // 20260320 ZJH 骨干网络的 4 个阶段
    Conv2d m_down1, m_down2, m_down3, m_down4;  // 20260320 ZJH 下采样卷积
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;     // 20260320 ZJH BN
    CSPBlock m_csp1, m_csp2, m_csp3, m_csp4;    // 20260320 ZJH CSP 块

    YOLOHead m_head;  // 20260320 ZJH 检测头
};

}  // namespace df
