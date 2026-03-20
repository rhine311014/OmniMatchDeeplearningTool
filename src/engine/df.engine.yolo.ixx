// 20260320 ZJH YOLOv5/v7/v8/v10 目标检测网络模块 — Phase 3 + Phase 5
// 支持多种 YOLO 变体：YOLOv5-Nano, YOLOv7-Tiny, YOLOv8-Nano, YOLOv10-Nano
// 骨干: Conv(stem) -> [特征提取块 + 下采样] x N
// 检测头: 锚点式 (v5/v7) 或解耦式 (v8/v10)
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

// ============================================================================
// 20260320 ZJH C2fBlock — YOLOv8 核心构建块（取代 CSP）
// 将特征图分为两半，一半直通，一半经过 Bottleneck，最后拼接
// 简化实现：reduce(1x1) -> split -> bottleneck(3x3+3x3) -> concat -> expand(1x1)
// 为保持输入输出通道一致，使用残差连接近似 C2f 行为
// ============================================================================
class C2fBlock : public Module {
public:
    // 20260320 ZJH 构造函数
    // nChannels: 输入和输出通道数（保持不变）
    C2fBlock(int nChannels)
        : m_convReduce(nChannels, nChannels / 2, 1, 1, 0, false),   // 20260320 ZJH 1x1 通道减半
          m_convExpand(nChannels / 2, nChannels, 1, 1, 0, false),    // 20260320 ZJH 1x1 通道恢复
          m_bnReduce(nChannels / 2),                                  // 20260320 ZJH Reduce BN
          m_bnExpand(nChannels),                                      // 20260320 ZJH Expand BN
          m_bottleneck1(nChannels / 2, nChannels / 2, 3, 1, 1, false),  // 20260320 ZJH Bottleneck 第一个 3x3
          m_bottleneck2(nChannels / 2, nChannels / 2, 3, 1, 1, false),  // 20260320 ZJH Bottleneck 第二个 3x3
          m_bnBot1(nChannels / 2),                                    // 20260320 ZJH Bottleneck BN1
          m_bnBot2(nChannels / 2),                                    // 20260320 ZJH Bottleneck BN2
          m_act(0.1f),                                                // 20260320 ZJH LeakyReLU 斜率 0.1
          m_nChannels(nChannels)                                      // 20260320 ZJH 保存通道数
    {}

    // 20260320 ZJH forward — C2f 块前向传播
    // input: [N, C, H, W]
    // 返回: [N, C, H, W]（通道数不变，残差连接）
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH 通道减半
        auto x = m_convReduce.forward(input);  // 20260320 ZJH [N,C,H,W] -> [N,C/2,H,W]
        x = m_bnReduce.forward(x);             // 20260320 ZJH BN
        x = m_act.forward(x);                  // 20260320 ZJH LeakyReLU
        // 20260320 ZJH Bottleneck: 两个 3x3 卷积
        x = m_bottleneck1.forward(x);          // 20260320 ZJH 3x3 卷积 1
        x = m_bnBot1.forward(x);               // 20260320 ZJH BN
        x = m_act.forward(x);                  // 20260320 ZJH LeakyReLU
        x = m_bottleneck2.forward(x);          // 20260320 ZJH 3x3 卷积 2
        x = m_bnBot2.forward(x);               // 20260320 ZJH BN
        x = m_act.forward(x);                  // 20260320 ZJH LeakyReLU
        // 20260320 ZJH 通道恢复
        x = m_convExpand.forward(x);           // 20260320 ZJH [N,C/2,H,W] -> [N,C,H,W]
        x = m_bnExpand.forward(x);             // 20260320 ZJH BN
        x = m_act.forward(x);                  // 20260320 ZJH LeakyReLU
        // 20260320 ZJH 残差连接
        x = tensorAdd(x, input);               // 20260320 ZJH output = x + input
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_convReduce.parameters());
        append(m_bnReduce.parameters());
        append(m_bottleneck1.parameters());
        append(m_bnBot1.parameters());
        append(m_bottleneck2.parameters());
        append(m_bnBot2.parameters());
        append(m_convExpand.parameters());
        append(m_bnExpand.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_convReduce.train(bMode); m_bnReduce.train(bMode);
        m_bottleneck1.train(bMode); m_bnBot1.train(bMode);
        m_bottleneck2.train(bMode); m_bnBot2.train(bMode);
        m_convExpand.train(bMode); m_bnExpand.train(bMode);
    }

private:
    Conv2d m_convReduce;     // 20260320 ZJH 1x1 通道减半
    Conv2d m_convExpand;     // 20260320 ZJH 1x1 通道恢复
    BatchNorm2d m_bnReduce, m_bnExpand;  // 20260320 ZJH Reduce/Expand BN
    Conv2d m_bottleneck1, m_bottleneck2; // 20260320 ZJH Bottleneck 3x3 卷积
    BatchNorm2d m_bnBot1, m_bnBot2;     // 20260320 ZJH Bottleneck BN
    LeakyReLU m_act;         // 20260320 ZJH LeakyReLU
    int m_nChannels;         // 20260320 ZJH 通道数
};

// ============================================================================
// 20260320 ZJH YOLOv8Head — YOLOv8 解耦检测头
// 分类和回归分支独立处理，无 anchor（anchor-free）
// 输入: [N, C, H, W]
// 输出: [N, H*W, 4+nClasses]（x, y, w, h, class_probs）
// ============================================================================
class YOLOv8Head : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数
    // nClasses: 类别数
    YOLOv8Head(int nInChannels, int nClasses)
        : m_nClasses(nClasses),
          // 20260320 ZJH 分类分支：两层 3x3 卷积 + 1x1 输出
          m_clsConv1(nInChannels, nInChannels, 3, 1, 1, false),   // 20260320 ZJH 分类特征提取 1
          m_clsConv2(nInChannels, nInChannels, 3, 1, 1, false),   // 20260320 ZJH 分类特征提取 2
          m_clsBn1(nInChannels), m_clsBn2(nInChannels),           // 20260320 ZJH 分类 BN
          m_clsOut(nInChannels, nClasses, 1, 1, 0, true),         // 20260320 ZJH 分类输出: nClasses 通道
          // 20260320 ZJH 回归分支：两层 3x3 卷积 + 1x1 输出
          m_regConv1(nInChannels, nInChannels, 3, 1, 1, false),   // 20260320 ZJH 回归特征提取 1
          m_regConv2(nInChannels, nInChannels, 3, 1, 1, false),   // 20260320 ZJH 回归特征提取 2
          m_regBn1(nInChannels), m_regBn2(nInChannels),           // 20260320 ZJH 回归 BN
          m_regOut(nInChannels, 4, 1, 1, 0, true),                // 20260320 ZJH 回归输出: 4 通道 (x,y,w,h)
          m_act(0.1f)                                              // 20260320 ZJH LeakyReLU
    {}

    // 20260320 ZJH forward — 解耦检测头前向传播
    // input: [N, C, H, W]
    // 返回: [N, H*W, 4+nClasses]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH 分类分支
        auto cls = m_clsConv1.forward(input);  // 20260320 ZJH [N,C,H,W]
        cls = m_clsBn1.forward(cls);           // 20260320 ZJH BN
        cls = m_act.forward(cls);              // 20260320 ZJH LeakyReLU
        cls = m_clsConv2.forward(cls);         // 20260320 ZJH [N,C,H,W]
        cls = m_clsBn2.forward(cls);           // 20260320 ZJH BN
        cls = m_act.forward(cls);              // 20260320 ZJH LeakyReLU
        cls = m_clsOut.forward(cls);           // 20260320 ZJH [N,nClasses,H,W]

        // 20260320 ZJH 回归分支
        auto reg = m_regConv1.forward(input);  // 20260320 ZJH [N,C,H,W]
        reg = m_regBn1.forward(reg);           // 20260320 ZJH BN
        reg = m_act.forward(reg);              // 20260320 ZJH LeakyReLU
        reg = m_regConv2.forward(reg);         // 20260320 ZJH [N,C,H,W]
        reg = m_regBn2.forward(reg);           // 20260320 ZJH BN
        reg = m_act.forward(reg);              // 20260320 ZJH LeakyReLU
        reg = m_regOut.forward(reg);           // 20260320 ZJH [N,4,H,W]

        // 20260320 ZJH 合并分类+回归输出并重塑
        int nBatch = input.shape(0);           // 20260320 ZJH 批次大小
        int nH = input.shape(2);               // 20260320 ZJH 特征图高度
        int nW = input.shape(3);               // 20260320 ZJH 特征图宽度
        int nSpatial = nH * nW;                // 20260320 ZJH 空间位置总数
        int nPredDim = 4 + m_nClasses;         // 20260320 ZJH 每个位置的预测维度

        // 20260320 ZJH 手动拼接 reg[N,4,H,W] 和 cls[N,nClasses,H,W] -> output[N,H*W,4+nClasses]
        auto regC = reg.contiguous();          // 20260320 ZJH 确保连续存储
        auto clsC = cls.contiguous();          // 20260320 ZJH 确保连续存储
        auto output = Tensor::zeros({nBatch, nSpatial, nPredDim});  // 20260320 ZJH 分配输出
        float* pOut = output.mutableFloatDataPtr();   // 20260320 ZJH 输出指针
        const float* pReg = regC.floatDataPtr();      // 20260320 ZJH 回归数据指针
        const float* pCls = clsC.floatDataPtr();      // 20260320 ZJH 分类数据指针

        // 20260320 ZJH 逐批次逐位置拷贝回归+分类数据
        for (int n = 0; n < nBatch; ++n) {
            for (int s = 0; s < nSpatial; ++s) {
                int nOutIdx = n * nSpatial * nPredDim + s * nPredDim;  // 20260320 ZJH 输出偏移
                // 20260320 ZJH 拷贝 4 个回归值 (x,y,w,h)
                for (int c = 0; c < 4; ++c) {
                    pOut[nOutIdx + c] = pReg[n * 4 * nSpatial + c * nSpatial + s];
                }
                // 20260320 ZJH 拷贝 nClasses 个分类值
                for (int c = 0; c < m_nClasses; ++c) {
                    pOut[nOutIdx + 4 + c] = pCls[n * m_nClasses * nSpatial + c * nSpatial + s];
                }
            }
        }
        return output;  // 20260320 ZJH [N, H*W, 4+nClasses]
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_clsConv1.parameters()); append(m_clsBn1.parameters());
        append(m_clsConv2.parameters()); append(m_clsBn2.parameters());
        append(m_clsOut.parameters());
        append(m_regConv1.parameters()); append(m_regBn1.parameters());
        append(m_regConv2.parameters()); append(m_regBn2.parameters());
        append(m_regOut.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_clsConv1.train(bMode); m_clsBn1.train(bMode);
        m_clsConv2.train(bMode); m_clsBn2.train(bMode);
        m_clsOut.train(bMode);
        m_regConv1.train(bMode); m_regBn1.train(bMode);
        m_regConv2.train(bMode); m_regBn2.train(bMode);
        m_regOut.train(bMode);
    }

private:
    int m_nClasses;              // 20260320 ZJH 类别数
    // 20260320 ZJH 分类分支
    Conv2d m_clsConv1, m_clsConv2;     // 20260320 ZJH 分类 3x3 卷积
    BatchNorm2d m_clsBn1, m_clsBn2;    // 20260320 ZJH 分类 BN
    Conv2d m_clsOut;                    // 20260320 ZJH 分类 1x1 输出
    // 20260320 ZJH 回归分支
    Conv2d m_regConv1, m_regConv2;     // 20260320 ZJH 回归 3x3 卷积
    BatchNorm2d m_regBn1, m_regBn2;    // 20260320 ZJH 回归 BN
    Conv2d m_regOut;                    // 20260320 ZJH 回归 1x1 输出
    LeakyReLU m_act;                    // 20260320 ZJH LeakyReLU
};

// ============================================================================
// 20260320 ZJH YOLOv8Nano — YOLOv8-Nano 完整网络（anchor-free, 解耦头）
// 骨干: stem(3x3) -> [down(3x3,stride=2) + C2f] x4
// 头: YOLOv8Head 输出检测结果（无 anchor）
// 输入: [N, 3, H, W]（H, W 应为 32 的倍数）
// 输出: [N, (H/16)*(W/16), 4+nClasses]
// ============================================================================
class YOLOv8Nano : public Module {
public:
    // 20260320 ZJH 构造函数
    // nNumClasses: 检测类别数，默认 20
    // nInChannels: 输入通道数，默认 3（RGB）
    YOLOv8Nano(int nNumClasses = 20, int nInChannels = 3)
        : m_stem(nInChannels, 16, 3, 1, 1, false),   // 20260320 ZJH stem: nIn->16
          m_stemBn(16),                                // 20260320 ZJH stem BN
          m_act(0.1f),                                 // 20260320 ZJH LeakyReLU
          m_down1(16, 32, 3, 2, 1, false),             // 20260320 ZJH 下采样 1: 16->32
          m_bn1(32),                                   // 20260320 ZJH BN1
          m_c2f1(32),                                  // 20260320 ZJH C2f 块 1
          m_down2(32, 64, 3, 2, 1, false),             // 20260320 ZJH 下采样 2: 32->64
          m_bn2(64),                                   // 20260320 ZJH BN2
          m_c2f2(64),                                  // 20260320 ZJH C2f 块 2
          m_down3(64, 128, 3, 2, 1, false),            // 20260320 ZJH 下采样 3: 64->128
          m_bn3(128),                                  // 20260320 ZJH BN3
          m_c2f3(128),                                 // 20260320 ZJH C2f 块 3
          m_down4(128, 256, 3, 2, 1, false),           // 20260320 ZJH 下采样 4: 128->256
          m_bn4(256),                                  // 20260320 ZJH BN4
          m_c2f4(256),                                 // 20260320 ZJH C2f 块 4
          m_head(256, nNumClasses)                     // 20260320 ZJH 解耦检测头
    {}

    // 20260320 ZJH forward — YOLOv8Nano 前向传播
    // input: [N, nInChannels, H, W]
    // 返回: [N, (H/16)*(W/16), 4+nClasses]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH Stem: [N,3,H,W] -> [N,16,H,W]
        auto x = m_stem.forward(input);
        x = m_stemBn.forward(x);
        x = m_act.forward(x);

        // 20260320 ZJH Stage 1: [N,16,H,W] -> [N,32,H/2,W/2]
        x = m_down1.forward(x);  x = m_bn1.forward(x);  x = m_act.forward(x);
        x = m_c2f1.forward(x);

        // 20260320 ZJH Stage 2: [N,32,H/2,W/2] -> [N,64,H/4,W/4]
        x = m_down2.forward(x);  x = m_bn2.forward(x);  x = m_act.forward(x);
        x = m_c2f2.forward(x);

        // 20260320 ZJH Stage 3: [N,64,H/4,W/4] -> [N,128,H/8,W/8]
        x = m_down3.forward(x);  x = m_bn3.forward(x);  x = m_act.forward(x);
        x = m_c2f3.forward(x);

        // 20260320 ZJH Stage 4: [N,128,H/8,W/8] -> [N,256,H/16,W/16]
        x = m_down4.forward(x);  x = m_bn4.forward(x);  x = m_act.forward(x);
        x = m_c2f4.forward(x);

        // 20260320 ZJH 解耦检测头: [N,256,H/16,W/16] -> [N,(H/16)*(W/16),4+nClasses]
        x = m_head.forward(x);
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_stem.parameters()); append(m_stemBn.parameters());
        append(m_down1.parameters()); append(m_bn1.parameters()); append(m_c2f1.parameters());
        append(m_down2.parameters()); append(m_bn2.parameters()); append(m_c2f2.parameters());
        append(m_down3.parameters()); append(m_bn3.parameters()); append(m_c2f3.parameters());
        append(m_down4.parameters()); append(m_bn4.parameters()); append(m_c2f4.parameters());
        append(m_head.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_stem.train(bMode); m_stemBn.train(bMode);
        m_down1.train(bMode); m_bn1.train(bMode); m_c2f1.train(bMode);
        m_down2.train(bMode); m_bn2.train(bMode); m_c2f2.train(bMode);
        m_down3.train(bMode); m_bn3.train(bMode); m_c2f3.train(bMode);
        m_down4.train(bMode); m_bn4.train(bMode); m_c2f4.train(bMode);
        m_head.train(bMode);
    }

private:
    Conv2d m_stem;             // 20260320 ZJH Stem 卷积
    BatchNorm2d m_stemBn;      // 20260320 ZJH Stem BN
    LeakyReLU m_act;           // 20260320 ZJH 共用 LeakyReLU
    Conv2d m_down1, m_down2, m_down3, m_down4;   // 20260320 ZJH 下采样卷积
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;      // 20260320 ZJH BN
    C2fBlock m_c2f1, m_c2f2, m_c2f3, m_c2f4;     // 20260320 ZJH C2f 块
    YOLOv8Head m_head;         // 20260320 ZJH 解耦检测头
};

// ============================================================================
// 20260320 ZJH ELANBlock — YOLOv7 核心构建块（高效层聚合网络）
// 多分支特征提取：4 个卷积分支，输出拼接后降维
// 简化实现：conv1 + conv2(从conv1) + conv3(从conv2) + conv4(从conv3)
//           -> 各分支相加（近似拼接效果）-> convOut 降维
// ============================================================================
class ELANBlock : public Module {
public:
    // 20260320 ZJH 构造函数
    // nInChannels: 输入通道数
    // nOutChannels: 输出通道数
    ELANBlock(int nInChannels, int nOutChannels)
        : m_conv1(nInChannels, nInChannels, 1, 1, 0, false),     // 20260320 ZJH 分支1: 1x1 变换
          m_conv2(nInChannels, nInChannels, 3, 1, 1, false),     // 20260320 ZJH 分支2: 3x3 从分支1
          m_conv3(nInChannels, nInChannels, 3, 1, 1, false),     // 20260320 ZJH 分支3: 3x3 从分支2
          m_conv4(nInChannels, nInChannels, 3, 1, 1, false),     // 20260320 ZJH 分支4: 3x3 从分支3
          m_bn1(nInChannels), m_bn2(nInChannels),
          m_bn3(nInChannels), m_bn4(nInChannels),
          m_convOut(nInChannels, nOutChannels, 1, 1, 0, false),  // 20260320 ZJH 合并后 1x1 降维
          m_bnOut(nOutChannels),                                  // 20260320 ZJH 输出 BN
          m_act(0.1f)                                             // 20260320 ZJH LeakyReLU
    {}

    // 20260320 ZJH forward — ELAN 块前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, H, W]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH 分支 1: 1x1 变换
        auto b1 = m_conv1.forward(input);   b1 = m_bn1.forward(b1);   b1 = m_act.forward(b1);
        // 20260320 ZJH 分支 2: 从分支1经过 3x3
        auto b2 = m_conv2.forward(b1);      b2 = m_bn2.forward(b2);   b2 = m_act.forward(b2);
        // 20260320 ZJH 分支 3: 从分支2经过 3x3
        auto b3 = m_conv3.forward(b2);      b3 = m_bn3.forward(b3);   b3 = m_act.forward(b3);
        // 20260320 ZJH 分支 4: 从分支3经过 3x3
        auto b4 = m_conv4.forward(b3);      b4 = m_bn4.forward(b4);   b4 = m_act.forward(b4);
        // 20260320 ZJH 聚合：将所有分支相加（近似多分支拼接效果）
        auto agg = tensorAdd(tensorAdd(b1, b2), tensorAdd(b3, b4));
        // 20260320 ZJH 1x1 降维到输出通道数
        auto out = m_convOut.forward(agg);   out = m_bnOut.forward(out);   out = m_act.forward(out);
        return out;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_conv1.parameters()); append(m_bn1.parameters());
        append(m_conv2.parameters()); append(m_bn2.parameters());
        append(m_conv3.parameters()); append(m_bn3.parameters());
        append(m_conv4.parameters()); append(m_bn4.parameters());
        append(m_convOut.parameters()); append(m_bnOut.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode); m_bn1.train(bMode);
        m_conv2.train(bMode); m_bn2.train(bMode);
        m_conv3.train(bMode); m_bn3.train(bMode);
        m_conv4.train(bMode); m_bn4.train(bMode);
        m_convOut.train(bMode); m_bnOut.train(bMode);
    }

private:
    Conv2d m_conv1, m_conv2, m_conv3, m_conv4;   // 20260320 ZJH 4 个分支卷积
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;      // 20260320 ZJH 分支 BN
    Conv2d m_convOut;          // 20260320 ZJH 合并后降维
    BatchNorm2d m_bnOut;       // 20260320 ZJH 输出 BN
    LeakyReLU m_act;           // 20260320 ZJH LeakyReLU
};

// ============================================================================
// 20260320 ZJH YOLOv7Tiny — YOLOv7-Tiny 检测网络（ELAN + anchor-based）
// 骨干: stem(3x3) -> [down(3x3,stride=2) + ELAN] x3
// 头: YOLOHead（复用 anchor-based 检测头）
// 输入: [N, 3, H, W]（H, W 应为 16 的倍数）
// 输出: [N, (H/8)*(W/8)*3, 5+nClasses]
// 注: Tiny 版本比 Nano 版本少一级下采样（总下采样 8 倍）
// ============================================================================
class YOLOv7Tiny : public Module {
public:
    // 20260320 ZJH 构造函数
    // nNumClasses: 检测类别数，默认 20
    // nInChannels: 输入通道数，默认 3
    YOLOv7Tiny(int nNumClasses = 20, int nInChannels = 3)
        : m_stem(nInChannels, 32, 3, 1, 1, false),    // 20260320 ZJH stem: nIn->32
          m_stemBn(32),                                 // 20260320 ZJH stem BN
          m_act(0.1f),                                  // 20260320 ZJH LeakyReLU
          m_down1(32, 64, 3, 2, 1, false),              // 20260320 ZJH 下采样 1: 32->64
          m_dbn1(64),                                   // 20260320 ZJH BN1
          m_elan1(64, 64),                              // 20260320 ZJH ELAN 块 1: 64->64
          m_down2(64, 128, 3, 2, 1, false),             // 20260320 ZJH 下采样 2: 64->128
          m_dbn2(128),                                  // 20260320 ZJH BN2
          m_elan2(128, 128),                            // 20260320 ZJH ELAN 块 2: 128->128
          m_down3(128, 256, 3, 2, 1, false),            // 20260320 ZJH 下采样 3: 128->256
          m_dbn3(256),                                  // 20260320 ZJH BN3
          m_elan3(256, 256),                            // 20260320 ZJH ELAN 块 3: 256->256
          m_pool(2, 2, 0),                              // 20260320 ZJH MaxPool 2x2（可选，此处不使用）
          m_head(256, 3, nNumClasses)                   // 20260320 ZJH 检测头: 3 anchors
    {}

    // 20260320 ZJH forward — YOLOv7Tiny 前向传播
    // input: [N, nInChannels, H, W]
    // 返回: [N, (H/8)*(W/8)*3, 5+nClasses]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH Stem: [N,3,H,W] -> [N,32,H,W]
        auto x = m_stem.forward(input);
        x = m_stemBn.forward(x);
        x = m_act.forward(x);

        // 20260320 ZJH Stage 1: [N,32,H,W] -> [N,64,H/2,W/2]
        x = m_down1.forward(x);  x = m_dbn1.forward(x);  x = m_act.forward(x);
        x = m_elan1.forward(x);

        // 20260320 ZJH Stage 2: [N,64,H/2,W/2] -> [N,128,H/4,W/4]
        x = m_down2.forward(x);  x = m_dbn2.forward(x);  x = m_act.forward(x);
        x = m_elan2.forward(x);

        // 20260320 ZJH Stage 3: [N,128,H/4,W/4] -> [N,256,H/8,W/8]
        x = m_down3.forward(x);  x = m_dbn3.forward(x);  x = m_act.forward(x);
        x = m_elan3.forward(x);

        // 20260320 ZJH 检测头: [N,256,H/8,W/8] -> [N,(H/8)*(W/8)*3,5+nClasses]
        x = m_head.forward(x);
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_stem.parameters()); append(m_stemBn.parameters());
        append(m_down1.parameters()); append(m_dbn1.parameters()); append(m_elan1.parameters());
        append(m_down2.parameters()); append(m_dbn2.parameters()); append(m_elan2.parameters());
        append(m_down3.parameters()); append(m_dbn3.parameters()); append(m_elan3.parameters());
        append(m_head.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_stem.train(bMode); m_stemBn.train(bMode);
        m_down1.train(bMode); m_dbn1.train(bMode); m_elan1.train(bMode);
        m_down2.train(bMode); m_dbn2.train(bMode); m_elan2.train(bMode);
        m_down3.train(bMode); m_dbn3.train(bMode); m_elan3.train(bMode);
        m_head.train(bMode);
    }

private:
    Conv2d m_stem;             // 20260320 ZJH Stem 卷积
    BatchNorm2d m_stemBn;      // 20260320 ZJH Stem BN
    LeakyReLU m_act;           // 20260320 ZJH 共用 LeakyReLU
    Conv2d m_down1, m_down2, m_down3;             // 20260320 ZJH 下采样卷积
    BatchNorm2d m_dbn1, m_dbn2, m_dbn3;           // 20260320 ZJH 下采样 BN
    ELANBlock m_elan1, m_elan2, m_elan3;          // 20260320 ZJH ELAN 块
    MaxPool2d m_pool;          // 20260320 ZJH MaxPool（保留备用）
    YOLOHead m_head;           // 20260320 ZJH anchor-based 检测头
};

// ============================================================================
// 20260320 ZJH SCDownBlock — YOLOv10 空间-通道解耦下采样模块
// 先通过 1x1 卷积变换通道，再通过 depthwise 3x3 stride=2 做空间下采样
// 输入: [N, Cin, H, W]
// 输出: [N, Cout, H/2, W/2]
// ============================================================================
class SCDownBlock : public Module {
public:
    // 20260320 ZJH 构造函数
    // nIn: 输入通道数
    // nOut: 输出通道数
    SCDownBlock(int nIn, int nOut)
        : m_pointwise(nIn, nOut, 1, 1, 0, false),      // 20260320 ZJH 1x1 通道变换
          m_depthwise(nOut, nOut, 3, 2, 1, false),      // 20260320 ZJH 3x3 stride=2 空间下采样
          m_bn1(nOut),                                   // 20260320 ZJH pointwise BN
          m_bn2(nOut),                                   // 20260320 ZJH depthwise BN
          m_act(0.1f)                                    // 20260320 ZJH LeakyReLU
    {}

    // 20260320 ZJH forward — 空间-通道解耦下采样前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, H/2, W/2]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH 1x1 通道变换
        auto x = m_pointwise.forward(input);   // 20260320 ZJH [N,Cin,H,W] -> [N,Cout,H,W]
        x = m_bn1.forward(x);                  // 20260320 ZJH BN
        x = m_act.forward(x);                  // 20260320 ZJH LeakyReLU
        // 20260320 ZJH 3x3 空间下采样
        x = m_depthwise.forward(x);            // 20260320 ZJH [N,Cout,H,W] -> [N,Cout,H/2,W/2]
        x = m_bn2.forward(x);                  // 20260320 ZJH BN
        x = m_act.forward(x);                  // 20260320 ZJH LeakyReLU
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_pointwise.parameters()); append(m_bn1.parameters());
        append(m_depthwise.parameters()); append(m_bn2.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_pointwise.train(bMode); m_bn1.train(bMode);
        m_depthwise.train(bMode); m_bn2.train(bMode);
    }

private:
    Conv2d m_pointwise;        // 20260320 ZJH 1x1 通道变换
    Conv2d m_depthwise;        // 20260320 ZJH 3x3 stride=2 空间下采样
    BatchNorm2d m_bn1, m_bn2;  // 20260320 ZJH BN
    LeakyReLU m_act;           // 20260320 ZJH LeakyReLU
};

// ============================================================================
// 20260320 ZJH YOLOv10Nano — YOLOv10-Nano 检测网络（NMS-free, 解耦下采样）
// 骨干: stem(3x3) -> SCDown(解耦下采样) x3 + C2f 块
// 头: YOLOv8Head（复用解耦检测头）
// 输入: [N, 3, H, W]（H, W 应为 16 的倍数）
// 输出: [N, (H/16)*(W/16), 4+nClasses]
// 关键特点：NMS-free 一对一匹配，空间-通道解耦下采样
// ============================================================================
class YOLOv10Nano : public Module {
public:
    // 20260320 ZJH 构造函数
    // nNumClasses: 检测类别数，默认 20
    // nInChannels: 输入通道数，默认 3
    YOLOv10Nano(int nNumClasses = 20, int nInChannels = 3)
        : m_stem(nInChannels, 16, 3, 1, 1, false),    // 20260320 ZJH stem: nIn->16
          m_stemBn(16),                                 // 20260320 ZJH stem BN
          m_act(0.1f),                                  // 20260320 ZJH LeakyReLU
          m_down1(16, 32),                              // 20260320 ZJH SCDown 1: 16->32, /2
          m_c2f1(32),                                   // 20260320 ZJH C2f 块 1
          m_down2(32, 128),                             // 20260320 ZJH SCDown 2: 32->128, /2
          m_c2f2(128),                                  // 20260320 ZJH C2f 块 2
          m_down3(128, 256),                            // 20260320 ZJH SCDown 3: 128->256, /2
          m_c2f3(256),                                  // 20260320 ZJH C2f 块 3
          m_head(256, nNumClasses)                      // 20260320 ZJH 解耦检测头
    {}

    // 20260320 ZJH forward — YOLOv10Nano 前向传播
    // input: [N, nInChannels, H, W]
    // 返回: [N, (H/16)*(W/16), 4+nClasses]
    Tensor forward(const Tensor& input) override {
        // 20260320 ZJH Stem: [N,3,H,W] -> [N,16,H,W]
        auto x = m_stem.forward(input);
        x = m_stemBn.forward(x);
        x = m_act.forward(x);

        // 20260320 ZJH Stage 1: [N,16,H,W] -> [N,32,H/2,W/2]
        x = m_down1.forward(x);               // 20260320 ZJH SCDown 解耦下采样
        x = m_c2f1.forward(x);                // 20260320 ZJH C2f 特征提取

        // 20260320 ZJH Stage 2: [N,32,H/2,W/2] -> [N,128,H/4,W/4]
        x = m_down2.forward(x);
        x = m_c2f2.forward(x);

        // 20260320 ZJH Stage 3: [N,128,H/4,W/4] -> [N,256,H/8,W/8]
        x = m_down3.forward(x);
        x = m_c2f3.forward(x);

        // 20260320 ZJH 检测头: [N,256,H/8,W/8] -> [N,(H/8)*(W/8),4+nClasses]
        x = m_head.forward(x);
        return x;
    }

    // 20260320 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_stem.parameters()); append(m_stemBn.parameters());
        append(m_down1.parameters()); append(m_c2f1.parameters());
        append(m_down2.parameters()); append(m_c2f2.parameters());
        append(m_down3.parameters()); append(m_c2f3.parameters());
        append(m_head.parameters());
        return vecResult;
    }

    // 20260320 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_stem.train(bMode); m_stemBn.train(bMode);
        m_down1.train(bMode); m_c2f1.train(bMode);
        m_down2.train(bMode); m_c2f2.train(bMode);
        m_down3.train(bMode); m_c2f3.train(bMode);
        m_head.train(bMode);
    }

private:
    Conv2d m_stem;             // 20260320 ZJH Stem 卷积
    BatchNorm2d m_stemBn;      // 20260320 ZJH Stem BN
    LeakyReLU m_act;           // 20260320 ZJH 共用 LeakyReLU
    SCDownBlock m_down1, m_down2, m_down3;        // 20260320 ZJH 解耦下采样
    C2fBlock m_c2f1, m_c2f2, m_c2f3;             // 20260320 ZJH C2f 块
    YOLOv8Head m_head;         // 20260320 ZJH 解耦检测头
};

}  // namespace df
