// 20260321 ZJH 实例分割引擎模块 — Phase 5B
// 实现 ROI Align + 简化 Mask R-CNN 风格的实例分割
// 架构: FPN backbone → RPN 候选框 → ROI Align → 分类/回归/Mask 三分支
// 以及 YOLACT 风格单阶段方案: backbone → prototype masks + 系数预测 → 线性组合
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <limits>

export module om.engine.instance_seg;

// 20260321 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// ROI Align — 区域特征对齐提取
// =========================================================

// 20260321 ZJH ROIAlign — 从特征图中按 ROI 坐标提取固定大小的特征
// 使用双线性插值避免量化误差（比 ROI Pooling 更精确）
class ROIAlign {
public:
    // 20260321 ZJH 构造函数
    // nOutputH: 输出特征高度
    // nOutputW: 输出特征宽度
    // fSpatialScale: 空间缩放因子（特征图尺寸 / 原图尺寸）
    // nSamplingRatio: 每个 bin 的采样点数（0 表示自适应）
    ROIAlign(int nOutputH, int nOutputW, float fSpatialScale = 1.0f / 16.0f,
             int nSamplingRatio = 2)
        : m_nOutputH(nOutputH), m_nOutputW(nOutputW),
          m_fSpatialScale(fSpatialScale), m_nSamplingRatio(nSamplingRatio) {}

    // 20260321 ZJH forward — 从特征图中提取 ROI 特征
    // features: [1, C, H, W] 特征图（单张图像）
    // rois: [N, 4] ROI 坐标 (x1, y1, x2, y2)，原图坐标系
    // 返回: [N, C, outputH, outputW] 对齐后的 ROI 特征
    Tensor forward(const Tensor& features, const Tensor& rois) {
        auto cf = features.contiguous();
        auto cr = rois.contiguous();
        auto vecFeatShape = cf.shapeVec();
        int nC = vecFeatShape[1];       // 20260321 ZJH 通道数
        int nFeatH = vecFeatShape[2];   // 20260321 ZJH 特征图高度
        int nFeatW = vecFeatShape[3];   // 20260321 ZJH 特征图宽度

        auto vecRoiShape = cr.shapeVec();
        int nNumRois = vecRoiShape[0];  // 20260321 ZJH ROI 数量

        auto result = Tensor::zeros({nNumRois, nC, m_nOutputH, m_nOutputW});
        const float* pFeat = cf.floatDataPtr();
        const float* pRois = cr.floatDataPtr();
        float* pOut = result.mutableFloatDataPtr();

        // 20260321 ZJH 逐 ROI 处理
        for (int n = 0; n < nNumRois; ++n) {
            // 20260321 ZJH 获取 ROI 坐标并缩放到特征图尺度
            float fX1 = pRois[n * 4 + 0] * m_fSpatialScale;
            float fY1 = pRois[n * 4 + 1] * m_fSpatialScale;
            float fX2 = pRois[n * 4 + 2] * m_fSpatialScale;
            float fY2 = pRois[n * 4 + 3] * m_fSpatialScale;

            float fRoiW = fX2 - fX1;  // 20260321 ZJH ROI 宽度（特征图尺度）
            float fRoiH = fY2 - fY1;  // 20260321 ZJH ROI 高度（特征图尺度）

            float fBinH = fRoiH / static_cast<float>(m_nOutputH);  // 20260321 ZJH 每个 bin 高度
            float fBinW = fRoiW / static_cast<float>(m_nOutputW);  // 20260321 ZJH 每个 bin 宽度

            int nSampH = m_nSamplingRatio;  // 20260321 ZJH 垂直采样点数
            int nSampW = m_nSamplingRatio;  // 20260321 ZJH 水平采样点数

            // 20260321 ZJH 逐通道、逐 bin 提取特征
            for (int c = 0; c < nC; ++c) {
                for (int oh = 0; oh < m_nOutputH; ++oh) {
                    for (int ow = 0; ow < m_nOutputW; ++ow) {
                        float fSum = 0.0f;
                        int nCount = 0;

                        // 20260321 ZJH bin 内均匀采样并双线性插值
                        for (int sh = 0; sh < nSampH; ++sh) {
                            for (int sw = 0; sw < nSampW; ++sw) {
                                // 20260321 ZJH 计算采样点在特征图上的坐标
                                float fY = fY1 + fBinH * (oh + (sh + 0.5f) / nSampH);
                                float fX = fX1 + fBinW * (ow + (sw + 0.5f) / nSampW);

                                // 20260321 ZJH 双线性插值
                                fSum += bilinearInterpolate(pFeat, nC, nFeatH, nFeatW, c, fY, fX);
                                nCount++;
                            }
                        }

                        // 20260321 ZJH 取平均作为该 bin 的输出
                        int nOutIdx = n * nC * m_nOutputH * m_nOutputW +
                                      c * m_nOutputH * m_nOutputW +
                                      oh * m_nOutputW + ow;
                        pOut[nOutIdx] = fSum / static_cast<float>(nCount);
                    }
                }
            }
        }

        return result;
    }

private:
    int m_nOutputH;        // 20260321 ZJH 输出高度
    int m_nOutputW;        // 20260321 ZJH 输出宽度
    float m_fSpatialScale; // 20260321 ZJH 空间缩放因子
    int m_nSamplingRatio;  // 20260321 ZJH 每 bin 采样点数

    // 20260321 ZJH 双线性插值 — 在特征图上的浮点坐标处采样
    static float bilinearInterpolate(const float* pData, int nC, int nH, int nW,
                                      int c, float fY, float fX) {
        // 20260321 ZJH 边界裁剪
        if (fY < -1.0f || fY > static_cast<float>(nH) ||
            fX < -1.0f || fX > static_cast<float>(nW)) {
            return 0.0f;
        }

        fY = std::max(fY, 0.0f);
        fX = std::max(fX, 0.0f);

        int nYLow = static_cast<int>(fY);      // 20260321 ZJH 下方行索引
        int nXLow = static_cast<int>(fX);      // 20260321 ZJH 左侧列索引
        int nYHigh = nYLow + 1;                 // 20260321 ZJH 上方行索引
        int nXHigh = nXLow + 1;                 // 20260321 ZJH 右侧列索引

        // 20260321 ZJH 边界保护
        if (nYLow >= nH - 1) { nYLow = nH - 1; nYHigh = nH - 1; fY = static_cast<float>(nYLow); }
        if (nXLow >= nW - 1) { nXLow = nW - 1; nXHigh = nW - 1; fX = static_cast<float>(nXLow); }

        float fLY = fY - nYLow;  // 20260321 ZJH 垂直插值权重
        float fLX = fX - nXLow;  // 20260321 ZJH 水平插值权重
        float fHY = 1.0f - fLY;
        float fHX = 1.0f - fLX;

        // 20260321 ZJH 四个角点的值
        float v1 = pData[c * nH * nW + nYLow * nW + nXLow];
        float v2 = pData[c * nH * nW + nYLow * nW + nXHigh];
        float v3 = pData[c * nH * nW + nYHigh * nW + nXLow];
        float v4 = pData[c * nH * nW + nYHigh * nW + nXHigh];

        // 20260321 ZJH 双线性插值结果
        return fHY * fHX * v1 + fHY * fLX * v2 + fLY * fHX * v3 + fLY * fLX * v4;
    }
};

// =========================================================
// ProtoNet — Prototype Mask 生成网络
// =========================================================

// 20260321 ZJH ProtoNet — 生成全局 prototype masks
// YOLACT 核心: 从 FPN 特征生成 K 个 prototype masks
// 每个实例的最终 mask = 系数向量与 prototypes 的线性组合
class ProtoNet : public Module {
public:
    // 20260321 ZJH 构造函数
    // nInChannels: 输入特征通道数
    // nNumPrototypes: prototype 数量（默认 32）
    ProtoNet(int nInChannels, int nNumPrototypes = 32)
        : m_nNumPrototypes(nNumPrototypes)
    {
        // 20260321 ZJH 3 层卷积逐步提炼 prototype
        m_pConv1 = std::make_shared<Conv2d>(nInChannels, 256, 3, 1, 1);
        m_pBn1 = std::make_shared<BatchNorm2d>(256);
        m_pConv2 = std::make_shared<Conv2d>(256, 256, 3, 1, 1);
        m_pBn2 = std::make_shared<BatchNorm2d>(256);
        m_pConv3 = std::make_shared<Conv2d>(256, nNumPrototypes, 1, 1, 0);  // 1x1 投影

        registerModule("proto_conv1", m_pConv1);
        registerModule("proto_bn1", m_pBn1);
        registerModule("proto_conv2", m_pConv2);
        registerModule("proto_bn2", m_pBn2);
        registerModule("proto_conv3", m_pConv3);
    }

    // 20260321 ZJH forward — 生成 prototype masks
    // input: [B, C, H, W] FPN 特征
    // 返回: [B, K, H, W] K 个 prototype masks
    Tensor forward(const Tensor& input) override {
        auto x = tensorReLU(m_pBn1->forward(m_pConv1->forward(input)));
        x = tensorReLU(m_pBn2->forward(m_pConv2->forward(x)));
        x = m_pConv3->forward(x);  // 20260321 ZJH 不加激活，sigmoid 在组装时做
        return x;
    }

    // 20260406 ZJH 获取 prototype 数量（外部解析预测结果时使用）
    int numPrototypes() const { return m_nNumPrototypes; }

private:
    int m_nNumPrototypes;  // 20260321 ZJH prototype 数量
    std::shared_ptr<Conv2d> m_pConv1, m_pConv2, m_pConv3;  // 20260406 ZJH 3 层卷积（逐步提炼 prototype 特征）
    std::shared_ptr<BatchNorm2d> m_pBn1, m_pBn2;  // 20260406 ZJH 前两层的 BN（第三层无 BN，直接输出）
};

// =========================================================
// InstanceHead — 实例分割预测头
// =========================================================

// 20260321 ZJH InstanceHead — 预测边界框 + 类别 + mask 系数
// 对每个检测结果预测 K 个系数，线性组合 prototype masks 得到实例 mask
class InstanceHead : public Module {
public:
    // 20260321 ZJH 构造函数
    // nInChannels: 输入特征通道数
    // nNumClasses: 类别数（不含背景）
    // nNumPrototypes: prototype 数量
    // nNumAnchors: 每个位置的锚框数量
    InstanceHead(int nInChannels, int nNumClasses, int nNumPrototypes = 32,
                  int nNumAnchors = 3)
        : m_nNumClasses(nNumClasses), m_nNumPrototypes(nNumPrototypes),
          m_nNumAnchors(nNumAnchors)
    {
        // 20260321 ZJH 共享特征提取
        m_pSharedConv1 = std::make_shared<Conv2d>(nInChannels, 256, 3, 1, 1);
        m_pSharedBn1 = std::make_shared<BatchNorm2d>(256);
        m_pSharedConv2 = std::make_shared<Conv2d>(256, 256, 3, 1, 1);
        m_pSharedBn2 = std::make_shared<BatchNorm2d>(256);

        // 20260321 ZJH 分类头: 预测每个锚框的类别分数
        int nClsOut = nNumAnchors * nNumClasses;
        m_pClsConv = std::make_shared<Conv2d>(256, nClsOut, 1, 1, 0);

        // 20260321 ZJH 回归头: 预测每个锚框的边界框偏移 (dx, dy, dw, dh)
        int nBboxOut = nNumAnchors * 4;
        m_pBboxConv = std::make_shared<Conv2d>(256, nBboxOut, 1, 1, 0);

        // 20260321 ZJH 系数头: 预测每个锚框的 prototype 系数
        int nCoeffOut = nNumAnchors * nNumPrototypes;
        m_pCoeffConv = std::make_shared<Conv2d>(256, nCoeffOut, 1, 1, 0);

        registerModule("shared_conv1", m_pSharedConv1);
        registerModule("shared_bn1", m_pSharedBn1);
        registerModule("shared_conv2", m_pSharedConv2);
        registerModule("shared_bn2", m_pSharedBn2);
        registerModule("cls_conv", m_pClsConv);
        registerModule("bbox_conv", m_pBboxConv);
        registerModule("coeff_conv", m_pCoeffConv);
    }

    // 20260321 ZJH forward — 多分支预测
    // input: [B, C, H, W] 特征图
    // 返回: 分类+回归+系数拼接后的张量
    Tensor forward(const Tensor& input) override {
        // 20260321 ZJH 共享特征
        auto x = tensorReLU(m_pSharedBn1->forward(m_pSharedConv1->forward(input)));
        x = tensorReLU(m_pSharedBn2->forward(m_pSharedConv2->forward(x)));

        // 20260321 ZJH 三分支预测
        auto clsOut = m_pClsConv->forward(x);    // [B, A*C, H, W]
        auto bboxOut = m_pBboxConv->forward(x);   // [B, A*4, H, W]
        auto coeffOut = m_pCoeffConv->forward(x);  // [B, A*K, H, W]

        // 20260321 ZJH 拼接所有输出沿通道维度
        auto combined = tensorConcatChannels(clsOut, bboxOut);
        combined = tensorConcatChannels(combined, coeffOut);
        return combined;
    }

    // 20260406 ZJH 获取类别数（用于解析预测通道布局）
    int numClasses() const { return m_nNumClasses; }
    // 20260406 ZJH 获取 prototype 数量（用于解析 mask 系数维度）
    int numPrototypes() const { return m_nNumPrototypes; }
    // 20260406 ZJH 获取锚框数量（用于解析预测网格结构）
    int numAnchors() const { return m_nNumAnchors; }

private:
    int m_nNumClasses;     // 20260321 ZJH 类别数
    int m_nNumPrototypes;  // 20260321 ZJH prototype 数量
    int m_nNumAnchors;     // 20260321 ZJH 锚框数量

    std::shared_ptr<Conv2d> m_pSharedConv1, m_pSharedConv2;  // 20260406 ZJH 共享特征提取卷积（两层 3x3）
    std::shared_ptr<BatchNorm2d> m_pSharedBn1, m_pSharedBn2;  // 20260406 ZJH 共享特征 BN
    std::shared_ptr<Conv2d> m_pClsConv, m_pBboxConv, m_pCoeffConv;  // 20260406 ZJH 分类/回归/系数三分支 1x1 预测头
};

// =========================================================
// InstanceSegOutput — 实例分割结构化输出
// =========================================================

// 20260330 ZJH 实例分割结构化输出（推理时需要 prototypes + predictions 两者才能组装 mask）
struct InstanceSegOutput {
    Tensor prototypes;   // 20260330 ZJH [B, K, H/4, W/4] prototype masks
    Tensor predictions;  // 20260330 ZJH [B, nPreds, 5+K+C] 检测预测 (bbox + coeffs + class)
};

// =========================================================
// SimpleInstanceSeg — 简化实例分割网络
// =========================================================

// 20260321 ZJH SimpleInstanceSeg — YOLACT 风格简化实例分割
// 架构: 编码器 → ProtoNet + InstanceHead
// 编码器提取多尺度特征，ProtoNet 生成 prototype masks，
// InstanceHead 预测检测框和 mask 系数
class SimpleInstanceSeg : public Module {
public:
    // 20260321 ZJH 构造函数
    // nInChannels: 输入通道数（通常 3 RGB 或 1 灰度）
    // nNumClasses: 类别数（不含背景）
    // nNumPrototypes: prototype mask 数量
    SimpleInstanceSeg(int nInChannels, int nNumClasses, int nNumPrototypes = 32)
        : m_nNumClasses(nNumClasses), m_nNumPrototypes(nNumPrototypes)
    {
        // 20260321 ZJH 简化编码器: 4 组卷积 + 池化
        m_pEnc1 = std::make_shared<Conv2d>(nInChannels, 64, 3, 1, 1);
        m_pBnEnc1 = std::make_shared<BatchNorm2d>(64);
        m_pEnc2 = std::make_shared<Conv2d>(64, 128, 3, 1, 1);
        m_pBnEnc2 = std::make_shared<BatchNorm2d>(128);
        m_pEnc3 = std::make_shared<Conv2d>(128, 256, 3, 1, 1);
        m_pBnEnc3 = std::make_shared<BatchNorm2d>(256);
        m_pEnc4 = std::make_shared<Conv2d>(256, 256, 3, 1, 1);
        m_pBnEnc4 = std::make_shared<BatchNorm2d>(256);

        // 20260321 ZJH ProtoNet: 从浅层特征生成 prototypes
        m_pProtoNet = std::make_shared<ProtoNet>(128, nNumPrototypes);

        // 20260321 ZJH InstanceHead: 从深层特征预测检测和系数
        m_pHead = std::make_shared<InstanceHead>(256, nNumClasses, nNumPrototypes);

        // 20260321 ZJH 注册子模块
        registerModule("enc1", m_pEnc1);
        registerModule("bn_enc1", m_pBnEnc1);
        registerModule("enc2", m_pEnc2);
        registerModule("bn_enc2", m_pBnEnc2);
        registerModule("enc3", m_pEnc3);
        registerModule("bn_enc3", m_pBnEnc3);
        registerModule("enc4", m_pEnc4);
        registerModule("bn_enc4", m_pBnEnc4);
        registerModule("proto_net", m_pProtoNet);
        registerModule("instance_head", m_pHead);
    }

    // 20260321 ZJH forward — 前向传播
    // input: [B, C, H, W] 输入图像
    // 返回: 拼接的预测结果张量
    Tensor forward(const Tensor& input) override {
        // 20260321 ZJH 编码器
        auto x1 = tensorReLU(m_pBnEnc1->forward(m_pEnc1->forward(input)));
        auto x1p = tensorMaxPool2d(x1, 2, 2, 0);  // [B, 64, H/2, W/2]

        auto x2 = tensorReLU(m_pBnEnc2->forward(m_pEnc2->forward(x1p)));
        auto x2p = tensorMaxPool2d(x2, 2, 2, 0);  // [B, 128, H/4, W/4]

        auto x3 = tensorReLU(m_pBnEnc3->forward(m_pEnc3->forward(x2p)));
        auto x3p = tensorMaxPool2d(x3, 2, 2, 0);  // [B, 256, H/8, W/8]

        auto x4 = tensorReLU(m_pBnEnc4->forward(m_pEnc4->forward(x3p)));

        // 20260321 ZJH ProtoNet: 从 x2（H/4 分辨率）生成 prototypes
        auto prototypes = m_pProtoNet->forward(x2);  // [B, K, H/4, W/4]

        // 20260321 ZJH InstanceHead: 从 x4（H/8 分辨率）预测
        auto predictions = m_pHead->forward(x4);

        // 20260321 ZJH 返回 prototypes 和 predictions 的通道拼接
        // 实际推理时分别解析
        return predictions;
    }

    // 20260330 ZJH 推理专用前向：同时返回 prototypes 和 predictions
    // forward() 只返回 predictions（训练用），forwardInfer() 返回两者（推理组装 mask 用）
    InstanceSegOutput forwardInfer(const Tensor& input) {
        // 20260330 ZJH 编码器路径（与 forward() 相同）
        auto x1 = tensorReLU(m_pBnEnc1->forward(m_pEnc1->forward(input)));
        auto x1p = tensorMaxPool2d(x1, 2, 2, 0);  // 20260330 ZJH [B, 64, H/2, W/2]

        auto x2 = tensorReLU(m_pBnEnc2->forward(m_pEnc2->forward(x1p)));
        auto x2p = tensorMaxPool2d(x2, 2, 2, 0);  // 20260330 ZJH [B, 128, H/4, W/4]

        auto x3 = tensorReLU(m_pBnEnc3->forward(m_pEnc3->forward(x2p)));
        auto x3p = tensorMaxPool2d(x3, 2, 2, 0);  // 20260330 ZJH [B, 256, H/8, W/8]

        auto x4 = tensorReLU(m_pBnEnc4->forward(m_pEnc4->forward(x3p)));

        // 20260330 ZJH ProtoNet: 从 x2（H/4 分辨率）生成 prototypes
        auto prototypes = m_pProtoNet->forward(x2);  // 20260330 ZJH [B, K, H/4, W/4]

        // 20260330 ZJH InstanceHead: 从 x4（H/8 分辨率）预测检测框 + 类别 + mask 系数
        auto predictions = m_pHead->forward(x4);

        // 20260330 ZJH 同时返回 prototypes 和 predictions，推理时用于组装实例 mask
        InstanceSegOutput output;
        output.prototypes = prototypes;
        output.predictions = predictions;
        return output;
    }

    // 20260321 ZJH getPrototypes — 单独获取 prototype masks（推理时使用）
    // input: [B, C, H, W] 输入图像
    // 返回: [B, K, H/4, W/4] K 个 prototype masks
    Tensor getPrototypes(const Tensor& input) {
        auto x1 = tensorReLU(m_pBnEnc1->forward(m_pEnc1->forward(input)));  // 20260406 ZJH 编码器层1
        auto x1p = tensorMaxPool2d(x1, 2, 2, 0);  // 20260406 ZJH 2x 下采样
        auto x2 = tensorReLU(m_pBnEnc2->forward(m_pEnc2->forward(x1p)));  // 20260406 ZJH 编码器层2
        return m_pProtoNet->forward(x2);  // 20260406 ZJH 从浅层特征生成 prototypes
    }

    // 20260406 ZJH 获取类别数
    int numClasses() const { return m_nNumClasses; }
    // 20260406 ZJH 获取 prototype 数量
    int numPrototypes() const { return m_nNumPrototypes; }

private:
    int m_nNumClasses;  // 20260406 ZJH 类别数（不含背景）
    int m_nNumPrototypes;  // 20260406 ZJH prototype mask 数量

    // 20260321 ZJH 编码器: 4 组 Conv + BN + ReLU + MaxPool
    std::shared_ptr<Conv2d> m_pEnc1, m_pEnc2, m_pEnc3, m_pEnc4;  // 20260406 ZJH 编码器卷积（通道: nIn→64→128→256→256）
    std::shared_ptr<BatchNorm2d> m_pBnEnc1, m_pBnEnc2, m_pBnEnc3, m_pBnEnc4;  // 20260406 ZJH 编码器 BN

    // 20260321 ZJH ProtoNet 和 InstanceHead
    std::shared_ptr<ProtoNet> m_pProtoNet;  // 20260406 ZJH ProtoNet: 从浅层特征(x2)生成 K 个 prototype masks
    std::shared_ptr<InstanceHead> m_pHead;  // 20260406 ZJH InstanceHead: 从深层特征(x4)预测检测+mask系数
};

// =========================================================
// NMS — 非极大值抑制
// =========================================================

// 20260321 ZJH 检测结果结构
struct InstanceResult {
    int nClassId;               // 20260321 ZJH 类别 ID
    float fScore;               // 20260321 ZJH 置信度分数
    float fX1, fY1, fX2, fY2;  // 20260321 ZJH 边界框坐标 (xyxy)
    std::vector<float> vecMaskCoeffs;  // 20260321 ZJH mask 系数向量
    std::vector<float> vecMask;  // 20260330 ZJH H×W 二值 mask（后处理填充）
    int nMaskH = 0;              // 20260330 ZJH mask 高度
    int nMaskW = 0;              // 20260330 ZJH mask 宽度
};

// 20260321 ZJH computeIoU — 计算两个边界框的 IoU
// 20260330 ZJH FIX-6: 添加 std::max(0.0f, ...) 保护，防止退化框（x2<x1 或 y2<y1）产生负面积
float computeIoU(const InstanceResult& a, const InstanceResult& b) {
    float fX1 = std::max(a.fX1, b.fX1);
    float fY1 = std::max(a.fY1, b.fY1);
    float fX2 = std::min(a.fX2, b.fX2);
    float fY2 = std::min(a.fY2, b.fY2);

    float fInter = std::max(0.0f, fX2 - fX1) * std::max(0.0f, fY2 - fY1);
    // 20260330 ZJH FIX-6: 单个框的面积也需保护，防止退化框 (x2 < x1) 导致负面积
    float fWa = std::max(0.0f, a.fX2 - a.fX1);  // 20260330 ZJH 框 A 宽度（非负）
    float fHa = std::max(0.0f, a.fY2 - a.fY1);  // 20260330 ZJH 框 A 高度（非负）
    float fAreaA = fWa * fHa;                     // 20260330 ZJH 框 A 面积
    float fWb = std::max(0.0f, b.fX2 - b.fX1);  // 20260330 ZJH 框 B 宽度（非负）
    float fHb = std::max(0.0f, b.fY2 - b.fY1);  // 20260330 ZJH 框 B 高度（非负）
    float fAreaB = fWb * fHb;                     // 20260330 ZJH 框 B 面积
    float fUnion = fAreaA + fAreaB - fInter;

    return (fUnion > 0.0f) ? fInter / fUnion : 0.0f;
}

// 20260321 ZJH instanceNMS — 实例级 NMS
// 按置信度排序，逐个保留，抑制高 IoU 重叠的低分检测
std::vector<InstanceResult> instanceNMS(std::vector<InstanceResult>& vecResults,
                                         float fIoUThreshold = 0.5f) {
    // 20260321 ZJH 按分数降序排列
    std::sort(vecResults.begin(), vecResults.end(),
              [](const InstanceResult& a, const InstanceResult& b) {
                  return a.fScore > b.fScore;
              });

    std::vector<bool> vecSuppressed(vecResults.size(), false);  // 20260406 ZJH 抑制标记数组
    std::vector<InstanceResult> vecKept;  // 20260406 ZJH 保留的检测结果

    // 20260406 ZJH 遍历所有候选框，按分数从高到低逐个处理
    for (size_t i = 0; i < vecResults.size(); ++i) {
        if (vecSuppressed[i]) continue;  // 20260406 ZJH 已被抑制则跳过
        vecKept.push_back(vecResults[i]);  // 20260406 ZJH 保留当前最高分候选

        // 20260406 ZJH 与后续所有同类候选计算 IoU，抑制高重叠的低分候选
        for (size_t j = i + 1; j < vecResults.size(); ++j) {
            if (vecSuppressed[j]) continue;  // 20260406 ZJH 已被抑制则跳过
            if (vecResults[i].nClassId == vecResults[j].nClassId) {  // 20260406 ZJH 仅同类之间做 NMS
                float fIoU = computeIoU(vecResults[i], vecResults[j]);  // 20260406 ZJH 计算 IoU
                if (fIoU > fIoUThreshold) {
                    vecSuppressed[j] = true;  // 20260406 ZJH IoU 超过阈值则抑制
                }
            }
        }
    }

    return vecKept;  // 20260406 ZJH 返回 NMS 后保留的检测结果
}

// =========================================================
// assembleMasks — 从 prototypes 和系数组装实例 masks
// =========================================================

// 20260321 ZJH assembleMasks — 线性组合 prototype masks 生成实例 masks
// prototypes: [K, H, W] prototype masks
// coeffs: [N, K] 每个实例的系数向量
// 返回: [N, H, W] 实例 masks（sigmoid 后的概率图）
Tensor assembleMasks(const Tensor& prototypes, const Tensor& coeffs) {
    auto cp = prototypes.contiguous();
    auto cc = coeffs.contiguous();
    auto vecPShape = cp.shapeVec();
    auto vecCShape = cc.shapeVec();

    int nK = vecPShape[0];   // 20260321 ZJH prototype 数量
    int nH = vecPShape[1];   // 20260321 ZJH mask 高度
    int nW = vecPShape[2];   // 20260321 ZJH mask 宽度
    int nN = vecCShape[0];   // 20260321 ZJH 实例数量

    auto result = Tensor::zeros({nN, nH, nW});
    const float* pProto = cp.floatDataPtr();
    const float* pCoeff = cc.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260321 ZJH mask[n][h][w] = sigmoid(sum_k(coeff[n][k] * proto[k][h][w]))
    for (int n = 0; n < nN; ++n) {
        for (int h = 0; h < nH; ++h) {
            for (int w = 0; w < nW; ++w) {
                float fSum = 0.0f;
                for (int k = 0; k < nK; ++k) {
                    fSum += pCoeff[n * nK + k] * pProto[k * nH * nW + h * nW + w];
                }
                // 20260321 ZJH sigmoid 激活
                pOut[n * nH * nW + h * nW + w] = 1.0f / (1.0f + std::exp(-fSum));
            }
        }
    }

    return result;
}

// =========================================================
// 实例分割评估指标
// =========================================================

// 20260321 ZJH computeMaskIoU — 计算二值 mask 之间的 IoU
float computeMaskIoU(const std::vector<float>& vecMaskA,
                      const std::vector<float>& vecMaskB,
                      float fThreshold = 0.5f) {
    int nInter = 0;  // 20260406 ZJH 交集像素数
    int nUnion = 0;  // 20260406 ZJH 并集像素数
    // 20260406 ZJH 遍历所有像素，统计二值化后的交集和并集
    for (size_t i = 0; i < vecMaskA.size(); ++i) {
        bool bA = vecMaskA[i] > fThreshold;  // 20260406 ZJH mask A 在该像素是否为正
        bool bB = vecMaskB[i] > fThreshold;  // 20260406 ZJH mask B 在该像素是否为正
        if (bA && bB) nInter++;  // 20260406 ZJH 两者都为正则交集加 1
        if (bA || bB) nUnion++;  // 20260406 ZJH 任一为正则并集加 1
    }
    return (nUnion > 0) ? static_cast<float>(nInter) / static_cast<float>(nUnion) : 0.0f;  // 20260406 ZJH 返回 IoU，避免除零
}

// =========================================================
// decodeInstances — 实例分割后处理
// =========================================================

// 20260330 ZJH 实例分割后处理：从 raw output 解码出实例列表
// 步骤: 置信度过滤 → NMS → assembleMasks → crop to bbox → resize to original
// output: forwardInfer() 返回的 InstanceSegOutput
// fConfThresh: 置信度阈值，低于此值的预测被过滤
// fNmsThresh: NMS IoU 阈值
// nOrigH/nOrigW: 原始图像尺寸（>0 时将 mask resize 到原始尺寸）
// nNumClasses: 类别数（用于解析 predictions 通道布局）
// nNumPrototypes: prototype 数量（用于解析 mask 系数）
// nNumAnchors: 每个位置的锚框数量
std::vector<InstanceResult> decodeInstances(
    const InstanceSegOutput& output,
    int nNumClasses,
    int nNumPrototypes,
    int nNumAnchors = 3,
    float fConfThresh = 0.5f,
    float fNmsThresh = 0.5f,
    int nOrigH = 0, int nOrigW = 0)
{
    // 20260330 ZJH 提取 predictions 张量的形状信息
    auto pred = output.predictions.contiguous();
    auto vecPredShape = pred.shapeVec();
    // 20260330 ZJH predictions 形状: [B, A*C + A*4 + A*K, featH, featW]
    int nB = vecPredShape[0];           // 20260330 ZJH batch size（通常为 1）
    int nTotalChannels = vecPredShape[1];  // 20260330 ZJH 总通道数
    int nFeatH = vecPredShape[2];       // 20260330 ZJH 特征图高度
    int nFeatW = vecPredShape[3];       // 20260330 ZJH 特征图宽度

    // 20260330 ZJH 通道布局: [A*nNumClasses | A*4 | A*nNumPrototypes]
    int nClsChannels = nNumAnchors * nNumClasses;      // 20260330 ZJH 分类通道数
    int nBboxChannels = nNumAnchors * 4;                // 20260330 ZJH 回归通道数
    int nCoeffChannels = nNumAnchors * nNumPrototypes;  // 20260330 ZJH 系数通道数

    // 20260330 ZJH 提取 prototypes 形状
    auto proto = output.prototypes.contiguous();
    auto vecProtoShape = proto.shapeVec();
    // 20260330 ZJH prototypes 形状: [B, K, protoH, protoW]
    int nProtoH = vecProtoShape[2];  // 20260330 ZJH prototype mask 高度
    int nProtoW = vecProtoShape[3];  // 20260330 ZJH prototype mask 宽度

    const float* pPred = pred.floatDataPtr();
    const float* pProto = proto.floatDataPtr();

    // 20260330 ZJH 只处理 batch 0（单张推理）
    std::vector<InstanceResult> vecCandidates;

    // 20260330 ZJH 遍历特征图每个位置和每个锚框，提取预测
    for (int ah = 0; ah < nFeatH; ++ah) {
        for (int aw = 0; aw < nFeatW; ++aw) {
            for (int a = 0; a < nNumAnchors; ++a) {
                // 20260330 ZJH 提取当前锚框的分类分数
                // 通道排列: cls=[0..A*C-1], bbox=[A*C..A*C+A*4-1], coeff=[A*C+A*4..]
                int nBestClass = -1;       // 20260330 ZJH 最高分类别
                float fBestScore = -1.0f;  // 20260330 ZJH 最高分数

                for (int c = 0; c < nNumClasses; ++c) {
                    // 20260330 ZJH 分类通道索引: anchor a 的第 c 类 → 通道 a * nNumClasses + c
                    int nChIdx = a * nNumClasses + c;
                    // 20260330 ZJH 在 [B, Channels, H, W] 布局中的线性索引
                    int nIdx = nChIdx * nFeatH * nFeatW + ah * nFeatW + aw;
                    // 20260330 ZJH sigmoid 将 logit 转为概率
                    float fVal = pPred[nIdx];
                    float fProb = 1.0f / (1.0f + std::exp(-fVal));

                    if (fProb > fBestScore) {
                        fBestScore = fProb;
                        nBestClass = c;
                    }
                }

                // 20260330 ZJH 置信度过滤：低于阈值的预测直接跳过
                if (fBestScore < fConfThresh) continue;

                // 20260330 ZJH 提取边界框偏移 (dx, dy, dw, dh)
                float fBbox[4];
                for (int bi = 0; bi < 4; ++bi) {
                    int nChIdx = nClsChannels + a * 4 + bi;
                    int nIdx = nChIdx * nFeatH * nFeatW + ah * nFeatW + aw;
                    fBbox[bi] = pPred[nIdx];
                }

                // 20260330 ZJH 将偏移解码为绝对坐标 (简化: 直接用 grid 坐标 + offset)
                // 中心坐标 = (grid + sigmoid(offset)) / featSize，宽高 = exp(offset)
                float fCx = (static_cast<float>(aw) + 1.0f / (1.0f + std::exp(-fBbox[0]))) / static_cast<float>(nFeatW);
                float fCy = (static_cast<float>(ah) + 1.0f / (1.0f + std::exp(-fBbox[1]))) / static_cast<float>(nFeatH);
                // 20260330 ZJH 夹紧宽高偏移防止 exp() 溢出（>10 → exp(10)≈22000 已远超正常范围）
                float fClampedW = std::min(fBbox[2], 10.0f);
                float fClampedH = std::min(fBbox[3], 10.0f);
                float fW = std::exp(fClampedW) / static_cast<float>(nFeatW);
                float fH = std::exp(fClampedH) / static_cast<float>(nFeatH);

                // 20260330 ZJH 转换为 xyxy 格式（归一化 [0,1] 坐标）
                float fX1 = std::max(0.0f, fCx - fW * 0.5f);
                float fY1 = std::max(0.0f, fCy - fH * 0.5f);
                float fX2 = std::min(1.0f, fCx + fW * 0.5f);
                float fY2 = std::min(1.0f, fCy + fH * 0.5f);

                // 20260330 ZJH 提取 mask 系数向量
                std::vector<float> vecCoeffs(nNumPrototypes);
                for (int ki = 0; ki < nNumPrototypes; ++ki) {
                    int nChIdx = nClsChannels + nBboxChannels + a * nNumPrototypes + ki;
                    int nIdx = nChIdx * nFeatH * nFeatW + ah * nFeatW + aw;
                    vecCoeffs[ki] = pPred[nIdx];
                }

                // 20260330 ZJH 构建候选实例
                InstanceResult inst;
                inst.nClassId = nBestClass;
                inst.fScore = fBestScore;
                inst.fX1 = fX1;
                inst.fY1 = fY1;
                inst.fX2 = fX2;
                inst.fY2 = fY2;
                inst.vecMaskCoeffs = std::move(vecCoeffs);
                vecCandidates.push_back(std::move(inst));
            }
        }
    }

    // 20260330 ZJH NMS: 抑制重叠检测框（复用现有 instanceNMS 函数）
    auto vecKept = instanceNMS(vecCandidates, fNmsThresh);

    // 20260330 ZJH 为存活实例组装 mask
    // 从 prototypes[0] 提取 batch=0 的 [K, protoH, protoW]
    // 每个实例: mask = sigmoid(coeffs · prototypes)，然后 crop 到 bbox 并 resize
    for (auto& inst : vecKept) {
        // 20260330 ZJH 构建系数张量 [1, K]
        auto coeffTensor = Tensor::zeros({1, nNumPrototypes});
        float* pCoeffData = coeffTensor.mutableFloatDataPtr();
        for (int ki = 0; ki < nNumPrototypes; ++ki) {
            pCoeffData[ki] = inst.vecMaskCoeffs[ki];
        }

        // 20260330 ZJH 提取 batch=0 的 prototypes [K, protoH, protoW]
        // 手动从 [B, K, protoH, protoW] 中取第 0 个 batch
        auto protoSlice = Tensor::zeros({nNumPrototypes, nProtoH, nProtoW});
        float* pSliceData = protoSlice.mutableFloatDataPtr();
        int nProtoPlaneSize = nProtoH * nProtoW;  // 20260330 ZJH 单个 prototype 的像素数
        for (int ki = 0; ki < nNumPrototypes; ++ki) {
            for (int pi = 0; pi < nProtoPlaneSize; ++pi) {
                // 20260330 ZJH 源索引: batch=0 的 [K, H, W] 数据
                pSliceData[ki * nProtoPlaneSize + pi] = pProto[ki * nProtoPlaneSize + pi];
            }
        }

        // 20260330 ZJH 调用 assembleMasks 线性组合得到 [1, protoH, protoW] 的 sigmoid mask
        auto maskTensor = assembleMasks(protoSlice, coeffTensor);
        const float* pMaskData = maskTensor.floatDataPtr();

        // 20260330 ZJH 确定输出 mask 尺寸
        int nOutH = (nOrigH > 0) ? nOrigH : nProtoH;  // 20260330 ZJH 目标高度
        int nOutW = (nOrigW > 0) ? nOrigW : nProtoW;  // 20260330 ZJH 目标宽度

        // 20260330 ZJH 计算 bbox 在 prototype 分辨率下的像素坐标
        int nCropX1 = static_cast<int>(inst.fX1 * nProtoW);
        int nCropY1 = static_cast<int>(inst.fY1 * nProtoH);
        int nCropX2 = static_cast<int>(inst.fX2 * nProtoW);
        int nCropY2 = static_cast<int>(inst.fY2 * nProtoH);
        // 20260330 ZJH 边界保护
        nCropX1 = std::max(0, std::min(nCropX1, nProtoW - 1));
        nCropY1 = std::max(0, std::min(nCropY1, nProtoH - 1));
        nCropX2 = std::max(nCropX1 + 1, std::min(nCropX2, nProtoW));
        nCropY2 = std::max(nCropY1 + 1, std::min(nCropY2, nProtoH));

        int nCropW = nCropX2 - nCropX1;  // 20260330 ZJH 裁剪区域宽度
        int nCropH = nCropY2 - nCropY1;  // 20260330 ZJH 裁剪区域高度

        // 20260330 ZJH 生成输出尺寸的二值 mask
        // 先裁剪 prototype mask 到 bbox 区域，再双线性插值 resize 到原始图像尺寸
        inst.nMaskH = nOutH;
        inst.nMaskW = nOutW;
        inst.vecMask.resize(static_cast<size_t>(nOutH) * nOutW, 0.0f);

        // 20260330 ZJH bbox 在输出尺寸下的像素坐标（用于只在 bbox 内填充 mask）
        int nOutX1 = static_cast<int>(inst.fX1 * nOutW);
        int nOutY1 = static_cast<int>(inst.fY1 * nOutH);
        int nOutX2 = static_cast<int>(inst.fX2 * nOutW);
        int nOutY2 = static_cast<int>(inst.fY2 * nOutH);
        nOutX1 = std::max(0, std::min(nOutX1, nOutW - 1));
        nOutY1 = std::max(0, std::min(nOutY1, nOutH - 1));
        nOutX2 = std::max(nOutX1 + 1, std::min(nOutX2, nOutW));
        nOutY2 = std::max(nOutY1 + 1, std::min(nOutY2, nOutH));

        // 20260330 ZJH 双线性插值：从裁剪的 prototype mask resize 到 bbox 输出区域
        int nBoxOutW = nOutX2 - nOutX1;  // 20260330 ZJH 输出 bbox 宽度
        int nBoxOutH = nOutY2 - nOutY1;  // 20260330 ZJH 输出 bbox 高度

        for (int oy = 0; oy < nBoxOutH; ++oy) {
            for (int ox = 0; ox < nBoxOutW; ++ox) {
                // 20260330 ZJH 映射输出像素到裁剪区域的浮点坐标
                float fSrcY = static_cast<float>(nCropY1) +
                              static_cast<float>(oy) * static_cast<float>(nCropH) / static_cast<float>(nBoxOutH);
                float fSrcX = static_cast<float>(nCropX1) +
                              static_cast<float>(ox) * static_cast<float>(nCropW) / static_cast<float>(nBoxOutW);

                // 20260330 ZJH 双线性插值采样 prototype mask
                int nSrcY0 = static_cast<int>(fSrcY);
                int nSrcX0 = static_cast<int>(fSrcX);
                int nSrcY1 = std::min(nSrcY0 + 1, nProtoH - 1);
                int nSrcX1 = std::min(nSrcX0 + 1, nProtoW - 1);
                nSrcY0 = std::min(nSrcY0, nProtoH - 1);
                nSrcX0 = std::min(nSrcX0, nProtoW - 1);

                float fFracY = fSrcY - static_cast<float>(nSrcY0);  // 20260330 ZJH 垂直插值权重
                float fFracX = fSrcX - static_cast<float>(nSrcX0);  // 20260330 ZJH 水平插值权重

                // 20260330 ZJH 四角取值（mask 数据在 maskTensor 中索引: batch0, h, w）
                float v00 = pMaskData[nSrcY0 * nProtoW + nSrcX0];
                float v01 = pMaskData[nSrcY0 * nProtoW + nSrcX1];
                float v10 = pMaskData[nSrcY1 * nProtoW + nSrcX0];
                float v11 = pMaskData[nSrcY1 * nProtoW + nSrcX1];

                // 20260330 ZJH 双线性插值结果
                float fVal = v00 * (1.0f - fFracY) * (1.0f - fFracX) +
                             v01 * (1.0f - fFracY) * fFracX +
                             v10 * fFracY * (1.0f - fFracX) +
                             v11 * fFracY * fFracX;

                // 20260330 ZJH 二值化阈值 0.5 并写入输出 mask
                int nOutIdx = (nOutY1 + oy) * nOutW + (nOutX1 + ox);
                inst.vecMask[nOutIdx] = (fVal > 0.5f) ? 1.0f : 0.0f;
            }
        }
    }

    return vecKept;
}

}  // namespace om
