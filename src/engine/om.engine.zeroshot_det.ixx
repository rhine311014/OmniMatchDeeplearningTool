// 20260322 ZJH 零样本目标检测模块 — 基于模板匹配的目标检测
// 核心思路：用户提供参考模板图像 → 特征提取 → 滑动窗口 + 余弦相似度匹配 → NMS 去重
// 不依赖 CLIP 或任何外部预训练模型，纯 C++ 特征匹配实现
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <numeric>

export module om.engine.zeroshot_det;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// 20260322 ZJH DetectionResult — 检测结果结构体
// 包含边界框坐标、匹配分数、类别 ID
struct DetectionResult {
    float fX = 0.0f;   // 20260322 ZJH 边界框左上角 X（归一化坐标，0~1）
    float fY = 0.0f;   // 20260322 ZJH 边界框左上角 Y（归一化坐标，0~1）
    float fW = 0.0f;   // 20260322 ZJH 边界框宽度（归一化坐标，0~1）
    float fH = 0.0f;   // 20260322 ZJH 边界框高度（归一化坐标，0~1）
    float fScore = 0.0f;  // 20260322 ZJH 匹配分数 [0, 1]，越高越匹配
    int nClassId = 0;     // 20260322 ZJH 类别 ID
};

// 20260322 ZJH DetFeatureExtractorCNN — 检测用轻量特征提取 CNN（4层）
// 结构与 ZeroShotAD 中的 FeatureExtractorCNN 相同但独立定义，避免模块间循环依赖
// Conv(in,32,3,1,1)+BN+ReLU+Pool → Conv(32,64)+BN+ReLU+Pool → Conv(64,128)+BN+ReLU+Pool → Conv(128,256)+BN+ReLU
// 输出 [B, 256, H/8, W/8]
class DetFeatureExtractorCNN : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入通道数，灰度 1，RGB 3
    DetFeatureExtractorCNN(int nInChannels = 1)
        : m_conv1(nInChannels, 32, 3, 1, 1),  // 20260322 ZJH 第1层卷积
          m_bn1(32),                            // 20260322 ZJH 第1层 BN
          m_relu1(),                             // 20260322 ZJH 第1层 ReLU
          m_pool1(2),                            // 20260322 ZJH 第1层池化
          m_conv2(32, 64, 3, 1, 1),             // 20260322 ZJH 第2层卷积
          m_bn2(64),                             // 20260322 ZJH 第2层 BN
          m_relu2(),                             // 20260322 ZJH 第2层 ReLU
          m_pool2(2),                            // 20260322 ZJH 第2层池化
          m_conv3(64, 128, 3, 1, 1),            // 20260322 ZJH 第3层卷积
          m_bn3(128),                            // 20260322 ZJH 第3层 BN
          m_relu3(),                             // 20260322 ZJH 第3层 ReLU
          m_pool3(2),                            // 20260322 ZJH 第3层池化
          m_conv4(128, 256, 3, 1, 1),           // 20260322 ZJH 第4层卷积
          m_bn4(256),                            // 20260322 ZJH 第4层 BN
          m_relu4()                              // 20260322 ZJH 第4层 ReLU
    {
        // 20260322 ZJH 注册子模块以递归管理参数和训练/评估模式
        registerModule("conv1", std::make_shared<Conv2d>(m_conv1));
        registerModule("bn1",   std::make_shared<BatchNorm2d>(m_bn1));
        registerModule("conv2", std::make_shared<Conv2d>(m_conv2));
        registerModule("bn2",   std::make_shared<BatchNorm2d>(m_bn2));
        registerModule("conv3", std::make_shared<Conv2d>(m_conv3));
        registerModule("bn3",   std::make_shared<BatchNorm2d>(m_bn3));
        registerModule("conv4", std::make_shared<Conv2d>(m_conv4));
        registerModule("bn4",   std::make_shared<BatchNorm2d>(m_bn4));
    }

    // 20260322 ZJH forward — 前向传播
    // input: [B, C, H, W]
    // 返回: [B, 256, H/8, W/8]
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 第1层：Conv+BN+ReLU+Pool
        Tensor x = m_conv1.forward(input);   // 20260322 ZJH [B, 32, H, W]
        x = m_bn1.forward(x);                // 20260322 ZJH BN
        x = m_relu1.forward(x);              // 20260322 ZJH ReLU
        x = m_pool1.forward(x);              // 20260322 ZJH [B, 32, H/2, W/2]

        // 20260322 ZJH 第2层：Conv+BN+ReLU+Pool
        x = m_conv2.forward(x);              // 20260322 ZJH [B, 64, H/2, W/2]
        x = m_bn2.forward(x);
        x = m_relu2.forward(x);
        x = m_pool2.forward(x);              // 20260322 ZJH [B, 64, H/4, W/4]

        // 20260322 ZJH 第3层：Conv+BN+ReLU+Pool
        x = m_conv3.forward(x);              // 20260322 ZJH [B, 128, H/4, W/4]
        x = m_bn3.forward(x);
        x = m_relu3.forward(x);
        x = m_pool3.forward(x);              // 20260322 ZJH [B, 128, H/8, W/8]

        // 20260322 ZJH 第4层：Conv+BN+ReLU
        x = m_conv4.forward(x);              // 20260322 ZJH [B, 256, H/8, W/8]
        x = m_bn4.forward(x);
        x = m_relu4.forward(x);

        return x;  // 20260322 ZJH 返回特征图 [B, 256, H/8, W/8]
    }

private:
    Conv2d m_conv1;       // 20260322 ZJH 第1层卷积
    BatchNorm2d m_bn1;    // 20260322 ZJH 第1层 BN
    ReLU m_relu1;          // 20260322 ZJH 第1层 ReLU
    MaxPool2d m_pool1;     // 20260322 ZJH 第1层池化

    Conv2d m_conv2;       // 20260322 ZJH 第2层卷积
    BatchNorm2d m_bn2;    // 20260322 ZJH 第2层 BN
    ReLU m_relu2;          // 20260322 ZJH 第2层 ReLU
    MaxPool2d m_pool2;     // 20260322 ZJH 第2层池化

    Conv2d m_conv3;       // 20260322 ZJH 第3层卷积
    BatchNorm2d m_bn3;    // 20260322 ZJH 第3层 BN
    ReLU m_relu3;          // 20260322 ZJH 第3层 ReLU
    MaxPool2d m_pool3;     // 20260322 ZJH 第3层池化

    Conv2d m_conv4;       // 20260322 ZJH 第4层卷积
    BatchNorm2d m_bn4;    // 20260322 ZJH 第4层 BN
    ReLU m_relu4;          // 20260322 ZJH 第4层 ReLU
};

// 20260322 ZJH ZeroShotObjectDetector — 零样本目标检测器（模板匹配方式）
// 使用方式：
// 1. registerTemplate() 注册参考模板图像（每个类别可注册多个模板）
// 2. detect() 对测试图像执行检测，返回所有匹配结果
// 检测原理：
// - 对模板提取全局特征向量（GAP 池化到 [256] 维）
// - 对测试图像提取特征图 [256, H', W']
// - 在特征图的每个空间位置提取 [256] 维向量，与模板特征计算余弦相似度
// - 超过阈值的位置生成检测框，最后 NMS 去重
class ZeroShotObjectDetector {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数
    ZeroShotObjectDetector(int nInChannels = 1)
        : m_extractor(nInChannels), m_nInChannels(nInChannels) {
        // 20260322 ZJH 设置特征提取器为评估模式
        m_extractor.eval();
    }

    // 20260322 ZJH registerTemplate — 注册参考模板图像
    // 提取每个模板的全局特征向量（GAP），并取平均作为该类别的代表特征
    // nClassId: 类别 ID
    // vecTemplates: 该类别的模板图像列表，每个形状 [1, C, H, W]
    void registerTemplate(int nClassId, const std::vector<Tensor>& vecTemplates) {
        // 20260322 ZJH 检查输入有效性
        if (vecTemplates.empty()) {
            throw std::invalid_argument("ZeroShotObjectDetector::registerTemplate — empty template list");
        }

        // 20260322 ZJH 提取所有模板的特征并取平均
        // 首先提取第一个模板确定通道数
        Tensor firstFeat = m_extractor.forward(vecTemplates[0]);  // 20260322 ZJH [1, 256, H', W']
        int nC = firstFeat.shape(1);  // 20260322 ZJH 通道数 = 256

        // 20260322 ZJH 创建平均特征向量 [256]
        std::vector<float> vecAvgFeature(static_cast<size_t>(nC), 0.0f);  // 20260322 ZJH 初始化为 0

        // 20260322 ZJH 遍历所有模板，提取特征并做 GAP（全局平均池化）
        for (size_t i = 0; i < vecTemplates.size(); ++i) {
            Tensor feat = m_extractor.forward(vecTemplates[i]);  // 20260322 ZJH [1, 256, H', W']
            Tensor cFeat = feat.contiguous();                     // 20260322 ZJH 确保连续内存
            const float* pFeat = cFeat.floatDataPtr();            // 20260322 ZJH 数据指针

            int nFeatH = cFeat.shape(2);  // 20260322 ZJH 特征图高度
            int nFeatW = cFeat.shape(3);  // 20260322 ZJH 特征图宽度
            int nSpatial = nFeatH * nFeatW;  // 20260322 ZJH 空间总像素数

            // 20260322 ZJH 对每个通道做全局平均池化（GAP）
            for (int c = 0; c < nC; ++c) {
                float fSum = 0.0f;  // 20260322 ZJH 通道内像素值累加
                for (int s = 0; s < nSpatial; ++s) {
                    fSum += pFeat[c * nSpatial + s];  // 20260322 ZJH 累加 [c, h, w] 位置的值
                }
                // 20260322 ZJH 累加到平均特征向量
                vecAvgFeature[static_cast<size_t>(c)] += fSum / static_cast<float>(nSpatial);
            }
        }

        // 20260322 ZJH 取平均（除以模板数）
        float fTemplateCount = static_cast<float>(vecTemplates.size());  // 20260322 ZJH 模板数量
        for (size_t c = 0; c < vecAvgFeature.size(); ++c) {
            vecAvgFeature[c] /= fTemplateCount;  // 20260322 ZJH 除以模板数得到平均特征
        }

        // 20260322 ZJH L2 归一化特征向量
        float fNorm = 0.0f;  // 20260322 ZJH L2 范数
        for (float f : vecAvgFeature) {
            fNorm += f * f;  // 20260322 ZJH 累加平方
        }
        fNorm = std::sqrt(fNorm + 1e-8f);  // 20260322 ZJH 求平方根 + eps 防零
        for (size_t c = 0; c < vecAvgFeature.size(); ++c) {
            vecAvgFeature[c] /= fNorm;  // 20260322 ZJH 归一化
        }

        // 20260322 ZJH 存储到模板信息列表
        // 将 vector<float> 转为 Tensor [256]
        Tensor featTensor = Tensor::fromData(vecAvgFeature.data(),
                                              {nC});  // 20260322 ZJH [256] 归一化特征向量

        TemplateInfo info;               // 20260322 ZJH 创建模板信息
        info.nClassId = nClassId;        // 20260322 ZJH 类别 ID
        info.features = featTensor;      // 20260322 ZJH 归一化特征向量
        info.nFeatDim = nC;              // 20260322 ZJH 特征维度
        m_vecTemplates.push_back(info);  // 20260322 ZJH 添加到模板列表
    }

    // 20260322 ZJH detect — 执行目标检测
    // 在测试图像的特征图上滑动，每个位置提取特征向量与所有模板计算余弦相似度
    // testImage: [1, C, H, W] 测试图像张量
    // fScoreThreshold: 匹配分数阈值，默认 0.5
    // 返回: 检测结果列表（已经过 NMS 去重）
    std::vector<DetectionResult> detect(const Tensor& testImage, float fScoreThreshold = 0.5f) {
        // 20260322 ZJH 检查是否已注册模板
        if (m_vecTemplates.empty()) {
            return {};  // 20260322 ZJH 无模板则返回空结果
        }

        // 20260322 ZJH 提取测试图像特征图
        Tensor feat = m_extractor.forward(testImage);  // 20260322 ZJH [1, 256, H', W']
        Tensor cFeat = feat.contiguous();               // 20260322 ZJH 确保连续内存
        const float* pFeat = cFeat.floatDataPtr();       // 20260322 ZJH 数据指针

        int nFeatH = cFeat.shape(2);   // 20260322 ZJH 特征图高度
        int nFeatW = cFeat.shape(3);   // 20260322 ZJH 特征图宽度
        int nSpatial = nFeatH * nFeatW;  // 20260322 ZJH 空间总像素数

        // 20260322 ZJH 获取原始图像尺寸用于坐标归一化
        int nImgH = testImage.shape(2);  // 20260322 ZJH 原始图像高度
        int nImgW = testImage.shape(3);  // 20260322 ZJH 原始图像宽度

        // 20260322 ZJH 计算每个特征图位置对应的感受野大小（粗略估计）
        // 经过 3 次 2x 池化，特征图每个位置对应原图 8x8 区域
        float fCellH = static_cast<float>(nImgH) / static_cast<float>(nFeatH);  // 20260322 ZJH 每个特征位置对应的原图高度
        float fCellW = static_cast<float>(nImgW) / static_cast<float>(nFeatW);  // 20260322 ZJH 每个特征位置对应的原图宽度

        // 20260322 ZJH 对特征图的每个空间位置和每个模板计算相似度
        std::vector<DetectionResult> vecResults;  // 20260322 ZJH 候选检测结果

        for (int h = 0; h < nFeatH; ++h) {
            for (int w = 0; w < nFeatW; ++w) {
                // 20260322 ZJH 提取当前位置的特征向量 [256]
                // 特征布局: [C, H', W']，当前位置线性索引 = c * H'*W' + h * W' + w
                // 遍历所有注册模板
                for (size_t t = 0; t < m_vecTemplates.size(); ++t) {
                    const TemplateInfo& tplInfo = m_vecTemplates[t];        // 20260322 ZJH 当前模板信息
                    const float* pTemplate = tplInfo.features.floatDataPtr();  // 20260322 ZJH 模板特征指针
                    int nDim = tplInfo.nFeatDim;                              // 20260322 ZJH 特征维度

                    // 20260322 ZJH 计算余弦相似度
                    float fDot = 0.0f;    // 20260322 ZJH 点积
                    float fNormA = 0.0f;  // 20260322 ZJH 测试特征 L2 范数
                    for (int c = 0; c < nDim; ++c) {
                        float fVal = pFeat[c * nSpatial + h * nFeatW + w];  // 20260322 ZJH 测试特征 [c, h, w]
                        fDot += fVal * pTemplate[c];                         // 20260322 ZJH 点积累加
                        fNormA += fVal * fVal;                               // 20260322 ZJH 范数累加
                    }
                    fNormA = std::sqrt(fNormA + 1e-8f);  // 20260322 ZJH 测试特征 L2 范数

                    // 20260322 ZJH 余弦相似度 = dot / (||a|| * ||b||)
                    // 模板特征已归一化，||b|| = 1
                    float fSimilarity = fDot / fNormA;  // 20260322 ZJH 余弦相似度

                    // 20260322 ZJH 将余弦相似度映射到 [0, 1] 范围
                    // 余弦相似度范围 [-1, 1]，映射为 (sim + 1) / 2
                    float fScore = (fSimilarity + 1.0f) * 0.5f;  // 20260322 ZJH 归一化到 [0, 1]

                    // 20260322 ZJH 超过阈值则生成检测框
                    if (fScore >= fScoreThreshold) {
                        DetectionResult det;  // 20260322 ZJH 创建检测结果

                        // 20260322 ZJH 计算边界框归一化坐标
                        // 检测框中心在 (h * cellH + cellH/2, w * cellW + cellW/2)
                        // 框大小为 cellH * 2 x cellW * 2（覆盖周围 2x2 个 cell）
                        float fBoxW = fCellW * 2.0f;  // 20260322 ZJH 框宽度
                        float fBoxH = fCellH * 2.0f;  // 20260322 ZJH 框高度
                        float fCenterX = static_cast<float>(w) * fCellW + fCellW * 0.5f;  // 20260322 ZJH 中心 X
                        float fCenterY = static_cast<float>(h) * fCellH + fCellH * 0.5f;  // 20260322 ZJH 中心 Y

                        // 20260322 ZJH 转为左上角坐标并归一化
                        det.fX = std::max(0.0f, (fCenterX - fBoxW * 0.5f) / static_cast<float>(nImgW));
                        det.fY = std::max(0.0f, (fCenterY - fBoxH * 0.5f) / static_cast<float>(nImgH));
                        det.fW = std::min(fBoxW / static_cast<float>(nImgW), 1.0f - det.fX);
                        det.fH = std::min(fBoxH / static_cast<float>(nImgH), 1.0f - det.fY);
                        det.fScore = fScore;              // 20260322 ZJH 匹配分数
                        det.nClassId = tplInfo.nClassId;  // 20260322 ZJH 类别 ID

                        vecResults.push_back(det);  // 20260322 ZJH 添加候选框
                    }
                }
            }
        }

        // 20260322 ZJH 执行 NMS 去重
        return nms(vecResults, 0.5f);  // 20260322 ZJH IoU 阈值 0.5
    }

    // 20260322 ZJH nms — 非极大值抑制
    // 按分数降序排列，依次保留最高分框并移除 IoU 超过阈值的重叠框
    // vecResults: 候选检测结果列表（会被排序修改）
    // fIoUThreshold: IoU 阈值，默认 0.5
    // 返回: NMS 后的检测结果
    static std::vector<DetectionResult> nms(std::vector<DetectionResult>& vecResults,
                                             float fIoUThreshold = 0.5f) {
        // 20260322 ZJH 按分数降序排序
        std::sort(vecResults.begin(), vecResults.end(),
                  [](const DetectionResult& a, const DetectionResult& b) {
                      return a.fScore > b.fScore;  // 20260322 ZJH 分数高的在前
                  });

        // 20260322 ZJH 标记是否被抑制
        std::vector<bool> vecSuppressed(vecResults.size(), false);  // 20260322 ZJH 抑制标记
        std::vector<DetectionResult> vecKept;  // 20260322 ZJH 保留的检测结果

        for (size_t i = 0; i < vecResults.size(); ++i) {
            if (vecSuppressed[i]) {
                continue;  // 20260322 ZJH 已被抑制，跳过
            }

            vecKept.push_back(vecResults[i]);  // 20260322 ZJH 保留当前最高分框

            // 20260322 ZJH 抑制与当前框 IoU 超过阈值的后续框
            for (size_t j = i + 1; j < vecResults.size(); ++j) {
                if (vecSuppressed[j]) {
                    continue;  // 20260322 ZJH 已被抑制，跳过
                }

                // 20260322 ZJH 仅对同类别执行 NMS
                if (vecResults[i].nClassId != vecResults[j].nClassId) {
                    continue;  // 20260322 ZJH 不同类别不做 NMS
                }

                // 20260322 ZJH 计算 IoU
                float fIoU = computeIoU(vecResults[i], vecResults[j]);  // 20260322 ZJH 交并比
                if (fIoU > fIoUThreshold) {
                    vecSuppressed[j] = true;  // 20260322 ZJH 抑制重叠框
                }
            }
        }

        return vecKept;  // 20260322 ZJH 返回 NMS 后的结果
    }

    // 20260322 ZJH templateCount — 返回已注册的模板数量
    size_t templateCount() const {
        return m_vecTemplates.size();  // 20260322 ZJH 返回模板列表大小
    }

    // 20260322 ZJH clearTemplates — 清除所有注册的模板
    void clearTemplates() {
        m_vecTemplates.clear();  // 20260322 ZJH 清空模板列表
    }

private:
    // 20260322 ZJH TemplateInfo — 模板信息内部结构
    struct TemplateInfo {
        int nClassId = 0;        // 20260322 ZJH 类别 ID
        Tensor features;         // 20260322 ZJH 归一化特征向量 [256]
        int nFeatDim = 256;      // 20260322 ZJH 特征维度
    };

    // 20260322 ZJH 检测用特征提取 CNN
    DetFeatureExtractorCNN m_extractor;

    // 20260322 ZJH 输入通道数
    int m_nInChannels = 1;

    // 20260322 ZJH 已注册的模板信息列表
    std::vector<TemplateInfo> m_vecTemplates;

    // 20260322 ZJH computeIoU — 计算两个检测框的交并比（IoU）
    // 输入坐标为归一化坐标 (x, y, w, h)
    // a: 检测框 A
    // b: 检测框 B
    // 返回: IoU 值 [0, 1]
    static float computeIoU(const DetectionResult& a, const DetectionResult& b) {
        // 20260322 ZJH 计算交集区域
        float fX1 = std::max(a.fX, b.fX);                     // 20260322 ZJH 交集左边界
        float fY1 = std::max(a.fY, b.fY);                     // 20260322 ZJH 交集上边界
        float fX2 = std::min(a.fX + a.fW, b.fX + b.fW);      // 20260322 ZJH 交集右边界
        float fY2 = std::min(a.fY + a.fH, b.fY + b.fH);      // 20260322 ZJH 交集下边界

        // 20260322 ZJH 交集面积（无交集时为 0）
        float fInterW = std::max(0.0f, fX2 - fX1);  // 20260322 ZJH 交集宽度
        float fInterH = std::max(0.0f, fY2 - fY1);  // 20260322 ZJH 交集高度
        float fInterArea = fInterW * fInterH;         // 20260322 ZJH 交集面积

        // 20260322 ZJH 并集面积 = A + B - 交集
        float fAreaA = a.fW * a.fH;                    // 20260322 ZJH 框 A 面积
        float fAreaB = b.fW * b.fH;                    // 20260322 ZJH 框 B 面积
        float fUnionArea = fAreaA + fAreaB - fInterArea;  // 20260322 ZJH 并集面积

        // 20260322 ZJH 返回 IoU，防除零
        if (fUnionArea < 1e-8f) {
            return 0.0f;  // 20260322 ZJH 面积为零时 IoU = 0
        }
        return fInterArea / fUnionArea;  // 20260322 ZJH 返回交并比
    }
};

}  // namespace om
