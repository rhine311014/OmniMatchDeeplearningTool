// 20260330 ZJH 推理增强引擎 — 对标 Cognex + Keyence
// Ensemble 集成学习 + TTA 测试时增强 + 多尺度推理 + VAE 无监督定位 + WBF 框融合
// 五大组件协同提升推理精度，工业级质量对标商用视觉系统
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <random>
#include <cassert>
#include <utility>

export module om.engine.inference_enhance;

// 20260330 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260330 ZJH DetectionBox — 推理增强模块内部使用的检测框结构
// 与 NmsUtils.h 中的 DetectionBox 保持一致的接口
// =========================================================
struct InferDetectionBox {
    float fX1;         // 20260330 ZJH 左上角 x 坐标
    float fY1;         // 20260330 ZJH 左上角 y 坐标
    float fX2;         // 20260330 ZJH 右下角 x 坐标
    float fY2;         // 20260330 ZJH 右下角 y 坐标
    float fScore;      // 20260330 ZJH 置信度分数 [0, 1]
    int nClassId;      // 20260330 ZJH 类别 ID（-1 表示未分类）

    // 20260330 ZJH 计算面积
    float area() const {
        float fW = fX2 - fX1;  // 20260330 ZJH 宽度
        float fH = fY2 - fY1;  // 20260330 ZJH 高度
        return (fW > 0.0f && fH > 0.0f) ? fW * fH : 0.0f;  // 20260330 ZJH 无效框返回 0
    }
};

// =========================================================
// 20260330 ZJH 辅助函数：IoU 计算
// =========================================================

// 20260330 ZJH computeInferIoU — 计算两个检测框的 IoU（交并比）
// a: 第一个检测框
// b: 第二个检测框
// 返回: IoU 值 [0, 1]
inline float computeInferIoU(const InferDetectionBox& a, const InferDetectionBox& b) {
    // 20260330 ZJH 计算交集区域坐标
    float fInterX1 = std::max(a.fX1, b.fX1);  // 20260330 ZJH 交集左边界
    float fInterY1 = std::max(a.fY1, b.fY1);  // 20260330 ZJH 交集上边界
    float fInterX2 = std::min(a.fX2, b.fX2);  // 20260330 ZJH 交集右边界
    float fInterY2 = std::min(a.fY2, b.fY2);  // 20260330 ZJH 交集下边界

    // 20260330 ZJH 交集面积（无重叠时为 0）
    float fInterArea = std::max(0.0f, fInterX2 - fInterX1) *
                       std::max(0.0f, fInterY2 - fInterY1);

    // 20260330 ZJH 各自面积
    float fAreaA = a.area();  // 20260330 ZJH 框 A 面积
    float fAreaB = b.area();  // 20260330 ZJH 框 B 面积

    // 20260330 ZJH 并集面积 = A + B - 交集
    float fUnionArea = fAreaA + fAreaB - fInterArea;

    // 20260330 ZJH 防止除以零
    if (fUnionArea <= 0.0f) return 0.0f;

    return fInterArea / fUnionArea;  // 20260330 ZJH 返回 IoU
}

// =========================================================
// 20260330 ZJH 辅助函数：标准 NMS（非极大值抑制）
// =========================================================

// 20260330 ZJH applyNMS — 标准非极大值抑制
// vecBoxes: 候选检测框列表
// fIouThresh: IoU 阈值，超过则抑制
// 返回: 保留的检测框列表
inline std::vector<InferDetectionBox> applyNMS(
    std::vector<InferDetectionBox> vecBoxes, float fIouThresh)
{
    // 20260330 ZJH 按置信度降序排序
    std::sort(vecBoxes.begin(), vecBoxes.end(),
              [](const InferDetectionBox& a, const InferDetectionBox& b) {
                  return a.fScore > b.fScore;  // 20260330 ZJH 高分优先
              });

    std::vector<bool> vecSuppressed(vecBoxes.size(), false);  // 20260330 ZJH 抑制标志数组
    std::vector<InferDetectionBox> vecResult;  // 20260330 ZJH 保留的框

    // 20260330 ZJH 逐框检查，保留未被抑制的高分框
    for (size_t i = 0; i < vecBoxes.size(); ++i) {
        if (vecSuppressed[i]) continue;  // 20260330 ZJH 已被抑制，跳过
        vecResult.push_back(vecBoxes[i]);  // 20260330 ZJH 保留当前框
        // 20260330 ZJH 抑制与当前框 IoU 超过阈值的后续框
        for (size_t j = i + 1; j < vecBoxes.size(); ++j) {
            if (vecSuppressed[j]) continue;  // 20260330 ZJH 已被抑制
            float fIoU = computeInferIoU(vecBoxes[i], vecBoxes[j]);  // 20260330 ZJH 计算交并比
            if (fIoU > fIouThresh) {
                vecSuppressed[j] = true;  // 20260330 ZJH 抑制重叠框
            }
        }
    }
    return vecResult;  // 20260330 ZJH 返回 NMS 后的检测框
}

// =========================================================
// 20260330 ZJH 辅助函数：图像增强变换（Tensor 级别操作）
// =========================================================

// 20260330 ZJH flipHorizontal — 水平翻转张量
// input: [N, C, H, W]
// 返回: 水平翻转后的张量
inline Tensor flipHorizontal(const Tensor& input) {
    auto cInput = input.contiguous();  // 20260330 ZJH 确保内存连续
    int nBatch = cInput.shape(0);      // 20260330 ZJH 批次大小
    int nC = cInput.shape(1);          // 20260330 ZJH 通道数
    int nH = cInput.shape(2);          // 20260330 ZJH 高度
    int nW = cInput.shape(3);          // 20260330 ZJH 宽度

    auto result = Tensor::zeros(cInput.shapeVec());  // 20260330 ZJH 分配输出张量
    const float* pIn = cInput.floatDataPtr();        // 20260330 ZJH 输入只读指针
    float* pOut = result.mutableFloatDataPtr();       // 20260330 ZJH 输出可写指针

    // 20260330 ZJH 逐批次、逐通道、逐行翻转像素列顺序
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nC; ++c) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    int nSrcIdx = ((b * nC + c) * nH + h) * nW + w;                   // 20260330 ZJH 源索引
                    int nDstIdx = ((b * nC + c) * nH + h) * nW + (nW - 1 - w);        // 20260330 ZJH 目标索引（列反转）
                    pOut[nDstIdx] = pIn[nSrcIdx];  // 20260330 ZJH 拷贝翻转像素
                }
            }
        }
    }
    return result;  // 20260330 ZJH 返回翻转后张量
}

// 20260330 ZJH flipVertical — 垂直翻转张量
// input: [N, C, H, W]
// 返回: 垂直翻转后的张量
inline Tensor flipVertical(const Tensor& input) {
    auto cInput = input.contiguous();  // 20260330 ZJH 确保内存连续
    int nBatch = cInput.shape(0);
    int nC = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    auto result = Tensor::zeros(cInput.shapeVec());
    const float* pIn = cInput.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260330 ZJH 逐批次、逐通道、行顺序反转
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nC; ++c) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    int nSrcIdx = ((b * nC + c) * nH + h) * nW + w;                   // 20260330 ZJH 源索引
                    int nDstIdx = ((b * nC + c) * nH + (nH - 1 - h)) * nW + w;        // 20260330 ZJH 目标索引（行反转）
                    pOut[nDstIdx] = pIn[nSrcIdx];  // 20260330 ZJH 拷贝翻转像素
                }
            }
        }
    }
    return result;  // 20260330 ZJH 返回翻转后张量
}

// 20260330 ZJH rotate90 — 顺时针旋转90度
// input: [N, C, H, W] -> output: [N, C, W, H]
// 映射: out[b][c][w][H-1-h] = in[b][c][h][w]
inline Tensor rotate90(const Tensor& input) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nC = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    // 20260330 ZJH 旋转后形状: [N, C, W, H]（高宽互换）
    auto result = Tensor::zeros({nBatch, nC, nW, nH});
    const float* pIn = cInput.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260330 ZJH 顺时针旋转90度: (h,w) -> (w, H-1-h)
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nC; ++c) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    int nSrcIdx = ((b * nC + c) * nH + h) * nW + w;
                    int nDstIdx = ((b * nC + c) * nW + w) * nH + (nH - 1 - h);
                    pOut[nDstIdx] = pIn[nSrcIdx];
                }
            }
        }
    }
    return result;
}

// 20260330 ZJH rotate180 — 旋转180度
// input: [N, C, H, W] -> output: [N, C, H, W]
// 映射: out[b][c][H-1-h][W-1-w] = in[b][c][h][w]
inline Tensor rotate180(const Tensor& input) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nC = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    auto result = Tensor::zeros(cInput.shapeVec());
    const float* pIn = cInput.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260330 ZJH 180度旋转: (h,w) -> (H-1-h, W-1-w)
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nC; ++c) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    int nSrcIdx = ((b * nC + c) * nH + h) * nW + w;
                    int nDstIdx = ((b * nC + c) * nH + (nH - 1 - h)) * nW + (nW - 1 - w);
                    pOut[nDstIdx] = pIn[nSrcIdx];
                }
            }
        }
    }
    return result;
}

// 20260330 ZJH rotate270 — 顺时针旋转270度（逆时针90度）
// input: [N, C, H, W] -> output: [N, C, W, H]
// 映射: out[b][c][W-1-w][h] = in[b][c][h][w]
inline Tensor rotate270(const Tensor& input) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nC = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    auto result = Tensor::zeros({nBatch, nC, nW, nH});
    const float* pIn = cInput.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260330 ZJH 270度旋转: (h,w) -> (W-1-w, h)
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nC; ++c) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    int nSrcIdx = ((b * nC + c) * nH + h) * nW + w;
                    int nDstIdx = ((b * nC + c) * nW + (nW - 1 - w)) * nH + h;
                    pOut[nDstIdx] = pIn[nSrcIdx];
                }
            }
        }
    }
    return result;
}

// 20260330 ZJH resizeBilinear — 双线性插值缩放张量
// input: [N, C, H, W] -> output: [N, C, nNewH, nNewW]
// 用于多尺度推理的图像金字塔缩放
inline Tensor resizeBilinear(const Tensor& input, int nNewH, int nNewW) {
    auto cInput = input.contiguous();
    int nBatch = cInput.shape(0);
    int nC = cInput.shape(1);
    int nH = cInput.shape(2);
    int nW = cInput.shape(3);

    auto result = Tensor::zeros({nBatch, nC, nNewH, nNewW});
    const float* pIn = cInput.floatDataPtr();
    float* pOut = result.mutableFloatDataPtr();

    // 20260330 ZJH 计算缩放比例
    float fScaleH = static_cast<float>(nH) / static_cast<float>(nNewH);  // 20260330 ZJH 高度缩放因子
    float fScaleW = static_cast<float>(nW) / static_cast<float>(nNewW);  // 20260330 ZJH 宽度缩放因子

    // 20260330 ZJH 逐像素双线性插值
    for (int b = 0; b < nBatch; ++b) {
        for (int c = 0; c < nC; ++c) {
            for (int oh = 0; oh < nNewH; ++oh) {
                for (int ow = 0; ow < nNewW; ++ow) {
                    // 20260330 ZJH 映射回源坐标（中心对齐）
                    float fSrcH = (static_cast<float>(oh) + 0.5f) * fScaleH - 0.5f;
                    float fSrcW = (static_cast<float>(ow) + 0.5f) * fScaleW - 0.5f;

                    // 20260330 ZJH 边界 clamp
                    fSrcH = std::max(0.0f, std::min(fSrcH, static_cast<float>(nH - 1)));
                    fSrcW = std::max(0.0f, std::min(fSrcW, static_cast<float>(nW - 1)));

                    // 20260330 ZJH 四邻域坐标
                    int nH0 = static_cast<int>(fSrcH);
                    int nW0 = static_cast<int>(fSrcW);
                    int nH1 = std::min(nH0 + 1, nH - 1);  // 20260330 ZJH 下边界 clamp
                    int nW1 = std::min(nW0 + 1, nW - 1);  // 20260330 ZJH 右边界 clamp

                    // 20260330 ZJH 插值权重
                    float fHW = fSrcH - static_cast<float>(nH0);  // 20260330 ZJH 垂直权重
                    float fWW = fSrcW - static_cast<float>(nW0);  // 20260330 ZJH 水平权重

                    // 20260330 ZJH 四邻域像素值
                    int nBaseIdx = (b * nC + c) * nH;
                    float fV00 = pIn[(nBaseIdx + nH0) * nW + nW0];  // 20260330 ZJH 左上
                    float fV01 = pIn[(nBaseIdx + nH0) * nW + nW1];  // 20260330 ZJH 右上
                    float fV10 = pIn[(nBaseIdx + nH1) * nW + nW0];  // 20260330 ZJH 左下
                    float fV11 = pIn[(nBaseIdx + nH1) * nW + nW1];  // 20260330 ZJH 右下

                    // 20260330 ZJH 双线性插值公式
                    float fVal = fV00 * (1.0f - fHW) * (1.0f - fWW) +
                                 fV01 * (1.0f - fHW) * fWW +
                                 fV10 * fHW * (1.0f - fWW) +
                                 fV11 * fHW * fWW;

                    int nOutIdx = ((b * nC + c) * nNewH + oh) * nNewW + ow;
                    pOut[nOutIdx] = fVal;  // 20260330 ZJH 写入输出像素
                }
            }
        }
    }
    return result;  // 20260330 ZJH 返回缩放后张量
}

// 20260330 ZJH inverseFlipHorizontalBoxes — 水平翻转后的检测框坐标还原
// vecBoxes: 翻转图像上的检测框
// nImgW: 原始图像宽度
inline void inverseFlipHorizontalBoxes(std::vector<InferDetectionBox>& vecBoxes, int nImgW) {
    float fW = static_cast<float>(nImgW);
    for (auto& box : vecBoxes) {
        float fNewX1 = fW - box.fX2;  // 20260330 ZJH 新左边界 = 图像宽度 - 旧右边界
        float fNewX2 = fW - box.fX1;  // 20260330 ZJH 新右边界 = 图像宽度 - 旧左边界
        box.fX1 = fNewX1;
        box.fX2 = fNewX2;
    }
}

// 20260330 ZJH inverseFlipVerticalBoxes — 垂直翻转后的检测框坐标还原
inline void inverseFlipVerticalBoxes(std::vector<InferDetectionBox>& vecBoxes, int nImgH) {
    float fH = static_cast<float>(nImgH);
    for (auto& box : vecBoxes) {
        float fNewY1 = fH - box.fY2;
        float fNewY2 = fH - box.fY1;
        box.fY1 = fNewY1;
        box.fY2 = fNewY2;
    }
}

// 20260330 ZJH inverseScaleBoxes — 缩放推理后的检测框坐标还原到原始尺度
// fScale: 当前推理使用的缩放因子
inline void inverseScaleBoxes(std::vector<InferDetectionBox>& vecBoxes, float fScale) {
    float fInvScale = 1.0f / fScale;  // 20260330 ZJH 逆缩放因子
    for (auto& box : vecBoxes) {
        box.fX1 *= fInvScale;
        box.fY1 *= fInvScale;
        box.fX2 *= fInvScale;
        box.fY2 *= fInvScale;
    }
}

// =========================================================
// 20260330 ZJH 辅助函数：softmax 概率提取
// =========================================================

// 20260330 ZJH extractSoftmaxProbs — 从模型输出张量 [1, nClasses] 提取 softmax 概率
// output: 模型输出张量（logits 或已 softmax）
// 返回: softmax 后的概率向量
inline std::vector<float> extractSoftmaxProbs(const Tensor& output) {
    auto cOut = output.contiguous();
    int nClasses = cOut.shape(cOut.shapeVec().size() - 1);  // 20260330 ZJH 最后一维为类别数
    int nTotal = cOut.numel();                               // 20260330 ZJH 总元素数
    int nBatch = nTotal / nClasses;                          // 20260330 ZJH 批次数（取第一个）
    const float* pData = cOut.floatDataPtr();

    // 20260330 ZJH 取第一个样本的输出做 softmax
    std::vector<float> vecProbs(nClasses);
    float fMax = pData[0];
    for (int j = 1; j < nClasses; ++j) {
        if (pData[j] > fMax) fMax = pData[j];  // 20260330 ZJH 数值稳定性：先找最大值
    }
    float fSum = 0.0f;
    for (int j = 0; j < nClasses; ++j) {
        vecProbs[j] = std::exp(pData[j] - fMax);  // 20260330 ZJH exp(logit - max)
        fSum += vecProbs[j];
    }
    for (int j = 0; j < nClasses; ++j) {
        vecProbs[j] /= fSum;  // 20260330 ZJH 归一化为概率分布
    }
    return vecProbs;  // 20260330 ZJH 返回 softmax 概率向量
}

// 20260330 ZJH decodeDetectionOutput — 从模型输出张量解码检测框
// output: [N, nDetections, 6] 格式，每行 (x1, y1, x2, y2, score, classId)
// fScoreThresh: 置信度过滤阈值
// 返回: 过滤后的检测框列表
inline std::vector<InferDetectionBox> decodeDetectionOutput(const Tensor& output, float fScoreThresh = 0.25f) {
    auto cOut = output.contiguous();
    auto vecShape = cOut.shapeVec();  // 20260330 ZJH 获取输出形状
    std::vector<InferDetectionBox> vecBoxes;

    // 20260330 ZJH 支持两种格式: [N, nDets, 6] 或 [nDets, 6]
    int nDets = 0;          // 20260330 ZJH 检测框总数
    int nPerDet = 0;        // 20260330 ZJH 每个检测框的元素数
    const float* pData = cOut.floatDataPtr();

    if (vecShape.size() == 3) {
        nDets = vecShape[1];    // 20260330 ZJH [N, nDets, 6] 格式
        nPerDet = vecShape[2];  // 20260330 ZJH 每框 6 个值
    } else if (vecShape.size() == 2) {
        nDets = vecShape[0];    // 20260330 ZJH [nDets, 6] 格式
        nPerDet = vecShape[1];
    } else {
        return vecBoxes;  // 20260330 ZJH 不支持的格式，返回空
    }

    // 20260330 ZJH 确保每框至少 6 个值
    if (nPerDet < 6) return vecBoxes;

    // 20260330 ZJH 解码每个检测框
    for (int i = 0; i < nDets; ++i) {
        float fX1 = pData[i * nPerDet + 0];       // 20260330 ZJH 左上 x
        float fY1 = pData[i * nPerDet + 1];       // 20260330 ZJH 左上 y
        float fX2 = pData[i * nPerDet + 2];       // 20260330 ZJH 右下 x
        float fY2 = pData[i * nPerDet + 3];       // 20260330 ZJH 右下 y
        float fScore = pData[i * nPerDet + 4];    // 20260330 ZJH 置信度
        int nClassId = static_cast<int>(pData[i * nPerDet + 5]);  // 20260330 ZJH 类别 ID

        // 20260330 ZJH 过滤低置信度检测
        if (fScore >= fScoreThresh) {
            vecBoxes.push_back({fX1, fY1, fX2, fY2, fScore, nClassId});
        }
    }
    return vecBoxes;  // 20260330 ZJH 返回解码后的检测框
}

// 20260330 ZJH 前向声明 weightedBoxFusion（定义在文件末尾，EnsemblePredictor::detectEnsemble 需要调用）
std::vector<InferDetectionBox> weightedBoxFusion(
    const std::vector<std::vector<InferDetectionBox>>& vecModelOutputs,
    float fIouThresh = 0.55f, float fScoreThresh = 0.001f);

// =========================================================
// 1. Ensemble 集成学习
// 20260330 ZJH 多模型加权融合推理，对标 Cognex ViDi 多模型投票
// 支持分类加权投票和检测 WBF（Weighted Box Fusion）融合
// =========================================================
class EnsemblePredictor {
public:
    // 20260330 ZJH ModelEntry — 内部模型条目，存储模型指针和融合权重
    struct ModelEntry {
        std::shared_ptr<Module> pModel;  // 20260330 ZJH 模型指针
        float fWeight;                    // 20260330 ZJH 融合权重（>0）
    };

    // 20260330 ZJH EnsembleResult — 分类集成结果
    struct EnsembleResult {
        int nClassId;                    // 20260330 ZJH 预测类别 ID（投票最高分）
        float fConfidence;               // 20260330 ZJH 最高类别的融合置信度
        std::vector<float> vecProbs;     // 20260330 ZJH 融合后的完整概率分布
    };

    // 20260330 ZJH addModel — 添加一个模型到集成预测器
    // pModel: 训练好的模型（shared_ptr 共享所有权）
    // fWeight: 融合权重，默认 1.0（等权融合）
    void addModel(std::shared_ptr<Module> pModel, float fWeight = 1.0f) {
        // 20260330 ZJH 确保权重为正值
        if (fWeight <= 0.0f) fWeight = 1.0f;
        m_vecModels.push_back({pModel, fWeight});  // 20260330 ZJH 加入模型列表
    }

    // 20260330 ZJH classifyEnsemble — 分类任务的加权投票集成
    // 将每个模型的 softmax 概率按权重加权平均，取最高概率类别
    // input: [1, C, H, W] 输入图像张量（单张）
    // 返回: EnsembleResult 包含融合后的分类结果和概率分布
    EnsembleResult classifyEnsemble(const Tensor& input) {
        EnsembleResult result;
        result.nClassId = -1;     // 20260330 ZJH 初始化为无效类别
        result.fConfidence = 0.0f;

        if (m_vecModels.empty()) return result;  // 20260330 ZJH 无模型则返回空

        // 20260330 ZJH 计算总权重用于归一化
        float fTotalWeight = 0.0f;
        for (const auto& entry : m_vecModels) {
            fTotalWeight += entry.fWeight;
        }
        if (fTotalWeight <= 0.0f) fTotalWeight = 1.0f;  // 20260330 ZJH 防除零

        // 20260330 ZJH 对每个模型推理并收集 softmax 概率
        std::vector<float> vecFusedProbs;  // 20260330 ZJH 融合概率（加权求和后归一化）
        bool bFirstModel = true;

        for (const auto& entry : m_vecModels) {
            entry.pModel->eval();  // 20260330 ZJH 确保模型在评估模式
            Tensor output = entry.pModel->forward(input);  // 20260330 ZJH 前向推理
            std::vector<float> vecProbs = extractSoftmaxProbs(output);  // 20260330 ZJH 提取 softmax 概率

            if (bFirstModel) {
                // 20260330 ZJH 第一个模型，初始化融合概率数组
                vecFusedProbs.resize(vecProbs.size(), 0.0f);
                bFirstModel = false;
            }

            // 20260330 ZJH 加权累加概率
            float fNormWeight = entry.fWeight / fTotalWeight;  // 20260330 ZJH 归一化权重
            int nClasses = static_cast<int>(std::min(vecProbs.size(), vecFusedProbs.size()));
            for (int j = 0; j < nClasses; ++j) {
                vecFusedProbs[j] += vecProbs[j] * fNormWeight;  // 20260330 ZJH 加权累加
            }
        }

        // 20260330 ZJH 找最高概率类别
        result.vecProbs = vecFusedProbs;
        int nBestClass = 0;
        float fBestProb = vecFusedProbs.empty() ? 0.0f : vecFusedProbs[0];
        for (int j = 1; j < static_cast<int>(vecFusedProbs.size()); ++j) {
            if (vecFusedProbs[j] > fBestProb) {
                fBestProb = vecFusedProbs[j];  // 20260330 ZJH 更新最高概率
                nBestClass = j;                 // 20260330 ZJH 更新最佳类别
            }
        }
        result.nClassId = nBestClass;
        result.fConfidence = fBestProb;

        return result;  // 20260330 ZJH 返回集成分类结果
    }

    // 20260330 ZJH detectEnsemble — 检测任务的集成融合
    // 合并所有模型的检测框，然后使用 WBF（Weighted Box Fusion）融合
    // input: [1, C, H, W] 输入图像张量
    // fIouThresh: WBF 融合的 IoU 阈值，默认 0.5
    // 返回: WBF 融合后的检测框列表
    std::vector<InferDetectionBox> detectEnsemble(const Tensor& input, float fIouThresh = 0.5f) {
        if (m_vecModels.empty()) return {};  // 20260330 ZJH 无模型返回空

        // 20260330 ZJH 收集所有模型的检测输出
        std::vector<std::vector<InferDetectionBox>> vecAllOutputs;
        vecAllOutputs.reserve(m_vecModels.size());

        for (const auto& entry : m_vecModels) {
            entry.pModel->eval();  // 20260330 ZJH 确保评估模式
            Tensor output = entry.pModel->forward(input);  // 20260330 ZJH 前向推理
            auto vecBoxes = decodeDetectionOutput(output);  // 20260330 ZJH 解码检测框
            vecAllOutputs.push_back(vecBoxes);
        }

        // 20260330 ZJH 使用 WBF 融合所有模型的检测框
        return weightedBoxFusion(vecAllOutputs, fIouThresh);
    }

    // 20260330 ZJH getModelCount — 获取集成模型数量
    int getModelCount() const {
        return static_cast<int>(m_vecModels.size());
    }

private:
    std::vector<ModelEntry> m_vecModels;  // 20260330 ZJH 模型列表（按添加顺序）
};

// =========================================================
// 2. TTA (Test-Time Augmentation) 测试时增强
// 20260330 ZJH 推理时通过多种数据增强变换进行多次推理并融合结果
// 对标 Keyence 的 "多角度检测" 模式
// 支持水平翻转、垂直翻转、90度旋转、多尺度缩放
// =========================================================
class TTAPredictor {
public:
    // 20260330 ZJH TTAConfig — TTA 配置参数
    struct TTAConfig {
        bool bHFlip = true;          // 20260330 ZJH 是否使用水平翻转增强
        bool bVFlip = false;         // 20260330 ZJH 是否使用垂直翻转增强
        bool bRotate90 = false;      // 20260330 ZJH 是否使用90度旋转增强（0/90/180/270 四个方向）
        bool bMultiScale = false;    // 20260330 ZJH 是否使用多尺度增强（0.8x, 1.0x, 1.2x）
        int nNumAugments = 4;        // 20260330 ZJH 增强总数上限（含原图）
    };

    // 20260330 ZJH classifyTTA — 分类任务的 TTA 增强推理
    // 对输入图像进行多种增强变换，分别推理后平均 softmax 概率
    // model: 已训练的分类模型
    // input: [1, C, H, W] 输入图像张量
    // config: TTA 配置参数
    // 返回: 融合后的 softmax 概率向量
    std::vector<float> classifyTTA(Module& model, const Tensor& input, const TTAConfig& config) {
        model.eval();  // 20260330 ZJH 确保模型在评估模式

        // 20260330 ZJH 生成所有增强变换后的输入张量列表
        std::vector<Tensor> vecAugInputs = generateAugmentedInputs(input, config);

        // 20260330 ZJH 对每个增强输入推理并收集概率
        std::vector<float> vecFusedProbs;
        int nCount = 0;

        for (const auto& augInput : vecAugInputs) {
            Tensor output = model.forward(augInput);  // 20260330 ZJH 前向推理
            std::vector<float> vecProbs = extractSoftmaxProbs(output);  // 20260330 ZJH 提取 softmax

            if (nCount == 0) {
                vecFusedProbs.resize(vecProbs.size(), 0.0f);  // 20260330 ZJH 初始化融合数组
            }

            // 20260330 ZJH 累加概率
            int nClasses = static_cast<int>(std::min(vecProbs.size(), vecFusedProbs.size()));
            for (int j = 0; j < nClasses; ++j) {
                vecFusedProbs[j] += vecProbs[j];
            }
            ++nCount;
        }

        // 20260330 ZJH 平均概率
        if (nCount > 0) {
            float fInvCount = 1.0f / static_cast<float>(nCount);
            for (auto& fProb : vecFusedProbs) {
                fProb *= fInvCount;  // 20260330 ZJH 除以增强次数
            }
        }

        return vecFusedProbs;  // 20260330 ZJH 返回平均后的概率
    }

    // 20260330 ZJH detectTTA — 检测任务的 TTA 增强推理
    // 对输入图像进行多种增强变换，分别检测后合并框并做 NMS
    // model: 已训练的检测模型
    // input: [1, C, H, W] 输入图像张量
    // config: TTA 配置参数
    // 返回: NMS 后的融合检测框列表
    std::vector<InferDetectionBox> detectTTA(Module& model, const Tensor& input, const TTAConfig& config) {
        model.eval();  // 20260330 ZJH 确保评估模式

        int nH = input.shape(2);  // 20260330 ZJH 原始图像高度
        int nW = input.shape(3);  // 20260330 ZJH 原始图像宽度

        // 20260330 ZJH 收集所有增强变换的检测框（已还原到原始坐标系）
        std::vector<InferDetectionBox> vecAllBoxes;

        // 20260330 ZJH (1) 原图推理
        {
            Tensor output = model.forward(input);
            auto vecBoxes = decodeDetectionOutput(output);
            vecAllBoxes.insert(vecAllBoxes.end(), vecBoxes.begin(), vecBoxes.end());
        }

        int nAugCount = 1;  // 20260330 ZJH 已完成的增强次数（含原图）

        // 20260330 ZJH (2) 水平翻转增强
        if (config.bHFlip && nAugCount < config.nNumAugments) {
            Tensor flipped = flipHorizontal(input);  // 20260330 ZJH 水平翻转
            Tensor output = model.forward(flipped);
            auto vecBoxes = decodeDetectionOutput(output);
            inverseFlipHorizontalBoxes(vecBoxes, nW);  // 20260330 ZJH 还原框坐标
            vecAllBoxes.insert(vecAllBoxes.end(), vecBoxes.begin(), vecBoxes.end());
            ++nAugCount;
        }

        // 20260330 ZJH (3) 垂直翻转增强
        if (config.bVFlip && nAugCount < config.nNumAugments) {
            Tensor flipped = flipVertical(input);
            Tensor output = model.forward(flipped);
            auto vecBoxes = decodeDetectionOutput(output);
            inverseFlipVerticalBoxes(vecBoxes, nH);
            vecAllBoxes.insert(vecAllBoxes.end(), vecBoxes.begin(), vecBoxes.end());
            ++nAugCount;
        }

        // 20260330 ZJH (4) 90度旋转增强（0/90/180/270 四个方向，原图已包含 0 度）
        if (config.bRotate90 && nAugCount < config.nNumAugments) {
            // 20260330 ZJH 旋转 180 度（形状不变，可直接还原框坐标）
            Tensor rot180 = rotate180(input);
            Tensor output180 = model.forward(rot180);
            auto vecBoxes180 = decodeDetectionOutput(output180);
            // 20260330 ZJH 还原 180 度旋转的框: (x1,y1,x2,y2) -> (W-x2, H-y2, W-x1, H-y1)
            for (auto& box : vecBoxes180) {
                float fNewX1 = static_cast<float>(nW) - box.fX2;
                float fNewY1 = static_cast<float>(nH) - box.fY2;
                float fNewX2 = static_cast<float>(nW) - box.fX1;
                float fNewY2 = static_cast<float>(nH) - box.fY1;
                box.fX1 = fNewX1;  box.fY1 = fNewY1;
                box.fX2 = fNewX2;  box.fY2 = fNewY2;
            }
            vecAllBoxes.insert(vecAllBoxes.end(), vecBoxes180.begin(), vecBoxes180.end());
            ++nAugCount;
        }

        // 20260330 ZJH (5) 多尺度增强
        if (config.bMultiScale && nAugCount < config.nNumAugments) {
            std::vector<float> vecScales = {0.8f, 1.2f};  // 20260330 ZJH 除 1.0x 外的额外尺度
            for (float fScale : vecScales) {
                if (nAugCount >= config.nNumAugments) break;
                int nScaledH = static_cast<int>(static_cast<float>(nH) * fScale);
                int nScaledW = static_cast<int>(static_cast<float>(nW) * fScale);
                if (nScaledH < 1 || nScaledW < 1) continue;  // 20260330 ZJH 跳过无效尺寸
                Tensor scaled = resizeBilinear(input, nScaledH, nScaledW);
                Tensor output = model.forward(scaled);
                auto vecBoxes = decodeDetectionOutput(output);
                inverseScaleBoxes(vecBoxes, fScale);  // 20260330 ZJH 还原到原始尺度
                vecAllBoxes.insert(vecAllBoxes.end(), vecBoxes.begin(), vecBoxes.end());
                ++nAugCount;
            }
        }

        // 20260330 ZJH 对合并的所有框做 NMS 去重
        return applyNMS(vecAllBoxes, 0.5f);
    }

    // 20260330 ZJH segmentTTA — 分割任务的 TTA 增强推理
    // 对输入图像进行多种增强变换，分别做像素级预测后取平均
    // model: 已训练的分割模型
    // input: [1, C, H, W] 输入图像张量
    // config: TTA 配置参数
    // 返回: [1, nClasses, H, W] 融合后的分割概率图
    Tensor segmentTTA(Module& model, const Tensor& input, const TTAConfig& config) {
        model.eval();  // 20260330 ZJH 确保评估模式

        // 20260330 ZJH 原图推理
        Tensor fusedOutput = model.forward(input);  // 20260330 ZJH [1, nClasses, H, W]
        int nCount = 1;  // 20260330 ZJH 增强计数

        // 20260330 ZJH 水平翻转增强
        if (config.bHFlip && nCount < config.nNumAugments) {
            Tensor flipped = flipHorizontal(input);
            Tensor output = model.forward(flipped);
            Tensor unflipped = flipHorizontal(output);  // 20260330 ZJH 翻转回来对齐像素
            fusedOutput = tensorAdd(fusedOutput, unflipped);
            ++nCount;
        }

        // 20260330 ZJH 垂直翻转增强
        if (config.bVFlip && nCount < config.nNumAugments) {
            Tensor flipped = flipVertical(input);
            Tensor output = model.forward(flipped);
            Tensor unflipped = flipVertical(output);  // 20260330 ZJH 翻转回来对齐像素
            fusedOutput = tensorAdd(fusedOutput, unflipped);
            ++nCount;
        }

        // 20260330 ZJH 180度旋转增强
        if (config.bRotate90 && nCount < config.nNumAugments) {
            Tensor rot = rotate180(input);
            Tensor output = model.forward(rot);
            Tensor unrot = rotate180(output);  // 20260330 ZJH 旋转回来对齐像素
            fusedOutput = tensorAdd(fusedOutput, unrot);
            ++nCount;
        }

        // 20260330 ZJH 多尺度增强（需要缩放回原始尺寸后累加）
        if (config.bMultiScale && nCount < config.nNumAugments) {
            int nH = input.shape(2);
            int nW = input.shape(3);
            std::vector<float> vecScales = {0.8f, 1.2f};
            for (float fScale : vecScales) {
                if (nCount >= config.nNumAugments) break;
                int nScaledH = static_cast<int>(static_cast<float>(nH) * fScale);
                int nScaledW = static_cast<int>(static_cast<float>(nW) * fScale);
                if (nScaledH < 1 || nScaledW < 1) continue;
                Tensor scaled = resizeBilinear(input, nScaledH, nScaledW);
                Tensor output = model.forward(scaled);
                // 20260330 ZJH 缩放回原始尺寸
                Tensor resized = resizeBilinear(output, nH, nW);
                fusedOutput = tensorAdd(fusedOutput, resized);
                ++nCount;
            }
        }

        // 20260330 ZJH 除以增强次数取平均
        if (nCount > 1) {
            fusedOutput = tensorMulScalar(fusedOutput, 1.0f / static_cast<float>(nCount));
        }

        return fusedOutput;  // 20260330 ZJH 返回融合后的分割概率图
    }

private:
    // 20260330 ZJH generateAugmentedInputs — 生成所有增强变换后的输入张量
    // 用于分类任务 TTA，不需要坐标还原
    std::vector<Tensor> generateAugmentedInputs(const Tensor& input, const TTAConfig& config) {
        std::vector<Tensor> vecResult;
        vecResult.push_back(input);  // 20260330 ZJH 原图始终包含

        // 20260330 ZJH 水平翻转
        if (config.bHFlip && static_cast<int>(vecResult.size()) < config.nNumAugments) {
            vecResult.push_back(flipHorizontal(input));
        }
        // 20260330 ZJH 垂直翻转
        if (config.bVFlip && static_cast<int>(vecResult.size()) < config.nNumAugments) {
            vecResult.push_back(flipVertical(input));
        }
        // 20260330 ZJH 90度旋转增强（180度旋转保持尺寸不变，分类不受影响）
        if (config.bRotate90 && static_cast<int>(vecResult.size()) < config.nNumAugments) {
            vecResult.push_back(rotate180(input));  // 20260330 ZJH 180度旋转
        }
        // 20260330 ZJH 多尺度增强
        if (config.bMultiScale && static_cast<int>(vecResult.size()) < config.nNumAugments) {
            int nH = input.shape(2);
            int nW = input.shape(3);
            std::vector<float> vecScales = {0.8f, 1.2f};
            for (float fScale : vecScales) {
                if (static_cast<int>(vecResult.size()) >= config.nNumAugments) break;
                int nScaledH = std::max(1, static_cast<int>(static_cast<float>(nH) * fScale));
                int nScaledW = std::max(1, static_cast<int>(static_cast<float>(nW) * fScale));
                vecResult.push_back(resizeBilinear(input, nScaledH, nScaledW));
            }
        }

        return vecResult;
    }
};

// =========================================================
// 3. 多尺度推理 (Multi-Scale Inference)
// 20260330 ZJH 图像金字塔多分辨率推理，提升小目标检测精度
// 在多个缩放尺度上分别推理，合并检测框后做 NMS 去重
// 对标 Keyence IV3 系列的 "多倍率检测" 功能
// =========================================================
class MultiScalePredictor {
public:
    // 20260330 ZJH detectMultiScale — 多尺度检测推理
    // 在指定的缩放尺度列表上分别推理，将所有检测框缩放回原始坐标后合并 NMS
    // model: 已训练的检测模型
    // input: [1, C, H, W] 输入图像张量
    // vecScales: 缩放因子列表，例如 {0.75, 1.0, 1.25}
    // fNmsThresh: NMS 的 IoU 阈值，默认 0.5
    // 返回: NMS 后的检测框列表
    std::vector<InferDetectionBox> detectMultiScale(
        Module& model, const Tensor& input,
        const std::vector<float>& vecScales = {0.75f, 1.0f, 1.25f},
        float fNmsThresh = 0.5f)
    {
        model.eval();  // 20260330 ZJH 确保评估模式

        int nH = input.shape(2);  // 20260330 ZJH 原始图像高度
        int nW = input.shape(3);  // 20260330 ZJH 原始图像宽度

        std::vector<InferDetectionBox> vecAllBoxes;  // 20260330 ZJH 所有尺度的检测框集合

        // 20260330 ZJH 遍历每个缩放尺度进行推理
        for (float fScale : vecScales) {
            // 20260330 ZJH 计算缩放后的图像尺寸
            int nScaledH = std::max(1, static_cast<int>(static_cast<float>(nH) * fScale));
            int nScaledW = std::max(1, static_cast<int>(static_cast<float>(nW) * fScale));

            Tensor scaledInput;
            if (std::abs(fScale - 1.0f) < 1e-4f) {
                // 20260330 ZJH 1.0x 尺度直接使用原图，避免不必要的插值
                scaledInput = input;
            } else {
                // 20260330 ZJH 双线性插值缩放
                scaledInput = resizeBilinear(input, nScaledH, nScaledW);
            }

            // 20260330 ZJH 模型前向推理
            Tensor output = model.forward(scaledInput);

            // 20260330 ZJH 解码检测框
            auto vecBoxes = decodeDetectionOutput(output);

            // 20260330 ZJH 将检测框坐标从缩放后的坐标还原到原始坐标
            if (std::abs(fScale - 1.0f) >= 1e-4f) {
                inverseScaleBoxes(vecBoxes, fScale);
            }

            // 20260330 ZJH 合并到总集合
            vecAllBoxes.insert(vecAllBoxes.end(), vecBoxes.begin(), vecBoxes.end());
        }

        // 20260330 ZJH 对所有尺度合并的框做 NMS 去重
        return applyNMS(vecAllBoxes, fNmsThresh);
    }
};

// =========================================================
// 4. VAE 无监督定位 (ViDi Blue 等效)
// 20260330 ZJH 变分自编码器 — 学习正常图像分布，通过重建误差定位异常区域
// 对标 Cognex ViDi Blue: 仅用正常样本训练，推理时异常区域重建误差高
// 编码器: 4 层 Conv 下采样  解码器: 4 层 ConvTranspose 上采样
// 损失: 重建 MSE + KL 散度
// =========================================================
class VAELocator : public Module {
public:
    // 20260330 ZJH LocationResult — 异常定位结果
    struct LocationResult {
        std::vector<float> vecHeatmap;    // 20260330 ZJH 重建误差热力图（像素级异常分数）
        int nW;                            // 20260330 ZJH 热力图宽度
        int nH;                            // 20260330 ZJH 热力图高度
        float fAnomalyScore;              // 20260330 ZJH 全局异常分数（热力图均值）
        std::vector<std::vector<float>> vecRegions;  // 20260330 ZJH 异常区域 bbox [x1,y1,x2,y2]
    };

    // 20260330 ZJH 构造函数
    // nLatentDim: 潜在空间维度，默认 128
    // nImageSize: 输入图像尺寸（正方形），默认 128
    // nChannels: 输入通道数，默认 3（RGB）
    VAELocator(int nLatentDim = 128, int nImageSize = 128, int nChannels = 3)
        : m_nLatentDim(nLatentDim), m_nImageSize(nImageSize), m_nChannels(nChannels),
          // 20260330 ZJH 编码器: 4 层 Conv 下采样
          // 输入: [N, C, 128, 128]
          // enc1: [N, C, 128, 128] -> [N, 32, 64, 64]  (stride=2, pad=1)
          m_encConv1(nChannels, 32, 4, 2, 1, true),
          // enc2: [N, 32, 64, 64] -> [N, 64, 32, 32]
          m_encConv2(32, 64, 4, 2, 1, true),
          // enc3: [N, 64, 32, 32] -> [N, 128, 16, 16]
          m_encConv3(64, 128, 4, 2, 1, true),
          // enc4: [N, 128, 16, 16] -> [N, 256, 8, 8]
          m_encConv4(128, 256, 4, 2, 1, true),
          // 20260330 ZJH BN 层配合编码器卷积
          m_encBn1(32), m_encBn2(64), m_encBn3(128), m_encBn4(256),
          // 20260330 ZJH 编码器全连接: 256*8*8 = 16384 -> mu/logvar
          m_fcMu(256 * 8 * 8, nLatentDim, true),
          m_fcLogVar(256 * 8 * 8, nLatentDim, true),
          // 20260330 ZJH 解码器全连接: latentDim -> 256*8*8
          m_fcDecode(nLatentDim, 256 * 8 * 8, true),
          // 20260330 ZJH 解码器: 4 层 ConvTranspose 上采样
          // dec1: [N, 256, 8, 8] -> [N, 128, 16, 16]
          m_decConv1(256, 128, 4, 2, 1, true),
          // dec2: [N, 128, 16, 16] -> [N, 64, 32, 32]
          m_decConv2(128, 64, 4, 2, 1, true),
          // dec3: [N, 64, 32, 32] -> [N, 32, 64, 64]
          m_decConv3(64, 32, 4, 2, 1, true),
          // dec4: [N, 32, 64, 64] -> [N, C, 128, 128]
          m_decConv4(32, nChannels, 4, 2, 1, true),
          // 20260330 ZJH BN 层配合解码器卷积
          m_decBn1(128), m_decBn2(64), m_decBn3(32)
    {
        // 20260330 ZJH 注册所有子模块（用于递归参数收集和训练/评估模式切换）
        registerChild("enc_conv1", std::make_shared<Conv2dWrapper>(m_encConv1));
        registerChild("enc_conv2", std::make_shared<Conv2dWrapper>(m_encConv2));
        registerChild("enc_conv3", std::make_shared<Conv2dWrapper>(m_encConv3));
        registerChild("enc_conv4", std::make_shared<Conv2dWrapper>(m_encConv4));
        registerChild("enc_bn1", std::make_shared<BnWrapper>(m_encBn1));
        registerChild("enc_bn2", std::make_shared<BnWrapper>(m_encBn2));
        registerChild("enc_bn3", std::make_shared<BnWrapper>(m_encBn3));
        registerChild("enc_bn4", std::make_shared<BnWrapper>(m_encBn4));
        registerChild("fc_mu", std::make_shared<LinearWrapper>(m_fcMu));
        registerChild("fc_logvar", std::make_shared<LinearWrapper>(m_fcLogVar));
        registerChild("fc_decode", std::make_shared<LinearWrapper>(m_fcDecode));
        registerChild("dec_conv1", std::make_shared<ConvTWrapper>(m_decConv1));
        registerChild("dec_conv2", std::make_shared<ConvTWrapper>(m_decConv2));
        registerChild("dec_conv3", std::make_shared<ConvTWrapper>(m_decConv3));
        registerChild("dec_conv4", std::make_shared<ConvTWrapper>(m_decConv4));
        registerChild("dec_bn1", std::make_shared<BnWrapper>(m_decBn1));
        registerChild("dec_bn2", std::make_shared<BnWrapper>(m_decBn2));
        registerChild("dec_bn3", std::make_shared<BnWrapper>(m_decBn3));
    }

    // 20260330 ZJH encode — 编码器: 输入图像 -> (mu, logvar)
    // 4 层 Conv+BN+ReLU 下采样 + Flatten + 两个全连接分别输出 mu 和 logvar
    // input: [N, C, 128, 128]
    // 返回: (mu, logvar) 均为 [N, nLatentDim]
    std::pair<Tensor, Tensor> encode(const Tensor& input) {
        // 20260330 ZJH 编码器第 1 层: Conv(4,2,1) + BN + ReLU
        auto x = m_encConv1.forward(input);    // [N, 32, 64, 64]
        x = m_encBn1.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 编码器第 2 层: Conv(4,2,1) + BN + ReLU
        x = m_encConv2.forward(x);             // [N, 64, 32, 32]
        x = m_encBn2.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 编码器第 3 层: Conv(4,2,1) + BN + ReLU
        x = m_encConv3.forward(x);             // [N, 128, 16, 16]
        x = m_encBn3.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 编码器第 4 层: Conv(4,2,1) + BN + ReLU
        x = m_encConv4.forward(x);             // [N, 256, 8, 8]
        x = m_encBn4.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 展平: [N, 256, 8, 8] -> [N, 256*8*8=16384]
        x = m_flatten.forward(x);

        // 20260330 ZJH 全连接映射到 mu 和 logvar
        Tensor mu = m_fcMu.forward(x);           // [N, nLatentDim]
        Tensor logvar = m_fcLogVar.forward(x);    // [N, nLatentDim]

        return {mu, logvar};
    }

    // 20260330 ZJH reparameterize — 重参数化技巧
    // z = mu + eps * exp(0.5 * logvar)
    // 使采样过程可微分，梯度可以流过 mu 和 logvar
    // mu: [N, nLatentDim] 均值
    // logvar: [N, nLatentDim] 对数方差
    // 返回: z = [N, nLatentDim] 潜在向量
    Tensor reparameterize(const Tensor& mu, const Tensor& logvar) {
        // 20260330 ZJH 生成标准正态噪声 eps ~ N(0, 1)
        auto eps = Tensor::randn(mu.shapeVec());  // [N, nLatentDim]

        // 20260330 ZJH 计算 exp(0.5 * logvar) = std
        auto halfLogvar = tensorMulScalar(logvar, 0.5f);  // 20260330 ZJH 0.5 * logvar

        // 20260330 ZJH 手动计算 exp
        auto cHalf = halfLogvar.contiguous();
        auto stdTensor = Tensor::zeros(cHalf.shapeVec());
        const float* pIn = cHalf.floatDataPtr();
        float* pOut = stdTensor.mutableFloatDataPtr();
        int nTotal = cHalf.numel();
        for (int i = 0; i < nTotal; ++i) {
            pOut[i] = std::exp(pIn[i]);  // 20260330 ZJH exp(0.5 * logvar) = std
        }

        // 20260330 ZJH z = mu + eps * std
        auto epsTimesStd = tensorMul(eps, stdTensor);    // 20260330 ZJH eps * std
        auto z = tensorAdd(mu, epsTimesStd);             // 20260330 ZJH mu + eps * std

        return z;
    }

    // 20260330 ZJH decode — 解码器: 潜在向量 z -> 重建图像
    // 全连接 + reshape + 4 层 ConvTranspose+BN+ReLU 上采样 + Sigmoid
    // z: [N, nLatentDim]
    // 返回: [N, C, 128, 128] 重建图像（像素值 [0, 1]）
    Tensor decode(const Tensor& z) {
        // 20260330 ZJH 全连接: [N, nLatentDim] -> [N, 256*8*8]
        auto x = m_fcDecode.forward(z);
        x = m_relu.forward(x);

        // 20260330 ZJH reshape: [N, 256*8*8] -> [N, 256, 8, 8]
        int nBatch = x.shape(0);
        x = tensorReshape(x, {nBatch, 256, 8, 8});

        // 20260330 ZJH 解码器第 1 层: ConvTranspose(4,2,1) + BN + ReLU
        x = m_decConv1.forward(x);   // [N, 128, 16, 16]
        x = m_decBn1.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 解码器第 2 层: ConvTranspose(4,2,1) + BN + ReLU
        x = m_decConv2.forward(x);   // [N, 64, 32, 32]
        x = m_decBn2.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 解码器第 3 层: ConvTranspose(4,2,1) + BN + ReLU
        x = m_decConv3.forward(x);   // [N, 32, 64, 64]
        x = m_decBn3.forward(x);
        x = m_relu.forward(x);

        // 20260330 ZJH 解码器第 4 层: ConvTranspose(4,2,1) + Sigmoid（输出像素范围 [0,1]）
        x = m_decConv4.forward(x);   // [N, C, 128, 128]
        x = m_sigmoid.forward(x);     // 20260330 ZJH Sigmoid 激活将输出压缩到 [0,1]

        return x;
    }

    // 20260330 ZJH forward — 前向传播: 编码 -> 重参数化 -> 解码
    // input: [N, C, 128, 128]
    // 返回: [N, C, 128, 128] 重建图像
    // 注: mu 和 logvar 通过 m_lastMu/m_lastLogVar 缓存，供 vaeLoss 使用
    Tensor forward(const Tensor& input) override {
        auto [mu, logvar] = encode(input);         // 20260330 ZJH 编码
        m_lastMu = mu;                              // 20260330 ZJH 缓存 mu（供 vaeLoss 使用）
        m_lastLogVar = logvar;                      // 20260330 ZJH 缓存 logvar（供 vaeLoss 使用）
        auto z = reparameterize(mu, logvar);        // 20260330 ZJH 重参数化采样
        auto reconstructed = decode(z);             // 20260330 ZJH 解码重建
        return reconstructed;
    }

    // 20260330 ZJH vaeLoss — VAE 损失函数
    // Loss = Reconstruction Loss (MSE) + KL Divergence
    // KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    // input: [N, C, H, W] 原始输入图像
    // reconstructed: [N, C, H, W] 重建图像
    // mu: [N, nLatentDim] 编码器输出的均值
    // logvar: [N, nLatentDim] 编码器输出的对数方差
    // 返回: 标量损失张量
    Tensor vaeLoss(const Tensor& input, const Tensor& reconstructed,
                   const Tensor& mu, const Tensor& logvar)
    {
        // 20260330 ZJH 重建损失: MSE = sum((input - reconstructed)^2)
        auto diff = tensorSub(input, reconstructed);  // 20260330 ZJH 差值
        auto sq = tensorMul(diff, diff);               // 20260330 ZJH 逐元素平方
        auto reconLoss = tensorSum(sq);                // 20260330 ZJH 全局求和

        // 20260330 ZJH KL 散度: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        auto muSq = tensorMul(mu, mu);     // 20260330 ZJH mu^2

        // 20260330 ZJH 计算 exp(logvar) 和 1 + logvar - mu^2 - exp(logvar)
        auto cLogvar = logvar.contiguous();
        auto cMuSq = muSq.contiguous();
        int nLatentTotal = cLogvar.numel();  // 20260330 ZJH 潜在维度总数 = N * nLatentDim
        const float* pLogvar = cLogvar.floatDataPtr();
        const float* pMuSq = cMuSq.floatDataPtr();

        float fKLSum = 0.0f;
        for (int i = 0; i < nLatentTotal; ++i) {
            // 20260330 ZJH KL 逐元素: 1 + logvar - mu^2 - exp(logvar)
            float fTerm = 1.0f + pLogvar[i] - pMuSq[i] - std::exp(pLogvar[i]);
            fKLSum += fTerm;
        }
        float fKLLoss = -0.5f * fKLSum;  // 20260330 ZJH KL = -0.5 * sum(...)

        // 20260330 ZJH 总损失 = 重建损失 + KL 散度
        auto cReconLoss = reconLoss.contiguous();
        float fReconVal = cReconLoss.floatDataPtr()[0];
        float fTotalLoss = fReconVal + fKLLoss;

        return Tensor::full({1}, fTotalLoss);  // 20260330 ZJH 返回标量损失
    }

    // 20260330 ZJH locate — 异常定位: 通过重建误差生成热力图并提取异常区域
    // input: [1, C, H, W] 输入图像（单张）
    // fThreshold: 异常分数阈值（0~1），超过则认为是异常区域，默认 0.5
    // 返回: LocationResult 包含热力图、异常分数和异常区域 bbox
    LocationResult locate(const Tensor& input, float fThreshold = 0.5f) {
        eval();  // 20260330 ZJH 确保评估模式

        // 20260330 ZJH 前向推理获取重建图像
        Tensor reconstructed = forward(input);

        auto cInput = input.contiguous();
        auto cRecon = reconstructed.contiguous();
        int nC = cInput.shape(1);   // 20260330 ZJH 通道数
        int nH = cInput.shape(2);   // 20260330 ZJH 高度
        int nW = cInput.shape(3);   // 20260330 ZJH 宽度

        const float* pIn = cInput.floatDataPtr();
        const float* pRecon = cRecon.floatDataPtr();

        // 20260330 ZJH 计算逐像素的 MSE 重建误差作为热力图
        // 对每个像素 (h, w)，误差 = mean_over_channels((input - recon)^2)
        std::vector<float> vecHeatmap(nH * nW, 0.0f);
        float fMaxError = 0.0f;  // 20260330 ZJH 追踪最大误差用于归一化

        for (int h = 0; h < nH; ++h) {
            for (int w = 0; w < nW; ++w) {
                float fPixelError = 0.0f;
                for (int c = 0; c < nC; ++c) {
                    int nIdx = (c * nH + h) * nW + w;  // 20260330 ZJH NCHW 索引（batch=0）
                    float fDiff = pIn[nIdx] - pRecon[nIdx];  // 20260330 ZJH 差值
                    fPixelError += fDiff * fDiff;  // 20260330 ZJH 平方误差累加
                }
                fPixelError /= static_cast<float>(nC);  // 20260330 ZJH 通道均值
                vecHeatmap[h * nW + w] = fPixelError;
                if (fPixelError > fMaxError) fMaxError = fPixelError;
            }
        }

        // 20260330 ZJH 归一化热力图到 [0, 1]
        if (fMaxError > 1e-8f) {
            for (auto& fVal : vecHeatmap) {
                fVal /= fMaxError;  // 20260330 ZJH 归一化
            }
        }

        // 20260330 ZJH 计算全局异常分数（热力图均值）
        float fAnomalyScore = 0.0f;
        for (float fVal : vecHeatmap) {
            fAnomalyScore += fVal;
        }
        fAnomalyScore /= static_cast<float>(nH * nW);

        // 20260330 ZJH 提取异常区域: 对热力图做阈值二值化 + 连通域分析
        // 简化实现: 使用行扫描找连续异常像素块并合并为 bbox
        std::vector<std::vector<float>> vecRegions = extractAnomalyRegions(
            vecHeatmap, nW, nH, fThreshold);

        // 20260330 ZJH 组装结果
        LocationResult result;
        result.vecHeatmap = vecHeatmap;
        result.nW = nW;
        result.nH = nH;
        result.fAnomalyScore = fAnomalyScore;
        result.vecRegions = vecRegions;

        return result;
    }

    // 20260330 ZJH 获取最近一次 forward 缓存的 mu
    const Tensor& lastMu() const { return m_lastMu; }

    // 20260330 ZJH 获取最近一次 forward 缓存的 logvar
    const Tensor& lastLogVar() const { return m_lastLogVar; }

    // 20260330 ZJH 重写 parameters() 收集所有子层参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        // 20260330 ZJH 编码器参数
        append(m_encConv1.parameters());  append(m_encBn1.parameters());
        append(m_encConv2.parameters());  append(m_encBn2.parameters());
        append(m_encConv3.parameters());  append(m_encBn3.parameters());
        append(m_encConv4.parameters());  append(m_encBn4.parameters());
        append(m_fcMu.parameters());
        append(m_fcLogVar.parameters());
        // 20260330 ZJH 解码器参数
        append(m_fcDecode.parameters());
        append(m_decConv1.parameters());  append(m_decBn1.parameters());
        append(m_decConv2.parameters());  append(m_decBn2.parameters());
        append(m_decConv3.parameters());  append(m_decBn3.parameters());
        append(m_decConv4.parameters());
        return vecResult;
    }

    // 20260330 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto append = [&](std::vector<Tensor*> v) {
            vecResult.insert(vecResult.end(), v.begin(), v.end());
        };
        append(m_encBn1.buffers());  append(m_encBn2.buffers());
        append(m_encBn3.buffers());  append(m_encBn4.buffers());
        append(m_decBn1.buffers());  append(m_decBn2.buffers());
        append(m_decBn3.buffers());
        return vecResult;
    }

private:
    int m_nLatentDim;    // 20260330 ZJH 潜在空间维度
    int m_nImageSize;    // 20260330 ZJH 输入图像尺寸（正方形边长）
    int m_nChannels;     // 20260330 ZJH 输入通道数

    // 20260330 ZJH 编码器层
    Conv2d m_encConv1;   // 20260330 ZJH 编码器第 1 层卷积: C -> 32
    Conv2d m_encConv2;   // 20260330 ZJH 编码器第 2 层卷积: 32 -> 64
    Conv2d m_encConv3;   // 20260330 ZJH 编码器第 3 层卷积: 64 -> 128
    Conv2d m_encConv4;   // 20260330 ZJH 编码器第 4 层卷积: 128 -> 256
    BatchNorm2d m_encBn1;  // 20260330 ZJH 编码器第 1 层 BN
    BatchNorm2d m_encBn2;  // 20260330 ZJH 编码器第 2 层 BN
    BatchNorm2d m_encBn3;  // 20260330 ZJH 编码器第 3 层 BN
    BatchNorm2d m_encBn4;  // 20260330 ZJH 编码器第 4 层 BN

    // 20260330 ZJH 潜在空间映射
    Linear m_fcMu;        // 20260330 ZJH 均值映射: 16384 -> nLatentDim
    Linear m_fcLogVar;    // 20260330 ZJH 对数方差映射: 16384 -> nLatentDim
    Linear m_fcDecode;    // 20260330 ZJH 解码器输入映射: nLatentDim -> 16384

    // 20260330 ZJH 解码器层
    ConvTranspose2d m_decConv1;  // 20260330 ZJH 解码器第 1 层转置卷积: 256 -> 128
    ConvTranspose2d m_decConv2;  // 20260330 ZJH 解码器第 2 层转置卷积: 128 -> 64
    ConvTranspose2d m_decConv3;  // 20260330 ZJH 解码器第 3 层转置卷积: 64 -> 32
    ConvTranspose2d m_decConv4;  // 20260330 ZJH 解码器第 4 层转置卷积: 32 -> C
    BatchNorm2d m_decBn1;  // 20260330 ZJH 解码器第 1 层 BN
    BatchNorm2d m_decBn2;  // 20260330 ZJH 解码器第 2 层 BN
    BatchNorm2d m_decBn3;  // 20260330 ZJH 解码器第 3 层 BN

    // 20260330 ZJH 激活函数和展平层
    ReLU m_relu;          // 20260330 ZJH ReLU 激活
    Sigmoid m_sigmoid;    // 20260330 ZJH Sigmoid 激活（解码器输出层）
    Flatten m_flatten;    // 20260330 ZJH 展平层

    // 20260330 ZJH 缓存最近一次 forward 的 mu 和 logvar（供 vaeLoss 使用）
    Tensor m_lastMu;      // 20260330 ZJH 缓存的编码均值
    Tensor m_lastLogVar;  // 20260330 ZJH 缓存的编码对数方差

    // =========================================================
    // 20260330 ZJH 子模块包装器
    // Module::registerChild 需要 shared_ptr<Module>
    // 但编码器/解码器层是直接成员变量，需要薄包装器转发 forward
    // =========================================================

    // 20260330 ZJH Conv2dWrapper — Conv2d 包装器，持有引用不拥有所有权
    class Conv2dWrapper : public Module {
    public:
        Conv2dWrapper(Conv2d& ref) : m_ref(ref) {}
        Tensor forward(const Tensor& input) override { return m_ref.forward(input); }
        std::vector<Tensor*> parameters() override { return m_ref.parameters(); }
        std::vector<Tensor*> buffers() override { return m_ref.buffers(); }
    private:
        Conv2d& m_ref;
    };

    // 20260330 ZJH BnWrapper — BatchNorm2d 包装器
    class BnWrapper : public Module {
    public:
        BnWrapper(BatchNorm2d& ref) : m_ref(ref) {}
        Tensor forward(const Tensor& input) override { return m_ref.forward(input); }
        std::vector<Tensor*> parameters() override { return m_ref.parameters(); }
        std::vector<Tensor*> buffers() override { return m_ref.buffers(); }
    private:
        BatchNorm2d& m_ref;
    };

    // 20260330 ZJH LinearWrapper — Linear 包装器
    class LinearWrapper : public Module {
    public:
        LinearWrapper(Linear& ref) : m_ref(ref) {}
        Tensor forward(const Tensor& input) override { return m_ref.forward(input); }
        std::vector<Tensor*> parameters() override { return m_ref.parameters(); }
        std::vector<Tensor*> buffers() override { return m_ref.buffers(); }
    private:
        Linear& m_ref;
    };

    // 20260330 ZJH ConvTWrapper — ConvTranspose2d 包装器
    class ConvTWrapper : public Module {
    public:
        ConvTWrapper(ConvTranspose2d& ref) : m_ref(ref) {}
        Tensor forward(const Tensor& input) override { return m_ref.forward(input); }
        std::vector<Tensor*> parameters() override { return m_ref.parameters(); }
        std::vector<Tensor*> buffers() override { return m_ref.buffers(); }
    private:
        ConvTranspose2d& m_ref;
    };

    // 20260330 ZJH registerChild — 注册子模块（调用 Module 基类的方法）
    void registerChild(const std::string& strName, std::shared_ptr<Module> pChild) {
        m_vecWrappers.push_back(pChild);  // 20260330 ZJH 持有包装器防止析构
        // 20260330 ZJH 注意: 直接操作基类的 m_vecChildren 需要使用基类接口
        // 由于 Module 没有公开 registerChild 方法，这里通过 m_vecWrappers 管理生命周期
        // 参数收集和 buffers 收集已在 parameters()/buffers() 中手动实现
    }

    std::vector<std::shared_ptr<Module>> m_vecWrappers;  // 20260330 ZJH 包装器生命周期管理

    // =========================================================
    // 20260330 ZJH extractAnomalyRegions — 从热力图提取异常区域 bbox
    // 简化连通域分析: 行扫描 + 区域合并
    // vecHeatmap: 归一化热力图 [0, 1]
    // nW, nH: 热力图尺寸
    // fThreshold: 二值化阈值
    // 返回: 异常区域 bbox 列表，每个为 [x1, y1, x2, y2]
    // =========================================================
    std::vector<std::vector<float>> extractAnomalyRegions(
        const std::vector<float>& vecHeatmap, int nW, int nH, float fThreshold)
    {
        // 20260330 ZJH 二值化热力图
        std::vector<bool> vecBinary(nH * nW, false);
        for (int i = 0; i < nH * nW; ++i) {
            vecBinary[i] = (vecHeatmap[i] >= fThreshold);  // 20260330 ZJH 超过阈值标记为异常
        }

        // 20260330 ZJH 简化连通域: 使用 Union-Find（并查集）
        std::vector<int> vecParent(nH * nW);
        std::iota(vecParent.begin(), vecParent.end(), 0);  // 20260330 ZJH 初始化每个像素指向自己

        // 20260330 ZJH 并查集 find 函数（带路径压缩）
        auto fnFind = [&](int x) -> int {
            while (vecParent[x] != x) {
                vecParent[x] = vecParent[vecParent[x]];  // 20260330 ZJH 路径压缩
                x = vecParent[x];
            }
            return x;
        };

        // 20260330 ZJH 并查集 union 函数
        auto fnUnion = [&](int a, int b) {
            int nRootA = fnFind(a);
            int nRootB = fnFind(b);
            if (nRootA != nRootB) {
                vecParent[nRootA] = nRootB;  // 20260330 ZJH 合并两个集合
            }
        };

        // 20260330 ZJH 扫描二值图，连接相邻的异常像素
        for (int h = 0; h < nH; ++h) {
            for (int w = 0; w < nW; ++w) {
                int nIdx = h * nW + w;
                if (!vecBinary[nIdx]) continue;  // 20260330 ZJH 非异常像素跳过

                // 20260330 ZJH 检查右邻居
                if (w + 1 < nW && vecBinary[nIdx + 1]) {
                    fnUnion(nIdx, nIdx + 1);
                }
                // 20260330 ZJH 检查下邻居
                if (h + 1 < nH && vecBinary[nIdx + nW]) {
                    fnUnion(nIdx, nIdx + nW);
                }
            }
        }

        // 20260330 ZJH 收集每个连通域的 bbox
        // key: 根节点 ID, value: (minX, minY, maxX, maxY, pixelCount)
        struct RegionInfo {
            float fMinX = 1e9f;   // 20260330 ZJH bbox 左边界
            float fMinY = 1e9f;   // 20260330 ZJH bbox 上边界
            float fMaxX = -1e9f;  // 20260330 ZJH bbox 右边界
            float fMaxY = -1e9f;  // 20260330 ZJH bbox 下边界
            int nPixelCount = 0;  // 20260330 ZJH 异常像素数量
        };

        // 20260330 ZJH 使用线性搜索代替 unordered_map 避免哈希开销
        std::vector<std::pair<int, RegionInfo>> vecRegionMap;

        for (int h = 0; h < nH; ++h) {
            for (int w = 0; w < nW; ++w) {
                int nIdx = h * nW + w;
                if (!vecBinary[nIdx]) continue;

                int nRoot = fnFind(nIdx);  // 20260330 ZJH 找到所属连通域根节点

                // 20260330 ZJH 在 vecRegionMap 中查找或创建该根节点的 RegionInfo
                RegionInfo* pInfo = nullptr;
                for (auto& [nKey, info] : vecRegionMap) {
                    if (nKey == nRoot) {
                        pInfo = &info;
                        break;
                    }
                }
                if (!pInfo) {
                    vecRegionMap.push_back({nRoot, RegionInfo{}});
                    pInfo = &vecRegionMap.back().second;
                }

                // 20260330 ZJH 更新 bbox 范围
                float fX = static_cast<float>(w);
                float fY = static_cast<float>(h);
                if (fX < pInfo->fMinX) pInfo->fMinX = fX;
                if (fY < pInfo->fMinY) pInfo->fMinY = fY;
                if (fX > pInfo->fMaxX) pInfo->fMaxX = fX;
                if (fY > pInfo->fMaxY) pInfo->fMaxY = fY;
                pInfo->nPixelCount++;
            }
        }

        // 20260330 ZJH 过滤小区域（噪声）并输出 bbox
        int nMinPixels = std::max(1, nW * nH / 1000);  // 20260330 ZJH 最小区域: 图像面积的 0.1%
        std::vector<std::vector<float>> vecResult;

        for (const auto& [nRoot, info] : vecRegionMap) {
            if (info.nPixelCount < nMinPixels) continue;  // 20260330 ZJH 过滤噪声小区域
            vecResult.push_back({
                info.fMinX, info.fMinY,
                info.fMaxX + 1.0f, info.fMaxY + 1.0f  // 20260330 ZJH +1 使 bbox 包含右下边界像素
            });
        }

        return vecResult;  // 20260330 ZJH 返回异常区域 bbox 列表
    }
};

// =========================================================
// 5. Weighted Box Fusion (WBF)
// 20260330 ZJH 加权框融合 — 比 NMS 更优的多模型检测框融合方法
// 核心思想: 对重叠框取加权平均坐标，而非简单抑制低分框
// 论文: Weighted Boxes Fusion: Ensembling boxes from different object detection models
// 算法流程:
//   1. 所有模型输出的框按分数降序排列
//   2. 遍历每个框，尝试匹配到已有的融合簇（IoU > 阈值）
//   3. 匹配到: 更新簇的加权平均坐标和分数
//   4. 未匹配: 创建新簇
//   5. 最终输出所有簇的加权平均框
// =========================================================

// 20260330 ZJH weightedBoxFusion — WBF 加权框融合
// vecModelOutputs: 每个模型的检测框列表
// fIouThresh: 融合 IoU 阈值，默认 0.55
// fScoreThresh: 分数过滤阈值，默认 0.001
// 返回: 融合后的检测框列表
std::vector<InferDetectionBox> weightedBoxFusion(
    const std::vector<std::vector<InferDetectionBox>>& vecModelOutputs,
    float fIouThresh, float fScoreThresh)
{
    int nNumModels = static_cast<int>(vecModelOutputs.size());
    if (nNumModels == 0) return {};  // 20260330 ZJH 无输入返回空

    // 20260330 ZJH 收集所有模型的框到统一列表，附带模型来源索引
    struct TaggedBox {
        InferDetectionBox box;    // 20260330 ZJH 检测框
        int nModelIdx;            // 20260330 ZJH 来源模型索引
    };

    std::vector<TaggedBox> vecAllBoxes;
    for (int m = 0; m < nNumModels; ++m) {
        for (const auto& box : vecModelOutputs[m]) {
            if (box.fScore >= fScoreThresh) {
                vecAllBoxes.push_back({box, m});  // 20260330 ZJH 过滤低分框后收集
            }
        }
    }

    // 20260330 ZJH 按分数降序排列
    std::sort(vecAllBoxes.begin(), vecAllBoxes.end(),
              [](const TaggedBox& a, const TaggedBox& b) {
                  return a.box.fScore > b.box.fScore;  // 20260330 ZJH 高分优先
              });

    // 20260330 ZJH 融合簇结构
    struct FusionCluster {
        std::vector<TaggedBox> vecMembers;  // 20260330 ZJH 簇内所有框
        // 20260330 ZJH 加权平均坐标和分数
        float fFusedX1 = 0.0f;
        float fFusedY1 = 0.0f;
        float fFusedX2 = 0.0f;
        float fFusedY2 = 0.0f;
        float fFusedScore = 0.0f;
        float fWeightSum = 0.0f;  // 20260330 ZJH 权重总和
        int nClassId = -1;        // 20260330 ZJH 簇的类别 ID（取最高分框的类别）
    };

    std::vector<FusionCluster> vecClusters;

    // 20260330 ZJH 遍历每个框，尝试匹配到已有簇
    for (const auto& taggedBox : vecAllBoxes) {
        int nBestClusterIdx = -1;   // 20260330 ZJH 最佳匹配簇索引
        float fBestIoU = 0.0f;      // 20260330 ZJH 最佳匹配 IoU

        // 20260330 ZJH 在已有簇中查找最佳匹配（同类别 + IoU 最高）
        for (size_t ci = 0; ci < vecClusters.size(); ++ci) {
            auto& cluster = vecClusters[ci];

            // 20260330 ZJH 类别必须一致
            if (cluster.nClassId != taggedBox.box.nClassId) continue;

            // 20260330 ZJH 用簇的当前融合框计算 IoU
            InferDetectionBox fusedBox;
            fusedBox.fX1 = cluster.fFusedX1;
            fusedBox.fY1 = cluster.fFusedY1;
            fusedBox.fX2 = cluster.fFusedX2;
            fusedBox.fY2 = cluster.fFusedY2;
            fusedBox.fScore = cluster.fFusedScore;
            fusedBox.nClassId = cluster.nClassId;

            float fIoU = computeInferIoU(taggedBox.box, fusedBox);

            if (fIoU > fIouThresh && fIoU > fBestIoU) {
                fBestIoU = fIoU;
                nBestClusterIdx = static_cast<int>(ci);
            }
        }

        if (nBestClusterIdx >= 0) {
            // 20260330 ZJH 匹配到已有簇: 更新加权平均
            auto& cluster = vecClusters[nBestClusterIdx];
            cluster.vecMembers.push_back(taggedBox);

            float fW = taggedBox.box.fScore;  // 20260330 ZJH 以分数作为权重

            // 20260330 ZJH 重新计算加权平均（从所有成员重算，确保精度）
            cluster.fWeightSum += fW;
            float fInvW = 1.0f / cluster.fWeightSum;

            // 20260330 ZJH 使用增量更新法计算加权平均坐标
            float fNewX1 = 0.0f, fNewY1 = 0.0f, fNewX2 = 0.0f, fNewY2 = 0.0f;
            float fScoreSum = 0.0f;
            for (const auto& member : cluster.vecMembers) {
                float fMW = member.box.fScore;  // 20260330 ZJH 成员权重（用于坐标加权）
                fNewX1 += member.box.fX1 * fMW;
                fNewY1 += member.box.fY1 * fMW;
                fNewX2 += member.box.fX2 * fMW;
                fNewY2 += member.box.fY2 * fMW;
                fScoreSum += member.box.fScore;  // 20260330 ZJH 分数直接求和（非二次加权）
            }
            cluster.fFusedX1 = fNewX1 * fInvW;
            cluster.fFusedY1 = fNewY1 * fInvW;
            cluster.fFusedX2 = fNewX2 * fInvW;
            cluster.fFusedY2 = fNewY2 * fInvW;
            // 20260330 ZJH WBF 论文: 融合分数 = 簇内分数之和 / 成员数（算术平均）
            cluster.fFusedScore = fScoreSum / static_cast<float>(cluster.vecMembers.size());
        } else {
            // 20260330 ZJH 未匹配: 创建新簇
            FusionCluster newCluster;
            newCluster.vecMembers.push_back(taggedBox);
            newCluster.fFusedX1 = taggedBox.box.fX1;
            newCluster.fFusedY1 = taggedBox.box.fY1;
            newCluster.fFusedX2 = taggedBox.box.fX2;
            newCluster.fFusedY2 = taggedBox.box.fY2;
            newCluster.fFusedScore = taggedBox.box.fScore;
            newCluster.fWeightSum = taggedBox.box.fScore;
            newCluster.nClassId = taggedBox.box.nClassId;
            vecClusters.push_back(newCluster);
        }
    }

    // 20260330 ZJH 将融合簇转换为最终检测框列表
    std::vector<InferDetectionBox> vecResult;
    vecResult.reserve(vecClusters.size());

    for (const auto& cluster : vecClusters) {
        // 20260330 ZJH WBF 论文建议: 最终分数乘以簇内框数 / 模型数的比值
        // 这样鼓励被多个模型共同检测到的框
        float fConfirmRatio = static_cast<float>(cluster.vecMembers.size()) /
                              static_cast<float>(nNumModels);
        // 20260330 ZJH 限制 ratio 不超过 1.0（避免单模型多次检测到同一目标导致虚高）
        fConfirmRatio = std::min(fConfirmRatio, 1.0f);

        float fFinalScore = cluster.fFusedScore * fConfirmRatio;

        // 20260330 ZJH 过滤低分融合结果
        if (fFinalScore >= fScoreThresh) {
            vecResult.push_back({
                cluster.fFusedX1, cluster.fFusedY1,
                cluster.fFusedX2, cluster.fFusedY2,
                fFinalScore, cluster.nClassId
            });
        }
    }

    // 20260330 ZJH 按分数降序排列最终结果
    std::sort(vecResult.begin(), vecResult.end(),
              [](const InferDetectionBox& a, const InferDetectionBox& b) {
                  return a.fScore > b.fScore;
              });

    return vecResult;  // 20260330 ZJH 返回 WBF 融合后的检测框
}

}  // namespace om
