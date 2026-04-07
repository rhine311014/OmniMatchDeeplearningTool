#pragma once
// 20260330 ZJH 通用 NMS（非极大值抑制）工具
// 适用于 YOLO 检测、实例分割、OCR 等所有需要去重叠框的场景
// 提供: 标准 NMS / Soft-NMS / 类别感知 NMS / Batched NMS

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>

namespace om {

// =========================================================
// DetectionBox — 通用检测框结构
// =========================================================

// 20260330 ZJH 通用检测框 — 比 InstanceResult 更轻量，不绑定 mask 系数
struct DetectionBox {
    float fX1;         // 20260330 ZJH 左上角 x 坐标
    float fY1;         // 20260330 ZJH 左上角 y 坐标
    float fX2;         // 20260330 ZJH 右下角 x 坐标
    float fY2;         // 20260330 ZJH 右下角 y 坐标
    float fScore;      // 20260330 ZJH 置信度分数 [0, 1]
    int nClassId;      // 20260330 ZJH 类别 ID（-1 表示未分类）
    int nBatchIdx;     // 20260330 ZJH 批次索引（用于 Batched NMS，单图设为 0）

    // 20260330 ZJH 计算面积
    float area() const {
        float fW = fX2 - fX1;  // 20260330 ZJH 宽度
        float fH = fY2 - fY1;  // 20260330 ZJH 高度
        return (fW > 0.0f && fH > 0.0f) ? fW * fH : 0.0f;  // 20260330 ZJH 无效框返回 0
    }
};

// =========================================================
// IoU 计算
// =========================================================

// 20260330 ZJH computeBoxIoU — 计算两个检测框的 IoU（交并比）
// a: 第一个检测框
// b: 第二个检测框
// 返回: IoU 值 [0, 1]
inline float computeBoxIoU(const DetectionBox& a, const DetectionBox& b) {
    // 20260330 ZJH 计算交集区域坐标
    float fInterX1 = std::max(a.fX1, b.fX1);  // 20260330 ZJH 交集左边界
    float fInterY1 = std::max(a.fY1, b.fY1);  // 20260330 ZJH 交集上边界
    float fInterX2 = std::min(a.fX2, b.fX2);  // 20260330 ZJH 交集右边界
    float fInterY2 = std::min(a.fY2, b.fY2);  // 20260330 ZJH 交集下边界

    // 20260330 ZJH 交集面积（无重叠时为 0）
    float fInterArea = std::max(0.0f, fInterX2 - fInterX1) *
                       std::max(0.0f, fInterY2 - fInterY1);

    // 20260330 ZJH 各自面积
    float fAreaA = a.area();
    float fAreaB = b.area();

    // 20260330 ZJH 并集面积 = A + B - 交集
    float fUnionArea = fAreaA + fAreaB - fInterArea;

    // 20260330 ZJH 防除零
    return (fUnionArea > 0.0f) ? fInterArea / fUnionArea : 0.0f;
}

// =========================================================
// 标准 NMS — Greedy 贪心非极大值抑制
// =========================================================

// 20260330 ZJH nms — 标准 Greedy NMS
// 按置信度降序排列，保留最高分框，抑制与之 IoU 超过阈值的框
// vecBoxes: 输入检测框列表（会被修改排序）
// fIoUThreshold: IoU 阈值（默认 0.45，YOLO 常用值）
// fScoreThreshold: 置信度过滤阈值（低于此值的框直接丢弃）
// bClassAware: 是否按类别分别做 NMS（true = 同类别内抑制，false = 全局抑制）
// 返回: NMS 后保留的检测框列表
inline std::vector<DetectionBox> nms(const std::vector<DetectionBox>& vecBoxes,  // 20260330 ZJH 改为 const 引用，不修改输入
                                      float fIoUThreshold = 0.45f,
                                      float fScoreThreshold = 0.25f,
                                      bool bClassAware = true) {
    // 20260330 ZJH 步骤 1: 过滤低置信度框
    std::vector<DetectionBox> vecFiltered;
    vecFiltered.reserve(vecBoxes.size());  // 20260330 ZJH 预分配避免多次扩容
    for (const auto& box : vecBoxes) {
        if (box.fScore >= fScoreThreshold) {  // 20260330 ZJH 只保留高于阈值的框
            vecFiltered.push_back(box);
        }
    }

    // 20260330 ZJH 空列表直接返回
    if (vecFiltered.empty()) {
        return {};
    }

    // 20260330 ZJH 步骤 2: 按置信度降序排列
    std::sort(vecFiltered.begin(), vecFiltered.end(),
              [](const DetectionBox& a, const DetectionBox& b) {
                  return a.fScore > b.fScore;  // 20260330 ZJH 高分在前
              });

    // 20260330 ZJH 步骤 3: Greedy 抑制
    std::vector<bool> vecSuppressed(vecFiltered.size(), false);  // 20260330 ZJH 抑制标记
    std::vector<DetectionBox> vecKept;  // 20260330 ZJH 保留结果
    vecKept.reserve(vecFiltered.size());

    for (size_t i = 0; i < vecFiltered.size(); ++i) {
        if (vecSuppressed[i]) continue;  // 20260330 ZJH 已被抑制，跳过
        vecKept.push_back(vecFiltered[i]);  // 20260330 ZJH 保留当前最高分框

        // 20260330 ZJH 对后续所有框检查 IoU
        for (size_t j = i + 1; j < vecFiltered.size(); ++j) {
            if (vecSuppressed[j]) continue;  // 20260330 ZJH 已被抑制，跳过

            // 20260330 ZJH 类别感知: 只抑制同类别的框
            if (bClassAware && vecFiltered[i].nClassId != vecFiltered[j].nClassId) {
                continue;  // 20260330 ZJH 不同类别不互相抑制
            }

            // 20260330 ZJH 计算 IoU，超阈值则抑制
            float fIoU = computeBoxIoU(vecFiltered[i], vecFiltered[j]);
            if (fIoU > fIoUThreshold) {
                vecSuppressed[j] = true;  // 20260330 ZJH 标记为被抑制
            }
        }
    }

    return vecKept;  // 20260330 ZJH 返回去重后的检测框
}

// =========================================================
// Soft-NMS — 软非极大值抑制
// =========================================================

// 20260330 ZJH Soft-NMS 衰减方式
enum class SoftNmsMethod {
    Linear,    // 20260330 ZJH 线性衰减: score *= (1 - IoU)
    Gaussian   // 20260330 ZJH 高斯衰减: score *= exp(-IoU^2 / sigma)
};

// 20260330 ZJH softNms — 软非极大值抑制
// 不直接删除重叠框，而是降低其置信度，保留更多检测结果
// 适用于密集遮挡场景（如行人检测、密排零件检测）
// vecBoxes: 输入检测框列表（会被修改，分数会被衰减）
// fIoUThreshold: IoU 阈值（仅 Linear 方式使用，高于此值才衰减）
// fScoreThreshold: 最终过滤阈值（衰减后低于此值的框被移除）
// method: 衰减方式（Linear / Gaussian）
// fSigma: 高斯衰减的 sigma 参数（默认 0.5）
// 返回: Soft-NMS 后保留的检测框列表
inline std::vector<DetectionBox> softNms(const std::vector<DetectionBox>& vecBoxes,  // 20260330 ZJH 改为 const 引用
                                          float fIoUThreshold = 0.3f,
                                          float fScoreThreshold = 0.01f,
                                          SoftNmsMethod method = SoftNmsMethod::Gaussian,
                                          float fSigma = 0.5f) {
    // 20260330 ZJH 工作副本，避免破坏原始数据
    std::vector<DetectionBox> vecWork(vecBoxes.begin(), vecBoxes.end());
    std::vector<DetectionBox> vecKept;  // 20260330 ZJH 保留结果

    // 20260330 ZJH 逐轮选取当前最高分框
    while (!vecWork.empty()) {
        // 20260330 ZJH 找到当前最高分框的索引
        int nMaxIdx = 0;
        float fMaxScore = vecWork[0].fScore;
        for (int i = 1; i < static_cast<int>(vecWork.size()); ++i) {
            if (vecWork[i].fScore > fMaxScore) {
                fMaxScore = vecWork[i].fScore;
                nMaxIdx = i;
            }
        }

        // 20260330 ZJH 取出最高分框并加入结果
        DetectionBox bestBox = vecWork[nMaxIdx];
        vecWork.erase(vecWork.begin() + nMaxIdx);  // 20260330 ZJH 从工作列表移除

        if (bestBox.fScore < fScoreThreshold) {
            continue;  // 20260330 ZJH 分数太低，跳过
        }

        vecKept.push_back(bestBox);  // 20260330 ZJH 保留

        // 20260330 ZJH 衰减剩余框的分数
        for (auto& box : vecWork) {
            float fIoU = computeBoxIoU(bestBox, box);

            if (method == SoftNmsMethod::Linear) {
                // 20260330 ZJH 线性衰减: IoU > 阈值时，score *= (1 - IoU)
                if (fIoU > fIoUThreshold) {
                    box.fScore *= (1.0f - fIoU);
                }
            } else {
                // 20260330 ZJH 高斯衰减: score *= exp(-IoU^2 / sigma)
                box.fScore *= std::exp(-(fIoU * fIoU) / fSigma);
            }
        }

        // 20260330 ZJH 移除衰减后低于阈值的框
        vecWork.erase(
            std::remove_if(vecWork.begin(), vecWork.end(),
                           [fScoreThreshold](const DetectionBox& b) {
                               return b.fScore < fScoreThreshold;  // 20260330 ZJH 低分剔除
                           }),
            vecWork.end());
    }

    return vecKept;  // 20260330 ZJH 返回 Soft-NMS 结果
}

// =========================================================
// Batched NMS — 多图批量 NMS
// =========================================================

// 20260330 ZJH batchedNms — 按批次索引分组做 NMS
// 适用于批量推理场景，每张图的检测结果独立做 NMS
// vecBoxes: 所有图像的检测框列表（需设置 nBatchIdx 字段）
// fIoUThreshold: IoU 阈值
// fScoreThreshold: 置信度过滤阈值
// bClassAware: 是否类别感知
// 返回: 所有图像 NMS 后的合并结果
inline std::vector<DetectionBox> batchedNms(const std::vector<DetectionBox>& vecBoxes,  // 20260330 ZJH 改为 const 引用
                                             float fIoUThreshold = 0.45f,
                                             float fScoreThreshold = 0.25f,
                                             bool bClassAware = true) {
    // 20260330 ZJH 找到最大批次索引
    int nMaxBatch = 0;
    for (const auto& box : vecBoxes) {
        if (box.nBatchIdx > nMaxBatch) {
            nMaxBatch = box.nBatchIdx;  // 20260330 ZJH 记录最大批次号
        }
    }

    std::vector<DetectionBox> vecAllKept;  // 20260330 ZJH 合并结果

    // 20260330 ZJH 逐批次分别做 NMS
    for (int b = 0; b <= nMaxBatch; ++b) {
        // 20260330 ZJH 收集当前批次的框
        std::vector<DetectionBox> vecBatch;
        for (const auto& box : vecBoxes) {
            if (box.nBatchIdx == b) {
                vecBatch.push_back(box);
            }
        }

        if (vecBatch.empty()) continue;  // 20260330 ZJH 空批次跳过

        // 20260330 ZJH 对当前批次做标准 NMS
        auto vecKept = nms(vecBatch, fIoUThreshold, fScoreThreshold, bClassAware);

        // 20260330 ZJH 合并到总结果
        vecAllKept.insert(vecAllKept.end(), vecKept.begin(), vecKept.end());
    }

    return vecAllKept;  // 20260330 ZJH 返回所有批次的 NMS 结果
}

// =========================================================
// YOLO 后处理: 解码 + NMS 一体化
// =========================================================

// 20260330 ZJH YoloDecodeParams — YOLO 解码参数
struct YoloDecodeParams {
    int nNumClasses;           // 20260330 ZJH 类别数量
    float fConfThreshold;      // 20260330 ZJH 置信度阈值（obj * cls_score）
    float fNmsIoUThreshold;    // 20260330 ZJH NMS 的 IoU 阈值
    int nInputWidth;           // 20260330 ZJH 模型输入宽度（用于坐标反归一化）
    int nInputHeight;          // 20260330 ZJH 模型输入高度
};

// 20260330 ZJH yoloDecodeAndNms — YOLO 检测输出解码 + NMS
// 将 YOLO 模型的原始输出 [N, P, 5+C] 解码为检测框列表并做 NMS
// pPredictions: 原始预测数据指针，布局 [nBatch, nPreds, 5+C]
//               每个预测: [cx, cy, w, h, objectness, cls0, cls1, ..., clsC-1]
//               cx/cy/w/h 为归一化坐标 [0,1]
// nBatch: 批次大小
// nPreds: 每张图的预测数量
// params: 解码参数
// 返回: NMS 后的检测框列表
inline std::vector<DetectionBox> yoloDecodeAndNms(const float* pPredictions,
                                                   int nBatch, int nPreds,
                                                   const YoloDecodeParams& params) {
    // 20260330 ZJH 防御性检查: 空指针或无效维度直接返回空
    if (!pPredictions || nBatch <= 0 || nPreds <= 0 || params.nNumClasses <= 0) return {};

    int nPerPred = 5 + params.nNumClasses;  // 20260330 ZJH 每个预测的元素数
    std::vector<DetectionBox> vecAllBoxes;   // 20260330 ZJH 所有解码后的框
    vecAllBoxes.reserve(static_cast<size_t>(nBatch) * static_cast<size_t>(nPreds) / 4);  // 20260330 ZJH 预估保留约 25%（size_t 防溢出）

    // 20260330 ZJH 逐批次、逐预测解码
    for (int b = 0; b < nBatch; ++b) {
        for (int p = 0; p < nPreds; ++p) {
            // 20260330 ZJH 定位当前预测的起始位置
            const float* pCur = pPredictions + (b * nPreds + p) * nPerPred;

            // 20260330 ZJH 读取中心坐标和宽高（归一化值）
            float fCx = pCur[0];  // 20260330 ZJH 中心 x
            float fCy = pCur[1];  // 20260330 ZJH 中心 y
            float fW  = pCur[2];  // 20260330 ZJH 宽度
            float fH  = pCur[3];  // 20260330 ZJH 高度

            // 20260330 ZJH 读取 objectness（经过 sigmoid）
            float fObj = 1.0f / (1.0f + std::exp(-pCur[4]));  // 20260330 ZJH sigmoid(objectness)

            // 20260330 ZJH 找到最高分类别
            int nBestClass = 0;        // 20260330 ZJH 最佳类别索引
            float fBestClsScore = -1e9f;  // 20260330 ZJH 最佳类别分数
            for (int c = 0; c < params.nNumClasses; ++c) {
                float fClsRaw = pCur[5 + c];  // 20260330 ZJH 原始类别分数
                if (fClsRaw > fBestClsScore) {
                    fBestClsScore = fClsRaw;
                    nBestClass = c;
                }
            }

            // 20260330 ZJH sigmoid 归一化类别分数
            float fClsProb = 1.0f / (1.0f + std::exp(-fBestClsScore));

            // 20260330 ZJH 综合置信度 = objectness × class_probability
            float fConfidence = fObj * fClsProb;

            // 20260330 ZJH 低于阈值的预测直接跳过（提前过滤减少 NMS 负担）
            if (fConfidence < params.fConfThreshold) {
                continue;
            }

            // 20260330 ZJH 将中心坐标 + 宽高转换为左上角 + 右下角坐标
            float fX1 = (fCx - fW * 0.5f) * static_cast<float>(params.nInputWidth);
            float fY1 = (fCy - fH * 0.5f) * static_cast<float>(params.nInputHeight);
            float fX2 = (fCx + fW * 0.5f) * static_cast<float>(params.nInputWidth);
            float fY2 = (fCy + fH * 0.5f) * static_cast<float>(params.nInputHeight);

            // 20260330 ZJH 裁剪到图像边界
            fX1 = std::max(0.0f, fX1);
            fY1 = std::max(0.0f, fY1);
            fX2 = std::min(static_cast<float>(params.nInputWidth), fX2);
            fY2 = std::min(static_cast<float>(params.nInputHeight), fY2);

            // 20260330 ZJH 构造检测框并加入列表
            DetectionBox box;
            box.fX1 = fX1;
            box.fY1 = fY1;
            box.fX2 = fX2;
            box.fY2 = fY2;
            box.fScore = fConfidence;
            box.nClassId = nBestClass;
            box.nBatchIdx = b;
            vecAllBoxes.push_back(box);
        }
    }

    // 20260330 ZJH 对解码后的框执行类别感知 NMS
    if (nBatch > 1) {
        return batchedNms(vecAllBoxes, params.fNmsIoUThreshold, params.fConfThreshold, true);
    } else {
        return nms(vecAllBoxes, params.fNmsIoUThreshold, params.fConfThreshold, true);
    }
}

}  // namespace om
