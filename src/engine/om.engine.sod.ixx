// 20260330 ZJH SOD (Sliding Over Detection) 大图滑窗检测模块
// 将超大图像切分为重叠的固定尺寸 patch，逐 patch 推理后合并结果
// 适用于: 工业高分辨率缺陷检测、卫星/航拍图像分析、PCB 元器件定位
// 参考: HikRobot SetSODParam/GetSODParam 接口设计
// 支持: 检测结果合并(坐标偏移+NMS) + 分割结果合并(逐像素投票)
module;

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

export module om.engine.sod;

// 20260330 ZJH 导入张量模块（用于 extractPatchTensor 工厂方法）
import om.engine.tensor;

export namespace om {

// =========================================================
// SODParams — SOD 滑窗参数配置
// =========================================================

// 20260330 ZJH SODParams — 滑窗检测参数，参考 HikRobot SOD 参数设计
// nPatchWidth/nPatchHeight: 每个滑窗的尺寸，通常与模型输入尺寸一致
// fOverlapRatio: 相邻滑窗的重叠率，越大召回越高但推理越慢
// fNmsThreshold: 合并检测结果时的 NMS IoU 阈值
struct SODParams {
    int nPatchWidth = 640;         // 20260330 ZJH 滑窗宽度（像素），默认 640
    int nPatchHeight = 640;        // 20260330 ZJH 滑窗高度（像素），默认 640
    float fOverlapRatio = 0.25f;   // 20260330 ZJH 重叠率 [0, 0.5]，默认 25%
    float fNmsThreshold = 0.5f;    // 20260330 ZJH 合并时的 NMS IoU 阈值
    float fScoreThreshold = 0.25f; // 20260330 ZJH 合并时的置信度过滤阈值
    bool bClassAware = true;       // 20260330 ZJH NMS 是否按类别分别处理

    // 20260330 ZJH validate — 校验参数合法性
    // 返回: true=参数合法，false=参数有误
    bool validate() const {
        if (nPatchWidth <= 0 || nPatchHeight <= 0) return false;       // 20260330 ZJH 尺寸必须正
        if (fOverlapRatio < 0.0f || fOverlapRatio > 0.5f) return false; // 20260330 ZJH 重叠率范围
        if (fNmsThreshold < 0.0f || fNmsThreshold > 1.0f) return false; // 20260330 ZJH NMS 阈值范围
        if (fScoreThreshold < 0.0f || fScoreThreshold > 1.0f) return false; // 20260330 ZJH 置信度阈值范围
        return true;  // 20260330 ZJH 所有校验通过
    }
};

// =========================================================
// PatchInfo — 单个滑窗切片的坐标和尺寸信息
// =========================================================

// 20260330 ZJH PatchInfo — 描述原图上一个切片的位置和尺寸
// 边缘切片可能小于 patch 尺寸，此时需要填充
struct PatchInfo {
    int nX;           // 20260330 ZJH 切片左上角 x 坐标（原图坐标系）
    int nY;           // 20260330 ZJH 切片左上角 y 坐标（原图坐标系）
    int nWidth;       // 20260330 ZJH 切片实际宽度（不含填充）
    int nHeight;      // 20260330 ZJH 切片实际高度（不含填充）
    int nPadRight;    // 20260330 ZJH 右侧填充量（边缘切片可能需要，以凑齐 patch 尺寸）
    int nPadBottom;   // 20260330 ZJH 底部填充量（边缘切片可能需要）
    int nPatchIdx;    // 20260330 ZJH 切片索引（用于关联结果）
};

// =========================================================
// DetectionBox — 通用检测框结构（与 NmsUtils.h 保持一致）
// =========================================================

// 20260330 ZJH SODDetectionBox — SOD 模块内部使用的检测框结构
// 与 NmsUtils.h 中的 DetectionBox 保持相同字段，避免跨模块头文件依赖
struct SODDetectionBox {
    float fX1;         // 20260330 ZJH 左上角 x 坐标（原图坐标系）
    float fY1;         // 20260330 ZJH 左上角 y 坐标（原图坐标系）
    float fX2;         // 20260330 ZJH 右下角 x 坐标（原图坐标系）
    float fY2;         // 20260330 ZJH 右下角 y 坐标（原图坐标系）
    float fScore;      // 20260330 ZJH 置信度分数 [0, 1]
    int nClassId;      // 20260330 ZJH 类别 ID（-1 表示未分类）
    int nPatchIdx;     // 20260330 ZJH 来源切片索引（用于追溯）

    // 20260330 ZJH area — 计算检测框面积
    float area() const {
        float fW = fX2 - fX1;  // 20260330 ZJH 宽度
        float fH = fY2 - fY1;  // 20260330 ZJH 高度
        return (fW > 0.0f && fH > 0.0f) ? fW * fH : 0.0f;  // 20260330 ZJH 无效框返回 0
    }
};

// =========================================================
// generatePatches — 根据图像尺寸和 SOD 参数生成切片列表
// =========================================================

// 20260330 ZJH generatePatches — 生成覆盖整个图像的滑窗切片列表
// 从左上角开始，按步长（patch尺寸 × (1 - 重叠率)）滑动
// 边缘切片如果不足 patch 尺寸，记录填充量（推理时用零填充或镜像填充）
// nImageW: 原图宽度（像素）
// nImageH: 原图高度（像素）
// params: SOD 滑窗参数
// 返回: 切片信息列表，覆盖整个图像
inline std::vector<PatchInfo> generatePatches(int nImageW, int nImageH,
                                               const SODParams& params) {
    // 20260330 ZJH 参数校验
    if (nImageW <= 0 || nImageH <= 0) {
        return {};  // 20260330 ZJH 空图像返回空列表
    }
    if (params.nPatchWidth <= 0 || params.nPatchHeight <= 0) {
        throw std::invalid_argument("SOD: patch size must be positive");
    }

    // 20260330 ZJH 计算水平和垂直步长
    // 步长 = 切片尺寸 × (1 - 重叠率)，向下取整
    // 重叠率 0.25 时步长为切片尺寸的 75%
    float fClampedOverlap = std::max(0.0f, std::min(0.5f, params.fOverlapRatio));
    int nStrideX = std::max(1, static_cast<int>(
        std::floor(static_cast<float>(params.nPatchWidth) * (1.0f - fClampedOverlap))));
    int nStrideY = std::max(1, static_cast<int>(
        std::floor(static_cast<float>(params.nPatchHeight) * (1.0f - fClampedOverlap))));

    std::vector<PatchInfo> vecPatches;  // 20260330 ZJH 结果列表
    int nPatchIdx = 0;                  // 20260330 ZJH 切片序号

    // 20260330 ZJH 逐行逐列生成切片
    for (int nY = 0; nY < nImageH; nY += nStrideY) {
        for (int nX = 0; nX < nImageW; nX += nStrideX) {
            PatchInfo patch;
            patch.nX = nX;  // 20260330 ZJH 切片左上角 x
            patch.nY = nY;  // 20260330 ZJH 切片左上角 y

            // 20260330 ZJH 计算切片实际宽高（边缘可能不足 patch 尺寸）
            int nActualW = std::min(params.nPatchWidth, nImageW - nX);
            int nActualH = std::min(params.nPatchHeight, nImageH - nY);
            patch.nWidth = nActualW;    // 20260330 ZJH 实际有效宽度
            patch.nHeight = nActualH;   // 20260330 ZJH 实际有效高度

            // 20260330 ZJH 计算填充量（凑齐 patch 尺寸，推理时需要）
            patch.nPadRight = params.nPatchWidth - nActualW;    // 20260330 ZJH 右侧填充
            patch.nPadBottom = params.nPatchHeight - nActualH;  // 20260330 ZJH 底部填充

            patch.nPatchIdx = nPatchIdx++;  // 20260330 ZJH 分配切片序号

            vecPatches.push_back(patch);
        }
    }

    return vecPatches;  // 20260330 ZJH 返回覆盖全图的切片列表
}

// =========================================================
// extractPatch — 从原图像素数据中提取单个切片（CHW 布局）
// =========================================================

// 20260330 ZJH extractPatch — 从 CHW float 图像中裁剪指定区域并填充到 patch 尺寸
// 输入图像布局: [C, H, W]（通道优先，已归一化为 float）
// 边缘切片不足 patch 尺寸的部分用零填充（黑色填充）
// vecImage: 输入图像的 CHW float 数据
// nC: 通道数（通常为 3）
// nH: 图像高度
// nW: 图像宽度
// patch: 切片信息（坐标 + 尺寸 + 填充量）
// nPatchW: 目标 patch 宽度（等于 SODParams::nPatchWidth）
// nPatchH: 目标 patch 高度（等于 SODParams::nPatchHeight）
// 返回: [C × nPatchH × nPatchW] 的 float 数组，边缘零填充
inline std::vector<float> extractPatch(const std::vector<float>& vecImage,
                                        int nC, int nH, int nW,
                                        const PatchInfo& patch,
                                        int nPatchW, int nPatchH) {
    // 20260330 ZJH 校验输入数据尺寸
    int nExpectedSize = nC * nH * nW;  // 20260330 ZJH 期望的像素总数
    if (static_cast<int>(vecImage.size()) < nExpectedSize) {
        throw std::invalid_argument("SOD extractPatch: image data size mismatch");
    }

    // 20260330 ZJH 分配输出缓冲区并初始化为零（零填充边缘）
    int nOutputSize = nC * nPatchH * nPatchW;  // 20260330 ZJH 输出 patch 像素总数
    std::vector<float> vecPatch(nOutputSize, 0.0f);

    // 20260330 ZJH 逐通道逐行复制有效像素区域
    for (int c = 0; c < nC; ++c) {
        for (int y = 0; y < patch.nHeight; ++y) {
            // 20260330 ZJH 源偏移: CHW 布局中 (c, patch.nY+y, patch.nX) 的位置
            int nSrcOffset = c * nH * nW + (patch.nY + y) * nW + patch.nX;
            // 20260330 ZJH 目标偏移: CHW 布局中 (c, y, 0) 的位置
            int nDstOffset = c * nPatchH * nPatchW + y * nPatchW;
            // 20260330 ZJH 复制一行有效像素（宽度 = patch.nWidth）
            std::memcpy(&vecPatch[nDstOffset], &vecImage[nSrcOffset],
                        sizeof(float) * patch.nWidth);
        }
    }

    return vecPatch;  // 20260330 ZJH 返回提取的 patch 数据
}

// 20260330 ZJH extractPatchTensor — 从 CHW 图像提取切片并包装为 Tensor
// 返回形状 [1, C, nPatchH, nPatchW] 的 Tensor，可直接输入模型推理
// vecImage: 输入图像 CHW float 数据
// nC: 通道数
// nH: 图像高度
// nW: 图像宽度
// patch: 切片信息
// nPatchW: 目标 patch 宽度
// nPatchH: 目标 patch 高度
// 返回: [1, C, nPatchH, nPatchW] 形状的 Tensor
inline Tensor extractPatchTensor(const std::vector<float>& vecImage,
                                  int nC, int nH, int nW,
                                  const PatchInfo& patch,
                                  int nPatchW, int nPatchH) {
    // 20260330 ZJH 先提取原始 float 数据
    auto vecPatch = extractPatch(vecImage, nC, nH, nW, patch, nPatchW, nPatchH);
    // 20260330 ZJH 包装为 [1, C, H, W] 形状的 Tensor
    return Tensor::fromData(vecPatch.data(), {1, nC, nPatchH, nPatchW});
}

// =========================================================
// computeSODBoxIoU — SOD 内部 IoU 计算（避免跨模块依赖）
// =========================================================

// 20260330 ZJH computeSODBoxIoU — 计算两个 SOD 检测框的 IoU
// a: 第一个检测框
// b: 第二个检测框
// 返回: IoU 值 [0, 1]
inline float computeSODBoxIoU(const SODDetectionBox& a, const SODDetectionBox& b) {
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
// mergeDetections — 合并多个切片的检测结果
// =========================================================

// 20260330 ZJH mergeDetections — 将各切片的检测框映射回原图坐标并做 NMS 去重
// 工作流程:
//   1. 对每个切片的检测框坐标加上切片偏移量（映射到原图坐标系）
//   2. 汇总所有切片的检测框
//   3. 按置信度排序后执行 Greedy NMS 去除重叠区域的重复检测
// vecPatchResults: 每个切片的检测框列表，坐标相对于切片
// vecPatches: 对应的切片信息列表（提供偏移坐标）
// params: SOD 参数（NMS 阈值、置信度阈值、类别感知开关）
// 返回: 去重后的原图坐标检测框列表
inline std::vector<SODDetectionBox> mergeDetections(
        const std::vector<std::vector<SODDetectionBox>>& vecPatchResults,
        const std::vector<PatchInfo>& vecPatches,
        const SODParams& params,
        int nImageW = 0, int nImageH = 0) {  // 20260330 ZJH 新增原图尺寸参数，用于边界裁剪
    // 20260330 ZJH 校验切片数量一致性
    if (vecPatchResults.size() != vecPatches.size()) {
        throw std::invalid_argument("SOD mergeDetections: results/patches size mismatch");
    }

    // 20260330 ZJH 第一步: 坐标偏移 — 将切片坐标映射到原图坐标
    std::vector<SODDetectionBox> vecAllBoxes;  // 20260330 ZJH 汇总所有切片的检测框
    for (size_t i = 0; i < vecPatchResults.size(); ++i) {
        float fOffsetX = static_cast<float>(vecPatches[i].nX);  // 20260330 ZJH x 偏移
        float fOffsetY = static_cast<float>(vecPatches[i].nY);  // 20260330 ZJH y 偏移

        for (const auto& box : vecPatchResults[i]) {
            SODDetectionBox mappedBox;
            mappedBox.fX1 = box.fX1 + fOffsetX;      // 20260330 ZJH 映射左上角 x
            mappedBox.fY1 = box.fY1 + fOffsetY;      // 20260330 ZJH 映射左上角 y
            mappedBox.fX2 = box.fX2 + fOffsetX;      // 20260330 ZJH 映射右下角 x
            mappedBox.fY2 = box.fY2 + fOffsetY;      // 20260330 ZJH 映射右下角 y

            // 20260330 ZJH 裁剪到原图边界（防止越界坐标）
            if (nImageW > 0 && nImageH > 0) {
                mappedBox.fX1 = std::max(0.0f, mappedBox.fX1);                         // 20260330 ZJH 左边界
                mappedBox.fY1 = std::max(0.0f, mappedBox.fY1);                         // 20260330 ZJH 上边界
                mappedBox.fX2 = std::min(static_cast<float>(nImageW), mappedBox.fX2);  // 20260330 ZJH 右边界
                mappedBox.fY2 = std::min(static_cast<float>(nImageH), mappedBox.fY2);  // 20260330 ZJH 下边界
            }
            mappedBox.fScore = box.fScore;             // 20260330 ZJH 保持原始置信度
            mappedBox.nClassId = box.nClassId;         // 20260330 ZJH 保持原始类别
            mappedBox.nPatchIdx = vecPatches[i].nPatchIdx;  // 20260330 ZJH 记录来源切片
            vecAllBoxes.push_back(mappedBox);
        }
    }

    // 20260330 ZJH 第二步: 置信度过滤
    std::vector<SODDetectionBox> vecFiltered;
    vecFiltered.reserve(vecAllBoxes.size());
    for (const auto& box : vecAllBoxes) {
        if (box.fScore >= params.fScoreThreshold) {  // 20260330 ZJH 只保留高于阈值的框
            vecFiltered.push_back(box);
        }
    }

    if (vecFiltered.empty()) {
        return {};  // 20260330 ZJH 无有效检测框
    }

    // 20260330 ZJH 第三步: 按置信度降序排列
    std::sort(vecFiltered.begin(), vecFiltered.end(),
              [](const SODDetectionBox& a, const SODDetectionBox& b) {
                  return a.fScore > b.fScore;  // 20260330 ZJH 高分在前
              });

    // 20260330 ZJH 第四步: Greedy NMS 去重
    std::vector<bool> vecSuppressed(vecFiltered.size(), false);  // 20260330 ZJH 抑制标记
    std::vector<SODDetectionBox> vecKept;  // 20260330 ZJH 保留结果
    vecKept.reserve(vecFiltered.size());

    for (size_t i = 0; i < vecFiltered.size(); ++i) {
        if (vecSuppressed[i]) continue;  // 20260330 ZJH 已被抑制，跳过
        vecKept.push_back(vecFiltered[i]);  // 20260330 ZJH 保留当前最高分框

        // 20260330 ZJH 对后续所有框检查 IoU
        for (size_t j = i + 1; j < vecFiltered.size(); ++j) {
            if (vecSuppressed[j]) continue;  // 20260330 ZJH 已被抑制，跳过

            // 20260330 ZJH 类别感知: 只抑制同类别的框
            if (params.bClassAware && vecFiltered[i].nClassId != vecFiltered[j].nClassId) {
                continue;  // 20260330 ZJH 不同类别不互相抑制
            }

            // 20260330 ZJH 计算 IoU，超阈值则抑制
            float fIoU = computeSODBoxIoU(vecFiltered[i], vecFiltered[j]);
            if (fIoU > params.fNmsThreshold) {
                vecSuppressed[j] = true;  // 20260330 ZJH 标记为被抑制
            }
        }
    }

    return vecKept;  // 20260330 ZJH 返回合并去重后的检测框
}

// =========================================================
// mergeSegmentations — 合并多个切片的分割结果
// =========================================================

// 20260330 ZJH mergeSegmentations — 将各切片的分割掩码拼接到原图尺寸
// 重叠区域采用"投票机制": 对每个像素统计各切片给出的类别投票，取最多票数的类别
// 这样可以消除切片边缘处的分割不一致问题
// vecPatchMasks: 每个切片的分割掩码（int 类别 ID 数组，尺寸 = patch.nHeight × patch.nWidth）
// vecPatches: 对应的切片信息列表
// nImageH: 原图高度
// nImageW: 原图宽度
// nNumClasses: 类别总数（用于投票计数，0 表示背景类）
// 返回: 原图尺寸的分割掩码 [nImageH × nImageW]，每个像素为类别 ID
inline std::vector<int> mergeSegmentations(
        const std::vector<std::vector<int>>& vecPatchMasks,
        const std::vector<PatchInfo>& vecPatches,
        int nImageH, int nImageW, int nNumClasses) {
    // 20260330 ZJH 校验切片数量一致性
    if (vecPatchMasks.size() != vecPatches.size()) {
        throw std::invalid_argument("SOD mergeSegmentations: masks/patches size mismatch");
    }
    if (nNumClasses <= 0) {
        throw std::invalid_argument("SOD mergeSegmentations: nNumClasses must be positive");
    }

    // 20260330 ZJH 分配投票计数矩阵: [nImageH × nImageW × nNumClasses]
    // 每个像素位置对每个类别记录得票数
    int nPixelCount = nImageH * nImageW;  // 20260330 ZJH 原图像素总数
    std::vector<int> vecVotes(nPixelCount * nNumClasses, 0);  // 20260330 ZJH 全零初始化

    // 20260330 ZJH 逐切片累加投票
    for (size_t p = 0; p < vecPatchMasks.size(); ++p) {
        const auto& vecMask = vecPatchMasks[p];
        const auto& patch = vecPatches[p];

        // 20260330 ZJH 校验 mask 尺寸
        int nExpected = patch.nHeight * patch.nWidth;
        if (static_cast<int>(vecMask.size()) < nExpected) {
            continue;  // 20260330 ZJH 跳过尺寸不匹配的 mask（容错）
        }

        // 20260330 ZJH 逐像素累加投票（只处理有效区域，忽略填充部分）
        for (int y = 0; y < patch.nHeight; ++y) {
            for (int x = 0; x < patch.nWidth; ++x) {
                // 20260330 ZJH 原图坐标
                int nImgY = patch.nY + y;
                int nImgX = patch.nX + x;

                // 20260330 ZJH 边界检查（防止越界）
                if (nImgY >= nImageH || nImgX >= nImageW) continue;

                // 20260330 ZJH 获取该像素在此切片中的预测类别
                int nClassId = vecMask[y * patch.nWidth + x];
                // 20260330 ZJH 类别 ID 范围检查
                if (nClassId < 0 || nClassId >= nNumClasses) continue;

                // 20260330 ZJH 对该像素的该类别投票 +1
                int nVoteIdx = (nImgY * nImageW + nImgX) * nNumClasses + nClassId;
                vecVotes[nVoteIdx] += 1;
            }
        }
    }

    // 20260330 ZJH 逐像素取最多票数的类别作为最终结果
    std::vector<int> vecResult(nPixelCount, 0);  // 20260330 ZJH 默认为背景类 (0)
    for (int i = 0; i < nPixelCount; ++i) {
        int nBestClass = 0;       // 20260330 ZJH 最高票类别
        int nBestCount = 0;       // 20260330 ZJH 最高票数
        for (int c = 0; c < nNumClasses; ++c) {
            int nCount = vecVotes[i * nNumClasses + c];
            if (nCount > nBestCount) {
                nBestCount = nCount;   // 20260330 ZJH 更新最高票数
                nBestClass = c;        // 20260330 ZJH 更新最高票类别
            }
        }
        vecResult[i] = nBestClass;  // 20260330 ZJH 写入最终类别
    }

    return vecResult;  // 20260330 ZJH 返回原图尺寸的分割掩码
}

// =========================================================
// mergeAnomalyScores — 合并多个切片的异常分数图
// =========================================================

// 20260330 ZJH mergeAnomalyScores — 将各切片的异常分数热图拼接到原图尺寸
// 重叠区域采用"取最大值"策略: 保留最高异常分数，避免漏检
// 适用于异常检测任务（异常检测模型输出的是逐像素异常分数而非类别）
// vecPatchScores: 每个切片的异常分数图 (float，尺寸 = patch.nHeight × patch.nWidth)
// vecPatches: 对应的切片信息列表
// nImageH: 原图高度
// nImageW: 原图宽度
// 返回: 原图尺寸的异常分数图 [nImageH × nImageW]
inline std::vector<float> mergeAnomalyScores(
        const std::vector<std::vector<float>>& vecPatchScores,
        const std::vector<PatchInfo>& vecPatches,
        int nImageH, int nImageW) {
    // 20260330 ZJH 校验切片数量一致性
    if (vecPatchScores.size() != vecPatches.size()) {
        throw std::invalid_argument("SOD mergeAnomalyScores: scores/patches size mismatch");
    }

    // 20260330 ZJH 分配原图尺寸的分数图，初始化为 -1（表示未覆盖）
    int nPixelCount = nImageH * nImageW;
    std::vector<float> vecResult(nPixelCount, -1.0f);

    // 20260330 ZJH 逐切片合并分数
    for (size_t p = 0; p < vecPatchScores.size(); ++p) {
        const auto& vecScores = vecPatchScores[p];
        const auto& patch = vecPatches[p];

        // 20260330 ZJH 校验分数图尺寸
        int nExpected = patch.nHeight * patch.nWidth;
        if (static_cast<int>(vecScores.size()) < nExpected) {
            continue;  // 20260330 ZJH 跳过尺寸不匹配的分数图
        }

        // 20260330 ZJH 逐像素取最大值（重叠区域保留最高分）
        for (int y = 0; y < patch.nHeight; ++y) {
            for (int x = 0; x < patch.nWidth; ++x) {
                int nImgY = patch.nY + y;  // 20260330 ZJH 原图 y 坐标
                int nImgX = patch.nX + x;  // 20260330 ZJH 原图 x 坐标

                // 20260330 ZJH 边界检查
                if (nImgY >= nImageH || nImgX >= nImageW) continue;

                float fScore = vecScores[y * patch.nWidth + x];  // 20260330 ZJH 当前切片分数
                int nImgIdx = nImgY * nImageW + nImgX;            // 20260330 ZJH 原图像素索引

                // 20260330 ZJH 取最大值：重叠区域保留最高异常分数
                if (fScore > vecResult[nImgIdx]) {
                    vecResult[nImgIdx] = fScore;
                }
            }
        }
    }

    // 20260330 ZJH 将未覆盖区域（-1）设为 0（理论上全图都会被覆盖）
    for (auto& fVal : vecResult) {
        if (fVal < 0.0f) fVal = 0.0f;
    }

    return vecResult;  // 20260330 ZJH 返回原图尺寸的异常分数图
}

// =========================================================
// SODPipeline — SOD 全流程编排器（无状态工具类）
// =========================================================

// 20260330 ZJH SODPipeline — 封装 SOD 全流程的便捷接口
// 使用方法:
//   1. 构造时传入 SODParams
//   2. 调用 prepare() 生成切片列表
//   3. 用户逐切片调用模型推理
//   4. 调用 mergeDetectionResults() / mergeSegmentationResults() 合并结果
class SODPipeline {
public:
    // 20260330 ZJH 构造函数 — 保存 SOD 参数
    explicit SODPipeline(const SODParams& params)
        : m_params(params) {
        // 20260330 ZJH 构造时不做任何操作，等待 prepare() 调用
    }

    // 20260330 ZJH prepare — 根据图像尺寸生成切片列表
    // nImageW: 原图宽度
    // nImageH: 原图高度
    // 返回: 切片列表引用（内部存储）
    const std::vector<PatchInfo>& prepare(int nImageW, int nImageH) {
        m_nImageW = nImageW;  // 20260330 ZJH 缓存原图尺寸
        m_nImageH = nImageH;
        m_vecPatches = generatePatches(nImageW, nImageH, m_params);
        return m_vecPatches;  // 20260330 ZJH 返回切片列表引用
    }

    // 20260330 ZJH patchCount — 返回切片总数
    int patchCount() const {
        return static_cast<int>(m_vecPatches.size());
    }

    // 20260330 ZJH getPatch — 获取指定索引的切片信息
    const PatchInfo& getPatch(int nIdx) const {
        return m_vecPatches.at(nIdx);
    }

    // 20260330 ZJH params — 获取 SOD 参数（只读）
    const SODParams& params() const {
        return m_params;
    }

    // 20260330 ZJH mergeDetectionResults — 合并检测结果
    // vecPatchResults: 各切片的检测框列表
    // 返回: 合并去重后的原图坐标检测框
    std::vector<SODDetectionBox> mergeDetectionResults(
            const std::vector<std::vector<SODDetectionBox>>& vecPatchResults) const {
        return mergeDetections(vecPatchResults, m_vecPatches, m_params, m_nImageW, m_nImageH);
    }

    // 20260330 ZJH mergeSegmentationResults — 合并分割结果
    // vecPatchMasks: 各切片的分割掩码
    // nNumClasses: 类别总数
    // 返回: 原图尺寸的分割掩码
    std::vector<int> mergeSegmentationResults(
            const std::vector<std::vector<int>>& vecPatchMasks,
            int nNumClasses) const {
        return mergeSegmentations(vecPatchMasks, m_vecPatches, m_nImageH, m_nImageW, nNumClasses);
    }

    // 20260330 ZJH mergeAnomalyResults — 合并异常分数图结果
    // vecPatchScores: 各切片的异常分数图
    // 返回: 原图尺寸的异常分数图
    std::vector<float> mergeAnomalyResults(
            const std::vector<std::vector<float>>& vecPatchScores) const {
        return mergeAnomalyScores(vecPatchScores, m_vecPatches, m_nImageH, m_nImageW);
    }

private:
    SODParams m_params;                // 20260330 ZJH SOD 参数配置
    std::vector<PatchInfo> m_vecPatches;  // 20260330 ZJH 切片列表缓存
    int m_nImageW = 0;                 // 20260330 ZJH 原图宽度缓存
    int m_nImageH = 0;                 // 20260330 ZJH 原图高度缓存
};

}  // namespace om
