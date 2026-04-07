#pragma once
// 20260330 ZJH 多 ROI 推理支持 — 对标海康 ROIAssistant
// 支持在单张图像上定义多个感兴趣区域（最多 256 个），
// 分别裁剪送入模型推理，并将检测结果映射回原图坐标
// 应用场景: 大幅面图像分区检测、多工位同时检测、局部精细检测

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// 20260330 ZJH 引用同目录下的 DetectionBox 定义
#include "NmsUtils.h"

namespace om {

// =========================================================
// ROI 形状类型
// =========================================================

// 20260330 ZJH ROI 形状类型枚举 — 定义支持的 ROI 几何形状
enum class ROIType {
    Rectangle,   // 20260330 ZJH 矩形（含旋转矩形）
    Circle,      // 20260330 ZJH 圆形
    Polygon,     // 20260330 ZJH 任意多边形
    Fan          // 20260330 ZJH 扇形（圆弧区域）
};

// =========================================================
// RectROI — 矩形感兴趣区域
// =========================================================

// 20260330 ZJH 矩形 ROI 描述结构
// 使用左上角坐标 + 宽高 + 旋转角度定义
struct RectROI {
    float fX;              // 20260330 ZJH 左上角 x 坐标（像素）
    float fY;              // 20260330 ZJH 左上角 y 坐标（像素）
    float fWidth;          // 20260330 ZJH ROI 宽度（像素）
    float fHeight;         // 20260330 ZJH ROI 高度（像素）
    float fAngle = 0.0f;   // 20260330 ZJH 旋转角度（度，默认 0 = 无旋转）
    std::string strName;   // 20260330 ZJH ROI 名称（可选，用于显示和日志）
    bool bEnabled = true;  // 20260330 ZJH 是否启用（false 时跳过推理）

    // 20260330 ZJH 计算 ROI 面积
    float area() const {
        return fWidth * fHeight;  // 20260330 ZJH 宽 × 高
    }

    // 20260330 ZJH 检查 ROI 是否有效（宽高必须为正）
    bool isValid() const {
        return fWidth > 0.0f && fHeight > 0.0f;  // 20260330 ZJH 宽高需大于零
    }
};

// =========================================================
// MultiROIManager — 多 ROI 管理器
// =========================================================

// 20260330 ZJH MultiROIManager — 管理多个 ROI 的添加、删除、裁剪和坐标映射
// 最多支持 256 个 ROI（对标海康 ROIAssistant 规格）
class MultiROIManager {
public:
    // 20260330 ZJH 最大 ROI 数量限制
    static constexpr int MAX_ROIS = 256;

    // =====================================================
    // 添加 / 删除 / 清空
    // =====================================================

    // 20260330 ZJH addRectROI — 添加一个矩形 ROI
    // roi: 要添加的矩形 ROI 描述
    // 返回: 新 ROI 的索引（0-based），失败返回 -1
    int addRectROI(const RectROI& roi) {
        // 20260330 ZJH 检查是否超过上限
        if (static_cast<int>(m_vecROIs.size()) >= MAX_ROIS) {
            return -1;  // 20260330 ZJH 已满，拒绝添加
        }
        // 20260330 ZJH 检查 ROI 有效性
        if (!roi.isValid()) {
            return -1;  // 20260330 ZJH 无效 ROI（宽高 <= 0）
        }
        m_vecROIs.push_back(roi);  // 20260330 ZJH 加入列表末尾
        return static_cast<int>(m_vecROIs.size()) - 1;  // 20260330 ZJH 返回新索引
    }

    // 20260330 ZJH removeROI — 按索引删除一个 ROI
    // nIndex: 要删除的 ROI 索引
    void removeROI(int nIndex) {
        // 20260330 ZJH 范围检查
        if (nIndex < 0 || nIndex >= static_cast<int>(m_vecROIs.size())) {
            return;  // 20260330 ZJH 无效索引，安全忽略
        }
        // 20260330 ZJH 从列表中删除
        m_vecROIs.erase(m_vecROIs.begin() + nIndex);
    }

    // 20260330 ZJH clearAll — 清空所有 ROI
    void clearAll() {
        m_vecROIs.clear();  // 20260330 ZJH 清空列表
    }

    // =====================================================
    // 查询接口
    // =====================================================

    // 20260330 ZJH getROICount — 获取当前 ROI 数量
    int getROICount() const {
        return static_cast<int>(m_vecROIs.size());
    }

    // 20260330 ZJH getROI — 获取指定索引的 ROI
    // nIndex: ROI 索引
    // 返回: ROI 副本（越界时返回默认空 ROI）
    RectROI getROI(int nIndex) const {
        // 20260330 ZJH 范围检查
        if (nIndex < 0 || nIndex >= static_cast<int>(m_vecROIs.size())) {
            return RectROI{};  // 20260330 ZJH 越界返回默认值
        }
        return m_vecROIs[nIndex];  // 20260330 ZJH 返回副本
    }

    // 20260330 ZJH setROI — 修改指定索引的 ROI
    // nIndex: ROI 索引
    // roi: 新的 ROI 描述
    // 返回: 是否修改成功
    bool setROI(int nIndex, const RectROI& roi) {
        // 20260330 ZJH 范围检查
        if (nIndex < 0 || nIndex >= static_cast<int>(m_vecROIs.size())) {
            return false;  // 20260330 ZJH 越界
        }
        // 20260330 ZJH 有效性检查
        if (!roi.isValid()) {
            return false;  // 20260330 ZJH 无效 ROI
        }
        m_vecROIs[nIndex] = roi;  // 20260330 ZJH 覆盖
        return true;
    }

    // 20260330 ZJH getEnabledROIs — 获取所有启用的 ROI 索引列表
    // 返回: 启用的 ROI 索引 vector
    std::vector<int> getEnabledROIs() const {
        std::vector<int> vecIndices;  // 20260330 ZJH 结果列表
        vecIndices.reserve(m_vecROIs.size());  // 20260330 ZJH 预分配
        for (int i = 0; i < static_cast<int>(m_vecROIs.size()); ++i) {
            if (m_vecROIs[i].bEnabled) {
                vecIndices.push_back(i);  // 20260330 ZJH 只收集启用的
            }
        }
        return vecIndices;
    }

    // =====================================================
    // 图像裁剪
    // =====================================================

    // 20260330 ZJH cropROI — 从 CHW float 图像中裁剪指定 ROI 区域并缩放到目标尺寸
    // vecImage: 输入图像数据（CHW 布局，float，像素值已归一化）
    // nC: 通道数（1 或 3）
    // nH: 图像高度
    // nW: 图像宽度
    // nROIIndex: 要裁剪的 ROI 索引
    // nTargetSize: 目标正方形边长（模型输入尺寸，如 224 或 640）
    // 返回: 裁剪并缩放后的图像数据（CHW 布局，nTargetSize × nTargetSize）
    std::vector<float> cropROI(const std::vector<float>& vecImage,
                               int nC, int nH, int nW,
                               int nROIIndex, int nTargetSize) const {
        // 20260330 ZJH 参数校验
        if (nROIIndex < 0 || nROIIndex >= static_cast<int>(m_vecROIs.size())) {
            throw std::out_of_range("MultiROIManager::cropROI — ROI index out of range");
        }
        // 20260330 ZJH 验证输入图像尺寸
        if (static_cast<int>(vecImage.size()) != nC * nH * nW) {
            throw std::invalid_argument("MultiROIManager::cropROI — image size mismatch");
        }

        const RectROI& roi = m_vecROIs[nROIIndex];  // 20260330 ZJH 取出目标 ROI

        // 20260330 ZJH 计算裁剪区域的整数像素坐标（裁剪到图像边界）
        int nRoiX1 = std::max(0, static_cast<int>(std::floor(roi.fX)));
        int nRoiY1 = std::max(0, static_cast<int>(std::floor(roi.fY)));
        int nRoiX2 = std::min(nW, static_cast<int>(std::ceil(roi.fX + roi.fWidth)));
        int nRoiY2 = std::min(nH, static_cast<int>(std::ceil(roi.fY + roi.fHeight)));

        // 20260330 ZJH 裁剪区域的实际尺寸
        int nCropW = nRoiX2 - nRoiX1;  // 20260330 ZJH 裁剪宽度
        int nCropH = nRoiY2 - nRoiY1;  // 20260330 ZJH 裁剪高度

        // 20260330 ZJH 退化情况: 裁剪区域为空
        if (nCropW <= 0 || nCropH <= 0) {
            return std::vector<float>(nC * nTargetSize * nTargetSize, 0.0f);
        }

        // 20260330 ZJH 分配输出缓冲区（CHW 布局）
        std::vector<float> vecCropped(nC * nTargetSize * nTargetSize, 0.0f);

        // 20260330 ZJH 双线性插值缩放: 将 nCropW×nCropH 缩放到 nTargetSize×nTargetSize
        float fScaleX = static_cast<float>(nCropW) / static_cast<float>(nTargetSize);
        float fScaleY = static_cast<float>(nCropH) / static_cast<float>(nTargetSize);

        for (int c = 0; c < nC; ++c) {
            for (int ty = 0; ty < nTargetSize; ++ty) {
                for (int tx = 0; tx < nTargetSize; ++tx) {
                    // 20260330 ZJH 计算源图像中的浮点坐标（对齐到裁剪区域内部）
                    float fSrcX = nRoiX1 + (tx + 0.5f) * fScaleX - 0.5f;
                    float fSrcY = nRoiY1 + (ty + 0.5f) * fScaleY - 0.5f;

                    // 20260330 ZJH 双线性插值的四个邻居像素坐标
                    int nX0 = static_cast<int>(std::floor(fSrcX));
                    int nY0 = static_cast<int>(std::floor(fSrcY));
                    int nX1 = nX0 + 1;
                    int nY1 = nY0 + 1;

                    // 20260330 ZJH 裁剪到图像边界（防止越界）
                    nX0 = std::max(0, std::min(nX0, nW - 1));
                    nY0 = std::max(0, std::min(nY0, nH - 1));
                    nX1 = std::max(0, std::min(nX1, nW - 1));
                    nY1 = std::max(0, std::min(nY1, nH - 1));

                    // 20260330 ZJH 插值权重
                    float fAlpha = fSrcX - std::floor(fSrcX);  // 20260330 ZJH 水平权重
                    float fBeta  = fSrcY - std::floor(fSrcY);  // 20260330 ZJH 垂直权重

                    // 20260330 ZJH CHW 布局下的偏移计算
                    int nPlaneOffset = c * nH * nW;  // 20260330 ZJH 当前通道在源图像中的起始偏移

                    // 20260330 ZJH 读取四个邻居像素值
                    float fVal00 = vecImage[nPlaneOffset + nY0 * nW + nX0];  // 20260330 ZJH 左上
                    float fVal01 = vecImage[nPlaneOffset + nY0 * nW + nX1];  // 20260330 ZJH 右上
                    float fVal10 = vecImage[nPlaneOffset + nY1 * nW + nX0];  // 20260330 ZJH 左下
                    float fVal11 = vecImage[nPlaneOffset + nY1 * nW + nX1];  // 20260330 ZJH 右下

                    // 20260330 ZJH 双线性插值公式
                    float fInterp = fVal00 * (1.0f - fAlpha) * (1.0f - fBeta)
                                  + fVal01 * fAlpha * (1.0f - fBeta)
                                  + fVal10 * (1.0f - fAlpha) * fBeta
                                  + fVal11 * fAlpha * fBeta;

                    // 20260330 ZJH 写入输出（CHW 布局）
                    vecCropped[c * nTargetSize * nTargetSize + ty * nTargetSize + tx] = fInterp;
                }
            }
        }

        return vecCropped;  // 20260330 ZJH 返回裁剪并缩放后的图像
    }

    // =====================================================
    // 坐标映射
    // =====================================================

    // 20260330 ZJH mapToOriginal — 将 ROI 内的检测框坐标映射回原图坐标系
    // 检测框在 ROI 裁剪图上的坐标需要加上 ROI 左上角偏移
    // vecBoxes: 检测框列表（会被就地修改，坐标从 ROI 局部转为原图全局）
    // nROIIndex: ROI 索引
    // nModelInputSize: 模型输入尺寸（用于计算缩放比例，0 = 不缩放）
    void mapToOriginal(std::vector<DetectionBox>& vecBoxes, int nROIIndex,
                       int nModelInputSize = 0) const {
        // 20260330 ZJH 范围检查
        if (nROIIndex < 0 || nROIIndex >= static_cast<int>(m_vecROIs.size())) {
            return;  // 20260330 ZJH 无效索引，不做映射
        }

        const RectROI& roi = m_vecROIs[nROIIndex];  // 20260330 ZJH 取出对应 ROI

        // 20260330 ZJH 计算缩放比例（模型输入尺寸到 ROI 实际尺寸）
        float fScaleX = 1.0f;  // 20260330 ZJH 默认不缩放
        float fScaleY = 1.0f;
        if (nModelInputSize > 0) {
            fScaleX = roi.fWidth / static_cast<float>(nModelInputSize);   // 20260330 ZJH 水平缩放
            fScaleY = roi.fHeight / static_cast<float>(nModelInputSize);  // 20260330 ZJH 垂直缩放
        }

        // 20260330 ZJH 逐框映射坐标
        for (auto& box : vecBoxes) {
            // 20260330 ZJH 先缩放回 ROI 尺寸，再平移到原图坐标
            box.fX1 = box.fX1 * fScaleX + roi.fX;  // 20260330 ZJH 左上 x
            box.fY1 = box.fY1 * fScaleY + roi.fY;  // 20260330 ZJH 左上 y
            box.fX2 = box.fX2 * fScaleX + roi.fX;  // 20260330 ZJH 右下 x
            box.fY2 = box.fY2 * fScaleY + roi.fY;  // 20260330 ZJH 右下 y
        }
    }

    // 20260330 ZJH mapPointToOriginal — 将 ROI 内的单点坐标映射回原图
    // fX, fY: ROI 局部坐标
    // nROIIndex: ROI 索引
    // nModelInputSize: 模型输入尺寸
    // outX, outY: 输出原图坐标
    void mapPointToOriginal(float fX, float fY, int nROIIndex,
                            int nModelInputSize,
                            float& outX, float& outY) const {
        // 20260330 ZJH 范围检查
        if (nROIIndex < 0 || nROIIndex >= static_cast<int>(m_vecROIs.size())) {
            outX = fX;  // 20260330 ZJH 无效索引，原样返回
            outY = fY;
            return;
        }

        const RectROI& roi = m_vecROIs[nROIIndex];

        // 20260330 ZJH 缩放 + 平移
        float fScaleX = (nModelInputSize > 0) ? roi.fWidth / static_cast<float>(nModelInputSize) : 1.0f;
        float fScaleY = (nModelInputSize > 0) ? roi.fHeight / static_cast<float>(nModelInputSize) : 1.0f;
        outX = fX * fScaleX + roi.fX;
        outY = fY * fScaleY + roi.fY;
    }

    // =====================================================
    // 批量推理辅助
    // =====================================================

    // 20260330 ZJH cropAllEnabled — 裁剪所有启用的 ROI，返回批量图像数据
    // vecImage: 输入 CHW float 图像
    // nC, nH, nW: 通道数、高度、宽度
    // nTargetSize: 模型输入尺寸
    // vecROIIndices: 输出参数，记录每个裁剪对应的 ROI 索引
    // 返回: 拼接后的批量图像数据 [N, C, nTargetSize, nTargetSize]
    std::vector<float> cropAllEnabled(const std::vector<float>& vecImage,
                                      int nC, int nH, int nW,
                                      int nTargetSize,
                                      std::vector<int>& vecROIIndices) const {
        vecROIIndices.clear();  // 20260330 ZJH 清空输出索引

        // 20260330 ZJH 收集所有启用的 ROI 索引
        std::vector<int> vecEnabled = getEnabledROIs();

        // 20260330 ZJH 单张裁剪图的元素数量
        int nCropElements = nC * nTargetSize * nTargetSize;

        // 20260330 ZJH 分配批量输出缓冲区
        std::vector<float> vecBatch(vecEnabled.size() * nCropElements, 0.0f);

        // 20260330 ZJH 逐 ROI 裁剪并拷贝到批量缓冲区
        for (size_t i = 0; i < vecEnabled.size(); ++i) {
            int nIdx = vecEnabled[i];  // 20260330 ZJH 当前 ROI 索引
            auto vecCrop = cropROI(vecImage, nC, nH, nW, nIdx, nTargetSize);

            // 20260330 ZJH 拷贝到批量缓冲区对应位置
            std::copy(vecCrop.begin(), vecCrop.end(),
                      vecBatch.begin() + static_cast<ptrdiff_t>(i * nCropElements));

            vecROIIndices.push_back(nIdx);  // 20260330 ZJH 记录索引映射
        }

        return vecBatch;  // 20260330 ZJH 返回 [N, C, H, W] 批量数据
    }

private:
    std::vector<RectROI> m_vecROIs;  // 20260330 ZJH ROI 列表（有序）
};

}  // namespace om
