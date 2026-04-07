// 20260330 ZJH 智能标注工具 — 对标 MVTec "Hover & Click" + "Automated Mask Suggestion"
// 功能: 鼠标悬停自动生成候选区域、前景/背景点击生成 mask、Canny 边缘候选区域、简化 SLIC 超像素
// 纯 C++ 实现，不依赖 OpenCV/Halcon 等第三方库
#pragma once

#include <QObject>     // 20260330 ZJH 信号槽基类
#include <QPointF>     // 20260330 ZJH 浮点坐标
#include <QRectF>      // 20260330 ZJH 矩形区域
#include <QPolygonF>   // 20260330 ZJH 多边形区域
#include <QImage>      // 20260330 ZJH 图像数据
#include <QVector>     // 20260330 ZJH 动态数组

// 20260330 ZJH 智能标注工具
// 提供四种辅助标注机制:
//   1. Hover 模式: 鼠标悬停时基于灰度相似性自动推荐候选区域
//   2. Click 模式: 前景/背景点击 → FloodFill 生成精确 mask
//   3. 边缘检测: Canny 边缘 → 轮廓闭合 → 候选矩形列表
//   4. 超像素: 简化 SLIC 算法 → 候选多边形列表
class SmartLabelTool : public QObject
{
    Q_OBJECT

public:
    // 20260330 ZJH 构造函数
    explicit SmartLabelTool(QObject* pParent = nullptr);

    // 20260330 ZJH 析构函数
    ~SmartLabelTool() override = default;

    // 20260330 ZJH 设置当前图像（后续所有操作基于此图像）
    // 参数: image - 输入图像（支持 RGB/灰度）
    void setImage(const QImage& image);

    // 20260330 ZJH 获取当前图像是否已设置
    bool hasImage() const;

    // ===== Hover 模式 =====

    // 20260330 ZJH 鼠标悬停时自动生成候选区域
    // 以悬停点为种子，在局部窗口内查找灰度相似的连通区域，返回外接矩形
    // 参数: ptPos - 鼠标在图像坐标系中的位置
    // 返回: 推荐的候选区域矩形（无效时返回空矩形）
    QRectF suggestRegionAt(const QPointF& ptPos);

    // ===== Click 模式 =====

    // 20260330 ZJH 点击前景/背景点，使用 FloodFill 生成 mask
    // 前景点标记"属于目标"的区域，背景点标记"不属于目标"的区域
    // 通过多轮 FloodFill 合并前景、剔除背景，生成二值 mask
    // 参数: vecFgPoints - 前景点列表（用户标记的"属于目标"的点）
    //       vecBgPoints - 背景点列表（用户标记的"不属于目标"的点）
    // 返回: 二值 mask 图像（白色=目标区域, 黑色=背景）
    QImage generateMaskFromPoints(const QVector<QPointF>& vecFgPoints,
                                  const QVector<QPointF>& vecBgPoints);

    // ===== 边缘检测辅助 =====

    // 20260330 ZJH Canny 边缘 → 轮廓闭合 → 候选区域列表
    // 对图像执行 Sobel 梯度 + 非极大值抑制 + 双阈值 + 连通域分析
    // 参数: fThreshold - 边缘检测低阈值（高阈值 = 2 * fThreshold）
    // 返回: 候选区域矩形列表
    QVector<QRectF> detectCandidateRegions(float fThreshold = 50.0f);

    // ===== 超像素分割 =====

    // 20260330 ZJH 简化 SLIC 超像素生成候选区域
    // 使用 K-means 聚类将图像分割为若干超像素区域
    // 参数: nNumRegions - 目标超像素数量
    // 返回: 每个超像素区域的多边形边界
    QVector<QPolygonF> generateSuperpixels(int nNumRegions = 100);

    // ===== 参数调节 =====

    // 20260330 ZJH 设置 FloodFill 灰度容差（默认 20）
    void setFloodFillTolerance(int nTolerance);

    // 20260330 ZJH 设置 Hover 模式的搜索窗口半径（默认 50px）
    void setHoverSearchRadius(int nRadius);

signals:
    // 20260330 ZJH 候选区域推荐信号（Hover 模式触发）
    void regionSuggested(const QRectF& rect);

    // 20260330 ZJH mask 生成完成信号（Click 模式触发）
    void maskGenerated(const QImage& mask);

private:
    // ===== 内部工具函数 =====

    // 20260330 ZJH 将输入图像转为灰度图（内部缓存）
    void ensureGrayscale();

    // 20260330 ZJH FloodFill 算法：从种子点出发，填充灰度相似的连通区域
    // 参数: matGray - 灰度图像数据指针
    //       nWidth, nHeight - 图像尺寸
    //       ptSeed - 种子点
    //       nTolerance - 灰度容差
    //       pVisited - 访问标记数组（输出）
    void floodFill(const uchar* matGray, int nWidth, int nHeight,
                   const QPoint& ptSeed, int nTolerance,
                   QVector<bool>& pVisited);

    // 20260330 ZJH Sobel 3x3 梯度计算
    // 参数: matGray - 灰度图, nWidth/nHeight - 尺寸
    //       matGx, matGy - 输出 x/y 方向梯度（float 数组）
    void sobelGradient(const uchar* matGray, int nWidth, int nHeight,
                       QVector<float>& matGx, QVector<float>& matGy);

    // 20260330 ZJH 非极大值抑制（边缘细化）
    void nonMaxSuppression(const QVector<float>& matMag,
                           const QVector<float>& matAngle,
                           int nWidth, int nHeight,
                           QVector<float>& matNms);

    // 20260330 ZJH 双阈值 + 连通域标记
    void doubleThreshold(const QVector<float>& matNms,
                         int nWidth, int nHeight,
                         float fLow, float fHigh,
                         QVector<bool>& matEdge);

    // 20260330 ZJH 从二值边缘图提取连通域外接矩形
    QVector<QRectF> extractBoundingRects(const QVector<bool>& matEdge,
                                          int nWidth, int nHeight);

    // ===== 成员变量 =====
    QImage m_image;           // 20260330 ZJH 原始输入图像
    QImage m_grayImage;       // 20260330 ZJH 灰度缓存图像
    bool m_bGrayValid = false; // 20260330 ZJH 灰度缓存是否有效

    int m_nFloodTolerance = 20;  // 20260330 ZJH FloodFill 灰度容差
    int m_nHoverRadius = 50;     // 20260330 ZJH Hover 搜索窗口半径（像素）
};
