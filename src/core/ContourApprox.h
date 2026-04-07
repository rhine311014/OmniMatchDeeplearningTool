#pragma once
// 20260330 ZJH 轮廓近似与分析 — 对标海康 ContourApproxCpp
// 纯 C++ 实现，零第三方依赖
// 功能: Douglas-Peucker 轮廓简化、二值 mask 轮廓提取、
//       面积/周长计算、最小外接矩形、凸包
// 应用场景: 分割后处理、缺陷轮廓分析、尺寸测量

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace om {

// =========================================================
// Point2f — 二维浮点坐标点
// =========================================================

// 20260330 ZJH 二维浮点坐标点
struct Point2f {
    float fX = 0.0f;  // 20260330 ZJH x 坐标
    float fY = 0.0f;  // 20260330 ZJH y 坐标

    // 20260330 ZJH 向量减法
    Point2f operator-(const Point2f& other) const {
        return { fX - other.fX, fY - other.fY };
    }

    // 20260330 ZJH 向量加法
    Point2f operator+(const Point2f& other) const {
        return { fX + other.fX, fY + other.fY };
    }

    // 20260330 ZJH 标量乘法
    Point2f operator*(float fScale) const {
        return { fX * fScale, fY * fScale };
    }
};

// =========================================================
// RotatedRect — 最小外接旋转矩形
// =========================================================

// 20260330 ZJH 旋转矩形描述结构
struct RotatedRect {
    Point2f ptCenter;      // 20260330 ZJH 中心点坐标
    float fWidth = 0.0f;   // 20260330 ZJH 宽度（较长边）
    float fHeight = 0.0f;  // 20260330 ZJH 高度（较短边）
    float fAngle = 0.0f;   // 20260330 ZJH 旋转角度（度，逆时针为正）
};

// =========================================================
// 内部辅助函数（匿名命名空间，不导出）
// =========================================================

namespace detail {

// 20260330 ZJH 计算点到线段的垂直距离
// pt: 目标点
// ptLineStart: 线段起点
// ptLineEnd: 线段终点
// 返回: 点到线段所在直线的垂直距离
inline float pointToLineDistance(const Point2f& pt,
                                const Point2f& ptLineStart,
                                const Point2f& ptLineEnd) {
    // 20260330 ZJH 线段向量
    float fDx = ptLineEnd.fX - ptLineStart.fX;
    float fDy = ptLineEnd.fY - ptLineStart.fY;

    // 20260330 ZJH 线段长度
    float fLen = std::sqrt(fDx * fDx + fDy * fDy);

    // 20260330 ZJH 退化情况: 线段长度为零（起终点重合）
    if (fLen < 1e-8f) {
        // 20260330 ZJH 返回点到起点的欧氏距离
        float fPx = pt.fX - ptLineStart.fX;
        float fPy = pt.fY - ptLineStart.fY;
        return std::sqrt(fPx * fPx + fPy * fPy);
    }

    // 20260330 ZJH 使用叉积计算点到直线的距离: |cross| / |line|
    float fCross = std::abs((pt.fX - ptLineStart.fX) * fDy - (pt.fY - ptLineStart.fY) * fDx);
    return fCross / fLen;
}

// 20260330 ZJH Douglas-Peucker 递归实现
// vecPoints: 原始点集（引用，只读）
// nStart: 当前段起始索引
// nEnd: 当前段终止索引
// fEpsilon: 简化阈值（距离容差）
// vecKeep: 输出标记数组（true = 保留该点）
// 20260330 ZJH 新增 nMaxDepth 参数防止恶意/退化输入导致栈溢出
inline void douglasPuckerRecursive(const std::vector<Point2f>& vecPoints,
                                   int nStart, int nEnd, float fEpsilon,
                                   std::vector<bool>& vecKeep,
                                   int nDepth = 0, int nMaxDepth = 64) {
    // 20260330 ZJH 基线情况: 段内无中间点
    if (nEnd - nStart <= 1) {
        return;  // 20260330 ZJH 无中间点可检查
    }

    // 20260330 ZJH 递归深度保护: 超过最大深度停止细分，防止栈溢出
    if (nDepth >= nMaxDepth) {
        return;  // 20260330 ZJH 达到深度上限，停止递归
    }

    // 20260330 ZJH 找到距离基线最远的中间点
    float fMaxDist = 0.0f;   // 20260330 ZJH 最大距离
    int nMaxIdx = nStart;    // 20260330 ZJH 最远点索引

    for (int i = nStart + 1; i < nEnd; ++i) {
        float fDist = pointToLineDistance(vecPoints[i], vecPoints[nStart], vecPoints[nEnd]);
        if (fDist > fMaxDist) {
            fMaxDist = fDist;  // 20260330 ZJH 更新最大距离
            nMaxIdx = i;        // 20260330 ZJH 更新最远点索引
        }
    }

    // 20260330 ZJH 如果最大距离超过阈值，保留该点并递归两段
    if (fMaxDist > fEpsilon) {
        vecKeep[nMaxIdx] = true;  // 20260330 ZJH 标记保留
        // 20260330 ZJH 递归处理左半段（起点到最远点），深度 +1
        douglasPuckerRecursive(vecPoints, nStart, nMaxIdx, fEpsilon, vecKeep, nDepth + 1, nMaxDepth);
        // 20260330 ZJH 递归处理右半段（最远点到终点），深度 +1
        douglasPuckerRecursive(vecPoints, nMaxIdx, nEnd, fEpsilon, vecKeep, nDepth + 1, nMaxDepth);
    }
    // 20260330 ZJH 否则: 最大距离 <= epsilon，该段所有中间点被丢弃
}

// 20260330 ZJH 二维叉积 (用于凸包排序)
// O -> A -> B 的叉积值
inline float cross2D(const Point2f& ptO, const Point2f& ptA, const Point2f& ptB) {
    return (ptA.fX - ptO.fX) * (ptB.fY - ptO.fY) -
           (ptA.fY - ptO.fY) * (ptB.fX - ptO.fX);
}

}  // namespace detail

// =========================================================
// approxContour — Douglas-Peucker 轮廓简化
// =========================================================

// 20260330 ZJH approxContour — 使用 Douglas-Peucker 算法简化轮廓
// 将密集的轮廓点集简化为少量关键点，保持轮廓整体形状
// vecPoints: 输入轮廓点集（有序）
// fEpsilon: 简化阈值（越大越简化，单位: 像素）
//           推荐值: 轮廓周长的 0.5%~2%
// 返回: 简化后的轮廓点集
inline std::vector<Point2f> approxContour(const std::vector<Point2f>& vecPoints,
                                          float fEpsilon) {
    // 20260330 ZJH 退化情况: 点太少无需简化
    if (vecPoints.size() <= 2) {
        return vecPoints;  // 20260330 ZJH 原样返回
    }

    // 20260330 ZJH 初始化保留标记: 首尾点必须保留
    int nSize = static_cast<int>(vecPoints.size());
    std::vector<bool> vecKeep(nSize, false);
    vecKeep[0] = true;            // 20260330 ZJH 保留起点
    vecKeep[nSize - 1] = true;    // 20260330 ZJH 保留终点

    // 20260330 ZJH 递归执行 Douglas-Peucker 算法
    detail::douglasPuckerRecursive(vecPoints, 0, nSize - 1, fEpsilon, vecKeep);

    // 20260330 ZJH 收集保留的点
    std::vector<Point2f> vecResult;
    vecResult.reserve(nSize / 4);  // 20260330 ZJH 预估简化后约为原始的 25%
    for (int i = 0; i < nSize; ++i) {
        if (vecKeep[i]) {
            vecResult.push_back(vecPoints[i]);  // 20260330 ZJH 只收集标记为保留的点
        }
    }

    return vecResult;  // 20260330 ZJH 返回简化后的轮廓
}

// =========================================================
// findContours — 从二值 mask 提取轮廓点
// =========================================================

// 20260330 ZJH findContours — 从二值 mask 图像提取外轮廓点集
// 使用 Moore 邻域追踪算法（8-连通）提取连通区域的外边界
// vecMask: 二值 mask（0=背景, 非0=前景），行主序存储
// nW: mask 宽度
// nH: mask 高度
// 返回: 轮廓点集列表（每个连通区域一条轮廓）
inline std::vector<std::vector<Point2f>> findContours(const std::vector<uint8_t>& vecMask,
                                                       int nW, int nH) {
    // 20260330 ZJH 参数校验
    if (static_cast<int>(vecMask.size()) != nW * nH) {
        throw std::invalid_argument("findContours — mask size mismatch");
    }

    std::vector<std::vector<Point2f>> vecContours;  // 20260330 ZJH 所有轮廓的集合

    // 20260330 ZJH 已访问标记（防止同一轮廓被重复提取）
    std::vector<bool> vecVisited(nW * nH, false);

    // 20260330 ZJH Moore 邻域 8 个方向偏移（顺时针: 右、右下、下、左下、左、左上、上、右上）
    static const int s_arrDx[8] = { 1,  1,  0, -1, -1, -1,  0,  1 };
    static const int s_arrDy[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };

    // 20260330 ZJH 扫描所有像素寻找轮廓起始点
    for (int y = 0; y < nH; ++y) {
        for (int x = 0; x < nW; ++x) {
            int nPixelIdx = y * nW + x;  // 20260330 ZJH 当前像素索引

            // 20260330 ZJH 跳过条件: 背景像素、已访问、不是左边界
            // 左边界判定: 像素为前景，且左邻像素为背景或已到左边界
            if (vecMask[nPixelIdx] == 0) continue;        // 20260330 ZJH 背景跳过
            if (vecVisited[nPixelIdx]) continue;            // 20260330 ZJH 已处理跳过
            if (x > 0 && vecMask[y * nW + (x - 1)] != 0) continue;  // 20260330 ZJH 非左边界跳过

            // 20260330 ZJH 找到一个新轮廓的起始点，开始 Moore 邻域追踪
            std::vector<Point2f> vecContour;  // 20260330 ZJH 当前轮廓的点集
            int nCurX = x;     // 20260330 ZJH 当前追踪位置 x
            int nCurY = y;     // 20260330 ZJH 当前追踪位置 y
            int nDir = 0;      // 20260330 ZJH 初始搜索方向（右）

            int nMaxIter = nW * nH * 2;  // 20260330 ZJH 最大迭代次数（防止死循环）
            int nIter = 0;               // 20260330 ZJH 当前迭代计数

            do {
                // 20260330 ZJH 记录当前点
                vecContour.push_back({ static_cast<float>(nCurX), static_cast<float>(nCurY) });
                vecVisited[nCurY * nW + nCurX] = true;  // 20260330 ZJH 标记已访问

                // 20260330 ZJH 从上一步的反方向开始顺时针搜索下一个前景邻居
                int nStartDir = (nDir + 5) % 8;  // 20260330 ZJH 反方向再逆时针转一步
                bool bFound = false;  // 20260330 ZJH 是否找到下一个前景点

                for (int d = 0; d < 8; ++d) {
                    int nTestDir = (nStartDir + d) % 8;  // 20260330 ZJH 顺时针逐个方向测试
                    int nNx = nCurX + s_arrDx[nTestDir];  // 20260330 ZJH 邻居 x
                    int nNy = nCurY + s_arrDy[nTestDir];  // 20260330 ZJH 邻居 y

                    // 20260330 ZJH 边界检查
                    if (nNx < 0 || nNx >= nW || nNy < 0 || nNy >= nH) continue;

                    // 20260330 ZJH 找到前景邻居
                    if (vecMask[nNy * nW + nNx] != 0) {
                        nCurX = nNx;   // 20260330 ZJH 移动到邻居
                        nCurY = nNy;
                        nDir = nTestDir;  // 20260330 ZJH 记录移动方向
                        bFound = true;
                        break;
                    }
                }

                // 20260330 ZJH 孤立点（无前景邻居），结束追踪
                if (!bFound) break;

                ++nIter;
            } while ((nCurX != x || nCurY != y) && nIter < nMaxIter);
            // 20260330 ZJH 回到起始点或达到最大迭代次数时停止

            // 20260330 ZJH 只保留有效轮廓（至少 3 个点才构成区域）
            if (vecContour.size() >= 3) {
                vecContours.push_back(std::move(vecContour));
            }
        }
    }

    return vecContours;  // 20260330 ZJH 返回所有提取的轮廓
}

// =========================================================
// contourArea — 计算轮廓面积（Shoelace 公式）
// =========================================================

// 20260330 ZJH contourArea — 使用 Shoelace（鞋带）公式计算多边形面积
// vecContour: 有序轮廓点集（顺时针或逆时针均可）
// 返回: 面积（始终为正值，单位: 像素²）
inline float contourArea(const std::vector<Point2f>& vecContour) {
    int nSize = static_cast<int>(vecContour.size());  // 20260330 ZJH 点数
    // 20260330 ZJH 少于 3 个点无法构成多边形
    if (nSize < 3) {
        return 0.0f;
    }

    // 20260330 ZJH Shoelace 公式: Area = 0.5 * |Σ(xi*yi+1 - xi+1*yi)|
    float fSum = 0.0f;
    for (int i = 0; i < nSize; ++i) {
        int j = (i + 1) % nSize;  // 20260330 ZJH 下一个点（循环到首点）
        // 20260330 ZJH 叉积累加
        fSum += vecContour[i].fX * vecContour[j].fY;
        fSum -= vecContour[j].fX * vecContour[i].fY;
    }

    return std::abs(fSum) * 0.5f;  // 20260330 ZJH 取绝对值并除以 2
}

// =========================================================
// contourPerimeter — 计算轮廓周长
// =========================================================

// 20260330 ZJH contourPerimeter — 计算闭合轮廓的周长
// vecContour: 有序轮廓点集
// bClosed: 是否闭合（true = 首尾相连，默认 true）
// 返回: 周长（单位: 像素）
inline float contourPerimeter(const std::vector<Point2f>& vecContour, bool bClosed = true) {
    int nSize = static_cast<int>(vecContour.size());
    if (nSize < 2) {
        return 0.0f;  // 20260330 ZJH 单点无周长
    }

    float fPerimeter = 0.0f;  // 20260330 ZJH 累加周长

    // 20260330 ZJH 逐段计算相邻点之间的欧氏距离
    int nEnd = bClosed ? nSize : nSize - 1;  // 20260330 ZJH 闭合时多算首尾连线
    for (int i = 0; i < nEnd; ++i) {
        int j = (i + 1) % nSize;  // 20260330 ZJH 下一个点
        float fDx = vecContour[j].fX - vecContour[i].fX;
        float fDy = vecContour[j].fY - vecContour[i].fY;
        fPerimeter += std::sqrt(fDx * fDx + fDy * fDy);  // 20260330 ZJH 欧氏距离
    }

    return fPerimeter;  // 20260330 ZJH 返回总周长
}

// =========================================================
// minAreaRect — 计算最小外接旋转矩形
// =========================================================

// 20260330 ZJH minAreaRect — 计算轮廓点集的最小面积外接矩形
// 使用旋转卡尺（Rotating Calipers）算法，基于凸包
// vecContour: 输入轮廓点集
// 返回: 最小外接旋转矩形（中心、宽、高、角度）
inline RotatedRect minAreaRect(const std::vector<Point2f>& vecContour);

// 20260330 ZJH 前置声明 convexHull，因为 minAreaRect 依赖它
inline std::vector<Point2f> convexHull(const std::vector<Point2f>& vecPoints);

// =========================================================
// convexHull — 计算轮廓凸包
// =========================================================

// 20260330 ZJH convexHull — 使用 Andrew's Monotone Chain 算法计算凸包
// 时间复杂度: O(n log n)，其中 n 为输入点数
// vecPoints: 输入点集
// 返回: 凸包顶点（逆时针排列）
inline std::vector<Point2f> convexHull(const std::vector<Point2f>& vecPoints) {
    int nSize = static_cast<int>(vecPoints.size());
    // 20260330 ZJH 退化情况
    if (nSize < 3) {
        return vecPoints;  // 20260330 ZJH 不足 3 点无法构成凸包
    }

    // 20260330 ZJH 复制并按 x 坐标排序（x 相同时按 y 排序）
    std::vector<Point2f> vecSorted = vecPoints;
    std::sort(vecSorted.begin(), vecSorted.end(),
              [](const Point2f& a, const Point2f& b) {
                  return (a.fX < b.fX) || (a.fX == b.fX && a.fY < b.fY);
              });

    // 20260330 ZJH Andrew's Monotone Chain 算法
    std::vector<Point2f> vecHull(2 * nSize);  // 20260330 ZJH 最多 2n 个点
    int k = 0;  // 20260330 ZJH 当前凸包点数

    // 20260330 ZJH 构建下凸壳（从左到右）
    for (int i = 0; i < nSize; ++i) {
        // 20260330 ZJH 如果新点在已有方向的右侧（非左转），弹出上一个点
        while (k >= 2 && detail::cross2D(vecHull[k - 2], vecHull[k - 1], vecSorted[i]) <= 0.0f) {
            --k;
        }
        vecHull[k++] = vecSorted[i];  // 20260330 ZJH 压入新点
    }

    // 20260330 ZJH 构建上凸壳（从右到左）
    int nLower = k + 1;  // 20260330 ZJH 下凸壳大小 + 1（上凸壳的起始判断点）
    for (int i = nSize - 2; i >= 0; --i) {
        while (k >= nLower && detail::cross2D(vecHull[k - 2], vecHull[k - 1], vecSorted[i]) <= 0.0f) {
            --k;
        }
        vecHull[k++] = vecSorted[i];  // 20260330 ZJH 压入新点
    }

    // 20260330 ZJH 最后一个点与第一个点重合，去掉
    vecHull.resize(k - 1);
    return vecHull;  // 20260330 ZJH 返回凸包顶点（逆时针）
}

// 20260330 ZJH minAreaRect 完整实现
// 基于凸包的旋转卡尺算法，遍历凸包每条边作为候选方向，
// 找出最小面积的外接矩形
inline RotatedRect minAreaRect(const std::vector<Point2f>& vecContour) {
    // 20260330 ZJH 退化情况
    if (vecContour.empty()) {
        return RotatedRect{};  // 20260330 ZJH 空轮廓返回默认值
    }
    if (vecContour.size() == 1) {
        return RotatedRect{ vecContour[0], 0.0f, 0.0f, 0.0f };  // 20260330 ZJH 单点
    }
    if (vecContour.size() == 2) {
        // 20260330 ZJH 两点: 矩形退化为线段
        float fDx = vecContour[1].fX - vecContour[0].fX;
        float fDy = vecContour[1].fY - vecContour[0].fY;
        float fLen = std::sqrt(fDx * fDx + fDy * fDy);
        float fAngle = std::atan2(fDy, fDx) * 180.0f / 3.14159265f;
        Point2f ptCenter = { (vecContour[0].fX + vecContour[1].fX) * 0.5f,
                             (vecContour[0].fY + vecContour[1].fY) * 0.5f };
        return RotatedRect{ ptCenter, fLen, 0.0f, fAngle };
    }

    // 20260330 ZJH 步骤 1: 计算凸包
    auto vecHull = convexHull(vecContour);
    int nHullSize = static_cast<int>(vecHull.size());

    if (nHullSize < 3) {
        // 20260330 ZJH 凸包退化（共线点），返回 AABB
        float fMinX = vecHull[0].fX, fMaxX = vecHull[0].fX;
        float fMinY = vecHull[0].fY, fMaxY = vecHull[0].fY;
        for (const auto& pt : vecHull) {
            fMinX = std::min(fMinX, pt.fX);
            fMaxX = std::max(fMaxX, pt.fX);
            fMinY = std::min(fMinY, pt.fY);
            fMaxY = std::max(fMaxY, pt.fY);
        }
        return RotatedRect{
            { (fMinX + fMaxX) * 0.5f, (fMinY + fMaxY) * 0.5f },
            fMaxX - fMinX, fMaxY - fMinY, 0.0f
        };
    }

    // 20260330 ZJH 步骤 2: 旋转卡尺 — 遍历凸包每条边作为参考方向
    float fBestArea = std::numeric_limits<float>::max();  // 20260330 ZJH 最小面积记录
    RotatedRect bestRect;  // 20260330 ZJH 最优矩形

    for (int i = 0; i < nHullSize; ++i) {
        int j = (i + 1) % nHullSize;  // 20260330 ZJH 下一个凸包顶点

        // 20260330 ZJH 当前边的方向向量
        float fEdgeX = vecHull[j].fX - vecHull[i].fX;
        float fEdgeY = vecHull[j].fY - vecHull[i].fY;

        // 20260330 ZJH 归一化方向向量
        float fEdgeLen = std::sqrt(fEdgeX * fEdgeX + fEdgeY * fEdgeY);
        if (fEdgeLen < 1e-10f) continue;  // 20260330 ZJH 重合点跳过
        float fNx = fEdgeX / fEdgeLen;  // 20260330 ZJH 单位方向 x
        float fNy = fEdgeY / fEdgeLen;  // 20260330 ZJH 单位方向 y

        // 20260330 ZJH 将所有凸包点投影到该方向和垂直方向
        float fMinProj = std::numeric_limits<float>::max();   // 20260330 ZJH 沿边方向最小投影
        float fMaxProj = -std::numeric_limits<float>::max();  // 20260330 ZJH 沿边方向最大投影
        float fMinPerp = std::numeric_limits<float>::max();   // 20260330 ZJH 垂直方向最小投影
        float fMaxPerp = -std::numeric_limits<float>::max();  // 20260330 ZJH 垂直方向最大投影

        for (int k = 0; k < nHullSize; ++k) {
            // 20260330 ZJH 相对于参考点的偏移
            float fRelX = vecHull[k].fX - vecHull[i].fX;
            float fRelY = vecHull[k].fY - vecHull[i].fY;

            // 20260330 ZJH 沿边方向的投影（点积）
            float fProj = fRelX * fNx + fRelY * fNy;
            // 20260330 ZJH 垂直方向的投影（叉积的 z 分量）
            float fPerp = fRelX * (-fNy) + fRelY * fNx;

            // 20260330 ZJH 更新极值
            fMinProj = std::min(fMinProj, fProj);
            fMaxProj = std::max(fMaxProj, fProj);
            fMinPerp = std::min(fMinPerp, fPerp);
            fMaxPerp = std::max(fMaxPerp, fPerp);
        }

        // 20260330 ZJH 计算候选矩形面积
        float fCandW = fMaxProj - fMinProj;   // 20260330 ZJH 沿边方向的跨度
        float fCandH = fMaxPerp - fMinPerp;   // 20260330 ZJH 垂直方向的跨度
        float fCandArea = fCandW * fCandH;    // 20260330 ZJH 面积

        // 20260330 ZJH 如果面积更小，更新最优解
        if (fCandArea < fBestArea) {
            fBestArea = fCandArea;

            // 20260330 ZJH 计算矩形中心在原图坐标系下的位置
            float fMidProj = (fMinProj + fMaxProj) * 0.5f;  // 20260330 ZJH 投影方向中点
            float fMidPerp = (fMinPerp + fMaxPerp) * 0.5f;  // 20260330 ZJH 垂直方向中点
            float fCenterX = vecHull[i].fX + fMidProj * fNx + fMidPerp * (-fNy);
            float fCenterY = vecHull[i].fY + fMidProj * fNy + fMidPerp * fNx;

            // 20260330 ZJH 计算旋转角度（边方向相对于 x 轴的角度）
            float fAngle = std::atan2(fNy, fNx) * 180.0f / 3.14159265f;

            bestRect.ptCenter = { fCenterX, fCenterY };
            bestRect.fWidth = fCandW;
            bestRect.fHeight = fCandH;
            bestRect.fAngle = fAngle;
        }
    }

    return bestRect;  // 20260330 ZJH 返回最小面积外接矩形
}

// =========================================================
// 辅助度量函数
// =========================================================

// 20260330 ZJH contourCircularity — 计算轮廓圆度
// 圆度 = 4π × 面积 / 周长²，完美圆的圆度为 1.0
// vecContour: 输入轮廓点集
// 返回: 圆度值 [0, 1]
inline float contourCircularity(const std::vector<Point2f>& vecContour) {
    float fArea = contourArea(vecContour);         // 20260330 ZJH 计算面积
    float fPerim = contourPerimeter(vecContour);   // 20260330 ZJH 计算周长

    // 20260330 ZJH 防除零
    if (fPerim < 1e-8f) {
        return 0.0f;
    }

    // 20260330 ZJH 4πA / P²
    return 4.0f * 3.14159265f * fArea / (fPerim * fPerim);
}

// 20260330 ZJH boundingRect — 计算轴对齐外接矩形（AABB）
// vecContour: 输入轮廓点集
// outX, outY: 输出左上角坐标
// outW, outH: 输出宽高
inline void boundingRect(const std::vector<Point2f>& vecContour,
                         float& outX, float& outY, float& outW, float& outH) {
    if (vecContour.empty()) {
        outX = outY = outW = outH = 0.0f;  // 20260330 ZJH 空轮廓
        return;
    }

    float fMinX = vecContour[0].fX;
    float fMaxX = vecContour[0].fX;
    float fMinY = vecContour[0].fY;
    float fMaxY = vecContour[0].fY;

    // 20260330 ZJH 遍历求极值
    for (const auto& pt : vecContour) {
        fMinX = std::min(fMinX, pt.fX);
        fMaxX = std::max(fMaxX, pt.fX);
        fMinY = std::min(fMinY, pt.fY);
        fMaxY = std::max(fMaxY, pt.fY);
    }

    outX = fMinX;
    outY = fMinY;
    outW = fMaxX - fMinX;
    outH = fMaxY - fMinY;
}

}  // namespace om
