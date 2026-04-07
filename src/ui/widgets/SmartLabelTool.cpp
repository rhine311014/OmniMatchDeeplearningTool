// 20260330 ZJH SmartLabelTool 实现
// 智能标注工具: FloodFill + Canny 边缘 + 简化 SLIC 超像素
// 纯 C++ 实现，不依赖 OpenCV/Halcon

#include "ui/widgets/SmartLabelTool.h"

#include <QtMath>    // 20260330 ZJH qSqrt, qAtan2 等数学函数
#include <QQueue>    // 20260330 ZJH BFS 队列（FloodFill 使用）
#include <QSet>      // 20260330 ZJH 去重集合
#include <algorithm> // 20260330 ZJH std::min, std::max, std::clamp


// 20260330 ZJH 构造函数
SmartLabelTool::SmartLabelTool(QObject* pParent)
    : QObject(pParent)
{
}

// 20260330 ZJH 设置当前图像
void SmartLabelTool::setImage(const QImage& image)
{
    m_image = image;           // 20260330 ZJH 保存原始图像
    m_bGrayValid = false;      // 20260330 ZJH 灰度缓存失效，下次使用时重新生成
}

// 20260330 ZJH 检查图像是否已设置
bool SmartLabelTool::hasImage() const
{
    return !m_image.isNull();  // 20260330 ZJH 非空即有效
}

// 20260330 ZJH 确保灰度图缓存有效
void SmartLabelTool::ensureGrayscale()
{
    if (m_bGrayValid) return;  // 20260330 ZJH 缓存有效则跳过

    if (m_image.isNull()) {
        m_grayImage = QImage();  // 20260330 ZJH 空图像无法转换
        return;
    }

    // 20260330 ZJH 使用 Qt 内置格式转换为 8 位灰度
    m_grayImage = m_image.convertToFormat(QImage::Format_Grayscale8);
    m_bGrayValid = true;  // 20260330 ZJH 标记缓存有效
}

// 20260330 ZJH 设置 FloodFill 灰度容差
void SmartLabelTool::setFloodFillTolerance(int nTolerance)
{
    m_nFloodTolerance = std::clamp(nTolerance, 1, 255);  // 20260330 ZJH 限制到合法范围
}

// 20260330 ZJH 设置 Hover 搜索窗口半径
void SmartLabelTool::setHoverSearchRadius(int nRadius)
{
    m_nHoverRadius = std::clamp(nRadius, 10, 500);  // 20260330 ZJH 限制到合法范围
}

// ===== Hover 模式: 鼠标悬停自动推荐候选区域 =====

QRectF SmartLabelTool::suggestRegionAt(const QPointF& ptPos)
{
    ensureGrayscale();  // 20260330 ZJH 确保灰度缓存
    if (m_grayImage.isNull()) return QRectF();  // 20260330 ZJH 无图像返回空

    int nW = m_grayImage.width();   // 20260330 ZJH 图像宽度
    int nH = m_grayImage.height();  // 20260330 ZJH 图像高度
    int nSeedX = static_cast<int>(ptPos.x());  // 20260330 ZJH 种子点 X
    int nSeedY = static_cast<int>(ptPos.y());  // 20260330 ZJH 种子点 Y

    // 20260330 ZJH 检查种子点是否在图像范围内
    if (nSeedX < 0 || nSeedX >= nW || nSeedY < 0 || nSeedY >= nH) {
        return QRectF();  // 20260330 ZJH 越界返回空
    }

    const uchar* pGray = m_grayImage.constBits();  // 20260330 ZJH 灰度数据指针

    // 20260330 ZJH 在 Hover 搜索窗口内执行 FloodFill
    QVector<bool> vecVisited(nW * nH, false);
    floodFill(pGray, nW, nH, QPoint(nSeedX, nSeedY), m_nFloodTolerance, vecVisited);

    // 20260330 ZJH 计算填充区域的外接矩形
    int nMinX = nW, nMinY = nH, nMaxX = 0, nMaxY = 0;  // 20260330 ZJH 外接矩形边界
    int nPixelCount = 0;  // 20260330 ZJH 填充像素计数

    // 20260330 ZJH 限定搜索范围在 Hover 窗口内
    int nStartX = std::max(0, nSeedX - m_nHoverRadius);
    int nStartY = std::max(0, nSeedY - m_nHoverRadius);
    int nEndX = std::min(nW - 1, nSeedX + m_nHoverRadius);
    int nEndY = std::min(nH - 1, nSeedY + m_nHoverRadius);

    for (int y = nStartY; y <= nEndY; ++y) {
        for (int x = nStartX; x <= nEndX; ++x) {
            if (vecVisited[y * nW + x]) {
                nMinX = std::min(nMinX, x);  // 20260330 ZJH 更新左边界
                nMinY = std::min(nMinY, y);  // 20260330 ZJH 更新上边界
                nMaxX = std::max(nMaxX, x);  // 20260330 ZJH 更新右边界
                nMaxY = std::max(nMaxY, y);  // 20260330 ZJH 更新下边界
                ++nPixelCount;
            }
        }
    }

    // 20260330 ZJH 至少需要 4 个像素才认为是有效区域
    if (nPixelCount < 4) return QRectF();

    QRectF rectResult(nMinX, nMinY, nMaxX - nMinX + 1, nMaxY - nMinY + 1);
    emit regionSuggested(rectResult);  // 20260330 ZJH 发射信号通知 UI
    return rectResult;
}

// ===== Click 模式: 前景/背景点生成 mask =====

QImage SmartLabelTool::generateMaskFromPoints(const QVector<QPointF>& vecFgPoints,
                                               const QVector<QPointF>& vecBgPoints)
{
    ensureGrayscale();  // 20260330 ZJH 确保灰度缓存
    if (m_grayImage.isNull()) return QImage();  // 20260330 ZJH 无图像返回空

    int nW = m_grayImage.width();   // 20260330 ZJH 图像宽度
    int nH = m_grayImage.height();  // 20260330 ZJH 图像高度
    const uchar* pGray = m_grayImage.constBits();  // 20260330 ZJH 灰度数据

    // 20260330 ZJH 创建前景 mask（白色=前景）
    QVector<bool> vecFgMask(nW * nH, false);

    // 20260330 ZJH 对每个前景点执行 FloodFill，合并结果
    for (const QPointF& ptFg : vecFgPoints) {
        int nX = static_cast<int>(ptFg.x());  // 20260330 ZJH 前景点 X
        int nY = static_cast<int>(ptFg.y());  // 20260330 ZJH 前景点 Y
        if (nX < 0 || nX >= nW || nY < 0 || nY >= nH) continue;  // 20260330 ZJH 越界跳过

        QVector<bool> vecVisited(nW * nH, false);
        floodFill(pGray, nW, nH, QPoint(nX, nY), m_nFloodTolerance, vecVisited);

        // 20260330 ZJH 合并到前景 mask（OR 操作）
        for (int i = 0; i < nW * nH; ++i) {
            if (vecVisited[i]) vecFgMask[i] = true;
        }
    }

    // 20260330 ZJH 对每个背景点执行 FloodFill，从前景中剔除
    for (const QPointF& ptBg : vecBgPoints) {
        int nX = static_cast<int>(ptBg.x());  // 20260330 ZJH 背景点 X
        int nY = static_cast<int>(ptBg.y());  // 20260330 ZJH 背景点 Y
        if (nX < 0 || nX >= nW || nY < 0 || nY >= nH) continue;  // 20260330 ZJH 越界跳过

        QVector<bool> vecVisited(nW * nH, false);
        floodFill(pGray, nW, nH, QPoint(nX, nY), m_nFloodTolerance, vecVisited);

        // 20260330 ZJH 从前景 mask 中剔除背景区域
        for (int i = 0; i < nW * nH; ++i) {
            if (vecVisited[i]) vecFgMask[i] = false;
        }
    }

    // 20260330 ZJH 生成输出 mask 图像（白色=目标, 黑色=背景）
    QImage maskImage(nW, nH, QImage::Format_Grayscale8);
    uchar* pMask = maskImage.bits();  // 20260330 ZJH mask 数据指针
    int nStride = maskImage.bytesPerLine();  // 20260330 ZJH 行字节数（可能有对齐填充）

    for (int y = 0; y < nH; ++y) {
        uchar* pRow = pMask + y * nStride;  // 20260330 ZJH 当前行指针
        for (int x = 0; x < nW; ++x) {
            pRow[x] = vecFgMask[y * nW + x] ? 255 : 0;  // 20260330 ZJH 前景白色，背景黑色
        }
    }

    emit maskGenerated(maskImage);  // 20260330 ZJH 发射信号通知 UI
    return maskImage;
}

// ===== 边缘检测: Canny → 候选区域 =====

QVector<QRectF> SmartLabelTool::detectCandidateRegions(float fThreshold)
{
    ensureGrayscale();  // 20260330 ZJH 确保灰度缓存
    if (m_grayImage.isNull()) return {};  // 20260330 ZJH 无图像返回空

    int nW = m_grayImage.width();   // 20260330 ZJH 图像宽度
    int nH = m_grayImage.height();  // 20260330 ZJH 图像高度
    const uchar* pGray = m_grayImage.constBits();  // 20260330 ZJH 灰度数据

    // 20260330 ZJH Step 1: Sobel 3x3 梯度计算
    QVector<float> vecGx(nW * nH, 0.0f);  // 20260330 ZJH x 方向梯度
    QVector<float> vecGy(nW * nH, 0.0f);  // 20260330 ZJH y 方向梯度
    sobelGradient(pGray, nW, nH, vecGx, vecGy);

    // 20260330 ZJH Step 2: 计算梯度幅值和方向
    QVector<float> vecMag(nW * nH, 0.0f);    // 20260330 ZJH 梯度幅值
    QVector<float> vecAngle(nW * nH, 0.0f);  // 20260330 ZJH 梯度方向（弧度）

    for (int i = 0; i < nW * nH; ++i) {
        vecMag[i] = qSqrt(vecGx[i] * vecGx[i] + vecGy[i] * vecGy[i]);  // 20260330 ZJH L2 范数
        vecAngle[i] = qAtan2(vecGy[i], vecGx[i]);  // 20260330 ZJH 梯度方向
    }

    // 20260330 ZJH Step 3: 非极大值抑制（边缘细化）
    QVector<float> vecNms(nW * nH, 0.0f);
    nonMaxSuppression(vecMag, vecAngle, nW, nH, vecNms);

    // 20260330 ZJH Step 4: 双阈值 + 边缘连接
    float fLow = fThreshold;       // 20260330 ZJH 低阈值
    float fHigh = fThreshold * 2;  // 20260330 ZJH 高阈值（经典比例 1:2）
    QVector<bool> vecEdge(nW * nH, false);
    doubleThreshold(vecNms, nW, nH, fLow, fHigh, vecEdge);

    // 20260330 ZJH Step 5: 从边缘图提取连通域外接矩形
    return extractBoundingRects(vecEdge, nW, nH);
}

// ===== 超像素: 简化 SLIC =====

QVector<QPolygonF> SmartLabelTool::generateSuperpixels(int nNumRegions)
{
    ensureGrayscale();  // 20260330 ZJH 确保灰度缓存
    if (m_grayImage.isNull()) return {};  // 20260330 ZJH 无图像返回空

    int nW = m_grayImage.width();   // 20260330 ZJH 图像宽度
    int nH = m_grayImage.height();  // 20260330 ZJH 图像高度
    const uchar* pGray = m_grayImage.constBits();  // 20260330 ZJH 灰度数据

    // 20260330 ZJH 计算超像素网格间距 S = sqrt(N / K)
    int nTotalPixels = nW * nH;  // 20260330 ZJH 总像素数
    nNumRegions = std::clamp(nNumRegions, 4, nTotalPixels / 4);  // 20260330 ZJH 限制范围
    int nStep = static_cast<int>(qSqrt(static_cast<double>(nTotalPixels) / nNumRegions));
    if (nStep < 2) nStep = 2;  // 20260330 ZJH 最小步长 2

    // 20260330 ZJH 初始化聚类中心（均匀网格采样）
    struct ClusterCenter {
        float fX, fY;  // 20260330 ZJH 中心坐标
        float fGray;   // 20260330 ZJH 灰度值
    };
    QVector<ClusterCenter> vecCenters;  // 20260330 ZJH 聚类中心列表

    for (int y = nStep / 2; y < nH; y += nStep) {
        for (int x = nStep / 2; x < nW; x += nStep) {
            ClusterCenter center;
            center.fX = static_cast<float>(x);      // 20260330 ZJH 中心 X
            center.fY = static_cast<float>(y);      // 20260330 ZJH 中心 Y
            center.fGray = static_cast<float>(pGray[y * nW + x]);  // 20260330 ZJH 中心灰度
            vecCenters.append(center);
        }
    }

    int nK = vecCenters.size();  // 20260330 ZJH 实际聚类数
    if (nK == 0) return {};      // 20260330 ZJH 防御空结果

    // 20260330 ZJH 像素→聚类标签映射
    QVector<int> vecLabels(nTotalPixels, -1);  // 20260330 ZJH -1 表示未分配
    QVector<float> vecDist(nTotalPixels, 1e30f);  // 20260330 ZJH 最小距离

    float fM = 10.0f;  // 20260330 ZJH 灰度权重参数（控制紧凑度）

    // 20260330 ZJH 迭代 5 轮（SLIC 通常 5-10 轮收敛）
    for (int nIter = 0; nIter < 5; ++nIter) {
        // 20260330 ZJH 重置距离
        vecDist.fill(1e30f);

        // 20260330 ZJH 对每个聚类中心，在 2S x 2S 窗口内分配像素
        for (int k = 0; k < nK; ++k) {
            int nCx = static_cast<int>(vecCenters[k].fX);  // 20260330 ZJH 中心 X
            int nCy = static_cast<int>(vecCenters[k].fY);  // 20260330 ZJH 中心 Y
            float fCg = vecCenters[k].fGray;                // 20260330 ZJH 中心灰度

            // 20260330 ZJH 搜索窗口边界
            int nX0 = std::max(0, nCx - nStep);
            int nY0 = std::max(0, nCy - nStep);
            int nX1 = std::min(nW - 1, nCx + nStep);
            int nY1 = std::min(nH - 1, nCy + nStep);

            for (int y = nY0; y <= nY1; ++y) {
                for (int x = nX0; x <= nX1; ++x) {
                    float fGray = static_cast<float>(pGray[y * nW + x]);  // 20260330 ZJH 像素灰度
                    // 20260330 ZJH SLIC 距离 = sqrt(dc^2/m^2 + ds^2/S^2)
                    float fDc = (fGray - fCg) / fM;  // 20260330 ZJH 归一化灰度距离
                    float fDx = static_cast<float>(x - nCx) / nStep;  // 20260330 ZJH 归一化空间距离 X
                    float fDy = static_cast<float>(y - nCy) / nStep;  // 20260330 ZJH 归一化空间距离 Y
                    float fDist = fDc * fDc + fDx * fDx + fDy * fDy;  // 20260330 ZJH 平方距离

                    int nIdx = y * nW + x;  // 20260330 ZJH 像素线性索引
                    if (fDist < vecDist[nIdx]) {
                        vecDist[nIdx] = fDist;     // 20260330 ZJH 更新最小距离
                        vecLabels[nIdx] = k;       // 20260330 ZJH 分配到聚类 k
                    }
                }
            }
        }

        // 20260330 ZJH 重新计算聚类中心
        QVector<float> vecSumX(nK, 0.0f);    // 20260330 ZJH X 坐标累加
        QVector<float> vecSumY(nK, 0.0f);    // 20260330 ZJH Y 坐标累加
        QVector<float> vecSumG(nK, 0.0f);    // 20260330 ZJH 灰度累加
        QVector<int>   vecCount(nK, 0);       // 20260330 ZJH 像素计数

        for (int y = 0; y < nH; ++y) {
            for (int x = 0; x < nW; ++x) {
                int nLabel = vecLabels[y * nW + x];
                if (nLabel >= 0 && nLabel < nK) {
                    vecSumX[nLabel] += x;     // 20260330 ZJH 累加 X
                    vecSumY[nLabel] += y;     // 20260330 ZJH 累加 Y
                    vecSumG[nLabel] += pGray[y * nW + x];  // 20260330 ZJH 累加灰度
                    vecCount[nLabel]++;        // 20260330 ZJH 计数
                }
            }
        }

        // 20260330 ZJH 更新聚类中心为均值
        for (int k = 0; k < nK; ++k) {
            if (vecCount[k] > 0) {
                vecCenters[k].fX = vecSumX[k] / vecCount[k];
                vecCenters[k].fY = vecSumY[k] / vecCount[k];
                vecCenters[k].fGray = vecSumG[k] / vecCount[k];
            }
        }
    }

    // 20260330 ZJH 从标签图提取每个超像素的边界多边形（使用外接矩形的四角简化）
    QVector<QPolygonF> vecResult;
    for (int k = 0; k < nK; ++k) {
        int nMinX = nW, nMinY = nH, nMaxX = 0, nMaxY = 0;  // 20260330 ZJH 外接矩形
        int nCount = 0;  // 20260330 ZJH 像素计数

        for (int y = 0; y < nH; ++y) {
            for (int x = 0; x < nW; ++x) {
                if (vecLabels[y * nW + x] == k) {
                    nMinX = std::min(nMinX, x);
                    nMinY = std::min(nMinY, y);
                    nMaxX = std::max(nMaxX, x);
                    nMaxY = std::max(nMaxY, y);
                    ++nCount;
                }
            }
        }

        // 20260330 ZJH 至少 4 个像素才输出
        if (nCount < 4) continue;

        // 20260330 ZJH 简化: 用外接矩形四角作为多边形
        QPolygonF polygon;
        polygon << QPointF(nMinX, nMinY)
                << QPointF(nMaxX, nMinY)
                << QPointF(nMaxX, nMaxY)
                << QPointF(nMinX, nMaxY);
        vecResult.append(polygon);
    }

    return vecResult;
}

// ===== 内部工具函数 =====

// 20260330 ZJH FloodFill BFS 实现
void SmartLabelTool::floodFill(const uchar* matGray, int nWidth, int nHeight,
                                const QPoint& ptSeed, int nTolerance,
                                QVector<bool>& pVisited)
{
    // 20260330 ZJH 获取种子点灰度值
    int nSeedGray = matGray[ptSeed.y() * nWidth + ptSeed.x()];

    // 20260330 ZJH BFS 队列
    QQueue<QPoint> queue;
    queue.enqueue(ptSeed);
    pVisited[ptSeed.y() * nWidth + ptSeed.x()] = true;  // 20260330 ZJH 标记种子已访问

    // 20260330 ZJH 4 邻域偏移（上下左右）
    static const int s_arrDx[] = {0, 0, -1, 1};
    static const int s_arrDy[] = {-1, 1, 0, 0};

    while (!queue.isEmpty()) {
        QPoint ptCur = queue.dequeue();  // 20260330 ZJH 取出当前点

        // 20260330 ZJH 遍历 4 邻域
        for (int d = 0; d < 4; ++d) {
            int nNx = ptCur.x() + s_arrDx[d];  // 20260330 ZJH 邻居 X
            int nNy = ptCur.y() + s_arrDy[d];  // 20260330 ZJH 邻居 Y

            // 20260330 ZJH 边界检查
            if (nNx < 0 || nNx >= nWidth || nNy < 0 || nNy >= nHeight) continue;

            int nIdx = nNy * nWidth + nNx;  // 20260330 ZJH 邻居线性索引
            if (pVisited[nIdx]) continue;    // 20260330 ZJH 已访问跳过

            // 20260330 ZJH 检查灰度相似性（容差范围内才扩展）
            int nNeighborGray = matGray[nIdx];
            if (qAbs(nNeighborGray - nSeedGray) <= nTolerance) {
                pVisited[nIdx] = true;       // 20260330 ZJH 标记已访问
                queue.enqueue(QPoint(nNx, nNy));  // 20260330 ZJH 加入队列
            }
        }
    }
}

// 20260330 ZJH Sobel 3x3 梯度计算
void SmartLabelTool::sobelGradient(const uchar* matGray, int nWidth, int nHeight,
                                    QVector<float>& matGx, QVector<float>& matGy)
{
    // 20260330 ZJH Sobel 核:
    // Gx = [-1 0 1; -2 0 2; -1 0 1]
    // Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    for (int y = 1; y < nHeight - 1; ++y) {
        for (int x = 1; x < nWidth - 1; ++x) {
            // 20260330 ZJH 取 3x3 邻域灰度值
            float fP00 = matGray[(y - 1) * nWidth + (x - 1)];  // 20260330 ZJH 左上
            float fP01 = matGray[(y - 1) * nWidth + x];        // 20260330 ZJH 上
            float fP02 = matGray[(y - 1) * nWidth + (x + 1)];  // 20260330 ZJH 右上
            float fP10 = matGray[y * nWidth + (x - 1)];        // 20260330 ZJH 左
            float fP12 = matGray[y * nWidth + (x + 1)];        // 20260330 ZJH 右
            float fP20 = matGray[(y + 1) * nWidth + (x - 1)];  // 20260330 ZJH 左下
            float fP21 = matGray[(y + 1) * nWidth + x];        // 20260330 ZJH 下
            float fP22 = matGray[(y + 1) * nWidth + (x + 1)];  // 20260330 ZJH 右下

            int nIdx = y * nWidth + x;
            matGx[nIdx] = -fP00 + fP02 - 2 * fP10 + 2 * fP12 - fP20 + fP22;  // 20260330 ZJH X 梯度
            matGy[nIdx] = -fP00 - 2 * fP01 - fP02 + fP20 + 2 * fP21 + fP22;  // 20260330 ZJH Y 梯度
        }
    }
}

// 20260330 ZJH 非极大值抑制
void SmartLabelTool::nonMaxSuppression(const QVector<float>& matMag,
                                        const QVector<float>& matAngle,
                                        int nWidth, int nHeight,
                                        QVector<float>& matNms)
{
    for (int y = 1; y < nHeight - 1; ++y) {
        for (int x = 1; x < nWidth - 1; ++x) {
            int nIdx = y * nWidth + x;
            float fAngle = matAngle[nIdx];  // 20260330 ZJH 梯度方向
            float fMag = matMag[nIdx];      // 20260330 ZJH 梯度幅值

            // 20260330 ZJH 将角度归一化到 [0, pi)
            if (fAngle < 0) fAngle += static_cast<float>(M_PI);

            // 20260330 ZJH 根据梯度方向选择比较邻居
            float fN1 = 0.0f, fN2 = 0.0f;  // 20260330 ZJH 梯度方向上的两个邻居幅值
            if (fAngle < M_PI / 8 || fAngle >= 7 * M_PI / 8) {
                // 20260330 ZJH 水平方向（0度）: 比较左右
                fN1 = matMag[nIdx - 1];
                fN2 = matMag[nIdx + 1];
            } else if (fAngle < 3 * M_PI / 8) {
                // 20260330 ZJH 45度方向: 比较左上和右下
                fN1 = matMag[(y - 1) * nWidth + (x + 1)];
                fN2 = matMag[(y + 1) * nWidth + (x - 1)];
            } else if (fAngle < 5 * M_PI / 8) {
                // 20260330 ZJH 垂直方向（90度）: 比较上下
                fN1 = matMag[(y - 1) * nWidth + x];
                fN2 = matMag[(y + 1) * nWidth + x];
            } else {
                // 20260330 ZJH 135度方向: 比较右上和左下
                fN1 = matMag[(y - 1) * nWidth + (x - 1)];
                fN2 = matMag[(y + 1) * nWidth + (x + 1)];
            }

            // 20260330 ZJH 当前点比两个邻居都大时保留，否则抑制
            matNms[nIdx] = (fMag >= fN1 && fMag >= fN2) ? fMag : 0.0f;
        }
    }
}

// 20260330 ZJH 双阈值 + 连通域边缘追踪
void SmartLabelTool::doubleThreshold(const QVector<float>& matNms,
                                      int nWidth, int nHeight,
                                      float fLow, float fHigh,
                                      QVector<bool>& matEdge)
{
    // 20260330 ZJH 第一遍: 标记强边缘（≥高阈值）
    QQueue<QPoint> queue;  // 20260330 ZJH BFS 队列，用于从强边缘连接弱边缘

    for (int y = 0; y < nHeight; ++y) {
        for (int x = 0; x < nWidth; ++x) {
            int nIdx = y * nWidth + x;
            if (matNms[nIdx] >= fHigh) {
                matEdge[nIdx] = true;  // 20260330 ZJH 标记为强边缘
                queue.enqueue(QPoint(x, y));  // 20260330 ZJH 加入 BFS 起点
            }
        }
    }

    // 20260330 ZJH 第二遍: BFS 从强边缘出发，连接相邻的弱边缘
    static const int s_arrDx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int s_arrDy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    while (!queue.isEmpty()) {
        QPoint pt = queue.dequeue();
        for (int d = 0; d < 8; ++d) {
            int nNx = pt.x() + s_arrDx[d];
            int nNy = pt.y() + s_arrDy[d];
            if (nNx < 0 || nNx >= nWidth || nNy < 0 || nNy >= nHeight) continue;

            int nIdx = nNy * nWidth + nNx;
            // 20260330 ZJH 弱边缘（≥低阈值且尚未标记）连接为边缘
            if (!matEdge[nIdx] && matNms[nIdx] >= fLow) {
                matEdge[nIdx] = true;
                queue.enqueue(QPoint(nNx, nNy));
            }
        }
    }
}

// 20260330 ZJH 从二值边缘图提取连通域外接矩形
QVector<QRectF> SmartLabelTool::extractBoundingRects(const QVector<bool>& matEdge,
                                                      int nWidth, int nHeight)
{
    QVector<QRectF> vecResult;
    QVector<bool> vecVisited(nWidth * nHeight, false);  // 20260330 ZJH 访问标记

    // 20260330 ZJH 4 邻域偏移
    static const int s_arrDx[] = {0, 0, -1, 1};
    static const int s_arrDy[] = {-1, 1, 0, 0};

    for (int y = 0; y < nHeight; ++y) {
        for (int x = 0; x < nWidth; ++x) {
            int nIdx = y * nWidth + x;
            if (!matEdge[nIdx] || vecVisited[nIdx]) continue;  // 20260330 ZJH 非边缘或已访问跳过

            // 20260330 ZJH BFS 提取连通域
            QQueue<QPoint> queue;
            queue.enqueue(QPoint(x, y));
            vecVisited[nIdx] = true;

            int nMinX = x, nMinY = y, nMaxX = x, nMaxY = y;  // 20260330 ZJH 外接矩形
            int nCount = 0;

            while (!queue.isEmpty()) {
                QPoint pt = queue.dequeue();
                nMinX = std::min(nMinX, pt.x());
                nMinY = std::min(nMinY, pt.y());
                nMaxX = std::max(nMaxX, pt.x());
                nMaxY = std::max(nMaxY, pt.y());
                ++nCount;

                for (int d = 0; d < 4; ++d) {
                    int nNx = pt.x() + s_arrDx[d];
                    int nNy = pt.y() + s_arrDy[d];
                    if (nNx < 0 || nNx >= nWidth || nNy < 0 || nNy >= nHeight) continue;
                    int nNIdx = nNy * nWidth + nNx;
                    if (!vecVisited[nNIdx] && matEdge[nNIdx]) {
                        vecVisited[nNIdx] = true;
                        queue.enqueue(QPoint(nNx, nNy));
                    }
                }
            }

            // 20260330 ZJH 过滤太小的连通域（噪声），至少 20 像素
            if (nCount >= 20) {
                vecResult.append(QRectF(nMinX, nMinY,
                                        nMaxX - nMinX + 1, nMaxY - nMinY + 1));
            }
        }
    }

    return vecResult;
}
