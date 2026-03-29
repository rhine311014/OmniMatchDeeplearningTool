// 20260323 ZJH GradCAMOverlay — Grad-CAM 热力图叠加控件实现
// 20260324 ZJH 新增二值化缺陷图模式（Binary mode）

#include "ui/widgets/GradCAMOverlay.h"
#include "ui/widgets/ThemeColors.h"    // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>   // 20260323 ZJH 绘图引擎
#include <algorithm>  // 20260323 ZJH std::clamp
#include <cmath>      // 20260323 ZJH 数学函数

// 20260323 ZJH 构造函数
GradCAMOverlay::GradCAMOverlay(QWidget* pParent)
    : QWidget(pParent)
{
    setMinimumSize(200, 200);
    rebuildColorLUT();  // 20260324 ZJH 初始化默认色彩映射的查找表
}

// 20260323 ZJH 设置原始图像
void GradCAMOverlay::setOriginalImage(const QImage& image)
{
    m_imgOriginal = image;
    updateComposite();  // 20260323 ZJH 重新合成
    update();
}

// 20260323 ZJH 设置 Grad-CAM 热力图数据
void GradCAMOverlay::setHeatmap(const QVector<double>& vecHeatmap, int nWidth, int nHeight)
{
    m_vecHeatmap = vecHeatmap;
    m_nHeatmapW = nWidth;
    m_nHeatmapH = nHeight;
    updateComposite();
    update();
}

// 20260323 ZJH 设置叠加透明度
void GradCAMOverlay::setOverlayAlpha(double dAlpha)
{
    m_dAlpha = std::clamp(dAlpha, 0.0, 1.0);
    updateComposite();
    update();
}

// 20260323 ZJH 设置色彩映射
void GradCAMOverlay::setColorMap(ColorMapType type)
{
    m_colorMapType = type;
    rebuildColorLUT();  // 20260324 ZJH 色彩映射类型变化时重建查找表
    updateComposite();
    update();
}

// 20260324 ZJH 设置显示模式（热力图 / 二值化缺陷图）
void GradCAMOverlay::setDisplayMode(GradCAMDisplayMode eMode)
{
    if (m_eDisplayMode == eMode) {
        return;  // 20260324 ZJH 模式未变化，不重复合成
    }
    m_eDisplayMode = eMode;  // 20260324 ZJH 记录新模式
    updateComposite();        // 20260324 ZJH 重新合成图像
    update();                 // 20260324 ZJH 触发重绘
}

// 20260324 ZJH 获取当前显示模式
GradCAMDisplayMode GradCAMOverlay::displayMode() const
{
    return m_eDisplayMode;
}

// 20260324 ZJH 设置二值化阈值
void GradCAMOverlay::setThreshold(double dThreshold)
{
    dThreshold = std::clamp(dThreshold, 0.0, 1.0);  // 20260324 ZJH 限制到 [0, 1]
    if (qFuzzyCompare(m_dThreshold, dThreshold)) {
        return;  // 20260324 ZJH 阈值未变化
    }
    m_dThreshold = dThreshold;  // 20260324 ZJH 更新阈值
    // 20260324 ZJH 仅在二值化模式下才需要重新合成
    if (m_eDisplayMode == GradCAMDisplayMode::Binary) {
        updateComposite();
        update();
    }
}

// 20260324 ZJH 获取当前二值化阈值
double GradCAMOverlay::threshold() const
{
    return m_dThreshold;
}

// 20260323 ZJH 清空数据
void GradCAMOverlay::clear()
{
    m_imgOriginal = QImage();
    m_vecHeatmap.clear();
    m_nHeatmapW = 0;
    m_nHeatmapH = 0;
    m_imgComposite = QImage();
    update();
}

// 20260323 ZJH 获取合成图像
// 20260324 ZJH 返回 const 引用避免不必要的 QImage 深拷贝
const QImage& GradCAMOverlay::compositeImage() const
{
    return m_imgComposite;
}

// 20260323 ZJH 推荐最小尺寸
QSize GradCAMOverlay::minimumSizeHint() const
{
    return QSize(200, 200);
}

// 20260324 ZJH 返回控件的推荐尺寸（300x300），供布局管理器参考
QSize GradCAMOverlay::sizeHint() const
{
    return QSize(300, 300);  // 20260324 ZJH 热力图叠加控件推荐正方形尺寸
}

// 20260323 ZJH 绘制事件
void GradCAMOverlay::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    // 20260323 ZJH 背景
    painter.fillRect(rect(), QColor("#1a1d24"));

    if (m_imgComposite.isNull()) {
        // 20260323 ZJH 无数据时显示占位文字
        painter.setPen(QColor("#64748b"));
        painter.setFont(QFont(ThemeColors::s_strFontFamily, 10));  // 20260324 ZJH 使用共享字体族名
        painter.drawText(rect(), Qt::AlignCenter, "No Grad-CAM data");
        return;
    }

    // 20260323 ZJH 等比缩放绘制合成图像
    QSize imgSize = m_imgComposite.size();
    QSize widgetSize = size();
    imgSize.scale(widgetSize, Qt::KeepAspectRatio);

    int nX = (widgetSize.width() - imgSize.width()) / 2;
    int nY = (widgetSize.height() - imgSize.height()) / 2;

    painter.drawImage(QRect(nX, nY, imgSize.width(), imgSize.height()), m_imgComposite);
}

// 20260323 ZJH 将热力图转换为彩色 QImage
QImage GradCAMOverlay::heatmapToColorImage() const
{
    if (m_vecHeatmap.isEmpty() || m_nHeatmapW <= 0 || m_nHeatmapH <= 0) {
        return QImage();
    }

    QImage imgColor(m_nHeatmapW, m_nHeatmapH, QImage::Format_ARGB32);

    // 20260324 ZJH 使用预计算的颜色查找表代替逐像素调用 colorMap()，大幅减少分支开销
    for (int y = 0; y < m_nHeatmapH; ++y) {
        QRgb* pLine = reinterpret_cast<QRgb*>(imgColor.scanLine(y));  // 20260323 ZJH scanLine 指针直接写入
        for (int x = 0; x < m_nHeatmapW; ++x) {
            int nIdx = y * m_nHeatmapW + x;
            double dVal = (nIdx < m_vecHeatmap.size()) ? m_vecHeatmap[nIdx] : 0.0;
            // 20260324 ZJH 将 [0, 1] 映射到 LUT 索引 [0, 255]
            int nLutIdx = static_cast<int>(std::clamp(dVal, 0.0, 1.0) * 255.0);
            if (nLutIdx > 255) nLutIdx = 255;  // 20260324 ZJH 防止浮点精度导致越界
            pLine[x] = m_arrColorLUT[nLutIdx];
        }
    }

    return imgColor;
}

// 20260324 ZJH 将归一化热力图转换为二值化 QImage
// 缺陷（>=阈值）= 黑色(0,0,0)，背景（<阈值）= 白色(255,255,255)
QImage GradCAMOverlay::heatmapToBinaryImage() const
{
    // 20260324 ZJH 检查热力图数据有效性
    if (m_vecHeatmap.isEmpty() || m_nHeatmapW <= 0 || m_nHeatmapH <= 0) {
        return QImage();  // 20260324 ZJH 无数据返回空图像
    }

    // 20260324 ZJH 创建二值化图像
    QImage imgBinary(m_nHeatmapW, m_nHeatmapH, QImage::Format_RGB32);

    // 20260324 ZJH 预定义黑白颜色值
    const QRgb rgbBlack = qRgb(0, 0, 0);      // 20260324 ZJH 缺陷像素（黑色）
    const QRgb rgbWhite = qRgb(255, 255, 255); // 20260324 ZJH 背景像素（白色）

    // 20260324 ZJH 逐像素二值化：使用 scanLine 指针直接写入
    for (int y = 0; y < m_nHeatmapH; ++y) {
        QRgb* pLine = reinterpret_cast<QRgb*>(imgBinary.scanLine(y));
        for (int x = 0; x < m_nHeatmapW; ++x) {
            int nIdx = y * m_nHeatmapW + x;  // 20260324 ZJH 行优先索引
            double dVal = (nIdx < m_vecHeatmap.size()) ? m_vecHeatmap[nIdx] : 0.0;
            // 20260324 ZJH >=阈值的像素为缺陷（黑色），<阈值为背景（白色）
            pLine[x] = (dVal >= m_dThreshold) ? rgbBlack : rgbWhite;
        }
    }

    return imgBinary;  // 20260324 ZJH 返回纯黑白二值化图像
}

// 20260324 ZJH 预计算 256 级颜色查找表，在色彩映射类型变更时调用
void GradCAMOverlay::rebuildColorLUT()
{
    m_arrColorLUT.resize(256);  // 20260324 ZJH 分配 256 个条目
    for (int i = 0; i < 256; ++i) {
        double dVal = static_cast<double>(i) / 255.0;  // 20260324 ZJH 索引映射到 [0, 1]
        QColor c = colorMap(dVal);                       // 20260324 ZJH 调用现有色彩映射函数
        m_arrColorLUT[i] = qRgb(c.red(), c.green(), c.blue());  // 20260324 ZJH 存储为 QRgb
    }
}

// 20260323 ZJH 色彩映射
QColor GradCAMOverlay::colorMap(double dValue) const
{
    dValue = std::clamp(dValue, 0.0, 1.0);  // 20260323 ZJH 限制到 [0, 1]

    int nR = 0, nG = 0, nB = 0;

    switch (m_colorMapType) {
        case ColorMapType::Jet: {
            // 20260323 ZJH Jet: 蓝→青→绿→黄→红
            if (dValue < 0.25) {
                nR = 0;
                nG = static_cast<int>(255 * (dValue / 0.25));
                nB = 255;
            } else if (dValue < 0.5) {
                nR = 0;
                nG = 255;
                nB = static_cast<int>(255 * (1.0 - (dValue - 0.25) / 0.25));
            } else if (dValue < 0.75) {
                nR = static_cast<int>(255 * ((dValue - 0.5) / 0.25));
                nG = 255;
                nB = 0;
            } else {
                nR = 255;
                nG = static_cast<int>(255 * (1.0 - (dValue - 0.75) / 0.25));
                nB = 0;
            }
            break;
        }
        case ColorMapType::Viridis: {
            // 20260323 ZJH Viridis 简化近似：深紫→蓝→绿→黄
            if (dValue < 0.33) {
                double t = dValue / 0.33;
                nR = static_cast<int>(68 + (33 - 68) * t);
                nG = static_cast<int>(1 + (145 - 1) * t);
                nB = static_cast<int>(84 + (140 - 84) * t);
            } else if (dValue < 0.66) {
                double t = (dValue - 0.33) / 0.33;
                nR = static_cast<int>(33 + (94 - 33) * t);
                nG = static_cast<int>(145 + (201 - 145) * t);
                nB = static_cast<int>(140 + (98 - 140) * t);
            } else {
                double t = (dValue - 0.66) / 0.34;
                nR = static_cast<int>(94 + (253 - 94) * t);
                nG = static_cast<int>(201 + (231 - 201) * t);
                nB = static_cast<int>(98 + (37 - 98) * t);
            }
            break;
        }
        case ColorMapType::Hot: {
            // 20260323 ZJH Hot: 黑→红→黄→白
            if (dValue < 0.33) {
                nR = static_cast<int>(255 * (dValue / 0.33));
                nG = 0;
                nB = 0;
            } else if (dValue < 0.66) {
                nR = 255;
                nG = static_cast<int>(255 * ((dValue - 0.33) / 0.33));
                nB = 0;
            } else {
                nR = 255;
                nG = 255;
                nB = static_cast<int>(255 * ((dValue - 0.66) / 0.34));
            }
            break;
        }
        case ColorMapType::Turbo: {
            // 20260323 ZJH Turbo 简化近似
            if (dValue < 0.25) {
                double t = dValue / 0.25;
                nR = static_cast<int>(48 + (190 - 48) * t);
                nG = static_cast<int>(18 + (75 - 18) * t);
                nB = static_cast<int>(59 + (230 - 59) * t);
            } else if (dValue < 0.5) {
                double t = (dValue - 0.25) / 0.25;
                nR = static_cast<int>(190 * (1 - t) + 20 * t);
                nG = static_cast<int>(75 + (180 - 75) * t);
                nB = static_cast<int>(230 * (1 - t));
            } else if (dValue < 0.75) {
                double t = (dValue - 0.5) / 0.25;
                nR = static_cast<int>(20 + (240 - 20) * t);
                nG = static_cast<int>(180 + (220 - 180) * t);
                nB = static_cast<int>(0 + (20) * t);
            } else {
                double t = (dValue - 0.75) / 0.25;
                nR = static_cast<int>(240 + (15) * t);
                nG = static_cast<int>(220 * (1 - t) + 10 * t);
                nB = static_cast<int>(20 * (1 - t));
            }
            break;
        }
    }

    return QColor(std::clamp(nR, 0, 255), std::clamp(nG, 0, 255), std::clamp(nB, 0, 255));
}

// 20260323 ZJH 合成原始图像和热力图
// 20260324 ZJH 根据显示模式选择热力图叠加或二值化缺陷图
void GradCAMOverlay::updateComposite()
{
    // 20260324 ZJH 二值化模式：不需要原始图像，仅生成纯黑白图
    if (m_eDisplayMode == GradCAMDisplayMode::Binary) {
        if (m_vecHeatmap.isEmpty()) {
            // 20260324 ZJH 无热力图数据时显示原图（如果有的话）
            m_imgComposite = m_imgOriginal;
            return;
        }

        // 20260324 ZJH 生成二值化图像并缩放到原始图像尺寸（如果有原始图像）
        QImage imgBinary = heatmapToBinaryImage();
        if (!m_imgOriginal.isNull()) {
            // 20260324 ZJH 缩放到原始图像大小以保持一致的显示尺寸
            m_imgComposite = imgBinary.scaled(m_imgOriginal.size(),
                                               Qt::IgnoreAspectRatio,
                                               Qt::SmoothTransformation);
        } else {
            // 20260324 ZJH 无原始图像时直接使用原始分辨率
            m_imgComposite = imgBinary;
        }
        return;
    }

    // 20260324 ZJH 以下为原有热力图叠加模式（Heatmap mode）

    if (m_imgOriginal.isNull()) {
        m_imgComposite = QImage();
        return;
    }

    if (m_vecHeatmap.isEmpty()) {
        m_imgComposite = m_imgOriginal;  // 20260323 ZJH 无热力图时显示原图
        return;
    }

    // 20260323 ZJH 将热力图转为彩色图像并缩放到原始图像大小
    QImage imgHeat = heatmapToColorImage();
    QImage imgHeatScaled = imgHeat.scaled(m_imgOriginal.size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    // 20260323 ZJH 创建合成图像
    QImage imgOrig = m_imgOriginal.convertToFormat(QImage::Format_ARGB32);
    m_imgComposite = QImage(imgOrig.size(), QImage::Format_ARGB32);

    // 20260323 ZJH 使用 scanLine 指针直接混合像素，性能比 pixelColor 快 100-500 倍
    int nAlpha256 = static_cast<int>(m_dAlpha * 256);        // 20260323 ZJH 定点化 alpha
    int nInvAlpha256 = 256 - nAlpha256;                       // 20260323 ZJH 定点化 (1-alpha)
    for (int y = 0; y < imgOrig.height(); ++y) {
        const QRgb* pOrig = reinterpret_cast<const QRgb*>(imgOrig.constScanLine(y));
        const QRgb* pHeat = reinterpret_cast<const QRgb*>(imgHeatScaled.constScanLine(y));
        QRgb* pDst = reinterpret_cast<QRgb*>(m_imgComposite.scanLine(y));
        for (int x = 0; x < imgOrig.width(); ++x) {
            int nR = (nInvAlpha256 * qRed(pOrig[x]) + nAlpha256 * qRed(pHeat[x])) >> 8;
            int nG = (nInvAlpha256 * qGreen(pOrig[x]) + nAlpha256 * qGreen(pHeat[x])) >> 8;
            int nB = (nInvAlpha256 * qBlue(pOrig[x]) + nAlpha256 * qBlue(pHeat[x])) >> 8;
            pDst[x] = qRgb(nR, nG, nB);
        }
    }
}
