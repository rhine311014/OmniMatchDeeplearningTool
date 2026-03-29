// 20260323 ZJH ConfidenceHistogramChart — 置信度分布直方图控件实现
// QPainter 自绘直方图，支持单色和双色（正确/错误）模式

#include "ui/widgets/ConfidenceHistogramChart.h"
#include "ui/widgets/ThemeColors.h"               // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>     // 20260323 ZJH 绘图引擎
#include <QMouseEvent>  // 20260323 ZJH 鼠标事件
#include <algorithm>    // 20260323 ZJH std::max

// 20260323 ZJH 构造函数
ConfidenceHistogramChart::ConfidenceHistogramChart(QWidget* pParent)
    : QWidget(pParent)
{
    setMouseTracking(true);
    setMinimumSize(300, 200);
}

// 20260323 ZJH 设置单色置信度数据
void ConfidenceHistogramChart::setData(const QVector<double>& vecConfidences, int nBins)
{
    m_nBins = nBins;
    m_bDualColor = false;
    computeBins(vecConfidences, nBins, m_vecBinCounts);

    // 20260323 ZJH 计算最大箱体计数
    m_nMaxCount = 0;
    for (int n : m_vecBinCounts) {
        m_nMaxCount = std::max(m_nMaxCount, n);
    }

    m_vecCorrectCounts.clear();
    m_vecIncorrectCounts.clear();
    update();
}

// 20260323 ZJH 设置双色置信度数据
void ConfidenceHistogramChart::setDataWithCorrectness(
    const QVector<double>& vecCorrect,
    const QVector<double>& vecIncorrect,
    int nBins)
{
    m_nBins = nBins;
    m_bDualColor = true;

    computeBins(vecCorrect, nBins, m_vecCorrectCounts);
    computeBins(vecIncorrect, nBins, m_vecIncorrectCounts);

    // 20260323 ZJH 合并为总计数并求最大值
    m_vecBinCounts.resize(nBins);
    m_nMaxCount = 0;
    for (int i = 0; i < nBins; ++i) {
        int nC = (i < m_vecCorrectCounts.size()) ? m_vecCorrectCounts[i] : 0;
        int nI = (i < m_vecIncorrectCounts.size()) ? m_vecIncorrectCounts[i] : 0;
        m_vecBinCounts[i] = nC + nI;
        m_nMaxCount = std::max(m_nMaxCount, m_vecBinCounts[i]);
    }

    update();
}

// 20260323 ZJH 清空数据
void ConfidenceHistogramChart::clear()
{
    m_vecBinCounts.clear();
    m_vecCorrectCounts.clear();
    m_vecIncorrectCounts.clear();
    m_nMaxCount = 0;
    m_nHoverBin = -1;
    update();
}

// 20260323 ZJH 设置柱体颜色
void ConfidenceHistogramChart::setBarColor(const QColor& color)
{
    m_colorBar = color;
    update();
}

// 20260323 ZJH 推荐最小尺寸
QSize ConfidenceHistogramChart::minimumSizeHint() const
{
    return QSize(300, 200);
}

// 20260324 ZJH 返回控件的推荐尺寸（400x300），供布局管理器参考
QSize ConfidenceHistogramChart::sizeHint() const
{
    return QSize(400, 300);  // 20260324 ZJH 图表类控件推荐尺寸
}

// 20260323 ZJH 绘制事件
void ConfidenceHistogramChart::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 20260323 ZJH 背景
    painter.fillRect(rect(), QColor("#1a1d24"));

    // 20260323 ZJH 标题
    painter.setPen(QColor("#e2e8f0"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 11, QFont::Bold));
    painter.drawText(QRectF(0, 4, width(), m_nMarginTop - 4), Qt::AlignCenter, "Confidence Distribution");

    // 20260323 ZJH 空数据提示
    if (m_vecBinCounts.isEmpty() || m_nMaxCount == 0) {
        painter.setPen(QColor("#64748b"));
        painter.setFont(QFont(ThemeColors::s_strFontFamily, 10));
        painter.drawText(rect(), Qt::AlignCenter, "No data");
        return;
    }

    // 20260323 ZJH 绘图区域
    QRectF plotRect(m_nMarginLeft, m_nMarginTop,
                    width() - m_nMarginLeft - m_nMarginRight,
                    height() - m_nMarginTop - m_nMarginBottom);

    // 20260323 ZJH 绘制 Y 轴网格线
    painter.setPen(QPen(QColor("#334155"), 1, Qt::DotLine));
    int nYTicks = 4;  // 20260323 ZJH Y 轴刻度数
    for (int i = 1; i <= nYTicks; ++i) {
        double dY = plotRect.bottom() - (static_cast<double>(i) / nYTicks) * plotRect.height();
        painter.drawLine(QPointF(plotRect.left(), dY), QPointF(plotRect.right(), dY));
    }

    // 20260323 ZJH Y 轴标签
    painter.setPen(QColor("#94a3b8"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 8));
    for (int i = 0; i <= nYTicks; ++i) {
        int nVal = static_cast<int>(static_cast<double>(i) / nYTicks * m_nMaxCount);
        double dY = plotRect.bottom() - (static_cast<double>(i) / nYTicks) * plotRect.height();
        painter.drawText(QRectF(0, dY - 8, m_nMarginLeft - 5, 16),
                        Qt::AlignRight | Qt::AlignVCenter, QString::number(nVal));
    }

    // 20260323 ZJH 绘制柱体
    double dBarWidth = plotRect.width() / m_nBins;  // 20260323 ZJH 每个箱体宽度
    double dGap = dBarWidth * 0.1;                   // 20260323 ZJH 间隙

    for (int i = 0; i < m_nBins; ++i) {
        double dX = plotRect.left() + i * dBarWidth + dGap;
        double dW = dBarWidth - 2 * dGap;

        if (m_bDualColor) {
            // 20260323 ZJH 双色模式：正确（下方）+ 错误（上方）堆叠
            int nC = (i < m_vecCorrectCounts.size()) ? m_vecCorrectCounts[i] : 0;
            int nI = (i < m_vecIncorrectCounts.size()) ? m_vecIncorrectCounts[i] : 0;

            // 20260323 ZJH 正确预测柱（绿色）
            if (nC > 0) {
                double dH = (static_cast<double>(nC) / m_nMaxCount) * plotRect.height();
                QRectF barRect(dX, plotRect.bottom() - dH, dW, dH);
                QColor c = (i == m_nHoverBin) ? m_colorCorrect.lighter(130) : m_colorCorrect;
                painter.fillRect(barRect, c);
            }

            // 20260323 ZJH 错误预测柱（红色，堆叠在绿色上方）
            if (nI > 0) {
                double dHc = (static_cast<double>(nC) / m_nMaxCount) * plotRect.height();
                double dHi = (static_cast<double>(nI) / m_nMaxCount) * plotRect.height();
                QRectF barRect(dX, plotRect.bottom() - dHc - dHi, dW, dHi);
                QColor c = (i == m_nHoverBin) ? m_colorIncorrect.lighter(130) : m_colorIncorrect;
                painter.fillRect(barRect, c);
            }
        } else {
            // 20260323 ZJH 单色模式
            int nCount = m_vecBinCounts[i];
            if (nCount > 0) {
                double dH = (static_cast<double>(nCount) / m_nMaxCount) * plotRect.height();
                QRectF barRect(dX, plotRect.bottom() - dH, dW, dH);
                QColor c = (i == m_nHoverBin) ? m_colorBar.lighter(130) : m_colorBar;
                painter.fillRect(barRect, c);
            }
        }
    }

    // 20260323 ZJH X 轴标签
    painter.setPen(QColor("#94a3b8"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 8));
    for (int i = 0; i <= m_nBins; i += qMax(m_nBins / 4, 1)) {  // 20260323 ZJH qMax 防止步长为 0 导致死循环
        double dVal = static_cast<double>(i) / m_nBins;
        double dX = plotRect.left() + i * dBarWidth;
        painter.drawText(QRectF(dX - 15, plotRect.bottom() + 4, 30, 14),
                        Qt::AlignCenter, QString::number(dVal, 'f', 1));
    }

    // 20260323 ZJH X 轴标题
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 9));
    painter.drawText(QRectF(m_nMarginLeft, height() - 16, plotRect.width(), 14),
                    Qt::AlignCenter, "Confidence");

    // 20260323 ZJH 坐标轴边框
    painter.setPen(QPen(QColor("#64748b"), 1.5));
    painter.drawRect(plotRect);

    // 20260323 ZJH 悬停提示
    if (m_nHoverBin >= 0 && m_nHoverBin < m_nBins) {
        double dLow = static_cast<double>(m_nHoverBin) / m_nBins;
        double dHigh = static_cast<double>(m_nHoverBin + 1) / m_nBins;
        QString strTip = QString("[%1, %2): %3")
                             .arg(dLow, 0, 'f', 2)
                             .arg(dHigh, 0, 'f', 2)
                             .arg(m_vecBinCounts[m_nHoverBin]);

        painter.setPen(QColor("#ffffff"));
        painter.setFont(QFont(ThemeColors::s_strFontFamily, 9, QFont::Bold));
        double dTipX = plotRect.left() + m_nHoverBin * dBarWidth + dBarWidth / 2;
        painter.drawText(QRectF(dTipX - 40, m_nMarginTop - 2, 80, 14),
                        Qt::AlignCenter, strTip);
    }

    // 20260323 ZJH 双色模式图例
    if (m_bDualColor) {
        int nLx = static_cast<int>(plotRect.right()) - 130;
        int nLy = m_nMarginTop + 5;
        painter.fillRect(QRect(nLx - 4, nLy - 2, 134, 36), QColor(26, 29, 36, 200));
        painter.fillRect(QRect(nLx, nLy + 2, 10, 10), m_colorCorrect);
        painter.setPen(QColor("#e2e8f0"));
        painter.setFont(QFont(ThemeColors::s_strFontFamily, 8));
        painter.drawText(nLx + 14, nLy + 11, "Correct");
        painter.fillRect(QRect(nLx, nLy + 18, 10, 10), m_colorIncorrect);
        painter.drawText(nLx + 14, nLy + 27, "Incorrect");
    }
}

// 20260323 ZJH 鼠标移动事件
void ConfidenceHistogramChart::mouseMoveEvent(QMouseEvent* pEvent)
{
    if (m_vecBinCounts.isEmpty()) return;

    QRectF plotRect(m_nMarginLeft, m_nMarginTop,
                    width() - m_nMarginLeft - m_nMarginRight,
                    height() - m_nMarginTop - m_nMarginBottom);

    double dX = pEvent->position().toPoint().x() - plotRect.left();
    double dBarWidth = plotRect.width() / m_nBins;

    int nNewHover = -1;
    if (dX >= 0 && dX < plotRect.width() && plotRect.contains(pEvent->position().toPoint())) {
        nNewHover = static_cast<int>(dX / dBarWidth);
        if (nNewHover >= m_nBins) nNewHover = -1;
    }

    if (nNewHover != m_nHoverBin) {
        m_nHoverBin = nNewHover;
        update();
    }
}

// 20260323 ZJH 鼠标离开事件
void ConfidenceHistogramChart::leaveEvent(QEvent* /*pEvent*/)
{
    m_nHoverBin = -1;
    update();
}

// 20260323 ZJH 将置信度值分箱
void ConfidenceHistogramChart::computeBins(
    const QVector<double>& vecValues, int nBins,
    QVector<int>& vecBinCounts) const
{
    vecBinCounts.fill(0, nBins);  // 20260323 ZJH 初始化全零

    for (double dVal : vecValues) {
        // 20260323 ZJH 将 [0, 1] 映射到 [0, nBins-1]
        int nBin = static_cast<int>(dVal * nBins);
        if (nBin >= nBins) nBin = nBins - 1;  // 20260323 ZJH 边界值归入最后一箱
        if (nBin < 0) nBin = 0;
        vecBinCounts[nBin]++;
    }
}
