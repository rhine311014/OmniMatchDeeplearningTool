// 20260323 ZJH LabelPieChart — 标签分布饼图控件实现
// QPainter 自绘饼图，支持悬停高亮和图例

#include "ui/widgets/LabelPieChart.h"
#include "ui/widgets/ThemeColors.h"    // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>       // 20260323 ZJH 绘图引擎
#include <QMouseEvent>    // 20260323 ZJH 鼠标事件
#include <cmath>          // 20260323 ZJH atan2, sqrt
#include <numbers>        // 20260323 ZJH std::numbers::pi (C++20)

// 20260323 ZJH 构造函数
LabelPieChart::LabelPieChart(QWidget* pParent)
    : QWidget(pParent)
{
    setMouseTracking(true);  // 20260323 ZJH 启用鼠标追踪
    setMinimumSize(200, 200);
}

// 20260323 ZJH 设置饼图数据（从映射）
void LabelPieChart::setData(const QMap<QString, int>& mapData)
{
    m_vecSlices.clear();
    m_nTotal = 0;
    const auto& colors = defaultColors();
    int nIdx = 0;

    // 20260323 ZJH 遍历映射，生成扇区数据
    for (auto it = mapData.begin(); it != mapData.end(); ++it) {
        PieSlice slice;
        slice.strName = it.key();
        slice.nCount = it.value();
        slice.color = colors[nIdx % colors.size()];
        m_vecSlices.append(slice);
        m_nTotal += it.value();
        ++nIdx;
    }

    update();  // 20260323 ZJH 触发重绘
}

// 20260323 ZJH 设置饼图数据（带自定义颜色）
void LabelPieChart::setSlices(const QVector<PieSlice>& vecSlices)
{
    m_vecSlices = vecSlices;
    m_nTotal = 0;
    for (const auto& s : m_vecSlices) {
        m_nTotal += s.nCount;
    }
    update();
}

// 20260323 ZJH 清空数据
void LabelPieChart::clear()
{
    m_vecSlices.clear();
    m_nTotal = 0;
    m_nHoverIndex = -1;
    update();
}

// 20260323 ZJH 推荐最小尺寸
QSize LabelPieChart::minimumSizeHint() const
{
    return QSize(200, 200);
}

// 20260324 ZJH 返回控件的推荐尺寸（400x300），供布局管理器参考
QSize LabelPieChart::sizeHint() const
{
    return QSize(400, 300);  // 20260324 ZJH 图表类控件推荐尺寸
}

// 20260323 ZJH 绘制事件
void LabelPieChart::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 20260323 ZJH 背景
    painter.fillRect(rect(), QColor("#1a1d24"));

    // 20260323 ZJH 空数据时显示占位文字
    if (m_vecSlices.isEmpty() || m_nTotal == 0) {
        painter.setPen(QColor("#64748b"));
        painter.setFont(QFont(ThemeColors::s_strFontFamily, 11));  // 20260324 ZJH 使用共享字体族名
        painter.drawText(rect(), Qt::AlignCenter, "No data");
        return;
    }

    // 20260323 ZJH 计算饼图区域（留出右侧图例空间）
    int nLegendWidth = 120;  // 20260323 ZJH 图例宽度
    int nAvailable = qMin(width() - nLegendWidth - 20, height() - 20);
    int nDiameter = qMax(nAvailable, 80);  // 20260323 ZJH 饼图直径
    int nCx = (width() - nLegendWidth) / 2;  // 20260323 ZJH 圆心 X
    int nCy = height() / 2;                    // 20260323 ZJH 圆心 Y

    m_pieRect = QRectF(nCx - nDiameter / 2.0, nCy - nDiameter / 2.0, nDiameter, nDiameter);

    // 20260323 ZJH 绘制扇区
    int nStartAngle = 90 * 16;  // 20260323 ZJH 从12点方向开始（Qt 角度单位: 1/16 度）

    for (int i = 0; i < m_vecSlices.size(); ++i) {
        double dFrac = static_cast<double>(m_vecSlices[i].nCount) / m_nTotal;  // 20260323 ZJH 占比
        int nSpanAngle = static_cast<int>(dFrac * 360 * 16);  // 20260323 ZJH 跨度角

        QRectF pieR = m_pieRect;
        // 20260323 ZJH 悬停时扇区向外偏移 8px
        if (i == m_nHoverIndex) {
            double dMidAngle = (nStartAngle + nSpanAngle / 2.0) / 16.0 * std::numbers::pi / 180.0;
            pieR.translate(8 * cos(dMidAngle), -8 * sin(dMidAngle));
        }

        // 20260323 ZJH 填充扇区
        QColor fillColor = m_vecSlices[i].color;
        if (i == m_nHoverIndex) fillColor = fillColor.lighter(130);  // 20260323 ZJH 悬停变亮
        painter.setPen(QPen(QColor("#1a1d24"), 2));  // 20260323 ZJH 扇区间隔线
        painter.setBrush(fillColor);
        painter.drawPie(pieR, nStartAngle, nSpanAngle);

        nStartAngle += nSpanAngle;  // 20260323 ZJH 累加起始角
    }

    // 20260323 ZJH 绘制图例
    int nLegendX = width() - nLegendWidth;
    int nLegendY = 20;
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 9));  // 20260324 ZJH 使用共享字体族名

    for (int i = 0; i < m_vecSlices.size(); ++i) {
        int nY = nLegendY + i * 22;  // 20260323 ZJH 行间距 22px

        // 20260323 ZJH 颜色方块
        painter.fillRect(QRect(nLegendX, nY + 2, 12, 12), m_vecSlices[i].color);

        // 20260323 ZJH 标签名 + 数量
        painter.setPen(QColor("#e2e8f0"));
        double dPct = static_cast<double>(m_vecSlices[i].nCount) / m_nTotal * 100;
        QString strText = m_vecSlices[i].strName + " (" + QString::number(dPct, 'f', 1) + "%)";
        painter.drawText(nLegendX + 16, nY + 13, strText);
    }
}

// 20260323 ZJH 鼠标移动事件
void LabelPieChart::mouseMoveEvent(QMouseEvent* pEvent)
{
    int nNewHover = sliceAtPos(pEvent->position().toPoint());  // 20260323 ZJH 计算悬停扇区
    if (nNewHover != m_nHoverIndex) {
        m_nHoverIndex = nNewHover;
        update();  // 20260323 ZJH 仅在变化时重绘
    }
}

// 20260323 ZJH 鼠标离开事件
void LabelPieChart::leaveEvent(QEvent* /*pEvent*/)
{
    m_nHoverIndex = -1;  // 20260323 ZJH 清除悬停
    update();
}

// 20260323 ZJH 计算鼠标位置对应的扇区索引
int LabelPieChart::sliceAtPos(const QPoint& pt) const
{
    if (m_vecSlices.isEmpty() || m_nTotal == 0) return -1;

    // 20260323 ZJH 计算相对于饼图中心的偏移
    double dCx = m_pieRect.center().x();
    double dCy = m_pieRect.center().y();
    double dDx = pt.x() - dCx;
    double dDy = pt.y() - dCy;
    double dR = m_pieRect.width() / 2.0;

    // 20260323 ZJH 判断是否在饼图圆内
    if (dDx * dDx + dDy * dDy > dR * dR) return -1;

    // 20260323 ZJH 计算角度（从12点方向顺时针，映射到 Qt 的角度系统）
    double dAngle = atan2(-dDy, dDx) * 180.0 / std::numbers::pi;  // 20260323 ZJH 转为度
    if (dAngle < 0) dAngle += 360.0;

    // 20260323 ZJH 从 90 度开始（12 点方向）
    double dCumAngle = 90.0;
    for (int i = 0; i < m_vecSlices.size(); ++i) {
        double dSpan = static_cast<double>(m_vecSlices[i].nCount) / m_nTotal * 360.0;
        double dEndAngle = dCumAngle + dSpan;

        // 20260323 ZJH 角度归一化后判断是否在该扇区内
        double dNormAngle = fmod(dAngle, 360.0);
        double dNormStart = fmod(dCumAngle, 360.0);
        double dNormEnd = fmod(dEndAngle, 360.0);

        bool bInSlice = false;
        if (dNormStart <= dNormEnd) {
            bInSlice = (dNormAngle >= dNormStart && dNormAngle < dNormEnd);
        } else {
            bInSlice = (dNormAngle >= dNormStart || dNormAngle < dNormEnd);
        }

        if (bInSlice) return i;
        dCumAngle = dEndAngle;
    }

    return -1;
}

// 20260323 ZJH 默认颜色循环
const QVector<QColor>& LabelPieChart::defaultColors()
{
    static const QVector<QColor> s_colors = {
        QColor("#3b82f6"), QColor("#f59e0b"), QColor("#10b981"), QColor("#ef4444"),
        QColor("#8b5cf6"), QColor("#06b6d4"), QColor("#ec4899"), QColor("#84cc16"),
        QColor("#f97316"), QColor("#14b8a6"), QColor("#a855f7"), QColor("#64748b")
    };
    return s_colors;
}
