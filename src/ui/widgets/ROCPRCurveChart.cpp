// 20260323 ZJH ROCPRCurveChart — ROC/PR 曲线图表控件实现
// QPainter 自绘 ROC 曲线和 PR 曲线，支持多类别叠加

#include "ui/widgets/ROCPRCurveChart.h"
#include "ui/widgets/ThemeColors.h"     // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>      // 20260323 ZJH 绘图引擎
#include <QPainterPath>  // 20260323 ZJH 路径绘制
#include <QMouseEvent>   // 20260323 ZJH 鼠标事件
#include <cmath>         // 20260323 ZJH 数学函数

// 20260323 ZJH 构造函数
ROCPRCurveChart::ROCPRCurveChart(QWidget* pParent)
    : QWidget(pParent)
{
    // 20260324 ZJH 移除 setMouseTracking(true)：mouseMoveEvent 当前未使用悬停交互，
    // 启用追踪会导致不必要的事件调用开销
    // 20260323 ZJH 设置最小尺寸
    setMinimumSize(300, 250);
}

// 20260323 ZJH 设置曲线类型
void ROCPRCurveChart::setCurveType(CurveType type)
{
    m_curveType = type;  // 20260323 ZJH 更新曲线类型
    update();            // 20260323 ZJH 触发重绘
}

// 20260323 ZJH 添加一条曲线
void ROCPRCurveChart::addCurve(const CurveData& curve)
{
    CurveData curveWithColor = curve;  // 20260323 ZJH 复制曲线数据

    // 20260323 ZJH 如果未指定颜色，使用默认颜色循环
    if (!curveWithColor.color.isValid()) {
        const auto& colors = defaultColors();
        curveWithColor.color = colors[m_vecCurves.size() % colors.size()];
    }

    m_vecCurves.append(curveWithColor);  // 20260323 ZJH 添加到曲线列表
    update();  // 20260323 ZJH 触发重绘
}

// 20260323 ZJH 清空所有曲线
void ROCPRCurveChart::clear()
{
    m_vecCurves.clear();  // 20260323 ZJH 清空曲线数据
    update();             // 20260323 ZJH 触发重绘
}

// 20260323 ZJH 设置图例显示状态
void ROCPRCurveChart::setShowLegend(bool bShow)
{
    m_bShowLegend = bShow;
    update();
}

// 20260323 ZJH 推荐最小尺寸
QSize ROCPRCurveChart::minimumSizeHint() const
{
    return QSize(300, 250);
}

// 20260324 ZJH 返回控件的推荐尺寸（400x300），供布局管理器参考
QSize ROCPRCurveChart::sizeHint() const
{
    return QSize(400, 300);  // 20260324 ZJH 图表类控件推荐尺寸
}

// 20260323 ZJH 绘制事件
void ROCPRCurveChart::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);                          // 20260323 ZJH 创建画笔
    painter.setRenderHint(QPainter::Antialiasing);   // 20260323 ZJH 抗锯齿

    // 20260323 ZJH 背景填充
    painter.fillRect(rect(), QColor("#1a1d24"));

    // 20260323 ZJH 计算绘图区域
    m_plotRect = QRectF(
        m_nMarginLeft,
        m_nMarginTop,
        width() - m_nMarginLeft - m_nMarginRight,
        height() - m_nMarginTop - m_nMarginBottom
    );

    // 20260323 ZJH 绘制参考线（ROC: 对角线）
    drawReferenceLine(painter);

    // 20260323 ZJH 绘制坐标轴和网格
    drawAxes(painter);

    // 20260323 ZJH 绘制所有曲线
    drawCurves(painter);

    // 20260323 ZJH 绘制图例
    if (m_bShowLegend && !m_vecCurves.isEmpty()) {
        drawLegend(painter);
    }

    // 20260323 ZJH 绘制标题
    painter.setPen(QColor("#e2e8f0"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 11, QFont::Bold));
    QString strTitle = (m_curveType == CurveType::ROC) ? "ROC Curve" : "Precision-Recall Curve";
    painter.drawText(QRectF(0, 4, width(), m_nMarginTop - 4), Qt::AlignCenter, strTitle);
}

// 20260323 ZJH 鼠标移动事件（预留悬停高亮扩展）
// 20260324 ZJH 保留此覆写以备将来实现曲线上最近点悬停高亮和坐标提示功能
void ROCPRCurveChart::mouseMoveEvent(QMouseEvent* /*pEvent*/)
{
    // 20260324 ZJH 预留：将来可在此实现鼠标悬停时查找最近曲线点并显示坐标提示
    // 需要时重新启用构造函数中的 setMouseTracking(true) 来激活
}

// 20260323 ZJH 数据坐标到像素坐标转换
QPointF ROCPRCurveChart::dataToPixel(double dX, double dY) const
{
    double dPxX = m_plotRect.left() + dX * m_plotRect.width();   // 20260323 ZJH X 映射
    double dPxY = m_plotRect.bottom() - dY * m_plotRect.height(); // 20260323 ZJH Y 映射（反转）
    return QPointF(dPxX, dPxY);
}

// 20260323 ZJH 绘制坐标轴和网格线
void ROCPRCurveChart::drawAxes(QPainter& painter) const
{
    // 20260323 ZJH 网格线（5x5）
    painter.setPen(QPen(QColor("#334155"), 1, Qt::DotLine));
    for (int i = 1; i <= 4; ++i) {
        double dVal = i * 0.25;  // 20260323 ZJH 0.25 步进
        // 20260323 ZJH 水平网格线
        QPointF ptLeft = dataToPixel(0.0, dVal);
        QPointF ptRight = dataToPixel(1.0, dVal);
        painter.drawLine(ptLeft, ptRight);
        // 20260323 ZJH 垂直网格线
        QPointF ptTop = dataToPixel(dVal, 0.0);
        QPointF ptBottom = dataToPixel(dVal, 1.0);
        painter.drawLine(ptTop, ptBottom);
    }

    // 20260323 ZJH 坐标轴边框
    painter.setPen(QPen(QColor("#64748b"), 1.5));
    painter.drawRect(m_plotRect);

    // 20260323 ZJH 刻度标签
    painter.setPen(QColor("#94a3b8"));
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 8));

    for (int i = 0; i <= 4; ++i) {
        double dVal = i * 0.25;
        QString strVal = QString::number(dVal, 'f', 2);

        // 20260323 ZJH Y 轴标签
        QPointF ptY = dataToPixel(0.0, dVal);
        painter.drawText(QRectF(0, ptY.y() - 8, m_nMarginLeft - 5, 16),
                        Qt::AlignRight | Qt::AlignVCenter, strVal);

        // 20260323 ZJH X 轴标签
        QPointF ptX = dataToPixel(dVal, 0.0);
        painter.drawText(QRectF(ptX.x() - 20, m_plotRect.bottom() + 4, 40, 16),
                        Qt::AlignCenter, strVal);
    }

    // 20260323 ZJH 轴标题
    painter.setFont(QFont(ThemeColors::s_strFontFamily, 9));
    if (m_curveType == CurveType::ROC) {
        // 20260323 ZJH ROC: X=FPR, Y=TPR
        painter.drawText(QRectF(m_nMarginLeft, height() - 18, m_plotRect.width(), 16),
                        Qt::AlignCenter, "False Positive Rate");
        painter.save();
        painter.translate(12, m_nMarginTop + m_plotRect.height() / 2);
        painter.rotate(-90);
        painter.drawText(QRectF(-60, 0, 120, 16), Qt::AlignCenter, "True Positive Rate");
        painter.restore();
    } else {
        // 20260323 ZJH PR: X=Recall, Y=Precision
        painter.drawText(QRectF(m_nMarginLeft, height() - 18, m_plotRect.width(), 16),
                        Qt::AlignCenter, "Recall");
        painter.save();
        painter.translate(12, m_nMarginTop + m_plotRect.height() / 2);
        painter.rotate(-90);
        painter.drawText(QRectF(-60, 0, 120, 16), Qt::AlignCenter, "Precision");
        painter.restore();
    }
}

// 20260323 ZJH 绘制所有曲线
void ROCPRCurveChart::drawCurves(QPainter& painter) const
{
    for (const auto& curve : m_vecCurves) {
        if (curve.vecPoints.size() < 2) continue;  // 20260323 ZJH 至少需要 2 个点

        // 20260323 ZJH 构建曲线路径
        QPainterPath path;
        QPointF ptFirst = dataToPixel(curve.vecPoints[0].first, curve.vecPoints[0].second);
        path.moveTo(ptFirst);

        for (int i = 1; i < curve.vecPoints.size(); ++i) {
            QPointF pt = dataToPixel(curve.vecPoints[i].first, curve.vecPoints[i].second);
            path.lineTo(pt);
        }

        // 20260323 ZJH 绘制曲线填充（半透明）
        QPainterPath fillPath = path;
        QPointF ptLast = dataToPixel(curve.vecPoints.last().first, 0.0);
        QPointF ptOrigin = dataToPixel(curve.vecPoints.first().first, 0.0);
        fillPath.lineTo(ptLast);
        fillPath.lineTo(ptOrigin);
        fillPath.closeSubpath();

        QColor fillColor = curve.color;
        fillColor.setAlpha(30);  // 20260323 ZJH 半透明填充
        painter.fillPath(fillPath, fillColor);

        // 20260323 ZJH 绘制曲线线条
        painter.setPen(QPen(curve.color, 2.0));
        painter.drawPath(path);
    }
}

// 20260323 ZJH 绘制图例
void ROCPRCurveChart::drawLegend(QPainter& painter) const
{
    int nLegendX = static_cast<int>(m_plotRect.right()) - 160;  // 20260323 ZJH 图例 X 位置
    int nLegendY = static_cast<int>(m_plotRect.top()) + 10;     // 20260323 ZJH 图例 Y 位置
    int nLineHeight = 18;  // 20260323 ZJH 每行高度

    // 20260323 ZJH 图例背景
    int nLegendHeight = m_vecCurves.size() * nLineHeight + 8;
    painter.fillRect(QRect(nLegendX - 4, nLegendY - 4, 164, nLegendHeight),
                     QColor(26, 29, 36, 200));

    painter.setFont(QFont(ThemeColors::s_strFontFamily, 8));

    for (int i = 0; i < m_vecCurves.size(); ++i) {
        int nY = nLegendY + i * nLineHeight;  // 20260323 ZJH 当前行 Y

        // 20260323 ZJH 颜色方块
        painter.fillRect(QRect(nLegendX, nY + 2, 12, 12), m_vecCurves[i].color);

        // 20260323 ZJH 类别名 + AUC 值
        painter.setPen(QColor("#e2e8f0"));
        QString strText = m_vecCurves[i].strClassName +
                          " (AUC=" + QString::number(m_vecCurves[i].dAUC, 'f', 3) + ")";
        painter.drawText(nLegendX + 16, nY + 13, strText);
    }
}

// 20260323 ZJH 绘制参考线
void ROCPRCurveChart::drawReferenceLine(QPainter& painter) const
{
    painter.setPen(QPen(QColor("#475569"), 1.5, Qt::DashLine));

    if (m_curveType == CurveType::ROC) {
        // 20260323 ZJH ROC: 对角线 (0,0)→(1,1) 表示随机分类器
        painter.drawLine(dataToPixel(0.0, 0.0), dataToPixel(1.0, 1.0));
    } else {
        // 20260323 ZJH PR: 水平线（表示随机分类器的平均精确率）
        // 不画参考线，PR 曲线没有标准参考线
    }
}

// 20260323 ZJH 默认曲线颜色循环
const QVector<QColor>& ROCPRCurveChart::defaultColors()
{
    static const QVector<QColor> s_colors = {
        QColor("#3b82f6"),  // 20260323 ZJH 蓝色
        QColor("#f59e0b"),  // 20260323 ZJH 橙色
        QColor("#10b981"),  // 20260323 ZJH 绿色
        QColor("#8b5cf6"),  // 20260323 ZJH 紫色
        QColor("#ef4444"),  // 20260323 ZJH 红色
        QColor("#06b6d4"),  // 20260323 ZJH 青色
        QColor("#ec4899"),  // 20260323 ZJH 粉色
        QColor("#84cc16")   // 20260323 ZJH 黄绿色
    };
    return s_colors;
}
