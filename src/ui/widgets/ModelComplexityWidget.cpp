// 20260323 ZJH ModelComplexityWidget — 模型复杂度信息控件实现

#include "ui/widgets/ModelComplexityWidget.h"

#include <QPainter>  // 20260323 ZJH 绘图引擎

// 20260323 ZJH 构造函数
ModelComplexityWidget::ModelComplexityWidget(QWidget* pParent)
    : QWidget(pParent)
{
    setFixedHeight(160);  // 20260323 ZJH 固定高度
    setMinimumWidth(180);
}

// 20260323 ZJH 设置模型信息
void ModelComplexityWidget::setInfo(const ModelComplexityInfo& info)
{
    m_info = info;
    m_bHasData = true;
    update();
}

// 20260323 ZJH 清空显示
void ModelComplexityWidget::clear()
{
    m_info = ModelComplexityInfo();
    m_bHasData = false;
    update();
}

// 20260323 ZJH 绘制事件
void ModelComplexityWidget::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 20260323 ZJH 背景
    painter.fillRect(rect(), QColor("#1a1d24"));

    if (!m_bHasData) {
        // 20260323 ZJH 无数据时显示占位文字
        painter.setPen(QColor("#64748b"));
        painter.setFont(QFont("Segoe UI", 10));
        painter.drawText(rect(), Qt::AlignCenter, "No model selected");
        return;
    }

    int nX = 8;   // 20260323 ZJH 左边距
    int nY = 8;   // 20260323 ZJH 起始 Y

    // 20260323 ZJH 架构名称标题
    painter.setPen(QColor("#e2e8f0"));
    painter.setFont(QFont("Segoe UI", 10, QFont::Bold));
    painter.drawText(nX, nY + 14, m_info.strArchitecture);
    nY += 24;

    // 20260323 ZJH 绘制各项指标
    painter.setFont(QFont("Segoe UI", 9));

    auto drawRow = [&](const QString& strLabel, const QString& strValue) {
        painter.setPen(QColor("#94a3b8"));
        painter.drawText(nX, nY + 12, strLabel);
        painter.setPen(QColor("#e2e8f0"));
        painter.drawText(width() / 2, nY + 12, strValue);
        nY += 20;
    };

    drawRow("Parameters:", formatParams(m_info.nTotalParams));
    drawRow("Trainable:", formatParams(m_info.nTrainableParams));
    drawRow("FLOPs:", formatFLOPs(m_info.dFLOPs));
    drawRow("Memory:", QString::number(m_info.dMemoryMB, 'f', 1) + " MB");
    drawRow("Layers:", QString::number(m_info.nLayers));
    drawRow("Input:", QString::number(m_info.nInputSize) + "x" + QString::number(m_info.nInputSize));
}

// 20260323 ZJH 格式化参数量
QString ModelComplexityWidget::formatParams(qint64 nParams)
{
    if (nParams >= 1000000000) {
        return QString::number(nParams / 1.0e9, 'f', 2) + "B";  // 20260323 ZJH 十亿
    } else if (nParams >= 1000000) {
        return QString::number(nParams / 1.0e6, 'f', 2) + "M";  // 20260323 ZJH 百万
    } else if (nParams >= 1000) {
        return QString::number(nParams / 1.0e3, 'f', 1) + "K";  // 20260323 ZJH 千
    }
    return QString::number(nParams);
}

// 20260323 ZJH 格式化 FLOPs
QString ModelComplexityWidget::formatFLOPs(double dFLOPs)
{
    if (dFLOPs >= 1000.0) {
        return QString::number(dFLOPs / 1000.0, 'f', 2) + " TFLOPs";
    } else if (dFLOPs >= 1.0) {
        return QString::number(dFLOPs, 'f', 2) + " GFLOPs";
    } else if (dFLOPs >= 0.001) {
        return QString::number(dFLOPs * 1000.0, 'f', 1) + " MFLOPs";
    }
    return "N/A";
}
