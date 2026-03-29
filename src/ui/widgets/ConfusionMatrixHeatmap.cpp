// 20260322 ZJH ConfusionMatrixHeatmap 实现
// 使用 QPainter 自绘混淆矩阵热力图，支持行/列归一化和鼠标悬停高亮

#include "ui/widgets/ConfusionMatrixHeatmap.h"  // 20260322 ZJH 类声明
#include "ui/widgets/ThemeColors.h"             // 20260324 ZJH 共享主题颜色和字体族名

#include <QPainter>       // 20260322 ZJH 绘图引擎
#include <QPaintEvent>    // 20260322 ZJH 绘图事件
#include <QMouseEvent>    // 20260322 ZJH 鼠标事件
#include <QToolTip>       // 20260322 ZJH 鼠标悬停提示（预留）
#include <QtMath>         // 20260322 ZJH qMax 等数学函数

#include <algorithm>      // 20260322 ZJH std::max_element

// 20260322 ZJH 构造函数
ConfusionMatrixHeatmap::ConfusionMatrixHeatmap(QWidget* pParent)
    : QWidget(pParent)
    , m_eNormMode(NormMode::Count)  // 20260322 ZJH 默认显示原始计数
    , m_ptHoverCell(-1, -1)          // 20260322 ZJH 初始无悬停
{
    // 20260322 ZJH 启用鼠标追踪，不需要按住按钮即可触发 mouseMoveEvent
    setMouseTracking(true);

    // 20260322 ZJH 设置暗色背景
    setAutoFillBackground(true);
    QPalette pal = palette();  // 20260322 ZJH 获取当前调色板
    pal.setColor(QPalette::Window, QColor("#22262e"));  // 20260322 ZJH 暗灰背景
    setPalette(pal);  // 20260322 ZJH 应用调色板
}

// 20260322 ZJH 设置混淆矩阵数据和类别名称
void ConfusionMatrixHeatmap::setData(const QVector<QVector<int>>& matData,
                                     const QStringList& vecClassNames)
{
    m_matData = matData;              // 20260322 ZJH 保存矩阵数据
    m_vecClassNames = vecClassNames;  // 20260322 ZJH 保存类别名称
    m_ptHoverCell = QPoint(-1, -1);   // 20260322 ZJH 重置悬停状态
    m_bNormCacheDirty = true;         // 20260323 ZJH 标记缓存失效
    update();  // 20260322 ZJH 触发重绘
}

// 20260322 ZJH 设置归一化模式
void ConfusionMatrixHeatmap::setNormMode(int nMode)
{
    // 20260324 ZJH 范围校验：nMode 必须在 [0, 2] 之间，否则忽略无效值直接返回
    if (nMode < 0 || nMode > 2) {
        return;  // 20260324 ZJH 非法模式值，不改变当前状态
    }

    // 20260322 ZJH 将整数参数转为枚举值
    m_eNormMode = static_cast<NormMode>(nMode);
    m_bNormCacheDirty = true;  // 20260323 ZJH 模式变化时标记缓存失效
    update();  // 20260322 ZJH 触发重绘以更新显示
}

// 20260322 ZJH 清空数据
void ConfusionMatrixHeatmap::clear()
{
    m_matData.clear();          // 20260322 ZJH 清空矩阵
    m_vecClassNames.clear();    // 20260322 ZJH 清空类别名称
    m_bNormCacheDirty = true;   // 20260323 ZJH 标记缓存失效
    m_ptHoverCell = QPoint(-1, -1);  // 20260322 ZJH 重置悬停
    update();  // 20260322 ZJH 触发重绘
}

// 20260322 ZJH 推荐最小尺寸
QSize ConfusionMatrixHeatmap::minimumSizeHint() const
{
    int nN = m_vecClassNames.size();  // 20260322 ZJH 类别数量
    if (nN == 0) {
        return QSize(200, 200);  // 20260322 ZJH 无数据时最小 200x200
    }
    // 20260322 ZJH 每个单元格至少 40x40，加上标签边距和内边距
    int nSize = s_nLabelMargin + s_nPadding * 2 + nN * 40;
    return QSize(nSize, nSize);  // 20260322 ZJH 正方形
}

// 20260322 ZJH 自绘混淆矩阵热力图
void ConfusionMatrixHeatmap::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);  // 20260322 ZJH 创建绘图对象
    painter.setRenderHint(QPainter::Antialiasing, true);  // 20260322 ZJH 启用抗锯齿

    int nN = m_vecClassNames.size();  // 20260322 ZJH 类别数量

    // 20260322 ZJH 无数据时显示占位提示
    if (nN == 0 || m_matData.isEmpty()) {
        painter.setPen(QColor("#64748b"));  // 20260322 ZJH 灰色文字
        painter.setFont(QFont(ThemeColors::s_strFontFamily, 12));  // 20260324 ZJH 中号字体，使用共享字体族名
        painter.drawText(rect(), Qt::AlignCenter, QStringLiteral("暂无混淆矩阵数据"));
        return;  // 20260322 ZJH 无数据直接返回
    }

    // 20260322 ZJH 计算绘制区域
    int nAvailW = width() - s_nLabelMargin - s_nPadding * 2;   // 20260322 ZJH 可用宽度
    int nAvailH = height() - s_nLabelMargin - s_nPadding * 2;  // 20260322 ZJH 可用高度
    int nCellSize = qMin(nAvailW, nAvailH) / nN;  // 20260322 ZJH 单元格大小（正方形）
    if (nCellSize < 20) {
        nCellSize = 20;  // 20260322 ZJH 最小 20px
    }

    // 20260322 ZJH 矩阵绘制起始坐标
    int nStartX = s_nPadding + s_nLabelMargin;  // 20260322 ZJH 左侧留标签空间
    int nStartY = s_nPadding + s_nLabelMargin;  // 20260322 ZJH 顶部留标签空间

    // 20260322 ZJH 获取归一化矩阵
    QVector<QVector<double>> matNorm = normalizedMatrix();

    // 20260322 ZJH 找出矩阵最大值（用于颜色映射缩放）
    double dMaxVal = 0;
    for (int r = 0; r < nN; ++r) {
        for (int c = 0; c < nN; ++c) {
            if (matNorm[r][c] > dMaxVal) {
                dMaxVal = matNorm[r][c];  // 20260322 ZJH 更新最大值
            }
        }
    }
    if (dMaxVal < 1e-9) {
        dMaxVal = 1.0;  // 20260322 ZJH 全零矩阵时避免除零
    }

    // 20260322 ZJH 绘制矩阵单元格
    QFont fontCell(ThemeColors::s_strFontFamily, 9);  // 20260324 ZJH 单元格数值字体，使用共享字体族名
    painter.setFont(fontCell);

    for (int r = 0; r < nN; ++r) {
        for (int c = 0; c < nN; ++c) {
            // 20260322 ZJH 计算当前单元格矩形
            QRect rectCell(nStartX + c * nCellSize,
                           nStartY + r * nCellSize,
                           nCellSize, nCellSize);

            // 20260322 ZJH 根据归一化值计算颜色
            double dColorVal = matNorm[r][c] / dMaxVal;  // 20260322 ZJH 映射到 0~1
            QColor cellColor = valueToColor(dColorVal);  // 20260322 ZJH 获取渐变颜色

            // 20260322 ZJH 检查是否为悬停行或列
            bool bHighlight = (m_ptHoverCell.x() >= 0 &&
                              (r == m_ptHoverCell.y() || c == m_ptHoverCell.x()));

            if (bHighlight) {
                // 20260322 ZJH 悬停行/列时增加亮度
                cellColor = cellColor.lighter(130);
            }

            // 20260322 ZJH 填充单元格背景
            painter.fillRect(rectCell, cellColor);

            // 20260322 ZJH 绘制单元格边框
            painter.setPen(QPen(QColor("#333842"), 1));  // 20260322 ZJH 暗色边框
            painter.drawRect(rectCell);

            // 20260322 ZJH 绘制单元格内数值文字
            QString strText;
            if (m_eNormMode == NormMode::Count) {
                strText = QString::number(m_matData[r][c]);  // 20260322 ZJH 原始计数
            } else {
                strText = QString::number(matNorm[r][c], 'f', 2);  // 20260322 ZJH 归一化值保留2位小数
            }

            // 20260322 ZJH 根据背景亮度选择文字颜色（深色背景用白字，浅色用黑字）
            double dLuminance = 0.299 * cellColor.redF() +
                                0.587 * cellColor.greenF() +
                                0.114 * cellColor.blueF();
            painter.setPen(dLuminance > 0.5 ? QColor("#1a1d24") : QColor("#f1f5f9"));

            painter.drawText(rectCell, Qt::AlignCenter, strText);
        }
    }

    // 20260322 ZJH 绘制行标签（左侧，真实类别 — Y 轴）
    QFont fontLabel(ThemeColors::s_strFontFamily, 8);  // 20260324 ZJH 标签字体，使用共享字体族名
    painter.setFont(fontLabel);
    painter.setPen(QColor("#94a3b8"));  // 20260322 ZJH 浅灰色标签文字

    for (int r = 0; r < nN; ++r) {
        // 20260322 ZJH 行标签在单元格左侧居中显示
        QRect rectLabel(s_nPadding,
                        nStartY + r * nCellSize,
                        s_nLabelMargin - 4,
                        nCellSize);
        // 20260322 ZJH 高亮悬停行标签
        if (m_ptHoverCell.y() == r) {
            painter.setPen(QColor("#e2e8f0"));  // 20260322 ZJH 白色高亮
        } else {
            painter.setPen(QColor("#94a3b8"));  // 20260322 ZJH 正常灰色
        }
        painter.drawText(rectLabel, Qt::AlignRight | Qt::AlignVCenter, m_vecClassNames[r]);
    }

    // 20260322 ZJH 绘制列标签（顶部，预测类别 — X 轴）
    for (int c = 0; c < nN; ++c) {
        // 20260322 ZJH 列标签需要旋转 45 度显示以节省空间
        painter.save();  // 20260322 ZJH 保存当前坐标变换状态

        // 20260322 ZJH 移动原点到列标签中心位置
        int nCenterX = nStartX + c * nCellSize + nCellSize / 2;
        int nCenterY = nStartY - 4;
        painter.translate(nCenterX, nCenterY);
        painter.rotate(-45);  // 20260322 ZJH 逆时针旋转 45 度

        // 20260322 ZJH 高亮悬停列标签
        if (m_ptHoverCell.x() == c) {
            painter.setPen(QColor("#e2e8f0"));  // 20260322 ZJH 白色高亮
        } else {
            painter.setPen(QColor("#94a3b8"));  // 20260322 ZJH 正常灰色
        }
        painter.drawText(QRect(-60, -10, 60, 20), Qt::AlignRight | Qt::AlignVCenter,
                         m_vecClassNames[c]);

        painter.restore();  // 20260322 ZJH 恢复坐标变换
    }

    // 20260322 ZJH 绘制轴标题
    QFont fontTitle(ThemeColors::s_strFontFamily, 9, QFont::Bold);  // 20260324 ZJH 轴标题字体，使用共享字体族名
    painter.setFont(fontTitle);
    painter.setPen(QColor("#cbd5e1"));  // 20260322 ZJH 较亮的标题颜色

    // 20260322 ZJH Y 轴标题（"真实类别"）—— 竖向绘制在最左侧
    painter.save();
    painter.translate(s_nPadding - 2, nStartY + nN * nCellSize / 2);
    painter.rotate(-90);  // 20260322 ZJH 旋转 90 度
    painter.drawText(QRect(-60, -14, 120, 20), Qt::AlignCenter, QStringLiteral("真实类别"));
    painter.restore();

    // 20260322 ZJH X 轴标题（"预测类别"）—— 绘制在底部中央
    painter.drawText(QRect(nStartX, nStartY + nN * nCellSize + 4,
                           nN * nCellSize, 20),
                     Qt::AlignCenter, QStringLiteral("预测类别"));
}

// 20260322 ZJH 鼠标移动事件：更新悬停单元格并触发重绘
void ConfusionMatrixHeatmap::mouseMoveEvent(QMouseEvent* pEvent)
{
    QPoint ptNew = cellAtPos(pEvent->position().toPoint());  // 20260322 ZJH 计算当前鼠标所在的单元格
    if (ptNew != m_ptHoverCell) {
        m_ptHoverCell = ptNew;  // 20260322 ZJH 更新悬停单元格
        update();  // 20260322 ZJH 触发重绘以更新高亮
    }
    QWidget::mouseMoveEvent(pEvent);  // 20260322 ZJH 调用基类处理
}

// 20260322 ZJH 鼠标离开事件：取消悬停高亮
void ConfusionMatrixHeatmap::leaveEvent(QEvent* pEvent)
{
    m_ptHoverCell = QPoint(-1, -1);  // 20260322 ZJH 重置悬停状态
    update();  // 20260322 ZJH 触发重绘以取消高亮
    QWidget::leaveEvent(pEvent);  // 20260322 ZJH 调用基类处理
}

// 20260322 ZJH 根据归一化值返回渐变颜色（深蓝 → 青 → 黄 → 红）
QColor ConfusionMatrixHeatmap::valueToColor(double dValue) const
{
    // 20260322 ZJH 限制值范围 [0, 1]
    dValue = qBound(0.0, dValue, 1.0);

    // 20260322 ZJH 四段渐变色带：深蓝 → 蓝 → 青绿 → 橙 → 红
    // 参考 viridis 配色方案
    struct ColorStop {
        double dPos;   // 20260322 ZJH 位置 [0, 1]
        int nR, nG, nB;  // 20260322 ZJH RGB 颜色值
    };

    static const ColorStop arrStops[] = {
        { 0.0,  13,  27,  42 },   // 20260322 ZJH 深蓝 #0d1b2a
        { 0.25, 30,  80, 140 },   // 20260322 ZJH 蓝色 #1e508c
        { 0.50, 40, 170, 160 },   // 20260322 ZJH 青绿 #28aaa0
        { 0.75, 240, 180,  40 },  // 20260322 ZJH 橙黄 #f0b428
        { 1.0,  230,  57,  70 }   // 20260322 ZJH 红色 #e63946
    };

    static const int nStops = 5;  // 20260322 ZJH 色带节点数量

    // 20260322 ZJH 找到 dValue 所在的区间并线性插值
    for (int i = 0; i < nStops - 1; ++i) {
        if (dValue <= arrStops[i + 1].dPos) {
            // 20260322 ZJH 计算区间内的插值系数 t
            double dRange = arrStops[i + 1].dPos - arrStops[i].dPos;
            double t = (dRange > 1e-9) ? (dValue - arrStops[i].dPos) / dRange : 0.0;

            // 20260322 ZJH 线性插值 RGB 各通道
            int nR = static_cast<int>(arrStops[i].nR + t * (arrStops[i + 1].nR - arrStops[i].nR));
            int nG = static_cast<int>(arrStops[i].nG + t * (arrStops[i + 1].nG - arrStops[i].nG));
            int nB = static_cast<int>(arrStops[i].nB + t * (arrStops[i + 1].nB - arrStops[i].nB));

            return QColor(qBound(0, nR, 255), qBound(0, nG, 255), qBound(0, nB, 255));
        }
    }

    // 20260322 ZJH 超出范围时返回最高色（红色）
    return QColor(230, 57, 70);
}

// 20260322 ZJH 计算归一化后的矩阵
QVector<QVector<double>> ConfusionMatrixHeatmap::normalizedMatrix() const
{
    // 20260323 ZJH 使用缓存避免每帧重算 O(N^2)
    if (!m_bNormCacheDirty) return m_matNormCache;
    m_bNormCacheDirty = false;

    int nN = m_matData.size();  // 20260322 ZJH 矩阵大小
    QVector<QVector<double>> matNorm(nN, QVector<double>(nN, 0.0));

    if (nN == 0) {
        return matNorm;  // 20260322 ZJH 空矩阵直接返回
    }

    if (m_eNormMode == NormMode::Count) {
        // 20260322 ZJH 计数模式：直接转为 double
        for (int r = 0; r < nN; ++r) {
            for (int c = 0; c < nN; ++c) {
                matNorm[r][c] = static_cast<double>(m_matData[r][c]);
            }
        }
    } else if (m_eNormMode == NormMode::RowNorm) {
        // 20260322 ZJH 行归一化：每行除以行总和
        for (int r = 0; r < nN; ++r) {
            double dRowSum = 0;
            for (int c = 0; c < nN; ++c) {
                dRowSum += m_matData[r][c];  // 20260322 ZJH 累加行总和
            }
            for (int c = 0; c < nN; ++c) {
                matNorm[r][c] = (dRowSum > 0) ? m_matData[r][c] / dRowSum : 0.0;
            }
        }
    } else if (m_eNormMode == NormMode::ColumnNorm) {
        // 20260322 ZJH 列归一化：每列除以列总和
        for (int c = 0; c < nN; ++c) {
            double dColSum = 0;
            for (int r = 0; r < nN; ++r) {
                dColSum += m_matData[r][c];  // 20260322 ZJH 累加列总和
            }
            for (int r = 0; r < nN; ++r) {
                matNorm[r][c] = (dColSum > 0) ? m_matData[r][c] / dColSum : 0.0;
            }
        }
    }

    m_matNormCache = matNorm;  // 20260323 ZJH 写入缓存
    return matNorm;  // 20260322 ZJH 返回归一化后的矩阵
}

// 20260322 ZJH 根据鼠标位置计算单元格索引
QPoint ConfusionMatrixHeatmap::cellAtPos(const QPoint& ptPos) const
{
    int nN = m_vecClassNames.size();  // 20260322 ZJH 类别数量
    if (nN == 0) {
        return QPoint(-1, -1);  // 20260322 ZJH 无数据
    }

    // 20260322 ZJH 计算单元格大小（与 paintEvent 一致）
    int nAvailW = width() - s_nLabelMargin - s_nPadding * 2;
    int nAvailH = height() - s_nLabelMargin - s_nPadding * 2;
    int nCellSize = qMin(nAvailW, nAvailH) / nN;
    if (nCellSize < 20) {
        nCellSize = 20;
    }

    int nStartX = s_nPadding + s_nLabelMargin;
    int nStartY = s_nPadding + s_nLabelMargin;

    // 20260322 ZJH 计算鼠标所在的列和行
    int nCol = (ptPos.x() - nStartX) / nCellSize;
    int nRow = (ptPos.y() - nStartY) / nCellSize;

    // 20260322 ZJH 检查边界
    if (nCol >= 0 && nCol < nN && nRow >= 0 && nRow < nN) {
        return QPoint(nCol, nRow);  // 20260322 ZJH 有效单元格
    }

    return QPoint(-1, -1);  // 20260322 ZJH 超出矩阵范围
}
