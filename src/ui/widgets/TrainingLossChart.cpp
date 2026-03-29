// 20260322 ZJH TrainingLossChart 实现
// 初始化 Qt Charts 图表、坐标轴、系列，实时绘制训练/验证损失曲线

#include "ui/widgets/TrainingLossChart.h"  // 20260322 ZJH TrainingLossChart 类声明
#include "ui/widgets/ThemeColors.h"        // 20260324 ZJH 共享主题颜色和字体族名

#include <QVBoxLayout>              // 20260322 ZJH 垂直布局容纳 QChartView
#include <QtCharts/QChart>         // 20260322 ZJH 图表核心类
#include <QtCharts/QChartView>     // 20260322 ZJH 图表视图控件
#include <QtCharts/QLineSeries>    // 20260322 ZJH 折线系列
#include <QtCharts/QValueAxis>     // 20260322 ZJH 数值坐标轴
#include <QtCharts/QLegend>        // 20260322 ZJH 图例
#include <QtCharts/QLegendMarker>  // 20260322 ZJH 图例标记

#include <algorithm>  // 20260322 ZJH std::max_element 用于找最大值

// 20260322 ZJH 构造函数，初始化图表和坐标轴
TrainingLossChart::TrainingLossChart(QWidget* pParent)
    : QWidget(pParent)
    , m_pChart(nullptr)
    , m_pChartView(nullptr)
    , m_pTrainSeries(nullptr)
    , m_pValSeries(nullptr)
    , m_pAxisX(nullptr)
    , m_pAxisY(nullptr)
{
    // 20260322 ZJH 1. 创建训练损失折线系列（蓝色 #2563eb）
    m_pTrainSeries = new QLineSeries(this);
    m_pTrainSeries->setName(QStringLiteral("训练损失"));  // 20260322 ZJH 图例中显示的名称
    QPen penTrain(QColor("#2563eb"));  // 20260322 ZJH 蓝色画笔
    penTrain.setWidth(2);  // 20260322 ZJH 线宽 2px
    m_pTrainSeries->setPen(penTrain);

    // 20260322 ZJH 2. 创建验证损失折线系列（橙色 #f59e0b）
    m_pValSeries = new QLineSeries(this);
    m_pValSeries->setName(QStringLiteral("验证损失"));  // 20260322 ZJH 图例中显示的名称
    QPen penVal(QColor("#f59e0b"));  // 20260322 ZJH 橙色画笔
    penVal.setWidth(2);  // 20260322 ZJH 线宽 2px
    m_pValSeries->setPen(penVal);

    // 20260322 ZJH 3. 创建 X 轴（Epoch）
    m_pAxisX = new QValueAxis(this);
    m_pAxisX->setTitleText(QStringLiteral("轮次 (Epoch)"));  // 20260324 ZJH 轴标题汉化
    m_pAxisX->setLabelFormat("%d");  // 20260322 ZJH 整数格式
    m_pAxisX->setRange(0, 10);       // 20260322 ZJH 初始范围 0~10
    m_pAxisX->setGridLineVisible(true);  // 20260322 ZJH 显示网格线
    m_pAxisX->setGridLineColor(QColor("#333842"));  // 20260322 ZJH 暗色网格线
    m_pAxisX->setLabelsColor(QColor("#94a3b8"));    // 20260322 ZJH 浅灰标签文字
    m_pAxisX->setTitleBrush(QBrush(QColor("#94a3b8")));  // 20260322 ZJH 标题颜色

    // 20260322 ZJH 4. 创建 Y 轴（Loss）
    m_pAxisY = new QValueAxis(this);
    m_pAxisY->setTitleText(QStringLiteral("损失 (Loss)"));  // 20260324 ZJH 轴标题汉化
    m_pAxisY->setLabelFormat("%.4f");  // 20260322 ZJH 4位小数
    m_pAxisY->setRange(0, 1.0);        // 20260322 ZJH 初始范围 0~1
    m_pAxisY->setGridLineVisible(true);  // 20260322 ZJH 显示网格线
    m_pAxisY->setGridLineColor(QColor("#333842"));  // 20260322 ZJH 暗色网格线
    m_pAxisY->setLabelsColor(QColor("#94a3b8"));    // 20260322 ZJH 浅灰标签文字
    m_pAxisY->setTitleBrush(QBrush(QColor("#94a3b8")));  // 20260322 ZJH 标题颜色

    // 20260322 ZJH 5. 创建图表对象
    m_pChart = new QChart();
    m_pChart->addSeries(m_pTrainSeries);  // 20260322 ZJH 添加训练损失系列
    m_pChart->addSeries(m_pValSeries);    // 20260322 ZJH 添加验证损失系列

    // 20260322 ZJH 6. 将坐标轴关联到系列
    m_pChart->addAxis(m_pAxisX, Qt::AlignBottom);  // 20260322 ZJH X轴在底部
    m_pChart->addAxis(m_pAxisY, Qt::AlignLeft);    // 20260322 ZJH Y轴在左侧
    m_pTrainSeries->attachAxis(m_pAxisX);  // 20260322 ZJH 训练系列绑定 X 轴
    m_pTrainSeries->attachAxis(m_pAxisY);  // 20260322 ZJH 训练系列绑定 Y 轴
    m_pValSeries->attachAxis(m_pAxisX);    // 20260322 ZJH 验证系列绑定 X 轴
    m_pValSeries->attachAxis(m_pAxisY);    // 20260322 ZJH 验证系列绑定 Y 轴

    // 20260322 ZJH 7. 设置暗色背景
    m_pChart->setBackgroundBrush(QBrush(QColor("#22262e")));  // 20260322 ZJH 图表区域暗色背景
    m_pChart->setPlotAreaBackgroundBrush(QBrush(QColor("#1a1d24")));  // 20260322 ZJH 绘图区稍深
    m_pChart->setPlotAreaBackgroundVisible(true);  // 20260322 ZJH 显示绘图区背景

    // 20260322 ZJH 8. 去掉图表标题（由外部提供上下文）
    m_pChart->setTitle(QString());

    // 20260322 ZJH 9. 配置图例位置在右上角
    m_pChart->legend()->setVisible(true);    // 20260322 ZJH 显示图例
    m_pChart->legend()->setAlignment(Qt::AlignTop);  // 20260322 ZJH 图例对齐到顶部
    m_pChart->legend()->setLabelColor(QColor("#e2e8f0"));  // 20260322 ZJH 图例文字颜色
    m_pChart->legend()->setFont(QFont(ThemeColors::s_strFontFamily, 9));  // 20260324 ZJH 图例字体，使用共享字体族名

    // 20260322 ZJH 10. 去掉图表边距
    m_pChart->setMargins(QMargins(4, 4, 4, 4));  // 20260322 ZJH 紧凑边距

    // 20260322 ZJH 11. 创建图表视图
    m_pChartView = new QChartView(m_pChart, this);
    m_pChartView->setRenderHint(QPainter::Antialiasing);  // 20260322 ZJH 启用抗锯齿
    m_pChartView->setBackgroundBrush(QBrush(QColor("#22262e")));  // 20260322 ZJH 视图背景

    // 20260322 ZJH 12. 将图表视图放入布局
    QVBoxLayout* pLayout = new QVBoxLayout(this);
    pLayout->setContentsMargins(0, 0, 0, 0);  // 20260322 ZJH 无边距
    pLayout->addWidget(m_pChartView);          // 20260322 ZJH 填满整个控件
}

// 20260322 ZJH 添加训练损失数据点
void TrainingLossChart::addTrainPoint(int nEpoch, double dLoss)
{
    // 20260322 ZJH 存储原始数据（用于平滑计算）
    m_vecTrainRaw.append(QPointF(nEpoch, dLoss));

    if (m_bSmoothing) {
        // 20260322 ZJH 启用平滑时，重新计算 EMA 并更新显示系列
        applySmoothingToSeries();
    } else {
        // 20260322 ZJH 未启用平滑时，直接追加到显示系列
        m_pTrainSeries->append(nEpoch, dLoss);
    }

    // 20260322 ZJH 自动调整坐标轴范围
    updateAxisRanges();
}

// 20260322 ZJH 添加验证损失数据点
void TrainingLossChart::addValPoint(int nEpoch, double dLoss)
{
    // 20260322 ZJH 存储原始数据（用于平滑计算）
    m_vecValRaw.append(QPointF(nEpoch, dLoss));

    if (m_bSmoothing) {
        // 20260322 ZJH 启用平滑时，重新计算 EMA 并更新显示系列
        applySmoothingToSeries();
    } else {
        // 20260322 ZJH 未启用平滑时，直接追加到显示系列
        m_pValSeries->append(nEpoch, dLoss);
    }

    // 20260322 ZJH 自动调整坐标轴范围
    updateAxisRanges();
}

// 20260322 ZJH 清空所有数据点并重置坐标轴
void TrainingLossChart::clear()
{
    // 20260322 ZJH 清空原始数据
    m_vecTrainRaw.clear();
    m_vecValRaw.clear();

    // 20260322 ZJH 清空显示系列
    m_pTrainSeries->clear();
    m_pValSeries->clear();

    // 20260322 ZJH 重置坐标轴为默认范围
    m_pAxisX->setRange(0, 10);
    m_pAxisY->setRange(0, 1.0);
}

// 20260322 ZJH 设置是否启用 EMA 平滑
void TrainingLossChart::setSmoothing(bool bEnabled)
{
    // 20260322 ZJH 记录平滑状态
    m_bSmoothing = bEnabled;

    if (bEnabled) {
        // 20260322 ZJH 切换到平滑模式：重新用 EMA 计算并刷新显示
        applySmoothingToSeries();
    } else {
        // 20260322 ZJH 切换到原始数据模式：将原始数据直接填充到显示系列
        m_pTrainSeries->clear();
        for (const auto& pt : m_vecTrainRaw) {
            m_pTrainSeries->append(pt);  // 20260322 ZJH 逐点追加原始训练数据
        }

        m_pValSeries->clear();
        for (const auto& pt : m_vecValRaw) {
            m_pValSeries->append(pt);  // 20260322 ZJH 逐点追加原始验证数据
        }
    }

    // 20260322 ZJH 刷新坐标轴范围
    updateAxisRanges();
}

// 20260322 ZJH 自动缩放坐标轴范围
void TrainingLossChart::updateAxisRanges()
{
    // 20260322 ZJH 收集所有数据点
    double dMaxEpoch = 10.0;   // 20260322 ZJH X轴最小范围 10
    double dMaxLoss  = 0.1;    // 20260322 ZJH Y轴最小范围 0.1

    // 20260322 ZJH 遍历原始训练数据找最大 epoch 和最大 loss
    for (const auto& pt : m_vecTrainRaw) {
        if (pt.x() > dMaxEpoch) dMaxEpoch = pt.x();  // 20260322 ZJH 更新最大 epoch
        if (pt.y() > dMaxLoss)  dMaxLoss  = pt.y();   // 20260322 ZJH 更新最大 loss
    }

    // 20260322 ZJH 遍历原始验证数据找最大 epoch 和最大 loss
    for (const auto& pt : m_vecValRaw) {
        if (pt.x() > dMaxEpoch) dMaxEpoch = pt.x();  // 20260322 ZJH 更新最大 epoch
        if (pt.y() > dMaxLoss)  dMaxLoss  = pt.y();   // 20260322 ZJH 更新最大 loss
    }

    // 20260322 ZJH 给 Y 轴留 10% 的上方空间避免曲线贴顶
    dMaxLoss *= 1.1;

    // 20260322 ZJH 更新坐标轴范围
    m_pAxisX->setRange(0, dMaxEpoch + 1);  // 20260322 ZJH X轴多留1个 epoch
    m_pAxisY->setRange(0, dMaxLoss);        // 20260322 ZJH Y轴从0到最大 loss * 1.1
}

// 20260322 ZJH 将原始数据应用 EMA 平滑后更新显示系列
void TrainingLossChart::applySmoothingToSeries()
{
    // 20260322 ZJH EMA 公式: smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i-1]
    // alpha 越小，曲线越平滑

    // 20260322 ZJH 处理训练损失系列
    m_pTrainSeries->clear();  // 20260322 ZJH 清空当前显示
    if (!m_vecTrainRaw.isEmpty()) {
        double dSmoothed = m_vecTrainRaw.first().y();  // 20260322 ZJH 第一个点直接作为初始值
        for (const auto& pt : m_vecTrainRaw) {
            dSmoothed = m_dSmoothAlpha * pt.y() + (1.0 - m_dSmoothAlpha) * dSmoothed;  // 20260322 ZJH EMA 计算
            m_pTrainSeries->append(pt.x(), dSmoothed);  // 20260322 ZJH 追加平滑后的点
        }
    }

    // 20260322 ZJH 处理验证损失系列
    m_pValSeries->clear();  // 20260322 ZJH 清空当前显示
    if (!m_vecValRaw.isEmpty()) {
        double dSmoothed = m_vecValRaw.first().y();  // 20260322 ZJH 第一个点直接作为初始值
        for (const auto& pt : m_vecValRaw) {
            dSmoothed = m_dSmoothAlpha * pt.y() + (1.0 - m_dSmoothAlpha) * dSmoothed;  // 20260322 ZJH EMA 计算
            m_pValSeries->append(pt.x(), dSmoothed);  // 20260322 ZJH 追加平滑后的点
        }
    }
}
