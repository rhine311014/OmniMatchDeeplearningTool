// 20260322 ZJH TrainingLossChart — 训练损失曲线图表控件
// 使用 Qt Charts 绘制训练损失和验证损失的双线折线图
// 支持实时添加数据点、自动缩放坐标轴、指数移动平均平滑
// 暗色背景适配 (#22262e)，图例右上角显示

#pragma once

#include <QWidget>          // 20260322 ZJH 基础控件基类
#include <QVector>          // 20260322 ZJH 存储历史损失数据点

// 20260322 ZJH Qt6 Charts 类前向声明（Qt6 中 Charts 类位于全局命名空间）
class QChart;
class QChartView;
class QLineSeries;
class QValueAxis;

// 20260322 ZJH 训练损失曲线图表控件
// 持有两条折线：训练损失（蓝色 #2563eb）和验证损失（橙色 #f59e0b）
// 支持实时添加数据点、自动缩放、EMA 平滑
class TrainingLossChart : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化图表和坐标轴
    // 参数: pParent - 父控件指针
    explicit TrainingLossChart(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~TrainingLossChart() override = default;

    // 20260322 ZJH 添加训练损失数据点
    // 参数: nEpoch - 当前训练轮次（X轴）
    //       dLoss  - 训练集损失值（Y轴）
    void addTrainPoint(int nEpoch, double dLoss);

    // 20260322 ZJH 添加验证损失数据点
    // 参数: nEpoch - 当前训练轮次（X轴）
    //       dLoss  - 验证集损失值（Y轴）
    void addValPoint(int nEpoch, double dLoss);

    // 20260322 ZJH 清空所有数据点并重置坐标轴
    void clear();

    // 20260322 ZJH 设置是否启用 EMA 指数移动平均平滑
    // 参数: bEnabled - true 启用平滑显示，false 显示原始数据
    void setSmoothing(bool bEnabled);

private:
    // 20260322 ZJH 自动缩放坐标轴范围以适应全部数据点
    void updateAxisRanges();

    // 20260322 ZJH 将原始数据应用 EMA 平滑后更新显示系列
    void applySmoothingToSeries();

    QChart*      m_pChart;        // 20260322 ZJH 图表对象（持有系列和坐标轴）
    QChartView*  m_pChartView;   // 20260322 ZJH 图表视图（嵌入到布局中显示）
    QLineSeries* m_pTrainSeries; // 20260322 ZJH 训练损失折线（蓝色 #2563eb）
    QLineSeries* m_pValSeries;   // 20260322 ZJH 验证损失折线（橙色 #f59e0b）
    QValueAxis*  m_pAxisX;       // 20260322 ZJH X轴：Epoch
    QValueAxis*  m_pAxisY;       // 20260322 ZJH Y轴：Loss

    // 20260322 ZJH 原始数据存储（平滑模式下用于计算 EMA）
    QVector<QPointF> m_vecTrainRaw;  // 20260322 ZJH 训练损失原始数据点
    QVector<QPointF> m_vecValRaw;    // 20260322 ZJH 验证损失原始数据点

    bool m_bSmoothing = false;       // 20260322 ZJH 是否启用 EMA 平滑（默认关闭）
    double m_dSmoothAlpha = 0.3;     // 20260322 ZJH EMA 平滑系数（越小越平滑）
};
