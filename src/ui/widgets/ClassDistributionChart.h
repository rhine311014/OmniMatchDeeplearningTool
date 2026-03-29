// 20260322 ZJH ClassDistributionChart — 标签类别分布水平条形图控件
// 基于 Qt Charts 绘制各类别在数据集中的图像数量分布
// 支持动态刷新数据、无数据时显示占位文字
// Qt6 中 Charts 类位于全局命名空间（通过 QT_BEGIN/END_NAMESPACE 包裹）

#pragma once

#include <QWidget>   // 20260322 ZJH QWidget 基类
#include <QMap>      // 20260322 ZJH 标签名->数量映射

// 20260322 ZJH Qt6 Charts 类前向声明（Qt6 中 Charts 类位于全局命名空间）
class QChartView;
class QChart;
class QBarSeries;
class QBarCategoryAxis;
class QValueAxis;
class QLabel;
class QStackedWidget;

// 20260322 ZJH 类别分布水平条形图控件
// 使用方式：
//   ClassDistributionChart* pChart = new ClassDistributionChart(this);
//   pChart->updateData({{"cat", 50}, {"dog", 30}});
class ClassDistributionChart : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化图表布局
    // 参数: pParent - 父控件指针
    explicit ClassDistributionChart(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构，Qt 对象树管理子控件
    ~ClassDistributionChart() override = default;

public slots:
    // 20260322 ZJH 刷新条形图数据
    // 参数: mapData - 标签名 -> 图像数量映射表
    //       mapData 为空时显示占位文字 "暂无数据"
    void updateData(const QMap<QString, int>& mapData);

    // 20260322 ZJH 清空图表，显示占位文字
    void clear();

private:
    // 20260322 ZJH 初始化图表组件（QChart/QBarSeries/坐标轴/视图）
    void setupChart();

    // 20260322 ZJH 切换到图表视图（有数据时调用）
    void showChart();

    // 20260322 ZJH 切换到占位文字视图（无数据时调用）
    void showPlaceholder();

    // ===== 成员变量 =====

    QStackedWidget*     m_pStack;        // 20260322 ZJH 图表与占位文字切换容器
    QChartView*         m_pChartView;    // 20260322 ZJH Qt Charts 图表视图
    QChart*             m_pChart;        // 20260322 ZJH 图表对象
    QBarSeries*         m_pBarSeries;    // 20260322 ZJH 条形图系列
    QBarCategoryAxis*   m_pAxisY;        // 20260322 ZJH Y 轴（标签名称分类轴）
    QValueAxis*         m_pAxisX;        // 20260322 ZJH X 轴（图像数量数值轴）
    QLabel*             m_pPlaceholder;  // 20260322 ZJH 无数据时的占位文字标签
};
