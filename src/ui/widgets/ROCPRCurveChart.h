// 20260323 ZJH ROCPRCurveChart — ROC/PR 曲线图表控件
// QWidget 子类，QPainter 自绘 ROC 曲线和 PR 曲线
// 支持多类别叠加显示、AUC 标注、交互悬停
#pragma once

#include <QWidget>   // 20260323 ZJH 控件基类
#include <QVector>   // 20260323 ZJH 动态数组
#include <QPair>     // 20260323 ZJH 点对
#include <QString>   // 20260323 ZJH 字符串
#include <QColor>    // 20260323 ZJH 颜色

// 20260323 ZJH 单条曲线数据
struct CurveData
{
    QString strClassName;                         // 20260323 ZJH 类别名称
    QVector<QPair<double, double>> vecPoints;     // 20260323 ZJH 曲线数据点 (x, y)
    double dAUC = 0.0;                            // 20260323 ZJH 曲线下面积
    QColor color;                                 // 20260323 ZJH 曲线颜色
};

// 20260323 ZJH 曲线类型枚举
enum class CurveType
{
    ROC = 0,  // 20260323 ZJH ROC 曲线（FPR vs TPR）
    PR        // 20260323 ZJH PR 曲线（Recall vs Precision）
};

// 20260323 ZJH ROC/PR 曲线图表控件
class ROCPRCurveChart : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit ROCPRCurveChart(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~ROCPRCurveChart() override = default;

    // 20260323 ZJH 设置曲线类型（ROC 或 PR）
    void setCurveType(CurveType type);

    // 20260323 ZJH 添加一条曲线
    void addCurve(const CurveData& curve);

    // 20260323 ZJH 清空所有曲线
    void clear();

    // 20260323 ZJH 设置是否显示 AUC 图例
    void setShowLegend(bool bShow);

    // 20260323 ZJH 获取推荐的最小尺寸
    QSize minimumSizeHint() const override;

    // 20260324 ZJH 返回控件的推荐尺寸，供布局管理器参考
    QSize sizeHint() const override;

protected:
    // 20260323 ZJH 绘制事件：绘制坐标轴、网格、曲线、图例
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260323 ZJH 鼠标移动事件：悬停高亮
    void mouseMoveEvent(QMouseEvent* pEvent) override;

private:
    // 20260323 ZJH 将数据坐标 (0~1, 0~1) 转换为绘图区域像素坐标
    QPointF dataToPixel(double dX, double dY) const;

    // 20260323 ZJH 绘制坐标轴和网格线
    void drawAxes(QPainter& painter) const;

    // 20260323 ZJH 绘制所有曲线
    void drawCurves(QPainter& painter) const;

    // 20260323 ZJH 绘制图例
    void drawLegend(QPainter& painter) const;

    // 20260323 ZJH 绘制对角参考线（ROC: 随机分类器线）
    void drawReferenceLine(QPainter& painter) const;

    CurveType m_curveType = CurveType::ROC;   // 20260323 ZJH 当前曲线类型
    QVector<CurveData> m_vecCurves;           // 20260323 ZJH 所有曲线数据
    bool m_bShowLegend = true;                // 20260323 ZJH 是否显示图例

    // 20260323 ZJH 绘图区域边距（像素）
    int m_nMarginLeft = 55;     // 20260323 ZJH 左边距（Y 轴标签空间）
    int m_nMarginRight = 20;    // 20260323 ZJH 右边距
    int m_nMarginTop = 30;      // 20260323 ZJH 上边距（标题空间）
    int m_nMarginBottom = 40;   // 20260323 ZJH 下边距（X 轴标签空间）

    // 20260323 ZJH 绘图区域缓存（paintEvent 中计算）
    mutable QRectF m_plotRect;

    // 20260323 ZJH 预设曲线颜色循环
    static const QVector<QColor>& defaultColors();
};
