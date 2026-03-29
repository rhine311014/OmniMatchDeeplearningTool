// 20260323 ZJH ConfidenceHistogramChart — 置信度分布直方图控件
// QWidget 子类，QPainter 自绘直方图
// 显示模型预测置信度的分布情况，用于评估模型校准性
#pragma once

#include <QWidget>   // 20260323 ZJH 控件基类
#include <QVector>   // 20260323 ZJH 动态数组
#include <QColor>    // 20260323 ZJH 颜色

// 20260323 ZJH 置信度分布直方图控件
class ConfidenceHistogramChart : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit ConfidenceHistogramChart(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~ConfidenceHistogramChart() override = default;

    // 20260323 ZJH 设置置信度数据
    // 参数: vecConfidences - 所有预测的置信度值 [0, 1]
    //       nBins - 直方图箱数（默认 20）
    void setData(const QVector<double>& vecConfidences, int nBins = 20);

    // 20260323 ZJH 设置正确/错误预测的置信度数据（双色直方图）
    void setDataWithCorrectness(
        const QVector<double>& vecCorrect,
        const QVector<double>& vecIncorrect,
        int nBins = 20);

    // 20260323 ZJH 清空数据
    void clear();

    // 20260323 ZJH 设置柱体颜色
    void setBarColor(const QColor& color);

    // 20260323 ZJH 推荐最小尺寸
    QSize minimumSizeHint() const override;

    // 20260324 ZJH 返回控件的推荐尺寸，供布局管理器参考
    QSize sizeHint() const override;

protected:
    // 20260323 ZJH 绘制事件
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260323 ZJH 鼠标移动事件（悬停提示）
    void mouseMoveEvent(QMouseEvent* pEvent) override;

    // 20260323 ZJH 鼠标离开事件
    void leaveEvent(QEvent* pEvent) override;

private:
    // 20260323 ZJH 将置信度值分箱
    void computeBins(const QVector<double>& vecValues, int nBins,
                     QVector<int>& vecBinCounts) const;

    QVector<int> m_vecBinCounts;         // 20260323 ZJH 总箱体计数
    QVector<int> m_vecCorrectCounts;     // 20260323 ZJH 正确预测箱体计数
    QVector<int> m_vecIncorrectCounts;   // 20260323 ZJH 错误预测箱体计数
    bool m_bDualColor = false;           // 20260323 ZJH 是否双色模式
    int m_nBins = 20;                    // 20260323 ZJH 箱数
    int m_nMaxCount = 0;                 // 20260323 ZJH 最大箱体计数（用于 Y 轴缩放）

    QColor m_colorBar = QColor("#3b82f6");       // 20260323 ZJH 单色模式柱体颜色
    QColor m_colorCorrect = QColor("#10b981");   // 20260323 ZJH 正确预测颜色（绿）
    QColor m_colorIncorrect = QColor("#ef4444"); // 20260323 ZJH 错误预测颜色（红）

    // 20260323 ZJH 绘图区域边距
    int m_nMarginLeft = 50;
    int m_nMarginRight = 15;
    int m_nMarginTop = 30;
    int m_nMarginBottom = 35;

    int m_nHoverBin = -1;  // 20260323 ZJH 悬停箱体索引（-1=无）
};
