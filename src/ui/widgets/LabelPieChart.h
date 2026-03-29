// 20260323 ZJH LabelPieChart — 标签分布饼图控件
// QWidget 子类，QPainter 自绘饼图，显示各标签占比
// 支持悬停高亮、图例、动画展开
#pragma once

#include <QWidget>      // 20260323 ZJH 控件基类
#include <QVector>      // 20260323 ZJH 动态数组
#include <QColor>       // 20260323 ZJH 颜色
#include <QString>      // 20260323 ZJH 字符串
#include <QMap>         // 20260323 ZJH 有序映射

// 20260323 ZJH 饼图数据项
struct PieSlice
{
    QString strName;    // 20260323 ZJH 标签名称
    int nCount = 0;     // 20260323 ZJH 数量
    QColor color;       // 20260323 ZJH 颜色
};

// 20260323 ZJH 标签分布饼图控件
class LabelPieChart : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit LabelPieChart(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~LabelPieChart() override = default;

    // 20260323 ZJH 设置饼图数据
    // 参数: mapData - 标签名 → 数量 映射
    void setData(const QMap<QString, int>& mapData);

    // 20260323 ZJH 设置饼图数据（带自定义颜色）
    void setSlices(const QVector<PieSlice>& vecSlices);

    // 20260323 ZJH 清空数据
    void clear();

    // 20260323 ZJH 推荐最小尺寸
    QSize minimumSizeHint() const override;

    // 20260324 ZJH 返回控件的推荐尺寸，供布局管理器参考
    QSize sizeHint() const override;

protected:
    // 20260323 ZJH 绘制事件
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260323 ZJH 鼠标移动事件（悬停高亮）
    void mouseMoveEvent(QMouseEvent* pEvent) override;

    // 20260323 ZJH 鼠标离开事件
    void leaveEvent(QEvent* pEvent) override;

private:
    // 20260323 ZJH 计算鼠标位置对应的扇区索引（-1 表示无）
    int sliceAtPos(const QPoint& pt) const;

    QVector<PieSlice> m_vecSlices;  // 20260323 ZJH 扇区数据列表
    int m_nTotal = 0;               // 20260323 ZJH 总数量
    int m_nHoverIndex = -1;         // 20260323 ZJH 悬停高亮的扇区索引（-1=无）

    // 20260323 ZJH 饼图区域缓存（paintEvent 中计算）
    mutable QRectF m_pieRect;

    // 20260323 ZJH 默认颜色循环
    static const QVector<QColor>& defaultColors();
};
