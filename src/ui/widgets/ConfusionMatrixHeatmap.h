// 20260322 ZJH ConfusionMatrixHeatmap — 混淆矩阵热力图控件
// 使用 QPainter 自绘混淆矩阵，不依赖 Qt Charts
// 支持三种归一化模式（计数/行归一化/列归一化）、鼠标悬停高亮行列、
// 蓝→红颜色渐变映射

#pragma once

#include <QWidget>       // 20260322 ZJH 基础控件基类
#include <QVector>       // 20260322 ZJH 矩阵数据存储
#include <QStringList>   // 20260322 ZJH 类别名称列表
#include <QPoint>        // 20260322 ZJH 鼠标悬停单元格索引

// 20260322 ZJH 归一化模式枚举
// 控制混淆矩阵中每格显示的数值类型
enum class NormMode : int
{
    Count      = 0,  // 20260322 ZJH 显示原始计数值
    RowNorm    = 1,  // 20260322 ZJH 按行归一化（每行和为 1.0，表示真实类别的预测分布）
    ColumnNorm = 2   // 20260322 ZJH 按列归一化（每列和为 1.0，表示预测类别的来源分布）
};

// 20260322 ZJH 混淆矩阵热力图控件
// 接收 NxN 整数矩阵和 N 个类别名称，用 QPainter 自绘热力图
// 颜色映射：0 值 → 深蓝 #0d1b2a，高值 → 红色 #e63946
class ConfusionMatrixHeatmap : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父控件指针
    explicit ConfusionMatrixHeatmap(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构
    ~ConfusionMatrixHeatmap() override = default;

    // 20260322 ZJH 设置混淆矩阵数据和类别名称
    // 参数: matData - NxN 整数矩阵（行=真实类别，列=预测类别）
    //       vecClassNames - N 个类别名称
    void setData(const QVector<QVector<int>>& matData, const QStringList& vecClassNames);

    // 20260322 ZJH 设置归一化模式
    // 参数: nMode - 0=计数, 1=行归一化, 2=列归一化
    void setNormMode(int nMode);

    // 20260322 ZJH 清空数据
    void clear();

    // 20260322 ZJH 推荐最小尺寸
    QSize minimumSizeHint() const override;

protected:
    // 20260322 ZJH 自绘混淆矩阵热力图
    void paintEvent(QPaintEvent* pEvent) override;

    // 20260322 ZJH 鼠标移动事件：高亮悬停行列
    void mouseMoveEvent(QMouseEvent* pEvent) override;

    // 20260322 ZJH 鼠标离开事件：取消高亮
    void leaveEvent(QEvent* pEvent) override;

private:
    // 20260322 ZJH 根据归一化值 [0, 1] 返回颜色（深蓝→红色渐变）
    // 参数: dValue - 归一化后的值（0.0 ~ 1.0）
    // 返回: 对应的渐变颜色
    QColor valueToColor(double dValue) const;

    // 20260322 ZJH 计算归一化后的矩阵（根据当前归一化模式）
    // 返回: 归一化后的浮点矩阵
    QVector<QVector<double>> normalizedMatrix() const;

    // 20260322 ZJH 根据鼠标位置计算所在单元格的行列索引
    // 参数: ptPos - 鼠标坐标（控件内坐标）
    // 返回: QPoint(列, 行)，无效区域返回 (-1, -1)
    QPoint cellAtPos(const QPoint& ptPos) const;

    QVector<QVector<int>> m_matData;   // 20260322 ZJH 原始整数矩阵数据
    QStringList m_vecClassNames;       // 20260322 ZJH 类别名称列表
    NormMode m_eNormMode = NormMode::Count;  // 20260322 ZJH 当前归一化模式

    // 20260323 ZJH 缓存归一化矩阵，避免每帧 paintEvent 重算 O(N^2)
    mutable QVector<QVector<double>> m_matNormCache;
    mutable bool m_bNormCacheDirty = true;

    QPoint m_ptHoverCell;  // 20260322 ZJH 鼠标悬停的单元格索引 (col, row)，(-1,-1) 表示无悬停

    // 20260322 ZJH 布局常量
    static constexpr int s_nLabelMargin = 80;  // 20260322 ZJH 左侧和顶部类别标签预留空间（像素）
    static constexpr int s_nPadding     = 10;  // 20260322 ZJH 控件边距（像素）
};
