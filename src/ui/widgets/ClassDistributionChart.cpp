// 20260322 ZJH ClassDistributionChart 实现
// 水平条形图：每个标签一个条形，X 轴为数量，Y 轴为标签名
// 无数据时显示居中占位文字 "暂无数据"

#include "ui/widgets/ClassDistributionChart.h"  // 20260322 ZJH 自身声明

// 20260322 ZJH Qt6 Charts 头文件（Qt6.10.1 中 Charts 类位于全局命名空间，直接 include 即可使用）
#include <QtCharts/QChartView>
#include <QtCharts/QChart>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QValueAxis>

#include <QVBoxLayout>      // 20260322 ZJH 垂直布局，撑满父控件
#include <QLabel>           // 20260322 ZJH 无数据占位文字
#include <QStackedWidget>   // 20260322 ZJH 图表/占位 视图切换

// 20260322 ZJH 条形图颜色预设（循环使用，覆盖多类别场景）
static const QColor s_arrBarColors[] = {
    QColor(0x26, 0x63, 0xeb),  // 20260322 ZJH 蓝色  — 默认主色
    QColor(0xf5, 0x9e, 0x0b),  // 20260322 ZJH 橙色
    QColor(0x10, 0xb9, 0x81),  // 20260322 ZJH 青绿色
    QColor(0xef, 0x44, 0x44),  // 20260322 ZJH 红色
    QColor(0x8b, 0x5c, 0xf6),  // 20260322 ZJH 紫色
    QColor(0xec, 0x48, 0x99),  // 20260322 ZJH 粉色
    QColor(0x06, 0xb6, 0xd4),  // 20260322 ZJH 青色
    QColor(0x84, 0xcc, 0x16),  // 20260322 ZJH 黄绿色
};
static constexpr int s_nColorCount = static_cast<int>(sizeof(s_arrBarColors) / sizeof(s_arrBarColors[0]));  // 20260322 ZJH 颜色总数

// ============================================================================
// 20260322 ZJH 构造函数
// ============================================================================
ClassDistributionChart::ClassDistributionChart(QWidget* pParent)
    : QWidget(pParent)
    , m_pStack(nullptr)
    , m_pChartView(nullptr)
    , m_pChart(nullptr)
    , m_pBarSeries(nullptr)
    , m_pAxisY(nullptr)
    , m_pAxisX(nullptr)
    , m_pPlaceholder(nullptr)
{
    // 20260322 ZJH 主布局：撑满父控件，无边距
    QVBoxLayout* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(0, 0, 0, 0);
    pMainLayout->setSpacing(0);

    // 20260322 ZJH 创建图表/占位切换容器
    m_pStack = new QStackedWidget(this);
    pMainLayout->addWidget(m_pStack);

    // 20260322 ZJH 初始化图表组件
    setupChart();

    // 20260322 ZJH 创建占位文字标签
    m_pPlaceholder = new QLabel(QStringLiteral("暂无数据"), this);
    m_pPlaceholder->setAlignment(Qt::AlignCenter);  // 20260322 ZJH 居中显示
    m_pPlaceholder->setStyleSheet(QStringLiteral(
        "QLabel {"
        "  color: #475569;"           // 20260322 ZJH 暗灰色文字，与暗色主题协调
        "  font-size: 12pt;"
        "  background: transparent;"
        "  border: none;"
        "}"
    ));

    // 20260322 ZJH 将图表视图和占位标签加入切换容器
    m_pStack->addWidget(m_pChartView);    // 20260322 ZJH 索引 0 — 图表视图
    m_pStack->addWidget(m_pPlaceholder); // 20260322 ZJH 索引 1 — 占位文字

    // 20260322 ZJH 初始状态显示占位文字
    showPlaceholder();
}

// ============================================================================
// 20260322 ZJH 初始化图表组件
// 创建 QChart、QBarSeries（水平条形使用 QBarSet 单条实现）、XY 轴并组装
// ============================================================================
void ClassDistributionChart::setupChart()
{
    // 20260322 ZJH 创建图表对象
    m_pChart = new QChart();
    m_pChart->setBackgroundBrush(QColor(0x1e, 0x22, 0x30));  // 20260322 ZJH 深色背景，与主题统一
    m_pChart->setBackgroundRoundness(0);                      // 20260322 ZJH 无圆角，紧凑
    m_pChart->legend()->hide();                               // 20260322 ZJH 隐藏图例，节省空间
    m_pChart->setMargins(QMargins(4, 4, 4, 4));              // 20260322 ZJH 最小内边距，紧凑布局
    m_pChart->setTitle(QString());                            // 20260322 ZJH 无标题，由外部 GroupBox 代替

    // 20260322 ZJH 创建条形图系列（后续动态添加 QBarSet）
    m_pBarSeries = new QBarSeries();
    m_pChart->addSeries(m_pBarSeries);

    // 20260322 ZJH 创建分类轴（Y 轴 — 标签名称，显示在左侧作为水平条形的行标签）
    m_pAxisY = new QBarCategoryAxis();
    m_pAxisY->setLabelsColor(QColor(0xe2, 0xe8, 0xf0));  // 20260322 ZJH 浅色标签文字
    m_pAxisY->setGridLineVisible(false);                  // 20260322 ZJH 不显示网格线
    m_pAxisY->setLinePenColor(QColor(0x2a, 0x2d, 0x35)); // 20260322 ZJH 轴线颜色
    m_pChart->addAxis(m_pAxisY, Qt::AlignLeft);           // 20260322 ZJH Y 轴挂到图表左侧
    m_pBarSeries->attachAxis(m_pAxisY);                   // 20260322 ZJH 系列绑定 Y 轴

    // 20260322 ZJH 创建数值轴（X 轴 — 图像数量，显示在底部）
    m_pAxisX = new QValueAxis();
    m_pAxisX->setLabelFormat(QStringLiteral("%d"));       // 20260322 ZJH 整数格式
    m_pAxisX->setLabelsColor(QColor(0xe2, 0xe8, 0xf0));  // 20260322 ZJH 浅色文字
    m_pAxisX->setGridLineColor(QColor(0x2a, 0x2d, 0x35));// 20260322 ZJH 网格线暗色
    m_pAxisX->setLinePenColor(QColor(0x2a, 0x2d, 0x35)); // 20260322 ZJH 轴线颜色
    m_pAxisX->setMin(0);                                  // 20260322 ZJH 数量最小为 0
    m_pChart->addAxis(m_pAxisX, Qt::AlignBottom);         // 20260322 ZJH X 轴挂到图表底部
    m_pBarSeries->attachAxis(m_pAxisX);                   // 20260322 ZJH 系列绑定 X 轴

    // 20260322 ZJH 创建图表视图（不带缩放/滚动，保持紧凑）
    m_pChartView = new QChartView(m_pChart);
    m_pChartView->setRenderHint(QPainter::Antialiasing);  // 20260322 ZJH 抗锯齿渲染
    m_pChartView->setBackgroundBrush(QColor(0x1e, 0x22, 0x30));  // 20260322 ZJH 同图表背景色
    m_pChartView->setFrameShape(QFrame::NoFrame);         // 20260322 ZJH 无边框
    m_pChartView->setMinimumHeight(120);                  // 20260322 ZJH 最小高度保证可读性
}

// ============================================================================
// 20260322 ZJH 刷新条形图数据
// 参数: mapData — 标签名 -> 图像数量映射（key: 标签名, value: 数量）
// ============================================================================
void ClassDistributionChart::updateData(const QMap<QString, int>& mapData)
{
    // 20260322 ZJH 空数据时直接显示占位文字
    if (mapData.isEmpty()) {
        showPlaceholder();
        return;
    }

    // 20260322 ZJH 清空旧系列数据
    m_pBarSeries->clear();

    // 20260322 ZJH 清空分类轴的旧标签
    m_pAxisY->clear();

    // 20260322 ZJH 收集标签名列表（QMap 默认按 key 字母序排列）
    QStringList listLabels;
    int nMaxCount = 0;  // 20260322 ZJH 记录最大数量，用于设置 X 轴范围
    for (auto it = mapData.constBegin(); it != mapData.constEnd(); ++it) {
        listLabels.append(it.key());                 // 20260322 ZJH 收集标签名
        nMaxCount = qMax(nMaxCount, it.value());     // 20260322 ZJH 更新最大值
    }

    // 20260322 ZJH 为每个标签创建独立的 QBarSet，实现多色条形
    int nColorIdx = 0;  // 20260322 ZJH 颜色轮换索引
    for (const QString& strLabel : listLabels) {
        int nCount = mapData.value(strLabel, 0);  // 20260322 ZJH 获取该标签的图像数量

        // 20260322 ZJH 每个标签创建一个独立 QBarSet，设置不同颜色
        QBarSet* pBarSet = new QBarSet(strLabel);
        QColor clrBar = s_arrBarColors[nColorIdx % s_nColorCount];  // 20260322 ZJH 循环取颜色
        pBarSet->setColor(clrBar);                   // 20260322 ZJH 填充色
        pBarSet->setBorderColor(clrBar.darker(120)); // 20260322 ZJH 边框色略深
        pBarSet->setLabelColor(QColor(0xe2, 0xe8, 0xf0));  // 20260322 ZJH 标签文字浅色
        *pBarSet << nCount;  // 20260322 ZJH 填入数量值（每个 barSet 只有一个值）

        m_pBarSeries->append(pBarSet);  // 20260322 ZJH 将 barSet 加入系列
        ++nColorIdx;                    // 20260322 ZJH 切换下一个颜色
    }

    // 20260322 ZJH 设置 Y 轴分类标签（各标签名）
    m_pAxisY->setCategories(listLabels);

    // 20260322 ZJH 设置 X 轴范围（0 到最大数量，留 10% 余量）
    m_pAxisX->setRange(0, nMaxCount > 0 ? static_cast<double>(nMaxCount) * 1.1 : 10.0);

    // 20260322 ZJH 切换到图表视图显示
    showChart();
}

// ============================================================================
// 20260322 ZJH 清空图表，显示占位文字
// ============================================================================
void ClassDistributionChart::clear()
{
    // 20260322 ZJH 清空系列和轴数据
    m_pBarSeries->clear();
    m_pAxisY->clear();

    // 20260322 ZJH 显示占位文字
    showPlaceholder();
}

// ============================================================================
// 20260322 ZJH 私有辅助 — 切换到图表视图
// ============================================================================
void ClassDistributionChart::showChart()
{
    m_pStack->setCurrentWidget(m_pChartView);  // 20260322 ZJH 显示图表（索引 0）
}

// ============================================================================
// 20260322 ZJH 私有辅助 — 切换到占位文字视图
// ============================================================================
void ClassDistributionChart::showPlaceholder()
{
    m_pStack->setCurrentWidget(m_pPlaceholder);  // 20260322 ZJH 显示占位文字（索引 1）
}
