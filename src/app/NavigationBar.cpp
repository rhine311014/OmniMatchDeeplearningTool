// 20260322 ZJH 顶部导航栏实现
// 创建 8 个均分宽度的页签按钮，管理选中状态样式，绘制底部蓝色指示线

#include "NavigationBar.h"
#include <QPushButton>    // 20260322 ZJH 页签按钮控件
#include <QHBoxLayout>    // 20260322 ZJH 水平均分布局
#include <QPainter>       // 20260322 ZJH 在 paintEvent 中绘制指示线
#include <QPaintEvent>    // 20260322 ZJH paintEvent 参数类型

namespace OmniMatch {

// 20260322 ZJH 8 个页签的中文显示名称，顺序与 PageIndex 枚举严格对应
static const QStringList s_arrPageNames = {
    QStringLiteral("项目"),    // PageIndex::Project    = 0
    QStringLiteral("图库"),    // PageIndex::Gallery    = 1
    QStringLiteral("图像"),    // PageIndex::Image      = 2
    QStringLiteral("检查"),    // PageIndex::Inspection = 3
    QStringLiteral("拆分"),    // PageIndex::Split      = 4
    QStringLiteral("训练"),    // PageIndex::Training   = 5
    QStringLiteral("评估"),    // PageIndex::Evaluation = 6
    QStringLiteral("导出")     // PageIndex::Export     = 7
};

// 20260322 ZJH 选中状态颜色常量
static constexpr char s_strColorActive[]   = "#2563eb";  // 选中按钮蓝色背景/指示线颜色
static constexpr char s_strColorInactive[] = "#94a3b8";  // 未选中按钮文字颜色（灰色）
static constexpr int  s_nNavHeight         = 40;         // 导航栏固定高度（像素）
static constexpr int  s_nIndicatorHeight   = 3;          // 底部指示线高度（像素）

// 20260324 ZJH 返回指定索引的页签中文名称，越界返回空 QString
QString NavigationBar::pageName(int nIndex)
{
    // 20260324 ZJH 边界检查，防止索引越界访问列表
    if (nIndex < 0 || nIndex >= s_arrPageNames.size()) {
        return QString();  // 20260324 ZJH 无效索引，返回空字符串
    }
    return s_arrPageNames[nIndex];  // 20260324 ZJH 返回对应的中文页签名称
}

// 20260322 ZJH 构造函数，初始化列表设置默认索引，调用 setupUI 完成界面搭建
NavigationBar::NavigationBar(QWidget* pParent)
    : QWidget(pParent)
    , m_nCurrentIndex(0)   // 默认选中第一个页签（项目页）
{
    // 20260324 ZJH 预构建选中/未选中样式字符串，避免 updateButtonStyles 每次重新拼接
    buildStyleCache();

    setupUI();  // 20260322 ZJH 创建按钮、布局、样式
}

// 20260322 ZJH 初始化导航栏界面：设置尺寸、创建布局、生成 8 个页签按钮
void NavigationBar::setupUI()
{
    // 20260322 ZJH 固定高度 40px，宽度由父容器决定
    setFixedHeight(s_nNavHeight);
    setObjectName("navigationBar");  // 20260322 ZJH 供 QSS 精确选择器使用

    // 20260322 ZJH 水平布局，无边距无间距，使按钮紧密排列并均分宽度
    m_pLayout = new QHBoxLayout(this);
    m_pLayout->setContentsMargins(0, 0, 0, 0);  // 无边距，按钮顶格排列
    m_pLayout->setSpacing(0);                    // 按钮之间无间距

    // 20260322 ZJH 循环创建 8 个页签按钮
    for (int i = 0; i < PageIndex::Count; ++i) {
        QPushButton* pBtn = new QPushButton(s_arrPageNames[i], this);
        pBtn->setObjectName("navButton");           // 20260322 ZJH QSS 选择器名称
        pBtn->setCheckable(false);                  // 20260322 ZJH 样式由代码控制，不用 Qt 自带 checked 状态
        pBtn->setCursor(Qt::PointingHandCursor);    // 20260322 ZJH 鼠标悬停变手型，提示可点击
        pBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);  // 20260322 ZJH 水平扩展，均分宽度
        pBtn->setFixedHeight(s_nNavHeight);         // 20260322 ZJH 按钮高度与导航栏一致

        // 20260322 ZJH 用 lambda 捕获索引 i，点击时检查是否需要切换，避免重复触发
        connect(pBtn, &QPushButton::clicked, this, [this, i]() {
            if (m_nCurrentIndex != i) {    // 20260322 ZJH 只有切换到不同页签才触发
                m_nCurrentIndex = i;       // 20260322 ZJH 更新内部当前索引
                updateButtonStyles();      // 20260322 ZJH 刷新所有按钮视觉样式
                update();                  // 20260322 ZJH 触发 paintEvent 更新指示线位置
                emit pageChanged(i);       // 20260322 ZJH 通知外部（MainWindow）切换页面
            }
        });

        m_arrButtons.append(pBtn);    // 20260322 ZJH 加入按钮列表，便于后续统一操作
        m_pLayout->addWidget(pBtn);   // 20260322 ZJH 加入水平布局，触发等宽分配
    }

    // 20260322 ZJH 设置初始选中样式（第 0 个按钮高亮）
    updateButtonStyles();
}

// 20260322 ZJH 程序化设置当前选中页签（不发射 pageChanged，避免循环触发）
void NavigationBar::setCurrentIndex(int nIndex)
{
    // 20260322 ZJH 边界检查：索引必须在合法范围内
    if (nIndex < 0 || nIndex >= m_arrButtons.size()) {
        return;  // 索引越界，静默忽略
    }

    m_nCurrentIndex = nIndex;   // 20260322 ZJH 记录新的选中索引
    updateButtonStyles();        // 20260322 ZJH 刷新按钮颜色
    update();                    // 20260322 ZJH 重绘以更新指示线位置
}

// 20260322 ZJH 返回当前选中的页签索引
int NavigationBar::currentIndex() const
{
    return m_nCurrentIndex;  // 直接返回成员变量
}

// 20260322 ZJH 设置指定按钮启用/禁用（如未打开项目时禁用后续页签）
void NavigationBar::setPageEnabled(int nIndex, bool bEnabled)
{
    // 20260322 ZJH 边界检查，防止越界访问列表
    if (nIndex >= 0 && nIndex < m_arrButtons.size()) {
        m_arrButtons[nIndex]->setEnabled(bEnabled);  // Qt 自动处理禁用时的视觉置灰
    }
}

// 20260324 ZJH 构建并缓存选中/未选中按钮样式字符串
// 缓存后 updateButtonStyles() 只需直接赋值，无需每次重新拼接 QString
void NavigationBar::buildStyleCache()
{
    // 20260324 ZJH 选中状态：蓝色背景，白色加粗文字，无边框，无圆角（扁平风格）
    m_strActiveStyle = QStringLiteral(
        "QPushButton {"
        "  background-color: %1;"     // 蓝色背景
        "  color: #ffffff;"           // 白色文字
        "  font-weight: 600;"         // 半粗体，视觉权重感
        "  border: none;"             // 无边框
        "  padding: 0px 8px;"         // 左右内边距
        "}"
    ).arg(s_strColorActive);

    // 20260324 ZJH 未选中状态：透明背景，灰色文字，悬停时文字变白
    m_strInactiveStyle = QStringLiteral(
        "QPushButton {"
        "  background-color: transparent;"  // 透明背景，透出导航栏底色
        "  color: %1;"                      // 灰色文字
        "  font-weight: 400;"               // 普通字重
        "  border: none;"
        "  padding: 0px 8px;"
        "}"
        "QPushButton:hover {"
        "  background-color: rgba(37, 99, 235, 0.15);"  // 悬停浅蓝高亮
        "  color: #ffffff;"                              // 悬停文字变白
        "}"
        "QPushButton:disabled {"
        "  color: #3d4452;"  // 禁用状态更深的灰色，明确不可点击
        "}"
    ).arg(s_strColorInactive);
}

// 20260322 ZJH 刷新所有按钮的选中/未选中样式
// 20260324 ZJH 使用预缓存的样式字符串，避免每次页面切换时重复构建 QString
void NavigationBar::updateButtonStyles()
{
    for (int i = 0; i < m_arrButtons.size(); ++i) {
        QPushButton* pBtn = m_arrButtons[i];  // 20260322 ZJH 取当前按钮指针

        if (i == m_nCurrentIndex) {
            pBtn->setStyleSheet(m_strActiveStyle);    // 20260324 ZJH 使用缓存的选中样式
        } else {
            pBtn->setStyleSheet(m_strInactiveStyle);  // 20260324 ZJH 使用缓存的未选中样式
        }
    }
}

// 20260322 ZJH 在导航栏底部绘制当前选中页签的蓝色指示线
// 指示线宽度与对应按钮相同，位于导航栏最底部
void NavigationBar::paintEvent(QPaintEvent* pEvent)
{
    // 20260322 ZJH 先调用基类绘制背景（由 QSS 控制），再叠加指示线
    QWidget::paintEvent(pEvent);

    // 20260322 ZJH 边界检查，防止按钮列表为空时绘制崩溃
    if (m_nCurrentIndex < 0 || m_nCurrentIndex >= m_arrButtons.size()) {
        return;  // 无效索引，跳过绘制
    }

    QPushButton* pActiveBtn = m_arrButtons[m_nCurrentIndex];  // 取当前选中按钮
    if (!pActiveBtn) {
        return;  // 空指针保护
    }

    // 20260322 ZJH 将按钮在自身坐标系中的矩形，映射到导航栏（this）坐标系
    const QRect btnRect = pActiveBtn->geometry();  // 按钮相对于父控件（this）的矩形

    // 20260322 ZJH 计算指示线矩形：与按钮等宽，位于导航栏最底部
    const QRect indicatorRect(
        btnRect.left(),                        // 与按钮左边对齐
        height() - s_nIndicatorHeight,        // 紧贴底部
        btnRect.width(),                       // 与按钮等宽
        s_nIndicatorHeight                     // 固定高度 3px
    );

    QPainter painter(this);  // 20260322 ZJH 在当前 widget 上绘制
    painter.setRenderHint(QPainter::Antialiasing, false);  // 20260322 ZJH 水平线不需要抗锯齿
    painter.fillRect(indicatorRect, QColor(s_strColorActive));  // 20260322 ZJH 填充蓝色指示线
}

}  // namespace OmniMatch
