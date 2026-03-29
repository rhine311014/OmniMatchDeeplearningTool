// 20260322 ZJH BasePage 实现
// 提供生命周期回调的默认空实现和三栏布局辅助方法

#include "ui/pages/BasePage.h"  // 20260322 ZJH BasePage 类声明

#include <QVBoxLayout>   // 20260322 ZJH 页面根布局
#include <QSplitter>     // 20260322 ZJH 三栏可拖拽分割器
#include <QList>         // 20260322 ZJH QSplitter::setSizes 参数类型

// 20260322 ZJH 构造函数，初始化基类和默认面板宽度
// 参数 pParent: 父控件指针，传入 QStackedWidget 时自动管理生命周期
BasePage::BasePage(QWidget* pParent)
    : QWidget(pParent)          // 20260322 ZJH 初始化 QWidget 基类
    , m_pProject(nullptr)       // 20260322 ZJH 初始时无项目加载
    , m_pSplitter(nullptr)      // 20260322 ZJH 分割器延迟到 setupThreeColumnLayout 创建
    , m_nLeftWidth(280)         // 20260322 ZJH 左侧面板默认宽度 280px
    , m_nRightWidth(250)        // 20260322 ZJH 右侧面板默认宽度 250px
{
}

// ===== 生命周期回调默认空实现 =====

// 20260322 ZJH 页面切换到前台时调用，基类空实现，子类按需重写
void BasePage::onEnter()
{
    // 20260322 ZJH 默认空操作，子类可在此刷新数据/恢复动画等
}

// 20260322 ZJH 页面离开前台时调用，基类空实现，子类按需重写
void BasePage::onLeave()
{
    // 20260322 ZJH 默认空操作，子类可在此暂停后台任务/保存草稿等
}

// 20260324 ZJH 项目加载 — Template Method 模式（final，子类不可重写）
// 先完成基类簿记（保存项目指针），再调用子类扩展钩子
void BasePage::onProjectLoaded(Project* pProject)
{
    m_pProject = pProject;    // 20260324 ZJH 保存项目指针（弱引用，不持有所有权）
    onProjectLoadedImpl();    // 20260324 ZJH 调用子类扩展点
}

// 20260324 ZJH 项目关闭 — Template Method 模式（final，子类不可重写）
// 先调用子类扩展钩子（此时 m_pProject 仍有效），再清空指针
void BasePage::onProjectClosed()
{
    onProjectClosedImpl();    // 20260324 ZJH 先调用子类扩展点（可能需要 m_pProject）
    m_pProject = nullptr;     // 20260324 ZJH 清空项目指针
}

// 20260324 ZJH 项目加载扩展点默认空实现，子类按需重写
void BasePage::onProjectLoadedImpl()
{
    // 20260324 ZJH 默认空操作
}

// 20260324 ZJH 项目关闭扩展点默认空实现，子类按需重写
void BasePage::onProjectClosedImpl()
{
    // 20260324 ZJH 默认空操作
}

// ===== 三栏布局辅助 =====

// 20260322 ZJH 设置三栏可拖拽布局
// 内部创建 QSplitter 并根据传入的面板控件配置初始大小
void BasePage::setupThreeColumnLayout(QWidget* pLeft, QWidget* pCenter, QWidget* pRight)
{
    // 20260322 ZJH 创建页面根布局（垂直），无边距使内容充满整个页面区域
    QVBoxLayout* pLayout = new QVBoxLayout(this);
    pLayout->setContentsMargins(0, 0, 0, 0);  // 20260322 ZJH 页面无边距，最大化利用空间
    pLayout->setSpacing(0);                     // 20260322 ZJH 无间距

    // 20260322 ZJH 创建水平分割器，允许用户拖拽调整面板宽度
    m_pSplitter = new QSplitter(Qt::Horizontal, this);
    m_pSplitter->setHandleWidth(1);  // 20260322 ZJH 分割线宽度 1px，极简风格
    m_pSplitter->setChildrenCollapsible(false);  // 20260322 ZJH 禁止拖拽折叠面板

    // 20260322 ZJH 添加左侧面板
    m_pSplitter->addWidget(pLeft);

    // 20260322 ZJH 添加中央工作区
    m_pSplitter->addWidget(pCenter);

    // 20260322 ZJH 准备初始大小列表
    QList<int> arrSizes;

    if (pRight) {
        // 20260322 ZJH 三栏模式：左 + 中央（自适应占剩余空间） + 右
        m_pSplitter->addWidget(pRight);
        arrSizes << m_nLeftWidth << 1000 << m_nRightWidth;
    } else {
        // 20260322 ZJH 两栏模式：左 + 中央（占满剩余空间）
        arrSizes << m_nLeftWidth << 1000;
    }

    // 20260322 ZJH 设置各面板的初始宽度比例
    m_pSplitter->setSizes(arrSizes);

    // 20260325 ZJH 三栏等比缩放：所有面板都设 stretch factor，窗口缩放时按比例分配空间
    // 比例 ≈ left:center:right = 6:14:5（基于设计宽度 300:700:250）
    m_pSplitter->setStretchFactor(0, 6);   // 20260325 ZJH 左面板占比 6（~24%）
    m_pSplitter->setStretchFactor(1, 14);  // 20260325 ZJH 中央区域占比 14（~56%）
    if (pRight) {
        m_pSplitter->setStretchFactor(2, 5);  // 20260325 ZJH 右面板占比 5（~20%）
    }

    // 20260322 ZJH 将分割器添加到页面布局中
    pLayout->addWidget(m_pSplitter);
}

// 20260322 ZJH 更新左侧面板宽度
// 仅在分割器已创建后生效，否则仅记录数值供后续 setup 使用
void BasePage::setLeftPanelWidth(int nWidth)
{
    m_nLeftWidth = nWidth;  // 20260322 ZJH 记录左侧面板目标宽度

    // 20260322 ZJH 如果分割器已创建，立即更新大小
    if (m_pSplitter && m_pSplitter->count() >= 2) {
        QList<int> arrSizes = m_pSplitter->sizes();  // 20260322 ZJH 获取当前各面板大小
        arrSizes[0] = nWidth;  // 20260322 ZJH 更新左侧面板宽度
        m_pSplitter->setSizes(arrSizes);  // 20260322 ZJH 应用新大小
    }
}

// 20260322 ZJH 更新右侧面板宽度
// 仅在三栏模式下（分割器有 3 个子控件）生效
void BasePage::setRightPanelWidth(int nWidth)
{
    m_nRightWidth = nWidth;  // 20260322 ZJH 记录右侧面板目标宽度

    // 20260322 ZJH 如果分割器已创建且存在右侧面板（3 个子控件），立即更新
    if (m_pSplitter && m_pSplitter->count() >= 3) {
        QList<int> arrSizes = m_pSplitter->sizes();  // 20260322 ZJH 获取当前各面板大小
        arrSizes[2] = nWidth;  // 20260322 ZJH 更新右侧面板宽度（索引 2）
        m_pSplitter->setSizes(arrSizes);  // 20260322 ZJH 应用新大小
    }
}
