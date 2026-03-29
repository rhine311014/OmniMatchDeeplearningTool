// 20260322 ZJH BasePage — 所有工作流页面的基类
// 提供统一的生命周期回调（onEnter/onLeave/onProjectLoaded/onProjectClosed）
// 以及三栏布局辅助方法，子类重写回调即可实现特定页面逻辑

#pragma once

#include <QWidget>       // 20260322 ZJH 基础窗口控件基类
#include <QVBoxLayout>   // 20260322 ZJH 垂直布局（预留子类使用）
#include <QSplitter>     // 20260322 ZJH 三栏可拖拽分割器

// 20260322 ZJH 前向声明 Project 类，避免头文件循环依赖
class Project;

// 20260322 ZJH 页面基类，所有 8 个工作流页面均继承自此类
// 基类提供：
//   1. 生命周期回调（页面切换、项目加载/关闭）
//   2. 三栏布局辅助（左侧面板 + 中央工作区 + 可选右侧面板）
//   3. 持有当前项目指针（不拥有所有权，由 Application 管理）
class BasePage : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，设置默认面板宽度
    explicit BasePage(QWidget* pParent = nullptr);

    // 20260322 ZJH 虚析构，确保子类析构正确调用
    virtual ~BasePage() = default;

    // ===== 生命周期回调（子类重写） =====

    // 20260322 ZJH 页面切换到前台时调用（如初始化/刷新显示内容）
    virtual void onEnter();

    // 20260322 ZJH 页面离开前台时调用（如暂停后台任务、保存临时状态）
    virtual void onLeave();

    // 20260324 ZJH 项目加载后调用 — Template Method 模式
    // 基类负责保存 m_pProject 指针，然后调用 onProjectLoadedImpl() 供子类扩展
    // 声明为 final，子类不可重写，改为重写 onProjectLoadedImpl()
    virtual void onProjectLoaded(Project* pProject) final;

    // 20260324 ZJH 项目关闭时调用 — Template Method 模式
    // 基类负责清空 m_pProject 指针，然后调用 onProjectClosedImpl() 供子类扩展
    // 声明为 final，子类不可重写，改为重写 onProjectClosedImpl()
    virtual void onProjectClosed() final;

protected:
    // ===== 生命周期扩展钩子（子类重写） =====

    // 20260324 ZJH 项目加载后的子类扩展点，基类已完成 m_pProject 赋值
    // 子类在此读取项目数据并填充 UI，无需手动调用基类版本
    virtual void onProjectLoadedImpl();

    // 20260324 ZJH 项目关闭时的子类扩展点，此时 m_pProject 仍有效
    // 子类在此清空与项目相关的 UI 内容，无需手动调用基类版本
    // 返回后基类自动将 m_pProject 置空
    virtual void onProjectClosedImpl();

    // ===== 三栏布局辅助 =====

    // 20260322 ZJH 设置三栏可拖拽布局：左侧面板 + 中央工作区 + 可选右侧面板
    // 参数 pLeft: 左侧面板控件（如属性面板/文件树）
    // 参数 pCenter: 中央主工作区控件（如图像画布/图表区域）
    // 参数 pRight: 右侧面板控件（如属性检查器），传 nullptr 则不创建右栏
    void setupThreeColumnLayout(QWidget* pLeft, QWidget* pCenter, QWidget* pRight = nullptr);

    // 20260322 ZJH 设置左侧面板初始宽度（像素）
    void setLeftPanelWidth(int nWidth);

    // 20260322 ZJH 设置右侧面板初始宽度（像素）
    void setRightPanelWidth(int nWidth);

    // 20260322 ZJH 当前项目指针（弱引用，不持有所有权，由 Application 管理生命周期）
    Project* m_pProject = nullptr;

private:
    // 20260322 ZJH 三栏分割器，由 setupThreeColumnLayout() 创建
    QSplitter* m_pSplitter = nullptr;

    // 20260322 ZJH 左侧面板初始宽度（默认 280px）
    int m_nLeftWidth = 280;

    // 20260322 ZJH 右侧面板初始宽度（默认 250px）
    int m_nRightWidth = 250;
};
