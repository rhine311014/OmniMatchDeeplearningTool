#pragma once
// 20260322 ZJH 顶部导航栏组件 — OmniMatch 工作流页签
// 包含 8 个页签按钮：项目/图库/图像/检查/拆分/训练/评估/导出
// 选中按钮蓝色背景 #2563eb，未选中透明背景灰色文字 #94a3b8
// 高度固定 40px，按钮宽度均分，底部绘制蓝色指示线

#include <QWidget>
#include <QList>

// 20260322 ZJH 前向声明，减少头文件依赖层级
class QPushButton;
class QHBoxLayout;

namespace OmniMatch {

// 20260322 ZJH 顶部水平导航栏，承载 8 个工作流页签按钮
// 通过 pageChanged(int) 信号通知 MainWindow 切换中央页面堆叠
class NavigationBar : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 页面索引常量，与按钮顺序一一对应
    enum PageIndex {
        Project    = 0,  // 项目管理页
        Gallery    = 1,  // 图库浏览页
        Image      = 2,  // 图像预处理页
        Inspection = 3,  // 数据检查页
        Split      = 4,  // 数据集拆分页
        Training   = 5,  // 模型训练页
        Evaluation = 6,  // 模型评估页
        Export     = 7,  // 模型导出页
        Count      = 8   // 页签总数，用于循环边界
    };

    // 20260324 ZJH 页签总数常量，供外部引用（消除魔数 8）
    static constexpr int kPageCount = PageIndex::Count;

    // 20260324 ZJH 获取指定索引的页签中文名称
    // 参数 nIndex: 页面索引（0 ~ kPageCount-1），越界返回空字符串
    static QString pageName(int nIndex);

    // 20260322 ZJH 构造函数，创建全部按钮并初始化布局
    explicit NavigationBar(QWidget* pParent = nullptr);

    // 20260324 ZJH 析构函数，Qt 对象树自动管理子对象内存，使用 default
    ~NavigationBar() override = default;

    // 20260322 ZJH 程序化切换当前选中页签（不触发 pageChanged 信号）
    // 参数 nIndex: 目标页面索引，越界时忽略
    void setCurrentIndex(int nIndex);

    // 20260322 ZJH 返回当前高亮的页签索引
    int currentIndex() const;

    // 20260322 ZJH 控制指定页签按钮的可用状态
    // 参数 nIndex: 目标页面索引；bEnabled: true=可点击，false=置灰
    void setPageEnabled(int nIndex, bool bEnabled);

signals:
    // 20260322 ZJH 用户主动点击页签时发射，携带新页面索引
    // 监听者（通常是 MainWindow）据此切换 QStackedWidget 当前页
    void pageChanged(int nIndex);

protected:
    // 20260322 ZJH 重写 paintEvent，在导航栏底部绘制当前页签的蓝色指示线
    void paintEvent(QPaintEvent* pEvent) override;

private:
    // 20260322 ZJH 创建 8 个页签按钮并配置布局和样式
    void setupUI();

    // 20260322 ZJH 刷新所有按钮的选中/未选中视觉样式
    void updateButtonStyles();

    // 20260324 ZJH 构建并缓存选中/未选中样式字符串，主题切换后重新调用
    void buildStyleCache();

    QHBoxLayout* m_pLayout = nullptr;   // 水平等分布局，按钮宽度均分
    QList<QPushButton*> m_arrButtons;   // 8 个页签按钮，顺序与 PageIndex 枚举对应
    int m_nCurrentIndex = 0;            // 当前选中页签索引，默认 0（项目页）

    // 20260324 ZJH 缓存选中/未选中样式字符串，避免每次 updateButtonStyles 重复构建
    QString m_strActiveStyle;    // 20260324 ZJH 选中状态按钮样式（蓝色背景白色文字）
    QString m_strInactiveStyle;  // 20260324 ZJH 未选中状态按钮样式（透明背景灰色文字）
};

}  // namespace OmniMatch
