// 20260322 ZJH SplitPage — 数据集拆分页面
// 提供训练/验证/测试集比例滑块、预设按钮、执行拆分功能
// 拆分后刷新统计卡片、标签分布表格和条形图
// 继承自 BasePage，使用三栏布局（左配置面板 + 中央统计视图 + 右说明面板）

#pragma once

#include "ui/pages/BasePage.h"  // 20260322 ZJH 页面基类（三栏布局 + 生命周期回调）

// 20260322 ZJH Qt 控件前向声明，避免不必要的头文件展开
class QLabel;
class QLineEdit;
class QSlider;
class QSpinBox;
class QCheckBox;
class QPushButton;
class QGroupBox;
class QTableWidget;
class QFrame;
class ClassDistributionChart;  // 20260322 ZJH 类别分布条形图控件

// 20260322 ZJH 数据集拆分配置页面
// 三栏布局：
//   左栏(280px): 拆分配置 + 快速预设 + 操作按钮
//   中栏:         统计卡片 + 标签分布表格 + 分布图表
//   右栏(220px): 使用说明 + 拆分状态
class SplitPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化三栏 UI 布局
    explicit SplitPage(QWidget* pParent = nullptr);

    // 20260322 ZJH 默认析构，Qt 对象树管理子控件
    ~SplitPage() override = default;

    // ===== 生命周期回调 =====

    // 20260322 ZJH 页面切换到前台时调用（刷新统计显示）
    void onEnter() override;

    // 20260324 ZJH 项目加载后调用，从 ImageDataset 读取当前拆分状态（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时清空所有显示（Template Method 扩展点）
    void onProjectClosedImpl() override;

private slots:
    // ===== 滑块/微调框联动槽 =====

    // 20260322 ZJH 训练集滑块值变化 → 同步微调框 + 更新测试集显示
    void onTrainSliderChanged(int nValue);

    // 20260322 ZJH 训练集微调框值变化 → 同步滑块 + 更新测试集显示
    void onTrainSpinChanged(int nValue);

    // 20260322 ZJH 验证集滑块值变化 → 同步微调框 + 更新测试集显示
    void onValSliderChanged(int nValue);

    // 20260322 ZJH 验证集微调框值变化 → 同步滑块 + 更新测试集显示
    void onValSpinChanged(int nValue);

    // ===== 预设按钮槽 =====

    // 20260322 ZJH 应用 70/15/15 预设
    void onPreset701515();

    // 20260322 ZJH 应用 80/10/10 预设
    void onPreset801010();

    // 20260322 ZJH 应用 60/20/20 预设
    void onPreset602020();

    // ===== 操作按钮槽 =====

    // 20260322 ZJH 执行拆分：调用 ImageDataset::autoSplit，刷新统计
    void onExecuteSplit();

    // 20260322 ZJH 重置拆分：将全部图像设为 Unassigned，刷新统计
    void onResetSplit();

private:
    // ===== UI 构建私有方法 =====

    // 20260322 ZJH 构建左侧配置面板（拆分配置 + 预设 + 操作按钮）
    QWidget* buildLeftPanel();

    // 20260322 ZJH 构建中央内容区（统计卡片 + 表格 + 图表）
    QWidget* buildCenterPanel();

    // 20260322 ZJH 构建右侧说明面板（说明文字 + 拆分状态）
    QWidget* buildRightPanel();

    // 20260322 ZJH 创建单个统计卡片 QFrame
    // 参数: strTitle - 卡片标题（如 "训练集"）
    //       color    - 主题颜色（标题栏/数字颜色）
    //       ppCount  - 输出参数：数量 QLabel 指针
    //       ppPct    - 输出参数：百分比 QLabel 指针
    // 返回: 构建好的卡片 QFrame（调用方将其加入布局）
    QFrame* createStatCard(const QString& strTitle, const QColor& color,
                           QLabel** ppCount, QLabel** ppPct);

    // ===== 数据刷新私有方法 =====

    // 20260322 ZJH 根据当前项目数据刷新统计卡片、表格和图表
    void refreshStats();

    // 20260322 ZJH 更新测试集百分比显示（根据训练集 + 验证集计算）
    void updateTestLabel();

    // 20260322 ZJH 应用预设比例（抑制信号避免联动死循环）
    void applyPreset(int nTrain, int nVal);

    // ===== 左栏控件 =====

    QLineEdit*   m_pSplitNameEdit;    // 20260322 ZJH 拆分名称输入框
    QSlider*     m_pTrainSlider;      // 20260322 ZJH 训练集比例滑块（50~95）
    QSpinBox*    m_pTrainSpin;        // 20260322 ZJH 训练集比例微调框
    QSlider*     m_pValSlider;        // 20260322 ZJH 验证集比例滑块（0~30）
    QSpinBox*    m_pValSpin;          // 20260322 ZJH 验证集比例微调框
    QLabel*      m_pTestLabel;        // 20260322 ZJH 测试集比例显示标签（只读）
    QCheckBox*   m_pStratifiedCheck;  // 20260322 ZJH 分层采样复选框

    QPushButton* m_pPreset701515Btn;  // 20260322 ZJH 70/15/15 预设按钮
    QPushButton* m_pPreset801010Btn;  // 20260322 ZJH 80/10/10 预设按钮
    QPushButton* m_pPreset602020Btn;  // 20260322 ZJH 60/20/20 预设按钮

    QPushButton* m_pExecuteBtn;       // 20260322 ZJH 执行拆分主按钮（蓝色）
    QPushButton* m_pResetBtn;         // 20260322 ZJH 重置拆分按钮

    // ===== 中栏控件 =====

    QLabel*      m_pTrainCount;       // 20260322 ZJH 训练集统计卡片 — 数量标签
    QLabel*      m_pTrainPct;         // 20260322 ZJH 训练集统计卡片 — 百分比标签
    QLabel*      m_pValCount;         // 20260322 ZJH 验证集统计卡片 — 数量标签
    QLabel*      m_pValPct;           // 20260322 ZJH 验证集统计卡片 — 百分比标签
    QLabel*      m_pTestCount;        // 20260322 ZJH 测试集统计卡片 — 数量标签
    QLabel*      m_pTestPct;          // 20260322 ZJH 测试集统计卡片 — 百分比标签

    QTableWidget*          m_pDistTable;   // 20260322 ZJH 标签分布明细表格（列：标签|训练|验证|测试|总计）
    ClassDistributionChart* m_pChart;      // 20260322 ZJH 类别分布条形图

    // ===== 右栏控件 =====

    QLabel*      m_pSplitStatusLabel;   // 20260322 ZJH 拆分状态标签（已拆分/未拆分）

    // ===== 状态标志 =====

    bool m_bUpdating = false;  // 20260322 ZJH 滑块/微调框联动时防止递归触发
};
