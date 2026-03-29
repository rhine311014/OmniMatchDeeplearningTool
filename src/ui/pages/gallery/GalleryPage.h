// 20260322 ZJH GalleryPage — 图库浏览页面
// BasePage 子类，三栏布局展示数据集中的全部图像缩略图
// 左面板：统计信息 + 标签过滤 + 拆分过滤 + 标签管理
// 中央面板：工具栏（导入/删除/排序/搜索/缩略图滑块）+ QListView 缩略图网格
// 支持拖放导入、多选操作、双击跳转标注页
// 20260324 ZJH 优化：添加 50ms 防抖定时器，合并快速连续的数据变更信号为一次刷新
#pragma once

#include "ui/pages/BasePage.h"       // 20260322 ZJH 页面基类

#include <QListView>                 // 20260322 ZJH 缩略图列表视图
#include <QStandardItemModel>        // 20260322 ZJH 标准条目模型
#include <QSortFilterProxyModel>     // 20260322 ZJH 排序/过滤代理模型
#include <QLabel>                    // 20260322 ZJH 统计文本标签
#include <QCheckBox>                 // 20260322 ZJH 过滤复选框
#include <QPushButton>               // 20260322 ZJH 按钮
#include <QComboBox>                 // 20260322 ZJH 排序下拉框
#include <QLineEdit>                 // 20260322 ZJH 搜索框
#include <QSlider>                   // 20260322 ZJH 缩略图大小滑块
#include <QVBoxLayout>               // 20260322 ZJH 垂直布局
#include <QGroupBox>                 // 20260322 ZJH 分组框
#include <QSet>                      // 20260322 ZJH 标签过滤集合
#include <QTimer>                    // 20260324 ZJH 防抖定时器

// 20260322 ZJH 前向声明
class ImageDataset;
class ThumbnailDelegate;

// 20260322 ZJH 自定义排序过滤代理模型
// 支持按文件名搜索 + 标签过滤 + 拆分过滤
class GalleryFilterProxy : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    explicit GalleryFilterProxy(QObject* pParent = nullptr);

    // 20260322 ZJH 设置搜索关键词（匹配文件名）
    void setSearchText(const QString& strText);

    // 20260322 ZJH 设置标签过滤集合（空集 = 全部显示）
    void setLabelFilter(const QSet<int>& setLabelIds);

    // 20260322 ZJH 设置拆分过滤集合（空集 = 全部显示）
    void setSplitFilter(const QSet<int>& setSplitTypes);

    // 20260322 ZJH 清除所有过滤条件
    void clearFilters();

protected:
    // 20260322 ZJH 过滤行：同时满足搜索+标签+拆分条件才显示
    bool filterAcceptsRow(int nSourceRow, const QModelIndex& sourceParent) const override;

    // 20260322 ZJH 自定义排序比较（按名称/标签/拆分排序）
    bool lessThan(const QModelIndex& left, const QModelIndex& right) const override;

private:
    QString m_strSearchText;       // 20260322 ZJH 当前搜索关键词
    QSet<int> m_setLabelFilter;    // 20260322 ZJH 标签过滤集合
    QSet<int> m_setSplitFilter;    // 20260322 ZJH 拆分过滤集合
};

// 20260322 ZJH 图库浏览页面
class GalleryPage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化三栏布局和所有子控件
    explicit GalleryPage(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~GalleryPage() override = default;

    // ===== 生命周期回调 =====

    // 20260322 ZJH 页面进入前台时刷新显示
    void onEnter() override;

    // 20260322 ZJH 页面离开前台
    void onLeave() override;

    // 20260324 ZJH 项目加载后绑定数据集信号并刷新（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时清空所有内容（Template Method 扩展点）
    void onProjectClosedImpl() override;

protected:
    // 20260322 ZJH 拖放进入事件：检测是否有可导入的文件
    void dragEnterEvent(QDragEnterEvent* pEvent) override;

    // 20260322 ZJH 拖放移动事件：持续接受拖入
    void dragMoveEvent(QDragMoveEvent* pEvent) override;

    // 20260322 ZJH 拖放释放事件：处理文件/文件夹导入
    void dropEvent(QDropEvent* pEvent) override;

private slots:
    // 20260322 ZJH 导入图像文件（打开文件对话框）
    void onImportImages();

    // 20260322 ZJH 导入文件夹（打开文件夹对话框）
    void onImportFolder();

    // 20260322 ZJH 删除选中的图像
    void onDeleteSelected();

    // 20260322 ZJH 搜索框文本变化时实时过滤
    void onSearchTextChanged(const QString& strText);

    // 20260322 ZJH 排序方式变化
    void onSortChanged(int nIndex);

    // 20260322 ZJH 缩略图大小滑块值变化
    void onThumbnailSizeChanged(int nSize);

    // 20260322 ZJH 缩略图双击 → 跳转标注页
    void onItemDoubleClicked(const QModelIndex& index);

    // 20260322 ZJH 数据集内容变化时刷新模型
    void onDatasetChanged();

    // 20260322 ZJH 标签列表变化时刷新过滤面板
    void onLabelsChanged();

    // 20260322 ZJH 标签过滤复选框状态变化
    void onLabelFilterChanged();

    // 20260322 ZJH 拆分过滤复选框状态变化
    void onSplitFilterChanged();

private:
    // 20260322 ZJH 创建左侧面板（统计+过滤+标签管理）
    QWidget* createLeftPanel();

    // 20260322 ZJH 创建中央面板（工具栏+缩略图网格）
    QWidget* createCenterPanel();

    // 20260322 ZJH 刷新缩略图模型（从数据集重建全部条目）
    void refreshModel();

    // 20260322 ZJH 更新统计标签显示
    void updateStatistics();

    // 20260322 ZJH 重建标签过滤面板中的复选框列表
    void rebuildLabelFilters();

    // ===== 左面板组件 =====

    QGroupBox*  m_pStatsGroup;          // 20260322 ZJH "统计" 分组框
    QLabel*     m_pLblTotalImages;      // 20260322 ZJH 总图像数
    QLabel*     m_pLblLabeledCount;     // 20260322 ZJH 已标注数
    QLabel*     m_pLblUnlabeledCount;   // 20260322 ZJH 未标注数
    QLabel*     m_pLblTrainCount;       // 20260322 ZJH 训练集数量
    QLabel*     m_pLblValCount;         // 20260322 ZJH 验证集数量
    QLabel*     m_pLblTestCount;        // 20260322 ZJH 测试集数量

    QGroupBox*   m_pLabelFilterGroup;   // 20260322 ZJH "标签过滤" 分组框
    QVBoxLayout* m_pLabelFilterLayout;  // 20260322 ZJH 标签过滤内部布局
    QCheckBox*   m_pChkAllLabels;       // 20260322 ZJH "全部" 复选框
    QVector<QCheckBox*> m_vecLabelCheckboxes;  // 20260322 ZJH 各标签复选框列表

    QGroupBox*  m_pSplitFilterGroup;    // 20260322 ZJH "拆分过滤" 分组框
    QCheckBox*  m_pChkSplitAll;         // 20260322 ZJH 拆分 - 全部
    QCheckBox*  m_pChkSplitTrain;       // 20260322 ZJH 拆分 - 训练
    QCheckBox*  m_pChkSplitVal;         // 20260322 ZJH 拆分 - 验证
    QCheckBox*  m_pChkSplitTest;        // 20260322 ZJH 拆分 - 测试

    QPushButton* m_pBtnManageLabels;    // 20260322 ZJH "标签管理" 按钮

    // ===== 中央面板组件 =====

    QPushButton* m_pBtnImportImages;    // 20260322 ZJH "导入图像" 按钮
    QPushButton* m_pBtnImportFolder;    // 20260322 ZJH "导入文件夹" 按钮
    QPushButton* m_pBtnDeleteSelected;  // 20260322 ZJH "删除选中" 按钮
    QComboBox*   m_pCmbSort;            // 20260322 ZJH 排序下拉框
    QLineEdit*   m_pEdtSearch;          // 20260322 ZJH 搜索框
    QSlider*     m_pSliderThumbSize;    // 20260322 ZJH 缩略图大小滑块

    QListView*              m_pListView;       // 20260322 ZJH 缩略图列表视图
    QStandardItemModel*     m_pModel;          // 20260322 ZJH 底层数据模型
    GalleryFilterProxy*     m_pProxyModel;     // 20260322 ZJH 过滤排序代理模型
    ThumbnailDelegate*      m_pDelegate;       // 20260322 ZJH 缩略图绘制代理

    // ===== 数据层引用 =====

    ImageDataset* m_pDataset = nullptr;  // 20260322 ZJH 当前项目的数据集（弱引用，不持有所有权）

    // 20260324 ZJH 导入图像后自动保存项目到 .dfproj
    void autoSaveProject();

    // 20260324 ZJH 防抖定时器：合并快速连续的数据集变更信号为一次刷新
    // dataChanged + imagesAdded + imagesRemoved + splitChanged 可能在同一操作中连续触发
    QTimer* m_pRefreshDebounce = nullptr;
};
