// 20260322 ZJH ImagePage — 图像标注页面
// BasePage 子类，三栏布局：左侧标签/工具/标注列表面板、中央图像查看器、右侧信息面板
// 支持矩形/多边形标注绘制、撤销重做、图像导航、缩放控制
#pragma once

#include "ui/pages/BasePage.h"   // 20260322 ZJH 页面基类

#include <QLabel>         // 20260322 ZJH 文字标签
#include <QComboBox>      // 20260322 ZJH 下拉选择框
#include <QPushButton>    // 20260322 ZJH 按钮
#include <QButtonGroup>   // 20260322 ZJH 互斥按钮组
#include <QSlider>        // 20260322 ZJH 滑块
#include <QListWidget>    // 20260322 ZJH 列表控件
#include <QCheckBox>      // 20260322 ZJH 复选框
#include <QFutureWatcher> // 20260324 ZJH 异步图像加载完成通知

// 20260322 ZJH 前向声明
class ZoomableGraphicsView;
class AnnotationController;
class ImageDataset;

// 20260322 ZJH 图像标注页面
// 布局：
//   左面板 (250px): 标签管理 + 工具选择 + 标注列表
//   中央面板: 导航栏 + ZoomableGraphicsView 图像查看器
//   右面板 (220px): 缩略图预览 + 缩放控制 + 文件信息 + 鼠标信息
class ImagePage : public BasePage
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    explicit ImagePage(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~ImagePage() override = default;

    // 20260322 ZJH 事件过滤器：拦截 viewport 鼠标事件转发给 AnnotationController
    bool eventFilter(QObject* pWatched, QEvent* pEvent) override;

    // ===== 生命周期回调 =====

    // 20260322 ZJH 页面进入前台时刷新显示
    void onEnter() override;

    // 20260322 ZJH 页面离开前台时保存状态
    void onLeave() override;

    // 20260324 ZJH 项目加载后绑定数据集（Template Method 扩展点）
    void onProjectLoadedImpl() override;

    // 20260324 ZJH 项目关闭时清空（Template Method 扩展点）
    void onProjectClosedImpl() override;

public slots:
    // 20260322 ZJH 加载指定 UUID 的图像到查看器
    // 参数: strUuid - 图像 UUID
    void loadImage(const QString& strUuid);

    // 20260322 ZJH 按索引导航到图像
    // 参数: nIndex - 图像在数据集中的索引（0-based）
    void navigateToImage(int nIndex);

private slots:
    // 20260322 ZJH 上一张图像
    void onPrevImage();

    // 20260322 ZJH 下一张图像
    void onNextImage();

    // 20260322 ZJH 适应视图
    void onFitView();

    // 20260322 ZJH 1:1 实际大小
    void onActualSize();

    // 20260322 ZJH 缩放变化时更新 UI
    void onZoomChanged(double dPercent);

    // 20260322 ZJH 鼠标位置变化时更新坐标信息
    void onMousePositionChanged(const QPointF& ptScene, const QPoint& ptImage);

    // 20260322 ZJH 像素值变化时更新像素信息
    void onPixelValueChanged(const QPoint& ptImage, int nGray, QRgb rgbValue);

    // 20260322 ZJH 工具按钮组切换
    void onToolChanged(int nId);

    // 20260322 ZJH 标签选择变化
    void onLabelComboChanged(int nIndex);

    // 20260322 ZJH 标注数量变化时刷新列表
    void onAnnotationCountChanged(int nCount);

    // 20260322 ZJH 删除选中标注
    void onDeleteAnnotation();

    // 20260322 ZJH 显示/隐藏标注切换
    void onToggleAnnotationVisibility(bool bVisible);

    // 20260322 ZJH 缩放滑块值变化
    void onZoomSliderChanged(int nValue);

    // 20260322 ZJH 笔刷大小变化
    void onBrushSizeChanged(int nValue);

    // 20260324 ZJH 异步图像加载完成回调
    // 读取 m_pImageWatcher 的结果，设置到图像查看器并绑定标注控制器
    void onAsyncImageLoaded();

private:
    // 20260322 ZJH 创建左侧面板
    QWidget* createLeftPanel();

    // 20260322 ZJH 创建中央面板
    QWidget* createCenterPanel();

    // 20260322 ZJH 创建右侧面板
    QWidget* createRightPanel();

    // 20260322 ZJH 刷新标签下拉框
    void refreshLabelCombo();

    // 20260322 ZJH 刷新标注列表
    void refreshAnnotationList();

    // 20260322 ZJH 更新导航栏索引文字
    void updateNavLabel();

    // 20260322 ZJH 创建分组框容器（统一样式）
    QWidget* createGroupBox(const QString& strTitle, QLayout* pLayout);

    // ===== 数据 =====
    ImageDataset* m_pDataset = nullptr;  // 20260322 ZJH 当前数据集指针
    int m_nCurrentIndex = -1;             // 20260322 ZJH 当前图像在数据集中的索引

    // 20260324 ZJH 异步图像加载成员
    QFutureWatcher<QImage>* m_pImageWatcher = nullptr;  // 20260324 ZJH 异步加载完成通知器
    int m_nPendingIndex = -1;                            // 20260324 ZJH 正在异步加载的图像索引（用于判断结果是否过期）
    QString m_strPendingUuid;                            // 20260324 ZJH 正在异步加载的图像 UUID
    QString m_strPendingFilePath;                        // 20260324 ZJH 正在异步加载的图像文件路径

    // ===== 左侧面板 =====
    QComboBox* m_pLabelCombo;             // 20260322 ZJH 标签选择下拉框
    QPushButton* m_pBtnAssignLabel;       // 20260322 ZJH 分配标签按钮
    QPushButton* m_pBtnManageLabels;      // 20260322 ZJH 管理标签按钮
    QButtonGroup* m_pToolGroup;           // 20260322 ZJH 工具按钮互斥组
    QPushButton* m_pBtnSelect;            // 20260322 ZJH 选择工具按钮(V)
    QPushButton* m_pBtnRect;              // 20260322 ZJH 矩形工具按钮(B)
    QPushButton* m_pBtnPolygon;           // 20260322 ZJH 多边形工具按钮(P)
    QPushButton* m_pBtnBrush;             // 20260322 ZJH 画笔工具按钮(D)
    QSlider* m_pBrushSlider;              // 20260322 ZJH 笔刷大小滑块
    QLabel* m_pBrushSizeLabel;            // 20260322 ZJH 笔刷大小数值标签
    QWidget* m_pBrushGroup;               // 20260322 ZJH 笔刷大小控件组（仅画笔工具可见）
    QListWidget* m_pAnnotationList;       // 20260322 ZJH 标注列表
    QPushButton* m_pBtnDeleteAnnotation;  // 20260322 ZJH 删除标注按钮
    QCheckBox* m_pChkShowAnnotations;     // 20260322 ZJH 显示/隐藏标注复选框

    // ===== 中央面板 =====
    ZoomableGraphicsView* m_pGraphicsView;  // 20260322 ZJH 图像查看器
    AnnotationController* m_pAnnotCtrl;     // 20260322 ZJH 标注控制器
    QPushButton* m_pBtnPrev;                // 20260322 ZJH 上一张按钮
    QPushButton* m_pBtnNext;                // 20260322 ZJH 下一张按钮
    QLabel* m_pNavLabel;                    // 20260322 ZJH 导航索引标签
    QPushButton* m_pBtnFit;                 // 20260322 ZJH 适应按钮
    QPushButton* m_pBtnActual;              // 20260322 ZJH 1:1 按钮
    QLabel* m_pZoomLabel;                   // 20260322 ZJH 缩放百分比标签

    // ===== 右侧面板 =====
    QLabel* m_pThumbnailLabel;      // 20260322 ZJH 缩略图预览
    QSlider* m_pZoomSlider;         // 20260322 ZJH 缩放滑块
    QLabel* m_pZoomValueLabel;      // 20260322 ZJH 缩放数值标签
    QLabel* m_pFileNameLabel;       // 20260322 ZJH 文件名
    QLabel* m_pFileSizeLabel;       // 20260322 ZJH 文件尺寸
    QLabel* m_pFileBytesLabel;      // 20260322 ZJH 文件大小（字节）
    QLabel* m_pFileDepthLabel;      // 20260322 ZJH 色深
    QLabel* m_pMousePosLabel;       // 20260322 ZJH 鼠标坐标
    QLabel* m_pPixelValueLabel;     // 20260322 ZJH 像素值
};
