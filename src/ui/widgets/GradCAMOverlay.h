// 20260323 ZJH GradCAMOverlay — Grad-CAM 热力图叠加控件
// 在原始图像上叠加 Grad-CAM 可解释性热力图
// 支持透明度调节和 Jet/Viridis 色彩映射
// 20260324 ZJH 新增二值化缺陷图模式（缺陷=黑色，背景=白色）
#pragma once

#include <QWidget>   // 20260323 ZJH 控件基类
#include <QImage>    // 20260323 ZJH 图像
#include <QVector>   // 20260323 ZJH 动态数组

// 20260323 ZJH 色彩映射枚举
enum class ColorMapType
{
    Jet = 0,     // 20260323 ZJH Jet 色彩映射（蓝→青→黄→红）
    Viridis,     // 20260323 ZJH Viridis 色彩映射（紫→蓝→绿→黄）
    Hot,         // 20260323 ZJH Hot 色彩映射（黑→红→黄→白）
    Turbo        // 20260323 ZJH Turbo 色彩映射（改进版 Jet）
};

// 20260324 ZJH 显示模式枚举
// Heatmap: 原有热力图叠加模式（色彩映射 + alpha 混合）
// Binary:  二值化缺陷图模式（缺陷=黑色，背景=白色，无 alpha 混合）
enum class GradCAMDisplayMode
{
    Heatmap = 0,  // 20260324 ZJH 热力图模式（原有行为）
    Binary  = 1   // 20260324 ZJH 二值化缺陷图（缺陷=黑色，背景=白色）
};

// 20260323 ZJH Grad-CAM 热力图叠加控件
class GradCAMOverlay : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit GradCAMOverlay(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~GradCAMOverlay() override = default;

    // 20260323 ZJH 设置原始图像
    void setOriginalImage(const QImage& image);

    // 20260323 ZJH 设置 Grad-CAM 热力图数据
    // 参数: vecHeatmap - 归一化热力图 [0, 1]，行优先存储
    //       nWidth - 热力图宽度
    //       nHeight - 热力图高度
    void setHeatmap(const QVector<double>& vecHeatmap, int nWidth, int nHeight);

    // 20260323 ZJH 设置热力图透明度
    // 参数: dAlpha - 透明度 [0, 1]，0=仅原图，1=仅热力图
    void setOverlayAlpha(double dAlpha);

    // 20260323 ZJH 设置色彩映射类型
    void setColorMap(ColorMapType type);

    // 20260324 ZJH 设置显示模式（热力图 / 二值化缺陷图）
    // 参数: eMode - 显示模式枚举
    void setDisplayMode(GradCAMDisplayMode eMode);

    // 20260324 ZJH 获取当前显示模式
    GradCAMDisplayMode displayMode() const;

    // 20260324 ZJH 设置二值化阈值
    // 参数: dThreshold - 阈值 [0, 1]，>=阈值的像素视为缺陷（黑色），<阈值视为背景（白色）
    void setThreshold(double dThreshold);

    // 20260324 ZJH 获取当前二值化阈值
    double threshold() const;

    // 20260323 ZJH 清空数据
    void clear();

    // 20260323 ZJH 获取叠加后的合成图像
    // 20260324 ZJH 返回 const 引用避免不必要的 QImage 深拷贝
    const QImage& compositeImage() const;

    // 20260323 ZJH 推荐最小尺寸
    QSize minimumSizeHint() const override;

    // 20260324 ZJH 返回控件的推荐尺寸，供布局管理器参考
    QSize sizeHint() const override;

protected:
    // 20260323 ZJH 绘制事件
    void paintEvent(QPaintEvent* pEvent) override;

private:
    // 20260323 ZJH 将归一化热力图转换为彩色 QImage
    QImage heatmapToColorImage() const;

    // 20260324 ZJH 将归一化热力图转换为二值化 QImage
    // 缺陷（>=阈值）= 黑色，背景（<阈值）= 白色
    QImage heatmapToBinaryImage() const;

    // 20260323 ZJH 色彩映射：将 [0, 1] 映射为 RGB
    QColor colorMap(double dValue) const;

    // 20260323 ZJH 合成原始图像和热力图
    void updateComposite();

    // 20260324 ZJH 预计算 256 级颜色查找表，避免 heatmapToColorImage 逐像素调用 colorMap()
    void rebuildColorLUT();

    QImage m_imgOriginal;             // 20260323 ZJH 原始图像
    QVector<double> m_vecHeatmap;     // 20260323 ZJH 热力图数据 [0, 1]
    int m_nHeatmapW = 0;              // 20260323 ZJH 热力图宽度
    int m_nHeatmapH = 0;              // 20260323 ZJH 热力图高度
    double m_dAlpha = 0.5;            // 20260323 ZJH 叠加透明度
    ColorMapType m_colorMapType = ColorMapType::Jet;  // 20260323 ZJH 色彩映射类型
    QImage m_imgComposite;            // 20260323 ZJH 合成图像缓存
    QVector<QRgb> m_arrColorLUT;      // 20260324 ZJH 256 级预计算颜色查找表

    // 20260324 ZJH 显示模式和二值化阈值
    GradCAMDisplayMode m_eDisplayMode = GradCAMDisplayMode::Binary;  // 20260324 ZJH 默认二值化模式
    double m_dThreshold = 0.5;  // 20260324 ZJH 二值化阈值，默认 0.5
};
