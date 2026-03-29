// 20260323 ZJH AugmentationPreviewDialog — 数据增强预览对话框
// 实时预览数据增强效果，包括亮度、翻转、旋转、裁剪等变换
// 左侧参数面板 + 右侧3x3网格预览
// 20260324 ZJH 优化：添加 150ms 防抖定时器，避免滑块拖动时频繁刷新
#pragma once

#include <QDialog>      // 20260323 ZJH 对话框基类
#include <QImage>       // 20260323 ZJH 图像
#include <QLabel>       // 20260323 ZJH 标签
#include <QSlider>      // 20260323 ZJH 滑块
#include <QCheckBox>    // 20260323 ZJH 复选框
#include <QGridLayout>  // 20260323 ZJH 网格布局
#include <QVector>      // 20260323 ZJH 动态数组
#include <QTimer>       // 20260324 ZJH 防抖定时器

// 20260323 ZJH 数据增强预览对话框
class AugmentationPreviewDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    // 参数: imgSample - 用于预览的样本图像
    //       pParent - 父窗口
    explicit AugmentationPreviewDialog(const QImage& imgSample, QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~AugmentationPreviewDialog() override = default;

    // 20260323 ZJH 获取亮度调整范围
    double brightnessRange() const;

    // 20260323 ZJH 获取翻转概率
    double flipProbability() const;

    // 20260323 ZJH 获取旋转角度范围
    double rotationRange() const;

    // 20260323 ZJH 获取是否启用各项增强
    bool isBrightnessEnabled() const;
    bool isFlipEnabled() const;
    bool isRotationEnabled() const;
    bool isCropEnabled() const;
    bool isNoiseEnabled() const;

    // 20260324 ZJH 获取新增增强项的启用状态
    bool isVerticalFlipEnabled() const;
    bool isSaturationEnabled() const;
    bool isRandomErasingEnabled() const;
    bool isGaussianBlurEnabled() const;
    bool isAffineEnabled() const;

private slots:
    // 20260323 ZJH 刷新预览网格
    void refreshPreview();

    // 20260324 ZJH 参数变化时的防抖入口（重启 150ms 定时器）
    void scheduleRefresh();

private:
    // 20260323 ZJH 创建参数面板
    QWidget* createParamsPanel();

    // 20260323 ZJH 创建预览网格面板
    QWidget* createPreviewPanel();

    // 20260323 ZJH 应用增强到图像并返回变换后的图像
    QImage applyAugmentation(const QImage& imgSrc) const;

    // 20260324 ZJH 亮度调整（原地修改，避免额外深拷贝）
    void adjustBrightness(QImage& img, double dFactor) const;

    // 20260323 ZJH 水平翻转
    QImage flipHorizontal(const QImage& img) const;

    // 20260324 ZJH 垂直翻转
    QImage flipVertical(const QImage& img) const;

    // 20260323 ZJH 旋转
    QImage rotateImage(const QImage& img, double dAngleDeg) const;

    // 20260324 ZJH 添加高斯噪声（原地修改，避免额外深拷贝）
    void addNoise(QImage& img, double dSigma) const;

    // 20260324 ZJH 饱和度/色调抖动（原地修改 HSV 通道）
    void adjustSaturationHue(QImage& img, double dSatFactor, double dHueShift) const;

    // 20260324 ZJH 随机擦除：在图像上填充灰色矩形
    void applyRandomErasing(QImage& img, double dRatio) const;

    // 20260324 ZJH 随机缩放裁剪：裁剪后缩放回原尺寸
    QImage applyRandomCrop(const QImage& img, double dMinScale) const;

    // 20260324 ZJH 高斯模糊近似（3x3 均值模糊多次叠加）
    void applyGaussianBlur(QImage& img, double dSigma) const;

    // 20260324 ZJH 仿射变换（剪切+平移）
    QImage applyAffine(const QImage& img, double dShearDeg, double dTranslate) const;

    QImage m_imgSample;                // 20260323 ZJH 样本图像
    QVector<QLabel*> m_vecPreviewLabels;  // 20260323 ZJH 预览标签列表（3x3=9个）

    // 20260323 ZJH 参数控件（基础）
    QCheckBox* m_pChkBrightness;       // 20260323 ZJH 亮度启用
    QSlider*   m_pSldBrightness;       // 20260323 ZJH 亮度范围滑块
    QCheckBox* m_pChkFlip;             // 20260323 ZJH 水平翻转启用
    QSlider*   m_pSldFlipProb;         // 20260323 ZJH 翻转概率滑块
    QCheckBox* m_pChkRotation;         // 20260323 ZJH 旋转启用
    QSlider*   m_pSldRotation;         // 20260323 ZJH 旋转角度滑块
    QCheckBox* m_pChkCrop;             // 20260323 ZJH 随机裁剪启用
    QCheckBox* m_pChkNoise;            // 20260323 ZJH 噪声启用

    // 20260324 ZJH 参数控件（扩展）
    QCheckBox* m_pChkVerticalFlip;     // 20260324 ZJH 垂直翻转启用
    QCheckBox* m_pChkSaturation;       // 20260324 ZJH 饱和度/色调抖动启用
    QSlider*   m_pSldSaturation;       // 20260324 ZJH 饱和度强度滑块
    QCheckBox* m_pChkRandomErasing;    // 20260324 ZJH 随机擦除启用
    QSlider*   m_pSldErasingRatio;     // 20260324 ZJH 擦除面积比例滑块
    QCheckBox* m_pChkGaussianBlur;     // 20260324 ZJH 高斯模糊启用
    QSlider*   m_pSldBlurSigma;        // 20260324 ZJH 模糊 sigma 滑块
    QCheckBox* m_pChkAffine;           // 20260324 ZJH 仿射变换启用
    QSlider*   m_pSldShearDeg;         // 20260324 ZJH 剪切角度滑块

    // 20260324 ZJH 防抖定时器：滑块拖动时延迟 150ms 再刷新预览
    QTimer* m_pDebounceTimer = nullptr;
};
