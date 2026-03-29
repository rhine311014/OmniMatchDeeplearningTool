// 20260323 ZJH AugmentationPreviewDialog — 数据增强预览对话框实现

#include "ui/dialogs/AugmentationPreviewDialog.h"

#include <QVBoxLayout>      // 20260323 ZJH 垂直布局
#include <QHBoxLayout>      // 20260323 ZJH 水平布局
#include <QPushButton>      // 20260323 ZJH 按钮
#include <QGroupBox>        // 20260323 ZJH 分组框
#include <QDialogButtonBox> // 20260323 ZJH 标准按钮组
#include <QTransform>       // 20260323 ZJH 图像变换
#include <QColor>           // 20260324 ZJH QColor HSV 色彩空间操作
#include <QRandomGenerator> // 20260323 ZJH 随机数生成器
#include <algorithm>        // 20260323 ZJH std::clamp
#include <cmath>            // 20260324 ZJH std::sqrt, std::tan, std::round

// 20260323 ZJH 构造函数
AugmentationPreviewDialog::AugmentationPreviewDialog(const QImage& imgSample, QWidget* pParent)
    : QDialog(pParent)
    , m_imgSample(imgSample.width() > 400 || imgSample.height() > 400
                  ? imgSample.scaled(400, 400, Qt::KeepAspectRatio, Qt::SmoothTransformation)
                  : imgSample)  // 20260323 ZJH 预缩放到 400px 以内，避免 9 次全分辨率增强
{
    setWindowTitle(QStringLiteral("数据增强预览"));  // 20260324 ZJH 窗口标题汉化
    setMinimumSize(800, 600);

    // 20260324 ZJH 初始化防抖定时器（150ms SingleShot）
    // 滑块拖动时每次值变化只重启定时器，定时器到期才真正执行 refreshPreview()
    m_pDebounceTimer = new QTimer(this);
    m_pDebounceTimer->setSingleShot(true);  // 20260324 ZJH 单次触发模式
    m_pDebounceTimer->setInterval(150);     // 20260324 ZJH 150ms 防抖间隔
    connect(m_pDebounceTimer, &QTimer::timeout, this, &AugmentationPreviewDialog::refreshPreview);

    // 20260324 ZJH 暗色主题样式表（使用 QStringLiteral 避免运行时 Latin-1 → UTF-16 转换）
    setStyleSheet(QStringLiteral(
        "QDialog { background: #1e2028; color: #e2e8f0; }"
        "QGroupBox { border: 1px solid #334155; border-radius: 6px; margin-top: 10px; padding-top: 14px; color: #94a3b8; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        "QCheckBox { color: #e2e8f0; }"
        "QSlider::groove:horizontal { background: #334155; height: 6px; border-radius: 3px; }"
        "QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -4px 0; border-radius: 7px; }"));

    // 20260323 ZJH 主布局：左参数 + 右预览
    QHBoxLayout* pMainLayout = new QHBoxLayout(this);

    pMainLayout->addWidget(createParamsPanel(), 0);   // 20260323 ZJH 左侧参数面板
    pMainLayout->addWidget(createPreviewPanel(), 1);   // 20260323 ZJH 右侧预览面板

    // 20260323 ZJH 初始刷新预览
    refreshPreview();
}

// 20260323 ZJH 创建参数面板
QWidget* AugmentationPreviewDialog::createParamsPanel()
{
    QWidget* pPanel = new QWidget;
    QVBoxLayout* pLayout = new QVBoxLayout(pPanel);
    pPanel->setFixedWidth(240);

    // 20260323 ZJH 亮度调整分组
    QGroupBox* pGrpBrightness = new QGroupBox(QStringLiteral("亮度"));  // 20260324 ZJH 汉化
    QVBoxLayout* pBriLayout = new QVBoxLayout(pGrpBrightness);
    m_pChkBrightness = new QCheckBox(QStringLiteral("启用"));  // 20260324 ZJH 汉化
    m_pChkBrightness->setChecked(true);
    m_pSldBrightness = new QSlider(Qt::Horizontal);
    m_pSldBrightness->setRange(5, 50);  // 20260323 ZJH 5%~50% 范围
    m_pSldBrightness->setValue(20);
    pBriLayout->addWidget(m_pChkBrightness);
    pBriLayout->addWidget(new QLabel(QStringLiteral("范围: ")));  // 20260324 ZJH 汉化
    pBriLayout->addWidget(m_pSldBrightness);
    pLayout->addWidget(pGrpBrightness);

    // 20260323 ZJH 翻转分组
    QGroupBox* pGrpFlip = new QGroupBox(QStringLiteral("水平翻转"));  // 20260324 ZJH 汉化
    QVBoxLayout* pFlipLayout = new QVBoxLayout(pGrpFlip);
    m_pChkFlip = new QCheckBox(QStringLiteral("启用"));  // 20260324 ZJH 汉化
    m_pChkFlip->setChecked(true);
    m_pSldFlipProb = new QSlider(Qt::Horizontal);
    m_pSldFlipProb->setRange(0, 100);
    m_pSldFlipProb->setValue(50);
    pFlipLayout->addWidget(m_pChkFlip);
    pFlipLayout->addWidget(new QLabel(QStringLiteral("概率:")));  // 20260324 ZJH 汉化
    pFlipLayout->addWidget(m_pSldFlipProb);
    pLayout->addWidget(pGrpFlip);

    // 20260323 ZJH 旋转分组
    QGroupBox* pGrpRotation = new QGroupBox(QStringLiteral("旋转"));  // 20260324 ZJH 汉化
    QVBoxLayout* pRotLayout = new QVBoxLayout(pGrpRotation);
    m_pChkRotation = new QCheckBox(QStringLiteral("启用"));  // 20260324 ZJH 汉化
    m_pChkRotation->setChecked(true);
    m_pSldRotation = new QSlider(Qt::Horizontal);
    m_pSldRotation->setRange(0, 45);  // 20260323 ZJH 0~45 度
    m_pSldRotation->setValue(15);
    pRotLayout->addWidget(m_pChkRotation);
    pRotLayout->addWidget(new QLabel(QStringLiteral("最大角度:")));  // 20260324 ZJH 汉化
    pRotLayout->addWidget(m_pSldRotation);
    pLayout->addWidget(pGrpRotation);

    // 20260324 ZJH 几何变换扩展
    QGroupBox* pGrpGeomExt = new QGroupBox(QStringLiteral("几何变换扩展"));
    QVBoxLayout* pGeomExtLayout = new QVBoxLayout(pGrpGeomExt);
    m_pChkVerticalFlip = new QCheckBox(QStringLiteral("垂直翻转"));  // 20260324 ZJH 垂直翻转
    m_pChkCrop = new QCheckBox(QStringLiteral("随机裁剪"));  // 20260324 ZJH 汉化
    m_pChkAffine = new QCheckBox(QStringLiteral("仿射变换"));  // 20260324 ZJH 仿射变换
    m_pSldShearDeg = new QSlider(Qt::Horizontal);  // 20260324 ZJH 剪切角度
    m_pSldShearDeg->setRange(0, 30);  // 20260324 ZJH 0~30度
    m_pSldShearDeg->setValue(10);
    pGeomExtLayout->addWidget(m_pChkVerticalFlip);
    pGeomExtLayout->addWidget(m_pChkCrop);
    pGeomExtLayout->addWidget(m_pChkAffine);
    pGeomExtLayout->addWidget(new QLabel(QStringLiteral("剪切角度:")));
    pGeomExtLayout->addWidget(m_pSldShearDeg);
    pLayout->addWidget(pGrpGeomExt);

    // 20260324 ZJH 颜色/噪声扩展
    QGroupBox* pGrpColorNoise = new QGroupBox(QStringLiteral("颜色 / 噪声"));
    QVBoxLayout* pColorNoiseLayout = new QVBoxLayout(pGrpColorNoise);
    m_pChkSaturation = new QCheckBox(QStringLiteral("饱和度/色调抖动"));  // 20260324 ZJH 饱和度启用
    m_pSldSaturation = new QSlider(Qt::Horizontal);  // 20260324 ZJH 饱和度强度
    m_pSldSaturation->setRange(5, 80);  // 20260324 ZJH 5%~80%
    m_pSldSaturation->setValue(20);
    m_pChkNoise = new QCheckBox(QStringLiteral("高斯噪声"));  // 20260324 ZJH 汉化
    m_pChkGaussianBlur = new QCheckBox(QStringLiteral("高斯模糊"));  // 20260324 ZJH 模糊
    m_pSldBlurSigma = new QSlider(Qt::Horizontal);  // 20260324 ZJH 模糊 sigma
    m_pSldBlurSigma->setRange(1, 30);  // 20260324 ZJH sigma 0.1~3.0 (值/10)
    m_pSldBlurSigma->setValue(10);
    m_pChkRandomErasing = new QCheckBox(QStringLiteral("随机擦除"));  // 20260324 ZJH 擦除
    m_pSldErasingRatio = new QSlider(Qt::Horizontal);  // 20260324 ZJH 擦除面积比例
    m_pSldErasingRatio->setRange(5, 40);  // 20260324 ZJH 5%~40%
    m_pSldErasingRatio->setValue(15);
    pColorNoiseLayout->addWidget(m_pChkSaturation);
    pColorNoiseLayout->addWidget(new QLabel(QStringLiteral("强度:")));
    pColorNoiseLayout->addWidget(m_pSldSaturation);
    pColorNoiseLayout->addWidget(m_pChkNoise);
    pColorNoiseLayout->addWidget(m_pChkGaussianBlur);
    pColorNoiseLayout->addWidget(new QLabel(QStringLiteral("Sigma:")));
    pColorNoiseLayout->addWidget(m_pSldBlurSigma);
    pColorNoiseLayout->addWidget(m_pChkRandomErasing);
    pColorNoiseLayout->addWidget(new QLabel(QStringLiteral("擦除面积:")));
    pColorNoiseLayout->addWidget(m_pSldErasingRatio);
    pLayout->addWidget(pGrpColorNoise);

    // 20260323 ZJH 刷新按钮
    QPushButton* pBtnRefresh = new QPushButton(QStringLiteral("刷新预览"));  // 20260324 ZJH 汉化
    // 20260324 ZJH 按钮样式表使用 QStringLiteral
    pBtnRefresh->setStyleSheet(QStringLiteral(
        "QPushButton { background: #2563eb; color: white; border-radius: 4px; padding: 8px; }"
        "QPushButton:hover { background: #1d4ed8; }"));
    connect(pBtnRefresh, &QPushButton::clicked, this, &AugmentationPreviewDialog::refreshPreview);
    pLayout->addWidget(pBtnRefresh);

    pLayout->addStretch();

    // 20260323 ZJH OK/Cancel
    QDialogButtonBox* pBtnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(pBtnBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(pBtnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    pLayout->addWidget(pBtnBox);

    // 20260324 ZJH 参数变化时通过防抖定时器延迟刷新（复选框 + 滑块）
    // 复选框切换和滑块拖动都走 scheduleRefresh()，150ms 内的连续变化合并为一次刷新
    connect(m_pChkBrightness, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkFlip, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkRotation, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkCrop, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkNoise, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldBrightness, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldFlipProb, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldRotation, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);

    // 20260324 ZJH 新增控件的防抖连接
    connect(m_pChkVerticalFlip, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkSaturation, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldSaturation, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkRandomErasing, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldErasingRatio, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkGaussianBlur, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldBlurSigma, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pChkAffine, &QCheckBox::toggled, this, &AugmentationPreviewDialog::scheduleRefresh);
    connect(m_pSldShearDeg, &QSlider::valueChanged, this, &AugmentationPreviewDialog::scheduleRefresh);

    return pPanel;
}

// 20260323 ZJH 创建预览面板
QWidget* AugmentationPreviewDialog::createPreviewPanel()
{
    QWidget* pPanel = new QWidget;
    QGridLayout* pGrid = new QGridLayout(pPanel);
    pGrid->setSpacing(4);

    // 20260323 ZJH 创建 3x3 预览网格
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            QLabel* pLbl = new QLabel;
            pLbl->setAlignment(Qt::AlignCenter);
            // 20260324 ZJH 预览标签样式使用 QStringLiteral
            pLbl->setStyleSheet(QStringLiteral("QLabel { background: #22262e; border: 1px solid #334155; border-radius: 4px; }"));
            pLbl->setMinimumSize(150, 150);
            pLbl->setScaledContents(true);
            pGrid->addWidget(pLbl, r, c);
            m_vecPreviewLabels.append(pLbl);
        }
    }

    return pPanel;
}

// 20260324 ZJH 参数变化时的防抖入口：重启 150ms 定时器
// 滑块拖动期间可能每秒触发数十次 valueChanged，通过定时器合并为一次 refreshPreview
void AugmentationPreviewDialog::scheduleRefresh()
{
    m_pDebounceTimer->start();  // 20260324 ZJH 重启定时器（如果已经在运行则重新计时）
}

// 20260323 ZJH 刷新预览网格
void AugmentationPreviewDialog::refreshPreview()
{
    for (QLabel* pLbl : m_vecPreviewLabels) {
        QImage imgAug = applyAugmentation(m_imgSample);  // 20260323 ZJH 应用随机增强
        QPixmap pix = QPixmap::fromImage(imgAug);
        pLbl->setPixmap(pix.scaled(pLbl->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

// 20260323 ZJH 应用增强
QImage AugmentationPreviewDialog::applyAugmentation(const QImage& imgSrc) const
{
    if (imgSrc.isNull()) return imgSrc;

    QImage imgResult = imgSrc.convertToFormat(QImage::Format_ARGB32);
    QRandomGenerator* pRng = QRandomGenerator::global();

    // 20260323 ZJH 亮度调整
    if (m_pChkBrightness->isChecked()) {
        double dRange = m_pSldBrightness->value() / 100.0;
        double dFactor = 1.0 + pRng->bounded(2.0 * dRange) - dRange;
        // 20260324 ZJH 原地修改，避免额外深拷贝
        adjustBrightness(imgResult, dFactor);
    }

    // 20260323 ZJH 水平翻转
    if (m_pChkFlip->isChecked()) {
        double dProb = m_pSldFlipProb->value() / 100.0;
        if (pRng->bounded(1.0) < dProb) {
            imgResult = flipHorizontal(imgResult);
        }
    }

    // 20260323 ZJH 旋转
    if (m_pChkRotation->isChecked()) {
        double dMaxAngle = m_pSldRotation->value();
        double dAngle = pRng->bounded(2.0 * dMaxAngle) - dMaxAngle;
        imgResult = rotateImage(imgResult, dAngle);
    }

    // 20260324 ZJH 垂直翻转
    if (m_pChkVerticalFlip->isChecked()) {
        if (pRng->bounded(1.0) < 0.5) {
            imgResult = flipVertical(imgResult);  // 20260324 ZJH 50% 概率垂直翻转
        }
    }

    // 20260324 ZJH 仿射变换（剪切+平移）
    if (m_pChkAffine->isChecked()) {
        double dShear = m_pSldShearDeg->value();  // 20260324 ZJH 剪切角度上限
        imgResult = applyAffine(imgResult, dShear, 0.1);  // 20260324 ZJH 平移比例固定 0.1
    }

    // 20260324 ZJH 随机缩放裁剪
    if (m_pChkCrop->isChecked()) {
        imgResult = applyRandomCrop(imgResult, 0.7);  // 20260324 ZJH 最小缩放 0.7
    }

    // 20260324 ZJH 饱和度/色调抖动
    if (m_pChkSaturation->isChecked()) {
        double dSatRange = m_pSldSaturation->value() / 100.0;  // 20260324 ZJH 滑块值转比例
        double dSatFactor = 1.0 + pRng->bounded(2.0 * dSatRange) - dSatRange;  // 20260324 ZJH 随机因子
        double dHueShift = (pRng->bounded(1.0) - 0.5) * 0.1;  // 20260324 ZJH 色调随机偏移
        adjustSaturationHue(imgResult, dSatFactor, dHueShift);
    }

    // 20260323 ZJH 高斯噪声
    if (m_pChkNoise->isChecked()) {
        // 20260324 ZJH 原地修改，避免额外深拷贝
        addNoise(imgResult, 15.0);
    }

    // 20260324 ZJH 高斯模糊
    if (m_pChkGaussianBlur->isChecked()) {
        double dSigma = m_pSldBlurSigma->value() / 10.0;  // 20260324 ZJH 滑块值/10 = 实际 sigma
        applyGaussianBlur(imgResult, dSigma);
    }

    // 20260324 ZJH 随机擦除
    if (m_pChkRandomErasing->isChecked()) {
        double dRatio = m_pSldErasingRatio->value() / 100.0;  // 20260324 ZJH 滑块值转面积比例
        applyRandomErasing(imgResult, dRatio);
    }

    return imgResult;
}

// 20260324 ZJH 亮度调整（原地修改，避免额外深拷贝）
// 参数: img - 待修改的图像引用
//       dFactor - 亮度倍率（>1 变亮，<1 变暗）
void AugmentationPreviewDialog::adjustBrightness(QImage& img, double dFactor) const
{
    for (int y = 0; y < img.height(); ++y) {
        QRgb* pLine = reinterpret_cast<QRgb*>(img.scanLine(y));  // 20260323 ZJH scanLine 直接操作像素
        for (int x = 0; x < img.width(); ++x) {
            int nR = std::clamp(static_cast<int>(qRed(pLine[x]) * dFactor), 0, 255);
            int nG = std::clamp(static_cast<int>(qGreen(pLine[x]) * dFactor), 0, 255);
            int nB = std::clamp(static_cast<int>(qBlue(pLine[x]) * dFactor), 0, 255);
            pLine[x] = qRgba(nR, nG, nB, qAlpha(pLine[x]));
        }
    }
}

// 20260323 ZJH 水平翻转
QImage AugmentationPreviewDialog::flipHorizontal(const QImage& img) const
{
    return img.flipped(Qt::Horizontal);  // 20260323 ZJH Qt6 推荐 flipped() 替代 mirrored()
}

// 20260324 ZJH 垂直翻转
QImage AugmentationPreviewDialog::flipVertical(const QImage& img) const
{
    return img.flipped(Qt::Vertical);  // 20260324 ZJH 垂直翻转（Qt6 推荐 flipped 替代 mirrored）
}

// 20260323 ZJH 旋转
QImage AugmentationPreviewDialog::rotateImage(const QImage& img, double dAngleDeg) const
{
    QTransform transform;
    transform.rotate(dAngleDeg);
    return img.transformed(transform, Qt::SmoothTransformation);
}

// 20260324 ZJH 添加高斯噪声（原地修改，避免额外深拷贝）
// 参数: img - 待修改的图像引用
//       dSigma - 噪声标准差
void AugmentationPreviewDialog::addNoise(QImage& img, double dSigma) const
{
    QRandomGenerator* pRng = QRandomGenerator::global();  // 20260323 ZJH 全局随机数生成器

    for (int y = 0; y < img.height(); ++y) {
        QRgb* pLine = reinterpret_cast<QRgb*>(img.scanLine(y));  // 20260323 ZJH scanLine 直接操作像素
        for (int x = 0; x < img.width(); ++x) {
            // 20260323 ZJH 中心极限定理近似高斯噪声（3 个均匀随机数求和）
            double dNoise = (pRng->bounded(1.0) + pRng->bounded(1.0) +
                            pRng->bounded(1.0) - 1.5) * dSigma * 2.0;
            int nN = static_cast<int>(dNoise);
            int nR = std::clamp(qRed(pLine[x]) + nN, 0, 255);
            int nG = std::clamp(qGreen(pLine[x]) + nN, 0, 255);
            int nB = std::clamp(qBlue(pLine[x]) + nN, 0, 255);
            pLine[x] = qRgba(nR, nG, nB, qAlpha(pLine[x]));
        }
    }
}

// 20260324 ZJH 饱和度/色调抖动（原地修改 HSV 通道）
// 参数: img - 待修改的 ARGB32 图像引用
//       dSatFactor - 饱和度倍率（>1 增强，<1 降低）
//       dHueShift - 色调偏移量（-0.5~0.5 范围）
void AugmentationPreviewDialog::adjustSaturationHue(QImage& img, double dSatFactor, double dHueShift) const
{
    for (int y = 0; y < img.height(); ++y) {
        QRgb* pLine = reinterpret_cast<QRgb*>(img.scanLine(y));  // 20260324 ZJH scanLine 直接操作像素
        for (int x = 0; x < img.width(); ++x) {
            QColor color(pLine[x]);  // 20260324 ZJH 从 QRgb 构建 QColor
            double dH = color.hueF();           // 20260324 ZJH 色调 0~1
            double dS = color.saturationF();     // 20260324 ZJH 饱和度 0~1
            double dV = color.valueF();          // 20260324 ZJH 明度 0~1

            // 20260324 ZJH 调整饱和度
            dS = std::clamp(dS * dSatFactor, 0.0, 1.0);

            // 20260324 ZJH 调整色调（循环映射到 0~1）
            dH = dH + dHueShift;  // 20260324 ZJH 色调偏移
            if (dH < 0.0) dH += 1.0;  // 20260324 ZJH 负值回绕
            if (dH > 1.0) dH -= 1.0;  // 20260324 ZJH 超值回绕

            // 20260324 ZJH 重建颜色
            QColor newColor = QColor::fromHsvF(dH, dS, dV, color.alphaF());
            pLine[x] = newColor.rgba();  // 20260324 ZJH 写回像素
        }
    }
}

// 20260324 ZJH 随机擦除：在图像上填充灰色矩形
// 参数: img - 待修改的图像引用
//       dRatio - 最大擦除面积占总面积的比例
void AugmentationPreviewDialog::applyRandomErasing(QImage& img, double dRatio) const
{
    QRandomGenerator* pRng = QRandomGenerator::global();  // 20260324 ZJH 全局随机数生成器

    int nW = img.width();   // 20260324 ZJH 图像宽度
    int nH = img.height();  // 20260324 ZJH 图像高度
    double dArea = nW * nH * dRatio;  // 20260324 ZJH 最大擦除面积
    double dAspect = 0.5 + pRng->bounded(1.5);  // 20260324 ZJH 宽高比 0.5~2.0

    // 20260324 ZJH 计算擦除矩形尺寸
    int nEraseH = static_cast<int>(std::sqrt(dArea / dAspect));  // 20260324 ZJH 擦除高度
    int nEraseW = static_cast<int>(nEraseH * dAspect);           // 20260324 ZJH 擦除宽度
    nEraseH = std::min(nEraseH, nH);  // 20260324 ZJH 限制不超过图像高度
    nEraseW = std::min(nEraseW, nW);  // 20260324 ZJH 限制不超过图像宽度

    if (nEraseW <= 0 || nEraseH <= 0) return;  // 20260324 ZJH 尺寸无效直接返回

    // 20260324 ZJH 随机确定擦除位置
    int nX = pRng->bounded(nW - nEraseW + 1);  // 20260324 ZJH 左上角 X
    int nY = pRng->bounded(nH - nEraseH + 1);  // 20260324 ZJH 左上角 Y

    // 20260324 ZJH 填充灰色像素（128, 128, 128）
    QRgb grayPixel = qRgb(128, 128, 128);  // 20260324 ZJH 灰色填充值
    for (int y = nY; y < nY + nEraseH; ++y) {
        QRgb* pLine = reinterpret_cast<QRgb*>(img.scanLine(y));  // 20260324 ZJH scanLine 直接操作
        for (int x = nX; x < nX + nEraseW; ++x) {
            pLine[x] = grayPixel;  // 20260324 ZJH 填充灰色
        }
    }
}

// 20260324 ZJH 随机缩放裁剪：裁剪后缩放回原尺寸
// 参数: img - 输入图像
//       dMinScale - 最小缩放比例（如 0.7 表示裁剪区域最少为原始面积的 70%）
// 返回: 裁剪并缩放后的图像（尺寸与原图相同）
QImage AugmentationPreviewDialog::applyRandomCrop(const QImage& img, double dMinScale) const
{
    QRandomGenerator* pRng = QRandomGenerator::global();  // 20260324 ZJH 随机数生成器

    int nW = img.width();   // 20260324 ZJH 原始宽度
    int nH = img.height();  // 20260324 ZJH 原始高度

    // 20260324 ZJH 随机缩放因子在 [dMinScale, 1.0] 之间
    double dScale = dMinScale + pRng->bounded(1.0 - dMinScale);
    int nCropW = static_cast<int>(nW * dScale);  // 20260324 ZJH 裁剪宽度
    int nCropH = static_cast<int>(nH * dScale);  // 20260324 ZJH 裁剪高度

    if (nCropW <= 0 || nCropH <= 0) return img;  // 20260324 ZJH 安全检查

    // 20260324 ZJH 随机裁剪位置
    int nX = pRng->bounded(std::max(1, nW - nCropW + 1));  // 20260324 ZJH 左上角 X
    int nY = pRng->bounded(std::max(1, nH - nCropH + 1));  // 20260324 ZJH 左上角 Y

    // 20260324 ZJH 裁剪并缩放回原始尺寸
    QImage imgCropped = img.copy(nX, nY, nCropW, nCropH);  // 20260324 ZJH 裁剪
    return imgCropped.scaled(nW, nH, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);  // 20260324 ZJH 缩放回原尺寸
}

// 20260324 ZJH 高斯模糊近似（3x3 均值模糊多次叠加）
// 参数: img - 待修改的图像引用
//       dSigma - 目标 sigma（叠加 3x3 次数 = round(sigma)，至少 1 次）
void AugmentationPreviewDialog::applyGaussianBlur(QImage& img, double dSigma) const
{
    int nPasses = std::max(1, static_cast<int>(std::round(dSigma)));  // 20260324 ZJH 模糊叠加次数

    for (int pass = 0; pass < nPasses; ++pass) {
        QImage imgCopy = img.copy();  // 20260324 ZJH 拷贝用于读取邻域像素

        for (int y = 1; y < img.height() - 1; ++y) {
            QRgb* pDstLine = reinterpret_cast<QRgb*>(img.scanLine(y));  // 20260324 ZJH 目标行

            for (int x = 1; x < img.width() - 1; ++x) {
                int nSumR = 0, nSumG = 0, nSumB = 0;  // 20260324 ZJH 通道累加器

                // 20260324 ZJH 3x3 邻域求和
                for (int dy = -1; dy <= 1; ++dy) {
                    const QRgb* pSrcLine = reinterpret_cast<const QRgb*>(imgCopy.constScanLine(y + dy));
                    for (int dx = -1; dx <= 1; ++dx) {
                        QRgb px = pSrcLine[x + dx];  // 20260324 ZJH 邻域像素
                        nSumR += qRed(px);
                        nSumG += qGreen(px);
                        nSumB += qBlue(px);
                    }
                }

                // 20260324 ZJH 取平均值（3x3 = 9 个像素）
                pDstLine[x] = qRgba(nSumR / 9, nSumG / 9, nSumB / 9, qAlpha(pDstLine[x]));
            }
        }
    }
}

// 20260324 ZJH 仿射变换（剪切+平移）
// 参数: img - 输入图像
//       dShearDeg - 最大剪切角度（度）
//       dTranslate - 最大平移比例
// 返回: 变换后的图像
QImage AugmentationPreviewDialog::applyAffine(const QImage& img, double dShearDeg, double dTranslate) const
{
    QRandomGenerator* pRng = QRandomGenerator::global();  // 20260324 ZJH 随机数生成器

    // 20260324 ZJH 随机剪切角度在 [-dShearDeg, +dShearDeg] 之间
    double dShear = (pRng->bounded(2.0) - 1.0) * dShearDeg;
    double dShearRad = dShear * 3.14159265 / 180.0;  // 20260324 ZJH 角度转弧度
    double dShearFactor = std::tan(dShearRad);         // 20260324 ZJH 剪切因子

    // 20260324 ZJH 随机平移像素偏移
    double dTxFrac = (pRng->bounded(2.0) - 1.0) * dTranslate;  // 20260324 ZJH X 方向比例
    double dTyFrac = (pRng->bounded(2.0) - 1.0) * dTranslate;  // 20260324 ZJH Y 方向比例
    double dTx = dTxFrac * img.width();   // 20260324 ZJH X 方向像素偏移
    double dTy = dTyFrac * img.height();  // 20260324 ZJH Y 方向像素偏移

    // 20260324 ZJH 构建 QTransform（剪切 + 平移）
    QTransform transform;
    transform.shear(dShearFactor, 0);  // 20260324 ZJH 水平剪切
    transform.translate(dTx, dTy);     // 20260324 ZJH 平移

    return img.transformed(transform, Qt::SmoothTransformation);  // 20260324 ZJH 应用变换
}

// 20260323 ZJH Getter 方法
double AugmentationPreviewDialog::brightnessRange() const { return m_pSldBrightness->value() / 100.0; }
double AugmentationPreviewDialog::flipProbability() const { return m_pSldFlipProb->value() / 100.0; }
double AugmentationPreviewDialog::rotationRange() const { return m_pSldRotation->value(); }
bool AugmentationPreviewDialog::isBrightnessEnabled() const { return m_pChkBrightness->isChecked(); }
bool AugmentationPreviewDialog::isFlipEnabled() const { return m_pChkFlip->isChecked(); }
bool AugmentationPreviewDialog::isRotationEnabled() const { return m_pChkRotation->isChecked(); }
bool AugmentationPreviewDialog::isCropEnabled() const { return m_pChkCrop->isChecked(); }
bool AugmentationPreviewDialog::isNoiseEnabled() const { return m_pChkNoise->isChecked(); }

// 20260324 ZJH 新增 Getter 方法
bool AugmentationPreviewDialog::isVerticalFlipEnabled() const { return m_pChkVerticalFlip->isChecked(); }
bool AugmentationPreviewDialog::isSaturationEnabled() const { return m_pChkSaturation->isChecked(); }
bool AugmentationPreviewDialog::isRandomErasingEnabled() const { return m_pChkRandomErasing->isChecked(); }
bool AugmentationPreviewDialog::isGaussianBlurEnabled() const { return m_pChkGaussianBlur->isChecked(); }
bool AugmentationPreviewDialog::isAffineEnabled() const { return m_pChkAffine->isChecked(); }
