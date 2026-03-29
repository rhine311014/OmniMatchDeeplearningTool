// 20260323 ZJH ModelComplexityWidget — 模型复杂度信息控件
// 显示模型的参数量、FLOPs、内存占用、层数等关键指标
// 用于训练页和导出页的右侧面板
#pragma once

#include <QWidget>   // 20260323 ZJH 控件基类
#include <QString>   // 20260323 ZJH 字符串

// 20260323 ZJH 模型复杂度信息
struct ModelComplexityInfo
{
    QString strArchitecture;       // 20260323 ZJH 模型架构名称
    qint64 nTotalParams = 0;      // 20260323 ZJH 总参数量
    qint64 nTrainableParams = 0;  // 20260323 ZJH 可训练参数量
    double dFLOPs = 0.0;          // 20260323 ZJH 浮点运算量（GFLOPs）
    double dMemoryMB = 0.0;       // 20260323 ZJH 预估内存占用（MB）
    int nLayers = 0;              // 20260323 ZJH 层数
    int nInputSize = 224;         // 20260323 ZJH 输入尺寸
};

// 20260323 ZJH 模型复杂度信息控件
class ModelComplexityWidget : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit ModelComplexityWidget(QWidget* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~ModelComplexityWidget() override = default;

    // 20260323 ZJH 设置模型复杂度信息
    void setInfo(const ModelComplexityInfo& info);

    // 20260323 ZJH 清空显示
    void clear();

protected:
    // 20260323 ZJH 绘制事件
    void paintEvent(QPaintEvent* pEvent) override;

private:
    // 20260323 ZJH 格式化参数量为人类可读字符串（如 "11.2M", "1.5B"）
    static QString formatParams(qint64 nParams);

    // 20260323 ZJH 格式化 FLOPs
    static QString formatFLOPs(double dFLOPs);

    ModelComplexityInfo m_info;   // 20260323 ZJH 当前模型信息
    bool m_bHasData = false;     // 20260323 ZJH 是否有数据
};
