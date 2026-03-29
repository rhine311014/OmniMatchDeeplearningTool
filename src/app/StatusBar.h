#pragma once
// 20260322 ZJH 底部状态栏组件 — OmniMatch 操作反馈与进度展示
// 布局：左侧消息文本 | 弹性空间 | 标注进度标签 | 进度条（隐藏/显示） | GPU 信息标签
// 高度固定 28px，背景色 #13151a（深黑色，比主背景更深，形成视觉层次）

#include <QWidget>
#include <QString>

// 20260322 ZJH 前向声明，避免在头文件中引入实现头
class QLabel;
class QProgressBar;
class QHBoxLayout;

namespace OmniMatch {

// 20260322 ZJH 底部状态栏，统一显示操作消息、标注进度、训练进度和 GPU 信息
class StatusBar : public QWidget
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数，初始化全部子控件和布局
    explicit StatusBar(QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数，子控件由 Qt 对象树管理
    ~StatusBar() override;

public slots:
    // 20260322 ZJH 设置左侧操作消息文本（如 "正在加载数据集..."）
    void setMessage(const QString& strMsg);

    // 20260322 ZJH 设置标注进度文本（如 "已标注图像: 80/100 (80%)"）
    void setLabelProgress(const QString& strInfo);

    // 20260322 ZJH 设置并显示训练/推理进度条（0-100），-1 表示隐藏进度条
    void setProgress(int nPercent);

    // 20260322 ZJH 清空所有状态信息，隐藏进度条（回到空白初始状态）
    void clearMessage();

    // 20260322 ZJH 更新右侧 GPU 信息文本（如 "GPU: RTX 4090 | VRAM 8.2/24 GB"）
    void setGpuInfo(const QString& strInfo);

private:
    // 20260322 ZJH 创建并布局全部子控件
    void setupUI();

    QLabel*      m_pLblMessage   = nullptr;  // 左侧操作消息标签
    QLabel*      m_pLblProgress  = nullptr;  // 标注进度文本标签（进度条左侧）
    QProgressBar* m_pProgressBar = nullptr;  // 训练/推理进度条，默认隐藏
    QLabel*      m_pLblGpuInfo   = nullptr;  // 最右侧 GPU 显存/算力信息标签
};

}  // namespace OmniMatch
