// 20260322 ZJH 底部状态栏实现
// 水平排列：消息 | 弹性间距 | 标注进度 | 进度条 | GPU信息
// 固定高度 28px，背景色 #13151a

#include "StatusBar.h"
#include <QLabel>        // 20260322 ZJH 消息/进度/GPU 文本标签
#include <QProgressBar>  // 20260322 ZJH 训练推理进度条
#include <QHBoxLayout>   // 20260322 ZJH 水平布局

namespace OmniMatch {

// 20260322 ZJH 状态栏固定高度（像素）
static constexpr int s_nStatusBarHeight = 28;

// 20260322 ZJH 构造函数，调用 setupUI 完成全部初始化
StatusBar::StatusBar(QWidget* pParent)
    : QWidget(pParent)
{
    setupUI();  // 20260322 ZJH 创建子控件、布局、样式
}

// 20260322 ZJH 析构函数，子控件由 Qt 对象树自动释放
StatusBar::~StatusBar()
{
}

// 20260322 ZJH 初始化状态栏界面布局和所有子控件
void StatusBar::setupUI()
{
    // 20260322 ZJH 固定高度，宽度跟随父容器自适应
    setFixedHeight(s_nStatusBarHeight);
    setObjectName("statusBar");  // 20260322 ZJH 供 QSS 精确选择样式

    // 20260322 ZJH 设置背景色 #13151a（深黑，比主背景 #0d1117 略浅，形成底部层次感）
    setStyleSheet(QStringLiteral(
        "#statusBar {"
        "  background-color: #13151a;"
        "  border-top: 1px solid #1e2230;"  // 20260322 ZJH 顶部 1px 分割线
        "}"
    ));

    // 20260322 ZJH 水平布局，紧凑边距
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    pLayout->setContentsMargins(10, 2, 10, 2);  // 20260322 ZJH 左右 10px，上下 2px 边距
    pLayout->setSpacing(12);                     // 20260322 ZJH 控件间距 12px

    // 20260322 ZJH 左侧消息标签 — 显示当前操作状态提示
    m_pLblMessage = new QLabel(this);
    m_pLblMessage->setObjectName("statusMessage");  // 20260322 ZJH QSS 选择器
    m_pLblMessage->setStyleSheet(QStringLiteral(
        "#statusMessage { color: #94a3b8; font-size: 12px; }"  // 20260322 ZJH 灰色小字
    ));
    pLayout->addWidget(m_pLblMessage);           // 20260322 ZJH 靠左添加

    // 20260322 ZJH 中间弹性空间，将右侧控件推到末尾
    pLayout->addStretch(1);

    // 20260322 ZJH 标注进度标签 — 显示图像标注完成率
    m_pLblProgress = new QLabel(this);
    m_pLblProgress->setObjectName("labelProgress");  // 20260322 ZJH QSS 选择器
    m_pLblProgress->setStyleSheet(QStringLiteral(
        "#labelProgress { color: #64748b; font-size: 12px; }"  // 20260322 ZJH 偏暗灰色
    ));
    pLayout->addWidget(m_pLblProgress);

    // 20260322 ZJH 训练/推理进度条 — 默认隐藏，需要时通过 setProgress() 显示
    m_pProgressBar = new QProgressBar(this);
    m_pProgressBar->setFixedWidth(160);      // 20260322 ZJH 固定宽度 160px，不占用过多空间
    m_pProgressBar->setFixedHeight(14);      // 20260322 ZJH 固定高度 14px，与状态栏高度协调
    m_pProgressBar->setRange(0, 100);        // 20260322 ZJH 百分比范围 0-100
    m_pProgressBar->setValue(0);             // 20260322 ZJH 初始值 0
    m_pProgressBar->setTextVisible(true);    // 20260322 ZJH 显示百分比文字
    m_pProgressBar->setObjectName("statusProgressBar");  // 20260322 ZJH QSS 选择器
    m_pProgressBar->setStyleSheet(QStringLiteral(
        "#statusProgressBar {"
        "  background-color: #1e2230;"          // 20260322 ZJH 进度条底色
        "  border: none;"
        "  border-radius: 3px;"
        "  color: #ffffff;"
        "  font-size: 11px;"
        "}"
        "#statusProgressBar::chunk {"
        "  background-color: #2563eb;"          // 20260322 ZJH 进度填充蓝色
        "  border-radius: 3px;"
        "}"
    ));
    m_pProgressBar->hide();                  // 20260322 ZJH 初始隐藏，仅在训练时显示
    pLayout->addWidget(m_pProgressBar);

    // 20260322 ZJH GPU 信息标签 — 显示显卡型号和显存使用量
    m_pLblGpuInfo = new QLabel(this);
    m_pLblGpuInfo->setObjectName("gpuInfo");  // 20260322 ZJH QSS 选择器
    m_pLblGpuInfo->setStyleSheet(QStringLiteral(
        "#gpuInfo { color: #4ade80; font-size: 12px; }"  // 20260322 ZJH 绿色，对应 GPU 运行状态
    ));
    pLayout->addWidget(m_pLblGpuInfo);        // 20260322 ZJH 最右侧
}

// 20260322 ZJH 设置左侧消息文本（如操作提示、错误信息等）
void StatusBar::setMessage(const QString& strMsg)
{
    m_pLblMessage->setText(strMsg);  // 20260322 ZJH 直接设置 QLabel 文本
}

// 20260322 ZJH 设置标注进度文本（如 "已标注: 60/100"）
void StatusBar::setLabelProgress(const QString& strInfo)
{
    m_pLblProgress->setText(strInfo);  // 20260322 ZJH 更新标注进度标签内容
}

// 20260322 ZJH 设置进度条数值并控制显示/隐藏
// 参数 nPercent: 0-100 显示并更新进度；<0 则隐藏进度条
void StatusBar::setProgress(int nPercent)
{
    if (nPercent < 0) {
        // 20260322 ZJH 负值表示操作结束，隐藏进度条以节省状态栏空间
        m_pProgressBar->hide();
    } else {
        // 20260322 ZJH 正常范围，Qt 内部会将超出 0-100 的值 clamp 到边界
        m_pProgressBar->setValue(nPercent);
        m_pProgressBar->show();  // 20260322 ZJH 显示进度条
    }
}

// 20260322 ZJH 清空全部状态信息，回到初始空白状态
void StatusBar::clearMessage()
{
    m_pLblMessage->clear();    // 20260322 ZJH 清空操作消息
    m_pLblProgress->clear();   // 20260322 ZJH 清空标注进度文本
    m_pLblGpuInfo->clear();    // 20260322 ZJH 清空 GPU 信息
    m_pProgressBar->hide();    // 20260322 ZJH 隐藏进度条
    m_pProgressBar->setValue(0);  // 20260322 ZJH 重置进度条数值
}

// 20260322 ZJH 更新右侧 GPU 信息显示文本
void StatusBar::setGpuInfo(const QString& strInfo)
{
    m_pLblGpuInfo->setText(strInfo);  // 20260322 ZJH 直接更新 GPU 信息标签
}

}  // namespace OmniMatch
