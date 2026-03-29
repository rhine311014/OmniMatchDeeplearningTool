// 20260323 ZJH HelpDialog — 帮助对话框实现

#include "ui/dialogs/HelpDialog.h"

#include <QVBoxLayout>      // 20260323 ZJH 垂直布局
#include <QLabel>           // 20260323 ZJH 标签
#include <QTextBrowser>     // 20260323 ZJH 富文本浏览器
#include <QTableWidget>     // 20260323 ZJH 表格控件
#include <QHeaderView>      // 20260323 ZJH 表格头
#include <QDialogButtonBox> // 20260323 ZJH 按钮组

// 20260323 ZJH 构造函数
HelpDialog::HelpDialog(QWidget* pParent)
    : QDialog(pParent)
{
    setWindowTitle("OmniMatch 帮助");  // 20260324 ZJH 窗口标题汉化
    setMinimumSize(600, 500);

    // 20260323 ZJH 暗色主题
    setStyleSheet("QDialog { background: #1e2028; color: #e2e8f0; }"
                  "QTabWidget::pane { border: 1px solid #334155; background: #1e2028; }"
                  "QTabBar::tab { background: #22262e; color: #94a3b8; padding: 8px 20px; border: 1px solid #334155; }"
                  "QTabBar::tab:selected { background: #2563eb; color: white; }"
                  "QTextBrowser { background: #22262e; color: #e2e8f0; border: none; padding: 12px; }"
                  "QTableWidget { background: #22262e; color: #e2e8f0; border: none; gridline-color: #334155; }"
                  "QHeaderView::section { background: #1a1d24; color: #94a3b8; padding: 6px; border: 1px solid #334155; }");

    QVBoxLayout* pLayout = new QVBoxLayout(this);

    // 20260323 ZJH 标签页
    QTabWidget* pTabs = new QTabWidget;
    pTabs->addTab(createGuideTab(), "用户指南");      // 20260324 ZJH 标签页标题汉化
    pTabs->addTab(createShortcutsTab(), "快捷键");    // 20260324 ZJH 标签页标题汉化
    pTabs->addTab(createAboutTab(), "关于");           // 20260324 ZJH 标签页标题汉化
    pLayout->addWidget(pTabs);

    // 20260323 ZJH 关闭按钮
    QDialogButtonBox* pBtnBox = new QDialogButtonBox(QDialogButtonBox::Close);
    connect(pBtnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    pLayout->addWidget(pBtnBox);
}

// 20260323 ZJH 使用指南标签页
QWidget* HelpDialog::createGuideTab()
{
    QTextBrowser* pBrowser = new QTextBrowser;
    pBrowser->setOpenExternalLinks(true);

    // 20260324 ZJH 用户指南内容全部汉化
    QString strHtml;
    strHtml += "<h2 style='color: #3b82f6;'>OmniMatch 快速入门指南</h2>";
    strHtml += "<h3>1. 创建项目</h3>";
    strHtml += "<p>在<b>项目</b>页面点击<b>新建项目</b>。选择任务类型（分类、"
               "目标检测、分割、异常检测等），并设置项目目录。</p>";
    strHtml += "<h3>2. 导入图像</h3>";
    strHtml += "<p>切换到<b>图库</b>页面。使用工具栏导入图像或文件夹，"
               "也可以直接拖拽文件到图库中。</p>";
    strHtml += "<h3>3. 标注图像</h3>";
    strHtml += "<p>双击图像打开<b>图像</b>页面。使用标注工具"
               "（矩形、多边形、笔刷）标注感兴趣区域，从下拉菜单中选择标签。</p>";
    strHtml += "<h3>4. 划分数据集</h3>";
    strHtml += "<p>进入<b>划分</b>页面。设置训练/验证/测试比例并点击执行。"
               "分层采样确保类别分布均衡。</p>";
    strHtml += "<h3>5. 训练模型</h3>";
    strHtml += "<p>在<b>训练</b>页面，选择模型架构，配置超参数，"
               "点击开始训练。实时监控损失曲线。</p>";
    strHtml += "<h3>6. 评估</h3>";
    strHtml += "<p>训练完成后，切换到<b>评估</b>页面运行模型评估。"
               "查看准确率、混淆矩阵、ROC 曲线和其他指标。</p>";
    strHtml += "<h3>7. 导出</h3>";
    strHtml += "<p>在<b>导出</b>页面将训练好的模型导出为 ONNX、TensorRT 或其他格式。</p>";

    pBrowser->setHtml(strHtml);
    return pBrowser;
}

// 20260323 ZJH 快捷键标签页
QWidget* HelpDialog::createShortcutsTab()
{
    QWidget* pTab = new QWidget;
    QVBoxLayout* pLayout = new QVBoxLayout(pTab);

    QTableWidget* pTable = new QTableWidget;
    pTable->setColumnCount(2);
    pTable->setHorizontalHeaderLabels({"快捷键", "操作"});  // 20260324 ZJH 表头汉化
    pTable->horizontalHeader()->setStretchLastSection(true);
    pTable->verticalHeader()->hide();
    pTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    pTable->setSelectionMode(QAbstractItemView::NoSelection);

    // 20260323 ZJH 快捷键数据
    struct ShortcutEntry { QString strKey; QString strAction; };
    // 20260324 ZJH 快捷键操作描述全部汉化
    QVector<ShortcutEntry> vecShortcuts = {
        {"Ctrl+N",   "新建项目"},
        {"Ctrl+O",   "打开项目"},
        {"Ctrl+S",   "保存项目"},
        {"Ctrl+W",   "关闭项目"},
        {"Ctrl+Q",   "退出应用程序"},
        {"Ctrl+Z",   "撤销"},
        {"Ctrl+Y",   "重做"},
        {"Alt+1~8",  "切换到页面 1-8"},
        {"F1",       "显示快捷键帮助"},
        {"F11",      "切换全屏"},
        {"V",        "选择工具"},
        {"B",        "矩形工具"},
        {"P",        "多边形工具"},
        {"D",        "笔刷工具"},
        {"Delete",   "删除选中标注"},
        {"Ctrl+0",   "适应视图"},
        {"Ctrl+1",   "实际大小 (100%)"},
    };

    pTable->setRowCount(vecShortcuts.size());
    for (int i = 0; i < vecShortcuts.size(); ++i) {
        pTable->setItem(i, 0, new QTableWidgetItem(vecShortcuts[i].strKey));
        pTable->setItem(i, 1, new QTableWidgetItem(vecShortcuts[i].strAction));
    }

    pTable->resizeColumnsToContents();
    pLayout->addWidget(pTable);

    return pTab;
}

// 20260323 ZJH 关于标签页
QWidget* HelpDialog::createAboutTab()
{
    QWidget* pTab = new QWidget;
    QVBoxLayout* pLayout = new QVBoxLayout(pTab);
    pLayout->setAlignment(Qt::AlignCenter);

    // 20260323 ZJH 产品名称
    QLabel* pLblTitle = new QLabel("OmniMatch");
    pLblTitle->setStyleSheet("font-size: 28px; font-weight: bold; color: #3b82f6;");
    pLblTitle->setAlignment(Qt::AlignCenter);
    pLayout->addWidget(pLblTitle);

    // 20260324 ZJH 版本号（使用 CMake 注入的 OM_VERSION 宏），汉化版本前缀
    QLabel* pLblVersion = new QLabel(QStringLiteral("版本 ") + QStringLiteral(OM_VERSION));
    pLblVersion->setStyleSheet("font-size: 14px; color: #94a3b8;");
    pLblVersion->setAlignment(Qt::AlignCenter);
    pLayout->addWidget(pLblVersion);

    pLayout->addSpacing(20);

    // 20260324 ZJH 描述汉化
    QLabel* pLblDesc = new QLabel(
        "纯 C++ 深度学习视觉平台\n\n"
        "面向工业视觉的全流程深度学习工具，\n"
        "基于 C++23 模块、Qt6 Widgets\n"
        "和自研张量/自动微分引擎从零构建。\n\n"
        "支持：分类、目标检测、分割、\n"
        "异常检测、OCR、实例分割、\n"
        "零样本检测和零样本异常检测。"
    );
    pLblDesc->setAlignment(Qt::AlignCenter);
    pLblDesc->setStyleSheet("color: #e2e8f0; line-height: 1.6;");
    pLayout->addWidget(pLblDesc);

    pLayout->addSpacing(20);

    // 20260324 ZJH 技术栈汉化
    QLabel* pLblTech = new QLabel(
        "技术栈: C++23 | Qt 6.10.1 | CMake 3.30+ | MSVC 14.50\n"
        "引擎: 自研 Tensor + Autograd + CNN/RNN/ViT/GAN/UNet"
    );
    pLblTech->setAlignment(Qt::AlignCenter);
    pLblTech->setStyleSheet("color: #64748b; font-size: 11px;");
    pLayout->addWidget(pLblTech);

    pLayout->addStretch();

    return pTab;
}
