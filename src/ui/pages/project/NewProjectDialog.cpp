// 20260403 ZJH NewProjectDialog 实现 — Halcon 风格新建项目对话框
// 卡片网格选择任务类型 + 右侧描述面板 + 底部项目信息表单
// 参考 MVTec Deep Learning Tool / Halcon DL Tool 24.05 UI 设计

#include "ui/pages/project/NewProjectDialog.h"  // 20260403 ZJH 类声明

#include <QVBoxLayout>      // 20260403 ZJH 垂直布局
#include <QHBoxLayout>      // 20260403 ZJH 水平布局
#include <QGridLayout>      // 20260403 ZJH 网格布局（任务卡片）
#include <QFileDialog>      // 20260403 ZJH 文件保存对话框
#include <QMessageBox>      // 20260403 ZJH 输入验证错误提示
#include <QStandardPaths>   // 20260403 ZJH 获取文档目录作为默认路径
#include <QFrame>           // 20260403 ZJH 分隔线
#include <QScrollArea>      // 20260403 ZJH 卡片区域滚动

// 20260403 ZJH 构造函数：初始化 Halcon 风格对话框 UI
NewProjectDialog::NewProjectDialog(QWidget* pParent)
    : QDialog(pParent)                // 20260403 ZJH 初始化 QDialog 基类
    , m_nSelectedIndex(0)             // 20260403 ZJH 默认选中第一张卡片（分类）
    , m_pLblDescTitle(nullptr)        // 20260403 ZJH 描述面板标题
    , m_pLblDescBody(nullptr)         // 20260403 ZJH 描述面板正文
    , m_pEdtName(nullptr)             // 20260403 ZJH 项目名称
    , m_pEdtPath(nullptr)             // 20260403 ZJH 项目路径
    , m_pBtnBrowse(nullptr)           // 20260403 ZJH 浏览按钮
    , m_pEdtDescription(nullptr)      // 20260403 ZJH 项目描述
    , m_pChkRelativePaths(nullptr)    // 20260403 ZJH 相对路径复选框
    , m_pBtnCancel(nullptr)           // 20260403 ZJH 取消按钮
    , m_pBtnCreate(nullptr)           // 20260403 ZJH 创建按钮
{
    initTaskCards();  // 20260403 ZJH 初始化 10 种任务类型卡片数据
    setupUi();        // 20260403 ZJH 创建并组装 UI 控件
}

// 20260403 ZJH 初始化任务类型卡片数据（10 种任务类型）
// 每种任务包含：中文标题、副标题、描述面板标题、详细说明
void NewProjectDialog::initTaskCards()
{
    m_vecCardInfos = {
        // 20260403 ZJH 1. 图像分类
        {
            om::TaskType::Classification,
            QStringLiteral("分类"),
            QString(),
            QStringLiteral("分类"),
            QStringLiteral("将图像分配到预定义的类别中。\n\n"
                           "分类是最基本的深度学习任务，模型学习将整幅图像映射到一个类别标签。"
                           "适用于产品质量分级、缺陷类型识别、物料分拣等场景。")
        },
        // 20260403 ZJH 2. 异常检测
        {
            om::TaskType::AnomalyDetection,
            QStringLiteral("异常检测"),
            QStringLiteral("全局上下文异常值检测 (Global\nContext Anomaly Detection)"),
            QStringLiteral("异常检测（全局上下文异常值检测）"),
            QStringLiteral("检测图像中与正常样本不同的异常区域。\n\n"
                           "异常检测仅需正常样本进行训练，模型学习正常模式后，能自动识别偏离正常的区域。"
                           "适用于表面缺陷检测、纹理异常检测等无法预知所有缺陷类型的场景。")
        },
        // 20260403 ZJH 3. 目标检测 - 轴对齐矩形
        {
            om::TaskType::ObjectDetection,
            QStringLiteral("对象检测"),
            QStringLiteral("轴对齐矩形"),
            QStringLiteral("对象检测（轴对齐矩形）"),
            QStringLiteral("检测给定类别的对象并在图像中对应定位。\n\n"
                           "目标检测使用轴对齐的矩形边界框标记对象位置和类别。"
                           "适用于零件计数、装配完整性检查、多目标定位等场景。")
        },
        // 20260403 ZJH 4. 语义分割
        {
            om::TaskType::SemanticSegmentation,
            QStringLiteral("语义分割"),
            QString(),
            QStringLiteral("语义分割"),
            QStringLiteral("对图像中的每个像素进行分类。\n\n"
                           "语义分割为图像中每个像素赋予一个类别标签，但不区分同类的不同实例。"
                           "适用于缺陷区域精确测量、材料区域分割、表面覆盖率计算等场景。")
        },
        // 20260403 ZJH 5. 实例分割
        {
            om::TaskType::InstanceSegmentation,
            QStringLiteral("实例分割"),
            QStringLiteral("轴对齐边界框"),
            QStringLiteral("实例分割（带轴对齐边界框）"),
            QStringLiteral("检测给定类别的对象并在图像中对应定位。\n\n"
                           "实例分割是对象检测的一种特殊情况，其中模型还会预测不同的对象实例，并"
                           "将找到的实例分配到图像内的对应区域。")
        },
        // 20260403 ZJH 6. Deep OCR
        {
            om::TaskType::DeepOCR,
            QStringLiteral("Deep OCR"),
            QString(),
            QStringLiteral("Deep OCR"),
            QStringLiteral("检测并识别图像中的文本内容。\n\n"
                           "Deep OCR 结合文本检测和字符识别两个阶段，"
                           "能够定位图像中任意方向的文本区域并识别其内容。"
                           "适用于印刷质量检查、序列号读取、标签验证等场景。")
        },
        // 20260403 ZJH 7. 零样本缺陷检测
        {
            om::TaskType::ZeroShotDefectDetection,
            QStringLiteral("零样本缺陷检测"),
            QStringLiteral("CLIP 文本引导"),
            QStringLiteral("零样本缺陷检测（CLIP 文本引导）"),
            QStringLiteral("无需训练，使用文本描述驱动缺陷检测。\n\n"
                           "基于 CLIP 视觉-语言模型，通过自然语言描述目标缺陷类型即可进行检测，"
                           "无需收集和标注缺陷样本。适用于新产品快速上线、小批量多品种场景。")
        },
        // 20260403 ZJH 8. 零样本目标检测
        {
            om::TaskType::ZeroShotObjectDetection,
            QStringLiteral("零样本目标检测"),
            QStringLiteral("开放词汇检测"),
            QStringLiteral("零样本目标检测（开放词汇检测）"),
            QStringLiteral("无需训练，使用文本描述驱动目标检测。\n\n"
                           "基于开放词汇检测模型，通过文本描述即可检测任意类别的对象，"
                           "无需针对特定类别进行训练。适用于通用物体检测、快速原型验证等场景。")
        },
        // 20260403 ZJH 9. 图像检索
        {
            om::TaskType::ImageRetrieval,
            QStringLiteral("图像检索"),
            QStringLiteral("特征嵌入搜索"),
            QStringLiteral("图像检索（特征嵌入搜索）"),
            QStringLiteral("通过特征相似度搜索匹配图像。\n\n"
                           "使用深度学习提取图像特征嵌入向量，通过余弦相似度搜索最相似的图像。"
                           "适用于相似缺陷查找、产品型号匹配、历史案例检索等场景。")
        },
        // 20260403 ZJH 10. 无监督分割
        {
            om::TaskType::UnsupervisedSegmentation,
            QStringLiteral("无监督分割"),
            QStringLiteral("自动分区"),
            QStringLiteral("无监督分割（自动分区）"),
            QStringLiteral("无需标注数据，自动将图像分割为语义区域。\n\n"
                           "基于 SAM 等基础模型，无需人工标注即可将图像自动分割为有意义的区域。"
                           "适用于未知场景的快速分析、自动标注辅助、区域发现等场景。")
        }
    };
}

// 20260403 ZJH 初始化对话框布局和控件
void NewProjectDialog::setupUi()
{
    // 20260403 ZJH 设置对话框标题和大小
    setWindowTitle(QStringLiteral("创建新项目"));
    setMinimumSize(1100, 750);  // 20260403 ZJH 最小尺寸，匹配 Halcon 风格宽屏布局
    resize(1100, 780);          // 20260403 ZJH 默认尺寸

    // 20260403 ZJH 设置对话框暗色主题样式（与项目整体主题一致）
    setStyleSheet(QStringLiteral(
        "QDialog {"
        "  background-color: #1a1d23;"    // 20260403 ZJH 深色背景
        "  color: #e2e8f0;"               // 20260403 ZJH 浅色文字
        "}"
        "QLabel {"
        "  color: #94a3b8;"               // 20260403 ZJH 标签灰色文字
        "  background: transparent;"      // 20260403 ZJH 标签透明背景
        "}"
        "QLineEdit {"
        "  background-color: #262a35;"    // 20260403 ZJH 输入框暗色背景
        "  color: #e2e8f0;"               // 20260403 ZJH 输入文字颜色
        "  border: 1px solid #3b4252;"    // 20260403 ZJH 边框
        "  border-radius: 4px;"
        "  padding: 6px 10px;"
        "  font-size: 10pt;"
        "}"
        "QLineEdit:focus {"
        "  border-color: #00bcd4;"        // 20260403 ZJH 聚焦时青色边框（匹配强调色）
        "}"
        "QTextEdit {"
        "  background-color: #262a35;"    // 20260403 ZJH 多行输入暗色背景
        "  color: #e2e8f0;"               // 20260403 ZJH 文字颜色
        "  border: 1px solid #3b4252;"    // 20260403 ZJH 边框
        "  border-radius: 4px;"
        "  padding: 6px 10px;"
        "  font-size: 10pt;"
        "}"
        "QTextEdit:focus {"
        "  border-color: #00bcd4;"        // 20260403 ZJH 聚焦时青色边框
        "}"
        "QCheckBox {"
        "  color: #94a3b8;"               // 20260403 ZJH 复选框文字灰色
        "  font-size: 10pt;"
        "  spacing: 6px;"                 // 20260403 ZJH 复选框与文字间距
        "}"
        "QCheckBox::indicator {"
        "  width: 16px; height: 16px;"    // 20260403 ZJH 复选框指示器大小
        "  border: 1px solid #3b4252;"    // 20260403 ZJH 边框
        "  border-radius: 3px;"
        "  background-color: #262a35;"    // 20260403 ZJH 背景
        "}"
        "QCheckBox::indicator:checked {"
        "  background-color: #00bcd4;"    // 20260403 ZJH 选中时青色
        "  border-color: #00bcd4;"        // 20260403 ZJH 选中时边框
        "}"
    ));

    // 20260403 ZJH 创建主垂直布局
    QVBoxLayout* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(28, 24, 28, 20);  // 20260403 ZJH 对话框内边距
    pMainLayout->setSpacing(16);                       // 20260403 ZJH 控件间距

    // ===== 1. 顶部标题区 =====

    // 20260403 ZJH "创建新项目" 大标题
    QLabel* pLblTitle = new QLabel(QStringLiteral("创建新项目"), this);
    pLblTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 18pt; font-weight: bold; }"));  // 20260403 ZJH 白色大标题
    pMainLayout->addWidget(pLblTitle);

    // ===== 2. 导入数据集区域 =====

    // 20260403 ZJH 导入数据集提示行（水平布局：按钮 + 说明文字）
    QHBoxLayout* pImportLayout = new QHBoxLayout();
    pImportLayout->setSpacing(16);  // 20260403 ZJH 按钮与文字间距

    // 20260403 ZJH "从 DL 数据集创建项目（可选）：" 小标题
    QLabel* pLblImportHint = new QLabel(
        QStringLiteral("从 DL 数据集创建项目（可选）："), this);
    pLblImportHint->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; }"));  // 20260403 ZJH 灰色小字
    pMainLayout->addWidget(pLblImportHint);

    // 20260403 ZJH 导入数据集按钮
    QPushButton* pBtnImport = new QPushButton(QStringLiteral("  导入数据集"), this);
    pBtnImport->setFixedSize(180, 44);  // 20260403 ZJH 固定大小
    pBtnImport->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #262a35;"    // 20260403 ZJH 暗色背景
        "  color: #e2e8f0;"               // 20260403 ZJH 白色文字
        "  border: 1px solid #3b4252;"    // 20260403 ZJH 边框
        "  border-radius: 6px;"
        "  font-size: 10pt;"
        "  text-align: left;"             // 20260403 ZJH 文字左对齐
        "  padding-left: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2d3340;"    // 20260403 ZJH 悬停时稍亮
        "  border-color: #4b5563;"
        "}"
    ));

    // 20260403 ZJH 导入数据集说明文字
    QLabel* pLblImportDesc = new QLabel(
        QStringLiteral("从现有 DL 数据集创建项目。 数据集定义支持的 DL 方法。"), this);
    pLblImportDesc->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 9pt; }"));  // 20260403 ZJH 灰色说明

    pImportLayout->addWidget(pBtnImport);
    pImportLayout->addWidget(pLblImportDesc, 1);  // 20260403 ZJH 说明文字填充剩余空间
    pMainLayout->addLayout(pImportLayout);

    // ===== 3. 深度学习方法标题 =====

    QLabel* pLblMethodTitle = new QLabel(QStringLiteral("深度学习方法："), this);
    pLblMethodTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; }"));  // 20260403 ZJH 灰色小标题
    pMainLayout->addWidget(pLblMethodTitle);

    // ===== 4. 卡片网格 + 描述面板（水平布局） =====

    QHBoxLayout* pMiddleLayout = new QHBoxLayout();
    pMiddleLayout->setSpacing(20);  // 20260403 ZJH 卡片区与描述面板间距

    // 20260403 ZJH 4a. 左侧任务卡片网格（2 列）
    QGridLayout* pCardGrid = new QGridLayout();
    pCardGrid->setSpacing(8);  // 20260403 ZJH 卡片间距
    pCardGrid->setContentsMargins(0, 0, 0, 0);

    // 20260403 ZJH 遍历任务卡片数据，创建按钮并添加到网格
    for (int i = 0; i < m_vecCardInfos.size(); ++i) {
        QPushButton* pCard = createTaskCard(m_vecCardInfos[i], i);  // 20260403 ZJH 创建卡片
        m_vecCardButtons.append(pCard);  // 20260403 ZJH 保存到列表

        int nRow = i / 2;  // 20260403 ZJH 行号（每行 2 张卡片）
        int nCol = i % 2;  // 20260403 ZJH 列号（0 或 1）
        pCardGrid->addWidget(pCard, nRow, nCol);  // 20260403 ZJH 添加到网格
    }

    // 20260403 ZJH 将卡片网格包装到 QWidget 中，方便控制大小
    QWidget* pCardWidget = new QWidget(this);
    pCardWidget->setLayout(pCardGrid);
    pCardWidget->setMinimumWidth(480);   // 20260403 ZJH 卡片区最小宽度
    pCardWidget->setMaximumWidth(560);   // 20260403 ZJH 卡片区最大宽度

    pMiddleLayout->addWidget(pCardWidget);

    // 20260403 ZJH 4b. 右侧描述面板
    QFrame* pDescPanel = new QFrame(this);
    pDescPanel->setStyleSheet(QStringLiteral(
        "QFrame {"
        "  background-color: #262a35;"    // 20260403 ZJH 暗色背景
        "  border: 1px solid #3b4252;"    // 20260403 ZJH 边框
        "  border-radius: 8px;"
        "}"
    ));

    QVBoxLayout* pDescLayout = new QVBoxLayout(pDescPanel);
    pDescLayout->setContentsMargins(20, 20, 20, 20);  // 20260403 ZJH 内边距
    pDescLayout->setSpacing(12);                        // 20260403 ZJH 标题与正文间距

    // 20260403 ZJH 描述面板标题
    m_pLblDescTitle = new QLabel(this);
    m_pLblDescTitle->setStyleSheet(QStringLiteral(
        "QLabel { color: #e2e8f0; font-size: 13pt; font-weight: bold; border: none; }"));
    m_pLblDescTitle->setWordWrap(true);  // 20260403 ZJH 允许换行
    pDescLayout->addWidget(m_pLblDescTitle);

    // 20260403 ZJH 描述面板正文
    m_pLblDescBody = new QLabel(this);
    m_pLblDescBody->setStyleSheet(QStringLiteral(
        "QLabel { color: #94a3b8; font-size: 10pt; line-height: 160%; border: none; }"));
    m_pLblDescBody->setWordWrap(true);         // 20260403 ZJH 允许换行
    m_pLblDescBody->setAlignment(Qt::AlignTop | Qt::AlignLeft);  // 20260403 ZJH 顶部左对齐
    pDescLayout->addWidget(m_pLblDescBody, 1);  // 20260403 ZJH stretch=1 填满剩余空间

    pMiddleLayout->addWidget(pDescPanel, 1);  // 20260403 ZJH stretch=1 描述面板填充

    pMainLayout->addLayout(pMiddleLayout, 1);  // 20260403 ZJH stretch=1 中部区域填充

    // ===== 5. 分隔线 =====

    QFrame* pSeparator = new QFrame(this);
    pSeparator->setFrameShape(QFrame::HLine);    // 20260403 ZJH 水平线
    pSeparator->setStyleSheet(QStringLiteral(
        "QFrame { color: #3b4252; }"));           // 20260403 ZJH 灰色分隔线
    pMainLayout->addWidget(pSeparator);

    // ===== 6. 底部表单区 =====

    // 20260403 ZJH 6a. 项目名称 + 项目文件路径（水平排列）
    QHBoxLayout* pNamePathLayout = new QHBoxLayout();
    pNamePathLayout->setSpacing(16);  // 20260403 ZJH 名称与路径间距

    // 20260403 ZJH 项目名称组
    QVBoxLayout* pNameGroup = new QVBoxLayout();
    pNameGroup->setSpacing(4);
    QLabel* pLblName = new QLabel(QStringLiteral("项目名称"), this);
    pLblName->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; }"));
    m_pEdtName = new QLineEdit(this);
    m_pEdtName->setText(QStringLiteral("新项目"));  // 20260403 ZJH 默认项目名
    m_pEdtName->setPlaceholderText(QStringLiteral("输入项目名称"));
    pNameGroup->addWidget(pLblName);
    pNameGroup->addWidget(m_pEdtName);

    pNamePathLayout->addLayout(pNameGroup, 2);  // 20260403 ZJH stretch=2 名称占比

    // 20260403 ZJH 项目文件路径组
    QVBoxLayout* pPathGroup = new QVBoxLayout();
    pPathGroup->setSpacing(4);
    QLabel* pLblPath = new QLabel(QStringLiteral("项目文件路径"), this);
    pLblPath->setStyleSheet(QStringLiteral(
        "QLabel { color: #64748b; font-size: 9pt; }"));

    QHBoxLayout* pPathRow = new QHBoxLayout();
    pPathRow->setSpacing(8);

    m_pEdtPath = new QLineEdit(this);
    // 20260403 ZJH 默认路径为用户文档目录
    m_strDefaultDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    m_pEdtPath->setText(m_strDefaultDir + QStringLiteral("/新项目.omdl"));
    m_pEdtPath->setPlaceholderText(QStringLiteral("选择项目存储路径"));
    pPathRow->addWidget(m_pEdtPath, 1);  // 20260403 ZJH stretch=1 填充

    m_pBtnBrowse = new QPushButton(QStringLiteral("浏览..."), this);
    m_pBtnBrowse->setFixedWidth(80);  // 20260403 ZJH 按钮固定宽度
    m_pBtnBrowse->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #262a35;"
        "  color: #e2e8f0;"
        "  border: 1px solid #3b4252;"
        "  border-radius: 4px;"
        "  padding: 6px 12px;"
        "  font-size: 10pt;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2d3340;"
        "  border-color: #4b5563;"
        "}"
    ));
    pPathRow->addWidget(m_pBtnBrowse);

    pPathGroup->addWidget(pLblPath);
    pPathGroup->addLayout(pPathRow);

    pNamePathLayout->addLayout(pPathGroup, 3);  // 20260403 ZJH stretch=3 路径占比

    pMainLayout->addLayout(pNamePathLayout);

    // 20260403 ZJH 6b. 项目描述 + 复选框/按钮（水平排列）
    QHBoxLayout* pBottomLayout = new QHBoxLayout();
    pBottomLayout->setSpacing(16);

    // 20260403 ZJH 项目描述文本框（左侧）
    QVBoxLayout* pDescInputGroup = new QVBoxLayout();
    pDescInputGroup->setSpacing(0);

    m_pEdtDescription = new QTextEdit(this);
    m_pEdtDescription->setPlaceholderText(QStringLiteral("输入对项目的说明"));
    m_pEdtDescription->setFixedHeight(70);  // 20260403 ZJH 固定高度，约 3 行
    pDescInputGroup->addWidget(m_pEdtDescription);

    pBottomLayout->addLayout(pDescInputGroup, 3);  // 20260403 ZJH stretch=3

    // 20260403 ZJH 右侧：复选框 + 按钮组
    QVBoxLayout* pRightGroup = new QVBoxLayout();
    pRightGroup->setSpacing(12);

    // 20260403 ZJH 保存相对路径复选框
    m_pChkRelativePaths = new QCheckBox(
        QStringLiteral("保存对应于项目的图像路径"), this);
    pRightGroup->addWidget(m_pChkRelativePaths, 0, Qt::AlignRight);

    // 20260403 ZJH 按钮行（取消 + 创建项目）
    QHBoxLayout* pBtnLayout = new QHBoxLayout();
    pBtnLayout->setSpacing(12);
    pBtnLayout->addStretch(1);  // 20260403 ZJH 弹性空间将按钮推到右侧

    m_pBtnCancel = new QPushButton(QStringLiteral("取消"), this);
    m_pBtnCancel->setFixedSize(90, 36);  // 20260403 ZJH 取消按钮固定大小
    m_pBtnCancel->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #262a35;"    // 20260403 ZJH 暗色背景
        "  color: #e2e8f0;"               // 20260403 ZJH 白色文字
        "  border: 1px solid #3b4252;"    // 20260403 ZJH 边框
        "  border-radius: 18px;"          // 20260403 ZJH 圆角胶囊形
        "  font-size: 10pt;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2d3340;"
        "  border-color: #4b5563;"
        "}"
    ));

    m_pBtnCreate = new QPushButton(QStringLiteral("创建项目"), this);
    m_pBtnCreate->setFixedSize(110, 36);  // 20260403 ZJH 创建按钮固定大小
    m_pBtnCreate->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #00bcd4;"    // 20260403 ZJH 青色强调色
        "  color: white;"                 // 20260403 ZJH 白色文字
        "  border: none;"
        "  border-radius: 18px;"          // 20260403 ZJH 圆角胶囊形
        "  font-size: 10pt;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "  background-color: #26c6da;"    // 20260403 ZJH 悬停时亮青
        "}"
        "QPushButton:pressed {"
        "  background-color: #0097a7;"    // 20260403 ZJH 按下时深青
        "}"
    ));

    pBtnLayout->addWidget(m_pBtnCancel);
    pBtnLayout->addWidget(m_pBtnCreate);
    pRightGroup->addLayout(pBtnLayout);

    pBottomLayout->addLayout(pRightGroup, 2);  // 20260403 ZJH stretch=2

    pMainLayout->addLayout(pBottomLayout);

    // ===== 7. 信号槽连接 =====

    // 20260403 ZJH 浏览按钮 → 文件保存对话框
    connect(m_pBtnBrowse, &QPushButton::clicked,
            this, &NewProjectDialog::onBrowsePath);

    // 20260403 ZJH 项目名称变化 → 自动更新文件路径
    connect(m_pEdtName, &QLineEdit::textChanged,
            this, &NewProjectDialog::onNameChanged);

    // 20260403 ZJH 创建按钮 → 验证并接受
    connect(m_pBtnCreate, &QPushButton::clicked,
            this, &NewProjectDialog::onAccept);

    // 20260403 ZJH 取消按钮 → 关闭对话框
    connect(m_pBtnCancel, &QPushButton::clicked,
            this, &QDialog::reject);

    // ===== 8. 初始化选中状态 =====

    updateCardSelection();     // 20260403 ZJH 高亮默认选中卡片
    updateDescriptionPanel();  // 20260403 ZJH 显示默认选中卡片的描述
}

// 20260403 ZJH 创建单个任务卡片按钮
// 参数: info - 卡片信息结构体
//       nIndex - 卡片在列表中的索引
// 返回: 创建的 QPushButton 指针
QPushButton* NewProjectDialog::createTaskCard(const TaskCardInfo& info, int nIndex)
{
    QPushButton* pCard = new QPushButton(this);
    pCard->setFixedHeight(62);      // 20260403 ZJH 卡片固定高度
    pCard->setMinimumWidth(220);    // 20260403 ZJH 卡片最小宽度
    pCard->setCursor(Qt::PointingHandCursor);  // 20260403 ZJH 手型光标

    // 20260403 ZJH 构建卡片显示文本（标题 + 副标题）
    QString strCardText;
    if (info.strSubtitle.isEmpty()) {
        strCardText = info.strTitle;  // 20260403 ZJH 无副标题时只显示标题
    } else {
        strCardText = info.strTitle + QStringLiteral("\n") + info.strSubtitle;  // 20260403 ZJH 标题+副标题
    }
    pCard->setText(strCardText);

    // 20260403 ZJH 卡片默认样式（未选中状态）
    pCard->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #262a35;"    // 20260403 ZJH 暗色背景
        "  color: #e2e8f0;"               // 20260403 ZJH 白色文字
        "  border: 2px solid #3b4252;"    // 20260403 ZJH 灰色边框
        "  border-radius: 8px;"           // 20260403 ZJH 圆角
        "  text-align: left;"             // 20260403 ZJH 文字左对齐
        "  padding: 10px 14px;"           // 20260403 ZJH 内边距
        "  font-size: 10pt;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2d3340;"    // 20260403 ZJH 悬停时稍亮
        "  border-color: #4b5563;"
        "}"
    ));

    // 20260403 ZJH 连接点击信号，使用 lambda 传递索引
    connect(pCard, &QPushButton::clicked, this, [this, nIndex]() {
        onTaskCardClicked(nIndex);
    });

    return pCard;
}

// ===== 数据访问 =====

// 20260403 ZJH 获取项目名称
QString NewProjectDialog::projectName() const
{
    return m_pEdtName->text().trimmed();  // 20260403 ZJH 去除首尾空白
}

// 20260403 ZJH 获取任务类型
om::TaskType NewProjectDialog::taskType() const
{
    // 20260403 ZJH 从当前选中卡片获取任务类型
    if (m_nSelectedIndex >= 0 && m_nSelectedIndex < m_vecCardInfos.size()) {
        return m_vecCardInfos[m_nSelectedIndex].eType;
    }
    return om::TaskType::Classification;  // 20260403 ZJH 防御性默认值
}

// 20260403 ZJH 获取项目文件路径
QString NewProjectDialog::projectPath() const
{
    return m_pEdtPath->text().trimmed();  // 20260403 ZJH 去除首尾空白
}

// 20260403 ZJH 获取项目描述
QString NewProjectDialog::projectDescription() const
{
    return m_pEdtDescription->toPlainText().trimmed();  // 20260403 ZJH 纯文本内容
}

// 20260403 ZJH 获取是否保存相对路径
bool NewProjectDialog::saveRelativeImagePaths() const
{
    return m_pChkRelativePaths->isChecked();  // 20260403 ZJH 复选框状态
}

// ===== 槽函数 =====

// 20260403 ZJH 任务卡片点击处理
void NewProjectDialog::onTaskCardClicked(int nIndex)
{
    // 20260403 ZJH 更新选中索引
    if (nIndex >= 0 && nIndex < m_vecCardInfos.size()) {
        m_nSelectedIndex = nIndex;    // 20260403 ZJH 记录选中索引
        updateCardSelection();        // 20260403 ZJH 更新视觉高亮
        updateDescriptionPanel();     // 20260403 ZJH 更新描述面板内容
    }
}

// 20260403 ZJH 浏览按钮点击：弹出文件保存对话框
void NewProjectDialog::onBrowsePath()
{
    // 20260403 ZJH 弹出保存文件对话框，选择 .omdl 项目文件位置
    QString strFilePath = QFileDialog::getSaveFileName(
        this,                                                 // 20260403 ZJH 父窗口
        QStringLiteral("选择项目文件路径"),                     // 20260403 ZJH 对话框标题
        m_pEdtPath->text(),                                   // 20260403 ZJH 初始路径
        QStringLiteral("OmniMatch 项目文件 (*.omdl)")          // 20260403 ZJH 文件过滤器
    );

    // 20260403 ZJH 用户未取消则更新路径
    if (!strFilePath.isEmpty()) {
        // 20260403 ZJH 确保后缀为 .omdl
        if (!strFilePath.endsWith(QStringLiteral(".omdl"), Qt::CaseInsensitive)) {
            strFilePath += QStringLiteral(".omdl");
        }
        m_pEdtPath->setText(strFilePath);  // 20260403 ZJH 更新路径输入框
    }
}

// 20260403 ZJH 项目名称变化时自动更新文件路径
void NewProjectDialog::onNameChanged(const QString& strText)
{
    Q_UNUSED(strText);  // 20260403 ZJH 参数未直接使用，通过 projectName() 获取
    updateFilePath();   // 20260403 ZJH 重新生成文件路径
}

// 20260403 ZJH 验证输入并接受对话框
void NewProjectDialog::onAccept()
{
    // 20260403 ZJH 1. 检查项目名称非空
    if (projectName().isEmpty()) {
        QMessageBox::warning(this,
            QStringLiteral("输入错误"),
            QStringLiteral("请输入项目名称。"));
        m_pEdtName->setFocus();
        return;
    }

    // 20260403 ZJH 2. 检查项目路径非空
    if (projectPath().isEmpty()) {
        QMessageBox::warning(this,
            QStringLiteral("输入错误"),
            QStringLiteral("请选择项目文件路径。"));
        m_pEdtPath->setFocus();
        return;
    }

    accept();  // 20260403 ZJH 验证通过，关闭对话框并返回 Accepted
}

// ===== 私有辅助函数 =====

// 20260403 ZJH 更新所有卡片的选中高亮状态
void NewProjectDialog::updateCardSelection()
{
    for (int i = 0; i < m_vecCardButtons.size(); ++i) {
        QPushButton* pCard = m_vecCardButtons[i];
        if (i == m_nSelectedIndex) {
            // 20260403 ZJH 选中状态：青色边框高亮
            pCard->setStyleSheet(QStringLiteral(
                "QPushButton {"
                "  background-color: #262a35;"
                "  color: #e2e8f0;"
                "  border: 2px solid #00bcd4;"    // 20260403 ZJH 青色强调边框
                "  border-radius: 8px;"
                "  text-align: left;"
                "  padding: 10px 14px;"
                "  font-size: 10pt;"
                "}"
            ));
        } else {
            // 20260403 ZJH 未选中状态：灰色边框
            pCard->setStyleSheet(QStringLiteral(
                "QPushButton {"
                "  background-color: #262a35;"
                "  color: #e2e8f0;"
                "  border: 2px solid #3b4252;"    // 20260403 ZJH 灰色边框
                "  border-radius: 8px;"
                "  text-align: left;"
                "  padding: 10px 14px;"
                "  font-size: 10pt;"
                "}"
                "QPushButton:hover {"
                "  background-color: #2d3340;"
                "  border-color: #4b5563;"
                "}"
            ));
        }
    }
}

// 20260403 ZJH 更新右侧描述面板内容
void NewProjectDialog::updateDescriptionPanel()
{
    // 20260403 ZJH 根据当前选中索引获取卡片信息
    if (m_nSelectedIndex >= 0 && m_nSelectedIndex < m_vecCardInfos.size()) {
        const TaskCardInfo& info = m_vecCardInfos[m_nSelectedIndex];
        m_pLblDescTitle->setText(info.strDescTitle);  // 20260403 ZJH 更新标题
        m_pLblDescBody->setText(info.strDescBody);    // 20260403 ZJH 更新正文
    }
}

// 20260403 ZJH 根据项目名称自动生成文件路径
void NewProjectDialog::updateFilePath()
{
    QString strName = projectName();  // 20260403 ZJH 获取当前项目名
    if (strName.isEmpty()) {
        strName = QStringLiteral("新项目");  // 20260403 ZJH 名称为空时使用默认名
    }
    // 20260403 ZJH 在默认目录下生成 .omdl 文件路径
    m_pEdtPath->setText(m_strDefaultDir + QStringLiteral("/") + strName + QStringLiteral(".omdl"));
}
