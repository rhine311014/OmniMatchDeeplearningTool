// 20260322 ZJH NewProjectDialog 实现
// 新建项目对话框 UI 初始化、路径浏览和输入验证

#include "ui/pages/project/NewProjectDialog.h"  // 20260322 ZJH 类声明

#include <QFormLayout>     // 20260322 ZJH 表单布局（标签-控件对齐）
#include <QVBoxLayout>     // 20260322 ZJH 垂直布局
#include <QHBoxLayout>     // 20260322 ZJH 水平布局（路径行）
#include <QFileDialog>     // 20260322 ZJH 文件夹选择对话框
#include <QMessageBox>     // 20260322 ZJH 输入验证错误提示
#include <QStandardPaths>  // 20260322 ZJH 获取文档目录作为默认路径

// 20260322 ZJH 构造函数：初始化对话框 UI
NewProjectDialog::NewProjectDialog(QWidget* pParent)
    : QDialog(pParent)       // 20260322 ZJH 初始化 QDialog 基类
    , m_pEdtName(nullptr)    // 20260322 ZJH 各控件指针初始化为空
    , m_pCmbTaskType(nullptr)
    , m_pEdtPath(nullptr)
    , m_pBtnBrowse(nullptr)
    , m_pBtnBox(nullptr)
{
    setupUi();  // 20260322 ZJH 创建并组装 UI 控件
}

// 20260322 ZJH 初始化对话框布局和控件
void NewProjectDialog::setupUi()
{
    // 20260322 ZJH 设置对话框标题
    setWindowTitle(QStringLiteral("新建项目"));

    // 20260322 ZJH 设置对话框固定宽度，高度自适应
    setFixedWidth(480);

    // 20260322 ZJH 设置对话框暗色主题样式
    setStyleSheet(QStringLiteral(
        "QDialog {"
        "  background-color: #1a1d23;"   // 20260322 ZJH 深色背景
        "  color: #e2e8f0;"              // 20260322 ZJH 浅色文字
        "}"
        "QLabel {"
        "  color: #94a3b8;"              // 20260322 ZJH 标签灰色文字
        "  font-size: 10pt;"
        "}"
        "QLineEdit {"
        "  background-color: #262a35;"   // 20260322 ZJH 输入框暗色背景
        "  color: #e2e8f0;"              // 20260322 ZJH 输入文字颜色
        "  border: 1px solid #3b4252;"   // 20260322 ZJH 边框
        "  border-radius: 4px;"
        "  padding: 6px 10px;"
        "  font-size: 10pt;"
        "}"
        "QLineEdit:focus {"
        "  border-color: #2563eb;"       // 20260322 ZJH 聚焦时蓝色边框
        "}"
        "QComboBox {"
        "  background-color: #262a35;"   // 20260322 ZJH 下拉框暗色背景
        "  color: #e2e8f0;"              // 20260322 ZJH 下拉文字颜色
        "  border: 1px solid #3b4252;"   // 20260322 ZJH 边框
        "  border-radius: 4px;"
        "  padding: 6px 10px;"
        "  font-size: 10pt;"
        "}"
        "QComboBox::drop-down {"
        "  border: none;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background-color: #262a35;"   // 20260322 ZJH 下拉列表背景
        "  color: #e2e8f0;"              // 20260322 ZJH 下拉列表文字
        "  selection-background-color: #2563eb;"  // 20260322 ZJH 选中蓝色
        "}"
        "QPushButton {"
        "  background-color: #2563eb;"   // 20260322 ZJH 蓝色按钮
        "  color: white;"                // 20260322 ZJH 白色文字
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 6px 16px;"
        "  font-size: 10pt;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3b82f6;"   // 20260322 ZJH 悬停时亮蓝
        "}"
        "QPushButton:pressed {"
        "  background-color: #1d4ed8;"   // 20260322 ZJH 按下时深蓝
        "}"
    ));

    // 20260322 ZJH 创建主垂直布局
    QVBoxLayout* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(24, 20, 24, 20);  // 20260322 ZJH 对话框内边距
    pMainLayout->setSpacing(16);  // 20260322 ZJH 控件间距

    // 20260322 ZJH 创建表单布局（标签-控件左右对齐）
    QFormLayout* pFormLayout = new QFormLayout();
    pFormLayout->setSpacing(12);  // 20260322 ZJH 行间距
    pFormLayout->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);  // 20260322 ZJH 标签右对齐

    // 20260322 ZJH 1. 项目名称输入框
    m_pEdtName = new QLineEdit(this);
    m_pEdtName->setText(QStringLiteral("新建项目"));           // 20260322 ZJH 默认项目名
    m_pEdtName->setPlaceholderText(QStringLiteral("输入项目名称"));  // 20260322 ZJH 占位提示
    pFormLayout->addRow(QStringLiteral("项目名称:"), m_pEdtName);    // 20260322 ZJH 添加到表单

    // 20260322 ZJH 2. 任务类型下拉框（8 种任务类型）
    m_pCmbTaskType = new QComboBox(this);
    // 20260322 ZJH 遍历所有 TaskType 枚举值，填充中文名称
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::AnomalyDetection),
                            static_cast<int>(om::TaskType::AnomalyDetection));       // 20260322 ZJH 异常检测
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::Classification),
                            static_cast<int>(om::TaskType::Classification));         // 20260322 ZJH 图像分类
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::ObjectDetection),
                            static_cast<int>(om::TaskType::ObjectDetection));         // 20260322 ZJH 目标检测
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::SemanticSegmentation),
                            static_cast<int>(om::TaskType::SemanticSegmentation));   // 20260322 ZJH 语义分割
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::InstanceSegmentation),
                            static_cast<int>(om::TaskType::InstanceSegmentation));   // 20260322 ZJH 实例分割
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::DeepOCR),
                            static_cast<int>(om::TaskType::DeepOCR));                // 20260322 ZJH 深度 OCR
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::ZeroShotDefectDetection),
                            static_cast<int>(om::TaskType::ZeroShotDefectDetection));// 20260322 ZJH 零样本缺陷检测
    m_pCmbTaskType->addItem(om::taskTypeToString(om::TaskType::ZeroShotObjectDetection),
                            static_cast<int>(om::TaskType::ZeroShotObjectDetection));// 20260322 ZJH 零样本目标检测

    // 20260322 ZJH 默认选中"图像分类"（索引 1）
    m_pCmbTaskType->setCurrentIndex(1);
    pFormLayout->addRow(QStringLiteral("任务类型:"), m_pCmbTaskType);  // 20260322 ZJH 添加到表单

    // 20260322 ZJH 3. 项目路径（输入框 + 浏览按钮水平排列）
    QHBoxLayout* pPathLayout = new QHBoxLayout();
    pPathLayout->setSpacing(8);  // 20260322 ZJH 输入框和按钮间距

    m_pEdtPath = new QLineEdit(this);
    // 20260322 ZJH 默认路径为用户文档目录下的 OmniMatch 子目录
    QString strDefaultPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                             + "/OmniMatch";
    m_pEdtPath->setText(strDefaultPath);  // 20260322 ZJH 设置默认路径
    m_pEdtPath->setPlaceholderText(QStringLiteral("选择项目存储路径"));  // 20260322 ZJH 占位提示
    pPathLayout->addWidget(m_pEdtPath, 1);  // 20260322 ZJH stretch=1，占满剩余空间

    m_pBtnBrowse = new QPushButton(QStringLiteral("浏览..."), this);
    m_pBtnBrowse->setFixedWidth(80);  // 20260322 ZJH 按钮固定宽度
    pPathLayout->addWidget(m_pBtnBrowse);  // 20260322 ZJH 添加浏览按钮

    pFormLayout->addRow(QStringLiteral("项目路径:"), pPathLayout);  // 20260322 ZJH 添加到表单

    pMainLayout->addLayout(pFormLayout);  // 20260322 ZJH 将表单加入主布局

    // 20260322 ZJH 4. 弹性空间
    pMainLayout->addStretch(1);

    // 20260322 ZJH 5. 确定/取消按钮框
    m_pBtnBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    m_pBtnBox->button(QDialogButtonBox::Ok)->setText(QStringLiteral("确定"));        // 20260322 ZJH 中文化
    m_pBtnBox->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("取消"));     // 20260322 ZJH 中文化
    pMainLayout->addWidget(m_pBtnBox);

    // 20260322 ZJH 连接信号槽
    connect(m_pBtnBrowse, &QPushButton::clicked,
            this, &NewProjectDialog::onBrowsePath);    // 20260322 ZJH 浏览按钮 → 文件夹选择

    connect(m_pBtnBox, &QDialogButtonBox::accepted,
            this, &NewProjectDialog::onAccept);        // 20260322 ZJH 确定 → 验证并接受

    connect(m_pBtnBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);                   // 20260322 ZJH 取消 → 关闭对话框
}

// ===== 数据访问 =====

// 20260322 ZJH 获取项目名称
QString NewProjectDialog::projectName() const
{
    return m_pEdtName->text().trimmed();  // 20260322 ZJH 去除首尾空白
}

// 20260322 ZJH 获取任务类型
om::TaskType NewProjectDialog::taskType() const
{
    // 20260322 ZJH 从下拉框的 userData 获取枚举值
    int nType = m_pCmbTaskType->currentData().toInt();  // 20260322 ZJH 读取关联数据
    return static_cast<om::TaskType>(nType);             // 20260322 ZJH 转换为枚举类型
}

// 20260322 ZJH 获取项目路径
QString NewProjectDialog::projectPath() const
{
    return m_pEdtPath->text().trimmed();  // 20260322 ZJH 去除首尾空白
}

// ===== 槽函数 =====

// 20260322 ZJH 浏览按钮点击：弹出文件夹选择对话框
void NewProjectDialog::onBrowsePath()
{
    // 20260322 ZJH 以当前路径作为初始目录
    QString strDir = QFileDialog::getExistingDirectory(
        this,                                    // 20260322 ZJH 父窗口
        QStringLiteral("选择项目存储目录"),         // 20260322 ZJH 对话框标题
        m_pEdtPath->text(),                       // 20260322 ZJH 初始目录
        QFileDialog::ShowDirsOnly                 // 20260322 ZJH 仅显示目录
    );

    // 20260322 ZJH 用户未取消则更新路径
    if (!strDir.isEmpty()) {
        m_pEdtPath->setText(strDir);  // 20260322 ZJH 更新路径输入框
    }
}

// 20260322 ZJH 验证输入并接受对话框
void NewProjectDialog::onAccept()
{
    // 20260322 ZJH 1. 检查项目名称非空
    if (projectName().isEmpty()) {
        QMessageBox::warning(this,
            QStringLiteral("输入错误"),
            QStringLiteral("请输入项目名称。"));  // 20260322 ZJH 名称为空提示
        m_pEdtName->setFocus();  // 20260322 ZJH 聚焦到名称输入框
        return;  // 20260322 ZJH 不关闭对话框
    }

    // 20260322 ZJH 2. 检查项目路径非空
    if (projectPath().isEmpty()) {
        QMessageBox::warning(this,
            QStringLiteral("输入错误"),
            QStringLiteral("请选择项目存储路径。"));  // 20260322 ZJH 路径为空提示
        m_pEdtPath->setFocus();  // 20260322 ZJH 聚焦到路径输入框
        return;  // 20260322 ZJH 不关闭对话框
    }

    accept();  // 20260322 ZJH 验证通过，关闭对话框并返回 QDialog::Accepted
}
