// 20260322 ZJH LabelManagementDialog 实现
// 标签管理对话框：表格列表 + 增删改 + 颜色选择

#include "ui/dialogs/LabelManagementDialog.h"  // 20260322 ZJH 类声明
#include "core/data/ImageDataset.h"             // 20260322 ZJH 数据集管理
#include "core/data/LabelInfo.h"                // 20260322 ZJH 标签信息

#include <QVBoxLayout>       // 20260322 ZJH 垂直布局
#include <QHBoxLayout>       // 20260322 ZJH 水平布局
#include <QHeaderView>       // 20260322 ZJH 表头视图
#include <QColorDialog>      // 20260322 ZJH 颜色选择对话框
#include <QInputDialog>      // 20260322 ZJH 输入对话框
#include <QMessageBox>       // 20260322 ZJH 消息对话框
#include <QPixmap>           // 20260322 ZJH 位图
#include <QPainter>          // 20260322 ZJH 绘图器
#include <QIcon>             // 20260322 ZJH 图标

// 20260322 ZJH 构造函数
LabelManagementDialog::LabelManagementDialog(ImageDataset* pDataset,
                                               QWidget* pParent)
    : QDialog(pParent)
    , m_pTable(nullptr)
    , m_pBtnAdd(nullptr)
    , m_pBtnDelete(nullptr)
    , m_pButtonBox(nullptr)
    , m_pDataset(pDataset)
    , m_bUpdating(false)
{
    // 20260322 ZJH 设置对话框标题和大小
    setWindowTitle(QStringLiteral("标签管理"));
    setMinimumSize(500, 400);
    resize(560, 460);

    // 20260322 ZJH 暗色主题样式
    setStyleSheet(QStringLiteral(
        "QDialog {"
        "  background-color: #1a1d24;"
        "  color: #e2e8f0;"
        "}"
        "QTableWidget {"
        "  background-color: #13151a;"
        "  color: #e2e8f0;"
        "  gridline-color: #2a2d35;"
        "  border: 1px solid #2a2d35;"
        "  selection-background-color: #2563eb;"
        "}"
        "QTableWidget::item {"
        "  padding: 4px 8px;"
        "}"
        "QHeaderView::section {"
        "  background-color: #1e2230;"
        "  color: #94a3b8;"
        "  border: 1px solid #2a2d35;"
        "  padding: 6px;"
        "  font-weight: bold;"
        "}"
        "QPushButton {"
        "  background-color: #2a2d35;"
        "  color: #e2e8f0;"
        "  border: 1px solid #3a3d45;"
        "  border-radius: 4px;"
        "  padding: 6px 16px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3a3d45;"
        "}"
    ));

    // 20260322 ZJH 主布局
    QVBoxLayout* pMainLayout = new QVBoxLayout(this);
    pMainLayout->setContentsMargins(12, 12, 12, 12);
    pMainLayout->setSpacing(8);

    // ===== 表格 =====

    // 20260322 ZJH 创建标签表格（4列：颜色/名称/图像数/操作）
    m_pTable = new QTableWidget(0, 4, this);
    m_pTable->setHorizontalHeaderLabels({
        QStringLiteral("颜色"),
        QStringLiteral("名称"),
        QStringLiteral("图像数"),
        QStringLiteral("ID")
    });

    // 20260322 ZJH 设置列宽
    m_pTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Fixed);
    m_pTable->setColumnWidth(0, 80);    // 20260324 ZJH 颜色列加宽方便点击
    m_pTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);  // 20260322 ZJH 名称列自适应
    m_pTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Fixed);
    m_pTable->setColumnWidth(2, 80);    // 20260322 ZJH 图像数列固定宽度
    m_pTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Fixed);
    m_pTable->setColumnWidth(3, 50);    // 20260322 ZJH ID 列固定宽度

    m_pTable->setSelectionBehavior(QTableWidget::SelectRows);         // 20260322 ZJH 整行选中
    m_pTable->setSelectionMode(QTableWidget::SingleSelection);        // 20260322 ZJH 单选
    m_pTable->verticalHeader()->setVisible(false);                    // 20260322 ZJH 隐藏行号

    // 20260324 ZJH 单击颜色列即弹出颜色选择器（比双击更直观）
    connect(m_pTable, &QTableWidget::cellClicked,
            this, &LabelManagementDialog::onCellClicked);
    // 20260322 ZJH 双击名称列进入编辑模式
    connect(m_pTable, &QTableWidget::cellDoubleClicked,
            this, &LabelManagementDialog::onCellDoubleClicked);
    connect(m_pTable, &QTableWidget::cellChanged,
            this, &LabelManagementDialog::onCellChanged);

    pMainLayout->addWidget(m_pTable, 1);  // 20260322 ZJH stretch=1

    // ===== 操作按钮行 =====

    QHBoxLayout* pBtnLayout = new QHBoxLayout();
    pBtnLayout->setSpacing(8);

    // 20260322 ZJH "添加标签" 按钮
    m_pBtnAdd = new QPushButton(QStringLiteral("添加标签"), this);
    m_pBtnAdd->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #2563eb;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 6px 16px;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "  background-color: #3b82f6;"
        "}"
    ));
    connect(m_pBtnAdd, &QPushButton::clicked,
            this, &LabelManagementDialog::onAddLabel);
    pBtnLayout->addWidget(m_pBtnAdd);

    // 20260322 ZJH "删除标签" 按钮
    m_pBtnDelete = new QPushButton(QStringLiteral("删除标签"), this);
    m_pBtnDelete->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  background-color: #dc2626;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 6px 16px;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "  background-color: #ef4444;"
        "}"
    ));
    connect(m_pBtnDelete, &QPushButton::clicked,
            this, &LabelManagementDialog::onDeleteLabel);
    pBtnLayout->addWidget(m_pBtnDelete);

    pBtnLayout->addStretch(1);  // 20260322 ZJH 弹性空间
    pMainLayout->addLayout(pBtnLayout);

    // ===== OK/Cancel 按钮组 =====

    m_pButtonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    pMainLayout->addWidget(m_pButtonBox);

    // 20260322 ZJH 连接按钮信号
    connect(m_pButtonBox, &QDialogButtonBox::accepted,
            this, &LabelManagementDialog::onAccept);
    connect(m_pButtonBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);

    // 20260322 ZJH 初始化表格内容
    refreshTable();
}

// 20260324 ZJH 添加标签 — 输入名称 + 自选颜色
void LabelManagementDialog::onAddLabel()
{
    if (!m_pDataset) {
        return;
    }

    // 20260322 ZJH 弹出输入对话框获取标签名称
    bool bOk = false;
    QString strName = QInputDialog::getText(
        this,
        QStringLiteral("添加标签"),
        QStringLiteral("标签名称:"),
        QLineEdit::Normal,
        QStringLiteral(""),
        &bOk);

    if (!bOk || strName.trimmed().isEmpty()) {
        return;  // 20260322 ZJH 用户取消或输入为空
    }

    // 20260324 ZJH 预设默认颜色（从预设列表循环选取），作为颜色对话框初始值
    QVector<QColor> vecColors = defaultLabelColors();
    int nColorIndex = m_pDataset->labels().size() % vecColors.size();
    QColor defaultColor = vecColors[nColorIndex];

    // 20260324 ZJH 弹出颜色选择对话框让用户自定义标签颜色
    QColor selectedColor = QColorDialog::getColor(
        defaultColor, this, QStringLiteral("选择标签颜色"));

    // 20260324 ZJH 用户取消颜色选择时使用预设颜色
    if (!selectedColor.isValid()) {
        selectedColor = defaultColor;
    }

    // 20260322 ZJH 创建新标签
    LabelInfo newLabel;
    newLabel.strName = strName.trimmed();
    newLabel.color = selectedColor;  // 20260324 ZJH 使用用户选择的颜色

    // 20260322 ZJH 添加到数据集
    m_pDataset->addLabel(newLabel);

    // 20260322 ZJH 刷新表格
    refreshTable();
}

// 20260322 ZJH 删除选中标签
void LabelManagementDialog::onDeleteLabel()
{
    if (!m_pDataset || !m_pTable) {
        return;
    }

    // 20260322 ZJH 获取选中行
    int nRow = m_pTable->currentRow();
    if (nRow < 0) {
        QMessageBox::information(this,
            QStringLiteral("删除标签"),
            QStringLiteral("请先选中要删除的标签行。"));
        return;
    }

    // 20260322 ZJH 获取标签 ID
    QTableWidgetItem* pIdItem = m_pTable->item(nRow, 3);
    if (!pIdItem) {
        return;
    }
    int nLabelId = pIdItem->text().toInt();

    // 20260322 ZJH 确认删除
    QString strName = m_pTable->item(nRow, 1)->text();
    int nRet = QMessageBox::question(this,
        QStringLiteral("确认删除"),
        QStringLiteral("确定要删除标签 \"%1\" 吗？").arg(strName),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (nRet != QMessageBox::Yes) {
        return;  // 20260322 ZJH 用户取消
    }

    // 20260322 ZJH 从数据集中移除标签
    m_pDataset->removeLabel(nLabelId);

    // 20260322 ZJH 刷新表格
    refreshTable();
}

// 20260324 ZJH 单击颜色列弹出颜色选择器
void LabelManagementDialog::onCellClicked(int nRow, int nCol)
{
    if (!m_pDataset || nCol != 0) {
        return;  // 20260324 ZJH 仅处理颜色列（第 0 列）
    }

    QTableWidgetItem* pIdItem = m_pTable->item(nRow, 3);
    if (!pIdItem) {
        return;
    }
    int nLabelId = pIdItem->text().toInt();

    // 20260324 ZJH 获取当前颜色
    LabelInfo* pLabel = m_pDataset->findLabel(nLabelId);
    if (!pLabel) {
        return;
    }

    // 20260324 ZJH 弹出颜色选择对话框
    QColor newColor = QColorDialog::getColor(
        pLabel->color, this, QStringLiteral("选择标签颜色"));

    if (newColor.isValid()) {
        // 20260324 ZJH 更新标签颜色
        pLabel->color = newColor;
        m_pDataset->updateLabel(*pLabel);

        // 20260324 ZJH 刷新表格
        refreshTable();
    }
}

// 20260322 ZJH 表格单元格双击
void LabelManagementDialog::onCellDoubleClicked(int nRow, int nCol)
{
    Q_UNUSED(nRow);
    // 20260324 ZJH 颜色列已由 onCellClicked 处理，双击仅保留名称列的默认编辑行为
    if (nCol == 0) {
        return;  // 20260324 ZJH 颜色列已由单击处理，避免重复弹出
    }
    // 20260322 ZJH 双击名称列（nCol == 1）由默认编辑行为处理
}

// 20260322 ZJH 表格单元格内容变化（名称编辑完成）
void LabelManagementDialog::onCellChanged(int nRow, int nCol)
{
    // 20260322 ZJH 刷新过程中忽略变化信号
    if (m_bUpdating || !m_pDataset) {
        return;
    }

    if (nCol == 1) {
        // 20260322 ZJH 名称列被编辑
        QTableWidgetItem* pNameItem = m_pTable->item(nRow, nCol);
        QTableWidgetItem* pIdItem = m_pTable->item(nRow, 3);
        if (!pNameItem || !pIdItem) {
            return;
        }

        int nLabelId = pIdItem->text().toInt();
        QString strNewName = pNameItem->text().trimmed();

        if (strNewName.isEmpty()) {
            // 20260322 ZJH 名称不能为空，恢复原名
            refreshTable();
            return;
        }

        // 20260322 ZJH 更新标签名称
        LabelInfo* pLabel = m_pDataset->findLabel(nLabelId);
        if (pLabel) {
            pLabel->strName = strNewName;
            m_pDataset->updateLabel(*pLabel);
        }
    }
}

// 20260322 ZJH 确定按钮点击
void LabelManagementDialog::onAccept()
{
    accept();  // 20260322 ZJH 关闭对话框
}

// 20260322 ZJH 刷新表格内容
void LabelManagementDialog::refreshTable()
{
    if (!m_pTable || !m_pDataset) {
        return;
    }

    m_bUpdating = true;  // 20260322 ZJH 设置刷新标志，防止 cellChanged 信号递归

    m_pTable->setRowCount(0);  // 20260322 ZJH 清空表格

    // 20260322 ZJH 获取标签分布统计
    QMap<int, int> mapDist = m_pDataset->labelDistribution();

    // 20260322 ZJH 遍历所有标签
    const QVector<LabelInfo>& vecLabels = m_pDataset->labels();
    for (const LabelInfo& label : vecLabels) {
        int nRow = m_pTable->rowCount();
        m_pTable->insertRow(nRow);  // 20260322 ZJH 添加新行

        // 20260322 ZJH 列 0：颜色块
        QTableWidgetItem* pColorItem = new QTableWidgetItem();
        pColorItem->setIcon(createColorIcon(label.color, 24));  // 20260324 ZJH 颜色图标加大方便点击
        pColorItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);  // 20260322 ZJH 不可编辑
        pColorItem->setToolTip(label.color.name());  // 20260322 ZJH 提示颜色值
        m_pTable->setItem(nRow, 0, pColorItem);

        // 20260322 ZJH 列 1：标签名称（可编辑）
        QTableWidgetItem* pNameItem = new QTableWidgetItem(label.strName);
        pNameItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsEditable);
        m_pTable->setItem(nRow, 1, pNameItem);

        // 20260322 ZJH 列 2：图像数
        int nCount = mapDist.value(label.nId, 0);
        QTableWidgetItem* pCountItem = new QTableWidgetItem(QString::number(nCount));
        pCountItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);  // 20260322 ZJH 不可编辑
        pCountItem->setTextAlignment(Qt::AlignCenter);
        m_pTable->setItem(nRow, 2, pCountItem);

        // 20260322 ZJH 列 3：标签 ID（隐含数据）
        QTableWidgetItem* pIdItem = new QTableWidgetItem(QString::number(label.nId));
        pIdItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);  // 20260322 ZJH 不可编辑
        pIdItem->setTextAlignment(Qt::AlignCenter);
        m_pTable->setItem(nRow, 3, pIdItem);
    }

    m_bUpdating = false;  // 20260322 ZJH 恢复刷新标志
}

// 20260322 ZJH 创建颜色块图标
QIcon LabelManagementDialog::createColorIcon(const QColor& color, int nSize)
{
    QPixmap pixmap(nSize, nSize);  // 20260322 ZJH 创建 nSize x nSize 位图
    pixmap.fill(Qt::transparent);   // 20260322 ZJH 透明背景

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setBrush(color);                        // 20260322 ZJH 设置填充颜色
    painter.setPen(color.darker(130));               // 20260322 ZJH 设置深色描边
    painter.drawRoundedRect(1, 1, nSize - 2, nSize - 2, 3, 3);  // 20260322 ZJH 绘制圆角矩形
    painter.end();

    return QIcon(pixmap);  // 20260322 ZJH 返回图标
}
