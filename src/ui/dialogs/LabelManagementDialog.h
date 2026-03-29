// 20260322 ZJH LabelManagementDialog — 标签管理对话框
// QDialog 子类，通过 QTableWidget 管理数据集中的标签
// 支持添加标签、删除标签、双击修改颜色和名称
#pragma once

#include <QDialog>           // 20260322 ZJH 对话框基类
#include <QTableWidget>      // 20260322 ZJH 表格控件
#include <QPushButton>       // 20260322 ZJH 按钮
#include <QDialogButtonBox>  // 20260322 ZJH 标准按钮组
#include <QVector>           // 20260322 ZJH 向量容器

// 20260322 ZJH 前向声明
class ImageDataset;
struct LabelInfo;

// 20260322 ZJH 标签管理对话框
// 功能：
//   - QTableWidget 显示标签列表（颜色块/名称/图像数/操作）
//   - 添加标签（输入名称，自动分配颜色）
//   - 删除选中标签
//   - 双击颜色块弹出 QColorDialog
//   - 双击名称可编辑
class LabelManagementDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pDataset - 当前数据集指针（用于读取/修改标签）
    //       pParent  - 父窗口
    explicit LabelManagementDialog(ImageDataset* pDataset,
                                   QWidget* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~LabelManagementDialog() override = default;

private slots:
    // 20260322 ZJH 添加标签按钮点击
    void onAddLabel();

    // 20260322 ZJH 删除选中标签按钮点击
    void onDeleteLabel();

    // 20260324 ZJH 单击颜色列弹出 QColorDialog 选色
    void onCellClicked(int nRow, int nCol);

    // 20260322 ZJH 表格单元格双击（名称列→编辑）
    void onCellDoubleClicked(int nRow, int nCol);

    // 20260322 ZJH 表格单元格内容变化（名称编辑完成后更新数据集）
    void onCellChanged(int nRow, int nCol);

    // 20260322 ZJH 确定按钮点击
    void onAccept();

private:
    // 20260322 ZJH 刷新表格内容
    void refreshTable();

    // 20260322 ZJH 创建颜色块图标
    QIcon createColorIcon(const QColor& color, int nSize = 16);

    QTableWidget*     m_pTable;          // 20260322 ZJH 标签表格
    QPushButton*      m_pBtnAdd;         // 20260322 ZJH "添加标签" 按钮
    QPushButton*      m_pBtnDelete;      // 20260322 ZJH "删除标签" 按钮
    QDialogButtonBox* m_pButtonBox;      // 20260322 ZJH OK/Cancel 按钮组

    ImageDataset*     m_pDataset;        // 20260322 ZJH 数据集指针（弱引用）
    bool              m_bUpdating;       // 20260322 ZJH 刷新标志，防止信号递归
};
