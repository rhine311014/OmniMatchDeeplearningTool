// 20260403 ZJH NewProjectDialog — Halcon 风格新建项目对话框
// 卡片网格选择任务类型 + 右侧描述面板 + 底部项目信息表单
// 参考 MVTec Deep Learning Tool / Halcon DL Tool 24.05 UI 设计
#pragma once

#include <QDialog>          // 20260403 ZJH 对话框基类
#include <QLineEdit>        // 20260403 ZJH 文本输入框
#include <QTextEdit>        // 20260403 ZJH 多行文本输入（项目描述）
#include <QPushButton>      // 20260403 ZJH 按钮
#include <QLabel>           // 20260403 ZJH 标签
#include <QCheckBox>        // 20260403 ZJH 复选框
#include <QVector>          // 20260403 ZJH 卡片按钮列表

#include "core/DLTypes.h"   // 20260403 ZJH TaskType 枚举

// 20260403 ZJH 任务类型卡片信息结构体
// 每张卡片对应一种深度学习任务类型
struct TaskCardInfo
{
    om::TaskType eType;       // 20260403 ZJH 任务类型枚举值
    QString      strTitle;    // 20260403 ZJH 卡片标题（中文）
    QString      strSubtitle; // 20260403 ZJH 卡片副标题（英文/补充说明）
    QString      strDescTitle;// 20260403 ZJH 描述面板标题
    QString      strDescBody; // 20260403 ZJH 描述面板详细说明
};

// 20260403 ZJH Halcon 风格新建项目对话框
// 布局：
//   顶部: "创建新项目" 标题 + 导入数据集按钮
//   中部: 任务类型卡片网格(左) + 选中任务描述面板(右)
//   底部: 项目名称 + 项目文件路径 + 项目描述 + 复选框 + 取消/创建按钮
class NewProjectDialog : public QDialog
{
    Q_OBJECT

public:
    // 20260403 ZJH 构造函数，初始化 Halcon 风格对话框 UI
    // 参数: pParent - 父窗口
    explicit NewProjectDialog(QWidget* pParent = nullptr);

    // 20260403 ZJH 默认析构
    ~NewProjectDialog() override = default;

    // ===== 数据访问 =====

    // 20260403 ZJH 获取用户输入的项目名称
    // 返回: 项目名称字符串
    QString projectName() const;

    // 20260403 ZJH 获取用户选择的任务类型
    // 返回: TaskType 枚举值
    om::TaskType taskType() const;

    // 20260403 ZJH 获取用户选择的项目文件路径（含 .omdl 后缀）
    // 返回: 项目文件绝对路径
    QString projectPath() const;

    // 20260403 ZJH 获取用户输入的项目描述
    // 返回: 项目描述字符串
    QString projectDescription() const;

    // 20260403 ZJH 获取是否保存对应于项目的图像路径
    // 返回: true 表示保存相对路径，false 表示绝对路径
    bool saveRelativeImagePaths() const;

private slots:
    // 20260403 ZJH 任务卡片点击：切换选中状态并更新描述面板
    // 参数: nIndex - 被点击卡片的索引
    void onTaskCardClicked(int nIndex);

    // 20260403 ZJH 浏览按钮点击：弹出文件保存对话框
    void onBrowsePath();

    // 20260403 ZJH 项目名称变化时自动更新文件路径
    void onNameChanged(const QString& strText);

    // 20260403 ZJH 验证输入合法性并接受对话框
    void onAccept();

private:
    // 20260403 ZJH 初始化对话框布局和控件
    void setupUi();

    // 20260403 ZJH 初始化任务类型卡片数据（10 种任务类型）
    void initTaskCards();

    // 20260403 ZJH 创建单个任务卡片按钮
    // 参数: info - 卡片信息
    //       nIndex - 卡片索引（用于信号映射）
    // 返回: 创建的按钮指针
    QPushButton* createTaskCard(const TaskCardInfo& info, int nIndex);

    // 20260403 ZJH 更新选中卡片的高亮状态
    void updateCardSelection();

    // 20260403 ZJH 更新右侧描述面板内容
    void updateDescriptionPanel();

    // 20260403 ZJH 根据项目名称和默认目录自动生成文件路径
    void updateFilePath();

    // ===== 成员变量 =====

    QVector<TaskCardInfo>  m_vecCardInfos;    // 20260403 ZJH 任务卡片数据列表
    QVector<QPushButton*>  m_vecCardButtons;  // 20260403 ZJH 任务卡片按钮列表
    int                    m_nSelectedIndex;  // 20260403 ZJH 当前选中卡片索引

    QLabel*     m_pLblDescTitle;   // 20260403 ZJH 描述面板标题标签
    QLabel*     m_pLblDescBody;    // 20260403 ZJH 描述面板详细说明标签

    QLineEdit*  m_pEdtName;        // 20260403 ZJH 项目名称输入框
    QLineEdit*  m_pEdtPath;        // 20260403 ZJH 项目文件路径输入框
    QPushButton* m_pBtnBrowse;     // 20260403 ZJH 路径浏览按钮
    QTextEdit*  m_pEdtDescription; // 20260403 ZJH 项目描述多行文本框
    QCheckBox*  m_pChkRelativePaths; // 20260403 ZJH 保存相对图像路径复选框

    QPushButton* m_pBtnCancel;     // 20260403 ZJH 取消按钮
    QPushButton* m_pBtnCreate;     // 20260403 ZJH 创建项目按钮

    QString     m_strDefaultDir;   // 20260403 ZJH 默认项目目录（用户文档目录）
};
