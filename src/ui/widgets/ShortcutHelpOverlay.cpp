// 20260322 ZJH ShortcutHelpOverlay 实现
// 快捷键参考面板：半透明遮罩 + 中央卡片 + 分组快捷键表格

#include "ui/widgets/ShortcutHelpOverlay.h"  // 20260322 ZJH 类声明

#include <QPainter>      // 20260322 ZJH 绘图器
#include <QPaintEvent>   // 20260322 ZJH 绘制事件
#include <QKeyEvent>     // 20260322 ZJH 按键事件
#include <QMouseEvent>   // 20260322 ZJH 鼠标事件
#include <QFontMetrics>  // 20260322 ZJH 字体度量
#include <QEvent>        // 20260324 ZJH QEvent::Resize 事件类型

// 20260322 ZJH 快捷键表格数据结构
struct ShortcutEntry {
    QString strKey;      // 20260322 ZJH 快捷键
    QString strAction;   // 20260322 ZJH 功能描述
};

// 20260322 ZJH 快捷键分组数据结构
struct ShortcutGroup {
    QString strTitle;                    // 20260322 ZJH 分组标题
    QVector<ShortcutEntry> vecEntries;   // 20260322 ZJH 快捷键条目列表
};

// 20260322 ZJH 全局快捷键表格定义
// 20260323 ZJH 静态常量避免每次 paintEvent 重新分配数十个 QString 对象
static const QVector<ShortcutGroup>& getShortcutGroups()
{
    static const QVector<ShortcutGroup> s_groups = {
        {
            QStringLiteral("文件"),
            {
                { QStringLiteral("Ctrl+N"), QStringLiteral("新建项目") },
                { QStringLiteral("Ctrl+O"), QStringLiteral("打开项目") },
                { QStringLiteral("Ctrl+S"), QStringLiteral("保存") },
                { QStringLiteral("Ctrl+W"), QStringLiteral("关闭项目") },
                { QStringLiteral("Ctrl+Q"), QStringLiteral("退出") },
            }
        },
        {
            QStringLiteral("页面导航"),
            {
                { QStringLiteral("Alt+1~8"), QStringLiteral("切换页面") },
                { QStringLiteral("Ctrl+G"),  QStringLiteral("跳转图像") },
                { QStringLiteral("Ctrl+F"),  QStringLiteral("搜索") },
            }
        },
        {
            QStringLiteral("标注工具"),
            {
                { QStringLiteral("V"), QStringLiteral("选择") },
                { QStringLiteral("B"), QStringLiteral("矩形") },
                { QStringLiteral("P"), QStringLiteral("多边形") },
                { QStringLiteral("D"), QStringLiteral("画笔") },
                { QStringLiteral("E"), QStringLiteral("橡皮") },
                { QStringLiteral("Ctrl+Z"), QStringLiteral("撤销") },
                { QStringLiteral("Ctrl+Y"), QStringLiteral("重做") },
                { QStringLiteral("Ctrl+C"), QStringLiteral("复制标注") },
                { QStringLiteral("Ctrl+V"), QStringLiteral("粘贴标注") },
            }
        },
        {
            QStringLiteral("训练控制"),
            {
                { QStringLiteral("F5"),  QStringLiteral("开始训练") },
                { QStringLiteral("F6"),  QStringLiteral("暂停") },
                { QStringLiteral("F7"),  QStringLiteral("停止") },
                { QStringLiteral("F11"), QStringLiteral("全屏") },
            }
        }
    };
    return s_groups;
}

// 20260322 ZJH 构造函数
ShortcutHelpOverlay::ShortcutHelpOverlay(QWidget* pParent)
    : QWidget(pParent)
{
    // 20260322 ZJH 设置为无边框窗口属性
    setWindowFlags(Qt::Widget);
    // 20260322 ZJH 设置焦点策略为强焦点，接收键盘事件
    setFocusPolicy(Qt::StrongFocus);
    // 20260324 ZJH 安装事件过滤器到父控件，监听 Resize 事件同步遮罩大小
    if (pParent) {
        pParent->installEventFilter(this);  // 20260324 ZJH 父控件 resize 时自动调整遮罩
    }
    // 20260322 ZJH 初始隐藏
    hide();
}

// 20260322 ZJH 显示遮罩
void ShortcutHelpOverlay::showOverlay()
{
    if (parentWidget()) {
        // 20260322 ZJH 调整大小覆盖父窗口
        resize(parentWidget()->size());
        move(0, 0);
    }
    show();       // 20260322 ZJH 显示
    raise();      // 20260322 ZJH 置于最前
    setFocus();   // 20260322 ZJH 获取焦点以接收键盘事件
}

// 20260322 ZJH 自绘事件
void ShortcutHelpOverlay::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 20260322 ZJH 1. 绘制半透明黑色遮罩
    painter.fillRect(rect(), QColor(0, 0, 0, 180));

    // 20260322 ZJH 2. 计算中央卡片区域
    int nCardWidth = 640;   // 20260322 ZJH 卡片宽度
    int nCardHeight = 520;  // 20260322 ZJH 卡片高度
    int nCardX = (width() - nCardWidth) / 2;    // 20260322 ZJH 卡片水平居中
    int nCardY = (height() - nCardHeight) / 2;  // 20260322 ZJH 卡片垂直居中
    QRect cardRect(nCardX, nCardY, nCardWidth, nCardHeight);

    // 20260322 ZJH 3. 绘制卡片背景（暗色圆角矩形）
    painter.setBrush(QColor(26, 29, 36));          // 20260322 ZJH 暗色背景 #1a1d24
    painter.setPen(QColor(42, 45, 53));             // 20260322 ZJH 边框 #2a2d35
    painter.drawRoundedRect(cardRect, 12, 12);      // 20260322 ZJH 12px 圆角

    // 20260322 ZJH 4. 绘制标题
    QFont titleFont;
    titleFont.setPixelSize(20);
    titleFont.setBold(true);
    painter.setFont(titleFont);
    painter.setPen(QColor(226, 232, 240));  // 20260322 ZJH 白色文字 #e2e8f0

    int nTitleY = nCardY + 36;  // 20260322 ZJH 标题 Y 位置
    painter.drawText(QRect(nCardX, nCardY + 16, nCardWidth, 30),
                     Qt::AlignCenter,
                     QStringLiteral("快捷键参考"));

    // 20260322 ZJH 5. 绘制分组快捷键表格
    // 20260324 ZJH 使用 const 引用避免每次 paintEvent 拷贝整个 vector
    const auto& vecGroups = getShortcutGroups();

    QFont groupFont;
    groupFont.setPixelSize(14);
    groupFont.setBold(true);

    QFont entryFont;
    entryFont.setPixelSize(12);
    entryFont.setBold(false);

    QFont keyFont;
    keyFont.setPixelSize(12);
    keyFont.setBold(true);
    keyFont.setFamily(QStringLiteral("Consolas"));

    int nCurrentY = nTitleY + 24;   // 20260322 ZJH 当前绘制 Y 位置
    int nLeftCol = nCardX + 24;     // 20260322 ZJH 左列起始 X
    int nRightCol = nCardX + nCardWidth / 2 + 12;  // 20260322 ZJH 右列起始 X
    int nColWidth = nCardWidth / 2 - 36;  // 20260322 ZJH 单列宽度

    // 20260322 ZJH 双列布局绘制快捷键分组
    int nGroupIndex = 0;
    int nStartY = nCurrentY;  // 20260322 ZJH 记录起始 Y，用于右列对齐

    for (const ShortcutGroup& group : vecGroups) {
        // 20260322 ZJH 决定是左列还是右列
        int nBaseX = (nGroupIndex < 2) ? nLeftCol : nRightCol;
        if (nGroupIndex == 2) {
            nCurrentY = nStartY;  // 20260322 ZJH 右列回到顶部
        }

        // 20260322 ZJH 绘制分组标题
        painter.setFont(groupFont);
        painter.setPen(QColor(37, 99, 235));  // 20260322 ZJH 蓝色标题 #2563eb
        painter.drawText(nBaseX, nCurrentY + 14, group.strTitle);
        nCurrentY += 22;  // 20260322 ZJH 标题下方间距

        // 20260322 ZJH 绘制分隔线
        painter.setPen(QPen(QColor(42, 45, 53), 1));
        painter.drawLine(nBaseX, nCurrentY, nBaseX + nColWidth, nCurrentY);
        nCurrentY += 6;

        // 20260322 ZJH 绘制各快捷键条目
        for (const ShortcutEntry& entry : group.vecEntries) {
            // 20260322 ZJH 绘制快捷键（等宽字体，高亮颜色）
            painter.setFont(keyFont);
            painter.setPen(QColor(96, 165, 250));  // 20260322 ZJH 蓝色亮色 #60a5fa

            // 20260322 ZJH 绘制快捷键背景块
            QFontMetrics fmKey(keyFont);
            int nKeyWidth = fmKey.horizontalAdvance(entry.strKey) + 12;
            QRect keyBgRect(nBaseX, nCurrentY, nKeyWidth, 18);
            painter.setBrush(QColor(30, 34, 48));  // 20260322 ZJH 深色背景 #1e2230
            painter.setPen(QColor(42, 45, 53));
            painter.drawRoundedRect(keyBgRect, 3, 3);

            painter.setPen(QColor(96, 165, 250));
            painter.drawText(QRect(nBaseX + 6, nCurrentY, nKeyWidth - 12, 18),
                             Qt::AlignLeft | Qt::AlignVCenter, entry.strKey);

            // 20260322 ZJH 绘制功能描述
            painter.setFont(entryFont);
            painter.setPen(QColor(148, 163, 184));  // 20260322 ZJH 灰色文字 #94a3b8
            painter.drawText(QRect(nBaseX + nKeyWidth + 8, nCurrentY, nColWidth - nKeyWidth - 8, 18),
                             Qt::AlignLeft | Qt::AlignVCenter, entry.strAction);

            nCurrentY += 22;  // 20260322 ZJH 条目间距
        }

        nCurrentY += 8;  // 20260322 ZJH 分组间距
        ++nGroupIndex;
    }

    // 20260322 ZJH 6. 绘制底部提示文字
    painter.setFont(entryFont);
    painter.setPen(QColor(100, 116, 139));  // 20260322 ZJH 深灰色 #64748b
    painter.drawText(QRect(nCardX, nCardY + nCardHeight - 36, nCardWidth, 24),
                     Qt::AlignCenter,
                     QStringLiteral("按任意键或点击关闭"));
}

// 20260324 ZJH 事件过滤器：监听父控件 Resize 事件，自动同步遮罩大小
// 参数 pWatched: 被监听的对象（父控件）
// 参数 pEvent: 事件对象
// 返回: false（不拦截事件，继续正常分发）
bool ShortcutHelpOverlay::eventFilter(QObject* pWatched, QEvent* pEvent)
{
    // 20260324 ZJH 仅处理父控件的 Resize 事件，且遮罩处于可见状态时才同步大小
    if (pWatched == parentWidget() && pEvent->type() == QEvent::Resize) {
        if (isVisible()) {
            resize(parentWidget()->size());  // 20260324 ZJH 同步调整为父控件新大小
            move(0, 0);                       // 20260324 ZJH 确保左上角对齐
        }
    }
    return QWidget::eventFilter(pWatched, pEvent);  // 20260324 ZJH 继续正常事件分发
}

// 20260322 ZJH 按键事件：任意键关闭
void ShortcutHelpOverlay::keyPressEvent(QKeyEvent* /*pEvent*/)
{
    hide();  // 20260322 ZJH 隐藏遮罩
}

// 20260322 ZJH 鼠标点击事件：点击关闭
void ShortcutHelpOverlay::mousePressEvent(QMouseEvent* /*pEvent*/)
{
    hide();  // 20260322 ZJH 隐藏遮罩
}
