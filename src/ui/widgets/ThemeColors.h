// 20260323 ZJH ThemeColors — 主题颜色常量集中定义
// 所有 QPainter 自绘控件共享的颜色、字体和绘图参数
// 避免硬编码颜色字符串分散在数十个文件中
#pragma once

#include <QColor>   // 20260323 ZJH QColor 类型
#include <QFont>    // 20260323 ZJH QFont 类型
#include <QString>  // 20260324 ZJH QString 类型，用于字体族名
#include <QVector>  // 20260323 ZJH QVector 类型

namespace ThemeColors {

// 20260324 ZJH 跨平台 CJK 字体族名，避免各控件硬编码不同字体
#ifdef Q_OS_WIN
    static const QString s_strFontFamily = QStringLiteral("Microsoft YaHei");   // 20260324 ZJH Windows 默认中文字体
#elif defined(Q_OS_MACOS)
    static const QString s_strFontFamily = QStringLiteral("PingFang SC");       // 20260324 ZJH macOS 默认中文字体
#else
    static const QString s_strFontFamily = QStringLiteral("Noto Sans CJK SC"); // 20260324 ZJH Linux 默认中文字体
#endif

// ===== 背景 =====
inline const QColor kBackground("#1a1d24");       // 20260323 ZJH 控件深色背景
inline const QColor kCardBackground("#22262e");    // 20260323 ZJH 卡片/面板背景
inline const QColor kDialogBackground("#1e2028");  // 20260323 ZJH 对话框背景

// ===== 文字 =====
inline const QColor kTextPrimary("#e2e8f0");       // 20260323 ZJH 主文字色
inline const QColor kTextSecondary("#94a3b8");     // 20260323 ZJH 次要文字/标签
inline const QColor kTextPlaceholder("#64748b");   // 20260323 ZJH 占位文字/禁用态

// ===== 边框/网格 =====
inline const QColor kBorder("#334155");            // 20260323 ZJH 边框/网格线
inline const QColor kAxisLine("#64748b");          // 20260323 ZJH 坐标轴线

// ===== 强调色 =====
inline const QColor kPrimaryBlue("#3b82f6");       // 20260323 ZJH 主强调色（蓝）
inline const QColor kAccentBlue("#2563eb");        // 20260323 ZJH 按钮/导航蓝
inline const QColor kSuccess("#10b981");           // 20260323 ZJH 成功/绿色
inline const QColor kWarning("#f59e0b");           // 20260323 ZJH 警告/橙色
inline const QColor kError("#ef4444");             // 20260323 ZJH 错误/红色
inline const QColor kPurple("#8b5cf6");            // 20260323 ZJH 紫色
inline const QColor kCyan("#06b6d4");              // 20260323 ZJH 青色
inline const QColor kPink("#ec4899");              // 20260323 ZJH 粉色
inline const QColor kLime("#84cc16");              // 20260323 ZJH 黄绿色
inline const QColor kOrange("#f97316");            // 20260323 ZJH 橙红色
inline const QColor kTeal("#14b8a6");              // 20260323 ZJH 青绿色

// ===== 共享图表颜色循环 =====
inline const QVector<QColor>& chartPalette() {
    static const QVector<QColor> s_palette = {
        kPrimaryBlue, kWarning, kSuccess, kError,
        kPurple, kCyan, kPink, kLime,
        kOrange, kTeal, QColor("#a855f7"), kTextPlaceholder
    };
    return s_palette;
}

}  // namespace ThemeColors
