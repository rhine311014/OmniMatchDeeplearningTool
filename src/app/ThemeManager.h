#pragma once
// 20260322 ZJH 主题管理器 — OmniMatch 深色/亮色主题切换
// 采用单例模式，全局唯一实例，负责加载 QSS 并应用到整个 QApplication

#include <QObject>
#include <QString>

// 20260322 ZJH 前向声明，避免不必要的头文件依赖
class QApplication;

// 20260322 ZJH OmniMatch 命名空间，与 OmniMatch 项目隔离
namespace OmniMatch {

// 20260322 ZJH 主题管理器类，单例模式
// 职责：从 Qt 资源系统加载 QSS 样式表，切换时发射 themeChanged 信号通知订阅者
class ThemeManager : public QObject
{
    Q_OBJECT

public:
    // 20260322 ZJH 主题枚举：暗色（默认）和亮色两种
    enum class Theme {
        Dark,   // 暗色主题，主背景 #0d1117，参考 MVTec DL Tool 深色风格
        Light   // 亮色主题，主背景 #f8fafc，预留明亮模式
    };

    // 20260322 ZJH 获取单例指针（Meyer's Singleton，线程安全）
    static ThemeManager* instance();

    // 20260322 ZJH 应用指定主题到 QApplication 全局样式
    // 参数 eTheme: 目标主题枚举值
    void applyTheme(Theme eTheme);

    // 20260322 ZJH 返回当前正在使用的主题枚举值
    Theme currentTheme() const;

signals:
    // 20260322 ZJH 主题切换完成后发射，携带新主题枚举值
    // 监听者可根据此信号动态更新无法用 QSS 覆盖的自绘控件
    void themeChanged(Theme eTheme);

private:
    // 20260322 ZJH 私有构造函数，阻止外部直接构造实例（单例）
    explicit ThemeManager(QObject* pParent = nullptr);

    // 20260322 ZJH 从 Qt 资源文件路径读取 QSS 文本内容
    // 参数 strPath: 形如 ":/themes/dark_theme.qss" 的资源路径
    // 返回: 完整 QSS 字符串，失败时返回空字符串
    QString loadStyleSheet(const QString& strPath);

    Theme m_eCurrentTheme = Theme::Dark;  // 当前主题，默认暗色
};

}  // namespace OmniMatch
