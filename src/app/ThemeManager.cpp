// 20260322 ZJH 主题管理器实现
// 负责从 Qt 资源系统加载 QSS 并通过 qApp->setStyleSheet() 全局应用

#include "ThemeManager.h"
#include <QApplication>   // 20260322 ZJH qApp 宏，访问全局 QApplication 实例
#include <QFile>          // 20260322 ZJH 读取 Qt 资源文件
#include <QTextStream>    // 20260322 ZJH 文本流读取 QSS 内容
#include <QDebug>         // 20260324 ZJH qDebug 输出未知主题警告

namespace OmniMatch {

// 20260324 ZJH 获取单例实例
// 使用堆分配 + qApp 父对象，确保在 QApplication 析构前销毁
// 避免 Meyer's Singleton（静态局部变量）在 QApplication 退出后才析构导致未定义行为
ThemeManager* ThemeManager::instance()
{
    // 20260324 ZJH 堆分配单例，以 qApp 为父对象
    // QObject 父子机制保证在 QApplication 析构时自动 delete 此单例
    static ThemeManager* s_pInst = new ThemeManager(qApp);
    return s_pInst;  // 20260324 ZJH 返回唯一实例指针
}

// 20260322 ZJH 私有构造函数（外部无法直接 new ThemeManager）
ThemeManager::ThemeManager(QObject* pParent)
    : QObject(pParent)
    , m_eCurrentTheme(Theme::Dark)  // 默认暗色主题
{
}

// 20260322 ZJH 应用目标主题到 QApplication 全局样式表
void ThemeManager::applyTheme(Theme eTheme)
{
    QString strQssPath;  // 待加载的 QSS 资源路径

    // 20260322 ZJH 根据主题枚举选择对应的 QSS 资源路径
    switch (eTheme) {
    case Theme::Dark:
        // 20260322 ZJH 暗色主题 QSS 路径，前缀 ":/" 表示 Qt 资源系统
        strQssPath = QStringLiteral(":/themes/dark_theme.qss");
        break;
    case Theme::Light:
        // 20260322 ZJH 亮色主题 QSS 路径
        strQssPath = QStringLiteral(":/themes/light_theme.qss");
        break;
    default:
        // 20260324 ZJH 未知主题枚举值，输出警告并回退到暗色主题
        qDebug() << "[ThemeManager] Unknown theme value" << static_cast<int>(eTheme)
                 << ", falling back to Dark theme";
        strQssPath = QStringLiteral(":/themes/dark_theme.qss");
        eTheme = Theme::Dark;  // 20260324 ZJH 修正为暗色主题，确保后续记录正确
        break;
    }

    // 20260322 ZJH 加载 QSS 文件内容
    const QString strStyleSheet = loadStyleSheet(strQssPath);

    if (!strStyleSheet.isEmpty()) {
        // 20260322 ZJH 将样式表应用到 QApplication，影响所有 QWidget
        qApp->setStyleSheet(strStyleSheet);
    }

    m_eCurrentTheme = eTheme;      // 20260322 ZJH 记录当前生效主题
    emit themeChanged(eTheme);     // 20260322 ZJH 通知订阅者主题已变更
}

// 20260322 ZJH 返回当前正在使用的主题枚举值
ThemeManager::Theme ThemeManager::currentTheme() const
{
    return m_eCurrentTheme;  // 直接返回成员变量
}

// 20260322 ZJH 从 Qt 资源路径读取完整 QSS 文本
QString ThemeManager::loadStyleSheet(const QString& strPath)
{
    QFile file(strPath);  // 20260322 ZJH 用资源路径构造 QFile

    // 20260322 ZJH 以只读文本模式打开文件；失败说明资源未正确嵌入
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return QString();  // 返回空字符串，调用方判空后不应用样式
    }

    QTextStream stream(&file);  // 20260322 ZJH 文本流包装文件，方便整体读取
    stream.setEncoding(QStringConverter::Utf8);  // 20260322 ZJH 强制 UTF-8，确保中文字符正确

    return stream.readAll();  // 20260322 ZJH 一次性读取全部 QSS 文本并返回
}

}  // namespace OmniMatch
