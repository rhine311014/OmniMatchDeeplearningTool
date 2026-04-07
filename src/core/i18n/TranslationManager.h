// 20260323 ZJH TranslationManager — 国际化管理器
// 管理应用程序的多语言支持（中文/英文/日文/德文）
// 使用 Qt 翻译系统 (QTranslator) + 内置字符串映射双机制
// 支持系统 locale 自动检测 + 运行时动态切换（无需重启）
// 单例模式，全局可访问
#pragma once

#include <QObject>       // 20260323 ZJH QObject 基类
#include <QString>       // 20260323 ZJH 字符串
#include <QMap>          // 20260323 ZJH 映射
#include <QVector>       // 20260330 ZJH 多语言翻译向量

// 20260330 ZJH 支持的语言枚举（扩展为 4 种语言）
enum class AppLanguage
{
    Chinese  = 0,   // 20260323 ZJH 简体中文（默认）
    English  = 1,   // 20260323 ZJH 英文
    Japanese = 2,   // 20260330 ZJH 日文 (ja_JP)
    German   = 3    // 20260330 ZJH 德文 (de_DE)
};

// 20260330 ZJH 翻译条目：存储一个 key 在所有语言下的翻译文本
// 索引: 0=Chinese, 1=English, 2=Japanese, 3=German
struct TranslationEntry {
    QString arrTexts[4];  // 20260330 ZJH 四种语言的翻译文本数组
};

// 20260323 ZJH 国际化管理器
class TranslationManager : public QObject
{
    Q_OBJECT

public:
    // 20260323 ZJH 获取单例实例（Meyer's Singleton）
    static TranslationManager& instance();

    // 20260323 ZJH 设置当前语言
    // 参数: lang - 目标语言
    void setLanguage(AppLanguage lang);

    // 20260323 ZJH 获取当前语言
    AppLanguage currentLanguage() const;

    // 20260323 ZJH 翻译字符串
    // 参数: strKey - 翻译键名
    // 返回: 当前语言的翻译文本，找不到则返回键名本身
    QString translate(const QString& strKey) const;

    // 20260323 ZJH 获取语言显示名称
    static QString languageDisplayName(AppLanguage lang);

    // 20260330 ZJH 获取所有支持的语言列表
    static QVector<AppLanguage> supportedLanguages();

    // 20260330 ZJH 从系统 locale 自动检测语言
    // 检查 QLocale::system().name() 的前缀匹配：zh→Chinese, ja→Japanese, de→German, 否则→English
    // 返回: 自动检测到的语言
    static AppLanguage detectSystemLanguage();

    // 20260330 ZJH 语言枚举 ↔ locale 字符串互转
    // 参数: lang - 语言枚举
    // 返回: 对应的 locale 字符串（如 "zh_CN", "en_US", "ja_JP", "de_DE"）
    static QString languageToLocale(AppLanguage lang);

    // 20260330 ZJH locale 字符串 → 语言枚举
    // 参数: strLocale - locale 字符串
    // 返回: 匹配的语言枚举（找不到则返回 English）
    static AppLanguage localeToLanguage(const QString& strLocale);

signals:
    // 20260323 ZJH 语言切换信号（动态切换时所有 UI 组件应监听此信号并刷新文本）
    // 参数: lang - 新语言
    void languageChanged(AppLanguage lang);

private:
    // 20260323 ZJH 私有构造函数（单例）
    TranslationManager();

    // 20260323 ZJH 禁止拷贝
    TranslationManager(const TranslationManager&) = delete;
    TranslationManager& operator=(const TranslationManager&) = delete;

    // 20260323 ZJH 加载内置翻译字典
    void loadBuiltinTranslations();

    AppLanguage m_currentLang = AppLanguage::Chinese;  // 20260323 ZJH 当前语言
    // 20260330 ZJH 内置翻译字典: key → TranslationEntry (4语言)
    QMap<QString, TranslationEntry> m_mapTranslations;
};

// 20260323 ZJH 全局翻译宏（便捷使用）
#define OM_TR(key) TranslationManager::instance().translate(key)
