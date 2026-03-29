// 20260323 ZJH TranslationManager — 国际化管理器
// 管理应用程序的多语言支持（中文/英文）
// 使用 Qt 翻译系统 (QTranslator) + 内置字符串映射双机制
// 单例模式，全局可访问
#pragma once

#include <QObject>       // 20260323 ZJH QObject 基类
#include <QString>       // 20260323 ZJH 字符串
#include <QMap>          // 20260323 ZJH 映射

// 20260323 ZJH 支持的语言枚举
enum class AppLanguage
{
    Chinese = 0,   // 20260323 ZJH 简体中文（默认）
    English        // 20260323 ZJH 英文
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

signals:
    // 20260323 ZJH 语言切换信号
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
    // 20260323 ZJH 内置翻译字典: key → {Chinese, English}
    QMap<QString, QPair<QString, QString>> m_mapTranslations;
};

// 20260323 ZJH 全局翻译宏（便捷使用）
#define OM_TR(key) TranslationManager::instance().translate(key)
