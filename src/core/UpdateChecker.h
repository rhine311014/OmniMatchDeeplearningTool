// 20260330 ZJH 版本更新检查器
// 通过 HTTP 请求检查 GitHub Releases 或自定义更新服务器的最新版本
// 支持语义版本号比较（SemVer: major.minor.patch）
// 非阻塞异步检查，结果通过信号返回
#pragma once

#include <QObject>          // 20260330 ZJH 信号槽基类
#include <QString>          // 20260330 ZJH 字符串
#include <QNetworkAccessManager>  // 20260330 ZJH HTTP 网络请求管理器

// 20260330 ZJH 版本更新检查器
// 使用方法:
//   UpdateChecker checker;
//   connect(&checker, &UpdateChecker::updateCheckFinished, this, &MyWidget::onUpdateResult);
//   checker.checkForUpdates();
class UpdateChecker : public QObject
{
    Q_OBJECT

public:
    // 20260330 ZJH 构造函数
    explicit UpdateChecker(QObject* pParent = nullptr);

    // 20260330 ZJH 析构函数
    ~UpdateChecker() override = default;

    // 20260330 ZJH 版本信息结构体
    struct VersionInfo {
        QString strCurrentVersion;   // 20260330 ZJH 当前应用版本号（如 "2.1.0"）
        QString strLatestVersion;    // 20260330 ZJH 服务器上的最新版本号
        bool bUpdateAvailable;       // 20260330 ZJH 是否有新版本可用
        QString strDownloadUrl;      // 20260330 ZJH 新版本下载地址
        QString strReleaseNotes;     // 20260330 ZJH 新版本更新日志
    };

    // 20260330 ZJH 发起异步更新检查
    // 向更新服务器发送 GET 请求，解析 JSON 响应获取最新版本信息
    // 结果通过 updateCheckFinished 或 updateCheckFailed 信号返回
    void checkForUpdates();

    // 20260330 ZJH 获取当前应用版本号
    QString currentVersion() const;

    // 20260330 ZJH 设置当前应用版本号（默认从编译时宏读取）
    void setCurrentVersion(const QString& strVersion);

    // 20260330 ZJH 设置更新检查 URL（默认为 GitHub Releases API）
    void setUpdateUrl(const QString& strUrl);

    // 20260330 ZJH 比较语义版本号: 判断 strLatest 是否比 strCurrent 新
    // 支持格式: "major.minor.patch"（如 "2.1.0" vs "2.0.0"）
    // 参数: strCurrent - 当前版本, strLatest - 待比较版本
    // 返回: true 表示 strLatest 比 strCurrent 新
    static bool isNewerVersion(const QString& strCurrent, const QString& strLatest);

signals:
    // 20260330 ZJH 更新检查完成信号（成功获取到版本信息）
    void updateCheckFinished(const UpdateChecker::VersionInfo& info);

    // 20260330 ZJH 更新检查失败信号（网络错误/解析错误）
    void updateCheckFailed(const QString& strError);

private slots:
    // 20260330 ZJH HTTP 响应回调
    void onNetworkReply();

private:
    // 20260330 ZJH 解析 JSON 响应，提取版本信息
    // 参数: arrData - HTTP 响应 body 数据
    // 返回: 解析后的 VersionInfo（解析失败时 bUpdateAvailable = false）
    VersionInfo parseResponse(const QByteArray& arrData) const;

    QNetworkAccessManager* m_pNetworkManager;  // 20260330 ZJH HTTP 网络管理器
    QString m_strCurrentVersion;               // 20260330 ZJH 当前版本号
    QString m_strUpdateUrl;                    // 20260330 ZJH 更新检查 URL
};
