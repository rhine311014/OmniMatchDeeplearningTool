// 20260330 ZJH UpdateChecker 实现
// 异步 HTTP 版本检查 + JSON 解析 + 语义版本号比较

#include "core/UpdateChecker.h"

#include <QNetworkReply>      // 20260330 ZJH HTTP 响应对象
#include <QNetworkRequest>    // 20260330 ZJH HTTP 请求对象
#include <QJsonDocument>      // 20260330 ZJH JSON 文档解析
#include <QJsonObject>        // 20260330 ZJH JSON 对象
#include <QJsonArray>         // 20260330 ZJH JSON 数组（GitHub Releases API 返回数组）
#include <QUrl>               // 20260330 ZJH URL 构造
#include <QStringList>        // 20260330 ZJH 字符串分割


// 20260330 ZJH 默认当前版本（编译时确定）
static const QString s_strDefaultVersion = QStringLiteral("2.1.0");

// 20260330 ZJH 默认更新检查 URL（GitHub Releases API）
static const QString s_strDefaultUpdateUrl =
    QStringLiteral("https://api.github.com/repos/omnimatch/omnimatch-dl-tool/releases/latest");


// 20260330 ZJH 构造函数
UpdateChecker::UpdateChecker(QObject* pParent)
    : QObject(pParent)
    , m_pNetworkManager(new QNetworkAccessManager(this))  // 20260330 ZJH 创建网络管理器
    , m_strCurrentVersion(s_strDefaultVersion)             // 20260330 ZJH 初始化默认版本
    , m_strUpdateUrl(s_strDefaultUpdateUrl)                // 20260330 ZJH 初始化默认 URL
{
}

// 20260330 ZJH 获取当前版本号
QString UpdateChecker::currentVersion() const
{
    return m_strCurrentVersion;
}

// 20260330 ZJH 设置当前版本号
void UpdateChecker::setCurrentVersion(const QString& strVersion)
{
    m_strCurrentVersion = strVersion;  // 20260330 ZJH 覆盖默认版本
}

// 20260330 ZJH 设置更新检查 URL
void UpdateChecker::setUpdateUrl(const QString& strUrl)
{
    m_strUpdateUrl = strUrl;  // 20260330 ZJH 覆盖默认 URL
}

// 20260330 ZJH 发起异步更新检查
void UpdateChecker::checkForUpdates()
{
    // 20260330 ZJH 检查 URL 是否有效
    QUrl url(m_strUpdateUrl);
    if (!url.isValid()) {
        emit updateCheckFailed(QStringLiteral("Invalid update URL: ") + m_strUpdateUrl);
        return;
    }

    // 20260330 ZJH 构造 HTTP GET 请求
    QNetworkRequest request(url);
    // 20260330 ZJH 设置 User-Agent（GitHub API 要求）
    request.setRawHeader("User-Agent", "OmniMatch-DL-Tool/" + m_strCurrentVersion.toUtf8());
    // 20260330 ZJH 请求 JSON 格式响应
    request.setRawHeader("Accept", "application/vnd.github.v3+json");

    // 20260330 ZJH 发送异步 GET 请求
    QNetworkReply* pReply = m_pNetworkManager->get(request);

    // 20260330 ZJH 连接响应完成信号
    connect(pReply, &QNetworkReply::finished, this, &UpdateChecker::onNetworkReply);
}

// 20260330 ZJH HTTP 响应回调
void UpdateChecker::onNetworkReply()
{
    // 20260330 ZJH 获取 QNetworkReply 发送者
    QNetworkReply* pReply = qobject_cast<QNetworkReply*>(sender());
    if (!pReply) {
        emit updateCheckFailed(QStringLiteral("Internal error: null reply"));
        return;
    }

    // 20260330 ZJH 确保 reply 对象在函数结束后被释放
    pReply->deleteLater();

    // 20260330 ZJH 检查网络错误
    if (pReply->error() != QNetworkReply::NoError) {
        emit updateCheckFailed(
            QStringLiteral("Network error: ") + pReply->errorString());
        return;
    }

    // 20260330 ZJH 读取响应数据
    QByteArray arrData = pReply->readAll();

    // 20260330 ZJH 检查响应是否为空
    if (arrData.isEmpty()) {
        emit updateCheckFailed(QStringLiteral("Empty response from update server"));
        return;
    }

    // 20260330 ZJH 解析 JSON 响应
    VersionInfo info = parseResponse(arrData);

    // 20260330 ZJH 发射结果信号
    emit updateCheckFinished(info);
}

// 20260330 ZJH 解析 JSON 响应
UpdateChecker::VersionInfo UpdateChecker::parseResponse(const QByteArray& arrData) const
{
    VersionInfo info;
    info.strCurrentVersion = m_strCurrentVersion;  // 20260330 ZJH 填入当前版本
    info.bUpdateAvailable = false;                  // 20260330 ZJH 默认无更新
    info.strLatestVersion = m_strCurrentVersion;    // 20260330 ZJH 默认等于当前版本

    // 20260330 ZJH 解析 JSON
    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(arrData, &parseError);

    if (parseError.error != QJsonParseError::NoError) {
        // 20260330 ZJH JSON 解析失败，返回默认值
        return info;
    }

    QJsonObject obj;

    // 20260330 ZJH 兼容两种格式: GitHub Releases API (对象) 和自定义 API (可能是数组)
    if (doc.isObject()) {
        obj = doc.object();  // 20260330 ZJH 直接使用对象
    } else if (doc.isArray()) {
        QJsonArray arr = doc.array();
        if (!arr.isEmpty()) {
            obj = arr.first().toObject();  // 20260330 ZJH 取数组第一个元素
        }
    }

    if (obj.isEmpty()) {
        return info;  // 20260330 ZJH 空对象，返回默认值
    }

    // 20260330 ZJH 提取版本号（GitHub 格式: "tag_name": "v2.2.0"）
    QString strTagName = obj.value(QStringLiteral("tag_name")).toString();
    // 20260330 ZJH 去除可能的 "v" 前缀（如 "v2.2.0" → "2.2.0"）
    if (strTagName.startsWith(QLatin1Char('v')) || strTagName.startsWith(QLatin1Char('V'))) {
        strTagName = strTagName.mid(1);
    }
    info.strLatestVersion = strTagName;

    // 20260330 ZJH 提取下载地址（优先 "browser_download_url"，回退到 "html_url"）
    if (obj.contains(QStringLiteral("assets"))) {
        QJsonArray arrAssets = obj.value(QStringLiteral("assets")).toArray();
        if (!arrAssets.isEmpty()) {
            QJsonObject firstAsset = arrAssets.first().toObject();
            info.strDownloadUrl = firstAsset.value(
                QStringLiteral("browser_download_url")).toString();
        }
    }
    // 20260330 ZJH 回退到 HTML URL
    if (info.strDownloadUrl.isEmpty()) {
        info.strDownloadUrl = obj.value(QStringLiteral("html_url")).toString();
    }

    // 20260330 ZJH 提取更新日志
    info.strReleaseNotes = obj.value(QStringLiteral("body")).toString();

    // 20260330 ZJH 比较版本号，判断是否有新版本
    info.bUpdateAvailable = isNewerVersion(m_strCurrentVersion, strTagName);

    return info;
}

// 20260330 ZJH 语义版本号比较
bool UpdateChecker::isNewerVersion(const QString& strCurrent, const QString& strLatest)
{
    // 20260330 ZJH 分割版本号为 major.minor.patch 三段
    QStringList listCurrent = strCurrent.split(QLatin1Char('.'));
    QStringList listLatest = strLatest.split(QLatin1Char('.'));

    // 20260330 ZJH 补齐缺失段为 0（如 "2.1" → "2.1.0"）
    while (listCurrent.size() < 3) listCurrent.append(QStringLiteral("0"));
    while (listLatest.size() < 3) listLatest.append(QStringLiteral("0"));

    // 20260330 ZJH 逐段比较: major → minor → patch
    for (int i = 0; i < 3; ++i) {
        int nCur = listCurrent[i].toInt();   // 20260330 ZJH 当前版本段
        int nLat = listLatest[i].toInt();    // 20260330 ZJH 最新版本段

        if (nLat > nCur) {
            return true;   // 20260330 ZJH 最新版本更高
        } else if (nLat < nCur) {
            return false;  // 20260330 ZJH 当前版本更高（本地比服务器新）
        }
        // 20260330 ZJH 相等则继续比较下一段
    }

    return false;  // 20260330 ZJH 完全相同，无需更新
}
