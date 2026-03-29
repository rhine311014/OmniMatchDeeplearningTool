// 20260323 ZJH ResourceMonitorWidget — 系统资源监控控件实现
// 定时查询 CPU/内存/GPU 使用情况并以条形图形式显示
// 20260324 ZJH 重构：将 QProcess 从同步 waitForFinished() 改为异步 finished() 信号回调
// 消除每次刷新时阻塞 UI 线程 500ms~1s 的性能问题

#include "ui/widgets/ResourceMonitorWidget.h"

#include <QPainter>              // 20260323 ZJH 绘图引擎
#include <QProcess>              // 20260323 ZJH 跨平台进程调用
#include <QRegularExpression>    // 20260323 ZJH 正则表达式解析输出

// 20260323 ZJH 构造函数
ResourceMonitorWidget::ResourceMonitorWidget(QWidget* pParent)
    : QWidget(pParent)
    , m_pRefreshTimer(new QTimer(this))
{
    setFixedHeight(100);
    setMinimumWidth(180);
    connect(m_pRefreshTimer, &QTimer::timeout, this, &ResourceMonitorWidget::onRefresh);
}

// 20260324 ZJH 析构函数，终止并清理异步进程
ResourceMonitorWidget::~ResourceMonitorWidget()
{
    // 20260324 ZJH 停止定时器防止析构期间再次触发
    m_pRefreshTimer->stop();

    // 20260324 ZJH 终止正在运行的 CPU 查询进程
    if (m_pCpuProcess) {
        m_pCpuProcess->kill();  // 20260324 ZJH 强制终止
        m_pCpuProcess->waitForFinished(500);  // 20260324 ZJH 短暂等待进程退出
    }

    // 20260324 ZJH 终止正在运行的内存查询进程
    if (m_pMemProcess) {
        m_pMemProcess->kill();  // 20260324 ZJH 强制终止
        m_pMemProcess->waitForFinished(500);  // 20260324 ZJH 短暂等待进程退出
    }
}

// 20260324 ZJH 返回控件的推荐尺寸（250x120），供布局管理器参考
QSize ResourceMonitorWidget::sizeHint() const
{
    return QSize(250, 120);  // 20260324 ZJH 资源监控控件推荐尺寸（4行条形图 + 边距）
}

// 20260323 ZJH 开始监控
void ResourceMonitorWidget::startMonitoring(int nIntervalMs)
{
    onRefresh();
    m_pRefreshTimer->start(nIntervalMs);
}

// 20260323 ZJH 停止监控
void ResourceMonitorWidget::stopMonitoring()
{
    m_pRefreshTimer->stop();
}

// 20260323 ZJH 手动更新数据
void ResourceMonitorWidget::updateSnapshot(const ResourceSnapshot& snapshot)
{
    m_snapshot = snapshot;
    update();
}

// 20260324 ZJH 定时刷新回调：启动异步查询（不阻塞 UI 线程）
void ResourceMonitorWidget::onRefresh()
{
    // 20260324 ZJH 启动 CPU 和内存异步查询
    startCpuQuery();
    startMemQuery();

    // 20260324 ZJH GPU 信息预留（需要 nvidia-smi 或 nvml）
    m_snapshot.dGpuPercent  = 0.0;
    m_snapshot.dVramUsedMB  = 0.0;
    m_snapshot.dVramTotalMB = 0.0;

    // 20260324 ZJH 立即使用缓存的快照数据重绘（异步回调到达时会再次 update）
    update();
}

// 20260324 ZJH 启动异步 CPU 查询
void ResourceMonitorWidget::startCpuQuery()
{
    // 20260324 ZJH 如果上一次查询仍在运行，跳过本次（避免进程堆积）
    if (m_bCpuQueryPending) {
        return;  // 20260324 ZJH 上次查询未完成，使用缓存值
    }

    // 20260324 ZJH 延迟创建进程对象（首次使用时创建）
    if (!m_pCpuProcess) {
        m_pCpuProcess = new QProcess(this);  // 20260324 ZJH 父对象为 this，自动销毁
        m_pCpuProcess->setProcessChannelMode(QProcess::MergedChannels);
        // 20260324 ZJH 连接 finished 信号到回调槽
        connect(m_pCpuProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                this, &ResourceMonitorWidget::onCpuQueryFinished);
    }

    m_bCpuQueryPending = true;  // 20260324 ZJH 标记查询正在进行

#if defined(Q_OS_WIN)
    // 20260323 ZJH Windows: wmic cpu get LoadPercentage
    m_pCpuProcess->start("wmic", {"cpu", "get", "LoadPercentage", "/value"});
#elif defined(Q_OS_LINUX)
    // 20260323 ZJH Linux: 读取 /proc/stat 计算 CPU 使用率
    m_pCpuProcess->start("sh", {"-c", "grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}'"});
#elif defined(Q_OS_MACOS)
    m_pCpuProcess->start("sh", {"-c", "ps -A -o %cpu | awk '{s+=$1} END {print s}'"});
#endif
}

// 20260324 ZJH 启动异步内存查询
void ResourceMonitorWidget::startMemQuery()
{
    // 20260324 ZJH 如果上一次查询仍在运行，跳过本次
    if (m_bMemQueryPending) {
        return;  // 20260324 ZJH 上次查询未完成，使用缓存值
    }

    // 20260324 ZJH 延迟创建进程对象
    if (!m_pMemProcess) {
        m_pMemProcess = new QProcess(this);  // 20260324 ZJH 父对象为 this，自动销毁
        m_pMemProcess->setProcessChannelMode(QProcess::MergedChannels);
        // 20260324 ZJH 连接 finished 信号到回调槽
        connect(m_pMemProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                this, &ResourceMonitorWidget::onMemQueryFinished);
    }

    m_bMemQueryPending = true;  // 20260324 ZJH 标记查询正在进行

#if defined(Q_OS_WIN)
    m_pMemProcess->start("wmic", {"OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/value"});
#elif defined(Q_OS_LINUX)
    m_pMemProcess->start("sh", {"-c", "free -m | grep Mem | awk '{print $2,$3}'"});
#elif defined(Q_OS_MACOS)
    m_pMemProcess->start("sh", {"-c", "vm_stat | head -5"});
#endif
}

// 20260324 ZJH CPU 查询进程完成回调（在主线程调用，不阻塞 UI）
void ResourceMonitorWidget::onCpuQueryFinished(int /*nExitCode*/)
{
    m_bCpuQueryPending = false;  // 20260324 ZJH 标记查询完成

    if (!m_pCpuProcess) {
        return;  // 20260324 ZJH 防御性检查
    }

    QString strOutput = m_pCpuProcess->readAllStandardOutput().trimmed();
    // 20260323 ZJH 从输出提取数字
    QRegularExpression re("(\\d+\\.?\\d*)");
    auto match = re.match(strOutput);
    if (match.hasMatch()) {
        m_snapshot.dCpuPercent = match.captured(1).toDouble();
    }

    // 20260324 ZJH 异步数据到达，触发重绘显示最新值
    update();
}

// 20260324 ZJH 内存查询进程完成回调（在主线程调用，不阻塞 UI）
void ResourceMonitorWidget::onMemQueryFinished(int /*nExitCode*/)
{
    m_bMemQueryPending = false;  // 20260324 ZJH 标记查询完成

    if (!m_pMemProcess) {
        return;  // 20260324 ZJH 防御性检查
    }

    QString strOutput = m_pMemProcess->readAllStandardOutput();

#if defined(Q_OS_WIN)
    // 20260323 ZJH 解析 wmic 输出（KB 单位）
    QRegularExpression reFree("FreePhysicalMemory=(\\d+)");
    QRegularExpression reTotal("TotalVisibleMemorySize=(\\d+)");
    auto matchFree = reFree.match(strOutput);
    auto matchTotal = reTotal.match(strOutput);
    if (matchTotal.hasMatch()) {
        m_snapshot.dMemTotalMB = matchTotal.captured(1).toDouble() / 1024.0;
    }
    if (matchFree.hasMatch()) {
        double dFreeMB = matchFree.captured(1).toDouble() / 1024.0;
        m_snapshot.dMemUsedMB = m_snapshot.dMemTotalMB - dFreeMB;
    }
#elif defined(Q_OS_LINUX)
    QStringList parts = strOutput.trimmed().split(QRegularExpression("\\s+"));
    if (parts.size() >= 2) {
        m_snapshot.dMemTotalMB = parts[0].toDouble();
        m_snapshot.dMemUsedMB  = parts[1].toDouble();
    }
#endif

    // 20260324 ZJH 异步数据到达，触发重绘显示最新值
    update();
}

// 20260323 ZJH 绘制事件
void ResourceMonitorWidget::paintEvent(QPaintEvent* /*pEvent*/)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(rect(), QColor("#1a1d24"));

    double dMemPercent = (m_snapshot.dMemTotalMB > 0) ?
        (m_snapshot.dMemUsedMB / m_snapshot.dMemTotalMB * 100.0) : 0.0;
    double dVramPercent = (m_snapshot.dVramTotalMB > 0) ?
        (m_snapshot.dVramUsedMB / m_snapshot.dVramTotalMB * 100.0) : 0.0;

    // 20260323 ZJH 缓存字体避免每帧构造
    static const QFont s_fontLabel("Segoe UI", 8);

    drawResourceBar(painter, s_fontLabel, 4,  "CPU",  m_snapshot.dCpuPercent, QColor("#3b82f6"));
    drawResourceBar(painter, s_fontLabel, 28, "MEM",  dMemPercent,            QColor("#10b981"));
    drawResourceBar(painter, s_fontLabel, 52, "GPU",  m_snapshot.dGpuPercent, QColor("#f59e0b"));
    drawResourceBar(painter, s_fontLabel, 76, "VRAM", dVramPercent,           QColor("#8b5cf6"));
}

// 20260323 ZJH 绘制单行资源条
void ResourceMonitorWidget::drawResourceBar(
    QPainter& painter, const QFont& font, int nY, const QString& strLabel,
    double dPercent, const QColor& color) const
{
    int nLabelW = 40;
    int nBarX = nLabelW + 4;
    int nBarW = width() - nBarX - 45;
    int nBarH = 14;

    painter.setPen(QColor("#94a3b8"));
    painter.setFont(font);
    painter.drawText(QRectF(4, nY, nLabelW, 20), Qt::AlignLeft | Qt::AlignVCenter, strLabel);

    QRectF bgRect(nBarX, nY + 3, nBarW, nBarH);
    painter.fillRect(bgRect, QColor("#22262e"));

    double dFillW = nBarW * qMin(dPercent, 100.0) / 100.0;
    if (dFillW > 0) {
        QRectF fillRect(nBarX, nY + 3, dFillW, nBarH);
        QColor fillColor = (dPercent > 80.0) ? QColor("#ef4444") : color;
        painter.fillRect(fillRect, fillColor);
    }

    painter.setPen(QColor("#e2e8f0"));
    painter.drawText(QRectF(nBarX + nBarW + 4, nY, 40, 20),
                    Qt::AlignLeft | Qt::AlignVCenter,
                    QString::number(dPercent, 'f', 0) + "%");
}
