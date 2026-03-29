// 20260323 ZJH ResourceMonitorWidget — 系统资源监控控件
// 显示 CPU 使用率、内存使用、GPU 使用率和显存占用
// 定时刷新，带迷你条形图可视化
#pragma once

#include <QWidget>   // 20260323 ZJH 控件基类
#include <QTimer>    // 20260323 ZJH 定时刷新
#include <QLabel>    // 20260323 ZJH 文本标签

// 20260324 ZJH 前向声明异步进程
class QProcess;

// 20260323 ZJH 系统资源快照
struct ResourceSnapshot
{
    double dCpuPercent = 0.0;      // 20260323 ZJH CPU 使用率 (0~100)
    double dMemUsedMB = 0.0;       // 20260323 ZJH 内存使用（MB）
    double dMemTotalMB = 0.0;      // 20260323 ZJH 内存总量（MB）
    double dGpuPercent = 0.0;      // 20260323 ZJH GPU 使用率 (0~100)
    double dVramUsedMB = 0.0;      // 20260323 ZJH 显存使用（MB）
    double dVramTotalMB = 0.0;     // 20260323 ZJH 显存总量（MB）
};

// 20260323 ZJH 系统资源监控控件
class ResourceMonitorWidget : public QWidget
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit ResourceMonitorWidget(QWidget* pParent = nullptr);

    // 20260324 ZJH 析构函数，清理异步进程
    ~ResourceMonitorWidget() override;

    // 20260323 ZJH 开始监控（默认每 2 秒刷新一次）
    void startMonitoring(int nIntervalMs = 2000);

    // 20260323 ZJH 停止监控
    void stopMonitoring();

    // 20260323 ZJH 手动更新数据
    void updateSnapshot(const ResourceSnapshot& snapshot);

    // 20260324 ZJH 返回控件的推荐尺寸，供布局管理器参考
    QSize sizeHint() const override;

protected:
    // 20260323 ZJH 绘制事件
    void paintEvent(QPaintEvent* pEvent) override;

private slots:
    // 20260323 ZJH 定时刷新回调
    void onRefresh();

    // 20260324 ZJH CPU 查询进程完成回调
    void onCpuQueryFinished(int nExitCode);

    // 20260324 ZJH 内存查询进程完成回调
    void onMemQueryFinished(int nExitCode);

private:
    // 20260323 ZJH 绘制单行资源条
    void drawResourceBar(QPainter& painter, const QFont& font, int nY, const QString& strLabel,
                         double dPercent, const QColor& color) const;

    // 20260324 ZJH 启动异步 CPU 查询
    void startCpuQuery();

    // 20260324 ZJH 启动异步内存查询
    void startMemQuery();

    ResourceSnapshot m_snapshot;   // 20260323 ZJH 当前资源快照
    QTimer* m_pRefreshTimer;       // 20260323 ZJH 刷新定时器

    // 20260324 ZJH 异步查询进程指针（非阻塞，通过 finished 信号回调结果）
    QProcess* m_pCpuProcess = nullptr;   // 20260324 ZJH CPU 查询进程
    QProcess* m_pMemProcess = nullptr;   // 20260324 ZJH 内存查询进程
    bool m_bCpuQueryPending = false;     // 20260324 ZJH CPU 查询是否正在进行
    bool m_bMemQueryPending = false;     // 20260324 ZJH 内存查询是否正在进行
};
