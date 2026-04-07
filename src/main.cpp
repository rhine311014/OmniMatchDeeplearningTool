// 20260403 ZJH OmniMatch Deep Learning Tool — Qt6 应用入口
// 阶段机 + tickAnimation() 同步重绘，保证启动动画在任何阻塞期间都不冻结
// 核心原理: repaint() 直接调用 paintEvent()，不依赖事件循环

#include <QApplication>      // 20260322 ZJH Qt 应用基类
#include <QTimer>             // 20260322 ZJH 延迟定时器
#include <QIcon>              // 20260401 ZJH 应用图标
#include <QSvgRenderer>       // 20260401 ZJH SVG 渲染器
#include <QPainter>           // 20260401 ZJH 绘制 SVG 到 QPixmap
#include <QtConcurrent>       // 20260403 ZJH 后台线程执行 GPU 检测

#include "app/Application.h"           // 20260322 ZJH 全局事件总线
#include "app/ThemeManager.h"          // 20260322 ZJH 主题管理器
#include "app/MainWindow.h"            // 20260322 ZJH 主窗口
#include "ui/widgets/SplashScreen.h"   // 20260324 ZJH 启动画面

// 20260406 ZJH 应用程序入口函数
// 参数 argc: 命令行参数个数; argv: 命令行参数字符串数组
int main(int argc, char* argv[])
{
    // 20260406 ZJH 创建 Qt 应用对象，初始化事件循环和全局 qApp 指针
    QApplication app(argc, argv);

    // 20260328 ZJH 设置应用信息
    app.setApplicationName("OmniMatch Deep Learning Tool");
    app.setApplicationVersion(OM_VERSION);
    app.setOrganizationName("OmniMatch");

    // 20260401 ZJH 设置应用图标（SVG → 多尺寸位图 QIcon）
    {
        QIcon appIcon;  // 20260406 ZJH 多尺寸图标容器
        // 20260406 ZJH 从 Qt 资源系统加载 SVG 图标文件
        QSvgRenderer svgRenderer(QStringLiteral(":/icons/app_icon.svg"));
        if (svgRenderer.isValid()) {  // 20260406 ZJH SVG 加载成功才进行渲染
            // 20260406 ZJH 遍历 4 个常用图标尺寸，逐一渲染为位图
            for (int nSize : {16, 32, 48, 256}) {
                QPixmap pixmap(nSize, nSize);  // 20260406 ZJH 创建指定尺寸的透明画布
                pixmap.fill(Qt::transparent);  // 20260406 ZJH 填充透明背景
                QPainter painter(&pixmap);     // 20260406 ZJH 在画布上创建绘图器
                svgRenderer.render(&painter);  // 20260406 ZJH 将 SVG 渲染到画布
                painter.end();                 // 20260406 ZJH 结束绘制，释放绘图资源
                appIcon.addPixmap(pixmap);     // 20260406 ZJH 将该尺寸位图添加到 QIcon
            }
        }
        app.setWindowIcon(appIcon);  // 20260406 ZJH 设置全局应用图标（影响标题栏和任务栏）
    }

    // 20260403 ZJH 创建并显示启动画面
    auto* pSplash = new SplashScreen();
    pSplash->setStatus(QStringLiteral("正在初始化..."));
    pSplash->show();

    // 20260403 ZJH ===== 阶段机定时器 =====
    // 每个阻塞操作前后调用 pSplash->tickAnimation() 同步刷新粒子
    // tickAnimation() = updateParticles() + repaint()，直接调用 paintEvent，不经过事件循环
    // 这样即使页面构造函数耗时 100ms，也只会在单帧内卡顿，前后帧连贯

    // 20260406 ZJH 创建阶段机驱动定时器（堆分配，由 stage 14 手动删除）
    auto* pStageTimer = new QTimer();
    pStageTimer->setInterval(50);  // 20260403 ZJH 50ms 间隔，每阶段之间留足时间让 DWM 合成粒子帧

    int* pnStage = new int(0);                         // 20260406 ZJH 阶段计数器（堆分配，lambda 按值捕获指针）
    MainWindow** ppMainWindow = new MainWindow*(nullptr);  // 20260406 ZJH 主窗口指针的堆指针（跨阶段传递）
    QFuture<void>* pGpuFuture = new QFuture<void>();   // 20260406 ZJH GPU 检测异步任务句柄

    // 20260406 ZJH 连接定时器超时信号到阶段机 lambda，按值捕获所有堆指针
    QObject::connect(pStageTimer, &QTimer::timeout, [=]() {
        switch (*pnStage) {

        case 0:  // 20260406 ZJH 阶段0: 初始化 Application 单例
            pSplash->setStatus(QStringLiteral("正在初始化..."));
            pSplash->setProgress(5);
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞前刷新
            Application::instance();   // 20260406 ZJH 触发 Application 单例构造
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞后刷新
            break;

        case 1:  // 20260406 ZJH 阶段1: 异步启动 GPU 硬件检测
            pSplash->setStatus(QStringLiteral("正在检测 GPU 硬件..."));
            pSplash->setProgress(10);
            // 20260406 ZJH 在后台线程执行 GPU 探测，避免阻塞 UI 线程
            *pGpuFuture = QtConcurrent::run([]() {
                Application::instance()->initializePerformance();
            });
            break;

        case 2:  // 20260406 ZJH 阶段2: 等待 GPU 检测完成
            if (!pGpuFuture->isFinished()) {
                pSplash->tickAnimation();  // 20260403 ZJH 等待期间持续刷新粒子
                return;  // 20260403 ZJH 不推进阶段，下次定时器触发再检查
            }
            pSplash->setProgress(20);  // 20260406 ZJH GPU 检测完成，推进进度到 20%
            break;

        case 3:  // 20260406 ZJH 阶段3: 加载暗色主题 QSS
            pSplash->setStatus(QStringLiteral("正在加载主题..."));
            pSplash->setProgress(25);
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞前刷新
            // 20260406 ZJH 初始化主题管理器并应用默认暗色主题
            OmniMatch::ThemeManager::instance()->applyTheme(
                OmniMatch::ThemeManager::Theme::Dark);
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞后刷新
            break;

        case 4:  // 20260406 ZJH 阶段4: 创建主窗口骨架（延迟模式，不创建页面）
            pSplash->setStatus(QStringLiteral("正在创建窗口..."));
            pSplash->setProgress(30);
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞前刷新
            *ppMainWindow = new MainWindow(true);  // 20260406 ZJH true=延迟页面创建
            (*ppMainWindow)->setAttribute(Qt::WA_DeleteOnClose);  // 20260406 ZJH 关闭窗口时自动释放
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞后刷新
            break;

        case 5:  case 6:  case 7:  case 8:   // 20260406 ZJH 阶段5-12: 逐步创建 8 个页面
        case 9:  case 10: case 11: case 12:
        {
            int nPageIndex = *pnStage - 5;      // 20260406 ZJH 阶段号减 5 得到页面索引 (0-7)
            int nProgress = 35 + nPageIndex * 7; // 20260406 ZJH 线性计算进度百分比 (35%~91%)
            pSplash->setStatus(QStringLiteral("正在加载页面 %1/8 ...").arg(nPageIndex + 1));
            pSplash->setProgress(nProgress);
            pSplash->tickAnimation();  // 20260403 ZJH 页面构造前刷新 — 关键!
            (*ppMainWindow)->initPage(nPageIndex);
            pSplash->tickAnimation();  // 20260403 ZJH 页面构造后刷新 — 关键!
            break;
        }

        case 13:  // 20260406 ZJH 阶段13: 信号槽连接和最终初始化
            pSplash->setStatus(QStringLiteral("正在连接组件..."));
            pSplash->setProgress(95);
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞前刷新
            (*ppMainWindow)->finalizeInit();  // 20260406 ZJH 连接信号槽、初始化动画和状态栏
            pSplash->tickAnimation();  // 20260403 ZJH 阻塞后刷新
            break;

        case 14:  // 20260406 ZJH 阶段14: 启动完成，淡出启动画面并显示主窗口
        {
            pStageTimer->stop();  // 20260406 ZJH 停止阶段机定时器
            pSplash->setStatus(QStringLiteral("启动完成"));
            pSplash->setProgress(100);  // 20260406 ZJH 进度达到 100%

            MainWindow* pMW = *ppMainWindow;  // 20260406 ZJH 取出主窗口指针
            // 20260403 ZJH 设置主窗口图标（与应用图标一致，确保标题栏显示图标）
            pMW->setWindowIcon(QApplication::windowIcon());

            // 20260406 ZJH 连接启动画面淡出完成信号：淡出结束后显示主窗口并释放启动画面
            QObject::connect(pSplash, &SplashScreen::fadeOutFinished,
                             pSplash, [pMW, pSplash]() {
                pMW->show();             // 20260406 ZJH 显示主窗口
                pSplash->deleteLater();  // 20260406 ZJH 延迟释放启动画面内存
            });
            pSplash->fadeOut();  // 20260406 ZJH 开始启动画面淡出动画

            // 20260406 ZJH 释放阶段机辅助堆变量（定时器已停止，不会再访问）
            delete pnStage;
            delete ppMainWindow;
            delete pGpuFuture;
            delete pStageTimer;
            return;  // 20260406 ZJH 退出 lambda，不再推进阶段
        }

        default:  // 20260406 ZJH 未知阶段，静默跳过
            break;
        }

        ++(*pnStage);  // 20260406 ZJH 推进到下一阶段
    });

    pStageTimer->start();  // 20260406 ZJH 启动阶段机定时器，开始分步初始化
    return app.exec();     // 20260406 ZJH 进入 Qt 事件循环，程序在此阻塞直到退出
}
