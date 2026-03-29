// 20260328 ZJH OmniMatch Deep Learning Tool — Qt6 应用入口
// 初始化全局事件总线 Application，探测 GPU 硬件
// 显示启动画面 → 应用暗色主题 → 创建主窗口 → 渐隐启动画面 → 显示主窗口

#include <QApplication>  // 20260322 ZJH Qt 应用基类
#include <QTimer>         // 20260322 ZJH 延迟定时器

#include "app/Application.h"           // 20260322 ZJH 全局事件总线
#include "app/ThemeManager.h"          // 20260322 ZJH 主题管理器
#include "app/MainWindow.h"            // 20260322 ZJH 主窗口
#include "ui/widgets/SplashScreen.h"   // 20260324 ZJH 启动画面

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // 20260328 ZJH 设置应用信息 — OmniMatch Deep Learning Tool
    app.setApplicationName("OmniMatch Deep Learning Tool");  // 20260328 ZJH 应用名称
    // 20260324 ZJH 使用 CMake 注入的 OM_VERSION 宏替代硬编码版本号
    app.setApplicationVersion(OM_VERSION);
    app.setOrganizationName("OmniMatch");  // 20260328 ZJH 组织名称

    // 20260324 ZJH 创建并显示启动画面（无边框置顶窗口，展示品牌和加载进度）
    auto* pSplash = new SplashScreen();
    pSplash->setStatus(QStringLiteral("正在初始化..."));
    pSplash->show();
    app.processEvents();  // 20260324 ZJH 强制刷新事件队列，确保启动画面立即可见

    // 20260324 ZJH 初始化全局事件总线 + GPU 硬件检测
    pSplash->setStatus(QStringLiteral("正在检测 GPU 硬件..."));
    app.processEvents();  // 20260324 ZJH 刷新状态文本
    Application::instance();
    Application::instance()->initializePerformance();

    // 20260324 ZJH 应用暗色主题
    pSplash->setStatus(QStringLiteral("正在加载主题..."));
    app.processEvents();  // 20260324 ZJH 刷新状态文本
    OmniMatch::ThemeManager::instance()->applyTheme(OmniMatch::ThemeManager::Theme::Dark);

    // 20260324 ZJH 创建主窗口（WA_DeleteOnClose 确保退出时释放内存）
    pSplash->setStatus(QStringLiteral("正在加载界面..."));
    app.processEvents();  // 20260324 ZJH 刷新状态文本
    auto* pMainWindow = new MainWindow();
    pMainWindow->setAttribute(Qt::WA_DeleteOnClose);

    // 20260324 ZJH 启动画面渐隐完成后显示主窗口并释放启动画面
    QObject::connect(pSplash, &SplashScreen::fadeOutFinished, pSplash, [pMainWindow, pSplash]() {
        pMainWindow->show();      // 20260324 ZJH 渐隐完成，显示主窗口
        pSplash->deleteLater();   // 20260324 ZJH 释放启动画面内存
    });

    // 20260324 ZJH 启动渐隐动画（500ms 过渡到主窗口）
    pSplash->setStatus(QStringLiteral("启动完成"));
    pSplash->fadeOut();

    return app.exec();
}
