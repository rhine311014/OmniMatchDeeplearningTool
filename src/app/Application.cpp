// 20260322 ZJH Application 单例实现
// 全局事件总线 + 项目管理 + GPU 硬件探测
// GPU 检测采用动态加载 nvcuda.dll 的方式，不依赖 CUDA SDK

#include "app/Application.h"
#include "core/project/Project.h"  // 20260322 ZJH 完整类型定义，使 unique_ptr<Project> 析构合法

#include <QApplication>  // 20260324 ZJH 提供 qApp 宏（全局 QApplication 指针）
#include <QLibrary>      // 20260323 ZJH Qt 跨平台动态库加载（替代 Windows LoadLibraryA）
#include <QDebug>        // 20260322 ZJH 用于 qDebug 日志输出

// 20260324 ZJH 获取进程内唯一的 Application 实例
// 使用堆分配 + qApp 父对象，确保在 QApplication 析构前销毁
// 避免 Meyer's Singleton（静态局部变量）在 QApplication 退出后才析构导致未定义行为
Application* Application::instance()
{
    // 20260324 ZJH 堆分配单例，以 qApp 为父对象
    // QObject 父子机制保证在 QApplication 析构时自动 delete 此单例
    static Application* s_pInst = new Application(qApp);
    return s_pInst;
}

// 20260322 ZJH 私有构造函数：初始化成员变量，不执行耗时操作
// 耗时操作（如 GPU 探测）延迟到 initializePerformance() 中执行
Application::Application(QObject* pParent)
    : QObject(pParent)             // 20260322 ZJH 初始化 QObject 基类
    , m_pCurrentProject(nullptr)   // 20260322 ZJH 初始时无打开的项目
    , m_strGpuName("No GPU detected")  // 20260322 ZJH GPU 名称默认为未检测到
    , m_nGpuVramMB(0)              // 20260322 ZJH GPU 显存默认为 0
    , m_bHasGpu(false)             // 20260322 ZJH 默认无 GPU
{
    qDebug() << "[OmniMatch] Application singleton constructed";  // 20260322 ZJH 构造日志
}

// 20260322 ZJH 析构函数必须在 .cpp 中定义
// 原因：unique_ptr<Project> 的析构需要 Project 类型完整，而 .h 中仅前向声明
Application::~Application() = default;

// ===== 项目管理 =====

// 20260322 ZJH 获取当前项目指针
// 返回值：指向当前 Project 的裸指针，未打开项目时为 nullptr
Project* Application::currentProject() const
{
    return m_pCurrentProject.get();  // 20260322 ZJH 从 unique_ptr 获取裸指针，不转移所有权
}

// 20260322 ZJH 设置（替换）当前项目
// pProject 为 nullptr 时等同于关闭项目；传入新项目时旧项目自动析构
void Application::setCurrentProject(std::unique_ptr<Project> pProject)
{
    // 20260322 ZJH 旧项目被 unique_ptr 自动释放
    m_pCurrentProject = std::move(pProject);
}

// 20260322 ZJH 检查当前是否有有效的已打开项目
// 只检查指针非空，具体有效性验证由调用者负责
bool Application::hasValidProject() const
{
    // 20260322 ZJH unique_ptr 转 bool，非空则返回 true
    return m_pCurrentProject != nullptr;
}

// ===== 硬件信息 getter =====

// 20260322 ZJH 返回 GPU 显卡名称（需先调用 initializePerformance）
QString Application::gpuName() const
{
    return m_strGpuName;  // 20260322 ZJH 返回探测得到的 GPU 名称
}

// 20260322 ZJH 返回 GPU 显存大小（MB，需先调用 initializePerformance）
size_t Application::gpuVramMB() const
{
    return m_nGpuVramMB;  // 20260322 ZJH 返回探测得到的显存容量
}

// 20260322 ZJH 返回是否存在可用的 NVIDIA GPU
bool Application::hasGpu() const
{
    return m_bHasGpu;  // 20260322 ZJH 返回 GPU 存在标志
}

// ===== 信号通知方法 =====
// 20260324 ZJH 以下方法封装信号发射，禁止外部直接 emit Application 的信号
// 确保信号只从 Application 对象自身内部发射，符合 Qt 信号-槽所有权规范

// 20260324 ZJH 通知项目已创建
void Application::notifyProjectCreated(Project* pProject)
{
    emit projectCreated(pProject);  // 20260324 ZJH 从自身发射信号
}

// 20260324 ZJH 通知项目已打开
void Application::notifyProjectOpened(Project* pProject)
{
    emit projectOpened(pProject);  // 20260324 ZJH 从自身发射信号
}

// 20260324 ZJH 通知项目已关闭
void Application::notifyProjectClosed()
{
    emit projectClosed();  // 20260324 ZJH 从自身发射信号
}

// 20260324 ZJH 通知项目已保存
void Application::notifyProjectSaved()
{
    emit projectSaved();  // 20260324 ZJH 从自身发射信号
}

// 20260324 ZJH 通知请求导航到指定页面
void Application::notifyNavigateToPage(int nPageIndex)
{
    emit requestNavigateToPage(nPageIndex);  // 20260324 ZJH 从自身发射信号
}

// 20260324 ZJH 通知请求打开指定图像
// 20260404 ZJH 扩展: 可选传递标注 UUID，用于检查页实例双击后跳转到图像页并选中标注
void Application::notifyOpenImage(const QString& strImageUuid, const QString& strAnnotationUuid)
{
    emit requestOpenImage(strImageUuid, strAnnotationUuid);  // 20260404 ZJH 携带标注 UUID
}

// 20260324 ZJH 通知全局设置已变更
void Application::notifyGlobalSettingsChanged()
{
    emit globalSettingsChanged();  // 20260324 ZJH 从自身发射信号
}

// 20260324 ZJH 通知评估任务完成
void Application::notifyEvaluationCompleted()
{
    emit evaluationCompleted();  // 20260324 ZJH 从自身发射信号
}

// ===== GPU 硬件探测 =====

// 20260322 ZJH 初始化硬件性能信息
// 通过动态加载 nvcuda.dll 检测 NVIDIA GPU，无需安装 CUDA SDK
// 支持平台：Windows（其他平台跳过探测）
void Application::initializePerformance()
{
    qDebug() << "[OmniMatch] initializePerformance: 开始 GPU 探测";

    // 20260323 ZJH 使用 QLibrary 跨平台动态加载 CUDA Driver（替代 Windows LoadLibraryA）
    QLibrary cudaLib("nvcuda");  // 20260323 ZJH Qt 自动追加平台后缀（.dll / .so）
    if (!cudaLib.load()) {
        qDebug() << "[OmniMatch] nvcuda 未找到，系统无 NVIDIA GPU";
        m_bHasGpu    = false;
        m_strGpuName = QStringLiteral("No GPU detected");
        m_nGpuVramMB = 0;
        return;
    }

    // 20260323 ZJH CUDA Driver API 函数指针类型（最小子集）
    using CuInitFn           = int(*)(unsigned int);
    using CuDeviceGetCountFn = int(*)(int*);
    using CuDeviceGetNameFn  = int(*)(char*, int, int);
    using CuDeviceTotalMemFn = int(*)(size_t*, int);

    // 20260323 ZJH 通过 QLibrary::resolve 获取函数地址（跨平台替代 GetProcAddress）
    auto fnCuInit           = reinterpret_cast<CuInitFn>(cudaLib.resolve("cuInit"));
    auto fnCuDeviceGetCount = reinterpret_cast<CuDeviceGetCountFn>(cudaLib.resolve("cuDeviceGetCount"));
    auto fnCuDeviceGetName  = reinterpret_cast<CuDeviceGetNameFn>(cudaLib.resolve("cuDeviceGetName"));
    auto fnCuDeviceTotalMem = reinterpret_cast<CuDeviceTotalMemFn>(cudaLib.resolve("cuDeviceTotalMem_v2"));

    if (!fnCuInit || !fnCuDeviceGetCount || !fnCuDeviceGetName || !fnCuDeviceTotalMem) {
        qDebug() << "[OmniMatch] CUDA Driver API 函数解析失败";
        m_bHasGpu    = false;
        m_strGpuName = QStringLiteral("Driver too old");
        m_nGpuVramMB = 0;
        return;  // 20260323 ZJH QLibrary 析构自动卸载库
    }

    // 20260323 ZJH cuInit(0) 初始化 CUDA 驱动
    if (fnCuInit(0) != 0) {
        qDebug() << "[OmniMatch] cuInit 失败";
        m_bHasGpu    = false;
        m_strGpuName = QStringLiteral("CUDA init failed");
        m_nGpuVramMB = 0;
        return;
    }

    // 20260323 ZJH 获取 CUDA 设备数量
    int nDeviceCount = 0;
    fnCuDeviceGetCount(&nDeviceCount);
    if (nDeviceCount <= 0) {
        qDebug() << "[OmniMatch] 未找到 CUDA 设备";
        m_bHasGpu    = false;
        m_strGpuName = QStringLiteral("No CUDA device");
        m_nGpuVramMB = 0;
        return;
    }

    // 20260323 ZJH 读取第一个 GPU 名称和显存
    char szGpuName[256] = {};
    fnCuDeviceGetName(szGpuName, static_cast<int>(sizeof(szGpuName)), 0);

    size_t nVramBytes = 0;
    fnCuDeviceTotalMem(&nVramBytes, 0);

    m_bHasGpu    = true;
    m_strGpuName = QString::fromLocal8Bit(szGpuName);
    m_nGpuVramMB = nVramBytes / (1024ULL * 1024ULL);

    qDebug() << "[OmniMatch] GPU:" << m_strGpuName << "|" << m_nGpuVramMB << "MB";
    // 20260323 ZJH QLibrary 析构时自动卸载库，无需手动 FreeLibrary
}
