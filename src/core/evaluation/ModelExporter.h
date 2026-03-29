// 20260323 ZJH ModelExporter — 模型导出器
// 将训练好的模型导出为 ONNX/TensorRT/OpenVINO/自研 DFM 格式
// 提供导出进度回调和格式兼容性检查
#pragma once

#include <QString>      // 20260323 ZJH 字符串
#include <QObject>      // 20260323 ZJH Qt 对象系统
#include <QStringList>  // 20260323 ZJH 字符串列表

// 20260323 ZJH 导出格式枚举
enum class ExportFormat
{
    ONNX = 0,       // 20260323 ZJH Open Neural Network Exchange
    TensorRT,       // 20260323 ZJH NVIDIA TensorRT 引擎
    OpenVINO,       // 20260323 ZJH Intel OpenVINO IR
    NativeDFM       // 20260323 ZJH OmniMatch 自研 .dfm 模型格式
};

// 20260323 ZJH 导出精度枚举
enum class ExportPrecision
{
    FP32 = 0,   // 20260323 ZJH 32位浮点
    FP16,       // 20260323 ZJH 16位浮点（半精度）
    INT8        // 20260323 ZJH 8位整数量化
};

// 20260323 ZJH 导出配置
struct ExportConfig
{
    ExportFormat format = ExportFormat::ONNX;          // 20260323 ZJH 导出格式
    ExportPrecision precision = ExportPrecision::FP32;  // 20260323 ZJH 精度
    QString strModelName;          // 20260323 ZJH 模型名称
    QString strOutputDir;          // 20260323 ZJH 输出目录
    bool bDynamicBatch = false;    // 20260323 ZJH 是否支持动态批量
    int nMinBatch = 1;             // 20260323 ZJH 最小批量大小
    int nMaxBatch = 16;            // 20260323 ZJH 最大批量大小
};

// 20260323 ZJH 导出结果
struct ExportResult
{
    bool bSuccess = false;         // 20260323 ZJH 是否成功
    QString strOutputPath;         // 20260323 ZJH 输出文件完整路径
    qint64 nFileSizeBytes = 0;     // 20260323 ZJH 文件大小（字节）
    double dExportTimeSeconds = 0; // 20260323 ZJH 导出耗时（秒）
    QString strErrorMessage;       // 20260323 ZJH 错误信息（失败时填充）
    QString strMessage;            // 20260324 ZJH 信息性消息（如模拟导出说明，成功时可选填充）
};

// 20260323 ZJH 模型导出器
// 管理模型导出流程，提供进度回调
class ModelExporter : public QObject
{
    Q_OBJECT

public:
    // 20260323 ZJH 构造函数
    explicit ModelExporter(QObject* pParent = nullptr);

    // 20260323 ZJH 析构函数
    ~ModelExporter() override = default;

    // 20260323 ZJH 执行模型导出
    // 参数: config - 导出配置
    //       strModelPath - 源模型文件路径
    // 返回: 导出结果
    ExportResult exportModel(const ExportConfig& config, const QString& strModelPath);

    // 20260323 ZJH 获取指定格式的文件扩展名
    static QString formatExtension(ExportFormat format);

    // 20260323 ZJH 获取格式显示名称
    static QString formatDisplayName(ExportFormat format);

    // 20260323 ZJH 获取支持的精度列表
    static QStringList supportedPrecisions(ExportFormat format);

    // 20260323 ZJH 检查格式是否可用（依赖检查）
    static bool isFormatAvailable(ExportFormat format);

signals:
    // 20260323 ZJH 导出进度更新
    // 参数: nPercent - 进度百分比 (0-100)
    //       strStage - 当前阶段描述
    void progressUpdated(int nPercent, const QString& strStage);

    // 20260323 ZJH 导出日志输出
    void logMessage(const QString& strMessage);
};
