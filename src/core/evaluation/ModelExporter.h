// 20260323 ZJH ModelExporter — 模型导出器
// 将训练好的模型导出为 ONNX/TensorRT/OpenVINO/自研 OMM 格式
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
    NativeOMM       // 20260330 ZJH OmniMatch 自研 .omm 模型格式（v4，含架构元数据）
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

    // 20260402 ZJH [OPT-2.4] 模型量化导出 — PTQ/QAT 支持
    // 对标 NVIDIA Model Optimizer: FP16/INT8/NVFP4 量化

    // 20260402 ZJH 量化模式枚举
    enum class QuantizationMode {
        DynamicPTQ,     // 20260402 ZJH 动态量化（无需校准数据，权重 INT8 + 激活 FP32，速度 1.5-2x）
        StaticPTQ,      // 20260402 ZJH 静态量化（需校准数据，权重+激活 INT8，速度 2-4x）
        QAT             // 20260402 ZJH 量化感知训练（训练时插入伪量化节点，精度最高，速度同 StaticPTQ）
    };

    // 20260402 ZJH 量化配置
    struct QuantizeConfig {
        QuantizationMode eMode = QuantizationMode::StaticPTQ;  // 20260402 ZJH 量化模式
        QString strCalibDataDir;          // 20260402 ZJH 校准数据目录（StaticPTQ 需要）
        int nCalibBatchSize = 8;          // 20260402 ZJH 校准批次大小
        int nCalibNumBatches = 50;        // 20260402 ZJH 校准批次数量
        QString strCalibMethod = "entropy";  // 20260402 ZJH 校准方法: "minmax"/"entropy"/"percentile"
        float fPercentile = 99.99f;       // 20260402 ZJH Percentile 校准百分位
        bool bPerChannel = true;          // 20260402 ZJH 是否逐通道量化（精度更高但稍慢）
    };

    // 20260402 ZJH 量化结果
    struct QuantizeResult {
        bool bSuccess = false;            // 20260402 ZJH 是否成功
        QString strOutputPath;            // 20260402 ZJH 量化后模型路径
        qint64 nOrigSizeBytes = 0;        // 20260402 ZJH 原始模型大小
        qint64 nQuantSizeBytes = 0;       // 20260402 ZJH 量化后模型大小
        float fCompressionRatio = 1.0f;   // 20260402 ZJH 压缩比
        QString strErrorMessage;          // 20260402 ZJH 错误信息
    };

    // 20260402 ZJH 执行模型量化
    // strOnnxPath: 源 ONNX 模型路径
    // config: 量化配置
    // 返回: 量化结果
    QuantizeResult quantizeModel(const QString& strOnnxPath, const QuantizeConfig& config);

    // 20260402 ZJH TensorRT 一键优化（ONNX → TRT engine）
    // strOnnxPath: ONNX 模型路径
    // ePrecision: 精度 (FP32/FP16/INT8)
    // strCalibDir: INT8 校准数据目录（FP16/FP32 可为空）
    // 返回: 导出结果，strOutputPath 为 .trt 文件路径
    ExportResult optimizeTensorRT(const QString& strOnnxPath, ExportPrecision ePrecision,
                                   const QString& strCalibDir = QString());

signals:
    // 20260323 ZJH 导出进度更新
    // 参数: nPercent - 进度百分比 (0-100)
    //       strStage - 当前阶段描述
    void progressUpdated(int nPercent, const QString& strStage);

    // 20260323 ZJH 导出日志输出
    void logMessage(const QString& strMessage);
};
