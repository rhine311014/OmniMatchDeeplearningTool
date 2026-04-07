// 20260322 ZJH ProjectSerializer 实现
// Project ←→ JSON 文件（.dfproj）双向序列化
// 使用 QJsonDocument / QJsonObject / QJsonArray 进行 JSON 读写

#include "core/project/ProjectSerializer.h"  // 20260322 ZJH ProjectSerializer 类声明
#include "core/project/Project.h"            // 20260322 ZJH Project 完整类定义
#include "core/DLTypes.h"                    // 20260322 ZJH 枚举类型定义
#include "core/data/ImageDataset.h"          // 20260322 ZJH 数据集类
#include "core/data/ImageEntry.h"            // 20260322 ZJH 图像条目
#include "core/data/LabelInfo.h"             // 20260322 ZJH 标签信息
#include "core/data/Annotation.h"            // 20260322 ZJH 标注数据
#include "core/training/TrainingConfig.h"       // 20260324 ZJH 训练配置结构体（序列化用）
#include "core/evaluation/EvaluationResult.h"  // 20260324 ZJH 评估结果结构体（序列化用）

#include <QJsonDocument>   // 20260322 ZJH JSON 文档解析/生成
#include <QJsonObject>     // 20260322 ZJH JSON 对象
#include <QJsonArray>      // 20260322 ZJH JSON 数组
#include <QFile>           // 20260322 ZJH 文件读写
#include <QFileInfo>       // 20260324 ZJH 文件路径信息（用于路径重定位 absoluteFilePath）
#include <QDebug>          // 20260322 ZJH 调试日志

// 20260322 ZJH 项目文件格式版本号
static const QString s_strFormatVersion = QStringLiteral("2.0.0");

// 20260322 ZJH 将单个标注序列化为 JSON 对象
// 参数: annotation - 标注数据引用
// 返回: 包含标注信息的 QJsonObject
static QJsonObject annotationToJson(const Annotation& annotation)
{
    QJsonObject obj;  // 20260322 ZJH 标注 JSON 对象
    obj["uuid"]    = annotation.strUuid;                            // 20260322 ZJH 唯一标识
    obj["type"]    = static_cast<int>(annotation.eType);            // 20260322 ZJH 标注类型（枚举转 int）
    obj["labelId"] = annotation.nLabelId;                           // 20260322 ZJH 关联标签 ID

    // 20260322 ZJH 序列化边界矩形（x, y, width, height）
    QJsonObject rectObj;  // 20260322 ZJH 矩形 JSON 对象
    rectObj["x"]      = annotation.rectBounds.x();       // 20260322 ZJH 左上角 X
    rectObj["y"]      = annotation.rectBounds.y();       // 20260322 ZJH 左上角 Y
    rectObj["width"]  = annotation.rectBounds.width();   // 20260322 ZJH 宽度
    rectObj["height"] = annotation.rectBounds.height();  // 20260322 ZJH 高度
    obj["bounds"] = rectObj;  // 20260322 ZJH 嵌入边界矩形

    // 20260322 ZJH 序列化多边形顶点列表（仅 Polygon 类型有数据）
    if (!annotation.polygon.isEmpty()) {
        QJsonArray polyArr;  // 20260322 ZJH 多边形点数组
        for (const QPointF& pt : annotation.polygon) {
            QJsonObject ptObj;  // 20260322 ZJH 单个点对象
            ptObj["x"] = pt.x();  // 20260322 ZJH 点的 X 坐标
            ptObj["y"] = pt.y();  // 20260322 ZJH 点的 Y 坐标
            polyArr.append(ptObj);  // 20260322 ZJH 追加到数组
        }
        obj["polygon"] = polyArr;  // 20260322 ZJH 嵌入多边形数据
    }

    // 20260322 ZJH 序列化文字内容（仅 TextArea 类型有数据）
    if (!annotation.strText.isEmpty()) {
        obj["text"] = annotation.strText;  // 20260322 ZJH OCR 文本
    }

    // 20260330 ZJH 序列化缺陷严重度（对标海康 VisionTrain 缺陷分级）
    if (annotation.eSeverity != DefectSeverity::None) {
        obj["severity"] = static_cast<int>(annotation.eSeverity);
    }

    // 20260402 ZJH 序列化旋转角度（仅 RotatedRect 类型或角度非零时保存）
    if (std::abs(annotation.fAngle) > 0.01f) {
        obj["angle"] = static_cast<double>(annotation.fAngle);
    }

    return obj;  // 20260322 ZJH 返回标注 JSON 对象
}

// 20260322 ZJH 从 JSON 对象反序列化标注数据
// 参数: obj - 标注 JSON 对象
// 返回: 恢复的 Annotation 结构体
static Annotation annotationFromJson(const QJsonObject& obj)
{
    // 20260322 ZJH 根据类型创建标注
    // 20260324 ZJH 验证标注类型枚举值范围，防止非法 JSON 数据导致未定义行为
    int nTypeVal = obj["type"].toInt();  // 20260324 ZJH 读取标注类型整数值
    if (nTypeVal < 0 || nTypeVal > 4) {  // 20260402 ZJH 扩展到 4 (RotatedRect)
        // 20260324 ZJH 无效标注类型，使用 Rect 作为安全默认值
        qDebug() << "[ProjectSerializer] annotationFromJson: 无效标注类型" << nTypeVal << "，使用默认 Rect";
        nTypeVal = 0;
    }
    auto eType = static_cast<AnnotationType>(nTypeVal);  // 20260324 ZJH 安全转换标注类型枚举
    Annotation annotation(eType);  // 20260322 ZJH 创建标注（自动生成 UUID）

    // 20260322 ZJH 恢复保存的 UUID（覆盖自动生成的）
    annotation.strUuid = obj["uuid"].toString();   // 20260322 ZJH 唯一标识
    annotation.nLabelId = obj["labelId"].toInt(-1); // 20260322 ZJH 标签 ID

    // 20260322 ZJH 恢复边界矩形
    if (obj.contains("bounds")) {
        QJsonObject rectObj = obj["bounds"].toObject();  // 20260322 ZJH 读取矩形对象
        annotation.rectBounds = QRectF(
            rectObj["x"].toDouble(),       // 20260322 ZJH X 坐标
            rectObj["y"].toDouble(),       // 20260322 ZJH Y 坐标
            rectObj["width"].toDouble(),   // 20260322 ZJH 宽度
            rectObj["height"].toDouble()   // 20260322 ZJH 高度
        );
    }

    // 20260322 ZJH 恢复多边形顶点
    if (obj.contains("polygon")) {
        QJsonArray polyArr = obj["polygon"].toArray();  // 20260322 ZJH 读取多边形数组
        for (const QJsonValue& val : polyArr) {
            QJsonObject ptObj = val.toObject();  // 20260322 ZJH 读取点对象
            annotation.polygon.append(QPointF(
                ptObj["x"].toDouble(),  // 20260322 ZJH X 坐标
                ptObj["y"].toDouble()   // 20260322 ZJH Y 坐标
            ));
        }
    }

    // 20260322 ZJH 恢复文字内容
    if (obj.contains("text")) {
        annotation.strText = obj["text"].toString();  // 20260322 ZJH OCR 文本
    }

    // 20260330 ZJH 恢复缺陷严重度（向后兼容：旧文件无此字段默认 None）
    if (obj.contains("severity")) {
        annotation.eSeverity = static_cast<DefectSeverity>(obj["severity"].toInt(0));
    }

    // 20260402 ZJH 恢复旋转角度（向后兼容：旧文件无此字段默认 0.0 = 轴对齐）
    if (obj.contains("angle")) {
        annotation.fAngle = static_cast<float>(obj["angle"].toDouble(0.0));
    }

    return annotation;  // 20260322 ZJH 返回恢复的标注
}

// ===== 公共接口实现 =====

// 20260322 ZJH 将项目序列化保存为 JSON 文件
bool ProjectSerializer::save(const Project* pProject, const QString& strFilePath)
{
    // 20260322 ZJH 空指针检查
    if (!pProject) {
        qDebug() << "[ProjectSerializer] save: 项目指针为空";  // 20260322 ZJH 空指针日志
        return false;  // 20260322 ZJH 无法保存空项目
    }

    // 20260322 ZJH 构建顶层 JSON 对象
    QJsonObject root;  // 20260322 ZJH 根 JSON 对象

    // 20260322 ZJH 写入文件格式版本
    root["version"]  = s_strFormatVersion;  // 20260322 ZJH 格式版本 "2.0.0"

    // 20260322 ZJH 写入项目元数据
    root["name"]     = pProject->name();                                // 20260322 ZJH 项目名称
    root["taskType"] = static_cast<int>(pProject->taskType());          // 20260322 ZJH 任务类型（枚举转 int）
    root["state"]    = static_cast<int>(pProject->state());             // 20260322 ZJH 项目状态（枚举转 int）
    root["path"]     = pProject->path();                                // 20260322 ZJH 项目路径

    // 20260403 ZJH 序列化项目描述（仅非空时写入，向后兼容旧格式）
    if (!pProject->description().isEmpty()) {
        root["description"] = pProject->description();                 // 20260403 ZJH 项目描述
    }

    // 20260322 ZJH 序列化标签列表
    const ImageDataset* pDataset = pProject->dataset();  // 20260322 ZJH 获取数据集
    QJsonArray labelsArr;  // 20260322 ZJH 标签 JSON 数组
    if (pDataset) {
        for (const LabelInfo& label : pDataset->labels()) {
            QJsonObject labelObj;  // 20260322 ZJH 单个标签对象
            labelObj["id"]      = label.nId;                 // 20260322 ZJH 标签 ID
            labelObj["name"]    = label.strName;             // 20260322 ZJH 标签名称
            labelObj["color"]   = label.color.name();        // 20260322 ZJH 颜色转 "#rrggbb" 字符串
            labelObj["visible"] = label.bVisible;            // 20260322 ZJH 可见性
            labelsArr.append(labelObj);  // 20260322 ZJH 追加到数组
        }
    }
    root["labels"] = labelsArr;  // 20260322 ZJH 嵌入标签数组

    // 20260322 ZJH 序列化图像条目列表
    QJsonArray imagesArr;  // 20260322 ZJH 图像 JSON 数组
    if (pDataset) {
        for (const ImageEntry& entry : pDataset->images()) {
            QJsonObject imgObj;  // 20260322 ZJH 单个图像对象
            imgObj["uuid"]         = entry.strUuid;                          // 20260322 ZJH UUID
            imgObj["filePath"]     = entry.strFilePath;                      // 20260322 ZJH 绝对路径
            imgObj["relativePath"] = entry.strRelativePath;                  // 20260322 ZJH 相对路径
            imgObj["width"]        = entry.nWidth;                           // 20260322 ZJH 宽度
            imgObj["height"]       = entry.nHeight;                          // 20260322 ZJH 高度
            imgObj["channels"]     = entry.nChannels;                        // 20260322 ZJH 通道数
            imgObj["fileSize"]     = static_cast<qint64>(entry.nFileSize);   // 20260322 ZJH 文件大小
            imgObj["labelId"]      = entry.nLabelId;                         // 20260322 ZJH 图像级标签
            imgObj["split"]        = static_cast<int>(entry.eSplit);         // 20260322 ZJH 拆分类型

            // 20260322 ZJH 序列化标注列表
            if (!entry.vecAnnotations.isEmpty()) {
                QJsonArray annosArr;  // 20260322 ZJH 标注数组
                for (const Annotation& anno : entry.vecAnnotations) {
                    annosArr.append(annotationToJson(anno));  // 20260322 ZJH 逐个序列化标注
                }
                imgObj["annotations"] = annosArr;  // 20260322 ZJH 嵌入标注数组
            }

            imagesArr.append(imgObj);  // 20260322 ZJH 追加到图像数组
        }
    }
    root["images"] = imagesArr;  // 20260322 ZJH 嵌入图像数组

    // 20260322 ZJH 序列化数据集拆分统计
    QJsonObject splitsObj;  // 20260322 ZJH 拆分统计对象
    if (pDataset) {
        splitsObj["train"]      = pDataset->countBySplit(om::SplitType::Train);       // 20260322 ZJH 训练集数量
        splitsObj["validation"] = pDataset->countBySplit(om::SplitType::Validation);  // 20260322 ZJH 验证集数量
        splitsObj["test"]       = pDataset->countBySplit(om::SplitType::Test);        // 20260322 ZJH 测试集数量
        splitsObj["unassigned"] = pDataset->countBySplit(om::SplitType::Unassigned);  // 20260322 ZJH 未分配数量
    }
    root["splits"] = splitsObj;  // 20260322 ZJH 嵌入拆分统计

    // 20260324 ZJH 序列化训练配置（TrainingConfig 全字段）
    {
        const TrainingConfig& cfg = pProject->trainingConfig();  // 20260324 ZJH 获取训练配置引用
        QJsonObject cfgObj;  // 20260324 ZJH 训练配置 JSON 对象

        // 20260324 ZJH 框架与模型
        cfgObj["framework"]    = static_cast<int>(cfg.eFramework);     // 20260324 ZJH 训练框架枚举
        cfgObj["modelCapability"] = static_cast<int>(cfg.eModelCapability);  // 20260330 ZJH 模型能力等级
        cfgObj["architecture"] = static_cast<int>(cfg.eArchitecture);  // 20260324 ZJH 模型架构枚举
        cfgObj["device"]       = static_cast<int>(cfg.eDevice);        // 20260324 ZJH 设备类型枚举
        cfgObj["anomalyMode"]  = static_cast<int>(cfg.eAnomalyMode);   // 20260330 ZJH 异常检测训练模式
        cfgObj["optimizer"]    = static_cast<int>(cfg.eOptimizer);     // 20260324 ZJH 优化器枚举
        cfgObj["scheduler"]    = static_cast<int>(cfg.eScheduler);     // 20260324 ZJH 调度器枚举

        // 20260324 ZJH 超参数
        cfgObj["learningRate"] = cfg.dLearningRate;  // 20260324 ZJH 学习率
        cfgObj["batchSize"]    = cfg.nBatchSize;     // 20260324 ZJH 批量大小
        cfgObj["epochs"]       = cfg.nEpochs;        // 20260324 ZJH 训练轮次
        cfgObj["resolutionPreset"] = static_cast<int>(cfg.eResolutionPreset);  // 20260330 ZJH 分辨率预设
        cfgObj["inputSize"]    = cfg.nInputSize;      // 20260324 ZJH 输入尺寸
        cfgObj["patience"]     = cfg.nPatience;       // 20260324 ZJH 早停耐心值

        // 20260331 ZJH 迁移学习参数（骨干冻结 + 分层学习率）
        cfgObj["freezeEpochs"]          = cfg.nFreezeEpochs;           // 20260331 ZJH 骨干冻结轮数
        cfgObj["backboneLrMultiplier"]  = cfg.dBackboneLrMultiplier;   // 20260331 ZJH 骨干 LR 倍率
        cfgObj["cropSize"]              = cfg.nCropSize;               // 20260401 ZJH Crop 尺寸

        // 20260330 ZJH 预训练模型与标识
        cfgObj["pretrainedModelPath"] = cfg.strPretrainedModelPath;  // 20260330 ZJH 预训练模型路径
        cfgObj["modelTag"]            = cfg.strModelTag;              // 20260330 ZJH 模型标识

        // 20260330 ZJH 数据增强预设
        cfgObj["augPreset"]    = static_cast<int>(cfg.eAugPreset);   // 20260330 ZJH 增强预设模式

        // 20260324 ZJH 数据增强（基础）
        cfgObj["augmentation"]   = cfg.bAugmentation;    // 20260324 ZJH 启用增强标志
        cfgObj["augBrightness"]  = cfg.dAugBrightness;   // 20260324 ZJH 亮度增强幅度
        cfgObj["augContrast"]    = cfg.dAugContrast;     // 20260324 ZJH 对比度增强幅度
        cfgObj["augFlipProb"]    = cfg.dAugFlipProb;     // 20260324 ZJH 翻转概率
        cfgObj["augRotation"]    = cfg.dAugRotation;     // 20260324 ZJH 旋转角度

        // 20260324 ZJH 数据增强（扩展 — 几何变换）
        cfgObj["augVerticalFlipProb"] = cfg.dAugVerticalFlipProb;  // 20260324 ZJH 垂直翻转概率
        cfgObj["augAffine"]           = cfg.bAugAffine;            // 20260324 ZJH 仿射变换开关
        cfgObj["augShearDeg"]         = cfg.dAugShearDeg;          // 20260324 ZJH 剪切角度
        cfgObj["augTranslate"]        = cfg.dAugTranslate;         // 20260324 ZJH 平移比例
        cfgObj["augRandomCrop"]       = cfg.bAugRandomCrop;        // 20260324 ZJH 随机裁剪开关
        cfgObj["augCropScale"]        = cfg.dAugCropScale;         // 20260324 ZJH 裁剪缩放比例

        // 20260324 ZJH 数据增强（扩展 — 颜色变换）
        cfgObj["augColorJitter"] = cfg.bAugColorJitter;  // 20260324 ZJH 颜色抖动开关
        cfgObj["augSaturation"]  = cfg.dAugSaturation;   // 20260324 ZJH 饱和度抖动
        cfgObj["augHue"]         = cfg.dAugHue;           // 20260324 ZJH 色调抖动

        // 20260324 ZJH 数据增强（扩展 — 噪声/遮挡）
        cfgObj["augGaussianNoise"]    = cfg.bAugGaussianNoise;      // 20260324 ZJH 高斯噪声开关
        cfgObj["augGaussianNoiseStd"] = cfg.dAugGaussianNoiseStd;   // 20260324 ZJH 噪声标准差
        cfgObj["augGaussianBlur"]     = cfg.bAugGaussianBlur;       // 20260324 ZJH 高斯模糊开关
        cfgObj["augBlurSigma"]        = cfg.dAugBlurSigma;          // 20260324 ZJH 模糊 sigma
        cfgObj["augRandomErasing"]    = cfg.bAugRandomErasing;      // 20260324 ZJH 随机擦除开关
        cfgObj["augErasingProb"]      = cfg.dAugErasingProb;        // 20260324 ZJH 擦除概率
        cfgObj["augErasingRatio"]     = cfg.dAugErasingRatio;       // 20260324 ZJH 擦除面积比例

        // 20260324 ZJH 数据增强（扩展 — 高级混合）
        cfgObj["augMixup"]       = cfg.bAugMixup;        // 20260324 ZJH Mixup 开关
        cfgObj["augMixupAlpha"]  = cfg.dAugMixupAlpha;   // 20260324 ZJH Mixup alpha
        cfgObj["augCutMix"]      = cfg.bAugCutMix;       // 20260324 ZJH CutMix 开关
        cfgObj["augCutMixAlpha"] = cfg.dAugCutMixAlpha;  // 20260324 ZJH CutMix alpha

        // 20260324 ZJH 导出选项
        cfgObj["exportOnnx"] = cfg.bExportOnnx;  // 20260324 ZJH 自动导出 ONNX 标志

        root["trainingConfig"] = cfgObj;  // 20260324 ZJH 嵌入训练配置对象
    }

    // 20260324 ZJH 序列化训练历史记录（每个 Epoch 一条）
    {
        QJsonArray historyArr;  // 20260324 ZJH 训练历史 JSON 数组
        for (const EpochRecord& record : pProject->trainingHistory()) {
            QJsonObject recObj;  // 20260324 ZJH 单条记录对象
            recObj["epoch"]       = record.nEpoch;        // 20260324 ZJH 轮次编号
            recObj["trainLoss"]   = record.dTrainLoss;    // 20260324 ZJH 训练损失
            recObj["valLoss"]     = record.dValLoss;      // 20260324 ZJH 验证损失
            recObj["valAccuracy"] = record.dValAccuracy;  // 20260324 ZJH 验证准确率
            historyArr.append(recObj);  // 20260324 ZJH 追加到数组
        }
        root["trainingHistory"] = historyArr;  // 20260324 ZJH 嵌入训练历史数组
    }

    // 20260324 ZJH 序列化最佳模型检查点路径
    root["bestModelPath"] = pProject->bestModelPath();  // 20260324 ZJH 最佳模型路径

    // 20260324 ZJH 序列化评估结果（仅在已有评估结果时写入）
    if (pProject->hasEvaluationResult()) {
        const EvaluationResult& eval = pProject->evaluationResult();  // 20260324 ZJH 获取评估结果引用
        QJsonObject evalObj;  // 20260324 ZJH 评估结果 JSON 对象

        // 20260324 ZJH 核心指标
        evalObj["accuracy"]  = eval.dAccuracy;   // 20260324 ZJH 准确率
        evalObj["precision"] = eval.dPrecision;  // 20260324 ZJH 精确率
        evalObj["recall"]    = eval.dRecall;     // 20260324 ZJH 召回率
        evalObj["f1Score"]   = eval.dF1Score;    // 20260324 ZJH F1 分数
        evalObj["meanAP"]    = eval.dMeanAP;     // 20260324 ZJH mAP
        evalObj["mIoU"]      = eval.dMIoU;       // 20260324 ZJH mIoU
        evalObj["auc"]       = eval.dAUC;        // 20260324 ZJH AUC

        // 20260324 ZJH 性能统计
        evalObj["avgLatencyMs"]  = eval.dAvgLatencyMs;   // 20260324 ZJH 平均推理延迟（ms）
        evalObj["throughputFPS"] = eval.dThroughputFPS;  // 20260324 ZJH 推理吞吐量（FPS）

        // 20260324 ZJH 统计摘要
        evalObj["totalImages"]      = eval.nTotalImages;       // 20260324 ZJH 评估图像总数
        evalObj["correct"]          = eval.nCorrect;           // 20260324 ZJH 正确预测数量
        evalObj["totalTimeSeconds"] = eval.dTotalTimeSeconds;  // 20260324 ZJH 评估总耗时（秒）

        // 20260324 ZJH 类别名称列表
        QJsonArray classNamesArr;  // 20260324 ZJH 类别名称 JSON 数组
        for (const QString& strClassName : eval.vecClassNames) {
            classNamesArr.append(strClassName);  // 20260324 ZJH 追加类别名称
        }
        evalObj["classNames"] = classNamesArr;  // 20260324 ZJH 嵌入类别名称数组

        // 20260324 ZJH 混淆矩阵（二维 JSON 数组，行=真实类别，列=预测类别）
        QJsonArray confusionArr;  // 20260324 ZJH 混淆矩阵 JSON 数组
        for (const QVector<int>& vecRow : eval.matConfusion) {
            QJsonArray rowArr;  // 20260324 ZJH 混淆矩阵一行
            for (int nVal : vecRow) {
                rowArr.append(nVal);  // 20260324 ZJH 追加单元格值
            }
            confusionArr.append(rowArr);  // 20260324 ZJH 追加一行到矩阵
        }
        evalObj["confusion"] = confusionArr;  // 20260324 ZJH 嵌入混淆矩阵

        // 20260324 ZJH 每类精确率
        QJsonArray precPerClassArr;  // 20260324 ZJH 每类精确率 JSON 数组
        for (double dVal : eval.vecPrecisionPerClass) {
            precPerClassArr.append(dVal);  // 20260324 ZJH 追加精确率值
        }
        evalObj["precisionPerClass"] = precPerClassArr;  // 20260324 ZJH 嵌入每类精确率

        // 20260324 ZJH 每类召回率
        QJsonArray recPerClassArr;  // 20260324 ZJH 每类召回率 JSON 数组
        for (double dVal : eval.vecRecallPerClass) {
            recPerClassArr.append(dVal);  // 20260324 ZJH 追加召回率值
        }
        evalObj["recallPerClass"] = recPerClassArr;  // 20260324 ZJH 嵌入每类召回率

        // 20260324 ZJH 每类 F1 分数
        QJsonArray f1PerClassArr;  // 20260324 ZJH 每类 F1 JSON 数组
        for (double dVal : eval.vecF1PerClass) {
            f1PerClassArr.append(dVal);  // 20260324 ZJH 追加 F1 值
        }
        evalObj["f1PerClass"] = f1PerClassArr;  // 20260324 ZJH 嵌入每类 F1

        root["evaluation"] = evalObj;  // 20260324 ZJH 嵌入评估结果对象到根
    }

    // 20260322 ZJH 生成 JSON 文档
    QJsonDocument doc(root);  // 20260322 ZJH 将根对象包装为 QJsonDocument

    // 20260322 ZJH 写入文件
    QFile file(strFilePath);  // 20260322 ZJH 目标文件
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "[ProjectSerializer] save: 无法打开文件" << strFilePath;  // 20260322 ZJH 打开失败日志
        return false;  // 20260322 ZJH 文件打开失败
    }

    // 20260322 ZJH 写入格式化 JSON（Indented 便于人类阅读）
    // 20260324 ZJH 检查写入返回值，确保数据完整写入磁盘
    QByteArray jsonData = doc.toJson(QJsonDocument::Indented);  // 20260324 ZJH 生成格式化 JSON 字节数据
    qint64 nBytesWritten = file.write(jsonData);  // 20260324 ZJH 写入并获取实际写入字节数
    file.close();  // 20260322 ZJH 关闭文件
    if (nBytesWritten != jsonData.size()) {
        qDebug() << "[ProjectSerializer] save: 写入不完整，期望" << jsonData.size() << "实际" << nBytesWritten;  // 20260324 ZJH 写入失败日志
        return false;  // 20260324 ZJH 写入失败
    }

    qDebug() << "[ProjectSerializer] save: 成功" << strFilePath;  // 20260322 ZJH 保存成功日志
    return true;  // 20260322 ZJH 返回成功
}

// 20260322 ZJH 从 JSON 文件反序列化恢复项目
std::unique_ptr<Project> ProjectSerializer::load(const QString& strFilePath)
{
    // 20260322 ZJH 打开文件
    QFile file(strFilePath);  // 20260322 ZJH 源文件
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "[ProjectSerializer] load: 无法打开文件" << strFilePath;  // 20260322 ZJH 打开失败日志
        return nullptr;  // 20260322 ZJH 文件打开失败
    }

    // 20260324 ZJH 检查文件大小，防止超大文件导致内存耗尽（上限 100MB）
    constexpr qint64 s_nMaxProjectFileSize = 100 * 1024 * 1024;  // 20260324 ZJH 100MB 上限
    if (file.size() > s_nMaxProjectFileSize) {
        qDebug() << "[ProjectSerializer] load: 文件过大" << file.size() << "字节，上限" << s_nMaxProjectFileSize;  // 20260324 ZJH 文件过大日志
        file.close();  // 20260324 ZJH 关闭文件
        return nullptr;  // 20260324 ZJH 拒绝加载超大文件
    }

    // 20260322 ZJH 读取全部内容
    QByteArray data = file.readAll();  // 20260322 ZJH 读取文件全部字节
    file.close();  // 20260322 ZJH 关闭文件

    // 20260322 ZJH 解析 JSON
    QJsonParseError parseError;  // 20260322 ZJH 解析错误信息
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);  // 20260322 ZJH 解析 JSON
    if (parseError.error != QJsonParseError::NoError) {
        qDebug() << "[ProjectSerializer] load: JSON 解析错误" << parseError.errorString();  // 20260322 ZJH 解析错误日志
        return nullptr;  // 20260322 ZJH 解析失败
    }

    // 20260322 ZJH 获取根对象
    QJsonObject root = doc.object();  // 20260322 ZJH JSON 根对象

    // 20260325 ZJH 检查文件格式版本号（宽松模式：缺失或不匹配仅警告，不拒绝加载）
    // 旧版项目文件可能没有 version 字段，或版本号不同——仍尝试加载
    QString strFileVersion = root["version"].toString();
    if (!strFileVersion.isEmpty() && strFileVersion != s_strFormatVersion) {
        qDebug() << "[ProjectSerializer] load: 版本不同，文件:" << strFileVersion
                 << "当前:" << s_strFormatVersion << "（尝试兼容加载）";
    }

    // 20260322 ZJH 创建 Project 实例
    auto pProject = std::make_unique<Project>();

    // 20260322 ZJH 恢复项目元数据
    pProject->setName(root["name"].toString());  // 20260322 ZJH 项目名称

    // 20260403 ZJH 恢复项目描述（可选字段，旧项目文件可能不存在此字段）
    if (root.contains("description")) {
        pProject->setDescription(root["description"].toString());  // 20260403 ZJH 项目描述
    }

    // 20260324 ZJH 验证 TaskType 枚举值范围（0-7），防止非法 JSON 数据导致未定义行为
    int nTaskTypeVal = root["taskType"].toInt();  // 20260324 ZJH 读取任务类型整数值
    if (nTaskTypeVal < 0 || nTaskTypeVal > 9) {
        qDebug() << "[ProjectSerializer] load: 无效任务类型" << nTaskTypeVal;  // 20260324 ZJH 非法枚举日志
        return nullptr;  // 20260324 ZJH 任务类型非法，拒绝加载
    }
    pProject->setTaskType(static_cast<om::TaskType>(nTaskTypeVal));  // 20260324 ZJH 安全转换任务类型

    // 20260324 ZJH 验证 ProjectState 枚举值范围（0-5），防止非法 JSON 数据导致未定义行为
    int nStateVal = root["state"].toInt();  // 20260324 ZJH 读取项目状态整数值
    if (nStateVal < 0 || nStateVal > 5) {
        qDebug() << "[ProjectSerializer] load: 无效项目状态" << nStateVal;  // 20260324 ZJH 非法枚举日志
        return nullptr;  // 20260324 ZJH 项目状态非法，拒绝加载
    }
    pProject->setState(static_cast<om::ProjectState>(nStateVal));  // 20260324 ZJH 安全转换项目状态

    // 20260325 ZJH 项目路径：始终使用 .dfproj 文件的实际所在目录
    // JSON 中存储的 "path" 可能是旧的绝对路径（项目被移动/复制后过时）
    // 用实际文件位置确保图像相对路径重定位基于正确的根目录
    {
        QString strActualDir = QFileInfo(strFilePath).absolutePath();
        pProject->setPath(strActualDir);
    }

    // 20260323 ZJH 恢复标签列表（直接插入保留原始 ID，不走 addLabel 自增逻辑）
    ImageDataset* pDataset = pProject->dataset();  // 20260322 ZJH 获取数据集
    QJsonArray labelsArr = root["labels"].toArray();  // 20260322 ZJH 标签数组
    int nMaxLabelId = -1;  // 20260323 ZJH 追踪最大标签 ID，用于恢复自增计数器
    for (const QJsonValue& val : labelsArr) {
        QJsonObject labelObj = val.toObject();  // 20260322 ZJH 单个标签对象
        LabelInfo label;  // 20260322 ZJH 标签信息
        label.nId      = labelObj["id"].toInt();                   // 20260322 ZJH 标签 ID
        label.strName  = labelObj["name"].toString();              // 20260322 ZJH 标签名称
        label.color    = QColor(labelObj["color"].toString());     // 20260322 ZJH 颜色（从 "#rrggbb" 恢复）
        label.bVisible = labelObj["visible"].toBool(true);         // 20260322 ZJH 可见性（默认 true）
        pDataset->insertLabelDirect(label);  // 20260323 ZJH 直接插入保留原始 ID
        if (label.nId > nMaxLabelId) nMaxLabelId = label.nId;  // 20260323 ZJH 更新最大 ID
    }
    // 20260323 ZJH 恢复自增计数器，确保后续新标签 ID 不与已有冲突
    pDataset->setNextLabelId(nMaxLabelId + 1);

    // 20260322 ZJH 恢复图像条目列表
    QJsonArray imagesArr = root["images"].toArray();  // 20260322 ZJH 图像数组
    for (const QJsonValue& val : imagesArr) {
        QJsonObject imgObj = val.toObject();  // 20260322 ZJH 单个图像对象
        ImageEntry entry;  // 20260322 ZJH 图像条目（构造时自动生成 UUID）

        // 20260322 ZJH 恢复保存的 UUID（覆盖自动生成的）
        entry.strUuid         = imgObj["uuid"].toString();                        // 20260322 ZJH UUID
        entry.strFilePath     = imgObj["filePath"].toString();                    // 20260322 ZJH 绝对路径
        entry.strRelativePath = imgObj["relativePath"].toString();                // 20260322 ZJH 相对路径

        // 20260324 ZJH 检查相对路径是否包含 ".." 目录遍历，防止恶意项目文件读取项目目录外的文件
        if (entry.strRelativePath.contains("..")) {
            qDebug() << "[ProjectSerializer] load: 跳过包含路径遍历的条目" << entry.strRelativePath;  // 20260324 ZJH 路径遍历日志
            continue;  // 20260324 ZJH 跳过可疑路径条目
        }

        // 20260324 ZJH 路径重定位：优先使用项目目录 + 相对路径（支持项目跨机器/目录迁移）
        // 如果相对路径非空且项目路径已知，尝试从项目目录重建绝对路径
        if (!entry.strRelativePath.isEmpty() && !pProject->path().isEmpty()) {
            QString strRelocatedPath = pProject->path() + QStringLiteral("/") + entry.strRelativePath;  // 20260324 ZJH 重定位路径
            if (QFile::exists(strRelocatedPath)) {
                // 20260324 ZJH 重定位路径存在，更新绝对路径（项目已迁移到新位置）
                entry.strFilePath = QFileInfo(strRelocatedPath).absoluteFilePath();
            }
            // 20260324 ZJH 重定位路径不存在时保留原始 strFilePath（回退到旧绝对路径）
        }
        entry.nWidth          = imgObj["width"].toInt();                          // 20260322 ZJH 宽度
        entry.nHeight         = imgObj["height"].toInt();                         // 20260322 ZJH 高度
        entry.nChannels       = imgObj["channels"].toInt();                       // 20260322 ZJH 通道数
        entry.nFileSize       = imgObj["fileSize"].toInteger();                   // 20260322 ZJH 文件大小
        entry.nLabelId        = imgObj["labelId"].toInt(-1);                      // 20260322 ZJH 图像级标签
        // 20260324 ZJH 验证 SplitType 枚举值范围（0-3），非法值使用 Unassigned 作为安全默认
        int nSplitVal = imgObj["split"].toInt();  // 20260324 ZJH 读取拆分类型整数值
        if (nSplitVal < 0 || nSplitVal > 3) {
            qDebug() << "[ProjectSerializer] load: 无效拆分类型" << nSplitVal << "，使用默认 Unassigned";  // 20260324 ZJH 非法拆分类型日志
            nSplitVal = 3;  // 20260324 ZJH 默认为 Unassigned (3)
        }
        entry.eSplit          = static_cast<om::SplitType>(nSplitVal);  // 20260324 ZJH 安全转换拆分类型

        // 20260322 ZJH 恢复标注列表
        if (imgObj.contains("annotations")) {
            QJsonArray annosArr = imgObj["annotations"].toArray();  // 20260322 ZJH 标注数组
            for (const QJsonValue& annoVal : annosArr) {
                entry.vecAnnotations.append(annotationFromJson(annoVal.toObject()));  // 20260322 ZJH 逐个反序列化
            }
        }

        pDataset->addImage(entry);  // 20260322 ZJH 添加图像到数据集
    }

    // 20260324 ZJH 反序列化训练配置（兼容旧版本文件：不存在则使用默认值）
    if (root.contains("trainingConfig")) {
        QJsonObject cfgObj = root["trainingConfig"].toObject();  // 20260324 ZJH 训练配置 JSON 对象
        TrainingConfig cfg;  // 20260324 ZJH 使用默认值初始化

        // 20260324 ZJH 框架与模型
        cfg.eFramework    = static_cast<om::TrainingFramework>(cfgObj["framework"].toInt(static_cast<int>(cfg.eFramework)));
        // 20260330 ZJH 模型能力等级（兼容旧版：不存在则使用默认 Normal）
        if (cfgObj.contains("modelCapability"))
            cfg.eModelCapability = static_cast<om::ModelCapability>(cfgObj["modelCapability"].toInt(static_cast<int>(cfg.eModelCapability)));
        cfg.eArchitecture = static_cast<om::ModelArchitecture>(cfgObj["architecture"].toInt(static_cast<int>(cfg.eArchitecture)));
        cfg.eDevice       = static_cast<om::DeviceType>(cfgObj["device"].toInt(static_cast<int>(cfg.eDevice)));
        // 20260330 ZJH 异常检测训练模式（兼容旧版：不存在则使用默认 Fast）
        if (cfgObj.contains("anomalyMode"))
            cfg.eAnomalyMode = static_cast<om::AnomalyTrainingMode>(cfgObj["anomalyMode"].toInt(static_cast<int>(cfg.eAnomalyMode)));
        cfg.eOptimizer    = static_cast<om::OptimizerType>(cfgObj["optimizer"].toInt(static_cast<int>(cfg.eOptimizer)));
        cfg.eScheduler    = static_cast<om::SchedulerType>(cfgObj["scheduler"].toInt(static_cast<int>(cfg.eScheduler)));

        // 20260324 ZJH 超参数
        cfg.dLearningRate = cfgObj["learningRate"].toDouble(cfg.dLearningRate);
        cfg.nBatchSize    = cfgObj["batchSize"].toInt(cfg.nBatchSize);
        cfg.nEpochs       = cfgObj["epochs"].toInt(cfg.nEpochs);
        // 20260330 ZJH 分辨率预设（兼容旧版：不存在则使用默认）
        if (cfgObj.contains("resolutionPreset"))
            cfg.eResolutionPreset = static_cast<om::InputResolutionPreset>(cfgObj["resolutionPreset"].toInt(static_cast<int>(cfg.eResolutionPreset)));
        cfg.nInputSize    = cfgObj["inputSize"].toInt(cfg.nInputSize);
        cfg.nPatience     = cfgObj["patience"].toInt(cfg.nPatience);

        // 20260331 ZJH 迁移学习参数（兼容旧版：不存在则使用默认值）
        cfg.nFreezeEpochs = cfgObj["freezeEpochs"].toInt(cfg.nFreezeEpochs);
        cfg.dBackboneLrMultiplier = cfgObj["backboneLrMultiplier"].toDouble(cfg.dBackboneLrMultiplier);
        cfg.nCropSize = cfgObj["cropSize"].toInt(cfg.nCropSize);  // 20260401 ZJH Crop 尺寸

        // 20260330 ZJH 预训练模型与标识（兼容旧版：不存在则为空）
        if (cfgObj.contains("pretrainedModelPath"))
            cfg.strPretrainedModelPath = cfgObj["pretrainedModelPath"].toString();
        if (cfgObj.contains("modelTag"))
            cfg.strModelTag = cfgObj["modelTag"].toString();

        // 20260330 ZJH 数据增强预设（兼容旧版：不存在则使用默认 Default）
        if (cfgObj.contains("augPreset"))
            cfg.eAugPreset = static_cast<om::AugmentationPreset>(cfgObj["augPreset"].toInt(static_cast<int>(cfg.eAugPreset)));

        // 20260324 ZJH 数据增强（基础）
        cfg.bAugmentation  = cfgObj["augmentation"].toBool(cfg.bAugmentation);
        cfg.dAugBrightness = cfgObj["augBrightness"].toDouble(cfg.dAugBrightness);
        cfg.dAugContrast   = cfgObj["augContrast"].toDouble(cfg.dAugContrast);
        cfg.dAugFlipProb   = cfgObj["augFlipProb"].toDouble(cfg.dAugFlipProb);
        cfg.dAugRotation   = cfgObj["augRotation"].toDouble(cfg.dAugRotation);

        // 20260324 ZJH 数据增强（扩展 — 几何变换，contains 检查兼容旧版文件）
        if (cfgObj.contains("augVerticalFlipProb"))
            cfg.dAugVerticalFlipProb = cfgObj["augVerticalFlipProb"].toDouble(cfg.dAugVerticalFlipProb);
        if (cfgObj.contains("augAffine"))
            cfg.bAugAffine = cfgObj["augAffine"].toBool(cfg.bAugAffine);
        if (cfgObj.contains("augShearDeg"))
            cfg.dAugShearDeg = cfgObj["augShearDeg"].toDouble(cfg.dAugShearDeg);
        if (cfgObj.contains("augTranslate"))
            cfg.dAugTranslate = cfgObj["augTranslate"].toDouble(cfg.dAugTranslate);
        if (cfgObj.contains("augRandomCrop"))
            cfg.bAugRandomCrop = cfgObj["augRandomCrop"].toBool(cfg.bAugRandomCrop);
        if (cfgObj.contains("augCropScale"))
            cfg.dAugCropScale = cfgObj["augCropScale"].toDouble(cfg.dAugCropScale);

        // 20260324 ZJH 数据增强（扩展 — 颜色变换，contains 检查兼容旧版文件）
        if (cfgObj.contains("augColorJitter"))
            cfg.bAugColorJitter = cfgObj["augColorJitter"].toBool(cfg.bAugColorJitter);
        if (cfgObj.contains("augSaturation"))
            cfg.dAugSaturation = cfgObj["augSaturation"].toDouble(cfg.dAugSaturation);
        if (cfgObj.contains("augHue"))
            cfg.dAugHue = cfgObj["augHue"].toDouble(cfg.dAugHue);

        // 20260324 ZJH 数据增强（扩展 — 噪声/遮挡，contains 检查兼容旧版文件）
        if (cfgObj.contains("augGaussianNoise"))
            cfg.bAugGaussianNoise = cfgObj["augGaussianNoise"].toBool(cfg.bAugGaussianNoise);
        if (cfgObj.contains("augGaussianNoiseStd"))
            cfg.dAugGaussianNoiseStd = cfgObj["augGaussianNoiseStd"].toDouble(cfg.dAugGaussianNoiseStd);
        if (cfgObj.contains("augGaussianBlur"))
            cfg.bAugGaussianBlur = cfgObj["augGaussianBlur"].toBool(cfg.bAugGaussianBlur);
        if (cfgObj.contains("augBlurSigma"))
            cfg.dAugBlurSigma = cfgObj["augBlurSigma"].toDouble(cfg.dAugBlurSigma);
        if (cfgObj.contains("augRandomErasing"))
            cfg.bAugRandomErasing = cfgObj["augRandomErasing"].toBool(cfg.bAugRandomErasing);
        if (cfgObj.contains("augErasingProb"))
            cfg.dAugErasingProb = cfgObj["augErasingProb"].toDouble(cfg.dAugErasingProb);
        if (cfgObj.contains("augErasingRatio"))
            cfg.dAugErasingRatio = cfgObj["augErasingRatio"].toDouble(cfg.dAugErasingRatio);

        // 20260324 ZJH 数据增强（扩展 — 高级混合，contains 检查兼容旧版文件）
        if (cfgObj.contains("augMixup"))
            cfg.bAugMixup = cfgObj["augMixup"].toBool(cfg.bAugMixup);
        if (cfgObj.contains("augMixupAlpha"))
            cfg.dAugMixupAlpha = cfgObj["augMixupAlpha"].toDouble(cfg.dAugMixupAlpha);
        if (cfgObj.contains("augCutMix"))
            cfg.bAugCutMix = cfgObj["augCutMix"].toBool(cfg.bAugCutMix);
        if (cfgObj.contains("augCutMixAlpha"))
            cfg.dAugCutMixAlpha = cfgObj["augCutMixAlpha"].toDouble(cfg.dAugCutMixAlpha);

        // 20260324 ZJH 导出选项
        cfg.bExportOnnx = cfgObj["exportOnnx"].toBool(cfg.bExportOnnx);

        pProject->setTrainingConfig(cfg);  // 20260324 ZJH 应用训练配置到项目
    }

    // 20260324 ZJH 反序列化训练历史记录（兼容旧版本文件：不存在则保持空列表）
    if (root.contains("trainingHistory")) {
        QJsonArray historyArr = root["trainingHistory"].toArray();  // 20260324 ZJH 训练历史 JSON 数组
        for (const QJsonValue& val : historyArr) {
            QJsonObject recObj = val.toObject();  // 20260324 ZJH 单条记录对象
            EpochRecord record;  // 20260324 ZJH 创建记录
            record.nEpoch       = recObj["epoch"].toInt();             // 20260324 ZJH 轮次编号
            record.dTrainLoss   = recObj["trainLoss"].toDouble();      // 20260324 ZJH 训练损失
            record.dValLoss     = recObj["valLoss"].toDouble();        // 20260324 ZJH 验证损失
            record.dValAccuracy = recObj["valAccuracy"].toDouble();    // 20260324 ZJH 验证准确率
            pProject->addEpochRecord(record);  // 20260324 ZJH 添加到训练历史
        }
    }

    // 20260324 ZJH 反序列化最佳模型路径（兼容旧版本文件：不存在则为空）
    if (root.contains("bestModelPath")) {
        pProject->setBestModelPath(root["bestModelPath"].toString());  // 20260324 ZJH 恢复最佳模型路径
    }

    // 20260324 ZJH 反序列化评估结果（兼容旧版本文件：不存在则无评估结果）
    if (root.contains("evaluation")) {
        QJsonObject evalObj = root["evaluation"].toObject();  // 20260324 ZJH 评估结果 JSON 对象
        EvaluationResult eval;  // 20260324 ZJH 创建评估结果实例

        // 20260324 ZJH 核心指标
        eval.dAccuracy  = evalObj["accuracy"].toDouble();   // 20260324 ZJH 准确率
        eval.dPrecision = evalObj["precision"].toDouble();  // 20260324 ZJH 精确率
        eval.dRecall    = evalObj["recall"].toDouble();     // 20260324 ZJH 召回率
        eval.dF1Score   = evalObj["f1Score"].toDouble();    // 20260324 ZJH F1 分数
        eval.dMeanAP    = evalObj["meanAP"].toDouble();     // 20260324 ZJH mAP
        eval.dMIoU      = evalObj["mIoU"].toDouble();       // 20260324 ZJH mIoU
        eval.dAUC       = evalObj["auc"].toDouble();        // 20260324 ZJH AUC

        // 20260324 ZJH 性能统计
        eval.dAvgLatencyMs  = evalObj["avgLatencyMs"].toDouble();   // 20260324 ZJH 平均延迟
        eval.dThroughputFPS = evalObj["throughputFPS"].toDouble();  // 20260324 ZJH 吞吐量

        // 20260324 ZJH 统计摘要
        eval.nTotalImages      = evalObj["totalImages"].toInt();         // 20260324 ZJH 评估图像总数
        eval.nCorrect          = evalObj["correct"].toInt();             // 20260324 ZJH 正确预测数量
        eval.dTotalTimeSeconds = evalObj["totalTimeSeconds"].toDouble(); // 20260324 ZJH 评估总耗时

        // 20260324 ZJH 类别名称列表
        QJsonArray classNamesArr = evalObj["classNames"].toArray();  // 20260324 ZJH 类别名称数组
        for (const QJsonValue& val : classNamesArr) {
            eval.vecClassNames.append(val.toString());  // 20260324 ZJH 恢复类别名称
        }

        // 20260324 ZJH 混淆矩阵（二维数组）
        QJsonArray confusionArr = evalObj["confusion"].toArray();  // 20260324 ZJH 混淆矩阵数组
        for (const QJsonValue& rowVal : confusionArr) {
            QJsonArray rowArr = rowVal.toArray();  // 20260324 ZJH 一行数据
            QVector<int> vecRow;  // 20260324 ZJH 存储一行值
            for (const QJsonValue& cellVal : rowArr) {
                vecRow.append(cellVal.toInt());  // 20260324 ZJH 恢复单元格值
            }
            eval.matConfusion.append(vecRow);  // 20260324 ZJH 追加一行到混淆矩阵
        }

        // 20260324 ZJH 每类精确率
        QJsonArray precPerClassArr = evalObj["precisionPerClass"].toArray();  // 20260324 ZJH 每类精确率数组
        for (const QJsonValue& val : precPerClassArr) {
            eval.vecPrecisionPerClass.append(val.toDouble());  // 20260324 ZJH 恢复精确率
        }

        // 20260324 ZJH 每类召回率
        QJsonArray recPerClassArr = evalObj["recallPerClass"].toArray();  // 20260324 ZJH 每类召回率数组
        for (const QJsonValue& val : recPerClassArr) {
            eval.vecRecallPerClass.append(val.toDouble());  // 20260324 ZJH 恢复召回率
        }

        // 20260324 ZJH 每类 F1 分数
        QJsonArray f1PerClassArr = evalObj["f1PerClass"].toArray();  // 20260324 ZJH 每类 F1 数组
        for (const QJsonValue& val : f1PerClassArr) {
            eval.vecF1PerClass.append(val.toDouble());  // 20260324 ZJH 恢复 F1
        }

        pProject->setEvaluationResult(eval);  // 20260324 ZJH 将反序列化的评估结果设置到项目
    }

    // 20260324 ZJH 加载完成后重置脏标志（刚从磁盘加载的项目无需保存）
    pProject->setDirty(false);

    qDebug() << "[ProjectSerializer] load: 成功，项目:" << pProject->name()
             << "图像:" << pDataset->imageCount()
             << "标签:" << pDataset->labels().size();  // 20260322 ZJH 加载成功日志

    return pProject;  // 20260322 ZJH 返回恢复的项目（unique_ptr 所有权转移给调用者）
}
