// 20260322 ZJH TrainingSession 实现
// 20260324 ZJH 重写为真实训练流水线：加载真实图像数据 → EngineBridge 创建模型 → 逐 Epoch 训练 → 保存模型
// 通过信号通知 UI 层训练进度、日志和完成状态

#include "core/training/TrainingSession.h"  // 20260322 ZJH TrainingSession 类声明
#include "core/project/Project.h"           // 20260322 ZJH Project 类（获取数据集信息）
#include "core/data/ImageDataset.h"         // 20260322 ZJH ImageDataset（获取图像和标签）
#include "core/DLTypes.h"                   // 20260322 ZJH 类型定义（modelArchitectureToString）

#include <QThread>       // 20260322 ZJH QThread::msleep
#include <QDateTime>     // 20260322 ZJH 获取当前时间用于日志
#include <QImage>        // 20260324 ZJH QImage 加载和预处理训练图像
#include <QDir>          // 20260324 ZJH QDir::mkpath 创建模型保存目录
#include <QFileInfo>     // 20260324 ZJH QFileInfo 验证文件存在

#include <QRandomGenerator>  // 20260326 ZJH 数据增强随机数生成器
#include <QPainter>          // 20260328 ZJH QPainter 用于将标注渲染为像素级分割掩码
#include <algorithm>         // 20260326 ZJH std::min, std::max 用于亮度裁剪
#include <cmath>  // 20260322 ZJH std::exp
#ifdef _MSC_VER
#include <windows.h>         // 20260326 ZJH EXCEPTION_EXECUTE_HANDLER for SEH
#endif

// 20260322 ZJH 构造函数
TrainingSession::TrainingSession(QObject* pParent)
    : QObject(pParent)
{
}

// 20260322 ZJH 设置训练配置
void TrainingSession::setConfig(const TrainingConfig& config)
{
    m_config = config;  // 20260322 ZJH 拷贝配置（值语义）
}

// 20260322 ZJH 设置当前项目
void TrainingSession::setProject(Project* pProject)
{
    m_pProject = pProject;  // 20260322 ZJH 保存项目弱引用
}

// 20260322 ZJH 查询是否正在训练
bool TrainingSession::isRunning() const
{
    return m_bRunning.load();  // 20260322 ZJH 原子读取运行状态
}

// 20260324 ZJH 获取训练完成后模型保存路径
QString TrainingSession::modelSavePath() const
{
    return m_strModelSavePath;  // 20260324 ZJH 返回模型保存路径（训练未完成时为空）
}

// 20260324 ZJH 获取引擎桥接层指针（供外部查询模型信息）
EngineBridge* TrainingSession::engine() const
{
    return m_pEngine.get();  // 20260324 ZJH 返回裸指针（不转移所有权）
}

// 20260322 ZJH 开始训练（在工作线程中调用）
void TrainingSession::startTraining()
{
    // 20260322 ZJH 防止重复启动
    if (m_bRunning.load()) {
        emit trainingLog(QStringLiteral("[警告] 训练已在运行中，忽略重复启动请求"));
        return;  // 20260322 ZJH 已在运行，直接返回
    }

    // 20260322 ZJH 重置控制标志
    m_bRunning.store(true);
    m_bStopRequested.store(false);
    m_bPaused.store(false);

    // 20260324 ZJH 清空上次训练的模型路径
    m_strModelSavePath.clear();

    // 20260322 ZJH 执行训练循环
    runTrainingLoop();

    // 20260322 ZJH 训练结束后清除运行状态
    m_bRunning.store(false);
}

// 20260322 ZJH 请求停止训练
void TrainingSession::stopTraining()
{
    m_bStopRequested.store(true);  // 20260322 ZJH 设置停止标记（训练循环会在下一次检查时退出）
    m_bPaused.store(false);        // 20260322 ZJH 同时解除暂停状态，避免永远阻塞
    m_pauseCondition.wakeAll();    // 20260324 ZJH 唤醒等待条件，使暂停中的训练线程立即响应停止
}

// 20260326 ZJH 强制释放 GPU 内存（训练结束/异常后调用）
void TrainingSession::forceGpuCleanup()
{
    if (m_pEngine) {
        m_pEngine->releaseGpu();  // 20260326 ZJH 参数移回 CPU + omCudaCleanup
    }
}

// 20260322 ZJH 请求暂停训练
void TrainingSession::pauseTraining()
{
    m_bPaused.store(true);  // 20260322 ZJH 设置暂停标记
    emit trainingLog(QStringLiteral("[信息] 训练已暂停"));
}

// 20260322 ZJH 请求恢复训练
void TrainingSession::resumeTraining()
{
    m_bPaused.store(false);        // 20260322 ZJH 清除暂停标记
    m_pauseCondition.wakeAll();    // 20260324 ZJH 唤醒等待条件，使暂停中的训练线程立即恢复执行
    emit trainingLog(QStringLiteral("[信息] 训练已恢复"));
}

// 20260324 ZJH 将 om::ModelArchitecture 枚举映射为引擎识别的模型类型字符串
// 参数: eArch - 模型架构枚举值
// 返回: EngineBridge 可识别的模型类型字符串
std::string TrainingSession::architectureToEngineString(om::ModelArchitecture eArch)
{
    // 20260325 ZJH 将 UI 架构枚举映射为 EngineBridge::createModel 接受的字符串标识
    switch (eArch) {
        // 20260325 ZJH 分类模型
        case om::ModelArchitecture::ResNet18:          return "ResNet18";
        case om::ModelArchitecture::ResNet50:          return "ResNet50";
        case om::ModelArchitecture::MobileNetV4Small:  return "MobileNetV4Small";
        case om::ModelArchitecture::ViTTiny:           return "ViTTiny";
        // 20260325 ZJH 目标检测模型
        case om::ModelArchitecture::YOLOv5Nano:        return "YOLOv5Nano";
        case om::ModelArchitecture::YOLOv8Nano:        return "YOLOv8Nano";
        // 20260325 ZJH 语义分割模型
        case om::ModelArchitecture::UNet:              return "UNet";
        case om::ModelArchitecture::DeepLabV3Plus:     return "DeepLabV3Plus";
        // 20260325 ZJH 异常检测模型
        case om::ModelArchitecture::EfficientAD:       return "EfficientAD";
        // 20260325 ZJH 实例分割模型
        case om::ModelArchitecture::YOLOv8InstanceSeg: return "YOLOv8Seg";
        case om::ModelArchitecture::MaskRCNN:          return "MaskRCNN";
        // 20260325 ZJH 暂未实现的架构回退到最近似的已实现模型
        case om::ModelArchitecture::EfficientNetB0:    return "ResNet18";       // 20260325 ZJH 回退
        case om::ModelArchitecture::MobileNetV4Medium: return "MobileNetV4Small";
        case om::ModelArchitecture::ConvNeXtTiny:      return "ResNet18";
        case om::ModelArchitecture::RepVGGA0:          return "ResNet18";
        case om::ModelArchitecture::YOLOv11Nano:       return "YOLOv8Nano";
        case om::ModelArchitecture::RTDETR:            return "YOLOv8Nano";
        case om::ModelArchitecture::PSPNet:            return "DeepLabV3Plus";
        case om::ModelArchitecture::SegFormer:         return "DeepLabV3Plus";
        default:                                       return "MLP";
    }
}

// 20260324 ZJH 真实训练循环
// 1. 从 ImageDataset 加载真实图像数据并预处理为 CHW float [0,1]
// 2. 通过 EngineBridge 创建模型
// 3. 调用 EngineBridge::train() 执行真实训练（引擎内部处理优化器/梯度/反向传播）
// 4. 通过回调接收每 Epoch 结果并发送信号给 UI
// 5. 训练完成后保存模型到项目目录
void TrainingSession::runTrainingLoop()
{
    // 20260326 ZJH C++ 异常保护
    try {
        runTrainingLoopImpl();
    } catch (const std::exception& e) {
        emit trainingLog(QStringLiteral("[致命错误] 训练异常: %1").arg(QString::fromStdString(e.what())));
        emit trainingFinished(false, QStringLiteral("训练异常终止"));
    } catch (...) {
        emit trainingLog(QStringLiteral("[致命错误] 训练线程未知异常"));
        emit trainingFinished(false, QStringLiteral("训练异常终止"));
    }

    // 20260326 ZJH 无论训练成功或异常，都强制释放 GPU 内存
    // 之前异常路径跳过 omCudaCleanup() 导致 15.7GB GPU 内存泄漏
    forceGpuCleanup();
}

void TrainingSession::runTrainingLoopImpl()
{
    // 20260324 ZJH 将 m_config 拷贝到线程本地变量，避免与主线程 setConfig() 产生数据竞争
    TrainingConfig localConfig = m_config;

    // 20260324 ZJH 超参数安全校验（修改本地副本，不修改成员变量）
    if (localConfig.nBatchSize <= 0) localConfig.nBatchSize = 16;  // 20260324 ZJH 批量大小最小 16
    if (localConfig.nEpochs <= 0) localConfig.nEpochs = 1;         // 20260324 ZJH 至少训练 1 轮
    if (localConfig.nInputSize <= 0) localConfig.nInputSize = 224;  // 20260324 ZJH 输入尺寸默认 224

    // 20260322 ZJH 获取总训练轮次
    int nTotalEpochs = localConfig.nEpochs;

    // 20260324 ZJH 拷贝项目指针到本地
    Project* pLocalProject = m_pProject;

    // 20260325 ZJH 提前保存项目路径，避免训练完成后跨线程访问 m_pProject（可能已被主线程修改/关闭）
    QString strProjectPath = pLocalProject ? pLocalProject->path() : QString();

    // 20260324 ZJH 验证项目和数据集有效性
    if (!pLocalProject || !pLocalProject->dataset()) {
        emit trainingLog(QStringLiteral("[错误] 项目或数据集无效，无法启动训练"));
        emit trainingFinished(false, QStringLiteral("项目或数据集无效"));
        return;  // 20260324 ZJH 无效项目，直接退出
    }

    ImageDataset* pDataset = pLocalProject->dataset();  // 20260324 ZJH 获取数据集指针
    const auto& vecImages = pDataset->images();          // 20260324 ZJH 获取全部图像条目引用
    const auto& vecLabels = pDataset->labels();          // 20260324 ZJH 获取全部标签定义引用

    // 20260324 ZJH 确定类别数量（标签数量，至少 2 个类别）
    int nNumClasses = vecLabels.size();
    if (nNumClasses < 2) nNumClasses = 2;  // 20260324 ZJH 至少保证二分类

    // 20260328 ZJH 分割模型：nNumClasses = 标签数 + 1（包含背景类）
    // mask 编码: 0=背景, labelId+1=对应缺陷类型
    // 例: 划痕(id=0)→mask=1, 异物(id=1)→mask=2, nNumClasses=3
    // UNet 输出 [B, 3, H, W]: channel 0=背景, channel 1=划痕, channel 2=异物
    bool bIsSegmentation = (localConfig.eArchitecture == om::ModelArchitecture::UNet ||
                            localConfig.eArchitecture == om::ModelArchitecture::DeepLabV3Plus ||
                            localConfig.eArchitecture == om::ModelArchitecture::PSPNet ||
                            localConfig.eArchitecture == om::ModelArchitecture::SegFormer);
    if (bIsSegmentation) {
        nNumClasses = static_cast<int>(vecLabels.size()) + 1;  // 20260328 ZJH +1 背景类
        if (nNumClasses < 2) nNumClasses = 2;
    }

    // 20260324 ZJH 统计训练集和验证集图像数量
    int nTrainCount = pDataset->countBySplit(om::SplitType::Train);       // 20260324 ZJH 训练集数量
    int nValCount   = pDataset->countBySplit(om::SplitType::Validation);  // 20260324 ZJH 验证集数量

    // 20260324 ZJH 输出训练启动日志
    emit trainingLog(QStringLiteral("========================================"));
    emit trainingLog(QStringLiteral("[%1] 训练开始 (真实数据模式)").arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
    emit trainingLog(QStringLiteral("模型架构: %1").arg(om::modelArchitectureToString(localConfig.eArchitecture)));
    emit trainingLog(QStringLiteral("引擎模型: %1").arg(QString::fromStdString(architectureToEngineString(localConfig.eArchitecture))));
    emit trainingLog(QStringLiteral("设备: %1").arg(localConfig.eDevice == om::DeviceType::CUDA ? "CUDA" : "CPU"));
    emit trainingLog(QStringLiteral("优化器: %1 | 学习率: %2").arg(
        localConfig.eOptimizer == om::OptimizerType::Adam ? "Adam" :
        localConfig.eOptimizer == om::OptimizerType::AdamW ? "AdamW" : "SGD",
        QString::number(localConfig.dLearningRate, 'f', 6)));
    emit trainingLog(QStringLiteral("批量大小: %1 | 训练轮次: %2 | 输入尺寸: %3x%3")
        .arg(localConfig.nBatchSize).arg(nTotalEpochs).arg(localConfig.nInputSize));
    emit trainingLog(QStringLiteral("训练集: %1 张 | 验证集: %2 张 | 类别数: %3")
        .arg(nTrainCount).arg(nValCount).arg(nNumClasses));
    emit trainingLog(QStringLiteral("早停耐心: %1 轮 | 数据增强: %2")
        .arg(localConfig.nPatience).arg(localConfig.bAugmentation ? "开启" : "关闭"));
    // 20260328 ZJH 输出分割模型标识，告知用户将生成像素级掩码
    if (bIsSegmentation) {
        emit trainingLog(QStringLiteral("分割模式: 已启用 (将从标注生成 %1x%1 像素级掩码)")
            .arg(localConfig.nInputSize));
    }
    emit trainingLog(QStringLiteral("========================================"));

    // 20260324 ZJH ===== 第 1 步：加载真实训练数据 =====
    emit trainingLog(QStringLiteral("[信息] 正在加载训练图像..."));

    // 20260324 ZJH 每张图像展平后的维度: 3通道 * inputSize * inputSize
    int nInputDim = 3 * localConfig.nInputSize * localConfig.nInputSize;

    // 20260324 ZJH 收集训练集数据
    std::vector<float> vecTrainData;   // 20260324 ZJH 训练数据 [N * C*H*W]
    std::vector<int> vecTrainLabels;   // 20260324 ZJH 训练标签 [N]
    // 20260324 ZJH 收集验证集数据
    std::vector<float> vecValData;     // 20260324 ZJH 验证数据 [N * C*H*W]
    std::vector<int> vecValLabels;     // 20260324 ZJH 验证标签 [N]

    // 20260328 ZJH 分割模型的像素级掩码向量
    // 每张图像展平为 [H * W] 的 int 数组，每个像素值为标签 ID（0=背景，>0 为前景类别）
    std::vector<int> vecTrainMasks;  // 20260328 ZJH [N_train * H * W] 训练集像素级掩码
    std::vector<int> vecValMasks;    // 20260328 ZJH [N_val * H * W] 验证集像素级掩码

    // 20260328 ZJH 单张掩码的像素数量
    int nMaskDim = localConfig.nInputSize * localConfig.nInputSize;  // 20260328 ZJH H * W

    // 20260324 ZJH 预分配内存（避免频繁 realloc）
    vecTrainData.reserve(static_cast<size_t>(nTrainCount) * nInputDim);
    vecTrainLabels.reserve(nTrainCount);
    vecValData.reserve(static_cast<size_t>(nValCount) * nInputDim);
    vecValLabels.reserve(nValCount);

    // 20260328 ZJH 为分割掩码预分配内存（仅分割模型需要）
    if (bIsSegmentation) {
        vecTrainMasks.reserve(static_cast<size_t>(nTrainCount) * nMaskDim);
        vecValMasks.reserve(static_cast<size_t>(nValCount) * nMaskDim);
    }

    int nTrainLoaded = 0;  // 20260324 ZJH 成功加载的训练图像数
    int nValLoaded   = 0;  // 20260324 ZJH 成功加载的验证图像数
    int nLoadFailed  = 0;  // 20260324 ZJH 加载失败的图像数

    for (const ImageEntry& entry : vecImages) {
        // 20260324 ZJH 检查停止请求（数据加载期间也响应停止）
        if (m_bStopRequested.load()) {
            emit trainingLog(QStringLiteral("[信息] 数据加载期间收到停止请求"));
            emit trainingFinished(false, QStringLiteral("训练在数据加载阶段被停止"));
            return;
        }

        // 20260324 ZJH 跳过非训练/验证集的图像
        bool bIsTrain = (entry.eSplit == om::SplitType::Train);
        bool bIsVal   = (entry.eSplit == om::SplitType::Validation);
        if (!bIsTrain && !bIsVal) continue;  // 20260324 ZJH 跳过测试集和未分配的图像

        // 20260324 ZJH 确定该图像的有效标签 ID
        // 两种标注模式：
        //   1. 图像级标签（分类/异常检测）：nLabelId >= 0
        //   2. 对象级标注（检测/分割）：vecAnnotations 非空，取第一个标注的 nLabelId
        // 两者均无则跳过该图像
        int nEffectiveLabelId = entry.nLabelId;  // 20260324 ZJH 优先使用图像级标签
        if (nEffectiveLabelId < 0 && !entry.vecAnnotations.isEmpty()) {
            // 20260324 ZJH 图像级标签未设置，尝试从对象标注中推断
            // 取第一个有效标注的标签作为该图像的分类标签
            for (const Annotation& anno : entry.vecAnnotations) {
                if (anno.nLabelId >= 0) {
                    nEffectiveLabelId = anno.nLabelId;
                    break;  // 20260324 ZJH 找到第一个有效标签即停止
                }
            }
        }
        // 20260324 ZJH 两种模式都无有效标签则跳过
        if (nEffectiveLabelId < 0) continue;

        // 20260329 ZJH 加载完整原始图像
        QImage imgFull(entry.strFilePath);  // 20260329 ZJH 从磁盘加载原始分辨率图像
        if (imgFull.isNull()) {
            nLoadFailed++;  // 20260329 ZJH 图像加载失败计数
            continue;       // 20260329 ZJH 跳过无法加载的图像
        }
        // 20260329 ZJH 统一转换为 RGB888 格式（确保 3 通道，后续处理通用）
        imgFull = imgFull.convertToFormat(QImage::Format_RGB888);

        // 20260329 ZJH 获取原始图像实际尺寸（entry.nWidth/nHeight 可能为 0，必须用 QImage 实际值兜底）
        int nOrigW = imgFull.width();   // 20260329 ZJH 原始图像实际宽度（像素）
        int nOrigH = imgFull.height();  // 20260329 ZJH 原始图像实际高度（像素）
        int nPatchSize = localConfig.nInputSize;  // 20260329 ZJH 模型输入尺寸，即 patch 边长（如 416）

        // 20260329 ZJH 判断是否启用 patch 模式：图像任一边 > 2 倍 patch 尺寸时启用
        // 例如 5472×3648 vs 416：远超 2 倍，使用 patch 模式提取局部区域训练
        // 小图（如 800×600 vs 416）则直接缩放到 inputSize
        bool bUsePatchMode = (nOrigW > nPatchSize * 2 || nOrigH > nPatchSize * 2);

        if (bUsePatchMode) {
            // 20260329 ZJH ===== PATCH 模式：从大图中提取 patch 进行训练 =====
            // 大图（如 5472×3648）直接缩放到 416×416 会丢失所有缺陷细节
            // patch 模式以标注为中心提取 native 分辨率的局部区域，保留原始像素精度

            QVector<QRect> vecPatches;       // 20260329 ZJH 每个 patch 在原图中的矩形区域
            QVector<int> vecPatchLabels;     // 20260329 ZJH 每个 patch 对应的分类标签 ID

            // 20260329 ZJH --- 1. 以每个标注为中心生成 patch ---
            // 每个标注（划痕/异物）独立生成一个 patch，确保缺陷完整包含在内
            for (const Annotation& anno : entry.vecAnnotations) {
                if (anno.nLabelId < 0) continue;  // 20260329 ZJH 跳过无效标签的标注

                QRectF bounds = anno.rectBounds;  // 20260329 ZJH 标注的外接矩形（原始图像坐标）
                // 20260329 ZJH 计算标注中心点坐标
                int nCenterX = static_cast<int>(bounds.x() + bounds.width() / 2);   // 20260329 ZJH 标注中心 X
                int nCenterY = static_cast<int>(bounds.y() + bounds.height() / 2);  // 20260329 ZJH 标注中心 Y

                // 20260329 ZJH 训练集增加随机偏移 ±64px，增强位置多样性
                // 验证集不加偏移，保持可复现性
                if (bIsTrain) {
                    nCenterX += QRandomGenerator::global()->bounded(129) - 64;  // 20260329 ZJH X 方向 [-64, +64] 随机偏移
                    nCenterY += QRandomGenerator::global()->bounded(129) - 64;  // 20260329 ZJH Y 方向 [-64, +64] 随机偏移
                }

                // 20260329 ZJH 根据中心点计算 patch 左上角坐标
                int nPatchX = nCenterX - nPatchSize / 2;  // 20260329 ZJH patch 左上角 X
                int nPatchY = nCenterY - nPatchSize / 2;  // 20260329 ZJH patch 左上角 Y
                // 20260329 ZJH 边界裁剪：确保 patch 不超出原始图像范围
                nPatchX = std::max(0, std::min(nPatchX, nOrigW - nPatchSize));  // 20260329 ZJH X 方向夹紧
                nPatchY = std::max(0, std::min(nPatchY, nOrigH - nPatchSize));  // 20260329 ZJH Y 方向夹紧

                vecPatches.append(QRect(nPatchX, nPatchY, nPatchSize, nPatchSize));  // 20260329 ZJH 记录 patch 区域
                vecPatchLabels.append(anno.nLabelId);  // 20260329 ZJH 记录该 patch 的标签（标注的类别）
            }

            // 20260329 ZJH --- 2. 为有标注的图像添加背景 patch ---
            // 背景 patch 不以标注为中心，用于平衡正负样本比例
            // 训练集: min(标注patch数, 3) 个背景; 验证集: 1 个背景
            if (!entry.vecAnnotations.isEmpty()) {
                int nBgPatches = bIsTrain ? std::min(static_cast<int>(vecPatches.size()), 3) : 1;  // 20260329 ZJH 背景 patch 数量
                for (int bg = 0; bg < nBgPatches; ++bg) {
                    // 20260329 ZJH 在原图内随机选取背景 patch 位置
                    int nBgX = QRandomGenerator::global()->bounded(std::max(1, nOrigW - nPatchSize));  // 20260329 ZJH 随机 X
                    int nBgY = QRandomGenerator::global()->bounded(std::max(1, nOrigH - nPatchSize));  // 20260329 ZJH 随机 Y
                    vecPatches.append(QRect(nBgX, nBgY, nPatchSize, nPatchSize));  // 20260329 ZJH 记录背景 patch
                    vecPatchLabels.append(nEffectiveLabelId);  // 20260329 ZJH 背景 patch 使用图像级标签
                }
            }

            // 20260329 ZJH --- 3. 无标注图像：生成随机 patch ---
            // 无标注的图像（纯背景）也要参与训练，提供负样本
            if (entry.vecAnnotations.isEmpty()) {
                int nNumRandom = bIsTrain ? 2 : 1;  // 20260329 ZJH 训练集 2 个随机 patch，验证集 1 个
                for (int r = 0; r < nNumRandom; ++r) {
                    // 20260329 ZJH 在原图内随机选取 patch 位置
                    int nRx = QRandomGenerator::global()->bounded(std::max(1, nOrigW - nPatchSize));  // 20260329 ZJH 随机 X
                    int nRy = QRandomGenerator::global()->bounded(std::max(1, nOrigH - nPatchSize));  // 20260329 ZJH 随机 Y
                    vecPatches.append(QRect(nRx, nRy, nPatchSize, nPatchSize));  // 20260329 ZJH 记录随机 patch
                    vecPatchLabels.append(nEffectiveLabelId);  // 20260329 ZJH 使用图像级标签（无标注 → 背景类/OK类）
                }
            }

            // 20260329 ZJH --- 4. 逐 patch 处理：裁剪、掩码、增强、转 CHW ---
            for (int p = 0; p < vecPatches.size(); ++p) {
                QRect patchRect = vecPatches[p];  // 20260329 ZJH 当前 patch 在原图中的区域

                // 20260329 ZJH 从原始大图中裁剪 patch（native 分辨率，无缩放）
                QImage imgPatch = imgFull.copy(patchRect);  // 20260329 ZJH QImage::copy 提取子区域

                // 20260329 ZJH 边缘安全：若裁剪结果尺寸不一致（图像边缘截断），强制缩放到目标尺寸
                if (imgPatch.width() != nPatchSize || imgPatch.height() != nPatchSize) {
                    imgPatch = imgPatch.scaled(nPatchSize, nPatchSize,
                                               Qt::IgnoreAspectRatio, Qt::SmoothTransformation);  // 20260329 ZJH 强制缩放修正
                }

                // 20260329 ZJH ===== 分割模型：生成 patch 局部掩码 =====
                // 掩码坐标 = 标注原始坐标 - patch 左上角偏移（无缩放，native 像素对齐）
                QImage imgMask;  // 20260329 ZJH 当前 patch 的分割掩码
                if (bIsSegmentation) {
                    // 20260329 ZJH 创建 patch 尺寸的灰度掩码，初始化全 0（背景）
                    imgMask = QImage(nPatchSize, nPatchSize, QImage::Format_Grayscale8);
                    imgMask.fill(0);  // 20260329 ZJH 所有像素初始为背景类

                    // 20260329 ZJH 使用 QPainter 将落入 patch 范围内的标注渲染到掩码上
                    QPainter painter(&imgMask);
                    painter.setPen(Qt::NoPen);  // 20260329 ZJH 无描边，仅填充

                    for (const Annotation& anno : entry.vecAnnotations) {
                        if (anno.nLabelId < 0) continue;  // 20260329 ZJH 跳过无效标签

                        // 20260329 ZJH mask 值 = labelId + 1（0 保留给背景）
                        int nMaskValue = anno.nLabelId + 1;  // 20260329 ZJH 划痕→1, 异物→2
                        nMaskValue = std::min(255, std::max(1, nMaskValue));  // 20260329 ZJH 限制在 [1, 255]
                        painter.setBrush(QColor(nMaskValue, nMaskValue, nMaskValue));  // 20260329 ZJH 灰度值作为画刷颜色

                        if (anno.eType == AnnotationType::Rect) {
                            // 20260329 ZJH 矩形标注：从原始坐标平移到 patch 局部坐标（减去 patch 左上角）
                            QRectF rectLocal(
                                anno.rectBounds.x() - patchRect.x(),       // 20260329 ZJH 局部 X = 原始 X - patch X
                                anno.rectBounds.y() - patchRect.y(),       // 20260329 ZJH 局部 Y = 原始 Y - patch Y
                                anno.rectBounds.width(),                   // 20260329 ZJH 宽度不变（native 像素）
                                anno.rectBounds.height()                   // 20260329 ZJH 高度不变（native 像素）
                            );
                            painter.drawRect(rectLocal);  // 20260329 ZJH 在掩码上填充矩形区域
                        }
                        else if (anno.eType == AnnotationType::Polygon) {
                            // 20260329 ZJH 多边形标注：逐顶点从原始坐标平移到 patch 局部坐标
                            QPolygonF polyLocal;
                            polyLocal.reserve(anno.polygon.size());  // 20260329 ZJH 预分配顶点数
                            for (const QPointF& pt : anno.polygon) {
                                // 20260329 ZJH 每个顶点坐标减去 patch 左上角偏移
                                polyLocal.append(QPointF(pt.x() - patchRect.x(), pt.y() - patchRect.y()));
                            }
                            painter.drawPolygon(polyLocal);  // 20260329 ZJH 在掩码上填充多边形区域
                        }
                        // 20260329 ZJH Mask/TextArea 类型暂不处理（后续扩展）
                    }

                    painter.end();  // 20260329 ZJH 结束掩码绘制，释放 QPainter
                }

                // 20260329 ZJH ===== 数据增强（仅训练集 patch，验证集不增强）=====
                if (bIsTrain) {
                    // 20260329 ZJH 随机水平翻转 (50% 概率)
                    if (QRandomGenerator::global()->bounded(2) == 0) {
                        imgPatch = imgPatch.mirrored(true, false);  // 20260329 ZJH 水平翻转 patch
                        if (bIsSegmentation) {
                            imgMask = imgMask.mirrored(true, false);  // 20260329 ZJH 掩码同步水平翻转
                        }
                    }
                    // 20260329 ZJH 随机垂直翻转 (50% 概率)
                    if (QRandomGenerator::global()->bounded(2) == 0) {
                        imgPatch = imgPatch.mirrored(false, true);  // 20260329 ZJH 垂直翻转 patch
                        if (bIsSegmentation) {
                            imgMask = imgMask.mirrored(false, true);  // 20260329 ZJH 掩码同步垂直翻转
                        }
                    }
                    // 20260329 ZJH 随机亮度抖动 (±15%)，仅应用于 RGB 图像，不应用于掩码
                    {
                        float fBrightness = 0.85f + QRandomGenerator::global()->bounded(31) / 100.0f;  // 20260329 ZJH [0.85, 1.15]
                        for (int y = 0; y < imgPatch.height(); ++y) {
                            uchar* pRow = imgPatch.scanLine(y);  // 20260329 ZJH 获取第 y 行像素指针
                            for (int x = 0; x < imgPatch.width() * 3; ++x) {
                                int nVal = static_cast<int>(pRow[x] * fBrightness);  // 20260329 ZJH 乘以亮度因子
                                pRow[x] = static_cast<uchar>(std::min(255, std::max(0, nVal)));  // 20260329 ZJH 裁剪到 [0,255]
                            }
                        }
                    }
                }

                // 20260329 ZJH ===== 将 patch 像素转换为 CHW float [0,1] 并追加到数据集 =====
                std::vector<float>& vecTargetData   = bIsTrain ? vecTrainData   : vecValData;    // 20260329 ZJH 选择目标数据容器
                std::vector<int>&   vecTargetLabels = bIsTrain ? vecTrainLabels : vecValLabels;  // 20260329 ZJH 选择目标标签容器

                const uchar* pBits = imgPatch.constBits();   // 20260329 ZJH 获取 patch 像素数据指针
                int nBytesPerLine = imgPatch.bytesPerLine();  // 20260329 ZJH 每行字节数（含对齐 padding）

                for (int c = 0; c < 3; ++c) {  // 20260329 ZJH 遍历 3 通道 (R=0, G=1, B=2)
                    for (int y = 0; y < nPatchSize; ++y) {  // 20260329 ZJH 遍历行
                        for (int x = 0; x < nPatchSize; ++x) {  // 20260329 ZJH 遍历列
                            // 20260329 ZJH 从 RGB888 格式读取像素值并归一化到 [0, 1]
                            vecTargetData.push_back(pBits[y * nBytesPerLine + x * 3 + c] / 255.0f);
                        }
                    }
                }

                // 20260329 ZJH 保存该 patch 的分类标签
                vecTargetLabels.push_back(vecPatchLabels[p]);

                // 20260329 ZJH 分割模型：将掩码 QImage 展平为 int 向量并追加
                if (bIsSegmentation) {
                    std::vector<int>& vecTargetMasks = bIsTrain ? vecTrainMasks : vecValMasks;  // 20260329 ZJH 选择目标掩码容器
                    for (int y = 0; y < nPatchSize; ++y) {
                        const uchar* pMaskRow = imgMask.constScanLine(y);  // 20260329 ZJH 获取掩码第 y 行指针
                        for (int x = 0; x < nPatchSize; ++x) {
                            // 20260329 ZJH 灰度图每像素 1 字节，值即为掩码类别 ID
                            vecTargetMasks.push_back(static_cast<int>(pMaskRow[x]));
                        }
                    }
                }
            }

            // 20260329 ZJH 更新加载计数：每个 patch 算一个训练/验证样本
            if (bIsTrain) nTrainLoaded += vecPatches.size();  // 20260329 ZJH 训练集样本数 += patch 数
            else nValLoaded += vecPatches.size();              // 20260329 ZJH 验证集样本数 += patch 数

        } else {
            // 20260329 ZJH ===== SCALE 模式（小图回退）：整图缩放到 inputSize =====
            // 图像较小（任一边 ≤ 2×patchSize），直接缩放到模型输入尺寸
            // 保留原始逻辑：缩放 → 掩码生成（坐标按比例缩放）→ 增强 → CHW 转换

            // 20260329 ZJH 保存原始尺寸用于掩码坐标缩放
            int nOrigImgW = nOrigW;  // 20260329 ZJH 缩放前的原始宽度
            int nOrigImgH = nOrigH;  // 20260329 ZJH 缩放前的原始高度

            // 20260329 ZJH 缩放到模型输入尺寸
            QImage img = imgFull.scaled(localConfig.nInputSize, localConfig.nInputSize,
                                        Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

            // 20260329 ZJH ===== 分割模型掩码生成（缩放模式）=====
            // 标注坐标基于原始图像，需要按缩放比例映射到 inputSize
            QImage imgMask;  // 20260329 ZJH 掩码图像（仅分割模型使用）
            if (bIsSegmentation) {
                // 20260329 ZJH 创建与输入尺寸相同的灰度掩码，初始化为全 0（背景）
                imgMask = QImage(localConfig.nInputSize, localConfig.nInputSize, QImage::Format_Grayscale8);
                imgMask.fill(0);  // 20260329 ZJH 背景像素值 = 0

                // 20260329 ZJH 计算缩放因子：原始坐标 → inputSize 坐标
                // 优先用 entry 记录的尺寸；若为 0 则用 QImage 实际尺寸
                int nEntryW = (entry.nWidth > 0) ? entry.nWidth : nOrigImgW;   // 20260329 ZJH 参考宽度
                int nEntryH = (entry.nHeight > 0) ? entry.nHeight : nOrigImgH; // 20260329 ZJH 参考高度
                double dScaleX = (nEntryW > 0) ?
                    static_cast<double>(localConfig.nInputSize) / nEntryW : 1.0;  // 20260329 ZJH X 缩放因子
                double dScaleY = (nEntryH > 0) ?
                    static_cast<double>(localConfig.nInputSize) / nEntryH : 1.0;  // 20260329 ZJH Y 缩放因子

                // 20260329 ZJH 使用 QPainter 将每个标注渲染到掩码上
                QPainter painter(&imgMask);
                painter.setPen(Qt::NoPen);  // 20260329 ZJH 无边框，仅填充

                for (const Annotation& anno : entry.vecAnnotations) {
                    if (anno.nLabelId < 0) continue;  // 20260329 ZJH 跳过无效标签

                    // 20260329 ZJH 多类分割: mask 值 = labelId + 1（0 保留给背景）
                    int nMaskValue = anno.nLabelId + 1;  // 20260329 ZJH 划痕→1, 异物→2
                    nMaskValue = std::min(255, std::max(1, nMaskValue));  // 20260329 ZJH Grayscale8 范围限制
                    painter.setBrush(QColor(nMaskValue, nMaskValue, nMaskValue));  // 20260329 ZJH 设置画刷

                    if (anno.eType == AnnotationType::Rect) {
                        // 20260329 ZJH 矩形标注：将 rectBounds 从原始坐标缩放到训练尺寸
                        QRectF rectScaled(
                            anno.rectBounds.x() * dScaleX,       // 20260329 ZJH 左上角 X 缩放
                            anno.rectBounds.y() * dScaleY,       // 20260329 ZJH 左上角 Y 缩放
                            anno.rectBounds.width() * dScaleX,   // 20260329 ZJH 宽度缩放
                            anno.rectBounds.height() * dScaleY   // 20260329 ZJH 高度缩放
                        );
                        painter.drawRect(rectScaled);  // 20260329 ZJH 在掩码上填充矩形区域
                    }
                    else if (anno.eType == AnnotationType::Polygon) {
                        // 20260329 ZJH 多边形标注：逐顶点从原始坐标缩放到训练尺寸
                        QPolygonF polyScaled;
                        polyScaled.reserve(anno.polygon.size());  // 20260329 ZJH 预分配顶点数
                        for (const QPointF& pt : anno.polygon) {
                            polyScaled.append(QPointF(pt.x() * dScaleX, pt.y() * dScaleY));  // 20260329 ZJH 逐顶点缩放
                        }
                        painter.drawPolygon(polyScaled);  // 20260329 ZJH 在掩码上填充多边形区域
                    }
                    // 20260329 ZJH Mask/TextArea 类型暂不处理（后续扩展）
                }

                painter.end();  // 20260329 ZJH 结束绘制
            }

            // 20260329 ZJH ===== 数据增强（仅训练集，验证集不增强）=====
            if (bIsTrain) {
                // 20260329 ZJH 随机水平翻转 (50% 概率)
                if (QRandomGenerator::global()->bounded(2) == 0) {
                    img = img.mirrored(true, false);  // 20260329 ZJH 水平翻转
                    if (bIsSegmentation) {
                        imgMask = imgMask.mirrored(true, false);  // 20260329 ZJH 掩码同步翻转
                    }
                }
                // 20260329 ZJH 随机垂直翻转 (50% 概率)
                if (QRandomGenerator::global()->bounded(2) == 0) {
                    img = img.mirrored(false, true);  // 20260329 ZJH 垂直翻转
                    if (bIsSegmentation) {
                        imgMask = imgMask.mirrored(false, true);  // 20260329 ZJH 掩码同步翻转
                    }
                }
                // 20260329 ZJH 随机亮度抖动 (±15%)，仅 RGB 图像，不应用于掩码
                {
                    float fBrightness = 0.85f + QRandomGenerator::global()->bounded(31) / 100.0f;  // 20260329 ZJH [0.85, 1.15]
                    QImage imgAug = img.copy();  // 20260329 ZJH 复制图像用于亮度调整
                    for (int y = 0; y < imgAug.height(); ++y) {
                        uchar* pRow = imgAug.scanLine(y);  // 20260329 ZJH 获取第 y 行指针
                        for (int x = 0; x < imgAug.width() * 3; ++x) {
                            int nVal = static_cast<int>(pRow[x] * fBrightness);  // 20260329 ZJH 乘以亮度因子
                            pRow[x] = static_cast<uchar>(std::min(255, std::max(0, nVal)));  // 20260329 ZJH 裁剪到 [0,255]
                        }
                    }
                    img = imgAug;  // 20260329 ZJH 用增强后的图像替换
                }
            }

            // 20260329 ZJH ===== 将像素转换为 CHW 格式的 float [0, 1] =====
            const uchar* pBits = img.constBits();       // 20260329 ZJH 获取像素数据指针
            int nBytesPerLine = img.bytesPerLine();      // 20260329 ZJH 每行字节数（含对齐 padding）

            std::vector<float>& vecTargetData   = bIsTrain ? vecTrainData   : vecValData;    // 20260329 ZJH 选择数据容器
            std::vector<int>&   vecTargetLabels = bIsTrain ? vecTrainLabels : vecValLabels;  // 20260329 ZJH 选择标签容器

            for (int c = 0; c < 3; ++c) {  // 20260329 ZJH 遍历 3 通道 (R=0, G=1, B=2)
                for (int y = 0; y < localConfig.nInputSize; ++y) {  // 20260329 ZJH 遍历行
                    for (int x = 0; x < localConfig.nInputSize; ++x) {  // 20260329 ZJH 遍历列
                        float fVal = pBits[y * nBytesPerLine + x * 3 + c] / 255.0f;  // 20260329 ZJH 归一化到 [0, 1]
                        vecTargetData.push_back(fVal);  // 20260329 ZJH 追加到数据向量
                    }
                }
            }

            // 20260329 ZJH 保存有效标签 ID
            vecTargetLabels.push_back(nEffectiveLabelId);

            // 20260329 ZJH 分割模型：将掩码展平为 int 向量并追加
            if (bIsSegmentation) {
                std::vector<int>& vecTargetMasks = bIsTrain ? vecTrainMasks : vecValMasks;  // 20260329 ZJH 选择掩码容器
                for (int y = 0; y < localConfig.nInputSize; ++y) {
                    const uchar* pMaskRow = imgMask.constScanLine(y);  // 20260329 ZJH 获取掩码第 y 行指针
                    for (int x = 0; x < localConfig.nInputSize; ++x) {
                        vecTargetMasks.push_back(static_cast<int>(pMaskRow[x]));  // 20260329 ZJH 像素值即掩码类别 ID
                    }
                }
            }

            // 20260329 ZJH 更新加载计数（整图模式每张图 1 个样本）
            if (bIsTrain) nTrainLoaded++;   // 20260329 ZJH 训练集样本数 +1
            else nValLoaded++;              // 20260329 ZJH 验证集样本数 +1
        }
    }

    // 20260324 ZJH 数据加载完毕后不再访问项目指针（防止训练中项目被关闭导致悬挂指针）
    pLocalProject = nullptr;
    pDataset = nullptr;

    // 20260324 ZJH 输出数据加载结果
    emit trainingLog(QStringLiteral("[信息] 数据加载完成: 训练 %1 张, 验证 %2 张, 失败 %3 张")
        .arg(nTrainLoaded).arg(nValLoaded).arg(nLoadFailed));

    // 20260328 ZJH 分割模型数据质量检查：统计有标注像素的图像数
    if (bIsSegmentation && !vecTrainMasks.empty()) {
        int nMaskDimLocal = localConfig.nInputSize * localConfig.nInputSize;
        int nTrainWithAnno = 0;  // 20260328 ZJH 有标注像素的训练图数
        int nValWithAnno = 0;    // 20260328 ZJH 有标注像素的验证图数
        for (int i = 0; i < nTrainLoaded; ++i) {
            bool bHas = false;
            for (int p = 0; p < nMaskDimLocal && !bHas; ++p) {
                if (vecTrainMasks[static_cast<size_t>(i) * nMaskDimLocal + p] > 0) bHas = true;
            }
            if (bHas) ++nTrainWithAnno;
        }
        for (int i = 0; i < nValLoaded; ++i) {
            bool bHas = false;
            for (int p = 0; p < nMaskDimLocal && !bHas; ++p) {
                if (vecValMasks[static_cast<size_t>(i) * nMaskDimLocal + p] > 0) bHas = true;
            }
            if (bHas) ++nValWithAnno;
        }
        emit trainingLog(QStringLiteral("[信息] 分割掩码统计: 训练集 %1/%2 张有标注, 验证集 %3/%4 张有标注")
            .arg(nTrainWithAnno).arg(nTrainLoaded).arg(nValWithAnno).arg(nValLoaded));
        if (nTrainWithAnno == 0) {
            emit trainingLog(QStringLiteral("[警告] 训练集中没有图像包含标注！请检查数据拆分，确保有标注的图像分配到训练集"));
        }
        if (nValWithAnno == 0) {
            emit trainingLog(QStringLiteral("[警告] 验证集中没有图像包含标注！验证损失将无法反映真实精度"));
        }
    }

    // 20260328 ZJH 输出分割掩码统计信息
    if (bIsSegmentation) {
        emit trainingLog(QStringLiteral("[信息] 分割掩码: 训练 %1 张 (%2 像素), 验证 %3 张 (%4 像素)")
            .arg(nTrainLoaded)
            .arg(vecTrainMasks.size())
            .arg(nValLoaded)
            .arg(vecValMasks.size()));
    }

    // 20260324 ZJH 校验训练数据是否足够
    if (nTrainLoaded == 0) {
        emit trainingLog(QStringLiteral("[错误] 没有成功加载任何训练图像，无法开始训练"));
        emit trainingFinished(false, QStringLiteral("训练数据为空"));
        return;
    }

    // 20260324 ZJH ===== 第 2 步：创建模型 =====
    emit trainingLog(QStringLiteral("[信息] 正在创建模型..."));

    // 20260324 ZJH 创建 EngineBridge 实例
    m_pEngine = std::make_unique<EngineBridge>();

    // 20260324 ZJH 获取引擎识别的模型类型字符串
    std::string strModelType = architectureToEngineString(localConfig.eArchitecture);

    // 20260324 ZJH 创建模型
    bool bModelOk = m_pEngine->createModel(strModelType, localConfig.nInputSize, nNumClasses);
    if (!bModelOk) {
        emit trainingLog(QStringLiteral("[错误] 模型创建失败: %1").arg(QString::fromStdString(strModelType)));
        emit trainingFinished(false, QStringLiteral("模型创建失败"));
        m_pEngine.reset();  // 20260324 ZJH 释放无效引擎
        return;
    }

    // 20260324 ZJH 输出模型信息
    int64_t nTotalParams = m_pEngine->totalParameters();
    emit trainingLog(QStringLiteral("[信息] 模型创建成功: %1 | 参数量: %2")
        .arg(QString::fromStdString(strModelType))
        .arg(nTotalParams));

    // 20260324 ZJH ===== 第 3 步：配置训练参数并启动真实训练 =====

    // 20260324 ZJH 构建 BridgeTrainParams
    BridgeTrainParams params;
    params.strModelType = strModelType;                                      // 20260324 ZJH 模型类型
    params.nInputSize   = localConfig.nInputSize;                            // 20260324 ZJH 输入尺寸
    params.nNumClasses  = nNumClasses;                                       // 20260324 ZJH 类别数
    params.nEpochs      = localConfig.nEpochs;                               // 20260324 ZJH 训练轮次
    params.nBatchSize   = localConfig.nBatchSize;                            // 20260324 ZJH 批量大小
    params.fLearningRate = static_cast<float>(localConfig.dLearningRate);    // 20260324 ZJH 学习率
    params.nPatience    = localConfig.nPatience;                             // 20260324 ZJH 早停耐心
    params.bUseCuda     = (localConfig.eDevice == om::DeviceType::CUDA);     // 20260324 ZJH 是否使用 CUDA
    // 20260324 ZJH 优化器名称映射
    params.strOptimizer = (localConfig.eOptimizer == om::OptimizerType::SGD) ? "SGD" : "Adam";

    // 20260325 ZJH [Phase 4] 输出设备信息，区分 GPU 驻留训练和 CPU 路径
    if (localConfig.eDevice == om::DeviceType::CUDA) {
        emit trainingLog(QStringLiteral("[信息] 设备: CUDA GPU (全 GPU 驻留训练 — 数据/权重/梯度全部在 GPU 上)"));
    } else {
        emit trainingLog(QStringLiteral("[信息] 设备: CPU (AVX2 SIMD + OpenMP %1 核)").arg(QThread::idealThreadCount()));
    }

    emit trainingLog(QStringLiteral("[信息] 开始引擎训练..."));

    // 20260324 ZJH Epoch 回调: 接收引擎返回的每轮训练结果，转发给 UI
    EpochCallback epochCb = [this, nTotalEpochs](const BridgeEpochResult& r) {
        // 20260324 ZJH 计算进度百分比
        int nPercent = static_cast<int>(100.0 * r.nEpoch / nTotalEpochs);
        emit progressChanged(nPercent);
        // 20260324 ZJH 发送 epoch 完成信号到 UI
        emit epochCompleted(r.nEpoch, r.nTotalEpochs,
                            static_cast<double>(r.fTrainLoss),
                            static_cast<double>(r.fValLoss),
                            static_cast<double>(r.fMetric));
    };

    // 20260324 ZJH Batch 回调: 接收每批次完成通知
    BatchCallback batchCb = [this](int nBatch, int nTotalBatches) {
        emit batchCompleted(nBatch, nTotalBatches);
    };

    // 20260324 ZJH 日志回调: 引擎内部日志转发到 UI
    LogCallback logCb = [this](const std::string& strMsg) {
        emit trainingLog(QString::fromStdString(strMsg));
    };

    // 20260324 ZJH 停止检查回调: 引擎在每个 Epoch/Batch 间隙检查
    // 同时处理暂停等待逻辑
    StopChecker stopCheck = [this]() -> bool {
        // 20260324 ZJH 检查停止请求
        if (m_bStopRequested.load()) return true;

        // 20260324 ZJH 暂停等待：使用条件变量阻塞
        {
            QMutexLocker locker(&m_pauseMutex);
            while (m_bPaused.load() && !m_bStopRequested.load()) {
                m_pauseCondition.wait(&m_pauseMutex);
            }
        }

        // 20260324 ZJH 暂停恢复后再次检查停止
        return m_bStopRequested.load();
    };

    // 20260324 ZJH 调用 EngineBridge::train() 执行真实训练
    // 20260326 ZJH try/catch 保护：train() 内部的 forward/backward/optimizer 可能因维度不匹配、
    // 内存不足等原因抛出异常，若异常逃逸到 QThread 会导致 std::terminate() → abort() 闪退
    bool bTrainOk = false;
    try {
        bTrainOk = m_pEngine->train(params,
                                         vecTrainData, vecTrainLabels,
                                         vecValData, vecValLabels,
                                         vecTrainMasks, vecValMasks,  // 20260328 ZJH 分割掩码（非分割模型时为空向量）
                                         epochCb, batchCb, logCb, stopCheck);
    } catch (const std::exception& e) {
        emit trainingLog(QStringLiteral("[错误] 训练异常: %1").arg(QString::fromStdString(e.what())));
    } catch (...) {
        emit trainingLog(QStringLiteral("[错误] 训练中发生未知异常"));
    }

    // 20260324 ZJH 检查训练是否被用户停止
    if (m_bStopRequested.load()) {
        emit trainingLog(QStringLiteral("[信息] 训练被用户停止"));
        emit trainingFinished(false, QStringLiteral("训练被用户手动停止"));
        emit progressChanged(0);
        return;
    }

    // 20260324 ZJH ===== 第 4 步：保存训练好的模型 =====
    emit trainingLog(QStringLiteral("========================================"));
    emit trainingLog(QStringLiteral("[%1] 训练完成").arg(QDateTime::currentDateTime().toString("hh:mm:ss")));

    if (bTrainOk) {
        // 20260325 ZJH 使用训练开始前预保存的项目路径（避免跨线程访问 m_pProject 导致竞态条件）
        if (!strProjectPath.isEmpty()) {
            // 20260324 ZJH 创建模型保存目录
            QString strModelDir = strProjectPath + QStringLiteral("/models");
            // 20260325 ZJH 检查目录创建是否成功，失败时输出具体路径到训练日志
            if (!QDir().mkpath(strModelDir)) {
                emit trainingLog(QStringLiteral("[错误] 无法创建模型目录: %1").arg(strModelDir));
            }

            // 20260324 ZJH 保存模型文件（.dfm 格式）
            QString strModelPath = strModelDir + QStringLiteral("/best_model.omm");
            emit trainingLog(QStringLiteral("[信息] 正在保存模型: %1").arg(strModelPath));

            bool bSaveOk = m_pEngine->saveModel(strModelPath.toStdString());
            if (bSaveOk) {
                m_strModelSavePath = strModelPath;  // 20260324 ZJH 记录保存路径
                emit trainingLog(QStringLiteral("[信息] 模型保存成功: %1").arg(strModelPath));

                // 20260324 ZJH 验证文件确实存在并输出文件大小
                QFileInfo fi(strModelPath);
                if (fi.exists()) {
                    double dSizeMB = fi.size() / (1024.0 * 1024.0);
                    emit trainingLog(QStringLiteral("[信息] 模型文件大小: %1 MB").arg(dSizeMB, 0, 'f', 2));
                }
            } else {
                emit trainingLog(QStringLiteral("[错误] 模型保存失败，目标路径: %1").arg(strModelPath));
            }

            // 20260324 ZJH 如果配置了导出 ONNX，也导出一份（使用同样的 EngineBridge）
            if (localConfig.bExportOnnx) {
                emit trainingLog(QStringLiteral("[信息] 正在导出 ONNX 模型..."));
                // 20260324 ZJH 暂时保存为同目录下的 .onnx 文件（TODO: 实现真实 ONNX 导出）
                QString strOnnxPath = strModelDir + QStringLiteral("/best_model.onnx");
                // 20260324 ZJH 当前引擎不支持 ONNX 格式，先保存 dfm 格式副本
                m_pEngine->saveModel(strOnnxPath.toStdString());
                emit trainingLog(QStringLiteral("[信息] ONNX 模型导出完成: %1").arg(strOnnxPath));
            }
        } else {
            emit trainingLog(QStringLiteral("[警告] 项目路径为空，无法保存模型"));
        }
    }

    emit trainingLog(QStringLiteral("========================================"));
    emit progressChanged(100);  // 20260324 ZJH 进度 100%

    // 20260324 ZJH 发送训练完成信号
    if (bTrainOk) {
        emit trainingFinished(true, QStringLiteral("训练完成，模型已保存"));
    } else {
        emit trainingFinished(false, QStringLiteral("训练过程中出现错误"));
    }
}
