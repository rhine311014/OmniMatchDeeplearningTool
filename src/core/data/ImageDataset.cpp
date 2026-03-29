// 20260322 ZJH ImageDataset 完整实现
// 提供数据集的图像管理、标签管理、拆分、统计和导入功能

#include "core/data/ImageDataset.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>           // 20260324 ZJH QFile::copy / QFile::exists（图像复制到项目目录）
#include <QFileInfo>
#include <QImageReader>
#include <QSet>
#include <QRandomGenerator>
#include <QDebug>
#include <QtConcurrent>    // 20260324 ZJH QtConcurrent::blockingMapped 并行文件拷贝
#include <algorithm>

// 20260324 ZJH 支持的图像文件裸扩展名集合（统一数据源）
// importFromFolder 和 importFiles 均从此集合派生，消除重复定义
const QSet<QString>& ImageDataset::supportedExtensions()
{
    // 20260324 ZJH 静态局部变量，首次调用时初始化，后续直接返回引用
    static const QSet<QString> s_setExtensions = {
        QStringLiteral("png"),   // 20260324 ZJH PNG 格式
        QStringLiteral("jpg"),   // 20260324 ZJH JPEG 格式
        QStringLiteral("jpeg"),  // 20260324 ZJH JPEG 格式（长扩展名）
        QStringLiteral("bmp"),   // 20260324 ZJH BMP 格式
        QStringLiteral("tif"),   // 20260324 ZJH TIFF 格式
        QStringLiteral("tiff")   // 20260324 ZJH TIFF 格式（长扩展名）
    };
    return s_setExtensions;  // 20260324 ZJH 返回静态集合的常量引用
}

// 20260324 ZJH 将裸扩展名集合转为 glob 通配符过滤器列表（如 "*.png"）
// 供 QDirIterator 的 nameFilters 参数使用
QStringList ImageDataset::extensionGlobFilters()
{
    QStringList vecFilters;  // 20260324 ZJH 结果列表
    // 20260324 ZJH 遍历裸扩展名，前置 "*." 构造 glob 模式
    for (const QString& strExt : supportedExtensions()) {
        vecFilters.append(QStringLiteral("*.") + strExt);
    }
    return vecFilters;  // 20260324 ZJH 返回通配符过滤器列表
}

// 20260324 ZJH 根据 QImage::Format 推断通道数
// 将分散在 importFromFolder 和 importFiles 中的重复判断逻辑统一提取
int ImageDataset::channelsFromFormat(QImage::Format fmt)
{
    // 20260324 ZJH 灰度/索引色格式 -> 1 通道
    if (fmt == QImage::Format_Grayscale8 || fmt == QImage::Format_Grayscale16 ||
        fmt == QImage::Format_Mono || fmt == QImage::Format_MonoLSB ||
        fmt == QImage::Format_Indexed8) {
        return 1;
    }
    // 20260324 ZJH 带 Alpha 通道的格式 -> 4 通道
    if (fmt == QImage::Format_ARGB32 || fmt == QImage::Format_ARGB32_Premultiplied ||
        fmt == QImage::Format_RGBA8888 || fmt == QImage::Format_RGBA8888_Premultiplied ||
        fmt == QImage::Format_RGBA64 || fmt == QImage::Format_RGBA64_Premultiplied) {
        return 4;
    }
    // 20260324 ZJH 其他格式默认 RGB -> 3 通道
    return 3;
}

// 20260322 ZJH 构造函数
ImageDataset::ImageDataset(QObject* pParent)
    : QObject(pParent)         // 20260322 ZJH 初始化 QObject 基类
    , m_nNextLabelId(0)        // 20260322 ZJH 标签 ID 从 0 开始自增
{
}

// 20260322 ZJH 析构函数
ImageDataset::~ImageDataset() = default;

// 20260324 ZJH 设置项目路径，导入图像时会将文件复制到该路径下的 images/ 子目录
void ImageDataset::setProjectPath(const QString& strPath)
{
    m_strProjectPath = strPath;  // 20260324 ZJH 记录项目根目录路径
}

// 20260324 ZJH 获取当前项目路径
QString ImageDataset::projectPath() const
{
    return m_strProjectPath;  // 20260324 ZJH 返回项目根目录路径
}

// 20260324 ZJH 获取磁盘缩略图缓存目录路径
// 返回 <项目路径>/thumbnails/，首次调用时自动创建目录
// 缩略图磁盘缓存使应用重启后无需从原图重新生成缩略图，显著加速画廊页面首屏渲染
QString ImageDataset::thumbnailCachePath() const
{
    // 20260324 ZJH 项目路径为空时无法确定缓存目录
    if (m_strProjectPath.isEmpty()) {
        return QString();  // 20260324 ZJH 返回空字符串表示缓存不可用
    }

    // 20260324 ZJH 构建缩略图缓存目录路径
    QString strThumbDir = m_strProjectPath + QStringLiteral("/thumbnails");

    // 20260324 ZJH 确保目录存在（首次调用时创建）
    QDir dir(strThumbDir);
    if (!dir.exists()) {
        dir.mkpath(QStringLiteral("."));  // 20260324 ZJH 创建目录（含所有父目录）
    }

    return strThumbDir;  // 20260324 ZJH 返回缩略图缓存目录绝对路径
}

// 20260324 ZJH 将源图像文件复制到项目 images/ 子目录
// 如果 m_strProjectPath 为空则不执行复制（向后兼容）
// 如果同名文件已存在则追加 _1、_2 等后缀避免覆盖
// 参数: strSourcePath - 源图像文件绝对路径
//       strSubDir     - images/ 下的子目录名（可为空，用于保留文件夹分类结构）
// 返回: 复制成功返回目标文件绝对路径，失败返回空字符串
QString ImageDataset::copyImageToProject(const QString& strSourcePath, const QString& strSubDir)
{
    // 20260324 ZJH 项目路径为空时跳过复制，保持原始路径
    if (m_strProjectPath.isEmpty()) {
        return QString();  // 20260324 ZJH 无项目上下文，返回空表示未复制
    }

    // 20260324 ZJH 构建目标目录路径：<项目路径>/images/ 或 <项目路径>/images/<子目录>/
    QString strDestDir = m_strProjectPath + QStringLiteral("/images");  // 20260324 ZJH 基础 images 目录
    if (!strSubDir.isEmpty()) {
        strDestDir += QStringLiteral("/") + strSubDir;  // 20260324 ZJH 追加子目录（如类别文件夹名）
    }

    // 20260324 ZJH 确保目标目录存在（含所有父目录）
    QDir destDir(strDestDir);  // 20260324 ZJH 目标目录对象
    if (!destDir.exists()) {
        if (!destDir.mkpath(QStringLiteral("."))) {
            qDebug() << "[ImageDataset] copyImageToProject: 无法创建目标目录" << strDestDir;  // 20260324 ZJH 创建目录失败日志
            return QString();  // 20260324 ZJH 目录创建失败，返回空
        }
    }

    // 20260324 ZJH 提取源文件名信息
    QFileInfo srcFileInfo(strSourcePath);  // 20260324 ZJH 源文件信息
    QString strBaseName = srcFileInfo.completeBaseName();  // 20260324 ZJH 不含扩展名的文件名（如 "img001"）
    QString strSuffix   = srcFileInfo.suffix();             // 20260324 ZJH 文件扩展名（如 "png"）

    // 20260324 ZJH 构建目标文件路径
    QString strDestPath = strDestDir + QStringLiteral("/") + strBaseName + QStringLiteral(".") + strSuffix;

    // 20260324 ZJH 检查源文件与目标路径是否相同（已在项目目录中则跳过复制）
    QFileInfo destFileInfo(strDestPath);  // 20260324 ZJH 目标文件信息
    if (QFileInfo(strSourcePath).absoluteFilePath() == destFileInfo.absoluteFilePath()) {
        return strDestPath;  // 20260324 ZJH 源文件已在目标位置，无需复制
    }

    // 20260324 ZJH 如果目标文件已存在（同名冲突），追加递增后缀避免覆盖
    if (QFile::exists(strDestPath)) {
        int nSuffix = 1;  // 20260324 ZJH 后缀递增起始值
        QString strNewDestPath;  // 20260324 ZJH 带后缀的新目标路径
        do {
            // 20260324 ZJH 构造带后缀的文件名，如 "img001_1.png"、"img001_2.png"
            strNewDestPath = strDestDir + QStringLiteral("/") + strBaseName
                + QStringLiteral("_") + QString::number(nSuffix) + QStringLiteral(".") + strSuffix;
            ++nSuffix;  // 20260324 ZJH 递增后缀
        } while (QFile::exists(strNewDestPath));  // 20260324 ZJH 循环直到找到不冲突的文件名
        strDestPath = strNewDestPath;  // 20260324 ZJH 使用不冲突的路径
    }

    // 20260324 ZJH 执行文件复制
    bool bCopyOk = QFile::copy(strSourcePath, strDestPath);  // 20260324 ZJH 复制文件
    if (!bCopyOk) {
        qDebug() << "[ImageDataset] copyImageToProject: 复制失败" << strSourcePath << "->" << strDestPath;  // 20260324 ZJH 复制失败日志
        return QString();  // 20260324 ZJH 复制失败，返回空
    }

    return strDestPath;  // 20260324 ZJH 返回复制后的目标文件绝对路径
}

// 20260324 ZJH 从 m_vecImages 完全重建 UUID 到索引的哈希映射
// 用于批量移除后索引偏移需要全量刷新的场景
void ImageDataset::rebuildUuidIndex()
{
    m_mapUuidToIndex.clear();  // 20260324 ZJH 清空旧映射
    m_mapUuidToIndex.reserve(m_vecImages.size());  // 20260324 ZJH 预分配哈希桶减少 rehash
    // 20260324 ZJH 遍历全部图像条目，建立 UUID -> 索引映射
    for (int i = 0; i < m_vecImages.size(); ++i) {
        m_mapUuidToIndex.insert(m_vecImages[i].strUuid, i);
    }
}

// ===== 图像管理 =====

// 20260322 ZJH 添加单张图像到数据集末尾
void ImageDataset::addImage(const ImageEntry& entry)
{
    // 20260324 ZJH 新图像索引 = 当前列表末尾位置
    int nNewIndex = m_vecImages.size();
    m_vecImages.append(entry);  // 20260322 ZJH 拷贝图像条目到列表末尾
    // 20260324 ZJH 更新 UUID 索引映射
    m_mapUuidToIndex.insert(entry.strUuid, nNewIndex);
    emit imagesAdded(1);        // 20260322 ZJH 通知外部：添加了 1 张图像
    emit dataChanged();         // 20260322 ZJH 通知外部：数据发生变更
}

// 20260322 ZJH 批量添加多张图像
void ImageDataset::addImages(const QVector<ImageEntry>& vecEntries)
{
    // 20260322 ZJH 空列表时不做任何操作，避免发出无意义信号
    if (vecEntries.isEmpty()) {
        return;  // 20260322 ZJH 无图像需添加，直接返回
    }
    // 20260324 ZJH 记录追加起始索引，用于后续更新 UUID 索引
    int nStartIndex = m_vecImages.size();
    m_vecImages.append(vecEntries);                  // 20260322 ZJH 批量追加到列表末尾
    // 20260324 ZJH 逐条更新 UUID 索引映射（比 rebuildUuidIndex 高效，仅处理新增部分）
    for (int i = 0; i < vecEntries.size(); ++i) {
        m_mapUuidToIndex.insert(vecEntries[i].strUuid, nStartIndex + i);
    }
    emit imagesAdded(static_cast<int>(vecEntries.size()));  // 20260322 ZJH 通知添加数量
    emit dataChanged();                              // 20260322 ZJH 通知数据变更
}

// 20260322 ZJH 按 UUID 移除单张图像
void ImageDataset::removeImage(const QString& strUuid)
{
    // 20260324 ZJH 使用 UUID 索引 O(1) 查找，取代线性扫描
    auto it = m_mapUuidToIndex.find(strUuid);
    if (it == m_mapUuidToIndex.end()) {
        return;  // 20260322 ZJH 未找到匹配 UUID，静默忽略
    }

    int nIndex = it.value();  // 20260324 ZJH 获取图像在向量中的索引
    m_vecImages.removeAt(nIndex);  // 20260322 ZJH 移除匹配项
    // 20260324 ZJH 移除后索引发生偏移，全量重建哈希映射
    rebuildUuidIndex();
    emit imagesRemoved(1);    // 20260322 ZJH 通知移除 1 张
    emit dataChanged();       // 20260322 ZJH 通知数据变更
}

// 20260322 ZJH 批量按 UUID 移除多张图像
void ImageDataset::removeImages(const QVector<QString>& vecUuids)
{
    // 20260322 ZJH 空列表时不做任何操作
    if (vecUuids.isEmpty()) {
        return;  // 20260322 ZJH 无图像需移除
    }

    // 20260322 ZJH 将待删除 UUID 存入 QSet，提升查找效率（O(1) vs O(n)）
    QSet<QString> setUuids(vecUuids.begin(), vecUuids.end());

    int nRemoved = 0;  // 20260322 ZJH 实际移除计数

    // 20260322 ZJH 从后往前遍历，避免删除元素后索引偏移
    for (int i = m_vecImages.size() - 1; i >= 0; --i) {
        if (setUuids.contains(m_vecImages[i].strUuid)) {
            m_vecImages.removeAt(i);  // 20260322 ZJH 移除匹配项
            ++nRemoved;               // 20260322 ZJH 计数 +1
        }
    }

    // 20260322 ZJH 有实际移除时才发信号
    if (nRemoved > 0) {
        // 20260324 ZJH 批量移除后全量重建 UUID 索引（多处索引偏移，逐条更新不划算）
        rebuildUuidIndex();
        emit imagesRemoved(nRemoved);  // 20260322 ZJH 通知移除数量
        emit dataChanged();            // 20260322 ZJH 通知数据变更
    }
}

// 20260322 ZJH 按 UUID 查找图像（可修改版）
// 20260324 ZJH 改用 m_mapUuidToIndex 哈希映射实现 O(1) 查找
ImageEntry* ImageDataset::findImage(const QString& strUuid)
{
    auto it = m_mapUuidToIndex.find(strUuid);  // 20260324 ZJH O(1) 哈希查找
    if (it != m_mapUuidToIndex.end()) {
        return &m_vecImages[it.value()];  // 20260324 ZJH 通过索引直接定位
    }
    return nullptr;  // 20260322 ZJH 未找到，返回空指针
}

// 20260322 ZJH 按 UUID 查找图像（只读版）
// 20260324 ZJH 改用 m_mapUuidToIndex 哈希映射实现 O(1) 查找
const ImageEntry* ImageDataset::findImage(const QString& strUuid) const
{
    auto it = m_mapUuidToIndex.find(strUuid);  // 20260324 ZJH O(1) 哈希查找
    if (it != m_mapUuidToIndex.end()) {
        return &m_vecImages[it.value()];  // 20260324 ZJH 通过索引直接定位
    }
    return nullptr;  // 20260322 ZJH 未找到
}

// 20260324 ZJH 按 UUID 查找图像在向量中的索引
// 使用 m_mapUuidToIndex 实现 O(1) 查找
int ImageDataset::findImageIndex(const QString& strUuid) const
{
    auto it = m_mapUuidToIndex.find(strUuid);  // 20260324 ZJH O(1) 哈希查找
    if (it != m_mapUuidToIndex.end()) {
        return it.value();  // 20260324 ZJH 返回向量索引
    }
    return -1;  // 20260324 ZJH 未找到，返回 -1
}

// 20260322 ZJH 获取图像总数
int ImageDataset::imageCount() const
{
    return static_cast<int>(m_vecImages.size());  // 20260322 ZJH 返回向量大小
}

// 20260322 ZJH 获取全部图像条目的常量引用
const QVector<ImageEntry>& ImageDataset::images() const
{
    return m_vecImages;  // 20260322 ZJH 返回内部向量的常量引用
}

// ===== 标签管理 =====

// 20260322 ZJH 添加新标签到数据集
// 自动分配递增 ID，忽略传入的 nId
void ImageDataset::addLabel(const LabelInfo& label)
{
    LabelInfo newLabel = label;             // 20260322 ZJH 拷贝传入的标签信息
    newLabel.nId = m_nNextLabelId;          // 20260322 ZJH 分配自增 ID
    ++m_nNextLabelId;                       // 20260322 ZJH 递增下一个可用 ID

    // 20260322 ZJH 如果颜色无效（未设置），自动分配预设颜色
    if (!newLabel.color.isValid()) {
        QVector<QColor> vecColors = defaultLabelColors();  // 20260322 ZJH 获取预设颜色列表
        // 20260322 ZJH 按 ID 取模循环分配颜色
        int nColorIndex = newLabel.nId % vecColors.size();
        newLabel.color = vecColors[nColorIndex];  // 20260322 ZJH 分配对应颜色
    }

    m_vecLabels.append(newLabel);  // 20260322 ZJH 添加到标签列表
    emit labelsChanged();          // 20260322 ZJH 通知标签变更
    emit dataChanged();            // 20260322 ZJH 通知数据变更
}

// 20260323 ZJH 直接插入标签保留原始 ID（反序列化专用，不触发信号）
void ImageDataset::insertLabelDirect(const LabelInfo& label)
{
    m_vecLabels.append(label);  // 20260323 ZJH 直接追加，保留传入的 nId
}

// 20260323 ZJH 设置下一个标签自增 ID
void ImageDataset::setNextLabelId(int nNextId)
{
    m_nNextLabelId = nNextId;  // 20260323 ZJH 恢复自增计数器
}

// 20260324 ZJH 检查指定标签是否被任何图像条目或标注对象引用
bool ImageDataset::isLabelInUse(int nId) const
{
    // 20260324 ZJH 遍历所有图像，检查图像级标签和标注级标签
    for (const auto& entry : m_vecImages) {
        // 20260324 ZJH 检查图像级标签（分类/异常检测任务）
        if (entry.nLabelId == nId) {
            return true;  // 20260324 ZJH 该标签被图像级标签引用
        }
        // 20260324 ZJH 检查标注级标签（检测/分割任务）
        for (const auto& annotation : entry.vecAnnotations) {
            if (annotation.nLabelId == nId) {
                return true;  // 20260324 ZJH 该标签被某条标注引用
            }
        }
    }
    return false;  // 20260324 ZJH 无任何引用
}

// 20260322 ZJH 按 ID 移除标签
// 20260324 ZJH 若标签仍被引用，级联将引用该标签的 nLabelId 置为 -1
void ImageDataset::removeLabel(int nId)
{
    // 20260322 ZJH 遍历查找匹配 ID 的标签
    for (int i = 0; i < m_vecLabels.size(); ++i) {
        if (m_vecLabels[i].nId == nId) {
            // 20260324 ZJH 孤儿防护：若标签仍被引用，级联清除引用
            if (isLabelInUse(nId)) {
                qDebug("ImageDataset::removeLabel: label %d is in use, cascading nLabelId to -1", nId);
                // 20260324 ZJH 遍历所有图像，将引用该标签的 nLabelId 置为 -1（未分配）
                for (auto& entry : m_vecImages) {
                    // 20260324 ZJH 清除图像级标签引用
                    if (entry.nLabelId == nId) {
                        entry.nLabelId = -1;
                    }
                    // 20260324 ZJH 清除标注级标签引用
                    for (auto& annotation : entry.vecAnnotations) {
                        if (annotation.nLabelId == nId) {
                            annotation.nLabelId = -1;
                        }
                    }
                }
            }
            m_vecLabels.removeAt(i);  // 20260322 ZJH 移除匹配项
            emit labelsChanged();      // 20260322 ZJH 通知标签变更
            emit dataChanged();        // 20260322 ZJH 通知数据变更
            return;                    // 20260322 ZJH ID 唯一，找到即返回
        }
    }
    // 20260322 ZJH 未找到匹配 ID，静默忽略
}

// 20260322 ZJH 更新已有标签信息
void ImageDataset::updateLabel(const LabelInfo& label)
{
    // 20260322 ZJH 按 nId 查找并更新
    for (int i = 0; i < m_vecLabels.size(); ++i) {
        if (m_vecLabels[i].nId == label.nId) {
            m_vecLabels[i] = label;  // 20260322 ZJH 整体替换标签信息
            emit labelsChanged();     // 20260322 ZJH 通知标签变更
            emit dataChanged();       // 20260322 ZJH 通知数据变更
            return;                   // 20260322 ZJH 找到即返回
        }
    }
    // 20260322 ZJH 未找到匹配 ID，静默忽略
}

// 20260322 ZJH 获取全部标签列表
const QVector<LabelInfo>& ImageDataset::labels() const
{
    return m_vecLabels;  // 20260322 ZJH 返回内部标签向量的常量引用
}

// 20260322 ZJH 按 ID 查找标签
LabelInfo* ImageDataset::findLabel(int nId)
{
    // 20260322 ZJH 线性查找匹配 ID 的标签
    for (int i = 0; i < m_vecLabels.size(); ++i) {
        if (m_vecLabels[i].nId == nId) {
            return &m_vecLabels[i];  // 20260322 ZJH 返回匹配项指针
        }
    }
    return nullptr;  // 20260322 ZJH 未找到
}

// ===== 数据集拆分 =====

// 20260322 ZJH 手动设置单张图像的拆分类型
void ImageDataset::assignSplit(const QString& strUuid, om::SplitType eSplit)
{
    // 20260322 ZJH 查找目标图像
    ImageEntry* pEntry = findImage(strUuid);
    if (pEntry != nullptr) {
        pEntry->eSplit = eSplit;  // 20260322 ZJH 设置拆分类型
        emit splitChanged();      // 20260322 ZJH 通知拆分变更
        emit dataChanged();       // 20260322 ZJH 通知数据变更
    }
    // 20260322 ZJH 未找到图像时静默忽略
}

// 20260322 ZJH 自动拆分数据集
// fTrainRatio + fValRatio <= 1.0，剩余部分为测试集
// bStratified=true 时按标签分布均匀分配，保证各子集标签比例一致
// 20260324 ZJH 新增 nSeed 参数：>= 0 时使用指定种子实现可复现拆分，< 0 时使用全局随机数
void ImageDataset::autoSplit(float fTrainRatio, float fValRatio, bool bStratified, int nSeed)
{
    // 20260323 ZJH 参数验证
    if (fTrainRatio < 0.0f) fTrainRatio = 0.0f;
    if (fValRatio < 0.0f) fValRatio = 0.0f;
    if (fTrainRatio + fValRatio > 1.0f) {
        // 20260323 ZJH 超过 100% 时按比例缩放
        float fSum = fTrainRatio + fValRatio;
        fTrainRatio /= fSum;
        fValRatio /= fSum;
    }

    // 20260322 ZJH 无图像时直接返回
    if (m_vecImages.isEmpty()) {
        return;
    }

    // 20260324 ZJH 根据 nSeed 参数选择随机数生成器
    // nSeed >= 0 时使用指定种子创建本地生成器，确保可复现拆分
    // nSeed < 0 时使用全局随机数生成器（原有行为）
    QRandomGenerator localRng;                         // 20260324 ZJH 本地随机数生成器（种子模式时使用）
    QRandomGenerator* pRng = nullptr;                  // 20260324 ZJH 实际使用的随机数生成器指针
    if (nSeed >= 0) {
        localRng = QRandomGenerator(static_cast<quint32>(nSeed));  // 20260324 ZJH 使用指定种子初始化
        pRng = &localRng;                              // 20260324 ZJH 指向本地生成器
    } else {
        pRng = QRandomGenerator::global();             // 20260324 ZJH 使用全局随机数生成器
    }

    if (bStratified) {
        // 20260328 ZJH 分层采样模式：按标签分组后分别拆分
        // 分割场景特殊处理：图像级标签(nLabelId)可能全为 -1，
        // 此时按"有无标注(vecAnnotations)"分为两组，确保有标注的图优先进训练集/验证集
        QMap<int, QVector<int>> mapLabelToIndices;  // 20260322 ZJH labelId -> 图像索引列表

        // 20260328 ZJH 检测是否为分割场景（所有图像级标签均为-1，但有标注存在）
        bool bAllUnlabeled = true;   // 20260328 ZJH 是否所有图像级标签都为 -1
        bool bHasAnyAnno = false;    // 20260328 ZJH 是否有任何图像带标注
        for (int i = 0; i < m_vecImages.size(); ++i) {
            if (m_vecImages[i].nLabelId >= 0) bAllUnlabeled = false;
            if (!m_vecImages[i].vecAnnotations.isEmpty()) bHasAnyAnno = true;
        }

        // 20260328 ZJH 智能分组策略：优先按标签分组，无标签时按标注分组
        // 目的：有标注/标签的图像优先分配到训练集和验证集
        // 适用所有模型类型（分类/分割/检测），不仅限于分割
        for (int i = 0; i < m_vecImages.size(); ++i) {
            int nGroupKey;
            if (m_vecImages[i].nLabelId >= 0) {
                // 20260328 ZJH 有图像级标签: 按标签分组（分类模型标准路径）
                nGroupKey = m_vecImages[i].nLabelId;
            } else if (!m_vecImages[i].vecAnnotations.isEmpty()) {
                // 20260328 ZJH 无图像标签但有标注: 归为"有标注"组(key=9999)
                // 确保分割/检测项目的已标注图像集中分配到训练集和验证集
                nGroupKey = 9999;
            } else {
                // 20260328 ZJH 无标签也无标注: 归为"无标注"组(key=-1)
                // 作为纯背景样本，按比例分配到各子集
                nGroupKey = -1;
            }
            mapLabelToIndices[nGroupKey].append(i);
        }

        // 20260322 ZJH 对每个标签组独立进行随机拆分
        for (auto it = mapLabelToIndices.begin(); it != mapLabelToIndices.end(); ++it) {
            QVector<int>& vecIndices = it.value();  // 20260322 ZJH 该标签组的图像索引列表

            // 20260322 ZJH 随机打乱索引顺序（Fisher-Yates 洗牌）
            for (int i = vecIndices.size() - 1; i > 0; --i) {
                int j = static_cast<int>(pRng->bounded(i + 1));  // 20260322 ZJH 生成 [0, i] 随机数
                std::swap(vecIndices[i], vecIndices[j]);          // 20260322 ZJH 交换元素
            }

            int nTotal = vecIndices.size();  // 20260322 ZJH 该组图像总数
            // 20260322 ZJH 计算训练集和验证集的边界索引
            int nTrainEnd = static_cast<int>(nTotal * fTrainRatio);               // 20260322 ZJH 训练集结束位置
            int nValEnd   = nTrainEnd + static_cast<int>(nTotal * fValRatio);     // 20260322 ZJH 验证集结束位置

            // 20260324 ZJH 小组保护：确保每个分组至少分配 1 张图像到每个子集
            if (nTotal >= 3) {
                // 20260324 ZJH 3 张及以上时，保证 train/val/test 各至少 1 张
                if (nTrainEnd < 1) nTrainEnd = 1;                          // 20260324 ZJH 训练集至少 1 张
                if (nValEnd <= nTrainEnd) nValEnd = nTrainEnd + 1;         // 20260324 ZJH 验证集至少 1 张
                if (nValEnd >= nTotal) nValEnd = nTotal - 1;               // 20260324 ZJH 测试集至少 1 张
                if (nTrainEnd >= nValEnd) nTrainEnd = nValEnd - 1;         // 20260324 ZJH 再次校正训练集边界
            } else if (nTotal == 2) {
                // 20260324 ZJH 仅 2 张：分配 1 张 train + 1 张 val，无 test
                nTrainEnd = 1;
                nValEnd   = 2;
            } else if (nTotal == 1) {
                // 20260324 ZJH 仅 1 张：全部分配到 train
                nTrainEnd = 1;
                nValEnd   = 1;
            }

            // 20260322 ZJH 分配拆分类型
            for (int i = 0; i < nTotal; ++i) {
                int nIdx = vecIndices[i];  // 20260322 ZJH 取出实际图像索引
                if (i < nTrainEnd) {
                    m_vecImages[nIdx].eSplit = om::SplitType::Train;       // 20260322 ZJH 训练集
                } else if (i < nValEnd) {
                    m_vecImages[nIdx].eSplit = om::SplitType::Validation;  // 20260322 ZJH 验证集
                } else {
                    m_vecImages[nIdx].eSplit = om::SplitType::Test;        // 20260322 ZJH 测试集
                }
            }
        }
    } else {
        // 20260322 ZJH 非分层模式：全部图像混合随机拆分
        // 生成索引列表并随机打乱
        QVector<int> vecIndices(m_vecImages.size());  // 20260322 ZJH 索引列表
        for (int i = 0; i < vecIndices.size(); ++i) {
            vecIndices[i] = i;  // 20260322 ZJH 填充顺序索引
        }

        // 20260322 ZJH 随机打乱（Fisher-Yates 洗牌）
        for (int i = vecIndices.size() - 1; i > 0; --i) {
            int j = static_cast<int>(pRng->bounded(i + 1));  // 20260322 ZJH 随机位置
            std::swap(vecIndices[i], vecIndices[j]);          // 20260322 ZJH 交换
        }

        int nTotal    = vecIndices.size();                                      // 20260322 ZJH 图像总数
        int nTrainEnd = static_cast<int>(nTotal * fTrainRatio);                 // 20260322 ZJH 训练集边界
        int nValEnd   = nTrainEnd + static_cast<int>(nTotal * fValRatio);       // 20260322 ZJH 验证集边界

        // 20260324 ZJH 小组保护：确保每个分组至少分配 1 张图像到每个子集
        if (nTotal >= 3) {
            // 20260324 ZJH 3 张及以上时，保证 train/val/test 各至少 1 张
            if (nTrainEnd < 1) nTrainEnd = 1;                          // 20260324 ZJH 训练集至少 1 张
            if (nValEnd <= nTrainEnd) nValEnd = nTrainEnd + 1;         // 20260324 ZJH 验证集至少 1 张
            if (nValEnd >= nTotal) nValEnd = nTotal - 1;               // 20260324 ZJH 测试集至少 1 张
            if (nTrainEnd >= nValEnd) nTrainEnd = nValEnd - 1;         // 20260324 ZJH 再次校正训练集边界
        } else if (nTotal == 2) {
            // 20260324 ZJH 仅 2 张：分配 1 张 train + 1 张 val，无 test
            nTrainEnd = 1;
            nValEnd   = 2;
        } else if (nTotal == 1) {
            // 20260324 ZJH 仅 1 张：全部分配到 train
            nTrainEnd = 1;
            nValEnd   = 1;
        }

        // 20260322 ZJH 分配拆分类型
        for (int i = 0; i < nTotal; ++i) {
            int nIdx = vecIndices[i];  // 20260322 ZJH 取出实际图像索引
            if (i < nTrainEnd) {
                m_vecImages[nIdx].eSplit = om::SplitType::Train;       // 20260322 ZJH 训练集
            } else if (i < nValEnd) {
                m_vecImages[nIdx].eSplit = om::SplitType::Validation;  // 20260322 ZJH 验证集
            } else {
                m_vecImages[nIdx].eSplit = om::SplitType::Test;        // 20260322 ZJH 测试集
            }
        }
    }

    emit splitChanged();  // 20260322 ZJH 通知拆分变更
    emit dataChanged();   // 20260322 ZJH 通知数据变更
}

// 20260322 ZJH 统计指定拆分类型的图像数量
int ImageDataset::countBySplit(om::SplitType eSplit) const
{
    int nCount = 0;  // 20260322 ZJH 计数器
    // 20260322 ZJH 遍历所有图像，统计匹配的拆分类型
    for (const auto& entry : m_vecImages) {
        if (entry.eSplit == eSplit) {
            ++nCount;  // 20260322 ZJH 匹配则计数 +1
        }
    }
    return nCount;  // 20260322 ZJH 返回统计结果
}

// ===== 统计 =====

// 20260322 ZJH 获取已标注的图像数量
int ImageDataset::labeledCount() const
{
    int nCount = 0;  // 20260322 ZJH 计数器
    // 20260322 ZJH 遍历所有图像，调用 isLabeled() 判断
    for (const auto& entry : m_vecImages) {
        if (entry.isLabeled()) {
            ++nCount;  // 20260322 ZJH 已标注则计数 +1
        }
    }
    return nCount;  // 20260322 ZJH 返回已标注数量
}

// 20260322 ZJH 获取未标注的图像数量
int ImageDataset::unlabeledCount() const
{
    // 20260322 ZJH 总数减去已标注数
    return imageCount() - labeledCount();
}

// 20260322 ZJH 获取标签分布统计（labelId -> 使用次数）
QMap<int, int> ImageDataset::labelDistribution() const
{
    QMap<int, int> mapDistribution;  // 20260322 ZJH 标签 ID -> 使用计数

    // 20260322 ZJH 遍历所有图像，统计图像级标签
    for (const auto& entry : m_vecImages) {
        if (entry.nLabelId >= 0) {
            // 20260322 ZJH 有图像级标签，累加计数
            mapDistribution[entry.nLabelId]++;
        }
        // 20260322 ZJH 同时统计对象级标注中的标签
        for (const auto& annotation : entry.vecAnnotations) {
            if (annotation.nLabelId >= 0) {
                mapDistribution[annotation.nLabelId]++;  // 20260322 ZJH 累加标注标签计数
            }
        }
    }

    return mapDistribution;  // 20260322 ZJH 返回分布统计
}

// ===== 导入 =====

// 20260324 ZJH 从文件夹导入图像 — 三阶段并行流水线
// 阶段 1（串行）：扫描目录收集文件列表 + 创建标签映射
// 阶段 2（并行）：QtConcurrent 多线程文件拷贝到项目目录
// 阶段 3（串行）：构建 ImageEntry + 标签赋值 + 批量入库
void ImageDataset::importFromFolder(const QString& strFolderPath, bool bRecursive)
{
    // 20260322 ZJH 验证文件夹路径有效性
    QDir dir(strFolderPath);
    if (!dir.exists()) {
        return;  // 20260322 ZJH 文件夹不存在，直接返回
    }

    // ===== 阶段 1: 串行扫描目录，收集文件列表 =====

    // 20260322 ZJH 配置目录迭代器标志
    QDirIterator::IteratorFlags flags = QDirIterator::NoIteratorFlags;  // 20260322 ZJH 默认不递归
    if (bRecursive) {
        flags = QDirIterator::Subdirectories;  // 20260322 ZJH 启用递归扫描
    }

    // 20260322 ZJH 创建目录迭代器，仅筛选文件，按图像扩展名过滤
    QDirIterator it(strFolderPath, extensionGlobFilters(), QDir::Files, flags);

    // 20260324 ZJH 扫描结果：源路径 + 子目录名（用于并行拷贝）
    struct ScanItem {
        QString strSourcePath;   // 源文件绝对路径
        QString strSubDir;       // 目标子目录名（根目录图像为空）
        QString strParentName;   // 父文件夹名（用于标签映射）
        bool bInRootDir;         // 是否在导入根目录
    };
    QVector<ScanItem> vecScanItems;

    // 20260322 ZJH 遍历所有匹配的图像文件
    while (it.hasNext()) {
        QString strFilePath = it.next();  // 20260322 ZJH 获取下一个匹配文件的路径
        QFileInfo fileInfo(strFilePath);  // 20260322 ZJH 构造文件信息对象

        // 20260322 ZJH 检查图像所在子文件夹名，用作自动标签
        QString strParentName = fileInfo.dir().dirName();  // 20260322 ZJH 获取父目录名
        bool bInRootDir = (fileInfo.dir().absolutePath() == dir.absolutePath());

        ScanItem item;
        item.strSourcePath = fileInfo.absoluteFilePath();
        item.strParentName = strParentName;
        item.bInRootDir = bInRootDir;
        // 20260324 ZJH 确定子目录名用于复制时保留文件夹结构
        item.strSubDir = (!bInRootDir && !strParentName.isEmpty()) ? strParentName : QString();
        vecScanItems.append(item);
    }

    if (vecScanItems.isEmpty()) {
        return;  // 20260324 ZJH 无匹配图像
    }

    // ===== 阶段 2: 多线程并行文件拷贝 =====

    // 20260324 ZJH 预创建所有需要的子目录（串行，因为 mkpath 有竞态风险）
    if (!m_strProjectPath.isEmpty()) {
        QSet<QString> setSubDirs;
        for (const ScanItem& item : vecScanItems) {
            setSubDirs.insert(item.strSubDir);  // 20260324 ZJH 收集去重的子目录名
        }
        for (const QString& strSub : setSubDirs) {
            QString strDestDir = m_strProjectPath + QStringLiteral("/images");
            if (!strSub.isEmpty()) {
                strDestDir += QStringLiteral("/") + strSub;
            }
            QDir(strDestDir).mkpath(QStringLiteral("."));  // 20260324 ZJH 预创建目录
        }
    }

    // 20260324 ZJH 并行拷贝：每个文件独立拷贝到目标目录，线程安全
    // QtConcurrent::blockingMapped 利用全部 CPU 核心并行执行 I/O 操作
    // 返回每个文件的拷贝结果（目标路径 + 文件大小）
    struct CopyResult {
        QString strDestPath;   // 拷贝后的目标绝对路径（空表示未拷贝/失败）
        qint64 nFileSize;      // 文件大小
    };

    // 20260324 ZJH 捕获项目路径到局部变量（lambda 中不能使用 this->成员 跨线程）
    QString strProjectPath = m_strProjectPath;

    QVector<CopyResult> vecCopyResults = QtConcurrent::blockingMapped(
        vecScanItems,
        [strProjectPath](const ScanItem& item) -> CopyResult {
            CopyResult result;

            // 20260324 ZJH 项目路径为空时不拷贝（向后兼容）
            if (strProjectPath.isEmpty()) {
                result.strDestPath = QString();
                result.nFileSize = QFileInfo(item.strSourcePath).size();
                return result;
            }

            // 20260324 ZJH 构建目标路径
            QString strDestDir = strProjectPath + QStringLiteral("/images");
            if (!item.strSubDir.isEmpty()) {
                strDestDir += QStringLiteral("/") + item.strSubDir;
            }

            QFileInfo srcInfo(item.strSourcePath);
            QString strBaseName = srcInfo.completeBaseName();
            QString strSuffix = srcInfo.suffix();
            QString strDestPath = strDestDir + QStringLiteral("/") + strBaseName + QStringLiteral(".") + strSuffix;

            // 20260324 ZJH 源文件已在目标位置则跳过
            if (QFileInfo(item.strSourcePath).absoluteFilePath() == QFileInfo(strDestPath).absoluteFilePath()) {
                result.strDestPath = strDestPath;
                result.nFileSize = srcInfo.size();
                return result;
            }

            // 20260324 ZJH 同名文件冲突处理：追加 _1, _2 ... 后缀
            if (QFile::exists(strDestPath)) {
                int nSuffix = 1;
                QString strNewPath;
                do {
                    strNewPath = strDestDir + QStringLiteral("/") + strBaseName
                               + QStringLiteral("_") + QString::number(nSuffix)
                               + QStringLiteral(".") + strSuffix;
                    nSuffix++;
                } while (QFile::exists(strNewPath));
                strDestPath = strNewPath;
            }

            // 20260324 ZJH 执行文件拷贝
            if (QFile::copy(item.strSourcePath, strDestPath)) {
                result.strDestPath = strDestPath;
                result.nFileSize = QFileInfo(strDestPath).size();
            } else {
                // 20260324 ZJH 拷贝失败，保留原始路径
                result.strDestPath = QString();
                result.nFileSize = srcInfo.size();
            }
            return result;
        }
    );

    // ===== 阶段 3: 串行构建 ImageEntry + 标签赋值 + 批量入库 =====

    // 20260322 ZJH 存储已创建的子目录标签映射（子目录名 -> 标签 ID）
    QMap<QString, int> mapFolderToLabel;

    // 20260322 ZJH 收集待添加的图像条目
    QVector<ImageEntry> vecNewEntries;
    vecNewEntries.reserve(vecScanItems.size());  // 20260324 ZJH 预分配避免扩容

    for (int i = 0; i < vecScanItems.size(); ++i) {
        const ScanItem& scanItem = vecScanItems[i];
        const CopyResult& copyResult = vecCopyResults[i];

        // 20260322 ZJH 创建图像条目
        ImageEntry entry;

        // 20260324 ZJH 根据拷贝结果设置路径
        if (!copyResult.strDestPath.isEmpty()) {
            entry.strFilePath = copyResult.strDestPath;
            QDir projDir(m_strProjectPath);
            entry.strRelativePath = projDir.relativeFilePath(copyResult.strDestPath);
        } else {
            entry.strFilePath = scanItem.strSourcePath;
            entry.strRelativePath = dir.relativeFilePath(scanItem.strSourcePath);
        }

        entry.nFileSize = copyResult.nFileSize;

        // 20260324 ZJH 延迟元数据读取优化：nWidth/nHeight/nChannels 保持 0
        // 首次显示时由 ImagePage::onAsyncImageLoaded 回填

        if (!scanItem.bInRootDir && !scanItem.strParentName.isEmpty()) {
            // 20260322 ZJH 图像在子文件夹中，需要创建或查找对应标签
            if (!mapFolderToLabel.contains(scanItem.strParentName)) {
                // 20260322 ZJH 该子文件夹尚未创建标签，新建一个
                LabelInfo newLabel;
                newLabel.strName = scanItem.strParentName;  // 20260322 ZJH 标签名 = 文件夹名

                // 20260322 ZJH 检查标签是否已存在于数据集中（避免重复）
                bool bExists = false;
                for (const auto& existingLabel : m_vecLabels) {
                    if (existingLabel.strName == scanItem.strParentName) {
                        mapFolderToLabel[scanItem.strParentName] = existingLabel.nId;
                        bExists = true;
                        break;
                    }
                }

                if (!bExists) {
                    // 20260322 ZJH 标签不存在，创建新标签
                    addLabel(newLabel);  // 20260322 ZJH addLabel 内部自动分配 ID 和颜色
                    mapFolderToLabel[scanItem.strParentName] = m_nNextLabelId - 1;
                }
            }

            // 20260322 ZJH 将标签 ID 赋值给图像
            entry.nLabelId = mapFolderToLabel[scanItem.strParentName];
        }

        vecNewEntries.append(entry);  // 20260322 ZJH 添加到待插入列表
    }

    // 20260322 ZJH 批量添加收集到的图像条目
    if (!vecNewEntries.isEmpty()) {
        addImages(vecNewEntries);  // 20260322 ZJH 调用批量添加（内部触发信号）
    }
}

// 20260324 ZJH 从文件路径列表导入图像 — 并行拷贝版本
// 阶段 1（串行）：过滤有效文件
// 阶段 2（并行）：QtConcurrent 多线程文件拷贝
// 阶段 3（串行）：构建 ImageEntry + 批量入库
void ImageDataset::importFiles(const QStringList& vecPaths)
{
    // 20260322 ZJH 空列表直接返回
    if (vecPaths.isEmpty()) {
        return;
    }

    // ===== 阶段 1: 串行过滤有效文件 =====
    const QSet<QString>& setExtensions = supportedExtensions();
    QStringList vecValidPaths;  // 20260324 ZJH 经过过滤的有效文件路径

    for (const QString& strPath : vecPaths) {
        QFileInfo fileInfo(strPath);
        if (!fileInfo.exists() || !fileInfo.isFile()) {
            continue;
        }
        if (!setExtensions.contains(fileInfo.suffix().toLower())) {
            continue;
        }
        vecValidPaths.append(fileInfo.absoluteFilePath());
    }

    if (vecValidPaths.isEmpty()) {
        return;
    }

    // ===== 阶段 2: 并行文件拷贝 =====

    // 20260324 ZJH 预创建 images/ 目录
    if (!m_strProjectPath.isEmpty()) {
        QDir(m_strProjectPath + QStringLiteral("/images")).mkpath(QStringLiteral("."));
    }

    struct CopyResult {
        QString strDestPath;
        qint64 nFileSize;
    };

    QString strProjectPath = m_strProjectPath;

    QVector<CopyResult> vecCopyResults = QtConcurrent::blockingMapped(
        vecValidPaths,
        [strProjectPath](const QString& strSourcePath) -> CopyResult {
            CopyResult result;

            if (strProjectPath.isEmpty()) {
                result.strDestPath = QString();
                result.nFileSize = QFileInfo(strSourcePath).size();
                return result;
            }

            QString strDestDir = strProjectPath + QStringLiteral("/images");
            QFileInfo srcInfo(strSourcePath);
            QString strBaseName = srcInfo.completeBaseName();
            QString strSuffix = srcInfo.suffix();
            QString strDestPath = strDestDir + QStringLiteral("/") + strBaseName + QStringLiteral(".") + strSuffix;

            // 20260324 ZJH 已在目标位置则跳过
            if (QFileInfo(strSourcePath).absoluteFilePath() == QFileInfo(strDestPath).absoluteFilePath()) {
                result.strDestPath = strDestPath;
                result.nFileSize = srcInfo.size();
                return result;
            }

            // 20260324 ZJH 同名文件冲突
            if (QFile::exists(strDestPath)) {
                int nSuffix = 1;
                QString strNewPath;
                do {
                    strNewPath = strDestDir + QStringLiteral("/") + strBaseName
                               + QStringLiteral("_") + QString::number(nSuffix)
                               + QStringLiteral(".") + strSuffix;
                    nSuffix++;
                } while (QFile::exists(strNewPath));
                strDestPath = strNewPath;
            }

            if (QFile::copy(strSourcePath, strDestPath)) {
                result.strDestPath = strDestPath;
                result.nFileSize = QFileInfo(strDestPath).size();
            } else {
                result.strDestPath = QString();
                result.nFileSize = srcInfo.size();
            }
            return result;
        }
    );

    // ===== 阶段 3: 串行构建 ImageEntry + 批量入库 =====
    QVector<ImageEntry> vecNewEntries;
    vecNewEntries.reserve(vecValidPaths.size());

    for (int i = 0; i < vecValidPaths.size(); ++i) {
        const CopyResult& copyResult = vecCopyResults[i];

        ImageEntry entry;
        if (!copyResult.strDestPath.isEmpty()) {
            entry.strFilePath = copyResult.strDestPath;
            QDir projDir(m_strProjectPath);
            entry.strRelativePath = projDir.relativeFilePath(copyResult.strDestPath);
        } else {
            entry.strFilePath = vecValidPaths[i];
        }
        entry.nFileSize = copyResult.nFileSize;
        // 20260324 ZJH 延迟元数据：nWidth/nHeight/nChannels 保持 0

        vecNewEntries.append(entry);
    }

    if (!vecNewEntries.isEmpty()) {
        addImages(vecNewEntries);
    }
}
