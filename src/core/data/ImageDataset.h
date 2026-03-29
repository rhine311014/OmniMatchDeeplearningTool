// 20260322 ZJH ImageDataset — 数据集管理类
// 管理项目中的全部图像条目和标签定义
// 提供图像增删查、标签管理、数据集拆分、统计分析和文件夹导入功能
// 继承 QObject 以支持信号槽通知机制
#pragma once

#include <QObject>
#include <QVector>
#include <QMap>
#include <QHash>
#include <QStringList>
#include <QSet>
#include <QImage>

#include "core/DLTypes.h"
#include "core/data/LabelInfo.h"
#include "core/data/ImageEntry.h"

// 20260322 ZJH 数据集管理类
// 持有图像列表和标签列表，提供完整的数据管理 API
class ImageDataset : public QObject
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父 QObject（用于 Qt 对象树内存管理）
    explicit ImageDataset(QObject* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~ImageDataset() override;

    // ===== 图像管理 =====

    // 20260322 ZJH 添加单张图像到数据集
    // 参数: entry - 图像条目（按值传入，内部拷贝存储）
    void addImage(const ImageEntry& entry);

    // 20260322 ZJH 批量添加多张图像到数据集
    // 参数: vecEntries - 图像条目列表
    void addImages(const QVector<ImageEntry>& vecEntries);

    // 20260322 ZJH 按 UUID 移除单张图像
    // 参数: strUuid - 要移除的图像 UUID
    void removeImage(const QString& strUuid);

    // 20260322 ZJH 批量按 UUID 移除多张图像
    // 参数: vecUuids - 要移除的图像 UUID 列表
    void removeImages(const QVector<QString>& vecUuids);

    // 20260322 ZJH 按 UUID 查找图像（可修改版）
    // 参数: strUuid - 目标图像 UUID
    // 返回: 指向图像条目的指针，未找到返回 nullptr
    ImageEntry* findImage(const QString& strUuid);

    // 20260322 ZJH 按 UUID 查找图像（只读版）
    // 参数: strUuid - 目标图像 UUID
    // 返回: 指向图像条目的常量指针，未找到返回 nullptr
    const ImageEntry* findImage(const QString& strUuid) const;

    // 20260324 ZJH 按 UUID 查找图像在向量中的索引
    // 参数: strUuid - 目标图像 UUID
    // 返回: 图像在 m_vecImages 中的索引，未找到返回 -1
    int findImageIndex(const QString& strUuid) const;

    // 20260322 ZJH 获取数据集中的图像总数
    // 返回: 图像条目总数
    int imageCount() const;

    // 20260322 ZJH 获取全部图像条目的常量引用
    // 返回: 图像条目向量的常量引用（用于遍历/显示）
    const QVector<ImageEntry>& images() const;

    // ===== 标签管理 =====

    // 20260322 ZJH 添加新标签到数据集
    // 参数: label - 标签信息（nId 会被自动分配，传入值被忽略）
    void addLabel(const LabelInfo& label);

    // 20260323 ZJH 直接插入标签保留原始 ID（仅限反序列化使用）
    // 参数: label - 标签信息（保留 nId 不变）
    void insertLabelDirect(const LabelInfo& label);

    // 20260323 ZJH 设置下一个标签自增 ID（反序列化恢复后调用）
    void setNextLabelId(int nNextId);

    // 20260324 ZJH 检查指定标签是否被任何图像或标注引用
    // 参数: nId - 要检查的标签 ID
    // 返回: true 表示该标签仍被至少一个 ImageEntry 或 Annotation 引用
    bool isLabelInUse(int nId) const;

    // 20260322 ZJH 按 ID 移除标签
    // 参数: nId - 要移除的标签 ID
    // 20260324 ZJH 若标签仍被引用，级联将引用该标签的 nLabelId 置为 -1（未分配）
    void removeLabel(int nId);

    // 20260322 ZJH 更新已有标签信息（按 nId 匹配）
    // 参数: label - 新的标签信息（nId 必须已存在）
    void updateLabel(const LabelInfo& label);

    // 20260322 ZJH 获取全部标签列表的常量引用
    // 返回: 标签信息向量的常量引用
    const QVector<LabelInfo>& labels() const;

    // 20260322 ZJH 按 ID 查找标签（可修改版）
    // 参数: nId - 目标标签 ID
    // 返回: 指向标签信息的指针，未找到返回 nullptr
    LabelInfo* findLabel(int nId);

    // ===== 数据集拆分 =====

    // 20260322 ZJH 手动为单张图像设置拆分类型
    // 参数: strUuid - 图像 UUID; eSplit - 目标拆分类型
    void assignSplit(const QString& strUuid, om::SplitType eSplit);

    // 20260322 ZJH 自动拆分数据集为训练/验证/测试集
    // 参数: fTrainRatio - 训练集比例（0.0~1.0）
    //       fValRatio   - 验证集比例（0.0~1.0），剩余为测试集
    //       bStratified  - 是否分层采样（true 时按标签分布均匀分配）
    //       nSeed        - 随机种子（>= 0 时使用指定种子实现可复现拆分，< 0 时使用全局随机数生成器）
    void autoSplit(float fTrainRatio, float fValRatio, bool bStratified = true, int nSeed = -1);

    // 20260322 ZJH 统计指定拆分类型的图像数量
    // 参数: eSplit - 拆分类型
    // 返回: 该类型的图像数量
    int countBySplit(om::SplitType eSplit) const;

    // ===== 统计 =====

    // 20260322 ZJH 获取已标注的图像数量
    // 返回: 已标注图像数量（isLabeled() == true）
    int labeledCount() const;

    // 20260322 ZJH 获取未标注的图像数量
    // 返回: 未标注图像数量（isLabeled() == false）
    int unlabeledCount() const;

    // 20260322 ZJH 获取标签分布统计
    // 返回: 映射表 labelId -> 使用该标签的图像数量
    QMap<int, int> labelDistribution() const;

    // ===== 导入 =====

    // 20260322 ZJH 从文件夹导入图像
    // 递归扫描 png/jpg/jpeg/bmp/tif/tiff 文件
    // 子文件夹名作为类别名自动创建标签并赋值给该文件夹下的图像
    // 参数: strFolderPath - 文件夹绝对路径
    //       bRecursive     - 是否递归扫描子目录（默认 true）
    void importFromFolder(const QString& strFolderPath, bool bRecursive = true);

    // 20260322 ZJH 从文件路径列表导入图像
    // 不自动创建标签，仅加载图像元数据
    // 参数: vecPaths - 图像文件绝对路径列表
    void importFiles(const QStringList& vecPaths);

signals:
    // 20260322 ZJH 数据集内容发生变更（通用信号，任何修改都会触发）
    void dataChanged();

    // 20260322 ZJH 图像添加完成信号
    // 参数: nCount - 本次添加的图像数量
    void imagesAdded(int nCount);

    // 20260322 ZJH 图像移除完成信号
    // 参数: nCount - 本次移除的图像数量
    void imagesRemoved(int nCount);

    // 20260322 ZJH 标签列表发生变更信号
    void labelsChanged();

    // 20260322 ZJH 数据集拆分发生变更信号
    void splitChanged();

public:
    // 20260324 ZJH 支持的图像文件裸扩展名集合（统一数据源，不含通配符前缀）
    // importFromFolder 和 importFiles 共用此集合，避免扩展名列表重复定义
    static const QSet<QString>& supportedExtensions();

    // 20260324 ZJH 将 supportedExtensions() 转换为 glob 通配符过滤器列表（如 "*.png"）
    // 用于 QDirIterator 过滤
    static QStringList extensionGlobFilters();

    // 20260324 ZJH 根据 QImage::Format 推断图像通道数
    // 参数: fmt - QImageReader 读取到的图像格式
    // 返回: 通道数（1=灰度, 3=RGB, 4=RGBA）
    static int channelsFromFormat(QImage::Format fmt);

    // 20260324 ZJH 设置项目路径，导入图像时会将文件复制到该路径下的 images/ 子目录
    // 参数: strPath - 项目根目录绝对路径
    void setProjectPath(const QString& strPath);

    // 20260324 ZJH 获取当前项目路径
    // 返回: 项目根目录绝对路径，未设置时返回空字符串
    QString projectPath() const;

    // 20260324 ZJH 获取磁盘缩略图缓存目录路径
    // 返回: <项目路径>/thumbnails/，未设置项目路径时返回空字符串
    // 首次调用时自动创建目录
    QString thumbnailCachePath() const;

private:
    QVector<ImageEntry> m_vecImages;   // 20260322 ZJH 全部图像条目列表
    QVector<LabelInfo>  m_vecLabels;   // 20260322 ZJH 全部标签信息列表
    int m_nNextLabelId = 0;            // 20260322 ZJH 下一个可用的标签 ID（自增）

    // 20260324 ZJH 项目根目录路径（用于将导入图像复制到项目内 images/ 子目录）
    // 为空时导入行为退化为仅记录原始路径（向后兼容）
    QString m_strProjectPath;

    // 20260324 ZJH UUID 到向量索引的哈希映射，提供 O(1) 查找
    // 在 addImage/addImages/removeImage/removeImages/clear 中同步更新
    QHash<QString, int> m_mapUuidToIndex;

    // 20260324 ZJH 从 m_vecImages 完全重建 UUID 索引
    // 用于批量移除后索引偏移需要全量刷新的场景
    void rebuildUuidIndex();

    // 20260324 ZJH 将源图像文件复制到项目 images/ 子目录
    // 参数: strSourcePath - 源图像文件绝对路径
    //       strSubDir     - images/ 下的子目录名（可为空，用于保留文件夹结构）
    // 返回: 复制成功时返回目标文件绝对路径，失败时返回空字符串
    QString copyImageToProject(const QString& strSourcePath, const QString& strSubDir = QString());
};
