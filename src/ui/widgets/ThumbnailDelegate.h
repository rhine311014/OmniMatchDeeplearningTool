// 20260322 ZJH ThumbnailDelegate — 缩略图网格项自定义绘制代理
// 用于 GalleryPage 的 QListView，以图标模式展示图像缩略图
// 功能：图像加载+缓存、文件名截断、标签色块、拆分标记、选中高亮
// 20260324 ZJH 优化：增大 QPixmapCache 限制，缓存未命中时绘制占位符并异步加载
#pragma once

#include <QStyledItemDelegate>  // 20260322 ZJH 自定义绘制代理基类
#include <QSet>                 // 20260324 ZJH 跟踪正在异步加载的缓存键
#include <QPointer>             // 20260324 ZJH 线程安全弱引用，防止异步回调访问已销毁对象

// 20260322 ZJH 缩略图模型的自定义数据角色
// 约定 QStandardItem 中通过 setData(value, role) 存储各字段
enum ThumbnailRoles {
    FilePathRole   = Qt::UserRole + 1,  // 20260322 ZJH 图像文件绝对路径 (QString)
    UuidRole       = Qt::UserRole + 2,  // 20260322 ZJH 图像唯一标识 UUID (QString)
    LabelIdRole    = Qt::UserRole + 3,  // 20260322 ZJH 标签 ID (int, -1 表示未标注)
    LabelColorRole = Qt::UserRole + 4,  // 20260322 ZJH 标签颜色 (QColor)
    LabelNameRole  = Qt::UserRole + 5,  // 20260322 ZJH 标签名称 (QString)
    SplitTypeRole  = Qt::UserRole + 6,  // 20260322 ZJH 拆分类型 (int, 对应 om::SplitType)
    FileNameRole   = Qt::UserRole + 7,  // 20260322 ZJH 文件名 (QString, 不含路径)
    ThumbDirRole   = Qt::UserRole + 100  // 20260324 ZJH 缩略图磁盘缓存目录路径 (QString)
};

// 20260322 ZJH 缩略图绘制代理
// 在 QListView(IconMode) 中以网格形式绘制：
//   - 中央：等比缩放的图像缩略图（QPixmapCache 缓存）
//   - 下方：文件名文本（超长时截断加省略号）
//   - 左上角：标签颜色圆角矩形（仅在有标签时显示）
//   - 右下角：拆分类型标记文字（T/V/E）
//   - 选中时：蓝色边框高亮
class ThumbnailDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    // 20260322 ZJH 构造函数
    // 参数: pParent - 父对象（通常为 QListView）
    explicit ThumbnailDelegate(QObject* pParent = nullptr);

    // 20260322 ZJH 析构函数
    ~ThumbnailDelegate() override = default;

    // 20260322 ZJH 自定义绘制方法，绘制缩略图网格项的全部内容
    // 参数: pPainter - 绘图器; option - 绘制选项（含位置/大小/状态）; index - 模型索引
    void paint(QPainter* pPainter,
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override;

    // 20260322 ZJH 返回每个网格项的推荐大小
    // 大小 = 缩略图尺寸 + 文件名文本行高 + 上下边距
    // 参数: option - 样式选项; index - 模型索引
    // 返回: 推荐的控件大小
    QSize sizeHint(const QStyleOptionViewItem& option,
                   const QModelIndex& index) const override;

    // 20260322 ZJH 设置缩略图目标大小（正方形边长，像素）
    // 参数: nSize - 缩略图边长（80~320）
    void setThumbnailSize(int nSize);

    // 20260322 ZJH 获取当前缩略图大小
    // 返回: 缩略图边长（像素）
    int thumbnailSize() const;

private:
    // 20260324 ZJH 异步加载缩略图：先检查磁盘缓存，未命中则从原图加载并保存到磁盘缓存
    // 提交到 QThreadPool 后台线程加载，完成后插入内存缓存并刷新视图
    // 参数: strFilePath     - 图像文件路径
    //       strCacheKey     - 内存缓存键
    //       nThumbSize      - 目标缩略图尺寸
    //       strUuid         - 图像 UUID（用于磁盘缓存文件名）
    //       strThumbDir     - 磁盘缩略图缓存目录路径（为空时跳过磁盘缓存）
    void asyncLoadThumbnail(const QString& strFilePath,
                            const QString& strCacheKey,
                            int nThumbSize,
                            const QString& strUuid,
                            const QString& strThumbDir) const;

    // 20260322 ZJH 当前缩略图目标边长（默认 160px）
    int m_nThumbSize = 160;

    // 20260322 ZJH 文件名文本区域高度（固定 20px）
    static constexpr int s_nTextHeight = 20;

    // 20260322 ZJH 网格项四周内边距（像素）
    static constexpr int s_nPadding = 6;

    // 20260322 ZJH 标签色块的宽高（像素）
    static constexpr int s_nLabelBadgeSize = 14;

    // 20260322 ZJH 拆分标记文字的字体大小（像素）
    static constexpr int s_nSplitFontSize = 10;

    // 20260324 ZJH 正在后台加载中的缓存键集合（mutable 因为 paint 是 const 方法）
    // 防止同一张图片被多次提交到线程池
    mutable QSet<QString> m_setPendingLoads;
};
