// 20260322 ZJH ThumbnailDelegate 实现
// 自定义绘制缩略图网格项：图像+文件名+标签色块+拆分标记+选中高亮
// 20260324 ZJH 优化：增大 QPixmapCache 至 200MB，缓存未命中时绘制占位符并异步加载

#include "ui/widgets/ThumbnailDelegate.h"  // 20260322 ZJH 类声明

#include <QPainter>          // 20260322 ZJH 绘图器
#include <QPixmap>           // 20260322 ZJH 位图
#include <QPixmapCache>      // 20260322 ZJH 全局像素图缓存
#include <QApplication>      // 20260322 ZJH 应用程序实例（获取调色板）
#include <QStyle>            // 20260322 ZJH 样式接口
#include <QFileInfo>         // 20260322 ZJH 文件信息（文件名提取）
#include <QFontMetrics>      // 20260322 ZJH 字体度量（文本截断计算）
#include <QImageReader>      // 20260322 ZJH 图像读取器（高效预缩放加载）
#include <QFile>             // 20260324 ZJH 磁盘缓存文件存在检查
#include <QThreadPool>       // 20260324 ZJH 线程池，用于异步加载缩略图
#include <QAbstractItemView> // 20260324 ZJH 获取父视图用于触发 viewport update
#include <QTimer>            // 20260324 ZJH 延迟触发视图更新
#include <algorithm>         // 20260404 ZJH std::min（标注轮廓缩放）
#include <cmath>             // 20260404 ZJH std::max（裁剪区域计算）

// 20260322 ZJH 构造函数
ThumbnailDelegate::ThumbnailDelegate(QObject* pParent)
    : QStyledItemDelegate(pParent)  // 20260322 ZJH 调用基类构造
{
    // 20260324 ZJH 增大 QPixmapCache 限制到 200MB（默认 10MB 远远不够数百张缩略图）
    // 每张 160x160 ARGB32 缩略图约 100KB，200MB 可缓存约 2000 张
    QPixmapCache::setCacheLimit(200 * 1024);  // 20260324 ZJH 参数单位为 KB
}

// 20260322 ZJH 自定义绘制：缩略图 + 文件名 + 标签色块 + 拆分标记 + 选中高亮
void ThumbnailDelegate::paint(QPainter* pPainter,
                              const QStyleOptionViewItem& option,
                              const QModelIndex& index) const
{
    // 20260322 ZJH 保存绘图器状态，函数结束时恢复
    pPainter->save();

    // 20260322 ZJH 启用抗锯齿，使圆角和文字更平滑
    pPainter->setRenderHint(QPainter::Antialiasing, true);
    pPainter->setRenderHint(QPainter::SmoothPixmapTransform, true);

    // 20260322 ZJH 计算网格项的内部区域（去除内边距）
    QRect rcItem = option.rect;  // 20260322 ZJH 完整网格项矩形
    QRect rcInner = rcItem.adjusted(s_nPadding, s_nPadding, -s_nPadding, -s_nPadding);  // 20260322 ZJH 去除四周内边距后的区域

    // 20260322 ZJH 1. 绘制背景和选中高亮边框
    bool bSelected = option.state & QStyle::State_Selected;  // 20260322 ZJH 判断是否被选中
    bool bHovered  = option.state & QStyle::State_MouseOver; // 20260322 ZJH 判断是否鼠标悬停

    if (bSelected) {
        // 20260322 ZJH 选中时绘制蓝色半透明背景 + 蓝色边框
        pPainter->setPen(QPen(QColor(37, 99, 235), 2));  // 20260322 ZJH 2px 蓝色边框
        pPainter->setBrush(QColor(37, 99, 235, 30));     // 20260322 ZJH 半透明蓝色背景
        pPainter->drawRoundedRect(rcItem.adjusted(1, 1, -1, -1), 4, 4);  // 20260322 ZJH 圆角矩形
    } else if (bHovered) {
        // 20260322 ZJH 悬停时绘制浅灰色半透明背景
        pPainter->setPen(QPen(QColor(100, 116, 139, 100), 1));  // 20260322 ZJH 1px 灰色边框
        pPainter->setBrush(QColor(100, 116, 139, 20));          // 20260322 ZJH 半透明灰色背景
        pPainter->drawRoundedRect(rcItem.adjusted(1, 1, -1, -1), 4, 4);
    }

    // 20260322 ZJH 2. 计算缩略图绘制区域（内部区域上方，正方形）
    int nThumbAreaWidth  = rcInner.width();    // 20260322 ZJH 缩略图区域可用宽度
    int nThumbAreaHeight = rcInner.height() - s_nTextHeight;  // 20260322 ZJH 减去底部文件名区域
    QRect rcThumbArea(rcInner.left(), rcInner.top(), nThumbAreaWidth, nThumbAreaHeight);

    // 20260322 ZJH 3. 加载缩略图（使用 QPixmapCache 缓存）
    QString strFilePath = index.data(FilePathRole).toString();  // 20260322 ZJH 从模型获取文件路径

    // 20260402 ZJH 检查是否为实例视图裁剪模式（Halcon 风格）
    QRectF rectCrop = index.data(CropRectRole).toRectF();  // 20260402 ZJH 裁剪矩形（空 = 整图）
    bool bCropMode = rectCrop.isValid() && !rectCrop.isEmpty();  // 20260402 ZJH 是否需要裁剪

    // 20260402 ZJH 缓存键: 裁剪模式包含裁剪坐标，确保不同实例有不同缓存
    QString strCacheKey;
    if (bCropMode) {
        strCacheKey = QStringLiteral("crop_%1_%2_%3_%4_%5_%6")
            .arg(strFilePath)
            .arg(static_cast<int>(rectCrop.x())).arg(static_cast<int>(rectCrop.y()))
            .arg(static_cast<int>(rectCrop.width())).arg(static_cast<int>(rectCrop.height()))
            .arg(m_nThumbSize);
    } else {
        strCacheKey = QStringLiteral("thumb_%1_%2")
            .arg(strFilePath)
            .arg(m_nThumbSize);  // 20260322 ZJH 缓存键包含路径和大小
    }

    QPixmap pixThumb;  // 20260322 ZJH 缩略图位图
    if (!QPixmapCache::find(strCacheKey, &pixThumb)) {
        // 20260324 ZJH 缓存未命中：绘制灰色占位符，并异步加载图像

        // 20260402 ZJH 裁剪模式: 同步加载裁剪区域（小区域快速加载不会阻塞）
        if (bCropMode && !strFilePath.isEmpty()) {
            QImage imgFull(strFilePath);  // 20260402 ZJH 加载原图
            if (!imgFull.isNull()) {
                // 20260402 ZJH 裁剪标注区域（带 10% padding 防止边缘截断）
                int nPadX = static_cast<int>(rectCrop.width() * 0.1);   // 20260402 ZJH 10% 水平 padding
                int nPadY = static_cast<int>(rectCrop.height() * 0.1);  // 20260402 ZJH 10% 垂直 padding
                QRect rcCropInt(
                    std::max(0, static_cast<int>(rectCrop.x()) - nPadX),
                    std::max(0, static_cast<int>(rectCrop.y()) - nPadY),
                    static_cast<int>(rectCrop.width()) + 2 * nPadX,
                    static_cast<int>(rectCrop.height()) + 2 * nPadY
                );
                // 20260402 ZJH 裁剪到图像边界内
                rcCropInt = rcCropInt.intersected(imgFull.rect());
                QImage imgCropped = imgFull.copy(rcCropInt);  // 20260402 ZJH 裁剪

                // 20260402 ZJH 缩放到缩略图大小
                QImage imgScaled = imgCropped.scaled(m_nThumbSize, m_nThumbSize,
                    Qt::KeepAspectRatio, Qt::SmoothTransformation);
                pixThumb = QPixmap::fromImage(imgScaled);
                QPixmapCache::insert(strCacheKey, pixThumb);  // 20260402 ZJH 缓存
            }
        }

        if (pixThumb.isNull()) {
            // 20260324 ZJH 绘制占位矩形（裁剪模式加载失败或整图模式缓存未命中）
            pPainter->setPen(QPen(QColor(71, 85, 105), 1));
            pPainter->setBrush(QColor(30, 34, 48));
            pPainter->drawRect(rcThumbArea.adjusted(4, 4, -4, -4));

            pPainter->setPen(QColor(100, 116, 139));
            QFont fontPlaceholder = pPainter->font();
            fontPlaceholder.setPixelSize(11);
            pPainter->setFont(fontPlaceholder);
            pPainter->drawText(rcThumbArea, Qt::AlignCenter, QStringLiteral("..."));

            // 20260324 ZJH 整图模式异步加载
            if (!bCropMode && !strFilePath.isEmpty()) {
                QString strUuid     = index.data(UuidRole).toString();
                QString strThumbDir = index.data(ThumbDirRole).toString();
                asyncLoadThumbnail(strFilePath, strCacheKey, m_nThumbSize, strUuid, strThumbDir);
            }
        }
    }

    if (!pixThumb.isNull()) {
        // 20260322 ZJH 4. 绘制缩略图（居中于缩略图区域）
        int nOffsetX = rcThumbArea.left() + (nThumbAreaWidth - pixThumb.width()) / 2;
        int nOffsetY = rcThumbArea.top() + (nThumbAreaHeight - pixThumb.height()) / 2;
        pPainter->drawPixmap(nOffsetX, nOffsetY, pixThumb);  // 20260322 ZJH 绘制缩略图

        // 20260404 ZJH 实例视图: 绘制标注轮廓+半透明填充（对标 Halcon 检查页彩色轮廓叠加）
        if (bCropMode) {
            int nLabelIdForBorder = index.data(LabelIdRole).toInt();
            if (nLabelIdForBorder >= 0) {
                QColor clrBorder = index.data(LabelColorRole).value<QColor>();
                if (!clrBorder.isValid()) clrBorder = QColor(37, 99, 235);

                // 20260404 ZJH 尝试获取标注多边形轮廓
                QPolygonF polyAnnotation = index.data(AnnotationPolyRole).value<QPolygonF>();

                if (!polyAnnotation.isEmpty()) {
                    // 20260404 ZJH 将标注多边形从原图坐标变换到缩略图坐标
                    // 需要: 1) 减去裁剪区域偏移  2) 按缩放比例缩放  3) 平移到绘制位置
                    int nPadX = static_cast<int>(rectCrop.width() * 0.1);   // 20260404 ZJH 与裁剪 padding 一致
                    int nPadY = static_cast<int>(rectCrop.height() * 0.1);
                    // 20260404 ZJH 重建实际裁剪区域（与缓存加载时相同的计算）
                    double dCropX = std::max(0.0, rectCrop.x() - nPadX);
                    double dCropY = std::max(0.0, rectCrop.y() - nPadY);
                    double dCropW = rectCrop.width() + 2.0 * nPadX;
                    double dCropH = rectCrop.height() + 2.0 * nPadY;

                    // 20260404 ZJH 计算缩放比（保持宽高比，取较小的缩放因子）
                    double dScaleX = static_cast<double>(pixThumb.width()) / dCropW;
                    double dScaleY = static_cast<double>(pixThumb.height()) / dCropH;
                    double dScale = std::min(dScaleX, dScaleY);

                    // 20260404 ZJH 变换多边形顶点: (imgX, imgY) → (thumbX, thumbY)
                    QPolygonF polyTransformed;
                    for (const QPointF& pt : polyAnnotation) {
                        double tx = nOffsetX + (pt.x() - dCropX) * dScale;
                        double ty = nOffsetY + (pt.y() - dCropY) * dScale;
                        polyTransformed << QPointF(tx, ty);
                    }

                    // 20260404 ZJH 绘制半透明填充 + 轮廓线（Halcon 风格）
                    QColor clrFill = clrBorder;
                    clrFill.setAlpha(50);  // 20260404 ZJH 半透明填充（alpha=50，不遮挡图像细节）
                    pPainter->setPen(QPen(clrBorder, 1.5));  // 20260404 ZJH 1.5px 轮廓线
                    pPainter->setBrush(clrFill);
                    pPainter->drawPolygon(polyTransformed);  // 20260404 ZJH 绘制闭合多边形
                } else {
                    // 20260404 ZJH 无多边形数据时退化为简单矩形边框
                    pPainter->setPen(QPen(clrBorder, 2));
                    pPainter->setBrush(Qt::NoBrush);
                    pPainter->drawRect(nOffsetX, nOffsetY, pixThumb.width(), pixThumb.height());
                }
            }
        }
    }

    // 20260322 ZJH 5. 左上角标签色块（仅在有标签时绘制）
    int nLabelId = index.data(LabelIdRole).toInt();  // 20260322 ZJH 获取标签 ID
    if (nLabelId >= 0) {
        // 20260322 ZJH 获取标签颜色
        QColor clrLabel = index.data(LabelColorRole).value<QColor>();
        if (!clrLabel.isValid()) {
            clrLabel = QColor(37, 99, 235);  // 20260322 ZJH 默认蓝色
        }

        // 20260322 ZJH 绘制圆角色块矩形
        QRect rcBadge(rcThumbArea.left() + 4, rcThumbArea.top() + 4,
                      s_nLabelBadgeSize, s_nLabelBadgeSize);
        pPainter->setPen(Qt::NoPen);
        pPainter->setBrush(clrLabel);
        pPainter->drawRoundedRect(rcBadge, 3, 3);  // 20260322 ZJH 3px 圆角
    }

    // 20260322 ZJH 6. 右下角拆分标记文字（T/V/E）
    int nSplitType = index.data(SplitTypeRole).toInt();  // 20260322 ZJH 获取拆分类型
    // 20260322 ZJH 仅对 Train(0)/Validation(1)/Test(2) 绘制标记，Unassigned(3) 不绘制
    if (nSplitType >= 0 && nSplitType <= 2) {
        // 20260322 ZJH 拆分标记字符映射
        static const QString s_arrSplitChars[] = {
            QStringLiteral("T"),  // 20260322 ZJH Train
            QStringLiteral("V"),  // 20260322 ZJH Validation
            QStringLiteral("E")   // 20260322 ZJH tEst (用 E 表示 Test，避免与 Train 的 T 混淆)
        };
        // 20260322 ZJH 拆分标记颜色映射
        static const QColor s_arrSplitColors[] = {
            QColor(34, 197, 94),   // 20260322 ZJH Train: 绿色
            QColor(251, 191, 36),  // 20260322 ZJH Validation: 黄色
            QColor(99, 102, 241)   // 20260322 ZJH Test: 紫色
        };

        QString strSplitChar = s_arrSplitChars[nSplitType];  // 20260322 ZJH 拆分字符
        QColor clrSplit = s_arrSplitColors[nSplitType];       // 20260322 ZJH 拆分颜色

        // 20260322 ZJH 设置拆分标记字体
        QFont fontSplit = pPainter->font();
        fontSplit.setPixelSize(s_nSplitFontSize);  // 20260322 ZJH 10px 字体大小
        fontSplit.setBold(true);                    // 20260322 ZJH 粗体
        pPainter->setFont(fontSplit);

        // 20260322 ZJH 计算标记绘制位置（右下角偏移）
        QRect rcSplitBg(rcThumbArea.right() - 18, rcThumbArea.bottom() - 18, 16, 16);
        pPainter->setPen(Qt::NoPen);
        pPainter->setBrush(QColor(0, 0, 0, 160));  // 20260322 ZJH 半透明黑色背景
        pPainter->drawRoundedRect(rcSplitBg, 3, 3);

        pPainter->setPen(clrSplit);  // 20260322 ZJH 设置文字颜色
        pPainter->drawText(rcSplitBg, Qt::AlignCenter, strSplitChar);  // 20260322 ZJH 绘制标记文字
    }

    // 20260322 ZJH 7. 底部文件名文本（截断+省略号）
    QString strFileName = index.data(FileNameRole).toString();  // 20260322 ZJH 获取文件名
    if (strFileName.isEmpty()) {
        // 20260322 ZJH 备选：从文件路径提取文件名
        strFileName = QFileInfo(strFilePath).fileName();
    }

    // 20260322 ZJH 计算文件名文本绘制区域
    QRect rcText(rcInner.left(), rcInner.bottom() - s_nTextHeight,
                 rcInner.width(), s_nTextHeight);

    // 20260322 ZJH 设置文件名字体
    QFont fontName = pPainter->font();
    fontName.setPixelSize(11);  // 20260322 ZJH 11px 小字体
    fontName.setBold(false);
    pPainter->setFont(fontName);

    // 20260322 ZJH 使用 QFontMetrics 截断超长文件名
    QFontMetrics fm(fontName);
    QString strElided = fm.elidedText(strFileName, Qt::ElideMiddle, rcText.width());

    // 20260322 ZJH 设置文字颜色（选中时白色，未选中时浅灰）
    pPainter->setPen(bSelected ? QColor(255, 255, 255) : QColor(203, 213, 225));
    pPainter->drawText(rcText, Qt::AlignHCenter | Qt::AlignVCenter, strElided);

    // 20260322 ZJH 恢复绘图器状态
    pPainter->restore();
}

// 20260324 ZJH 异步加载缩略图：先检查磁盘缓存，未命中则从原图加载并保存到磁盘缓存
// 在 QThreadPool 后台线程中加载图像，完成后通过 QTimer::singleShot 回到主线程插入内存缓存并刷新视图
// 磁盘缓存使用 <项目路径>/thumbnails/<uuid>_<size>.jpg 格式，JPEG quality=85 平衡质量和速度
// 应用重启后从磁盘缓存加载（~1-5ms/张）远快于从原图生成（~20-100ms/张）
void ThumbnailDelegate::asyncLoadThumbnail(const QString& strFilePath,
                                           const QString& strCacheKey,
                                           int nThumbSize,
                                           const QString& strUuid,
                                           const QString& strThumbDir) const
{
    // 20260324 ZJH 检查是否已在加载队列中，避免重复提交
    if (m_setPendingLoads.contains(strCacheKey)) {
        return;  // 20260324 ZJH 已经在加载，跳过
    }

    m_setPendingLoads.insert(strCacheKey);  // 20260324 ZJH 标记为正在加载

    // 20260324 ZJH 使用 QPointer 弱引用代替裸指针，防止异步回调时 delegate 已销毁
    // QPointer 在对象销毁后自动置 nullptr，QTimer::singleShot 会检查 context 对象存活性
    QPointer<QObject> pWeakRef = const_cast<ThumbnailDelegate*>(this);

    // 20260324 ZJH 构建磁盘缓存文件路径（如果缩略图缓存目录可用）
    // 格式: <项目路径>/thumbnails/<uuid>_<size>.jpg
    QString strDiskCachePath;  // 20260324 ZJH 磁盘缓存路径（为空表示不使用磁盘缓存）
    if (!strThumbDir.isEmpty() && !strUuid.isEmpty()) {
        strDiskCachePath = strThumbDir + QStringLiteral("/") + strUuid
            + QStringLiteral("_") + QString::number(nThumbSize) + QStringLiteral(".jpg");
    }

    QThreadPool::globalInstance()->start([strFilePath, strCacheKey, nThumbSize, pWeakRef, strDiskCachePath]() {
        // 20260324 ZJH 此 lambda 在后台线程执行

        QImage imgThumb;  // 20260324 ZJH 缩略图结果

        // 20260324 ZJH 第一步：尝试从磁盘缓存加载（远快于解码原图）
        if (!strDiskCachePath.isEmpty() && QFile::exists(strDiskCachePath)) {
            imgThumb = QImage(strDiskCachePath);  // 20260324 ZJH 从磁盘缓存加载 JPEG 缩略图
            // 20260324 ZJH 磁盘缓存命中：无需解码原图，加载速度约 1-5ms
        }

        // 20260324 ZJH 第二步：磁盘缓存未命中或加载失败，从原图生成缩略图
        bool bNeedSaveToDisk = false;  // 20260324 ZJH 是否需要保存到磁盘缓存
        if (imgThumb.isNull()) {
            // 20260324 ZJH 从原图加载并预缩放
            QImageReader reader(strFilePath);
            reader.setAutoTransform(true);  // 20260324 ZJH 自动应用 EXIF 旋转
            // 20260324 ZJH 设置预缩放尺寸，让解码器在解码时直接缩放，大幅节省内存
            reader.setScaledSize(QSize(nThumbSize, nThumbSize));

            imgThumb = reader.read();  // 20260324 ZJH 读取图像（后台线程，不阻塞 UI）
            bNeedSaveToDisk = true;     // 20260324 ZJH 标记需要保存到磁盘缓存
        }

        // 20260324 ZJH 等比缩放到目标大小并转换为 QPixmap
        QPixmap pixResult;
        if (!imgThumb.isNull()) {
            // 20260324 ZJH 等比缩放（磁盘缓存加载的可能尺寸不完全匹配，统一缩放）
            QImage imgScaled = imgThumb.scaled(
                nThumbSize, nThumbSize,
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation);

            pixResult = QPixmap::fromImage(imgScaled);

            // 20260324 ZJH 第三步：将生成的缩略图保存到磁盘缓存（仅从原图新生成时保存）
            if (bNeedSaveToDisk && !strDiskCachePath.isEmpty()) {
                // 20260324 ZJH JPEG quality=85 平衡文件大小和图像质量
                // 160x160 缩略图约 5-15KB，1000 张约 5-15MB 磁盘占用
                imgScaled.save(strDiskCachePath, "JPEG", 85);  // 20260324 ZJH 保存到磁盘缓存
            }
        }

        // 20260324 ZJH 检查 delegate 是否仍然存活，避免向已销毁对象投递回调
        if (pWeakRef.isNull()) {
            return;  // 20260324 ZJH delegate 已销毁，放弃缓存插入（缓存是全局的，下次会重新加载）
        }

        // 20260324 ZJH 回到主线程：插入内存缓存并触发视图刷新
        // QTimer::singleShot 的 context 参数会在投递前再次检查对象存活性
        QTimer::singleShot(0, pWeakRef.data(), [strCacheKey, pixResult, pWeakRef]() {
            // 20260324 ZJH 此 lambda 在主线程执行
            if (!pixResult.isNull()) {
                QPixmapCache::insert(strCacheKey, pixResult);  // 20260324 ZJH 插入全局内存缓存
            }

            // 20260324 ZJH 再次检查存活性（QTimer 投递与执行之间可能有间隔）
            if (pWeakRef.isNull()) {
                return;  // 20260324 ZJH delegate 已销毁，跳过 UI 更新
            }

            // 20260324 ZJH 从待加载集合中移除
            auto* pDelegate = qobject_cast<ThumbnailDelegate*>(pWeakRef.data());
            if (pDelegate) {
                pDelegate->m_setPendingLoads.remove(strCacheKey);
            }

            // 20260324 ZJH 触发父视图 viewport 重绘以显示已加载的缩略图
            auto* pView = qobject_cast<QAbstractItemView*>(pWeakRef->parent());
            if (pView && pView->viewport()) {
                pView->viewport()->update();  // 20260324 ZJH 刷新视图
            }
        });
    });
}

// 20260322 ZJH 返回每个网格项的推荐大小
QSize ThumbnailDelegate::sizeHint(const QStyleOptionViewItem& /*option*/,
                                  const QModelIndex& /*index*/) const
{
    // 20260322 ZJH 网格项总大小 = 缩略图大小 + 上下内边距 + 文本行高
    int nWidth  = m_nThumbSize + s_nPadding * 2;  // 20260322 ZJH 宽度 = 缩略图 + 左右边距
    int nHeight = m_nThumbSize + s_nPadding * 2 + s_nTextHeight;  // 20260322 ZJH 高度 = 缩略图 + 上下边距 + 文本
    return QSize(nWidth, nHeight);
}

// 20260322 ZJH 设置缩略图目标大小
void ThumbnailDelegate::setThumbnailSize(int nSize)
{
    // 20260322 ZJH 限制范围在 80~320 像素之间
    m_nThumbSize = qBound(80, nSize, 320);
}

// 20260322 ZJH 获取当前缩略图大小
int ThumbnailDelegate::thumbnailSize() const
{
    return m_nThumbSize;
}
