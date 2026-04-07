// 20260330 ZJH AI 数据合成引擎 — 对标 Keyence AI 图像生成（核心卖点）
// 从少量真实缺陷样本自动生成大量合成训练数据
// 三种策略: CopyPaste合成 / 几何+光度增强合成 / GAN生成
// 纯 C++23 实现，零外部依赖（不依赖 OpenCV/Halcon）
module;

#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <array>
#include <iostream>
#include <cassert>

export module om.engine.data_synthesis;

// 20260330 ZJH 导入引擎内部模块：标注结构（BBoxAnnotation）和张量
import om.engine.annotation;

export namespace om {

// =========================================================================
// 20260330 ZJH 前向声明 / 数据类型
// =========================================================================

// 20260330 ZJH BBox — 合成管线用的像素坐标标注框
// 与 om.engine.annotation 中的 BBoxAnnotation（归一化坐标）不同，此处使用像素坐标
// 方便在固定尺寸图像上进行裁剪/粘贴运算
struct BBox {
    float fX = 0.0f;          // 20260330 ZJH 左上角 X（像素坐标）
    float fY = 0.0f;          // 20260330 ZJH 左上角 Y（像素坐标）
    float fW = 0.0f;          // 20260330 ZJH 宽度（像素）
    float fH = 0.0f;          // 20260330 ZJH 高度（像素）
    int nClassId = 0;          // 20260330 ZJH 类别 ID（从 0 开始）
    std::string strClassName;  // 20260330 ZJH 类别名称
    float fConfidence = 1.0f;  // 20260330 ZJH 置信度（合成数据默认 1.0）
};

// =========================================================================
// 20260330 ZJH 内部工具函数 — 双线性采样、高斯模糊、随机数
// =========================================================================

namespace detail {

// 20260330 ZJH 线程局部随机数引擎，避免多线程竞争
// 每个线程拥有独立的 Mersenne Twister 引擎
inline std::mt19937& rng() {
    static thread_local std::mt19937 s_gen{std::random_device{}()};  // 20260330 ZJH 用硬件随机种子初始化
    return s_gen;
}

// 20260330 ZJH 均匀分布随机浮点数 [fMin, fMax]
inline float randFloat(float fMin, float fMax) {
    std::uniform_real_distribution<float> dist(fMin, fMax);  // 20260330 ZJH 构造均匀分布
    return dist(rng());
}

// 20260330 ZJH 均匀分布随机整数 [nMin, nMax]
inline int randInt(int nMin, int nMax) {
    std::uniform_int_distribution<int> dist(nMin, nMax);  // 20260330 ZJH 构造整数均匀分布
    return dist(rng());
}

// 20260330 ZJH 正态分布随机浮点数 N(fMean, fStddev)
inline float randNormal(float fMean, float fStddev) {
    std::normal_distribution<float> dist(fMean, fStddev);  // 20260330 ZJH 构造高斯分布
    return dist(rng());
}

// 20260330 ZJH clamp — 将值限制在 [fMin, fMax] 范围内
inline float clampf(float fVal, float fMin, float fMax) {
    return std::max(fMin, std::min(fMax, fVal));  // 20260330 ZJH 标准 clamp 实现
}

// 20260330 ZJH 双线性采样 — 从 CHW 格式的浮点图像中按亚像素坐标采样
// vecImage: [C, H, W] 格式的图像数据，值域 [0,1]
// nC, nH, nW: 通道数、高度、宽度
// fY, fX: 亚像素行列坐标（可以是小数）
// nChannel: 要采样的通道索引
// 返回: 双线性插值后的像素值
inline float bilinearSample(const std::vector<float>& vecImage,
                            int nC, int nH, int nW,
                            float fY, float fX, int nChannel)
{
    // 20260330 ZJH 计算四个最近邻像素的整数坐标
    int nY0 = static_cast<int>(std::floor(fY));  // 20260330 ZJH 左上角行
    int nX0 = static_cast<int>(std::floor(fX));  // 20260330 ZJH 左上角列
    int nY1 = nY0 + 1;  // 20260330 ZJH 右下角行
    int nX1 = nX0 + 1;  // 20260330 ZJH 右下角列

    // 20260330 ZJH 计算小数部分（插值权重）
    float fDy = fY - static_cast<float>(nY0);  // 20260330 ZJH 行方向小数偏移
    float fDx = fX - static_cast<float>(nX0);  // 20260330 ZJH 列方向小数偏移

    // 20260330 ZJH 边界 clamp：超出图像范围时取最近边界像素
    auto clampY = [nH](int y) { return std::max(0, std::min(nH - 1, y)); };
    auto clampX = [nW](int x) { return std::max(0, std::min(nW - 1, x)); };

    // 20260330 ZJH CHW 索引计算：channel * H * W + row * W + col
    size_t nBase = static_cast<size_t>(nChannel) * static_cast<size_t>(nH) * static_cast<size_t>(nW);
    float f00 = vecImage[nBase + static_cast<size_t>(clampY(nY0)) * static_cast<size_t>(nW) + static_cast<size_t>(clampX(nX0))];  // 20260330 ZJH 左上
    float f01 = vecImage[nBase + static_cast<size_t>(clampY(nY0)) * static_cast<size_t>(nW) + static_cast<size_t>(clampX(nX1))];  // 20260330 ZJH 右上
    float f10 = vecImage[nBase + static_cast<size_t>(clampY(nY1)) * static_cast<size_t>(nW) + static_cast<size_t>(clampX(nX0))];  // 20260330 ZJH 左下
    float f11 = vecImage[nBase + static_cast<size_t>(clampY(nY1)) * static_cast<size_t>(nW) + static_cast<size_t>(clampX(nX1))];  // 20260330 ZJH 右下

    // 20260330 ZJH 双线性插值公式: (1-dy)*(1-dx)*f00 + (1-dy)*dx*f01 + dy*(1-dx)*f10 + dy*dx*f11
    float fResult = (1.0f - fDy) * (1.0f - fDx) * f00
                  + (1.0f - fDy) * fDx * f01
                  + fDy * (1.0f - fDx) * f10
                  + fDy * fDx * f11;
    return fResult;
}

// 20260330 ZJH 1D 高斯模糊 — 对一维数组做高斯平滑
// vecData: 输入/输出数组
// nLen: 数组长度
// fSigma: 高斯标准差
// 返回: 平滑后的数组（原地修改并返回引用）
inline void gaussianSmooth1D(std::vector<float>& vecData, int nLen, float fSigma) {
    if (fSigma <= 0.0f || nLen <= 1) return;  // 20260330 ZJH sigma<=0 无需平滑

    // 20260330 ZJH 计算核半径：3-sigma 规则覆盖 99.7% 的高斯分布
    int nRadius = static_cast<int>(std::ceil(3.0f * fSigma));
    if (nRadius < 1) nRadius = 1;  // 20260330 ZJH 最小核半径为 1

    // 20260330 ZJH 构建归一化高斯核
    std::vector<float> vecKernel(static_cast<size_t>(2 * nRadius + 1));
    float fSum = 0.0f;  // 20260330 ZJH 核权重总和（用于归一化）
    for (int i = -nRadius; i <= nRadius; ++i) {
        float fVal = std::exp(-0.5f * static_cast<float>(i * i) / (fSigma * fSigma));
        vecKernel[static_cast<size_t>(i + nRadius)] = fVal;
        fSum += fVal;
    }
    // 20260330 ZJH 归一化核权重，使总和为 1.0
    for (auto& v : vecKernel) v /= fSum;

    // 20260330 ZJH 卷积（使用临时缓冲区避免读写冲突）
    std::vector<float> vecTemp(static_cast<size_t>(nLen));
    for (int i = 0; i < nLen; ++i) {
        float fAcc = 0.0f;  // 20260330 ZJH 加权累加器
        for (int k = -nRadius; k <= nRadius; ++k) {
            // 20260330 ZJH 边界 clamp：超出范围时取最近边界值
            int nIdx = std::max(0, std::min(nLen - 1, i + k));
            fAcc += vecData[static_cast<size_t>(nIdx)] * vecKernel[static_cast<size_t>(k + nRadius)];
        }
        vecTemp[static_cast<size_t>(i)] = fAcc;
    }
    vecData = std::move(vecTemp);  // 20260330 ZJH 移动赋值回原数组
}

// 20260330 ZJH 2D 高斯模糊 — 对单通道 H×W 浮点图像做高斯平滑
// 使用可分离卷积优化: 先行方向再列方向，复杂度从 O(H*W*K^2) 降为 O(H*W*2K)
// vecData: [H, W] 的单通道数据
// nH, nW: 图像尺寸
// fSigma: 高斯标准差
inline void gaussianSmooth2D(std::vector<float>& vecData, int nH, int nW, float fSigma) {
    if (fSigma <= 0.0f) return;  // 20260330 ZJH sigma<=0 无需平滑

    // 20260330 ZJH 构建高斯核
    int nRadius = static_cast<int>(std::ceil(3.0f * fSigma));
    if (nRadius < 1) nRadius = 1;
    std::vector<float> vecKernel(static_cast<size_t>(2 * nRadius + 1));
    float fSum = 0.0f;
    for (int i = -nRadius; i <= nRadius; ++i) {
        float fVal = std::exp(-0.5f * static_cast<float>(i * i) / (fSigma * fSigma));
        vecKernel[static_cast<size_t>(i + nRadius)] = fVal;
        fSum += fVal;
    }
    for (auto& v : vecKernel) v /= fSum;

    // 20260330 ZJH 第一遍：水平方向卷积（逐行处理）
    std::vector<float> vecTemp(vecData.size());
    for (int y = 0; y < nH; ++y) {
        for (int x = 0; x < nW; ++x) {
            float fAcc = 0.0f;
            for (int k = -nRadius; k <= nRadius; ++k) {
                int nSrcX = std::max(0, std::min(nW - 1, x + k));  // 20260330 ZJH 水平 clamp
                fAcc += vecData[static_cast<size_t>(y * nW + nSrcX)]
                      * vecKernel[static_cast<size_t>(k + nRadius)];
            }
            vecTemp[static_cast<size_t>(y * nW + x)] = fAcc;
        }
    }

    // 20260330 ZJH 第二遍：垂直方向卷积（逐列处理）
    for (int y = 0; y < nH; ++y) {
        for (int x = 0; x < nW; ++x) {
            float fAcc = 0.0f;
            for (int k = -nRadius; k <= nRadius; ++k) {
                int nSrcY = std::max(0, std::min(nH - 1, y + k));  // 20260330 ZJH 垂直 clamp
                fAcc += vecTemp[static_cast<size_t>(nSrcY * nW + x)]
                      * vecKernel[static_cast<size_t>(k + nRadius)];
            }
            vecData[static_cast<size_t>(y * nW + x)] = fAcc;
        }
    }
}

// 20260330 ZJH 旋转 CHW 图像 — 绕中心旋转指定角度（双线性插值）
// vecSrc: 源图像 [C, H, W]
// nC, nH, nW: 通道/高度/宽度
// fAngleDeg: 旋转角度（度，逆时针为正）
// 返回: 旋转后的图像（同尺寸，超出部分填 0）
inline std::vector<float> rotateCHW(const std::vector<float>& vecSrc,
                                     int nC, int nH, int nW, float fAngleDeg)
{
    std::vector<float> vecDst(vecSrc.size(), 0.0f);  // 20260330 ZJH 初始化为黑色
    float fAngleRad = fAngleDeg * 3.14159265f / 180.0f;  // 20260330 ZJH 角度转弧度
    float fCosA = std::cos(fAngleRad);  // 20260330 ZJH 旋转矩阵的 cos 分量
    float fSinA = std::sin(fAngleRad);  // 20260330 ZJH 旋转矩阵的 sin 分量
    float fCx = static_cast<float>(nW) * 0.5f;  // 20260330 ZJH 旋转中心 X
    float fCy = static_cast<float>(nH) * 0.5f;  // 20260330 ZJH 旋转中心 Y

    // 20260330 ZJH 反向映射：对目标图像的每个像素，计算其在源图像中的位置
    for (int y = 0; y < nH; ++y) {
        for (int x = 0; x < nW; ++x) {
            // 20260330 ZJH 将目标坐标转为相对中心的偏移
            float fDx = static_cast<float>(x) - fCx;
            float fDy = static_cast<float>(y) - fCy;
            // 20260330 ZJH 反向旋转得到源坐标（转置旋转矩阵）
            float fSrcX = fCosA * fDx + fSinA * fDy + fCx;
            float fSrcY = -fSinA * fDx + fCosA * fDy + fCy;
            // 20260330 ZJH 检查源坐标是否在图像范围内
            // 20260330 ZJH 允许边界像素: 上限改为 nW/nH（bilinear 内部会安全处理边界）
            if (fSrcX >= 0.0f && fSrcX < static_cast<float>(nW) &&
                fSrcY >= 0.0f && fSrcY < static_cast<float>(nH)) {
                // 20260330 ZJH 逐通道双线性采样
                for (int c = 0; c < nC; ++c) {
                    size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nH * nW)
                                   + static_cast<size_t>(y * nW + x);
                    vecDst[nDstIdx] = bilinearSample(vecSrc, nC, nH, nW, fSrcY, fSrcX, c);
                }
            }
        }
    }
    return vecDst;
}

// 20260330 ZJH 缩放 CHW 图像 — 将源图像缩放到目标尺寸（双线性插值）
// vecSrc: 源图像 [C, srcH, srcW]
// nC: 通道数
// nSrcH, nSrcW: 源尺寸
// nDstH, nDstW: 目标尺寸
// 返回: 缩放后的图像 [C, nDstH, nDstW]
inline std::vector<float> resizeCHW(const std::vector<float>& vecSrc,
                                     int nC, int nSrcH, int nSrcW,
                                     int nDstH, int nDstW)
{
    std::vector<float> vecDst(static_cast<size_t>(nC * nDstH * nDstW), 0.0f);
    // 20260330 ZJH 计算缩放因子
    float fScaleY = static_cast<float>(nSrcH) / static_cast<float>(nDstH);
    float fScaleX = static_cast<float>(nSrcW) / static_cast<float>(nDstW);

    for (int y = 0; y < nDstH; ++y) {
        for (int x = 0; x < nDstW; ++x) {
            // 20260330 ZJH 反向映射到源图像坐标
            float fSrcY = (static_cast<float>(y) + 0.5f) * fScaleY - 0.5f;
            float fSrcX = (static_cast<float>(x) + 0.5f) * fScaleX - 0.5f;
            for (int c = 0; c < nC; ++c) {
                size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nDstH * nDstW)
                               + static_cast<size_t>(y * nDstW + x);
                vecDst[nDstIdx] = bilinearSample(vecSrc, nC, nSrcH, nSrcW, fSrcY, fSrcX, c);
            }
        }
    }
    return vecDst;
}

// 20260330 ZJH 计算图像某通道的 mean 和 std
// vecImage: [C, H, W] 格式
// nC, nH, nW: 通道/高/宽
// nChannel: 要计算统计量的通道
// fMean, fStd: 输出的均值和标准差
inline void channelMeanStd(const std::vector<float>& vecImage,
                           int nC, int nH, int nW, int nChannel,
                           float& fMean, float& fStd)
{
    size_t nBase = static_cast<size_t>(nChannel) * static_cast<size_t>(nH * nW);
    size_t nPixels = static_cast<size_t>(nH * nW);
    double dSum = 0.0;   // 20260330 ZJH 用 double 累加避免浮点精度损失
    double dSumSq = 0.0;
    for (size_t i = 0; i < nPixels; ++i) {
        double v = static_cast<double>(vecImage[nBase + i]);
        dSum += v;
        dSumSq += v * v;
    }
    fMean = static_cast<float>(dSum / static_cast<double>(nPixels));
    double dVar = dSumSq / static_cast<double>(nPixels) - static_cast<double>(fMean) * static_cast<double>(fMean);
    fStd = static_cast<float>(std::sqrt(std::max(0.0, dVar)));  // 20260330 ZJH 防止浮点误差导致负方差
    if (fStd < 1e-7f) fStd = 1e-7f;  // 20260330 ZJH 防止除零
}

}  // namespace detail

// =========================================================================
// 20260330 ZJH 策略1: CopyPaste 缺陷合成
// =========================================================================

// 20260330 ZJH CopyPaste 合成配置
struct CopyPasteConfig {
    int nNumSynthPerDefect = 10;    // 20260330 ZJH 每个缺陷源生成的合成图数量
    float fScaleMin = 0.7f;         // 20260330 ZJH 缺陷缩放下限
    float fScaleMax = 1.3f;         // 20260330 ZJH 缺陷缩放上限
    float fRotateRange = 30.0f;     // 20260330 ZJH 旋转角度范围（±度）
    float fBlendAlpha = 0.9f;       // 20260330 ZJH 融合透明度（0=完全透明, 1=完全不透明）
    bool bPoissonBlend = false;     // 20260330 ZJH 泊松融合默认关闭（CPU 太慢），用 alpha 混合替代
    bool bRandomPosition = true;    // 20260330 ZJH 是否随机选择粘贴位置
};

// 20260330 ZJH CopyPaste 缺陷合成器
// 核心思路：从标注图像中裁剪缺陷区域，经过随机变换后粘贴到正常样本上
// 生成带有真实缺陷纹理的合成训练数据
class DefectCopyPaste {
public:
    // 20260330 ZJH 缺陷 patch 数据结构
    // 存储从标注图像中裁剪出的缺陷区域及其二值 mask
    struct DefectPatch {
        std::vector<float> vecPixels;    // 20260330 ZJH 缺陷像素 [C, H, W] CHW 格式
        std::vector<uint8_t> vecMask;    // 20260330 ZJH 二值 mask [H, W]（255=缺陷, 0=背景）
        int nW = 0;                      // 20260330 ZJH patch 宽度
        int nH = 0;                      // 20260330 ZJH patch 高度
        int nC = 0;                      // 20260330 ZJH 通道数
        std::string strClassName;        // 20260330 ZJH 缺陷类别名称
        int nClassId = 0;               // 20260330 ZJH 缺陷类别 ID
    };

    // 20260330 ZJH extractDefects — 从标注图像中提取缺陷 patch
    // 根据 BBox 标注裁剪出缺陷区域，生成对应的矩形二值 mask
    // vecImage: 源图像 [C, H, W] CHW 格式 float [0,1]
    // nC, nH, nW: 通道/高度/宽度
    // vecAnnotations: 缺陷标注框列表
    // 返回: 提取出的缺陷 patch 列表
    static std::vector<DefectPatch> extractDefects(
        const std::vector<float>& vecImage, int nC, int nH, int nW,
        const std::vector<BBox>& vecAnnotations)
    {
        std::vector<DefectPatch> vecPatches;  // 20260330 ZJH 结果列表
        vecPatches.reserve(vecAnnotations.size());  // 20260330 ZJH 预分配

        for (const auto& bbox : vecAnnotations) {
            // 20260330 ZJH 将 BBox 像素坐标转为整数，并 clamp 到图像边界
            int nX0 = std::max(0, static_cast<int>(std::floor(bbox.fX)));
            int nY0 = std::max(0, static_cast<int>(std::floor(bbox.fY)));
            int nX1 = std::min(nW, static_cast<int>(std::ceil(bbox.fX + bbox.fW)));
            int nY1 = std::min(nH, static_cast<int>(std::ceil(bbox.fY + bbox.fH)));

            int nPatchW = nX1 - nX0;  // 20260330 ZJH patch 实际宽度
            int nPatchH = nY1 - nY0;  // 20260330 ZJH patch 实际高度

            // 20260330 ZJH 跳过过小的 patch（面积不足 4 像素）
            if (nPatchW < 2 || nPatchH < 2) continue;

            DefectPatch patch;  // 20260330 ZJH 当前缺陷 patch
            patch.nW = nPatchW;
            patch.nH = nPatchH;
            patch.nC = nC;
            patch.strClassName = bbox.strClassName;
            patch.nClassId = bbox.nClassId;

            // 20260330 ZJH 从源图像裁剪像素数据（CHW 格式）
            patch.vecPixels.resize(static_cast<size_t>(nC * nPatchH * nPatchW));
            for (int c = 0; c < nC; ++c) {
                for (int y = 0; y < nPatchH; ++y) {
                    for (int x = 0; x < nPatchW; ++x) {
                        // 20260330 ZJH 源图像的 CHW 索引
                        size_t nSrcIdx = static_cast<size_t>(c) * static_cast<size_t>(nH * nW)
                                       + static_cast<size_t>((nY0 + y) * nW + (nX0 + x));
                        // 20260330 ZJH patch 的 CHW 索引
                        size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nPatchH * nPatchW)
                                       + static_cast<size_t>(y * nPatchW + x);
                        patch.vecPixels[nDstIdx] = vecImage[nSrcIdx];
                    }
                }
            }

            // 20260330 ZJH 生成矩形二值 mask（整个 patch 区域全为 255）
            // 后续可扩展为基于颜色/纹理的精细 mask 提取
            patch.vecMask.assign(static_cast<size_t>(nPatchH * nPatchW), 255);

            vecPatches.push_back(std::move(patch));
        }

        return vecPatches;
    }

    // 20260330 ZJH synthesize — 将单个缺陷 patch 合成到正常图像上
    // 执行流程: 缩放 → 旋转 → 选择粘贴位置 → 梯度域融合 → 输出
    // vecNormalImage: 正常（无缺陷）背景图像 [C, H, W]
    // nC, nH, nW: 通道/高度/宽度
    // defect: 要粘贴的缺陷 patch
    // config: 合成参数配置
    // fPasteX, fPasteY: 可选的粘贴位置（<0 表示随机选择）
    // 返回: 合成后的图像 [C, H, W] 及粘贴位置的 BBox
    static std::pair<std::vector<float>, BBox> synthesizeOne(
        const std::vector<float>& vecNormalImage, int nC, int nH, int nW,
        const DefectPatch& defect, const CopyPasteConfig& config,
        float fPasteX = -1.0f, float fPasteY = -1.0f)
    {
        // 20260330 ZJH Step 1: 随机缩放缺陷 patch
        float fScale = detail::randFloat(config.fScaleMin, config.fScaleMax);
        int nScaledW = std::max(2, static_cast<int>(std::round(defect.nW * fScale)));
        int nScaledH = std::max(2, static_cast<int>(std::round(defect.nH * fScale)));

        // 20260330 ZJH 缩放缺陷像素
        std::vector<float> vecScaledPatch = detail::resizeCHW(
            defect.vecPixels, defect.nC, defect.nH, defect.nW, nScaledH, nScaledW);

        // 20260330 ZJH 缩放 mask（用最近邻插值保持二值性）
        std::vector<uint8_t> vecScaledMask(static_cast<size_t>(nScaledH * nScaledW));
        for (int y = 0; y < nScaledH; ++y) {
            for (int x = 0; x < nScaledW; ++x) {
                int nSrcY = static_cast<int>(static_cast<float>(y) * static_cast<float>(defect.nH) / static_cast<float>(nScaledH));
                int nSrcX = static_cast<int>(static_cast<float>(x) * static_cast<float>(defect.nW) / static_cast<float>(nScaledW));
                nSrcY = std::min(nSrcY, defect.nH - 1);
                nSrcX = std::min(nSrcX, defect.nW - 1);
                vecScaledMask[static_cast<size_t>(y * nScaledW + x)] =
                    defect.vecMask[static_cast<size_t>(nSrcY * defect.nW + nSrcX)];
            }
        }

        // 20260330 ZJH Step 2: 随机旋转（如果旋转范围 > 0）
        if (config.fRotateRange > 0.0f) {
            float fAngle = detail::randFloat(-config.fRotateRange, config.fRotateRange);
            if (std::abs(fAngle) > 1.0f) {  // 20260330 ZJH 角度太小则跳过旋转
                vecScaledPatch = detail::rotateCHW(vecScaledPatch, nC, nScaledH, nScaledW, fAngle);
                // 20260330 ZJH 旋转 mask：先转为 float，旋转后二值化
                std::vector<float> vecMaskFloat(vecScaledMask.size());
                for (size_t i = 0; i < vecScaledMask.size(); ++i) {
                    vecMaskFloat[i] = vecScaledMask[i] > 127 ? 1.0f : 0.0f;
                }
                // 20260330 ZJH 用 1 通道 CHW 格式旋转 mask
                std::vector<float> vecRotMask = detail::rotateCHW(vecMaskFloat, 1, nScaledH, nScaledW, fAngle);
                for (size_t i = 0; i < vecScaledMask.size(); ++i) {
                    vecScaledMask[i] = vecRotMask[i] > 0.3f ? 255 : 0;  // 20260330 ZJH 阈值二值化
                }
            }
        }

        // 20260330 ZJH Step 3: 确定粘贴位置
        int nPasteX, nPasteY;
        if (config.bRandomPosition && fPasteX < 0.0f) {
            // 20260330 ZJH 随机位置（确保缺陷 patch 大部分在图像内）
            int nMarginX = std::max(0, nW - nScaledW);
            int nMarginY = std::max(0, nH - nScaledH);
            nPasteX = (nMarginX > 0) ? detail::randInt(0, nMarginX) : 0;
            nPasteY = (nMarginY > 0) ? detail::randInt(0, nMarginY) : 0;
        } else {
            nPasteX = static_cast<int>(fPasteX);
            nPasteY = static_cast<int>(fPasteY);
        }

        // 20260330 ZJH Step 4: 混合粘贴
        // 复制正常图像作为输出基底
        std::vector<float> vecResult = vecNormalImage;

        if (config.bPoissonBlend) {
            // 20260330 ZJH 梯度域融合（泊松融合近似）
            // 原理：在缺陷 mask 内部区域，使用缺陷 patch 的梯度替换背景梯度
            // 然后通过迭代求解泊松方程重建像素值，使边界无缝过渡
            // 此处使用简化的 Jacobi 迭代法（200 次迭代，确保大 patch 充分收敛）
            poissonBlend(vecResult, vecScaledPatch, vecScaledMask,
                         nC, nH, nW, nScaledH, nScaledW,
                         nPasteX, nPasteY, config.fBlendAlpha);
        } else {
            // 20260330 ZJH 简单 alpha 混合：dst = alpha * src + (1-alpha) * dst
            alphaBlend(vecResult, vecScaledPatch, vecScaledMask,
                       nC, nH, nW, nScaledH, nScaledW,
                       nPasteX, nPasteY, config.fBlendAlpha);
        }

        // 20260330 ZJH 构建合成结果的 BBox 标注
        BBox synthBBox;
        synthBBox.fX = static_cast<float>(nPasteX);
        synthBBox.fY = static_cast<float>(nPasteY);
        synthBBox.fW = static_cast<float>(nScaledW);
        synthBBox.fH = static_cast<float>(nScaledH);
        synthBBox.nClassId = defect.nClassId;
        synthBBox.strClassName = defect.strClassName;
        synthBBox.fConfidence = 1.0f;  // 20260330 ZJH 合成数据置信度固定为 1.0

        return {vecResult, synthBBox};
    }

    // 20260330 ZJH synthesize — 便捷接口，只返回图像
    static std::vector<float> synthesize(
        const std::vector<float>& vecNormalImage, int nC, int nH, int nW,
        const DefectPatch& defect, const CopyPasteConfig& config)
    {
        return synthesizeOne(vecNormalImage, nC, nH, nW, defect, config).first;
    }

    // 20260330 ZJH batchSynthesize — 批量合成
    // 从正常图像集和缺陷 patch 集的笛卡尔积中随机抽样合成
    // vecNormalImages: 正常图像列表（每张 [C, H, W]）
    // vecDefects: 缺陷 patch 列表
    // nC, nH, nW: 图像尺寸
    // config: 合成配置
    // 返回: 合成图像及其标注的列表
    static std::vector<std::pair<std::vector<float>, std::vector<BBox>>> batchSynthesize(
        const std::vector<std::vector<float>>& vecNormalImages,
        const std::vector<DefectPatch>& vecDefects,
        int nC, int nH, int nW,
        const CopyPasteConfig& config)
    {
        std::vector<std::pair<std::vector<float>, std::vector<BBox>>> vecResults;

        if (vecNormalImages.empty() || vecDefects.empty()) {
            return vecResults;  // 20260330 ZJH 空输入直接返回
        }

        // 20260330 ZJH 对每个缺陷 patch，在随机正常图像上生成 nNumSynthPerDefect 张合成图
        for (const auto& defect : vecDefects) {
            for (int i = 0; i < config.nNumSynthPerDefect; ++i) {
                // 20260330 ZJH 随机选择一张正常背景图
                int nBgIdx = detail::randInt(0, static_cast<int>(vecNormalImages.size()) - 1);
                const auto& vecBg = vecNormalImages[static_cast<size_t>(nBgIdx)];

                // 20260330 ZJH 合成一张
                auto [vecSynthImg, synthBBox] = synthesizeOne(vecBg, nC, nH, nW, defect, config);

                // 20260330 ZJH 存储结果（每张合成图有一个 BBox）
                vecResults.push_back({std::move(vecSynthImg), {synthBBox}});
            }
        }

        return vecResults;
    }

private:
    // 20260330 ZJH alphaBlend — 简单 alpha 混合
    // 在 mask 区域内将 patch 以 alpha 透明度叠加到目标图像
    static void alphaBlend(std::vector<float>& vecDst,
                           const std::vector<float>& vecPatch,
                           const std::vector<uint8_t>& vecMask,
                           int nDstC, int nDstH, int nDstW,
                           int nPatchH, int nPatchW,
                           int nPasteX, int nPasteY, float fAlpha)
    {
        for (int y = 0; y < nPatchH; ++y) {
            int nDstY = nPasteY + y;  // 20260330 ZJH 目标图像行坐标
            if (nDstY < 0 || nDstY >= nDstH) continue;  // 20260330 ZJH 越界跳过

            for (int x = 0; x < nPatchW; ++x) {
                int nDstX = nPasteX + x;  // 20260330 ZJH 目标图像列坐标
                if (nDstX < 0 || nDstX >= nDstW) continue;  // 20260330 ZJH 越界跳过

                // 20260330 ZJH 检查 mask：只在缺陷区域内粘贴
                if (vecMask[static_cast<size_t>(y * nPatchW + x)] < 128) continue;

                // 20260330 ZJH 逐通道 alpha 混合
                for (int c = 0; c < nDstC; ++c) {
                    size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nDstH * nDstW)
                                   + static_cast<size_t>(nDstY * nDstW + nDstX);
                    size_t nSrcIdx = static_cast<size_t>(c) * static_cast<size_t>(nPatchH * nPatchW)
                                   + static_cast<size_t>(y * nPatchW + x);
                    // 20260330 ZJH alpha 混合公式: dst = alpha * src + (1-alpha) * dst
                    vecDst[nDstIdx] = fAlpha * vecPatch[nSrcIdx]
                                    + (1.0f - fAlpha) * vecDst[nDstIdx];
                }
            }
        }
    }

    // 20260330 ZJH poissonBlend — 梯度域无缝融合（泊松方程 Jacobi 求解）
    // 原理:
    //   1. 在缺陷 mask 内部，使用缺陷 patch 的拉普拉斯算子（梯度散度）作为引导
    //   2. 边界条件由背景图像提供
    //   3. 用 Jacobi 迭代求解离散泊松方程: Δf = Δg（g=patch, f=输出）
    //   这样 mask 内部的梯度与 patch 一致，但边界与背景无缝过渡
    static void poissonBlend(std::vector<float>& vecDst,
                             const std::vector<float>& vecPatch,
                             const std::vector<uint8_t>& vecMask,
                             int nDstC, int nDstH, int nDstW,
                             int nPatchH, int nPatchW,
                             int nPasteX, int nPasteY, float fAlpha)
    {
        // 20260330 ZJH 逐通道处理（泊松方程对每个通道独立求解）
        for (int c = 0; c < nDstC; ++c) {
            // 20260330 ZJH 提取目标图像中粘贴区域对应的单通道子图
            // 为迭代求解创建工作缓冲区
            std::vector<float> vecWork(static_cast<size_t>(nPatchH * nPatchW), 0.0f);

            // 20260330 ZJH 初始化工作缓冲区：mask 内用 patch 值，mask 外用背景值
            for (int y = 0; y < nPatchH; ++y) {
                for (int x = 0; x < nPatchW; ++x) {
                    int nDstY = nPasteY + y;
                    int nDstX = nPasteX + x;
                    size_t nLocalIdx = static_cast<size_t>(y * nPatchW + x);

                    if (nDstY >= 0 && nDstY < nDstH && nDstX >= 0 && nDstX < nDstW) {
                        size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nDstH * nDstW)
                                       + static_cast<size_t>(nDstY * nDstW + nDstX);
                        // 20260330 ZJH 初始值为背景（Jacobi 迭代的初始猜测）
                        vecWork[nLocalIdx] = vecDst[nDstIdx];
                    }
                }
            }

            // 20260330 ZJH 预计算 patch 的拉普拉斯（离散 Laplacian）
            // Laplacian(g) = g(y-1,x) + g(y+1,x) + g(y,x-1) + g(y,x+1) - 4*g(y,x)
            std::vector<float> vecLaplacian(static_cast<size_t>(nPatchH * nPatchW), 0.0f);
            for (int y = 1; y < nPatchH - 1; ++y) {
                for (int x = 1; x < nPatchW - 1; ++x) {
                    size_t nIdx = static_cast<size_t>(c) * static_cast<size_t>(nPatchH * nPatchW);
                    float fCenter = vecPatch[nIdx + static_cast<size_t>(y * nPatchW + x)];
                    float fTop    = vecPatch[nIdx + static_cast<size_t>((y - 1) * nPatchW + x)];
                    float fBot    = vecPatch[nIdx + static_cast<size_t>((y + 1) * nPatchW + x)];
                    float fLeft   = vecPatch[nIdx + static_cast<size_t>(y * nPatchW + (x - 1))];
                    float fRight  = vecPatch[nIdx + static_cast<size_t>(y * nPatchW + (x + 1))];
                    // 20260330 ZJH 离散拉普拉斯算子
                    vecLaplacian[static_cast<size_t>(y * nPatchW + x)] =
                        fTop + fBot + fLeft + fRight - 4.0f * fCenter;
                }
            }

            // 20260330 ZJH 构建 mask 边界集合：mask 内且至少有一个邻居在 mask 外
            // 这些位置使用 Dirichlet 边界条件（固定为背景值）
            std::vector<bool> vecIsInterior(static_cast<size_t>(nPatchH * nPatchW), false);
            for (int y = 1; y < nPatchH - 1; ++y) {
                for (int x = 1; x < nPatchW - 1; ++x) {
                    size_t nIdx = static_cast<size_t>(y * nPatchW + x);
                    if (vecMask[nIdx] >= 128) {
                        // 20260330 ZJH 检查四邻域是否全在 mask 内（内部点）
                        bool bAllNeighInMask =
                            vecMask[static_cast<size_t>((y - 1) * nPatchW + x)] >= 128 &&
                            vecMask[static_cast<size_t>((y + 1) * nPatchW + x)] >= 128 &&
                            vecMask[static_cast<size_t>(y * nPatchW + (x - 1))] >= 128 &&
                            vecMask[static_cast<size_t>(y * nPatchW + (x + 1))] >= 128;
                        vecIsInterior[nIdx] = bAllNeighInMask;
                    }
                }
            }

            // 20260330 ZJH Jacobi 迭代求解泊松方程
            // 离散泊松: f(y,x) = (1/4) * [f(y-1,x) + f(y+1,x) + f(y,x-1) + f(y,x+1) - Lap_g(y,x)]
            // 边界点: 使用背景图像值（Dirichlet 条件）
            constexpr int nMaxIter = 20;  // 20260330 ZJH 迭代次数（20 次，256x256 已足够收敛，200 次太慢）
            std::vector<float> vecNext(vecWork.size());

            for (int iter = 0; iter < nMaxIter; ++iter) {
                vecNext = vecWork;  // 20260330 ZJH 复制当前值

                for (int y = 1; y < nPatchH - 1; ++y) {
                    for (int x = 1; x < nPatchW - 1; ++x) {
                        size_t nIdx = static_cast<size_t>(y * nPatchW + x);

                        // 20260330 ZJH 只更新 mask 内部点（边界固定）
                        if (!vecIsInterior[nIdx]) continue;

                        // 20260330 ZJH 读取四邻域当前值
                        float fTop   = vecWork[static_cast<size_t>((y - 1) * nPatchW + x)];
                        float fBot   = vecWork[static_cast<size_t>((y + 1) * nPatchW + x)];
                        float fLeft  = vecWork[static_cast<size_t>(y * nPatchW + (x - 1))];
                        float fRight = vecWork[static_cast<size_t>(y * nPatchW + (x + 1))];

                        // 20260330 ZJH Jacobi 更新公式
                        float fLap = vecLaplacian[nIdx];
                        vecNext[nIdx] = 0.25f * (fTop + fBot + fLeft + fRight - fLap);

                        // 20260330 ZJH clamp 到 [0, 1] 防止数值溢出
                        vecNext[nIdx] = detail::clampf(vecNext[nIdx], 0.0f, 1.0f);
                    }
                }

                std::swap(vecWork, vecNext);  // 20260330 ZJH 双缓冲交换
            }

            // 20260330 ZJH 将融合结果写回目标图像
            for (int y = 0; y < nPatchH; ++y) {
                for (int x = 0; x < nPatchW; ++x) {
                    int nDstY = nPasteY + y;
                    int nDstX = nPasteX + x;
                    if (nDstY < 0 || nDstY >= nDstH || nDstX < 0 || nDstX >= nDstW) continue;

                    size_t nMaskIdx = static_cast<size_t>(y * nPatchW + x);
                    if (vecMask[nMaskIdx] < 128) continue;  // 20260330 ZJH mask 外不修改

                    size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nDstH * nDstW)
                                   + static_cast<size_t>(nDstY * nDstW + nDstX);
                    size_t nLocalIdx = static_cast<size_t>(y * nPatchW + x);

                    // 20260330 ZJH 用 fAlpha 控制融合结果与背景的混合比例
                    vecDst[nDstIdx] = fAlpha * vecWork[nLocalIdx]
                                    + (1.0f - fAlpha) * vecDst[nDstIdx];
                }
            }
        }
    }
};

// =========================================================================
// 20260330 ZJH 策略2: 几何+光度增强合成
// =========================================================================

// 20260330 ZJH 增强合成配置
struct AugSynthConfig {
    int nNumVariants = 20;           // 20260330 ZJH 每张源图生成的变体数
    bool bElasticDeform = true;      // 20260330 ZJH 是否启用弹性变形（模拟材料形变）
    bool bPerspective = true;        // 20260330 ZJH 是否启用透视变换
    bool bColorTransfer = true;      // 20260330 ZJH 是否启用颜色迁移（匹配目标色调）
    float fNoiseLevel = 0.03f;       // 20260330 ZJH 高斯噪声标准差
    float fElasticAlpha = 30.0f;     // 20260330 ZJH 弹性变形强度
    float fElasticSigma = 4.0f;      // 20260330 ZJH 弹性变形平滑度
    float fPerspectiveMaxAngle = 15.0f;  // 20260330 ZJH 透视变换最大角度（度）
};

// 20260330 ZJH 增强合成器
// 通过弹性变形、透视变换、颜色迁移和噪声注入从单张缺陷图生成多个变体
class AugmentSynthesizer {
public:
    // 20260330 ZJH generateVariants — 从单张缺陷图像生成多个增强变体
    // vecSource: 源图像 [C, H, W]
    // nC, nH, nW: 通道/高度/宽度
    // config: 增强配置
    // vecTargetImages: 可选的目标图像列表（用于颜色迁移）
    // 返回: 生成的变体图像列表
    static std::vector<std::vector<float>> generateVariants(
        const std::vector<float>& vecSource, int nC, int nH, int nW,
        const AugSynthConfig& config,
        const std::vector<std::vector<float>>& vecTargetImages = {})
    {
        std::vector<std::vector<float>> vecResults;
        vecResults.reserve(static_cast<size_t>(config.nNumVariants));

        for (int i = 0; i < config.nNumVariants; ++i) {
            // 20260330 ZJH 从源图像开始，逐步叠加变换
            std::vector<float> vecVariant = vecSource;

            // 20260330 ZJH 50% 概率应用弹性变形
            if (config.bElasticDeform && detail::randFloat(0.0f, 1.0f) > 0.5f) {
                vecVariant = elasticDeform(vecVariant, nC, nH, nW,
                                           config.fElasticAlpha, config.fElasticSigma);
            }

            // 20260330 ZJH 50% 概率应用透视变换
            if (config.bPerspective && detail::randFloat(0.0f, 1.0f) > 0.5f) {
                vecVariant = perspectiveTransform(vecVariant, nC, nH, nW,
                                                   config.fPerspectiveMaxAngle);
            }

            // 20260330 ZJH 40% 概率应用颜色迁移（需要目标图像）
            if (config.bColorTransfer && !vecTargetImages.empty() &&
                detail::randFloat(0.0f, 1.0f) > 0.6f) {
                int nTargetIdx = detail::randInt(0, static_cast<int>(vecTargetImages.size()) - 1);
                vecVariant = colorTransfer(vecVariant, vecTargetImages[static_cast<size_t>(nTargetIdx)],
                                           nC, nH, nW);
            }

            // 20260330 ZJH 总是添加少量随机噪声（模拟传感器噪声）
            if (config.fNoiseLevel > 0.0f) {
                addGaussianNoise(vecVariant, config.fNoiseLevel);
            }

            // 20260330 ZJH 30% 概率随机水平翻转
            if (detail::randFloat(0.0f, 1.0f) > 0.7f) {
                flipHorizontal(vecVariant, nC, nH, nW);
            }

            // 20260330 ZJH 30% 概率随机亮度/对比度微调
            if (detail::randFloat(0.0f, 1.0f) > 0.7f) {
                float fBrightness = detail::randFloat(-0.1f, 0.1f);  // 20260330 ZJH 亮度偏移
                float fContrast = detail::randFloat(0.85f, 1.15f);    // 20260330 ZJH 对比度缩放
                adjustBrightnessContrast(vecVariant, fBrightness, fContrast);
            }

            vecResults.push_back(std::move(vecVariant));
        }

        return vecResults;
    }

    // 20260330 ZJH elasticDeform — 弹性变形
    // 通过生成随机位移场并用高斯核平滑，模拟材料/产品的真实形变
    // 这对工业缺陷（如划痕拉伸、褶皱变形）特别有效
    // vecImage: 输入图像 [C, H, W]
    // nC, nH, nW: 通道/高度/宽度
    // fAlpha: 变形强度（位移场的缩放因子，越大变形越剧烈）
    // fSigma: 高斯平滑标准差（越大变形越平滑/全局化）
    // 返回: 变形后的图像
    static std::vector<float> elasticDeform(
        const std::vector<float>& vecImage, int nC, int nH, int nW,
        float fAlpha = 30.0f, float fSigma = 4.0f)
    {
        size_t nPixels = static_cast<size_t>(nH * nW);

        // 20260330 ZJH Step 1: 生成随机位移场（水平和垂直各一个）
        // 初始值为 [-1, 1] 的均匀随机数
        std::vector<float> vecDx(nPixels);  // 20260330 ZJH 水平位移场
        std::vector<float> vecDy(nPixels);  // 20260330 ZJH 垂直位移场
        for (size_t i = 0; i < nPixels; ++i) {
            vecDx[i] = detail::randFloat(-1.0f, 1.0f);
            vecDy[i] = detail::randFloat(-1.0f, 1.0f);
        }

        // 20260330 ZJH Step 2: 用高斯核平滑位移场
        // 平滑后的位移场具有空间相关性，产生自然的变形效果
        detail::gaussianSmooth2D(vecDx, nH, nW, fSigma);
        detail::gaussianSmooth2D(vecDy, nH, nW, fSigma);

        // 20260330 ZJH Step 3: 缩放位移场（fAlpha 控制变形强度）
        for (size_t i = 0; i < nPixels; ++i) {
            vecDx[i] *= fAlpha;
            vecDy[i] *= fAlpha;
        }

        // 20260330 ZJH Step 4: 应用位移场 — 双线性采样
        std::vector<float> vecResult(vecImage.size(), 0.0f);
        for (int y = 0; y < nH; ++y) {
            for (int x = 0; x < nW; ++x) {
                size_t nPixIdx = static_cast<size_t>(y * nW + x);
                // 20260330 ZJH 计算源坐标 = 目标坐标 + 位移
                float fSrcX = static_cast<float>(x) + vecDx[nPixIdx];
                float fSrcY = static_cast<float>(y) + vecDy[nPixIdx];

                // 20260330 ZJH 逐通道双线性采样
                for (int c = 0; c < nC; ++c) {
                    size_t nDstIdx = static_cast<size_t>(c) * nPixels + nPixIdx;
                    vecResult[nDstIdx] = detail::bilinearSample(vecImage, nC, nH, nW, fSrcY, fSrcX, c);
                }
            }
        }

        return vecResult;
    }

    // 20260330 ZJH perspectiveTransform — 透视变换
    // 模拟相机角度变化导致的透视畸变
    // 通过对图像四角施加随机偏移，构建 3×3 单应性矩阵，然后做反向映射
    // vecImage: 输入图像 [C, H, W]
    // nC, nH, nW: 通道/高度/宽度
    // fMaxAngle: 最大透视角度（度），控制畸变程度
    // 返回: 变换后的图像
    static std::vector<float> perspectiveTransform(
        const std::vector<float>& vecImage, int nC, int nH, int nW,
        float fMaxAngle = 15.0f)
    {
        // 20260330 ZJH 将角度转为像素偏移量
        // 经验公式：最大偏移 ≈ 图像尺寸 × tan(angle)，但限制在合理范围内
        float fMaxShift = std::min(static_cast<float>(nW), static_cast<float>(nH))
                        * std::tan(fMaxAngle * 3.14159265f / 180.0f) * 0.3f;

        // 20260330 ZJH 源图像四角坐标
        float srcCorners[4][2] = {
            {0.0f, 0.0f},                                          // 20260330 ZJH 左上
            {static_cast<float>(nW - 1), 0.0f},                   // 20260330 ZJH 右上
            {static_cast<float>(nW - 1), static_cast<float>(nH - 1)},  // 20260330 ZJH 右下
            {0.0f, static_cast<float>(nH - 1)}                    // 20260330 ZJH 左下
        };

        // 20260330 ZJH 目标四角 = 源四角 + 随机偏移
        float dstCorners[4][2];
        for (int i = 0; i < 4; ++i) {
            dstCorners[i][0] = srcCorners[i][0] + detail::randFloat(-fMaxShift, fMaxShift);
            dstCorners[i][1] = srcCorners[i][1] + detail::randFloat(-fMaxShift, fMaxShift);
        }

        // 20260330 ZJH 计算 3×3 单应性矩阵 H，使得 dst = H * src
        // 使用 DLT（Direct Linear Transform）方法求解
        // 4 对点产生 8 个方程，求解 8 个未知数（h33 归一化为 1）
        std::array<float, 9> H = computeHomography(srcCorners, dstCorners);

        // 20260330 ZJH 计算逆单应性（用于反向映射）
        std::array<float, 9> Hinv = invertHomography(H);

        // 20260330 ZJH 反向映射：对每个目标像素，找到源图像中的对应位置
        std::vector<float> vecResult(vecImage.size(), 0.0f);
        for (int y = 0; y < nH; ++y) {
            for (int x = 0; x < nW; ++x) {
                // 20260330 ZJH 齐次坐标变换：[x', y', w'] = Hinv * [x, y, 1]
                float fDstX = static_cast<float>(x);
                float fDstY = static_cast<float>(y);
                float fSrcXh = Hinv[0] * fDstX + Hinv[1] * fDstY + Hinv[2];
                float fSrcYh = Hinv[3] * fDstX + Hinv[4] * fDstY + Hinv[5];
                float fW     = Hinv[6] * fDstX + Hinv[7] * fDstY + Hinv[8];

                // 20260330 ZJH 除以齐次坐标 w 得到欧氏坐标
                if (std::abs(fW) < 1e-8f) continue;  // 20260330 ZJH 防止除零
                float fSrcX = fSrcXh / fW;
                float fSrcY = fSrcYh / fW;

                // 20260330 ZJH 检查是否在源图像范围内
                // 20260330 ZJH 允许边界像素: 上限改为 nW/nH（bilinear 内部会安全处理边界）
                if (fSrcX < 0.0f || fSrcX >= static_cast<float>(nW) ||
                    fSrcY < 0.0f || fSrcY >= static_cast<float>(nH)) {
                    continue;  // 20260330 ZJH 超出范围填 0（黑色）
                }

                // 20260330 ZJH 双线性采样
                for (int c = 0; c < nC; ++c) {
                    size_t nDstIdx = static_cast<size_t>(c) * static_cast<size_t>(nH * nW)
                                   + static_cast<size_t>(y * nW + x);
                    vecResult[nDstIdx] = detail::bilinearSample(vecImage, nC, nH, nW, fSrcY, fSrcX, c);
                }
            }
        }

        return vecResult;
    }

    // 20260330 ZJH colorTransfer — 颜色迁移
    // 将源图像的颜色分布匹配到目标图像（逐通道 mean/std 匹配）
    // 原理：对每个通道 c，令 out_c = (src_c - mean_src_c) * (std_dst_c / std_src_c) + mean_dst_c
    // 这样输出图像保持源图像的结构，但色调与目标图像一致
    // vecSource: 源图像 [C, H, W]
    // vecTarget: 目标图像 [C, H, W]（提供色调参考）
    // nC, nH, nW: 通道/高度/宽度
    // 返回: 颜色迁移后的图像
    static std::vector<float> colorTransfer(
        const std::vector<float>& vecSource,
        const std::vector<float>& vecTarget,
        int nC, int nH, int nW)
    {
        std::vector<float> vecResult = vecSource;  // 20260330 ZJH 从源图像复制
        size_t nPixels = static_cast<size_t>(nH * nW);

        for (int c = 0; c < nC; ++c) {
            // 20260330 ZJH 计算源通道和目标通道的均值/标准差
            float fSrcMean, fSrcStd, fDstMean, fDstStd;
            detail::channelMeanStd(vecSource, nC, nH, nW, c, fSrcMean, fSrcStd);
            detail::channelMeanStd(vecTarget, nC, nH, nW, c, fDstMean, fDstStd);

            // 20260330 ZJH 逐像素变换：标准化后重新缩放到目标分布
            size_t nBase = static_cast<size_t>(c) * nPixels;
            float fScale = fDstStd / fSrcStd;  // 20260330 ZJH 标准差比值
            for (size_t i = 0; i < nPixels; ++i) {
                float fVal = vecResult[nBase + i];
                fVal = (fVal - fSrcMean) * fScale + fDstMean;  // 20260330 ZJH 均值-方差匹配
                vecResult[nBase + i] = detail::clampf(fVal, 0.0f, 1.0f);  // 20260330 ZJH clamp 到 [0,1]
            }
        }

        return vecResult;
    }

private:
    // 20260330 ZJH addGaussianNoise — 添加高斯噪声
    static void addGaussianNoise(std::vector<float>& vecImage, float fSigma) {
        for (auto& val : vecImage) {
            val += detail::randNormal(0.0f, fSigma);
            val = detail::clampf(val, 0.0f, 1.0f);  // 20260330 ZJH clamp 防溢出
        }
    }

    // 20260330 ZJH flipHorizontal — 水平翻转
    static void flipHorizontal(std::vector<float>& vecImage, int nC, int nH, int nW) {
        for (int c = 0; c < nC; ++c) {
            size_t nBase = static_cast<size_t>(c) * static_cast<size_t>(nH * nW);
            for (int y = 0; y < nH; ++y) {
                // 20260330 ZJH 从两端向中间交换像素
                for (int x = 0; x < nW / 2; ++x) {
                    size_t nIdx1 = nBase + static_cast<size_t>(y * nW + x);
                    size_t nIdx2 = nBase + static_cast<size_t>(y * nW + (nW - 1 - x));
                    std::swap(vecImage[nIdx1], vecImage[nIdx2]);
                }
            }
        }
    }

    // 20260330 ZJH adjustBrightnessContrast — 亮度/对比度调整
    // fBrightness: 亮度偏移量（加到每个像素上）
    // fContrast: 对比度缩放因子（以 0.5 为中心缩放）
    static void adjustBrightnessContrast(std::vector<float>& vecImage,
                                         float fBrightness, float fContrast)
    {
        for (auto& val : vecImage) {
            // 20260330 ZJH 对比度调整：(val - 0.5) * contrast + 0.5 + brightness
            val = (val - 0.5f) * fContrast + 0.5f + fBrightness;
            val = detail::clampf(val, 0.0f, 1.0f);
        }
    }

    // 20260330 ZJH computeHomography — 从 4 对点计算单应性矩阵（DLT 方法）
    // src[4][2]: 源四角坐标
    // dst[4][2]: 目标四角坐标
    // 返回: 3×3 单应性矩阵 H（行优先存储）
    static std::array<float, 9> computeHomography(const float src[4][2], const float dst[4][2]) {
        // 20260330 ZJH 构建 8×9 矩阵 A，使得 Ah = 0
        // 每对点贡献两行:
        //   [-sx, -sy, -1, 0, 0, 0, dx*sx, dx*sy, dx]
        //   [0, 0, 0, -sx, -sy, -1, dy*sx, dy*sy, dy]
        // 但我们用简化方法：设 h33=1，变成 8×8 线性方程组 Ax = b

        // 20260330 ZJH 8×8 矩阵 A 和 8×1 向量 b
        float A[8][8] = {};
        float b[8] = {};

        for (int i = 0; i < 4; ++i) {
            float sx = src[i][0], sy = src[i][1];
            float dx = dst[i][0], dy = dst[i][1];

            // 20260330 ZJH 第 2i 行: sx*h1 + sy*h2 + h3 - dx*sx*h7 - dx*sy*h8 = dx
            int r0 = 2 * i;
            A[r0][0] = sx;  A[r0][1] = sy;  A[r0][2] = 1.0f;
            A[r0][3] = 0.0f; A[r0][4] = 0.0f; A[r0][5] = 0.0f;
            A[r0][6] = -dx * sx;  A[r0][7] = -dx * sy;
            b[r0] = dx;

            // 20260330 ZJH 第 2i+1 行: sx*h4 + sy*h5 + h6 - dy*sx*h7 - dy*sy*h8 = dy
            int r1 = 2 * i + 1;
            A[r1][0] = 0.0f; A[r1][1] = 0.0f; A[r1][2] = 0.0f;
            A[r1][3] = sx;  A[r1][4] = sy;  A[r1][5] = 1.0f;
            A[r1][6] = -dy * sx;  A[r1][7] = -dy * sy;
            b[r1] = dy;
        }

        // 20260330 ZJH 高斯消元法求解 8×8 线性方程组
        // 采用列主元高斯消元，数值稳定性好
        float augmented[8][9];  // 20260330 ZJH 增广矩阵 [A | b]
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) augmented[i][j] = A[i][j];
            augmented[i][8] = b[i];
        }

        // 20260330 ZJH 前向消元（带列主元选取）
        for (int col = 0; col < 8; ++col) {
            // 20260330 ZJH 选取当前列绝对值最大的行作为主元
            int nMaxRow = col;
            float fMaxVal = std::abs(augmented[col][col]);
            for (int row = col + 1; row < 8; ++row) {
                if (std::abs(augmented[row][col]) > fMaxVal) {
                    fMaxVal = std::abs(augmented[row][col]);
                    nMaxRow = row;
                }
            }

            // 20260330 ZJH 交换行
            if (nMaxRow != col) {
                for (int j = 0; j < 9; ++j) {
                    std::swap(augmented[col][j], augmented[nMaxRow][j]);
                }
            }

            // 20260330 ZJH 主元为零，矩阵奇异，返回单位矩阵
            if (std::abs(augmented[col][col]) < 1e-10f) {
                return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
            }

            // 20260330 ZJH 消元：将当前列下方元素归零
            for (int row = col + 1; row < 8; ++row) {
                float fFactor = augmented[row][col] / augmented[col][col];
                for (int j = col; j < 9; ++j) {
                    augmented[row][j] -= fFactor * augmented[col][j];
                }
            }
        }

        // 20260330 ZJH 回代求解
        float x[8] = {};
        for (int i = 7; i >= 0; --i) {
            float fSum = augmented[i][8];
            for (int j = i + 1; j < 8; ++j) {
                fSum -= augmented[i][j] * x[j];
            }
            x[i] = fSum / augmented[i][i];
        }

        // 20260330 ZJH 组装 3×3 单应性矩阵（h33=1）
        // H = [h1 h2 h3; h4 h5 h6; h7 h8 1]
        return {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0f};
    }

    // 20260330 ZJH invertHomography — 求 3×3 单应性矩阵的逆
    // 使用代数余子式法（Cramer 规则）
    static std::array<float, 9> invertHomography(const std::array<float, 9>& H) {
        // 20260330 ZJH 3×3 矩阵行列式
        float fDet = H[0] * (H[4] * H[8] - H[5] * H[7])
                   - H[1] * (H[3] * H[8] - H[5] * H[6])
                   + H[2] * (H[3] * H[7] - H[4] * H[6]);

        if (std::abs(fDet) < 1e-10f) {
            // 20260330 ZJH 奇异矩阵，返回单位矩阵
            return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        }

        float fInvDet = 1.0f / fDet;  // 20260330 ZJH 行列式的倒数

        // 20260330 ZJH 伴随矩阵转置 / 行列式
        std::array<float, 9> Hinv;
        Hinv[0] = fInvDet * (H[4] * H[8] - H[5] * H[7]);
        Hinv[1] = fInvDet * (H[2] * H[7] - H[1] * H[8]);
        Hinv[2] = fInvDet * (H[1] * H[5] - H[2] * H[4]);
        Hinv[3] = fInvDet * (H[5] * H[6] - H[3] * H[8]);
        Hinv[4] = fInvDet * (H[0] * H[8] - H[2] * H[6]);
        Hinv[5] = fInvDet * (H[2] * H[3] - H[0] * H[5]);
        Hinv[6] = fInvDet * (H[3] * H[7] - H[4] * H[6]);
        Hinv[7] = fInvDet * (H[1] * H[6] - H[0] * H[7]);
        Hinv[8] = fInvDet * (H[0] * H[4] - H[1] * H[3]);

        return Hinv;
    }
};

// =========================================================================
// 20260330 ZJH 策略3: 简化 GAN 合成（DCGAN）
// =========================================================================

// 20260330 ZJH DefectGAN 合成器
// 训练一个 DCGAN 风格的生成器，从随机噪声生成逼真的缺陷图像
// 适用于缺陷样本数 > 50 的场景（少于 50 张 GAN 难以收敛）
// 内部使用简化的 Generator/Discriminator 架构（与 om.engine.gan 类似但独立实现）
class DefectGANSynthesizer {
public:
    // 20260330 ZJH 构造函数
    // nLatentDim: 潜在空间维度（噪声向量长度），默认 64
    DefectGANSynthesizer(int nLatentDim = 64)
        : m_nLatentDim(nLatentDim), m_bTrained(false),
          m_nImgC(0), m_nImgH(0), m_nImgW(0)
    {}

    // 20260330 ZJH train — 训练 GAN 生成器
    // 使用缺陷图像训练 DCGAN，训练完成后可用 generate() 生成新图像
    // vecDefectImages: 缺陷图像列表，每张为 [C, H, W] CHW 格式
    // nC, nH, nW: 通道/高度/宽度
    // nEpochs: 训练轮数
    void train(const std::vector<std::vector<float>>& vecDefectImages,
               int nC, int nH, int nW, int nEpochs = 200)
    {
        if (vecDefectImages.empty()) {
            std::cerr << "[DefectGANSynthesizer] ERROR: empty training set" << std::endl;
            return;
        }

        m_nImgC = nC;
        m_nImgH = nH;
        m_nImgW = nW;
        size_t nPixels = static_cast<size_t>(nC * nH * nW);

        // 20260330 ZJH 初始化生成器和判别器参数
        // Generator: z(latentDim) -> FC -> reshape -> 逐层上采样
        // 简化实现：使用单层全连接映射 latentDim -> nPixels
        // 生产环境应使用 om.engine.gan 中的多层 DCGAN 架构
        size_t nGenParams = static_cast<size_t>(m_nLatentDim) * nPixels + nPixels;  // 20260330 ZJH W + bias
        m_vecGenWeights.resize(static_cast<size_t>(m_nLatentDim) * nPixels);
        m_vecGenBias.resize(nPixels);

        // 20260330 ZJH Xavier 初始化
        float fGenScale = std::sqrt(2.0f / static_cast<float>(m_nLatentDim + nPixels));
        for (auto& w : m_vecGenWeights) w = detail::randNormal(0.0f, fGenScale);
        for (auto& b : m_vecGenBias) b = 0.0f;

        // 20260330 ZJH 判别器: nPixels -> FC -> 1 (sigmoid)
        m_vecDiscWeights.resize(nPixels);
        m_vecDiscBias.resize(1, 0.0f);
        float fDiscScale = std::sqrt(2.0f / static_cast<float>(nPixels));
        for (auto& w : m_vecDiscWeights) w = detail::randNormal(0.0f, fDiscScale);

        // 20260330 ZJH 训练循环
        float fLrG = 0.0002f;  // 20260330 ZJH 生成器学习率
        float fLrD = 0.0002f;  // 20260330 ZJH 判别器学习率
        int nDataSize = static_cast<int>(vecDefectImages.size());

        std::cout << "[DefectGANSynthesizer] Training on " << nDataSize
                  << " images for " << nEpochs << " epochs..." << std::endl;

        for (int epoch = 0; epoch < nEpochs; ++epoch) {
            float fDLossSum = 0.0f;  // 20260330 ZJH 判别器损失累计
            float fGLossSum = 0.0f;  // 20260330 ZJH 生成器损失累计

            for (int i = 0; i < nDataSize; ++i) {
                const auto& vecReal = vecDefectImages[static_cast<size_t>(i)];

                // 20260330 ZJH === 训练判别器 ===
                // 目标：D(real) -> 1, D(G(z)) -> 0

                // 20260330 ZJH 生成随机噪声向量 z
                std::vector<float> vecZ(static_cast<size_t>(m_nLatentDim));
                for (auto& z : vecZ) z = detail::randNormal(0.0f, 1.0f);

                // 20260330 ZJH 生成器前向：G(z) = sigmoid(W_g * z + b_g)
                std::vector<float> vecFake(nPixels, 0.0f);
                for (size_t p = 0; p < nPixels; ++p) {
                    float fSum = m_vecGenBias[p];
                    for (int j = 0; j < m_nLatentDim; ++j) {
                        fSum += m_vecGenWeights[p * static_cast<size_t>(m_nLatentDim) + static_cast<size_t>(j)] * vecZ[static_cast<size_t>(j)];
                    }
                    vecFake[p] = sigmoid(fSum);  // 20260330 ZJH sigmoid 激活输出 [0,1]
                }

                // 20260330 ZJH 判别器前向（真实图像）：D(real) = sigmoid(w_d · real + b_d)
                float fDReal = 0.0f;
                for (size_t p = 0; p < nPixels; ++p) {
                    fDReal += m_vecDiscWeights[p] * vecReal[p];
                }
                fDReal = sigmoid(fDReal + m_vecDiscBias[0]);

                // 20260330 ZJH 判别器前向（生成图像）
                float fDFake = 0.0f;
                for (size_t p = 0; p < nPixels; ++p) {
                    fDFake += m_vecDiscWeights[p] * vecFake[p];
                }
                fDFake = sigmoid(fDFake + m_vecDiscBias[0]);

                // 20260330 ZJH 判别器损失：-[log(D(real)) + log(1 - D(fake))]
                float fDLoss = -(std::log(fDReal + 1e-7f) + std::log(1.0f - fDFake + 1e-7f));
                fDLossSum += fDLoss;

                // 20260330 ZJH 判别器梯度更新
                // dL/dw_d = -(1-D(real))*real + D(fake)*fake
                float fGradDReal = -(1.0f - fDReal);  // 20260330 ZJH d(-log(D))/dD * dD/dpre
                float fGradDFake = fDFake;              // 20260330 ZJH d(-log(1-D))/dD * dD/dpre
                for (size_t p = 0; p < nPixels; ++p) {
                    m_vecDiscWeights[p] -= fLrD * (fGradDReal * vecReal[p] + fGradDFake * vecFake[p]);
                }
                m_vecDiscBias[0] -= fLrD * (fGradDReal + fGradDFake);

                // 20260330 ZJH === 训练生成器 ===
                // 目标：D(G(z)) -> 1（欺骗判别器）

                // 20260330 ZJH 重新生成噪声和图像
                for (auto& z : vecZ) z = detail::randNormal(0.0f, 1.0f);
                for (size_t p = 0; p < nPixels; ++p) {
                    float fSum = m_vecGenBias[p];
                    for (int j = 0; j < m_nLatentDim; ++j) {
                        fSum += m_vecGenWeights[p * static_cast<size_t>(m_nLatentDim) + static_cast<size_t>(j)] * vecZ[static_cast<size_t>(j)];
                    }
                    vecFake[p] = sigmoid(fSum);
                }

                // 20260330 ZJH 判别器对生成图像打分
                fDFake = 0.0f;
                for (size_t p = 0; p < nPixels; ++p) {
                    fDFake += m_vecDiscWeights[p] * vecFake[p];
                }
                fDFake = sigmoid(fDFake + m_vecDiscBias[0]);

                // 20260330 ZJH 生成器损失：-log(D(G(z)))
                float fGLoss = -std::log(fDFake + 1e-7f);
                fGLossSum += fGLoss;

                // 20260330 ZJH 生成器梯度：反向传播通过 D 和 G
                // dL/dG = -(1-D(fake)) * w_d
                // dL/dw_g = dL/dG * dG/dw_g（sigmoid 导数 * z）
                float fGradG = -(1.0f - fDFake);  // 20260330 ZJH dL/d(pre_D) 对 D 输入的梯度
                for (size_t p = 0; p < nPixels; ++p) {
                    // 20260330 ZJH dL/d(fake_p) = fGradG * w_d[p]
                    float fGradFakeP = fGradG * m_vecDiscWeights[p];
                    // 20260330 ZJH sigmoid 导数: fake_p * (1 - fake_p)
                    float fSigDeriv = vecFake[p] * (1.0f - vecFake[p]);
                    // 20260330 ZJH dL/d(pre_G_p) = fGradFakeP * fSigDeriv
                    float fGradPreG = fGradFakeP * fSigDeriv;

                    // 20260330 ZJH 更新生成器权重
                    for (int j = 0; j < m_nLatentDim; ++j) {
                        m_vecGenWeights[p * static_cast<size_t>(m_nLatentDim) + static_cast<size_t>(j)]
                            -= fLrG * fGradPreG * vecZ[static_cast<size_t>(j)];
                    }
                    m_vecGenBias[p] -= fLrG * fGradPreG;
                }
            }

            // 20260330 ZJH 每 50 个 epoch 打印一次训练进度
            if ((epoch + 1) % 50 == 0 || epoch == 0) {
                std::cout << "[DefectGANSynthesizer] Epoch " << (epoch + 1) << "/" << nEpochs
                          << " D_loss=" << (fDLossSum / static_cast<float>(nDataSize))
                          << " G_loss=" << (fGLossSum / static_cast<float>(nDataSize))
                          << std::endl;
            }
        }

        m_bTrained = true;
        std::cout << "[DefectGANSynthesizer] Training complete." << std::endl;
    }

    // 20260330 ZJH generate — 生成新的缺陷图像
    // nCount: 要生成的图像数量
    // 返回: 生成的图像列表，每张为 [C, H, W]
    std::vector<std::vector<float>> generate(int nCount) {
        std::vector<std::vector<float>> vecResults;
        if (!m_bTrained) {
            std::cerr << "[DefectGANSynthesizer] ERROR: model not trained" << std::endl;
            return vecResults;
        }

        size_t nPixels = static_cast<size_t>(m_nImgC * m_nImgH * m_nImgW);
        vecResults.reserve(static_cast<size_t>(nCount));

        for (int i = 0; i < nCount; ++i) {
            // 20260330 ZJH 生成随机噪声
            std::vector<float> vecZ(static_cast<size_t>(m_nLatentDim));
            for (auto& z : vecZ) z = detail::randNormal(0.0f, 1.0f);

            // 20260330 ZJH 生成器前向
            std::vector<float> vecImg(nPixels, 0.0f);
            for (size_t p = 0; p < nPixels; ++p) {
                float fSum = m_vecGenBias[p];
                for (int j = 0; j < m_nLatentDim; ++j) {
                    fSum += m_vecGenWeights[p * static_cast<size_t>(m_nLatentDim) + static_cast<size_t>(j)] * vecZ[static_cast<size_t>(j)];
                }
                vecImg[p] = sigmoid(fSum);
            }

            vecResults.push_back(std::move(vecImg));
        }

        return vecResults;
    }

    // 20260330 ZJH isTrained — 检查 GAN 是否已完成训练
    bool isTrained() const { return m_bTrained; }

private:
    int m_nLatentDim;                       // 20260330 ZJH 潜在空间维度
    bool m_bTrained;                        // 20260330 ZJH 是否已训练
    int m_nImgC, m_nImgH, m_nImgW;         // 20260330 ZJH 图像尺寸
    std::vector<float> m_vecGenWeights;     // 20260330 ZJH 生成器权重 [nPixels, latentDim]
    std::vector<float> m_vecGenBias;        // 20260330 ZJH 生成器偏置 [nPixels]
    std::vector<float> m_vecDiscWeights;    // 20260330 ZJH 判别器权重 [nPixels]
    std::vector<float> m_vecDiscBias;       // 20260330 ZJH 判别器偏置 [1]

    // 20260330 ZJH sigmoid 激活函数: 1 / (1 + exp(-x))
    static float sigmoid(float x) {
        // 20260330 ZJH clamp 输入防止 exp 溢出
        x = detail::clampf(x, -20.0f, 20.0f);
        return 1.0f / (1.0f + std::exp(-x));
    }
};

// =========================================================================
// 20260330 ZJH 统一合成管线
// =========================================================================

// 20260330 ZJH 数据合成管线 — 一键自动合成
// 自动选择最佳策略组合，从少量缺陷样本生成大量训练数据
// 对标 Keyence AI 图像生成功能（其核心卖点之一）
class DataSynthesisPipeline {
public:
    // 20260330 ZJH 合成结果结构体
    struct SynthesisResult {
        std::vector<std::vector<float>> vecImages;        // 20260330 ZJH 合成图像列表 [C, H, W]
        std::vector<std::vector<BBox>> vecAnnotations;    // 20260330 ZJH 对应的标注列表
        int nOriginalCount = 0;                           // 20260330 ZJH 原始缺陷样本数
        int nSynthesizedCount = 0;                        // 20260330 ZJH 合成生成数
        int nCopyPasteCount = 0;                          // 20260330 ZJH CopyPaste 策略贡献数
        int nAugmentCount = 0;                            // 20260330 ZJH 增强策略贡献数
        int nGanCount = 0;                                // 20260330 ZJH GAN 策略贡献数
    };

    // 20260330 ZJH autoSynthesize — 一键合成
    // 自动策略选择逻辑:
    //   1. 始终执行 CopyPaste 合成（效果最稳定，对样本数无要求）
    //   2. 始终执行几何+光度增强（零成本增加多样性）
    //   3. 当缺陷样本 > 50 张时启用 GAN 合成（需要足够数据训练 GAN）
    //   4. 按比例分配各策略的生成量，直到达到 nTargetTotal
    //
    // vecNormalImages: 正常（无缺陷）图像列表 [C, H, W]
    // vecDefectImages: 缺陷图像列表 [C, H, W]
    // vecDefectAnnotations: 每张缺陷图像的标注框列表
    // nC, nH, nW: 图像尺寸
    // nTargetTotal: 目标总样本数（含原始 + 合成）
    // 返回: SynthesisResult 包含所有合成图像和统计信息
    static SynthesisResult autoSynthesize(
        const std::vector<std::vector<float>>& vecNormalImages,
        const std::vector<std::vector<float>>& vecDefectImages,
        const std::vector<std::vector<BBox>>& vecDefectAnnotations,
        int nC, int nH, int nW,
        int nTargetTotal = 500)
    {
        SynthesisResult result;
        result.nOriginalCount = static_cast<int>(vecDefectImages.size());

        // 20260330 ZJH 输入验证
        if (vecDefectImages.empty()) {
            std::cerr << "[DataSynthesisPipeline] WARNING: no defect images provided" << std::endl;
            return result;
        }

        // 20260330 ZJH nTargetTotal 直接作为需要生成的数量（不再减去已有数量）
        // 用户指定"生成 N 张"就是新增 N 张，而非总量达到 N
        int nNeeded = std::max(1, nTargetTotal);  // 20260330 ZJH 至少生成 1 张

        std::cout << "[DataSynthesisPipeline] Need to synthesize " << nNeeded
                  << " images (have " << vecDefectImages.size()
                  << ", target " << nTargetTotal << ")" << std::endl;

        // 20260330 ZJH 策略分配比例
        bool bUseGAN = (vecDefectImages.size() >= 50);  // 20260330 ZJH GAN 需要 >=50 张
        float fCopyPasteRatio, fAugmentRatio, fGanRatio;

        if (bUseGAN) {
            // 20260330 ZJH 有足够数据时：40% CopyPaste + 30% 增强 + 30% GAN
            fCopyPasteRatio = 0.4f;
            fAugmentRatio = 0.3f;
            fGanRatio = 0.3f;
        } else {
            // 20260330 ZJH 数据不足时：60% CopyPaste + 40% 增强
            fCopyPasteRatio = 0.6f;
            fAugmentRatio = 0.4f;
            fGanRatio = 0.0f;
        }

        int nCopyPasteNeeded = static_cast<int>(std::ceil(nNeeded * fCopyPasteRatio));
        int nAugmentNeeded = static_cast<int>(std::ceil(nNeeded * fAugmentRatio));
        int nGanNeeded = bUseGAN ? (nNeeded - nCopyPasteNeeded - nAugmentNeeded) : 0;
        if (nGanNeeded < 0) nGanNeeded = 0;  // 20260330 ZJH 防止负数

        // 20260330 ZJH ====== 策略1: CopyPaste 合成 ======
        if (nCopyPasteNeeded > 0 && !vecNormalImages.empty() && !vecDefectAnnotations.empty()) {
            std::cout << "[DataSynthesisPipeline] CopyPaste: generating " << nCopyPasteNeeded << " images..." << std::endl;

            // 20260330 ZJH 从缺陷图像中提取所有 patch
            std::vector<DefectCopyPaste::DefectPatch> vecAllPatches;
            for (size_t i = 0; i < vecDefectImages.size() && i < vecDefectAnnotations.size(); ++i) {
                auto vecPatches = DefectCopyPaste::extractDefects(
                    vecDefectImages[i], nC, nH, nW, vecDefectAnnotations[i]);
                vecAllPatches.insert(vecAllPatches.end(),
                                     std::make_move_iterator(vecPatches.begin()),
                                     std::make_move_iterator(vecPatches.end()));
            }

            if (!vecAllPatches.empty()) {
                // 20260330 ZJH 计算每个 patch 需要生成多少张
                CopyPasteConfig cpConfig;
                int nPerPatch = std::max(1, nCopyPasteNeeded / static_cast<int>(vecAllPatches.size()));
                cpConfig.nNumSynthPerDefect = nPerPatch;

                auto vecSynth = DefectCopyPaste::batchSynthesize(
                    vecNormalImages, vecAllPatches, nC, nH, nW, cpConfig);

                // 20260330 ZJH 截取所需数量
                int nTake = std::min(nCopyPasteNeeded, static_cast<int>(vecSynth.size()));
                for (int i = 0; i < nTake; ++i) {
                    result.vecImages.push_back(std::move(vecSynth[static_cast<size_t>(i)].first));
                    result.vecAnnotations.push_back(std::move(vecSynth[static_cast<size_t>(i)].second));
                }
                result.nCopyPasteCount = nTake;
            }
        }

        // 20260330 ZJH ====== 策略2: 几何+光度增强 ======
        if (nAugmentNeeded > 0) {
            std::cout << "[DataSynthesisPipeline] Augmentation: generating " << nAugmentNeeded << " images..." << std::endl;

            AugSynthConfig augConfig;
            // 20260330 ZJH 计算每张源图需要的变体数
            int nPerImage = std::max(1, nAugmentNeeded / static_cast<int>(vecDefectImages.size()));
            augConfig.nNumVariants = nPerImage;

            int nGenerated = 0;
            for (size_t i = 0; i < vecDefectImages.size() && nGenerated < nAugmentNeeded; ++i) {
                auto vecVariants = AugmentSynthesizer::generateVariants(
                    vecDefectImages[i], nC, nH, nW, augConfig, vecNormalImages);

                for (auto& variant : vecVariants) {
                    if (nGenerated >= nAugmentNeeded) break;

                    result.vecImages.push_back(std::move(variant));

                    // 20260330 ZJH 增强变体继承原始标注（几何变换不改变缺陷位置的大致范围）
                    if (i < vecDefectAnnotations.size()) {
                        result.vecAnnotations.push_back(vecDefectAnnotations[i]);
                    } else {
                        result.vecAnnotations.push_back({});
                    }
                    ++nGenerated;
                }
            }
            result.nAugmentCount = nGenerated;
        }

        // 20260330 ZJH ====== 策略3: GAN 合成 ======
        if (nGanNeeded > 0 && bUseGAN) {
            std::cout << "[DataSynthesisPipeline] GAN: training and generating " << nGanNeeded << " images..." << std::endl;

            DefectGANSynthesizer gan(64);  // 20260330 ZJH 64 维潜在空间
            // 20260330 ZJH 训练 GAN（epoch 数根据数据量调整）
            int nEpochs = std::min(200, std::max(50, 5000 / static_cast<int>(vecDefectImages.size())));
            gan.train(vecDefectImages, nC, nH, nW, nEpochs);

            if (gan.isTrained()) {
                auto vecGenerated = gan.generate(nGanNeeded);
                for (auto& img : vecGenerated) {
                    result.vecImages.push_back(std::move(img));
                    result.vecAnnotations.push_back({});  // 20260330 ZJH GAN 生成的图像无精确标注
                }
                result.nGanCount = static_cast<int>(vecGenerated.size());
            }
        }

        result.nSynthesizedCount = static_cast<int>(result.vecImages.size());

        std::cout << "[DataSynthesisPipeline] Synthesis complete: "
                  << result.nSynthesizedCount << " images generated ("
                  << "CopyPaste=" << result.nCopyPasteCount
                  << " Augment=" << result.nAugmentCount
                  << " GAN=" << result.nGanCount << ")" << std::endl;

        return result;
    }
};

}  // namespace om
