// 20260320 ZJH 数据管线模块 — Phase 5
// Dataset/DataLoader/数据增强/图像加载，支持分类/检测/分割任务
module;

#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <functional>

export module om.engine.data_pipeline;

import om.engine.tensor;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260324 ZJH 图像加载：纯 C++ BMP 加载和基础操作
// GUI 应用中使用 Qt6 QImage 加载更多格式，此处提供引擎层独立后备方案
// =========================================================

// 20260320 ZJH RawImage — 原始图像数据结构
// 存储解码后的像素数据（float 归一化到 [0,1]）
struct RawImage {
    std::vector<float> vecData;  // 20260320 ZJH 像素数据 [C, H, W]，CHW 排列，float [0,1]
    int nWidth = 0;              // 20260320 ZJH 图像宽度
    int nHeight = 0;             // 20260320 ZJH 图像高度
    int nChannels = 0;           // 20260320 ZJH 通道数（1=灰度，3=RGB）

    // 20260320 ZJH 是否有效（已加载数据）
    bool isValid() const { return !vecData.empty() && nWidth > 0 && nHeight > 0; }
};

// 20260320 ZJH loadBMP — 简易 BMP 加载器（不依赖外部库）
// 支持 24 位无压缩 BMP
// 返回: RawImage 结构，CHW 格式，float [0,1]
RawImage loadBMP(const std::string& strPath) {
    RawImage img;
    std::ifstream file(strPath, std::ios::binary);
    if (!file.is_open()) return img;  // 20260320 ZJH 打开失败返回空

    // 20260320 ZJH 读取 BMP 文件头
    char header[54];
    file.read(header, 54);
    if (header[0] != 'B' || header[1] != 'M') return img;  // 20260320 ZJH 验证 BMP 魔数

    int nDataOffset = *reinterpret_cast<int*>(&header[10]);  // 20260320 ZJH 像素数据偏移
    img.nWidth = *reinterpret_cast<int*>(&header[18]);       // 20260320 ZJH 宽度
    img.nHeight = *reinterpret_cast<int*>(&header[22]);      // 20260320 ZJH 高度
    int nBpp = *reinterpret_cast<short*>(&header[28]);       // 20260320 ZJH 每像素位数

    if (nBpp != 24 && nBpp != 8) return img;  // 20260320 ZJH 仅支持 8 位灰度和 24 位 RGB

    img.nChannels = (nBpp == 24) ? 3 : 1;
    bool bFlipped = img.nHeight > 0;  // 20260320 ZJH BMP 通常底到顶存储
    if (img.nHeight < 0) img.nHeight = -img.nHeight;

    // 20260320 ZJH 每行字节数（4 字节对齐）
    int nRowBytes = ((img.nWidth * (nBpp / 8) + 3) / 4) * 4;

    file.seekg(nDataOffset);
    std::vector<unsigned char> vecRawPixels(static_cast<size_t>(nRowBytes * img.nHeight));
    file.read(reinterpret_cast<char*>(vecRawPixels.data()),
              static_cast<std::streamsize>(vecRawPixels.size()));

    // 20260320 ZJH 转换为 CHW float [0,1]
    img.vecData.resize(static_cast<size_t>(img.nChannels * img.nHeight * img.nWidth));
    for (int y = 0; y < img.nHeight; ++y) {
        int nSrcY = bFlipped ? (img.nHeight - 1 - y) : y;  // 20260320 ZJH 处理翻转
        for (int x = 0; x < img.nWidth; ++x) {
            int nSrcIdx = nSrcY * nRowBytes + x * (nBpp / 8);
            if (img.nChannels == 3) {
                // 20260320 ZJH BMP 存 BGR，转为 RGB CHW
                float fB = vecRawPixels[static_cast<size_t>(nSrcIdx)] / 255.0f;
                float fG = vecRawPixels[static_cast<size_t>(nSrcIdx + 1)] / 255.0f;
                float fR = vecRawPixels[static_cast<size_t>(nSrcIdx + 2)] / 255.0f;
                img.vecData[static_cast<size_t>(0 * img.nHeight * img.nWidth + y * img.nWidth + x)] = fR;
                img.vecData[static_cast<size_t>(1 * img.nHeight * img.nWidth + y * img.nWidth + x)] = fG;
                img.vecData[static_cast<size_t>(2 * img.nHeight * img.nWidth + y * img.nWidth + x)] = fB;
            } else {
                img.vecData[static_cast<size_t>(y * img.nWidth + x)] =
                    vecRawPixels[static_cast<size_t>(nSrcIdx)] / 255.0f;
            }
        }
    }

    return img;
}

// =========================================================
// 数据增强
// =========================================================

// 20260320 ZJH 前向声明 resizeImage（augmentImage 的随机缩放需要调用）
std::vector<float> resizeImage(const std::vector<float>& vecSrc,
                                int nC, int nSrcH, int nSrcW,
                                int nDstH, int nDstW);

// 20260320 ZJH AugmentConfig — 完整工业级数据增强配置
struct AugmentConfig {
    // ---- 几何变换 ----
    bool bRandomHFlip = true;          // 20260320 ZJH 随机水平翻转（镜像）
    bool bRandomVFlip = false;         // 20260320 ZJH 随机垂直翻转
    bool bRandomRotate90 = false;      // 20260320 ZJH 随机 90°/180°/270° 旋转
    bool bRandomRotate = false;        // 20260320 ZJH 随机任意角度旋转
    float fRotateRange = 15.0f;        // 20260320 ZJH 旋转角度范围 [-range, +range] 度
    bool bRandomScale = false;         // 20260320 ZJH 随机缩放
    float fScaleMin = 0.8f;            // 20260320 ZJH 最小缩放比
    float fScaleMax = 1.2f;            // 20260320 ZJH 最大缩放比
    bool bRandomTranslate = false;     // 20260320 ZJH 随机平移
    float fTranslateRange = 0.1f;      // 20260320 ZJH 平移比例 [0,1]
    bool bRandomShear = false;         // 20260320 ZJH 随机错切
    float fShearRange = 5.0f;          // 20260320 ZJH 错切角度范围（度）
    bool bRandomCrop = false;          // 20260320 ZJH 随机裁剪
    float fCropRatio = 0.9f;           // 20260320 ZJH 裁剪保留比例

    // ---- 颜色/灰度变换 ----
    bool bColorJitter = false;         // 20260320 ZJH 颜色抖动
    float fJitterBrightness = 0.1f;    // 20260320 ZJH 亮度变化幅度
    float fJitterContrast = 0.1f;      // 20260320 ZJH 对比度变化幅度
    float fJitterSaturation = 0.1f;    // 20260320 ZJH 饱和度变化幅度
    float fJitterHue = 0.02f;          // 20260320 ZJH 色调变化幅度
    bool bGammaCorrection = false;     // 20260320 ZJH 伽马校正
    float fGammaMin = 0.7f;            // 20260320 ZJH 最小伽马值
    float fGammaMax = 1.5f;            // 20260320 ZJH 最大伽马值
    bool bHistogramEQ = false;         // 20260320 ZJH 直方图均衡化
    bool bCLAHE = false;              // 20260320 ZJH 自适应直方图均衡（CLAHE）
    bool bInvert = false;              // 20260320 ZJH 随机反色
    float fInvertProb = 0.1f;          // 20260320 ZJH 反色概率
    bool bGrayscale = false;           // 20260320 ZJH 随机转灰度（RGB→Gray→RGB）
    float fGrayscaleProb = 0.1f;       // 20260320 ZJH 转灰度概率

    // ---- 噪声/模糊 ----
    bool bGaussianNoise = false;       // 20260320 ZJH 高斯噪声
    float fNoiseStd = 0.01f;           // 20260320 ZJH 噪声标准差
    bool bGaussianBlur = false;        // 20260320 ZJH 高斯模糊
    int nBlurKernelSize = 3;           // 20260320 ZJH 模糊核大小（奇数）
    float fBlurSigma = 1.0f;           // 20260320 ZJH 模糊 sigma
    bool bSaltPepper = false;          // 20260320 ZJH 椒盐噪声
    float fSaltPepperProb = 0.01f;     // 20260320 ZJH 椒盐概率

    // ---- 遮挡/混合（高级） ----
    bool bCutOut = false;              // 20260320 ZJH CutOut 随机遮挡
    int nCutOutSize = 8;               // 20260320 ZJH CutOut 遮挡区域大小
    int nCutOutCount = 1;              // 20260320 ZJH CutOut 遮挡数量
    bool bRandomErasing = false;       // 20260320 ZJH 随机擦除（Random Erasing）
    float fErasingProb = 0.5f;         // 20260320 ZJH 擦除概率
    float fErasingMinArea = 0.02f;     // 20260320 ZJH 最小擦除面积比
    float fErasingMaxArea = 0.33f;     // 20260320 ZJH 最大擦除面积比

    // ---- 归一化 ----
    bool bNormalize = true;            // 20260320 ZJH 是否归一化
    float fMeanR = 0.5f;
    float fMeanG = 0.5f;
    float fMeanB = 0.5f;
    float fStdR = 0.5f;
    float fStdG = 0.5f;
    float fStdB = 0.5f;
};

// 20260320 ZJH augmentImage — 完整工业级数据增强
// data: 输入图像数据 CHW 格式，float [0,1]，inplace 修改
void augmentImage(std::vector<float>& data, int nC, int nH, int nW,
                  const AugmentConfig& config) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    int nSpatial = nH * nW;

    // ====== 几何变换 ======

    // 20260320 ZJH 随机水平翻转（镜像）
    if (config.bRandomHFlip && dist01(gen) > 0.5f) {
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW / 2; ++x) {
                    size_t i1 = static_cast<size_t>(c * nSpatial + y * nW + x);
                    size_t i2 = static_cast<size_t>(c * nSpatial + y * nW + (nW - 1 - x));
                    std::swap(data[i1], data[i2]);
                }
    }

    // 20260320 ZJH 随机垂直翻转
    if (config.bRandomVFlip && dist01(gen) > 0.5f) {
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH / 2; ++y)
                for (int x = 0; x < nW; ++x) {
                    size_t i1 = static_cast<size_t>(c * nSpatial + y * nW + x);
                    size_t i2 = static_cast<size_t>(c * nSpatial + (nH - 1 - y) * nW + x);
                    std::swap(data[i1], data[i2]);
                }
    }

    // 20260320 ZJH 随机 90°/180°/270° 旋转
    if (config.bRandomRotate90 && nH == nW) {
        int nRot = static_cast<int>(dist01(gen) * 4.0f) % 4;  // 0/1/2/3
        if (nRot > 0) {
            std::vector<float> tmp(data.size());
            for (int r = 0; r < nRot; ++r) {
                for (int c = 0; c < nC; ++c)
                    for (int y = 0; y < nH; ++y)
                        for (int x = 0; x < nW; ++x) {
                            // 20260320 ZJH 顺时针 90°: (x,y) -> (y, W-1-x)
                            int ny = x, nx = nW - 1 - y;
                            tmp[static_cast<size_t>(c * nSpatial + ny * nW + nx)] =
                                data[static_cast<size_t>(c * nSpatial + y * nW + x)];
                        }
                data = tmp;
            }
        }
    }

    // 20260320 ZJH 随机任意角度旋转（双线性插值）
    if (config.bRandomRotate) {
        float fAngleDeg = (dist01(gen) * 2.0f - 1.0f) * config.fRotateRange;
        float fAngleRad = fAngleDeg * 3.14159265f / 180.0f;
        float fCos = std::cos(fAngleRad), fSin = std::sin(fAngleRad);
        float fCx = nW * 0.5f, fCy = nH * 0.5f;

        std::vector<float> rotated(data.size(), 0.0f);
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    // 20260320 ZJH 逆映射：从目标查源
                    float fSrcX = fCos * (x - fCx) + fSin * (y - fCy) + fCx;
                    float fSrcY = -fSin * (x - fCx) + fCos * (y - fCy) + fCy;
                    // 20260320 ZJH 双线性插值
                    int x0 = static_cast<int>(std::floor(fSrcX));
                    int y0 = static_cast<int>(std::floor(fSrcY));
                    float fx = fSrcX - x0, fy = fSrcY - y0;
                    auto getP = [&](int py, int px) -> float {
                        py = std::max(0, std::min(nH - 1, py));
                        px = std::max(0, std::min(nW - 1, px));
                        return data[static_cast<size_t>(c * nSpatial + py * nW + px)];
                    };
                    rotated[static_cast<size_t>(c * nSpatial + y * nW + x)] =
                        (1 - fy) * ((1 - fx) * getP(y0, x0) + fx * getP(y0, x0 + 1)) +
                        fy * ((1 - fx) * getP(y0 + 1, x0) + fx * getP(y0 + 1, x0 + 1));
                }
        data = rotated;
    }

    // 20260320 ZJH 随机缩放（通过 resize）
    if (config.bRandomScale) {
        float fScale = config.fScaleMin + dist01(gen) * (config.fScaleMax - config.fScaleMin);
        int nNewH = static_cast<int>(nH * fScale);
        int nNewW = static_cast<int>(nW * fScale);
        if (nNewH > 0 && nNewW > 0 && (nNewH != nH || nNewW != nW)) {
            auto scaled = resizeImage(data, nC, nH, nW, nNewH, nNewW);
            // 20260320 ZJH 中心裁剪或填充回原始尺寸
            data.assign(static_cast<size_t>(nC * nH * nW), 0.0f);
            int nOffY = (nNewH - nH) / 2, nOffX = (nNewW - nW) / 2;
            for (int c = 0; c < nC; ++c)
                for (int y = 0; y < nH; ++y)
                    for (int x = 0; x < nW; ++x) {
                        int sy = y + nOffY, sx = x + nOffX;
                        if (sy >= 0 && sy < nNewH && sx >= 0 && sx < nNewW)
                            data[static_cast<size_t>(c * nSpatial + y * nW + x)] =
                                scaled[static_cast<size_t>(c * nNewH * nNewW + sy * nNewW + sx)];
                    }
        }
    }

    // ====== 颜色/灰度变换 ======

    // 20260320 ZJH 亮度 + 对比度抖动
    if (config.bColorJitter) {
        float fBright = (dist01(gen) * 2.0f - 1.0f) * config.fJitterBrightness;
        float fContrast = 1.0f + (dist01(gen) * 2.0f - 1.0f) * config.fJitterContrast;
        for (auto& v : data) {
            v = fContrast * (v - 0.5f) + 0.5f + fBright;
            v = std::max(0.0f, std::min(1.0f, v));
        }
    }

    // 20260320 ZJH 伽马校正：out = in^gamma
    if (config.bGammaCorrection) {
        float fGamma = config.fGammaMin + dist01(gen) * (config.fGammaMax - config.fGammaMin);
        for (auto& v : data) {
            v = std::pow(std::max(0.0f, v), fGamma);
            v = std::min(1.0f, v);
        }
    }

    // 20260320 ZJH 直方图均衡化（逐通道）
    if (config.bHistogramEQ) {
        for (int c = 0; c < nC; ++c) {
            // 20260320 ZJH 计算直方图（256 bins）
            int hist[256] = {};
            for (int i = 0; i < nSpatial; ++i) {
                int bin = static_cast<int>(data[static_cast<size_t>(c * nSpatial + i)] * 255.0f);
                bin = std::max(0, std::min(255, bin));
                hist[bin]++;
            }
            // 20260320 ZJH CDF 累积
            int cdf[256];
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i) cdf[i] = cdf[i - 1] + hist[i];
            int nCdfMin = 0;
            for (int i = 0; i < 256; ++i) { if (cdf[i] > 0) { nCdfMin = cdf[i]; break; } }
            float fDenom = static_cast<float>(nSpatial - nCdfMin);
            if (fDenom < 1.0f) fDenom = 1.0f;
            // 20260320 ZJH 映射
            for (int i = 0; i < nSpatial; ++i) {
                int bin = static_cast<int>(data[static_cast<size_t>(c * nSpatial + i)] * 255.0f);
                bin = std::max(0, std::min(255, bin));
                data[static_cast<size_t>(c * nSpatial + i)] =
                    static_cast<float>(cdf[bin] - nCdfMin) / fDenom;
            }
        }
    }

    // 20260320 ZJH 随机反色
    if (config.bInvert && dist01(gen) < config.fInvertProb) {
        for (auto& v : data) v = 1.0f - v;
    }

    // 20260320 ZJH 随机转灰度（对 RGB 图像）
    if (config.bGrayscale && nC == 3 && dist01(gen) < config.fGrayscaleProb) {
        for (int i = 0; i < nSpatial; ++i) {
            float fGray = 0.299f * data[static_cast<size_t>(i)]
                        + 0.587f * data[static_cast<size_t>(nSpatial + i)]
                        + 0.114f * data[static_cast<size_t>(2 * nSpatial + i)];
            data[static_cast<size_t>(i)] = fGray;
            data[static_cast<size_t>(nSpatial + i)] = fGray;
            data[static_cast<size_t>(2 * nSpatial + i)] = fGray;
        }
    }

    // ====== 噪声/模糊 ======

    // 20260320 ZJH 高斯噪声
    if (config.bGaussianNoise) {
        std::normal_distribution<float> noiseDist(0.0f, config.fNoiseStd);
        for (auto& v : data) v = std::max(0.0f, std::min(1.0f, v + noiseDist(gen)));
    }

    // 20260320 ZJH 高斯模糊（3x3 或 5x5）
    if (config.bGaussianBlur) {
        int nK = config.nBlurKernelSize;
        if (nK % 2 == 0) nK++;
        int nR = nK / 2;
        float fSig = config.fBlurSigma;
        // 20260320 ZJH 生成高斯核
        std::vector<float> kernel(static_cast<size_t>(nK * nK));
        float fKernelSum = 0.0f;
        for (int ky = -nR; ky <= nR; ++ky)
            for (int kx = -nR; kx <= nR; ++kx) {
                float fVal = std::exp(-(kx * kx + ky * ky) / (2.0f * fSig * fSig));
                kernel[static_cast<size_t>((ky + nR) * nK + (kx + nR))] = fVal;
                fKernelSum += fVal;
            }
        for (auto& v : kernel) v /= fKernelSum;
        // 20260320 ZJH 逐通道卷积
        std::vector<float> blurred(data.size());
        for (int c = 0; c < nC; ++c)
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    float fSum = 0.0f;
                    for (int ky = -nR; ky <= nR; ++ky)
                        for (int kx = -nR; kx <= nR; ++kx) {
                            int sy = std::max(0, std::min(nH - 1, y + ky));
                            int sx = std::max(0, std::min(nW - 1, x + kx));
                            fSum += data[static_cast<size_t>(c * nSpatial + sy * nW + sx)]
                                  * kernel[static_cast<size_t>((ky + nR) * nK + (kx + nR))];
                        }
                    blurred[static_cast<size_t>(c * nSpatial + y * nW + x)] = fSum;
                }
        data = blurred;
    }

    // 20260320 ZJH 椒盐噪声
    if (config.bSaltPepper) {
        for (int i = 0; i < nC * nSpatial; ++i) {
            float r = dist01(gen);
            if (r < config.fSaltPepperProb * 0.5f) data[static_cast<size_t>(i)] = 0.0f;       // 椒
            else if (r < config.fSaltPepperProb) data[static_cast<size_t>(i)] = 1.0f;  // 盐
        }
    }

    // ====== 遮挡/擦除 ======

    // 20260320 ZJH CutOut 随机遮挡
    if (config.bCutOut) {
        for (int k = 0; k < config.nCutOutCount; ++k) {
            int nCx = static_cast<int>(dist01(gen) * nW);
            int nCy = static_cast<int>(dist01(gen) * nH);
            int nHalf = config.nCutOutSize / 2;
            for (int c = 0; c < nC; ++c)
                for (int y = std::max(0, nCy - nHalf); y < std::min(nH, nCy + nHalf); ++y)
                    for (int x = std::max(0, nCx - nHalf); x < std::min(nW, nCx + nHalf); ++x)
                        data[static_cast<size_t>(c * nSpatial + y * nW + x)] = 0.0f;
        }
    }

    // 20260320 ZJH Random Erasing（随机擦除）
    if (config.bRandomErasing && dist01(gen) < config.fErasingProb) {
        float fArea = nH * nW * (config.fErasingMinArea + dist01(gen) * (config.fErasingMaxArea - config.fErasingMinArea));
        float fRatio = 0.3f + dist01(gen) * 2.7f;  // 宽高比 [0.3, 3.0]
        int nErH = static_cast<int>(std::sqrt(fArea / fRatio));
        int nErW = static_cast<int>(std::sqrt(fArea * fRatio));
        nErH = std::min(nErH, nH); nErW = std::min(nErW, nW);
        int nY0 = static_cast<int>(dist01(gen) * (nH - nErH));
        int nX0 = static_cast<int>(dist01(gen) * (nW - nErW));
        std::normal_distribution<float> eraseDist(0.5f, 0.2f);
        for (int c = 0; c < nC; ++c)
            for (int y = nY0; y < nY0 + nErH; ++y)
                for (int x = nX0; x < nX0 + nErW; ++x)
                    data[static_cast<size_t>(c * nSpatial + y * nW + x)] = std::max(0.0f, std::min(1.0f, eraseDist(gen)));
    }

    // ====== 归一化（最后执行） ======
    if (config.bNormalize) {
        if (nC == 3) {
            float fMean[3] = {config.fMeanR, config.fMeanG, config.fMeanB};
            float fStd[3] = {config.fStdR, config.fStdG, config.fStdB};
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < nSpatial; ++i)
                    data[static_cast<size_t>(c * nSpatial + i)] =
                        (data[static_cast<size_t>(c * nSpatial + i)] - fMean[c]) / fStd[c];
        } else {
            for (auto& v : data) v = (v - config.fMeanR) / config.fStdR;
        }
    }
}

// 20260320 ZJH resizeImage — 简单最近邻缩放（CHW 格式）
// 将 [C, srcH, srcW] 缩放到 [C, dstH, dstW]
std::vector<float> resizeImage(const std::vector<float>& vecSrc,
                                int nC, int nSrcH, int nSrcW,
                                int nDstH, int nDstW) {
    std::vector<float> vecDst(static_cast<size_t>(nC * nDstH * nDstW));
    for (int c = 0; c < nC; ++c) {
        for (int dy = 0; dy < nDstH; ++dy) {
            int nSrcY = dy * nSrcH / nDstH;  // 20260320 ZJH 最近邻映射
            for (int dx = 0; dx < nDstW; ++dx) {
                int nSrcX = dx * nSrcW / nDstW;
                vecDst[static_cast<size_t>(c * nDstH * nDstW + dy * nDstW + dx)] =
                    vecSrc[static_cast<size_t>(c * nSrcH * nSrcW + nSrcY * nSrcW + nSrcX)];
            }
        }
    }
    return vecDst;
}

// =========================================================
// Dataset 基类和具体实现
// =========================================================

// 20260320 ZJH Dataset — 数据集抽象基类
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;                           // 20260320 ZJH 数据集大小
    virtual std::pair<Tensor, Tensor> getItem(size_t nIdx) = 0;  // 20260320 ZJH 获取 (input, target)
};

// 20260320 ZJH ImageClassificationDataset — 图像分类数据集
// 从文件夹结构加载：root/class_name/*.bmp|*.png|*.jpg
// 每个子文件夹名就是类别名，类别 ID 按字母排序
class ImageClassificationDataset : public Dataset {
public:
    // 20260320 ZJH 构造函数
    // strRootPath: 数据集根目录
    // nTargetH/nTargetW: 目标图像尺寸
    // nTargetC: 目标通道数（1=灰度, 3=RGB）
    // config: 数据增强配置
    ImageClassificationDataset(const std::string& strRootPath,
                                int nTargetH, int nTargetW, int nTargetC = 3,
                                AugmentConfig config = {})
        : m_strRootPath(strRootPath), m_nTargetH(nTargetH), m_nTargetW(nTargetW),
          m_nTargetC(nTargetC), m_config(config)
    {
        scanDirectory();  // 20260320 ZJH 扫描目录结构
    }

    size_t size() const override { return m_vecSamples.size(); }

    // 20260320 ZJH 获取第 nIdx 个样本
    // 返回: (image_tensor [C, H, W], label_tensor [numClasses] one-hot)
    std::pair<Tensor, Tensor> getItem(size_t nIdx) override {
        const auto& sample = m_vecSamples[nIdx];

        // 20260320 ZJH 加载图像
        RawImage img = loadBMP(sample.strPath);
        std::vector<float> vecData;
        int nC = 0, nH = 0, nW = 0;

        if (img.isValid()) {
            nC = img.nChannels;
            nH = img.nHeight;
            nW = img.nWidth;
            vecData = img.vecData;
        } else {
            // 20260320 ZJH 加载失败：返回黑色图像
            nC = m_nTargetC;
            nH = m_nTargetH;
            nW = m_nTargetW;
            vecData.resize(static_cast<size_t>(nC * nH * nW), 0.0f);
        }

        // 20260320 ZJH 缩放到目标尺寸
        if (nH != m_nTargetH || nW != m_nTargetW) {
            vecData = resizeImage(vecData, nC, nH, nW, m_nTargetH, m_nTargetW);
            nH = m_nTargetH;
            nW = m_nTargetW;
        }

        // 20260320 ZJH 通道数转换（RGB -> 灰度 或 灰度 -> RGB）
        if (nC == 3 && m_nTargetC == 1) {
            // 20260320 ZJH RGB 转灰度：0.299*R + 0.587*G + 0.114*B
            std::vector<float> vecGray(static_cast<size_t>(nH * nW));
            int nSpatial = nH * nW;
            for (int i = 0; i < nSpatial; ++i) {
                vecGray[static_cast<size_t>(i)] =
                    0.299f * vecData[static_cast<size_t>(i)] +
                    0.587f * vecData[static_cast<size_t>(nSpatial + i)] +
                    0.114f * vecData[static_cast<size_t>(2 * nSpatial + i)];
            }
            vecData = vecGray;
            nC = 1;
        } else if (nC == 1 && m_nTargetC == 3) {
            // 20260320 ZJH 灰度转 RGB：三通道复制
            int nSpatial = nH * nW;
            std::vector<float> vecRGB(static_cast<size_t>(3 * nSpatial));
            for (int i = 0; i < nSpatial; ++i) {
                vecRGB[static_cast<size_t>(i)] = vecData[static_cast<size_t>(i)];
                vecRGB[static_cast<size_t>(nSpatial + i)] = vecData[static_cast<size_t>(i)];
                vecRGB[static_cast<size_t>(2 * nSpatial + i)] = vecData[static_cast<size_t>(i)];
            }
            vecData = vecRGB;
            nC = 3;
        }

        // 20260320 ZJH 应用数据增强
        augmentImage(vecData, nC, nH, nW, m_config);

        // 20260320 ZJH 创建图像张量 [C, H, W]
        auto imgTensor = Tensor::fromData(vecData.data(), {nC, nH, nW});

        // 20260320 ZJH 创建 one-hot 标签张量 [numClasses]
        auto labelTensor = Tensor::zeros({static_cast<int>(m_vecClassNames.size())});
        labelTensor.mutableFloatDataPtr()[sample.nClassId] = 1.0f;

        return {imgTensor, labelTensor};
    }

    // 20260320 ZJH 获取类别数量
    int numClasses() const { return static_cast<int>(m_vecClassNames.size()); }

    // 20260320 ZJH 获取类别名列表
    const std::vector<std::string>& classNames() const { return m_vecClassNames; }

private:
    struct Sample {
        std::string strPath;  // 20260320 ZJH 文件路径
        int nClassId;         // 20260320 ZJH 类别 ID
    };

    // 20260320 ZJH 扫描目录结构，建立样本列表
    void scanDirectory() {
        namespace fs = std::filesystem;
        if (!fs::exists(m_strRootPath)) return;

        // 20260320 ZJH 收集子目录作为类别
        std::vector<std::string> vecDirs;
        for (const auto& entry : fs::directory_iterator(m_strRootPath)) {
            if (entry.is_directory()) {
                vecDirs.push_back(entry.path().filename().string());
            }
        }
        std::sort(vecDirs.begin(), vecDirs.end());  // 20260320 ZJH 按字母排序
        m_vecClassNames = vecDirs;

        // 20260320 ZJH 扫描每个类别目录下的图像文件
        for (int classId = 0; classId < static_cast<int>(vecDirs.size()); ++classId) {
            std::string strClassDir = m_strRootPath + "/" + vecDirs[static_cast<size_t>(classId)];
            for (const auto& entry : fs::directory_iterator(strClassDir)) {
                if (!entry.is_regular_file()) continue;
                std::string strExt = entry.path().extension().string();
                // 20260320 ZJH 支持的图像格式
                if (strExt == ".bmp" || strExt == ".BMP" ||
                    strExt == ".png" || strExt == ".PNG" ||
                    strExt == ".jpg" || strExt == ".JPG" ||
                    strExt == ".jpeg" || strExt == ".JPEG") {
                    m_vecSamples.push_back({entry.path().string(), classId});
                }
            }
        }
    }

    std::string m_strRootPath;              // 20260320 ZJH 数据集根目录
    int m_nTargetH;                         // 20260320 ZJH 目标高度
    int m_nTargetW;                         // 20260320 ZJH 目标宽度
    int m_nTargetC;                         // 20260320 ZJH 目标通道数
    AugmentConfig m_config;                 // 20260320 ZJH 数据增强配置
    std::vector<Sample> m_vecSamples;       // 20260320 ZJH 样本列表
    std::vector<std::string> m_vecClassNames;  // 20260320 ZJH 类别名列表
};

// =========================================================
// DataLoader — 数据加载器
// =========================================================

// 20260320 ZJH DataLoader — 批量数据加载器
// 支持 shuffle、批量加载，返回 (batchInput, batchTarget)
class DataLoader {
public:
    // 20260320 ZJH 构造函数
    // pDataset: 数据集指针（不拥有所有权）
    // nBatchSize: 批量大小
    // bShuffle: 是否随机打乱
    DataLoader(Dataset* pDataset, int nBatchSize, bool bShuffle = true)
        : m_pDataset(pDataset), m_nBatchSize(nBatchSize), m_bShuffle(bShuffle)
    {
        m_nNumSamples = static_cast<int>(pDataset->size());
        m_nNumBatches = (m_nNumSamples + nBatchSize - 1) / nBatchSize;  // 20260320 ZJH 向上取整

        // 20260320 ZJH 初始化索引序列
        m_vecIndices.resize(static_cast<size_t>(m_nNumSamples));
        for (int i = 0; i < m_nNumSamples; ++i)
            m_vecIndices[static_cast<size_t>(i)] = i;
    }

    // 20260320 ZJH reset — 重置到第一个 batch，可选 shuffle
    void reset() {
        m_nCurrentBatch = 0;
        if (m_bShuffle) {
            static thread_local std::mt19937 gen(std::random_device{}());
            std::shuffle(m_vecIndices.begin(), m_vecIndices.end(), gen);
        }
    }

    // 20260320 ZJH hasNext — 是否还有剩余 batch
    bool hasNext() const { return m_nCurrentBatch < m_nNumBatches; }

    // 20260320 ZJH next — 获取下一个 batch
    // 返回: (batchInput [N, ...], batchTarget [N, ...])
    std::pair<Tensor, Tensor> next() {
        int nStart = m_nCurrentBatch * m_nBatchSize;
        int nEnd = std::min(nStart + m_nBatchSize, m_nNumSamples);
        int nActualBatch = nEnd - nStart;

        // 20260320 ZJH 获取第一个样本确定形状
        auto [firstInput, firstTarget] = m_pDataset->getItem(
            static_cast<size_t>(m_vecIndices[static_cast<size_t>(nStart)]));

        auto vecInputShape = firstInput.shapeVec();
        auto vecTargetShape = firstTarget.shapeVec();

        // 20260320 ZJH 构建 batch 形状
        std::vector<int> vecBatchInputShape = {nActualBatch};
        vecBatchInputShape.insert(vecBatchInputShape.end(), vecInputShape.begin(), vecInputShape.end());
        std::vector<int> vecBatchTargetShape = {nActualBatch};
        vecBatchTargetShape.insert(vecBatchTargetShape.end(), vecTargetShape.begin(), vecTargetShape.end());

        auto batchInput = Tensor::zeros(vecBatchInputShape);
        auto batchTarget = Tensor::zeros(vecBatchTargetShape);

        int nInputSize = firstInput.numel();   // 20260320 ZJH 每个输入的元素数
        int nTargetSize = firstTarget.numel(); // 20260320 ZJH 每个目标的元素数

        // 20260320 ZJH 填入第一个样本
        {
            float* pInput = batchInput.mutableFloatDataPtr();
            float* pTarget = batchTarget.mutableFloatDataPtr();
            const float* pFI = firstInput.floatDataPtr();
            const float* pFT = firstTarget.floatDataPtr();
            for (int j = 0; j < nInputSize; ++j) pInput[j] = pFI[j];
            for (int j = 0; j < nTargetSize; ++j) pTarget[j] = pFT[j];
        }

        // 20260320 ZJH 填入后续样本
        for (int i = 1; i < nActualBatch; ++i) {
            int nIdx = m_vecIndices[static_cast<size_t>(nStart + i)];
            auto [inp, tgt] = m_pDataset->getItem(static_cast<size_t>(nIdx));
            auto ci = inp.contiguous();
            auto ct = tgt.contiguous();
            float* pInput = batchInput.mutableFloatDataPtr() + i * nInputSize;
            float* pTarget = batchTarget.mutableFloatDataPtr() + i * nTargetSize;
            const float* pI = ci.floatDataPtr();
            const float* pT = ct.floatDataPtr();
            for (int j = 0; j < nInputSize; ++j) pInput[j] = pI[j];
            for (int j = 0; j < nTargetSize; ++j) pTarget[j] = pT[j];
        }

        m_nCurrentBatch++;
        return {batchInput, batchTarget};
    }

    // 20260320 ZJH 获取 batch 总数
    int numBatches() const { return m_nNumBatches; }

    // 20260320 ZJH 获取数据集大小
    int numSamples() const { return m_nNumSamples; }

private:
    Dataset* m_pDataset;               // 20260320 ZJH 数据集指针
    int m_nBatchSize;                  // 20260320 ZJH 批量大小
    bool m_bShuffle;                   // 20260320 ZJH 是否随机打乱
    int m_nNumSamples;                 // 20260320 ZJH 样本总数
    int m_nNumBatches;                 // 20260320 ZJH batch 总数
    int m_nCurrentBatch = 0;           // 20260320 ZJH 当前 batch 索引
    std::vector<int> m_vecIndices;     // 20260320 ZJH 打乱后的索引序列
};

// 20260320 ZJH splitDataset — 按比例划分训练/验证/测试索引
// 返回: {trainIndices, valIndices, testIndices}
struct DatasetSplit {
    std::vector<size_t> vecTrainIndices;
    std::vector<size_t> vecValIndices;
    std::vector<size_t> vecTestIndices;
};

DatasetSplit splitDataset(size_t nTotalSize, float fTrainRatio = 0.8f,
                           float fValRatio = 0.1f, unsigned int nSeed = 42) {
    DatasetSplit split;
    std::vector<size_t> vecIndices(nTotalSize);
    for (size_t i = 0; i < nTotalSize; ++i) vecIndices[i] = i;

    // 20260320 ZJH 使用固定种子打乱（可复现）
    std::mt19937 gen(nSeed);
    std::shuffle(vecIndices.begin(), vecIndices.end(), gen);

    size_t nTrain = static_cast<size_t>(nTotalSize * fTrainRatio);
    size_t nVal = static_cast<size_t>(nTotalSize * fValRatio);

    split.vecTrainIndices.assign(vecIndices.begin(), vecIndices.begin() + nTrain);
    split.vecValIndices.assign(vecIndices.begin() + nTrain, vecIndices.begin() + nTrain + nVal);
    split.vecTestIndices.assign(vecIndices.begin() + nTrain + nVal, vecIndices.end());

    return split;
}

}  // namespace om
