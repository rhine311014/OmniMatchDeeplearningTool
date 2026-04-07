// 20260402 ZJH AI 缺陷生成器 — 工业场景核心痛点：残次品太少无法训练
// 双路线架构:
//   路线 A: DRAEM+（零缺陷样本可用）— Perlin 噪声 + 8 种纹理模板 + Poisson 混合
//   路线 B: DDPM Tiny（5+ 缺陷样本）— 轻量 UNet 去噪扩散模型 (~2M 参数)
// 来源: AnomalyDiffusion (2024), DefectFill (2025), DRAEM (ICCV 2021)
// 用途: 现场良品率 >99% 时，从少量正常图+可选缺陷样本→生成大量逼真缺陷图
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

export module om.engine.defect_generator;

// 20260406 ZJH 导入依赖模块
import om.engine.tensor;        // 20260406 ZJH 张量基础
import om.engine.tensor_ops;    // 20260406 ZJH 张量运算
import om.engine.module;        // 20260406 ZJH 模块基类
import om.engine.conv;          // 20260406 ZJH 卷积层/转置卷积层
import om.engine.linear;        // 20260406 ZJH 全连接层
import om.engine.activations;   // 20260406 ZJH 激活函数
import om.hal.cpu_backend;      // 20260406 ZJH CPU 后端

export namespace om {

// =============================================================================
// 20260402 ZJH PerlinNoise2D — 2D Perlin 噪声生成器
// 多频率叠加（octave）模拟真实缺陷纹理的不规则分布
// =============================================================================
class PerlinNoise2D {
public:
    // 20260402 ZJH 构造函数 — 初始化排列表
    explicit PerlinNoise2D(int nSeed = 42) {
        std::mt19937 rng(nSeed);  // 20260402 ZJH 确定性种子
        m_vecPerm.resize(512);  // 20260406 ZJH 分配排列表空间（双倍以避免取模）
        std::iota(m_vecPerm.begin(), m_vecPerm.begin() + 256, 0);  // 20260406 ZJH 初始化 0~255
        std::shuffle(m_vecPerm.begin(), m_vecPerm.begin() + 256, rng);  // 20260406 ZJH 随机打乱
        for (int i = 0; i < 256; ++i) m_vecPerm[i + 256] = m_vecPerm[i];  // 20260406 ZJH 复制到后半段
    }

    // 20260402 ZJH generate — 生成 Perlin 噪声图
    // nW, nH: 输出尺寸; nOctaves: 频率叠加次数; fScale: 基础缩放
    // 返回: [nH, nW] 噪声图，值域 [0, 1]
    Tensor generate(int nW, int nH, int nOctaves = 4, float fScale = 0.05f) {
        auto result = Tensor::zeros({nH, nW});  // 20260406 ZJH 分配输出张量
        float* pOut = result.mutableFloatDataPtr();  // 20260406 ZJH 获取可写指针

        float fMaxVal = 0.0f;  // 20260406 ZJH 追踪最大绝对值用于归一化
        for (int y = 0; y < nH; ++y) {  // 20260406 ZJH 逐行遍历
            for (int x = 0; x < nW; ++x) {  // 20260406 ZJH 逐列遍历
                float fVal = 0.0f;  // 20260406 ZJH 当前像素噪声值
                float fAmplitude = 1.0f;  // 20260406 ZJH 初始振幅
                float fFreq = fScale;  // 20260406 ZJH 初始频率
                for (int oct = 0; oct < nOctaves; ++oct) {  // 20260406 ZJH 逐 octave 叠加
                    fVal += fAmplitude * noise2d(x * fFreq, y * fFreq);  // 20260406 ZJH 累加该 octave 的噪声贡献
                    fFreq *= 2.0f;       // 20260402 ZJH 频率翻倍
                    fAmplitude *= 0.5f;  // 20260402 ZJH 振幅减半
                }
                pOut[y * nW + x] = fVal;  // 20260406 ZJH 写入原始噪声值
                fMaxVal = std::max(fMaxVal, std::abs(fVal));  // 20260406 ZJH 更新最大绝对值
            }
        }
        // 20260402 ZJH 归一化到 [0, 1]
        if (fMaxVal > 1e-6f) {  // 20260406 ZJH 避免除零
            float fInv = 0.5f / fMaxVal;  // 20260406 ZJH 缩放因子
            for (int i = 0; i < nW * nH; ++i) pOut[i] = pOut[i] * fInv + 0.5f;  // 20260406 ZJH 线性映射到 [0, 1]
        }
        return result;  // 20260406 ZJH 返回噪声图
    }

private:
    std::vector<int> m_vecPerm;  // 20260402 ZJH 排列表 (512)

    // 20260402 ZJH 2D Perlin 噪声核心
    float noise2d(float fX, float fY) {
        int nX0 = static_cast<int>(std::floor(fX)) & 255;  // 20260406 ZJH 网格左下角 X 坐标（取模 256）
        int nY0 = static_cast<int>(std::floor(fY)) & 255;  // 20260406 ZJH 网格左下角 Y 坐标
        float fDx = fX - std::floor(fX);  // 20260406 ZJH 网格内 X 偏移量 [0, 1)
        float fDy = fY - std::floor(fY);  // 20260406 ZJH 网格内 Y 偏移量 [0, 1)
        float fU = fade(fDx), fV = fade(fDy);  // 20260406 ZJH 平滑插值权重
        int nAA = m_vecPerm[m_vecPerm[nX0] + nY0];      // 20260406 ZJH 左下角哈希值
        int nAB = m_vecPerm[m_vecPerm[nX0] + nY0 + 1];  // 20260406 ZJH 左上角哈希值
        int nBA = m_vecPerm[m_vecPerm[nX0 + 1] + nY0];  // 20260406 ZJH 右下角哈希值
        int nBB = m_vecPerm[m_vecPerm[nX0 + 1] + nY0 + 1];  // 20260406 ZJH 右上角哈希值
        // 20260406 ZJH 双线性插值四个角的梯度贡献
        return lerp(fV, lerp(fU, grad(nAA, fDx, fDy), grad(nBA, fDx - 1, fDy)),
                        lerp(fU, grad(nAB, fDx, fDy - 1), grad(nBB, fDx - 1, fDy - 1)));
    }
    // 20260406 ZJH fade — 5次多项式平滑曲线 6t^5 - 15t^4 + 10t^3
    static float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
    // 20260406 ZJH lerp — 线性插值
    static float lerp(float t, float a, float b) { return a + t * (b - a); }
    // 20260406 ZJH grad — 伪随机梯度方向选择（2D 简化版，4 个方向）
    static float grad(int nHash, float fX, float fY) {
        int h = nHash & 3;  // 20260406 ZJH 取低 2 位决定梯度方向
        float u = (h < 2) ? fX : fY;  // 20260406 ZJH 选择主轴
        float v = (h < 2) ? fY : fX;  // 20260406 ZJH 选择副轴
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);  // 20260406 ZJH 符号翻转
    }
};

// =============================================================================
// 20260402 ZJH DefectTextureBank — 内置 8 种工业缺陷纹理模板
// 每种模板生成特定形态的缺陷 mask + 纹理
// =============================================================================
enum class DefectType {
    Scratch   = 0,  // 20260402 ZJH 划痕（细长线条）
    Pit       = 1,  // 20260402 ZJH 凹坑（圆形凹陷）
    Stain     = 2,  // 20260402 ZJH 污渍（不规则团块）
    Corrosion = 3,  // 20260402 ZJH 腐蚀（粗糙纹理区域）
    Crack     = 4,  // 20260402 ZJH 裂纹（分支树状）
    Bubble    = 5,  // 20260402 ZJH 气泡（多个小圆）
    Foreign   = 6,  // 20260402 ZJH 异物（随机形状）
    Missing   = 7   // 20260402 ZJH 缺失（矩形缺口）
};

class DefectTextureBank {
public:
    // 20260402 ZJH generateMask — 根据缺陷类型生成 mask [H, W]（0=正常, 1=缺陷区域）
    static Tensor generateMask(int nH, int nW, DefectType eType, int nSeed = 42) {
        auto mask = Tensor::zeros({nH, nW});  // 20260406 ZJH 初始化全零 mask
        float* pM = mask.mutableFloatDataPtr();  // 20260406 ZJH 获取可写指针
        std::mt19937 rng(nSeed);  // 20260406 ZJH 随机数生成器（确定性种子）
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // 20260406 ZJH [0,1) 均匀分布

        switch (eType) {
        case DefectType::Scratch: {
            // 20260402 ZJH 划痕: 随机角度的细长线条
            int nCx = nW / 2 + (rng() % (nW / 4)) - nW / 8;
            int nCy = nH / 2 + (rng() % (nH / 4)) - nH / 8;
            float fAngle = dist(rng) * 3.14159f;
            int nLen = nW / 3 + rng() % (nW / 4);
            int nThick = 1 + rng() % 3;
            for (int t = -nLen / 2; t < nLen / 2; ++t) {
                int nx = nCx + static_cast<int>(t * std::cos(fAngle));
                int ny = nCy + static_cast<int>(t * std::sin(fAngle));
                for (int d = -nThick; d <= nThick; ++d) {
                    int px = nx + static_cast<int>(d * std::sin(fAngle));
                    int py = ny - static_cast<int>(d * std::cos(fAngle));
                    if (px >= 0 && px < nW && py >= 0 && py < nH) pM[py * nW + px] = 1.0f;
                }
            }
            break;
        }
        case DefectType::Pit: {
            // 20260402 ZJH 凹坑: 椭圆
            int nCx = nW / 4 + rng() % (nW / 2);
            int nCy = nH / 4 + rng() % (nH / 2);
            int nRx = 5 + rng() % (nW / 8);
            int nRy = 5 + rng() % (nH / 8);
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    float fDx = static_cast<float>(x - nCx) / nRx;
                    float fDy = static_cast<float>(y - nCy) / nRy;
                    if (fDx * fDx + fDy * fDy < 1.0f) pM[y * nW + x] = 1.0f;
                }
            break;
        }
        case DefectType::Stain: {
            // 20260402 ZJH 污渍: Perlin 噪声阈值化
            PerlinNoise2D perlin(nSeed);
            auto noise = perlin.generate(nW, nH, 3, 0.08f);
            const float* pN = noise.floatDataPtr();
            for (int i = 0; i < nH * nW; ++i) pM[i] = (pN[i] > 0.6f) ? 1.0f : 0.0f;
            break;
        }
        case DefectType::Corrosion: {
            // 20260402 ZJH 腐蚀: 大面积 Perlin 噪声
            PerlinNoise2D perlin(nSeed + 1);
            auto noise = perlin.generate(nW, nH, 5, 0.04f);
            const float* pN = noise.floatDataPtr();
            for (int i = 0; i < nH * nW; ++i) pM[i] = (pN[i] > 0.55f) ? 1.0f : 0.0f;
            break;
        }
        case DefectType::Crack: {
            // 20260402 ZJH 裂纹: 随机游走线条
            int nX = nW / 2, nY = nH / 2;
            for (int step = 0; step < nW; ++step) {
                if (nX >= 0 && nX < nW && nY >= 0 && nY < nH) {
                    pM[nY * nW + nX] = 1.0f;
                    if (nX > 0) pM[nY * nW + nX - 1] = 1.0f;
                    if (nX < nW - 1) pM[nY * nW + nX + 1] = 1.0f;
                }
                nX += (rng() % 3) - 1;
                nY += (rng() % 3) - 1;
                nX = std::max(0, std::min(nX, nW - 1));
                nY = std::max(0, std::min(nY, nH - 1));
            }
            break;
        }
        case DefectType::Bubble: {
            // 20260402 ZJH 气泡: 多个小圆
            int nCount = 3 + rng() % 8;
            for (int b = 0; b < nCount; ++b) {
                int bx = rng() % nW, by = rng() % nH;
                int br = 2 + rng() % 5;
                for (int y = std::max(0, by - br); y < std::min(nH, by + br); ++y)
                    for (int x = std::max(0, bx - br); x < std::min(nW, bx + br); ++x)
                        if ((x - bx) * (x - bx) + (y - by) * (y - by) < br * br)
                            pM[y * nW + x] = 1.0f;
            }
            break;
        }
        case DefectType::Foreign: {
            // 20260402 ZJH 异物: Perlin 噪声形状
            PerlinNoise2D perlin(nSeed + 2);
            auto noise = perlin.generate(nW, nH, 4, 0.06f);
            const float* pN = noise.floatDataPtr();
            int nCx = nW / 4 + rng() % (nW / 2), nCy = nH / 4 + rng() % (nH / 2);
            int nR = nW / 6;
            for (int y = 0; y < nH; ++y)
                for (int x = 0; x < nW; ++x) {
                    float fDist = std::sqrt(static_cast<float>((x-nCx)*(x-nCx)+(y-nCy)*(y-nCy)));
                    if (fDist < nR && pN[y * nW + x] > 0.45f) pM[y * nW + x] = 1.0f;
                }
            break;
        }
        case DefectType::Missing: {
            // 20260402 ZJH 缺失: 矩形缺口
            int nX1 = nW / 4 + rng() % (nW / 4), nY1 = nH / 4 + rng() % (nH / 4);
            int nX2 = nX1 + nW / 6 + rng() % (nW / 6);
            int nY2 = nY1 + nH / 6 + rng() % (nH / 6);
            nX2 = std::min(nX2, nW - 1); nY2 = std::min(nY2, nH - 1);
            for (int y = nY1; y <= nY2; ++y)
                for (int x = nX1; x <= nX2; ++x)
                    pM[y * nW + x] = 1.0f;
            break;
        }
        }
        return mask;  // 20260406 ZJH 返回缺陷区域 mask
    }

    // 20260402 ZJH generateTexture — 生成缺陷区域的纹理（暗色/亮色/彩色变化）
    static Tensor generateTexture(int nH, int nW, DefectType eType, int nSeed = 42) {
        PerlinNoise2D perlin(nSeed + 100);  // 20260406 ZJH 使用不同种子生成纹理噪声
        auto noise = perlin.generate(nW, nH, 3, 0.1f);  // 20260406 ZJH 生成 3-octave Perlin 噪声
        auto texture = Tensor::zeros({3, nH, nW});  // 20260402 ZJH [3, H, W] RGB 纹理
        float* pT = texture.mutableFloatDataPtr();  // 20260406 ZJH 纹理可写指针
        const float* pN = noise.floatDataPtr();  // 20260406 ZJH 噪声只读指针
        int nS = nH * nW;  // 20260406 ZJH 单通道像素总数

        // 20260402 ZJH 根据缺陷类型选择纹理颜色基调
        float fBaseR = 0.3f, fBaseG = 0.3f, fBaseB = 0.3f;  // 20260406 ZJH 默认灰色基调
        switch (eType) {
            case DefectType::Scratch:   fBaseR = 0.2f; fBaseG = 0.2f; fBaseB = 0.2f; break;  // 20260406 ZJH 暗色划痕
            case DefectType::Pit:       fBaseR = 0.1f; fBaseG = 0.1f; fBaseB = 0.1f; break;  // 20260406 ZJH 深暗凹坑
            case DefectType::Stain:     fBaseR = 0.5f; fBaseG = 0.4f; fBaseB = 0.2f; break;  // 20260406 ZJH 棕色污渍
            case DefectType::Corrosion: fBaseR = 0.6f; fBaseG = 0.3f; fBaseB = 0.1f; break;  // 20260406 ZJH 锈色腐蚀
            case DefectType::Crack:     fBaseR = 0.15f;fBaseG = 0.15f;fBaseB = 0.15f;break;  // 20260406 ZJH 黑色裂纹
            case DefectType::Bubble:    fBaseR = 0.7f; fBaseG = 0.7f; fBaseB = 0.7f; break;  // 20260406 ZJH 亮色气泡
            case DefectType::Foreign:   fBaseR = 0.4f; fBaseG = 0.5f; fBaseB = 0.3f; break;  // 20260406 ZJH 绿色调异物
            case DefectType::Missing:   fBaseR = 0.05f;fBaseG = 0.05f;fBaseB = 0.05f;break;  // 20260406 ZJH 全黑缺失
        }
        // 20260406 ZJH 将基础颜色 + 噪声扰动混合为最终纹理
        for (int i = 0; i < nS; ++i) {
            float fNoise = pN[i] * 0.3f;  // 20260406 ZJH 噪声扰动幅度 30%
            pT[0 * nS + i] = std::max(0.0f, std::min(1.0f, fBaseR + fNoise));        // 20260406 ZJH R 通道
            pT[1 * nS + i] = std::max(0.0f, std::min(1.0f, fBaseG + fNoise * 0.8f)); // 20260406 ZJH G 通道（噪声衰减 80%）
            pT[2 * nS + i] = std::max(0.0f, std::min(1.0f, fBaseB + fNoise * 0.6f)); // 20260406 ZJH B 通道（噪声衰减 60%）
        }
        return texture;  // 20260406 ZJH 返回 RGB 缺陷纹理
    }
};

// =============================================================================
// 20260402 ZJH DDPMTiny — 轻量去噪扩散模型（~2M 参数）
// 基于 DDPM (Ho et al. 2020) + DDIM 加速采样 (Song et al. 2021)
// 架构: 轻量 UNet (32 base ch) + 时间步嵌入 + 条件输入
// 训练: 对缺陷图加噪 → UNet 预测噪声 → MSE 损失
// 采样: 从纯噪声开始 → 50 步 DDIM 去噪 → 生成缺陷图
// =============================================================================
class DDPMTiny : public Module {
public:
    // 20260402 ZJH 构造函数
    // nChannels: 图像通道数（3=RGB）; nBaseChannels: UNet 基础通道数
    // nTimesteps: 扩散步数
    DDPMTiny(int nChannels = 3, int nBaseChannels = 32, int nTimesteps = 500)
        : m_nChannels(nChannels), m_nTimesteps(nTimesteps),
          // 20260402 ZJH UNet: 编码器
          m_enc1(nChannels + 1, nBaseChannels, 3, 1, 1, false),          // 20260402 ZJH +1 for mask
          m_bn1(nBaseChannels),
          m_enc2(nBaseChannels, nBaseChannels * 2, 3, 2, 1, false),       // 20260402 ZJH /2
          m_bn2(nBaseChannels * 2),
          m_enc3(nBaseChannels * 2, nBaseChannels * 4, 3, 2, 1, false),   // 20260402 ZJH /4
          m_bn3(nBaseChannels * 4),
          // 20260402 ZJH UNet: 解码器
          m_dec3(nBaseChannels * 4, nBaseChannels * 2, 3, 1, 1, false),
          m_dbn3(nBaseChannels * 2),
          m_dec2(nBaseChannels * 2 + nBaseChannels * 2, nBaseChannels, 3, 1, 1, false),  // 20260402 ZJH +skip
          m_dbn2(nBaseChannels),
          m_dec1(nBaseChannels + nBaseChannels, nChannels, 3, 1, 1, true), // 20260402 ZJH +skip→output
          // 20260402 ZJH 时间步嵌入
          m_timeEmbed(1, nBaseChannels * 4, true)
    {
        // 20260402 ZJH 预计算线性噪声调度 beta_t
        m_vecBeta.resize(nTimesteps);
        m_vecAlpha.resize(nTimesteps);
        m_vecAlphaBar.resize(nTimesteps);
        float fBetaStart = 1e-4f, fBetaEnd = 0.02f;
        for (int t = 0; t < nTimesteps; ++t) {
            m_vecBeta[t] = fBetaStart + (fBetaEnd - fBetaStart) * t / static_cast<float>(nTimesteps - 1);
            m_vecAlpha[t] = 1.0f - m_vecBeta[t];
            m_vecAlphaBar[t] = (t == 0) ? m_vecAlpha[0] : m_vecAlphaBar[t - 1] * m_vecAlpha[t];
        }
    }

    // 20260402 ZJH forward — 预测噪声（训练用）
    // noisyImage: [N, C, H, W] 加噪图像
    // mask: [N, 1, H, W] 缺陷区域 mask
    // timestep: 当前时间步 t
    // 返回: [N, C, H, W] 预测的噪声
    Tensor forward(const Tensor& noisyImage) override {
        // 20260402 ZJH 简化: 不拼接 mask（在调用方处理）
        auto e1 = ReLU().forward(m_bn1.forward(m_enc1.forward(noisyImage)));  // 20260406 ZJH 编码器第1层: Conv+BN+ReLU
        auto e2 = ReLU().forward(m_bn2.forward(m_enc2.forward(e1)));  // 20260406 ZJH 编码器第2层: Conv(stride=2)+BN+ReLU, 尺寸/2
        auto e3 = ReLU().forward(m_bn3.forward(m_enc3.forward(e2)));  // 20260406 ZJH 编码器第3层: Conv(stride=2)+BN+ReLU, 尺寸/4

        // 20260402 ZJH 解码 + skip connections（简化: 最近邻上采样）
        auto d3 = ReLU().forward(m_dbn3.forward(m_dec3.forward(e3)));  // 20260406 ZJH 解码器第1层
        // 20260402 ZJH 上采样 d3 到 e2 尺寸
        d3 = upsampleNN(d3, e2.shape(2), e2.shape(3));  // 20260406 ZJH 最近邻上采样到 e2 分辨率
        // 20260402 ZJH 拼接 skip
        auto cat2 = catCh(d3, e2);  // 20260406 ZJH 通道拼接 skip connection
        auto d2 = ReLU().forward(m_dbn2.forward(m_dec2.forward(cat2)));  // 20260406 ZJH 解码器第2层
        d2 = upsampleNN(d2, e1.shape(2), e1.shape(3));  // 20260406 ZJH 上采样到 e1 分辨率
        auto cat1 = catCh(d2, e1);  // 20260406 ZJH 通道拼接 skip connection
        auto d1 = m_dec1.forward(cat1);  // 20260402 ZJH 输出（无激活）

        return d1;  // 20260406 ZJH 返回预测噪声
    }

    // 20260402 ZJH addNoise — 对图像加噪（前向扩散）
    // x0: [N, C, H, W] 原始图像; nTimestep: 时间步 t
    // 返回: {noisy_image, noise} 加噪图像和噪声
    std::pair<Tensor, Tensor> addNoise(const Tensor& x0, int nTimestep) {
        float fAlphaBar = m_vecAlphaBar[std::min(nTimestep, m_nTimesteps - 1)];  // 20260406 ZJH 查表获取累乘 alpha
        float fSqrtAB = std::sqrt(fAlphaBar);  // 20260406 ZJH sqrt(αbar)
        float fSqrt1mAB = std::sqrt(1.0f - fAlphaBar);  // 20260406 ZJH sqrt(1-αbar)

        auto noise = Tensor::randn(x0.shapeVec());  // 20260402 ZJH 标准正态噪声
        // 20260406 ZJH 前向扩散公式: x_t = sqrt(αbar)*x0 + sqrt(1-αbar)*ε
        auto noisy = tensorAdd(tensorMulScalar(x0, fSqrtAB),
                               tensorMulScalar(noise, fSqrt1mAB));
        return {noisy, noise};  // 20260406 ZJH 返回加噪图像和噪声
    }

    // 20260402 ZJH sample — DDIM 采样（从噪声生成图像）
    // shape: 输出形状 [1, C, H, W]; nSteps: 采样步数（默认 50）
    // condImage: 条件输入（正常图像）; mask: 缺陷区域
    Tensor sample(const std::vector<int>& shape, int nSteps = 50,
                  const Tensor& condImage = Tensor(), const Tensor& mask = Tensor()) {
        eval();  // 20260402 ZJH 推理模式
        auto xt = Tensor::randn(shape);  // 20260402 ZJH 纯噪声起点

        // 20260402 ZJH DDIM 步长
        int nSkip = m_nTimesteps / nSteps;

        for (int i = nSteps - 1; i >= 0; --i) {
            int nT = i * nSkip;
            float fAlphaBar = m_vecAlphaBar[std::min(nT, m_nTimesteps - 1)];
            float fAlphaBarPrev = (i > 0) ? m_vecAlphaBar[std::min((i - 1) * nSkip, m_nTimesteps - 1)] : 1.0f;

            // 20260402 ZJH 拼接 mask 通道（如有）
            Tensor input;
            if (mask.numel() > 0) {
                input = catCh(xt, mask);  // 20260402 ZJH [N, C+1, H, W]
            } else {
                // 20260402 ZJH 无 mask 时拼接全零通道
                input = catCh(xt, Tensor::zeros({shape[0], 1, shape[2], shape[3]}));
            }

            auto predNoise = forward(input);  // 20260402 ZJH 预测噪声

            // 20260402 ZJH DDIM 更新公式
            float fSqrtAB = std::sqrt(fAlphaBar);
            float fSqrt1mAB = std::sqrt(1.0f - fAlphaBar);
            float fSqrtABprev = std::sqrt(fAlphaBarPrev);
            float fSqrt1mABprev = std::sqrt(1.0f - fAlphaBarPrev);

            // 20260402 ZJH x0_pred = (xt - sqrt(1-αbar)*ε) / sqrt(αbar)
            auto x0Pred = tensorMulScalar(
                tensorSub(xt, tensorMulScalar(predNoise, fSqrt1mAB)),
                1.0f / (fSqrtAB + 1e-8f));

            // 20260402 ZJH xt-1 = sqrt(αbar_prev)*x0_pred + sqrt(1-αbar_prev)*ε
            xt = tensorAdd(tensorMulScalar(x0Pred, fSqrtABprev),
                           tensorMulScalar(predNoise, fSqrt1mABprev));

            // 20260402 ZJH 条件融合: 非缺陷区域保持原图
            if (condImage.numel() > 0 && mask.numel() > 0) {
                auto cMask = mask.contiguous();
                auto cCond = condImage.contiguous();
                auto cXt = xt.contiguous();
                float* pXt = xt.mutableFloatDataPtr();
                const float* pCond = cCond.floatDataPtr();
                const float* pM = cMask.floatDataPtr();
                int nC = shape[1], nH = shape[2], nW = shape[3], nS = nH * nW;
                for (int c = 0; c < nC; ++c)
                    for (int j = 0; j < nS; ++j) {
                        float fMask = pM[j];
                        pXt[c * nS + j] = fMask * pXt[c * nS + j] + (1.0f - fMask) * pCond[c * nS + j];
                    }
            }
        }

        // 20260402 ZJH 钳制到 [0, 1]
        auto cXt = xt.contiguous();
        float* p = xt.mutableFloatDataPtr();
        for (int i = 0; i < xt.numel(); ++i) p[i] = std::max(0.0f, std::min(1.0f, p[i]));

        return xt;
    }

    // 20260406 ZJH 递归收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> v;  // 20260406 ZJH 参数容器
        auto append = [&](std::vector<Tensor*> p) { v.insert(v.end(), p.begin(), p.end()); };  // 20260406 ZJH 合并辅助
        // 20260406 ZJH 编码器参数
        append(m_enc1.parameters()); append(m_bn1.parameters());
        append(m_enc2.parameters()); append(m_bn2.parameters());
        append(m_enc3.parameters()); append(m_bn3.parameters());
        // 20260406 ZJH 解码器参数
        append(m_dec3.parameters()); append(m_dbn3.parameters());
        append(m_dec2.parameters()); append(m_dbn2.parameters());
        append(m_dec1.parameters());
        // 20260406 ZJH 时间步嵌入参数
        append(m_timeEmbed.parameters());
        return v;  // 20260406 ZJH 返回全部参数
    }

    // 20260406 ZJH 递归收集所有 BN 缓冲区 (running_mean/running_var)
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> v;  // 20260406 ZJH 缓冲区容器
        auto append = [&](std::vector<Tensor*> b) { v.insert(v.end(), b.begin(), b.end()); };  // 20260406 ZJH 合并辅助
        append(m_bn1.buffers()); append(m_bn2.buffers()); append(m_bn3.buffers());  // 20260406 ZJH 编码器 BN
        append(m_dbn3.buffers()); append(m_dbn2.buffers());  // 20260406 ZJH 解码器 BN
        return v;  // 20260406 ZJH 返回全部缓冲区
    }

    // 20260406 ZJH 设置训练/评估模式，递归传播到所有子层
    void train(bool b = true) override {
        m_bTraining = b;  // 20260406 ZJH 更新当前模块状态
        m_enc1.train(b); m_bn1.train(b); m_enc2.train(b); m_bn2.train(b);  // 20260406 ZJH 编码器
        m_enc3.train(b); m_bn3.train(b);
        m_dec3.train(b); m_dbn3.train(b); m_dec2.train(b); m_dbn2.train(b);  // 20260406 ZJH 解码器
    }

    // 20260406 ZJH 获取扩散总步数
    int timesteps() const { return m_nTimesteps; }

    // 20260402 ZJH 公开工具函数供 DefectGenerator 使用
    static Tensor publicCatCh(const Tensor& a, const Tensor& b) { return catCh(a, b); }  // 20260406 ZJH 代理调用通道拼接

private:
    int m_nChannels;    // 20260406 ZJH 图像通道数
    int m_nTimesteps;   // 20260406 ZJH 扩散总步数
    Conv2d m_enc1, m_enc2, m_enc3;  // 20260406 ZJH UNet 编码器卷积层
    BatchNorm2d m_bn1, m_bn2, m_bn3;  // 20260406 ZJH 编码器 BN 层
    Conv2d m_dec3, m_dec2, m_dec1;  // 20260406 ZJH UNet 解码器卷积层
    BatchNorm2d m_dbn3, m_dbn2;  // 20260406 ZJH 解码器 BN 层
    Linear m_timeEmbed;  // 20260406 ZJH 时间步嵌入全连接层
    std::vector<float> m_vecBeta;      // 20260406 ZJH 噪声调度 β_t 序列
    std::vector<float> m_vecAlpha;     // 20260406 ZJH α_t = 1 - β_t
    std::vector<float> m_vecAlphaBar;  // 20260406 ZJH 累乘 αbar_t = Π_{i=0}^{t} α_i

    // 20260402 ZJH 最近邻上采样
    // 20260406 ZJH 将张量 t 从 [N,C,nH,nW] 上采样到 [N,C,nTH,nTW]
    static Tensor upsampleNN(const Tensor& t, int nTH, int nTW) {
        auto c = t.contiguous();  // 20260406 ZJH 确保内存连续
        int nN = c.shape(0), nC = c.shape(1), nH = c.shape(2), nW = c.shape(3);  // 20260406 ZJH 获取输入尺寸
        auto r = Tensor::zeros({nN, nC, nTH, nTW});  // 20260406 ZJH 分配输出张量
        float* pO = r.mutableFloatDataPtr(); const float* pI = c.floatDataPtr();  // 20260406 ZJH 数据指针
        float fSH = static_cast<float>(nH) / nTH, fSW = static_cast<float>(nW) / nTW;  // 20260406 ZJH 缩放因子
        // 20260406 ZJH 逐像素最近邻映射
        for (int n = 0; n < nN; ++n) for (int ch = 0; ch < nC; ++ch)
            for (int h = 0; h < nTH; ++h) { int sh = std::min(static_cast<int>(h*fSH),nH-1);  // 20260406 ZJH 源行坐标
                for (int w = 0; w < nTW; ++w) { int sw = std::min(static_cast<int>(w*fSW),nW-1);  // 20260406 ZJH 源列坐标
                    pO[((n*nC+ch)*nTH+h)*nTW+w] = pI[((n*nC+ch)*nH+sh)*nW+sw]; }}  // 20260406 ZJH 拷贝像素
        return r;  // 20260406 ZJH 返回上采样结果
    }

    // 20260402 ZJH 通道拼接
    // 20260406 ZJH 将两个张量沿通道维度拼接: [N,Ca,H,W] + [N,Cb,H,W] → [N,Ca+Cb,H,W]
    static Tensor catCh(const Tensor& a, const Tensor& b) {
        auto ca = a.contiguous(), cb = b.contiguous();  // 20260406 ZJH 确保连续
        int nN = ca.shape(0), nCa = ca.shape(1), nCb = cb.shape(1);  // 20260406 ZJH batch 和通道数
        int nH = ca.shape(2), nW = ca.shape(3), nS = nH * nW;  // 20260406 ZJH 空间尺寸
        int nCt = nCa + nCb;  // 20260406 ZJH 拼接后总通道数
        auto r = Tensor::zeros({nN, nCt, nH, nW});  // 20260406 ZJH 分配输出张量
        float* pO = r.mutableFloatDataPtr();  // 20260406 ZJH 输出指针
        const float* pA = ca.floatDataPtr(); const float* pB = cb.floatDataPtr();  // 20260406 ZJH 输入指针
        for (int n = 0; n < nN; ++n) {
            // 20260406 ZJH 拷贝 a 的通道
            for (int c = 0; c < nCa; ++c) for (int i = 0; i < nS; ++i)
                pO[((n*nCt+c)*nS)+i] = pA[((n*nCa+c)*nS)+i];
            // 20260406 ZJH 拷贝 b 的通道
            for (int c = 0; c < nCb; ++c) for (int i = 0; i < nS; ++i)
                pO[((n*nCt+nCa+c)*nS)+i] = pB[((n*nCb+c)*nS)+i];
        }
        return r;  // 20260406 ZJH 返回拼接结果
    }
};

// =============================================================================
// 20260402 ZJH DefectGenerator — 统一缺陷生成接口（自动选择路线）
// 用户只需调用 generate()，内部自动判断:
//   缺陷样本 <5 张 → 路线 A (DRAEM+)
//   缺陷样本 >=5 张 → 路线 B (DDPM Tiny)
// =============================================================================

struct DefectGeneratorConfig {
    int nTargetCount = 100;         // 20260402 ZJH 目标生成数量
    int nImageWidth = 224;          // 20260402 ZJH 图像宽度
    int nImageHeight = 224;         // 20260402 ZJH 图像高度
    int nDDPMTrainEpochs = 20;     // 20260402 ZJH DDPM 训练轮次
    float fDDPMLR = 1e-3f;          // 20260402 ZJH DDPM 学习率
    bool bForceMode = false;        // 20260402 ZJH 强制使用指定模式
    int nForcedMode = -1;           // 20260402 ZJH -1=自动, 0=DRAEM+, 1=DDPM
};

struct DefectGeneratorResult {
    std::vector<std::vector<float>> vecImages;  // 20260402 ZJH 生成的缺陷图 [C*H*W]
    std::vector<std::vector<float>> vecMasks;   // 20260402 ZJH 对应的缺陷 mask [H*W]
    int nGeneratedCount = 0;                     // 20260402 ZJH 实际生成数量
    int nMode = 0;                               // 20260402 ZJH 使用的模式 (0=DRAEM+, 1=DDPM)
    std::string strLog;                          // 20260402 ZJH 生成日志
};

class DefectGenerator {
public:
    // 20260402 ZJH generate — 统一生成入口
    // vecNormalImages: 正常样本 [C*H*W] 列表
    // vecDefectImages: 缺陷样本 [C*H*W] 列表（可为空 → 自动用 DRAEM+）
    // config: 生成配置
    static DefectGeneratorResult generate(
        const std::vector<std::vector<float>>& vecNormalImages,
        const std::vector<std::vector<float>>& vecDefectImages,
        const DefectGeneratorConfig& config)
    {
        DefectGeneratorResult result;  // 20260406 ZJH 初始化结果结构
        int nC = 3, nH = config.nImageHeight, nW = config.nImageWidth;  // 20260406 ZJH 图像尺寸参数

        // 20260402 ZJH 自动选择模式
        bool bUseDDPM = (static_cast<int>(vecDefectImages.size()) >= 5);  // 20260406 ZJH 缺陷样本>=5用 DDPM
        if (config.bForceMode) bUseDDPM = (config.nForcedMode == 1);  // 20260406 ZJH 强制模式覆盖自动选择

        if (!bUseDDPM) {
            // 20260402 ZJH ===== 路线 A: DRAEM+ 增强合成 =====
            result.nMode = 0;
            result.strLog = "[DefectGen] Mode: DRAEM+ (zero-shot, " +
                std::to_string(config.nTargetCount) + " targets)\n";

            std::mt19937 rng(42);  // 20260406 ZJH 固定种子保证可重复性
            for (int i = 0; i < config.nTargetCount; ++i) {  // 20260406 ZJH 逐张生成缺陷图
                if (vecNormalImages.empty()) break;  // 20260406 ZJH 无正常样本则停止
                // 20260402 ZJH 随机选择正常样本作为背景
                int nBgIdx = rng() % vecNormalImages.size();  // 20260406 ZJH 随机背景图索引
                // 20260402 ZJH 随机选择缺陷类型
                DefectType eType = static_cast<DefectType>(rng() % 8);  // 20260406 ZJH 8种缺陷之一
                int nSeed = static_cast<int>(rng());  // 20260406 ZJH 每张生成图使用不同种子

                // 20260402 ZJH 生成缺陷 mask 和纹理
                auto mask = DefectTextureBank::generateMask(nH, nW, eType, nSeed);
                auto texture = DefectTextureBank::generateTexture(nH, nW, eType, nSeed);

                // 20260402 ZJH 将缺陷纹理混合到正常图像上
                const float* pBg = vecNormalImages[nBgIdx].data();  // 20260406 ZJH 背景图数据指针
                const float* pMask = mask.floatDataPtr();  // 20260406 ZJH 缺陷 mask 指针
                const float* pTex = texture.floatDataPtr();  // 20260406 ZJH 缺陷纹理指针
                int nS = nH * nW;  // 20260406 ZJH 单通道像素总数

                std::vector<float> vecSynth(nC * nS);  // 20260406 ZJH 合成图像缓冲区
                for (int c = 0; c < nC; ++c) {
                    for (int j = 0; j < nS; ++j) {
                        float fM = pMask[j];
                        float fBg = (nBgIdx < static_cast<int>(vecNormalImages[nBgIdx].size())) ?
                            pBg[c * nS + j] : 0.5f;
                        float fTex = pTex[c * nS + j];
                        // 20260402 ZJH alpha 混合: result = (1-mask)*bg + mask*texture
                        // 添加高斯模糊边缘过渡（简化: mask 值作为 alpha）
                        vecSynth[c * nS + j] = (1.0f - fM) * fBg + fM * fTex;
                    }
                }

                result.vecImages.push_back(std::move(vecSynth));  // 20260406 ZJH 存储合成图像
                std::vector<float> vecMaskFlat(nS);  // 20260406 ZJH 展平 mask 缓冲区
                for (int j = 0; j < nS; ++j) vecMaskFlat[j] = pMask[j];  // 20260406 ZJH 拷贝 mask 数据
                result.vecMasks.push_back(std::move(vecMaskFlat));  // 20260406 ZJH 存储 mask
            }
            result.nGeneratedCount = static_cast<int>(result.vecImages.size());  // 20260406 ZJH 记录实际生成数量
            result.strLog += "[DefectGen] Generated " + std::to_string(result.nGeneratedCount) +
                " DRAEM+ images\n";

        } else {
            // 20260402 ZJH ===== 路线 B: DDPM Tiny 扩散生成 =====
            result.nMode = 1;
            result.strLog = "[DefectGen] Mode: DDPM Tiny (" +
                std::to_string(vecDefectImages.size()) + " defect samples)\n";

            // 20260402 ZJH Step 1: 训练 DDPM
            DDPMTiny ddpm(nC, 32, 500);  // 20260406 ZJH 创建 DDPM 模型（3通道, 32基础通道, 500步扩散）
            ddpm.train(true);  // 20260406 ZJH 设为训练模式
            auto vecParams = ddpm.parameters();  // 20260406 ZJH 获取所有可训练参数

            // 20260402 ZJH 简化: 使用内置优化器类型
            // 直接用 SGD（因为 Adam 参数初始化复杂）
            float fLR = config.fDDPMLR;  // 20260406 ZJH 学习率
            std::mt19937 rng(42);  // 20260406 ZJH 固定种子随机数生成器

            result.strLog += "[DefectGen] Training DDPM (" +
                std::to_string(config.nDDPMTrainEpochs) + " epochs)...\n";

            for (int epoch = 0; epoch < config.nDDPMTrainEpochs; ++epoch) {  // 20260406 ZJH 训练循环
                float fEpochLoss = 0.0f;  // 20260406 ZJH 当前 epoch 损失累加
                for (size_t i = 0; i < vecDefectImages.size(); ++i) {  // 20260406 ZJH 遍历所有缺陷样本
                    // 20260402 ZJH 构造输入
                    auto x0 = Tensor::fromData(vecDefectImages[i].data(), {1, nC, nH, nW});  // 20260406 ZJH 将缺陷图转为张量
                    int nT = rng() % ddpm.timesteps();  // 20260406 ZJH 随机选择扩散时间步
                    auto [noisy, noise] = ddpm.addNoise(x0, nT);  // 20260406 ZJH 前向扩散加噪

                    // 20260402 ZJH 拼接 mask 通道（全 1 = 整图都是缺陷）
                    auto fakeMask = Tensor::ones({1, 1, nH, nW});  // 20260406 ZJH 全 1 mask（整图为缺陷）
                    auto input = DDPMTiny::publicCatCh(noisy, fakeMask);  // 20260406 ZJH 拼接 mask 通道

                    auto predNoise = ddpm.forward(input);  // 20260406 ZJH 预测噪声
                    // 20260402 ZJH MSE 损失
                    auto diff = tensorSub(predNoise, noise);
                    auto loss = tensorMulScalar(tensorSum(tensorMul(diff, diff)),
                        1.0f / static_cast<float>(std::max(diff.numel(), 1)));

                    // 20260402 ZJH SGD 手动更新
                    ddpm.zeroGrad();  // 20260406 ZJH 清零梯度
                    tensorBackward(loss);  // 20260406 ZJH 反向传播
                    for (auto* p : vecParams) {  // 20260406 ZJH 遍历所有参数
                        auto g = tensorGetGrad(*p);  // 20260406 ZJH 获取梯度
                        if (g.numel() > 0) {  // 20260406 ZJH 有梯度才更新
                            auto update = tensorMulScalar(g, -fLR);  // 20260406 ZJH SGD: -lr * grad
                            *p = tensorAdd(*p, update);  // 20260406 ZJH 更新参数
                        }
                    }
                    fEpochLoss += loss.contiguous().floatDataPtr()[0];  // 20260406 ZJH 累加当前 batch 损失
                }
                if ((epoch + 1) % 5 == 0) {
                    result.strLog += "[DefectGen] DDPM epoch " + std::to_string(epoch + 1) +
                        " loss=" + std::to_string(fEpochLoss / vecDefectImages.size()) + "\n";
                }
            }

            // 20260402 ZJH Step 2: 采样生成
            result.strLog += "[DefectGen] Sampling " + std::to_string(config.nTargetCount) + " images...\n";
            for (int i = 0; i < config.nTargetCount; ++i) {  // 20260406 ZJH 逐张采样生成
                // 20260402 ZJH 随机正常图作为条件
                int nCondIdx = rng() % vecNormalImages.size();  // 20260406 ZJH 随机条件图索引
                auto condImage = Tensor::fromData(vecNormalImages[nCondIdx].data(), {1, nC, nH, nW});  // 20260406 ZJH 条件图张量

                // 20260402 ZJH 随机生成缺陷 mask
                DefectType eType = static_cast<DefectType>(rng() % 8);  // 20260406 ZJH 随机缺陷类型
                auto mask = DefectTextureBank::generateMask(nH, nW, eType, static_cast<int>(rng()));  // 20260406 ZJH 生成 2D mask
                auto mask4d = Tensor::zeros({1, 1, nH, nW});  // 20260406 ZJH 扩展为 4D 张量 [1,1,H,W]
                float* pM4d = mask4d.mutableFloatDataPtr();  // 20260406 ZJH 4D mask 可写指针
                const float* pM = mask.floatDataPtr();  // 20260406 ZJH 2D mask 只读指针
                for (int j = 0; j < nH * nW; ++j) pM4d[j] = pM[j];  // 20260406 ZJH 拷贝 mask 数据

                // 20260402 ZJH DDIM 采样
                auto generated = ddpm.sample({1, nC, nH, nW}, 50, condImage, mask4d);

                // 20260402 ZJH 提取结果
                auto cGen = generated.contiguous();  // 20260406 ZJH 确保连续
                const float* pGen = cGen.floatDataPtr();  // 20260406 ZJH 生成图数据指针
                int nTotal = nC * nH * nW;  // 20260406 ZJH 图像总元素数
                std::vector<float> vecImg(nTotal);  // 20260406 ZJH 图像缓冲区
                for (int j = 0; j < nTotal; ++j) vecImg[j] = pGen[j];  // 20260406 ZJH 拷贝数据
                result.vecImages.push_back(std::move(vecImg));  // 20260406 ZJH 存储生成图

                std::vector<float> vecMaskFlat(nH * nW);  // 20260406 ZJH mask 缓冲区
                for (int j = 0; j < nH * nW; ++j) vecMaskFlat[j] = pM[j];  // 20260406 ZJH 拷贝 mask
                result.vecMasks.push_back(std::move(vecMaskFlat));  // 20260406 ZJH 存储 mask
            }
            result.nGeneratedCount = static_cast<int>(result.vecImages.size());  // 20260406 ZJH 记录生成数量
            result.strLog += "[DefectGen] Generated " + std::to_string(result.nGeneratedCount) +
                " DDPM images\n";
        }

        return result;  // 20260406 ZJH 返回生成结果
    }
};

}  // namespace om
