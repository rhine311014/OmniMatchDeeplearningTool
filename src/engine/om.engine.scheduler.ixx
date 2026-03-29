// 20260320 ZJH 学习率调度器模块 — Phase 5
// 实现余弦退火、带预热的余弦退火、阶梯衰减等策略
module;

#include <cmath>
#include <algorithm>

export module om.engine.scheduler;

export namespace om {

// 20260320 ZJH LRScheduler — 学习率调度器基类
class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    // 20260320 ZJH step — 更新学习率（每个 epoch 调用一次）
    // nEpoch: 当前 epoch 编号（从 0 开始）
    // 返回: 当前应使用的学习率
    virtual float step(int nEpoch) = 0;

    // 20260320 ZJH 获取基础学习率
    float baseLR() const { return m_fBaseLR; }

protected:
    float m_fBaseLR = 0.01f;  // 20260320 ZJH 基础学习率
};

// 20260320 ZJH CosineAnnealingLR — 余弦退火学习率调度
// lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
class CosineAnnealingLR : public LRScheduler {
public:
    // 20260320 ZJH 构造函数
    // fBaseLR: 基础学习率
    // nTMax: 半周期（通常等于总 epoch 数）
    // fEtaMin: 最小学习率，默认 0
    CosineAnnealingLR(float fBaseLR, int nTMax, float fEtaMin = 0.0f)
        : m_nTMax(nTMax), m_fEtaMin(fEtaMin)
    {
        m_fBaseLR = fBaseLR;
    }

    float step(int nEpoch) override {
        if (m_nTMax <= 0) return m_fBaseLR;
        // 20260320 ZJH 余弦退火公式
        float fCos = std::cos(3.14159265f * static_cast<float>(nEpoch) / static_cast<float>(m_nTMax));
        return m_fEtaMin + 0.5f * (m_fBaseLR - m_fEtaMin) * (1.0f + fCos);
    }

private:
    int m_nTMax;       // 20260320 ZJH 半周期
    float m_fEtaMin;   // 20260320 ZJH 最小学习率
};

// 20260320 ZJH WarmupCosineAnnealingLR — 带线性预热的余弦退火
// 预热阶段：lr 从 0 线性增长到 base_lr
// 退火阶段：lr 按余弦从 base_lr 衰减到 eta_min
class WarmupCosineAnnealingLR : public LRScheduler {
public:
    // 20260320 ZJH 构造函数
    // fBaseLR: 基础学习率
    // nWarmupEpochs: 预热 epoch 数
    // nTotalEpochs: 总 epoch 数
    // fEtaMin: 最小学习率
    WarmupCosineAnnealingLR(float fBaseLR, int nWarmupEpochs, int nTotalEpochs,
                             float fEtaMin = 0.0f)
        : m_nWarmupEpochs(nWarmupEpochs), m_nTotalEpochs(nTotalEpochs), m_fEtaMin(fEtaMin)
    {
        m_fBaseLR = fBaseLR;
    }

    float step(int nEpoch) override {
        if (nEpoch < m_nWarmupEpochs) {
            // 20260320 ZJH 线性预热：lr = base_lr * epoch / warmup_epochs
            return m_fBaseLR * static_cast<float>(nEpoch + 1) / static_cast<float>(m_nWarmupEpochs);
        }
        // 20260320 ZJH 余弦退火
        int nCosineEpoch = nEpoch - m_nWarmupEpochs;
        int nCosineTotal = m_nTotalEpochs - m_nWarmupEpochs;
        if (nCosineTotal <= 0) return m_fBaseLR;
        float fCos = std::cos(3.14159265f * static_cast<float>(nCosineEpoch) / static_cast<float>(nCosineTotal));
        return m_fEtaMin + 0.5f * (m_fBaseLR - m_fEtaMin) * (1.0f + fCos);
    }

private:
    int m_nWarmupEpochs;   // 20260320 ZJH 预热 epoch 数
    int m_nTotalEpochs;    // 20260320 ZJH 总 epoch 数
    float m_fEtaMin;       // 20260320 ZJH 最小学习率
};

// 20260320 ZJH StepLR — 阶梯衰减学习率调度
// 每 step_size 个 epoch 乘以 gamma
class StepLR : public LRScheduler {
public:
    // 20260320 ZJH 构造函数
    // fBaseLR: 基础学习率
    // nStepSize: 衰减间隔（每多少个 epoch 衰减一次）
    // fGamma: 衰减因子，默认 0.1
    StepLR(float fBaseLR, int nStepSize, float fGamma = 0.1f)
        : m_nStepSize(nStepSize), m_fGamma(fGamma)
    {
        m_fBaseLR = fBaseLR;
    }

    float step(int nEpoch) override {
        int nDecays = nEpoch / m_nStepSize;  // 20260320 ZJH 已衰减次数
        float fLR = m_fBaseLR;
        for (int i = 0; i < nDecays; ++i) fLR *= m_fGamma;
        return fLR;
    }

private:
    int m_nStepSize;   // 20260320 ZJH 衰减间隔
    float m_fGamma;    // 20260320 ZJH 衰减因子
};

// 20260320 ZJH ExponentialLR — 指数衰减学习率调度
// lr = base_lr * gamma^epoch
class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(float fBaseLR, float fGamma = 0.95f)
        : m_fGamma(fGamma)
    {
        m_fBaseLR = fBaseLR;
    }

    float step(int nEpoch) override {
        return m_fBaseLR * std::pow(m_fGamma, static_cast<float>(nEpoch));
    }

private:
    float m_fGamma;  // 20260320 ZJH 衰减因子
};

}  // namespace om
