// 20260319 ZJH 优化器模块 — Phase 2 Part 1
// 实现 SGD（支持动量）和 Adam 两种优化器
// 20260325 ZJH Phase 3: 添加 GPU 路径，CUDA 参数直接在 GPU 上更新，消除 GPU↔CPU 数据乒乓
module;

#include <vector>
#include <cmath>

#include "om_types.h"  // 20260325 ZJH DeviceType 定义

export module om.engine.optimizer;

// 20260319 ZJH 导入依赖模块：张量类、张量运算、CPU 后端（直接操作数据指针）
import om.engine.tensor;
import om.engine.tensor_ops;
import om.hal.cpu_backend;
// 20260325 ZJH Phase 3: 导入 CUDA 后端用于 GPU 上的优化器 step
#ifdef OM_HAS_CUDA
import om.hal.cuda_backend;
#endif

export namespace om {

// 20260319 ZJH SGD — 随机梯度下降优化器，支持可选动量
// 参数更新规则：
//   无动量: param -= lr * grad
//   有动量: velocity = momentum * velocity + grad; param -= lr * velocity
// 20260325 ZJH Phase 3: GPU 参数使用 CUDABackend::sgdStep 直接在 GPU 上更新
class SGD {
public:
    // 20260319 ZJH 构造函数
    // vecParams: 需要优化的参数指针向量（由 Module::parameters() 获取）
    // fLr: 学习率
    // fMomentum: 动量系数，0.0 表示不使用动量
    SGD(std::vector<Tensor*> vecParams, float fLr, float fMomentum = 0.0f, float fWeightDecay = 0.0f)
        : m_vecParams(vecParams), m_fLr(fLr), m_fMomentum(fMomentum), m_fWeightDecay(fWeightDecay)
    {
        // 20260325 ZJH Phase 3: 速度缓冲区在参数所在设备上创建
        if (fMomentum > 0.0f) {
            m_vecVelocities.resize(vecParams.size());  // 20260319 ZJH 分配速度向量
            for (size_t i = 0; i < vecParams.size(); ++i) {
                // 20260325 ZJH 在参数的设备上创建速度缓冲区（GPU 参数 → GPU 速度）
                m_vecVelocities[i] = Tensor::zeros(vecParams[i]->shapeVec(), vecParams[i]->device());
            }
        }
    }

    // 20260331 ZJH addParams — 动态添加新参数到优化器（用于骨干解冻后加入低 LR 参数组）
    // fLrMultiplier: 该组参数的学习率倍率（实际 LR = baseLR × multiplier）
    void addParams(std::vector<Tensor*> vecNewParams, float fLrMultiplier = 1.0f) {
        for (auto* pParam : vecNewParams) {
            m_vecParams.push_back(pParam);  // 20260331 ZJH 追加到参数列表
            m_vecLrMultipliers.push_back(fLrMultiplier);  // 20260331 ZJH 记录对应倍率
            if (m_fMomentum > 0.0f) {
                // 20260331 ZJH 为新参数分配速度缓冲区（在参数所在设备上）
                m_vecVelocities.push_back(Tensor::zeros(pParam->shapeVec(), pParam->device()));
            }
        }
    }

    // 20260331 ZJH setParamLrMultipliers — 设置每个参数的学习率倍率向量
    // 用于分层学习率：骨干参数倍率 0.1，head 参数倍率 1.0
    void setParamLrMultipliers(const std::vector<float>& vecMultipliers) {
        m_vecLrMultipliers = vecMultipliers;  // 20260331 ZJH 直接覆盖
    }

    // 20260319 ZJH step — 执行一步参数更新
    // 遍历所有参数，根据梯度和学习率更新参数值
    // 20260325 ZJH Phase 3: GPU 参数使用 CUDABackend::sgdStep
    // 20260331 ZJH 支持分层学习率: 每个参数实际 LR = m_fLr × m_vecLrMultipliers[i]
    void step() {
        for (size_t i = 0; i < m_vecParams.size(); ++i) {
            auto grad = tensorGetGrad(*m_vecParams[i]);  // 20260319 ZJH 获取当前参数的梯度
            if (grad.numel() == 0) continue;  // 20260319 ZJH 无梯度则跳过该参数
            auto cGrad = grad.contiguous();  // 20260319 ZJH 确保梯度连续

            // 20260331 ZJH 计算此参数的实际学习率（baseLR × 倍率）
            float fEffectiveLr = m_fLr * getLrMultiplier(i);

            // 20260325 ZJH Phase 3: 检查参数是否在 CUDA 上
            if (m_vecParams[i]->isCuda()) {
#ifdef OM_HAS_CUDA
                // 20260325 ZJH GPU 路径：确保速度缓冲区在 GPU 上
                if (m_fMomentum > 0.0f && m_vecVelocities[i].isCpu()) {
                    m_vecVelocities[i] = m_vecVelocities[i].cuda();  // 20260325 ZJH 迁移速度到 GPU
                }
                // 20260325 ZJH 所有数据（参数/梯度/速度）都在 GPU 上，一次 kernel 完成更新
                float* pVel = (m_fMomentum > 0.0f) ? m_vecVelocities[i].mutableFloatDataPtr() : nullptr;
                CUDABackend::sgdStep(
                    m_vecParams[i]->mutableFloatDataPtr(),  // 20260325 ZJH GPU 参数指针
                    cGrad.floatDataPtr(),                    // 20260325 ZJH GPU 梯度指针
                    pVel,                                     // 20260325 ZJH GPU 速度指针（无动量时为 nullptr）
                    m_vecParams[i]->numel(),                 // 20260325 ZJH 元素总数
                    fEffectiveLr,                             // 20260331 ZJH 分层学习率
                    m_fMomentum);                             // 20260325 ZJH 动量系数
                // 20260402 ZJH Weight Decay (L2 正则化): param *= (1 - lr * wd)
                // 解耦式权重衰减，与 AdamW 风格一致，SGD 更新后缩放参数
                if (m_fWeightDecay > 0.0f) {
                    float fDecayFactor = 1.0f - fEffectiveLr * m_fWeightDecay;  // 20260402 ZJH 衰减因子
                    CUDABackend::mulScalar(
                        m_vecParams[i]->floatDataPtr(), fDecayFactor,
                        m_vecParams[i]->mutableFloatDataPtr(),
                        static_cast<size_t>(m_vecParams[i]->numel()));
                }
#endif
            } else {
                // 20260319 ZJH CPU 路径：原有逻辑不变
                float* pParam = m_vecParams[i]->mutableFloatDataPtr();  // 20260319 ZJH 参数可写指针
                const float* pGrad = cGrad.floatDataPtr();  // 20260319 ZJH 梯度只读指针
                int n = m_vecParams[i]->numel();  // 20260319 ZJH 参数元素总数

                if (m_fMomentum > 0.0f) {
                    // 20260319 ZJH 带动量的 SGD 更新
                    float* pVel = m_vecVelocities[i].mutableFloatDataPtr();  // 20260319 ZJH 速度可写指针
                    for (int j = 0; j < n; ++j) {
                        pVel[j] = m_fMomentum * pVel[j] + pGrad[j];  // 20260319 ZJH 更新速度：v = mu*v + grad
                        pParam[j] -= fEffectiveLr * pVel[j];  // 20260331 ZJH 使用分层 LR
                    }
                } else {
                    // 20260319 ZJH 普通 SGD 更新（无动量）
                    for (int j = 0; j < n; ++j) {
                        pParam[j] -= fEffectiveLr * pGrad[j];  // 20260331 ZJH 使用分层 LR
                    }
                }

                // 20260402 ZJH Weight Decay (L2 正则化): param *= (1 - lr * wd)
                // 解耦式权重衰减，SGD 更新后对参数施加衰减，防止权重过大导致过拟合
                if (m_fWeightDecay > 0.0f) {
                    float fDecayFactor = 1.0f - fEffectiveLr * m_fWeightDecay;  // 20260402 ZJH 衰减因子
                    for (int j = 0; j < n; ++j) {
                        pParam[j] *= fDecayFactor;  // 20260402 ZJH 缩放参数实现 L2 正则化
                    }
                }
            }
        }
    }

    // 20260319 ZJH zeroGrad — 清零所有参数的梯度
    void zeroGrad() {
        for (auto* pParam : m_vecParams) {
            tensorZeroGrad(*pParam);  // 20260319 ZJH 逐个清零参数梯度
        }
    }

    // 20260320 ZJH setLR — 动态设置学习率（用于学习率调度器）
    void setLR(float fLr) { m_fLr = fLr; }

    // 20260320 ZJH getLR — 获取当前学习率
    float getLR() const { return m_fLr; }

    // 20260326 ZJH 动态设置学习率（供 Cosine Annealing 调度器调用）
    void setLearningRate(float fNewLr) { m_fLr = fNewLr; }

private:
    // 20260331 ZJH 获取第 i 个参数的学习率倍率（无倍率向量时默认 1.0）
    float getLrMultiplier(size_t i) const {
        return (i < m_vecLrMultipliers.size()) ? m_vecLrMultipliers[i] : 1.0f;
    }

    std::vector<Tensor*> m_vecParams;      // 20260319 ZJH 参数指针列表
    float m_fLr;                            // 20260319 ZJH 学习率
    float m_fMomentum;                      // 20260319 ZJH 动量系数
    float m_fWeightDecay;                   // 20260402 ZJH L2 正则化系数（darknet 默认 5e-4）
    std::vector<Tensor> m_vecVelocities;   // 20260319 ZJH 速度缓冲区（动量模式下使用）
    std::vector<float> m_vecLrMultipliers; // 20260331 ZJH 每参数学习率倍率（分层学习率）
};

// 20260319 ZJH Adam — 自适应矩估计优化器
// 参数更新规则：
//   m = beta1 * m + (1 - beta1) * grad          (一阶矩估计)
//   v = beta2 * v + (1 - beta2) * grad^2        (二阶矩估计)
//   m_hat = m / (1 - beta1^t)                   (偏差校正)
//   v_hat = v / (1 - beta2^t)                   (偏差校正)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)   (参数更新)
// 20260325 ZJH Phase 3: GPU 参数使用 CUDABackend::adamStep 直接在 GPU 上更新
class Adam {
public:
    // 20260319 ZJH 构造函数
    // vecParams: 需要优化的参数指针向量
    // fLr: 学习率，默认 1e-3
    // fBeta1: 一阶矩衰减率，默认 0.9
    // fBeta2: 二阶矩衰减率，默认 0.999
    // fEps: 数值稳定性常数，默认 1e-8
    Adam(std::vector<Tensor*> vecParams, float fLr = 1e-3f, float fBeta1 = 0.9f,
         float fBeta2 = 0.999f, float fEps = 1e-8f)
        : m_vecParams(vecParams), m_fLr(fLr), m_fBeta1(fBeta1), m_fBeta2(fBeta2),
          m_fEps(fEps), m_nStep(0)
    {
        // 20260325 ZJH Phase 3: 一阶矩和二阶矩缓冲区在参数所在设备上创建
        m_vecM.resize(vecParams.size());  // 20260319 ZJH 一阶矩向量
        m_vecV.resize(vecParams.size());  // 20260319 ZJH 二阶矩向量
        for (size_t i = 0; i < vecParams.size(); ++i) {
            // 20260325 ZJH 在参数的设备上创建矩缓冲区（GPU 参数 → GPU m/v）
            m_vecM[i] = Tensor::zeros(vecParams[i]->shapeVec(), vecParams[i]->device());
            m_vecV[i] = Tensor::zeros(vecParams[i]->shapeVec(), vecParams[i]->device());
        }
    }

    // 20260331 ZJH addParams — 动态添加新参数到优化器（用于骨干解冻后加入低 LR 参数组）
    void addParams(std::vector<Tensor*> vecNewParams, float fLrMultiplier = 1.0f) {
        for (auto* pParam : vecNewParams) {
            m_vecParams.push_back(pParam);  // 20260331 ZJH 追加到参数列表
            m_vecLrMultipliers.push_back(fLrMultiplier);  // 20260331 ZJH 记录对应倍率
            // 20260331 ZJH 为新参数分配 m/v 缓冲区
            m_vecM.push_back(Tensor::zeros(pParam->shapeVec(), pParam->device()));
            m_vecV.push_back(Tensor::zeros(pParam->shapeVec(), pParam->device()));
        }
    }

    // 20260331 ZJH setParamLrMultipliers — 设置每个参数的学习率倍率向量
    void setParamLrMultipliers(const std::vector<float>& vecMultipliers) {
        m_vecLrMultipliers = vecMultipliers;
    }

    // 20260319 ZJH step — 执行一步 Adam 参数更新
    // 20260325 ZJH Phase 3: GPU 参数使用 CUDABackend::adamStep
    // 20260331 ZJH 支持分层学习率: 每个参数实际 LR = m_fLr × m_vecLrMultipliers[i]
    void step() {
        m_nStep++;  // 20260319 ZJH 递增步数计数器（用于偏差校正）

        for (size_t i = 0; i < m_vecParams.size(); ++i) {
            auto grad = tensorGetGrad(*m_vecParams[i]);  // 20260319 ZJH 获取当前参数的梯度
            if (grad.numel() == 0) continue;  // 20260319 ZJH 无梯度则跳过
            auto cGrad = grad.contiguous();  // 20260319 ZJH 确保梯度连续

            // 20260331 ZJH 计算此参数的实际学习率（baseLR × 倍率）
            float fEffectiveLr = m_fLr * getLrMultiplier(i);

            // 20260325 ZJH Phase 3: 检查参数是否在 CUDA 上
            if (m_vecParams[i]->isCuda()) {
#ifdef OM_HAS_CUDA
                // 20260325 ZJH 确保 m/v 缓冲区在 GPU 上（首次 step 时可能需要迁移）
                if (m_vecM[i].isCpu()) {
                    m_vecM[i] = m_vecM[i].cuda();  // 20260325 ZJH 迁移一阶矩到 GPU
                    m_vecV[i] = m_vecV[i].cuda();  // 20260325 ZJH 迁移二阶矩到 GPU
                }
                // 20260325 ZJH GPU 路径：参数/梯度/m/v 全部在 GPU 上，一次 kernel 完成 Adam 更新
                CUDABackend::adamStep(
                    m_vecParams[i]->mutableFloatDataPtr(),  // 20260325 ZJH GPU 参数指针
                    cGrad.floatDataPtr(),                    // 20260325 ZJH GPU 梯度指针
                    m_vecM[i].mutableFloatDataPtr(),          // 20260325 ZJH GPU 一阶矩指针
                    m_vecV[i].mutableFloatDataPtr(),          // 20260325 ZJH GPU 二阶矩指针
                    m_vecParams[i]->numel(),                 // 20260325 ZJH 元素总数
                    fEffectiveLr,                             // 20260331 ZJH 分层学习率
                    m_fBeta1,                                 // 20260325 ZJH 一阶矩衰减率
                    m_fBeta2,                                 // 20260325 ZJH 二阶矩衰减率
                    m_fEps,                                   // 20260325 ZJH 数值稳定性常数
                    m_nStep);                                 // 20260325 ZJH 当前步数
#endif
            } else {
                // 20260319 ZJH CPU 路径：原有 Adam 逻辑不变
                // 20260319 ZJH 计算 beta1^t 和 beta2^t 的幂次（偏差校正分母）
                float fBeta1Pow = std::pow(m_fBeta1, static_cast<float>(m_nStep));
                float fBeta2Pow = std::pow(m_fBeta2, static_cast<float>(m_nStep));

                float* pParam = m_vecParams[i]->mutableFloatDataPtr();  // 20260319 ZJH 参数可写指针
                const float* pGrad = cGrad.floatDataPtr();  // 20260319 ZJH 梯度只读指针
                float* pM = m_vecM[i].mutableFloatDataPtr();  // 20260319 ZJH 一阶矩可写指针
                float* pV = m_vecV[i].mutableFloatDataPtr();  // 20260319 ZJH 二阶矩可写指针
                int n = m_vecParams[i]->numel();  // 20260319 ZJH 参数元素总数

                for (int j = 0; j < n; ++j) {
                    // 20260319 ZJH 更新一阶矩：m = beta1 * m + (1 - beta1) * grad
                    pM[j] = m_fBeta1 * pM[j] + (1.0f - m_fBeta1) * pGrad[j];
                    // 20260319 ZJH 更新二阶矩：v = beta2 * v + (1 - beta2) * grad^2
                    pV[j] = m_fBeta2 * pV[j] + (1.0f - m_fBeta2) * pGrad[j] * pGrad[j];
                    // 20260319 ZJH 偏差校正
                    float fMhat = pM[j] / (1.0f - fBeta1Pow);  // 20260319 ZJH 校正一阶矩
                    float fVhat = pV[j] / (1.0f - fBeta2Pow);  // 20260319 ZJH 校正二阶矩
                    // 20260319 ZJH 参数更新：param -= lr * m_hat / (sqrt(v_hat) + eps)
                    pParam[j] -= fEffectiveLr * fMhat / (std::sqrt(fVhat) + m_fEps);
                }
            }
        }
    }

    // 20260319 ZJH zeroGrad — 清零所有参数的梯度
    void zeroGrad() {
        for (auto* pParam : m_vecParams) {
            tensorZeroGrad(*pParam);  // 20260319 ZJH 逐个清零参数梯度
        }
    }

    // 20260320 ZJH setLR — 动态设置学习率（用于学习率调度器）
    void setLR(float fLr) { m_fLr = fLr; }

    // 20260320 ZJH getLR — 获取当前学习率
    float getLR() const { return m_fLr; }

    // 20260326 ZJH 动态设置学习率（供 Cosine Annealing 调度器调用）
    void setLearningRate(float fNewLr) { m_fLr = fNewLr; }

private:
    // 20260331 ZJH 获取第 i 个参数的学习率倍率（无倍率向量时默认 1.0）
    float getLrMultiplier(size_t i) const {
        return (i < m_vecLrMultipliers.size()) ? m_vecLrMultipliers[i] : 1.0f;
    }

    std::vector<Tensor*> m_vecParams;  // 20260319 ZJH 参数指针列表
    float m_fLr;      // 20260319 ZJH 学习率
    float m_fBeta1;   // 20260319 ZJH 一阶矩衰减率
    float m_fBeta2;   // 20260319 ZJH 二阶矩衰减率
    float m_fEps;     // 20260319 ZJH 数值稳定性常数
    int m_nStep;      // 20260319 ZJH 当前步数计数器（从 0 开始，每次 step 递增）
    std::vector<Tensor> m_vecM;  // 20260319 ZJH 一阶矩（均值）缓冲区
    std::vector<Tensor> m_vecV;  // 20260319 ZJH 二阶矩（方差）缓冲区
    std::vector<float> m_vecLrMultipliers; // 20260331 ZJH 每参数学习率倍率（分层学习率）
};

// 20260320 ZJH AdamW — 带解耦权重衰减的 Adam 优化器
// 与标准 Adam 不同，AdamW 将权重衰减从梯度更新中解耦
// 参数更新：param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
// 20260325 ZJH Phase 3: GPU 参数使用 CUDABackend::adamStep + 额外权重衰减
class AdamW {
public:
    // 20260406 ZJH 构造函数
    // vecParams: 需要优化的参数指针向量（由 Module::parameters() 获取）
    // fLr: 学习率，默认 1e-3
    // fBeta1: 一阶矩衰减率，默认 0.9
    // fBeta2: 二阶矩衰减率，默认 0.999
    // fEps: 数值稳定性常数，默认 1e-8
    // fWeightDecay: 解耦权重衰减系数，默认 0.01
    AdamW(std::vector<Tensor*> vecParams, float fLr = 1e-3f, float fBeta1 = 0.9f,
          float fBeta2 = 0.999f, float fEps = 1e-8f, float fWeightDecay = 0.01f)
        : m_vecParams(vecParams), m_fLr(fLr), m_fBeta1(fBeta1), m_fBeta2(fBeta2),
          m_fEps(fEps), m_fWeightDecay(fWeightDecay), m_nStep(0)
    {
        // 20260325 ZJH Phase 3: 缓冲区在参数所在设备上创建
        m_vecM.resize(vecParams.size());  // 20260406 ZJH 一阶矩向量
        m_vecV.resize(vecParams.size());  // 20260406 ZJH 二阶矩向量
        for (size_t i = 0; i < vecParams.size(); ++i) {
            // 20260406 ZJH 在参数的设备上创建矩缓冲区（GPU 参数 → GPU m/v）
            m_vecM[i] = Tensor::zeros(vecParams[i]->shapeVec(), vecParams[i]->device());
            m_vecV[i] = Tensor::zeros(vecParams[i]->shapeVec(), vecParams[i]->device());
        }
    }

    // 20260331 ZJH addParams — 动态添加新参数到 AdamW（用于骨干解冻后加入低 LR 参数组）
    // vecNewParams: 新增参数指针向量
    // fLrMultiplier: 该组参数的学习率倍率（实际 LR = baseLR × multiplier）
    void addParams(std::vector<Tensor*> vecNewParams, float fLrMultiplier = 1.0f) {
        for (auto* pParam : vecNewParams) {
            m_vecParams.push_back(pParam);  // 20260406 ZJH 追加到参数列表
            m_vecLrMultipliers.push_back(fLrMultiplier);  // 20260406 ZJH 记录对应倍率
            // 20260406 ZJH 为新参数分配 m/v 缓冲区（在参数所在设备上）
            m_vecM.push_back(Tensor::zeros(pParam->shapeVec(), pParam->device()));
            m_vecV.push_back(Tensor::zeros(pParam->shapeVec(), pParam->device()));
        }
    }

    // 20260331 ZJH setParamLrMultipliers — 设置每个参数的学习率倍率向量
    void setParamLrMultipliers(const std::vector<float>& vecMultipliers) {
        m_vecLrMultipliers = vecMultipliers;
    }

    // 20260325 ZJH Phase 3: AdamW step 添加 GPU 路径
    // 20260331 ZJH 支持分层学习率: 每个参数实际 LR = m_fLr × m_vecLrMultipliers[i]
    void step() {
        m_nStep++;  // 20260406 ZJH 递增步数计数器（用于偏差校正）

        for (size_t i = 0; i < m_vecParams.size(); ++i) {
            auto grad = tensorGetGrad(*m_vecParams[i]);  // 20260406 ZJH 获取当前参数的梯度
            if (grad.numel() == 0) continue;  // 20260406 ZJH 无梯度则跳过该参数
            auto cGrad = grad.contiguous();  // 20260406 ZJH 确保梯度连续

            // 20260331 ZJH 计算此参数的实际学习率（baseLR × 倍率）
            float fEffectiveLr = m_fLr * getLrMultiplier(i);

            // 20260325 ZJH Phase 3: 检查参数是否在 CUDA 上
            if (m_vecParams[i]->isCuda()) {
#ifdef OM_HAS_CUDA
                // 20260325 ZJH 确保 m/v 缓冲区在 GPU 上
                if (m_vecM[i].isCpu()) {
                    m_vecM[i] = m_vecM[i].cuda();
                    m_vecV[i] = m_vecV[i].cuda();
                }
                // 20260325 ZJH GPU 路径：先用 CUDABackend::adamStep 完成 Adam 更新
                CUDABackend::adamStep(
                    m_vecParams[i]->mutableFloatDataPtr(),
                    cGrad.floatDataPtr(),
                    m_vecM[i].mutableFloatDataPtr(),
                    m_vecV[i].mutableFloatDataPtr(),
                    m_vecParams[i]->numel(),
                    fEffectiveLr, m_fBeta1, m_fBeta2, m_fEps, m_nStep);
                // 20260325 ZJH AdamW 额外步骤：解耦权重衰减 param *= (1 - lr * weight_decay)
                float fDecayFactor = 1.0f - fEffectiveLr * m_fWeightDecay;
                CUDABackend::mulScalar(
                    m_vecParams[i]->floatDataPtr(), fDecayFactor,
                    m_vecParams[i]->mutableFloatDataPtr(),
                    static_cast<size_t>(m_vecParams[i]->numel()));
#endif
            } else {
                // 20260320 ZJH CPU 路径：原有 AdamW 逻辑不变
                // 20260406 ZJH 计算 beta1^t 和 beta2^t 的幂次（偏差校正分母）
                float fBeta1Pow = std::pow(m_fBeta1, static_cast<float>(m_nStep));
                float fBeta2Pow = std::pow(m_fBeta2, static_cast<float>(m_nStep));

                float* pParam = m_vecParams[i]->mutableFloatDataPtr();  // 20260406 ZJH 参数可写指针
                const float* pGrad = cGrad.floatDataPtr();  // 20260406 ZJH 梯度只读指针
                float* pM = m_vecM[i].mutableFloatDataPtr();  // 20260406 ZJH 一阶矩可写指针
                float* pV = m_vecV[i].mutableFloatDataPtr();  // 20260406 ZJH 二阶矩可写指针
                int n = m_vecParams[i]->numel();  // 20260406 ZJH 参数元素总数

                for (int j = 0; j < n; ++j) {
                    // 20260320 ZJH 一阶矩和二阶矩更新
                    pM[j] = m_fBeta1 * pM[j] + (1.0f - m_fBeta1) * pGrad[j];  // 20260406 ZJH m = beta1*m + (1-beta1)*grad
                    pV[j] = m_fBeta2 * pV[j] + (1.0f - m_fBeta2) * pGrad[j] * pGrad[j];  // 20260406 ZJH v = beta2*v + (1-beta2)*grad²
                    float fMhat = pM[j] / (1.0f - fBeta1Pow);  // 20260406 ZJH 偏差校正一阶矩
                    float fVhat = pV[j] / (1.0f - fBeta2Pow);  // 20260406 ZJH 偏差校正二阶矩
                    // 20260320 ZJH AdamW 解耦权重衰减：使用分层 LR
                    // 20260406 ZJH 参数更新: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
                    pParam[j] -= fEffectiveLr * (fMhat / (std::sqrt(fVhat) + m_fEps) + m_fWeightDecay * pParam[j]);
                }
            }
        }
    }

    // 20260406 ZJH zeroGrad — 清零所有参数的梯度
    void zeroGrad() {
        for (auto* pParam : m_vecParams) tensorZeroGrad(*pParam);  // 20260406 ZJH 逐个清零参数梯度
    }

    // 20260406 ZJH setLR — 动态设置学习率（用于学习率调度器）
    void setLR(float fLr) { m_fLr = fLr; }
    // 20260406 ZJH getLR — 获取当前学习率
    float getLR() const { return m_fLr; }
    // 20260331 ZJH 动态设置学习率（供调度器调用）
    void setLearningRate(float fNewLr) { m_fLr = fNewLr; }

private:
    // 20260331 ZJH 获取第 i 个参数的学习率倍率（无倍率向量时默认 1.0）
    float getLrMultiplier(size_t i) const {
        return (i < m_vecLrMultipliers.size()) ? m_vecLrMultipliers[i] : 1.0f;  // 20260406 ZJH 默认倍率 1.0
    }

    std::vector<Tensor*> m_vecParams;  // 20260406 ZJH 参数指针列表
    float m_fLr, m_fBeta1, m_fBeta2, m_fEps, m_fWeightDecay;  // 20260406 ZJH 学习率、一阶矩衰减率、二阶矩衰减率、数值稳定常数、权重衰减系数
    int m_nStep;  // 20260406 ZJH 当前步数计数器（从 0 开始，每次 step 递增）
    std::vector<Tensor> m_vecM, m_vecV;  // 20260406 ZJH 一阶矩（均值）和二阶矩（方差）缓冲区
    std::vector<float> m_vecLrMultipliers; // 20260331 ZJH 每参数学习率倍率
};

}  // namespace om
