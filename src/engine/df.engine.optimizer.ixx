// 20260319 ZJH 优化器模块 — Phase 2 Part 1
// 实现 SGD（支持动量）和 Adam 两种优化器
module;

#include <vector>
#include <cmath>

export module df.engine.optimizer;

// 20260319 ZJH 导入依赖模块：张量类、张量运算、CPU 后端（直接操作数据指针）
import df.engine.tensor;
import df.engine.tensor_ops;
import df.hal.cpu_backend;

export namespace df {

// 20260319 ZJH SGD — 随机梯度下降优化器，支持可选动量
// 参数更新规则：
//   无动量: param -= lr * grad
//   有动量: velocity = momentum * velocity + grad; param -= lr * velocity
class SGD {
public:
    // 20260319 ZJH 构造函数
    // vecParams: 需要优化的参数指针向量（由 Module::parameters() 获取）
    // fLr: 学习率
    // fMomentum: 动量系数，0.0 表示不使用动量
    SGD(std::vector<Tensor*> vecParams, float fLr, float fMomentum = 0.0f)
        : m_vecParams(vecParams), m_fLr(fLr), m_fMomentum(fMomentum)
    {
        // 20260319 ZJH 若启用动量，为每个参数创建零初始化的速度缓冲区
        if (fMomentum > 0.0f) {
            m_vecVelocities.resize(vecParams.size());  // 20260319 ZJH 分配速度向量
            for (size_t i = 0; i < vecParams.size(); ++i) {
                m_vecVelocities[i] = Tensor::zeros(vecParams[i]->shapeVec());  // 20260319 ZJH 零初始化速度
            }
        }
    }

    // 20260319 ZJH step — 执行一步参数更新
    // 遍历所有参数，根据梯度和学习率更新参数值
    void step() {
        for (size_t i = 0; i < m_vecParams.size(); ++i) {
            auto grad = tensorGetGrad(*m_vecParams[i]);  // 20260319 ZJH 获取当前参数的梯度
            if (grad.numel() == 0) continue;  // 20260319 ZJH 无梯度则跳过该参数
            auto cGrad = grad.contiguous();  // 20260319 ZJH 确保梯度连续

            float* pParam = m_vecParams[i]->mutableFloatDataPtr();  // 20260319 ZJH 参数可写指针
            const float* pGrad = cGrad.floatDataPtr();  // 20260319 ZJH 梯度只读指针
            int n = m_vecParams[i]->numel();  // 20260319 ZJH 参数元素总数

            if (m_fMomentum > 0.0f) {
                // 20260319 ZJH 带动量的 SGD 更新
                float* pVel = m_vecVelocities[i].mutableFloatDataPtr();  // 20260319 ZJH 速度可写指针
                for (int j = 0; j < n; ++j) {
                    pVel[j] = m_fMomentum * pVel[j] + pGrad[j];  // 20260319 ZJH 更新速度：v = mu*v + grad
                    pParam[j] -= m_fLr * pVel[j];  // 20260319 ZJH 更新参数：param -= lr * v
                }
            } else {
                // 20260319 ZJH 普通 SGD 更新（无动量）
                for (int j = 0; j < n; ++j) {
                    pParam[j] -= m_fLr * pGrad[j];  // 20260319 ZJH param -= lr * grad
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

private:
    std::vector<Tensor*> m_vecParams;      // 20260319 ZJH 参数指针列表
    float m_fLr;                            // 20260319 ZJH 学习率
    float m_fMomentum;                      // 20260319 ZJH 动量系数
    std::vector<Tensor> m_vecVelocities;   // 20260319 ZJH 速度缓冲区（动量模式下使用）
};

// 20260319 ZJH Adam — 自适应矩估计优化器
// 参数更新规则：
//   m = beta1 * m + (1 - beta1) * grad          (一阶矩估计)
//   v = beta2 * v + (1 - beta2) * grad^2        (二阶矩估计)
//   m_hat = m / (1 - beta1^t)                   (偏差校正)
//   v_hat = v / (1 - beta2^t)                   (偏差校正)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)   (参数更新)
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
        // 20260319 ZJH 为每个参数创建零初始化的一阶矩和二阶矩缓冲区
        m_vecM.resize(vecParams.size());  // 20260319 ZJH 一阶矩向量
        m_vecV.resize(vecParams.size());  // 20260319 ZJH 二阶矩向量
        for (size_t i = 0; i < vecParams.size(); ++i) {
            m_vecM[i] = Tensor::zeros(vecParams[i]->shapeVec());  // 20260319 ZJH 零初始化一阶矩
            m_vecV[i] = Tensor::zeros(vecParams[i]->shapeVec());  // 20260319 ZJH 零初始化二阶矩
        }
    }

    // 20260319 ZJH step — 执行一步 Adam 参数更新
    void step() {
        m_nStep++;  // 20260319 ZJH 递增步数计数器（用于偏差校正）
        // 20260319 ZJH 计算 beta1^t 和 beta2^t 的幂次（偏差校正分母）
        float fBeta1Pow = std::pow(m_fBeta1, static_cast<float>(m_nStep));
        float fBeta2Pow = std::pow(m_fBeta2, static_cast<float>(m_nStep));

        for (size_t i = 0; i < m_vecParams.size(); ++i) {
            auto grad = tensorGetGrad(*m_vecParams[i]);  // 20260319 ZJH 获取当前参数的梯度
            if (grad.numel() == 0) continue;  // 20260319 ZJH 无梯度则跳过
            auto cGrad = grad.contiguous();  // 20260319 ZJH 确保梯度连续

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
                pParam[j] -= m_fLr * fMhat / (std::sqrt(fVhat) + m_fEps);
            }
        }
    }

    // 20260319 ZJH zeroGrad — 清零所有参数的梯度
    void zeroGrad() {
        for (auto* pParam : m_vecParams) {
            tensorZeroGrad(*pParam);  // 20260319 ZJH 逐个清零参数梯度
        }
    }

private:
    std::vector<Tensor*> m_vecParams;  // 20260319 ZJH 参数指针列表
    float m_fLr;      // 20260319 ZJH 学习率
    float m_fBeta1;   // 20260319 ZJH 一阶矩衰减率
    float m_fBeta2;   // 20260319 ZJH 二阶矩衰减率
    float m_fEps;     // 20260319 ZJH 数值稳定性常数
    int m_nStep;      // 20260319 ZJH 当前步数计数器（从 0 开始，每次 step 递增）
    std::vector<Tensor> m_vecM;  // 20260319 ZJH 一阶矩（均值）缓冲区
    std::vector<Tensor> m_vecV;  // 20260319 ZJH 二阶矩（方差）缓冲区
};

}  // namespace df
