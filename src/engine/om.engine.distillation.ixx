// 20260320 ZJH 知识蒸馏模块 — 教师-学生训练框架
// 支持 Logit 蒸馏（Hinton）、特征蒸馏（FitNet）、注意力蒸馏（AT）
// 目标：用大模型指导小模型，提高精度同时减小体积
module;

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

export module om.engine.distillation;

import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 蒸馏损失函数
// =========================================================

// 20260320 ZJH KLDivLoss — KL 散度损失
// 用于 Logit 蒸馏：比较教师和学生的软标签分布
// loss = KL(softmax(teacher/T) || softmax(student/T)) * T^2
class KLDivLoss {
public:
    // 20260320 ZJH forward — 计算 KL 散度损失
    // studentLogits: 学生模型输出 [N, C]
    // teacherLogits: 教师模型输出 [N, C]
    // fTemperature: 温度参数（越高越软，通常 3-20）
    // 返回: 标量损失
    Tensor forward(const Tensor& studentLogits, const Tensor& teacherLogits, float fTemperature) {
        auto cS = studentLogits.contiguous();
        auto cT = teacherLogits.contiguous();
        int nBatch = cS.shape(0);
        int nClasses = cS.shape(1);
        float fT = fTemperature;
        float fT2 = fT * fT;

        // 20260320 ZJH 计算软标签 softmax(logit / T)
        auto softStudent = Tensor::zeros({nBatch, nClasses});
        auto softTeacher = Tensor::zeros({nBatch, nClasses});
        const float* pS = cS.floatDataPtr();
        const float* pT = cT.floatDataPtr();
        float* pSS = softStudent.mutableFloatDataPtr();
        float* pST = softTeacher.mutableFloatDataPtr();

        for (int b = 0; b < nBatch; ++b) {
            // 20260320 ZJH 学生 softmax(logit/T)
            float fMaxS = pS[b * nClasses];
            float fMaxT = pT[b * nClasses];
            for (int c = 1; c < nClasses; ++c) {
                if (pS[b * nClasses + c] > fMaxS) fMaxS = pS[b * nClasses + c];
                if (pT[b * nClasses + c] > fMaxT) fMaxT = pT[b * nClasses + c];
            }
            float fSumS = 0, fSumT = 0;
            for (int c = 0; c < nClasses; ++c) {
                pSS[b * nClasses + c] = std::exp((pS[b * nClasses + c] - fMaxS) / fT);
                pST[b * nClasses + c] = std::exp((pT[b * nClasses + c] - fMaxT) / fT);
                fSumS += pSS[b * nClasses + c];
                fSumT += pST[b * nClasses + c];
            }
            for (int c = 0; c < nClasses; ++c) {
                pSS[b * nClasses + c] /= fSumS;
                pST[b * nClasses + c] /= fSumT;
            }
        }

        // 20260320 ZJH KL 散度：sum(teacher * log(teacher / student)) * T^2
        float fLoss = 0.0f;
        for (int i = 0; i < nBatch * nClasses; ++i) {
            float fTeach = pST[i];
            float fStud = std::max(pSS[i], 1e-7f);
            if (fTeach > 1e-7f) {
                fLoss += fTeach * std::log(fTeach / fStud);
            }
        }
        fLoss = fLoss * fT2 / static_cast<float>(nBatch);

        return Tensor::full({1}, fLoss);
    }
};

// 20260320 ZJH FeatureDistillLoss — 特征蒸馏损失（FitNet 风格）
// 学生的中间特征图 MSE 匹配教师的中间特征图
// 需要适配层将学生通道数映射到教师通道数
class FeatureDistillLoss {
public:
    // 20260320 ZJH forward — 计算特征蒸馏损失
    // studentFeat: 学生中间特征 [N, Cs, H, W]
    // teacherFeat: 教师中间特征 [N, Ct, H, W]（通过适配层后 Ct=Cs）
    // 返回: 标量 MSE 损失
    Tensor forward(const Tensor& studentFeat, const Tensor& teacherFeat) {
        auto cS = studentFeat.contiguous();
        auto cT = teacherFeat.contiguous();
        int nNumel = cS.numel();
        float fLoss = 0.0f;
        const float* pS = cS.floatDataPtr();
        const float* pT = cT.floatDataPtr();
        int nMin = std::min(nNumel, cT.numel());
        for (int i = 0; i < nMin; ++i) {
            float d = pS[i] - pT[i];
            fLoss += d * d;
        }
        return Tensor::full({1}, fLoss / static_cast<float>(nMin));
    }
};

// 20260320 ZJH AttentionDistillLoss — 注意力蒸馏损失（AT）
// 对比教师和学生的注意力图（特征图沿通道维求和的平方）
class AttentionDistillLoss {
public:
    Tensor forward(const Tensor& studentFeat, const Tensor& teacherFeat) {
        auto cS = studentFeat.contiguous();
        auto cT = teacherFeat.contiguous();
        int nBatch = cS.shape(0);
        int nCS = cS.shape(1);
        int nCT = cT.shape(1);
        int nH = cS.shape(2);
        int nW = cS.shape(3);
        int nSpatial = nH * nW;

        // 20260320 ZJH 计算注意力图：A(x) = sum_c(F_c^2) 沿通道维度
        auto attnS = Tensor::zeros({nBatch, 1, nH, nW});
        auto attnT = Tensor::zeros({nBatch, 1, nH, nW});
        float* pAS = attnS.mutableFloatDataPtr();
        float* pAT = attnT.mutableFloatDataPtr();
        const float* pS = cS.floatDataPtr();
        const float* pT = cT.floatDataPtr();

        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nCS; ++c)
                for (int s = 0; s < nSpatial; ++s)
                    pAS[n * nSpatial + s] += pS[(n * nCS + c) * nSpatial + s] * pS[(n * nCS + c) * nSpatial + s];
            int nHTmin = std::min(nH, cT.shape(2));
            int nWTmin = std::min(nW, cT.shape(3));
            for (int c = 0; c < nCT; ++c)
                for (int h = 0; h < nHTmin; ++h)
                    for (int w = 0; w < nWTmin; ++w)
                        pAT[n * nSpatial + h * nW + w] += pT[(n * nCT + c) * cT.shape(2) * cT.shape(3) + h * cT.shape(3) + w]
                                                          * pT[(n * nCT + c) * cT.shape(2) * cT.shape(3) + h * cT.shape(3) + w];
        }

        // 20260320 ZJH L2 归一化
        for (int n = 0; n < nBatch; ++n) {
            float fNormS = 0, fNormT = 0;
            for (int s = 0; s < nSpatial; ++s) {
                fNormS += pAS[n * nSpatial + s] * pAS[n * nSpatial + s];
                fNormT += pAT[n * nSpatial + s] * pAT[n * nSpatial + s];
            }
            fNormS = std::sqrt(fNormS + 1e-8f);
            fNormT = std::sqrt(fNormT + 1e-8f);
            for (int s = 0; s < nSpatial; ++s) {
                pAS[n * nSpatial + s] /= fNormS;
                pAT[n * nSpatial + s] /= fNormT;
            }
        }

        // 20260320 ZJH L2 距离
        float fLoss = 0.0f;
        for (int i = 0; i < nBatch * nSpatial; ++i) {
            float d = pAS[i] - pAT[i];
            fLoss += d * d;
        }
        return Tensor::full({1}, fLoss / static_cast<float>(nBatch));
    }
};

// =========================================================
// 蒸馏训练配置和管理器
// =========================================================

// 20260320 ZJH DistillConfig — 蒸馏训练配置
struct DistillConfig {
    float fTemperature = 4.0f;       // 20260320 ZJH 温度参数（Hinton 推荐 3-20）
    float fAlphaHard = 0.5f;         // 20260320 ZJH 硬标签损失权重
    float fAlphaSoft = 0.5f;         // 20260320 ZJH 软标签（KL）损失权重
    float fAlphaFeature = 0.0f;      // 20260320 ZJH 特征蒸馏损失权重（0=不用）
    float fAlphaAttention = 0.0f;    // 20260320 ZJH 注意力蒸馏损失权重（0=不用）
    bool bProgressiveTemp = false;   // 20260320 ZJH 渐进温度（从高到低）
    int nTotalEpochs = 100;          // 20260320 ZJH 总训练轮数（渐进温度用）
};

// 20260320 ZJH DistillationManager — 蒸馏训练管理器
// 组合多种蒸馏损失，管理教师-学生训练流程
class DistillationManager {
public:
    DistillationManager(const DistillConfig& config = {}) : m_config(config) {}

    // 20260320 ZJH computeLoss — 计算组合蒸馏损失
    // studentLogits: 学生输出 [N, C]
    // teacherLogits: 教师输出 [N, C]
    // targets: 真实标签 [N, C] one-hot
    // hardLoss: 预先计算的硬标签损失（如 CrossEntropy）
    // nCurrentEpoch: 当前 epoch（渐进温度用）
    // 返回: 组合损失值（float）
    float computeLoss(const Tensor& studentLogits, const Tensor& teacherLogits,
                       const Tensor& hardLoss, int nCurrentEpoch = 0) {
        float fTemp = m_config.fTemperature;
        // 20260320 ZJH 渐进温度：从高温逐渐降到低温
        if (m_config.bProgressiveTemp && m_config.nTotalEpochs > 1) {
            float fProgress = static_cast<float>(nCurrentEpoch) / static_cast<float>(m_config.nTotalEpochs - 1);
            fTemp = m_config.fTemperature * (1.0f - fProgress * 0.5f);  // 从 T 降到 T/2
            fTemp = std::max(1.0f, fTemp);
        }

        float fTotal = 0.0f;

        // 20260320 ZJH 硬标签损失
        float fHardLoss = hardLoss.item();
        fTotal += m_config.fAlphaHard * fHardLoss;

        // 20260320 ZJH KL 散度软标签损失
        if (m_config.fAlphaSoft > 0.0f) {
            auto klLoss = m_klLoss.forward(studentLogits, teacherLogits, fTemp);
            fTotal += m_config.fAlphaSoft * klLoss.item();
        }

        m_fLastHardLoss = fHardLoss;
        m_fLastSoftLoss = fTotal - m_config.fAlphaHard * fHardLoss;
        m_fLastTotalLoss = fTotal;
        m_fLastTemperature = fTemp;

        return fTotal;
    }

    // 20260320 ZJH computeFeatureLoss — 计算特征蒸馏损失
    float computeFeatureLoss(const Tensor& studentFeat, const Tensor& teacherFeat) {
        if (m_config.fAlphaFeature <= 0.0f) return 0.0f;
        auto loss = m_featLoss.forward(studentFeat, teacherFeat);
        return m_config.fAlphaFeature * loss.item();
    }

    // 20260320 ZJH computeAttentionLoss — 计算注意力蒸馏损失
    float computeAttentionLoss(const Tensor& studentFeat, const Tensor& teacherFeat) {
        if (m_config.fAlphaAttention <= 0.0f) return 0.0f;
        auto loss = m_attnLoss.forward(studentFeat, teacherFeat);
        return m_config.fAlphaAttention * loss.item();
    }

    // 20260320 ZJH 获取统计
    float lastHardLoss() const { return m_fLastHardLoss; }
    float lastSoftLoss() const { return m_fLastSoftLoss; }
    float lastTotalLoss() const { return m_fLastTotalLoss; }
    float lastTemperature() const { return m_fLastTemperature; }

    // 20260320 ZJH 计算压缩率
    static float compressionRatio(Module& teacher, Module& student) {
        int nTeacher = 0, nStudent = 0;
        for (auto* p : teacher.parameters()) nTeacher += p->numel();
        for (auto* p : student.parameters()) nStudent += p->numel();
        return nTeacher > 0 ? static_cast<float>(nTeacher) / static_cast<float>(nStudent) : 1.0f;
    }

    // 20260320 ZJH 计算模型大小（MB）
    static float modelSizeMB(Module& model) {
        int nParams = 0;
        for (auto* p : model.parameters()) nParams += p->numel();
        return static_cast<float>(nParams * 4) / (1024.0f * 1024.0f);
    }

private:
    DistillConfig m_config;
    KLDivLoss m_klLoss;
    FeatureDistillLoss m_featLoss;
    AttentionDistillLoss m_attnLoss;
    float m_fLastHardLoss = 0.0f;
    float m_fLastSoftLoss = 0.0f;
    float m_fLastTotalLoss = 0.0f;
    float m_fLastTemperature = 4.0f;
};

// =========================================================
// 模型剪枝（权重裁剪）
// =========================================================

// 20260320 ZJH ModelPruner — 权重剪枝
// 将绝对值最小的权重置零，减少有效参数
class ModelPruner {
public:
    // 20260320 ZJH prune — 对模型执行非结构化剪枝
    // fSparsity: 剪枝比例（0.3 = 剪掉 30% 的权重）
    // 返回: 实际剪枝后的稀疏率
    static float prune(Module& model, float fSparsity) {
        auto params = model.parameters();
        // 20260320 ZJH 收集所有权重绝对值
        std::vector<float> vecAbsWeights;
        int nTotal = 0;
        for (auto* p : params) {
            auto cp = p->contiguous();
            const float* pD = cp.floatDataPtr();
            int n = p->numel();
            nTotal += n;
            for (int i = 0; i < n; ++i) vecAbsWeights.push_back(std::abs(pD[i]));
        }

        // 20260320 ZJH 找阈值（排序后取百分位）
        std::sort(vecAbsWeights.begin(), vecAbsWeights.end());
        int nPruneIdx = static_cast<int>(vecAbsWeights.size() * fSparsity);
        nPruneIdx = std::min(nPruneIdx, static_cast<int>(vecAbsWeights.size()) - 1);
        float fThreshold = vecAbsWeights[nPruneIdx];

        // 20260320 ZJH 将小于阈值的权重置零
        int nPruned = 0;
        for (auto* p : params) {
            float* pD = p->mutableFloatDataPtr();
            int n = p->numel();
            for (int i = 0; i < n; ++i) {
                if (std::abs(pD[i]) <= fThreshold) {
                    pD[i] = 0.0f;
                    nPruned++;
                }
            }
        }

        return nTotal > 0 ? static_cast<float>(nPruned) / static_cast<float>(nTotal) : 0.0f;
    }

    // 20260320 ZJH sparsity — 计算当前模型稀疏率
    static float sparsity(Module& model) {
        int nZero = 0, nTotal = 0;
        for (auto* p : model.parameters()) {
            auto cp = p->contiguous();
            const float* pD = cp.floatDataPtr();
            int n = p->numel();
            nTotal += n;
            for (int i = 0; i < n; ++i) {
                if (std::abs(pD[i]) < 1e-8f) nZero++;
            }
        }
        return nTotal > 0 ? static_cast<float>(nZero) / static_cast<float>(nTotal) : 0.0f;
    }
};

// =========================================================
// 量化感知
// =========================================================

// 20260320 ZJH QuantizationHelper — 量化辅助工具
// INT8 伪量化：训练时模拟量化误差，推理时可真正量化
class QuantizationHelper {
public:
    // 20260320 ZJH fakeQuantize — 伪量化（训练用）
    // 将 float32 值 round 到 INT8 可表示的最近值
    static Tensor fakeQuantize(const Tensor& input, float fScale, int nZeroPoint) {
        auto ci = input.contiguous();
        auto result = Tensor::zeros(ci.shapeVec());
        const float* pIn = ci.floatDataPtr();
        float* pOut = result.mutableFloatDataPtr();
        int n = ci.numel();

        for (int i = 0; i < n; ++i) {
            // 20260320 ZJH 量化：q = round(x / scale) + zero_point
            float fQ = std::round(pIn[i] / fScale) + nZeroPoint;
            fQ = std::max(-128.0f, std::min(127.0f, fQ));
            // 20260320 ZJH 反量化：x' = (q - zero_point) * scale
            pOut[i] = (fQ - nZeroPoint) * fScale;
        }
        return result;
    }

    // 20260320 ZJH calibrate — 校准量化参数（对称 Min-Max 方案）
    // 返回 {scale, zero_point}，对称量化 zero_point=0
    static std::pair<float, int> calibrate(const Tensor& input) {
        auto ci = input.contiguous();
        const float* pD = ci.floatDataPtr();
        int n = ci.numel();

        float fAbsMax = 0.0f;
        for (int i = 0; i < n; ++i) {
            float a = std::abs(pD[i]);
            if (a > fAbsMax) fAbsMax = a;
        }

        // 20260320 ZJH 对称量化：scale = absMax / 127, zero_point = 0
        float fScale = fAbsMax / 127.0f;
        if (fScale < 1e-8f) fScale = 1e-8f;

        return {fScale, 0};
    }

    // 20260320 ZJH estimateCompression — 估算量化压缩率
    // FP32→INT8 = 4x, FP32→FP16 = 2x
    static float estimateCompression(bool bInt8 = true) {
        return bInt8 ? 4.0f : 2.0f;
    }
};

}  // namespace om
