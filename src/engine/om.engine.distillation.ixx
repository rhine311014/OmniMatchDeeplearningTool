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
        // 20260330 ZJH 使用 tensor ops 替代原始指针循环，支持 CUDA 设备
        auto eDevice = studentLogits.device();  // 20260330 ZJH 记住输入设备
        float fT = fTemperature;  // 20260330 ZJH 温度参数
        float fT2 = fT * fT;  // 20260330 ZJH T^2 系数
        int nBatch = studentLogits.shape(0);  // 20260330 ZJH 批量大小

        // 20260330 ZJH 缩放 logits: logit / T
        auto scaledStudent = tensorMulScalar(studentLogits, 1.0f / fT);  // 20260330 ZJH 学生 logits / T
        auto scaledTeacher = tensorMulScalar(teacherLogits, 1.0f / fT);  // 20260330 ZJH 教师 logits / T

        // 20260330 ZJH softmax 生成软标签
        auto tSoftStudent = tensorSoftmaxLastDim(scaledStudent);  // 20260330 ZJH 学生软标签
        auto tSoftTeacher = tensorSoftmaxLastDim(scaledTeacher);  // 20260330 ZJH 教师软标签

        // 20260330 ZJH KL 散度: sum(teacher * log(teacher / student)) * T^2 / N
        // 20260330 ZJH 搬到 CPU 做标量计算（KL 散度是标量归约，CPU 足够）
        auto cpuStudent = tSoftStudent.contiguous();
        auto cpuTeacher = tSoftTeacher.contiguous();
        if (cpuStudent.isCuda()) cpuStudent = cpuStudent.cpu();  // 20260330 ZJH CUDA→CPU
        if (cpuTeacher.isCuda()) cpuTeacher = cpuTeacher.cpu();  // 20260330 ZJH CUDA→CPU
        int nClasses = cpuStudent.shape(1);  // 20260330 ZJH 类别数
        const float* pSS = cpuStudent.floatDataPtr();  // 20260330 ZJH 学生软标签数据指针
        const float* pST = cpuTeacher.floatDataPtr();  // 20260330 ZJH 教师软标签数据指针

        float fLoss = 0.0f;  // 20260330 ZJH 累计损失
        for (int i = 0; i < nBatch * nClasses; ++i) {
            float fTeach = pST[i];  // 20260330 ZJH 教师概率
            float fStud = std::max(pSS[i], 1e-7f);  // 20260330 ZJH 学生概率（下界防 log(0)）
            if (fTeach > 1e-7f) {
                fLoss += fTeach * std::log(fTeach / fStud);  // 20260330 ZJH KL 项
            }
        }
        fLoss = fLoss * fT2 / static_cast<float>(nBatch);  // 20260330 ZJH 缩放 T^2/N

        return Tensor::full({1}, fLoss, eDevice);  // 20260330 ZJH 在原始设备上返回结果
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
        // 20260330 ZJH 使用 tensor ops 替代原始指针循环，支持 CUDA 设备
        auto eDevice = studentFeat.device();  // 20260330 ZJH 记住输入设备
        auto diff = tensorSub(studentFeat, teacherFeat);  // 20260330 ZJH 差值张量
        auto diffSq = tensorMul(diff, diff);  // 20260330 ZJH 逐元素平方
        auto tSum = tensorSum(diffSq);  // 20260330 ZJH 全局求和（返回标量张量）
        int nNumel = std::min(studentFeat.numel(), teacherFeat.numel());  // 20260330 ZJH 元素数
        // 20260330 ZJH 除以元素数得到 MSE
        return tensorMulScalar(tSum, 1.0f / static_cast<float>(std::max(1, nNumel)));
    }
};

// 20260320 ZJH AttentionDistillLoss — 注意力蒸馏损失（AT）
// 对比教师和学生的注意力图（特征图沿通道维求和的平方）
class AttentionDistillLoss {
public:
    Tensor forward(const Tensor& studentFeat, const Tensor& teacherFeat) {
        // 20260330 ZJH 使用 CPU 路径 + 设备感知分配，支持 CUDA 输入
        auto eDevice = studentFeat.device();  // 20260330 ZJH 记住输入设备
        // 20260330 ZJH 搬到 CPU 做注意力图计算（通道维归约，CPU 足够）
        auto cS = studentFeat.contiguous();
        auto cT = teacherFeat.contiguous();
        if (cS.isCuda()) cS = cS.cpu();  // 20260330 ZJH CUDA→CPU
        if (cT.isCuda()) cT = cT.cpu();  // 20260330 ZJH CUDA→CPU
        int nBatch = cS.shape(0);  // 20260330 ZJH 批量大小
        int nCS = cS.shape(1);  // 20260330 ZJH 学生通道数
        int nCT = cT.shape(1);  // 20260330 ZJH 教师通道数
        int nH = cS.shape(2);  // 20260330 ZJH 特征图高度
        int nW = cS.shape(3);  // 20260330 ZJH 特征图宽度
        int nSpatial = nH * nW;  // 20260330 ZJH 空间维度总数

        // 20260320 ZJH 计算注意力图：A(x) = sum_c(F_c^2) 沿通道维度
        auto attnS = Tensor::zeros({nBatch, 1, nH, nW});  // 20260330 ZJH CPU 上分配
        auto attnT = Tensor::zeros({nBatch, 1, nH, nW});  // 20260330 ZJH CPU 上分配
        float* pAS = attnS.mutableFloatDataPtr();  // 20260330 ZJH 学生注意力图指针
        float* pAT = attnT.mutableFloatDataPtr();  // 20260330 ZJH 教师注意力图指针
        const float* pS = cS.floatDataPtr();  // 20260330 ZJH 学生特征数据指针
        const float* pT = cT.floatDataPtr();  // 20260330 ZJH 教师特征数据指针

        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nCS; ++c)
                for (int s = 0; s < nSpatial; ++s)
                    pAS[n * nSpatial + s] += pS[(n * nCS + c) * nSpatial + s] * pS[(n * nCS + c) * nSpatial + s];
            int nHTmin = std::min(nH, cT.shape(2));  // 20260330 ZJH 教师特征高度上限
            int nWTmin = std::min(nW, cT.shape(3));  // 20260330 ZJH 教师特征宽度上限
            for (int c = 0; c < nCT; ++c)
                for (int h = 0; h < nHTmin; ++h)
                    for (int w = 0; w < nWTmin; ++w)
                        pAT[n * nSpatial + h * nW + w] += pT[(n * nCT + c) * cT.shape(2) * cT.shape(3) + h * cT.shape(3) + w]
                                                          * pT[(n * nCT + c) * cT.shape(2) * cT.shape(3) + h * cT.shape(3) + w];
        }

        // 20260320 ZJH L2 归一化
        for (int n = 0; n < nBatch; ++n) {
            float fNormS = 0, fNormT = 0;  // 20260330 ZJH L2 范数累加器
            for (int s = 0; s < nSpatial; ++s) {
                fNormS += pAS[n * nSpatial + s] * pAS[n * nSpatial + s];
                fNormT += pAT[n * nSpatial + s] * pAT[n * nSpatial + s];
            }
            fNormS = std::sqrt(fNormS + 1e-8f);  // 20260330 ZJH 加 epsilon 防零
            fNormT = std::sqrt(fNormT + 1e-8f);  // 20260330 ZJH 加 epsilon 防零
            for (int s = 0; s < nSpatial; ++s) {
                pAS[n * nSpatial + s] /= fNormS;  // 20260330 ZJH 归一化学生注意力图
                pAT[n * nSpatial + s] /= fNormT;  // 20260330 ZJH 归一化教师注意力图
            }
        }

        // 20260320 ZJH L2 距离
        float fLoss = 0.0f;  // 20260330 ZJH 累计损失
        for (int i = 0; i < nBatch * nSpatial; ++i) {
            float d = pAS[i] - pAT[i];  // 20260330 ZJH 注意力图差值
            fLoss += d * d;  // 20260330 ZJH 平方累加
        }
        return Tensor::full({1}, fLoss / static_cast<float>(nBatch), eDevice);  // 20260330 ZJH 在原始设备上返回
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
    // 20260406 ZJH 构造函数 — 初始化蒸馏管理器配置
    // config: 蒸馏训练配置（温度、各损失权重等），默认使用 DistillConfig 默认值
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
        float fTemp = m_config.fTemperature;  // 20260406 ZJH 当前温度值，初始化为配置中的温度
        // 20260320 ZJH 渐进温度：从高温逐渐降到低温
        if (m_config.bProgressiveTemp && m_config.nTotalEpochs > 1) {
            // 20260406 ZJH 计算训练进度比例 [0, 1]
            float fProgress = static_cast<float>(nCurrentEpoch) / static_cast<float>(m_config.nTotalEpochs - 1);
            fTemp = m_config.fTemperature * (1.0f - fProgress * 0.5f);  // 20260406 ZJH 线性衰减: 从 T 降到 T/2
            fTemp = std::max(1.0f, fTemp);  // 20260406 ZJH 温度下限为 1.0（避免退化为硬标签）
        }

        float fTotal = 0.0f;  // 20260406 ZJH 组合损失累积值

        // 20260320 ZJH 硬标签损失
        float fHardLoss = hardLoss.item();  // 20260406 ZJH 从张量中提取标量损失值
        fTotal += m_config.fAlphaHard * fHardLoss;  // 20260406 ZJH 加权硬标签损失

        // 20260320 ZJH KL 散度软标签损失
        if (m_config.fAlphaSoft > 0.0f) {
            // 20260406 ZJH 软标签权重 > 0 时才计算 KL 散度（避免无意义计算）
            auto klLoss = m_klLoss.forward(studentLogits, teacherLogits, fTemp);  // 20260406 ZJH 计算 KL 散度损失
            fTotal += m_config.fAlphaSoft * klLoss.item();  // 20260406 ZJH 加权软标签损失
        }

        m_fLastHardLoss = fHardLoss;  // 20260406 ZJH 记录本次硬标签损失（供外部查询）
        m_fLastSoftLoss = fTotal - m_config.fAlphaHard * fHardLoss;  // 20260406 ZJH 记录本次软标签损失
        m_fLastTotalLoss = fTotal;  // 20260406 ZJH 记录本次总损失
        m_fLastTemperature = fTemp;  // 20260406 ZJH 记录本次使用的温度

        return fTotal;  // 20260406 ZJH 返回组合损失值
    }

    // 20260320 ZJH computeFeatureLoss — 计算特征蒸馏损失
    // 20260406 ZJH studentFeat: 学生中间特征 [N, Cs, H, W]
    // 20260406 ZJH teacherFeat: 教师中间特征 [N, Ct, H, W]（通过适配层后 Ct=Cs）
    // 20260406 ZJH 返回: 加权后的特征蒸馏损失标量值
    float computeFeatureLoss(const Tensor& studentFeat, const Tensor& teacherFeat) {
        if (m_config.fAlphaFeature <= 0.0f) return 0.0f;  // 20260406 ZJH 特征蒸馏权重为0时跳过
        auto loss = m_featLoss.forward(studentFeat, teacherFeat);  // 20260406 ZJH 计算 MSE 特征损失
        return m_config.fAlphaFeature * loss.item();  // 20260406 ZJH 返回加权损失值
    }

    // 20260320 ZJH computeAttentionLoss — 计算注意力蒸馏损失
    // 20260406 ZJH studentFeat: 学生特征图 [N, Cs, H, W]
    // 20260406 ZJH teacherFeat: 教师特征图 [N, Ct, H, W]
    // 20260406 ZJH 返回: 加权后的注意力蒸馏损失标量值
    float computeAttentionLoss(const Tensor& studentFeat, const Tensor& teacherFeat) {
        if (m_config.fAlphaAttention <= 0.0f) return 0.0f;  // 20260406 ZJH 注意力蒸馏权重为0时跳过
        auto loss = m_attnLoss.forward(studentFeat, teacherFeat);  // 20260406 ZJH 计算注意力图 L2 距离损失
        return m_config.fAlphaAttention * loss.item();  // 20260406 ZJH 返回加权损失值
    }

    // 20260320 ZJH 获取统计
    float lastHardLoss() const { return m_fLastHardLoss; }  // 20260406 ZJH 返回上一次的硬标签损失
    float lastSoftLoss() const { return m_fLastSoftLoss; }  // 20260406 ZJH 返回上一次的软标签损失
    float lastTotalLoss() const { return m_fLastTotalLoss; }  // 20260406 ZJH 返回上一次的总损失
    float lastTemperature() const { return m_fLastTemperature; }  // 20260406 ZJH 返回上一次使用的温度

    // 20260320 ZJH 计算压缩率
    // 20260406 ZJH compressionRatio — 计算教师/学生模型参数数量比（压缩率）
    // 20260406 ZJH teacher: 教师模型（参数量更大）
    // 20260406 ZJH student: 学生模型（参数量更小）
    // 20260406 ZJH 返回: 压缩率 = 教师参数数 / 学生参数数
    static float compressionRatio(Module& teacher, Module& student) {
        int nTeacher = 0, nStudent = 0;  // 20260406 ZJH 教师/学生参数计数
        for (auto* p : teacher.parameters()) nTeacher += p->numel();  // 20260406 ZJH 累加教师参数数
        for (auto* p : student.parameters()) nStudent += p->numel();  // 20260406 ZJH 累加学生参数数
        return nTeacher > 0 ? static_cast<float>(nTeacher) / static_cast<float>(nStudent) : 1.0f;  // 20260406 ZJH 返回压缩率
    }

    // 20260320 ZJH 计算模型大小（MB）
    // 20260406 ZJH model: 待计算的模型
    // 20260406 ZJH 返回: FP32 参数占用的内存大小（MB）
    static float modelSizeMB(Module& model) {
        int nParams = 0;  // 20260406 ZJH 参数元素总数
        for (auto* p : model.parameters()) nParams += p->numel();  // 20260406 ZJH 累加所有参数元素数
        return static_cast<float>(nParams * 4) / (1024.0f * 1024.0f);  // 20260406 ZJH FP32=4字节，转MB
    }

private:
    DistillConfig m_config;                  // 20260406 ZJH 蒸馏配置（温度、权重等）
    KLDivLoss m_klLoss;                      // 20260406 ZJH KL 散度损失计算器
    FeatureDistillLoss m_featLoss;           // 20260406 ZJH 特征蒸馏 MSE 损失计算器
    AttentionDistillLoss m_attnLoss;         // 20260406 ZJH 注意力蒸馏 L2 损失计算器
    float m_fLastHardLoss = 0.0f;            // 20260406 ZJH 上一次硬标签损失值
    float m_fLastSoftLoss = 0.0f;            // 20260406 ZJH 上一次软标签损失值
    float m_fLastTotalLoss = 0.0f;           // 20260406 ZJH 上一次总损失值
    float m_fLastTemperature = 4.0f;         // 20260406 ZJH 上一次使用的温度值
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
        auto params = model.parameters();  // 20260406 ZJH 获取模型所有可训练参数
        // 20260320 ZJH 收集所有权重绝对值
        std::vector<float> vecAbsWeights;  // 20260406 ZJH 存放所有权重绝对值
        int nTotal = 0;  // 20260406 ZJH 参数元素总数
        for (auto* p : params) {
            auto cp = p->contiguous();  // 20260406 ZJH 确保参数连续内存
            const float* pD = cp.floatDataPtr();  // 20260406 ZJH 获取数据只读指针
            int n = p->numel();  // 20260406 ZJH 当前参数元素数
            nTotal += n;  // 20260406 ZJH 累加总元素数
            for (int i = 0; i < n; ++i) vecAbsWeights.push_back(std::abs(pD[i]));  // 20260406 ZJH 收集绝对值
        }

        // 20260320 ZJH 找阈值（排序后取百分位）
        std::sort(vecAbsWeights.begin(), vecAbsWeights.end());  // 20260406 ZJH 升序排列所有绝对值
        int nPruneIdx = static_cast<int>(vecAbsWeights.size() * fSparsity);  // 20260406 ZJH 计算剪枝分位点索引
        nPruneIdx = std::min(nPruneIdx, static_cast<int>(vecAbsWeights.size()) - 1);  // 20260406 ZJH 防越界
        float fThreshold = vecAbsWeights[nPruneIdx];  // 20260406 ZJH 剪枝阈值

        // 20260320 ZJH 将小于阈值的权重置零
        int nPruned = 0;  // 20260406 ZJH 已剪枝计数
        for (auto* p : params) {
            float* pD = p->mutableFloatDataPtr();  // 20260406 ZJH 获取可写数据指针
            int n = p->numel();  // 20260406 ZJH 当前参数元素数
            for (int i = 0; i < n; ++i) {
                if (std::abs(pD[i]) <= fThreshold) {
                    pD[i] = 0.0f;  // 20260406 ZJH 将绝对值低于阈值的权重置零
                    nPruned++;  // 20260406 ZJH 剪枝计数+1
                }
            }
        }

        return nTotal > 0 ? static_cast<float>(nPruned) / static_cast<float>(nTotal) : 0.0f;  // 20260406 ZJH 返回实际稀疏率
    }

    // 20260320 ZJH sparsity — 计算当前模型稀疏率
    // 20260406 ZJH sparsity — 计算模型当前的零值参数比例
    // 20260406 ZJH model: 待分析的模型
    // 20260406 ZJH 返回: 稀疏率 = 零值参数数 / 总参数数
    static float sparsity(Module& model) {
        int nZero = 0, nTotal = 0;  // 20260406 ZJH 零值计数 / 总参数计数
        for (auto* p : model.parameters()) {
            auto cp = p->contiguous();  // 20260406 ZJH 确保连续内存
            const float* pD = cp.floatDataPtr();  // 20260406 ZJH 数据只读指针
            int n = p->numel();  // 20260406 ZJH 元素数
            nTotal += n;  // 20260406 ZJH 累加总元素数
            for (int i = 0; i < n; ++i) {
                if (std::abs(pD[i]) < 1e-8f) nZero++;  // 20260406 ZJH 绝对值<epsilon视为零
            }
        }
        return nTotal > 0 ? static_cast<float>(nZero) / static_cast<float>(nTotal) : 0.0f;  // 20260406 ZJH 返回稀疏率
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
        auto ci = input.contiguous();  // 20260406 ZJH 确保输入连续
        auto result = Tensor::zeros(ci.shapeVec());  // 20260406 ZJH 分配与输入同形状的输出
        const float* pIn = ci.floatDataPtr();  // 20260406 ZJH 输入数据只读指针
        float* pOut = result.mutableFloatDataPtr();  // 20260406 ZJH 输出数据可写指针
        int n = ci.numel();  // 20260406 ZJH 元素总数

        // 20260406 ZJH 逐元素执行 量化→反量化 模拟
        for (int i = 0; i < n; ++i) {
            // 20260320 ZJH 量化：q = round(x / scale) + zero_point
            float fQ = std::round(pIn[i] / fScale) + nZeroPoint;  // 20260406 ZJH 映射到整数域
            fQ = std::max(-128.0f, std::min(127.0f, fQ));  // 20260406 ZJH 钳位到 INT8 范围 [-128, 127]
            // 20260320 ZJH 反量化：x' = (q - zero_point) * scale
            pOut[i] = (fQ - nZeroPoint) * fScale;  // 20260406 ZJH 映射回浮点域（带量化误差）
        }
        return result;  // 20260406 ZJH 返回伪量化后的张量
    }

    // 20260320 ZJH calibrate — 校准量化参数（对称 Min-Max 方案）
    // 返回 {scale, zero_point}，对称量化 zero_point=0
    static std::pair<float, int> calibrate(const Tensor& input) {
        auto ci = input.contiguous();  // 20260406 ZJH 确保输入连续
        const float* pD = ci.floatDataPtr();  // 20260406 ZJH 数据只读指针
        int n = ci.numel();  // 20260406 ZJH 元素总数

        float fAbsMax = 0.0f;  // 20260406 ZJH 最大绝对值追踪
        for (int i = 0; i < n; ++i) {
            float a = std::abs(pD[i]);  // 20260406 ZJH 当前元素绝对值
            if (a > fAbsMax) fAbsMax = a;  // 20260406 ZJH 更新最大绝对值
        }

        // 20260320 ZJH 对称量化：scale = absMax / 127, zero_point = 0
        float fScale = fAbsMax / 127.0f;  // 20260406 ZJH 计算缩放因子
        if (fScale < 1e-8f) fScale = 1e-8f;  // 20260406 ZJH 防除零下限

        return {fScale, 0};  // 20260406 ZJH 返回 {scale, zeroPoint}
    }

    // 20260320 ZJH estimateCompression — 估算量化压缩率
    // FP32→INT8 = 4x, FP32→FP16 = 2x
    // 20260406 ZJH estimateCompression — 估算量化压缩率
    // 20260406 ZJH bInt8: true=INT8(4倍压缩), false=FP16(2倍压缩)
    // 20260406 ZJH 返回: 压缩倍率
    static float estimateCompression(bool bInt8 = true) {
        return bInt8 ? 4.0f : 2.0f;  // 20260406 ZJH FP32→INT8=4x, FP32→FP16=2x
    }
};

}  // namespace om
