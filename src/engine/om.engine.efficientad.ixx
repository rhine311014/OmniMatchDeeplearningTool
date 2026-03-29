// 20260322 ZJH EfficientAD 异常检测模块
// 实现教师-学生异常检测网络 (EfficientAD)
// 核心思想: 教师网络提取正常样本特征，学生网络模仿教师输出
//           推理时教师-学生输出差异（MSE）= 异常分数图
// 教师网络: 预训练或在正常样本上训练的轻量 CNN
// 学生网络: 与教师结构相同但随机初始化，通过知识蒸馏学习正常模式
// 异常区域: 学生无法复现教师输出 → MSE 高 → 异常
module;

#include <vector>
#include <string>
#include <utility>
#include <string>
#include <memory>
#include <cmath>

export module om.engine.efficientad;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;

export namespace om {

// 20260322 ZJH EfficientADBackbone — EfficientAD 共用骨干网络
// 4 层 Conv+BN+ReLU+MaxPool 结构，提取多尺度特征
// 通道变化: nInChannels → 32 → 64 → 128 → 256
// 空间变化: 每层 MaxPool stride=2 下采样，输出空间尺寸 = 输入 / 16
class EfficientADBackbone : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    EfficientADBackbone(int nInChannels = 3)
        : m_conv1(nInChannels, 32, 3, 1, 1, false),  // 20260322 ZJH 层1: Cin→32, 3x3, pad=1
          m_bn1(32),
          m_pool1(2, 2, 0),                            // 20260322 ZJH MaxPool 2x2 stride=2
          m_conv2(32, 64, 3, 1, 1, false),             // 20260322 ZJH 层2: 32→64
          m_bn2(64),
          m_pool2(2, 2, 0),
          m_conv3(64, 128, 3, 1, 1, false),            // 20260322 ZJH 层3: 64→128
          m_bn3(128),
          m_pool3(2, 2, 0),
          m_conv4(128, 256, 3, 1, 1, false),           // 20260322 ZJH 层4: 128→256
          m_bn4(256),
          m_pool4(2, 2, 0)
    {}

    // 20260322 ZJH forward — 骨干网络前向传播
    // input: [N, Cin, H, W]
    // 返回: [N, 256, H/16, W/16] 特征图
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 层1: Conv3x3 → BN → ReLU → MaxPool
        auto x = m_conv1.forward(input);
        x = m_bn1.forward(x);
        x = m_relu.forward(x);
        x = m_pool1.forward(x);   // 20260322 ZJH 空间下采样 /2

        // 20260322 ZJH 层2: Conv3x3 → BN → ReLU → MaxPool
        x = m_conv2.forward(x);
        x = m_bn2.forward(x);
        x = m_relu.forward(x);
        x = m_pool2.forward(x);   // 20260322 ZJH 空间下采样 /4

        // 20260322 ZJH 层3: Conv3x3 → BN → ReLU → MaxPool
        x = m_conv3.forward(x);
        x = m_bn3.forward(x);
        x = m_relu.forward(x);
        x = m_pool3.forward(x);   // 20260322 ZJH 空间下采样 /8

        // 20260322 ZJH 层4: Conv3x3 → BN → ReLU → MaxPool
        x = m_conv4.forward(x);
        x = m_bn4.forward(x);
        x = m_relu.forward(x);
        x = m_pool4.forward(x);   // 20260322 ZJH 空间下采样 /16

        return x;  // 20260322 ZJH 返回 [N, 256, H/16, W/16] 特征图
    }

    // 20260322 ZJH 提取中间特征（不经过最后一层 MaxPool）
    // 用于更高分辨率的异常定位
    // 返回: [N, 256, H/8, W/8] 特征图
    Tensor forwardFeatures(const Tensor& input) {
        auto x = m_conv1.forward(input);
        x = m_bn1.forward(x);
        x = m_relu.forward(x);
        x = m_pool1.forward(x);

        x = m_conv2.forward(x);
        x = m_bn2.forward(x);
        x = m_relu.forward(x);
        x = m_pool2.forward(x);

        x = m_conv3.forward(x);
        x = m_bn3.forward(x);
        x = m_relu.forward(x);
        x = m_pool3.forward(x);

        // 20260322 ZJH 最后一层不做 MaxPool，保留更高空间分辨率
        x = m_conv4.forward(x);
        x = m_bn4.forward(x);
        x = m_relu.forward(x);

        return x;  // 20260322 ZJH 返回 [N, 256, H/8, W/8] 高分辨率特征图
    }

    // 20260322 ZJH 重写 parameters()
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        appendVec(m_conv1.parameters());  appendVec(m_bn1.parameters());
        appendVec(m_conv2.parameters());  appendVec(m_bn2.parameters());
        appendVec(m_conv3.parameters());  appendVec(m_bn3.parameters());
        appendVec(m_conv4.parameters());  appendVec(m_bn4.parameters());
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        appendVec(m_conv1.namedParameters(makeP("conv1")));
        appendVec(m_bn1.namedParameters(makeP("bn1")));
        appendVec(m_conv2.namedParameters(makeP("conv2")));
        appendVec(m_bn2.namedParameters(makeP("bn2")));
        appendVec(m_conv3.namedParameters(makeP("conv3")));
        appendVec(m_bn3.namedParameters(makeP("bn3")));
        appendVec(m_conv4.namedParameters(makeP("conv4")));
        appendVec(m_bn4.namedParameters(makeP("bn4")));
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集 BN running stats
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto appendVec = [&](std::vector<Tensor*> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        appendVec(m_bn1.buffers());  // 20260328 ZJH bn1 running_mean/running_var
        appendVec(m_bn2.buffers());  // 20260328 ZJH bn2 running_mean/running_var
        appendVec(m_bn3.buffers());  // 20260328 ZJH bn3 running_mean/running_var
        appendVec(m_bn4.buffers());  // 20260328 ZJH bn4 running_mean/running_var
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集 BN 命名缓冲区
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto appendBufs = [&](const std::string& strSubPrefix, Module& mod) {
            std::string strFullPrefix = strPrefix.empty() ? strSubPrefix : strPrefix + "." + strSubPrefix;
            auto vecBufs = mod.namedBuffers(strFullPrefix);
            vecResult.insert(vecResult.end(), vecBufs.begin(), vecBufs.end());
        };
        appendBufs("bn1", m_bn1);
        appendBufs("bn2", m_bn2);
        appendBufs("bn3", m_bn3);
        appendBufs("bn4", m_bn4);
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_conv1.train(bMode);  m_bn1.train(bMode);
        m_conv2.train(bMode);  m_bn2.train(bMode);
        m_conv3.train(bMode);  m_bn3.train(bMode);
        m_conv4.train(bMode);  m_bn4.train(bMode);
    }

private:
    Conv2d m_conv1, m_conv2, m_conv3, m_conv4;        // 20260322 ZJH 4 层 3x3 卷积
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;            // 20260322 ZJH 4 层 BN
    MaxPool2d m_pool1, m_pool2, m_pool3, m_pool4;      // 20260322 ZJH 4 层 MaxPool
    ReLU m_relu;                                         // 20260322 ZJH ReLU 激活（无状态，共用）
};

// 20260322 ZJH EfficientAD — 教师-学生异常检测网络
// 持有教师网络 (teacher) 和学生网络 (student) 两个相同结构的骨干
// 训练阶段:
//   1. 教师网络先在正常样本上预训练（或使用预训练权重冻结）
//   2. 学生网络通过模仿教师输出进行知识蒸馏训练
//   3. 损失函数: MSE(teacher_features, student_features)
// 推理阶段:
//   1. 同时前向计算教师和学生的特征
//   2. 异常分数 = MSE(teacher_features, student_features)
//   3. 正常区域学生能准确复现教师输出（MSE低），异常区域学生失败（MSE高）
class EfficientAD : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    EfficientAD(int nInChannels = 3)
        : m_pTeacher(std::make_shared<EfficientADBackbone>(nInChannels)),  // 20260322 ZJH 教师网络
          m_pStudent(std::make_shared<EfficientADBackbone>(nInChannels)),  // 20260322 ZJH 学生网络
          // 20260328 ZJH 异常分数统计量（标量张量），用于自适应阈值校准
          // 存为 Tensor buffer 而非 float，利用现有序列化器自动保存/加载
          m_tScoreMean(Tensor::zeros({1})),   // 20260328 ZJH 正常样本异常分数均值
          m_tScoreStd(Tensor::zeros({1})),    // 20260328 ZJH 正常样本异常分数标准差
          m_tThreshold(Tensor::zeros({1}))    // 20260328 ZJH 异常判定阈值 (mean + k*std)
    {}

    // 20260322 ZJH forward — 计算异常分数图
    // input: [N, Cin, H, W]
    // 返回: [N, 1, H/8, W/8] 异常分数图（逐像素 MSE）
    // 注意: forward 默认使用 forwardFeatures（高分辨率）计算异常分数
    Tensor forward(const Tensor& input) override {
        return computeAnomalyScore(input);  // 20260322 ZJH 委托给异常分数计算
    }

    // 20260322 ZJH computeAnomalyScore — 计算逐像素异常分数图
    // 对教师和学生的特征图逐元素求 MSE 差异
    // input: [N, Cin, H, W]
    // 返回: [N, 1, Hf, Wf] 异常分数图（Hf=H/8, Wf=W/8）
    Tensor computeAnomalyScore(const Tensor& input) {
        // 20260322 ZJH 教师特征（评估模式，不更新 BN 统计量）
        auto teacherFeats = m_pTeacher->forwardFeatures(input);  // 20260322 ZJH [N, 256, H/8, W/8]
        // 20260322 ZJH 学生特征
        auto studentFeats = m_pStudent->forwardFeatures(input);  // 20260322 ZJH [N, 256, H/8, W/8]

        // 20260322 ZJH 计算逐元素差异: diff = teacher - student
        auto diff = tensorSub(teacherFeats, studentFeats);  // 20260322 ZJH [N, 256, H/8, W/8]

        // 20260322 ZJH 计算逐元素平方: diff^2
        auto diffSq = tensorMul(diff, diff);  // 20260322 ZJH [N, 256, H/8, W/8]

        // 20260322 ZJH 沿通道维求均值得到异常分数图
        // 手动计算通道均值: [N, 256, H, W] → [N, 1, H, W]
        auto cDiffSq = diffSq.contiguous();
        int nBatch = cDiffSq.shape(0);       // 20260322 ZJH 批次大小
        int nChannels = cDiffSq.shape(1);    // 20260322 ZJH 通道数 (256)
        int nH = cDiffSq.shape(2);           // 20260322 ZJH 特征图高度
        int nW = cDiffSq.shape(3);           // 20260322 ZJH 特征图宽度

        // 20260322 ZJH 创建输出异常分数图 [N, 1, H, W]
        auto anomalyMap = Tensor::zeros({nBatch, 1, nH, nW});
        const float* pDiffSq = cDiffSq.floatDataPtr();    // 20260322 ZJH 平方差数据指针
        float* pOut = anomalyMap.mutableFloatDataPtr();     // 20260322 ZJH 输出数据指针

        int nSpatial = nH * nW;  // 20260322 ZJH 空间元素数
        float fInvC = 1.0f / static_cast<float>(nChannels);  // 20260322 ZJH 通道均值缩放因子

        // 20260322 ZJH 对每个 batch 和空间位置，沿通道维求均值
        for (int n = 0; n < nBatch; ++n) {
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    float fSum = 0.0f;  // 20260322 ZJH 通道维累加
                    for (int c = 0; c < nChannels; ++c) {
                        // 20260322 ZJH 索引: [n, c, h, w] = n*C*H*W + c*H*W + h*W + w
                        int nIdx = n * nChannels * nSpatial + c * nSpatial + h * nW + w;
                        fSum += pDiffSq[nIdx];
                    }
                    // 20260322 ZJH 输出索引: [n, 0, h, w] = n*H*W + h*W + w
                    pOut[n * nSpatial + h * nW + w] = fSum * fInvC;
                }
            }
        }

        return anomalyMap;  // 20260322 ZJH 返回异常分数图 [N, 1, H/8, W/8]
    }

    // 20260322 ZJH computeDistillationLoss — 计算知识蒸馏损失（训练学生时使用）
    // MSE(teacher_features, student_features) 全局平均
    // input: [N, Cin, H, W]
    // 返回: 标量损失张量 [1]
    Tensor computeDistillationLoss(const Tensor& input) {
        // 20260322 ZJH 教师特征（梯度不需要流过教师网络）
        auto teacherFeats = m_pTeacher->forwardFeatures(input);
        // 20260322 ZJH 学生特征（梯度需要流过学生网络）
        auto studentFeats = m_pStudent->forwardFeatures(input);

        // 20260322 ZJH 计算 MSE 损失: mean((teacher - student)^2)
        auto diff = tensorSub(teacherFeats, studentFeats);
        auto diffSq = tensorMul(diff, diff);

        // 20260322 ZJH 全局求和再除以元素数得到 MSE
        auto loss = tensorSum(diffSq);  // 20260322 ZJH 求和
        float fNumel = static_cast<float>(diffSq.numel());  // 20260322 ZJH 元素总数
        loss = tensorMulScalar(loss, 1.0f / fNumel);  // 20260322 ZJH 平均化

        return loss;  // 20260322 ZJH 返回 MSE 损失标量
    }

    // 20260322 ZJH computeImageAnomalyScore — 计算整张图像的异常分数（标量）
    // 取异常分数图的最大值作为图像级异常分数
    // input: [N, Cin, H, W]
    // 返回: 异常分数（float）
    float computeImageAnomalyScore(const Tensor& input) {
        auto anomalyMap = computeAnomalyScore(input);  // 20260322 ZJH [N, 1, H/8, W/8]
        return tensorMax(anomalyMap);  // 20260322 ZJH 取最大值作为图像级异常分数
    }

    // 20260328 ZJH calibrate — 根据正常样本的异常分数分布校准阈值
    // vecScores: 所有正常训练样本的图像级异常分数（max of anomaly map）
    // fK: sigma 倍数，默认 3.0（3-sigma 规则：99.7% 正常样本在阈值以下）
    // 校准公式: threshold = mean + k * std
    void calibrate(const std::vector<float>& vecScores, float fK = 3.0f) {
        if (vecScores.empty()) return;  // 20260328 ZJH 空列表不校准

        // 20260328 ZJH 计算均值
        double dSum = 0.0;
        for (float fScore : vecScores) dSum += static_cast<double>(fScore);
        float fMean = static_cast<float>(dSum / static_cast<double>(vecScores.size()));

        // 20260328 ZJH 计算标准差
        double dVarSum = 0.0;
        for (float fScore : vecScores) {
            double dDiff = static_cast<double>(fScore) - static_cast<double>(fMean);
            dVarSum += dDiff * dDiff;
        }
        float fStd = static_cast<float>(std::sqrt(dVarSum / static_cast<double>(vecScores.size())));

        // 20260328 ZJH 防止 std=0（所有分数相同时），设置最小 std 为 mean 的 10%
        if (fStd < fMean * 0.1f) fStd = fMean * 0.1f;
        // 20260328 ZJH 极端情况：mean 也为 0，使用绝对下限
        if (fStd < 1e-6f) fStd = 1e-6f;

        // 20260328 ZJH 计算阈值: mean + k * std
        float fThreshold = fMean + fK * fStd;

        // 20260328 ZJH 写入张量 buffer（序列化器自动保存）
        m_tScoreMean.mutableFloatDataPtr()[0] = fMean;
        m_tScoreStd.mutableFloatDataPtr()[0] = fStd;
        m_tThreshold.mutableFloatDataPtr()[0] = fThreshold;

        m_bCalibrated = true;  // 20260328 ZJH 标记已校准
    }

    // 20260328 ZJH isCalibrated — 是否已完成阈值校准
    bool isCalibrated() const {
        // 20260328 ZJH threshold > 0 说明已校准（加载旧模型时 threshold=0 表示未校准）
        return m_bCalibrated || m_tThreshold.floatDataPtr()[0] > 0.0f;
    }

    // 20260328 ZJH anomalyThreshold — 获取异常判定阈值
    // 未校准时返回 fallback 值 0.5f（向后兼容旧模型）
    float anomalyThreshold() const {
        float fThreshold = m_tThreshold.floatDataPtr()[0];
        return (fThreshold > 0.0f) ? fThreshold : 0.5f;  // 20260328 ZJH fallback
    }

    // 20260328 ZJH scoreMean — 获取正常样本异常分数均值
    float scoreMean() const { return m_tScoreMean.floatDataPtr()[0]; }

    // 20260328 ZJH scoreStd — 获取正常样本异常分数标准差
    float scoreStd() const { return m_tScoreStd.floatDataPtr()[0]; }

    // 20260322 ZJH getTeacher — 获取教师网络（用于预训练或冻结）
    std::shared_ptr<EfficientADBackbone> getTeacher() { return m_pTeacher; }

    // 20260322 ZJH getStudent — 获取学生网络（用于蒸馏训练）
    std::shared_ptr<EfficientADBackbone> getStudent() { return m_pStudent; }

    // 20260322 ZJH freezeTeacher — 冻结教师网络参数（训练学生时调用）
    // 将教师网络所有参数的 requiresGrad 设置为 false
    void freezeTeacher() {
        m_pTeacher->eval();  // 20260322 ZJH 设置为评估模式
        // 20260322 ZJH 注意: 冻结梯度需要在优化器中排除教师参数
        // 实际训练时只将学生参数传给优化器即可
    }

    // 20260322 ZJH studentParameters — 仅返回学生网络的参数（用于优化器）
    std::vector<Tensor*> studentParameters() {
        return m_pStudent->parameters();  // 20260322 ZJH 仅学生参数
    }

    // 20260322 ZJH 重写 parameters() — 返回教师+学生所有参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecResult;
        auto vecTeacher = m_pTeacher->parameters();
        vecResult.insert(vecResult.end(), vecTeacher.begin(), vecTeacher.end());
        auto vecStudent = m_pStudent->parameters();
        vecResult.insert(vecResult.end(), vecStudent.begin(), vecStudent.end());
        return vecResult;
    }

    // 20260322 ZJH 重写 namedParameters()
    std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecP) {
            vecResult.insert(vecResult.end(), vecP.begin(), vecP.end());
        };
        appendVec(m_pTeacher->namedParameters(makeP("teacher")));
        appendVec(m_pStudent->namedParameters(makeP("student")));
        return vecResult;
    }

    // 20260328 ZJH 重写 buffers() 收集教师+学生 BN running stats + 阈值统计量
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecResult;
        auto vecTeacher = m_pTeacher->buffers();  // 20260328 ZJH 教师网络 BN 缓冲区
        vecResult.insert(vecResult.end(), vecTeacher.begin(), vecTeacher.end());
        auto vecStudent = m_pStudent->buffers();  // 20260328 ZJH 学生网络 BN 缓冲区
        vecResult.insert(vecResult.end(), vecStudent.begin(), vecStudent.end());
        // 20260328 ZJH 异常阈值统计量（校准后序列化保存）
        vecResult.push_back(&m_tScoreMean);
        vecResult.push_back(&m_tScoreStd);
        vecResult.push_back(&m_tThreshold);
        return vecResult;
    }

    // 20260328 ZJH 重写 namedBuffers() 收集教师+学生 BN 命名缓冲区 + 阈值统计量
    std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") override {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        auto makeP = [&](const std::string& s) -> std::string {
            return strPrefix.empty() ? s : strPrefix + "." + s;
        };
        auto appendVec = [&](std::vector<std::pair<std::string, Tensor*>> vecB) {
            vecResult.insert(vecResult.end(), vecB.begin(), vecB.end());
        };
        appendVec(m_pTeacher->namedBuffers(makeP("teacher")));
        appendVec(m_pStudent->namedBuffers(makeP("student")));
        // 20260328 ZJH 阈值统计量作为命名缓冲区，序列化器自动保存/加载
        vecResult.push_back({makeP("anomaly_score_mean"), &m_tScoreMean});
        vecResult.push_back({makeP("anomaly_score_std"), &m_tScoreStd});
        vecResult.push_back({makeP("anomaly_threshold"), &m_tThreshold});
        return vecResult;
    }

    // 20260322 ZJH 重写 train()
    void train(bool bMode = true) override {
        m_bTraining = bMode;
        m_pTeacher->train(bMode);
        m_pStudent->train(bMode);
    }

private:
    std::shared_ptr<EfficientADBackbone> m_pTeacher;  // 20260322 ZJH 教师网络
    std::shared_ptr<EfficientADBackbone> m_pStudent;  // 20260322 ZJH 学生网络

    // 20260328 ZJH 异常阈值校准统计量（存为 Tensor buffer，序列化器自动保存/加载）
    Tensor m_tScoreMean;   // 20260328 ZJH 正常样本异常分数均值 [1]
    Tensor m_tScoreStd;    // 20260328 ZJH 正常样本异常分数标准差 [1]
    Tensor m_tThreshold;   // 20260328 ZJH 异常判定阈值 = mean + k*std [1]
    bool m_bCalibrated = false;  // 20260328 ZJH 当前会话是否已校准（非持久化）
};

}  // namespace om
