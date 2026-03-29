// 20260322 ZJH PatchCore 异常检测模块
// 实现 PatchCore 异常检测算法（核心思想: 正常样本特征记忆库 + 最近邻距离）
// 训练阶段: 仅使用 OK（正常）图像，提取 patch 级特征并存入 MemoryBank
// 推理阶段: 提取测试图像 patch 特征 → 与 MemoryBank 中最近邻的距离 → 异常分数
// 优势: 无需训练网络权重，仅需正常样本建库；少样本场景下表现优异
// PatchCore 不继承 Module（非前向网络），使用独立的特征提取器
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cstdint>

export module om.engine.patchcore;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;

export namespace om {

// 20260322 ZJH PatchCoreExtractor — PatchCore 特征提取 CNN
// 4 层 Conv+BN+ReLU+MaxPool 结构（与 EfficientADBackbone 类似但独立实现）
// 提取 patch 级特征用于与 MemoryBank 比较
// 输入: [N, Cin, H, W]
// 输出特征图: [N, 256, H/8, W/8]，每个空间位置对应一个 256 维 patch 特征向量
class PatchCoreExtractor : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    PatchCoreExtractor(int nInChannels = 3)
        : m_conv1(nInChannels, 32, 3, 1, 1, false),  // 20260322 ZJH 层1: Cin→32
          m_bn1(32),
          m_pool1(2, 2, 0),                            // 20260322 ZJH MaxPool 下采样 /2
          m_conv2(32, 64, 3, 1, 1, false),             // 20260322 ZJH 层2: 32→64
          m_bn2(64),
          m_pool2(2, 2, 0),                            // 20260322 ZJH 下采样 /4
          m_conv3(64, 128, 3, 1, 1, false),            // 20260322 ZJH 层3: 64→128
          m_bn3(128),
          m_pool3(2, 2, 0),                            // 20260322 ZJH 下采样 /8
          m_conv4(128, 256, 3, 1, 1, false),           // 20260322 ZJH 层4: 128→256
          m_bn4(256)
          // 20260322 ZJH 最后一层不做 MaxPool，保留较高空间分辨率用于 patch 特征提取
    {}

    // 20260322 ZJH forward — 前向传播提取特征图
    // input: [N, Cin, H, W]
    // 返回: [N, 256, H/8, W/8] 特征图
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 层1: Conv3x3 → BN → ReLU → MaxPool
        auto x = m_conv1.forward(input);
        x = m_bn1.forward(x);
        x = m_relu.forward(x);
        x = m_pool1.forward(x);

        // 20260322 ZJH 层2
        x = m_conv2.forward(x);
        x = m_bn2.forward(x);
        x = m_relu.forward(x);
        x = m_pool2.forward(x);

        // 20260322 ZJH 层3
        x = m_conv3.forward(x);
        x = m_bn3.forward(x);
        x = m_relu.forward(x);
        x = m_pool3.forward(x);

        // 20260322 ZJH 层4（不做 MaxPool）
        x = m_conv4.forward(x);
        x = m_bn4.forward(x);
        x = m_relu.forward(x);

        return x;  // 20260322 ZJH 返回 [N, 256, H/8, W/8]
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

    // 20260322 ZJH getFeatureDim — 获取特征向量维度
    int getFeatureDim() const { return 256; }

private:
    Conv2d m_conv1, m_conv2, m_conv3, m_conv4;    // 20260322 ZJH 4 层 3x3 卷积
    BatchNorm2d m_bn1, m_bn2, m_bn3, m_bn4;        // 20260322 ZJH 4 层 BN
    MaxPool2d m_pool1, m_pool2, m_pool3;            // 20260322 ZJH 3 层 MaxPool（层4不做池化）
    ReLU m_relu;                                     // 20260322 ZJH ReLU 激活
};

// 20260322 ZJH PatchCore — PatchCore 异常检测器
// 核心思想: 基于正常样本 patch 特征的记忆库 + 最近邻距离异常检测
// 不继承 Module（非标准前向网络）
// 使用流程:
//   1. 创建 PatchCore 实例
//   2. 调用 buildMemoryBank() 传入正常样本图像列表
//   3. 推理时调用 computeAnomalyScore() 或 computeAnomalyMap()
class PatchCore {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    // nMaxMemorySize: MemoryBank 最大容量（coreset 采样上限），默认 1000
    PatchCore(int nInChannels = 3, int nMaxMemorySize = 1000)
        : m_extractor(nInChannels),
          m_nMaxMemorySize(nMaxMemorySize),
          m_nFeatureDim(256)  // 20260322 ZJH 特征向量维度，与 PatchCoreExtractor 输出通道一致
    {
        m_extractor.eval();  // 20260322 ZJH 特征提取器设置为评估模式
    }

    // 20260322 ZJH buildMemoryBank — 构建正常样本特征记忆库
    // 遍历所有正常图像，提取 patch 特征并存入 MemoryBank
    // 如果特征总量超过 m_nMaxMemorySize，使用随机采样压缩
    // vecNormalImages: 正常样本图像张量列表，每个为 [1, Cin, H, W]
    void buildMemoryBank(const std::vector<Tensor>& vecNormalImages) {
        m_vecMemoryBank.clear();  // 20260322 ZJH 清空旧记忆库

        // 20260322 ZJH 遍历所有正常图像，提取 patch 特征
        for (const auto& image : vecNormalImages) {
            // 20260322 ZJH 前向提取特征图: [1, Cin, H, W] → [1, 256, Hf, Wf]
            auto featureMap = m_extractor.forward(image);
            auto cFeats = featureMap.contiguous();

            int nChannels = cFeats.shape(1);  // 20260322 ZJH 特征通道数 (256)
            int nH = cFeats.shape(2);          // 20260322 ZJH 特征图高度
            int nW = cFeats.shape(3);          // 20260322 ZJH 特征图宽度
            const float* pData = cFeats.floatDataPtr();  // 20260322 ZJH 数据指针

            // 20260322 ZJH 遍历特征图每个空间位置，提取 256 维 patch 特征向量
            for (int h = 0; h < nH; ++h) {
                for (int w = 0; w < nW; ++w) {
                    std::vector<float> vecFeature(nChannels);  // 20260322 ZJH 单个 patch 特征向量
                    for (int c = 0; c < nChannels; ++c) {
                        // 20260322 ZJH 索引: [0, c, h, w] = c*H*W + h*W + w
                        vecFeature[c] = pData[c * nH * nW + h * nW + w];
                    }
                    m_vecMemoryBank.push_back(std::move(vecFeature));  // 20260322 ZJH 添加到记忆库
                }
            }
        }

        // 20260322 ZJH Coreset 随机采样: 如果记忆库过大，随机采样压缩到 m_nMaxMemorySize
        if (static_cast<int>(m_vecMemoryBank.size()) > m_nMaxMemorySize) {
            coresetSubsampling();  // 20260322 ZJH 执行 coreset 采样
        }
    }

    // 20260322 ZJH computeAnomalyScore — 计算整张图像的异常分数（标量）
    // 对测试图像提取 patch 特征，与 MemoryBank 求最近邻距离
    // 取所有 patch 中最大的最近邻距离作为图像级异常分数
    // testImage: [1, Cin, H, W] 测试图像
    // 返回: 图像级异常分数（越高越异常）
    float computeAnomalyScore(const Tensor& testImage) {
        auto anomalyMap = computeAnomalyMap(testImage);  // 20260322 ZJH 获取异常热力图
        return tensorMax(anomalyMap);  // 20260322 ZJH 取最大值作为图像级分数
    }

    // 20260322 ZJH computeAnomalyMap — 计算像素级异常热力图
    // 对测试图像每个 patch 位置计算与 MemoryBank 的最近邻距离
    // testImage: [1, Cin, H, W] 测试图像
    // 返回: [1, 1, Hf, Wf] 异常热力图（Hf=H/8, Wf=W/8）
    Tensor computeAnomalyMap(const Tensor& testImage) {
        // 20260322 ZJH 提取测试图像特征图
        auto featureMap = m_extractor.forward(testImage);
        auto cFeats = featureMap.contiguous();

        int nChannels = cFeats.shape(1);  // 20260322 ZJH 特征通道数 (256)
        int nH = cFeats.shape(2);          // 20260322 ZJH 特征图高度
        int nW = cFeats.shape(3);          // 20260322 ZJH 特征图宽度
        const float* pData = cFeats.floatDataPtr();

        // 20260322 ZJH 创建异常热力图 [1, 1, Hf, Wf]
        auto anomalyMap = Tensor::zeros({1, 1, nH, nW});
        float* pOut = anomalyMap.mutableFloatDataPtr();

        // 20260322 ZJH 对每个空间位置计算与 MemoryBank 的最近邻距离
        for (int h = 0; h < nH; ++h) {
            for (int w = 0; w < nW; ++w) {
                // 20260322 ZJH 提取当前位置的 patch 特征向量
                std::vector<float> vecQuery(nChannels);
                for (int c = 0; c < nChannels; ++c) {
                    vecQuery[c] = pData[c * nH * nW + h * nW + w];
                }

                // 20260322 ZJH 在 MemoryBank 中查找最近邻
                float fMinDist = findNearestNeighborDistance(vecQuery);

                // 20260322 ZJH 写入异常分数
                pOut[h * nW + w] = fMinDist;
            }
        }

        return anomalyMap;  // 20260322 ZJH 返回异常热力图
    }

    // 20260322 ZJH getMemoryBankSize — 获取当前记忆库大小
    int getMemoryBankSize() const {
        return static_cast<int>(m_vecMemoryBank.size());
    }

    // 20260322 ZJH getExtractor — 获取特征提取器（用于预训练）
    PatchCoreExtractor& getExtractor() { return m_extractor; }

    // 20260322 ZJH getExtractorParameters — 获取特征提取器参数
    std::vector<Tensor*> getExtractorParameters() {
        return m_extractor.parameters();
    }

private:
    // 20260322 ZJH findNearestNeighborDistance — 在 MemoryBank 中查找最近邻距离
    // vecQuery: 查询特征向量 [feature_dim]
    // 返回: 与 MemoryBank 中最近特征的欧氏距离
    float findNearestNeighborDistance(const std::vector<float>& vecQuery) const {
        float fMinDistSq = std::numeric_limits<float>::max();  // 20260322 ZJH 最小距离平方初始化为最大值

        // 20260322 ZJH 遍历 MemoryBank 中所有特征向量
        for (const auto& vecMemory : m_vecMemoryBank) {
            float fDistSq = 0.0f;  // 20260322 ZJH 欧氏距离平方累加
            for (int d = 0; d < m_nFeatureDim; ++d) {
                float fDiff = vecQuery[d] - vecMemory[d];  // 20260322 ZJH 逐维差值
                fDistSq += fDiff * fDiff;  // 20260322 ZJH 累加差值平方
            }
            // 20260322 ZJH 更新最近邻
            if (fDistSq < fMinDistSq) {
                fMinDistSq = fDistSq;
            }
        }

        // 20260322 ZJH 返回欧氏距离（开平方）
        return std::sqrt(fMinDistSq);
    }

    // 20260322 ZJH coresetSubsampling — Coreset 随机采样压缩记忆库
    // 从 m_vecMemoryBank 中随机采样 m_nMaxMemorySize 个特征向量
    // 保留最具代表性的特征子集，减少推理时的搜索开销
    void coresetSubsampling() {
        int nTotal = static_cast<int>(m_vecMemoryBank.size());
        if (nTotal <= m_nMaxMemorySize) return;  // 20260322 ZJH 不需要采样

        // 20260322 ZJH 生成随机索引
        std::vector<int> vecIndices(nTotal);
        std::iota(vecIndices.begin(), vecIndices.end(), 0);  // 20260322 ZJH 填充 0, 1, 2, ..., n-1

        // 20260322 ZJH Fisher-Yates 洗牌（只需前 m_nMaxMemorySize 个）
        // 使用简单线性同余生成器（LCG），避免依赖 <random>（C++20 模块中可能不可见）
        uint32_t nSeed = 42u;  // 20260322 ZJH 固定种子保证可重复性
        for (int i = 0; i < m_nMaxMemorySize; ++i) {
            // 20260322 ZJH LCG: seed = seed * 1664525 + 1013904223（Numerical Recipes 参数）
            nSeed = nSeed * 1664525u + 1013904223u;
            int nRange = nTotal - i;  // 20260322 ZJH 剩余可选范围
            int j = i + static_cast<int>(nSeed % static_cast<uint32_t>(nRange));  // 20260322 ZJH 随机索引
            std::swap(vecIndices[i], vecIndices[j]);  // 20260322 ZJH 交换
        }

        // 20260322 ZJH 按采样索引构建新的记忆库
        std::vector<std::vector<float>> vecSampled;
        vecSampled.reserve(m_nMaxMemorySize);
        for (int i = 0; i < m_nMaxMemorySize; ++i) {
            vecSampled.push_back(std::move(m_vecMemoryBank[vecIndices[i]]));
        }

        m_vecMemoryBank = std::move(vecSampled);  // 20260322 ZJH 替换为采样后的记忆库
    }

    PatchCoreExtractor m_extractor;  // 20260322 ZJH 特征提取 CNN
    int m_nMaxMemorySize;            // 20260322 ZJH MemoryBank 最大容量
    int m_nFeatureDim;               // 20260322 ZJH 特征向量维度 (256)

    // 20260322 ZJH MemoryBank: 存储正常样本 patch 级特征向量
    // 每个元素是一个 256 维 float 向量
    // 大小: 最多 m_nMaxMemorySize 个特征向量
    std::vector<std::vector<float>> m_vecMemoryBank;
};

}  // namespace om
