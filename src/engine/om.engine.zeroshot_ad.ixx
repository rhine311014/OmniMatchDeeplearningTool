// 20260322 ZJH 零样本异常检测模块 — 基于特征分布的缺陷检测
// 核心思路：轻量 CNN 提取特征 → OK 图像建立正常分布（均值+方差）→ Mahalanobis 距离判定异常
// 只需 OK（正常）图像训练，无需标注，无需外部预训练模型或 ONNX
module;

#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <numeric>

export module om.engine.zeroshot_ad;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// 20260322 ZJH FeatureExtractorCNN — 轻量特征提取 CNN（4层，提取多尺度特征）
// 结构：Conv(in, 32, 3, 1, 1)+BN+ReLU+Pool → Conv(32, 64)+BN+ReLU+Pool → Conv(64, 128)+BN+ReLU+Pool → Conv(128, 256)+BN+ReLU
// 输出最终特征图 [B, 256, H/8, W/8]
class FeatureExtractorCNN : public Module {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，灰度图为 1，RGB 为 3
    FeatureExtractorCNN(int nInChannels = 1)
        : m_conv1(nInChannels, 32, 3, 1, 1),  // 20260322 ZJH 第1层卷积: [B, in, H, W] → [B, 32, H, W]
          m_bn1(32),                            // 20260322 ZJH 第1层批归一化
          m_relu1(),                             // 20260322 ZJH 第1层 ReLU 激活
          m_pool1(2),                            // 20260322 ZJH 第1层池化: [B, 32, H, W] → [B, 32, H/2, W/2]
          m_conv2(32, 64, 3, 1, 1),             // 20260322 ZJH 第2层卷积: [B, 32, H/2, W/2] → [B, 64, H/2, W/2]
          m_bn2(64),                             // 20260322 ZJH 第2层批归一化
          m_relu2(),                             // 20260322 ZJH 第2层 ReLU 激活
          m_pool2(2),                            // 20260322 ZJH 第2层池化: → [B, 64, H/4, W/4]
          m_conv3(64, 128, 3, 1, 1),            // 20260322 ZJH 第3层卷积: → [B, 128, H/4, W/4]
          m_bn3(128),                            // 20260322 ZJH 第3层批归一化
          m_relu3(),                             // 20260322 ZJH 第3层 ReLU 激活
          m_pool3(2),                            // 20260322 ZJH 第3层池化: → [B, 128, H/8, W/8]
          m_conv4(128, 256, 3, 1, 1),           // 20260322 ZJH 第4层卷积: → [B, 256, H/8, W/8]
          m_bn4(256),                            // 20260322 ZJH 第4层批归一化
          m_relu4()                              // 20260322 ZJH 第4层 ReLU 激活（无池化，保留空间分辨率）
    {
        // 20260322 ZJH 注册所有子模块，使参数管理和 train/eval 切换能递归传播
        registerModule("conv1", std::make_shared<Conv2d>(m_conv1));
        registerModule("bn1",   std::make_shared<BatchNorm2d>(m_bn1));
        registerModule("conv2", std::make_shared<Conv2d>(m_conv2));
        registerModule("bn2",   std::make_shared<BatchNorm2d>(m_bn2));
        registerModule("conv3", std::make_shared<Conv2d>(m_conv3));
        registerModule("bn3",   std::make_shared<BatchNorm2d>(m_bn3));
        registerModule("conv4", std::make_shared<Conv2d>(m_conv4));
        registerModule("bn4",   std::make_shared<BatchNorm2d>(m_bn4));
    }

    // 20260322 ZJH forward — 前向传播，提取特征图
    // input: [B, C, H, W] 输入图像张量
    // 返回: [B, 256, H/8, W/8] 特征图
    Tensor forward(const Tensor& input) override {
        // 20260322 ZJH 第1层：Conv+BN+ReLU+Pool
        Tensor x = m_conv1.forward(input);   // 20260322 ZJH [B, 32, H, W]
        x = m_bn1.forward(x);                // 20260322 ZJH 批归一化
        x = m_relu1.forward(x);              // 20260322 ZJH ReLU 激活
        x = m_pool1.forward(x);              // 20260322 ZJH [B, 32, H/2, W/2]

        // 20260322 ZJH 第2层：Conv+BN+ReLU+Pool
        x = m_conv2.forward(x);              // 20260322 ZJH [B, 64, H/2, W/2]
        x = m_bn2.forward(x);                // 20260322 ZJH 批归一化
        x = m_relu2.forward(x);              // 20260322 ZJH ReLU 激活
        x = m_pool2.forward(x);              // 20260322 ZJH [B, 64, H/4, W/4]

        // 20260322 ZJH 第3层：Conv+BN+ReLU+Pool
        x = m_conv3.forward(x);              // 20260322 ZJH [B, 128, H/4, W/4]
        x = m_bn3.forward(x);                // 20260322 ZJH 批归一化
        x = m_relu3.forward(x);              // 20260322 ZJH ReLU 激活
        x = m_pool3.forward(x);              // 20260322 ZJH [B, 128, H/8, W/8]

        // 20260322 ZJH 第4层：Conv+BN+ReLU（无池化，保留空间分辨率）
        x = m_conv4.forward(x);              // 20260322 ZJH [B, 256, H/8, W/8]
        x = m_bn4.forward(x);                // 20260322 ZJH 批归一化
        x = m_relu4.forward(x);              // 20260322 ZJH ReLU 激活

        return x;  // 20260322 ZJH 返回最终特征图 [B, 256, H/8, W/8]
    }

private:
    // 20260322 ZJH 第1层子模块：Conv(in, 32, 3, stride=1, pad=1) + BN + ReLU + MaxPool(2)
    Conv2d m_conv1;
    BatchNorm2d m_bn1;
    ReLU m_relu1;
    MaxPool2d m_pool1;

    // 20260322 ZJH 第2层子模块：Conv(32, 64, 3, stride=1, pad=1) + BN + ReLU + MaxPool(2)
    Conv2d m_conv2;
    BatchNorm2d m_bn2;
    ReLU m_relu2;
    MaxPool2d m_pool2;

    // 20260322 ZJH 第3层子模块：Conv(64, 128, 3, stride=1, pad=1) + BN + ReLU + MaxPool(2)
    Conv2d m_conv3;
    BatchNorm2d m_bn3;
    ReLU m_relu3;
    MaxPool2d m_pool3;

    // 20260322 ZJH 第4层子模块：Conv(128, 256, 3, stride=1, pad=1) + BN + ReLU
    Conv2d m_conv4;
    BatchNorm2d m_bn4;
    ReLU m_relu4;
};

// 20260322 ZJH ZeroShotAnomalyDetector — 零样本异常检测器
// 训练阶段：只用 OK 图像建立正常特征分布（Welford 在线算法计算均值+方差）
// 推理阶段：计算测试图像特征与正常分布的马氏距离作为异常分数
class ZeroShotAnomalyDetector {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，灰度图 1，RGB 3
    ZeroShotAnomalyDetector(int nInChannels = 1)
        : m_extractor(nInChannels), m_nInChannels(nInChannels) {
        // 20260322 ZJH 设置特征提取器为评估模式（BN 使用 running stats）
        m_extractor.eval();
    }

    // 20260322 ZJH train — 训练阶段：只用 OK 图像建立正常特征分布
    // 使用 Welford 在线算法逐张累加均值和方差，避免一次性加载所有特征到内存
    // vecOkImages: OK 图像张量列表，每个形状 [1, C, H, W]
    void train(const std::vector<Tensor>& vecOkImages) {
        // 20260322 ZJH 检查输入有效性
        if (vecOkImages.empty()) {
            throw std::invalid_argument("ZeroShotAnomalyDetector::train — empty OK image list");
        }

        // 20260322 ZJH 设置提取器为评估模式（使用 BN running stats 而非 batch stats）
        m_extractor.eval();

        // 20260322 ZJH 提取第一张 OK 图像的特征以确定特征图形状
        Tensor firstFeat = m_extractor.forward(vecOkImages[0]);  // 20260322 ZJH [1, 256, H', W']
        int nC = firstFeat.shape(1);  // 20260322 ZJH 通道数 = 256
        int nH = firstFeat.shape(2);  // 20260322 ZJH 特征图高度
        int nW = firstFeat.shape(3);  // 20260322 ZJH 特征图宽度

        // 20260322 ZJH 初始化均值和 M2 累加器（Welford 算法）
        // mean: 特征均值 [C, H', W']
        // m2: 差的平方和 [C, H', W']，最终 var = m2 / n
        m_meanFeatures = Tensor::zeros({nC, nH, nW});  // 20260322 ZJH 均值累加器
        Tensor m2 = Tensor::zeros({nC, nH, nW});       // 20260322 ZJH M2 累加器
        m_nTrainCount = 0;                              // 20260322 ZJH 已处理图像计数

        // 20260322 ZJH 获取可写指针用于在线更新
        float* pMean = m_meanFeatures.mutableFloatDataPtr();  // 20260322 ZJH 均值数据指针
        float* pM2   = m2.mutableFloatDataPtr();              // 20260322 ZJH M2 数据指针
        int nSpatialSize = nC * nH * nW;                      // 20260322 ZJH 特征图总元素数

        // 20260322 ZJH 遍历所有 OK 图像，逐张更新均值和方差
        for (size_t i = 0; i < vecOkImages.size(); ++i) {
            // 20260322 ZJH 提取当前图像的特征
            Tensor feat = m_extractor.forward(vecOkImages[i]);  // 20260322 ZJH [1, 256, H', W']
            Tensor cFeat = feat.contiguous();                    // 20260322 ZJH 确保连续内存
            const float* pFeat = cFeat.floatDataPtr();           // 20260322 ZJH 特征数据指针

            // 20260322 ZJH Welford 在线算法更新均值和 M2
            m_nTrainCount += 1;  // 20260322 ZJH 计数加 1
            float fN = static_cast<float>(m_nTrainCount);  // 20260322 ZJH 当前样本数

            for (int j = 0; j < nSpatialSize; ++j) {
                float fDelta = pFeat[j] - pMean[j];         // 20260322 ZJH 新值与旧均值的差
                pMean[j] += fDelta / fN;                     // 20260322 ZJH 更新均值：mean += delta / n
                float fDelta2 = pFeat[j] - pMean[j];        // 20260322 ZJH 新值与新均值的差
                pM2[j] += fDelta * fDelta2;                  // 20260322 ZJH 更新 M2：M2 += delta * delta2
            }
        }

        // 20260322 ZJH 计算方差：var = M2 / n（总体方差，非样本方差）
        m_varFeatures = Tensor::zeros({nC, nH, nW});          // 20260322 ZJH 分配方差张量
        float* pVar = m_varFeatures.mutableFloatDataPtr();     // 20260322 ZJH 方差数据指针
        float fN = static_cast<float>(m_nTrainCount);          // 20260322 ZJH 总样本数
        for (int j = 0; j < nSpatialSize; ++j) {
            // 20260322 ZJH 总体方差 = M2 / n，至少保留 eps 避免除零
            pVar[j] = (fN > 1.0f) ? (pM2[j] / fN) : 1.0f;   // 20260322 ZJH 单样本时方差设为 1
        }

        // 20260322 ZJH 标记训练完成
        m_bTrained = true;

        // 20260322 ZJH 保存特征图尺寸
        m_nFeatC = nC;  // 20260322 ZJH 特征通道数
        m_nFeatH = nH;  // 20260322 ZJH 特征图高度
        m_nFeatW = nW;  // 20260322 ZJH 特征图宽度
    }

    // 20260322 ZJH predict — 推理阶段：计算异常分数
    // 提取测试图像特征，与正常分布比较，返回标量异常分数（越大越异常）
    // testImage: [1, C, H, W] 测试图像张量
    // 返回: 异常分数（float 标量）
    float predict(const Tensor& testImage) {
        // 20260322 ZJH 检查是否已训练
        if (!m_bTrained) {
            throw std::runtime_error("ZeroShotAnomalyDetector::predict — model not trained");
        }

        // 20260322 ZJH 提取测试图像特征
        Tensor feat = m_extractor.forward(testImage);  // 20260322 ZJH [1, 256, H', W']
        Tensor cFeat = feat.contiguous();               // 20260322 ZJH 确保连续内存
        const float* pFeat = cFeat.floatDataPtr();       // 20260322 ZJH 特征数据指针

        // 20260322 ZJH 获取正常分布参数的数据指针
        const float* pMean = m_meanFeatures.floatDataPtr();  // 20260322 ZJH 均值指针
        const float* pVar  = m_varFeatures.floatDataPtr();   // 20260322 ZJH 方差指针

        // 20260322 ZJH 计算马氏距离：score = mean((feat - mean)^2 / (var + eps))
        int nSpatialSize = m_nFeatC * m_nFeatH * m_nFeatW;  // 20260322 ZJH 特征图总元素数
        float fScore = 0.0f;                                  // 20260322 ZJH 异常分数累加器
        constexpr float fEps = 1e-6f;                         // 20260322 ZJH 防除零常数

        for (int j = 0; j < nSpatialSize; ++j) {
            float fDiff = pFeat[j] - pMean[j];                // 20260322 ZJH 特征与均值的差
            fScore += (fDiff * fDiff) / (pVar[j] + fEps);     // 20260322 ZJH 累加归一化平方差
        }

        // 20260322 ZJH 取平均得到最终异常分数
        fScore /= static_cast<float>(nSpatialSize);           // 20260322 ZJH 平均化
        return fScore;  // 20260322 ZJH 返回异常分数
    }

    // 20260322 ZJH predictHeatmap — 生成像素级异常热力图
    // 对每个空间位置 (h, w) 沿通道维求 sum_c((feat[c,h,w] - mean[c,h,w])^2 / (var[c,h,w] + eps))
    // testImage: [1, C, H, W] 测试图像张量
    // 返回: [H', W'] 热力图张量，每个位置是该处的异常分数
    Tensor predictHeatmap(const Tensor& testImage) {
        // 20260322 ZJH 检查是否已训练
        if (!m_bTrained) {
            throw std::runtime_error("ZeroShotAnomalyDetector::predictHeatmap — model not trained");
        }

        // 20260322 ZJH 提取测试图像特征
        Tensor feat = m_extractor.forward(testImage);  // 20260322 ZJH [1, 256, H', W']
        Tensor cFeat = feat.contiguous();               // 20260322 ZJH 确保连续内存
        const float* pFeat = cFeat.floatDataPtr();       // 20260322 ZJH 特征数据指针

        // 20260322 ZJH 获取正常分布参数的数据指针
        const float* pMean = m_meanFeatures.floatDataPtr();  // 20260322 ZJH 均值指针
        const float* pVar  = m_varFeatures.floatDataPtr();   // 20260322 ZJH 方差指针

        // 20260322 ZJH 创建热力图张量 [H', W']
        Tensor heatmap = Tensor::zeros({m_nFeatH, m_nFeatW});  // 20260322 ZJH 初始化热力图
        float* pHeatmap = heatmap.mutableFloatDataPtr();        // 20260322 ZJH 热力图数据指针
        constexpr float fEps = 1e-6f;                           // 20260322 ZJH 防除零常数

        // 20260322 ZJH 对每个空间位置 (h, w) 沿通道维累加异常分数
        // 特征数据布局: [C, H', W']（batch=1 时忽略 batch 维）
        for (int h = 0; h < m_nFeatH; ++h) {
            for (int w = 0; w < m_nFeatW; ++w) {
                float fPixelScore = 0.0f;  // 20260322 ZJH 当前位置的异常分数
                for (int c = 0; c < m_nFeatC; ++c) {
                    // 20260322 ZJH 计算特征在 [c, h, w] 位置的线性索引
                    int nIdx = c * (m_nFeatH * m_nFeatW) + h * m_nFeatW + w;
                    float fDiff = pFeat[nIdx] - pMean[nIdx];               // 20260322 ZJH 特征与均值的差
                    fPixelScore += (fDiff * fDiff) / (pVar[nIdx] + fEps);  // 20260322 ZJH 累加归一化平方差
                }
                // 20260322 ZJH 写入热力图
                pHeatmap[h * m_nFeatW + w] = fPixelScore;  // 20260322 ZJH 热力图 [h, w] 位置
            }
        }

        return heatmap;  // 20260322 ZJH 返回异常热力图 [H', W']
    }

    // 20260322 ZJH setThreshold — 设置异常阈值
    // 异常分数超过此阈值则判定为 NG（异常）
    // fThreshold: 阈值，默认 0.5
    void setThreshold(float fThreshold) {
        m_fThreshold = fThreshold;  // 20260322 ZJH 更新阈值
    }

    // 20260322 ZJH threshold — 获取当前异常阈值
    float threshold() const {
        return m_fThreshold;  // 20260322 ZJH 返回阈值
    }

    // 20260322 ZJH isAnomaly — 判定是否异常
    // 调用 predict 获取异常分数，与阈值比较
    // testImage: [1, C, H, W] 测试图像张量
    // 返回: true=NG（异常），false=OK（正常）
    bool isAnomaly(const Tensor& testImage) {
        float fScore = predict(testImage);      // 20260322 ZJH 计算异常分数
        return fScore > m_fThreshold;            // 20260322 ZJH 分数 > 阈值则为异常
    }

    // 20260322 ZJH saveModel — 保存正常分布参数到二进制文件
    // 文件格式：[magic][nC][nH][nW][nTrainCount][threshold][mean 数据][var 数据]
    // strPath: 保存路径
    // 返回: true=成功, false=失败
    bool saveModel(const std::string& strPath) {
        // 20260322 ZJH 检查是否已训练
        if (!m_bTrained) {
            return false;  // 20260322 ZJH 未训练无法保存
        }

        // 20260322 ZJH 以二进制模式打开文件
        std::ofstream ofs(strPath, std::ios::binary);
        if (!ofs.is_open()) {
            return false;  // 20260322 ZJH 文件打开失败
        }

        // 20260322 ZJH 写入魔数用于文件格式校验
        const uint32_t nMagic = 0x5A534144;  // 20260322 ZJH "ZSAD" — ZeroShot Anomaly Detector
        ofs.write(reinterpret_cast<const char*>(&nMagic), sizeof(nMagic));

        // 20260322 ZJH 写入特征图尺寸和训练参数
        ofs.write(reinterpret_cast<const char*>(&m_nFeatC), sizeof(m_nFeatC));
        ofs.write(reinterpret_cast<const char*>(&m_nFeatH), sizeof(m_nFeatH));
        ofs.write(reinterpret_cast<const char*>(&m_nFeatW), sizeof(m_nFeatW));
        ofs.write(reinterpret_cast<const char*>(&m_nTrainCount), sizeof(m_nTrainCount));
        ofs.write(reinterpret_cast<const char*>(&m_fThreshold), sizeof(m_fThreshold));

        // 20260322 ZJH 写入均值数据
        int nSize = m_nFeatC * m_nFeatH * m_nFeatW;  // 20260322 ZJH 特征图总元素数
        ofs.write(reinterpret_cast<const char*>(m_meanFeatures.floatDataPtr()),
                  static_cast<std::streamsize>(nSize * sizeof(float)));

        // 20260322 ZJH 写入方差数据
        ofs.write(reinterpret_cast<const char*>(m_varFeatures.floatDataPtr()),
                  static_cast<std::streamsize>(nSize * sizeof(float)));

        ofs.close();  // 20260322 ZJH 关闭文件
        return true;  // 20260322 ZJH 保存成功
    }

    // 20260322 ZJH loadModel — 从二进制文件加载正常分布参数
    // strPath: 文件路径
    // 返回: true=成功, false=失败
    bool loadModel(const std::string& strPath) {
        // 20260322 ZJH 以二进制模式打开文件
        std::ifstream ifs(strPath, std::ios::binary);
        if (!ifs.is_open()) {
            return false;  // 20260322 ZJH 文件打开失败
        }

        // 20260322 ZJH 验证魔数
        uint32_t nMagic = 0;
        ifs.read(reinterpret_cast<char*>(&nMagic), sizeof(nMagic));
        if (nMagic != 0x5A534144) {
            return false;  // 20260322 ZJH 魔数不匹配，文件格式错误
        }

        // 20260322 ZJH 读取特征图尺寸和训练参数
        ifs.read(reinterpret_cast<char*>(&m_nFeatC), sizeof(m_nFeatC));
        ifs.read(reinterpret_cast<char*>(&m_nFeatH), sizeof(m_nFeatH));
        ifs.read(reinterpret_cast<char*>(&m_nFeatW), sizeof(m_nFeatW));
        ifs.read(reinterpret_cast<char*>(&m_nTrainCount), sizeof(m_nTrainCount));
        ifs.read(reinterpret_cast<char*>(&m_fThreshold), sizeof(m_fThreshold));

        // 20260322 ZJH 分配均值和方差张量
        int nSize = m_nFeatC * m_nFeatH * m_nFeatW;  // 20260322 ZJH 特征图总元素数
        m_meanFeatures = Tensor::zeros({m_nFeatC, m_nFeatH, m_nFeatW});
        m_varFeatures = Tensor::zeros({m_nFeatC, m_nFeatH, m_nFeatW});

        // 20260322 ZJH 读取均值数据
        ifs.read(reinterpret_cast<char*>(m_meanFeatures.mutableFloatDataPtr()),
                 static_cast<std::streamsize>(nSize * sizeof(float)));

        // 20260322 ZJH 读取方差数据
        ifs.read(reinterpret_cast<char*>(m_varFeatures.mutableFloatDataPtr()),
                 static_cast<std::streamsize>(nSize * sizeof(float)));

        ifs.close();  // 20260322 ZJH 关闭文件

        // 20260322 ZJH 标记已训练
        m_bTrained = true;

        return true;  // 20260322 ZJH 加载成功
    }

    // 20260322 ZJH isTrained — 查询模型是否已训练
    bool isTrained() const {
        return m_bTrained;  // 20260322 ZJH 返回训练状态
    }

    // 20260322 ZJH trainCount — 返回训练样本数
    int trainCount() const {
        return m_nTrainCount;  // 20260322 ZJH 返回已训练的 OK 图像数
    }

    // 20260322 ZJH featureShape — 返回特征图形状 {C, H', W'}
    std::vector<int> featureShape() const {
        return {m_nFeatC, m_nFeatH, m_nFeatW};  // 20260322 ZJH 返回特征图维度
    }

    // 20260322 ZJH autoThreshold — 自动计算阈值
    // 使用 OK 图像的异常分数统计（均值 + k * 标准差）设置阈值
    // vecOkImages: OK 图像列表（通常与训练集相同或子集）
    // fK: 标准差倍数，默认 3.0（3-sigma 规则）
    void autoThreshold(const std::vector<Tensor>& vecOkImages, float fK = 3.0f) {
        // 20260322 ZJH 检查是否已训练
        if (!m_bTrained) {
            throw std::runtime_error("ZeroShotAnomalyDetector::autoThreshold — model not trained");
        }

        // 20260322 ZJH 收集所有 OK 图像的异常分数
        std::vector<float> vecScores;  // 20260322 ZJH 异常分数列表
        vecScores.reserve(vecOkImages.size());

        for (size_t i = 0; i < vecOkImages.size(); ++i) {
            float fScore = predict(vecOkImages[i]);  // 20260322 ZJH 计算异常分数
            vecScores.push_back(fScore);              // 20260322 ZJH 记录分数
        }

        // 20260322 ZJH 计算分数的均值
        float fMean = 0.0f;
        for (float f : vecScores) {
            fMean += f;  // 20260322 ZJH 累加
        }
        fMean /= static_cast<float>(vecScores.size());  // 20260322 ZJH 求平均

        // 20260322 ZJH 计算分数的标准差
        float fVar = 0.0f;
        for (float f : vecScores) {
            float fD = f - fMean;     // 20260322 ZJH 差值
            fVar += fD * fD;          // 20260322 ZJH 累加平方差
        }
        fVar /= static_cast<float>(vecScores.size());  // 20260322 ZJH 方差
        float fStd = std::sqrt(fVar);                   // 20260322 ZJH 标准差

        // 20260322 ZJH 设置阈值为 mean + k * std
        m_fThreshold = fMean + fK * fStd;  // 20260322 ZJH 3-sigma 阈值
    }

private:
    // 20260322 ZJH 轻量特征提取 CNN
    FeatureExtractorCNN m_extractor;

    // 20260322 ZJH 输入图像通道数
    int m_nInChannels = 1;

    // 20260322 ZJH 正常特征分布参数
    Tensor m_meanFeatures;   // 20260322 ZJH [C, H', W'] 均值
    Tensor m_varFeatures;    // 20260322 ZJH [C, H', W'] 方差

    // 20260322 ZJH 模型状态
    bool m_bTrained = false;   // 20260322 ZJH 是否已训练
    float m_fThreshold = 0.5f; // 20260322 ZJH 异常阈值（predict 分数超过此值判定为 NG）
    int m_nTrainCount = 0;     // 20260322 ZJH 已训练的 OK 图像数

    // 20260322 ZJH 特征图尺寸（训练后确定）
    int m_nFeatC = 0;  // 20260322 ZJH 通道数（256）
    int m_nFeatH = 0;  // 20260322 ZJH 高度
    int m_nFeatW = 0;  // 20260322 ZJH 宽度
};

}  // namespace om
