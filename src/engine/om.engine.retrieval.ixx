// 20260330 ZJH 图像检索引擎模块 — 对标 HikRobot CNNRetrievalCpp
// 实现完整的图像检索流水线: CNN 特征提取 + 图库管理 + Top-K 余弦相似度检索
// 架构: FeatureExtractor(4-block CNN backbone) → L2 归一化 → GalleryManager → Top-K 匹配
// 支持: 图库增删改查 / 二进制序列化 / Triplet Loss 训练 / 多 ROI 检索
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <cassert>

export module om.engine.retrieval;

// 20260330 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.hal.cpu_backend;

export namespace om {

// =========================================================
// 20260330 ZJH 数据结构定义
// =========================================================

// 20260330 ZJH 图库样本 — 单张参考图像的特征向量 + 标签信息
// 特征向量已经过 L2 归一化，便于直接用内积计算余弦相似度
struct GallerySample {
    std::string strLabel;             // 20260330 ZJH 样本标签名（如 "OK", "NG_scratch"）
    std::vector<float> vecFeature;    // 20260330 ZJH L2 归一化后的特征向量
    int nLabelIndex = -1;             // 20260330 ZJH 标签在类别列表中的索引（-1 表示未分配）
    int nRoiIndex = 0;                // 20260330 ZJH ROI 区域索引（多 ROI 时使用，默认 0）
};

// 20260330 ZJH 检索匹配结果 — 单条匹配信息
struct RetrievalMatch {
    int nSampleIndex;      // 20260330 ZJH 匹配的样本在图库中的索引
    float fSimilarity;     // 20260330 ZJH 余弦相似度 [0, 1]（越大越相似）
    std::string strLabel;  // 20260330 ZJH 匹配样本的标签名
    int nLabelIndex;       // 20260330 ZJH 匹配样本的标签索引
};

// 20260330 ZJH ROI 区域定义 — 用于多 ROI 检索
// 坐标为归一化坐标 [0,1]，相对于输入图像尺寸
struct RetrievalROI {
    float fX;       // 20260330 ZJH 左上角 X 归一化坐标
    float fY;       // 20260330 ZJH 左上角 Y 归一化坐标
    float fWidth;   // 20260330 ZJH 宽度归一化比例
    float fHeight;  // 20260330 ZJH 高度归一化比例
    int nIndex;     // 20260330 ZJH ROI 编号（0 ~ 255，最多 256 个区域）
};

// 20260330 ZJH 图库统计信息
struct GalleryStats {
    int nTotalSamples;                           // 20260330 ZJH 总样本数
    int nClassCount;                             // 20260330 ZJH 类别数
    int nFeatureDim;                             // 20260330 ZJH 特征维度
    std::vector<std::string> vecClassNames;      // 20260330 ZJH 类别名列表
    std::vector<int> vecClassSampleCounts;       // 20260330 ZJH 每个类别的样本数
};

// =========================================================
// 20260330 ZJH 图库序列化格式（二进制）
// =========================================================
// 文件头: [魔数 "OMGR" 4字节] [版本号 uint32] [特征维度 uint32] [样本数 uint32] [类别数 uint32]
// 类别表: [类别数 × (名称长度 uint32 + 名称字节)]
// 样本表: [样本数 × (标签索引 int32 + ROI索引 int32 + 特征向量 float×dim)]

// 20260330 ZJH 图库文件魔数，标识 OmniMatch Gallery 格式
constexpr char GALLERY_MAGIC[4] = {'O', 'M', 'G', 'R'};

// 20260330 ZJH 图库文件当前版本号
constexpr uint32_t GALLERY_VERSION = 1;

// 20260330 ZJH 最大支持 ROI 数量（对标 HikRobot 256 区域限制）
constexpr int MAX_ROI_COUNT = 256;

// =========================================================
// 20260330 ZJH ConvBnReLU — 卷积 + 批归一化 + ReLU 复合块
// =========================================================

// 20260330 ZJH ConvBnReLU — 特征提取骨干网络的基础构建块
// 封装 Conv2d → BatchNorm2d → ReLU 三层结构，简化骨干网络搭建
class ConvBnReLU : public Module {
public:
    // 20260330 ZJH 构造函数
    // nIn: 输入通道数
    // nOut: 输出通道数
    // nKernel: 卷积核大小
    // nStride: 步幅，默认 1
    // nPad: 填充，默认 1（3×3 卷积 same padding）
    ConvBnReLU(int nIn, int nOut, int nKernel = 3, int nStride = 1, int nPad = 1)
        : m_conv(nIn, nOut, nKernel, nStride, nPad, true),  // 20260330 ZJH 带偏置卷积
          m_bn(nOut),                                         // 20260330 ZJH 批归一化
          m_relu()                                            // 20260330 ZJH ReLU 激活
    {
        // 20260330 ZJH 注册子模块，确保参数能被递归收集
        registerModule("conv", std::make_shared<Conv2d>(m_conv));
        registerModule("bn", std::make_shared<BatchNorm2d>(m_bn));
    }

    // 20260330 ZJH forward — 前向传播: Conv → BN → ReLU
    // input: [N, Cin, H, W]
    // 返回: [N, Cout, H', W']（H'/W' 取决于 stride 和 padding）
    Tensor forward(const Tensor& input) override {
        Tensor x = m_conv.forward(input);   // 20260330 ZJH 卷积
        x = m_bn.forward(x);                // 20260330 ZJH 批归一化
        x = m_relu.forward(x);              // 20260330 ZJH ReLU 激活
        return x;                            // 20260330 ZJH 返回激活后特征图
    }

    // 20260330 ZJH 递归收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecParams;                              // 20260330 ZJH 结果容器
        auto vecConv = m_conv.parameters();                          // 20260330 ZJH 卷积层参数
        auto vecBn = m_bn.parameters();                              // 20260330 ZJH BN 层参数
        vecParams.insert(vecParams.end(), vecConv.begin(), vecConv.end());    // 20260330 ZJH 合并卷积参数
        vecParams.insert(vecParams.end(), vecBn.begin(), vecBn.end());        // 20260330 ZJH 合并 BN 参数
        return vecParams;                                            // 20260330 ZJH 返回全部参数
    }

    // 20260330 ZJH 递归收集所有缓冲区（BN running stats）
    std::vector<Tensor*> buffers() override {
        return m_bn.buffers();  // 20260330 ZJH BN 层持有 running_mean 和 running_var
    }

private:
    Conv2d m_conv;        // 20260330 ZJH 卷积层
    BatchNorm2d m_bn;     // 20260330 ZJH 批归一化层
    ReLU m_relu;          // 20260330 ZJH ReLU 激活（无参数，不注册）
};

// =========================================================
// 20260330 ZJH FeatureExtractor — 特征提取骨干网络
// =========================================================

// 20260330 ZJH FeatureExtractor — 轻量级 CNN 提取定长特征向量
// 架构: 4 个下采样块（64→128→256→512）+ AdaptiveAvgPool + Linear 投影 + L2 归一化
// 每个块: ConvBnReLU(stride=1) → ConvBnReLU(stride=1) → MaxPool2d(2)
// 输出: [N, nFeatureDim] 的 L2 归一化特征向量
// 设计参考: 工业视觉特征提取通常使用轻量骨干，平衡速度与精度
class FeatureExtractor : public Module {
public:
    // 20260330 ZJH 构造函数
    // nInChannels: 输入图像通道数（1=灰度, 3=RGB）
    // nFeatureDim: 输出特征向量维度，默认 512
    FeatureExtractor(int nInChannels = 3, int nFeatureDim = 512)
        : m_nInChannels(nInChannels),    // 20260330 ZJH 保存输入通道数
          m_nFeatureDim(nFeatureDim),    // 20260330 ZJH 保存特征维度
          // 20260330 ZJH Block 1: 输入通道 → 64 通道，2 层卷积 + MaxPool 下采样
          m_block1Conv1(nInChannels, 64, 3, 1, 1),  // 20260330 ZJH 第1层: Cin→64, 3×3, stride=1, pad=1
          m_block1Conv2(64, 64, 3, 1, 1),            // 20260330 ZJH 第2层: 64→64, 3×3, stride=1, pad=1
          m_pool1(2),                                 // 20260330 ZJH MaxPool 2×2 下采样（H/2, W/2）
          // 20260330 ZJH Block 2: 64 → 128 通道
          m_block2Conv1(64, 128, 3, 1, 1),           // 20260330 ZJH 第1层: 64→128
          m_block2Conv2(128, 128, 3, 1, 1),          // 20260330 ZJH 第2层: 128→128
          m_pool2(2),                                 // 20260330 ZJH MaxPool 2×2 下采样
          // 20260330 ZJH Block 3: 128 → 256 通道
          m_block3Conv1(128, 256, 3, 1, 1),          // 20260330 ZJH 第1层: 128→256
          m_block3Conv2(256, 256, 3, 1, 1),          // 20260330 ZJH 第2层: 256→256
          m_pool3(2),                                 // 20260330 ZJH MaxPool 2×2 下采样
          // 20260330 ZJH Block 4: 256 → 512 通道
          m_block4Conv1(256, 512, 3, 1, 1),          // 20260330 ZJH 第1层: 256→512
          m_block4Conv2(512, 512, 3, 1, 1),          // 20260330 ZJH 第2层: 512→512
          m_pool4(2),                                 // 20260330 ZJH MaxPool 2×2 下采样
          // 20260330 ZJH 全局自适应平均池化: [N, 512, H', W'] → [N, 512, 1, 1]
          m_adaptivePool(1, 1),
          // 20260330 ZJH 展平层: [N, 512, 1, 1] → [N, 512]
          m_flatten(1),
          // 20260330 ZJH 线性投影层: 512 → nFeatureDim（当 nFeatureDim != 512 时进行维度变换）
          m_fc(512, nFeatureDim, false)               // 20260330 ZJH 无偏置，纯线性投影
    {
        // 20260330 ZJH 注册所有子模块，确保参数递归收集和训练模式切换正确
        registerModule("block1_conv1", std::make_shared<ConvBnReLU>(m_block1Conv1));
        registerModule("block1_conv2", std::make_shared<ConvBnReLU>(m_block1Conv2));
        registerModule("block2_conv1", std::make_shared<ConvBnReLU>(m_block2Conv1));
        registerModule("block2_conv2", std::make_shared<ConvBnReLU>(m_block2Conv2));
        registerModule("block3_conv1", std::make_shared<ConvBnReLU>(m_block3Conv1));
        registerModule("block3_conv2", std::make_shared<ConvBnReLU>(m_block3Conv2));
        registerModule("block4_conv1", std::make_shared<ConvBnReLU>(m_block4Conv1));
        registerModule("block4_conv2", std::make_shared<ConvBnReLU>(m_block4Conv2));
        registerModule("fc", std::make_shared<Linear>(m_fc));
    }

    // 20260330 ZJH forward — 前向传播: 输入图像 → L2 归一化特征向量
    // input: [N, C, H, W] 输入图像张量（推荐 H,W >= 32，因为有4次2x下采样）
    // 返回: [N, nFeatureDim] L2 归一化后的特征向量
    Tensor forward(const Tensor& input) override {
        // 20260330 ZJH Block 1: 输入 → 64 通道，2x 下采样
        Tensor x = m_block1Conv1.forward(input);   // 20260330 ZJH [N, Cin, H, W] → [N, 64, H, W]
        x = m_block1Conv2.forward(x);               // 20260330 ZJH [N, 64, H, W] → [N, 64, H, W]
        x = m_pool1.forward(x);                     // 20260330 ZJH [N, 64, H, W] → [N, 64, H/2, W/2]

        // 20260330 ZJH Block 2: 64 → 128 通道，2x 下采样
        x = m_block2Conv1.forward(x);               // 20260330 ZJH [N, 64, H/2, W/2] → [N, 128, H/2, W/2]
        x = m_block2Conv2.forward(x);               // 20260330 ZJH [N, 128, H/2, W/2] → [N, 128, H/2, W/2]
        x = m_pool2.forward(x);                     // 20260330 ZJH [N, 128, H/2, W/2] → [N, 128, H/4, W/4]

        // 20260330 ZJH Block 3: 128 → 256 通道，2x 下采样
        x = m_block3Conv1.forward(x);               // 20260330 ZJH [N, 128, H/4, W/4] → [N, 256, H/4, W/4]
        x = m_block3Conv2.forward(x);               // 20260330 ZJH [N, 256, H/4, W/4] → [N, 256, H/4, W/4]
        x = m_pool3.forward(x);                     // 20260330 ZJH [N, 256, H/4, W/4] → [N, 256, H/8, W/8]

        // 20260330 ZJH Block 4: 256 → 512 通道，2x 下采样
        x = m_block4Conv1.forward(x);               // 20260330 ZJH [N, 256, H/8, W/8] → [N, 512, H/8, W/8]
        x = m_block4Conv2.forward(x);               // 20260330 ZJH [N, 512, H/8, W/8] → [N, 512, H/8, W/8]
        x = m_pool4.forward(x);                     // 20260330 ZJH [N, 512, H/8, W/8] → [N, 512, H/16, W/16]

        // 20260330 ZJH 全局池化 + 展平: [N, 512, H/16, W/16] → [N, 512]
        x = m_adaptivePool.forward(x);              // 20260330 ZJH → [N, 512, 1, 1]
        x = m_flatten.forward(x);                   // 20260330 ZJH → [N, 512]

        // 20260330 ZJH 线性投影到目标特征维度
        x = m_fc.forward(x);                        // 20260330 ZJH [N, 512] → [N, nFeatureDim]

        // 20260330 ZJH L2 归一化: 使每个特征向量的 L2 范数为 1
        // 归一化后内积等价于余弦相似度，简化检索计算
        x = l2Normalize(x);                         // 20260330 ZJH L2 归一化

        return x;  // 20260330 ZJH 返回归一化特征向量
    }

    // 20260330 ZJH 获取特征维度
    int featureDim() const { return m_nFeatureDim; }

    // 20260330 ZJH 获取输入通道数
    int inChannels() const { return m_nInChannels; }

    // 20260330 ZJH 递归收集所有可训练参数
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> vecParams;  // 20260330 ZJH 结果容器
        // 20260330 ZJH 收集 4 个 block 的 8 个 ConvBnReLU 层参数
        auto addParams = [&vecParams](Module& mod) {
            auto v = mod.parameters();  // 20260330 ZJH 获取子模块参数
            vecParams.insert(vecParams.end(), v.begin(), v.end());  // 20260330 ZJH 合并
        };
        addParams(m_block1Conv1);  // 20260330 ZJH Block1 第1层
        addParams(m_block1Conv2);  // 20260330 ZJH Block1 第2层
        addParams(m_block2Conv1);  // 20260330 ZJH Block2 第1层
        addParams(m_block2Conv2);  // 20260330 ZJH Block2 第2层
        addParams(m_block3Conv1);  // 20260330 ZJH Block3 第1层
        addParams(m_block3Conv2);  // 20260330 ZJH Block3 第2层
        addParams(m_block4Conv1);  // 20260330 ZJH Block4 第1层
        addParams(m_block4Conv2);  // 20260330 ZJH Block4 第2层
        addParams(m_fc);           // 20260330 ZJH 线性投影层
        return vecParams;          // 20260330 ZJH 返回所有参数
    }

    // 20260330 ZJH 递归收集所有缓冲区（BN running stats）
    std::vector<Tensor*> buffers() override {
        std::vector<Tensor*> vecBufs;  // 20260330 ZJH 结果容器
        auto addBufs = [&vecBufs](Module& mod) {
            auto v = mod.buffers();  // 20260330 ZJH 获取子模块缓冲区
            vecBufs.insert(vecBufs.end(), v.begin(), v.end());  // 20260330 ZJH 合并
        };
        addBufs(m_block1Conv1);  // 20260330 ZJH Block1 BN stats
        addBufs(m_block1Conv2);  // 20260330 ZJH Block1 BN stats
        addBufs(m_block2Conv1);  // 20260330 ZJH Block2 BN stats
        addBufs(m_block2Conv2);  // 20260330 ZJH Block2 BN stats
        addBufs(m_block3Conv1);  // 20260330 ZJH Block3 BN stats
        addBufs(m_block3Conv2);  // 20260330 ZJH Block3 BN stats
        addBufs(m_block4Conv1);  // 20260330 ZJH Block4 BN stats
        addBufs(m_block4Conv2);  // 20260330 ZJH Block4 BN stats
        return vecBufs;          // 20260330 ZJH 返回所有缓冲区
    }

private:
    int m_nInChannels;           // 20260330 ZJH 输入通道数
    int m_nFeatureDim;           // 20260330 ZJH 输出特征维度

    // 20260330 ZJH Block 1: Cin → 64
    ConvBnReLU m_block1Conv1;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    ConvBnReLU m_block1Conv2;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    MaxPool2d m_pool1;           // 20260330 ZJH 2×2 最大池化

    // 20260330 ZJH Block 2: 64 → 128
    ConvBnReLU m_block2Conv1;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    ConvBnReLU m_block2Conv2;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    MaxPool2d m_pool2;           // 20260330 ZJH 2×2 最大池化

    // 20260330 ZJH Block 3: 128 → 256
    ConvBnReLU m_block3Conv1;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    ConvBnReLU m_block3Conv2;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    MaxPool2d m_pool3;           // 20260330 ZJH 2×2 最大池化

    // 20260330 ZJH Block 4: 256 → 512
    ConvBnReLU m_block4Conv1;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    ConvBnReLU m_block4Conv2;    // 20260330 ZJH 3×3 卷积 + BN + ReLU
    MaxPool2d m_pool4;           // 20260330 ZJH 2×2 最大池化

    // 20260330 ZJH 全局池化 + 投影
    AdaptiveAvgPool2d m_adaptivePool;  // 20260330 ZJH 自适应平均池化到 1×1
    Flatten m_flatten;                  // 20260330 ZJH 展平为 [N, 512]
    Linear m_fc;                        // 20260330 ZJH 线性投影 512→nFeatureDim

    // 20260330 ZJH l2Normalize — 对特征向量做 L2 归一化
    // 输入: [N, D] 特征张量
    // 返回: [N, D] 每行 L2 范数为 1 的归一化张量
    // 归一化后，内积 = 余弦相似度，避免检索时重复计算范数
    static Tensor l2Normalize(const Tensor& input) {
        const auto& vecShape = input.shapeVec();       // 20260330 ZJH 获取输入形状
        int nBatch = vecShape[0];                    // 20260330 ZJH batch 大小
        int nDim = vecShape[1];                      // 20260330 ZJH 特征维度
        Tensor output = Tensor::zeros({nBatch, nDim});  // 20260330 ZJH 分配输出张量
        const float* pIn = input.floatDataPtr();     // 20260330 ZJH 输入数据指针
        float* pOut = output.mutableFloatDataPtr();  // 20260330 ZJH 输出可写指针
        // 20260330 ZJH 逐 batch 计算 L2 范数并归一化
        for (int b = 0; b < nBatch; ++b) {
            float fNormSq = 0.0f;  // 20260330 ZJH 范数的平方和
            int nOffset = b * nDim;  // 20260330 ZJH 当前 batch 在一维数组中的偏移
            // 20260330 ZJH 计算 L2 范数平方
            for (int d = 0; d < nDim; ++d) {
                fNormSq += pIn[nOffset + d] * pIn[nOffset + d];  // 20260330 ZJH 累加平方
            }
            // 20260330 ZJH 范数值，加 eps 防止除零
            float fNorm = std::sqrt(fNormSq + 1e-12f);  // 20260330 ZJH sqrt(sum + eps)
            float fInvNorm = 1.0f / fNorm;              // 20260330 ZJH 取倒数，乘法替代除法
            // 20260330 ZJH 归一化每个元素
            for (int d = 0; d < nDim; ++d) {
                pOut[nOffset + d] = pIn[nOffset + d] * fInvNorm;  // 20260330 ZJH x / ||x||
            }
        }
        return output;  // 20260330 ZJH 返回归一化结果
    }
};

// =========================================================
// 20260330 ZJH GalleryManager — 图库管理器
// =========================================================

// 20260330 ZJH GalleryManager — 管理参考样本集 + 特征索引
// 核心职责:
//   1. 样本增删改查（按标签分类管理）
//   2. Top-K 余弦相似度检索
//   3. 图库二进制序列化（保存/加载）
//   4. 统计信息查询
// 存储: 向量线性扫描（适合工业场景 <10000 样本，无需 ANN 索引）
class GalleryManager {
public:
    // 20260330 ZJH 构造函数
    // nFeatureDim: 特征向量维度，默认 512
    GalleryManager(int nFeatureDim = 512)
        : m_nFeatureDim(nFeatureDim) {}  // 20260330 ZJH 保存特征维度

    // =========================================================
    // 样本管理
    // =========================================================

    // 20260330 ZJH addSample — 添加单个样本到图库
    // strLabel: 样本标签名
    // vecFeature: L2 归一化后的特征向量（长度必须等于 m_nFeatureDim）
    // nRoiIndex: ROI 区域索引（多 ROI 时使用，默认 0）
    // 说明: 自动维护类别列表和标签索引映射
    void addSample(const std::string& strLabel,
                   const std::vector<float>& vecFeature,
                   int nRoiIndex = 0) {
        // 20260330 ZJH 检查特征维度是否匹配
        if (static_cast<int>(vecFeature.size()) != m_nFeatureDim) {
            throw std::invalid_argument(
                "Feature dimension mismatch: expected " + std::to_string(m_nFeatureDim) +
                " but got " + std::to_string(vecFeature.size()));  // 20260330 ZJH 维度不匹配抛异常
        }
        // 20260330 ZJH 检查 ROI 索引范围
        if (nRoiIndex < 0 || nRoiIndex >= MAX_ROI_COUNT) {
            throw std::invalid_argument(
                "ROI index out of range [0, " + std::to_string(MAX_ROI_COUNT - 1) +
                "]: " + std::to_string(nRoiIndex));  // 20260330 ZJH ROI 索引越界
        }
        // 20260330 ZJH 查找或创建标签索引
        int nLabelIdx = getOrCreateLabelIndex(strLabel);  // 20260330 ZJH 获取/创建标签索引
        // 20260330 ZJH 构建样本并添加到图库
        GallerySample sample;                   // 20260330 ZJH 创建样本对象
        sample.strLabel = strLabel;              // 20260330 ZJH 设置标签名
        sample.vecFeature = vecFeature;          // 20260330 ZJH 复制特征向量
        sample.nLabelIndex = nLabelIdx;          // 20260330 ZJH 设置标签索引
        sample.nRoiIndex = nRoiIndex;            // 20260330 ZJH 设置 ROI 索引
        m_vecSamples.push_back(std::move(sample));  // 20260330 ZJH 移动添加到样本列表
    }

    // 20260330 ZJH addSampleBatch — 批量添加样本
    // strLabel: 所有样本共享的标签名
    // vecFeatures: 多个特征向量（每个长度 = m_nFeatureDim）
    // nRoiIndex: ROI 区域索引
    void addSampleBatch(const std::string& strLabel,
                        const std::vector<std::vector<float>>& vecFeatures,
                        int nRoiIndex = 0) {
        // 20260330 ZJH 逐个添加，复用 addSample 的校验逻辑
        for (const auto& vecFeat : vecFeatures) {
            addSample(strLabel, vecFeat, nRoiIndex);  // 20260330 ZJH 逐个添加
        }
    }

    // 20260330 ZJH deleteSample — 按索引删除单个样本
    // nIndex: 样本在 m_vecSamples 中的索引
    // 返回: true 删除成功, false 索引越界
    bool deleteSample(int nIndex) {
        // 20260330 ZJH 检查索引有效性
        if (nIndex < 0 || nIndex >= static_cast<int>(m_vecSamples.size())) {
            return false;  // 20260330 ZJH 索引越界，返回失败
        }
        // 20260330 ZJH 删除指定索引的样本
        m_vecSamples.erase(m_vecSamples.begin() + nIndex);  // 20260330 ZJH 从向量中移除
        rebuildLabelMap();  // 20260330 ZJH 重建标签映射（可能有类别被清空）
        return true;  // 20260330 ZJH 删除成功
    }

    // 20260330 ZJH deleteClass — 删除指定标签的所有样本
    // strLabel: 要删除的标签名
    // 返回: 被删除的样本数量
    int deleteClass(const std::string& strLabel) {
        int nDeletedCount = 0;  // 20260330 ZJH 删除计数
        // 20260330 ZJH 使用 erase-remove 惯用法删除匹配标签的样本
        auto itEnd = std::remove_if(m_vecSamples.begin(), m_vecSamples.end(),
            [&strLabel, &nDeletedCount](const GallerySample& sample) {
                if (sample.strLabel == strLabel) {
                    ++nDeletedCount;  // 20260330 ZJH 计数
                    return true;      // 20260330 ZJH 标记删除
                }
                return false;  // 20260330 ZJH 保留
            });
        m_vecSamples.erase(itEnd, m_vecSamples.end());  // 20260330 ZJH 实际删除
        // 20260330 ZJH 重建标签映射
        if (nDeletedCount > 0) {
            rebuildLabelMap();  // 20260330 ZJH 更新类别列表
        }
        return nDeletedCount;  // 20260330 ZJH 返回删除数量
    }

    // 20260330 ZJH clear — 清空图库所有样本和标签
    void clear() {
        m_vecSamples.clear();      // 20260330 ZJH 清空样本列表
        m_vecClassNames.clear();   // 20260330 ZJH 清空类别名列表
        m_mapLabelIndex.clear();   // 20260330 ZJH 清空标签索引映射
    }

    // =========================================================
    // 查询接口
    // =========================================================

    // 20260330 ZJH getSampleCount — 获取图库中的总样本数
    int getSampleCount() const {
        return static_cast<int>(m_vecSamples.size());  // 20260330 ZJH 返回样本向量大小
    }

    // 20260330 ZJH getClassCount — 获取图库中的类别数
    int getClassCount() const {
        return static_cast<int>(m_vecClassNames.size());  // 20260330 ZJH 返回类别名列表大小
    }

    // 20260330 ZJH getClassNames — 获取所有类别名称
    std::vector<std::string> getClassNames() const {
        return m_vecClassNames;  // 20260330 ZJH 返回类别名列表副本
    }

    // 20260330 ZJH getFeatureDim — 获取特征维度
    int getFeatureDim() const {
        return m_nFeatureDim;  // 20260330 ZJH 返回特征向量维度
    }

    // 20260330 ZJH getSample — 按索引获取样本引用
    // nIndex: 样本索引
    // 返回: 样本的 const 引用（越界时抛异常）
    const GallerySample& getSample(int nIndex) const {
        // 20260330 ZJH 边界检查
        if (nIndex < 0 || nIndex >= static_cast<int>(m_vecSamples.size())) {
            throw std::out_of_range(
                "Sample index out of range: " + std::to_string(nIndex));  // 20260330 ZJH 越界异常
        }
        return m_vecSamples[nIndex];  // 20260330 ZJH 返回样本引用
    }

    // 20260330 ZJH getClassSampleCount — 获取指定类别的样本数量
    // strLabel: 类别名
    // 返回: 该类别的样本数（类别不存在则返回 0）
    int getClassSampleCount(const std::string& strLabel) const {
        int nCount = 0;  // 20260330 ZJH 计数器
        // 20260330 ZJH 线性扫描计数
        for (const auto& sample : m_vecSamples) {
            if (sample.strLabel == strLabel) {
                ++nCount;  // 20260330 ZJH 匹配则计数
            }
        }
        return nCount;  // 20260330 ZJH 返回计数
    }

    // 20260330 ZJH getStats — 获取图库完整统计信息
    GalleryStats getStats() const {
        GalleryStats stats;                                     // 20260330 ZJH 统计信息对象
        stats.nTotalSamples = getSampleCount();                 // 20260330 ZJH 总样本数
        stats.nClassCount = getClassCount();                    // 20260330 ZJH 类别数
        stats.nFeatureDim = m_nFeatureDim;                      // 20260330 ZJH 特征维度
        stats.vecClassNames = m_vecClassNames;                  // 20260330 ZJH 类别名列表
        // 20260330 ZJH 统计每个类别的样本数
        stats.vecClassSampleCounts.resize(m_vecClassNames.size(), 0);  // 20260330 ZJH 初始化计数数组
        for (const auto& sample : m_vecSamples) {
            if (sample.nLabelIndex >= 0 &&
                sample.nLabelIndex < static_cast<int>(stats.vecClassSampleCounts.size())) {
                stats.vecClassSampleCounts[sample.nLabelIndex]++;  // 20260330 ZJH 对应类别计数+1
            }
        }
        return stats;  // 20260330 ZJH 返回统计信息
    }

    // =========================================================
    // Top-K 余弦相似度检索
    // =========================================================

    // 20260330 ZJH search — 使用查询特征向量在图库中检索 Top-K 最相似样本
    // vecQuery: L2 归一化后的查询特征向量
    // nTopK: 返回最相似的 K 个结果，默认 5
    // nRoiFilter: ROI 过滤（-1 表示不过滤，>=0 只匹配特定 ROI）
    // 返回: 按相似度降序排列的匹配结果列表
    // 原理: 因为特征向量已 L2 归一化，内积 = 余弦相似度
    std::vector<RetrievalMatch> search(const std::vector<float>& vecQuery,
                                        int nTopK = 5,
                                        int nRoiFilter = -1) const {
        // 20260330 ZJH 空图库直接返回空结果
        if (m_vecSamples.empty()) {
            return {};  // 20260330 ZJH 无样本可检索
        }
        // 20260330 ZJH 检查查询特征维度
        if (static_cast<int>(vecQuery.size()) != m_nFeatureDim) {
            throw std::invalid_argument(
                "Query feature dimension mismatch: expected " + std::to_string(m_nFeatureDim) +
                " but got " + std::to_string(vecQuery.size()));  // 20260330 ZJH 维度不匹配
        }
        // 20260330 ZJH 计算查询向量与所有样本的余弦相似度
        std::vector<RetrievalMatch> vecAllMatches;  // 20260330 ZJH 全部匹配结果
        vecAllMatches.reserve(m_vecSamples.size());  // 20260330 ZJH 预分配内存
        for (int i = 0; i < static_cast<int>(m_vecSamples.size()); ++i) {
            const auto& sample = m_vecSamples[i];  // 20260330 ZJH 当前样本引用
            // 20260330 ZJH ROI 过滤: 若指定了 ROI，只匹配同 ROI 的样本
            if (nRoiFilter >= 0 && sample.nRoiIndex != nRoiFilter) {
                continue;  // 20260330 ZJH 跳过不匹配的 ROI
            }
            // 20260330 ZJH 计算内积（L2 归一化后 = 余弦相似度）
            float fSim = computeDotProduct(vecQuery, sample.vecFeature);  // 20260330 ZJH 内积
            // 20260330 ZJH 构建匹配结果
            RetrievalMatch match;                // 20260330 ZJH 匹配结果对象
            match.nSampleIndex = i;               // 20260330 ZJH 样本索引
            match.fSimilarity = fSim;             // 20260330 ZJH 相似度
            match.strLabel = sample.strLabel;     // 20260330 ZJH 标签名
            match.nLabelIndex = sample.nLabelIndex;  // 20260330 ZJH 标签索引
            vecAllMatches.push_back(match);       // 20260330 ZJH 添加到结果列表
        }
        // 20260330 ZJH 按相似度降序排序
        std::sort(vecAllMatches.begin(), vecAllMatches.end(),
            [](const RetrievalMatch& a, const RetrievalMatch& b) {
                return a.fSimilarity > b.fSimilarity;  // 20260330 ZJH 降序排列
            });
        // 20260330 ZJH 截取 Top-K
        int nResultCount = std::min(nTopK, static_cast<int>(vecAllMatches.size()));  // 20260330 ZJH 实际返回数量
        vecAllMatches.resize(nResultCount);  // 20260330 ZJH 截断到 Top-K
        return vecAllMatches;  // 20260330 ZJH 返回 Top-K 匹配结果
    }

    // 20260330 ZJH searchByClass — 检索后按类别投票，返回最可能的类别
    // vecQuery: L2 归一化后的查询特征向量
    // nTopK: 参与投票的 Top-K 样本数
    // 返回: {类别名, 置信度} 对，置信度为 Top-K 中该类别的平均相似度
    std::pair<std::string, float> searchByClass(const std::vector<float>& vecQuery,
                                                 int nTopK = 5) const {
        // 20260330 ZJH 执行 Top-K 检索
        auto vecMatches = search(vecQuery, nTopK);  // 20260330 ZJH 获取 Top-K 匹配
        if (vecMatches.empty()) {
            return {"", 0.0f};  // 20260330 ZJH 无匹配返回空
        }
        // 20260330 ZJH 按类别统计相似度总和和出现次数
        std::unordered_map<std::string, float> mapClassSimSum;    // 20260330 ZJH 类别相似度累计
        std::unordered_map<std::string, int> mapClassCount;       // 20260330 ZJH 类别出现次数
        for (const auto& match : vecMatches) {
            mapClassSimSum[match.strLabel] += match.fSimilarity;  // 20260330 ZJH 累加相似度
            mapClassCount[match.strLabel]++;                       // 20260330 ZJH 计数
        }
        // 20260330 ZJH 找到平均相似度最高的类别
        std::string strBestLabel;   // 20260330 ZJH 最佳类别名
        float fBestScore = -1.0f;   // 20260330 ZJH 最佳平均相似度
        for (const auto& [strLabel, fSimSum] : mapClassSimSum) {
            float fAvgSim = fSimSum / static_cast<float>(mapClassCount[strLabel]);  // 20260330 ZJH 平均相似度
            if (fAvgSim > fBestScore) {
                fBestScore = fAvgSim;    // 20260330 ZJH 更新最佳
                strBestLabel = strLabel;  // 20260330 ZJH 更新最佳类别
            }
        }
        return {strBestLabel, fBestScore};  // 20260330 ZJH 返回最佳类别和置信度
    }

    // =========================================================
    // 图库序列化（二进制格式）
    // =========================================================

    // 20260330 ZJH saveGallery — 将图库保存到二进制文件
    // strPath: 输出文件路径（推荐扩展名 .omgr）
    // 返回: true 保存成功, false 保存失败
    // 格式: [Header] [ClassTable] [SampleTable]
    bool saveGallery(const std::string& strPath) const {
        // 20260330 ZJH 打开输出文件（二进制模式）
        std::ofstream ofs(strPath, std::ios::binary);  // 20260330 ZJH 二进制写模式
        if (!ofs.is_open()) {
            return false;  // 20260330 ZJH 文件打开失败
        }
        // 20260330 ZJH 写入文件头
        ofs.write(GALLERY_MAGIC, 4);  // 20260330 ZJH 魔数 "OMGR"
        writeUint32(ofs, GALLERY_VERSION);  // 20260330 ZJH 版本号
        writeUint32(ofs, static_cast<uint32_t>(m_nFeatureDim));  // 20260330 ZJH 特征维度
        writeUint32(ofs, static_cast<uint32_t>(m_vecSamples.size()));  // 20260330 ZJH 样本数
        writeUint32(ofs, static_cast<uint32_t>(m_vecClassNames.size()));  // 20260330 ZJH 类别数

        // 20260330 ZJH 写入类别表: 每个类别 = [名称长度 uint32] + [名称字节]
        for (const auto& strClassName : m_vecClassNames) {
            writeUint32(ofs, static_cast<uint32_t>(strClassName.size()));  // 20260330 ZJH 名称长度
            ofs.write(strClassName.data(), strClassName.size());            // 20260330 ZJH 名称内容
        }

        // 20260330 ZJH 写入样本表: 每个样本 = [标签索引 int32] + [ROI索引 int32] + [特征 float×dim]
        for (const auto& sample : m_vecSamples) {
            writeInt32(ofs, sample.nLabelIndex);  // 20260330 ZJH 标签索引
            writeInt32(ofs, sample.nRoiIndex);    // 20260330 ZJH ROI 索引
            // 20260330 ZJH 写入特征向量（连续 float 数组）
            ofs.write(reinterpret_cast<const char*>(sample.vecFeature.data()),
                      sample.vecFeature.size() * sizeof(float));  // 20260330 ZJH 特征数据
        }

        ofs.flush();  // 20260330 ZJH 刷新缓冲
        return ofs.good();  // 20260330 ZJH 返回写入是否成功
    }

    // 20260330 ZJH loadGallery — 从二进制文件加载图库
    // strPath: 输入文件路径
    // 返回: true 加载成功, false 加载失败（格式错误/文件不存在）
    // 说明: 加载前会清空当前图库
    bool loadGallery(const std::string& strPath) {
        // 20260330 ZJH 打开输入文件（二进制模式）
        std::ifstream ifs(strPath, std::ios::binary);  // 20260330 ZJH 二进制读模式
        if (!ifs.is_open()) {
            return false;  // 20260330 ZJH 文件打开失败
        }
        // 20260330 ZJH 读取并校验魔数
        char arrMagic[4] = {};  // 20260330 ZJH 魔数缓冲区
        ifs.read(arrMagic, 4);  // 20260330 ZJH 读取 4 字节
        if (std::memcmp(arrMagic, GALLERY_MAGIC, 4) != 0) {
            return false;  // 20260330 ZJH 魔数不匹配，非法文件
        }
        // 20260330 ZJH 读取版本号
        uint32_t nVersion = readUint32(ifs);  // 20260330 ZJH 版本号
        if (nVersion > GALLERY_VERSION) {
            return false;  // 20260330 ZJH 版本过高，不兼容
        }
        // 20260330 ZJH 读取基本信息
        uint32_t nFeatureDim = readUint32(ifs);    // 20260330 ZJH 特征维度
        uint32_t nSampleCount = readUint32(ifs);   // 20260330 ZJH 样本数
        uint32_t nClassCount = readUint32(ifs);     // 20260330 ZJH 类别数

        // 20260330 ZJH 清空当前图库并设置特征维度
        clear();  // 20260330 ZJH 清空现有数据
        m_nFeatureDim = static_cast<int>(nFeatureDim);  // 20260330 ZJH 更新特征维度

        // 20260330 ZJH 读取类别表
        m_vecClassNames.resize(nClassCount);  // 20260330 ZJH 预分配类别名列表
        for (uint32_t i = 0; i < nClassCount; ++i) {
            uint32_t nNameLen = readUint32(ifs);           // 20260330 ZJH 名称长度
            m_vecClassNames[i].resize(nNameLen);           // 20260330 ZJH 分配字符串空间
            ifs.read(m_vecClassNames[i].data(), nNameLen);  // 20260330 ZJH 读取名称内容
            m_mapLabelIndex[m_vecClassNames[i]] = static_cast<int>(i);  // 20260330 ZJH 建立映射
        }

        // 20260330 ZJH 读取样本表
        m_vecSamples.resize(nSampleCount);  // 20260330 ZJH 预分配样本列表
        for (uint32_t i = 0; i < nSampleCount; ++i) {
            m_vecSamples[i].nLabelIndex = readInt32(ifs);  // 20260330 ZJH 标签索引
            m_vecSamples[i].nRoiIndex = readInt32(ifs);    // 20260330 ZJH ROI 索引
            // 20260330 ZJH 恢复标签名
            if (m_vecSamples[i].nLabelIndex >= 0 &&
                m_vecSamples[i].nLabelIndex < static_cast<int>(m_vecClassNames.size())) {
                m_vecSamples[i].strLabel = m_vecClassNames[m_vecSamples[i].nLabelIndex];  // 20260330 ZJH 根据索引恢复标签名
            }
            // 20260330 ZJH 读取特征向量
            m_vecSamples[i].vecFeature.resize(nFeatureDim);  // 20260330 ZJH 分配特征空间
            ifs.read(reinterpret_cast<char*>(m_vecSamples[i].vecFeature.data()),
                     nFeatureDim * sizeof(float));  // 20260330 ZJH 读取特征数据
        }

        return ifs.good();  // 20260330 ZJH 返回读取是否成功
    }

private:
    int m_nFeatureDim;  // 20260330 ZJH 特征向量维度

    std::vector<GallerySample> m_vecSamples;                    // 20260330 ZJH 样本列表（线性存储）
    std::vector<std::string> m_vecClassNames;                   // 20260330 ZJH 类别名有序列表
    std::unordered_map<std::string, int> m_mapLabelIndex;       // 20260330 ZJH 标签名→索引映射

    // 20260330 ZJH getOrCreateLabelIndex — 获取或创建标签索引
    // 若标签已存在则返回已有索引，否则创建新索引
    int getOrCreateLabelIndex(const std::string& strLabel) {
        auto it = m_mapLabelIndex.find(strLabel);  // 20260330 ZJH 查找标签
        if (it != m_mapLabelIndex.end()) {
            return it->second;  // 20260330 ZJH 已存在，返回索引
        }
        // 20260330 ZJH 新标签：分配下一个索引
        int nNewIndex = static_cast<int>(m_vecClassNames.size());  // 20260330 ZJH 新索引
        m_vecClassNames.push_back(strLabel);      // 20260330 ZJH 添加到类别名列表
        m_mapLabelIndex[strLabel] = nNewIndex;    // 20260330 ZJH 更新映射
        return nNewIndex;  // 20260330 ZJH 返回新索引
    }

    // 20260330 ZJH rebuildLabelMap — 重建标签映射
    // 删除样本后可能导致某些类别消失，需要重建映射
    void rebuildLabelMap() {
        m_vecClassNames.clear();   // 20260330 ZJH 清空类别名列表
        m_mapLabelIndex.clear();   // 20260330 ZJH 清空映射
        // 20260330 ZJH 扫描所有样本，收集唯一标签
        for (auto& sample : m_vecSamples) {
            auto it = m_mapLabelIndex.find(sample.strLabel);  // 20260330 ZJH 查找标签
            if (it == m_mapLabelIndex.end()) {
                // 20260330 ZJH 新标签，分配索引
                int nIdx = static_cast<int>(m_vecClassNames.size());  // 20260330 ZJH 新索引
                m_vecClassNames.push_back(sample.strLabel);  // 20260330 ZJH 添加到列表
                m_mapLabelIndex[sample.strLabel] = nIdx;     // 20260330 ZJH 更新映射
                sample.nLabelIndex = nIdx;                   // 20260330 ZJH 更新样本索引
            } else {
                sample.nLabelIndex = it->second;  // 20260330 ZJH 使用已有索引
            }
        }
    }

    // 20260330 ZJH computeDotProduct — 计算两个向量的内积
    // 对于 L2 归一化向量，内积 = 余弦相似度
    static float computeDotProduct(const std::vector<float>& vecA,
                                    const std::vector<float>& vecB) {
        float fDot = 0.0f;  // 20260330 ZJH 内积累加器
        int nDim = static_cast<int>(vecA.size());  // 20260330 ZJH 向量维度
        // 20260330 ZJH 4 路展开减少循环开销（简单 SIMD 友好优化）
        int i = 0;  // 20260330 ZJH 循环索引
        int nDim4 = nDim - (nDim % 4);  // 20260330 ZJH 4 的倍数截断点
        for (; i < nDim4; i += 4) {
            fDot += vecA[i] * vecB[i]             // 20260330 ZJH 第 1 路
                  + vecA[i + 1] * vecB[i + 1]     // 20260330 ZJH 第 2 路
                  + vecA[i + 2] * vecB[i + 2]     // 20260330 ZJH 第 3 路
                  + vecA[i + 3] * vecB[i + 3];    // 20260330 ZJH 第 4 路
        }
        // 20260330 ZJH 处理尾部剩余元素
        for (; i < nDim; ++i) {
            fDot += vecA[i] * vecB[i];  // 20260330 ZJH 逐元素累加
        }
        return fDot;  // 20260330 ZJH 返回内积
    }

    // =========================================================
    // 20260330 ZJH 二进制读写辅助函数
    // =========================================================

    // 20260330 ZJH writeUint32 — 写入 uint32 到输出流（小端序）
    static void writeUint32(std::ofstream& ofs, uint32_t nVal) {
        ofs.write(reinterpret_cast<const char*>(&nVal), sizeof(uint32_t));  // 20260330 ZJH 直接写 4 字节
    }

    // 20260330 ZJH writeInt32 — 写入 int32 到输出流
    static void writeInt32(std::ofstream& ofs, int32_t nVal) {
        ofs.write(reinterpret_cast<const char*>(&nVal), sizeof(int32_t));  // 20260330 ZJH 直接写 4 字节
    }

    // 20260330 ZJH readUint32 — 从输入流读取 uint32
    static uint32_t readUint32(std::ifstream& ifs) {
        uint32_t nVal = 0;  // 20260330 ZJH 初始化
        ifs.read(reinterpret_cast<char*>(&nVal), sizeof(uint32_t));  // 20260330 ZJH 读取 4 字节
        return nVal;  // 20260330 ZJH 返回读取值
    }

    // 20260330 ZJH readInt32 — 从输入流读取 int32
    static int32_t readInt32(std::ifstream& ifs) {
        int32_t nVal = 0;  // 20260330 ZJH 初始化
        ifs.read(reinterpret_cast<char*>(&nVal), sizeof(int32_t));  // 20260330 ZJH 读取 4 字节
        return nVal;  // 20260330 ZJH 返回读取值
    }
};

// =========================================================
// 20260330 ZJH TripletLoss — 三元组损失函数
// =========================================================

// 20260330 ZJH TripletLoss — 特征学习的三元组损失
// 公式: L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
// 其中 d(a, b) = 1 - cosine_similarity(a, b) = 1 - dot(a, b) （L2 归一化向量）
// 目标: 拉近同类样本距离，推远异类样本距离，间隔至少 margin
// 用途: 训练 FeatureExtractor 学习可区分的特征表示
class TripletLoss {
public:
    // 20260330 ZJH 构造函数
    // fMargin: 三元组间隔阈值，默认 0.3
    // 间隔越大，要求正负样本之间的区分度越高
    TripletLoss(float fMargin = 0.3f)
        : m_fMargin(fMargin) {}  // 20260330 ZJH 保存间隔参数

    // 20260330 ZJH forward — 计算三元组损失
    // anchor: [N, D] 锚点特征（已 L2 归一化）
    // positive: [N, D] 正样本特征（与锚点同类）
    // negative: [N, D] 负样本特征（与锚点异类）
    // 返回: 标量损失张量（可反向传播）
    // 计算流程:
    //   1. dPos = 1 - dot(anchor, positive) — 锚点到正样本的距离
    //   2. dNeg = 1 - dot(anchor, negative) — 锚点到负样本的距离
    //   3. loss = mean(max(0, dPos - dNeg + margin))
    Tensor forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative) {
        const auto& vecShape = anchor.shapeVec();  // 20260330 ZJH 获取锚点形状
        int nBatch = vecShape[0];                // 20260330 ZJH batch 大小
        int nDim = vecShape[1];                  // 20260330 ZJH 特征维度

        // 20260330 ZJH 计算锚点-正样本相似度: [N] = sum(anchor * positive, dim=1)
        Tensor dotPos = tensorMul(anchor, positive);       // 20260330 ZJH 逐元素乘: [N, D]
        // 20260330 ZJH 计算锚点-负样本相似度: [N] = sum(anchor * negative, dim=1)
        Tensor dotNeg = tensorMul(anchor, negative);       // 20260330 ZJH 逐元素乘: [N, D]

        // 20260330 ZJH 需要沿 dim=1 求和得到内积
        // 手动计算逐 batch 的内积和最终损失
        const float* pAnchor = anchor.floatDataPtr();      // 20260330 ZJH 锚点数据
        const float* pPositive = positive.floatDataPtr();  // 20260330 ZJH 正样本数据
        const float* pNegative = negative.floatDataPtr();  // 20260330 ZJH 负样本数据

        // 20260330 ZJH 计算损失: 逐 batch 计算 max(0, dPos - dNeg + margin)
        float fTotalLoss = 0.0f;  // 20260330 ZJH 总损失累加器
        for (int b = 0; b < nBatch; ++b) {
            float fDotPos = 0.0f;  // 20260330 ZJH 正样本内积
            float fDotNeg = 0.0f;  // 20260330 ZJH 负样本内积
            int nOffset = b * nDim;  // 20260330 ZJH 当前 batch 偏移
            // 20260330 ZJH 计算内积
            for (int d = 0; d < nDim; ++d) {
                fDotPos += pAnchor[nOffset + d] * pPositive[nOffset + d];  // 20260330 ZJH anchor·positive
                fDotNeg += pAnchor[nOffset + d] * pNegative[nOffset + d];  // 20260330 ZJH anchor·negative
            }
            // 20260330 ZJH 余弦距离 = 1 - 余弦相似度
            float fDistPos = 1.0f - fDotPos;  // 20260330 ZJH 正样本距离
            float fDistNeg = 1.0f - fDotNeg;  // 20260330 ZJH 负样本距离
            // 20260330 ZJH 三元组损失: max(0, dPos - dNeg + margin)
            float fLoss = std::max(0.0f, fDistPos - fDistNeg + m_fMargin);  // 20260330 ZJH hinge loss
            fTotalLoss += fLoss;  // 20260330 ZJH 累加
        }
        // 20260330 ZJH 求平均损失
        float fMeanLoss = fTotalLoss / static_cast<float>(nBatch);  // 20260330 ZJH 均值

        // 20260330 ZJH 构建计算图用于反向传播
        // 使用张量运算构建可微分的损失
        // dPos[b] = 1 - sum(anchor[b] * positive[b])
        // dNeg[b] = 1 - sum(anchor[b] * negative[b])
        // loss = mean(max(0, dPos - dNeg + margin))

        // 20260330 ZJH 构造 dPos 和 dNeg 张量用于自动微分
        Tensor tDotPos = batchDot(anchor, positive);  // 20260330 ZJH [N, 1] 内积
        Tensor tDotNeg = batchDot(anchor, negative);  // 20260330 ZJH [N, 1] 内积

        // 20260330 ZJH dPos = 1 - dotPos, dNeg = 1 - dotNeg
        Tensor tOnes = Tensor::ones({nBatch, 1});  // 20260330 ZJH [N, 1] 全 1
        Tensor tDistPos = tensorSub(tOnes, tDotPos);  // 20260330 ZJH 正样本距离
        Tensor tDistNeg = tensorSub(tOnes, tDotNeg);  // 20260330 ZJH 负样本距离

        // 20260330 ZJH diff = dPos - dNeg + margin
        Tensor tMargin = Tensor::ones({nBatch, 1});  // 20260330 ZJH margin 张量
        float* pMargin = tMargin.mutableFloatDataPtr();  // 20260330 ZJH 可写指针
        for (int i = 0; i < nBatch; ++i) {
            pMargin[i] = m_fMargin;  // 20260330 ZJH 填充 margin 值
        }
        Tensor tDiff = tensorAdd(tensorSub(tDistPos, tDistNeg), tMargin);  // 20260330 ZJH dPos - dNeg + margin

        // 20260330 ZJH max(0, diff) 通过 ReLU 实现 hinge
        Tensor tHinge = tensorReLU(tDiff);  // 20260330 ZJH hinge loss [N, 1]

        // 20260330 ZJH mean reduction
        Tensor tLoss = tensorSum(tHinge);  // 20260330 ZJH sum([N, 1]) → 标量
        Tensor tScale = Tensor::ones({1});  // 20260330 ZJH 缩放因子
        float* pScale = tScale.mutableFloatDataPtr();  // 20260330 ZJH 可写指针
        pScale[0] = 1.0f / static_cast<float>(nBatch);  // 20260330 ZJH 1/N
        tLoss = tensorMul(tLoss, tScale);  // 20260330 ZJH 均值损失

        return tLoss;  // 20260330 ZJH 返回损失张量（支持 backward）
    }

    // 20260330 ZJH 获取间隔参数
    float margin() const { return m_fMargin; }

    // 20260330 ZJH 设置间隔参数
    void setMargin(float fMargin) { m_fMargin = fMargin; }

private:
    float m_fMargin;  // 20260330 ZJH 三元组间隔阈值

    // 20260330 ZJH batchDot — 批量内积: [N, D] × [N, D] → [N, 1]
    // 计算每个 batch 对应特征向量的内积
    static Tensor batchDot(const Tensor& a, const Tensor& b) {
        const auto& vecShape = a.shapeVec();    // 20260330 ZJH 获取形状
        int nBatch = vecShape[0];             // 20260330 ZJH batch 大小
        int nDim = vecShape[1];               // 20260330 ZJH 特征维度
        Tensor result = Tensor::zeros({nBatch, 1});  // 20260330 ZJH 输出 [N, 1]
        const float* pA = a.floatDataPtr();   // 20260330 ZJH a 数据
        const float* pB = b.floatDataPtr();   // 20260330 ZJH b 数据
        float* pR = result.mutableFloatDataPtr();  // 20260330 ZJH 输出可写指针
        // 20260330 ZJH 逐 batch 计算内积
        for (int n = 0; n < nBatch; ++n) {
            float fDot = 0.0f;  // 20260330 ZJH 内积累加器
            int nOffset = n * nDim;  // 20260330 ZJH 偏移
            for (int d = 0; d < nDim; ++d) {
                fDot += pA[nOffset + d] * pB[nOffset + d];  // 20260330 ZJH 逐元素乘累加
            }
            pR[n] = fDot;  // 20260330 ZJH 存储内积结果
        }
        return result;  // 20260330 ZJH 返回 [N, 1]
    }
};

// =========================================================
// 20260330 ZJH ImageRetrievalTool — 完整检索流水线
// =========================================================

// 20260330 ZJH ImageRetrievalTool — 端到端图像检索工具
// 封装 FeatureExtractor + GalleryManager，提供一站式图像检索 API
// 典型工作流:
//   1. 训练阶段: 使用 TripletLoss 训练 FeatureExtractor
//   2. 注册阶段: 对参考图像提取特征，注册到 GalleryManager
//   3. 检索阶段: 对查询图像提取特征，在 GalleryManager 中 Top-K 检索
//   4. 持久化: saveGallery/loadGallery 保存/加载图库
// 对标: HikRobot CNNRetrievalCpp（图库管理 + CNN 特征 + Top-K 检索 + 多 ROI）
class ImageRetrievalTool {
public:
    // 20260330 ZJH 构造函数
    // nFeatureDim: 特征向量维度，默认 512
    // nInChannels: 输入图像通道数（1=灰度, 3=RGB），默认 3
    ImageRetrievalTool(int nFeatureDim = 512, int nInChannels = 3)
        : m_encoder(nInChannels, nFeatureDim),  // 20260330 ZJH 初始化特征提取网络
          m_gallery(nFeatureDim),                // 20260330 ZJH 初始化图库管理器
          m_nFeatureDim(nFeatureDim),            // 20260330 ZJH 保存特征维度
          m_nInChannels(nInChannels)              // 20260330 ZJH 保存输入通道数
    {}

    // =========================================================
    // 特征提取
    // =========================================================

    // 20260330 ZJH extractFeature — 提取单张图像的特征向量
    // image: [1, C, H, W] 单张图像张量（已预处理，归一化到 [0,1] 或 [-1,1]）
    // 返回: 长度为 nFeatureDim 的 L2 归一化特征向量
    std::vector<float> extractFeature(const Tensor& image) {
        // 20260330 ZJH 设置评估模式（BN 使用 running stats，Dropout 透传）
        m_encoder.eval();  // 20260330 ZJH 切换到推理模式
        // 20260330 ZJH 前向传播提取特征
        Tensor tFeat = m_encoder.forward(image);  // 20260330 ZJH [1, D] 归一化特征
        // 20260330 ZJH 将 Tensor 转换为 std::vector<float>
        const float* pData = tFeat.floatDataPtr();  // 20260330 ZJH 特征数据指针
        int nDim = tFeat.numel();  // 20260330 ZJH 特征元素数
        std::vector<float> vecFeature(pData, pData + nDim);  // 20260330 ZJH 复制到 vector
        return vecFeature;  // 20260330 ZJH 返回特征向量
    }

    // 20260330 ZJH extractFeatureBatch — 批量提取特征
    // images: [N, C, H, W] 多张图像张量
    // 返回: N 个特征向量
    std::vector<std::vector<float>> extractFeatureBatch(const Tensor& images) {
        // 20260330 ZJH 设置评估模式
        m_encoder.eval();  // 20260330 ZJH 推理模式
        // 20260330 ZJH 前向传播提取特征
        Tensor tFeats = m_encoder.forward(images);  // 20260330 ZJH [N, D]
        const auto& vecShape = tFeats.shapeVec();       // 20260330 ZJH 获取形状
        int nBatch = vecShape[0];                     // 20260330 ZJH batch 大小
        int nDim = vecShape[1];                       // 20260330 ZJH 特征维度
        const float* pData = tFeats.floatDataPtr();   // 20260330 ZJH 数据指针
        // 20260330 ZJH 逐 batch 转换为 vector
        std::vector<std::vector<float>> vecResult(nBatch);  // 20260330 ZJH 结果容器
        for (int b = 0; b < nBatch; ++b) {
            vecResult[b].assign(pData + b * nDim, pData + (b + 1) * nDim);  // 20260330 ZJH 复制特征
        }
        return vecResult;  // 20260330 ZJH 返回所有特征向量
    }

    // 20260330 ZJH extractFeatureFromROI — 提取 ROI 区域的特征
    // image: [1, C, H, W] 输入图像
    // roi: ROI 区域定义（归一化坐标）
    // 返回: ROI 区域的 L2 归一化特征向量
    std::vector<float> extractFeatureFromROI(const Tensor& image, const RetrievalROI& roi) {
        // 20260330 ZJH 从输入图像中裁剪 ROI 区域
        Tensor tRoiCrop = cropROI(image, roi);  // 20260330 ZJH 裁剪 ROI
        // 20260330 ZJH 提取裁剪区域的特征
        return extractFeature(tRoiCrop);  // 20260330 ZJH 返回 ROI 特征
    }

    // =========================================================
    // 样本注册
    // =========================================================

    // 20260330 ZJH registerSample — 注册单张图像到图库
    // image: [1, C, H, W] 图像张量
    // strLabel: 样本标签名
    // nRoiIndex: ROI 索引（默认 0）
    void registerSample(const Tensor& image, const std::string& strLabel, int nRoiIndex = 0) {
        // 20260330 ZJH 提取特征
        std::vector<float> vecFeat = extractFeature(image);  // 20260330 ZJH 特征向量
        // 20260330 ZJH 添加到图库
        m_gallery.addSample(strLabel, vecFeat, nRoiIndex);  // 20260330 ZJH 注册到图库
    }

    // 20260330 ZJH registerSampleWithROI — 从图像 ROI 区域注册样本
    // image: [1, C, H, W] 图像张量
    // strLabel: 样本标签名
    // roi: ROI 区域定义
    void registerSampleWithROI(const Tensor& image, const std::string& strLabel,
                                const RetrievalROI& roi) {
        // 20260330 ZJH 提取 ROI 特征
        std::vector<float> vecFeat = extractFeatureFromROI(image, roi);  // 20260330 ZJH ROI 特征
        // 20260330 ZJH 添加到图库
        m_gallery.addSample(strLabel, vecFeat, roi.nIndex);  // 20260330 ZJH 注册到图库
    }

    // 20260330 ZJH registerSampleBatch — 批量注册图像
    // images: [N, C, H, W] 多张图像
    // strLabel: 所有图像共享的标签名
    void registerSampleBatch(const Tensor& images, const std::string& strLabel) {
        // 20260330 ZJH 批量提取特征
        auto vecFeats = extractFeatureBatch(images);  // 20260330 ZJH N 个特征向量
        // 20260330 ZJH 逐个添加到图库
        for (const auto& vecFeat : vecFeats) {
            m_gallery.addSample(strLabel, vecFeat);  // 20260330 ZJH 注册
        }
    }

    // =========================================================
    // 图像检索
    // =========================================================

    // 20260330 ZJH retrieve — 检索: 输入图像 → Top-K 匹配结果
    // image: [1, C, H, W] 查询图像张量
    // nTopK: 返回最相似的 K 个结果
    // 返回: 按相似度降序排列的匹配结果列表
    std::vector<RetrievalMatch> retrieve(const Tensor& image, int nTopK = 5) {
        // 20260330 ZJH 提取查询图像特征
        std::vector<float> vecQuery = extractFeature(image);  // 20260330 ZJH 查询特征
        // 20260330 ZJH 在图库中检索
        return m_gallery.search(vecQuery, nTopK);  // 20260330 ZJH 返回 Top-K 结果
    }

    // 20260330 ZJH retrieveWithROI — 对 ROI 区域进行检索
    // image: [1, C, H, W] 查询图像
    // roi: ROI 区域定义
    // nTopK: 返回数量
    // bFilterByROI: 是否只匹配同 ROI 索引的样本
    std::vector<RetrievalMatch> retrieveWithROI(const Tensor& image,
                                                 const RetrievalROI& roi,
                                                 int nTopK = 5,
                                                 bool bFilterByROI = true) {
        // 20260330 ZJH 提取 ROI 区域特征
        std::vector<float> vecQuery = extractFeatureFromROI(image, roi);  // 20260330 ZJH ROI 查询特征
        // 20260330 ZJH ROI 过滤参数
        int nRoiFilter = bFilterByROI ? roi.nIndex : -1;  // 20260330 ZJH -1 表示不过滤
        // 20260330 ZJH 在图库中检索
        return m_gallery.search(vecQuery, nTopK, nRoiFilter);  // 20260330 ZJH 返回结果
    }

    // 20260330 ZJH retrieveMultiROI — 多 ROI 区域检索（对标 HikRobot 256 区域能力）
    // image: [1, C, H, W] 查询图像
    // vecROIs: 多个 ROI 区域定义（最多 256 个）
    // nTopK: 每个 ROI 返回的匹配数量
    // 返回: 每个 ROI 的 Top-K 匹配结果列表
    std::vector<std::vector<RetrievalMatch>> retrieveMultiROI(
            const Tensor& image,
            const std::vector<RetrievalROI>& vecROIs,
            int nTopK = 5) {
        // 20260330 ZJH 检查 ROI 数量限制
        if (static_cast<int>(vecROIs.size()) > MAX_ROI_COUNT) {
            throw std::invalid_argument(
                "ROI count exceeds limit " + std::to_string(MAX_ROI_COUNT) +
                ": " + std::to_string(vecROIs.size()));  // 20260330 ZJH ROI 数量越界
        }
        // 20260330 ZJH 逐 ROI 检索
        std::vector<std::vector<RetrievalMatch>> vecResults;  // 20260330 ZJH 结果容器
        vecResults.reserve(vecROIs.size());  // 20260330 ZJH 预分配
        for (const auto& roi : vecROIs) {
            vecResults.push_back(retrieveWithROI(image, roi, nTopK, true));  // 20260330 ZJH 检索单个 ROI
        }
        return vecResults;  // 20260330 ZJH 返回所有 ROI 的结果
    }

    // 20260330 ZJH classify — 基于检索的分类: 输入图像 → 类别 + 置信度
    // image: [1, C, H, W] 查询图像
    // nTopK: 参与投票的 Top-K 样本数
    // 返回: {类别名, 置信度}
    std::pair<std::string, float> classify(const Tensor& image, int nTopK = 5) {
        // 20260330 ZJH 提取查询特征
        std::vector<float> vecQuery = extractFeature(image);  // 20260330 ZJH 查询特征
        // 20260330 ZJH 按类别投票
        return m_gallery.searchByClass(vecQuery, nTopK);  // 20260330 ZJH 返回分类结果
    }

    // =========================================================
    // 图库和模型访问
    // =========================================================

    // 20260330 ZJH gallery — 获取图库管理器引用
    GalleryManager& gallery() { return m_gallery; }

    // 20260330 ZJH gallery — const 版本
    const GalleryManager& gallery() const { return m_gallery; }

    // 20260330 ZJH encoder — 获取特征提取网络引用
    FeatureExtractor& encoder() { return m_encoder; }

    // 20260330 ZJH encoder — const 版本
    const FeatureExtractor& encoder() const { return m_encoder; }

    // 20260330 ZJH featureDim — 获取特征维度
    int featureDim() const { return m_nFeatureDim; }

    // 20260330 ZJH inChannels — 获取输入通道数
    int inChannels() const { return m_nInChannels; }

    // =========================================================
    // 图库持久化（代理到 GalleryManager）
    // =========================================================

    // 20260330 ZJH saveGallery — 保存图库到文件
    bool saveGallery(const std::string& strPath) const {
        return m_gallery.saveGallery(strPath);  // 20260330 ZJH 代理调用
    }

    // 20260330 ZJH loadGallery — 从文件加载图库
    bool loadGallery(const std::string& strPath) {
        return m_gallery.loadGallery(strPath);  // 20260330 ZJH 代理调用
    }

private:
    FeatureExtractor m_encoder;  // 20260330 ZJH CNN 特征提取骨干网络
    GalleryManager m_gallery;    // 20260330 ZJH 图库管理器
    int m_nFeatureDim;           // 20260330 ZJH 特征向量维度
    int m_nInChannels;           // 20260330 ZJH 输入图像通道数

    // 20260330 ZJH cropROI — 从图像中裁剪 ROI 区域
    // image: [1, C, H, W] 输入图像
    // roi: ROI 区域定义（归一化坐标 [0,1]）
    // 返回: [1, C, roiH, roiW] 裁剪后的图像区域
    // 注意: 简单实现使用最近邻采样，工业场景足够
    static Tensor cropROI(const Tensor& image, const RetrievalROI& roi) {
        const auto& vecShape = image.shapeVec();  // 20260330 ZJH 获取输入形状
        int nC = vecShape[1];                   // 20260330 ZJH 通道数
        int nH = vecShape[2];                   // 20260330 ZJH 图像高度
        int nW = vecShape[3];                   // 20260330 ZJH 图像宽度

        // 20260330 ZJH 将归一化坐标转换为像素坐标
        int nX0 = std::max(0, static_cast<int>(roi.fX * nW));       // 20260330 ZJH 左边界（裁剪到合法范围）
        int nY0 = std::max(0, static_cast<int>(roi.fY * nH));       // 20260330 ZJH 上边界
        int nRoiW = std::max(1, static_cast<int>(roi.fWidth * nW));  // 20260330 ZJH ROI 宽度（至少 1）
        int nRoiH = std::max(1, static_cast<int>(roi.fHeight * nH)); // 20260330 ZJH ROI 高度
        // 20260330 ZJH 裁剪到图像边界内
        nRoiW = std::min(nRoiW, nW - nX0);  // 20260330 ZJH 不超过右边界
        nRoiH = std::min(nRoiH, nH - nY0);  // 20260330 ZJH 不超过下边界

        // 20260330 ZJH 分配输出张量
        Tensor output = Tensor::zeros({1, nC, nRoiH, nRoiW});  // 20260330 ZJH [1, C, roiH, roiW]
        const float* pIn = image.floatDataPtr();                 // 20260330 ZJH 输入数据指针
        float* pOut = output.mutableFloatDataPtr();              // 20260330 ZJH 输出可写指针

        // 20260330 ZJH 逐通道逐行复制 ROI 区域像素
        for (int c = 0; c < nC; ++c) {
            for (int y = 0; y < nRoiH; ++y) {
                for (int x = 0; x < nRoiW; ++x) {
                    // 20260330 ZJH 源像素索引: [0, c, y0+y, x0+x]
                    int nSrcIdx = c * nH * nW + (nY0 + y) * nW + (nX0 + x);
                    // 20260330 ZJH 目标像素索引: [0, c, y, x]
                    int nDstIdx = c * nRoiH * nRoiW + y * nRoiW + x;
                    pOut[nDstIdx] = pIn[nSrcIdx];  // 20260330 ZJH 复制像素值
                }
            }
        }
        return output;  // 20260330 ZJH 返回裁剪后的 ROI 图像
    }
};

}  // namespace om
