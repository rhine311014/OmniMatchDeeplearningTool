// 20260322 ZJH PatchCore 异常检测模块
// 实现 PatchCore 异常检测算法（核心思想: 正常样本特征记忆库 + 最近邻距离）
// 训练阶段: 仅使用 OK（正常）图像，提取 patch 级特征并存入 MemoryBank
// 推理阶段: 提取测试图像 patch 特征 → 与 MemoryBank 中最近邻的距离 → 异常分数
// 优势: 无需训练网络权重，仅需正常样本建库；少样本场景下表现优异
// PatchCore 不继承 Module（非前向网络），使用独立的特征提取器
// 20260402 ZJH Phase 1.2: 支持 ImageNet 预训练 ResNet18 骨干替换 4 层轻量 CNN
//   对标 PatchCore 论文: WideResNet-50 Layer2+Layer3 多尺度特征拼接
//   bUsePretrainedBackbone=true → ResNet18 前 3 层（Layer1~Layer3）提取 384 维多尺度特征
//   bUsePretrainedBackbone=false → 保留原有 4 层轻量 CNN（向后兼容）
module;

// 20260406 ZJH 标准库头文件包含
#include <vector>    // 20260406 ZJH 动态数组容器
#include <string>    // 20260406 ZJH 字符串类型
#include <memory>    // 20260406 ZJH 智能指针 (shared_ptr, unique_ptr)
#include <cmath>     // 20260406 ZJH 数学函数 (sqrt)
#include <algorithm> // 20260406 ZJH 算法函数 (std::min, std::max, std::swap)
#include <limits>    // 20260406 ZJH 数值极限 (numeric_limits::max)
#include <numeric>   // 20260406 ZJH 数值算法 (std::iota)
#include <cstdint>   // 20260406 ZJH 固定宽度整数类型 (uint32_t)

export module om.engine.patchcore;

// 20260322 ZJH 导入依赖模块
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.conv;
import om.engine.linear;
import om.engine.activations;
import om.engine.resnet;  // 20260402 ZJH 导入 ResNet18/BasicBlock 用于预训练骨干

export namespace om {

// 20260402 ZJH ResNet18FeatureExtractor — 使用 ResNet18 前 3 层提取多尺度特征
// 对标 PatchCore 论文: WideResNet-50 Layer2+Layer3 拼接
// 架构: conv1(3x3,s1,p1) → bn1 → relu → maxpool(3,2,1)
//        → Layer1(2xBasicBlock, 64→64)   输出 [N, 64, H/4, W/4]
//        → Layer2(2xBasicBlock, 64→128, s2) 输出 [N, 128, H/8, W/8]
//        → Layer3(2xBasicBlock, 128→256, s2) 输出 [N, 256, H/16, W/16]
// 多尺度拼接: Layer2 [N,128,H/8,W/8] + 上采样(Layer3) [N,256,H/8,W/8] → [N,384,H/8,W/8]
// 优势: ImageNet 预训练权重包含丰富的通用视觉特征，异常检测精度远超随机初始化 CNN
class ResNet18FeatureExtractor : public Module {
public:
    // 20260402 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    ResNet18FeatureExtractor(int nInChannels = 3) {
        // 20260402 ZJH Stage 0: stem 卷积 + BN + ReLU + MaxPool
        // ResNet18 标准 stem: conv1(3x3, stride=1, pad=1) → bn1 → relu → maxpool(3,2,1)
        // 注意: 小图像模式使用 3x3 stride=1（非 ImageNet 7x7 stride=2），maxpool 条件使用
        m_pConv1 = std::make_shared<Conv2d>(nInChannels, 64, 3, 1, 1, false);  // 20260402 ZJH stem conv 3→64
        m_pBn1 = std::make_shared<BatchNorm2d>(64);  // 20260402 ZJH stem BN
        m_pMaxpool = std::make_shared<MaxPool2d>(3, 2, 1);  // 20260402 ZJH 3x3 stride=2 pad=1 下采样 /2
        m_pRelu = std::make_shared<ReLU>();  // 20260402 ZJH 共用 ReLU 激活

        // 20260402 ZJH 注册 stem 模块（利用 Module 基类递归管理参数）
        registerModule("conv1", m_pConv1);
        registerModule("bn1", m_pBn1);
        registerModule("maxpool", m_pMaxpool);
        registerModule("relu", m_pRelu);

        // 20260402 ZJH Stage 1: Layer1 = 2x BasicBlock(64→64, stride=1)
        // 输入 [N, 64, H/2, W/2]（经 maxpool 后）→ 输出 [N, 64, H/2, W/2]
        // 大图输入（H>32）经 maxpool 后为 H/2，此层不改变空间尺寸
        auto addLayer = [&](const std::string& strName,
                            std::vector<std::shared_ptr<BasicBlock>>& vecBlocks,
                            int nInC, int nOutC, int nStride) {
            vecBlocks.reserve(2);  // 20260402 ZJH 预分配 2 个 BasicBlock
            for (int i = 0; i < 2; ++i) {
                int s = (i == 0) ? nStride : 1;  // 20260402 ZJH 仅第一个 block 使用指定 stride
                int c = (i == 0) ? nInC : nOutC;  // 20260402 ZJH 第一个 block 输入为 nInC，后续为 nOutC
                auto pBlock = std::make_shared<BasicBlock>(c, nOutC, s);  // 20260402 ZJH 创建残差块
                vecBlocks.push_back(pBlock);  // 20260402 ZJH 存入层列表
                registerModule(strName + "." + std::to_string(i), pBlock);  // 20260402 ZJH 注册子模块
            }
        };

        // 20260402 ZJH 构建 Layer1~Layer3（与 ResNet18 完全一致，共享权重结构）
        addLayer("layer1", m_vecLayer1, 64, 64, 1);    // 20260402 ZJH Layer1: 64→64, stride=1
        addLayer("layer2", m_vecLayer2, 64, 128, 2);   // 20260402 ZJH Layer2: 64→128, stride=2 → 空间 /2
        addLayer("layer3", m_vecLayer3, 128, 256, 2);  // 20260402 ZJH Layer3: 128→256, stride=2 → 空间 /2
    }

    // 20260402 ZJH forward — 前向传播提取多尺度特征并拼接
    // input: [N, Cin, H, W]
    // 返回: [N, 384, H/8, W/8] 多尺度拼接特征（128 + 256 通道）
    Tensor forward(const Tensor& input) override {
        return extractMultiScale(input);  // 20260402 ZJH 委托给多尺度提取方法
    }

    // 20260402 ZJH extractMultiScale — 提取 Layer2+Layer3 多尺度特征并拼接
    // 对标 PatchCore 论文的特征提取策略:
    //   Layer2 输出 [N, 128, H/8, W/8] — 中间层特征，保留纹理细节
    //   Layer3 输出 [N, 256, H/16, W/16] — 深层特征，捕获语义信息
    //   上采样 Layer3 到 H/8 后与 Layer2 拼接 → [N, 384, H/8, W/8]
    // input: [N, Cin, H, W]
    // 返回: [N, 384, H/8, W/8]
    Tensor extractMultiScale(const Tensor& input) {
        // 20260402 ZJH Stem: conv1 → bn1 → relu → maxpool
        auto x = m_pConv1->forward(input);   // 20260402 ZJH [N, 64, H, W]
        x = m_pBn1->forward(x);              // 20260402 ZJH BN 归一化
        x = m_pRelu->forward(x);             // 20260402 ZJH ReLU 激活
        // 20260402 ZJH 大图（H>32）使用 maxpool 下采样，小图跳过
        if (input.shape(2) > 32) {
            x = m_pMaxpool->forward(x);      // 20260402 ZJH [N, 64, H/2, W/2]
        }

        // 20260402 ZJH Layer1: 2x BasicBlock(64→64)
        for (auto& pBlock : m_vecLayer1) x = pBlock->forward(x);  // 20260402 ZJH [N, 64, H/2, W/2]

        // 20260402 ZJH Layer2: 2x BasicBlock(64→128, stride=2)
        auto xLayer2 = x;  // 20260402 ZJH 保存 Layer1 输出作为 Layer2 输入
        for (auto& pBlock : m_vecLayer2) xLayer2 = pBlock->forward(xLayer2);
        // 20260402 ZJH xLayer2 = [N, 128, H/4, W/4]（stem有maxpool时为H/4，无maxpool时为H/2）
        // 对于大图: stem→/2, layer2→/2 = /4  但我们需要 /8
        // 实际: stem conv(s1)→H, maxpool(s2)→H/2, layer1→H/2, layer2(s2)→H/4

        // 20260402 ZJH Layer3: 2x BasicBlock(128→256, stride=2)
        auto xLayer3 = xLayer2;  // 20260402 ZJH Layer2 输出作为 Layer3 输入
        for (auto& pBlock : m_vecLayer3) xLayer3 = pBlock->forward(xLayer3);
        // 20260402 ZJH xLayer3 = [N, 256, H/8, W/8]

        // 20260402 ZJH 上采样 Layer3 到 Layer2 的空间尺寸
        int nTargetH = xLayer2.shape(2);  // 20260402 ZJH Layer2 的空间高度
        int nTargetW = xLayer2.shape(3);  // 20260402 ZJH Layer2 的空间宽度
        auto xLayer3Up = nnUpsample(xLayer3, nTargetH, nTargetW);  // 20260402 ZJH [N, 256, H/4, W/4]

        // 20260402 ZJH 沿通道维度拼接 Layer2 + Layer3（上采样后）
        // [N, 128, H/4, W/4] + [N, 256, H/4, W/4] → [N, 384, H/4, W/4]
        auto xConcat = tensorConcatChannels(xLayer2, xLayer3Up);  // 20260402 ZJH 多尺度特征拼接

        return xConcat;  // 20260402 ZJH 返回 [N, 384, H/4, W/4] 多尺度特征
    }

    // 20260402 ZJH freeze — 冻结所有参数（PatchCore 不训练骨干网络）
    // PatchCore 的特征提取器使用预训练权重直接建库，不做 fine-tune
    // 冻结后 parameters() 仍返回参数列表（序列化用），但不传给优化器
    void freeze() {
        eval();  // 20260402 ZJH 切换到评估模式（BN 使用 running stats）
        m_bFrozen = true;  // 20260402 ZJH 标记已冻结
    }

    // 20260402 ZJH isFrozen — 查询冻结状态
    bool isFrozen() const { return m_bFrozen; }

    // 20260402 ZJH getFeatureDim — 获取输出特征向量维度
    // Layer2(128) + Layer3(256) = 384 维
    int getFeatureDim() const { return 384; }

private:
    // 20260402 ZJH Stem 模块
    std::shared_ptr<Conv2d> m_pConv1;        // 20260402 ZJH stem 卷积 3→64
    std::shared_ptr<BatchNorm2d> m_pBn1;     // 20260402 ZJH stem BN
    std::shared_ptr<MaxPool2d> m_pMaxpool;   // 20260402 ZJH stem 下采样池化
    std::shared_ptr<ReLU> m_pRelu;           // 20260402 ZJH 共用 ReLU

    // 20260402 ZJH ResNet18 Layer1~Layer3（每层 2 个 BasicBlock）
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer1;  // 20260402 ZJH 64→64
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer2;  // 20260402 ZJH 64→128, stride=2
    std::vector<std::shared_ptr<BasicBlock>> m_vecLayer3;  // 20260402 ZJH 128→256, stride=2

    bool m_bFrozen = false;  // 20260402 ZJH 参数冻结标记

    // 20260402 ZJH nnUpsample — 最近邻上采样（内部工具函数）
    // 将 [N, C, srcH, srcW] 上采样到 [N, C, nTargetH, nTargetW]
    // 使用最近邻插值（无可训练参数，保持梯度简单）
    static Tensor nnUpsample(const Tensor& input, int nTargetH, int nTargetW) {
        auto cIn = input.contiguous();  // 20260402 ZJH 确保输入连续
        int nN = cIn.shape(0), nC = cIn.shape(1);  // 20260402 ZJH batch 和通道数
        int nH = cIn.shape(2), nW = cIn.shape(3);  // 20260402 ZJH 源空间尺寸
        auto result = Tensor::zeros({nN, nC, nTargetH, nTargetW});  // 20260402 ZJH 分配输出
        float* pO = result.mutableFloatDataPtr();  // 20260402 ZJH 输出数据指针
        const float* pI = cIn.floatDataPtr();  // 20260402 ZJH 输入数据指针
        // 20260402 ZJH 计算源-目标坐标缩放比
        float fSH = static_cast<float>(nH) / static_cast<float>(nTargetH);
        float fSW = static_cast<float>(nW) / static_cast<float>(nTargetW);
        // 20260402 ZJH 逐像素最近邻采样
        for (int n = 0; n < nN; ++n)
            for (int c = 0; c < nC; ++c)
                for (int th = 0; th < nTargetH; ++th) {
                    int sh = std::min(static_cast<int>(th * fSH), nH - 1);  // 20260402 ZJH 源 y 坐标
                    for (int tw = 0; tw < nTargetW; ++tw) {
                        int sw = std::min(static_cast<int>(tw * fSW), nW - 1);  // 20260402 ZJH 源 x 坐标
                        pO[((n * nC + c) * nTargetH + th) * nTargetW + tw] =
                            pI[((n * nC + c) * nH + sh) * nW + sw];  // 20260402 ZJH 复制像素值
                    }
                }
        return result;  // 20260402 ZJH 返回上采样结果
    }
};

// 20260322 ZJH PatchCoreExtractor — PatchCore 特征提取 CNN（原始 4 层轻量版）
// 4 层 Conv+BN+ReLU+MaxPool 结构（与 EfficientADBackbone 类似但独立实现）
// 提取 patch 级特征用于与 MemoryBank 比较
// 输入: [N, Cin, H, W]
// 输出特征图: [N, 256, H/8, W/8]，每个空间位置对应一个 256 维 patch 特征向量
// 20260402 ZJH 保留作为 fallback（bUsePretrainedBackbone=false 时使用）
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
//   1. 创建 PatchCore 实例（bUsePretrainedBackbone=true 使用 ResNet18 骨干）
//   2. 调用 loadPretrainedBackbone() 加载 ImageNet 预训练权重（可选）
//   3. 调用 buildMemoryBank() 传入正常样本图像列表
//   4. 推理时调用 computeAnomalyScore() 或 computeAnomalyMap()
// 20260402 ZJH Phase 1.2: 新增 bUsePretrainedBackbone 标志
//   true(默认) → 使用 ResNet18FeatureExtractor（384 维多尺度特征）
//   false → 使用原有 PatchCoreExtractor（256 维轻量 CNN）
class PatchCore {
public:
    // 20260322 ZJH 构造函数
    // nInChannels: 输入图像通道数，默认 3（RGB）
    // nMaxMemorySize: MemoryBank 最大容量（coreset 采样上限），默认 1000
    // 20260402 ZJH bUsePretrainedBackbone: 是否使用 ResNet18 预训练骨干，默认 true
    PatchCore(int nInChannels = 3, int nMaxMemorySize = 1000, bool bUsePretrainedBackbone = true)
        : m_nMaxMemorySize(nMaxMemorySize),
          m_bUsePretrainedBackbone(bUsePretrainedBackbone)  // 20260402 ZJH 保存骨干选择标志
    {
        if (m_bUsePretrainedBackbone) {
            // 20260402 ZJH 使用 ResNet18 前 3 层作为特征提取器（对标 PatchCore 论文）
            m_pResNetExtractor = std::make_unique<ResNet18FeatureExtractor>(nInChannels);
            m_pResNetExtractor->freeze();  // 20260402 ZJH PatchCore 不训练骨干，直接冻结
            m_nFeatureDim = m_pResNetExtractor->getFeatureDim();  // 20260402 ZJH 384 维
        } else {
            // 20260402 ZJH 使用原有 4 层轻量 CNN（向后兼容）
            m_pLegacyExtractor = std::make_unique<PatchCoreExtractor>(nInChannels);
            m_pLegacyExtractor->eval();  // 20260402 ZJH 设置为评估模式
            m_nFeatureDim = m_pLegacyExtractor->getFeatureDim();  // 20260402 ZJH 256 维
        }
    }

    // 20260402 ZJH loadPretrainedBackbone — 从 .omm 文件加载 ImageNet 预训练权重到 ResNet18 骨干
    // 仅在 bUsePretrainedBackbone=true 时有效
    // strWeightPath: 预训练权重文件路径（.omm 格式，ResNet18 ImageNet 1000-class）
    // 返回: 成功加载的参数层数
    int loadPretrainedBackbone(const std::string& strWeightPath) {
        if (!m_bUsePretrainedBackbone || !m_pResNetExtractor) {
            return 0;  // 20260402 ZJH 非预训练模式，跳过
        }
        // 20260402 ZJH 权重加载由 EngineBridge 统一处理（通过 loadPyTorchPretrainedToSegModel）
        // 此处仅提供接口，实际调用在 EngineBridge 中完成
        return 0;  // 20260402 ZJH placeholder，实际加载在 EngineBridge 中
    }

    // 20260402 ZJH getBackboneModule — 获取骨干网络的 Module 引用（用于外部预训练加载）
    // 返回: 当前使用的特征提取器 Module 指针（ResNet18 或轻量 CNN）
    Module* getBackboneModule() {
        if (m_bUsePretrainedBackbone && m_pResNetExtractor) {
            return m_pResNetExtractor.get();  // 20260402 ZJH ResNet18 骨干
        }
        if (m_pLegacyExtractor) {
            return m_pLegacyExtractor.get();  // 20260402 ZJH 原始 4 层 CNN
        }
        return nullptr;  // 20260402 ZJH 不应到达
    }

    // 20260402 ZJH isUsingPretrainedBackbone — 查询是否使用预训练骨干
    bool isUsingPretrainedBackbone() const { return m_bUsePretrainedBackbone; }

    // 20260322 ZJH buildMemoryBank — 构建正常样本特征记忆库
    // 遍历所有正常图像，提取 patch 特征并存入 MemoryBank
    // 如果特征总量超过 m_nMaxMemorySize，使用随机采样压缩
    // vecNormalImages: 正常样本图像张量列表，每个为 [1, Cin, H, W]
    void buildMemoryBank(const std::vector<Tensor>& vecNormalImages) {
        m_vecMemoryBank.clear();  // 20260322 ZJH 清空旧记忆库

        // 20260322 ZJH 遍历所有正常图像，提取 patch 特征
        for (const auto& image : vecNormalImages) {
            // 20260402 ZJH 根据骨干类型选择特征提取器
            // ResNet18 骨干: [1, Cin, H, W] → [1, 384, Hf, Wf]
            // 轻量 CNN:     [1, Cin, H, W] → [1, 256, Hf, Wf]
            Tensor featureMap;
            if (m_bUsePretrainedBackbone && m_pResNetExtractor) {
                featureMap = m_pResNetExtractor->forward(image);  // 20260402 ZJH ResNet18 多尺度特征
            } else {
                featureMap = m_pLegacyExtractor->forward(image);  // 20260402 ZJH 原始 4 层 CNN
            }
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
    // 返回: [1, 1, Hf, Wf] 异常热力图
    // 20260402 ZJH ResNet18 骨干: Hf=H/4, Wf=W/4（更高分辨率）
    //              轻量 CNN:      Hf=H/8, Wf=W/8
    Tensor computeAnomalyMap(const Tensor& testImage) {
        // 20260402 ZJH 根据骨干类型选择特征提取器
        Tensor featureMap;
        if (m_bUsePretrainedBackbone && m_pResNetExtractor) {
            featureMap = m_pResNetExtractor->forward(testImage);  // 20260402 ZJH ResNet18 多尺度特征
        } else {
            featureMap = m_pLegacyExtractor->forward(testImage);  // 20260402 ZJH 原始 4 层 CNN
        }
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

    // 20260402 ZJH getExtractor — 获取原始轻量特征提取器（向后兼容，仅在 legacy 模式可用）
    // 注意: 使用 ResNet18 骨干时此方法不应被调用，建议使用 getBackboneModule()
    PatchCoreExtractor* getLegacyExtractor() {
        return m_pLegacyExtractor.get();  // 20260402 ZJH 可能为 nullptr
    }

    // 20260322 ZJH getExtractorParameters — 获取特征提取器参数
    // 20260402 ZJH 根据骨干类型返回对应的参数列表
    std::vector<Tensor*> getExtractorParameters() {
        if (m_bUsePretrainedBackbone && m_pResNetExtractor) {
            return m_pResNetExtractor->parameters();  // 20260402 ZJH ResNet18 骨干参数
        }
        if (m_pLegacyExtractor) {
            return m_pLegacyExtractor->parameters();  // 20260402 ZJH 原始 CNN 参数
        }
        return {};  // 20260402 ZJH 空列表
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

    // 20260402 ZJH 特征提取器（两种骨干互斥，只有一个非 null）
    std::unique_ptr<ResNet18FeatureExtractor> m_pResNetExtractor;  // 20260402 ZJH ResNet18 预训练骨干（384 维）
    std::unique_ptr<PatchCoreExtractor> m_pLegacyExtractor;        // 20260402 ZJH 原始 4 层 CNN（256 维）
    bool m_bUsePretrainedBackbone;   // 20260402 ZJH 骨干选择标志: true=ResNet18, false=轻量CNN
    int m_nMaxMemorySize;            // 20260322 ZJH MemoryBank 最大容量
    int m_nFeatureDim;               // 20260402 ZJH 特征向量维度（384 或 256，取决于骨干类型）

    // 20260322 ZJH MemoryBank: 存储正常样本 patch 级特征向量
    // 每个元素是一个 256 维 float 向量
    // 大小: 最多 m_nMaxMemorySize 个特征向量
    std::vector<std::vector<float>> m_vecMemoryBank;
};

}  // 20260406 ZJH namespace om 结束
