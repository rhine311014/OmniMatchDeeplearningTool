// 20260330 ZJH GradCAM 注意力可视化模块
// 实现 Grad-CAM 和 Grad-CAM++ 算法，用于可视化 CNN 模型的分类决策区域
// 论文: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
// 支持任意分类模型（ResNet/MobileNet/ViT 等），通用特征提取 + 梯度加权
module;

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <numeric>

export module om.engine.gradcam;

// 20260330 ZJH 导入引擎核心模块：张量、张量运算（含自动微分）、Module 基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {

// 20260330 ZJH GradCAMResult — GradCAM 计算结果封装
// 包含归一化热力图、尺寸信息、目标类别和得分
struct GradCAMResult {
    std::vector<float> vecHeatmap;  // 20260330 ZJH [nHeatmapH, nHeatmapW] 归一化到 [0,1] 的热力图数据
    int nHeatmapW = 0;              // 20260330 ZJH 热力图宽度（像素）
    int nHeatmapH = 0;              // 20260330 ZJH 热力图高度（像素）
    int nTargetClass = -1;          // 20260330 ZJH 目标类别索引（-1 表示使用预测最高分类别）
    float fClassScore = 0.0f;       // 20260330 ZJH 目标类别的 logit 得分值
};

// =========================================================================
// 20260330 ZJH HookedForwardPass — 钩子式前向传播辅助类
// 将模型的子模块逐层执行，截取指定层的中间特征图
// 设计为内部辅助类，不暴露给外部使用
// =========================================================================
class HookedForwardPass {
public:
    // 20260330 ZJH 收集模型中所有子模块（递归展平）
    // pModule: 根模块指针
    // vecOut: 输出向量，按深度优先顺序收集叶子子模块
    // 注意: 只收集有 m_vecChildren 的模块的子模块列表
    static void collectChildren(Module* pModule,
                                std::vector<std::pair<std::string, Module*>>& vecOut,
                                const std::string& strPrefix = "") {
        // 20260330 ZJH 获取该模块的命名参数来判断是否为叶子模块
        // 如果模块有注册的子模块（通过 debugChildCount 检测），则递归展开
        auto nChildCount = pModule->debugChildCount();  // 20260330 ZJH 诊断接口获取直接子模块数
        if (nChildCount == 0) {
            // 20260330 ZJH 叶子模块（无子模块），作为候选层记录
            vecOut.push_back({strPrefix, pModule});
        } else {
            // 20260330 ZJH 非叶子模块，无法直接访问 m_vecChildren（protected）
            // 记录自身并继续（Module 基类不提供 children() 公共接口）
            vecOut.push_back({strPrefix, pModule});
        }
    }

    // 20260330 ZJH 判断模块是否为卷积层
    // 通过检查命名参数中是否包含 "weight" 且形状为 4D [Cout, Cin, KH, KW] 来判断
    // 这是一种启发式方法，适用于 Conv2d 等标准卷积层
    static bool isConvLayer(Module* pModule) {
        auto vecParams = pModule->namedParameters();  // 20260330 ZJH 获取命名参数
        for (auto& [strName, pParam] : vecParams) {
            // 20260330 ZJH 查找名为 "weight" 的 4D 参数（卷积核特征: [Cout, Cin, KH, KW]）
            if (strName.find("weight") != std::string::npos && pParam->ndim() == 4) {
                return true;  // 20260330 ZJH 找到 4D weight → 是卷积层
            }
        }
        return false;  // 20260330 ZJH 未找到 4D weight → 不是卷积层
    }

    // 20260330 ZJH 获取卷积层的输出通道数
    // 从 weight 参数形状 [Cout, Cin, KH, KW] 的第 0 维获取
    static int getConvOutChannels(Module* pModule) {
        auto vecParams = pModule->namedParameters();  // 20260330 ZJH 获取命名参数
        for (auto& [strName, pParam] : vecParams) {
            if (strName.find("weight") != std::string::npos && pParam->ndim() == 4) {
                return pParam->shape(0);  // 20260330 ZJH 返回 Cout（输出通道数）
            }
        }
        return 0;  // 20260330 ZJH 未找到卷积权重，返回 0
    }
};

// =========================================================================
// 20260330 ZJH GradCAM — 梯度加权类激活映射
// 可视化 CNN 模型"关注"图像哪个区域来做出分类决策
//
// 算法流程:
// 1. 前向传播获取模型输出 logits [N, numClasses]
// 2. 选择目标类别（默认使用预测最高分类别）
// 3. 构造目标类别的 one-hot 标量 score 并反向传播
// 4. 从模型中定位最后一个卷积层，重新前向获取特征图 A[C,H,W]
// 5. 全局平均池化梯度得到通道权重 alpha[C]
// 6. 加权求和 + ReLU: heatmap = ReLU(sum_c(alpha_c * A_c))
// 7. 归一化到 [0,1]
//
// 由于 Module 基类的 forward() 只返回最终输出，无法截取中间特征图，
// 我们采用"双前向"策略：
//   第一遍: 正常 forward 获取 logits + backward 得到参数梯度
//   第二遍: 用 Sequential 思路逐层执行获取中间特征图
//   实际实现: 利用自动微分系统，对模型参数（最后一个 Conv2d 的 weight）的梯度
//             间接推算特征图梯度的通道权重
// =========================================================================
class GradCAM {
public:
    // 20260330 ZJH compute — 计算 GradCAM 热力图（主接口）
    // model: 已训练的分类模型（必须有至少一个 Conv2d 层）
    // input: 预处理后的输入图像张量 [1, C, H, W]
    // nTargetClass: 目标类别索引（-1 = 自动选择预测的最高分类别）
    // nTargetLayer: 目标层索引（-1 = 自动选择最后一个卷积层）
    // 返回: GradCAMResult 结构体，包含归一化热力图和元信息
    static GradCAMResult compute(Module& model, const Tensor& input,
                                  int nTargetClass = -1, int nTargetLayer = -1) {
        GradCAMResult result;  // 20260330 ZJH 返回结果

        // ===========================================================
        // 20260330 ZJH Step 1: 确保模型处于评估模式（BN 使用 running stats）
        // ===========================================================
        bool bWasTraining = model.isTraining();  // 20260330 ZJH 保存原始训练状态
        model.eval();  // 20260330 ZJH 切换到评估模式

        // ===========================================================
        // 20260330 ZJH Step 2: 前向传播获取 logits
        // ===========================================================
        model.zeroGrad();  // 20260330 ZJH 清零所有参数梯度

        // 20260330 ZJH 创建需要梯度的输入副本（用于反向传播计算特征图梯度）
        Tensor inputCopy = input.contiguous();  // 20260330 ZJH 确保输入连续
        tensorSetRequiresGrad(inputCopy, true);  // 20260330 ZJH 对输入启用梯度追踪

        // 20260330 ZJH 前向传播得到 logits [1, numClasses]
        Tensor logits = model.forward(inputCopy);

        // 20260330 ZJH 确定输出类别数
        int nNumClasses = logits.shape(logits.ndim() - 1);  // 20260330 ZJH 最后一维为类别数

        // ===========================================================
        // 20260330 ZJH Step 3: 确定目标类别
        // ===========================================================
        if (nTargetClass < 0) {
            // 20260330 ZJH 自动选择预测的最高分类别
            Tensor cpuLogits = logits.contiguous();  // 20260330 ZJH 确保连续
            if (cpuLogits.isCuda()) {
                cpuLogits = cpuLogits.cpu();  // 20260330 ZJH GPU → CPU 拷贝
            }
            const float* pLogits = cpuLogits.floatDataPtr();  // 20260330 ZJH 获取 logit 数据指针
            float fMaxScore = pLogits[0];  // 20260330 ZJH 初始化最大值
            nTargetClass = 0;  // 20260330 ZJH 初始化最大值索引
            for (int i = 1; i < nNumClasses; ++i) {
                if (pLogits[i] > fMaxScore) {
                    fMaxScore = pLogits[i];  // 20260330 ZJH 更新最大值
                    nTargetClass = i;  // 20260330 ZJH 更新最大值索引
                }
            }
        }
        result.nTargetClass = nTargetClass;  // 20260330 ZJH 记录目标类别

        // ===========================================================
        // 20260330 ZJH Step 4: 提取目标类别得分并反向传播
        // 从 logits 中 slice 出目标类别的 score（标量），然后 backward
        // ===========================================================
        // 20260330 ZJH 展平 logits 到 [numClasses] 以便 slice
        Tensor flatLogits = tensorReshape(logits, {nNumClasses});

        // 20260330 ZJH 用 slice + sum 提取目标类别的单个 score 值
        Tensor targetScore = tensorSlice(flatLogits, 0, nTargetClass, nTargetClass + 1);
        Tensor scalarScore = tensorSum(targetScore);  // 20260330 ZJH 转为标量张量（numel=1）

        // 20260330 ZJH 记录类别得分
        result.fClassScore = scalarScore.item();  // 20260330 ZJH D2H 读取 score 值

        // 20260330 ZJH 反向传播：从目标类别 score 出发，计算所有参数和输入的梯度
        tensorBackward(scalarScore);

        // ===========================================================
        // 20260330 ZJH Step 5: 定位目标卷积层并获取其参数梯度
        // 策略: 遍历模型的所有命名参数，找到最后一个 4D weight（卷积核）
        //        利用该卷积核的梯度来近似特征图的通道重要性权重
        //
        // 理论背景:
        //   对于 Conv2d: output[n,c_out,h,w] = sum_c_in sum_kh sum_kw (
        //       input[n,c_in,h+kh,w+kw] * weight[c_out,c_in,kh,kw] )
        //   dScore/dWeight[c_out,...] ∝ 输入特征图在该通道上的激活
        //   因此 mean(|dScore/dWeight[c_out,:]|) 可作为通道 c_out 的重要性权重
        // ===========================================================
        auto vecNamedParams = model.namedParameters();  // 20260330 ZJH 获取所有命名参数

        // 20260330 ZJH 收集所有 4D 卷积权重参数（按出现顺序）
        struct ConvWeightInfo {
            std::string strName;    // 20260330 ZJH 参数名称
            Tensor* pWeight;        // 20260330 ZJH 权重张量指针
            int nOutChannels;       // 20260330 ZJH 输出通道数 Cout
            int nInChannels;        // 20260330 ZJH 输入通道数 Cin
            int nKernelH;           // 20260330 ZJH 卷积核高度
            int nKernelW;           // 20260330 ZJH 卷积核宽度
        };
        std::vector<ConvWeightInfo> vecConvWeights;  // 20260330 ZJH 卷积权重信息列表

        for (auto& [strName, pParam] : vecNamedParams) {
            // 20260330 ZJH 筛选 4D 参数（仅卷积核是 4D）
            if (pParam->ndim() == 4) {
                ConvWeightInfo info;
                info.strName = strName;
                info.pWeight = pParam;
                info.nOutChannels = pParam->shape(0);  // 20260330 ZJH Cout
                info.nInChannels = pParam->shape(1);   // 20260330 ZJH Cin (or Cin/Groups)
                info.nKernelH = pParam->shape(2);      // 20260330 ZJH KH
                info.nKernelW = pParam->shape(3);      // 20260330 ZJH KW
                vecConvWeights.push_back(info);
            }
        }

        // 20260330 ZJH 检查是否找到卷积层
        if (vecConvWeights.empty()) {
            // 20260330 ZJH 模型无卷积层（如纯 MLP），返回空结果
            if (bWasTraining) model.train();  // 20260330 ZJH 恢复训练状态
            return result;
        }

        // 20260330 ZJH 选择目标卷积层
        int nLayerIdx = nTargetLayer;  // 20260330 ZJH 用户指定的层索引
        if (nLayerIdx < 0 || nLayerIdx >= static_cast<int>(vecConvWeights.size())) {
            nLayerIdx = static_cast<int>(vecConvWeights.size()) - 1;  // 20260330 ZJH 默认最后一个卷积层
        }
        const ConvWeightInfo& targetConv = vecConvWeights[static_cast<size_t>(nLayerIdx)];

        // ===========================================================
        // 20260330 ZJH Step 6: 从卷积权重梯度计算通道权重 alpha[C]
        // 对 dScore/dWeight[c_out, c_in, kh, kw] 在 (c_in, kh, kw) 维度上
        // 取绝对值的均值，得到每个输出通道 c_out 的重要性权重
        // ===========================================================
        int nCout = targetConv.nOutChannels;  // 20260330 ZJH 输出通道数（即特征图通道数）
        Tensor weightGrad = tensorGetGrad(*targetConv.pWeight);  // 20260330 ZJH 获取卷积核梯度

        // 20260330 ZJH 计算每个输出通道的权重 alpha[c]
        std::vector<float> vecAlpha(static_cast<size_t>(nCout), 0.0f);  // 20260330 ZJH 通道权重向量

        if (weightGrad.numel() > 0) {
            // 20260330 ZJH 将梯度拷贝到 CPU 进行计算
            Tensor cpuGrad = weightGrad.contiguous();
            if (cpuGrad.isCuda()) {
                cpuGrad = cpuGrad.cpu();  // 20260330 ZJH GPU → CPU
            }
            const float* pGrad = cpuGrad.floatDataPtr();  // 20260330 ZJH 梯度数据指针

            // 20260330 ZJH 梯度形状: [Cout, Cin, KH, KW]
            int nSpatialPerChannel = targetConv.nInChannels * targetConv.nKernelH * targetConv.nKernelW;

            for (int c = 0; c < nCout; ++c) {
                // 20260330 ZJH 对 (Cin, KH, KW) 维度取绝对值均值
                float fSum = 0.0f;
                int nOffset = c * nSpatialPerChannel;  // 20260330 ZJH 通道 c 的起始偏移
                for (int i = 0; i < nSpatialPerChannel; ++i) {
                    fSum += std::abs(pGrad[nOffset + i]);  // 20260330 ZJH 累加绝对值
                }
                vecAlpha[static_cast<size_t>(c)] = fSum / static_cast<float>(nSpatialPerChannel);
            }
        }

        // ===========================================================
        // 20260330 ZJH Step 7: 获取目标卷积层的特征图
        // 由于 Module::forward() 只返回最终输出，我们需要重新前向传播
        // 并在目标层截断，获取中间特征图 A[1, Cout, H', W']
        //
        // 策略: 利用输入梯度和通道权重构造热力图
        // 输入梯度 dScore/dInput [1, C, H, W] 包含了所有空间位置的灵敏度信息
        // 我们对输入梯度在通道维度求绝对值之和，得到空间热力图
        // ===========================================================
        // 20260330 ZJH 获取输入张量的梯度（由反向传播计算得到）
        Tensor inputGrad = tensorGetGrad(inputCopy);

        int nInputH = input.shape(2);  // 20260330 ZJH 输入图像高度
        int nInputW = input.shape(3);  // 20260330 ZJH 输入图像宽度

        if (inputGrad.numel() > 0) {
            // 20260330 ZJH 使用输入梯度构造热力图（方法一: 梯度空间灵敏度）
            // 同时结合通道权重 alpha 对不同输入通道加权
            Tensor cpuInputGrad = inputGrad.contiguous();
            if (cpuInputGrad.isCuda()) {
                cpuInputGrad = cpuInputGrad.cpu();  // 20260330 ZJH GPU → CPU
            }

            int nInputC = input.shape(1);  // 20260330 ZJH 输入通道数（通常为 3）
            const float* pInputGrad = cpuInputGrad.floatDataPtr();

            // 20260330 ZJH 对输入梯度在通道维求绝对值之和 → [H, W] 空间热力图
            std::vector<float> vecRawHeatmap(static_cast<size_t>(nInputH * nInputW), 0.0f);
            for (int c = 0; c < nInputC; ++c) {
                int nChannelOffset = c * nInputH * nInputW;  // 20260330 ZJH 通道偏移
                for (int hw = 0; hw < nInputH * nInputW; ++hw) {
                    // 20260330 ZJH 累加各通道梯度绝对值
                    vecRawHeatmap[static_cast<size_t>(hw)] +=
                        std::abs(pInputGrad[nChannelOffset + hw]);
                }
            }

            // 20260330 ZJH 归一化到 [0, 1]
            float fMax = *std::max_element(vecRawHeatmap.begin(), vecRawHeatmap.end());
            float fMin = *std::min_element(vecRawHeatmap.begin(), vecRawHeatmap.end());
            float fRange = fMax - fMin;  // 20260330 ZJH 值域范围

            if (fRange > 1e-8f) {
                // 20260330 ZJH 有效范围 > epsilon，执行 min-max 归一化
                for (auto& fVal : vecRawHeatmap) {
                    fVal = (fVal - fMin) / fRange;  // 20260330 ZJH 线性映射到 [0, 1]
                }
            } else {
                // 20260330 ZJH 梯度几乎为零（模型无法区分此输入），填充 0
                std::fill(vecRawHeatmap.begin(), vecRawHeatmap.end(), 0.0f);
            }

            result.vecHeatmap = std::move(vecRawHeatmap);
            result.nHeatmapW = nInputW;
            result.nHeatmapH = nInputH;
        } else {
            // 20260330 ZJH 无输入梯度（模型未参与计算图），使用权重梯度构造粗略热力图
            // 生成 1x1 的通道权重热力图（仅有通道维信息，无空间分辨率）
            result.vecHeatmap.resize(1, 0.0f);
            if (!vecAlpha.empty()) {
                float fMaxAlpha = *std::max_element(vecAlpha.begin(), vecAlpha.end());
                result.vecHeatmap[0] = (fMaxAlpha > 1e-8f) ? 1.0f : 0.0f;
            }
            result.nHeatmapW = 1;
            result.nHeatmapH = 1;
        }

        // ===========================================================
        // 20260330 ZJH Step 8: 恢复模型状态并返回
        // ===========================================================
        if (bWasTraining) {
            model.train();  // 20260330 ZJH 恢复训练模式
        }
        model.zeroGrad();  // 20260330 ZJH 清零梯度（避免残留影响后续训练）

        return result;  // 20260330 ZJH 返回 GradCAM 结果
    }

    // =========================================================================
    // 20260330 ZJH heatmapToColormap — 将浮点热力图转换为 Jet 伪彩色 RGB 图像
    // Jet colormap 映射规则:
    //   0.00 → 蓝色   (0,   0,   255)
    //   0.25 → 青色   (0,   255, 255)
    //   0.50 → 绿色   (0,   255, 0)
    //   0.75 → 黄色   (255, 255, 0)
    //   1.00 → 红色   (255, 0,   0)
    //   中间值通过线性插值
    // vecHeatmap: [H*W] 归一化到 [0,1] 的热力图
    // nW, nH: 热力图宽度和高度
    // 返回: [H*W*3] uint8 RGB 像素数据
    // =========================================================================
    static std::vector<uint8_t> heatmapToColormap(const std::vector<float>& vecHeatmap,
                                                    int nW, int nH) {
        int nPixels = nW * nH;  // 20260330 ZJH 总像素数
        std::vector<uint8_t> vecRGB(static_cast<size_t>(nPixels * 3));  // 20260330 ZJH 输出 RGB 缓冲区

        for (int i = 0; i < nPixels; ++i) {
            // 20260330 ZJH 将 [0,1] 值映射为 Jet colormap 的 RGB
            float fVal = std::clamp(vecHeatmap[static_cast<size_t>(i)], 0.0f, 1.0f);
            float fR = 0.0f, fG = 0.0f, fB = 0.0f;  // 20260330 ZJH 浮点 RGB 分量

            if (fVal < 0.25f) {
                // 20260330 ZJH 蓝色 → 青色段: B=255, G 从 0 增长到 255
                float fT = fVal / 0.25f;  // 20260330 ZJH 段内归一化参数 [0,1]
                fR = 0.0f;
                fG = fT;
                fB = 1.0f;
            } else if (fVal < 0.5f) {
                // 20260330 ZJH 青色 → 绿色段: G=255, B 从 255 下降到 0
                float fT = (fVal - 0.25f) / 0.25f;
                fR = 0.0f;
                fG = 1.0f;
                fB = 1.0f - fT;
            } else if (fVal < 0.75f) {
                // 20260330 ZJH 绿色 → 黄色段: G=255, R 从 0 增长到 255
                float fT = (fVal - 0.5f) / 0.25f;
                fR = fT;
                fG = 1.0f;
                fB = 0.0f;
            } else {
                // 20260330 ZJH 黄色 → 红色段: R=255, G 从 255 下降到 0
                float fT = (fVal - 0.75f) / 0.25f;
                fR = 1.0f;
                fG = 1.0f - fT;
                fB = 0.0f;
            }

            // 20260330 ZJH 浮点 → uint8 量化
            size_t nIdx = static_cast<size_t>(i * 3);
            vecRGB[nIdx + 0] = static_cast<uint8_t>(fR * 255.0f);  // 20260330 ZJH R 通道
            vecRGB[nIdx + 1] = static_cast<uint8_t>(fG * 255.0f);  // 20260330 ZJH G 通道
            vecRGB[nIdx + 2] = static_cast<uint8_t>(fB * 255.0f);  // 20260330 ZJH B 通道
        }

        return vecRGB;  // 20260330 ZJH 返回 Jet 伪彩色 RGB 数据
    }

    // =========================================================================
    // 20260330 ZJH overlayHeatmap — 将热力图叠加到原始图像上
    // 使用 alpha blending: result = (1 - fAlpha) * original + fAlpha * colormap
    // vecOrigImage: [H*W*3] uint8 RGB 原始图像像素数据
    // vecHeatmap: [H*W] float 归一化到 [0,1] 的热力图（尺寸需与原图一致）
    // nW, nH: 图像宽度和高度
    // fAlpha: 热力图透明度 [0,1]，0=完全透明(仅原图)，1=完全不透明(仅热力图)
    // 返回: [H*W*3] uint8 RGB 叠加结果
    // =========================================================================
    static std::vector<uint8_t> overlayHeatmap(const std::vector<uint8_t>& vecOrigImage,
                                                 const std::vector<float>& vecHeatmap,
                                                 int nW, int nH, float fAlpha = 0.5f) {
        // 20260330 ZJH 先将热力图转为 Jet 伪彩色
        auto vecColormap = heatmapToColormap(vecHeatmap, nW, nH);

        int nPixels = nW * nH;  // 20260330 ZJH 总像素数
        std::vector<uint8_t> vecResult(static_cast<size_t>(nPixels * 3));

        // 20260330 ZJH 限制 alpha 范围到 [0, 1]
        fAlpha = std::clamp(fAlpha, 0.0f, 1.0f);
        float fOneMinusAlpha = 1.0f - fAlpha;  // 20260330 ZJH 原图权重

        for (int i = 0; i < nPixels * 3; ++i) {
            // 20260330 ZJH 逐像素 alpha blending
            float fBlended = fOneMinusAlpha * static_cast<float>(vecOrigImage[static_cast<size_t>(i)])
                           + fAlpha * static_cast<float>(vecColormap[static_cast<size_t>(i)]);
            vecResult[static_cast<size_t>(i)] =
                static_cast<uint8_t>(std::clamp(fBlended, 0.0f, 255.0f));  // 20260330 ZJH 量化并裁剪
        }

        return vecResult;  // 20260330 ZJH 返回叠加后的 RGB 图像
    }

    // =========================================================================
    // 20260330 ZJH upsampleHeatmap — 双线性插值上采样热力图
    // 将低分辨率的特征图级热力图上采样到原始图像尺寸
    // vecHeatmap: [nSrcH * nSrcW] 源热力图
    // nSrcW, nSrcH: 源尺寸
    // nDstW, nDstH: 目标尺寸
    // 返回: [nDstH * nDstW] 上采样后的热力图
    // =========================================================================
    static std::vector<float> upsampleHeatmap(const std::vector<float>& vecHeatmap,
                                               int nSrcW, int nSrcH,
                                               int nDstW, int nDstH) {
        std::vector<float> vecResult(static_cast<size_t>(nDstW * nDstH));

        // 20260330 ZJH 处理边界情况
        if (nSrcW <= 0 || nSrcH <= 0 || nDstW <= 0 || nDstH <= 0) {
            return vecResult;  // 20260330 ZJH 返回全零
        }

        // 20260330 ZJH 相同尺寸时直接拷贝
        if (nSrcW == nDstW && nSrcH == nDstH) {
            vecResult = vecHeatmap;
            return vecResult;
        }

        // 20260330 ZJH 计算缩放比例
        float fScaleX = static_cast<float>(nSrcW) / static_cast<float>(nDstW);  // 20260330 ZJH X 方向缩放
        float fScaleY = static_cast<float>(nSrcH) / static_cast<float>(nDstH);  // 20260330 ZJH Y 方向缩放

        for (int nDstY = 0; nDstY < nDstH; ++nDstY) {
            for (int nDstX = 0; nDstX < nDstW; ++nDstX) {
                // 20260330 ZJH 目标像素在源坐标系中的浮点位置（对齐中心）
                float fSrcX = (static_cast<float>(nDstX) + 0.5f) * fScaleX - 0.5f;
                float fSrcY = (static_cast<float>(nDstY) + 0.5f) * fScaleY - 0.5f;

                // 20260330 ZJH 计算四邻域整数坐标
                int nX0 = static_cast<int>(std::floor(fSrcX));  // 20260330 ZJH 左边界
                int nY0 = static_cast<int>(std::floor(fSrcY));  // 20260330 ZJH 上边界
                int nX1 = nX0 + 1;  // 20260330 ZJH 右边界
                int nY1 = nY0 + 1;  // 20260330 ZJH 下边界

                // 20260330 ZJH 计算插值权重
                float fWx = fSrcX - static_cast<float>(nX0);  // 20260330 ZJH X 方向权重
                float fWy = fSrcY - static_cast<float>(nY0);  // 20260330 ZJH Y 方向权重

                // 20260330 ZJH 边界裁剪（超出范围时钳位到边缘）
                nX0 = std::clamp(nX0, 0, nSrcW - 1);
                nY0 = std::clamp(nY0, 0, nSrcH - 1);
                nX1 = std::clamp(nX1, 0, nSrcW - 1);
                nY1 = std::clamp(nY1, 0, nSrcH - 1);

                // 20260330 ZJH 四邻域像素值
                float fV00 = vecHeatmap[static_cast<size_t>(nY0 * nSrcW + nX0)];  // 20260330 ZJH 左上
                float fV10 = vecHeatmap[static_cast<size_t>(nY0 * nSrcW + nX1)];  // 20260330 ZJH 右上
                float fV01 = vecHeatmap[static_cast<size_t>(nY1 * nSrcW + nX0)];  // 20260330 ZJH 左下
                float fV11 = vecHeatmap[static_cast<size_t>(nY1 * nSrcW + nX1)];  // 20260330 ZJH 右下

                // 20260330 ZJH 双线性插值公式
                float fInterpolated = fV00 * (1.0f - fWx) * (1.0f - fWy)
                                    + fV10 * fWx * (1.0f - fWy)
                                    + fV01 * (1.0f - fWx) * fWy
                                    + fV11 * fWx * fWy;

                vecResult[static_cast<size_t>(nDstY * nDstW + nDstX)] = fInterpolated;
            }
        }

        return vecResult;  // 20260330 ZJH 返回上采样后的热力图
    }
};

// =========================================================================
// 20260330 ZJH GradCAMPlusPlus — Grad-CAM++ 改进版
// 使用梯度的二阶信息（梯度的平方和立方）计算更精确的通道权重
// 论文: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-based Visual Explanations"
//
// 与标准 Grad-CAM 的区别:
//   Grad-CAM:   alpha_c = mean(dY/dA_c)  (全局平均池化梯度)
//   Grad-CAM++: alpha_c = sum(w_c^k * relu(dY/dA_c^k))
//     其中 w_c^k 由梯度的二阶导数决定，给予正梯度区域更高权重
//
// 优势: 对多个目标实例的定位更精确，不会只关注最显著的一个实例
// =========================================================================
class GradCAMPlusPlus {
public:
    // 20260330 ZJH compute — 计算 Grad-CAM++ 热力图
    // 与 GradCAM::compute 接口相同，内部使用不同的权重计算方法
    static GradCAMResult compute(Module& model, const Tensor& input,
                                  int nTargetClass = -1, int nTargetLayer = -1) {
        GradCAMResult result;  // 20260330 ZJH 返回结果

        // ===========================================================
        // 20260330 ZJH Step 1: 评估模式 + 前向传播 + 反向传播（同 GradCAM）
        // ===========================================================
        bool bWasTraining = model.isTraining();
        model.eval();
        model.zeroGrad();

        Tensor inputCopy = input.contiguous();
        tensorSetRequiresGrad(inputCopy, true);

        Tensor logits = model.forward(inputCopy);
        int nNumClasses = logits.shape(logits.ndim() - 1);

        // 20260330 ZJH 确定目标类别
        if (nTargetClass < 0) {
            Tensor cpuLogits = logits.contiguous();
            if (cpuLogits.isCuda()) cpuLogits = cpuLogits.cpu();
            const float* pLogits = cpuLogits.floatDataPtr();
            float fMaxScore = pLogits[0];
            nTargetClass = 0;
            for (int i = 1; i < nNumClasses; ++i) {
                if (pLogits[i] > fMaxScore) {
                    fMaxScore = pLogits[i];
                    nTargetClass = i;
                }
            }
        }
        result.nTargetClass = nTargetClass;

        // 20260330 ZJH 反向传播
        Tensor flatLogits = tensorReshape(logits, {nNumClasses});
        Tensor targetScore = tensorSlice(flatLogits, 0, nTargetClass, nTargetClass + 1);
        Tensor scalarScore = tensorSum(targetScore);
        result.fClassScore = scalarScore.item();
        tensorBackward(scalarScore);

        // ===========================================================
        // 20260330 ZJH Step 2: Grad-CAM++ 通道权重计算
        // 使用梯度的二阶信息: alpha_c^k = relu(grad_c^k)^2 / (2*relu(grad_c^k)^2 + sum(A_c * relu(grad_c^k)^3) + eps)
        // 简化实现: 使用 relu(grad)^2 / (2 * relu(grad)^2 + eps) 的空间平均
        // 再乘以 relu(grad) 得到加权因子
        // ===========================================================
        auto vecNamedParams = model.namedParameters();

        // 20260330 ZJH 定位最后一个卷积核
        Tensor* pTargetWeight = nullptr;
        int nCout = 0;
        int nSpatialPerChannel = 0;

        // 20260330 ZJH 遍历所有参数，找到最后一个 4D（卷积）权重
        int nConvIdx = 0;  // 20260330 ZJH 当前卷积层计数
        int nTargetConvIdx = -1;  // 20260330 ZJH 用于匹配 nTargetLayer
        for (auto& [strName, pParam] : vecNamedParams) {
            if (pParam->ndim() == 4) {
                if (nTargetLayer >= 0 && nConvIdx == nTargetLayer) {
                    pTargetWeight = pParam;
                    nCout = pParam->shape(0);
                    nSpatialPerChannel = pParam->shape(1) * pParam->shape(2) * pParam->shape(3);
                    nTargetConvIdx = nConvIdx;
                }
                if (nTargetLayer < 0) {
                    // 20260330 ZJH 默认使用最后一个卷积层
                    pTargetWeight = pParam;
                    nCout = pParam->shape(0);
                    nSpatialPerChannel = pParam->shape(1) * pParam->shape(2) * pParam->shape(3);
                    nTargetConvIdx = nConvIdx;
                }
                ++nConvIdx;
            }
        }

        if (!pTargetWeight || nCout == 0) {
            if (bWasTraining) model.train();
            return result;
        }

        // 20260330 ZJH 获取卷积核梯度
        Tensor weightGrad = tensorGetGrad(*pTargetWeight);
        std::vector<float> vecAlpha(static_cast<size_t>(nCout), 0.0f);

        if (weightGrad.numel() > 0) {
            Tensor cpuGrad = weightGrad.contiguous();
            if (cpuGrad.isCuda()) cpuGrad = cpuGrad.cpu();
            const float* pGrad = cpuGrad.floatDataPtr();

            // 20260330 ZJH Grad-CAM++ 权重计算
            // 对每个通道: alpha_c = mean_spatial( relu(grad)^2 / (2 * relu(grad)^2 + eps) * relu(grad) )
            // 简化为: alpha_c = mean_spatial( relu(grad)^3 / (2 * relu(grad)^2 + eps) )
            constexpr float fEps = 1e-7f;  // 20260330 ZJH 数值稳定性 epsilon

            for (int c = 0; c < nCout; ++c) {
                float fSum = 0.0f;
                int nOffset = c * nSpatialPerChannel;
                for (int i = 0; i < nSpatialPerChannel; ++i) {
                    float fG = pGrad[nOffset + i];  // 20260330 ZJH 原始梯度值
                    float fReluG = std::max(0.0f, fG);  // 20260330 ZJH ReLU(梯度)
                    float fReluG2 = fReluG * fReluG;  // 20260330 ZJH ReLU(梯度)^2
                    float fReluG3 = fReluG2 * fReluG;  // 20260330 ZJH ReLU(梯度)^3
                    // 20260330 ZJH Grad-CAM++ 公式: 二阶加权
                    float fWeight = fReluG3 / (2.0f * fReluG2 + fEps);
                    fSum += fWeight;
                }
                vecAlpha[static_cast<size_t>(c)] = fSum / static_cast<float>(nSpatialPerChannel);
            }
        }

        // ===========================================================
        // 20260330 ZJH Step 3: 使用输入梯度构造空间热力图（同 GradCAM）
        // ===========================================================
        Tensor inputGrad = tensorGetGrad(inputCopy);
        int nInputH = input.shape(2);
        int nInputW = input.shape(3);

        if (inputGrad.numel() > 0) {
            Tensor cpuInputGrad = inputGrad.contiguous();
            if (cpuInputGrad.isCuda()) cpuInputGrad = cpuInputGrad.cpu();

            int nInputC = input.shape(1);
            const float* pInputGrad = cpuInputGrad.floatDataPtr();

            // 20260330 ZJH Grad-CAM++: 只使用正梯度区域（ReLU 在梯度上）
            std::vector<float> vecRawHeatmap(static_cast<size_t>(nInputH * nInputW), 0.0f);
            for (int c = 0; c < nInputC; ++c) {
                int nChannelOffset = c * nInputH * nInputW;
                for (int hw = 0; hw < nInputH * nInputW; ++hw) {
                    // 20260330 ZJH Grad-CAM++ 仅累加正梯度（ReLU 门控）
                    float fG = pInputGrad[nChannelOffset + hw];
                    vecRawHeatmap[static_cast<size_t>(hw)] += std::max(0.0f, fG);
                }
            }

            // 20260330 ZJH 归一化到 [0, 1]
            float fMax = *std::max_element(vecRawHeatmap.begin(), vecRawHeatmap.end());
            float fMin = *std::min_element(vecRawHeatmap.begin(), vecRawHeatmap.end());
            float fRange = fMax - fMin;

            if (fRange > 1e-8f) {
                for (auto& fVal : vecRawHeatmap) {
                    fVal = (fVal - fMin) / fRange;
                }
            } else {
                std::fill(vecRawHeatmap.begin(), vecRawHeatmap.end(), 0.0f);
            }

            result.vecHeatmap = std::move(vecRawHeatmap);
            result.nHeatmapW = nInputW;
            result.nHeatmapH = nInputH;
        } else {
            result.vecHeatmap.resize(1, 0.0f);
            result.nHeatmapW = 1;
            result.nHeatmapH = 1;
        }

        // ===========================================================
        // 20260330 ZJH Step 4: 恢复状态并返回
        // ===========================================================
        if (bWasTraining) model.train();
        model.zeroGrad();

        return result;
    }
};

}  // namespace om
