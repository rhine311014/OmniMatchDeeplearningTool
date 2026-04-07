// 20260330 ZJH 模型剪枝与量化模块
// 提供非结构化剪枝（Magnitude Pruning）、结构化通道剪枝（Structured Channel Pruning）、
// 稀疏度分析、FP16/INT8 量化、反量化推理
// 参考: HikRobot prune_ratio 0.1~0.8 + INT8/FP16 精度选择设计
// 用途: 将训练好的模型压缩后部署到边缘设备（如工控机、嵌入式 GPU）
module;

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cstring>

export module om.engine.pruning;

// 20260330 ZJH 导入张量和模块基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {

// =========================================================
// PruneStrategy — 剪枝策略枚举
// =========================================================

// 20260330 ZJH PruneStrategy — 定义两种主流剪枝方式
// MagnitudeBased: 非结构化剪枝，按权重绝对值大小将最小的置零（产生稀疏矩阵）
// StructuredChannel: 结构化通道剪枝，按 L1 范数剪除整个输出通道（减少实际计算量）
enum class PruneStrategy {
    MagnitudeBased,     // 20260330 ZJH L1 范数最小的权重逐个置零
    StructuredChannel   // 20260330 ZJH 按输出通道 L1 范数排序，剪除整个通道
};

// =========================================================
// QuantizeMode — 量化精度枚举
// =========================================================

// 20260330 ZJH QuantizeMode — 参考 HikRobot 的 FP32/FP16/INT8 精度选择
enum class QuantizeMode {
    FP32,    // 20260330 ZJH 原始精度（无量化）
    FP16,    // 20260330 ZJH 半精度（16位浮点，适合 GPU 推理加速）
    INT8     // 20260330 ZJH 8位整数量化（适合 CPU/NPU 推理加速）
};

// =========================================================
// PruneConfig — 剪枝配置
// =========================================================

// 20260330 ZJH PruneConfig — 剪枝超参数配置
// 参考 HikRobot prune_ratio 0.1~0.8 范围
struct PruneConfig {
    float fPruneRatio = 0.3f;           // 20260330 ZJH 剪枝比例 [0.1, 0.8]，默认 30%
    PruneStrategy eStrategy = PruneStrategy::MagnitudeBased;  // 20260330 ZJH 默认非结构化
    bool bFinetuneAfterPrune = true;    // 20260330 ZJH 剪枝后是否微调恢复精度
    int nFinetuneEpochs = 10;           // 20260330 ZJH 微调 epoch 数

    // 20260330 ZJH validate — 校验配置合法性
    bool validate() const {
        if (fPruneRatio < 0.1f || fPruneRatio > 0.8f) return false;  // 20260330 ZJH 比例范围
        if (nFinetuneEpochs < 0) return false;  // 20260330 ZJH epoch 不能为负
        return true;
    }
};

// =========================================================
// SparsityInfo — 模型稀疏度统计信息
// =========================================================

// 20260330 ZJH SparsityInfo — 分析模型参数的稀疏程度
struct SparsityInfo {
    int nTotalParams;       // 20260330 ZJH 参数总数（所有层的权重元素数之和）
    int nZeroParams;        // 20260330 ZJH 零值参数数量
    float fSparsityRatio;   // 20260330 ZJH 稀疏率 = nZeroParams / nTotalParams
    int nTotalBytes;        // 20260330 ZJH 原始模型大小（字节，FP32 × 参数数）
    int nEffectiveBytes;    // 20260330 ZJH 非零参数占用（字节）
    int nNumLayers;         // 20260330 ZJH 参与统计的层数
};

// =========================================================
// QuantizedLayer — 单层量化结果
// =========================================================

// 20260330 ZJH QuantizedLayer — 保存单层量化后的权重和量化参数
struct QuantizedLayer {
    std::vector<int8_t> vecQuantizedWeights;   // 20260330 ZJH INT8 量化权重
    std::vector<uint16_t> vecFP16Weights;      // 20260330 ZJH FP16 半精度权重（以 uint16 存储）
    float fScale;                               // 20260330 ZJH 缩放因子（INT8: max(|w|)/127）
    int nZeroPoint;                             // 20260330 ZJH 零点偏移（对称量化时为 0）
    std::vector<int> vecShape;                  // 20260330 ZJH 原始权重形状
    int nNumElements;                           // 20260330 ZJH 元素总数
};

// =========================================================
// QuantizedModel — 量化后的完整模型
// =========================================================

// 20260330 ZJH QuantizedModel — 保存整个模型量化后的结果
struct QuantizedModel {
    std::vector<QuantizedLayer> vecLayers;      // 20260330 ZJH 每层量化结果
    std::vector<std::string> vecLayerNames;     // 20260330 ZJH 层名称列表
    QuantizeMode eMode;                         // 20260330 ZJH 量化模式
    int nTotalOriginalBytes;                    // 20260330 ZJH 原始模型大小（字节）
    int nTotalQuantizedBytes;                   // 20260330 ZJH 量化后模型大小（字节）
    float fCompressionRatio;                    // 20260330 ZJH 压缩比 = 原始/量化
};

// =========================================================
// analyzeSparsity — 统计模型稀疏度
// =========================================================

// 20260330 ZJH analyzeSparsity — 遍历模型所有参数，统计零值权重占比
// 用于: 剪枝前后对比、评估模型压缩效果
// model: 要分析的模型（通过 Module 基类接口访问参数）
// 返回: SparsityInfo 统计结构
inline SparsityInfo analyzeSparsity(Module& model) {
    SparsityInfo info;
    info.nTotalParams = 0;     // 20260330 ZJH 初始化计数器
    info.nZeroParams = 0;
    info.nNumLayers = 0;

    // 20260330 ZJH 获取模型所有参数（递归收集）
    auto vecParams = model.parameters();

    // 20260330 ZJH 逐参数统计
    for (auto* pParam : vecParams) {
        if (!pParam) continue;  // 20260330 ZJH 跳过空指针（防御性）

        // 20260330 ZJH 确保参数在 CPU 上以便读取数据
        Tensor cpuParam = pParam->contiguous();
        int nNumel = cpuParam.numel();  // 20260330 ZJH 当前参数的元素数
        if (nNumel <= 0) continue;      // 20260330 ZJH 跳过空参数

        const float* pData = cpuParam.floatDataPtr();  // 20260330 ZJH 获取数据指针
        info.nTotalParams += nNumel;  // 20260330 ZJH 累加总参数数
        info.nNumLayers++;             // 20260330 ZJH 累加层数

        // 20260330 ZJH 逐元素检查是否为零
        for (int i = 0; i < nNumel; ++i) {
            // 20260330 ZJH 用绝对值阈值判断零（浮点精度问题）
            if (std::fabs(pData[i]) < 1e-8f) {
                info.nZeroParams++;  // 20260330 ZJH 累加零值参数数
            }
        }
    }

    // 20260330 ZJH 计算派生指标
    if (info.nTotalParams > 0) {
        info.fSparsityRatio = static_cast<float>(info.nZeroParams) /
                              static_cast<float>(info.nTotalParams);
    } else {
        info.fSparsityRatio = 0.0f;
    }

    // 20260330 ZJH 计算字节数（FP32 每个参数 4 字节）
    info.nTotalBytes = info.nTotalParams * 4;
    info.nEffectiveBytes = (info.nTotalParams - info.nZeroParams) * 4;

    return info;  // 20260330 ZJH 返回稀疏度统计
}

// =========================================================
// pruneModelMagnitude — 非结构化幅度剪枝（内部实现）
// =========================================================

// 20260330 ZJH pruneModelMagnitude — 将绝对值最小的权重置零
// 工作原理:
//   1. 收集所有权重的绝对值
//   2. 按绝对值排序，找到剪枝阈值（第 pruneRatio × total 小的值）
//   3. 将绝对值低于阈值的权重置零
// model: 要剪枝的模型
// fPruneRatio: 剪枝比例 [0.1, 0.8]
// 返回: 实际被置零的参数数量
inline int pruneModelMagnitude(Module& model, float fPruneRatio) {
    // 20260330 ZJH 获取所有参数
    auto vecParams = model.parameters();

    // 20260330 ZJH 第一步: 收集所有权重的绝对值
    std::vector<float> vecAllAbsValues;  // 20260330 ZJH 存放所有权重绝对值
    int nTotalParams = 0;
    for (auto* pParam : vecParams) {
        if (!pParam) continue;
        Tensor cpuParam = pParam->contiguous();
        int nNumel = cpuParam.numel();
        if (nNumel <= 0) continue;

        const float* pData = cpuParam.floatDataPtr();
        nTotalParams += nNumel;
        for (int i = 0; i < nNumel; ++i) {
            vecAllAbsValues.push_back(std::fabs(pData[i]));
        }
    }

    if (vecAllAbsValues.empty()) {
        return 0;  // 20260330 ZJH 无参数可剪枝
    }

    // 20260330 ZJH 第二步: 排序找阈值
    // 使用 nth_element 而非完全排序，O(n) 复杂度
    int nPruneCount = static_cast<int>(
        std::floor(static_cast<float>(nTotalParams) * fPruneRatio));
    nPruneCount = std::max(0, std::min(nPruneCount, nTotalParams - 1));

    // 20260330 ZJH nth_element 将第 nPruneCount 小的元素放到正确位置
    std::nth_element(vecAllAbsValues.begin(),
                     vecAllAbsValues.begin() + nPruneCount,
                     vecAllAbsValues.end());
    float fThreshold = vecAllAbsValues[nPruneCount];  // 20260330 ZJH 剪枝阈值

    // 20260330 ZJH 第三步: 将低于阈值的权重置零
    int nPrunedCount = 0;  // 20260330 ZJH 实际剪枝计数
    for (auto* pParam : vecParams) {
        if (!pParam) continue;
        int nNumel = pParam->numel();
        if (nNumel <= 0) continue;

        float* pData = pParam->mutableFloatDataPtr();  // 20260330 ZJH 获取可写指针
        for (int i = 0; i < nNumel; ++i) {
            if (std::fabs(pData[i]) <= fThreshold) {
                pData[i] = 0.0f;    // 20260330 ZJH 将小权重置零
                nPrunedCount++;      // 20260330 ZJH 计数
            }
        }
    }

    return nPrunedCount;  // 20260330 ZJH 返回实际被剪枝的参数数量
}

// =========================================================
// pruneModelStructured — 结构化通道剪枝（内部实现）
// =========================================================

// 20260330 ZJH pruneModelStructured — 按输出通道 L1 范数剪枝
// 工作原理:
//   1. 对每个卷积/线性层，计算每个输出通道的 L1 范数
//   2. 将 L1 范数最小的通道整行置零
//   3. 整个通道置零后，后续层可以跳过对应计算（结构化稀疏）
// model: 要剪枝的模型
// fPruneRatio: 剪枝比例
// 返回: 实际被置零的参数数量
inline int pruneModelStructured(Module& model, float fPruneRatio) {
    auto vecParams = model.parameters();
    int nTotalPruned = 0;  // 20260330 ZJH 总剪枝参数数

    for (auto* pParam : vecParams) {
        if (!pParam) continue;

        auto vecShape = pParam->shapeVec();  // 20260330 ZJH 获取参数形状

        // 20260330 ZJH 只对 2D 以上的参数做通道剪枝（排除 1D bias）
        // 典型形状: Conv [OutC, InC, KH, KW], Linear [Out, In]
        if (vecShape.size() < 2) continue;

        int nOutChannels = vecShape[0];  // 20260330 ZJH 输出通道数（第 0 维）
        if (nOutChannels <= 1) continue;  // 20260330 ZJH 只有 1 个通道无法剪枝

        // 20260330 ZJH 计算每个输出通道内的元素数
        int nChannelSize = 1;
        for (size_t d = 1; d < vecShape.size(); ++d) {
            nChannelSize *= vecShape[d];
        }

        // 20260330 ZJH 计算每个输出通道的 L1 范数
        float* pData = pParam->mutableFloatDataPtr();
        std::vector<std::pair<float, int>> vecChannelNorms;  // 20260330 ZJH {L1范数, 通道索引}
        vecChannelNorms.reserve(nOutChannels);

        for (int c = 0; c < nOutChannels; ++c) {
            float fNorm = 0.0f;
            int nOffset = c * nChannelSize;  // 20260330 ZJH 通道起始偏移
            for (int i = 0; i < nChannelSize; ++i) {
                fNorm += std::fabs(pData[nOffset + i]);  // 20260330 ZJH 累加绝对值
            }
            vecChannelNorms.push_back({fNorm, c});
        }

        // 20260330 ZJH 按 L1 范数升序排列（最不重要的在前）
        std::sort(vecChannelNorms.begin(), vecChannelNorms.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;  // 20260330 ZJH 小范数在前
                  });

        // 20260330 ZJH 计算要剪除的通道数
        int nPruneChannels = static_cast<int>(
            std::floor(static_cast<float>(nOutChannels) * fPruneRatio));
        nPruneChannels = std::max(0, std::min(nPruneChannels, nOutChannels - 1));

        // 20260330 ZJH 将被剪除通道的所有权重置零
        for (int k = 0; k < nPruneChannels; ++k) {
            int nChanIdx = vecChannelNorms[k].second;  // 20260330 ZJH 被剪除的通道索引
            int nOffset = nChanIdx * nChannelSize;
            for (int i = 0; i < nChannelSize; ++i) {
                pData[nOffset + i] = 0.0f;  // 20260330 ZJH 整个通道置零
            }
            nTotalPruned += nChannelSize;  // 20260330 ZJH 累加剪枝参数数
        }
    }

    return nTotalPruned;  // 20260330 ZJH 返回总剪枝参数数
}

// =========================================================
// pruneModel — 统一剪枝入口
// =========================================================

// 20260330 ZJH pruneModel — 根据策略选择剪枝方式并执行
// model: 要剪枝的模型
// fPruneRatio: 剪枝比例 [0.1, 0.8]
// eStrategy: 剪枝策略（非结构化 / 结构化通道）
// 返回: 实际被剪枝（置零）的参数数量
inline int pruneModel(Module& model, float fPruneRatio,
                       PruneStrategy eStrategy = PruneStrategy::MagnitudeBased) {
    // 20260330 ZJH 校验剪枝比例范围
    if (fPruneRatio < 0.1f || fPruneRatio > 0.8f) {
        throw std::invalid_argument(
            "pruneModel: fPruneRatio must be in [0.1, 0.8], got " +
            std::to_string(fPruneRatio));
    }

    // 20260330 ZJH 根据策略分发
    switch (eStrategy) {
        case PruneStrategy::MagnitudeBased:
            return pruneModelMagnitude(model, fPruneRatio);
        case PruneStrategy::StructuredChannel:
            return pruneModelStructured(model, fPruneRatio);
        default:
            return pruneModelMagnitude(model, fPruneRatio);  // 20260330 ZJH 默认走非结构化
    }
}

// =========================================================
// FP16 量化辅助函数
// =========================================================

// 20260330 ZJH floatToHalf — 将 FP32 转 FP16（IEEE 754 半精度）
// 使用简化截断方案（非完整的非规格化处理，工业部署够用）
inline uint16_t floatToHalfSimple(float fValue) {
    uint32_t nBits = 0;
    std::memcpy(&nBits, &fValue, sizeof(float));  // 20260330 ZJH 位模式提取

    uint32_t nSign = (nBits >> 31) & 0x1;          // 20260330 ZJH 符号位
    int nExponent = static_cast<int>((nBits >> 23) & 0xFF) - 127;  // 20260330 ZJH 无偏指数
    uint32_t nMantissa = nBits & 0x7FFFFF;          // 20260330 ZJH 尾数（23位）

    uint16_t nHalf = static_cast<uint16_t>(nSign << 15);  // 20260330 ZJH 半精度符号位

    if (nExponent > 15) {
        // 20260330 ZJH 上溢→Inf
        nHalf |= 0x7C00;
    } else if (nExponent < -14) {
        // 20260330 ZJH 下溢→零
        // 简化处理，不做非规格化映射
    } else {
        // 20260330 ZJH 正常范围：指数加偏移15，尾数截断到10位
        uint16_t nHalfExp = static_cast<uint16_t>((nExponent + 15) << 10);
        uint16_t nHalfMan = static_cast<uint16_t>(nMantissa >> 13);
        nHalf |= nHalfExp | nHalfMan;
    }

    return nHalf;  // 20260330 ZJH 返回 FP16 位模式
}

// 20260330 ZJH halfToFloat — 将 FP16 转回 FP32
inline float halfToFloatSimple(uint16_t nHalf) {
    uint32_t nSign = (nHalf >> 15) & 0x1;                   // 20260330 ZJH 符号位
    uint32_t nExponent = (nHalf >> 10) & 0x1F;              // 20260330 ZJH 指数（5位）
    uint32_t nMantissa = nHalf & 0x3FF;                      // 20260330 ZJH 尾数（10位）

    uint32_t nBits = 0;
    if (nExponent == 0x1F) {
        // 20260330 ZJH 特殊值: Inf 或 NaN
        nBits = (nSign << 31) | (0xFF << 23) | (nMantissa << 13);
    } else if (nExponent == 0) {
        // 20260330 ZJH 零或非规格化数→简化为零
        nBits = (nSign << 31);
    } else {
        // 20260330 ZJH 正常范围：指数从偏移15转到偏移127，尾数扩展到23位
        uint32_t nFP32Exp = static_cast<uint32_t>(nExponent - 15 + 127);
        nBits = (nSign << 31) | (nFP32Exp << 23) | (nMantissa << 13);
    }

    float fResult = 0.0f;
    std::memcpy(&fResult, &nBits, sizeof(float));  // 20260330 ZJH 位模式还原为 float
    return fResult;
}

// =========================================================
// quantizeModel — 量化整个模型
// =========================================================

// 20260330 ZJH quantizeModel — 将模型权重量化为指定精度
// FP32→FP16: 逐元素转换为 IEEE 754 半精度
// FP32→INT8: 对称量化，scale = max(|w|) / 127，zeroPoint = 0
// model: 要量化的模型
// eMode: 目标精度
// 返回: QuantizedModel 结构（包含量化权重和元信息）
inline QuantizedModel quantizeModel(Module& model, QuantizeMode eMode) {
    QuantizedModel qmodel;
    qmodel.eMode = eMode;  // 20260330 ZJH 记录量化模式
    qmodel.nTotalOriginalBytes = 0;
    qmodel.nTotalQuantizedBytes = 0;

    // 20260330 ZJH FP32 模式: 不做量化，直接返回空模型
    if (eMode == QuantizeMode::FP32) {
        qmodel.fCompressionRatio = 1.0f;
        return qmodel;
    }

    // 20260330 ZJH 获取命名参数（带名称方便调试和反序列化）
    auto vecNamedParams = model.namedParameters();

    for (auto& [strName, pParam] : vecNamedParams) {
        if (!pParam) continue;

        Tensor cpuParam = pParam->contiguous();  // 20260330 ZJH 确保连续布局
        int nNumel = cpuParam.numel();
        if (nNumel <= 0) continue;

        const float* pData = cpuParam.floatDataPtr();  // 20260330 ZJH FP32 原始数据

        QuantizedLayer layer;
        layer.vecShape = cpuParam.shapeVec();  // 20260330 ZJH 保存原始形状
        layer.nNumElements = nNumel;

        // 20260330 ZJH 累加原始字节数
        qmodel.nTotalOriginalBytes += nNumel * 4;  // 20260330 ZJH FP32 = 4 bytes

        if (eMode == QuantizeMode::FP16) {
            // 20260330 ZJH FP16 量化: 逐元素 float→half
            layer.vecFP16Weights.resize(nNumel);
            for (int i = 0; i < nNumel; ++i) {
                layer.vecFP16Weights[i] = floatToHalfSimple(pData[i]);
            }
            layer.fScale = 1.0f;     // 20260330 ZJH FP16 不需要 scale
            layer.nZeroPoint = 0;     // 20260330 ZJH FP16 不需要 zero point
            qmodel.nTotalQuantizedBytes += nNumel * 2;  // 20260330 ZJH FP16 = 2 bytes
        } else {
            // 20260330 ZJH INT8 对称量化
            // step 1: 找到最大绝对值
            float fMaxAbs = 0.0f;
            for (int i = 0; i < nNumel; ++i) {
                float fAbs = std::fabs(pData[i]);
                if (fAbs > fMaxAbs) fMaxAbs = fAbs;
            }

            // step 2: 计算 scale = maxAbs / 127
            // 20260330 ZJH 防除零: 全零权重层 scale 设为 1
            layer.fScale = (fMaxAbs > 1e-10f) ? (fMaxAbs / 127.0f) : 1.0f;
            layer.nZeroPoint = 0;  // 20260330 ZJH 对称量化零点为 0

            // step 3: 逐元素量化 q = round(w / scale), clamp to [-128, 127]
            layer.vecQuantizedWeights.resize(nNumel);
            for (int i = 0; i < nNumel; ++i) {
                float fQuantized = std::round(pData[i] / layer.fScale);
                // 20260330 ZJH 钳位到 INT8 范围
                fQuantized = std::max(-128.0f, std::min(127.0f, fQuantized));
                layer.vecQuantizedWeights[i] = static_cast<int8_t>(fQuantized);
            }
            qmodel.nTotalQuantizedBytes += nNumel * 1;  // 20260330 ZJH INT8 = 1 byte
        }

        qmodel.vecLayerNames.push_back(strName);  // 20260330 ZJH 保存层名
        qmodel.vecLayers.push_back(std::move(layer));
    }

    // 20260330 ZJH 计算压缩比
    if (qmodel.nTotalQuantizedBytes > 0) {
        qmodel.fCompressionRatio = static_cast<float>(qmodel.nTotalOriginalBytes) /
                                   static_cast<float>(qmodel.nTotalQuantizedBytes);
    } else {
        qmodel.fCompressionRatio = 1.0f;
    }

    return qmodel;  // 20260330 ZJH 返回量化模型
}

// =========================================================
// dequantizeLayer — 反量化单层权重（INT8/FP16 → FP32）
// =========================================================

// 20260330 ZJH dequantizeLayer — 将量化权重恢复为 FP32 Tensor
// 用于: 反量化推理时临时恢复权重精度
// layer: 量化后的层数据
// eMode: 量化模式
// 返回: 恢复后的 FP32 Tensor
inline Tensor dequantizeLayer(const QuantizedLayer& layer, QuantizeMode eMode) {
    // 20260330 ZJH 分配 FP32 数据缓冲
    std::vector<float> vecDequantized(layer.nNumElements, 0.0f);

    if (eMode == QuantizeMode::FP16) {
        // 20260330 ZJH FP16→FP32: 逐元素 half→float
        for (int i = 0; i < layer.nNumElements; ++i) {
            vecDequantized[i] = halfToFloatSimple(layer.vecFP16Weights[i]);
        }
    } else if (eMode == QuantizeMode::INT8) {
        // 20260330 ZJH INT8→FP32: dequant = q * scale + zero_point * scale
        // 对称量化 zero_point=0, 所以 dequant = q * scale
        for (int i = 0; i < layer.nNumElements; ++i) {
            vecDequantized[i] = static_cast<float>(layer.vecQuantizedWeights[i]) * layer.fScale;
        }
    }

    // 20260330 ZJH 包装为 Tensor 返回
    return Tensor::fromData(vecDequantized.data(), layer.vecShape);
}

// =========================================================
// dequantizeAndForward — 反量化推理
// =========================================================

// 20260330 ZJH dequantizeAndForward — 将量化模型反量化后执行前向推理
// 工作流程:
//   1. 逐层将量化权重恢复为 FP32
//   2. 写回模型参数
//   3. 执行模型前向传播
//   4. （可选）恢复原始量化权重
// 注意: 此方法临时修改模型权重，非线程安全
// qmodel: 量化模型数据
// model: 原始模型（权重会被临时覆盖）
// input: 输入张量
// 返回: 推理输出张量
inline Tensor dequantizeAndForward(const QuantizedModel& qmodel,
                                    Module& model,
                                    const Tensor& input) {
    // 20260330 ZJH FP32 模式不需要反量化
    if (qmodel.eMode == QuantizeMode::FP32) {
        return model.forward(input);
    }

    // 20260330 ZJH 获取模型命名参数
    auto vecNamedParams = model.namedParameters();

    // 20260330 ZJH 逐层反量化并写回
    for (size_t i = 0; i < qmodel.vecLayers.size(); ++i) {
        // 20260330 ZJH 在命名参数中查找对应层
        const std::string& strLayerName = qmodel.vecLayerNames[i];
        for (auto& [strName, pParam] : vecNamedParams) {
            if (strName == strLayerName && pParam) {
                // 20260330 ZJH 反量化得到 FP32 Tensor
                Tensor dequantized = dequantizeLayer(qmodel.vecLayers[i], qmodel.eMode);
                // 20260330 ZJH 复制反量化数据到模型参数
                int nNumel = dequantized.numel();
                if (nNumel == pParam->numel()) {
                    const float* pSrc = dequantized.floatDataPtr();
                    float* pDst = pParam->mutableFloatDataPtr();
                    std::memcpy(pDst, pSrc, sizeof(float) * nNumel);
                }
                break;  // 20260330 ZJH 找到即跳出内层循环
            }
        }
    }

    // 20260330 ZJH 执行前向推理
    return model.forward(input);
}

// =========================================================
// PruneAndQuantizePipeline — 剪枝+量化一体化流水线
// =========================================================

// 20260330 ZJH PruneAndQuantizePipeline — 封装"剪枝→分析→量化"的完整工作流
// 使用方法:
//   1. 构造时传入剪枝配置和量化模式
//   2. 调用 execute() 对模型执行剪枝+量化
//   3. 通过 sparsityInfo()/quantizedModel() 获取结果
class PruneAndQuantizePipeline {
public:
    // 20260330 ZJH 构造函数 — 保存配置
    PruneAndQuantizePipeline(const PruneConfig& pruneConfig, QuantizeMode eQuantizeMode)
        : m_pruneConfig(pruneConfig), m_eQuantizeMode(eQuantizeMode) {}

    // 20260330 ZJH execute — 执行完整的剪枝+量化流水线
    // model: 目标模型（权重会被原地修改）
    // 返回: true=成功，false=参数校验失败
    bool execute(Module& model) {
        // 20260330 ZJH 校验剪枝配置
        if (!m_pruneConfig.validate()) {
            return false;
        }

        // 20260330 ZJH 步骤 1: 剪枝前稀疏度分析（基线）
        m_sparsityBefore = analyzeSparsity(model);

        // 20260330 ZJH 步骤 2: 执行剪枝
        m_nPrunedCount = pruneModel(model, m_pruneConfig.fPruneRatio,
                                     m_pruneConfig.eStrategy);

        // 20260330 ZJH 步骤 3: 剪枝后稀疏度分析
        m_sparsityAfter = analyzeSparsity(model);

        // 20260330 ZJH 步骤 4: 量化（如果不是 FP32）
        m_quantizedModel = quantizeModel(model, m_eQuantizeMode);

        m_bExecuted = true;  // 20260406 ZJH 标记已执行
        return true;  // 20260406 ZJH 返回成功
    }

    // 20260330 ZJH sparsityBefore — 剪枝前稀疏度
    const SparsityInfo& sparsityBefore() const { return m_sparsityBefore; }

    // 20260330 ZJH sparsityAfter — 剪枝后稀疏度
    const SparsityInfo& sparsityAfter() const { return m_sparsityAfter; }

    // 20260330 ZJH prunedCount — 被剪枝的参数数量
    int prunedCount() const { return m_nPrunedCount; }

    // 20260330 ZJH quantizedModel — 量化后的模型
    const QuantizedModel& quantizedModel() const { return m_quantizedModel; }

    // 20260330 ZJH isExecuted — 是否已执行
    bool isExecuted() const { return m_bExecuted; }

private:
    PruneConfig m_pruneConfig;              // 20260330 ZJH 剪枝配置
    QuantizeMode m_eQuantizeMode;           // 20260330 ZJH 量化模式
    SparsityInfo m_sparsityBefore;          // 20260330 ZJH 剪枝前稀疏度
    SparsityInfo m_sparsityAfter;           // 20260330 ZJH 剪枝后稀疏度
    int m_nPrunedCount = 0;                 // 20260330 ZJH 剪枝参数数
    QuantizedModel m_quantizedModel;        // 20260330 ZJH 量化结果
    bool m_bExecuted = false;               // 20260330 ZJH 执行标记
};

// =========================================================
// 20260402 ZJH [OPT-3.4] Conv+BN 融合 — 推理优化（RepVGG 核心技巧）
// 将训练时的 Conv + BatchNorm 合并为单个 Conv（零额外推理开销）
// 数学推导:
//   BN 输出: y = gamma * (conv_out - running_mean) / sqrt(running_var + eps) + beta
//   等价合并: y = (gamma / sqrt(var+eps)) * conv_out + (beta - gamma*mean/sqrt(var+eps))
//   即: new_weight = weight * gamma/sqrt(var+eps)
//       new_bias   = bias*gamma/sqrt(var+eps) + beta - gamma*mean/sqrt(var+eps)
// 效果: 推理时消除 BN 层的 mean/var/gamma/beta 计算，减少 ~20-30% 算子数
// 对标: PyTorch torch.nn.utils.fuse_conv_bn_eval(), TensorRT 自动融合
// =========================================================

// 20260402 ZJH fuseConvBN — 将 Conv 权重与 BN 参数融合（推理优化）
// convWeight: [Cout, Cin, KH, KW] 卷积核权重
// convBias: [Cout] 卷积偏置（可为空张量，此时视为全零）
// bnGamma: [Cout] BN gamma（缩放参数）
// bnBeta: [Cout] BN beta（偏移参数）
// bnMean: [Cout] BN running_mean
// bnVar: [Cout] BN running_var
// fEps: BN epsilon（数值稳定性常数）
// 返回: {fusedWeight, fusedBias} 融合后的卷积参数
inline std::pair<Tensor, Tensor> fuseConvBN(
    const Tensor& convWeight, const Tensor& convBias,
    const Tensor& bnGamma, const Tensor& bnBeta,
    const Tensor& bnMean, const Tensor& bnVar,
    float fEps = 1e-5f)
{
    int nCout = convWeight.shape(0);  // 20260402 ZJH 输出通道数
    int nWeightsPerChannel = convWeight.numel() / nCout;  // 20260402 ZJH 每通道权重数 = Cin*KH*KW

    // 20260402 ZJH 确保所有输入在 CPU 上且连续
    auto cW = convWeight.contiguous();
    auto cGamma = bnGamma.contiguous();
    auto cBeta = bnBeta.contiguous();
    auto cMean = bnMean.contiguous();
    auto cVar = bnVar.contiguous();

    const float* pW = cW.floatDataPtr();      // 20260402 ZJH Conv 权重指针
    const float* pGamma = cGamma.floatDataPtr();  // 20260402 ZJH BN gamma 指针
    const float* pBeta = cBeta.floatDataPtr();    // 20260402 ZJH BN beta 指针
    const float* pMean = cMean.floatDataPtr();    // 20260402 ZJH BN mean 指针
    const float* pVar = cVar.floatDataPtr();      // 20260402 ZJH BN var 指针

    // 20260402 ZJH 分配融合后的权重和偏置
    Tensor tFusedW = Tensor::zeros(convWeight.shapeVec());  // 20260402 ZJH [Cout, Cin, KH, KW]
    Tensor tFusedB = Tensor::zeros({nCout});                 // 20260402 ZJH [Cout]
    float* pFW = tFusedW.mutableFloatDataPtr();   // 20260402 ZJH 融合权重写入指针
    float* pFB = tFusedB.mutableFloatDataPtr();   // 20260402 ZJH 融合偏置写入指针

    // 20260402 ZJH 获取原始卷积偏置（无偏置时视为全零）
    bool bHasBias = (convBias.numel() > 0);
    const float* pBias = bHasBias ? convBias.contiguous().floatDataPtr() : nullptr;

    // 20260402 ZJH 逐通道融合
    for (int c = 0; c < nCout; ++c) {
        // 20260402 ZJH scale = gamma[c] / sqrt(var[c] + eps)
        float fScale = pGamma[c] / std::sqrt(pVar[c] + fEps);

        // 20260402 ZJH 融合权重: new_weight[c] = weight[c] * scale
        for (int j = 0; j < nWeightsPerChannel; ++j) {
            pFW[c * nWeightsPerChannel + j] = pW[c * nWeightsPerChannel + j] * fScale;
        }

        // 20260402 ZJH 融合偏置: new_bias[c] = (bias[c]) * scale + beta[c] - mean[c] * scale
        float fOrigBias = bHasBias ? pBias[c] : 0.0f;  // 20260402 ZJH 原始偏置
        pFB[c] = fOrigBias * fScale + pBeta[c] - pMean[c] * fScale;
    }

    return {tFusedW, tFusedB};  // 20260402 ZJH 返回融合后参数
}

// 20260402 ZJH fuseAllConvBN — 遍历模型，自动融合所有连续的 Conv+BN 对
// model: 待优化的模型（eval 模式下调用）
// 返回: 融合的 Conv+BN 对数
inline int fuseAllConvBN(Module& model) {
    // 20260402 ZJH 获取所有命名参数
    auto vecNamed = model.namedParameters();
    int nFused = 0;  // 20260402 ZJH 融合计数

    // 20260402 ZJH 扫描参数名，匹配 "xxx.weight" 和 "xxx_bn.gamma" 模式
    // 匹配规则: 如果存在 paramName="conv1.weight" 和 "bn1.gamma"
    // 则尝试融合 conv1 和 bn1
    // 注: 当前简化实现，完整版需要遍历模型子模块树
    for (auto& [strName, pParam] : vecNamed) {
        // 20260402 ZJH 跳过非权重参数
        if (strName.find(".weight") == std::string::npos) continue;
        // 20260402 ZJH 检查是否有 4D 形状（Conv 权重 [Cout,Cin,KH,KW]）
        if (pParam->shapeVec().size() != 4) continue;

        // 20260402 ZJH 提取基础名（去掉 ".weight" 后缀）
        std::string strBase = strName.substr(0, strName.find(".weight"));

        // 20260402 ZJH 查找对应的 BN 参数
        // 命名规则假设: conv1 → bn1, 或 conv1 → conv1_bn
        // 这里简化: 不做名称匹配，由 EngineBridge 在训练完成后显式调用
        // fuseConvBN 函数已提供，具体融合逻辑在 EngineBridge 的 BN 折叠代码中
        // 参见: EngineBridge.cpp 第 2741 行 "训练完成后自动将 BatchNorm 参数合并到前置 Conv"
        (void)strBase;  // 20260402 ZJH 避免未使用警告
    }

    return nFused;
}

}  // namespace om
