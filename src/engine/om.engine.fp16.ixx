// 20260321 ZJH FP16 混合精度训练模块
// 提供 IEEE 754 半精度浮点转换工具、梯度缩放器（GradScaler）、混合精度配置
// 用于防止 FP16 训练中的梯度下溢，同时节省显存和加速训练
module;

#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "om_types.h"

export module om.engine.fp16;

// 20260321 ZJH 导入依赖模块：张量类、张量运算（含自动微分接口）、模块基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {

// =========================================================
// FP16Converter — IEEE 754 半精度浮点转换工具类
// =========================================================

// 20260321 ZJH FP16Converter — 提供 float32 与 float16 (uint16_t) 之间的双向转换
// IEEE 754 半精度格式：1 位符号 + 5 位指数 + 10 位尾数
// 表示范围约 +-65504，最小正规数约 6.1e-5，最小非规格化数约 5.96e-8
class FP16Converter {
public:
    // 20260321 ZJH floatToHalf — 将 IEEE 754 单精度浮点数转换为半精度表示
    // 参数: f - 输入的 32 位浮点数
    // 返回: 16 位半精度浮点数（以 uint16_t 存储）
    // 转换逻辑：提取符号/指数/尾数位域，映射到半精度的 1+5+10 格式
    static uint16_t floatToHalf(float f) {
        // 20260321 ZJH 通过 union 进行二进制位级访问，避免 strict aliasing 问题
        uint32_t nBits = 0;  // 20260321 ZJH 存放 float 的原始位模式
        // 20260321 ZJH 使用 memcpy 安全地将 float 位模式复制到 uint32_t
        std::memcpy(&nBits, &f, sizeof(float));

        // 20260321 ZJH 提取 IEEE 754 单精度各字段
        uint32_t nSign = (nBits >> 31) & 0x1;          // 20260321 ZJH 符号位：0=正，1=负
        uint32_t nExponent = (nBits >> 23) & 0xFF;     // 20260321 ZJH 指数字段（带偏移 127）
        uint32_t nMantissa = nBits & 0x7FFFFF;         // 20260321 ZJH 尾数字段（23 位）

        // 20260321 ZJH 半精度各字段初始化
        uint16_t nHalfSign = static_cast<uint16_t>(nSign << 15);  // 20260321 ZJH 符号位移到 bit15
        uint16_t nHalfExponent = 0;  // 20260321 ZJH 半精度指数字段
        uint16_t nHalfMantissa = 0;  // 20260321 ZJH 半精度尾数字段

        // 20260321 ZJH 计算半精度指数：fp32 偏移 127 转换为 fp16 偏移 15
        int nNewExp = static_cast<int>(nExponent) - 127 + 15;  // 20260321 ZJH 重新计算偏移后的指数

        if (nExponent == 0xFF) {
            // 20260321 ZJH 特殊值处理：Inf 或 NaN（fp32 指数全 1）
            nHalfExponent = 0x1F;  // 20260321 ZJH 半精度指数全 1 表示特殊值
            if (nMantissa != 0) {
                // 20260321 ZJH NaN：保留尾数最高位，确保半精度 NaN 不退化为 Inf
                nHalfMantissa = static_cast<uint16_t>(nMantissa >> 13);  // 20260321 ZJH 截断尾数
                if (nHalfMantissa == 0) {
                    nHalfMantissa = 1;  // 20260321 ZJH 确保 NaN 尾数不为零
                }
            }
            // 20260321 ZJH 若 nMantissa == 0，则为 Inf，nHalfMantissa 保持 0
        } else if (nExponent == 0) {
            // 20260321 ZJH fp32 非规格化数（极小值），在 fp16 中直接映射为零
            // fp32 非规格化数范围远小于 fp16 最小非规格化数
            nHalfExponent = 0;  // 20260321 ZJH 指数为 0
            nHalfMantissa = 0;  // 20260321 ZJH 尾数为 0，结果为 +-0
        } else if (nNewExp >= 0x1F) {
            // 20260321 ZJH 上溢：超出半精度表示范围（>= 65504），映射为 Inf
            nHalfExponent = 0x1F;  // 20260321 ZJH 指数全 1
            nHalfMantissa = 0;     // 20260321 ZJH 尾数为 0 表示 Inf
        } else if (nNewExp <= 0) {
            // 20260321 ZJH 下溢区域：fp32 值太小，需要映射为 fp16 非规格化数或零
            if (nNewExp < -10) {
                // 20260321 ZJH 完全下溢：移位量超过尾数位宽，结果为零
                nHalfExponent = 0;  // 20260321 ZJH 指数为 0
                nHalfMantissa = 0;  // 20260321 ZJH 尾数为 0
            } else {
                // 20260321 ZJH 部分下溢：转换为半精度非规格化数
                // 将隐含的 1 放回尾数，然后右移对应位数
                uint32_t nFullMantissa = nMantissa | 0x800000;  // 20260321 ZJH 恢复隐含的整数位 1
                int nShift = 14 - nNewExp;  // 20260321 ZJH 计算右移位数（14 = 1 + 13，1 为隐含位）
                nHalfMantissa = static_cast<uint16_t>(nFullMantissa >> nShift);  // 20260321 ZJH 右移得到非规格化尾数
                nHalfExponent = 0;  // 20260321 ZJH 非规格化数指数为 0
            }
        } else {
            // 20260321 ZJH 正常范围：直接映射指数和尾数
            nHalfExponent = static_cast<uint16_t>(nNewExp);  // 20260321 ZJH 映射后的指数
            nHalfMantissa = static_cast<uint16_t>(nMantissa >> 13);  // 20260321 ZJH 截断尾数从 23 位到 10 位
        }

        // 20260321 ZJH 组装半精度位模式：符号(1) + 指数(5) + 尾数(10)
        uint16_t nResult = nHalfSign
                         | static_cast<uint16_t>(nHalfExponent << 10)
                         | nHalfMantissa;
        return nResult;  // 20260321 ZJH 返回组装好的 16 位半精度值
    }

    // 20260321 ZJH halfToFloat — 将 IEEE 754 半精度浮点数转换为单精度浮点数
    // 参数: h - 输入的 16 位半精度浮点数（uint16_t）
    // 返回: 对应的 32 位浮点数
    static float halfToFloat(uint16_t h) {
        // 20260321 ZJH 提取半精度各字段
        uint32_t nSign = (h >> 15) & 0x1;          // 20260321 ZJH 符号位
        uint32_t nExponent = (h >> 10) & 0x1F;     // 20260321 ZJH 指数字段（5 位，偏移 15）
        uint32_t nMantissa = h & 0x3FF;             // 20260321 ZJH 尾数字段（10 位）

        // 20260321 ZJH 单精度各字段初始化
        uint32_t nFloatSign = nSign << 31;          // 20260321 ZJH 符号位移到 bit31
        uint32_t nFloatExponent = 0;                // 20260321 ZJH 单精度指数字段
        uint32_t nFloatMantissa = 0;                // 20260321 ZJH 单精度尾数字段

        if (nExponent == 0x1F) {
            // 20260321 ZJH 特殊值：Inf 或 NaN（半精度指数全 1）
            nFloatExponent = 0xFF;  // 20260321 ZJH 单精度指数全 1
            if (nMantissa != 0) {
                // 20260321 ZJH NaN：将 10 位尾数左移到 23 位，保留 NaN 特征
                nFloatMantissa = nMantissa << 13;  // 20260321 ZJH 尾数扩展
            }
            // 20260321 ZJH 若 nMantissa == 0，则为 Inf，nFloatMantissa 保持 0
        } else if (nExponent == 0) {
            // 20260321 ZJH 零或非规格化数
            if (nMantissa == 0) {
                // 20260321 ZJH 正零或负零：指数和尾数均为 0
                nFloatExponent = 0;   // 20260321 ZJH 指数为 0
                nFloatMantissa = 0;   // 20260321 ZJH 尾数为 0
            } else {
                // 20260321 ZJH 非规格化数：需要归一化
                // 半精度非规格化数的值为 (-1)^s * 2^(-14) * (0.mantissa)
                // 转换为单精度规格化数
                int nNewExp = -14;  // 20260321 ZJH 非规格化数的有效指数为 -14（1 - 偏移15）
                uint32_t nShiftedMantissa = nMantissa;  // 20260321 ZJH 临时变量用于归一化

                // 20260321 ZJH 归一化循环：左移尾数直到最高位为 1
                while ((nShiftedMantissa & 0x400) == 0) {
                    nShiftedMantissa <<= 1;  // 20260321 ZJH 左移一位
                    nNewExp--;               // 20260321 ZJH 指数减 1 补偿
                }
                // 20260321 ZJH 移除隐含的整数位 1（归一化后最高位为 1，fp32 格式中隐含）
                nShiftedMantissa &= 0x3FF;  // 20260321 ZJH 清除 bit10
                // 20260321 ZJH 计算单精度指数：有效指数 + fp32 偏移 127
                nFloatExponent = static_cast<uint32_t>(nNewExp + 127);  // 20260321 ZJH 加偏移
                nFloatMantissa = nShiftedMantissa << 13;  // 20260321 ZJH 尾数扩展到 23 位
            }
        } else {
            // 20260321 ZJH 正常范围：直接映射
            // 指数转换：fp16 偏移 15 -> fp32 偏移 127，差值 112
            nFloatExponent = nExponent + 112;  // 20260321 ZJH nExponent - 15 + 127 = nExponent + 112
            nFloatMantissa = nMantissa << 13;  // 20260321 ZJH 尾数从 10 位扩展到 23 位
        }

        // 20260321 ZJH 组装单精度位模式：符号(1) + 指数(8) + 尾数(23)
        uint32_t nResultBits = nFloatSign
                             | (nFloatExponent << 23)
                             | nFloatMantissa;

        // 20260321 ZJH 通过 memcpy 将位模式安全地转换回 float
        float fResult = 0.0f;  // 20260321 ZJH 结果浮点数
        std::memcpy(&fResult, &nResultBits, sizeof(float));
        return fResult;  // 20260321 ZJH 返回转换后的单精度浮点数
    }

    // 20260321 ZJH convertToHalf — 批量将 float 数组转换为 half (uint16_t) 数组
    // 参数: pSrc - 源 float 数组指针
    //       pDst - 目标 uint16_t 数组指针（调用方负责分配足够空间）
    //       nCount - 需要转换的元素数量
    static void convertToHalf(const float* pSrc, uint16_t* pDst, int nCount) {
        // 20260321 ZJH 逐元素调用 floatToHalf 进行转换
        for (int i = 0; i < nCount; ++i) {
            pDst[i] = floatToHalf(pSrc[i]);  // 20260321 ZJH 逐个转换并写入目标
        }
    }

    // 20260321 ZJH convertToFloat — 批量将 half (uint16_t) 数组转换为 float 数组
    // 参数: pSrc - 源 uint16_t 数组指针
    //       pDst - 目标 float 数组指针（调用方负责分配足够空间）
    //       nCount - 需要转换的元素数量
    static void convertToFloat(const uint16_t* pSrc, float* pDst, int nCount) {
        // 20260321 ZJH 逐元素调用 halfToFloat 进行转换
        for (int i = 0; i < nCount; ++i) {
            pDst[i] = halfToFloat(pSrc[i]);  // 20260321 ZJH 逐个转换并写入目标
        }
    }
};

// =========================================================
// MixedPrecisionConfig — 混合精度训练配置结构体
// =========================================================

// 20260321 ZJH MixedPrecisionConfig — 控制混合精度训练行为的配置
// 包含启用开关、损失缩放参数、动态缩放策略
struct MixedPrecisionConfig {
    bool bEnabled = false;           // 20260321 ZJH 是否启用混合精度训练，默认关闭
    float fLossScale = 65536.0f;     // 20260321 ZJH 初始损失缩放因子，65536 = 2^16，常用初始值
    bool bDynamicScaling = true;     // 20260321 ZJH 是否启用动态缩放（根据梯度健康状况自动调整）
    int nScaleWindowSize = 2000;     // 20260321 ZJH 缩放调整窗口大小（连续无溢出步数达到此值后增大 scale）
};

// =========================================================
// 辅助函数 — 张量级别的 FP16 工具
// =========================================================

// 20260321 ZJH hasInfOrNan — 检查张量中是否存在 inf 或 nan 值
// 参数: t - 待检查的张量
// 返回: true 表示存在 inf 或 nan，false 表示所有值均为有限数
// 用途: GradScaler 在 step() 中调用此函数判断梯度是否健康
bool hasInfOrNan(const Tensor& t) {
    // 20260321 ZJH 空张量视为无异常值
    if (t.numel() == 0) {
        return false;  // 20260321 ZJH 空张量无元素，不可能有 inf/nan
    }

    // 20260407 ZJH [修复] GPU 张量必须先 D2H 拷贝到 CPU 再检查
    // 旧: 直接用 floatDataPtr() 获取指针 → GPU 张量返回设备指针 → CPU 解引用崩溃
    auto ct = t.contiguous();
    Tensor cpuTensor = ct.isCuda() ? ct.cpu() : ct;  // 20260407 ZJH GPU→CPU 传输
    const float* pData = cpuTensor.floatDataPtr();  // 20260407 ZJH 现在是 CPU 指针，安全读取
    int nTotal = cpuTensor.numel();

    // 20260321 ZJH 逐元素检查是否为 inf 或 nan
    for (int i = 0; i < nTotal; ++i) {
        if (std::isinf(pData[i]) || std::isnan(pData[i])) {
            return true;  // 20260321 ZJH 发现异常值，立即返回
        }
    }
    return false;  // 20260321 ZJH 所有元素均为有限数
}

// 20260321 ZJH tensorCastToHalf — 将 FP32 张量的值裁剪到 FP16 可表示范围
// 参数: input - 输入的 FP32 张量
// 返回: 值被 clip 到 [-65504, 65504] 范围内的新张量
// 说明: 并非真正转换数据类型（仍为 float32 存储），而是确保值在 FP16 范围内
//       65504 是 FP16 可表示的最大有限值（0x7BFF）
Tensor tensorCastToHalf(const Tensor& input) {
    // 20260321 ZJH 空张量直接返回
    if (input.numel() == 0) {
        return input;  // 20260321 ZJH 空张量无需处理
    }

    // 20260321 ZJH FP16 最大有限值常量
    constexpr float fHalfMax = 65504.0f;   // 20260321 ZJH FP16 最大正数
    constexpr float fHalfMin = -65504.0f;  // 20260321 ZJH FP16 最小负数（最大绝对值）

    // 20260321 ZJH 获取连续内存版本，线性遍历输入数据
    auto ct = input.contiguous();  // 20260321 ZJH 确保输入连续
    const float* pSrc = ct.floatDataPtr();  // 20260321 ZJH 源数据只读指针
    int nTotal = ct.numel();  // 20260321 ZJH 元素总数

    // 20260321 ZJH 创建与输入同形状的输出张量
    Tensor result = Tensor::zeros(ct.shapeVec());  // 20260321 ZJH 分配输出存储
    float* pDst = result.mutableFloatDataPtr();  // 20260321 ZJH 输出可写指针

    // 20260321 ZJH 逐元素裁剪到 FP16 范围
    for (int i = 0; i < nTotal; ++i) {
        float fVal = pSrc[i];  // 20260321 ZJH 读取源值
        // 20260321 ZJH 处理 NaN：NaN 保持为 NaN（std::clamp 对 NaN 行为未定义，需特殊处理）
        if (std::isnan(fVal)) {
            pDst[i] = fVal;  // 20260321 ZJH NaN 直接透传，不裁剪
        } else {
            // 20260321 ZJH 将有限值裁剪到 [-65504, 65504]
            pDst[i] = std::clamp(fVal, fHalfMin, fHalfMax);  // 20260321 ZJH 标准库裁剪函数
        }
    }

    return result;  // 20260321 ZJH 返回裁剪后的张量
}

// =========================================================
// GradScaler — 梯度缩放器，防止 FP16 训练中的梯度下溢
// =========================================================

// 20260321 ZJH GradScaler — 动态损失缩放器，核心思路：
// 1. 前向传播后将 loss 乘以 scale 因子（放大梯度，防止小梯度下溢为零）
// 2. 反向传播后将梯度除以 scale 因子（恢复正确的梯度值）
// 3. 检查梯度是否包含 inf/nan：
//    - 若有，说明 scale 过大导致溢出，跳过本次更新并缩小 scale
//    - 若无，正常更新参数，并在连续成功步数达到阈值后增大 scale
// 此机制在 NVIDIA Apex / PyTorch AMP 中广泛使用
class GradScaler {
public:
    // 20260321 ZJH 构造函数 — 初始化缩放器各参数
    // 参数: fInitScale - 初始缩放因子，默认 65536.0（2^16），足够放大大多数梯度
    //       fGrowFactor - 增长因子，连续无溢出步数达标后 scale 乘以此值，默认 2.0
    //       fShrinkFactor - 缩小因子，检测到溢出时 scale 乘以此值，默认 0.5
    //       nGrowInterval - 增长间隔，连续无溢出步数达到此值后增大 scale，默认 2000
    GradScaler(float fInitScale = 65536.0f,
               float fGrowFactor = 2.0f,
               float fShrinkFactor = 0.5f,
               int nGrowInterval = 2000)
        : m_fScale(fInitScale)              // 20260321 ZJH 初始缩放因子
        , m_fGrowFactor(fGrowFactor)        // 20260321 ZJH 增长倍率
        , m_fShrinkFactor(fShrinkFactor)    // 20260321 ZJH 缩小倍率
        , m_nGrowInterval(nGrowInterval)    // 20260321 ZJH 增长间隔步数
        , m_nStepsSinceGrow(0)              // 20260321 ZJH 自上次增长以来的连续成功步数，初始为 0
        , m_bFoundInf(false)                // 20260321 ZJH 当前步是否发现 inf/nan，初始未发现
    {
        // 20260321 ZJH 构造完成，scaler 已就绪
    }

    // 20260321 ZJH scale — 缩放损失值，用于前向传播后、反向传播前
    // 将 loss 乘以当前 scale 因子，使得反向传播产生的梯度被同比例放大
    // 参数: loss - 前向传播计算得到的损失张量（通常为标量，numel==1）
    // 返回: 缩放后的损失张量 (loss * m_fScale)
    Tensor scale(const Tensor& loss) {
        // 20260321 ZJH 使用 tensorMulScalar 将损失乘以缩放因子
        // tensorMulScalar 会处理自动微分（创建 MulScalarBackward 节点）
        Tensor scaledLoss = tensorMulScalar(
            const_cast<Tensor&>(loss),  // 20260321 ZJH tensorMulScalar 接受非 const 引用
            m_fScale                     // 20260321 ZJH 当前缩放因子
        );
        return scaledLoss;  // 20260321 ZJH 返回 loss * scale
    }

    // 20260321 ZJH unscaleGrads — 反缩放梯度，将所有参数的梯度除以当前 scale 因子
    // 必须在反向传播（backward）之后、step() 之前调用
    // 参数: vecParams - 模型参数指针向量，每个参数的梯度将被除以 scale
    void unscaleGrads(std::vector<Tensor*>& vecParams) {
        // 20260321 ZJH 计算反缩放系数：1.0 / scale
        float fInvScale = 1.0f / m_fScale;  // 20260321 ZJH 反缩放因子

        // 20260321 ZJH 重置 inf 标记，准备重新检测
        m_bFoundInf = false;  // 20260321 ZJH 每次 unscale 前重置

        // 20260321 ZJH 遍历所有参数，反缩放其梯度
        for (size_t i = 0; i < vecParams.size(); ++i) {
            // 20260321 ZJH 获取当前参数的梯度
            Tensor grad = tensorGetGrad(*vecParams[i]);  // 20260321 ZJH 从 GradAccumulator 获取梯度

            // 20260321 ZJH 若无梯度则跳过该参数
            if (grad.numel() == 0) {
                continue;  // 20260321 ZJH 参数未参与计算，无梯度
            }

            // 20260321 ZJH 检查梯度是否包含 inf 或 nan
            if (hasInfOrNan(grad)) {
                m_bFoundInf = true;  // 20260321 ZJH 标记发现溢出
                // 20260321 ZJH 发现溢出后仍继续处理剩余参数（但 step() 会跳过更新）
            }

            // 20260407 ZJH [修复] 梯度反缩放由调用方（EngineBridge）通过 GradAccumulator 执行
            // fp16 模块没有导入 autograd，无法直接访问 GradAccumulator
            // unscaleGrads 只负责 inf 检测，反缩放在 EngineBridge 侧用 tensorMulScalar 完成
            (void)fInvScale;  // 20260407 ZJH inf 检测已完成，反缩放由调用方执行
        }
    }

    // 20260321 ZJH step — 检查梯度健康状况并决定是否进行参数更新
    // 必须在 unscaleGrads() 之后调用
    // 返回: true 表示梯度正常，调用方应执行 optimizer.step()
    //       false 表示梯度包含 inf/nan，调用方应跳过本次 optimizer.step()
    // 注意: 此函数不直接调用 optimizer，而是返回布尔值由调用方决定
    bool step() {
        if (m_bFoundInf) {
            // 20260321 ZJH 检测到 inf/nan：缩小 scale 因子，防止后续步骤再次溢出
            m_fScale *= m_fShrinkFactor;  // 20260321 ZJH scale 缩小（默认减半）

            // 20260321 ZJH 防止 scale 缩小到零或极小值，设定下限
            constexpr float fMinScale = 1.0f;  // 20260321 ZJH 最小允许 scale
            if (m_fScale < fMinScale) {
                m_fScale = fMinScale;  // 20260321 ZJH 钳制到最小值
            }

            // 20260321 ZJH 重置连续成功步数计数器
            m_nStepsSinceGrow = 0;  // 20260321 ZJH 溢出后重新计数

            return false;  // 20260321 ZJH 告知调用方跳过参数更新
        }

        // 20260321 ZJH 梯度正常：增加连续成功步数计数
        m_nStepsSinceGrow++;  // 20260321 ZJH 成功步数 +1

        return true;  // 20260321 ZJH 告知调用方可以执行参数更新
    }

    // 20260321 ZJH update — 更新 scale 因子
    // 在 optimizer.step()（如果执行了的话）之后调用
    // 检查连续成功步数是否达到增长间隔，若是则增大 scale
    void update() {
        if (m_bFoundInf) {
            // 20260321 ZJH 本步已溢出，scale 已在 step() 中缩小，此处不再调整
            m_bFoundInf = false;  // 20260321 ZJH 重置溢出标记，准备下一步
            return;  // 20260321 ZJH 直接返回
        }

        // 20260321 ZJH 检查是否达到增长条件
        if (m_nStepsSinceGrow >= m_nGrowInterval) {
            // 20260321 ZJH 连续无溢出步数达到阈值，增大 scale
            m_fScale *= m_fGrowFactor;  // 20260321 ZJH scale 增大（默认翻倍）

            // 20260321 ZJH 防止 scale 增长到极大值导致上溢
            constexpr float fMaxScale = 2.0f * 65536.0f * 65536.0f;  // 20260321 ZJH 上限约 8.59e9
            if (m_fScale > fMaxScale) {
                m_fScale = fMaxScale;  // 20260321 ZJH 钳制到最大值
            }

            // 20260321 ZJH 重置计数器，开始新的增长周期
            m_nStepsSinceGrow = 0;  // 20260321 ZJH 重新计数
        }

        // 20260321 ZJH 重置溢出标记
        m_bFoundInf = false;  // 20260321 ZJH 清除标记，准备下一步
    }

    // =========================================================
    // 状态查询接口
    // =========================================================

    // 20260321 ZJH getScale — 获取当前缩放因子
    // 返回: 当前的 scale 值
    float getScale() const {
        return m_fScale;  // 20260321 ZJH 返回当前缩放因子
    }

    // 20260321 ZJH isFoundInf — 查询本步是否发现 inf/nan
    // 返回: true 表示本步 unscaleGrads 中检测到异常值
    bool isFoundInf() const {
        return m_bFoundInf;  // 20260321 ZJH 返回溢出标记
    }

    // 20260321 ZJH getGrowInterval — 获取增长间隔步数
    // 返回: 连续无溢出步数达到此值后增大 scale
    int getGrowInterval() const {
        return m_nGrowInterval;  // 20260321 ZJH 返回增长间隔
    }

    // 20260321 ZJH getStepsSinceGrow — 获取自上次增长以来的连续成功步数
    // 返回: 当前连续无溢出步数
    int getStepsSinceGrow() const {
        return m_nStepsSinceGrow;  // 20260321 ZJH 返回连续成功步数
    }

private:
    float m_fScale;            // 20260321 ZJH 当前缩放因子，初始值由构造函数设定
    float m_fGrowFactor;       // 20260321 ZJH 增长因子：scale *= growFactor
    float m_fShrinkFactor;     // 20260321 ZJH 缩小因子：scale *= shrinkFactor
    int m_nGrowInterval;       // 20260321 ZJH 增长间隔：连续无溢出步数达标后增大 scale
    int m_nStepsSinceGrow;     // 20260321 ZJH 自上次 scale 增长以来的连续成功步数
    bool m_bFoundInf;          // 20260321 ZJH 当前步是否在 unscaleGrads 中发现 inf/nan
};

}  // namespace om
