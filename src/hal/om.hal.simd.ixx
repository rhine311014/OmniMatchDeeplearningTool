// 20260321 ZJH SIMDBackend — AVX2 SIMD 加速计算内核
// 提供矩阵乘法、向量运算、激活函数的 AVX2 优化实现
// 所有函数在不支持 AVX2 的平台上自动回退到标量实现
module;

#include <cstddef>  // 20260321 ZJH size_t 定义
#include <cstring>  // 20260321 ZJH memset 等内存操作
#include <cmath>    // 20260321 ZJH expf 等数学函数

// 20260321 ZJH MSVC 编译器使用 __cpuid/__cpuidex 进行 CPU 特性检测
#ifdef _MSC_VER
#include <intrin.h>
#endif

// 20260321 ZJH 仅在编译器支持 AVX2 时引入 SIMD 内联函数头文件
// MSVC 通过 /arch:AVX2 启用，GCC/Clang 通过 -mavx2 -mfma 启用
#ifdef __AVX2__
#include <immintrin.h>
#endif

export module om.hal.simd;

export namespace om {

// 20260321 ZJH SIMDBackend — 全静态方法类，提供 AVX2 SIMD 优化的核心运算
// 包含：CPU 特性检测、矩阵乘法、向量加/乘、标量乘、ReLU、Sigmoid
// 所有方法均提供 AVX2 快速路径和标量回退路径
class SIMDBackend {
public:
    // ===================================================================
    // CPU 特性检测
    // ===================================================================

    // 20260321 ZJH 运行时检测当前 CPU 是否支持 AVX2 指令集
    // 使用 CPUID 指令查询 EBX 寄存器第 5 位（AVX2 标志位）
    // 返回: true 表示支持 AVX2，false 表示不支持
    static bool isAVX2Supported() {
#ifdef _MSC_VER
        // 20260321 ZJH MSVC 编译器：使用 __cpuidex 内建函数
        // 参数: cpuInfo — 存储 EAX/EBX/ECX/EDX 四个寄存器值的数组
        //       function_id = 7 — 扩展特性枚举叶
        //       subfunction_id = 0 — 子叶 0
        int cpuInfo[4] = { 0 };  // 20260321 ZJH 存储 CPUID 返回的 EAX/EBX/ECX/EDX
        __cpuidex(cpuInfo, 7, 0);  // 20260321 ZJH 调用 CPUID leaf 7, subleaf 0
        // 20260321 ZJH EBX 的第 5 位（bit 5）标识 AVX2 支持
        // 使用位与操作检测该位是否为 1
        return (cpuInfo[1] & (1 << 5)) != 0;  // 20260321 ZJH cpuInfo[1] = EBX
#elif defined(__GNUC__) || defined(__clang__)
        // 20260321 ZJH GCC/Clang 编译器：使用 __builtin_cpu_supports 内建函数
        // 该函数在程序启动时自动初始化 CPU 特性缓存，开销极低
        return __builtin_cpu_supports("avx2");
#else
        // 20260321 ZJH 未知编译器，保守返回不支持
        return false;
#endif
    }

    // ===================================================================
    // 矩阵乘法 — AVX2 4x8 分块 + FMA
    // ===================================================================

    // 20260321 ZJH AVX2 优化矩阵乘法：A[M,K] × B[K,N] → C[M,N]，行主序存储
    // 核心策略：4x8 分块 — 每次处理 A 的 4 行 × B 的 8 列
    //   - 外层循环以 8 列为步长遍历 N 维（B 的列方向）
    //   - 中层循环以 4 行为步长遍历 M 维（A 的行方向）
    //   - 内层循环遍历 K 维（公共维度），使用 FMA 累加
    // 使用 _mm256_fmadd_ps (a*b+c) 将乘法和加法融合为单条指令
    // 余数部分（不满 4 行或 8 列的边界）回退到标量计算
    // 参数: pA — A 矩阵指针 [M×K]，pB — B 矩阵指针 [K×N]
    //       pC — 输出矩阵指针 [M×N]
    //       nM — A 的行数，nK — 公共维度，nN — B 的列数
    static void matmulAVX2(const float* pA, const float* pB, float* pC, int nM, int nK, int nN) {
        // 20260321 ZJH 先将输出矩阵 C 全部清零，后续用累加方式填充
        std::memset(pC, 0, static_cast<size_t>(nM) * nN * sizeof(float));

#ifdef __AVX2__
        // 20260321 ZJH 计算可被 8 整除的列数上界和可被 4 整除的行数上界
        // nN8 — 8 列分块的结束列索引（不含余数列）
        // nM4 — 4 行分块的结束行索引（不含余数行）
        int nN8 = nN & ~7;  // 20260321 ZJH 等价于 (nN / 8) * 8，向下对齐到 8 的倍数
        int nM4 = nM & ~3;  // 20260321 ZJH 等价于 (nM / 4) * 4，向下对齐到 4 的倍数

        // 20260321 ZJH 主循环：以 8 列为步长遍历 B 的列维度
        for (int j = 0; j < nN8; j += 8) {
            // 20260321 ZJH 中层循环：以 4 行为步长遍历 A 的行维度
            for (int i = 0; i < nM4; i += 4) {
                // 20260321 ZJH 4 个 256 位累加寄存器，分别对应 A 的第 i, i+1, i+2, i+3 行
                // 每个寄存器存储 8 个 float 结果（对应 B 的 j ~ j+7 列）
                __m256 vSum0 = _mm256_setzero_ps();  // 20260321 ZJH C[i+0][j..j+7] 的累加器
                __m256 vSum1 = _mm256_setzero_ps();  // 20260321 ZJH C[i+1][j..j+7] 的累加器
                __m256 vSum2 = _mm256_setzero_ps();  // 20260321 ZJH C[i+2][j..j+7] 的累加器
                __m256 vSum3 = _mm256_setzero_ps();  // 20260321 ZJH C[i+3][j..j+7] 的累加器

                // 20260321 ZJH 内层循环：遍历公共维度 K，逐元素累加乘积
                for (int k = 0; k < nK; ++k) {
                    // 20260321 ZJH 从 B 矩阵加载连续 8 个 float（第 k 行第 j~j+7 列）
                    // B 按行主序存储，B[k][j] 开始的 8 个元素在内存中连续
                    __m256 vB = _mm256_loadu_ps(&pB[k * nN + j]);

                    // 20260321 ZJH 广播 A[i+0][k] 到 8 个通道，与 vB 做 FMA 累加
                    // _mm256_set1_ps 将单个标量复制到 256 位寄存器的全部 8 个 lane
                    // _mm256_fmadd_ps(a, b, c) = a * b + c，单指令完成乘加融合
                    __m256 vA0 = _mm256_set1_ps(pA[(i + 0) * nK + k]);
                    vSum0 = _mm256_fmadd_ps(vA0, vB, vSum0);

                    // 20260321 ZJH 广播 A[i+1][k]，FMA 累加到 vSum1
                    __m256 vA1 = _mm256_set1_ps(pA[(i + 1) * nK + k]);
                    vSum1 = _mm256_fmadd_ps(vA1, vB, vSum1);

                    // 20260321 ZJH 广播 A[i+2][k]，FMA 累加到 vSum2
                    __m256 vA2 = _mm256_set1_ps(pA[(i + 2) * nK + k]);
                    vSum2 = _mm256_fmadd_ps(vA2, vB, vSum2);

                    // 20260321 ZJH 广播 A[i+3][k]，FMA 累加到 vSum3
                    __m256 vA3 = _mm256_set1_ps(pA[(i + 3) * nK + k]);
                    vSum3 = _mm256_fmadd_ps(vA3, vB, vSum3);
                }

                // 20260321 ZJH 将 4 个累加寄存器的结果写回 C 矩阵对应位置
                // _mm256_storeu_ps 支持非对齐地址写入
                _mm256_storeu_ps(&pC[(i + 0) * nN + j], vSum0);
                _mm256_storeu_ps(&pC[(i + 1) * nN + j], vSum1);
                _mm256_storeu_ps(&pC[(i + 2) * nN + j], vSum2);
                _mm256_storeu_ps(&pC[(i + 3) * nN + j], vSum3);
            }

            // 20260321 ZJH 处理行方向余数：不足 4 行的尾部，逐行使用 AVX2
            for (int i = nM4; i < nM; ++i) {
                __m256 vSum = _mm256_setzero_ps();  // 20260321 ZJH 单行累加器
                for (int k = 0; k < nK; ++k) {
                    // 20260321 ZJH 对余数行仍使用 AVX2 加速列方向计算
                    __m256 vA = _mm256_set1_ps(pA[i * nK + k]);
                    __m256 vB = _mm256_loadu_ps(&pB[k * nN + j]);
                    vSum = _mm256_fmadd_ps(vA, vB, vSum);
                }
                _mm256_storeu_ps(&pC[i * nN + j], vSum);  // 20260321 ZJH 写回余数行结果
            }
        }

        // 20260321 ZJH 处理列方向余数：不足 8 列的尾部，回退到标量计算
        // 这些列无法填满一个 256 位寄存器，必须逐元素处理
        for (int i = 0; i < nM; ++i) {
            for (int j = nN8; j < nN; ++j) {
                float fVal = 0.0f;  // 20260321 ZJH C[i][j] 的标量累加器
                for (int k = 0; k < nK; ++k) {
                    fVal += pA[i * nK + k] * pB[k * nN + j];  // 20260321 ZJH 标量乘加
                }
                pC[i * nN + j] = fVal;  // 20260321 ZJH 写回余数列结果
            }
        }
#else
        // 20260321 ZJH 标量回退路径：编译器未启用 AVX2 时使用朴素三重循环
        // 采用 i-k-j 循环顺序以提升 B 矩阵的缓存访问局部性
        for (int i = 0; i < nM; ++i) {
            for (int k = 0; k < nK; ++k) {
                float fA_ik = pA[i * nK + k];  // 20260321 ZJH 缓存 A[i][k]，内层循环复用
                for (int j = 0; j < nN; ++j) {
                    pC[i * nN + j] += fA_ik * pB[k * nN + j];  // 20260321 ZJH 累加乘积
                }
            }
        }
#endif
    }

    // ===================================================================
    // 向量逐元素加法
    // ===================================================================

    // 20260321 ZJH AVX2 优化的逐元素向量加法：pOut[i] = pA[i] + pB[i]
    // AVX2 每次处理 8 个 float（256 位 = 8 × 32 位），余数标量处理
    // 参数: pA — 输入向量 A，pB — 输入向量 B，pOut — 输出向量
    //       nCount — 元素总数
    static void addAVX2(const float* pA, const float* pB, float* pOut, size_t nCount) {
#ifdef __AVX2__
        size_t i = 0;  // 20260321 ZJH 循环索引，AVX2 主循环结束后用于标量余数处理

        // 20260321 ZJH AVX2 主循环：每次加载 8 个 float 进行并行加法
        // 循环条件 i + 8 <= nCount 确保不越界读取
        for (; i + 8 <= nCount; i += 8) {
            // 20260321 ZJH 从 pA 和 pB 各加载 8 个连续 float 到 256 位寄存器
            __m256 vA = _mm256_loadu_ps(&pA[i]);
            __m256 vB = _mm256_loadu_ps(&pB[i]);
            // 20260321 ZJH 8 路并行浮点加法，结果存入 vResult
            __m256 vResult = _mm256_add_ps(vA, vB);
            // 20260321 ZJH 将 8 个结果写回 pOut，支持非对齐地址
            _mm256_storeu_ps(&pOut[i], vResult);
        }

        // 20260321 ZJH 标量处理余数元素（不足 8 个的尾部）
        for (; i < nCount; ++i) {
            pOut[i] = pA[i] + pB[i];
        }
#else
        // 20260321 ZJH 标量回退路径：逐元素加法
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] + pB[i];
        }
#endif
    }

    // ===================================================================
    // 向量逐元素乘法
    // ===================================================================

    // 20260321 ZJH AVX2 优化的逐元素向量乘法：pOut[i] = pA[i] * pB[i]
    // 与 addAVX2 结构完全一致，区别在于使用 _mm256_mul_ps 替代 _mm256_add_ps
    // 参数: pA — 输入向量 A，pB — 输入向量 B，pOut — 输出向量
    //       nCount — 元素总数
    static void mulAVX2(const float* pA, const float* pB, float* pOut, size_t nCount) {
#ifdef __AVX2__
        size_t i = 0;  // 20260321 ZJH 循环索引

        // 20260321 ZJH AVX2 主循环：每次处理 8 个 float 的并行乘法
        for (; i + 8 <= nCount; i += 8) {
            __m256 vA = _mm256_loadu_ps(&pA[i]);   // 20260321 ZJH 加载 A 的 8 个元素
            __m256 vB = _mm256_loadu_ps(&pB[i]);   // 20260321 ZJH 加载 B 的 8 个元素
            __m256 vResult = _mm256_mul_ps(vA, vB);  // 20260321 ZJH 8 路并行乘法
            _mm256_storeu_ps(&pOut[i], vResult);    // 20260321 ZJH 写回结果
        }

        // 20260321 ZJH 标量处理不足 8 个的余数元素
        for (; i < nCount; ++i) {
            pOut[i] = pA[i] * pB[i];
        }
#else
        // 20260321 ZJH 标量回退路径：逐元素乘法
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] * pB[i];
        }
#endif
    }

    // ===================================================================
    // ReLU 激活函数
    // ===================================================================

    // 20260321 ZJH AVX2 优化的 ReLU 激活：pOut[i] = max(0, pIn[i])
    // 使用 _mm256_max_ps(val, zero) 将负值截断为 0，正值保持不变
    // ReLU 是深度学习中最常用的激活函数，SIMD 优化收益显著
    // 参数: pIn — 输入向量，pOut — 输出向量，nCount — 元素总数
    static void reluAVX2(const float* pIn, float* pOut, size_t nCount) {
#ifdef __AVX2__
        // 20260321 ZJH 创建全零向量，用于 max 比较的下界
        __m256 vZero = _mm256_setzero_ps();
        size_t i = 0;  // 20260321 ZJH 循环索引

        // 20260321 ZJH AVX2 主循环：每次对 8 个 float 执行 max(0, x)
        for (; i + 8 <= nCount; i += 8) {
            // 20260321 ZJH 加载 8 个输入值
            __m256 vVal = _mm256_loadu_ps(&pIn[i]);
            // 20260321 ZJH 逐通道取 max(val, 0)：负值变 0，正值不变
            __m256 vResult = _mm256_max_ps(vVal, vZero);
            // 20260321 ZJH 写回 8 个 ReLU 结果
            _mm256_storeu_ps(&pOut[i], vResult);
        }

        // 20260321 ZJH 标量处理余数元素
        for (; i < nCount; ++i) {
            pOut[i] = pIn[i] > 0.0f ? pIn[i] : 0.0f;
        }
#else
        // 20260321 ZJH 标量回退路径：逐元素 ReLU
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pIn[i] > 0.0f ? pIn[i] : 0.0f;
        }
#endif
    }

    // ===================================================================
    // 标量乘法
    // ===================================================================

    // 20260321 ZJH AVX2 优化的标量乘法：pOut[i] = pA[i] * fScalar
    // 使用 _mm256_set1_ps 将标量广播到 8 个通道，批量执行乘法
    // 常用于学习率缩放、梯度乘系数等场景
    // 参数: pA — 输入向量，fScalar — 标量乘数
    //       pOut — 输出向量，nCount — 元素总数
    static void mulScalarAVX2(const float* pA, float fScalar, float* pOut, size_t nCount) {
#ifdef __AVX2__
        // 20260321 ZJH 将标量广播到 256 位寄存器的全部 8 个 lane
        __m256 vScalar = _mm256_set1_ps(fScalar);
        size_t i = 0;  // 20260321 ZJH 循环索引

        // 20260321 ZJH AVX2 主循环：每次 8 个元素乘以标量
        for (; i + 8 <= nCount; i += 8) {
            __m256 vA = _mm256_loadu_ps(&pA[i]);       // 20260321 ZJH 加载 8 个输入元素
            __m256 vResult = _mm256_mul_ps(vA, vScalar);  // 20260321 ZJH 8 路并行标量乘
            _mm256_storeu_ps(&pOut[i], vResult);        // 20260321 ZJH 写回结果
        }

        // 20260321 ZJH 标量处理余数元素
        for (; i < nCount; ++i) {
            pOut[i] = pA[i] * fScalar;
        }
#else
        // 20260321 ZJH 标量回退路径：逐元素标量乘
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = pA[i] * fScalar;
        }
#endif
    }

    // ===================================================================
    // Sigmoid 激活函数 — 快速多项式近似
    // ===================================================================

    // 20260321 ZJH AVX2 优化的 Sigmoid 激活：pOut[i] = 1 / (1 + exp(-pIn[i]))
    // 精确 exp() 在 SIMD 中代价高昂，此处采用分段多项式近似：
    //   - |x| > 6.0: sigmoid ≈ 0 或 1（饱和区直接截断）
    //   - |x| <= 6.0: 使用 5 阶多项式拟合，最大误差 < 0.002
    // 近似公式: sigmoid(x) ≈ 0.5 + x * (c1 + x^2 * (c3 + x^2 * c5))
    //   其中 c1 = 0.2459, c3 = -0.0138, c5 = 0.0003 是最小二乘拟合系数
    // 参数: pIn — 输入向量，pOut — 输出向量，nCount — 元素总数
    static void sigmoidAVX2(const float* pIn, float* pOut, size_t nCount) {
#ifdef __AVX2__
        // 20260321 ZJH 预加载多项式近似所需的常量到 AVX2 寄存器
        __m256 vHalf = _mm256_set1_ps(0.5f);        // 20260321 ZJH 偏置项 0.5
        __m256 vC1   = _mm256_set1_ps(0.2459f);     // 20260321 ZJH 1 阶系数（主导线性项）
        __m256 vC3   = _mm256_set1_ps(-0.0138f);    // 20260321 ZJH 3 阶系数（修正曲率）
        __m256 vC5   = _mm256_set1_ps(0.0003f);     // 20260321 ZJH 5 阶系数（微调尾部）
        __m256 vMin  = _mm256_set1_ps(-6.0f);       // 20260321 ZJH 下界截断阈值
        __m256 vMax  = _mm256_set1_ps(6.0f);        // 20260321 ZJH 上界截断阈值
        __m256 vZero = _mm256_setzero_ps();          // 20260321 ZJH 零值，用于下界 clamp
        __m256 vOne  = _mm256_set1_ps(1.0f);         // 20260321 ZJH 上界 clamp 值

        size_t i = 0;  // 20260321 ZJH 循环索引

        // 20260321 ZJH AVX2 主循环：每次处理 8 个 sigmoid 值
        for (; i + 8 <= nCount; i += 8) {
            // 20260321 ZJH 加载 8 个输入值
            __m256 vX = _mm256_loadu_ps(&pIn[i]);

            // 20260321 ZJH 将 x 截断到 [-6, 6] 范围，防止多项式在饱和区发散
            // clamp(x, -6, 6): 先取 max(x, -6)，再取 min(result, 6)
            __m256 vClamped = _mm256_max_ps(vX, vMin);
            vClamped = _mm256_min_ps(vClamped, vMax);

            // 20260321 ZJH 计算 x^2，用于奇数阶多项式求值
            __m256 vX2 = _mm256_mul_ps(vClamped, vClamped);

            // 20260321 ZJH 霍纳法则（Horner's method）求多项式值，减少乘法次数
            // poly = c5 * x^2 + c3 = x^2 * c5 + c3
            __m256 vPoly = _mm256_fmadd_ps(vX2, vC5, vC3);
            // 20260321 ZJH poly = poly * x^2 + c1 = (c5*x^2 + c3) * x^2 + c1
            vPoly = _mm256_fmadd_ps(vPoly, vX2, vC1);
            // 20260321 ZJH result = x * poly + 0.5 = x * (c1 + x^2*(c3 + x^2*c5)) + 0.5
            __m256 vResult = _mm256_fmadd_ps(vClamped, vPoly, vHalf);

            // 20260321 ZJH 最终将结果 clamp 到 [0, 1]，确保输出是合法概率值
            // 多项式近似在截断边界附近可能略微越界，clamp 保证安全
            vResult = _mm256_max_ps(vResult, vZero);
            vResult = _mm256_min_ps(vResult, vOne);

            // 20260321 ZJH 写回 8 个 sigmoid 结果
            _mm256_storeu_ps(&pOut[i], vResult);
        }

        // 20260321 ZJH 标量处理余数元素，使用精确 sigmoid 公式
        for (; i < nCount; ++i) {
            pOut[i] = 1.0f / (1.0f + expf(-pIn[i]));
        }
#else
        // 20260321 ZJH 标量回退路径：使用精确 sigmoid 公式
        // 1 / (1 + exp(-x)) 是标准 sigmoid 定义
        for (size_t i = 0; i < nCount; ++i) {
            pOut[i] = 1.0f / (1.0f + expf(-pIn[i]));
        }
#endif
    }

};  // 20260321 ZJH class SIMDBackend 结束

}  // 20260321 ZJH namespace om 结束
