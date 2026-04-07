// 20260330 ZJH 行业应用专用工具模块 — 对标 Cognex Assembly/Print/OCV
// 装配验证 (Assembly Verification): 检查产品组装完整性
// 印刷质量检测 (Print Quality Inspection): ISO 15416 条形码/文字质量分级
// OCV 光学字符验证 (Optical Character Verification): 读取+比对+批量校验
module;

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <map>
#include <regex>
#include <sstream>

export module om.engine.applications;

// 20260330 ZJH 导入依赖模块：张量类、张量运算、模块基类
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;

export namespace om {

// =========================================================
// 装配验证 (Assembly Verification)
// =========================================================

// 20260330 ZJH AssemblyTarget — 单个装配目标的期望描述
// 定义一个零件在产品中应处的位置、类型和是否必须存在
struct AssemblyTarget {
    std::string strPartName;     // 20260330 ZJH 零件名称，用于报告中标识
    float fX;                    // 20260330 ZJH 期望位置 X（归一化 [0,1]）
    float fY;                    // 20260330 ZJH 期望位置 Y（归一化 [0,1]）
    float fW;                    // 20260330 ZJH 期望宽度（归一化 [0,1]）
    float fH;                    // 20260330 ZJH 期望高度（归一化 [0,1]）
    int nClassId;                // 20260330 ZJH 期望分类 ID，用于检查错型
    float fPositionTolerance;    // 20260330 ZJH 位置容差（像素），中心偏差超此值判定为错位
    bool bRequired;              // 20260330 ZJH 是否必须存在，true 则缺失判定为不合格
};

// 20260330 ZJH AssemblyCheckResult — 装配验证结果
// 汇总全部零件的到位/正确/缺失/错位/错型情况
struct AssemblyCheckResult {
    bool bAllPresent;            // 20260330 ZJH 全部必需零件到位
    bool bAllCorrect;            // 20260330 ZJH 全部零件位置和类型正确
    std::vector<std::string> vecMissing;     // 20260330 ZJH 缺失零件名称列表
    std::vector<std::string> vecMisplaced;   // 20260330 ZJH 错位零件名称列表（在容差外）
    std::vector<std::string> vecWrongType;   // 20260330 ZJH 错型零件名称列表（类别不匹配）
    float fOverallScore;         // 20260330 ZJH 总分 [0,1]，合格零件占比
};

// 20260330 ZJH DetectionBox — 检测框（应用模块本地定义，避免跨模块耦合）
// 与 om.engine.metrics 中的同名结构保持字段兼容
struct DetectionBox {
    float fX1;         // 20260330 ZJH 左上角 x 坐标
    float fY1;         // 20260330 ZJH 左上角 y 坐标
    float fX2;         // 20260330 ZJH 右下角 x 坐标
    float fY2;         // 20260330 ZJH 右下角 y 坐标
    int nClassId;      // 20260330 ZJH 类别 ID
    float fConfidence; // 20260330 ZJH 置信度 [0,1]
};

// 20260330 ZJH AssemblyVerifier — 装配完整性验证器
// 工作流: setTemplate → verify(检测结果) → 比对每个零件的位置/类型
class AssemblyVerifier {
public:
    // 20260330 ZJH setTemplate — 设置期望装配模板
    // vecTargets: 所有期望零件的位置和类型描述
    void setTemplate(const std::vector<AssemblyTarget>& vecTargets) {
        m_vecTargets = vecTargets;  // 20260330 ZJH 保存装配模板
    }

    // 20260330 ZJH verify — 执行装配验证
    // vecDetections: 实际检测到的目标框列表
    // 返回: 装配检查结果，含缺失/错位/错型详情
    AssemblyCheckResult verify(const std::vector<DetectionBox>& vecDetections) const {
        AssemblyCheckResult result;  // 20260330 ZJH 初始化结果
        result.bAllPresent = true;   // 20260330 ZJH 初始假设全部到位
        result.bAllCorrect = true;   // 20260330 ZJH 初始假设全部正确
        result.fOverallScore = 0.0f; // 20260330 ZJH 总分初始化

        int nCorrectCount = 0;       // 20260330 ZJH 合格零件计数器

        // 20260330 ZJH 标记已匹配的检测框，避免重复匹配
        std::vector<bool> vecUsed(vecDetections.size(), false);

        // 20260330 ZJH 逐目标遍历模板，为每个零件找最优匹配
        for (const auto& target : m_vecTargets) {
            // 20260330 ZJH 计算目标中心点（归一化坐标）
            float fTargetCx = target.fX + target.fW * 0.5f;  // 20260330 ZJH 目标中心 X
            float fTargetCy = target.fY + target.fH * 0.5f;  // 20260330 ZJH 目标中心 Y

            int nBestIdx = -1;       // 20260330 ZJH 最优匹配检测框索引
            float fBestDist = 1e9f;  // 20260330 ZJH 最小距离（初始化为极大值）

            // 20260330 ZJH 遍历所有检测框，找距离最近的未使用框
            for (int i = 0; i < static_cast<int>(vecDetections.size()); ++i) {
                if (vecUsed[i]) {
                    continue;  // 20260330 ZJH 已被其他零件匹配，跳过
                }

                const auto& det = vecDetections[i];
                // 20260330 ZJH 计算检测框中心点
                float fDetCx = (det.fX1 + det.fX2) * 0.5f;  // 20260330 ZJH 检测中心 X
                float fDetCy = (det.fY1 + det.fY2) * 0.5f;  // 20260330 ZJH 检测中心 Y

                // 20260330 ZJH 欧几里得距离
                float fDx = fDetCx - fTargetCx;  // 20260330 ZJH X 偏差
                float fDy = fDetCy - fTargetCy;  // 20260330 ZJH Y 偏差
                float fDist = std::sqrt(fDx * fDx + fDy * fDy);  // 20260330 ZJH 计算距离

                // 20260330 ZJH 更新最近匹配
                if (fDist < fBestDist) {
                    fBestDist = fDist;  // 20260330 ZJH 更新最小距离
                    nBestIdx = i;       // 20260330 ZJH 更新最优索引
                }
            }

            // 20260330 ZJH 判定零件是否缺失
            if (nBestIdx < 0 || fBestDist > target.fPositionTolerance * 3.0f) {
                // 20260330 ZJH 未找到匹配，或距离远超容差的3倍 → 视为缺失
                if (target.bRequired) {
                    result.vecMissing.push_back(target.strPartName);  // 20260330 ZJH 记录缺失
                    result.bAllPresent = false;  // 20260330 ZJH 标记不完整
                    result.bAllCorrect = false;  // 20260330 ZJH 标记不正确
                }
                continue;  // 20260330 ZJH 处理下一个零件
            }

            // 20260330 ZJH 标记已使用
            vecUsed[nBestIdx] = true;

            const auto& bestDet = vecDetections[nBestIdx];

            // 20260330 ZJH 检查类型是否匹配
            bool bTypeCorrect = (bestDet.nClassId == target.nClassId);
            if (!bTypeCorrect) {
                result.vecWrongType.push_back(target.strPartName);  // 20260330 ZJH 记录错型
                result.bAllCorrect = false;  // 20260330 ZJH 标记不正确
                continue;  // 20260330 ZJH 类型错误不计入合格
            }

            // 20260330 ZJH 检查位置是否在容差内
            bool bPositionOk = (fBestDist <= target.fPositionTolerance);
            if (!bPositionOk) {
                result.vecMisplaced.push_back(target.strPartName);  // 20260330 ZJH 记录错位
                result.bAllCorrect = false;  // 20260330 ZJH 标记不正确
            } else {
                nCorrectCount++;  // 20260330 ZJH 类型和位置都正确，计入合格
            }
        }

        // 20260330 ZJH 计算总分：合格零件数 / 总零件数
        int nTotal = static_cast<int>(m_vecTargets.size());
        result.fOverallScore = (nTotal > 0)
            ? static_cast<float>(nCorrectCount) / static_cast<float>(nTotal)
            : 1.0f;  // 20260330 ZJH 无目标时默认满分

        return result;  // 20260330 ZJH 返回装配验证结果
    }

private:
    std::vector<AssemblyTarget> m_vecTargets;  // 20260330 ZJH 装配模板列表
};

// =========================================================
// 印刷质量检测 (Print Quality Inspection)
// =========================================================

// 20260330 ZJH PrintQualityResult — 印刷质量检测结果
// 参照 ISO 15416 (条形码) / ISO 15415 (二维码) 标准等级
struct PrintQualityResult {
    // 20260330 ZJH ISO 等级: A(4.0) 最优, B(3.0), C(2.0), D(1.0), F(0.0) 不合格
    enum Grade { A, B, C, D, F } eGrade;
    float fEdgeContrast;          // 20260330 ZJH 边缘对比度 [0,1]，条空过渡清晰度
    float fModulation;            // 20260330 ZJH 调制度 [0,1]，最小反射率差 / 最大反射率差
    float fDecodability;          // 20260330 ZJH 可解码性 [0,1]，解码余量
    float fDefects;               // 20260330 ZJH 缺陷度 [0,1]，0=无缺陷 1=严重缺陷
    float fSymbolContrast;        // 20260330 ZJH 符号对比度 [0,1]，最亮-最暗反射率差
    std::string strGradeReport;   // 20260330 ZJH 等级报告文本，人类可读
};

// 20260330 ZJH PrintQualityInspector — 印刷质量检测器
// 分析条形码/二维码/文字的印刷质量，给出 ISO 标准分级
class PrintQualityInspector {
public:
    // 20260330 ZJH inspectBarcode — 分析条形码/二维码印刷质量
    // vecImage: 浮点图像数据 [0,1] 归一化
    // nC, nH, nW: 通道数、高度、宽度
    // fX, fY, fW, fH: 条码区域（归一化坐标）
    // 返回: 印刷质量检测结果（ISO 等级+各项指标）
    PrintQualityResult inspectBarcode(const std::vector<float>& vecImage,
                                      int nC, int nH, int nW,
                                      float fX, float fY, float fW, float fH) {
        PrintQualityResult result;  // 20260330 ZJH 初始化结果

        // 20260330 ZJH 计算 ROI 的像素边界
        int nRoiX1 = static_cast<int>(fX * nW);          // 20260330 ZJH ROI 左边界
        int nRoiY1 = static_cast<int>(fY * nH);          // 20260330 ZJH ROI 上边界
        int nRoiX2 = static_cast<int>((fX + fW) * nW);   // 20260330 ZJH ROI 右边界
        int nRoiY2 = static_cast<int>((fY + fH) * nH);   // 20260330 ZJH ROI 下边界

        // 20260330 ZJH 边界裁剪，防止越界
        nRoiX1 = std::max(0, std::min(nRoiX1, nW - 1));
        nRoiY1 = std::max(0, std::min(nRoiY1, nH - 1));
        nRoiX2 = std::max(nRoiX1 + 1, std::min(nRoiX2, nW));
        nRoiY2 = std::max(nRoiY1 + 1, std::min(nRoiY2, nH));

        // 20260330 ZJH 提取 ROI 区域灰度值（多通道取第一通道）
        std::vector<float> vecRoi;  // 20260330 ZJH ROI 内像素灰度值
        float fMin = 1.0f;          // 20260330 ZJH 最暗像素值
        float fMax = 0.0f;          // 20260330 ZJH 最亮像素值
        float fSum = 0.0f;          // 20260330 ZJH 像素值总和

        // 20260330 ZJH 遍历 ROI 像素，取第一通道
        for (int y = nRoiY1; y < nRoiY2; ++y) {
            for (int x = nRoiX1; x < nRoiX2; ++x) {
                // 20260330 ZJH CHW 布局: channel=0, 偏移 = 0*H*W + y*W + x
                float fVal = vecImage[static_cast<size_t>(y) * nW + x];
                vecRoi.push_back(fVal);   // 20260330 ZJH 收集像素值
                fMin = std::min(fMin, fVal);  // 20260330 ZJH 更新最小值
                fMax = std::max(fMax, fVal);  // 20260330 ZJH 更新最大值
                fSum += fVal;                 // 20260330 ZJH 累加
            }
        }

        int nRoiSize = static_cast<int>(vecRoi.size());  // 20260330 ZJH ROI 像素总数

        // 20260330 ZJH 符号对比度 = 最亮 - 最暗
        result.fSymbolContrast = (nRoiSize > 0) ? (fMax - fMin) : 0.0f;

        // 20260330 ZJH 计算边缘对比度 — 水平方向相邻像素差的平均值
        float fEdgeSum = 0.0f;  // 20260330 ZJH 边缘差值总和
        int nEdgeCount = 0;     // 20260330 ZJH 边缘计数
        int nRoiW = nRoiX2 - nRoiX1;  // 20260330 ZJH ROI 宽度

        // 20260330 ZJH 逐行扫描，计算水平相邻像素差（模拟扫描线）
        for (int y = nRoiY1; y < nRoiY2; ++y) {
            for (int x = nRoiX1; x < nRoiX2 - 1; ++x) {
                float fCurr = vecImage[static_cast<size_t>(y) * nW + x];
                float fNext = vecImage[static_cast<size_t>(y) * nW + x + 1];
                float fDiff = std::abs(fNext - fCurr);  // 20260330 ZJH 相邻像素差
                fEdgeSum += fDiff;  // 20260330 ZJH 累加
                nEdgeCount++;       // 20260330 ZJH 计数
            }
        }
        // 20260330 ZJH 边缘对比度 = 平均梯度 / 符号对比度，归一化到 [0,1]
        float fAvgEdge = (nEdgeCount > 0) ? (fEdgeSum / nEdgeCount) : 0.0f;
        result.fEdgeContrast = (result.fSymbolContrast > 1e-6f)
            ? std::min(1.0f, fAvgEdge / result.fSymbolContrast * 2.0f)
            : 0.0f;

        // 20260330 ZJH 调制度 — 扫描线上局部极值差与全局对比度之比
        result.fModulation = computeModulation(vecRoi, nRoiW);

        // 20260330 ZJH 可解码性 — 基于调制度和符号对比度的综合评估
        // 简化模型: decodability ≈ min(modulation, symbolContrast)
        result.fDecodability = std::min(result.fModulation, result.fSymbolContrast);

        // 20260330 ZJH 缺陷度 — 检测不规则性（标准差 / 范围）
        float fMean = (nRoiSize > 0) ? (fSum / nRoiSize) : 0.0f;
        float fVariance = 0.0f;  // 20260330 ZJH 方差
        for (float fVal : vecRoi) {
            float fDiff = fVal - fMean;  // 20260330 ZJH 偏差
            fVariance += fDiff * fDiff;  // 20260330 ZJH 累加平方偏差
        }
        fVariance = (nRoiSize > 1) ? (fVariance / (nRoiSize - 1)) : 0.0f;
        float fStdDev = std::sqrt(fVariance);  // 20260330 ZJH 标准差

        // 20260330 ZJH 缺陷度: 标准差越大表示印刷越不均匀
        result.fDefects = (result.fSymbolContrast > 1e-6f)
            ? std::min(1.0f, fStdDev / result.fSymbolContrast)
            : 0.0f;

        // 20260330 ZJH 计算 ISO 等级
        result.eGrade = computeGrade(result);

        // 20260330 ZJH 生成等级报告文本
        result.strGradeReport = generateReport(result);

        return result;  // 20260330 ZJH 返回印刷质量检测结果
    }

    // 20260330 ZJH inspectText — 分析文字印刷质量
    // vecImage: 浮点图像数据 [0,1] 归一化
    // nC, nH, nW: 通道数、高度、宽度
    // strExpectedText: 期望文字（暂未使用，预留 OCR 对比）
    // 返回: 印刷质量检测结果（基于全图分析）
    PrintQualityResult inspectText(const std::vector<float>& vecImage,
                                   int nC, int nH, int nW,
                                   const std::string& strExpectedText) {
        // 20260330 ZJH 文字印刷质量 = 对整张图执行条码同等分析
        return inspectBarcode(vecImage, nC, nH, nW, 0.0f, 0.0f, 1.0f, 1.0f);
    }

private:
    // 20260330 ZJH computeModulation — 计算调制度
    // vecRoi: ROI 灰度值列表（行优先）
    // nRoiW: ROI 每行宽度
    // 返回: 调制度 [0,1]
    float computeModulation(const std::vector<float>& vecRoi, int nRoiW) const {
        if (vecRoi.empty() || nRoiW <= 1) {
            return 0.0f;  // 20260330 ZJH 数据不足，返回0
        }

        // 20260330 ZJH 取中间行作为代表扫描线
        int nRows = static_cast<int>(vecRoi.size()) / nRoiW;
        int nMidRow = nRows / 2;  // 20260330 ZJH 中间行索引

        // 20260330 ZJH 在扫描线上找局部极大/极小值
        float fMinLocal = 1.0f;   // 20260330 ZJH 局部最小值
        float fMaxLocal = 0.0f;   // 20260330 ZJH 局部最大值
        float fMinGlobal = 1.0f;  // 20260330 ZJH 全局最小值
        float fMaxGlobal = 0.0f;  // 20260330 ZJH 全局最大值

        int nStart = nMidRow * nRoiW;  // 20260330 ZJH 扫描线起始索引
        // 20260330 ZJH 遍历扫描线，查找极值
        for (int i = 0; i < nRoiW; ++i) {
            float fVal = vecRoi[nStart + i];  // 20260330 ZJH 当前像素值
            fMinGlobal = std::min(fMinGlobal, fVal);  // 20260330 ZJH 更新全局最小
            fMaxGlobal = std::max(fMaxGlobal, fVal);  // 20260330 ZJH 更新全局最大

            // 20260330 ZJH 检查局部极值（非边界点，前后变化方向相反）
            if (i > 0 && i < nRoiW - 1) {
                float fPrev = vecRoi[nStart + i - 1];
                float fNext = vecRoi[nStart + i + 1];
                // 20260330 ZJH 局部极大值
                if (fVal >= fPrev && fVal >= fNext) {
                    fMaxLocal = std::max(fMaxLocal, fVal);
                }
                // 20260330 ZJH 局部极小值
                if (fVal <= fPrev && fVal <= fNext) {
                    fMinLocal = std::min(fMinLocal, fVal);
                }
            }
        }

        // 20260330 ZJH 调制度 = (局部极大 - 局部极小) / (全局极大 - 全局极小)
        float fGlobalRange = fMaxGlobal - fMinGlobal;  // 20260330 ZJH 全局范围
        if (fGlobalRange < 1e-6f) {
            return 0.0f;  // 20260330 ZJH 无对比度，返回0
        }
        float fLocalRange = fMaxLocal - fMinLocal;  // 20260330 ZJH 局部范围
        return std::min(1.0f, fLocalRange / fGlobalRange);  // 20260330 ZJH 归一化到 [0,1]
    }

    // 20260330 ZJH computeGrade — 根据各项指标计算 ISO 等级
    // result: 已填充指标的结果结构
    // 返回: A/B/C/D/F 等级
    PrintQualityResult::Grade computeGrade(const PrintQualityResult& result) const {
        // 20260330 ZJH ISO 15416 综合评分 = 各项指标的最低分
        // 每项指标映射到 4.0(A)/3.0(B)/2.0(C)/1.0(D)/0.0(F) 分数

        // 20260330 ZJH 取最低项作为总等级（木桶原理）
        float fMinScore = 4.0f;  // 20260330 ZJH 初始化为最高分

        // 20260330 ZJH 符号对比度评分
        fMinScore = std::min(fMinScore, gradeMetric(result.fSymbolContrast));
        // 20260330 ZJH 调制度评分
        fMinScore = std::min(fMinScore, gradeMetric(result.fModulation));
        // 20260330 ZJH 可解码性评分
        fMinScore = std::min(fMinScore, gradeMetric(result.fDecodability));
        // 20260330 ZJH 边缘对比度评分
        fMinScore = std::min(fMinScore, gradeMetric(result.fEdgeContrast));
        // 20260330 ZJH 缺陷度评分（反向: 低缺陷=高分）
        fMinScore = std::min(fMinScore, gradeMetric(1.0f - result.fDefects));

        // 20260330 ZJH 分数转等级
        if (fMinScore >= 3.5f) return PrintQualityResult::A;       // 20260330 ZJH A 级: ≥3.5
        if (fMinScore >= 2.5f) return PrintQualityResult::B;       // 20260330 ZJH B 级: ≥2.5
        if (fMinScore >= 1.5f) return PrintQualityResult::C;       // 20260330 ZJH C 级: ≥1.5
        if (fMinScore >= 0.5f) return PrintQualityResult::D;       // 20260330 ZJH D 级: ≥0.5
        return PrintQualityResult::F;                               // 20260330 ZJH F 级: <0.5
    }

    // 20260330 ZJH gradeMetric — 单项指标评分 [0,4.0]
    // fValue: 指标值 [0,1]
    // 返回: ISO 分数 0.0/1.0/2.0/3.0/4.0
    float gradeMetric(float fValue) const {
        if (fValue >= 0.87f) return 4.0f;  // 20260330 ZJH A 级阈值
        if (fValue >= 0.70f) return 3.0f;  // 20260330 ZJH B 级阈值
        if (fValue >= 0.50f) return 2.0f;  // 20260330 ZJH C 级阈值
        if (fValue >= 0.25f) return 1.0f;  // 20260330 ZJH D 级阈值
        return 0.0f;                        // 20260330 ZJH F 级
    }

    // 20260330 ZJH generateReport — 生成人类可读的等级报告
    // result: 完整检测结果
    // 返回: 多行文本报告
    std::string generateReport(const PrintQualityResult& result) const {
        // 20260330 ZJH 等级名称映射
        static const char* s_arrGradeNames[] = {"A", "B", "C", "D", "F"};
        std::ostringstream oss;  // 20260330 ZJH 报告构建器

        oss << "=== Print Quality Report ===\n";
        oss << "Overall Grade: " << s_arrGradeNames[static_cast<int>(result.eGrade)] << "\n";
        oss << "Symbol Contrast: " << result.fSymbolContrast << "\n";
        oss << "Edge Contrast:   " << result.fEdgeContrast << "\n";
        oss << "Modulation:      " << result.fModulation << "\n";
        oss << "Decodability:    " << result.fDecodability << "\n";
        oss << "Defects:         " << result.fDefects << "\n";

        return oss.str();  // 20260330 ZJH 返回报告字符串
    }
};

// =========================================================
// OCV 光学字符验证 (Optical Character Verification)
// =========================================================

// 20260330 ZJH OCVResult — 光学字符验证结果
// 记录匹配状态、相似度、不匹配字符位置等
struct OCVResult {
    bool bMatch;                 // 20260330 ZJH 是否匹配（读取==期望）
    float fSimilarity;           // 20260330 ZJH 字符级相似度 [0,1]
    std::string strRead;         // 20260330 ZJH 实际读取的文字
    std::string strExpected;     // 20260330 ZJH 期望的文字
    std::vector<int> vecMismatchPositions;  // 20260330 ZJH 不匹配字符位置索引列表
    float fPrintQuality;         // 20260330 ZJH 印刷质量分数 [0,1]
};

// 20260330 ZJH OCVVerifier — 光学字符验证器
// 工作流: setExpectedPattern → verify(读取文字) → 逐字符比对
// 支持简单正则模式匹配(*, ?, [a-z] 等)
class OCVVerifier {
public:
    // 20260330 ZJH setExpectedPattern — 设置期望文字模式
    // strPattern: 期望文字或正则模式（支持 std::regex 语法）
    void setExpectedPattern(const std::string& strPattern) {
        m_strPattern = strPattern;  // 20260330 ZJH 保存模式串
    }

    // 20260330 ZJH verify — 验证单条读取文字
    // strReadText: OCR 引擎输出的读取结果
    // 返回: 验证结果（匹配/相似度/不匹配位置）
    OCVResult verify(const std::string& strReadText) const {
        OCVResult result;            // 20260330 ZJH 初始化结果
        result.strRead = strReadText;        // 20260330 ZJH 记录实际读取
        result.strExpected = m_strPattern;   // 20260330 ZJH 记录期望模式
        result.fPrintQuality = 1.0f;         // 20260330 ZJH 默认印刷质量满分

        // 20260330 ZJH 尝试正则匹配
        result.bMatch = matchPattern(strReadText);

        // 20260330 ZJH 计算字符级相似度（Levenshtein 距离的归一化互补值）
        int nMaxLen = static_cast<int>(std::max(strReadText.size(), m_strPattern.size()));
        if (nMaxLen == 0) {
            result.fSimilarity = 1.0f;  // 20260330 ZJH 两个空串完全匹配
            result.bMatch = true;
            return result;
        }

        // 20260330 ZJH 逐字符比对（简化版，不做编辑距离）
        int nMinLen = static_cast<int>(std::min(strReadText.size(), m_strPattern.size()));
        int nMatchCount = 0;  // 20260330 ZJH 匹配字符数

        // 20260330 ZJH 逐位置比较，记录不匹配位置
        for (int i = 0; i < nMinLen; ++i) {
            if (strReadText[i] == m_strPattern[i]) {
                nMatchCount++;  // 20260330 ZJH 字符匹配，计数+1
            } else {
                result.vecMismatchPositions.push_back(i);  // 20260330 ZJH 记录不匹配位置
            }
        }

        // 20260330 ZJH 长度差异部分全部视为不匹配
        for (int i = nMinLen; i < nMaxLen; ++i) {
            result.vecMismatchPositions.push_back(i);  // 20260330 ZJH 超出部分为不匹配
        }

        // 20260330 ZJH 相似度 = 匹配字符数 / 最大长度
        result.fSimilarity = static_cast<float>(nMatchCount) / static_cast<float>(nMaxLen);

        // 20260330 ZJH 印刷质量与相似度成正比（简化模型）
        result.fPrintQuality = result.fSimilarity;

        return result;  // 20260330 ZJH 返回验证结果
    }

    // 20260330 ZJH verifyBatch — 批量验证多条文字
    // vecTexts: OCR 引擎输出的多条读取结果
    // 返回: 每条文字对应的验证结果
    std::vector<OCVResult> verifyBatch(const std::vector<std::string>& vecTexts) const {
        std::vector<OCVResult> vecResults;  // 20260330 ZJH 结果列表
        vecResults.reserve(vecTexts.size());  // 20260330 ZJH 预分配内存

        // 20260330 ZJH 逐条验证
        for (const auto& strText : vecTexts) {
            vecResults.push_back(verify(strText));  // 20260330 ZJH 调用单条验证
        }

        return vecResults;  // 20260330 ZJH 返回批量结果
    }

private:
    std::string m_strPattern;  // 20260330 ZJH 期望文字模式

    // 20260330 ZJH matchPattern — 简单正则匹配
    // str: 待匹配的字符串
    // 返回: 是否匹配期望模式
    bool matchPattern(const std::string& str) const {
        if (m_strPattern.empty()) {
            return str.empty();  // 20260330 ZJH 空模式只匹配空串
        }

        // 20260330 ZJH 尝试 std::regex 匹配
        try {
            std::regex re(m_strPattern);  // 20260330 ZJH 编译正则表达式
            return std::regex_match(str, re);  // 20260330 ZJH 全串匹配
        } catch (const std::regex_error&) {
            // 20260330 ZJH 正则编译失败，回退为精确匹配
            return (str == m_strPattern);
        }
    }
};

}  // namespace om
