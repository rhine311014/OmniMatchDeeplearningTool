// 20260402 ZJH DataDiagnostics — 数据质量诊断工具
// 分析数据集质量（类别不平衡/重复图/损坏图），提供改进建议和自动平衡权重
// 用途: 训练前自动检测数据集问题，指导用户优化数据质量
// 对标: MVTec Deep Learning Tool 数据集统计 + Halcon train_dl_model 数据校验
#pragma once

#include <QString>       // 20260402 ZJH 诊断报告中的警告/建议文本
#include <QStringList>   // 20260402 ZJH 警告和建议列表
#include <vector>        // 20260402 ZJH 标签向量
#include <map>           // 20260402 ZJH 类别统计映射
#include <cmath>         // 20260402 ZJH sqrt/log 用于权重计算
#include <algorithm>     // 20260402 ZJH min_element/max_element
#include <numeric>       // 20260402 ZJH accumulate 求和

// 20260402 ZJH DataDiagReport — 数据质量诊断报告
// 包含数据集统计信息、问题检测结果和改进建议
struct DataDiagReport {
    int nTotalImages = 0;      // 20260402 ZJH 总图像数
    int nLabeledImages = 0;    // 20260402 ZJH 已标注图像数（标签 >= 0）
    int nUnlabeledImages = 0;  // 20260402 ZJH 未标注图像数（标签 < 0）

    // 20260402 ZJH 每个类别的样本计数 (classId → count)
    std::map<int, int> mapClassCounts;

    // 20260402 ZJH 类别不平衡检测结果
    bool bClassImbalanced = false;  // 20260402 ZJH 是否存在类别不平衡（max/min > 3）
    float fImbalanceRatio = 1.0f;   // 20260402 ZJH 不平衡比值（最多类 / 最少类的样本数）

    int nDuplicateImages = 0;  // 20260402 ZJH 重复图像数（预留，当前未实现像素级去重）
    int nCorruptImages = 0;    // 20260402 ZJH 损坏图像数（预留，当前未实现文件完整性校验）

    // 20260402 ZJH 诊断警告列表（问题描述）
    QStringList vecWarnings;
    // 20260402 ZJH 改进建议列表（可执行的优化操作）
    QStringList vecSuggestions;

    // 20260402 ZJH 自动平衡权重 — classId → 建议的 loss 加权系数
    // 逆频率加权: weight[c] = totalSamples / (nClasses × count[c])
    // 少数类权重高、多数类权重低，用于加权交叉熵损失函数
    std::map<int, float> mapClassWeights;
};

// 20260402 ZJH DataDiagnostics — 数据质量诊断工具类
// 提供静态方法，无状态，可在任意线程调用
class DataDiagnostics {
public:
    // 20260402 ZJH analyze — 分析数据集质量
    // 参数: vecLabels - 每张图像的类别标签（-1 表示未标注）
    //       nNumClasses - 类别总数（用于检测缺失类别）
    // 返回: 完整的诊断报告（统计信息 + 警告 + 建议 + 平衡权重）
    static DataDiagReport analyze(const std::vector<int>& vecLabels, int nNumClasses) {
        DataDiagReport report;  // 20260402 ZJH 初始化空报告

        // 20260402 ZJH 统计总图像数
        report.nTotalImages = static_cast<int>(vecLabels.size());

        // 20260402 ZJH 遍历标签，统计标注/未标注数量及每类计数
        for (int nLabel : vecLabels) {
            if (nLabel < 0) {
                // 20260402 ZJH 未标注图像（标签为负数）
                report.nUnlabeledImages++;
            } else {
                // 20260402 ZJH 已标注图像
                report.nLabeledImages++;
                report.mapClassCounts[nLabel]++;  // 20260402 ZJH 累加对应类别计数
            }
        }

        // 20260402 ZJH 检查: 数据集为空
        if (report.nTotalImages == 0) {
            report.vecWarnings.append(QStringLiteral("数据集为空，无法进行训练。"));
            report.vecSuggestions.append(QStringLiteral("请导入图像到数据集中。"));
            return report;  // 20260402 ZJH 空数据集无需后续检查
        }

        // 20260402 ZJH 检查: 未标注图像过多（超过 50%）
        if (report.nUnlabeledImages > report.nTotalImages / 2) {
            report.vecWarnings.append(QStringLiteral("超过 50%% 的图像未标注 (%1/%2)。")
                .arg(report.nUnlabeledImages).arg(report.nTotalImages));
            report.vecSuggestions.append(QStringLiteral("请标注更多图像，或启用半监督训练模式。"));
        }

        // 20260402 ZJH 检查: 已标注图像数量不足（少于每类 10 张）
        if (report.nLabeledImages > 0 && report.nLabeledImages < nNumClasses * 10) {
            report.vecWarnings.append(QStringLiteral("已标注图像数量较少 (%1)，平均每类不足 10 张。")
                .arg(report.nLabeledImages));
            report.vecSuggestions.append(QStringLiteral("建议启用数据增强和少样本学习模式。"));
        }

        // 20260402 ZJH 检查: 缺失类别（某些类别没有任何样本）
        for (int nClassId = 0; nClassId < nNumClasses; ++nClassId) {
            if (report.mapClassCounts.find(nClassId) == report.mapClassCounts.end()) {
                // 20260402 ZJH 类别 nClassId 没有任何样本
                report.mapClassCounts[nClassId] = 0;  // 20260402 ZJH 显式记录为 0
                report.vecWarnings.append(QStringLiteral("类别 %1 没有任何训练样本。").arg(nClassId));
                report.vecSuggestions.append(QStringLiteral("请为类别 %1 添加训练样本，或使用 AI 数据合成。").arg(nClassId));
            }
        }

        // 20260402 ZJH 检测类别不平衡（仅在有 ≥2 个非空类别时）
        if (report.mapClassCounts.size() >= 2) {
            int nMaxCount = 0;   // 20260402 ZJH 最多类的样本数
            int nMinCount = std::numeric_limits<int>::max();  // 20260402 ZJH 最少类的样本数（排除 0）
            int nMaxClass = 0;   // 20260402 ZJH 最多类的 ID
            int nMinClass = 0;   // 20260402 ZJH 最少类的 ID

            // 20260402 ZJH 遍历所有类别找最大最小
            for (const auto& [nClassId, nCount] : report.mapClassCounts) {
                if (nCount > nMaxCount) {
                    nMaxCount = nCount;  // 20260402 ZJH 更新最大
                    nMaxClass = nClassId;
                }
                if (nCount > 0 && nCount < nMinCount) {
                    nMinCount = nCount;  // 20260402 ZJH 更新最小（排除空类）
                    nMinClass = nClassId;
                }
            }

            // 20260402 ZJH 计算不平衡比值: max/min
            if (nMinCount > 0 && nMinCount < std::numeric_limits<int>::max()) {
                report.fImbalanceRatio = static_cast<float>(nMaxCount) / static_cast<float>(nMinCount);

                // 20260402 ZJH 不平衡比值 > 3 时警告
                // 经验值: <3 可接受, 3~10 需加权, >10 需过采样/合成
                if (report.fImbalanceRatio > 3.0f) {
                    report.bClassImbalanced = true;  // 20260402 ZJH 标记类别不平衡
                    report.vecWarnings.append(QStringLiteral(
                        "类别不平衡: 类 %1 有 %2 张，类 %3 仅 %4 张（比值 %5:1）。")
                        .arg(nMaxClass).arg(nMaxCount).arg(nMinClass).arg(nMinCount)
                        .arg(QString::number(static_cast<double>(report.fImbalanceRatio), 'f', 1)));
                    report.vecSuggestions.append(QStringLiteral(
                        "建议: (1) 为少数类采集更多样本 (2) 启用加权损失函数 (3) 使用 AI 数据合成补齐。"));
                }
            }
        }

        // 20260402 ZJH 计算自动平衡类别权重
        report.mapClassWeights = computeClassWeights(vecLabels, nNumClasses);

        return report;  // 20260402 ZJH 返回完整诊断报告
    }

    // 20260402 ZJH computeClassWeights — 计算逆频率类别权重
    // 公式: weight[c] = totalSamples / (nClasses × count[c])
    // 归一化: 使所有权重的均值 = 1.0（不改变整体梯度幅度）
    // 参数: vecLabels - 标签向量; nNumClasses - 类别总数
    // 返回: classId → weight 映射
    static std::map<int, float> computeClassWeights(const std::vector<int>& vecLabels, int nNumClasses) {
        std::map<int, float> mapWeights;  // 20260402 ZJH 结果权重映射

        // 20260402 ZJH 统计每类样本数
        std::vector<int> vecCounts(nNumClasses, 0);  // 20260402 ZJH 每类计数，初始化为 0
        int nTotalLabeled = 0;  // 20260402 ZJH 有效标注总数

        for (int nLabel : vecLabels) {
            // 20260402 ZJH 仅统计有效标签（0 ~ nNumClasses-1）
            if (nLabel >= 0 && nLabel < nNumClasses) {
                vecCounts[nLabel]++;
                nTotalLabeled++;
            }
        }

        // 20260402 ZJH 无有效标签时返回均匀权重
        if (nTotalLabeled == 0) {
            for (int c = 0; c < nNumClasses; ++c) {
                mapWeights[c] = 1.0f;  // 20260402 ZJH 均匀权重
            }
            return mapWeights;
        }

        // 20260402 ZJH 逆频率公式: w[c] = total / (nClasses × count[c])
        // 空类使用 max_weight 兜底（避免除以零）
        float fMaxWeight = 0.0f;  // 20260402 ZJH 记录最大权重（用于空类兜底）
        for (int c = 0; c < nNumClasses; ++c) {
            if (vecCounts[c] > 0) {
                // 20260402 ZJH 标准逆频率公式
                float fWeight = static_cast<float>(nTotalLabeled)
                              / (static_cast<float>(nNumClasses) * static_cast<float>(vecCounts[c]));
                mapWeights[c] = fWeight;
                if (fWeight > fMaxWeight) fMaxWeight = fWeight;  // 20260402 ZJH 更新最大值
            }
        }

        // 20260402 ZJH 空类兜底: 使用现有最大权重的 2 倍（强激励采集该类样本）
        for (int c = 0; c < nNumClasses; ++c) {
            if (vecCounts[c] == 0) {
                mapWeights[c] = fMaxWeight * 2.0f;  // 20260402 ZJH 空类权重 = 2 × 最大权重
            }
        }

        // 20260402 ZJH 归一化: 使权重均值 = 1.0（不改变整体梯度规模）
        float fSum = 0.0f;
        for (const auto& [nClassId, fWeight] : mapWeights) {
            fSum += fWeight;  // 20260402 ZJH 累加所有权重
        }
        float fMean = fSum / static_cast<float>(nNumClasses);  // 20260402 ZJH 权重均值
        if (fMean > 1e-7f) {
            for (auto& [nClassId, fWeight] : mapWeights) {
                fWeight /= fMean;  // 20260402 ZJH 归一化到均值=1
            }
        }

        return mapWeights;  // 20260402 ZJH 返回归一化后的类别权重
    }
};
