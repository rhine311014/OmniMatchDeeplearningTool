// 20260330 ZJH 混合判定引擎 — 传统视觉 + AI + 规则引擎
// 对标所有四家竞品的混合检测能力（Keyence/Cognex/Basler/MVTec）
// 三大组件:
//   1. RuleEngine    — 基于阈值/范围的规则判定（支持 7 种比较运算符）
//   2. HybridJudge   — 多源融合判定（4 种策略: AI优先/传统优先/投票/级联）
//   3. JudgeResult   — OK/NG/Uncertain 三态判定
module;

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>

export module om.engine.hybrid_judge;

export namespace om {

// 20260330 ZJH JudgeResult — 三态判定结果
// OK: 合格品; NG: 不合格品; Uncertain: 无法确定（需人工复判）
enum class JudgeResult { OK, NG, Uncertain };

// =========================================================
// 规则引擎 (Rule Engine)
// =========================================================

// 20260330 ZJH Rule — 单条判定规则
// 基于指标名称和比较运算符进行阈值判定
// 例如: "area" > 500 → NG, "confidence" InRange [0.3, 0.7] → Uncertain
struct Rule {
    std::string strName;       // 20260330 ZJH 规则名称（如"最小面积检查"）
    std::string strMetric;     // 20260330 ZJH 指标名称: "area", "confidence", "count", "brightness" 等
    // 20260330 ZJH 比较运算符枚举
    // GT: 大于, LT: 小于, GE: 大于等于, LE: 小于等于
    // EQ: 等于, NE: 不等于, InRange: 范围内 [fValue1, fValue2]
    enum Op { GT, LT, GE, LE, EQ, NE, InRange } eOp;
    float fValue1 = 0.0f;     // 20260330 ZJH 比较值1（所有运算符使用）
    float fValue2 = 0.0f;     // 20260330 ZJH 比较值2（仅 InRange 使用，表示上界）
    JudgeResult eOnMatch;      // 20260330 ZJH 匹配时的判定结果（OK 或 NG 或 Uncertain）
};

// 20260330 ZJH RuleEngine — 基于规则的判定引擎
// 维护一组规则，按顺序评估输入指标，返回第一个匹配规则的判定结果
// 若无规则匹配则返回 OK（默认合格）
class RuleEngine {
public:
    // 20260330 ZJH addRule — 添加一条判定规则
    // rule: 规则定义（名称、指标、运算符、阈值、判定结果）
    void addRule(const Rule& rule) {
        m_vecRules.push_back(rule);  // 20260330 ZJH 追加到规则列表末尾
    }

    // 20260330 ZJH removeRule — 按名称移除规则
    // strName: 要移除的规则名称
    // 返回: true 表示成功移除，false 表示未找到
    bool removeRule(const std::string& strName) {
        for (auto it = m_vecRules.begin(); it != m_vecRules.end(); ++it) {
            if (it->strName == strName) {
                m_vecRules.erase(it);  // 20260330 ZJH 找到并移除
                return true;
            }
        }
        return false;  // 20260330 ZJH 未找到该规则
    }

    // 20260330 ZJH clearRules — 清空所有规则
    void clearRules() {
        m_vecRules.clear();  // 20260330 ZJH 释放所有规则
    }

    // 20260330 ZJH getRuleCount — 获取当前规则数量
    // 返回: 规则条数
    int getRuleCount() const {
        return static_cast<int>(m_vecRules.size());  // 20260330 ZJH 返回规则数
    }

    // 20260330 ZJH evaluate — 对指标集合进行规则评估
    // 按规则添加顺序逐条评估，第一个匹配的规则决定最终判定
    // NG 优先: 任何一条规则判 NG 则立即返回 NG
    // 若有规则判 Uncertain 但无 NG，返回 Uncertain
    // 若全部不匹配则返回 OK（默认合格）
    // mapMetrics: 指标名称→值 的映射（如 {"area": 120.5, "confidence": 0.85}）
    // 返回: 最终判定结果
    JudgeResult evaluate(const std::map<std::string, float>& mapMetrics) const {
        bool bHasUncertain = false;  // 20260330 ZJH 是否有 Uncertain 匹配

        for (const auto& rule : m_vecRules) {
            // 20260330 ZJH 查找该规则对应的指标值
            auto it = mapMetrics.find(rule.strMetric);  // 20260330 ZJH 在指标映射中查找
            if (it == mapMetrics.end()) {
                continue;  // 20260330 ZJH 指标不存在，跳过此规则
            }

            float fVal = it->second;  // 20260330 ZJH 取出指标值
            bool bMatch = false;  // 20260330 ZJH 是否匹配当前规则

            // 20260330 ZJH 根据运算符类型判断是否匹配
            switch (rule.eOp) {
                case Rule::GT:
                    bMatch = (fVal > rule.fValue1);  // 20260330 ZJH 大于
                    break;
                case Rule::LT:
                    bMatch = (fVal < rule.fValue1);  // 20260330 ZJH 小于
                    break;
                case Rule::GE:
                    bMatch = (fVal >= rule.fValue1);  // 20260330 ZJH 大于等于
                    break;
                case Rule::LE:
                    bMatch = (fVal <= rule.fValue1);  // 20260330 ZJH 小于等于
                    break;
                case Rule::EQ:
                    bMatch = (std::abs(fVal - rule.fValue1) < 1e-6f);  // 20260330 ZJH 等于（浮点容差）
                    break;
                case Rule::NE:
                    bMatch = (std::abs(fVal - rule.fValue1) >= 1e-6f);  // 20260330 ZJH 不等于
                    break;
                case Rule::InRange:
                    bMatch = (fVal >= rule.fValue1 && fVal <= rule.fValue2);  // 20260330 ZJH 范围内 [v1, v2]
                    break;
            }

            if (bMatch) {
                // 20260330 ZJH NG 优先: 任何一条规则判 NG 则立即返回
                if (rule.eOnMatch == JudgeResult::NG) {
                    return JudgeResult::NG;  // 20260330 ZJH 立即判 NG
                }
                // 20260330 ZJH Uncertain 先记录，看后续是否有 NG
                if (rule.eOnMatch == JudgeResult::Uncertain) {
                    bHasUncertain = true;  // 20260330 ZJH 标记有不确定匹配
                }
            }
        }

        // 20260330 ZJH 若有 Uncertain 但无 NG，返回 Uncertain
        if (bHasUncertain) {
            return JudgeResult::Uncertain;
        }

        return JudgeResult::OK;  // 20260330 ZJH 全部规则均未匹配，默认合格
    }

private:
    std::vector<Rule> m_vecRules;  // 20260330 ZJH 规则列表（按添加顺序存储）
};

// =========================================================
// 混合判定器 (Hybrid Judge)
// =========================================================

// 20260330 ZJH HybridJudge — 多源融合判定器
// 综合 AI 判定、传统视觉判定和规则引擎结果，按策略融合输出最终判定
// 四种融合策略:
//   AiPriority:          AI 优先，低置信度时参考传统视觉
//   TraditionalPriority: 传统优先，异常时参考 AI
//   Voting:              投票制（AI + 传统 + 规则三方多数决）
//   Cascade:             级联制（传统初筛 → AI 精判 → 规则兜底）
class HybridJudge {
public:
    // 20260330 ZJH JudgeInput — 混合判定输入
    // 包含 AI、传统视觉、规则引擎三方的独立判定结果和量化指标
    struct JudgeInput {
        JudgeResult eAiResult;                        // 20260330 ZJH AI 模型判定结果
        float fAiConfidence;                          // 20260330 ZJH AI 模型置信度 [0,1]
        JudgeResult eTraditionalResult;               // 20260330 ZJH 传统视觉判定结果
        float fTraditionalScore;                      // 20260330 ZJH 传统视觉评分 [0,1]
        std::map<std::string, float> mapMetrics;      // 20260330 ZJH 量化指标集合（供规则引擎使用）
    };

    // 20260330 ZJH FusionStrategy — 融合策略枚举
    enum class FusionStrategy {
        AiPriority,           // 20260330 ZJH AI 优先，低置信度时参考传统视觉
        TraditionalPriority,  // 20260330 ZJH 传统优先，异常时参考 AI
        Voting,               // 20260330 ZJH 投票制（AI + 传统 + 规则多数决）
        Cascade               // 20260330 ZJH 级联制（传统初筛 → AI 精判 → 规则兜底）
    };

    // 20260330 ZJH judge — 执行混合判定
    // input: 包含 AI、传统、规则三方输入的结构体
    // 返回: 最终融合判定结果 (OK/NG/Uncertain)
    JudgeResult judge(const JudgeInput& input) const {
        // 20260330 ZJH 根据当前策略分发到对应的融合方法
        switch (m_eStrategy) {
            case FusionStrategy::AiPriority:
                return judgeAiPriority(input);         // 20260330 ZJH AI 优先策略
            case FusionStrategy::TraditionalPriority:
                return judgeTraditionalPriority(input); // 20260330 ZJH 传统优先策略
            case FusionStrategy::Voting:
                return judgeVoting(input);             // 20260330 ZJH 投票策略
            case FusionStrategy::Cascade:
                return judgeCascade(input);             // 20260330 ZJH 级联策略
        }
        return JudgeResult::Uncertain;  // 20260330 ZJH 兜底（不应到达）
    }

    // 20260330 ZJH setStrategy — 设置融合策略
    // eStrategy: 四种策略之一
    void setStrategy(FusionStrategy eStrategy) {
        m_eStrategy = eStrategy;  // 20260330 ZJH 更新策略
    }

    // 20260330 ZJH getStrategy — 获取当前融合策略
    // 返回: 当前策略枚举值
    FusionStrategy getStrategy() const {
        return m_eStrategy;  // 20260330 ZJH 返回当前策略
    }

    // 20260330 ZJH setRuleEngine — 设置规则引擎（用于投票和级联策略）
    // engine: 已配置好规则的 RuleEngine 实例
    void setRuleEngine(const RuleEngine& engine) {
        m_ruleEngine = engine;  // 20260330 ZJH 拷贝赋值
        m_bHasRuleEngine = true;  // 20260330 ZJH 标记已设置规则引擎
    }

    // 20260330 ZJH setConfidenceThreshold — 设置 AI 置信度阈值
    // 低于此阈值时 AI 判定不可信，需参考传统视觉
    // fThreshold: 置信度阈值（默认 0.7）
    void setConfidenceThreshold(float fThreshold) {
        m_fConfidenceThreshold = fThreshold;  // 20260330 ZJH 更新阈值
    }

    // 20260330 ZJH setTraditionalScoreThreshold — 设置传统视觉评分阈值
    // 低于此阈值时传统视觉判定不可信
    // fThreshold: 评分阈值（默认 0.6）
    void setTraditionalScoreThreshold(float fThreshold) {
        m_fTraditionalScoreThreshold = fThreshold;  // 20260330 ZJH 更新阈值
    }

private:
    // 20260330 ZJH AI 优先策略实现
    // 逻辑:
    //   1. AI 置信度 >= 阈值 → 直接采用 AI 判定
    //   2. AI 置信度 < 阈值 → 参考传统视觉:
    //      a. 传统和 AI 一致 → 采用
    //      b. 不一致 → 返回 Uncertain
    JudgeResult judgeAiPriority(const JudgeInput& input) const {
        // 20260330 ZJH AI 置信度足够高，直接采信 AI
        if (input.fAiConfidence >= m_fConfidenceThreshold) {
            return input.eAiResult;  // 20260330 ZJH 高置信度，直接用 AI 结果
        }

        // 20260330 ZJH AI 置信度不足，参考传统视觉
        if (input.eAiResult == input.eTraditionalResult) {
            return input.eAiResult;  // 20260330 ZJH 两者一致，采用共同判定
        }

        // 20260330 ZJH 传统视觉评分也足够且判 NG，倾向 NG（宁错杀不放过）
        if (input.eTraditionalResult == JudgeResult::NG
            && input.fTraditionalScore >= m_fTraditionalScoreThreshold) {
            return JudgeResult::NG;  // 20260330 ZJH 传统视觉高分判 NG，采信
        }

        return JudgeResult::Uncertain;  // 20260330 ZJH 无法确定，交人工复判
    }

    // 20260330 ZJH 传统优先策略实现
    // 逻辑:
    //   1. 传统视觉评分 >= 阈值 → 直接采用传统判定
    //   2. 传统视觉评分 < 阈值 → 参考 AI:
    //      a. AI 置信度高 → 采用 AI
    //      b. 否则 → Uncertain
    JudgeResult judgeTraditionalPriority(const JudgeInput& input) const {
        // 20260330 ZJH 传统视觉评分足够高，直接采信
        if (input.fTraditionalScore >= m_fTraditionalScoreThreshold) {
            return input.eTraditionalResult;  // 20260330 ZJH 高评分，直接用传统结果
        }

        // 20260330 ZJH 传统评分不足，参考 AI
        if (input.fAiConfidence >= m_fConfidenceThreshold) {
            return input.eAiResult;  // 20260330 ZJH AI 高置信度，采用 AI 结果
        }

        // 20260330 ZJH 两者均不可信
        if (input.eAiResult == input.eTraditionalResult) {
            return input.eAiResult;  // 20260330 ZJH 虽然分数低但结果一致，采用
        }

        return JudgeResult::Uncertain;  // 20260330 ZJH 无法确定
    }

    // 20260330 ZJH 投票策略实现
    // 逻辑: AI + 传统 + 规则引擎三方投票
    //   - 三方中至少两方判 NG → NG
    //   - 三方中至少两方判 OK → OK
    //   - 其余 → Uncertain
    JudgeResult judgeVoting(const JudgeInput& input) const {
        int nOkVotes = 0;   // 20260330 ZJH OK 票数
        int nNgVotes = 0;   // 20260330 ZJH NG 票数

        // 20260330 ZJH AI 投票
        if (input.eAiResult == JudgeResult::OK) nOkVotes++;       // 20260330 ZJH AI 投 OK
        else if (input.eAiResult == JudgeResult::NG) nNgVotes++;  // 20260330 ZJH AI 投 NG

        // 20260330 ZJH 传统视觉投票
        if (input.eTraditionalResult == JudgeResult::OK) nOkVotes++;       // 20260330 ZJH 传统投 OK
        else if (input.eTraditionalResult == JudgeResult::NG) nNgVotes++;  // 20260330 ZJH 传统投 NG

        // 20260330 ZJH 规则引擎投票（如果已配置）
        if (m_bHasRuleEngine) {
            JudgeResult eRuleResult = m_ruleEngine.evaluate(input.mapMetrics);  // 20260330 ZJH 规则评估
            if (eRuleResult == JudgeResult::OK) nOkVotes++;       // 20260330 ZJH 规则投 OK
            else if (eRuleResult == JudgeResult::NG) nNgVotes++;  // 20260330 ZJH 规则投 NG
        }

        // 20260330 ZJH 多数决
        int nTotalVoters = m_bHasRuleEngine ? 3 : 2;  // 20260330 ZJH 投票人数
        int nMajority = nTotalVoters / 2 + 1;  // 20260330 ZJH 多数所需票数

        if (nNgVotes >= nMajority) {
            return JudgeResult::NG;  // 20260330 ZJH NG 多数，判 NG
        }
        if (nOkVotes >= nMajority) {
            return JudgeResult::OK;  // 20260330 ZJH OK 多数，判 OK
        }

        return JudgeResult::Uncertain;  // 20260330 ZJH 无多数，不确定
    }

    // 20260330 ZJH 级联策略实现
    // 逻辑: 传统初筛 → AI 精判 → 规则兜底
    //   第1级: 传统视觉初筛
    //     - 传统判 NG 且高评分 → 直接 NG（明显缺陷无需 AI）
    //     - 传统判 OK 且高评分 → 进入第2级（可能有微小缺陷）
    //     - 传统低评分 → 进入第2级
    //   第2级: AI 精判
    //     - AI 高置信度 → 采用 AI 判定
    //     - AI 低置信度 → 进入第3级
    //   第3级: 规则兜底
    //     - 有规则引擎 → 用规则判定
    //     - 无规则引擎 → Uncertain
    JudgeResult judgeCascade(const JudgeInput& input) const {
        // ---- 第1级: 传统视觉初筛 ----
        // 20260330 ZJH 传统视觉对明显缺陷判定准确且快速
        if (input.fTraditionalScore >= m_fTraditionalScoreThreshold) {
            if (input.eTraditionalResult == JudgeResult::NG) {
                return JudgeResult::NG;  // 20260330 ZJH 明显 NG，直接拒绝
            }
            // 20260330 ZJH 传统判 OK 但不能完全确定，进入 AI 精判
        }

        // ---- 第2级: AI 精判 ----
        // 20260330 ZJH AI 对微小缺陷、纹理异常等传统方法难以捕捉的场景有优势
        if (input.fAiConfidence >= m_fConfidenceThreshold) {
            return input.eAiResult;  // 20260330 ZJH AI 高置信度，采用 AI 判定
        }

        // ---- 第3级: 规则兜底 ----
        // 20260330 ZJH AI 也不确定时，使用规则引擎根据量化指标做最终判定
        if (m_bHasRuleEngine) {
            JudgeResult eRuleResult = m_ruleEngine.evaluate(input.mapMetrics);  // 20260330 ZJH 规则评估
            if (eRuleResult != JudgeResult::Uncertain) {
                return eRuleResult;  // 20260330 ZJH 规则有明确判定，采用
            }
        }

        // 20260330 ZJH 三级均无法确定，交人工复判
        return JudgeResult::Uncertain;
    }

    FusionStrategy m_eStrategy = FusionStrategy::AiPriority;  // 20260330 ZJH 默认 AI 优先策略
    RuleEngine m_ruleEngine;                                   // 20260330 ZJH 规则引擎实例
    bool m_bHasRuleEngine = false;                             // 20260330 ZJH 是否已配置规则引擎
    float m_fConfidenceThreshold = 0.7f;                       // 20260330 ZJH AI 置信度阈值
    float m_fTraditionalScoreThreshold = 0.6f;                 // 20260330 ZJH 传统视觉评分阈值
};

}  // namespace om
