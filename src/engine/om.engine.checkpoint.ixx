// 20260320 ZJH 模型检查点模块 — Phase 6
// 自动保存最佳模型和定期检查点，支持训练恢复
module;

#include <string>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <vector>

export module om.engine.checkpoint;

import om.engine.tensor;
import om.engine.module;

export namespace om {

// 20260320 ZJH CheckpointManager — 检查点管理器
// 负责训练过程中的模型保存和恢复
class CheckpointManager {
public:
    // 20260320 ZJH 构造函数
    // strDir: 检查点保存目录
    // nSaveEveryN: 每 N 个 epoch 保存一次
    // bSaveBest: 是否保存最佳模型
    CheckpointManager(const std::string& strDir = "data/checkpoints",
                       int nSaveEveryN = 10, bool bSaveBest = true)
        : m_strDir(strDir), m_nSaveEveryN(nSaveEveryN), m_bSaveBest(bSaveBest)
    {
        std::filesystem::create_directories(m_strDir);
    }

    // 20260320 ZJH saveCheckpoint — 保存模型检查点
    // model: 要保存的模型
    // strName: 检查点名称（如 "best" 或 "epoch_10"）
    // nEpoch: 当前 epoch
    // fLoss: 当前损失
    // fMetric: 当前指标（准确率/mIoU 等）
    // 返回: 保存的文件路径
    std::string saveCheckpoint(Module& model, const std::string& strName,
                                int nEpoch, float fLoss, float fMetric) {
        std::string strPath = m_strDir + "/" + strName + ".dfckpt";
        std::ofstream file(strPath, std::ios::binary);
        if (!file.is_open()) return "";

        // 20260320 ZJH 写入魔数和版本
        const char magic[] = "DFCKPT01";
        file.write(magic, 8);

        // 20260320 ZJH 写入训练状态
        int32_t epoch = static_cast<int32_t>(nEpoch);
        file.write(reinterpret_cast<const char*>(&epoch), 4);
        file.write(reinterpret_cast<const char*>(&fLoss), 4);
        file.write(reinterpret_cast<const char*>(&fMetric), 4);

        // 20260320 ZJH 写入时间戳
        auto now = std::chrono::system_clock::now();
        int64_t nTimestamp = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();
        file.write(reinterpret_cast<const char*>(&nTimestamp), 8);

        // 20260320 ZJH 写入模型参数
        auto vecNamedParams = model.namedParameters();
        int32_t nParamCount = static_cast<int32_t>(vecNamedParams.size());
        file.write(reinterpret_cast<const char*>(&nParamCount), 4);

        for (const auto& [name, pTensor] : vecNamedParams) {
            auto ct = pTensor->contiguous();
            auto vecShape = ct.shapeVec();
            int nNumel = ct.numel();

            // 20260320 ZJH 参数名
            int32_t nNameLen = static_cast<int32_t>(name.size());
            file.write(reinterpret_cast<const char*>(&nNameLen), 4);
            file.write(name.c_str(), nNameLen);

            // 20260320 ZJH 形状
            int32_t nDims = static_cast<int32_t>(vecShape.size());
            file.write(reinterpret_cast<const char*>(&nDims), 4);
            for (auto s : vecShape) {
                int32_t d = static_cast<int32_t>(s);
                file.write(reinterpret_cast<const char*>(&d), 4);
            }

            // 20260320 ZJH 数据
            int32_t nBytes = static_cast<int32_t>(nNumel * 4);
            file.write(reinterpret_cast<const char*>(&nBytes), 4);
            file.write(reinterpret_cast<const char*>(ct.floatDataPtr()), nBytes);
        }

        file.close();
        m_strLastSavedPath = strPath;
        return strPath;
    }

    // 20260320 ZJH onEpochEnd — 每个 epoch 结束时调用
    // 根据配置决定是否保存检查点
    // 返回: 保存的文件路径（如未保存则为空）
    std::string onEpochEnd(Module& model, int nEpoch, float fLoss, float fMetric) {
        std::string strSaved;

        // 20260320 ZJH 保存最佳模型
        if (m_bSaveBest && fLoss < m_fBestLoss) {
            m_fBestLoss = fLoss;
            strSaved = saveCheckpoint(model, "best", nEpoch, fLoss, fMetric);
        }

        // 20260320 ZJH 定期保存
        if (m_nSaveEveryN > 0 && (nEpoch + 1) % m_nSaveEveryN == 0) {
            std::string strName = "epoch_" + std::to_string(nEpoch + 1);
            saveCheckpoint(model, strName, nEpoch, fLoss, fMetric);
        }

        return strSaved;
    }

    // 20260320 ZJH loadCheckpoint — 从检查点恢复模型参数
    // model: 要恢复的模型（结构必须与保存时一致）
    // strPath: 检查点文件路径
    // 返回: {epoch, loss, metric} 训练状态，失败返回 {-1, 0, 0}
    struct CheckpointInfo {
        int nEpoch = -1;
        float fLoss = 0.0f;
        float fMetric = 0.0f;
    };

    static CheckpointInfo loadCheckpoint(Module& model, const std::string& strPath) {
        CheckpointInfo info;
        std::ifstream file(strPath, std::ios::binary);
        if (!file.is_open()) return info;

        // 20260320 ZJH 验证魔数
        char magic[8];
        file.read(magic, 8);
        if (std::string(magic, 8) != "DFCKPT01") return info;

        // 20260320 ZJH 读取训练状态
        int32_t epoch;
        file.read(reinterpret_cast<char*>(&epoch), 4);
        file.read(reinterpret_cast<char*>(&info.fLoss), 4);
        file.read(reinterpret_cast<char*>(&info.fMetric), 4);
        info.nEpoch = epoch;

        // 20260320 ZJH 跳过时间戳
        int64_t ts;
        file.read(reinterpret_cast<char*>(&ts), 8);

        // 20260320 ZJH 读取参数
        int32_t nParamCount;
        file.read(reinterpret_cast<char*>(&nParamCount), 4);

        // 20260320 ZJH 建立模型参数名到指针的映射
        auto vecNamedParams = model.namedParameters();
        std::vector<std::pair<std::string, Tensor*>> mapParams(
            vecNamedParams.begin(), vecNamedParams.end());

        for (int32_t i = 0; i < nParamCount; ++i) {
            // 20260320 ZJH 读取参数名
            int32_t nNameLen;
            file.read(reinterpret_cast<char*>(&nNameLen), 4);
            std::string strName(static_cast<size_t>(nNameLen), '\0');
            file.read(strName.data(), nNameLen);

            // 20260320 ZJH 读取形状
            int32_t nDims;
            file.read(reinterpret_cast<char*>(&nDims), 4);
            std::vector<int> vecShape(static_cast<size_t>(nDims));
            for (int32_t d = 0; d < nDims; ++d) {
                int32_t dim;
                file.read(reinterpret_cast<char*>(&dim), 4);
                vecShape[static_cast<size_t>(d)] = dim;
            }

            // 20260320 ZJH 读取数据
            int32_t nBytes;
            file.read(reinterpret_cast<char*>(&nBytes), 4);
            int nNumel = nBytes / 4;

            // 20260320 ZJH 查找对应参数并恢复
            for (auto& [name, pTensor] : mapParams) {
                if (name == strName && pTensor->numel() == nNumel) {
                    file.read(reinterpret_cast<char*>(pTensor->mutableFloatDataPtr()), nBytes);
                    break;
                }
            }
            // 20260320 ZJH 如果没找到匹配参数，跳过数据
            if (!file.good()) break;
        }

        return info;
    }

    // 20260320 ZJH getLastSavedPath — 获取最后保存的路径
    const std::string& lastSavedPath() const { return m_strLastSavedPath; }

    // 20260320 ZJH getBestLoss — 获取最佳损失
    float bestLoss() const { return m_fBestLoss; }

    // 20260320 ZJH listCheckpoints — 列出所有检查点文件
    std::vector<std::string> listCheckpoints() const {
        std::vector<std::string> vecFiles;
        if (!std::filesystem::exists(m_strDir)) return vecFiles;
        for (const auto& entry : std::filesystem::directory_iterator(m_strDir)) {
            if (entry.path().extension() == ".dfckpt") {
                vecFiles.push_back(entry.path().string());
            }
        }
        return vecFiles;
    }

private:
    std::string m_strDir;               // 20260320 ZJH 检查点目录
    int m_nSaveEveryN;                  // 20260320 ZJH 定期保存间隔
    bool m_bSaveBest;                   // 20260320 ZJH 是否保存最佳
    float m_fBestLoss = 1e30f;          // 20260320 ZJH 最佳损失
    std::string m_strLastSavedPath;     // 20260320 ZJH 最后保存路径
};

}  // namespace om
