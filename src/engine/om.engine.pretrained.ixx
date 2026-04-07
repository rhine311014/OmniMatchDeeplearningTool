// 20260330 ZJH 预训练权重管理模块 — 对标 MVTec 12个预训练 .hdl 模型
// 完整功能: 列出可用模型 / 生成预训练权重(.omm) / 加载预训练权重(部分匹配) / 冻结骨干
// 依赖: om.engine.serializer (序列化) + om.engine.resnet/resnet50/mobilenet/vit/unet/efficientad (模型类)
module;

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>       // 20260331 ZJH std::ifstream（PyTorch .omm 直接读取）
#include <filesystem>    // 20260331 ZJH std::filesystem::path（UTF-8 路径）
#include <unordered_map> // 20260331 ZJH 参数名→指针映射
#include <cmath>
#include <algorithm>
#include <cstring>

export module om.engine.pretrained;

// 20260330 ZJH 导入全部依赖模块（pretrained 位于 om_engine_tools，tools 依赖 vision+core）
import om.engine.tensor;
import om.engine.tensor_ops;
import om.engine.module;
import om.engine.serializer;
import om.engine.resnet;
import om.engine.resnet50;
import om.engine.mobilenet;
import om.engine.vit;
import om.engine.unet;
import om.engine.efficientad;
import om.engine.segmodels;  // 20260331 ZJH DeepLabV3 预训练支持

export namespace om {

// 20260330 ZJH 预训练权重元信息
struct PretrainedInfo {
    std::string strModelType;     // 20260330 ZJH 模型类型名
    std::string strDataset;       // 20260330 ZJH 训练数据集
    int nNumClasses;              // 20260330 ZJH 原始类别数
    int nInputSize;               // 20260330 ZJH 标准输入尺寸
    std::string strDescription;   // 20260330 ZJH 模型描述
};

// 20260330 ZJH listAvailablePretrained — 返回所有可用的预训练模型列表
std::vector<PretrainedInfo> listAvailablePretrained() {
    return {
        {"ResNet18", "ImageNet", 1000, 224, "ResNet-18 (11.7M params, balanced)"},
        {"ResNet50", "ImageNet", 1000, 224, "ResNet-50 (25.6M params, high accuracy)"},
        {"MobileNetV4Small", "ImageNet", 1000, 224, "MobileNetV4-Small (3.4M params, edge)"},
        {"ViT", "ImageNet", 1000, 224, "ViT-Tiny (5.7M params, attention-based)"},
        {"UNet", "Random", 2, 256, "UNet (31M params, segmentation)"},
        {"UNet-Light", "Random", 2, 256, "UNet-Light base=32 (7M params)"},
        {"EfficientAD", "Random", 2, 256, "EfficientAD (anomaly detection)"},
        {"ResNet18-Seg", "ImageNet", 1000, 224, "ResNet-18 backbone for segmentation"},
    };
}

// 20260330 ZJH generatePretrainedWeights — 生成预训练权重文件(.omm)
// 创建指定类型的模型实例（构造时已 Kaiming 初始化），序列化为 .omm 文件
// strModelType: 模型类型名（如 "ResNet18"）
// strOutputPath: 输出文件路径
// nNumClasses: 输出类别数（默认 1000 = ImageNet）
// nInChannels: 输入通道数（默认 3 = RGB）
// 返回: 是否成功
bool generatePretrainedWeights(const std::string& strModelType,
                                const std::string& strOutputPath,
                                int nNumClasses = 1000,
                                int nInChannels = 3) {
    // 20260330 ZJH 根据类型创建模型
    std::shared_ptr<Module> pModel;  // 20260330 ZJH 模型指针
    int nBaseChannels = 64;  // 20260330 ZJH 默认基础通道数
    int nInputSize = 224;    // 20260330 ZJH 默认输入尺寸

    if (strModelType == "ResNet18" || strModelType == "ResNet18-Seg") {
        pModel = std::make_shared<ResNet18>(nNumClasses, nInChannels);
    } else if (strModelType == "ResNet50") {
        pModel = std::make_shared<ResNet50>(nNumClasses, nInChannels);
    } else if (strModelType == "MobileNetV4Small") {
        pModel = std::make_shared<MobileNetV4Small>(nNumClasses, nInChannels);
        nBaseChannels = 16;
    } else if (strModelType == "ViT") {
        pModel = std::make_shared<ViT>(224, 16, nInChannels, nNumClasses, 192, 6, 3, 384);
        nBaseChannels = 192;
    } else if (strModelType == "UNet") {
        pModel = std::make_shared<UNet>(nInChannels, nNumClasses, 64);
        nInputSize = 256;
    } else if (strModelType == "UNet-Light") {
        pModel = std::make_shared<UNet>(nInChannels, nNumClasses, 32);
        nBaseChannels = 32; nInputSize = 256;
    } else if (strModelType == "EfficientAD") {
        pModel = std::make_shared<EfficientAD>(nInChannels);
        nInputSize = 256;
    } else {
        std::cerr << "[Pretrained] ERROR: unknown model type '" << strModelType << "'" << std::endl;
        return false;
    }

    // 20260330 ZJH 构造元数据
    ModelMeta meta;  // 20260330 ZJH v4 格式元数据
    meta.strModelType = strModelType;
    meta.nModelTypeHash = ModelMeta::hashString(strModelType);
    meta.nBaseChannels = nBaseChannels;
    meta.nInputSize = nInputSize;
    meta.nNumClasses = nNumClasses;
    meta.nInChannels = nInChannels;

    // 20260330 ZJH 保存为 .omm 文件
    try {
        ModelSerializer::save(*pModel, strOutputPath, meta);
        auto vecParams = pModel->parameters();
        int nTotalParams = 0;
        for (auto* p : vecParams) nTotalParams += p->numel();
        std::cerr << "[Pretrained] generated " << strModelType << " -> " << strOutputPath
                  << " (" << nTotalParams << " params)" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Pretrained] save failed: " << e.what() << std::endl;
        return false;
    }
}

// 20260330 ZJH loadPretrainedWeights — 加载预训练权重到模型（自动跳过 shape 不匹配的层）
// model: 目标模型（已创建，可能类别数不同）
// strWeightPath: .omm 文件路径
// 返回: (加载成功的层数, 跳过的层数)
std::pair<int, int> loadPretrainedWeights(Module& model, const std::string& strWeightPath) {
    int nLoaded = 0, nSkipped = 0;  // 20260330 ZJH 计数器

    try {
        // 20260330 ZJH 先用 peekMeta 获取文件中的模型信息
        ModelMeta fileMeta;
        ModelSerializer::peekMeta(strWeightPath, fileMeta);

        // 20260330 ZJH 创建与文件相同架构的临时模型来加载全部权重
        std::shared_ptr<Module> pTempModel;
        if (fileMeta.strModelType == "ResNet18" || fileMeta.strModelType == "ResNet18-Seg") {
            pTempModel = std::make_shared<ResNet18>(fileMeta.nNumClasses, fileMeta.nInChannels);
        } else if (fileMeta.strModelType == "ResNet50") {
            pTempModel = std::make_shared<ResNet50>(fileMeta.nNumClasses, fileMeta.nInChannels);
        } else if (fileMeta.strModelType == "MobileNetV4Small") {
            pTempModel = std::make_shared<MobileNetV4Small>(fileMeta.nNumClasses, fileMeta.nInChannels);
        } else if (fileMeta.strModelType == "ViT") {
            pTempModel = std::make_shared<ViT>(224, 16, fileMeta.nInChannels, fileMeta.nNumClasses, 192, 6, 3, 384);
        } else if (fileMeta.strModelType == "UNet" || fileMeta.strModelType == "UNet-Light") {
            int nBase = (fileMeta.strModelType == "UNet-Light") ? 32 : 64;
            pTempModel = std::make_shared<UNet>(fileMeta.nInChannels, fileMeta.nNumClasses, nBase);
        } else if (fileMeta.strModelType == "EfficientAD") {
            pTempModel = std::make_shared<EfficientAD>(fileMeta.nInChannels);
        } else {
            std::cerr << "[Pretrained] unknown model type in file: " << fileMeta.strModelType << std::endl;
            return {0, 0};
        }

        // 20260330 ZJH 加载权重到临时模型
        ModelSerializer::load(*pTempModel, strWeightPath);

        // 20260330 ZJH 按名称匹配，将临时模型的参数拷贝到目标模型
        auto vecSrcParams = pTempModel->namedParameters();
        auto vecDstParams = model.namedParameters();

        // 20260330 ZJH 建立目标参数名→指针映射
        std::unordered_map<std::string, Tensor*> mapDst;
        for (auto& [strName, pTensor] : vecDstParams) {
            mapDst[strName] = pTensor;
        }

        // 20260330 ZJH 逐参数匹配拷贝
        for (auto& [strName, pSrcTensor] : vecSrcParams) {
            auto it = mapDst.find(strName);
            if (it == mapDst.end()) {
                nSkipped++;  // 20260330 ZJH 目标模型中无此参数（可能是新增层）
                continue;
            }
            Tensor* pDstTensor = it->second;
            // 20260330 ZJH 检查形状是否匹配
            if (pSrcTensor->numel() != pDstTensor->numel()) {
                nSkipped++;  // 20260330 ZJH 形状不匹配（如 FC 层类别数不同）
                std::cerr << "[Pretrained] skip '" << strName << "' shape mismatch: "
                          << pSrcTensor->numel() << " vs " << pDstTensor->numel() << std::endl;
                continue;
            }
            // 20260330 ZJH 拷贝参数数据
            auto cSrc = pSrcTensor->contiguous();
            std::memcpy(pDstTensor->mutableFloatDataPtr(),
                        cSrc.floatDataPtr(),
                        static_cast<size_t>(cSrc.numel()) * sizeof(float));
            nLoaded++;
        }

        // 20260330 ZJH 同样处理 buffers（BN running stats）
        auto vecSrcBufs = pTempModel->namedBuffers();
        auto vecDstBufs = model.namedBuffers();
        std::unordered_map<std::string, Tensor*> mapDstBuf;
        for (auto& [strName, pTensor] : vecDstBufs) {
            mapDstBuf[strName] = pTensor;
        }
        for (auto& [strName, pSrcTensor] : vecSrcBufs) {
            auto it = mapDstBuf.find(strName);
            if (it == mapDstBuf.end()) { nSkipped++; continue; }
            if (pSrcTensor->numel() != it->second->numel()) { nSkipped++; continue; }
            auto cSrc = pSrcTensor->contiguous();
            std::memcpy(it->second->mutableFloatDataPtr(),
                        cSrc.floatDataPtr(),
                        static_cast<size_t>(cSrc.numel()) * sizeof(float));
            nLoaded++;
        }

        std::cerr << "[Pretrained] loaded " << nLoaded << " layers, skipped " << nSkipped
                  << " from " << strWeightPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Pretrained] load failed: " << e.what() << std::endl;
    }

    return {nLoaded, nSkipped};
}

// 20260330 ZJH generateAllPretrained — 批量生成全部 8 种模型的预训练权重
// strOutputDir: 输出目录路径（如 "pretrained/"）
void generateAllPretrained(const std::string& strOutputDir) {
    auto vecModels = listAvailablePretrained();
    for (const auto& info : vecModels) {
        std::string strPath = strOutputDir + "/" + info.strModelType + ".omm";
        generatePretrainedWeights(info.strModelType, strPath, info.nNumClasses, 3);
    }
}

// 20260330 ZJH freezeBackbone — 冻结骨干网络（前 N 层不更新梯度）
void freezeBackbone(Module& model, int nFreezeUpTo) {
    auto vecParams = model.parameters();
    int nCount = 0;
    for (auto* pParam : vecParams) {
        if (nCount < nFreezeUpTo) {
            tensorSetRequiresGrad(*pParam, false);
        }
        ++nCount;
    }
    std::cerr << "[Pretrained] froze " << std::min(nFreezeUpTo, static_cast<int>(vecParams.size()))
              << "/" << vecParams.size() << " parameters" << std::endl;
}

// 20260330 ZJH unfreezeAll — 解冻所有参数
void unfreezeAll(Module& model) {
    for (auto* pParam : model.parameters()) {
        tensorSetRequiresGrad(*pParam, true);
    }
    std::cerr << "[Pretrained] unfroze all " << model.parameters().size() << " parameters" << std::endl;
}

// 20260331 ZJH ========== 跨架构预训练权重加载 ==========
// 将 PyTorch 导出的 ResNet18 权重加载到 DeepLabV3+ 编码器
// 处理: 名称映射 + BN weight/bias→gamma/beta + downsample→convDs/bnDs
//       + stem 7x7→3x3 center-crop + FC 层跳过

// 20260331 ZJH mapPyTorchNameToOmniMatch — PyTorch ResNet18 参数名 → DeepLabV3 参数名
// 返回空字符串表示跳过该参数（如 fc 层、不匹配的层）
// bIsBuffer: true 表示缓冲区（running_mean/running_var），false 表示参数
std::string mapPyTorchNameToOmniMatch(const std::string& strPyName, bool bIsBuffer) {
    std::string strResult = strPyName;  // 20260331 ZJH 初始复制

    // 20260331 ZJH 跳过 FC 层（DeepLabV3 无分类全连接层）
    if (strResult.find("fc.") == 0) return "";

    // 20260331 ZJH 跳过 num_batches_tracked 缓冲区（OmniMatch 不保存此字段）
    if (strResult.find("num_batches") != std::string::npos) return "";

    // 20260331 ZJH stem 映射: conv1 → stem, bn1(顶层) → bnStem
    // 注意: 仅映射顶层 bn1（非 layer 内部的 bn1）
    if (strResult == "conv1.weight") return "stem.weight";
    if (strResult == "bn1.weight") return "bnStem.gamma";
    if (strResult == "bn1.bias") return "bnStem.beta";
    if (strResult == "bn1.running_mean") return "bnStem.running_mean";
    if (strResult == "bn1.running_var") return "bnStem.running_var";

    // 20260331 ZJH layer 点号→下划线: layer1.0 → layer1_0, layer2.1 → layer2_1
    // PyTorch: "layer1.0.conv1.weight" → OmniMatch: "layer1_0.conv1.weight"
    for (int nLayer = 1; nLayer <= 4; ++nLayer) {
        for (int nBlock = 0; nBlock <= 1; ++nBlock) {
            std::string strPy = "layer" + std::to_string(nLayer) + "."
                              + std::to_string(nBlock) + ".";
            std::string strOm = "layer" + std::to_string(nLayer) + "_"
                              + std::to_string(nBlock) + ".";
            if (strResult.find(strPy) == 0) {
                strResult = strOm + strResult.substr(strPy.size());
                break;
            }
        }
    }

    // 20260331 ZJH downsample 映射: downsample.0 → convDs, downsample.1 → bnDs
    // PyTorch: "layer2_0.downsample.0.weight" → "layer2_0.convDs.weight"
    // PyTorch: "layer2_0.downsample.1.weight" → "layer2_0.bnDs.gamma"
    {
        auto nDsPos = strResult.find(".downsample.0.");
        if (nDsPos != std::string::npos) {
            strResult = strResult.substr(0, nDsPos) + ".convDs."
                      + strResult.substr(nDsPos + 14);  // 14 = strlen(".downsample.0.")
        }
    }
    {
        auto nDsPos = strResult.find(".downsample.1.");
        if (nDsPos != std::string::npos) {
            strResult = strResult.substr(0, nDsPos) + ".bnDs."
                      + strResult.substr(nDsPos + 14);  // 14 = strlen(".downsample.1.")
        }
    }

    // 20260331 ZJH BN 参数名映射: weight → gamma, bias → beta
    // 仅对 BN 层（名称含 bn/bnStem/bnDs）的 weight/bias 做映射
    // Conv 层的 weight/bias 保持不变
    {
        // 20260331 ZJH 检查是否为 BN 层的参数（通过父层名包含 "bn" 判断）
        auto nLastDot = strResult.rfind('.');
        if (nLastDot != std::string::npos) {
            std::string strParent = strResult.substr(0, nLastDot);  // 20260331 ZJH 父层名
            std::string strField = strResult.substr(nLastDot + 1);  // 20260331 ZJH 字段名
            // 20260331 ZJH 提取最末尾的层名（可能是嵌套的，如 "layer1_0.bn1"）
            auto nParentLastDot = strParent.rfind('.');
            std::string strLayerName = (nParentLastDot != std::string::npos)
                ? strParent.substr(nParentLastDot + 1) : strParent;
            // 20260331 ZJH 判断是否为 BN 层（名称以 "bn" 或 "Bn" 开头）
            bool bIsBnLayer = (strLayerName.find("bn") == 0 || strLayerName.find("Bn") == 0
                            || strLayerName.find("bN") == 0);
            if (bIsBnLayer) {
                if (strField == "weight") {
                    strResult = strParent + ".gamma";
                } else if (strField == "bias") {
                    strResult = strParent + ".beta";
                }
                // running_mean / running_var 保持不变
            }
        }
    }

    return strResult;
}

// 20260331 ZJH centerCrop7x7to3x3 — 从 PyTorch 7x7 卷积核中提取中心 3x3 区域
// PyTorch ResNet18 的 conv1 为 [64, 3, 7, 7]，OmniMatch DeepLabV3 的 stem 为 [64, 3, 3, 3]
// 提取每个 filter 的中心 3x3 patch（偏移 [2:5, 2:5]），保留主要响应区域
// pSrc: 源 7x7 数据 [Cout, Cin, 7, 7]
// pDst: 目标 3x3 数据 [Cout, Cin, 3, 3]
// nCout: 输出通道数
// nCin: 输入通道数
void centerCrop7x7to3x3(const float* pSrc, float* pDst, int nCout, int nCin) {
    for (int co = 0; co < nCout; ++co) {
        for (int ci = 0; ci < nCin; ++ci) {
            // 20260331 ZJH 源 7x7 kernel 在内存中的偏移
            const float* pKernel = pSrc + (co * nCin + ci) * 49;  // 49 = 7*7
            // 20260331 ZJH 目标 3x3 kernel 在内存中的偏移
            float* pOut = pDst + (co * nCin + ci) * 9;  // 9 = 3*3
            // 20260331 ZJH 提取中心 3x3: 行 [2,3,4], 列 [2,3,4]
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    pOut[r * 3 + c] = pKernel[(r + 2) * 7 + (c + 2)];
                }
            }
        }
    }
}

// 20260331 ZJH loadPyTorchPretrainedToSegModel — 从 PyTorch 导出的 .omm 加载权重到分割模型
// 直接读取 .omm 文件中的 PyTorch 命名权重，通过名称映射加载到 DeepLabV3+
// 支持: stem 7x7→3x3 center-crop, BN weight→gamma, downsample→convDs
// model: 目标分割模型（DeepLabV3 等）
// strWeightPath: PyTorch 导出的 .omm v4 文件路径
// 返回: (成功加载数, 跳过数)
std::pair<int, int> loadPyTorchPretrainedToSegModel(
    Module& model, const std::string& strWeightPath)
{
    int nLoaded = 0, nSkipped = 0;  // 20260331 ZJH 计数器

    try {
        // 20260331 ZJH 验证文件元数据
        ModelMeta fileMeta;
        bool bHasMeta = ModelSerializer::peekMeta(strWeightPath, fileMeta);
        if (bHasMeta) {
            std::cerr << "[Pretrained] PyTorch .omm meta: type=" << fileMeta.strModelType
                      << " classes=" << fileMeta.nNumClasses
                      << " input=" << fileMeta.nInputSize << std::endl;
        }

        // 20260331 ZJH 直接读取 .omm 文件中的所有命名张量
        // 不创建临时模型，直接从文件读取 PyTorch 命名的权重
        std::filesystem::path fsPath(
            reinterpret_cast<const char8_t*>(strWeightPath.c_str()));
        std::ifstream ifs(fsPath, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open pretrained file: " + strWeightPath);
        }

        // 20260331 ZJH 跳过 magic + version
        ifs.seekg(8);

        // 20260331 ZJH 读取 v4 元数据段（跳过）
        uint32_t nMetaCount = 0;
        ifs.read(reinterpret_cast<char*>(&nMetaCount), sizeof(uint32_t));
        ifs.seekg(static_cast<std::streamoff>(nMetaCount * sizeof(float)), std::ios::cur);

        // 20260331 ZJH 构建目标模型的参数和缓冲区映射
        auto vecDstParams = model.namedParameters();
        auto vecDstBufs = model.namedBuffers();
        std::unordered_map<std::string, Tensor*> mapDstParams;
        std::unordered_map<std::string, Tensor*> mapDstBufs;
        for (auto& [strName, pTensor] : vecDstParams) mapDstParams[strName] = pTensor;
        for (auto& [strName, pTensor] : vecDstBufs) mapDstBufs[strName] = pTensor;

        std::cerr << "[Pretrained] target model: " << mapDstParams.size()
                  << " params, " << mapDstBufs.size() << " buffers" << std::endl;

        // 20260331 ZJH ===== 读取并映射参数 =====
        uint32_t nNumParams = 0;
        ifs.read(reinterpret_cast<char*>(&nNumParams), sizeof(uint32_t));
        std::cerr << "[Pretrained] source file: " << nNumParams << " params" << std::endl;

        for (uint32_t p = 0; p < nNumParams; ++p) {
            // 20260331 ZJH 读取张量名
            uint32_t nNameLen = 0;
            ifs.read(reinterpret_cast<char*>(&nNameLen), sizeof(uint32_t));
            std::string strPyName(nNameLen, '\0');
            ifs.read(strPyName.data(), static_cast<std::streamsize>(nNameLen));

            // 20260331 ZJH 读取维度
            uint32_t nNumDims = 0;
            ifs.read(reinterpret_cast<char*>(&nNumDims), sizeof(uint32_t));
            std::vector<int> vecShape(nNumDims);
            for (uint32_t d = 0; d < nNumDims; ++d) {
                uint32_t nDim = 0;
                ifs.read(reinterpret_cast<char*>(&nDim), sizeof(uint32_t));
                vecShape[d] = static_cast<int>(nDim);
            }
            int nNumel = 1;
            for (int s : vecShape) nNumel *= s;

            // 20260331 ZJH 读取张量数据到临时缓冲区
            std::vector<float> vecData(static_cast<size_t>(nNumel));
            ifs.read(reinterpret_cast<char*>(vecData.data()),
                     static_cast<std::streamsize>(nNumel * sizeof(float)));

            // 20260331 ZJH 名称映射
            std::string strOmName = mapPyTorchNameToOmniMatch(strPyName, false);
            if (strOmName.empty()) {
                nSkipped++;  // 20260331 ZJH FC 层等无对应目标
                std::cerr << "[Pretrained] skip param '" << strPyName << "' (no target)" << std::endl;
                continue;
            }

            // 20260331 ZJH 查找目标参数
            auto it = mapDstParams.find(strOmName);
            if (it == mapDstParams.end()) {
                nSkipped++;
                std::cerr << "[Pretrained] skip param '" << strPyName << "' -> '"
                          << strOmName << "' (not found in target)" << std::endl;
                continue;
            }

            Tensor* pDst = it->second;

            // 20260331 ZJH 特殊处理: stem 7x7 → 3x3 center-crop
            if (strOmName == "stem.weight" && vecShape.size() == 4
                && vecShape[2] == 7 && vecShape[3] == 7) {
                int nCout = vecShape[0];  // 20260331 ZJH 输出通道数 (64)
                int nCin = vecShape[1];   // 20260331 ZJH 输入通道数 (3)
                // 20260331 ZJH 检查目标是 3x3
                if (pDst->shape(2) == 3 && pDst->shape(3) == 3
                    && pDst->shape(0) == nCout && pDst->shape(1) == nCin) {
                    centerCrop7x7to3x3(vecData.data(), pDst->mutableFloatDataPtr(), nCout, nCin);
                    nLoaded++;
                    std::cerr << "[Pretrained] loaded '" << strPyName << "' -> '" << strOmName
                              << "' [7x7 center-crop -> 3x3]" << std::endl;
                    continue;
                }
            }

            // 20260331 ZJH 标准形状匹配拷贝
            if (pDst->numel() != nNumel) {
                nSkipped++;
                std::cerr << "[Pretrained] skip param '" << strPyName << "' -> '" << strOmName
                          << "' shape mismatch: " << nNumel << " vs " << pDst->numel() << std::endl;
                continue;
            }

            std::memcpy(pDst->mutableFloatDataPtr(), vecData.data(),
                        static_cast<size_t>(nNumel) * sizeof(float));
            nLoaded++;
        }

        // 20260331 ZJH ===== 读取并映射缓冲区（BN running stats）=====
        uint32_t nNumBuffers = 0;
        ifs.read(reinterpret_cast<char*>(&nNumBuffers), sizeof(uint32_t));
        std::cerr << "[Pretrained] source file: " << nNumBuffers << " buffers" << std::endl;

        for (uint32_t b = 0; b < nNumBuffers; ++b) {
            // 20260331 ZJH 读取缓冲区名和数据
            uint32_t nNameLen = 0;
            ifs.read(reinterpret_cast<char*>(&nNameLen), sizeof(uint32_t));
            std::string strPyName(nNameLen, '\0');
            ifs.read(strPyName.data(), static_cast<std::streamsize>(nNameLen));

            uint32_t nNumDims = 0;
            ifs.read(reinterpret_cast<char*>(&nNumDims), sizeof(uint32_t));
            int nNumel = 1;
            for (uint32_t d = 0; d < nNumDims; ++d) {
                uint32_t nDim = 0;
                ifs.read(reinterpret_cast<char*>(&nDim), sizeof(uint32_t));
                nNumel *= static_cast<int>(nDim);
            }
            std::vector<float> vecData(static_cast<size_t>(nNumel));
            ifs.read(reinterpret_cast<char*>(vecData.data()),
                     static_cast<std::streamsize>(nNumel * sizeof(float)));

            // 20260331 ZJH 名称映射（缓冲区也走相同映射逻辑）
            std::string strOmName = mapPyTorchNameToOmniMatch(strPyName, true);
            if (strOmName.empty()) { nSkipped++; continue; }

            auto it = mapDstBufs.find(strOmName);
            if (it == mapDstBufs.end()) {
                nSkipped++;
                std::cerr << "[Pretrained] skip buffer '" << strPyName << "' -> '"
                          << strOmName << "' (not found)" << std::endl;
                continue;
            }
            if (it->second->numel() != nNumel) {
                nSkipped++;
                continue;
            }

            std::memcpy(it->second->mutableFloatDataPtr(), vecData.data(),
                        static_cast<size_t>(nNumel) * sizeof(float));
            nLoaded++;
        }

        ifs.close();

        std::cerr << "[Pretrained] PyTorch->OmniMatch: loaded " << nLoaded
                  << " tensors, skipped " << nSkipped << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Pretrained] PyTorch load failed: " << e.what() << std::endl;
    }

    return {nLoaded, nSkipped};
}

}  // namespace om
