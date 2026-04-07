// 20260330 ZJH ModelRegistry — 模型注册表实现
// 集中管理所有模型类型的工厂函数和元信息
// 新增模型只需在 registerAllModels() 中添加一行注册调用

// 20260330 ZJH 标准库头文件（必须在 import 之前，MSVC C++23 模块兼容性要求）
#include <iostream>    // 20260330 ZJH std::cerr 错误日志
#include <algorithm>   // 20260330 ZJH std::transform (未来可能需要)
#include <mutex>       // 20260330 ZJH FIX-5: std::mutex + std::lock_guard for ensureInitialized

// 20260330 ZJH 包含注册表头文件
#include "engine/bridge/ModelRegistry.h"

// 20260330 ZJH 导入 OmniMatch 引擎 C++23 模块（与 EngineBridge.cpp 相同的 import 列表）
// 每个模块提供对应的模型类定义
import om.engine.tensor;
import om.engine.module;
import om.engine.linear;
import om.engine.activations;
import om.engine.conv;
import om.engine.resnet;
import om.engine.resnet50;
import om.engine.mobilenet;
import om.engine.vit;
import om.engine.unet;
import om.engine.segmodels;
import om.engine.yolo;
import om.engine.instance_seg;
import om.engine.efficientad;
import om.engine.gcad;              // 20260402 ZJH GCAD 全局上下文异常检测
import om.engine.edge_extraction;  // 20260402 ZJH DL 边缘提取（对标 Halcon）
import om.engine.dinomaly;         // 20260402 ZJH Dinomaly (CVPR 2025 SOTA AD)
import om.engine.retrieval;        // 20260402 ZJH DL 图像检索
import om.engine.sam;              // 20260402 ZJH MobileSAM 无监督分割
import om.engine.convnext;         // 20260402 ZJH ConvNeXt-Tiny 现代 CNN 分类
import om.engine.sam2_unet;        // 20260402 ZJH SAM2-UNet 分割
import om.engine.dbnet;   // 20260402 ZJH DBNet 可微分二值化文本检测
import om.engine.crnn;    // 20260402 ZJH CRNN 文本识别

// =====================================================================
// 20260330 ZJH SimpleMLP 分类器定义（从 EngineBridge.cpp 迁移）
// 3 层全连接网络: inputDim -> 256 -> 128 -> numClasses
// 唯一不需要 4D 输入的模型（接受展平向量 [B, C*H*W]）
// =====================================================================
class SimpleMLP : public om::Module {
public:
    // 20260330 ZJH 构造函数
    // 参数: nInputDim - 展平输入维度 (3*H*W); nNumClasses - 输出类别数
    SimpleMLP(int nInputDim, int nNumClasses)
        : m_fc1(nInputDim, 256), m_fc2(256, 128), m_fc3(128, nNumClasses)
    {
        // 20260330 ZJH 注册子模块以便 parameters() 能遍历到所有权重
        registerModule("fc1", std::make_shared<om::Linear>(m_fc1));
        registerModule("fc2", std::make_shared<om::Linear>(m_fc2));
        registerModule("fc3", std::make_shared<om::Linear>(m_fc3));
    }

    // 20260330 ZJH 前向传播: fc1 -> relu -> fc2 -> relu -> fc3
    om::Tensor forward(const om::Tensor& input) override {
        auto x = m_fc1.forward(input);   // 20260330 ZJH 第一层全连接
        x = m_relu.forward(x);           // 20260330 ZJH ReLU 激活
        x = m_fc2.forward(x);            // 20260330 ZJH 第二层全连接
        x = m_relu.forward(x);           // 20260330 ZJH ReLU 激活
        x = m_fc3.forward(x);            // 20260330 ZJH 输出层（无激活，由 loss 函数处理 softmax）
        return x;
    }

private:
    om::Linear m_fc1, m_fc2, m_fc3;  // 20260330 ZJH 三层全连接层
    om::ReLU m_relu;                  // 20260330 ZJH 共享 ReLU 激活（无状态，可复用）
};

// =====================================================================
// 20260330 ZJH 内部函数: 注册所有内置模型
// 集中定义所有模型的工厂函数、元信息和别名
// 新增模型步骤:
//   1. 在此函数中添加 reg.registerModel(...) 调用
//   2. 如有别名，添加 reg.registerAlias(...) 调用
//   3. 无需修改 EngineBridge.cpp
// =====================================================================
static void registerAllModels(om::ModelRegistry& reg)
{
    // ===== 分类模型 =====

    // 20260330 ZJH MLP — 简单全连接分类器（唯一非 CNN 模型）
    // 工厂参数: nNumClasses, nInChannels(忽略), nInputSize
    // 展平输入维度 = nInChannels * nInputSize * nInputSize
    reg.registerModel("MLP",
        [](int nNumClasses, int nInChannels, int nInputSize) -> std::shared_ptr<om::Module> {
            // 20260330 ZJH 计算展平维度: 3 * H * W（MLP 不使用 4D 输入）
            int nInputDim = nInChannels * nInputSize * nInputSize;
            return std::make_shared<SimpleMLP>(nInputDim, nNumClasses);
        },
        {"MLP", om::ModelCategory::Classification, /*bIsCnn=*/false, /*nDefaultBaseChannels=*/0});
    // 20260330 ZJH "SimpleMLP" 是 MLP 的别名（旧版兼容）
    reg.registerAlias("SimpleMLP", "MLP");

    // 20260330 ZJH ResNet18 — 经典残差网络 18 层
    reg.registerModel("ResNet18",
        [](int nNumClasses, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::ResNet18>(nNumClasses);
        },
        {"ResNet18", om::ModelCategory::Classification, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // 20260330 ZJH ResNet50 — 残差网络 50 层（Bottleneck 结构）
    reg.registerModel("ResNet50",
        [](int nNumClasses, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::ResNet50>(nNumClasses);
        },
        {"ResNet50", om::ModelCategory::Classification, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // 20260330 ZJH MobileNetV4Small — 轻量级移动端分类模型
    reg.registerModel("MobileNetV4Small",
        [](int nNumClasses, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::MobileNetV4Small>(nNumClasses);
        },
        {"MobileNetV4Small", om::ModelCategory::Classification, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/32});

    // 20260330 ZJH ViTTiny — Vision Transformer (Tiny 配置)
    // patch_size=16, embed_dim=192, depth=12, heads=3
    reg.registerModel("ViTTiny",
        [](int nNumClasses, int nInChannels, int nInputSize) -> std::shared_ptr<om::Module> {
            // 20260330 ZJH ViT 需要 inputSize 计算 patch 数量
            return std::make_shared<om::ViT>(
                nInputSize, 16, nInChannels, nNumClasses, 192, 12, 3);
        },
        {"ViTTiny", om::ModelCategory::Classification, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/192});

    // 20260402 ZJH ConvNeXtTiny — 现代化纯 CNN（Liu et al., 2022，ImageNet Top-1 82.1%）
    // 4 Stage [3,3,9,3]，通道 [96,192,384,768]，DW-Conv7x7+LN+GELU
    reg.registerModel("ConvNeXtTiny",
        [](int nNumClasses, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::ConvNeXtTiny>(nNumClasses, nInChannels > 0 ? nInChannels : 3);
        },
        {"ConvNeXtTiny", om::ModelCategory::Classification, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/96});
    // 20260402 ZJH "ConvNeXt" 是 "ConvNeXtTiny" 的别名
    reg.registerAlias("ConvNeXt", "ConvNeXtTiny");

    // ===== 目标检测模型 =====

    // 20260330 ZJH YOLOv5Nano — 超轻量 YOLO 检测模型
    reg.registerModel("YOLOv5Nano",
        [](int nNumClasses, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::YOLOv5Nano>(nNumClasses);
        },
        {"YOLOv5Nano", om::ModelCategory::Detection, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/16});

    // 20260330 ZJH YOLOv8Nano — YOLOv8 Nano 检测模型
    reg.registerModel("YOLOv8Nano",
        [](int nNumClasses, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::YOLOv8Nano>(nNumClasses);
        },
        {"YOLOv8Nano", om::ModelCategory::Detection, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/16});

    // ===== 语义分割模型 =====

    // 20260330 ZJH UNet — 经典 U 形编解码分割网络
    // 根据输入尺寸自动选择通道数: >384 用 base=32（轻量），≤384 用 base=64（标准）
    reg.registerModel("UNet",
        [](int nNumClasses, int nInChannels, int nInputSize) -> std::shared_ptr<om::Module> {
            // 20260330 ZJH 大尺寸（>384）用轻量模型避免显存爆满和训练过慢
            int nBase = (nInputSize > 384) ? 32 : 64;
            return std::make_shared<om::UNet>(nInChannels, nNumClasses, nBase);
        },
        {"UNet", om::ModelCategory::Segmentation, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // 20260330 ZJH DeepLabV3+ — 带 ASPP 的分割模型
    reg.registerModel("DeepLabV3+",
        [](int nNumClasses, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::DeepLabV3>(nInChannels, nNumClasses);
        },
        {"DeepLabV3+", om::ModelCategory::Segmentation, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});
    // 20260330 ZJH 别名: "DeepLabV3Plus" 和 "DeepLabV3" 均映射到 "DeepLabV3+"
    reg.registerAlias("DeepLabV3Plus", "DeepLabV3+");
    reg.registerAlias("DeepLabV3", "DeepLabV3+");

    // 20260401 ZJH MobileSegNet — 轻量级工业分割网络（对标海康 ASI_SEG）
    // MobileNetV4-Small 编码器 + ASPPLite(3分支) + 低级特征融合解码器
    // 参数量 ~1.75M（vs DeepLabV3+ ~26M），推理速度 3-5 倍提升
    reg.registerModel("MobileSegNet",
        [](int nNumClasses, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::MobileSegNet>(nInChannels, nNumClasses);
        },
        {"MobileSegNet", om::ModelCategory::Segmentation, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/16});
    // 20260401 ZJH "MobileSeg" 是 "MobileSegNet" 的别名
    reg.registerAlias("MobileSeg", "MobileSegNet");

    // ===== 异常检测模型 =====

    // 20260330 ZJH EfficientAD — 高效异常检测（Teacher-Student 蒸馏 + 自编码器）
    reg.registerModel("EfficientAD",
        [](int /*nNumClasses*/, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::EfficientAD>();
        },
        {"EfficientAD", om::ModelCategory::AnomalyDetection, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // 20260402 ZJH GCAD — 全局上下文异常检测（布局+纹理双分支，对标 Halcon GCAD）
    reg.registerModel("GCAD",
        [](int /*nNumClasses*/, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::GCAD>();  // 20260402 ZJH 默认 3 通道 + 128 维嵌入
        },
        {"GCAD", om::ModelCategory::AnomalyDetection, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});
    reg.registerAlias("GlobalContextAD", "GCAD");  // 20260402 ZJH 别名

    // ===== OCR 模型 =====

    // 20260402 ZJH DBNet — 可微分二值化文本检测
    reg.registerModel("DBNet",
        [](int /*nNumClasses*/, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::DBNet>(nInChannels > 0 ? nInChannels : 3);
        },
        {"DBNet", om::ModelCategory::OCR, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // 20260402 ZJH CRNN — 文本识别（95 类 = ASCII 可打印 + blank）
    reg.registerModel("CRNN",
        [](int nNumClasses, int /*nInChannels*/, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            int nClasses = (nNumClasses > 0) ? nNumClasses : 96;  // 20260402 ZJH 默认 Industrial 字符集 (95+1blank)
            return std::make_shared<om::CRNN>(nClasses);
        },
        {"CRNN", om::ModelCategory::OCR, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});
    reg.registerAlias("DBNet+CRNN", "DBNet");  // 20260402 ZJH 端到端 OCR 别名（DBNet 检测 + CRNN 识别）

    // ===== DL 边缘提取模型 =====

    // 20260402 ZJH EdgeUNet — 轻量 U-Net 边缘提取（对标 Halcon Edge Extraction）
    reg.registerModel("EdgeUNet",
        [](int /*nNumClasses*/, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::EdgeExtractionNet>(nInChannels > 0 ? nInChannels : 3, 32);
        },
        {"EdgeUNet", om::ModelCategory::Segmentation, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/32});
    reg.registerAlias("EdgeExtraction", "EdgeUNet");

    // ===== CVPR 2025 前沿架构 =====

    // 20260402 ZJH Dinomaly — CVPR 2025 SOTA 异常检测 (99.6% AUROC MVTec AD)
    reg.registerModel("Dinomaly",
        [](int /*nNumClasses*/, int nInChannels, int nInputSize) -> std::shared_ptr<om::Module> {
            int nSize = (nInputSize > 0) ? nInputSize : 224;
            return std::make_shared<om::Dinomaly>(nInChannels > 0 ? nInChannels : 3, nSize);
        },
        {"Dinomaly", om::ModelCategory::AnomalyDetection, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/192});

    // 20260402 ZJH SAM2-UNet — SAM2 Hiera 编码器 + U-Net 解码器（分割精度 +20%）
    reg.registerModel("SAM2-UNet",
        [](int nNumClasses, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            return std::make_shared<om::SAM2UNet>(nInChannels > 0 ? nInChannels : 3, nNumClasses > 0 ? nNumClasses : 2);
        },
        {"SAM2-UNet", om::ModelCategory::Segmentation, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});
    reg.registerAlias("SAM2UNet", "SAM2-UNet");

    // ===== DL 图像检索 =====

    // 20260402 ZJH ImageRetrievalNet — 特征嵌入 + 余弦相似度搜索（对标海康 VisionMaster）
    reg.registerModel("ImageRetrieval",
        [](int nNumClasses, int nInChannels, int nInputSize) -> std::shared_ptr<om::Module> {
            // 20260402 ZJH 复用 ResNet18 作为特征提取 backbone，输出 512 维嵌入
            return std::make_shared<om::ResNet18>(nInChannels > 0 ? nInChannels : 3,
                                                   nNumClasses > 0 ? nNumClasses : 512);
        },
        {"ImageRetrieval", om::ModelCategory::Classification, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // ===== DL 无监督分割 =====

    // 20260402 ZJH MobileSAM — Segment Anything Model 轻量版（无需标注，自动分区）
    reg.registerModel("MobileSAM",
        [](int /*nNumClasses*/, int nInChannels, int nInputSize) -> std::shared_ptr<om::Module> {
            int nSize = (nInputSize > 0) ? nInputSize : 256;
            return std::make_shared<om::SAM>(nInChannels > 0 ? nInChannels : 3, nSize);
        },
        {"MobileSAM", om::ModelCategory::Segmentation, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});

    // ===== 实例分割模型 =====

    // 20260330 ZJH YOLOv8Seg — YOLO 实例分割（YOLACT 风格: 检测 + 原型 mask）
    reg.registerModel("YOLOv8Seg",
        [](int nNumClasses, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            // 20260330 ZJH SimpleInstanceSeg(inChannels=3, numClasses, numPrototypes=32)
            return std::make_shared<om::SimpleInstanceSeg>(nInChannels, nNumClasses, 32);
        },
        {"YOLOv8Seg", om::ModelCategory::InstanceSeg, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/16});
    // 20260330 ZJH "YOLOv8InstanceSeg" 是 "YOLOv8Seg" 的别名
    reg.registerAlias("YOLOv8InstanceSeg", "YOLOv8Seg");

    // 20260330 ZJH MaskRCNN — Mask R-CNN 实例分割
    reg.registerModel("MaskRCNN",
        [](int nNumClasses, int nInChannels, int /*nInputSize*/) -> std::shared_ptr<om::Module> {
            // 20260330 ZJH 当前实现复用 SimpleInstanceSeg
            return std::make_shared<om::SimpleInstanceSeg>(nInChannels, nNumClasses, 32);
        },
        {"MaskRCNN", om::ModelCategory::InstanceSeg, /*bIsCnn=*/true, /*nDefaultBaseChannels=*/64});
}

// =====================================================================
// 20260330 ZJH ModelRegistry 成员函数实现
// =====================================================================

namespace om {

// 20260330 ZJH 获取全局唯一实例（Meyers' Singleton）
ModelRegistry& ModelRegistry::instance()
{
    static ModelRegistry s_instance;  // 20260330 ZJH C++11 起静态局部变量初始化线程安全
    return s_instance;
}

// 20260330 ZJH 注册模型类型
void ModelRegistry::registerModel(const std::string& strType, ModelFactory factory, ModelInfo info)
{
    m_mapFactories[strType] = std::move(factory);  // 20260330 ZJH 存储工厂函数
    m_mapInfos[strType] = std::move(info);          // 20260330 ZJH 存储元信息
}

// 20260330 ZJH 注册别名
void ModelRegistry::registerAlias(const std::string& strAlias, const std::string& strPrimaryType)
{
    m_mapAliases[strAlias] = strPrimaryType;  // 20260330 ZJH 别名指向主名称
}

// 20260330 ZJH 解析别名: 如果是别名则返回主名称，否则原样返回
std::string ModelRegistry::resolveName(const std::string& strType) const
{
    auto it = m_mapAliases.find(strType);  // 20260330 ZJH 查找别名表
    if (it != m_mapAliases.end()) {
        return it->second;  // 20260330 ZJH 返回别名对应的主名称
    }
    return strType;  // 20260330 ZJH 不是别名，原样返回
}

// 20260330 ZJH 创建模型实例（返回 shared_ptr<void> 以避免 C++23 模块前向声明冲突）
std::shared_ptr<void> ModelRegistry::createModel(const std::string& strType, int nNumClasses,
                                                    int nInChannels, int nInputSize) const
{
    std::string strResolved = resolveName(strType);  // 20260330 ZJH 解析别名
    auto it = m_mapFactories.find(strResolved);      // 20260330 ZJH 查找工厂函数
    if (it == m_mapFactories.end()) {
        return nullptr;  // 20260330 ZJH 未注册的模型类型
    }
    return it->second(nNumClasses, nInChannels, nInputSize);  // 20260330 ZJH 调用工厂函数创建模型
}

// 20260330 ZJH 查询模型元信息
const ModelInfo* ModelRegistry::getModelInfo(const std::string& strType) const
{
    std::string strResolved = resolveName(strType);  // 20260330 ZJH 解析别名
    auto it = m_mapInfos.find(strResolved);          // 20260330 ZJH 查找元信息
    if (it == m_mapInfos.end()) {
        return nullptr;  // 20260330 ZJH 未注册的类型
    }
    return &(it->second);  // 20260330 ZJH 返回元信息指针
}

// 20260330 ZJH 获取所有已注册的主名称列表
std::vector<std::string> ModelRegistry::getRegisteredTypes() const
{
    std::vector<std::string> vecTypes;
    vecTypes.reserve(m_mapFactories.size());  // 20260330 ZJH 预分配避免多次扩容
    for (const auto& [strKey, factory] : m_mapFactories) {
        vecTypes.push_back(strKey);  // 20260330 ZJH 收集主名称（不含别名）
    }
    return vecTypes;
}

// 20260330 ZJH 检查模型类型是否已注册
bool ModelRegistry::isRegistered(const std::string& strType) const
{
    std::string strResolved = resolveName(strType);     // 20260330 ZJH 解析别名
    return m_mapFactories.count(strResolved) > 0;       // 20260330 ZJH 查找工厂表
}

// 20260330 ZJH 确保所有内置模型已注册（懒初始化，首次调用时执行）
// 20260330 ZJH FIX-5: 加 mutex 保护，防止多线程同时首次调用时重复注册
void ModelRegistry::ensureInitialized()
{
    static std::mutex s_initMutex;  // 20260330 ZJH FIX-5: 静态互斥锁，保护初始化过程
    std::lock_guard<std::mutex> lock(s_initMutex);  // 20260330 ZJH FIX-5: RAII 锁
    if (m_bInitialized) {
        return;  // 20260330 ZJH 已初始化，跳过
    }
    registerAllModels(*this);  // 20260330 ZJH 调用全局注册函数
    m_bInitialized = true;     // 20260330 ZJH 标记完成
}

}  // namespace om
