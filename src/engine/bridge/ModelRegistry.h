// 20260330 ZJH ModelRegistry — 模型注册表，替代 EngineBridge 中 200+ 行 if-else 字符串匹配链
// 采用静态注册表模式: 每种模型类型注册工厂函数 + 元信息
// 新增模型只需在 ModelRegistry.cpp 的 registerAllModels() 中添加一行，无需修改 EngineBridge.cpp
#pragma once

#include <string>       // 20260330 ZJH std::string
#include <functional>   // 20260330 ZJH std::function 工厂函数类型
#include <unordered_map>// 20260330 ZJH 哈希表存储注册项
#include <memory>       // 20260330 ZJH std::shared_ptr
#include <vector>       // 20260330 ZJH std::vector 返回已注册类型列表
#include <mutex>        // 20260330 ZJH FIX-5: std::mutex ensureInitialized 线程安全

namespace om {

// 20260330 ZJH 注意: 不前向声明 om::Module
// MSVC C++23 模块中前向声明与 module export 的类型不兼容（C2556）
// 改用 void 基类指针，由 .cpp 中 import 后做 static_pointer_cast

// 20260330 ZJH 模型任务类别枚举
// 用于替代 EngineSessionImpl 中的 bIsDetection/bIsSegmentation 等布尔标志组合
// EngineBridge 内部仍保留 bool flags（train/infer 路径依赖），但通过 eCategory 统一设置
enum class ModelCategory {
    Classification,    // 20260330 ZJH 分类: ResNet18, ResNet50, MobileNetV4Small, ViTTiny, MLP
    AnomalyDetection,  // 20260330 ZJH 异常检测: EfficientAD
    Segmentation,      // 20260330 ZJH 语义分割: UNet, DeepLabV3/DeepLabV3+
    Detection,         // 20260330 ZJH 目标检测: YOLOv5Nano, YOLOv8Nano
    InstanceSeg,       // 20260330 ZJH 实例分割: YOLOv8Seg, MaskRCNN
    OCR                // 20260330 ZJH 文字识别: CRNN (预留)
};

// 20260330 ZJH 模型元信息结构体
// 包含模型类型标识、任务类别、输入模式、默认通道数等
// 供 EngineBridge 在创建模型后自动设置 EngineSessionImpl 的标志位
struct ModelInfo {
    std::string strType;           // 20260330 ZJH 模型类型主名称（如 "ResNet18"）
    ModelCategory eCategory;       // 20260330 ZJH 任务类别（分类/检测/分割等）
    bool bIsCnn;                   // 20260330 ZJH 是否为 CNN（true=需要 4D 输入 [B,C,H,W]）
    int nDefaultBaseChannels;      // 20260330 ZJH 默认基础通道数（序列化匹配用）
};

// 20260330 ZJH 模型工厂函数签名
// 参数: nNumClasses - 类别数; nInChannels - 输入通道数（默认 3=RGB）; nInputSize - 空间尺寸
// 返回: 创建好的 Module 共享指针; 返回 nullptr 表示创建失败
// 注意: nInputSize 参数用于 ViT（需要计算 patch 数量）和 UNet（根据尺寸选择通道数）等
//       对不依赖 inputSize 的模型（如 ResNet）可忽略该参数
using ModelFactory = std::function<std::shared_ptr<void>(int nNumClasses, int nInChannels, int nInputSize)>;

// 20260330 ZJH ModelRegistry — 单例模式的模型注册表
// 职责: 维护 modelType -> (factory, info) 的映射
// 线程安全: 注册阶段在程序初始化时完成（单线程），查询阶段只读（天然线程安全）
class ModelRegistry {
public:
    // 20260330 ZJH 获取全局唯一实例（Meyers' Singleton，C++11 起线程安全）
    static ModelRegistry& instance();

    // 20260330 ZJH 注册模型类型
    // 参数: strType - 模型类型名称（主名称，如 "ResNet18"）
    //       factory - 工厂函数
    //       info - 模型元信息
    void registerModel(const std::string& strType, ModelFactory factory, ModelInfo info);

    // 20260330 ZJH 注册别名（映射到已注册的主名称）
    // 例: "SimpleMLP" -> "MLP", "DeepLabV3Plus" -> "DeepLabV3+"
    void registerAlias(const std::string& strAlias, const std::string& strPrimaryType);

    // 20260330 ZJH 创建模型实例（替代 if-else 链）
    // 参数: strType - 模型类型名称（支持主名称和别名）
    //       nNumClasses - 类别数
    //       nInChannels - 输入通道数（默认 3）
    //       nInputSize - 输入空间尺寸（默认 224）
    // 返回: 模型指针; nullptr 表示未注册的类型
    std::shared_ptr<void> createModel(const std::string& strType, int nNumClasses,
                                         int nInChannels = 3, int nInputSize = 224) const;

    // 20260330 ZJH 查询模型元信息
    // 返回: 指向 ModelInfo 的指针; nullptr 表示未注册
    const ModelInfo* getModelInfo(const std::string& strType) const;

    // 20260330 ZJH 获取所有已注册的主模型类型名称
    std::vector<std::string> getRegisteredTypes() const;

    // 20260330 ZJH 检查模型类型是否已注册（支持别名）
    bool isRegistered(const std::string& strType) const;

    // 20260330 ZJH 确保所有内置模型已注册（首次调用时自动执行）
    void ensureInitialized();

private:
    // 20260330 ZJH 私有构造函数（单例模式）
    ModelRegistry() = default;

    // 20260330 ZJH 解析别名，返回主名称
    // 如果 strType 是别名则返回对应主名称，否则原样返回
    std::string resolveName(const std::string& strType) const;

    // 20260330 ZJH 工厂函数映射: 主名称 -> 工厂函数
    std::unordered_map<std::string, ModelFactory> m_mapFactories;

    // 20260330 ZJH 模型元信息映射: 主名称 -> ModelInfo
    std::unordered_map<std::string, ModelInfo> m_mapInfos;

    // 20260330 ZJH 别名映射: 别名 -> 主名称
    std::unordered_map<std::string, std::string> m_mapAliases;

    // 20260330 ZJH 初始化标记（避免重复注册）
    bool m_bInitialized = false;
};

}  // namespace om
