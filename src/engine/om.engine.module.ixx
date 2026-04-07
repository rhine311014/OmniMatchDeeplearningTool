// 20260319 ZJH Module 基类模块 — Phase 2 Part 1
// 实现 PyTorch nn.Module 风格的模块基类，支持参数管理、子模块注册、训练/评估模式切换
// Sequential 容器支持链式前向传播
// 20260406 ZJH 全局模块片段（module; 之后、export module 之前），用于引入传统头文件
module;

#include <vector>   // 20260406 ZJH 动态数组容器，用于存储参数列表、子模块列表、缓冲区列表
#include <string>   // 20260406 ZJH 字符串类型，用于参数名称、子模块名称等标识符
#include <memory>   // 20260406 ZJH 智能指针 shared_ptr，用于子模块的共享所有权管理
#include <utility>  // 20260406 ZJH std::pair，用于构建 {名称, 指针} 键值对

// 20260406 ZJH 声明本文件为 C++20 模块接口单元，模块名 om.engine.module
// 20260406 ZJH 提供 Module 基类和 Sequential 容器，是整个引擎神经网络层次结构的根基
export module om.engine.module;

// 20260319 ZJH 导入张量类和张量运算模块（含自动微分接口）
import om.engine.tensor;
import om.engine.tensor_ops;

// 20260406 ZJH om 命名空间，OmniMatch 引擎所有公开符号的顶层命名空间
export namespace om {

// 20260319 ZJH Module — 模块基类，所有神经网络层继承此类
// 管理参数（Tensor*）和子模块（shared_ptr<Module>），支持递归遍历
class Module {
public:
    // 20260319 ZJH 虚析构，支持多态删除
    virtual ~Module() = default;

    // 20260319 ZJH forward — 纯虚前向传播接口，子类必须实现
    // input: 输入张量
    // 返回: 前向计算输出张量
    virtual Tensor forward(const Tensor& input) = 0;

    // 20260319 ZJH registerParameter — 注册模块参数，自动设置 requiresGrad=true
    // strName: 参数名称（用于标识和调试）
    // param: 参数张量的引用，实际存储在子类成员变量中
    void registerParameter(const std::string& strName, Tensor& param) {
        tensorSetRequiresGrad(param, true);  // 20260319 ZJH 自动开启梯度追踪
        m_vecParameters.push_back({strName, &param});  // 20260319 ZJH 保存名称和指针对
    }

    // 20260326 ZJH registerBuffer — 注册非梯度缓冲区（如 BN running stats）
    // 独立于 parameters()，不影响优化器
    // 通过 buffers() 方法收集，GPU 迁移时单独处理
    void registerBuffer(const std::string& strName, Tensor& buf) {
        m_vecBuffers.push_back({strName, &buf});  // 20260406 ZJH 保存缓冲区名称和指针对到列表
    }

    // 20260326 ZJH buffers — 递归收集本模块及所有子模块的缓冲区指针
    // 返回不参与梯度更新但需要随设备迁移的张量（如 BN running stats）
    virtual std::vector<Tensor*> buffers() {
        std::vector<Tensor*> vecResult;  // 20260406 ZJH 结果容器，存储所有缓冲区的原始指针
        // 20260406 ZJH 遍历本模块直接注册的缓冲区，strName 为名称（此处未使用），pBuf 为指针
        for (auto& [strName, pBuf] : m_vecBuffers) {
            vecResult.push_back(pBuf);  // 20260406 ZJH 添加缓冲区指针到结果
        }
        // 20260406 ZJH 遍历所有子模块，递归收集子模块的缓冲区
        for (auto& [strName, pChild] : m_vecChildren) {
            auto vecChildBufs = pChild->buffers();  // 20260406 ZJH 子模块递归调用 buffers()
            vecResult.insert(vecResult.end(), vecChildBufs.begin(), vecChildBufs.end());  // 20260406 ZJH 合并子模块缓冲区到结果
        }
        return vecResult;  // 20260406 ZJH 返回本模块及所有子模块的全部缓冲区指针
    }

    // 20260327 ZJH namedBuffers — 递归收集本模块及所有子模块的命名缓冲区
    // 返回: {名称, Tensor*} 对的向量，名称带层级前缀（如 "bn1.running_mean"）
    // 用于序列化保存 BN running stats 等非训练状态张量
    // 20260328 ZJH 增加 fallback 机制，与 namedParameters() 一致
    virtual std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") {
        std::vector<std::pair<std::string, Tensor*>> vecResult;  // 20260406 ZJH 结果容器，存储 {全限定名, 指针} 对
        // 20260327 ZJH 收集本模块直接注册的缓冲区
        for (auto& [strName, pBuf] : m_vecBuffers) {
            // 20260406 ZJH 构建全限定名：若有前缀则拼接 "prefix.name"，否则直接使用名称
            std::string strFullName = strPrefix.empty() ? strName : strPrefix + "." + strName;
            vecResult.push_back({strFullName, pBuf});  // 20260406 ZJH 添加命名缓冲区到结果
        }
        // 20260327 ZJH 递归收集子模块的命名缓冲区
        for (auto& [strName, pChild] : m_vecChildren) {
            // 20260406 ZJH 构建子模块前缀：将子模块名称追加到当前前缀后
            std::string strChildPrefix = strPrefix.empty() ? strName : strPrefix + "." + strName;
            auto vecChildBufs = pChild->namedBuffers(strChildPrefix);  // 20260406 ZJH 子模块递归调用
            vecResult.insert(vecResult.end(), vecChildBufs.begin(), vecChildBufs.end());  // 20260406 ZJH 合并子模块结果
        }

        // 20260328 ZJH Fallback: 模型重写了 buffers() 但未重写 namedBuffers()
        // 20260406 ZJH 如果递归收集为空，尝试从 buffers() 虚调用获取并自动编号命名
        if (vecResult.empty()) {
            auto vecBufs = buffers();  // 20260406 ZJH virtual dispatch 到子类重写版本
            // 20260406 ZJH 如果子类 buffers() 返回了非空结果，则自动生成编号命名
            if (!vecBufs.empty()) {
                vecResult.reserve(vecBufs.size());  // 20260406 ZJH 预分配空间，避免多次扩容
                // 20260406 ZJH 遍历所有缓冲区，生成 "buffer_0", "buffer_1", ... 编号名称
                for (size_t i = 0; i < vecBufs.size(); ++i) {
                    std::string strName = strPrefix.empty()
                        ? ("buffer_" + std::to_string(i))
                        : (strPrefix + ".buffer_" + std::to_string(i));
                    vecResult.push_back({strName, vecBufs[i]});  // 20260406 ZJH 添加自动命名的缓冲区
                }
            }
        }

        return vecResult;  // 20260406 ZJH 返回所有命名缓冲区列表
    }

    // 20260319 ZJH parameters — 递归收集本模块及所有子模块的参数指针
    // 返回: 所有参数的 Tensor* 向量，用于传给优化器
    // 声明为 virtual 以支持自定义参数收集（如 ResNet 的 BasicBlock）
    virtual std::vector<Tensor*> parameters() {
        std::vector<Tensor*> vecResult;  // 20260319 ZJH 结果容器
        // 20260319 ZJH 收集本模块直接注册的参数
        for (auto& [strName, pParam] : m_vecParameters) {
            vecResult.push_back(pParam);  // 20260319 ZJH 添加参数指针
        }
        // 20260319 ZJH 递归收集所有子模块的参数
        for (auto& [strName, pChild] : m_vecChildren) {
            auto vecChildParams = pChild->parameters();  // 20260319 ZJH 子模块递归调用
            vecResult.insert(vecResult.end(), vecChildParams.begin(), vecChildParams.end());  // 20260319 ZJH 合并到结果
        }
        return vecResult;  // 20260319 ZJH 返回所有参数指针
    }

    // 20260319 ZJH namedParameters — 递归收集本模块及所有子模块的命名参数
    // 返回: {名称, Tensor*} 对的向量，名称带层级前缀（如 "layer_0.weight"）
    // 声明为 virtual 以支持自定义命名参数收集（如 ResNet 的 BasicBlock）
    // 20260328 ZJH 增加 fallback 机制：若递归收集返回空但 parameters() 非空，
    //              自动从 parameters() 生成 "param_N" 编号命名，解决大量模型
    //              只重写 parameters() 未重写 namedParameters() 导致序列化空文件的 bug
    virtual std::vector<std::pair<std::string, Tensor*>> namedParameters(const std::string& strPrefix = "") {
        std::vector<std::pair<std::string, Tensor*>> vecResult;  // 20260406 ZJH 结果容器，存储 {全限定名, 指针} 对
        // 20260319 ZJH 收集本模块直接注册的参数
        for (auto& [strName, pParam] : m_vecParameters) {
            // 20260406 ZJH 构建全限定名：若有前缀则拼接 "prefix.name"，否则直接使用名称
            std::string strFullName = strPrefix.empty() ? strName : strPrefix + "." + strName;
            vecResult.push_back({strFullName, pParam});  // 20260406 ZJH 添加命名参数到结果
        }
        // 20260319 ZJH 递归收集子模块的命名参数
        for (auto& [strName, pChild] : m_vecChildren) {
            // 20260406 ZJH 构建子模块前缀：将子模块名称追加到当前前缀后
            std::string strChildPrefix = strPrefix.empty() ? strName : strPrefix + "." + strName;
            auto vecChildParams = pChild->namedParameters(strChildPrefix);  // 20260406 ZJH 子模块递归调用
            vecResult.insert(vecResult.end(), vecChildParams.begin(), vecChildParams.end());  // 20260406 ZJH 合并子模块结果
        }

        // 20260328 ZJH Fallback: 模型重写了 parameters() 但未重写 namedParameters()
        // 此时 m_vecParameters/m_vecChildren 为空，递归收集返回空列表
        // 但 parameters()（virtual dispatch 到子类）可能返回非空
        // 自动生成 "param_0", "param_1", ... 编号命名，确保序列化正常工作
        // 20260406 ZJH 如果递归收集为空，尝试从 parameters() 虚调用获取并自动编号命名
        if (vecResult.empty()) {
            auto vecParams = parameters();  // 20260328 ZJH virtual call → 子类重写版本
            // 20260406 ZJH 如果子类 parameters() 返回了非空结果，则自动生成编号命名
            if (!vecParams.empty()) {
                vecResult.reserve(vecParams.size());  // 20260406 ZJH 预分配空间，避免多次扩容
                // 20260406 ZJH 遍历所有参数，生成 "param_0", "param_1", ... 编号名称
                for (size_t i = 0; i < vecParams.size(); ++i) {
                    std::string strName = strPrefix.empty()
                        ? ("param_" + std::to_string(i))
                        : (strPrefix + ".param_" + std::to_string(i));
                    vecResult.push_back({strName, vecParams[i]});  // 20260406 ZJH 添加自动命名的参数
                }
            }
        }

        return vecResult;  // 20260406 ZJH 返回所有命名参数列表
    }

    // 20260319 ZJH registerModule — 注册子模块，用于递归参数管理和训练模式切换
    // strName: 子模块名称
    // pModule: 子模块的 shared_ptr
    void registerModule(const std::string& strName, std::shared_ptr<Module> pModule) {
        m_vecChildren.push_back({strName, pModule});  // 20260319 ZJH 保存名称和子模块对
    }

    // 20260319 ZJH train — 设置训练模式，递归传播到所有子模块
    // bMode: true 为训练模式，false 为评估模式
    // 声明为 virtual 以支持自定义训练模式切换（如 ResNet 的 BasicBlock）
    virtual void train(bool bMode = true) {
        m_bTraining = bMode;  // 20260319 ZJH 设置本模块的训练标志
        // 20260319 ZJH 递归设置所有子模块的训练模式
        for (auto& [strName, pChild] : m_vecChildren) {
            pChild->train(bMode);  // 20260319 ZJH 子模块递归调用
        }
    }

    // 20260319 ZJH eval — 设置评估模式（等价于 train(false)）
    void eval() {
        train(false);  // 20260319 ZJH 调用 train(false)
    }

    // 20260319 ZJH isTraining — 查询当前是否处于训练模式
    bool isTraining() const {
        return m_bTraining;  // 20260319 ZJH 返回训练标志
    }

    // 20260328 ZJH debugChildCount — 诊断用：返回直接子模块数量
    // 20260406 ZJH 返回: m_vecChildren 的元素数量，不递归统计孙模块
    size_t debugChildCount() const {
        return m_vecChildren.size();  // 20260406 ZJH 返回子模块列表大小
    }

    // 20260328 ZJH debugParamCount — 诊断用：返回直接注册参数数量
    // 20260406 ZJH 返回: m_vecParameters 的元素数量，不包含子模块的参数
    size_t debugParamCount() const {
        return m_vecParameters.size();  // 20260406 ZJH 返回参数列表大小
    }

    // 20260319 ZJH zeroGrad — 清零所有参数的梯度，递归包含子模块参数
    void zeroGrad() {
        // 20260319 ZJH 获取所有参数指针并逐个清零梯度
        for (auto* pParam : parameters()) {
            tensorZeroGrad(*pParam);  // 20260319 ZJH 调用 tensor_ops 的梯度清零接口
        }
    }

protected:
    bool m_bTraining = true;  // 20260319 ZJH 训练模式标志，默认为训练模式
    // 20260319 ZJH 本模块直接注册的参数列表（名称 + Tensor 指针）
    std::vector<std::pair<std::string, Tensor*>> m_vecParameters;
    // 20260326 ZJH 缓冲区列表（BN running stats 等非梯度张量，需随设备迁移）
    std::vector<std::pair<std::string, Tensor*>> m_vecBuffers;
    // 20260319 ZJH 子模块列表（名称 + shared_ptr<Module>）
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> m_vecChildren;
};

// 20260319 ZJH Sequential — 顺序容器，按添加顺序依次执行各子模块的 forward
class Sequential : public Module {
public:
    // 20260319 ZJH add — 添加一个子模块到顺序容器末尾
    // pModule: 要添加的子模块
    void add(std::shared_ptr<Module> pModule) {
        // 20260319 ZJH 以 "layer_N" 为名称注册子模块
        registerModule("layer_" + std::to_string(m_vecChildren.size()), pModule);
        m_vecLayers.push_back(pModule);  // 20260319 ZJH 同时保存到有序层列表
    }

    // 20260319 ZJH forward — 按添加顺序依次执行各层的前向传播
    // input: 初始输入张量
    // 返回: 最后一层的输出张量
    Tensor forward(const Tensor& input) override {
        Tensor x = input;  // 20260319 ZJH 初始化中间结果为输入
        // 20260319 ZJH 逐层前向传播
        for (auto& pLayer : m_vecLayers) {
            x = pLayer->forward(x);  // 20260319 ZJH 当前层输出作为下一层输入
        }
        return x;  // 20260319 ZJH 返回最终输出
    }

private:
    // 20260319 ZJH 有序层列表，保持添加顺序
    std::vector<std::shared_ptr<Module>> m_vecLayers;
};

}  // namespace om  // 20260406 ZJH 结束 om 命名空间
