// 20260319 ZJH Module 基类模块 — Phase 2 Part 1
// 实现 PyTorch nn.Module 风格的模块基类，支持参数管理、子模块注册、训练/评估模式切换
// Sequential 容器支持链式前向传播
module;

#include <vector>
#include <string>
#include <memory>
#include <utility>

export module om.engine.module;

// 20260319 ZJH 导入张量类和张量运算模块（含自动微分接口）
import om.engine.tensor;
import om.engine.tensor_ops;

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
        m_vecBuffers.push_back({strName, &buf});
    }

    // 20260326 ZJH buffers — 递归收集本模块及所有子模块的缓冲区指针
    // 返回不参与梯度更新但需要随设备迁移的张量（如 BN running stats）
    virtual std::vector<Tensor*> buffers() {
        std::vector<Tensor*> vecResult;
        for (auto& [strName, pBuf] : m_vecBuffers) {
            vecResult.push_back(pBuf);
        }
        for (auto& [strName, pChild] : m_vecChildren) {
            auto vecChildBufs = pChild->buffers();
            vecResult.insert(vecResult.end(), vecChildBufs.begin(), vecChildBufs.end());
        }
        return vecResult;
    }

    // 20260327 ZJH namedBuffers — 递归收集本模块及所有子模块的命名缓冲区
    // 返回: {名称, Tensor*} 对的向量，名称带层级前缀（如 "bn1.running_mean"）
    // 用于序列化保存 BN running stats 等非训练状态张量
    // 20260328 ZJH 增加 fallback 机制，与 namedParameters() 一致
    virtual std::vector<std::pair<std::string, Tensor*>> namedBuffers(const std::string& strPrefix = "") {
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        // 20260327 ZJH 收集本模块直接注册的缓冲区
        for (auto& [strName, pBuf] : m_vecBuffers) {
            std::string strFullName = strPrefix.empty() ? strName : strPrefix + "." + strName;
            vecResult.push_back({strFullName, pBuf});
        }
        // 20260327 ZJH 递归收集子模块的命名缓冲区
        for (auto& [strName, pChild] : m_vecChildren) {
            std::string strChildPrefix = strPrefix.empty() ? strName : strPrefix + "." + strName;
            auto vecChildBufs = pChild->namedBuffers(strChildPrefix);
            vecResult.insert(vecResult.end(), vecChildBufs.begin(), vecChildBufs.end());
        }

        // 20260328 ZJH Fallback: 模型重写了 buffers() 但未重写 namedBuffers()
        if (vecResult.empty()) {
            auto vecBufs = buffers();
            if (!vecBufs.empty()) {
                vecResult.reserve(vecBufs.size());
                for (size_t i = 0; i < vecBufs.size(); ++i) {
                    std::string strName = strPrefix.empty()
                        ? ("buffer_" + std::to_string(i))
                        : (strPrefix + ".buffer_" + std::to_string(i));
                    vecResult.push_back({strName, vecBufs[i]});
                }
            }
        }

        return vecResult;
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
        std::vector<std::pair<std::string, Tensor*>> vecResult;
        // 20260319 ZJH 收集本模块直接注册的参数
        for (auto& [strName, pParam] : m_vecParameters) {
            std::string strFullName = strPrefix.empty() ? strName : strPrefix + "." + strName;
            vecResult.push_back({strFullName, pParam});
        }
        // 20260319 ZJH 递归收集子模块的命名参数
        for (auto& [strName, pChild] : m_vecChildren) {
            std::string strChildPrefix = strPrefix.empty() ? strName : strPrefix + "." + strName;
            auto vecChildParams = pChild->namedParameters(strChildPrefix);
            vecResult.insert(vecResult.end(), vecChildParams.begin(), vecChildParams.end());
        }

        // 20260328 ZJH Fallback: 模型重写了 parameters() 但未重写 namedParameters()
        // 此时 m_vecParameters/m_vecChildren 为空，递归收集返回空列表
        // 但 parameters()（virtual dispatch 到子类）可能返回非空
        // 自动生成 "param_0", "param_1", ... 编号命名，确保序列化正常工作
        if (vecResult.empty()) {
            auto vecParams = parameters();  // 20260328 ZJH virtual call → 子类重写版本
            if (!vecParams.empty()) {
                vecResult.reserve(vecParams.size());
                for (size_t i = 0; i < vecParams.size(); ++i) {
                    std::string strName = strPrefix.empty()
                        ? ("param_" + std::to_string(i))
                        : (strPrefix + ".param_" + std::to_string(i));
                    vecResult.push_back({strName, vecParams[i]});
                }
            }
        }

        return vecResult;
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
    size_t debugChildCount() const {
        return m_vecChildren.size();
    }

    // 20260328 ZJH debugParamCount — 诊断用：返回直接注册参数数量
    size_t debugParamCount() const {
        return m_vecParameters.size();
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

}  // namespace om
