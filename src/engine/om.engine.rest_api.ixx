// 20260330 ZJH REST API 推理服务模块 — HTTP 接口暴露推理能力
// 便于 MES/SCADA/上位机集成，支持健康检查/单图推理/批量推理/模型信息
// 配方管理 (RecipeManager): 多模型切换 + 持久化
module;

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>

export module om.engine.rest_api;

// 20260330 ZJH 导入依赖模块：张量类、模块基类
import om.engine.tensor;
import om.engine.module;

export namespace om {

// =========================================================
// API 配置
// =========================================================

// 20260330 ZJH ApiConfig — REST API 服务配置
struct ApiConfig {
    int nPort = 8080;                    // 20260330 ZJH 监听端口
    std::string strHost = "0.0.0.0";     // 20260330 ZJH 绑定地址（0.0.0.0 监听所有网卡）
    int nMaxConcurrent = 4;              // 20260330 ZJH 最大并发推理数
    bool bEnableCORS = true;             // 20260330 ZJH 是否启用跨域访问（Web 前端集成需要）
};

// =========================================================
// HTTP 请求/响应
// =========================================================

// 20260330 ZJH HttpRequest — 简化 HTTP 请求结构
// 由外部 HTTP 服务器解析后填充，传入 InferenceServer 处理
struct HttpRequest {
    std::string strMethod;    // 20260330 ZJH 请求方法: GET/POST/OPTIONS
    std::string strPath;      // 20260330 ZJH 请求路径: /api/infer, /api/health 等
    std::string strBody;      // 20260330 ZJH 请求体: JSON 格式
    std::map<std::string, std::string> mapHeaders;  // 20260330 ZJH 请求头键值对
};

// 20260330 ZJH HttpResponse — 简化 HTTP 响应结构
// 由 InferenceServer 填充后，交给外部 HTTP 服务器发送
struct HttpResponse {
    int nStatusCode = 200;                          // 20260330 ZJH HTTP 状态码
    std::string strBody;                             // 20260330 ZJH 响应体（通常为 JSON）
    std::string strContentType = "application/json"; // 20260330 ZJH Content-Type 头
};

// =========================================================
// 推理服务器
// =========================================================

// 20260330 ZJH InferenceServer — REST API 推理服务核心
// 路由注册:
//   GET  /api/health     → 健康检查
//   POST /api/infer      → 单图推理（Base64 图像）
//   POST /api/batch      → 批量推理
//   GET  /api/model/info → 模型信息
//   GET  /api/doc        → API 文档
// 设计为同步处理，可嵌入 Qt 事件循环或独立线程
class InferenceServer {
public:
    // 20260330 ZJH 构造函数 — 初始化服务配置
    // config: API 配置（端口、地址、并发数、CORS）
    InferenceServer(const ApiConfig& config = {})
        : m_config(config), m_pModel(nullptr), m_nRequestCount(0) {
        // 20260330 ZJH 配置保存完毕，等待 setModel 和 handleRequest
    }

    // 20260330 ZJH setModel — 设置推理模型
    // pModel: 已加载的模型指针（shared_ptr 管理生命周期）
    void setModel(std::shared_ptr<Module> pModel) {
        m_pModel = pModel;  // 20260330 ZJH 保存模型引用
    }

    // 20260330 ZJH handleRequest — 处理 HTTP 请求（同步）
    // request: 解析后的 HTTP 请求
    // 返回: HTTP 响应
    HttpResponse handleRequest(const HttpRequest& request) {
        m_nRequestCount++;  // 20260330 ZJH 请求计数递增

        // 20260330 ZJH CORS 预检请求处理
        if (request.strMethod == "OPTIONS" && m_config.bEnableCORS) {
            return makeCorsResponse();  // 20260330 ZJH 返回 CORS 预检响应
        }

        // 20260330 ZJH 路由分发
        if (request.strPath == "/api/health" && request.strMethod == "GET") {
            return handleHealth();  // 20260330 ZJH 健康检查
        }
        if (request.strPath == "/api/infer" && request.strMethod == "POST") {
            return handleInfer(request);  // 20260330 ZJH 单图推理
        }
        if (request.strPath == "/api/batch" && request.strMethod == "POST") {
            return handleBatch(request);  // 20260330 ZJH 批量推理
        }
        if (request.strPath == "/api/model/info" && request.strMethod == "GET") {
            return handleModelInfo();  // 20260330 ZJH 模型信息
        }
        if (request.strPath == "/api/doc" && request.strMethod == "GET") {
            return handleDoc();  // 20260330 ZJH API 文档
        }

        // 20260330 ZJH 未匹配路由 → 404
        return makeErrorResponse(404, "Not Found: " + request.strPath);
    }

    // 20260330 ZJH generateApiDoc — 生成 API 文档（OpenAPI 风格简化版）
    // 返回: JSON 格式的 API 文档字符串
    std::string generateApiDoc() const {
        std::ostringstream oss;  // 20260330 ZJH 文档构建器

        oss << "{\n";
        oss << "  \"title\": \"OmniMatch Inference API\",\n";
        oss << "  \"version\": \"1.0.0\",\n";
        oss << "  \"host\": \"" << m_config.strHost << ":" << m_config.nPort << "\",\n";
        oss << "  \"endpoints\": [\n";

        // 20260330 ZJH 健康检查端点
        oss << "    {\n";
        oss << "      \"method\": \"GET\",\n";
        oss << "      \"path\": \"/api/health\",\n";
        oss << "      \"description\": \"Health check — returns server status and request count\"\n";
        oss << "    },\n";

        // 20260330 ZJH 单图推理端点
        oss << "    {\n";
        oss << "      \"method\": \"POST\",\n";
        oss << "      \"path\": \"/api/infer\",\n";
        oss << "      \"description\": \"Single image inference — accepts Base64 encoded image in JSON body\",\n";
        oss << "      \"request_body\": {\"image_base64\": \"string\", \"threshold\": \"float (optional, default 0.5)\"}\n";
        oss << "    },\n";

        // 20260330 ZJH 批量推理端点
        oss << "    {\n";
        oss << "      \"method\": \"POST\",\n";
        oss << "      \"path\": \"/api/batch\",\n";
        oss << "      \"description\": \"Batch inference — accepts array of Base64 images\",\n";
        oss << "      \"request_body\": {\"images\": [\"string\"], \"threshold\": \"float (optional)\"}\n";
        oss << "    },\n";

        // 20260330 ZJH 模型信息端点
        oss << "    {\n";
        oss << "      \"method\": \"GET\",\n";
        oss << "      \"path\": \"/api/model/info\",\n";
        oss << "      \"description\": \"Model information — returns model name, parameter count, input shape\"\n";
        oss << "    },\n";

        // 20260330 ZJH API 文档端点
        oss << "    {\n";
        oss << "      \"method\": \"GET\",\n";
        oss << "      \"path\": \"/api/doc\",\n";
        oss << "      \"description\": \"API documentation — this endpoint\"\n";
        oss << "    }\n";

        oss << "  ]\n";
        oss << "}\n";

        return oss.str();  // 20260330 ZJH 返回 API 文档 JSON
    }

private:
    ApiConfig m_config;                    // 20260330 ZJH 服务配置
    std::shared_ptr<Module> m_pModel;      // 20260330 ZJH 当前推理模型
    int m_nRequestCount;                   // 20260330 ZJH 累计请求计数

    // 20260330 ZJH handleHealth — 健康检查处理
    // 返回: 包含状态和请求计数的 JSON 响应
    HttpResponse handleHealth() const {
        HttpResponse resp;  // 20260330 ZJH 初始化响应
        resp.nStatusCode = 200;
        std::ostringstream oss;
        oss << "{\"status\": \"ok\", \"model_loaded\": " << (m_pModel ? "true" : "false")
            << ", \"request_count\": " << m_nRequestCount << "}";
        resp.strBody = oss.str();  // 20260330 ZJH 设置响应体
        return resp;
    }

    // 20260330 ZJH handleInfer — 单图推理处理
    // request: 包含 Base64 图像的 POST 请求
    // 返回: 推理结果 JSON
    HttpResponse handleInfer(const HttpRequest& request) const {
        // 20260330 ZJH 检查模型是否已加载
        if (!m_pModel) {
            return makeErrorResponse(503, "Model not loaded");  // 20260330 ZJH 服务不可用
        }

        // 20260330 ZJH 检查请求体是否为空
        if (request.strBody.empty()) {
            return makeErrorResponse(400, "Empty request body");  // 20260330 ZJH 错误请求
        }

        // 20260330 ZJH 简化 JSON 解析: 查找 "image_base64" 字段
        std::string strImageData = extractJsonString(request.strBody, "image_base64");
        if (strImageData.empty()) {
            return makeErrorResponse(400, "Missing 'image_base64' field");
        }

        // 20260330 ZJH 提取阈值（可选参数，默认 0.5）
        float fThreshold = extractJsonFloat(request.strBody, "threshold", 0.5f);

        // 20260330 ZJH 解码 Base64 图像数据
        std::vector<unsigned char> vecImageBytes = decodeBase64(strImageData);
        if (vecImageBytes.empty()) {
            return makeErrorResponse(400, "Invalid Base64 image data");
        }

        // 20260330 ZJH 构建推理输入张量（假设预处理后 224x224 RGB）
        // 实际部署时应根据模型输入规格动态调整
        int nInputSize = 224 * 224 * 3;  // 20260330 ZJH 默认输入尺寸
        Tensor inputTensor = Tensor::zeros({1, 3, 224, 224});  // 20260330 ZJH 创建输入张量
        float* pData = inputTensor.mutableFloatDataPtr();  // 20260330 ZJH 获取可写指针

        // 20260330 ZJH 填充张量数据（简化: 将字节归一化到 [0,1]）
        int nFillSize = std::min(static_cast<int>(vecImageBytes.size()), nInputSize);
        for (int i = 0; i < nFillSize; ++i) {
            pData[i] = static_cast<float>(vecImageBytes[i]) / 255.0f;
        }

        // 20260330 ZJH 执行前向推理
        Tensor outputTensor = m_pModel->forward(inputTensor);

        // 20260330 ZJH 格式化推理结果为 JSON
        HttpResponse resp;
        resp.nStatusCode = 200;
        std::ostringstream oss;
        oss << "{\"status\": \"ok\", \"threshold\": " << fThreshold
            << ", \"output_shape\": [";

        // 20260330 ZJH 输出张量形状
        auto vecShape = outputTensor.shapeVec();
        for (size_t i = 0; i < vecShape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << vecShape[i];
        }
        oss << "], \"output_size\": " << outputTensor.numel();

        // 20260330 ZJH 输出前 N 个值（最多10个，避免响应过大）
        oss << ", \"values\": [";
        const float* pOut = outputTensor.floatDataPtr();
        int nOutCount = std::min(10, static_cast<int>(outputTensor.numel()));
        for (int i = 0; i < nOutCount; ++i) {
            if (i > 0) oss << ", ";
            oss << pOut[i];
        }
        oss << "]}";

        resp.strBody = oss.str();
        return resp;  // 20260330 ZJH 返回推理结果
    }

    // 20260330 ZJH handleBatch — 批量推理处理
    // request: 包含多张 Base64 图像的 POST 请求
    // 返回: 各图推理结果数组 JSON
    HttpResponse handleBatch(const HttpRequest& request) const {
        // 20260330 ZJH 检查模型是否已加载
        if (!m_pModel) {
            return makeErrorResponse(503, "Model not loaded");
        }

        // 20260330 ZJH 检查请求体
        if (request.strBody.empty()) {
            return makeErrorResponse(400, "Empty request body");
        }

        // 20260330 ZJH 简化批量处理: 返回占位响应
        // 实际实现需解析 JSON 数组并逐图推理
        HttpResponse resp;
        resp.nStatusCode = 200;
        resp.strBody = "{\"status\": \"ok\", \"message\": \"Batch inference endpoint ready\", "
                        "\"max_concurrent\": " + std::to_string(m_config.nMaxConcurrent) + "}";
        return resp;
    }

    // 20260330 ZJH handleModelInfo — 模型信息处理
    // 返回: 模型参数量和加载状态 JSON
    HttpResponse handleModelInfo() const {
        HttpResponse resp;
        resp.nStatusCode = 200;
        std::ostringstream oss;

        if (m_pModel) {
            // 20260330 ZJH 统计模型参数量
            auto vecParams = m_pModel->parameters();
            int nTotalParams = 0;  // 20260330 ZJH 总参数数量
            for (const auto* pParam : vecParams) {
                nTotalParams += static_cast<int>(pParam->numel());  // 20260330 ZJH 累加每层参数数
            }

            oss << "{\"model_loaded\": true, \"total_parameters\": " << nTotalParams
                << ", \"num_layers\": " << vecParams.size() << "}";
        } else {
            oss << "{\"model_loaded\": false}";
        }

        resp.strBody = oss.str();
        return resp;
    }

    // 20260330 ZJH handleDoc — API 文档处理
    // 返回: API 文档 JSON
    HttpResponse handleDoc() const {
        HttpResponse resp;
        resp.nStatusCode = 200;
        resp.strBody = generateApiDoc();  // 20260330 ZJH 调用文档生成器
        return resp;
    }

    // 20260330 ZJH makeCorsResponse — 生成 CORS 预检响应
    // 返回: 允许跨域的 204 响应
    HttpResponse makeCorsResponse() const {
        HttpResponse resp;
        resp.nStatusCode = 204;  // 20260330 ZJH No Content
        resp.strBody = "";
        resp.strContentType = "";
        return resp;
    }

    // 20260330 ZJH makeErrorResponse — 生成错误响应
    // nCode: HTTP 状态码
    // strMessage: 错误消息
    // 返回: 包含错误信息的 JSON 响应
    static HttpResponse makeErrorResponse(int nCode, const std::string& strMessage) {
        HttpResponse resp;
        resp.nStatusCode = nCode;
        resp.strBody = "{\"error\": \"" + strMessage + "\"}";
        return resp;  // 20260330 ZJH 返回错误响应
    }

    // 20260330 ZJH extractJsonString — 从 JSON 字符串中提取指定字段的字符串值
    // strJson: JSON 文本
    // strKey: 要提取的键名
    // 返回: 字段值（不含引号），未找到返回空串
    static std::string extractJsonString(const std::string& strJson, const std::string& strKey) {
        // 20260330 ZJH 查找 "key": "value" 模式
        std::string strSearch = "\"" + strKey + "\"";  // 20260330 ZJH 构造搜索串
        size_t nPos = strJson.find(strSearch);
        if (nPos == std::string::npos) {
            return "";  // 20260330 ZJH 未找到键
        }

        // 20260330 ZJH 跳过键名和冒号
        nPos = strJson.find(':', nPos + strSearch.size());
        if (nPos == std::string::npos) {
            return "";  // 20260330 ZJH 无冒号
        }

        // 20260330 ZJH 跳过空白
        nPos++;
        while (nPos < strJson.size() && (strJson[nPos] == ' ' || strJson[nPos] == '\t')) {
            nPos++;
        }

        // 20260330 ZJH 检查是否为字符串值（以引号开头）
        if (nPos >= strJson.size() || strJson[nPos] != '"') {
            return "";  // 20260330 ZJH 非字符串值
        }

        // 20260330 ZJH 提取引号内的值
        nPos++;  // 20260330 ZJH 跳过开引号
        size_t nEnd = strJson.find('"', nPos);  // 20260330 ZJH 找闭引号
        if (nEnd == std::string::npos) {
            return "";  // 20260330 ZJH 无闭引号
        }

        return strJson.substr(nPos, nEnd - nPos);  // 20260330 ZJH 返回字段值
    }

    // 20260330 ZJH extractJsonFloat — 从 JSON 字符串中提取指定字段的浮点值
    // strJson: JSON 文本
    // strKey: 要提取的键名
    // fDefault: 未找到时的默认值
    // 返回: 字段浮点值
    static float extractJsonFloat(const std::string& strJson, const std::string& strKey,
                                  float fDefault) {
        // 20260330 ZJH 查找 "key": value 模式
        std::string strSearch = "\"" + strKey + "\"";
        size_t nPos = strJson.find(strSearch);
        if (nPos == std::string::npos) {
            return fDefault;  // 20260330 ZJH 未找到，返回默认值
        }

        // 20260330 ZJH 跳过键名和冒号
        nPos = strJson.find(':', nPos + strSearch.size());
        if (nPos == std::string::npos) {
            return fDefault;
        }

        // 20260330 ZJH 跳过空白
        nPos++;
        while (nPos < strJson.size() && (strJson[nPos] == ' ' || strJson[nPos] == '\t')) {
            nPos++;
        }

        // 20260330 ZJH 解析浮点数
        try {
            return std::stof(strJson.substr(nPos));  // 20260330 ZJH stof 自动截断到有效数字
        } catch (...) {
            return fDefault;  // 20260330 ZJH 解析失败，返回默认值
        }
    }

    // 20260330 ZJH decodeBase64 — Base64 解码
    // strBase64: Base64 编码字符串
    // 返回: 解码后的字节数组
    static std::vector<unsigned char> decodeBase64(const std::string& strBase64) {
        // 20260330 ZJH Base64 字符到 6-bit 值的查找表
        static const int s_arrDecodeTable[128] = {
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 0x00-0x0F
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 0x10-0x1F
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,  // 0x20-0x2F: +, /
            52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,  // 0x30-0x3F: 0-9
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, // 0x40-0x4F: A-O
            15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,  // 0x50-0x5F: P-Z
            -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,  // 0x60-0x6F: a-o
            41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1   // 0x70-0x7F: p-z
        };

        std::vector<unsigned char> vecResult;  // 20260330 ZJH 解码结果
        vecResult.reserve(strBase64.size() * 3 / 4);  // 20260330 ZJH 预分配（Base64 约 4:3 比例）

        int nBits = 0;       // 20260330 ZJH 当前累积的有效位数
        unsigned int nBuffer = 0;  // 20260330 ZJH 位缓冲区

        // 20260330 ZJH 逐字符解码
        for (char ch : strBase64) {
            if (ch == '=' || ch == '\n' || ch == '\r') {
                continue;  // 20260330 ZJH 跳过填充符和换行
            }
            if (ch < 0 || ch >= 128) {
                continue;  // 20260330 ZJH 跳过非 ASCII 字符
            }
            int nVal = s_arrDecodeTable[static_cast<int>(ch)];  // 20260330 ZJH 查表
            if (nVal < 0) {
                continue;  // 20260330 ZJH 跳过非法字符
            }

            nBuffer = (nBuffer << 6) | static_cast<unsigned int>(nVal);  // 20260330 ZJH 左移6位并合并
            nBits += 6;  // 20260330 ZJH 增加6个有效位

            // 20260330 ZJH 每满8位输出一个字节
            if (nBits >= 8) {
                nBits -= 8;  // 20260330 ZJH 消耗8位
                vecResult.push_back(static_cast<unsigned char>((nBuffer >> nBits) & 0xFF));
            }
        }

        return vecResult;  // 20260330 ZJH 返回解码后的字节数组
    }
};

// =========================================================
// 配方管理 (Recipe Manager)
// =========================================================

// 20260330 ZJH Recipe — 推理配方
// 一个配方绑定一个模型+参数集，可在产线切换不同产品时快速切换
struct Recipe {
    std::string strName;          // 20260330 ZJH 配方名称（唯一标识）
    std::string strModelPath;     // 20260330 ZJH 模型文件路径（.omm/.onnx）
    std::string strDescription;   // 20260330 ZJH 配方描述
    float fConfThreshold;         // 20260330 ZJH 置信度阈值 [0,1]
    std::map<std::string, std::string> mapParams;  // 20260330 ZJH 自定义参数键值对
};

// 20260330 ZJH RecipeManager — 配方管理器
// 支持添加/删除/切换/持久化配方，配合 InferenceServer 实现多产品切换
class RecipeManager {
public:
    // 20260330 ZJH addRecipe — 添加配方
    // recipe: 配方定义
    void addRecipe(const Recipe& recipe) {
        // 20260330 ZJH 检查是否已存在同名配方
        for (auto& existing : m_vecRecipes) {
            if (existing.strName == recipe.strName) {
                existing = recipe;  // 20260330 ZJH 同名则覆盖更新
                return;
            }
        }
        m_vecRecipes.push_back(recipe);  // 20260330 ZJH 新增配方
    }

    // 20260330 ZJH removeRecipe — 删除配方
    // strName: 配方名称
    void removeRecipe(const std::string& strName) {
        // 20260330 ZJH 查找并删除指定配方
        m_vecRecipes.erase(
            std::remove_if(m_vecRecipes.begin(), m_vecRecipes.end(),
                [&strName](const Recipe& r) { return r.strName == strName; }),
            m_vecRecipes.end()
        );

        // 20260330 ZJH 如果删除的是当前配方，清空当前配方名
        if (m_strCurrentRecipe == strName) {
            m_strCurrentRecipe.clear();
        }
    }

    // 20260330 ZJH getRecipe — 获取指定配方
    // strName: 配方名称
    // 返回: 配方定义（未找到则返回空配方）
    Recipe getRecipe(const std::string& strName) const {
        // 20260330 ZJH 线性搜索（配方数通常 <100，无需优化）
        for (const auto& recipe : m_vecRecipes) {
            if (recipe.strName == strName) {
                return recipe;  // 20260330 ZJH 找到，返回副本
            }
        }
        return Recipe{};  // 20260330 ZJH 未找到，返回空配方
    }

    // 20260330 ZJH listRecipes — 列出所有配方
    // 返回: 配方列表（副本）
    std::vector<Recipe> listRecipes() const {
        return m_vecRecipes;  // 20260330 ZJH 返回完整配方列表
    }

    // 20260330 ZJH switchRecipe — 切换当前配方
    // strName: 目标配方名称
    // 返回: 是否切换成功（配方存在则成功）
    bool switchRecipe(const std::string& strName) {
        // 20260330 ZJH 检查配方是否存在
        for (const auto& recipe : m_vecRecipes) {
            if (recipe.strName == strName) {
                m_strCurrentRecipe = strName;  // 20260330 ZJH 更新当前配方名
                return true;                    // 20260330 ZJH 切换成功
            }
        }
        return false;  // 20260330 ZJH 配方不存在，切换失败
    }

    // 20260330 ZJH getCurrentRecipe — 获取当前配方名称
    // 返回: 当前配方名称（未选择则为空串）
    std::string getCurrentRecipe() const {
        return m_strCurrentRecipe;  // 20260330 ZJH 返回当前配方名
    }

    // 20260330 ZJH saveRecipes — 持久化配方到文件
    // strPath: 目标文件路径（JSON 格式）
    // 返回: 是否保存成功
    bool saveRecipes(const std::string& strPath) const {
        std::ofstream ofs(strPath);  // 20260330 ZJH 打开文件
        if (!ofs.is_open()) {
            return false;  // 20260330 ZJH 文件打开失败
        }

        // 20260330 ZJH 手写 JSON 序列化（避免依赖第三方 JSON 库）
        ofs << "{\n";
        ofs << "  \"current_recipe\": \"" << m_strCurrentRecipe << "\",\n";
        ofs << "  \"recipes\": [\n";

        // 20260330 ZJH 逐个序列化配方
        for (size_t i = 0; i < m_vecRecipes.size(); ++i) {
            const auto& r = m_vecRecipes[i];
            ofs << "    {\n";
            ofs << "      \"name\": \"" << escapeJson(r.strName) << "\",\n";
            ofs << "      \"model_path\": \"" << escapeJson(r.strModelPath) << "\",\n";
            ofs << "      \"description\": \"" << escapeJson(r.strDescription) << "\",\n";
            ofs << "      \"conf_threshold\": " << r.fConfThreshold << ",\n";

            // 20260330 ZJH 序列化自定义参数
            ofs << "      \"params\": {";
            bool bFirst = true;  // 20260330 ZJH 首项标记（控制逗号）
            for (const auto& [key, val] : r.mapParams) {
                if (!bFirst) ofs << ", ";
                ofs << "\"" << escapeJson(key) << "\": \"" << escapeJson(val) << "\"";
                bFirst = false;
            }
            ofs << "}\n";

            ofs << "    }";
            if (i + 1 < m_vecRecipes.size()) ofs << ",";  // 20260330 ZJH 非末尾加逗号
            ofs << "\n";
        }

        ofs << "  ]\n";
        ofs << "}\n";

        return true;  // 20260330 ZJH 保存成功
    }

    // 20260330 ZJH loadRecipes — 从文件加载配方
    // strPath: 配方文件路径（JSON 格式）
    // 返回: 是否加载成功
    bool loadRecipes(const std::string& strPath) {
        std::ifstream ifs(strPath);  // 20260330 ZJH 打开文件
        if (!ifs.is_open()) {
            return false;  // 20260330 ZJH 文件打开失败
        }

        // 20260330 ZJH 读取全部内容
        std::string strContent((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());

        // 20260330 ZJH 简化解析：逐配方提取字段
        // 实际生产环境应使用 nlohmann/json 或 rapidjson
        m_vecRecipes.clear();  // 20260330 ZJH 清空现有配方

        // 20260330 ZJH 提取当前配方名
        m_strCurrentRecipe = extractField(strContent, "current_recipe");

        // 20260330 ZJH 查找每个配方块
        size_t nSearchPos = 0;  // 20260330 ZJH 搜索起始位置
        while (true) {
            // 20260330 ZJH 查找下一个 "name" 字段（每个配方的起始标志）
            size_t nNamePos = strContent.find("\"name\"", nSearchPos);
            if (nNamePos == std::string::npos) {
                break;  // 20260330 ZJH 无更多配方
            }

            // 20260330 ZJH 找到该配方块的结束位置
            size_t nBlockEnd = strContent.find('}', nNamePos);
            if (nBlockEnd == std::string::npos) {
                break;  // 20260330 ZJH JSON 格式异常
            }

            // 20260330 ZJH 提取配方块
            std::string strBlock = strContent.substr(nNamePos, nBlockEnd - nNamePos);

            Recipe recipe;
            recipe.strName = extractField(strBlock, "name");
            recipe.strModelPath = extractField(strBlock, "model_path");
            recipe.strDescription = extractField(strBlock, "description");

            // 20260330 ZJH 提取阈值
            std::string strThresh = extractField(strBlock, "conf_threshold");
            recipe.fConfThreshold = strThresh.empty() ? 0.5f : std::stof(strThresh);

            // 20260330 ZJH 添加到列表（跳过空名配方）
            if (!recipe.strName.empty()) {
                m_vecRecipes.push_back(recipe);
            }

            nSearchPos = nBlockEnd + 1;  // 20260330 ZJH 移动搜索位置
        }

        return true;  // 20260330 ZJH 加载成功
    }

private:
    std::vector<Recipe> m_vecRecipes;       // 20260330 ZJH 配方列表
    std::string m_strCurrentRecipe;         // 20260330 ZJH 当前激活配方名称

    // 20260330 ZJH escapeJson — JSON 字符串转义
    // str: 原始字符串
    // 返回: 转义后的字符串（双引号和反斜杠前加 \）
    static std::string escapeJson(const std::string& str) {
        std::string strResult;  // 20260330 ZJH 转义结果
        strResult.reserve(str.size() + 8);  // 20260330 ZJH 预分配（略大于原串）
        for (char ch : str) {
            if (ch == '"') {
                strResult += "\\\"";  // 20260330 ZJH 转义双引号
            } else if (ch == '\\') {
                strResult += "\\\\";  // 20260330 ZJH 转义反斜杠
            } else if (ch == '\n') {
                strResult += "\\n";   // 20260330 ZJH 转义换行
            } else if (ch == '\r') {
                strResult += "\\r";   // 20260330 ZJH 转义回车
            } else if (ch == '\t') {
                strResult += "\\t";   // 20260330 ZJH 转义制表符
            } else {
                strResult += ch;      // 20260330 ZJH 普通字符直接追加
            }
        }
        return strResult;
    }

    // 20260330 ZJH extractField — 简化 JSON 字段提取
    // strJson: JSON 文本
    // strKey: 键名
    // 返回: 字段值（字符串或数值的文本表示）
    static std::string extractField(const std::string& strJson, const std::string& strKey) {
        std::string strSearch = "\"" + strKey + "\"";  // 20260330 ZJH 构造搜索串
        size_t nPos = strJson.find(strSearch);
        if (nPos == std::string::npos) {
            return "";  // 20260330 ZJH 未找到
        }

        // 20260330 ZJH 跳过键名和冒号
        nPos = strJson.find(':', nPos + strSearch.size());
        if (nPos == std::string::npos) {
            return "";
        }
        nPos++;  // 20260330 ZJH 跳过冒号

        // 20260330 ZJH 跳过空白
        while (nPos < strJson.size() && (strJson[nPos] == ' ' || strJson[nPos] == '\t'
               || strJson[nPos] == '\n' || strJson[nPos] == '\r')) {
            nPos++;
        }

        if (nPos >= strJson.size()) {
            return "";
        }

        // 20260330 ZJH 字符串值（引号包裹）
        if (strJson[nPos] == '"') {
            nPos++;  // 20260330 ZJH 跳过开引号
            size_t nEnd = strJson.find('"', nPos);
            if (nEnd == std::string::npos) return "";
            return strJson.substr(nPos, nEnd - nPos);
        }

        // 20260330 ZJH 数值或布尔值（到逗号/大括号/换行截止）
        size_t nEnd = strJson.find_first_of(",}\n\r", nPos);
        if (nEnd == std::string::npos) {
            nEnd = strJson.size();
        }
        std::string strVal = strJson.substr(nPos, nEnd - nPos);
        // 20260330 ZJH 去除尾部空白
        while (!strVal.empty() && (strVal.back() == ' ' || strVal.back() == '\t')) {
            strVal.pop_back();
        }
        return strVal;
    }
};

}  // namespace om
