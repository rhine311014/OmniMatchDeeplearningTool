#pragma once
// 20260330 ZJH 数据格式转换器 — COCO JSON / Pascal VOC XML / YOLO txt 三格式互转
// 对标海康 VisionTrain 的数据导入导出 + MVTec DL Tool 的 Import/Export 功能
// 零第三方依赖：手写 JSON / XML 解析器，仅依赖 C++ 标准库

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstring>

namespace om {

// =========================================================================
// 20260330 ZJH 通用标注格式（内部中间表示）
// 所有外部格式先转为此中间表示，再输出为目标格式
// =========================================================================

// 20260330 ZJH 单个目标框（像素坐标）
struct BBox {
    float fX, fY, fW, fH;  // 20260330 ZJH 左上角坐标 + 宽高（像素坐标）
    int nClassId;           // 20260330 ZJH 类别 ID（从 0 开始）
    std::string strClassName;  // 20260330 ZJH 类别名称
    float fConfidence = 1.0f;  // 20260330 ZJH 置信度（默认 1.0 表示人工标注）
    // 20260330 ZJH 分割 mask（可选，COCO polygon 格式: 每个多边形是一组 [x1,y1,x2,y2,...] 像素坐标）
    std::vector<std::vector<float>> vecPolygons;
};

// 20260330 ZJH 单张图像标注（包含多个目标框）
struct ImageAnnotation {
    std::string strImagePath;  // 20260330 ZJH 图像文件的完整路径或相对路径
    std::string strImageName;  // 20260330 ZJH 图像文件名（不含路径）
    int nWidth, nHeight;       // 20260330 ZJH 图像宽高（像素）
    std::vector<BBox> vecBoxes;  // 20260330 ZJH 该图像上的所有目标框
};

// 20260330 ZJH 数据集标注（所有图像 + 类别列表）
struct DatasetAnnotation {
    std::vector<ImageAnnotation> vecImages;     // 20260330 ZJH 所有图像的标注
    std::vector<std::string> vecClassNames;     // 20260330 ZJH 类别名称列表（索引即 classId）
    std::string strDescription;                 // 20260330 ZJH 数据集描述信息
};

// =========================================================================
// 20260330 ZJH 内部 JSON 解析工具函数
// 手写轻量 JSON 解析：基于字符串搜索 + 括号匹配，不依赖 nlohmann_json
// =========================================================================
namespace detail {

// 20260330 ZJH 跳过空白字符，返回下一个非空白字符的位置
// 参数: strJson - JSON 字符串, nPos - 当前位置
// 返回: 下一个非空白字符的位置
inline size_t skipWhitespace(const std::string& strJson, size_t nPos) {
    while (nPos < strJson.size() && (strJson[nPos] == ' ' || strJson[nPos] == '\t' ||
           strJson[nPos] == '\n' || strJson[nPos] == '\r')) {
        ++nPos;  // 20260330 ZJH 跳过空格、制表符、换行
    }
    return nPos;  // 20260330 ZJH 返回第一个非空白位置
}

// 20260330 ZJH 提取带引号的字符串值（从当前引号位置到结束引号）
// 参数: strJson - JSON 字符串, nPos - 起始引号位置
// 返回: pair(提取的字符串, 结束引号后的位置)
inline std::pair<std::string, size_t> extractString(const std::string& strJson, size_t nPos) {
    if (nPos >= strJson.size() || strJson[nPos] != '"') {
        return {"", nPos};  // 20260330 ZJH 不是引号开头，返回空
    }
    ++nPos;  // 20260330 ZJH 跳过起始引号
    std::string strResult;  // 20260330 ZJH 存放提取结果
    while (nPos < strJson.size() && strJson[nPos] != '"') {
        if (strJson[nPos] == '\\' && nPos + 1 < strJson.size()) {
            // 20260330 ZJH 处理转义字符
            ++nPos;  // 20260330 ZJH 跳过反斜杠
            switch (strJson[nPos]) {
                case '"':  strResult += '"';  break;  // 20260330 ZJH 转义双引号
                case '\\': strResult += '\\'; break;  // 20260330 ZJH 转义反斜杠
                case '/':  strResult += '/';  break;  // 20260330 ZJH 转义斜杠
                case 'n':  strResult += '\n'; break;  // 20260330 ZJH 转义换行
                case 't':  strResult += '\t'; break;  // 20260330 ZJH 转义制表符
                default:   strResult += strJson[nPos]; break;  // 20260330 ZJH 未知转义，原样保留
            }
        } else {
            strResult += strJson[nPos];  // 20260330 ZJH 普通字符直接追加
        }
        ++nPos;  // 20260330 ZJH 移动到下一个字符
    }
    if (nPos < strJson.size()) ++nPos;  // 20260330 ZJH 跳过结束引号
    return {strResult, nPos};  // 20260330 ZJH 返回提取的字符串和结束位置
}

// 20260330 ZJH 提取数值（整数或浮点数）
// 参数: strJson - JSON 字符串, nPos - 数值起始位置
// 返回: pair(数值的double表示, 结束位置)
inline std::pair<double, size_t> extractNumber(const std::string& strJson, size_t nPos) {
    size_t nStart = nPos;  // 20260330 ZJH 记录起始位置
    // 20260330 ZJH 扫描数值字符：数字、小数点、正负号、科学记数法
    while (nPos < strJson.size() && (std::isdigit(strJson[nPos]) || strJson[nPos] == '.' ||
           strJson[nPos] == '-' || strJson[nPos] == '+' || strJson[nPos] == 'e' || strJson[nPos] == 'E')) {
        ++nPos;  // 20260330 ZJH 继续扫描
    }
    double dVal = 0.0;  // 20260330 ZJH 解析结果
    std::string strNum = strJson.substr(nStart, nPos - nStart);  // 20260330 ZJH 截取数值子串
    std::sscanf(strNum.c_str(), "%lf", &dVal);  // 20260330 ZJH 用 sscanf 解析
    return {dVal, nPos};  // 20260330 ZJH 返回数值和结束位置
}

// 20260330 ZJH 查找匹配的右括号（支持 [] 和 {} 嵌套）
// 参数: strJson - JSON 字符串, nPos - 左括号位置
// 返回: 匹配的右括号位置（含括号本身）
inline size_t findMatchingBracket(const std::string& strJson, size_t nPos) {
    if (nPos >= strJson.size()) return std::string::npos;  // 20260330 ZJH 越界检查
    char chOpen = strJson[nPos];   // 20260330 ZJH 左括号字符
    char chClose = (chOpen == '[') ? ']' : '}';  // 20260330 ZJH 对应右括号
    int nDepth = 1;  // 20260330 ZJH 嵌套深度计数
    bool bInString = false;  // 20260330 ZJH 是否在字符串内部（字符串内的括号不计数）
    ++nPos;  // 20260330 ZJH 跳过起始括号
    while (nPos < strJson.size() && nDepth > 0) {
        char ch = strJson[nPos];  // 20260330 ZJH 当前字符
        if (bInString) {
            // 20260330 ZJH 在字符串内部：只关注引号和转义
            if (ch == '\\') {
                ++nPos;  // 20260330 ZJH 跳过转义字符的下一个字符
            } else if (ch == '"') {
                bInString = false;  // 20260330 ZJH 字符串结束
            }
        } else {
            if (ch == '"') {
                bInString = true;  // 20260330 ZJH 进入字符串
            } else if (ch == chOpen) {
                ++nDepth;  // 20260330 ZJH 嵌套加深
            } else if (ch == chClose) {
                --nDepth;  // 20260330 ZJH 嵌套减少
                if (nDepth == 0) return nPos;  // 20260330 ZJH 找到匹配的右括号
            }
        }
        ++nPos;  // 20260330 ZJH 移动到下一个字符
    }
    return std::string::npos;  // 20260330 ZJH 未找到匹配括号
}

// 20260330 ZJH 在 JSON 对象中查找指定 key 对应的值起始位置
// 参数: strJson - JSON 字符串, strKey - 要查找的键名
// 返回: 值的起始位置（跳过冒号和空白）；未找到返回 npos
inline size_t findKey(const std::string& strJson, const std::string& strKey) {
    std::string strSearch = "\"" + strKey + "\"";  // 20260330 ZJH 构造带引号的搜索串
    size_t nPos = 0;  // 20260330 ZJH 搜索起始位置
    while (nPos < strJson.size()) {
        nPos = strJson.find(strSearch, nPos);  // 20260330 ZJH 查找 key
        if (nPos == std::string::npos) return std::string::npos;  // 20260330 ZJH 未找到
        // 20260330 ZJH 跳过 key 和冒号
        nPos += strSearch.size();  // 20260330 ZJH 跳过 "key"
        nPos = skipWhitespace(strJson, nPos);  // 20260330 ZJH 跳过空白
        if (nPos < strJson.size() && strJson[nPos] == ':') {
            ++nPos;  // 20260330 ZJH 跳过冒号
            nPos = skipWhitespace(strJson, nPos);  // 20260330 ZJH 跳过冒号后的空白
            return nPos;  // 20260330 ZJH 返回值起始位置
        }
        // 20260330 ZJH 没有冒号说明这不是一个 key，继续搜索
    }
    return std::string::npos;  // 20260330 ZJH 未找到
}

// 20260330 ZJH 提取 JSON 数组的各个元素子串
// 假设 nPos 指向 '[' 字符，返回数组中每个元素的子串
// 参数: strJson - JSON 字符串, nPos - '[' 位置
// 返回: 各元素的子串 vector
inline std::vector<std::string> extractArrayElements(const std::string& strJson, size_t nPos) {
    std::vector<std::string> vecElements;  // 20260330 ZJH 存放各元素子串
    if (nPos >= strJson.size() || strJson[nPos] != '[') return vecElements;  // 20260330 ZJH 非数组

    size_t nEnd = findMatchingBracket(strJson, nPos);  // 20260330 ZJH 找到 ']' 位置
    if (nEnd == std::string::npos) return vecElements;  // 20260330 ZJH 括号不匹配

    ++nPos;  // 20260330 ZJH 跳过 '['
    nPos = skipWhitespace(strJson, nPos);  // 20260330 ZJH 跳过空白
    if (nPos >= nEnd) return vecElements;  // 20260330 ZJH 空数组

    // 20260330 ZJH 逐个提取元素
    while (nPos < nEnd) {
        nPos = skipWhitespace(strJson, nPos);  // 20260330 ZJH 跳过元素前空白
        if (nPos >= nEnd) break;  // 20260330 ZJH 到达数组末尾

        size_t nElemStart = nPos;  // 20260330 ZJH 元素起始位置
        size_t nElemEnd = nPos;    // 20260330 ZJH 元素结束位置

        char ch = strJson[nPos];  // 20260330 ZJH 元素首字符
        if (ch == '{' || ch == '[') {
            // 20260330 ZJH 嵌套对象或数组：找到匹配的右括号
            nElemEnd = findMatchingBracket(strJson, nPos);
            if (nElemEnd == std::string::npos) break;  // 20260330 ZJH 括号不匹配
            ++nElemEnd;  // 20260330 ZJH 包含右括号
        } else if (ch == '"') {
            // 20260330 ZJH 字符串元素
            auto [str, end] = extractString(strJson, nPos);  // 20260330 ZJH 提取字符串
            nElemEnd = end;  // 20260330 ZJH 记录结束位置
        } else {
            // 20260330 ZJH 数值或布尔：扫描到逗号或 ']'
            while (nElemEnd < nEnd && strJson[nElemEnd] != ',' && strJson[nElemEnd] != ']') {
                ++nElemEnd;
            }
        }

        // 20260330 ZJH 截取元素子串并加入列表
        std::string strElem = strJson.substr(nElemStart, nElemEnd - nElemStart);
        // 20260330 ZJH 去除尾部空白
        while (!strElem.empty() && (strElem.back() == ' ' || strElem.back() == '\n' ||
               strElem.back() == '\r' || strElem.back() == '\t')) {
            strElem.pop_back();
        }
        if (!strElem.empty()) {
            vecElements.push_back(strElem);  // 20260330 ZJH 添加到结果
        }

        nPos = nElemEnd;  // 20260330 ZJH 移到元素结束位置
        nPos = skipWhitespace(strJson, nPos);  // 20260330 ZJH 跳过空白
        if (nPos < nEnd && strJson[nPos] == ',') {
            ++nPos;  // 20260330 ZJH 跳过逗号分隔符
        }
    }

    return vecElements;  // 20260330 ZJH 返回所有元素
}

// 20260330 ZJH 从 JSON 对象子串中提取指定 key 的字符串值
// 参数: strObj - JSON 对象子串, strKey - 键名
// 返回: 字符串值（未找到返回空串）
inline std::string getStringVal(const std::string& strObj, const std::string& strKey) {
    size_t nPos = findKey(strObj, strKey);  // 20260330 ZJH 定位 key 的值位置
    if (nPos == std::string::npos) return "";  // 20260330 ZJH 未找到
    auto [strVal, _] = extractString(strObj, nPos);  // 20260330 ZJH 提取字符串
    return strVal;  // 20260330 ZJH 返回值
}

// 20260330 ZJH 从 JSON 对象子串中提取指定 key 的数值
// 参数: strObj - JSON 对象子串, strKey - 键名
// 返回: double 值（未找到返回 0.0）
inline double getNumberVal(const std::string& strObj, const std::string& strKey) {
    size_t nPos = findKey(strObj, strKey);  // 20260330 ZJH 定位 key 的值位置
    if (nPos == std::string::npos) return 0.0;  // 20260330 ZJH 未找到
    auto [dVal, _] = extractNumber(strObj, nPos);  // 20260330 ZJH 提取数值
    return dVal;  // 20260330 ZJH 返回值
}

// 20260330 ZJH 从 JSON 对象子串中提取指定 key 的数组元素
// 参数: strObj - JSON 对象子串, strKey - 键名
// 返回: 数组元素子串列表（未找到返回空）
inline std::vector<std::string> getArrayVal(const std::string& strObj, const std::string& strKey) {
    size_t nPos = findKey(strObj, strKey);  // 20260330 ZJH 定位 key 的值位置
    if (nPos == std::string::npos) return {};  // 20260330 ZJH 未找到
    return extractArrayElements(strObj, nPos);  // 20260330 ZJH 提取数组元素
}

// 20260330 ZJH 读取整个文件内容到字符串
// 参数: strPath - 文件路径
// 返回: 文件内容字符串（失败返回空串）
inline std::string readFileContents(const std::string& strPath) {
    std::ifstream ifs(strPath, std::ios::binary);  // 20260330 ZJH 二进制模式打开
    if (!ifs.is_open()) return "";  // 20260330 ZJH 打开失败
    std::string strContent((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());  // 20260330 ZJH 一次性读入
    ifs.close();  // 20260330 ZJH 关闭文件
    return strContent;  // 20260330 ZJH 返回内容
}

// 20260330 ZJH XML 简易解析：提取指定标签的文本内容
// 参数: strXml - XML 字符串, strTag - 标签名, nStartPos - 搜索起始位置
// 返回: pair(标签文本内容, 结束标签后的位置)
inline std::pair<std::string, size_t> extractXmlTag(const std::string& strXml,
                                                     const std::string& strTag,
                                                     size_t nStartPos = 0) {
    std::string strOpen = "<" + strTag + ">";    // 20260330 ZJH 开始标签
    std::string strClose = "</" + strTag + ">";  // 20260330 ZJH 结束标签
    size_t nOpen = strXml.find(strOpen, nStartPos);  // 20260330 ZJH 查找开始标签
    if (nOpen == std::string::npos) return {"", std::string::npos};  // 20260330 ZJH 未找到
    size_t nContentStart = nOpen + strOpen.size();  // 20260330 ZJH 内容起始
    size_t nClose = strXml.find(strClose, nContentStart);  // 20260330 ZJH 查找结束标签
    if (nClose == std::string::npos) return {"", std::string::npos};  // 20260330 ZJH 未找到
    std::string strContent = strXml.substr(nContentStart, nClose - nContentStart);  // 20260330 ZJH 截取内容
    size_t nAfter = nClose + strClose.size();  // 20260330 ZJH 结束标签后的位置
    return {strContent, nAfter};  // 20260330 ZJH 返回内容和位置
}

// 20260330 ZJH XML 简易解析：提取所有匹配指定标签的子块
// 参数: strXml - XML 字符串, strTag - 标签名
// 返回: 所有匹配块的内容列表
inline std::vector<std::string> extractAllXmlBlocks(const std::string& strXml,
                                                     const std::string& strTag) {
    std::vector<std::string> vecBlocks;  // 20260330 ZJH 存放所有块
    size_t nPos = 0;  // 20260330 ZJH 搜索起始位置
    while (nPos < strXml.size()) {
        auto [strContent, nAfter] = extractXmlTag(strXml, strTag, nPos);  // 20260330 ZJH 提取下一个块
        if (nAfter == std::string::npos) break;  // 20260330 ZJH 没有更多块
        vecBlocks.push_back(strContent);  // 20260330 ZJH 添加到结果
        nPos = nAfter;  // 20260330 ZJH 移动到下一个搜索位置
    }
    return vecBlocks;  // 20260330 ZJH 返回所有块
}

// 20260330 ZJH XML 转义：将特殊字符转为 XML 实体
// 参数: strInput - 原始字符串
// 返回: 转义后的安全字符串
inline std::string xmlEscape(const std::string& strInput) {
    std::string strOut;  // 20260330 ZJH 输出缓冲
    strOut.reserve(strInput.size() + 16);  // 20260330 ZJH 预分配空间
    for (char ch : strInput) {
        switch (ch) {
            case '&':  strOut += "&amp;";  break;  // 20260330 ZJH & → &amp;
            case '<':  strOut += "&lt;";   break;  // 20260330 ZJH < → &lt;
            case '>':  strOut += "&gt;";   break;  // 20260330 ZJH > → &gt;
            case '"':  strOut += "&quot;"; break;  // 20260330 ZJH " → &quot;
            case '\'': strOut += "&apos;"; break;  // 20260330 ZJH ' → &apos;
            default:   strOut += ch;       break;  // 20260330 ZJH 普通字符
        }
    }
    return strOut;  // 20260330 ZJH 返回转义后的字符串
}

// 20260330 ZJH JSON 转义：将特殊字符转为 JSON 转义序列
// 参数: strInput - 原始字符串
// 返回: 转义后的安全字符串
inline std::string jsonEscape(const std::string& strInput) {
    std::string strOut;  // 20260330 ZJH 输出缓冲
    strOut.reserve(strInput.size() + 16);  // 20260330 ZJH 预分配空间
    for (char ch : strInput) {
        switch (ch) {
            case '"':  strOut += "\\\""; break;  // 20260330 ZJH " → \"
            case '\\': strOut += "\\\\"; break;  // 20260330 ZJH \ → \\
            case '\n': strOut += "\\n";  break;  // 20260330 ZJH 换行 → \n
            case '\r': strOut += "\\r";  break;  // 20260330 ZJH 回车 → \r
            case '\t': strOut += "\\t";  break;  // 20260330 ZJH 制表符 → \t
            default:   strOut += ch;     break;  // 20260330 ZJH 普通字符
        }
    }
    return strOut;  // 20260330 ZJH 返回转义后的字符串
}

// 20260330 ZJH 判断文件扩展名是否为常见图像格式
// 参数: strExt - 文件扩展名（含点号，如 ".jpg"）
// 返回: true 表示是图像文件
inline bool isImageExtension(const std::string& strExt) {
    std::string strLower = strExt;  // 20260330 ZJH 转小写用于比较
    std::transform(strLower.begin(), strLower.end(), strLower.begin(), ::tolower);
    // 20260330 ZJH 支持 JPEG/PNG/BMP/TIFF/WebP 格式
    return strLower == ".jpg" || strLower == ".jpeg" || strLower == ".png" ||
           strLower == ".bmp" || strLower == ".tif" || strLower == ".tiff" ||
           strLower == ".webp";
}

}  // namespace detail

// =========================================================================
// 20260330 ZJH COCO JSON 导入
// 支持 instances_train.json / instances_val.json 标准 COCO 格式
// 解析流程: images→id映射 → categories→id映射 → annotations→按 image_id 分组
// =========================================================================
inline DatasetAnnotation importCOCO(const std::string& strJsonPath) {
    DatasetAnnotation dataset;  // 20260330 ZJH 输出结果

    // 20260330 ZJH 读取整个 JSON 文件
    std::string strJson = detail::readFileContents(strJsonPath);
    if (strJson.empty()) return dataset;  // 20260330 ZJH 文件读取失败

    // ---- Step 1: 解析 categories 数组 → classId→className 映射 ----
    std::map<int, std::string> mapCategoryIdToName;  // 20260330 ZJH COCO category_id → 类别名
    std::map<int, int> mapCategoryIdToIndex;          // 20260330 ZJH COCO category_id → 连续 index
    {
        auto vecCatElems = detail::getArrayVal(strJson, "categories");  // 20260330 ZJH 提取 categories 数组
        for (const auto& strCat : vecCatElems) {
            int nId = static_cast<int>(detail::getNumberVal(strCat, "id"));      // 20260330 ZJH COCO 类别 ID
            std::string strName = detail::getStringVal(strCat, "name");           // 20260330 ZJH 类别名称
            mapCategoryIdToName[nId] = strName;  // 20260330 ZJH 记录映射
        }
        // 20260330 ZJH 按 COCO id 排序后分配连续 index（COCO 的 category_id 可能不连续）
        int nIdx = 0;
        for (auto& [nCatId, strName] : mapCategoryIdToName) {
            mapCategoryIdToIndex[nCatId] = nIdx;  // 20260330 ZJH 分配连续索引
            dataset.vecClassNames.push_back(strName);  // 20260330 ZJH 添加到类别列表
            ++nIdx;
        }
    }

    // ---- Step 2: 解析 images 数组 → imageId→ImageAnnotation 映射 ----
    std::map<int, size_t> mapImageIdToIdx;  // 20260330 ZJH COCO image_id → vecImages 索引
    {
        auto vecImgElems = detail::getArrayVal(strJson, "images");  // 20260330 ZJH 提取 images 数组
        for (const auto& strImg : vecImgElems) {
            int nId = static_cast<int>(detail::getNumberVal(strImg, "id"));   // 20260330 ZJH 图像 ID
            std::string strFileName = detail::getStringVal(strImg, "file_name");  // 20260330 ZJH 文件名
            int nW = static_cast<int>(detail::getNumberVal(strImg, "width"));     // 20260330 ZJH 图像宽度
            int nH = static_cast<int>(detail::getNumberVal(strImg, "height"));    // 20260330 ZJH 图像高度

            ImageAnnotation imgAnn;  // 20260330 ZJH 创建图像标注
            imgAnn.strImageName = strFileName;  // 20260330 ZJH 设置文件名
            imgAnn.strImagePath = strFileName;  // 20260330 ZJH 路径暂时设为文件名（导入后可修改）
            imgAnn.nWidth = nW;   // 20260330 ZJH 设置宽度
            imgAnn.nHeight = nH;  // 20260330 ZJH 设置高度

            mapImageIdToIdx[nId] = dataset.vecImages.size();  // 20260330 ZJH 记录索引映射
            dataset.vecImages.push_back(imgAnn);  // 20260330 ZJH 添加到结果
        }
    }

    // ---- Step 3: 解析 annotations 数组 → 按 image_id 分配到对应图像 ----
    {
        auto vecAnnElems = detail::getArrayVal(strJson, "annotations");  // 20260330 ZJH 提取 annotations 数组
        for (const auto& strAnn : vecAnnElems) {
            int nImageId = static_cast<int>(detail::getNumberVal(strAnn, "image_id"));     // 20260330 ZJH 所属图像 ID
            int nCategoryId = static_cast<int>(detail::getNumberVal(strAnn, "category_id"));  // 20260330 ZJH 类别 ID

            // 20260330 ZJH 查找对应的图像
            auto itImg = mapImageIdToIdx.find(nImageId);
            if (itImg == mapImageIdToIdx.end()) continue;  // 20260330 ZJH 未知 image_id，跳过

            // 20260330 ZJH 解析 bbox: [x, y, width, height]（COCO 格式为像素坐标）
            auto vecBboxElems = detail::getArrayVal(strAnn, "bbox");  // 20260330 ZJH 提取 bbox 数组
            if (vecBboxElems.size() < 4) continue;  // 20260330 ZJH bbox 不完整，跳过

            BBox box;  // 20260330 ZJH 创建目标框
            std::sscanf(vecBboxElems[0].c_str(), "%f", &box.fX);  // 20260330 ZJH 解析 x
            std::sscanf(vecBboxElems[1].c_str(), "%f", &box.fY);  // 20260330 ZJH 解析 y
            std::sscanf(vecBboxElems[2].c_str(), "%f", &box.fW);  // 20260330 ZJH 解析 width
            std::sscanf(vecBboxElems[3].c_str(), "%f", &box.fH);  // 20260330 ZJH 解析 height

            // 20260330 ZJH 设置类别信息
            auto itCat = mapCategoryIdToIndex.find(nCategoryId);
            if (itCat != mapCategoryIdToIndex.end()) {
                box.nClassId = itCat->second;  // 20260330 ZJH 连续类别索引
                box.strClassName = mapCategoryIdToName[nCategoryId];  // 20260330 ZJH 类别名称
            } else {
                box.nClassId = nCategoryId;  // 20260330 ZJH 回退使用原始 ID
            }

            // 20260330 ZJH 解析置信度（如有 score 字段）
            double dScore = detail::getNumberVal(strAnn, "score");
            box.fConfidence = (dScore > 0.0) ? static_cast<float>(dScore) : 1.0f;

            // 20260330 ZJH 解析分割多边形（可选）
            auto vecSegElems = detail::getArrayVal(strAnn, "segmentation");
            for (const auto& strPoly : vecSegElems) {
                // 20260330 ZJH 每个多边形是一个 float 数组 [x1,y1,x2,y2,...]
                if (strPoly.empty() || strPoly[0] != '[') continue;  // 20260330 ZJH 跳过非数组
                auto vecCoords = detail::extractArrayElements(strPoly, 0);  // 20260330 ZJH 提取坐标值
                std::vector<float> vecPoly;  // 20260330 ZJH 单个多边形的坐标
                for (const auto& strCoord : vecCoords) {
                    float fVal = 0.0f;  // 20260330 ZJH 坐标值
                    std::sscanf(strCoord.c_str(), "%f", &fVal);  // 20260330 ZJH 解析
                    vecPoly.push_back(fVal);  // 20260330 ZJH 添加
                }
                if (!vecPoly.empty()) {
                    box.vecPolygons.push_back(vecPoly);  // 20260330 ZJH 添加多边形
                }
            }

            dataset.vecImages[itImg->second].vecBoxes.push_back(box);  // 20260330 ZJH 添加到对应图像
        }
    }

    dataset.strDescription = "Imported from COCO JSON: " + strJsonPath;  // 20260330 ZJH 设置描述
    return dataset;  // 20260330 ZJH 返回结果
}

// =========================================================================
// 20260330 ZJH COCO JSON 导出
// 输出标准 COCO instances 格式，包含 images/categories/annotations 三个数组
// =========================================================================
inline bool exportCOCO(const DatasetAnnotation& dataset, const std::string& strOutputPath) {
    // 20260330 ZJH 确保输出目录存在
    std::filesystem::path outPath(strOutputPath);  // 20260330 ZJH 输出路径
    if (outPath.has_parent_path()) {
        std::filesystem::create_directories(outPath.parent_path());  // 20260330 ZJH 创建父目录
    }

    std::ofstream ofs(strOutputPath);  // 20260330 ZJH 打开输出文件
    if (!ofs.is_open()) return false;  // 20260330 ZJH 无法打开文件

    ofs << "{\n";  // 20260330 ZJH JSON 根对象开始

    // ---- categories 数组 ----
    ofs << "  \"categories\": [\n";
    for (size_t i = 0; i < dataset.vecClassNames.size(); ++i) {
        // 20260330 ZJH COCO 的 category_id 从 1 开始（非 0），但此处保持与内部 classId 一致
        ofs << "    {\"id\": " << i
            << ", \"name\": \"" << detail::jsonEscape(dataset.vecClassNames[i])
            << "\", \"supercategory\": \"none\"}";
        if (i + 1 < dataset.vecClassNames.size()) ofs << ",";  // 20260330 ZJH 逗号分隔
        ofs << "\n";
    }
    ofs << "  ],\n";

    // ---- images 数组 ----
    ofs << "  \"images\": [\n";
    for (size_t i = 0; i < dataset.vecImages.size(); ++i) {
        const auto& img = dataset.vecImages[i];  // 20260330 ZJH 当前图像
        ofs << "    {\"id\": " << i
            << ", \"file_name\": \"" << detail::jsonEscape(img.strImageName)
            << "\", \"width\": " << img.nWidth
            << ", \"height\": " << img.nHeight << "}";
        if (i + 1 < dataset.vecImages.size()) ofs << ",";  // 20260330 ZJH 逗号分隔
        ofs << "\n";
    }
    ofs << "  ],\n";

    // ---- annotations 数组 ----
    ofs << "  \"annotations\": [\n";
    int nAnnId = 0;  // 20260330 ZJH 全局标注 ID
    for (size_t nImgIdx = 0; nImgIdx < dataset.vecImages.size(); ++nImgIdx) {
        const auto& img = dataset.vecImages[nImgIdx];  // 20260330 ZJH 当前图像
        for (size_t nBoxIdx = 0; nBoxIdx < img.vecBoxes.size(); ++nBoxIdx) {
            const auto& box = img.vecBoxes[nBoxIdx];  // 20260330 ZJH 当前框
            float fArea = box.fW * box.fH;  // 20260330 ZJH 面积

            if (nAnnId > 0) ofs << ",\n";  // 20260330 ZJH 逗号分隔

            // 20260330 ZJH 写入标注 JSON 对象
            char arrBuf[512];
            std::snprintf(arrBuf, sizeof(arrBuf),
                "    {\"id\": %d, \"image_id\": %d, \"category_id\": %d, "
                "\"bbox\": [%.2f, %.2f, %.2f, %.2f], \"area\": %.2f, \"iscrowd\": 0",
                nAnnId, static_cast<int>(nImgIdx), box.nClassId,
                box.fX, box.fY, box.fW, box.fH, fArea);
            ofs << arrBuf;

            // 20260330 ZJH 写入分割多边形（如有）
            if (!box.vecPolygons.empty()) {
                ofs << ", \"segmentation\": [";
                for (size_t nPolyIdx = 0; nPolyIdx < box.vecPolygons.size(); ++nPolyIdx) {
                    if (nPolyIdx > 0) ofs << ", ";  // 20260330 ZJH 多边形间逗号分隔
                    ofs << "[";
                    const auto& vecPoly = box.vecPolygons[nPolyIdx];  // 20260330 ZJH 当前多边形
                    for (size_t k = 0; k < vecPoly.size(); ++k) {
                        if (k > 0) ofs << ", ";  // 20260330 ZJH 坐标间逗号分隔
                        char arrCoord[32];
                        std::snprintf(arrCoord, sizeof(arrCoord), "%.2f", vecPoly[k]);
                        ofs << arrCoord;
                    }
                    ofs << "]";
                }
                ofs << "]";
            } else {
                // 20260330 ZJH 没有分割多边形时输出空数组
                ofs << ", \"segmentation\": []";
            }

            ofs << "}";  // 20260330 ZJH 结束标注对象
            ++nAnnId;
        }
    }
    if (nAnnId > 0) ofs << "\n";  // 20260330 ZJH 最后一行换行
    ofs << "  ]\n";

    ofs << "}\n";  // 20260330 ZJH JSON 根对象结束
    ofs.close();   // 20260330 ZJH 关闭文件
    return true;   // 20260330 ZJH 导出成功
}

// =========================================================================
// 20260330 ZJH Pascal VOC XML 导入
// 从指定目录读取所有 .xml 文件，解析标准 VOC 格式
// 每个 XML 包含: <annotation> → <filename> + <size> + <object>*N
// =========================================================================
inline DatasetAnnotation importVOC(const std::string& strXmlDir) {
    DatasetAnnotation dataset;  // 20260330 ZJH 输出结果
    std::map<std::string, int> mapClassToId;  // 20260330 ZJH 类别名→classId 动态映射

    // 20260330 ZJH 检查目录是否存在
    if (!std::filesystem::exists(strXmlDir)) return dataset;

    // 20260330 ZJH 遍历目录下所有 .xml 文件
    for (const auto& entry : std::filesystem::directory_iterator(strXmlDir)) {
        if (!entry.is_regular_file()) continue;  // 20260330 ZJH 跳过非文件
        std::string strExt = entry.path().extension().string();  // 20260330 ZJH 获取扩展名
        if (strExt != ".xml" && strExt != ".XML") continue;  // 20260330 ZJH 只处理 XML

        // 20260330 ZJH 读取 XML 文件内容
        std::string strXml = detail::readFileContents(entry.path().string());
        if (strXml.empty()) continue;  // 20260330 ZJH 空文件跳过

        ImageAnnotation imgAnn;  // 20260330 ZJH 创建图像标注

        // 20260330 ZJH 提取文件名
        auto [strFilename, _1] = detail::extractXmlTag(strXml, "filename");
        imgAnn.strImageName = strFilename;   // 20260330 ZJH 设置文件名
        imgAnn.strImagePath = strFilename;   // 20260330 ZJH 路径暂设为文件名

        // 20260330 ZJH 提取图像尺寸
        auto [strSize, _2] = detail::extractXmlTag(strXml, "size");
        if (!strSize.empty()) {
            auto [strW, _w] = detail::extractXmlTag(strSize, "width");   // 20260330 ZJH 宽度
            auto [strH, _h] = detail::extractXmlTag(strSize, "height");  // 20260330 ZJH 高度
            // 20260330 ZJH FIX-4: stoi/stof 包裹 try-catch 防止格式异常导致崩溃
            try { imgAnn.nWidth = strW.empty() ? 0 : std::stoi(strW); } catch (...) { imgAnn.nWidth = 0; }   // 20260330 ZJH 解析宽度
            try { imgAnn.nHeight = strH.empty() ? 0 : std::stoi(strH); } catch (...) { imgAnn.nHeight = 0; }  // 20260330 ZJH 解析高度
        }

        // 20260330 ZJH 提取所有 <object> 块
        auto vecObjects = detail::extractAllXmlBlocks(strXml, "object");
        for (const auto& strObj : vecObjects) {
            BBox box;  // 20260330 ZJH 创建目标框

            // 20260330 ZJH 提取类别名称
            auto [strName, _n] = detail::extractXmlTag(strObj, "name");
            box.strClassName = strName;  // 20260330 ZJH 设置类别名

            // 20260330 ZJH 动态分配 classId
            auto itClass = mapClassToId.find(strName);
            if (itClass != mapClassToId.end()) {
                box.nClassId = itClass->second;  // 20260330 ZJH 已有类别
            } else {
                int nNewId = static_cast<int>(mapClassToId.size());  // 20260330 ZJH 新 ID
                mapClassToId[strName] = nNewId;  // 20260330 ZJH 注册新类别
                box.nClassId = nNewId;  // 20260330 ZJH 设置 ID
            }

            // 20260330 ZJH 提取边界框坐标 <bndbox>
            auto [strBndbox, _b] = detail::extractXmlTag(strObj, "bndbox");
            if (!strBndbox.empty()) {
                auto [strXmin, _x1] = detail::extractXmlTag(strBndbox, "xmin");
                auto [strYmin, _y1] = detail::extractXmlTag(strBndbox, "ymin");
                auto [strXmax, _x2] = detail::extractXmlTag(strBndbox, "xmax");
                auto [strYmax, _y2] = detail::extractXmlTag(strBndbox, "ymax");

                // 20260330 ZJH FIX-4: stof 包裹 try-catch 防止格式异常导致崩溃
                float fXmin = 0.0f, fYmin = 0.0f, fXmax = 0.0f, fYmax = 0.0f;
                try { fXmin = strXmin.empty() ? 0.0f : std::stof(strXmin); } catch (...) {}  // 20260330 ZJH 左上 x
                try { fYmin = strYmin.empty() ? 0.0f : std::stof(strYmin); } catch (...) {}  // 20260330 ZJH 左上 y
                try { fXmax = strXmax.empty() ? 0.0f : std::stof(strXmax); } catch (...) {}  // 20260330 ZJH 右下 x
                try { fYmax = strYmax.empty() ? 0.0f : std::stof(strYmax); } catch (...) {}  // 20260330 ZJH 右下 y

                // 20260330 ZJH VOC 使用 (xmin,ymin,xmax,ymax) 像素坐标 → 转为 (x,y,w,h)
                box.fX = fXmin;               // 20260330 ZJH 左上角 x
                box.fY = fYmin;               // 20260330 ZJH 左上角 y
                box.fW = fXmax - fXmin;       // 20260330 ZJH 宽度
                box.fH = fYmax - fYmin;       // 20260330 ZJH 高度
            }

            // 20260330 ZJH 提取置信度（VOC 扩展字段，标准格式不含此字段）
            auto [strConf, _c] = detail::extractXmlTag(strObj, "confidence");
            if (!strConf.empty()) {
                // 20260330 ZJH FIX-4: stof 包裹 try-catch
                try { box.fConfidence = std::stof(strConf); } catch (...) { box.fConfidence = 0.0f; }  // 20260330 ZJH 解析置信度
            }

            imgAnn.vecBoxes.push_back(box);  // 20260330 ZJH 添加目标框
        }

        dataset.vecImages.push_back(imgAnn);  // 20260330 ZJH 添加图像标注
    }

    // 20260330 ZJH 构建类别名称列表（按 classId 排序）
    dataset.vecClassNames.resize(mapClassToId.size());
    for (const auto& [strName, nId] : mapClassToId) {
        dataset.vecClassNames[nId] = strName;  // 20260330 ZJH 填充类别名
    }

    dataset.strDescription = "Imported from Pascal VOC XML: " + strXmlDir;  // 20260330 ZJH 描述
    return dataset;  // 20260330 ZJH 返回结果
}

// =========================================================================
// 20260330 ZJH Pascal VOC XML 导出
// 每张图像生成一个 .xml 文件，格式符合 Pascal VOC 标准
// =========================================================================
inline bool exportVOC(const DatasetAnnotation& dataset, const std::string& strOutputDir) {
    // 20260330 ZJH 创建输出目录
    std::filesystem::create_directories(strOutputDir);

    for (const auto& img : dataset.vecImages) {
        // 20260330 ZJH 从文件名推导输出 XML 文件名
        std::filesystem::path imgPath(img.strImageName);
        std::string strBaseName = imgPath.stem().string();      // 20260330 ZJH 无扩展名文件名
        std::string strOutPath = strOutputDir + "/" + strBaseName + ".xml";

        std::ofstream ofs(strOutPath);  // 20260330 ZJH 打开输出文件
        if (!ofs.is_open()) continue;   // 20260330 ZJH 无法打开则跳过

        // 20260330 ZJH 写入 XML 头部
        ofs << "<annotation>\n";
        ofs << "  <folder>images</folder>\n";
        ofs << "  <filename>" << detail::xmlEscape(img.strImageName) << "</filename>\n";
        ofs << "  <source>\n";
        ofs << "    <database>OmniMatch Export</database>\n";
        ofs << "  </source>\n";
        ofs << "  <size>\n";
        ofs << "    <width>" << img.nWidth << "</width>\n";
        ofs << "    <height>" << img.nHeight << "</height>\n";
        ofs << "    <depth>3</depth>\n";
        ofs << "  </size>\n";
        ofs << "  <segmented>0</segmented>\n";

        // 20260330 ZJH 写入每个目标对象
        for (const auto& box : img.vecBoxes) {
            // 20260330 ZJH 将 (x,y,w,h) 转为 VOC 的 (xmin,ymin,xmax,ymax)
            int nXmin = static_cast<int>(std::round(box.fX));              // 20260330 ZJH 左上 x
            int nYmin = static_cast<int>(std::round(box.fY));              // 20260330 ZJH 左上 y
            int nXmax = static_cast<int>(std::round(box.fX + box.fW));     // 20260330 ZJH 右下 x
            int nYmax = static_cast<int>(std::round(box.fY + box.fH));     // 20260330 ZJH 右下 y

            // 20260330 ZJH 裁剪到图像边界
            nXmin = std::max(0, std::min(nXmin, img.nWidth));
            nYmin = std::max(0, std::min(nYmin, img.nHeight));
            nXmax = std::max(0, std::min(nXmax, img.nWidth));
            nYmax = std::max(0, std::min(nYmax, img.nHeight));

            // 20260330 ZJH 确定类别名称
            std::string strClassName = box.strClassName;
            if (strClassName.empty() && box.nClassId >= 0 &&
                box.nClassId < static_cast<int>(dataset.vecClassNames.size())) {
                strClassName = dataset.vecClassNames[box.nClassId];  // 20260330 ZJH 从类别列表查
            }
            if (strClassName.empty()) {
                strClassName = "class_" + std::to_string(box.nClassId);  // 20260330 ZJH 兜底名称
            }

            ofs << "  <object>\n";
            ofs << "    <name>" << detail::xmlEscape(strClassName) << "</name>\n";
            ofs << "    <pose>Unspecified</pose>\n";
            ofs << "    <truncated>0</truncated>\n";
            ofs << "    <difficult>0</difficult>\n";
            ofs << "    <bndbox>\n";
            ofs << "      <xmin>" << nXmin << "</xmin>\n";
            ofs << "      <ymin>" << nYmin << "</ymin>\n";
            ofs << "      <xmax>" << nXmax << "</xmax>\n";
            ofs << "      <ymax>" << nYmax << "</ymax>\n";
            ofs << "    </bndbox>\n";
            ofs << "  </object>\n";
        }

        ofs << "</annotation>\n";  // 20260330 ZJH XML 根元素结束
        ofs.close();  // 20260330 ZJH 关闭文件
    }

    return true;  // 20260330 ZJH 导出成功
}

// =========================================================================
// 20260330 ZJH YOLO txt 导入
// 格式: 每行 "classId cx cy w h"（归一化到 [0,1]）
// 需要 classes.txt 提供类别名映射，imageDir 获取图像尺寸
// =========================================================================
inline DatasetAnnotation importYOLO(const std::string& strTxtDir,
                                     const std::string& strClassesFile,
                                     const std::string& strImageDir) {
    DatasetAnnotation dataset;  // 20260330 ZJH 输出结果

    // ---- Step 1: 读取 classes.txt → 类别名列表 ----
    {
        std::ifstream ifs(strClassesFile);  // 20260330 ZJH 打开类别文件
        if (ifs.is_open()) {
            std::string strLine;
            while (std::getline(ifs, strLine)) {
                // 20260330 ZJH 去除行尾空白
                while (!strLine.empty() && (strLine.back() == '\r' || strLine.back() == '\n' ||
                       strLine.back() == ' ')) {
                    strLine.pop_back();
                }
                if (!strLine.empty()) {
                    dataset.vecClassNames.push_back(strLine);  // 20260330 ZJH 添加类别名
                }
            }
            ifs.close();  // 20260330 ZJH 关闭文件
        }
    }

    // ---- Step 2: 建立 imageName→imagePath 映射（用于匹配 txt 和图像） ----
    std::map<std::string, std::string> mapBaseToImgPath;  // 20260330 ZJH 无扩展名→完整路径
    std::map<std::string, std::pair<int, int>> mapBaseToSize;  // 20260330 ZJH 无扩展名→(宽,高)
    if (std::filesystem::exists(strImageDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(strImageDir)) {
            if (!entry.is_regular_file()) continue;  // 20260330 ZJH 跳过非文件
            std::string strExt = entry.path().extension().string();  // 20260330 ZJH 扩展名
            if (!detail::isImageExtension(strExt)) continue;  // 20260330 ZJH 非图像跳过

            std::string strBase = entry.path().stem().string();  // 20260330 ZJH 无扩展名文件名
            mapBaseToImgPath[strBase] = entry.path().string();   // 20260330 ZJH 完整路径
            // 20260330 ZJH 图像尺寸需从文件头读取（此处先设 0，由调用方填充或使用默认值）
            // 实际项目中应调用 OpenCV 或 stb_image 读取尺寸
            mapBaseToSize[strBase] = {0, 0};
        }
    }

    // ---- Step 3: 遍历 txt 目录，解析每个标签文件 ----
    if (!std::filesystem::exists(strTxtDir)) return dataset;  // 20260330 ZJH 目录不存在

    for (const auto& entry : std::filesystem::directory_iterator(strTxtDir)) {
        if (!entry.is_regular_file()) continue;  // 20260330 ZJH 跳过非文件
        if (entry.path().extension().string() != ".txt") continue;  // 20260330 ZJH 只处理 .txt
        // 20260330 ZJH 跳过 classes.txt 本身
        if (entry.path().filename().string() == "classes.txt") continue;

        std::string strBaseName = entry.path().stem().string();  // 20260330 ZJH 无扩展名文件名

        ImageAnnotation imgAnn;  // 20260330 ZJH 创建图像标注
        imgAnn.strImageName = strBaseName;  // 20260330 ZJH 设置文件名

        // 20260330 ZJH 查找对应图像路径和尺寸
        auto itPath = mapBaseToImgPath.find(strBaseName);
        if (itPath != mapBaseToImgPath.end()) {
            imgAnn.strImagePath = itPath->second;  // 20260330 ZJH 设置图像路径
            imgAnn.strImageName = std::filesystem::path(itPath->second).filename().string();
        }
        auto itSize = mapBaseToSize.find(strBaseName);
        if (itSize != mapBaseToSize.end()) {
            imgAnn.nWidth = itSize->second.first;   // 20260330 ZJH 宽度
            imgAnn.nHeight = itSize->second.second;  // 20260330 ZJH 高度
        }

        // 20260330 ZJH 读取标签文件
        std::ifstream ifs(entry.path().string());
        if (!ifs.is_open()) continue;  // 20260330 ZJH 无法打开跳过

        std::string strLine;
        while (std::getline(ifs, strLine)) {
            if (strLine.empty()) continue;  // 20260330 ZJH 跳过空行
            // 20260330 ZJH 去除回车
            while (!strLine.empty() && (strLine.back() == '\r' || strLine.back() == '\n')) {
                strLine.pop_back();
            }
            if (strLine.empty()) continue;

            std::istringstream iss(strLine);  // 20260330 ZJH 解析行
            int nClassId = 0;
            float fCx = 0.0f, fCy = 0.0f, fW = 0.0f, fH = 0.0f;
            iss >> nClassId >> fCx >> fCy >> fW >> fH;  // 20260330 ZJH 读取 class cx cy w h

            BBox box;  // 20260330 ZJH 创建目标框
            box.nClassId = nClassId;  // 20260330 ZJH 类别 ID

            // 20260330 ZJH 设置类别名称
            if (nClassId >= 0 && nClassId < static_cast<int>(dataset.vecClassNames.size())) {
                box.strClassName = dataset.vecClassNames[nClassId];
            }

            // 20260330 ZJH YOLO 格式: cx cy w h 归一化 [0,1] → 反归一化为像素坐标
            if (imgAnn.nWidth > 0 && imgAnn.nHeight > 0) {
                // 20260330 ZJH 有图像尺寸：反归一化
                float fPixW = fW * imgAnn.nWidth;   // 20260330 ZJH 像素宽度
                float fPixH = fH * imgAnn.nHeight;  // 20260330 ZJH 像素高度
                box.fX = fCx * imgAnn.nWidth - fPixW * 0.5f;   // 20260330 ZJH 左上角 x
                box.fY = fCy * imgAnn.nHeight - fPixH * 0.5f;  // 20260330 ZJH 左上角 y
                box.fW = fPixW;  // 20260330 ZJH 宽度
                box.fH = fPixH;  // 20260330 ZJH 高度
            } else {
                // 20260330 ZJH 无图像尺寸：保持归一化坐标（用户后续需手动反归一化）
                box.fX = fCx - fW * 0.5f;  // 20260330 ZJH 归一化左上角 x
                box.fY = fCy - fH * 0.5f;  // 20260330 ZJH 归一化左上角 y
                box.fW = fW;               // 20260330 ZJH 归一化宽度
                box.fH = fH;               // 20260330 ZJH 归一化高度
            }

            imgAnn.vecBoxes.push_back(box);  // 20260330 ZJH 添加目标框
        }
        ifs.close();  // 20260330 ZJH 关闭文件

        dataset.vecImages.push_back(imgAnn);  // 20260330 ZJH 添加图像标注
    }

    dataset.strDescription = "Imported from YOLO txt: " + strTxtDir;  // 20260330 ZJH 描述
    return dataset;  // 20260330 ZJH 返回结果
}

// =========================================================================
// 20260330 ZJH YOLO txt 导出
// 每张图像一个 .txt（classId cx cy w h 归一化），同时输出 classes.txt
// =========================================================================
inline bool exportYOLO(const DatasetAnnotation& dataset, const std::string& strOutputDir) {
    // 20260330 ZJH 创建输出目录结构: labels/ + classes.txt
    std::string strLabelsDir = strOutputDir + "/labels";  // 20260330 ZJH 标签子目录
    std::filesystem::create_directories(strLabelsDir);

    // ---- Step 1: 写入 classes.txt ----
    {
        std::string strClassesPath = strOutputDir + "/classes.txt";
        std::ofstream ofs(strClassesPath);  // 20260330 ZJH 打开类别文件
        if (!ofs.is_open()) return false;   // 20260330 ZJH 打开失败
        for (const auto& strName : dataset.vecClassNames) {
            ofs << strName << "\n";  // 20260330 ZJH 每行一个类别
        }
        ofs.close();  // 20260330 ZJH 关闭文件
    }

    // ---- Step 2: 为每张图像写入标签 txt ----
    for (const auto& img : dataset.vecImages) {
        // 20260330 ZJH 推导输出文件名
        std::filesystem::path imgPath(img.strImageName);
        std::string strBaseName = imgPath.stem().string();  // 20260330 ZJH 无扩展名
        std::string strOutPath = strLabelsDir + "/" + strBaseName + ".txt";

        std::ofstream ofs(strOutPath);  // 20260330 ZJH 打开输出文件
        if (!ofs.is_open()) continue;   // 20260330 ZJH 无法打开跳过

        for (const auto& box : img.vecBoxes) {
            float fCx, fCy, fNormW, fNormH;  // 20260330 ZJH 归一化中心坐标和尺寸

            if (img.nWidth > 0 && img.nHeight > 0) {
                // 20260330 ZJH 像素坐标 → 归一化坐标
                fCx = (box.fX + box.fW * 0.5f) / img.nWidth;    // 20260330 ZJH 归一化中心 x
                fCy = (box.fY + box.fH * 0.5f) / img.nHeight;   // 20260330 ZJH 归一化中心 y
                fNormW = box.fW / img.nWidth;                     // 20260330 ZJH 归一化宽度
                fNormH = box.fH / img.nHeight;                    // 20260330 ZJH 归一化高度
            } else {
                // 20260330 ZJH 无图像尺寸：假设已经是归一化坐标
                fCx = box.fX + box.fW * 0.5f;   // 20260330 ZJH 中心 x
                fCy = box.fY + box.fH * 0.5f;   // 20260330 ZJH 中心 y
                fNormW = box.fW;                 // 20260330 ZJH 宽度
                fNormH = box.fH;                 // 20260330 ZJH 高度
            }

            // 20260330 ZJH 裁剪到 [0, 1] 范围
            fCx = std::max(0.0f, std::min(1.0f, fCx));
            fCy = std::max(0.0f, std::min(1.0f, fCy));
            fNormW = std::max(0.0f, std::min(1.0f, fNormW));
            fNormH = std::max(0.0f, std::min(1.0f, fNormH));

            // 20260330 ZJH 输出: classId cx cy w h
            char arrBuf[256];
            std::snprintf(arrBuf, sizeof(arrBuf), "%d %.6f %.6f %.6f %.6f",
                box.nClassId, fCx, fCy, fNormW, fNormH);
            ofs << arrBuf << "\n";
        }

        ofs.close();  // 20260330 ZJH 关闭文件
    }

    return true;  // 20260330 ZJH 导出成功
}

// =========================================================================
// 20260330 ZJH 一键格式互转
// 参数: strInputPath - 输入路径（COCO 为 .json 文件，VOC/YOLO 为目录）
//       strOutputPath - 输出路径（COCO 为 .json 文件，VOC/YOLO 为目录）
//       strFromFormat - 源格式 "coco" / "voc" / "yolo"
//       strToFormat   - 目标格式 "coco" / "voc" / "yolo"
// =========================================================================
inline bool convertFormat(const std::string& strInputPath,
                          const std::string& strOutputPath,
                          const std::string& strFromFormat,
                          const std::string& strToFormat) {
    // ---- Step 1: 导入为中间表示 ----
    DatasetAnnotation dataset;  // 20260330 ZJH 中间表示

    if (strFromFormat == "coco") {
        dataset = importCOCO(strInputPath);  // 20260330 ZJH 从 COCO JSON 导入
    } else if (strFromFormat == "voc") {
        dataset = importVOC(strInputPath);  // 20260330 ZJH 从 VOC XML 导入
    } else if (strFromFormat == "yolo") {
        // 20260330 ZJH YOLO 导入需要 classes.txt 和图像目录
        // 约定: strInputPath 为标签目录，classes.txt 在同级目录，images/ 在同级目录
        std::filesystem::path inputPath(strInputPath);
        std::string strParent = inputPath.parent_path().string();      // 20260330 ZJH 父目录
        std::string strClasses = strParent + "/classes.txt";           // 20260330 ZJH 类别文件
        std::string strImgDir = strParent + "/images";                 // 20260330 ZJH 图像目录
        // 20260330 ZJH 如果 classes.txt 在标签目录内，回退查找
        if (!std::filesystem::exists(strClasses)) {
            strClasses = strInputPath + "/classes.txt";
        }
        if (!std::filesystem::exists(strImgDir)) {
            strImgDir = strInputPath;  // 20260330 ZJH 回退为标签目录本身
        }
        dataset = importYOLO(strInputPath, strClasses, strImgDir);  // 20260330 ZJH 导入
    } else {
        return false;  // 20260330 ZJH 不支持的源格式
    }

    // 20260330 ZJH 检查导入结果
    if (dataset.vecImages.empty()) return false;  // 20260330 ZJH 无数据

    // ---- Step 2: 导出为目标格式 ----
    if (strToFormat == "coco") {
        return exportCOCO(dataset, strOutputPath);  // 20260330 ZJH 导出为 COCO
    } else if (strToFormat == "voc") {
        return exportVOC(dataset, strOutputPath);  // 20260330 ZJH 导出为 VOC
    } else if (strToFormat == "yolo") {
        return exportYOLO(dataset, strOutputPath);  // 20260330 ZJH 导出为 YOLO
    }

    return false;  // 20260330 ZJH 不支持的目标格式
}

// =========================================================================
// 20260330 ZJH 从 OmniMatch 项目导出为标准格式
// 读取 .omdl 项目目录中的 annotations/ 和 images.json，转为指定格式
// =========================================================================
inline bool exportFromProject(const std::string& strProjectDir,
                              const std::string& strOutputPath,
                              const std::string& strFormat) {
    DatasetAnnotation dataset;  // 20260330 ZJH 中间表示

    // 20260330 ZJH 读取 labels.json → 类别列表
    std::string strLabelsJson = detail::readFileContents(strProjectDir + "/labels.json");
    if (!strLabelsJson.empty()) {
        // 20260330 ZJH 提取 labels 数组
        auto vecLabelElems = detail::getArrayVal(strLabelsJson, "labels");
        for (const auto& strLabel : vecLabelElems) {
            // 20260330 ZJH 每个 label 对象提取 name 字段
            std::string strName = detail::getStringVal(strLabel, "name");
            if (!strName.empty()) {
                dataset.vecClassNames.push_back(strName);  // 20260330 ZJH 添加类别
            }
        }
    }

    // 20260330 ZJH 读取 images.json → 图像列表
    std::string strImagesJson = detail::readFileContents(strProjectDir + "/images.json");
    if (!strImagesJson.empty()) {
        auto vecImageElems = detail::getArrayVal(strImagesJson, "images");
        for (const auto& strImg : vecImageElems) {
            ImageAnnotation imgAnn;  // 20260330 ZJH 创建图像标注
            imgAnn.strImagePath = detail::getStringVal(strImg, "path");        // 20260330 ZJH 路径
            imgAnn.strImageName = detail::getStringVal(strImg, "filename");    // 20260330 ZJH 文件名
            imgAnn.nWidth = static_cast<int>(detail::getNumberVal(strImg, "width"));    // 20260330 ZJH 宽度
            imgAnn.nHeight = static_cast<int>(detail::getNumberVal(strImg, "height"));  // 20260330 ZJH 高度

            // 20260330 ZJH 读取对应的标注文件
            std::string strId = detail::getStringVal(strImg, "id");  // 20260330 ZJH 图像 ID
            if (!strId.empty()) {
                std::string strAnnPath = strProjectDir + "/annotations/" + strId + ".json";
                std::string strAnnJson = detail::readFileContents(strAnnPath);
                if (!strAnnJson.empty()) {
                    auto vecAnnElems = detail::getArrayVal(strAnnJson, "annotations");
                    for (const auto& strAnn : vecAnnElems) {
                        BBox box;  // 20260330 ZJH 创建目标框
                        box.nClassId = static_cast<int>(detail::getNumberVal(strAnn, "label_id"));
                        box.fConfidence = 1.0f;  // 20260330 ZJH 人工标注置信度为 1

                        // 20260330 ZJH 读取矩形坐标
                        box.fX = static_cast<float>(detail::getNumberVal(strAnn, "x"));
                        box.fY = static_cast<float>(detail::getNumberVal(strAnn, "y"));
                        box.fW = static_cast<float>(detail::getNumberVal(strAnn, "width"));
                        box.fH = static_cast<float>(detail::getNumberVal(strAnn, "height"));

                        // 20260330 ZJH 设置类别名
                        if (box.nClassId >= 0 && box.nClassId < static_cast<int>(dataset.vecClassNames.size())) {
                            box.strClassName = dataset.vecClassNames[box.nClassId];
                        }

                        imgAnn.vecBoxes.push_back(box);  // 20260330 ZJH 添加框
                    }
                }
            }

            dataset.vecImages.push_back(imgAnn);  // 20260330 ZJH 添加图像
        }
    }

    // 20260330 ZJH 导出为指定格式
    if (strFormat == "coco") {
        return exportCOCO(dataset, strOutputPath);
    } else if (strFormat == "voc") {
        return exportVOC(dataset, strOutputPath);
    } else if (strFormat == "yolo") {
        return exportYOLO(dataset, strOutputPath);
    }

    return false;  // 20260330 ZJH 不支持的格式
}

// =========================================================================
// 20260330 ZJH 从标准格式导入到 OmniMatch 项目
// 解析外部格式后写入 .omdl 目录结构（labels.json + images.json + annotations/）
// =========================================================================
inline bool importToProject(const std::string& strInputPath,
                            const std::string& strFormat,
                            const std::string& strProjectDir) {
    // ---- Step 1: 导入为中间表示 ----
    DatasetAnnotation dataset;  // 20260330 ZJH 中间表示

    if (strFormat == "coco") {
        dataset = importCOCO(strInputPath);
    } else if (strFormat == "voc") {
        dataset = importVOC(strInputPath);
    } else if (strFormat == "yolo") {
        // 20260330 ZJH YOLO 格式的目录约定（同 convertFormat）
        std::filesystem::path inputPath(strInputPath);
        std::string strParent = inputPath.parent_path().string();
        std::string strClasses = strParent + "/classes.txt";
        std::string strImgDir = strParent + "/images";
        if (!std::filesystem::exists(strClasses)) {
            strClasses = strInputPath + "/classes.txt";
        }
        if (!std::filesystem::exists(strImgDir)) {
            strImgDir = strInputPath;
        }
        dataset = importYOLO(strInputPath, strClasses, strImgDir);
    } else {
        return false;  // 20260330 ZJH 不支持的格式
    }

    if (dataset.vecImages.empty()) return false;  // 20260330 ZJH 无数据

    // ---- Step 2: 创建 .omdl 项目目录结构 ----
    std::filesystem::create_directories(strProjectDir);
    std::filesystem::create_directories(strProjectDir + "/annotations");

    // ---- Step 3: 写入 labels.json ----
    {
        std::string strLabelsPath = strProjectDir + "/labels.json";
        std::ofstream ofs(strLabelsPath);
        if (!ofs.is_open()) return false;  // 20260330 ZJH 无法打开

        ofs << "{\n  \"labels\": [\n";
        for (size_t i = 0; i < dataset.vecClassNames.size(); ++i) {
            ofs << "    {\"id\": " << i
                << ", \"name\": \"" << detail::jsonEscape(dataset.vecClassNames[i])
                << "\", \"color\": \"#" << std::hex;
            // 20260330 ZJH 自动分配颜色（基于索引的 HSV 色环映射）
            int nHue = static_cast<int>((i * 137) % 360);  // 20260330 ZJH 黄金角色环分布
            // 20260330 ZJH 简化 HSV→RGB: 固定 S=0.8 V=0.9
            float fH = nHue / 60.0f;
            int nSector = static_cast<int>(fH) % 6;
            float fFrac = fH - static_cast<int>(fH);
            int nV = 230;  // 20260330 ZJH V*255
            int nP = 46;   // 20260330 ZJH V*(1-S)*255
            int nQ = static_cast<int>(nV * (1.0f - 0.8f * fFrac));  // 20260330 ZJH V*(1-S*f)*255
            int nT = static_cast<int>(nV * (1.0f - 0.8f * (1.0f - fFrac)));  // 20260330 ZJH V*(1-S*(1-f))*255
            int nR = 0, nG = 0, nB = 0;
            switch (nSector) {
                case 0: nR = nV; nG = nT; nB = nP; break;
                case 1: nR = nQ; nG = nV; nB = nP; break;
                case 2: nR = nP; nG = nV; nB = nT; break;
                case 3: nR = nP; nG = nQ; nB = nV; break;
                case 4: nR = nT; nG = nP; nB = nV; break;
                case 5: nR = nV; nG = nP; nB = nQ; break;
            }
            char arrColor[8];
            std::snprintf(arrColor, sizeof(arrColor), "%02x%02x%02x", nR, nG, nB);
            ofs << std::dec;  // 20260330 ZJH 恢复十进制
            ofs << arrColor << "\"}";
            if (i + 1 < dataset.vecClassNames.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ]\n}\n";
        ofs.close();  // 20260330 ZJH 关闭文件
    }

    // ---- Step 4: 写入 images.json + 每张图像的 annotations/ID.json ----
    {
        std::string strImagesPath = strProjectDir + "/images.json";
        std::ofstream ofsImg(strImagesPath);
        if (!ofsImg.is_open()) return false;  // 20260330 ZJH 无法打开

        ofsImg << "{\n  \"images\": [\n";
        for (size_t i = 0; i < dataset.vecImages.size(); ++i) {
            const auto& img = dataset.vecImages[i];
            // 20260330 ZJH 生成简单的图像 ID（基于索引）
            char arrId[32];
            std::snprintf(arrId, sizeof(arrId), "img_%06d", static_cast<int>(i));
            std::string strId(arrId);

            ofsImg << "    {\"id\": \"" << strId
                   << "\", \"filename\": \"" << detail::jsonEscape(img.strImageName)
                   << "\", \"path\": \"" << detail::jsonEscape(img.strImagePath)
                   << "\", \"width\": " << img.nWidth
                   << ", \"height\": " << img.nHeight << "}";
            if (i + 1 < dataset.vecImages.size()) ofsImg << ",";
            ofsImg << "\n";

            // 20260330 ZJH 写入标注文件
            if (!img.vecBoxes.empty()) {
                std::string strAnnPath = strProjectDir + "/annotations/" + strId + ".json";
                std::ofstream ofsAnn(strAnnPath);
                if (ofsAnn.is_open()) {
                    ofsAnn << "{\n  \"annotations\": [\n";
                    for (size_t j = 0; j < img.vecBoxes.size(); ++j) {
                        const auto& box = img.vecBoxes[j];
                        char arrAnn[512];
                        std::snprintf(arrAnn, sizeof(arrAnn),
                            "    {\"label_id\": %d, \"type\": \"rect\", "
                            "\"x\": %.2f, \"y\": %.2f, \"width\": %.2f, \"height\": %.2f, "
                            "\"confidence\": %.4f}",
                            box.nClassId, box.fX, box.fY, box.fW, box.fH, box.fConfidence);
                        ofsAnn << arrAnn;
                        if (j + 1 < img.vecBoxes.size()) ofsAnn << ",";
                        ofsAnn << "\n";
                    }
                    ofsAnn << "  ]\n}\n";
                    ofsAnn.close();
                }
            }
        }
        ofsImg << "  ]\n}\n";
        ofsImg.close();
    }

    return true;  // 20260330 ZJH 导入成功
}

}  // namespace om
