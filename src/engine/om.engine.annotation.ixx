// 20260320 ZJH 图像标注数据结构与序列化模块 — Phase 5
// 支持多种标注类型：矩形框、多边形、画笔、点标注、图像级标签
// 支持多种导出格式：YOLO、VOC XML、COCO JSON、OmniMatch 内部 JSON
module;

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdio>
#include <cmath>
#include <algorithm>

export module om.engine.annotation;

export namespace om {

// 20260320 ZJH 标注类型枚举 — 定义支持的标注方式
enum class AnnotationType {
    BBox,        // 20260320 ZJH 矩形框（目标检测）
    Polygon,     // 20260320 ZJH 多边形（语义分割/实例分割）
    Brush,       // 20260320 ZJH 画笔（像素级标注）
    Point,       // 20260320 ZJH 点标注（关键点检测）
    ImageLabel   // 20260320 ZJH 图像级标签（分类）
};

// 20260320 ZJH 2D 点结构 — 归一化坐标 (0-1)
struct Point2f {
    float fX = 0.0f;  // 20260320 ZJH X 坐标（归一化 0-1）
    float fY = 0.0f;  // 20260320 ZJH Y 坐标（归一化 0-1）
};

// 20260320 ZJH 矩形框标注 — 用于目标检测任务
struct BBoxAnnotation {
    float fX = 0.0f;       // 20260320 ZJH 左上角 X（归一化 0-1）
    float fY = 0.0f;       // 20260320 ZJH 左上角 Y（归一化 0-1）
    float fWidth = 0.0f;   // 20260320 ZJH 宽度（归一化 0-1）
    float fHeight = 0.0f;  // 20260320 ZJH 高度（归一化 0-1）
    int nClassId = 0;       // 20260320 ZJH 类别 ID
    std::string strClassName;  // 20260320 ZJH 类别名称
};

// 20260320 ZJH 多边形标注 — 用于分割任务
struct PolygonAnnotation {
    std::vector<Point2f> vecPoints;  // 20260320 ZJH 顶点列表（归一化坐标）
    int nClassId = 0;                 // 20260320 ZJH 类别 ID
    std::string strClassName;         // 20260320 ZJH 类别名称
};

// 20260320 ZJH 单张图像的标注集合
struct ImageAnnotation {
    std::string strImagePath;               // 20260320 ZJH 图像文件路径
    int nImageWidth = 0;                     // 20260320 ZJH 图像原始宽度（像素）
    int nImageHeight = 0;                    // 20260320 ZJH 图像原始高度（像素）
    std::vector<BBoxAnnotation> vecBBoxes;   // 20260320 ZJH 矩形框标注列表
    std::vector<PolygonAnnotation> vecPolygons;  // 20260320 ZJH 多边形标注列表
    int nImageClassId = -1;                  // 20260320 ZJH 图像级类别（分类任务，-1 表示未标注）
};

// 20260320 ZJH 标注项目 — 包含所有标注信息
struct AnnotationProject {
    std::string strName;                         // 20260320 ZJH 项目名称
    std::vector<std::string> vecClassNames;      // 20260320 ZJH 类别名称列表
    std::vector<ImageAnnotation> vecAnnotations; // 20260320 ZJH 所有图像的标注
    AnnotationType annotType = AnnotationType::BBox;  // 20260320 ZJH 标注类型
};

// ============================================================================
// 20260320 ZJH 保存标注到 YOLO 格式
// 每张图像一个 .txt 文件，每行格式: class_id center_x center_y width height
// 所有坐标归一化到 [0, 1]
// ============================================================================
inline void saveAnnotationsYOLO(const AnnotationProject& project, const std::string& strOutputDir) {
    // 20260320 ZJH 创建输出目录
    std::filesystem::create_directories(strOutputDir);

    // 20260320 ZJH 遍历所有图像标注
    for (const auto& ann : project.vecAnnotations) {
        // 20260320 ZJH 从图像路径提取文件名（不含扩展名）
        std::filesystem::path imgPath(ann.strImagePath);
        std::string strBaseName = imgPath.stem().string();  // 20260320 ZJH 获取无扩展名的文件名

        // 20260320 ZJH 构造输出 .txt 文件路径
        std::string strOutPath = strOutputDir + "/" + strBaseName + ".txt";
        std::ofstream ofs(strOutPath);  // 20260320 ZJH 打开输出文件

        if (!ofs.is_open()) continue;  // 20260320 ZJH 无法打开文件则跳过

        // 20260320 ZJH 写入每个矩形框标注
        for (const auto& bbox : ann.vecBBoxes) {
            // 20260320 ZJH 计算 YOLO 格式的中心坐标
            float fCx = bbox.fX + bbox.fWidth * 0.5f;   // 20260320 ZJH 中心 X
            float fCy = bbox.fY + bbox.fHeight * 0.5f;  // 20260320 ZJH 中心 Y
            // 20260320 ZJH 写入: class_id cx cy w h
            char arrBuf[256];
            std::snprintf(arrBuf, sizeof(arrBuf), "%d %.6f %.6f %.6f %.6f",
                bbox.nClassId, fCx, fCy, bbox.fWidth, bbox.fHeight);
            ofs << arrBuf << "\n";
        }

        ofs.close();  // 20260320 ZJH 关闭文件
    }
}

// ============================================================================
// 20260320 ZJH 加载 YOLO 格式标注
// 从图像目录和标签目录读取标注数据
// ============================================================================
inline AnnotationProject loadAnnotationsYOLO(const std::string& strImageDir,
                                              const std::string& strLabelDir,
                                              const std::vector<std::string>& vecClassNames) {
    AnnotationProject project;  // 20260320 ZJH 创建标注项目
    project.strName = "YOLO Import";  // 20260320 ZJH 默认项目名
    project.vecClassNames = vecClassNames;  // 20260320 ZJH 设置类别名称
    project.annotType = AnnotationType::BBox;  // 20260320 ZJH YOLO 默认为矩形框

    // 20260320 ZJH 扫描图像目录
    if (!std::filesystem::exists(strImageDir)) return project;

    for (const auto& entry : std::filesystem::directory_iterator(strImageDir)) {
        if (!entry.is_regular_file()) continue;  // 20260320 ZJH 跳过非文件

        std::string strExt = entry.path().extension().string();  // 20260320 ZJH 获取扩展名
        // 20260320 ZJH 只处理常见图像格式
        if (strExt != ".jpg" && strExt != ".jpeg" && strExt != ".png" && strExt != ".bmp") continue;

        ImageAnnotation ann;  // 20260320 ZJH 创建图像标注
        ann.strImagePath = entry.path().string();  // 20260320 ZJH 设置图像路径

        // 20260320 ZJH 查找对应的标签文件
        std::string strBaseName = entry.path().stem().string();
        std::string strLabelPath = strLabelDir + "/" + strBaseName + ".txt";

        std::ifstream ifs(strLabelPath);  // 20260320 ZJH 打开标签文件
        if (ifs.is_open()) {
            std::string strLine;
            // 20260320 ZJH 逐行读取标注
            while (std::getline(ifs, strLine)) {
                if (strLine.empty()) continue;  // 20260320 ZJH 跳过空行

                std::istringstream iss(strLine);  // 20260320 ZJH 解析行
                int nClassId = 0;
                float fCx = 0.0f, fCy = 0.0f, fW = 0.0f, fH = 0.0f;
                iss >> nClassId >> fCx >> fCy >> fW >> fH;  // 20260320 ZJH 读取 class cx cy w h

                BBoxAnnotation bbox;
                bbox.nClassId = nClassId;          // 20260320 ZJH 类别 ID
                bbox.fX = fCx - fW * 0.5f;        // 20260320 ZJH 左上角 X = cx - w/2
                bbox.fY = fCy - fH * 0.5f;        // 20260320 ZJH 左上角 Y = cy - h/2
                bbox.fWidth = fW;                  // 20260320 ZJH 宽度
                bbox.fHeight = fH;                 // 20260320 ZJH 高度
                // 20260320 ZJH 设置类别名称
                if (nClassId >= 0 && nClassId < static_cast<int>(vecClassNames.size())) {
                    bbox.strClassName = vecClassNames[nClassId];
                }
                ann.vecBBoxes.push_back(bbox);  // 20260320 ZJH 添加到标注列表
            }
            ifs.close();  // 20260320 ZJH 关闭文件
        }

        project.vecAnnotations.push_back(ann);  // 20260320 ZJH 添加图像标注
    }

    return project;  // 20260320 ZJH 返回标注项目
}

// ============================================================================
// 20260320 ZJH 保存标注到 VOC XML 格式
// 每张图像一个 .xml 文件（Pascal VOC 格式）
// ============================================================================
inline void saveAnnotationsVOC(const AnnotationProject& project, const std::string& strOutputDir) {
    // 20260320 ZJH 创建输出目录
    std::filesystem::create_directories(strOutputDir);

    for (const auto& ann : project.vecAnnotations) {
        std::filesystem::path imgPath(ann.strImagePath);
        std::string strBaseName = imgPath.stem().string();
        std::string strOutPath = strOutputDir + "/" + strBaseName + ".xml";

        std::ofstream ofs(strOutPath);
        if (!ofs.is_open()) continue;

        // 20260320 ZJH 写入 XML 头
        ofs << "<annotation>\n";
        ofs << "  <folder>" << imgPath.parent_path().filename().string() << "</folder>\n";
        ofs << "  <filename>" << imgPath.filename().string() << "</filename>\n";
        ofs << "  <size>\n";
        ofs << "    <width>" << ann.nImageWidth << "</width>\n";
        ofs << "    <height>" << ann.nImageHeight << "</height>\n";
        ofs << "    <depth>3</depth>\n";
        ofs << "  </size>\n";

        // 20260320 ZJH 写入每个标注对象
        for (const auto& bbox : ann.vecBBoxes) {
            // 20260320 ZJH 将归一化坐标转换为像素坐标
            int nXmin = static_cast<int>(bbox.fX * ann.nImageWidth);
            int nYmin = static_cast<int>(bbox.fY * ann.nImageHeight);
            int nXmax = static_cast<int>((bbox.fX + bbox.fWidth) * ann.nImageWidth);
            int nYmax = static_cast<int>((bbox.fY + bbox.fHeight) * ann.nImageHeight);

            ofs << "  <object>\n";
            ofs << "    <name>" << bbox.strClassName << "</name>\n";
            ofs << "    <difficult>0</difficult>\n";
            ofs << "    <bndbox>\n";
            ofs << "      <xmin>" << nXmin << "</xmin>\n";
            ofs << "      <ymin>" << nYmin << "</ymin>\n";
            ofs << "      <xmax>" << nXmax << "</xmax>\n";
            ofs << "      <ymax>" << nYmax << "</ymax>\n";
            ofs << "    </bndbox>\n";
            ofs << "  </object>\n";
        }

        ofs << "</annotation>\n";
        ofs.close();
    }
}

// ============================================================================
// 20260320 ZJH 保存标注到 COCO JSON 格式
// 所有标注保存到一个 JSON 文件
// ============================================================================
inline void saveAnnotationsCOCO(const AnnotationProject& project, const std::string& strOutputPath) {
    std::ofstream ofs(strOutputPath);
    if (!ofs.is_open()) return;

    // 20260320 ZJH 手动构建 JSON（不依赖 nlohmann_json，避免模块间依赖）
    ofs << "{\n";

    // 20260320 ZJH 类别列表
    ofs << "  \"categories\": [\n";
    for (size_t i = 0; i < project.vecClassNames.size(); ++i) {
        ofs << "    {\"id\": " << i << ", \"name\": \"" << project.vecClassNames[i] << "\"}";
        if (i + 1 < project.vecClassNames.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    // 20260320 ZJH 图像列表
    ofs << "  \"images\": [\n";
    for (size_t i = 0; i < project.vecAnnotations.size(); ++i) {
        const auto& ann = project.vecAnnotations[i];
        std::filesystem::path imgPath(ann.strImagePath);
        ofs << "    {\"id\": " << i
            << ", \"file_name\": \"" << imgPath.filename().string()
            << "\", \"width\": " << ann.nImageWidth
            << ", \"height\": " << ann.nImageHeight << "}";
        if (i + 1 < project.vecAnnotations.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    // 20260320 ZJH 标注列表
    ofs << "  \"annotations\": [\n";
    int nAnnId = 0;  // 20260320 ZJH 全局标注 ID 计数器
    for (size_t imgIdx = 0; imgIdx < project.vecAnnotations.size(); ++imgIdx) {
        const auto& ann = project.vecAnnotations[imgIdx];
        for (size_t bIdx = 0; bIdx < ann.vecBBoxes.size(); ++bIdx) {
            const auto& bbox = ann.vecBBoxes[bIdx];
            // 20260320 ZJH COCO 使用像素坐标 [x, y, width, height]
            float fPxX = bbox.fX * ann.nImageWidth;
            float fPxY = bbox.fY * ann.nImageHeight;
            float fPxW = bbox.fWidth * ann.nImageWidth;
            float fPxH = bbox.fHeight * ann.nImageHeight;
            float fArea = fPxW * fPxH;  // 20260320 ZJH 面积

            if (nAnnId > 0) ofs << ",\n";
            char arrBuf[512];
            std::snprintf(arrBuf, sizeof(arrBuf),
                "    {\"id\": %d, \"image_id\": %d, \"category_id\": %d, "
                "\"bbox\": [%.2f, %.2f, %.2f, %.2f], \"area\": %.2f, \"iscrowd\": 0}",
                nAnnId, static_cast<int>(imgIdx), bbox.nClassId,
                fPxX, fPxY, fPxW, fPxH, fArea);
            ofs << arrBuf;
            ++nAnnId;
        }
    }
    if (nAnnId > 0) ofs << "\n";
    ofs << "  ]\n";

    ofs << "}\n";
    ofs.close();
}

// ============================================================================
// 20260320 ZJH 保存标注到 OmniMatch 内部 JSON 格式
// ============================================================================
inline void saveAnnotationsOM(const AnnotationProject& project, const std::string& strPath) {
    std::ofstream ofs(strPath);
    if (!ofs.is_open()) return;

    ofs << "{\n";
    ofs << "  \"name\": \"" << project.strName << "\",\n";
    ofs << "  \"type\": " << static_cast<int>(project.annotType) << ",\n";

    // 20260320 ZJH 类别名称
    ofs << "  \"classes\": [";
    for (size_t i = 0; i < project.vecClassNames.size(); ++i) {
        ofs << "\"" << project.vecClassNames[i] << "\"";
        if (i + 1 < project.vecClassNames.size()) ofs << ", ";
    }
    ofs << "],\n";

    // 20260320 ZJH 标注数据
    ofs << "  \"annotations\": [\n";
    for (size_t i = 0; i < project.vecAnnotations.size(); ++i) {
        const auto& ann = project.vecAnnotations[i];
        ofs << "    {\n";
        ofs << "      \"image\": \"" << ann.strImagePath << "\",\n";
        ofs << "      \"width\": " << ann.nImageWidth << ",\n";
        ofs << "      \"height\": " << ann.nImageHeight << ",\n";
        ofs << "      \"image_class\": " << ann.nImageClassId << ",\n";

        // 20260320 ZJH BBox 列表
        ofs << "      \"bboxes\": [";
        for (size_t j = 0; j < ann.vecBBoxes.size(); ++j) {
            const auto& b = ann.vecBBoxes[j];
            char arrBuf[256];
            std::snprintf(arrBuf, sizeof(arrBuf),
                "{\"class_id\": %d, \"x\": %.6f, \"y\": %.6f, \"w\": %.6f, \"h\": %.6f}",
                b.nClassId, b.fX, b.fY, b.fWidth, b.fHeight);
            ofs << arrBuf;
            if (j + 1 < ann.vecBBoxes.size()) ofs << ", ";
        }
        ofs << "],\n";

        // 20260320 ZJH 多边形列表
        ofs << "      \"polygons\": [";
        for (size_t j = 0; j < ann.vecPolygons.size(); ++j) {
            const auto& p = ann.vecPolygons[j];
            ofs << "{\"class_id\": " << p.nClassId << ", \"points\": [";
            for (size_t k = 0; k < p.vecPoints.size(); ++k) {
                char arrBuf[64];
                std::snprintf(arrBuf, sizeof(arrBuf), "[%.6f, %.6f]",
                    p.vecPoints[k].fX, p.vecPoints[k].fY);
                ofs << arrBuf;
                if (k + 1 < p.vecPoints.size()) ofs << ", ";
            }
            ofs << "]}";
            if (j + 1 < ann.vecPolygons.size()) ofs << ", ";
        }
        ofs << "]\n";

        ofs << "    }";
        if (i + 1 < project.vecAnnotations.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();
}

// ============================================================================
// 20260320 ZJH 加载 OmniMatch 内部 JSON 格式标注
// 简化 JSON 解析（手动解析，不依赖 nlohmann_json 避免模块依赖）
// ============================================================================
inline AnnotationProject loadAnnotationsOM(const std::string& strPath) {
    AnnotationProject project;

    std::ifstream ifs(strPath);
    if (!ifs.is_open()) return project;

    // 20260320 ZJH 读取整个文件内容
    std::string strContent((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    ifs.close();

    // 20260320 ZJH 简化解析：提取项目名称
    auto findStr = [&](const std::string& key) -> std::string {
        std::string strSearch = "\"" + key + "\": \"";
        auto pos = strContent.find(strSearch);
        if (pos == std::string::npos) return "";
        pos += strSearch.size();
        auto end = strContent.find("\"", pos);
        if (end == std::string::npos) return "";
        return strContent.substr(pos, end - pos);
    };

    project.strName = findStr("name");

    // 20260320 ZJH 提取类型
    {
        std::string strSearch = "\"type\": ";
        auto pos = strContent.find(strSearch);
        if (pos != std::string::npos) {
            pos += strSearch.size();
            project.annotType = static_cast<AnnotationType>(std::stoi(strContent.substr(pos, 1)));
        }
    }

    // 20260320 ZJH 注意：完整的 JSON 解析需要更复杂的实现
    // 此处提供基础的加载框架，实际使用时应配合 nlohmann_json
    return project;
}

}  // namespace om
