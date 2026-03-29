// 20260322 ZJH ProjectSerializer — 项目序列化工具
// 将 Project 对象序列化为 JSON 文件（.dfproj）
// 以及从 JSON 文件反序列化恢复 Project 对象
// 采用纯静态方法设计，无需实例化
#pragma once

#include <QString>
#include <memory>

// 20260322 ZJH 前向声明 Project 类，避免头文件依赖
class Project;

// 20260322 ZJH 项目序列化器（纯静态工具类）
// 职责：Project ←→ JSON 文件（.dfproj）双向转换
class ProjectSerializer
{
public:
    // 20260322 ZJH 将项目序列化保存为 JSON 文件
    // 参数: pProject    - 要保存的项目指针（只读）
    //       strFilePath - 目标 .dfproj 文件完整路径
    // 返回: true 保存成功，false 保存失败
    static bool save(const Project* pProject, const QString& strFilePath);

    // 20260322 ZJH 从 JSON 文件反序列化恢复项目
    // 参数: strFilePath - .dfproj 文件完整路径
    // 返回: 反序列化的 Project 对象（unique_ptr 所有权），失败返回 nullptr
    static std::unique_ptr<Project> load(const QString& strFilePath);
};
