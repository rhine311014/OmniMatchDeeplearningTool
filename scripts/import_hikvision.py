#!/usr/bin/env python3
"""
20260401 ZJH 海康 VisionTrain XML 标注 → OmniMatch .dfproj 导入工具
将海康的 FlawPolygonRoiParameter 多边形标注转换为 OmniMatch Annotation 格式

用法: python import_hikvision.py <海康项目目录> <OmniMatch项目目录>
示例: python import_hikvision.py D:/model1/model1/panel C:/Users/XL/Desktop/tese/新建项目

输入: 海康 base/1/images/ 下的 *.xml 和 *.jpg
输出: 更新 OmniMatch 项目的 images.json，添加多边形标注
"""

import os
import sys
import json
import uuid
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_hikvision_xml(xml_path: str) -> list:
    """解析海康 VisionTrain XML 标注文件，提取多边形标注

    返回: [{"label": "异物", "polygon": [(x1,y1), (x2,y2), ...] }, ...]
    """
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  [WARN] XML 解析失败: {xml_path}")
        return annotations

    # 20260401 ZJH 遍历所有 FlawPolygonRoiParameter 节点
    for item in root.iter():
        if 'FlawPolygonRoiParameter' in item.tag:
            # 20260401 ZJH 提取标签名
            flags_elem = item.find('flags')
            if flags_elem is None or not flags_elem.text:
                continue
            label = flags_elem.text.strip()

            # 20260401 ZJH 跳过 hard_background（海康的忽略区域）
            if label == 'hard_background':
                continue

            # 20260401 ZJH 提取多边形顶点
            points = []
            points_elem = item.find('_PolygonPoints')
            if points_elem is None:
                continue

            for pt in points_elem:
                if 'PolygonPoint' in pt.tag:
                    x_elem = pt.find('x')
                    y_elem = pt.find('y')
                    if x_elem is not None and y_elem is not None:
                        try:
                            x = float(x_elem.text)
                            y = float(y_elem.text)
                            points.append((x, y))
                        except (ValueError, TypeError):
                            continue

            if len(points) >= 3:  # 20260401 ZJH 至少 3 个顶点才是有效多边形
                annotations.append({
                    "label": label,
                    "polygon": points
                })

    return annotations


def import_hikvision_to_omnimatch(hik_dir: str, om_dir: str):
    """将海康项目标注导入 OmniMatch 项目

    1. 复制图像文件到 OmniMatch 项目 images/ 目录
    2. 解析海康 XML，转换标注为 OmniMatch Annotation 格式
    3. 更新 OmniMatch 项目的 .dfproj 文件
    """
    hik_images_dir = os.path.join(hik_dir, "base", "1", "images")
    if not os.path.isdir(hik_images_dir):
        print(f"[ERROR] 海康图像目录不存在: {hik_images_dir}")
        return

    # 20260401 ZJH 查找 OmniMatch 项目文件
    dfproj_files = list(Path(om_dir).glob("*.dfproj"))
    if not dfproj_files:
        print(f"[ERROR] 未找到 .dfproj 文件: {om_dir}")
        return
    dfproj_path = str(dfproj_files[0])
    print(f"[INFO] OmniMatch 项目: {dfproj_path}")

    # 20260401 ZJH 读取现有项目
    with open(dfproj_path, 'r', encoding='utf-8') as f:
        project = json.load(f)

    # 20260401 ZJH 收集海康标签 → OmniMatch labelId 映射
    # 海康标签: 异物(1), 脏污(2), 划痕(3) → OmniMatch labelId: 0, 1, 2
    hik_labels = {}  # label_name → labelId

    # 20260401 ZJH 从已有项目标签中获取映射
    existing_labels = {}
    if "labels" in project:
        for lbl in project["labels"]:
            existing_labels[lbl["name"]] = lbl["id"]

    # 20260401 ZJH 收集所有海康 XML 中的标签
    all_xml_files = sorted(Path(hik_images_dir).glob("*.xml"))
    all_hik_labels = set()
    for xml_file in all_xml_files:
        annos = parse_hikvision_xml(str(xml_file))
        for a in annos:
            all_hik_labels.add(a["label"])

    print(f"[INFO] 海康标签: {all_hik_labels}")
    print(f"[INFO] 现有OmniMatch标签: {existing_labels}")

    # 20260401 ZJH 建立标签映射
    next_id = max(existing_labels.values(), default=-1) + 1
    label_map = {}
    label_colors = {"异物": "#ff4444", "划痕": "#44ff44", "脏污": "#4444ff"}

    for lbl_name in sorted(all_hik_labels):
        if lbl_name in existing_labels:
            label_map[lbl_name] = existing_labels[lbl_name]
        else:
            label_map[lbl_name] = next_id
            color = label_colors.get(lbl_name, "#ffaa00")
            if "labels" not in project:
                project["labels"] = []
            project["labels"].append({
                "id": next_id,
                "name": lbl_name,
                "color": color
            })
            print(f"  [NEW LABEL] {lbl_name} → id={next_id}")
            next_id += 1

    print(f"[INFO] 标签映射: {label_map}")

    # 20260401 ZJH 处理每张图像
    all_jpg_files = sorted(Path(hik_images_dir).glob("*.jpg"))
    print(f"[INFO] 海康图像: {len(all_jpg_files)} 张")

    # 20260401 ZJH 确保 OmniMatch images 目录存在
    om_images_dir = os.path.join(om_dir, "images")
    os.makedirs(om_images_dir, exist_ok=True)

    # 20260401 ZJH 收集现有图像列表（避免重复导入）
    existing_images = {}
    if "images" in project:
        for img in project["images"]:
            fname = os.path.basename(img.get("path", ""))
            existing_images[fname] = img

    imported = 0
    total_annos = 0

    for jpg_file in all_jpg_files:
        fname = jpg_file.name
        xml_file = jpg_file.with_suffix('.xml')

        # 20260401 ZJH 解析标注
        annotations = []
        if xml_file.exists():
            annotations = parse_hikvision_xml(str(xml_file))

        # 20260401 ZJH 复制图像到 OmniMatch 目录
        dst_path = os.path.join(om_images_dir, fname)
        if not os.path.exists(dst_path):
            shutil.copy2(str(jpg_file), dst_path)

        # 20260401 ZJH 转换标注为 OmniMatch 格式
        om_annotations = []
        for anno in annotations:
            label_id = label_map.get(anno["label"], 0)
            poly_points = anno["polygon"]

            # 20260401 ZJH 计算外接矩形
            xs = [p[0] for p in poly_points]
            ys = [p[1] for p in poly_points]
            bounds = {
                "x": min(xs),
                "y": min(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys)
            }

            om_anno = {
                "uuid": str(uuid.uuid4()),
                "type": 1,  # Polygon
                "labelId": label_id,
                "bounds": bounds,
                "polygon": [{"x": p[0], "y": p[1]} for p in poly_points]
            }
            om_annotations.append(om_anno)

        total_annos += len(om_annotations)

        # 20260401 ZJH 更新或添加图像条目
        if fname in existing_images:
            # 更新现有条目的标注
            for img in project["images"]:
                if os.path.basename(img.get("path", "")) == fname:
                    img["annotations"] = om_annotations
                    break
        else:
            # 新增图像条目
            img_entry = {
                "path": f"images/{fname}",
                "labelId": -1,
                "split": 0,  # Unassigned
                "annotations": om_annotations
            }
            if "images" not in project:
                project["images"] = []
            project["images"].append(img_entry)

        imported += 1
        print(f"  [{imported}/{len(all_jpg_files)}] {fname}: {len(om_annotations)} 个标注")

    # 20260401 ZJH 保存更新后的项目文件
    with open(dfproj_path, 'w', encoding='utf-8') as f:
        json.dump(project, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 导入完成:")
    print(f"  图像: {imported} 张")
    print(f"  标注: {total_annos} 个多边形")
    print(f"  标签: {label_map}")
    print(f"  项目已保存: {dfproj_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python import_hikvision.py <海康项目目录> <OmniMatch项目目录>")
        print("示例: python import_hikvision.py D:/model1/model1/panel C:/Users/XL/Desktop/tese/新建项目")
        sys.exit(1)

    import_hikvision_to_omnimatch(sys.argv[1], sys.argv[2])
