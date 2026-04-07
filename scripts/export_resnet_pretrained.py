"""
20260331 ZJH 导出 PyTorch ResNet18 预训练权重为 OmniMatch .omm v4 格式
用途: DeepLabV3+ 编码器的预训练初始化，实现迁移学习
格式: [Magic "OMM\0"][Version=4][MetaCount uint32][Meta float×6]
      [NumParams uint32][Params...][NumBuffers uint32][Buffers...][CRC32]
每个张量: [NameLen uint32][Name bytes][NumDims uint32][Dim0..DimN uint32][float32 data]
"""
import torch
import torchvision
import struct
import os
import numpy as np


# CRC32 查表法（与 C++ ModelSerializer 完全一致）
def _build_crc32_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xEDB88320) if (crc & 1) else (crc >> 1)
        table.append(crc & 0xFFFFFFFF)
    return table

_CRC32_TABLE = _build_crc32_table()


def _update_crc(crc: int, data: bytes) -> int:
    """逐字节更新 CRC32（与 C++ updateCrc 完全一致）"""
    for b in data:
        crc = (_CRC32_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)) & 0xFFFFFFFF
    return crc


def _write_u32(f, crc: int, val: int) -> int:
    """写入 uint32 并更新 CRC"""
    data = struct.pack('<I', val)
    f.write(data)
    return _update_crc(crc, data)


def _write_bytes(f, crc: int, data: bytes) -> int:
    """写入原始字节并更新 CRC"""
    f.write(data)
    return _update_crc(crc, data)


def _write_tensor(f, crc: int, name: str, tensor_np: np.ndarray) -> int:
    """写入单个张量（名称 + 维度 + 数据），返回更新后的 CRC"""
    # 写入名称长度 + 名称
    name_bytes = name.encode('utf-8')
    crc = _write_u32(f, crc, len(name_bytes))
    crc = _write_bytes(f, crc, name_bytes)

    # 写入维度数 + 每个维度值
    shape = tensor_np.shape
    crc = _write_u32(f, crc, len(shape))
    for dim in shape:
        crc = _write_u32(f, crc, dim)

    # 写入 float32 数据
    flat = tensor_np.flatten().astype(np.float32)
    data = flat.tobytes()
    crc = _write_bytes(f, crc, data)

    return crc


def export_resnet18_to_omm(output_path: str):
    """下载 ResNet18 ImageNet 预训练权重并导出为 .omm v4 格式"""
    print("[1/4] 下载 ResNet18 预训练权重...")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    # 打印模型结构供对照
    print("\n[INFO] PyTorch ResNet18 参数名 → OmniMatch 名称映射:")
    params = []
    buffers = []

    for name, param in model.named_parameters():
        params.append((name, param.detach().cpu().numpy()))
        print(f"  param: {name}: {list(param.shape)}")

    for name, buf in model.named_buffers():
        if 'running_mean' in name or 'running_var' in name:
            buffers.append((name, buf.detach().cpu().numpy()))
            print(f"  buffer: {name}: {list(buf.shape)}")

    print(f"\n[2/4] 收集完成: {len(params)} 个参数, {len(buffers)} 个缓冲区")

    # ========== 构建 v4 元数据 ==========
    # ModelMeta::encode() 格式: [magic=42.0, typeHash, baseChannels, inputSize, numClasses, inChannels]
    # typeHash = FNV-1a("ResNet18")
    def fnv1a_hash(s: str) -> int:
        h = 2166136261
        for c in s.encode('utf-8'):
            h ^= c
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    model_type = "ResNet18"
    type_hash = fnv1a_hash(model_type)
    base_channels = 64
    input_size = 224
    num_classes = 1000
    in_channels = 3

    meta_floats = [
        42.0,                                               # magic
        struct.unpack('<f', struct.pack('<I', type_hash))[0],  # typeHash as float bits
        float(base_channels),
        float(input_size),
        float(num_classes),
        float(in_channels),
    ]

    # ========== 写入 .omm v4 ==========
    print(f"[3/4] 写入 v4 格式 {output_path}...")

    with open(output_path, 'wb') as f:
        crc = 0

        # 魔数 "OMM\0"
        magic = b'OMM\x00'
        f.write(magic)
        crc = _update_crc(crc, magic)

        # 版本号 4
        crc = _write_u32(f, crc, 4)

        # 元数据: MetaCount + Meta floats
        crc = _write_u32(f, crc, len(meta_floats))
        meta_data = struct.pack(f'<{len(meta_floats)}f', *meta_floats)
        crc = _write_bytes(f, crc, meta_data)

        # 参数数量
        crc = _write_u32(f, crc, len(params))

        # 写入所有参数
        for name, data in params:
            crc = _write_tensor(f, crc, name, data)

        # 缓冲区数量
        crc = _write_u32(f, crc, len(buffers))

        # 写入所有缓冲区
        for name, data in buffers:
            crc = _write_tensor(f, crc, name, data)

        # CRC32 校验和
        f.write(struct.pack('<I', crc))

    # ========== 验证 ==========
    file_size = os.path.getsize(output_path)
    total_params = sum(d.size for _, d in params)
    print(f"\n[4/4] 验证写入...")
    print(f"  文件: {output_path}")
    print(f"  大小: {file_size / 1024 / 1024:.1f} MB")
    print(f"  参数量: {total_params:,}")
    print(f"  参数数: {len(params)}")
    print(f"  缓冲区数: {len(buffers)}")

    # 读回验证 CRC
    with open(output_path, 'rb') as f:
        all_data = f.read()
    verify_crc = 0
    # CRC 覆盖 magic 到 CRC 之前的所有字节
    for b in all_data[:-4]:
        verify_crc = (_CRC32_TABLE[(verify_crc ^ b) & 0xFF] ^ (verify_crc >> 8)) & 0xFFFFFFFF
    saved_crc = struct.unpack('<I', all_data[-4:])[0]
    if verify_crc == saved_crc:
        print(f"  CRC32: OK (0x{saved_crc:08X})")
    else:
        print(f"  CRC32: FAIL (computed=0x{verify_crc:08X}, saved=0x{saved_crc:08X})")

    print("\n[完成] .omm v4 格式导出成功!")


if __name__ == '__main__':
    output = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'resnet18_imagenet.omm')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    export_resnet18_to_omm(output)
