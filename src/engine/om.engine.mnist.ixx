// 20260319 ZJH MNIST 数据加载器 — Phase 1D
// 解析 IDX 格式文件（MNIST 标准格式），加载训练集和测试集
// 用户需手动将 MNIST 数据文件放到 data/mnist/ 目录下
// 文件：train-images-idx3-ubyte, train-labels-idx1-ubyte,
//       t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
module;

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

export module om.engine.mnist;

// 20260319 ZJH 导入依赖模块：张量类
import om.engine.tensor;
import om.hal.cpu_backend;

// 20260319 ZJH 内部辅助函数（不导出）
namespace om::detail {

// 20260319 ZJH readUint32BigEndian — 从字节流中读取大端序 32 位无符号整数
// IDX 格式头部使用大端序存储整数字段
inline uint32_t readUint32BigEndian(std::ifstream& ifs) {
    uint8_t arrBytes[4];  // 20260319 ZJH 4 字节缓冲区
    ifs.read(reinterpret_cast<char*>(arrBytes), 4);  // 20260319 ZJH 读取 4 字节
    // 20260319 ZJH 大端序转换：高字节在前，低字节在后
    return (static_cast<uint32_t>(arrBytes[0]) << 24) |
           (static_cast<uint32_t>(arrBytes[1]) << 16) |
           (static_cast<uint32_t>(arrBytes[2]) << 8) |
           (static_cast<uint32_t>(arrBytes[3]));
}

}  // namespace om::detail

export namespace om {

// 20260319 ZJH MnistDataset — MNIST 数据集结构体
// m_images: 图像数据，形状 [N, 784]，像素值归一化到 [0, 1]
// m_labels: 标签数据，形状 [N, 10]，one-hot 编码
// m_nSamples: 样本总数
struct MnistDataset {
    Tensor m_images;    // 20260319 ZJH 图像张量 [N, 784]，float32，归一化到 [0,1]
    Tensor m_labels;    // 20260319 ZJH 标签张量 [N, 10]，float32，one-hot 编码
    int m_nSamples = 0; // 20260319 ZJH 样本总数
};

// 20260319 ZJH loadMnist — 加载 MNIST 数据集（图像 + 标签）
// strImagesPath: IDX3 格式图像文件路径（如 train-images-idx3-ubyte）
// strLabelsPath: IDX1 格式标签文件路径（如 train-labels-idx1-ubyte）
// 返回: MnistDataset 结构体，包含归一化图像和 one-hot 标签
// 若文件不存在或格式错误，抛出 std::runtime_error
MnistDataset loadMnist(const std::string& strImagesPath, const std::string& strLabelsPath) {
    // ===== 读取图像文件 =====
    // 20260319 ZJH 以二进制模式打开图像文件
    std::ifstream ifsImages(strImagesPath, std::ios::binary);
    if (!ifsImages.is_open()) {
        // 20260319 ZJH 文件不存在时输出清晰的指引信息
        throw std::runtime_error(
            "Cannot open MNIST images file: " + strImagesPath + "\n"
            "Please download MNIST dataset and place files at:\n"
            "  data/mnist/train-images-idx3-ubyte\n"
            "  data/mnist/train-labels-idx1-ubyte\n"
            "  data/mnist/t10k-images-idx3-ubyte\n"
            "  data/mnist/t10k-labels-idx1-ubyte\n"
            "Download from: http://yann.lecun.com/exdb/mnist/");
    }

    // 20260319 ZJH 读取图像文件头部：magic number（2051 表示图像）
    uint32_t nMagic = detail::readUint32BigEndian(ifsImages);
    if (nMagic != 2051) {
        throw std::runtime_error("Invalid MNIST images magic number: " + std::to_string(nMagic) + " (expected 2051)");
    }

    // 20260319 ZJH 读取样本数、行数、列数
    uint32_t nNumImages = detail::readUint32BigEndian(ifsImages);  // 20260319 ZJH 图像数量
    uint32_t nRows = detail::readUint32BigEndian(ifsImages);       // 20260319 ZJH 图像行数（28）
    uint32_t nCols = detail::readUint32BigEndian(ifsImages);       // 20260319 ZJH 图像列数（28）
    int nPixels = static_cast<int>(nRows * nCols);         // 20260319 ZJH 每张图像像素数（784）
    int nSamples = static_cast<int>(nNumImages);           // 20260319 ZJH 样本总数

    // 20260319 ZJH 读取所有图像的原始字节数据
    std::vector<uint8_t> vecRawImages(static_cast<size_t>(nSamples) * static_cast<size_t>(nPixels));
    ifsImages.read(reinterpret_cast<char*>(vecRawImages.data()),
                   static_cast<std::streamsize>(vecRawImages.size()));
    ifsImages.close();  // 20260319 ZJH 关闭图像文件

    // 20260319 ZJH 将 uint8 像素值归一化到 [0, 1] float32
    auto images = Tensor::zeros({nSamples, nPixels});  // 20260319 ZJH 分配图像张量
    float* pImageData = images.mutableFloatDataPtr();   // 20260319 ZJH 图像数据写入指针
    for (size_t i = 0; i < vecRawImages.size(); ++i) {
        // 20260319 ZJH 除以 255.0f 将 [0, 255] 映射到 [0.0, 1.0]
        pImageData[i] = static_cast<float>(vecRawImages[i]) / 255.0f;
    }

    // ===== 读取标签文件 =====
    // 20260319 ZJH 以二进制模式打开标签文件
    std::ifstream ifsLabels(strLabelsPath, std::ios::binary);
    if (!ifsLabels.is_open()) {
        throw std::runtime_error(
            "Cannot open MNIST labels file: " + strLabelsPath + "\n"
            "Please download MNIST dataset and place files at:\n"
            "  data/mnist/train-images-idx3-ubyte\n"
            "  data/mnist/train-labels-idx1-ubyte\n"
            "  data/mnist/t10k-images-idx3-ubyte\n"
            "  data/mnist/t10k-labels-idx1-ubyte\n"
            "Download from: http://yann.lecun.com/exdb/mnist/");
    }

    // 20260319 ZJH 读取标签文件头部：magic number（2049 表示标签）
    uint32_t nLabelMagic = detail::readUint32BigEndian(ifsLabels);
    if (nLabelMagic != 2049) {
        throw std::runtime_error("Invalid MNIST labels magic number: " + std::to_string(nLabelMagic) + " (expected 2049)");
    }

    // 20260319 ZJH 读取标签数量并验证与图像数一致
    uint32_t nNumLabels = detail::readUint32BigEndian(ifsLabels);
    if (nNumLabels != nNumImages) {
        throw std::runtime_error("MNIST images/labels count mismatch: " +
            std::to_string(nNumImages) + " images vs " + std::to_string(nNumLabels) + " labels");
    }

    // 20260319 ZJH 读取所有标签的原始字节数据
    std::vector<uint8_t> vecRawLabels(static_cast<size_t>(nSamples));
    ifsLabels.read(reinterpret_cast<char*>(vecRawLabels.data()),
                   static_cast<std::streamsize>(vecRawLabels.size()));
    ifsLabels.close();  // 20260319 ZJH 关闭标签文件

    // 20260319 ZJH 将标签转换为 one-hot 编码 [N, 10]
    int nClasses = 10;  // 20260319 ZJH MNIST 有 10 个类别（数字 0-9）
    auto labels = Tensor::zeros({nSamples, nClasses});  // 20260319 ZJH 分配标签张量
    float* pLabelData = labels.mutableFloatDataPtr();    // 20260319 ZJH 标签数据写入指针
    for (int i = 0; i < nSamples; ++i) {
        int nLabel = static_cast<int>(vecRawLabels[static_cast<size_t>(i)]);  // 20260319 ZJH 当前样本标签值
        // 20260319 ZJH 在对应类别位置设为 1.0，其余保持 0.0（one-hot 编码）
        pLabelData[i * nClasses + nLabel] = 1.0f;
    }

    // 20260319 ZJH 组装并返回数据集结构体
    MnistDataset dataset;
    dataset.m_images = images;      // 20260319 ZJH 归一化图像 [N, 784]
    dataset.m_labels = labels;      // 20260319 ZJH one-hot 标签 [N, 10]
    dataset.m_nSamples = nSamples;  // 20260319 ZJH 样本数量

    // 20260319 ZJH 输出加载信息
    std::printf("Loaded %d samples from %s\n", nSamples, strImagesPath.c_str());

    return dataset;  // 20260319 ZJH 返回加载完成的数据集
}

}  // namespace om
