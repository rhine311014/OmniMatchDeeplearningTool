// 20260319 ZJH TensorStorage — 张量原始内存持有者
// 通过 shared_ptr 引用计数管理生命周期
// 视图操作（reshape/slice/transpose）共享同一 Storage 实例
// 20260325 ZJH Phase 1 GPU-Resident 重写：支持 CUDA 设备内存分配/释放/拷贝
// 20260406 ZJH 全局模块片段（module; 声明之前的预处理指令区域）
module;

#include <cstddef>     // 20260406 ZJH size_t — 无符号整数类型
#include <cstdlib>     // 20260406 ZJH std::aligned_alloc / std::free — 对齐内存分配/释放（非 MSVC）
#include <cstring>     // 20260406 ZJH std::memcpy — 内存拷贝
#include <memory>      // 20260406 ZJH std::shared_ptr / std::make_shared — 引用计数智能指针
#include <stdexcept>   // 20260406 ZJH std::runtime_error / std::bad_alloc — 异常类

#include "om_types.h"  // 20260406 ZJH DeviceType 枚举定义（CPU / CUDA）

// 20260325 ZJH CUDA 内存管理函数前向声明（C 链接）
// 仅在编译时启用 CUDA 支持时生效，用于 GPU 内存分配/释放/传输
#ifdef OM_HAS_CUDA
extern "C" {
    // 20260325 ZJH GPU 内存分配（通过内存池）
    int omCudaMalloc(void** ppDevPtr, size_t nBytes);
    // 20260325 ZJH GPU 内存释放（归还到内存池）
    int omCudaFree(void* pDevPtr);
    // 20260325 ZJH Host -> Device 拷贝（同步）
    int omCudaCopyH2D(void* pDst, const void* pSrc, size_t nBytes);
    // 20260325 ZJH Device -> Host 拷贝（同步）
    int omCudaCopyD2H(void* pDst, const void* pSrc, size_t nBytes);
    // 20260325 ZJH Device -> Device 拷贝
    int omCudaCopyD2D(void* pDst, const void* pSrc, size_t nBytes);
}
#endif

// 20260406 ZJH 导出 TensorStorage 模块接口单元
export module om.engine.tensor_storage;

// 20260406 ZJH om 命名空间：OmniMatch 引擎全部公开类型和函数的顶层命名空间
export namespace om {

// 20260319 ZJH TensorStorage — 持有连续内存块，支持 CPU 分配
// 20260325 ZJH 扩展为 CPU/CUDA 双后端，根据 DeviceType 分配不同设备上的内存
class TensorStorage {
public:
    // 20260319 ZJH 构造 — 分配 nBytes 字节的 64 字节对齐内存
    // 20260325 ZJH 扩展为 GPU 感知构造函数，新增 eDevice 和 nDeviceId 参数
    //              默认值 CPU/0 保证所有现有代码无需修改即可编译
    explicit TensorStorage(size_t nBytes,
                           DeviceType eDevice = DeviceType::CPU,
                           int nDeviceId = 0)
        : m_nBytes(nBytes), m_pData(nullptr), m_deviceType(eDevice), m_nDeviceId(nDeviceId)
    {
        if (nBytes > 0) {
            // 20260325 ZJH 根据设备类型选择内存分配策略
            if (eDevice == DeviceType::CUDA) {
#ifdef OM_HAS_CUDA
                // 20260325 ZJH GPU 内存：通过 CUDA 内存池分配设备内存
                int nErr = omCudaMalloc(&m_pData, nBytes);
                if (nErr != 0 || !m_pData) {
                    throw std::runtime_error("TensorStorage: omCudaMalloc failed, bytes=" + std::to_string(nBytes));
                }
#else
                // 20260325 ZJH 未编译 CUDA 支持时，请求 GPU 内存直接抛异常
                throw std::runtime_error("TensorStorage: CUDA not available (OM_HAS_CUDA not defined)");
#endif
            } else {
                // 20260319 ZJH CPU 内存：64 字节对齐分配，适配 AVX-512 等 SIMD 指令
#ifdef _MSC_VER
                m_pData = _aligned_malloc(nBytes, 64);
#else
                m_pData = std::aligned_alloc(64, (nBytes + 63) & ~63);
#endif
                if (!m_pData) {
                    throw std::bad_alloc();
                }
            }
        }
    }

    // 20260319 ZJH 从外部数据构造（拷贝模式）
    // 20260325 ZJH 扩展为 GPU 感知：支持将 CPU 源数据拷贝到指定设备
    //              若目标设备为 CUDA，先在 GPU 分配内存，再通过 H2D 传输
    TensorStorage(const void* pSrcData, size_t nBytes,
                  DeviceType eDevice = DeviceType::CPU,
                  int nDeviceId = 0)
        : TensorStorage(nBytes, eDevice, nDeviceId)  // 20260325 ZJH 委托构造函数分配内存
    {
        if (pSrcData && nBytes > 0) {
            if (eDevice == DeviceType::CUDA) {
#ifdef OM_HAS_CUDA
                // 20260325 ZJH 源数据在 CPU 上，目标在 GPU 上，执行 H2D 传输
                int nErr = omCudaCopyH2D(m_pData, pSrcData, nBytes);
                if (nErr != 0) {
                    throw std::runtime_error("TensorStorage: omCudaCopyH2D failed during construction");
                }
#endif
            } else {
                // 20260319 ZJH CPU 路径：直接 memcpy 拷贝
                std::memcpy(m_pData, pSrcData, nBytes);
            }
        }
    }

    // 20260319 ZJH 禁止拷贝
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;

    // 20260319 ZJH 析构 — 释放对齐内存
    // 20260325 ZJH 根据设备类型选择释放策略：CUDA 使用 omCudaFree，CPU 使用 _aligned_free
    ~TensorStorage() {
        if (m_pData) {
            if (m_deviceType == DeviceType::CUDA) {
#ifdef OM_HAS_CUDA
                // 20260325 ZJH GPU 内存释放（归还到内存池，不实际调用 cudaFree）
                omCudaFree(m_pData);
#endif
            } else {
                // 20260319 ZJH CPU 内存释放
#ifdef _MSC_VER
                _aligned_free(m_pData);
#else
                std::free(m_pData);
#endif
            }
            m_pData = nullptr;
        }
    }

    // 20260406 ZJH 返回底层内存的常量指针（CPU 或 GPU 设备指针），供只读访问
    const void* data() const { return m_pData; }
    // 20260406 ZJH 返回底层内存的可写指针（CPU 或 GPU 设备指针），供写入访问
    void* mutableData() { return m_pData; }
    // 20260406 ZJH 返回已分配的总字节数
    size_t bytes() const { return m_nBytes; }
    // 20260406 ZJH 返回存储所在的设备类型（CPU 或 CUDA）
    DeviceType deviceType() const { return m_deviceType; }
    // 20260406 ZJH 返回设备编号（多 GPU 场景下区分不同 GPU，默认 0）
    int deviceId() const { return m_nDeviceId; }

    // 20260325 ZJH 创建指定设备上的数据副本，处理全部 4 种传输方向
    // eTargetDevice: 目标设备类型（CPU 或 CUDA）
    // nTargetDeviceId: 目标设备编号（多 GPU 场景）
    // 返回: 新分配的 TensorStorage shared_ptr，包含完整数据副本
    std::shared_ptr<TensorStorage> copyTo(DeviceType eTargetDevice,
                                           int nTargetDeviceId = 0) const {
        // 20260325 ZJH 在目标设备上分配同样大小的新存储
        auto pNew = std::make_shared<TensorStorage>(m_nBytes, eTargetDevice, nTargetDeviceId);

        if (m_nBytes == 0) {
            // 20260325 ZJH 空存储无需拷贝
            return pNew;
        }

        if (m_deviceType == DeviceType::CPU && eTargetDevice == DeviceType::CPU) {
            // 20260325 ZJH CPU → CPU：直接内存拷贝
            std::memcpy(pNew->mutableData(), m_pData, m_nBytes);
        }
#ifdef OM_HAS_CUDA
        else if (m_deviceType == DeviceType::CPU && eTargetDevice == DeviceType::CUDA) {
            // 20260325 ZJH CPU → GPU：Host to Device 传输
            int nErr = omCudaCopyH2D(pNew->mutableData(), m_pData, m_nBytes);
            if (nErr != 0) {
                throw std::runtime_error("TensorStorage::copyTo: omCudaCopyH2D failed");
            }
        }
        else if (m_deviceType == DeviceType::CUDA && eTargetDevice == DeviceType::CPU) {
            // 20260325 ZJH GPU → CPU：Device to Host 传输
            int nErr = omCudaCopyD2H(pNew->mutableData(), m_pData, m_nBytes);
            if (nErr != 0) {
                throw std::runtime_error("TensorStorage::copyTo: omCudaCopyD2H failed");
            }
        }
        else if (m_deviceType == DeviceType::CUDA && eTargetDevice == DeviceType::CUDA) {
            // 20260325 ZJH GPU → GPU：Device to Device 传输（可跨设备）
            int nErr = omCudaCopyD2D(pNew->mutableData(), m_pData, m_nBytes);
            if (nErr != 0) {
                throw std::runtime_error("TensorStorage::copyTo: omCudaCopyD2D failed");
            }
        }
#endif
        else {
            // 20260325 ZJH 不支持的传输方向（如未启用 CUDA 时请求 GPU 传输）
            throw std::runtime_error("TensorStorage::copyTo: unsupported device transfer direction");
        }

        return pNew;
    }

private:
    void* m_pData;          // 20260319 ZJH 对齐内存指针（CPU 或 GPU 设备指针）
    size_t m_nBytes;        // 20260319 ZJH 已分配字节数
    DeviceType m_deviceType; // 20260319 ZJH 所在设备类型
    int m_nDeviceId;        // 20260319 ZJH 设备编号（多 GPU 场景）
};

}  // namespace om
