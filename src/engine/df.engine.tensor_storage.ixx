// 20260319 ZJH TensorStorage — 张量原始内存持有者
// 通过 shared_ptr 引用计数管理生命周期
// 视图操作（reshape/slice/transpose）共享同一 Storage 实例
module;

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "df_types.h"

export module df.engine.tensor_storage;

export namespace df {

// 20260319 ZJH TensorStorage — 持有连续内存块，支持 CPU 分配
class TensorStorage {
public:
    // 20260319 ZJH 构造 — 分配 nBytes 字节的 64 字节对齐内存
    explicit TensorStorage(size_t nBytes)
        : m_nBytes(nBytes), m_pData(nullptr), m_deviceType(DeviceType::CPU), m_nDeviceId(0)
    {
        if (nBytes > 0) {
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

    // 20260319 ZJH 从外部数据构造（拷贝模式）
    TensorStorage(const void* pSrcData, size_t nBytes)
        : TensorStorage(nBytes)
    {
        if (pSrcData && nBytes > 0) {
            std::memcpy(m_pData, pSrcData, nBytes);
        }
    }

    // 20260319 ZJH 禁止拷贝
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;

    // 20260319 ZJH 析构 — 释放对齐内存
    ~TensorStorage() {
        if (m_pData) {
#ifdef _MSC_VER
            _aligned_free(m_pData);
#else
            std::free(m_pData);
#endif
            m_pData = nullptr;
        }
    }

    const void* data() const { return m_pData; }
    void* mutableData() { return m_pData; }
    size_t bytes() const { return m_nBytes; }
    DeviceType deviceType() const { return m_deviceType; }
    int deviceId() const { return m_nDeviceId; }

private:
    void* m_pData;          // 20260319 ZJH 对齐内存指针
    size_t m_nBytes;        // 20260319 ZJH 已分配字节数
    DeviceType m_deviceType; // 20260319 ZJH 所在设备类型
    int m_nDeviceId;        // 20260319 ZJH 设备编号（多 GPU 场景）
};

}  // namespace df
