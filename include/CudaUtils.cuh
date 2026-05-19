#pragma once
/**
 * @file CudaUtils.cuh
 * @brief CUDA error checking and utility macros for EdgePredict Engine
 */

#include <cuda_runtime.h>
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
#include <iostream>
#include <cstdlib>

namespace edgepredict {

// ============================================================================
// Profiling Macros (NVTX)
// ============================================================================

#ifdef USE_NVTX
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#else
    #define NVTX_PUSH(name)
    #define NVTX_POP()
#endif

// ============================================================================
// Error Checking Macros
// ============================================================================

/**
 * @brief Check CUDA API call for errors and crash with detailed message
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                      << "\n  File: " << __FILE__ \
                      << "\n  Line: " << __LINE__ \
                      << "\n  Call: " << #call << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Check for kernel launch errors
 */
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "[KERNEL ERROR] " << cudaGetErrorString(err) \
                      << "\n  File: " << __FILE__ \
                      << "\n  Line: " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Synchronize and check for errors (use sparingly - expensive)
 */
#define CUDA_SYNC_CHECK() \
    do { \
        cudaDeviceSynchronize(); \
        CUDA_CHECK_KERNEL(); \
    } while (0)

// ============================================================================
// Memory Management Helpers
// ============================================================================

/**
 * @brief RAII wrapper for device memory
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : m_ptr(nullptr), m_size(0) {}
    
    explicit DeviceBuffer(size_t count) : m_ptr(nullptr), m_size(0) {
        allocate(count);
    }
    
    ~DeviceBuffer() { free(); }
    
    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Move allowed
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : m_ptr(other.m_ptr), m_size(other.m_size) {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }
    
    void allocate(size_t count) {
        free();
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&m_ptr, count * sizeof(T)));
            m_size = count;
        }
    }
    
    void free() {
        if (m_ptr) {
            cudaFree(m_ptr);
            m_ptr = nullptr;
            m_size = 0;
        }
    }
    
    void copyFromHost(const T* hostData, size_t count) {
        if (count > m_size) allocate(count);
        CUDA_CHECK(cudaMemcpy(m_ptr, hostData, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copyToHost(T* hostData, size_t count) const {
        CUDA_CHECK(cudaMemcpy(hostData, m_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void copyFromHostAsync(const T* hostData, size_t count, cudaStream_t stream) {
        if (count > m_size) allocate(count);
        CUDA_CHECK(cudaMemcpyAsync(m_ptr, hostData, count * sizeof(T), 
                                   cudaMemcpyHostToDevice, stream));
    }
    
    void copyToHostAsync(T* hostData, size_t count, cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(hostData, m_ptr, count * sizeof(T), 
                                   cudaMemcpyDeviceToHost, stream));
    }
    
    void fill(int value) {
        if (m_ptr && m_size > 0) {
            CUDA_CHECK(cudaMemset(m_ptr, value, m_size * sizeof(T)));
        }
    }
    
    T* get() { return m_ptr; }
    const T* get() const { return m_ptr; }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }
    
    T* operator->() { return m_ptr; }
    const T* operator->() const { return m_ptr; }

private:
    T* m_ptr;
    size_t m_size;
};

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

/**
 * @brief Calculate optimal grid/block dimensions
 */
inline void getKernelDims(int n, int& gridSize, int& blockSize) {
    blockSize = 256;  // Default block size
    gridSize = (n + blockSize - 1) / blockSize;
}

/**
 * @brief Get device properties
 */
inline cudaDeviceProp getDeviceProps(int device = 0) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props;
}

/**
 * @brief Print GPU info
 */
inline void printGPUInfo(int device = 0) {
    cudaDeviceProp props = getDeviceProps(device);
    std::cout << "[GPU] " << props.name 
              << " | SM " << props.major << "." << props.minor
              << " | " << (props.totalGlobalMem / (1024*1024)) << " MB"
              << " | " << props.multiProcessorCount << " SMs"
              << std::endl;
}

} // namespace edgepredict
