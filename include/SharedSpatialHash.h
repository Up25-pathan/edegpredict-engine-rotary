#pragma once

#include "Types.h"
#include "CudaUtils.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace edgepredict {

struct HashConfig {
    double cellSize = 0.0002;
    double originX = 0.0;
    double originY = 0.0;
    double originZ = 0.0;
};

class SharedSpatialHash {
public:
    SharedSpatialHash() = default;
    ~SharedSpatialHash();

    SharedSpatialHash(const SharedSpatialHash&) = delete;
    SharedSpatialHash& operator=(const SharedSpatialHash&) = delete;

    SharedSpatialHash(SharedSpatialHash&& other) noexcept;
    SharedSpatialHash& operator=(SharedSpatialHash&& other) noexcept;

    void build(FEMNodeGPU* d_nodes, int numNodes, const HashConfig& config,
               cudaStream_t stream = 0);

    void scatterResults(FEMNodeGPU* d_dst, cudaStream_t stream = 0);

    FEMNodeGPU* getSortedNodes() { return m_sortedNodes.get(); }
    int* getCellStart() { return m_cellStart.get(); }
    int* getCellEnd() { return m_cellEnd.get(); }
    int getHashTableSize() const { return m_hashTableSize; }
    double getCellSize() const { return m_config.cellSize; }
    double getOriginX() const { return m_config.originX; }
    double getOriginY() const { return m_config.originY; }
    double getOriginZ() const { return m_config.originZ; }
    int getNumNodes() const { return m_numNodes; }

    void free();

private:
    void ensureCapacity(int numNodes);
    static int nextPrime(int n);

    DeviceBuffer<FEMNodeGPU> m_sortedNodes;
    DeviceBuffer<int> m_cellStart;
    DeviceBuffer<int> m_cellEnd;
    DeviceBuffer<int> m_nodeHashes;
    int m_numNodes = 0;
    int m_hashTableSize = 0;
    HashConfig m_config;
};

// ── GPU Kernel Declarations ──────────────────────────────────────────────

__global__ void computeNodeHashKernel(FEMNodeGPU* nodes, int numNodes,
    double originX, double originY, double originZ, double cellSize, int tableSize);

__global__ void scatterContactResultsKernel(
    const FEMNodeGPU* sorted, FEMNodeGPU* original, int numNodes);

struct CompareNodeByHash {
    __host__ __device__ bool operator()(const FEMNodeGPU& a, const FEMNodeGPU& b) const {
        return a.cellHash < b.cellHash;
    }
};

} // namespace edgepredict
