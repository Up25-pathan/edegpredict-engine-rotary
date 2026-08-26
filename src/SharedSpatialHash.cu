#include "SharedSpatialHash.h"
#include "CudaUtils.cuh"
#include <thrust/execution_policy.h>

namespace edgepredict {

// ── GPU Kernels ──────────────────────────────────────────────────────────

__global__ void computeNodeHashKernel(FEMNodeGPU* nodes, int numNodes,
    double originX, double originY, double originZ, double cellSize, int tableSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    int cx = max(0, (int)floor((nodes[i].x - originX) / cellSize));
    int cy = max(0, (int)floor((nodes[i].y - originY) / cellSize));
    int cz = max(0, (int)floor((nodes[i].z - originZ) / cellSize));

    unsigned long long h = (static_cast<unsigned long long>(cx) * 73856093ULL) ^
                           (static_cast<unsigned long long>(cy) * 19349663ULL) ^
                           (static_cast<unsigned long long>(cz) * 83492791ULL);
    nodes[i].cellHash = static_cast<int>((h % static_cast<unsigned long long>(tableSize) + static_cast<unsigned long long>(tableSize)) % static_cast<unsigned long long>(tableSize));

    nodes[i].fx = 0.0;
    nodes[i].fy = 0.0;
    nodes[i].fz = 0.0;
    nodes[i].heatAccumulator = 0.0;
    nodes[i].contactForceNormal = 0.0;
}

__global__ void scatterContactResultsKernel(
    const FEMNodeGPU* sorted, FEMNodeGPU* original, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    int oid = sorted[i].id;
    if (oid < 0 || oid >= numNodes) return;

    atomicAdd(&original[oid].fx, sorted[i].fx);
    atomicAdd(&original[oid].fy, sorted[i].fy);
    atomicAdd(&original[oid].fz, sorted[i].fz);
    atomicAdd(&original[oid].heatAccumulator, sorted[i].heatAccumulator);
    atomicAdd(&original[oid].contactForceNormal, sorted[i].contactForceNormal);

    if (sorted[i].penetrationDepth > original[oid].penetrationDepth) {
        atomicMaxDouble(&original[oid].penetrationDepth, sorted[i].penetrationDepth);
    }
    if (sorted[i].inContact) {
        original[oid].inContact = true;
    }
}

// Forward declaration (defined in ContactSolver.cu)
__global__ void findToolCellBoundsKernel(
    FEMNodeGPU* sortedNodes, int numNodes,
    int* cellStart, int* cellEnd, int hashTableSize);

// ── SharedSpatialHash Implementation ─────────────────────────────────────

SharedSpatialHash::~SharedSpatialHash() {
    free();
}

SharedSpatialHash::SharedSpatialHash(SharedSpatialHash&& other) noexcept
    : m_sortedNodes(std::move(other.m_sortedNodes))
    , m_cellStart(std::move(other.m_cellStart))
    , m_cellEnd(std::move(other.m_cellEnd))
    , m_nodeHashes(std::move(other.m_nodeHashes))
    , m_numNodes(other.m_numNodes)
    , m_hashTableSize(other.m_hashTableSize)
    , m_config(other.m_config)
{
    other.m_numNodes = 0;
    other.m_hashTableSize = 0;
}

SharedSpatialHash& SharedSpatialHash::operator=(SharedSpatialHash&& other) noexcept {
    if (this != &other) {
        free();
        m_sortedNodes = std::move(other.m_sortedNodes);
        m_cellStart = std::move(other.m_cellStart);
        m_cellEnd = std::move(other.m_cellEnd);
        m_nodeHashes = std::move(other.m_nodeHashes);
        m_numNodes = other.m_numNodes;
        m_hashTableSize = other.m_hashTableSize;
        m_config = other.m_config;
        other.m_numNodes = 0;
        other.m_hashTableSize = 0;
    }
    return *this;
}

void SharedSpatialHash::free() {
    m_sortedNodes.free();
    m_cellStart.free();
    m_cellEnd.free();
    m_nodeHashes.free();
    m_numNodes = 0;
    m_hashTableSize = 0;
}

void SharedSpatialHash::ensureCapacity(int numNodes) {
    if (numNodes > static_cast<int>(m_sortedNodes.size())) {
        m_sortedNodes.allocate(numNodes);
        m_nodeHashes.allocate(numNodes);
        int tableSize = nextPrime(std::max(numNodes * 2, 100003));
        if (tableSize != m_hashTableSize) {
            m_cellStart.allocate(tableSize);
            m_cellEnd.allocate(tableSize);
            m_hashTableSize = tableSize;
        }
        m_numNodes = numNodes;
    }
}

int SharedSpatialHash::nextPrime(int n) {
    auto isPrime = [](int x) -> bool {
        if (x < 2) return false;
        if (x == 2) return true;
        if (x % 2 == 0) return false;
        for (int i = 3; i * i <= x; i += 2)
            if (x % i == 0) return false;
        return true;
    };
    while (!isPrime(n)) ++n;
    return n;
}

void SharedSpatialHash::build(FEMNodeGPU* d_nodes, int numNodes,
                               const HashConfig& config, cudaStream_t stream) {
    ensureCapacity(numNodes);
    m_config = config;

    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;

    // Step 1: copy nodes to sorted buffer, compute hashes, zero accumulators
    CUDA_CHECK(cudaMemcpyAsync(m_sortedNodes.get(), d_nodes,
                                numNodes * sizeof(FEMNodeGPU),
                                cudaMemcpyDeviceToDevice, stream));

    computeNodeHashKernel<<<gridSize, blockSize, 0, stream>>>(
        m_sortedNodes.get(), numNodes,
        config.originX, config.originY, config.originZ,
        config.cellSize, m_hashTableSize);
    CUDA_CHECK_KERNEL();

    // Step 2: sort by cellhash
    thrust::device_ptr<FEMNodeGPU> nodePtr(m_sortedNodes.get());
    thrust::sort(thrust::cuda::par.on(stream), nodePtr, nodePtr + numNodes, CompareNodeByHash());

    // Step 3: build cell bounds
    CUDA_CHECK(cudaMemsetAsync(m_cellStart.get(), -1,
                                m_hashTableSize * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(m_cellEnd.get(), -1,
                                m_hashTableSize * sizeof(int), stream));

    // Reuse gridSize for cell bounds kernel (same node count)
    findToolCellBoundsKernel<<<gridSize, blockSize, 0, stream>>>(
        m_sortedNodes.get(), numNodes,
        m_cellStart.get(), m_cellEnd.get(), m_hashTableSize);
    CUDA_CHECK_KERNEL();

    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

void SharedSpatialHash::scatterResults(FEMNodeGPU* d_dst, cudaStream_t stream) {
    if (m_numNodes <= 0) return;

    int blockSize = 256;
    int gridSize = (m_numNodes + blockSize - 1) / blockSize;

    scatterContactResultsKernel<<<gridSize, blockSize, 0, stream>>>(
        m_sortedNodes.get(), d_dst, m_numNodes);
    CUDA_CHECK_KERNEL();
}

} // namespace edgepredict
