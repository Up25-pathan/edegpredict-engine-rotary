#pragma once

#include "Types.h"
#include <vector>
#include <cuda_runtime.h>

namespace edgepredict {

struct TriWPos;  // forward declaration (defined in ToolSDF.cu)

/**
 * @brief 3D Signed Distance Field for tool mesh collision detection.
 *
 * Builds an SDF from the tool's surface triangles at rest pose (θ = 0).
 * At runtime each particle position is transformed into tool-local SDF
 * space via setToolTransform() and sampled in O(1) via a 3D CUDA texture.
 *
 * The SDF replaces the old node-sphere contact detection (which suffered
 * from tunneling between nodes) with a continuous surface representation.
 */
class ToolSDF {
public:
    ToolSDF() = default;
    ~ToolSDF();

    ToolSDF(const ToolSDF&) = delete;
    ToolSDF& operator=(const ToolSDF&) = delete;

    ToolSDF(ToolSDF&& other) noexcept;
    ToolSDF& operator=(ToolSDF&& other) noexcept;

    /**
     * @brief Build the SDF from a tool surface mesh at rest pose.
     * @param nodes  Surface vertex positions (FEMNode, world coords)
     * @param triangles  Surface triangle connectivity
     * @param paddingMeters  Extra voxel padding around the bounding box
     * @param voxelsPerAxis  Target voxel count along longest axis
     * @return true if SDF was built successfully
     */
    bool build(const std::vector<FEMNode>& nodes,
               const std::vector<Triangle>& triangles,
               double paddingMeters = 0.002,
               int voxelsPerAxis = 128);

    /**
     * @brief Rebuild the SDF from updated vertex positions (e.g., worn geometry).
     * Frees the existing SDF and reconstructs it from the new positions
     * using the same triangle connectivity. The grid dimensions are
     * recomputed from the new bounding box.
     */
    bool rebuild(const std::vector<Vec3>& positions,
                 const std::vector<Triangle>& triangles,
                 double paddingMeters = 0.002,
                 int voxelsPerAxis = 128);

    /// Free all GPU resources
    void free();

    /// Set the current tool world → SDF-rest-frame transform
    void setToolTransform(float centerX, float centerY, float centerZ,
                          float cosAngle, float sinAngle);

    bool isValid() const { return m_texture != 0; }

    // ── Accessors for device kernels ──────────────────────────────────────
    cudaTextureObject_t getTexture()  const { return m_texture; }
    float3 getGridOrigin()            const { return m_origin; }
    float  getVoxelSize()             const { return m_voxelSize; }
    int3   getGridDims()              const { return m_dims; }

    // Tool transform (device-accessible via constant memory or kernel args)
    float getCenterX() const { return m_centerX; }
    float getCenterY() const { return m_centerY; }
    float getCenterZ() const { return m_centerZ; }
    float getCosAngle() const { return m_cosAngle; }
    float getSinAngle() const { return m_sinAngle; }

private:
    void buildFromVerts(const std::vector<float3>& verts,
                        const std::vector<Triangle>& triangles);
    void computeSDFCPU(const std::vector<TriWPos>& triData,
                       std::vector<float>& grid);
    void createTexture(const std::vector<float>& grid);

    cudaTextureObject_t m_texture = 0;
    cudaArray_t         m_sdfArray = nullptr;

    float3 m_origin = {0, 0, 0};
    float  m_voxelSize = 0.001f;
    int3   m_dims = {0, 0, 0};

    // Current world→SDF transform (set per step)
    float m_centerX = 0, m_centerY = 0, m_centerZ = 0;
    float m_cosAngle = 1.0f, m_sinAngle = 0.0f;
};

} // namespace edgepredict
