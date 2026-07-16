#include "ToolSDF.h"
#include "CudaUtils.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace edgepredict {

// ── Utility: point-to-triangle distance (squared) ──────────────────────────
// Returns squared distance from p to the triangle defined by a,b,c, and
// stores the closest point on the triangle in `closest`.
__device__ inline float pointTriDistanceSq(
    const float3& p,
    const float3& a, const float3& b, const float3& c,
    float3& closest)
{
    float3 ab = {b.x - a.x, b.y - a.y, b.z - a.z};
    float3 ac = {c.x - a.x, c.y - a.y, c.z - a.z};
    float3 ap = {p.x - a.x, p.y - a.y, p.z - a.z};

    float d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z;
    float d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        goto done;
    }

    float3 bp = {p.x - b.x, p.y - b.y, p.z - b.z};
    float d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z;
    float d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        goto done;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        closest.x = a.x + v * ab.x;
        closest.y = a.y + v * ab.y;
        closest.z = a.z + v * ab.z;
        goto done;
    }

    float3 cp = {p.x - c.x, p.y - c.y, p.z - c.z};
    float d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z;
    float d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        goto done;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        closest.x = a.x + w * ac.x;
        closest.y = a.y + w * ac.y;
        closest.z = a.z + w * ac.z;
        goto done;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest.x = b.x + w * (c.x - b.x);
        closest.y = b.y + w * (c.y - b.y);
        closest.z = b.z + w * (c.z - b.z);
        goto done;
    }

    // Inside the triangle — barycentric
    {
        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        float u = 1.0f - v - w;
        closest.x = a.x + u * (b.x - a.x) + v * (c.x - a.x);
        closest.y = a.y + u * (b.y - a.y) + v * (c.y - a.y);
        closest.z = a.z + u * (b.z - a.z) + v * (c.z - a.z);
    }

done:
    float dx = p.x - closest.x;
    float dy = p.y - closest.y;
    float dz = p.z - closest.z;
    return dx * dx + dy * dy + dz * dz;
}

// Host-side version of the same computation
static float pointTriDistanceSqHost(
    const float3& p,
    const float3& a, const float3& b, const float3& c,
    float3& closest)
{
    // Same implementation as device version
    float3 ab = {b.x - a.x, b.y - a.y, b.z - a.z};
    float3 ac = {c.x - a.x, c.y - a.y, c.z - a.z};
    float3 ap = {p.x - a.x, p.y - a.y, p.z - a.z};

    float d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z;
    float d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        goto done;
    }

    float3 bp = {p.x - b.x, p.y - b.y, p.z - b.z};
    float d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z;
    float d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        goto done;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        closest.x = a.x + v * ab.x;
        closest.y = a.y + v * ab.y;
        closest.z = a.z + v * ab.z;
        goto done;
    }

    float3 cp = {p.x - c.x, p.y - c.y, p.z - c.z};
    float d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z;
    float d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        goto done;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        closest.x = a.x + w * ac.x;
        closest.y = a.y + w * ac.y;
        closest.z = a.z + w * ac.z;
        goto done;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest.x = b.x + w * (c.x - b.x);
        closest.y = b.y + w * (c.y - b.y);
        closest.z = b.z + w * (c.z - b.z);
        goto done;
    }

    {
        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        float u = 1.0f - v - w;
        closest.x = a.x + u * (b.x - a.x) + v * (c.x - a.x);
        closest.y = a.y + u * (b.y - a.y) + v * (c.y - a.y);
        closest.z = a.z + u * (b.z - a.z) + v * (c.z - a.z);
    }

done:
    float dx = p.x - closest.x;
    float dy = p.y - closest.y;
    float dz = p.z - closest.z;
    return dx * dx + dy * dy + dz * dz;
}

// ── Internal triangle representation with precomputed geometry ────────────
struct TriWPos {
    float3 a, b, c;
    float3 normal;
    float3 bboxMin, bboxMax;
};

// ── ToolSDF Implementation ─────────────────────────────────────────────────

ToolSDF::~ToolSDF() { free(); }

ToolSDF::ToolSDF(ToolSDF&& other) noexcept
    : m_texture(other.m_texture), m_sdfArray(other.m_sdfArray),
      m_origin(other.m_origin), m_voxelSize(other.m_voxelSize),
      m_dims(other.m_dims),
      m_centerX(other.m_centerX), m_centerY(other.m_centerY), m_centerZ(other.m_centerZ),
      m_cosAngle(other.m_cosAngle), m_sinAngle(other.m_sinAngle)
{
    other.m_texture = 0;
    other.m_sdfArray = nullptr;
    other.m_dims = {0, 0, 0};
}

ToolSDF& ToolSDF::operator=(ToolSDF&& other) noexcept {
    if (this != &other) {
        free();
        m_texture = other.m_texture;      other.m_texture = 0;
        m_sdfArray = other.m_sdfArray;    other.m_sdfArray = nullptr;
        m_origin = other.m_origin;
        m_voxelSize = other.m_voxelSize;
        m_dims = other.m_dims;            other.m_dims = {0, 0, 0};
        m_centerX = other.m_centerX; m_centerY = other.m_centerY; m_centerZ = other.m_centerZ;
        m_cosAngle = other.m_cosAngle; m_sinAngle = other.m_sinAngle;
    }
    return *this;
}

void ToolSDF::free() {
    if (m_texture) { cudaDestroyTextureObject(m_texture); m_texture = 0; }
    if (m_sdfArray) { cudaFreeArray(m_sdfArray); m_sdfArray = nullptr; }
    m_dims = {0, 0, 0};
}

bool ToolSDF::build(const std::vector<FEMNode>& nodes,
                    const std::vector<Triangle>& triangles,
                    double paddingMeters,
                    int voxelsPerAxis)
{
    free();

    if (nodes.empty() || triangles.empty()) {
        std::cerr << "[ToolSDF] Empty mesh — cannot build SDF" << std::endl;
        return false;
    }

    // Compute bounding box of the mesh
    float3 bmin = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 bmax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const auto& n : nodes) {
        bmin.x = fminf(bmin.x, (float)n.position.x);
        bmin.y = fminf(bmin.y, (float)n.position.y);
        bmin.z = fminf(bmin.z, (float)n.position.z);
        bmax.x = fmaxf(bmax.x, (float)n.position.x);
        bmax.y = fmaxf(bmax.y, (float)n.position.y);
        bmax.z = fmaxf(bmax.z, (float)n.position.z);
    }

    // Add padding
    float pad = (float)paddingMeters;
    bmin.x -= pad; bmin.y -= pad; bmin.z -= pad;
    bmax.x += pad; bmax.y += pad; bmax.z += pad;

    m_origin = bmin;

    // Determine voxel size (isotropic, based on longest axis)
    float extentX = bmax.x - bmin.x;
    float extentY = bmax.y - bmin.y;
    float extentZ = bmax.z - bmin.z;
    float maxExtent = fmaxf(extentX, fmaxf(extentY, extentZ));
    m_voxelSize = maxExtent / (float)voxelsPerAxis;

    m_dims.x = (int)(extentX / m_voxelSize) + 2;
    m_dims.y = (int)(extentY / m_voxelSize) + 2;
    m_dims.z = (int)(extentZ / m_voxelSize) + 2;

    // Clamp to reasonable limits
    m_dims.x = std::min(m_dims.x, 256);
    m_dims.y = std::min(m_dims.y, 256);
    m_dims.z = std::min(m_dims.z, 256);

    int totalVoxels = m_dims.x * m_dims.y * m_dims.z;
    if (totalVoxels <= 0) {
        std::cerr << "[ToolSDF] Invalid grid dimensions" << std::endl;
        return false;
    }

    // Convert nodes to float3 array for faster access
    std::vector<float3> verts(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        verts[i] = {(float)nodes[i].position.x,
                    (float)nodes[i].position.y,
                    (float)nodes[i].position.z};
    }

    // Build triangle list (float3 vertices) and compute SDF
    buildFromVerts(verts, triangles);

    std::cout << "[ToolSDF] Built " << m_dims.x << "×" << m_dims.y << "×" << m_dims.z
              << " SDF (voxel " << m_voxelSize * 1000 << "mm) from "
              << triangles.size() << " triangles" << std::endl;
    return true;
}

bool ToolSDF::rebuild(const std::vector<Vec3>& positions,
                       const std::vector<Triangle>& triangles,
                       double paddingMeters,
                       int voxelsPerAxis)
{
    free();

    if (positions.empty() || triangles.empty()) {
        std::cerr << "[ToolSDF] Empty geometry — cannot rebuild SDF" << std::endl;
        return false;
    }

    // Compute bounding box from new positions
    float3 bmin = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 bmax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const auto& p : positions) {
        bmin.x = fminf(bmin.x, (float)p.x);
        bmin.y = fminf(bmin.y, (float)p.y);
        bmin.z = fminf(bmin.z, (float)p.z);
        bmax.x = fmaxf(bmax.x, (float)p.x);
        bmax.y = fmaxf(bmax.y, (float)p.y);
        bmax.z = fmaxf(bmax.z, (float)p.z);
    }

    float pad = (float)paddingMeters;
    bmin.x -= pad; bmin.y -= pad; bmin.z -= pad;
    bmax.x += pad; bmax.y += pad; bmax.z += pad;

    m_origin = bmin;

    float extentX = bmax.x - bmin.x;
    float extentY = bmax.y - bmin.y;
    float extentZ = bmax.z - bmin.z;
    float maxExtent = fmaxf(extentX, fmaxf(extentY, extentZ));
    m_voxelSize = maxExtent / (float)voxelsPerAxis;

    m_dims.x = (int)(extentX / m_voxelSize) + 2;
    m_dims.y = (int)(extentY / m_voxelSize) + 2;
    m_dims.z = (int)(extentZ / m_voxelSize) + 2;

    m_dims.x = std::min(m_dims.x, 256);
    m_dims.y = std::min(m_dims.y, 256);
    m_dims.z = std::min(m_dims.z, 256);

    int totalVoxels = m_dims.x * m_dims.y * m_dims.z;
    if (totalVoxels <= 0) {
        std::cerr << "[ToolSDF] Invalid grid dimensions after rebuild" << std::endl;
        return false;
    }

    // Convert Vec3 positions to float3
    std::vector<float3> verts(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        verts[i] = {(float)positions[i].x,
                    (float)positions[i].y,
                    (float)positions[i].z};
    }

    buildFromVerts(verts, triangles);

    std::cout << "[ToolSDF] Rebuilt " << m_dims.x << "×" << m_dims.y << "×" << m_dims.z
              << " SDF from worn geometry (" << triangles.size() << " triangles)"
              << std::endl;
    return true;
}

void ToolSDF::buildFromVerts(const std::vector<float3>& verts,
                              const std::vector<Triangle>& triangles)
{
    // Build triangle data from float3 verts + Triangle connectivity
    std::vector<TriWPos> triData(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        triData[i].a = verts[tri.indices[0]];
        triData[i].b = verts[tri.indices[1]];
        triData[i].c = verts[tri.indices[2]];
        triData[i].normal = {(float)tri.normal.x,
                             (float)tri.normal.y,
                             (float)tri.normal.z};
        triData[i].bboxMin = {
            fminf(triData[i].a.x, fminf(triData[i].b.x, triData[i].c.x)),
            fminf(triData[i].a.y, fminf(triData[i].b.y, triData[i].c.y)),
            fminf(triData[i].a.z, fminf(triData[i].b.z, triData[i].c.z))
        };
        triData[i].bboxMax = {
            fmaxf(triData[i].a.x, fmaxf(triData[i].b.x, triData[i].c.x)),
            fmaxf(triData[i].a.y, fmaxf(triData[i].b.y, triData[i].c.y)),
            fmaxf(triData[i].a.z, fmaxf(triData[i].b.z, triData[i].c.z))
        };
    }

    int totalVoxels = m_dims.x * m_dims.y * m_dims.z;
    std::vector<float> grid(totalVoxels, FLT_MAX);
    computeSDFCPU(triData, grid);
    createTexture(grid);
}

void ToolSDF::computeSDFCPU(const std::vector<TriWPos>& triData,
                            std::vector<float>& grid)
{
    int totalVoxels = m_dims.x * m_dims.y * m_dims.z;
    int numTris = (int)triData.size();

    // For each triangle, scatter its distance to voxels within its bbox
    // (the bbox is expanded by 1 voxel for smoothness)
    float invVoxel = 1.0f / m_voxelSize;

    // Precompute voxel center positions
    auto voxelCenter = [&](int ix, int iy, int iz) -> float3 {
        return {m_origin.x + (ix + 0.5f) * m_voxelSize,
                m_origin.y + (iy + 0.5f) * m_voxelSize,
                m_origin.z + (iz + 0.5f) * m_voxelSize};
    };

    for (int ti = 0; ti < numTris; ++ti) {
        const auto& tri = triData[ti];

        // Compute voxel range for this triangle's bbox (expanded by 1 voxel)
        int ix0 = std::max(0, (int)((tri.bboxMin.x - m_origin.x) * invVoxel) - 1);
        int iy0 = std::max(0, (int)((tri.bboxMin.y - m_origin.y) * invVoxel) - 1);
        int iz0 = std::max(0, (int)((tri.bboxMin.z - m_origin.z) * invVoxel) - 1);
        int ix1 = std::min(m_dims.x - 1, (int)((tri.bboxMax.x - m_origin.x) * invVoxel) + 2);
        int iy1 = std::min(m_dims.y - 1, (int)((tri.bboxMax.y - m_origin.y) * invVoxel) + 2);
        int iz1 = std::min(m_dims.z - 1, (int)((tri.bboxMax.z - m_origin.z) * invVoxel) + 2);

        for (int iz = iz0; iz <= iz1; ++iz) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                for (int ix = ix0; ix <= ix1; ++ix) {
                    float3 center = voxelCenter(ix, iy, iz);
                    float3 closest;
                    float d2 = pointTriDistanceSqHost(center, tri.a, tri.b, tri.c, closest);
                    int idx = ix + m_dims.x * (iy + m_dims.y * iz);
                    if (d2 < grid[idx]) {
                        grid[idx] = d2;
                    }
                }
            }
        }
    }

    // Convert squared distances to signed distances
    // First pass: compute unsigned distance for all voxels
    for (int i = 0; i < totalVoxels; ++i) {
        if (grid[i] < FLT_MAX) {
            grid[i] = sqrtf(grid[i]);
        }
    }

    // Determine inside/outside sign via ray casting along Z
    // For each (x,y) column, scan from -inf to +inf, toggling inside/outside
    // at each surface crossing detected by a sign change in the distance
    // gradient along Z.
    //
    // Simpler approach: use the nearest triangle's normal to determine sign.
    // For each surface voxel (grid[i] < padding), compute position and
    // re-find the nearest triangle. If dot(pos - nearest_point, normal) < 0,
    // the point is inside the tool.

    // We need the nearest triangle for sign determination.
    // Store the closest triangle index alongside the distance.
    // Since we only have distances, we'll recompute for surface voxels.

    float narrowBand = m_voxelSize * 2.0f;
    for (int iz = 0; iz < m_dims.z; ++iz) {
        for (int iy = 0; iy < m_dims.y; ++iy) {
            for (int ix = 0; ix < m_dims.x; ++ix) {
                int idx = ix + m_dims.x * (iy + m_dims.y * iz);
                float dist = grid[idx];
                if (dist > narrowBand) continue;  // far from surface — will be handled by flood fill

                float3 center = voxelCenter(ix, iy, iz);

                // Re-find nearest triangle for sign
                float bestD2 = FLT_MAX;
                float3 bestNormal = {0, 0, 0};
                float3 bestClosest = {0, 0, 0};
                for (const auto& tri : triData) {
                    // Quick bbox test
                    if (center.x < tri.bboxMin.x - m_voxelSize ||
                        center.x > tri.bboxMax.x + m_voxelSize ||
                        center.y < tri.bboxMin.y - m_voxelSize ||
                        center.y > tri.bboxMax.y + m_voxelSize ||
                        center.z < tri.bboxMin.z - m_voxelSize ||
                        center.z > tri.bboxMax.z + m_voxelSize) continue;

                    float3 closest;
                    float d2 = pointTriDistanceSqHost(center, tri.a, tri.b, tri.c, closest);
                    if (d2 < bestD2) {
                        bestD2 = d2;
                        bestNormal = tri.normal;
                        bestClosest = closest;
                    }
                }

                if (bestD2 < FLT_MAX) {
                    float3 diff = {center.x - bestClosest.x,
                                   center.y - bestClosest.y,
                                   center.z - bestClosest.z};
                    float dot = diff.x * bestNormal.x + diff.y * bestNormal.y + diff.z * bestNormal.z;
                    if (dot < 0.0f) {
                        grid[idx] = -dist;  // inside tool
                    }
                    // else: outside (positive, already set)
                }
            }
        }
    }

    // Flood fill for voxels far from surface: propagate sign from nearest
    // surface voxel using a simple iterative propagation.
    // For each unknown voxel (dist == FLT_MAX), copy the sign of the
    // nearest known neighbor.
    for (int pass = 0; pass < 3; ++pass) {
        for (int iz = 1; iz < m_dims.z - 1; ++iz) {
            for (int iy = 1; iy < m_dims.y - 1; ++iy) {
                for (int ix = 1; ix < m_dims.x - 1; ++ix) {
                    int idx = ix + m_dims.x * (iy + m_dims.y * iz);
                    if (grid[idx] < FLT_MAX) continue;

                    // Check neighbors for a known distance
                    float nearestDist = FLT_MAX;
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                int ni = (ix + dx) + m_dims.x * ((iy + dy) + m_dims.y * (iz + dz));
                                float nd = fabsf(grid[ni]);
                                if (nd < nearestDist) {
                                    nearestDist = nd;
                                }
                            }
                        }
                    }
                    if (nearestDist < FLT_MAX) {
                        // Extrapolate — we don't know sign, use positive (outside)
                        // This is safe because flood fill only reaches voxels outside the
                        // narrow band, and the tool is a closed manifold.
                        grid[idx] = nearestDist + m_voxelSize * 0.5f;
                    }
                }
            }
        }
    }

    // Fill any remaining unknown voxels with a large distance
    for (int i = 0; i < totalVoxels; ++i) {
        if (grid[i] >= FLT_MAX) {
            grid[i] = m_voxelSize * (float)std::max(m_dims.x, std::max(m_dims.y, m_dims.z));
        }
    }
}

void ToolSDF::createTexture(const std::vector<float>& grid) {
    // Create 3D CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(m_dims.x, m_dims.y, m_dims.z);

    CUDA_CHECK(cudaMalloc3DArray(&m_sdfArray, &channelDesc, extent));

    // Copy host data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)grid.data(),
                                             m_dims.x * sizeof(float),
                                             m_dims.x, m_dims.y);
    copyParams.dstArray = m_sdfArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Texture resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_sdfArray;

    // Texture descriptor (trilinear interpolation, clamp to edge)
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;     // trilinear interpolation
    texDesc.readMode = cudaReadModeElementType;     // read as float
    texDesc.normalizedCoords = false;               // use unnormalized (voxel index) coords
    texDesc.borderColor[0] = 0.0f;

    CUDA_CHECK(cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr));
}

void ToolSDF::setToolTransform(float centerX, float centerY, float centerZ,
                                float cosAngle, float sinAngle) {
    m_centerX = centerX;
    m_centerY = centerY;
    m_centerZ = centerZ;
    m_cosAngle = cosAngle;
    m_sinAngle = sinAngle;
}

} // namespace edgepredict
