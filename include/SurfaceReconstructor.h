#pragma once
/**
 * @file SurfaceReconstructor.h
 * @brief SPH-to-Mesh surface reconstruction using Marching Cubes
 *
 * Converts the SPH particle cloud into a solid triangulated mesh
 * for high-fidelity visualization of the machined workpiece.
 *
 * Pipeline:
 *   1. Build 3D scalar density field from particle positions
 *   2. Run Marching Cubes to extract iso-surface triangles
 *   3. Apply Laplacian smoothing for visual quality
 *   4. Interpolate per-vertex scalar data (temperature, damage, stress)
 */

#include "Types.h"
#include <vector>
#include <unordered_map>

namespace edgepredict {

/**
 * @brief Surface reconstruction parameters
 */
struct ReconstructionParams {
    double cellSize = 0.0003;        // Voxel grid cell size (m) — ~300μm default
    double isoValue = 0.1;           // Iso-surface threshold — lowered for sparse drilling chips
    double smoothingRadius = 0.001;  // Kernel radius for density splatting (m)
    int smoothingPasses = 2;         // Laplacian mesh smoothing passes
    bool interpolateScalars = true;  // Interpolate temperature/damage to vertices
    const Mesh* toolMesh = nullptr;  // Tool mesh for CSG carving (nullptr = skip carving)
    bool dynamicReconstruction = false; // If true, carve tool geometry out of density field
};

/**
 * @brief Marching Cubes surface reconstructor for SPH particle clouds
 */
class SurfaceReconstructor {
public:
    /**
     * @brief Reconstruct a solid triangulated mesh from SPH particles
     * @param particles Active SPH particles (INACTIVE particles are ignored)
     * @param params Reconstruction parameters
     * @return Triangulated surface mesh with per-vertex scalars
     */
    static Mesh reconstruct(const std::vector<MPMParticle>& particles,
                            const ReconstructionParams& params);

    /**
     * @brief Reconstruct with CSG tool carving for sharp cutting edges
     * @param particles Active SPH particles
     * @param toolMesh High-resolution tool surface mesh for carving
     * @param params Reconstruction parameters
     * @return Carved triangulated mesh with per-vertex scalars and material status
     */
    static Mesh reconstructWithCarving(const std::vector<MPMParticle>& particles,
                                       const Mesh& toolMesh,
                                       const ReconstructionParams& params);

private:
    /**
     * @brief Build a 3D scalar density field by splatting particles
     */
    static std::vector<double> buildDensityField(
        const std::vector<MPMParticle>& particles,
        const Vec3& gridMin, int nx, int ny, int nz,
        double cellSize, double radius);

    /**
     * @brief Carve tool geometry out of density field (in-place)
     * Sets density to 0 for voxels inside or near the tool surface
     */
    static void carveToolGeometry(std::vector<double>& field,
                                  const Vec3& gridMin, int nx, int ny, int nz,
                                  double cellSize, const Mesh& toolMesh);

    /**
     * @brief Extract iso-surface triangles using Marching Cubes
     */
    static Mesh marchingCubes(const std::vector<double>& field,
                              const Vec3& gridMin,
                              int nx, int ny, int nz,
                              double cellSize, double isoValue);

    /**
     * @brief Apply Laplacian smoothing to reduce staircase artifacts
     */
    static void smoothMesh(Mesh& mesh, int passes);

    /**
     * @brief Interpolate scalar fields from nearby particles to mesh vertices
     */
    static void interpolateScalars(Mesh& mesh,
                                   const std::vector<MPMParticle>& particles,
                                   double radius);

    /**
     * @brief Interpolate particle status onto mesh vertices as materialStatus scalar
     * 0 = workpiece block (ACTIVE), 1 = chip (CHIP), 2 = air (INACTIVE/other)
     */
    static void interpolateMaterialStatus(Mesh& mesh,
                                          const std::vector<MPMParticle>& particles,
                                          double radius);

    /**
     * @brief Standard Marching Cubes edge table (256 entries)
     */
    static const int edgeTable[256];

    /**
     * @brief Standard Marching Cubes triangle table (256 × 16 entries)
     */
    static const int triTable[256][16];
};

} // namespace edgepredict
