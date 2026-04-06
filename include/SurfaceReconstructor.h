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

namespace edgepredict {

/**
 * @brief Surface reconstruction parameters
 */
struct ReconstructionParams {
    double cellSize = 0.0003;        // Voxel grid cell size (m) — ~300μm default
    double isoValue = 0.5;           // Iso-surface threshold (0-1 normalized)
    double smoothingRadius = 0.001;  // Kernel radius for density splatting (m)
    int smoothingPasses = 2;         // Laplacian mesh smoothing passes
    bool interpolateScalars = true;  // Interpolate temperature/damage to vertices
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
    static Mesh reconstruct(const std::vector<SPHParticle>& particles,
                            const ReconstructionParams& params);

private:
    /**
     * @brief Build a 3D scalar density field by splatting particles
     */
    static std::vector<double> buildDensityField(
        const std::vector<SPHParticle>& particles,
        const Vec3& gridMin, int nx, int ny, int nz,
        double cellSize, double radius);

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
                                   const std::vector<SPHParticle>& particles,
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
