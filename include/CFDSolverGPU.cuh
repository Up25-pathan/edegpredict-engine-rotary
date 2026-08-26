#pragma once
/**
 * @file CFDSolverGPU.cuh
 * @brief GPU-accelerated CFD solver using CUDA
 * 
 * Implements Navier-Stokes equations on MAC grid:
 * - Semi-Lagrangian advection (stable)
 * - Jacobi/Red-Black Gauss-Seidel pressure solver
 * - Temperature advection and diffusion
 * - Coupling with SPH/FEM
 * 
 * Performance target: 100x faster than CPU version
 */

#include "Types.h"
#include "Config.h"
#include "CudaUtils.cuh"
#include "IPhysicsSolver.h"
#include "DeviceCoupling.h"

namespace edgepredict {

/**
 * @brief MAC grid cell for CFD
 * @note Temperature fields use double precision for consistent thermal coupling
 *       with FEM/Contact solvers (BUG-H4 fix). Velocity/pressure remain float.
 */
struct CFDCell {
    float u, v, w;          // Velocity components (staggered)
    float p;                // Pressure (cell center)
    double T;               // Temperature (double for thermal coupling consistency)
    float divergence;
    bool isSolid;           // Boundary flag
};

/**
 * @brief GPU CFD solver parameters
 */
struct CFDSolverGPUParams {
    int nx, ny, nz;         // Grid dimensions
    float dx;               // Cell size (m)
    float dt;               // Time step
    float specificHeat = 3800.0f;       // J/(kg*K)
    float heatTransferCoeff = 12000.0f; // W/(m^2*K)
    
    // Fluid properties
    float density;          // kg/m³
    float viscosity;        // Pa·s (dynamic)
    float thermalDiff;      // m²/s
    
    // Boundary conditions
    float inletVelocity;    // m/s
    double inletTemperature; // °C (double for thermal coupling)
    
    // Solver settings
    int maxPressureIters;   // Max pressure solve iterations
    float pressureTolerance;
};

/**
 * @brief GPU-accelerated CFD solver for machining coolant simulation
 * 
 * Implements Navier-Stokes equations on MAC grid for transient flow analysis.
 */
class CFDSolverGPU : public IPhysicsSolver, public IMetricsProvider {
public:
    CFDSolverGPU();
    ~CFDSolverGPU() override;
    
    // No copy
    CFDSolverGPU(const CFDSolverGPU&) = delete;
    CFDSolverGPU& operator=(const CFDSolverGPU&) = delete;
    
    // IPhysicsSolver interface
    std::string getName() const override { return "CFDSolverGPU"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void getBounds(double& minX, double& minY, double& minZ, 
                   double& maxX, double& maxY, double& maxZ) const override;
    void reset() override;
    
    // IMetricsProvider interface
    double getMaxStress() const override { return 0.0; } // CFD doesn't compute stress
    double getMaxTemperature() const override { return m_maxTemperature; }
    double getTotalKineticEnergy() const override;
    void syncMetrics() override;

    /**
     * @brief Get velocity at world position (interpolated)
     */
    Vec3 getVelocityAt(const Vec3& pos) const;
    
    /**
     * @brief Get temperature at world position
     */
    double getTemperatureAt(const Vec3& pos) const;
    // Batch-download the temperature grid once for efficient per-particle queries
    // Returns a host vector[ntotal] that can be indexed via (k*nx*ny + j*nx + i)
    struct GridParams { int nx, ny, nz; float dx; };
    GridParams getGridParams() const { return {m_params.nx, m_params.ny, m_params.nz, m_params.dx}; }
    std::vector<double> getTemperatureGrid() const;
    
    /**
     * @brief Set solid obstacle from particle positions
     */
    void setSolidObstacles(const double* particlePositions, int numParticles);

    // ── Device-Pointer Coupling (read device CouplingPoint* arrays directly) ──
    void setSolidObstaclesFromDevice(const CouplingPoint* d_points, int numPoints);
    void setHeatSourcesFromDevice(const CouplingPoint* d_points, int numPoints);
    void setParticleHeatSourcesFromDevice(const CouplingPoint* d_points, int numPoints,
                                          double particleDensity, double particleSpecificHeat);
    
    // ── Chip-Fluid Coupling (Roadblock #10) ────────────────────────────
    /**
     * @brief Register chip particle device data for fluid interaction.
     * Must be called once during setup (or when particle count changes).
     * @param d_particles Device pointer to MPMParticle array
     * @param numParticles Total particle count
     * @param smoothingRadius MPM smoothing radius (used for chip radius estimate)
     */
    void setChipParticleData(MPMParticle* d_particles, int numParticles, double smoothingRadius);

    /**
     * @brief Apply fluid drag forces to chip particles and set immersed
     * boundary velocity conditions in the CFD grid.
     * 
     * Two-way coupling:
     *   1. For each CHIP-status particle, sample CFD velocity, compute
     *      Stokes-drag force, write to particle->ext_f (consumed by MPM p2g).
     *   2. Override CFD velocity in chip-occupied cells to match particle
     *      velocity (immersed Dirichlet BC).
     * 
     * Call AFTER cfd->step() each macro-step. Forces take effect in the
     * NEXT step's MPM sub-cycling (one-step lag, standard for partitioned
     * coupling).
     */
    void applyChipFluidCoupling();

    /**
     * @brief (Re-)mark CFD solid fraction from chip particle positions.
     * Should be called before each CFD step to update obstacle geometry.
     */
    void markChipSolidFraction();

    /**
     * @brief Set heat sources from FEM temperature
     */
    void setHeatSources(const double* nodeTemperatures, const double* nodePositions, int numNodes);

    /**
     * @brief Add heat sources from MPM particle temperatures (accumulates, does not reset)
     */
    void setParticleHeatSources(const double* particleTemperatures, const double* particlePositions, int numParticles,
                                double particleDensity, double particleSpecificHeat);

    // ── Device-side CFD cooling for MPM particles ──────────────────────
    void applyCFDCoolingDevice(MPMParticle* d_particles, int numParticles,
                               double h, double ambientTemp, double dt,
                               double& outTotalCooled, double& outTotalHeatRemoved, int& outHotCount);


    
    /**
     * @brief Get maximum velocity (for CFL)
     */
    double getMaxVelocity() const { return (double)m_maxVelocity; }
    
    /**
     * @brief Copy results to host for export
     */
    void copyVelocityToHost(std::vector<Vec3>& velocities);
    void copyTemperatureToHost(std::vector<double>& temperatures);

private:
    // ... private members stay float for CUDA performance ...
    // Grid properties
    CFDSolverGPUParams m_params;
    int m_totalCells = 0;
    
    // Device memory
    float* d_u = nullptr;       // Velocity X (staggered at face)
    float* d_v = nullptr;       // Velocity Y
    float* d_w = nullptr;       // Velocity Z
    float* d_u_temp = nullptr;  // Temporary for advection
    float* d_v_temp = nullptr;
    float* d_w_temp = nullptr;
    float* d_p = nullptr;       // Pressure
    float* d_divergence = nullptr;
    double* d_T = nullptr;      // Temperature (double for thermal coupling)
    double* d_T_temp = nullptr;
    float* d_solid = nullptr;   // Solid volume fraction [0,1] (was bool*)
    
    // Pressure solver working buffers (persistent to avoid per-step alloc)
    float* d_residual = nullptr;    // L∞ residual scalar
    float* d_pOld = nullptr;        // Previous pressure for convergence check
    float* d_maxVel = nullptr;      // Max velocity reduction result
    float* h_pinnedResidual = nullptr;  // Pinned host copy for async graph readback
    float* d_scratchFloat = nullptr;    // Reusable scratch buffer (avoids per-call cudaMalloc)
    int m_scratchFloatCapacity = 0;
    double* d_coolTotal = nullptr;      // Reduction buffer for applyCFDCoolingDevice
    double* d_coolHeatRemoved = nullptr;
    int* d_coolHotCount = nullptr;
    double* h_pinnedCoolTotal = nullptr;
    double* h_pinnedCoolHeatRemoved = nullptr;
    int* h_pinnedCoolHotCount = nullptr;
    
    // Heat sources (double for consistent thermal coupling)
    double* d_heatSource = nullptr;
    
    // ── Chip-Fluid Coupling State (Roadblock #10) ──────────────────────
    MPMParticle* d_chipParticles = nullptr;   // Device pointer (borrowed, not owned)
    int m_numChipParticles = 0;
    float m_chipRadius = 5e-5f;               // 50µm typical chip radius
    
    // Simulation state
    double m_currentTime = 0.0;
    int m_currentStep = 0;
    bool m_isInitialized = false;
    float m_maxVelocity = 0.0f;
    double m_maxTemperature = 25.0;
    
    // CUDA resources
    cudaStream_t m_stream = nullptr;
    cudaGraphExec_t m_pressureGraph = nullptr;
    
    // Helper methods
    void allocateMemory();
    void freeMemory();
    void applyBoundaryConditions();
    void advectVelocity(float dt);
    void addForces(float dt);
    void computeDivergence();
    void solvePressure();
    void subtractPressureGradient();
    void advectTemperature(float dt);
    void diffuseTemperature(float dt);
    void updateMetrics();
    
    // Grid indexing
    __host__ __device__ inline int idx(int i, int j, int k) const {
        return i + j * m_params.nx + k * m_params.nx * m_params.ny;
    }
};

// ============================================================================
// CUDA Kernels (declarations)
// ============================================================================

/**
 * @brief Semi-Lagrangian advection kernel
 */
__global__ void advectVelocityKernel(
    const float* u, const float* v, const float* w,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz, float dx, float dt);

/**
 * @brief Divergence calculation kernel
 */
__global__ void computeDivergenceKernel(
    const float* u, const float* v, const float* w,
    float* div, int nx, int ny, int nz, float dx);

/**
 * @brief Jacobi pressure solver iteration
 */
__global__ void jacobiPressureKernel(
    const float* p, float* p_new, const float* div,
    const float* solid, int nx, int ny, int nz, float dx);

/**
 * @brief Red-Black Gauss-Seidel pressure solve
 */
__global__ void redBlackGaussSeidelKernel(
    float* p, const float* div, const float* solid,
    int nx, int ny, int nz, float dx, int color);

/**
 * @brief Pressure gradient subtraction
 */
__global__ void subtractGradientKernel(
    float* u, float* v, float* w, const float* p,
    const float* solid, int nx, int ny, int nz, float dx, float dt, float rho);

/**
 * @brief Temperature advection kernel (double precision)
 */
__global__ void advectTemperatureKernel(
    const double* T, double* T_out,
    const float* u, const float* v, const float* w,
    int nx, int ny, int nz, float dx, float dt);

/**
 * @brief Temperature diffusion kernel (implicit Jacobi, double precision)
 */
__global__ void diffuseTemperatureKernel(
    const double* T, double* T_out, const double* heatSource,
    int nx, int ny, int nz, float dx, float dt, float alpha);

/**
 * @brief Apply boundary conditions kernel
 */
__global__ void applyBoundaryKernel(
    float* u, float* v, float* w, double* T,
    int nx, int ny, int nz, float inletU, double inletT);

/**
 * @brief Find maximum velocity (reduction)
 */
__global__ void findMaxVelocityKernel(
    const float* u, const float* v, const float* w,
    float* maxVal, int n);

__global__ void pressureResidualKernel(
    const float* p, const float* pOld, float* residual, int n);

/**
 * @brief Set solid from coupling point device array
 */
__global__ void markSolidFromCouplingPointsKernel(
    float* solid, const CouplingPoint* points, int numPoints,
    int nx, int ny, int nz, float dx, float originX, float originY, float originZ);

/**
 * @brief Map coupling point heat sources (FEM-style: fluid density/specific heat)
 */
__global__ void mapCouplingPointHeatSourcesKernel(
    double* heatSource, const double* fluidTemp,
    const CouplingPoint* points, int numPoints,
    int nx, int ny, int nz, float dx,
    float fluidDensity, float fluidSpecificHeat, float heatTransferCoeff,
    float originX, float originY, float originZ);

/**
 * @brief Map coupling point heat sources (MPM-style: particle density/specific heat)
 */
__global__ void mapCouplingPointParticleHeatSourcesKernel(
    double* heatSource, const double* fluidTemp,
    const CouplingPoint* points, int numPoints,
    int nx, int ny, int nz, float dx,
    double particleDensity, double particleSpecificHeat, double heatTransferCoeff,
    float originX, float originY, float originZ);

/**
 * @brief Set solid from particles
 */
__global__ void markSolidFromParticlesKernel(
    float* solid, const float* particlePos, int numParticles,
    int nx, int ny, int nz, float dx, float originX, float originY, float originZ);

/**
 * @brief Map particle temperatures to heat source grid (accumulates into existing heatSource)
 * @note Temperature fields are double precision for consistent thermal coupling
 */
__global__ void mapParticleHeatSourcesKernel(
    double* heatSource, const double* fluidTemp,
    const float* particlePos, const double* particleTemp, int numParticles,
    int nx, int ny, int nz, float dx,
    double particleDensity, double particleSpecificHeat, double heatTransferCoeff,
    float originX, float originY, float originZ);

/**
 * @brief Map temperatures to heat source grid
 */
__global__ void mapHeatSourcesKernel(
    double* heatSource, const double* fluidTemp,
    const float* nodePos, const double* nodeTemp, int numNodes,
    int nx, int ny, int nz, float dx,
    float fluidDensity, float fluidSpecificHeat, float heatTransferCoeff,
    float originX, float originY, float originZ);

// ── Chip-Fluid Coupling Kernels (Roadblock #10) ───────────────────────────

/**
 * @brief Apply Stokes/Morison fluid drag force to CHIP-status MPM particles.
 * 
 * For each particle with status == CHIP:
 *   1. Sample CFD velocity at particle position (trilinear interpolation)
 *   2. Compute relative velocity: v_rel = v_fluid - v_particle
 *   3. Compute Reynolds number: Re = rho_f * |v_rel| * d_p / mu
 *   4. Compute drag coefficient: Cd = 24/Re + 6/(1+√Re) + 0.4 (Morison)
 *   5. Drag force: F_drag = 0.5 * Cd * rho_f * A * |v_rel| * v_rel
 *   6. Write F_drag to particle.ext_f fields (consumed by MPM p2g)
 *
 * @param particles  Device array of MPMParticle
 * @param numParticles Total particle count
 * @param u,v,w      CFD velocity fields (staggered MAC grid, cell-centered samples)
 * @param nx,ny,nz   CFD grid dimensions
 * @param dx         CFD cell size
 * @param fluidViscosity  Dynamic viscosity (Pa·s)
 * @param fluidDensity    Fluid density (kg/m³)
 * @param chipRadius      Characteristic chip radius (m) — typically 0.5 * smoothingRadius
 * @param originX,originY,originZ  CFD grid origin in world coords
 */
__global__ void applyFluidDragToChipsKernel(
    MPMParticle* particles, int numParticles,
    const float* u, const float* v, const float* w,
    int nx, int ny, int nz, float dx,
    float fluidViscosity, float fluidDensity,
    float chipRadius,
    float originX, float originY, float originZ);

/**
 * @brief Impose chip velocity as immersed Dirichlet BC in the CFD velocity field.
 * 
 * For each chip particle, map to the nearest CFD cell and set the fluid velocity
 * to match the particle velocity (no-slip condition on the chip surface).
 * Uses a Gaussian-weighted influence region (3×3×3 neighbourhood) so that a chip
 * spanning multiple cells smoothly imposes its velocity on the surrounding fluid.
 *
 * Call AFTER the pressure projection (subtractGradientKernel) to enforce the BC.
 *
 * @param particles  Device array of MPMParticle
 * @param numParticles Total particle count
 * @param u,v,w      CFD velocity fields (cell-centered, will be overwritten in chip cells)
 * @param nx,ny,nz   CFD grid dimensions
 * @param dx         CFD cell size
 * @param chipRadius Characteristic chip radius (m)
 * @param originX,originY,originZ  CFD grid origin
 */
__global__ void imposeChipVelocityBCKernel(
    const MPMParticle* particles, int numParticles,
    float* u, float* v, float* w,
    int nx, int ny, int nz, float dx,
    float chipRadius,
    float originX, float originY, float originZ);

__global__ void applyCFDCoolingKernel(
    MPMParticle* particles, int numParticles,
    double ambientTemp, double h, double dt,
    double* d_totalCooled, double* d_totalHeatRemoved, int* d_hotCount);

} // namespace edgepredict
