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

namespace edgepredict {

/**
 * @brief MAC grid cell for CFD
 */
struct CFDCell {
    float u, v, w;          // Velocity components (staggered)
    float p;                // Pressure (cell center)
    float T;                // Temperature
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
    
    // Fluid properties
    float density;          // kg/m³
    float viscosity;        // Pa·s (dynamic)
    float thermalDiff;      // m²/s
    
    // Boundary conditions
    float inletVelocity;    // m/s
    float inletTemperature; // °C
    
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
    double getMaxTemperature() const override { return (double)m_maxTemperature; }
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
    
    /**
     * @brief Set solid obstacle from particle positions
     */
    void setSolidObstacles(const double* particlePositions, int numParticles);
    
    /**
     * @brief Set heat sources from FEM temperature
     */
    void setHeatSources(const double* nodeTemperatures, const double* nodePositions, int numNodes);
    
    /**
     * @brief Get maximum velocity (for CFL)
     */
    double getMaxVelocity() const { return (double)m_maxVelocity; }
    
    /**
     * @brief Copy results to host for export
     */
    void copyVelocityToHost(std::vector<Vec3>& velocities);
    void copyTemperatureToHost(std::vector<float>& temperatures);

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
    float* d_T = nullptr;       // Temperature
    float* d_T_temp = nullptr;
    bool* d_solid = nullptr;    // Solid flags
    
    // Heat sources
    float* d_heatSource = nullptr;
    
    // Simulation state
    double m_currentTime = 0.0;
    int m_currentStep = 0;
    bool m_isInitialized = false;
    float m_maxVelocity = 0.0f;
    float m_maxTemperature = 25.0f;
    
    // CUDA resources
    cudaStream_t m_stream = nullptr;
    
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
    const bool* solid, int nx, int ny, int nz, float dx);

/**
 * @brief Red-Black Gauss-Seidel pressure solve
 */
__global__ void redBlackGaussSeidelKernel(
    float* p, const float* div, const bool* solid,
    int nx, int ny, int nz, float dx, int color);

/**
 * @brief Pressure gradient subtraction
 */
__global__ void subtractGradientKernel(
    float* u, float* v, float* w, const float* p,
    const bool* solid, int nx, int ny, int nz, float dx, float dt, float rho);

/**
 * @brief Temperature advection kernel
 */
__global__ void advectTemperatureKernel(
    const float* T, float* T_out,
    const float* u, const float* v, const float* w,
    int nx, int ny, int nz, float dx, float dt);

/**
 * @brief Temperature diffusion kernel (implicit Jacobi)
 */
__global__ void diffuseTemperatureKernel(
    const float* T, float* T_out, const float* heatSource,
    int nx, int ny, int nz, float dx, float dt, float alpha);

/**
 * @brief Apply boundary conditions kernel
 */
__global__ void applyBoundaryKernel(
    float* u, float* v, float* w, float* T,
    int nx, int ny, int nz, float inletU, float inletT);

/**
 * @brief Find maximum velocity (reduction)
 */
__global__ void findMaxVelocityKernel(
    const float* u, const float* v, const float* w,
    float* maxVal, int n);

/**
 * @brief Set solid from particles
 */
__global__ void markSolidFromParticlesKernel(
    bool* solid, const float* particlePos, int numParticles,
    int nx, int ny, int nz, float dx, float originX, float originY, float originZ);

/**
 * @brief Map temperatures to heat source grid
 */
__global__ void mapHeatSourcesKernel(
    float* heatSource, const float* nodePos, const float* nodeTemp, int numNodes,
    int nx, int ny, int nz, float dx, float originX, float originY, float originZ);

} // namespace edgepredict
