#pragma once
/**
 * @file MPMSolver.cuh
 * @brief Material Point Method (MPM) solver for workpiece simulation
 */

#include "Types.h"
#include "IPhysicsSolver.h"
#include "CudaUtils.cuh"
#include "AdiabaticShearModel.cuh"
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace edgepredict {

struct MPMResults {
    double maxStress = 0;
    double maxTemperature = 25.0;
    double totalKineticEnergy = 0;
    int activeParticleCount = 0;
    int chipParticleCount = 0;
    
    MPMResults() = default;
};

struct MPMKernelConfig {
    double dx;              // Grid cell spacing
    double invDx;           // 1.0 / dx
    double dt;
    double youngsModulus;
    double poissonsRatio;
    double yieldStrength;

    // Material properties for thermal evolution
    double density;
    double specificHeat;
    double taylorQuinney;    // Fraction of plastic work → heat (0.9)
    double meltingPoint;
    double ambientTemp;

    // Johnson-Cook strength model
    double jc_A;             // Yield strength (Pa)
    double jc_B;             // Strain hardening modulus (Pa)
    double jc_n;             // Strain hardening exponent
    double jc_C;             // Strain rate sensitivity
    double jc_m;             // Thermal softening exponent

    // Johnson-Cook failure / damage model
    double jc_D1, jc_D2, jc_D3, jc_D4, jc_D5;
    double damageThreshold;
    double referenceStrainRate;

    // LOD (Level of Detail) zoning
    double lodActiveRadius;  // Full physics radius (m)
    double lodNearRadius;    // Reduced physics radius (m)

    MPMKernelConfig()
        : dx(0.001), invDx(1000.0), dt(1e-6),
          youngsModulus(200e9), poissonsRatio(0.3), yieldStrength(500e6),
          density(7850.0), specificHeat(475.0), taylorQuinney(0.9),
          meltingPoint(1500.0), ambientTemp(25.0),
          jc_A(595e6), jc_B(580e6), jc_n(0.133), jc_C(0.023), jc_m(1.03),
          jc_D1(0.06), jc_D2(3.31), jc_D3(-1.96), jc_D4(0.018), jc_D5(0.58),
          damageThreshold(1.0), referenceStrainRate(1.0),
          lodActiveRadius(0.006), lodNearRadius(0.025) {}
};

class MPMSolver : public IPhysicsSolver, 
                  public IMetricsProvider,
                  public IThermalCoupling {
public:
    MPMSolver();
    ~MPMSolver() override;
    
    // Non-copyable
    MPMSolver(const MPMSolver&) = delete;
    MPMSolver& operator=(const MPMSolver&) = delete;
    
    // Tool collision
    void setToolNodes(FEMNodeGPU* nodes, int numNodes) {
        d_toolNodes = nodes;
        m_numToolNodes = numNodes;
    }

    // Getters for analytics
    MPMGridNode* getGridDevicePtr() const { return d_grid; }
    int getNumGridNodes() const { return m_numGridNodes; }
    int3 getGridDimensions() const { return make_int3(m_gridDimX, m_gridDimY, m_gridDimZ); }
    const MPMKernelConfig& getKernelConfig() const { return m_config; }
    const Vec3& getDomainMin() const { return m_domainMin; }

    // IPhysicsSolver interface
    std::string getName() const override { return "MPMSolver"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void reset() override;
    void getBounds(double& minX, double& minY, double& minZ, 
                   double& maxX, double& maxY, double& maxZ) const override;
    
    int getNodeCount() const override { return m_numGridNodes; }
    int getParticleCount() const override { return m_numParticles; }
    
    // IMetricsProvider interface
    double getMaxStress() const override { return m_maxStress; }
    double getMaxTemperature() const override { return m_maxTemperature; }
    double getTotalKineticEnergy() const override { return m_kineticEnergy; }
    void syncMetrics() override;
    
    // IThermalCoupling interface
    void applyHeatFlux(double x, double y, double z, double heatFlux) override;
    double getTemperatureAt(double x, double y, double z) const override;

    void initializeParticleBox(const Vec3& minBounds, const Vec3& maxBounds, double spacing);
    void initializeCylindricalWorkpiece(const Vec3& center, double radius, double length, double spacing, int axis = 2);

    std::vector<MPMParticle> getParticles();
    MPMParticle* getDeviceParticles() { return d_particles; }
    MPMGridNode* getDeviceGrid() { return d_grid; }
    
    void setToolPosition(const Vec3& pos) { m_toolPosition = pos; }
    Vec3 getToolPosition() const { return m_toolPosition; }
    void applyExternalForce(const Vec3& center, double radius, const Vec3& force);
    void cutParticles(const Vec3& planeNormal, double planeDist);
    void resetExternalForces();
    void updateResults();
    void setLODEnabled(bool enabled) { m_lodEnabled = enabled; }
    
    double getSmoothingRadius() const { return m_config.dx * 1.5; }

    std::vector<MPMParticle>& getHostParticles() { return h_particles; }
    void updateParticlesFromHost() { copyToDevice(); }

private:
    void allocateMemory(int particleCapacity, int gridCapacity);
    void freeMemory();
    void copyToDevice();
    void copyFromDevice();
    
    // MPM Core Steps
    void resetGrid();
    void p2g();
    void updateGrid(double dt);
    void g2p(double dt);
    
    // Configuration
    const Config* m_mainConfig = nullptr;
    MPMKernelConfig m_config;
    int m_maxParticles = 5000000;
    int m_numParticles = 0;
    
    int m_gridDimX, m_gridDimY, m_gridDimZ;
    int m_numGridNodes = 0;
    
    // GPU pointers
    MPMParticle* d_particles = nullptr;
    MPMGridNode* d_grid = nullptr;
    
    // Tool
    FEMNodeGPU* d_toolNodes = nullptr;
    int m_numToolNodes = 0;
    
    // Host data (pinned for fast transfer)
    MPMParticle* h_pinnedParticles = nullptr;
    std::vector<MPMParticle> h_particles;
    
    Vec3 m_domainMin;
    Vec3 m_domainMax;
    Vec3 m_particleMin;
    Vec3 m_particleMax;
    
    // Simulation state
    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    int m_currentStep = 0;
    Vec3 m_toolPosition;
    bool m_lodEnabled = true;
    
    MPMResults m_results;
    double m_maxStress = 0;
    double m_maxTemperature = 25.0;
    double m_kineticEnergy = 0;
    
    // Device-side bounds storage (atomic min/max update)
    double* d_boundsMin = nullptr;  // 3 doubles: x, y, z
    double* d_boundsMax = nullptr;  // 3 doubles: x, y, z
    
    AdiabaticShearModel m_adiabaticShearModel;
    cudaStream_t m_computeStream = nullptr;
};

// ============================================================================
// New Kernel Declarations
// ============================================================================

/**
 * @brief Apply external body force to particles within a radius of center.
 *        Force is distributed with linear falloff: w = 1 - d/R.
 *        Stored in particle's ext_f fields, consumed by p2gKernel.
 */
__global__ void applyExternalForceKernel(MPMParticle* particles, int numParticles,
    Vec3 center, double radius, Vec3 force);

/**
 * @brief Cut/remove particles on one side of a plane.
 *        Particles where planeNormal · pos > planeDist are set to INACTIVE.
 */
__global__ void cutParticlesKernel(MPMParticle* particles, int numParticles,
    Vec3 planeNormal, double planeDist);

/**
 * @brief Classify particles into LOD zones based on distance from tool.
 *        ACTIVE = near tool (full physics)
 *        NEAR   = medium distance (reduced frequency)
 *        FAR    = far from tool (kinematic only)
 */
__global__ void lodClassifyKernel(MPMParticle* particles, int numParticles,
    Vec3 toolPos, double activeRadius, double nearRadius);

/**
 * @brief Reset external force fields to zero (called at end of each step).
 */
__global__ void resetExternalForceKernel(MPMParticle* particles, int numParticles);

/**
 * @brief Update particle bounds (min/max) via atomic CAS operations.
 *        Used for air-gap detection and logging.
 */
__global__ void updateParticleBoundsKernel(MPMParticle* particles, int numParticles,
    double* minX, double* minY, double* minZ,
    double* maxX, double* maxY, double* maxZ);

/**
 * @brief Compute aggregate stats (max stress, max temp, kinetic energy) from particles.
 */
__global__ void computeParticleStatsKernel(MPMParticle* particles, int numParticles,
    double* maxStress, double* maxTemp, double* kinEnergy);

} // namespace edgepredict
