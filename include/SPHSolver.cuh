#pragma once
/**
 * @file SPHSolver.cuh
 * @brief Smoothed Particle Hydrodynamics solver for workpiece simulation
 * 
 * Clean v4 implementation:
 * - Particle-based flow simulation (Ti-6Al-4V / 4340 Steel)
 * - Spatial hash neighbor search with Thrust sorting
 * - Johnson-Cook damage and chip separation
 * - Level of Detail (LOD) for performance
 * - Adiabatic Shear Band (ASB) integration
 */

#include "Types.h"
#include "IPhysicsSolver.h"
#include "CudaUtils.cuh"
#include "AdiabaticShearModel.cuh"
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace edgepredict {

/**
 * @brief SPH simulation results
 */
struct SPHResults {
    double maxDensity = 0;
    double maxPressure = 0;
    double maxVelocity = 0;
    double maxTemperature = 25.0;
    double totalKineticEnergy = 0;
    int activeParticleCount = 0;
    int chipParticleCount = 0;
    int removedParticleCount = 0;
    
    SPHResults() = default;
};

/**
 * @brief SPH kernel configuration and precomputed constants
 */
struct SHKernelConfig {
    double h;               // Smoothing radius
    double h2, h3, h6, h9;  // Powers of h
    double poly6Coeff;      // Density kernel coefficient
    double spikyGradCoeff;  // Pressure gradient kernel coefficient
    double viscLapCoeff;    // Viscosity Laplacian coefficient
    double particleMass;    // Mass of a single particle
    double restDensity;     // ρ₀
    double gasStiffness;    // B in Tait EOS
    double viscosity;       // Dynamic viscosity (optional for chips)
    
    SHKernelConfig() : h(0.0005), h2(0), h3(0), h6(0), h9(0),
                        poly6Coeff(0), spikyGradCoeff(0), viscLapCoeff(0),
                        particleMass(0), restDensity(4430), gasStiffness(1e7), 
                        viscosity(0.1) {}
};

// Internal typedef for naming consistency in .cu file
using SPHKernelConfig = SHKernelConfig;

/**
 * @brief GPU-accelerated SPH solver for workpiece material
 */
class SPHSolver : public IPhysicsSolver, 
                  public IMetricsProvider,
                  public IThermalCoupling {
public:
    SPHSolver();
    ~SPHSolver() override;
    
    // Non-copyable
    SPHSolver(const SPHSolver&) = delete;
    SPHSolver& operator=(const SPHSolver&) = delete;
    
    // IPhysicsSolver interface
    std::string getName() const override { return "SPHSolver"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void reset() override;
    void getBounds(double& minX, double& minY, double& minZ, 
                   double& maxX, double& maxY, double& maxZ) const override;
    
    // Generic geometry query
    int getNodeCount() const override { return 0; }
    int getParticleCount() const override { return m_numParticles; }
    
    // IMetricsProvider interface
    double getMaxStress() const override { return m_maxStress; }
    double getMaxTemperature() const override { return m_maxTemperature; }
    double getTotalKineticEnergy() const override { return m_kineticEnergy; }
    void syncMetrics() override;
    
    // IThermalCoupling interface
    void applyHeatFlux(double x, double y, double z, double heatFlux) override;
    double getTemperatureAt(double x, double y, double z) const override;
    /**
     * @brief Initialize particle distribution as a rectangular box (default)
     */
    void initializeParticleBox(const Vec3& minBounds, const Vec3& maxBounds, double spacing);

    /**
     * @brief Initialize particle distribution as a solid cylinder (for drilling/reaming/boring workpiece)
     * 
     * @param center Center of the top face of the cylinder
     * @param radius Radius of the cylinder
     * @param length Length of the cylinder along Z-axis
     * @param spacing Distance between particles
     * @param axis Which axis the cylinder aligns with (0=X, 1=Y, 2=Z)
     */
    void initializeCylindricalWorkpiece(const Vec3& center, double radius, double length, double spacing, int axis = 2);

    std::vector<SPHParticle> getParticles();
    SPHParticle* getDeviceParticles() { return d_particles; }
    void setToolPosition(const Vec3& pos) { m_toolPosition = pos; }
    void applyExternalForce(const Vec3& center, double radius, const Vec3& force);
    void cutParticles(const Vec3& planeNormal, double planeDist);
    void updateResults();  // Internal sync point
    

    // Legacy compatibility attributes
    std::vector<SPHParticle>& getHostParticles() { return h_particles; }
    void updateParticlesFromHost() { copyToDevice(); }

private:
    void allocateMemory(int capacity);
    void freeMemory();
    void copyToDevice();
    void copyFromDevice();
    void buildSpatialHash();
    
    // Configuration
    const Config* m_mainConfig = nullptr;
    SPHKernelConfig m_config;
    int m_maxParticles = 5000000;
    int m_numParticles = 0;
    
    // GPU pointers
    SPHParticle* d_particles = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;
    int* d_particleOrder = nullptr;
    
    // Host data (pinned for fast transfer)
    SPHParticle* h_pinnedParticles = nullptr;
    std::vector<SPHParticle> h_particles;
    
    // Spatial Hash Params
    int m_hashTableSize = 1000003;
    double m_cellSize = 0.0005;
    Vec3 m_domainMin;
    Vec3 m_domainMax;
    
    // Simulation state
    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    int m_currentStep = 0;
    Vec3 m_toolPosition;
    
    // Global Results (v3 compatibility)
    SPHResults m_results;
    
    // Interface Results (v4 metrics)
    double m_maxStress = 0;
    double m_maxTemperature = 25.0;
    double m_kineticEnergy = 0;
    int m_activeCount = 0;
    int m_chipCount = 0;
    
    // LOD and Parameters
    bool m_lodEnabled = false;
    double m_lodActiveRadius = 0.002;
    double m_lodNearRadius = 0.010;
    int m_lodNearSkipSteps = 2;
    int m_lodFarSkipSteps = 10;
    
    // Damage/Chip separation
    bool m_damageEnabled = false;
    double m_damageThreshold = 1.0;
    double m_jc_D1, m_jc_D2, m_jc_D3, m_jc_D4, m_jc_D5;
    double m_refStrainRate = 1.0;
    double m_refTemp = 25.0;
    double m_meltTemp = 1660.0;
    
    // Sub-models
    AdiabaticShearModel m_adiabaticShearModel;
    cudaStream_t m_computeStream = nullptr;
};

} // namespace edgepredict
