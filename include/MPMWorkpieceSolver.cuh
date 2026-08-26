#pragma once
/**
 * @file MPMWorkpieceSolver.cuh
 * @brief CUDA Material Point Method workpiece solver.
 *
 * GPU-first MPM implementation for metal cutting. Particles carry material
 * history; a background grid solves momentum, stress divergence, tool contact,
 * fixture constraints, and thermal contact response.
 */

#include "IPhysicsSolver.h"
#include "Types.h"
#include <cuda_runtime.h>
#include <vector>

namespace edgepredict {

struct MPMParticleGPU {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
    double mass;
    double volume;
    double density;
    double temperature;
    double plasticStrain;
    double strainRate;
    double damage;
    double residualStress;
    double stress_xx, stress_yy, stress_zz;
    double stress_xy, stress_xz, stress_yz;
    double backstress_xx, backstress_yy, backstress_zz;
    double backstress_xy, backstress_xz, backstress_yz;
    int id;
    ParticleStatus status;

    EP_HOST_DEVICE MPMParticleGPU()
        : x(0), y(0), z(0), vx(0), vy(0), vz(0), fx(0), fy(0), fz(0),
          mass(0), volume(0), density(0), temperature(25.0),
          plasticStrain(0), strainRate(0), damage(0), residualStress(0),
          stress_xx(0), stress_yy(0), stress_zz(0), stress_xy(0), stress_xz(0), stress_yz(0),
          backstress_xx(0), backstress_yy(0), backstress_zz(0),
          backstress_xy(0), backstress_xz(0), backstress_yz(0),
          id(-1), status(ParticleStatus::ACTIVE) {}
};

struct MPMGridNodeGPU {
    double mass;
    double vx, vy, vz;
    double fx, fy, fz;
    double temperatureMass;
    int fixed;

    EP_HOST_DEVICE MPMGridNodeGPU()
        : mass(0), vx(0), vy(0), vz(0), fx(0), fy(0), fz(0),
          temperatureMass(0), fixed(0) {}
};

struct MPMGridParams {
    double cellSize = 0.0004;
    double minX = 0, minY = 0, minZ = 0;
    int nx = 1, ny = 1, nz = 1;
};

struct MPMMaterialParams {
    double density = 7850.0;
    double young = 210e9;
    double poisson = 0.29;
    double shear = 81e9;
    double bulk = 175e9;
    double specificHeat = 475.0;
    double meltTemp = 1460.0;
    double refTemp = 25.0;
    double jcA = 595e6;
    double jcB = 580e6;
    double jcN = 0.133;
    double jcC = 0.023;
    double jcM = 1.03;
    double refStrainRate = 1.0;
    double failureStrain = 0.3;
    double damageThreshold = 1.0;
    double criticalPlasticStrain = 2.0;
    double thermalChipRatio = 0.98;
    double toolContactRadius = 0.0006;
    double contactDamping = 0.15;
    double maxContactTempRise = 25.0;
    double thermalConductivity = 42.6;        // W/mK — AISI 4140 steel @ 25°C
    double kinematicHardeningModulus = 0.0;   // Pa — 0 = isotropic only
    int thermalClampEnabled = 1;
};

class MPMWorkpieceSolver : public IPhysicsSolver,
                           public IMetricsProvider,
                           public IThermalCoupling {
public:
    MPMWorkpieceSolver() = default;
    ~MPMWorkpieceSolver() override;

    MPMWorkpieceSolver(const MPMWorkpieceSolver&) = delete;
    MPMWorkpieceSolver& operator=(const MPMWorkpieceSolver&) = delete;

    std::string getName() const override { return "MPMWorkpieceSolver"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void reset() override;
    void getBounds(double& minX, double& minY, double& minZ,
                   double& maxX, double& maxY, double& maxZ) const override;
    int getParticleCount() const override { return m_numParticles; }

    double getMaxStress() const override { return m_maxStress; }
    double getMaxTemperature() const override { return m_maxTemperature; }
    double getTotalKineticEnergy() const override { return m_kineticEnergy; }
    void syncMetrics() override { updateMetrics(); }

    void applyHeatFlux(double x, double y, double z, double heatFlux) override;
    double getTemperatureAt(double x, double y, double z) const override;
    void setParticleTemperatures(const double* temperatures, int count);

    void initializeParticleBox(const Vec3& minBounds, const Vec3& maxBounds, double spacing);
    void initializeCylindricalWorkpiece(const Vec3& center, double radius,
                                        double length, double spacing, int axis = 2);
    void setToolMesh(const Mesh& mesh, double contactRadius);
    void applyExternalForce(const Vec3& center, double radius, const Vec3& force);
    std::vector<MPMParticle> getParticlesForExport();
    double getSmoothingRadius() const { return m_gridParams.cellSize; }
    double getTotalContactForce() const { return m_totalContactForce; }
    void compactParticles();

    /// Set the current tool tip position (for geometric chip separation)
    void setToolTipPosition(double x, double y, double z) {
        m_toolTipPosX = x; m_toolTipPosY = y; m_toolTipPosZ = z;
    }

private:
    void allocateParticles(int capacity);
    void allocateGrid();
    void freeMemory();
    void uploadParticles();
    void downloadParticles();
    void rebuildGridFromHostBounds();
    void clearGrid();
    void updateMetrics();

    const Config* m_config = nullptr;
    std::vector<MPMParticleGPU> h_particles;
    std::vector<Vec3> h_toolSamples;

    MPMParticleGPU* d_particles = nullptr;
    MPMParticleGPU* d_tempParticles = nullptr;
    MPMGridNodeGPU* d_grid = nullptr;
    Vec3* d_toolSamples = nullptr;
    double* d_metrics = nullptr; // maxStress, maxTemp, kineticEnergy
    double* d_contactForce = nullptr;

    int m_capacity = 0;
    int m_numParticles = 0;
    int m_numGridNodes = 0;
    int m_numToolSamples = 0;
    int m_compactionCount = 0;

    MPMGridParams m_gridParams;
    MPMMaterialParams m_material;

    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    int m_currentStep = 0;
    double m_particleSpacing = 0.00032;

    double m_maxStress = 0.0;
    double m_maxTemperature = 25.0;
    double m_kineticEnergy = 0.0;
    double m_totalContactForce = 0.0;

    // Tool tip position (used by chip fracture/separation kernel)
    double m_toolTipPosX = 0.0, m_toolTipPosY = 0.0, m_toolTipPosZ = 0.0;
};

} // namespace edgepredict
