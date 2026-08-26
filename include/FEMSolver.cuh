#pragma once
/**
 * @file FEMSolver.cuh
 * @brief Finite Element Method solver for tool stress analysis
 * 
 * Key improvements over v3:
 * - Actually computes stress (not returning 0!)
 * - Proper element stiffness matrix
 * - Spring creation from mesh triangles
 * - Wear model integration
 */

#include "Types.h"
#include "IPhysicsSolver.h"
#include "CudaUtils.cuh"
#include "DeviceCoupling.h"
#include <vector>

namespace edgepredict {

// Forward declarations or use from Types.h
struct FEMElement;
struct FEMNodeGPU;
class Config;

/**
 * @brief FEM simulation results
 */
struct FEMResults {
    double maxStress = 0;           // Von Mises stress (Pa)
    double avgStress = 0;
    double maxDisplacement = 0;     // m
    double maxTemperature = 0;      // °C
    double maxWear = 0;             // m
    double totalKineticEnergy = 0;  // J
    int numContactNodes = 0;
};

/**
 * @brief Material properties for FEM
 */
struct FEMMaterialProps {
    double youngsModulus = 600e9;       // Carbide tool (~600 GPa)
    double poissonsRatio = 0.22;
    double density = 14500;              // kg/m³ (carbide)
    double thermalConductivity = 80;     // W/(m·K)
    double specificHeat = 200;           // J/(kg·K)
    double yieldStrength = 4e9;          // ~4 GPa compressive
};

/**
 * @brief GPU-accelerated FEM solver for tool stress analysis
 * 
 * Uses spring-mass system for fast dynamic simulation.
 * More accurate than v3's non-functional implementation.
 */
class FEMSolver : public IPhysicsSolver, public IMetricsProvider, public IThermalCoupling {
public:
    FEMSolver();
    ~FEMSolver() override;
    
    FEMSolver(const FEMSolver&) = delete;
    FEMSolver& operator=(const FEMSolver&) = delete;
    
    // IPhysicsSolver interface
    std::string getName() const override { return "FEMSolver"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void reset() override;
    void getBounds(double& minX, double& minY, double& minZ, 
                   double& maxX, double& maxY, double& maxZ) const override;
    
    // IMetricsProvider interface
    double getMaxStress() const override { return m_results.maxStress; }
    double getMaxTemperature() const override { return m_results.maxTemperature; }
    double getTotalKineticEnergy() const override { return m_results.totalKineticEnergy; }
    void syncMetrics() override { updateResults(); }
    
    // IThermalCoupling interface
    void applyHeatFlux(double x, double y, double z, double heatFlux) override;
    double getTemperatureAt(double x, double y, double z) const override;
    
    // FEM-specific methods
    
    /**
     * @brief Initialize from mesh (creates springs from triangles)
     */
    void initializeFromMesh(const Mesh& mesh);
    
    /**
     * @brief Set tool position/orientation
     */
    void setToolTransform(const Vec3& position, const Vec3& axis, double angle);
    
    /**
     * @brief Apply external force at a node
     */
    void applyNodeForce(int nodeIndex, const Vec3& force);
    
    /**
     * @brief Apply contact force (from SPH interaction)
     */
    void applyContactForce(int nodeIndex, const Vec3& force, double heatFlux);
    
    /**
     * @brief Get node data (copies from GPU)
     */
    std::vector<FEMNodeGPU> getNodes();
    
    /**
     * @brief Get current results
     */
    const FEMResults& getResults() const { return m_results; }
    
    /**
     * @brief Export to mesh for visualization
     */
    void exportToMesh(Mesh& mesh);  // non-const: calls copyFromDevice()
    
    /**
     * @brief Access device node pointer (for interaction kernel)
     */
    FEMNodeGPU* getDeviceNodes() { return d_nodes; }
    int getNodeCount() const { return m_numNodes; }

    // Device-side coupling point array (for thermal coupling with CFD without PCIe round-trips)
    CouplingPoint* getDeviceCouplingPoints() { return d_couplingPoints; }
    void syncCouplingPoints();
    
    /**
     * @brief Set all node velocities to a uniform value (rigid-body G-Code velocity)
     */
    void setAllNodeVelocities(double vx, double vy, double vz);

    /**
     * @brief Set rigid-body velocities: translation plus spindle rotation.
     */
    void setRigidBodyNodeVelocities(const Vec3& linearVelocity,
                                    double angularVelocityZ,
                                    double centerX,
                                    double centerY);

    /**
     * @brief Set material properties
     */
    void setMaterial(const FEMMaterialProps& props) { m_material = props; }

    /**
     * @brief Apply dynamic mass scaling (AMS)
     * @param factor Multiplier for inertial density (1.0 = no scaling)
     */
    void setDynamicMassScaling(double factor);
    
    /**
     * @brief Translate all mesh nodes by given offset
     */
    void translateMesh(double dx, double dy, double dz);
    
    /**
     * @brief Rotate mesh around Z axis about a center point
     */
    void rotateAroundZ(double angle, double centerX, double centerY);
    
    // === ANCHORED PHYSICS: Dynamic Spindle Coupling ===
    /**
     * @brief Initialize driven nodes (virtual collet/clamp)
     * @param topFraction Fraction of logic to clamp (e.g., 0.2 = top 20%)
     */
    void initializeDrivenNodes(double topFraction);
    
    /**
     * @brief Configure spindle dynamics (opt-in physics)
     */
    void setSpindleDynamicsConfig(bool enabled, double stiffness, double damping);

    /**
     * @brief Update virtual spindle state for current timestep
     */
    void setVirtualSpindleState(const Vec3& pos, const Vec3& vel);

    /**
     * @brief Advance internal clock without running physics.
     * Fix L: Used during air-cutting macro-steps to keep FEM time synchronized
     * with the engine's simulation clock.
     */
    void advanceTime(double dt) { m_currentTime += dt; }
    void setCoolantHTC(double htc) { m_coolantHTC = htc; }
    cudaEvent_t getStepEvent() const { return m_stepEvent; }

    // === TWO-WAY KINEMATIC COUPLING: Cutting Force Feedback ===
    /**
     * @brief Apply resultant cutting force from strategy to tool cutting zone.
     *
     * Zeroes any residual contact forces on surface nodes, then distributes
     * the lumped cutting force (Fc + Ft + Ff) across non-driven surface nodes
     * in the bottom 30% of the tool with quadratic falloff weighting.
     * Must be called before step(dt) when spindle dynamics is enabled.
     */
    void applyCuttingForces(const Vec3& resultantForce);

private:
    int m_thermalSkip = 10;  // apply conduction only every N steps

    void allocateMemory(int maxNodes, int maxTetNodes, int maxElements);
    void freeMemory();
    void copyToDevice();
    void copyFromDevice();
    void createElementsFromMesh(const Mesh& mesh);
    void computeElementForces();
    void integrate(double dt);
    void updateStress();
    void updateWear(double dt);
    void applyDamping();
    void updateResults();
    
    // Host data
    std::vector<FEMNodeGPU> h_nodes;            // Surface
    std::vector<FEMNodeGPU> h_tetNodes;         // Volumetric interior
    std::vector<FEMElement> h_elements;
    std::vector<FEMEmbedConstraint> h_embedConstraints;
    std::vector<Triangle> m_meshTriangles;  // Store original triangles for export
    
    // Device data
    FEMNodeGPU* d_nodes = nullptr;              // Surface (receives SPH force)
    FEMNodeGPU* d_tetNodes = nullptr;           // Volumetric physics driver
    FEMElement* d_elements = nullptr;
    FEMEmbedConstraint* d_embedConstraints = nullptr;
    CouplingPoint* d_couplingPoints = nullptr;
    
    // Configuration
    FEMMaterialProps m_material;
    double m_globalDamping = 0.1;
    int m_maxNodes = 50000;
    int m_maxElements = 200000;
    double m_massScalingFactor = 1.0;
    double m_stiffnessScalingFactor = 1.0;
    int m_numNodes = 0;              // Total surface nodes
    int m_numTetNodes = 0;           // Total vol nodes
    int m_numElements = 0;           // Total tets
    int m_numEmbeds = 0;             // Typically == m_numNodes
    
    // Tool transform
    Vec3 m_toolPosition;
    Vec3 m_toolAxis;
    double m_toolAngle = 0;
    double m_toolAngularVelocity = 0;

    // CUDA streams
    cudaStream_t m_computeStream = nullptr;
    cudaEvent_t m_stepEvent = nullptr;

    // State
    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    double m_stableTimeStep = 1e-6;
    int m_currentStep = 0;
    double m_refTemp = 25.0;
    double m_typicalElementLength = 0.001;  // Fix 8: replaces static local
    
    // === ANCHORED PHYSICS: Spindle State ===
    bool m_spindleDynamicsEnabled = true;
    double m_spindleStiffness = 5e7;
    double m_spindleDamping = 1e4;
    Vec3 m_spindlePos;
    Vec3 m_spindleVel;
    
    FEMResults m_results;
    double m_coolantHTC = 1.0e3;  // W/m²K, updated from CoolantHardeningModel

    // === TWO-WAY KINEMATIC COUPLING: Device buffers ===
    double* d_cuttingMinMax = nullptr;    // [minZ, maxZ] device buffer
    int*    d_cuttingZoneCount = nullptr; // number of cutting-zone nodes
};

/**
 * @brief Synchronise FEM node (x,y,z,temperature) into a CouplingPoint array
 *        for device-side thermal coupling with CFD — eliminates PCIe round-trips.
 */
__global__ void syncFEMCouplingPointsKernel(
    const FEMNodeGPU* nodes, CouplingPoint* out, int numNodes);

/**
 * @brief Set rigid-body velocity for every FEM node on the GPU.
 *        Replaces the CPU copyFromDevice/for-loop/copyToDevice bottleneck.
 */
__global__ void setRigidBodyVelocitiesKernel(
    FEMNodeGPU* nodes, int numNodes,
    double vx, double vy, double vz,
    double omegaZ, double centerX, double centerY);

/**
 * @brief Translate all mesh nodes by a fixed offset directly on GPU.
 *        Replaces the CPU copyFromDevice/for-loop/copyToDevice bottleneck.
 */
__global__ void translateMeshKernel(FEMNodeGPU* nodes, int numNodes,
    double dx, double dy, double dz);

/**
 * @brief Rotate all mesh nodes around the Z axis on GPU.
 *        Replaces the CPU copyFromDevice/for-loop/copyToDevice bottleneck.
 */
__global__ void rotateAroundZKernel(FEMNodeGPU* nodes, int numNodes,
    double angle, double centerX, double centerY);

} // namespace edgepredict
