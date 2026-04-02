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
#include <vector>

namespace edgepredict {

// Forward declarations or use from Types.h
struct FEMSpring;
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
    
    /**
     * @brief Set material properties
     */
    void setMaterial(const FEMMaterialProps& props) { m_material = props; }
    
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

private:
    void allocateMemory(int numNodes, int numSprings);
    void freeMemory();
    void copyToDevice();
    void copyFromDevice();
    void createSpringsFromMesh(const Mesh& mesh);
    void computeSpringForces();
    void integrate(double dt);
    void updateStress();
    void updateWear(double dt);
    void applyDamping();
    void updateResults();
    
    // Host data
    std::vector<FEMNodeGPU> h_nodes;
    std::vector<FEMSpring> h_springs;
    std::vector<Triangle> m_meshTriangles;  // Store original triangles for export
    
    // Device data
    FEMNodeGPU* d_nodes = nullptr;
    FEMSpring* d_springs = nullptr;
    
    // Configuration
    FEMMaterialProps m_material;
    double m_globalDamping = 0.1;
    int m_maxNodes = 50000;
    int m_maxSprings = 200000;
    double m_massScalingFactor = 1.0;      // Increase mass -> larger dt -> faster
    double m_stiffnessScalingFactor = 1.0; // Decrease stiffness -> larger dt -> faster
    int m_numNodes = 0;
    int m_numSprings = 0;
    
    // Tool transform
    Vec3 m_toolPosition;
    Vec3 m_toolAxis;
    double m_toolAngle = 0;
    double m_toolAngularVelocity = 0;

    // CUDA streams
    cudaStream_t m_computeStream = nullptr;

    // State
    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    double m_stableTimeStep = 1e-6;
    int m_currentStep = 0;
    double m_refTemp = 25.0;
    
    // === ANCHORED PHYSICS: Spindle State ===
    bool m_spindleDynamicsEnabled = false;
    double m_spindleStiffness = 5e7;
    double m_spindleDamping = 1e4;
    Vec3 m_spindlePos;
    Vec3 m_spindleVel;
    
    FEMResults m_results;
};

} // namespace edgepredict
