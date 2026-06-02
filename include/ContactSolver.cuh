#pragma once
/**
 * @file ContactSolver.cuh
 * @brief Tool-workpiece contact detection and force computation
 * 
 * Key improvements over v3:
 * - Spatial hash acceleration (not O(N×M) brute force)
 * - Proper Coulomb friction (not velocity damping)
 * - Heat generation at contact
 * - ToolCoatingModel integration
 */

#include "Types.h"
#include "CudaUtils.cuh"
#include <vector>

namespace edgepredict {

// Forward declarations to break dependency cycles
class MPMSolver;
class FEMSolver;
class ToolCoatingModel;

/**
 * @brief Contact detection configuration
 */
struct ContactConfig {
    double contactRadius = 0.0002;      // Detection radius (m)
    double contactStiffness = 1e7;      // Contact stiffness (N/m)
    double contactDampingRatio = 0.25;  // Normal contact damping ratio
    double frictionCoefficient = 0.3;   // Coulomb friction
    double heatPartition = 0.5;         // Fraction of heat to tool
    double plasticWorkFraction = 0.9;   // Fraction of plastic work converted to heat
    double workpieceShearYield = 378.2e6;  // τ_max = σ_yield/√3 (Pa) — default Steel 4140 (655/√3)
    double maxContactPressure = 4.0e9;     // Plastic pressure cap for stress reporting (Pa)
    int    engagementRampSteps = 200;      // Gradual force ramp over first N contact steps
    double maxPenetrationFraction = 0.5;   // Max penetration as fraction of contactRadius
    
    // Thermal properties
    double toolSpecificHeat = 200.0;       // J/(kg·K)
    double workSpecificHeat = 475.0;       // J/(kg·K)
    double workPhysicalDensity = 7850.0;   // kg/m^3, used for contact thermal mass
    double toolMeltTemp = 2870.0;          // °C
    double workMeltTemp = 1460.0;          // °C
};

/**
 * @brief Contact event for post-processing
 */
struct ContactEvent {
    int particleIndex;
    int nodeIndex;
    Vec3 position;
    Vec3 normal;
    Vec3 force;
    double penetration;
    double heatGenerated;
};

/**
 * @brief Contact solver for SPH-FEM interaction
 */
class ContactSolver {
public:
    ContactSolver();
    ~ContactSolver();
    
    // Non-copyable
    ContactSolver(const ContactSolver&) = delete;
    ContactSolver& operator=(const ContactSolver&) = delete;
    
    /**
     * @brief Initialize with SPH and FEM solvers
     */
    void initialize(MPMSolver* sph, FEMSolver* fem, const ContactConfig& config);
    
    /**
     * @brief Detect and resolve contacts
     * @param dt Time step
     */
    void resolveContacts(double dt);
    
    /**
     * @brief Get results
     */
    int getContactCount() const { return m_numContacts; }
    double getHeatGenerated() const { return m_totalHeatGenerated; }
    double getTotalContactForce() const { return m_totalContactForce; }
    double getEstimatedContactPressure() const;
    std::vector<ContactEvent> getContactEvents() const;
    void applyCoatingWearToMesh(Mesh& mesh) const;
    
    /**
     * @brief Setters / Getters
     */
    void setConfig(const ContactConfig& config) { m_config = config; }
    const ContactConfig& getConfig() const { return m_config; }
    void setToolCoatingModel(ToolCoatingModel* model) { m_toolCoatingModel = model; }

private:
    void buildSpatialHash();
    void launchContactKernel(double dt);
    void transferResults();
    
    // References
    MPMSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ToolCoatingModel* m_toolCoatingModel = nullptr;
    
    ContactConfig m_config;
    
    // Spatial hash for tool nodes
    int* d_toolCellStart = nullptr;
    int* d_toolCellEnd = nullptr;
    int* d_toolNodeOrder = nullptr;
    int* d_toolNodeHashes = nullptr;       // Per-node cell hash
    FEMNodeGPU* d_sortedToolNodes = nullptr; // Sorted copy for cache-coherent access
    int m_hashTableSize = 100003;          // Prime for hash table
    double m_cellSize = 0.0002;            // Set to contactRadius at initialize
    
    // Results (device)
    int* d_numContacts = nullptr;
    double* d_totalHeat = nullptr;
    double* d_totalForce = nullptr;
    
    // Results (host)
    int m_numContacts = 0;
    double m_totalHeatGenerated = 0;
    double m_totalContactForce = 0;
    
    bool m_isInitialized = false;
    int m_contactAge = 0;  // Steps since first contact — for engagement ramp
    int m_lastContactStep = 0;  // Last step with active contacts — for ramp reset hysteresis
    int m_resolveCallCount = 0;  // Total resolveContacts calls — internal step counter
    bool m_spatialHashReady = false; // Set true after first successful hash build
};

// ============================================================================
// Free function for launching contact kernel (called directly)
// ============================================================================

void launchContactInteraction(
    MPMParticle* particles, int numParticles,
    FEMNodeGPU* nodes, int numNodes,
    double contactRadius, double contactStiffness,
    double friction, double heatPartition, double dt
);

} // namespace edgepredict
