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
class SPHSolver;
class FEMSolver;
class ToolCoatingModel;

/**
 * @brief Contact detection configuration
 */
struct ContactConfig {
    double contactRadius = 0.0002;      // Detection radius (m)
    double contactStiffness = 1e7;      // Contact stiffness (N/m)
    double frictionCoefficient = 0.3;   // Coulomb friction
    double heatPartition = 0.5;         // Fraction of heat to tool
    double plasticWorkFraction = 0.9;   // Fraction of plastic work converted to heat
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
    void initialize(SPHSolver* sph, FEMSolver* fem, const ContactConfig& config);
    
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
    std::vector<ContactEvent> getContactEvents() const;
    
    /**
     * @brief Setters
     */
    void setConfig(const ContactConfig& config) { m_config = config; }
    void setToolCoatingModel(ToolCoatingModel* model) { m_toolCoatingModel = model; }

private:
    void buildSpatialHash();
    void launchContactKernel(double dt);
    void transferResults();
    
    // References
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ToolCoatingModel* m_toolCoatingModel = nullptr;
    
    ContactConfig m_config;
    
    // Spatial hash for tool nodes
    int* d_toolCellStart = nullptr;
    int* d_toolCellEnd = nullptr;
    int* d_toolNodeOrder = nullptr;
    int m_hashTableSize = 100000;
    double m_cellSize = 0.0002;
    
    // Results (device)
    int* d_numContacts = nullptr;
    double* d_totalHeat = nullptr;
    double* d_totalForce = nullptr;
    
    // Results (host)
    int m_numContacts = 0;
    double m_totalHeatGenerated = 0;
    double m_totalContactForce = 0;
    
    bool m_isInitialized = false;
};

// ============================================================================
// Free function for launching contact kernel (called directly)
// ============================================================================

void launchContactInteraction(
    SPHParticle* particles, int numParticles,
    FEMNodeGPU* nodes, int numNodes,
    double contactRadius, double contactStiffness,
    double friction, double heatPartition, double dt
);

} // namespace edgepredict
