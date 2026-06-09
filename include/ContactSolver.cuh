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
    double workpieceShearYield = 378.2e6;  // τ_max = σ_yield/√3 (Pa) — fallback when JC is disabled
    double maxContactPressure = 4.0e9;     // Plastic pressure cap for stress reporting (Pa)
    // Johnson-Cook flow stress for contact shear cap (replaces fixed shearYield when enabled)
    double jc_A = 553.0e6;      // Yield stress (Pa) — AISI 1045 defaults
    double jc_B = 600.0e6;      // Hardening modulus (Pa)
    double jc_n = 0.234;        // Hardening exponent
    double jc_C = 0.013;        // Strain rate sensitivity
    double jc_m = 1.0;          // Thermal softening exponent
    double jc_strainRateRef = 1.0;   // Reference strain rate (1/s)
    double jc_T_ref = 25.0;          // Reference temperature (°C)
    double jc_T_melt = 1520.0;       // Melt temperature (°C)
    bool jcEnabled = true;           // Use per-particle JC flow stress instead of fixed shearYield
    int    engagementRampSteps = 200;      // Gradual force ramp over first N contact steps
    double maxPenetrationFraction = 0.5;   // Max penetration as fraction of contactRadius
    
    // Thermal properties
    double toolSpecificHeat = 200.0;       // J/(kg·K)
    double workSpecificHeat = 475.0;       // J/(kg·K)
    double workPhysicalDensity = 7850.0;   // kg/m^3, used for contact thermal mass
    double toolMeltTemp = 2870.0;          // °C
    double workMeltTemp = 1460.0;          // °C
    double contactHTC = 1.0e5;             // W/m²K (updated each step from CFD/hardening model)
    // Radiation (Item 5)
    double surfaceEmissivity = 0.7;
    double stefanBoltzmann = 5.670374419e-8; // W/m²K⁴
    double ambientTemperature = 25.0;        // °C
    // Phase change (Items 6, 9)
    double toolPhysicalDensity = 14500.0;    // kg/m³ — tungsten carbide
    double latentHeatFusion = 0.0;           // J/kg (0 = disabled, ~280 kJ/kg for carbide)
    double solidusTemp = 0.0;                // °C — mushy zone start (0 = use single melt temp)
    double liquidusTemp = 0.0;               // °C — mushy zone end (0 = use single melt temp)
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
    void setHeatPartition(double hp) { m_config.heatPartition = hp; }
    void setContactHTC(double htc) { m_config.contactHTC = htc; }
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
    double friction, double heatPartition, double dt,
    double contactHTC = 1.0e5,
    bool   jcEnabled = true,
    double jc_A = 553.0e6, double jc_B = 600.0e6, double jc_n = 0.234,
    double jc_C = 0.013, double jc_m = 1.0,
    double jc_strainRateRef = 1.0, double jc_T_ref = 25.0, double jc_T_melt = 1520.0
);

} // namespace edgepredict
