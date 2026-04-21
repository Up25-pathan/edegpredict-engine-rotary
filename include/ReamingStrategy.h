#pragma once
/**
 * @file ReamingStrategy.h
 * @brief Reaming operation strategy — multi-flute precision hole finishing
 * 
 * Reaming is a finishing operation that:
 *   - Uses 4-8 straight or helical flutes
 *   - Removes only 0.1-0.5 mm radial stock (very light DOC)
 *   - Generates near-zero radial forces (tool guided by hole)
 *   - Measures quality by hole roundness / h7 tolerance
 *   - Runs at lower speed and feed than drilling
 *
 * Force model is similar to milling (multi-flute engagement) but with
 * clamped DOC limits and roundness as the primary output metric.
 */

#include "IMachiningStrategy.h"
#include "MillingStrategy.h"  // Reuses Flute struct
#include <vector>

namespace edgepredict {

/**
 * @brief Reamer types
 */
enum class ReamerType {
    HAND_REAMER,         // Straight flute hand reamer
    MACHINE_REAMER,      // Spiral/helical flute machine reamer
    SHELL_REAMER,        // Large-diameter shell reamer
    ADJUSTABLE_REAMER,   // Expansion reamer
    TAPERED_REAMER       // For tapered holes
};

/**
 * @brief Reaming operation strategy
 */
class ReamingStrategy : public IMachiningStrategy {
public:
    ReamingStrategy();
    ~ReamingStrategy() override = default;
    
    // IMachiningStrategy interface
    std::string getName() const override { return "ReamingStrategy"; }
    MachiningType getType() const override { return MachiningType::REAMING; }
    
    bool initialize(const Config& config) override;
    void setToolGeometry(const ToolGeometry& geometry) override;
    void connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                        ContactSolver* contact, CFDSolverGPU* cfd) override;
    void updateConditions(const MachineState& state, double dt) override;
    std::vector<CuttingEdge> getActiveCuttingEdges() const override;
    MachiningOutput computeOutput() const override;
    void applyKinematics(double dt) override;
    bool isInitialized() const override { return m_initialized; }
    void reset() override;
    void applyAdaptiveControl(double feedMultiplier, double speedMultiplier) override;
    
    // Anchored Physics: Virtual Spindle Interface
    Vec3 getSpindlePosition() const override { return m_spindlePosition; }
    Vec3 getSpindleVelocity() const override { return m_spindleVelocity; }
    
    // Reaming-specific methods
    void setReamerType(ReamerType type) { m_reamerType = type; }
    int getEngagedFluteCount() const;
    double getChipLoad(int fluteIndex) const;
    
    /** @brief Get computed hole roundness deviation (µm) */
    double getHoleRoundness() const { return m_holeRoundnessUm; }
    
    /** @brief Check if h7 tolerance is met */
    bool meetsH7Tolerance() const;

private:
    // Solver connections
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ContactSolver* m_contact = nullptr;
    CFDSolverGPU* m_cfd = nullptr;
    
    // Configuration
    ToolGeometry m_geometry;
    ReamerType m_reamerType = ReamerType::MACHINE_REAMER;
    
    // Reaming-specific parameters
    int m_numFlutes = 6;                  // Typical: 4-8 flutes
    double m_reamDiameter = 0.010;        // Reamer diameter (m)
    double m_maxRadialDOC = 0.0005;       // max DOC = 0.5mm (hard-clamped)
    double m_radialStock = 0.0002;        // Stock to remove per side (m)
    double m_h7ToleranceMm = 0.015;       // h7 tolerance band (mm)
    double m_pilotHoleDiameter = 0.0098;  // Pre-existing hole diameter (m)
    
    // Flute data (reuses Flute struct from MillingStrategy)
    std::vector<Flute> m_flutes;
    
    // State
    CuttingConditions m_conditions;
    double m_toolRotationAngle = 0;
    double m_currentDepth = 0;
    
    // Quality metrics
    mutable double m_holeRoundnessUm = 0;    // Computed roundness (µm)
    mutable double m_h7OutputMm = 0;          // Computed h7 tolerance (mm)
    
    // Material
    double m_specificCuttingForce = 2000e6; // Pa
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // Anchored Physics: Virtual Spindle Transform
    Vec3 m_spindlePosition;
    Vec3 m_spindleVelocity;
    Vec3 m_prevSpindlePosition;
    
    // Cached
    mutable double m_cachedTorque = 0;
    mutable double m_cachedThrust = 0;
    
    // Helpers
    void initializeFlutes();
    void updateFluteEngagement();
    double computeChipLoadAtAngle(double angle) const;
    Vec3 computeFluteForce(const Flute& flute) const;
    double computeHoleRoundness() const;
};

} // namespace edgepredict
