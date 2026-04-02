#pragma once
/**
 * @file BoringStrategy.h
 * @brief Boring operation strategy — single-point internal turning
 * 
 * Boring enlarges a pre-existing hole with a single-point insert on a
 * rotating boring bar.  Key differences vs drilling:
 *   - Cutting forces point RADIALLY outward from bore centerline
 *   - Tool deflection of the boring bar is the dominant quality limiter
 *     (long thin bar → cantilever bending)
 *   - Single cutting edge, not 2 lips + chisel
 *   - Axial feed is slow; radial depth defines material removal
 */

#include "IMachiningStrategy.h"

namespace edgepredict {

/**
 * @brief Boring bar / head types
 */
enum class BoringType {
    SINGLE_POINT,     // Standard single-point boring bar
    TWIN_INSERT,      // Two-insert boring head (balanced forces)
    MICRO_BORING,     // High-precision micro boring bar (< 6mm)
    LINE_BORING       // Multi-journal line boring
};

/**
 * @brief Boring operation strategy
 */
class BoringStrategy : public IMachiningStrategy {
public:
    BoringStrategy();
    ~BoringStrategy() override = default;
    
    // IMachiningStrategy interface
    std::string getName() const override { return "BoringStrategy"; }
    MachiningType getType() const override { return MachiningType::BORING; }
    
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
    
    // Boring-specific methods
    void setBoringType(BoringType type) { m_boringType = type; }
    
    /** @brief Get radial cutting force (dominant component) */
    double getRadialForce() const;
    
    /** @brief Get boring bar deflection (quality metric) */
    double getBarDeflection() const;
    
    /** @brief Get torque */
    double getTorque() const;

private:
    // Solver connections
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ContactSolver* m_contact = nullptr;
    CFDSolverGPU* m_cfd = nullptr;
    
    // Configuration
    ToolGeometry m_geometry;
    BoringType m_boringType = BoringType::SINGLE_POINT;
    
    // Boring bar geometry (from config)
    double m_barLength = 0.040;         // Boring bar overhang (m) — default 40mm
    double m_barDiameter = 0.010;       // Boring bar shank diameter (m) — default 10mm
    double m_barYoungsModulus = 600e9;   // Bar material E (Pa) — carbide
    double m_radialDepthOfCut = 0.001;  // Radial DOC into bore wall (m)
    double m_boreStartDiameter = 0.020; // Pre-existing hole diameter (m)
    double m_insertLeadAngle = 90.0;    // Lead angle κ (degrees) — 90° typical
    double m_insertNoseRadius = 0.4e-3; // Insert nose radius (m)
    
    // State
    CuttingConditions m_conditions;
    double m_toolRotationAngle = 0;
    double m_currentDepth = 0;          // Axial penetration depth (m)
    
    // Material
    double m_specificCuttingForce = 2000e6; // Pa
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // Anchored Physics: Virtual Spindle Transform
    Vec3 m_spindlePosition;
    Vec3 m_spindleVelocity;
    Vec3 m_prevSpindlePosition;
    
    // Cached forces
    mutable double m_cachedRadialForce = 0;
    mutable double m_cachedTangentialForce = 0;
    mutable double m_cachedAxialForce = 0;
    mutable double m_cachedTorque = 0;
    mutable double m_cachedBarDeflection = 0;  // meters
    
    // Helpers
    double computeRadialForce() const;
    double computeTangentialForce() const;
    double computeAxialForce() const;
    double computeBarDeflection(double radialForce) const;
};

} // namespace edgepredict
