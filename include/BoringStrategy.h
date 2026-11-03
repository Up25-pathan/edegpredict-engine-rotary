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
#include "EdgeSubgridModel.h"
#include "BUEModel.h"
#include "ChatterDynamics.h"

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
    void connectSolvers(MPMSolver* sph, FEMSolver* fem, 
                        ContactSolver* contact, CFDSolverGPU* cfd) override;
    void updateConditions(const MachineState& state, double dt) override;
    std::vector<CuttingEdge> getActiveCuttingEdges() const override;
    MachiningOutput computeOutput() const override;
    void applyKinematics(double dt) override;
    bool isInitialized() const override { return m_initialized; }
    void reset() override;
    void applyAdaptiveControl(double feedMultiplier, double speedMultiplier) override;
    
    // Edge Subgrid Model Interface
    void setEdgeSubgridModel(EdgeSubgridModel* model) override { m_edgeSubgrid = model; }
    const EdgeSubgridModel* getEdgeSubgridModel() const override { return m_edgeSubgrid; }

    // BUE Model Interface
    void setBUEModel(BUEModel* model) override { m_bue = model; }
    const BUEModel* getBUEModel() const override { return m_bue; }

    // Chatter Dynamics Interface
    void setChatterDynamics(ChatterDynamics* chatter) override { m_chatter = chatter; }

    // Correction methods
    void applyEdgeSubgridCorrections() override;
    void applyBUECorrections() override;
    void applyChatterModulation(double dt) override;

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
    MPMSolver* m_sph = nullptr;
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

    // Material properties for edge subgrid (from config, replaces hardcoded Ti-6Al-4V)
    double m_workYieldPa = 880e6;
    double m_workYoungsPa = 113.8e9;
    double m_toolYoungsPa = 600e9;
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    double m_configRPM = 1000.0;  // Fallback RPM from config (G-Code lacks S-word)
    
    // Anchored Physics: Virtual Spindle Transform
    Vec3 m_spindlePosition;
    Vec3 m_spindleVelocity;
    Vec3 m_prevSpindlePosition;
    
    // Edge Subgrid, BUE, Chatter models (non-owning)
    EdgeSubgridModel* m_edgeSubgrid = nullptr;
    BUEModel* m_bue = nullptr;
    ChatterDynamics* m_chatter = nullptr;

    // Deflection warning throttle (replaces static local in const method)
    mutable int m_deflectionWarnCount = 0;
    
    // Cached forces
    mutable double m_cachedRadialForce = 0;
    mutable double m_cachedTangentialForce = 0;
    mutable double m_cachedAxialForce = 0;
    mutable double m_cachedTorque = 0;
    mutable double m_cachedBarDeflection = 0;  // meters
    
    // ── P1 FIX: Dynamic Boring Bar Vibration Model ──────────────────────────
    // State-space model for boring bar vibration (cantilever beam, first mode)
    //  m*ẍ + c*ẋ + k*x = F_radial(t)
    // where F_radial(t) is the time-varying radial cutting force
    // Natural frequency: f_n = (1/2π) * sqrt(k/m_eff) = (1.875²/2π) * sqrt(EI/ρAL⁴)
    // Dynamic amplification at forcing frequency f: A = 1/sqrt((1-r²)² + (2ζr)²)
    // where r = f/f_n, ζ = damping ratio (~0.01-0.05 for carbide/steel bars)
    // Regenerative chatter: F(t) depends on x(t - T) where T = 60/RPM (tooth period)
    double m_barNaturalFrequency = 0;     // Hz, first bending mode
    double m_barDampingRatio = 0.02;      // Structural damping ratio (2% typical)
    double m_barEffectiveMass = 0;        // Modal effective mass (kg)
    double m_barStiffness = 0;            // Modal stiffness (N/m)
    double m_barVibrationDisplacement = 0; // Current vibration displacement (m)
    double m_barVibrationVelocity = 0;    // Current vibration velocity (m/s)
    double m_prevRadialForce = 0;         // Force at previous step for regenerative effect
    
    // Helpers
    double computeRadialForce() const;
    double computeTangentialForce() const;
    double computeAxialForce() const;
    double computeBarDeflection(double radialForce) const;
    void updateBarDynamics(double dt, double radialForce);  // P1: dynamic deflection update
};

} // namespace edgepredict
