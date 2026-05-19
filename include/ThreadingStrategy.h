#pragma once
/**
 * @file ThreadingStrategy.h
 * @brief Threading operation strategy — helical multi-pass thread cutting
 * 
 * Threading is fundamentally different from all other strategies:
 *   - Helical toolpath synchronized with spindle rotation
 *   - Multi-pass with incrementing radial depth per pass
 *   - V-shaped chip geometry (60° metric, 55° BSP)
 *   - Insert profile matches thread form
 *   - Spring passes at end (zero infeed) for final tolerance
 *
 * Supports: ISO metric (M), UN/UNC/UNF, BSP/BSPP
 */

#include "IMachiningStrategy.h"

namespace edgepredict {

/**
 * @brief Thread form standards
 */
enum class ThreadForm {
    ISO_METRIC,     // 60° form angle (M threads)
    UN_UNIFIED,     // 60° form angle (UNC/UNF)
    BSP_PARALLEL,   // 55° form angle (BSPP)
    BSP_TAPER,      // 55° form angle (BSPT)
    TRAPEZOIDAL,    // 30° form angle (Tr threads)
    ACME            // 29° form angle
};

/**
 * @brief Threading method
 */
enum class ThreadMethod {
    SINGLE_POINT,     // External/internal single-point threading
    THREAD_MILLING,   // Helical interpolation with rotating cutter
    TAPPING,          // Tap threading (rigid or floating)
    CHASING           // Multi-tooth die head
};

/**
 * @brief Infeed method for multi-pass threading
 */
enum class InfeedMethod {
    RADIAL,           // Straight-in (radial infeed)
    FLANK,            // 29°/30° flank infeed (ISO recommended)
    MODIFIED_FLANK,   // Alternating flank (best chip control)
    INCREMENTAL       // Constant area per pass
};

/**
 * @brief Threading operation strategy
 */
class ThreadingStrategy : public IMachiningStrategy {
public:
    ThreadingStrategy();
    ~ThreadingStrategy() override = default;
    
    // IMachiningStrategy interface
    std::string getName() const override { return "ThreadingStrategy"; }
    MachiningType getType() const override { return MachiningType::THREADING; }
    
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
    
    // Threading-specific methods
    void setThreadForm(ThreadForm form) { m_threadForm = form; }
    void setInfeedMethod(InfeedMethod method) { m_infeedMethod = method; }
    
    /** @brief Get current pass number (1-based) */
    int getCurrentPass() const { return m_currentPass; }
    
    /** @brief Get total pass count */
    int getTotalPasses() const { return m_numPasses; }
    
    /** @brief Get current radial depth for this pass */
    double getCurrentPassDepth() const;
    
    /** @brief Get thread depth achieved so far */
    double getThreadDepthAchieved() const { return m_threadDepthAchieved; }

private:
    // Solver connections
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ContactSolver* m_contact = nullptr;
    CFDSolverGPU* m_cfd = nullptr;
    
    // Configuration
    ToolGeometry m_geometry;
    ThreadForm m_threadForm = ThreadForm::ISO_METRIC;
    ThreadMethod m_threadMethod = ThreadMethod::SINGLE_POINT;
    InfeedMethod m_infeedMethod = InfeedMethod::MODIFIED_FLANK;
    
    // Thread geometry (from config)
    double m_threadPitchMm = 1.5;            // Thread pitch (mm)
    double m_threadFormAngleDeg = 60.0;      // Form angle (degrees)
    double m_majorDiameterMm = 10.0;         // Major diameter (mm)
    double m_threadLengthMm = 20.0;          // Thread engagement length (mm)
    
    // Multi-pass parameters
    int m_numPasses = 6;                     // Total passes (roughing + spring)
    int m_numSpringPasses = 1;               // Spring passes at zero infeed
    int m_currentPass = 0;                   // Current pass (0 = not started)
    double m_totalThreadDepth = 0;           // Full thread depth (computed, m)
    double m_threadDepthAchieved = 0;        // Accumulated depth so far (m)
    std::vector<double> m_passDepths;        // Radial depth per pass (m)
    
    // State
    CuttingConditions m_conditions;
    double m_toolRotationAngle = 0;
    double m_helicalPhase = 0;               // Accumulated helical phase (radians)
    double m_axialPositionInPass = 0;        // Where along the thread length we are (m)
    bool m_passInProgress = false;
    
    // Material
    double m_specificCuttingForce = 2000e6;  // Pa
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // Anchored Physics: Virtual Spindle Transform
    Vec3 m_spindlePosition;
    Vec3 m_spindleVelocity;
    Vec3 m_prevSpindlePosition;
    
    // Cached forces
    mutable double m_cachedTangentialForce = 0;
    mutable double m_cachedRadialForce = 0;
    mutable double m_cachedTorque = 0;
    
    // Helpers
    void computePassDepths();
    double computeVChipArea() const;
    Vec3 computeHelicalPosition(double phase) const;
    double getFormHalfAngle() const;
    double computeMinorDiameter() const;
};

} // namespace edgepredict
