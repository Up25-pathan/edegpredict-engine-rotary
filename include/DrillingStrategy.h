#pragma once
/**
 * @file DrillingStrategy.h
 * @brief Drilling operation strategy
 * 
 * Handles:
 * - Twist drill geometry (point angle, chisel edge)
 * - Indexable drill inserts
 * - Axial thrust + torque calculation
 * - Web thickness effects on chisel edge force
 */

#include "IMachiningStrategy.h"

namespace edgepredict {

/**
 * @brief Drill types
 */
enum class DrillType {
    TWIST_DRILL,        // Standard twist drill
    INDEXABLE,          // With inserts
    SPADE,              // Flat blade
    GUN_DRILL,          // Deep hole
    CENTER_DRILL        // For centering
};

/**
 * @brief Drilling strategy
 */
class DrillingStrategy : public IMachiningStrategy {
public:
    DrillingStrategy();
    ~DrillingStrategy() override = default;
    
    // IMachiningStrategy interface
    std::string getName() const override { return "DrillingStrategy"; }
    MachiningType getType() const override { return MachiningType::DRILLING; }
    
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
    
    // === ANCHORED PHYSICS: Virtual Spindle Interface ===
    Vec3 getSpindlePosition() const override { return m_spindlePosition; }
    Vec3 getSpindleVelocity() const override { return m_spindleVelocity; }
    
    // Drilling-specific methods
    
    /**
     * @brief Set drill type
     */
    void setDrillType(DrillType type) { m_drillType = type; }
    
    /**
     * @brief Get thrust force (axial)
     */
    double getThrustForce() const;
    
    /**
     * @brief Get torque
     */
    double getTorque() const;
    
    /**
     * @brief Get power required
     */
    double getPower() const;

private:
    // Solver connections
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ContactSolver* m_contact = nullptr;
    CFDSolverGPU* m_cfd = nullptr;
    
    // Configuration
    ToolGeometry m_geometry;
    DrillType m_drillType = DrillType::TWIST_DRILL;
    
    // State
    CuttingConditions m_conditions;
    double m_drillRotationAngle = 0;
    double m_currentDepth = 0;
    
    // Material
    double m_specificCuttingForce = 2000e6;
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // === ANCHORED PHYSICS: Virtual Spindle Transform ===
    Vec3 m_spindlePosition;     // Absolute position from G-Code
    Vec3 m_spindleVelocity;     // Computed velocity
    Vec3 m_prevSpindlePosition;
    
    // Cached forces
    mutable double m_cachedThrust = 0;
    mutable double m_cachedTorque = 0;
    
    // Helpers
    double computeChiselEdgeForce() const;
    double computeCuttingLipForce() const;
};

} // namespace edgepredict
