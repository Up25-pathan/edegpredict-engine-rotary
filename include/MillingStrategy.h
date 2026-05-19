#pragma once
/**
 * @file MillingStrategy.h
 * @brief Milling operation strategy for end mills and face mills
 * 
 * Handles:
 * - Multi-flute engagement (4, 6, 8 flutes)
 * - Tool rotation + stationary/moving workpiece
 * - Helix angle effects
 * - Up/down milling (climb vs conventional)
 * - Entry/exit chip thickness variation
 */

#include "IMachiningStrategy.h"
#include <vector>

namespace edgepredict {

/**
 * @brief End mill types
 */
enum class EndMillType {
    FLAT_END,       // Flat bottom
    BALL_NOSE,      // Hemispherical end
    BULL_NOSE,      // Corner radius
    CHAMFER,        // V-shaped
    ROUGHING        // Serrated edges
};

/**
 * @brief Milling mode
 */
enum class MillingMode {
    CLIMB,          // Down milling (cutter direction = feed direction)
    CONVENTIONAL,   // Up milling (cutter opposite to feed)
    SLOT,           // Full slot engagement
    FACE            // Face milling
};

/**
 * @brief Individual flute on the cutter
 */
struct Flute {
    int index = 0;
    double angularPosition = 0;     // rad, position around tool
    double helixAngle = 30;         // degrees
    double rakeAngle = 10;          // degrees
    bool isEngaged = false;
    double engagementStart = 0;     // rad
    double engagementEnd = 0;       // rad
    double currentChipLoad = 0;     // m
};

/**
 * @brief Milling operation strategy
 */
class MillingStrategy : public IMachiningStrategy {
public:
    MillingStrategy();
    ~MillingStrategy() override = default;
    
    // IMachiningStrategy interface
    std::string getName() const override { return "MillingStrategy"; }
    MachiningType getType() const override { return MachiningType::MILLING; }
    
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
    
    // Milling-specific methods
    
    /**
     * @brief Set end mill type
     */
    void setEndMillType(EndMillType type) { m_endMillType = type; }
    
    /**
     * @brief Set milling mode
     */
    void setMillingMode(MillingMode mode) { m_millingMode = mode; }
    
    /**
     * @brief Get number of flutes currently engaged
     */
    int getEngagedFluteCount() const;
    
    /**
     * @brief Get instantaneous chip load for a flute
     */
    double getChipLoad(int fluteIndex) const;
    
    /**
     * @brief Get radial immersion (ae/D ratio)
     */
    double getRadialImmersion() const;
    
    /**
     * @brief Get engagement arc angle
     */
    double getEngagementAngle() const;

private:
    // Solver connections
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ContactSolver* m_contact = nullptr;
    CFDSolverGPU* m_cfd = nullptr;
    
    // Configuration
    ToolGeometry m_geometry;
    EndMillType m_endMillType = EndMillType::FLAT_END;
    MillingMode m_millingMode = MillingMode::CLIMB;
    
    // Flute data
    std::vector<Flute> m_flutes;
    
    // Current state
    CuttingConditions m_conditions;
    double m_toolRotationAngle = 0;
    
    // Material properties
    double m_specificCuttingForce = 2000e6;  // Pa
    double m_frictionCoeff = 0.4;
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // === ANCHORED PHYSICS: Virtual Spindle Transform ===
    Vec3 m_spindlePosition;     // Absolute position from G-Code
    Vec3 m_spindleVelocity;     // Computed velocity
    Vec3 m_prevSpindlePosition;
    
    // Helper methods
    void initializeFlutes();
    void updateFluteEngagement();
    double computeChipLoadAtAngle(double angle) const;
    double computeEngagementArc() const;
    Vec3 computeFluteForce(const Flute& flute) const;
};

} // namespace edgepredict
