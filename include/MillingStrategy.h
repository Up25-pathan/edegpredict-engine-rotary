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
#include "EdgeSubgridModel.h"
#include "BUEModel.h"
#include "ChatterDynamics.h"
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
    void connectSolvers(MPMSolver* sph, FEMSolver* fem, 
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
    MPMSolver* m_sph = nullptr;
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

    // Fix 2: Material properties for edge subgrid (replaces hardcoded Ti-6Al-4V)
    double m_workYieldPa = 880e6;
    double m_workYoungsPa = 113.8e9;
    double m_toolYoungsPa = 600e9;
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // === ANCHORED PHYSICS: Virtual Spindle Transform ===
    Vec3 m_spindlePosition;     // Absolute position from G-Code
    Vec3 m_spindleVelocity;     // Computed velocity
    Vec3 m_prevSpindlePosition;
    
    // Edge Subgrid Model
    void setEdgeSubgridModel(EdgeSubgridModel* model) override { m_edgeSubgrid = model; }
    const EdgeSubgridModel* getEdgeSubgridModel() const override { return m_edgeSubgrid; }

    // BUE Model
    void setBUEModel(BUEModel* model) override { m_bue = model; }
    const BUEModel* getBUEModel() const override { return m_bue; }

    // Chatter Dynamics
    void setChatterDynamics(ChatterDynamics* chatter) override { m_chatter = chatter; }

    // Correction methods
    void applyEdgeSubgridCorrections() override;
    void applyBUECorrections() override;
    void applyChatterModulation(double dt) override;

    EdgeSubgridModel* m_edgeSubgrid = nullptr;
    BUEModel* m_bue = nullptr;
    ChatterDynamics* m_chatter = nullptr;

    // Helper methods
    void initializeFlutes();
    void updateFluteEngagement();
    double computeChipLoadAtAngle(double angle) const;
    double computeEngagementArc() const;
    Vec3 computeFluteForce(const Flute& flute) const;
};

} // namespace edgepredict
