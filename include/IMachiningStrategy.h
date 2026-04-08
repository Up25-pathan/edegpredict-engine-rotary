#pragma once
/**
 * @file IMachiningStrategy.h
 * @brief Clean interface for machining operation strategies
 * 
 * Design principles:
 * 1. Strategy pattern - different implementations for milling/drilling/reaming/threading/boring
 * 2. No circular dependencies - uses forward declarations
 * 3. Composition over inheritance - strategies compose with solvers
 * 4. Single responsibility - each strategy handles one operation type
 */

#include "Types.h"
#include "Config.h"
#include <memory>
#include <string>
#include <vector>

namespace edgepredict {

// Forward declarations - no circular dependencies!
class SPHSolver;
class FEMSolver;
class ContactSolver;
class CFDSolverGPU;

/**
 * @brief Tool geometry parameters for different operations
 */
struct ToolGeometry {
    // Common parameters
    double rakeAngle = 0;           // degrees (positive = relief)
    double clearanceAngle = 7;      // degrees
    double noseRadius = 0.4e-3;     // m (0.4mm for inserts)
    

    
    // Milling specific
    int numFlutes = 4;              // Number of cutting edges
    double helixAngle = 30;         // degrees
    double fluteLength = 0.02;      // m
    double cutterDiameter = 0.01;   // m (10mm)
    
    // Drilling specific
    double pointAngle = 118;        // degrees (standard twist drill)
    double chiselEdgeRatio = 0.18;  // chisel/diameter ratio
    double webThickness = 0;        // m
    

};

/**
 * @brief Cutting conditions at current time
 */
struct CuttingConditions {
    double cuttingSpeed = 0;        // m/s
    double feedRate = 0;            // m/min or mm/tooth
    double depthOfCut = 0;          // m
    double widthOfCut = 0;          // m (for milling)
    
    double spindleSpeed = 0;        // rad/s
    double toolRotationAngle = 0;   // current rotation (rad)
    
    // === ANCHORED PHYSICS: Virtual Spindle Transform ===
    Vec3 spindlePosition;           // Absolute position from G-Code (meters)
    Vec3 spindleVelocity;           // Velocity vector derived from G-Code path
    bool isGCodeDriven = false;     // True when G-Code provides absolute positions

    Vec3 toolPosition;
    Vec3 toolVelocity;
    Vec3 feedDirection;
    Vec3 cuttingDirection;
};

/**
 * @brief Cutting edge representation
 */
struct CuttingEdge {
    Vec3 startPoint;
    Vec3 endPoint;
    Vec3 normal;                    // Pointing into workpiece
    Vec3 rakeDirection;             // Chip flow direction
    double localRakeAngle;          // May vary along edge
    double engagement;              // 0-1, how much edge is cutting
    bool isActive = false;          // Currently in cut?
};

/**
 * @brief Process outputs from machining
 */
struct MachiningOutput {
    // Forces
    Vec3 cuttingForce;              // Fc - tangential
    Vec3 thrustForce;               // Ft - radial
    Vec3 feedForce;                 // Ff - axial
    double torque = 0;              // N·m
    double power = 0;               // W
    
    // Chip formation
    double chipThickness = 0;       // m
    double chipWidth = 0;           // m
    double chipVelocity = 0;        // m/s
    double materialRemovalRate = 0; // mm³/s
    
    // Quality metrics
    double theoreticalRoughness = 0; // µm Ra
    double actualRoughness = 0;      // µm Ra (from simulation)
    
    // Tool state
    double maxToolStress = 0;       // Pa
    double maxToolTemperature = 0;  // °C
    double wearRate = 0;            // m/s
    
    // Operation-specific quality metrics
    double h7ToleranceOutputMm = 0; // Reaming: computed h7 tolerance deviation (mm)
    double barDeflectionUm = 0;     // Boring: boring bar deflection (µm)
};

/**
 * @brief Interface for machining operation strategies
 * 
 * Each strategy implements operation-specific:
 * - Tool geometry interpretation
 * - Kinematics (tool vs workpiece rotation)
 * - Cutting edge engagement calculation
 * - Force/torque prediction
 */
class IMachiningStrategy {
public:
    virtual ~IMachiningStrategy() = default;
    
    /**
     * @brief Get strategy name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get machining type
     */
    virtual MachiningType getType() const = 0;
    
    /**
     * @brief Initialize strategy with configuration
     */
    virtual bool initialize(const Config& config) = 0;
    
    /**
     * @brief Check if strategy is already initialized
     */
    virtual bool isInitialized() const = 0;
    
    /**
     * @brief Set tool geometry
     */
    virtual void setToolGeometry(const ToolGeometry& geometry) = 0;
    
    /**
     * @brief Connect physics solvers (dependency injection)
     */
    virtual void connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                 ContactSolver* contact, CFDSolverGPU* cfd) = 0;
    
    /**
     * @brief Update cutting conditions from G-code/machine state
     */
    virtual void updateConditions(const MachineState& state, double dt) = 0;
    
    /**
     * @brief Get active cutting edges for current conditions
     */
    virtual std::vector<CuttingEdge> getActiveCuttingEdges() const = 0;
    
    /**
     * @brief Compute cutting forces for current state
     */
    virtual MachiningOutput computeOutput() const = 0;
    
    /**
     * @brief Apply operation-specific transformations to tool/workpiece
     * 
     * For rotary operations: rotate tool
     */
    virtual void applyKinematics(double dt) = 0;
    
    /**
     * @brief Reset to initial state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Apply adaptive control multipliers (from OptimizationManager)
     */
    virtual void applyAdaptiveControl(double feedMultiplier, double speedMultiplier) = 0;

    // === ANCHORED PHYSICS: Virtual Spindle Interface ===
    /**
     * @brief Get the virtual spindle absolute position (from G-Code)
     */
    virtual Vec3 getSpindlePosition() const = 0;
    
    /**
     * @brief Get the virtual spindle absolute velocity
     */
    virtual Vec3 getSpindleVelocity() const = 0;
};

/**
 * @brief Factory for creating machining strategies
 */
class MachiningStrategyFactory {
public:
    /**
     * @brief Create strategy for given machining type
     */
    static std::unique_ptr<IMachiningStrategy> create(MachiningType type);
    
    /**
     * @brief Create strategy from config
     */
    static std::unique_ptr<IMachiningStrategy> createFromConfig(const Config& config);
};

} // namespace edgepredict
