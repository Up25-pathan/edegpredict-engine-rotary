#pragma once
/**
 * @file OptimizationManager.h
 * @brief Adaptive control and process optimization with PID control
 * 
 * Implements real-time optimization of:
 * - Feed rate to maintain constant chip load
 * - Spindle speed to avoid chatter
 * - Cutting depth adjustment
 */

#include "Types.h"
#include "Config.h"
#include <vector>
#include <deque>
#include <functional>

namespace edgepredict {

/**
 * @brief PID controller parameters
 */
struct PIDParams {
    double Kp = 1.0;    // Proportional gain
    double Ki = 0.1;    // Integral gain
    double Kd = 0.01;   // Derivative gain
    double outputMin = 0.0;
    double outputMax = 1.0;
    double integralMax = 10.0;  // Anti-windup
};

/**
 * @brief PID controller implementation
 */
class PIDController {
public:
    PIDController() = default;
    PIDController(const PIDParams& params);
    
    /**
     * @brief Compute control output
     * @param setpoint Desired value
     * @param measured Current measured value
     * @param dt Time step (s)
     * @return Control output
     */
    double compute(double setpoint, double measured, double dt);
    
    /**
     * @brief Reset controller state
     */
    void reset();
    
    /**
     * @brief Set parameters
     */
    void setParams(const PIDParams& params) { m_params = params; }
    
private:
    PIDParams m_params;
    double m_integral = 0;
    double m_prevError = 0;
    bool m_firstCall = true;
};

/**
 * @brief Optimization targets and limits
 */
struct OptimizationTargets {
    // Stress limits
    double maxToolStress = 2e9;         // Pa (2 GPa)
    double targetToolStress = 1e9;      // Pa (1 GPa) - desired operating point
    
    // Temperature limits
    double maxToolTemperature = 800.0;   // °C
    double targetToolTemperature = 400.0; // °C
    double maxWorkpieceTemp = 500.0;     // °C
    
    // Wear limits
    double maxWearRate = 1e-6;           // m/s
    double maxTotalWear = 0.3e-3;        // m (0.3mm)
    
    // Force limits
    double maxCuttingForce = 5000.0;     // N
    double targetCuttingForce = 2000.0;  // N
    
    // Quality targets
    double targetMRR = 10.0;             // Material removal rate (cm³/min)
    double targetSurfaceRoughness = 1.6; // Ra (μm)
    
    // Feed/speed limits
    double minFeedRate = 0.01;           // mm/rev
    double maxFeedRate = 0.5;            // mm/rev
    double minSpindleSpeed = 100;        // RPM
    double maxSpindleSpeed = 10000;      // RPM
};

/**
 * @brief Adaptive control command
 */
struct AdaptiveCommand {
    double feedRateMultiplier = 1.0;     // Multiply current feed rate by this
    double spindleSpeedMultiplier = 1.0; // Multiply current spindle speed by this
    double depthMultiplier = 1.0;        // Multiply cutting depth by this
    
    bool coolantOn = true;
    double coolantFlowMultiplier = 1.0;
    
    bool emergencyStop = false;
    std::string reason;
};

/**
 * @brief Process monitoring data
 */
struct ProcessState {
    double time = 0;
    double toolStress = 0;
    double toolTemperature = 25;
    double workpieceTemperature = 25;
    double cuttingForce = 0;
    double wearRate = 0;
    double totalWear = 0;
    double materialRemovalRate = 0;
    double power = 0;
    
    double feedRate = 0.1;
    double spindleSpeed = 1000;
    double cuttingDepth = 0.5;
};

/**
 * @brief Optimization and adaptive control manager
 */
class OptimizationManager {
public:
    OptimizationManager();
    ~OptimizationManager() = default;
    
    /**
     * @brief Initialize from configuration
     */
    void initialize(const Config& config);
    
    /**
     * @brief Update with current process state and get control command
     * @param state Current process measurements
     * @param dt Time step (s)
     * @return Adaptive control command
     */
    AdaptiveCommand update(const ProcessState& state, double dt);
    
    /**
     * @brief Set optimization targets
     */
    void setTargets(const OptimizationTargets& targets) { m_targets = targets; }
    
    /**
     * @brief Set PID parameters for feed control
     */
    void setFeedPID(const PIDParams& params) { m_feedPID.setParams(params); }
    
    /**
     * @brief Set PID parameters for speed control
     */
    void setSpeedPID(const PIDParams& params) { m_speedPID.setParams(params); }
    
    /**
     * @brief Enable/disable adaptive control
     */
    void setEnabled(bool enabled) { m_enabled = enabled; }
    
    /**
     * @brief Get optimization statistics
     */
    double getTotalEnergyConsumed() const { return m_totalEnergy; }
    double getTotalMaterialRemoved() const { return m_totalMaterialRemoved; }
    double getAverageMRR() const;
    double getEstimatedToolLifeRemaining() const;
    
    /**
     * @brief Reset state
     */
    void reset();
    
    /**
     * @brief Set callback for optimization events
     */
    using OptimizationCallback = std::function<void(const std::string& event, double value)>;
    void setCallback(OptimizationCallback callback) { m_callback = callback; }

private:
    OptimizationTargets m_targets;
    
    PIDController m_feedPID;      // Controls feed based on force
    PIDController m_speedPID;     // Controls speed based on temperature
    PIDController m_depthPID;     // Controls depth based on stress
    
    bool m_enabled = true;
    
    // History for trend analysis
    std::deque<ProcessState> m_history;
    static constexpr size_t MAX_HISTORY = 100;
    
    // Statistics
    double m_totalEnergy = 0;
    double m_totalMaterialRemoved = 0;
    double m_totalTime = 0;
    
    OptimizationCallback m_callback;
    
    // Internal methods
    void checkLimits(const ProcessState& state, AdaptiveCommand& cmd);
    void applyTrendCorrection(AdaptiveCommand& cmd);
    double estimateChatterRisk(const ProcessState& state);
};

/**
 * @brief Energy monitor for sustainability tracking
 */
class EnergyMonitor {
public:
    EnergyMonitor() = default;
    
    /**
     * @brief Update with current power consumption
     */
    void update(double power, double dt);
    
    /**
     * @brief Get total energy consumed (kWh)
     */
    double getTotalEnergy() const { return m_totalEnergy / 3600000.0; }
    
    /**
     * @brief Get peak power (kW)
     */
    double getPeakPower() const { return m_peakPower / 1000.0; }
    
    /**
     * @brief Get average power (kW)
     */
    double getAveragePower() const;
    
    /**
     * @brief Get CO2 emissions (kg)
     * @param co2PerKWh CO2 per kWh (default: 0.5 kg/kWh)
     */
    double getCO2Emissions(double co2PerKWh = 0.5) const;
    
    /**
     * @brief Reset statistics
     */
    void reset();

private:
    double m_totalEnergy = 0;       // Joules
    double m_peakPower = 0;         // Watts
    double m_totalTime = 0;         // seconds
    int m_sampleCount = 0;
};

} // namespace edgepredict
