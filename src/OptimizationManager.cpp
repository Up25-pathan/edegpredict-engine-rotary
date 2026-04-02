/**
 * @file OptimizationManager.cpp
 * @brief Adaptive optimization and PID control implementation
 */

#include "OptimizationManager.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace edgepredict {

// ============================================================================
// PIDController Implementation
// ============================================================================

PIDController::PIDController(const PIDParams& params) : m_params(params) {}

double PIDController::compute(double setpoint, double measured, double dt) {
    if (dt <= 0) return 0;
    
    double error = setpoint - measured;
    
    // Proportional term
    double P = m_params.Kp * error;
    
    // Integral term (with anti-windup)
    m_integral += error * dt;
    m_integral = std::clamp(m_integral, -m_params.integralMax, m_params.integralMax);
    double I = m_params.Ki * m_integral;
    
    // Derivative term
    double D = 0;
    if (!m_firstCall) {
        D = m_params.Kd * (error - m_prevError) / dt;
    }
    m_prevError = error;
    m_firstCall = false;
    
    // Total output
    double output = P + I + D;
    return std::clamp(output, m_params.outputMin, m_params.outputMax);
}

void PIDController::reset() {
    m_integral = 0;
    m_prevError = 0;
    m_firstCall = true;
}

// ============================================================================
// OptimizationManager Implementation
// ============================================================================

OptimizationManager::OptimizationManager() {
    // Initialize PID controllers with defaults
    PIDParams feedParams;
    feedParams.Kp = 0.5;
    feedParams.Ki = 0.05;
    feedParams.Kd = 0.01;
    feedParams.outputMin = 0.5;   // Min 50% of feed
    feedParams.outputMax = 1.5;   // Max 150% of feed
    m_feedPID.setParams(feedParams);
    
    PIDParams speedParams;
    speedParams.Kp = 0.3;
    speedParams.Ki = 0.02;
    speedParams.Kd = 0.005;
    speedParams.outputMin = 0.7;  // Min 70% of speed
    speedParams.outputMax = 1.2;  // Max 120% of speed
    m_speedPID.setParams(speedParams);
    
    PIDParams depthParams;
    depthParams.Kp = 0.2;
    depthParams.Ki = 0.01;
    depthParams.Kd = 0.002;
    depthParams.outputMin = 0.5;
    depthParams.outputMax = 1.2;
    m_depthPID.setParams(depthParams);
}

void OptimizationManager::initialize(const Config& config) {
    const auto& optParams = config.getOptimization();
    
    m_enabled = optParams.enabled;
    
    // Set targets from config
    m_targets.maxToolStress = optParams.maxStress;
    m_targets.maxToolTemperature = optParams.maxTemperature;
    m_targets.maxTotalWear = optParams.maxWear;
    
    // PID from config
    PIDParams feedParams;
    feedParams.Kp = optParams.feedPID_Kp;
    feedParams.Ki = optParams.feedPID_Ki;
    feedParams.Kd = optParams.feedPID_Kd;
    feedParams.outputMin = 0.5;
    feedParams.outputMax = 1.5;
    m_feedPID.setParams(feedParams);
    
    std::cout << "[OptimizationManager] Initialized, enabled=" << m_enabled << std::endl;
}

AdaptiveCommand OptimizationManager::update(const ProcessState& state, double dt) {
    AdaptiveCommand cmd;
    
    // Add to history
    m_history.push_back(state);
    if (m_history.size() > MAX_HISTORY) {
        m_history.pop_front();
    }
    
    // Update statistics
    m_totalTime += dt;
    m_totalEnergy += state.power * dt;
    m_totalMaterialRemoved += state.materialRemovalRate * dt / 60.0;  // cm³/min to cm³/s
    
    if (!m_enabled) {
        return cmd;  // No adjustment
    }
    
    // === Safety Checks First ===
    checkLimits(state, cmd);
    if (cmd.emergencyStop) {
        return cmd;
    }
    
    // === Feed Rate Control (based on cutting force) ===
    // Reduce feed if force is too high, increase if too low
    double forceNormalized = state.cuttingForce / m_targets.targetCuttingForce;
    
    // We want forceNormalized ≈ 1.0
    // If force > target, reduce feed (output < 1)
    // If force < target, increase feed (output > 1)
    double feedAdjust = m_feedPID.compute(1.0, forceNormalized, dt);
    cmd.feedRateMultiplier = feedAdjust;
    
    // === Spindle Speed Control (based on temperature) ===
    double tempNormalized = state.toolTemperature / m_targets.targetToolTemperature;
    
    // If temp > target, reduce speed (less heat generation)
    double speedAdjust = m_speedPID.compute(1.0, tempNormalized, dt);
    cmd.spindleSpeedMultiplier = speedAdjust;
    
    // === Depth Control (based on stress) ===
    double stressNormalized = state.toolStress / m_targets.targetToolStress;
    double depthAdjust = m_depthPID.compute(1.0, stressNormalized, dt);
    cmd.depthMultiplier = depthAdjust;
    
    // === Coolant Control ===
    if (state.toolTemperature > m_targets.targetToolTemperature * 0.8) {
        cmd.coolantOn = true;
        cmd.coolantFlowMultiplier = 1.0 + (state.toolTemperature - m_targets.targetToolTemperature * 0.8) / 
                                          (m_targets.targetToolTemperature * 0.4);
        cmd.coolantFlowMultiplier = std::clamp(cmd.coolantFlowMultiplier, 1.0, 2.0);
    }
    
    // === Chatter Detection ===
    double chatterRisk = estimateChatterRisk(state);
    if (chatterRisk > 0.7) {
        // Reduce speed slightly to avoid chatter
        cmd.spindleSpeedMultiplier *= 0.9;
        
        if (m_callback) {
            m_callback("chatter_warning", chatterRisk);
        }
    }
    
    // === Trend Correction ===
    applyTrendCorrection(cmd);
    
    return cmd;
}

void OptimizationManager::checkLimits(const ProcessState& state, AdaptiveCommand& cmd) {
    // Emergency stop conditions
    if (state.toolStress > m_targets.maxToolStress * 1.2) {
        cmd.emergencyStop = true;
        cmd.reason = "Tool stress exceeded 120% of limit";
        return;
    }
    
    if (state.toolTemperature > m_targets.maxToolTemperature) {
        cmd.emergencyStop = true;
        cmd.reason = "Tool temperature exceeded limit";
        return;
    }
    
    if (state.totalWear > m_targets.maxTotalWear) {
        cmd.emergencyStop = true;
        cmd.reason = "Tool wear exceeded limit";
        return;
    }
    
    if (state.cuttingForce > m_targets.maxCuttingForce * 1.5) {
        cmd.emergencyStop = true;
        cmd.reason = "Cutting force exceeded 150% of limit";
        return;
    }
    
    // Warning levels - reduce aggressively
    if (state.toolStress > m_targets.maxToolStress * 0.9) {
        cmd.feedRateMultiplier *= 0.8;
        cmd.depthMultiplier *= 0.9;
        
        if (m_callback) {
            m_callback("stress_warning", state.toolStress);
        }
    }
    
    if (state.toolTemperature > m_targets.maxToolTemperature * 0.85) {
        cmd.spindleSpeedMultiplier *= 0.9;
        cmd.coolantFlowMultiplier = 1.5;
        
        if (m_callback) {
            m_callback("temperature_warning", state.toolTemperature);
        }
    }
    
    if (state.wearRate > m_targets.maxWearRate * 0.8) {
        cmd.feedRateMultiplier *= 0.85;
        
        if (m_callback) {
            m_callback("wear_warning", state.wearRate);
        }
    }
}

void OptimizationManager::applyTrendCorrection(AdaptiveCommand& cmd) {
    if (m_history.size() < 10) return;
    
    // Calculate temperature trend
    double tempSum = 0, tempTimeSum = 0;
    double t = 0;
    for (const auto& s : m_history) {
        tempSum += s.toolTemperature;
        tempTimeSum += s.toolTemperature * t;
        t += 1;
    }
    
    double n = static_cast<double>(m_history.size());
    double tSum = n * (n - 1) / 2;
    double tSqSum = n * (n - 1) * (2 * n - 1) / 6;
    
    double tempTrend = (n * tempTimeSum - tSum * tempSum) / (n * tSqSum - tSum * tSum);
    
    // If temperature is rising fast, preemptively reduce speed
    if (tempTrend > 1.0) {  // Rising more than 1°C per sample
        cmd.spindleSpeedMultiplier *= 0.95;
        cmd.coolantFlowMultiplier *= 1.1;
    }
    
    // Similar for stress
    double stressSum = 0, stressTimeSum = 0;
    t = 0;
    for (const auto& s : m_history) {
        stressSum += s.toolStress;
        stressTimeSum += s.toolStress * t;
        t += 1;
    }
    
    double stressTrend = (n * stressTimeSum - tSum * stressSum) / (n * tSqSum - tSum * tSum);
    
    if (stressTrend > 1e7) {  // Rising fast (10 MPa/sample)
        cmd.feedRateMultiplier *= 0.95;
        cmd.depthMultiplier *= 0.95;
    }
}

double OptimizationManager::estimateChatterRisk(const ProcessState& state) {
    if (m_history.size() < 20) return 0;
    
    // Calculate force variance
    std::vector<double> recentForces;
    for (auto it = m_history.rbegin(); it != m_history.rend() && recentForces.size() < 20; ++it) {
        recentForces.push_back(it->cuttingForce);
    }
    
    double mean = std::accumulate(recentForces.begin(), recentForces.end(), 0.0) / recentForces.size();
    
    double variance = 0;
    for (double f : recentForces) {
        variance += (f - mean) * (f - mean);
    }
    variance /= recentForces.size();
    
    double stdDev = std::sqrt(variance);
    
    // High variance relative to mean indicates chatter
    double coeffVar = (mean > 0) ? stdDev / mean : 0;
    
    // Normalize to 0-1 risk
    return std::clamp(coeffVar * 5.0, 0.0, 1.0);
}

double OptimizationManager::getAverageMRR() const {
    if (m_totalTime <= 0) return 0;
    return m_totalMaterialRemoved / (m_totalTime / 60.0);  // cm³/min
}

double OptimizationManager::getEstimatedToolLifeRemaining() const {
    if (m_history.empty()) return 1000;  // Unknown, return large value
    
    // Get current wear rate
    double currentWear = m_history.back().totalWear;
    double currentWearRate = m_history.back().wearRate;
    
    if (currentWearRate <= 0) return 1000;
    
    double remainingWear = m_targets.maxTotalWear - currentWear;
    if (remainingWear <= 0) return 0;
    
    return remainingWear / currentWearRate;  // seconds
}

void OptimizationManager::reset() {
    m_feedPID.reset();
    m_speedPID.reset();
    m_depthPID.reset();
    m_history.clear();
    m_totalEnergy = 0;
    m_totalMaterialRemoved = 0;
    m_totalTime = 0;
}

// ============================================================================
// EnergyMonitor Implementation
// ============================================================================

void EnergyMonitor::update(double power, double dt) {
    m_totalEnergy += power * dt;
    m_totalTime += dt;
    m_sampleCount++;
    
    if (power > m_peakPower) {
        m_peakPower = power;
    }
}

double EnergyMonitor::getAveragePower() const {
    if (m_totalTime <= 0) return 0;
    return (m_totalEnergy / m_totalTime) / 1000.0;  // Watts to kW
}

double EnergyMonitor::getCO2Emissions(double co2PerKWh) const {
    return getTotalEnergy() * co2PerKWh;
}

void EnergyMonitor::reset() {
    m_totalEnergy = 0;
    m_peakPower = 0;
    m_totalTime = 0;
    m_sampleCount = 0;
}

} // namespace edgepredict
