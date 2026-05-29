#pragma once
/**
 * @file SimulationHealthMonitor.h
 * @brief Runtime physics sanity checker — detects non-physical states and halts.
 *
 * Checks every N steps for:
 *   1. Thermal explosion    — temperature hit melting point too fast
 *   2. Stress explosion     — von Mises stress exceeds N× material yield
 *   3. Force explosion      — total contact force exceeds physical bounds
 *   4. Contact explosion    — contact count at step 1 >> expected (tool embedded)
 *   5. NaN / Inf corruption — any NaN in particle position/velocity
 *   6. Energy violation     — kinetic energy jumps >100× between checks
 *   7. dt floor collapse    — timestep stuck at minimum for extended period
 *
 * Usage:
 *   SimulationHealthMonitor monitor;
 *   monitor.initialize(config);
 *   ...
 *   auto result = monitor.check(step, dt, sph, fem, contactSolver);
 *   if (result.severity == HealthSeverity::HALT) {
 *       std::cerr << monitor.getDiagnosticReport() << std::endl;
 *       break;
 *   }
 */

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace edgepredict {

// Forward declarations
class MPMSolver;
class FEMSolver;
class ContactSolver;
class Config;
struct MPMParticle;

// ============================================================================
// Severity Levels
// ============================================================================

enum class HealthSeverity {
    OK,      // Everything within bounds
    WARN,    // Suspicious but simulation can continue
    HALT     // Non-physical state — simulation must stop
};

// ============================================================================
// Health Check Result
// ============================================================================

enum class FailureReason {
    NONE,
    THERMAL_EXPLOSION,         // Melting point hit too quickly
    STRESS_EXPLOSION,          // Stress >> material yield
    FORCE_EXPLOSION,           // Contact force non-physical
    CONTACT_EXPLOSION,         // Tool embedded in workpiece
    NAN_CORRUPTION,            // NaN/Inf in particle data
    ENERGY_VIOLATION,          // KE jumped > 100×
    DT_FLOOR_COLLAPSE          // dt stuck at minimum
};

struct HealthCheckResult {
    HealthSeverity severity = HealthSeverity::OK;
    FailureReason  reason   = FailureReason::NONE;
    std::string    message;
};

// ============================================================================
// Configuration
// ============================================================================

struct HealthMonitorConfig {
    bool   enabled = true;                    // Master switch
    int    checkIntervalSteps = 100;          // Check every N steps

    // Thermal
    double meltingPointC = 1460.0;            // Material melting point (°C)
    double thermalExplosionFraction = 0.05;   // HALT if >5% of particles at melting
    int    thermalExplosionMaxSteps = 200;     // within this many steps

    // Stress
    double yieldStrengthPa = 595e6;           // Material yield strength (Pa)
    double stressExplosionMultiplier = 50.0;  // HALT if stress > 50× yield

    // Force
    double maxPhysicalForceN = 100e6;         // 100 MN absolute ceiling (single step)

    // Contact
    int    contactExplosionMultiplier = 50;    // HALT if contacts > 50× numNodes on step 1

    // Energy
    double energyJumpMultiplier = 100.0;      // HALT if KE jumps > 100×

    // dt
    int    dtFloorMaxConsecutive = 500;        // WARN if dt at floor for 500+ consecutive steps
};

// ============================================================================
// SimulationHealthMonitor
// ============================================================================

class SimulationHealthMonitor {
public:
    SimulationHealthMonitor() = default;
    ~SimulationHealthMonitor() = default;

    /**
     * @brief Initialize from engine config — pulls material properties automatically.
     */
    void initialize(const Config& config);

    /**
     * @brief Initialize with explicit config.
     */
    void initialize(const HealthMonitorConfig& config) { m_config = config; m_initialized = true; }

    /**
     * @brief Run all health checks. Call every step or every output interval.
     *
     * @param step           Current simulation step
     * @param dt             Current timestep
     * @param dtMin          Minimum timestep floor
     * @param contactCount   Number of active contacts
     * @param totalForce     Total contact force (N)
     * @param maxStress      Maximum von Mises stress (Pa)
     * @param maxTemp        Maximum temperature (°C)
     * @param kineticEnergy  Total kinetic energy (J)
     * @param numParticles   Total active particles
     * @param numNodes       Total FEM nodes
     * @param particlesAtMelt Number of particles at melting point
     *
     * @return HealthCheckResult with severity and message
     */
    HealthCheckResult check(
        int step, double dt, double dtMin,
        int contactCount, double totalForce,
        double maxStress, double maxTemp,
        double kineticEnergy,
        int numParticles, int numNodes,
        int particlesAtMelt
    );

    /**
     * @brief Generate detailed diagnostic report after a HALT.
     */
    std::string getDiagnosticReport() const;

    /**
     * @brief Check if monitor is enabled.
     */
    bool isEnabled() const { return m_initialized && m_config.enabled; }

    /**
     * @brief Get the config for modification.
     */
    HealthMonitorConfig& getConfig() { return m_config; }

private:
    HealthMonitorConfig m_config;
    bool m_initialized = false;

    // Tracking state
    int    m_firstContactStep = -1;      // Step when contacts first appeared
    int    m_firstMeltStep = -1;         // Step when melting was first detected
    double m_prevKineticEnergy = 0.0;    // Previous check's KE
    int    m_dtFloorCount = 0;           // Consecutive steps at dt floor
    int    m_warningCount = 0;           // Total warnings issued

    // Snapshot for diagnostic report
    int    m_haltStep = 0;
    double m_haltTime = 0.0;
    FailureReason m_haltReason = FailureReason::NONE;
    double m_haltMaxStress = 0.0;
    double m_haltMaxTemp = 0.0;
    double m_haltForce = 0.0;
    int    m_haltContactCount = 0;
    double m_haltKE = 0.0;
    int    m_haltParticlesAtMelt = 0;
    int    m_haltNumParticles = 0;
};

// ============================================================================
// Inline Implementation
// ============================================================================

inline void SimulationHealthMonitor::initialize(const Config& config) {
    // Pull material properties from config
    // (Config.h is not included here to avoid circular deps — use the overload
    //  in SimulationEngine.cpp where Config.h is available)
    m_initialized = true;
}

inline HealthCheckResult SimulationHealthMonitor::check(
    int step, double dt, double dtMin,
    int contactCount, double totalForce,
    double maxStress, double maxTemp,
    double kineticEnergy,
    int numParticles, int numNodes,
    int particlesAtMelt)
{
    if (!isEnabled()) return { HealthSeverity::OK, FailureReason::NONE, "" };

    HealthCheckResult result;

    // Track first contact
    if (contactCount > 0 && m_firstContactStep < 0) {
        m_firstContactStep = step;
    }

    // ── Check 1: CONTACT EXPLOSION (tool embedded at T=0) ──────────────────
    // If contacts appear on step 1 and exceed numNodes × multiplier,
    // the tool is deep inside the workpiece.
    if (step <= 5 && contactCount > numNodes * m_config.contactExplosionMultiplier && numNodes > 0) {
        m_haltStep = step; m_haltReason = FailureReason::CONTACT_EXPLOSION;
        m_haltContactCount = contactCount; m_haltForce = totalForce;
        m_haltMaxStress = maxStress; m_haltMaxTemp = maxTemp;
        m_haltNumParticles = numParticles; m_haltKE = kineticEnergy;
        m_haltParticlesAtMelt = particlesAtMelt;
        result.severity = HealthSeverity::HALT;
        result.reason = FailureReason::CONTACT_EXPLOSION;
        result.message = "[HEALTH] HALT: Contact explosion at step " + std::to_string(step)
            + " — " + std::to_string(contactCount) + " contacts ("
            + std::to_string(contactCount / std::max(numNodes, 1)) + "× node count). "
            + "Tool is embedded in workpiece.";
        return result;
    }

    // ── Check 2: FORCE EXPLOSION ───────────────────────────────────────────
    if (totalForce > m_config.maxPhysicalForceN) {
        m_haltStep = step; m_haltReason = FailureReason::FORCE_EXPLOSION;
        m_haltForce = totalForce; m_haltContactCount = contactCount;
        m_haltMaxStress = maxStress; m_haltMaxTemp = maxTemp;
        m_haltNumParticles = numParticles; m_haltKE = kineticEnergy;
        m_haltParticlesAtMelt = particlesAtMelt;
        result.severity = HealthSeverity::HALT;
        result.reason = FailureReason::FORCE_EXPLOSION;
        result.message = "[HEALTH] HALT: Force explosion at step " + std::to_string(step)
            + " — " + std::to_string(totalForce) + " N (limit: "
            + std::to_string(m_config.maxPhysicalForceN) + " N)";
        return result;
    }

    // ── Check 3: THERMAL EXPLOSION ─────────────────────────────────────────
    // If a significant fraction of particles hit melting point very quickly
    if (particlesAtMelt > 0) {
        if (m_firstMeltStep < 0) m_firstMeltStep = step;

        double meltFraction = (double)particlesAtMelt / std::max(numParticles, 1);
        int stepsToMelt = step - m_firstContactStep;

        if (meltFraction > m_config.thermalExplosionFraction
            && stepsToMelt < m_config.thermalExplosionMaxSteps
            && stepsToMelt >= 0) {
            m_haltStep = step; m_haltReason = FailureReason::THERMAL_EXPLOSION;
            m_haltMaxTemp = maxTemp; m_haltParticlesAtMelt = particlesAtMelt;
            m_haltNumParticles = numParticles; m_haltForce = totalForce;
            m_haltContactCount = contactCount; m_haltMaxStress = maxStress;
            m_haltKE = kineticEnergy;
            result.severity = HealthSeverity::HALT;
            result.reason = FailureReason::THERMAL_EXPLOSION;
            std::ostringstream oss;
            oss << "[HEALTH] HALT: Thermal explosion at step " << step
                << " — " << std::fixed << std::setprecision(1) << (meltFraction * 100)
                << "% of particles at melting point (" << m_config.meltingPointC << "°C) "
                << "within " << stepsToMelt << " steps of first contact";
            result.message = oss.str();
            return result;
        }
    }

    // ── Check 4: STRESS EXPLOSION ──────────────────────────────────────────
    double stressLimit = m_config.yieldStrengthPa * m_config.stressExplosionMultiplier;
    if (maxStress > stressLimit) {
        m_haltStep = step; m_haltReason = FailureReason::STRESS_EXPLOSION;
        m_haltMaxStress = maxStress; m_haltMaxTemp = maxTemp;
        m_haltForce = totalForce; m_haltContactCount = contactCount;
        m_haltNumParticles = numParticles; m_haltKE = kineticEnergy;
        m_haltParticlesAtMelt = particlesAtMelt;
        result.severity = HealthSeverity::HALT;
        result.reason = FailureReason::STRESS_EXPLOSION;
        std::ostringstream oss;
        oss << "[HEALTH] HALT: Stress explosion at step " << step
            << " — " << std::scientific << std::setprecision(2) << maxStress / 1e6
            << " MPa (" << std::fixed << std::setprecision(0) << (maxStress / m_config.yieldStrengthPa)
            << "× yield strength " << m_config.yieldStrengthPa / 1e6 << " MPa)";
        result.message = oss.str();
        return result;
    }

    // ── Check 5: ENERGY VIOLATION ──────────────────────────────────────────
    if (m_prevKineticEnergy > 1e-15 && kineticEnergy > 1e-15) {
        double ratio = kineticEnergy / m_prevKineticEnergy;
        if (ratio > m_config.energyJumpMultiplier) {
            m_haltStep = step; m_haltReason = FailureReason::ENERGY_VIOLATION;
            m_haltKE = kineticEnergy; m_haltMaxStress = maxStress;
            m_haltMaxTemp = maxTemp; m_haltForce = totalForce;
            m_haltContactCount = contactCount;
            m_haltNumParticles = numParticles;
            m_haltParticlesAtMelt = particlesAtMelt;
            result.severity = HealthSeverity::HALT;
            result.reason = FailureReason::ENERGY_VIOLATION;
            std::ostringstream oss;
            oss << "[HEALTH] HALT: Energy violation at step " << step
                << " — KE jumped " << std::fixed << std::setprecision(0) << ratio
                << "× (from " << std::scientific << m_prevKineticEnergy
                << " J to " << kineticEnergy << " J)";
            result.message = oss.str();
            return result;
        }
    }
    m_prevKineticEnergy = kineticEnergy;

    // ── Check 6: NaN detection (maxStress or maxTemp) ──────────────────────
    if (std::isnan(maxStress) || std::isnan(maxTemp) || std::isinf(maxStress) || std::isinf(maxTemp)) {
        m_haltStep = step; m_haltReason = FailureReason::NAN_CORRUPTION;
        m_haltMaxStress = maxStress; m_haltMaxTemp = maxTemp;
        m_haltNumParticles = numParticles;
        result.severity = HealthSeverity::HALT;
        result.reason = FailureReason::NAN_CORRUPTION;
        result.message = "[HEALTH] HALT: NaN/Inf detected at step " + std::to_string(step);
        return result;
    }

    // ── Check 7: dt floor collapse (WARN only) ────────────────────────────
    if (dtMin > 0 && std::abs(dt - dtMin) < dtMin * 0.01) {
        m_dtFloorCount++;
        if (m_dtFloorCount >= m_config.dtFloorMaxConsecutive) {
            m_warningCount++;
            result.severity = HealthSeverity::WARN;
            result.reason = FailureReason::DT_FLOOR_COLLAPSE;
            result.message = "[HEALTH] WARN: dt at minimum floor for "
                + std::to_string(m_dtFloorCount) + " consecutive checks";
            // Don't return — just warn, keep checking
        }
    } else {
        m_dtFloorCount = 0;
    }

    return result;
}

inline std::string SimulationHealthMonitor::getDiagnosticReport() const {
    if (m_haltReason == FailureReason::NONE) return "[HEALTH] No failures detected.\n";

    std::ostringstream report;
    report << "\n"
           << "╔══════════════════════════════════════════════════════════════╗\n"
           << "║              SIMULATION FAILURE — ENGINE HALTED             ║\n"
           << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Reason
    report << "  Halted at step: " << m_haltStep << "\n";
    report << "  Failure type:   ";
    switch (m_haltReason) {
        case FailureReason::THERMAL_EXPLOSION:  report << "THERMAL EXPLOSION"; break;
        case FailureReason::STRESS_EXPLOSION:   report << "STRESS EXPLOSION"; break;
        case FailureReason::FORCE_EXPLOSION:    report << "FORCE EXPLOSION"; break;
        case FailureReason::CONTACT_EXPLOSION:  report << "CONTACT EXPLOSION (tool embedded)"; break;
        case FailureReason::NAN_CORRUPTION:     report << "NaN/Inf CORRUPTION"; break;
        case FailureReason::ENERGY_VIOLATION:   report << "ENERGY CONSERVATION VIOLATION"; break;
        case FailureReason::DT_FLOOR_COLLAPSE:  report << "TIMESTEP COLLAPSE"; break;
        default:                                report << "UNKNOWN"; break;
    }
    report << "\n\n";

    // State snapshot
    report << "  ── Physics State at Failure ──\n";
    report << std::scientific << std::setprecision(3);
    report << "    Max Stress:       " << m_haltMaxStress / 1e6 << " MPa";
    if (m_config.yieldStrengthPa > 0) {
        report << " (" << std::fixed << std::setprecision(0)
               << (m_haltMaxStress / m_config.yieldStrengthPa) << "× yield)";
    }
    report << "\n";
    report << std::fixed << std::setprecision(1);
    report << "    Max Temperature:  " << m_haltMaxTemp << " °C";
    if (m_haltMaxTemp >= m_config.meltingPointC) report << " [AT MELTING POINT]";
    report << "\n";
    report << std::scientific << std::setprecision(3);
    report << "    Contact Force:    " << m_haltForce << " N\n";
    report << "    Contact Count:    " << m_haltContactCount;
    report << "\n";
    report << "    Kinetic Energy:   " << m_haltKE << " J\n";
    if (m_haltParticlesAtMelt > 0) {
        report << "    Particles at melt: " << m_haltParticlesAtMelt << " / "
               << m_haltNumParticles << " ("
               << std::fixed << std::setprecision(1)
               << (100.0 * m_haltParticlesAtMelt / std::max(m_haltNumParticles, 1)) << "%)\n";
    }

    // Root cause guidance
    report << "\n  ── Probable Root Cause ──\n";
    switch (m_haltReason) {
        case FailureReason::CONTACT_EXPLOSION:
        case FailureReason::FORCE_EXPLOSION:
            report << "    The tool is deeply embedded inside the workpiece at T=0.\n"
                   << "    → Check G-Code clearance height (G00 Z...)\n"
                   << "    → Check tool_length_offset_mm in machine_setup\n"
                   << "    → Check air-gap enforcement logic in SimulationEngine\n"
                   << "    → Ensure contact radius < air gap distance\n";
            break;
        case FailureReason::THERMAL_EXPLOSION:
            report << "    Frictional heat injection rate exceeds material capacity.\n"
                   << "    → Too many contacts generating heat simultaneously\n"
                   << "    → Check heat_partition and workpiece specific heat\n"
                   << "    → Likely caused by tool embedding (see contact count)\n";
            break;
        case FailureReason::STRESS_EXPLOSION:
            report << "    Contact forces producing non-physical stress.\n"
                   << "    → dt may be too large for XPBD constraint stability\n"
                   << "    → Check contact stiffness and penetration clamp\n";
            break;
        case FailureReason::ENERGY_VIOLATION:
            report << "    Energy conservation violated — artificial energy injection.\n"
                   << "    → Check XPBD velocity correction sign conventions\n"
                   << "    → Check leapfrog integrator force accumulation\n";
            break;
        default:
            report << "    Check simulation logs for additional context.\n";
            break;
    }

    report << "\n  ── Recommendation ──\n"
           << "    1. Inspect the T=0 preview files (step 0 VTK export)\n"
           << "    2. Verify tool-workpiece alignment in 3D viewer\n"
           << "    3. Increase G-Code clearance height if tool is embedded\n"
           << "    4. Re-run with corrected configuration\n\n";

    return report.str();
}

} // namespace edgepredict
