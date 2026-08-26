#pragma once

#include "Types.h"
#include "Config.h"
#include <cmath>
#include <vector>
#include <string>

namespace edgepredict {

class MPMSolver;
class FEMSolver;
class CFDSolverGPU;

/**
 * @brief Runtime state for coolant hardening model
 */
struct CoolantHardeningState {
    double jetStagnationHTC = 0.0;            // Current stagnation zone HTC
    double wallJetHTC = 0.0;                  // Current wall jet zone HTC
    double boilingRegimeHTC = 0.0;            // Current boiling regime HTC
    double capillaryPenetrationDepth = 0.0;   // Current wicking depth (m)
    double effectiveCoolantHTC = 0.0;         // Combined effective HTC
    double meanCoolantTemperature = 25.0;     // Running mean coolant temp (°C)
    double lastCoolingApplied = 0.0;          // Total °C of cooling applied last step
    double lastHeatRemovedJ = 0.0;            // Total heat energy removed last step (J)
    int cooledParticleCount = 0;              // Particles cooled last step
    int updateCount = 0;
};

/**
 * @brief Coolant hardening model (Gap #7a–7f)
 *
 * Implements six sub-items of Gap #7 for the edgepredict engine:
 *   a) Jet impingement cooling mechanics
 *   b) Film boiling / Leidenfrost regime transition
 *   c) Capillary penetration (wicking into tool-workpiece gap)
 *   d) Thermal shock hardening (quench hardness on MPM particles)
 *   e) MPM–CFD bidirectional coupling (two-way heat exchange)
 *   f) Flow-regime-dependent HTC (local Re/Pr based cooling)
 *
 * All sub-items can be independently enabled/disabled via JSON config.
 */
class CoolantHardeningModel {
public:
    CoolantHardeningModel();
    ~CoolantHardeningModel() = default;

    void initialize(const CoolantHardeningConfig& params);
    void configure(const Config& config);

    /**
     * @brief Update all coolant hardening physics each timestep
     * @param mpm       MPM solver (workpiece particles)
     * @param fem       FEM solver (tool nodes)
     * @param cfd       CFD solver (coolant grid)
     * @param dt        Timestep (s)
     */
    void update(MPMSolver* mpm, FEMSolver* fem, CFDSolverGPU* cfd, double dt);

    bool isEnabled() const { return m_params.enabled; }
    const CoolantHardeningState& getState() const { return m_state; }
    double getEffectiveHTC() const { return m_state.effectiveCoolantHTC; }

    // Per-gap enable queries (for JSON-driven on/off)
    bool isJetImpingementEnabled() const { return m_params.jetImpingementEnabled; }
    bool isFilmBoilingEnabled() const { return m_params.filmBoilingEnabled; }
    bool isCapillaryWickingEnabled() const { return m_params.capillaryWickingEnabled; }
    bool isThermalShockEnabled() const { return m_params.thermalShockEnabled; }
    bool isBidirectionalCouplingEnabled() const { return m_params.bidirectionalCouplingEnabled; }
    bool isFlowRegimeHTCEnabled() const { return m_params.flowRegimeHTCEnabled; }

    void reset();

private:
    CoolantHardeningConfig m_params;
    CoolantHardeningState m_state;
    bool m_initialized = false;

    // Gap #7a: Compute jet impingement HTC at given position
    double computeJetImpingementHTC(const Vec3& position, const Vec3& jetOrigin) const;

    // Gap #7b: Compute boiling regime HTC based on surface temperature
    double computeBoilingHTC(double surfaceTemperatureC) const;
    double computeRadiationHTC(double surfaceTemperatureC, double ambientC = 25.0) const;

    // Gap #7c: Update capillary penetration depth
    double updateCapillaryPenetration(double dt, double contactGapNm);

    // Gap #7d: Compute quench hardness correction for an MPM particle
    double computeQuenchHardnessIncrease(double particleTemp, double coolingRate) const;

    // Gap #7e: Apply convective cooling from CFD to MPM particles
    void applyCFDCoolingToParticles(MPMSolver* mpm, CFDSolverGPU* cfd, double dt);

    // Gap #7f: Compute flow-regime HTC from local Re/Pr
    double computeFlowRegimeHTC(double velocity, double tempFilm, double charLength) const;
};

} // namespace edgepredict
