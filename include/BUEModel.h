#pragma once

#include "Types.h"
#include "Config.h"
#include <cmath>
#include <vector>

namespace edgepredict {

enum class BUEPhase {
    NONE,
    NUCLEATION,
    GROWTH,
    STABLE,
    BREAKOFF,
    RECOVERY
};

struct BUEState {
    BUEPhase phase = BUEPhase::NONE;
    double height = 0.0;
    double width = 0.0;
    double coverageFraction = 0.0;
    double effectiveRakeChangeDeg = 0.0;
    double effectiveEdgeRadiusMultiplier = 1.0;
    double adhesionStrength = 0.0;
    double formationTime = 0.0;
    double breakoffCount = 0.0;
    double timeSinceLastBreakoff = 0.0;
    double stabilityIndex = 1.0;
    double roughnessPenaltyUm = 0.0;
    double chippingRiskIncrease = 0.0;
    double layerThicknessUm = 0.0;
    bool breakoffEvent = false;

    bool isActive() const { return phase != BUEPhase::NONE; }
};

struct BUEParams {
    bool enabled = true;

    double nucleationTemperatureC = 150.0;
    double nucleationStressPa = 500e6;
    double nucleationSpeedLow = 0.5;
    double nucleationSpeedHigh = 2.0;

    double growthRateConstant = 1e-8;
    double maxStableHeightUm = 200.0;
    double maxStableWidthUm = 500.0;

    double breakoffHeightRatio = 0.9;
    double breakoffForceFluctuation = 0.15;
    double recoveryTimeS = 0.005;

    double adhesionWorkAl = 0.15;
    double adhesionWorkSteel = 0.08;
    double adhesionWorkTi = 0.05;
    double adhesionWorkInconel = 0.04;

    double roughnessMultiplier = 3.0;
    double chippingRiskIncrease = 0.25;
};

class BUEModel {
public:
    BUEModel();
    ~BUEModel() = default;

    void initialize(const BUEParams& params);
    void configure(const Config& config);

    BUEState update(
        double dt,
        double uncutChipThickness,
        double cuttingSpeed,
        double temperatureC,
        double normalStressPa,
        double shearStressPa,
        double toolRakeAngleRad,
        const std::string& workpieceMaterial
    );

    const BUEState& getState() const { return m_state; }
    BUEPhase getPhase() const { return m_state.phase; }

    double getEffectiveRakeCorrectionDeg() const;
    double getForceModulationFactor() const;
    double getRoughnessPenalty() const;
    double getChippingRiskIncrease() const;
    bool isActive() const { return m_state.isActive(); }
    bool isEnabled() const { return m_params.enabled; }

    void reset();

private:
    BUEParams m_params;
    BUEState m_state;
    bool m_initialized = false;

    double getAdhesionWork(const std::string& material) const;
    bool checkNucleation(double speed, double temp, double stress,
                         double adhesionWork) const;
    void updateGrowth(double dt, double speed, double temp, double stress);
    void updatePhase(double speed, double temp, double stress);
    void updateEffects();
    void triggerBreakoff();
};

} // namespace edgepredict
