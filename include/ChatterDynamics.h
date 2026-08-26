#pragma once

#include "Types.h"
#include "Config.h"
#include <vector>
#include <cmath>

namespace edgepredict {

struct ModalParams {
    double massX = 0.5;
    double stiffnessX = 1e7;
    double dampingRatioX = 0.02;

    double massY = 0.5;
    double stiffnessY = 1e7;
    double dampingRatioY = 0.02;

    double massZ = 0.5;
    double stiffnessZ = 1e7;
    double dampingRatioZ = 0.02;

    double couplingFactor = 0.0;
};

struct StabilityLobe {
    double spindleRPM = 0.0;
    double maxDepthOfCutMm = 0.0;
    double chatterFrequencyHz = 0.0;
    int lobeNumber = 0;
    bool isStable = false;
};

struct ChatterState {
    Vec3 displacement;
    Vec3 velocity;
    Vec3 acceleration;

    Vec3 previousDisplacement;
    double timeSinceLastToothPass = 0.0;
    double chipThicknessModulation = 0.0;

    double chatterFrequencyHz = 0.0;
    double stabilityParameter = 1.0;
    bool chattering = false;
    double chatterGrowthRate = 0.0;
    double vibrationEnergy = 0.0;

    std::vector<StabilityLobe> stabilityLobes;
    double recommendedRPM = 0.0;
    double maxStableDepthMm = 0.0;
};

struct ChatterParams {
    bool enabled = true;
    bool computeStabilityLobes = true;
    int numTeeth = 4;
    double radialImmersion = 1.0;
    double cuttingForceCoeffX = 1.0;
    double specificCuttingForce = 2000e6;

    double maxChatterFrequency = 10000.0;
    double minChatterFrequency = 50.0;
    int lobeRangeStart = 0;
    int lobeRangeEnd = 10;
    double stabilitySafetyFactor = 0.8;
};

class ChatterDynamics {
public:
    ChatterDynamics();
    ~ChatterDynamics() = default;

    void initialize(const ChatterParams& chatterParams,
                    const ModalParams& modalParams);
    void configure(const Config& config);

    void updateDynamics(
        double dt,
        double spindleRPM,
        double depthOfCut,
        double widthOfCut,
        double feedPerTooth,
        double cuttingForceMagnitude,
        const Vec3& cuttingDirection,
        const Vec3& thrustDirection
    );

    double getChipThicknessModulation() const;
    Vec3 getVibrationDisplacement() const;
    Vec3 getVibrationVelocity() const;
    bool isChattering() const { return m_state.chattering; }
    double getStabilityParameter() const { return m_state.stabilityParameter; }
    double getChatterFrequency() const { return m_state.chatterFrequencyHz; }

    void computeStabilityLobes(
        double maxDepthMm,
        double minRPM,
        double maxRPM
    );

    const ChatterState& getState() const { return m_state; }
    const ModalParams& getModalParams() const { return m_modal; }
    bool isEnabled() const { return m_params.enabled; }

    void reset();

private:
    ChatterParams m_params;
    ModalParams m_modal;
    ChatterState m_state;
    bool m_initialized = false;

    void solveRegenerativeChatter(double dt, double chipThickness);
    double computeDepthOfCutLimit(double rpm, double freq, int lobe) const;
    double computePhaseShift(double rpm, double freq) const;

    // Circular buffer for displacement history (non-static to avoid overflow bugs)
    double m_dispHistory[10] = {0};
    int m_histIdx = 0;
};

} // namespace edgepredict
