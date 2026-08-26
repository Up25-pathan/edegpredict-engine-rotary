#pragma once

#include "Types.h"
#include "Config.h"
#include <cmath>

namespace edgepredict {

struct EdgeSubgridParams {
    double edgeRadius = 20e-6;
    bool enabled = true;
    double minimumStableRakeDeg = -60.0;
    double maxPloughingFraction = 0.85;
    double criticalRatio = 0.0;
};

struct EdgeSubgridResult {
    double effectiveRakeAngleRad = 0.0;
    double ploughingForceN = 0.0;
    double sizeEffectFactor = 1.0;
    double deadMetalZoneHeight = 0.0;
    double edgeTemperatureFactor = 1.0;
    bool ploughingDominant = false;
    double correctedSpecificEnergyPa = 0.0;
};

class EdgeSubgridModel {
public:
    EdgeSubgridModel();
    ~EdgeSubgridModel() = default;

    void initialize(const EdgeSubgridParams& params);
    void configure(const Config& config);

    EdgeSubgridResult compute(
        double uncutChipThickness,
        double normalStressPa,
        double slidingVelocity,
        double temperatureC,
        double materialYieldPa,
        double materialYoungsModulus,
        double toolYoungsModulus
    ) const;

    double getEffectiveRakeAngle(double uncutChipThickness) const;
    double getPloughingForce(double uncutChipThickness, double normalStress,
                             double widthOfCut) const;
    double getSizeEffectFactor(double uncutChipThickness) const;

    bool isEnabled() const { return m_params.enabled; }
    double getEdgeRadius() const { return m_params.edgeRadius; }

private:
    EdgeSubgridParams m_params;
    bool m_initialized = false;

    double computeDeadMetalZoneHeight(double uncutChipThickness) const;
    double computeCriticalRatio() const;
};

} // namespace edgepredict
