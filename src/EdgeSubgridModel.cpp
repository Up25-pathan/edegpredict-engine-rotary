#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "EdgeSubgridModel.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace edgepredict {

using edgepredict::constants::PI;

EdgeSubgridModel::EdgeSubgridModel() = default;

void EdgeSubgridModel::initialize(const EdgeSubgridParams& params) {
    m_params = params;
    m_initialized = true;
}

void EdgeSubgridModel::configure(const Config& config) {
    const auto& j = config.getJson();
    if (!j.contains("edge_subgrid")) {
        std::cout << "[EdgeSubgrid] Not configured (no edge_subgrid section)" << std::endl;
        return;
    }

    const auto& es = j["edge_subgrid"];
    m_params.enabled = es.value("enabled", true);
    m_params.edgeRadius = es.value("edge_radius_mm", 0.020) / 1000.0;
    m_params.minimumStableRakeDeg = es.value("minimum_stable_rake_deg", -60.0);
    m_params.maxPloughingFraction = es.value("max_ploughing_fraction", 0.85);
    m_params.criticalRatio = es.value("critical_ratio", 0.0);
    m_initialized = true;
    std::cout << "[EdgeSubgrid] Enabled=" << m_params.enabled
              << " edgeRadius=" << m_params.edgeRadius * 1e6 << "um" << std::endl;
}

double EdgeSubgridModel::computeCriticalRatio() const {
    if (m_params.criticalRatio > 0.0) return m_params.criticalRatio;
    return 0.3;
}

double EdgeSubgridModel::computeDeadMetalZoneHeight(
    double uncutChipThickness) const
{
    double ratio = uncutChipThickness / m_params.edgeRadius;
    if (ratio > 2.0) return 0.0;

    double clamped = std::max(0.05, std::min(1.0, ratio));
    return m_params.edgeRadius * (1.0 - clamped) * 0.6;
}

EdgeSubgridResult EdgeSubgridModel::compute(
    double uncutChipThickness,
    double normalStressPa,
    double slidingVelocity,
    double temperatureC,
    double materialYieldPa,
    double materialYoungsModulus,
    double toolYoungsModulus) const
{
    EdgeSubgridResult result;

    if (!m_params.enabled || uncutChipThickness <= 0) {
        result.effectiveRakeAngleRad = 0.0;
        result.sizeEffectFactor = 1.0;
        result.ploughingDominant = false;
        return result;
    }

    double ratio = uncutChipThickness / m_params.edgeRadius;
    result.ploughingDominant = (ratio < 1.0);

    double degRange = 45.0;
    double effectiveRakeDeg = -degRange * std::exp(-ratio * 1.5);
    double minRake = m_params.minimumStableRakeDeg;
    effectiveRakeDeg = std::max(effectiveRakeDeg, minRake);
    result.effectiveRakeAngleRad = effectiveRakeDeg * PI / 180.0;

    result.deadMetalZoneHeight = computeDeadMetalZoneHeight(uncutChipThickness);

    double c = computeCriticalRatio();
    result.sizeEffectFactor = 1.0 + (c / std::max(ratio, 1e-12)) * 0.3;
    result.sizeEffectFactor = std::min(result.sizeEffectFactor, 5.0);

    double E_star = 1.0 / (
        (1.0 - 0.22 * 0.22) / std::max(toolYoungsModulus, 1e9) +
        (1.0 - 0.34 * 0.34) / std::max(materialYoungsModulus, 1e9)
    );
    double a_h = std::sqrt(m_params.edgeRadius * uncutChipThickness);
    double maxHertzPa = (2.0 * E_star) / PI * std::sqrt(
        uncutChipThickness / std::max(m_params.edgeRadius, 1e-12));
    double hertzStress = std::min(maxHertzPa, 5.0 * materialYieldPa);

    if (result.ploughingDominant) {
        double pRatio = 1.0 - ratio;
        double pf = pRatio * hertzStress * m_params.edgeRadius * 1e-4;
        double limit = normalStressPa * m_params.maxPloughingFraction;
        result.ploughingForceN = (pf < limit) ? pf : limit;
    }

    double tRef = 25.0;
    double tMelt = 1660.0;
    double T_star = (temperatureC - tRef) / (tMelt - tRef);
    T_star = std::max(0.0, std::min(1.0, T_star));
    result.edgeTemperatureFactor = 1.0 + 0.5 * std::exp(-ratio) * (1.0 + T_star);

    result.correctedSpecificEnergyPa = materialYieldPa * result.sizeEffectFactor;

    return result;
}

double EdgeSubgridModel::getEffectiveRakeAngle(
    double uncutChipThickness) const
{
    if (!m_params.enabled) return 0.0;
    double ratio = uncutChipThickness / m_params.edgeRadius;
    double deg = -45.0 * std::exp(-ratio * 1.5);
    return std::max(deg, m_params.minimumStableRakeDeg) * PI / 180.0;
}

double EdgeSubgridModel::getPloughingForce(
    double uncutChipThickness,
    double normalStress,
    double widthOfCut) const
{
    if (!m_params.enabled || uncutChipThickness <= 0) return 0.0;
    double ratio = uncutChipThickness / m_params.edgeRadius;
    if (ratio > 1.0) return 0.0;

    double pRatio = 1.0 - ratio;
    return pRatio * normalStress * m_params.edgeRadius * widthOfCut * 0.5;
}

double EdgeSubgridModel::getSizeEffectFactor(
    double uncutChipThickness) const
{
    if (!m_params.enabled || uncutChipThickness <= 0) return 1.0;
    double ratio = uncutChipThickness / m_params.edgeRadius;
    double c = computeCriticalRatio();
    double factor = 1.0 + (c / std::max(ratio, 1e-12)) * 0.3;
    return std::min(factor, 5.0);
}

} // namespace edgepredict
