#include "BUEModel.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace edgepredict {

static constexpr double PI = 3.14159265358979323846;

BUEModel::BUEModel() = default;

void BUEModel::initialize(const BUEParams& params) {
    m_params = params;
    m_initialized = true;
    m_state = BUEState();
}

void BUEModel::configure(const Config& config) {
    const auto& j = config.getJson();
    if (!j.contains("bue_model")) {
        std::cout << "[BUEModel] Not configured (no bue_model section)" << std::endl;
        return;
    }

    const auto& bue = j["bue_model"];
    m_params.enabled = bue.value("enabled", true);
    m_params.nucleationTemperatureC = bue.value("nucleation_temperature_C", 150.0);
    m_params.nucleationStressPa = bue.value("nucleation_stress_Pa", 500e6);
    m_params.nucleationSpeedLow = bue.value("nucleation_speed_low_m_s", 0.5);
    m_params.nucleationSpeedHigh = bue.value("nucleation_speed_high_m_s", 2.0);
    m_params.growthRateConstant = bue.value("growth_rate_constant", 1e-8);
    m_params.maxStableHeightUm = bue.value("max_stable_height_um", 200.0);
    m_params.adhesionWorkAl = bue.value("adhesion_work_Al", 0.15);
    m_params.adhesionWorkSteel = bue.value("adhesion_work_steel", 0.08);
    m_params.adhesionWorkTi = bue.value("adhesion_work_Ti", 0.05);
    m_params.adhesionWorkInconel = bue.value("adhesion_work_Inconel", 0.04);
    m_params.roughnessMultiplier = bue.value("roughness_multiplier", 3.0);
    m_params.chippingRiskIncrease = bue.value("chipping_risk_increase", 0.25);

    m_initialized = true;
    std::cout << "[BUEModel] Enabled=" << m_params.enabled
              << " nucleationTemp=" << m_params.nucleationTemperatureC << "C" << std::endl;
}

double BUEModel::getAdhesionWork(const std::string& material) const {
    std::string m = material;
    std::transform(m.begin(), m.end(), m.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (m.find("alum") != std::string::npos || m.find("al") == 0)
        return m_params.adhesionWorkAl;
    if (m.find("inconel") != std::string::npos || m.find("inco") != std::string::npos)
        return m_params.adhesionWorkInconel;
    if (m.find("titanium") != std::string::npos || m.find("ti") == 0)
        return m_params.adhesionWorkTi;
    if (m.find("steel") != std::string::npos || m.find("1045") != std::string::npos)
        return m_params.adhesionWorkSteel;

    return 0.10;
}

bool BUEModel::checkNucleation(
    double speed, double temp, double stress, double adhesionWork) const
{
    if (!m_initialized || !m_params.enabled) return false;
    if (speed < m_params.nucleationSpeedLow || speed > m_params.nucleationSpeedHigh)
        return false;
    if (temp < m_params.nucleationTemperatureC)
        return false;
    if (stress < m_params.nucleationStressPa * 0.5)
        return false;

    double tempFactor = (temp - m_params.nucleationTemperatureC) /
                        (500.0 - m_params.nucleationTemperatureC);
    tempFactor = std::max(0.0, std::min(1.0, tempFactor));

    double speedFactor = 1.0 - (speed - m_params.nucleationSpeedLow) /
                                (m_params.nucleationSpeedHigh - m_params.nucleationSpeedLow);
    speedFactor = std::max(0.0, std::min(1.0, speedFactor));

    double stressFactor = stress / (m_params.nucleationStressPa * 2.0);
    stressFactor = std::min(1.0, stressFactor);

    double prob = adhesionWork * tempFactor * speedFactor * stressFactor * 2.0;

    return prob > 0.3;
}

void BUEModel::updateGrowth(double dt, double speed, double temp, double stress) {
    double growthRate = m_params.growthRateConstant * speed *
                        (1.0 + 0.5 * (temp - m_params.nucleationTemperatureC) / 200.0);
    growthRate *= (stress / m_params.nucleationStressPa);
    growthRate = std::max(0.0, growthRate);

    m_state.height += growthRate * dt;
    m_state.width += growthRate * dt * 0.3;

    double maxHeight = m_params.maxStableHeightUm * 1e-6;
    double maxWidth = m_params.maxStableWidthUm * 1e-6;

    m_state.height = std::min(m_state.height, maxHeight);
    m_state.width = std::min(m_state.width, maxWidth);

    m_state.coverageFraction = m_state.height / maxHeight;
    m_state.coverageFraction = std::min(1.0, m_state.coverageFraction);
}

void BUEModel::triggerBreakoff() {
    m_state.phase = BUEPhase::BREAKOFF;
    m_state.breakoffCount += 1.0;
    m_state.timeSinceLastBreakoff = 0.0;
}

void BUEModel::updatePhase(double speed, double temp, double stress) {
    double maxHeight = m_params.maxStableHeightUm * 1e-6;

    switch (m_state.phase) {
    case BUEPhase::NONE:
        break;

    case BUEPhase::NUCLEATION:
        if (m_state.height > maxHeight * 0.1) {
            m_state.phase = BUEPhase::GROWTH;
        }
        break;

    case BUEPhase::GROWTH:
        if (m_state.height >= maxHeight * m_params.breakoffHeightRatio) {
            triggerBreakoff();
        } else if (m_state.coverageFraction > 0.6) {
            m_state.phase = BUEPhase::STABLE;
        }
        break;

    case BUEPhase::STABLE: {
        double fluct = std::abs(std::sin(m_state.formationTime * 1000.0)) *
                       m_params.breakoffForceFluctuation;
        double threshold = m_params.breakoffForceFluctuation *
                          (1.0 + 0.5 * (1.0 - m_state.coverageFraction));
        if (fluct > threshold || m_state.height >= maxHeight) {
            triggerBreakoff();
        }
        break;
    }

    case BUEPhase::BREAKOFF:
        if (m_state.timeSinceLastBreakoff > m_params.recoveryTimeS) {
            m_state.height *= 0.3;
            m_state.width *= 0.3;
            m_state.coverageFraction *= 0.3;
            m_state.phase = BUEPhase::RECOVERY;
        }
        break;

    case BUEPhase::RECOVERY:
        if (m_state.height < maxHeight * 0.05) {
            m_state.phase = BUEPhase::NONE;
        } else {
            m_state.phase = BUEPhase::GROWTH;
        }
        break;
    }
}

void BUEModel::updateEffects() {
    double maxHeight = m_params.maxStableHeightUm * 1e-6;
    double heightRatio = m_state.height / std::max(maxHeight, 1e-12);

    m_state.effectiveRakeChangeDeg = -15.0 * heightRatio;
    m_state.effectiveEdgeRadiusMultiplier = 1.0 + 2.0 * heightRatio;
    m_state.effectiveEdgeRadiusMultiplier = std::min(
        m_state.effectiveEdgeRadiusMultiplier, 4.0);

    if (m_state.phase == BUEPhase::BREAKOFF) {
        m_state.stabilityIndex = 0.2;
    } else if (m_state.phase == BUEPhase::NUCLEATION) {
        m_state.stabilityIndex = 0.6;
    } else if (m_state.phase == BUEPhase::STABLE) {
        m_state.stabilityIndex = 0.8;
    } else if (m_state.phase == BUEPhase::GROWTH) {
        m_state.stabilityIndex = 0.5;
    } else {
        m_state.stabilityIndex = 1.0;
    }
}

BUEState BUEModel::update(
    double dt,
    double uncutChipThickness,
    double cuttingSpeed,
    double temperatureC,
    double normalStressPa,
    double shearStressPa,
    double toolRakeAngleRad,
    const std::string& workpieceMaterial)
{
    if (!m_initialized || !m_params.enabled || uncutChipThickness <= 0 || dt <= 0) {
        return BUEState();
    }

    double adhesionWork = getAdhesionWork(workpieceMaterial);

    m_state.formationTime += dt;
    m_state.timeSinceLastBreakoff += dt;

    if (!m_state.isActive()) {
        if (checkNucleation(cuttingSpeed, temperatureC, normalStressPa, adhesionWork)) {
            m_state.phase = BUEPhase::NUCLEATION;
            m_state.adhesionStrength = adhesionWork;
            std::cout << "[BUE] Nucleation at T=" << temperatureC
                      << "C, speed=" << cuttingSpeed << " m/s" << std::endl;
        }
    }

    if (m_state.isActive()) {
        updateGrowth(dt, cuttingSpeed, temperatureC, normalStressPa);
        updatePhase(cuttingSpeed, temperatureC, normalStressPa);
        updateEffects();
    }

    return m_state;
}

double BUEModel::getEffectiveRakeCorrectionDeg() const {
    return m_state.effectiveRakeChangeDeg;
}

double BUEModel::getForceModulationFactor() const {
    if (m_state.phase == BUEPhase::BREAKOFF)
        return 1.0 + m_params.breakoffForceFluctuation * 3.0;
    if (m_state.phase == BUEPhase::GROWTH || m_state.phase == BUEPhase::STABLE)
        return 1.0 - m_state.coverageFraction * 0.3;
    return 1.0;
}

double BUEModel::getRoughnessPenalty() const {
    if (!m_state.isActive()) return 1.0;
    double base = 1.0 + (m_params.roughnessMultiplier - 1.0) * m_state.coverageFraction;
    if (m_state.phase == BUEPhase::BREAKOFF)
        return base * 1.5;
    return base;
}

double BUEModel::getChippingRiskIncrease() const {
    if (!m_state.isActive()) return 0.0;
    double risk = m_params.chippingRiskIncrease * m_state.coverageFraction;
    if (m_state.phase == BUEPhase::BREAKOFF)
        risk += m_params.chippingRiskIncrease * 0.5;
    return risk;
}

void BUEModel::reset() {
    m_state = BUEState();
}

} // namespace edgepredict

void BUEModel::extrapolateSteadyState(double remainingTimeS, double projectedTemperatureC) {
    if (!m_initialized || !m_params.enabled) return;
    
    if (projectedTemperatureC > m_params.nucleationTemperatureC) {
        // Assume BUE reaches its stable maximum during the remaining cut time
        m_state.phase = BUEPhase::STABLE;
        m_state.height = m_params.maxStableHeightUm;
        m_state.width = m_params.maxStableWidthUm;
        m_state.coverageFraction = 1.0;
        m_state.effectiveRakeChangeDeg = -15.0 * (m_state.height / m_params.maxStableHeightUm);
        m_state.roughnessPenaltyUm = m_params.roughnessMultiplier * (m_state.height / 100.0);
        m_state.chippingRiskIncrease = m_params.chippingRiskIncrease;
    } else {
        m_state.phase = BUEPhase::NONE;
    }
}
