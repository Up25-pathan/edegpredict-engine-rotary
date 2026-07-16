#include "ChatterDynamics.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

namespace edgepredict {

using edgepredict::constants::PI;

ChatterDynamics::ChatterDynamics() = default;

void ChatterDynamics::initialize(
    const ChatterParams& chatterParams,
    const ModalParams& modalParams)
{
    m_params = chatterParams;
    m_modal = modalParams;
    m_state = ChatterState();
    m_initialized = true;
}

void ChatterDynamics::configure(const Config& config) {
    const auto& j = config.getJson();
    if (!j.contains("chatter_dynamics")) {
        std::cout << "[ChatterDynamics] Not configured (no chatter_dynamics section)" << std::endl;
        return;
    }

    const auto& cd = j["chatter_dynamics"];

    m_params.enabled = cd.value("enabled", true);
    m_params.numTeeth = cd.value("num_teeth", 4);
    m_params.radialImmersion = cd.value("radial_immersion", 1.0);
    m_params.specificCuttingForce = cd.value("specific_cutting_force_Pa", 2000e6);
    m_params.stabilitySafetyFactor = cd.value("stability_safety_factor", 0.8);

    if (cd.contains("modal")) {
        const auto& mod = cd["modal"];
        m_modal.massX = mod.value("mass_X_kg", 0.5);
        m_modal.stiffnessX = mod.value("stiffness_X_N_m", 1e7);
        m_modal.dampingRatioX = mod.value("damping_ratio_X", 0.02);
        m_modal.massY = mod.value("mass_Y_kg", 0.5);
        m_modal.stiffnessY = mod.value("stiffness_Y_N_m", 1e7);
        m_modal.dampingRatioY = mod.value("damping_ratio_Y", 0.02);
        m_modal.massZ = mod.value("mass_Z_kg", 0.5);
        m_modal.stiffnessZ = mod.value("stiffness_Z_N_m", 1e7);
        m_modal.dampingRatioZ = mod.value("damping_ratio_Z", 0.02);
        m_modal.couplingFactor = mod.value("coupling_factor", 0.0);
    }

    m_initialized = true;
    std::cout << "[ChatterDynamics] Enabled=" << m_params.enabled
              << " teeth=" << m_params.numTeeth << " safetyFactor="
              << m_params.stabilitySafetyFactor << std::endl;
}

double ChatterDynamics::computePhaseShift(double rpm, double freq) const {
    double toothPassFreq = rpm / 60.0 * m_params.numTeeth;
    double phase = 2.0 * PI * freq / toothPassFreq;
    while (phase > 2.0 * PI) phase -= 2.0 * PI;
    return phase;
}

double ChatterDynamics::computeDepthOfCutLimit(
    double rpm, double freq, int lobe) const
{
    double omega = 2.0 * PI * freq;
    double k_s = m_params.specificCuttingForce;

    double Kx = m_params.cuttingForceCoeffX;
    double ae_over_D = m_params.radialImmersion;

    double kx = k_s * ae_over_D * Kx;

    double Re_G = 0.0;
    double Im_G = 0.0;

    double k_x = m_modal.stiffnessX;
    double zeta_x = m_modal.dampingRatioX;
    double omega_n_x = std::sqrt(k_x / std::max(m_modal.massX, 1e-9));
    double r_x = omega / omega_n_x;
    double Re_Gx = (1.0 - r_x * r_x) / (k_x * std::pow(1.0 - r_x * r_x, 2.0) +
                   k_x * std::pow(2.0 * zeta_x * r_x, 2.0));
    double Im_Gx = -(2.0 * zeta_x * r_x) / (k_x * std::pow(1.0 - r_x * r_x, 2.0) +
                   k_x * std::pow(2.0 * zeta_x * r_x, 2.0));

    double k_y = m_modal.stiffnessY;
    double zeta_y = m_modal.dampingRatioY;
    double omega_n_y = std::sqrt(k_y / std::max(m_modal.massY, 1e-9));
    double r_y = omega / omega_n_y;
    double Re_Gy = (1.0 - r_y * r_y) / (k_y * std::pow(1.0 - r_y * r_y, 2.0) +
                   k_y * std::pow(2.0 * zeta_y * r_y, 2.0));
    double Im_Gy = -(2.0 * zeta_y * r_y) / (k_y * std::pow(1.0 - r_y * r_y, 2.0) +
                   k_y * std::pow(2.0 * zeta_y * r_y, 2.0));

    Re_G = Re_Gx * 0.5 + Re_Gy * 0.5;
    Im_G = Im_Gx * 0.5 + Im_Gy * 0.5;

    double phi = computePhaseShift(rpm, freq);
    double Re_FR = Re_G * (1.0 - std::cos(phi)) + Im_G * std::sin(phi);

    if (std::abs(Re_FR) < 1e-15) return 1e6;

    double Kt = k_s;
    double b_lim = -1.0 / (2.0 * Kt * m_params.numTeeth * Re_FR);
    b_lim *= m_params.stabilitySafetyFactor;

    return std::max(0.0, b_lim) * 1000.0;
}

void ChatterDynamics::computeStabilityLobes(
    double maxDepthMm, double minRPM, double maxRPM)
{
    if (!m_initialized || !m_params.computeStabilityLobes) return;

    m_state.stabilityLobes.clear();
    int numSteps = 200;

    double bestDepth = 0.0;
    double bestRPM = minRPM;

    for (int lobe = m_params.lobeRangeStart; lobe <= m_params.lobeRangeEnd; ++lobe) {
        for (int i = 0; i < numSteps; ++i) {
            double rpm = minRPM + (maxRPM - minRPM) * i / numSteps;
            double toothPassFreq = rpm / 60.0 * m_params.numTeeth;

            for (int mode = 0; mode < 3; ++mode) {
                double freq = 0.0;
                double k = 0.0, m = 0.0, zeta = 0.0;

                switch (mode) {
                case 0:
                    k = m_modal.stiffnessX;
                    m = m_modal.massX;
                    zeta = m_modal.dampingRatioX;
                    break;
                case 1:
                    k = m_modal.stiffnessY;
                    m = m_modal.massY;
                    zeta = m_modal.dampingRatioY;
                    break;
                case 2:
                    k = m_modal.stiffnessZ;
                    m = m_modal.massZ;
                    zeta = m_modal.dampingRatioZ;
                    break;
                }

                double omega_n = std::sqrt(k / std::max(m, 1e-9));
                double omega_d = omega_n * std::sqrt(1.0 - zeta * zeta);

                double f_lower = omega_d / (2.0 * PI) * 0.7;
                double f_upper = omega_d / (2.0 * PI) * 1.3;
                f_lower = std::max(f_lower, m_params.minChatterFrequency);
                f_upper = std::min(f_upper, m_params.maxChatterFrequency);

                for (int fi = 0; fi < 20; ++fi) {
                    double freqTry = f_lower + (f_upper - f_lower) * fi / 19.0;
                    double phase = computePhaseShift(rpm, freqTry);
                    double epsilon = 2.0 * PI - phase - 2.0 * PI * lobe;

                    if (std::abs(epsilon) < 0.3) {
                        StabilityLobe lobeData;
                        lobeData.spindleRPM = rpm;
                        lobeData.chatterFrequencyHz = freqTry;
                        lobeData.lobeNumber = lobe;
                        lobeData.maxDepthOfCutMm = computeDepthOfCutLimit(
                            rpm, freqTry, lobe);
                        lobeData.isStable = true;
                        m_state.stabilityLobes.push_back(lobeData);

                        if (lobeData.maxDepthOfCutMm > bestDepth) {
                            bestDepth = lobeData.maxDepthOfCutMm;
                            bestRPM = rpm;
                        }
                        break;
                    }
                }
            }
        }
    }

    m_state.maxStableDepthMm = std::min(bestDepth, maxDepthMm);
    m_state.recommendedRPM = bestRPM;

    if (!m_state.stabilityLobes.empty()) {
        std::cout << "[Chatter] Stability lobes computed: "
                  << m_state.stabilityLobes.size() << " points, "
                  << "max stable depth=" << m_state.maxStableDepthMm << " mm "
                  << "@ " << m_state.recommendedRPM << " RPM" << std::endl;
    }
}

void ChatterDynamics::updateDynamics(
    double dt,
    double spindleRPM,
    double depthOfCut,
    double widthOfCut,
    double feedPerTooth,
    double cuttingForceMagnitude,
    const Vec3& cuttingDirection,
    const Vec3& thrustDirection)
{
    if (!m_initialized || !m_params.enabled || dt <= 0) return;

    m_state.timeSinceLastToothPass += dt;

    double toothPassInterval = 60.0 / (spindleRPM * m_params.numTeeth);
    bool toothImpact = (m_state.timeSinceLastToothPass >= toothPassInterval);

    if (toothImpact) {
        m_state.timeSinceLastToothPass = 0.0;
    }

    double chipThickness = feedPerTooth + m_state.chipThicknessModulation;

    double dampingX = 2.0 * m_modal.dampingRatioX *
                      std::sqrt(m_modal.massX * m_modal.stiffnessX);
    double dampingY = 2.0 * m_modal.dampingRatioY *
                      std::sqrt(m_modal.massY * m_modal.stiffnessY);
    double dampingZ = 2.0 * m_modal.dampingRatioZ *
                      std::sqrt(m_modal.massZ * m_modal.stiffnessZ);

    double forceMag = 0.0;
    if (toothImpact && chipThickness > 0) {
        forceMag = m_params.specificCuttingForce * chipThickness * depthOfCut;
    }

    Vec3 force(
        forceMag * cuttingDirection.x + cuttingForceMagnitude * thrustDirection.x * 0.3,
        forceMag * cuttingDirection.y + cuttingForceMagnitude * thrustDirection.y * 0.3,
        forceMag * cuttingDirection.z + cuttingForceMagnitude * thrustDirection.z * 0.3
    );

    m_state.previousDisplacement = m_state.displacement;

    double accelX = (force.x - dampingX * m_state.velocity.x -
                     m_modal.stiffnessX * m_state.displacement.x) /
                    std::max(m_modal.massX, 1e-9);
    double accelY = (force.y - dampingY * m_state.velocity.y -
                     m_modal.stiffnessY * m_state.displacement.y) /
                    std::max(m_modal.massY, 1e-9);
    double accelZ = (force.z - dampingZ * m_state.velocity.z -
                     m_modal.stiffnessZ * m_state.displacement.z) /
                    std::max(m_modal.massZ, 1e-9);

    m_state.acceleration = Vec3(accelX, accelY, accelZ);

    m_state.velocity.x += m_state.acceleration.x * dt;
    m_state.velocity.y += m_state.acceleration.y * dt;
    m_state.velocity.z += m_state.acceleration.z * dt;

    m_state.displacement.x += m_state.velocity.x * dt;
    m_state.displacement.y += m_state.velocity.y * dt;
    m_state.displacement.z += m_state.velocity.z * dt;

    // Clamp displacement to physically reasonable range (0.5mm max deflection)
    // Prevents unbounded growth when regenerative chatter feedback diverges
    // numerically (forward Euler integration of harmonic oscillator + force feedback
    // can grow without bound even at moderate damping ratios).
    {
        double maxDisp = 5.0e-4; // 0.5 mm
        double dispMag = m_state.displacement.length();
        if (dispMag > maxDisp) {
            m_state.displacement = m_state.displacement * (maxDisp / dispMag);
            // Damp velocity on clamping to dissipate excess kinetic energy
            m_state.velocity.x *= 0.5;
            m_state.velocity.y *= 0.5;
            m_state.velocity.z *= 0.5;
        }
    }

    m_state.vibrationEnergy = 0.5 * m_modal.massX *
        (m_state.velocity.x * m_state.velocity.x +
         m_state.velocity.y * m_state.velocity.y +
         m_state.velocity.z * m_state.velocity.z) +
        0.5 * m_modal.stiffnessX *
        (m_state.displacement.x * m_state.displacement.x +
         m_state.displacement.y * m_state.displacement.y +
         m_state.displacement.z * m_state.displacement.z);

    solveRegenerativeChatter(dt, chipThickness);

    if (toothImpact) {
        double displacementDiff = (m_state.displacement - m_state.previousDisplacement).length();
        m_state.chipThicknessModulation = displacementDiff * 0.5;
    }
}

void ChatterDynamics::solveRegenerativeChatter(double dt, double chipThickness) {
    if (!m_initialized || dt <= 0) return;
    // No cutting = no regenerative chatter — reset state and exit
    if (chipThickness <= 0.0) {
        m_state.chattering = false;
        m_state.chatterGrowthRate = 0.0;
        m_state.stabilityParameter = 1.0;
        return;
    }

    double k = m_modal.stiffnessX;
    double m = m_modal.massX;
    double zeta = m_modal.dampingRatioX;
    double omega_n = std::sqrt(k / std::max(m, 1e-9));

    double growth = 0.0;
    double prevDisp = m_state.previousDisplacement.length();
    double currDisp = m_state.displacement.length();

    int samples = 10;

    m_dispHistory[m_histIdx % samples] = currDisp;
    m_histIdx++;

    if (m_histIdx >= samples) {
        double mean = 0.0;
        for (int i = 0; i < samples; ++i) mean += m_dispHistory[i];
        mean /= samples;

        if (mean > 1e-12) {
            double variance = 0.0;
            for (int i = 0; i < samples; ++i)
                variance += (m_dispHistory[i] - mean) * (m_dispHistory[i] - mean);
            variance /= samples;
            growth = std::sqrt(variance) / mean;
        }
    }

    double toothPassFreq = 60.0 / (m_state.timeSinceLastToothPass + 1e-9);

    double dominantFreq = omega_n / (2.0 * PI);
    for (double f = m_params.minChatterFrequency;
         f < m_params.maxChatterFrequency; f += 10.0) {
        double response = 1.0 / std::sqrt(
            std::pow(1.0 - std::pow(f / (omega_n / (2.0 * PI)), 2.0), 2.0) +
            std::pow(2.0 * zeta * f / (omega_n / (2.0 * PI)), 2.0));
        if (response > 3.0 && f > dominantFreq * 0.8 && f < dominantFreq * 1.2) {
            dominantFreq = f;
        }
    }

    m_state.chatterFrequencyHz = dominantFreq;
    m_state.chatterGrowthRate = growth;

    double threshold = 0.3;
    if (growth > threshold && m_histIdx >= samples) {
        m_state.chattering = true;
        m_state.stabilityParameter = 1.0 - std::min(1.0, (growth - threshold) / threshold);
    } else {
        if (m_state.chattering && growth < threshold * 0.3) {
            m_state.chattering = false;
        }
        m_state.stabilityParameter = 1.0;
    }

    m_state.stabilityParameter = std::max(0.0, std::min(1.0, m_state.stabilityParameter));

    if (m_state.chattering && m_histIdx % 20 == 0) {
        std::cout << "[Chatter] ACTIVE at f=" << m_state.chatterFrequencyHz
                  << " Hz, growth=" << m_state.chatterGrowthRate
                  << ", stability=" << m_state.stabilityParameter
                  << std::endl;
    }
}

double ChatterDynamics::getChipThicknessModulation() const {
    return m_state.chipThicknessModulation;
}

Vec3 ChatterDynamics::getVibrationDisplacement() const {
    return m_state.displacement;
}

Vec3 ChatterDynamics::getVibrationVelocity() const {
    return m_state.velocity;
}

void ChatterDynamics::reset() {
    m_state = ChatterState();
}

} // namespace edgepredict
