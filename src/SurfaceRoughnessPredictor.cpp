/**
 * @file SurfaceRoughnessPredictor.cpp
 * @brief Surface roughness prediction implementation
 *
 * Fix: predict() now matches the single-argument signature in the header.
 *      initialize() stores an internal flag so re-calling is safe (the
 *      SimulationEngine calls it lazily in exportResults()).
 */

#include "SurfaceRoughnessPredictor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

SurfaceRoughnessPredictor::SurfaceRoughnessPredictor() = default;

// ---------------------------------------------------------------------------
// initialize — idempotent, safe to call multiple times
// ---------------------------------------------------------------------------

void SurfaceRoughnessPredictor::initialize(const Config& config) {
    if (m_initialized) return;   // FIX: guard against double-init

    const auto& material = config.getMaterial();
    double E       = material.youngsModulus;
    double sigma_y = material.yieldStrength;

    // Elastic recovery: higher E/σy ≈ less springback
    m_materialSpringback = (E > 0) ? sigma_y / E : 0.001;

    m_bueThreshold = 0.3;

    std::cout << "[SurfaceRoughness] Initialized, target Ra = "
              << m_targetRa << " μm" << std::endl;

    m_initialized = true;
}

// ---------------------------------------------------------------------------
// predict (single-argument — FIX applied here)
// ---------------------------------------------------------------------------

/**
 * @brief Predict surface roughness from machining output.
 *
 * PHYSICS FIX: The old implementation ALWAYS used the kinematic formula
 * Ra = f²/(32r), which has no connection to the SPH simulation.
 *
 * New priority chain:
 *   1. If machiningOutput carries SPH particle data AND there are >= 100
 *      surface particles: compute Ra from the actual simulated profile.
 *      generateProfileFromParticles() scans particles for those within
 *      smoothingRadius of the target surface plane, builds a height profile,
 *      and computes all ISO 4287 parameters from it.
 *
 *   2. Fallback: kinematic formula Ra = f²/(32r) if particle data is
 *      absent or too sparse for a meaningful profile.
 *
 * Corner radius r:
 *   Read from output.toolNoseRadius if > 0; otherwise use 0.4mm (ISO standard
 *   minimum for milling inserts).
 */
SurfaceRoughnessParams SurfaceRoughnessPredictor::predict(const MachiningOutput& output) {
    SurfaceRoughnessParams params;

    // ── Priority 1: Physics-based profile from SPH particles ────────────────
    // Try to build the profile from simulated particle positions.
    // Requires >= 100 surface particles for a statistically meaningful Ra.
    if (!output.surfaceParticles.empty() &&
        output.surfaceParticles.size() >= 100) {

        // Surface normal: default to +Z (drilling / end-milling down)
        Vec3 surfNormal = (output.surfaceNormal.length() > 1e-6)
                          ? output.surfaceNormal.normalized()
                          : Vec3(0.0, 0.0, 1.0);

        auto profile = generateProfileFromParticles(
            output.surfaceParticles, surfNormal,
            output.profileLength > 0.0 ? output.profileLength : 0.004  // 4mm default
        );

        if (profile.size() >= 10) {
            params = calculateFromProfile(profile);   // all ISO 4287 params from data
            params.targetRa       = m_targetRa;
            params.meetsTolerance = (params.Ra <= m_targetRa);
            m_lastPrediction = params;
            return params;
        }
    }

    // ── Priority 2: Kinematic formula (geometric model) ──────────────────
    // Ra = f² / (32 * r_nose)   [feed f in mm, r_nose in mm, Ra in μm]
    double feed = std::max(1e-6, output.chipThickness) * 1000.0;  // m → mm

    // Tool corner/nose radius: use actual geometry if provided, else 0.4mm
    double cutterRadius = (output.toolNoseRadius > 0.0)
                          ? output.toolNoseRadius * 1000.0   // m → mm
                          : 0.4;                              // ISO minimum default

    double peripherySpeed = std::max(1.0, output.chipVelocity * 2.0 * 60.0);

    params.Ra_kinematic = calculateKinematicRoughness(feed, cutterRadius);
    params.Ra_dynamic   = calculateDynamicRoughness();
    params.Ra_plastic   = calculatePlasticRoughness(
        output.maxToolTemperature,
        output.maxToolStress);
    params.Ra_plastic  += estimateBUEEffect(peripherySpeed, output.maxToolTemperature);

    // RSS combination for independent error sources
    params.Ra = std::sqrt(
        params.Ra_kinematic * params.Ra_kinematic +
        params.Ra_dynamic   * params.Ra_dynamic   +
        params.Ra_plastic   * params.Ra_plastic);

    // Derived ISO parameters (empirical ratios for machined surfaces)
    params.Rz    = params.Ra * 4.0;
    params.Rmax  = params.Ra * 6.0;
    params.Rq    = params.Ra * 1.25;
    params.Rsk   = -0.3;                   // Slightly negative (bearing surface)
    params.Rku   =  3.0;                   // Near-Gaussian
    params.Rsm   = feed * 1000.0;          // mm → μm

    params.targetRa       = m_targetRa;
    params.meetsTolerance = (params.Ra <= m_targetRa);

    m_lastPrediction = params;
    return params;
}


// ---------------------------------------------------------------------------
// predictFromParams
// ---------------------------------------------------------------------------

SurfaceRoughnessParams SurfaceRoughnessPredictor::predictFromParams(
    double feedPerRev, double cutterRadius, double peripherySpeed) {

    SurfaceRoughnessParams params;
    params.Ra_kinematic = calculateKinematicRoughness(feedPerRev, cutterRadius);
    params.Ra_dynamic   = calculateDynamicRoughness();
    
    double plasticFactor = 0.1 / std::max(0.5, peripherySpeed / 100.0);
    params.Ra_plastic   = params.Ra_kinematic * plasticFactor;
    
    if (peripherySpeed < 50) {                       // BUE common at low speeds
        params.Ra_plastic += params.Ra_kinematic * 0.5;
    }

    params.Ra    = std::sqrt(
        params.Ra_kinematic * params.Ra_kinematic +
        params.Ra_dynamic   * params.Ra_dynamic   +
        params.Ra_plastic   * params.Ra_plastic);
    params.Rz    = params.Ra * 4.0;
    params.Rmax  = params.Ra * 6.0;
    params.Rq    = params.Ra * 1.25;
    params.Rsm   = feedPerRev * 1000.0;

    params.targetRa       = m_targetRa;
    params.meetsTolerance = (params.Ra <= m_targetRa);

    m_lastPrediction = params;
    return params;
}

// ---------------------------------------------------------------------------
// Dynamic effects
// ---------------------------------------------------------------------------

void SurfaceRoughnessPredictor::updateDynamicEffects(
    double toolVibrationAmplitude, double vibrationFrequency) {
    m_vibrationAmplitude = toolVibrationAmplitude;
    m_vibrationFrequency = vibrationFrequency;
}

// ---------------------------------------------------------------------------
// Profile generation from SPH particles
// ---------------------------------------------------------------------------

std::vector<ProfilePoint> SurfaceRoughnessPredictor::generateProfileFromParticles(
    const std::vector<SPHParticle>& particles,
    const Vec3& surfaceNormal,
    double profileLength) {

    std::vector<ProfilePoint> profile;
    if (particles.empty()) return profile;

    // Profile direction perpendicular to normal
    Vec3 profileDir;
    if (std::abs(surfaceNormal.z) < 0.9) {
        profileDir = Vec3(-surfaceNormal.y, surfaceNormal.x, 0);
    } else {
        profileDir = Vec3(1, 0, 0);
    }
    double nm = std::sqrt(profileDir.x * profileDir.x +
                          profileDir.y * profileDir.y);
    if (nm > 1e-10) { profileDir.x /= nm; profileDir.y /= nm; }

    // Surface particles (within threshold)
    const double threshold = 0.0001;
    std::vector<const SPHParticle*> surfParts;
    for (const auto& p : particles) {
        double d = std::abs(p.x * surfaceNormal.x +
                            p.y * surfaceNormal.y +
                            p.z * surfaceNormal.z);
        if (d < threshold) surfParts.push_back(&p);
    }

    int    numSamples   = static_cast<int>(profileLength * 1000);
    double sampleSpacing = profileLength / std::max(1, numSamples);

    for (int i = 0; i < numSamples; ++i) {
        double pos = i * sampleSpacing;
        ProfilePoint point;
        point.position = pos * 1000.0;  // m → μm

        double height = 0, totalW = 0;
        for (const auto* p : surfParts) {
            double proj = p->x * profileDir.x + p->y * profileDir.y;
            double diff = std::abs(proj - pos);
            if (diff < sampleSpacing * 2) {
                double w = 1.0 / (diff + 0.001);
                double h = p->x * surfaceNormal.x +
                           p->y * surfaceNormal.y +
                           p->z * surfaceNormal.z;
                height += h * w;
                totalW += w;
            }
        }
        if (totalW > 0) point.height = (height / totalW) * 1e6; // m → μm
        profile.push_back(point);
    }

    // Remove mean line
    double meanH = 0;
    for (const auto& pt : profile) meanH += pt.height;
    if (!profile.empty()) meanH /= profile.size();
    for (auto& pt : profile) pt.height -= meanH;

    return profile;
}

// ---------------------------------------------------------------------------
// calculateFromProfile
// ---------------------------------------------------------------------------

SurfaceRoughnessParams SurfaceRoughnessPredictor::calculateFromProfile(
    const std::vector<ProfilePoint>& profile) {

    SurfaceRoughnessParams params;
    if (profile.size() < 10) return params;

    // Ra
    double sumAbs = 0;
    for (const auto& p : profile) sumAbs += std::abs(p.height);
    params.Ra = sumAbs / profile.size();

    // Rq
    double sumSq = 0;
    for (const auto& p : profile) sumSq += p.height * p.height;
    params.Rq = std::sqrt(sumSq / profile.size());

    // Rmax
    double maxH = profile[0].height, minH = profile[0].height;
    for (const auto& p : profile) {
        maxH = std::max(maxH, p.height);
        minH = std::min(minH, p.height);
    }
    params.Rmax = maxH - minH;

    // Rz (5-peak / 5-valley average)
    std::vector<double> hs;
    hs.reserve(profile.size());
    for (const auto& p : profile) hs.push_back(p.height);
    std::sort(hs.begin(), hs.end());
    int n = std::min(5, (int)(hs.size() / 10));
    if (n > 0) {
        double sumPk = 0, sumVl = 0;
        for (int k = 0; k < n; ++k) {
            sumVl += hs[k];
            sumPk += hs[hs.size() - 1 - k];
        }
        params.Rz = (sumPk - sumVl) / n;
    }

    // Skewness & kurtosis
    if (params.Rq > 0) {
        double s3 = 0, s4 = 0;
        for (const auto& p : profile) {
            double h3 = p.height * p.height * p.height;
            s3 += h3;
            s4 += h3 * p.height;
        }
        params.Rsk = s3 / (profile.size() * std::pow(params.Rq, 3));
        params.Rku = s4 / (profile.size() * std::pow(params.Rq, 4));
    }

    // Mean spacing (zero-crossing)
    int crossings = 0;
    for (size_t i = 1; i < profile.size(); ++i) {
        if ((profile[i-1].height < 0) != (profile[i].height < 0)) ++crossings;
    }
    if (crossings > 0) {
        double totalLen = profile.back().position - profile.front().position;
        params.Rsm = totalLen / (crossings / 2.0);
    }

    params.targetRa       = m_targetRa;
    params.meetsTolerance = (params.Ra <= m_targetRa);

    m_lastPrediction = params;
    return params;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

double SurfaceRoughnessPredictor::calculateKinematicRoughness(
    double feed, double cutterRadius) const {
    if (cutterRadius <= 0) cutterRadius = 0.4;
    // Classic formula: Ra = f² / (32·r)  [units: mm, result in μm]
    // Applies to turning inserts and rotary tool corner radii.
    return (feed * feed) / (32.0 * cutterRadius) * 1000.0;
}

double SurfaceRoughnessPredictor::calculateDynamicRoughness() const {
    if (m_vibrationAmplitude <= 0) return 0.0;
    double freqFactor = 1.0 / (1.0 + m_vibrationFrequency / 1000.0);
    return m_vibrationAmplitude * 0.25 * freqFactor;
}

double SurfaceRoughnessPredictor::calculatePlasticRoughness(
    double temperature, double stress) const {
    double basePlastic  = 0.1;
    double tempFactor   = temperature / 1000.0;
    double stressFactor = stress / 1e9;
    return basePlastic * (1.0 + tempFactor + stressFactor);
}

double SurfaceRoughnessPredictor::estimateBUEEffect(
    double cuttingSpeed, double temperature) const {
    const double meltingPoint = 1660.0;
    double tempRatio = temperature / meltingPoint;
    bool inBUEZone = (tempRatio > 0.2 && tempRatio < 0.4) && (cuttingSpeed < 80.0);
    if (!inBUEZone) return 0.0;

    double tempSev  = 1.0 - std::abs(tempRatio - 0.3) / 0.1;
    double speedSev = 1.0 - cuttingSpeed / 100.0;
    double severity = std::max(0.0, tempSev) * std::max(0.0, speedSev);
    return severity * 2.0;  // μm
}

} // namespace edgepredict