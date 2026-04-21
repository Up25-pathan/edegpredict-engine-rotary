#pragma once
/**
 * @file SurfaceRoughnessPredictor.h
 * @brief Surface roughness prediction from simulation data
 *
 * Fix: predict() takes only a MachiningOutput — the previous SimulationEngine
 *      code erroneously called predict(output, config) which does not exist.
 *      The Config is consumed once in initialize() and stored internally.
 */

#include "Types.h"
#include "Config.h"
#include "IMachiningStrategy.h"
#include <vector>
#include <functional>

namespace edgepredict {

/**
 * @brief Surface roughness parameters (ISO 4287)
 */
struct SurfaceRoughnessParams {
    double Ra    = 0;   // Arithmetic average roughness (μm)
    double Rz    = 0;   // Average max height of profile (μm)
    double Rmax  = 0;   // Maximum height of profile (μm)
    double Rq    = 0;   // RMS roughness (μm)
    double Rsk   = 0;   // Skewness
    double Rku   = 0;   // Kurtosis
    double Rsm   = 0;   // Mean spacing of peaks (μm)

    // Components
    double Ra_kinematic = 0;
    double Ra_dynamic   = 0;
    double Ra_plastic   = 0;

    // Quality
    bool   meetsTolerance = false;
    double targetRa       = 0;
};

/**
 * @brief Surface profile point
 */
struct ProfilePoint {
    double position;    // Distance along profile (μm)
    double height;      // Height deviation from mean (μm)
};

/**
 * @brief Surface roughness predictor
 */
class SurfaceRoughnessPredictor {
public:
    SurfaceRoughnessPredictor();
    ~SurfaceRoughnessPredictor() = default;

    /**
     * @brief Read configuration (call once before predict()).
     *        Safe to call multiple times — subsequent calls are no-ops.
     */
    void initialize(const Config& config);

    void setStrategy(IMachiningStrategy* strategy) { m_strategy = strategy; }

    /**
     * @brief Predict roughness from current machining output.
     * FIX: single-argument version — Config is stored from initialize().
     */
    SurfaceRoughnessParams predict(const MachiningOutput& output);

    /**
     * @brief Predict roughness from explicit parameters (useful for testing).
     */
    SurfaceRoughnessParams predictFromParams(double feedPerRev,
                                              double noseRadius,
                                              double cuttingSpeed);

    void updateDynamicEffects(double toolVibrationAmplitude,
                               double vibrationFrequency);

    std::vector<ProfilePoint> generateProfileFromParticles(
        const std::vector<SPHParticle>& particles,
        const Vec3& surfaceNormal,
        double profileLength);

    SurfaceRoughnessParams calculateFromProfile(
        const std::vector<ProfilePoint>& profile);

    void setTargetRa(double targetRa) { m_targetRa = targetRa; }

    const SurfaceRoughnessParams& getLastPrediction() const {
        return m_lastPrediction;
    }

private:
    IMachiningStrategy* m_strategy = nullptr;

    double m_targetRa          = 1.6;
    double m_materialSpringback = 0.1;
    double m_bueThreshold      = 0.3;

    double m_vibrationAmplitude = 0;
    double m_vibrationFrequency = 0;
    bool   m_hasBUE             = false;
    bool   m_initialized        = false;   // guard against double-init

    SurfaceRoughnessParams m_lastPrediction;

    double calculateKinematicRoughness(double feed, double noseRadius) const;
    double calculateDynamicRoughness()  const;
    double calculatePlasticRoughness(double temperature, double stress) const;
    double estimateBUEEffect(double cuttingSpeed, double temperature) const;
};

} // namespace edgepredict