/**
 * @file AdiabaticShearModel.cuh
 * @brief Adiabatic Shear Band model for serrated chip formation
 * 
 * In machining of Ti-6Al-4V and Inconel, chips often form in a serrated
 * pattern due to adiabatic shear bands. These occur when:
 * 
 * 1. Thermal softening exceeds strain hardening (dσ/dε < 0)
 * 2. Heat cannot dissipate fast enough (adiabatic condition)
 * 3. Localized shear occurs in narrow bands (~10-50μm)
 * 4. Temperature in band approaches 0.9 * T_melt
 * 
 * This results in periodic "teeth" in the chip, which is a signature
 * of machining difficult-to-cut materials.
 */

#pragma once

#include "Types.h"
// MPMSolver.cuh removed to break circular dependency (MPMParticle is now in Types.h)

namespace edgepredict {

// AdiabaticShearConfig is now defined in Types.h
struct AdiabaticShearConfig;

/**
 * @brief GPU kernel parameters for shear band computation
 */
struct ShearBandParams {
    double criticalStrainRate;
    double softeningThreshold;
    double taylorQuinney;
    double thermalDiffusivity;
    double bandWidth;
    double maxTempRatio;
    double meltingTemp;
    double specificHeat;
    double density;
};

/**
 * @brief Adiabatic Shear Band model manager
 * 
 * Detects and applies adiabatic shear band effects to SPH particles.
 * Key effects:
 * - Localized heating in shear bands
 * - Reduced flow stress in bands (thermal softening)
 * - Periodic segmentation of chip
 */
class AdiabaticShearModel {
public:
    AdiabaticShearModel();
    ~AdiabaticShearModel();
    
    /**
     * @brief Initialize with configuration
     */
    void initialize(const AdiabaticShearConfig& config, double meltTemp, 
                    double specificHeat, double density);
    
    /**
     * @brief Update shear band effects on particles
     * @param particles Device pointer to SPH particles
     * @param numParticles Number of particles
     * @param dt Time step
     */
    void update(MPMParticle* d_particles, int numParticles, double dt);
    
    /**
     * @brief Check if shear band model is enabled
     */
    bool isEnabled() const { return m_config.enabled; }
    
    /**
     * @brief Get estimated shear band spacing
     * Based on Komanduri-Hou model: λ = 2π * sqrt(k*ρ*Cp / τ*γ̇)
     */
    double getSegmentationSpacing() const { return m_segmentationSpacing; }
    
private:
    AdiabaticShearConfig m_config;
    ShearBandParams m_params;
    double m_segmentationSpacing = 0;
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Detect shear band onset based on thermal softening criterion
 */
__global__ void detectShearBandsKernel(MPMParticle* particles, int numParticles,
                                        ShearBandParams params);

/**
 * @brief Apply adiabatic heating in shear bands
 * Temperature rise: ΔT = (β * τ * γ̇ * dt) / (ρ * Cp)
 */
__global__ void applyAdiabaticHeatingKernel(MPMParticle* particles, int numParticles,
                                             ShearBandParams params, double dt);

/**
 * @brief Apply thermal softening to flow stress
 */
__global__ void applyThermalSofteningKernel(MPMParticle* particles, int numParticles,
                                             double meltTemp, double refTemp);

} // namespace edgepredict
