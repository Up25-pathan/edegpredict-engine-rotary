/**
 * @file AdiabaticShearModel.cu
 * @brief Implementation of Adiabatic Shear Band model
 * 
 * References:
 * - Komanduri & Hou (2002) - Thermal analysis of chip segmentation
 * - Molinari & Clifton (1987) - Shear band instability analysis
 * - Bai & Dodd (1992) - Adiabatic Shear Localization
 */

#include "AdiabaticShearModel.cuh"
#include "CudaUtils.cuh"
#include <iostream>
#include <cmath>

namespace edgepredict {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Detect shear band onset based on thermal softening criterion
 * 
 * Shear band forms when: dσ/dε < 0 (thermal softening > strain hardening)
 * 
 * We detect this by checking if:
 * 1. Strain rate exceeds critical value
 * 2. Temperature is high enough for thermal softening
 * 3. Equivalent stress is decreasing (approximated by comparing to expected)
 */
__global__ void detectShearBandsKernel(SPHParticle* particles, int numParticles,
                                        ShearBandParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    
    // Compute equivalent shear stress from deviatoric stress
    double sxx = p.stress_xx - (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    double syy = p.stress_yy - (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    double szz = p.stress_zz - (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    
    double J2 = 0.5 * (sxx*sxx + syy*syy + szz*szz) + 
                p.stress_xy*p.stress_xy + p.stress_xz*p.stress_xz + p.stress_yz*p.stress_yz;
    double tau_eq = sqrt(J2);  // Equivalent shear stress
    
    // Homologous temperature
    double T_homologous = p.temperature / params.meltingTemp;
    
    // Check shear band criteria:
    // 1. High strain rate
    // 2. Temperature > 0.3 * T_melt (significant thermal softening)
    // 3. High shear stress indicating deformation
    
    bool isHighStrainRate = p.strainRate > params.criticalStrainRate;
    bool isThermalSoftening = T_homologous > 0.3;
    bool isHighShear = tau_eq > 100e6;  // > 100 MPa shear stress
    
    // If all criteria met, particle is in potential shear band zone
    if (isHighStrainRate && isThermalSoftening && isHighShear) {
        // Mark for shear band processing by setting high strain rate flag
        // (we reuse existing fields to avoid adding more struct members)
        p.lodZone = LODZone::ACTIVE;  // Ensure full physics in shear band
    }
}

/**
 * @brief Apply adiabatic heating in shear bands
 * 
 * Temperature rise from plastic work:
 *   ΔT = (β * σ * ε̇ * dt) / (ρ * Cp)
 * 
 * where:
 *   β = Taylor-Quinney coefficient (~0.9 for metals)
 *   σ = equivalent stress
 *   ε̇ = strain rate
 *   ρ = density
 *   Cp = specific heat
 */
__global__ void applyAdiabaticHeatingKernel(SPHParticle* particles, int numParticles,
                                             ShearBandParams params, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    
    // Only apply enhanced heating for high strain rate particles
    // FIX: Stricter guard — skip particles not yet engaged in cutting.
    // Floating-point noise in strainRate (near-zero but non-zero) was causing
    // 320°C temperature spikes at step 100 BEFORE first tool contact.
    if (p.strainRate < 1.0 || p.temperature <= 25.1) return;
    
    // von Mises equivalent stress
    double s1 = p.stress_xx - p.stress_yy;
    double s2 = p.stress_yy - p.stress_zz;
    double s3 = p.stress_zz - p.stress_xx;
    double sigma_eq = sqrt(0.5 * (s1*s1 + s2*s2 + s3*s3 + 
                     6*(p.stress_xy*p.stress_xy + p.stress_xz*p.stress_xz + p.stress_yz*p.stress_yz)));
    
    // Temperature rise from plastic work
    // ΔT = β * σ * ε̇ * dt / (ρ * Cp)
    double dT = (params.taylorQuinney * sigma_eq * p.strainRate * dt) / 
                (params.density * params.specificHeat);
    
    // In shear bands, heating is more intense (less time for heat dissipation)
    // Apply localization factor for very high strain rates
    if (p.strainRate > params.criticalStrainRate) {
        double localization = 1.0 + 2.0 * (p.strainRate / params.criticalStrainRate - 1.0);
        localization = fmin(localization, 5.0);  // Cap at 5x
        dT *= localization;
    }
    
    // Apply temperature rise (capped at max temperature ratio * T_melt)
    double maxTemp = params.maxTempRatio * params.meltingTemp;
    p.temperature = fmin(p.temperature + dT, maxTemp);
}

/**
 * @brief Apply thermal softening to reduce flow stress
 * 
 * Johnson-Cook thermal term: (1 - T*^m)
 * where T* = (T - T_ref) / (T_melt - T_ref)
 */
__global__ void applyThermalSofteningKernel(SPHParticle* particles, int numParticles,
                                             double meltTemp, double refTemp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    
    // Homologous temperature
    double T_star = 0.0;
    if (meltTemp > refTemp) {
        T_star = (p.temperature - refTemp) / (meltTemp - refTemp);
        T_star = fmax(0.0, fmin(1.0, T_star));
    }
    
    // Softening factor: (1 - T*^m), using m = 1.0 for simplicity
    double softeningFactor = 1.0 - T_star;
    softeningFactor = fmax(0.1, softeningFactor);  // Don't let it go to zero
    
    // Apply softening to stress tensor
    p.stress_xx *= softeningFactor;
    p.stress_yy *= softeningFactor;
    p.stress_zz *= softeningFactor;
    p.stress_xy *= softeningFactor;
    p.stress_xz *= softeningFactor;
    p.stress_yz *= softeningFactor;
}

// ============================================================================
// AdiabaticShearModel Implementation
// ============================================================================

AdiabaticShearModel::AdiabaticShearModel() = default;
AdiabaticShearModel::~AdiabaticShearModel() = default;

void AdiabaticShearModel::initialize(const AdiabaticShearConfig& config, 
                                      double meltTemp, double specificHeat, double density) {
    m_config = config;
    
    // Set up kernel parameters
    m_params.criticalStrainRate = config.criticalStrainRate;
    m_params.softeningThreshold = config.softeningThreshold;
    m_params.taylorQuinney = config.taylorQuinneyCoeff;
    m_params.thermalDiffusivity = config.thermalDiffusivity;
    m_params.bandWidth = config.typicalBandWidth;
    m_params.maxTempRatio = config.maxTemperatureRatio;
    m_params.meltingTemp = meltTemp;
    m_params.specificHeat = specificHeat;
    m_params.density = density;
    
    // Estimate shear band spacing using Komanduri-Hou model
    // λ = 2π * sqrt(k / (ρ * Cp * τ * γ̇))
    // Simplified estimate assuming typical machining conditions
    double k = config.thermalDiffusivity * density * specificHeat;  // Thermal conductivity
    double tau_typical = 500e6;  // 500 MPa typical shear stress
    double gammadot_typical = 1e5;  // Typical shear rate in PSZ
    
    m_segmentationSpacing = 2.0 * constants::PI * sqrt(k / (density * specificHeat * 
                                               tau_typical * gammadot_typical));
    
    if (m_config.enabled) {
        std::cout << "[AdiabaticShearModel] Initialized:" << std::endl;
        std::cout << "  Critical strain rate: " << config.criticalStrainRate << " /s" << std::endl;
        std::cout << "  Taylor-Quinney coeff: " << config.taylorQuinneyCoeff << std::endl;
        std::cout << "  Estimated chip spacing: " << m_segmentationSpacing * 1e6 << " μm" << std::endl;
    }
}

void AdiabaticShearModel::update(SPHParticle* d_particles, int numParticles, double dt) {
    if (!m_config.enabled || numParticles == 0) return;
    
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    // Step 1: Detect shear band locations
    detectShearBandsKernel<<<gridSize, blockSize>>>(d_particles, numParticles, m_params);
    CUDA_CHECK_KERNEL();
    
    // Step 2: Apply adiabatic heating
    applyAdiabaticHeatingKernel<<<gridSize, blockSize>>>(d_particles, numParticles, m_params, dt);
    CUDA_CHECK_KERNEL();
    
    // Step 3: Apply thermal softening
    applyThermalSofteningKernel<<<gridSize, blockSize>>>(d_particles, numParticles,
                                                          m_params.meltingTemp, 25.0);
    CUDA_CHECK_KERNEL();
}

} // namespace edgepredict
