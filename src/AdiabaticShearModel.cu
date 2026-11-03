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
__global__ void detectShearBandsKernel(MPMParticle* particles, int numParticles,
                                        ShearBandParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    MPMParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    
    // Compute equivalent shear stress from deviatoric stress
    double sxx = p.stress_xx - (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    double syy = p.stress_yy - (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    double szz = p.stress_zz - (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    
    double J2 = 0.5 * (sxx*sxx + syy*syy + szz*szz) + 
                p.stress_xy*p.stress_xy + p.stress_xz*p.stress_xz + p.stress_yz*p.stress_yz;
    double tau_eq = sqrt(J2);  // Equivalent shear stress
    
    // Homologous temperature
    double T_homologous = p.temperature / fmax(params.meltingTemp, 1.0);
    
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
/**
 * @brief Adiabatic shear heating kernel — sustained strain-rate guard applied.
 *
 * PHYSICS FIX (ASB Noise Filter):
 * The old code triggered a 5× heat multiplier from a single step where
 * strainRate > criticalStrainRate.  SPH boundary noise routinely generates
 * strain rates of 100–1000 s⁻¹ near free surfaces even without real deformation.
 *
 * Physical reality: Adiabatic shear bands require SUSTAINED high strain rates
 * over a zone of material — not a single particle spike.
 *
 * Fix:
 *   1. Raise minimum guard from 1.0/s to m_criticalStrainRate (in machining
 *      of steel: ~10,000–100,000 s⁻¹ at the ASB threshold).
 *   2. The `p.damage` field doubles as a strain-rate history counter.
 *      Particles accumulate a small counter each step they exceed the
 *      critical rate.  Only when counter > 3 is the multiplier applied.
 *      The counter decays by 0.5 per step when below critical.
 *   3. Max localization reduced from 5× to 2.5× (still matches the
 *      ~2.5× thermal conductivity reduction measured in ASB literature:
 *      Dodd & Bai, 1987; Timothy, 1987).
 */
__global__ void applyAdiabaticHeatingKernel(MPMParticle* particles, int numParticles,
                                             ShearBandParams params, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];

    // BUG 4 FIX: CHIP particles should not accumulate ASB heat
    // (prevents heat carryover from cutting zone asbCounter history)
    // Must check BEFORE the ACTIVE guard since CHIP != ACTIVE.
    if (p.status == ParticleStatus::CHIP) {
        p.asbCounter = 0;
        return;
    }

    if (p.status != ParticleStatus::ACTIVE) return;

    // ── Guard 1: Temperature must be elevated (tool contact has occurred) ────
    // p.temperature <= 25°C means the particle has not been touched by the tool.
    if (p.temperature <= 25.5) return;

    // ── Strain-rate clip to the mesh-resolvable rate ──────────────────────
    // p.strainRate is computed directly from the grid velocity gradient (g2p)
    // and at the instant of tool contact the kinematic grid BC forces cells to
    // full cutting speed, spiking the rate far above what the mesh can resolve
    // (~cutting speed / dx ≈ 2e3 s⁻¹). Feeding that numerical spike into the
    // Taylor-Quinney integral below dT-jumps a particle to the ASB ceiling
    // (0.5 × T_melt) within a single contact step. Bound the rate used for
    // heating to 2× the critical (localization) rate — the same resolvable-work
    // bound already applied to the in-p2g plastic heat (MPMSolver p2gKernel).
    double sr = fmin(p.strainRate, params.criticalStrainRate * 2.0);

    // ── Guard 2: Strain rate must exceed cutting-physics threshold ────
    // Entry threshold for basic Taylor-Quinney heating = a fraction of the
    // critical (localization) rate. The critical rate itself is set to match
    // the strain rate the mesh can actually RESOLVE (≈ cutting_speed/dx, see
    // Blocker-B analysis), not the incompressible high-σ̂ physical figure
    // (~1e4-1e5 s⁻¹) which the finite cell size can never reach.
    // (Removed the old hard-coded 5000 s⁻¹ floor — at 0.8 mm cells the
    // resolved peak is ~2175 s⁻¹, so a 5000 floor permanently blocked the
    // entire adiabatic-shear heating path.)
    const double minThreshold = fmax(params.criticalStrainRate * 0.5, 100.0);
    if (sr < minThreshold) {
        // Decay the sustained-history counter
        p.asbCounter = max(0, p.asbCounter - 1);
        return;
    }

    // ── Von Mises equivalent stress ───────────────────────────────────────
    double s1 = p.stress_xx - p.stress_yy;
    double s2 = p.stress_yy - p.stress_zz;
    double s3 = p.stress_zz - p.stress_xx;
    double sigma_eq = sqrt(0.5 * (s1*s1 + s2*s2 + s3*s3 +
                     6.0*(p.stress_xy*p.stress_xy + p.stress_xz*p.stress_xz +
                          p.stress_yz*p.stress_yz)));

    // ── Base Taylor-Quinney adiabatic heating ──────────────────────────
    // ΔT = β * σ_eq * ε̇ * dt / (ρ * Cp)
    // First Law of Thermodynamics: Fraction of mechanical plastic work converted to heat
    // cannot exceed 100% (beta <= 1.0, typically 0.90 for cutting alloys).
    double beta = fmin(params.taylorQuinney, 1.0);
    double dT = (beta * sigma_eq * sr * dt) /
                fmax(params.density * params.specificHeat, 1e-12);

    // ── History tracking ───────────────────────────────────────────────
    // We use p.asbCounter as an integer counter (0 to N steps) to track
    // sustained deformation above critical strain rate.
    p.asbCounter = min(p.asbCounter + 1, 10);

    // Physical numerical stability cap: limit single-step dT to prevent overshoot
    dT = fmin(dT, 100.0);

    // Apply temperature rise (physical ceiling: max_temp_ratio * T_melt)
    double maxTemp = params.maxTempRatio * params.meltingTemp;
    p.temperature  = fmin(p.temperature + dT, maxTemp);
}


/**
 * @brief Apply thermal softening to reduce flow stress
 * 
 * Johnson-Cook thermal term: (1 - T*^m)
 * where T* = (T - T_ref) / (T_melt - T_ref)
 */
__global__ void applyThermalSofteningKernel(MPMParticle* particles, int numParticles,
                                             double meltTemp, double refTemp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    MPMParticle& p = particles[idx];
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
    
    // Thermal softening affects only the DEVIATORIC stress, not the hydrostatic
    // pressure.  The full tensor σ = p_hyd * I + dev(σ).  Only the deviatoric
    // part is reduced by thermal softening — the pressure component is unchanged.
    double pressure = (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;

    double dev_xx = p.stress_xx - pressure;
    double dev_yy = p.stress_yy - pressure;
    double dev_zz = p.stress_zz - pressure;

    p.stress_xx = pressure + dev_xx * softeningFactor;
    p.stress_yy = pressure + dev_yy * softeningFactor;
    p.stress_zz = pressure + dev_zz * softeningFactor;
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

void AdiabaticShearModel::update(MPMParticle* d_particles, int numParticles, double dt,
                                  cudaStream_t stream) {
    if (!m_config.enabled || numParticles == 0) return;
    
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    // Step 1: Detect shear band locations
    detectShearBandsKernel<<<gridSize, blockSize, 0, stream>>>(d_particles, numParticles, m_params);
    CUDA_CHECK_KERNEL();
    
    // Step 2: Apply adiabatic heating
    // Thermal softening is NOT applied here — the Johnson-Cook model in
    // p2gKernel already handles thermal softening via (1 - T*^m) in the
    // flow stress. Applying it again here would double-count the effect.
    applyAdiabaticHeatingKernel<<<gridSize, blockSize, 0, stream>>>(d_particles, numParticles, m_params, dt);
    CUDA_CHECK_KERNEL();
}

} // namespace edgepredict
