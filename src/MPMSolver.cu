/**
 * @file MPMSolver.cu
 * @brief Material Point Method (MPM) CUDA implementation with APIC & Johnson-Cook
 */

#include "MPMSolver.cuh"
#include "Config.h"
#include "PhysicsConstants.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace edgepredict {

// Gravity constant (Z-up frame, drilling downward)
static __device__ const double GRAVITY_Z = -9.81;

// --- Math Helpers ---

__device__ inline void getWeightsAndGradients(double xp, double invDx, int& baseNode, double w[3], double dw[3]) {
    double x_frac = xp * invDx;
    baseNode = (int)floor(x_frac - 0.5);
    double d0 = x_frac - baseNode;
    double d1 = d0 - 1.0;
    double d2 = d0 - 2.0;

    w[0] = 0.5 * (1.5 - d0) * (1.5 - d0);
    w[1] = 0.75 - d1 * d1;
    w[2] = 0.5 * (1.5 + d2) * (1.5 + d2);

    dw[0] = -(1.5 - d0) * invDx;
    dw[1] = -2.0 * d1 * invDx;
    dw[2] = (1.5 + d2) * invDx;
}

__device__ inline double det3x3(const double A[9]) {
    return A[0]*(A[4]*A[8] - A[5]*A[7]) -
           A[1]*(A[3]*A[8] - A[5]*A[6]) +
           A[2]*(A[3]*A[7] - A[4]*A[6]);
}

__device__ inline void mul3x3(const double A[9], const double B[9], double C[9]) {
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            C[i*3+j] = A[i*3+0]*B[0*3+j] + A[i*3+1]*B[1*3+j] + A[i*3+2]*B[2*3+j];
        }
    }
}

__device__ inline void inv3x3(const double A[9], double invA[9]) {
    double det = det3x3(A);
    if (fabs(det) < 1e-12) {
        invA[0] = invA[4] = invA[8] = 1.0;
        invA[1] = invA[2] = invA[3] = 0.0;
        invA[5] = invA[6] = invA[7] = 0.0;
        return;
    }
    double invDet = 1.0 / det;
    invA[0] =  (A[4]*A[8] - A[5]*A[7]) * invDet;
    invA[1] = -(A[1]*A[8] - A[2]*A[7]) * invDet;
    invA[2] =  (A[1]*A[5] - A[2]*A[4]) * invDet;
    invA[3] = -(A[3]*A[8] - A[5]*A[6]) * invDet;
    invA[4] =  (A[0]*A[8] - A[2]*A[6]) * invDet;
    invA[5] = -(A[0]*A[5] - A[2]*A[3]) * invDet;
    invA[6] =  (A[3]*A[7] - A[4]*A[6]) * invDet;
    invA[7] = -(A[0]*A[7] - A[1]*A[6]) * invDet;
    invA[8] =  (A[0]*A[4] - A[1]*A[3]) * invDet;
}

// --- Kernels ---

/**
 * @brief Apply viscoelastic damping and Dirichlet BCs in the fixture zone.
 *
 * Bottommost grid layer (gz == 0): hard velocity zero — represents rigid vise jaw.
 * Above layers within fixtureThickness: quadratic-tapered velocity damping via
 *   v *= 1 / (1 + sigma * dt)
 *   where sigma = dampingCoeff * (1 - zRel)^2
 *
 * This smoothly absorbs incoming stress waves and prevents non-physical reflections.
 */
__global__ void applyFixtureBCKernel(MPMGridNode* grid, int3 dims,
    Vec3 domainMin, double dx,
    Vec3 workpieceMin, Vec3 workpieceMax,
    double fixtureThickness, double dampingCoeff, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numNodes = dims.x * dims.y * dims.z;
    if (idx >= numNodes) return;

    int rem = idx % (dims.x * dims.y);
    int gx = rem % dims.x;
    int gy = rem / dims.x;
    int gz = idx / (dims.x * dims.y);

    double worldX = gx * dx + domainMin.x;
    double worldY = gy * dx + domainMin.y;
    double worldZ = gz * dx + domainMin.z;

    MPMGridNode& node = grid[idx];
    if (node.mass < 1e-12) return;

    // Check if node is within lateral workpiece bounds (with 1-cell margin)
    if (worldX < workpieceMin.x - dx || worldX > workpieceMax.x + dx ||
        worldY < workpieceMin.y - dx || worldY > workpieceMax.y + dx) {
        return;
    }

    // 1. Rigid Vise Baseplate: bottom layer of workpiece is clamped rigidly (zero velocity)
    if (worldZ <= workpieceMin.z + 0.5 * dx) {
        node.vx = 0.0; node.vy = 0.0; node.vz = 0.0;
        node.px = 0.0; node.py = 0.0; node.pz = 0.0;
        return;
    }

    // 2. Viscoelastic absorption layer in the bottom fixture zone:
    double zRel = (worldZ - workpieceMin.z) / fixtureThickness;
    if (zRel >= 0.0 && zRel <= 1.0) {
        double sigma = dampingCoeff * (1.0 - zRel) * (1.0 - zRel);
        double factor = 1.0 / (1.0 + sigma * dt);
        node.vx *= factor;
        node.vy *= factor;
        node.vz *= factor;
        node.px *= factor;
        node.py *= factor;
        node.pz *= factor;
    }

    // 3. Lateral Vise Jaws: CNC vises clamp along Y (front and back jaws)
    bool atMinYJaw = (worldY <= workpieceMin.y + 0.5 * dx);
    bool atMaxYJaw = (worldY >= workpieceMax.y - 0.5 * dx);
    if ((atMinYJaw || atMaxYJaw) && worldZ <= workpieceMax.z - (workpieceMax.z - workpieceMin.z) * 0.3) {
        node.vy = 0.0;
        node.py = 0.0;
    }
}

__global__ void resetGridKernel(MPMGridNode* grid, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    grid[idx].mass = 0;
    grid[idx].vx = 0; grid[idx].vy = 0; grid[idx].vz = 0;
    grid[idx].px = 0; grid[idx].py = 0; grid[idx].pz = 0;
    grid[idx].fx = 0; grid[idx].fy = 0; grid[idx].fz = 0;
    grid[idx].isTool = 0;
    grid[idx].tvx = 0; grid[idx].tvy = 0; grid[idx].tvz = 0;
}

__global__ void p2gKernel(MPMParticle* particles, int numParticles, MPMGridNode* grid, int3 dimensions, MPMKernelConfig config, Vec3 domainMin, double dt) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= numParticles) return;
    
    MPMParticle& p = particles[pIdx];
    if (p.status == ParticleStatus::INACTIVE) return;

    // [FIX Robust NaN hygiene] A NaN position would fail every comparison below
    // (the kill-zone check sees positions "inside" the grid, indexing OOB grid
    // cells) and poison the whole transfer. Kill the particle before it can.
    if (!(p.x == p.x) || !(p.y == p.y) || !(p.z == p.z)) {
        p.status = ParticleStatus::INACTIVE;
        return;
    }

    // Boundary Kill Zone check to prevent CUDA out-of-bounds crash
    int cx = (int)floor((p.x - domainMin.x) * config.invDx);
    int cy = (int)floor((p.y - domainMin.y) * config.invDx);
    int cz = (int)floor((p.z - domainMin.z) * config.invDx);
    if (cx < 1 || cx >= dimensions.x - 2 || 
        cy < 1 || cy >= dimensions.y - 2 || 
        cz < 1 || cz >= dimensions.z - 2) {
        p.status = ParticleStatus::INACTIVE;
        return;
    }

    // LOD: FAR zone particles only contribute mass and momentum (kinematic), no stress
    bool skipStress = (p.lodZone == LODZone::ZONE_FAR);
    // FIXED_BOUNDARY particles: kinematic only (clamped, no deformation)
    // CHIP particles: advected only (stress-free debris)
    bool isFixed = (p.status == ParticleStatus::FIXED_BOUNDARY);
    bool isChip = (p.status == ParticleStatus::CHIP);
    if (isFixed || isChip) skipStress = true;

    // ── Kinematic Moving Window (System 3) ─────────────────────────────────
    // Far-field ACTIVE material is FROZEN: it contributes grid MASS only (no
    // momentum, no stress), statically pinning the bulk workpiece outside the
    // moving window around the tool tip. Only in-window (ACTIVE/NEAR) particles
    // drive the deformation physics. Chips are never frozen (they advect away).
    bool frozen = (p.status == ParticleStatus::ACTIVE && p.lodZone == LODZone::ZONE_FAR);
    if (frozen) skipStress = true;

    int bx, by, bz;
    double wx[3], wy[3], wz[3];
    double dwx[3], dwy[3], dwz[3];

    getWeightsAndGradients(p.x - domainMin.x, config.invDx, bx, wx, dwx);
    getWeightsAndGradients(p.y - domainMin.y, config.invDx, by, wy, dwy);
    getWeightsAndGradients(p.z - domainMin.z, config.invDx, bz, wz, dwz);

    // --- Internal force computation (stress-based) ---
    // Only for ACTIVE and NEAR zone particles. FAR particles skip stress
    // (they have negligible effect on tool engagement).
    double sigma[9] = {0};
    double V0 = p.volume;
    double J = 1.0;  // Default: identity deformation (no volume change)

    if (!skipStress && p.status == ParticleStatus::ACTIVE) {
        // Elastic deformation gradient: F_e = F * inv(F_p)
        double invFp[9];
        inv3x3(p.F_p, invFp);
        double Fe[9];
        mul3x3(p.F, invFp, Fe);

        J = det3x3(Fe);
        // [FIX Metals are nearly incompressible.] Clamp J to prevent infinite
        // pressures (volumetric locking) during massive element distortion.
        // Softened to a smooth saturation (unit slope near J=1, asymptotic to
        // [0.9, 1.1]) instead of the old hard bang-bang snap: the hard clamp
        // converted grid-scale ringing into a saturated alternating hydrostatic
        // limit-cycle that pumped the whole field to ±clamp (the "explosion").
        {
            double Jabs = J;
            double softening = edgepredict::physics::J_CLAMP_SOFTENING;
            if (Jabs > 1.0) J = 1.0 + softening * tanh((Jabs - 1.0) / softening);
            else            J = 1.0 - softening * tanh((1.0 - Jabs) / softening);
        }

        // [FIX Hydrostatic blow-up] The J-clamp above bounds ONLY the volume
        // ratio, not individual F_e components: heavily sheared elements keep
        // Fe[0]·Fe[4]·Fe[8] ≈ 1 while off-diagonal entries diverge, making
        // B = F_e·F_eᵀ (and hence tr σ) reach 1e14 Pa — the pathology the
        // mean-stress clamp has to paper over downstream. Kill it at the
        // source: NaN-guard + bound F_e components and the trace of B before
        // they ever reach the stress / P2G force path.
        // Numerical hygiene: sanitize NaN/Inf entries to identity without
        // non-physical component-wise clipping of genuine large-strain shear
        for (int i = 0; i < 9; ++i) {
            double a = Fe[i];
            if (!(a == a) || fabs(a) > 1e4) {
                Fe[i] = (i % 4 == 0) ? 1.0 : 0.0;
            }
        }

        // Lamé parameters from config (Neo-Hookean)
        double mu = config.youngsModulus / (2.0 * (1.0 + config.poissonsRatio));
        double lambda = config.youngsModulus * config.poissonsRatio /
                        ((1.0 + config.poissonsRatio) * (1.0 - 2.0 * config.poissonsRatio));

        double B[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B[i*3+j] = Fe[i*3+0]*Fe[j*3+0] + Fe[i*3+1]*Fe[j*3+1] + Fe[i*3+2]*Fe[j*3+2];
            }
        }
        // Objective emergency guard for elastic distortional energy:
        // Plastic radial return naturally relaxes deviatoric elastic strain into F_p.
        // Cap is set generously (50.0) purely to stop unphysical numerical explosions.
        double trB = B[0] + B[4] + B[8];
        const double trBEmergencyCap = 50.0;
        if (!(trB == trB) || trB > trBEmergencyCap) {
            double s = (trB > 3.0) ? sqrt((trBEmergencyCap - 3.0) / (trB - 3.0)) : 1.0;
            B[0] = 1.0 + (B[0] - 1.0) * s;
            B[4] = 1.0 + (B[4] - 1.0) * s;
            B[8] = 1.0 + (B[8] - 1.0) * s;
            B[1] *= s; B[2] *= s; B[3] *= s; B[5] *= s; B[6] *= s; B[7] *= s;
        }

        // Compressible Neo-Hookean Cauchy stress:
        //   σ = μ/J · (B - I) + λ·ln(J)/J · I
        //   = μ/J · B + (λ·ln(J)/J - μ/J) · I
        double J_safe = fmax(J, 1e-12);
        double Jpow = (J_safe - 1.0);
        double mu_over_J = mu / J_safe;
        for (int i = 0; i < 9; ++i) {
            sigma[i] = mu_over_J * B[i];
            if (i % 4 == 0) {
                sigma[i] += lambda * Jpow - mu_over_J;
            }
        }

        // ================================================================
        // Johnson-Cook Plasticity Return Mapping (with kinematic hardening)
        // ================================================================
        double tr_sigma = sigma[0] + sigma[4] + sigma[8];
        // FIX: persist hydrostatic pressure (mean stress) for exports/triaxiality.
        // Guard against the pathological deviatoric-distortion case: det(Fe) stays
        // within [0.9,1.1] (J-clamp) yet individual Fe components can diverge on
        // massively sheared elements, making tr_sigma (and hence p.pressure)
        // non-finite/e-24-order. A mean stress beyond ~±10 GPa is not physical for
        // this steel (yield ≈ 2 GPa), so clamp and NaN-guard here. The SAME clamped
        // meanStress is substituted back into the diagonal stress used for the P2G
        // internal force and JC triaxiality (see below) — otherwise only the export
        // is clamped and the force path can still explode.
        double meanStress = tr_sigma / 3.0;

        // [FIX Field-blowing on contact] Linear BULK viscosity added to the mean
        // stress that feeds the P2G internal force:
        //   sigma_damp = -coef * rho*c*dx * div(v)
        // div(v) taken from the APIC affine matrix (one-step lag). Dissipative
        // only for volumetric (dilational/breathing/checkerboard) modes; zero for
        // rigid motion and simple shear, so cutting forces are essentially
        // unchanged. This damps the grid-scale energy that the contact step-input
        // would otherwise pump into a field-wide saturated pressure pattern.
        double divV = p.C[0] + p.C[4] + p.C[8];
        if (!(divV == divV) || fabs(divV) > 1.0e6) divV = 0.0;
        double rho_p = p.mass / fmax(V0 * J, 1.0e-30);
        double cp = sqrt((lambda + 2.0 * mu) / fmax(rho_p, 1.0e-3));
        meanStress -= config.linearViscosityCoef * rho_p * cp * config.dx * divV;
        // [FIX Robust pressure] Replace the old hard ±1e10 Pa clamp (~15× yield,
        // ~9× the J-clamped Neo-Hookean limit for this steel) with a
        // strength-scaled cap. Bulk mean stress beyond ~3 GPa is not physical for
        // 4140; anything larger is the F_e/tr(σ) runaway that the distortional
        // caps above now stop at the source. The SAME clamped meanStress still
        // feeds the P2G internal force and JC triaxiality, so no unphysical
        // spike can reach the force path.
        double meanStressLimit = fmax(2.5e9, 3.5 * config.jc_A);
        if (!(meanStress == meanStress) || fabs(meanStress) > meanStressLimit)
            meanStress = (meanStress >= 0) ? meanStressLimit : -meanStressLimit;
        p.pressure = meanStress;
        double s[9];
        for (int i = 0; i < 9; ++i) s[i] = sigma[i];
        s[0] -= tr_sigma / 3.0;
        s[4] -= tr_sigma / 3.0;
        s[8] -= tr_sigma / 3.0;

        // Rebuild the total stress from the (trace-free) deviatoric part plus the
        // CLAMPED mean stress. This guarantees the unclamped tr_sigma never reaches
        // the P2G internal force even in the elastic branch (q <= Y), which does not
        // pass through the radial-return rebuild below.
        sigma[0] = s[0] + meanStress;
        sigma[4] = s[4] + meanStress;
        sigma[8] = s[8] + meanStress;
        sigma[1] = s[1]; sigma[2] = s[2]; sigma[3] = s[3];
        sigma[5] = s[5]; sigma[6] = s[6]; sigma[7] = s[7];

        // Shifted deviatoric stress (for Bauschinger / kinematic hardening)
        double eta_x  = s[0] - p.backstress_xx;
        double eta_y  = s[4] - p.backstress_yy;
        double eta_z  = s[8] - p.backstress_zz;
        double eta_xy = s[1] - p.backstress_xy;
        double eta_xz = s[2] - p.backstress_xz;
        double eta_yz = s[5] - p.backstress_yz;

        // Equivalent von Mises stress on SHIFTED stress: q = sqrt(1.5 * η_ij * η_ij)
        double q = sqrt(1.5 * (eta_x*eta_x + eta_y*eta_y + eta_z*eta_z +
                      2.0 * (eta_xy*eta_xy + eta_xz*eta_xz + eta_yz*eta_yz)));

        // Johnson-Cook flow stress
        double T_star = (p.temperature - config.ambientTemp) /
                        fmax(config.meltingPoint - config.ambientTemp, 1.0);
        if (T_star < 0) T_star = 0;
        if (T_star > 1) T_star = 1;

        double edot_star = fmax(p.strainRate / fmax(config.referenceStrainRate, 1e-12), 1e-5);
        double thermal_factor = 1.0 - pow(T_star, config.jc_m);
        thermal_factor = fmax(thermal_factor, 0.05); // Maintain 5% residual strength at melting point to prevent explosion
        double Y = (config.jc_A + config.jc_B * pow(p.plasticStrain, config.jc_n)) *
                   (1.0 + config.jc_C * log(edot_star)) *
                   thermal_factor;

        double deltaEps = 0.0;
        if (q > Y && q > 1e-12) {
            double scale = Y / q;
            // Update deviatoric stress: s_new = α + Y/q * η
            s[0] = p.backstress_xx + scale * eta_x;
            s[4] = p.backstress_yy + scale * eta_y;
            s[8] = p.backstress_zz + scale * eta_z;
            s[1] = p.backstress_xy + scale * eta_xy;
            s[2] = p.backstress_xz + scale * eta_xz;
            s[5] = p.backstress_yz + scale * eta_yz;
            for (int i = 0; i < 9; ++i) {
                sigma[i] = s[i];
                // Use the CLAMPED hydrostatic mean stress here: tr_sigma can run
                // away to ~1e14 Pa on massively sheared elements (det(Fe) stays in
                // [0.9,1.1] while Fe components diverge), and this sigma feeds the
                // P2G internal force directly. Clamping meanStress bounds the whole
                // tensor (the deviatoric part is already capped at the yield stress).
                if (i % 4 == 0) sigma[i] += meanStress;
            }

            // Equivalent plastic strain increment (radial return magnitude)
            deltaEps = (q - Y) / fmax(3.0 * mu, 1e-12);
            
            // [FIX] Limit maximum plastic strain increment per micro-step 
            // to prevent explicit integration explosions.
            if (deltaEps > edgepredict::physics::MAX_PLASTIC_STRAIN_INCREMENT) 
                deltaEps = edgepredict::physics::MAX_PLASTIC_STRAIN_INCREMENT;

            if (deltaEps > 0) {
                // Evolve backstress (Prager-Ziegler kinematic hardening)
                if (config.kinematicHardeningModulus > 0.0) {
                    double n_scale = config.kinematicHardeningModulus * deltaEps / q;
                    p.backstress_xx += n_scale * eta_x;
                    p.backstress_yy += n_scale * eta_y;
                    p.backstress_zz += n_scale * eta_z;
                    p.backstress_xy += n_scale * eta_xy;
                    p.backstress_xz += n_scale * eta_xz;
                    p.backstress_yz += n_scale * eta_yz;
                }

                p.plasticStrain += deltaEps;

                // ============================================================
                // UPDATE PLASTIC DEFORMATION GRADIENT
                //   F_p_new = (I + (3/2) * Δε_pl / q * s_dev) * F_p
                //   where s_dev is the deviatoric stress (flow direction)
                //   This ensures F_e = F * inv(F_p) only contains elastic strain
                // ============================================================
                {
                    // Multiplicative Radial-Return Plasticity (Volume-preserving)
                    double s_norm = sqrt(s[0]*s[0] + s[4]*s[4] + s[8]*s[8] + 2.0*(s[1]*s[1] + s[2]*s[2] + s[5]*s[5]));
                    double alpha = (s_norm > 1e-12) ? (1.5 * deltaEps / s_norm) : 0.0;
                    
                    double I_alpha_s[9] = {0};
                    for(int i=0; i<9; ++i) I_alpha_s[i] = alpha * s[i];
                    I_alpha_s[0] += 1.0; I_alpha_s[4] += 1.0; I_alpha_s[8] += 1.0;
                    
                    double newFp[9] = {0};
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            newFp[i*3+j] = 0;
                            for (int k = 0; k < 3; ++k) {
                                newFp[i*3+j] += I_alpha_s[i*3+k] * p.F_p[k*3+j];
                            }
                        }
                    }
                    
                    // Enforce incompressibility det(F_p) = 1
                    double detFp = newFp[0]*(newFp[4]*newFp[8] - newFp[5]*newFp[7]) -
                                   newFp[1]*(newFp[3]*newFp[8] - newFp[5]*newFp[6]) +
                                   newFp[2]*(newFp[3]*newFp[7] - newFp[4]*newFp[6]);
                    double cbrtFp = cbrt(fmax(detFp, 1e-12));
                    for(int i=0; i<9; ++i) p.F_p[i] = newFp[i] / cbrtFp;
                }

                // ============================================================
                // THERMAL EVOLUTION: Plastic work → heat
                //   ΔT = β * σ_eq * Δε_pl / (ρ * Cp)
                //   σ_eq = Y (yield stress after radial return)
                //   β = Taylor-Quinney coefficient (~0.9 for metals)
                //
                // PHYSICS FIX (melting-clamp runaway): this base Taylor-Quinney
                // heat is *in addition* to the dedicated adiabatic-shear model
                // (AdiabaticShearModel.cu, which now carries the properly bounded
                // plastic-work→heat for the shear zone). The deltaEps cap of 0.05
                // per micro-step corresponds to a strain rate of 1e5 s⁻¹ — an
                // order of magnitude above the ~2175 s⁻¹ the 0.8 mm mesh can
                // actually resolve — so it accumulated thousands of K and slammed
                // the whole workpiece onto the melting clamp (1314 °C). Bound the
                // per-step rise by the *resolvable* plastic work instead.
                // ============================================================
                // PHYSICS FIX (melting-clamp runaway): this base Taylor-Quinney
                // heat is disabled if the particle is actively undergoing adiabatic
                // shear localization. The ASB model provides the physical heat.
                double dT = 0.0;
                
                if (p.asbCounter == 0) {
                    dT = config.taylorQuinney * Y * deltaEps /
                         fmax(config.physicalDensity * config.specificHeat, 1e-12);

                    // Physical per-step plastic work ceiling consistent with the
                    // resolvable strain rate (≈ ε̇_res·dt). deltaEps has already been
                    // capped at 0.05 above; tighten to the resolvable Δε_pl so a
                    // single step cannot over-deposit heat (~0.5 K/step at 2175 s⁻¹).
                    double resolvableDeltaEps = fmax(p.strainRate * config.dt, 1e-12);
                    double deltaEpsEff = fmin(deltaEps, resolvableDeltaEps * 5.0);
                    double dTBase = config.taylorQuinney * Y * deltaEpsEff /
                                   fmax(config.physicalDensity * config.specificHeat, 1e-12);
                    if (dTBase < dT) dT = dTBase;

                    // Safety ceiling: allow rapid adiabatic shear heating, bounded by melting point
                    if (dT > 25.0) dT = 25.0;
                }

                p.temperature += dT;

                // Clamp temperature to physical range
                if (p.temperature > config.meltingPoint)
                    p.temperature = config.meltingPoint;

                // ============================================================
                // DAMAGE: Johnson-Cook failure criterion
                //   ε_f = [D1 + D2·exp(D3·σ*)][1 + D4·ln(ε̇*)][1 + D5·T*]
                //   σ* = pressure / σ_eq  (stress triaxiality)
                //   D = Σ(Δε_pl / ε_f), failure when D > damageThreshold
                // ============================================================
                // JC failure incubation uses the clamped mean stress (same bound as
                // p.pressure) so an unphysical hydrostatic spike cannot collapse
                // ε_f = D1 + D2·exp(D3·σ*) to zero and force spurious chip formation.
                double pressure = meanStress;
                double triaxiality = pressure / fmax(q, 1e-12);

                double eps_f = (config.jc_D1 + config.jc_D2 * exp(config.jc_D3 * triaxiality)) *
                               (1.0 + config.jc_D4 * log(fmax(edot_star, 1.0))) *
                               (1.0 + config.jc_D5 * T_star);

                if (eps_f > 1e-12) {
                    // [FIX Damage avalanche] Mirror the heat path: the raw Δε cap
                    // (0.05 per sub-step ≈ 1e5 s⁻¹) is far above the ~2e3 s⁻¹ the
                    // 0.8 mm grid can resolve, so a few macro steps could ratchet a
                    // whole region past the failure threshold at once. Accumulate
                    // damage at the RESOLVABLE equivalent plastic-work rate instead.
                    double resolvableDeltaEps = fmax(p.strainRate * dt, 1e-12);
                    double deltaEpsDam = fmin(deltaEps, resolvableDeltaEps * edgepredict::physics::DAMAGE_INCREMENT_FACTOR);
                    p.damage += deltaEpsDam / eps_f;

                    // Gradual stress degradation (CDM): when D > 0.7, reduce
                    // stress linearly so it reaches zero at D = damageThreshold.
                    // This avoids the numerical shock of binary CHIP transition.
                    if (p.damage > 0.7 * config.damageThreshold) {
                        double degrade = (config.damageThreshold - p.damage) /
                                         (config.damageThreshold * 0.3);
                        degrade = fmax(0.0, fmin(1.0, degrade));
                        for (int i = 0; i < 9; ++i) sigma[i] *= degrade;
                        // Also record the residual stress at this point
                        p.residualStress = fmax(p.residualStress,
                            sqrt(1.5 * (s[0]*s[0] + s[4]*s[4] + s[8]*s[8] +
                                 2.0 * (s[1]*s[1] + s[2]*s[2] + s[5]*s[5]))));
                    }

                    // Failure: particle transitions to chip or inactive
                    if (p.damage >= config.damageThreshold) {
                        p.status = ParticleStatus::CHIP;
                        skipStress = true;
                    }
                }
            }
        }

        // Store updated stress back to particle
        p.stress_xx = sigma[0]; p.stress_yy = sigma[4]; p.stress_zz = sigma[8];
        p.stress_xy = sigma[1]; p.stress_xz = sigma[2]; p.stress_yz = sigma[5];
    }

    // ================================================================
    // Particle → Grid transfer (APIC)
    // ================================================================
    for (int i = 0; i < 3; ++i) {
        int gx = bx + i;
        if (gx < 0 || gx >= dimensions.x) continue;
        for (int j = 0; j < 3; ++j) {
            int gy = by + j;
            if (gy < 0 || gy >= dimensions.y) continue;
            for (int k = 0; k < 3; ++k) {
                int gz = bz + k;
                if (gz < 0 || gz >= dimensions.z) continue;

                double weight = wx[i] * wy[j] * wz[k];
                double dW_dx = dwx[i] * wy[j] * wz[k];
                double dW_dy = wx[i] * dwy[j] * wz[k];
                double dW_dz = wx[i] * wy[j] * dwz[k];

                int gIdx = gx + dimensions.x * (gy + dimensions.y * gz);

                double dpos_x = (gx * config.dx + domainMin.x) - p.x;
                double dpos_y = (gy * config.dx + domainMin.y) - p.y;
                double dpos_z = (gz * config.dx + domainMin.z) - p.z;

                // APIC affine velocity. Chips advect with the pure grid velocity:
                // their C matrix is stale (g2p skips CHIP particles), so reusing
                // it would inject a frozen affine velocity into the debris field.
                double affine_vx = 0.0, affine_vy = 0.0, affine_vz = 0.0;
                if (!isChip) {
                    affine_vx = p.C[0]*dpos_x + p.C[1]*dpos_y + p.C[2]*dpos_z;
                    affine_vy = p.C[3]*dpos_x + p.C[4]*dpos_y + p.C[5]*dpos_z;
                    affine_vz = p.C[6]*dpos_x + p.C[7]*dpos_y + p.C[8]*dpos_z;
                }

                atomicAdd(&grid[gIdx].mass, weight * p.mass);
                // FIXED_BOUNDARY / frozen-FAR particles contribute mass but NOT
                // momentum (v=0), pinning the bulk workpiece as kinematic support.
                if (!isFixed && !frozen) {
                    atomicAdd(&grid[gIdx].px, weight * p.mass * (p.vx + affine_vx));
                    atomicAdd(&grid[gIdx].py, weight * p.mass * (p.vy + affine_vy));
                    atomicAdd(&grid[gIdx].pz, weight * p.mass * (p.vz + affine_vz));
                }

                // Internal force from stress divergence
                double fx_int = 0, fy_int = 0, fz_int = 0;
                if (!skipStress) {
                    fx_int = -V0 * J * (sigma[0]*dW_dx + sigma[1]*dW_dy + sigma[2]*dW_dz);
                    fy_int = -V0 * J * (sigma[3]*dW_dx + sigma[4]*dW_dy + sigma[5]*dW_dz);
                    fz_int = -V0 * J * (sigma[6]*dW_dx + sigma[7]*dW_dy + sigma[8]*dW_dz);
                }

                // External body force (from applyExternalForce, e.g., cutting reaction)
                double fx_ext = weight * p.ext_fx;
                double fy_ext = weight * p.ext_fy;
                double fz_ext = weight * p.ext_fz;

                atomicAdd(&grid[gIdx].fx, fx_int + fx_ext);
                atomicAdd(&grid[gIdx].fy, fy_int + fy_ext);
                atomicAdd(&grid[gIdx].fz, fz_int + fz_ext);
            }
        }
    }
}

__global__ void updateGridKernel(MPMGridNode* grid, int numNodes, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    MPMGridNode& g = grid[idx];
    if (g.mass > 1e-12) {
        // BUG 3 FIX: Gravity along -Z (tool drills downward in Z-up frame)
        // Was: g.py += g.mass * -9.81 * dt;  // Wrong axis for Z-up drilling
        g.pz += g.mass * GRAVITY_Z * dt;
        
        g.px += g.fx * dt;
        g.py += g.fy * dt;
        g.pz += g.fz * dt;
        
        g.vx = g.px / g.mass;
        g.vy = g.py / g.mass;
        g.vz = g.pz / g.mass;
    }
}

__global__ void rasterizeToolKernel(FEMNodeGPU* toolNodes, int numToolNodes, MPMGridNode* grid, int3 dimensions, MPMKernelConfig config, Vec3 domainMin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numToolNodes) return;
    
    FEMNodeGPU& node = toolNodes[idx];
    
    int gx = (int)round((node.x - domainMin.x) * config.invDx);
    int gy = (int)round((node.y - domainMin.y) * config.invDx);
    int gz = (int)round((node.z - domainMin.z) * config.invDx);
    
    // Spread tool influence to a 2x2x2 neighborhood to ensure solid boundary
    for(int i=0; i<2; ++i) {
        int cx = gx + i;
        if (cx < 0 || cx >= dimensions.x) continue;
        for(int j=0; j<2; ++j) {
            int cy = gy + j;
            if (cy < 0 || cy >= dimensions.y) continue;
            for(int k=0; k<2; ++k) {
                int cz = gz + k;
                if (cz < 0 || cz >= dimensions.z) continue;
                
                int gIdx = cx + dimensions.x * (cy + dimensions.y * cz);
                grid[gIdx].isTool = 1;
                grid[gIdx].tvx = node.vx;
                grid[gIdx].tvy = node.vy;
                grid[gIdx].tvz = node.vz;
            }
        }
    }
}

__global__ void applyGridCollisionKernel(MPMGridNode* grid, int numNodes, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    MPMGridNode& g = grid[idx];
    if (g.mass > 1e-12 && g.isTool == 1) {
        // ── FIX F4 (Chisel-Punch → Shear Chip): One-Sided Signorini Contact ──────
        //
        // OLD code: "Sticky/Kinematic collision: overwrite velocity with tool velocity"
        //   g.vx = g.tvx; g.vy = g.tvy; g.vz = g.tvz;
        //
        // WHY THAT WAS WRONG:
        //   For chisel-tip nodes, tool velocity ≈ (0, 0, -feed_rate) — pure axial
        //   translation, no rotational component. After G2P interpolation, every
        //   workpiece particle near the tool got the same downward velocity with no
        //   tangential shear. This produced pure compression (chisel punch) instead
        //   of helical chip shear. Von Mises stress saturated at 1.93 GPa (pressure-
        //   dominated), temperature pinned at 730°C, and 3D showed explosion not chips.
        //
        // CORRECT PHYSICS: A drill is a ROTATING tool that shears material tangentially.
        //   At the grid level this means: a workpiece grid node touching the tool
        //   must NOT move INTO the tool (no-penetration), but IS free to slide
        //   tangentially (enabling the shear strain that makes chips, not plugs).
        //
        // IMPLEMENTATION: Signorini (one-sided) contact constraint.
        //   1. Compute relative velocity: dv = v_grid - v_tool
        //   2. If dv points INTO the tool (dv · n̂ < 0), remove that component.
        //   3. Leave the tangential component intact — this carries the rotational
        //      cutting speed that drives shear chip formation.
        // Enforce one-sided no-penetration boundary condition against the advancing tool:
        // In the feed direction (Z), workpiece material cannot penetrate upward into the descending drill point.
        if (g.vz > g.tvz) {
            g.vz = g.tvz;
            g.pz = g.mass * g.vz;
        }
    }
}


__global__ void interpolateToolNodesKernel(FEMNodeGPU* out, const FEMNodeGPU* start,
    const FEMNodeGPU* end, int numNodes, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    out[idx].x = start[idx].x + alpha * (end[idx].x - start[idx].x);
    out[idx].y = start[idx].y + alpha * (end[idx].y - start[idx].y);
    out[idx].z = start[idx].z + alpha * (end[idx].z - start[idx].z);
    out[idx].ox = start[idx].ox + alpha * (end[idx].ox - start[idx].ox);
    out[idx].oy = start[idx].oy + alpha * (end[idx].oy - start[idx].oy);
    out[idx].oz = start[idx].oz + alpha * (end[idx].oz - start[idx].oz);
    // BUG FIX (Blocker-A velocity build-up): the rigid-body velocity (linear
    // feed + ω×r rotation) is constant over the macro step; carry it straight
    // through to the interpolated tool nodes. Previously vx/vy/vz were never
    // written here, leaving uninitialized/stale values that rasterizeToolKernel
    // then spread onto the MPM grid as a wrong tool-velocity BC — starving the
    // workpiece of cutting speed (~500× too slow), crushing strain rate so the
    // adiabatic-shear plastic heat could never fire.
    out[idx].vx = end[idx].vx;
    out[idx].vy = end[idx].vy;
    out[idx].vz = end[idx].vz;
    out[idx].fx = end[idx].fx;
    out[idx].fy = end[idx].fy;
    out[idx].fz = end[idx].fz;
    out[idx].temperature = end[idx].temperature;
}

__global__ void g2pKernel(MPMParticle* particles, int numParticles, MPMGridNode* grid, int3 dimensions, MPMKernelConfig config, double dt, Vec3 domainMin) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= numParticles) return;
    
    MPMParticle& p = particles[pIdx];
    if (p.status == ParticleStatus::INACTIVE) return;

    // FIXED_BOUNDARY: keep zero velocity and fixed position
    if (p.status == ParticleStatus::FIXED_BOUNDARY) {
        p.vx = 0; p.vy = 0; p.vz = 0;
        return;
    }

    // ── Kinematic Moving Window (System 3): freeze far-field material ──────
    // Active particles outside the moving window (ZONE_FAR) are frozen: zero
    // velocity, no advection. The bulk workpiece stays a rigid kinematic
    // support; only the window around the tool tip does deforming physics.
    // Chips (debris) still advect away from the tool.
    if (p.status == ParticleStatus::ACTIVE && p.lodZone == LODZone::ZONE_FAR) {
        p.vx = 0; p.vy = 0; p.vz = 0;
        p.fx = 0; p.fy = 0; p.fz = 0;
        p.strainRate = 0;
        return;
    }

    int bx, by, bz;
    double wx[3], wy[3], wz[3];
    double dwx[3], dwy[3], dwz[3];

    getWeightsAndGradients(p.x - domainMin.x, config.invDx, bx, wx, dwx);
    getWeightsAndGradients(p.y - domainMin.y, config.invDx, by, wy, dwy);
    getWeightsAndGradients(p.z - domainMin.z, config.invDx, bz, wz, dwz);
    
    double vx = 0, vy = 0, vz = 0;
    double B[9] = {0}; 
    double L[9] = {0}; 
    
    for (int i = 0; i < 3; ++i) {
        int gx = bx + i;
        if (gx < 0 || gx >= dimensions.x) continue;
        for (int j = 0; j < 3; ++j) {
            int gy = by + j;
            if (gy < 0 || gy >= dimensions.y) continue;
            for (int k = 0; k < 3; ++k) {
                int gz = bz + k;
                if (gz < 0 || gz >= dimensions.z) continue;
                
                int gIdx = gx + dimensions.x * (gy + dimensions.y * gz);
                MPMGridNode& g = grid[gIdx];
                if (g.mass < 1e-12) continue;
                
                double weight = wx[i] * wy[j] * wz[k];
                double dW_dx = dwx[i] * wy[j] * wz[k];
                double dW_dy = wx[i] * dwy[j] * wz[k];
                double dW_dz = wx[i] * wy[j] * dwz[k];
                
                vx += weight * g.vx;
                vy += weight * g.vy;
                vz += weight * g.vz;
                
                double dpos_x = (gx * config.dx + domainMin.x) - p.x;
                double dpos_y = (gy * config.dx + domainMin.y) - p.y;
                double dpos_z = (gz * config.dx + domainMin.z) - p.z;
                
                B[0] += weight * g.vx * dpos_x; B[1] += weight * g.vx * dpos_y; B[2] += weight * g.vx * dpos_z;
                B[3] += weight * g.vy * dpos_x; B[4] += weight * g.vy * dpos_y; B[5] += weight * g.vy * dpos_z;
                B[6] += weight * g.vz * dpos_x; B[7] += weight * g.vz * dpos_y; B[8] += weight * g.vz * dpos_z;
                
                L[0] += g.vx * dW_dx; L[1] += g.vx * dW_dy; L[2] += g.vx * dW_dz;
                L[3] += g.vy * dW_dx; L[4] += g.vy * dW_dy; L[5] += g.vy * dW_dz;
                L[6] += g.vz * dW_dx; L[7] += g.vz * dW_dy; L[8] += g.vz * dW_dz;
            }
        }
    }
    
    p.vx = vx; p.vy = vy; p.vz = vz;

    // [FIX Robust NaN/Inf hygiene] The grid velocity is AMS-scaled and can be
    // unbounded if any upstream kernel corrupted it. Guard the interpolated
    // velocity: NaN/Inf → zero, and cap speed far above any physical machining
    // velocity (~2 m/s) so it can never seed a field-wide blow-up.
    {
        const double vCap = 200.0;  // pure safety guard, ≫ physical speeds
        double v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
        if (!(v2 == v2) || v2 > vCap * vCap) {
            double s = (v2 > 0.0 && v2 == v2) ? (vCap / sqrt(v2)) : 0.0;
            p.vx *= s; p.vy *= s; p.vz *= s;
        }
    }

    // [FIX NaN position guard] Advect from the (finite) pre-update position and
    // NaN-guard the result — a NaN position later fails the p2g kill-zone
    // comparisons (all false) and indexes grid cells out of bounds.
    double nxp = p.x + p.vx * dt, nyp = p.y + p.vy * dt, nzp = p.z + p.vz * dt;
    bool badPos = !(nxp == nxp) || !(nyp == nyp) || !(nzp == nzp) ||
                  fabs(nxp) > 1e6 || fabs(nyp) > 1e6 || fabs(nzp) > 1e6;
    if (badPos) {
        // Numerically corrupt step — pin the particle at its old state.
        p.vx = p.vy = p.vz = 0.0;
    } else {
        p.x = nxp; p.y = nyp; p.z = nzp;
    }

    // CHIP particles: advect but don't update deformation gradient
    if (p.status == ParticleStatus::CHIP) {
        p.strainRate = 0;
        return;
    }
    
    // APIC inertia tensor inverse D_p^{-1} for quadratic B-splines is 4 / dx^2 (Jiang et al. 2015)
    double D_inv = 4.0 * config.invDx * config.invDx;
    for(int i=0; i<9; ++i) p.C[i] = B[i] * D_inv;

    // BUG 3 FIX: Zero APIC C matrix for ZONE_FAR particles.
    // Far-field particles carry angular momentum purely from SPH
    // advection/diffusion of the velocity field (not from tool contact),
    // causing the bulk workpiece to rotate. They should advect
    // kinematically but not participate in APIC angular transfer.
    if (p.lodZone == LODZone::ZONE_FAR) {
        for (int i = 0; i < 9; ++i) p.C[i] = 0.0;
    } else {
        // P3 FIX: Physics-based APIC C-matrix decay.
        // Decay rate based on material damping, not arbitrary 0.99.
        // For elastic wave propagation, high-frequency numerical noise should
        // decay at a rate proportional to the bulk viscosity coefficient.
        // Decay factor = exp(-damping_coeff * dt) where damping_coeff ~ 10-100/s
        double apiDecay = exp(-edgepredict::physics::FIXTURE_DAMPING_COEFF * 1e-3 * dt);
        // Manual clamp instead of std::clamp (not allowed in __global__ without --expt-relaxed-constexpr)
        if (apiDecay < 0.95) apiDecay = 0.95;
        else if (apiDecay > 0.999) apiDecay = 0.999;
        for (int i = 0; i < 9; ++i) p.C[i] *= apiDecay;
    }
    
    double I_Ldt[9];
    for(int i=0; i<9; ++i) {
        double v_inc = L[i] * dt;
        if (i%4 == 0) I_Ldt[i] = 1.0; else I_Ldt[i] = 0.0;
        // [FIX J-drift / field saturation] The multiplicative update below
        // compounds EVERY substep; raw L*dt from the (un-damped, grid-ringing)
        // velocity field let det(F) drift out of the physical band so every
        // particle hit the mean-stress clamp and the whole field "boiled".
        // Bound the per-step increment (~0.05 ≫ any legit strain at these dt)
        // so J can only move a small, physical amount per step.
        if (!(v_inc == v_inc)) v_inc = 0.0;
        if (v_inc >  0.05) v_inc =  0.05;
        if (v_inc < -0.05) v_inc = -0.05;
        I_Ldt[i] += v_inc;
    }
    
    double F_new[9];
    mul3x3(I_Ldt, p.F, F_new);
    // [FIX F clamp] NaN/corrupt deformation gradients feed directly into the
    // F_e → B → tr(σ) blow-up path next p2g. NaN entries → identity; bound the
    // magnitude so a single bad element cannot inject ~1e14 Pa into the mesh.
    for (int i = 0; i < 9; ++i) {
        double a = F_new[i];
        if (!(a == a)) {
            a = (i % 4 == 0) ? 1.0 : 0.0;
        } else if (a > 100.0) {
            a = 100.0;
        } else if (a < -100.0) {
            a = -100.0;
        }
        p.F[i] = a;
    }
    
    p.strainRate = sqrt(2.0/3.0 * (L[0]*L[0] + L[4]*L[4] + L[8]*L[8] + 0.5*(L[1]+L[3])*(L[1]+L[3]) + 0.5*(L[2]+L[6])*(L[2]+L[6]) + 0.5*(L[5]+L[7])*(L[5]+L[7])));
}

// ============================================================================
// External Force Kernel
// ============================================================================

__global__ void applyExternalForceKernel(MPMParticle* particles, int numParticles,
    Vec3 center, double radius, Vec3 force) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE && p.status != ParticleStatus::CHIP) return;

    double dx = p.x - center.x;
    double dy = p.y - center.y;
    double dz = p.z - center.z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);

    if (dist < radius) {
        // Linear falloff: 1 at center, 0 at radius
        double w = 1.0 - dist / radius;
        p.ext_fx = force.x * w;
        p.ext_fy = force.y * w;
        p.ext_fz = force.z * w;
    }
}

// ============================================================================
// Plane Cutting Kernel
// ============================================================================

__global__ void cutParticlesKernel(MPMParticle* particles, int numParticles,
    Vec3 planeNormal, double planeDist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE && p.status != ParticleStatus::CHIP) return;

    // Remove particle if it's on the "positive" side of the plane
    double signedDist = planeNormal.x * p.x + planeNormal.y * p.y + planeNormal.z * p.z - planeDist;
    if (signedDist > 0) {
        p.status = ParticleStatus::INACTIVE;
    }
}

// ============================================================================
// LOD Classification Kernel
// ============================================================================

__global__ void lodClassifyKernel(MPMParticle* particles, int numParticles,
    const Vec3* toolPosPtr, double activeRadius, double nearRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE && p.status != ParticleStatus::CHIP) return;

    // Tool position is read through a pointer each kernel launch so the CUDA
    // graph replay sees the CURRENT position, not the one baked at record time.
    const Vec3 toolPos = *toolPosPtr;

    double dx = p.x - toolPos.x;
    double dy = p.y - toolPos.y;
    double dz = p.z - toolPos.z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);

    if (dist <= activeRadius) {
        p.lodZone = LODZone::ACTIVE;
    } else if (dist <= nearRadius) {
        p.lodZone = LODZone::ZONE_NEAR;
    } else {
        p.lodZone = LODZone::ZONE_FAR;
    }
}

// ============================================================================
// Reset External Force Kernel
// ============================================================================

__global__ void resetExternalForceKernel(MPMParticle* particles, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    particles[idx].ext_fx = 0;
    particles[idx].ext_fy = 0;
    particles[idx].ext_fz = 0;
}

// ============================================================================
// Particle Bounds Update Kernel
// ============================================================================

__global__ void updateParticleBoundsKernel(MPMParticle* particles, int numParticles,
    double* minX, double* minY, double* minZ,
    double* maxX, double* maxY, double* maxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE && p.status != ParticleStatus::CHIP) return;

    atomicMinDouble(minX, p.x);
    atomicMinDouble(minY, p.y);
    atomicMinDouble(minZ, p.z);
    atomicMaxDouble(maxX, p.x);
    atomicMaxDouble(maxY, p.y);
    atomicMaxDouble(maxZ, p.z);
}

// ============================================================================
// Coupling Point Sync Kernel (Device-side thermal coupling with CFD)
// ============================================================================

__global__ void syncMPMCouplingPointsKernel(
    const MPMParticle* particles, CouplingPoint* out, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    out[idx].x = particles[idx].x;
    out[idx].y = particles[idx].y;
    out[idx].z = particles[idx].z;
    out[idx].temperature = particles[idx].temperature;
}

// ============================================================================
// Particle Stats Kernel (max stress, max temp, kinetic energy)
// ============================================================================

__global__ void computeParticleStatsKernel(MPMParticle* particles, int numParticles,
    double* maxStress, double* maxTemp, double* kinEnergy, double* internalEnergy) {
    __shared__ double sMaxStress[256];
    __shared__ double sMaxTemp[256];
    __shared__ double sKinEnergy[256];
    __shared__ double sIntEnergy[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double localMaxStress = 0.0;
    double localMaxTemp = 0.0;
    double localKinEnergy = 0.0;
    double localIntEnergy = 0.0;

    if (idx < numParticles) {
        MPMParticle& p = particles[idx];
        if (p.status == ParticleStatus::ACTIVE || p.status == ParticleStatus::CHIP) {
            // Von Mises stress from deviatoric stress
            double trace = p.stress_xx + p.stress_yy + p.stress_zz;
            double sxx = p.stress_xx - trace / 3.0;
            double syy = p.stress_yy - trace / 3.0;
            double szz = p.stress_zz - trace / 3.0;
            double sigma_vm = sqrt(1.5 * (sxx*sxx + syy*syy + szz*szz +
                2.0*(p.stress_xy*p.stress_xy + p.stress_xz*p.stress_xz + p.stress_yz*p.stress_yz)));
            localMaxStress = sigma_vm;
            localMaxTemp = p.temperature;
            localKinEnergy = 0.5 * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
            // Internal energy: elastic strain energy + plastic work
            // Elastic strain energy density = sigma_vm² / (6 * G) where G = shear modulus
            double elasticEnergy = (sigma_vm * sigma_vm * p.mass) / (6.0 * 80e9); // G ≈ 80 GPa for steel
            // Plastic work = V * sum(sigma_vm * d_eps_p). We approximate with total plastic strain
            double plasticWork = p.volume * p.plasticStrain * sigma_vm;
            localIntEnergy = elasticEnergy + plasticWork;
        }
    }

    sMaxStress[tid] = localMaxStress;
    sMaxTemp[tid] = localMaxTemp;
    sKinEnergy[tid] = localKinEnergy;
    sIntEnergy[tid] = localIntEnergy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sMaxStress[tid + s] > sMaxStress[tid]) sMaxStress[tid] = sMaxStress[tid + s];
            if (sMaxTemp[tid + s] > sMaxTemp[tid]) sMaxTemp[tid] = sMaxTemp[tid + s];
            sKinEnergy[tid] += sKinEnergy[tid + s];
            sIntEnergy[tid] += sIntEnergy[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxDouble(maxStress, sMaxStress[0]);
        atomicMaxDouble(maxTemp, sMaxTemp[0]);
        atomicAdd(kinEnergy, sKinEnergy[0]);
        atomicAdd(internalEnergy, sIntEnergy[0]);
    }
}

__global__ void countMeltingParticlesKernel(MPMParticle* particles, int numParticles,
    double meltingPoint, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        MPMParticle& p = particles[idx];
        if (p.status == ParticleStatus::ACTIVE &&
            p.temperature >= meltingPoint - 1.0) {
            atomicAdd(count, 1);
        }
    }
}

__global__ void countParticleStatusKernel(MPMParticle* particles, int numParticles,
    int* activeCount, int* chipCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        MPMParticle& p = particles[idx];
        if (p.status == ParticleStatus::ACTIVE) {
            atomicAdd(activeCount, 1);
        } else if (p.status == ParticleStatus::CHIP) {
            atomicAdd(chipCount, 1);
        }
    }
}

__global__ void avgSurfaceTempKernel(MPMParticle* particles, int numParticles,
    double* sumTemp, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        MPMParticle& p = particles[idx];
        if (p.status != ParticleStatus::INACTIVE && p.temperature > 30.0) {
            atomicAdd(sumTemp, p.temperature);
            atomicAdd(count, 1);
        }
    }
}

// --- MPMSolver Methods ---

MPMSolver::MPMSolver() : m_amsMassScalingFactor(1.0) {}
MPMSolver::~MPMSolver() { freeMemory(); }

bool MPMSolver::initialize(const Config& config) {
    m_mainConfig = &config;

    const auto& sph = config.getSPH();
    const auto& mat = config.getMaterial();
    const auto& tm  = config.getToolMaterial();

    // P11: MPM particle capacity now comes from config (sph_parameters.
    // max_particles). It was a fixed 5M member default regardless of config,
    // so a 500k-particle config still allocated a 5M-particle GPU buffer.
    m_maxParticles = std::max(100, sph.maxParticles);

    // ── Auto-size domain from workpiece bounds + tool envelope ────────────
    // OLD: hardcoded 20cm cube m_domainMin = Vec3(-0.1,-0.1,-0.1) wasted 99.9%
    // of memory/compute on empty cells.
    // NEW: Compute tight bounds from actual workpiece geometry with margin for
    // chip debris ejection. Falls back to 20cm cube if geometry unavailable.
    {
        const auto& j = config.getJson();
        double toolDiameter = 0.010; // 10mm default

        // Try to get tool diameter from config
        if (j.contains("machining_parameters")) {
            toolDiameter = j["machining_parameters"].value("tool_diameter_mm", 10.0) / 1000.0;
        }

        // Workpiece bounds from config — prefer the actual workpiece_geometry
        // box (width/height/depth) so the domain tightly encloses the particle
        // block instead of wasting grid cells on a 40mm default.
        // The box is centred on Z=0 with its top face at Z=0 (main.cpp places
        // workMin = (-w/2,-h/2,-depth), workMax = (w/2,h/2,0)).
        {
            double wMm = 15.0, hMm = 15.0, dMm = 15.0;
            bool haveGeo = false;
            if (j.contains("workpiece_geometry")) {
                const auto& wg = j["workpiece_geometry"];
                if (wg.is_object()) {
                    wMm = wg.value("width_mm", 15.0);
                    hMm = wg.value("height_mm", 15.0);
                    dMm = wg.value("depth_mm", 15.0);
                    if (wMm > 0 && hMm > 0 && dMm > 0) haveGeo = true;
                }
            }
            double depthMm = dMm;
            if (j.contains("machining_parameters")) {
                auto& mp = j["machining_parameters"];
                if (!haveGeo) depthMm = mp.value("depth_of_cut_mm", 15.0);
            }
            if (!haveGeo) {
                // Fallback: 40mm default cube
                double wrkSizeMm = 40.0;
                if (j.contains("machining_parameters")) {
                    auto& mp = j["machining_parameters"];
                    wrkSizeMm = mp.value("workpiece_size_mm", 40.0);
                }
                wMm = hMm = wrkSizeMm;
            }
            double halfX = wMm / 2.0 / 1000.0;
            double halfY = hMm / 2.0 / 1000.0;
            Vec3 wpMin(-halfX, -halfY, -(depthMm + 2.0) / 1000.0);
            Vec3 wpMax( halfX,  halfY,  0.005); // 5mm above surface for chips

            // Margin: 2× tool diameter around workpiece for chip ejection envelope
            double margin = 2.0 * toolDiameter;
            m_domainMin = Vec3(wpMin.x - margin, wpMin.y - margin, wpMin.z - margin);
            m_domainMax = Vec3(wpMax.x + margin, wpMax.y + margin, wpMax.z + margin * 0.5);

            std::cout << "[MPMSolver] Auto-sized domain: ["
                      << m_domainMin.x*1e3 << "," << m_domainMin.y*1e3 << ","
                      << m_domainMin.z*1e3 << "] → ["
                      << m_domainMax.x*1e3 << "," << m_domainMax.y*1e3 << ","
                      << m_domainMax.z*1e3 << "] mm" << std::endl;
        }
    }

    // ── Decouple grid dx from smoothing radius ──────────────────────────────
    // OLD: m_config.dx = max(smoothingRadius * 2, 0.0005) — refining particles
    // forced smaller dx which forced smaller dt (CFL). New: independent config.
    if (sph.gridSpacing > 0.0) {
        m_config.dx = sph.gridSpacing;
        std::cout << "[MPMSolver] Grid spacing (decoupled): "
                  << m_config.dx * 1e3 << " mm" << std::endl;
    } else {
        m_config.dx = std::max(sph.smoothingRadius * 2.0, 0.0005);
    }
    m_config.invDx = 1.0 / m_config.dx;

    m_gridDimX = (int)((m_domainMax.x - m_domainMin.x) * m_config.invDx) + 1;
    m_gridDimY = (int)((m_domainMax.y - m_domainMin.y) * m_config.invDx) + 1;
    m_gridDimZ = (int)((m_domainMax.z - m_domainMin.z) * m_config.invDx) + 1;
    m_numGridNodes = m_gridDimX * m_gridDimY * m_gridDimZ;

    std::cout << "[MPMSolver] Grid: " << m_gridDimX << "×" << m_gridDimY
              << "×" << m_gridDimZ << " = " << m_numGridNodes
              << " nodes (dx=" << m_config.dx*1e3 << "mm)" << std::endl;

    // --- Populate kernel config from Config object ---
    m_config.youngsModulus  = mat.youngsModulus;
    m_config.poissonsRatio  = mat.poissonsRatio;
    m_config.yieldStrength  = mat.yieldStrength;

    m_config.physicalDensity = mat.density;
    m_config.density        = mat.density * sph.numericalMassScalingFactor;
    m_config.specificHeat   = mat.specificHeat;
    m_config.taylorQuinney  = 0.9;
    m_config.meltingPoint   = mat.meltingPoint;
    m_config.ambientTemp    = config.getMachining().ambientTemperature;

    if (sph.numericalMassScalingFactor > 1.0) {
        std::cout << "[MPMSolver] Numerical mass scaling: "
                  << sph.numericalMassScalingFactor
                  << "x (CFL dt x" << std::sqrt(sph.numericalMassScalingFactor)
                  << ", thermal density unchanged)" << std::endl;
    }

    m_config.jc_A = mat.jc_A;
    m_config.jc_B = mat.jc_B;
    m_config.jc_n = mat.jc_n;
    m_config.jc_C = mat.jc_C;
    m_config.jc_m = mat.jc_m;

    m_config.jc_D1 = sph.jc_D1;
    m_config.jc_D2 = sph.jc_D2;
    m_config.jc_D3 = sph.jc_D3;
    m_config.jc_D4 = sph.jc_D4;
    m_config.jc_D5 = sph.jc_D5;
    m_config.damageThreshold = sph.damageThreshold;
    m_config.referenceStrainRate = sph.referenceStrainRate;

    m_config.lodActiveRadius = sph.lodActiveRadius;
    m_config.lodNearRadius   = sph.lodNearRadius;
    // Bulk-viscosity checkerboard damper (see MPMSolver.cuh).
    m_config.linearViscosityCoef = 0.12;
    m_lodEnabled             = sph.lodEnabled;

    // ── Adiabatic shear model (plastic-work rate heating) ───────────────────
    // Reads an optional "adiabatic_shear" section. When absent it defaults to
    // ENABLED with a cutting-physics strain-rate threshold so the strong
    // Taylor-Quinney rate heating (σ_eq·ε̇·dt) contributes the dominant
    // plastic-work heat that friction-only contact heat omits.
    {
        AdiabaticShearConfig asCfg;
        if (config.getJson().contains("adiabatic_shear")) {
            const auto& as = config.getJson()["adiabatic_shear"];
            asCfg.enabled              = as.value("enabled", true);
            asCfg.criticalStrainRate   = as.value("critical_strain_rate", 5000.0);
            asCfg.softeningThreshold   = as.value("softening_threshold", 0.95);
            asCfg.taylorQuinneyCoeff   = as.value("taylor_quinney", 0.9);
            asCfg.maxTemperatureRatio  = as.value("max_temp_ratio", 0.9);
            asCfg.critStrain           = as.value("critical_strain", 0.8);
            asCfg.tempThreshold        = as.value("temperature_threshold_C", 600.0);
        } else {
            asCfg.enabled             = true;
            asCfg.criticalStrainRate  = 5000.0;
            asCfg.softeningThreshold  = 0.95;
            asCfg.taylorQuinneyCoeff  = 0.9;
            asCfg.maxTemperatureRatio = 0.9;
            asCfg.critStrain          = 0.8;
            asCfg.tempThreshold       = 600.0;
        }
        asCfg.thermalDiffusivity =
            mat.thermalConductivity / std::max(mat.density * mat.specificHeat, 1e-12);
        asCfg.typicalBandWidth   = 20e-6;
        m_adiabaticShearModel.initialize(asCfg, mat.meltingPoint,
                                         mat.specificHeat, mat.density);
    }

    // Fixture / viscoelastic damping boundary parameters
    m_fixtureThickness      = config.getMachineSetup().fixtureLayerThickness;
    m_fixtureDampingCoeff   = 1e7;  // 1/s — absorbs ~97% over 20 steps @ 20ns, tuned for steel c≈5000m/s

    // Allocate device memory for bounds and stats
    allocateMemory(m_maxParticles, m_numGridNodes);

    // Create per-solver CUDA stream
    CUDA_CHECK(cudaStreamCreate(&m_computeStream));

    // Initialize bounds device memory (for atomic min/max)
    cudaMalloc(&d_boundsMin, 3 * sizeof(double));
    cudaMalloc(&d_boundsMax, 3 * sizeof(double));

    // Device mirror of m_toolPosition, read by capture-looped kernels (LOD)
    cudaMalloc(&d_toolPos, sizeof(Vec3));

    // Initialize particle bounds to empty sentinels. Workpiece creation fills
    // these with the actual particle extents before the first simulation step.
    const double inf = std::numeric_limits<double>::infinity();
    m_boundsSeedMin[0] = inf; m_boundsSeedMin[1] = inf; m_boundsSeedMin[2] = inf;
    m_boundsSeedMax[0] = -inf; m_boundsSeedMax[1] = -inf; m_boundsSeedMax[2] = -inf;
    cudaMemcpy(d_boundsMin, m_boundsSeedMin, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundsMax, m_boundsSeedMax, 3 * sizeof(double), cudaMemcpyHostToDevice);

    m_particleMin = Vec3(m_boundsSeedMin[0], m_boundsSeedMin[1], m_boundsSeedMin[2]);
    m_particleMax = Vec3(m_boundsSeedMax[0], m_boundsSeedMax[1], m_boundsSeedMax[2]);

    m_isInitialized = true;
    return true;
}

void MPMSolver::allocateMemory(int particleCapacity, int gridCapacity) {
    cudaMalloc(&d_particles, particleCapacity * sizeof(MPMParticle));
    cudaMalloc(&d_grid, gridCapacity * sizeof(MPMGridNode));
    cudaMalloc(&d_couplingPoints, particleCapacity * sizeof(CouplingPoint));
    // Persistent reduction buffers (allocated once, zero runtime cudaMalloc)
    CUDA_CHECK(cudaMalloc(&d_redMaxStress, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_redMaxTemp, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_redKinEnergy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_redIntEnergy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_meltCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_activeCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chipCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sumTemp, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_surfaceCount, sizeof(int)));
    h_particles.resize(particleCapacity);
    // Pinned host memory for fast async transfers
    if (h_pinnedParticles == nullptr) {
        cudaMallocHost(&h_pinnedParticles, particleCapacity * sizeof(MPMParticle));
    }
}

void MPMSolver::freeMemory() {
    if (d_particles) cudaFree(d_particles);
    if (d_grid) cudaFree(d_grid);
    if (d_couplingPoints) cudaFree(d_couplingPoints);
    if (d_redMaxStress) cudaFree(d_redMaxStress);
    if (d_redMaxTemp) cudaFree(d_redMaxTemp);
    if (d_redKinEnergy) cudaFree(d_redKinEnergy);
    if (d_redIntEnergy) cudaFree(d_redIntEnergy);
    if (d_boundsMin) cudaFree(d_boundsMin);
    if (d_boundsMax) cudaFree(d_boundsMax);
    if (d_toolPos) cudaFree(d_toolPos);
    if (d_startToolNodes) cudaFree(d_startToolNodes);
    if (d_interpToolNodes) cudaFree(d_interpToolNodes);
    if (d_meltCount) cudaFree(d_meltCount);
    if (d_activeCount) cudaFree(d_activeCount);
    if (d_chipCount) cudaFree(d_chipCount);
    if (d_sumTemp) cudaFree(d_sumTemp);
    if (d_surfaceCount) cudaFree(d_surfaceCount);
    if (m_stepGraphExec) { cudaGraphExecDestroy(m_stepGraphExec); m_stepGraphExec = nullptr; }
    m_stepGraphRecorded = false;
    m_lastGraphDt = 0.0;
    if (h_pinnedParticles) cudaFreeHost(h_pinnedParticles);
    if (m_computeStream) cudaStreamDestroy(m_computeStream);
    h_pinnedParticles = nullptr;
    m_computeStream = nullptr;
    d_startToolNodes = nullptr;
    d_interpToolNodes = nullptr;
    d_meltCount = nullptr;
    d_activeCount = nullptr;
    d_chipCount = nullptr;
    d_sumTemp = nullptr;
    d_surfaceCount = nullptr;
}

void MPMSolver::copyToDevice() {
    cudaMemcpyAsync(d_particles, h_particles.data(), m_numParticles * sizeof(MPMParticle), cudaMemcpyHostToDevice, m_computeStream);
    cudaStreamSynchronize(m_computeStream);
}

void MPMSolver::copyFromDevice() {
    cudaMemcpyAsync(h_particles.data(), d_particles, m_numParticles * sizeof(MPMParticle), cudaMemcpyDeviceToHost, m_computeStream);
    cudaStreamSynchronize(m_computeStream);
}

void MPMSolver::beginInterpolatedToolMotion(FEMNodeGPU* femNodes, int numNodes) {
    if (!d_startToolNodes || m_numToolNodes != numNodes) {
        if (d_startToolNodes) cudaFree(d_startToolNodes);
        if (d_interpToolNodes) cudaFree(d_interpToolNodes);
        cudaMalloc(&d_startToolNodes, numNodes * sizeof(FEMNodeGPU));
        cudaMalloc(&d_interpToolNodes, numNodes * sizeof(FEMNodeGPU));
        m_numToolNodes = numNodes;
    }
    cudaMemcpyAsync(d_startToolNodes, femNodes, numNodes * sizeof(FEMNodeGPU), cudaMemcpyDeviceToDevice, m_computeStream);
    cudaStreamSynchronize(m_computeStream);
}

void MPMSolver::interpolateToolForSubStep(FEMNodeGPU* femNodes, double alpha) {
    if (!d_interpToolNodes || !d_startToolNodes || m_numToolNodes == 0) return;
    int threads = 256;
    int blocks = (m_numToolNodes + threads - 1) / threads;
    interpolateToolNodesKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_interpToolNodes, d_startToolNodes, femNodes, m_numToolNodes, alpha);
    CUDA_CHECK_KERNEL();
    d_toolNodes = d_interpToolNodes;
}

void MPMSolver::endInterpolatedToolMotion(FEMNodeGPU* femNodes) {
    d_toolNodes = femNodes;
}

__global__ void chipFractureAndSeparationKernel(
    MPMParticle* particles, int count,
    double toolTipZ, double toolCenterX, double toolCenterY, double toolRadiusMax,
    double cellSize, double damageThreshold, double minSeparationDamage)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MPMParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE) return;

    // ── Physical Chip Separation ──────────────────────────────────────────
    // Material is sheared at the cutting lips and evacuated along the flutes in +Z.
    // A particle separates into a chip when:
    // 1. It is in the active shear zone or flute evacuation chamber (Z >= toolTipZ - 0.5 * cellSize)
    // 2. It is within the radial footprint of the cutting lips (rTool <= toolRadiusMax + cellSize)
    // 3. It has accumulated sufficient ductile damage (D >= minSeparationDamage ~ 0.90 * threshold)
    if (p.status == ParticleStatus::ACTIVE) {
        double dtx = p.x - toolCenterX;
        double dty = p.y - toolCenterY;
        double rTool = sqrt(dtx * dtx + dty * dty);

        // Cutting zone: at or above the cutting lip along Z (flute evacuation),
        // within the tool cutting diameter, and having reached ductile failure.
        bool inShearOrFluteZone = (p.z >= toolTipZ - 0.5 * cellSize) && 
                                  (rTool <= toolRadiusMax + cellSize);

        if (inShearOrFluteZone && p.damage >= minSeparationDamage) {
            p.status = ParticleStatus::CHIP;
            p.residualStress = fmax(p.residualStress,
                sqrt(p.stress_xx * p.stress_xx + p.stress_yy * p.stress_yy +
                     p.stress_zz * p.stress_zz));
            // Cohesive softening: relieve stress gradually to prevent shock waves
            p.stress_xx *= 0.1;
            p.stress_yy *= 0.1;
            p.stress_zz *= 0.1;
            p.stress_xy *= 0.1;
            p.stress_xz *= 0.1;
            p.stress_yz *= 0.1;
            p.damage = fmax(p.damage, damageThreshold);
            return;
        }
        return;
    }

    // ── Chip fracture (CHIP particles only) ────────────────────────────────
    // CHIP particles with continued damage accumulation or high residual
    // stress get a velocity perturbation to simulate further fragmentation.
    // This prevents long continuous chips and produces realistic chip breakage.
    if (p.status == ParticleStatus::CHIP) {
        if (p.damage < damageThreshold * 1.5) return;

        // Fracture: impart a random-ish perturbation based on local state.
        // Use the particle ID to seed a deterministic pseudo-random direction.
        unsigned int seed = (unsigned int)(p.id * 2654435761U);
        double rx = (double)((seed >> 16) & 0xFF) / 255.0 - 0.5;
        double ry = (double)((seed >> 8) & 0xFF) / 255.0 - 0.5;
        double rz = (double)(seed & 0xFF) / 255.0 - 0.5;

        // Perturbation magnitude: fraction of the current velocity magnitude
        // (bounded — old formula `speed*0.3 + 0.5` could fling debris at
        // arbitrary speeds when the AMS had inflated the density-scaled field).
        double speed = sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
        double perturbMag = fmin(speed * edgepredict::physics::CHIP_PERTURB_SPEED_FACTOR 
                                 + edgepredict::physics::CHIP_PERTURB_BASE, 
                                 edgepredict::physics::CHIP_PERTURB_MAX);

        p.vx += rx * perturbMag;
        p.vy += ry * perturbMag;
        p.vz += rz * perturbMag;

        // [FIX Debris scatter] Cap chip speed so fractured chips cannot blow
        // across the whole domain (physical chip speeds here are ~1-2 m/s;
        // cap is generous at 10 m/s).
        double v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
        const double chipVCap = edgepredict::physics::CHIP_SPEED_CAP;
        if (v2 > chipVCap * chipVCap) {
            double s = chipVCap / sqrt(v2);
            p.vx *= s; p.vy *= s; p.vz *= s;
        }

        // Reset damage to prevent repeated fracturing of the same particle
        p.damage = damageThreshold * edgepredict::physics::DAMAGE_RESET_FRACTION;
    }
}

void MPMSolver::step(double dt) {
    if (!m_isInitialized || m_numParticles == 0) return;

    int3 gridDim = {m_gridDimX, m_gridDimY, m_gridDimZ};
    int gridThreads = 256;
    int gridBlocks = (m_numGridNodes + gridThreads - 1) / gridThreads;

    int partThreads = 256;
    int partBlocks = (m_numParticles + partThreads - 1) / partThreads;

    // Host seed for the captured H2D reset copies. These MUST be member
    // buffers (stable address for the graph's lifetime), not stack arrays
    // (the captured address was previously re-read on every graph replay
    // after the stack frame had been destroyed).
    const double inf = std::numeric_limits<double>::infinity();
    m_boundsSeedMin[0] = inf; m_boundsSeedMin[1] = inf; m_boundsSeedMin[2] = inf;
    m_boundsSeedMax[0] = -inf; m_boundsSeedMax[1] = -inf; m_boundsSeedMax[2] = -inf;

    // ── Refresh device tool position for LOD classification ────────────────
    // Must run before EVERY graph execute (record AND replay): the graph was
    // recorded once with d_toolPos as the kernel argument address, so the
    // LOD window tracks the tool as it moves even though the graph is not
    // re-recorded. Without this the window anchor freezes at record-time.
    CUDA_CHECK(cudaMemcpyAsync(d_toolPos, &m_toolPosition, sizeof(Vec3),
                               cudaMemcpyHostToDevice, m_computeStream));

    // ── CUDA Graph: record once per distinct dt, replay for remaining sub-steps ──
    // The graph captures ALL kernel launches + cudaMemcpyAsync on m_computeStream.
    // CUDA_CHECK_KERNEL() is harmless inside capture (kernels are recorded, not run).

    if (!m_stepGraphRecorded || std::fabs(dt - m_lastGraphDt) > 1e-20) {
        // Teardown old graph if dt changed
        if (m_stepGraphExec) {
            cudaGraphExecDestroy(m_stepGraphExec);
            m_stepGraphExec = nullptr;
        }
        m_stepGraphRecorded = false;

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(m_computeStream, cudaStreamCaptureModeGlobal));

        // ── Captured kernel launch sequence ──────────────────────────────────
        if (m_lodEnabled) {
            lodClassifyKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
                d_particles, m_numParticles, d_toolPos,
                m_config.lodActiveRadius, m_config.lodNearRadius);
            CUDA_CHECK_KERNEL();
        }

        resetGridKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(d_grid, m_numGridNodes);
        CUDA_CHECK_KERNEL();

        p2gKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles, d_grid, gridDim, m_config, m_domainMin, dt);
        CUDA_CHECK_KERNEL();

        updateGridKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(d_grid, m_numGridNodes, dt);
        CUDA_CHECK_KERNEL();

        if (m_fixtureThickness > 0.0) {
            applyFixtureBCKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(
                d_grid, gridDim, m_domainMin, m_config.dx,
                m_workpieceMin, m_workpieceMax,
                m_fixtureThickness, m_fixtureDampingCoeff, dt);
            CUDA_CHECK_KERNEL();
        }

        // Kinematic boundary condition note:
        // Tool-workpiece non-penetration, Coulomb friction, and contact force
        // are resolved via the high-order Signed Distance Field (ToolSDF) and
        // Lagrangian XPBD solver in ContactSolver (every mechanical step dt).
        // Overwriting Eulerian background grid nodes with tool rotational velocity
        // (g.vx = g.tvx) imposed an unphysical sticky boundary condition that
        // violently tore the workpiece and fought against XPBD repulsion.
        // The background grid now integrates continuum momentum freely.

        g2pKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles, d_grid, gridDim, m_config, dt, m_domainMin);
        CUDA_CHECK_KERNEL();

        if (m_adiabaticShearModel.isEnabled()) {
            m_adiabaticShearModel.update(d_particles, m_numParticles, dt, m_computeStream);
        }

        cudaMemcpyAsync(d_boundsMin, m_boundsSeedMin, 3 * sizeof(double), cudaMemcpyHostToDevice, m_computeStream);
        cudaMemcpyAsync(d_boundsMax, m_boundsSeedMax, 3 * sizeof(double), cudaMemcpyHostToDevice, m_computeStream);
        updateParticleBoundsKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles,
            d_boundsMin, d_boundsMin + 1, d_boundsMin + 2,
            d_boundsMax, d_boundsMax + 1, d_boundsMax + 2);
        CUDA_CHECK_KERNEL();

        resetExternalForceKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles);
        CUDA_CHECK_KERNEL();
        // ── End of captured sequence ─────────────────────────────────────────

        CUDA_CHECK(cudaStreamEndCapture(m_computeStream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&m_stepGraphExec, graph, NULL, NULL, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        m_stepGraphRecorded = true;
        m_lastGraphDt = dt;
    }

    // Execute the graph (either freshly recorded or replayed)
    // No cudaStreamSynchronize — the graph runs fully async. Bounding-box
    // telemetry is read by updateResults() after the macro-step completes.
    CUDA_CHECK(cudaGraphLaunch(m_stepGraphExec, m_computeStream));

    // Launch chip fracture & separation outside the graph every 2 steps
    if (m_currentStep % 2 == 0) {
        chipFractureAndSeparationKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles,
            m_toolPosition.z, m_toolPosition.x, m_toolPosition.y, m_toolRadiusMax,
            m_config.dx, m_config.damageThreshold,
            fmax(1e-3, 0.90 * m_config.damageThreshold));
        CUDA_CHECK_KERNEL();
    }

    m_currentTime += dt;
    m_currentStep++;
}

double MPMSolver::getStableTimeStep() const {
    // CFL condition for MPM: dt ≤ CFL * dx / c
    // where c = sqrt(E/ρ) is the dilatational wave speed
    // Uses m_adaptiveCFL (set by frequency-domain AMS) instead of fixed 0.2
    // MASS SCALING: Use scaled density for CFL (dt ∝ sqrt(rho_scaled)),
    // but physical density for all physics (handled by keeping m_config.density = physicalDensity)
    double rho_cfl = m_config.physicalDensity * m_amsMassScalingFactor;
    double waveSpeed = sqrt(m_config.youngsModulus / fmax(rho_cfl, 1.0));
    if (!(waveSpeed == waveSpeed) || waveSpeed <= 0.0)
        waveSpeed = sqrt(m_config.youngsModulus / fmax(m_config.physicalDensity, 1.0));
    double cfl = m_adaptiveCFL;
    double dt_cfl = cfl * m_config.dx / fmax(waveSpeed, 1.0);
    double dt = fmax(dt_cfl, 1e-10);
    if (!(dt == dt)) dt = 1e-10;
    return dt;
}

double MPMSolver::getWaveSpeed() const {
    return sqrt(m_config.youngsModulus / m_config.density);
}

double MPMSolver::getPhysicalWaveSpeed() const {
    return sqrt(m_config.youngsModulus / m_config.physicalDensity);
}

double MPMSolver::getHighestFrequency() const {
    // Estimate highest eigenfrequency from CFL condition:
    //   dt_cfl = CFL * dx / c_wave
    //   The highest frequency resolved by the grid is f_max ≈ c_wave / (2 * dx)
    //   (Nyquist limit for a wave on the grid with spacing dx)
    // Using unscaled physical density so the frequency estimate is
    // independent of AMS — this represents the TRUE material frequency.
    double waveSpeed = sqrt(m_config.youngsModulus / m_config.physicalDensity);
    double dx = fmax(m_config.dx, 1e-10);
    return waveSpeed / (2.0 * dx);
}

__global__ void updateParticleMassKernel(MPMParticle* particles, int numParticles, double newDensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    MPMParticle& p = particles[idx];
    if (p.status == ParticleStatus::ACTIVE || p.status == ParticleStatus::CHIP || p.status == ParticleStatus::FIXED_BOUNDARY) {
        p.density = newDensity;
        p.mass = p.volume * newDensity;
    }
}

void MPMSolver::setDynamicMassScaling(double factor) {
    // Mass scaling for explicit dynamics: ONLY affects CFL time step, NOT physics.
    // Physical density (m_config.physicalDensity) is used for ALL physics calculations:
    //   - Stress/strain, momentum, kinetic energy, contact forces, heat capacity
    // Scaled density (m_config.density) is used ONLY for CFL: dt ∝ sqrt(density)
    // This is the correct explicit mass scaling approach — it accelerates time
    // integration without corrupting the physics.
    // [FIX] A NaN/Inf/negative factor from the AMS PID would silently corrupt
    // the density field. Clamp to a sane range instead.
    if (!(factor == factor) || factor < 1.0) factor = 1.0;
    if (factor > 1.0e6) factor = 1.0e6;
    m_amsMassScalingFactor = factor;  // Store for CFL calculation only
    // m_config.density remains = physicalDensity (no change to particle masses)
    // m_config.physicalDensity is the ONLY density used in physics kernels
}

void MPMSolver::reset() {
    m_currentTime = 0;
    m_currentStep = 0;
}

void MPMSolver::getBounds(double& minX, double& minY, double& minZ, double& maxX, double& maxY, double& maxZ) const {
    // Bounds are updated every step via updateParticleBoundsKernel and copied
    // to m_particleMin/m_particleMax in step(). This getter returns cached values.
    minX = m_particleMin.x; minY = m_particleMin.y; minZ = m_particleMin.z;
    maxX = m_particleMax.x; maxY = m_particleMax.y; maxZ = m_particleMax.z;
}

void MPMSolver::syncMetrics() {
    if (m_numParticles == 0) return;

    double initVal = 0.0;
    cudaMemcpy(d_redMaxStress, &initVal, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_redMaxTemp, &initVal, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_redKinEnergy, &initVal, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_redIntEnergy, &initVal, sizeof(double), cudaMemcpyHostToDevice);

    int partThreads = 256;
    int partBlocks = (m_numParticles + partThreads - 1) / partThreads;
    computeParticleStatsKernel<<<partBlocks, partThreads>>>(
        d_particles, m_numParticles,
        d_redMaxStress, d_redMaxTemp, d_redKinEnergy, d_redIntEnergy);
    CUDA_CHECK_KERNEL();

    cudaMemcpy(&m_maxStress, d_redMaxStress, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&m_maxTemperature, d_redMaxTemp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&m_kineticEnergy, d_redKinEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&m_internalEnergy, d_redIntEnergy, sizeof(double), cudaMemcpyDeviceToHost);
}

void MPMSolver::applyHeatFlux(double x, double y, double z, double heatFlux) {
    // Not yet implemented — requires a kernel that distributes heat flux
    // to nearby particles. For now, thermal coupling from CFD/external sources
    // is handled through the grid or direct temperature boundary conditions.
}

void MPMSolver::setParticleTemperatures(const double* temperatures, int count) {
    if (!d_particles || count <= 0 || count > m_numParticles) return;
    // Ensure host particles are up-to-date
    copyFromDevice();
    // Update temperatures on host
    for (int i = 0; i < count; ++i) {
        h_particles[i].temperature = temperatures[i];
    }
    // Upload back to device
    copyToDevice();
}

double MPMSolver::getTemperatureAt(double x, double y, double z) const {
    // Nearest-neighbor lookup on host particles.
    // This is called infrequently (CFD coupling), so O(N) scan is acceptable.
    double closestDist = 1e30;
    double temp = m_config.ambientTemp;
    for (int i = 0; i < m_numParticles; ++i) {
        const MPMParticle& p = h_particles[i];
        if (p.status != ParticleStatus::ACTIVE) continue;
        double dx = p.x - x, dy = p.y - y, dz = p.z - z;
        double d = dx*dx + dy*dy + dz*dz;
        if (d < closestDist) {
            closestDist = d;
            temp = p.temperature;
        }
    }
    return temp;
}

void MPMSolver::initializeParticleBox(const Vec3& minBounds, const Vec3& maxBounds, double spacing) {
    m_particleMin = minBounds;
    m_particleMax = maxBounds;
    m_workpieceMin = minBounds;
    m_workpieceMax = maxBounds;
    m_numParticles = 0;

    double density = m_mainConfig
        ? m_mainConfig->getMaterial().density *
              m_mainConfig->getSPH().numericalMassScalingFactor
        : 7850.0;
    double pMass = density * spacing * spacing * spacing;

    // Mark bottom 10% as fixed boundary
    double fixedThreshold = minBounds.z + (maxBounds.z - minBounds.z) * 0.1;

    for (double z = minBounds.z; z <= maxBounds.z; z += spacing) {
        for (double y = minBounds.y; y <= maxBounds.y; y += spacing) {
            for (double x = minBounds.x; x <= maxBounds.x; x += spacing) {
                if (m_numParticles >= m_maxParticles) break;
                MPMParticle p;
                p.x = x; p.y = y; p.z = z;
                p.mass = pMass;
                p.density = density;
                p.volume = spacing * spacing * spacing;
                p.temperature = m_config.ambientTemp;
                p.strainRate = 0.0;  // Initialize to reference for
                                                               // first-step JC rate term
                // Unique ID — seeds deterministic per-particle chip fracture
                // perturbations (previously unset → identical for all particles)
                p.id = m_numParticles;
                // Bottom layer = fixed boundary (clamped by vice/fixture)
                if (z <= fixedThreshold) {
                    p.status = ParticleStatus::FIXED_BOUNDARY;
                }
                h_particles[m_numParticles++] = p;
            }
        }
    }
    copyToDevice();
    double hostMin[3] = {m_particleMin.x, m_particleMin.y, m_particleMin.z};
    double hostMax[3] = {m_particleMax.x, m_particleMax.y, m_particleMax.z};
    cudaMemcpy(d_boundsMin, hostMin, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundsMax, hostMax, 3 * sizeof(double), cudaMemcpyHostToDevice);
}

void MPMSolver::initializeCylindricalWorkpiece(const Vec3& center, double radius, double length, double spacing, int axis) {
    m_particleMin = Vec3(center.x - radius, center.y - radius, center.z - length);
    m_particleMax = Vec3(center.x + radius, center.y + radius, center.z);
    m_workpieceMin = m_particleMin;
    m_workpieceMax = m_particleMax;
    m_numParticles = 0;

    double density = m_mainConfig
        ? m_mainConfig->getMaterial().density *
              m_mainConfig->getSPH().numericalMassScalingFactor
        : 7850.0;
    double pMass = density * spacing * spacing * spacing;

    // Mark bottom 10% of cylinder as fixed boundary
    double fixedThreshold = center.z - length * 0.1;

    double r2 = radius * radius;
    for (double dz = 0; dz <= length; dz += spacing) {
        for (double dy = -radius; dy <= radius; dy += spacing) {
            for (double dx = -radius; dx <= radius; dx += spacing) {
                if (dx*dx + dy*dy <= r2) {
                    if (m_numParticles >= m_maxParticles) break;
                    MPMParticle p;
                    p.x = center.x + dx;
                    p.y = center.y + dy;
                    p.z = center.z - dz;
                    p.mass = pMass;
                    p.density = density;
                    p.volume = spacing * spacing * spacing;
                    p.temperature = m_config.ambientTemp;
                    p.strainRate = 0.0;
                    // Unique ID — seeds deterministic per-particle chip fracture
                    p.id = m_numParticles;
                    // Bottom fixture layer = fixed boundary
                    if (p.z <= fixedThreshold) {
                        p.status = ParticleStatus::FIXED_BOUNDARY;
                    }
                    h_particles[m_numParticles++] = p;
                }
            }
        }
    }
    copyToDevice();
    double hostMin[3] = {m_particleMin.x, m_particleMin.y, m_particleMin.z};
    double hostMax[3] = {m_particleMax.x, m_particleMax.y, m_particleMax.z};
    cudaMemcpy(d_boundsMin, hostMin, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundsMax, hostMax, 3 * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<MPMParticle> MPMSolver::getParticles() {
    copyFromDevice();
    return std::vector<MPMParticle>(h_particles.begin(), h_particles.begin() + m_numParticles);
}

int MPMSolver::getMeltingParticleCount(double meltingPoint) {
    if (m_numParticles == 0 || !d_particles || !d_meltCount) return 0;
    cudaMemsetAsync(d_meltCount, 0, sizeof(int), m_computeStream);
    int threads = 256;
    int blocks = (m_numParticles + threads - 1) / threads;
    countMeltingParticlesKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_particles, m_numParticles, meltingPoint, d_meltCount);
    CUDA_CHECK_KERNEL();
    int result = 0;
    cudaMemcpyAsync(&result, d_meltCount, sizeof(int), cudaMemcpyDeviceToHost, m_computeStream);
    cudaStreamSynchronize(m_computeStream);
    return result;
}

double MPMSolver::getAvgSurfaceTemp() {
    if (m_numParticles == 0 || !d_particles || !d_sumTemp || !d_surfaceCount) return 25.0;
    cudaMemsetAsync(d_sumTemp, 0, sizeof(double), m_computeStream);
    cudaMemsetAsync(d_surfaceCount, 0, sizeof(int), m_computeStream);
    int threads = 256;
    int blocks = (m_numParticles + threads - 1) / threads;
    avgSurfaceTempKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_particles, m_numParticles, d_sumTemp, d_surfaceCount);
    CUDA_CHECK_KERNEL();
    double sum = 0.0;
    int count = 0;
    cudaMemcpyAsync(&sum, d_sumTemp, sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);
    cudaMemcpyAsync(&count, d_surfaceCount, sizeof(int), cudaMemcpyDeviceToHost, m_computeStream);
    cudaStreamSynchronize(m_computeStream);
    return count > 0 ? sum / count : 25.0;
}

void MPMSolver::applyExternalForce(const Vec3& center, double radius, const Vec3& force) {
    if (m_numParticles == 0 || force.length() < 1e-12) return;

    int partThreads = 256;
    int partBlocks = (m_numParticles + partThreads - 1) / partThreads;

    applyExternalForceKernel<<<partBlocks, partThreads>>>(
        d_particles, m_numParticles, center, radius, force);
    CUDA_CHECK_KERNEL();
}

void MPMSolver::resetExternalForces() {
    if (m_numParticles == 0) return;
    int partThreads = 256;
    int partBlocks = (m_numParticles + partThreads - 1) / partThreads;
    resetExternalForceKernel<<<partBlocks, partThreads>>>(d_particles, m_numParticles);
    CUDA_CHECK_KERNEL();
}

void MPMSolver::cutParticles(const Vec3& planeNormal, double planeDist) {
    if (m_numParticles == 0) return;

    int partThreads = 256;
    int partBlocks = (m_numParticles + partThreads - 1) / partThreads;

    cutParticlesKernel<<<partBlocks, partThreads>>>(
        d_particles, m_numParticles, planeNormal, planeDist);
    CUDA_CHECK_KERNEL();
}

void MPMSolver::syncCouplingPoints() {
    if (!m_isInitialized || m_numParticles == 0 || !d_couplingPoints) return;
    int threads = 256;
    int blocks = (m_numParticles + threads - 1) / threads;
    syncMPMCouplingPointsKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_particles, d_couplingPoints, m_numParticles);
    CUDA_CHECK_KERNEL();
}

void MPMSolver::updateResults() {
    if (!m_isInitialized || m_numParticles == 0) return;

    // Use GPU reduction instead of full D2H copy for particle counts (pre-allocated)
    if (!d_activeCount || !d_chipCount) return;
    cudaMemsetAsync(d_activeCount, 0, sizeof(int), m_computeStream);
    cudaMemsetAsync(d_chipCount, 0, sizeof(int), m_computeStream);

    int threads = 256;
    int blocks = (m_numParticles + threads - 1) / threads;
    countParticleStatusKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_particles, m_numParticles, d_activeCount, d_chipCount);
    CUDA_CHECK_KERNEL();

    cudaMemcpyAsync(&m_results.activeParticleCount, d_activeCount, sizeof(int),
                    cudaMemcpyDeviceToHost, m_computeStream);
    cudaMemcpyAsync(&m_results.chipParticleCount, d_chipCount, sizeof(int),
                    cudaMemcpyDeviceToHost, m_computeStream);

    // Update particle bounds from device-side atomic min/max
    double hMin[3], hMax[3];
    cudaMemcpyAsync(hMin, d_boundsMin, 3 * sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);
    cudaMemcpyAsync(hMax, d_boundsMax, 3 * sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);

    // Sync metrics (stress, temp, kinetic energy) — also async D2H of 4 scalars
    cudaMemcpyAsync(&m_maxStress, d_redMaxStress, sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);
    cudaMemcpyAsync(&m_maxTemperature, d_redMaxTemp, sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);
    cudaMemcpyAsync(&m_kineticEnergy, d_redKinEnergy, sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);
    cudaMemcpyAsync(&m_internalEnergy, d_redIntEnergy, sizeof(double), cudaMemcpyDeviceToHost, m_computeStream);

    cudaStreamSynchronize(m_computeStream);

    m_particleMin = Vec3(hMin[0], hMin[1], hMin[2]);
    m_particleMax = Vec3(hMax[0], hMax[1], hMax[2]);
    m_results.maxStress = m_maxStress;
    m_results.maxTemperature = m_maxTemperature;
    m_results.totalKineticEnergy = m_kineticEnergy;
}

} // namespace edgepredict
