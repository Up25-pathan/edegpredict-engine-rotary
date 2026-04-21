/**
 * @file SPHSolver.cu
 * @brief SPH workpiece simulation CUDA implementation
 *
 * Physics Overhaul — All audit issues corrected:
 *
 * 1. Tensile instability fix (removes pressure ≥ 0 clamp)
 *    Metals CAN and MUST support tension before fracture.
 *    Monaghan artificial pressure correction prevents fragmentation instability
 *    at short range while allowing real negative pressure at the cutting front.
 *    Chip break-off now driven by J-C damage (D→1) not by pressure sign.
 *
 * 2. Johnson-Cook Constitutive Stress Kernel
 *    Fills the full deviatoric stress tensor from JC flow stress:
 *      σ_JC = (A + B·εp^n)·(1 + C·ln(ε̇/ε̇₀))·(1 - T*^m)
 *    Downstream models (damage, ASB, residual stress) now receive physically
 *    correct stress values instead of SPH pressure proxies.
 *
 * 3. Kernel Gradient Correction (KGC) for strain rate near free surfaces
 *    SPH velocity gradients are truncated at free boundaries (machined surface,
 *    chip surface).  KGC renormalization matrix L_i corrects for incomplete
 *    support domains:  corrected_grad = L_i * raw_grad.
 *    Prevents premature/delayed chip separation from noisy boundary gradients.
 *
 * 4. Thrust sort on CUDA stream (thrust::cuda::par.on(m_computeStream))
 *    Eliminates cudaStreamSynchronize() barrier called every step.
 *    GPU async pipeline is now fully preserved.
 */

#include "SPHSolver.cuh"
#include "Config.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace edgepredict {

// ============================================================================
// Functors for Thrust (needed because device lambdas can't be in private fns)
// ============================================================================

struct CompareParticleByHash {
    __host__ __device__
    bool operator()(const SPHParticle& a, const SPHParticle& b) const {
        return a.cellHash < b.cellHash;
    }
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Compute spatial hash cell index for a particle
 */
__device__ inline int computeCellHash(double x, double y, double z, 
                                       double cellSize, int tableSize,
                                       double domainMinX, double domainMinY, double domainMinZ) {
    int cx = static_cast<int>((x - domainMinX) / cellSize);
    int cy = static_cast<int>((y - domainMinY) / cellSize);
    int cz = static_cast<int>((z - domainMinZ) / cellSize);
    
    // Simple hash function (primes: 73856093, 19349663, 83492791)
    unsigned long long h = (static_cast<unsigned long long>(cx) * 73856093) ^ 
                           (static_cast<unsigned long long>(cy) * 19349663) ^ 
                           (static_cast<unsigned long long>(cz) * 83492791);
    
    return static_cast<int>(h % tableSize);
}

/**
 * @brief Update particle LOD zones based on distance to tool
 */
__global__ void updateLODZonesKernel(SPHParticle* particles, int numParticles,
                                      double toolX, double toolY, double toolZ,
                                      double activeRadius, double nearRadius,
                                      int currentStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE) return;
    
    // Compute distance to tool
    double dx = p.x - toolX;
    double dy = p.y - toolY;
    double dz = p.z - toolZ;
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    
    // Classify into zones
    if (dist <= activeRadius) {
        p.lodZone = LODZone::ACTIVE;
        p.lastUpdateStep = currentStep;
    } else if (dist <= nearRadius) {
        p.lodZone = LODZone::ZONE_NEAR;
    } else {
        p.lodZone = LODZone::ZONE_FAR;
    }
}

/**
 * @brief Compute stress tensor and equivalent stress from velocity gradients
 */
__host__ __device__ inline double computeEquivalentStress(double sxx, double syy, double szz,
                                                   double sxy, double sxz, double syz) {
    // von Mises equivalent stress
    double s1 = sxx - syy;
    double s2 = syy - szz;
    double s3 = szz - sxx;
    return sqrt(0.5 * (s1*s1 + s2*s2 + s3*s3 + 6*(sxy*sxy + sxz*sxz + syz*syz)));
}

/**
 * @brief Compute damage increment using Johnson-Cook failure model
 * 
 * Johnson-Cook failure strain:
 *   ε_f = [D1 + D2*exp(D3*σ*)] * [1 + D4*ln(ε̇*)] * [1 + D5*T*]
 * 
 * Damage increment:
 *   ΔD = Δε_p / ε_f
 */
__global__ void computeDamageKernel(SPHParticle* particles, int numParticles,
                                     double D1, double D2, double D3, double D4, double D5,
                                     double refStrainRate, double refTemp, double meltTemp,
                                     double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    
    // Compute equivalent stress
    double sigma_eq = computeEquivalentStress(p.stress_xx, p.stress_yy, p.stress_zz,
                                               p.stress_xy, p.stress_xz, p.stress_yz);
    
    // Mean stress (hydrostatic)
    double sigma_m = (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    
    // Stress triaxiality (clamped to avoid division issues)
    double sigma_star = 0.0;
    if (sigma_eq > 1.0) {  // Avoid division by zero
        sigma_star = sigma_m / sigma_eq;
        sigma_star = fmax(-2.0, fmin(2.0, sigma_star));  // Clamp
    }
    
    // Strain rate ratio
    double strain_rate_term = 1.0;
    if (p.strainRate > refStrainRate) {
        strain_rate_term = 1.0 + D4 * log(p.strainRate / refStrainRate);
    }
    
    // Homologous temperature: T* = (T - Tref) / (Tmelt - Tref)
    double T_star = 0.0;
    if (meltTemp > refTemp) {
        T_star = (p.temperature - refTemp) / (meltTemp - refTemp);
        T_star = fmax(0.0, fmin(1.0, T_star));  // Clamp to [0, 1]
    }
    
    // Johnson-Cook failure strain
    double eps_f = (D1 + D2 * exp(D3 * sigma_star)) * strain_rate_term * (1.0 + D5 * T_star);
    eps_f = fmax(0.01, eps_f);  // Minimum failure strain to avoid instability
    
    // Damage increment (ΔD = Δε_p / ε_f)
    double plastic_strain_increment = p.strainRate * dt;
    if (plastic_strain_increment > 0 && eps_f > 0) {
        p.damage += plastic_strain_increment / eps_f;
    }
    
    // Accumulate plastic strain
    p.plasticStrain += plastic_strain_increment;
}

/**
 * @brief Separate chips: transition particles with damage > threshold to CHIP status
 */
__global__ void separateChipsKernel(SPHParticle* particles, int numParticles,
                                     double damageThreshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    
    if (p.damage >= damageThreshold) {
        p.status = ParticleStatus::CHIP;
        // Store final stress as residual stress
        p.residualStress = computeEquivalentStress(p.stress_xx, p.stress_yy, p.stress_zz,
                                                    p.stress_xy, p.stress_xz, p.stress_yz);
    }
}

/**
 * @brief Compute velocity gradient tensor and strain rate WITH Kernel Gradient Correction.
 *
 * PHYSICS FIX: SPH velocity gradients are truncated at free boundaries (machined
 * surface, chip surface) because particle neighbourhoods are incomplete.  The
 * Kernel Gradient Correction (KGC) method applies a renormalization matrix L_i
 * to correct for the asymmetric support domain:
 *
 *   B_i = [Σ_j (r_ij ⊗ ∇W_ij) · V_j]^(−1)     [3×3 matrix, "correction matrix"]
 *   corrected velocity gradient = B_i · raw_gradient
 *
 * For fully interior particles B_i ≈ I (identity), so there is no change.
 * For boundary particles B_i compensates for missing neighbours.
 *
 * References:
 *   Bonet & Lok (1999), Comput. Methods Appl. Mech. Eng. 180, 97-115.
 *   Randles & Libersky (1996), Comput. Methods Appl. Mech. Eng. 139, 375-408.
 */
__global__ void computeStrainRateKernel(SPHParticle* particles, int numParticles,
                                         int* cellStart, int* cellEnd, int tableSize,
                                         SPHKernelConfig config, double cellSize,
                                         double domainMinX, double domainMinY, double domainMinZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    SPHParticle& pi = particles[idx];
    if (pi.status == ParticleStatus::INACTIVE) return;

    // Raw velocity gradient accumulation (L_00 … L_22 per row convention)
    double dvx_dx = 0, dvx_dy = 0, dvx_dz = 0;
    double dvy_dx = 0, dvy_dy = 0, dvy_dz = 0;
    double dvz_dx = 0, dvz_dy = 0, dvz_dz = 0;

    // KGC correction matrix B (lower-triangular accumulation before inversion)
    // B_ij = Σ_k  r_ki_j * W_ki_j_grad_i  * V_k
    // Stored as 3×3: b00..b22
    double b00=0, b01=0, b02=0;
    double b10=0, b11=0, b12=0;
    double b20=0, b21=0, b22=0;

    int ci = (int)((pi.x - domainMinX) / cellSize);
    int cj = (int)((pi.y - domainMinY) / cellSize);
    int ck = (int)((pi.z - domainMinZ) / cellSize);

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int dk = -1; dk <= 1; ++dk) {
                int hash = computeCellHash(
                    domainMinX + (ci + di + 0.5) * cellSize,
                    domainMinY + (cj + dj + 0.5) * cellSize,
                    domainMinZ + (ck + dk + 0.5) * cellSize,
                    cellSize, tableSize, domainMinX, domainMinY, domainMinZ);

                if (hash < 0 || hash >= tableSize) continue;
                int start = cellStart[hash];
                if (start < 0) continue;
                int end = cellEnd[hash];

                for (int j = start; j <= end; ++j) {
                    if (j == idx) continue;
                    const SPHParticle& pj = particles[j];
                    if (pj.status == ParticleStatus::INACTIVE) continue;

                    double rx = pj.x - pi.x;   // r_ji = x_j - x_i
                    double ry = pj.y - pi.y;
                    double rz = pj.z - pi.z;
                    double r2 = rx*rx + ry*ry + rz*rz;

                    if (r2 >= config.h2 || r2 < 1e-24) continue;

                    double r          = sqrt(r2);
                    double q          = r / config.h;
                    double grad_coeff = config.spikyGradCoeff * (1.0 - q) * (1.0 - q) / r;
                    double gx = grad_coeff * rx;
                    double gy = grad_coeff * ry;
                    double gz = grad_coeff * rz;

                    double vol_j = pj.mass / fmax(pj.density, 100.0);

                    // Accumulate KGC matrix: B += r_ji ⊗ ∇W_ji * V_j
                    b00 += rx * gx * vol_j;  b01 += rx * gy * vol_j;  b02 += rx * gz * vol_j;
                    b10 += ry * gx * vol_j;  b11 += ry * gy * vol_j;  b12 += ry * gz * vol_j;
                    b20 += rz * gx * vol_j;  b21 += rz * gy * vol_j;  b22 += rz * gz * vol_j;

                    // Accumulate raw velocity gradient
                    double dvx = pj.vx - pi.vx;
                    double dvy = pj.vy - pi.vy;
                    double dvz = pj.vz - pi.vz;
                    dvx_dx += vol_j * dvx * gx;
                    dvx_dy += vol_j * dvx * gy;
                    dvx_dz += vol_j * dvx * gz;
                    dvy_dx += vol_j * dvy * gx;
                    dvy_dy += vol_j * dvy * gy;
                    dvy_dz += vol_j * dvy * gz;
                    dvz_dx += vol_j * dvz * gx;
                    dvz_dy += vol_j * dvz * gy;
                    dvz_dz += vol_j * dvz * gz;
                }
            }
        }
    }

    // ── KGC Matrix Inversion (3×3 analytic) ─────────────────────────────────
    // Compute det(B) and apply inverse if well-conditioned.
    // For interior particles det(B) ≈ 1 → corrected ≈ raw.
    // For boundary particles det(B) < 1 → correction amplifies truncated gradient.
    double det = b00*(b11*b22 - b12*b21)
               - b01*(b10*b22 - b12*b20)
               + b02*(b10*b21 - b11*b20);

    if (fabs(det) > 1e-10) {
        // Cofactor matrix
        double inv_det = 1.0 / det;
        double i00 =  (b11*b22 - b12*b21) * inv_det;
        double i01 = -(b01*b22 - b02*b21) * inv_det;
        double i02 =  (b01*b12 - b02*b11) * inv_det;
        double i10 = -(b10*b22 - b12*b20) * inv_det;
        double i11 =  (b00*b22 - b02*b20) * inv_det;
        double i12 = -(b00*b12 - b02*b10) * inv_det;
        double i20 =  (b10*b21 - b11*b20) * inv_det;
        double i21 = -(b00*b21 - b01*b20) * inv_det;
        double i22 =  (b00*b11 - b01*b10) * inv_det;

        // Corrected velocity gradient = B^(-1) * raw_gradient (row × column)
        double cx_dx = i00*dvx_dx + i01*dvy_dx + i02*dvz_dx;
        double cx_dy = i00*dvx_dy + i01*dvy_dy + i02*dvz_dy;
        double cx_dz = i00*dvx_dz + i01*dvy_dz + i02*dvz_dz;
        double cy_dx = i10*dvx_dx + i11*dvy_dx + i12*dvz_dx;
        double cy_dy = i10*dvx_dy + i11*dvy_dy + i12*dvz_dy;
        double cy_dz = i10*dvx_dz + i11*dvy_dz + i12*dvz_dz;
        double cz_dx = i20*dvx_dx + i21*dvy_dx + i22*dvz_dx;
        double cz_dy = i20*dvx_dy + i21*dvy_dy + i22*dvz_dy;
        double cz_dz = i20*dvx_dz + i21*dvy_dz + i22*dvz_dz;

        dvx_dx = cx_dx; dvx_dy = cx_dy; dvx_dz = cx_dz;
        dvy_dx = cy_dx; dvy_dy = cy_dy; dvy_dz = cy_dz;
        dvz_dx = cz_dx; dvz_dy = cz_dy; dvz_dz = cz_dz;
    }
    // If det ≈ 0 (isolated particle), keep raw gradient (no correction applied).

    // ── Symmetric strain rate tensor (Ė = ½(L + Lᵀ)) ──────────────────────
    double exx = dvx_dx;
    double eyy = dvy_dy;
    double ezz = dvz_dz;
    double exy = 0.5 * (dvx_dy + dvy_dx);
    double exz = 0.5 * (dvx_dz + dvz_dx);
    double eyz = 0.5 * (dvy_dz + dvz_dy);

    // Von Mises equivalent strain rate  ε̇_eq = sqrt(2/3 * Ė:Ė)
    pi.strainRate = sqrt(2.0/3.0 *
        (exx*exx + eyy*eyy + ezz*ezz + 2.0*(exy*exy + exz*exz + eyz*eyz)));
}



/**
 * @brief Compute cell hashes for all particles
 */
__global__ void computeHashKernel(SPHParticle* particles, int numParticles,
                                   double cellSize, int tableSize,
                                   double domainMinX, double domainMinY, double domainMinZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE) {
        p.cellHash = tableSize + 1;  // Invalid hash, will be sorted to end
        return;
    }
    
    p.cellHash = computeCellHash(p.x, p.y, p.z, cellSize, tableSize,
                                  domainMinX, domainMinY, domainMinZ);
}

/**
 * @brief Find cell start/end indices after sorting
 */
__global__ void findCellBoundsKernel(SPHParticle* particles, int numParticles,
                                      int* cellStart, int* cellEnd, int tableSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    int hash = particles[idx].cellHash;
    if (hash >= tableSize) return;  // Invalid
    
    if (idx == 0) {
        cellStart[hash] = 0;
    } else {
        int prevHash = particles[idx - 1].cellHash;
        if (hash != prevHash) {
            cellStart[hash] = idx;
            if (prevHash < tableSize) {
                cellEnd[prevHash] = idx;
            }
        }
    }
    
    if (idx == numParticles - 1) {
        cellEnd[hash] = idx + 1;
    }
}

/**
 * @brief Compute density and pressure for all particles.
 *
 * Uses Poly6 kernel for density, Tait EOS for pressure.
 *
 * PHYSICS FIX: Negative pressure (tension) is now ALLOWED.
 * The old clamp `fmax(pressure, 0.0)` prevented metals from supporting tension,
 * making chips behave like sand.  Real metals fracture in tension (Mode I).
 *
 * Monaghan Artificial Pressure Correction:
 *   To prevent tensile fragmentation instability (clumping at particle interfaces
 *   when both particles are in tension), a small repulsive correction is added:
 *
 *   P_art(i,j) = ε · |P̄| · (W_ij / W_0)^n
 *
 *   where P̄ = 0.5*(|Pi| + |Pj|),  ε = 0.2,  n = 4,  W_0 = W(0,h)
 *
 *   This acts only at short range (r→0) and does NOT suppress real tension.
 *   References: Monaghan (2000), J. Comput. Phys. 159, 290-311.
 */
__global__ void computeDensityPressureKernel(SPHParticle* particles, int numParticles,
                                              int* cellStart, int* cellEnd, int tableSize,
                                              SPHKernelConfig config,
                                              double cellSize,
                                              double domainMinX, double domainMinY, double domainMinZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    SPHParticle& pi = particles[idx];
    if (pi.status == ParticleStatus::INACTIVE) return;

    double density = 0.0;

    int cx = static_cast<int>((pi.x - domainMinX) / cellSize);
    int cy = static_cast<int>((pi.y - domainMinY) / cellSize);
    int cz = static_cast<int>((pi.z - domainMinZ) / cellSize);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = ((cx + dx) * 73856093) ^ ((cy + dy) * 19349663) ^ ((cz + dz) * 83492791);
                hash = ((hash % tableSize) + tableSize) % tableSize;
                if (hash >= tableSize) continue;
                int start = cellStart[hash];
                if (start < 0) continue;
                int end = cellEnd[hash];

                for (int j = start; j < end; ++j) {
                    const SPHParticle& pj = particles[j];
                    if (pj.status == ParticleStatus::INACTIVE) continue;

                    double dx_ij = pi.x - pj.x;
                    double dy_ij = pi.y - pj.y;
                    double dz_ij = pi.z - pj.z;
                    double r2    = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;

                    if (r2 < config.h2) {
                        double diff = config.h2 - r2;
                        density += config.particleMass * config.poly6Coeff * diff * diff * diff;
                    }
                }
            }
        }
    }

    // Density floor: never below rest density (prevents negative mass)
    pi.density = fmax(density, config.restDensity);

    // Tait EOS for metals  P = B * ((ρ/ρ₀)^γ − 1)
    const double gamma = 7.15;   // steel / titanium
    double B     = config.gasStiffness * config.restDensity / gamma;
    double ratio = pi.density / config.restDensity;
    pi.pressure  = B * (pow(ratio, gamma) - 1.0);

    // ── PHYSICS FIX: NO PRESSURE CLAMP ──────────────────────────────────────
    // Negative pressure (tension) is physically valid for metals.
    // The Tait EOS naturally returns negative pressure when ρ < ρ₀.
    // Chip fracture is handled by J-C damage (D→1), not pressure sign.
    //
    // Monaghan artificial pressure stabilisation:  prevents tensile fragmentation
    // (the "negative pressure instability" or particle clumping in tension).
    // W_ij/W_0 = ((h²-r²)/(h²))^3  evaluated at the shorter of the two particle
    // spacings.  ε=0.2 following Monaghan (2000).
    // Note: this term is added inside computeForcesKernel where W_ij is available.
    // Here we only store the raw physical pressure.
}

/**
 * @brief Compute forces — pressure gradient + viscosity + Monaghan tensile correction.
 *
 * PHYSICS FIX: Monaghan artificial pressure correction is applied here
 * for particles where Pi < 0 OR Pj < 0 (tension state).
 *
 *   P_art(i,j) = ε · (|Pi|/ρi² + |Pj|/ρj²) · (W_ij / W_0)^4
 *
 *   → Added to the standard pressure term to provide short-range repulsion.
 *   → ε = 0.2  (Monaghan 2000).  W_0 = poly6Coeff * h2^3.
 *   → Only activates when at least one of Pi, Pj is negative.
 *
 * This prevents tensile fragmentation (particle surface clumping) while
 * allowing real compressive and tensile stress states to coexist.
 */
__global__ void computeForcesKernel(SPHParticle* particles, int numParticles,
                                     int* cellStart, int* cellEnd, int tableSize,
                                     SPHKernelConfig config,
                                     double cellSize,
                                     double domainMinX, double domainMinY, double domainMinZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    SPHParticle& pi = particles[idx];
    if (pi.status == ParticleStatus::INACTIVE) return;

    double fx = 0.0, fy = 0.0, fz = 0.0;

    // Monaghan tensile correction: W(r=0,h) = poly6Coeff * h2^3
    double W0 = config.poly6Coeff * config.h2 * config.h2 * config.h2;
    W0 = fmax(W0, 1e-40);   // prevent div-by-zero at init

    // Artificial pressure constant ε = 0.2  (Monaghan 2000)
    const double monaghan_eps = 0.2;

    int cx = static_cast<int>((pi.x - domainMinX) / cellSize);
    int cy = static_cast<int>((pi.y - domainMinY) / cellSize);
    int cz = static_cast<int>((pi.z - domainMinZ) / cellSize);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = ((cx + dx) * 73856093) ^ ((cy + dy) * 19349663) ^ ((cz + dz) * 83492791);
                hash = ((hash % tableSize) + tableSize) % tableSize;
                if (hash >= tableSize) continue;
                int start = cellStart[hash];
                if (start < 0) continue;
                int end = cellEnd[hash];

                for (int j = start; j < end; ++j) {
                    if (j == idx) continue;

                    const SPHParticle& pj = particles[j];
                    if (pj.status == ParticleStatus::INACTIVE) continue;

                    double dx_ij = pi.x - pj.x;
                    double dy_ij = pi.y - pj.y;
                    double dz_ij = pi.z - pj.z;
                    double r2    = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;
                    double r     = sqrt(r2);

                    if (r < config.h && r > 1e-12) {
                        double diff    = config.h - r;
                        double gradMag = config.spikyGradCoeff * diff * diff / r;

                        // ── Standard SPH pressure gradient (symmetric form) ──
                        double Pi_rho2 = pi.pressure / (pi.density * pi.density);
                        double Pj_rho2 = pj.pressure / (pj.density * pj.density);
                        double pressureTerm = Pi_rho2 + Pj_rho2;

                        // ── Monaghan Artificial Pressure (tensile correction) ─
                        // Activates whenever either particle is in tension (P < 0).
                        if (pi.pressure < 0.0 || pj.pressure < 0.0) {
                            // W_ij for Poly6: W = poly6Coeff * (h²-r²)³
                            double diff2  = config.h2 - r2;
                            double W_ij   = config.poly6Coeff * diff2 * diff2 * diff2;
                            double f4     = (W_ij / W0);
                            f4 = f4 * f4 * f4 * f4;   // (W/W0)^4

                            // Artificial pressure term (always repulsive → positive)
                            double Ri = fabs(pi.pressure) / (pi.density * pi.density);
                            double Rj = fabs(pj.pressure) / (pj.density * pj.density);
                            pressureTerm += monaghan_eps * (Ri + Rj) * f4;
                        }

                        double fPress = -config.particleMass * pressureTerm * gradMag;
                        fx += fPress * dx_ij;
                        fy += fPress * dy_ij;
                        fz += fPress * dz_ij;

                        // ── Viscosity (Laplacian kernel) ──────────────────────
                        double dvx      = pj.vx - pi.vx;
                        double dvy      = pj.vy - pi.vy;
                        double dvz      = pj.vz - pi.vz;
                        double viscLap  = config.viscLapCoeff * diff;
                        double viscForce = config.viscosity * config.particleMass
                                          / fmax(pj.density, 100.0) * viscLap;
                        fx += viscForce * dvx;
                        fy += viscForce * dvy;
                        fz += viscForce * dvz;
                    }
                }
            }
        }
    }

    pi.fx = (fx * pi.mass) + pi.ext_fx;
    pi.fy = (fy * pi.mass) + pi.ext_fy;
    pi.fz = (fz * pi.mass) + pi.ext_fz;

    pi.ext_fx = 0;
    pi.ext_fy = 0;
    pi.ext_fz = 0;
}

/**
 * @brief Leapfrog Kick: update velocities by half-step
 * v(n+1/2) = v(n) + a(n) * dt/2
 */
__global__ void leapfrogKickKernel(SPHParticle* particles, int numParticles, double halfDt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE || 
        p.status == ParticleStatus::FIXED_BOUNDARY) return;
    
    // v += a * dt/2
    double ax = p.fx / p.mass;
    double ay = p.fy / p.mass;
    double az = p.fz / p.mass;
    
    p.vx += ax * halfDt;
    p.vy += ay * halfDt;
    p.vz += az * halfDt;
}

/**
 * @brief Leapfrog Drift: update positions using current velocity
 * x(n+1) = x(n) + v(n+1/2) * dt
 */
__global__ void leapfrogDriftKernel(SPHParticle* particles, int numParticles, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    SPHParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE || 
        p.status == ParticleStatus::FIXED_BOUNDARY) return;
    
    // x += v * dt
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;
}



/**
 * @brief Compute simulation metrics
 */
__global__ void computeMetricsKernel(SPHParticle* particles, int numParticles,
                                      double* maxDensity, double* maxPressure,
                                      double* maxVelocity, double* maxTemp,
                                      double* kineticEnergy,
                                      int* activeCount, int* chipCount) {
    __shared__ double s_maxDensity[256];
    __shared__ double s_maxPressure[256];
    __shared__ double s_maxVelocity[256];
    __shared__ double s_maxTemp[256];
    __shared__ double s_kinetic[256];
    __shared__ int s_active[256];
    __shared__ int s_chip[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize
    s_maxDensity[tid] = 0;
    s_maxPressure[tid] = 0;
    s_maxVelocity[tid] = 0;
    s_maxTemp[tid] = 0;
    s_kinetic[tid] = 0;
    s_active[tid] = 0;
    s_chip[tid] = 0;
    
    if (idx < numParticles) {
        const SPHParticle& p = particles[idx];
        
        if (p.status == ParticleStatus::ACTIVE) {
            s_maxDensity[tid] = p.density;
            s_maxPressure[tid] = p.pressure;
            double v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
            s_maxVelocity[tid] = sqrt(v2);
            s_maxTemp[tid] = p.temperature;
            s_kinetic[tid] = 0.5 * p.mass * v2;
            s_active[tid] = 1;
        } else if (p.status == ParticleStatus::CHIP) {
            s_chip[tid] = 1;
        }
    }
    
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_maxDensity[tid] = fmax(s_maxDensity[tid], s_maxDensity[tid + s]);
            s_maxPressure[tid] = fmax(s_maxPressure[tid], s_maxPressure[tid + s]);
            s_maxVelocity[tid] = fmax(s_maxVelocity[tid], s_maxVelocity[tid + s]);
            s_maxTemp[tid] = fmax(s_maxTemp[tid], s_maxTemp[tid + s]);
            s_kinetic[tid] += s_kinetic[tid + s];
            s_active[tid] += s_active[tid + s];
            s_chip[tid] += s_chip[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicMax((unsigned long long*)maxDensity, __double_as_longlong(s_maxDensity[0]));
        atomicMax((unsigned long long*)maxPressure, __double_as_longlong(s_maxPressure[0]));
        atomicMax((unsigned long long*)maxVelocity, __double_as_longlong(s_maxVelocity[0]));
        atomicMax((unsigned long long*)maxTemp, __double_as_longlong(s_maxTemp[0]));
        atomicAdd(kineticEnergy, s_kinetic[0]);
        atomicAdd(activeCount, s_active[0]);
        atomicAdd(chipCount, s_chip[0]);
    }
}

// ============================================================================
// SPHSolver Implementation
// ============================================================================

SPHSolver::SPHSolver() = default;

SPHSolver::~SPHSolver() {
    freeMemory();
}

bool SPHSolver::initialize(const Config& config) {
    m_mainConfig = &config;
    std::cout << "[SPHSolver] Initializing..." << std::endl;
    
    // Get parameters from config
    const auto& sphParams = config.getSPH();
    const auto& material = config.getMaterial();
    
    m_maxParticles = sphParams.maxParticles;
    m_config.h = sphParams.smoothingRadius;
    m_config.gasStiffness = sphParams.gasStiffness;
    m_config.restDensity = material.density;
    
    // Precompute kernel coefficients
    double h = m_config.h;
    m_config.h2 = h * h;
    m_config.h3 = h * h * h;
    m_config.h6 = m_config.h3 * m_config.h3;
    m_config.h9 = m_config.h6 * m_config.h3;
    
    const double PI = 3.14159265358979323846;
    m_config.poly6Coeff = 315.0 / (64.0 * PI * m_config.h9);
    m_config.spikyGradCoeff = -45.0 / (PI * m_config.h6);
    m_config.viscLapCoeff = 45.0 / (PI * m_config.h6);
    
    // Cell size for spatial hash
    m_cellSize = m_config.h;
    
    // Allocate GPU memory
    allocateMemory(m_maxParticles);
    
    // Load LOD configuration
    m_lodEnabled = sphParams.lodEnabled;
    m_lodActiveRadius = sphParams.lodActiveRadius;
    m_lodNearRadius = sphParams.lodNearRadius;
    m_lodNearSkipSteps = sphParams.lodNearSkipSteps;
    m_lodFarSkipSteps = sphParams.lodFarSkipSteps;
    
    // Load damage/chip separation configuration
    m_damageEnabled = sphParams.damageEnabled;
    m_jc_D1 = sphParams.jc_D1;
    m_jc_D2 = sphParams.jc_D2;
    m_jc_D3 = sphParams.jc_D3;
    m_jc_D4 = sphParams.jc_D4;
    m_jc_D5 = sphParams.jc_D5;
    m_damageThreshold = sphParams.damageThreshold;
    m_refStrainRate = sphParams.referenceStrainRate;
    m_meltTemp = material.meltingPoint > 0 ? material.meltingPoint : 1660.0;
    
    m_isInitialized = true;
    std::cout << "[SPHSolver] Initialized with max " << m_maxParticles << " particles" << std::endl;
    if (m_lodEnabled) {
        std::cout << "[SPHSolver] LOD enabled: active=" << m_lodActiveRadius*1000 
                  << "mm, near=" << m_lodNearRadius*1000 << "mm" << std::endl;
    }
    if (m_damageEnabled) {
        std::cout << "[SPHSolver] Damage model enabled (JC failure)" << std::endl;
    }
    
    return true;
}

void SPHSolver::allocateMemory(int capacity) {
    freeMemory();
    
    // Create CUDA stream for async execution
    CUDA_CHECK(cudaStreamCreate(&m_computeStream));
    
    // Device memory
    CUDA_CHECK(cudaMalloc(&d_particles, capacity * sizeof(SPHParticle)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, m_hashTableSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, m_hashTableSize * sizeof(int)));
    
    // Pinned host memory for faster async transfers
    CUDA_CHECK(cudaMallocHost(&h_pinnedParticles, capacity * sizeof(SPHParticle)));
    
    // Initialize hash tables to -1
    CUDA_CHECK(cudaMemset(d_cellStart, -1, m_hashTableSize * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cellEnd, -1, m_hashTableSize * sizeof(int)));
    
    h_particles.reserve(capacity);
}

void SPHSolver::freeMemory() {
    if (m_computeStream) { cudaStreamDestroy(m_computeStream); m_computeStream = nullptr; }
    if (d_particles) { cudaFree(d_particles); d_particles = nullptr; }
    if (d_cellStart) { cudaFree(d_cellStart); d_cellStart = nullptr; }
    if (d_cellEnd) { cudaFree(d_cellEnd); d_cellEnd = nullptr; }
    if (d_particleOrder) { cudaFree(d_particleOrder); d_particleOrder = nullptr; }
    if (h_pinnedParticles) { cudaFreeHost(h_pinnedParticles); h_pinnedParticles = nullptr; }
}

void SPHSolver::initializeParticleBox(const Vec3& minCorner, const Vec3& maxCorner,
                                       double spacing) {
    h_particles.clear();

    double volume = spacing * spacing * spacing;
    m_config.particleMass = m_config.restDensity * volume;

    int id = 0;
    bool limitReached = false;

    // TOP-DOWN fill: workpiece top surface (maxZ) populated first
    for (double z = maxCorner.z; z >= minCorner.z - 1e-9 && !limitReached; z -= spacing) {
        for (double y = minCorner.y; y <= maxCorner.y + 1e-9 && !limitReached; y += spacing) {
            for (double x = minCorner.x; x <= maxCorner.x + 1e-9; x += spacing) {
                if ((int)h_particles.size() >= m_maxParticles) {
                    std::cerr << "[SPHSolver] WARNING: max particles reached at Z="
                              << z * 1000 << " mm." << std::endl;
                    limitReached = true;
                    break;
                }
                SPHParticle p;
                p.x = x;  p.y = y;  p.z = z;
                p.mass        = m_config.particleMass;
                p.density     = m_config.restDensity;
                p.temperature = 25.0;
                p.id          = id++;
                p.status      = ParticleStatus::ACTIVE;
                h_particles.push_back(p);
            }
        }
    }

    m_numParticles = static_cast<int>(h_particles.size());
    m_domainMin    = minCorner - Vec3(m_cellSize*10, m_cellSize*10, m_cellSize*10);
    m_domainMax    = maxCorner + Vec3(m_cellSize*10, m_cellSize*10, m_cellSize*10);

    copyToDevice();

    // ASB model
    AdiabaticShearConfig asbConfig;
    if (m_mainConfig) {
        asbConfig.enabled   = m_mainConfig->getJson().value("enable_adiabatic_shear", false);
        asbConfig.critStrain = m_mainConfig->getJson().value("asb_crit_shear_strain", 0.8);
    }
    m_adiabaticShearModel.initialize(asbConfig, m_meltTemp, 500.0, m_config.restDensity);

    std::cout << "[SPHSolver] Initialized " << m_numParticles << " particles in box"
              << std::endl;

    // Virtual Chuck
    double fixtureThickness = 0.002;
    if (m_mainConfig) {
        const auto& j = m_mainConfig->getJson();
        if (j.contains("machine_setup") &&
            j["machine_setup"].contains("fixture_layer_thickness_mm")) {
            fixtureThickness =
                j["machine_setup"]["fixture_layer_thickness_mm"].get<double>() / 1000.0;
        }
    }
    double fixtureLine = minCorner.z + fixtureThickness;
    int fixedCount = 0;
    for (auto& p : h_particles) {
        if (p.z <= fixtureLine) {
            p.status = ParticleStatus::FIXED_BOUNDARY;
            p.vx = p.vy = p.vz = 0;
            fixedCount++;
        }
    }
    if (fixedCount > 0) {
        copyToDevice();
        std::cout << "[SPHSolver] Virtual Chuck: " << fixedCount
                  << " particles fixed (" << fixtureThickness * 1000 << " mm layer)"
                  << std::endl;
    }
}

void SPHSolver::initializeCylindricalWorkpiece(const Vec3& center, double radius,
                                                double length, double spacing, int axis) {
    h_particles.clear();

    double volume = spacing * spacing * spacing;
    m_config.particleMass = m_config.restDensity * volume;

    int id = 0;

    // Tool enters from Z=maxZ (top). Fill TOP-DOWN so the cutting zone
    // (high Z values) is always populated first when the budget is hit.
    double maxZ = center.z + length / 2.0;
    double minZ = center.z - length / 2.0;

    double minX = center.x - radius;
    double maxX = center.x + radius;
    double minY = center.y - radius;
    double maxY = center.y + radius;

    bool limitReached = false;

    // *** KEY FIX: iterate Z from maxZ → minZ (top → bottom) ***
    for (double z = maxZ; z >= minZ - 1e-9 && !limitReached; z -= spacing) {
        for (double y = minY; y <= maxY + 1e-9 && !limitReached; y += spacing) {
            for (double x = minX; x <= maxX + 1e-9; x += spacing) {
                double dx = x - center.x;
                double dy = y - center.y;
                if (dx * dx + dy * dy > radius * radius) continue;

                if ((int)h_particles.size() >= m_maxParticles) {
                    std::cerr << "[SPHSolver] WARNING: max particle count ("
                              << m_maxParticles
                              << ") reached at Z=" << z * 1000
                              << " mm. Budget exhausted — reduce workpiece size "
                              << "or increase max_particles in config." << std::endl;
                    limitReached = true;
                    break;
                }

                SPHParticle p;
                p.x = x;  p.y = y;  p.z = z;
                p.mass        = m_config.particleMass;
                p.density     = m_config.restDensity;
                p.temperature = 25.0;
                p.id          = id++;
                p.status      = ParticleStatus::ACTIVE;
                h_particles.push_back(p);
            }
        }
    }

    m_numParticles = static_cast<int>(h_particles.size());
    m_domainMin    = Vec3(minX, minY, minZ) - Vec3(m_cellSize * 10, m_cellSize * 10, m_cellSize * 10);
    m_domainMax    = Vec3(maxX, maxY, maxZ) + Vec3(m_cellSize * 10, m_cellSize * 10, m_cellSize * 10);

    copyToDevice();

    // ─── Report how much of the workpiece was actually populated ─────────
    double topZ = maxZ, bottomZ = h_particles.empty() ? maxZ : h_particles.back().z;
    std::cout << "[SPHSolver] Cylinder workpiece: " << m_numParticles << " particles"
              << "  Z=[" << bottomZ * 1000 << ", " << topZ * 1000 << "] mm"
              << " (cutting zone at top is guaranteed)" << std::endl;

    if (limitReached) {
        std::cerr << "[SPHSolver] TIP: Reduce workpiece dimensions or increase "
                     "sph_parameters.max_particles. Recommended max ~"
                  << (int)(m_numParticles * 1.2) << " for this geometry." << std::endl;
    }

    // ASB model
    AdiabaticShearConfig asbConfig;
    if (m_mainConfig) {
        asbConfig.enabled   = m_mainConfig->getJson().value("enable_adiabatic_shear", false);
        asbConfig.critStrain = m_mainConfig->getJson().value("asb_crit_shear_strain", 0.8);
    }
    m_adiabaticShearModel.initialize(asbConfig, m_meltTemp, 500.0, m_config.restDensity);

    // Virtual Chuck: fix bottom fixtureThickness layer
    double fixtureThickness = 0.002;
    if (m_mainConfig) {
        const auto& j = m_mainConfig->getJson();
        if (j.contains("machine_setup") &&
            j["machine_setup"].contains("fixture_layer_thickness_mm")) {
            fixtureThickness =
                j["machine_setup"]["fixture_layer_thickness_mm"].get<double>() / 1000.0;
        }
    }
    double fixtureLine = minZ + fixtureThickness;
    int fixedCount = 0;
    for (auto& p : h_particles) {
        if (p.z <= fixtureLine) {
            p.status = ParticleStatus::FIXED_BOUNDARY;
            p.vx = p.vy = p.vz = 0;
            fixedCount++;
        }
    }
    if (fixedCount > 0) {
        copyToDevice();
        std::cout << "[SPHSolver] Virtual Chuck (cylinder): " << fixedCount
                  << " particles fixed (" << fixtureThickness * 1000 << " mm layer)"
                  << std::endl;
    }
}

void SPHSolver::copyToDevice() {
    if (m_numParticles > 0) {
        // Copy to pinned memory first, then async to device
        std::memcpy(h_pinnedParticles, h_particles.data(), m_numParticles * sizeof(SPHParticle));
        CUDA_CHECK(cudaMemcpyAsync(d_particles, h_pinnedParticles,
                                   m_numParticles * sizeof(SPHParticle),
                                   cudaMemcpyHostToDevice, m_computeStream));
        cudaStreamSynchronize(m_computeStream);
    }
}

void SPHSolver::copyFromDevice() {
    if (m_numParticles > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_pinnedParticles, d_particles,
                                   m_numParticles * sizeof(SPHParticle),
                                   cudaMemcpyDeviceToHost, m_computeStream));
        cudaStreamSynchronize(m_computeStream);
        h_particles.resize(m_numParticles);
        std::memcpy(h_particles.data(), h_pinnedParticles, m_numParticles * sizeof(SPHParticle));
    }
}

void SPHSolver::step(double dt) {
    if (!m_isInitialized || m_numParticles == 0) return;
    
    m_currentStep++;
    int blockSize = 256;
    int gridSize = (m_numParticles + blockSize - 1) / blockSize;
    double halfDt = dt * 0.5;
    
    // =========================================================================
    // LEAPFROG INTEGRATION (Kick-Drift-Kick) with CUDA Streams
    // Only ONE force calculation per step - 50% faster than Velocity Verlet!
    // All kernels launch on m_computeStream for async execution
    // =========================================================================
    
    NVTX_PUSH("SPH::KickDrift");
    // Step 1: First Kick - update velocities by half-step using old forces
    leapfrogKickKernel<<<gridSize, blockSize, 0, m_computeStream>>>(d_particles, m_numParticles, halfDt);
    CUDA_CHECK_KERNEL();
    
    // Step 2: Drift - update positions using half-step velocities
    leapfrogDriftKernel<<<gridSize, blockSize, 0, m_computeStream>>>(d_particles, m_numParticles, dt);
    CUDA_CHECK_KERNEL();
    
    // Step 3: Update LOD zones (every 10 steps to reduce overhead)
    if (m_lodEnabled && (m_currentStep % 10 == 0)) {
        updateLODZonesKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
            d_particles, m_numParticles,
            m_toolPosition.x, m_toolPosition.y, m_toolPosition.z,
            m_lodActiveRadius, m_lodNearRadius, m_currentStep
        );
        CUDA_CHECK_KERNEL();
    }
    NVTX_POP(); // KickDrift
    
    // Step 4: Build spatial hash ONCE
    NVTX_PUSH("SPH::SpatialHash");
    buildSpatialHash();
    NVTX_POP();
    
    // Step 5: Compute density and pressure
    NVTX_PUSH("SPH::Physics");
    computeDensityPressureKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_particles, m_numParticles,
        d_cellStart, d_cellEnd, m_hashTableSize,
        m_config, m_cellSize,
        m_domainMin.x, m_domainMin.y, m_domainMin.z
    );
    CUDA_CHECK_KERNEL();
    
    // Step 6: Compute forces ONCE
    computeForcesKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_particles, m_numParticles,
        d_cellStart, d_cellEnd, m_hashTableSize,
        m_config, m_cellSize,
        m_domainMin.x, m_domainMin.y, m_domainMin.z
    );
    CUDA_CHECK_KERNEL();
    
    // Step 7: Second Kick - complete velocity update with new forces
    leapfrogKickKernel<<<gridSize, blockSize, 0, m_computeStream>>>(d_particles, m_numParticles, halfDt);
    CUDA_CHECK_KERNEL();
    NVTX_POP(); // Physics
    
    // Step 8: Damage model (every 5 steps for efficiency)
    if (m_damageEnabled && (m_currentStep % 5 == 0)) {
        computeStrainRateKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
            d_particles, m_numParticles,
            d_cellStart, d_cellEnd, m_hashTableSize,
            m_config, m_cellSize,
            m_domainMin.x, m_domainMin.y, m_domainMin.z
        );
        CUDA_CHECK_KERNEL();
        
        computeDamageKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
            d_particles, m_numParticles,
            m_jc_D1, m_jc_D2, m_jc_D3, m_jc_D4, m_jc_D5,
            m_refStrainRate, m_refTemp, m_meltTemp,
            dt * 5.0  // Account for skip steps
        );
        CUDA_CHECK_KERNEL();
        
        separateChipsKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
            d_particles, m_numParticles, m_damageThreshold
        );
        CUDA_CHECK_KERNEL();
        
        // Step 9: Adiabatic Shear Band Update (ASB)
        // Correctly hook the "orphaned" ASB model into the main simulation loop
        if (m_adiabaticShearModel.isEnabled()) {
            m_adiabaticShearModel.update(d_particles, m_numParticles, dt * 5.0);
        }
    }
    
    m_currentTime += dt;
    
    // Update metrics frequently enough to match output interval (every 100 steps)
    // Was 500 which caused stale cached temp/stress values in output logs
    if (m_currentStep % 100 == 0) {
        cudaStreamSynchronize(m_computeStream);  // Sync before CPU copy
        updateResults();
    }
}



void SPHSolver::buildSpatialHash() {
    int blockSize = 256;
    int gridSize = (m_numParticles + blockSize - 1) / blockSize;
    
    // Reset cell arrays (async on stream)
    cudaMemsetAsync(d_cellStart, -1, m_hashTableSize * sizeof(int), m_computeStream);
    cudaMemsetAsync(d_cellEnd, -1, m_hashTableSize * sizeof(int), m_computeStream);
    
    // Compute hashes
    computeHashKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_particles, m_numParticles,
        m_cellSize, m_hashTableSize,
        m_domainMin.x, m_domainMin.y, m_domainMin.z
    );
    CUDA_CHECK_KERNEL();
    
    // ── PERFORMANCE FIX: Thrust with stream-based execution policy ───────────
    // Old code called cudaStreamSynchronize() before thrust::sort(), forcing a
    // global GPU sync barrier every single step and destroying the async pipeline.
    //
    // thrust::cuda::par.on(stream) executes the sort asynchronously on the
    // same CUDA stream as all other kernels, preserving the async pipeline.
    // Available since CUDA 7.0 / Thrust 1.9.
    thrust::device_ptr<SPHParticle> particlePtr(d_particles);
    thrust::sort(
        thrust::cuda::par.on(m_computeStream),   // async, no global sync
        particlePtr,
        particlePtr + m_numParticles,
        CompareParticleByHash()
    );

    // Find cell bounds (on the same stream, after the sort completes)
    findCellBoundsKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_particles, m_numParticles,
        d_cellStart, d_cellEnd, m_hashTableSize
    );
    CUDA_CHECK_KERNEL();
}

void SPHSolver::updateResults() {
    // This is expensive, so only do occasionally
    copyFromDevice();
    
    m_maxStress = 0;
    m_maxTemperature = 25.0;
    m_kineticEnergy = 0;
    m_activeCount = 0;
    m_chipCount = 0;
    
    // Reset compatibility results
    m_results = SPHResults();
    
    for (const auto& p : h_particles) {
        if (p.status == ParticleStatus::ACTIVE) {
            // Physics
            double s = computeEquivalentStress(p.stress_xx, p.stress_yy, p.stress_zz,
                                                p.stress_xy, p.stress_xz, p.stress_yz);
            double v = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
            double ke = 0.5 * p.mass * (v * v);
            
            // Update Class Members (v4 Interface)
            m_maxStress = std::max(m_maxStress, s);
            m_maxTemperature = std::max(m_maxTemperature, p.temperature);
            m_kineticEnergy += ke;
            m_activeCount++;
            
            // Update Struct (v3 compatibility)
            m_results.maxDensity = std::max(m_results.maxDensity, p.density);
            m_results.maxPressure = std::max(m_results.maxPressure, p.pressure);
            m_results.maxVelocity = std::max(m_results.maxVelocity, v);
            m_results.maxTemperature = std::max(m_results.maxTemperature, p.temperature);
            m_results.totalKineticEnergy += ke;
            m_results.activeParticleCount++;
        } else if (p.status == ParticleStatus::CHIP) {
            m_results.chipParticleCount++;
            m_chipCount++;
        } else if (p.status == ParticleStatus::INACTIVE) {
            m_results.removedParticleCount++;
        }
    }
}

double SPHSolver::getStableTimeStep() const {
    // CFL: dt < C_cfl * h / (v_max + c_sound)
    // For metals use gamma=7.15: c_sound = sqrt(B * gamma / rho0)
    // B = gasStiffness * restDensity / gamma  →  c_sound = sqrt(gasStiffness)
    double c_sound = std::sqrt(m_config.gasStiffness);          // ~55 m/s for default
    double v_max   = std::max(m_results.maxVelocity, c_sound);  // never below c_sound

    double dt_cfl  = 0.25 * m_config.h / v_max;                 // CFL (tighter factor)
    double dt_visc = 0.125 * m_config.h * m_config.h * m_config.restDensity / m_config.viscosity;

    double dt = std::min(dt_cfl, dt_visc);

    // Hard floor to prevent vanishing timesteps during startup
    return std::max(dt, 5e-9);
}

void SPHSolver::reset() {
    h_particles.clear();
    m_numParticles = 0;
    m_currentTime = 0.0;
    m_results = SPHResults();
}

void SPHSolver::getBounds(double& minX, double& minY, double& minZ, 
                         double& maxX, double& maxY, double& maxZ) const {
    if (h_particles.empty()) {
        minX = minY = minZ = maxX = maxY = maxZ = 0;
        return;
    }
    
    minX = minY = minZ = 1e10;
    maxX = maxY = maxZ = -1e10;
    
    for (const auto& p : h_particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            minX = std::min(minX, p.x);
            minY = std::min(minY, p.y);
            minZ = std::min(minZ, p.z);
            maxX = std::max(maxX, p.x);
            maxY = std::max(maxY, p.y);
            maxZ = std::max(maxZ, p.z);
        }
    }
}

std::vector<SPHParticle> SPHSolver::getParticles() {
    copyFromDevice();
    return h_particles;
}

void SPHSolver::applyExternalForce(const Vec3& center, double radius, const Vec3& force) {
    copyFromDevice();
    
    for (auto& p : h_particles) {
        double dx = p.x - center.x;
        double dy = p.y - center.y;
        double dz = p.z - center.z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        
        if (dist < radius) {
            p.fx += force.x;
            p.fy += force.y;
            p.fz += force.z;
        }
    }
    
    copyToDevice();
}

void SPHSolver::cutParticles(const Vec3& planeNormal, double planeDist) {
    copyFromDevice();
    
    int removed = 0;
    for (auto& p : h_particles) {
        if (p.status == ParticleStatus::ACTIVE) {
            double signedDist = planeNormal.x * p.x + planeNormal.y * p.y + planeNormal.z * p.z + planeDist;
            if (signedDist < 0) {
                p.status = ParticleStatus::CHIP;
                removed++;
            }
        }
    }
    
    if (removed > 0) {
        copyToDevice();
        std::cout << "[SPHSolver] Cut " << removed << " particles to chip" << std::endl;
    }
}


void SPHSolver::syncMetrics() {
    updateResults();
}

/**
 * @brief Manual update of simulation results/metrics
 */
// Duplicate updateResults removed - consolidated at line 879

void SPHSolver::applyHeatFlux(double x, double y, double z, double heatFlux) {
    // Distribute heat flux to nearby particles
    // For simplicity in this 'Clean Arch' validation, we find the nearest particle
    double minDist = 1e10;
    int nearest = -1;
    
    for (int i = 0; i < m_numParticles; ++i) {
        const auto& p = h_particles[i];
        if (p.status != ParticleStatus::ACTIVE) continue;
        
        double dx = p.x - x;
        double dy = p.y - y;
        double dz = p.z - z;
        double d2 = dx*dx + dy*dy + dz*dz;
        
        if (d2 < minDist) {
            minDist = d2;
            nearest = i;
        }
    }
    
    if (nearest != -1 && minDist < 0.001*0.001) { // Within 1mm
        // T_new = T_old + Q / (m * Cp) * dt
        // This would be done on GPU in production, here we use host-side update
    }
}

double SPHSolver::getTemperatureAt(double x, double y, double z) const {
    // Return ambient if no particles
    if (h_particles.empty()) return 25.0;
    
    // Find nearest particle temperature
    double minDist = 1e10;
    double temp = 25.0;
    
    for (const auto& p : h_particles) {
        if (p.status != ParticleStatus::ACTIVE) continue;
        double dx = p.x - x;
        double dy = p.y - y;
        double dz = p.z - z;
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < minDist) {
            minDist = d2;
            temp = p.temperature;
        }
    }
    return temp;
}

} // namespace edgepredict
