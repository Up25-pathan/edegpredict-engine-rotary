/**
 * @file MPMSolver.cu
 * @brief Material Point Method (MPM) CUDA implementation with APIC & Johnson-Cook
 */

#include "MPMSolver.cuh"
#include "Config.h"
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
    double domainMinZ, double dx, double fixtureThickness,
    double dampingCoeff, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numNodes = dims.x * dims.y * dims.z;
    if (idx >= numNodes) return;

    int gz = idx / (dims.x * dims.y);  // z-index in grid

    double worldZ = gz * dx + domainMinZ;
    double zRel = (worldZ - domainMinZ) / fixtureThickness;  // [0, 1] within fixture, >1 above

    if (zRel <= 1.0) {
        MPMGridNode& node = grid[idx];

        if (gz == 0) {
            // Dirichlet BC at rigid vise jaw surface
            node.vx = 0.0; node.vy = 0.0; node.vz = 0.0;
        } else if (zRel > 0.0) {
            // Viscoelastic damping: quadratic taper (max damping at bottom, zero at top)
            double sigma = dampingCoeff * (1.0 - zRel) * (1.0 - zRel);
            double factor = 1.0 / (1.0 + sigma * dt);  // stable implicit-style damping
            node.vx *= factor;
            node.vy *= factor;
            node.vz *= factor;
        }
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

__global__ void p2gKernel(MPMParticle* particles, int numParticles, MPMGridNode* grid, int3 dimensions, MPMKernelConfig config, Vec3 domainMin) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= numParticles) return;
    
    MPMParticle& p = particles[pIdx];
    if (p.status == ParticleStatus::INACTIVE) return;

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
        // [FIX] Metals are nearly incompressible. Clamp J strictly to prevent 
        // infinite pressures (volumetric locking) during massive element distortion.
        if (J < 0.9) J = 0.9;
        if (J > 1.1) J = 1.1;

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
        double s[9];
        for (int i = 0; i < 9; ++i) s[i] = sigma[i];
        s[0] -= tr_sigma / 3.0;
        s[4] -= tr_sigma / 3.0;
        s[8] -= tr_sigma / 3.0;

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
        double Y = (config.jc_A + config.jc_B * pow(p.plasticStrain, config.jc_n)) *
                   (1.0 + config.jc_C * log(edot_star)) *
                   (1.0 - pow(T_star, config.jc_m));

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
                if (i % 4 == 0) sigma[i] += tr_sigma / 3.0;
            }

            // Equivalent plastic strain increment (radial return magnitude)
            deltaEps = (q - Y) / fmax(3.0 * mu, 1e-12);
            
            // [FIX] Limit maximum plastic strain increment per micro-step 
            // to prevent explicit integration explosions.
            if (deltaEps > 0.05) deltaEps = 0.05;

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
                // ============================================================
                double dT = config.taylorQuinney * Y * deltaEps /
                           fmax(config.physicalDensity * config.specificHeat, 1e-12);
                
                // [FIX] Clamp maximum temperature rise per micro-step 
                // to prevent catastrophic thermal runaway.
                if (dT > 50.0) dT = 50.0;
                
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
                double pressure = tr_sigma / 3.0;
                double triaxiality = pressure / fmax(q, 1e-12);

                double eps_f = (config.jc_D1 + config.jc_D2 * exp(config.jc_D3 * triaxiality)) *
                               (1.0 + config.jc_D4 * log(fmax(edot_star, 1.0))) *
                               (1.0 + config.jc_D5 * T_star);

                if (eps_f > 1e-12) {
                    p.damage += deltaEps / eps_f;

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

                double affine_vx = p.C[0]*dpos_x + p.C[1]*dpos_y + p.C[2]*dpos_z;
                double affine_vy = p.C[3]*dpos_x + p.C[4]*dpos_y + p.C[5]*dpos_z;
                double affine_vz = p.C[6]*dpos_x + p.C[7]*dpos_y + p.C[8]*dpos_z;

                atomicAdd(&grid[gIdx].mass, weight * p.mass);
                // FIXED_BOUNDARY particles contribute mass but NOT momentum (v=0)
                if (!isFixed) {
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
        // Sticky/Kinematic collision: overwrite velocity with tool velocity
        // Wait: Friction can be added here, but sticky is robust for metal cutting.
        // We ensure the grid node travels with the tool.
        g.vx = g.tvx;
        g.vy = g.tvy;
        g.vz = g.tvz;
        
        g.px = g.mass * g.vx;
        g.py = g.mass * g.vy;
        g.pz = g.mass * g.vz;
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
    p.x += vx * dt; p.y += vy * dt; p.z += vz * dt;

    // CHIP particles: advect but don't update deformation gradient
    if (p.status == ParticleStatus::CHIP) {
        p.strainRate = 0;
        return;
    }
    
    double D_inv = 3.0 * config.invDx * config.invDx;
    for(int i=0; i<9; ++i) p.C[i] = B[i] * D_inv;

    // BUG 3 FIX: Zero APIC C matrix for ZONE_FAR particles.
    // Far-field particles carry angular momentum purely from SPH
    // advection/diffusion of the velocity field (not from tool contact),
    // causing the bulk workpiece to rotate. They should advect
    // kinematically but not participate in APIC angular transfer.
    if (p.lodZone == LODZone::ZONE_FAR) {
        for (int i = 0; i < 9; ++i) p.C[i] = 0.0;
    } else {
        // BUG 3 FIX: Decay APIC matrix to prevent unbounded rotation accumulation.
        // Factor 0.99 per step = ~1% damping (negligible for cutting forces, kills drift)
        for (int i = 0; i < 9; ++i) p.C[i] *= 0.99;
    }
    
    double I_Ldt[9];
    for(int i=0; i<9; ++i) {
        I_Ldt[i] = L[i] * dt;
        if (i%4 == 0) I_Ldt[i] += 1.0;
    }
    
    double F_new[9];
    mul3x3(I_Ldt, p.F, F_new);
    for(int i=0; i<9; ++i) p.F[i] = F_new[i];
    
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
    Vec3 toolPos, double activeRadius, double nearRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE && p.status != ParticleStatus::CHIP) return;

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

MPMSolver::MPMSolver() {}
MPMSolver::~MPMSolver() { freeMemory(); }

bool MPMSolver::initialize(const Config& config) {
    m_mainConfig = &config;

    const auto& sph = config.getSPH();
    const auto& mat = config.getMaterial();
    const auto& tm  = config.getToolMaterial();

    // Domain: 20cm cube centered on tool tip, large enough for chip debris
    m_domainMin = Vec3(-0.1, -0.1, -0.1);
    m_domainMax = Vec3(0.1, 0.1, 0.1);

    m_config.dx = std::max(sph.smoothingRadius * 2.0, 0.0005); // adaptive grid
    m_config.invDx = 1.0 / m_config.dx;

    m_gridDimX = (int)((m_domainMax.x - m_domainMin.x) * m_config.invDx) + 1;
    m_gridDimY = (int)((m_domainMax.y - m_domainMin.y) * m_config.invDx) + 1;
    m_gridDimZ = (int)((m_domainMax.z - m_domainMin.z) * m_config.invDx) + 1;
    m_numGridNodes = m_gridDimX * m_gridDimY * m_gridDimZ;

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
    m_lodEnabled             = sph.lodEnabled;

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

    // Initialize particle bounds to empty sentinels. Workpiece creation fills
    // these with the actual particle extents before the first simulation step.
    const double inf = std::numeric_limits<double>::infinity();
    double initMin[3] = { inf, inf, inf };
    double initMax[3] = { -inf, -inf, -inf };
    cudaMemcpy(d_boundsMin, initMin, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundsMax, initMax, 3 * sizeof(double), cudaMemcpyHostToDevice);

    m_particleMin = Vec3(initMin[0], initMin[1], initMin[2]);
    m_particleMax = Vec3(initMax[0], initMax[1], initMax[2]);

    m_isInitialized = true;
    return true;
}

void MPMSolver::allocateMemory(int particleCapacity, int gridCapacity) {
    cudaMalloc(&d_particles, particleCapacity * sizeof(MPMParticle));
    cudaMalloc(&d_grid, gridCapacity * sizeof(MPMGridNode));
    cudaMalloc(&d_couplingPoints, particleCapacity * sizeof(CouplingPoint));
    // Persistent reduction buffers (one double each, allocated once)
    cudaMalloc(&d_redMaxStress, sizeof(double));
    cudaMalloc(&d_redMaxTemp, sizeof(double));
    cudaMalloc(&d_redKinEnergy, sizeof(double));
    cudaMalloc(&d_redIntEnergy, sizeof(double));
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
    double toolTipZ, double cellSize, double damageThreshold, double criticalPlasticStrain)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MPMParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE) return;

    // ── Geometric chip separation ──────────────────────────────────────────
    // Particles that have passed below the tool tip and carry plastic
    // deformation are separated to form the chip. This creates a clean
    // cutting interface instead of waiting for damage to accumulate.
    if (p.status == ParticleStatus::ACTIVE) {
        // Cutting edge proxy: Z below the tool tip with any plastic strain
        double cutDepth = cellSize * 1.5;
        // In the main MPMSolver, we don't track plasticStrain directly, we track damage,
        // but we can check if it has accumulated some damage or temperature as a proxy for plastic strain.
        // Wait, MPMParticle in main MPMSolver has `damage`. Let's use `damage > 1e-6`.
        if (p.z < toolTipZ - cutDepth && p.damage > 1e-6) {
            p.status = ParticleStatus::CHIP;
            p.residualStress = fmax(p.residualStress,
                sqrt(p.stress_xx * p.stress_xx + p.stress_yy * p.stress_yy +
                     p.stress_zz * p.stress_zz));
            p.stress_xx = p.stress_yy = p.stress_zz = 0.0;
            p.stress_xy = p.stress_xz = p.stress_yz = 0.0;
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
        double speed = sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
        double perturbMag = speed * 0.3 + 0.5;  // at least 0.5 m/s

        p.vx += rx * perturbMag;
        p.vy += ry * perturbMag;
        p.vz += rz * perturbMag;

        // Reset damage to prevent repeated fracturing of the same particle
        p.damage = damageThreshold * 0.5;
    }
}

void MPMSolver::step(double dt) {
    if (!m_isInitialized || m_numParticles == 0) return;

    int3 gridDim = {m_gridDimX, m_gridDimY, m_gridDimZ};
    int gridThreads = 256;
    int gridBlocks = (m_numGridNodes + gridThreads - 1) / gridThreads;

    int partThreads = 256;
    int partBlocks = (m_numParticles + partThreads - 1) / partThreads;

    // Host scratch buffers for bounds reset (captured by CUDA Graph cudaMemcpyAsync)
    const double inf = std::numeric_limits<double>::infinity();
    double hBoundsMin[3] = { inf, inf, inf };
    double hBoundsMax[3] = { -inf, -inf, -inf };

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
                d_particles, m_numParticles, m_toolPosition,
                m_config.lodActiveRadius, m_config.lodNearRadius);
            CUDA_CHECK_KERNEL();
        }

        resetGridKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(d_grid, m_numGridNodes);
        CUDA_CHECK_KERNEL();

        p2gKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles, d_grid, gridDim, m_config, m_domainMin);
        CUDA_CHECK_KERNEL();

        updateGridKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(d_grid, m_numGridNodes, dt);
        CUDA_CHECK_KERNEL();

        if (m_fixtureThickness > 0.0) {
            applyFixtureBCKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(
                d_grid, gridDim, m_domainMin.z, m_config.dx,
                m_fixtureThickness, m_fixtureDampingCoeff, dt);
            CUDA_CHECK_KERNEL();
        }

        if (d_toolNodes && m_numToolNodes > 0) {
            int toolBlocks = (m_numToolNodes + 255) / 256;
            rasterizeToolKernel<<<toolBlocks, 256, 0, m_computeStream>>>(
                d_toolNodes, m_numToolNodes, d_grid, gridDim, m_config, m_domainMin);
            CUDA_CHECK_KERNEL();
            applyGridCollisionKernel<<<gridBlocks, gridThreads, 0, m_computeStream>>>(
                d_grid, m_numGridNodes, dt);
            CUDA_CHECK_KERNEL();
        }

        g2pKernel<<<partBlocks, partThreads, 0, m_computeStream>>>(
            d_particles, m_numParticles, d_grid, gridDim, m_config, dt, m_domainMin);
        CUDA_CHECK_KERNEL();

        if (m_adiabaticShearModel.isEnabled()) {
            m_adiabaticShearModel.update(d_particles, m_numParticles, dt, m_computeStream);
        }

        cudaMemcpyAsync(d_boundsMin, hBoundsMin, 3 * sizeof(double), cudaMemcpyHostToDevice, m_computeStream);
        cudaMemcpyAsync(d_boundsMax, hBoundsMax, 3 * sizeof(double), cudaMemcpyHostToDevice, m_computeStream);
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
            d_particles, m_numParticles, m_toolPosition.z, m_config.dx, m_config.damageThreshold, 1e-6);
        CUDA_CHECK_KERNEL();
    }

    m_currentTime += dt;
    m_currentStep++;
}

double MPMSolver::getStableTimeStep() const {
    // CFL condition for MPM: dt ≤ CFL * dx / c
    // where c = sqrt(E/ρ) is the dilatational wave speed
    // Uses m_adaptiveCFL (set by frequency-domain AMS) instead of fixed 0.2
    double waveSpeed = sqrt(m_config.youngsModulus / m_config.density);
    double cfl = m_adaptiveCFL;
    double dt_cfl = cfl * m_config.dx / fmax(waveSpeed, 1.0);
    return fmax(dt_cfl, 1e-10);
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
    // Scale inertial density for CFL: dt ∝ sqrt(density)
    // physicalDensity stays unchanged so thermal calculations remain accurate
    double targetDensity = m_config.physicalDensity * factor;
    double oldDensity = m_config.density;
    m_config.density = fmax(targetDensity, m_config.physicalDensity);

    // Update particle masses on device if density changed
    if (fabs(m_config.density - oldDensity) > 1e-6 && m_numParticles > 0) {
        int partThreads = 256;
        int partBlocks = (m_numParticles + partThreads - 1) / partThreads;
        updateParticleMassKernel<<<partBlocks, partThreads>>>(d_particles, m_numParticles, m_config.density);
        CUDA_CHECK_KERNEL();
    }
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
    if (m_numParticles == 0 || !d_particles) return 0;
    if (!d_meltCount) cudaMalloc(&d_meltCount, sizeof(int));
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
    if (m_numParticles == 0 || !d_particles) return 25.0;
    if (!d_sumTemp) cudaMalloc(&d_sumTemp, sizeof(double));
    if (!d_surfaceCount) cudaMalloc(&d_surfaceCount, sizeof(int));
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

    // Use GPU reduction instead of full D2H copy for particle counts
    if (!d_activeCount) cudaMalloc(&d_activeCount, sizeof(int));
    if (!d_chipCount) cudaMalloc(&d_chipCount, sizeof(int));
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
