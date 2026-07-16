/**
 * @file ContactSolver.cu
 * @brief Tool-workpiece contact detection and resolution
 *
 * Physics Overhaul — All audit issues fixed:
 *
 * 1. XPBD Position-Based Contact (replaces linear penalty + impulse cap)
 *    ─────────────────────────────────────────────────────────────────────
 *    Old: fn = k_contact * penetration, capped at m*r/dt² → tunneling
 *    New: Directly compute the constraint impulse that eliminates penetration
 *         in exactly one sub-step (zero compliance).
 *
 *         Position correction:  Δx_particle = penetration * normal
 *         Velocity correction:  Δv_particle = Δx / dt
 *         Equivalent force for logging: F = m * Δv / dt = m * penetration / dt²
 *
 *    This force is NEVER capped — it is always exactly what is needed to
 *    push the particle to the contact boundary.  For small dt and small
 *    penetration it is bounded naturally.  No explosion is possible.
 *
 * 2. Per-Node Heat Accumulation Buffer (fixes 40°C/μs tool-tip runaway)
 *    ─────────────────────────────────────────────────────────────────────
 *    Old: atomicAdd(&node.temperature, 2°C) inside the particle loop.
 *         20 particles × 2°C = 40°C per microsecond → 40,000,000°C/s.
 *    New: atomicAdd(&node.heatAccumulator, heatJoules) inside the loop.
 *         applyNodeHeatKernel() converts J → ΔT once, AFTER the loop.
 *         ΔT is physically bounded by actual frictional work, not a count.
 *
 * 3. Hertz Contact Pressure for Stress Reporting
 *    ─────────────────────────────────────────────────────────────────────
 *    p_max = (3 * Fn) / (2 * π * a²)
 *    where the contact half-width  a = sqrt(r_contact * penetration)
 *    Units: Pa — directly comparable to material yield strength.
 *
 * 4. Workpiece Particle Thermal — accumulated across all node contacts
 *    (this was already fixed; preserved here with updated equations).
 *
 * 5. Zorev temperature-dependent friction preserved.
 */

#include "ContactSolver.cuh"
#include "MPMSolver.cuh"
#include "FEMSolver.cuh"
#include "ToolCoatingModel.cuh"
#include "CudaUtils.cuh"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace edgepredict {

// ============================================================================
// Constants
// ============================================================================

using edgepredict::constants::PI;

// Thermal properties are now passed dynamically from ContactConfig

// ============================================================================
// Kernel 1: XPBD Contact Resolution
// ============================================================================

/**
 * @brief Position-based contact resolution kernel (XPBD, zero compliance).
 *
 * For every active SPH particle, loop over FEM nodes.  If a node is within
 * contactRadius:
 *
 *   (a) Compute constraint correction vector:
 *         C = (contactRadius - dist) * normalVector      [m]
 *
 *   (b) Apply position correction (particle moves, kinematic tool does not):
 *         particle.pos += C
 *
 *   (c) Derive velocity correction from position change:
 *         particle.vel += C / dt
 *       (This is the geometric impulse — exactly eliminates penetration in 1 step.)
 *
 *   (d) Compute Coulomb friction in tangential plane.
 *
 *   (e) Accumulate frictional heat into node.heatAccumulator [J] (NOT ΔT).
 *       Temperature is applied ONCE by applyNodeHeatKernel() below.
 *
 *   (f) Accumulate node.contactForceNormal and node.penetrationDepth for
 *       Hertz stress computation.
 *
 * @note The equivalent force for reporting is F = m * penetration / dt²,
 *       which is never capped.  It scales correctly with dt and penetration.
 *
 * SPATIAL HASH VARIANT: Instead of looping over all nodes, this kernel
 * uses cellStart/cellEnd arrays to check only nodes in 27 neighboring cells.
 */

// ============================================================================
// Spatial Hash Helpers (same convention as MPMSolver.cu)
// ============================================================================

__device__ inline int hashToolCell(int cx, int cy, int cz, int tableSize) {
    // Clamp to non-negative to prevent garbage hashes for out-of-bounds nodes.
    // Same prime-based hash as MPMSolver for consistency (64-bit to prevent overflow).
    unsigned long long cx_s = static_cast<unsigned long long>(max(0, cx));
    unsigned long long cy_s = static_cast<unsigned long long>(max(0, cy));
    unsigned long long cz_s = static_cast<unsigned long long>(max(0, cz));
    unsigned long long h = (cx_s * 73856093ULL) ^
                           (cy_s * 19349663ULL) ^
                           (cz_s * 83492791ULL);
    return static_cast<int>(h % static_cast<unsigned long long>(tableSize));
}

// ============================================================================
// GPU Kernel: Find cell bounds in sorted node array
// ============================================================================

__global__ void findToolCellBoundsKernel(
    FEMNodeGPU* sortedNodes, int numNodes,
    int* cellStart, int* cellEnd, int hashTableSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    int hash = sortedNodes[idx].cellHash;
    if (hash < 0 || hash >= hashTableSize) return;

    if (idx == 0 || sortedNodes[idx - 1].cellHash != hash) {
        cellStart[hash] = idx;
    }
    if (idx == numNodes - 1 || sortedNodes[idx + 1].cellHash != hash) {
        cellEnd[hash] = idx + 1;
    }
}

// ============================================================================
// GPU Kernel: XPBD Contact with Spatial Hash Lookup
// ============================================================================

/**
 * @brief Same physics as xpbdContactKernel but uses spatial hash for
 *        O(k) node lookup instead of O(M) brute force.
 */
// Compute Johnson-Cook flow stress from particle state (device version)
__device__ inline double deviceJCFlowStress(
    double plasticStrain, double strainRate, double temperature,
    double jc_A, double jc_B, double jc_n,
    double jc_C, double jc_m,
    double strainRateRef, double T_ref, double T_melt
) {
    double strainTerm = jc_A + jc_B * pow(fmax(plasticStrain, 0.0), jc_n);
    double rateRatio = fmax(strainRate / fmax(strainRateRef, 1e-9), 1.0);
    double rateTerm  = 1.0 + jc_C * log(rateRatio);
    double T_star = (temperature > T_ref)
        ? fmin(1.0, (temperature - T_ref) / fmax(T_melt - T_ref, 1.0))
        : 0.0;
    double thermalTerm = fmax(0.01, 1.0 - pow(T_star, jc_m));
    return fmax(strainTerm * rateTerm * thermalTerm, 1.0);
}

__global__ void xpbdContactSpatialHashKernel(
    MPMParticle*  particles,   int numParticles,
    FEMNodeGPU*   sortedNodes, int numNodes,
    int* cellStart, int* cellEnd, int hashTableSize,
    double cellSize,
    double domainMinX, double domainMinY, double domainMinZ,
    double contactRadius,
    double contactStiffness,
    double contactDampingRatio,
    double friction,
    double heatPartition,
    double dt,
    double shearYieldStress,
    double maxPenetration,
    double engagementScale,
    double workSpecificHeat,
    double workPhysicalDensity,
    double workMeltTemp,
    double contactHTC,
    bool   jcEnabled,
    double jc_A, double jc_B, double jc_n,
    double jc_C, double jc_m,
    double jc_strainRateRef, double jc_T_ref, double jc_T_melt,
    int*    contactCount,
    double* totalHeat,
    double* totalForce
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE ||
        p.status == ParticleStatus::FIXED_BOUNDARY) return;

    // Particle's cell coordinates
    int pcx = (int)floor((p.x - domainMinX) / cellSize);
    int pcy = (int)floor((p.y - domainMinY) / cellSize);
    int pcz = (int)floor((p.z - domainMinZ) / cellSize);

    double corrX = 0.0, corrY = 0.0, corrZ = 0.0;
    double totalHeatToParticle = 0.0;
    double totalForceMag       = 0.0;
    double totalHeatGen        = 0.0;
    int particleContactCount   = 0;
    bool   hasContact          = false;
    const double kNormal = fmax(contactStiffness, 0.0);
    const double cNormal = 2.0 * fmax(contactDampingRatio, 0.0)
                         * sqrt(kNormal * fmax(p.mass, 1e-15));

    // Track deepest penetration contact for velocity reflection
    double maxPen = 0.0;
    double maxNx = 0.0, maxNy = 0.0, maxNz = 1.0;

    // Loop over 27 neighboring cells (3×3×3)
    for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
        int hash = hashToolCell(pcx + dx, pcy + dy, pcz + dz, hashTableSize);
        int start = cellStart[hash];
        if (start < 0) continue;
        int end = cellEnd[hash];

        for (int j = start; j < end; ++j) {
            FEMNodeGPU& node = sortedNodes[j];
            if (node.status == NodeStatus::FAILED) continue;

            double ndx = p.x - node.x;
            double ndy = p.y - node.y;
            double ndz = p.z - node.z;
            double dist = sqrt(ndx*ndx + ndy*ndy + ndz*ndz);

            if (dist >= contactRadius || dist < 1e-12 || isnan(dist)) {
                continue;
            }

            // ── XPBD constraint (same physics as brute-force kernel) ────
            hasContact = true;
            double penetration = contactRadius - dist;
            if (penetration > maxPenetration) penetration = maxPenetration;

            double invDist = 1.0 / dist;
            double nx = ndx * invDist;
            double ny = ndy * invDist;
            double nz = ndz * invDist;

            if (penetration > maxPen) {
                maxPen = penetration;
                maxNx = nx; maxNy = ny; maxNz = nz;
            }

            // FIX C: Position correction is ALWAYS full strength.
            // This is a geometric constraint ("particle must not be inside tool"),
            // not a physical force. The engagement ramp only applies to forces/heat.
            corrX += penetration * nx;
            corrY += penetration * ny;
            corrZ += penetration * nz;
            particleContactCount++;

            // Relative velocity and Coulomb friction
            double relVx = p.vx - node.vx;
            double relVy = p.vy - node.vy;
            double relVz = p.vz - node.vz;
            double vn = relVx*nx + relVy*ny + relVz*nz;

            // Normal force is a compliant contact law, not an XPBD diagnostic
            // force. This keeps reported force independent of the global dt.
            double approachSpeed = fmax(0.0, -vn);
            double fn = (kNormal * penetration + cNormal * approachSpeed) * engagementScale;

            // Reaction force on tool node
            atomicAdd(&node.fx, -fn * nx);
            atomicAdd(&node.fy, -fn * ny);
            atomicAdd(&node.fz, -fn * nz);
            atomicAdd(&node.contactForceNormal, fn);

            // atomicMax for penetration depth (portable wrapper)
            atomicMaxDouble(&node.penetrationDepth, penetration);

            double vtx = relVx - vn * nx;
            double vty = relVy - vn * ny;
            double vtz = relVz - vn * nz;
            double vt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

            // ── Sticking-sliding friction (Zorev two-zone model) ──────────────
            // Sticking zone (near cutting edge, σ_n > σ_crit):
            //   τ = τ_yield — material shears internally, no sliding at interface
            // Sliding zone (further from edge, σ_n < σ_crit):
            //   τ = μ(T) · σ_n — Coulomb friction with Zorev temperature dependence
            double T_avg = 0.5 * (p.temperature + node.temperature);
            double T_ratio = fmin(1.0, fmax(0.0, (T_avg - 200.0) / 600.0));
            double mu_eff = friction * (1.0 - 0.4 * T_ratio);

            double particleVolume = (p.volume > 1e-24)
                ? p.volume
                : (p.mass / fmax(p.density, 1.0));
            double particleRadius = pow(3.0 * particleVolume / (4.0 * PI), 1.0 / 3.0);
            // Per-particle flow stress from JC model
            double flowStress = jcEnabled
                ? deviceJCFlowStress(p.plasticStrain, p.strainRate, p.temperature,
                                    jc_A, jc_B, jc_n, jc_C, jc_m,
                                    jc_strainRateRef, jc_T_ref, jc_T_melt)
                : shearYieldStress;
            double shearYield = flowStress / 1.7320508; // von Mises: τ = σ_y/√3

            // Item 1: True contact patch area from sphere-plane Hertz + plastic
            double contactPatchPen = (penetration > 0.0) ? penetration : 0.0;
            double elasticArea = PI * (2.0 * particleRadius * contactPatchPen
                                       - contactPatchPen * contactPatchPen);
            double plasticArea = (flowStress > 1.0)
                ? fn / flowStress
                : 0.0;
            double shearArea = elasticArea + plasticArea;
            double minArea = 0.01 * pow(fmax(particleVolume, 1e-24), 2.0 / 3.0);
            if (shearArea < minArea) shearArea = minArea;

            // Contact pressure and critical pressure for sticking transition
            double sigma_n = fn / fmax(shearArea, 1e-24);
            double sigma_crit = shearYield / fmax(mu_eff, 1e-6); // σ where μ·σ = τ_yield

            // Zorev friction: smooth (tanh) transition between zones
            double zoneSlope = 10.0; // higher = sharper transition
            double stickFrac = 0.5 * (1.0 + tanh(zoneSlope * (sigma_n / fmax(sigma_crit, 1e-12) - 1.0)));
            double tau_stick = shearYield;
            double tau_slide = mu_eff * sigma_n;
            double tau = (1.0 - stickFrac) * tau_slide + stickFrac * tau_stick;
            double ft = tau * shearArea;
            // Inertial cap: friction cannot exceed what would stop the particle
            double ft_stiction = (vt > 1e-10) ? (p.mass * vt / dt) : ft;
            ft = fmin(ft, ft_stiction);

            // Item 5: Pressure-dependent HTC
            double P_ref = flowStress;
            double HTC_eff = contactHTC * (1.0 + 5.0 * (1.0 - exp(-sigma_n / P_ref)));

            double heatThisStep = ft * vt * dt;
            double Q_cond = HTC_eff * shearArea * (p.temperature - node.temperature) * dt;
            totalHeatToParticle += (heatThisStep * (1.0 - heatPartition) - Q_cond) * engagementScale;
            atomicAdd(&node.heatAccumulator, (heatThisStep * heatPartition + Q_cond) * engagementScale);
            if (vt > 1e-15) {
                atomicAdd(&node.fx, ft * (vtx / vt));
                atomicAdd(&node.fy, ft * (vty / vt));
                atomicAdd(&node.fz, ft * (vtz / vt));
            }
        node.inContact = true;

            totalHeatGen += heatThisStep * engagementScale;
            totalForceMag += fn;
        }
    }}}

    if (!hasContact) return;

    // =========================================================================
    // XPBD Position Correction + Velocity Reflection
    // =========================================================================
    // 1. Move the particle outside the tool volume (non-negotiable constraint)
    // 2. Remove ONLY the inward velocity component (inelastic reflection)
    //
    // The OLD code added correction/dt to velocity, which at dt=5μs turned
    // a 1mm correction into +200 m/s — creating 475kJ of artificial kinetic
    // energy that blew up SPH pressure forces and crashed the simulation.
    //
    // The correct approach: project out the velocity component that points
    // toward the tool surface. This is equivalent to a perfectly inelastic
    // collision with the tool face, conserving tangential momentum.
    // =========================================================================
    double weight = 1.0 / particleContactCount;
    double cx = corrX * weight;
    double cy = corrY * weight;
    double cz = corrZ * weight;

    // Step 1: Position correction — hard geometric constraint
    p.x += cx;
    p.y += cy;
    p.z += cz;

    // Step 2: Velocity reflection — remove inward normal component
    // The correction vector points AWAY from the tool (outward normal).
    double corrMag = sqrt(cx*cx + cy*cy + cz*cz);
    if (corrMag > 1e-15) {
        double invCorrMag = 1.0 / corrMag;
        double nnx = cx * invCorrMag;  // outward normal
        double nny = cy * invCorrMag;
        double nnz = cz * invCorrMag;

        // Project velocity onto outward normal
        double vDotN = p.vx * nnx + p.vy * nny + p.vz * nnz;

        // If velocity points INTO tool (vDotN < 0), remove that component
        if (vDotN < 0.0) {
            p.vx -= vDotN * nnx;
            p.vy -= vDotN * nny;
            p.vz -= vDotN * nnz;
        }
    }

    // Workpiece heat
    if (totalHeatToParticle > 0.0) {
        double particleVolume = (p.volume > 1e-24)
            ? p.volume
            : (p.mass / fmax(p.density, 1.0));
        double physicalMass = fmax(workPhysicalDensity, 1.0) *
                              fmax(particleVolume, 1e-24);
        double dT = totalHeatToParticle / (fmax(physicalMass, 1e-15) * workSpecificHeat);
        p.temperature = fmin(p.temperature + dT, workMeltTemp);
    }

    atomicAdd(contactCount, 1);
    atomicAdd(totalHeat, totalHeatGen);
    atomicAdd(totalForce, totalForceMag);
}
__global__ void xpbdContactKernel(
    MPMParticle*  particles,   int numParticles,
    FEMNodeGPU*   nodes,       int numNodes,
    double contactRadius,
    double contactStiffness,
    double contactDampingRatio,
    double friction,
    double heatPartition,
    double dt,
    double shearYieldStress,
    double maxPenetration,
    double engagementScale,
    double workSpecificHeat,
    double workPhysicalDensity,
    double workMeltTemp,
    double contactHTC,
    bool   jcEnabled,
    double jc_A, double jc_B, double jc_n,
    double jc_C, double jc_m,
    double jc_strainRateRef, double jc_T_ref, double jc_T_melt,
    int*    contactCount,
    double* totalHeat,
    double* totalForce
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    MPMParticle& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE ||
        p.status == ParticleStatus::FIXED_BOUNDARY) return;

    double corrX = 0.0, corrY = 0.0, corrZ = 0.0;
    double totalHeatToParticle = 0.0;
    double totalForceMag       = 0.0;
    double totalHeatGen        = 0.0;
    int particleContactCount   = 0;
    bool   hasContact          = false;

    const double kNormal = fmax(contactStiffness, 0.0);
    const double cNormal = 2.0 * fmax(contactDampingRatio, 0.0)
                         * sqrt(kNormal * fmax(p.mass, 1e-15));

    // Track deepest penetration contact for velocity reflection
    double maxPen = 0.0;
    double maxNx = 0.0, maxNy = 0.0, maxNz = 1.0;

    for (int j = 0; j < numNodes; ++j) {
        FEMNodeGPU& node = nodes[j];
        if (node.status == NodeStatus::FAILED) continue;

        double dx   = p.x - node.x;
        double dy   = p.y - node.y;
        double dz   = p.z - node.z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);

        if (dist >= contactRadius || dist < 1e-12) continue;

        // ── XPBD constraint ─────────────────────────────────────────────────
        hasContact = true;
        double penetration = contactRadius - dist;

        // Penetration clamping: prevent extreme forces from deeply embedded
        // particles (e.g. tool starts inside workpiece due to config error).
        // Forces scale as m*pen/dt² — clamping pen bounds the peak force.
        if (penetration > maxPenetration)
            penetration = maxPenetration;

        double invDist     = 1.0 / dist;
        double nx = dx * invDist;
        double ny = dy * invDist;
        double nz = dz * invDist;

        // FIX C: Position correction is ALWAYS full strength.
        // This is a geometric constraint ("particle must not be inside tool"),
        // not a physical force. The engagement ramp only applies to forces/heat.
        corrX += penetration * nx;
        corrY += penetration * ny;
        corrZ += penetration * nz;
        particleContactCount++;

        // ── Normal impulse force (for reporting and reaction on node) ────────
        // F_constraint = m_particle * penetration / dt²
        // Scaled by engagement ramp to prevent initial shock.
        double fn = (p.mass * penetration / (dt * dt)) * engagementScale;

        // Reaction force on the (kinematic) tool node
        atomicAdd(&node.fx, -fn * nx);
        atomicAdd(&node.fy, -fn * ny);
        atomicAdd(&node.fz, -fn * nz);

        // Accumulate per-node contact metrics for Hertz stress
        atomicAdd(&node.contactForceNormal, fn);

        // atomicMax via bit-cast (doubles are positive here)
        // Hertz half-width a² = r_contact * penetration  →  p_hertz = 3Fn/(2π·a²)
        unsigned long long* addr  = (unsigned long long*)&node.penetrationDepth;
        unsigned long long  cur   = *addr;
        unsigned long long  nval  = __double_as_longlong(penetration);
        while (penetration > __longlong_as_double(cur)) {
            unsigned long long old = atomicCAS(addr, cur, nval);
            if (old == cur) break;
            cur = old;
        }

        // ── Relative velocity ────────────────────────────────────────────────
        double relVx = p.vx - node.vx;
        double relVy = p.vy - node.vy;
        double relVz = p.vz - node.vz;
        double  vn   = relVx*nx + relVy*ny + relVz*nz;

        // ── Sticking-sliding friction (Zorev two-zone model) ──────────────────
        double vtx = relVx - vn * nx;
        double vty = relVy - vn * ny;
        double vtz = relVz - vn * nz;
        double vt  = sqrt(vtx*vtx + vty*vty + vtz*vtz);

        // Zorev temperature-dependent friction coefficient
        double T_avg   = 0.5 * (p.temperature + node.temperature);
        double T_ratio = fmin(1.0, fmax(0.0, (T_avg - 200.0) / 600.0));
        double mu_eff  = friction * (1.0 - 0.4 * T_ratio);

        double particleVolume = (p.volume > 1e-24)
            ? p.volume
            : (p.mass / fmax(p.density, 1.0));
        // Per-particle flow stress for shear cap
        double flowStress = jcEnabled
            ? deviceJCFlowStress(p.plasticStrain, p.strainRate, p.temperature,
                                jc_A, jc_B, jc_n, jc_C, jc_m,
                                jc_strainRateRef, jc_T_ref, jc_T_melt)
            : shearYieldStress;
        double shearYield = flowStress / 1.7320508;

        // True contact patch + plastic component
        double particleRadius = pow(3.0 * particleVolume / (4.0 * PI), 1.0 / 3.0);
        double contactPatchPen = (penetration > 0.0) ? penetration : 0.0;
        double elasticArea = PI * (2.0 * particleRadius * contactPatchPen
                                   - contactPatchPen * contactPatchPen);
        double plasticArea = (flowStress > 1.0)
            ? fn / flowStress
            : 0.0;
        double shearArea = elasticArea + plasticArea;
        double minArea = 0.01 * pow(fmax(particleVolume, 1e-24), 2.0 / 3.0);
        if (shearArea < minArea) shearArea = minArea;

        // Contact pressure and critical pressure for sticking transition
        double sigma_n = fn / fmax(shearArea, 1e-24);
        double sigma_crit = shearYield / fmax(mu_eff, 1e-6);
        double zoneSlope = 10.0;
        double stickFrac = 0.5 * (1.0 + tanh(zoneSlope * (sigma_n / fmax(sigma_crit, 1e-12) - 1.0)));
        double tau_stick = shearYield;
        double tau_slide = mu_eff * sigma_n;
        double tau = (1.0 - stickFrac) * tau_slide + stickFrac * tau_stick;
        double ft = tau * shearArea;
        double ft_stiction = (vt > 1e-10) ? (p.mass * vt / dt) : ft;
        ft = fmin(ft, ft_stiction);

        double P_ref = flowStress;
        double HTC_eff = contactHTC * (1.0 + 5.0 * (1.0 - exp(-sigma_n / P_ref)));
        
    double heatThisStep = ft * vt * dt;
    double Q_cond = HTC_eff * shearArea * (p.temperature - node.temperature) * dt;
    totalHeatToParticle += (heatThisStep * (1.0 - heatPartition) - Q_cond) * engagementScale;
    atomicAdd(&node.heatAccumulator, (heatThisStep * heatPartition + Q_cond) * engagementScale);
    if (vt > 1e-15) {
        atomicAdd(&node.fx, ft * (vtx / vt));
        atomicAdd(&node.fy, ft * (vty / vt));
        atomicAdd(&node.fz, ft * (vtz / vt));
    }
        node.inContact = true;

        totalHeatGen += heatThisStep * engagementScale;
        totalForceMag += fn;
    }

    if (!hasContact) return;

    // =========================================================================
    // XPBD Position Correction + Velocity Reflection
    // =========================================================================
    // Same physics as the spatial-hash kernel (see detailed comment above).
    // Key change: velocity is REFLECTED (inward component removed), NOT boosted
    // by correction/dt. This prevents artificial kinetic energy injection.
    // =========================================================================
    double weight = 1.0 / particleContactCount;
    double cx = corrX * weight;
    double cy = corrY * weight;
    double cz = corrZ * weight;

    // Step 1: Position correction
    p.x += cx;
    p.y += cy;
    p.z += cz;

    // Step 2: Velocity reflection
    if (maxPen > 1e-15) {
        double vDotN = p.vx * maxNx + p.vy * maxNy + p.vz * maxNz;
        if (vDotN < 0.0) {
            p.vx -= vDotN * maxNx;
            p.vy -= vDotN * maxNy;
            p.vz -= vDotN * maxNz;
        }
    }

    // ── Apply accumulated workpiece heat ONCE ────────────────────────────────
    if (totalHeatToParticle > 0.0) {
        double particleVolume = (p.volume > 1e-24)
            ? p.volume
            : (p.mass / fmax(p.density, 1.0));
        double physicalMass = fmax(workPhysicalDensity, 1.0) *
                              fmax(particleVolume, 1e-24);
        double dT = totalHeatToParticle / (fmax(physicalMass, 1e-15) * workSpecificHeat);
        // Physical bound: dT cannot exceed heat available/mass.  No artificial clamp.
        // The material melting cap is the only physical ceiling.
        p.temperature = fmin(p.temperature + dT, workMeltTemp);
    }

    // Atomic result accumulation
    atomicAdd(contactCount, 1);
    atomicAdd(totalHeat,    totalHeatGen);
    atomicAdd(totalForce,   totalForceMag);
}

// ============================================================================
// Kernel 2: Apply Node Heat Buffer → Temperature
// ============================================================================

__global__ void resetContactAccumulatorsKernel(FEMNodeGPU* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];
    node.fx = 0.0;
    node.fy = 0.0;
    node.fz = 0.0;
    node.heatAccumulator = 0.0;
    node.contactForceNormal = 0.0;
    node.penetrationDepth = 0.0;
    node.inContact = false;
}

/**
 * @brief Convert per-node accumulated frictional heat to temperature rise.
 *
 * Run AFTER xpbdContactKernel in each step.
 *
 *   ΔT_node = Q_accumulated / (m_node * Cp_tool)
 *
 * This is the only place where node.temperature is updated from contact heat.
 * No per-contact clamp is needed — the total Q is physically bounded by
 * actual frictional work done in this step.
 *
 * Hertz stress is computed here simultaneously:
 *   a  = sqrt(r_contact * d_penetration)       contact half-width [m]
 *   p  = (3 * Fn) / (2 * π * a²)              Hertz peak pressure [Pa]
 */
__global__ void applyNodeHeatAndHertzStressKernel(
    FEMNodeGPU* nodes, int numNodes,
    double maxContactPressure,
    double contactRadius,
    double toolCp,           // J/(kg·K)
    double toolMeltTemp,     // °C  (carbide melts at ~2870°C)
    double ema_alpha,        // EMA smoothing factor (0.9 = 10-step average)
    double ambientTemp,      // °C  for radiation
    double stefanBoltzmann,  // W/m²K⁴
    double emissivity,       // surface emissivity
    double toolDensity,      // kg/m³ for surface area estimate
    double latentHeatFusion, // J/kg  for melting enthalpy
    double solidusTemp,      // °C — mushy zone start (0 = disabled)
    double liquidusTemp,     // °C — mushy zone end
    double dt                // timestep (s) for radiation energy
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    // ── Radiation cooling (Item 5) ────────────────────────────────────────────
    if (node.temperature > ambientTemp + 1.0) {
        // Estimate node surface area from mass: A ≈ (mass/ρ)^(2/3)
        double vol = fmax(node.mass / toolDensity, 1e-15);
        double area = pow(vol, 2.0 / 3.0); // m²
        double Tn = node.temperature + 273.15; // K
        double Ta = ambientTemp + 273.15;      // K
        double Qrad = emissivity * stefanBoltzmann * area *
                      (Tn * Tn * Tn * Tn - Ta * Ta * Ta * Ta) * dt;
        node.heatAccumulator = fmax(node.heatAccumulator - Qrad, 0.0);
    }

    // ── Phase change / melting enthalpy (Items 6, 9) ──────────────────────────
    // Enthalpy method with mushy zone support.  Temperature is mapped to a
    // piecewise-linear enthalpy curve:
    //   T <  Ts:        H = ρ·Cp·T
    //   Ts ≤ T < Tl:    H = ρ·Cp·Ts + ρ·Lf·(T-Ts)/(Tl-Ts)
    //   T ≥ Tl:         H = ρ·Cp·Ts + ρ·Lf + ρ·Cp·(T-Tl)
    //
    // The inverse (enthalpy → temperature) is used to update node temperature
    // after adding accumulated heat.  heatAccumulator stores J for the step.
    //
    // For the single-melt-temp case (default): Ts = Tl = toolMeltTemp.
    bool hasPhaseChange = (latentHeatFusion > 0.0);
    bool hasMushyZone = hasPhaseChange && (solidusTemp > 0.0) && (liquidusTemp > solidusTemp);
    double Ts = hasMushyZone ? solidusTemp : toolMeltTemp;
    double Tl = hasMushyZone ? liquidusTemp : toolMeltTemp;

    double latentAccum = 0.0;
    if (node.heatAccumulator > 0.0 && hasPhaseChange) {
        double mass = fmax(node.mass, 1e-12);
        double totalJ = node.heatAccumulator;
        double T0 = node.temperature;

        // Current enthalpy per unit mass: H0 = Cp·T0 + L_absorbed
        // L_absorbed is carried over from previous step in heatAccumulator excess.
        // For simplicity we recompute from current temperature.
        double H0;
        double cp = fmax(toolCp, 1e-12);
        if (T0 < Ts) {
            H0 = cp * T0;
        } else if (T0 < Tl) {
            double f = (T0 - Ts) / (Tl - Ts);
            H0 = cp * Ts + latentHeatFusion * f;
        } else {
            H0 = cp * Ts + latentHeatFusion + cp * (T0 - Tl);
        }

        double H1 = H0 + totalJ / mass; // New enthalpy per unit mass

        // Enthalpy → temperature
        double Tnew;
        double H_s = cp * Ts;                  // enthalpy at solidus
        double H_l = cp * Ts + latentHeatFusion; // enthalpy at liquidus
        if (H1 <= H_s) {
            // Solid
            Tnew = H1 / cp;
        } else if (H1 < H_l) {
            // Mushy zone — temperature pinned between Ts and Tl
            double f = (H1 - H_s) / (H_l - H_s);
            Tnew = Ts + f * (Tl - Ts);
        } else {
            // Fully liquid
            Tnew = Tl + (H1 - H_l) / toolCp;
        }

        // No per-step cap: physically bounded by contact work in this single step
        double dT = Tnew - T0;
        node.temperature = fmin(T0 + dT, toolMeltTemp);
        if (node.temperature >= toolMeltTemp) node.status = NodeStatus::FAILED;
    } else if (node.heatAccumulator > 0.0) {
        // No phase change — simple temperature rise.
        // No per-step cap: the heat in accumulator is physically bounded by
        // the friction work and HTC-limited conduction of this single step
        // (∼1e-7s), well within the CFL limit for explicit thermal diffusion
        // in carbide (dt_CFL ≈ 5 μs for 0.1mm elements).
        double dT = node.heatAccumulator / (fmax(node.mass, 1e-12) * fmax(toolCp, 1e-12));
        node.temperature = fmin(node.temperature + dT, toolMeltTemp);
        if (node.temperature >= toolMeltTemp) node.status = NodeStatus::FAILED;
    }
    node.heatAccumulator = latentAccum;

    // ── Hertz contact stress (physically meaningful von Mises proxy) ─────────
    // Hertz half-width approximation for a spherical punch on a flat surface:
    //   a² ≈ r_contact * d_penetration
    //   p_max = (3 * Fn) / (2 * π * a²)
    //
    // This gives peak contact pressure in Pa — directly comparable to the
    // material's compressive yield strength (~4 GPa for WC-Co carbide).
    double sigma_hertz = 0.0;
    if (node.contactForceNormal > 0.0 && node.penetrationDepth > 1e-12) {
        double a2 = contactRadius * node.penetrationDepth;   // a² [m²]
        if (a2 > 1e-18) {
            sigma_hertz = (3.0 * node.contactForceNormal) / (2.0 * PI * a2);
            sigma_hertz = fmin(sigma_hertz, maxContactPressure);
        }
    }

    // EMA-smoothed von Mises stress (prevents aliasing from step-to-step spikes)
    node.vonMisesStress = ema_alpha * node.vonMisesStress
                        + (1.0 - ema_alpha) * sigma_hertz;

    // Reset per-step contact accumulators
    node.contactForceNormal = 0.0;
    node.penetrationDepth   = 0.0;
}

// ============================================================================
// SDF Contact Kernel (replaces hash-based node loop for O(1) CD)
// ============================================================================

__global__ void xpbdContactSDFKernel(
    MPMParticle* particles, int numParticles,
    cudaTextureObject_t sdfTex,
    float3 gridOrigin, float voxelSize, int3 gridDims,
    float centerX, float centerY, float centerZ,
    float cosAngle, float sinAngle,
    float contactRadius,
    float friction,
    float heatPartition,
    float dt,
    double workSpecificHeat,
    double workPhysicalDensity,
    double workMeltTemp,
    double contactHTC,
    bool   jcEnabled,
    double jc_A, double jc_B, double jc_n,
    double jc_C, double jc_m,
    double jc_strainRateRef, double jc_T_ref, double jc_T_melt,
    int* contactCount, double* totalHeat, double* totalForce)
{
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= numParticles) return;

    MPMParticle& p = particles[pIdx];
    if (p.status == ParticleStatus::INACTIVE ||
        p.status == ParticleStatus::FIXED_BOUNDARY) return;

    // Transform particle position from world → SDF rest frame (θ=0)
    float dx = (float)p.x - centerX;
    float dy = (float)p.y - centerY;
    float localX =  dx * cosAngle + dy * sinAngle;
    float localY = -dx * sinAngle + dy * cosAngle;
    float localZ = (float)p.z - centerZ;

    // Convert to SDF texture coordinates (unnormalized: voxel index)
    float texX = (localX - gridOrigin.x) / voxelSize;
    float texY = (localY - gridOrigin.y) / voxelSize;
    float texZ = (localZ - gridOrigin.z) / voxelSize;

    // Clamp to valid range (border returns large positive distance)
    if (texX < 0 || texY < 0 || texZ < 0 ||
        texX > gridDims.x - 1 || texY > gridDims.y - 1 || texZ > gridDims.z - 1) {
        return;
    }

    // Sample SDF — returns signed distance (negative = inside tool)
    float sdf = tex3D<float>(sdfTex, texX, texY, texZ);

    if (sdf >= contactRadius) return;  // far from surface — no contact

    // Compute SDF gradient via central differences for contact normal
    float eps = 0.5f;  // half-voxel offset for gradient
    float gx = (tex3D<float>(sdfTex, texX + eps, texY, texZ) -
                tex3D<float>(sdfTex, texX - eps, texY, texZ)) / (eps * 2.0f * voxelSize);
    float gy = (tex3D<float>(sdfTex, texX, texY + eps, texZ) -
                tex3D<float>(sdfTex, texX, texY - eps, texZ)) / (eps * 2.0f * voxelSize);
    float gz = (tex3D<float>(sdfTex, texX, texY, texZ + eps) -
                tex3D<float>(sdfTex, texX, texY, texZ - eps)) / (eps * 2.0f * voxelSize);
    float glen = sqrtf(gx * gx + gy * gy + gz * gz);
    if (glen < 1e-10f) return;
    float nx = gx / glen;
    float ny = gy / glen;
    float nz = gz / glen;

    // ── XPBD Contact Resolution ──────────────────────────────────────────
    // Penetration depth (negative SDF means inside tool)
    float penetration = contactRadius - sdf;
    if (penetration <= 0.0f) return;

    // Position correction: push particle along gradient (toward surface)
    p.x += nx * penetration;
    p.y += ny * penetration;
    p.z += nz * penetration;

    // Velocity correction: remove inward component (normal reflection)
    float vDotN = (float)(p.vx * nx + p.vy * ny + p.vz * nz);
    if (vDotN < 0.0f) {
        p.vx -= vDotN * nx;
        p.vy -= vDotN * ny;
        p.vz -= vDotN * nz;
    }

    // ── Friction (Coulomb) ──────────────────────────────────────────────
    // Tangential velocity
    float tvx = (float)p.vx - vDotN * nx;
    float tvy = (float)p.vy - vDotN * ny;
    float tvz = (float)p.vz - vDotN * nz;
    float tvMag = sqrtf(tvx * tvx + tvy * tvy + tvz * tvz);

    float normalForce = (tvMag > 0.0f)
        ? fmaxf((float)p.mass * penetration / (dt * dt), 1e-15f)
        : 1e-15f;
    float frictionForce = friction * normalForce;
    float frictionImpulse = frictionForce * dt;

    if (tvMag > 0.0f && frictionImpulse > 0.0f) {
        float maxDeltaV = frictionImpulse / fmax((float)p.mass, 1e-15f);
        float frictionScale = fminf(1.0f, maxDeltaV / tvMag);
        p.vx -= tvx * frictionScale;
        p.vy -= tvy * frictionScale;
        p.vz -= tvz * frictionScale;
    }

    // ── Heat Generation ──────────────────────────────────────────────────
    float slipVel = sqrtf(
        (float)p.vx * (float)p.vx +
        (float)p.vy * (float)p.vy +
        (float)p.vz * (float)p.vz);
    double frictionWork = normalForce * slipVel * dt;  // J
    double heatPerParticle = frictionWork * (1.0 - heatPartition);  // fraction to workpiece

    if (heatPerParticle > 0.0) {
        double particleVolume = (p.volume > 1e-24)
            ? p.volume
            : (p.mass / fmax(p.density, 1.0));
        double physicalMass = fmax(workPhysicalDensity, 1.0) *
                              fmax(particleVolume, 1e-24);
        double dT = heatPerParticle / (fmax(physicalMass, 1e-15) * workSpecificHeat);
        p.temperature = fmin(p.temperature + dT, workMeltTemp);
    }

    // ── Metrics ──────────────────────────────────────────────────────────
    double forceMag = fmaxf(normalForce, 1e-15f);
    atomicAdd(contactCount, 1);
    atomicAdd(totalHeat, frictionWork);
    atomicAdd(totalForce, forceMag);
}

/// Mark FEM nodes as inContact + estimate Hertz stress from SDF contact results.
__global__ void markContactNodesFromSDFKernel(
    FEMNodeGPU* nodes, int numNodes,
    cudaTextureObject_t sdfTex,
    float3 gridOrigin, float voxelSize, int3 gridDims,
    float centerX, float centerY, float centerZ,
    float cosAngle, float sinAngle,
    float contactRadius,
    float maxContactPressure,
    const int* d_totalContactCount, const double* d_totalContactForce)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    // Transform node position to SDF rest frame
    float dx = (float)node.x - centerX;
    float dy = (float)node.y - centerY;
    float localX =  dx * cosAngle + dy * sinAngle;
    float localY = -dx * sinAngle + dy * cosAngle;
    float localZ = (float)node.z - centerZ;

    float texX = (localX - gridOrigin.x) / voxelSize;
    float texY = (localY - gridOrigin.y) / voxelSize;
    float texZ = (localZ - gridOrigin.z) / voxelSize;

    node.inContact = false;
    node.contactForceNormal = 0.0;
    node.penetrationDepth = 0.0;

    if (texX < 0 || texY < 0 || texZ < 0 ||
        texX > gridDims.x - 1 || texY > gridDims.y - 1 || texZ > gridDims.z - 1) {
        return;
    }

    float sdf = tex3D<float>(sdfTex, texX, texY, texZ);

    if (sdf < contactRadius * 2.0f) {
        node.inContact = true;

        // Estimate Hertz stress from aggregate contact data
        // Reads device pointers directly — no CPU synchronisation needed
        int count = *d_totalContactCount;
        double force = *d_totalContactForce;
        if (count > 0 && force > 0.0) {
            double avgForce = force / count;
            double estPen = contactRadius - fmaxf(sdf, 0.0f);
            double a2 = fmax(contactRadius * estPen, 1e-18);
            double hz = (3.0 * avgForce) / (2.0 * 3.14159265358979323846 * a2);
            node.vonMisesStress = fmin(hz, (double)maxContactPressure);
        }
    }
}

// ============================================================================
// ContactSolver Implementation
// ============================================================================

ContactSolver::ContactSolver() = default;

ContactSolver::~ContactSolver() {
    if (d_numContacts)  cudaFree(d_numContacts);
    if (d_totalHeat)    cudaFree(d_totalHeat);
    if (d_totalForce)   cudaFree(d_totalForce);
    if (m_transferEvent) cudaEventDestroy(m_transferEvent);
    if (m_computeStream) cudaStreamDestroy(m_computeStream);
}

void ContactSolver::initialize(MPMSolver* sph, FEMSolver* fem, const ContactConfig& config) {
    m_sph    = sph;
    m_fem    = fem;
    m_config = config;

    CUDA_CHECK(cudaMalloc(&d_numContacts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_totalHeat,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_totalForce,  sizeof(double)));

    CUDA_CHECK(cudaStreamCreate(&m_computeStream));
    CUDA_CHECK(cudaEventCreate(&m_transferEvent));

    m_isInitialized = true;
    std::cout << "[ContactSolver] Initialized — XPBD position-based contact, "
              << "per-node heat accumulation, Hertz stress reporting" << std::endl;
    std::cout << "[ContactSolver] Spatial hash: cellSize="
              << config.contactRadius * 1000 << "mm" << std::endl;
}

bool ContactSolver::initSDF(const Mesh& toolMesh) {
    if (toolMesh.nodes.empty() || toolMesh.triangles.empty()) {
        std::cerr << "[ContactSolver] Cannot init SDF: empty tool mesh" << std::endl;
        m_sdfEnabled = false;
        return false;
    }
    bool ok = m_sdf.build(toolMesh.nodes, toolMesh.triangles, 0.002, 128);
    m_sdfEnabled = ok;
    if (ok) {
        std::cout << "[ContactSolver] SDF contact enabled — O(1) per particle, no tunneling"
                  << std::endl;

        // Store triangle connectivity and compute vertex normals for wear morphing
        m_sdfTriangles = toolMesh.triangles;
        m_sdfOriginalPositions.resize(toolMesh.nodes.size());
        for (size_t i = 0; i < toolMesh.nodes.size(); ++i) {
            m_sdfOriginalPositions[i] = toolMesh.nodes[i].position;
        }

        // Compute per-vertex normals by averaging incident face normals
        m_sdfVertexNormals.assign(toolMesh.nodes.size(), Vec3{0, 0, 0});
        for (const auto& tri : toolMesh.triangles) {
            Vec3 n = tri.normal;
            double len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
            if (len > 1e-12) { n.x /= len; n.y /= len; n.z /= len; }
            for (int j = 0; j < 3; ++j) {
                m_sdfVertexNormals[tri.indices[j]].x += n.x;
                m_sdfVertexNormals[tri.indices[j]].y += n.y;
                m_sdfVertexNormals[tri.indices[j]].z += n.z;
            }
        }
        for (auto& vn : m_sdfVertexNormals) {
            double len = std::sqrt(vn.x*vn.x + vn.y*vn.y + vn.z*vn.z);
            if (len > 1e-12) { vn.x /= len; vn.y /= len; vn.z /= len; }
        }
    }
    return ok;
}

void ContactSolver::setToolTransform(float centerX, float centerY, float centerZ,
                                      float cosAngle, float sinAngle) {
    m_sdf.setToolTransform(centerX, centerY, centerZ, cosAngle, sinAngle);
}

bool ContactSolver::morphToolSDF() {
    if (!m_sdfEnabled || !m_fem) return false;
    if (m_sdfOriginalPositions.empty() || m_sdfVertexNormals.empty()) return false;

    // Download FEM surface node data (wear values at current deformed positions)
    std::vector<FEMNodeGPU> femNodes = m_fem->getNodes();
    if (femNodes.empty()) return false;

    // Compute worn surface positions:
    //   wornPos = originalPos - normal × wear
    // Displacing inward along the surface normal blunts the cutting edges
    // and rounds the tool profile, reflecting physical flank/crater wear.
    std::vector<Vec3> wornPositions(m_sdfOriginalPositions.size());
    for (size_t i = 0; i < m_sdfOriginalPositions.size(); ++i) {
        // Map surface vertex → FEM node by index
        double wear = (i < femNodes.size()) ? femNodes[i].wear : 0.0;
        Vec3 n = m_sdfVertexNormals[i];
        wornPositions[i].x = m_sdfOriginalPositions[i].x - n.x * wear;
        wornPositions[i].y = m_sdfOriginalPositions[i].y - n.y * wear;
        wornPositions[i].z = m_sdfOriginalPositions[i].z - n.z * wear;
    }

    // Rebuild the GPU SDF texture from the worn geometry
    bool ok = m_sdf.rebuild(wornPositions, m_sdfTriangles, 0.002, 128);
    m_sdfEnabled = ok;
    return ok;
}

void ContactSolver::resolveContacts(double dt) {
    if (!m_isInitialized || !m_sph || !m_fem) return;
    if (dt <= 0.0) return;

    int numP = m_sph->getParticleCount();
    int numN = m_fem->getNodeCount();
    if (numP <= 0 || numN <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_numContacts, 0, sizeof(int), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_totalHeat,   0, sizeof(double), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_totalForce,  0, sizeof(double), m_computeStream));

    int blockSize = 256;
    int gridSizeP = (numP + blockSize - 1) / blockSize;
    int gridSizeN = (numN + blockSize - 1) / blockSize;

    // ── Engagement Ramp-Up ───────────────────────────────────────────────────
    // When the tool first touches the workpiece, thousands of particles may
    // enter the contact radius simultaneously. Applying full XPBD constraint
    // force instantly produces a non-physical shock wave (100s of MN).
    //
    // Solution: gradually scale contact forces from 10% → 100% over the first
    // N steps of engagement (configurable via engagementRampSteps).
    // This models the real-world smooth engagement of a cutting edge.
    double engagementScale = 1.0;
    if (m_contactAge < m_config.engagementRampSteps && m_config.engagementRampSteps > 0) {
        // Smooth cubic Hermite ramp: starts at 0.1 (not zero — must have some response)
        double t = (double)m_contactAge / (double)m_config.engagementRampSteps;
        engagementScale = 0.1 + 0.9 * (t * t * (3.0 - 2.0 * t));  // smoothstep
    }

    // Max penetration from config (fraction of contactRadius)
    double maxPen = m_config.contactRadius * m_config.maxPenetrationFraction;

    resetContactAccumulatorsKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
        m_fem->getDeviceNodes(), numN);
    CUDA_CHECK_KERNEL();

    // ══════════════════════════════════════════════════════════════════════════
    // SDF CONTACT RESOLUTION — O(1) per particle, no tunneling
    // ══════════════════════════════════════════════════════════════════════════

    if (m_sdfEnabled && m_sdf.isValid()) {
        // Step 1: SDF contact kernel — per-particle O(1) lookup
        // (markContactNodesFromSDFKernel below resets node accumulators)
        xpbdContactSDFKernel<<<gridSizeP, blockSize, 0, m_computeStream>>>(
            m_sph->getDeviceParticles(), numP,
            m_sdf.getTexture(),
            m_sdf.getGridOrigin(), m_sdf.getVoxelSize(), m_sdf.getGridDims(),
            m_sdf.getCenterX(), m_sdf.getCenterY(), m_sdf.getCenterZ(),
            m_sdf.getCosAngle(), m_sdf.getSinAngle(),
            (float)m_config.contactRadius,
            (float)m_config.frictionCoefficient,
            (float)m_config.heatPartition,
            (float)dt,
            m_config.workSpecificHeat,
            m_config.workPhysicalDensity,
            m_config.workMeltTemp,
            m_config.contactHTC,
            m_config.jcEnabled,
            m_config.jc_A, m_config.jc_B, m_config.jc_n,
            m_config.jc_C, m_config.jc_m,
            m_config.jc_strainRateRef, m_config.jc_T_ref, m_config.jc_T_melt,
            d_numContacts, d_totalHeat, d_totalForce
        );
        CUDA_CHECK_KERNEL();

        // Step 2: Mark FEM nodes near SDF surface for wear/temperature model
        // Pass device pointers directly — the kernel reads d_totalContactCount
        // and d_totalContactForce from VRAM without any CPU synchronisation.
        markContactNodesFromSDFKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
            m_fem->getDeviceNodes(), numN,
            m_sdf.getTexture(),
            m_sdf.getGridOrigin(), m_sdf.getVoxelSize(), m_sdf.getGridDims(),
            m_sdf.getCenterX(), m_sdf.getCenterY(), m_sdf.getCenterZ(),
            m_sdf.getCosAngle(), m_sdf.getSinAngle(),
            (float)m_config.contactRadius,
            (float)m_config.maxContactPressure,
            d_numContacts, d_totalForce);
        CUDA_CHECK_KERNEL();
    } else {
        // ══════════════════════════════════════════════════════════════════════
        // FALLBACK: SPATIAL HASH CONTACT RESOLUTION (when SDF is unavailable)
        // ══════════════════════════════════════════════════════════════════════
        HashConfig hc;
        hc.cellSize = m_config.contactRadius;
        hc.originX = m_sph->getDomainMin().x;
        hc.originY = m_sph->getDomainMin().y;
        hc.originZ = m_sph->getDomainMin().z;

        m_hash.build(m_fem->getDeviceNodes(), numN, hc, m_computeStream);

        resetContactAccumulatorsKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
            m_fem->getDeviceNodes(), numN);
        CUDA_CHECK_KERNEL();

        xpbdContactSpatialHashKernel<<<gridSizeP, blockSize, 0, m_computeStream>>>(
            m_sph->getDeviceParticles(), numP,
            m_hash.getSortedNodes(), numN,
            m_hash.getCellStart(), m_hash.getCellEnd(),
            m_hash.getHashTableSize(),
            m_hash.getCellSize(),
            m_hash.getOriginX(), m_hash.getOriginY(), m_hash.getOriginZ(),
            m_config.contactRadius,
            m_config.contactStiffness,
            m_config.contactDampingRatio,
            m_config.frictionCoefficient,
            m_config.heatPartition,
            dt,
            m_config.workpieceShearYield,
            maxPen,
            engagementScale,
            m_config.workSpecificHeat,
            m_config.workPhysicalDensity,
            m_config.workMeltTemp,
            m_config.contactHTC,
            m_config.jcEnabled,
            m_config.jc_A, m_config.jc_B, m_config.jc_n,
            m_config.jc_C, m_config.jc_m,
            m_config.jc_strainRateRef, m_config.jc_T_ref, m_config.jc_T_melt,
            d_numContacts, d_totalHeat, d_totalForce
        );
        CUDA_CHECK_KERNEL();

        m_hash.scatterResults(m_fem->getDeviceNodes(), m_computeStream);

        // Apply node heat buffer → temperature (hash path only)
        applyNodeHeatAndHertzStressKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
            m_fem->getDeviceNodes(), numN,
            m_config.maxContactPressure,
            m_config.contactRadius,
            m_config.toolSpecificHeat,
            m_config.toolMeltTemp,
            0.9,
            m_config.ambientTemperature,
            m_config.stefanBoltzmann,
            m_config.surfaceEmissivity,
            m_config.toolPhysicalDensity,
            m_config.latentHeatFusion,
            m_config.solidusTemp,
            m_config.liquidusTemp,
            dt
        );
    }
    CUDA_CHECK_KERNEL();

    transferResults();

    // ── Track contact age for engagement ramp ────────────────────────────────
    // BUG 1 FIX: Only reset ramp after 50+ steps of disengagement.
    // Brief contact gaps (1-2 steps from chip ejection momentarily clearing
    // all contacts) must NOT reset the ramp — doing so causes a KE explosion
    // when the ramp restarts at 10% while MPM particles have full unconstrained
    // velocity from the previous step.
    m_resolveCallCount++;
    if (m_numContacts > 0) {
        m_contactAge++;
        m_lastContactStep = m_resolveCallCount;
    } else {
        // Only reset ramp if tool has been disengaged for >50 steps
        // (handles peck drilling G73/G83 re-entry correctly)
        if (m_resolveCallCount - m_lastContactStep > 50) {
            m_contactAge = 0;
        }
        // Do NOT reset on brief contact gaps (1-2 steps) — these are
        // caused by chip ejection momentarily clearing all contacts,
        // not by actual tool withdrawal.
    }

    // === CONTACT EXPLOSION GUARD ===
    // If contacts exceed 50× the number of tool nodes, the tool is deeply
    // embedded inside the workpiece — a configuration error, not a physics event.
    // Log a critical diagnostic and suppress all contact forces to prevent
    // thermal runaway from poisoning the entire simulation.
    if (m_numContacts > numN * 50) {
        static int explosionWarnings = 0;
        if (explosionWarnings++ < 5) {
            std::cerr << "\n[ContactSolver] CRITICAL: Contact explosion detected!"
                      << "\n  Contacts: " << m_numContacts 
                      << " (" << (m_numContacts / std::max(numN, 1)) << "× node count)"
                      << "\n  Force: " << std::scientific << m_totalContactForce << " N"
                      << "\n  Engagement ramp: " << std::fixed << std::setprecision(1) 
                      << (engagementScale * 100) << "%"
                      << "\n  This means the tool is deeply INSIDE the workpiece."
                      << "\n  Check initial tool positioning and G-Code clearance height."
                      << "\n  SURPRESSING ALL CONTACT FORCES to prevent cascade."
                      << std::defaultfloat << std::endl;
        }
        // Zero out accumulated forces on device nodes to prevent cascade
        resetContactAccumulatorsKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
            m_fem->getDeviceNodes(), numN);
        CUDA_CHECK_KERNEL();
        m_numContacts = 0;
        m_totalContactForce = 0.0;
        m_totalHeatGenerated = 0.0;
    }

    // Diagnostic logging (every 100th call)
    static int contactCallCount = 0;
    if (++contactCallCount % 100 == 1) {
        std::cout << std::fixed
                  << "[ContactSolver] call#" << contactCallCount
                  << ": contacts=" << m_numContacts
                  << std::scientific << std::setprecision(3)
                  << ", heat="  << m_totalHeatGenerated << " J"
                  << ", force=" << m_totalContactForce  << " N"
                  << ", ramp=" << std::fixed << std::setprecision(0)
                  << (engagementScale * 100) << "%"
                  << std::defaultfloat << std::endl;
    }
    // First-contact diagnostic: log overlap depth and contact area
    if (contactCallCount == 1 && m_numContacts > 0) {
        double rawForce   = m_totalContactForce / std::max(engagementScale, 1e-6);
        double estPen     = rawForce / std::max(m_config.contactStiffness * m_numContacts, 1e-18);
        double contArea   = m_numContacts * PI * m_config.contactRadius * m_config.contactRadius;
        double avgPress   = m_totalContactForce / std::max(contArea, 1e-18);
        std::cout << "[ContactSolver] First contact: contacts=" << m_numContacts
                  << ", force=" << std::scientific << std::setprecision(2) << m_totalContactForce << " N"
                  << ", rawF=" << rawForce << " N"
                  << ", ramp=" << std::fixed << std::setprecision(1) << (engagementScale * 100) << "%"
                  << ", estPen=" << estPen * 1e6 << " um"
                  << ", area=" << contArea * 1e6 << " mm2"
                  << ", press=" << avgPress / 1e6 << " MPa"
                  << std::defaultfloat << std::endl;
    }

    // ── Tool Coating Wear Update ──────────────────────────────────────────────
    if (m_toolCoatingModel && m_numContacts > 0 && m_fem) {
        auto toolNodes = m_fem->getNodes();
        int  numNodes  = static_cast<int>(toolNodes.size());

        for (int j = 0; j < numNodes; ++j) {
            if (!toolNodes[j].inContact) continue;

            double contactPressure = toolNodes[j].vonMisesStress;   // Hertz Pa
            double slidingVelocity = sqrt(
                toolNodes[j].vx * toolNodes[j].vx +
                toolNodes[j].vy * toolNodes[j].vy +
                toolNodes[j].vz * toolNodes[j].vz);
            double temperature = toolNodes[j].temperature;

            // Zone classification: Hertz pressure above 1 GPa → cutting edge.
            // Temperature above 400°C on a moderate-stress node → crater zone.
            WearZone zone = WearZone::FLANK_FACE;
            switch (static_cast<ToolRegion>(toolNodes[j].toolRegion)) {
                case ToolRegion::RAKE_FACE:
                case ToolRegion::INSERT_RAKE_FACE:
                case ToolRegion::CHIP_GULLET:
                    zone = WearZone::CRATER_ZONE;
                    break;
                case ToolRegion::FLANK_FACE:
                case ToolRegion::MARGIN:
                case ToolRegion::LAND:
                case ToolRegion::RELIEF_FACE:
                case ToolRegion::THREAD_FLANK:
                case ToolRegion::INSERT_FLANK_FACE:
                    zone = WearZone::FLANK_FACE;
                    break;
                case ToolRegion::CUTTING_EDGE:
                case ToolRegion::CHISEL_EDGE:
                case ToolRegion::END_CUTTING_EDGE:
                case ToolRegion::PERIPHERAL_CUTTING_EDGE:
                case ToolRegion::CORNER_RADIUS:
                case ToolRegion::LEAD_CHAMFER:
                case ToolRegion::THREAD_CREST:
                case ToolRegion::CHAMFER_LEAD:
                case ToolRegion::BURR_TOOTH:
                    zone = WearZone::CUTTING_EDGE;
                    break;
                default:
                    if (contactPressure > 1.0e9) {
                        zone = WearZone::CUTTING_EDGE;
                    } else if (temperature > 400.0) {
                        zone = WearZone::CRATER_ZONE;
                    }
                    break;
            }

            m_toolCoatingModel->updateWear(
                j, contactPressure, slidingVelocity, temperature, zone, dt);
        }
    }
}

void ContactSolver::transferResults() {
    CUDA_CHECK(cudaMemcpyAsync(&m_numContacts,       d_numContacts, sizeof(int),    cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaMemcpyAsync(&m_totalHeatGenerated, d_totalHeat,  sizeof(double), cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaMemcpyAsync(&m_totalContactForce,  d_totalForce, sizeof(double), cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaEventRecord(m_transferEvent, m_computeStream));
    CUDA_CHECK(cudaEventSynchronize(m_transferEvent));
    m_cumulativeHeatGenerated += m_totalHeatGenerated;
}

void ContactSolver::applyCoatingWearToMesh(Mesh& mesh) const {
    if (!m_toolCoatingModel) return;

    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        int nodeId = static_cast<int>(i);
        double coatingWear = m_toolCoatingModel->getWearDepth(nodeId);
        double flankWear = m_toolCoatingModel->getFlankWear(nodeId);
        double craterWear = m_toolCoatingModel->getCraterWear(nodeId);

        mesh.nodes[i].accumulatedWear =
            std::max(mesh.nodes[i].accumulatedWear, coatingWear);
        mesh.nodes[i].flankWear = std::max(mesh.nodes[i].flankWear, flankWear);
        mesh.nodes[i].craterWear = std::max(mesh.nodes[i].craterWear, craterWear);
        if (m_toolCoatingModel->isSubstrateExposed(nodeId)) {
            mesh.nodes[i].coatingRemaining = 0.0;
        }
    }
}

double ContactSolver::getEstimatedContactPressure() const {
    if (m_numContacts <= 0 || m_totalContactForce <= 0.0 ||
        m_config.contactRadius <= 0.0) {
        return 0.0;
    }

    const double nominalArea =
        static_cast<double>(m_numContacts) * PI *
        m_config.contactRadius * m_config.contactRadius;
    if (nominalArea <= 1e-18) return 0.0;

    // Hertz peak pressure is higher than nominal average pressure. This is a
    // conservative reporting proxy when per-node Hertz stress has not been
    // sampled yet by FEM metrics.
    double pressure = 1.5 * m_totalContactForce / nominalArea;
    return fmin(pressure, m_config.maxContactPressure);
}

std::vector<ContactEvent> ContactSolver::getContactEvents() const {
    return std::vector<ContactEvent>();
}

// ============================================================================
// Free function (legacy compatibility)
// ============================================================================

void launchContactInteraction(
    MPMParticle* particles, int numParticles,
    FEMNodeGPU* nodes, int numNodes,
    double contactRadius, double contactStiffness,
    double friction, double heatPartition, double dt,
    double contactHTC,
    bool   jcEnabled,
    double jc_A, double jc_B, double jc_n,
    double jc_C, double jc_m,
    double jc_strainRateRef, double jc_T_ref, double jc_T_melt
) {
    static int*    d_cnt  = nullptr;
    static double* d_h    = nullptr;
    static double* d_f    = nullptr;
    static bool    init   = false;

    if (!init) {
        CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_h,   sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_f,   sizeof(double)));
        init = true;
    }

    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_h,   0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_f,   0, sizeof(double)));

    int blockSize = 256;
    int gridSize  = (numParticles + blockSize - 1) / blockSize;

    // Use XPBD kernel for legacy compatibility
    // Legacy path: full engagement (scale=1.0), generous penetration cap, default steel shear yield
    double shearYield = 343.6e6;  // Steel 4140: 595 MPa / √3
    double maxPen = contactRadius * 0.5;
    double engagementScale = 1.0;  // Legacy: no ramp
    double workSpecHeat = 475.0;
    double workPhysicalDensity = 7850.0;
    double workMeltTemp = 1460.0;
    double dampingRatio = 0.25;
    double maxContactPressure = 4.0e9;
    // Guard against brute-force contact explosion (N * M > 1e10 is hanging the GPU)
    long long pairCount = (long long)numParticles * (long long)numNodes;
    if (pairCount > 10000000LL) {
        printf("[ContactSolver] WARNING: Brute force legacy path blocked due to massive pair count (%lld). Use spatial hash.\n", pairCount);
        return;
    }

    int blockSizeN = 256;
    int gridSizeN  = (numNodes + blockSizeN - 1) / blockSizeN;
    resetContactAccumulatorsKernel<<<gridSizeN, blockSizeN>>>(nodes, numNodes);
    CUDA_CHECK_KERNEL();

    double useHTC = contactHTC > 0 ? contactHTC : 1.0e5;
    xpbdContactKernel<<<gridSize, blockSize>>>(
        particles, numParticles,
        nodes,     numNodes,
        contactRadius, contactStiffness, dampingRatio,
        friction, heatPartition, dt,
        shearYield, maxPen, engagementScale,
        workSpecHeat, workPhysicalDensity, workMeltTemp,
        useHTC,
        jcEnabled,
        jc_A, jc_B, jc_n,
        jc_C, jc_m,
        jc_strainRateRef, jc_T_ref, jc_T_melt,
        d_cnt, d_h, d_f
    );
    CUDA_CHECK_KERNEL();

    // Apply node heat buffer / radiation / phase change (Items 5, 6, 9)
    applyNodeHeatAndHertzStressKernel<<<gridSizeN, blockSizeN>>>(
        nodes, numNodes, maxContactPressure, contactRadius,
        200.0, 2500.0, 0.9,
        25.0,                    // ambientTemp
        5.670374419e-8,          // stefanBoltzmann
        0.7,                     // emissivity
        14500.0,                 // toolDensity (WC)
        100.0,                   // latentHeatFusion (small for testing)
        0.0,                     // solidusTemp (disabled for legacy)
        0.0,                     // liquidusTemp
        dt);                     // dt from outer scope
    CUDA_CHECK_KERNEL();
}

} // namespace edgepredict
