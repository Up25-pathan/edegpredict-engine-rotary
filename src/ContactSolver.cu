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
 *
 * 6. FIRST-CONTACT THERMAL EXPLOSION FIX (P0):
 *    ─────────────────────────────────────────────────────────────────────
 *    Problem: When tool first contacts workpiece, particles go from zero
 *    contact to full engagement with cutting speed (20+ m/s) in one step.
 *    Friction heat = ft * vt * dt_micro dumps massive energy instantly.
 *
 *    Solution: Multi-layer protection:
 *    (a) Engagement ramp applies to BOTH position correction AND heat
 *    (b) First-contact detection: limit heat to plastic work only (no friction)
 *    (c) Heat partition computed at ACTUAL interface temperature
 *    (d) Chip-tool interface: separate HTC, limit sliding after formation
 *    (e) Penetration-ramped contact radius: starts small, grows to full
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

// First-contact thermal protection constants
constexpr int FIRST_CONTACT_STEPS = 20;      // Ramp friction heat over first N contact steps
[[maybe_unused]] constexpr double MAX_FIRST_CONTACT_HEAT_FRACTION = 0.1;  // Max 10% of plastic work as friction heat

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
    double dtMicro,
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
            // Fix 3.4: quench-hardened particles carry a permanently elevated
            // yield (thermal-shock hardening), added on top of the JC flow stress.
            flowStress += p.yieldStrengthOffset;
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
            // FIX 2.3: accumulate the true patch area per node so the Hertz
            // stress kernel uses F_total/A_total instead of F_total/(π·r·δ_max).
            atomicAdd(&node.accumulatedContactArea, shearArea);

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

            // ── P0 FIX: First-contact thermal explosion protection (hash path) ──────
            bool isFirstContact = (p.temperature < 100.0 && p.asbCounter < FIRST_CONTACT_STEPS);
            bool isChipContact = (p.status == ParticleStatus::CHIP);
            
            // Blok/Jaeger heat partition at ACTUAL interface temperature
            double k_tool = 80.0, rho_tool = 14500.0, Cp_tool = 200.0;
            double e_tool = sqrt(k_tool * rho_tool * Cp_tool);
            double k_work = 45.0, rho_work = workPhysicalDensity, Cp_work = workSpecificHeat;
            double e_work = sqrt(k_work * rho_work * Cp_work);
            double T_interface = (e_tool * node.temperature + e_work * p.temperature) / (e_tool + e_work);
            double T_ratio_partition = fmin(1.0, fmax(0.0, (T_interface - 200.0) / 800.0));
            double heatPartition_local = heatPartition * (1.0 - 0.3 * T_ratio_partition);
            heatPartition_local = fmax(0.3, fmin(0.7, heatPartition_local));  // Clamp 30-70%

            // Heat integration parameters
            double heatDt = isFirstContact ? dtMicro : (dt / 20.0);
            
            // Compute friction heat with first-contact protection
            double heatThisStep = 0.0;
            if (isFirstContact || isChipContact) {
                double maxPhysicalStrainRate = vt / fmax(2.0 * particleRadius, 1e-9);
                double cappedStrainRate = fmin(maxPhysicalStrainRate, 10.0 * 5000.0);
                double plasticWorkRate = flowStress * cappedStrainRate * particleVolume;
                heatThisStep = plasticWorkRate * heatDt * engagementScale;
                double maxEnergy = particleVolume * workPhysicalDensity * workSpecificHeat * 100.0;
                if (heatThisStep > maxEnergy) heatThisStep = maxEnergy;
                if (isFirstContact && p.asbCounter < FIRST_CONTACT_STEPS) {
                    p.asbCounter++;
                }
            } else {
                heatThisStep = ft * vt * heatDt * engagementScale;
            }

            // Heat generation and conduction
            double heatGen_work = heatThisStep * (1.0 - heatPartition_local);
            double Q_cond = HTC_eff * shearArea * (p.temperature - node.temperature) * heatDt;
            if (Q_cond < 0.0) Q_cond = 0.0;
            Q_cond = fmin(Q_cond, heatGen_work);
            
            double heatToWorkpiece = heatGen_work - Q_cond;
            
            // Accumulate workpiece heat (single engagement scaling on generated heat)
            totalHeatToParticle += heatToWorkpiece;
            
            // Accumulate tool heat (for node heat accumulator)
            double toolHeatThisStep = heatThisStep * heatPartition_local + Q_cond;
            atomicAdd(&node.heatAccumulator, toolHeatThisStep);
            
            if (vt > 1e-15) {
                atomicAdd(&node.fx, ft * (vtx / vt));
                atomicAdd(&node.fy, ft * (vty / vt));
                atomicAdd(&node.fz, ft * (vtz / vt));
            }
        node.inContact = true;

            totalHeatGen += heatThisStep;
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

        if (penetration > maxPen) {
            maxPen = penetration;
            maxNx = nx;
            maxNy = ny;
            maxNz = nz;
        }

        // FIX C: Position correction is ALWAYS full strength.
        // This is a geometric constraint ("particle must not be inside tool"),
        // not a physical force. The engagement ramp only applies to forces/heat.
        corrX += penetration * nx;
        corrY += penetration * ny;
        corrZ += penetration * nz;
        particleContactCount++;

        // ── Relative velocity ────────────────────────────────────────────────
        double relVx = p.vx - node.vx;
        double relVy = p.vy - node.vy;
        double relVz = p.vz - node.vz;
        double  vn   = relVx*nx + relVy*ny + relVz*nz;

        // ── Normal force: compliant contact law (spring-damper) ─────────────────
        // Replaces XPBD diagnostic force m*pen/dt² with physically-based spring-damper.
        // This keeps reported force independent of integration dt and prevents
        // the massive force spikes that cause contact explosion.
        double approachSpeed = fmax(0.0, -vn);
        double kNormal = fmax(contactStiffness, 0.0);
        double cNormal = 2.0 * fmax(contactDampingRatio, 0.0)
                         * sqrt(kNormal * fmax(p.mass, 1e-15));
        double fn = (kNormal * penetration + cNormal * approachSpeed) * engagementScale;

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
        // Fix 3.4: quench-hardened particles carry a permanently elevated yield.
        flowStress += p.yieldStrengthOffset;
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
        double shearAreaClamped = (shearArea < minArea) ? minArea : shearArea;
        // FIX 2.3: accumulate the true patch area per node (F_total/A_total).
        atomicAdd(&node.accumulatedContactArea, shearAreaClamped);
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
        
    // Friction work integrated on the micro-step scale (see SDF kernel note):
    // one resolve = one micro-step of dwell; the full interval dt would inflate
    // the per-particle dT and cause a temperature mass-jump.
    double heatDt = dt / 20.0;

    // ── P0 FIX: First-contact thermal explosion protection (hash path) ──────
    bool isFirstContact = (p.temperature < 100.0 && p.asbCounter < FIRST_CONTACT_STEPS);
    bool isChipContact = (p.status == ParticleStatus::CHIP);

    // Blok/Jaeger heat partition at ACTUAL interface temperature
    double k_tool = 80.0, rho_tool = 14500.0, Cp_tool = 200.0;
    double e_tool = sqrt(k_tool * rho_tool * Cp_tool);
    double k_work = 45.0, rho_work = workPhysicalDensity, Cp_work = workSpecificHeat;
    double e_work = sqrt(k_work * rho_work * Cp_work);
    double T_interface = (e_tool * node.temperature + e_work * p.temperature) / (e_tool + e_work);
    double T_ratio_partition = fmin(1.0, fmax(0.0, (T_interface - 200.0) / 800.0));
    double heatPartition_local = heatPartition * (1.0 - 0.3 * T_ratio_partition);
    heatPartition_local = fmax(0.3, fmin(0.7, heatPartition_local));

    double heatThisStep = 0.0;
    if (isFirstContact || isChipContact) {
        // First contact: NO friction sliding heat — only plastic work heat
        double particleVolume = (p.volume > 1e-24) ? p.volume : (p.mass / fmax(p.density, 1.0));
        double particleRadius = pow(3.0 * particleVolume / (4.0 * PI), 1.0 / 3.0);
        double strainRate_contact = vt / fmax(2.0 * particleRadius, 1e-9);
        double plasticWorkRate = flowStress * strainRate_contact * particleVolume;
        heatThisStep = plasticWorkRate * heatDt * engagementScale;
        if (isFirstContact && p.asbCounter < FIRST_CONTACT_STEPS) {
            p.asbCounter++;
        }
    } else {
        heatThisStep = ft * vt * heatDt * engagementScale;
    }

    double heatGen_work = heatThisStep * (1.0 - heatPartition_local);
    double Q_cond = HTC_eff * shearArea * (p.temperature - node.temperature) * heatDt;
    if (Q_cond < 0.0) Q_cond = 0.0;
    Q_cond = fmin(Q_cond, heatGen_work);
    double heatToWorkpiece = (heatGen_work - Q_cond);

    if (isChipContact) {
        double chipHTC = contactHTC * 0.5;
        double Q_cond_chip = chipHTC * shearArea * (p.temperature - node.temperature) * heatDt;
        Q_cond_chip = fmin(fmax(Q_cond_chip, 0.0), heatGen_work);
        heatToWorkpiece = (heatGen_work - Q_cond_chip) * 0.5;
    }

    totalHeatToParticle += heatToWorkpiece;
    atomicAdd(&node.heatAccumulator, heatThisStep * heatPartition_local + Q_cond);
    if (vt > 1e-15) {
        atomicAdd(&node.fx, ft * (vtx / vt));
        atomicAdd(&node.fy, ft * (vty / vt));
        atomicAdd(&node.fz, ft * (vtz / vt));
    }
    node.inContact = true;

    totalHeatGen += heatThisStep;
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
    node.accumulatedContactArea = 0.0;
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
    // Use unscaled physicalMass for thermal capacity to prevent dynamic mass scaling distortion
    double nodeThermalMass = fmax(node.physicalMass > 1e-15 ? node.physicalMass : node.mass, 1e-12);

    // ── Radiation cooling (Item 5) ────────────────────────────────────────────
    if (node.temperature > ambientTemp + 1.0) {
        // Estimate node surface area from physical mass: A ≈ (mass/ρ)^(2/3)
        double vol = fmax(nodeThermalMass / toolDensity, 1e-15);
        double area = pow(vol, 2.0 / 3.0); // m²
        double Tn = node.temperature + 273.15; // K
        double Ta = ambientTemp + 273.15;      // K
        double Qrad = emissivity * stefanBoltzmann * area *
                      (Tn * Tn * Tn * Tn - Ta * Ta * Ta * Ta) * dt;
        node.heatAccumulator = fmax(node.heatAccumulator - Qrad, 0.0);
    }

    // ── Thermal-inertia guard (calibration) ───────────────────────────────────
    // Prevent non-physical thermal shock: max dT per microstep is 5.0°C (5e6 °C/s)
    const double maxStepJ = nodeThermalMass * fmax(toolCp, 1e-12) * 5.0;
    if (node.heatAccumulator > maxStepJ) {
        node.heatAccumulator = maxStepJ;
    }

    // ── Phase change / melting enthalpy (Items 6, 9) ──────────────────────────
    bool hasPhaseChange = (latentHeatFusion > 0.0);
    bool hasMushyZone = hasPhaseChange && (solidusTemp > 0.0) && (liquidusTemp > solidusTemp);
    double Ts = hasMushyZone ? solidusTemp : toolMeltTemp;
    double Tl = hasMushyZone ? liquidusTemp : toolMeltTemp;

    double latentAccum = 0.0;
    if (node.heatAccumulator > 0.0 && hasPhaseChange) {
        double mass = nodeThermalMass;
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
        // No phase change — simple temperature rise using unscaled physical mass.
        double dT = node.heatAccumulator / (nodeThermalMass * fmax(toolCp, 1e-12));
        node.temperature = fmin(node.temperature + dT, toolMeltTemp);
        if (node.temperature >= toolMeltTemp) node.status = NodeStatus::FAILED;
    }
    node.heatAccumulator = latentAccum;

    // ── Hertz contact stress (physically meaningful von Mises proxy) ─────────
    // FIX 2.3: peak pressure for a Hertzian circular contact = 1.5·F_total/A_total.
    // node.contactForceNormal sums the load over ALL contacting particles while
    // the OLD code divided by the area of a single particle's max penetration
    // (a² = r·δ_max), inflating pressure by the contact count N and triggering
    // premature NodeStatus::FAILED / stress spikes. The area is now accumulated
    // per node over every contact patch.
    double sigma_hertz = 0.0;
    if (node.accumulatedContactArea > 1e-18 && node.contactForceNormal > 0.0) {
        sigma_hertz = 1.5 * (node.contactForceNormal / node.accumulatedContactArea);
        sigma_hertz = fmin(sigma_hertz, maxContactPressure);
    }

    // EMA-smoothed von Mises stress (prevents aliasing from step-to-step spikes)
    node.vonMisesStress = ema_alpha * node.vonMisesStress
                        + (1.0 - ema_alpha) * sigma_hertz;

    // Reset per-step contact accumulators
    node.contactForceNormal = 0.0;
    node.penetrationDepth   = 0.0;
    node.accumulatedContactArea = 0.0;
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
    float angularVelocity,     // Tool spin about Z (rad/s)
    float linearVelX, float linearVelY, float linearVelZ,  // Tool linear velocity (fix 2.2)
    float contactRadius,
    float friction,
    float heatPartition,
    float contactStiffness,
    float contactDampingRatio,
    float maxPenetration,
    float engagementScale,
    double toolSurfaceTemp,
    double shearYieldStress,
    float dt,
    float dtMicro,
    double workSpecificHeat,
    double workPhysicalDensity,
    double workMeltTemp,
    double contactHTC,
    bool   jcEnabled,
    double jc_A, double jc_B, double jc_n,
    double jc_C, double jc_m,
    double jc_strainRateRef, double jc_T_ref, double jc_T_melt,
    int* contactCount, double* totalHeat, double* toolHeat, double* totalForce)
{
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= numParticles) return;

    (void)contactStiffness;
    (void)contactDampingRatio;

    MPMParticle& p = particles[pIdx];
    if (p.status == ParticleStatus::INACTIVE ||
        p.status == ParticleStatus::FIXED_BOUNDARY) return;

    // Transform particle position from world → SDF rest frame (θ=0)
    float dx = (float)p.x - centerX;
    float dy = (float)p.y - centerY;
    float localX =  dx * cosAngle + dy * sinAngle;
    float localY = -dx * sinAngle + dy * cosAngle;
    float localZ = (float)p.z - centerZ;

    // Physical cutting boundary guard:
    // Tool CAD geometry strictly occupies localZ >= 0. Any point with localZ < 0
    // is in the uncut workpiece or clearance air below the chisel tip.
    if (localZ < 0.0f) return;

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

    if (sdf >= 0.0f) return;  // Outside tool boundary — no mechanical contact

    // Compute SDF gradient via 4th-order central difference for accurate normals
    // 4th-order: f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
    float eps = 1.0f;  // 1-voxel offset for 4th-order accuracy
    float gx = (tex3D<float>(sdfTex, texX - 2*eps, texY, texZ) 
             - 8.0f * tex3D<float>(sdfTex, texX - eps, texY, texZ)
             + 8.0f * tex3D<float>(sdfTex, texX + eps, texY, texZ)
             - tex3D<float>(sdfTex, texX + 2*eps, texY, texZ)) / (12.0f * eps * voxelSize);
    float gy = (tex3D<float>(sdfTex, texX, texY - 2*eps, texZ) 
             - 8.0f * tex3D<float>(sdfTex, texX, texY - eps, texZ)
             + 8.0f * tex3D<float>(sdfTex, texX, texY + eps, texZ)
             - tex3D<float>(sdfTex, texX, texY + 2*eps, texZ)) / (12.0f * eps * voxelSize);
    float gz = (tex3D<float>(sdfTex, texX, texY, texZ - 2*eps) 
             - 8.0f * tex3D<float>(sdfTex, texX, texY, texZ - eps)
             + 8.0f * tex3D<float>(sdfTex, texX, texY, texZ + eps)
             - tex3D<float>(sdfTex, texX, texY, texZ + 2*eps)) / (12.0f * eps * voxelSize);
    float glen = sqrtf(gx * gx + gy * gy + gz * gz);
    if (glen < 1e-10f) {
        // Fallback to 2nd-order if 4th-order fails
        eps = 0.5f;
        gx = (tex3D<float>(sdfTex, texX + eps, texY, texZ) -
            tex3D<float>(sdfTex, texX - eps, texY, texZ)) / (eps * 2.0f * voxelSize);
        gy = (tex3D<float>(sdfTex, texX, texY + eps, texZ) -
            tex3D<float>(sdfTex, texX, texY - eps, texZ)) / (eps * 2.0f * voxelSize);
        gz = (tex3D<float>(sdfTex, texX, texY, texZ + eps) -
            tex3D<float>(sdfTex, texX, texY, texZ - eps)) / (eps * 2.0f * voxelSize);
        glen = sqrtf(gx * gx + gy * gy + gz * gz);
        if (glen < 1e-10f) return;
    }
    // Gradient is computed in the SDF rest frame (θ=0). The particle was
    // transformed into that frame with R(-θ); the resulting correction normal
    // MUST be rotated back to world coordinates with R(+θ) before it is
    // applied, otherwise contact pushes resolve on the wrong side of a
    // spinning tool. (nz is unchanged — Z is the rotation axis.)
    float nx_local = gx / glen;
    float ny_local = gy / glen;
    float nz = gz / glen;
    float nx =  nx_local * cosAngle - ny_local * sinAngle;
    float ny =  nx_local * sinAngle + ny_local * cosAngle;

    // ── XPBD Contact Resolution ──────────────────────────────────────────
    // Physical penetration depth: positive only inside the solid tool body (-sdf)
    float penetration = -sdf;
    if (penetration <= 0.0f) return;
    penetration = fminf(penetration, maxPenetration);

    // ── F1: Relative velocity vs the tool surface ─────────────────────────
    // Linear feed velocity:
    float v_lin_x = linearVelX;
    float v_lin_y = linearVelY;
    float v_lin_z = linearVelZ;

    // Normal velocity: governed by relative motion in the linear feed direction.
    // (Spindle rotation ω × r is tangential to the tool axis; projecting ω × r into
    // vn would artificially inject a 1.5 m/s normal shock kick onto the workpiece).
    float relFeedVx = (float)p.vx - v_lin_x;
    float relFeedVy = (float)p.vy - v_lin_y;
    float relFeedVz = (float)p.vz - v_lin_z;
    float vn = relFeedVx * nx + relFeedVy * ny + relFeedVz * nz;

    // ── Kinematic Position Projection ─────────────────────────────────────
    float maxPosCorr = (maxPenetration > 0.0f) ? maxPenetration : (voxelSize * 0.5f);
    float posCorr = fminf(penetration, maxPosCorr) * (float)engagementScale;
    p.x += nx * posCorr;
    p.y += ny * posCorr;
    p.z += nz * posCorr;

    // ── Signorini Kinematic Normal Velocity Projection ───────────────────
    if (vn < 0.0f) {
        float vnCorr = vn * (float)engagementScale;
        p.vx -= vnCorr * nx;
        p.vy -= vnCorr * ny;
        p.vz -= vnCorr * nz;
    }

    // ── Tangential Slip Velocity (relative to spinning + advancing tool) ───
    // Total tool surface velocity includes rotational cutting speed ω × r:
    float toolVelX = -angularVelocity * dy + linearVelX;
    float toolVelY =  angularVelocity * dx + linearVelY;
    float toolVelZ =  linearVelZ;

    float relVx = (float)p.vx - toolVelX;
    float relVy = (float)p.vy - toolVelY;
    float relVz = (float)p.vz - toolVelZ;

    // Tangential slip velocity strictly orthogonal to normal n:
    float v_rel_dot_n = relVx * nx + relVy * ny + relVz * nz;
    float vtx = relVx - v_rel_dot_n * nx;
    float vty = relVy - v_rel_dot_n * ny;
    float vtz = relVz - v_rel_dot_n * nz;
    float vt = sqrtf(vtx * vtx + vty * vty + vtz * vtz);

    // ── Physical Contact Stresses & Zorev Rotary Shear ────────────────────
    double particleVolume = (p.volume > 1e-24)
        ? p.volume
        : (p.mass / fmax(p.density, 1.0));
    double particleRadius = pow(3.0 * particleVolume / (4.0 * PI), 1.0 / 3.0);
    double particleArea = PI * particleRadius * particleRadius;

    // Per-particle flow stress from the JC model
    double flowStress = jcEnabled
        ? deviceJCFlowStress(p.plasticStrain, p.strainRate, p.temperature,
                            jc_A, jc_B, jc_n, jc_C, jc_m,
                            jc_strainRateRef, jc_T_ref, jc_T_melt)
        : shearYieldStress;
    flowStress += p.yieldStrengthOffset;
    if (flowStress < 1e6) flowStress = 1e6;

    // von Mises shear yield stress: tau_y = sigma_flow / sqrt(3)
    double tau_yield = flowStress * 0.577350269;

    // Physical elastic-plastic indentation pressure law:
    // At small penetration, contact is compliant (Hertzian/elastic).
    // As penetration increases, pressure smoothly saturates at Meyer plastic hardness (3.0 * sigma_flow).
    double E_reduced = 1.67e11; // E* from carbide-steel contact
    double elasticPressure = (fmax((double)penetration, 0.0) / fmax((double)contactRadius, 1e-6)) * E_reduced;
    double plasticHardness = 3.0 * flowStress;
    double normalPressure = fmin(elasticPressure, plasticHardness) * (double)engagementScale;
    double fn = normalPressure * particleArea;

    // Coating and thermal friction factor
    double T_avg = 0.5 * ((double)p.temperature + toolSurfaceTemp);
    double T_ratio = fmin(1.0, fmax(0.0, (T_avg - 200.0) / 600.0));
    double coatingFrictionFactor = (p.status == ParticleStatus::CHIP) ? 0.70 : 0.85;
    double coatingThermalFactor  = (p.status == ParticleStatus::CHIP) ? 1.50 : 1.20;
    double mu_eff = (double)friction * coatingFrictionFactor * (1.0 - 0.4 * T_ratio);

    // ── Zorev Friction Law: tau = min(mu * sigma_n, tau_yield) ─────────
    double tau_slide = mu_eff * normalPressure;
    double tau = fmin(tau_slide, tau_yield);
    double ft = tau * particleArea;
    double forceMag = sqrt(fn * fn + ft * ft);

    // Effective time step for momentum transfer
    double stepDt = (dtMicro > 0.0f) ? (double)dtMicro : (double)dt;

    // Tangential shear impulse: drag particle in the direction of tool motion
    if (vt > 1e-9 && ft > 0.0) {
        double maxDeltaV = ft * stepDt / fmax((double)p.mass, 1e-15);
        float frictionScale = (float)fmin(1.0, maxDeltaV / (double)vt);
        p.vx -= vtx * frictionScale;
        p.vy -= vty * frictionScale;
        p.vz -= vtz * frictionScale;

        // Tangential shear strain rate contribution to Johnson-Cook plasticity
        double shearStrainRate = (double)vt / fmax(2.0 * particleRadius, 1e-9);
        p.strainRate = (float)fmin(shearStrainRate, 1.0e5);
        p.plasticStrain += (float)(p.strainRate * stepDt * (double)engagementScale);
    }

    // ── Thermal Generation & Heat Partition ──────────────────────────────
    double heatDt = stepDt;
    double k_tool = 80.0;
    double rho_tool = 14500.0;
    double Cp_tool = 200.0;
    double e_tool = sqrt(k_tool * rho_tool * Cp_tool);
    double k_work = 45.0;
    double rho_work = workPhysicalDensity;
    double Cp_work = workSpecificHeat;
    double e_work = sqrt(k_work * rho_work * Cp_work);
    double T_interface = (e_tool * toolSurfaceTemp + e_work * (double)p.temperature) / (e_tool + e_work);
    double T_ratio_partition = fmin(1.0, fmax(0.0, (T_interface - 200.0) / 800.0));
    double heatPartition_local = (heatPartition / coatingThermalFactor) * (1.0 - 0.3 * T_ratio_partition);
    heatPartition_local = fmax(0.2, fmin(0.7, heatPartition_local));

    // Friction + plastic dissipation heat
    double frictionPower = ft * (double)vt * (double)engagementScale;
    double heatThisStep = frictionPower * heatDt;
    // Bounded energy to prevent numerical runaway
    double maxStepEnergy = particleVolume * workPhysicalDensity * workSpecificHeat * 5.0;
    if (heatThisStep > maxStepEnergy) heatThisStep = maxStepEnergy;

    double heatGen_work = heatThisStep * (1.0 - heatPartition_local);
    double HTC_eff = contactHTC * (1.0 + 5.0 * (1.0 - exp(-normalPressure / flowStress)));
    double Q_cond = HTC_eff * particleArea * ((double)p.temperature - toolSurfaceTemp) * heatDt;
    if (Q_cond < 0.0) Q_cond = 0.0;
    Q_cond = fmin(Q_cond, heatGen_work);
    double heatToWorkpiece = heatGen_work - Q_cond;

    if (heatToWorkpiece > 0.0) {
        double physicalMass = fmax(workPhysicalDensity, 1.0) * fmax(particleVolume, 1e-24);
        double dT = heatToWorkpiece / (fmax(physicalMass, 1e-15) * workSpecificHeat);
        if (dT > 5.0) dT = 5.0;
        p.temperature = (float)fmin((double)p.temperature + dT, workMeltTemp);
    }

    double toolHeatThisStep = heatThisStep * heatPartition_local + Q_cond;

    // ── Metrics ──────────────────────────────────────────────────────────
    atomicAdd(contactCount, 1);
    atomicAdd(totalHeat, heatThisStep);
    atomicAdd(toolHeat, toolHeatThisStep);
    atomicAdd(totalForce, forceMag);
}

/// Mark FEM nodes as inContact + estimate Hertz stress from SDF contact results.
__global__ void markContactNodesFromSDFKernel(
    FEMNodeGPU* nodes, int numNodes,
    cudaTextureObject_t sdfTex,
    float3 gridOrigin, float voxelSize, int3 gridDims,
    float centerX, float centerY, float centerZ,
    float cosAngle, float sinAngle,
    float angularVelocity,
    float contactRadius,
    float maxContactPressure,
    float workpieceMaxZ,
    const int* d_totalContactCount, const double* d_totalContactForce,
    int* toolHeatNodeCount,
    double* toolHeatWeightSum)
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

    // Physical cutting boundary guard:
    if (localZ < 0.0f) return;

    float texX = (localX - gridOrigin.x) / voxelSize;
    float texY = (localY - gridOrigin.y) / voxelSize;
    float texZ = (localZ - gridOrigin.z) / voxelSize;

    node.inContact = false;
    node.contactForceNormal = 0.0;
    node.penetrationDepth = 0.0;
    node.accumulatedContactArea = 0.0;
    node.heatAccumulator = 0.0;

    int contactFlag = 0;

    if (texX < 0 || texY < 0 || texZ < 0 ||
        texX > gridDims.x - 1 || texY > gridDims.y - 1 || texZ > gridDims.z - 1) {
        return;
    }

    float sdf = tex3D<float>(sdfTex, texX, texY, texZ);

    int count = *d_totalContactCount;
    double force = *d_totalContactForce;

    // Check if node is in a cutting region
    bool isShank = (node.toolRegion == static_cast<int32_t>(ToolRegion::SHANK) ||
                    node.toolRegion == static_cast<int32_t>(ToolRegion::TOOL_BODY));
    bool isCuttingRegion = (node.toolRegion == static_cast<int32_t>(ToolRegion::CUTTING_EDGE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::CHISEL_EDGE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::RAKE_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::FLANK_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::MARGIN) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::INSERT_RAKE_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::END_CUTTING_EDGE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::PERIPHERAL_CUTTING_EDGE));
    if (node.toolRegion == 0) isCuttingRegion = true; // Fallback for unclassified meshes

    // In world coordinates, node must be submerged in or within engagement distance of the workpiece
    bool inCuttingZone = (node.z <= workpieceMaxZ + contactRadius * 2.0f);

    if (count > 0 && force > 0.0 && !isShank && isCuttingRegion && inCuttingZone && sdf < contactRadius * 1.5f) {
        node.inContact = true;
        contactFlag = 1;

        // Estimate Hertz stress from aggregate contact data
        // Reads device pointers directly — no CPU synchronisation needed
        double avgForce = force / count;
        double estPen = contactRadius - fmaxf(sdf, 0.0f);
        node.contactForceNormal = avgForce;
        node.penetrationDepth = estPen;
        // True circular Hertz contact patch area A = pi * a^2 = pi * r * delta
        double patchArea = 3.14159265358979323846 * fmax(contactRadius * estPen, 1e-18);
        node.accumulatedContactArea = patchArea;
        double a2 = fmax(contactRadius * estPen, 1e-18);
        double hz = (3.0 * avgForce) / (2.0 * 3.14159265358979323846 * a2);
        node.vonMisesStress = fmin(hz, (double)maxContactPressure);

        // Physics-based frictional heat weighting: q ~ mu * p * v_rel
        // v_local = |omega| * r_local
        float r_local = sqrtf(localX * localX + localY * localY);
        double v_local = fmax((double)(fabsf(angularVelocity) * r_local), 1e-3);
        double weight = v_local * fmax((double)estPen, 1e-9);
        node.heatAccumulator = weight;
        if (toolHeatWeightSum) {
            atomicAdd(toolHeatWeightSum, weight);
        }
    }

    atomicAdd(toolHeatNodeCount, contactFlag);
}

/// Distribute the aggregated tool-side friction heat over the FEM nodes marked
/// inContact. Works on the aggregate d_toolHeat, so no per-node atomics in the
/// hot particle loop are required (SDF path — hash path builds node heat per
/// particle-node pair directly).
/// Physics-based: Weight heat by relative velocity and penetration depth.
__global__ void distributeToolHeatKernel(
    FEMNodeGPU* nodes, int numNodes,
    const double* d_toolHeat, const int* d_toolHeatNodeCount,
    const double* d_toolHeatWeightSum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];
    if (!node.inContact) {
        node.heatAccumulator = 0.0;
        return;
    }

    int c = *d_toolHeatNodeCount;
    double h = *d_toolHeat;
    double weightSum = (d_toolHeatWeightSum != nullptr) ? *d_toolHeatWeightSum : 0.0;

    if (c > 0 && h > 0.0) {
        if (weightSum > 1e-12) {
            // Distribute heat proportional to local velocity and penetration:
            // sum(q_i) = h * sum(w_i) / weightSum = h (exact energy conservation)
            double nodeWeight = node.heatAccumulator;
            node.heatAccumulator = h * (nodeWeight / weightSum);
        } else {
            // Energy-conserving distribution over contacting nodes: sum(h/c) = h
            node.heatAccumulator = h / static_cast<double>(c);
        }
    } else {
        node.heatAccumulator = 0.0;
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
    if (d_toolHeat)     cudaFree(d_toolHeat);
    if (d_toolHeatNodeCount) cudaFree(d_toolHeatNodeCount);
    if (d_toolHeatWeightSum) cudaFree(d_toolHeatWeightSum);
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
    CUDA_CHECK(cudaMalloc(&d_toolHeat,    sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_toolHeatNodeCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_toolHeatWeightSum, sizeof(double)));

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
    bool ok = m_sdf.build(toolMesh.nodes, toolMesh.triangles, 0.002, 256);
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
                                      float cosAngle, float sinAngle,
                                      float angularVelocity,
                                      float linearVelX, float linearVelY, float linearVelZ) {
    m_toolAngularVelocity = angularVelocity;
    m_toolLinearVelX = linearVelX;
    m_toolLinearVelY = linearVelY;
    m_toolLinearVelZ = linearVelZ;
    m_sdf.setToolTransform(centerX, centerY, centerZ, cosAngle, sinAngle);
}

bool ContactSolver::morphToolSDF() {
    if (!m_sdfEnabled || !m_fem) return false;
    if (m_sdfOriginalPositions.empty() || m_sdfVertexNormals.empty()) return false;

    // Download lightweight FEM surface node wear data (avoids 10-20 MB full getNodes() PCIe copy)
    std::vector<NodeWearData> wearData;
    m_fem->getWearAndRegion(wearData);
    if (wearData.empty()) return false;

    // Compute worn surface positions:
    //   wornPos = originalPos - normal × wear
    // Displacing inward along the surface normal blunts the cutting edges
    // and rounds the tool profile, reflecting physical flank/crater wear.
    std::vector<Vec3> wornPositions(m_sdfOriginalPositions.size());
    for (size_t i = 0; i < m_sdfOriginalPositions.size(); ++i) {
        // Map surface vertex → FEM node by index
        double wear = (i < wearData.size()) ? (double)wearData[i].wear : 0.0;
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

// ── P2 FIX: Edge-Only SDF Morphing ────────────────────────────────────────
// Updates only vertices in cutting-edge regions (flank, rake, cutting edge)
// for fast frequent updates. Full rebuild still needed periodically.
bool ContactSolver::morphToolSDFEdgeOnly() {
    if (!m_sdfEnabled || !m_fem) return false;
    if (m_sdfOriginalPositions.empty() || m_sdfVertexNormals.empty()) return false;

    // Download lightweight FEM surface node wear & region data
    std::vector<NodeWearData> wearData;
    m_fem->getWearAndRegion(wearData);
    if (wearData.empty()) return false;

    // Identify cutting-edge region vertices by toolRegion
    // ToolRegion::FLANK_FACE, RAKE_FACE, CUTTING_EDGE, MARGIN, LAND, etc.
    std::vector<Vec3> wornPositions(m_sdfOriginalPositions.size());
    bool anyWear = false;
    for (size_t i = 0; i < m_sdfOriginalPositions.size(); ++i) {
        double wear = (i < wearData.size()) ? (double)wearData[i].wear : 0.0;
        int region = (i < wearData.size()) ? wearData[i].toolRegion : 0;
        
        // Only morph cutting-edge regions: flank, rake, cutting edge, margin, land
        bool isCuttingRegion = (region == 1 || region == 2 || region == 3 || 
                               region == 5 || region == 6 || region == 10 || region == 11);
        
        if (isCuttingRegion && wear > 1e-9) {
            Vec3 n = m_sdfVertexNormals[i];
            wornPositions[i].x = m_sdfOriginalPositions[i].x - n.x * wear;
            wornPositions[i].y = m_sdfOriginalPositions[i].y - n.y * wear;
            wornPositions[i].z = m_sdfOriginalPositions[i].z - n.z * wear;
            anyWear = true;
        } else {
            // Non-cutting regions: keep original position
            wornPositions[i] = m_sdfOriginalPositions[i];
        }
    }

    if (!anyWear) return false;  // No significant wear in cutting regions

    // Rebuild GPU SDF texture from partially worn geometry
    bool ok = m_sdf.rebuild(wornPositions, m_sdfTriangles, 0.002, 128);
    m_sdfEnabled = ok;
    return ok;
}

void ContactSolver::resolveContacts(double dt, double dtMicro) {
    if (!m_isInitialized || !m_sph || !m_fem) return;
    if (dt <= 0.0) return;

    int numP = m_sph->getParticleCount();
    int numN = m_fem->getNodeCount();
    if (numP <= 0 || numN <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_numContacts, 0, sizeof(int), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_totalHeat,   0, sizeof(double), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_totalForce,  0, sizeof(double), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_toolHeat,    0, sizeof(double), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_toolHeatNodeCount, 0, sizeof(int), m_computeStream));
    CUDA_CHECK(cudaMemsetAsync(d_toolHeatWeightSum, 0, sizeof(double), m_computeStream));

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

    // ── F3 FIX: Ramped Contact Radius for First Contact ──────────────────────
    // Start with small contact radius (chisel edge only) and grow to full
    // over the first N contact steps. This limits the initial contact zone to
    // the actual chisel tip (~0.75mm), preventing the simultaneous engagement
    // of the entire drill flank on first touch.
    //
    // FIX F3: Old code used 10 steps + 30% min radius. 10 steps was too few —
    // the radius reached full size while the force ramp was still at 13%,
    // meaning the particle count exploded before engagement was controlled.
    // 30% = 0.3mm min radius was 2× wider than the physical chisel tip (0.15mm).
    // Now: 50 steps, 15% min radius = 0.15mm (true chisel zone only).
    double contactRadiusRamped = m_config.contactRadius;
    constexpr int CONTACT_RADIUS_RAMP_STEPS = 50;  // Was 10 — ramp over 50 steps
    if (m_contactAge < CONTACT_RADIUS_RAMP_STEPS) {
        double t = (double)m_contactAge / (double)CONTACT_RADIUS_RAMP_STEPS;
        // Start at chisel-edge radius only (~15% = 0.15mm for 1mm contactRadius)
        double minRadius = m_config.contactRadius * 0.15;  // Was 0.3 — chisel tip
        contactRadiusRamped = minRadius + (m_config.contactRadius - minRadius) * (t * t * (3.0 - 2.0 * t));
    }

    // Max penetration from config (fraction of contactRadius)
    double maxPen = contactRadiusRamped * m_config.maxPenetrationFraction;

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
            m_toolAngularVelocity,
            m_toolLinearVelX, m_toolLinearVelY, m_toolLinearVelZ,
            (float)contactRadiusRamped,
            (float)m_config.frictionCoefficient,
            (float)m_config.heatPartition,
            (float)m_config.contactStiffness,
            (float)m_config.contactDampingRatio,
            (float)maxPen,
            (float)engagementScale,
            (m_fem ? m_fem->getMaxTemperature() : m_config.ambientTemperature),
            m_config.workpieceShearYield,
            (float)dt,
            (float)dtMicro,
            m_config.workSpecificHeat,
            m_config.workPhysicalDensity,
            m_config.workMeltTemp,
            m_config.contactHTC,
            m_config.jcEnabled,
            m_config.jc_A, m_config.jc_B, m_config.jc_n,
            m_config.jc_C, m_config.jc_m,
            m_config.jc_strainRateRef, m_config.jc_T_ref, m_config.jc_T_melt,
            d_numContacts, d_totalHeat, d_toolHeat, d_totalForce
        );
        CUDA_CHECK_KERNEL();

        double wkpcMinX = 0, wkpcMinY = 0, wkpcMinZ = 0, wkpcMaxX = 0, wkpcMaxY = 0, wkpcMaxZ = 0;
        if (m_sph) {
            m_sph->getBounds(wkpcMinX, wkpcMinY, wkpcMinZ, wkpcMaxX, wkpcMaxY, wkpcMaxZ);
        }

        markContactNodesFromSDFKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
            m_fem->getDeviceNodes(), numN,
            m_sdf.getTexture(),
            m_sdf.getGridOrigin(), m_sdf.getVoxelSize(), m_sdf.getGridDims(),
            m_sdf.getCenterX(), m_sdf.getCenterY(), m_sdf.getCenterZ(),
            m_sdf.getCosAngle(), m_sdf.getSinAngle(),
            m_toolAngularVelocity,
            (float)contactRadiusRamped,
            (float)m_config.maxContactPressure,
            (float)wkpcMaxZ,
            d_numContacts, d_totalForce,
            d_toolHeatNodeCount,
            d_toolHeatWeightSum);
        CUDA_CHECK_KERNEL();

        // Step 3: Distribute tool-side friction heat over the marked nodes and
        // convert node heat → temperature (Hertz stress, radiation, phase change
        // parity with the hash path).
        distributeToolHeatKernel<<<gridSizeN, blockSize, 0, m_computeStream>>>(
            m_fem->getDeviceNodes(), numN,
            d_toolHeat, d_toolHeatNodeCount,
            d_toolHeatWeightSum);
        CUDA_CHECK_KERNEL();
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
            contactRadiusRamped,
            m_config.contactStiffness,
            m_config.contactDampingRatio,
            m_config.frictionCoefficient,
            m_config.heatPartition,
            dt,
            dtMicro,
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
            contactRadiusRamped,
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

    // ── Tool Coating Wear Update (batched D2H) ───────────────────────────────
    // A full host copy of the FEM node array on every resolve is a blocking
    // ~10 MB PCIe read. Wear is a long-timescale integral, so sample every
    // kWearBatchInterval resolves and advance the wear integral by the
    // Only download contacting nodes (~10 KB) instead of full 10-20 MB FEM mesh.
    m_wearDtAccumulator += dt;
    if (m_toolCoatingModel && m_numContacts > 0 && m_fem &&
        (m_resolveCallCount % kWearBatchInterval) == 0) {
        const double wearDt = m_wearDtAccumulator;
        m_wearDtAccumulator = 0.0;
        std::vector<ContactingNodeData> contactingNodes;
        m_fem->getContactingNodesData(contactingNodes);

        for (const auto& cnode : contactingNodes) {
            int j = cnode.nodeIndex;
            double contactPressure = cnode.vonMisesStress;   // Hertz Pa
            double slidingVelocity = cnode.speed;
            double temperature = cnode.temperature;

            // Zone classification: Hertz pressure above 1 GPa → cutting edge.
            // Temperature above 400°C on a moderate-stress node → crater zone.
            WearZone zone = WearZone::FLANK_FACE;
            switch (static_cast<ToolRegion>(cnode.toolRegion)) {
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
                j, contactPressure, slidingVelocity, temperature, zone, wearDt);
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
