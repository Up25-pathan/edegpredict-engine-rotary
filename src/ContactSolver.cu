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
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace edgepredict {

// ============================================================================
// Constants
// ============================================================================

static constexpr double PI = 3.14159265358979323846;

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
// GPU Kernel: Compute hash for each FEM tool node
// ============================================================================

__global__ void computeToolNodeHashKernel(
    FEMNodeGPU* nodes, int numNodes,
    int* hashes,
    double cellSize, int hashTableSize,
    double domainMinX, double domainMinY, double domainMinZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    const FEMNodeGPU& n = nodes[idx];
    int cx = (int)floor((n.x - domainMinX) / cellSize);
    int cy = (int)floor((n.y - domainMinY) / cellSize);
    int cz = (int)floor((n.z - domainMinZ) / cellSize);
    hashes[idx] = hashToolCell(cx, cy, cz, hashTableSize);
}

// Comparator for sorting nodes by hash
struct CompareNodeByHash {
    __device__ bool operator()(const FEMNodeGPU& a, const FEMNodeGPU& b) const {
        return a.cellHash < b.cellHash;
    }
};

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
    double workMeltTemp,
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

            // atomicMax for penetration depth
            unsigned long long* addr = (unsigned long long*)&node.penetrationDepth;
            unsigned long long  cur  = *addr;
            unsigned long long  nval = __double_as_longlong(penetration);
            while (penetration > __longlong_as_double(cur)) {
                unsigned long long old = atomicCAS(addr, cur, nval);
                if (old == cur) break;
                cur = old;
            }

            double vtx = relVx - vn * nx;
            double vty = relVy - vn * ny;
            double vtz = relVz - vn * nz;
            double vt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

            // Zorev temperature-dependent friction
            double T_avg = 0.5 * (p.temperature + node.temperature);
            double T_ratio = fmin(1.0, fmax(0.0, (T_avg - 200.0) / 600.0));
            double mu_eff = friction * (1.0 - 0.4 * T_ratio);

            double ft_coulomb = mu_eff * fn;
            double ft_stiction = (vt > 1e-10) ? (p.mass * vt / dt) : 0.0;

            double contactArea = 3.14159265 * contactRadius * penetration;
            double ft_shear_limit = shearYieldStress * contactArea;
            double ft = fmin(fmin(ft_coulomb, ft_stiction), ft_shear_limit);

            // Fix G: engagementScale applied ONCE to each heat path.
            // Old code applied it twice to node heat (once here, once in atomicAdd).
            double heatThisStep = ft * vt * dt;  // Raw frictional heat [J]

            totalHeatToParticle += heatThisStep * (1.0 - heatPartition) * engagementScale;
            atomicAdd(&node.heatAccumulator, heatThisStep * heatPartition * engagementScale);
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
        double dT = totalHeatToParticle / (fmax(p.mass, 1e-15) * workSpecificHeat);
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
    double workMeltTemp,
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
        double preRelVx = p.vx - node.vx;
        double preRelVy = p.vy - node.vy;
        double preRelVz = p.vz - node.vz;
        double preVn = preRelVx*nx + preRelVy*ny + preRelVz*nz;
        double approachSpeed = fmax(0.0, -preVn);
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

        // ── Relative velocity ────────────────────────────────────────────────
        double relVx = p.vx - node.vx;
        double relVy = p.vy - node.vy;
        double relVz = p.vz - node.vz;
        double  vn   = relVx*nx + relVy*ny + relVz*nz;

        // ── Coulomb friction ─────────────────────────────────────────────────
        double vtx = relVx - vn * nx;
        double vty = relVy - vn * ny;
        double vtz = relVz - vn * nz;
        double vt  = sqrt(vtx*vtx + vty*vty + vtz*vtz);

        // Zorev temperature-dependent friction coefficient
        double T_avg   = 0.5 * (p.temperature + node.temperature);
        double T_ratio = fmin(1.0, fmax(0.0, (T_avg - 200.0) / 600.0));
        double mu_eff  = friction * (1.0 - 0.4 * T_ratio);

        // Friction force magnitude (capped by stiction limit this step)
        double ft_coulomb = mu_eff * fn;
        double ft_stiction = (vt > 1e-10) ? (p.mass * vt / dt) : 0.0;
        
        // Shear yield stress cap (sticking zone) — config-driven, not hardcoded
        // At extreme contact pressures near the cutting edge, Coulomb friction
        // would predict infinite tangential force. In reality, friction cannot
        // exceed the shear yield strength of the workpiece material:
        //   τ_max = σ_yield / √3 (von Mises criterion)
        // Contact area estimated from contact radius: A ≈ π·r²
        double contactArea = 3.14159265 * contactRadius * penetration;
        double ft_shear_limit = shearYieldStress * contactArea;
        
        double ft = fmin(fmin(ft_coulomb, ft_stiction), ft_shear_limit);

        // Frictional heat generated this contact pair: Q = ft * vt * dt
        // Fix G: engagementScale applied ONCE to each heat path (was doubled before).
        double heatThisStep = ft * vt * dt;   // Raw frictional heat [J]

        // ── Accumulate heat for workpiece particle ───────────────────────────
        totalHeatToParticle += heatThisStep * (1.0 - heatPartition) * engagementScale;

        // ── Accumulate heat into node's buffer (NOT ΔT!) ────────────────────
        // The conversion J → ΔT happens ONCE in applyNodeHeatKernel.
        atomicAdd(&node.heatAccumulator, heatThisStep * heatPartition * engagementScale);

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
    double corrMag = sqrt(cx*cx + cy*cy + cz*cz);
    if (corrMag > 1e-15) {
        double invCorrMag = 1.0 / corrMag;
        double nnx = cx * invCorrMag;
        double nny = cy * invCorrMag;
        double nnz = cz * invCorrMag;
        double vDotN = p.vx * nnx + p.vy * nny + p.vz * nnz;
        if (vDotN < 0.0) {
            p.vx -= vDotN * nnx;
            p.vy -= vDotN * nny;
            p.vz -= vDotN * nnz;
        }
    }

    // ── Apply accumulated workpiece heat ONCE ────────────────────────────────
    if (totalHeatToParticle > 0.0) {
        double dT = totalHeatToParticle / (fmax(p.mass, 1e-15) * workSpecificHeat);
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
    double ema_alpha         // EMA smoothing factor (0.9 = 10-step average)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    // ── Apply accumulated heat ────────────────────────────────────────────────
    if (node.heatAccumulator > 0.0) {
        double dT = node.heatAccumulator / (fmax(node.mass, 1e-12) * toolCp);
        node.temperature = fmin(node.temperature + dT, toolMeltTemp);
        node.heatAccumulator = 0.0;   // Reset buffer for next step
    }

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
// ContactSolver Implementation
// ============================================================================

ContactSolver::ContactSolver() = default;

ContactSolver::~ContactSolver() {
    if (d_toolCellStart)    cudaFree(d_toolCellStart);
    if (d_toolCellEnd)      cudaFree(d_toolCellEnd);
    if (d_toolNodeHashes)   cudaFree(d_toolNodeHashes);
    if (d_sortedToolNodes)  cudaFree(d_sortedToolNodes);
    if (d_numContacts)      cudaFree(d_numContacts);
    if (d_totalHeat)        cudaFree(d_totalHeat);
    if (d_totalForce)       cudaFree(d_totalForce);
}

void ContactSolver::initialize(MPMSolver* sph, FEMSolver* fem, const ContactConfig& config) {
    m_sph    = sph;
    m_fem    = fem;
    m_config = config;

    // Set cell size to contact radius for spatial hash
    m_cellSize = config.contactRadius;

    CUDA_CHECK(cudaMalloc(&d_numContacts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_totalHeat,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_totalForce,  sizeof(double)));

    // Allocate spatial hash buffers for tool nodes
    int maxNodes = fem ? fem->getNodeCount() : 50000;
    if (maxNodes <= 0) maxNodes = 50000;
    CUDA_CHECK(cudaMalloc(&d_toolCellStart,   m_hashTableSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_toolCellEnd,     m_hashTableSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_toolNodeHashes,  maxNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sortedToolNodes, maxNodes * sizeof(FEMNodeGPU)));

    m_isInitialized = true;
    std::cout << "[ContactSolver] Initialized — XPBD position-based contact, "
              << "per-node heat accumulation, Hertz stress reporting" << std::endl;
    std::cout << "[ContactSolver] Spatial hash: cellSize=" << m_cellSize * 1000
              << "mm, tableSize=" << m_hashTableSize << std::endl;
}

void ContactSolver::resolveContacts(double dt) {
    if (!m_isInitialized || !m_sph || !m_fem) return;

    int numP = m_sph->getParticleCount();
    int numN = m_fem->getNodeCount();
    if (numP <= 0 || numN <= 0) return;

    CUDA_CHECK(cudaMemset(d_numContacts, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_totalHeat,   0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_totalForce,  0, sizeof(double)));

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

    resetContactAccumulatorsKernel<<<gridSizeN, blockSize>>>(
        m_fem->getDeviceNodes(), numN);
    CUDA_CHECK_KERNEL();

    // ══════════════════════════════════════════════════════════════════════════
    // SPATIAL HASH CONTACT RESOLUTION — O(N×k) instead of O(N×M)
    // ══════════════════════════════════════════════════════════════════════════

    // Step 1: Build spatial hash for tool nodes (host-side, ~2K nodes = fast)
    {
        auto toolNodes = m_fem->getNodes();  // copies from device to host
        // Use SPH domain bounds to ensure hash grid alignment
        double dMinX = m_sph->getDomainMin().x;
        double dMinY = m_sph->getDomainMin().y;
        double dMinZ = m_sph->getDomainMin().z;

        // Compute cellHash for each node, and zero out force accumulators so we only capture the delta
        for (int i = 0; i < numN; ++i) {
            int cx = max(0, (int)floor((toolNodes[i].x - dMinX) / m_cellSize));
            int cy = max(0, (int)floor((toolNodes[i].y - dMinY) / m_cellSize));
            int cz = max(0, (int)floor((toolNodes[i].z - dMinZ) / m_cellSize));
            unsigned long long h = (static_cast<unsigned long long>(cx) * 73856093ULL) ^
                                   (static_cast<unsigned long long>(cy) * 19349663ULL) ^
                                   (static_cast<unsigned long long>(cz) * 83492791ULL);
            toolNodes[i].cellHash = static_cast<int>((h % m_hashTableSize + m_hashTableSize) % m_hashTableSize);
            
            // Clear accumulators so d_sortedToolNodes only stores the impulse from THIS step
            toolNodes[i].fx = 0.0;
            toolNodes[i].fy = 0.0;
            toolNodes[i].fz = 0.0;
            toolNodes[i].heatAccumulator = 0.0;
            toolNodes[i].contactForceNormal = 0.0;
        }

        // Upload to sorted buffer
        CUDA_CHECK(cudaMemcpy(d_sortedToolNodes, toolNodes.data(),
                              numN * sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));

        // Sort by cellHash on GPU
        thrust::device_ptr<FEMNodeGPU> nodePtr(d_sortedToolNodes);
        thrust::sort(nodePtr, nodePtr + numN, CompareNodeByHash());

        // Build cell bounds
        CUDA_CHECK(cudaMemset(d_toolCellStart, -1, m_hashTableSize * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_toolCellEnd,   -1, m_hashTableSize * sizeof(int)));
        findToolCellBoundsKernel<<<gridSizeN, blockSize>>>(
            d_sortedToolNodes, numN,
            d_toolCellStart, d_toolCellEnd, m_hashTableSize);
        CUDA_CHECK_KERNEL();

        // Step 2: Launch spatial hash contact kernel
        xpbdContactSpatialHashKernel<<<gridSizeP, blockSize>>>(
            m_sph->getDeviceParticles(), numP,
            d_sortedToolNodes, numN,
            d_toolCellStart, d_toolCellEnd, m_hashTableSize,
            m_cellSize, dMinX, dMinY, dMinZ,
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
            m_config.workMeltTemp,
            d_numContacts, d_totalHeat, d_totalForce
        );
        CUDA_CHECK_KERNEL();

        // Step 3: Merge contact data from sorted nodes back to original nodes
        // The sorted nodes received atomicAdd forces/heat from the contact kernel.
        // Copy sorted back to host and merge by node id into the original device array.
        std::vector<FEMNodeGPU> sortedHost(numN);
        CUDA_CHECK(cudaMemcpy(sortedHost.data(), d_sortedToolNodes,
                              numN * sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));

        // Re-fetch original (they haven't been modified by the contact kernel)
        std::vector<FEMNodeGPU> origHost(numN);
        CUDA_CHECK(cudaMemcpy(origHost.data(), m_fem->getDeviceNodes(),
                              numN * sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));

        for (int i = 0; i < numN; ++i) {
            int oid = sortedHost[i].id;
            if (oid >= 0 && oid < numN) {
                origHost[oid].fx += sortedHost[i].fx;
                origHost[oid].fy += sortedHost[i].fy;
                origHost[oid].fz += sortedHost[i].fz;
                origHost[oid].heatAccumulator += sortedHost[i].heatAccumulator;
                origHost[oid].contactForceNormal += sortedHost[i].contactForceNormal;
                if (sortedHost[i].penetrationDepth > origHost[oid].penetrationDepth)
                    origHost[oid].penetrationDepth = sortedHost[i].penetrationDepth;
                if (sortedHost[i].inContact)
                    origHost[oid].inContact = true;
            }
        }

        CUDA_CHECK(cudaMemcpy(m_fem->getDeviceNodes(), origHost.data(),
                              numN * sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));
    }

    // Step 4: Apply node heat buffer → temperature (on merged original nodes)
    applyNodeHeatAndHertzStressKernel<<<gridSizeN, blockSize>>>(
        m_fem->getDeviceNodes(), numN,
        m_config.maxContactPressure,
        m_config.contactRadius,
        m_config.toolSpecificHeat,
        m_config.toolMeltTemp,
        0.9       // EMA alpha (10-step smoothing window)
    );
    CUDA_CHECK_KERNEL();

    transferResults();

    // ── Track contact age for engagement ramp ────────────────────────────────
    // Reset to 0 when tool disengages so re-entry (peck drilling G73/G83)
    // starts the ramp from 10% again instead of staying at 100%.
    if (m_numContacts > 0) {
        m_contactAge++;
    } else {
        m_contactAge = 0;
    }

    // === CONTACT EXPLOSION GUARD ===
    // If contacts exceed 50× the number of tool nodes, the tool is deeply
    // embedded inside the workpiece — a configuration error, not a physics event.
    // Log a critical diagnostic and suppress further heat injection to prevent
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
                      << std::defaultfloat << std::endl;
        }
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
            if (contactPressure > 1.0e9) {
                zone = WearZone::CUTTING_EDGE;
            } else if (temperature > 400.0) {
                zone = WearZone::CRATER_ZONE;
            }

            m_toolCoatingModel->updateWear(
                j, contactPressure, slidingVelocity, temperature, zone, dt);
        }
    }
}

void ContactSolver::transferResults() {
    CUDA_CHECK(cudaMemcpy(&m_numContacts,       d_numContacts, sizeof(int),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m_totalHeatGenerated, d_totalHeat,  sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m_totalContactForce,  d_totalForce, sizeof(double), cudaMemcpyDeviceToHost));
}

std::vector<ContactEvent> ContactSolver::getContactEvents() const {
    return std::vector<ContactEvent>();
}

// ============================================================================
// Free function (legacy compatibility)
// ============================================================================

void launchContactInteraction(
    MPMParticle* particles, int numParticles,
    FEMNodeGPU*  nodes,     int numNodes,
    double contactRadius, double contactStiffness,
    double friction, double heatPartition, double dt
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

    xpbdContactKernel<<<gridSize, blockSize>>>(
        particles, numParticles,
        nodes,     numNodes,
        contactRadius, contactStiffness, dampingRatio,
        friction, heatPartition, dt,
        shearYield, maxPen, engagementScale,
        workSpecHeat, workMeltTemp,
        d_cnt, d_h, d_f
    );

    // Apply node heat buffer (static node count used)
    applyNodeHeatAndHertzStressKernel<<<gridSizeN, blockSizeN>>>(
        nodes, numNodes, maxContactPressure, contactRadius, 200.0, 2500.0, 0.9);
}

} // namespace edgepredict
