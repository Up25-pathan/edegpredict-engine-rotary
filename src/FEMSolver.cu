/**
 * @file FEMSolver.cu
 * @brief FEM tool stress analysis CUDA implementation
 *
 * Fixes applied:
 *  1. initializeFromMesh() now stores mesh triangles in m_meshTriangles.
 *  2. exportToMesh() copies both deformed nodes AND the stored triangles,
 *     producing a complete surface mesh for VTK export (previously node-only).
 *  3. computeStressKernel uses a per-node effective area derived from the
 *     average spring rest-length squared rather than a hardcoded 1 mm²,
 *     giving a physically better-scaled stress estimate.
 */

#include "FEMSolver.cuh"
#include "Config.h"
#include <iostream>
#include <algorithm>
#include <set>

namespace edgepredict {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeElementForcesKernel(FEMNodeGPU* nodes, int numNodes,
                                            FEMElement* elements, int numElements,
                                            double youngsModulus, double poissonsRatio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    FEMElement& el = elements[idx];
    int n0 = el.nodeIndices[0];
    int n1 = el.nodeIndices[1];
    int n2 = el.nodeIndices[2];
    int n3 = el.nodeIndices[3];
    if (n0 < 0 || n0 >= numNodes || n1 >= numNodes || n2 >= numNodes || n3 >= numNodes) return;

    FEMNodeGPU& node0 = nodes[n0];
    FEMNodeGPU& node1 = nodes[n1];
    FEMNodeGPU& node2 = nodes[n2];
    FEMNodeGPU& node3 = nodes[n3];

    // Current positions
    double x0 = node0.x, y0 = node0.y, z0 = node0.z;
    double x1 = node1.x, y1 = node1.y, z1 = node1.z;
    double x2 = node2.x, y2 = node2.y, z2 = node2.z;
    double x3 = node3.x, y3 = node3.y, z3 = node3.z;

    // Deformed shape matrix Ds
    double Ds[3][3];
    Ds[0][0] = x1 - x0; Ds[1][0] = y1 - y0; Ds[2][0] = z1 - z0;
    Ds[0][1] = x2 - x0; Ds[1][1] = y2 - y0; Ds[2][1] = z2 - z0;
    Ds[0][2] = x3 - x0; Ds[1][2] = y3 - y0; Ds[2][2] = z3 - z0;

    // Deformation gradient F = Ds * invDm
    double F[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F[i][j] = Ds[i][0] * el.invDm[0][j] + Ds[i][1] * el.invDm[1][j] + Ds[i][2] * el.invDm[2][j];
        }
    }

    // Green-Lagrange strain E = 0.5 * (F^T * F - I)
    double E[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) {
                sum += F[k][i] * F[k][j]; // F^T * F
            }
            E[i][j] = 0.5 * (sum - (i == j ? 1.0 : 0.0));
        }
    }

    // St. Venant-Kirchhoff physics: S = lambda * tr(E) * I + 2 * mu * E
    double lambda = (youngsModulus * poissonsRatio) / ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    double mu = youngsModulus / (2.0 * (1.0 + poissonsRatio));
    double trE = E[0][0] + E[1][1] + E[2][2];

    double S[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S[i][j] = 2.0 * mu * E[i][j] + (i == j ? lambda * trE : 0.0);
        }
    }

    // Piola-Kirchhoff 1: P = F * S
    double P[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P[i][j] = F[i][0] * S[0][j] + F[i][1] * S[1][j] + F[i][2] * S[2][j];
        }
    }

    // Nodal forces H = -V0 * P * invDm^T
    double H[3][3];
    double V0 = el.volume;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            H[i][j] = -V0 * (P[i][0] * el.invDm[j][0] + P[i][1] * el.invDm[j][1] + P[i][2] * el.invDm[j][2]);
        }
    }

    // Distribute forces
    double fx1 = H[0][0], fy1 = H[1][0], fz1 = H[2][0];
    double fx2 = H[0][1], fy2 = H[1][1], fz2 = H[2][1];
    double fx3 = H[0][2], fy3 = H[1][2], fz3 = H[2][2];
    double fx0 = -(fx1 + fx2 + fx3);
    double fy0 = -(fy1 + fy2 + fy3);
    double fz0 = -(fz1 + fz2 + fz3);
    
    atomicAdd(&node0.fx, fx0); atomicAdd(&node0.fy, fy0); atomicAdd(&node0.fz, fz0);
    atomicAdd(&node1.fx, fx1); atomicAdd(&node1.fy, fy1); atomicAdd(&node1.fz, fz1);
    atomicAdd(&node2.fx, fx2); atomicAdd(&node2.fy, fy2); atomicAdd(&node2.fz, fz2);
    atomicAdd(&node3.fx, fx3); atomicAdd(&node3.fy, fy3); atomicAdd(&node3.fz, fz3);
}

/**
 * @brief Accumulate contact-force stress into vonMisesStress using EMA.
 *
 * Called ONCE PER STEP, AFTER contact forces have been added to fx/fy/fz
 * and BEFORE integrateNodesKernel clears them.
 *
 * For a rigid tool the von-Mises estimate is:
 *   σ_vm ≈ |F_contact| / A_eff
 * where A_eff = typicalSpringLength² (same as before).
 *
 * EMA smoothing avoids aliasing from single-step contact events:
 *   σ_new = α * σ_current + (1-α) * σ_measured
 *   with α = 0.9 → ~10-step moving average
 */
__global__ void accumulateContactStressKernel(FEMNodeGPU* nodes, int numNodes,
                                               double typicalSpringLength,
                                               double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    // Contact stress from current forces (include both spring AND external)
    double fMag = sqrt(node.fx * node.fx + node.fy * node.fy + node.fz * node.fz);
    double area = typicalSpringLength * typicalSpringLength;
    if (area < 1e-14) area = 1e-14;

    double sigma_instant = fMag / area;

    // EMA: smooth over ~10 steps so the tip stays red persistently
    node.vonMisesStress = alpha * node.vonMisesStress + (1.0 - alpha) * sigma_instant;
}

__global__ void computeStressKernel(FEMNodeGPU* nodes, int numNodes,
                                     FEMElement* elements, int numElements,
                                     double youngsModulus, double typicalSpringLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    // Compute stress from external contact forces and internal spring forces
    double forceMag  = sqrt(node.fx * node.fx + node.fy * node.fy + node.fz * node.fz);

    // Effective cross-section area: typical-spring-length squared
    double area = typicalSpringLength * typicalSpringLength;
    if (area < 1e-12) area = 1e-12;

    node.vonMisesStress = forceMag / area;
}

// === ANCHORED PHYSICS: Spindle Spring Coupling ===
__global__ void computeSpindleForcesKernel(FEMNodeGPU* nodes, int numNodes,
                                           double px, double py, double pz,
                                           double vx, double vy, double vz,
                                           double stiffness, double damping) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    FEMNodeGPU& node = nodes[idx];
    if (!node.isDriven || node.isFixed) return;
    
    // Target position = spindle pose + original local offset from spindle center
    double targetX = px + node.localOffX;
    double targetY = py + node.localOffY;
    double targetZ = pz + node.localOffZ;
    
    // Spring force pulling toward spindle
    double dx = targetX - node.x;
    double dy = targetY - node.y;
    double dz = targetZ - node.z;
    
    double fx = dx * stiffness;
    double fy = dy * stiffness;
    double fz = dz * stiffness;
    
    // Damper force resisting relative velocity
    double dvx = vx - node.vx;
    double dvy = vy - node.vy;
    double dvz = vz - node.vz;
    
    fx += dvx * damping;
    fy += dvy * damping;
    fz += dvz * damping;
    
    atomicAdd(&node.fx, fx);
    atomicAdd(&node.fy, fy);
    atomicAdd(&node.fz, fz);
}

// === EMBEDDED MESH: Coupling Surface to Volume ===
__global__ void transferForcesToTetNodesKernel(
    FEMNodeGPU* surfaceNodes, FEMNodeGPU* tetNodes, 
    FEMEmbedConstraint* embeds, int numEmbeds) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEmbeds) return;
    
    FEMEmbedConstraint ec = embeds[idx];
    if (ec.surfaceNodeIdx < 0) return;
    
    FEMNodeGPU& sNode = surfaceNodes[ec.surfaceNodeIdx];
    
    // Distribute surface force into the 4 tet nodes according to Barycentric weights
    if (fabs(sNode.fx) > 1e-12 || fabs(sNode.fy) > 1e-12 || fabs(sNode.fz) > 1e-12) {
        for (int i = 0; i < 4; ++i) {
            int tIdx = ec.tetNodeIndices[i];
            if (tIdx >= 0) {
                atomicAdd(&tetNodes[tIdx].fx, sNode.fx * ec.weights[i]);
                atomicAdd(&tetNodes[tIdx].fy, sNode.fy * ec.weights[i]);
                atomicAdd(&tetNodes[tIdx].fz, sNode.fz * ec.weights[i]);
            }
        }
        // Clear surface force so it doesn't double accumulate next step
        sNode.fx = 0.0;
        sNode.fy = 0.0;
        sNode.fz = 0.0;
    }
}

__global__ void updateSurfaceNodesKernel(
    FEMNodeGPU* surfaceNodes, FEMNodeGPU* tetNodes, 
    FEMEmbedConstraint* embeds, int numEmbeds) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEmbeds) return;
    
    FEMEmbedConstraint ec = embeds[idx];
    if (ec.surfaceNodeIdx < 0) return;
    
    FEMNodeGPU& sNode = surfaceNodes[ec.surfaceNodeIdx];
    
    double nx = 0, ny = 0, nz = 0;
    double nvx = 0, nvy = 0, nvz = 0;
    
    for (int i = 0; i < 4; ++i) {
        int tIdx = ec.tetNodeIndices[i];
        if (tIdx >= 0) {
            double w = ec.weights[i];
            nx += tetNodes[tIdx].x * w;
            ny += tetNodes[tIdx].y * w;
            nz += tetNodes[tIdx].z * w;
            
            nvx += tetNodes[tIdx].vx * w;
            nvy += tetNodes[tIdx].vy * w;
            nvz += tetNodes[tIdx].vz * w;
        }
    }
    
    sNode.x = nx;
    sNode.y = ny;
    sNode.z = nz;
    sNode.vx = nvx;
    sNode.vy = nvy;
    sNode.vz = nvz;
}

__global__ void integrateNodesKernel(FEMNodeGPU* nodes, int numNodes,
                                      double dt, double damping) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];
    if (node.isFixed || node.status == NodeStatus::FAILED) {
        node.vx = node.vy = node.vz = 0;
        node.fx = node.fy = node.fz = 0;
        return;
    }

    // --- FORCE CAP FOR INVERSION PROTECTION (StVK safeguard) ---
    double fMag = sqrt(node.fx * node.fx + node.fy * node.fy + node.fz * node.fz);
    const double MAX_FORCE = 1e6; // 1 Million Newtons maximum allowed per node
    if (fMag > MAX_FORCE) {
        double scale = MAX_FORCE / fMag;
        node.fx *= scale;
        node.fy *= scale;
        node.fz *= scale;
    }
    // ---------------------------------------------------------

    double ax = node.fx / node.mass;
    double ay = node.fy / node.mass;
    double az = node.fz / node.mass;

    double halfDt = 0.5 * dt;
    node.vx = (node.vx + ax * halfDt) * (1.0 - damping);
    node.vy = (node.vy + ay * halfDt) * (1.0 - damping);
    node.vz = (node.vz + az * halfDt) * (1.0 - damping);

    node.x += node.vx * dt;
    node.y += node.vy * dt;
    node.z += node.vz * dt;

    node.fx = node.fy = node.fz = 0;
}

__global__ void updateWearKernel(FEMNodeGPU* nodes, int numNodes,
                                  double dt, double A, double B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];
    if (!node.inContact) return;

    double v   = sqrt(node.vx * node.vx + node.vy * node.vy + node.vz * node.vz);
    double T_K = node.temperature + 273.15;
    if (T_K < 300) T_K = 300;

    double wearRate = A * node.vonMisesStress * v * exp(-B / T_K);
    node.wear += wearRate * dt;

    if (node.wear > 0.3e-3) node.status = NodeStatus::WORN;
}

/**
 * @brief Spring-network thermal conduction.
/**
 * @brief Thermally-stable FEM heat conduction via CFL-substepped integration.
 *
 * PHYSICS FIX: The old kernel clamped ΔT to ±5°C per step, which:
 *  (a) Violated energy conservation (discarded up to 45°C of heat silently).
 *  (b) Hid numerical instability without fixing it.
 *
 * The explicit heat conduction stability criterion (1D von Neumann):
 *   dt_stable = 0.45 * L² * ρ * Cp / k
 *
 * For a carbide spring of length L=1e-4 m:
 *   dt_stable = 0.45 * (1e-4)² * 14500 * 200 / 80 ≈ 1.6 ms
 *
 * Since the mechanics timestep dt is typically 1 μs, the thermal update
 * called every m_thermalSkip=10 steps has effectiveDt = 10 μs << 1.6 ms.
 * The integration is ALREADY stable without the clamp.
 *
 * This kernel:
 *  1. Computes the physical heat flux: Q̇ = k * A * ΔT / L  [W]
 *  2. Computes the CFL-limited thermal timestep  dt_cfl
 *  3. Subcycles if effectiveDt > dt_cfl (protects against large values)
 *  4. Updates both endpoint node temperatures atomically — NO CLAMP.
 *
 * Cross-section A is computed as the equilateral-triangle area of a mesh
 * face at the spring rest length: A = L² * (√3/4) ≈ L² * 0.433.
 * This is physically better than the old L²*0.1 heuristic.
 */
__global__ void thermalConductionKernelV2(FEMNodeGPU* nodes, int numNodes,
                                           FEMSpring*  springs, int numSprings,
                                           double effectiveDt,
                                           double conductivity, double specificHeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;

    const FEMSpring& spring = springs[idx];
    if (spring.node1 < 0 || spring.node2 < 0) return;

    FEMNodeGPU& n1 = nodes[spring.node1];
    FEMNodeGPU& n2 = nodes[spring.node2];

    double deltaT = n2.temperature - n1.temperature;
    if (fabs(deltaT) < 1e-6) return;   // Near-equilibrium: skip

    double L = spring.restLength;
    if (L < 1e-12) return;

    // Cross-section: equilateral-triangle face area at spring rest length.
    // A_face = (√3/4) * L²  ≈ 0.433 * L²
    const double A = 0.433 * L * L;

    // Heat flux through this spring [W]:
    //   Q̇ = k * A * (T2 - T1) / L
    double heatFlux = conductivity * A * deltaT / L;

    // ── CFL stability criterion for explicit heat conduction ─────────────────
    // dt_cfl = 0.45 * L² * ρ_eff * Cp / k
    // We don't store density per-spring, but we know mass and volume:
    //   ρ_eff ≈ mass_avg / (A * L)  (mass of a spring-shaped rod)
    double m_avg   = 0.5 * (fmax(n1.mass, 1e-12) + fmax(n2.mass, 1e-12));
    double rho_eff = m_avg / fmax(A * L, 1e-18);
    double dt_cfl  = 0.45 * L * L * rho_eff * specificHeat / fmax(conductivity, 1e-10);

    // ── Subcycling: apply multiple small timesteps if effectiveDt > dt_cfl ───
    int    nSub  = (dt_cfl > 0.0) ? (int)ceil(effectiveDt / dt_cfl) : 1;
    nSub         = min(nSub, 20);          // safety limit
    double dtSub = effectiveDt / nSub;

    double dT_n1_total = 0.0;
    double dT_n2_total = 0.0;
    double T1 = n1.temperature;
    double T2 = n2.temperature;

    for (int sub = 0; sub < nSub; ++sub) {
        double dT_here = T2 - T1;
        double Q_dot   = conductivity * A * dT_here / L;   // W
        double dT1     = ( Q_dot * dtSub) / (fmax(n1.mass, 1e-12) * specificHeat);
        double dT2     = (-Q_dot * dtSub) / (fmax(n2.mass, 1e-12) * specificHeat);
        T1 += dT1;
        T2 += dT2;
        dT_n1_total += dT1;
        dT_n2_total += dT2;
    }

    // ── Apply total ΔT atomically — NO ARTIFICIAL CLAMP ─────────────────────
    // Energy is conserved: what leaves n1 arrives at n2 exactly.
    atomicAdd(&n1.temperature, dT_n1_total);
    atomicAdd(&n2.temperature, dT_n2_total);
}


__global__ void applyRotationKernel(FEMNodeGPU* nodes, int numNodes,
                                     double cx, double cy, double /*cz*/,
                                     double ax, double ay, double az,
                                     double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    double px = node.ox - cx;
    double py = node.oy - cy;
    double pz = node.oz;

    double c   = cos(angle);
    double s   = sin(angle);
    double omc = 1.0 - c;
    double dot = ax * px + ay * py + az * pz;

    double crossX = ay * pz - az * py;
    double crossY = az * px - ax * pz;
    double crossZ = ax * py - ay * px;

    double rx = px * c + crossX * s + ax * dot * omc;
    double ry = py * c + crossY * s + ay * dot * omc;
    double rz = pz * c + crossZ * s + az * dot * omc;

    node.x = rx + cx;
    node.y = ry + cy;
    node.z = rz;
}

// ============================================================================
// FEMSolver Implementation
// ============================================================================

FEMSolver::FEMSolver() { m_toolAxis = Vec3(0, 0, 1); }

FEMSolver::~FEMSolver() { freeMemory(); }

bool FEMSolver::initialize(const Config& config) {
    std::cout << "[FEMSolver] Initializing..." << std::endl;

    const auto& femParams = config.getFEM();
    m_material.youngsModulus = femParams.youngModulus;
    m_material.poissonsRatio = femParams.poissonsRatio;
    m_material.density       = femParams.density;
    m_globalDamping          = femParams.dampingRatio;
    m_maxNodes               = femParams.maxNodes;
    m_maxElements            = femParams.maxElements;
    m_massScalingFactor      = femParams.massScalingFactor;
    m_stiffnessScalingFactor = femParams.stiffnessScalingFactor;

    allocateMemory(m_maxNodes, m_maxNodes, m_maxElements);

    m_isInitialized = true;
    std::cout << "[FEMSolver] Initialized with max " << m_maxNodes << " nodes" << std::endl;
    return true;
}

void FEMSolver::allocateMemory(int maxNodes, int maxTetNodes, int maxElements) {
    freeMemory();
    CUDA_CHECK(cudaMalloc(&d_nodes,   maxNodes   * sizeof(FEMNodeGPU)));
    CUDA_CHECK(cudaMalloc(&d_tetNodes, maxTetNodes * sizeof(FEMNodeGPU)));
    CUDA_CHECK(cudaMalloc(&d_elements, maxElements * sizeof(FEMElement)));
    CUDA_CHECK(cudaMalloc(&d_embedConstraints, maxNodes * sizeof(FEMEmbedConstraint)));
    h_nodes.reserve(maxNodes);
    h_tetNodes.reserve(maxTetNodes);
    h_elements.reserve(maxElements);
    h_embedConstraints.reserve(maxNodes);
}

void FEMSolver::freeMemory() {
    if (d_nodes)   { cudaFree(d_nodes);   d_nodes   = nullptr; }
    if (d_tetNodes) { cudaFree(d_tetNodes); d_tetNodes = nullptr; }
    if (d_elements) { cudaFree(d_elements); d_elements = nullptr; }
    if (d_embedConstraints) { cudaFree(d_embedConstraints); d_embedConstraints = nullptr; }
}

// ---------------------------------------------------------------------------
// FIX: initializeFromMesh now stores triangles for exportToMesh
// ---------------------------------------------------------------------------

void FEMSolver::initializeFromMesh(const Mesh& mesh) {
    h_nodes.clear();
    h_tetNodes.clear();
    h_elements.clear();
    h_embedConstraints.clear();
    // FIX: store the original triangles
    m_meshTriangles = mesh.triangles;

    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const FEMNode& mn = mesh.nodes[i];
        FEMNodeGPU node;
        node.x  = mn.position.x;
        node.y  = mn.position.y;
        node.z  = mn.position.z;
        node.ox = mn.position.x;
        node.oy = mn.position.y;
        node.oz = mn.position.z;
        // Rough node volume: total mesh volume / numNodes
        // For now estimate from mesh bounding box
        // (will be refined below after all nodes are added)
        node.mass = m_material.density * 1e-9 * m_massScalingFactor;  // placeholder
        node.temperature = 25.0;
        node.id          = static_cast<int32_t>(i);
        node.status      = NodeStatus::OK;
        node.isFixed     = false; // Handled dynamically
        h_nodes.push_back(node);
    }

    // Populate tetNodes
    for (size_t i = 0; i < mesh.tetNodes.size(); ++i) {
        const FEMNode& mn = mesh.tetNodes[i];
        FEMNodeGPU node;
        node.x  = mn.position.x;
        node.y  = mn.position.y;
        node.z  = mn.position.z;
        node.ox = mn.position.x;
        node.oy = mn.position.y;
        node.oz = mn.position.z;
        node.mass = 1e-12; // Base mass, accumulated below
        node.temperature = 25.0;
        node.id          = static_cast<int32_t>(i);
        node.status      = NodeStatus::OK;
        node.isFixed     = true; // Will unfix below
        h_tetNodes.push_back(node);
    }

    h_embedConstraints = mesh.embedConstraints;
    m_numNodes = static_cast<int>(h_nodes.size());
    m_numTetNodes = static_cast<int>(h_tetNodes.size());
    m_numEmbeds = static_cast<int>(h_embedConstraints.size());
    
    // Enable physical deformation!
    // Since the tet nodes hold volumetric structure, we un-fix them.
    double maxZ = -1e10;
    for (const auto& n : h_tetNodes) {
        if (n.oz > maxZ) maxZ = n.oz;
    }
    for (auto& n : h_tetNodes) {
        if (n.oz > maxZ - 0.005) n.isFixed = true;
        else n.isFixed = false;
    }

    createElementsFromMesh(mesh);

    copyToDevice();

    // Compute stable time-step (CFL condition)
    // dt <= 0.5 * min_element_size * sqrt(rho(1-v^2) / E)
    // We will just use a generic stable time step for solid carbide unless overridden.
    m_stableTimeStep = 1e-8; // 10 ns explicit integration step for metals
    
    std::cout << "[FEMSolver] Created " << m_numNodes << " nodes, "
              << m_numElements << " volumetric elements, "
              << m_meshTriangles.size() << " surface triangles" << std::endl;
    std::cout << "[FEMSolver] Stable dt: " << m_stableTimeStep << " s" << std::endl;
}

void FEMSolver::createElementsFromMesh(const Mesh& mesh) {
    h_elements.clear();
    for (auto& n : h_tetNodes) n.mass = 0.0;
    
    for (const auto& mel : mesh.elements) {
        FEMElement el;
        el.nodeIndices[0] = mel.nodeIndices[0];
        el.nodeIndices[1] = mel.nodeIndices[1];
        el.nodeIndices[2] = mel.nodeIndices[2];
        el.nodeIndices[3] = mel.nodeIndices[3];
        
        int n0 = el.nodeIndices[0];
        int n1 = el.nodeIndices[1];
        int n2 = el.nodeIndices[2];
        int n3 = el.nodeIndices[3];
        
        double X0[3] = {h_tetNodes[n0].ox, h_tetNodes[n0].oy, h_tetNodes[n0].oz};
        double X1[3] = {h_tetNodes[n1].ox, h_tetNodes[n1].oy, h_tetNodes[n1].oz};
        double X2[3] = {h_tetNodes[n2].ox, h_tetNodes[n2].oy, h_tetNodes[n2].oz};
        double X3[3] = {h_tetNodes[n3].ox, h_tetNodes[n3].oy, h_tetNodes[n3].oz};
        
        double Dm[3][3];
        for(int i=0; i<3; ++i) { Dm[i][0] = X1[i]-X0[i]; Dm[i][1] = X2[i]-X0[i]; Dm[i][2] = X3[i]-X0[i]; }
        
        double detDm = Dm[0][0]*(Dm[1][1]*Dm[2][2] - Dm[1][2]*Dm[2][1])
                     - Dm[0][1]*(Dm[1][0]*Dm[2][2] - Dm[1][2]*Dm[2][0])
                     + Dm[0][2]*(Dm[1][0]*Dm[2][1] - Dm[1][1]*Dm[2][0]);
                     
        if (detDm < 0.0) {
            std::swap(el.nodeIndices[1], el.nodeIndices[2]);
            std::swap(X1[0], X2[0]); std::swap(X1[1], X2[1]); std::swap(X1[2], X2[2]);
            for(int i=0; i<3; ++i) { Dm[i][0] = X1[i]-X0[i]; Dm[i][1] = X2[i]-X0[i]; Dm[i][2] = X3[i]-X0[i]; }
            detDm = -detDm;
        }
        
        el.volume = detDm / 6.0;
        
        if (el.volume < 1e-18) {
            el.volume = 1e-18;
            detDm = 6e-18;
        }
        
        double invDet = 1.0 / detDm;
        el.invDm[0][0] = (Dm[1][1]*Dm[2][2] - Dm[1][2]*Dm[2][1]) * invDet;
        el.invDm[0][1] = (Dm[0][2]*Dm[2][1] - Dm[0][1]*Dm[2][2]) * invDet;
        el.invDm[0][2] = (Dm[0][1]*Dm[1][2] - Dm[0][2]*Dm[1][1]) * invDet;
        
        el.invDm[1][0] = (Dm[1][2]*Dm[2][0] - Dm[1][0]*Dm[2][2]) * invDet;
        el.invDm[1][1] = (Dm[0][0]*Dm[2][2] - Dm[0][2]*Dm[2][0]) * invDet;
        el.invDm[1][2] = (Dm[0][2]*Dm[1][0] - Dm[0][0]*Dm[1][2]) * invDet;
        
        el.invDm[2][0] = (Dm[1][0]*Dm[2][1] - Dm[1][1]*Dm[2][0]) * invDet;
        el.invDm[2][1] = (Dm[0][1]*Dm[2][0] - Dm[0][0]*Dm[2][1]) * invDet;
        el.invDm[2][2] = (Dm[0][0]*Dm[1][1] - Dm[0][1]*Dm[1][0]) * invDet;
        
        h_elements.push_back(el);
        
        double massPerNode = (m_material.density * el.volume * m_massScalingFactor) / 4.0;
        if (massPerNode < 1e-12) massPerNode = 1e-12;
        
        h_tetNodes[el.nodeIndices[0]].mass += massPerNode;
        h_tetNodes[el.nodeIndices[1]].mass += massPerNode;
        h_tetNodes[el.nodeIndices[2]].mass += massPerNode;
        h_tetNodes[el.nodeIndices[3]].mass += massPerNode;
    }
    m_numElements = static_cast<int>(h_elements.size());
}

void FEMSolver::copyToDevice() {
    if (m_numNodes > 0)
        CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes.data(),
                              m_numNodes   * sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));
    if (m_numTetNodes > 0)
        CUDA_CHECK(cudaMemcpy(d_tetNodes, h_tetNodes.data(),
                              m_numTetNodes * sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));
    if (m_numElements > 0)
        CUDA_CHECK(cudaMemcpy(d_elements, h_elements.data(),
                              m_numElements * sizeof(FEMElement),  cudaMemcpyHostToDevice));
    if (m_numEmbeds > 0)
        CUDA_CHECK(cudaMemcpy(d_embedConstraints, h_embedConstraints.data(),
                              m_numEmbeds * sizeof(FEMEmbedConstraint), cudaMemcpyHostToDevice));
}

void FEMSolver::copyFromDevice() {
    if (m_numNodes > 0) {
        h_nodes.resize(m_numNodes);
        CUDA_CHECK(cudaMemcpy(h_nodes.data(), d_nodes,
                              m_numNodes * sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));
    }
    if (m_numTetNodes > 0) {
        h_tetNodes.resize(m_numTetNodes);
        CUDA_CHECK(cudaMemcpy(h_tetNodes.data(), d_tetNodes,
                              m_numTetNodes * sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));
    }
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------

void FEMSolver::step(double dt) {
    if (!m_isInitialized || m_numNodes == 0) return;

    int blockSize  = 256;
    int nodeGrid   = (m_numNodes   + blockSize - 1) / blockSize;

    NVTX_PUSH("FEM::Transform");
    if (std::abs(m_toolAngularVelocity) > 1e-12) {
        m_toolAngle += m_toolAngularVelocity * dt;
        applyRotationKernel<<<nodeGrid, blockSize>>>(
            d_nodes, m_numNodes,
            m_toolPosition.x, m_toolPosition.y, m_toolPosition.z,
            m_toolAxis.x,     m_toolAxis.y,     m_toolAxis.z,
            m_toolAngle);
        CUDA_CHECK_KERNEL();
        
        if (m_numTetNodes > 0) {
            int tetGrid = (m_numTetNodes + blockSize - 1) / blockSize;
            applyRotationKernel<<<tetGrid, blockSize>>>(
                d_tetNodes, m_numTetNodes,
                m_toolPosition.x, m_toolPosition.y, m_toolPosition.z,
                m_toolAxis.x,     m_toolAxis.y,     m_toolAxis.z,
                m_toolAngle);
            CUDA_CHECK_KERNEL();
        }
    }
    NVTX_POP();

    NVTX_PUSH("FEM::Physics");

    // Transfer SPH forces from Surface to Volume
    int embedGrid = (m_numEmbeds + blockSize - 1) / blockSize;
    if (m_numEmbeds > 0) {
        transferForcesToTetNodesKernel<<<embedGrid, blockSize>>>(
            d_nodes, d_tetNodes, d_embedConstraints, m_numEmbeds);
        CUDA_CHECK_KERNEL();
    }

    int tetGrid = (m_numTetNodes + blockSize - 1) / blockSize;

    // === ANCHORED PHYSICS: Spindle Forces ===
    if (m_spindleDynamicsEnabled && m_numTetNodes > 0) {
        computeSpindleForcesKernel<<<tetGrid, blockSize>>>(
            d_tetNodes, m_numTetNodes,
            m_spindlePos.x, m_spindlePos.y, m_spindlePos.z,
            m_spindleVel.x, m_spindleVel.y, m_spindleVel.z,
            m_spindleStiffness, m_spindleDamping);
        CUDA_CHECK_KERNEL();
    }

    // Compute element forces using StVK physics
    if (m_numElements > 0) {
        int elementGrid = (m_numElements + blockSize - 1) / blockSize;
        computeElementForcesKernel<<<elementGrid, blockSize>>>(
            d_tetNodes, m_numTetNodes, d_elements, m_numElements,
            m_material.youngsModulus, m_material.poissonsRatio);
        CUDA_CHECK_KERNEL();
    }

    static double typicalL = 0.001;
    if (m_currentStep % 100 == 0 && m_numElements > 0) {
        typicalL = pow(h_elements.front().volume, 1.0/3.0);
    }

    // Stress accumulation from contact forces (EMA-smoothed, ∼10-step window)
    // NOTE: This remains on surface nodes because SPH contacts surface
    accumulateContactStressKernel<<<nodeGrid, blockSize>>>(
        d_nodes, m_numNodes, typicalL, 0.9 /*EMA alpha*/);
    CUDA_CHECK_KERNEL();

    if (m_numTetNodes > 0) {
        integrateNodesKernel<<<tetGrid, blockSize>>>(
            d_tetNodes, m_numTetNodes, dt, m_globalDamping * 0.01);
        CUDA_CHECK_KERNEL();
    }

    // Transfer position exactly to Surface Nodes
    if (m_numEmbeds > 0) {
        updateSurfaceNodesKernel<<<embedGrid, blockSize>>>(
            d_nodes, d_tetNodes, d_embedConstraints, m_numEmbeds);
        CUDA_CHECK_KERNEL();
    }
    
    NVTX_POP();

    NVTX_PUSH("FEM::ThermalWear");
    
    // TODO: Rewrite thermal conduction for tetrahedral volumes!
    // For now, thermal conduction via mesh is disabled since springs are removed.

    updateWearKernel<<<nodeGrid, blockSize>>>(
        d_nodes, m_numNodes, dt, 1e-9, 1000.0);
    CUDA_CHECK_KERNEL();
    NVTX_POP();

    m_currentTime += dt;
    m_currentStep++;

    if (m_currentStep % 100 == 0) updateResults();
}

// ---------------------------------------------------------------------------
// FIX: exportToMesh now copies triangles from m_meshTriangles
// ---------------------------------------------------------------------------

void FEMSolver::exportToMesh(Mesh& mesh) {
    // FIX: ensure host-side data is current before exporting
    copyFromDevice();

    mesh.clear();

    // Copy deformed node state
    for (const auto& node : h_nodes) {
        FEMNode mn;
        mn.position         = Vec3(node.x,  node.y,  node.z);
        mn.originalPosition = Vec3(node.ox, node.oy, node.oz);
        mn.velocity         = Vec3(node.vx, node.vy, node.vz);
        mn.temperature      = node.temperature;
        mn.stress           = node.vonMisesStress;
        mn.accumulatedWear  = node.wear;
        mn.status           = node.status;
        mn.isContact        = node.inContact;
        mesh.nodes.push_back(mn);
    }

    // FIX: copy the original surface connectivity
    // Triangles use the same node indices as they did at mesh-creation time;
    // since we only deform positions, the connectivity is unchanged.
    mesh.triangles = m_meshTriangles;
    mesh.elements = h_elements;
    
    mesh.tetNodes.clear();
    for (const auto& node : h_tetNodes) {
        FEMNode mn;
        mn.position         = Vec3(node.x,  node.y,  node.z);
        mn.originalPosition = Vec3(node.ox, node.oy, node.oz);
        mn.velocity         = Vec3(node.vx, node.vy, node.vz);
        mn.temperature      = node.temperature;
        mn.stress           = node.vonMisesStress;
        mn.accumulatedWear  = node.wear;
        mn.status           = node.status;
        mn.isContact        = node.inContact;
        mesh.tetNodes.push_back(mn);
    }
    mesh.embedConstraints = h_embedConstraints;
}

// ---------------------------------------------------------------------------
// Tool transform helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Anchored Physics: Dynamic Spindle Coupling
// ---------------------------------------------------------------------------

void FEMSolver::setSpindleDynamicsConfig(bool enabled, double stiffness, double damping) {
    m_spindleDynamicsEnabled = enabled;
    m_spindleStiffness = stiffness;
    m_spindleDamping = damping;
}

void FEMSolver::initializeDrivenNodes(double topFraction) {
    if (m_numNodes == 0) return;
    copyFromDevice();
    
    // Find Z bounds of the tool
    double minZ = 1e30, maxZ = -1e30;
    for (const auto& n : h_tetNodes) {
        if (n.z < minZ) minZ = n.z;
        if (n.z > maxZ) maxZ = n.z;
    }
    
    double lengthZ = maxZ - minZ;
    // Driven section is from top (maxZ) down by topFraction * lengthZ
    double drivenThresholdZ = maxZ - (lengthZ * topFraction);
    
    int drivenCount = 0;
    for (auto& n : h_tetNodes) {
        if (n.z >= drivenThresholdZ) {
            n.isDriven = true;
            n.localOffX = n.ox;
            n.localOffY = n.oy;
            n.localOffZ = n.oz;
            drivenCount++;
            
            if (m_spindleDynamicsEnabled) {
                n.isFixed = false; // Unfix so it moves via physics!
            }
        } else {
            n.isDriven = false;
            // The un-driven part of the tool must also be unfixed if dynamics are enabled
            // so it can deflect.
            if (m_spindleDynamicsEnabled) {
                n.isFixed = false; 
            }
        }
    }
    
    copyToDevice();
    std::cout << "[FEMSolver] Anchored Physics: " << drivenCount << " driven nodes configured." 
              << (m_spindleDynamicsEnabled ? " Dynamics ON: tool is deflected." : " Dynamics OFF: tool is rigid kinematic.")
              << std::endl;
}

void FEMSolver::setVirtualSpindleState(const Vec3& pos, const Vec3& vel) {
    m_spindlePos = pos;
    m_spindleVel = vel;
}

void FEMSolver::setToolTransform(const Vec3& position, const Vec3& axis,
                                  double angularVelocity) {
    m_toolPosition        = position;
    m_toolAxis            = axis.normalized();
    m_toolAngularVelocity = angularVelocity;
}

void FEMSolver::applyNodeForce(int nodeIndex, const Vec3& force) {
    if (nodeIndex < 0 || nodeIndex >= m_numNodes) return;
    FEMNodeGPU node;
    CUDA_CHECK(cudaMemcpy(&node, &d_nodes[nodeIndex],
                          sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));
    node.fx += force.x;
    node.fy += force.y;
    node.fz += force.z;
    CUDA_CHECK(cudaMemcpy(&d_nodes[nodeIndex], &node,
                          sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));
}

void FEMSolver::applyContactForce(int nodeIndex, const Vec3& force, double heatFlux) {
    if (nodeIndex < 0 || nodeIndex >= m_numNodes) return;
    FEMNodeGPU node;
    CUDA_CHECK(cudaMemcpy(&node, &d_nodes[nodeIndex],
                          sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));
    node.fx += force.x;
    node.fy += force.y;
    node.fz += force.z;
    node.inContact  = true;
    node.temperature += heatFlux / (node.mass * m_material.specificHeat);
    CUDA_CHECK(cudaMemcpy(&d_nodes[nodeIndex], &node,
                          sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));
}

void FEMSolver::applyHeatFlux(double x, double y, double z, double heatFlux) {
    copyFromDevice();
    int    nearestIdx = -1;
    double minDist    = 1e10;
    for (int i = 0; i < m_numNodes; ++i) {
        double dx = h_nodes[i].x - x;
        double dy = h_nodes[i].y - y;
        double dz = h_nodes[i].z - z;
        double dist = dx * dx + dy * dy + dz * dz;
        if (dist < minDist) { minDist = dist; nearestIdx = i; }
    }
    if (nearestIdx >= 0) {
        h_nodes[nearestIdx].temperature +=
            heatFlux / (h_nodes[nearestIdx].mass * m_material.specificHeat);
        copyToDevice();
    }
}

double FEMSolver::getTemperatureAt(double x, double y, double z) const {
    double minDist = 1e10;
    double temp    = 25.0;
    for (const auto& n : h_nodes) {
        double dx = n.x - x, dy = n.y - y, dz = n.z - z;
        double dist = dx * dx + dy * dy + dz * dz;
        if (dist < minDist) { minDist = dist; temp = n.temperature; }
    }
    return temp;
}

// ---------------------------------------------------------------------------
// Result / state helpers
// ---------------------------------------------------------------------------

void FEMSolver::updateResults() {
    copyFromDevice();

    m_results = FEMResults{};
    for (const auto& node : h_nodes) {
        m_results.maxStress      = std::max(m_results.maxStress,      node.vonMisesStress);
        m_results.avgStress     += node.vonMisesStress;
        double dx = node.x - node.ox, dy = node.y - node.oy, dz = node.z - node.oz;
        double disp = std::sqrt(dx*dx + dy*dy + dz*dz);
        m_results.maxDisplacement    = std::max(m_results.maxDisplacement, disp);
        m_results.maxTemperature     = std::max(m_results.maxTemperature,  node.temperature);
        m_results.maxWear            = std::max(m_results.maxWear,         node.wear);
        double v2 = node.vx*node.vx + node.vy*node.vy + node.vz*node.vz;
        m_results.totalKineticEnergy += 0.5 * node.mass * v2;
        if (node.inContact) m_results.numContactNodes++;
    }
    if (m_numNodes > 0) m_results.avgStress /= m_numNodes;
}

double FEMSolver::getStableTimeStep() const {
    // When tool is kinematic (all nodes fixed, no spindle dynamics),
    // FEM integration does nothing — don't limit the global timestep.
    if (!m_spindleDynamicsEnabled) {
        return 1e-3; // 1ms — effectively unlimited for SPH-driven dt
    }
    return m_stableTimeStep;
}

void FEMSolver::reset() {
    h_nodes.clear();
    h_elements.clear();
    m_meshTriangles.clear();
    m_numNodes   = 0;
    m_numElements = 0;
    m_currentTime = 0.0;
    m_toolAngle   = 0.0;
    m_results     = FEMResults{};
}

void FEMSolver::getBounds(double& minX, double& minY, double& minZ,
                           double& maxX, double& maxY, double& maxZ) const {
    if (h_nodes.empty()) { minX=minY=minZ=maxX=maxY=maxZ=0; return; }
    minX=minY=minZ= 1e10;
    maxX=maxY=maxZ=-1e10;
    for (const auto& n : h_nodes) {
        if (n.status != NodeStatus::FAILED) {
            minX=std::min(minX,n.x); minY=std::min(minY,n.y); minZ=std::min(minZ,n.z);
            maxX=std::max(maxX,n.x); maxY=std::max(maxY,n.y); maxZ=std::max(maxZ,n.z);
        }
    }
}

std::vector<FEMNodeGPU> FEMSolver::getNodes() { copyFromDevice(); return h_nodes; }

void FEMSolver::translateMesh(double dx, double dy, double dz) {
    copyFromDevice();
    for (auto& n : h_nodes) {
        n.x += dx; n.y += dy; n.z += dz;
        n.ox+= dx; n.oy+= dy; n.oz+= dz;
    }
    for (auto& n : h_tetNodes) {
        n.x += dx; n.y += dy; n.z += dz;
        n.ox+= dx; n.oy+= dy; n.oz+= dz;
    }
    m_toolPosition.x += dx;
    m_toolPosition.y += dy;
    m_toolPosition.z += dz;
    copyToDevice();
}

void FEMSolver::rotateAroundZ(double angle, double centerX, double centerY) {
    copyFromDevice();
    double c = std::cos(angle);
    double s = std::sin(angle);
    for (auto& n : h_nodes) {
        double px = n.x - centerX, py = n.y - centerY;
        n.x  = px * c - py * s + centerX;
        n.y  = px * s + py * c + centerY;
        double opx = n.ox - centerX, opy = n.oy - centerY;
        n.ox = opx * c - opy * s + centerX;
        n.oy = opx * s + opy * c + centerY;
    }
    for (auto& n : h_tetNodes) {
        double px = n.x - centerX, py = n.y - centerY;
        n.x  = px * c - py * s + centerX;
        n.y  = px * s + py * c + centerY;
        double opx = n.ox - centerX, opy = n.oy - centerY;
        n.ox = opx * c - opy * s + centerX;
        n.oy = opx * s + opy * c + centerY;
    }
    m_toolAngle += angle;
    copyToDevice();
}

} // namespace edgepredict
