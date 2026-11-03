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
#include "DeviceCoupling.h"
#include "Config.h"
#include <iostream>
#include <algorithm>
#include <set>

namespace edgepredict {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Thermal conduction through tetrahedral element edges.
 *
 * Each tet has 4 nodes → 6 edges. For each edge, heat flows by Fourier's law:
 *   Q = k · (A/L) · ΔT · dt
 * where A ≈ L² (cross-section estimate) and L is edge length.
 *
 * Replaces the disabled spring-based thermal conduction.
 */
__global__ void thermalConductionTetKernel(
    FEMNodeGPU* nodes, int numNodes,
    FEMElement* elements, int numElements,
    double thermalConductivity, double specificHeat, double dt) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    const FEMElement& el = elements[idx];

    // Exact lumped-mass FEM thermal integration using shape function gradients
    int n0 = el.nodeIndices[0];
    int n1 = el.nodeIndices[1];
    int n2 = el.nodeIndices[2];
    int n3 = el.nodeIndices[3];

    if (n0 < 0 || n0 >= numNodes || n3 >= numNodes) return;

    // Shape function gradients ∇N (3x4 matrix)
    // For a tetrahedron, ∇N_1, ∇N_2, ∇N_3 are the columns of invDm^T (or rows of invDm)
    // ∇N_0 = -(∇N_1 + ∇N_2 + ∇N_3)
    double gradN[3][4];
    for (int i = 0; i < 3; ++i) {
        gradN[i][1] = el.invDm[0][i];
        gradN[i][2] = el.invDm[1][i];
        gradN[i][3] = el.invDm[2][i];
        gradN[i][0] = -(gradN[i][1] + gradN[i][2] + gradN[i][3]);
    }

    double T[4] = { nodes[n0].temperature, nodes[n1].temperature, nodes[n2].temperature, nodes[n3].temperature };

    // K matrix: K_ij = ∫ k ∇N_i · ∇N_j dV = k * V * (∇N_i · ∇N_j)
    // Heat flux Q_i = -dt * Σ K_ij T_j
    double Q[4] = {0, 0, 0, 0};
    double kVdt = thermalConductivity * el.volume * dt;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double dot = gradN[0][i]*gradN[0][j] + gradN[1][i]*gradN[1][j] + gradN[2][i]*gradN[2][j];
            Q[i] -= kVdt * dot * T[j];
        }
    }

    if (nodes[n0].physicalMass > 1e-15 && specificHeat > 0) atomicAdd(&nodes[n0].conductionBuffer, Q[0]);
    if (nodes[n1].physicalMass > 1e-15 && specificHeat > 0) atomicAdd(&nodes[n1].conductionBuffer, Q[1]);
    if (nodes[n2].physicalMass > 1e-15 && specificHeat > 0) atomicAdd(&nodes[n2].conductionBuffer, Q[2]);
    if (nodes[n3].physicalMass > 1e-15 && specificHeat > 0) atomicAdd(&nodes[n3].conductionBuffer, Q[3]);
}

__global__ void applyTetHeatKernel(
    FEMNodeGPU* nodes, int numNodes, double specificHeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];
    // Use physicalMass (unscaled) for thermal calculations
    if (node.physicalMass > 1e-15 && specificHeat > 0) {
        double dT = (node.heatAccumulator + node.conductionBuffer) / (node.physicalMass * specificHeat);
        if (dT > 25.0) dT = 25.0; // Max 25°C per microstep (~25,000,000 °C/s physical ceiling)
        node.temperature += dT;
        node.heatAccumulator = 0.0;
        node.conductionBuffer = 0.0;
        // Physically cap at tungsten carbide melting/decomposition point (~2870°C) and min 20°C
        if (node.temperature > 2870.0) node.temperature = 2870.0;
        if (node.temperature < 20.0)   node.temperature = 20.0;
    }
}

/**
 * @brief Enforce thermal boundary conditions on FEM nodes.
 *
 * Dirichlet BC (Item 3): Nodes in the SHANK region are clamped to ambient
 *   temperature, modeling the large thermal mass of the spindle/holder.
 *
 * Convective Robin BC (Item 4): Boundary nodes exposed to coolant have
 *   Q_cool = HTC · A_eff · (T - T_ambient) subtracted each thermal step.
 *
 * Radiative BC (Item 8): All externally exposed nodes lose heat via
 *   Q_rad = ε·σ·A_eff·(T⁴ - Tₐ⁴)·dt  (combined into the same kernel).
 */
__global__ void applyThermalBCKernel(
    FEMNodeGPU* nodes, int numNodes,
    double ambientTemp,
    double coolantHTC,      // from CoolantHardeningModel
    double stefanBoltzmann,
    double emissivity,
    double toolDensity,
    double specificHeat,
    double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    // Item 3: Dirichlet BC — shank nodes fixed at ambient temperature
    // Only apply to nodes far from cutting edge and not in active contact
    // to prevent the entire tool from being clamped to ambient (infinite heat sink).
    bool isCuttingRegion = (node.toolRegion == static_cast<int32_t>(ToolRegion::RAKE_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::INSERT_RAKE_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::CUTTING_EDGE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::FLANK_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::MARGIN) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::LAND) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::RELIEF_FACE) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::THREAD_FLANK) ||
                            node.toolRegion == static_cast<int32_t>(ToolRegion::CHIP_GULLET));
    
    if (node.toolRegion == static_cast<int32_t>(ToolRegion::SHANK) && !isCuttingRegion && !node.inContact) {
        node.temperature = ambientTemp;
        return; // No cooling/heating needed for clamped nodes
    }

    // Estimate node surface area from PHYSICAL mass (same approach as contact kernel)
    double vol = fmax(node.physicalMass / toolDensity, 1e-15);
    double area = pow(vol, 2.0 / 3.0);

    // Item 4: Convective Robin BC — coolant flow on exposed surfaces
    if (coolantHTC > 0.0 && node.temperature > ambientTemp + 0.1) {
        double dT = node.temperature - ambientTemp;
        double Q_cool = coolantHTC * area * dT * dt;
        double massCp = fmax(node.physicalMass * specificHeat, 1e-15);
        double cooling = fmin(Q_cool / massCp, dT * 0.5); // max 50% per step
        node.temperature -= cooling;
    }

    // Item 8: Stefan-Boltzmann radiation on all exposed nodes
    if (emissivity > 0.0 && node.temperature > ambientTemp + 10.0) {
        double Tn = node.temperature + 273.15;
        double Ta = ambientTemp + 273.15;
        double Tn4 = Tn * Tn * Tn * Tn;
        double Ta4 = Ta * Ta * Ta * Ta;
        double Q_rad = emissivity * stefanBoltzmann * area * (Tn4 - Ta4) * dt;
        double massCp = fmax(node.physicalMass * specificHeat, 1e-15);
        double dT_rad = fmin(Q_rad / massCp, (node.temperature - ambientTemp) * 0.3);
        node.temperature -= dT_rad;
    }
}

/**
 * @brief Corotational Linear Elastic element forces kernel.
 *
 * Replaces StVK to eliminate inversion instability under extreme compression.
 * Extracts rotation R from the deformation gradient F via polar decomposition
 * (cross-product method), then applies Hooke's law in the rotated frame.
 *
 * Stable for arbitrarily large rotations and moderate compression.
 * If det(F) < 0 (inverted element), clamps F to prevent explosion.
 */
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

    // Deformed shape matrix Ds
    double Ds[3][3];
    Ds[0][0] = node1.x - node0.x; Ds[1][0] = node1.y - node0.y; Ds[2][0] = node1.z - node0.z;
    Ds[0][1] = node2.x - node0.x; Ds[1][1] = node2.y - node0.y; Ds[2][1] = node2.z - node0.z;
    Ds[0][2] = node3.x - node0.x; Ds[1][2] = node3.y - node0.y; Ds[2][2] = node3.z - node0.z;

    // Deformation gradient F = Ds * invDm
    double F[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            F[i][j] = Ds[i][0]*el.invDm[0][j] + Ds[i][1]*el.invDm[1][j] + Ds[i][2]*el.invDm[2][j];

    // ── Element erosion: weaken or kill highly distorted/inverted elements ──
    double detF = F[0][0]*(F[1][1]*F[2][2] - F[1][2]*F[2][1])
                - F[0][1]*(F[1][0]*F[2][2] - F[1][2]*F[2][0])
                + F[0][2]*(F[1][0]*F[2][1] - F[1][1]*F[2][0]);
    double erosion = 1.0;
    if (detF < 0.1) {
        // Progressive stiffness degradation as element inverts:
        //   erosion = (detF / 0.1)²  →  0 at detF=0, 1 at detF=0.1
        erosion = detF / 0.1;
        erosion = erosion * erosion; // quadratic for smooth transition
        double scale = cbrt(0.1 / fmax(detF, 1e-9));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F[i][j] *= scale;
    }
    if (erosion < 1e-6) {
        // Fix 9: Fully eroded element — just return, no need for 12 atomicAdds of zero
        return;
    }

    // ── Polar decomposition: extract rotation R from F ──
    // Higham's iteration for R = F * (F^T F)^{-1/2} with scaling
    double R[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = F[i][j];

    for (int iter = 0; iter < 20; ++iter) {
        double cof[3][3];
        cof[0][0] =  R[1][1]*R[2][2] - R[1][2]*R[2][1];
        cof[0][1] = -(R[1][0]*R[2][2] - R[1][2]*R[2][0]);
        cof[0][2] =  R[1][0]*R[2][1] - R[1][1]*R[2][0];
        cof[1][0] = -(R[0][1]*R[2][2] - R[0][2]*R[2][1]);
        cof[1][1] =  R[0][0]*R[2][2] - R[0][2]*R[2][0];
        cof[1][2] = -(R[0][0]*R[2][1] - R[0][1]*R[2][0]);
        cof[2][0] =  R[0][1]*R[1][2] - R[0][2]*R[1][1];
        cof[2][1] = -(R[0][0]*R[1][2] - R[0][2]*R[1][0]);
        cof[2][2] =  R[0][0]*R[1][1] - R[0][1]*R[1][0];

        double detR = R[0][0]*cof[0][0] + R[0][1]*cof[0][1] + R[0][2]*cof[0][2];
        if (fabs(detR) < 1e-12) break;
        double invDetR = 1.0 / detR;

        // Norms for scaling
        double normR1 = 0, normRInf = 0;
        double normC1 = 0, normCInf = 0;
        for(int i=0; i<3; ++i) {
            double sumR = fabs(R[i][0]) + fabs(R[i][1]) + fabs(R[i][2]);
            double sumC = (fabs(cof[0][i]) + fabs(cof[1][i]) + fabs(cof[2][i])) * invDetR;
            if(sumR > normRInf) normRInf = sumR;
            if(sumC > normC1) normC1 = sumC;
        }
        for(int j=0; j<3; ++j) {
            double sumR = fabs(R[0][j]) + fabs(R[1][j]) + fabs(R[2][j]);
            double sumC = (fabs(cof[j][0]) + fabs(cof[j][1]) + fabs(cof[j][2])) * invDetR;
            if(sumR > normR1) normR1 = sumR;
            if(sumC > normCInf) normCInf = sumC;
        }

        double gamma = pow((normC1 * normCInf) / fmax(1e-12, normR1 * normRInf), 0.25);
        double igamma = 1.0 / gamma;
        
        double maxDiff = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double nval = 0.5 * (gamma * R[i][j] + igamma * cof[j][i] * invDetR);
                double diff = fabs(R[i][j] - nval);
                if (diff > maxDiff) maxDiff = diff;
                R[i][j] = nval;
            }
        }
        if (maxDiff < 1e-6) break;
    }

    // ── Corotational Cauchy strain: ε = R^T·F − I ──
    double eps[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < 3; ++k)
                sum += R[k][i] * F[k][j];  // R^T * F
            eps[i][j] = sum - (i == j ? 1.0 : 0.0);
        }
    }

    // ── Hooke's law in rotated frame: σ = λ·tr(ε)·I + 2μ·ε ──
    double E_eff = youngsModulus * erosion;
    double lambda = (E_eff * poissonsRatio) / ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    double mu = E_eff / (2.0 * (1.0 + poissonsRatio));
    double trEps = eps[0][0] + eps[1][1] + eps[2][2];

    double sigma[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sigma[i][j] = 2.0*mu*eps[i][j] + (i == j ? lambda*trEps : 0.0);

    // ── Rotate stress back: P = R · σ ──
    double P[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            P[i][j] = R[i][0]*sigma[0][j] + R[i][1]*sigma[1][j] + R[i][2]*sigma[2][j];

    // ── FIX 1.4: symmetric Cauchy stress for nodal reporting ──
    // P = R·σ is the (non-symmetric) first Piola-Kirchhoff for the corotational
    // force (H = -V0·P·invDmᵀ is correct for linear tetrahedra). But the code
    // accumulated the six components of P into SYMMETRIC stress storage and fed
    // them into the von Mises yield formula — dropping the complementary shears
    // (P10, P20, P21) and making the reported stress rotate arbitrarily with the
    // tool angle θ (violates material frame indifference). Accumulate the true
    // symmetric Cauchy stress σ_c = R·σ·Rᵀ instead (detF≈1 in the corotational
    // small-strain setting, so no 1/detF factor).
    double sigCauchy[3][3] = {{0.0}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    sigCauchy[i][j] += R[i][k] * sigma[k][l] * R[j][l];

    // ── Nodal forces: H = −V₀ · P · invDmᵀ ──
    double H[3][3];
    double V0 = el.volume;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            H[i][j] = -V0 * (P[i][0]*el.invDm[j][0] + P[i][1]*el.invDm[j][1] + P[i][2]*el.invDm[j][2]);

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

    // Accumulate element Cauchy stress (symmetric global frame) to nodes
    // Volume-weighted: each node gets elementStress * elementVolume / 4
    double w = V0 * 0.25;
    atomicAdd(&node0.stress_xx, sigCauchy[0][0] * w); atomicAdd(&node0.stress_yy, sigCauchy[1][1] * w); atomicAdd(&node0.stress_zz, sigCauchy[2][2] * w);
    atomicAdd(&node0.stress_xy, sigCauchy[0][1] * w); atomicAdd(&node0.stress_xz, sigCauchy[0][2] * w); atomicAdd(&node0.stress_yz, sigCauchy[1][2] * w);
    atomicAdd(&node0.stressWeight, w);
    atomicAdd(&node1.stress_xx, sigCauchy[0][0] * w); atomicAdd(&node1.stress_yy, sigCauchy[1][1] * w); atomicAdd(&node1.stress_zz, sigCauchy[2][2] * w);
    atomicAdd(&node1.stress_xy, sigCauchy[0][1] * w); atomicAdd(&node1.stress_xz, sigCauchy[0][2] * w); atomicAdd(&node1.stress_yz, sigCauchy[1][2] * w);
    atomicAdd(&node1.stressWeight, w);
    atomicAdd(&node2.stress_xx, sigCauchy[0][0] * w); atomicAdd(&node2.stress_yy, sigCauchy[1][1] * w); atomicAdd(&node2.stress_zz, sigCauchy[2][2] * w);
    atomicAdd(&node2.stress_xy, sigCauchy[0][1] * w); atomicAdd(&node2.stress_xz, sigCauchy[0][2] * w); atomicAdd(&node2.stress_yz, sigCauchy[1][2] * w);
    atomicAdd(&node2.stressWeight, w);
    atomicAdd(&node3.stress_xx, sigCauchy[0][0] * w); atomicAdd(&node3.stress_yy, sigCauchy[1][1] * w); atomicAdd(&node3.stress_zz, sigCauchy[2][2] * w);
    atomicAdd(&node3.stress_xy, sigCauchy[0][1] * w); atomicAdd(&node3.stress_xz, sigCauchy[0][2] * w); atomicAdd(&node3.stress_yz, sigCauchy[1][2] * w);
    atomicAdd(&node3.stressWeight, w);
}

// accumulateContactStressKernel removed: stress is correctly computed 
// from Hertzian contact mechanics in ContactSolver.cu, and this kernel 
// was double-counting raw forces that are cleared during integration.


// ── Normalize element-averaged stress and compute von Mises ──
__global__ void normalizeNodeStressKernel(FEMNodeGPU* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    FEMNodeGPU& n = nodes[idx];
    if (n.stressWeight > 1e-30) {
        double invW = 1.0 / n.stressWeight;
        n.stress_xx *= invW; n.stress_yy *= invW; n.stress_zz *= invW;
        n.stress_xy *= invW; n.stress_xz *= invW; n.stress_yz *= invW;

        // von Mises from full Cauchy stress tensor
        double s1 = n.stress_xx - n.stress_yy;
        double s2 = n.stress_yy - n.stress_zz;
        double s3 = n.stress_zz - n.stress_xx;
        double seq = sqrt(0.5 * (s1*s1 + s2*s2 + s3*s3
                        + 6.0*(n.stress_xy*n.stress_xy
                             + n.stress_xz*n.stress_xz
                             + n.stress_yz*n.stress_yz)));
        // Exact continuum von Mises stress from Cauchy stress tensor
        n.vonMisesStress = seq;
    }
}

// ── P3 FIX: Tool Plasticity Kernel ──────────────────────────────────────────
// Performs J2 radial return mapping on nodes in cutting-edge regions.
// Only activates for tool regions: CUTTING_EDGE(3), RAKE_FACE(1,2), FLANK_FACE(5),
// MARGIN(10), LAND(11), CHIP_GULLET(4), RELIEF_FACE(6).
// Uses von Mises yield with isotropic hardening: sigma_y = sigma_y0 + H * eps_p_eq
__global__ void toolPlasticityKernel(FEMNodeGPU* nodes, int numNodes,
                                      double youngsModulus, double poissonsRatio,
                                      double initialYieldStress, double hardeningModulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];
    
    // Skip non-cutting regions
    // ToolRegion enum: UNKNOWN=0, RAKE_FACE=1, INSERT_RAKE_FACE=2, CUTTING_EDGE=3,
    // CHIP_GULLET=4, FLANK_FACE=5, RELIEF_FACE=6, MARGIN=10, LAND=11, SHANK=12, etc.
    int region = node.toolRegion;
    bool isCuttingRegion = (region == 1 || region == 2 || region == 3 || 
                           region == 4 || region == 5 || region == 6 || 
                           region == 10 || region == 11);
    
    if (!isCuttingRegion) {
        // Initialize yield stress for cutting regions
        if (node.yieldStress <= 0.0) {
            node.yieldStress = initialYieldStress;
            node.hardeningModulus = hardeningModulus;
        }
        return;
    }
    
    // Initialize yield stress and hardening if first time
    if (node.yieldStress <= 0.0) {
        node.yieldStress = initialYieldStress;
        node.hardeningModulus = hardeningModulus;
        node.equivalentPlasticStrain = 0.0;
    }
    
    // Current stress tensor
    double s_xx = node.stress_xx;
    double s_yy = node.stress_yy;
    double s_zz = node.stress_zz;
    double s_xy = node.stress_xy;
    double s_xz = node.stress_xz;
    double s_yz = node.stress_yz;
    
    // Von Mises equivalent stress
    double trace = s_xx + s_yy + s_zz;
    double dev_xx = s_xx - trace / 3.0;
    double dev_yy = s_yy - trace / 3.0;
    double dev_zz = s_zz - trace / 3.0;
    double seq = sqrt(0.5 * (dev_xx*dev_xx + dev_yy*dev_yy + dev_zz*dev_zz +
                6.0*(s_xy*s_xy + s_xz*s_xz + s_yz*s_yz)));
    
    // Yield stress with isotropic hardening
    double sigma_y = node.yieldStress + node.hardeningModulus * node.equivalentPlasticStrain;
    
    // Check yield condition
    if (seq <= sigma_y * 1.001) return;  // Below yield (small tolerance)
    
    // ── Radial Return Mapping (J2 Plasticity) ────────────────────────────────
    // Deviatoric stress direction
    double n_xx = (seq > 1e-12) ? dev_xx / seq : 0.0;
    double n_yy = (seq > 1e-12) ? dev_yy / seq : 0.0;
    double n_zz = (seq > 1e-12) ? dev_zz / seq : 0.0;
    double n_xy = (seq > 1e-12) ? s_xy / seq : 0.0;
    double n_xz = (seq > 1e-12) ? s_xz / seq : 0.0;
    double n_yz = (seq > 1e-12) ? s_yz / seq : 0.0;
    
    // Elastic shear modulus
    double G = youngsModulus / (2.0 * (1.0 + poissonsRatio));
    
    // Plastic strain increment: delta_eps = (seq - sigma_y) / (3G)
    double delta_eps = (seq - sigma_y) / (3.0 * G);
    if (delta_eps < 1e-12) delta_eps = 1e-12;
    
    // Cap plastic strain increment per step
    delta_eps = fmin(delta_eps, 0.05);
    
    // Update deviatoric stress: s_new = s_trial - 2G * delta_eps * n
    s_xx = trace / 3.0 + dev_xx - 2.0 * G * delta_eps * n_xx;
    s_yy = trace / 3.0 + dev_yy - 2.0 * G * delta_eps * n_yy;
    s_zz = trace / 3.0 + dev_zz - 2.0 * G * delta_eps * n_zz;
    s_xy = s_xy - 2.0 * G * delta_eps * n_xy;
    s_xz = s_xz - 2.0 * G * delta_eps * n_xz;
    s_yz = s_yz - 2.0 * G * delta_eps * n_yz;
    
    // Update plastic strain tensor
    double deps_factor = 1.5 * delta_eps / fmax(seq, 1e-12);
    node.plasticStrain_xx += deps_factor * dev_xx;
    node.plasticStrain_yy += deps_factor * dev_yy;
    node.plasticStrain_zz += deps_factor * dev_zz;
    node.plasticStrain_xy += deps_factor * s_xy;
    node.plasticStrain_xz += deps_factor * s_xz;
    node.plasticStrain_yz += deps_factor * s_yz;
    
    // Update equivalent plastic strain
    node.equivalentPlasticStrain += delta_eps;
    
    // Update yield stress (isotropic hardening)
    node.yieldStress = initialYieldStress + hardeningModulus * node.equivalentPlasticStrain;
    
    // Write back corrected stress
    node.stress_xx = s_xx;
    node.stress_yy = s_yy;
    node.stress_zz = s_zz;
    node.stress_xy = s_xy;
    node.stress_xz = s_xz;
    node.stress_yz = s_yz;
    
    // Recompute von Mises
    dev_xx = s_xx - trace / 3.0;
    dev_yy = s_yy - trace / 3.0;
    dev_zz = s_zz - trace / 3.0;
    node.vonMisesStress = sqrt(0.5 * (dev_xx*dev_xx + dev_yy*dev_yy + dev_zz*dev_zz +
                        6.0*(s_xy*s_xy + s_xz*s_xz + s_yz*s_yz)));
}

__global__ void resetNodeStressKernel(FEMNodeGPU* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    FEMNodeGPU& n = nodes[idx];
    n.stress_xx = 0; n.stress_yy = 0; n.stress_zz = 0;
    n.stress_xy = 0; n.stress_xz = 0; n.stress_yz = 0;
    n.stressWeight = 0;
    n.vonMisesStress = 0;
}

// === ANCHORED PHYSICS: Spindle Spring Coupling ===
__global__ void computeSpindleForcesKernel(FEMNodeGPU* nodes, int numNodes,
                                           double px, double py, double pz,
                                           double vx, double vy, double vz,
                                           double stiffness, double damping,
                                           double ang, double ax, double ay, double az) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    FEMNodeGPU& node = nodes[idx];
    if (!node.isDriven || node.isFixed) return;

    // FIX 1.3: rotate the captured (unrotated) local offset by the CURRENT
    // tool orientation before using it as the spring anchor. The old code
    // anchored to the static unrotated offset, so once the spindle rotated,
    // the 50 MN/m springs pulled every driven collet node back toward its
    // initial orientation — an artificial restoring torque that drove
    // high-frequency torsional vibration. This is the same axis-angle
    // rotation applyRotationKernel applies to the node positions, so the
    // spring target now follows the rotating tool.
    double c = cos(ang), s = sin(ang), omc = 1.0 - c;
    double ox = node.localOffX, oy = node.localOffY, oz = node.localOffZ;
    double dot = ax * ox + ay * oy + az * oz;
    double crossX = ay * oz - az * oy;
    double crossY = az * ox - ax * oz;
    double crossZ = ax * oy - ay * ox;
    double rotOffX = ox * c + crossX * s + ax * dot * omc;
    double rotOffY = oy * c + crossY * s + ay * dot * omc;
    double rotOffZ = oz * c + crossZ * s + az * dot * omc;

    // Target position = spindle pose + rotated local offset from spindle center
    double targetX = px + rotOffX;
    double targetY = py + rotOffY;
    double targetZ = pz + rotOffZ;
    
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

__global__ void transferSurfaceHeatToTetNodesKernel(
    FEMNodeGPU* surfaceNodes, FEMNodeGPU* tetNodes,
    FEMEmbedConstraint* embeds, int numEmbeds,
    double specificHeat, double transferFraction)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEmbeds) return;

    FEMEmbedConstraint ec = embeds[idx];
    if (ec.surfaceNodeIdx < 0) return;

    FEMNodeGPU& sNode = surfaceNodes[ec.surfaceNodeIdx];
    if (sNode.temperature <= 25.01) return;

    double tetTemp = 0.0;
    for (int i = 0; i < 4; ++i) {
        int tIdx = ec.tetNodeIndices[i];
        if (tIdx >= 0) tetTemp += tetNodes[tIdx].temperature * ec.weights[i];
    }

    double deltaT = sNode.temperature - tetTemp;
    if (deltaT <= 0.0) return;

    double physMass = fmax(sNode.physicalMass, 1e-12);
    double q = deltaT * physMass * specificHeat * transferFraction;
    sNode.temperature -= q / (physMass * specificHeat);

    for (int i = 0; i < 4; ++i) {
        int tIdx = ec.tetNodeIndices[i];
        if (tIdx >= 0) {
            atomicAdd(&tetNodes[tIdx].heatAccumulator, q * ec.weights[i]);
        }
    }
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

    // ── Velocity-based energy guard (replaces unphysical 1MN force cap) ──
    // The corotational model with det(F) guard prevents most explosions,
    // but as a safety net, cap acceleration to prevent runaway.
    double fMag = sqrt(node.fx*node.fx + node.fy*node.fy + node.fz*node.fz);
    double maxAccel = 1e8;  // ~100km/s² — physical for impact dynamics
    double maxForce = node.mass * maxAccel;
    if (fMag > maxForce && fMag > 0) {
        double scale = maxForce / fMag;
        node.fx *= scale;
        node.fy *= scale;
        node.fz *= scale;
    }

    double invMass = 1.0 / fmax(node.mass, 1e-12);
    double ax = node.fx * invMass;
    double ay = node.fy * invMass;
    double az = node.fz * invMass;

    // Symplectic Euler Integration
    node.vx += ax * dt;
    node.vy += ay * dt;
    node.vz += az * dt;
    
    // Position update with full velocity
    node.x += node.vx * dt;
    node.y += node.vy * dt;
    node.z += node.vz * dt;
    
    // dt-scaled damping: damping is the per-microsecond fraction (0.001 = 0.1%/μs).
    // Without dt-scaling, a 0.1s macro-step would also remove only 0.1% → far too
    // little damping for the physical time elapsed.  Use exp() for consistency
    // across all step sizes (CFL micro-steps and air-cutting macro-steps).
    double dampFactor = exp(-damping * dt);
    node.vx *= dampFactor;
    node.vy *= dampFactor;
    node.vz *= dampFactor;

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

    // Unscaled physical thermal masses
    double m1 = fmax(n1.physicalMass > 1e-15 ? n1.physicalMass : n1.mass, 1e-12);
    double m2 = fmax(n2.physicalMass > 1e-15 ? n2.physicalMass : n2.mass, 1e-12);

    // ── CFL stability criterion for explicit heat conduction ─────────────────
    // dt_cfl = 0.45 * L² * ρ_eff * Cp / k
    // We don't store density per-spring, but we know mass and volume:
    //   ρ_eff ≈ mass_avg / (A * L)  (mass of a spring-shaped rod)
    double m_avg   = 0.5 * (m1 + m2);
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
        double dT1     = ( Q_dot * dtSub) / (m1 * specificHeat);
        double dT2     = (-Q_dot * dtSub) / (m2 * specificHeat);
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
                                     double cx, double cy, double cz,
                                     double ax, double ay, double az,
                                     double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    double c   = cos(angle);
    double s   = sin(angle);
    double omc = 1.0 - c;

    double px = node.x - cx;
    double py = node.y - cy;
    double pz = node.z - cz;
    double dot = ax * px + ay * py + az * pz;
    double crossX = ay * pz - az * py;
    double crossY = az * px - ax * pz;
    double crossZ = ax * py - ay * px;

    double rx = px * c + crossX * s + ax * dot * omc;
    double ry = py * c + crossY * s + ay * dot * omc;
    double rz = pz * c + crossZ * s + az * dot * omc;

    node.x = rx + cx;
    node.y = ry + cy;
    node.z = rz + cz;

    double opx = node.ox - cx;
    double opy = node.oy - cy;
    double opz = node.oz - cz;
    double odot = ax * opx + ay * opy + az * opz;
    double ocrossX = ay * opz - az * opy;
    double ocrossY = az * opx - ax * opz;
    double ocrossZ = ax * opy - ay * opx;

    node.ox = opx * c + ocrossX * s + ax * odot * omc + cx;
    node.oy = opy * c + ocrossY * s + ay * odot * omc + cy;
    node.oz = opz * c + ocrossZ * s + az * odot * omc + cz;
}

// ============================================================================
// FEMSolver Implementation
// ============================================================================

// ============================================================================
// TWO-WAY KINEMATIC COUPLING: Cutting Force Distribution Kernels
// ============================================================================

__global__ void findToolBoundsZKernel(
    FEMNodeGPU* nodes, int numNodes,
    double* minZ, double* maxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    double z = nodes[idx].z;
    atomicMinDouble(minZ, z);
    atomicMaxDouble(maxZ, z);
}

__global__ void countCuttingZoneNodesKernel(
    FEMNodeGPU* nodes, int numNodes,
    double toolMinZ, double toolMaxZ,
    int* zoneCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    FEMNodeGPU& node = nodes[idx];
    if (node.isDriven || node.isFixed) return;
    double toolLen = toolMaxZ - toolMinZ;
    if (toolLen < 1e-12) return;
    double zFrac = (node.z - toolMinZ) / toolLen;
    if (zFrac <= 0.3) {
        atomicAdd(zoneCount, 1);
    }
}

__global__ void resetFEMNodeForcesKernel(FEMNodeGPU* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    nodes[idx].fx = 0.0;
    nodes[idx].fy = 0.0;
    nodes[idx].fz = 0.0;
}

__global__ void initBoundsZKernel(double* minZ, double* maxZ) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *minZ = 1e30;
        *maxZ = -1e30;
    }
}

__global__ void applyCuttingForcesDistributedKernel(
    FEMNodeGPU* nodes, int numNodes,
    double fx, double fy, double fz,
    const double* d_minMax,
    int zoneCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    if (zoneCount <= 0) return;
    FEMNodeGPU& node = nodes[idx];
    if (node.isDriven || node.isFixed) return;
    double toolMinZ = d_minMax[0];
    double toolMaxZ = d_minMax[1];
    double toolLen = toolMaxZ - toolMinZ;
    if (toolLen < 1e-12) return;
    double zFrac = (node.z - toolMinZ) / toolLen;
    if (zFrac > 0.3) return;
    double share = 1.0 / (double)zoneCount;
    atomicAdd(&node.fx, fx * share);
    atomicAdd(&node.fy, fy * share);
    atomicAdd(&node.fz, fz * share);
}

// ── Rigid-body velocity kernel (device-side kinematics, no CPU round-trip) ──

__global__ void setRigidBodyVelocitiesKernel(
    FEMNodeGPU* nodes, int numNodes,
    double vx, double vy, double vz,
    double omegaZ, double centerX, double centerY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    double rx = nodes[idx].x - centerX;
    double ry = nodes[idx].y - centerY;
    nodes[idx].vx = vx - omegaZ * ry;
    nodes[idx].vy = vy + omegaZ * rx;
    nodes[idx].vz = vz;
}

__global__ void translateMeshKernel(FEMNodeGPU* nodes, int numNodes,
    double dx, double dy, double dz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    nodes[idx].x += dx; nodes[idx].y += dy; nodes[idx].z += dz;
    nodes[idx].ox += dx; nodes[idx].oy += dy; nodes[idx].oz += dz;
}

__global__ void rotateAroundZKernel(FEMNodeGPU* nodes, int numNodes,
    double angle, double centerX, double centerY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    double px = nodes[idx].x - centerX;
    double py = nodes[idx].y - centerY;
    double c = cos(angle);
    double s = sin(angle);
    nodes[idx].x = px * c - py * s + centerX;
    nodes[idx].y = px * s + py * c + centerY;
    double opx = nodes[idx].ox - centerX;
    double opy = nodes[idx].oy - centerY;
    nodes[idx].ox = opx * c - opy * s + centerX;
    nodes[idx].oy = opx * s + opy * c + centerY;
}

// ============================================================================
// Coupling Point Sync Kernel (Device-side thermal coupling with CFD)
// ============================================================================

__global__ void syncFEMCouplingPointsKernel(
    const FEMNodeGPU* nodes, CouplingPoint* out, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    out[idx].x = nodes[idx].x;
    out[idx].y = nodes[idx].y;
    out[idx].z = nodes[idx].z;
    out[idx].temperature = nodes[idx].temperature;
}

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

    CUDA_CHECK(cudaStreamCreate(&m_computeStream));
    CUDA_CHECK(cudaEventCreate(&m_stepEvent));

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
    CUDA_CHECK(cudaMalloc(&d_couplingPoints, maxNodes * sizeof(CouplingPoint)));
    CUDA_CHECK(cudaMalloc(&d_cuttingMinMax, 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cuttingZoneCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wearData, maxNodes * sizeof(NodeWearData)));
    CUDA_CHECK(cudaMalloc(&d_contactingNodes, maxNodes * sizeof(ContactingNodeData)));
    CUDA_CHECK(cudaMalloc(&d_contactingCount, sizeof(int)));
    int htcBlocks = (maxNodes + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_htcBlockMax, htcBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_htcBlockSum, htcBlocks * sizeof(double)));
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
    if (d_couplingPoints) { cudaFree(d_couplingPoints); d_couplingPoints = nullptr; }
    if (d_cuttingMinMax) { cudaFree(d_cuttingMinMax); d_cuttingMinMax = nullptr; }
    if (d_cuttingZoneCount) { cudaFree(d_cuttingZoneCount); d_cuttingZoneCount = nullptr; }
    if (d_wearData) { cudaFree(d_wearData); d_wearData = nullptr; }
    if (d_contactingNodes) { cudaFree(d_contactingNodes); d_contactingNodes = nullptr; }
    if (d_contactingCount) { cudaFree(d_contactingCount); d_contactingCount = nullptr; }
    if (d_htcBlockMax) { cudaFree(d_htcBlockMax); d_htcBlockMax = nullptr; }
    if (d_htcBlockSum) { cudaFree(d_htcBlockSum); d_htcBlockSum = nullptr; }
    if (m_stepEvent) { cudaEventDestroy(m_stepEvent); m_stepEvent = nullptr; }
    if (m_computeStream) { cudaStreamDestroy(m_computeStream); m_computeStream = nullptr; }
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
        FEMNodeGPU node{};
        node.x  = mn.position.x;
        node.y  = mn.position.y;
        node.z  = mn.position.z;
node.ox = mn.position.x;
        node.oy = mn.position.y;
        node.oz = mn.position.z;
        // Rough node volume: total mesh volume / numNodes
        // For now estimate from mesh bounding box
        // (will be refined below after all nodes are added)
        double physicalMass = m_material.density * 1e-9;  // placeholder physical mass
        node.mass = physicalMass * m_massScalingFactor;  // scaled mass for mechanics
        node.physicalMass = physicalMass;  // physical mass for thermal
        node.temperature = 25.0;
        node.id          = static_cast<int32_t>(i);
        node.status      = NodeStatus::OK;
        node.isFixed     = false; // Handled dynamically
        node.toolRegion  = static_cast<int32_t>(mn.toolRegion);
        h_nodes.push_back(node);
    }
    
    // Populate tetNodes
    for (size_t i = 0; i < mesh.tetNodes.size(); ++i) {
        const FEMNode& mn = mesh.tetNodes[i];
        FEMNodeGPU node{};
        node.x  = mn.position.x;
        node.y  = mn.position.y;
        node.z  = mn.position.z;
        node.ox = mn.position.x;
        node.oy = mn.position.y;
        node.oz = mn.position.z;
        node.mass = 1e-6; // Base mass, accumulated below in createElementsFromMesh
        node.physicalMass = 1e-6;
        node.temperature = 25.0;
        node.id          = static_cast<int32_t>(i);
        node.status      = NodeStatus::OK;
        node.isFixed     = true; // Will unfix below
        node.toolRegion  = static_cast<int32_t>(mn.toolRegion);
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

    // =========================================================================
    // Fix B: Propagate tet node mass to surface nodes via embed constraints.
    //
    // Surface nodes (h_nodes) are used by the ContactSolver for tool-workpiece
    // interaction. They had placeholder mass = density * 1e-9 * massScaling,
    // which caused F = m*a → a = F/1e-9 → nodes reaching 1e12 m/s².
    //
    // Tet nodes (h_tetNodes) get correct mass from element volumes in
    // createElementsFromMesh(). This loop interpolates tet mass to surface
    // nodes using the barycentric weights in each embed constraint.
    // =========================================================================
    int massFixed = 0;
    for (int i = 0; i < m_numEmbeds; ++i) {
        const auto& ec = h_embedConstraints[i];
        if (ec.surfaceNodeIdx < 0 || ec.surfaceNodeIdx >= m_numNodes) continue;
        
        double totalPhysicalMass = 0.0;
        for (int k = 0; k < 4; ++k) {
            int ti = ec.tetNodeIndices[k];
            if (ti >= 0 && ti < m_numTetNodes) {
                totalPhysicalMass += h_tetNodes[ti].physicalMass * ec.weights[k];
            }
        }
        if (totalPhysicalMass > 1e-15) {
            h_nodes[ec.surfaceNodeIdx].physicalMass = totalPhysicalMass;
            h_nodes[ec.surfaceNodeIdx].mass = totalPhysicalMass * m_massScalingFactor;
            massFixed++;
        }
    }
    // Safety guard: ensure ALL surface nodes have at least a baseline physical mass
    // derived from tributary volume to prevent thermal singularities at the tip.
    double fallbackPhysicalMass = fmax(m_material.density * 1e-9, 1e-6);
    double fallbackMass = fallbackPhysicalMass * m_massScalingFactor;
    for (int i = 0; i < m_numNodes; ++i) {
        if (h_nodes[i].physicalMass < 1e-12) {
            h_nodes[i].mass = fallbackMass;
            h_nodes[i].physicalMass = fallbackPhysicalMass;
            massFixed++;
        }
    }
    std::cout << "[FEMSolver] Fix B: Propagated physical mass to "
              << massFixed << "/" << m_numNodes << " surface nodes" << std::endl;
    // Precompute cutting zone count for distributed cutting force feedback
    double sMinZ = 1e30, sMaxZ = -1e30;
    for (const auto& n : h_nodes) {
        if (n.oz < sMinZ) sMinZ = n.oz;
        if (n.oz > sMaxZ) sMaxZ = n.oz;
    }
    double sLengthZ = sMaxZ - sMinZ;
    m_cachedCuttingZoneCount = 0;
    for (const auto& n : h_nodes) {
        if (!n.isDriven && !n.isFixed && sLengthZ > 1e-12) {
            double zFrac = (n.oz - sMinZ) / sLengthZ;
            if (zFrac <= 0.3) {
                m_cachedCuttingZoneCount++;
            }
        }
    }
    if (m_cachedCuttingZoneCount == 0 && !h_nodes.empty()) {
        m_cachedCuttingZoneCount = (int)h_nodes.size();
    }

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
    for (auto& n : h_tetNodes) {
        n.mass = 0.0;
        n.physicalMass = 0.0;
    }
    
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
        double physicalMassPerNode = (m_material.density * el.volume) / 4.0;
        if (massPerNode < 1e-12) massPerNode = 1e-12;
        if (physicalMassPerNode < 1e-15) physicalMassPerNode = 1e-15;
        
        for (int k = 0; k < 4; ++k) {
            int ni = el.nodeIndices[k];
            h_tetNodes[ni].mass += massPerNode;
            h_tetNodes[ni].physicalMass += physicalMassPerNode;
        }
    }
    m_numElements = static_cast<int>(h_elements.size());

    // Baseline safeguard for any nodes with zero or negligible physical mass
    double minPhysMass = fmax(m_material.density * 1e-12, 1e-10);
    for (auto& n : h_tetNodes) {
        if (n.physicalMass < minPhysMass) {
            n.physicalMass = minPhysMass;
            n.mass = minPhysMass * m_massScalingFactor;
        }
    }
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
        double incrementalAngle = m_toolAngularVelocity * dt;
        m_toolAngle += incrementalAngle;
        applyRotationKernel<<<nodeGrid, blockSize, 0, m_computeStream>>>(
            d_nodes, m_numNodes,
            m_toolPosition.x, m_toolPosition.y, m_toolPosition.z,
            m_toolAxis.x,     m_toolAxis.y,     m_toolAxis.z,
            incrementalAngle);
        CUDA_CHECK_KERNEL();
        
        if (m_numTetNodes > 0) {
            int tetGrid = (m_numTetNodes + blockSize - 1) / blockSize;
            applyRotationKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(
                d_tetNodes, m_numTetNodes,
                m_toolPosition.x, m_toolPosition.y, m_toolPosition.z,
                m_toolAxis.x,     m_toolAxis.y,     m_toolAxis.z,
                incrementalAngle);
            CUDA_CHECK_KERNEL();
        }
    }
    NVTX_POP();

    NVTX_PUSH("FEM::Physics");

    int embedGrid = (m_numEmbeds + blockSize - 1) / blockSize;
    int tetGrid   = (m_numTetNodes + blockSize - 1) / blockSize;

    // =========================================================================
    // KINEMATIC vs DYNAMIC tool physics
    // =========================================================================
    // When spindle dynamics is OFF, the tool is a RIGID KINEMATIC body:
    //   - Position is driven by G-Code translation/rotation (not internal forces)
    //   - Contact forces are computed for REPORTING (Hertz stress, heat) only
    //   - Internal StVK element forces must NOT be computed or integrated
    //   - Node integration must NOT run (would cause stress explosion from
    //     contact forces being applied to 620GPa carbide elements)
    //
    // When spindle dynamics is ON, the tool deflects physically:
    //   - Spindle spring forces + contact forces → element forces → integration
    //   - Full FEM pipeline runs
    // =========================================================================

    if (m_spindleDynamicsEnabled) {
        // --- DYNAMIC MODE: Full FEM physics pipeline ---
        // Sub-cycle mechanics so each integration step stays within the explicit
        // CFL (m_stableTimeStep ≈ 10 ns for carbide). Otherwise a macro dt of
        // 10 µs is 1000× over the stability limit for the leapfrog integrator.
        int mechSub = 1;
        if (dt > m_stableTimeStep && m_stableTimeStep > 1e-15) {
            mechSub = static_cast<int>(std::ceil(dt / m_stableTimeStep));
            if (mechSub > 100000) {
                std::cerr << "[FEMSolver] WARNING: macro dt exceeds FEM CFL by "
                          << mechSub << "×; capping mechanical sub-steps at 100000.\n";
                mechSub = 100000;
            }
        }
        const double dtMech = dt / static_cast<double>(mechSub);
        for (int mech = 0; mech < mechSub; ++mech) {
            // Transfer contact forces from surface to volumetric nodes
            if (m_numEmbeds > 0) {
                transferForcesToTetNodesKernel<<<embedGrid, blockSize, 0, m_computeStream>>>(
                    d_nodes, d_tetNodes, d_embedConstraints, m_numEmbeds);
                CUDA_CHECK_KERNEL();
            }

            // Spindle spring-damper forces
            if (m_numTetNodes > 0) {
                computeSpindleForcesKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(
                    d_tetNodes, m_numTetNodes,
                    m_spindlePos.x, m_spindlePos.y, m_spindlePos.z,
                    m_spindleVel.x, m_spindleVel.y, m_spindleVel.z,
                    m_spindleStiffness, m_spindleDamping,
                    m_toolAngle, m_toolAxis.x, m_toolAxis.y, m_toolAxis.z);
                CUDA_CHECK_KERNEL();
            }

            // Zero stress accumulators before element force computation
            if (m_numTetNodes > 0) {
                resetNodeStressKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(d_tetNodes, m_numTetNodes);
                CUDA_CHECK_KERNEL();
            }

            // Internal elastic forces (StVK corotational) — also accumulates element stress
            if (m_numElements > 0) {
                int elementGrid = (m_numElements + blockSize - 1) / blockSize;
                computeElementForcesKernel<<<elementGrid, blockSize, 0, m_computeStream>>>(
                    d_tetNodes, m_numTetNodes, d_elements, m_numElements,
                    m_material.youngsModulus, m_material.poissonsRatio);
                CUDA_CHECK_KERNEL();
            }

            // Normalize element-averaged stress to nodes and compute von Mises
            if (m_numTetNodes > 0) {
                normalizeNodeStressKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(d_tetNodes, m_numTetNodes);
                CUDA_CHECK_KERNEL();
            }
            
            // ── P3 FIX: Tool Plasticity at Cutting Edge ────────────────────────
            // Radial return mapping for nodes in cutting-edge regions
            if (m_numTetNodes > 0) {
                toolPlasticityKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(
                    d_tetNodes, m_numTetNodes,
                    m_material.youngsModulus, m_material.poissonsRatio,
                    m_material.yieldStrength,  // Initial yield stress
                    m_material.yieldStrength * 0.1);  // Hardening modulus = 10% of yield
                CUDA_CHECK_KERNEL();
            }

            // Explicit time integration (leapfrog)
            if (m_numTetNodes > 0) {
                integrateNodesKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(
                    d_tetNodes, m_numTetNodes, dtMech, m_globalDamping * 0.01);
                CUDA_CHECK_KERNEL();
            }

            // Transfer deformed tet positions back to surface nodes
            if (m_numEmbeds > 0) {
                updateSurfaceNodesKernel<<<embedGrid, blockSize, 0, m_computeStream>>>(
                    d_nodes, d_tetNodes, d_embedConstraints, m_numEmbeds);
                CUDA_CHECK_KERNEL();
            }
        }
    }
    // else: KINEMATIC MODE — skip entire mechanical pipeline.
    // Tool nodes are moved only by translateMesh() / rotateAroundZ() from G-Code.
    // Contact forces and Hertz contact stresses are computed by ContactSolver on surface nodes.

    // Update typical element length for metrics (both modes)
    if (m_currentStep % 100 == 0 && m_numElements > 0) {
        m_typicalElementLength = pow(h_elements.front().volume, 1.0/3.0);
    }
    
    NVTX_POP();

    NVTX_PUSH("FEM::ThermalWear");

    if (m_numEmbeds > 0 && m_numTetNodes > 0) {
        transferSurfaceHeatToTetNodesKernel<<<embedGrid, blockSize, 0, m_computeStream>>>(
            d_nodes, d_tetNodes, d_embedConstraints, m_numEmbeds,
            m_material.specificHeat, 0.85);
        CUDA_CHECK_KERNEL();

        applyTetHeatKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(
            d_tetNodes, m_numTetNodes, m_material.specificHeat);
        CUDA_CHECK_KERNEL();
    }
    
    // Item 7: Thermal sub-cycling through tetrahedral element edges
    // Adaptive CFL-based sub-stepping: α·Δt / h² ≤ 0.25 for 3D explicit diffusion.
    // The conduction kernel accumulates into node.conductionBuffer which is
    // applied by applyTetHeatKernel after each sub-step.
    if (m_numElements > 0 && m_currentStep % m_thermalSkip == 0) {
        double alpha = (m_material.density > 0.0 && m_material.specificHeat > 0.0)
            ? m_material.thermalConductivity / (m_material.density * m_material.specificHeat)
            : 0.0;
        double h = fmax(m_typicalElementLength, 0.0001); // 0.1mm min
        double dt_thermal = dt * m_thermalSkip;
        // CFL: α·Δt / h² ≤ 0.25  =>  N = ceil(α·dt_thermal / (0.25·h²))
        int Nsub = 1;
        if (alpha > 0.0) {
            double cfl = alpha * dt_thermal / (h * h);
            if (cfl > 0.25) Nsub = static_cast<int>(ceil(cfl / 0.25));
        }
        double dt_sub = dt_thermal / Nsub;
        int elementGrid = (m_numElements + blockSize - 1) / blockSize;
        for (int sub = 0; sub < Nsub; ++sub) {
            thermalConductionTetKernel<<<elementGrid, blockSize, 0, m_computeStream>>>(
                d_tetNodes, m_numTetNodes,
                d_elements, m_numElements,
                m_material.thermalConductivity,
                m_material.specificHeat,
                dt_sub);
            CUDA_CHECK_KERNEL();
            int tetGrid = (m_numTetNodes + blockSize - 1) / blockSize;
            applyTetHeatKernel<<<tetGrid, blockSize, 0, m_computeStream>>>(
                d_tetNodes, m_numTetNodes, m_material.specificHeat);
            CUDA_CHECK_KERNEL();
        }
    }

    // Items 3, 4, 8: Thermal BC enforcement — Dirichlet (SHANK), convective, radiative
    // Runs exactly ONCE per step, after all heat accumulation and diffusion.
    if (m_numTetNodes > 0) {
        int tetGridBC = (m_numTetNodes + blockSize - 1) / blockSize;
        applyThermalBCKernel<<<tetGridBC, blockSize, 0, m_computeStream>>>(
            d_tetNodes, m_numTetNodes,
            25.0,                           // ambientTemp
            m_coolantHTC,                   // from CoolantHardeningModel (set via engine)
            5.670374419e-8,                 // stefanBoltzmann
            0.7,                            // emissivity
            m_material.density,             // toolDensity
            m_material.specificHeat,
            dt);
        CUDA_CHECK_KERNEL();
    }

    // F3: The FE updateWearKernel is DISABLED — zone-accurate flank (Archard)
    // and crater (Usui) wear are driven solely by ToolCoatingModel through
    // ContactSolver::applyCoatingWearToMesh. Running this crude A·σ·v·exp(-B/T)
    // loop on top double-counted wear (accumulatedWear ≈ 30 µm while the same
    // value leaked into flank/crater) and let the extrapolation run away to
    // 0.284 m. node.wear stays 0; accumulatedWear = coating wear depth.
    // updateWearKernel(__VA_ARGS__) is retained for reference but not launched.
    NVTX_POP();

    cudaEventRecord(m_stepEvent, m_computeStream);
    m_currentTime += dt;
    m_currentStep++;

    if (m_currentStep % 100 == 0) {
        cudaEventSynchronize(m_stepEvent);
        updateResults();
    }
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
        mn.toolRegion       = static_cast<ToolRegion>(node.toolRegion);
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
        mn.toolRegion       = static_cast<ToolRegion>(node.toolRegion);
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
    
    // Configure surface nodes h_nodes and cache cutting zone count
    double sMinZ = 1e30, sMaxZ = -1e30;
    for (const auto& n : h_nodes) {
        if (n.oz < sMinZ) sMinZ = n.oz;
        if (n.oz > sMaxZ) sMaxZ = n.oz;
    }
    double sLengthZ = sMaxZ - sMinZ;
    double sDrivenThresholdZ = sMaxZ - (sLengthZ * topFraction);

    m_cachedCuttingZoneCount = 0;
    for (auto& n : h_nodes) {
        if (n.oz >= sDrivenThresholdZ) {
            n.isDriven = true;
        } else {
            n.isDriven = false;
        }
        if (!n.isDriven && !n.isFixed && sLengthZ > 1e-12) {
            double zFrac = (n.oz - sMinZ) / sLengthZ;
            if (zFrac <= 0.3) {
                m_cachedCuttingZoneCount++;
            }
        }
    }
    if (m_cachedCuttingZoneCount == 0 && !h_nodes.empty()) {
        m_cachedCuttingZoneCount = (int)h_nodes.size();
    }
    
    copyToDevice();
    std::cout << "[FEMSolver] Anchored Physics: " << drivenCount << " driven nodes configured, "
              << m_cachedCuttingZoneCount << " cutting-zone surface nodes cached."
              << (m_spindleDynamicsEnabled ? " Dynamics ON: tool is deflected." : " Dynamics OFF: tool is rigid kinematic.")
              << std::endl;
}

void FEMSolver::setVirtualSpindleState(const Vec3& pos, const Vec3& vel) {
    m_spindlePos = pos;
    m_spindleVel = vel;
}

void FEMSolver::applyCuttingForces(const Vec3& resultantForce) {
    if (!m_isInitialized || m_numNodes == 0 || !m_spindleDynamicsEnabled) return;

    int blockSize = 256;
    int gridSize  = (m_numNodes + blockSize - 1) / blockSize;

    // 1. Zero surface node force accumulators (discard old contact forces)
    resetFEMNodeForcesKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_nodes, m_numNodes);
    CUDA_CHECK_KERNEL();

    // If resultant force is negligible or no cutting zone nodes exist, skip distribution.
    if (resultantForce.lengthSq() < 1e-24 || m_cachedCuttingZoneCount <= 0) return;

    // 2. Initialize device buffer for Z bounds directly on GPU
    initBoundsZKernel<<<1, 1, 0, m_computeStream>>>(d_cuttingMinMax, d_cuttingMinMax + 1);
    CUDA_CHECK_KERNEL();

    // 3. Find tool Z bounds on device
    findToolBoundsZKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_nodes, m_numNodes, d_cuttingMinMax, d_cuttingMinMax + 1);
    CUDA_CHECK_KERNEL();

    // 4. Apply distributed cutting forces fully asynchronously on GPU (zero PCIe stalls)
    applyCuttingForcesDistributedKernel<<<gridSize, blockSize, 0, m_computeStream>>>(
        d_nodes, m_numNodes,
        resultantForce.x, resultantForce.y, resultantForce.z,
        d_cuttingMinMax, m_cachedCuttingZoneCount);
    CUDA_CHECK_KERNEL();
}

void FEMSolver::setToolTransform(const Vec3& position, const Vec3& axis,
                                  double angularVelocity) {
    m_toolPosition        = position;
    m_toolAxis            = axis.normalized();
    m_toolAngularVelocity = angularVelocity;
}

void FEMSolver::setAllNodeVelocities(double vx, double vy, double vz) {
    copyFromDevice();
    for (auto& n : h_nodes) {
        n.vx = vx; n.vy = vy; n.vz = vz;
    }
    for (auto& n : h_tetNodes) {
        n.vx = vx; n.vy = vy; n.vz = vz;
    }
    copyToDevice();
}

void FEMSolver::setRigidBodyNodeVelocities(const Vec3& linearVelocity,
                                           double angularVelocityZ,
                                           double centerX,
                                           double centerY) {
    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;
    setRigidBodyVelocitiesKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_nodes, m_numNodes,
        linearVelocity.x, linearVelocity.y, linearVelocity.z,
        angularVelocityZ, centerX, centerY);
    CUDA_CHECK_KERNEL();
    int tetBlocks = (m_numTetNodes + threads - 1) / threads;
    if (m_numTetNodes > 0) {
        setRigidBodyVelocitiesKernel<<<tetBlocks, threads, 0, m_computeStream>>>(
            d_tetNodes, m_numTetNodes,
            linearVelocity.x, linearVelocity.y, linearVelocity.z,
            angularVelocityZ, centerX, centerY);
        CUDA_CHECK_KERNEL();
    }
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
        m_results.maxStress          = std::max(m_results.maxStress,      node.vonMisesStress);
        m_results.avgStress         += node.vonMisesStress;
        double dx = node.x - node.ox, dy = node.y - node.oy, dz = node.z - node.oz;
        double disp = std::sqrt(dx*dx + dy*dy + dz*dz);
        m_results.maxDisplacement    = std::max(m_results.maxDisplacement, disp);
        m_results.maxTemperature     = std::max(m_results.maxTemperature,  node.temperature);
        m_results.maxWear            = std::max(m_results.maxWear,         node.wear);
        double v2 = node.vx*node.vx + node.vy*node.vy + node.vz*node.vz;
        m_results.totalKineticEnergy += 0.5 * node.mass * v2;
        if (node.inContact) m_results.numContactNodes++;
    }
    // Volumetric tet node stress is only relevant when spindle dynamics (internal FEM deformation) is enabled.
    if (m_spindleDynamicsEnabled) {
        for (const auto& node : h_tetNodes) {
            m_results.maxStress = std::max(m_results.maxStress, node.vonMisesStress);
        }
    }
    for (const auto& node : h_tetNodes) {
        m_results.maxTemperature = std::max(m_results.maxTemperature, node.temperature);
    }
    if (m_numNodes > 0) m_results.avgStress /= m_numNodes;
}

double FEMSolver::getStableTimeStep() const {
    // When tool is kinematic (all nodes fixed, no spindle dynamics),
    // FEM integration does nothing — don't limit the global timestep.
    if (!m_spindleDynamicsEnabled) {
        return 1e-3; // 1ms — effectively unlimited for SPH-driven dt
    }
    // Mass scaling increases CFL: dt ∝ sqrt(massScalingFactor)
    return m_stableTimeStep * sqrt(fmax(m_massScalingFactor, 1.0));
}

void FEMSolver::setDynamicMassScaling(double factor) {
    m_massScalingFactor = fmax(factor, 1.0);
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
    const_cast<FEMSolver*>(this)->copyFromDevice();
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

void FEMSolver::syncCouplingPoints() {
    if (!m_isInitialized || m_numNodes == 0 || !d_couplingPoints) return;
    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;
    syncFEMCouplingPointsKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_nodes, d_couplingPoints, m_numNodes);
    CUDA_CHECK_KERNEL();
}

std::vector<FEMNodeGPU> FEMSolver::getNodes() { copyFromDevice(); return h_nodes; }

void FEMSolver::translateMesh(double dx, double dy, double dz) {
    if (m_numNodes == 0) { m_toolPosition.x += dx; m_toolPosition.y += dy; m_toolPosition.z += dz; return; }
    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;
    translateMeshKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_nodes, m_numNodes, dx, dy, dz);
    if (m_numTetNodes > 0) {
        int tetBlocks = (m_numTetNodes + threads - 1) / threads;
        translateMeshKernel<<<tetBlocks, threads, 0, m_computeStream>>>(
            d_tetNodes, m_numTetNodes, dx, dy, dz);
    }
    m_toolPosition.x += dx;
    m_toolPosition.y += dy;
    m_toolPosition.z += dz;
}

void FEMSolver::rotateAroundZ(double angle, double centerX, double centerY) {
    if (m_numNodes == 0) { m_toolAngle += angle; return; }
    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;
    rotateAroundZKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_nodes, m_numNodes, angle, centerX, centerY);
    if (m_numTetNodes > 0) {
        int tetBlocks = (m_numTetNodes + threads - 1) / threads;
        rotateAroundZKernel<<<tetBlocks, threads, 0, m_computeStream>>>(
            d_tetNodes, m_numTetNodes, angle, centerX, centerY);
    }
    m_toolAngle += angle;
}

__global__ void computeJetImpingementHTCKernel(
    const FEMNodeGPU* __restrict__ nodes,
    int numNodes,
    double jetOriginX, double jetOriginY, double jetOriginZ,
    double dNozzle,
    double vJet,
    double stagnationFactor,
    double* __restrict__ blockMaxHTC,
    double* __restrict__ blockSumHTC)
{
    extern __shared__ double sdata[];
    double* s_max = sdata;
    double* s_sum = &sdata[blockDim.x];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double myMax = 0.0;
    double mySum = 0.0;

    const double nu = 8.9e-4 / 1050.0; // MU_WATER_25C / 1050.0
    const double Re = vJet * dNozzle / nu;

    if (Re >= 1.0 && idx < numNodes) {
        const double Pr = 6.0;   // PR_WATER_25C
        const double k = 0.606;  // K_WATER_25C
        const double kOverD = (k / dNozzle) * stagnationFactor;
        const double Nu_stag = 0.7 * sqrt(Re) * pow(Pr, 0.33);
        const double htc_stag = Nu_stag * kOverD;
        const double Nu_wall_coeff = 0.5 * pow(Re, 0.6) * pow(Pr, 0.33);

        double dx = nodes[idx].x - jetOriginX;
        double dy = nodes[idx].y - jetOriginY;
        double dz = nodes[idx].z - jetOriginZ;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double rRatio = r / dNozzle;

        double htc = (rRatio < 0.5) ? htc_stag : (Nu_wall_coeff * pow(rRatio, -0.5) * kOverD);
        myMax = htc;
        mySum = htc;
    }

    s_max[tid] = myMax;
    s_sum[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMaxHTC[blockIdx.x] = s_max[0];
        blockSumHTC[blockIdx.x] = s_sum[0];
    }
}

void FEMSolver::computeJetImpingementHTC(
    const Vec3& jetOrigin,
    double dNozzle,
    double vJet,
    double stagnationFactor,
    double& outMaxHTC,
    double& outAvgHTC)
{
    outMaxHTC = 0.0;
    outAvgHTC = 0.0;
    if (!m_isInitialized || m_numNodes <= 0 || !d_nodes || !d_htcBlockMax || !d_htcBlockSum) return;

    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;

    size_t sharedBytes = 2 * threads * sizeof(double);
    computeJetImpingementHTCKernel<<<blocks, threads, sharedBytes, m_computeStream>>>(
        d_nodes, m_numNodes,
        jetOrigin.x, jetOrigin.y, jetOrigin.z,
        dNozzle, vJet, stagnationFactor,
        d_htcBlockMax, d_htcBlockSum);

    std::vector<double> h_blockMax(blocks);
    std::vector<double> h_blockSum(blocks);
    CUDA_CHECK(cudaMemcpyAsync(h_blockMax.data(), d_htcBlockMax, blocks * sizeof(double), cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaMemcpyAsync(h_blockSum.data(), d_htcBlockSum, blocks * sizeof(double), cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaStreamSynchronize(m_computeStream));

    double maxVal = 0.0;
    double sumVal = 0.0;
    for (int b = 0; b < blocks; ++b) {
        if (h_blockMax[b] > maxVal) maxVal = h_blockMax[b];
        sumVal += h_blockSum[b];
    }

    outMaxHTC = maxVal;
    outAvgHTC = (m_numNodes > 0) ? (sumVal / (double)m_numNodes) : 0.0;
}

__global__ void extractWearAndRegionKernel(
    const FEMNodeGPU* __restrict__ nodes,
    int numNodes,
    NodeWearData* __restrict__ outData)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numNodes) return;
    outData[idx].wear = (float)nodes[idx].wear;
    outData[idx].toolRegion = nodes[idx].toolRegion;
}

void FEMSolver::getWearAndRegion(std::vector<NodeWearData>& outData) {
    if (!m_isInitialized || m_numNodes <= 0 || !d_nodes || !d_wearData) {
        outData.clear();
        return;
    }
    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;
    extractWearAndRegionKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_nodes, m_numNodes, d_wearData);

    outData.resize(m_numNodes);
    CUDA_CHECK(cudaMemcpyAsync(outData.data(), d_wearData, m_numNodes * sizeof(NodeWearData),
                               cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaStreamSynchronize(m_computeStream));
}

__global__ void extractContactingNodesKernel(
    const FEMNodeGPU* __restrict__ nodes,
    int numNodes,
    ContactingNodeData* __restrict__ outData,
    int* __restrict__ counter,
    int maxOut)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numNodes) return;
    if (!nodes[idx].inContact) return;

    int slot = atomicAdd(counter, 1);
    if (slot < maxOut) {
        double vx = nodes[idx].vx;
        double vy = nodes[idx].vy;
        double vz = nodes[idx].vz;
        float speed = (float)sqrt(vx * vx + vy * vy + vz * vz);

        outData[slot].nodeIndex = idx;
        outData[slot].vonMisesStress = (float)nodes[idx].vonMisesStress;
        outData[slot].speed = speed;
        outData[slot].temperature = (float)nodes[idx].temperature;
        outData[slot].toolRegion = nodes[idx].toolRegion;
    }
}

void FEMSolver::getContactingNodesData(std::vector<ContactingNodeData>& outContactNodes) {
    outContactNodes.clear();
    if (!m_isInitialized || m_numNodes <= 0 || !d_nodes || !d_contactingNodes || !d_contactingCount) return;

    CUDA_CHECK(cudaMemsetAsync(d_contactingCount, 0, sizeof(int), m_computeStream));

    int threads = 256;
    int blocks = (m_numNodes + threads - 1) / threads;
    int maxOut = m_numNodes;
    extractContactingNodesKernel<<<blocks, threads, 0, m_computeStream>>>(
        d_nodes, m_numNodes, d_contactingNodes, d_contactingCount, maxOut);

    int count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_contactingCount, sizeof(int), cudaMemcpyDeviceToHost, m_computeStream));
    CUDA_CHECK(cudaStreamSynchronize(m_computeStream));

    if (count > maxOut) count = maxOut;
    if (count > 0) {
        outContactNodes.resize(count);
        CUDA_CHECK(cudaMemcpyAsync(outContactNodes.data(), d_contactingNodes,
                                   count * sizeof(ContactingNodeData), cudaMemcpyDeviceToHost, m_computeStream));
        CUDA_CHECK(cudaStreamSynchronize(m_computeStream));
    }
}

} // namespace edgepredict
