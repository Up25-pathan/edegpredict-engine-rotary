/**
 * @file ContactSolver.cu
 * @brief Tool-workpiece contact detection and resolution
 */

#include "ContactSolver.cuh"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ToolCoatingModel.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace edgepredict {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Brute-force O(N×M) contact kernel (standard version)
 */
__global__ void bruteForceContactKernel(
    SPHParticle* particles, int numParticles,
    FEMNodeGPU*  nodes,     int numNodes,
    double contactRadius, double contactStiffness,
    double friction,      double heatPartition, double dt,
    int* contactCount, double* totalHeat, double* totalForce
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    SPHParticle& particle = particles[idx];
    if (particle.status == ParticleStatus::INACTIVE ||
        particle.status == ParticleStatus::FIXED_BOUNDARY) return;

    double totalFx = 0, totalFy = 0, totalFz = 0;
    double heatGen  = 0;
    bool   hasContact = false;
    // Accumulate total raw heat destined for this particle across ALL node contacts.
    // Temperature is applied ONCE after the loop — NOT per node — to prevent
    // dense drill-tip nodes from multiplying ΔT by N_nodes × 200°C.
    double totalHeatToParticle = 0.0;

    for (int j = 0; j < numNodes; ++j) {
        FEMNodeGPU& node = nodes[j];
        if (node.status == NodeStatus::FAILED) continue;

        double dx = particle.x - node.x;
        double dy = particle.y - node.y;
        double dz = particle.z - node.z;
        double dist = sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < contactRadius && dist > 1e-12) {
            hasContact = true;

            double penetration = contactRadius - dist;
            double nx = dx / dist;
            double ny = dy / dist;
            double nz = dz / dist;

            // ─────────────────────────────────────────────────────────────
            // FIX: LINEAR penalty instead of Hertz (fn = k * d not k * d²)
            //
            // Choose k so that the natural frequency of the contact spring
            // is much higher than the simulation frequency but the force
            // remains within impulse bounds:
            //   k_safe = m_particle / (dt^2 * safety_factor)
            //
            // contactStiffness is now passed as a "material stiffness" hint
            // (E * r ~ 200e9 * 1e-4 = 2e7 N/m), but we always cap per
            // the impulse limit so there is no explosion.
            // ─────────────────────────────────────────────────────────────
            double fn = contactStiffness * penetration;  // LINEAR

            // Impulse cap: max force that resolves penetration in one step
            // without overshoot.  F_max = m * penetration / dt^2  (old Hertz cap)
            // Better impulse cap: F_max = m * (contactRadius / dt) / dt
            // = m * contactRadius / dt^2
            double maxFn = (particle.mass * contactRadius) / (dt * dt + 1e-30);
            if (fn > maxFn) fn = maxFn;

            // Relative velocity
            double relVx = particle.vx - node.vx;
            double relVy = particle.vy - node.vy;
            double relVz = particle.vz - node.vz;
            double vn    = relVx * nx + relVy * ny + relVz * nz;

            // Velocity damping in normal direction (prevent bouncing)
            // Restitution coefficient ~ 0.1 for metal-on-metal impact
            if (vn < 0) {
                double dampFn = -0.3 * particle.mass * vn / (dt + 1e-30);
                fn = fmax(fn + dampFn, 0.0);
            }

            // Tangential (friction)
            double vtx = relVx - vn * nx;
            double vty = relVy - vn * ny;
            double vtz = relVz - vn * nz;
            double vt  = sqrt(vtx * vtx + vty * vty + vtz * vtz);

            // Temperature-dependent friction (Zorev model)
            double T_contact = 0.5 * (particle.temperature + node.temperature);
            double T_ratio   = fmin(1.0, fmax(0.0, (T_contact - 200.0) / 600.0));
            double mu_eff    = friction * (1.0 - 0.4 * T_ratio);

            double ff = mu_eff * fn;
            // Stiction: clamp friction to what's needed to stop sliding this step
            double ffMax = (particle.mass * vt) / (dt + 1e-30);
            if (ff > ffMax) ff = ffMax;

            double ffx = 0, ffy = 0, ffz = 0;
            if (vt > 1e-10) {
                ffx = -ff * vtx / vt;
                ffy = -ff * vty / vt;
                ffz = -ff * vtz / vt;
            }

            double fx = fn * nx + ffx;
            double fy = fn * ny + ffy;
            double fz = fn * nz + ffz;

            totalFx += fx;
            totalFy += fy;
            totalFz += fz;

            // Reaction on tool node
            atomicAdd(&node.fx, -fx);
            atomicAdd(&node.fy, -fy);
            atomicAdd(&node.fz, -fz);

            // Frictional heat generation — accumulate raw heat for this contact pair.
            // Temperature applied AFTER the node loop (see below).
            double heatRate     = ff * vt;          // W per contact pair
            double heatThisStep = heatRate * dt;
            heatGen += heatThisStep;

            // Tool node temperature rise (2°C/step cap per contact pair)
            double toolCp     = 200.0;  // J/(kg·K) carbide
            double toolDeltaT = (heatThisStep * heatPartition) /
                                 (fmax(node.mass, 1e-8) * toolCp);
            toolDeltaT = fmin(toolDeltaT, 2.0);   // clamp 2°C/step per node contact
            atomicAdd(&node.temperature, toolDeltaT);
            node.inContact = true;

            // Accumulate heat fraction going to the workpiece particle
            totalHeatToParticle += heatThisStep * (1.0 - heatPartition);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Workpiece particle temperature update — applied ONCE after all node
    // contacts have been processed.  This prevents dense FEM meshes from
    // multiplying the clamp by N_contacts.
    //
    //   workCp = 475 J/(kg·K)  — AISI 4140 Steel (NOT Ti-6Al-4V)
    //   max ΔT = 5°C/step total (physically: heat spreads over bulk)
    //   temp cap = 1460°C      — Steel 4140 melting point
    // ─────────────────────────────────────────────────────────────────────
    if (totalHeatToParticle > 0.0) {
        const double workCp   = 475.0;   // J/(kg·K) Steel 4140
        double workDeltaT = totalHeatToParticle /
                            (fmax(particle.mass, 1e-15) * workCp);
        workDeltaT = fmin(workDeltaT, 5.0);   // clamp total ΔT to 5°C/step
        particle.temperature = fmin(particle.temperature + workDeltaT, 1460.0);
    }

    // Accumulate external forces (not overwrite)
    particle.ext_fx += totalFx;
    particle.ext_fy += totalFy;
    particle.ext_fz += totalFz;

    if (hasContact) {
        atomicAdd(contactCount, 1);
        atomicAdd(totalHeat,    heatGen);
        double fmag = sqrt(totalFx * totalFx + totalFy * totalFy + totalFz * totalFz);
        atomicAdd(totalForce,   fmag);
    }
}

// ============================================================================
// ContactSolver Implementation
// ============================================================================

ContactSolver::ContactSolver() = default;

ContactSolver::~ContactSolver() {
    if (d_toolCellStart) cudaFree(d_toolCellStart);
    if (d_toolCellEnd) cudaFree(d_toolCellEnd);
    if (d_numContacts) cudaFree(d_numContacts);
    if (d_totalHeat) cudaFree(d_totalHeat);
    if (d_totalForce) cudaFree(d_totalForce);
}

void ContactSolver::initialize(SPHSolver* sph, FEMSolver* fem, const ContactConfig& config) {
    m_sph = sph;
    m_fem = fem;
    m_config = config;
    
    // Scalar results
    CUDA_CHECK(cudaMalloc(&d_numContacts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_totalHeat,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_totalForce,  sizeof(double)));
    
    m_isInitialized = true;
    std::cout << "[ContactSolver] Initialized (Clean Arch)" << std::endl;
}

void ContactSolver::resolveContacts(double dt) {
    if (!m_isInitialized || !m_sph || !m_fem) return;
    
    int numP = m_sph->getParticleCount();
    int numN = m_fem->getNodeCount();
    if (numP <= 0 || numN <= 0) return;
    
    CUDA_CHECK(cudaMemset(d_numContacts, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_totalHeat, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_totalForce, 0, sizeof(double)));
    
    int blockSize = 256;
    int gridSize = (numP + blockSize - 1) / blockSize;
    
    bruteForceContactKernel<<<gridSize, blockSize>>>(
        m_sph->getDeviceParticles(), numP,
        m_fem->getDeviceNodes(), numN,
        m_config.contactRadius, m_config.contactStiffness,
        m_config.frictionCoefficient, m_config.heatPartition, dt,
        d_numContacts, d_totalHeat, d_totalForce
    );
    CUDA_CHECK_KERNEL();
    
    transferResults();
    
    // Diagnostic logging (every 100th call)
    static int contactCallCount = 0;
    if (++contactCallCount % 100 == 1) {
        std::cout << std::fixed
                  << "[ContactSolver] call#" << contactCallCount
                  << ": contacts=" << m_numContacts
                  << std::scientific << std::setprecision(3)
                  << ", heat=" << m_totalHeatGenerated << " J"
                  << ", force=" << m_totalContactForce << " N"
                  << std::defaultfloat << std::endl;
    }
    
    // === Tool Coating Wear Update ===
    // Now that contact forces and temperatures are resolved on GPU,
    // run the coating wear model on host for each node in contact.
    if (m_toolCoatingModel && m_numContacts > 0 && m_fem) {
        auto toolNodes = m_fem->getNodes();
        int numNodes = static_cast<int>(toolNodes.size());
        
        for (int j = 0; j < numNodes; ++j) {
            if (!toolNodes[j].inContact) continue;
            
            // Contact pressure estimated from von Mises stress at contact
            double contactPressure = toolNodes[j].vonMisesStress;
            
            // Sliding velocity from relative node velocity
            double slidingVelocity = sqrt(
                toolNodes[j].vx * toolNodes[j].vx + 
                toolNodes[j].vy * toolNodes[j].vy + 
                toolNodes[j].vz * toolNodes[j].vz);
            
            double temperature = toolNodes[j].temperature;
            
            // Determine wear zone from node position relative to tool geometry
            // Nodes near the tip = cutting edge / flank face
            // Nodes further up = rake face / crater zone
            WearZone zone = WearZone::FLANK_FACE;
            
            // Heuristic: nodes with high tangential velocity are on the rake face
            // nodes with high normal contact are on the flank face
            if (toolNodes[j].vonMisesStress > 1e9) {
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
    CUDA_CHECK(cudaMemcpy(&m_numContacts, d_numContacts, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m_totalHeatGenerated, d_totalHeat, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m_totalContactForce, d_totalForce, sizeof(double), cudaMemcpyDeviceToHost));
}

std::vector<ContactEvent> ContactSolver::getContactEvents() const {
    return std::vector<ContactEvent>();
}

// ============================================================================
// Free function
// ============================================================================

void launchContactInteraction(
    SPHParticle* particles, int numParticles,
    FEMNodeGPU* nodes, int numNodes,
    double contactRadius, double contactStiffness,
    double friction, double heatPartition, double dt
) {
    static int* d_cnt = nullptr;
    static double* d_h = nullptr;
    static double* d_f = nullptr;
    static bool init = false;
    
    if (!init) {
        CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_h, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_f, sizeof(double)));
        init = true;
    }
    
    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_h, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_f, 0, sizeof(double)));
    
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    bruteForceContactKernel<<<gridSize, blockSize>>>(
        particles, numParticles,
        nodes, numNodes,
        contactRadius, contactStiffness,
        friction, heatPartition, dt,
        d_cnt, d_h, d_f
    );
}

} // namespace edgepredict
