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
    if (particle.status == ParticleStatus::INACTIVE) return;

    double px = particle.x, py = particle.y, pz = particle.z;
    double totalFx = 0, totalFy = 0, totalFz = 0;
    double heatGen  = 0;
    bool   hasContact = false;

    for (int j = 0; j < numNodes; ++j) {
        FEMNodeGPU& node = nodes[j];
        if (node.status == NodeStatus::FAILED) continue;

        double dx_pn = px - node.x;
        double dy_pn = py - node.y;
        double dz_pn = pz - node.z;
        double dist  = sqrt(dx_pn * dx_pn + dy_pn * dy_pn + dz_pn * dz_pn);

        if (dist < contactRadius && dist > 1e-12) {
            hasContact = true;

            double penetration = contactRadius - dist;
            double nx = dx_pn / dist;
            double ny = dy_pn / dist;
            double nz = dz_pn / dist;

            // Hertzian contact force
            double fn = contactStiffness * penetration * penetration;

            // Anti-explosion cap (Impulse limit):
            // Maximum force that can push the particle back out of the boundary without 
            // overshooting in a single explicit timestep. F = m * d / dt^2
            double maxFn = (particle.mass * penetration) / (dt * dt);
            if (fn > maxFn) {
                fn = maxFn;
            }

            // Relative velocity
            double relVx = particle.vx - node.vx;
            double relVy = particle.vy - node.vy;
            double relVz = particle.vz - node.vz;

            // Normal component
            double vn = relVx * nx + relVy * ny + relVz * nz;

            // Tangential component
            double vtx = relVx - vn * nx;
            double vty = relVy - vn * ny;
            double vtz = relVz - vn * nz;
            double vt  = sqrt(vtx * vtx + vty * vty + vtz * vtz);

            // Friction
            double ff = friction * fn;
            double maxFriction = (particle.mass * vt) / dt;
            if (ff > maxFriction) ff = maxFriction;

            double ffx = 0, ffy = 0, ffz = 0;
            if (vt > 1e-12) {
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

            // Apply reaction to tool node
            atomicAdd(&node.fx, -fx);
            atomicAdd(&node.fy, -fy);
            atomicAdd(&node.fz, -fz);

            // Frictional heat
            double heat = 0.9 * ff * vt * dt;
            heatGen += heat;

            // Apply heat to tool node
            atomicAdd(&node.temperature, (heat * heatPartition) / (node.mass * 200.0));
            node.inContact = true;
        }
    }

    particle.fx += totalFx;
    particle.fy += totalFy;
    particle.fz += totalFz;

    if (heatGen > 0) {
        particle.temperature += (heatGen * (1.0 - heatPartition)) / (particle.mass * 500.0);
    }

    if (hasContact) {
        atomicAdd(contactCount, 1);
        atomicAdd(totalHeat,   heatGen);
        atomicAdd(totalForce,  sqrt(totalFx * totalFx + totalFy * totalFy + totalFz * totalFz));
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
        std::cout << "[ContactSolver] step " << contactCallCount
                  << ": " << m_numContacts << " contacts"
                  << ", heat=" << m_totalHeatGenerated
                  << ", force=" << m_totalContactForce << std::endl;
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
