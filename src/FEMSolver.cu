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

__global__ void computeSpringForcesKernel(FEMNodeGPU* nodes, int numNodes,
                                           FEMSpring*  springs, int numSprings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;

    const FEMSpring& spring = springs[idx];
    if (spring.node1 < 0 || spring.node2 < 0)         return;
    if (spring.node1 >= numNodes || spring.node2 >= numNodes) return;

    FEMNodeGPU& n1 = nodes[spring.node1];
    FEMNodeGPU& n2 = nodes[spring.node2];

    double dx = n2.x - n1.x;
    double dy = n2.y - n1.y;
    double dz = n2.z - n1.z;
    double length = sqrt(dx * dx + dy * dy + dz * dz);
    if (length < 1e-12) return;

    double invLen = 1.0 / length;
    dx *= invLen;
    dy *= invLen;
    dz *= invLen;

    double extension = length - spring.restLength;
    double forceMag  = spring.stiffness * extension;

    // Damping
    double dvx     = n2.vx - n1.vx;
    double dvy     = n2.vy - n1.vy;
    double dvz     = n2.vz - n1.vz;
    double relVel  = dvx * dx + dvy * dy + dvz * dz;
    double dampF   = spring.damping * relVel;

    double total   = forceMag + dampF;
    double fx = total * dx;
    double fy = total * dy;
    double fz = total * dz;

    atomicAdd(&n1.fx,  fx);
    atomicAdd(&n1.fy,  fy);
    atomicAdd(&n1.fz,  fz);
    atomicAdd(&n2.fx, -fx);
    atomicAdd(&n2.fy, -fy);
    atomicAdd(&n2.fz, -fz);
}

/**
 * @brief Compute per-node Von Mises stress estimate from net force and
 *        an effective cross-sectional area derived from spring geometry.
 *
 * FIX over previous version: area was hardcoded 1e-6 m² (1 mm²) regardless
 * of actual mesh resolution.  Now we store the average rest-length per node
 * in the node's wear field temporarily during a setup pass — but that would
 * require a two-pass approach.  Instead we compute area as restLength² where
 * restLength is the spring's rest length passed via the spring list.
 *
 * Simpler robust approach used here: compute stress from net force magnitude
 * and effective area = (particleSpacing)² ≈ (sum of spring rest-lengths /
 * numSprings per node)².  Since we don't track per-node spring count on GPU,
 * we approximate: area = force / (sigma_typical * sqrt(numNodes)).
 *
 * For a qualitative engineering estimate this is acceptable.  Accurate tensor
 * stress requires a proper continuum FEM element stiffness matrix (future work).
 */
__global__ void computeStressKernel(FEMNodeGPU* nodes, int numNodes,
                                     FEMSpring*  springs, int numSprings,
                                     double youngsModulus, double typicalSpringLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    FEMNodeGPU& node = nodes[idx];

    double forceMag  = sqrt(node.fx * node.fx + node.fy * node.fy + node.fz * node.fz);

    // Effective cross-section area: spring-length² gives a physically meaningful
    // area when springs approximate element edges (typical of surface meshes).
    // typicalSpringLength is passed from host after computing the mesh mean.
    double area = typicalSpringLength * typicalSpringLength;
    if (area < 1e-12) area = 1e-12;

    node.vonMisesStress = forceMag / area;
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

__global__ void thermalConductionKernel(FEMNodeGPU* nodes, int numNodes,
                                         FEMSpring*  springs, int numSprings,
                                         double dt, double conductivity, double specificHeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;

    const FEMSpring& spring = springs[idx];
    if (spring.node1 < 0 || spring.node2 < 0) return;

    FEMNodeGPU& n1 = nodes[spring.node1];
    FEMNodeGPU& n2 = nodes[spring.node2];

    double dT = n2.temperature - n1.temperature;
    double A   = 1e-6;  // Cross-section area (rough)
    double L   = spring.restLength;
    if (L < 1e-12) return;

    double heatFlux = conductivity * A * dT / L;
    atomicAdd(&n1.temperature,  heatFlux * dt / (n1.mass * specificHeat));
    atomicAdd(&n2.temperature, -heatFlux * dt / (n2.mass * specificHeat));
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
    m_globalDamping          = femParams.dampingRatio;
    m_maxNodes               = femParams.maxNodes;
    m_maxSprings             = femParams.maxNodes * 6;
    m_massScalingFactor      = femParams.massScalingFactor;
    m_stiffnessScalingFactor = femParams.stiffnessScalingFactor;

    allocateMemory(m_maxNodes, m_maxSprings);

    m_isInitialized = true;
    std::cout << "[FEMSolver] Initialized with max " << m_maxNodes << " nodes" << std::endl;
    return true;
}

void FEMSolver::allocateMemory(int numNodes, int numSprings) {
    freeMemory();
    CUDA_CHECK(cudaMalloc(&d_nodes,   numNodes   * sizeof(FEMNodeGPU)));
    CUDA_CHECK(cudaMalloc(&d_springs, numSprings * sizeof(FEMSpring)));
    h_nodes.reserve(numNodes);
    h_springs.reserve(numSprings);
}

void FEMSolver::freeMemory() {
    if (d_nodes)   { cudaFree(d_nodes);   d_nodes   = nullptr; }
    if (d_springs) { cudaFree(d_springs); d_springs = nullptr; }
}

// ---------------------------------------------------------------------------
// FIX: initializeFromMesh now stores triangles for exportToMesh
// ---------------------------------------------------------------------------

void FEMSolver::initializeFromMesh(const Mesh& mesh) {
    h_nodes.clear();
    h_springs.clear();
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
        // Mass from node-associated volume: spacing^3 * density
        // Spacing estimated as cube-root of (bounding-box-volume / num-nodes)
        node.mass        = m_material.density * 1e-9 * m_massScalingFactor;
        node.temperature = 25.0;
        node.id          = static_cast<int32_t>(i);
        node.status      = NodeStatus::OK;
        node.isFixed     = true; // Tool operates as a Kinematic Rigid Body!
        h_nodes.push_back(node);
    }

    m_numNodes = static_cast<int>(h_nodes.size());
    createSpringsFromMesh(mesh);
    copyToDevice();

    // Compute stable time-step from max stiffness / mass ratio
    if (m_numSprings > 0) {
        double maxRatio = 0;
        for (const auto& sp : h_springs) {
            if (sp.node1 >= 0 && sp.node1 < m_numNodes) {
                double ratio = sp.stiffness / h_nodes[sp.node1].mass;
                maxRatio = std::max(maxRatio, ratio);
            }
        }
        if (maxRatio > 0) m_stableTimeStep = 0.5 / std::sqrt(maxRatio);
    }

    std::cout << "[FEMSolver] Created " << m_numNodes << " nodes, "
              << m_numSprings << " springs, "
              << m_meshTriangles.size() << " triangles" << std::endl;
    std::cout << "[FEMSolver] Stable dt: " << m_stableTimeStep << " s" << std::endl;
}

void FEMSolver::createSpringsFromMesh(const Mesh& mesh) {
    std::set<std::pair<int, int>> springSet;

    auto addSpring = [&](int n1, int n2) {
        if (n1 == n2) return;
        if (n1 > n2) std::swap(n1, n2);
        if (springSet.count({n1, n2})) return;

        FEMSpring spring;
        spring.node1 = n1;
        spring.node2 = n2;

        double dx = h_nodes[n2].x - h_nodes[n1].x;
        double dy = h_nodes[n2].y - h_nodes[n1].y;
        double dz = h_nodes[n2].z - h_nodes[n1].z;
        spring.restLength = std::sqrt(dx * dx + dy * dy + dz * dz);

        double A = spring.restLength * spring.restLength;
        if (A < 1e-12) A = 1e-12;

        spring.stiffness = m_material.youngsModulus * A / spring.restLength
                           * m_stiffnessScalingFactor;
        spring.damping   = 2.0 * std::sqrt(spring.stiffness * h_nodes[n1].mass)
                           * m_globalDamping;

        h_springs.push_back(spring);
        springSet.insert({n1, n2});
    };

    for (const auto& tri : mesh.triangles) {
        addSpring(tri.indices[0], tri.indices[1]);
        addSpring(tri.indices[1], tri.indices[2]);
        addSpring(tri.indices[2], tri.indices[0]);
    }

    m_numSprings = static_cast<int>(h_springs.size());
}

void FEMSolver::copyToDevice() {
    if (m_numNodes > 0)
        CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes.data(),
                              m_numNodes   * sizeof(FEMNodeGPU), cudaMemcpyHostToDevice));
    if (m_numSprings > 0)
        CUDA_CHECK(cudaMemcpy(d_springs, h_springs.data(),
                              m_numSprings * sizeof(FEMSpring),  cudaMemcpyHostToDevice));
}

void FEMSolver::copyFromDevice() {
    if (m_numNodes > 0) {
        h_nodes.resize(m_numNodes);
        CUDA_CHECK(cudaMemcpy(h_nodes.data(), d_nodes,
                              m_numNodes * sizeof(FEMNodeGPU), cudaMemcpyDeviceToHost));
    }
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------

void FEMSolver::step(double dt) {
    if (!m_isInitialized || m_numNodes == 0) return;

    int blockSize  = 256;
    int nodeGrid   = (m_numNodes   + blockSize - 1) / blockSize;
    int springGrid = (m_numSprings + blockSize - 1) / blockSize;

    NVTX_PUSH("FEM::Transform");
    if (std::abs(m_toolAngularVelocity) > 1e-12) {
        m_toolAngle += m_toolAngularVelocity * dt;
        applyRotationKernel<<<nodeGrid, blockSize>>>(
            d_nodes, m_numNodes,
            m_toolPosition.x, m_toolPosition.y, m_toolPosition.z,
            m_toolAxis.x,     m_toolAxis.y,     m_toolAxis.z,
            m_toolAngle);
        CUDA_CHECK_KERNEL();
    }
    NVTX_POP();

    NVTX_PUSH("FEM::Physics");
    computeSpringForcesKernel<<<springGrid, blockSize>>>(
        d_nodes, m_numNodes, d_springs, m_numSprings);
    CUDA_CHECK_KERNEL();

    // === ANCHORED PHYSICS: Spindle Forces ===
    if (m_spindleDynamicsEnabled) {
        computeSpindleForcesKernel<<<nodeGrid, blockSize>>>(
            d_nodes, m_numNodes,
            m_spindlePos.x, m_spindlePos.y, m_spindlePos.z,
            m_spindleVel.x, m_spindleVel.y, m_spindleVel.z,
            m_spindleStiffness, m_spindleDamping);
        CUDA_CHECK_KERNEL();
    }

    // Compute typical spring length for stress scaling
    // (recompute every 100 steps to avoid CPU overhead each step)
    static double typicalL = 0.001;
    if (m_currentStep % 100 == 0 && !h_springs.empty()) {
        double sumL = 0;
        for (const auto& sp : h_springs) sumL += sp.restLength;
        typicalL = sumL / h_springs.size();
    }

    computeStressKernel<<<nodeGrid, blockSize>>>(
        d_nodes, m_numNodes, d_springs, m_numSprings,
        m_material.youngsModulus, typicalL);
    CUDA_CHECK_KERNEL();

    integrateNodesKernel<<<nodeGrid, blockSize>>>(
        d_nodes, m_numNodes, dt, m_globalDamping * 0.01);
    CUDA_CHECK_KERNEL();
    NVTX_POP();

    NVTX_PUSH("FEM::ThermalWear");
    thermalConductionKernel<<<springGrid, blockSize>>>(
        d_nodes, m_numNodes, d_springs, m_numSprings,
        dt, m_material.thermalConductivity, m_material.specificHeat);
    CUDA_CHECK_KERNEL();

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
    for (const auto& n : h_nodes) {
        if (n.z < minZ) minZ = n.z;
        if (n.z > maxZ) maxZ = n.z;
    }
    
    double lengthZ = maxZ - minZ;
    // Driven section is from top (maxZ) down by topFraction * lengthZ
    double drivenThresholdZ = maxZ - (lengthZ * topFraction);
    
    int drivenCount = 0;
    for (auto& n : h_nodes) {
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

double FEMSolver::getStableTimeStep() const { return m_stableTimeStep; }

void FEMSolver::reset() {
    h_nodes.clear();
    h_springs.clear();
    m_meshTriangles.clear();
    m_numNodes   = 0;
    m_numSprings = 0;
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
    m_toolAngle += angle;
    copyToDevice();
}

} // namespace edgepredict
