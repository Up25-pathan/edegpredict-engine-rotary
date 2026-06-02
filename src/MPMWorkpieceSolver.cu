#include "MPMWorkpieceSolver.cuh"
#include "Config.h"
#include "CudaUtils.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace edgepredict {

namespace {

__host__ __device__ double clampDevice(double v, double lo, double hi) {
    return fmax(lo, fmin(hi, v));
}

__host__ __device__ double equivalentStress(double xx, double yy, double zz,
                                            double xy, double xz, double yz) {
    double s1 = xx - yy;
    double s2 = yy - zz;
    double s3 = zz - xx;
    return sqrt(0.5 * (s1*s1 + s2*s2 + s3*s3 + 6.0 * (xy*xy + xz*xz + yz*yz)));
}

__host__ __device__ int gridIndex(int i, int j, int k, const MPMGridParams& gp) {
    return (k * gp.ny + j) * gp.nx + i;
}

__host__ __device__ void nodeCoords(int idx, const MPMGridParams& gp, int& i, int& j, int& k) {
    i = idx % gp.nx;
    int t = idx / gp.nx;
    j = t % gp.ny;
    k = t / gp.ny;
}

__host__ __device__ Vec3 nodePosition(int i, int j, int k, const MPMGridParams& gp) {
    return Vec3(gp.minX + i * gp.cellSize,
                gp.minY + j * gp.cellSize,
                gp.minZ + k * gp.cellSize);
}

__device__ double johnsonCookStress(const MPMParticleGPU& p, const MPMMaterialParams& mat) {
    double strainTerm = mat.jcA + mat.jcB * pow(fmax(p.plasticStrain, 0.0), mat.jcN);
    double rateRatio = fmax(p.strainRate / fmax(mat.refStrainRate, 1e-9), 1.0);
    double rateTerm = 1.0 + mat.jcC * log(rateRatio);
    double tStar = clampDevice((p.temperature - mat.refTemp) / fmax(mat.meltTemp - mat.refTemp, 1.0), 0.0, 1.0);
    double thermalTerm = fmax(0.01, 1.0 - pow(tStar, mat.jcM));
    return strainTerm * rateTerm * thermalTerm;
}

__device__ void computeWeights(const MPMParticleGPU& p, const MPMGridParams& gp,
                               int indices[8], double weights[8], Vec3 grads[8]) {
    double rx = (p.x - gp.minX) / gp.cellSize;
    double ry = (p.y - gp.minY) / gp.cellSize;
    double rz = (p.z - gp.minZ) / gp.cellSize;
    int baseX = static_cast<int>(floor(rx));
    int baseY = static_cast<int>(floor(ry));
    int baseZ = static_cast<int>(floor(rz));
    double fx = rx - baseX;
    double fy = ry - baseY;
    double fz = rz - baseZ;
    int n = 0;
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                int i = baseX + dx;
                int j = baseY + dy;
                int k = baseZ + dz;
                if (i < 0 || j < 0 || k < 0 || i >= gp.nx || j >= gp.ny || k >= gp.nz) {
                    indices[n] = -1;
                    weights[n] = 0.0;
                    grads[n] = Vec3::zero();
                    ++n;
                    continue;
                }
                double wx = dx ? fx : 1.0 - fx;
                double wy = dy ? fy : 1.0 - fy;
                double wz = dz ? fz : 1.0 - fz;
                double dwx = dx ? 1.0 / gp.cellSize : -1.0 / gp.cellSize;
                double dwy = dy ? 1.0 / gp.cellSize : -1.0 / gp.cellSize;
                double dwz = dz ? 1.0 / gp.cellSize : -1.0 / gp.cellSize;
                indices[n] = gridIndex(i, j, k, gp);
                weights[n] = wx * wy * wz;
                grads[n] = Vec3(dwx * wy * wz, wx * dwy * wz, wx * wy * dwz);
                ++n;
            }
        }
    }
}

__global__ void clearMPMGridKernel(MPMGridNodeGPU* grid, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    grid[idx] = MPMGridNodeGPU();
}

__global__ void mpmP2GKernel(MPMParticleGPU* particles, int numParticles,
                             MPMGridNodeGPU* grid, MPMGridParams gp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    const MPMParticleGPU& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE) return;

    int ids[8];
    double ws[8];
    Vec3 grads[8];
    computeWeights(p, gp, ids, ws, grads);

    for (int n = 0; n < 8; ++n) {
        int gid = ids[n];
        if (gid < 0 || ws[n] <= 0.0) continue;
        double m = p.mass * ws[n];
        atomicAdd(&grid[gid].mass, m);
        atomicAdd(&grid[gid].vx, p.vx * m);
        atomicAdd(&grid[gid].vy, p.vy * m);
        atomicAdd(&grid[gid].vz, p.vz * m);
        atomicAdd(&grid[gid].temperatureMass, p.temperature * m);

        if (p.status == ParticleStatus::ACTIVE || p.status == ParticleStatus::FIXED_BOUNDARY) {
            Vec3 gx = grads[n];
            double fX = -(p.stress_xx * gx.x + p.stress_xy * gx.y + p.stress_xz * gx.z) * p.volume;
            double fY = -(p.stress_xy * gx.x + p.stress_yy * gx.y + p.stress_yz * gx.z) * p.volume;
            double fZ = -(p.stress_xz * gx.x + p.stress_yz * gx.y + p.stress_zz * gx.z) * p.volume;
            atomicAdd(&grid[gid].fx, fX + p.fx * ws[n]);
            atomicAdd(&grid[gid].fy, fY + p.fy * ws[n]);
            atomicAdd(&grid[gid].fz, fZ + p.fz * ws[n]);
        } else if (p.status == ParticleStatus::CHIP) {
            atomicAdd(&grid[gid].fx, p.fx * ws[n]);
            atomicAdd(&grid[gid].fy, p.fy * ws[n]);
            atomicAdd(&grid[gid].fz, p.fz * ws[n]);
        }
    }
}

__global__ void mpmGridUpdateKernel(MPMGridNodeGPU* grid, int numNodes,
                                    Vec3* toolSamples, int numToolSamples,
                                    MPMGridParams gp, MPMMaterialParams mat,
                                    double fixtureThickness, double dt,
                                    double* totalContactForce) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    MPMGridNodeGPU& node = grid[idx];
    if (node.mass <= 1e-18) return;

    node.vx /= node.mass;
    node.vy /= node.mass;
    node.vz /= node.mass;

    int i, j, k;
    nodeCoords(idx, gp, i, j, k);
    Vec3 pos = nodePosition(i, j, k, gp);
    bool fixed = pos.z <= gp.minZ + fixtureThickness + gp.cellSize;
    if (fixed) {
        node.fixed = 1;
        node.vx = node.vy = node.vz = 0.0;
        return;
    }

    node.vx += (node.fx / node.mass) * dt;
    node.vy += (node.fy / node.mass) * dt;
    node.vz += (node.fz / node.mass) * dt;

    if (toolSamples && numToolSamples > 0) {
        double r2 = mat.toolContactRadius * mat.toolContactRadius;
        Vec3 bestN;
        bool hit = false;
        for (int s = 0; s < numToolSamples; ++s) {
            Vec3 d = pos - toolSamples[s];
            double d2 = d.lengthSq();
            if (d2 < r2 && d2 > 1e-20) {
                r2 = d2;
                bestN = d.normalized();
                hit = true;
            }
        }
        if (hit) {
            double vn = node.vx * bestN.x + node.vy * bestN.y + node.vz * bestN.z;
            if (vn < 0.0) {
                if (totalContactForce) {
                    atomicAdd(totalContactForce, fabs(node.mass * vn / fmax(dt, 1e-15)));
                }
                node.vx -= vn * bestN.x;
                node.vy -= vn * bestN.y;
                node.vz -= vn * bestN.z;
                double heat = 0.5 * node.mass * vn * vn * mat.contactDamping;
                double dT = heat / fmax(node.mass * mat.specificHeat, 1e-12);
                if (mat.thermalClampEnabled) dT = fmin(dT, mat.maxContactTempRise);
                node.temperatureMass += dT * node.mass;
            }
        }
    }
}

__global__ void mpmG2PKernel(MPMParticleGPU* particles, int numParticles,
                             MPMGridNodeGPU* grid, MPMGridParams gp,
                             MPMMaterialParams mat, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    MPMParticleGPU& p = particles[idx];
    if (p.status == ParticleStatus::INACTIVE) return;

    int ids[8];
    double ws[8];
    Vec3 grads[8];
    computeWeights(p, gp, ids, ws, grads);

    Vec3 v;
    double temp = 0.0;
    double wsum = 0.0;
    double L[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

    for (int n = 0; n < 8; ++n) {
        int gid = ids[n];
        if (gid < 0 || ws[n] <= 0.0) continue;
        const MPMGridNodeGPU& node = grid[gid];
        if (node.mass <= 1e-18) continue;
        Vec3 nv(node.vx, node.vy, node.vz);
        v += nv * ws[n];
        temp += (node.temperatureMass / node.mass) * ws[n];
        wsum += ws[n];
        L[0][0] += nv.x * grads[n].x; L[0][1] += nv.x * grads[n].y; L[0][2] += nv.x * grads[n].z;
        L[1][0] += nv.y * grads[n].x; L[1][1] += nv.y * grads[n].y; L[1][2] += nv.y * grads[n].z;
        L[2][0] += nv.z * grads[n].x; L[2][1] += nv.z * grads[n].y; L[2][2] += nv.z * grads[n].z;
    }

    if (p.status == ParticleStatus::FIXED_BOUNDARY) {
        p.vx = p.vy = p.vz = 0.0;
        p.fx = p.fy = p.fz = 0.0;
        return;
    }

    p.vx = v.x; p.vy = v.y; p.vz = v.z;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;
    if (wsum > 1e-8) p.temperature = fmin(fmax(temp / wsum, mat.refTemp), mat.meltTemp);

    if (p.status == ParticleStatus::ACTIVE) {
        double Dxx = L[0][0], Dyy = L[1][1], Dzz = L[2][2];
        double Dxy = 0.5 * (L[0][1] + L[1][0]);
        double Dxz = 0.5 * (L[0][2] + L[2][0]);
        double Dyz = 0.5 * (L[1][2] + L[2][1]);
        double tr = Dxx + Dyy + Dzz;
        double devXX = Dxx - tr / 3.0;
        double devYY = Dyy - tr / 3.0;
        double devZZ = Dzz - tr / 3.0;
        double eqRate = sqrt(2.0 / 3.0 *
            (devXX*devXX + devYY*devYY + devZZ*devZZ + 2.0 * (Dxy*Dxy + Dxz*Dxz + Dyz*Dyz)));
        p.strainRate = eqRate;

        double pressureInc = mat.bulk * tr * dt;
        p.stress_xx += (2.0 * mat.shear * devXX) * dt + pressureInc;
        p.stress_yy += (2.0 * mat.shear * devYY) * dt + pressureInc;
        p.stress_zz += (2.0 * mat.shear * devZZ) * dt + pressureInc;
        p.stress_xy += 2.0 * mat.shear * Dxy * dt;
        p.stress_xz += 2.0 * mat.shear * Dxz * dt;
        p.stress_yz += 2.0 * mat.shear * Dyz * dt;

        double seq = equivalentStress(p.stress_xx, p.stress_yy, p.stress_zz,
                                      p.stress_xy, p.stress_xz, p.stress_yz);
        double flow = johnsonCookStress(p, mat);
        if (seq > flow && seq > 1e-9) {
            double scale = flow / seq;
            p.stress_xx *= scale; p.stress_yy *= scale; p.stress_zz *= scale;
            p.stress_xy *= scale; p.stress_xz *= scale; p.stress_yz *= scale;
            double dp = (seq - flow) / fmax(3.0 * mat.shear, 1.0);
            p.plasticStrain += dp;
            double dT = 0.9 * flow * dp / fmax(mat.density * mat.specificHeat, 1.0);
            if (mat.thermalClampEnabled) dT = fmin(dT, mat.maxContactTempRise);
            p.temperature = fmin(p.temperature + dT, mat.meltTemp);
        }
        if (eqRate > 0.0) {
            p.damage += (eqRate * dt) / fmax(mat.failureStrain, 1e-3);
        }
    }

    if (p.status == ParticleStatus::ACTIVE &&
        (p.damage >= mat.damageThreshold ||
         p.plasticStrain >= mat.criticalPlasticStrain ||
         p.temperature >= mat.meltTemp * mat.thermalChipRatio)) {
        p.status = ParticleStatus::CHIP;
        p.residualStress = equivalentStress(p.stress_xx, p.stress_yy, p.stress_zz,
                                            p.stress_xy, p.stress_xz, p.stress_yz);
        p.stress_xx = p.stress_yy = p.stress_zz = 0.0;
        p.stress_xy = p.stress_xz = p.stress_yz = 0.0;
        p.damage = fmax(p.damage, mat.damageThreshold);
    }

    p.fx = p.fy = p.fz = 0.0;
}

__global__ void applyMPMExternalForceKernel(MPMParticleGPU* particles, int n,
                                            Vec3 center, double radius, Vec3 force) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    MPMParticleGPU& p = particles[idx];
    if (p.status != ParticleStatus::ACTIVE) return;
    Vec3 d(p.x - center.x, p.y - center.y, p.z - center.z);
    if (d.lengthSq() <= radius * radius) {
        p.fx += force.x;
        p.fy += force.y;
        p.fz += force.z;
    }
}

} // namespace

MPMWorkpieceSolver::~MPMWorkpieceSolver() {
    freeMemory();
}

bool MPMWorkpieceSolver::initialize(const Config& config) {
    m_config = &config;
    const auto& material = config.getMaterial();
    const auto& sph = config.getSPH();
    const auto& safety = config.getPhysicsSafety();

    m_material.density = material.density;
    m_material.young = material.youngsModulus;
    m_material.poisson = material.poissonsRatio;
    m_material.shear = m_material.young / (2.0 * (1.0 + m_material.poisson));
    m_material.bulk = m_material.young / (3.0 * (1.0 - 2.0 * m_material.poisson));
    m_material.specificHeat = material.specificHeat;
    m_material.meltTemp = material.meltingPoint;
    m_material.refTemp = config.getMachining().ambientTemperature;
    m_material.jcA = material.jc_A;
    m_material.jcB = material.jc_B;
    m_material.jcN = material.jc_n;
    m_material.jcC = material.jc_C;
    m_material.jcM = material.jc_m;
    m_material.refStrainRate = sph.referenceStrainRate;
    m_material.failureStrain = material.failureStrain;
    m_material.damageThreshold = sph.damageThreshold;
    m_material.criticalPlasticStrain = sph.criticalPlasticStrain;
    m_material.thermalChipRatio = sph.thermalSofteningChipRatio;
    m_material.maxContactTempRise = safety.maxContactTempRisePerStep;
    m_material.thermalClampEnabled = safety.thermalClampEnabled ? 1 : 0;

    m_gridParams.cellSize = sph.smoothingRadius;
    m_particleSpacing = sph.smoothingRadius * sph.particleSpacingFactor;
    m_material.toolContactRadius = sph.smoothingRadius * 1.5;
    m_isInitialized = true;
    CUDA_CHECK(cudaMalloc(&d_metrics, 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_contactForce, sizeof(double)));
    std::cout << "[MPM] CUDA solver initialized, dx="
              << m_gridParams.cellSize * 1000.0 << " mm" << std::endl;
    return true;
}

void MPMWorkpieceSolver::allocateParticles(int capacity) {
    if (capacity <= m_capacity) return;
    if (d_particles) cudaFree(d_particles);
    CUDA_CHECK(cudaMalloc(&d_particles, capacity * sizeof(MPMParticleGPU)));
    m_capacity = capacity;
}

void MPMWorkpieceSolver::freeMemory() {
    if (d_particles) { cudaFree(d_particles); d_particles = nullptr; }
    if (d_grid) { cudaFree(d_grid); d_grid = nullptr; }
    if (d_toolSamples) { cudaFree(d_toolSamples); d_toolSamples = nullptr; }
    if (d_metrics) { cudaFree(d_metrics); d_metrics = nullptr; }
    if (d_contactForce) { cudaFree(d_contactForce); d_contactForce = nullptr; }
    m_capacity = m_numParticles = m_numGridNodes = m_numToolSamples = 0;
}

void MPMWorkpieceSolver::uploadParticles() {
    allocateParticles(static_cast<int>(h_particles.size()));
    m_numParticles = static_cast<int>(h_particles.size());
    if (m_numParticles > 0) {
        CUDA_CHECK(cudaMemcpy(d_particles, h_particles.data(),
                              m_numParticles * sizeof(MPMParticleGPU), cudaMemcpyHostToDevice));
    }
}

void MPMWorkpieceSolver::downloadParticles() {
    if (m_numParticles <= 0) return;
    h_particles.resize(m_numParticles);
    CUDA_CHECK(cudaMemcpy(h_particles.data(), d_particles,
                          m_numParticles * sizeof(MPMParticleGPU), cudaMemcpyDeviceToHost));
}

void MPMWorkpieceSolver::allocateGrid() {
    int n = m_gridParams.nx * m_gridParams.ny * m_gridParams.nz;
    if (n == m_numGridNodes && d_grid) return;
    if (d_grid) cudaFree(d_grid);
    m_numGridNodes = n;
    CUDA_CHECK(cudaMalloc(&d_grid, m_numGridNodes * sizeof(MPMGridNodeGPU)));
}

void MPMWorkpieceSolver::rebuildGridFromHostBounds() {
    if (h_particles.empty()) return;
    Vec3 mn(1e30, 1e30, 1e30), mx(-1e30, -1e30, -1e30);
    for (const auto& p : h_particles) {
        if (p.status == ParticleStatus::INACTIVE) continue;
        mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
        mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
    }
    double pad = m_gridParams.cellSize * 4.0;
    m_gridParams.minX = mn.x - pad;
    m_gridParams.minY = mn.y - pad;
    m_gridParams.minZ = mn.z - pad;
    m_gridParams.nx = std::max(4, static_cast<int>(std::ceil((mx.x - mn.x + 2 * pad) / m_gridParams.cellSize)) + 1);
    m_gridParams.ny = std::max(4, static_cast<int>(std::ceil((mx.y - mn.y + 2 * pad) / m_gridParams.cellSize)) + 1);
    m_gridParams.nz = std::max(4, static_cast<int>(std::ceil((mx.z - mn.z + 2 * pad) / m_gridParams.cellSize)) + 1);
    allocateGrid();
}

void MPMWorkpieceSolver::initializeParticleBox(const Vec3& minBounds, const Vec3& maxBounds, double spacing) {
    h_particles.clear();
    m_particleSpacing = spacing;
    int maxParticles = m_config ? m_config->getSPH().maxParticles : 100000;
    double vol = spacing * spacing * spacing;
    int id = 0;
    for (double z = maxBounds.z; z >= minBounds.z - 1e-12 && static_cast<int>(h_particles.size()) < maxParticles; z -= spacing) {
        for (double y = minBounds.y; y <= maxBounds.y + 1e-12 && static_cast<int>(h_particles.size()) < maxParticles; y += spacing) {
            for (double x = minBounds.x; x <= maxBounds.x + 1e-12 && static_cast<int>(h_particles.size()) < maxParticles; x += spacing) {
                MPMParticleGPU p;
                p.x = x; p.y = y; p.z = z;
                p.mass = m_material.density * vol;
                p.volume = vol;
                p.density = m_material.density;
                p.temperature = m_material.refTemp;
                p.id = id++;
                if (m_config && z <= minBounds.z + m_config->getMachineSetup().fixtureLayerThickness) {
                    p.status = ParticleStatus::FIXED_BOUNDARY;
                }
                h_particles.push_back(p);
            }
        }
    }
    rebuildGridFromHostBounds();
    uploadParticles();
    std::cout << "[MPM] CUDA particles initialized in box: " << m_numParticles << std::endl;
}

void MPMWorkpieceSolver::initializeCylindricalWorkpiece(const Vec3& center, double radius,
                                                        double length, double spacing, int axis) {
    (void)axis;
    h_particles.clear();
    m_particleSpacing = spacing;
    int maxParticles = m_config ? m_config->getSPH().maxParticles : 100000;
    double vol = spacing * spacing * spacing;
    int id = 0;
    double minZ = center.z - length * 0.5;
    double maxZ = center.z + length * 0.5;
    for (double z = maxZ; z >= minZ - 1e-12 && static_cast<int>(h_particles.size()) < maxParticles; z -= spacing) {
        for (double y = center.y - radius; y <= center.y + radius + 1e-12 && static_cast<int>(h_particles.size()) < maxParticles; y += spacing) {
            for (double x = center.x - radius; x <= center.x + radius + 1e-12 && static_cast<int>(h_particles.size()) < maxParticles; x += spacing) {
                double dx = x - center.x, dy = y - center.y;
                if (dx * dx + dy * dy > radius * radius) continue;
                MPMParticleGPU p;
                p.x = x; p.y = y; p.z = z;
                p.mass = m_material.density * vol;
                p.volume = vol;
                p.density = m_material.density;
                p.temperature = m_material.refTemp;
                p.id = id++;
                if (m_config && z <= minZ + m_config->getMachineSetup().fixtureLayerThickness) {
                    p.status = ParticleStatus::FIXED_BOUNDARY;
                }
                h_particles.push_back(p);
            }
        }
    }
    rebuildGridFromHostBounds();
    uploadParticles();
    std::cout << "[MPM] CUDA particles initialized in cylinder: " << m_numParticles << std::endl;
}

void MPMWorkpieceSolver::clearGrid() {
    int block = 256;
    int grid = (m_numGridNodes + block - 1) / block;
    clearMPMGridKernel<<<grid, block>>>(d_grid, m_numGridNodes);
    CUDA_CHECK_KERNEL();
}

void MPMWorkpieceSolver::step(double dt) {
    if (!m_isInitialized || m_numParticles <= 0 || !d_particles || !d_grid) return;
    if (m_currentStep % 25 == 0) {
        downloadParticles();
        rebuildGridFromHostBounds();
        uploadParticles();
    }

    clearGrid();
    int block = 256;
    int pGrid = (m_numParticles + block - 1) / block;
    int nGrid = (m_numGridNodes + block - 1) / block;
    mpmP2GKernel<<<pGrid, block>>>(d_particles, m_numParticles, d_grid, m_gridParams);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemset(d_contactForce, 0, sizeof(double)));
    double fixture = m_config ? m_config->getMachineSetup().fixtureLayerThickness : 0.002;
    mpmGridUpdateKernel<<<nGrid, block>>>(d_grid, m_numGridNodes, d_toolSamples, m_numToolSamples,
                                          m_gridParams, m_material, fixture, dt, d_contactForce);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(&m_totalContactForce, d_contactForce, sizeof(double), cudaMemcpyDeviceToHost));
    mpmG2PKernel<<<pGrid, block>>>(d_particles, m_numParticles, d_grid, m_gridParams, m_material, dt);
    CUDA_CHECK_KERNEL();
    m_currentTime += dt;
    ++m_currentStep;
    if (m_currentStep % 25 == 0) updateMetrics();
}

void MPMWorkpieceSolver::setToolMesh(const Mesh& mesh, double contactRadius) {
    h_toolSamples.clear();
    m_material.toolContactRadius = std::max(contactRadius, m_gridParams.cellSize);
    int stride = std::max(1, static_cast<int>(mesh.nodes.size() / 4000));
    for (size_t i = 0; i < mesh.nodes.size(); i += static_cast<size_t>(stride)) {
        h_toolSamples.push_back(mesh.nodes[i].position);
    }
    if (d_toolSamples) {
        cudaFree(d_toolSamples);
        d_toolSamples = nullptr;
    }
    m_numToolSamples = static_cast<int>(h_toolSamples.size());
    if (m_numToolSamples > 0) {
        CUDA_CHECK(cudaMalloc(&d_toolSamples, m_numToolSamples * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(d_toolSamples, h_toolSamples.data(),
                              m_numToolSamples * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
}

void MPMWorkpieceSolver::applyExternalForce(const Vec3& center, double radius, const Vec3& force) {
    if (!d_particles || m_numParticles <= 0) return;
    int block = 256;
    int grid = (m_numParticles + block - 1) / block;
    applyMPMExternalForceKernel<<<grid, block>>>(d_particles, m_numParticles, center, radius, force);
    CUDA_CHECK_KERNEL();
}

double MPMWorkpieceSolver::getStableTimeStep() const {
    double waveSpeed = std::sqrt(std::max(m_material.young / std::max(m_material.density, 1.0), 1.0));
    double dt = 0.35 * m_gridParams.cellSize / waveSpeed;
    if (m_config) {
        dt = std::max(m_config->getSimulation().minTimeStep,
                      std::min(m_config->getSimulation().maxTimeStep, dt));
    }
    return dt;
}

void MPMWorkpieceSolver::reset() {
    h_particles.clear();
    h_toolSamples.clear();
    m_numParticles = 0;
    m_currentTime = 0.0;
    m_currentStep = 0;
    m_maxStress = 0.0;
    m_maxTemperature = m_material.refTemp;
    m_kineticEnergy = 0.0;
}

void MPMWorkpieceSolver::getBounds(double& minX, double& minY, double& minZ,
                                   double& maxX, double& maxY, double& maxZ) const {
    minX = minY = minZ = 1e30;
    maxX = maxY = maxZ = -1e30;
    for (const auto& p : h_particles) {
        if (p.status == ParticleStatus::INACTIVE) continue;
        minX = std::min(minX, p.x); minY = std::min(minY, p.y); minZ = std::min(minZ, p.z);
        maxX = std::max(maxX, p.x); maxY = std::max(maxY, p.y); maxZ = std::max(maxZ, p.z);
    }
    if (h_particles.empty()) minX = minY = minZ = maxX = maxY = maxZ = 0.0;
}

void MPMWorkpieceSolver::updateMetrics() {
    downloadParticles();
    m_maxStress = 0.0;
    m_maxTemperature = m_material.refTemp;
    m_kineticEnergy = 0.0;
    for (const auto& p : h_particles) {
        if (p.status == ParticleStatus::INACTIVE) continue;
        double s = equivalentStress(p.stress_xx, p.stress_yy, p.stress_zz,
                                    p.stress_xy, p.stress_xz, p.stress_yz);
        m_maxStress = std::max(m_maxStress, s);
        m_maxTemperature = std::max(m_maxTemperature, p.temperature);
        m_kineticEnergy += 0.5 * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
    }
}

std::vector<SPHParticle> MPMWorkpieceSolver::getParticlesForExport() {
    downloadParticles();
    std::vector<SPHParticle> out;
    out.reserve(h_particles.size());
    for (const auto& p : h_particles) {
        SPHParticle sp;
        sp.x = p.x; sp.y = p.y; sp.z = p.z;
        sp.vx = p.vx; sp.vy = p.vy; sp.vz = p.vz;
        sp.fx = p.fx; sp.fy = p.fy; sp.fz = p.fz;
        sp.mass = p.mass;
        sp.density = p.density;
        sp.temperature = p.temperature;
        sp.plasticStrain = p.plasticStrain;
        sp.strainRate = p.strainRate;
        sp.damage = p.damage;
        sp.residualStress = p.residualStress;
        sp.stress_xx = p.stress_xx; sp.stress_yy = p.stress_yy; sp.stress_zz = p.stress_zz;
        sp.stress_xy = p.stress_xy; sp.stress_xz = p.stress_xz; sp.stress_yz = p.stress_yz;
        sp.pressure = -(p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
        sp.id = p.id;
        sp.status = p.status;
        out.push_back(sp);
    }
    return out;
}

void MPMWorkpieceSolver::applyHeatFlux(double x, double y, double z, double heatFlux) {
    downloadParticles();
    Vec3 q(x, y, z);
    double best = 1e30;
    int nearest = -1;
    for (int i = 0; i < m_numParticles; ++i) {
        Vec3 p(h_particles[i].x, h_particles[i].y, h_particles[i].z);
        double d2 = (p - q).lengthSq();
        if (d2 < best) { best = d2; nearest = i; }
    }
    if (nearest >= 0 && best <= m_gridParams.cellSize * m_gridParams.cellSize) {
        auto& p = h_particles[nearest];
        p.temperature = std::min(p.temperature +
            heatFlux / std::max(p.mass * m_material.specificHeat, 1e-12), m_material.meltTemp);
        uploadParticles();
    }
}

double MPMWorkpieceSolver::getTemperatureAt(double x, double y, double z) const {
    Vec3 q(x, y, z);
    double best = 1e30;
    double temp = m_material.refTemp;
    for (const auto& hp : h_particles) {
        Vec3 p(hp.x, hp.y, hp.z);
        double d2 = (p - q).lengthSq();
        if (d2 < best) { best = d2; temp = hp.temperature; }
    }
    return temp;
}

} // namespace edgepredict
