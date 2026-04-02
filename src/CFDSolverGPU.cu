/**
 * @file CFDSolverGPU.cu
 * @brief GPU-accelerated CFD solver implementation
 */

#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>

namespace edgepredict {

// ============================================================================
// CUDA Kernels Implementation
// ============================================================================

__device__ inline int clamp(int val, int minVal, int maxVal) {
    return max(minVal, min(maxVal, val));
}

__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ inline float trilinearInterp(
    const float* field, int nx, int ny, int nz,
    float x, float y, float z) {
    
    // Get cell indices
    int i0 = clamp((int)x, 0, nx - 2);
    int j0 = clamp((int)y, 0, ny - 2);
    int k0 = clamp((int)z, 0, nz - 2);
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;
    
    // Fractional parts
    float fx = x - i0;
    float fy = y - j0;
    float fz = z - k0;
    
    // Indices
    int idx000 = i0 + j0 * nx + k0 * nx * ny;
    int idx100 = i1 + j0 * nx + k0 * nx * ny;
    int idx010 = i0 + j1 * nx + k0 * nx * ny;
    int idx110 = i1 + j1 * nx + k0 * nx * ny;
    int idx001 = i0 + j0 * nx + k1 * nx * ny;
    int idx101 = i1 + j0 * nx + k1 * nx * ny;
    int idx011 = i0 + j1 * nx + k1 * nx * ny;
    int idx111 = i1 + j1 * nx + k1 * nx * ny;
    
    // Trilinear interpolation
    float c00 = lerp(field[idx000], field[idx100], fx);
    float c10 = lerp(field[idx010], field[idx110], fx);
    float c01 = lerp(field[idx001], field[idx101], fx);
    float c11 = lerp(field[idx011], field[idx111], fx);
    
    float c0 = lerp(c00, c10, fy);
    float c1 = lerp(c01, c11, fy);
    
    return lerp(c0, c1, fz);
}

__global__ void advectVelocityKernel(
    const float* u, const float* v, const float* w,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz, float dx, float dt) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    // Semi-Lagrangian: trace back
    float x = i - dt * u[idx] / dx;
    float y = j - dt * v[idx] / dx;
    float z = k - dt * w[idx] / dx;
    
    // Clamp to grid
    x = fmaxf(0.5f, fminf(nx - 1.5f, x));
    y = fmaxf(0.5f, fminf(ny - 1.5f, y));
    z = fmaxf(0.5f, fminf(nz - 1.5f, z));
    
    // Interpolate
    u_out[idx] = trilinearInterp(u, nx, ny, nz, x, y, z);
    v_out[idx] = trilinearInterp(v, nx, ny, nz, x, y, z);
    w_out[idx] = trilinearInterp(w, nx, ny, nz, x, y, z);
}

__global__ void computeDivergenceKernel(
    const float* u, const float* v, const float* w,
    float* div, int nx, int ny, int nz, float dx) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    // Central difference
    float du_dx = (u[idx + 1] - u[idx - 1]) / (2.0f * dx);
    float dv_dy = (v[idx + nx] - v[idx - nx]) / (2.0f * dx);
    float dw_dz = (w[idx + nx*ny] - w[idx - nx*ny]) / (2.0f * dx);
    
    div[idx] = du_dx + dv_dy + dw_dz;
}

__global__ void jacobiPressureKernel(
    const float* p, float* p_new, const float* div,
    const bool* solid, int nx, int ny, int nz, float dx) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i == 0 || i >= nx-1 || j == 0 || j >= ny-1 || k == 0 || k >= nz-1) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    if (solid[idx]) {
        p_new[idx] = 0;
        return;
    }
    
    // Jacobi iteration: p = (sum neighbors - dx² * div) / 6
    float p_sum = p[idx-1] + p[idx+1] + 
                  p[idx-nx] + p[idx+nx] + 
                  p[idx-nx*ny] + p[idx+nx*ny];
    
    p_new[idx] = (p_sum - dx * dx * div[idx]) / 6.0f;
}

__global__ void redBlackGaussSeidelKernel(
    float* p, const float* div, const bool* solid,
    int nx, int ny, int nz, float dx, int color) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i == 0 || i >= nx-1 || j == 0 || j >= ny-1 || k == 0 || k >= nz-1) return;
    
    // Red-black coloring: only process cells matching color
    if ((i + j + k) % 2 != color) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    if (solid[idx]) {
        p[idx] = 0;
        return;
    }
    
    float p_sum = p[idx-1] + p[idx+1] + 
                  p[idx-nx] + p[idx+nx] + 
                  p[idx-nx*ny] + p[idx+nx*ny];
    
    p[idx] = (p_sum - dx * dx * div[idx]) / 6.0f;
}

__global__ void subtractGradientKernel(
    float* u, float* v, float* w, const float* p,
    const bool* solid, int nx, int ny, int nz, float dx, float dt, float rho) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i == 0 || i >= nx-1 || j == 0 || j >= ny-1 || k == 0 || k >= nz-1) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    if (solid[idx]) return;
    
    float scale = dt / (rho * dx);
    
    // Pressure gradient
    float dp_dx = (p[idx+1] - p[idx-1]) / 2.0f;
    float dp_dy = (p[idx+nx] - p[idx-nx]) / 2.0f;
    float dp_dz = (p[idx+nx*ny] - p[idx-nx*ny]) / 2.0f;
    
    u[idx] -= scale * dp_dx;
    v[idx] -= scale * dp_dy;
    w[idx] -= scale * dp_dz;
}

__global__ void advectTemperatureKernel(
    const float* T, float* T_out,
    const float* u, const float* v, const float* w,
    int nx, int ny, int nz, float dx, float dt) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    // Trace back
    float x = i - dt * u[idx] / dx;
    float y = j - dt * v[idx] / dx;
    float z = k - dt * w[idx] / dx;
    
    x = fmaxf(0.5f, fminf(nx - 1.5f, x));
    y = fmaxf(0.5f, fminf(ny - 1.5f, y));
    z = fmaxf(0.5f, fminf(nz - 1.5f, z));
    
    T_out[idx] = trilinearInterp(T, nx, ny, nz, x, y, z);
}

__global__ void diffuseTemperatureKernel(
    const float* T, float* T_out, const float* heatSource,
    int nx, int ny, int nz, float dx, float dt, float alpha) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i == 0 || i >= nx-1 || j == 0 || j >= ny-1 || k == 0 || k >= nz-1) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    // Laplacian
    float lap = (T[idx-1] + T[idx+1] + 
                 T[idx-nx] + T[idx+nx] + 
                 T[idx-nx*ny] + T[idx+nx*ny] - 6.0f * T[idx]) / (dx * dx);
    
    // Heat source
    float source = heatSource ? heatSource[idx] : 0.0f;
    
    // Explicit diffusion
    T_out[idx] = T[idx] + dt * (alpha * lap + source);
}

__global__ void applyBoundaryKernel(
    float* u, float* v, float* w, float* T,
    int nx, int ny, int nz, float inletU, float inletT) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    
    if (idx >= n) return;
    
    int k = idx / (nx * ny);
    int j = (idx - k * nx * ny) / nx;
    int i = idx - k * nx * ny - j * nx;
    
    // Inlet (z = 0)
    if (k == 0) {
        u[idx] = 0;
        v[idx] = 0;
        w[idx] = inletU;
        T[idx] = inletT;
    }
    
    // Outlet (z = nz-1) - zero gradient
    if (k == nz - 1) {
        int prevIdx = idx - nx * ny;
        u[idx] = u[prevIdx];
        v[idx] = v[prevIdx];
        w[idx] = w[prevIdx];
        T[idx] = T[prevIdx];
    }
    
    // Side walls - no slip
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        u[idx] = 0;
        v[idx] = 0;
        w[idx] = 0;
    }
}

__global__ void markSolidFromParticlesKernel(
    bool* solid, const float* particlePos, int numParticles,
    int nx, int ny, int nz, float dx, float originX, float originY, float originZ) {
    
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParticles) return;
    
    // Particle position
    float px = particlePos[pid * 3 + 0];
    float py = particlePos[pid * 3 + 1];
    float pz = particlePos[pid * 3 + 2];
    
    // Convert to grid indices
    int i = (int)((px - originX) / dx);
    int j = (int)((py - originY) / dx);
    int k = (int)((pz - originZ) / dx);
    
    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        solid[idx] = true;
    }
}

// Reduction kernel for max velocity
__global__ void findMaxVelocityKernel(
    const float* u, const float* v, const float* w,
    float* maxVal, int n) {
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float localMax = 0;
    if (i < n) {
        float mag = sqrtf(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
        localMax = mag;
    }
    sdata[tid] = localMax;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)maxVal, __float_as_int(sdata[0]));
    }
}

// ============================================================================
// CFDSolverGPU Implementation
// ============================================================================

// ============================================================================
// CFDSolverGPU Implementation
// ============================================================================

CFDSolverGPU::CFDSolverGPU() = default;

CFDSolverGPU::~CFDSolverGPU() {
    freeMemory();
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
}

bool CFDSolverGPU::initialize(const Config& config) {
    std::cout << "[CFDSolverGPU] Initializing GPU CFD solver..." << std::endl;
    
    const auto& cfd = config.getCFD();
    if (!cfd.enabled) return false;
    
    m_params.nx = cfd.gridX;
    m_params.ny = cfd.gridY;
    m_params.nz = cfd.gridZ;
    m_params.dx = static_cast<float>(cfd.cellSize);
    
    m_params.density = static_cast<float>(cfd.fluidDensity);
    m_params.viscosity = static_cast<float>(cfd.dynamicViscosity);
    m_params.thermalDiff = static_cast<float>(cfd.fluidThermalConductivity / 
                           (cfd.fluidDensity * cfd.fluidSpecificHeat));
    
    m_params.inletVelocity = static_cast<float>(cfd.inletVelocity);
    m_params.inletTemperature = static_cast<float>(cfd.inletTemperature);
    
    m_params.maxPressureIters = 50;
    m_params.pressureTolerance = 1e-5f;
    
    m_totalCells = m_params.nx * m_params.ny * m_params.nz;
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    
    // Allocate memory
    allocateMemory();
    
    std::cout << "[CFDSolverGPU] Grid: " << m_params.nx << "x" << m_params.ny << "x" << m_params.nz
              << " (" << m_totalCells << " cells)" << std::endl;
    
    m_isInitialized = true;
    return true;
}

void CFDSolverGPU::allocateMemory() {
    size_t size = m_totalCells * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_u, size));
    CUDA_CHECK(cudaMalloc(&d_v, size));
    CUDA_CHECK(cudaMalloc(&d_w, size));
    CUDA_CHECK(cudaMalloc(&d_u_temp, size));
    CUDA_CHECK(cudaMalloc(&d_v_temp, size));
    CUDA_CHECK(cudaMalloc(&d_w_temp, size));
    CUDA_CHECK(cudaMalloc(&d_p, size));
    CUDA_CHECK(cudaMalloc(&d_divergence, size));
    CUDA_CHECK(cudaMalloc(&d_T, size));
    CUDA_CHECK(cudaMalloc(&d_T_temp, size));
    CUDA_CHECK(cudaMalloc(&d_solid, m_totalCells * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_heatSource, size));
    
    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_u, 0, size));
    CUDA_CHECK(cudaMemset(d_v, 0, size));
    CUDA_CHECK(cudaMemset(d_w, 0, size));
    CUDA_CHECK(cudaMemset(d_p, 0, size));
    CUDA_CHECK(cudaMemset(d_T, 0, size));
    CUDA_CHECK(cudaMemset(d_solid, 0, m_totalCells * sizeof(bool)));
}

void CFDSolverGPU::freeMemory() {
    if (d_u) cudaFree(d_u);
    if (d_v) cudaFree(d_v);
    if (d_w) cudaFree(d_w);
    if (d_u_temp) cudaFree(d_u_temp);
    if (d_v_temp) cudaFree(d_v_temp);
    if (d_w_temp) cudaFree(d_w_temp);
    if (d_p) cudaFree(d_p);
    if (d_divergence) cudaFree(d_divergence);
    if (d_T) cudaFree(d_T);
    if (d_T_temp) cudaFree(d_T_temp);
    if (d_solid) cudaFree(d_solid);
    if (d_heatSource) cudaFree(d_heatSource);
}

void CFDSolverGPU::step(double dt) {
    if (!m_isInitialized) return;
    
    float fdt = (float)dt;
    m_params.dt = fdt;
    
    // CFD pipeline (all on GPU)
    applyBoundaryConditions();
    advectVelocity(fdt);
    addForces(fdt);
    computeDivergence();
    solvePressure();
    subtractPressureGradient();
    advectTemperature(fdt);
    diffuseTemperature(fdt);
    
    m_currentTime += dt;
    m_currentStep++;
    
    // Metrics update every 10 steps for performance
    if (m_currentStep % 10 == 0) {
        updateMetrics();
    }
}

double CFDSolverGPU::getStableTimeStep() const {
    // CFL condition: dt < dx / max_velocity
    if (m_maxVelocity < 1e-6f) return 0.01;
    return (double)(0.5f * m_params.dx / m_maxVelocity);
}

void CFDSolverGPU::getBounds(double& minX, double& minY, double& minZ, 
                             double& maxX, double& maxY, double& maxZ) const {
    minX = minY = minZ = 0.0;
    maxX = (double)(m_params.nx * m_params.dx);
    maxY = (double)(m_params.ny * m_params.dx);
    maxZ = (double)(m_params.nz * m_params.dx);
}

void CFDSolverGPU::reset() {
    if (!m_isInitialized) return;
    size_t size = m_totalCells * sizeof(float);
    cudaMemset(d_u, 0, size);
    cudaMemset(d_v, 0, size);
    cudaMemset(d_w, 0, size);
    cudaMemset(d_p, 0, size);
    cudaMemset(d_T, 0, size);
    cudaMemset(d_solid, 0, m_totalCells * sizeof(bool));
    m_currentTime = 0.0;
    m_currentStep = 0;
}

double CFDSolverGPU::getTotalKineticEnergy() const {
    return 0.5 * m_params.density * m_maxVelocity * m_maxVelocity * (m_totalCells * std::pow(m_params.dx, 3));
}

void CFDSolverGPU::syncMetrics() {
    updateMetrics();
    cudaStreamSynchronize(m_stream);
}

void CFDSolverGPU::applyBoundaryConditions() {
    int n = m_totalCells;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    applyBoundaryKernel<<<blocks, threads, 0, m_stream>>>(
        d_u, d_v, d_w, d_T, m_params.nx, m_params.ny, m_params.nz,
        m_params.inletVelocity, m_params.inletTemperature);
}

void CFDSolverGPU::advectVelocity(float dt) {
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (m_params.nx + threads.x - 1) / threads.x,
        (m_params.ny + threads.y - 1) / threads.y,
        (m_params.nz + threads.z - 1) / threads.z
    );
    
    advectVelocityKernel<<<blocks, threads, 0, m_stream>>>(
        d_u, d_v, d_w, d_u_temp, d_v_temp, d_w_temp,
        m_params.nx, m_params.ny, m_params.nz, m_params.dx, dt);
    
    // Swap buffers
    std::swap(d_u, d_u_temp);
    std::swap(d_v, d_v_temp);
    std::swap(d_w, d_w_temp);
}

void CFDSolverGPU::addForces(float dt) {
    // Body forces (gravity etc) could be added here if needed
}

void CFDSolverGPU::computeDivergence() {
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (m_params.nx + threads.x - 1) / threads.x,
        (m_params.ny + threads.y - 1) / threads.y,
        (m_params.nz + threads.z - 1) / threads.z
    );
    
    computeDivergenceKernel<<<blocks, threads, 0, m_stream>>>(
        d_u, d_v, d_w, d_divergence,
        m_params.nx, m_params.ny, m_params.nz, m_params.dx);
}

void CFDSolverGPU::solvePressure() {
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (m_params.nx + threads.x - 1) / threads.x,
        (m_params.ny + threads.y - 1) / threads.y,
        (m_params.nz + threads.z - 1) / threads.z
    );
    
    for (int iter = 0; iter < m_params.maxPressureIters; ++iter) {
        redBlackGaussSeidelKernel<<<blocks, threads, 0, m_stream>>>(
            d_p, d_divergence, d_solid,
            m_params.nx, m_params.ny, m_params.nz, m_params.dx, 0);
        
        redBlackGaussSeidelKernel<<<blocks, threads, 0, m_stream>>>(
            d_p, d_divergence, d_solid,
            m_params.nx, m_params.ny, m_params.nz, m_params.dx, 1);
    }
}

void CFDSolverGPU::subtractPressureGradient() {
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (m_params.nx + threads.x - 1) / threads.x,
        (m_params.ny + threads.y - 1) / threads.y,
        (m_params.nz + threads.z - 1) / threads.z
    );
    
    subtractGradientKernel<<<blocks, threads, 0, m_stream>>>(
        d_u, d_v, d_w, d_p, d_solid,
        m_params.nx, m_params.ny, m_params.nz,
        m_params.dx, m_params.dt, m_params.density);
}

void CFDSolverGPU::advectTemperature(float dt) {
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (m_params.nx + threads.x - 1) / threads.x,
        (m_params.ny + threads.y - 1) / threads.y,
        (m_params.nz + threads.z - 1) / threads.z
    );
    
    advectTemperatureKernel<<<blocks, threads, 0, m_stream>>>(
        d_T, d_T_temp, d_u, d_v, d_w,
        m_params.nx, m_params.ny, m_params.nz, m_params.dx, dt);
    
    std::swap(d_T, d_T_temp);
}

void CFDSolverGPU::diffuseTemperature(float dt) {
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (m_params.nx + threads.x - 1) / threads.x,
        (m_params.ny + threads.y - 1) / threads.y,
        (m_params.nz + threads.z - 1) / threads.z
    );
    
    diffuseTemperatureKernel<<<blocks, threads, 0, m_stream>>>(
        d_T, d_T_temp, d_heatSource,
        m_params.nx, m_params.ny, m_params.nz,
        m_params.dx, dt, m_params.thermalDiff);
    
    std::swap(d_T, d_T_temp);
}

void CFDSolverGPU::updateMetrics() {
    int n = m_totalCells;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    float* d_maxVel;
    cudaMalloc(&d_maxVel, sizeof(float));
    cudaMemset(d_maxVel, 0, sizeof(float));
    
    findMaxVelocityKernel<<<blocks, threads, threads * sizeof(float), m_stream>>>(
        d_u, d_v, d_w, d_maxVel, n);
    
    cudaMemcpy(&m_maxVelocity, d_maxVel, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_maxVel);
}

void CFDSolverGPU::setSolidObstacles(const double* particlePositions, int numParticles) {
    if (!m_isInitialized || !particlePositions || numParticles == 0) return;
    
    cudaMemset(d_solid, 0, m_totalCells * sizeof(bool));
    
    std::vector<float> fPos(numParticles * 3);
    for (int i = 0; i < numParticles * 3; ++i) fPos[i] = (float)particlePositions[i];
    
    float* d_particlePos;
    cudaMalloc(&d_particlePos, numParticles * 3 * sizeof(float));
    cudaMemcpy(d_particlePos, fPos.data(), numParticles * 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (numParticles + threads - 1) / threads;
    markSolidFromParticlesKernel<<<blocks, threads, 0, m_stream>>>(
        d_solid, d_particlePos, numParticles,
        m_params.nx, m_params.ny, m_params.nz, m_params.dx,
        0, 0, 0);
    
    cudaFree(d_particlePos);
}

void CFDSolverGPU::setHeatSources(const double* nodeTemperatures, int numNodes) {
    if (!m_isInitialized || !nodeTemperatures) return;
    
    std::vector<float> fTemp(numNodes);
    for (int i = 0; i < numNodes; ++i) fTemp[i] = (float)nodeTemperatures[i];
    
    cudaMemcpy(d_heatSource, fTemp.data(), 
               std::min((int)numNodes, m_totalCells) * sizeof(float), 
               cudaMemcpyHostToDevice);
}

Vec3 CFDSolverGPU::getVelocityAt(const Vec3& pos) const {
    std::vector<float> u_host(m_totalCells), v_host(m_totalCells), w_host(m_totalCells);
    cudaMemcpy(u_host.data(), d_u, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_host.data(), d_v, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_host.data(), d_w, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    
    int i = std::clamp(static_cast<int>(pos.x / m_params.dx), 0, m_params.nx - 1);
    int j = std::clamp(static_cast<int>(pos.y / m_params.dx), 0, m_params.ny - 1);
    int k = std::clamp(static_cast<int>(pos.z / m_params.dx), 0, m_params.nz - 1);
    int idx = i + j * m_params.nx + k * m_params.nx * m_params.ny;
    
    return Vec3(u_host[idx], v_host[idx], w_host[idx]);
}

double CFDSolverGPU::getTemperatureAt(const Vec3& pos) const {
    std::vector<float> T_host(m_totalCells);
    cudaMemcpy(T_host.data(), d_T, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    
    int i = std::clamp(static_cast<int>(pos.x / m_params.dx), 0, m_params.nx - 1);
    int j = std::clamp(static_cast<int>(pos.y / m_params.dx), 0, m_params.ny - 1);
    int k = std::clamp(static_cast<int>(pos.z / m_params.dx), 0, m_params.nz - 1);
    int idx = i + j * m_params.nx + k * m_params.nx * m_params.ny;
    
    return (double)T_host[idx];
}

void CFDSolverGPU::copyVelocityToHost(std::vector<Vec3>& velocities) {
    velocities.resize(m_totalCells);
    std::vector<float> u_host(m_totalCells), v_host(m_totalCells), w_host(m_totalCells);
    cudaMemcpy(u_host.data(), d_u, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_host.data(), d_v, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_host.data(), d_w, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m_totalCells; ++i) velocities[i] = Vec3(u_host[i], v_host[i], w_host[i]);
}

void CFDSolverGPU::copyTemperatureToHost(std::vector<float>& temperatures) {
    temperatures.resize(m_totalCells);
    cudaMemcpy(temperatures.data(), d_T, m_totalCells * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace edgepredict
