#pragma once
/**
 * @file Types.h
 * @brief Core type definitions for EdgePredict Engine v4
 * 
 * IMPORTANT: This is the SINGLE source of truth for all enums and basic types.
 * Do NOT define these types anywhere else to avoid conflicts.
 */

#include <cstdint>
#include <cmath>
#include <vector>

// ============================================================================
// CUDA Compatibility Macros
// ============================================================================
#ifdef __CUDACC__
    #define EP_HOST_DEVICE __host__ __device__
    #define EP_DEVICE __device__
    #define EP_HOST __host__
#else
    #define EP_HOST_DEVICE
    #define EP_DEVICE
    #define EP_HOST
#endif

namespace edgepredict {

// ============================================================================
// Enums (Single Definition - No Duplicates!)
// ============================================================================

/**
 * @brief Status of SPH particles in the simulation
 */
enum class ParticleStatus : int32_t {
    ACTIVE = 0,         // Active workpiece material
    INACTIVE = 1,       // Removed/deleted particle
    CHIP = 2,           // Detached chip particle
    BOUNDARY = 3,       // Boundary/ghost particle
    FIXED_BOUNDARY = 4  // Clamped by virtual chuck/vise — zero velocity, participates in density
};

/**
 * @brief Level of Detail zone for particles
 * Particles far from the tool can use reduced physics updates
 */
enum class LODZone : int32_t {
    ACTIVE = 0,         // Near tool: full physics every step
    ZONE_NEAR = 1,      // Medium distance: reduced update frequency
    ZONE_FAR = 2        // Far from tool: minimal updates
};

/**
 * @brief Status of FEM nodes
 */
enum class NodeStatus : int32_t {
    OK = 0,             // Normal operating condition
    WORN = 1,           // Wear threshold exceeded
    FAILED = 2,         // Catastrophic failure
    FIXED = 3           // Fixed boundary condition
};

/**
 * @brief Type of machining operation
 */
enum class MachiningType {
    MILLING,
    DRILLING,
    REAMING,
    THREADING,
    BORING
};

/**
 * @brief Configuration for Adiabatic Shear Band (ASB) module
 */
struct AdiabaticShearConfig {
    bool enabled;
    double criticalStrainRate;
    double softeningThreshold;
    double taylorQuinneyCoeff;
    double thermalDiffusivity;
    double typicalBandWidth;
    double maxTemperatureRatio;
    double critStrain;
    double tempThreshold;
    
    EP_HOST_DEVICE AdiabaticShearConfig() 
        : enabled(false), criticalStrainRate(1e4), softeningThreshold(0.95),
          taylorQuinneyCoeff(0.9), thermalDiffusivity(2.9e-6), typicalBandWidth(20e-6),
          maxTemperatureRatio(0.9), critStrain(0.8), tempThreshold(600.0) {}
};

/**
 * @brief Wear zones on the cutting tool
 */
enum class WearZone {
    NONE,
    RAKE_FACE,
    FLANK_FACE,
    CUTTING_EDGE,
    CRATER_ZONE,
    CHIP_GROOVE
};


// ============================================================================
// CUDA-Compatible Vector Types
// ============================================================================

/**
 * @brief 3D vector for GPU computation (replaces Eigen in device code)
 */
struct Vec3 {
    double x, y, z;
    
    EP_HOST_DEVICE Vec3() : x(0), y(0), z(0) {}
    EP_HOST_DEVICE Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    EP_HOST_DEVICE Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    EP_HOST_DEVICE Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    EP_HOST_DEVICE Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    EP_HOST_DEVICE Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    
    EP_HOST_DEVICE Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    EP_HOST_DEVICE Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    EP_HOST_DEVICE Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    
    EP_HOST_DEVICE double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    EP_HOST_DEVICE Vec3 cross(const Vec3& o) const { 
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; 
    }
    EP_HOST_DEVICE double lengthSq() const { return x * x + y * y + z * z; }
    EP_HOST_DEVICE double length() const { return sqrt(lengthSq()); }
    EP_HOST_DEVICE Vec3 normalized() const { 
        double len = length();
        return len > 1e-12 ? (*this / len) : Vec3(0, 0, 0);
    }
    
    // Zero vector
    EP_HOST_DEVICE static Vec3 zero() { return {0, 0, 0}; }
};

EP_HOST_DEVICE inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

/**
 * @brief SPH particle (GPU-optimized layout)
 */
struct SPHParticle {
    // Position
    double x, y, z;
    
    // Velocity
    double vx, vy, vz;
    
    // Half-step velocity (for Verlet)
    double vhx, vhy, vhz;
    
    // Force/acceleration
    double fx, fy, fz;
    
    // Properties
    double density;
    double pressure;
    double temperature;
    double mass;
    
    // Stress tensor components (symmetric: 6 components)
    double stress_xx, stress_yy, stress_zz;
    double stress_xy, stress_xz, stress_yz;
    
    // Strain and damage (for chip separation)
    double plasticStrain;       // Accumulated equivalent plastic strain
    double strainRate;          // Current strain rate
    double damage;              // Accumulated damage (0-1, >1 = failed)
    double residualStress;      // Residual stress after deformation
    
    // State
    int32_t id;
    ParticleStatus status;
    int32_t cellHash;       // Spatial hash for neighbor search
    LODZone lodZone;        // Level of Detail zone
    int32_t lastUpdateStep; // Last step when full physics was computed
    
    EP_HOST_DEVICE SPHParticle() 
        : x(0), y(0), z(0),
          vx(0), vy(0), vz(0),
          vhx(0), vhy(0), vhz(0),
          fx(0), fy(0), fz(0),
          density(0), pressure(0), temperature(25.0), mass(0),
          stress_xx(0), stress_yy(0), stress_zz(0),
          stress_xy(0), stress_xz(0), stress_yz(0),
          plasticStrain(0), strainRate(0), damage(0), residualStress(0),
          id(-1), status(ParticleStatus::ACTIVE), cellHash(0),
          lodZone(LODZone::ACTIVE), lastUpdateStep(0) {}
};

// ============================================================================
// FEM Structures
// ============================================================================

/**
 * @brief Spring connecting two FEM nodes
 */
struct FEMSpring {
    int node1, node2;
    double restLength;
    double stiffness;
    double damping;
    
    EP_HOST_DEVICE FEMSpring()
        : node1(-1), node2(-1), restLength(0), stiffness(0), damping(0) {}
};

/**
 * @brief FEM node for GPU-side physics (compact layout)
 */
struct FEMNodeGPU {
    // Position
    double x, y, z;
    double ox, oy, oz;  // Original position
    
    // Velocity
    double vx, vy, vz;
    
    // Force/acceleration
    double fx, fy, fz;
    
    // Properties
    double mass;
    double temperature;
    double vonMisesStress;
    double wear;
    
    // State
    int32_t id;
    NodeStatus status;
    bool isFixed;
    bool inContact;
    
    // Anchored Physics: spindle coupling
    bool isDriven;          // True = connected to virtual spindle via spring
    double localOffX;       // Offset from spindle center (original geometry)
    double localOffY;
    double localOffZ;
    
    EP_HOST_DEVICE FEMNodeGPU()
        : x(0), y(0), z(0), ox(0), oy(0), oz(0),
          vx(0), vy(0), vz(0),
          fx(0), fy(0), fz(0),
          mass(1e-9), temperature(25.0), vonMisesStress(0), wear(0),
          id(-1), status(NodeStatus::OK), isFixed(false), inContact(false),
          isDriven(false), localOffX(0), localOffY(0), localOffZ(0) {}
};

/**
 * @brief FEM node for general tool stress analysis (Host-side)
 */
struct FEMNode {
    Vec3 position;
    Vec3 originalPosition;
    Vec3 velocity;
    Vec3 force;
    
    double mass;
    double temperature;
    double stress;           // Von Mises stress (Pa)
    double accumulatedWear;  // Usui wear depth (m)
    
    NodeStatus status;
    bool isContact;          // Currently in contact with workpiece
    
    EP_HOST_DEVICE FEMNode()
        : position(), originalPosition(), velocity(), force(),
          mass(1e-9), temperature(25.0), stress(0), accumulatedWear(0),
          status(NodeStatus::OK), isContact(false) {}
};

/**
 * @brief Tetrahedral element for FEM
 */
struct FEMElement {
    int32_t nodeIndices[4];  // Tetrahedron has 4 nodes
    double volume;
    double stiffness;
    
    EP_HOST_DEVICE FEMElement() : volume(0), stiffness(0) {
        for (int i = 0; i < 4; ++i) nodeIndices[i] = -1;
    }
};

/**
 * @brief Triangle element for surface representation
 */
struct Triangle {
    int32_t indices[3];
    Vec3 normal;
    
    EP_HOST_DEVICE Triangle() {
        indices[0] = indices[1] = indices[2] = -1;
    }
};

// ============================================================================
// Mesh Structure
// ============================================================================

/**
 * @brief Surface mesh for geometry representation
 */
struct Mesh {
    std::vector<FEMNode> nodes;
    std::vector<Triangle> triangles;
    
    void clear() {
        nodes.clear();
        triangles.clear();
    }
    
    bool empty() const { return nodes.empty(); }
    size_t nodeCount() const { return nodes.size(); }
    size_t triangleCount() const { return triangles.size(); }
};

// ============================================================================
// Machine State (from G-Code)
// ============================================================================

/**
 * @brief Current machine state from G-Code interpreter
 */
struct MachineState {
    Vec3 position;          // Current tool position (m)
    double feedRate;        // Feed rate (m/s)
    double spindleRPM;      // Spindle speed (RPM)
    bool isActive;          // Is the machine currently moving?
    
    MachineState() : position(), feedRate(0), spindleRPM(0), isActive(false) {}
};

// ============================================================================
// Simulation Constants
// ============================================================================

namespace constants {
    constexpr double PI = 3.14159265358979323846;
    constexpr double GRAVITY = 9.81;                    // m/s^2
    constexpr double BOLTZMANN = 1.380649e-23;          // J/K
    constexpr double STEFAN_BOLTZMANN = 5.670374e-8;    // W/(m^2·K^4)
    
    // Default material properties (Ti-6Al-4V)
    constexpr double DEFAULT_DENSITY = 4430.0;          // kg/m^3
    constexpr double DEFAULT_SPECIFIC_HEAT = 526.3;     // J/(kg·K)
    constexpr double DEFAULT_THERMAL_CONDUCTIVITY = 6.7; // W/(m·K)
    constexpr double DEFAULT_MELTING_POINT = 1660.0;    // °C
}

} // namespace edgepredict
