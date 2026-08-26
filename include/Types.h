#pragma once
/**
 * @file Types.h
 * @brief Core type definitions for EdgePredict Engine v4
 * 
 * IMPORTANT: This is the SINGLE source of truth for all enums and basic types.
 * Do NOT define these types anywhere else to avoid conflicts.
 */

#include <cstdint>
#include <cstddef>
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
 * @brief Status of MPM particles in the simulation
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

enum class ToolRegion : int32_t {
    UNKNOWN = 0,
    TOOL_BODY = 1,
    CUTTING_EDGE = 2,
    CHISEL_EDGE = 3,
    RAKE_FACE = 4,
    FLANK_FACE = 5,
    MARGIN = 6,
    FLUTE = 7,
    SHANK = 8,

    // Milling/end-mill family
    END_CUTTING_EDGE = 9,
    PERIPHERAL_CUTTING_EDGE = 10,
    BALL_NOSE = 11,
    CORNER_RADIUS = 12,

    // Reaming/countersink/counterbore family
    LEAD_CHAMFER = 13,
    LAND = 14,
    COUNTERSINK_FACE = 15,
    COUNTERBORE_FACE = 16,

    // Tapping/thread-milling family
    THREAD_CREST = 17,
    THREAD_FLANK = 18,
    THREAD_ROOT = 19,
    RELIEF_FACE = 20,
    CHAMFER_LEAD = 21,

    // Indexable rotary cutters
    INSERT_RAKE_FACE = 22,
    INSERT_FLANK_FACE = 23,
    INSERT_SEAT = 24,
    CHIP_GULLET = 25,

    // Burr/deburring family
    BURR_TOOTH = 26,
    BURR_GULLET = 27,

    REGION_COUNT = 28
};

EP_HOST_DEVICE inline int toolRegionToInt(ToolRegion region) {
    return static_cast<int>(region);
}


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
    EP_HOST_DEVICE Vec3 operator-() const { return {-x, -y, -z}; }
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
    EP_HOST_DEVICE double length() const { return std::sqrt(lengthSq()); }
    EP_HOST_DEVICE Vec3 normalized() const { 
        double len = length();
        return len > 1e-12 ? (*this / len) : Vec3(0, 0, 0);
    }
    
    // Zero vector
    EP_HOST_DEVICE static Vec3 zero() { return {0, 0, 0}; }
};

EP_HOST_DEVICE inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

/**
 * @brief MPM particle (GPU-optimized layout)
 */
struct MPMParticle {
    // Position
    double x, y, z;
    
    // Velocity
    double vx, vy, vz;
    
    // Force/acceleration
    double fx, fy, fz;
    double ext_fx, ext_fy, ext_fz;
    
    // Properties
    double density;
    double pressure;     // SPH compatibility
    double volume;       // Volume occupied by the particle
    double mass;
    double temperature;
    
    // Deformation gradients (3x3 matrices flattened)
    double F[9];         // Elastic deformation gradient
    double F_p[9];       // Plastic deformation gradient
    
    // APIC affine momentum matrix (3x3 flattened)
    double C[9];
    
    // Stress tensor components (symmetric: 6 components)
    double stress_xx, stress_yy, stress_zz;
    double stress_xy, stress_xz, stress_yz;
    
    // Kinematic hardening backstress (deviatoric, 6 components, Prager-Ziegler)
    double backstress_xx, backstress_yy, backstress_zz;
    double backstress_xy, backstress_xz, backstress_yz;
    
    // Strain and damage
    double plasticStrain;       // Accumulated equivalent plastic strain
    double strainRate;          // Current strain rate
    double damage;              // Accumulated damage (0-1, >1 = failed)
    double residualStress;      // Residual stress after deformation
    
    // State
    int32_t id;
    ParticleStatus status;
    int32_t cellHash;       // Spatial hash for neighbor/grid search
    LODZone lodZone;        // Level of Detail zone
    int32_t lastUpdateStep; // Last step when full physics was computed
    uint8_t asbCounter;     // Counter for adiabatic shear band localization
    
    EP_HOST_DEVICE MPMParticle() 
        : x(0), y(0), z(0),
          vx(0), vy(0), vz(0),
          fx(0), fy(0), fz(0), ext_fx(0), ext_fy(0), ext_fz(0),
          density(0), pressure(0), volume(0), mass(0), temperature(25.0),
          stress_xx(0), stress_yy(0), stress_zz(0),
          stress_xy(0), stress_xz(0), stress_yz(0),
          backstress_xx(0), backstress_yy(0), backstress_zz(0),
          backstress_xy(0), backstress_xz(0), backstress_yz(0),
          plasticStrain(0), strainRate(0), damage(0), residualStress(0),
          id(-1), status(ParticleStatus::ACTIVE), cellHash(0),
          lodZone(LODZone::ACTIVE), lastUpdateStep(0), asbCounter(0) 
    {
        // Initialize F and F_p to Identity (diagonal at 0,4,8 in row-major 3x3), C to zero
        for (int i=0; i<9; i++) {
            F[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
            F_p[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
            C[i] = 0.0;
        }
    }
};

/**
 * @brief Background grid node for MPM
 */
struct MPMGridNode {
    double mass;
    double vx, vy, vz;   // Velocity
    double px, py, pz;   // Momentum
    double fx, fy, fz;   // Force
    
    int32_t isTool;      // 1 if inside tool
    double tvx, tvy, tvz; // Tool velocity at this node
    
    EP_HOST_DEVICE MPMGridNode()
        : mass(0), vx(0), vy(0), vz(0), px(0), py(0), pz(0), fx(0), fy(0), fz(0),
          isTool(0), tvx(0), tvy(0), tvz(0) {}
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
 *
 * Physics overhaul additions:
 *  - heatAccumulator: accumulates raw frictional heat (J) from all particle
 *    contacts in a single step before the temperature is updated. This prevents
 *    the N_contacts × clamp thermal explosion where a dense drill tip gained
 *    40°C/μs. Heat is applied ONCE per step in a dedicated kernel.
 *  - contactForceNormal: magnitude of net normal contact force (N) used by the
 *    Hertz contact stress formula: p_max = (3F)/(2π·a²).
 *  - penetrationDepth: maximum penetration in this step — used as the Hertz
 *    contact half-width proxy: a ≈ sqrt(r_contact * d_penetration).
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

    // Full Cauchy stress tensor (accumulated from elements, volume-weighted)
    double stress_xx, stress_yy, stress_zz;
    double stress_xy, stress_xz, stress_yz;
    double stressWeight;  // Sum of element volumes contributing to this node's stress

    // --- Physics Overhaul: Per-Node Heat Buffer ---
    // Frictional heat (J) accumulated across ALL particle contacts this step.
    // Applied atomically inside the contact kernel; temperature updated ONCE
    // in applyNodeHeatKernel after all contacts are resolved.
    double heatAccumulator;
    double conductionBuffer;   // J (reset to 0 each step by applyNodeHeatKernel)

    // --- Physics Overhaul: Hertz Contact Metrics ---
    // Net normal contact force magnitude and max penetration used to compute
    // the Hertz contact pressure: p_max = (3*Fn)/(2*pi*a^2),  a=sqrt(r*d).
    double contactForceNormal;  // N  (accumulated via atomicAdd)
    double penetrationDepth;    // m  (max penetration; atomic max)

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
    int32_t toolRegion;

    // Spatial hash (used by ContactSolver for O(N×k) neighbor search)
    int cellHash;

    EP_HOST_DEVICE FEMNodeGPU()
        : x(0), y(0), z(0), ox(0), oy(0), oz(0),
          vx(0), vy(0), vz(0),
          fx(0), fy(0), fz(0),
          mass(1e-9), temperature(25.0), vonMisesStress(0), wear(0),
          stress_xx(0), stress_yy(0), stress_zz(0),
          stress_xy(0), stress_xz(0), stress_yz(0), stressWeight(0),
          heatAccumulator(0.0), conductionBuffer(0.0), contactForceNormal(0.0), penetrationDepth(0.0),
          id(-1), status(NodeStatus::OK), isFixed(false), inContact(false),
          isDriven(false), localOffX(0), localOffY(0), localOffZ(0),
          toolRegion(static_cast<int32_t>(ToolRegion::UNKNOWN)),
          cellHash(0) {}
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
    double flankWear;
    double craterWear;
    double chippingRisk;
    double coatingRemaining;
    ToolRegion toolRegion;
    
    NodeStatus status;
    bool isContact;          // Currently in contact with workpiece
    
    EP_HOST_DEVICE FEMNode()
        : position(), originalPosition(), velocity(), force(),
          mass(1e-9), temperature(25.0), stress(0), accumulatedWear(0),
          flankWear(0), craterWear(0), chippingRisk(0), coatingRemaining(1.0),
          toolRegion(ToolRegion::UNKNOWN),
          status(NodeStatus::OK), isContact(false) {}
};

/**
 * @brief Tetrahedral element for FEM
 */
struct FEMElement {
    int32_t nodeIndices[4];  // Tetrahedron has 4 nodes
    double volume;           // Initial volume
    double invDm[3][3];      // Inverse of reference shape matrix

    EP_HOST_DEVICE FEMElement() : volume(0) {
        for (int i = 0; i < 4; ++i) nodeIndices[i] = -1;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                invDm[i][j] = 0.0;
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

/**
 * @brief Barycentric constraint binding a surface node to a tetrahedral element
 */
struct FEMEmbedConstraint {
    int32_t surfaceNodeIdx;
    int32_t tetNodeIndices[4];
    double weights[4];
    
    EP_HOST_DEVICE FEMEmbedConstraint() {
        surfaceNodeIdx = -1;
        for (int i = 0; i < 4; ++i) {
            tetNodeIndices[i] = -1;
            weights[i] = 0.0;
        }
    }
};

// ============================================================================
// Mesh Structure
// ============================================================================

/**
 * @brief Surface and Volume mesh for geometry representation
 */
struct Mesh {
    std::vector<FEMNode> nodes;           // High-resolution surface nodes (Visual/Contact)
    std::vector<Triangle> triangles;      // Original surface triangles
    
    std::vector<FEMNode> tetNodes;        // Internal tetrahedral physics nodes
    std::vector<FEMElement> elements;     // Volume tetrahedral physical elements
    std::vector<FEMEmbedConstraint> embedConstraints; // Coupler structure linking surface -> tet

    std::vector<int> materialStatus;      // Per-vertex: 0=workpiece block, 1=chip, 2=air
    
    void clear() {
        nodes.clear();
        triangles.clear();
        tetNodes.clear();
        elements.clear();
        embedConstraints.clear();
        materialStatus.clear();
    }
    
    bool empty() const { return nodes.empty() && tetNodes.empty(); }
    size_t nodeCount() const { return nodes.size(); }
    size_t triangleCount() const { return triangles.size(); }
    size_t tetNodeCount() const { return tetNodes.size(); }
    size_t elementCount() const { return elements.size(); }
    size_t embedCount() const { return embedConstraints.size(); }
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
    int motionMode;         // 0=rapid(G00), 1=linear(G01), 2=CW arc(G02), 3=CCW arc(G03)
    bool isRapid;           // Convenience: true during G00 (no cutting forces)
    
    MachineState() : position(), feedRate(0), spindleRPM(0), isActive(false), 
                     motionMode(0), isRapid(false) {}
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
