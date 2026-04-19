/**
 * @file BoringStrategy.cpp
 * @brief Boring operation strategy implementation
 *
 * Boring enlarges a pre-existing hole.  The critical physics difference
 * from drilling is that the dominant force is RADIAL (outward from bore
 * center), and the boring bar's cantilever deflection under that load
 * directly limits achievable bore tolerance.
 *
 * Force model:
 *   Ft (tangential) = Kc * ap * f * sin(κ)
 *   Fr (radial)     = Kc * ap * f * cos(κ)
 *   Fa (axial)      = Ft * tan(λ)          (helix/back-rake driven)
 *
 * Deflection model:
 *   δ = Fr * L³ / (3 * E * I)
 *   I = π * d⁴ / 64                        (solid round bar)
 */

#include "BoringStrategy.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;
}

BoringStrategy::BoringStrategy() = default;

bool BoringStrategy::initialize(const Config& config) {
    std::cout << "[BoringStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Cutting conditions from config
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;
    m_conditions.feedRate = machining.feedRateMmMin / 60.0 / 1000.0;  // m/s
    
    // Specific cutting force from material (Johnson-Cook A as base)
    m_specificCuttingForce = 2.5 * material.jc_A;
    
    // Read boring-specific parameters from JSON
    const auto& j = config.getJson();
    if (j.contains("machining_parameters")) {
        const auto& mp = j["machining_parameters"];
        m_barLength = mp.value("boring_bar_length_mm", 40.0) / 1000.0;
        m_barDiameter = mp.value("boring_bar_diameter_mm", 10.0) / 1000.0;
        m_boreStartDiameter = mp.value("bore_start_diameter_mm", 20.0) / 1000.0;
        m_radialDepthOfCut = mp.value("radial_depth_of_cut_mm", 1.0) / 1000.0;
        m_insertLeadAngle = mp.value("insert_lead_angle_deg", 90.0);
        m_insertNoseRadius = mp.value("insert_nose_radius_mm", 0.4) / 1000.0;
    }
    
    // Bar material modulus (default carbide; override for steel bars)
    if (j.contains("machining_parameters")) {
        const auto& mp = j["machining_parameters"];
        std::string barMaterial = mp.value("boring_bar_material", "carbide");
        if (barMaterial == "steel" || barMaterial == "HSS") {
            m_barYoungsModulus = 210e9;  // Steel
        } else if (barMaterial == "carbide") {
            m_barYoungsModulus = 600e9;  // Carbide
        } else if (barMaterial == "heavy_metal") {
            m_barYoungsModulus = 350e9;  // Tungsten heavy alloy (anti-vibration)
        }
    }
    
    // Default cutter diameter = bore start diameter + 2 * radial DOC
    if (m_geometry.cutterDiameter == 0.01) {
        m_geometry.cutterDiameter = m_boreStartDiameter + 2.0 * m_radialDepthOfCut;
    }
    
    // Spindle tracking init
    m_spindlePosition = m_conditions.toolPosition;
    m_prevSpindlePosition = m_spindlePosition;
    m_spindleVelocity = Vec3::zero();
    
    m_initialized = true;
    
    double LD = m_barLength / m_barDiameter;
    std::cout << "[BoringStrategy] Bar L=" << m_barLength * 1000 << "mm, D=" 
              << m_barDiameter * 1000 << "mm, L/D=" << std::fixed << std::setprecision(1) << LD
              << ", bore Ø" << m_boreStartDiameter * 1000 << "mm" << std::endl;
    std::cout << "[BoringStrategy] DOC(radial)=" << m_radialDepthOfCut * 1000 
              << "mm, κ=" << m_insertLeadAngle << "°" << std::endl;
    
    if (LD > 4.0) {
        std::cerr << "[BoringStrategy] WARNING: L/D ratio " << LD 
                  << " exceeds 4:1 — expect significant deflection and possible chatter" << std::endl;
    }
    
    return true;
}

void BoringStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
}

void BoringStrategy::connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                     ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void BoringStrategy::updateConditions(const MachineState& state, double dt) {
    m_conditions.spindleSpeed = state.spindleRPM * 2.0 * PI / 60.0 * m_adaptiveSpeedMultiplier;
    
    // Anchored Physics: G-Code-driven position
    m_prevSpindlePosition = m_spindlePosition;
    
    if (state.isActive) {
        m_spindlePosition = state.position;
        m_conditions.isGCodeDriven = true;
    } else {
        m_conditions.isGCodeDriven = false;
    }
    
    if (dt > 1e-12) {
        m_spindleVelocity = (m_spindlePosition - m_prevSpindlePosition) / dt;
    }
    
    m_conditions.spindlePosition = m_spindlePosition;
    m_conditions.spindleVelocity = m_spindleVelocity;
    m_conditions.toolPosition = m_spindlePosition;
    
    // Tool rotation
    m_toolRotationAngle += m_conditions.spindleSpeed * dt;
    if (m_toolRotationAngle > 2.0 * PI) {
        m_toolRotationAngle -= 2.0 * PI;
    }
    m_conditions.toolRotationAngle = m_toolRotationAngle;
    
    // Feed per revolution
    double feedMmPerMin = state.feedRate * 60000.0;  // m/s to mm/min
    double rpm = state.spindleRPM;
    if (rpm > 0) {
        m_conditions.feedRate = feedMmPerMin / 1000.0 / rpm;  // m/rev
    }
    
    // Cutting speed at bore periphery
    double R = (m_boreStartDiameter / 2.0) + m_radialDepthOfCut;
    m_conditions.cuttingSpeed = R * m_conditions.spindleSpeed;
    
    // Axial depth tracking
    m_currentDepth += feedMmPerMin / 60.0 * dt;
    
    // Directions — boring feeds axially downward
    m_conditions.feedDirection = Vec3(0, 0, -1);
    // Cutting direction rotates with tool
    m_conditions.cuttingDirection = Vec3(
        -std::sin(m_toolRotationAngle),
        std::cos(m_toolRotationAngle),
        0
    );
}

std::vector<CuttingEdge> BoringStrategy::getActiveCuttingEdges() const {
    std::vector<CuttingEdge> edges;
    
    double R = (m_boreStartDiameter / 2.0) + m_radialDepthOfCut;
    double kappaRad = m_insertLeadAngle * DEG_TO_RAD;
    
    // Single-point boring insert
    CuttingEdge insert;
    double angle = m_toolRotationAngle;
    
    // Insert is mounted at the bore periphery
    insert.startPoint = Vec3(
        m_conditions.toolPosition.x + R * std::cos(angle),
        m_conditions.toolPosition.y + R * std::sin(angle),
        m_conditions.toolPosition.z
    );
    
    // Edge extends along the insert (axial direction modulated by lead angle)
    double edgeLen = m_radialDepthOfCut / std::sin(kappaRad);
    insert.endPoint = Vec3(
        m_conditions.toolPosition.x + (R - m_radialDepthOfCut) * std::cos(angle),
        m_conditions.toolPosition.y + (R - m_radialDepthOfCut) * std::sin(angle),
        m_conditions.toolPosition.z - edgeLen * std::cos(kappaRad)
    );
    
    // Normal points radially OUTWARD from bore center (into bore wall)
    insert.normal = Vec3(std::cos(angle), std::sin(angle), 0);
    
    // Rake direction (tangential + slightly helical if back rake)
    insert.rakeDirection = Vec3(
        -std::sin(angle),
        std::cos(angle),
        0
    );
    
    insert.localRakeAngle = m_geometry.rakeAngle;
    insert.engagement = 1.0;
    insert.isActive = true;
    
    edges.push_back(insert);
    
    return edges;
}

// ---------------------------------------------------------------------------
// Force Model
// ---------------------------------------------------------------------------

double BoringStrategy::computeRadialForce() const {
    // Fr = Kc * ap * f * cos(κ)
    double ap = m_radialDepthOfCut;
    double f = m_conditions.feedRate;
    double kappaRad = m_insertLeadAngle * DEG_TO_RAD;
    
    // Kienzle correction for chip thickness
    double h = f * std::sin(kappaRad);  // uncut chip thickness
    double h_mm = h * 1000.0;
    if (h_mm < 0.01) h_mm = 0.01;
    double mc = 0.25;
    double Kc = m_specificCuttingForce * std::pow(h_mm, -mc);
    
    return Kc * ap * f * std::cos(kappaRad);
}

double BoringStrategy::computeTangentialForce() const {
    // Ft = Kc * ap * f * sin(κ)
    double ap = m_radialDepthOfCut;
    double f = m_conditions.feedRate;
    double kappaRad = m_insertLeadAngle * DEG_TO_RAD;
    
    double h = f * std::sin(kappaRad);
    double h_mm = h * 1000.0;
    if (h_mm < 0.01) h_mm = 0.01;
    double mc = 0.25;
    double Kc = m_specificCuttingForce * std::pow(h_mm, -mc);
    
    return Kc * ap * f * std::sin(kappaRad);
}

double BoringStrategy::computeAxialForce() const {
    // Fa is typically 20-30% of Ft for boring
    return computeTangentialForce() * 0.25;
}

double BoringStrategy::computeBarDeflection(double radialForce) const {
    // Cantilever beam deflection: δ = F * L³ / (3 * E * I)
    // Moment of inertia for solid round bar: I = π * d⁴ / 64
    double d = m_barDiameter;
    double I = PI * d * d * d * d / 64.0;
    double L = m_barLength;
    double E = m_barYoungsModulus;
    
    if (E * I < 1e-30) return 0.0;  // Safety
    
    return std::abs(radialForce) * L * L * L / (3.0 * E * I);
}

MachiningOutput BoringStrategy::computeOutput() const {
    MachiningOutput output;
    
    double Fr = computeRadialForce();
    double Ft = computeTangentialForce();
    double Fa = computeAxialForce();
    
    m_cachedRadialForce = Fr;
    m_cachedTangentialForce = Ft;
    m_cachedAxialForce = Fa;
    
    // Bar deflection from radial force
    double deflection = computeBarDeflection(Fr);
    m_cachedBarDeflection = deflection;
    
    // Log if deflection is critical
    if (deflection > 10e-6) {  // > 10 µm
        static int warnCount = 0;
        if (warnCount++ % 100 == 0) {
            std::cerr << "[BoringStrategy] WARNING: Bar deflection = " 
                      << deflection * 1e6 << " µm (>10µm threshold)" << std::endl;
        }
    }
    
    // Transform forces to global coordinates based on tool rotation angle
    double angle = m_toolRotationAngle;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    
    // Radial force points outward from bore center
    output.cuttingForce = Vec3(
        Ft * (-sinA) + Fr * cosA,
        Ft * cosA + Fr * sinA,
        0
    );
    output.thrustForce = Vec3(Fr * cosA, Fr * sinA, 0);
    output.feedForce = Vec3(0, 0, -Fa);
    
    // Torque = tangential force × radius
    double R = (m_boreStartDiameter / 2.0) + m_radialDepthOfCut;
    m_cachedTorque = Ft * R;
    output.torque = m_cachedTorque;
    
    // Power = torque × angular velocity
    output.power = m_cachedTorque * m_conditions.spindleSpeed;
    
    // Chip formation
    double kappaRad = m_insertLeadAngle * DEG_TO_RAD;
    output.chipThickness = m_conditions.feedRate * std::sin(kappaRad) * 2.0;
    output.chipWidth = m_radialDepthOfCut / std::sin(kappaRad);
    output.chipVelocity = m_conditions.cuttingSpeed * 0.5;
    
    // MRR = π * D_avg * ap * f * n
    double Davg = m_boreStartDiameter + m_radialDepthOfCut;
    double n = m_conditions.spindleSpeed / (2.0 * PI);  // rev/s
    output.materialRemovalRate = PI * Davg * m_radialDepthOfCut * 
                                  m_conditions.feedRate * n * 1e9;  // mm³/s
    
    // Surface roughness from nose radius: Ra ≈ f² / (32 * r_nose)
    double f_mm = m_conditions.feedRate * n * 1000.0;  // mm/rev
    if (m_insertNoseRadius > 0) {
        double r_mm = m_insertNoseRadius * 1000.0;
        output.theoreticalRoughness = (f_mm * f_mm) / (32.0 * r_mm);
    }
    
    // Get actual stress/temp from FEM solver
    if (m_fem) {
        output.maxToolStress = m_fem->getMaxStress();
        output.maxToolTemperature = m_fem->getMaxTemperature();
    }
    
    return output;
}

void BoringStrategy::applyKinematics(double dt) {
    // === DIGITAL TWIN: G-Code Kinematic Deferral ===
    if (m_conditions.isGCodeDriven) {
        m_toolRotationAngle += m_conditions.spindleSpeed * dt;
        if (m_toolRotationAngle > 2.0 * PI) {
            m_toolRotationAngle -= 2.0 * PI;
        }
        return;
    }
    
    // Legacy fallback
    if (m_fem) {
        double angle = m_conditions.spindleSpeed * dt;
        m_fem->rotateAroundZ(angle, m_conditions.toolPosition.x,
                             m_conditions.toolPosition.y);
        
        double feed = m_conditions.feedRate * m_conditions.spindleSpeed / (2 * PI) * dt;
        m_fem->translateMesh(0, 0, -feed);
        m_conditions.toolPosition.z -= feed;
        m_spindlePosition = m_conditions.toolPosition;
    }
}

void BoringStrategy::reset() {
    m_toolRotationAngle = 0;
    m_currentDepth = 0;
    m_conditions = CuttingConditions();
    m_spindlePosition = Vec3::zero();
    m_prevSpindlePosition = Vec3::zero();
    m_spindleVelocity = Vec3::zero();
    m_cachedRadialForce = 0;
    m_cachedTangentialForce = 0;
    m_cachedAxialForce = 0;
    m_cachedTorque = 0;
    m_cachedBarDeflection = 0;
    m_initialized = false;
}

void BoringStrategy::applyAdaptiveControl(double feedMultiplier, double speedMultiplier) {
    m_adaptiveFeedMultiplier = feedMultiplier;
    m_adaptiveSpeedMultiplier = speedMultiplier;
}

double BoringStrategy::getRadialForce() const {
    return m_cachedRadialForce;
}

double BoringStrategy::getBarDeflection() const {
    return m_cachedBarDeflection;
}

double BoringStrategy::getTorque() const {
    return m_cachedTorque;
}

} // namespace edgepredict
