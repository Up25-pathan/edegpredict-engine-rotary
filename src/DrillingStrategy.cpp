/**
 * @file DrillingStrategy.cpp
 * @brief Drilling operation strategy implementation
 */

#include "DrillingStrategy.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;
}

DrillingStrategy::DrillingStrategy() = default;

bool DrillingStrategy::initialize(const Config& config) {
    std::cout << "[DrillingStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Conditions from config
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;
    m_conditions.feedRate = machining.feedRateMmMin / 60.0 / 1000.0;  // m/s
    
    // Specific cutting force
    m_specificCuttingForce = 2.5 * material.jc_A;
    
    // Default drill geometry
    if (m_geometry.pointAngle == 0) {
        m_geometry.pointAngle = 118.0;  // Standard twist drill
        m_geometry.chiselEdgeRatio = 0.18;
        m_geometry.cutterDiameter = 0.01;  // 10mm
        m_geometry.helixAngle = 30.0;
    }
    
    // Compute web thickness from chisel edge ratio
    m_geometry.webThickness = m_geometry.cutterDiameter * m_geometry.chiselEdgeRatio;
    
    // === ANCHORED PHYSICS: Spindle tracking ===
    m_spindlePosition = m_conditions.toolPosition;
    m_prevSpindlePosition = m_spindlePosition;
    m_spindleVelocity = Vec3::zero();
    
    m_initialized = true;
    
    std::cout << "[DrillingStrategy] Drill D=" << m_geometry.cutterDiameter * 1000 
              << "mm, point=" << m_geometry.pointAngle << "°" << std::endl;
    
    return true;
}

void DrillingStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
}

void DrillingStrategy::connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                       ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void DrillingStrategy::updateConditions(const MachineState& state, double dt) {
    m_conditions.spindleSpeed = state.spindleRPM * 2.0 * PI / 60.0 * m_adaptiveSpeedMultiplier;
    
    // === ANCHORED PHYSICS: G-Code-driven position ===
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
    
    // Drill rotation
    m_drillRotationAngle += m_conditions.spindleSpeed * dt;
    if (m_drillRotationAngle > 2 * PI) {
        m_drillRotationAngle -= 2 * PI;
    }
    m_conditions.toolRotationAngle = m_drillRotationAngle;
    
    // Feed per revolution
    double feedMmPerMin = state.feedRate * 60000.0;  // m/s to mm/min
    double rpm = state.spindleRPM;
    m_conditions.feedRate = feedMmPerMin / 1000.0 / rpm;  // m/rev
    
    // Cutting speed at periphery
    double R = m_geometry.cutterDiameter / 2.0;
    m_conditions.cuttingSpeed = R * m_conditions.spindleSpeed;
    
    // Update depth
    m_currentDepth += feedMmPerMin / 60.0 * dt;
    
    // Directions
    m_conditions.feedDirection = Vec3(0, 0, -1);  // Drilling down
    m_conditions.cuttingDirection = Vec3(
        -std::sin(m_drillRotationAngle),
        std::cos(m_drillRotationAngle),
        0
    );
}

std::vector<CuttingEdge> DrillingStrategy::getActiveCuttingEdges() const {
    std::vector<CuttingEdge> edges;
    
    double R = m_geometry.cutterDiameter / 2.0;
    double pointRad = m_geometry.pointAngle * DEG_TO_RAD / 2.0;
    double webR = m_geometry.webThickness / 2.0;
    
    // Two main cutting lips (for twist drill)
    for (int i = 0; i < 2; ++i) {
        CuttingEdge lip;
        
        double angle = m_drillRotationAngle + i * PI;
        
        // Lip starts at web, ends at periphery
        lip.startPoint = Vec3(
            m_conditions.toolPosition.x + webR * std::cos(angle),
            m_conditions.toolPosition.y + webR * std::sin(angle),
            m_conditions.toolPosition.z - webR / std::tan(pointRad)
        );
        
        lip.endPoint = Vec3(
            m_conditions.toolPosition.x + R * std::cos(angle),
            m_conditions.toolPosition.y + R * std::sin(angle),
            m_conditions.toolPosition.z
        );
        
        // Normal (into workpiece, along drill axis)
        lip.normal = Vec3(0, 0, -1);
        
        // Rake direction (along helix)
        double helixRad = m_geometry.helixAngle * DEG_TO_RAD;
        lip.rakeDirection = Vec3(
            -std::sin(angle) * std::cos(helixRad),
            std::cos(angle) * std::cos(helixRad),
            std::sin(helixRad)
        );
        
        lip.localRakeAngle = m_geometry.helixAngle;  // Approximate
        lip.engagement = 1.0;
        lip.isActive = true;
        
        edges.push_back(lip);
    }
    
    // Chisel edge (at drill center)
    CuttingEdge chisel;
    chisel.startPoint = Vec3(
        m_conditions.toolPosition.x - webR,
        m_conditions.toolPosition.y,
        m_conditions.toolPosition.z - webR / std::tan(pointRad)
    );
    chisel.endPoint = Vec3(
        m_conditions.toolPosition.x + webR,
        m_conditions.toolPosition.y,
        m_conditions.toolPosition.z - webR / std::tan(pointRad)
    );
    chisel.normal = Vec3(0, 0, -1);
    chisel.rakeDirection = Vec3(0, 1, 0);
    chisel.localRakeAngle = -30.0;  // Chisel has negative rake
    chisel.engagement = 1.0;
    chisel.isActive = true;
    edges.push_back(chisel);
    
    return edges;
}

MachiningOutput DrillingStrategy::computeOutput() const {
    MachiningOutput output;
    
    // Thrust force = chisel edge + cutting lips
    double Fz_chisel = computeChiselEdgeForce();
    double Fz_lips = computeCuttingLipForce();
    double Fz = Fz_chisel + Fz_lips;
    
    // Torque from cutting lips only (chisel contributes little torque)
    double D = m_geometry.cutterDiameter;
    double f = m_conditions.feedRate;
    double kc = m_specificCuttingForce;
    
    // Using Shaw's formula adapted for drilling
    // Torque = (Kc * f * D²) / 8
    double Mc = (kc * f * D * D) / 8.0;
    
    m_cachedThrust = Fz;
    m_cachedTorque = Mc;
    
    output.thrustForce = Vec3(0, 0, -Fz);  // Axial (downward)
    output.cuttingForce = Vec3(0, 0, 0);   // Tangential forces cancel
    output.feedForce = Vec3(0, 0, -Fz);    // Same as thrust for drilling
    output.torque = Mc;
    
    // Power = Mc * ω
    output.power = Mc * m_conditions.spindleSpeed;
    
    // Chip formation
    // h = f * sin(κ), where κ = point angle / 2
    double kappa = m_geometry.pointAngle * DEG_TO_RAD / 2.0;
    output.chipThickness = f * std::sin(kappa) * 2.0;  // With compression
    output.chipWidth = (D - m_geometry.webThickness) / 2.0;
    
    // MRR = π/4 * D² * feed_rate_per_min
    double Vf = f * m_conditions.spindleSpeed / (2 * PI) * 60.0;  // m/min
    output.materialRemovalRate = PI / 4.0 * D * D * Vf * 1e9;  // mm³/min
    
    // Get actual values from solvers
    if (m_fem) {
        output.maxToolStress = m_fem->getMaxStress();
        output.maxToolTemperature = m_fem->getMaxTemperature();
    }
    
    return output;
}

double DrillingStrategy::computeChiselEdgeForce() const {
    // Chisel edge contributes significantly to thrust (up to 50%)
    // Empirical: Fz_chisel = C * Kc * f * w
    // where w = web thickness, C ≈ 1.2-1.5
    
    double C = 1.3;
    double f = m_conditions.feedRate;
    double w = m_geometry.webThickness;
    
    return C * m_specificCuttingForce * f * w;
}

double DrillingStrategy::computeCuttingLipForce() const {
    // Cutting lips thrust force
    // Fz_lip = Kc * f * (D - w) / 2 * sin(κ)
    
    double D = m_geometry.cutterDiameter;
    double w = m_geometry.webThickness;
    double f = m_conditions.feedRate;
    double kappa = m_geometry.pointAngle * DEG_TO_RAD / 2.0;
    
    return m_specificCuttingForce * f * (D - w) / 2.0 * std::sin(kappa);
}

void DrillingStrategy::applyKinematics(double dt) {
    // === DIGITAL TWIN: G-Code Kinematic Deferral ===
    if (m_conditions.isGCodeDriven) {
        m_drillRotationAngle += m_conditions.spindleSpeed * dt;
        if (m_drillRotationAngle > 2 * PI) {
            m_drillRotationAngle -= 2 * PI;
        }
        return;
    }
    
    // Legacy fallback: manual kinematics when no G-Code
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

void DrillingStrategy::reset() {
    m_drillRotationAngle = 0;
    m_currentDepth = 0;
    m_conditions = CuttingConditions();
    m_spindlePosition = Vec3::zero();
    m_prevSpindlePosition = Vec3::zero();
    m_spindleVelocity = Vec3::zero();
    m_cachedThrust = 0;
    m_cachedTorque = 0;
    m_initialized = false;
}

void DrillingStrategy::applyAdaptiveControl(double feedMultiplier, double speedMultiplier) {
    m_adaptiveFeedMultiplier = feedMultiplier;
    m_adaptiveSpeedMultiplier = speedMultiplier;
    // Don't modify conditions here — updateConditions() already applies these multipliers
}

double DrillingStrategy::getThrustForce() const {
    return m_cachedThrust;
}

double DrillingStrategy::getTorque() const {
    return m_cachedTorque;
}

double DrillingStrategy::getPower() const {
    return m_cachedTorque * m_conditions.spindleSpeed;
}

} // namespace edgepredict
