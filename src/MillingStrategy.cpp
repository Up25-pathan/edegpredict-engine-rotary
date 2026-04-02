/**
 * @file MillingStrategy.cpp
 * @brief Milling operation strategy implementation
 */

#include "MillingStrategy.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;
}

MillingStrategy::MillingStrategy() = default;

bool MillingStrategy::initialize(const Config& config) {
    std::cout << "[MillingStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Set conditions from config
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;  // RPM to rad/s
    m_conditions.depthOfCut = machining.depthOfCutMm / 1000.0;    // mm to m
    
    // Feed rate for milling is in mm/min (total), convert to m/s
    double feedMmPerMin = machining.feedRateMmMin;
    m_conditions.feedRate = feedMmPerMin / 60.0 / 1000.0;  // m/s
    
    // Parse Initial Position (with safe JSON access)
    std::vector<double> initPos = {0.0, 0.0, 0.0};
    if (config.getJson().contains("machining_parameters") &&
        config.getJson()["machining_parameters"].contains("initial_tool_position_mm")) {
         initPos = config.getJson()["machining_parameters"]["initial_tool_position_mm"].get<std::vector<double>>();
    }
    m_spindlePosition = m_conditions.toolPosition;
    m_prevSpindlePosition = m_spindlePosition;
    m_spindleVelocity = Vec3::zero();
    
    // Specific cutting force from material
    m_specificCuttingForce = 2.5 * material.jc_A;
    
    // Default geometry if not set
    if (m_geometry.numFlutes == 0) {
        m_geometry.numFlutes = 4;
        m_geometry.cutterDiameter = 0.01;  // 10mm
        m_geometry.helixAngle = 30.0;
        m_geometry.fluteLength = 0.025;    // 25mm
        m_geometry.rakeAngle = 10.0;
    }
    
    // Initialize flutes
    initializeFlutes();
    
    m_initialized = true;
    
    std::cout << "[MillingStrategy] Cutter: D=" << m_geometry.cutterDiameter * 1000 
              << "mm, " << m_geometry.numFlutes << " flutes"
              << ", helix=" << m_geometry.helixAngle << "°" << std::endl;
    std::cout << "[MillingStrategy] Kc=" << m_specificCuttingForce / 1e6 << " MPa" << std::endl;
    
    return true;
}

void MillingStrategy::initializeFlutes() {
    m_flutes.clear();
    m_flutes.resize(m_geometry.numFlutes);
    
    double angleStep = 2.0 * PI / m_geometry.numFlutes;
    
    for (int i = 0; i < m_geometry.numFlutes; ++i) {
        m_flutes[i].index = i;
        m_flutes[i].angularPosition = i * angleStep;
        m_flutes[i].helixAngle = m_geometry.helixAngle;
        m_flutes[i].rakeAngle = m_geometry.rakeAngle;
        m_flutes[i].isEngaged = false;
    }
}

void MillingStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
    initializeFlutes();
}

void MillingStrategy::connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                      ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void MillingStrategy::updateConditions(const MachineState& state, double dt) {
    // Update from machine state
    m_conditions.spindleSpeed = state.spindleRPM * 2.0 * PI / 60.0 * m_adaptiveSpeedMultiplier;
    
    // === ANCHORED PHYSICS: G-Code-driven position ===
    m_prevSpindlePosition = m_spindlePosition;
    
    // If state has active feed, assume it's G-Code driven
    if (state.isActive) {
        m_spindlePosition = state.position;
        m_conditions.isGCodeDriven = true;
    } else {
        m_conditions.isGCodeDriven = false;
    }
    
    // Compute velocity from position delta (finite difference)
    if (dt > 1e-12) {
        m_spindleVelocity = (m_spindlePosition - m_prevSpindlePosition) / dt;
    }
    
    m_conditions.spindlePosition = m_spindlePosition;
    m_conditions.spindleVelocity = m_spindleVelocity;
    m_conditions.toolPosition = m_spindlePosition;
    
    // Tool rotation
    m_toolRotationAngle += m_conditions.spindleSpeed * dt;
    if (m_toolRotationAngle > 2 * PI) {
        m_toolRotationAngle -= 2 * PI;
    }
    m_conditions.toolRotationAngle = m_toolRotationAngle;
    
    // Cutting speed at tool periphery
    double radius = m_geometry.cutterDiameter / 2.0;
    m_conditions.cuttingSpeed = radius * m_conditions.spindleSpeed;
    
    // Width of cut (radial depth)
    m_conditions.widthOfCut = m_geometry.cutterDiameter * getRadialImmersion();
    
    // Feed direction (typically along X or Y)
    m_conditions.feedDirection = Vec3(1, 0, 0);  // Default X+
    
    // Cutting direction rotates with tool
    m_conditions.cuttingDirection = Vec3(
        -std::sin(m_toolRotationAngle),
        std::cos(m_toolRotationAngle),
        0
    );
    
    // Update flute positions and engagement
    for (auto& flute : m_flutes) {
        flute.angularPosition = m_toolRotationAngle + 
                                 (flute.index * 2.0 * PI / m_geometry.numFlutes);
        
        // Normalize to [0, 2π]
        while (flute.angularPosition > 2 * PI) flute.angularPosition -= 2 * PI;
        while (flute.angularPosition < 0) flute.angularPosition += 2 * PI;
    }
    
    updateFluteEngagement();
}

void MillingStrategy::updateFluteEngagement() {
    // Calculate engagement arc based on milling mode and radial immersion
    double ae = m_conditions.widthOfCut;                    // Radial depth
    double D = m_geometry.cutterDiameter;
    double R = D / 2.0;
    
    // Engagement start/end angles
    double phiStart, phiEnd;
    
    double immersion = ae / D;
    
    if (m_millingMode == MillingMode::SLOT) {
        // Full slot: 0 to 180 degrees
        phiStart = 0;
        phiEnd = PI;
    } else if (m_millingMode == MillingMode::CLIMB) {
        // Climb (down) milling: chip starts thick, ends thin
        phiEnd = PI;
        phiStart = std::acos(1.0 - 2.0 * immersion);
    } else if (m_millingMode == MillingMode::CONVENTIONAL) {
        // Conventional (up) milling: chip starts thin, ends thick
        phiStart = 0;
        phiEnd = std::acos(1.0 - 2.0 * immersion);
    } else {
        // Face milling: symmetric
        double halfAngle = std::asin(ae / D);
        phiStart = PI / 2.0 - halfAngle;
        phiEnd = PI / 2.0 + halfAngle;
    }
    
    // Check each flute for engagement
    for (auto& flute : m_flutes) {
        // Flute position relative to feed direction
        double fluteAngle = flute.angularPosition;
        
        // Check if flute is within engagement arc
        if (fluteAngle >= phiStart && fluteAngle <= phiEnd) {
            flute.isEngaged = true;
            flute.engagementStart = phiStart;
            flute.engagementEnd = phiEnd;
            
            // Compute instantaneous chip load at this angle
            flute.currentChipLoad = computeChipLoadAtAngle(fluteAngle);
        } else {
            flute.isEngaged = false;
            flute.currentChipLoad = 0;
        }
    }
}

double MillingStrategy::computeChipLoadAtAngle(double angle) const {
    // Instantaneous chip thickness in milling
    // h(φ) = fz * sin(φ)
    // where fz = feed per tooth
    
    double feedPerTooth = m_conditions.feedRate / (m_geometry.numFlutes * 
                          m_conditions.spindleSpeed / (2.0 * PI));  // m/tooth
    
    return feedPerTooth * std::sin(angle);
}

std::vector<CuttingEdge> MillingStrategy::getActiveCuttingEdges() const {
    std::vector<CuttingEdge> edges;
    
    double radius = m_geometry.cutterDiameter / 2.0;
    
    for (const auto& flute : m_flutes) {
        if (!flute.isEngaged) continue;
        
        CuttingEdge edge;
        
        // Edge position on cutter periphery
        double angle = flute.angularPosition;
        
        edge.startPoint = Vec3(
            m_conditions.toolPosition.x + radius * std::cos(angle),
            m_conditions.toolPosition.y + radius * std::sin(angle),
            m_conditions.toolPosition.z
        );
        
        // End point accounts for helix
        double helixRad = flute.helixAngle * DEG_TO_RAD;
        double zOffset = m_geometry.fluteLength;
        double angleOffset = zOffset * std::tan(helixRad) / radius;
        
        edge.endPoint = Vec3(
            m_conditions.toolPosition.x + radius * std::cos(angle + angleOffset),
            m_conditions.toolPosition.y + radius * std::sin(angle + angleOffset),
            m_conditions.toolPosition.z - zOffset
        );
        
        // Normal points radially outward
        edge.normal = Vec3(std::cos(angle), std::sin(angle), 0);
        
        // Rake direction (tangent with helix angle)
        edge.rakeDirection = Vec3(
            -std::sin(angle) * std::cos(helixRad),
            std::cos(angle) * std::cos(helixRad),
            -std::sin(helixRad)
        );
        
        edge.localRakeAngle = flute.rakeAngle;
        edge.engagement = flute.currentChipLoad > 0 ? 1.0 : 0;
        edge.isActive = true;
        
        edges.push_back(edge);
    }
    
    return edges;
}

MachiningOutput MillingStrategy::computeOutput() const {
    MachiningOutput output;
    
    // Sum forces from all engaged flutes
    Vec3 totalCuttingForce(0, 0, 0);
    Vec3 totalThrustForce(0, 0, 0);
    Vec3 totalFeedForce(0, 0, 0);
    
    double totalTorque = 0;
    double maxChipLoad = 0;
    
    for (const auto& flute : m_flutes) {
        if (!flute.isEngaged) continue;
        
        Vec3 fluteForce = computeFluteForce(flute);
        
        // Transform to global coordinates
        double angle = flute.angularPosition;
        double cosA = std::cos(angle);
        double sinA = std::sin(angle);
        
        // Tangential force (cutting direction)
        double Ft = fluteForce.x;
        // Radial force
        double Fr = fluteForce.y;
        // Axial force
        double Fa = fluteForce.z;
        
        // Transform to X-Y-Z
        totalCuttingForce.x += Ft * (-sinA) + Fr * cosA;
        totalCuttingForce.y += Ft * cosA + Fr * sinA;
        totalFeedForce.z += Fa;
        
        // Torque from tangential force
        double radius = m_geometry.cutterDiameter / 2.0;
        totalTorque += Ft * radius;
        
        if (flute.currentChipLoad > maxChipLoad) {
            maxChipLoad = flute.currentChipLoad;
        }
    }
    
    output.cuttingForce = totalCuttingForce;
    output.feedForce = totalFeedForce;
    output.thrustForce = totalThrustForce;
    output.torque = totalTorque;
    
    // Power
    output.power = totalTorque * m_conditions.spindleSpeed;
    
    // Chip parameters (average)
    output.chipThickness = maxChipLoad * 2.0;  // Compression
    output.chipWidth = m_conditions.depthOfCut;
    output.chipVelocity = m_conditions.cuttingSpeed / 2.0;
    
    // MRR = ae * ap * Vf
    output.materialRemovalRate = m_conditions.widthOfCut * m_conditions.depthOfCut * 
                                  m_conditions.feedRate * 1e9;  // mm³/s
    
    // Surface roughness estimate
    // Ra ≈ fz² / (32 * corner_radius) for flat end
    // For ball nose: Ra depends on step-over
    if (m_endMillType == EndMillType::BALL_NOSE) {
        double stepOver = m_conditions.widthOfCut;
        double R = m_geometry.cutterDiameter / 2.0;
        output.theoreticalRoughness = (stepOver * stepOver) / (8.0 * R) * 1e6;  // μm
    } else if (m_geometry.noseRadius > 0) {
        double fz = maxChipLoad * 1000;  // mm
        double r = m_geometry.noseRadius * 1000;  // mm
        output.theoreticalRoughness = (fz * fz) / (32.0 * r);
    }
    
    // Get actual values from solvers
    if (m_fem) {
        output.maxToolStress = m_fem->getMaxStress();
        output.maxToolTemperature = m_fem->getMaxTemperature();
    }
    
    return output;
}

Vec3 MillingStrategy::computeFluteForce(const Flute& flute) const {
    // Cutting force on single flute
    // Using mechanistic model: F = Kc * b * h
    
    double h = flute.currentChipLoad;  // Chip thickness
    double b = m_conditions.depthOfCut;  // Chip width (axial)
    
    if (h <= 0) return Vec3(0, 0, 0);
    
    // Kienzle correction for chip thickness
    double mc = 0.25;
    double h_mm = h * 1000;
    if (h_mm < 0.01) h_mm = 0.01;
    double Kc_corrected = m_specificCuttingForce * std::pow(h_mm, -mc);
    
    // Tangential (cutting) force
    double Ft = Kc_corrected * b * h;
    
    // Radial force (depends on rake angle)
    double Kr = 0.4;  // Typical ratio
    double Fr = Kr * Ft;
    
    // Axial force (depends on helix angle)
    double helixRad = flute.helixAngle * DEG_TO_RAD;
    double Fa = Ft * std::tan(helixRad);
    
    return Vec3(Ft, Fr, Fa);
}

void MillingStrategy::applyKinematics(double dt) {
    // In milling: tool rotates, workpiece typically moves for feed
    
    // Rotate tool mesh (FEM)
    if (m_fem) {
        double angle = m_conditions.spindleSpeed * dt;
        m_fem->rotateAroundZ(angle, m_conditions.toolPosition.x, 
                             m_conditions.toolPosition.y);
                             
        // === ANCHORED PHYSICS: Translate FEM to G-Code position ===
        if (m_conditions.isGCodeDriven) {
            Vec3 delta = m_spindlePosition - m_prevSpindlePosition;
            if (delta.lengthSq() > 1e-18) {
                m_fem->translateMesh(delta.x, delta.y, delta.z);
            }
        }
    }
    
    // Legacy fallback: Move workpiece in feed direction if not G-Code driven
    if (!m_conditions.isGCodeDriven) {
        // Update tool position for feed
        m_conditions.toolPosition.x += m_conditions.feedRate * dt * m_conditions.feedDirection.x;
        m_conditions.toolPosition.y += m_conditions.feedRate * dt * m_conditions.feedDirection.y;
        m_conditions.toolPosition.z += m_conditions.feedRate * dt * m_conditions.feedDirection.z;
        m_spindlePosition = m_conditions.toolPosition;
    }
}

void MillingStrategy::reset() {
    m_toolRotationAngle = 0;
    m_conditions = CuttingConditions();
    m_spindlePosition = Vec3::zero();
    m_prevSpindlePosition = Vec3::zero();
    m_spindleVelocity = Vec3::zero();
    initializeFlutes();
    m_initialized = false;
}

void MillingStrategy::applyAdaptiveControl(double feedMultiplier, double speedMultiplier) {
    m_adaptiveFeedMultiplier = feedMultiplier;
    m_adaptiveSpeedMultiplier = speedMultiplier;
    // Don't modify conditions here — updateConditions() already applies these multipliers
}

int MillingStrategy::getEngagedFluteCount() const {
    int count = 0;
    for (const auto& flute : m_flutes) {
        if (flute.isEngaged) ++count;
    }
    return count;
}

double MillingStrategy::getChipLoad(int fluteIndex) const {
    if (fluteIndex >= 0 && fluteIndex < static_cast<int>(m_flutes.size())) {
        return m_flutes[fluteIndex].currentChipLoad;
    }
    return 0;
}

double MillingStrategy::getRadialImmersion() const {
    // ae/D ratio (0 to 0.5 for side milling, 1.0 for slot)
    if (m_millingMode == MillingMode::SLOT) {
        return 1.0;
    }
    // Default to 50% radial immersion
    return 0.5;
}

double MillingStrategy::getEngagementAngle() const {
    double immersion = getRadialImmersion();
    if (m_millingMode == MillingMode::SLOT) {
        return PI;
    }
    return std::acos(1.0 - 2.0 * immersion);
}

} // namespace edgepredict
