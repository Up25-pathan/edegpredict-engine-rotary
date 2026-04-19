/**
 * @file ReamingStrategy.cpp
 * @brief Reaming operation strategy implementation
 *
 * Reaming is a multi-flute finish pass with very light depth of cut (0.1-0.5mm).
 * All flutes engage simultaneously as the reamer is guided by the existing hole.
 * The quality metric is hole roundness / h7 tolerance, not stock removal rate.
 *
 * Force model:
 *   Per-flute: F = Kc * ap * h(φ)  where h(φ) = fz * sin(φ)
 *   Radial forces nearly cancel out due to uniform flute spacing
 *   Thrust (axial) force is very small — reamer self-feeds from chamfered entry
 *
 * Roundness model:
 *   Deviation from perfect circle driven by:
 *   - Flute force imbalance (odd vs even flute count)
 *   - Tool runout (spindle TIR)
 *   - Entry taper/chamfer engagement
 */

#include "ReamingStrategy.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;
}

ReamingStrategy::ReamingStrategy() = default;

bool ReamingStrategy::initialize(const Config& config) {
    std::cout << "[ReamingStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Cutting conditions
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;
    m_conditions.feedRate = machining.feedRateMmMin / 60.0 / 1000.0;  // m/s
    
    // Specific cutting force
    m_specificCuttingForce = 2.5 * material.jc_A;
    
    // Read reaming-specific parameters from JSON
    const auto& j = config.getJson();
    if (j.contains("machining_parameters")) {
        const auto& mp = j["machining_parameters"];
        m_numFlutes = mp.value("num_reamer_flutes", 6);
        m_reamDiameter = mp.value("reamer_diameter_mm", 10.0) / 1000.0;
        m_radialStock = mp.value("radial_stock_mm", 0.2) / 1000.0;
        m_h7ToleranceMm = mp.value("h7_tolerance_mm", 0.015);
        m_pilotHoleDiameter = mp.value("pilot_hole_diameter_mm", 
                                        (m_reamDiameter * 1000.0 - 2.0 * m_radialStock * 1000.0)) / 1000.0;
    }
    
    // Clamp radial stock to reaming limits (0.1 - 0.5 mm)
    m_radialStock = std::max(0.0001, std::min(0.0005, m_radialStock));
    
    // Clamp flute count to valid range
    m_numFlutes = std::max(4, std::min(8, m_numFlutes));
    
    // Set geometry
    m_geometry.cutterDiameter = m_reamDiameter;
    m_geometry.numFlutes = m_numFlutes;
    m_geometry.helixAngle = 10.0;  // Reamers have low helix (0-15°)
    m_geometry.rakeAngle = 5.0;    // Low positive rake
    
    // DOC for reaming = radial stock
    m_conditions.depthOfCut = m_radialStock;
    
    initializeFlutes();
    
    // Spindle tracking
    m_spindlePosition = m_conditions.toolPosition;
    m_prevSpindlePosition = m_spindlePosition;
    m_spindleVelocity = Vec3::zero();
    
    m_initialized = true;
    
    std::cout << "[ReamingStrategy] Reamer Ø" << m_reamDiameter * 1000 << "mm, " 
              << m_numFlutes << " flutes, stock=" << m_radialStock * 1000 
              << "mm, h7 target=±" << m_h7ToleranceMm << "mm" << std::endl;
    
    return true;
}

void ReamingStrategy::initializeFlutes() {
    m_flutes.clear();
    m_flutes.resize(m_numFlutes);
    
    double angleStep = 2.0 * PI / m_numFlutes;
    
    // Unequal pitch spacing for chatter suppression (realistic reamers)
    // Deviation from equal spacing: ±2° per flute (typical)
    for (int i = 0; i < m_numFlutes; ++i) {
        m_flutes[i].index = i;
        // Slight unequal spacing for vibration damping
        double deviation = (i % 2 == 0) ? 2.0 * DEG_TO_RAD : -2.0 * DEG_TO_RAD;
        m_flutes[i].angularPosition = i * angleStep + deviation;
        m_flutes[i].helixAngle = m_geometry.helixAngle;
        m_flutes[i].rakeAngle = m_geometry.rakeAngle;
        m_flutes[i].isEngaged = false;
    }
}

void ReamingStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
    initializeFlutes();
}

void ReamingStrategy::connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                      ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void ReamingStrategy::updateConditions(const MachineState& state, double dt) {
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
    
    // Cutting speed at periphery
    double R = m_reamDiameter / 2.0;
    m_conditions.cuttingSpeed = R * m_conditions.spindleSpeed;
    
    // Feed per tooth
    double rpm = state.spindleRPM;
    if (rpm > 0) {
        double feedMmPerMin = state.feedRate * 60000.0;
        m_conditions.feedRate = feedMmPerMin / 1000.0 / rpm;  // m/rev
    }
    
    m_currentDepth += state.feedRate * dt;
    
    m_conditions.feedDirection = Vec3(0, 0, -1);
    m_conditions.cuttingDirection = Vec3(
        -std::sin(m_toolRotationAngle),
        std::cos(m_toolRotationAngle),
        0
    );
    
    // Update flute positions
    for (auto& flute : m_flutes) {
        flute.angularPosition = m_toolRotationAngle + 
            (flute.index * 2.0 * PI / m_numFlutes);
        while (flute.angularPosition > 2.0 * PI) flute.angularPosition -= 2.0 * PI;
        while (flute.angularPosition < 0) flute.angularPosition += 2.0 * PI;
    }
    
    updateFluteEngagement();
}

void ReamingStrategy::updateFluteEngagement() {
    // In reaming, ALL flutes engage simultaneously (full circumference cut)
    // because the reamer fills the bore. Engagement arc = full 360°.
    
    for (auto& flute : m_flutes) {
        flute.isEngaged = true;  // All flutes are always engaged
        flute.engagementStart = 0;
        flute.engagementEnd = 2.0 * PI;
        flute.currentChipLoad = computeChipLoadAtAngle(flute.angularPosition);
    }
}

double ReamingStrategy::computeChipLoadAtAngle(double angle) const {
    // h(φ) = fz * sin(φ) for each flute
    // Feed per tooth = feed_per_rev / num_flutes
    double n_rps = m_conditions.spindleSpeed / (2.0 * PI);
    double fz = (n_rps > 0) ? m_conditions.feedRate / m_numFlutes : 0;
    
    // Chip load modulated by angular position
    double h = fz * std::abs(std::sin(angle));
    
    // Clamp to reaming DOC range (very thin chips)
    return std::min(h, m_maxRadialDOC);
}

std::vector<CuttingEdge> ReamingStrategy::getActiveCuttingEdges() const {
    std::vector<CuttingEdge> edges;
    
    double R = m_reamDiameter / 2.0;
    
    for (const auto& flute : m_flutes) {
        if (!flute.isEngaged) continue;
        
        CuttingEdge edge;
        double angle = flute.angularPosition;
        
        edge.startPoint = Vec3(
            m_conditions.toolPosition.x + R * std::cos(angle),
            m_conditions.toolPosition.y + R * std::sin(angle),
            m_conditions.toolPosition.z
        );
        
        // Short flute length for reamers (radial stock depth only)
        double helixRad = flute.helixAngle * DEG_TO_RAD;
        double fluteLen = m_geometry.fluteLength > 0 ? m_geometry.fluteLength : 0.015;
        double angleOffset = fluteLen * std::tan(helixRad) / R;
        
        edge.endPoint = Vec3(
            m_conditions.toolPosition.x + R * std::cos(angle + angleOffset),
            m_conditions.toolPosition.y + R * std::sin(angle + angleOffset),
            m_conditions.toolPosition.z - fluteLen
        );
        
        edge.normal = Vec3(std::cos(angle), std::sin(angle), 0);
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

Vec3 ReamingStrategy::computeFluteForce(const Flute& flute) const {
    double h = flute.currentChipLoad;
    double b = m_radialStock;  // Axial depth = radial stock for reaming
    
    if (h <= 0) return Vec3(0, 0, 0);
    
    // Kienzle correction
    double h_mm = h * 1000.0;
    if (h_mm < 0.001) h_mm = 0.001;  // Very small chips in reaming
    double mc = 0.25;
    double Kc = m_specificCuttingForce * std::pow(h_mm, -mc);
    
    // Tangential force (very light cutting)
    double Ft = Kc * b * h;
    
    // Radial force — very small in reaming (tool guided by hole)
    double Kr = 0.15;  // Much lower than milling (0.4)
    double Fr = Kr * Ft;
    
    // Axial force — minimal with reamer chamfered entry
    double Fa = 0.1 * Ft;
    
    return Vec3(Ft, Fr, Fa);
}

double ReamingStrategy::computeHoleRoundness() const {
    // Roundness deviation model for reaming:
    // Sources: flute force imbalance, runout, elastic recovery
    
    // 1. Force imbalance from unequal pitch spacing
    double maxForce = 0, minForce = 1e30;
    for (const auto& flute : m_flutes) {
        Vec3 f = computeFluteForce(flute);
        double mag = std::sqrt(f.x * f.x + f.y * f.y);
        maxForce = std::max(maxForce, mag);
        minForce = std::min(minForce, mag);
    }
    double forceImbalance = (maxForce - minForce) / 
                            (maxForce > 0 ? maxForce : 1.0);
    
    // 2. Elastic springback (workpiece material)
    double elasticRecovery = m_radialStock * 0.02;  // ~2% elastic recovery
    
    // 3. Combined roundness deviation (µm)
    double roundness = (forceImbalance * m_reamDiameter * 0.5 + elasticRecovery) * 1e6;
    
    return roundness;
}

MachiningOutput ReamingStrategy::computeOutput() const {
    MachiningOutput output;
    
    Vec3 totalForce(0, 0, 0);
    double totalTorque = 0;
    
    for (const auto& flute : m_flutes) {
        if (!flute.isEngaged) continue;
        
        Vec3 ff = computeFluteForce(flute);
        double angle = flute.angularPosition;
        double cosA = std::cos(angle);
        double sinA = std::sin(angle);
        
        double Ft = ff.x;
        double Fr = ff.y;
        double Fa = ff.z;
        
        totalForce.x += Ft * (-sinA) + Fr * cosA;
        totalForce.y += Ft * cosA + Fr * sinA;
        totalForce.z += Fa;
        
        double R = m_reamDiameter / 2.0;
        totalTorque += Ft * R;
    }
    
    output.cuttingForce = totalForce;
    output.thrustForce = Vec3(0, 0, totalForce.z);  // Very small
    output.feedForce = Vec3(0, 0, -totalForce.z);
    m_cachedTorque = totalTorque;
    m_cachedThrust = totalForce.z;
    output.torque = totalTorque;
    output.power = totalTorque * m_conditions.spindleSpeed;
    
    // Chip parameters (very thin chips)
    double maxChip = 0;
    for (const auto& flute : m_flutes) {
        if (flute.currentChipLoad > maxChip) maxChip = flute.currentChipLoad;
    }
    output.chipThickness = maxChip;
    output.chipWidth = m_radialStock;
    output.chipVelocity = m_conditions.cuttingSpeed * 0.4;
    
    // MRR (very low for reaming — finish operation)
    double n_rps = m_conditions.spindleSpeed / (2.0 * PI);
    output.materialRemovalRate = PI * m_reamDiameter * m_radialStock * 
                                  m_conditions.feedRate * n_rps * 1e9;
    
    // Surface roughness (reaming achieves Ra 0.4-1.6 µm typically)
    double fz_mm = (n_rps > 0) ? (m_conditions.feedRate / m_numFlutes * 1000.0) : 0;
    double r_mm = m_geometry.noseRadius * 1000.0;
    if (r_mm > 0) {
        output.theoreticalRoughness = (fz_mm * fz_mm) / (32.0 * r_mm * m_numFlutes);
    } else {
        output.theoreticalRoughness = 0.8;  // Typical reaming Ra
    }
    
    // Hole quality metrics
    m_holeRoundnessUm = computeHoleRoundness();
    
    // h7 tolerance check
    // h7 tolerance band depends on diameter:
    //   Ø6-10mm: 0 to -15µm
    //   Ø10-18mm: 0 to -18µm
    //   Ø18-30mm: 0 to -21µm
    double toleranceBand;
    double d_mm = m_reamDiameter * 1000.0;
    if (d_mm <= 10) toleranceBand = 0.015;
    else if (d_mm <= 18) toleranceBand = 0.018;
    else if (d_mm <= 30) toleranceBand = 0.021;
    else if (d_mm <= 50) toleranceBand = 0.025;
    else toleranceBand = 0.030;
    
    m_h7OutputMm = m_holeRoundnessUm / 1000.0;  // µm to mm
    output.h7ToleranceOutputMm = m_h7OutputMm;
    
    // Actual values from solvers
    if (m_fem) {
        output.maxToolStress = m_fem->getMaxStress();
        output.maxToolTemperature = m_fem->getMaxTemperature();
    }
    
    return output;
}

void ReamingStrategy::applyKinematics(double dt) {
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
        
        double feed = m_conditions.feedRate * m_conditions.spindleSpeed / (2.0 * PI) * dt;
        m_fem->translateMesh(0, 0, -feed);
        m_conditions.toolPosition.z -= feed;
        m_spindlePosition = m_conditions.toolPosition;
    }
}

void ReamingStrategy::reset() {
    m_toolRotationAngle = 0;
    m_currentDepth = 0;
    m_conditions = CuttingConditions();
    m_spindlePosition = Vec3::zero();
    m_prevSpindlePosition = Vec3::zero();
    m_spindleVelocity = Vec3::zero();
    m_cachedTorque = 0;
    m_cachedThrust = 0;
    m_holeRoundnessUm = 0;
    m_h7OutputMm = 0;
    initializeFlutes();
    m_initialized = false;
}

void ReamingStrategy::applyAdaptiveControl(double feedMultiplier, double speedMultiplier) {
    m_adaptiveFeedMultiplier = feedMultiplier;
    m_adaptiveSpeedMultiplier = speedMultiplier;
}

int ReamingStrategy::getEngagedFluteCount() const {
    int count = 0;
    for (const auto& flute : m_flutes) {
        if (flute.isEngaged) ++count;
    }
    return count;
}

double ReamingStrategy::getChipLoad(int fluteIndex) const {
    if (fluteIndex >= 0 && fluteIndex < static_cast<int>(m_flutes.size())) {
        return m_flutes[fluteIndex].currentChipLoad;
    }
    return 0;
}

bool ReamingStrategy::meetsH7Tolerance() const {
    double d_mm = m_reamDiameter * 1000.0;
    double toleranceBand;
    if (d_mm <= 10) toleranceBand = 0.015;
    else if (d_mm <= 18) toleranceBand = 0.018;
    else if (d_mm <= 30) toleranceBand = 0.021;
    else if (d_mm <= 50) toleranceBand = 0.025;
    else toleranceBand = 0.030;
    
    return m_h7OutputMm <= toleranceBand;
}

} // namespace edgepredict
