/**
 * @file ThreadingStrategy.cpp
 * @brief Threading operation strategy implementation
 *
 * Threading uses a helical toolpath synchronized with spindle rotation.
 * The tool follows: P(t) = (R·cos(ωt), R·sin(ωt), -pitch · ωt / (2π))
 *
 * Multi-pass depth scheduling uses CONSTANT-AREA infeed (ISO recommended):
 *   depth_n = total_depth × √(n/N)
 * This keeps the chip cross-sectional area approximately constant per pass,
 * which keeps cutting forces uniform and prevents the first pass from being
 * too heavy.
 *
 * V-chip geometry:
 *   Cross-section area = depth² × tan(α)
 *   where α = half of thread form angle (30° for metric)
 */

#include "ThreadingStrategy.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>
#include <iomanip>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;
}

ThreadingStrategy::ThreadingStrategy() = default;

bool ThreadingStrategy::initialize(const Config& config) {
    std::cout << "[ThreadingStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Cutting conditions
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;
    m_conditions.feedRate = machining.feedRateMmMin / 60.0 / 1000.0;  // m/s
    
    // Specific cutting force
    m_specificCuttingForce = 2.5 * material.jc_A;
    
    // Read threading-specific parameters from JSON
    const auto& j = config.getJson();
    if (j.contains("machining_parameters")) {
        const auto& mp = j["machining_parameters"];
        m_threadPitchMm = mp.value("thread_pitch_mm", 1.5);
        m_threadFormAngleDeg = mp.value("thread_form_angle_deg", 60.0);
        m_majorDiameterMm = mp.value("thread_major_diameter_mm", 10.0);
        m_threadLengthMm = mp.value("thread_length_mm", 20.0);
        m_numPasses = mp.value("num_threading_passes", 6);
        m_numSpringPasses = mp.value("num_spring_passes", 1);
        
        // Thread form selection
        std::string formStr = mp.value("thread_form", "metric");
        if (formStr == "metric" || formStr == "M")     m_threadForm = ThreadForm::ISO_METRIC;
        else if (formStr == "UN" || formStr == "UNC")   m_threadForm = ThreadForm::UN_UNIFIED;
        else if (formStr == "BSP" || formStr == "BSPP") m_threadForm = ThreadForm::BSP_PARALLEL;
        else if (formStr == "BSPT")                     m_threadForm = ThreadForm::BSP_TAPER;
        else if (formStr == "Tr" || formStr == "trapezoidal") m_threadForm = ThreadForm::TRAPEZOIDAL;
        else if (formStr == "ACME")                     m_threadForm = ThreadForm::ACME;
        
        // Infeed method
        std::string infeedStr = mp.value("thread_infeed_method", "modified_flank");
        if (infeedStr == "radial")           m_infeedMethod = InfeedMethod::RADIAL;
        else if (infeedStr == "flank")       m_infeedMethod = InfeedMethod::FLANK;
        else if (infeedStr == "modified_flank") m_infeedMethod = InfeedMethod::MODIFIED_FLANK;
        else if (infeedStr == "incremental") m_infeedMethod = InfeedMethod::INCREMENTAL;
    }
    
    // Set thread form angle based on standard
    switch (m_threadForm) {
        case ThreadForm::ISO_METRIC:
        case ThreadForm::UN_UNIFIED:
            m_threadFormAngleDeg = 60.0;
            break;
        case ThreadForm::BSP_PARALLEL:
        case ThreadForm::BSP_TAPER:
            m_threadFormAngleDeg = 55.0;
            break;
        case ThreadForm::TRAPEZOIDAL:
            m_threadFormAngleDeg = 30.0;
            break;
        case ThreadForm::ACME:
            m_threadFormAngleDeg = 29.0;
            break;
    }
    
    // Compute total thread depth from pitch and form angle
    // For ISO metric: H = P × √3/2 = P × 0.866, and depth = 5/8 × H = 0.5413 × P
    double P = m_threadPitchMm / 1000.0;  // meters
    double alpha = getFormHalfAngle();
    
    if (m_threadForm == ThreadForm::ISO_METRIC || m_threadForm == ThreadForm::UN_UNIFIED) {
        double H = P * std::sqrt(3.0) / 2.0;
        m_totalThreadDepth = 5.0 / 8.0 * H;  // Effective thread depth
    } else if (m_threadForm == ThreadForm::BSP_PARALLEL || m_threadForm == ThreadForm::BSP_TAPER) {
        double H = P * 0.960491;  // BSP: H = 0.960491 × P
        m_totalThreadDepth = 0.640327 * P;
    } else {
        // Generic: depth = P / (2 × tan(α))
        m_totalThreadDepth = P / (2.0 * std::tan(alpha));
    }
    
    // Ensure minimum passes for material
    m_numPasses = std::max(m_numPasses, 3);
    m_numSpringPasses = std::max(m_numSpringPasses, 1);
    
    // Compute pass depths using constant-area method
    computePassDepths();
    
    // Tool geometry defaults for threading insert
    m_geometry.cutterDiameter = m_majorDiameterMm / 1000.0;
    m_geometry.rakeAngle = 0;  // Neutral rake for threading inserts
    
    // Spindle tracking
    m_spindlePosition = m_conditions.toolPosition;
    m_prevSpindlePosition = m_spindlePosition;
    m_spindleVelocity = Vec3::zero();
    
    m_currentPass = 0;
    m_threadDepthAchieved = 0;
    m_helicalPhase = 0;
    
    m_initialized = true;
    
    std::cout << "[ThreadingStrategy] Thread: M" << m_majorDiameterMm 
              << " × " << m_threadPitchMm << "mm pitch"
              << ", form angle=" << m_threadFormAngleDeg << "°"
              << ", depth=" << m_totalThreadDepth * 1e6 << "µm"
              << ", " << m_numPasses << " passes + " << m_numSpringPasses << " spring"
              << std::endl;
    
    // Log pass schedule
    std::cout << "[ThreadingStrategy] Pass schedule (µm cumulative): ";
    for (int i = 0; i < static_cast<int>(m_passDepths.size()); ++i) {
        std::cout << std::fixed << std::setprecision(0) << m_passDepths[i] * 1e6;
        if (i < static_cast<int>(m_passDepths.size()) - 1) std::cout << " → ";
    }
    std::cout << std::endl;
    
    return true;
}

void ThreadingStrategy::computePassDepths() {
    m_passDepths.clear();
    
    int cuttingPasses = m_numPasses - m_numSpringPasses;
    if (cuttingPasses < 1) cuttingPasses = 1;
    
    // Constant-area infeed: depth_n = total_depth × √(n/N)
    // This gives decreasing incremental depths per pass (heavy first, light last)
    for (int i = 1; i <= cuttingPasses; ++i) {
        double cumulativeDepth = m_totalThreadDepth * std::sqrt(static_cast<double>(i) / cuttingPasses);
        m_passDepths.push_back(cumulativeDepth);
    }
    
    // Spring passes (same final depth — zero additional infeed)
    for (int i = 0; i < m_numSpringPasses; ++i) {
        m_passDepths.push_back(m_totalThreadDepth);
    }
}

double ThreadingStrategy::getFormHalfAngle() const {
    return (m_threadFormAngleDeg / 2.0) * DEG_TO_RAD;
}

double ThreadingStrategy::computeMinorDiameter() const {
    return (m_majorDiameterMm / 1000.0) - 2.0 * m_totalThreadDepth;
}

double ThreadingStrategy::getCurrentPassDepth() const {
    if (m_currentPass <= 0 || m_currentPass > static_cast<int>(m_passDepths.size())) {
        return 0;
    }
    return m_passDepths[m_currentPass - 1];
}

double ThreadingStrategy::computeVChipArea() const {
    // V-shaped chip cross-sectional area
    // For a V-profile with half-angle α and depth d:
    //   Area = d² × tan(α)
    // The depth per pass is the INCREMENTAL depth (not cumulative)
    
    double currentCumulativeDepth = getCurrentPassDepth();
    double prevCumulativeDepth = (m_currentPass > 1) ? m_passDepths[m_currentPass - 2] : 0;
    double incrementalDepth = currentCumulativeDepth - prevCumulativeDepth;
    
    if (incrementalDepth <= 0) return 0;  // Spring pass
    
    double alpha = getFormHalfAngle();
    
    // For modified flank infeed, the chip is not a symmetric V
    // but a parallelogram. Effective area is slightly larger.
    double areaMultiplier = 1.0;
    if (m_infeedMethod == InfeedMethod::MODIFIED_FLANK) {
        areaMultiplier = 1.1;  // ~10% larger due to alternating flank
    } else if (m_infeedMethod == InfeedMethod::FLANK) {
        areaMultiplier = 0.9;  // Cleaner chip formation
    }
    
    return incrementalDepth * incrementalDepth * std::tan(alpha) * areaMultiplier;
}

Vec3 ThreadingStrategy::computeHelicalPosition(double phase) const {
    // Helical toolpath: P(t) = (R·cos(φ), R·sin(φ), -pitch · φ / (2π))
    double R = (m_majorDiameterMm / 1000.0) / 2.0;
    double pitch = m_threadPitchMm / 1000.0;
    
    return Vec3(
        R * std::cos(phase),
        R * std::sin(phase),
        -pitch * phase / (2.0 * PI)
    );
}

void ThreadingStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
}

void ThreadingStrategy::connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                        ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void ThreadingStrategy::updateConditions(const MachineState& state, double dt) {
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
    
    // Tool rotation (synchronized with spindle for thread cutting)
    double dAngle = m_conditions.spindleSpeed * dt;
    m_toolRotationAngle += dAngle;
    if (m_toolRotationAngle > 2.0 * PI) {
        m_toolRotationAngle -= 2.0 * PI;
    }
    m_conditions.toolRotationAngle = m_toolRotationAngle;
    
    // Helical phase accumulates continuously (for multi-revolution threading)
    m_helicalPhase += dAngle;
    
    // Thread feed rate = pitch × spindle_speed / (2π) — SYNCHRONIZED
    double pitch = m_threadPitchMm / 1000.0;
    m_conditions.feedRate = pitch;  // m/rev (feed = pitch for threading)
    
    // Cutting speed at thread major diameter
    double R = (m_majorDiameterMm / 1000.0) / 2.0;
    m_conditions.cuttingSpeed = R * m_conditions.spindleSpeed;
    
    // Track axial position within current pass
    m_axialPositionInPass += pitch * dAngle / (2.0 * PI);
    
    // Auto-advance pass when thread length is reached
    double threadLength = m_threadLengthMm / 1000.0;
    if (m_axialPositionInPass >= threadLength && !m_passDepths.empty()) {
        if (m_currentPass < static_cast<int>(m_passDepths.size())) {
            m_currentPass++;
            m_threadDepthAchieved = getCurrentPassDepth();
            m_axialPositionInPass = 0;
            
            bool isSpring = (m_currentPass > (m_numPasses - m_numSpringPasses));
            std::cout << "[ThreadingStrategy] Pass " << m_currentPass << "/" << m_numPasses
                      << (isSpring ? " (spring)" : "")
                      << " — depth=" << m_threadDepthAchieved * 1e6 << "µm" << std::endl;
        }
    }
    
    // Start first pass automatically
    if (m_currentPass == 0) {
        m_currentPass = 1;
        m_threadDepthAchieved = getCurrentPassDepth();
    }
    
    // Feed direction = axial (Z) for threading
    m_conditions.feedDirection = Vec3(0, 0, -1);
    m_conditions.cuttingDirection = Vec3(
        -std::sin(m_toolRotationAngle),
        std::cos(m_toolRotationAngle),
        0
    );
}

std::vector<CuttingEdge> ThreadingStrategy::getActiveCuttingEdges() const {
    std::vector<CuttingEdge> edges;
    
    double R = (m_majorDiameterMm / 1000.0) / 2.0;
    double alpha = getFormHalfAngle();
    double currentDepth = getCurrentPassDepth();
    
    // Threading insert — V-shaped cutting edge
    CuttingEdge vEdge;
    double angle = m_toolRotationAngle;
    
    // Insert center at thread periphery
    Vec3 insertCenter(
        m_conditions.toolPosition.x + R * std::cos(angle),
        m_conditions.toolPosition.y + R * std::sin(angle),
        m_conditions.toolPosition.z - m_axialPositionInPass
    );
    
    // V-edge: two sides of the thread form
    // Leading flank
    vEdge.startPoint = Vec3(
        insertCenter.x,
        insertCenter.y,
        insertCenter.z + currentDepth * std::tan(alpha)
    );
    
    // Trailing flank
    vEdge.endPoint = Vec3(
        insertCenter.x,
        insertCenter.y,
        insertCenter.z - currentDepth * std::tan(alpha)
    );
    
    // Normal points radially inward (into material)
    vEdge.normal = Vec3(-std::cos(angle), -std::sin(angle), 0);
    
    vEdge.rakeDirection = Vec3(
        -std::sin(angle),
        std::cos(angle),
        0
    );
    
    vEdge.localRakeAngle = m_geometry.rakeAngle;
    vEdge.engagement = (currentDepth > 0) ? 1.0 : 0.0;
    vEdge.isActive = (currentDepth > 0);
    
    edges.push_back(vEdge);
    
    return edges;
}

MachiningOutput ThreadingStrategy::computeOutput() const {
    MachiningOutput output;
    
    // V-chip area for current pass
    double chipArea = computeVChipArea();
    
    if (chipArea <= 0) {
        // Spring pass — essentially zero cutting force
        output.cuttingForce = Vec3(0, 0, 0);
        output.thrustForce = Vec3(0, 0, 0);
        output.feedForce = Vec3(0, 0, 0);
        output.torque = 0;
        output.power = 0;
        return output;
    }
    
    // Equivalent chip thickness for Kienzle correction
    double alpha = getFormHalfAngle();
    double currentDepth = getCurrentPassDepth();
    double prevDepth = (m_currentPass > 1) ? m_passDepths[m_currentPass - 2] : 0;
    double incrementalDepth = currentDepth - prevDepth;
    
    double h_eq = chipArea / (2.0 * incrementalDepth / std::cos(alpha));  // Equivalent chip thickness
    double h_mm = h_eq * 1000.0;
    if (h_mm < 0.01) h_mm = 0.01;
    
    double mc = 0.25;
    double Kc = m_specificCuttingForce * std::pow(h_mm, -mc);
    
    // Tangential force = Kc × chip_area
    double Ft = Kc * chipArea;
    
    // Radial force from V-profile: Fr = Ft × tan(α)
    double Fr = Ft * std::tan(alpha);
    
    // Axial force (from thread pitch helix angle)
    double pitch = m_threadPitchMm / 1000.0;
    double R = (m_majorDiameterMm / 1000.0) / 2.0;
    double helixAngle = std::atan(pitch / (2.0 * PI * R));
    double Fa = Ft * std::tan(helixAngle);
    
    m_cachedTangentialForce = Ft;
    m_cachedRadialForce = Fr;
    
    // Transform to global
    double angle = m_toolRotationAngle;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    
    output.cuttingForce = Vec3(
        Ft * (-sinA) + Fr * cosA,
        Ft * cosA + Fr * sinA,
        0
    );
    output.thrustForce = Vec3(Fr * cosA, Fr * sinA, 0);
    output.feedForce = Vec3(0, 0, -Fa);
    
    m_cachedTorque = Ft * R;
    output.torque = m_cachedTorque;
    output.power = m_cachedTorque * m_conditions.spindleSpeed;
    
    // Chip formation
    output.chipThickness = h_eq * 2.0;
    output.chipWidth = 2.0 * incrementalDepth * std::tan(alpha);
    output.chipVelocity = m_conditions.cuttingSpeed * 0.5;
    
    // MRR for threading
    double n_rps = m_conditions.spindleSpeed / (2.0 * PI);
    output.materialRemovalRate = chipArea * pitch * n_rps * 1e9;  // mm³/s
    
    // Get FEM values
    if (m_fem) {
        output.maxToolStress = m_fem->getMaxStress();
        output.maxToolTemperature = m_fem->getMaxTemperature();
    }
    
    return output;
}

void ThreadingStrategy::applyKinematics(double dt) {
    // === DIGITAL TWIN: G-Code Kinematic Deferral ===
    if (m_conditions.isGCodeDriven) {
        double dAngle = m_conditions.spindleSpeed * dt;
        m_toolRotationAngle += dAngle;
        if (m_toolRotationAngle > 2.0 * PI) {
            m_toolRotationAngle -= 2.0 * PI;
        }
        m_helicalPhase += dAngle;
        return;
    }
    
    // Legacy fallback
    if (m_fem) {
        double angle = m_conditions.spindleSpeed * dt;
        m_fem->rotateAroundZ(angle, m_conditions.toolPosition.x,
                             m_conditions.toolPosition.y);
        
        double pitch = m_threadPitchMm / 1000.0;
        double axialFeed = pitch * angle / (2.0 * PI);
        m_fem->translateMesh(0, 0, -axialFeed);
        m_conditions.toolPosition.z -= axialFeed;
        m_spindlePosition = m_conditions.toolPosition;
    }
}

void ThreadingStrategy::reset() {
    m_toolRotationAngle = 0;
    m_helicalPhase = 0;
    m_axialPositionInPass = 0;
    m_currentPass = 0;
    m_threadDepthAchieved = 0;
    m_passInProgress = false;
    m_conditions = CuttingConditions();
    m_spindlePosition = Vec3::zero();
    m_prevSpindlePosition = Vec3::zero();
    m_spindleVelocity = Vec3::zero();
    m_cachedTangentialForce = 0;
    m_cachedRadialForce = 0;
    m_cachedTorque = 0;
    m_initialized = false;
}

void ThreadingStrategy::applyAdaptiveControl(double feedMultiplier, double speedMultiplier) {
    m_adaptiveFeedMultiplier = feedMultiplier;
    m_adaptiveSpeedMultiplier = speedMultiplier;
}

} // namespace edgepredict
