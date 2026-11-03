/**
 * @file DrillingStrategy.cpp
 * @brief Drilling operation strategy implementation
 */

#include "DrillingStrategy.h"
#include "EdgeSubgridModel.h"
#include "BUEModel.h"
#include "ChatterDynamics.h"
#include "MPMSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include "PhysicsConstants.h"
#include <iostream>
#include <cmath>

namespace edgepredict {

namespace {
    using constants::PI;
    using edgepredict::physics::DEG_TO_RAD;
    using edgepredict::physics::ZOREV_TEMP_START;
    using edgepredict::physics::ZOREV_TEMP_RANGE;
    using edgepredict::physics::ZOREV_FRICTION_REDUCTION;
    using edgepredict::physics::MERCHANT_BASE_ANGLE;
    using edgepredict::physics::KIENZLE_EXPONENT;
    using edgepredict::physics::DRILLING_CONTACT_RADIUS_MULTIPLIER;
    using edgepredict::physics::CHIP_COMPRESSION_MIN;
    using edgepredict::physics::CHIP_COMPRESSION_MAX;
}

DrillingStrategy::DrillingStrategy() = default;

bool DrillingStrategy::initialize(const Config& config) {
    std::cout << "[DrillingStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Conditions from config
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;
    m_conditions.feedRate = machining.feedRateMmMin / 60.0 / 1000.0;  // m/s
    m_configRPM = machining.rpm;  // Store config RPM as fallback
    
    // Specific cutting force
    m_specificCuttingForce = 2.5 * material.jc_A;

    // Fix 2: Store material properties for edge subgrid (replaces hardcoded Ti-6Al-4V)
    m_workYieldPa = material.yieldStrength;
    m_workYoungsPa = material.youngsModulus;
    m_toolYoungsPa = config.getToolMaterial().youngsModulus;
    
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

void DrillingStrategy::connectSolvers(MPMSolver* sph, FEMSolver* fem, 
                                       ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void DrillingStrategy::updateConditions(const MachineState& state, double dt) {
    // Robustness: drill_DE10.gcode normally carries an S-word (S3501 M03), but
    // if a G-Code lacks it, state.spindleRPM is 0. Falling back to the config
    // RPM keeps the peripheral cutting speed R·ω valid — otherwise the AMS
    // contact-frequency estimate collapses to 0 (pinning it at the 1 kHz floor),
    // the torque, and the shear-chipping model.
    double rpm = state.spindleRPM;
    if (rpm < 1.0) rpm = m_configRPM;
    if (rpm < 1.0) rpm = 1.0;

    m_conditions.spindleSpeed = rpm * 2.0 * PI / 60.0 * m_adaptiveSpeedMultiplier;

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
    
    double feedMmPerMin = state.feedRate * 60000.0 * m_adaptiveFeedMultiplier;  // m/s to mm/min
    m_conditions.feedRate = feedMmPerMin / 60.0 / 1000.0;  // m/s
    
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
    double pointRad = std::clamp(m_geometry.pointAngle * DEG_TO_RAD / 2.0,
                                  DEG_TO_RAD * 0.5, DEG_TO_RAD * 44.5);
    double webR = m_geometry.webThickness / 2.0;
    
    // Two main cutting lips (for twist drill)
    for (int i = 0; i < 2; ++i) {
        CuttingEdge lip;
        
        double angle = m_drillRotationAngle + i * PI;
        
        // Lip starts at web, ends at periphery
        // FIX 4.4: restore the true drill cone profile. The apex (tip) is the
        // LOWEST point (z = toolPosition.z). Along the lip the surface rises
        // toward the shank: z(r) = z_apex + r/tan(κ). The old code placed the
        // web point BELOW the apex (-webR/tanκ) and clamped the periphery flat
        // at the apex, so the axial elevation over the lip was a fraction of
        // the physical value and the drill behaved like a ~170° point. (pointRad
        // here is already the half-angle κ.)
        double tanKappa = std::tan(pointRad);
        lip.startPoint = Vec3(
            m_conditions.toolPosition.x + webR * std::cos(angle),
            m_conditions.toolPosition.y + webR * std::sin(angle),
            m_conditions.toolPosition.z + webR / tanKappa
        );
        
        lip.endPoint = Vec3(
            m_conditions.toolPosition.x + R * std::cos(angle),
            m_conditions.toolPosition.y + R * std::sin(angle),
            m_conditions.toolPosition.z + R / tanKappa
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
    
    // Chisel edge (at drill center) — sits at the apex of the point cone
    double tanKappaChisel = std::tan(pointRad);
    CuttingEdge chisel;
    chisel.startPoint = Vec3(
        m_conditions.toolPosition.x - webR,
        m_conditions.toolPosition.y,
        m_conditions.toolPosition.z + webR * 0.5 / tanKappaChisel
    );
    chisel.endPoint = Vec3(
        m_conditions.toolPosition.x + webR,
        m_conditions.toolPosition.y,
        m_conditions.toolPosition.z + webR * 0.5 / tanKappaChisel
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
    double omega = fmax(m_conditions.spindleSpeed, 1e-6);
    double f_rev = f / (omega / (2.0 * constants::PI));  // m/s → m/rev
    
    // Shaw's formula for drilling torque: Mc = kc * f_rev * D² / 8
    // f_rev in m/rev, D in m → result in N·m
    // Convert to mm-based units for Shaw formula
    double f_rev_mm = f_rev * 1000.0;
    double D_mm = D * 1000.0;
    double Mc_Nmm = (kc / 1e6) * f_rev_mm * D_mm * D_mm / 8.0;
    double Mc = Mc_Nmm / 1000.0;
    
    m_cachedThrust = Fz;
    m_cachedTorque = Mc;
    
    output.thrustForce = Vec3(0, 0, -Fz);  // Axial (downward)
    output.cuttingForce = Vec3(0, 0, 0);   // Tangential forces cancel
    output.feedForce = Vec3(0, 0, 0);      // In drilling, axial force is thrustForce; feedForce is zeroed to prevent double-counting
    output.torque = Mc;
    
    // Periphery (tangential) cutting speed — used for contact-frequency
    // estimation in the AMS spectral-radius controller. This is the true
    // speed of the cutting edge passing through the material (R×ω), NOT the
    // axial feed rate (which is ~200× smaller and pins the frequency too low).
    output.cuttingSpeed = m_conditions.cuttingSpeed;
    
    // Power = Mc * ω
    output.power = Mc * m_conditions.spindleSpeed;
    
    // Chip formation
    // h = f * sin(κ), where κ = point angle / 2
    double kappa = std::clamp(m_geometry.pointAngle * DEG_TO_RAD / 2.0,
                               DEG_TO_RAD * 0.5, DEG_TO_RAD * 44.5);
    // Chip compression ratio from Merchant's shear-angle analysis with dynamic flow stress.
    // Standard sheet-shear relation: r = t0/tc = sin(φ)/cos(φ−α), so
    //   tc = t0 · cos(φ−α)/sin(φ),   φ = 45° + (α−β)/2   (Merchant), β = atan(μ).
    // The previous code divided a stress (Pa) by a speed (m/s scaled ×1e6) —
    // a dimensional mismatch that forced the shear angle onto its 15° clamp
    // and pinned chipCompression at 1/tan(15°) ≈ 3.73 (the "0.0004" artifact).
    double rakeRad = m_geometry.helixAngle * DEG_TO_RAD;
    double frictionModRad = std::atan(m_frictionCoef);
    
    // Get dynamic flow stress from MPM particles in cutting zone
    double flowStress = m_specificCuttingForce;  // fallback
    if (m_sph) {
        // Get average flow stress from MPM particles in cutting zone
        // This would require sampling particles near the cutting edge
        // For now use specificCuttingForce which is based on material JC_A
        // TODO: Sample MPM particles in cutting zone for dynamic flow stress
    }
    
    double shearAngleRad = edgepredict::physics::MERCHANT_BASE_ANGLE + (rakeRad - frictionModRad) / 2.0;
    shearAngleRad = std::clamp(shearAngleRad, 15.0 * DEG_TO_RAD, 55.0 * DEG_TO_RAD);
    double chipCompression = std::cos(shearAngleRad - rakeRad) /
                             std::sin(shearAngleRad);
    // Clamp to physically reasonable range
    chipCompression = std::clamp(chipCompression, edgepredict::physics::CHIP_COMPRESSION_MIN, edgepredict::physics::CHIP_COMPRESSION_MAX);
    output.chipThickness = f_rev * std::sin(kappa) * chipCompression;
    output.chipWidth = (D - m_geometry.webThickness) / 2.0;
    output.chipVelocity = m_conditions.cuttingSpeed * 0.4;
    
    // MRR = π/4 * D² * Vf where Vf = axial feed speed
    // f is in m/s → convert to mm/s: Vf (mm/s) = f * 1000
    // D is in m → convert to mm: D_mm = D * 1000
    // MRR = π/4 * D_mm² * Vf  (mm³/s)
    double Vf_mm_s = f * 1000.0;
    output.materialRemovalRate = (PI / 4.0) * (D * 1000.0) * (D * 1000.0) * Vf_mm_s;
    
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
    // Convert m/s to m/rev
    double omega = fmax(m_conditions.spindleSpeed, 1e-6);
    double f_rev = m_conditions.feedRate / (omega / (2.0 * PI));
    double w = m_geometry.webThickness;
    
    return C * m_specificCuttingForce * f_rev * w;
}

double DrillingStrategy::computeCuttingLipForce() const {
    // Cutting lips thrust force
    // Fz_lip = Kc * f * (D - w) / 2 * sin(κ)
    
    double D = m_geometry.cutterDiameter;
    double w = m_geometry.webThickness;
    // Convert m/s to m/rev
    double omega = fmax(m_conditions.spindleSpeed, 1e-6);
    double f_rev = m_conditions.feedRate / (omega / (2.0 * PI));
    double kappa = std::clamp(m_geometry.pointAngle * DEG_TO_RAD / 2.0,
                               DEG_TO_RAD * 0.5, DEG_TO_RAD * 44.5);
    
    return m_specificCuttingForce * f_rev * (D - w) / 2.0 * std::sin(kappa);
}

void DrillingStrategy::applyKinematics(double dt) {
    // === DIGITAL TWIN: G-Code Kinematic Deferral ===
    // Rotation is already updated in updateConditions() — do not double-increment.
    if (m_conditions.isGCodeDriven) {
        return;
    }
    
    // Legacy fallback: manual kinematics when no G-Code
    if (m_fem) {
        double angle = m_conditions.spindleSpeed * dt;
        m_fem->rotateAroundZ(angle, m_conditions.toolPosition.x,
                             m_conditions.toolPosition.y);
        
        double feed = m_conditions.feedRate * dt;
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

// ---------------------------------------------------------------------------
// Edge Subgrid Model Corrections
// ---------------------------------------------------------------------------

void DrillingStrategy::applyEdgeSubgridCorrections() {
    if (!m_edgeSubgrid || !m_edgeSubgrid->isEnabled()) return;

    double chipPerRev = m_conditions.feedRate /
        std::max(m_conditions.spindleSpeed / (2.0 * PI), 1e-12);
    double uncutChipThickness = chipPerRev * std::sin(
        m_geometry.pointAngle * DEG_TO_RAD / 2.0);

    if (uncutChipThickness <= 0) return;

    // FIX 3.5: engaged lip length (web radius → periphery) replaces the
    // hardcoded 1e-4 m ploughing width.
    double lipWidth = (m_geometry.cutterDiameter / 2.0) - (m_geometry.webThickness / 2.0);

    auto result = m_edgeSubgrid->compute(
        uncutChipThickness, lipWidth,
        m_specificCuttingForce * 0.5,
        m_conditions.cuttingSpeed,
        m_currentDepth > 0.001 ? 300.0 : 25.0,
        m_workYieldPa, m_workYoungsPa, m_toolYoungsPa  // Fix 2: use actual material properties
    );

    m_conditions.effectiveRakeAngleRad = result.effectiveRakeAngleRad;
    m_conditions.ploughingForceN = result.ploughingForceN;
    m_conditions.sizeEffectFactor = result.sizeEffectFactor;
    m_conditions.edgeTemperatureFactor = result.edgeTemperatureFactor;
}

// ---------------------------------------------------------------------------
// BUE Model Corrections
// ---------------------------------------------------------------------------

void DrillingStrategy::applyBUECorrections() {
    if (!m_bue || !m_bue->isEnabled() || !m_bue->isActive()) return;

    m_conditions.bueEffectiveRakeChangeDeg = m_bue->getEffectiveRakeCorrectionDeg();
    m_conditions.bueCoverageFraction = m_bue->getState().coverageFraction;
    m_conditions.bueForceModulation = m_bue->getForceModulationFactor();
}

// ---------------------------------------------------------------------------
// Chatter Modulation
// ---------------------------------------------------------------------------

void DrillingStrategy::applyChatterModulation(double dt) {
    if (!m_chatter || !m_chatter->isEnabled()) return;

    m_conditions.chipThicknessModulation = m_chatter->getChipThicknessModulation();
    double disp = m_chatter->getVibrationDisplacement().length();
    m_conditions.vibrationDisplacementUm = disp * 1e6;
    m_conditions.chattering = m_chatter->isChattering();
    m_conditions.stabilityParameter = m_chatter->getStabilityParameter();
}

} // namespace edgepredict
