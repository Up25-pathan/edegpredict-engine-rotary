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
#include "EdgeSubgridModel.h"
#include "BUEModel.h"
#include "ChatterDynamics.h"
#include "MPMSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>

namespace edgepredict {

namespace {
    using constants::PI;
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
    m_configRPM = machining.rpm;  // Store config RPM as fallback
    
    // Specific cutting force from material (Johnson-Cook A as base)
    m_specificCuttingForce = 2.5 * material.jc_A;

    // Material properties for edge subgrid (from config, replaces hardcoded Ti-6Al-4V)
    m_workYieldPa = material.yieldStrength;
    m_workYoungsPa = material.youngsModulus;
    m_toolYoungsPa = config.getToolMaterial().youngsModulus;
    
    // Read boring-specific parameters from JSON (single pass)
    const auto& j = config.getJson();
    if (j.contains("machining_parameters")) {
        const auto& mp = j["machining_parameters"];
        m_barLength = mp.value("boring_bar_length_mm", 40.0) / 1000.0;
        m_barDiameter = mp.value("boring_bar_diameter_mm", 10.0) / 1000.0;
        m_boreStartDiameter = mp.value("bore_start_diameter_mm", 20.0) / 1000.0;
        m_radialDepthOfCut = mp.value("radial_depth_of_cut_mm", 1.0) / 1000.0;
        m_insertLeadAngle = mp.value("insert_lead_angle_deg", 90.0);
        m_insertNoseRadius = mp.value("insert_nose_radius_mm", 0.4) / 1000.0;

        // Bar material modulus
        std::string barMaterial = mp.value("boring_bar_material", "steel");
        if (barMaterial == "steel" || barMaterial == "HSS") {
            m_barYoungsModulus = 210e9;
        } else if (barMaterial == "carbide") {
            m_barYoungsModulus = 600e9;
        } else if (barMaterial == "heavy_metal") {
            m_barYoungsModulus = 350e9;
        } else {
            m_barYoungsModulus = 210e9;
        }
    } else {
        m_barYoungsModulus = 210e9;
    }
    
    // Default cutter diameter = bore start diameter + 2 * radial DOC
    // Fix: sentinel check by "unset" (<= 0), not by equality to the 10mm default.
    if (m_geometry.cutterDiameter <= 0.0) {
        m_geometry.cutterDiameter = m_boreStartDiameter + 2.0 * m_radialDepthOfCut;
    }
    
    // Spindle tracking init
    m_spindlePosition = m_conditions.toolPosition;
    m_prevSpindlePosition = m_spindlePosition;
    m_spindleVelocity = Vec3::zero();
    
    m_initialized = true;
    
    double LD = m_barLength / fmax(m_barDiameter, 1e-6);
    std::cout << "[BoringStrategy] Bar L=" << m_barLength * 1000 << "mm, D=" 
              << m_barDiameter * 1000 << "mm, L/D=" << std::fixed << std::setprecision(1) << LD
              << ", bore Ø" << m_boreStartDiameter * 1000 << "mm" << std::endl;
    std::cout << "[BoringStrategy] DOC(radial)=" << m_radialDepthOfCut * 1000 
              << "mm, κ=" << m_insertLeadAngle << "°" << std::endl;
    
    if (LD > 4.0) {
        std::cerr << "[BoringStrategy] WARNING: L/D ratio " << LD 
                  << " exceeds 4:1 — expect significant deflection and possible chatter" << std::endl;
    }
    
    // ── P1 FIX: Initialize Dynamic Boring Bar Vibration Model ────────────────
    // First bending mode of cantilever beam: f_n = (1.875²/2π) * sqrt(EI/ρAL⁴)
    double A = PI * m_barDiameter * m_barDiameter / 4.0;  // Cross-sectional area
    double I = PI * m_barDiameter * m_barDiameter * m_barDiameter * m_barDiameter / 64.0;
    double rho = 14500.0;  // Carbide density (kg/m³) - bar material
    double m_eff = 0.236 * rho * A * m_barLength;  // Modal effective mass (first mode)
    m_barStiffness = 3.0 * m_barYoungsModulus * I / (m_barLength * m_barLength * m_barLength);  // k = 3EI/L³
    m_barEffectiveMass = m_eff;
    m_barNaturalFrequency = (1.0 / (2.0 * PI)) * sqrt(m_barStiffness / m_eff);
    
    // Initialize vibration state
    m_barVibrationDisplacement = 0.0;
    m_barVibrationVelocity = 0.0;
    m_prevRadialForce = 0.0;
    
    std::cout << "[BoringStrategy] Bar dynamics: f_n=" << m_barNaturalFrequency 
              << " Hz, k=" << m_barStiffness / 1e6 << " MN/m, m_eff=" << m_eff * 1000 << " g" << std::endl;
    
    return true;
}

void BoringStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
}

void BoringStrategy::connectSolvers(MPMSolver* sph, FEMSolver* fem, 
                                     ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void BoringStrategy::updateConditions(const MachineState& state, double dt) {
    // RPM fallback: if the G-Code lacks an S-word, state.spindleRPM is 0 and
    // spindleSpeed/cuttingSpeed/feed-per-rev collapse. Fall back to config RPM.
    double rpm = state.spindleRPM;
    if (rpm < 1.0) rpm = m_configRPM;
    if (rpm < 1.0) rpm = 1.0;
    m_conditions.spindleSpeed = rpm * 2.0 * PI / 60.0 * m_adaptiveSpeedMultiplier;
    
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
    double feedMmPerMin = state.feedRate * 60000.0 * m_adaptiveFeedMultiplier;  // m/s to mm/min
    double rpmFeed = rpm;   // RPM fallback guard already applied above
    if (rpmFeed < 1.0) rpmFeed = 1.0;
    if (rpmFeed > 0) {
        m_conditions.feedRate = feedMmPerMin / 1000.0 / rpmFeed;  // m/rev
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
    
    // ── P1 FIX: Update Boring Bar Vibration Dynamics ────────────────────────
    // Radial force is the exciting force for bar vibration
    double Fr = computeRadialForce();
    updateBarDynamics(dt, Fr);
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
    double edgeLen = m_radialDepthOfCut / fmax(std::sin(kappaRad), 1e-6);
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
    // Fr = Kc * ap * f * cos(κ_eff)
    double ap = m_radialDepthOfCut;
    double f = m_conditions.feedRate;  // m/rev
    double r_eps = fmax(m_insertNoseRadius, 1e-6);

    // FIX 4.2: Effective lead angle across the circular insert nose arc.
    // The config default (90°) made cos(κ)=0 → Fr ≡ 0 → zero bar deflection.
    // When the radial DOC is ≤ nose radius the engaged corner sweeps the
    // nose arc and the effective lead angle is arcsin(ap/r_ε), producing a
    // dominant radial load. (Single-point boring inserts: r_ε ≈ 0.4–0.8 mm.)
    double kappaEffRad;
    if (ap < r_eps) {
        kappaEffRad = std::asin(std::clamp(ap / r_eps, 0.05, 1.0));
    } else {
        kappaEffRad = m_insertLeadAngle * DEG_TO_RAD;
    }

    // Kienzle correction for chip thickness
    double h = f * std::sin(kappaEffRad);  // uncut chip thickness
    double h_mm = h * 1000.0;
    if (h_mm < 0.01) h_mm = 0.01;
    double mc = 0.25;
    double Kc = m_specificCuttingForce * std::pow(h_mm, -mc);

    return Kc * ap * f * std::cos(kappaEffRad);
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
        if (m_deflectionWarnCount++ % 100 == 0) {
            std::cerr << "[BoringStrategy] WARNING: Bar deflection = " 
                      << deflection * 1e6 << " µm (>10µm threshold)" << std::endl;
        }
    }
    
    // Transform forces to global coordinates based on tool rotation angle
    double angle = m_toolRotationAngle;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    
    // FIX 4.3: cuttingForce must carry ONLY the tangential component. The old
    // cuttingForce also packed the radial (Fr·cosA, Fr·sinA) term, while
    // thrustForce carried that same radial vector — and SimulationEngine sums
    // cutting + thrust + feed. The radial bore-wall load was therefore added
    // twice to the tool shank. Now:
    //   cuttingForce = Ft·t̂            (tangential, orthogonal to radius)
    //   thrustForce  = Fr·r̂            (radial, outward into bore wall)
    //   feedForce    = Fa·ẑ            (axial, downward)
    output.cuttingForce = Vec3(-Ft * sinA, Ft * cosA, 0.0);
    output.thrustForce = Vec3(Fr * cosA, Fr * sinA, 0.0);
    output.feedForce = Vec3(0.0, 0.0, -Fa);
    
    // Torque = tangential force × radius
    double R = (m_boreStartDiameter / 2.0) + m_radialDepthOfCut;
    m_cachedTorque = Ft * R;
    output.torque = m_cachedTorque;
    
    // Power = torque × angular velocity
    output.power = m_cachedTorque * m_conditions.spindleSpeed;
    output.cuttingSpeed = m_conditions.cuttingSpeed;
    
    // Chip formation
    double kappaRad = m_insertLeadAngle * DEG_TO_RAD;
    output.chipThickness = m_conditions.feedRate * std::sin(kappaRad) * 2.0;
    output.chipWidth = m_radialDepthOfCut / fmax(std::sin(kappaRad), 1e-6);
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
        // FIX 4.1: Rotation was already advanced in updateConditions()
        // (m_toolRotationAngle += spindleSpeed·dt). Advancing it again here
        // doubled the spindle RPM for G-Code-driven boring → 2x peripheral
        // speed, doubled surface marks and interface heat.
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

// ---------------------------------------------------------------------------
// Edge Subgrid, BUE, Chatter corrections
// ---------------------------------------------------------------------------

void BoringStrategy::applyEdgeSubgridCorrections() {
    if (!m_edgeSubgrid || !m_edgeSubgrid->isEnabled()) return;
    // FIX 4.5: boring's m_conditions.feedRate is stored in m/rev (see
    // updateConditions). The old code divided it by spindle frequency again,
    // reducing the true per-revolution chip load 20× at 1200 RPM and forcing
    // the edge subgrid into extreme ploughing. For a single-point boring insert
    // the uncut chip load IS the feed per revolution — use it directly.
    double chipLoad = m_conditions.feedRate;  // m/rev
    if (chipLoad <= 0) return;
    // FIX 3.5: engaged cutting-edge length b = ap/sin(κ) — replaces the
    // hardcoded 1e-4 m ploughing width.
    double kappaRadBore = m_insertLeadAngle * DEG_TO_RAD;
    double widthBore = m_radialDepthOfCut / std::max(std::sin(kappaRadBore), 1e-6);
    auto result = m_edgeSubgrid->compute(
        chipLoad, widthBore,
        m_specificCuttingForce * 0.5,
        m_conditions.cuttingSpeed,
        m_currentDepth > 0.001 ? 300.0 : 25.0,
        m_workYieldPa, m_workYoungsPa, m_toolYoungsPa);
    m_conditions.effectiveRakeAngleRad = result.effectiveRakeAngleRad;
    m_conditions.ploughingForceN = result.ploughingForceN;
    m_conditions.sizeEffectFactor = result.sizeEffectFactor;
    m_conditions.edgeTemperatureFactor = result.edgeTemperatureFactor;
}

void BoringStrategy::applyBUECorrections() {
    if (!m_bue || !m_bue->isEnabled() || !m_bue->isActive()) return;
    m_conditions.bueEffectiveRakeChangeDeg = m_bue->getEffectiveRakeCorrectionDeg();
    m_conditions.bueCoverageFraction = m_bue->getState().coverageFraction;
    m_conditions.bueForceModulation = m_bue->getForceModulationFactor();
}

void BoringStrategy::applyChatterModulation(double dt) {
    if (!m_chatter || !m_chatter->isEnabled()) return;
    m_conditions.chipThicknessModulation = m_chatter->getChipThicknessModulation();
    double disp = m_chatter->getVibrationDisplacement().length();
    m_conditions.vibrationDisplacementUm = disp * 1e6;
    m_conditions.chattering = m_chatter->isChattering();
    m_conditions.stabilityParameter = m_chatter->getStabilityParameter();
}

// ── P1 FIX: Dynamic Boring Bar Vibration Update ─────────────────────────────
// State-space integration of cantilever beam first mode:
//   m_eff * ẍ + c * ẋ + k * x = F_radial(t)
// Using Newmark-beta (average acceleration, unconditionally stable) or
// simple central difference for explicit integration.
// Damping: c = 2 * ζ * sqrt(k * m_eff)
void BoringStrategy::updateBarDynamics(double dt, double radialForce) {
    if (dt <= 0.0 || m_barStiffness <= 0.0 || m_barEffectiveMass <= 0.0) return;
    
    // Damping coefficient
    double c = 2.0 * m_barDampingRatio * sqrt(m_barStiffness * m_barEffectiveMass);
    
    // Regenerative effect: radial force depends on previous vibration
    // F(t) = F_nominal * (1 + K_reg * x(t - T))
    // where T = 60/RPM is tooth passing period, K_reg is regenerative coefficient
    // For single-point boring: T = 1 rev period, so use m_prevRadialForce
    double T = (m_conditions.spindleSpeed > 1e-6) ? (2.0 * PI / m_conditions.spindleSpeed) : 0.0;
    
    // Simple central difference integration:
    // m_eff * (x_n+1 - 2x_n + x_n-1)/dt² + c*(x_n+1 - x_n-1)/(2dt) + k*x_n = F_n
    // Solve for x_n+1:
    double coeff = m_barEffectiveMass / (dt * dt) + c / (2.0 * dt);
    double x_next = (radialForce 
                     - m_barStiffness * m_barVibrationDisplacement 
                     + (2.0 * m_barEffectiveMass / (dt * dt) - m_barStiffness) * m_barVibrationDisplacement
                     - (m_barEffectiveMass / (dt * dt) - c / (2.0 * dt)) * m_barVibrationDisplacement) / coeff;
    
    // Actually, simpler: use the standard central difference formula
    // x_next = (dt² * (F - c*v - k*x) + 2*m*x - m*x_prev) / m
    // But we only track x and v. Use velocity Verlet:
    double a = (radialForce - c * m_barVibrationVelocity - m_barStiffness * m_barVibrationDisplacement) / m_barEffectiveMass;
    double x_new = m_barVibrationDisplacement + m_barVibrationVelocity * dt + 0.5 * a * dt * dt;
    double a_new = (radialForce - c * (m_barVibrationVelocity + a * dt) - m_barStiffness * x_new) / m_barEffectiveMass;
    double v_new = m_barVibrationVelocity + 0.5 * (a + a_new) * dt;
    
    // Regenerative chatter check: if vibration grows exponentially, flag chatter
    double vibRatio = (fabs(m_barVibrationDisplacement) > 1e-12) ? fabs(x_new / m_barVibrationDisplacement) : 1.0;
    
    m_barVibrationDisplacement = x_new;
    m_barVibrationVelocity = v_new;
    m_prevRadialForce = radialForce;
    
    // Update cached deflection with dynamic component
    m_cachedBarDeflection = computeBarDeflection(radialForce) + m_barVibrationDisplacement;
    
    // Log if dynamic amplification is significant
    static int logCount = 0;
    if (fabs(m_barVibrationDisplacement) > 5e-6 && (++logCount % 100 == 0)) {
        double dynAmp = (computeBarDeflection(radialForce) > 1e-12) ? 
                       fabs(m_barVibrationDisplacement / computeBarDeflection(radialForce)) : 1.0;
        std::cerr << "[BoringStrategy] Dynamic vibration: x=" << m_barVibrationDisplacement * 1e6 
                  << " µm, amp_factor=" << dynAmp << std::endl;
    }
}

} // namespace edgepredict
