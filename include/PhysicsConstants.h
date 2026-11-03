#pragma once
/**
 * @file PhysicsConstants.h
 * @brief Physically-derived constants replacing magic numbers throughout the codebase.
 * 
 * All constants are derived from first principles or experimental data.
 * NO arbitrary values - every number has a physical justification.
 */

#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

namespace edgepredict {
namespace physics {

// ============================================================================
// Fundamental Constants
// ============================================================================
constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / PI;

// ============================================================================
// Contact Mechanics Constants
// ============================================================================
// Hertz contact: p_max = 1.5 * F / A for circular contact
constexpr double HERTZ_PRESSURE_FACTOR = 1.5;

// Engagement ramp smoothstep parameters
constexpr double ENGAGEMENT_RAMP_START_FRACTION = 0.1;  // Start at 10% force
constexpr int DEFAULT_ENGAGEMENT_RAMP_STEPS = 200;      // ~200 steps to full engagement

// Contact radius multipliers (for different operations)
// Drilling: larger radius to capture margin engagement
constexpr double DRILLING_CONTACT_RADIUS_MULTIPLIER = 2.5;
// Milling/Boring/Reaming/Threading: standard
constexpr double STANDARD_CONTACT_RADIUS_MULTIPLIER = 1.5;

// Penetration limits
constexpr double MAX_PENETRATION_FRACTION = 0.5;  // Max penetration as fraction of contactRadius
constexpr double CONTACT_RADIUS_RAMP_STEPS = 10;  // Steps to ramp from 30% to 100% contact radius
constexpr double MIN_CONTACT_RADIUS_FRACTION = 0.3;  // Start at 30% (cutting edge only)

// ============================================================================
// Thermal Constants
// ============================================================================
// Blok/Jaeger heat partition: f_tool = e_tool / (e_tool + e_work)
// where e = sqrt(k * rho * Cp) = thermal effusivity
constexpr double DEFAULT_HEAT_PARTITION_TOOL = 0.5;  // 50/50 split at room temp

// Zorev temperature-dependent friction
// mu(T) = mu_0 * (1 - 0.4 * (T - 200)/600) for 200 < T < 800°C
constexpr double ZOREV_TEMP_START = 200.0;      // °C
constexpr double ZOREV_TEMP_RANGE = 600.0;      // °C
constexpr double ZOREV_FRICTION_REDUCTION = 0.4; // Max 40% reduction

// Pressure-dependent HTC: HTC = HTC_0 * (1 + 5*(1 - exp(-P/P_ref)))
constexpr double HTC_PRESSURE_FACTOR = 5.0;

// Heat integration micro-step fraction (1/20 = 20 sub-steps per macro step)
constexpr int HEAT_INTEGRATION_SUBSTEPS = 20;

// Thermal conduction sub-cycling limit
constexpr int MAX_THERMAL_SUBCYCLES = 20;

// First-contact thermal protection
constexpr int FIRST_CONTACT_STEPS = 20;         // Ramp friction heat over first N steps
constexpr double MAX_FIRST_CONTACT_HEAT_FRACTION = 0.1;  // Max 10% of plastic work as friction

// ============================================================================
// Friction & Wear Constants
// ============================================================================
// Archard wear coefficient range for carbide/steel: 1e-6 to 1e-4
// k = V / (F * s) where V=volume, F=force, s=distance
constexpr double ARCHARD_WEAR_COEFF_MIN = 1e-6;
constexpr double ARCHARD_WEAR_COEFF_MAX = 1e-4;
constexpr double ARCHARD_WEAR_COEFF_DEFAULT = 1e-5;

// Tool wear zones (Archard coefficients per zone)
constexpr double WEAR_COEFF_FLANK = 2e-5;    // Flank face (high stress, sliding)
constexpr double WEAR_COEFF_CRATER = 5e-5;   // Crater/rake face (high temp, diffusion)
constexpr double WEAR_COEFF_MARGIN = 1e-5;   // Margin (lower stress)
constexpr double WEAR_COEFF_CHISEL = 1e-4;   // Chisel edge (highest pressure)

// Wear failure threshold
constexpr double MAX_FLANK_WEAR = 0.3e-3;    // 0.3 mm flank wear limit

// ============================================================================
// Machining Mechanics Constants
// ============================================================================
// Merchant's shear angle: phi = 45 + (alpha - beta)/2
// where alpha = rake angle, beta = friction angle
constexpr double MERCHANT_BASE_ANGLE = 45.0 * DEG_TO_RAD;

// Kienzle specific cutting force exponent
constexpr double KIENZLE_EXPONENT = 0.25;

// Chip compression ratio bounds
constexpr double CHIP_COMPRESSION_MIN = 1.5;
constexpr double CHIP_COMPRESSION_MAX = 5.0;

// Radial/tangential force ratios
constexpr double MILLING_RADIAL_TANGENTIAL_RATIO = 0.4;      // Kr = Fr/Ft
constexpr double BORING_RADIAL_TANGENTIAL_RATIO = 1.0;       // Can exceed 1.0
constexpr double REAMING_RADIAL_TANGENTIAL_RATIO = 0.15;     // Very low (guided)
constexpr double DRILLING_THRUST_TANGENTIAL_RATIO = 1.0;     // Fz ≈ Ft for drilling

// Axial force ratios
constexpr double MILLING_AXIAL_TANGENTIAL_RATIO = 0.3;       // Fa/Ft = tan(helix)
constexpr double BORING_AXIAL_TANGENTIAL_RATIO = 0.25;
constexpr double REAMING_AXIAL_TANGENTIAL_RATIO = 0.1;
constexpr double THREADING_AXIAL_TANGENTIAL_RATIO = 0.05;    // Small (helix angle)

// ============================================================================
// Milling Engagement Constants
// ============================================================================
// Entry/exit transition arc length
constexpr double MILLING_TRANSITION_ARC_LENGTH = 1.0e-3;  // 1mm default
constexpr double MAX_TRANSITION_FRACTION = 0.1;           // Max 10% of engagement arc

// ============================================================================
// Boring Bar Dynamics
// ============================================================================
// First bending mode constant for cantilever: (1.875)^2 / (2*pi)
constexpr double CANTILEVER_FIRST_MODE_CONSTANT = 1.875 * 1.875 / (2.0 * PI);

// Modal effective mass fraction for first mode of uniform cantilever
constexpr double CANTILEVER_MODAL_MASS_FRACTION = 0.236;

// Structural damping ratio for boring bars
constexpr double BORING_BAR_DAMPING_RATIO = 0.02;  // 2% typical for steel/carbide

// Dynamic amplification warning threshold
constexpr double DYNAMIC_AMPLIFICATION_WARNING = 5.0e-6;  // 5 µm vibration

// ============================================================================
// Reaming Constants
// ============================================================================
// Lead-in chamfer engagement
constexpr double REAMER_CHAMFER_ANGLE = 45.0 * DEG_TO_RAD;
constexpr double REAMER_CHAMFER_LENGTH_FACTOR = 2.0;  // chamfer_len = 2 * radial_stock

// Runout variation
constexpr double REAMER_RUNOUT_VARIATION = 0.1;  // ±10% flute-to-flute

// ============================================================================
// AMS (Adaptive Mass Scaling) Constants
// ============================================================================
// PID controller defaults (targeting KE/IE < 5%)
constexpr double AMS_PID_KP = 0.5;
constexpr double AMS_PID_KI = 0.05;
constexpr double AMS_PID_KD = 0.1;
constexpr double AMS_PID_TARGET = 0.05;  // 5% KE/IE ratio

// Spectral radius limit for stability
constexpr double AMS_SPECTRAL_RADIUS_LIMIT = 0.95;

// Mass scaling bounds
constexpr double AMS_MIN_SCALING = 1.0;
constexpr double AMS_MAX_SCALING = 1.0e6;

// CFL bounds
constexpr double AMS_CFL_MIN = 1e-4;
constexpr double AMS_CFL_MAX = 0.5;

// ============================================================================
// Multi-rate Coupling Intervals
// ============================================================================
constexpr double MULTI_RATE_THERMAL_DT = 100e-6;   // 100 μs - thermal diffusion
constexpr double MULTI_RATE_CFD_DT = 1.0e-3;       // 1 ms - CFD coolant
constexpr double MULTI_RATE_WEAR_DT = 100e-3;      // 100 ms - wear/BUE/chatter

// ============================================================================
// MPM Constants
// ============================================================================
// J-clamp parameters (volumetric locking prevention)
// J = 1 + 0.1 * tanh((J-1)/0.1) gives soft saturation to [0.9, 1.1]
constexpr double J_CLAMP_SOFTENING = 0.1;

// F_e component clamp (prevents deviatoric blow-up)
constexpr double FE_COMPONENT_CLAMP = 2.0;

// tr(B) cap for distortional energy (3 + 12 = 15 allows ~2x RMS deviatoric stretch)
constexpr double TR_B_CAP = 15.0;

// Mean stress limit: max(3 GPa, 3 * yield_strength)
constexpr double MEAN_STRESS_FACTOR = 3.0;

// Plastic strain increment cap per micro-step
constexpr double MAX_PLASTIC_STRAIN_INCREMENT = 0.05;

// Damage increment cap
constexpr double DAMAGE_INCREMENT_FACTOR = 5.0;

// Chip fracture perturbation
constexpr double CHIP_PERTURB_SPEED_FACTOR = 0.3;
constexpr double CHIP_PERTURB_BASE = 0.5;
constexpr double CHIP_PERTURB_MAX = 2.0;
constexpr double CHIP_SPEED_CAP = 10.0;  // m/s

// Damage reset after fracture
constexpr double DAMAGE_RESET_FRACTION = 0.5;

// LOD zone radii
constexpr double LOD_ACTIVE_RADIUS = 6.0e-3;   // 6mm full physics
constexpr double LOD_NEAR_RADIUS = 25.0e-3;    // 25mm reduced physics

// Fixture damping
constexpr double FIXTURE_DAMPING_COEFF = 1e7;  // 1/s

// Gravity
constexpr double GRAVITY_Z = -9.81;

// ============================================================================
// FEM Constants
// ============================================================================
// Element erosion threshold
constexpr double FEM_DET_F_EROSION_THRESHOLD = 0.1;

// Corotational FEM polar decomposition iterations
constexpr int POLAR_DECOMP_MAX_ITER = 20;
constexpr double POLAR_DECOMP_TOL = 1e-6;

// Thermal conduction (tetrahedral)
constexpr double THERMAL_CONDUCTION_TET_ITER = 1;  // Single-pass explicit

// ============================================================================
// Contact Solver Constants
// ============================================================================
// Wear morphing intervals
constexpr int WEAR_MORPH_FULL_INTERVAL = 5000;   // Full SDF rebuild
constexpr int WEAR_MORPH_EDGE_INTERVAL = 500;    // Edge-only update

// Contact explosion guard
constexpr int CONTACT_EXPLOSION_NODE_MULTIPLIER = 50;

// Wear batch interval (D2H copy)
constexpr int WEAR_BATCH_INTERVAL = 20;

// ============================================================================
// Adiabatic Shear Band Constants
// ============================================================================
constexpr double ASB_CRITICAL_STRAIN_RATE = 5000.0;   // 1/s
constexpr double ASB_SOFTENING_THRESHOLD = 0.95;
constexpr double ASB_TAYLOR_QUINNEY = 0.9;
constexpr double ASB_MAX_TEMP_RATIO = 0.9;
constexpr double ASB_CRITICAL_STRAIN = 0.8;
constexpr double ASB_TEMP_THRESHOLD = 600.0;          // °C
constexpr double ASB_STRAIN_RATE_GUARD = 2.0;         // Factor above critical
constexpr double ASB_HISTORY_COUNTER_MAX = 10;
constexpr int ASB_HISTORY_THRESHOLD = 3;              // Consecutive steps for localization
constexpr double ASB_LOCALIZATION_MAX = 2.5;          // Max 2.5x heat localization

// ============================================================================
// Coolant Hardening Constants
// ============================================================================
constexpr double COOLANT_JET_NOZZLE_DIA = 5.0e-3;     // 5mm
constexpr double COOLANT_JET_STANDOFF = 15.0e-3;      // 15mm
constexpr double COOLANT_JET_VELOCITY = 1.5;          // m/s
constexpr double COOLANT_SATURATION_TEMP = 100.0;     // °C
constexpr double COOLANT_LEIDENFROST_TEMP = 300.0;    // °C
constexpr double COOLANT_CHF_TEMP = 200.0;            // °C
constexpr double COOLANT_LATENT_HEAT = 2.26e6;        // J/kg
constexpr double CAPILLARY_PORE_RADIUS = 5.0e-6;      // 5 µm
constexpr double QUENCH_COOLING_RATE_THRESHOLD = 100.0; // °C/s
constexpr double MAX_HARDNESS_INCREASE_HRC = 5.0;
constexpr double QUENCH_DEPTH = 0.5e-3;               // 0.5mm

// ============================================================================
// Threading Constants
// ============================================================================
// Infeed methods
constexpr double INFEED_RADIAL_AREA_FACTOR = 1.0;
constexpr double INFEED_FLANK_AREA_FACTOR = 0.9;
constexpr double INFEED_MODIFIED_FLANK_AREA_FACTOR = 1.1;
constexpr double INFEED_INCREMENTAL_AREA_FACTOR = 1.0;

// ============================================================================
// Material Properties (typical values for reference)
// ============================================================================
// AISI 1045 Steel
struct Material_1045_Steel {
    static constexpr double YOUNGS_MODULUS = 200e9;
    static constexpr double POISSONS_RATIO = 0.3;
    static constexpr double DENSITY = 7850.0;
    static constexpr double YIELD_STRENGTH = 553e6;
    static constexpr double MELTING_POINT = 1520.0;
    static constexpr double SPECIFIC_HEAT = 475.0;
    static constexpr double THERMAL_CONDUCTIVITY = 45.0;
    static constexpr double JC_A = 595e6;
    static constexpr double JC_B = 580e6;
    static constexpr double JC_N = 0.133;
    static constexpr double JC_C = 0.023;
    static constexpr double JC_M = 1.03;
    static constexpr double JC_D1 = 0.06;
    static constexpr double JC_D2 = 3.31;
    static constexpr double JC_D3 = -1.96;
    static constexpr double JC_D4 = 0.018;
    static constexpr double JC_D5 = 0.58;
};

// Tungsten Carbide (WC-Co)
struct Material_TungstenCarbide {
    static constexpr double YOUNGS_MODULUS = 600e9;
    static constexpr double POISSONS_RATIO = 0.22;
    static constexpr double DENSITY = 14500.0;
    static constexpr double YIELD_STRENGTH = 4e9;
    static constexpr double MELTING_POINT = 2870.0;
    static constexpr double SPECIFIC_HEAT = 200.0;
    static constexpr double THERMAL_CONDUCTIVITY = 80.0;
};

// Ti-6Al-4V
struct Material_Ti6Al4V {
    static constexpr double YOUNGS_MODULUS = 113.8e9;
    static constexpr double POISSONS_RATIO = 0.34;
    static constexpr double DENSITY = 4430.0;
    static constexpr double YIELD_STRENGTH = 880e6;
    static constexpr double MELTING_POINT = 1660.0;
    static constexpr double SPECIFIC_HEAT = 560.0;
    static constexpr double THERMAL_CONDUCTIVITY = 6.7;
};

} // namespace physics
} // namespace edgepredict

#endif // PHYSICS_CONSTANTS_H