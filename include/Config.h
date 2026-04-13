#pragma once
/**
 * @file Config.h
 * @brief Configuration management for EdgePredict Engine
 */

#include "Types.h"
#include "json.hpp"
#include <string>
#include <map>
#include <array>
#include <memory>
#include <stdexcept>

namespace edgepredict {

// Use fully-qualified name to avoid conflicts
using json = nlohmann::json;

/**
 * @brief Workpiece material properties from configuration
 */
struct MaterialProperties {
    std::string name = "Ti-6Al-4V";
    double density = constants::DEFAULT_DENSITY;           // kg/m³
    double youngsModulus = 113.8e9;                        // Pa
    double poissonsRatio = 0.34;
    double yieldStrength = 880e6;                          // Pa
    double specificHeat = constants::DEFAULT_SPECIFIC_HEAT; // J/(kg·K)
    double thermalConductivity = constants::DEFAULT_THERMAL_CONDUCTIVITY; // W/(m·K)
    double meltingPoint = constants::DEFAULT_MELTING_POINT; // °C
    
    // Johnson-Cook plasticity parameters (for any material)
    double jc_A = 880e6;   // Yield strength (Pa)
    double jc_B = 290e6;   // Strain hardening (Pa)
    double jc_n = 0.47;    // Strain hardening exponent
    double jc_C = 0.015;   // Strain rate sensitivity
    double jc_m = 1.0;     // Thermal softening exponent
    
    // Failure parameters
    double failureStrain = 0.3;
    double fractureToughness = 55e6;  // Pa·m^0.5
};

/**
 * @brief Tool material properties - separate from workpiece
 */
struct ToolMaterialProperties {
    std::string name = "Carbide";
    double density = 14500;                    // kg/m³ (tungsten carbide)
    double youngsModulus = 600e9;              // Pa
    double poissonsRatio = 0.22;
    double specificHeat = 200;                 // J/(kg·K)
    double thermalConductivity = 80;           // W/(m·K)
    double meltingPoint = 2870;                // °C
    double yieldStrength = 4e9;                // Pa (compressive)
    
    // Usui wear parameters (tool-specific)
    double usui_A = 1e-9;
    double usui_B = 1000.0;
    
    // Tool coating (optional)
    std::string coating = "none";              // TiN, TiAlN, AlCrN, etc.
    double coatingThickness = 0;               // m
};

/**
 * @brief Simulation parameters
 */
struct SimulationParams {
    int numSteps = 1000;
    double timeStepDuration = 1e-6;      // seconds
    int outputIntervalSteps = 100;
    double minTimeStep = 1e-12;
    double maxTimeStep = 1e-4;
    bool previewSetup = false;
};

/**
 * @brief Machining parameters
 */
struct MachiningParams {
    MachiningType type = MachiningType::MILLING;
    double rpm = 1000.0;
    double feedRateMmMin = 300.0;
    double depthOfCutMm = 2.0;
    double ambientTemperature = 25.0;    // °C
};

/**
 * @brief SPH solver parameters
 */
struct SPHParams {
    double smoothingRadius = 0.0001;     // m (100 microns)
    double gasStiffness = 3000.0;
    int maxParticles = 100000;
    double particleSpacingFactor = 0.8;  // Relative to smoothing radius
    
    // Level of Detail (LOD) parameters
    bool lodEnabled = true;
    double lodActiveRadius = 0.002;      // 2mm - full physics zone
    double lodNearRadius = 0.01;         // 10mm - reduced updates zone  
    int lodNearSkipSteps = 5;            // Update every N steps in NEAR zone
    int lodFarSkipSteps = 20;            // Update every N steps in FAR zone
    
    // Damage/Chip Separation Model (Johnson-Cook failure)
    // Default values calibrated for AISI 4140 Steel (common structural machining material)
    // Override via sph_parameters.jc_D1..D5 in config JSON for other materials
    bool damageEnabled = true;
    double jc_D1 = 0.06;                  // JC failure D1 (AISI 4140)
    double jc_D2 = 3.31;                  // JC failure D2 (AISI 4140)
    double jc_D3 = -1.96;                 // JC failure D3 (AISI 4140)
    double jc_D4 = 0.018;                 // JC failure D4 (AISI 4140)
    double jc_D5 = 0.58;                  // JC failure D5 (AISI 4140)
    double damageThreshold = 1.0;         // Damage threshold for separation
    double referenceStrainRate = 1.0;     // Reference strain rate (1/s)
};

/**
 * @brief FEM solver parameters
 */
struct FEMParams {
    double youngModulus = 200e9;         // Tool material (carbide)
    double dampingRatio = 0.1;
    double massScalingFactor = 1.0;      // Increase for speed (default: no scaling)
    double stiffnessScalingFactor = 1.0; // Decrease for speed (default: no scaling)
    int maxNodes = 50000;
    int maxElements = 200000;
    
    // Tool Coating Model
    bool coatingEnabled = false;
    double coatingThickness1 = 4e-6;     // First layer (TiAlN) 4 microns
    double coatingThickness2 = 2e-6;     // Second layer (TiN) 2 microns
    double coatingHardness1 = 3300;      // HV for TiAlN
    double coatingHardness2 = 2500;      // HV for TiN
    double substrateHardness = 1500;     // HV for WC-Co substrate
    
    // Wear Model
    bool wearModelEnabled = true;
    double flankWearCoeff = 1e-12;       // Flank wear coefficient
    double craterWearActivation = 180000;// Crater wear activation energy (J/mol)
};

/**
 * @brief CFD solver parameters (coolant simulation)
 */
struct CFDParams {
    bool enabled = false;
    int gridX = 50;                          // Grid cells in X
    int gridY = 50;                          // Grid cells in Y
    int gridZ = 50;                          // Grid cells in Z
    double cellSize = 0.001;                 // m
    
    // Coolant properties (configurable from input!)
    std::string coolantType = "Water-Glycol"; // Water, Oil, MWF, Emulsion, Custom
    double inletVelocity = 1.0;              // m/s
    double inletTemperature = 20.0;          // °C
    
    // Custom coolant properties (used when coolantType = "Custom")
    double fluidDensity = 1050.0;            // kg/m³
    double dynamicViscosity = 0.003;         // Pa·s
    double fluidSpecificHeat = 3800.0;       // J/(kg·K)
    double fluidThermalConductivity = 0.5;   // W/(m·K)
};

/**
 * @brief Optimization/adaptive control parameters
 */
struct OptimizationParams {
    bool enabled = false;
    
    // Limits
    double maxStress = 2e9;                  // Pa
    double maxTemperature = 800.0;           // °C
    double maxWear = 0.3e-3;                 // m
    double maxForce = 5000;                  // N
    
    // PID gains for feed control
    double feedPID_Kp = 0.5;
    double feedPID_Ki = 0.05;
    double feedPID_Kd = 0.01;
    
    // PID gains for speed control
    double speedPID_Kp = 0.3;
    double speedPID_Ki = 0.02;
    double speedPID_Kd = 0.005;
};

/**
 * @brief Machine setup parameters (CNC environment for tool simulation precision)
 * 
 * Mimics a CNC controller's offset page. Every field has a safe default
 * so existing configs without "machine_setup" run identically to before.
 */
/**
 * @brief Workpiece geometry parameters (replaces hardcoded dimensions)
 */
struct WorkpieceGeometryParams {
    std::string shape = "auto";          // "auto", "cylinder", "box"
    double radiusMm = 0;                 // Cylinder radius (mm), 0 = use default
    double lengthMm = 0;                 // Cylinder length or box depth (mm)
    double widthMm = 0;                  // Box width X (mm)
    double heightMm = 0;                 // Box height Y (mm)
    double depthMm = 0;                  // Box depth Z (mm)
};

struct MachineSetupParams {
    // Work Coordinate Systems (WCS) - mimics G54-G59 offset page on CNC controller
    // 6 registers: G54(0) through G59(5) — each stores [X, Y, Z] in meters
    std::array<std::array<double, 3>, 6> workOffsets = {{}};
    
    // Legacy alias for backward compatibility
    double workOffsetG54[3] = {0.0, 0.0, 0.0};
    
    // Tool length compensation (H-register value)
    double toolLengthOffset = 0.0;                 // meters (from spindle face to tip)
    
    // Spindle dynamics (only active when enableSpindleDynamics = true)
    double spindleStiffness = 5e7;                 // N/m (typical: 2e7 - 1e8)
    double spindleDamping = 1e4;                   // N·s/m
    bool enableSpindleDynamics = false;             // Opt-in: false = rigid tool (safe default)
    
    // Fixturing
    double fixtureLayerThickness = 0.002;          // meters (bottom 2mm of workpiece)
    
    // Tool alignment
    bool autoAlignToolTip = true;                   // Auto-center tool tip to origin
    int alignAxis = -1;                             // -1 = auto-detect from machining type
    
    // Driven node configuration (for spindle dynamics)
    double drivenNodeFraction = 0.2;               // Top 20% of tool = collet region
};

/**
 * @brief File paths configuration
 */
struct FilePaths {
    std::string toolGeometry;
    std::string workpieceGeometry;
    std::string gcodeFile;
    std::string outputDirectory = "output";
    std::string outputResults = "results.json";
};

/**
 * @brief Main configuration class
 * 
 * Provides validated access to all simulation parameters.
 * Parses JSON configuration files.
 */
class Config {
public:
    Config() = default;
    ~Config() = default;
    
    /**
     * @brief Load configuration from JSON file
     * @param path Path to JSON configuration file
     * @throws std::runtime_error if file cannot be loaded or parsed
     */
    void loadFromFile(const std::string& path);
    
    /**
     * @brief Load configuration from JSON string
     * @param jsonStr JSON string
     */
    void loadFromString(const std::string& jsonStr);
    
    /**
     * @brief Check if configuration is valid
     */
    bool isValid() const { return m_isValid; }
    
    /**
     * @brief Get validation errors
     */
    const std::vector<std::string>& getErrors() const { return m_errors; }
    
    // Accessors
    const std::string& getSimulationName() const { return m_simulationName; }
    const MaterialProperties& getMaterial() const { return m_material; }
    const SimulationParams& getSimulation() const { return m_simulation; }
    const MachiningParams& getMachining() const { return m_machining; }
    const SPHParams& getSPH() const { return m_sph; }
    const FEMParams& getFEM() const { return m_fem; }
    const CFDParams& getCFD() const { return m_cfd; }
    const FilePaths& getFilePaths() const { return m_filePaths; }
    const ToolMaterialProperties& getToolMaterial() const { return m_toolMaterial; }
    const OptimizationParams& getOptimization() const { return m_optimization; }
    const MachineSetupParams& getMachineSetup() const { return m_machineSetup; }
    const WorkpieceGeometryParams& getWorkpieceGeometry() const { return m_workpieceGeometry; }
    const json& getJson() const { return m_json; }
    
    // Convenience accessors
    MachiningType getMachiningType() const { return m_machining.type; }
    double getTimeStep() const { return m_simulation.timeStepDuration; }
    int getNumSteps() const { return m_simulation.numSteps; }
    
private:
    void parseJson(const nlohmann::json& j);
    void validate();
    
    std::string m_simulationName = "EdgePredict Simulation";
    std::string m_configPath;
    
    MaterialProperties m_material;
    SimulationParams m_simulation;
    MachiningParams m_machining;
    SPHParams m_sph;
    FEMParams m_fem;
    CFDParams m_cfd;
    FilePaths m_filePaths;
    ToolMaterialProperties m_toolMaterial;
    OptimizationParams m_optimization;
    MachineSetupParams m_machineSetup;
    WorkpieceGeometryParams m_workpieceGeometry;
    
    bool m_isValid = false;
    std::vector<std::string> m_errors;
    
    // Store raw json for custom access
    json m_json;
};

} // namespace edgepredict
