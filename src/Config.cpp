/**
 * @file Config.cpp
 * @brief Configuration parsing implementation
 */

#include "Config.h"
#include <fstream>
#include <iostream>

namespace edgepredict {

void Config::loadFromFile(const std::string& path) {
    m_configPath = path;
    m_errors.clear();
    
    std::ifstream file(path);
    if (!file.is_open()) {
        m_errors.push_back("Cannot open config file: " + path);
        m_isValid = false;
        throw std::runtime_error(m_errors.back());
    }
    
    try {
        json j = json::parse(file);
        parseJson(j);
        validate();
    } catch (const json::parse_error& e) {
        m_errors.push_back("JSON parse error: " + std::string(e.what()));
        m_isValid = false;
        throw std::runtime_error(m_errors.back());
    }
    
    std::cout << "[Config] Loaded: " << path << std::endl;
}

void Config::loadFromString(const std::string& jsonStr) {
    m_errors.clear();
    
    try {
        json j = json::parse(jsonStr);
        parseJson(j);
        validate();
    } catch (const json::parse_error& e) {
        m_errors.push_back("JSON parse error: " + std::string(e.what()));
        m_isValid = false;
        throw std::runtime_error(m_errors.back());
    }
}

void Config::parseJson(const json& j) {
    m_json = j;
    
    // Simulation name
    m_simulationName = j.value("simulation_name", "EdgePredict Simulation");
    
    // Machining type
    std::string machType = j.value("machining_type", "milling");
    if (machType == "milling") m_machining.type = MachiningType::MILLING;
    else if (machType == "drilling") m_machining.type = MachiningType::DRILLING;
    else if (machType == "reaming") m_machining.type = MachiningType::REAMING;
    else if (machType == "threading") m_machining.type = MachiningType::THREADING;
    else if (machType == "boring") m_machining.type = MachiningType::BORING;
    else m_machining.type = MachiningType::MILLING;
    
    // Simulation parameters
    if (j.contains("simulation_parameters")) {
        auto& sp = j["simulation_parameters"];
        m_simulation.numSteps = sp.value("num_steps", 1000);
        m_simulation.timeStepDuration = sp.value("time_step_duration_s", 1e-6);
        m_simulation.outputIntervalSteps = sp.value("output_interval_steps", 100);
        m_simulation.minTimeStep = sp.value("min_time_step_s", 1e-12);
        m_simulation.maxTimeStep = sp.value("max_time_step_s", 1e-4);
        m_simulation.previewSetup = sp.value("preview_setup", false);
    }
    
    // Machining parameters
    if (j.contains("machining_parameters")) {
        auto& mp = j["machining_parameters"];
        m_machining.rpm = mp.value("rpm", 1000.0);
        m_machining.feedRateMmMin = mp.value("feed_rate_mm_min", 300.0);
        m_machining.depthOfCutMm = mp.value("depth_of_cut_mm", 2.0);
    }
    
    // File paths
    if (j.contains("file_paths")) {
        auto& fp = j["file_paths"];
        m_filePaths.toolGeometry = fp.value("tool_geometry", "");
        m_filePaths.workpieceGeometry = fp.value("workpiece_geometry", "");
        m_filePaths.outputDirectory = fp.value("output_directory", "output");
        m_filePaths.outputResults = fp.value("output_results", "results.json");
        m_filePaths.gcodeFile = fp.value("gcode_file", "");
    }
    
    // G-Code file fallback: also check top-level for backward compatibility
    if (m_filePaths.gcodeFile.empty()) {
        m_filePaths.gcodeFile = j.value("gcode_file", "");
    }
    
    // Material properties
    if (j.contains("material_properties")) {
        auto& mat = j["material_properties"];
        m_material.name = mat.value("name", "Unknown");
        m_material.density = mat.value("density_kg_m3", constants::DEFAULT_DENSITY);
        m_material.youngsModulus = mat.value("youngs_modulus_Pa", 113.8e9);
        m_material.specificHeat = mat.value("specific_heat_J_kgK", constants::DEFAULT_SPECIFIC_HEAT);
        m_material.thermalConductivity = mat.value("thermal_conductivity_W_mK", 
                                                    constants::DEFAULT_THERMAL_CONDUCTIVITY);
        m_material.meltingPoint = mat.value("melting_point_C", constants::DEFAULT_MELTING_POINT);
        
        // Johnson-Cook plasticity
        if (mat.contains("johnson_cook_plasticity")) {
            auto& jc = mat["johnson_cook_plasticity"];
            m_material.jc_A = jc.value("A_yield_strength_MPa", 880.0) * 1e6;
            m_material.jc_B = jc.value("B_strain_hardening_MPa", 290.0) * 1e6;
            m_material.jc_n = jc.value("n_strain_hardening_exp", 0.47);
            m_material.jc_C = jc.value("C_strain_rate_sensitivity", 0.015);
            m_material.jc_m = jc.value("m_thermal_softening_exp", 1.0);
        }
        
        // Failure parameters
        m_material.yieldStrength = mat.value("yield_strength_MPa", 880.0) * 1e6;
        m_material.failureStrain = mat.value("failure_strain", 0.3);
        m_material.fractureToughness = mat.value("fracture_toughness_MPa_sqrt_m", 55.0) * 1e6;
    }
    
    // Tool material properties (separate from workpiece!)
    if (j.contains("tool_material")) {
        auto& tm = j["tool_material"];
        m_toolMaterial.name = tm.value("name", "Carbide");
        m_toolMaterial.density = tm.value("density_kg_m3", 14500.0);
        m_toolMaterial.youngsModulus = tm.value("youngs_modulus_GPa", 600.0) * 1e9;
        m_toolMaterial.poissonsRatio = tm.value("poissons_ratio", 0.22);
        m_toolMaterial.specificHeat = tm.value("specific_heat_J_kgK", 200.0);
        m_toolMaterial.thermalConductivity = tm.value("thermal_conductivity_W_mK", 80.0);
        m_toolMaterial.meltingPoint = tm.value("melting_point_C", 2870.0);
        m_toolMaterial.yieldStrength = tm.value("yield_strength_GPa", 4.0) * 1e9;
        
        // Usui wear parameters (tool-specific)
        if (tm.contains("usui_wear")) {
            auto& usui = tm["usui_wear"];
            m_toolMaterial.usui_A = usui.value("A", 1e-9);
            m_toolMaterial.usui_B = usui.value("B", 1000.0);
        }
        
        // Coating
        m_toolMaterial.coating = tm.value("coating", "none");
        m_toolMaterial.coatingThickness = tm.value("coating_thickness_um", 0.0) * 1e-6;
    }
    
    // Physics parameters (legacy compatibility)
    if (j.contains("physics_parameters")) {
        auto& pp = j["physics_parameters"];
        m_machining.ambientTemperature = pp.value("ambient_temperature_C", 25.0);
    }
    
    // SPH parameters
    if (j.contains("sph_parameters")) {
        auto& sp = j["sph_parameters"];
        m_sph.smoothingRadius = sp.value("smoothing_radius_m", 0.0001);
        m_sph.gasStiffness = sp.value("gas_stiffness", 3000.0);
        m_sph.maxParticles = sp.value("max_particles", 100000);
        
        // LOD parameters
        m_sph.lodEnabled = sp.value("lod_enabled", true);
        m_sph.lodActiveRadius = sp.value("lod_active_radius_m", 0.002);
        m_sph.lodNearRadius = sp.value("lod_near_radius_m", 0.01);
        m_sph.lodNearSkipSteps = sp.value("lod_near_skip_steps", 5);
        m_sph.lodFarSkipSteps = sp.value("lod_far_skip_steps", 20);
        
        // Damage/Chip Separation parameters
        m_sph.damageEnabled = sp.value("damage_enabled", true);
        m_sph.jc_D1 = sp.value("jc_D1", 0.05);
        m_sph.jc_D2 = sp.value("jc_D2", 3.44);
        m_sph.jc_D3 = sp.value("jc_D3", -2.12);
        m_sph.jc_D4 = sp.value("jc_D4", 0.002);
        m_sph.jc_D5 = sp.value("jc_D5", 0.61);
        m_sph.damageThreshold = sp.value("damage_threshold", 1.0);
        m_sph.referenceStrainRate = sp.value("reference_strain_rate", 1.0);
    }
    
    // FEM parameters
    if (j.contains("fem_parameters")) {
        auto& fp = j["fem_parameters"];
        m_fem.youngModulus = fp.value("youngs_modulus_Pa", 200e9);
        m_fem.dampingRatio = fp.value("damping_ratio", 0.1);
        m_fem.maxNodes = fp.value("max_nodes", 50000);
        m_fem.massScalingFactor = fp.value("mass_scaling_factor", 1.0);
        m_fem.stiffnessScalingFactor = fp.value("stiffness_scaling_factor", 1.0);
    }
    
    // CFD parameters
    if (j.contains("cfd_parameters")) {
        auto& cp = j["cfd_parameters"];
        m_cfd.enabled = cp.value("enabled", false);
        m_cfd.gridX = cp.value("grid_x", 50);
        m_cfd.gridY = cp.value("grid_y", 50);
        m_cfd.gridZ = cp.value("grid_z", 50);
        m_cfd.cellSize = cp.value("cell_size_m", 0.001);
        
        // Coolant type and properties
        m_cfd.coolantType = cp.value("coolant_type", "Water-Glycol");
        m_cfd.inletVelocity = cp.value("inlet_velocity_m_s", 1.0);
        m_cfd.inletTemperature = cp.value("inlet_temperature_C", 20.0);
        
        // Custom coolant properties
        if (cp.contains("custom_properties") || m_cfd.coolantType == "Custom") {
            auto& custom = cp.contains("custom_properties") ? cp["custom_properties"] : cp;
            m_cfd.fluidDensity = custom.value("density_kg_m3", 1050.0);
            m_cfd.dynamicViscosity = custom.value("dynamic_viscosity_Pa_s", 0.003);
            m_cfd.fluidSpecificHeat = custom.value("specific_heat_J_kgK", 3800.0);
            m_cfd.fluidThermalConductivity = custom.value("thermal_conductivity_W_mK", 0.5);
        }
    }
    
    // Optimization parameters
    if (j.contains("optimization")) {
        auto& opt = j["optimization"];
        m_optimization.enabled = opt.value("enabled", false);
        
        // Limits
        m_optimization.maxStress = opt.value("max_stress_GPa", 2.0) * 1e9;
        m_optimization.maxTemperature = opt.value("max_temperature_C", 800.0);
        m_optimization.maxWear = opt.value("max_wear_mm", 0.3) * 1e-3;
        m_optimization.maxForce = opt.value("max_force_N", 5000.0);
        
        // PID for feed
        if (opt.contains("feed_pid")) {
            auto& pid = opt["feed_pid"];
            m_optimization.feedPID_Kp = pid.value("Kp", 0.5);
            m_optimization.feedPID_Ki = pid.value("Ki", 0.05);
            m_optimization.feedPID_Kd = pid.value("Kd", 0.01);
        }
        
        // PID for speed
        if (opt.contains("speed_pid")) {
            auto& pid = opt["speed_pid"];
            m_optimization.speedPID_Kp = pid.value("Kp", 0.3);
            m_optimization.speedPID_Ki = pid.value("Ki", 0.02);
            m_optimization.speedPID_Kd = pid.value("Kd", 0.005);
        }
    }
    
    // Machine setup parameters (Anchored Physics)
    if (j.contains("machine_setup")) {
        auto& ms = j["machine_setup"];
        
        // G54-G59 work offsets (new multi-WCS format)
        const char* wcsKeys[] = {"G54", "G55", "G56", "G57", "G58", "G59"};
        for (int i = 0; i < 6; ++i) {
            if (ms.contains(wcsKeys[i])) {
                auto offsets = ms[wcsKeys[i]].get<std::vector<double>>();
                for (size_t k = 0; k < std::min(offsets.size(), (size_t)3); ++k) {
                    m_machineSetup.workOffsets[i][k] = offsets[k] / 1000.0; // mm to m
                }
            }
        }
        
        // Legacy single G54 key (backward compat)
        if (ms.contains("work_offset_G54")) {
            auto offsets = ms["work_offset_G54"].get<std::vector<double>>();
            for (size_t i = 0; i < std::min(offsets.size(), (size_t)3); ++i) {
                m_machineSetup.workOffsetG54[i] = offsets[i] / 1000.0;
                m_machineSetup.workOffsets[0][i] = offsets[i] / 1000.0; // Copy to G54 slot
            }
        }
        
        // Tool length offset
        m_machineSetup.toolLengthOffset = ms.value("tool_length_offset_mm", 0.0) / 1000.0;
        
        // Spindle dynamics (opt-in)
        m_machineSetup.enableSpindleDynamics = ms.value("enable_spindle_dynamics", false);
        m_machineSetup.spindleStiffness = ms.value("spindle_stiffness_N_m", 5e7);
        m_machineSetup.spindleDamping = ms.value("spindle_damping_N_s_m", 1e4);
        
        // Fixturing
        m_machineSetup.fixtureLayerThickness = ms.value("fixture_layer_thickness_mm", 2.0) / 1000.0;
        
        // Tool alignment
        m_machineSetup.autoAlignToolTip = ms.value("auto_align_tool_tip", true);
        
        // Alignment axis: auto-detect from machining type when -1
        std::string axisStr = ms.value("align_axis", "auto");
        if (axisStr == "X" || axisStr == "x") m_machineSetup.alignAxis = 0;
        else if (axisStr == "Y" || axisStr == "y") m_machineSetup.alignAxis = 1;
        else if (axisStr == "Z" || axisStr == "z") m_machineSetup.alignAxis = 2;
        else m_machineSetup.alignAxis = -1; // auto
        
        // Driven node fraction for spindle coupling
        m_machineSetup.drivenNodeFraction = ms.value("driven_node_fraction", 0.2);
        
        std::cout << "[Config] Machine setup loaded:"
                  << " G54=[" << m_machineSetup.workOffsets[0][0]*1000 
                  << "," << m_machineSetup.workOffsets[0][1]*1000 
                  << "," << m_machineSetup.workOffsets[0][2]*1000 << "]mm"
                  << " TLO=" << m_machineSetup.toolLengthOffset*1000 << "mm"
                  << " SpindleDyn=" << (m_machineSetup.enableSpindleDynamics ? "ON" : "OFF")
                  << std::endl;
        
        // Log all configured WCS offsets
        for (int i = 0; i < 6; ++i) {
            if (std::abs(m_machineSetup.workOffsets[i][0]) > 1e-9 ||
                std::abs(m_machineSetup.workOffsets[i][1]) > 1e-9 ||
                std::abs(m_machineSetup.workOffsets[i][2]) > 1e-9) {
                std::cout << "  G" << (54+i) << " = ["
                          << m_machineSetup.workOffsets[i][0]*1000 << ", "
                          << m_machineSetup.workOffsets[i][1]*1000 << ", "
                          << m_machineSetup.workOffsets[i][2]*1000 << "] mm" << std::endl;
            }
        }
    }
    
    // Workpiece geometry parameters (replaces hardcoded dimensions)
    if (j.contains("workpiece_geometry")) {
        auto& wg = j["workpiece_geometry"];
        m_workpieceGeometry.shape = wg.value("shape", "auto");
        m_workpieceGeometry.radiusMm = wg.value("radius_mm", 0.0);
        m_workpieceGeometry.lengthMm = wg.value("length_mm", 0.0);
        m_workpieceGeometry.widthMm = wg.value("width_mm", 0.0);
        m_workpieceGeometry.heightMm = wg.value("height_mm", 0.0);
        m_workpieceGeometry.depthMm = wg.value("depth_mm", 0.0);
        
        std::cout << "[Config] Workpiece geometry: shape=" << m_workpieceGeometry.shape;
        if (m_workpieceGeometry.radiusMm > 0)
            std::cout << " R=" << m_workpieceGeometry.radiusMm << "mm";
        if (m_workpieceGeometry.lengthMm > 0)
            std::cout << " L=" << m_workpieceGeometry.lengthMm << "mm";
        std::cout << std::endl;
    } else if (j.contains("machining_parameters")) {
        // Legacy: read shape from machining_parameters
        auto& mp = j["machining_parameters"];
        if (mp.contains("workpiece_shape")) {
            m_workpieceGeometry.shape = mp["workpiece_shape"].get<std::string>();
        }
    }
}

void Config::validate() {
    m_errors.clear();
    m_isValid = true;
    
    // Validate optional fields (warnings only)
    if (m_filePaths.toolGeometry.empty()) {
        std::cerr << "[Config] Note: No tool_geometry specified (will use solver defaults)" << std::endl;
    }
    
    // Validate ranges
    if (m_simulation.numSteps <= 0) {
        m_errors.push_back("simulation_parameters.num_steps must be positive");
        m_isValid = false;
    }
    
    if (m_simulation.timeStepDuration <= 0) {
        m_errors.push_back("simulation_parameters.time_step_duration_s must be positive");
        m_isValid = false;
    }
    
    if (m_machining.rpm < 0) {
        m_errors.push_back("machining_parameters.rpm cannot be negative");
        m_isValid = false;
    }
    
    if (m_material.density <= 0) {
        m_errors.push_back("material_properties.density_kg_m3 must be positive");
        m_isValid = false;
    }
    
    // Warnings (not errors)
    if (m_simulation.numSteps > 10000000) {
        std::cerr << "[Config] Warning: Very large num_steps (" 
                  << m_simulation.numSteps << ") - simulation may take a long time" 
                  << std::endl;
    }
    
    if (m_isValid) {
        std::cout << "[Config] Validation passed" << std::endl;
        std::cout << "  Workpiece: " << m_material.name << std::endl;
        std::cout << "  Tool: " << m_toolMaterial.name << std::endl;
        std::cout << "  Steps: " << m_simulation.numSteps << std::endl;
        std::cout << "  RPM: " << m_machining.rpm << std::endl;
        if (m_cfd.enabled) {
            std::cout << "  CFD: Enabled (" << m_cfd.coolantType << ")" << std::endl;
        }
        if (m_optimization.enabled) {
            std::cout << "  Optimization: Enabled" << std::endl;
        }
    } else {
        std::cerr << "[Config] Validation FAILED:" << std::endl;
        for (const auto& err : m_errors) {
            std::cerr << "  - " << err << std::endl;
        }
    }
}

} // namespace edgepredict
