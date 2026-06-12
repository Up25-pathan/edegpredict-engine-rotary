#include "ToolRNDAnalyzer.h"
#include "BUEModel.h"
#include "ChatterDynamics.h"
#include "EdgeSubgridModel.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace edgepredict {

namespace {
double clamp01(double v) {
    return std::max(0.0, std::min(1.0, v));
}

double vonMisesParticle(const MPMParticle& p) {
    double mean = (p.stress_xx + p.stress_yy + p.stress_zz) / 3.0;
    double sxx = p.stress_xx - mean;
    double syy = p.stress_yy - mean;
    double szz = p.stress_zz - mean;
    return std::sqrt(1.5 * (sxx*sxx + syy*syy + szz*szz +
        2.0 * (p.stress_xy*p.stress_xy + p.stress_xz*p.stress_xz + p.stress_yz*p.stress_yz)));
}

nlohmann::json buildRegionJson(const ToolRegionReport& report) {
    nlohmann::json regions = nlohmann::json::array();
    for (const auto& agg : report.regions) {
        if (agg.nodeCount == 0) continue;
        nlohmann::json r;
        r["name"] = ToolRegionClassifier::regionName(agg.region);
        r["id"] = static_cast<int>(agg.region);
        r["node_count"] = agg.nodeCount;
        r["contact_node_count"] = agg.contactNodeCount;
        r["max_stress_MPa"] = agg.maxStress / 1e6;
        r["avg_stress_MPa"] = agg.avgStress / 1e6;
        r["max_temperature_C"] = agg.maxTemperature;
        r["avg_temperature_C"] = agg.avgTemperature;
        r["max_wear_um"] = agg.maxWear * 1e6;
        r["max_flank_wear_um"] = agg.maxFlankWear * 1e6;
        r["max_crater_wear_um"] = agg.maxCraterWear * 1e6;
        r["max_chipping_risk"] = agg.maxChippingRisk;
        r["min_coating_remaining_percent"] = agg.minCoatingRemaining * 100.0;
        regions.push_back(r);
    }
    return regions;
}

nlohmann::json buildGeometryJson(const Mesh& mesh) {
    nlohmann::json root;
    nlohmann::json regions;
    if (mesh.nodes.empty()) {
        root["diameter_mm_estimate"] = 0.0;
        root["axial_span_mm"] = 0.0;
        root["surface_node_count"] = 0;
        root["triangle_count"] = mesh.triangles.size();
        root["regions"] = regions;
        return root;
    }

    const int regionCount = static_cast<int>(ToolRegion::REGION_COUNT);
    struct Bounds {
        int count = 0;
        double minZ = 1e30, maxZ = -1e30;
        double minR = 1e30, maxR = 0.0;
    };
    std::vector<Bounds> bounds(regionCount);

    double globalMinZ = 1e30, globalMaxZ = -1e30, globalMaxR = 0.0;
    for (const auto& node : mesh.nodes) {
        int idx = static_cast<int>(node.toolRegion);
        if (idx < 0 || idx >= regionCount) idx = 0;
        double r = std::sqrt(node.position.x * node.position.x + node.position.y * node.position.y);
        auto& b = bounds[idx];
        b.count++;
        b.minZ = std::min(b.minZ, node.position.z);
        b.maxZ = std::max(b.maxZ, node.position.z);
        b.minR = std::min(b.minR, r);
        b.maxR = std::max(b.maxR, r);
        globalMinZ = std::min(globalMinZ, node.position.z);
        globalMaxZ = std::max(globalMaxZ, node.position.z);
        globalMaxR = std::max(globalMaxR, r);
    }

    root["diameter_mm_estimate"] = globalMaxR * 2000.0;
    root["axial_span_mm"] = (globalMaxZ - globalMinZ) * 1000.0;
    root["surface_node_count"] = mesh.nodes.size();
    root["triangle_count"] = mesh.triangles.size();

    for (int i = 0; i < regionCount; ++i) {
        const auto& b = bounds[i];
        if (b.count == 0) continue;
        nlohmann::json r;
        r["node_count"] = b.count;
        r["z_min_mm"] = b.minZ * 1000.0;
        r["z_max_mm"] = b.maxZ * 1000.0;
        r["radius_min_mm"] = b.minR * 1000.0;
        r["radius_max_mm"] = b.maxR * 1000.0;
        regions[ToolRegionClassifier::regionName(static_cast<ToolRegion>(i))] = r;
    }
    root["regions"] = regions;
    return root;
}

nlohmann::json buildFailureWearJson(const Mesh& mesh, const Config& config) {
    const auto& tool = config.getToolMaterial();
    const double ambient = config.getMachining().ambientTemperature;
    double yield = std::max(tool.yieldStrength, 1.0);
    double melt = std::max(tool.meltingPoint, ambient + 1.0);

    double maxStress = 0.0, maxTemp = 0.0, maxWear = 0.0;
    double maxChipRisk = 0.0, minCoating = 1.0, maxThermalRisk = 0.0;
    double maxFlank = 0.0, maxCrater = 0.0;
    std::string criticalRegion = "unknown";

    for (const auto& node : mesh.nodes) {
        maxStress = std::max(maxStress, node.stress);
        maxTemp = std::max(maxTemp, node.temperature);
        maxWear = std::max(maxWear, node.accumulatedWear);
        maxFlank = std::max(maxFlank, node.flankWear);
        maxCrater = std::max(maxCrater, node.craterWear);
        minCoating = std::min(minCoating, node.coatingRemaining);
        maxThermalRisk = std::max(
            maxThermalRisk,
            clamp01((node.temperature - ambient) / std::max(melt - ambient, 1.0)));
        if (node.chippingRisk > maxChipRisk) {
            maxChipRisk = node.chippingRisk;
            criticalRegion = ToolRegionClassifier::regionName(node.toolRegion);
        }
    }

    double stressUtil = maxStress / yield;
    double wearLimit = std::max(config.getOptimization().maxWear, 0.3e-3);
    double wearUtil = maxWear / wearLimit;
    double coatingFailureRisk = clamp01((1.0 - minCoating) * 0.75 + maxThermalRisk * 0.25);
    double designRisk = clamp01(0.38 * clamp01(stressUtil) +
                                0.22 * maxChipRisk +
                                0.18 * clamp01(wearUtil) +
                                0.14 * coatingFailureRisk +
                                0.08 * maxThermalRisk);

    nlohmann::json root;
    root["max_stress_MPa"] = maxStress / 1e6;
    root["stress_utilization"] = stressUtil;
    root["max_temperature_C"] = maxTemp;
    root["thermal_overload_risk"] = maxThermalRisk;
    root["max_wear_um"] = maxWear * 1e6;
    root["max_flank_wear_um"] = maxFlank * 1e6;
    root["max_crater_wear_um"] = maxCrater * 1e6;
    root["max_chipping_risk"] = maxChipRisk;
    root["critical_chipping_region"] = criticalRegion;
    root["min_coating_remaining_percent"] = minCoating * 100.0;
    root["coating_failure_risk"] = coatingFailureRisk;
    root["overall_design_risk_score"] = designRisk;
    root["risk_score_note"] =
        "0..1 engineering risk score derived from simulated stress, temperature, wear, coating remaining, and chipping risk; requires calibration for certified tool-life prediction.";
    return root;
}

nlohmann::json buildChipHeatJson(const ToolRNDAnalysisInput& input) {
    const auto& particles = *input.particles;
    const auto& config = *input.config;
    const double ambient = config.getMachining().ambientTemperature;
    const double physicalDensity = std::max(config.getMaterial().density, 1.0);
    const auto physicalMass = [physicalDensity](const MPMParticle& p) {
        if (p.volume > 1e-24) return physicalDensity * p.volume;
        if (p.density > 1.0) return p.mass * (physicalDensity / p.density);
        return p.mass;
    };

    int activeCount = 0, chipCount = 0, damagedCount = 0;
    double chipMass = 0.0, chipVolume = 0.0, chipTempMass = 0.0, chipMaxTemp = ambient;
    double chipSpeedMass = 0.0, chipDamage = 0.0, chipPlastic = 0.0;
    Vec3 chipMomentum;
    double activeThermalJ = 0.0, chipThermalJ = 0.0;

    for (const auto& p : particles) {
        bool active = p.status == ParticleStatus::ACTIVE;
        bool chip = p.status == ParticleStatus::CHIP;
        if (!active && !chip) continue;

        double dT = std::max(0.0, p.temperature - ambient);
        double pPhysicalMass = physicalMass(p);
        double thermal = pPhysicalMass * config.getMaterial().specificHeat * dT;
        if (active) {
            activeCount++;
            activeThermalJ += thermal;
        }
        if (chip) {
            chipCount++;
            chipMass += pPhysicalMass;
            chipVolume += p.volume;
            chipTempMass += p.temperature * pPhysicalMass;
            chipMaxTemp = std::max(chipMaxTemp, p.temperature);
            double speed = std::sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
            chipSpeedMass += speed * pPhysicalMass;
            chipMomentum += Vec3(p.vx, p.vy, p.vz) * pPhysicalMass;
            chipDamage += p.damage;
            chipPlastic += p.plasticStrain;
            chipThermalJ += thermal;
        }
        if (p.damage > 0.05 || vonMisesParticle(p) > config.getMaterial().yieldStrength) {
            damagedCount++;
        }
    }

    double toolThermalJ = 0.0;
    for (const auto& node : input.toolMesh->nodes) {
        double dT = std::max(0.0, node.temperature - ambient);
        toolThermalJ += node.mass * config.getToolMaterial().specificHeat * dT;
    }

    double totalThermalJ = std::max(1e-12, chipThermalJ + activeThermalJ + toolThermalJ);
    double chipRatio = chipMass > 0.0 ? chipTempMass / chipMass : ambient;
    double chipSpeed = chipMass > 0.0 ? chipSpeedMass / chipMass : 0.0;
    Vec3 evacuation = chipMomentum.length() > 1e-12 ? chipMomentum.normalized() : Vec3::zero();

    double theoreticalChipThickness =
        input.config->getMachining().feedRateMmMin /
        std::max(input.config->getMachining().rpm, 1.0) / 1000.0;
    double chipThicknessProxy = chipCount > 0
        ? std::cbrt(std::max(chipVolume / std::max(chipCount, 1), 0.0))
        : 0.0;
    double segmentationIndex = chipCount > 0
        ? clamp01((chipDamage / chipCount) * 0.55 + (chipPlastic / chipCount) * 0.45)
        : 0.0;

    nlohmann::json root;
    root["chip_particle_count"] = chipCount;
    root["active_particle_count"] = activeCount;
    root["damaged_or_yielded_particle_count"] = damagedCount;
    root["chip_mass_kg"] = chipMass;
    root["chip_volume_mm3"] = chipVolume * 1e9;
    root["mean_chip_temperature_C"] = chipRatio;
    root["max_chip_temperature_C"] = chipMaxTemp;
    root["mean_chip_velocity_m_s"] = chipSpeed;
    root["chip_thickness_proxy_mm"] = chipThicknessProxy * 1000.0;
    root["theoretical_feed_per_rev_mm"] = theoreticalChipThickness * 1000.0;
    root["segmentation_index"] = segmentationIndex;
    root["evacuation_direction"] = {evacuation.x, evacuation.y, evacuation.z};

    nlohmann::json heat;
    heat["tool_J"] = toolThermalJ;
    heat["chip_J"] = chipThermalJ;
    heat["workpiece_J"] = activeThermalJ;
    heat["contact_generated_J"] = input.contactHeatJ;
    heat["tool_percent"] = 100.0 * toolThermalJ / totalThermalJ;
    heat["chip_percent"] = 100.0 * chipThermalJ / totalThermalJ;
    heat["workpiece_percent"] = 100.0 * activeThermalJ / totalThermalJ;
    heat["note"] = "Thermal partition is estimated from stored sensible heat at export time; coolant removal and boundary losses are not yet closed-energy-calibrated.";
    root["heat_partition"] = heat;
    return root;
}
} // namespace

void ToolRNDAnalyzer::exportJson(const std::string& path,
                                 const ToolRNDAnalysisInput& input) {
    if (!input.toolMesh || !input.particles || !input.config) return;

    nlohmann::json root;
    root["schema"] = "edgepredict_tool_rnd_report_v1";
    root["simulation_time_s"] = input.currentTimeS;
    root["tool_family"] = input.regionReport.familyName;
    root["classification_confidence"] = input.regionReport.classificationConfidence;
    root["regions"] = buildRegionJson(input.regionReport);
    root["geometry"] = buildGeometryJson(*input.toolMesh);
    root["failure_wear"] = buildFailureWearJson(*input.toolMesh, *input.config);
    root["chip_heat"] = buildChipHeatJson(input);

    // Edge subgrid model
    if (input.edgeSubgrid) {
        nlohmann::json es;
        es["enabled"] = input.edgeSubgrid->isEnabled();
        es["edge_radius_um"] = input.edgeSubgrid->getEdgeRadius() * 1e6;
        root["edge_subgrid"] = es;
    }

    // BUE model
    if (input.bueModel) {
        nlohmann::json bue;
        bue["enabled"] = input.bueModel->isEnabled();
        bue["active"] = input.bueModel->isActive();
        const auto& s = input.bueModel->getState();
        bue["phase"] = static_cast<int>(s.phase);
        bue["coverage_fraction"] = s.coverageFraction;
        bue["layer_thickness_um"] = s.layerThicknessUm * 1e6;
        bue["breakoff_event"] = s.breakoffEvent;
        bue["roughness_penalty_um"] = s.roughnessPenaltyUm;
        bue["chipping_risk_increase"] = s.chippingRiskIncrease;
        bue["force_modulation"] = input.bueModel->getForceModulationFactor();
        root["bue_model"] = bue;
    }

    // Chatter dynamics
    if (input.chatter) {
        nlohmann::json ch;
        ch["enabled"] = input.chatter->isEnabled();
        ch["chattering"] = input.chatter->isChattering();
        ch["stability_parameter"] = input.chatter->getStabilityParameter();
        ch["chip_thickness_modulation"] = input.chatter->getChipThicknessModulation();
        auto disp = input.chatter->getVibrationDisplacement();
        ch["vibration_displacement_x_um"] = disp.x * 1e6;
        ch["vibration_displacement_y_um"] = disp.y * 1e6;
        ch["vibration_displacement_z_um"] = disp.z * 1e6;
        root["chatter_dynamics"] = ch;
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ToolRNDAnalyzer] Failed to write report: " << path << std::endl;
        return;
    }
    file << root.dump(2);
}

} // namespace edgepredict
