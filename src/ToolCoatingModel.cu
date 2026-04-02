/**
 * @file ToolCoatingModel.cu
 * @brief Implementation of multi-layer tool coating wear model
 * 
 * Wear models implemented:
 * - Flank wear (VB): Archard-type abrasive wear
 * - Crater wear (KT): Diffusion-dominated, Arrhenius temperature dependence
 * 
 * References:
 * - Archard (1953) - Wear theory
 * - Usui (1984) - Tool wear equation for metal cutting
 */

#include "ToolCoatingModel.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace edgepredict {

// Constants
const double GAS_CONSTANT = 8.314;  // J/(mol·K)

ToolCoatingModel::ToolCoatingModel() = default;
ToolCoatingModel::~ToolCoatingModel() = default;

void ToolCoatingModel::initialize(int numNodes) {
    m_numNodes = numNodes;
    m_nodeStates.resize(numNodes);
    
    // Initialize all nodes with fresh coating
    for (auto& state : m_nodeStates) {
        state = NodeCoatingState();
    }
    
    std::cout << "[ToolCoatingModel] Initialized for " << numNodes << " nodes" << std::endl;
}

void ToolCoatingModel::addLayer(const CoatingLayer& layer) {
    m_layers.push_back(layer);
}

void ToolCoatingModel::addLayer(const std::string& name, double thickness, double hardness) {
    // Industrial defaults for multi-layer coatings (TiAlN / TiN based)
    double k = 15.0;            // Average conductivity W/(m·K)
    double maxT = 800.0;        // Oxidation resistance typical
    double adhesion = 100.0;    // Adhesion strength MPa
    double wearRes = 1.0;       // Base wear resistance
    
    m_layers.emplace_back(name, thickness, hardness, k, maxT, adhesion, wearRes);
}

void ToolCoatingModel::setupTiAlNCoating() {
    m_layers.clear();
    
    // Outer layer: TiAlN (excellent wear resistance)
    m_layers.emplace_back("TiAlN", 4e-6,  // 4 microns
                          3300,           // HV3300
                          5.0,            // W/(m·K)
                          800,            // Max temp 800°C
                          150,            // Adhesion 150 MPa
                          2.0);           // 2x wear resistance
    
    // Middle layer: TiN (good adhesion)
    m_layers.emplace_back("TiN", 2e-6,    // 2 microns
                          2000,           // HV2000
                          20.0,           // W/(m·K)
                          600,            // Max temp 600°C
                          200,            // Adhesion 200 MPa
                          1.5);           // 1.5x wear resistance
    
    // Substrate: WC-Co carbide
    m_layers.emplace_back("WC-Co", 1e-3,  // Effectively infinite
                          1500,           // HV1500
                          80.0,           // W/(m·K)
                          1000,           // Max temp 1000°C
                          0,              // N/A
                          1.0);           // Baseline
    
    std::cout << "[ToolCoatingModel] Setup TiAlN/TiN/WC-Co coating stack" << std::endl;
}

void ToolCoatingModel::setupTiNCoating() {
    m_layers.clear();
    
    m_layers.emplace_back("TiN", 3e-6,    // 3 microns
                          2000,           // HV2000
                          20.0,           // W/(m·K)
                          600,            // Max temp
                          200, 1.5);
    
    m_layers.emplace_back("WC-Co", 1e-3,
                          1500, 80.0, 1000, 0, 1.0);
}

void ToolCoatingModel::setupUncoatedCarbide() {
    m_layers.clear();
    m_layers.emplace_back("WC-Co", 1e-3,
                          1500, 80.0, 1000, 0, 1.0);
}

void ToolCoatingModel::setupDiamondCoating() {
    m_layers.clear();
    
    // CVD Diamond coating (excellent for aluminum, composites)
    m_layers.emplace_back("CVD-Diamond", 10e-6,  // 10 microns
                          10000,          // HV10000 (extremely hard)
                          2000,           // W/(m·K) (excellent thermal)
                          700,            // Max temp 700°C (graphitizes)
                          50,             // Lower adhesion
                          5.0);           // 5x wear resistance
    
    m_layers.emplace_back("WC-Co", 1e-3,
                          1500, 80.0, 1000, 0, 1.0);
    
    std::cout << "[ToolCoatingModel] Setup CVD-Diamond/WC-Co coating" << std::endl;
}

void ToolCoatingModel::updateWear(int nodeId, double contactPressure, double slidingVelocity,
                                   double temperature, WearZone zone, double dt) {
    if (nodeId < 0 || nodeId >= m_numNodes) return;
    if (contactPressure <= 0 || slidingVelocity <= 0) return;
    
    NodeCoatingState& state = m_nodeStates[nodeId];
    state.zone = zone;
    state.contactTemperature = temperature;
    
    // Get current active layer
    int layerIdx = state.activeLayerIndex;
    if (layerIdx >= static_cast<int>(m_layers.size())) return;
    
    // Compute wear rate for current layer
    double wearRate = computeWearRate(layerIdx, contactPressure, slidingVelocity, temperature);
    
    // Apply wear increment
    double wearIncrement = wearRate * dt;
    state.wearDepth += wearIncrement;
    
    // Update zone-specific wear metrics
    if (zone == WearZone::FLANK_FACE || zone == WearZone::CUTTING_EDGE) {
        // Flank wear: VB is the wear land width
        // Approximate: VB ≈ wear_depth / tan(clearance_angle)
        // For typical 7° clearance: tan(7°) ≈ 0.12
        state.flankWearWidth += wearIncrement / 0.12;
    } else if (zone == WearZone::RAKE_FACE) {
        // Crater wear: KT is the depth of crater
        // Diffusion wear is temperature-dependent
        double tempK = temperature + 273.15;
        double diffusionFactor = exp(-m_craterActivationEnergy / (GAS_CONSTANT * tempK));
        state.craterDepth += wearIncrement * diffusionFactor * 1e6;  // Enhanced for crater
    }
    
    // Check if current layer is worn through
    checkLayerTransition(nodeId);
}

double ToolCoatingModel::computeWearRate(int layerIndex, double pressure, double velocity,
                                          double temperature) const {
    if (layerIndex >= static_cast<int>(m_layers.size())) return 0;
    
    const CoatingLayer& layer = m_layers[layerIndex];
    
    // Archard wear equation: V = K * F * s / H
    //   V = wear volume
    //   K = wear coefficient
    //   F = normal force
    //   s = sliding distance
    //   H = hardness
    //
    // As rate: dV/dt = K * P * v / H
    //   P = pressure (force/area)
    //   v = sliding velocity
    
    double baseRate = m_flankWearCoeff * pressure * velocity / layer.hardness;
    
    // Apply layer's wear resistance factor
    baseRate /= layer.wearResistance;
    
    // Temperature acceleration factor (wear increases with temperature)
    if (temperature > layer.maxTemperature) {
        // Coating degraded, wear accelerates
        double overheat = (temperature - layer.maxTemperature) / layer.maxTemperature;
        baseRate *= (1.0 + 5.0 * overheat);  // Up to 5x faster wear when overheated
    }
    
    return baseRate;
}

void ToolCoatingModel::checkLayerTransition(int nodeId) {
    NodeCoatingState& state = m_nodeStates[nodeId];
    int layerIdx = state.activeLayerIndex;
    
    if (layerIdx >= static_cast<int>(m_layers.size())) return;
    
    // Check if we've worn through the current layer
    double wornInCurrentLayer = state.wearDepth;
    
    // Subtract thickness of all previous layers
    for (int i = 0; i < layerIdx; ++i) {
        wornInCurrentLayer -= m_layers[i].thickness;
    }
    
    // If worn through current layer, transition to next
    if (wornInCurrentLayer > m_layers[layerIdx].thickness) {
        if (layerIdx < static_cast<int>(m_layers.size()) - 1) {
            state.activeLayerIndex++;
            std::cout << "[ToolCoatingModel] Node " << nodeId 
                      << ": Layer worn through, now on " 
                      << m_layers[state.activeLayerIndex].name << std::endl;
        }
    }
}

double ToolCoatingModel::getWearDepth(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return 0;
    return m_nodeStates[nodeId].wearDepth;
}

double ToolCoatingModel::getCraterWear(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return 0;
    return m_nodeStates[nodeId].craterDepth;
}

double ToolCoatingModel::getFlankWear(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return 0;
    return m_nodeStates[nodeId].flankWearWidth;
}

int ToolCoatingModel::getActiveLayer(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return 0;
    return m_nodeStates[nodeId].activeLayerIndex;
}

bool ToolCoatingModel::isSubstrateExposed(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return false;
    // Substrate is the last layer
    return m_nodeStates[nodeId].activeLayerIndex >= static_cast<int>(m_layers.size()) - 1;
}

double ToolCoatingModel::getMaxFlankWear() const {
    double maxVB = 0;
    for (const auto& state : m_nodeStates) {
        maxVB = std::max(maxVB, state.flankWearWidth);
    }
    return maxVB;
}

double ToolCoatingModel::getMaxCraterWear() const {
    double maxKT = 0;
    for (const auto& state : m_nodeStates) {
        maxKT = std::max(maxKT, state.craterDepth);
    }
    return maxKT;
}

double ToolCoatingModel::getEffectiveHardness(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return 1500;  // Default WC-Co
    int layerIdx = m_nodeStates[nodeId].activeLayerIndex;
    if (layerIdx >= static_cast<int>(m_layers.size())) return 1500;
    return m_layers[layerIdx].hardness;
}

double ToolCoatingModel::getEffectiveThermalConductivity(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return 80.0;  // Default WC-Co
    int layerIdx = m_nodeStates[nodeId].activeLayerIndex;
    if (layerIdx >= static_cast<int>(m_layers.size())) return 80.0;
    return m_layers[layerIdx].thermalConductivity;
}

void ToolCoatingModel::reset() {
    for (auto& state : m_nodeStates) {
        state = NodeCoatingState();
    }
    // Reset layer thicknesses
    for (auto& layer : m_layers) {
        layer.currentThickness = layer.thickness;
    }
}

std::string ToolCoatingModel::getCoatingDescription() const {
    std::string desc;
    for (size_t i = 0; i < m_layers.size(); ++i) {
        if (i > 0) desc += " / ";
        desc += m_layers[i].name;
        if (m_layers[i].thickness < 1e-4) {
            desc += " (" + std::to_string(int(m_layers[i].thickness * 1e6)) + "μm)";
        }
    }
    return desc;
}

} // namespace edgepredict
