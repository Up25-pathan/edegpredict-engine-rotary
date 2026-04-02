/**
 * @file ToolCoatingModel.cuh
 * @brief Multi-layer tool coating model with progressive wear
 * 
 * Modern cutting tools use multi-layer coatings like:
 * - TiAlN (outer): High hardness, oxidation resistant up to 800°C
 * - TiN (inner): Good adhesion, moderate hardness
 * - WC-Co (substrate): Tough carbide base
 * 
 * As the outer layer wears through, the next layer is exposed with
 * different wear characteristics.
 */

#pragma once

#include "Types.h"
#include <vector>
#include <string>

namespace edgepredict {

/**
 * @brief Single coating layer properties
 */
struct CoatingLayer {
    std::string name;              // Material name (e.g., "TiAlN")
    double thickness;              // Layer thickness (m)
    double currentThickness;       // Remaining thickness (m)
    double hardness;               // Vickers hardness (HV)
    double thermalConductivity;    // W/(m·K)
    double maxTemperature;         // Max operating temp before degradation (°C)
    double adhesionStrength;       // Adhesion strength (MPa)
    double wearResistance;         // Relative wear resistance (higher = better)
    
    CoatingLayer(const std::string& n = "", double t = 0, double h = 0,
                 double k = 0, double maxT = 0, double adh = 0, double wr = 1.0)
        : name(n), thickness(t), currentThickness(t), hardness(h),
          thermalConductivity(k), maxTemperature(maxT), 
          adhesionStrength(adh), wearResistance(wr) {}
};

/* WearZone is now centrally defined in Types.h */

/**
 * @brief Per-node coating state (stored on each FEM tool node)
 */
struct NodeCoatingState {
    int activeLayerIndex;          // Currently exposed layer (0 = outermost)
    double wearDepth;              // Total wear depth (m)
    double craterDepth;            // Crater wear KT (m)
    double flankWearWidth;         // Flank wear VB (m)
    double contactTemperature;     // Interface temperature (°C)
    WearZone zone;                 // Which wear zone this node is in
    
    NodeCoatingState() 
        : activeLayerIndex(0), wearDepth(0), craterDepth(0), 
          flankWearWidth(0), contactTemperature(25.0), zone(WearZone::NONE) {}
};

/**
 * @brief Tool Coating Model manager
 * 
 * Manages multi-layer coating wear with:
 * - Layer-dependent wear rates
 * - Automatic layer transition when worn through
 * - Separate tracking of crater (KT) and flank (VB) wear
 */
class ToolCoatingModel {
public:
    ToolCoatingModel();
    ~ToolCoatingModel();
    
    /**
     * @brief Initialize with standard coating stack
     * @param numNodes Number of tool FEM nodes
     */
    void initialize(int numNodes);
    
    /**
     * @brief Add a coating layer (add from outside to inside)
     */
    void addLayer(const CoatingLayer& layer);
    
    /**
     * @brief Simplified addLayer helper
     */
    void addLayer(const std::string& name, double thickness, double hardness);
    
    /**
     * @brief Set up common coating configurations
     */
    void setupTiAlNCoating();       // TiAlN over TiN over WC-Co
    void setupTiNCoating();         // TiN over WC-Co
    void setupUncoatedCarbide();    // Just WC-Co (no coating)
    void setupDiamondCoating();     // CVD Diamond over WC-Co
    
    /**
     * @brief Update wear for a specific node
     * @param nodeId Tool node ID
     * @param contactPressure Normal pressure at contact (Pa)
     * @param slidingVelocity Relative sliding velocity (m/s)
     * @param temperature Interface temperature (°C)
     * @param zone Which wear zone
     * @param dt Time step (s)
     */
    void updateWear(int nodeId, double contactPressure, double slidingVelocity,
                   double temperature, WearZone zone, double dt);
    
    /**
     * @brief Get total wear depth at node
     */
    double getWearDepth(int nodeId) const;
    
    /**
     * @brief Get crater wear depth KT at node
     */
    double getCraterWear(int nodeId) const;
    
    /**
     * @brief Get flank wear width VB at node
     */
    double getFlankWear(int nodeId) const;
    
    /**
     * @brief Get currently active layer at node
     */
    int getActiveLayer(int nodeId) const;
    
    /**
     * @brief Check if substrate is exposed at node
     */
    bool isSubstrateExposed(int nodeId) const;
    
    /**
     * @brief Get maximum flank wear across all nodes
     */
    double getMaxFlankWear() const;
    
    /**
     * @brief Get maximum crater depth across all nodes
     */
    double getMaxCraterWear() const;
    
    /**
     * @brief Get effective hardness at node (depends on active layer)
     */
    double getEffectiveHardness(int nodeId) const;
    
    /**
     * @brief Get effective thermal conductivity at node
     */
    double getEffectiveThermalConductivity(int nodeId) const;
    
    /**
     * @brief Reset all coating to initial state
     */
    void reset();
    
    /**
     * @brief Get coating stack description
     */
    std::string getCoatingDescription() const;

private:
    /**
     * @brief Compute wear rate for current layer
     * Uses Archard-type wear model modified for coating
     */
    double computeWearRate(int layerIndex, double pressure, double velocity, 
                           double temperature) const;
    
    /**
     * @brief Check and update layer transitions
     */
    void checkLayerTransition(int nodeId);
    
    std::vector<CoatingLayer> m_layers;
    std::vector<NodeCoatingState> m_nodeStates;
    int m_numNodes = 0;
    
    // Wear model coefficients
    double m_flankWearCoeff = 1e-12;      // Flank wear K coefficient
    double m_craterWearCoeff = 5e-13;     // Crater wear K coefficient
    double m_craterActivationEnergy = 180000; // J/mol for diffusion wear
};

} // namespace edgepredict
