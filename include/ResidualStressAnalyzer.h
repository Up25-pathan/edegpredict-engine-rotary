/**
 * @file ResidualStressAnalyzer.h
 * @brief Analyzer for computing and outputting residual stress in machined surfaces
 * 
 * Residual stresses arise from:
 * 1. Mechanical plastic deformation
 * 2. Thermal gradients during cooling
 * 3. Phase transformations (for steel)
 * 
 * This analyzer computes stress profiles vs depth from machined surface,
 * which is critical for predicting fatigue life and part performance.
 */

#pragma once

#include "Types.h"
#include "SPHSolver.cuh"
#include <vector>
#include <string>

namespace edgepredict {

/**
 * @brief Stress profile point (stress vs depth from surface)
 */
struct StressProfilePoint {
    double depth;           // Distance from surface (m)
    double stressAxial;     // Stress in cutting direction (Pa)
    double stressFeed;      // Stress in feed direction (Pa)
    double stressRadial;    // Stress normal to surface (Pa)
    double stressVonMises;  // von Mises equivalent (Pa)
    double temperature;     // Temperature at this depth (°C)
    
    StressProfilePoint() 
        : depth(0), stressAxial(0), stressFeed(0), stressRadial(0),
          stressVonMises(0), temperature(25.0) {}
};

/**
 * @brief Surface region for local stress analysis
 */
struct SurfaceRegion {
    Vec3 center;            // Center of region
    Vec3 normal;            // Surface normal (pointing outward)
    double area;            // Region area (m²)
    double avgStress;       // Average surface stress (Pa)
    double maxStress;       // Maximum stress (Pa)
    double avgTemp;         // Average temperature (°C)
    
    SurfaceRegion()
        : center(0,0,0), normal(0,1,0), area(0), 
          avgStress(0), maxStress(0), avgTemp(25.0) {}
};

/**
 * @brief Residual Stress Analyzer
 * 
 * Analyzes stress distribution in machined workpiece surface.
 * Provides:
 * - Stress vs depth profiles
 * - Surface stress maps
 * - Stress statistics
 */
class ResidualStressAnalyzer {
public:
    ResidualStressAnalyzer();
    ~ResidualStressAnalyzer();
    
    /**
     * @brief Configure analyzer
     * @param maxDepth Maximum depth to analyze (m)
     * @param depthResolution Number of depth levels
     */
    void configure(double maxDepth, int depthResolution);
    
    /**
     * @brief Analyze stress distribution from SPH particles
     * @param particles Host-side particle array
     * @param surfaceNormal Normal vector of machined surface
     * @param surfacePoint A point on the machined surface
     */
    void analyzeFromParticles(const std::vector<SPHParticle>& particles,
                              const Vec3& surfaceNormal,
                              const Vec3& surfacePoint);
    
    /**
     * @brief High-level analysis (entry point for Engine)
     */
    void analyze(const Config& config);
    
    /**
     * @brief Get stress profile (stress vs depth)
     */
    const std::vector<StressProfilePoint>& getStressProfile() const { return m_profile; }
    
    /**
     * @brief Get surface stress (at depth=0)
     */
    double getSurfaceStress() const { return m_surfaceStress; }
    
    /**
     * @brief Get maximum compressive stress (negative = compressive)
     */
    double getMaxCompressiveStress() const { return m_maxCompressive; }
    
    /**
     * @brief Get maximum tensile stress (positive = tensile)
     */
    double getMaxTensileStress() const { return m_maxTensile; }
    
    /**
     * @brief Get depth where stress changes from compressive to tensile
     */
    double getCrossoverDepth() const { return m_crossoverDepth; }
    
    /**
     * @brief Get affected layer depth (where stress > 10% of max)
     */
    double getAffectedLayerDepth() const { return m_affectedDepth; }
    
    /**
     * @brief Export stress profile to CSV file
     */
    bool exportToCSV(const std::string& filename) const;
    
    /**
     * @brief Get profile summary as string
     */
    std::string getSummary() const;
    
    /**
     * @brief Estimate mechanical contribution to residual stress
     * Based on plastic strain and constraint factor
     */
    static double estimateMechanicalStress(double plasticStrain, double youngModulus,
                                            double constraintFactor = 0.5);
    
    /**
     * @brief Estimate thermal contribution to residual stress
     * Based on temperature change and thermal expansion
     */
    static double estimateThermalStress(double deltaT, double youngModulus,
                                         double thermalExpansion, double constraintFactor = 0.5);

private:
    /**
     * @brief Compute distance from particle to surface plane
     */
    double distanceToSurface(const SPHParticle& p, const Vec3& normal, 
                              const Vec3& surfacePoint) const;
    
    /**
     * @brief Compute von Mises equivalent stress from components
     */
    double computeVonMises(double sxx, double syy, double szz,
                           double sxy, double sxz, double syz) const;
    
    std::vector<StressProfilePoint> m_profile;
    
    // Configuration
    double m_maxDepth = 0.001;      // 1mm default
    int m_depthResolution = 50;     // 50 depth levels
    
    // Results
    double m_surfaceStress = 0;
    double m_maxCompressive = 0;
    double m_maxTensile = 0;
    double m_crossoverDepth = 0;
    double m_affectedDepth = 0;
};

} // namespace edgepredict
