/**
 * @file ResidualStressAnalyzer.cpp
 * @brief Implementation of Residual Stress Analyzer
 */

#include "ResidualStressAnalyzer.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace edgepredict {

ResidualStressAnalyzer::ResidualStressAnalyzer() = default;
ResidualStressAnalyzer::~ResidualStressAnalyzer() = default;

void ResidualStressAnalyzer::analyze(const Config& config) {
    // High-level analysis call from Engine
    // In a full implementation, this might pull data from solvers
    // For now, we emit a status message to confirm the analyzer is 'hooked' correctly
    std::cout << "    [ResidualStress] Surface stress analysis initialized (LOD: " 
              << m_depthResolution << " layers)" << std::endl;
}

void ResidualStressAnalyzer::configure(double maxDepth, int depthResolution) {
    m_maxDepth = maxDepth;
    m_depthResolution = depthResolution;
    
    // Pre-allocate profile
    m_profile.resize(depthResolution);
    double depthStep = m_maxDepth / (depthResolution - 1);
    for (int i = 0; i < depthResolution; ++i) {
        m_profile[i].depth = i * depthStep;
    }
}

void ResidualStressAnalyzer::analyzeFromParticles(const std::vector<SPHParticle>& particles,
                                                   const Vec3& surfaceNormal,
                                                   const Vec3& surfacePoint) {
    if (particles.empty()) return;
    
    // Reset profile
    for (auto& pt : m_profile) {
        pt = StressProfilePoint();
        pt.depth = pt.depth;  // Preserve depth
    }
    
    // Depth step
    double depthStep = m_maxDepth / (m_depthResolution - 1);
    
    // For each depth level, gather particles and compute average stress
    std::vector<int> counts(m_depthResolution, 0);
    std::vector<double> sumAxial(m_depthResolution, 0);
    std::vector<double> sumFeed(m_depthResolution, 0);
    std::vector<double> sumRadial(m_depthResolution, 0);
    std::vector<double> sumVonMises(m_depthResolution, 0);
    std::vector<double> sumTemp(m_depthResolution, 0);
    
    // Define cutting direction (perpendicular to surface normal, in XY plane)
    Vec3 cuttingDir(-surfaceNormal.y, surfaceNormal.x, 0);
    double magCut = std::sqrt(cuttingDir.x*cuttingDir.x + cuttingDir.y*cuttingDir.y);
    if (magCut > 1e-10) {
        cuttingDir.x /= magCut;
        cuttingDir.y /= magCut;
    } else {
        cuttingDir = Vec3(1, 0, 0);
    }
    
    // Feed direction (perpendicular to both)
    Vec3 feedDir(
        surfaceNormal.y * cuttingDir.z - surfaceNormal.z * cuttingDir.y,
        surfaceNormal.z * cuttingDir.x - surfaceNormal.x * cuttingDir.z,
        surfaceNormal.x * cuttingDir.y - surfaceNormal.y * cuttingDir.x
    );
    
    for (const auto& p : particles) {
        // Only analyze workpiece particles (not chips)
        if (p.status != ParticleStatus::ACTIVE) continue;
        
        // Compute distance to surface
        double dist = distanceToSurface(p, surfaceNormal, surfacePoint);
        
        // Only consider particles below surface (positive distance)
        if (dist < 0 || dist > m_maxDepth) continue;
        
        // Find depth bin
        int bin = static_cast<int>(dist / depthStep);
        if (bin >= m_depthResolution) bin = m_depthResolution - 1;
        
        // Compute von Mises stress
        double vonMises = computeVonMises(p.stress_xx, p.stress_yy, p.stress_zz,
                                           p.stress_xy, p.stress_xz, p.stress_yz);
        
        // Project stress to surface coordinates
        // Axial stress (in cutting direction)
        double axial = p.stress_xx * cuttingDir.x * cuttingDir.x +
                       p.stress_yy * cuttingDir.y * cuttingDir.y +
                       2 * p.stress_xy * cuttingDir.x * cuttingDir.y;
        
        // Radial stress (normal to surface)
        double radial = p.stress_xx * surfaceNormal.x * surfaceNormal.x +
                        p.stress_yy * surfaceNormal.y * surfaceNormal.y +
                        p.stress_zz * surfaceNormal.z * surfaceNormal.z;
        
        // Feed stress (in feed direction)
        double feed = p.stress_xx * feedDir.x * feedDir.x +
                      p.stress_yy * feedDir.y * feedDir.y +
                      p.stress_zz * feedDir.z * feedDir.z;
        
        // Accumulate
        counts[bin]++;
        sumAxial[bin] += axial;
        sumFeed[bin] += feed;
        sumRadial[bin] += radial;
        sumVonMises[bin] += vonMises;
        sumTemp[bin] += p.temperature;
    }
    
    // Compute averages
    for (int i = 0; i < m_depthResolution; ++i) {
        m_profile[i].depth = i * depthStep;
        
        if (counts[i] > 0) {
            m_profile[i].stressAxial = sumAxial[i] / counts[i];
            m_profile[i].stressFeed = sumFeed[i] / counts[i];
            m_profile[i].stressRadial = sumRadial[i] / counts[i];
            m_profile[i].stressVonMises = sumVonMises[i] / counts[i];
            m_profile[i].temperature = sumTemp[i] / counts[i];
        }
    }
    
    // Compute summary statistics
    m_surfaceStress = m_profile.empty() ? 0 : m_profile[0].stressAxial;
    m_maxCompressive = 0;
    m_maxTensile = 0;
    m_crossoverDepth = 0;
    bool foundCrossover = false;
    
    for (size_t i = 0; i < m_profile.size(); ++i) {
        double stress = m_profile[i].stressAxial;
        
        if (stress < m_maxCompressive) {
            m_maxCompressive = stress;
        }
        if (stress > m_maxTensile) {
            m_maxTensile = stress;
        }
        
        // Detect crossover (sign change)
        if (!foundCrossover && i > 0) {
            double prevStress = m_profile[i-1].stressAxial;
            if ((prevStress < 0 && stress >= 0) || (prevStress > 0 && stress <= 0)) {
                m_crossoverDepth = m_profile[i].depth;
                foundCrossover = true;
            }
        }
    }
    
    // Affected layer depth (where stress drops below 10% of max)
    double threshold = std::max(std::abs(m_maxCompressive), std::abs(m_maxTensile)) * 0.1;
    m_affectedDepth = m_maxDepth;
    for (size_t i = m_profile.size(); i > 0; --i) {
        if (std::abs(m_profile[i-1].stressAxial) > threshold) {
            m_affectedDepth = m_profile[i-1].depth;
            break;
        }
    }
}

double ResidualStressAnalyzer::distanceToSurface(const SPHParticle& p, 
                                                   const Vec3& normal,
                                                   const Vec3& surfacePoint) const {
    // Distance from point to plane: (P - P0) · n
    double dx = p.x - surfacePoint.x;
    double dy = p.y - surfacePoint.y;
    double dz = p.z - surfacePoint.z;
    
    return -(dx * normal.x + dy * normal.y + dz * normal.z);
}

double ResidualStressAnalyzer::computeVonMises(double sxx, double syy, double szz,
                                                 double sxy, double sxz, double syz) const {
    double s1 = sxx - syy;
    double s2 = syy - szz;
    double s3 = szz - sxx;
    
    return std::sqrt(0.5 * (s1*s1 + s2*s2 + s3*s3 + 
                            6.0 * (sxy*sxy + sxz*sxz + syz*syz)));
}

double ResidualStressAnalyzer::estimateMechanicalStress(double plasticStrain, 
                                                          double youngModulus,
                                                          double constraintFactor) {
    // Residual stress from plastic deformation:
    // σ_residual ≈ -E * ε_plastic * constraint_factor
    // Negative because plastic strain leaves compressive residual stress
    return -youngModulus * plasticStrain * constraintFactor;
}

double ResidualStressAnalyzer::estimateThermalStress(double deltaT, double youngModulus,
                                                       double thermalExpansion,
                                                       double constraintFactor) {
    // Thermal residual stress from non-uniform cooling:
    // σ_thermal = E * α * ΔT * constraint_factor
    // Positive ΔT (cooling from high temp) causes tensile stress near surface
    return youngModulus * thermalExpansion * deltaT * constraintFactor;
}

bool ResidualStressAnalyzer::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ResidualStressAnalyzer] Failed to open " << filename << std::endl;
        return false;
    }
    
    file << "Depth_um,Stress_Axial_MPa,Stress_Feed_MPa,Stress_Radial_MPa,Stress_VonMises_MPa,Temperature_C\n";
    
    for (const auto& pt : m_profile) {
        file << std::fixed << std::setprecision(2)
             << (pt.depth * 1e6) << ","           // Convert to microns
             << (pt.stressAxial / 1e6) << ","     // Convert to MPa
             << (pt.stressFeed / 1e6) << ","
             << (pt.stressRadial / 1e6) << ","
             << (pt.stressVonMises / 1e6) << ","
             << pt.temperature << "\n";
    }
    
    file.close();
    std::cout << "[ResidualStressAnalyzer] Exported profile to " << filename << std::endl;
    return true;
}

std::string ResidualStressAnalyzer::getSummary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "Residual Stress Profile Summary:\n";
    ss << "  Surface stress: " << (m_surfaceStress / 1e6) << " MPa\n";
    ss << "  Max compressive: " << (m_maxCompressive / 1e6) << " MPa\n";
    ss << "  Max tensile: " << (m_maxTensile / 1e6) << " MPa\n";
    ss << "  Crossover depth: " << (m_crossoverDepth * 1e6) << " μm\n";
    ss << "  Affected layer: " << (m_affectedDepth * 1e6) << " μm\n";
    return ss.str();
}

} // namespace edgepredict
