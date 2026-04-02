#pragma once
/**
 * @file VTKExporter.h
 * @brief VTK format exporter for visualization
 */

#include "Types.h"
#include "SimulationEngine.h"
#include "SPHSolver.cuh"
#include <string>
#include <vector>

namespace edgepredict {

/**
 * @brief Exports simulation data to VTK format for visualization
 */
class VTKExporter : public IExporter {
public:
    VTKExporter(const std::string& outputDir);
    ~VTKExporter() override = default;
    
    // IExporter interface
    std::string getName() const override { return "VTKExporter"; }
    void exportStep(int step, double time, const Mesh& mesh) override;
    void exportFinal(const Config& config, const Mesh& mesh) override;
    void exportParticles(int step, double time, const std::vector<SPHParticle>& particles) override;
    void exportMetrics(int step, double time, double maxStress, double maxTemp) override;
    
    // Additional export methods
    
    /**
     * @brief Export mesh with scalar fields
     */
    void exportMesh(const std::string& filename, const Mesh& mesh,
                    const std::vector<double>& stress = {},
                    const std::vector<double>& temperature = {},
                    const std::vector<double>& wear = {});
    
    /**
     * @brief Export SPH particles as point cloud
     */
    void exportParticles(const std::string& filename,
                         const std::vector<SPHParticle>& particles);
    
    /**
     * @brief Set output directory
     */
    void setOutputDirectory(const std::string& dir) { m_outputDir = dir; }

private:
    std::string m_outputDir;
    int m_stepCounter = 0;
};

} // namespace edgepredict
