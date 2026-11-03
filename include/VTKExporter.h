#pragma once
/**
 * @file VTKExporter.h
 * @brief Modern VTU format exporter for visualization
 *
 * Supports three export modes:
 *   - "ascii"      : Human-readable XML VTU (slow, large, debuggable)
 *   - "binary"     : Base64-encoded raw binary VTU (fast, medium size)
 *   - "compressed" : zlib-compressed Base64 VTU (fast, smallest files)
 *
 * Generates a .pvd manifest for timeline scrubbing in ParaView.
 */

#include "Types.h"
#include "SimulationEngine.h"
#include "MPMSolver.cuh"
#include <string>
#include <vector>
#include <utility>

namespace edgepredict {

/**
 * @brief Exports simulation data to VTU format for visualization
 */
class VTKExporter : public IExporter {
public:
    VTKExporter(const std::string& outputDir);
    ~VTKExporter() override;
    
    // IExporter interface
    std::string getName() const override { return "VTKExporter"; }
    void exportStep(int step, double time, const Mesh& mesh) override;
    void exportFinal(const Config& config, const Mesh& mesh) override;
    void exportParticles(int step, double time, const std::vector<MPMParticle>& particles) override;
    void exportMetrics(int step, double time, double maxStress, double maxTemp) override;
    
    /**
     * @brief Export the reconstructed solid workpiece as a VTU mesh
     */
    void exportReconstructedWorkpiece(const Mesh& mesh);
    void exportReconstructedWorkpiece(int step, double time, const Mesh& mesh) override;
    
    /**
     * @brief Export mesh with scalar fields
     */
    void exportMesh(const std::string& filename, const Mesh& mesh,
                    const std::vector<double>& stress = {},
                    const std::vector<double>& temperature = {},
                    const std::vector<double>& wear = {},
                    const std::vector<int>& materialStatus = {});
    
    /**
     * @brief Export SPH particles as point cloud
     */
    void exportParticles(const std::string& filename,
                         const std::vector<MPMParticle>& particles);
    
    /**
     * @brief Set output directory
     */
    void setOutputDirectory(const std::string& dir) { m_outputDir = dir; }

    /**
     * @brief Set export mode: "ascii", "binary", or "compressed"
     */
    void setExportMode(const std::string& mode);

    /**
     * @brief Write the .pvd manifest file linking all exported timesteps
     */
    void writePVDManifest() const;

private:
    std::string m_outputDir;
    int m_stepCounter = 0;
    std::string m_exportMode = "compressed";  // "ascii" | "binary" | "compressed"

    // PVD manifest entries: (timestep, filename_relative)
    std::vector<std::pair<double, std::string>> m_pvdEntries;
    std::vector<std::pair<double, std::string>> m_pvdParticleEntries;

    // VTU writing helpers
    void writeVTUMesh(const std::string& filename, const Mesh& mesh,
                      const std::vector<double>& stress,
                      const std::vector<double>& temperature,
                      const std::vector<double>& wear,
                      const std::vector<int>& materialStatus);
    void writeVTUParticles(const std::string& filename,
                           const std::vector<MPMParticle>& particles);

    // Encoding helpers
    static std::string base64Encode(const unsigned char* data, size_t len);
    static std::vector<unsigned char> zlibCompress(const unsigned char* data, size_t len);

    // Write a DataArray tag with data in the configured mode
    template<typename T>
    void writeDataArray(std::ofstream& file, const std::string& name,
                        int numComponents, const std::vector<T>& data) const;
};

} // namespace edgepredict
