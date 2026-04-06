#pragma once
/**
 * @file HDF5Exporter.h
 * @brief HDF5 file export for time-series simulation data
 * 
 * Stores simulation data in efficient binary format for:
 * - Post-processing in Python/MATLAB
 * - Large dataset handling
 * - Time-series analysis
 */

#include "Types.h"
#include "SimulationEngine.h"
#include "SPHSolver.cuh"
#include <string>
#include <vector>

#ifdef HAS_HDF5
#include <H5Cpp.h>
#endif

namespace edgepredict {

/**
 * @brief HDF5 dataset structure
 */
struct HDF5SimulationData {
    std::vector<double> particleX, particleY, particleZ;
    std::vector<double> particleVx, particleVy, particleVz;
    std::vector<double> particleTemp, particlePressure;
    std::vector<int> particleStatus;
    
    std::vector<double> nodeX, nodeY, nodeZ;
    std::vector<double> nodeStress, nodeTemp, nodeWear;
    
    double time;
    int step;
    double maxStress, maxTemp;
};

/**
 * @brief Exports simulation data to HDF5 format
 */
class HDF5Exporter : public IExporter {
public:
    HDF5Exporter(const std::string& outputDir);
    ~HDF5Exporter() override;
    
    // IExporter interface
    std::string getName() const override { return "HDF5Exporter"; }
    void exportStep(int step, double time, const Mesh& mesh) override;
    void exportFinal(const Config& config, const Mesh& mesh) override;
    
    // HDF5-specific methods
    
    /**
     * @brief Export particle data
     */
    void exportParticles(int step, double time, const std::vector<SPHParticle>& particles);
    
    /**
     * @brief Export FEM node data
     */
    void exportNodes(int step, double time, const std::vector<FEMNode>& nodes);
    
    /**
     * @brief Export time-series scalar
     */
    void appendTimeSeries(const std::string& name, double time, double value);
    
    /**
     * @brief Create a new file for this simulation
     */
    void createFile(const std::string& filename);
    
    /**
     * @brief Close current file
     */
    void closeFile();
    
    /**
     * @brief Set compression level (0-9)
     */
    void setCompressionLevel(int level) { m_compressionLevel = level; }

private:
    std::string m_outputDir;
    std::string m_currentFile;
    int m_compressionLevel = 4;
    
#ifdef HAS_HDF5
    std::unique_ptr<H5::H5File> m_file;
    void writeDataset(const std::string& name, const std::vector<double>& data, int step);
    void writeDataset(const std::string& name, const std::vector<int>& data, int step);
#endif
    
    // Fallback: simple binary format when HDF5 not available
    void writeBinaryFallback(int step, double time, const Mesh& mesh);
};

} // namespace edgepredict
