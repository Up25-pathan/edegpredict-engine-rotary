#pragma once

#include "Types.h"
#include "SimulationEngine.h"
#include "SPHSolver.cuh"
#include <string>
#include <vector>

namespace edgepredict {

/**
 * @brief Exporter that writes simulation data strictly to standard output 
 * as Newline-Delimited JSON (NDJSON) for native Tauri/Node.js pipeline integration.
 */
class StdoutIPCExporter : public IExporter {
public:
    StdoutIPCExporter() = default;
    ~StdoutIPCExporter() override = default;
    
    // IExporter base interface
    std::string getName() const override { return "StdoutIPCExporter"; }
    void exportStep(int step, double time, const Mesh& mesh) override;
    void exportFinal(const Config& config, const Mesh& mesh) override;
    
    // Extended IPC hooks
    void exportParticles(int step, double time, const std::vector<SPHParticle>& particles) override;
    void exportMetrics(int step, double time, double maxStress, double maxTemp) override;
};

} // namespace edgepredict
