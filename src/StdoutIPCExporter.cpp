#include "StdoutIPCExporter.h"
#include "Base64.h"
#include "json.hpp"
#include <iostream>

using json = nlohmann::json;

namespace edgepredict {

void StdoutIPCExporter::exportStep(int step, double time, const Mesh& mesh) {
    (void)time;
    // Downstream consumers usually prioritize particles over mesh updates for live rendering
    // But we could dump the tool mesh deformation here if requested.
    // For now, it's largely static so we won't bloat the IPC pipe.
}

void StdoutIPCExporter::exportFinal(const Config& config, const Mesh& mesh) {
    (void)config;
    (void)mesh;
    
    json j;
    j["type"] = "final_mesh_ready";
    j["message"] = "Simulation complete - final mesh safely exported to VTK/HDF5 storage.";
    std::fputs(j.dump().c_str(), stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);
}

void StdoutIPCExporter::exportMetrics(int step, double time, double maxStress, double maxTemp) {
    json j;
    j["type"] = "metrics";
    j["step"] = step;
    j["time"] = time;
    j["max_stress_mpa"] = maxStress / 1e6;
    j["max_temp_c"] = maxTemp;
    
    std::fputs(j.dump().c_str(), stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);
}

void StdoutIPCExporter::exportParticles(int step, double time, const std::vector<SPHParticle>& particles) {
    // Count active
    int activeCount = 0;
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            activeCount++;
        }
    }
    
    if (activeCount == 0) return;
    
    std::vector<float> positions;
    std::vector<float> temps;
    positions.reserve(activeCount * 3);
    temps.reserve(activeCount);
    
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            positions.push_back(static_cast<float>(p.x));
            positions.push_back(static_cast<float>(p.y));
            positions.push_back(static_cast<float>(p.z));
            temps.push_back(static_cast<float>(p.temperature));
        }
    }
    
    std::string b64_pos = edgepredict::utils::base64_encode(
        reinterpret_cast<const unsigned char*>(positions.data()), 
        positions.size() * sizeof(float)
    );
    
    std::string b64_temps = edgepredict::utils::base64_encode(
        reinterpret_cast<const unsigned char*>(temps.data()), 
        temps.size() * sizeof(float)
    );
    
    json j;
    j["type"] = "particles";
    j["step"] = step;
    j["count"] = activeCount;
    j["positions_b64"] = b64_pos;
    j["temps_b64"] = b64_temps;
    
    std::fputs(j.dump().c_str(), stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);
}

} // namespace edgepredict
