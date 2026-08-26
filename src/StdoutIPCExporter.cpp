#include "StdoutIPCExporter.h"
#include "Base64.h"
#include "json.hpp"
#include <iostream>
#include <algorithm>
#include <unordered_map>

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

void StdoutIPCExporter::exportDetailedMetrics(int step, double time,
                                              double workpieceStress, double workpieceTemp,
                                              double toolStress, double toolTemp,
                                              double coolantTemp) {
    json j;
    j["type"] = "metrics";
    j["step"] = step;
    j["time"] = time;
    j["workpiece_stress_mpa"] = workpieceStress / 1e6;
    j["workpiece_temp_c"] = workpieceTemp;
    j["tool_stress_mpa"] = toolStress / 1e6;
    j["tool_temp_c"] = toolTemp;
    j["coolant_temp_c"] = coolantTemp;
    j["max_stress_mpa"] = std::max(workpieceStress, toolStress) / 1e6;
    j["max_temp_c"] = std::max(std::max(workpieceTemp, toolTemp), coolantTemp);

    std::fputs(j.dump().c_str(), stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);
}

void StdoutIPCExporter::exportParticles(int step, double time, const std::vector<MPMParticle>& particles) {
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

void StdoutIPCExporter::exportReconstructedWorkpiece(int step, double time, const Mesh& mesh) {
    (void)time;
    if (mesh.nodes.empty() || mesh.triangles.empty()) return;

    // Fast vertex-clustering decimation: keep at most 5000 triangles
    size_t maxVerts = 2500;
    size_t maxTris = 5000;
    size_t vertexStride = 1;
    size_t triStride = 1;
    if (mesh.nodes.size() > maxVerts) vertexStride = mesh.nodes.size() / maxVerts + 1;
    if (mesh.triangles.size() > maxTris) triStride = mesh.triangles.size() / maxTris + 1;

    std::vector<float> pos;
    std::vector<int> triIndices;
    std::vector<float> vertTemps;
    std::vector<int> vertStatus;

    // Build remap from original vertex index to decimated index
    std::unordered_map<int, int> remap;
    for (size_t vi = 0; vi < mesh.nodes.size(); vi += vertexStride) {
        int newId = static_cast<int>(pos.size() / 3);
        remap[static_cast<int>(vi)] = newId;
        pos.push_back(static_cast<float>(mesh.nodes[vi].position.x));
        pos.push_back(static_cast<float>(mesh.nodes[vi].position.y));
        pos.push_back(static_cast<float>(mesh.nodes[vi].position.z));
        vertTemps.push_back(static_cast<float>(mesh.nodes[vi].temperature));
        int ms = (!mesh.materialStatus.empty() && vi < mesh.materialStatus.size())
                 ? mesh.materialStatus[vi] : 0;
        vertStatus.push_back(ms);
    }

    for (size_t ti = 0; ti < mesh.triangles.size(); ti += triStride) {
        const auto& tri = mesh.triangles[ti];
        auto it0 = remap.find(tri.indices[0]);
        auto it1 = remap.find(tri.indices[1]);
        auto it2 = remap.find(tri.indices[2]);
        if (it0 != remap.end() && it1 != remap.end() && it2 != remap.end()) {
            triIndices.push_back(it0->second);
            triIndices.push_back(it1->second);
            triIndices.push_back(it2->second);
        }
    }

    int numVerts = static_cast<int>(pos.size() / 3);
    int numTris = static_cast<int>(triIndices.size() / 3);
    if (numVerts == 0 || numTris == 0) return;

    std::string b64Pos = edgepredict::utils::base64_encode(
        reinterpret_cast<const unsigned char*>(pos.data()),
        pos.size() * sizeof(float));
    std::string b64Tris = edgepredict::utils::base64_encode(
        reinterpret_cast<const unsigned char*>(triIndices.data()),
        triIndices.size() * sizeof(int));
    std::string b64Temps = edgepredict::utils::base64_encode(
        reinterpret_cast<const unsigned char*>(vertTemps.data()),
        vertTemps.size() * sizeof(float));
    std::string b64Status = edgepredict::utils::base64_encode(
        reinterpret_cast<const unsigned char*>(vertStatus.data()),
        vertStatus.size() * sizeof(int));

    json j;
    j["type"] = "reconstructed_mesh";
    j["step"] = step;
    j["num_vertices"] = numVerts;
    j["num_triangles"] = numTris;
    j["positions_b64"] = b64Pos;
    j["triangles_b64"] = b64Tris;
    j["temperatures_b64"] = b64Temps;
    j["material_status_b64"] = b64Status;

    std::fputs(j.dump().c_str(), stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);
}

} // namespace edgepredict
