/**
 * @file VTKExporter.cpp
 * @brief Modern VTU export implementation (ascii / binary / compressed)
 *
 * Replaces the legacy VTK ASCII writer with the XML-based .vtu standard.
 * Supports three modes controlled by JSON config "vtu_export_mode":
 *   - "ascii"      : format="ascii" — human-readable, large files
 *   - "binary"     : format="appended" encoding="base64" — fast, medium
 *   - "compressed" : Base64 + zlib — fast, smallest files (~10× smaller)
 *
 * Generates a .pvd manifest for timeline scrubbing in ParaView.
 */

#include "VTKExporter.h"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

// zlib for compressed mode (available in CUDA toolkit / system)
#ifdef _WIN32
#include <zlib.h>
#else
#include <zlib.h>
#endif

namespace edgepredict {

// ─── Base64 encoding table ──────────────────────────────────────────────────
static const char b64chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string VTKExporter::base64Encode(const unsigned char* data, size_t len) {
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = ((unsigned int)data[i]) << 16;
        if (i + 1 < len) n |= ((unsigned int)data[i + 1]) << 8;
        if (i + 2 < len) n |= ((unsigned int)data[i + 2]);
        result.push_back(b64chars[(n >> 18) & 0x3F]);
        result.push_back(b64chars[(n >> 12) & 0x3F]);
        result.push_back((i + 1 < len) ? b64chars[(n >> 6) & 0x3F] : '=');
        result.push_back((i + 2 < len) ? b64chars[n & 0x3F] : '=');
    }
    return result;
}

std::vector<unsigned char> VTKExporter::zlibCompress(const unsigned char* data, size_t len) {
    uLongf compLen = compressBound(static_cast<uLong>(len)) + 1024; // Extra padding to prevent Z_BUF_ERROR
    std::vector<unsigned char> compressed(compLen);
    int ret = compress(compressed.data(), &compLen, data, static_cast<uLong>(len));
    if (ret != Z_OK) {
        std::cerr << "[VTKExporter] zlib compression failed (code " << ret << "), falling back to raw" << std::endl;
        return {};
    }
    compressed.resize(compLen);
    return compressed;
}

// ─── Constructor / Destructor ───────────────────────────────────────────────

VTKExporter::VTKExporter(const std::string& outputDir) 
    : m_outputDir(outputDir) {
    std::filesystem::create_directories(outputDir);
}

VTKExporter::~VTKExporter() {
    // Write PVD manifest on destruction (end of simulation)
    writePVDManifest();
}

void VTKExporter::setExportMode(const std::string& mode) {
    if (mode == "ascii" || mode == "binary" || mode == "compressed") {
        m_exportMode = mode;
    } else {
        std::cerr << "[VTKExporter] Unknown export mode '" << mode
                  << "', using 'compressed'" << std::endl;
        m_exportMode = "compressed";
    }
    std::cout << "[VTKExporter] Export mode: " << m_exportMode << std::endl;
}

// ─── DataArray writer (templated for float/double/int) ──────────────────────
template<typename T>
void VTKExporter::writeDataArray(std::ofstream& file, const std::string& name,
                                  int numComponents, const std::vector<T>& data) const {
    // Determine VTK type string
    std::string typeStr;
    if constexpr (std::is_same_v<T, double>) typeStr = "Float64";
    else if constexpr (std::is_same_v<T, float>) typeStr = "Float32";
    else if constexpr (std::is_same_v<T, int>) typeStr = "Int32";
    else if constexpr (std::is_same_v<T, unsigned char>) typeStr = "UInt8";
    else typeStr = "Float64";

    if (m_exportMode == "ascii") {
        file << "          <DataArray type=\"" << typeStr << "\" Name=\"" << name
             << "\" NumberOfComponents=\"" << numComponents << "\" format=\"ascii\">\n";
        file << "            ";
        for (size_t i = 0; i < data.size(); ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                file << std::scientific << std::setprecision(6) << data[i];
            } else {
                file << data[i];
            }
            if (i + 1 < data.size()) file << " ";
            if ((i + 1) % 12 == 0 && i + 1 < data.size()) file << "\n            ";
        }
        file << "\n          </DataArray>\n";
    } else {
        // Binary or compressed: encode data as Base64
        const unsigned char* rawPtr = reinterpret_cast<const unsigned char*>(data.data());
        size_t rawLen = data.size() * sizeof(T);

        bool isCompressed = false;
        std::string encoded;

        if (m_exportMode == "compressed") {
            auto compressed = zlibCompress(rawPtr, rawLen);
            if (!compressed.empty()) {
                isCompressed = true;
                // VTK compressed block layout (vtkZLibDataCompressor, inline
                // "binary" format): a 4 x uint32 header followed by the zlib
                // stream.  header[] = [numBlocks, blockSize, lastBlockSize,
                // compressedSize] where compressedSize is the zlib byte length.
                uint32_t header[4] = {1, (uint32_t)rawLen, (uint32_t)rawLen,
                                      (uint32_t)compressed.size()};
                std::vector<unsigned char> hdrVec(
                    reinterpret_cast<unsigned char*>(header),
                    reinterpret_cast<unsigned char*>(header) + 16);
                // CRITICAL: VTK's reader decodes the header and the payload as
                // TWO SEPARATE base64 blocks concatenated back-to-back (e.g.
                // "AQAA...AAAA==eF5j..."). Encoding them as a single contiguous
                // base64 stream makes vtkXMLDataReader fail ("data array ...
                // may be too short") and yields no visible geometry.
                encoded = base64Encode(hdrVec.data(), hdrVec.size())
                        + base64Encode(compressed.data(), compressed.size());
            } else {
                // Fallback to raw binary
                uint32_t sizeHeader = static_cast<uint32_t>(rawLen);
                std::vector<unsigned char> hdr(
                    reinterpret_cast<unsigned char*>(&sizeHeader),
                    reinterpret_cast<unsigned char*>(&sizeHeader) + 4);
                encoded = base64Encode(hdr.data(), hdr.size())
                        + base64Encode(rawPtr, rawLen);
            }
        } else {
            // Raw binary: 4-byte size header followed by raw bytes, each
            // base64-encoded separately and concatenated (VTK-compatible).
            uint32_t sizeHeader = static_cast<uint32_t>(rawLen);
            std::vector<unsigned char> hdr(
                reinterpret_cast<unsigned char*>(&sizeHeader),
                reinterpret_cast<unsigned char*>(&sizeHeader) + 4);
            encoded = base64Encode(hdr.data(), hdr.size())
                    + base64Encode(rawPtr, rawLen);
        }

        std::string compStr = isCompressed
            ? " compressor=\"vtkZLibDataCompressor\"" : "";
        file << "          <DataArray type=\"" << typeStr << "\" Name=\"" << name
             << "\" NumberOfComponents=\"" << numComponents
             << "\" format=\"binary\"" << compStr << ">\n";
        file << "            " << encoded << "\n";
        file << "          </DataArray>\n";
    }
}

// ─── IExporter interface ────────────────────────────────────────────────────

void VTKExporter::exportStep(int step, double time, const Mesh& mesh) {
    std::ostringstream filename;
    filename << m_outputDir << "/mesh_" << std::setfill('0') << std::setw(6) << step << ".vtu";
    
    std::vector<double> stress, temperature, wear;
    stress.reserve(mesh.nodes.size());
    temperature.reserve(mesh.nodes.size());
    wear.reserve(mesh.nodes.size());
    
    for (const auto& node : mesh.nodes) {
        stress.push_back(node.stress);
        temperature.push_back(node.temperature);
        wear.push_back(node.accumulatedWear);
    }
    
    writeVTUMesh(filename.str(), mesh, stress, temperature, wear, mesh.materialStatus);

    // Record for PVD manifest
    std::string relName = "mesh_" + std::to_string(step) + ".vtu";
    std::ostringstream relSS;
    relSS << "mesh_" << std::setfill('0') << std::setw(6) << step << ".vtu";
    m_pvdEntries.push_back({time, relSS.str()});
    m_stepCounter++;
}

void VTKExporter::exportFinal(const Config& config, const Mesh& mesh) {
    (void)config;
    std::string filename = m_outputDir + "/tool_final.vtu";
    
    std::vector<double> stress, temperature, wear;
    stress.reserve(mesh.nodes.size());
    temperature.reserve(mesh.nodes.size());
    wear.reserve(mesh.nodes.size());
    
    for (const auto& node : mesh.nodes) {
        stress.push_back(node.stress);
        temperature.push_back(node.temperature);
        wear.push_back(node.accumulatedWear);
    }
    
    writeVTUMesh(filename, mesh, stress, temperature, wear, {});
    std::cout << "[VTKExporter] Exported final mesh to: " << filename << std::endl;

    // Write PVD manifest
    writePVDManifest();
}

void VTKExporter::exportReconstructedWorkpiece(const Mesh& mesh) {
    exportReconstructedWorkpiece(-1, 0.0, mesh);
}

void VTKExporter::exportReconstructedWorkpiece(int step, double time, const Mesh& mesh) {
    (void)time;
    std::string filename;
    if (step >= 0) {
        std::ostringstream ss;
        ss << m_outputDir << "/workpiece_" << std::setfill('0') << std::setw(6) << step << ".vtu";
        filename = ss.str();
    } else {
        filename = m_outputDir + "/workpiece_final.vtu";
    }
    
    std::vector<double> stress, temperature, wear;
    stress.reserve(mesh.nodes.size());
    temperature.reserve(mesh.nodes.size());
    wear.reserve(mesh.nodes.size());
    
    for (const auto& node : mesh.nodes) {
        stress.push_back(node.stress);
        temperature.push_back(node.temperature);
        wear.push_back(node.accumulatedWear);
    }
    
    writeVTUMesh(filename, mesh, stress, temperature, wear, mesh.materialStatus);
    std::cout << "[VTKExporter] Exported reconstructed workpiece mesh to: " << filename << std::endl;
}

void VTKExporter::exportParticles(int step, double time,
                                   const std::vector<MPMParticle>& particles) {
    std::ostringstream filename;
    filename << m_outputDir << "/particles_" << std::setfill('0') << std::setw(6) << step << ".vtu";
    writeVTUParticles(filename.str(), particles);

    std::ostringstream relSS;
    relSS << "particles_" << std::setfill('0') << std::setw(6) << step << ".vtu";
    m_pvdParticleEntries.push_back({time, relSS.str()});
}

void VTKExporter::exportMetrics(int step, double time, double maxStress, double maxTemp) {
    (void)step; (void)time; (void)maxStress; (void)maxTemp;
}

// Legacy interface delegates to VTU writer
void VTKExporter::exportMesh(const std::string& filename, const Mesh& mesh,
                              const std::vector<double>& stress,
                              const std::vector<double>& temperature,
                              const std::vector<double>& wear,
                              const std::vector<int>& materialStatus) {
    // Remap .vtk extension to .vtu
    std::string vtuFile = filename;
    auto pos = vtuFile.rfind(".vtk");
    if (pos != std::string::npos) {
        vtuFile.replace(pos, 4, ".vtu");
    }
    writeVTUMesh(vtuFile, mesh, stress, temperature, wear, materialStatus);
}

void VTKExporter::exportParticles(const std::string& filename,
                                   const std::vector<MPMParticle>& particles) {
    std::string vtuFile = filename;
    auto pos = vtuFile.rfind(".vtk");
    if (pos != std::string::npos) {
        vtuFile.replace(pos, 4, ".vtu");
    }
    writeVTUParticles(vtuFile, particles);
}

// ─── VTU Mesh Writer ────────────────────────────────────────────────────────

void VTKExporter::writeVTUMesh(const std::string& filename, const Mesh& mesh,
                                const std::vector<double>& stress,
                                const std::vector<double>& temperature,
                                const std::vector<double>& wear,
                                const std::vector<int>& materialStatus) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[VTKExporter] Failed to open: " << filename << std::endl;
        return;
    }

    int numPoints = static_cast<int>(mesh.nodes.size());
    int numCells = 0;
    int cellType = 10; // VTK_TETRA
    int nodesPerCell = 4;

    if (!mesh.triangles.empty()) {
        numCells = static_cast<int>(mesh.triangles.size());
        cellType = 5;  // VTK_TRIANGLE
        nodesPerCell = 3;
    } else if (!mesh.elements.empty()) {
        numCells = static_cast<int>(mesh.elements.size());
        cellType = 10; // VTK_TETRA
        nodesPerCell = 4;
    }

    // XML VTU header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\"";
    if (m_exportMode == "compressed") {
        file << " compressor=\"vtkZLibDataCompressor\"";
    }
    file << ">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << numPoints
         << "\" NumberOfCells=\"" << numCells << "\">\n";

    // ── Points ──────────────────────────────────────────────────────────
    file << "      <Points>\n";
    {
        std::vector<double> coords;
        coords.reserve(numPoints * 3);
        for (const auto& node : mesh.nodes) {
            coords.push_back(node.position.x);
            coords.push_back(node.position.y);
            coords.push_back(node.position.z);
        }
        writeDataArray(file, "Points", 3, coords);
    }
    file << "      </Points>\n";

    // ── Cells ───────────────────────────────────────────────────────────
    file << "      <Cells>\n";
    if (numCells > 0) {
        // Connectivity
        std::vector<int> connectivity;
        connectivity.reserve(numCells * nodesPerCell);
        if (!mesh.triangles.empty()) {
            for (const auto& tri : mesh.triangles) {
                connectivity.push_back(tri.indices[0]);
                connectivity.push_back(tri.indices[1]);
                connectivity.push_back(tri.indices[2]);
            }
        } else {
            for (const auto& el : mesh.elements) {
                connectivity.push_back(el.nodeIndices[0]);
                connectivity.push_back(el.nodeIndices[1]);
                connectivity.push_back(el.nodeIndices[2]);
                connectivity.push_back(el.nodeIndices[3]);
            }
        }
        writeDataArray(file, "connectivity", 1, connectivity);

        // Offsets
        std::vector<int> offsets;
        offsets.reserve(numCells);
        for (int i = 1; i <= numCells; ++i) {
            offsets.push_back(i * nodesPerCell);
        }
        writeDataArray(file, "offsets", 1, offsets);

        // Types
        std::vector<unsigned char> types(numCells, static_cast<unsigned char>(cellType));
        writeDataArray(file, "types", 1, types);
    }
    file << "      </Cells>\n";

    // ── Point Data ──────────────────────────────────────────────────────
    file << "      <PointData>\n";

    // Von Mises Stress (MPa)
    if (stress.size() == (size_t)numPoints) {
        std::vector<double> stressMPa;
        stressMPa.reserve(numPoints);
        for (double s : stress) stressMPa.push_back(s / 1e6);
        writeDataArray(file, "VonMisesStress_MPa", 1, stressMPa);
    }

    // Temperature (°C)
    if (temperature.size() == (size_t)numPoints) {
        writeDataArray(file, "Temperature_C", 1, temperature);
    }

    // Wear (mm)
    if (wear.size() == (size_t)numPoints) {
        std::vector<double> wearMm;
        wearMm.reserve(numPoints);
        for (double w : wear) wearMm.push_back(w * 1000.0);
        writeDataArray(file, "Wear_mm", 1, wearMm);
    }

    // Tool Region ID
    {
        std::vector<int> regionId;
        regionId.reserve(numPoints);
        for (const auto& node : mesh.nodes) {
            regionId.push_back(static_cast<int>(node.toolRegion));
        }
        writeDataArray(file, "ToolRegionId", 1, regionId);
    }

    // Chipping Risk
    {
        std::vector<double> chipping;
        chipping.reserve(numPoints);
        for (const auto& node : mesh.nodes) {
            chipping.push_back(node.chippingRisk);
        }
        writeDataArray(file, "ChippingRisk", 1, chipping);
    }

    // Coating Remaining (%)
    {
        std::vector<double> coating;
        coating.reserve(numPoints);
        for (const auto& node : mesh.nodes) {
            coating.push_back(node.coatingRemaining * 100.0);
        }
        writeDataArray(file, "CoatingRemaining_percent", 1, coating);
    }

    // Flank Wear (μm)
    {
        std::vector<double> flank;
        flank.reserve(numPoints);
        for (const auto& node : mesh.nodes) {
            flank.push_back(node.flankWear * 1e6);
        }
        writeDataArray(file, "FlankWear_um", 1, flank);
    }

    // Crater Wear (μm)
    {
        std::vector<double> crater;
        crater.reserve(numPoints);
        for (const auto& node : mesh.nodes) {
            crater.push_back(node.craterWear * 1e6);
        }
        writeDataArray(file, "CraterWear_um", 1, crater);
    }

    // Material Status
    if (materialStatus.size() == (size_t)numPoints) {
        writeDataArray(file, "MaterialStatus", 1, materialStatus);
    }

    // Displacement magnitude (mm)
    {
        std::vector<double> dispMag;
        dispMag.reserve(numPoints);
        for (const auto& node : mesh.nodes) {
            Vec3 disp = node.position - node.originalPosition;
            dispMag.push_back(disp.length() * 1000.0);
        }
        writeDataArray(file, "Displacement_mm", 1, dispMag);
    }

    // Displacement vector
    {
        std::vector<double> dispVec;
        dispVec.reserve(numPoints * 3);
        for (const auto& node : mesh.nodes) {
            Vec3 disp = node.position - node.originalPosition;
            dispVec.push_back(disp.x);
            dispVec.push_back(disp.y);
            dispVec.push_back(disp.z);
        }
        writeDataArray(file, "Displacement", 3, dispVec);
    }

    file << "      </PointData>\n";

    // Close tags
    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";
    file.close();
}

// ─── VTU Particle Writer ────────────────────────────────────────────────────

void VTKExporter::writeVTUParticles(const std::string& filename,
                                     const std::vector<MPMParticle>& particles) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[VTKExporter] Failed to open: " << filename << std::endl;
        return;
    }

    // Count active particles
    int numActive = 0;
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) numActive++;
    }

    // XML VTU header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\"";
    if (m_exportMode == "compressed") {
        file << " compressor=\"vtkZLibDataCompressor\"";
    }
    file << ">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << numActive
         << "\" NumberOfCells=\"" << numActive << "\">\n";

    // Points
    file << "      <Points>\n";
    {
        std::vector<double> coords;
        coords.reserve(numActive * 3);
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) {
                coords.push_back(p.x);
                coords.push_back(p.y);
                coords.push_back(p.z);
            }
        }
        writeDataArray(file, "Points", 3, coords);
    }
    file << "      </Points>\n";

    // Cells (vertices — one cell per point)
    file << "      <Cells>\n";
    {
        std::vector<int> connectivity(numActive);
        std::vector<int> offsets(numActive);
        std::vector<unsigned char> types(numActive, 1); // VTK_VERTEX
        for (int i = 0; i < numActive; ++i) {
            connectivity[i] = i;
            offsets[i] = i + 1;
        }
        writeDataArray(file, "connectivity", 1, connectivity);
        writeDataArray(file, "offsets", 1, offsets);
        writeDataArray(file, "types", 1, types);
    }
    file << "      </Cells>\n";

    // Point Data
    file << "      <PointData>\n";

    // Temperature
    {
        std::vector<double> temp;
        temp.reserve(numActive);
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) temp.push_back(p.temperature);
        }
        writeDataArray(file, "Temperature_C", 1, temp);
    }

    // Pressure (MPa)
    {
        std::vector<double> pressure;
        pressure.reserve(numActive);
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) pressure.push_back(p.pressure / 1e6);
        }
        writeDataArray(file, "Pressure_MPa", 1, pressure);
    }

    // Velocity magnitude
    {
        std::vector<double> vmag;
        vmag.reserve(numActive);
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) {
                vmag.push_back(std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz));
            }
        }
        writeDataArray(file, "Velocity_m_s", 1, vmag);
    }

    // Status
    {
        std::vector<int> status;
        status.reserve(numActive);
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) {
                status.push_back(static_cast<int>(p.status));
            }
        }
        writeDataArray(file, "ParticleStatus", 1, status);
    }

    // Velocity vector
    {
        std::vector<double> vel;
        vel.reserve(numActive * 3);
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) {
                vel.push_back(p.vx);
                vel.push_back(p.vy);
                vel.push_back(p.vz);
            }
        }
        writeDataArray(file, "Velocity", 3, vel);
    }

    file << "      </PointData>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";
    file.close();
}

// ─── PVD Manifest Writer ────────────────────────────────────────────────────

void VTKExporter::writePVDManifest() const {
    // Mesh timeline
    if (!m_pvdEntries.empty()) {
        std::string pvdPath = m_outputDir + "/simulation_mesh.pvd";
        std::ofstream pvd(pvdPath);
        if (pvd.is_open()) {
            pvd << "<?xml version=\"1.0\"?>\n";
            pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            pvd << "  <Collection>\n";
            for (const auto& [time, fname] : m_pvdEntries) {
                pvd << "    <DataSet timestep=\"" << std::scientific << std::setprecision(8) << time
                    << "\" file=\"" << fname << "\"/>\n";
            }
            pvd << "  </Collection>\n";
            pvd << "</VTKFile>\n";
            pvd.close();
            std::cout << "[VTKExporter] Wrote PVD manifest: " << pvdPath
                      << " (" << m_pvdEntries.size() << " timesteps)" << std::endl;
        }
    }

    // Particle timeline
    if (!m_pvdParticleEntries.empty()) {
        std::string pvdPath = m_outputDir + "/simulation_particles.pvd";
        std::ofstream pvd(pvdPath);
        if (pvd.is_open()) {
            pvd << "<?xml version=\"1.0\"?>\n";
            pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            pvd << "  <Collection>\n";
            for (const auto& [time, fname] : m_pvdParticleEntries) {
                pvd << "    <DataSet timestep=\"" << std::scientific << std::setprecision(8) << time
                    << "\" file=\"" << fname << "\"/>\n";
            }
            pvd << "  </Collection>\n";
            pvd << "</VTKFile>\n";
            pvd.close();
            std::cout << "[VTKExporter] Wrote particle PVD manifest: " << pvdPath
                      << " (" << m_pvdParticleEntries.size() << " timesteps)" << std::endl;
        }
    }
}

} // namespace edgepredict
