/**
 * @file VTKExporter.cpp
 * @brief VTK file format export implementation
 */

#include "VTKExporter.h"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace edgepredict {

VTKExporter::VTKExporter(const std::string& outputDir) 
    : m_outputDir(outputDir) {
    std::filesystem::create_directories(outputDir);
}

void VTKExporter::exportStep(int step, double time, const Mesh& mesh) {
    (void)time;
    std::ostringstream filename;
    filename << m_outputDir << "/mesh_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    
    // Extract stress and temperature from mesh nodes
    std::vector<double> stress, temperature, wear;
    stress.reserve(mesh.nodes.size());
    temperature.reserve(mesh.nodes.size());
    wear.reserve(mesh.nodes.size());
    
    for (const auto& node : mesh.nodes) {
        stress.push_back(node.stress);
        temperature.push_back(node.temperature);
        wear.push_back(node.accumulatedWear);
    }
    
    exportMesh(filename.str(), mesh, stress, temperature, wear);
    m_stepCounter++;
}

void VTKExporter::exportFinal(const Config& config, const Mesh& mesh) {
    (void)config;
    std::string filename = m_outputDir + "/tool_final.vtk";
    
    std::vector<double> stress, temperature, wear;
    stress.reserve(mesh.nodes.size());
    temperature.reserve(mesh.nodes.size());
    wear.reserve(mesh.nodes.size());
    
    for (const auto& node : mesh.nodes) {
        stress.push_back(node.stress);
        temperature.push_back(node.temperature);
        wear.push_back(node.accumulatedWear);
    }
    
    exportMesh(filename, mesh, stress, temperature, wear);
    std::cout << "[VTKExporter] Exported final mesh to: " << filename << std::endl;
}

void VTKExporter::exportMesh(const std::string& filename, const Mesh& mesh,
                              const std::vector<double>& stress,
                              const std::vector<double>& temperature,
                              const std::vector<double>& wear) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[VTKExporter] Failed to open: " << filename << std::endl;
        return;
    }
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "EdgePredict Tool Mesh\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    
    // Points
    file << "POINTS " << mesh.nodes.size() << " double\n";
    for (const auto& node : mesh.nodes) {
        file << std::scientific << std::setprecision(8)
             << node.position.x << " " 
             << node.position.y << " " 
             << node.position.z << "\n";
    }
    
    // Cells (triangles)
    if (!mesh.triangles.empty()) {
        int numCells = static_cast<int>(mesh.triangles.size());
        file << "\nCELLS " << numCells << " " << (numCells * 4) << "\n";
        for (const auto& tri : mesh.triangles) {
            file << "3 " << tri.indices[0] << " " << tri.indices[1] << " " << tri.indices[2] << "\n";
        }
        
        file << "\nCELL_TYPES " << numCells << "\n";
        for (int i = 0; i < numCells; ++i) {
            file << "5\n";  // VTK_TRIANGLE
        }
    }
    
    // Point data (scalars)
    if (!stress.empty() || !temperature.empty() || !wear.empty()) {
        file << "\nPOINT_DATA " << mesh.nodes.size() << "\n";
        
        // Stress field
        if (stress.size() == mesh.nodes.size()) {
            file << "SCALARS VonMisesStress_MPa double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (double s : stress) {
                file << std::scientific << std::setprecision(6) << (s / 1e6) << "\n";
            }
        }
        
        // Temperature field
        if (temperature.size() == mesh.nodes.size()) {
            file << "\nSCALARS Temperature_C double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (double t : temperature) {
                file << std::fixed << std::setprecision(2) << t << "\n";
            }
        }
        
        // Wear field
        if (wear.size() == mesh.nodes.size()) {
            file << "\nSCALARS Wear_mm double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (double w : wear) {
                file << std::scientific << std::setprecision(6) << (w * 1000.0) << "\n";  // m to mm
            }
        }
        
        // Displacement magnitude
        file << "\nSCALARS Displacement_mm double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& node : mesh.nodes) {
            Vec3 disp = node.position - node.originalPosition;
            double mag = disp.length() * 1000.0;  // m to mm
            file << std::scientific << std::setprecision(6) << mag << "\n";
        }
        
        // Displacement vector
        file << "\nVECTORS Displacement double\n";
        for (const auto& node : mesh.nodes) {
            Vec3 disp = node.position - node.originalPosition;
            file << std::scientific << std::setprecision(6)
                 << disp.x << " " << disp.y << " " << disp.z << "\n";
        }
    }
    
    file.close();
}


void VTKExporter::exportParticles(int step, double time,
                                   const std::vector<SPHParticle>& particles) {
    (void)time;
    std::ostringstream filename;
    filename << m_outputDir << "/particles_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    exportParticles(filename.str(), particles);
}

void VTKExporter::exportMetrics(int step, double time, double maxStress, double maxTemp) {
    // Optionally log metrics to a separate file, but here we just store them
    (void)step; (void)time; (void)maxStress; (void)maxTemp;
}

void VTKExporter::exportParticles(const std::string& filename,
                                   const std::vector<SPHParticle>& particles) {
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
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "EdgePredict SPH Particles\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    
    // Points
    file << "POINTS " << numActive << " double\n";
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            file << std::scientific << std::setprecision(8)
                 << p.x << " " << p.y << " " << p.z << "\n";
        }
    }
    
    // Vertices (for point cloud)
    file << "\nVERTICES " << numActive << " " << (numActive * 2) << "\n";
    for (int i = 0; i < numActive; ++i) {
        file << "1 " << i << "\n";
    }
    
    // Scalar data
    file << "\nPOINT_DATA " << numActive << "\n";
    
    // Temperature
    file << "SCALARS Temperature_C double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            file << std::fixed << std::setprecision(2) << p.temperature << "\n";
        }
    }
    
    // Pressure
    file << "\nSCALARS Pressure_MPa double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            file << std::scientific << std::setprecision(6) << (p.pressure / 1e6) << "\n";
        }
    }
    
    // Velocity magnitude
    file << "\nSCALARS Velocity_m_s double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            double vmag = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
            file << std::scientific << std::setprecision(6) << vmag << "\n";
        }
    }
    
    // Status (as integer)
    file << "\nSCALARS ParticleStatus int 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            file << static_cast<int>(p.status) << "\n";
        }
    }
    
    // Velocity vector
    file << "\nVECTORS Velocity double\n";
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            file << std::scientific << std::setprecision(6)
                 << p.vx << " " << p.vy << " " << p.vz << "\n";
        }
    }
    
    file.close();
}

} // namespace edgepredict
