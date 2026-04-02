/**
 * @file HDF5Exporter.cpp
 * @brief HDF5 export implementation with fallback to binary format
 */

#include "HDF5Exporter.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace edgepredict {

HDF5Exporter::HDF5Exporter(const std::string& outputDir) 
    : m_outputDir(outputDir) {
    std::filesystem::create_directories(outputDir);
}

HDF5Exporter::~HDF5Exporter() {
    closeFile();
}

void HDF5Exporter::createFile(const std::string& filename) {
    m_currentFile = m_outputDir + "/" + filename;
    
#ifdef HAS_HDF5
    try {
        m_file = std::make_unique<H5::H5File>(m_currentFile, H5F_ACC_TRUNC);
        
        // Create groups
        m_file->createGroup("/particles");
        m_file->createGroup("/nodes");
        m_file->createGroup("/timeseries");
        m_file->createGroup("/metadata");
        
        std::cout << "[HDF5Exporter] Created file: " << m_currentFile << std::endl;
    } catch (const H5::Exception& e) {
        std::cerr << "[HDF5Exporter] Failed to create file: " << e.getDetailMsg() << std::endl;
    }
#else
    std::cout << "[HDF5Exporter] HDF5 not available, using binary fallback" << std::endl;
    m_currentFile = m_outputDir + "/" + filename + ".bin";
#endif
}

void HDF5Exporter::closeFile() {
#ifdef HAS_HDF5
    if (m_file) {
        m_file->close();
        m_file.reset();
    }
#endif
}

void HDF5Exporter::exportStep(int step, double time, const Mesh& mesh) {
#ifdef HAS_HDF5
    if (!m_file) {
        createFile("simulation.h5");
    }
    
    try {
        // Create step group
        std::string stepGroup = "/particles/step_" + std::to_string(step);
        m_file->createGroup(stepGroup);
        
        // Write time attribute
        hsize_t dims[1] = {1};
        H5::DataSpace attrSpace(1, dims);
        H5::Attribute timeAttr = m_file->openGroup(stepGroup).createAttribute(
            "time", H5::PredType::NATIVE_DOUBLE, attrSpace);
        timeAttr.write(H5::PredType::NATIVE_DOUBLE, &time);
        
        // Write node data
        if (!mesh.nodes.empty()) {
            std::vector<double> x, y, z, stress, temp, wear;
            x.reserve(mesh.nodes.size());
            y.reserve(mesh.nodes.size());
            z.reserve(mesh.nodes.size());
            stress.reserve(mesh.nodes.size());
            temp.reserve(mesh.nodes.size());
            wear.reserve(mesh.nodes.size());
            
            for (const auto& node : mesh.nodes) {
                x.push_back(node.position.x);
                y.push_back(node.position.y);
                z.push_back(node.position.z);
                stress.push_back(node.stress);
                temp.push_back(node.temperature);
                wear.push_back(node.accumulatedWear);
            }
            
            hsize_t nodeDims[1] = {mesh.nodes.size()};
            H5::DataSpace nodeSpace(1, nodeDims);
            
            std::string nodeGroup = "/nodes/step_" + std::to_string(step);
            m_file->createGroup(nodeGroup);
            
            auto createDataset = [&](const std::string& name, const std::vector<double>& data) {
                H5::DSetCreatPropList plist;
                plist.setChunk(1, nodeDims);
                plist.setDeflate(m_compressionLevel);
                
                H5::DataSet ds = m_file->createDataSet(
                    nodeGroup + "/" + name, H5::PredType::NATIVE_DOUBLE, nodeSpace, plist);
                ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            };
            
            createDataset("x", x);
            createDataset("y", y);
            createDataset("z", z);
            createDataset("stress", stress);
            createDataset("temperature", temp);
            createDataset("wear", wear);
        }
        
    } catch (const H5::Exception& e) {
        std::cerr << "[HDF5Exporter] Error exporting step: " << e.getDetailMsg() << std::endl;
    }
#else
    writeBinaryFallback(step, time, mesh);
#endif
}

void HDF5Exporter::exportFinal(const Config& config, const Mesh& mesh) {
    (void)mesh;
#ifdef HAS_HDF5
    if (!m_file) return;
    
    try {
        // Write simulation metadata
        H5::Group metadata = m_file->openGroup("/metadata");
        
        hsize_t dims[1] = {1};
        H5::DataSpace attrSpace(1, dims);
        
        // Material properties
        const auto& material = config.getMaterial();
        double density = material.density;
        double yieldStrength = material.yieldStrength;
        
        H5::Attribute densityAttr = metadata.createAttribute("density", H5::PredType::NATIVE_DOUBLE, attrSpace);
        densityAttr.write(H5::PredType::NATIVE_DOUBLE, &density);
        
        H5::Attribute yieldAttr = metadata.createAttribute("yield_strength", H5::PredType::NATIVE_DOUBLE, attrSpace);
        yieldAttr.write(H5::PredType::NATIVE_DOUBLE, &yieldStrength);
        
        // Simulation parameters
        const auto& sim = config.getSimulation();
        double dt = sim.timeStepDuration;
        double duration = sim.numSteps * sim.timeStepDuration;
        
        H5::Attribute dtAttr = metadata.createAttribute("time_step", H5::PredType::NATIVE_DOUBLE, attrSpace);
        dtAttr.write(H5::PredType::NATIVE_DOUBLE, &dt);
        
        H5::Attribute durAttr = metadata.createAttribute("duration", H5::PredType::NATIVE_DOUBLE, attrSpace);
        durAttr.write(H5::PredType::NATIVE_DOUBLE, &duration);
        
        std::cout << "[HDF5Exporter] Wrote final metadata" << std::endl;
        
    } catch (const H5::Exception& e) {
        std::cerr << "[HDF5Exporter] Error exporting metadata: " << e.getDetailMsg() << std::endl;
    }
    
    closeFile();
#else
    std::cout << "[HDF5Exporter] Final export (binary fallback)" << std::endl;
#endif
}

void HDF5Exporter::exportParticles(int step, double time, const std::vector<SPHParticle>& particles) {
    (void)time;
#ifdef HAS_HDF5
    if (!m_file) {
        createFile("simulation.h5");
    }
    
    try {
        std::string groupName = "/particles/step_" + std::to_string(step);
        
        // Check if group exists, create if not
        try {
            m_file->openGroup(groupName);
        } catch (...) {
            m_file->createGroup(groupName);
        }
        
        // Prepare data arrays
        std::vector<double> x, y, z, vx, vy, vz, temp, pressure;
        std::vector<int> status;
        
        for (const auto& p : particles) {
            if (p.status == ParticleStatus::INACTIVE) continue;
            
            x.push_back(p.x);
            y.push_back(p.y);
            z.push_back(p.z);
            vx.push_back(p.vx);
            vy.push_back(p.vy);
            vz.push_back(p.vz);
            temp.push_back(p.temperature);
            pressure.push_back(p.pressure);
            status.push_back(static_cast<int>(p.status));
        }
        
        if (x.empty()) return;
        
        hsize_t dims[1] = {x.size()};
        H5::DataSpace space(1, dims);
        
        H5::DSetCreatPropList plist;
        plist.setChunk(1, dims);
        plist.setDeflate(m_compressionLevel);
        
        auto writeDouble = [&](const std::string& name, const std::vector<double>& data) {
            H5::DataSet ds = m_file->createDataSet(groupName + "/" + name, 
                                                    H5::PredType::NATIVE_DOUBLE, space, plist);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        writeDouble("x", x);
        writeDouble("y", y);
        writeDouble("z", z);
        writeDouble("vx", vx);
        writeDouble("vy", vy);
        writeDouble("vz", vz);
        writeDouble("temperature", temp);
        writeDouble("pressure", pressure);
        
        H5::DataSet statusDs = m_file->createDataSet(groupName + "/status", 
                                                      H5::PredType::NATIVE_INT, space, plist);
        statusDs.write(status.data(), H5::PredType::NATIVE_INT);
        
    } catch (const H5::Exception& e) {
        std::cerr << "[HDF5Exporter] Error exporting particles: " << e.getDetailMsg() << std::endl;
    }
#endif
}

void HDF5Exporter::exportNodes(int step, double time, const std::vector<FEMNode>& nodes) {
    (void)time;
#ifdef HAS_HDF5
    if (!m_file) {
        createFile("simulation.h5");
    }
    
    try {
        std::string groupName = "/nodes/step_" + std::to_string(step);
        
        try {
            m_file->openGroup(groupName);
        } catch (...) {
            m_file->createGroup(groupName);
        }
        
        std::vector<double> x, y, z, stress, temp, wear;
        
        for (const auto& node : nodes) {
            x.push_back(node.position.x);
            y.push_back(node.position.y);
            z.push_back(node.position.z);
            stress.push_back(node.stress);
            temp.push_back(node.temperature);
            wear.push_back(node.accumulatedWear);
        }
        
        if (x.empty()) return;
        
        hsize_t dims[1] = {x.size()};
        H5::DataSpace space(1, dims);
        
        H5::DSetCreatPropList plist;
        plist.setChunk(1, dims);
        plist.setDeflate(m_compressionLevel);
        
        auto writeDouble = [&](const std::string& name, const std::vector<double>& data) {
            H5::DataSet ds = m_file->createDataSet(groupName + "/" + name, 
                                                    H5::PredType::NATIVE_DOUBLE, space, plist);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        writeDouble("x", x);
        writeDouble("y", y);
        writeDouble("z", z);
        writeDouble("stress", stress);
        writeDouble("temperature", temp);
        writeDouble("wear", wear);
        
    } catch (const H5::Exception& e) {
        std::cerr << "[HDF5Exporter] Error exporting nodes: " << e.getDetailMsg() << std::endl;
    }
#endif
}

void HDF5Exporter::appendTimeSeries(const std::string& name, double time, double value) {
#ifdef HAS_HDF5
    if (!m_file) return;
    
    try {
        std::string dsName = "/timeseries/" + name;
        
        // Check if dataset exists
        bool exists = false;
        try {
            m_file->openDataSet(dsName);
            exists = true;
        } catch (...) {}
        
        if (!exists) {
            // Create extensible dataset
            hsize_t dims[2] = {0, 2};
            hsize_t maxDims[2] = {H5S_UNLIMITED, 2};
            H5::DataSpace space(2, dims, maxDims);
            
            hsize_t chunk[2] = {100, 2};
            H5::DSetCreatPropList plist;
            plist.setChunk(2, chunk);
            plist.setDeflate(m_compressionLevel);
            
            m_file->createDataSet(dsName, H5::PredType::NATIVE_DOUBLE, space, plist);
        }
        
        // Append data
        H5::DataSet ds = m_file->openDataSet(dsName);
        H5::DataSpace fileSpace = ds.getSpace();
        
        hsize_t currentDims[2];
        fileSpace.getSimpleExtentDims(currentDims);
        
        hsize_t newDims[2] = {currentDims[0] + 1, 2};
        ds.extend(newDims);
        
        hsize_t offset[2] = {currentDims[0], 0};
        hsize_t count[2] = {1, 2};
        
        fileSpace = ds.getSpace();
        fileSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
        
        H5::DataSpace memSpace(2, count);
        double data[2] = {time, value};
        ds.write(data, H5::PredType::NATIVE_DOUBLE, memSpace, fileSpace);
        
    } catch (const H5::Exception& e) {
        std::cerr << "[HDF5Exporter] Error appending time series: " << e.getDetailMsg() << std::endl;
    }
#endif
}

void HDF5Exporter::writeBinaryFallback(int step, double time, const Mesh& mesh) {
    std::ostringstream filename;
    filename << m_outputDir << "/step_" << std::setfill('0') << std::setw(6) << step << ".bin";
    
    std::string filename_str = filename.str();
    std::ofstream file(filename_str.c_str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[HDF5Exporter] Failed to open binary file: " << filename.str() << std::endl;
        return;
    }
    
    // Header
    file.write(reinterpret_cast<const char*>(&step), sizeof(int));
    file.write(reinterpret_cast<const char*>(&time), sizeof(double));
    
    int numNodes = static_cast<int>(mesh.nodes.size());
    file.write(reinterpret_cast<const char*>(&numNodes), sizeof(int));
    
    // Node data
    for (const auto& node : mesh.nodes) {
        file.write(reinterpret_cast<const char*>(&node.position.x), sizeof(double));
        file.write(reinterpret_cast<const char*>(&node.position.y), sizeof(double));
        file.write(reinterpret_cast<const char*>(&node.position.z), sizeof(double));
        file.write(reinterpret_cast<const char*>(&node.stress), sizeof(double));
        file.write(reinterpret_cast<const char*>(&node.temperature), sizeof(double));
        file.write(reinterpret_cast<const char*>(&node.accumulatedWear), sizeof(double));
    }
    
    file.close();
}

} // namespace edgepredict
