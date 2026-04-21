/**
 * @file GeometryLoader.cpp
 * @brief Geometry loading implementation
 */

#include "GeometryLoader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_map>

// OpenCASCADE includes for CAD loading
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>

namespace edgepredict {

GeometryLoader::GeometryLoader() = default;
GeometryLoader::~GeometryLoader() = default;

bool GeometryLoader::detectFormat(const std::string& path, std::string& format) {
    size_t dotPos = path.rfind('.');
    if (dotPos == std::string::npos) {
        m_lastError = "No file extension found";
        return false;
    }
    
    format = path.substr(dotPos + 1);
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);
    return true;
}

bool GeometryLoader::load(const std::string& path, Mesh& mesh) {
    std::string format;
    if (!detectFormat(path, format)) {
        return false;
    }
    
    std::cout << "[GeometryLoader] Loading: " << path << " (format: " << format << ")" << std::endl;
    
    if (format == "stl") {
        return loadSTL(path, mesh);
    } else if (format == "step" || format == "stp") {
        return loadSTEP(path, mesh);
    } else if (format == "iges" || format == "igs") {
        return loadIGES(path, mesh);
    } else {
        m_lastError = "Unsupported format: " + format;
        return false;
    }
}

bool GeometryLoader::loadSTL(const std::string& path, Mesh& mesh) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Cannot open file: " + path;
        return false;
    }
    
    mesh.clear();
    
    // Check if binary or ASCII STL
    char header[80];
    file.read(header, 80);
    
    // Read number of triangles (binary STL)
    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));
    
    // Check if this looks like binary (file size should match)
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t expectedBinarySize = 84 + numTriangles * 50;
    
    bool isBinary = (fileSize == expectedBinarySize);
    file.seekg(0, std::ios::beg);
    
    if (isBinary) {
        // Binary STL
        file.seekg(84);  // Skip header + triangle count
        
        std::unordered_map<size_t, int> vertexMap;
        
        for (uint32_t i = 0; i < numTriangles; ++i) {
            // Read normal (not used, we compute our own)
            float normal[3];
            file.read(reinterpret_cast<char*>(normal), 12);
            
            Triangle tri;
            
            // Read 3 vertices
            for (int v = 0; v < 3; ++v) {
                float vertex[3];
                file.read(reinterpret_cast<char*>(vertex), 12);
                
                // Create vertex hash for deduplication
                Vec3 pos(vertex[0], vertex[1], vertex[2]);
                size_t hash = std::hash<double>{}(pos.x) ^ 
                             (std::hash<double>{}(pos.y) << 1) ^
                             (std::hash<double>{}(pos.z) << 2);
                
                auto it = vertexMap.find(hash);
                if (it != vertexMap.end()) {
                    tri.indices[v] = it->second;
                } else {
                    int idx = static_cast<int>(mesh.nodes.size());
                    FEMNode node;
                    node.position = pos;
                    node.originalPosition = pos;
                    mesh.nodes.push_back(node);
                    vertexMap[hash] = idx;
                    tri.indices[v] = idx;
                }
            }
            
            mesh.triangles.push_back(tri);
            
            // Skip attribute byte count
            uint16_t attr;
            file.read(reinterpret_cast<char*>(&attr), 2);
        }
    } else {
        // ASCII STL - simple parser
        file.seekg(0);
        std::string line;
        
        std::unordered_map<size_t, int> vertexMap;
        Triangle currentTri;
        int vertexIdx = 0;
        
        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            
            if (line.rfind("vertex", 0) == 0) {
                float x, y, z;
                if (sscanf(line.c_str(), "vertex %f %f %f", &x, &y, &z) == 3) {
                    Vec3 pos(x, y, z);
                    size_t hash = std::hash<double>{}(pos.x) ^ 
                                 (std::hash<double>{}(pos.y) << 1) ^
                                 (std::hash<double>{}(pos.z) << 2);
                    
                    auto it = vertexMap.find(hash);
                    if (it != vertexMap.end()) {
                        currentTri.indices[vertexIdx] = it->second;
                    } else {
                        int idx = static_cast<int>(mesh.nodes.size());
                        FEMNode node;
                        node.position = pos;
                        node.originalPosition = pos;
                        mesh.nodes.push_back(node);
                        vertexMap[hash] = idx;
                        currentTri.indices[vertexIdx] = idx;
                    }
                    
                    vertexIdx++;
                    if (vertexIdx == 3) {
                        mesh.triangles.push_back(currentTri);
                        vertexIdx = 0;
                    }
                }
            }
        }
    }
    
    // Compute normals
    computeNormals(mesh);
    
    std::cout << "[GeometryLoader] Loaded STL: " << mesh.nodeCount() << " nodes, " 
              << mesh.triangleCount() << " triangles" << std::endl;
    
    return true;
}

bool GeometryLoader::loadSTEP(const std::string& path, Mesh& mesh) {
    STEPControl_Reader reader;
    
    IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
    if (status != IFSelect_RetDone) {
        m_lastError = "Failed to read STEP file: " + path;
        return false;
    }
    
    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    
    if (shape.IsNull()) {
        m_lastError = "STEP file produced null shape";
        return false;
    }
    
    return meshFromCADShape(shape, mesh, m_defaultDeflection);
}

bool GeometryLoader::loadIGES(const std::string& path, Mesh& mesh) {
    IGESControl_Reader reader;
    
    IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
    if (status != IFSelect_RetDone) {
        m_lastError = "Failed to read IGES file: " + path;
        return false;
    }
    
    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    
    if (shape.IsNull()) {
        m_lastError = "IGES file produced null shape";
        return false;
    }
    
    return meshFromCADShape(shape, mesh, m_defaultDeflection);
}

bool GeometryLoader::meshFromCADShape(const TopoDS_Shape& shape, Mesh& mesh, double deflection) {
    mesh.clear();
    
    // Tessellate the shape
    BRepMesh_IncrementalMesh(shape, deflection);
    
    // Map to deduplicate vertices
    std::unordered_map<size_t, int> vertexMap;
    
    // Extract triangles from all faces
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        TopLoc_Location location;
        Handle(Poly_Triangulation) triangulation = 
            BRep_Tool::Triangulation(TopoDS::Face(explorer.Current()), location);
        
        if (triangulation.IsNull()) continue;
        
        for (int i = 1; i <= triangulation->NbTriangles(); ++i) {
            Standard_Integer n1, n2, n3;
            triangulation->Triangle(i).Get(n1, n2, n3);
            
            Triangle tri;
            Standard_Integer nodeIndices[3] = {n1, n2, n3};
            
            for (int v = 0; v < 3; ++v) {
                gp_Pnt point = triangulation->Node(nodeIndices[v]).Transformed(location);
                Vec3 pos(point.X(), point.Y(), point.Z());
                
                // Hash for deduplication
                size_t hash = std::hash<double>{}(pos.x * 1000) ^ 
                             (std::hash<double>{}(pos.y * 1000) << 1) ^
                             (std::hash<double>{}(pos.z * 1000) << 2);
                
                auto it = vertexMap.find(hash);
                if (it != vertexMap.end()) {
                    tri.indices[v] = it->second;
                } else {
                    int idx = static_cast<int>(mesh.nodes.size());
                    FEMNode node;
                    node.position = pos;
                    node.originalPosition = pos;
                    mesh.nodes.push_back(node);
                    vertexMap[hash] = idx;
                    tri.indices[v] = idx;
                }
            }
            
            mesh.triangles.push_back(tri);
        }
    }
    
    computeNormals(mesh);
    
    std::cout << "[GeometryLoader] Loaded CAD: " << mesh.nodeCount() << " nodes, " 
              << mesh.triangleCount() << " triangles" << std::endl;
    
    return true;
}

void GeometryLoader::scaleMesh(Mesh& mesh, double scale) {
    for (auto& node : mesh.nodes) {
        node.position = node.position * scale;
        node.originalPosition = node.originalPosition * scale;
    }
}

void GeometryLoader::computeNormals(Mesh& mesh) {
    for (auto& tri : mesh.triangles) {
        if (tri.indices[0] >= 0 && tri.indices[0] < static_cast<int>(mesh.nodes.size()) &&
            tri.indices[1] >= 0 && tri.indices[1] < static_cast<int>(mesh.nodes.size()) &&
            tri.indices[2] >= 0 && tri.indices[2] < static_cast<int>(mesh.nodes.size())) {
            
            const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
            const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
            const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
            
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            tri.normal = edge1.cross(edge2).normalized();
        }
    }
}

void GeometryLoader::getBoundingBox(const Mesh& mesh, Vec3& minCorner, Vec3& maxCorner) {
    if (mesh.nodes.empty()) {
        minCorner = Vec3::zero();
        maxCorner = Vec3::zero();
        return;
    }
    
    minCorner = mesh.nodes[0].position;
    maxCorner = mesh.nodes[0].position;
    
    for (const auto& node : mesh.nodes) {
        minCorner.x = std::min(minCorner.x, node.position.x);
        minCorner.y = std::min(minCorner.y, node.position.y);
        minCorner.z = std::min(minCorner.z, node.position.z);
        maxCorner.x = std::max(maxCorner.x, node.position.x);
        maxCorner.y = std::max(maxCorner.y, node.position.y);
        maxCorner.z = std::max(maxCorner.z, node.position.z);
    }
}

void GeometryLoader::alignToTip(Mesh& mesh, int alignAxis) {
    if (mesh.nodes.empty()) return;
    
    // Clamp axis to valid range
    if (alignAxis < 0 || alignAxis > 2) alignAxis = 2;
    
    // Step 1: Find minimum coordinate along the align axis (= the cutting tip)
    double minAlignCoord = 1e30;
    double sumX = 0, sumY = 0;
    
    for (const auto& node : mesh.nodes) {
        double alignCoord;
        switch (alignAxis) {
            case 0: alignCoord = node.position.x; break;
            case 1: alignCoord = node.position.y; break;
            default: alignCoord = node.position.z; break;
        }
        if (alignCoord < minAlignCoord) minAlignCoord = alignCoord;
        sumX += node.position.x;
        sumY += node.position.y;
    }
    
    // Step 2: Compute centroid for X,Y centering
    double n = static_cast<double>(mesh.nodes.size());
    double centroidX = sumX / n;
    double centroidY = sumY / n;
    
    // Step 3: Build translation so tip = (0,0,0)
    Vec3 offset;
    switch (alignAxis) {
        case 0: // Tool points along X
            offset = Vec3(-minAlignCoord, -centroidY, 0);
            break;
        case 1: // Tool points along Y
            offset = Vec3(-centroidX, -minAlignCoord, 0);
            break;
        default: // Tool points along Z
            offset = Vec3(-centroidX, -centroidY, -minAlignCoord);
            break;
    }
    
    // Step 4: Apply to all vertices
    for (auto& node : mesh.nodes) {
        node.position = node.position + offset;
        node.originalPosition = node.originalPosition + offset;
    }
    
    const char* axisName[] = {"X", "Y", "Z"};
    std::cout << "[GeometryLoader] alignToTip(axis=" << axisName[alignAxis] 
              << "): offset=(" << offset.x*1000 << ", " << offset.y*1000 
              << ", " << offset.z*1000 << ") mm" << std::endl;
    
    // Verify tip is at origin
    Vec3 minC, maxC;
    getBoundingBox(mesh, minC, maxC);
    std::cout << "[GeometryLoader] Post-align bounds: "
              << "X[" << minC.x*1000 << "," << maxC.x*1000 << "] "
              << "Y[" << minC.y*1000 << "," << maxC.y*1000 << "] "
              << "Z[" << minC.z*1000 << "," << maxC.z*1000 << "] mm" << std::endl;
}

void GeometryLoader::applyMachineOffsets(Mesh& mesh, const double g54[3], 
                                          double toolLengthOffset, int alignAxis) {
    if (mesh.nodes.empty()) return;
    
    // Build offset: G54 position + tool length along the correct axis
    Vec3 offset(g54[0], g54[1], g54[2]);
    switch (alignAxis) {
        case 0: offset.x += toolLengthOffset; break;
        case 1: offset.y += toolLengthOffset; break;
        default: offset.z += toolLengthOffset; break;
    }
    
    // Skip if zero offset
    if (std::abs(offset.x) < 1e-12 && std::abs(offset.y) < 1e-12 && std::abs(offset.z) < 1e-12) {
        return;
    }
    
    for (auto& node : mesh.nodes) {
        node.position = node.position + offset;
        node.originalPosition = node.originalPosition + offset;
    }
    
    std::cout << "[GeometryLoader] Machine offsets: G54=(" 
              << g54[0]*1000 << "," << g54[1]*1000 << "," << g54[2]*1000 
              << ")mm TLO=" << toolLengthOffset*1000 << "mm" << std::endl;
}

void GeometryLoader::voxelizeToTetMesh(Mesh& mesh, double elementSize, int maxNodes) {
    if (mesh.nodes.empty() || mesh.triangles.empty()) return;
    
    Vec3 minC, maxC;
    getBoundingBox(mesh, minC, maxC);
    
    // Add margin
    minC = minC - Vec3(elementSize, elementSize, elementSize);
    maxC = maxC + Vec3(elementSize, elementSize, elementSize);
    
    int nx = std::max(1, static_cast<int>((maxC.x - minC.x) / elementSize));
    int ny = std::max(1, static_cast<int>((maxC.y - minC.y) / elementSize));
    int nz = std::max(1, static_cast<int>((maxC.z - minC.z) / elementSize));
    
    // Auto-scale elementSize if maxNodes exceeded.
    // Each voxel produces ~1 node.
    long long totalVoxels = static_cast<long long>(nx) * ny * nz;
    if (totalVoxels > maxNodes * 2) {
        double scaleFactor = std::cbrt(static_cast<double>(totalVoxels) / (maxNodes * 2));
        elementSize *= scaleFactor;
        nx = std::max(1, static_cast<int>((maxC.x - minC.x) / elementSize));
        ny = std::max(1, static_cast<int>((maxC.y - minC.y) / elementSize));
        nz = std::max(1, static_cast<int>((maxC.z - minC.z) / elementSize));
        std::cout << "[GeometryLoader] Warning: elementSize increased to " << elementSize 
                  << " to stay under maxNodes limit." << std::endl;
    }
    
    auto isInside = [&](const Vec3& p) -> bool {
        int intersections = 0;
        Vec3 rayDir(1.0, 1.2345e-4, 3.4567e-5); // slightly off-axis to avoid perfect edge hits
        rayDir = rayDir.normalized();
        
        for (const auto& tri : mesh.triangles) {
            Vec3 v0 = mesh.nodes[tri.indices[0]].position;
            Vec3 v1 = mesh.nodes[tri.indices[1]].position;
            Vec3 v2 = mesh.nodes[tri.indices[2]].position;
            
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec3 h = rayDir.cross(edge2);
            double a = edge1.dot(h);
            
            if (a > -1e-8 && a < 1e-8) continue;
                
            double f = 1.0 / a;
            Vec3 s = p - v0;
            double u = f * s.dot(h);
            
            if (u < 0.0 || u > 1.0) continue;
                
            Vec3 q = s.cross(edge1);
            double v = f * rayDir.dot(q);
            
            if (v < 0.0 || u + v > 1.0) continue;
                
            double t = f * edge2.dot(q);
            if (t > 1e-8) {
                intersections++;
            }
        }
        return (intersections % 2) == 1;
    };
    
    // Grid 3D array mapping (i,j,k) to node id
    std::vector<int> nodeGrid((nx+1) * (ny+1) * (nz+1), -1);
    auto getNodeIdx = [&](int i, int j, int k) -> int& {
        return nodeGrid[i + j*(nx+1) + k*(nx+1)*(ny+1)];
    };
    
    Mesh newMesh;
    std::vector<bool> activeVoxels(nx * ny * nz, false);
    
    std::cout << "[GeometryLoader] Voxelizing tool... (" << nx << "x" << ny << "x" << nz << ")" << std::endl;
    int voxelCount = 0;
    
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Vec3 center = minC + Vec3((i+0.5)*elementSize, (j+0.5)*elementSize, (k+0.5)*elementSize);
                if (isInside(center)) {
                    activeVoxels[i + j*nx + k*nx*ny] = true;
                    voxelCount++;
                }
            }
        }
    }
    
    std::cout << "[GeometryLoader] Active voxels: " << voxelCount << std::endl;
    
    // Create nodes for active voxels. We make sure all 8 corners are created.
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (activeVoxels[i + j*nx + k*nx*ny]) {
                    for (int dk = 0; dk <= 1; ++dk) {
                        for (int dj = 0; dj <= 1; ++dj) {
                            for (int di = 0; di <= 1; ++di) {
                                int& nIdx = getNodeIdx(i+di, j+dj, k+dk);
                                if (nIdx == -1) {
                                    FEMNode node;
                                    node.position = minC + Vec3((i+di)*elementSize, (j+dj)*elementSize, (k+dk)*elementSize);
                                    node.originalPosition = node.position;
                                    node.status = NodeStatus::OK;
                                    nIdx = static_cast<int>(newMesh.nodes.size());
                                    newMesh.nodes.push_back(node);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 6-split of a cube into tetrahedra
    const int tetSplits[6][4] = {
        {0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7},
        {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}
    };
    
    std::vector<int> voxelToFirstTet(nx * ny * nz, -1);
    
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int voxIdx = i + j*nx + k*nx*ny;
                if (activeVoxels[voxIdx]) {
                    voxelToFirstTet[voxIdx] = static_cast<int>(newMesh.elements.size());
                    
                    int c[8];
                    c[0] = getNodeIdx(i, j, k);
                    c[1] = getNodeIdx(i+1, j, k);
                    c[2] = getNodeIdx(i, j+1, k);
                    c[3] = getNodeIdx(i+1, j+1, k);
                    c[4] = getNodeIdx(i, j, k+1);
                    c[5] = getNodeIdx(i+1, j, k+1);
                    c[6] = getNodeIdx(i, j+1, k+1);
                    c[7] = getNodeIdx(i+1, j+1, k+1);
                    
                    for (int t = 0; t < 6; ++t) {
                        FEMElement el;
                        el.nodeIndices[0] = c[tetSplits[t][0]];
                        el.nodeIndices[1] = c[tetSplits[t][1]];
                        el.nodeIndices[2] = c[tetSplits[t][2]];
                        el.nodeIndices[3] = c[tetSplits[t][3]];
                        newMesh.elements.push_back(el);
                    }
                }
            }
        }
    }
    
    std::cout << "[GeometryLoader] Volumetric Mesh Generated: " << newMesh.nodes.size() 
              << " nodes, " << newMesh.elements.size() << " tetrahedra" << std::endl;
              
    mesh.tetNodes = newMesh.nodes;
    mesh.elements = newMesh.elements;
    
    // --- BUILD KINEMATIC EMBEDDING CONSTRAINTS ---
    auto computeBarycentric = [](const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, double w[4]) {
        Vec3 vab = b - a;
        Vec3 vac = c - a;
        Vec3 vad = d - a;
        
        double vol6 = vab.dot(vac.cross(vad));
        if (std::abs(vol6) < 1e-15) {
            w[0] = 0.25; w[1] = 0.25; w[2] = 0.25; w[3] = 0.25;
            return;
        }
        
        double inv6Vol = 1.0 / vol6;
        w[1] = (p - a).dot(vac.cross(vad)) * inv6Vol;
        w[2] = vab.dot((p - a).cross(vad)) * inv6Vol;
        w[3] = vab.dot(vac.cross(p - a)) * inv6Vol;
        w[0] = 1.0 - w[1] - w[2] - w[3];
    };
    
    mesh.embedConstraints.clear();
    for (int nIdx = 0; nIdx < (int)mesh.nodes.size(); ++nIdx) {
        Vec3 p = mesh.nodes[nIdx].position;
        
        int i = static_cast<int>(std::floor((p.x - minC.x) / elementSize));
        int j = static_cast<int>(std::floor((p.y - minC.y) / elementSize));
        int k = static_cast<int>(std::floor((p.z - minC.z) / elementSize));
        
        int bestTet = -1;
        double bestMinW = -1e10;
        double bestW[4] = {1, 0, 0, 0};
        
        // Search outward up to a few voxels away
        for (int r = 0; r <= 5; ++r) {
            for (int dk = -r; dk <= r; ++dk) {
                for (int dj = -r; dj <= r; ++dj) {
                    for (int di = -r; di <= r; ++di) {
                        int ci = i + di; int cj = j + dj; int ck = k + dk;
                        if (ci < 0 || ci >= nx || cj < 0 || cj >= ny || ck < 0 || ck >= nz) continue;
                        
                        int firstTet = voxelToFirstTet[ci + cj*nx + ck*nx*ny];
                        if (firstTet != -1) {
                            for(int t = 0; t < 6; ++t) {
                                int tetIdx = firstTet + t;
                                const auto& el = mesh.elements[tetIdx];
                                double w[4];
                                computeBarycentric(p, 
                                    mesh.tetNodes[el.nodeIndices[0]].position,
                                    mesh.tetNodes[el.nodeIndices[1]].position,
                                    mesh.tetNodes[el.nodeIndices[2]].position,
                                    mesh.tetNodes[el.nodeIndices[3]].position,
                                    w);
                                    
                                double minW = std::min({w[0], w[1], w[2], w[3]});
                                if (minW > bestMinW) {
                                    bestMinW = minW;
                                    bestTet = tetIdx;
                                    bestW[0] = w[0]; bestW[1] = w[1]; bestW[2] = w[2]; bestW[3] = w[3];
                                }
                            }
                        }
                    }
                }
            }
            if (bestMinW >= -1e-4) break; // Solidly inside or extremely close
        }
        
        FEMEmbedConstraint ec;
        ec.surfaceNodeIdx = nIdx;
        if (bestTet != -1) {
            ec.tetNodeIndices[0] = mesh.elements[bestTet].nodeIndices[0];
            ec.tetNodeIndices[1] = mesh.elements[bestTet].nodeIndices[1];
            ec.tetNodeIndices[2] = mesh.elements[bestTet].nodeIndices[2];
            ec.tetNodeIndices[3] = mesh.elements[bestTet].nodeIndices[3];
        }
        ec.weights[0] = bestW[0]; 
        ec.weights[1] = bestW[1]; 
        ec.weights[2] = bestW[2]; 
        ec.weights[3] = bestW[3];
        mesh.embedConstraints.push_back(ec);
    }
    
    std::cout << "[GeometryLoader] Embedded Mesh Constraints: " << mesh.embedConstraints.size() << std::endl;
}

} // namespace edgepredict
