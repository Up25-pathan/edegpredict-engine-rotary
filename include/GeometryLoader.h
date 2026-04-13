#pragma once
/**
 * @file GeometryLoader.h
 * @brief Geometry loading for STL and CAD files (STEP/IGES)
 */

#include "Types.h"
#include "Config.h"
#include <string>
#include <memory>

// Forward declare OpenCASCADE types to avoid heavy includes
class TopoDS_Shape;

namespace edgepredict {

/**
 * @brief Loads geometry from various file formats into Mesh structure
 */
class GeometryLoader {
public:
    GeometryLoader();
    ~GeometryLoader();
    
    /**
     * @brief Load mesh from file (auto-detect format)
     * @param path Path to geometry file
     * @param mesh Output mesh
     * @return true if loading succeeded
     */
    bool load(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Load mesh from STL file
     */
    bool loadSTL(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Load mesh from STEP file
     */
    bool loadSTEP(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Load mesh from IGES file
     */
    bool loadIGES(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Apply scale factor to mesh
     * @param mesh Mesh to scale
     * @param scale Scale factor (e.g., 0.001 for mm to m)
     */
    static void scaleMesh(Mesh& mesh, double scale);
    
    /**
     * @brief Compute mesh normals
     */
    static void computeNormals(Mesh& mesh);
    
    /**
     * @brief Compute mesh bounding box
     */
    static void getBoundingBox(const Mesh& mesh, Vec3& minCorner, Vec3& maxCorner);
    
    /**
     * @brief Align tool mesh so cutting tip is at exact origin (0,0,0)
     * 
     * Finds the minimum coordinate along the align axis (the tip),
     * centers X/Y, and translates so tip = origin.
     * Mimics CNC tool length measurement from spindle face.
     * 
     * @param mesh Mesh to align (modified in-place)
     * @param alignAxis 0=X, 1=Y, 2=Z (default Z). -1=auto is resolved by caller.
     */
    static void alignToTip(Mesh& mesh, int alignAxis = 2);
    
    /**
     * @brief Apply Work Coordinate System offset (G54) and tool length offset
     * @param mesh Mesh to offset
     * @param g54 Work offset [X, Y, Z] in meters
     * @param toolLengthOffset Tool length in meters (added along Z)
     * @param alignAxis Which axis the tool length offset applies to
     */
    static void applyMachineOffsets(Mesh& mesh, const double g54[3], 
                                     double toolLengthOffset, int alignAxis = 2);
    
    /**
     * @brief Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }

private:
    bool detectFormat(const std::string& path, std::string& format);
    bool meshFromCADShape(const TopoDS_Shape& shape, Mesh& mesh, double deflection);
    
    std::string m_lastError;
    double m_defaultDeflection = 0.05;  // Tessellation tolerance (mm)
};

} // namespace edgepredict
