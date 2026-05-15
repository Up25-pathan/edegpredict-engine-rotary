#pragma once
/**
 * @file CoordinateSystem.h
 * @brief CNC-style hierarchical coordinate system for Vericut-level alignment
 *
 * Implements a 3-tier coordinate hierarchy:
 *   MCS (Machine Coordinate System) → WCS (Work Coordinate System, G54-G59) → TCS (Tool Coordinate System)
 *
 * All transforms use 4×4 homogeneous matrices for correct composition
 * of translations, rotations, and scales.
 */

#include "Types.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <array>

namespace edgepredict {

// ============================================================================
// 4×4 Homogeneous Transformation Matrix
// ============================================================================

/**
 * @brief 4×4 homogeneous transformation matrix (column-major storage)
 *
 * Layout (row-major for readability, stored column-major):
 *   | m[0]  m[4]  m[8]   m[12] |
 *   | m[1]  m[5]  m[9]   m[13] |
 *   | m[2]  m[6]  m[10]  m[14] |
 *   | m[3]  m[7]  m[11]  m[15] |
 *
 * Column-major matches OpenGL/GPU conventions.
 * m[12,13,14] = translation components.
 */
struct Mat4 {
    double m[16];

    // --- Constructors ---

    /** @brief Default: identity matrix */
    Mat4() {
        std::memset(m, 0, sizeof(m));
        m[0] = m[5] = m[10] = m[15] = 1.0;
    }

    /** @brief Construct from 16 values (column-major) */
    explicit Mat4(const double* data) {
        std::memcpy(m, data, sizeof(m));
    }

    // --- Static Factories ---

    static Mat4 identity() { return Mat4(); }

    static Mat4 translation(double tx, double ty, double tz) {
        Mat4 result;
        result.m[12] = tx;
        result.m[13] = ty;
        result.m[14] = tz;
        return result;
    }

    static Mat4 translation(const Vec3& t) {
        return translation(t.x, t.y, t.z);
    }

    static Mat4 scale(double sx, double sy, double sz) {
        Mat4 result;
        result.m[0]  = sx;
        result.m[5]  = sy;
        result.m[10] = sz;
        return result;
    }

    static Mat4 uniformScale(double s) {
        return scale(s, s, s);
    }

    static Mat4 rotationX(double radians) {
        Mat4 result;
        double c = std::cos(radians);
        double s = std::sin(radians);
        result.m[5]  =  c;  result.m[9]  = -s;
        result.m[6]  =  s;  result.m[10] =  c;
        return result;
    }

    static Mat4 rotationY(double radians) {
        Mat4 result;
        double c = std::cos(radians);
        double s = std::sin(radians);
        result.m[0]  =  c;  result.m[8]  =  s;
        result.m[2]  = -s;  result.m[10] =  c;
        return result;
    }

    static Mat4 rotationZ(double radians) {
        Mat4 result;
        double c = std::cos(radians);
        double s = std::sin(radians);
        result.m[0] =  c;  result.m[4] = -s;
        result.m[1] =  s;  result.m[5] =  c;
        return result;
    }

    /**
     * @brief Rotation around arbitrary axis (Rodrigues formula)
     */
    static Mat4 rotationAxis(const Vec3& axis, double radians) {
        Vec3 a = axis.normalized();
        double c = std::cos(radians);
        double s = std::sin(radians);
        double t = 1.0 - c;

        Mat4 result;
        result.m[0]  = t * a.x * a.x + c;
        result.m[1]  = t * a.x * a.y + s * a.z;
        result.m[2]  = t * a.x * a.z - s * a.y;

        result.m[4]  = t * a.x * a.y - s * a.z;
        result.m[5]  = t * a.y * a.y + c;
        result.m[6]  = t * a.y * a.z + s * a.x;

        result.m[8]  = t * a.x * a.z + s * a.y;
        result.m[9]  = t * a.y * a.z - s * a.x;
        result.m[10] = t * a.z * a.z + c;

        return result;
    }

    // --- Composition ---

    /** @brief Matrix multiplication (this * other) */
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k) {
                    sum += m[k * 4 + row] * other.m[col * 4 + k];
                }
                result.m[col * 4 + row] = sum;
            }
        }
        return result;
    }

    // --- Transform Operations ---

    /** @brief Transform a point (applies translation) */
    Vec3 transformPoint(const Vec3& p) const {
        double w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
        if (std::abs(w) < 1e-15) w = 1.0;
        return Vec3(
            (m[0] * p.x + m[4] * p.y + m[8]  * p.z + m[12]) / w,
            (m[1] * p.x + m[5] * p.y + m[9]  * p.z + m[13]) / w,
            (m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]) / w
        );
    }

    /** @brief Transform a direction vector (ignores translation) */
    Vec3 transformDirection(const Vec3& d) const {
        return Vec3(
            m[0] * d.x + m[4] * d.y + m[8]  * d.z,
            m[1] * d.x + m[5] * d.y + m[9]  * d.z,
            m[2] * d.x + m[6] * d.y + m[10] * d.z
        );
    }

    /** @brief Extract translation component */
    Vec3 getTranslation() const {
        return Vec3(m[12], m[13], m[14]);
    }

    /** @brief Set translation component */
    void setTranslation(const Vec3& t) {
        m[12] = t.x;
        m[13] = t.y;
        m[14] = t.z;
    }

    /**
     * @brief Compute inverse (for rigid transforms: transpose rotation + negate translation)
     * For general affine transforms, uses cofactor expansion.
     */
    Mat4 inverse() const {
        // For rigid-body transforms (rotation + translation only), use fast path
        // General 4x4 inverse via cofactor expansion
        Mat4 inv;
        double det;

        inv.m[0] = m[5]  * m[10] * m[15] - m[5]  * m[11] * m[14] -
                   m[9]  * m[6]  * m[15] + m[9]  * m[7]  * m[14] +
                   m[13] * m[6]  * m[11] - m[13] * m[7]  * m[10];

        inv.m[4] = -m[4]  * m[10] * m[15] + m[4]  * m[11] * m[14] +
                    m[8]  * m[6]  * m[15] - m[8]  * m[7]  * m[14] -
                    m[12] * m[6]  * m[11] + m[12] * m[7]  * m[10];

        inv.m[8] = m[4]  * m[9] * m[15] - m[4]  * m[11] * m[13] -
                   m[8]  * m[5] * m[15] + m[8]  * m[7]  * m[13] +
                   m[12] * m[5] * m[11] - m[12] * m[7]  * m[9];

        inv.m[12] = -m[4]  * m[9] * m[14] + m[4]  * m[10] * m[13] +
                     m[8]  * m[5] * m[14] - m[8]  * m[6]  * m[13] -
                     m[12] * m[5] * m[10] + m[12] * m[6]  * m[9];

        inv.m[1] = -m[1]  * m[10] * m[15] + m[1]  * m[11] * m[14] +
                    m[9]  * m[2]  * m[15] - m[9]  * m[3]  * m[14] -
                    m[13] * m[2]  * m[11] + m[13] * m[3]  * m[10];

        inv.m[5] = m[0]  * m[10] * m[15] - m[0]  * m[11] * m[14] -
                   m[8]  * m[2]  * m[15] + m[8]  * m[3]  * m[14] +
                   m[12] * m[2]  * m[11] - m[12] * m[3]  * m[10];

        inv.m[9] = -m[0]  * m[9] * m[15] + m[0]  * m[11] * m[13] +
                    m[8]  * m[1] * m[15] - m[8]  * m[3]  * m[13] -
                    m[12] * m[1] * m[11] + m[12] * m[3]  * m[9];

        inv.m[13] = m[0]  * m[9] * m[14] - m[0]  * m[10] * m[13] -
                    m[8]  * m[1] * m[14] + m[8]  * m[2]  * m[13] +
                    m[12] * m[1] * m[10] - m[12] * m[2]  * m[9];

        inv.m[2] = m[1]  * m[6] * m[15] - m[1]  * m[7] * m[14] -
                   m[5]  * m[2] * m[15] + m[5]  * m[3] * m[14] +
                   m[13] * m[2] * m[7]  - m[13] * m[3] * m[6];

        inv.m[6] = -m[0]  * m[6] * m[15] + m[0]  * m[7] * m[14] +
                    m[4]  * m[2] * m[15] - m[4]  * m[3] * m[14] -
                    m[12] * m[2] * m[7]  + m[12] * m[3] * m[6];

        inv.m[10] = m[0]  * m[5] * m[15] - m[0]  * m[7] * m[13] -
                    m[4]  * m[1] * m[15] + m[4]  * m[3] * m[13] +
                    m[12] * m[1] * m[7]  - m[12] * m[3] * m[5];

        inv.m[14] = -m[0]  * m[5] * m[14] + m[0]  * m[6] * m[13] +
                     m[4]  * m[1] * m[14] - m[4]  * m[2] * m[13] -
                     m[12] * m[1] * m[6]  + m[12] * m[2] * m[5];

        inv.m[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] +
                    m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
                    m[9] * m[2] * m[7]  + m[9] * m[3] * m[6];

        inv.m[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] -
                   m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
                   m[8] * m[2] * m[7]  - m[8] * m[3] * m[6];

        inv.m[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] +
                     m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
                     m[8] * m[1] * m[7]  + m[8] * m[3] * m[5];

        inv.m[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] -
                    m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
                    m[8] * m[1] * m[6]  - m[8] * m[2] * m[5];

        det = m[0] * inv.m[0] + m[1] * inv.m[4] + m[2] * inv.m[8] + m[3] * inv.m[12];

        if (std::abs(det) < 1e-15) {
            return Mat4::identity(); // Singular matrix, return identity
        }

        det = 1.0 / det;
        for (int i = 0; i < 16; i++) {
            inv.m[i] *= det;
        }
        return inv;
    }

    /** @brief Print matrix for debugging */
    void print(const std::string& label = "") const {
        if (!label.empty()) std::cout << "[" << label << "]" << std::endl;
        for (int row = 0; row < 4; ++row) {
            std::cout << "  | ";
            for (int col = 0; col < 4; ++col) {
                std::cout << std::fixed << std::setprecision(6) << m[col * 4 + row] << " ";
            }
            std::cout << "|" << std::endl;
        }
    }
};

// ============================================================================
// Bounding Box
// ============================================================================

/**
 * @brief Axis-aligned bounding box
 */
struct BoundingBox {
    Vec3 min;
    Vec3 max;

    BoundingBox() : min(1e30, 1e30, 1e30), max(-1e30, -1e30, -1e30) {}
    BoundingBox(const Vec3& mn, const Vec3& mx) : min(mn), max(mx) {}

    Vec3 size() const { return max - min; }
    Vec3 center() const { return (min + max) * 0.5; }
    double volume() const {
        Vec3 s = size();
        return s.x * s.y * s.z;
    }

    bool isValid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }

    void expand(const Vec3& point) {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }
};

// ============================================================================
// CNC Coordinate System (3-Tier Hierarchy)
// ============================================================================

/**
 * @brief CNC-style coordinate system with MCS → WCS (G54-G59) → TCS hierarchy
 *
 * Mimics a real CNC controller's coordinate management:
 * - MCS: Machine home (absolute 0,0,0)
 * - WCS: Work offset (datum point on the workpiece) — 6 registers (G54-G59)
 * - TCS: Tool offset (from spindle gauge line to tool tip)
 *
 * The compound transform for the tool tip in world space is:
 *   T_world = T_mcs * T_wcs[active] * T_tcs
 */
class CNCCoordinateSystem {
public:
    static constexpr int NUM_WCS = 6;  // G54 through G59

    CNCCoordinateSystem() {
        m_mcsToWorld = Mat4::identity();
        m_tcsOffset = Mat4::identity();
        for (int i = 0; i < NUM_WCS; ++i) {
            m_wcsOffsets[i] = Mat4::identity();
        }
        m_activeWCS = 0; // G54
    }

    // --- WCS Management (G54-G59) ---

    /**
     * @brief Set work coordinate offset
     * @param index 0=G54, 1=G55, ..., 5=G59
     * @param offset Translation offset in meters
     */
    void setWCS(int index, const Vec3& offset) {
        if (index >= 0 && index < NUM_WCS) {
            m_wcsOffsets[index] = Mat4::translation(offset);
            std::cout << "[CoordSys] Set G" << (54 + index)
                      << " = (" << offset.x * 1000 << ", " << offset.y * 1000
                      << ", " << offset.z * 1000 << ") mm" << std::endl;
        }
    }

    /**
     * @brief Set work coordinate offset with rotation
     * @param index WCS index (0-5)
     * @param offset Translation in meters
     * @param rotation Rotation matrix (e.g., for rotated fixtures)
     */
    void setWCS(int index, const Vec3& offset, const Mat4& rotation) {
        if (index >= 0 && index < NUM_WCS) {
            m_wcsOffsets[index] = Mat4::translation(offset) * rotation;
        }
    }

    /** @brief Switch active WCS (called when G54-G59 is parsed from G-Code) */
    void setActiveWCS(int index) {
        if (index >= 0 && index < NUM_WCS) {
            m_activeWCS = index;
            std::cout << "[CoordSys] Active WCS: G" << (54 + index) << std::endl;
        }
    }

    int getActiveWCS() const { return m_activeWCS; }

    // --- Tool Coordinate System ---

    /**
     * @brief Set tool length offset
     * @param tlo Tool length in meters (spindle face to tip)
     * @param axis Axis along which the tool extends (0=X, 1=Y, 2=Z)
     */
    void setToolLengthOffset(double tlo, int axis = 2) {
        Vec3 offset;
        switch (axis) {
            case 0: offset = Vec3(tlo, 0, 0); break;
            case 1: offset = Vec3(0, tlo, 0); break;
            default: offset = Vec3(0, 0, tlo); break;
        }
        m_tcsOffset = Mat4::translation(offset);
        std::cout << "[CoordSys] TLO = " << tlo * 1000 << " mm (axis "
                  << axis << ")" << std::endl;
    }

    /** @brief Set MCS transform (usually identity unless machine has a base offset) */
    void setMCS(const Mat4& mcs) { m_mcsToWorld = mcs; }

    // --- Compound Transform Queries ---

    /** @brief Get the full tool-tip-to-world transform */
    Mat4 getToolToWorld() const {
        return m_mcsToWorld * m_wcsOffsets[m_activeWCS] * m_tcsOffset;
    }

    /** @brief Get the workpiece-origin-to-world transform */
    Mat4 getWorkpieceToWorld() const {
        return m_mcsToWorld * m_wcsOffsets[m_activeWCS];
    }

    /** @brief Get the active WCS offset matrix */
    const Mat4& getActiveWCSMatrix() const {
        return m_wcsOffsets[m_activeWCS];
    }

    /** @brief Get the TCS offset matrix */
    const Mat4& getTCSMatrix() const { return m_tcsOffset; }

    // --- Mesh and Particle Transforms ---

    /**
     * @brief Align a tool mesh so its cutting tip is at the origin,
     *        then apply TCS and WCS transforms.
     * @param mesh Tool mesh to align (modified in-place)
     * @param alignAxis 0=X, 1=Y, 2=Z
     */
    void alignToolFromCAD(Mesh& mesh, int alignAxis = 2) {
        if (mesh.nodes.empty()) return;
        if (alignAxis < 0 || alignAxis > 2) alignAxis = 2;

        // Step 1: Find tip (min along align axis) and centroid
        double minAlignCoord = 1e30;
        double sumPerp1 = 0, sumPerp2 = 0;

        for (const auto& node : mesh.nodes) {
            double coord;
            switch (alignAxis) {
                case 0: coord = node.position.x; break;
                case 1: coord = node.position.y; break;
                default: coord = node.position.z; break;
            }
            if (coord < minAlignCoord) minAlignCoord = coord;

            // Sum perpendicular axes for centering
            if (alignAxis == 0) { sumPerp1 += node.position.y; sumPerp2 += node.position.z; }
            else if (alignAxis == 1) { sumPerp1 += node.position.x; sumPerp2 += node.position.z; }
            else { sumPerp1 += node.position.x; sumPerp2 += node.position.y; }
        }

        double n = static_cast<double>(mesh.nodes.size());
        double centPerp1 = sumPerp1 / n;
        double centPerp2 = sumPerp2 / n;

        // Step 2: Build alignment transform (tip → origin, centered)
        Mat4 alignT;
        switch (alignAxis) {
            case 0: alignT = Mat4::translation(-minAlignCoord, -centPerp1, -centPerp2); break;
            case 1: alignT = Mat4::translation(-centPerp1, -minAlignCoord, -centPerp2); break;
            default: alignT = Mat4::translation(-centPerp1, -centPerp2, -minAlignCoord); break;
        }

        // Step 3: Apply alignment, then WCS+TCS
        Mat4 fullTransform = getToolToWorld() * alignT;

        transformMesh(mesh, fullTransform);

        std::cout << "[CoordSys] Aligned tool (axis=" << alignAxis
                  << ") through compound MCS*WCS*TCS transform" << std::endl;
    }

    /**
     * @brief Apply a 4×4 transform to all mesh nodes
     */
    static void transformMesh(Mesh& mesh, const Mat4& transform) {
        for (auto& node : mesh.nodes) {
            node.position = transform.transformPoint(node.position);
            node.originalPosition = transform.transformPoint(node.originalPosition);
        }
        // Re-compute triangle normals
        for (auto& tri : mesh.triangles) {
            if (tri.indices[0] >= 0 && tri.indices[0] < static_cast<int>(mesh.nodes.size()) &&
                tri.indices[1] >= 0 && tri.indices[1] < static_cast<int>(mesh.nodes.size()) &&
                tri.indices[2] >= 0 && tri.indices[2] < static_cast<int>(mesh.nodes.size())) {
                const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
                const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
                const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
                tri.normal = (v1 - v0).cross(v2 - v0).normalized();
            }
        }
    }

    /**
     * @brief Apply a 4×4 transform to all SPH particles
     */
    static void transformParticles(std::vector<SPHParticle>& particles, const Mat4& transform) {
        for (auto& p : particles) {
            Vec3 pos = transform.transformPoint(Vec3(p.x, p.y, p.z));
            p.x = pos.x;
            p.y = pos.y;
            p.z = pos.z;

            // Transform velocity direction too
            Vec3 vel = transform.transformDirection(Vec3(p.vx, p.vy, p.vz));
            p.vx = vel.x;
            p.vy = vel.y;
            p.vz = vel.z;
        }
    }

    /**
     * @brief Compute bounding box of a mesh
     */
    static BoundingBox computeBoundingBox(const Mesh& mesh) {
        BoundingBox bb;
        for (const auto& node : mesh.nodes) {
            bb.expand(node.position);
        }
        return bb;
    }

    /**
     * @brief Compute bounding box of SPH particles
     */
    static BoundingBox computeBoundingBox(const std::vector<SPHParticle>& particles) {
        BoundingBox bb;
        for (const auto& p : particles) {
            if (p.status != ParticleStatus::INACTIVE) {
                bb.expand(Vec3(p.x, p.y, p.z));
            }
        }
        return bb;
    }

private:
    Mat4 m_mcsToWorld;              // Machine Coordinate System → world
    Mat4 m_wcsOffsets[NUM_WCS];     // G54(0) through G59(5)
    int  m_activeWCS;               // Currently active WCS index
    Mat4 m_tcsOffset;               // Tool Coordinate System (TLO)
};

} // namespace edgepredict
