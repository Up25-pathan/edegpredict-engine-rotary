#include "ToolRegionClassifier.h"
#include "CoordinateSystem.h"
#include "json.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <queue>
#include <map>
#include <set>
#include <stack>

namespace edgepredict {

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / PI;

double clamp01(double v) {
    return std::max(0.0, std::min(1.0, v));
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool containsAny(const std::string& haystack, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (haystack.find(needle) != std::string::npos) return true;
    }
    return false;
}

double getJsonDouble(const Config& config,
                     const char* section,
                     const char* key,
                     double fallback) {
    const auto& j = config.getJson();
    if (!j.contains(section) || !j[section].contains(key)) return fallback;
    return j[section].value(key, fallback);
}

bool isEdgeRegion(ToolRegion region) {
    return region == ToolRegion::CUTTING_EDGE ||
           region == ToolRegion::CHISEL_EDGE ||
           region == ToolRegion::END_CUTTING_EDGE ||
           region == ToolRegion::PERIPHERAL_CUTTING_EDGE ||
           region == ToolRegion::CORNER_RADIUS ||
           region == ToolRegion::LEAD_CHAMFER ||
           region == ToolRegion::THREAD_CREST ||
           region == ToolRegion::CHAMFER_LEAD ||
           region == ToolRegion::BURR_TOOTH;
}

bool isRakeRegion(ToolRegion region) {
    return region == ToolRegion::RAKE_FACE ||
           region == ToolRegion::INSERT_RAKE_FACE ||
           region == ToolRegion::CHIP_GULLET;
}

bool isFlankRegion(ToolRegion region) {
    return region == ToolRegion::FLANK_FACE ||
           region == ToolRegion::MARGIN ||
           region == ToolRegion::LAND ||
           region == ToolRegion::RELIEF_FACE ||
           region == ToolRegion::THREAD_FLANK ||
           region == ToolRegion::INSERT_FLANK_FACE;
}

} // namespace

const char* ToolRegionClassifier::familyName(RotaryToolFamily family) {
    switch (family) {
        case RotaryToolFamily::DRILLING: return "drilling";
        case RotaryToolFamily::REAMING: return "reaming";
        case RotaryToolFamily::MILLING: return "milling";
        case RotaryToolFamily::THREADING: return "threading";
        case RotaryToolFamily::BORING: return "boring";
        case RotaryToolFamily::COUNTERSINK_COUNTERBORE: return "countersink_counterbore";
        case RotaryToolFamily::BURR_DEBURRING: return "burr_deburring";
        case RotaryToolFamily::UNKNOWN:
        default: return "unknown";
    }
}

const char* ToolRegionClassifier::regionName(ToolRegion region) {
    switch (region) {
        case ToolRegion::TOOL_BODY: return "tool_body";
        case ToolRegion::CUTTING_EDGE: return "cutting_edge";
        case ToolRegion::CHISEL_EDGE: return "chisel_edge";
        case ToolRegion::RAKE_FACE: return "rake_face";
        case ToolRegion::FLANK_FACE: return "flank_face";
        case ToolRegion::MARGIN: return "margin";
        case ToolRegion::FLUTE: return "flute";
        case ToolRegion::SHANK: return "shank";
        case ToolRegion::END_CUTTING_EDGE: return "end_cutting_edge";
        case ToolRegion::PERIPHERAL_CUTTING_EDGE: return "peripheral_cutting_edge";
        case ToolRegion::BALL_NOSE: return "ball_nose";
        case ToolRegion::CORNER_RADIUS: return "corner_radius";
        case ToolRegion::LEAD_CHAMFER: return "lead_chamfer";
        case ToolRegion::LAND: return "land";
        case ToolRegion::COUNTERSINK_FACE: return "countersink_face";
        case ToolRegion::COUNTERBORE_FACE: return "counterbore_face";
        case ToolRegion::THREAD_CREST: return "thread_crest";
        case ToolRegion::THREAD_FLANK: return "thread_flank";
        case ToolRegion::THREAD_ROOT: return "thread_root";
        case ToolRegion::RELIEF_FACE: return "relief_face";
        case ToolRegion::CHAMFER_LEAD: return "chamfer_lead";
        case ToolRegion::INSERT_RAKE_FACE: return "insert_rake_face";
        case ToolRegion::INSERT_FLANK_FACE: return "insert_flank_face";
        case ToolRegion::INSERT_SEAT: return "insert_seat";
        case ToolRegion::CHIP_GULLET: return "chip_gullet";
        case ToolRegion::BURR_TOOTH: return "burr_tooth";
        case ToolRegion::BURR_GULLET: return "burr_gullet";
        case ToolRegion::UNKNOWN:
        default: return "unknown";
    }
}

// ============================================================================
// Phase 1: PCA Auto-Alignment
// ============================================================================

ToolRegionClassifier::PCAResult ToolRegionClassifier::computePCA(
    const std::vector<FEMNode>& nodes) {

    PCAResult result;
    size_t n = nodes.size();
    if (n == 0) {
        result.transform = Mat4::identity();
        return result;
    }

    // Centroid
    Vec3 centroid;
    for (const auto& node : nodes) {
        centroid += node.position;
    }
    centroid = centroid * (1.0 / n);
    result.centroid = centroid;

    // Covariance matrix (3x3)
    double cov[3][3] = {{0}};
    for (const auto& node : nodes) {
        Vec3 d = node.position - centroid;
        cov[0][0] += d.x * d.x;
        cov[0][1] += d.x * d.y;
        cov[0][2] += d.x * d.z;
        cov[1][1] += d.y * d.y;
        cov[1][2] += d.y * d.z;
        cov[2][2] += d.z * d.z;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    double invN = 1.0 / n;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cov[i][j] *= invN;

    // Jacobi eigenvalue decomposition for 3x3 symmetric matrix
    double A[3][3];
    double V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A[i][j] = cov[i][j];

    // Jacobi iteration
    const int MAX_ITER = 50;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        double maxOff = std::abs(A[0][1]);
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (std::abs(A[i][j]) > maxOff) {
                    maxOff = std::abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (maxOff < 1e-15) break;

        double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
        double t = (theta >= 0 ? 1.0 : -1.0) /
                   (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        // Apply rotation
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        A[p][p] = c * c * app + s * s * aqq - 2.0 * s * c * apq;
        A[q][q] = s * s * app + c * c * aqq + 2.0 * s * c * apq;
        A[p][q] = A[q][p] = (c * c - s * s) * apq + s * c * (app - aqq);

        for (int r = 0; r < 3; ++r) {
            if (r != p && r != q) {
                double arp = A[r][p], arq = A[r][q];
                A[r][p] = A[p][r] = c * arp - s * arq;
                A[r][q] = A[q][r] = s * arp + c * arq;
            }
        }

        // Update eigenvectors
        for (int r = 0; r < 3; ++r) {
            double vrp = V[r][p], vrq = V[r][q];
            V[r][p] = c * vrp - s * vrq;
            V[r][q] = s * vrp + c * vrq;
        }
    }

    result.eigenvalues[0] = A[0][0];
    result.eigenvalues[1] = A[1][1];
    result.eigenvalues[2] = A[2][2];

    // Sort eigenvalues descending, reorder eigenvectors
    int order[3] = {0, 1, 2};
    if (A[0][0] < A[1][1]) std::swap(order[0], order[1]);
    if (A[order[0]][order[0]] < A[order[2]][order[2]]) std::swap(order[0], order[2]);
    if (A[order[1]][order[1]] < A[order[2]][order[2]]) std::swap(order[1], order[2]);

    result.primaryAxis = Vec3(V[0][order[0]], V[1][order[0]], V[2][order[0]]).normalized();
    result.secondaryAxis = Vec3(V[0][order[1]], V[1][order[1]], V[2][order[1]]).normalized();
    result.tertiaryAxis = result.primaryAxis.cross(result.secondaryAxis).normalized();
    result.secondaryAxis = result.tertiaryAxis.cross(result.primaryAxis).normalized();

    // Build alignment transform: primaryAxis -> Z, tip (min along Z after rotation) -> origin
    // First: rotate so primary axis aligns to Z
    Vec3 zAxis(0, 0, 1);
    double cosAngle = result.primaryAxis.dot(zAxis);
    double angle = std::acos(clamp01(cosAngle)); // clamp to [-1,1] for numerical safety
    Vec3 rotAxis = result.primaryAxis.cross(zAxis);
    Mat4 rotMat;
    if (rotAxis.lengthSq() > 1e-20) {
        rotMat = Mat4::rotationAxis(rotAxis, angle);
    } else if (cosAngle < 0) {
        // 180 degree flip — rotate around X
        rotMat = Mat4::rotationX(PI);
    }

    // Find tip after rotation
    Vec3 tip(1e30, 1e30, 1e30);
    for (const auto& node : nodes) {
        Vec3 p = rotMat.transformPoint(node.position);
        if (p.z < tip.z) {
            tip = p;
        }
    }

    // Translate tip to origin
    result.transform = Mat4::translation(-tip) * rotMat;

    return result;
}

// ============================================================================
// Phase 2: Normal Orientation
// ============================================================================

ToolRegionClassifier::MeshTopology ToolRegionClassifier::buildTopology(
    const Mesh& mesh) {

    MeshTopology topo;
    int V = static_cast<int>(mesh.nodes.size());
    int F = static_cast<int>(mesh.triangles.size());

    topo.vertexNormals.resize(V, Vec3::zero());
    topo.vertexFaces.resize(V);
    topo.vertexVertices.resize(V);

    // Build face adjacency per edge for half-edge structure
    struct EdgeKey {
        int a, b;
        bool operator==(const EdgeKey& o) const {
            return a == o.a && b == o.b;
        }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const {
            return static_cast<size_t>(k.a) ^ (static_cast<size_t>(k.b) << 16);
        }
    };

    std::unordered_map<EdgeKey, int, EdgeKeyHash> edgeToFace;

    for (int f = 0; f < F; ++f) {
        const auto& tri = mesh.triangles[f];
        int i0 = tri.indices[0];
        int i1 = tri.indices[1];
        int i2 = tri.indices[2];

        if (i0 < 0 || i1 < 0 || i2 < 0) continue;
        if (i0 >= V || i1 >= V || i2 >= V) continue;

        // Accumulate area-weighted normals
        const Vec3& v0 = mesh.nodes[i0].position;
        const Vec3& v1 = mesh.nodes[i1].position;
        const Vec3& v2 = mesh.nodes[i2].position;
        Vec3 faceNormal = (v1 - v0).cross(v2 - v0);
        topo.vertexNormals[i0] += faceNormal;
        topo.vertexNormals[i1] += faceNormal;
        topo.vertexNormals[i2] += faceNormal;

        // Vertex-face adjacency
        topo.vertexFaces[i0].push_back(f);
        topo.vertexFaces[i1].push_back(f);
        topo.vertexFaces[i2].push_back(f);

        // Vertex-vertex adjacency (edge neighbors)
        auto addEdge = [&](int a, int b) {
            // Deduplicate
            bool found = false;
            for (int v : topo.vertexVertices[a]) {
                if (v == b) { found = true; break; }
            }
            if (!found) topo.vertexVertices[a].push_back(b);
        };
        addEdge(i0, i1);
        addEdge(i1, i0);
        addEdge(i1, i2);
        addEdge(i2, i1);
        addEdge(i2, i0);
        addEdge(i0, i2);

        // Half-edge map
        std::array<std::pair<int,int>, 3> edges = {{
            {std::min(i0,i1), std::max(i0,i1)},
            {std::min(i1,i2), std::max(i1,i2)},
            {std::min(i2,i0), std::max(i2,i0)}
        }};
        for (const auto& e : edges) {
            EdgeKey key{e.first, e.second};
            auto it = edgeToFace.find(key);
            if (it == edgeToFace.end()) {
                edgeToFace[key] = f;
            } else {
                // Found the other side — record half-edge
                HalfEdge he;
                he.vertA = e.first;
                he.vertB = e.second;
                he.faceA = it->second;
                he.faceB = f;
                topo.halfEdges.push_back(he);
                edgeToFace.erase(it);
            }
        }
    }

    // Remaining boundary edges
    for (const auto& kv : edgeToFace) {
        HalfEdge he;
        he.vertA = kv.first.a;
        he.vertB = kv.first.b;
        he.faceA = kv.second;
        he.faceB = -1;
        topo.halfEdges.push_back(he);
    }

    // Normalize vertex normals
    for (auto& n : topo.vertexNormals) {
        double len = n.length();
        if (len > 1e-20) n = n * (1.0 / len);
    }

    return topo;
}

void ToolRegionClassifier::orientNormalsOutward(
    std::vector<Vec3>& normals,
    const std::vector<FEMNode>& nodes,
    const std::vector<Triangle>& triangles,
    const MeshTopology& topo) {

    int V = static_cast<int>(nodes.size());
    if (V == 0) return;

    // Orient each face so normal points radially outward from the Z-axis.
    // This is the correct convention for rotary cutting tools:
    // - Flank faces have normals with positive radial component
    // - Rake faces (flutes) also have normals with positive radial component
    //   (they point away from the tool center, toward the chip flow direction)
    // - The only exception is the core/web region where radial length ≈ 0
    normals.assign(V, Vec3::zero());
    for (size_t f = 0; f < triangles.size(); ++f) {
        const auto& tri = triangles[f];
        if (tri.indices[0] < 0 || tri.indices[0] >= V ||
            tri.indices[1] < 0 || tri.indices[1] >= V ||
            tri.indices[2] < 0 || tri.indices[2] >= V) continue;

        const Vec3& v0 = nodes[tri.indices[0]].position;
        const Vec3& v1 = nodes[tri.indices[1]].position;
        const Vec3& v2 = nodes[tri.indices[2]].position;

        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 cross = e1.cross(e2);
        double area = cross.length();

        Vec3 faceNormal = (area > 1e-20) ? cross * (1.0 / area) : Vec3(0, 0, 1);

        // Flip if normal points radially inward
        Vec3 center = (v0 + v1 + v2) * (1.0 / 3.0);
        double rLen = std::sqrt(center.x * center.x + center.y * center.y);
        if (rLen > 1e-12) {
            Vec3 radialDir(center.x / rLen, center.y / rLen, 0.0);
            if (faceNormal.dot(radialDir) < 0) {
                faceNormal = faceNormal * (-1.0);
            }
        }

        // Accumulate area-weighted normals
        normals[tri.indices[0]] += faceNormal * area;
        normals[tri.indices[1]] += faceNormal * area;
        normals[tri.indices[2]] += faceNormal * area;
    }
    for (auto& n : normals) {
        double len = n.length();
        if (len > 1e-20) n = n * (1.0 / len);
    }
}

// ============================================================================
// Phase 3: Dihedral Edge Detection + Curvature
// ============================================================================

double ToolRegionClassifier::computeDihedralAngle(
    const Vec3& nA, const Vec3& nB, const Vec3& edgeDir) {

    double cosAngle = clamp01(nA.dot(nB));
    double angle = std::acos(cosAngle);

    // Determine convex vs concave using edge direction
    // For a convex edge, the angle is > 0 and the face normals diverge
    // We use the sign of (nA × nB) · edgeDir
    Vec3 cross = nA.cross(nB);
    double sign = cross.dot(edgeDir);
    return (sign >= 0) ? angle : -angle;
}

ToolRegionClassifier::CurvatureData ToolRegionClassifier::computeCurvature(
    const Mesh& mesh,
    const MeshTopology& topo,
    const std::vector<Vec3>& normals) {

    int V = static_cast<int>(mesh.nodes.size());
    CurvatureData data;
    data.meanCurvature.assign(V, 0.0);
    data.gaussianCurvature.assign(V, 0.0);
    data.dihedralMax.assign(V, 0.0);
    data.isSharpEdge.assign(V, false);

    if (V == 0 || topo.halfEdges.empty()) return data;

    // Compute face normals for dihedral calculation
    std::vector<Vec3> faceNormals(mesh.triangles.size());
    for (size_t f = 0; f < mesh.triangles.size(); ++f) {
        const auto& tri = mesh.triangles[f];
        const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
        const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
        const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
        Vec3 cross = (v1 - v0).cross(v2 - v0);
        double len = cross.length();
        if (len > 1e-20) {
            faceNormals[f] = cross * (1.0 / len);
        }
    }

    // Orient face normals outward using radial direction (correct for rotary tools).
    // For a Z-aligned tool, "outward" means the normal has a positive radial component:
    // n · (cx, cy, 0) > 0 where (cx, cy) is the face center's radial vector from the Z-axis.
    // This correctly handles concave flute surfaces (normals point radially inward = negative
    // radial dot, so they get flipped to point outward), which centroid-based orientation
    // would get wrong.
    for (size_t f = 0; f < faceNormals.size(); ++f) {
        const auto& tri = mesh.triangles[f];
        if (tri.indices[0] < 0 || tri.indices[0] >= V) continue;
        if (tri.indices[1] < 0 || tri.indices[1] >= V) continue;
        if (tri.indices[2] < 0 || tri.indices[2] >= V) continue;
        const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
        const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
        const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
        Vec3 center = (v0 + v1 + v2) * (1.0 / 3.0);
        // Radial direction from Z-axis: (cx, cy, 0) normalized
        double rLen = std::sqrt(center.x * center.x + center.y * center.y);
        if (rLen > 1e-12) {
            Vec3 radialDir(center.x / rLen, center.y / rLen, 0.0);
            if (faceNormals[f].dot(radialDir) < 0) {
                faceNormals[f] = faceNormals[f] * (-1.0);
            }
        }
        // If on-axis (r ≈ 0), keep normal as-is (chisel edge / tip apex)
    }

    // Phase 3a: Dihedral angle at each half-edge
    // Track per-vertex max dihedral and per-vertex sharpness
    for (const auto& he : topo.halfEdges) {
        if (he.faceA < 0 || he.faceB < 0) continue;
        if (he.faceA >= (int)faceNormals.size() || he.faceB >= (int)faceNormals.size()) continue;

        Vec3 edgeDir = (mesh.nodes[he.vertB].position - mesh.nodes[he.vertA].position).normalized();
        double dihedral = computeDihedralAngle(faceNormals[he.faceA], faceNormals[he.faceB], edgeDir);

        double absDihedral = std::abs(dihedral);
        data.dihedralMax[he.vertA] = std::max(data.dihedralMax[he.vertA], absDihedral);
        data.dihedralMax[he.vertB] = std::max(data.dihedralMax[he.vertB], absDihedral);
    }

    // Phase 3b: Cotangent Laplacian mean curvature
    // H(v) = (1/2) * |sum_j (cot α_j + cot β_j) * (v - v_j)|
    // We compute per-vertex mean curvature normal
    for (int v = 0; v < V; ++v) {
        const Vec3& vi = mesh.nodes[v].position;
        Vec3 laplacianSum;

        for (int nbrIdx : topo.vertexVertices[v]) {
            const Vec3& vj = mesh.nodes[nbrIdx].position;

            // Find the two faces sharing edge (v, nbrIdx)
            int faceA = -1, faceB = -1;
            for (int f : topo.vertexFaces[v]) {
                const auto& tri = mesh.triangles[f];
                int i0 = tri.indices[0], i1 = tri.indices[1], i2 = tri.indices[2];
                bool hasV = (i0 == v || i1 == v || i2 == v);
                bool hasNbr = (i0 == nbrIdx || i1 == nbrIdx || i2 == nbrIdx);
                if (hasV && hasNbr) {
                    if (faceA < 0) faceA = f;
                    else faceB = f;
                }
            }

            if (faceA < 0) continue;

            // For each adjacent face, compute cotangent of angle opposite edge (v, vj)
            auto cotAngle = [&](int face, int viIdx, int vjIdx) -> double {
                const auto& tri = mesh.triangles[face];
                int idxs[3] = {tri.indices[0], tri.indices[1], tri.indices[2]};
                int k = -1;
                for (int t = 0; t < 3; ++t) {
                    if (idxs[t] != viIdx && idxs[t] != vjIdx) {
                        k = idxs[t];
                        break;
                    }
                }
                if (k < 0) return 0.0;

                const Vec3& vk = mesh.nodes[k].position;
                const Vec3& viPos = mesh.nodes[viIdx].position;
                const Vec3& vjPos = mesh.nodes[vjIdx].position;
                double ax = viPos.x - vk.x, ay = viPos.y - vk.y, az = viPos.z - vk.z;
                double bx = vjPos.x - vk.x, by = vjPos.y - vk.y, bz = vjPos.z - vk.z;
                double dot = ax * bx + ay * by + az * bz;
                double crossX = ay * bz - az * by;
                double crossY = az * bx - ax * bz;
                double crossZ = ax * by - ay * bx;
                double crossLen = std::sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
                if (crossLen < 1e-20) return 0.0;
                return dot / crossLen;
            };

            double cotSum = cotAngle(faceA, v, nbrIdx);
            if (faceB >= 0) cotSum += cotAngle(faceB, v, nbrIdx);

            Vec3 diff(vi.x - vj.x, vi.y - vj.y, vi.z - vj.z);
            laplacianSum = laplacianSum + diff * cotSum;
        }

        data.meanCurvature[v] = 0.5 * laplacianSum.length();
    }

    // Phase 3c: Gaussian curvature via angle defect
    // K(v) = (2π - sum(θ_j)) / (1/3 * sum(area of incident triangles))
    for (int v = 0; v < V; ++v) {
        const Vec3& vi = mesh.nodes[v].position;
        double angleSum = 0.0;
        double areaSum = 0.0;

        for (int f : topo.vertexFaces[v]) {
            const auto& tri = mesh.triangles[f];
            int i0 = tri.indices[0], i1 = tri.indices[1], i2 = tri.indices[2];

            const Vec3* vv[3] = {&mesh.nodes[i0].position,
                                 &mesh.nodes[i1].position,
                                 &mesh.nodes[i2].position};

            // Find which index corresponds to v
            int localIdx = -1;
            if (i0 == v) localIdx = 0;
            else if (i1 == v) localIdx = 1;
            else if (i2 == v) localIdx = 2;

            if (localIdx < 0) continue;

            // Angle at v in this triangle
            const Vec3& a = *vv[(localIdx + 1) % 3];
            const Vec3& b = *vv[(localIdx + 2) % 3];
            Vec3 e1 = a - vi;
            Vec3 e2 = b - vi;
            double e1len = e1.length();
            double e2len = e2.length();
            if (e1len < 1e-20 || e2len < 1e-20) continue;

            double cosAngle = clamp01(e1.dot(e2) / (e1len * e2len));
            angleSum += std::acos(cosAngle);

            // Area contribution (1/3 of triangle area per vertex)
            areaSum += e1.cross(e2).length() * (1.0 / 6.0);
        }

        double defect = 2.0 * PI - angleSum;
        data.gaussianCurvature[v] = (areaSum > 1e-20) ? defect / areaSum : 0.0;
    }

    // Phase 3d: Sharp edge classification
    // A vertex is a "sharp edge" if its max dihedral angle exceeds threshold
    // Automatic threshold detection: compute mean + std of dihedralMax
    double meanDihedral = 0.0;
    int countNonZero = 0;
    for (int v = 0; v < V; ++v) {
        if (data.dihedralMax[v] > 1e-6) {
            meanDihedral += data.dihedralMax[v];
            countNonZero++;
        }
    }
    if (countNonZero > 0) meanDihedral /= countNonZero;

    double varDihedral = 0.0;
    for (int v = 0; v < V; ++v) {
        if (data.dihedralMax[v] > 1e-6) {
            double d = data.dihedralMax[v] - meanDihedral;
            varDihedral += d * d;
        }
    }
    if (countNonZero > 0) varDihedral /= countNonZero;
    double stdDihedral = std::sqrt(varDihedral);

    // Threshold: mean + 1.0 * std (catches outliers = sharp edges)
    double sharpThreshold = std::max(meanDihedral + stdDihedral, 20.0 * DEG_TO_RAD);

    for (int v = 0; v < V; ++v) {
        data.isSharpEdge[v] = (data.dihedralMax[v] > sharpThreshold);
    }

    return data;
}

// ============================================================================
// Phase 4: Kinematic Envelope
// ============================================================================

ToolRegionClassifier::EnvelopeData ToolRegionClassifier::detectEnvelope(
    const Mesh& mesh, double zBinSize) {

    int V = static_cast<int>(mesh.nodes.size());
    EnvelopeData data;
    data.onEnvelope.assign(V, false);
    data.maxRadius = 0.0;
    data.zBinSize = zBinSize;

    if (V == 0) return data;

    // Find Z range
    double minZ = 1e30, maxZ = -1e30;
    for (const auto& node : mesh.nodes) {
        minZ = std::min(minZ, node.position.z);
        maxZ = std::max(maxZ, node.position.z);
    }

    double span = maxZ - minZ;
    if (span < 1e-12) {
        for (auto& v : data.onEnvelope) v = true;
        return data;
    }

    zBinSize = fmax(zBinSize, 1e-12);
    int numBins = std::max(1, static_cast<int>(std::ceil(span / zBinSize)));
    data.zSliceMaxRadius.assign(numBins, 0.0);

    // Bin mapping
    auto getBin = [&](double z) -> int {
        int bin = static_cast<int>((z - minZ) / zBinSize);
        return std::max(0, std::min(numBins - 1, bin));
    };

    // First pass: find max radius per Z-slice
    for (const auto& node : mesh.nodes) {
        double r = std::sqrt(node.position.x * node.position.x +
                             node.position.y * node.position.y);
        int bin = getBin(node.position.z);
        if (r > data.zSliceMaxRadius[bin]) {
            data.zSliceMaxRadius[bin] = r;
        }
        if (r > data.maxRadius) data.maxRadius = r;
    }

    // Second pass: mark envelope nodes within tolerance of max radius in their slice
    // Tolerance scales with tool size: 5μm or 0.5% of radius, whichever is larger
    double tolerance = std::max(5e-6, data.maxRadius * 0.005);

    for (int i = 0; i < V; ++i) {
        const auto& node = mesh.nodes[i];
        double r = std::sqrt(node.position.x * node.position.x +
                             node.position.y * node.position.y);
        int bin = getBin(node.position.z);
        double sliceMax = data.zSliceMaxRadius[bin];
        if (sliceMax > 0 && r >= sliceMax - tolerance) {
            data.onEnvelope[i] = true;
        }
    }

    return data;
}

// ============================================================================
// Phase 5: Seed-Based Region Growing
// ============================================================================

ToolRegionClassifier::NormalClass ToolRegionClassifier::classifyNormalVsRotation(
    const Vec3& normal, const Vec3& position, double rotationDirection) {

    // For a tool rotating around Z-axis (positive = CCW when viewed from +Z):
    // Tangential velocity at position (x,y,z) is v = ω × r = ω * (-y, x, 0)
    // For CCW rotation (ω > 0): v_tang = (-y, x, 0)
    Vec3 tangent(-position.y, position.x, 0.0);
    double tLen = tangent.length();
    if (tLen < 1e-20) return NormalClass::NEUTRAL;
    tangent = tangent * (rotationDirection / tLen);

    double alignment = normal.dot(tangent);

    // Rake face: normal opposes tangential velocity (chip flows up the rake)
    // alignment < 0 means normal generally opposes tangent
    // Flank face: normal aligns with or trails tangential velocity
    // alignment > 0 means normal generally aligns with tangent
    // For clearance/relief: flank normal has a component pointing "away" from cut
    // which, due to the helix, appears as a positive dot with tangent

    constexpr double RAKE_THRESHOLD = -0.15;
    constexpr double FLANK_THRESHOLD = 0.15;

    if (alignment < RAKE_THRESHOLD) return NormalClass::RAKE_LIKE;
    if (alignment > FLANK_THRESHOLD) return NormalClass::FLANK_LIKE;
    return NormalClass::NEUTRAL;
}

std::vector<int> ToolRegionClassifier::findCuttingEdgeSeeds(
    const std::vector<FEMNode>& nodes,
    const MeshTopology& topo,
    const CurvatureData& curv,
    const EnvelopeData& env,
    const std::vector<Vec3>& normals,
    RotaryToolFamily family) {

    std::vector<int> seeds;
    int V = static_cast<int>(nodes.size());

    // Seed criteria (combined):
    // 1. Sharp edge (high dihedral angle)
    // 2. On kinematic envelope (high radius for that Z-slice)
    // 3. High mean curvature
    // 4. Normal transitions between rake-like and flank-like across the edge

    // Compute edge-based transition: for each half-edge where both vertices
    // are on envelope and at least one is sharp, check if normals on either
    // side of the edge belong to different classes
    for (const auto& he : topo.halfEdges) {
        if (he.vertA < 0 || he.vertB >= V) continue;
        if (he.vertB < 0 || he.vertB >= V) continue;

        bool aOnEnv = env.onEnvelope[he.vertA];
        bool bOnEnv = env.onEnvelope[he.vertB];
        bool aSharp = curv.isSharpEdge[he.vertA];
        bool bSharp = curv.isSharpEdge[he.vertB];

        if (!aOnEnv && !bOnEnv) continue;
        if (!aSharp && !bSharp) continue;

        seeds.push_back(he.vertA);
        seeds.push_back(he.vertB);
    }

    // Deduplicate
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

    // If no seeds found via dihedral, fall back to high-curvature envelope vertices
    if (seeds.empty()) {
        // Compute curvature statistics
        double meanCurv = 0.0;
        int countCurv = 0;
        for (int v = 0; v < V; ++v) {
            if (curv.meanCurvature[v] > 1e-8) {
                meanCurv += curv.meanCurvature[v];
                countCurv++;
            }
        }
        if (countCurv > 0) meanCurv /= countCurv;

        for (int v = 0; v < V; ++v) {
            if (env.onEnvelope[v] &&
                curv.meanCurvature[v] > meanCurv * 1.5) {
                seeds.push_back(v);
            }
        }
    }

    return seeds;
}

ToolRegionClassifier::ClassificationGraph ToolRegionClassifier::growRegions(
    const Mesh& mesh,
    const MeshTopology& topo,
    const CurvatureData& curv,
    const EnvelopeData& env,
    const std::vector<Vec3>& normals,
    RotaryToolFamily family) {

    int V = static_cast<int>(mesh.nodes.size());
    ClassificationGraph graph;
    graph.nodeCount = V;
    graph.assigned.assign(V, ToolRegion::UNKNOWN);
    graph.assignmentConfidence.assign(V, 0.0);

    if (V == 0) return graph;

    // 1. Find seeds (cutting edges)
    std::vector<int> seeds = findCuttingEdgeSeeds(mesh.nodes, topo, curv, env, normals, family);

    // Mark seeds as CUTTING_EDGE placeholder
    for (int v : seeds) {
        graph.assigned[v] = ToolRegion::CUTTING_EDGE;
        graph.assignmentConfidence[v] = 0.8;
    }

    // 2. BFS region growing from seeds
    // For each seed, classify neighbors based on normal vs tangential velocity
    struct QueueItem {
        int vertex;
        ToolRegion parentRegion;
        double parentConfidence;
    };

    std::queue<QueueItem> queue;
    std::vector<bool> visited(V, false);

    // Push all seeds
    for (int v : seeds) {
        queue.push({v, ToolRegion::CUTTING_EDGE, 0.8});
        visited[v] = true;
    }

    // BFS with propagation
    while (!queue.empty()) {
        auto item = queue.front();
        queue.pop();

        int v = item.vertex;
        const Vec3& pos = mesh.nodes[v].position;

        for (int nbr : topo.vertexVertices[v]) {
            if (visited[nbr]) continue;
            if (nbr < 0 || nbr >= V) continue;

            // Classify neighbor based on its normal relative to rotation
            NormalClass nc = classifyNormalVsRotation(normals[nbr], pos);

            ToolRegion assignedRegion;
            double confidence;

            // On-envelope vertices that are not sharp are typically margin/land
            if (env.onEnvelope[nbr] && !curv.isSharpEdge[nbr]) {
                assignedRegion = ToolRegion::MARGIN;
                confidence = 0.6;
            } else if (nc == NormalClass::RAKE_LIKE) {
                assignedRegion = ToolRegion::RAKE_FACE;
                confidence = 0.65;
            } else if (nc == NormalClass::FLANK_LIKE) {
                assignedRegion = ToolRegion::FLANK_FACE;
                confidence = 0.65;
            } else if (curv.meanCurvature[nbr] > 0.01) {
                // High curvature but neutral = corner radius
                assignedRegion = ToolRegion::CORNER_RADIUS;
                confidence = 0.5;
            } else if (env.onEnvelope[nbr]) {
                assignedRegion = ToolRegion::MARGIN;
                confidence = 0.5;
            } else {
                // Interior: check if concave (flute) or convex (tool body)
                if (curv.gaussianCurvature[nbr] < -1.0) {
                    assignedRegion = ToolRegion::FLUTE;
                    confidence = 0.55;
                } else {
                    assignedRegion = ToolRegion::TOOL_BODY;
                    confidence = 0.4;
                }
            }

            graph.assigned[nbr] = assignedRegion;
            graph.assignmentConfidence[nbr] = confidence;
            visited[nbr] = true;

            queue.push({nbr, assignedRegion, confidence});
        }
    }

    // 3. Assign remaining unvisited vertices
    for (int v = 0; v < V; ++v) {
        if (!visited[v]) {
            if (curv.meanCurvature[v] > 0.01) {
                graph.assigned[v] = ToolRegion::CORNER_RADIUS;
                graph.assignmentConfidence[v] = 0.3;
            } else if (curv.gaussianCurvature[v] < -1.0) {
                graph.assigned[v] = ToolRegion::FLUTE;
                graph.assignmentConfidence[v] = 0.35;
            } else {
                graph.assigned[v] = ToolRegion::TOOL_BODY;
                graph.assignmentConfidence[v] = 0.3;
            }
        }
    }

    return graph;
}

// ============================================================================
// Phase 6: Family-Specific Mapping
// ============================================================================

void ToolRegionClassifier::assignFamilyRegions(
    ClassificationGraph& graph,
    const Mesh& mesh,
    const MeshTopology& topo,
    const CurvatureData& curv,
    const EnvelopeData& env,
    const std::vector<Vec3>& normals,
    RotaryToolFamily family,
    const Config& config) {

    int V = static_cast<int>(mesh.nodes.size());

    // Map generic topology to family-specific region IDs
    // We work with the graph.assigned vector which currently has generic labels
    // (CUTTING_EDGE, RAKE_FACE, FLANK_FACE, MARGIN, FLUTE, TOOL_BODY, etc.)

    double maxRadius = env.maxRadius;
    double span = 0.0;
    double minZ = 1e30, maxZ = -1e30;
    for (const auto& node : mesh.nodes) {
        minZ = std::min(minZ, node.position.z);
        maxZ = std::max(maxZ, node.position.z);
    }
    span = maxZ - minZ;

    // Config-driven parameters
    double shankStartRel = getJsonDouble(config, "machining_parameters",
                                         "shank_start_rel", 0.82);
    double pointAngleDeg = getJsonDouble(config, "machining_parameters",
                                         "point_angle_deg", 118.0);
    double chiselRatio = getJsonDouble(config, "machining_parameters",
                                       "chisel_edge_ratio", 0.18);
    double webRadius = maxRadius * chiselRatio;
    bool ballNose = containsAny(getDescriptorString(config),
                                {"ball nose", "ballnose", "spherical"});
    bool indexable = containsAny(getDescriptorString(config),
                                 {"indexable", "insert"});

    const double pointHalfAngleRad = 0.5 * pointAngleDeg * DEG_TO_RAD;
    const double pointHeight = maxRadius / std::max(std::tan(pointHalfAngleRad), 1e-6);
    const double pointZoneHeight = std::min(pointHeight * 1.5, span * 0.4);

    for (int v = 0; v < V; ++v) {
        ToolRegion generic = graph.assigned[v];
        const Vec3& pos = mesh.nodes[v].position;
        double r = std::sqrt(pos.x * pos.x + pos.y * pos.y);
        double zRel = (span > 1e-12) ? (pos.z - minZ) / span : 0.0;
        double zFromTip = pos.z - minZ;
        double rRel = (maxRadius > 1e-12) ? r / maxRadius : 0.0;

        // Shank detection (universal)
        if (zRel > shankStartRel) {
            graph.assigned[v] = ToolRegion::SHANK;
            graph.assignmentConfidence[v] = 0.9;
            continue;
        }

        switch (family) {
            case RotaryToolFamily::DRILLING: {
                // Drill-specific mapping
                if (zFromTip <= pointZoneHeight && r <= webRadius) {
                    graph.assigned[v] = ToolRegion::CHISEL_EDGE;
                    graph.assignmentConfidence[v] = 0.8;
                } else if (generic == ToolRegion::CUTTING_EDGE || generic == ToolRegion::MARGIN) {
                    if (zFromTip <= pointZoneHeight) {
                        graph.assigned[v] = ToolRegion::CUTTING_EDGE;
                        graph.assignmentConfidence[v] = 0.85;
                    } else {
                        graph.assigned[v] = ToolRegion::MARGIN;
                        graph.assignmentConfidence[v] = 0.7;
                    }
                } else if (generic == ToolRegion::RAKE_FACE) {
                    graph.assigned[v] = ToolRegion::RAKE_FACE;
                    graph.assignmentConfidence[v] = 0.75;
                } else if (generic == ToolRegion::FLANK_FACE) {
                    graph.assigned[v] = ToolRegion::FLANK_FACE;
                    graph.assignmentConfidence[v] = 0.75;
                } else if (generic == ToolRegion::FLUTE || generic == ToolRegion::TOOL_BODY) {
                    if (zFromTip <= pointZoneHeight) {
                        // Within point region — already handled above
                    }
                    graph.assigned[v] = ToolRegion::FLUTE;
                    graph.assignmentConfidence[v] = 0.6;
                }
                break;
            }

            case RotaryToolFamily::REAMING: {
                if (zFromTip <= getJsonDouble(config, "machining_parameters",
                                              "lead_length_mm", 1.0) / 1000.0) {
                    if (rRel > 0.7) {
                        graph.assigned[v] = ToolRegion::LEAD_CHAMFER;
                        graph.assignmentConfidence[v] = 0.8;
                    } else {
                        graph.assigned[v] = ToolRegion::FLUTE;
                        graph.assignmentConfidence[v] = 0.6;
                    }
                } else if (generic == ToolRegion::MARGIN || rRel > 0.92) {
                    graph.assigned[v] = ToolRegion::MARGIN;
                    graph.assignmentConfidence[v] = 0.75;
                } else if (rRel > 0.82) {
                    graph.assigned[v] = ToolRegion::LAND;
                    graph.assignmentConfidence[v] = 0.7;
                } else if (generic == ToolRegion::RAKE_FACE) {
                    graph.assigned[v] = ToolRegion::RAKE_FACE;
                    graph.assignmentConfidence[v] = 0.65;
                } else if (generic == ToolRegion::FLANK_FACE) {
                    graph.assigned[v] = ToolRegion::FLANK_FACE;
                    graph.assignmentConfidence[v] = 0.65;
                } else {
                    graph.assigned[v] = ToolRegion::FLUTE;
                    graph.assignmentConfidence[v] = 0.55;
                }
                break;
            }

            case RotaryToolFamily::MILLING: {
                double endZoneHeight = std::min(maxRadius * 0.60, span * 0.14);
                if (indexable && rRel > 0.72 && zRel < 0.55) {
                    if (generic == ToolRegion::RAKE_FACE) {
                        graph.assigned[v] = ToolRegion::INSERT_RAKE_FACE;
                    } else if (generic == ToolRegion::FLANK_FACE) {
                        graph.assigned[v] = ToolRegion::INSERT_FLANK_FACE;
                    } else {
                        graph.assigned[v] = ToolRegion::PERIPHERAL_CUTTING_EDGE;
                    }
                    graph.assignmentConfidence[v] = 0.8;
                } else if (zFromTip <= endZoneHeight) {
                    if (ballNose && rRel < 0.62) {
                        graph.assigned[v] = ToolRegion::BALL_NOSE;
                        graph.assignmentConfidence[v] = 0.8;
                    } else if (rRel > 0.82 || generic == ToolRegion::CUTTING_EDGE) {
                        graph.assigned[v] = ToolRegion::END_CUTTING_EDGE;
                        graph.assignmentConfidence[v] = 0.8;
                    } else if (rRel > 0.7) {
                        graph.assigned[v] = ToolRegion::CORNER_RADIUS;
                        graph.assignmentConfidence[v] = 0.65;
                    } else {
                        graph.assigned[v] = ToolRegion::END_CUTTING_EDGE;
                        graph.assignmentConfidence[v] = 0.6;
                    }
                } else if (rRel > 0.88 || generic == ToolRegion::MARGIN) {
                    graph.assigned[v] = ToolRegion::PERIPHERAL_CUTTING_EDGE;
                    graph.assignmentConfidence[v] = 0.75;
                } else if (generic == ToolRegion::RAKE_FACE) {
                    graph.assigned[v] = ToolRegion::RAKE_FACE;
                    graph.assignmentConfidence[v] = 0.7;
                } else if (generic == ToolRegion::FLANK_FACE) {
                    graph.assigned[v] = ToolRegion::FLANK_FACE;
                    graph.assignmentConfidence[v] = 0.7;
                } else if (rRel < 0.72 && zRel < 0.76) {
                    graph.assigned[v] = ToolRegion::FLUTE;
                    graph.assignmentConfidence[v] = 0.6;
                } else {
                    graph.assigned[v] = ToolRegion::TOOL_BODY;
                    graph.assignmentConfidence[v] = 0.5;
                }
                break;
            }

            case RotaryToolFamily::THREADING: {
                double leadHeight = getJsonDouble(config, "machining_parameters",
                                                  "lead_length_mm", 0.5) / 1000.0;
                if (zFromTip <= leadHeight) {
                    graph.assigned[v] = ToolRegion::CHAMFER_LEAD;
                    graph.assignmentConfidence[v] = 0.8;
                } else if (rRel > 0.90 && normals[v].dot(
                    Vec3(pos.x / std::max(r, 1e-12), pos.y / std::max(r, 1e-12), 0)) > 0.35) {
                    graph.assigned[v] = ToolRegion::THREAD_CREST;
                    graph.assignmentConfidence[v] = 0.7;
                } else if (rRel > 0.72 && std::abs(normals[v].z) > 0.25) {
                    graph.assigned[v] = ToolRegion::THREAD_FLANK;
                    graph.assignmentConfidence[v] = 0.7;
                } else if (rRel < 0.68 && zRel < 0.76) {
                    graph.assigned[v] = ToolRegion::THREAD_ROOT;
                    graph.assignmentConfidence[v] = 0.65;
                } else if (generic == ToolRegion::RAKE_FACE) {
                    graph.assigned[v] = ToolRegion::RAKE_FACE;
                    graph.assignmentConfidence[v] = 0.6;
                } else if (generic == ToolRegion::FLANK_FACE) {
                    graph.assigned[v] = ToolRegion::RELIEF_FACE;
                    graph.assignmentConfidence[v] = 0.6;
                } else {
                    graph.assigned[v] = ToolRegion::FLUTE;
                    graph.assignmentConfidence[v] = 0.55;
                }
                break;
            }

            case RotaryToolFamily::BORING:
            case RotaryToolFamily::COUNTERSINK_COUNTERBORE: {
                bool countersink = containsAny(getDescriptorString(config),
                                               {"countersink", "counter sink"});
                bool counterbore = containsAny(getDescriptorString(config),
                                               {"counterbore", "spotface", "spot face"});

                if (countersink && zFromTip <= pointZoneHeight) {
                    if (rRel > 0.42) {
                        graph.assigned[v] = ToolRegion::COUNTERSINK_FACE;
                        graph.assignmentConfidence[v] = 0.8;
                    } else {
                        graph.assigned[v] = ToolRegion::CUTTING_EDGE;
                        graph.assignmentConfidence[v] = 0.7;
                    }
                } else if (counterbore && rRel > 0.85) {
                    graph.assigned[v] = ToolRegion::COUNTERBORE_FACE;
                    graph.assignmentConfidence[v] = 0.8;
                } else if (generic == ToolRegion::CUTTING_EDGE ||
                           (zFromTip <= pointZoneHeight && rRel > 0.70)) {
                    graph.assigned[v] = ToolRegion::CUTTING_EDGE;
                    graph.assignmentConfidence[v] = 0.75;
                } else if (generic == ToolRegion::RAKE_FACE) {
                    graph.assigned[v] = ToolRegion::RAKE_FACE;
                    graph.assignmentConfidence[v] = 0.65;
                } else if (generic == ToolRegion::FLANK_FACE) {
                    graph.assigned[v] = ToolRegion::FLANK_FACE;
                    graph.assignmentConfidence[v] = 0.65;
                } else if (rRel > 0.88) {
                    graph.assigned[v] = ToolRegion::MARGIN;
                    graph.assignmentConfidence[v] = 0.6;
                } else {
                    graph.assigned[v] = ToolRegion::CHIP_GULLET;
                    graph.assignmentConfidence[v] = 0.55;
                }
                break;
            }

            case RotaryToolFamily::BURR_DEBURRING: {
                if (rRel > 0.76) {
                    if (std::abs(normals[v].dot(
                        Vec3(-pos.y / std::max(r, 1e-12), pos.x / std::max(r, 1e-12), 0))) > 0.16 ||
                        std::abs(normals[v].z) > 0.35) {
                        graph.assigned[v] = ToolRegion::BURR_TOOTH;
                        graph.assignmentConfidence[v] = 0.75;
                    } else {
                        graph.assigned[v] = ToolRegion::BURR_GULLET;
                        graph.assignmentConfidence[v] = 0.65;
                    }
                } else {
                    graph.assigned[v] = ToolRegion::BURR_GULLET;
                    graph.assignmentConfidence[v] = 0.55;
                }
                break;
            }

            case RotaryToolFamily::UNKNOWN:
            default: {
                if (generic == ToolRegion::CUTTING_EDGE ||
                    (zFromTip <= pointZoneHeight && rRel > 0.65)) {
                    graph.assigned[v] = ToolRegion::CUTTING_EDGE;
                    graph.assignmentConfidence[v] = 0.7;
                } else if (generic == ToolRegion::MARGIN || rRel > 0.88) {
                    graph.assigned[v] = ToolRegion::MARGIN;
                    graph.assignmentConfidence[v] = 0.6;
                } else if (generic == ToolRegion::RAKE_FACE) {
                    graph.assigned[v] = ToolRegion::RAKE_FACE;
                    graph.assignmentConfidence[v] = 0.6;
                } else if (generic == ToolRegion::FLANK_FACE) {
                    graph.assigned[v] = ToolRegion::FLANK_FACE;
                    graph.assignmentConfidence[v] = 0.6;
                } else {
                    graph.assigned[v] = ToolRegion::FLUTE;
                    graph.assignmentConfidence[v] = 0.45;
                }
                break;
            }
        }
    }
}

// ============================================================================
// Main classify() entry point
// ============================================================================

void ToolRegionClassifier::classify(Mesh& mesh, const Config& config) {
    if (mesh.nodes.empty()) return;

    // Phase 1: PCA Auto-Alignment
    PCAResult pca = computePCA(mesh.nodes);
    if (pca.transform.m[0] != 1.0 || pca.transform.m[5] != 1.0 || pca.transform.m[10] != 1.0) {
        // Apply PCA alignment if it's not identity
        for (auto& node : mesh.nodes) {
            node.position = pca.transform.transformPoint(node.position);
            node.originalPosition = pca.transform.transformPoint(node.originalPosition);
        }
        // Recompute triangle normals
        for (auto& tri : mesh.triangles) {
            if (tri.indices[0] >= 0 && tri.indices[0] < (int)mesh.nodes.size() &&
                tri.indices[1] >= 0 && tri.indices[1] < (int)mesh.nodes.size() &&
                tri.indices[2] >= 0 && tri.indices[2] < (int)mesh.nodes.size()) {
                const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
                const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
                const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
                tri.normal = (v1 - v0).cross(v2 - v0).normalized();
            }
        }
    }

    // Phase 2: Build topology and orient normals
    MeshTopology topo = buildTopology(mesh);
    std::vector<Vec3> normals = topo.vertexNormals;
    orientNormalsOutward(normals, mesh.nodes, mesh.triangles, topo);

    // Phase 3: Curvature computation
    CurvatureData curv = computeCurvature(mesh, topo, normals);

    // Phase 4: Kinematic envelope
    double zBinSize = std::max(1e-5, pca.eigenvalues[0] > 0
        ? std::sqrt(pca.eigenvalues[0]) * 0.02 : 1e-4);
    EnvelopeData env = detectEnvelope(mesh, zBinSize);

    // Phase 5: Region growing
    RotaryToolFamily family = detectFamily(config);
    ClassificationGraph graph = growRegions(mesh, topo, curv, env, normals, family);

    // Phase 6: Family-specific assignment
    assignFamilyRegions(graph, mesh, topo, curv, env, normals, family, config);

    // Write results to mesh
    int classified = 0;
    for (int i = 0; i < (int)mesh.nodes.size() && i < graph.nodeCount; ++i) {
        mesh.nodes[i].toolRegion = graph.assigned[i];
        if (graph.assigned[i] != ToolRegion::UNKNOWN) classified++;
    }

    // Apply JSON overrides last (highest priority)
    applyAssistedOverrides(mesh, config);

    // Re-count after overrides
    classified = 0;
    for (const auto& node : mesh.nodes) {
        if (node.toolRegion != ToolRegion::UNKNOWN) classified++;
    }

    std::cout << "[ToolRegionClassifier] Family=" << familyName(family)
              << ", PCA aligned tool: primary axis=("
              << std::fixed << std::setprecision(3)
              << pca.primaryAxis.x << ", " << pca.primaryAxis.y << ", "
              << pca.primaryAxis.z << ")"
              << ", classified " << classified << "/" << mesh.nodes.size()
              << " CAD nodes via topological analysis"
              << " (confidence=" << std::fixed << std::setprecision(2)
              << (100.0 * classified / std::max<size_t>(mesh.nodes.size(), 1))
              << "%)" << std::endl;
}

// ============================================================================
// Family detection (unchanged from original)
// ============================================================================

std::string ToolRegionClassifier::getDescriptorString(const Config& config) {
    const auto& j = config.getJson();
    std::string descriptor;
    auto append = [&](const std::string& value) {
        if (!value.empty()) {
            if (!descriptor.empty()) descriptor += " ";
            descriptor += value;
        }
    };

    append(j.value("rotary_tool_family", ""));
    append(j.value("tool_family", ""));
    append(j.value("tool_type", ""));
    if (j.contains("machining_parameters")) {
        const auto& mp = j["machining_parameters"];
        append(mp.value("tool_type", ""));
        append(mp.value("tool_family", ""));
        append(mp.value("cutter_type", ""));
    }
    return lowerCopy(descriptor);
}

RotaryToolFamily ToolRegionClassifier::detectFamily(const Config& config) {
    const std::string d = getDescriptorString(config);

    if (containsAny(d, {"burr", "deburr", "rotary file"})) {
        return RotaryToolFamily::BURR_DEBURRING;
    }
    if (containsAny(d, {"countersink", "counter sink", "counterbore", "spotface", "spot face"})) {
        return RotaryToolFamily::COUNTERSINK_COUNTERBORE;
    }
    if (containsAny(d, {"tap", "thread mill", "threadmill", "thread"})) {
        return RotaryToolFamily::THREADING;
    }
    if (containsAny(d, {"ream"})) {
        return RotaryToolFamily::REAMING;
    }
    if (containsAny(d, {"boring", "bore"})) {
        return RotaryToolFamily::BORING;
    }
    if (containsAny(d, {"end mill", "endmill", "ball nose", "bull nose", "face mill",
                        "shell mill", "milling", "slot drill", "roughing"})) {
        return RotaryToolFamily::MILLING;
    }
    if (containsAny(d, {"drill", "holemaking", "gun drill", "center drill", "spot drill"})) {
        return RotaryToolFamily::DRILLING;
    }

    switch (config.getMachiningType()) {
        case MachiningType::DRILLING: return RotaryToolFamily::DRILLING;
        case MachiningType::REAMING: return RotaryToolFamily::REAMING;
        case MachiningType::THREADING: return RotaryToolFamily::THREADING;
        case MachiningType::BORING: return RotaryToolFamily::BORING;
        case MachiningType::MILLING: return RotaryToolFamily::MILLING;
        default: return RotaryToolFamily::UNKNOWN;
    }
}

// ============================================================================
// JSON overrides (unchanged from original)
// ============================================================================

ToolRegion ToolRegionClassifier::parseRegionName(const std::string& name) {
    std::string key = lowerCopy(name);
    std::replace(key.begin(), key.end(), '-', '_');
    std::replace(key.begin(), key.end(), ' ', '_');

    const int regionCount = static_cast<int>(ToolRegion::REGION_COUNT);
    for (int i = 0; i < regionCount; ++i) {
        ToolRegion region = static_cast<ToolRegion>(i);
        if (key == regionName(region)) return region;
    }

    if (key == "edge" || key == "cutting_lip" || key == "lip") return ToolRegion::CUTTING_EDGE;
    if (key == "chisel" || key == "web") return ToolRegion::CHISEL_EDGE;
    if (key == "rake" || key == "chip_face") return ToolRegion::RAKE_FACE;
    if (key == "flank" || key == "relief") return ToolRegion::FLANK_FACE;
    if (key == "body") return ToolRegion::TOOL_BODY;
    return ToolRegion::UNKNOWN;
}

void ToolRegionClassifier::applyAssistedOverrides(Mesh& mesh, const Config& config) {
    const auto& j = config.getJson();
    if (!j.contains("tool_region_overrides") || !j["tool_region_overrides"].is_array()) {
        return;
    }

    int touched = 0;
    for (const auto& rule : j["tool_region_overrides"]) {
        if (!rule.is_object()) continue;
        ToolRegion region = ToolRegion::UNKNOWN;
        if (rule.contains("region")) region = parseRegionName(rule.value("region", ""));
        if (region == ToolRegion::UNKNOWN && rule.contains("name")) {
            region = parseRegionName(rule.value("name", ""));
        }
        if (region == ToolRegion::UNKNOWN) continue;

        if (rule.contains("node_ids") && rule["node_ids"].is_array()) {
            for (const auto& idv : rule["node_ids"]) {
                int id = idv.get<int>();
                if (id >= 0 && id < static_cast<int>(mesh.nodes.size())) {
                    mesh.nodes[id].toolRegion = region;
                    touched++;
                }
            }
            continue;
        }

        bool hasBox = rule.contains("min_mm") && rule.contains("max_mm") &&
                      rule["min_mm"].is_array() && rule["max_mm"].is_array() &&
                      rule["min_mm"].size() >= 3 && rule["max_mm"].size() >= 3;
        Vec3 minBox, maxBox;
        if (hasBox) {
            minBox = Vec3(rule["min_mm"][0].get<double>() / 1000.0,
                          rule["min_mm"][1].get<double>() / 1000.0,
                          rule["min_mm"][2].get<double>() / 1000.0);
            maxBox = Vec3(rule["max_mm"][0].get<double>() / 1000.0,
                          rule["max_mm"][1].get<double>() / 1000.0,
                          rule["max_mm"][2].get<double>() / 1000.0);
        }

        bool hasZ = rule.contains("z_range_mm") && rule["z_range_mm"].is_array() &&
                    rule["z_range_mm"].size() >= 2;
        double zMin = hasZ ? rule["z_range_mm"][0].get<double>() / 1000.0 : -1e30;
        double zMax = hasZ ? rule["z_range_mm"][1].get<double>() / 1000.0 :  1e30;

        bool hasRadius = rule.contains("radius_range_mm") &&
                         rule["radius_range_mm"].is_array() &&
                         rule["radius_range_mm"].size() >= 2;
        double rMin = hasRadius ? rule["radius_range_mm"][0].get<double>() / 1000.0 : 0.0;
        double rMax = hasRadius ? rule["radius_range_mm"][1].get<double>() / 1000.0 : 1e30;

        for (auto& node : mesh.nodes) {
            const Vec3& p = node.position;
            bool match = true;
            if (hasBox) {
                match = p.x >= minBox.x && p.x <= maxBox.x &&
                        p.y >= minBox.y && p.y <= maxBox.y &&
                        p.z >= minBox.z && p.z <= maxBox.z;
            }
            if (match && hasZ) match = p.z >= zMin && p.z <= zMax;
            if (match && hasRadius) {
                double r = std::sqrt(p.x * p.x + p.y * p.y);
                match = r >= rMin && r <= rMax;
            }
            if (match && (hasBox || hasZ || hasRadius)) {
                node.toolRegion = region;
                touched++;
            }
        }
    }

    if (touched > 0) {
        std::cout << "[ToolRegionClassifier] Applied assisted CAD overrides to "
                  << touched << " nodes" << std::endl;
    }
}

// ============================================================================
// Risk fields annotation
// ============================================================================

void ToolRegionClassifier::annotateRiskFields(Mesh& mesh, const Config& config) {
    const auto& tool = config.getToolMaterial();
    const double ambient = config.getMachining().ambientTemperature;
    const double yield = std::max(tool.yieldStrength, 1.0);
    const double melt = std::max(tool.meltingPoint, ambient + 1.0);
    const double coatingThickness = tool.coatingThickness > 0.0
        ? tool.coatingThickness
        : 4.0e-6;

    for (auto& node : mesh.nodes) {
        const double wearRatio = node.accumulatedWear / std::max(coatingThickness, 1e-12);
        const double stressRatio = node.stress / yield;
        const double tempRatio = (node.temperature - ambient) / (melt - ambient);
        const double regionMultiplier = isEdgeRegion(node.toolRegion) ? 1.25 :
            (isRakeRegion(node.toolRegion) || isFlankRegion(node.toolRegion)) ? 1.0 : 0.45;

        node.chippingRisk = clamp01(regionMultiplier *
            (0.58 * stressRatio + 0.25 * clamp01(tempRatio) + 0.17 * clamp01(wearRatio)));
        node.coatingRemaining = clamp01(1.0 - wearRatio);

        if (isFlankRegion(node.toolRegion) || isEdgeRegion(node.toolRegion)) {
            node.flankWear = std::max(node.flankWear, node.accumulatedWear);
        }
        if (isRakeRegion(node.toolRegion)) {
            node.craterWear = std::max(node.craterWear, node.accumulatedWear);
        }
    }
}

// ============================================================================
// Build report & export
// ============================================================================

ToolRegionReport ToolRegionClassifier::buildReport(const Mesh& mesh, const Config& config) {
    ToolRegionReport report;
    report.family = detectFamily(config);
    report.familyName = familyName(report.family);
    report.totalNodes = static_cast<int>(mesh.nodes.size());

    const int regionCount = static_cast<int>(ToolRegion::REGION_COUNT);
    report.regions.resize(regionCount);
    for (int i = 0; i < regionCount; ++i) {
        report.regions[i].region = static_cast<ToolRegion>(i);
    }

    for (const auto& node : mesh.nodes) {
        int idx = static_cast<int>(node.toolRegion);
        if (idx < 0 || idx >= regionCount) idx = 0;
        auto& agg = report.regions[idx];
        agg.nodeCount++;
        if (node.isContact) agg.contactNodeCount++;
        agg.maxStress = std::max(agg.maxStress, node.stress);
        agg.avgStress += node.stress;
        agg.maxTemperature = std::max(agg.maxTemperature, node.temperature);
        agg.avgTemperature += node.temperature;
        agg.maxWear = std::max(agg.maxWear, node.accumulatedWear);
        agg.maxFlankWear = std::max(agg.maxFlankWear, node.flankWear);
        agg.maxCraterWear = std::max(agg.maxCraterWear, node.craterWear);
        agg.maxChippingRisk = std::max(agg.maxChippingRisk, node.chippingRisk);
        agg.minCoatingRemaining = std::min(agg.minCoatingRemaining, node.coatingRemaining);
        if (node.toolRegion != ToolRegion::UNKNOWN) report.classifiedNodes++;
    }

    for (auto& agg : report.regions) {
        if (agg.nodeCount > 0) {
            agg.avgStress /= agg.nodeCount;
            agg.avgTemperature /= agg.nodeCount;
        }
    }

    report.classificationConfidence = report.totalNodes > 0
        ? static_cast<double>(report.classifiedNodes) / report.totalNodes
        : 0.0;
    return report;
}

void ToolRegionClassifier::exportReportJson(const std::string& path,
                                            const ToolRegionReport& report) {
    nlohmann::json root;
    root["tool_family"] = report.familyName;
    root["tool_family_id"] = static_cast<int>(report.family);
    root["classification_confidence"] = report.classificationConfidence;
    root["classified_nodes"] = report.classifiedNodes;
    root["total_nodes"] = report.totalNodes;
    root["regions"] = nlohmann::json::array();

    for (const auto& agg : report.regions) {
        if (agg.nodeCount == 0) continue;
        nlohmann::json r;
        r["name"] = regionName(agg.region);
        r["id"] = static_cast<int>(agg.region);
        r["node_count"] = agg.nodeCount;
        r["contact_node_count"] = agg.contactNodeCount;
        r["max_stress_MPa"] = agg.maxStress / 1e6;
        r["avg_stress_MPa"] = agg.avgStress / 1e6;
        r["max_temperature_C"] = agg.maxTemperature;
        r["avg_temperature_C"] = agg.avgTemperature;
        r["max_wear_um"] = agg.maxWear * 1e6;
        r["max_flank_wear_um"] = agg.maxFlankWear * 1e6;
        r["max_crater_wear_um"] = agg.maxCraterWear * 1e6;
        r["max_chipping_risk"] = agg.maxChippingRisk;
        r["min_coating_remaining_percent"] = agg.minCoatingRemaining * 100.0;
        root["regions"].push_back(r);
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ToolRegionClassifier] Failed to write report: " << path << std::endl;
        return;
    }
    file << root.dump(2);
}

} // namespace edgepredict
