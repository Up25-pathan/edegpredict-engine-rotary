#pragma once

#include "Types.h"
#include "Config.h"
#include "CoordinateSystem.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>

namespace edgepredict {

enum class RotaryToolFamily {
    UNKNOWN,
    DRILLING,
    REAMING,
    MILLING,
    THREADING,
    BORING,
    COUNTERSINK_COUNTERBORE,
    BURR_DEBURRING
};

struct ToolRegionAggregate {
    ToolRegion region = ToolRegion::UNKNOWN;
    int nodeCount = 0;
    int contactNodeCount = 0;
    double maxStress = 0.0;
    double avgStress = 0.0;
    double maxTemperature = 0.0;
    double avgTemperature = 0.0;
    double maxWear = 0.0;
    double maxFlankWear = 0.0;
    double maxCraterWear = 0.0;
    double maxChippingRisk = 0.0;
    double minCoatingRemaining = 1.0;
};

struct ToolRegionReport {
    RotaryToolFamily family = RotaryToolFamily::UNKNOWN;
    std::string familyName;
    double classificationConfidence = 0.0;
    int classifiedNodes = 0;
    int totalNodes = 0;
    std::vector<ToolRegionAggregate> regions;
};

class ToolRegionClassifier {
public:
    static void classify(Mesh& mesh, const Config& config);
    static void annotateRiskFields(Mesh& mesh, const Config& config);
    static ToolRegionReport buildReport(const Mesh& mesh, const Config& config);
    static void exportReportJson(const std::string& path, const ToolRegionReport& report);
    static const char* regionName(ToolRegion region);
    static const char* familyName(RotaryToolFamily family);

private:
    // ----------------------------------------------------------------
    // Phase 1: PCA Auto-Alignment
    // ----------------------------------------------------------------
    struct PCAResult {
        Mat4 transform;         // Aligns tool axis to Z, tip to origin
        Vec3 centroid;
        Vec3 primaryAxis;       // Largest eigenvector
        Vec3 secondaryAxis;
        Vec3 tertiaryAxis;
        double eigenvalues[3];
    };
    static PCAResult computePCA(const std::vector<FEMNode>& nodes);

    // ----------------------------------------------------------------
    // Phase 2: Normal Orientation
    // ----------------------------------------------------------------
    struct HalfEdge {
        int vertA, vertB;
        int faceA, faceB;   // -1 if boundary
    };
    struct MeshTopology {
        std::vector<Vec3> vertexNormals;           // Per-vertex area-weighted normal
        std::vector<std::vector<int>> vertexFaces; // Faces adjacent to each vertex
        std::vector<std::vector<int>> vertexVertices; // Edge-adjacent vertices
        std::vector<HalfEdge> halfEdges;
    };
    static MeshTopology buildTopology(const Mesh& mesh);
    static void orientNormalsOutward(std::vector<Vec3>& normals,
                                     const std::vector<FEMNode>& nodes,
                                     const std::vector<Triangle>& triangles,
                                     const MeshTopology& topo);

    // ----------------------------------------------------------------
    // Phase 3: Dihedral Edge Detection + Curvature
    // ----------------------------------------------------------------
    struct CurvatureData {
        std::vector<double> meanCurvature;       // Cotangent Laplacian H
        std::vector<double> gaussianCurvature;   // Angle defect K
        std::vector<double> dihedralMax;         // Max dihedral angle at vertex (rad)
        std::vector<bool> isSharpEdge;           // Dihedral > threshold
    };
    static CurvatureData computeCurvature(const Mesh& mesh,
                                          const MeshTopology& topo,
                                          const std::vector<Vec3>& normals);

    // ----------------------------------------------------------------
    // Phase 4: Kinematic Envelope
    // ----------------------------------------------------------------
    struct EnvelopeData {
        std::vector<bool> onEnvelope;
        double maxRadius;
        std::vector<double> zSliceMaxRadius; // Per Z-slice
        double zBinSize;
    };
    static EnvelopeData detectEnvelope(const Mesh& mesh, double zBinSize = 0.0001);

    // ----------------------------------------------------------------
    // Phase 5: Seed-Based Region Growing
    // ----------------------------------------------------------------
    struct ClassificationGraph {
        int nodeCount;
        std::vector<ToolRegion> assigned;
        std::vector<double> assignmentConfidence;
    };
    static ClassificationGraph growRegions(
        const Mesh& mesh,
        const MeshTopology& topo,
        const CurvatureData& curv,
        const EnvelopeData& env,
        const std::vector<Vec3>& normals,
        RotaryToolFamily family);

    // ----------------------------------------------------------------
    // Seed identification per tool family
    // ----------------------------------------------------------------
    static std::vector<int> findCuttingEdgeSeeds(const std::vector<FEMNode>& nodes,
                                                  const MeshTopology& topo,
                                                  const CurvatureData& curv,
                                                  const EnvelopeData& env,
                                                  const std::vector<Vec3>& normals,
                                                  RotaryToolFamily family);

    // ----------------------------------------------------------------
    // Phase 6: Family mapping
    // ----------------------------------------------------------------
    static void assignFamilyRegions(ClassificationGraph& graph,
                                    const Mesh& mesh,
                                    const MeshTopology& topo,
                                    const CurvatureData& curv,
                                    const EnvelopeData& env,
                                    const std::vector<Vec3>& normals,
                                    RotaryToolFamily family,
                                    const Config& config);

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------
    static RotaryToolFamily detectFamily(const Config& config);
    static void applyAssistedOverrides(Mesh& mesh, const Config& config);
    static ToolRegion parseRegionName(const std::string& name);
    static std::string getDescriptorString(const Config& config);

    // Dihedral angle between two faces (0..PI)
    static double computeDihedralAngle(const Vec3& nA, const Vec3& nB, const Vec3& edgeDir);

    // Normal direction relative to tangential velocity of rotation
    enum class NormalClass { UNKNOWN, RAKE_LIKE, FLANK_LIKE, NEUTRAL };
    static NormalClass classifyNormalVsRotation(const Vec3& normal,
                                                 const Vec3& position,
                                                 double rotationDirection = 1.0);
};

} // namespace edgepredict
