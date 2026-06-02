#pragma once

#include "Types.h"
#include "Config.h"
#include <string>
#include <vector>

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
    struct NodeContext;
    struct GeometryContext;

    static RotaryToolFamily detectFamily(const Config& config);
    static ToolRegion classifyNode(const FEMNode& node,
                                   const Vec3& normal,
                                   const GeometryContext& ctx,
                                   RotaryToolFamily family);
    static ToolRegion classifyDrillingNode(const NodeContext& n, const GeometryContext& ctx);
    static ToolRegion classifyReamingNode(const NodeContext& n, const GeometryContext& ctx);
    static ToolRegion classifyMillingNode(const NodeContext& n, const GeometryContext& ctx);
    static ToolRegion classifyThreadingNode(const NodeContext& n, const GeometryContext& ctx);
    static ToolRegion classifyBoringNode(const NodeContext& n, const GeometryContext& ctx);
    static ToolRegion classifyBurrNode(const NodeContext& n, const GeometryContext& ctx);
    static ToolRegion classifyGenericRotaryNode(const NodeContext& n, const GeometryContext& ctx);
    static void applyAssistedOverrides(Mesh& mesh, const Config& config);
    static ToolRegion parseRegionName(const std::string& name);
};

} // namespace edgepredict
