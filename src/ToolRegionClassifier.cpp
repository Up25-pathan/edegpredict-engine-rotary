#include "ToolRegionClassifier.h"
#include "json.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>

namespace edgepredict {

namespace {
constexpr double PI = 3.14159265358979323846;

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

std::string getDescriptorString(const Config& config) {
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

std::vector<Vec3> computeNodeNormals(const Mesh& mesh) {
    std::vector<Vec3> normals(mesh.nodes.size(), Vec3::zero());
    for (const auto& tri : mesh.triangles) {
        int a = tri.indices[0];
        int b = tri.indices[1];
        int c = tri.indices[2];
        if (a < 0 || b < 0 || c < 0 ||
            a >= static_cast<int>(mesh.nodes.size()) ||
            b >= static_cast<int>(mesh.nodes.size()) ||
            c >= static_cast<int>(mesh.nodes.size())) {
            continue;
        }

        const Vec3& v0 = mesh.nodes[a].position;
        const Vec3& v1 = mesh.nodes[b].position;
        const Vec3& v2 = mesh.nodes[c].position;
        Vec3 areaNormal = (v1 - v0).cross(v2 - v0);
        normals[a] += areaNormal;
        normals[b] += areaNormal;
        normals[c] += areaNormal;
    }

    for (auto& n : normals) {
        n = n.normalized();
    }
    return normals;
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

struct ToolRegionClassifier::NodeContext {
    const FEMNode& node;
    Vec3 normal;
    double zRel = 0.0;
    double zFromTip = 0.0;
    double r = 0.0;
    double rRel = 0.0;
    double radialNormal = 0.0;
    double tangentialNormal = 0.0;
    double axialNormal = 0.0;
};

struct ToolRegionClassifier::GeometryContext {
    double minZ = 0.0;
    double maxZ = 0.0;
    double span = 0.0;
    double maxRadius = 0.0;
    double pointZoneHeight = 0.0;
    double webRadius = 0.0;
    double leadZoneHeight = 0.0;
    double endZoneHeight = 0.0;
    double shankStartRel = 0.82;
    std::string descriptor;
    bool ballNose = false;
    bool indexable = false;
    bool countersink = false;
    bool counterbore = false;
};

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

void ToolRegionClassifier::classify(Mesh& mesh, const Config& config) {
    if (mesh.nodes.empty()) return;

    GeometryContext ctx;
    ctx.minZ = 1e30;
    ctx.maxZ = -1e30;
    for (const auto& node : mesh.nodes) {
        ctx.minZ = std::min(ctx.minZ, node.position.z);
        ctx.maxZ = std::max(ctx.maxZ, node.position.z);
        ctx.maxRadius = std::max(ctx.maxRadius, std::sqrt(
            node.position.x * node.position.x + node.position.y * node.position.y));
    }

    ctx.span = ctx.maxZ - ctx.minZ;
    if (ctx.maxRadius <= 1e-12 || ctx.span <= 1e-12) {
        for (auto& node : mesh.nodes) node.toolRegion = ToolRegion::UNKNOWN;
        return;
    }

    ctx.descriptor = getDescriptorString(config);
    ctx.ballNose = containsAny(ctx.descriptor, {"ball nose", "ballnose", "spherical"});
    ctx.indexable = containsAny(ctx.descriptor, {"indexable", "insert"});
    ctx.countersink = containsAny(ctx.descriptor, {"countersink", "counter sink"});
    ctx.counterbore = containsAny(ctx.descriptor, {"counterbore", "spotface", "spot face"});

    double pointAngleDeg = getJsonDouble(config, "machining_parameters",
                                         "point_angle_deg", 118.0);
    double chiselRatio = getJsonDouble(config, "machining_parameters",
                                       "chisel_edge_ratio", 0.18);
    pointAngleDeg = std::max(60.0, std::min(160.0, pointAngleDeg));
    chiselRatio = std::max(0.05, std::min(0.45, chiselRatio));

    const double pointHalfAngleRad = 0.5 * pointAngleDeg * PI / 180.0;
    const double pointHeight = ctx.maxRadius / std::max(std::tan(pointHalfAngleRad), 1e-6);
    ctx.pointZoneHeight = std::max(ctx.span * 0.08, std::min(ctx.span * 0.35, pointHeight * 1.5));
    ctx.webRadius = ctx.maxRadius * chiselRatio;
    ctx.leadZoneHeight = std::max(ctx.span * 0.04, std::min(ctx.span * 0.18, ctx.maxRadius * 0.75));
    ctx.endZoneHeight = std::max(ctx.span * 0.035, std::min(ctx.span * 0.14, ctx.maxRadius * 0.60));

    auto normals = computeNodeNormals(mesh);
    RotaryToolFamily family = detectFamily(config);
    int classified = 0;
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        mesh.nodes[i].toolRegion = classifyNode(mesh.nodes[i], normals[i], ctx, family);
        if (mesh.nodes[i].toolRegion != ToolRegion::UNKNOWN) classified++;
    }

    applyAssistedOverrides(mesh, config);
    classified = 0;
    for (const auto& node : mesh.nodes) {
        if (node.toolRegion != ToolRegion::UNKNOWN) classified++;
    }

    std::cout << "[ToolRegionClassifier] Family=" << familyName(family)
              << ", classified " << classified << "/" << mesh.nodes.size()
              << " CAD nodes (confidence=" << std::fixed << std::setprecision(2)
              << (100.0 * classified / std::max<size_t>(mesh.nodes.size(), 1))
              << "%)" << std::endl;
}

ToolRegion ToolRegionClassifier::classifyNode(const FEMNode& node,
                                              const Vec3& normal,
                                              const GeometryContext& ctx,
                                              RotaryToolFamily family) {
    const double r = std::sqrt(node.position.x * node.position.x +
                               node.position.y * node.position.y);
    Vec3 radial = r > 1e-12
        ? Vec3(node.position.x / r, node.position.y / r, 0.0)
        : Vec3::zero();
    Vec3 tangent = r > 1e-12
        ? Vec3(-node.position.y / r, node.position.x / r, 0.0)
        : Vec3::zero();

    NodeContext n{
        node,
        normal,
        (node.position.z - ctx.minZ) / std::max(ctx.span, 1e-12),
        node.position.z - ctx.minZ,
        r,
        r / std::max(ctx.maxRadius, 1e-12),
        normal.dot(radial),
        normal.dot(tangent),
        normal.z
    };

    switch (family) {
        case RotaryToolFamily::DRILLING: return classifyDrillingNode(n, ctx);
        case RotaryToolFamily::REAMING: return classifyReamingNode(n, ctx);
        case RotaryToolFamily::MILLING: return classifyMillingNode(n, ctx);
        case RotaryToolFamily::THREADING: return classifyThreadingNode(n, ctx);
        case RotaryToolFamily::BORING:
        case RotaryToolFamily::COUNTERSINK_COUNTERBORE: return classifyBoringNode(n, ctx);
        case RotaryToolFamily::BURR_DEBURRING: return classifyBurrNode(n, ctx);
        case RotaryToolFamily::UNKNOWN:
        default: return classifyGenericRotaryNode(n, ctx);
    }
}

ToolRegion ToolRegionClassifier::classifyDrillingNode(const NodeContext& n,
                                                      const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (n.zFromTip <= ctx.pointZoneHeight) {
        if (n.r <= ctx.webRadius) return ToolRegion::CHISEL_EDGE;
        if (std::abs(n.tangentialNormal) > 0.18) {
            return n.tangentialNormal < 0.0 ? ToolRegion::RAKE_FACE : ToolRegion::FLANK_FACE;
        }
        return ToolRegion::CUTTING_EDGE;
    }
    if (n.rRel > 0.88) return ToolRegion::MARGIN;
    if (n.zRel < 0.72) return ToolRegion::FLUTE;
    return ToolRegion::TOOL_BODY;
}

ToolRegion ToolRegionClassifier::classifyReamingNode(const NodeContext& n,
                                                     const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (n.zFromTip <= ctx.leadZoneHeight) {
        if (n.rRel > 0.72) return ToolRegion::LEAD_CHAMFER;
        return ToolRegion::FLUTE;
    }
    if (n.rRel > 0.92) return ToolRegion::MARGIN;
    if (n.rRel > 0.82) return ToolRegion::LAND;
    if (std::abs(n.tangentialNormal) > 0.22) {
        return n.tangentialNormal < 0.0 ? ToolRegion::RAKE_FACE : ToolRegion::FLANK_FACE;
    }
    if (n.zRel < 0.76) return ToolRegion::FLUTE;
    return ToolRegion::TOOL_BODY;
}

ToolRegion ToolRegionClassifier::classifyMillingNode(const NodeContext& n,
                                                     const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (ctx.indexable && n.rRel > 0.72 && n.zRel < 0.55) {
        if (n.tangentialNormal < -0.18) return ToolRegion::INSERT_RAKE_FACE;
        if (n.tangentialNormal > 0.18) return ToolRegion::INSERT_FLANK_FACE;
        return ToolRegion::PERIPHERAL_CUTTING_EDGE;
    }
    if (n.zFromTip <= ctx.endZoneHeight) {
        if (ctx.ballNose && n.rRel < 0.62) return ToolRegion::BALL_NOSE;
        if (n.rRel > 0.82) return ToolRegion::CORNER_RADIUS;
        return ToolRegion::END_CUTTING_EDGE;
    }
    if (n.rRel > 0.88) return ToolRegion::PERIPHERAL_CUTTING_EDGE;
    if (std::abs(n.tangentialNormal) > 0.24) {
        return n.tangentialNormal < 0.0 ? ToolRegion::RAKE_FACE : ToolRegion::FLANK_FACE;
    }
    if (n.rRel < 0.72 && n.zRel < 0.76) return ToolRegion::FLUTE;
    return ToolRegion::TOOL_BODY;
}

ToolRegion ToolRegionClassifier::classifyThreadingNode(const NodeContext& n,
                                                       const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (n.zFromTip <= ctx.leadZoneHeight) return ToolRegion::CHAMFER_LEAD;
    if (n.rRel > 0.90 && n.radialNormal > 0.35) return ToolRegion::THREAD_CREST;
    if (n.rRel > 0.72 && std::abs(n.axialNormal) > 0.25) return ToolRegion::THREAD_FLANK;
    if (n.rRel < 0.68 && n.zRel < 0.76) return ToolRegion::THREAD_ROOT;
    if (std::abs(n.tangentialNormal) > 0.22) {
        return n.tangentialNormal < 0.0 ? ToolRegion::RAKE_FACE : ToolRegion::RELIEF_FACE;
    }
    if (n.zRel < 0.76) return ToolRegion::FLUTE;
    return ToolRegion::TOOL_BODY;
}

ToolRegion ToolRegionClassifier::classifyBoringNode(const NodeContext& n,
                                                    const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (ctx.countersink && n.zFromTip <= ctx.pointZoneHeight) {
        if (n.rRel > 0.42) return ToolRegion::COUNTERSINK_FACE;
        return ToolRegion::CUTTING_EDGE;
    }
    if (ctx.counterbore && n.rRel > 0.85) return ToolRegion::COUNTERBORE_FACE;
    if (n.zFromTip <= ctx.leadZoneHeight && n.rRel > 0.70) return ToolRegion::CUTTING_EDGE;
    if (std::abs(n.tangentialNormal) > 0.22) {
        return n.tangentialNormal < 0.0 ? ToolRegion::RAKE_FACE : ToolRegion::FLANK_FACE;
    }
    if (n.rRel > 0.88) return ToolRegion::MARGIN;
    if (n.zRel < 0.72) return ToolRegion::CHIP_GULLET;
    return ToolRegion::TOOL_BODY;
}

ToolRegion ToolRegionClassifier::classifyBurrNode(const NodeContext& n,
                                                  const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (n.rRel > 0.76) {
        if (std::abs(n.tangentialNormal) > 0.16 || std::abs(n.axialNormal) > 0.35) {
            return ToolRegion::BURR_TOOTH;
        }
        return ToolRegion::BURR_GULLET;
    }
    if (n.zRel < 0.78) return ToolRegion::BURR_GULLET;
    return ToolRegion::TOOL_BODY;
}

ToolRegion ToolRegionClassifier::classifyGenericRotaryNode(const NodeContext& n,
                                                           const GeometryContext& ctx) {
    if (n.zRel > ctx.shankStartRel) return ToolRegion::SHANK;
    if (n.zFromTip <= ctx.leadZoneHeight && n.rRel > 0.65) return ToolRegion::CUTTING_EDGE;
    if (n.rRel > 0.88) return ToolRegion::MARGIN;
    if (std::abs(n.tangentialNormal) > 0.22) {
        return n.tangentialNormal < 0.0 ? ToolRegion::RAKE_FACE : ToolRegion::FLANK_FACE;
    }
    if (n.zRel < 0.76) return ToolRegion::FLUTE;
    return ToolRegion::TOOL_BODY;
}

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
