#pragma once

#include "Types.h"
#include "Config.h"
#include "ToolRegionClassifier.h"
#include "EdgeSubgridModel.h"
#include "BUEModel.h"
#include "ChatterDynamics.h"
#include <string>
#include <vector>

namespace edgepredict {

struct ToolRNDAnalysisInput {
    const Mesh* toolMesh = nullptr;
    const std::vector<MPMParticle>* particles = nullptr;
    const Config* config = nullptr;
    double contactHeatJ = 0.0;
    double contactForceN = 0.0;
    double currentTimeS = 0.0;
    ToolRegionReport regionReport;

    // Advanced model state (optional)
    const EdgeSubgridModel* edgeSubgrid = nullptr;
    const BUEModel* bueModel = nullptr;
    const ChatterDynamics* chatter = nullptr;
};

class ToolRNDAnalyzer {
public:
    static void exportJson(const std::string& path, const ToolRNDAnalysisInput& input);
};

} // namespace edgepredict
