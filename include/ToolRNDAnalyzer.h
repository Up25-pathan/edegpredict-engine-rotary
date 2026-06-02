#pragma once

#include "Types.h"
#include "Config.h"
#include "ToolRegionClassifier.h"
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
};

class ToolRNDAnalyzer {
public:
    static void exportJson(const std::string& path, const ToolRNDAnalysisInput& input);
};

} // namespace edgepredict
