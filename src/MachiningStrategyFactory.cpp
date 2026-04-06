/**
 * @file MachiningStrategyFactory.cpp
 * @brief Factory implementation for creating machining strategies
 * 
 * Each MachiningType maps to exactly one Strategy with correct kinematics.
 * No silent fallbacks — wrong physics is worse than crashing.
 */

#include "IMachiningStrategy.h"
#include "MillingStrategy.h"
#include "DrillingStrategy.h"
#include "BoringStrategy.h"
#include "ReamingStrategy.h"
#include "ThreadingStrategy.h"
#include <iostream>
#include <stdexcept>

namespace edgepredict {

std::unique_ptr<IMachiningStrategy> MachiningStrategyFactory::create(MachiningType type) {
    switch (type) {
        case MachiningType::MILLING:
            std::cout << "[Factory] Creating MillingStrategy" << std::endl;
            return std::make_unique<MillingStrategy>();
            
        case MachiningType::DRILLING:
            std::cout << "[Factory] Creating DrillingStrategy" << std::endl;
            return std::make_unique<DrillingStrategy>();
            
        case MachiningType::BORING:
            std::cout << "[Factory] Creating BoringStrategy" << std::endl;
            return std::make_unique<BoringStrategy>();

        case MachiningType::REAMING:
            std::cout << "[Factory] Creating ReamingStrategy" << std::endl;
            return std::make_unique<ReamingStrategy>();

        case MachiningType::THREADING:
            std::cout << "[Factory] Creating ThreadingStrategy" << std::endl;
            return std::make_unique<ThreadingStrategy>();
            
        default:
            throw std::runtime_error(
                "[Factory] FATAL: Unknown MachiningType — cannot create strategy. "
                "Check machining_type in your config JSON.");
    }
}

std::unique_ptr<IMachiningStrategy> MachiningStrategyFactory::createFromConfig(const Config& config) {
    auto strategy = create(config.getMachiningType());
    if (strategy) {
        strategy->initialize(config);
    }
    return strategy;
}

} // namespace edgepredict
