/**
 * @file MachiningStrategyFactory.cpp
 * @brief Factory implementation for creating machining strategies
 */

#include "IMachiningStrategy.h"
#include "MillingStrategy.h"
#include "DrillingStrategy.h"
#include <iostream>

namespace edgepredict {

std::unique_ptr<IMachiningStrategy> MachiningStrategyFactory::create(MachiningType type) {
    switch (type) {
        case MachiningType::MILLING:
            std::cout << "[Factory] Creating MillingStrategy" << std::endl;
            return std::make_unique<MillingStrategy>();
            
        case MachiningType::DRILLING:
            std::cout << "[Factory] Creating DrillingStrategy" << std::endl;
            return std::make_unique<DrillingStrategy>();
            
        case MachiningType::REAMING:
        case MachiningType::THREADING:
        case MachiningType::BORING:
            // For now, use milling as base for these rotary ops until their specifics are implemented
            std::cout << "[Factory] Operation not yet fully implemented, using MillingStrategy as base" << std::endl;
            return std::make_unique<MillingStrategy>();
            
        default:
            std::cout << "[Factory] Unknown type, defaulting to MillingStrategy" << std::endl;
            return std::make_unique<MillingStrategy>();
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
