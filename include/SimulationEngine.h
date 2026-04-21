#pragma once
/**
 * @file SimulationEngine.h
 * @brief Main simulation orchestrator
 *
 * Kept intentionally slim — all actual physics is delegated to solvers.
 * Fix: added forward declarations for SPHSolver and FEMSolver so that
 *      dynamic_cast in SimulationEngine.cpp resolves without a circular include.
 *      (SimulationEngine.cpp includes the full headers; the .h only needs fwd decls.)
 */

#include "Types.h"
#include "Config.h"
#include "IPhysicsSolver.h"
#include "IMachiningStrategy.h"
#include "CoordinateSystem.h"
#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include "AsyncExporter.h"
#include "ResidualStressAnalyzer.h"

namespace edgepredict {

// Forward declarations — full headers included in SimulationEngine.cpp
class GeometryLoader;
class GCodeInterpreter;
class IExporter;
class ContactSolver;
class CFDSolverGPU;
class OptimizationManager;
class SurfaceRoughnessPredictor;
class ResidualStressAnalyzer;
class SPHSolver;
class FEMSolver;
struct SPHParticle;

/**
 * @brief Simulation step callback
 */
using StepCallback       = std::function<void(int step, double time,
                                               double maxStress, double maxTemp)>;
using CompletionCallback = std::function<void(bool success, const std::string& message)>;

/**
 * @brief Main simulation engine orchestrator
 */
class SimulationEngine {
public:
    SimulationEngine();
    ~SimulationEngine();

    SimulationEngine(const SimulationEngine&)            = delete;
    SimulationEngine& operator=(const SimulationEngine&) = delete;

    // Initialisation
    bool initialize(const std::string& configPath);
    bool initialize(const Config& config);

    // Run / control
    void run();
    bool step(double dt);
    void stop();

    bool isRunning()      const { return m_isRunning;     }
    bool isInitialized()  const { return m_isInitialized; }

    // Dependency injection
    void addSolver(std::unique_ptr<IPhysicsSolver> solver);
    void addExporter(std::unique_ptr<IExporter> exporter);
    void setStrategy(std::unique_ptr<IMachiningStrategy> strategy);
    void setToolMesh(const Mesh& mesh);

    IMachiningStrategy* getStrategy() const { return m_strategy.get(); }

    void setOptimizationManager(std::unique_ptr<OptimizationManager> manager);
    void setRoughnessPredictor(std::unique_ptr<SurfaceRoughnessPredictor> predictor);
    void setStressAnalyzer(std::unique_ptr<ResidualStressAnalyzer> analyzer);

    void setContactSolver(ContactSolver* solver);
    void setCFDSolver(CFDSolverGPU* solver);
    void setCoordinateSystem(const CNCCoordinateSystem& cs) { m_coordinateSystem = cs; }
    const CNCCoordinateSystem& getCoordinateSystem() const { return m_coordinateSystem; }

    void setStepCallback(StepCallback callback)           { m_stepCallback       = callback; }
    void setCompletionCallback(CompletionCallback callback){ m_completionCallback = callback; }

    // Accessors
    const Config& getConfig()      const { return m_config;       }
    Config& getMutableConfig()           { return m_config;       }
    const Mesh&   getToolMesh()    const { return m_toolMesh;     }
    double getCurrentTime()        const { return m_currentTime;  }
    int    getCurrentStep()        const { return m_currentStep;  }
    double getMaxStress()          const { return m_maxStress;    }
    double getMaxTemperature()     const { return m_maxTemperature; }
    double getElapsedWallTime()    const;

private:
    void loadGeometry();
    void initializeSolvers();
    void validateAlignment();
    void runMainLoop();
    void preview();
    void exportResults();
    void updateMetrics();
    double computeAdaptiveTimeStep() const;
    double computeAirGap() const;
    void handleAirCutting(double& dt);

    // Config & geometry
    Config m_config;
    Mesh   m_toolMesh;   // updated from FEM solver each output interval

    std::unique_ptr<GeometryLoader>    m_geometryLoader;
    std::unique_ptr<GCodeInterpreter>  m_gcodeInterpreter;

    // Machining strategy
    std::unique_ptr<IMachiningStrategy> m_strategy;

    // Solvers (owned)
    std::vector<std::unique_ptr<IPhysicsSolver>> m_solvers;

    // Exporters (owned)
    std::vector<std::unique_ptr<IExporter>> m_exporters;
    std::unique_ptr<AsyncExporter>          m_asyncExporter;

    // Non-owning pointers (lifetimes managed by main())
    ContactSolver*  m_contactSolver = nullptr;
    CFDSolverGPU*   m_cfdSolver    = nullptr;

    // Analytics & optimisation
    std::unique_ptr<OptimizationManager>      m_optimizationManager;
    std::unique_ptr<SurfaceRoughnessPredictor> m_roughnessPredictor;
    std::unique_ptr<ResidualStressAnalyzer>   m_stressAnalyzer;

    // Callbacks
    StepCallback       m_stepCallback;
    CompletionCallback m_completionCallback;

    // State
    bool   m_isInitialized = false;
    bool   m_isRunning     = false;
    bool   m_shouldStop    = false;
    bool   m_skipSPHThisStep = false;  // Set by handleAirCutting to skip SPH during approach
    double m_currentTime   = 0.0;
    int    m_currentStep   = 0;
    double m_maxStress     = 0.0;
    double m_maxTemperature = 0.0;

    std::chrono::high_resolution_clock::time_point m_startTime;
    
    // CNC coordinate system (shared from main for runtime WCS switching)
    CNCCoordinateSystem m_coordinateSystem;
};

// ---------------------------------------------------------------------------
// IExporter interface (defined here as it is tightly coupled to the engine)
// ---------------------------------------------------------------------------

class IExporter {
public:
    virtual ~IExporter() = default;

    virtual std::string getName() const = 0;
    virtual void exportStep(int step, double time, const Mesh& mesh) = 0;
    virtual void exportFinal(const Config& config, const Mesh& mesh) = 0;

    // Optional extensions
    virtual void exportParticles(int step, double time,
                                  const std::vector<SPHParticle>& particles) {}
    virtual void exportMetrics(int step, double time,
                                double maxStress, double maxTemp) {}
};

} // namespace edgepredict