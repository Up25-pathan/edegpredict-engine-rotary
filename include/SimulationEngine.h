#pragma once
/**
 * @file SimulationEngine.h
 * @brief Main simulation orchestrator
 *
 * Kept intentionally slim — all actual physics is delegated to solvers.
 * Fix: added forward declarations for MPMSolver and FEMSolver so that
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
#include <mutex>
#include "AsyncExporter.h"
#include "ResidualStressAnalyzer.h"
#include "SimulationHealthMonitor.h"

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
class MPMSolver;
class FEMSolver;
struct MPMParticle;
class EdgeSubgridModel;
class BUEModel;
class CoolantHardeningModel;
class ChatterDynamics;

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

    // Advanced tool models (forward to strategy if already registered)
    void setEdgeSubgridModel(EdgeSubgridModel* model);
    void setBUEModel(BUEModel* model);
    void setCoolantHardeningModel(CoolantHardeningModel* model) { m_coolantHardening = model; }
    void setChatterDynamics(ChatterDynamics* chatter);

    EdgeSubgridModel* getEdgeSubgridModel() const { return m_edgeSubgrid; }
    BUEModel* getBUEModel() const { return m_bueModel; }
    CoolantHardeningModel* getCoolantHardeningModel() const { return m_coolantHardening; }
    ChatterDynamics* getChatterDynamics() const { return m_chatterDynamics; }

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
    double getMaxWorkpieceStress() const { return m_workpieceMaxStress; }
    double getMaxWorkpieceTemperature() const { return m_workpieceMaxTemperature; }
    double getMaxToolStress() const { return m_toolMaxStress; }
    double getMaxToolTemperature() const { return m_toolMaxTemperature; }
    double getMaxCoolantTemperature() const { return m_coolantMaxTemperature; }
    double getMaxToolWear() const { return m_toolMaxWear; }
    double getElapsedWallTime()    const;

private:
    void loadGeometry();
    void initializeSolvers();
    void validateAlignment();
    void runMainLoop();
    void preview();
    void exportResults();
    void updateMetrics();
    double computeAdaptiveTimeStep();
    int computeSubStepCount(double macroDt) const;
    double computeAirGap() const;
    double getCurrentCutDepthMm() const;
    bool hasReachedTargetDepth() const;
    void handleAirCutting(double& dt);
    void recordContactEngagement(double dt);
    void cacheSolverPointers();
    bool shouldStop() const;
    void setShouldStop(bool v);

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
    mutable MPMSolver* m_mpmSolver = nullptr;   // cached from m_solvers
    mutable FEMSolver* m_femSolver = nullptr;   // cached from m_solvers
    EdgeSubgridModel* m_edgeSubgrid = nullptr;
    BUEModel*        m_bueModel    = nullptr;
    CoolantHardeningModel* m_coolantHardening = nullptr;
    ChatterDynamics* m_chatterDynamics = nullptr;

    // Coolant delivery state, driven by the G-code program (M8 = ON, M9 = OFF).
    // When no program is loaded, falls back to the coolant model's enable flag.
    bool m_coolantActive = true;

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
    mutable std::mutex m_shouldStopMutex;  // guards m_shouldStop across threads
    bool   m_skipPhysicsThisStep = false;  // Set by handleAirCutting to skip ALL solvers during approach
    bool   m_contactEngaged = false;       // Latched once tool/workpiece contact begins
    double m_contactStartTime = -1.0;
    double m_contactStartToolZ = 0.0;
    double m_currentTime   = 0.0;
    int    m_currentStep   = 0;
    double m_maxStress     = 0.0;
    double m_maxTemperature = 0.0;
    double m_workpieceMaxStress = 0.0;
    double m_workpieceMaxTemperature = 0.0;
    double m_toolMaxStress = 0.0;
    double m_toolMaxTemperature = 0.0;
    double m_coolantMaxTemperature = 0.0;
    double m_toolMaxWear = 0.0;
    double m_toolCurrentStress = 0.0;
    double m_workpieceCurrentStress = 0.0;
    double m_toolCurrentTemperature = 0.0;
    double m_workpieceCurrentTemperature = 0.0;
    std::string m_dtLimiterName;

    // Adaptive Mass Scaling (AMS) state
    double m_amsMassScaling = 1.0;
    double m_amsKEIERatio = 0.0;       // Kinetic/Internal energy ratio (monitor)
    int    m_amsSubStepCounter = 0;     // Current micro-step within macro-step
    double m_amsMacroDt = 0.0;          // Engine-level timestep (may be > MPM CFL)
    double m_amsMicroDt = 0.0;          // MPM sub-cycling timestep

    // Frequency-domain AMS tracking
    double m_amsHighestFrequency = 0.0;  // Estimated highest eigenfrequency (Hz)
    double m_amsSpectralRadius = 0.0;    // Estimated spectral radius (CFL-based, NOT KE ratio)
    double m_amsAdaptiveCFL = 0.2;       // Current adaptive CFL number

    // ── AMS PID Controller ──────────────────────────────────────────────────
    // Replaces the oscillatory multiplicative 0.8/1.05 CFL control with a
    // proper PID controller targeting KE/IE < 5%. Anti-windup clamped integral.
    struct AMSPIDController {
        double kp = 0.5, ki = 0.05, kd = 0.1;
        double integral = 0.0;
        double prev_error = 0.0;
        double target = 0.05;  // KE/IE < 5%

        void configure(double p, double i, double d, double t) {
            kp = p; ki = i; kd = d; target = t;
            integral = 0.0; prev_error = 0.0;
        }

        double update(double keie) {
            double error = target - keie;
            integral = std::clamp(integral + error, -2.0, 2.0); // anti-windup
            double deriv = error - prev_error;
            prev_error = error;
            return kp * error + ki * integral + kd * deriv;
        }

        void reset() { integral = 0.0; prev_error = 0.0; }
    };
    AMSPIDController m_amsPID;

    // ── Multi-rate time integration accumulators (5-tier hierarchy) ──────────
    // P2 FIX: Track physical simulation time for multi-rate coupling, not adaptive dt
    double m_thermalAccumulator = 0.0;   // Tier 3: thermal diffusion (accumulates physical time)
    double m_cfdAccumulator = 0.0;       // Tier 4: CFD coolant flow (accumulates physical time)
    double m_wearAccumulator = 0.0;      // Tier 5: wear/BUE/chatter (accumulates physical time)
    double m_lastThermalCouplingTime = 0.0;  // Last simulation time thermal coupling ran
    double m_lastCfdCouplingTime = 0.0;      // Last simulation time CFD coupling ran

    // ── CoolantHardening D2H batch cache ────────────────────────────────────
    // Avoids 10 MB/step host copies of FEM nodes; only fetches every N steps.
    int m_coolantBatchInterval = 100;    // Only fetch FEM nodes every N steps
    int m_coolantLastBatchStep = -1;     // Last step when nodes were fetched

    // Tool rotation tracking for SDF transform
    double m_currentToolAngle = 0.0;  // cumulative Z-rotation (rad)

    // Dynamic tool wear morphing (SDF rebuild interval)
    int m_wearMorphStepInterval = 5000;  // Increased from 500: wear changes are gradual
    int m_lastWearMorphStep = 0;

    // Interpolated tool motion for sub-cycling (GPU buffer)
    FEMNodeGPU* d_toolStartSnapshot = nullptr;  // device buffer: FEM node positions at start of macro-step

    // Air-cutting log throttle (replaces static locals in handleAirCutting)
    double m_airCutLogTime_gcode = -1.0;
    double m_airCutLogTime_legacy = -1.0;

    // Final-results export latch: exportResults() may fire early from the
    // health-halt path inside the main loop; run() must not export twice.
    bool   m_finalResultsExported = false;

    // Steady-State Extrapolation Module
    static constexpr int STRESS_HISTORY_CAPACITY = 1000;
    std::array<double, STRESS_HISTORY_CAPACITY> m_stressHistory{};
    int m_stressHistoryIdx = 0;
    int m_stressHistoryCount = 0;   // entries written (for mean/stddev before full)
    bool m_steadyStateReached = false;
    bool m_extrapolationFired = false;

    // Thermal coupling convergence tracking
    int    m_lastThermalIters = 0;         // iterations taken in last thermal loop
    double m_lastThermalResidual = 0.0;    // max |ΔT| across all nodes/particles (°C)
    bool   m_thermalConverged = false;     // did the last thermal loop converge?

    std::chrono::high_resolution_clock::time_point m_startTime;

    // Simulation Health Monitor — detects non-physical states and halts
    SimulationHealthMonitor m_healthMonitor;
    
    // Two-way kinematic coupling: resultant cutting force from strategy
    Vec3 m_cuttingForceResultant;

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
                                  const std::vector<MPMParticle>& particles) {}
    virtual void exportMetrics(int step, double time,
                                double maxStress, double maxTemp) {}
    virtual void exportDetailedMetrics(int step, double time,
                                       double workpieceStress, double workpieceTemp,
                                       double toolStress, double toolTemp,
                                       double coolantTemp) {
        double maxStress = workpieceStress > toolStress ? workpieceStress : toolStress;
        double maxTemp = workpieceTemp > toolTemp ? workpieceTemp : toolTemp;
        if (coolantTemp > maxTemp) maxTemp = coolantTemp;
        exportMetrics(step, time, maxStress, maxTemp);
    }

    virtual void exportReconstructedWorkpiece(int step, double time,
                                              const Mesh& mesh) {
        (void)step;
        (void)time;
        (void)mesh;
    }
};

} // namespace edgepredict
