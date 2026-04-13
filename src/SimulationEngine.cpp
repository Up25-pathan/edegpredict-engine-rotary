/**
 * @file SimulationEngine.cpp
 * @brief Simulation orchestrator implementation
 *
 * Fixes applied:
 *  - exportResults(): predict(output) not predict(output,config); use
 *    analyzeFromParticles() instead of non-existent analyze(config).
 *  - runMainLoop(): update m_toolMesh from FEM before each async export.
 *  - initializeSolvers(): guard JSON key access with contains() check.
 *  - exportResults(): initialise predictors, get SPH particles for analysis.
 */

#include "SimulationEngine.h"
#include "GeometryLoader.h"
#include "GCodeInterpreter.h"
#include "FEMSolver.cuh"
#include "SPHSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include "OptimizationManager.h"
#include "SurfaceRoughnessPredictor.h"
#include "ResidualStressAnalyzer.h"
#include "SurfaceReconstructor.h"
#include "VTKExporter.h"
#include "CudaUtils.cuh"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <chrono>

namespace edgepredict {

SimulationEngine::SimulationEngine()
    : m_geometryLoader(std::make_unique<GeometryLoader>()),
      m_gcodeInterpreter(std::make_unique<GCodeInterpreter>()),
      m_asyncExporter(std::make_unique<AsyncExporter>()) {
    m_startTime = std::chrono::high_resolution_clock::now();
}

SimulationEngine::~SimulationEngine() = default;

// ---------------------------------------------------------------------------
// Dependency Injection (Setters)
// ---------------------------------------------------------------------------

void SimulationEngine::setOptimizationManager(std::unique_ptr<OptimizationManager> manager) {
    m_optimizationManager = std::move(manager);
}

void SimulationEngine::setRoughnessPredictor(std::unique_ptr<SurfaceRoughnessPredictor> predictor) {
    m_roughnessPredictor = std::move(predictor);
}

void SimulationEngine::setStressAnalyzer(std::unique_ptr<ResidualStressAnalyzer> analyzer) {
    m_stressAnalyzer = std::move(analyzer);
}

// ---------------------------------------------------------------------------
// Internal: 3D Preview (T=0 Export)
// ---------------------------------------------------------------------------

void SimulationEngine::preview() {
    std::cout << "[Engine] Generating 3D Preview (T=0)..." << std::endl;
    
    updateMetrics();
    
    // Force FEM mesh export to internal Mesh structure
    for (const auto& solver : m_solvers) {
        if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
            fem->exportToMesh(m_toolMesh);
            break;
        }
    }
    
    // Get initial particle cloud
    std::vector<SPHParticle> initialParticles;
    for (const auto& solver : m_solvers) {
        if (auto* sph = dynamic_cast<SPHSolver*>(solver.get())) {
            initialParticles = sph->getParticles();
            break;
        }
    }
    
    // Export via all exporters (this creates mesh_000000.vtk, particles_000000.vtk, etc.)
    for (auto& exporter : m_exporters) {
        exporter->exportStep(0, 0.0, m_toolMesh);
        exporter->exportParticles(0, 0.0, initialParticles);
    }
    
    std::cout << "[Engine] Preview exported. Output directory: " 
              << m_config.getFilePaths().outputDirectory << std::endl;
}

// ---------------------------------------------------------------------------
// Internal: finalisation
// ---------------------------------------------------------------------------

bool SimulationEngine::initialize(const std::string& configPath) {
    try {
        m_config.loadFromFile(configPath);
        return initialize(m_config);
    } catch (const std::exception& e) {
        std::cerr << "[Engine] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool SimulationEngine::initialize(const Config& config) {
    m_config = config;

    if (!m_config.isValid()) {
        std::cerr << "[Engine] Invalid configuration" << std::endl;
        return false;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v4.0 (Clean Architecture)   " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Simulation: " << m_config.getSimulationName() << std::endl;

    m_isInitialized = true;
    return true;
}

// ---------------------------------------------------------------------------
// Dependency injection
// ---------------------------------------------------------------------------

void SimulationEngine::addSolver(std::unique_ptr<IPhysicsSolver> solver) {
    if (solver) {
        std::cout << "[Engine] Added solver: " << solver->getName() << std::endl;
        m_solvers.push_back(std::move(solver));
    }
}

void SimulationEngine::addExporter(std::unique_ptr<IExporter> exporter) {
    if (exporter) {
        std::cout << "[Engine] Added exporter: " << exporter->getName() << std::endl;
        m_exporters.push_back(std::move(exporter));
    }
}

void SimulationEngine::setStrategy(std::unique_ptr<IMachiningStrategy> strategy) {
    if (strategy) {
        std::cout << "[Engine] Set strategy: " << strategy->getName() << std::endl;
        m_strategy = std::move(strategy);
    }
}

void SimulationEngine::setContactSolver(ContactSolver* solver) {
    m_contactSolver = solver;
    if (solver) std::cout << "[Engine] Contact solver registered" << std::endl;
}

void SimulationEngine::setCFDSolver(CFDSolverGPU* solver) {
    m_cfdSolver = solver;
    if (solver) std::cout << "[Engine] CFD solver registered" << std::endl;
}

void SimulationEngine::setToolMesh(const Mesh& mesh) {
    m_toolMesh = mesh;
    std::cout << "[Engine] Registered tool mesh with " << m_toolMesh.nodes.size() 
              << " nodes and " << m_toolMesh.triangles.size() << " triangles" << std::endl;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

void SimulationEngine::run() {
    if (!m_isInitialized) {
        std::cerr << "[Engine] Not initialized!" << std::endl;
        if (m_completionCallback) m_completionCallback(false, "Not initialized");
        return;
    }

    m_startTime = std::chrono::high_resolution_clock::now();
    m_isRunning  = true;
    m_shouldStop = false;

    try {
        loadGeometry();
        initializeSolvers();
        validateAlignment();

        m_asyncExporter = std::make_unique<AsyncExporter>();
        m_asyncExporter->start();

        runMainLoop();

        if (m_asyncExporter) {
            m_asyncExporter->flush();
            m_asyncExporter->stop();
        }

        exportResults();

        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Simulation completed successfully!" << std::endl;
        std::cout << "Wall time: " << std::fixed << std::setprecision(2)
                  << getElapsedWallTime() << " seconds" << std::endl;
        std::cout << "==================================================" << std::endl;

        if (m_completionCallback) m_completionCallback(true, "Completed successfully");

    } catch (const std::exception& e) {
        std::cerr << "[Engine] FATAL ERROR: " << e.what() << std::endl;
        if (m_completionCallback) m_completionCallback(false, e.what());
    }

    m_isRunning = false;
}

// ---------------------------------------------------------------------------
// Single step (external control)
// ---------------------------------------------------------------------------

bool SimulationEngine::step(double dt) {
    if (!m_isInitialized || m_solvers.empty()) return false;

    NVTX_PUSH("SimulationEngine::step");

    // --- G-Code Driven Kinematics (Digital Twin Foundation) ---
    // The engine acts as the CNC controller, computing exact displacement
    // per timestep and applying it directly to the FEM mesh.
    MachineState state;
    state.spindleRPM = m_config.getMachining().rpm;
    state.feedRate   = m_config.getMachining().feedRateMmMin / 60000.0;
    
    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        auto prevSnap = m_gcodeInterpreter->getStateAtTime(m_currentTime);
        auto nextSnap = m_gcodeInterpreter->advanceByDt(dt);
        
        // Exact displacement this timestep (micron-level precision)
        Vec3 displacement = nextSnap.position - prevSnap.position;
        
        state.position   = nextSnap.position;
        state.spindleRPM = nextSnap.spindleRPM;
        state.feedRate   = nextSnap.feedRate;
        state.isActive   = true;
        
        // Flag rapids so strategies can skip force calculations
        state.motionMode = nextSnap.motionMode;
        state.isRapid    = (nextSnap.motionMode == 0);
        
        // Apply displacement directly to FEM tool mesh
        for (auto& solver : m_solvers) {
            FEMSolver* fem = dynamic_cast<FEMSolver*>(solver.get());
            if (!fem) continue;
            
            // Translate mesh by exact G-Code displacement
            if (displacement.lengthSq() > 1e-24) {
                fem->translateMesh(displacement.x, displacement.y, displacement.z);
            }
            
            // Apply rotation based on spindle RPM
            if (nextSnap.spindleRPM > 0 && dt > 0) {
                double angleRad = nextSnap.spindleRPM * 2.0 * 3.14159265358979323846 / 60.0 * dt;
                fem->rotateAroundZ(angleRad, nextSnap.position.x, nextSnap.position.y);
            }
            
            // Update spindle dynamics coupling
            Vec3 velocity = (dt > 1e-15) ? displacement / dt : Vec3::zero();
            fem->setVirtualSpindleState(nextSnap.position, velocity);
            break; // Only one FEM solver
        }
        
        // Sync WCS if G-Code switched it
        if (m_gcodeInterpreter->getActiveWCS() != m_coordinateSystem.getActiveWCS()) {
            m_coordinateSystem.setActiveWCS(m_gcodeInterpreter->getActiveWCS());
        }
    }
    
    // --- Strategy: forces, engagement, thermal coupling ---
    if (m_strategy) {
        m_strategy->updateConditions(state, dt);
        m_strategy->applyKinematics(dt);
    }

    // --- Physics solvers ---
    for (auto& solver : m_solvers) solver->step(dt);

    // === CFD Coupling: Feed SPH positions and FEM temperatures to coolant grid ===
    // This runs every step for maximum simulation accuracy (real-world fidelity).
    if (m_cfdSolver) {
        SPHSolver* sph = nullptr;
        FEMSolver* fem = nullptr;
        for (auto& solver : m_solvers) {
            if (!sph) sph = dynamic_cast<SPHSolver*>(solver.get());
            if (!fem) fem = dynamic_cast<FEMSolver*>(solver.get());
        }
        
        if (sph && sph->getParticleCount() > 0) {
            // Pack SPH positions for solid obstacle marking in CFD grid
            auto particles = sph->getParticles();
            std::vector<double> positions(particles.size() * 3);
            for (size_t i = 0; i < particles.size(); ++i) {
                positions[i*3+0] = particles[i].x;
                positions[i*3+1] = particles[i].y;
                positions[i*3+2] = particles[i].z;
            }
            m_cfdSolver->setSolidObstacles(positions.data(), 
                                            static_cast<int>(particles.size()));
        }
        
        if (fem && fem->getNodeCount() > 0) {
            // Feed FEM node temperatures and POSITIONS as heat sources to coolant
            auto nodes = fem->getNodes();
            std::vector<double> temps(nodes.size());
            std::vector<double> pos(nodes.size() * 3);
            for (size_t i = 0; i < nodes.size(); ++i) {
                temps[i] = nodes[i].temperature;
                pos[i*3+0] = nodes[i].x;
                pos[i*3+1] = nodes[i].y;
                pos[i*3+2] = nodes[i].z;
            }
            m_cfdSolver->setHeatSources(temps.data(), pos.data(),
                                         static_cast<int>(nodes.size()));
        }
    }

    if (m_contactSolver) m_contactSolver->resolveContacts(dt);

    // --- Adaptive optimisation (every 50 steps) ---
    if (m_optimizationManager && m_strategy && (m_currentStep % 50 == 0)) {
        ProcessState pState;
        pState.time            = m_currentTime;
        pState.toolStress      = m_maxStress;
        pState.toolTemperature = m_maxTemperature;

        auto output = m_strategy->computeOutput();
        pState.cuttingForce = std::sqrt(
            output.cuttingForce.x * output.cuttingForce.x +
            output.cuttingForce.y * output.cuttingForce.y +
            output.cuttingForce.z * output.cuttingForce.z);
        pState.materialRemovalRate = output.materialRemovalRate;

        auto cmd = m_optimizationManager->update(pState, dt * 50.0);
        m_strategy->applyAdaptiveControl(cmd.feedRateMultiplier,
                                         cmd.spindleSpeedMultiplier);
    }

    m_currentTime += dt;
    m_currentStep++;

    NVTX_POP();
    return true;
}

void SimulationEngine::stop() { m_shouldStop = true; }

double SimulationEngine::getElapsedWallTime() const {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - m_startTime;
    return elapsed.count();
}

// ---------------------------------------------------------------------------
// Internal: load geometry
// ---------------------------------------------------------------------------

void SimulationEngine::loadGeometry() {
    // Skip if solvers already have geometry (pre-initialised in main.cpp)
    for (const auto& solver : m_solvers) {
        if (solver->getNodeCount() > 0 || solver->getParticleCount() > 0) {
            std::cout << "[Engine] Solvers already have geometry – skipping loadGeometry()" << std::endl;
            return;
        }
    }

    std::cout << "[Engine] Loading geometry..." << std::endl;

    const auto& fp = m_config.getFilePaths();
    if (fp.toolGeometry.empty()) {
        std::cout << "[Engine] No tool geometry specified – using solver-generated defaults" << std::endl;
        return;
    }
    if (!std::filesystem::exists(fp.toolGeometry)) {
        std::cerr << "[Engine] WARNING: Tool geometry file not found: " << fp.toolGeometry << std::endl;
        return;
    }
    std::cout << "[Engine] Tool geometry: " << fp.toolGeometry << std::endl;
}

// ---------------------------------------------------------------------------
// Internal: initialise solvers
// ---------------------------------------------------------------------------

void SimulationEngine::initializeSolvers() {
    std::cout << "[Engine] Initializing " << m_solvers.size() << " solver(s)..." << std::endl;

    for (auto& solver : m_solvers) {
        if (solver->getNodeCount() > 0 || solver->getParticleCount() > 0) {
            std::cout << "  - " << solver->getName()
                      << " (already initialized, skipping)" << std::endl;
            continue;
        }
        std::cout << "  - " << solver->getName() << "..." << std::endl;
        if (!solver->initialize(m_config))
            throw std::runtime_error("Failed to initialize solver: " + solver->getName());
    }

    // Strategy
    if (m_strategy) {
        if (!m_strategy->isInitialized()) {
            std::cout << "  - " << m_strategy->getName() << " (strategy)..." << std::endl;
            if (!m_strategy->initialize(m_config))
                throw std::runtime_error("Failed to initialize strategy: " + m_strategy->getName());
        } else {
            std::cout << "  - " << m_strategy->getName()
                      << " (already connected, skipping)" << std::endl;
        }
    }

    // Apply initial tool offset from config (with safe JSON access)
    const auto& j = m_config.getJson();
    if (j.contains("machining_parameters") &&
        j["machining_parameters"].contains("initial_tool_position_mm")) {

        auto initPos = j["machining_parameters"]["initial_tool_position_mm"]
                           .get<std::vector<double>>();
        if (initPos.size() >= 3) {
            double offX = initPos[0] / 1000.0;
            double offY = initPos[1] / 1000.0;
            double offZ = initPos[2] / 1000.0;

            if (std::abs(offX) > 1e-9 || std::abs(offY) > 1e-9 || std::abs(offZ) > 1e-9) {
                std::cout << "[Engine] Applying initial tool offset: ("
                          << offX << ", " << offY << ", " << offZ << ") m" << std::endl;
                for (auto& solver : m_solvers) {
                    if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                        fem->translateMesh(offX, offY, offZ);
                    }
                }
            }
        }
    }

    std::cout << "[Engine] All solvers initialized" << std::endl;
}

// ---------------------------------------------------------------------------
// Internal: pre-flight geometry validation (Anchored Physics)
// ---------------------------------------------------------------------------

void SimulationEngine::validateAlignment() {
    if (m_solvers.size() < 2) return;
    
    SPHSolver* sph = nullptr;
    FEMSolver* fem = nullptr;
    for (auto& solver : m_solvers) {
        if (!sph) sph = dynamic_cast<SPHSolver*>(solver.get());
        if (!fem) fem = dynamic_cast<FEMSolver*>(solver.get());
    }
    if (!sph || !fem) return;
    
    double sMinX, sMinY, sMinZ, sMaxX, sMaxY, sMaxZ;
    double tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ;
    sph->getBounds(sMinX, sMinY, sMinZ, sMaxX, sMaxY, sMaxZ);
    fem->getBounds(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ);
    
    std::cout << "[Engine] Pre-flight - SPH Bounds: X[" << sMinX << ", " << sMaxX << "] "
              << "Y[" << sMinY << ", " << sMaxY << "] "
              << "Z[" << sMinZ << ", " << sMaxZ << "]" << std::endl;
    std::cout << "[Engine] Pre-flight - FEM Bounds: X[" << tMinX << ", " << tMaxX << "] "
              << "Y[" << tMinY << ", " << tMaxY << "] "
              << "Z[" << tMinZ << ", " << tMaxZ << "]" << std::endl;
    
    // Check 1: Full geometric overlap = tool deeply embedded in workpiece
    bool xOverlap = (tMinX < sMaxX) && (tMaxX > sMinX);
    bool yOverlap = (tMinY < sMaxY) && (tMaxY > sMinY);
    bool zOverlap = (tMinZ < sMaxZ) && (tMaxZ > sMinZ);
    
    if (xOverlap && yOverlap && zOverlap) {
        double penX = std::min(tMaxX, sMaxX) - std::max(tMinX, sMinX);
        double penY = std::min(tMaxY, sMaxY) - std::max(tMinY, sMinY);
        double penZ = std::min(tMaxZ, sMaxZ) - std::max(tMinZ, sMinZ);
        double overlapVol = std::max(0.0, penX) * std::max(0.0, penY) * std::max(0.0, penZ);
        
        double toolVol = (tMaxX-tMinX) * (tMaxY-tMinY) * (tMaxZ-tMinZ);
        
        if (toolVol > 1e-18 && overlapVol > toolVol * 0.5) {
            throw std::runtime_error(
                "[FATAL] Kinematic Setup Invalid: Tool deeply embedded in workpiece at T=0. "
                "Overlap volume: " + std::to_string(overlapVol * 1e9) + " mm^3. "
                "Check G54 offset, tool length offset, and G-Code starting position.");
        }
        
        double minPen = std::min({penX, penY, penZ});
        std::cout << "[Engine] Pre-flight: Tool near workpiece (penetration: "
                  << minPen * 1000 << " mm) — OK for cutting start" << std::endl;
    }
    
    // Check 2: Tool unreachably far from workpiece
    double gap = computeAirGap();
    if (gap > 0.1) {
        std::cerr << "[Engine] WARNING: Tool is " << gap * 1000 
                  << " mm from workpiece. Excessive air-cutting expected." << std::endl;
    }
    
    std::cout << "[Engine] Pre-flight alignment validation PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Internal: main simulation loop
// ---------------------------------------------------------------------------

void SimulationEngine::runMainLoop() {
    const auto& simParams  = m_config.getSimulation();
    const int   totalSteps = simParams.numSteps;
    const int   outInterval = simParams.outputIntervalSteps;

    std::cout << "[Engine] Starting simulation (" << totalSteps << " steps)..." << std::endl;

    m_currentStep       = 0;
    m_currentTime       = 0.0;
    m_maxTemperature    = m_config.getMachining().ambientTemperature;

    // G-Code
    const std::string gcodePath = m_config.getFilePaths().gcodeFile;
    if (!gcodePath.empty()) {
        m_gcodeInterpreter = std::make_unique<GCodeInterpreter>();
        if (m_gcodeInterpreter->loadFile(gcodePath)) {
            std::cout << "[Engine] Using G-Code for toolpath control" << std::endl;

            // ----------------------------------------------------------------
            // TIME BUDGET CHECK: make sure the configured num_steps * max_dt
            // is sufficient to cover a meaningful portion of the toolpath.
            // ----------------------------------------------------------------
            double gcDuration = m_gcodeInterpreter->getTotalDuration();
            double maxDt      = m_config.getSimulation().maxTimeStep;
            double physWindow = static_cast<double>(totalSteps) * maxDt;

            std::cout << "  Segments: " << m_gcodeInterpreter->getSegmentCount() << std::endl;
            std::cout << "  Duration: " << gcDuration << " seconds" << std::endl;
            std::cout << "  Physics window: " << physWindow << " s  ("
                      << std::fixed << std::setprecision(1)
                      << (physWindow / gcDuration * 100.0) << "% of toolpath)" << std::endl;

            if (physWindow < gcDuration * 0.05) {
                std::cerr << "\n[Engine] CRITICAL: Physics window (" << physWindow
                          << " s) covers only "
                          << (physWindow / gcDuration * 100.0) << "% of the G-Code path ("
                          << gcDuration << " s).\n"
                          << "  The tool will NEVER reach the workpiece in this run.\n"
                          << "  FIX: Increase num_steps or max_time_step_s in your JSON config.\n"
                          << "  Recommended: num_steps=" << static_cast<int>(gcDuration / maxDt) + 1000
                          << " to cover full toolpath.\n" << std::endl;
            } else if (physWindow < gcDuration * 0.2) {
                std::cerr << "[Engine] WARNING: Physics window covers only "
                          << (physWindow / gcDuration * 100.0) << "% of G-Code path. "
                          << "Consider increasing num_steps for full engagement." << std::endl;
            }
        } else {
            std::cerr << "[Engine] Failed to load G-Code: "
                      << m_gcodeInterpreter->getLastError() << std::endl;
        }
    }

    // ====================================================================
    // NEW: INITIAL KINEMATIC SYNC (Fixes the Rubber-Band Effect)
    // ====================================================================
    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        // Get the very first position from the G-Code (e.g., Z=50 clearance)
        auto startState = m_gcodeInterpreter->getStateAtTime(0.0);
        Vec3 startPos = startState.position;
        
        // If the G-code doesn't start exactly at 0,0,0, move the mesh there
        if (startPos.lengthSq() > 1e-12) {
            std::cout << "[Engine] Syncing tool mesh to G-Code start position: (" 
                      << startPos.x*1000 << ", " << startPos.y*1000 << ", " 
                      << startPos.z*1000 << ") mm" << std::endl;
                      
            for (auto& solver : m_solvers) {
                if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                    // Physically move the CAD mesh to the starting clearance plane
                    fem->translateMesh(startPos.x, startPos.y, startPos.z);
                    // Tell the spindle dynamics that we are comfortably resting here
                    fem->setVirtualSpindleState(startPos, Vec3::zero());
                }
            }
        }
    }
    // ====================================================================

    // === DIGITAL TWIN: Initial 3D Preview (Step 0) ===
    preview();
    
    if (simParams.previewSetup) {
        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "[Engine] PREVIEW COMPLETE. Inspect step 0 files." << std::endl;
        std::cout << "Proceeding to simulation in 3 seconds..." << std::endl;
        std::cout << "----------------------------------------------------------\n" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    while (m_currentStep < totalSteps && !m_shouldStop) {
        double dt = computeAdaptiveTimeStep();
        handleAirCutting(dt);
        step(dt);

        if (m_currentStep % outInterval == 0 || m_currentStep == 1) {
            updateMetrics();

            double percent = (100.0 * m_currentStep) / totalSteps;
            std::cout << "  Step " << std::setw(7) << m_currentStep
                      << " | " << std::fixed << std::setprecision(1) << percent << "%"
                      << " | dt: " << std::scientific << std::setprecision(2) << dt
                      << " | Stress: " << std::scientific
                      << std::setprecision(2) << m_maxStress / 1e6 << " MPa"
                      << " | Temp: " << std::fixed << std::setprecision(1)
                      << m_maxTemperature << " C"
                      << std::endl;

            // ----------------------------------------------------------------
            // FIX: update m_toolMesh from the FEM solver before capturing it
            // ----------------------------------------------------------------
            for (const auto& solver : m_solvers) {
                if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                    fem->exportToMesh(m_toolMesh);
                    break;
                }
            }

            // Collect SPH particles for live export
            std::vector<SPHParticle> capturedParticles;
            for (const auto& solver : m_solvers) {
                if (auto* sph = dynamic_cast<SPHSolver*>(solver.get())) {
                    capturedParticles = sph->getParticles();
                    break;
                }
            }

            // Async export (non-blocking)
            for (auto& exporter : m_exporters) {
                int                      capturedStep    = m_currentStep;
                double                   capturedTime    = m_currentTime;
                double                   capturedStress  = m_maxStress;
                double                   capturedTemp    = m_maxTemperature;
                Mesh                     capturedMesh    = m_toolMesh;   // now has FEM data
                std::vector<SPHParticle> capturedPts     = capturedParticles;
                IExporter*               exporterPtr     = exporter.get();

                m_asyncExporter->enqueue(
                    [exporterPtr, capturedStep, capturedTime,
                     capturedMesh, capturedPts,
                     capturedStress, capturedTemp]() {
                        exporterPtr->exportStep(capturedStep, capturedTime, capturedMesh);
                        exporterPtr->exportMetrics(capturedStep, capturedTime,
                                                   capturedStress, capturedTemp);
                        if (!capturedPts.empty()) {
                            exporterPtr->exportParticles(capturedStep, capturedTime, capturedPts);
                        }
                    },
                    capturedStep, capturedTime);
            }

            if (m_stepCallback)
                m_stepCallback(m_currentStep, m_currentTime, m_maxStress, m_maxTemperature);
        }
    }

    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Internal: export final results
// FIX: removed non-existent predict(output, config) and analyze(config) calls.
//      Use predict(output) and analyzeFromParticles() with particles from SPH.
// ---------------------------------------------------------------------------

void SimulationEngine::exportResults() {
    std::cout << "[Engine] Exporting final results..." << std::endl;

    // --- Update tool mesh one last time ---
    for (const auto& solver : m_solvers) {
        if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
            fem->exportToMesh(m_toolMesh);
            break;
        }
    }

    // --- Collect final particle data ---
    std::vector<SPHParticle> finalParticles;
    for (const auto& solver : m_solvers) {
        if (auto* sph = dynamic_cast<SPHSolver*>(solver.get())) {
            finalParticles = sph->getParticles();
            break;
        }
    }

    // --- Analytics ---
    if (m_strategy) {
        auto output = m_strategy->computeOutput();

        // Surface roughness predictor
        if (m_roughnessPredictor) {
            // Initialise with config before first use
            m_roughnessPredictor->initialize(m_config);

            // FIX: predict() takes a single MachiningOutput argument
            auto roughness = m_roughnessPredictor->predict(output);
            std::cout << "  [Roughness] Ra = " << std::fixed << std::setprecision(3)
                      << roughness.Ra << " μm"
                      << (roughness.meetsTolerance ? "  [PASS]" : "  [FAIL]")
                      << std::endl;
        }

        // Residual stress analyzer
        // FIX: use analyzeFromParticles() (the method that actually exists)
        if (m_stressAnalyzer && !finalParticles.empty()) {
            m_stressAnalyzer->configure(0.001 /*maxDepth m*/, 50 /*levels*/);

            // Default surface normal = +Y (machined top face)
            Vec3 surfaceNormal(0.0, 1.0, 0.0);
            Vec3 surfacePoint(0.0, 0.0, 0.0);

            // Try to find the actual top surface from SPH bounds
            double minX, minY, minZ, maxX, maxY, maxZ;
            m_solvers.front()->getBounds(minX, minY, minZ, maxX, maxY, maxZ);
            surfacePoint = Vec3((minX + maxX) * 0.5, maxY, (minZ + maxZ) * 0.5);

            m_stressAnalyzer->analyzeFromParticles(finalParticles,
                                                    surfaceNormal, surfacePoint);
            std::cout << m_stressAnalyzer->getSummary();

            const std::string csvPath =
                m_config.getFilePaths().outputDirectory + "/residual_stress_profile.csv";
            m_stressAnalyzer->exportToCSV(csvPath);
        }
    }

    // --- Surface Reconstruction ---
    if (!finalParticles.empty()) {
        std::cout << "[Engine] Reconstructing machined surface..." << std::endl;
        ReconstructionParams reconParams;
        reconParams.smoothingRadius = m_config.getSPH().smoothingRadius;
        reconParams.cellSize = reconParams.smoothingRadius * 3.0; // 300μm default
        
        Mesh reconstructedMesh = SurfaceReconstructor::reconstruct(finalParticles, reconParams);
        
        for (auto& exporter : m_exporters) {
            if (auto* vtk = dynamic_cast<VTKExporter*>(exporter.get())) {
                vtk->exportReconstructedWorkpiece(reconstructedMesh);
            }
        }
    }

    // --- Create output directory ---
    const auto& outputDir = m_config.getFilePaths().outputDirectory;
    std::filesystem::create_directories(outputDir);

    // --- Final export from all exporters ---
    for (auto& exporter : m_exporters) {
        std::cout << "  - " << exporter->getName() << "..." << std::endl;
        exporter->exportFinal(m_config, m_toolMesh);
    }

    std::cout << "[Engine] Results saved to: " << outputDir << std::endl;
}

// ---------------------------------------------------------------------------
// Internal: metrics
// ---------------------------------------------------------------------------

void SimulationEngine::updateMetrics() {
    for (const auto& solver : m_solvers) {
        auto* metrics = dynamic_cast<IMetricsProvider*>(solver.get());
        if (metrics) {
            metrics->syncMetrics();
            m_maxStress      = std::max(m_maxStress,      metrics->getMaxStress());
            m_maxTemperature = std::max(m_maxTemperature, metrics->getMaxTemperature());
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: adaptive time-step
// ---------------------------------------------------------------------------

double SimulationEngine::computeAdaptiveTimeStep() const {
    const auto& simParams = m_config.getSimulation();
    double dt = simParams.timeStepDuration;

    for (const auto& solver : m_solvers) {
        double solverDt = solver->getStableTimeStep();
        if (solverDt > 0) dt = std::min(dt, solverDt);
    }

    return std::max(simParams.minTimeStep, std::min(simParams.maxTimeStep, dt));
}

// ---------------------------------------------------------------------------
// Internal: air-gap detection
// ---------------------------------------------------------------------------

double SimulationEngine::computeAirGap() const {
    if (m_solvers.size() < 2) return 0.0;

    double minX1, minY1, minZ1, maxX1, maxY1, maxZ1;
    double minX2, minY2, minZ2, maxX2, maxY2, maxZ2;
    m_solvers[0]->getBounds(minX1, minY1, minZ1, maxX1, maxY1, maxZ1);
    m_solvers[1]->getBounds(minX2, minY2, minZ2, maxX2, maxY2, maxZ2);

    double dx = std::max(0.0, std::max(minX1 - maxX2, minX2 - maxX1));
    double dy = std::max(0.0, std::max(minY1 - maxY2, minY2 - maxY1));
    double dz = std::max(0.0, std::max(minZ1 - maxZ2, minZ2 - maxZ1));
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void SimulationEngine::handleAirCutting(double& dt) {
    if (!m_contactSolver || m_contactSolver->getContactCount() > 0) return;

    // When G-Code is active, the toolpath timing is already defined by the
    // interpreter. The rapid G00 already moves at max traverse rate (10 m/s).
    // Applying an ADDITIONAL dt boost causes the tool to teleport by 0.9mm in
    // a single step — the entire clearance gap — embedding it into the surface
    // and generating 224+ contacts and 74kN thermal runaway in one step.
    // DO NOT boost dt when G-Code controls the kinematics.
    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        // Still log the air-cutting status so the user can see approach progress
        double gap = computeAirGap();
        if (gap > 0.0005) {
            static double lastSkipLog = -1.0;
            if (m_currentTime - lastSkipLog > 0.05) {
                std::cout << "[Engine] Air-cutting (gap: " << gap * 1000.0
                          << " mm @ T=" << std::fixed << std::setprecision(4)
                          << m_currentTime << " s)" << std::endl;
                lastSkipLog = m_currentTime;
            }
        }
        return;  // No dt override — G-Code controls the approach rate
    }

    // Legacy path: no G-Code loaded, manual kinematics only.
    // Safe to boost dt since there is no interpolated trajectory to maintain.
    double gap = computeAirGap();
    if (gap > 0.0005) {
        static double lastSkipLog = -1.0;
        if (m_currentTime - lastSkipLog > 0.1) {
            std::cout << "[Engine] Air-cutting (gap: " << gap * 1000.0
                      << " mm). Accelerating..." << std::endl;
            lastSkipLog = m_currentTime;
        }
        double baseDt = computeAdaptiveTimeStep();
        dt = std::min(baseDt * 5.0, gap / 10.0);
    }
}

} // namespace edgepredict