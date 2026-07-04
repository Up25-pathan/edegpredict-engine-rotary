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
#include "EdgeSubgridModel.h"
#include "BUEModel.h"
#include "CoolantHardeningModel.h"
#include "MaterialModels.h"
#include "ChatterDynamics.h"
#include "FEMSolver.cuh"
#include "MPMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include "OptimizationManager.h"
#include "SurfaceRoughnessPredictor.h"
#include "ResidualStressAnalyzer.h"
#include "SurfaceReconstructor.h"
#include "ToolRegionClassifier.h"
#include "ToolRNDAnalyzer.h"
#include "VTKExporter.h"
#include "CudaUtils.cuh"
#include "Config.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <filesystem>
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

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
    if (m_contactSolver) m_contactSolver->applyCoatingWearToMesh(m_toolMesh);
    ToolRegionClassifier::annotateRiskFields(m_toolMesh, m_config);
    
    // Get initial particle cloud
    std::vector<MPMParticle> initialParticles;
    for (const auto& solver : m_solvers) {
        if (auto* sph = dynamic_cast<MPMSolver*>(solver.get())) {
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
        strategy->setEdgeSubgridModel(m_edgeSubgrid);
        strategy->setBUEModel(m_bueModel);
        strategy->setChatterDynamics(m_chatterDynamics);
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

void SimulationEngine::setEdgeSubgridModel(EdgeSubgridModel* model) {
    m_edgeSubgrid = model;
    if (m_strategy) m_strategy->setEdgeSubgridModel(model);
    if (model) std::cout << "[Engine] Edge subgrid model registered" << std::endl;
}

void SimulationEngine::setBUEModel(BUEModel* model) {
    m_bueModel = model;
    if (m_strategy) m_strategy->setBUEModel(model);
    if (model) std::cout << "[Engine] BUE model registered" << std::endl;
}

void SimulationEngine::setChatterDynamics(ChatterDynamics* chatter) {
    m_chatterDynamics = chatter;
    if (m_strategy) m_strategy->setChatterDynamics(chatter);
    if (chatter) std::cout << "[Engine] Chatter dynamics registered" << std::endl;
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

    // Reset air-cutting log throttles (Fix 4: prevents stale state across runs)
    m_airCutLogTime_gcode = -1.0;
    m_airCutLogTime_legacy = -1.0;

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
        // === FIX: Single-clock kinematics (eliminates dual-clock drift) ===
        // Use m_currentTime as the ONLY canonical clock. No advanceByDt().
        auto prevSnap = m_gcodeInterpreter->getStateAtTime(m_currentTime);
        auto nextSnap = m_gcodeInterpreter->getStateAtTime(m_currentTime + dt);
        
        // Exact displacement this timestep (micron-level precision)
        Vec3 displacement = nextSnap.position - prevSnap.position;
        if (displacement.length() > 0.001) {
            std::cout << "[DBUG] LARGE DISPLACEMENT: " << displacement.length()*1000.0
                      << "mm at T=" << m_currentTime << " dt=" << dt << std::endl;
        }
        
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
            double angularVelocity = nextSnap.spindleRPM * 2.0 * 3.14159265358979323846 / 60.0;
            if (nextSnap.spindleRPM > 0 && dt > 0) {
                double angleRad = angularVelocity * dt;
                fem->rotateAroundZ(angleRad, nextSnap.position.x, nextSnap.position.y);
            }
            
            // Update spindle dynamics coupling
            Vec3 velocity = (dt > 1e-15) ? displacement / dt : Vec3::zero();
            fem->setVirtualSpindleState(nextSnap.position, velocity);
            
            // Set full rigid-body tool velocity: feed plus spindle tangential
            // velocity. Contact heat/friction and MPM collision both depend on
            // local surface speed, not just axial feed speed.
            fem->setRigidBodyNodeVelocities(
                velocity, angularVelocity, nextSnap.position.x, nextSnap.position.y);
            break; // Only one FEM solver
        }
        
        // Update MPM solver tool position for LOD classification
        for (auto& solver : m_solvers) {
            if (auto* mpm = dynamic_cast<MPMSolver*>(solver.get())) {
                mpm->setToolPosition(state.position);
                break;
            }
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

        // Compute output to get forces for advanced models
        MachiningOutput initialOutput;
        if (m_chatterDynamics || m_bueModel) {
            initialOutput = m_strategy->computeOutput();
        }

        // Drive chatter dynamics time step (regenerative displacement)
        if (m_chatterDynamics && m_chatterDynamics->isEnabled()) {
            double cuttingForceMag = std::sqrt(
                initialOutput.cuttingForce.x * initialOutput.cuttingForce.x +
                initialOutput.cuttingForce.y * initialOutput.cuttingForce.y +
                initialOutput.cuttingForce.z * initialOutput.cuttingForce.z);
            double rpm = m_config.getMachining().rpm;
            double doc = m_config.getMachining().depthOfCutMm / 1000.0;
            double woc = doc * 0.5;
            double fpt = initialOutput.chipThickness;
            m_chatterDynamics->updateDynamics(
                dt, rpm, doc, woc, fpt,
                cuttingForceMag, Vec3(1, 0, 0), Vec3(0, 0, 1));
        }

        // Apply BUE model (nucleation/growth at cutting interface)
        if (m_bueModel && m_bueModel->isEnabled()) {
            double tempC = m_toolMaxTemperature;
            double shearStress = m_toolMaxStress / 1.7320508; // Fix 3: von Mises → shear: τ = σ_VM / √3
            double speed = m_config.getMachining().rpm;
            double chipThickness = initialOutput.chipThickness;
            double rakeAngleRad = 0.0;
            std::string material = "steel";
            try {
                auto c = m_config.getJson();
                if (c.contains("material") && c["material"].contains("type")) {
                    material = c["material"]["type"].get<std::string>();
                }
            } catch (...) {}
            m_bueModel->update(dt, chipThickness, speed, tempC,
                                shearStress, shearStress * 0.5,
                                rakeAngleRad, material);
        }

        // Apply edge subgrid corrections (effective rake, ploughing, size effect)
        m_strategy->applyEdgeSubgridCorrections();

        // Apply BUE corrections (force modulation, effective rake change)
        m_strategy->applyBUECorrections();

        // Apply chatter modulation (chip thickness modulation)
        m_strategy->applyChatterModulation(dt);
    }

    // --- Physics solvers ---
    // During air-cutting approach, skip ALL physics solvers (SPH + FEM + CFD).
    // The tool moves purely via G-Code kinematics — no internal forces.
    // FEM's stable dt is ~1e-8s; the macro timestep can be 1ms = 100,000× CFL violation.
    if (!m_skipPhysicsThisStep) {
        // Find solvers
        MPMSolver* mpm = nullptr;
        FEMSolver* fem = nullptr;
        CFDSolverGPU* cfd = nullptr;
        for (auto& solver : m_solvers) {
            if (!mpm) mpm = dynamic_cast<MPMSolver*>(solver.get());
            if (!fem) fem = dynamic_cast<FEMSolver*>(solver.get());
            if (!cfd) cfd = dynamic_cast<CFDSolverGPU*>(solver.get());
        }

        // ── AMS controller: compute mass‑scaling factor to reach target dt ──
        if (m_config.getSimulation().amsEnabled) {
            double naturalDt = std::numeric_limits<double>::max();
            for (auto& solver : m_solvers) {
                double sd = solver->getStableTimeStep();
                if (sd <= 0) continue;
                // Undo AMS contribution for MPM/FEM (CFD is unaffected by AMS)
                bool isAmsParticipant = (dynamic_cast<MPMSolver*>(solver.get()) != nullptr
                                      || dynamic_cast<FEMSolver*>(solver.get()) != nullptr);
                if (isAmsParticipant && m_amsMassScaling > 1.0) {
                    sd /= std::sqrt(m_amsMassScaling);
                }
                if (sd < naturalDt) naturalDt = sd;
            }
            if (naturalDt > 0 && naturalDt < std::numeric_limits<double>::max()) {
                const auto& sim = m_config.getSimulation();
                double maxTarget = naturalDt * std::sqrt(sim.amsMaxScalingFactor);
                double targetDt = std::min(sim.amsTargetTimeStep, maxTarget);
                targetDt = std::max(targetDt, naturalDt);
                double factor = (targetDt / naturalDt) * (targetDt / naturalDt);
                factor = std::max(std::min(factor, sim.amsMaxScalingFactor), 1.0);
                if (std::fabs(factor - m_amsMassScaling) > 0.01) {
                    for (auto& s : m_solvers) {
                        if (auto* p = dynamic_cast<MPMSolver*>(s.get())) p->setDynamicMassScaling(factor);
                        if (auto* p = dynamic_cast<FEMSolver*>(s.get())) p->setDynamicMassScaling(factor);
                    }
                    m_amsMassScaling = factor;
                }
            }
            dt = computeAdaptiveTimeStep();
        }
        m_amsMacroDt = dt;

        // Pass tool geometry to MPM solver for grid-based collision
        if (mpm && fem) {
            mpm->setToolNodes(fem->getDeviceNodes(), fem->getNodeCount());
        }

        // ── Sub‑cycling: MPM micro‑steps, FEM macro‑step ─────────────────────
        const int nSubSteps = (m_config.getSimulation().subCyclingSteps > 0 && mpm)
            ? m_config.getSimulation().subCyclingSteps : 1;
        const double microDt = dt / nSubSteps;
        m_amsMicroDt = microDt;

        m_amsSubStepCounter = 0;
        for (int sub = 0; sub < nSubSteps; ++sub, ++m_amsSubStepCounter) {
            if (mpm) mpm->step(microDt);
            if (fem) fem->advanceTime(microDt);
        }
        // FEM + CFD physics at macro dt
        if (fem) fem->step(dt);
        if (cfd) cfd->step(dt);

        // ── KE/IE ratio monitor ──────────────────────────────────────────────
        if (mpm) {
            mpm->syncMetrics();
            const double ke = mpm->getTotalKineticEnergy();
            const double ie = mpm->getTotalInternalEnergy();
            m_amsKEIERatio = (ie > 1e-30) ? (ke / ie) : 0.0;
            if (m_amsKEIERatio > 0.05 && m_amsMassScaling > 1.0) {
                double reduced = m_amsMassScaling * (0.05 / m_amsKEIERatio);
                reduced = std::max(reduced, 1.0);
                for (auto& s : m_solvers) {
                    if (auto* p = dynamic_cast<MPMSolver*>(s.get())) p->setDynamicMassScaling(reduced);
                    if (auto* p = dynamic_cast<FEMSolver*>(s.get())) p->setDynamicMassScaling(reduced);
                }
                m_amsMassScaling = reduced;
            }
        }

        // ── Implicit predictor–corrector thermal coupling ─────────────────────
        // Iterate the thermal subsystem (contact heat → CFD → coolant cooling)
        // to convergence so that T-dependent properties, heat partition, boiling
        // regime, and radiation all use consistent temperatures within a step.
        const int MAX_IMPLICIT_ITER = 3;
        double iterToolTemp = 0.0, iterWorkTemp = 0.0;
        for (int iter = 0; iter < MAX_IMPLICIT_ITER; ++iter) {
            if (m_contactEngaged && m_contactSolver) {
                // 1. Blok/Jaeger heat partition
                {
                    double toolTemp = fmax(iterToolTemp > 0.0 ? iterToolTemp : 25.0, 25.0);
                    double workTemp = fmax(iterWorkTemp > 0.0 ? iterWorkTemp : 25.0, 25.0);
                    const auto& mat = m_config.getMaterial();
                    const auto& tmat = m_config.getToolMaterial();
                    double effTool = thermalEffusivity(
                        toolTemp, tmat.density,
                        tmat.thermalConductivityTable, tmat.specificHeatTable,
                        tmat.thermalConductivity, tmat.specificHeat);
                    double effWork = thermalEffusivity(
                        workTemp, mat.density,
                        mat.thermalConductivityTable, mat.specificHeatTable,
                        mat.thermalConductivity, mat.specificHeat);
                    m_contactSolver->setHeatPartition(blokHeatPartition(effTool, effWork));
                }
                // 2. Flow-regime contact HTC
                if (m_coolantHardening) {
                    double htc = m_coolantHardening->getEffectiveHTC();
                    m_contactSolver->setContactHTC(htc);
                    // Also feed coolant HTC to FEM solver for convective BC
                    for (auto& solver : m_solvers) {
                        if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                            fem->setCoolantHTC(htc);
                            break;
                        }
                    }
                }
                m_contactSolver->resolveContacts(dt);
            }

            // 3. CFD heat sources (FEM nodes + MPM particles)
            if (m_cfdSolver) {
                MPMSolver* sph = nullptr;
                FEMSolver* fem = nullptr;
                for (auto& solver : m_solvers) {
                    if (!sph) sph = dynamic_cast<MPMSolver*>(solver.get());
                    if (!fem) fem = dynamic_cast<FEMSolver*>(solver.get());
                }

                if (sph && sph->getParticleCount() > 0) {
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

                if (sph && sph->getParticleCount() > 0) {
                    auto particles = sph->getParticles();
                    std::vector<double> particleTemps(particles.size());
                    std::vector<double> particlePos(particles.size() * 3);
                    #pragma omp parallel for if(particles.size() > 10000)
                    for (int i = 0; i < (int)particles.size(); ++i) {
                        particleTemps[i] = particles[i].temperature;
                        particlePos[i*3+0] = particles[i].x;
                        particlePos[i*3+1] = particles[i].y;
                        particlePos[i*3+2] = particles[i].z;
                    }
                    const auto& mat = m_config.getMaterial();
                    m_cfdSolver->setParticleHeatSources(
                        particleTemps.data(), particlePos.data(),
                        static_cast<int>(particles.size()),
                        mat.density, mat.specificHeat);
                }
            }

            // 4. Coolant hardening model (applies convective cooling)
            if (m_coolantHardening) {
                MPMSolver* mpm = nullptr;
                FEMSolver* fem = nullptr;
                for (auto& solver : m_solvers) {
                    if (!mpm) mpm = dynamic_cast<MPMSolver*>(solver.get());
                    if (!fem) fem = dynamic_cast<FEMSolver*>(solver.get());
                }
                m_coolantHardening->update(mpm, fem, m_cfdSolver, dt);
            }

            // 5. Read current solver temperatures for convergence check
            {
                double newToolTemp = 0.0, newWorkTemp = 0.0;
                for (auto& solver : m_solvers) {
                    auto* metrics = dynamic_cast<IMetricsProvider*>(solver.get());
                    if (!metrics) continue;
                    double t = metrics->getMaxTemperature();
                    if (dynamic_cast<FEMSolver*>(solver.get()))
                        newToolTemp = fmax(newToolTemp, t);
                    else if (dynamic_cast<MPMSolver*>(solver.get()))
                        newWorkTemp = fmax(newWorkTemp, t);
                }
                if (iter == 0) {
                    iterToolTemp = newToolTemp;
                    iterWorkTemp = newWorkTemp;
                } else {
                    double dTool = fabs(newToolTemp - iterToolTemp);
                    double dWork = fabs(newWorkTemp - iterWorkTemp);
                    if (dTool < 1.0 && dWork < 1.0) break;
                    iterToolTemp = newToolTemp;
                    iterWorkTemp = newWorkTemp;
                }
            }
        }
    }
    m_skipPhysicsThisStep = false;

    // (Contact solver already ran above, before SPH — see FIX B)

    // --- Adaptive optimisation (every 50 steps) ---
    if (m_optimizationManager && m_strategy && (m_currentStep % 50 == 0)) {
        ProcessState pState;
        pState.time            = m_currentTime;
        pState.toolStress      = m_toolMaxStress;
        pState.toolTemperature = m_toolMaxTemperature;

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

    // --- Steady-State Extrapolation Module ---
    if (!m_steadyStateReached && m_contactEngaged) {
        m_stressHistory.push_back(m_toolMaxStress);
        if (m_stressHistory.size() > 1000) {
            m_stressHistory.erase(m_stressHistory.begin());
        }

        // Only check for steady state after 5000 steps and depth > 0.5mm
        double currentDepthMm = getCurrentCutDepthMm();
        if (m_currentStep > 5000 && currentDepthMm > 0.5 && m_stressHistory.size() == 1000) {
            double sum = 0.0;
            for (double s : m_stressHistory) sum += s;
            double mean = sum / 1000.0;
            
            double variance = 0.0;
            for (double s : m_stressHistory) variance += (s - mean) * (s - mean);
            double stddev = std::sqrt(variance / 1000.0);
            
            double cv = (mean > 1e-6) ? (stddev / mean) : 1.0;
            if (cv < 0.02) {
                m_steadyStateReached = true;
            }
        }
    }

    if (m_steadyStateReached && !m_extrapolationFired) {
        double currentDepthMm = getCurrentCutDepthMm();

        if (m_config.getSimulation().fidelityMode == "standard") {
            if (m_chatterDynamics && m_chatterDynamics->isEnabled() && m_chatterDynamics->isChattering()) {
                std::cout << "\n[Analytics] WARNING: Unstable Chatter Detected at depth " << currentDepthMm << " mm." << std::endl;
                std::cout << "[Analytics] CV-based steady-state is invalid under chatter. Detection will retry.\n" << std::endl;
                // Do NOT set m_extrapolationFired — let detection retry
            } else {
                m_extrapolationFired = true;
                std::cout << "\n============================================================" << std::endl;
                std::cout << "[Extrapolation] Steady-State achieved at depth " << currentDepthMm << " mm." << std::endl;
                std::cout << "[Extrapolation] Standard Mode active: Halting explicit physics." << std::endl;

                double targetDepthMm = m_config.getMachining().depthOfCutMm;
                double remainingDistanceMm = std::max(0.0, targetDepthMm - currentDepthMm);
                double remainingDepthM = remainingDistanceMm / 1000.0;

                // Feed rate fallback: prefer G-Code feed, fall back to config
                double feedRateMs = state.feedRate;
                if (feedRateMs < 1e-12) feedRateMs = m_config.getMachining().feedRateMmMin / 60000.0;
                double remainingTime = (feedRateMs > 1e-12) ? (remainingDistanceMm / (feedRateMs * 1000.0)) : 0.0;

                // Fix 7: Material-specific thermal time constants for extrapolation
                // τ = L²·ρ·Cp / k (characteristic thermal diffusion time)
                double T_amb = m_config.getMachining().ambientTemperature;
                auto& extrapMat = m_config.getMaterial();
                auto& extrapTM  = m_config.getToolMaterial();
                double L_tool = 0.005;  // 5mm tool body half-thickness
                double L_work = 0.010;  // 10mm workpiece thermal penetration depth
                double tauTool = L_tool * L_tool * extrapTM.density * extrapTM.specificHeat /
                                 std::max(extrapTM.thermalConductivity, 1.0);
                double tauWork = L_work * L_work * extrapMat.density * extrapMat.specificHeat /
                                 std::max(extrapMat.thermalConductivity, 1.0);
                double tauCool = m_config.getSimulation().thermalTimeConstant * 0.1; // Coolant equilibrates fast
                double projectedFinalTime = m_currentTime + remainingTime;

                // Tool temperature (carbide: fast response, small τ)
                double projectedToolTemp = T_amb + (m_toolMaxTemperature - T_amb) * (1.0 - std::exp(-projectedFinalTime / tauTool));
                // Workpiece temperature (steel/Ti: slower response, larger τ)
                double projectedWorkTemp = T_amb + (m_workpieceMaxTemperature - T_amb) * (1.0 - std::exp(-projectedFinalTime / tauWork));
                // Coolant temperature (fluid: very fast response)
                double projectedCoolantTemp = T_amb + (m_coolantMaxTemperature - T_amb) * (1.0 - std::exp(-projectedFinalTime / tauCool));

                // Tool wear: linear projection from current accumulation rate
                double cuttingTimeSoFar = m_currentTime - m_contactStartTime;
                double projectedWear = m_toolMaxWear;
                if (cuttingTimeSoFar > 1e-12 && remainingTime > 1e-12) {
                    double wearRate = m_toolMaxWear / cuttingTimeSoFar;
                    projectedWear += wearRate * remainingTime;
                }

                // Fast-forward BUE Model
                if (m_bueModel && m_bueModel->isEnabled()) {
                    m_bueModel->extrapolateSteadyState(remainingTime, projectedToolTemp);
                    std::cout << "[Extrapolation] Fast-forwarded Built-Up Edge to steady state.\n";
                }

                std::cout << "[Extrapolation] Mathematical projection calculated." << std::endl;
                std::cout << "[Extrapolation] Projected Tool Temp:      " << projectedToolTemp << " C" << std::endl;
                std::cout << "[Extrapolation] Projected Workpiece Temp: " << projectedWorkTemp << " C" << std::endl;
                std::cout << "[Extrapolation] Projected Coolant Temp:   " << projectedCoolantTemp << " C" << std::endl;
                std::cout << "[Extrapolation] Projected Max Wear:       " << projectedWear * 1e6 << " um" << std::endl;
                std::cout << "[Extrapolation] Time constant tau (Tool):   " << tauTool << " s" << std::endl;
                std::cout << "[Extrapolation] Time constant tau (Work):   " << tauWork << " s" << std::endl;
                std::cout << "============================================================\n" << std::endl;

                // Override all extrapolated quantities for export
                m_toolMaxTemperature = projectedToolTemp;
                m_workpieceMaxTemperature = projectedWorkTemp;
                m_coolantMaxTemperature = projectedCoolantTemp;
                m_toolMaxWear = projectedWear;

                // ── Synthetic VTK steps for remaining depth ────────────────
                // Move the tool incrementally to target depth and export VTK
                // at each interval, so the output matches scientific mode's
                // full-cut progression (tool mesh + static particle cloud).
                const double depthPerSynthStepMm = 0.5;  // ~0.5mm per VTK file
                const int syntheticSteps = std::max(1,
                    static_cast<int>(std::ceil(remainingDistanceMm / depthPerSynthStepMm)));
                const double depthPerStepM = remainingDepthM / syntheticSteps;
                const double timePerStep = remainingTime / syntheticSteps;

                std::vector<MPMParticle> synthParticles;
                for (const auto& s : m_solvers) {
                    if (auto* mpm = dynamic_cast<MPMSolver*>(s.get())) {
                        synthParticles = mpm->getParticles();
                        break;
                    }
                }

                std::cout << "[Extrapolation] Generating " << syntheticSteps
                          << " synthetic VTK steps for remaining "
                          << remainingDistanceMm << " mm..." << std::endl;

                // Capture mesh value range for progressive field scaling
                Mesh baseMesh;
                for (const auto& s : m_solvers) {
                    if (auto* fem = dynamic_cast<FEMSolver*>(s.get())) {
                        fem->exportToMesh(baseMesh);
                        break;
                    }
                }
                double baseMaxTemp = 0.0, baseMaxWear = 0.0;
                for (const auto& n : baseMesh.nodes) {
                    if (n.temperature > baseMaxTemp) baseMaxTemp = n.temperature;
                    if (n.accumulatedWear > baseMaxWear) baseMaxWear = n.accumulatedWear;
                }
                double tempScaleFinal = (baseMaxTemp > 1.0)
                    ? m_toolMaxTemperature / baseMaxTemp : 1.0;
                double wearScaleFinal = (baseMaxWear > 1e-15)
                    ? m_toolMaxWear / baseMaxWear : 1.0;

                for (int si = 0; si < syntheticSteps; ++si) {
                    for (auto& s : m_solvers) {
                        if (auto* fem = dynamic_cast<FEMSolver*>(s.get())) {
                            fem->translateMesh(0.0, 0.0, -depthPerStepM);
                            break;
                        }
                    }
                    m_currentStep++;
                    m_currentTime += timePerStep;

                    // Export at each synthetic step (no output interval gating)
                    Mesh synthMesh;
                    for (const auto& s : m_solvers) {
                        if (auto* fem = dynamic_cast<FEMSolver*>(s.get())) {
                            fem->exportToMesh(synthMesh);
                            break;
                        }
                    }
                    if (m_contactSolver) m_contactSolver->applyCoatingWearToMesh(synthMesh);

                    // Progressively scale fields: linear ramp from 1.0 up to final ratio
                    double progress = static_cast<double>(si + 1) / syntheticSteps;
                    double tempScale = 1.0 + (tempScaleFinal - 1.0) * progress;
                    double wearScale = 1.0 + (wearScaleFinal - 1.0) * progress;
                    for (auto& n : synthMesh.nodes) {
                        n.temperature = 25.0 + (n.temperature - 25.0) * tempScale;
                        n.accumulatedWear *= wearScale;
                    }

                    for (auto& exporter : m_exporters) {
                        exporter->exportStep(m_currentStep, m_currentTime, synthMesh);
                        exporter->exportParticles(m_currentStep, m_currentTime, synthParticles);
                        exporter->exportDetailedMetrics(
                            m_currentStep, m_currentTime,
                            m_workpieceMaxStress, m_workpieceMaxTemperature,
                            m_toolMaxStress, m_toolMaxTemperature,
                            m_coolantMaxTemperature);
                    }
                }

                std::cout << "[Extrapolation] Final tool at target depth: "
                          << targetDepthMm << " mm" << std::endl;
                m_shouldStop = true;
            }
        } else {
            m_extrapolationFired = true;
            std::cout << "\n[Analytics] Steady-State detected at " << currentDepthMm
                      << " mm. Scientific mode active: Continuing full simulation.\n" << std::endl;
        }
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
    
    MPMSolver* sph = nullptr;
    FEMSolver* fem = nullptr;
    for (auto& solver : m_solvers) {
        if (!sph) sph = dynamic_cast<MPMSolver*>(solver.get());
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
    const int   configuredSteps = simParams.numSteps;
    int         totalSteps = configuredSteps;
    const int   outInterval = simParams.outputIntervalSteps;

    std::cout << "[Engine] Starting simulation (" << configuredSteps << " configured steps)..." << std::endl;
    std::cout << "[Engine] Physics patch: contact-lock + split-material-metrics + rotary-velocity v3" << std::endl;

    m_currentStep       = 0;
    m_currentTime       = 0.0;
    m_maxStress         = 0.0;
    m_workpieceMaxStress = 0.0;
    m_toolMaxStress     = 0.0;
    m_maxTemperature    = m_config.getMachining().ambientTemperature;
    m_workpieceMaxTemperature = m_config.getMachining().ambientTemperature;
    m_toolMaxTemperature = m_config.getMachining().ambientTemperature;
    m_coolantMaxTemperature = m_config.getMachining().ambientTemperature;
    m_toolMaxWear       = 0.0;
    m_contactEngaged    = false;
    m_contactStartTime  = -1.0;
    m_contactStartToolZ = 0.0;
    m_skipPhysicsThisStep = false;

    // ── Initialize Simulation Health Monitor ──────────────────────────────
    {
        HealthMonitorConfig hmc;
        hmc.enabled = true;
        hmc.meltingPointC = m_config.getMaterial().meltingPoint;
        hmc.yieldStrengthPa = m_config.getMaterial().yieldStrength;
        hmc.maxPhysicalForceN = 100e6;  // 100 MN absolute ceiling
        hmc.thermalExplosionFraction = 0.05;  // 5% of particles at melt
        hmc.thermalExplosionMaxSteps = 200;
        hmc.stressExplosionMultiplier = 50.0;
        hmc.contactExplosionMultiplier = 50;
        hmc.checkIntervalSteps = simParams.outputIntervalSteps;

        // Allow JSON override
        const auto& j = m_config.getJson();
        if (j.contains("health_monitor")) {
            const auto& hj = j["health_monitor"];
            hmc.enabled = hj.value("enabled", hmc.enabled);
            hmc.maxPhysicalForceN = hj.value("max_force_N", hmc.maxPhysicalForceN);
            hmc.stressExplosionMultiplier = hj.value("stress_multiplier", hmc.stressExplosionMultiplier);
            hmc.thermalExplosionFraction = hj.value("thermal_fraction", hmc.thermalExplosionFraction);
            hmc.contactExplosionMultiplier = hj.value("contact_multiplier", hmc.contactExplosionMultiplier);
        }

        m_healthMonitor.initialize(hmc);
        if (hmc.enabled) {
            std::cout << "[HealthMonitor] Enabled — yield=" << hmc.yieldStrengthPa / 1e6
                      << "MPa, melt=" << hmc.meltingPointC << "°C, maxForce="
                      << hmc.maxPhysicalForceN / 1e6 << "MN" << std::endl;
        }
    }

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
    // INITIAL KINEMATIC SYNC — Pre-position tool at clearance height
    // ====================================================================
    // The G-Code starts at machine home (0,0,0). The first G00 rapid moves
    // the tool to the clearance height (e.g., Z=+2mm). We skip all initial
    // G00 rapids and pre-position the tool at the END of the last rapid
    // (= START of the first cutting move). This ensures the tool starts
    // ABOVE the workpiece with a proper air gap.
    //
    // CRITICAL FIX: Previous code scanned by time with a step > rapid duration
    // (267μs step vs 200μs rapid), skipping the entire rapid. Now uses
    // getFirstCuttingMoveTime() which iterates segments directly.
    // ====================================================================
    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        // Get the EXACT time boundary between initial rapids and first cutting move
        double cutStartTime = m_gcodeInterpreter->getFirstCuttingMoveTime();
        
        // Get the EXACT position at the end of the last initial rapid
        // (This is the clearance height position, e.g., (0, 0, +2mm))
        Vec3 clearancePos;
        if (cutStartTime > 1e-12) {
            // Use getSegmentEndPosition to get the exact endpoint of the last rapid
            // (avoids floating-point interpolation errors at segment boundaries)
            clearancePos = m_gcodeInterpreter->getSegmentEndPosition(cutStartTime - 1e-9);
        } else {
            // No initial rapids — first segment is already a cutting move
            // Use the start position of the first segment
            clearancePos = m_gcodeInterpreter->getStateAtTime(0.0).position;
        }
        
        // Set engine clock to start at this time (skip rapid approach)
        m_currentTime = cutStartTime;
        
        std::cout << "[Engine] Pre-positioning tool at G-Code clearance height" << std::endl;
        std::cout << "  First cutting move starts at T=" << std::fixed << std::setprecision(6)
                  << cutStartTime*1000 << " ms" << std::endl;
        std::cout << "  Tool clearance position: (" << clearancePos.x*1000 << ", " 
                  << clearancePos.y*1000 << ", " << clearancePos.z*1000 << ") mm" << std::endl;
        
        // Pre-position the tool mesh at clearance
        if (clearancePos.lengthSq() > 1e-18) {
            for (auto& solver : m_solvers) {
                if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                    fem->translateMesh(clearancePos.x, clearancePos.y, clearancePos.z);
                    fem->setVirtualSpindleState(clearancePos, Vec3::zero());
                }
            }
        }
        
        // === POST-SYNC VALIDATION ===
        // Verify the tool is NOT overlapping the workpiece at T=0
        double tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ;
        double wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ;
        bool foundFem = false, foundSph = false;
        for (const auto& solver : m_solvers) {
            if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                fem->getBounds(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ);
                foundFem = true;
            }
            if (auto* sph = dynamic_cast<MPMSolver*>(solver.get())) {
                sph->getBounds(wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ);
                foundSph = true;
            }
        }
        if (foundFem && foundSph) {
            // Check Z overlap (tool tip vs workpiece top)
            double toolTipZ = tMinZ;
            double workpieceTopZ = wMaxZ;
            double gap = toolTipZ - workpieceTopZ;
            std::cout << "[Engine] Post-sync gap: tool tip Z=" << toolTipZ*1000 
                      << "mm, workpiece top Z=" << workpieceTopZ*1000 
                      << "mm, gap=" << gap*1000 << "mm" << std::endl;

            // ── SAFE ENGAGEMENT: Enforce minimum air gap ─────────────────
            // The contact solver's detection radius must be SMALLER than the
            // gap, otherwise particles register as "in contact" before the
            // tool physically reaches the workpiece.
            //
            // Required gap = 2 × contactRadius to ensure clean approach.
            // If gap is insufficient, auto-translate tool upward.
            double requiredGap = 0.0;
            if (m_contactSolver) {
                // Read contact radius from the solver's current config
                double contactRadius = m_config.getSPH().smoothingRadius * 1.5;
                requiredGap = contactRadius * 2.0;
            } else {
                // Fallback: use SPH smoothing radius as proxy
                requiredGap = m_config.getSPH().smoothingRadius * 3.0;
            }

            if (gap < requiredGap) {
                double correction = requiredGap - gap + 1e-6; // +1μm safety
                std::cout << "[Engine] AUTO-FIX: Gap (" << gap*1000 << "mm) < required ("
                          << requiredGap*1000 << "mm). Translating tool UP by " 
                          << correction*1000 << "mm to prevent contact explosion." << std::endl;
                
                for (auto& solver : m_solvers) {
                    if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                        fem->translateMesh(0.0, 0.0, correction);
                        fem->getBounds(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ);
                        gap = tMinZ - workpieceTopZ;
                        std::cout << "[Engine] Corrected gap: " << gap*1000 << "mm" << std::endl;
                    }
                }
            }
            
            if (gap < 0) {
                std::cerr << "\n[Engine] CRITICAL WARNING: Tool is " 
                          << (-gap)*1000 << "mm INSIDE the workpiece at T=0!\n"
                          << "  This will cause contact explosion. Check G-Code clearance height\n"
                          << "  and workpiece dimensions.\n" << std::endl;
            } else {
                std::cout << "[Engine] Post-sync validation PASSED: " 
                          << gap*1000 << "mm air gap" << std::endl;
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

    if (simParams.runToTargetDepth && m_config.getMachining().depthOfCutMm > 0.0) {
        const double feedRate = m_config.getMachining().feedRateMmMin / 60000.0;
        double stableDt = std::max(simParams.minTimeStep, computeAdaptiveTimeStep());
        if (simParams.amsEnabled) {
            const double maxAchievableDt = stableDt * std::sqrt(simParams.amsMaxScalingFactor);
            const double amsDt = std::min(simParams.amsTargetTimeStep, maxAchievableDt);
            stableDt = std::max(stableDt, amsDt);
        }
        if (feedRate > 1e-12 && stableDt > 1e-15) {
            const double targetTime =
                (m_config.getMachining().depthOfCutMm / 1000.0) / feedRate;
            const long long estimatedCutSteps =
                static_cast<long long>(std::ceil(targetTime / stableDt));
            const long long estimatedBudget =
                static_cast<long long>(std::ceil(estimatedCutSteps * 1.10)) + 1000;
            const int cappedBudget = static_cast<int>(
                std::min<long long>(simParams.maxAutoExtendedSteps, estimatedBudget));
            totalSteps = std::max(totalSteps, cappedBudget);

            std::cout << "[Engine] Depth-governed run enabled: target="
                      << std::fixed << std::setprecision(3)
                      << m_config.getMachining().depthOfCutMm << " mm, estimated cut time="
                      << targetTime << " s"
                      << (simParams.amsEnabled ? " (AMS-scaled dt)" : "")
                      << ", step budget=" << std::defaultfloat << totalSteps << std::endl;
        }
    }

    // ── OUTPUT HEADER: clarify metric semantics ─────────────────────────
    std::cout << "  Legend:  Wkpc[pk] = peak von Mises stress across all workpiece particles (MPa)"
              << "\n           Tool[pk]= peak von Mises stress across all tool FEM nodes (MPa)"
              << "\n           dt[xxx]  = adaptive timestep, limiter in brackets (e.g. SPH, FEM, CFD)"
              << "\n           HeatGen  = cumulative contact heat generated (J)"
              << "\n           Cooling  = per-step coolant dT on particles (C), proving live coupling"
              << std::endl;

    bool extensionLogged = false;
    while (m_currentStep < totalSteps && !m_shouldStop) {
        if (hasReachedTargetDepth()) {
            std::cout << "[Engine] Target depth reached: "
                      << std::fixed << std::setprecision(3)
                      << getCurrentCutDepthMm() << " mm at step " << m_currentStep
                      << " (t=" << m_currentTime << "s). Stopping." << std::endl;
            break;
        }

        if (!extensionLogged && m_currentStep >= configuredSteps &&
            simParams.runToTargetDepth && !hasReachedTargetDepth()) {
            std::cout << "[Engine] Configured step count reached, continuing because "
                      << "run_to_target_depth=true and current cut depth is "
                      << std::fixed << std::setprecision(3)
                      << getCurrentCutDepthMm() << " mm of "
                      << m_config.getMachining().depthOfCutMm << " mm." << std::endl;
            extensionLogged = true;
        }

        // === G-Code Early Termination ===
        // Stop when the G-Code toolpath is fully consumed (+ 5% buffer).
        // Prevents thousands of empty steps after the tool has retracted.
        if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
            double gcDuration = m_gcodeInterpreter->getTotalDuration();
            if (m_currentTime >= gcDuration * 1.05) {
                if (simParams.runToTargetDepth && !hasReachedTargetDepth()) {
                    std::cerr << "[Engine] WARNING: G-Code ended at "
                              << getCurrentCutDepthMm() << " mm, before requested "
                              << m_config.getMachining().depthOfCutMm
                              << " mm. Check the G-Code Z depth or depth_of_cut_mm." << std::endl;
                } else {
                    std::cout << "[Engine] G-Code path complete at step " << m_currentStep
                              << " (t=" << m_currentTime << "s). Stopping." << std::endl;
                }
                break;
            }
        }
        double dt = computeAdaptiveTimeStep();
        handleAirCutting(dt);
        step(dt);

        if (m_currentStep % outInterval == 0 || m_currentStep == 1) {
            updateMetrics();

            double percent = (100.0 * m_currentStep) / totalSteps;
            double heatGen = m_contactSolver ? m_contactSolver->getHeatGenerated() : 0.0;
            double coolingEnergy = (m_coolantHardening && m_coolantHardening->isEnabled())
                                   ? m_coolantHardening->getState().lastHeatRemovedJ : 0.0;
            int cooledCount = (m_coolantHardening && m_coolantHardening->isEnabled())
                              ? m_coolantHardening->getState().cooledParticleCount : 0;

            std::cout << "  Step " << std::setw(7) << m_currentStep
                      << " | " << std::fixed << std::setprecision(1) << percent << "%"
                      << " | dt: " << std::scientific << std::setprecision(2) << dt
                      << " [" << m_dtLimiterName << "]"
                      << " | Wkpc[pk]: " << std::scientific << std::setprecision(2)
                      << m_workpieceCurrentStress / 1e6 << " MPa, "
                      << std::fixed << std::setprecision(1)
                      << m_workpieceMaxTemperature << " C"
                      << " | Tool[pk]: " << std::scientific << std::setprecision(2)
                      << m_toolCurrentStress / 1e6 << " MPa, "
                      << std::fixed << std::setprecision(1)
                      << m_toolMaxTemperature << " C"
                      << " | Coolant: " << std::fixed << std::setprecision(1)
                      << m_coolantMaxTemperature << " C"
                      << " | HeatGen: " << std::scientific << std::setprecision(2)
                      << heatGen << " J";
            if (coolingEnergy > 0.0) {
                std::cout << " | Qrem: " << std::scientific << std::setprecision(2)
                          << coolingEnergy << " J (" << cooledCount << " particles)";
            }

            if (m_contactStartTime >= 0.0 && m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
                auto nowSnap = m_gcodeInterpreter->getStateAtTime(m_currentTime);
                double cutDepthMm = std::max(0.0, (m_contactStartToolZ - nowSnap.position.z) * 1000.0);
                std::cout << " | CutDepth: " << std::fixed << std::setprecision(4)
                          << cutDepthMm << " mm";
            }

            std::cout << std::endl;

            // ── HEALTH MONITOR CHECK ─────────────────────────────────────
            if (m_healthMonitor.isEnabled()) {
                // Count particles at melting point (quick scan from SPH)
                int particlesAtMelt = 0;
                int numParticles = 0;
                int numNodes = 0;
                double kineticEnergy = 0;
                for (const auto& solver : m_solvers) {
                    if (auto* sph = dynamic_cast<MPMSolver*>(solver.get())) {
                        numParticles = sph->getParticleCount();
                        kineticEnergy = sph->getTotalKineticEnergy();
                        // Count particles at melt from the last metrics sync
                        const auto& particles = sph->getHostParticles();
                        for (const auto& p : particles) {
                            if (p.status == ParticleStatus::ACTIVE &&
                                p.temperature >= m_config.getMaterial().meltingPoint - 1.0) {
                                particlesAtMelt++;
                            }
                        }
                    }
                    if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                        numNodes = fem->getNodeCount();
                    }
                }

                int contactCount = m_contactSolver ? m_contactSolver->getContactCount() : 0;
                double totalForce = m_contactSolver ? m_contactSolver->getTotalContactForce() : 0;
                double materialMaxTemperature =
                    std::max(m_workpieceMaxTemperature, m_toolMaxTemperature);

                auto healthResult = m_healthMonitor.check(
                    m_currentStep, dt, m_config.getSimulation().minTimeStep,
                    contactCount, totalForce,
                    m_maxStress, materialMaxTemperature,
                    kineticEnergy, numParticles, numNodes,
                    particlesAtMelt
                );

                if (healthResult.severity == HealthSeverity::HALT) {
                    std::cerr << "\n" << healthResult.message << std::endl;
                    std::cerr << m_healthMonitor.getDiagnosticReport();
                    // Export what we have before stopping
                    exportResults();
                    m_shouldStop = true;
                    break;
                } else if (healthResult.severity == HealthSeverity::WARN) {
                    std::cerr << healthResult.message << std::endl;
                }
            }

            // ----------------------------------------------------------------
            // FIX: update m_toolMesh from the FEM solver before capturing it
            // ----------------------------------------------------------------
            for (const auto& solver : m_solvers) {
                if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                    fem->exportToMesh(m_toolMesh);
                    break;
                }
            }
            if (m_contactSolver) m_contactSolver->applyCoatingWearToMesh(m_toolMesh);
            ToolRegionClassifier::annotateRiskFields(m_toolMesh, m_config);

            // Collect SPH particles for live export
            std::vector<MPMParticle> capturedParticles;
            for (const auto& solver : m_solvers) {
                if (auto* sph = dynamic_cast<MPMSolver*>(solver.get())) {
                    capturedParticles = sph->getParticles();
                    break;
                }
            }

            // Async export (non-blocking)
            for (auto& exporter : m_exporters) {
                int                      capturedStep    = m_currentStep;
                double                   capturedTime    = m_currentTime;
                double                   capturedWorkStress = m_workpieceMaxStress;
                double                   capturedWorkTemp = m_workpieceMaxTemperature;
                double                   capturedToolStress = m_toolMaxStress;
                double                   capturedToolTemp = m_toolMaxTemperature;
                double                   capturedCoolantTemp = m_coolantMaxTemperature;
                Mesh                     capturedMesh    = m_toolMesh;   // now has FEM data
                std::vector<MPMParticle> capturedPts     = capturedParticles;
                IExporter*               exporterPtr     = exporter.get();

                m_asyncExporter->enqueue(
                    [exporterPtr, capturedStep, capturedTime,
                     capturedMesh, capturedPts,
                     capturedWorkStress, capturedWorkTemp,
                     capturedToolStress, capturedToolTemp,
                     capturedCoolantTemp]() {
                        exporterPtr->exportStep(capturedStep, capturedTime, capturedMesh);
                        exporterPtr->exportDetailedMetrics(
                            capturedStep, capturedTime,
                            capturedWorkStress, capturedWorkTemp,
                            capturedToolStress, capturedToolTemp,
                            capturedCoolantTemp);
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
    if (m_contactSolver) m_contactSolver->applyCoatingWearToMesh(m_toolMesh);

    // When extrapolated, scale per-node fields to match projected final values
    // so the tool mesh (and thus RND report + VTK) reflects the full-cut state.
    if (m_extrapolationFired && m_config.getSimulation().fidelityMode == "standard") {
        double tempScale = 1.0, wearScale = 1.0;
        double meshMaxTemp = 0.0, meshMaxWear = 0.0;
        for (const auto& n : m_toolMesh.nodes) {
            if (n.temperature > meshMaxTemp) meshMaxTemp = n.temperature;
            if (n.accumulatedWear > meshMaxWear) meshMaxWear = n.accumulatedWear;
        }
        if (meshMaxTemp > 1.0) tempScale = m_toolMaxTemperature / meshMaxTemp;
        if (meshMaxWear > 1e-15) wearScale = m_toolMaxWear / meshMaxWear;
        for (auto& n : m_toolMesh.nodes) {
            n.temperature = 25.0 + (n.temperature - 25.0) * tempScale;
            n.accumulatedWear *= wearScale;
        }
    }

    ToolRegionClassifier::annotateRiskFields(m_toolMesh, m_config);

    // --- Collect final particle data ---
    std::vector<MPMParticle> finalParticles;
    for (const auto& solver : m_solvers) {
        if (auto* sph = dynamic_cast<MPMSolver*>(solver.get())) {
            finalParticles = sph->getParticles();
            break;
        }
    }

    const double ambient = m_config.getMachining().ambientTemperature;
    bool cuttingOccurred =
        m_contactEngaged ||
        m_workpieceMaxStress > 1.0e3 ||
        m_toolMaxStress > 1.0e3 ||
        m_workpieceMaxTemperature > ambient + 0.5 ||
        m_toolMaxTemperature > ambient + 0.5;
    if (!cuttingOccurred) {
        for (const auto& p : finalParticles) {
            double stressMag = std::sqrt(
                p.stress_xx * p.stress_xx + p.stress_yy * p.stress_yy +
                p.stress_zz * p.stress_zz + 2.0 * (
                p.stress_xy * p.stress_xy + p.stress_xz * p.stress_xz +
                p.stress_yz * p.stress_yz));
            if (p.status == ParticleStatus::CHIP ||
                p.plasticStrain > 1.0e-8 ||
                p.damage > 1.0e-8 ||
                stressMag > 1.0e3 ||
                p.temperature > ambient + 0.5) {
                cuttingOccurred = true;
                break;
            }
        }
    }

    // --- Analytics ---
    if (m_strategy) {
        auto output = m_strategy->computeOutput();

        // Surface roughness predictor
        if (m_roughnessPredictor) {
            if (!cuttingOccurred) {
                std::cout << "  [Roughness] N/A - no cutting occurred" << std::endl;
            } else {
            // Initialise with config before first use
            m_roughnessPredictor->initialize(m_config);

            // Fix N: Populate surfaceParticles from SPH for physics-based roughness.
            // Without this, surfaceParticles is empty and roughness uses only the
            // kinematic formula (f²/32r), ignoring actual material deformation.
            if (!finalParticles.empty()) {
                // Find workpiece top surface Z coordinate
                double wpMaxZ = -1e20;
                for (const auto& solver : m_solvers) {
                    if (auto* sph = dynamic_cast<MPMSolver*>(solver.get())) {
                        double mnX, mnY, mnZ, mxX, mxY, mxZ;
                        sph->getBounds(mnX, mnY, mnZ, mxX, mxY, mxZ);
                        wpMaxZ = mxZ;
                        break;
                    }
                }
                // Collect particles within 2× smoothing radius of top surface
                double surfThreshold = m_config.getSPH().smoothingRadius * 2.0;
                for (const auto& p : finalParticles) {
                    if (p.status == ParticleStatus::ACTIVE &&
                        std::abs(p.z - wpMaxZ) < surfThreshold) {
                        output.surfaceParticles.push_back(p);
                    }
                }
                output.surfaceNormal = Vec3(0, 0, 1);
                output.profileLength = 0.004;  // 4mm scan length
                output.toolNoseRadius = m_config.getMachining().depthOfCutMm > 0
                    ? m_config.getMachining().depthOfCutMm / 1000.0 * 0.1
                    : 0.4e-3;
            }

            // FIX: predict() takes a single MachiningOutput argument
            auto roughness = m_roughnessPredictor->predict(output);
            std::cout << "  [Roughness] Ra = " << std::fixed << std::setprecision(3)
                      << roughness.Ra << " μm"
                      << (roughness.meetsTolerance ? "  [PASS]" : "  [FAIL]")
                      << std::endl;
            }
        }

        // Residual stress analyzer
        // FIX: use analyzeFromParticles() (the method that actually exists)
        if (m_stressAnalyzer && !cuttingOccurred) {
            std::cout << "  [ResidualStress] N/A - no cutting occurred" << std::endl;
        } else if (m_stressAnalyzer && !finalParticles.empty()) {
            m_stressAnalyzer->configure(0.001 /*maxDepth m*/, 50 /*levels*/);

            // FIX: Surface normal must match actual workpiece orientation.
            // For Z-up workpiece (drill enters from top), the machined surface
            // normal is +Z, not +Y. Using (0,1,0) caused distanceToSurface()
            // to find zero particles within the Y plane → all-zero residual stress.
            double minX, minY, minZ, maxX, maxY, maxZ;
            m_solvers.front()->getBounds(minX, minY, minZ, maxX, maxY, maxZ);

            // Auto-detect: the top surface is the one with the largest extent
            // For drilling into a block, the top face is at Z=0 (maxZ)
            Vec3 surfaceNormal(0.0, 0.0, 1.0);  // Z-up for drilling/milling
            Vec3 surfacePoint((minX + maxX) * 0.5, (minY + maxY) * 0.5, maxZ);

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

    // --- Tool R&D semantic report ---
    {
        auto regionReport = ToolRegionClassifier::buildReport(m_toolMesh, m_config);
        const std::string reportPath = outputDir + "/tool_region_report.json";
        ToolRegionClassifier::exportReportJson(reportPath, regionReport);
        ToolRNDAnalysisInput rndInput;
        rndInput.toolMesh = &m_toolMesh;
        rndInput.particles = &finalParticles;
        rndInput.config = &m_config;
        rndInput.contactHeatJ = m_contactSolver ? m_contactSolver->getCumulativeHeatGenerated() : 0.0;
        rndInput.contactForceN = m_contactSolver ? m_contactSolver->getTotalContactForce() : 0.0;
        rndInput.currentTimeS = m_currentTime;
        rndInput.regionReport = regionReport;
        rndInput.edgeSubgrid = m_edgeSubgrid;
        rndInput.bueModel = m_bueModel;
        rndInput.chatter = m_chatterDynamics;
        const std::string rndReportPath = outputDir + "/tool_rnd_report.json";
        ToolRNDAnalyzer::exportJson(rndReportPath, rndInput);
        
        try {
            std::ifstream in(rndReportPath);
            if (in.is_open()) {
                nlohmann::json j;
                in >> j;
                in.close();
                
                j["fidelity_mode"] = m_config.getSimulation().fidelityMode;
                if (m_extrapolationFired && m_config.getSimulation().fidelityMode == "standard") {
                    j["extrapolated"] = true;
                    j["steady_state_depth_mm"] = getCurrentCutDepthMm();
                } else {
                    j["extrapolated"] = false;
                }
                
                std::ofstream out(rndReportPath);
                out << j.dump(4) << std::endl;
            }
        } catch (...) {}
        std::cout << "  [ToolRegions] Report exported: " << reportPath
                  << " (confidence=" << std::fixed << std::setprecision(1)
                  << regionReport.classificationConfidence * 100.0 << "%)"
                  << std::endl;
        std::cout << "  [ToolRND] Report exported: " << rndReportPath << std::endl;
    }

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
    const double ambient = m_config.getMachining().ambientTemperature;

    for (const auto& solver : m_solvers) {
        auto* metrics = dynamic_cast<IMetricsProvider*>(solver.get());
        if (metrics) {
            metrics->syncMetrics();
            const double stress = metrics->getMaxStress();
            const double temperature = metrics->getMaxTemperature();

            if (dynamic_cast<MPMSolver*>(solver.get())) {
                m_workpieceCurrentStress = stress;
                m_workpieceMaxStress = std::max(m_workpieceMaxStress, stress);
                m_workpieceMaxTemperature =
                    std::max(m_workpieceMaxTemperature, std::max(temperature, ambient));
            } else if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                m_toolCurrentStress = stress;
                m_toolMaxStress = std::max(m_toolMaxStress, stress);
                m_toolMaxTemperature =
                    std::max(m_toolMaxTemperature, std::max(temperature, ambient));
                m_toolMaxWear = std::max(m_toolMaxWear, fem->getResults().maxWear);
            } else if (dynamic_cast<CFDSolverGPU*>(solver.get())) {
                m_coolantMaxTemperature =
                    std::max(m_coolantMaxTemperature, temperature);
            }
        }
    }

    if (m_contactSolver && m_contactSolver->getContactCount() > 0) {
        double contactPressure = m_contactSolver->getEstimatedContactPressure();
        m_toolCurrentStress = std::max(m_toolCurrentStress, contactPressure);
        m_workpieceCurrentStress = std::max(m_workpieceCurrentStress, contactPressure);
        m_toolMaxStress = std::max(m_toolMaxStress, contactPressure);
        m_workpieceMaxStress = std::max(m_workpieceMaxStress, contactPressure);
        m_toolMaxTemperature = std::max(m_toolMaxTemperature, ambient);
        m_workpieceMaxTemperature = std::max(m_workpieceMaxTemperature, ambient);
    }

    m_maxStress = std::max(m_workpieceMaxStress, m_toolMaxStress);
    m_maxTemperature = std::max({
        m_workpieceMaxTemperature,
        m_toolMaxTemperature,
        m_coolantMaxTemperature
    });
}

// ---------------------------------------------------------------------------
// Internal: adaptive time-step
// ---------------------------------------------------------------------------

double SimulationEngine::computeAdaptiveTimeStep() {
    const auto& simParams = m_config.getSimulation();
    double dt = simParams.timeStepDuration;
    m_dtLimiterName = "requested";

    for (const auto& solver : m_solvers) {
        double solverDt = solver->getStableTimeStep();
        if (solverDt > 0 && solverDt < dt) {
            dt = solverDt;
            m_dtLimiterName = solver->getName();
        }
    }

    if (dt <= simParams.minTimeStep) {
        dt = simParams.minTimeStep;
        m_dtLimiterName = "min_dt";
    }
    if (dt >= simParams.maxTimeStep) {
        dt = simParams.maxTimeStep;
        m_dtLimiterName = "max_dt";
    }

    return dt;
}

// ---------------------------------------------------------------------------
// Internal: air-gap detection
// ---------------------------------------------------------------------------

double SimulationEngine::computeAirGap() const {
    if (m_solvers.size() < 2) return 0.0;

    const IPhysicsSolver* solverA = nullptr;
    const IPhysicsSolver* solverB = nullptr;
    const MPMSolver* mpm = nullptr;
    const FEMSolver* fem = nullptr;
    for (const auto& solver : m_solvers) {
        if (!solver) continue;
        if (!mpm) mpm = dynamic_cast<const MPMSolver*>(solver.get());
        if (!fem) fem = dynamic_cast<const FEMSolver*>(solver.get());
    }

    if (mpm && fem) {
        solverA = mpm;
        solverB = fem;
    } else {
        solverA = m_solvers[0].get();
        solverB = m_solvers[1].get();
    }

    double minX1, minY1, minZ1, maxX1, maxY1, maxZ1;
    double minX2, minY2, minZ2, maxX2, maxY2, maxZ2;
    solverA->getBounds(minX1, minY1, minZ1, maxX1, maxY1, maxZ1);
    solverB->getBounds(minX2, minY2, minZ2, maxX2, maxY2, maxZ2);
    
    // Detect corrupt bounds in either solver
    for (int s = 0; s < 2; s++) {
        double mnX = (s==0)?minX1:minX2, mxX = (s==0)?maxX1:maxX2;
        double mnY = (s==0)?minY1:minY2, mxY = (s==0)?maxY1:maxY2;
        double mnZ = (s==0)?minZ1:minZ2, mxZ = (s==0)?maxZ1:maxZ2;
        if (std::abs(mnX) > 1e3 || std::abs(mxX) > 1e3 ||
            std::abs(mnY) > 1e3 || std::abs(mxY) > 1e3 ||
            std::abs(mnZ) > 1e3 || std::abs(mxZ) > 1e3) {
            std::cerr << "[DEBUG] CORRUPT BOUNDS solver[" << s << "]: X[" << mnX << ", " << mxX << "] "
                      << "Y[" << mnY << ", " << mxY << "] "
                      << "Z[" << mnZ << ", " << mxZ << "]" << std::endl;
        }
    }

    double dx = std::max(0.0, std::max(minX1 - maxX2, minX2 - maxX1));
    double dy = std::max(0.0, std::max(minY1 - maxY2, minY2 - maxY1));
    double dz = std::max(0.0, std::max(minZ1 - maxZ2, minZ2 - maxZ1));
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double SimulationEngine::getCurrentCutDepthMm() const {
    if (m_contactStartTime < 0.0) return 0.0;

    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        auto snap = m_gcodeInterpreter->getStateAtTime(m_currentTime);
        return std::max(0.0, (m_contactStartToolZ - snap.position.z) * 1000.0);
    }

    double feedRate = m_config.getMachining().feedRateMmMin / 60000.0;
    double elapsed = std::max(0.0, m_currentTime - m_contactStartTime);
    return elapsed * feedRate * 1000.0;
}

bool SimulationEngine::hasReachedTargetDepth() const {
    const auto& sim = m_config.getSimulation();
    const double targetDepthMm = m_config.getMachining().depthOfCutMm;
    if (!sim.runToTargetDepth || targetDepthMm <= 0.0) return false;
    return getCurrentCutDepthMm() + sim.targetDepthToleranceMm >= targetDepthMm;
}

void SimulationEngine::recordContactEngagement(double dt) {
    if (m_contactStartTime >= 0.0) return;

    m_contactStartTime = m_currentTime;
    double feedRate = m_config.getMachining().feedRateMmMin / 60000.0;

    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        auto snap = m_gcodeInterpreter->getStateAtTime(m_currentTime);
        m_contactStartToolZ = snap.position.z;
        if (snap.feedRate > 1e-12) feedRate = snap.feedRate;
    }

    const double targetDepthMm = m_config.getMachining().depthOfCutMm;
    const double currentDepthMm = getCurrentCutDepthMm();
    const double remainingDepthMm =
        targetDepthMm > 0.0 ? std::max(0.0, targetDepthMm - currentDepthMm) : 0.0;
    const long long estimatedRemainingSteps =
        (dt > 1e-15 && feedRate > 1e-12)
            ? static_cast<long long>(std::ceil((remainingDepthMm / 1000.0) / (feedRate * dt)))
            : 0;

    std::cout << "[Engine] Contact LOCKED - cut depth "
              << std::fixed << std::setprecision(3) << currentDepthMm << " mm";
    if (targetDepthMm > 0.0) {
        std::cout << " of requested " << targetDepthMm << " mm"
                  << " (est. " << estimatedRemainingSteps << " physics steps remaining)";
    }
    std::cout << "." << std::endl;
}

void SimulationEngine::handleAirCutting(double& dt) {
    // =====================================================================
    // CONTACT / SURFACE-REACHED CHECK
    // =====================================================================
    // Exit air-cutting mode when:
    //   1. Contact solver reports contacts (physics already running)
    //   2. Solver AABBs overlap (gap ≤ 0) — tool reached workpiece surface
    //
    // When gap reaches 0 during the physics-skipped approach, dt falls back
    // to the CFL-limited value from computeAdaptiveTimeStep().  Physics then
    // runs at CFL dt (stable) for the remaining step budget.
    // =====================================================================
    if (m_contactEngaged) {
        m_skipPhysicsThisStep = false;
        return;
    }

    double gap = computeAirGap();
    const bool contactReported =
        (m_contactSolver && m_contactSolver->getContactCount() > 0);

    if (contactReported) {
        m_contactEngaged = true;
        m_skipPhysicsThisStep = false;
        for (const auto& solver : m_solvers) {
            if (auto* mpm = dynamic_cast<MPMSolver*>(solver.get())) {
                double mpmDt = mpm->getStableTimeStep();
                const auto& sp = m_config.getSimulation();
                dt = std::max(sp.minTimeStep, std::min(sp.maxTimeStep, mpmDt));
                break;
            }
        }
        recordContactEngagement(dt);
        std::cout << "[Engine] Contact reported at step "
                  << m_currentStep << " (T=" << m_currentTime << "s). "
                  << "Physics engaged at dt=" << std::scientific << dt << "s."
                  << std::endl;
        return;
    }

    if (gap <= 0) {
            m_contactEngaged = true;
            m_skipPhysicsThisStep = false;
            // Tool has reached the workpiece.
            // Use MPM's CFL dt (~3.4e-7s) for the physics phase, NOT FEM's
            // overly-conservative 1e-8s (FEM is kinematic, dt limit doesn't
            // apply).  At CFL dt, 600k remaining steps give ~0.2s of cutting
            // (1.8mm penetration) — enough for meaningful stress results.
            for (const auto& solver : m_solvers) {
                if (auto* mpm = dynamic_cast<MPMSolver*>(solver.get())) {
                    double mpmDt = mpm->getStableTimeStep();
                    const auto& sp = m_config.getSimulation();
                    dt = std::max(sp.minTimeStep, std::min(sp.maxTimeStep, mpmDt));
                    break;
                }
            }
            recordContactEngagement(dt);
            std::cout << "[Engine] Tool reached workpiece at step "
                      << m_currentStep << " (T=" << m_currentTime << "s). "
                      << "Physics engaged at dt=" << std::scientific << dt << "s."
                      << std::endl;
        return;
    }
    
    // =========================================================================
    // AIR-CUTTING HANDOFF THRESHOLD
    // =========================================================================
    // Use a FIXED small handoff gap (100μm = 0.1mm).
    // 
    // Previous approach used contactRadius × 1.2 as the handoff, but this
    // fails when the SPH auto-scaler increases spacing (e.g., large workpieces
    // with limited particle budgets). Auto-scaled contactRadius can reach
    // 1.8mm, making the handoff at 2.15mm — larger than the air-cutting
    // macro-dt tier boundaries, so the tool would never close the gap.
    //
    // Configurable gap from simulation_parameters.handoff_gap_mm (default 0.1mm).
    // Small enough for genuine proximity, large enough to avoid overshoot at 1ms
    // macro-dt (at 525mm/min, 1ms = 8.75μm per step).
    // =========================================================================
    const double handoffGap = m_config.getSimulation().handoffGapMm / 1000.0;
    
    if (gap < handoffGap) {
        // =====================================================================
        // PROXIMITY APPROACH (gap < handoffGap):
        // Use graduated dt to close the remaining gap quickly.
        // Physics skipped — MPM unstable at dt > CFL limit.
        // =====================================================================
        double proximityDt = 1.0e-6;  // default 1μs
        if (gap > 2.0e-5) {
            proximityDt = 1.0e-3;     // gap > 20μm → 1ms (8.75μm/step)
        } else if (gap > 5.0e-6) {
            proximityDt = 1.0e-4;     // gap > 5μm → 100μs (0.875μm/step)
        }
        dt = proximityDt;
        m_skipPhysicsThisStep = true;
        {
            for (auto& solver : m_solvers) {
                if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                    fem->advanceTime(dt);
                    break;
                }
            }
        }
        return;
    }

    if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
        // =====================================================================
        // G-CODE APPROACH ACCELERATION
        // =====================================================================
        // The SPH CFL condition forces dt ≈ 1e-7s, but the G-code toolpath
        // needs seconds to execute. At dt=1e-7, the tool would need BILLIONS
        // of steps to traverse the air gap — never reaching the workpiece.
        //
        // FIX: During air-cutting, use a MACRO timestep and skip ALL physics.
        // This advances only the G-code interpolator and FEM tool mesh,
        // closing the gap in ~20 steps instead of 10 billion.
        //
        // Graduated dt tiers based on ABSOLUTE gap distance:
        //   > 5mm   → 100ms/step  (closes 50mm gaps in ~57 steps at 525mm/min)
        //   > 2mm   →  50ms/step  (closes 3mm in ~7 steps)
        //   > 0.5mm →  10ms/step  (fine approach)
        //   > 0.2mm →   1ms/step  (micro approach — ~8.75μm/step at 525mm/min)
        //   else    →  return to physics (handled by handoffGap check above)
        // =====================================================================
        double macroDt;
        
        if (gap > 0.005) {
            macroDt = 0.1;       // 100ms for huge gaps (>5mm)
        } else if (gap > 0.002) {
            macroDt = 0.05;      // 50ms for large gaps (2-5mm)
        } else if (gap > 0.0005) {
            macroDt = 0.01;      // 10ms for medium gaps (0.5-2mm)
        } else if (gap > handoffGap + 0.00005) {
            macroDt = 0.001;     // 1ms for fine approach (just above handoff)
        } else {
            // =====================================================================
            // Close the [handoffGap, 0.15mm] dead zone with graduated dt
            // (physics skipped — MPM unstable at dt > CFL).
            // =====================================================================
            double closeDt = 1.0e-6;
            if (gap > 2.0e-5) {
                closeDt = 1.0e-3;    // gap > 20μm → 1ms
            } else if (gap > 5.0e-6) {
                closeDt = 1.0e-4;    // gap > 5μm → 100μs
            }
            dt = closeDt;
            m_skipPhysicsThisStep = true;
            {
                for (auto& solver : m_solvers) {
                    if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                        fem->advanceTime(dt);
                        break;
                    }
                }
            }
            return;
        }
        
        dt = macroDt;

        // Skip ALL physics during air-cutting:
        // - SPH: workpiece particles are stationary, no need to compute forces
        // - FEM: tool is kinematic (G-Code driven), no internal elastic forces needed
        // - CFD: no thermal sources during approach
        m_skipPhysicsThisStep = true;

        // Fix L: Advance FEM internal clock to stay in sync.
        // During air-cutting, step() is skipped, so FEM's m_currentTime freezes.
        // When contact begins, FEM would have wrong timestamps for all
        // time-dependent computations (thermal decay, wear rate, etc.).
        for (auto& solver : m_solvers) {
            if (auto* fem = dynamic_cast<FEMSolver*>(solver.get())) {
                fem->advanceTime(macroDt);
                break;
            }
        }
        
        // Fix 4: member variable replaces static local (prevents stale state across runs)
        if (m_currentTime - m_airCutLogTime_gcode > 0.01 || m_currentStep < 10) {
            std::cout << "[Engine] Air-cutting (gap: " << std::fixed << std::setprecision(2)
                      << gap * 1000.0 << " mm, dt=" << std::scientific << dt 
                      << " @ T=" << std::fixed << std::setprecision(4) 
                      << m_currentTime << " s)" << std::endl;
            m_airCutLogTime_gcode = m_currentTime;
        }
        return;
    }

    // Legacy path: no G-Code loaded, manual kinematics only.
    if (gap > 0.0005) {
        // Fix 4: member variable replaces static local (prevents stale state across runs)
        if (m_currentTime - m_airCutLogTime_legacy > 0.1) {
            std::cout << "[Engine] Air-cutting (gap: " << gap * 1000.0
                      << " mm). Accelerating..." << std::endl;
            m_airCutLogTime_legacy = m_currentTime;
        }
        double baseDt = computeAdaptiveTimeStep();
        dt = std::min(baseDt * 5.0, gap / 10.0);
    }
}

} // namespace edgepredict
