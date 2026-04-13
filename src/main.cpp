/**
 * @file main.cpp
 * @brief EdgePredict Engine v4 – Main Entry Point
 *
 * Fixes applied vs previous version:
 *  1. CFD solver: ownership correctly transferred to engine via addSolver()
 *     AFTER the non-owning pointer is stored in strategy/engine.  The order
 *     was correct before but is now explicitly commented.
 *  2. Strategy: connectSolvers() is called with real solver pointers (not null).
 *     Pointers remain valid until end of main() because engine owns the solvers
 *     in its vector.
 *  3. JSON key access guarded with contains() throughout to prevent null-JSON
 *     exceptions on configs that omit optional sections.
 *  4. IPC mode: stdout redirect is restored cleanly before return.
 *  5. ToolCoatingModel lifetime: unique_ptr lives for entire main() scope,
 *     outlasting engine.run().
 */

#include "SimulationEngine.h"
#include "Config.h"
#include "GeometryLoader.h"
#include "CoordinateSystem.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include "ToolCoatingModel.cuh"
#include "OptimizationManager.h"
#include "SurfaceRoughnessPredictor.h"
#include "ResidualStressAnalyzer.h"
#include "IMachiningStrategy.h"
#include "VTKExporter.h"
#include "StdoutIPCExporter.h"
#include "CudaUtils.cuh"

#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>
#include <sstream>

using namespace edgepredict;

// ---------------------------------------------------------------------------
// Banner / usage
// ---------------------------------------------------------------------------

void printBanner() {
    std::cout << R"(
  ______    _            _____              _ _      _   
 |  ____|  | |          |  __ \            | (_)    | |  
 | |__   __| | __ _  ___| |__) | __ ___  __| |_  ___| |_ 
 |  __| / _` |/ _` |/ _ \  ___/ '__/ _ \/ _` | |/ __| __|
 | |___| (_| | (_| |  __/ |   | | |  __/ (_| | | (__| |_ 
 |______\__,_|\__, |\___|_|   |_|  \___|\__,_|_|\___|\__|
               __/ |                                     
              |___/         Engine v4.0 (Clean Arch)     
)" << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <config.json> [options]\n"
              << "\nOptions:\n"
              << "  --help, -h       Show this help\n"
              << "  --gpu-info       Print GPU info and exit\n"
              << "  --validate       Validate config only\n"
              << "  --ipc            IPC mode (stdout = NDJSON)\n"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Parametric tool mesh generation
// Produces ~1 000-node triangulated surface when no STL/STEP file is given.
// ---------------------------------------------------------------------------

void generateParametricToolMesh(Mesh& mesh, const Config& config) {
    mesh.clear();

    const double PI   = 3.14159265358979323846;
    MachiningType type = config.getMachiningType();

    auto addNode = [&](double x, double y, double z) -> int {
        int idx = static_cast<int>(mesh.nodes.size());
        FEMNode node;
        node.position         = Vec3(x, y, z);
        node.originalPosition = Vec3(x, y, z);
        mesh.nodes.push_back(node);
        return idx;
    };

    auto addTri = [&](int a, int b, int c) {
        Triangle tri;
        tri.indices[0] = a;
        tri.indices[1] = b;
        tri.indices[2] = c;
        const Vec3& v0 = mesh.nodes[a].position;
        const Vec3& v1 = mesh.nodes[b].position;
        const Vec3& v2 = mesh.nodes[c].position;
        tri.normal = (v1 - v0).cross(v2 - v0).normalized();
        mesh.triangles.push_back(tri);
    };

    // Tool geometry parameters based on machining type — all from config when available
    double radius, length;
    bool hasCone = false;
    double coneH  = 0.0;
    const auto& j = config.getJson();
    
    switch (type) {
        case MachiningType::DRILLING:
            radius  = 0.005;   // 10mm diameter / 2
            length  = 0.025;   // 25mm flute length
            hasCone = true;
            coneH   = radius / std::tan(59.0 * PI / 180.0);  // 118° point angle
            break;
            
        case MachiningType::BORING: {
            // Boring bar: long thin cylinder (L/D ≈ 4:1 default, configurable)
            double barLenMm = 40.0;
            double barDiaMm = 10.0;
            if (j.contains("machining_parameters")) {
                barLenMm = j["machining_parameters"].value("boring_bar_length_mm", 40.0);
                barDiaMm = j["machining_parameters"].value("boring_bar_diameter_mm", 10.0);
            }
            radius  = (barDiaMm / 2.0) / 1000.0;
            length  = barLenMm / 1000.0;
            hasCone = false;
            break;
        }
        
        case MachiningType::REAMING: {
            // Reamer: flat-bottom multi-flute cylinder
            double reamDiaMm = 10.0;
            if (j.contains("machining_parameters")) {
                reamDiaMm = j["machining_parameters"].value("reamer_diameter_mm", 10.0);
            }
            radius  = (reamDiaMm / 2.0) / 1000.0;
            length  = 0.020;  // 20mm flute length (typical)
            hasCone = false;  // Flat bottom with chamfer entry (no drill point)
            break;
        }
        
        case MachiningType::THREADING: {
            // Thread mill: short body with V-profile tip
            double majorDiaMm = 10.0;
            if (j.contains("machining_parameters")) {
                majorDiaMm = j["machining_parameters"].value("thread_major_diameter_mm", 10.0);
            }
            radius  = (majorDiaMm / 2.0 * 0.8) / 1000.0;  // Insert body smaller than thread
            length  = 0.015;  // Short body
            hasCone = true;   // V-shaped tip for thread form
            double formAngle = 60.0;
            if (j.contains("machining_parameters")) {
                formAngle = j["machining_parameters"].value("thread_form_angle_deg", 60.0);
            }
            coneH = radius / std::tan(formAngle / 2.0 * PI / 180.0);
            break;
        }
        
        default:  // MILLING
            radius  = 0.005;   // 10mm diameter / 2
            length  = 0.025;   // 25mm flute length
            hasCone = false;
            break;
    }

    int nCirc  = 32;
    int nAxial = 20;

    // Cylindrical body
    int bodyStart = static_cast<int>(mesh.nodes.size());
    for (int j = 0; j <= nAxial; ++j) {
        double z = -(double)j / nAxial * length;
        for (int i = 0; i < nCirc; ++i) {
            double theta = 2.0 * PI * i / nCirc;
            addNode(radius * std::cos(theta), radius * std::sin(theta), z);
        }
    }
    for (int j = 0; j < nAxial; ++j) {
        for (int i = 0; i < nCirc; ++i) {
            int i2 = (i + 1) % nCirc;
            int a  = bodyStart + j * nCirc + i;
            int b  = bodyStart + j * nCirc + i2;
            int c  = bodyStart + (j + 1) * nCirc + i;
            int d  = bodyStart + (j + 1) * nCirc + i2;
            addTri(a, b, d);
            addTri(a, d, c);
        }
    }

    // Top cap
    if (hasCone) {
        int apex = addNode(0, 0, coneH);
        for (int i = 0; i < nCirc; ++i) {
            int i2 = (i + 1) % nCirc;
            addTri(apex, bodyStart + i, bodyStart + i2);
        }
    } else {
        int center = addNode(0, 0, 0);
        for (int i = 0; i < nCirc; ++i) {
            int i2 = (i + 1) % nCirc;
            addTri(center, bodyStart + i2, bodyStart + i);
        }
    }

    // Bottom cap
    int bottomRing  = bodyStart + nAxial * nCirc;
    int bottomCenter = addNode(0, 0, -length);
    for (int i = 0; i < nCirc; ++i) {
        int i2 = (i + 1) % nCirc;
        addTri(bottomCenter, bottomRing + i, bottomRing + i2);
    }

    // Tool name for logging
    const char* toolName;
    switch (type) {
        case MachiningType::DRILLING:   toolName = "drill";       break;
        case MachiningType::BORING:     toolName = "boring bar";  break;
        case MachiningType::REAMING:    toolName = "reamer";      break;
        case MachiningType::THREADING:  toolName = "thread mill"; break;
        default:                        toolName = "endmill";     break;
    }
    std::cout << "[Main] Generated parametric " << toolName
              << " mesh: " << mesh.nodeCount() << " nodes, "
              << mesh.triangleCount() << " triangles" << std::endl;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string configPath;
    bool gpuInfoOnly   = false;
    bool validateOnly  = false;
    bool ipcMode       = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
        else if (arg == "--gpu-info")             { gpuInfoOnly  = true; }
        else if (arg == "--validate")             { validateOnly = true; }
        else if (arg == "--ipc")                  { ipcMode      = true; }
        else if (arg[0] != '-')                   { configPath   = arg;  }
    }

    if (!ipcMode) printBanner();

    if (gpuInfoOnly) { printGPUInfo(); return 0; }

    if (configPath.empty()) {
        configPath = "input.json";
        if (!ipcMode)
            std::cout << "[Main] Using default config: " << configPath << std::endl;
    }

    if (!ipcMode) { printGPUInfo(); std::cout << std::endl; }

    // IPC mode: redirect std::cout to std::cerr so pure JSON lines reach stdout
    std::streambuf* origCout = nullptr;
    if (ipcMode) {
        origCout = std::cout.rdbuf();
        std::cout.rdbuf(std::cerr.rdbuf());
        std::cerr << "[Main] IPC mode — stdout reserved for NDJSON" << std::endl;
    }

    int exitCode = 0;
    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        // ------------------------------------------------------------------
        // 1. Engine + config
        // ------------------------------------------------------------------
        SimulationEngine engine;
        if (!engine.initialize(configPath)) {
            std::cerr << "[FATAL] Failed to initialize engine\n";
            exitCode = 1;
            goto cleanup;
        }

        if (validateOnly) {
            std::cout << "[Main] Configuration is valid.\n";
            goto cleanup;
        }

        {
            const Config& config = engine.getConfig();

            // ------------------------------------------------------------------
            // 2. Create physics solvers
            // ------------------------------------------------------------------
            auto sphSolver = std::make_unique<SPHSolver>();
            auto femSolver = std::make_unique<FEMSolver>();

            sphSolver->initialize(config);
            femSolver->initialize(config);

            // Workpiece generation — config-driven shape and dimensions
            const auto& sphCfg = config.getSPH();
            double spacing = sphCfg.smoothingRadius * sphCfg.particleSpacingFactor;
            
            // ================================================================
            // === DIGITAL TWIN: Config-driven workpiece sizing ===
            // Replaces hardcoded 20mm radius / 50mm length with config values
            // ================================================================
            const auto& wpGeo = config.getWorkpieceGeometry();
            
            // Determine shape
            std::string workpieceShape = wpGeo.shape;
            bool useCylinder;
            if (workpieceShape == "cylinder") {
                useCylinder = true;
            } else if (workpieceShape == "box") {
                useCylinder = false;
            } else {
                // Auto-detect: drilling/boring/reaming/threading use cylinder
                MachiningType mtype = config.getMachiningType();
                useCylinder = (mtype == MachiningType::DRILLING || 
                               mtype == MachiningType::BORING ||
                               mtype == MachiningType::REAMING || 
                               mtype == MachiningType::THREADING);
            }
            
            // Check for workpiece CAD file first
            bool workpieceFromCAD = false;
            GeometryLoader geoLoader;
            
            if (!config.getFilePaths().workpieceGeometry.empty()) {
                Mesh workpieceMesh;
                if (geoLoader.load(config.getFilePaths().workpieceGeometry, workpieceMesh)) {
                    GeometryLoader::scaleMesh(workpieceMesh, 0.001); // mm → m
                    auto bb = CNCCoordinateSystem::computeBoundingBox(workpieceMesh);
                    std::cout << "[Main] Workpiece CAD bounding box: "
                              << "X[" << bb.min.x*1000 << ", " << bb.max.x*1000 << "] "
                              << "Y[" << bb.min.y*1000 << ", " << bb.max.y*1000 << "] "
                              << "Z[" << bb.min.z*1000 << ", " << bb.max.z*1000 << "] mm" << std::endl;
                    sphSolver->initializeParticleBox(bb.min, bb.max, spacing);
                    workpieceFromCAD = true;
                } else {
                    std::cerr << "[Warning] Workpiece geometry load failed: "
                              << geoLoader.getLastError() << std::endl;
                }
            }
            
            if (!workpieceFromCAD) {
                if (useCylinder) {
                    // Config-driven or default dimensions
                    double radius = wpGeo.radiusMm > 0 ? wpGeo.radiusMm / 1000.0 : 0.003;
                    double length = wpGeo.lengthMm > 0 ? wpGeo.lengthMm / 1000.0 : 0.005;
                    // Center moved down so top of cylinder is at Z=0 (CNC G54 origin)
                    Vec3 center(0.0, 0.0, -length / 2.0); 
                    sphSolver->initializeCylindricalWorkpiece(center, radius, length, spacing, 2);
                    std::cout << "[Main] Workpiece: Cylinder R=" << radius*1000 
                              << "mm L=" << length*1000 << "mm" << std::endl;
                } else {
                    // Config-driven or default box dimensions
                    double sizeX = wpGeo.widthMm > 0 ? wpGeo.widthMm / 1000.0 : 0.008;
                    double sizeY = wpGeo.heightMm > 0 ? wpGeo.heightMm / 1000.0 : 0.008;
                    double sizeZ = wpGeo.depthMm > 0 ? wpGeo.depthMm / 1000.0 : 0.005;
                    // Top of box at Z=0
                    Vec3 workMin(-sizeX/2, -sizeY/2, -sizeZ);
                    Vec3 workMax( sizeX/2,  sizeY/2,  0.0);
                    sphSolver->initializeParticleBox(workMin, workMax, spacing);
                    std::cout << "[Main] Workpiece: Box " << sizeX*1000 << "x" 
                              << sizeY*1000 << "x" << sizeZ*1000 << " mm" << std::endl;
                }
            }

            // Tool geometry
            Mesh toolMesh;

            if (!config.getFilePaths().toolGeometry.empty()) {
                if (geoLoader.load(config.getFilePaths().toolGeometry, toolMesh)) {
                    GeometryLoader::scaleMesh(toolMesh, 0.001); // mm → m
                    std::cout << "[Main] Tool geometry loaded from CAD: "
                              << toolMesh.nodeCount() << " nodes, "
                              << toolMesh.triangleCount() << " triangles" << std::endl;
                } else {
                    // The config validator already confirmed the file exists,
                    // so if we get here it's a format/OCC parsing failure.
                    throw std::runtime_error(
                        "[FATAL] Failed to load tool geometry '" +
                        config.getFilePaths().toolGeometry + "': " +
                        geoLoader.getLastError() +
                        ". Fix the CAD file or check OpenCASCADE installation.");
                }
            } else {
                // No tool geometry path — only reached when Config validation is
                // explicitly relaxed. Generate parametric mesh as debug fallback.
                generateParametricToolMesh(toolMesh, config);
            }
            
            // ------------------------------------------------------------------
            // === DIGITAL TWIN: 4×4 Matrix Coordinate System (MCS/WCS/TCS) ===
            // Replaces ad-hoc alignToTip + applyMachineOffsets with proper
            // CNC coordinate hierarchy using 4×4 homogeneous transforms.
            // ------------------------------------------------------------------
            CNCCoordinateSystem coordSystem;
            
            // Load all WCS offsets from config (G54-G59)
            const auto& mSetup = config.getMachineSetup();
            for (int i = 0; i < CNCCoordinateSystem::NUM_WCS; ++i) {
                Vec3 wcsOff(mSetup.workOffsets[i][0], 
                            mSetup.workOffsets[i][1], 
                            mSetup.workOffsets[i][2]);
                if (wcsOff.lengthSq() > 1e-18) {
                    coordSystem.setWCS(i, wcsOff);
                }
            }
            coordSystem.setActiveWCS(0); // Start with G54
            
            // Auto-detect alignment axis
            int alignAxis = mSetup.alignAxis;
            if (alignAxis == -1) {
                alignAxis = 2; // Z for all rotary tools
            }
            
            // Set TLO
            int tloAxis = (alignAxis != -1) ? alignAxis : 2;
            coordSystem.setToolLengthOffset(mSetup.toolLengthOffset, tloAxis);
            
            // Align tool: tip → origin, then apply compound MCS*WCS*TCS transform
            if (mSetup.autoAlignToolTip) {
                coordSystem.alignToolFromCAD(toolMesh, alignAxis);
            } else {
                // Manual: just apply WCS+TCS without alignment
                Mat4 transform = coordSystem.getToolToWorld();
                CNCCoordinateSystem::transformMesh(toolMesh, transform);
            }
            
            // Store coordinate system in engine for runtime use
            engine.setCoordinateSystem(coordSystem);
            
            // Now that Mesh is perfectly aligned to world, initialize physics!
            femSolver->initializeFromMesh(toolMesh);

            // *** NEW: offset tool tip to be just above the workpiece surface ***
            // Without G-Code the workpiece top is at Z=0 and the tool tip is at Z=0.
            // They are in immediate contact at T=0.  Offset by one smoothing radius
            // so the first timestep produces a well-defined first-contact event.
            {
                // safeOffset must be > contactRadius (= h) to avoid step-0 contact.
                // contactRadius = 1.0 * h, so we offset by 1.5*h to give a clean approach.
                double safeOffset = sphCfg.smoothingRadius * 1.5;
                std::cout << "[Main] Offsetting tool tip by +" << safeOffset * 1000
                          << " mm to avoid T=0 hard contact" << std::endl;
                femSolver->translateMesh(0.0, 0.0, safeOffset);
            }
            
            // ------------------------------------------------------------------
            // === ANCHORED PHYSICS: Phase D Spindle Dynamics Setup ===
            // ------------------------------------------------------------------
            femSolver->setSpindleDynamicsConfig(mSetup.enableSpindleDynamics, 
                                                mSetup.spindleStiffness, 
                                                mSetup.spindleDamping);
            femSolver->initializeDrivenNodes(mSetup.drivenNodeFraction);
            
            double tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ;
            femSolver->getBounds(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ);
            std::cout << "[Main] Final aligned Tool bounds: "
                      << "X[" << tMinX*1000 << ", " << tMaxX*1000 << "] "
                      << "Y[" << tMinY*1000 << ", " << tMaxY*1000 << "] "
                      << "Z[" << tMinZ*1000 << ", " << tMaxZ*1000 << "] mm" << std::endl;
            
            // Link tool mesh to engine for VTK export
            engine.setToolMesh(toolMesh);

            // ------------------------------------------------------------------
            // 3. Contact solver (stack-allocated, outlives engine.run())
            //    Contact radius = 1.5 * smoothing radius for reliable engagement
            // ------------------------------------------------------------------
            ContactSolver contactSolver;
            ContactConfig  contactCfg;
            // contactRadius = 1.0 × h (one smoothing radius = standard SPH contact zone)
            // Using 1.5×h was too large: particles far from the tool still received forces,
            // causing 22 contacts at step 0 and immediate thermal runaway.
            contactCfg.contactRadius = sphCfg.smoothingRadius * 1.0;
            // k = E* × contactRadius  (reduced modulus, Ti-64 vs Carbide)
            // E* ≈ 1/(((1-v1²)/E1) + ((1-v2²)/E2))  ≈ 80 GPa
            // contactRadius ≈ smoothingRadius * 1.5
            double E_reduced = 80e9;
            contactCfg.contactStiffness = E_reduced * sphCfg.smoothingRadius * 1.5;
            contactCfg.frictionCoefficient = 0.3;
            contactSolver.initialize(sphSolver.get(), femSolver.get(), contactCfg);

            // ------------------------------------------------------------------
            // 4. Tool coating model (lifetime spans all of main body)
            // ------------------------------------------------------------------
            auto coatingModel = std::make_unique<ToolCoatingModel>();
            coatingModel->initialize(femSolver->getNodeCount());
            coatingModel->addLayer("TiAlN", 3e-6, 3300.0);
            coatingModel->addLayer("TiN",   1e-6, 2000.0);
            contactSolver.setToolCoatingModel(coatingModel.get());

            engine.setContactSolver(&contactSolver);

            // ------------------------------------------------------------------
            // 5. CFD solver (optional)
            //    Order matters: store raw pointer first, THEN transfer ownership.
            // ------------------------------------------------------------------
            std::unique_ptr<CFDSolverGPU> cfdSolver;
            if (config.getCFD().enabled) {
                try {
                    cfdSolver = std::make_unique<CFDSolverGPU>();
                    cfdSolver->initialize(config);
                    // Store non-owning pointer in engine BEFORE the move
                    engine.setCFDSolver(cfdSolver.get());
                    std::cout << "[Main] CFD coolant solver enabled ("
                              << config.getCFD().coolantType << ")" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[Main] CFD init failed: " << e.what()
                              << " — continuing without coolant" << std::endl;
                    cfdSolver.reset();
                }
            }

            // ------------------------------------------------------------------
            // 6. Machining strategy
            //    createFromConfig() calls initialize(config) internally.
            //    connectSolvers() is called with VALID pointers (solvers still
            //    owned by their unique_ptrs in this scope, not yet moved).
            // ------------------------------------------------------------------
            auto strategy = MachiningStrategyFactory::createFromConfig(config);
            if (strategy) {
                // Raw pointers are valid — unique_ptrs not yet moved to engine
                strategy->connectSolvers(sphSolver.get(), femSolver.get(),
                                         &contactSolver,
                                         cfdSolver ? cfdSolver.get() : nullptr);
                engine.setStrategy(std::move(strategy));
            }

            // ------------------------------------------------------------------
            // 7. Optimisation & analytics
            // ------------------------------------------------------------------
            auto optMgr = std::make_unique<OptimizationManager>();
            optMgr->initialize(config);
            engine.setOptimizationManager(std::move(optMgr));

            engine.setRoughnessPredictor(std::make_unique<SurfaceRoughnessPredictor>());
            engine.setStressAnalyzer(std::make_unique<ResidualStressAnalyzer>());

            // ------------------------------------------------------------------
            // 8. Exporters
            // ------------------------------------------------------------------
            engine.addExporter(
                std::make_unique<VTKExporter>(config.getFilePaths().outputDirectory));

            if (ipcMode)
                engine.addExporter(std::make_unique<StdoutIPCExporter>());

            // ------------------------------------------------------------------
            // 9. Transfer solver ownership to engine
            //    (raw pointers stored in strategy/contactSolver remain valid
            //     because engine's vector is the new owner and lives until
            //     engine is destroyed at end of this scope)
            // ------------------------------------------------------------------
            engine.addSolver(std::move(sphSolver));
            engine.addSolver(std::move(femSolver));
            if (cfdSolver)
                engine.addSolver(std::move(cfdSolver));

            // ------------------------------------------------------------------
            // 10. Run
            // ------------------------------------------------------------------
            std::cout << "\n[Main] Starting simulation...\n" << std::endl;
            engine.run();

            // Final timing / metrics
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = endTime - startTime;
            std::cout << "\n[Main] Total execution time: "
                      << std::fixed << std::setprecision(2)
                      << elapsed.count() << " s\n";
            std::cout << "[Main] Max stress:       "
                      << engine.getMaxStress() / 1e6 << " MPa\n";
            std::cout << "[Main] Max temperature:  "
                      << engine.getMaxTemperature() << " °C\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
        exitCode = 1;
    }

cleanup:
    // Restore stdout if redirected
    if (ipcMode && origCout) std::cout.rdbuf(origCout);
    return exitCode;
}