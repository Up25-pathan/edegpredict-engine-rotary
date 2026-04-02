# EdgePredict Engine — Future Advancement Roadmap

**Document Version**: 1.0  
**Date**: February 2026  
**Status**: Planning (Post-Engine Validation)  
**Author**: OMR Systems Engineering

---

## Current State (v5 — What We Have Today)

| Component | Status | Technology |
|-----------|--------|------------|
| SPH Workpiece Solver | ✅ Complete | CUDA, Leapfrog, Spatial Hash |
| FEM Tool Solver  | ✅ Complete | CUDA, Mass-Spring, Usui Wear |
| Contact Detection | ✅ Complete | CUDA, Penalty Method |
| CFD Coolant | ✅ Complete | Finite Difference Grid |
| Machining Strategies | ✅ Complete | Milling, Turning, Drilling |
| Damage & Chip Separation | ✅ Complete | Johnson-Cook Failure |
| Performance Optimizations | ✅ Complete | Streams, Pinned Memory |

> **Before implementing ANY advancement below, the base engine must be validated through Docker build + test simulations.**

---

## Advancement #1: ML Surrogate Model ⭐ (Highest Impact)

### What Is It?
A **neural network trained on simulation results** that can predict output (forces, temperature, wear) in **5 milliseconds** instead of 30 minutes.

### Why It Matters

```
TODAY:
  User changes RPM → clicks "Run" → waits 30 min → sees result

WITH SURROGATE:
  User drags RPM slider → result updates LIVE (like a video game)
```

### How It Works (Step by Step)

**Step 1: Generate Training Data**
Run 500-2000 simulations with different parameters. Record inputs and outputs into a CSV file.

```
Input: Speed=100, Feed=0.1, DOC=1.0, Material=Ti-6Al-4V
Output: Force=450N, Temp=520°C, Wear=0.12mm, Roughness=1.2µm
```

**Step 2: Train Neural Network (Python + PyTorch)**
Feed the CSV into a neural network. The network learns the patterns: *"when speed goes up, force drops but temperature rises."*

```
Architecture: 12 inputs → 128 neurons → 256 neurons → 128 neurons → 8 outputs
Training: Adam optimizer, 500 epochs, ~2 hours on GPU
```

**Step 3: Export to ONNX**
Convert the trained PyTorch model to ONNX format (a universal ML model format that C++ can load).

**Step 4: C++ Inference**
Load the ONNX model in the EdgePredict engine using ONNX Runtime. Now every prediction takes ~5ms.

**Step 5: UI Integration**
Connect the surrogate to UI sliders. As user drags a slider, call `surrogate.predict()` and update the dashboard instantly.

### Implementation Timeline
| Phase | Duration | What |
|-------|----------|------|
| Data Generation | 2 weeks | Run 500+ simulations, export CSV |
| Training Pipeline | 2 weeks | Python script, PyTorch model |
| C++ Integration | 2 weeks | ONNX Runtime, preprocessor |
| UI Connection | 2 weeks | API for sliders, live charts |
| Validation | 2 weeks | Compare ML vs full sim accuracy |
| **Total** | **10 weeks** | |

### Dependencies
- PyTorch 2.0+, ONNX Runtime 1.16+, scikit-learn 1.3+

### Detailed Design Document
📄 [ML_Surrogate_Model_Design.md](file:///d:/EDGE_PREDICT-simulation%20software/edgepredict-engine-v3-main/docs/ML_Surrogate_Model_Design.md) (1,301 lines, full code)

---

## Advancement #2: Digital Twin Mode

### What Is It?
Connect EdgePredict to a **real CNC machine** via OPC-UA or MTConnect protocol. The simulation runs in parallel with the actual machining, comparing predicted vs actual forces/temperatures in real-time.

### Why It Matters
- Detect tool breakage **before** it happens
- Validate simulation accuracy against real data
- Provide real-time "what-if" analysis during production

### How It Works
```
CNC Machine (Real)          EdgePredict (Virtual)
    │                              │
    ├── Spindle Load: 45% ────────►├── Predicted Load: 47%  ✅ Match
    ├── Vibration: 0.3g ──────────►├── Predicted: 0.28g     ✅ Match
    ├── Tool Temp: 380°C ─────────►├── Predicted: 520°C     ⚠️ Sensor error?
    │                              │
    └── ALARM if deviation > 20%
```

### Implementation Steps
1. Add OPC-UA client library (open62541)
2. Create `DigitalTwinManager` class
3. Build real-time comparison dashboard
4. Add alert system for deviations

### Timeline: ~12 weeks | Priority: Medium

---

## Advancement #3: Multi-GPU Scaling

### What Is It?
Split the simulation across **2-4 GPUs** on the same machine. Each GPU handles a portion of the particles/nodes.

### Why It Matters
- 2x-4x faster simulations
- Handle 10M+ particles (currently limited to ~5M on single GPU)
- Enable full-workpiece simulation instead of just the cutting zone

### How It Works
```
GPU 0: SPH Particles 0 - 2,500,000      (Left half of workpiece)
GPU 1: SPH Particles 2,500,001 - 5,000,000  (Right half of workpiece)
       ↕ Exchange boundary particles via NVLink/PCIe
GPU 2: FEM Nodes (entire tool)
GPU 3: CFD Grid (coolant flow)
```

### Implementation Steps
1. Domain decomposition algorithm
2. Inter-GPU communication (NCCL or cudaMemcpyPeer)
3. Load balancing (particles move between GPUs as tool advances)
4. Synchronization barriers between timesteps

### Existing Foundation
📄 [MultiGPUManager.cuh](file:///d:/EDGE_PREDICT-simulation%20software/edgepredict-engine-v3-main/include/MultiGPUManager.cuh) — Header already exists

### Timeline: ~8 weeks | Priority: Medium-Low

---

## Advancement #4: Adaptive Mesh Refinement (AMR)

### What Is It?
Dynamically **add more particles** near the cutting zone and **remove particles** far away. This keeps the simulation accurate where it matters and fast everywhere else.

### Why It Matters
- 3x-5x faster without losing accuracy
- Automatically handles different workpiece sizes
- No manual "zone setup" needed

### How It Works
```
Before AMR:                    After AMR:
● ● ● ● ● ● ● ● ●           ·   ·   ·   ·   ·
● ● ● ● ● ● ● ● ●           · · · · · · · · ·
● ● ● ● ● ● ● ● ●           · ● ● ● ● ● ● · ·
● ● ● ● ● ● ● ● ●    →      · ●●●●●●●●●●● ·   ← Dense near tool
● ● ● ●[TOOL]● ● ●           · ●●●●[TOOL]●●●● ·
● ● ● ● ● ● ● ● ●           · ●●●●●●●●●●● ·
● ● ● ● ● ● ● ● ●           · ● ● ● ● ● ● · ·
● ● ● ● ● ● ● ● ●           · · · · · · · · ·
                              ·   ·   ·   ·   ·   ← Sparse far away
```

### Implementation Steps
1. Particle splitting kernel (1 particle → 4 children near tool)
2. Particle merging kernel (4 particles → 1 parent far from tool)
3. Conservation enforcement (mass, momentum, energy must be preserved)
4. Smoothing length adaptation (h varies with local density)

### Timeline: ~6 weeks | Priority: High

---

## Advancement #5: Cloud HPC Mode

### What Is It?
Run EdgePredict on **cloud GPU instances** (AWS, Azure, GCP) for users who don't have local NVIDIA GPUs.

### Why It Matters
- Removes hardware barrier — anyone can use EdgePredict
- Scale to massive simulations (A100/H100 GPUs in cloud)
- SaaS business model opportunity

### How It Works
```
User's Browser/App → REST API → Cloud Worker (GPU VM) → Results back
                                     │
                                     ├── AWS: p4d.24xlarge (8x A100)
                                     ├── Azure: NC96ads (4x A100)
                                     └── GCP: a2-ultragpu-8g (8x A100)
```

### Implementation Steps
1. Dockerize engine (already done ✅)
2. Build REST API layer (FastAPI or gRPC)
3. Job queue system (Redis + Celery)
4. Result streaming (WebSocket for live progress)
5. Authentication & billing integration

### Timeline: ~10 weeks | Priority: Medium

---

## Advancement #6: AI Material Database

### What Is It?
Use a **Large Language Model** to automatically extract material properties from research papers, datasheets, and databases like MatWeb.

### Why It Matters
- Currently: User must find Johnson-Cook parameters manually (hours of searching)
- With AI: User types "Inconel 625" → system finds and fills all parameters automatically

### How It Works
```
User types: "Inconel 625 at 800°C"
        │
        ▼
   LLM Agent searches:
   ├── MatWeb database
   ├── Research papers (Google Scholar)
   └── Internal simulation results
        │
        ▼
   Returns:  A=648 MPa, B=1320 MPa, n=0.52, C=0.006, m=1.1
             Density=8440 kg/m³, Cp=410 J/kgK, k=9.8 W/mK
             Confidence: 92% (3 sources agree)
```

### Implementation Steps
1. Build material property database (PostgreSQL)
2. Web scraper for MatWeb/research papers
3. LLM-based extraction pipeline
4. Confidence scoring (how many sources agree)
5. UI integration (auto-complete in material dropdown)

### Timeline: ~8 weeks | Priority: Low (nice-to-have)

---

## Advancement #7: Real-Time 3D Visualization

### What Is It?
A **WebGL/Vulkan-based 3D viewer** that shows the simulation running live — particles flowing, chips forming, temperature fields glowing.

### Why It Matters
- Currently: Results are post-processed in ParaView (manual, slow)
- With live viewer: See the simulation as it runs, rotate/zoom in real-time

### How It Works
```
EdgePredict Engine (CUDA) ──stream──► Render Engine (Vulkan/WebGL)
        │                                      │
        ├── Particle positions                  ├── Point cloud render
        ├── Temperature field                   ├── Color-mapped heatmap
        ├── Tool mesh                          ├── Wireframe overlay
        └── Contact forces                     └── Vector arrows
```

### Existing Foundation
📄 [live_viewer.py](file:///d:/EDGE_PREDICT-simulation%20software/edgepredict-engine-v3-main/live_viewer.py) — Basic Python viewer already exists

### Implementation Steps
1. CUDA-OpenGL interop (zero-copy render from GPU simulation data)
2. WebGL/Three.js frontend (browser-based)
3. WebSocket streaming (position data at 30 FPS)
4. Color mapping (temperature → red gradient, stress → blue gradient)
5. VTK export for ParaView compatibility

### Timeline: ~8 weeks | Priority: High

---

## Advancement #8: Smart Parameter Recommendation

### What Is It?
An **AI assistant** that recommends optimal machining parameters based on the selected tool, material, and operation type — before the user even runs a simulation.

### Why It Matters
- New users don't know what RPM/feed to use
- Experienced users can validate their intuition
- Reduces simulation iterations from 10 to 2-3

### How It Works
```
User selects:  Tool = Sandvik DE10, Material = Ti-6Al-4V, Op = Drilling
        │
        ▼
   Recommendation Engine:
   ├── Sandvik catalog data → RPM: 800-1200, Feed: 0.15-0.25
   ├── Historical simulations → Best at RPM=950, Feed=0.18
   └── ML Surrogate → Predicted Force=380N, Temp=510°C ✅ Safe
        │
        ▼
   Shows: "Recommended: RPM=950, Feed=0.18 mm/rev
           Expected Force: 380N, Temperature: 510°C
           Tool Life: ~45 minutes"
```

### Implementation Steps
1. Build recommendation database from tool catalogs
2. Link to ML Surrogate for instant validation
3. Rule engine for safety constraints
4. UI integration (auto-fill with "Recommended" badge)

### Timeline: ~6 weeks | Priority: Medium (after ML Surrogate)

---

## Advancement #9: Extended Machining Strategies ⭐ (High Priority)

### What Is It?
Expand beyond Drilling, Milling, and Turning to cover **all standard CNC operations**. Each new strategy reuses the existing SPH/FEM/Contact physics but implements a different **tool path and geometry**.

### Currently Supported (v5)
| Strategy | File | Tool Type |
|----------|------|-----------|
| Turning | `TurningStrategy.cpp` | Single-point insert |
| Milling | `MillingStrategy.cpp` | End mill / Face mill |
| Drilling | `DrillingStrategy.cpp` | Twist drill / Indexable |

### New Strategies to Add

#### 9a. Boring Strategy
**What**: Internal diameter machining — like turning, but inside a hole.
**Tool**: Boring bar with indexable insert.
**Physics**: Same as turning, but forces push outward (radial). Tool deflection is critical because boring bars are long and thin.

```
                    ┌─────────────────┐
   Boring Bar ═════╪═►  [INSERT]      │  ← Workpiece (hole)
                    │    ↑ cutting     │
                    │    radial force  │
                    └─────────────────┘
```

**Implementation**: Copy `TurningStrategy.cpp`, invert radial direction, add tool deflection compensation.
**Effort**: ~1 week

---

#### 9b. Reaming Strategy
**What**: Precision hole finishing after drilling. Removes 0.1-0.5mm material for tight tolerances (H7).
**Tool**: Multi-flute reamer (4-8 flutes).
**Physics**: Very light cuts, low forces. Surface quality and roundness are critical outputs.

```
   Reamer (6 flutes)
      │╲╲╲╲╲╲│
      │╱╱╱╱╱╱│  ← existing drilled hole
      │╲╲╲╲╲╲│     removes ~0.2mm per side
      │╱╱╱╱╱╱│
```

**Implementation**: Similar to drilling but with multi-flute engagement and much smaller DOC.
**Effort**: ~1.5 weeks

---

#### 9c. Threading Strategy
**What**: Cutting internal or external threads (M8, M10, etc.).
**Tool**: Single-point thread insert OR thread mill OR tap.
**Physics**: Helical tool path. Multiple passes at increasing depth. V-shaped chip geometry.

```
   Single-Point Threading (Lathe):
   Pass 1: ─╲─────╲─────╲─────    depth = 0.2mm
   Pass 2: ──╲────╲────╲────      depth = 0.4mm
   Pass 3: ───╲───╲───╲───        depth = 0.6mm (final)
```

**Implementation**: New `ThreadingStrategy.cpp` with helical path generator, multi-pass logic, and thread pitch parameter.
**Effort**: ~2 weeks

---

#### 9d. Grooving & Parting Strategy
**What**: Cutting narrow slots (grooving) or cutting off finished parts (parting).
**Tool**: Narrow grooving insert (2-6mm width).
**Physics**: Plunge cutting — tool moves radially into workpiece. Extreme chip compression in narrow slot. High risk of vibration (chatter).

```
   Parting Off:
   ┌──────────────┐
   │  Workpiece   │
   │    ┌───┐     │
   │    │INS│ ← plunge radially
   │    └───┘     │
   │              │
   └──────────────┘
```

**Implementation**: New `GroovingStrategy.cpp` with radial plunge path, chip breaking logic, and chatter detection.
**Effort**: ~2 weeks

---

#### 9e. Face Turning Strategy
**What**: Machining a flat face on the end of a cylindrical workpiece.
**Tool**: Same turning insert, but tool moves radially (X-axis) instead of axially (Z-axis).
**Physics**: Cutting speed changes continuously (maximum at outer diameter, zero at center). This means force and temperature vary across the pass.

```
   Face Turning:
   ┌─────────┐
   │         │
   │  ←──────│ INSERT  (moves toward center)
   │         │
   └─────────┘
   RPM constant, but surface speed drops toward center
```

**Implementation**: Modify `TurningStrategy.cpp` to swap feed direction from Z to X. Add variable surface speed calculation.
**Effort**: ~1 week

---

### Combined Timeline for All New Strategies

| Strategy | Base Code | New File | Effort |
|----------|-----------|----------|--------|
| Boring | Copy TurningStrategy | `BoringStrategy.cpp` | 1 week |
| Face Turning | Modify TurningStrategy | `FaceTurningStrategy.cpp` | 1 week |
| Reaming | Copy DrillingStrategy | `ReamingStrategy.cpp` | 1.5 weeks |
| Threading | New implementation | `ThreadingStrategy.cpp` | 2 weeks |
| Grooving/Parting | New implementation | `GroovingStrategy.cpp` | 2 weeks |
| **Total** | | | **~7.5 weeks** |

### What Changes in the Engine
1. New `.cpp` files implementing `IMachiningStrategy` interface
2. Update `Config.cpp` to recognize new `machining_type` values
3. Update `SimulationEngine` factory to instantiate new strategies
4. New input templates for each operation type

---

## Priority Roadmap Summary

```
                    2026                           2027
            Q1          Q2          Q3          Q4          Q1
            ├───────────┼───────────┼───────────┼───────────┤
Engine:     ████████                                        
Validate     ▲ NOW                                          
                                                            
#9 Strat:       ████████                                    
#4 AMR:         ██████                                      
#1 ML:              ██████████                              
#7 3D View:             ████████                            
#8 Smart:                   ██████                          
#2 Digital:                     ████████████                
#5 Cloud:                           ██████████              
#3 Multi-GPU:                           ████████            
#6 AI MatDB:                                ████████        
```

| # | Feature | Impact | Effort | Priority |
|---|---------|--------|--------|----------|
| 9 | Extended Machining Strategies | ⭐⭐⭐⭐⭐ | 7.5 weeks | 🔴 Do First |
| 4 | Adaptive Mesh Refinement | ⭐⭐⭐⭐ | 6 weeks | 🔴 Do First |
| 1 | ML Surrogate Model | ⭐⭐⭐⭐⭐ | 10 weeks | 🔴 Do First |
| 7 | Real-Time 3D Visualization | ⭐⭐⭐⭐ | 8 weeks | 🟡 Do Next |
| 8 | Smart Parameter Recommendation | ⭐⭐⭐ | 6 weeks | 🟡 Do Next |
| 2 | Digital Twin Mode | ⭐⭐⭐⭐⭐ | 12 weeks | 🟢 Do Later |
| 5 | Cloud HPC Mode | ⭐⭐⭐⭐ | 10 weeks | 🟢 Do Later |
| 3 | Multi-GPU Scaling | ⭐⭐⭐ | 8 weeks | 🔵 Optional |
| 6 | AI Material Database | ⭐⭐ | 8 weeks | 🔵 Optional |

---

> **Remember**: None of these should be started until the base engine passes Docker build + test simulation validation. Get the foundation right first, then add these features one at a time.
