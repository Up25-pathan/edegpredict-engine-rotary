# EdgePredict Engine v4 — Fix All Critical Issues

## Background

After deep analysis of the entire project codebase (~24 source files, ~8000+ LOC) and thorough review of `analysis_results.md` (which was produced from a previous AI analysis of the simulation engine), I've confirmed **every issue** listed is real and can be traced to specific lines of code. The engine has a clean architecture but produces **meaningless simulation results** because the core physics loop is broken — the tool and workpiece never interact.

## Summary of All Issues

| # | Issue | Severity | Files Affected |
|---|-------|----------|----------------|
| 1 | ContactSolver created but never called in sim loop | 🔴 **CRITICAL** | `main.cpp`, `SimulationEngine.cpp/.h` |
| 2 | MachiningStrategy never created/registered | 🔴 **CRITICAL** | `main.cpp` |
| 3 | `gcode_file` parsed from wrong JSON path | 🟡 **HIGH** | `Config.cpp` |
| 4 | `loadGeometry()` throws on empty tool path | 🟡 **HIGH** | `SimulationEngine.cpp` |
| 5 | FEM stress model is over-simplified (hardcoded L0=1mm) | 🟠 **MEDIUM** | `FEMSolver.cu` |
| 6 | CUDA architecture set to 75, needs 120 for RTX 5060 Ti | 🟡 **HIGH** | `CMakeLists.txt` |
| 7 | Metrics report zeroes (consequence of #1) | 🔴 Auto-fixes with #1 | `SimulationEngine.cpp` |

---

## Proposed Changes

### Phase 1: Core Physics Integration (Issues #1, #2 — MOST CRITICAL)

These two fixes will make the simulation actually **do something real**.

---

#### Issue #1: Wire ContactSolver into SimulationEngine

**Root Cause**: In `main.cpp:134-139`, the `ContactSolver` is created as a **local variable on the stack** and initialized with raw pointers to the solvers. But it is never passed to the `SimulationEngine`, and `SimulationEngine::step()` (line 123-166) never calls `contactSolver.resolveContacts(dt)`. SPH particles and FEM nodes exist in the same space but **never interact**.

**Fix**:

##### [MODIFY] [SimulationEngine.h](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/include/SimulationEngine.h)
- Add `#include "ContactSolver.cuh"` 
- Add `void setContactSolver(ContactSolver* solver);` public method
- Add `ContactSolver* m_contactSolver = nullptr;` private member

##### [MODIFY] [SimulationEngine.cpp](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/src/SimulationEngine.cpp)
- Add `setContactSolver()` implementation
- Modify `step()` method to call `m_contactSolver->resolveContacts(dt)` **between** SPH and FEM solver steps (the critical physics coupling point)
- The contact solver must run AFTER all solvers step (so positions are updated) but BEFORE the next kick — so we add it right after the solver loop at line 157

##### [MODIFY] [main.cpp](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/src/main.cpp)
- After creating the `contactSolver` (line 134-139), register it with the engine via `engine.setContactSolver(&contactSolver)`
- Move the `contactSolver` to be created **before** the solvers are moved into the engine, since we need the raw pointers

---

#### Issue #2: Create and Register MachiningStrategy

**Root Cause**: `MachiningStrategyFactory::create()` and `createFromConfig()` exist and work perfectly (in `MachiningStrategyFactory.cpp:14-45`), but **nobody calls them** in `main.cpp`. Without a strategy, `m_strategy` is null in `SimulationEngine::step()` (line 130), so no kinematics are applied — no tool rotation for milling/drilling, no workpiece rotation for turning.

**Fix**:

##### [MODIFY] [main.cpp](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/src/main.cpp)
- Add `#include "IMachiningStrategy.h"` (which contains the factory)
- After initializing solvers and before running, create the strategy:
  ```cpp
  auto strategy = MachiningStrategyFactory::createFromConfig(config);
  strategy->connectSolvers(sphSolver.get(), femSolver.get(), &contactSolver, nullptr);
  engine.setStrategy(std::move(strategy));
  ```

---

### Phase 2: Config & Loading Fixes (Issues #3, #4)

---

#### Issue #3: gcode_file Parsed from Wrong JSON Path

**Root Cause**: In `Config.cpp:92`:
```cpp
m_filePaths.gcodeFile = j.value("gcode_file", "");
```
This reads from the **top-level** JSON. But in **both** `input.json:24` and `input_turning.json:24`, `gcode_file` is nested under `file_paths`:
```json
"file_paths": {
    "gcode_file": "turning_test.gcode"
}
```

**Fix**:

##### [MODIFY] [Config.cpp](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/src/Config.cpp)
- Move `gcode_file` parsing inside the `file_paths` block (line 83-89):
  ```cpp
  m_filePaths.gcodeFile = fp.value("gcode_file", "");
  ```
- Keep the top-level fallback for backward compatibility:
  ```cpp
  if (m_filePaths.gcodeFile.empty()) {
      m_filePaths.gcodeFile = j.value("gcode_file", "");
  }
  ```

---

#### Issue #4: loadGeometry() Throws on Empty Tool Path

**Root Cause**: In `SimulationEngine.cpp:183-184`:
```cpp
if (filePaths.toolGeometry.empty()) {
    throw std::runtime_error("No tool geometry specified");
}
```
The turning test case (`input_turning.json:22`) has `"tool_geometry": ""`, which would crash immediately.

**Fix**:

##### [MODIFY] [SimulationEngine.cpp](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/src/SimulationEngine.cpp)
- Instead of throwing, log a warning and continue (the FEM solver can still generate default tool geometry — it handles missing meshes by working with node count = 0):
  ```cpp
  if (filePaths.toolGeometry.empty()) {
      std::cout << "[Engine] No tool geometry specified — using generated default" << std::endl;
      return; // Skip file loading, use solver-generated nodes
  }
  ```

---

### Phase 3: Build System Fix (Issue #6)

---

#### Issue #6: CUDA Architecture Mismatch

**Root Cause**: `CMakeLists.txt:13` defaults to `CMAKE_CUDA_ARCHITECTURES=75` (Turing/RTX 20xx). The user's RTX 5060 Ti is Blackwell SM 12.0, requiring arch `120`.

**Fix**:

##### [MODIFY] [CMakeLists.txt](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/CMakeLists.txt)
- Change default from `75` to `120`:
  ```cmake
  set(CMAKE_CUDA_ARCHITECTURES "120")  # RTX 5060 Ti (Blackwell)
  ```
- This can still be overridden via `-DCMAKE_CUDA_ARCHITECTURES=XX` from the command line

---

### Phase 4: FEM Stress Improvement (Issue #5) — Lower Priority

---

#### Issue #5: FEM Stress Model Over-Simplified

**Root Cause**: In `FEMSolver.cu:131-148`, the `computeStressKernel` uses:
```cuda
double L0 = 0.001;  // 1mm hardcoded
double strain = displacement / L0;
double stress = youngsModulus * strain;
```
This is a very rough approximation. It doesn't use actual spring extensions or the real element connectivity.

**Fix** (Phase 4 — can be deferred since the spring-mass model does work correctly for force propagation):

##### [MODIFY] [FEMSolver.cu](file:///c:/EDGEPREDICT/edgepredict-engine-v3-main/src/FEMSolver.cu)
- Use actual spring-based strain tensor instead of displacement-based heuristic
- For each node, average the strain from connected springs
- This requires passing spring connectivity info to the stress kernel (a bigger refactor)

> [!IMPORTANT]
> **I recommend deferring Issue #5** to a later phase. The current simplified model will produce order-of-magnitude correct stress values once contact is working. We should fix Issues #1-4 & #6 first, build, validate, then improve the stress model.

---

## User Review Required

> [!IMPORTANT]
> **Phase ordering**: I plan to fix Issues #1-4 and #6 in this session (code changes only — no compilation possible until CUDA Toolkit + CMake are installed). Issue #5 (FEM stress improvement) would be a separate follow-up. Does this priority make sense?

> [!WARNING]
> **Build environment**: Your system still needs CUDA Toolkit 13.0 and CMake installed before we can compile and test. Should I just fix the code now and we handle the build environment separately?

> [!IMPORTANT]
> **ContactSolver ownership design**: I'm proposing `SimulationEngine` holds a **raw pointer** (non-owning) to the `ContactSolver`, which continues to be owned by `main()`. This is because `ContactSolver` holds raw pointers to SPH and FEM solvers — if we moved ownership, the lifetime management gets complex. Are you OK with this approach, or want me to use `shared_ptr` instead?

---

## Verification Plan

### After Implementing Code Fixes
1. **Static verification**: Trace the call flow from `main() → engine.run() → step()` to confirm `contactSolver.resolveContacts(dt)` is called
2. **Config verification**: Confirm `gcode_file` is parsed from `file_paths` in both JSON inputs
3. **Empty tool path**: Confirm `loadGeometry()` doesn't throw when `tool_geometry` is empty

### After Build Environment is Ready
1. **Compile** with `cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..` + `cmake --build .`
2. **Run minimal test**: `edgepredict-engine.exe input_turning.json` — verify:
   - No crash on empty tool geometry
   - G-code is loaded (check for `"Using G-Code for toolpath control"` message)
   - Strategy is created (check for `"[Factory] Creating TurningStrategy"` message)
   - Contact forces appear non-zero in output
   - Temperature rises above 25°C (friction heat)
   - Stress values are non-zero
3. **Run drilling test**: `edgepredict-engine.exe input.json` — requires tool geometry file
