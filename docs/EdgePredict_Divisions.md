# EdgePredict — Product Divisions & Licensing Architecture

**OMR Systems | Confidential**  
**Date**: March 2026

---

## 1. Executive Summary: "One Engine, Multiple UI Editions"

From a software engineering perspective, **EdgePredict uses a single, unified Simulation Engine (v5+)**. We do not maintain four different simulators. The core engine contains the physics for every single strategy.

However, from a **commercial and UI perspective**, a customer who only makes drills doesn't want an interface cluttered with grinding wheels and turning inserts. 

Therefore, we use **UI-Level Gating (License Tiers)**. A user downloads the "EdgePredict" software, but their license key unlocks a specific UI division. 

### Proposed Product Division Names

1. **EdgePredict Rotary™** (Round Tools: Milling, Drilling, etc.)
2. **EdgePredict Turning™** (Stationary Inserts: Lathes, Boring, etc.)
3. **EdgePredict Grinding™** (Abrasives: Surface, Cylindrical, etc.)
4. **EdgePredict Advanced™** (Specialty: EDM, Laser, Broaching)

---

## 2. Platform Architecture

```text
 ┌───────────────────────────────────────────────────────────────┐
 │                   SINGLE DOWNLOAD / INSTALL                   │
 │                                                               │
 │   ┌─────────────────────── UI LAYER ──────────────────────┐   │
 │   │ (Interface changes & tools unlock based on License Key) │   │
 │   │                                                       │   │
 │   │  [EdgePredict]   [EdgePredict]   [EdgePredict]        │   │
 │   │  [  ROTARY   ]   [  TURNING  ]   [  GRINDING ]  ...   │   │
 │   └──────────────────────────┬────────────────────────────┘   │
 │                              │                                │
 │   ┌──────────────────────────▼────────────────────────────┐   │
 │   │               EDGEPREDICT CORE ENGINE                 │   │
 │   │                 (All Physics Built-In)                │   │
 │   │    SPH Solver │ FEM Solver │ CFD │ Contact │ Wear     │   │
 │   └───────────────────────────────────────────────────────┘   │
 └───────────────────────────────────────────────────────────────┘
```

---

## 3. The Four Divisions In Detail

### 🔵 1. EdgePredict Rotary™
**Target Customers**: OSG, Emuge-Franken, Guhring, Dormer Pramet
**Core Focus**: Rotating tools, stationary workpiece. 

| UI Unlocks | Sub-Operations / Tool Types |
|------------|-----------------------------|
| **Drilling** | Twist Drill, Indexable Drill, Step Drill, Center Drill |
| **Milling**  | End Mill, Face Mill, Ball Nose, Chamfer Mill, T-Slot |
| **Reaming**  | Straight Flute Reamer, Spiral Flute Reamer |
| **Threading**| Thread Mill, Tap (Cut & Form) |
| **Boring**   | Fine Boring Heads (Rotating type) |

---

### 🟠 2. EdgePredict Turning™
**Target Customers**: Iscar, Tungaloy, Kyocera, Sandvik CoroTurn
**Core Focus**: Rotating workpiece, stationary tool feeding into it.

| UI Unlocks | Sub-Operations / Tool Types |
|------------|-----------------------------|
| **OD Turning**| External cylindrical (CNMG, DNMG, WNMG inserts) |
| **ID Boring** | Internal hole machining (CCMT, DCMT inserts) |
| **Face Turn** | Machining flat ends, varying surface speed computations |
| **Grooving**  | Deep slotting, circlip grooves (Plunge mechanics) |
| **Parting Off**| Severing the finished part |
| **Lathe Thread**| External/Internal single-point threading (16ER/IR inserts) |

---

### 🟢 3. EdgePredict Grinding™
**Target Customers**: Norton, 3M, Tyrolit, Winterthur
**Core Focus**: Multi-grain abrasive scratching, friction-dominated heat.

| UI Unlocks | Sub-Operations / Tool Types |
|------------|-----------------------------|
| **Surface**   | Flat plane finishing |
| **Cylindrical**| Outside/Inside diameter precision finishing |
| **Creep Feed**| Deep, slow passes for aerospace alloys (e.g., turbine roots) |
| **Tool Grind**| Simulating the manufacturing of drills/mills themselves |

---

### 🔴 4. EdgePredict Advanced™
**Target Customers**: GF Machining Solutions, Makino, Trumpf
**Core Focus**: Non-cutting physics, thermal/fluid ablation.

| UI Unlocks | Sub-Operations / Tool Types |
|------------|-----------------------------|
| **EDM**        | Wire EDM, Sinker EDM (Spark Erosion) |
| **Laser**      | Laser Cutting, Laser Drilling (Thermal Ablation) |
| **Waterjet**   | Abrasive Waterjet (High-pressure fluid/particle dynamics) |
| **Broaching**  | Multi-tooth linear shearing (Keyways, Splines) |
| **Hobbing**    | Complex synchronized generation of gear teeth |

---

## 4. UI Gating Workflows

When a user launches the software, the UI automatically adapts:

1. **Verify License**: e.g., "License: EdgePredict Turning"
2. **Hide Irrelevant Modules**: The "Tool Library" hides all End Mills, Drills, and Grinding Wheels. It only shows Turning Inserts and Boring Bars.
3. **Filter Parameters**: The setup screen asks for "Lathe RPM" instead of "Spindle RPM".
4. **Upsell Banner**: *(Optional)* "Working on a mill-turn machine? Upgrade to the **EdgePredict Complete** bundle to unlock Rotary tools."

## 5. Engineering Advantage

By keeping everything in **one engine**:
1. **Zero Duplication**: A physics patch to SPH heat generation fixes bugs in both Turning and Milling instantly.
2. **Easy Packaging**: Only one `.exe` and one Docker container to maintain and distribute.
3. **Seamless Upgrades**: When a user buys a new division, they just enter a text code. No new downloads needed. The UI simply un-hides buttons.
