# ML Surrogate Model for EdgePredict Engine

## Technical Design Document

**Version**: 1.0  
**Date**: January 30, 2026  
**Author**: EdgePredict Development Team  
**Status**: Future Implementation (Post Engine Validation)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [What is a Surrogate Model](#3-what-is-a-surrogate-model)
4. [How It Works](#4-how-it-works)
5. [Architecture](#5-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference System](#7-inference-system)
8. [Integration with EdgePredict](#8-integration-with-edgepredict)
9. [Use Cases](#9-use-cases)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Technical Specifications](#11-technical-specifications)
12. [Testing & Validation](#12-testing--validation)
13. [Appendix](#13-appendix)

---

## 1. Executive Summary

### What Is This?

A **Machine Learning Surrogate Model** is a neural network trained to **approximate the behavior of the full physics simulation**. Once trained, it can predict simulation outputs (cutting forces, temperature, wear, surface roughness) in **milliseconds** instead of minutes/hours.

### Key Benefits

| Metric | Full Simulation | Surrogate Model | Improvement |
|--------|----------------|-----------------|-------------|
| Single prediction time | 5-60 minutes | 5-50 ms | **1000-10000x faster** |
| Parameter optimization | Days | Minutes | **100x faster** |
| Real-time UI updates | Not possible | Possible | **Enables new features** |
| Batch predictions (1000) | 80+ hours | 5 seconds | **Enables optimization** |

### When to Use?

| Use Case | Use Surrogate? |
|----------|---------------|
| Quick parameter exploration | Yes |
| Real-time UI slider updates | Yes |
| Parameter optimization | Yes |
| Final validation simulation | No - Use full simulation |
| New material/tool not in training | No - Use full simulation |

---

## 2. Problem Statement

### The Challenge

EdgePredict's physics simulation provides highly accurate results, but each simulation run takes significant time:

```
CURRENT SIMULATION TIMES

  Simple turning operation:     ~5 minutes
  Complex milling with CFD:     ~30 minutes
  Full tool life prediction:    ~2 hours

  PROBLEM: Users cannot explore parameters interactively
```

### User Pain Points

1. **Waiting for results**: User changes cutting speed, waits 5+ minutes for new results
2. **Manual optimization**: Finding optimal parameters requires many trial runs
3. **No real-time feedback**: Cannot see how changes affect results immediately
4. **Expensive exploration**: Testing 100 parameter combinations = 8+ hours

### The Solution

Train a neural network on simulation data. The network learns the **patterns** in the data and can predict results instantly for new parameter combinations.

---

## 3. What is a Surrogate Model

### Definition

A **Surrogate Model** (also called metamodel, response surface, or emulator) is a simplified mathematical model that approximates the input-output behavior of a complex simulation.

### Visual Explanation

```
                     TRADITIONAL APPROACH

  Input Parameters --> Full Physics Simulation --> Results
  (speed, feed, DOC)      (5-60 minutes)         (forces, temp,
                          +-----------+           wear, roughness)
                          | SPH Solver|
                          | FEM Solver|
                          | CFD Solver|
                          | Contact   |
                          | Thermal   |
                          +-----------+


                     WITH SURROGATE MODEL

  Input Parameters --> Neural Network --> Results
  (speed, feed, DOC)    (5-50 ms)       (forces, temp,
                        +-------+        wear, roughness)
                        | ##### |
                        |#######|  Trained on 500+
                        | ##### |  simulation results
                        +-------+

  1000-10000x FASTER!
```

### Types of Surrogate Models

| Type | Pros | Cons | Best For |
|------|------|------|----------|
| **Neural Network** | Handles complex patterns, scales well | Needs more training data | Complex multi-output problems |
| **Gaussian Process** | Provides uncertainty estimates | Slow with large datasets | Small datasets, optimization |
| **Random Forest** | Fast training, interpretable | Less smooth predictions | Quick prototyping |
| **Polynomial Response Surface** | Simple, fast | Only for smooth relationships | Simple problems |

**Recommendation for EdgePredict**: Neural Network (specifically, a Multi-Layer Perceptron or MLP)

---

## 4. How It Works

### Phase 1: Training Data Generation

We run the full EdgePredict simulation many times with different input parameters and record the results.

```
DESIGN OF EXPERIMENTS (DOE) - PARAMETER SAMPLING

  Parameter Space:
  +---------------------------------------------+
  | Cutting Speed:  50 - 300 m/min              |
  | Feed Rate:      0.05 - 0.3 mm/rev           |
  | Depth of Cut:   0.5 - 5.0 mm                |
  | Tool Rake:      -10 deg to +15 deg          |
  | Coolant:        Dry, Flood, MQL             |
  | Material:       Ti-6Al-4V, Inconel 718...   |
  +---------------------------------------------+

  Sampling Method: Latin Hypercube Sampling (LHS)
  Samples: 500-2000 simulation runs
```

**Example Training Dataset:**

| Run | V (m/min) | f (mm/rev) | DOC (mm) | Rake | Coolant | Fx (N) | Fy (N) | Fz (N) | T_max (C) | VB (mm) | Ra (um) |
|-----|-----------|------------|----------|------|---------|--------|--------|--------|-----------|---------|---------|
| 1 | 100 | 0.10 | 1.0 | 5 | Flood | 450 | 220 | 180 | 520 | 0.12 | 1.2 |
| 2 | 150 | 0.10 | 1.0 | 5 | Flood | 380 | 185 | 152 | 610 | 0.18 | 1.1 |
| 3 | 100 | 0.20 | 1.0 | 5 | Flood | 620 | 310 | 248 | 490 | 0.15 | 1.8 |
| 4 | 100 | 0.10 | 2.0 | 5 | Flood | 890 | 445 | 356 | 545 | 0.14 | 1.3 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 500 | 200 | 0.15 | 1.5 | 0 | MQL | 420 | 205 | 168 | 680 | 0.22 | 1.4 |

### Phase 2: Data Preprocessing

Before training, we need to:

1. **Normalize inputs**: Scale all inputs to 0-1 range
2. **Normalize outputs**: Scale outputs for stable training
3. **Handle categorical variables**: One-hot encode material, coolant type
4. **Split data**: 80% training, 10% validation, 10% test

```python
# Pseudocode for preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numerical features
numerical_features = ['speed', 'feed', 'depth', 'rake_angle']
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[numerical_features])

# Categorical features
categorical_features = ['material', 'coolant_type', 'tool_coating']
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(data[categorical_features])

# Combine
X = np.concatenate([X_numerical, X_categorical], axis=1)

# Output normalization
y_scaler = StandardScaler()
y = y_scaler.fit_transform(data[output_columns])
```

### Phase 3: Neural Network Training

```
       INPUTS                    NEURAL NETWORK                    OUTPUTS
  +-------------+           +----------------------+          +-----------------+
  | Cutting     |           |                      |          |                 |
  | Speed       |---+       |  Input Layer (12)    |      +-->| Cutting Force   |
  +-------------+   |       |        |             |      |   | Fx, Fy, Fz      |
  | Feed Rate   |---+       |        v             |      |   +-----------------+
  +-------------+   |       |  Hidden Layer 1      |      |   | Temperature     |
  | Depth of    |---+       |  (128 neurons, ReLU) |      |   | Max, Avg        |
  | Cut         |   |       |        |             |      |   +-----------------+
  +-------------+   +------>|        v             |------+-->| Tool Wear Rate  |
  | Rake Angle  |---+       |  Hidden Layer 2      |      |   | VB (mm/min)     |
  +-------------+   |       |  (256 neurons, ReLU) |      |   +-----------------+
  | Material    |---+       |        |             |      |   | Surface         |
  | (one-hot)   |   |       |        v             |      |   | Roughness Ra    |
  +-------------+   |       |  Hidden Layer 3      |      |   +-----------------+
  | Coolant     |---+       |  (128 neurons, ReLU) |      +-->| Chip Type       |
  | (one-hot)   |   |       |        |             |          | (classification)|
  +-------------+   |       |        v             |          +-----------------+
  | Tool        |---+       |  Output Layer (8)    |
  | Coating     |           |  (Linear activation) |
  +-------------+           +----------------------+
```

**Training Configuration:**

```python
# PyTorch model definition
class MachiningPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # Mean Squared Error for regression
epochs = 500
batch_size = 32
```

### Phase 4: Model Export to ONNX

After training, export the model to ONNX format for C++ inference:

```python
# Export trained model to ONNX
import torch.onnx

dummy_input = torch.randn(1, input_dim)
torch.onnx.export(
    model,
    dummy_input,
    "machining_surrogate.onnx",
    input_names=['parameters'],
    output_names=['predictions'],
    dynamic_axes={'parameters': {0: 'batch'}, 'predictions': {0: 'batch'}}
)
```

### Phase 5: C++ Inference with ONNX Runtime

```cpp
// C++ inference using ONNX Runtime
#include <onnxruntime_cxx_api.h>

class SurrogatePredictor {
public:
    SurrogatePredictor(const std::string& modelPath) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SurrogateModel");
        session_ = Ort::Session(env_, modelPath.c_str(), Ort::SessionOptions{});
    }
    
    PredictionResult predict(const MachiningParameters& params) {
        // Prepare input tensor
        std::vector<float> input = prepareInput(params);
        
        // Run inference (takes ~5ms)
        auto output = session_.Run(...);
        
        // Parse output
        return parseOutput(output);
    }
    
private:
    Ort::Env env_;
    Ort::Session session_;
};
```

---

## 5. Architecture

### System Overview

```
                        EDGEPREDICT SURROGATE SYSTEM

  +---------------------------------------------------------------------+
  |                      TRAINING SUBSYSTEM                              |
  |                      (Python + PyTorch)                              |
  |  +-------------+   +-------------+   +-------------+                |
  |  |   DOE       |   |  Training   |   |   Model     |                |
  |  |  Sampler    |-->|  Pipeline   |-->|   Export    |                |
  |  |             |   |             |   |  (ONNX)     |                |
  |  +-------------+   +-------------+   +------+------+                |
  +-------------------------------------------|---------------------------+
                                              |
                                              v
  +-----------------------------------------------------------------------+
  |                      INFERENCE SUBSYSTEM                               |
  |                      (C++ + ONNX Runtime)                              |
  |  +-------------+   +-------------+   +-------------+                  |
  |  |   Model     |   |  Surrogate  |   |   Result    |                  |
  |  |   Loader    |-->|  Predictor  |-->|   Cache     |                  |
  |  |             |   |             |   |             |                  |
  |  +-------------+   +-------------+   +-------------+                  |
  |                          |                                             |
  |                          v                                             |
  |  +---------------------------------------------------------------------+
  |  |                    API LAYER                                        |
  |  |  predict()  |  batchPredict()  |  getConfidence()  |  retrain()    |
  |  +---------------------------------------------------------------------+
  +-----------------------------------------------------------------------+
```

### Component Details

#### 1. TrainingDataGenerator

Manages automated simulation runs for training data collection.

```cpp
class TrainingDataGenerator {
public:
    // Configure parameter ranges
    void setParameterRange(const std::string& param, double min, double max);
    
    // Generate sample points using Latin Hypercube Sampling
    std::vector<ParameterSet> generateSamples(int numSamples);
    
    // Run simulations and collect results
    TrainingDataset runSimulations(const std::vector<ParameterSet>& samples);
    
    // Export to CSV/HDF5 for Python training
    void exportDataset(const TrainingDataset& data, const std::string& path);
    
private:
    std::map<std::string, std::pair<double, double>> parameterRanges_;
    SimulationEngine& engine_;
};
```

#### 2. SurrogatePredictor

C++ class for fast inference.

```cpp
class SurrogatePredictor {
public:
    // Load ONNX model
    bool loadModel(const std::string& onnxPath);
    
    // Single prediction (returns in ~5ms)
    PredictionResult predict(const MachiningParameters& params);
    
    // Batch prediction (more efficient for many samples)
    std::vector<PredictionResult> batchPredict(
        const std::vector<MachiningParameters>& params);
    
    // Get confidence/uncertainty estimate
    double getConfidence(const MachiningParameters& params);
    
    // Check if parameters are within training range
    bool isInTrainingRange(const MachiningParameters& params);
    
    // Get model metadata
    ModelMetadata getMetadata() const;
    
private:
    Ort::Env env_;
    Ort::Session session_;
    InputNormalizer normalizer_;
    OutputDenormalizer denormalizer_;
};
```

#### 3. SurrogateModelManager

High-level manager for the surrogate system.

```cpp
class SurrogateModelManager {
public:
    // Initialize with model path
    void initialize(const std::string& modelDirectory);
    
    // Get predictor for specific material/operation combination
    SurrogatePredictor& getPredictor(
        const std::string& material,
        const std::string& operation);
    
    // Check if surrogate should be used (vs full simulation)
    bool shouldUseSurrogate(const MachiningParameters& params);
    
    // Hybrid mode: use surrogate with validation
    PredictionResult predictWithValidation(
        const MachiningParameters& params,
        SimulationEngine& engine,
        double validationProbability = 0.05);  // 5% full sim for validation
    
    // List available models
    std::vector<ModelInfo> listModels() const;
    
private:
    std::map<std::string, SurrogatePredictor> predictors_;
};
```

---

## 6. Training Pipeline

### Workflow Diagram

```
                         TRAINING PIPELINE WORKFLOW

  +------------+    +------------+    +------------+    +------------+
  |   Define   |    |  Generate  |    |    Run     |    |   Export   |
  | Parameter  |--->|   Sample   |--->|Simulations |--->|  Dataset   |
  |  Ranges    |    |   Points   |    |            |    |            |
  +------------+    +------------+    +------------+    +------------+
       |                                                      |
       | input.json                                          | .csv/.h5
       |                                                      |
       v                                                      v
  +--------------------------------------------------------------------+
  |                        PYTHON TRAINING                              |
  |  +----------+   +----------+   +----------+   +----------+         |
  |  |   Load   |   |   Train  |   | Validate |   |  Export  |         |
  |  |   Data   |-->|   Model  |-->|   Model  |-->|   ONNX   |         |
  |  |          |   |          |   |          |   |          |         |
  |  +----------+   +----------+   +----------+   +----------+         |
  +--------------------------------------------------------------------+
                                           |
                                           v
  +--------------------------------------------------------------------+
  |                        C++ INTEGRATION                              |
  |  +----------+   +----------+   +----------+                        |
  |  |   Load   |   |   Test   |   |  Deploy  |                        |
  |  |   ONNX   |-->| Accuracy |-->|  to App  |                        |
  |  |          |   |          |   |          |                        |
  |  +----------+   +----------+   +----------+                        |
  +--------------------------------------------------------------------+
```

### Training Script (Python)

```python
"""
train_surrogate.py - Training script for EdgePredict Surrogate Model
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import onnx
import onnxruntime as ort

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_training_data(csv_path):
    """Load simulation results from CSV"""
    df = pd.read_csv(csv_path)
    
    # Input columns
    numerical_inputs = ['speed_mpm', 'feed_mmrev', 'depth_mm', 'rake_deg']
    categorical_inputs = ['material', 'coolant', 'coating']
    
    # Output columns
    outputs = ['force_x', 'force_y', 'force_z', 
               'temp_max', 'temp_avg',
               'wear_rate', 'roughness_ra']
    
    return df, numerical_inputs, categorical_inputs, outputs

# ============================================================================
# 2. PREPROCESS DATA
# ============================================================================

def create_preprocessor(numerical_cols, categorical_cols):
    """Create sklearn preprocessor for inputs"""
    return ColumnTransformer([
        ('numerical', StandardScaler(), numerical_cols),
        ('categorical', OneHotEncoder(sparse=False), categorical_cols)
    ])

# ============================================================================
# 3. DEFINE MODEL
# ============================================================================

class MachiningPredictor(nn.Module):
    """Neural network for machining parameter prediction"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=500, lr=0.001):
    """Train the surrogate model"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += loss_fn(predictions, y_batch).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    return model

# ============================================================================
# 5. EXPORT TO ONNX
# ============================================================================

def export_to_onnx(model, input_dim, output_path):
    """Export trained model to ONNX format"""
    
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['machining_parameters'],
        output_names=['predictions'],
        dynamic_axes={
            'machining_parameters': {0: 'batch_size'},
            'predictions': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    # Verify export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model exported to {output_path}")

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    df, num_cols, cat_cols, output_cols = load_training_data("training_data.csv")
    
    # Preprocess
    preprocessor = create_preprocessor(num_cols, cat_cols)
    X = preprocessor.fit_transform(df)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(df[output_cols])
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Create model
    model = MachiningPredictor(input_dim=X.shape[1], output_dim=len(output_cols))
    
    # Train
    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)
    model = train_model(model, train_loader, val_loader)
    
    # Export
    export_to_onnx(model, X.shape[1], "machining_surrogate.onnx")
    
    # Save preprocessors for inference
    save_preprocessor(preprocessor, "input_preprocessor.pkl")
    save_preprocessor(y_scaler, "output_scaler.pkl")
```

---

## 7. Inference System

### C++ Implementation

```cpp
// File: include/SurrogateModel.h

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace edgepredict {

/**
 * Input parameters for surrogate prediction
 */
struct MachiningParameters {
    double cuttingSpeed;      // m/min
    double feedRate;          // mm/rev
    double depthOfCut;        // mm
    double rakeAngle;         // degrees
    std::string material;     // e.g., "Ti-6Al-4V"
    std::string coolant;      // e.g., "Flood", "MQL", "Dry"
    std::string toolCoating;  // e.g., "TiAlN", "TiN", "Uncoated"
};

/**
 * Prediction results from surrogate model
 */
struct PredictionResult {
    // Forces
    double forceX;            // Cutting force (N)
    double forceY;            // Feed force (N)
    double forceZ;            // Thrust force (N)
    
    // Thermal
    double maxTemperature;    // deg C
    double avgTemperature;    // deg C
    
    // Wear & Quality
    double wearRate;          // mm/min (flank wear rate)
    double surfaceRoughness;  // Ra (um)
    
    // Metadata
    double confidence;        // 0-1, prediction confidence
    double inferenceTimeMs;   // Time taken for prediction
    bool inTrainingRange;     // True if inputs within training range
};

/**
 * Neural network surrogate model for instant predictions
 */
class SurrogateModel {
public:
    SurrogateModel();
    ~SurrogateModel();
    
    /**
     * Load ONNX model from file
     */
    bool loadModel(const std::string& modelPath, 
                   const std::string& configPath);
    
    /**
     * Get single prediction (fast, ~5ms)
     */
    PredictionResult predict(const MachiningParameters& params);
    
    /**
     * Batch prediction (more efficient for multiple samples)
     */
    std::vector<PredictionResult> batchPredict(
        const std::vector<MachiningParameters>& params);
    
    /**
     * Check if parameters are within training range
     */
    bool isInTrainingRange(const MachiningParameters& params) const;
    
    /**
     * Get model information
     */
    struct ModelInfo {
        std::string name;
        std::string version;
        int numInputs;
        int numOutputs;
        std::vector<std::string> materials;
        std::vector<std::string> operations;
        double trainingAccuracy;
    };
    ModelInfo getModelInfo() const;
    
    /**
     * Check if model is loaded and ready
     */
    bool isReady() const { return m_isReady; }

private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;
    
    // Preprocessing parameters
    struct NormalizationParams {
        std::vector<double> means;
        std::vector<double> stds;
        std::map<std::string, std::vector<double>> categoricalEncodings;
    };
    NormalizationParams m_inputNorm;
    NormalizationParams m_outputNorm;
    
    // Training range for validation
    struct ParameterRange {
        double min, max;
    };
    std::map<std::string, ParameterRange> m_trainingRanges;
    
    // State
    bool m_isReady = false;
    ModelInfo m_modelInfo;
    
    // Helper methods
    std::vector<float> preprocessInput(const MachiningParameters& params) const;
    PredictionResult postprocessOutput(const std::vector<float>& rawOutput) const;
};

} // namespace edgepredict
```

### Example Usage

```cpp
// Example: Real-time UI integration

#include "SurrogateModel.h"
#include <iostream>
#include <chrono>

int main() {
    edgepredict::SurrogateModel surrogate;
    
    // Load model
    if (!surrogate.loadModel("models/titanium_turning.onnx",
                             "models/titanium_turning_config.json")) {
        std::cerr << "Failed to load surrogate model\n";
        return 1;
    }
    
    // User changes speed slider in UI
    edgepredict::MachiningParameters params;
    params.cuttingSpeed = 120.0;    // m/min
    params.feedRate = 0.15;         // mm/rev
    params.depthOfCut = 1.5;        // mm
    params.rakeAngle = 5.0;         // degrees
    params.material = "Ti-6Al-4V";
    params.coolant = "Flood";
    params.toolCoating = "TiAlN";
    
    // Get instant prediction
    auto start = std::chrono::high_resolution_clock::now();
    auto result = surrogate.predict(params);
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Display results (available in ~5ms!)
    std::cout << "=== SURROGATE PREDICTION ===\n";
    std::cout << "Cutting Force: " << result.forceX << " N\n";
    std::cout << "Max Temperature: " << result.maxTemperature << " C\n";
    std::cout << "Wear Rate: " << result.wearRate << " mm/min\n";
    std::cout << "Surface Roughness: " << result.surfaceRoughness << " um\n";
    std::cout << "Confidence: " << result.confidence * 100 << "%\n";
    std::cout << "Prediction Time: " << elapsed << " ms\n";
    
    // Check if within training range
    if (!result.inTrainingRange) {
        std::cout << "Warning: Parameters outside training range\n";
        std::cout << "   Consider running full simulation for validation\n";
    }
    
    return 0;
}
```

---

## 8. Integration with EdgePredict

### UI Integration Architecture

```
                           EDGEPREDICT UI APP

  +----------------------------------------------------------------------+
  |                     PARAMETER CONTROL PANEL                          |
  |  +------------------------------------------------------------------+|
  |  | Cutting Speed: [========*=========] 150 m/min                    ||
  |  | Feed Rate:     [====*=============] 0.12 mm/rev                  ||
  |  | Depth of Cut:  [===========*======] 2.0 mm                       ||
  |  +------------------------------------------------------------------+|
  +---------------------------------|-------------------------------------+
                                    | OnSliderChange
                                    v
  +----------------------------------------------------------------------+
  |                     SURROGATE PREDICTOR                              |
  |                     (C++ / ONNX Runtime)                             |
  |                     Prediction Time: ~5ms                            |
  +---------------------------------|-------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  |                     LIVE RESULTS DISPLAY                             |
  |  +------------------------------------------------------------------+|
  |  |  Cutting Force:  ========....  485 N                             ||
  |  |  Temperature:    =============  612 C  [!] HIGH                  ||
  |  |  Wear Rate:      ====..........  0.18 mm/min                     ||
  |  |  Surface Ra:     ===...........  1.2 um [OK]                     ||
  |  |                                                                  ||
  |  |  Confidence: 94%  |  Within Training Range: YES                  ||
  |  +------------------------------------------------------------------+|
  +----------------------------------------------------------------------+

  +----------------------------------------------------------------------+
  |                     ACTION BUTTONS                                   |
  |  [ Run Full Simulation ]  [ Optimize Parameters ]  [ Export ]       |
  +----------------------------------------------------------------------+
```

### API for UI Integration

```cpp
// API exposed to UI application

class EdgePredictAPI {
public:
    /**
     * Quick prediction for UI (uses surrogate if available)
     * Returns in ~5ms
     */
    PredictionResult quickPredict(const MachiningParameters& params) {
        if (m_surrogate.isReady() && m_surrogate.isInTrainingRange(params)) {
            return m_surrogate.predict(params);
        } else {
            // Fallback to fast estimate or cached result
            return getCachedEstimate(params);
        }
    }
    
    /**
     * Full simulation (accurate but slow)
     */
    SimulationResult runFullSimulation(
        const MachiningParameters& params,
        std::function<void(double progress)> progressCallback) {
        // Run EdgePredict engine
        return m_engine.run(params, progressCallback);
    }
    
    /**
     * Optimize parameters using surrogate
     */
    MachiningParameters optimize(
        const std::string& objective,
        const std::map<std::string, std::string>& constraints) {
        // Use surrogate for fast optimization (10000+ evaluations)
        return m_optimizer.optimize(m_surrogate, objective, constraints);
    }
    
private:
    SurrogateModel m_surrogate;
    SimulationEngine m_engine;
    ParameterOptimizer m_optimizer;
};
```

---

## 9. Use Cases

### Use Case 1: Real-Time Parameter Exploration

**Scenario**: User wants to see how changing cutting speed affects results

```
User Action: Drags speed slider from 100 to 200 m/min

Timeline:
  0 ms: Slider change detected
  2 ms: New parameters sent to surrogate
  7 ms: Prediction received
  8 ms: UI updated with new force/temp/wear values

Result: Smooth, responsive UI with instant feedback
```

### Use Case 2: Parameter Optimization

**Scenario**: Find optimal speed/feed for minimum wear while maintaining surface quality

```cpp
// Define optimization problem
problem.minimize("wear_rate");
problem.constraint("surface_roughness", "<", 1.6);  // Ra < 1.6 um
problem.constraint("cutting_force", "<", 600);      // Force < 600 N
problem.constraint("temperature", "<", 700);        // Temp < 700 C

// Parameter bounds
problem.setRange("cutting_speed", 80, 250);    // m/min
problem.setRange("feed_rate", 0.08, 0.25);     // mm/rev
problem.setRange("depth_of_cut", 0.5, 3.0);    // mm

// Run optimization (evaluates 10,000 parameter combinations)
// With surrogate: ~50 seconds
// Without surrogate: ~800 hours!
auto optimal = optimizer.solve(problem, surrogate);

std::cout << "Optimal Settings:\n";
std::cout << "  Speed: " << optimal.cuttingSpeed << " m/min\n";
std::cout << "  Feed: " << optimal.feedRate << " mm/rev\n";
std::cout << "  DOC: " << optimal.depthOfCut << " mm\n";
std::cout << "  Expected Wear Rate: " << optimal.predictedWear << " mm/min\n";
```

### Use Case 3: Process Capability Analysis

**Scenario**: Analyze sensitivity of results to parameter variations

```cpp
// Monte Carlo analysis
std::vector<MachiningParameters> samples;

// Generate 10,000 samples with +/-5% variation
for (int i = 0; i < 10000; i++) {
    MachiningParameters p = nominal;
    p.cuttingSpeed *= (1 + randomUniform(-0.05, 0.05));
    p.feedRate *= (1 + randomUniform(-0.05, 0.05));
    samples.push_back(p);
}

// Batch predict (takes ~5 seconds with surrogate)
auto results = surrogate.batchPredict(samples);

// Analyze distribution
auto stats = computeStatistics(results);
std::cout << "Force: " << stats.mean << " +/- " << stats.stddev << " N\n";
std::cout << "Cpk: " << stats.cpk << "\n";
```

### Use Case 4: Quick Material Comparison

**Scenario**: Compare 5 different materials for the same operation

```cpp
std::vector<std::string> materials = {
    "Ti-6Al-4V", "Inconel 718", "AISI 4340", "Al 7075", "Stainless 316"
};

std::cout << "Material Comparison (V=150, f=0.1, DOC=1.5):\n";
std::cout << std::setw(15) << "Material" 
          << std::setw(10) << "Force" 
          << std::setw(10) << "Temp" 
          << std::setw(10) << "Wear\n";

for (const auto& material : materials) {
    params.material = material;
    auto result = surrogate.predict(params);  // ~5ms each
    
    std::cout << std::setw(15) << material
              << std::setw(10) << result.forceX
              << std::setw(10) << result.maxTemperature
              << std::setw(10) << result.wearRate << "\n";
}

// Total time: ~25ms for all 5 materials!
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | Design training data schema | CSV format specification |
| 1.2 | Implement TrainingDataGenerator | C++ class for automated runs |
| 1.3 | Create DOE sampling module | Latin Hypercube sampler |
| 1.4 | Export simulation results | CSV/HDF5 export function |

### Phase 2: Training Pipeline (Week 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | Python training script | train_surrogate.py |
| 2.2 | Data preprocessing module | Normalization pipeline |
| 2.3 | Neural network architecture | PyTorch model |
| 2.4 | ONNX export function | Model export script |
| 2.5 | Model validation metrics | Accuracy reporting |

### Phase 3: C++ Inference (Week 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | ONNX Runtime integration | CMake setup |
| 3.2 | SurrogatePredictor class | C++ inference class |
| 3.3 | Input preprocessing | Normalization in C++ |
| 3.4 | Output postprocessing | Denormalization in C++ |
| 3.5 | Confidence estimation | Uncertainty quantification |

### Phase 4: Integration (Week 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 | API design for UI | EdgePredictAPI class |
| 4.2 | Model management | Model loading/switching |
| 4.3 | Hybrid mode | Surrogate + validation runs |
| 4.4 | Optimization integration | Parameter optimizer |
| 4.5 | Documentation | User guide |

### Phase 5: Validation & Deployment (Week 9-10)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1 | Generate training data | 500+ simulation runs |
| 5.2 | Train production models | ONNX models for each material |
| 5.3 | Accuracy validation | Comparison with test simulations |
| 5.4 | Performance benchmarking | Latency measurements |
| 5.5 | Release | Integrated feature |

---

## 11. Technical Specifications

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| ONNX Runtime | 1.16+ | C++ neural network inference |
| PyTorch | 2.0+ | Python training |
| ONNX | 1.14+ | Model format |
| NumPy | 1.24+ | Data processing |
| Scikit-learn | 1.3+ | Preprocessing |
| Pandas | 2.0+ | Data loading |

### Model Specifications

```yaml
# Model Configuration
architecture:
  type: "MLP"
  input_dim: 12  # 4 numerical + 8 one-hot encoded
  hidden_layers: [128, 256, 128]
  output_dim: 8
  activation: "ReLU"
  dropout: 0.1
  
training:
  optimizer: "Adam"
  learning_rate: 0.001
  batch_size: 32
  epochs: 500
  early_stopping_patience: 50
  
performance:
  inference_time_ms: 5-10
  accuracy_r2: 0.95+
  training_time_hours: 1-2
```

### File Structure

```
edgepredict-engine/
|-- include/
|   |-- SurrogateModel.h
|   |-- TrainingDataGenerator.h
|   +-- ONNXPredictor.h
|-- src/
|   |-- SurrogateModel.cpp
|   |-- TrainingDataGenerator.cpp
|   +-- ONNXPredictor.cpp
|-- python/
|   |-- train_surrogate.py
|   |-- preprocess.py
|   |-- evaluate.py
|   +-- requirements.txt
|-- models/
|   |-- titanium_turning.onnx
|   |-- titanium_turning_config.json
|   |-- steel_milling.onnx
|   +-- ...
+-- data/
    +-- training_samples/
        |-- titanium_turning_500samples.csv
        +-- ...
```

---

## 12. Testing & Validation

### Accuracy Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| R2 Score | > 0.95 | Coefficient of determination |
| RMSE (Forces) | < 50 N | Root mean square error |
| RMSE (Temperature) | < 30 C | |
| RMSE (Wear) | < 0.02 mm/min | |
| Max Error | < 15% | Worst-case percentage error |

### Test Cases

```cpp
// Unit test example
TEST(SurrogateModel, PredictionAccuracy) {
    SurrogateModel model;
    model.loadModel("models/test_model.onnx", "models/test_config.json");
    
    // Known test case
    MachiningParameters params;
    params.cuttingSpeed = 150.0;
    params.feedRate = 0.15;
    params.depthOfCut = 1.5;
    params.material = "Ti-6Al-4V";
    
    // Run full simulation for ground truth
    auto groundTruth = runFullSimulation(params);
    
    // Compare with surrogate
    auto prediction = model.predict(params);
    
    EXPECT_NEAR(prediction.forceX, groundTruth.forceX, 50);  // +/-50 N
    EXPECT_NEAR(prediction.maxTemperature, groundTruth.maxTemp, 30);  // +/-30 C
    EXPECT_NEAR(prediction.wearRate, groundTruth.wearRate, 0.02);  // +/-0.02 mm/min
}

TEST(SurrogateModel, InferenceSpeed) {
    SurrogateModel model;
    model.loadModel("models/production.onnx", "models/production_config.json");
    
    MachiningParameters params = getRandomParams();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        model.predict(params);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double avgMs = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    EXPECT_LT(avgMs, 10);  // Must be < 10ms per prediction
}
```

---

## 13. Appendix

### A. Sample Training Data Format

```csv
run_id,speed_mpm,feed_mmrev,depth_mm,rake_deg,material,coolant,coating,force_x,force_y,force_z,temp_max,temp_avg,wear_rate,roughness_ra
1,100,0.10,1.0,5,Ti-6Al-4V,Flood,TiAlN,450,220,180,520,385,0.12,1.2
2,150,0.10,1.0,5,Ti-6Al-4V,Flood,TiAlN,380,185,152,610,445,0.18,1.1
3,100,0.20,1.0,5,Ti-6Al-4V,Flood,TiAlN,620,310,248,490,360,0.15,1.8
...
```

### B. ONNX Model Metadata

```json
{
  "model_name": "edgepredict_titanium_turning_v1",
  "version": "1.0.0",
  "created": "2026-01-30",
  "input_features": {
    "numerical": ["speed_mpm", "feed_mmrev", "depth_mm", "rake_deg"],
    "categorical": ["material", "coolant", "coating"]
  },
  "output_features": [
    "force_x", "force_y", "force_z",
    "temp_max", "temp_avg",
    "wear_rate", "roughness_ra"
  ],
  "training_data": {
    "num_samples": 500,
    "parameter_ranges": {
      "speed_mpm": [50, 300],
      "feed_mmrev": [0.05, 0.3],
      "depth_mm": [0.5, 5.0]
    }
  },
  "performance": {
    "r2_score": 0.963,
    "inference_time_ms": 5.2
  }
}
```

### C. Error Handling

```cpp
// Handle out-of-range predictions
auto result = surrogate.predict(params);

if (!result.inTrainingRange) {
    // Show warning to user
    ui.showWarning("Parameters outside training range. Results may be less accurate.");
    
    // Suggest running full simulation
    if (ui.askUser("Run full simulation for validation?")) {
        auto fullResult = engine.run(params);
        ui.displayResults(fullResult);
    }
}

// Handle low confidence
if (result.confidence < 0.8) {
    ui.showWarning("Low prediction confidence (" + 
                   std::to_string(result.confidence * 100) + "%)");
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-30 | EdgePredict Team | Initial document |

---

**End of Document**
