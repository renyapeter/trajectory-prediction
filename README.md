# Multi-Modal Trajectory Prediction (LSTM + PyTorch)

## Overview

> **🏆 Judges Note: Refactored Codebase**
> We have taken the original monolithic code and completely refactored it into a clean, modular, and production-ready Python project. You can find this improved architecture in the `refactored_pipeline` directory.
> 
> Please see the **[Refactored Pipeline README](./refactored_pipeline/README.md)** for detailed instructions on how to set it up, train, and run evaluations on the new modular structure!

This project predicts future trajectories of pedestrians and cyclists in an urban autonomous driving setting. Instead of a single path, the model generates multiple possible future trajectories (K=3) to capture real-world uncertainty.

The system is built using LSTM-based temporal modeling, enhanced with agent-centric transformations, velocity features, and a Winner-Take-All (WTA) loss.

## Problem
- Input: 2 seconds of past motion  
- Output: 3 seconds of future trajectory  
- Goal: Predict multiple plausible future paths  


## Dataset
**nuScenes v1.0-mini**

Data used:
- Position (x, y)
- Derived velocity (dx, dy)
- Basic environmental context

## Preprocessing (Key Idea)

### Agent-Centric Transformation
- Shift last observed point to (0, 0)
- Rotate trajectory to align with the X-axis

### Why?
- Removes dependency on global position and orientation  
- Improves generalization and training stability  


## Model

**MultiModal LSTM (PyTorch)**

- Shared encoder
- Separate decoders for each trajectory mode
- Outputs 3 trajectories

### Input
(x, y, dx, dy)


### Output
3 × future trajectory sequences


---

## Loss Function

**Winner-Take-All (WTA)**

- Computes error across all predicted trajectories  
- Only penalizes the best matching trajectory  

This encourages diverse and realistic predictions.

---

## Metrics

- ADE (Average Displacement Error): Mean distance between predicted and actual trajectory points  
- FDE (Final Displacement Error): Distance between predicted and actual final position  

---

## Performance

| Model   | MinADE | MinFDE |
|--------|--------|--------|
| Keras  | 13.46 m | 24.46 m |
| PyTorch | 0.358 m | 0.483 m |

Approximate improvement: 97–98%

---

## Repository Structure
├── best_model.pt # Trained PyTorch model
└── README.md

Features Implemented
Multi-modal trajectory prediction
Agent-centric preprocessing
Velocity-based inputs
Risk and collision analysis (in development)
Visualization of predicted paths


Limitations
No map overlay (dataset limitation)
No social interaction modeling
Limited dataset size (mini version)
Future Work


Social pooling (multi-agent interaction)
Transformer-based models
Full nuScenes dataset
Goal-conditioned prediction
