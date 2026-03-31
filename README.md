# trajectory-prediction
Multi-Modal Trajectory Prediction (LSTM + PyTorch)
Overview

This project predicts future trajectories of pedestrians and cyclists in an urban autonomous driving setting. Instead of a single path, the model generates multiple possible future trajectories (K=3) to capture real-world uncertainty.
Built using LSTM-based temporal modeling, enhanced with agent-centric transformations, velocity features, and a Winner-Take-All (WTA) loss.

Problem
Input: 2 seconds of past motion
Output: 3 seconds of future trajectory
Goal: Predict multiple plausible future paths
Dataset
nuScenes v1.0-mini
Uses:
Position (x, y)
Derived velocity (dx, dy)
Basic environmental context
Preprocessing (Key Idea)
Agent-Centric Transformation
Shift last observed point → (0,0)
Rotate trajectory → align with X-axis
Why?
Removes dependency on global position/orientation
Improves generalization and stability
Model
MultiModal LSTM (PyTorch)
Shared encoder + separate decoders
Outputs 3 trajectories
Input:
(x, y, dx, dy)
Output:
3 × future trajectory sequences
Loss Function
Winner-Take-All (WTA)
Computes error across all predictions
Only penalizes the best matching trajectory

 Encourages diverse and realistic outputs

Metrics
ADE – average distance error
FDE – final position error
Performance
Model	MinADE	MinFDE
Keras	13.46 m	24.46 m
PyTorch	0.358 m	0.483 m
~97–98% improvement

Repo Structure
├── best_model.pt   # Trained PyTorch model
└── README.md
Usage
import torch

model = torch.load("best_model.pt")
model.eval()
Input format:
(batch_size, timesteps, 4)
→ [x, y, dx, dy]
Features Implemented
Multi-modal trajectory prediction
Agent-centric preprocessing
Velocity-based inputs
Risk & collision analysis (in development)
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
