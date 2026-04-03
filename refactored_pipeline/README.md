# NuScenes Trajectory Prediction (Multi-Modal PyTorch Pipeline)

This repository contains a modular Python project for vehicle trajectory prediction using the NuScenes dataset. 

## Architecture Overview

The core model is a **MultiModalLSTM**. 
- **Encoder:** An LSTM that reads past positional and velocity data.
- **Decoders:** `K` separate LSTM decoders (where K=3) exist to generate multiple possible future trajectories independently. 
- **Loss:** It uses a custom **Winner-Takes-All (WTA) Loss** where only the predicted mode that is closest to the ground truth trajectory is penalized for regression error. This teaches the model to accurately predict distinctly different possible driving behaviors.

## Setup Instructions

### 1. Install Dependencies
Make sure you have PyTorch, NumPy, and basic ML packages installed:
```bash
pip install torch numpy
```

### 2. Download Data
Download and extract the `v1.0-mini` NuScenes dataset into a directory within the root project, or define the path using the `NUSCENES_MINI_PATH` environment variable. The code naturally looks for `v1.0-mini/sample_annotation.json` inside the working directory.

```bash
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xvzf v1.0-mini.tgz
```

## Running the Pipeline

### Testing the Pipeline
You can trigger `test_pipeline.py` to test the entire data ingestion, dataloader fetching, and model forward pass using a single batch of real data.

```bash
python test_pipeline.py
```
*Expected Output:* You should see the shapes of the real input tensor pass through the Multi-Modal model to produce output tensors indicating predictions and confidence scores.

### Training the Model
To initiate the training process and produce a `best_model.pt` saved within the `checkpoints` directory, run:
```bash
python train.py
```

### Evaluating the Model
You can calculate the MinADE (Minimum Average Displacement Error) and MinFDE (Minimum Final Displacement Error) validation metrics using:
```bash
python evaluate.py
```

## Metrics

Based on recent runs, the model achieves the following performance:
- **Best MinADE (Train)**: 0.3477
- **Validation MinADE**: 0.3524
- **Validation MinFDE**: 0.4483

**Evaluation Output:**
```text
... Loaded checkpoint.
    Validation MinADE: 0.3524
    Validation MinFDE: 0.4483
```

**Training Output:**
```text
Starting training on cuda...
Ep   0 | loss: 0.4755 | minADE: 0.3789 | minFDE: 0.4962
Ep  10 | loss: 0.4098 | minADE: 0.3771 | minFDE: 0.4845
Ep  20 | loss: 0.3994 | minADE: 0.3582 | minFDE: 0.4732
Ep  30 | loss: 0.3961 | minADE: 0.3594 | minFDE: 0.4736
Ep  40 | loss: 0.3945 | minADE: 0.3619 | minFDE: 0.4781
Ep  50 | loss: 0.3931 | minADE: 0.3628 | minFDE: 0.4787
Ep  60 | loss: 0.3934 | minADE: 0.3628 | minFDE: 0.4787
Ep  70 | loss: 0.3933 | minADE: 0.3629 | minFDE: 0.4788
Ep  80 | loss: 0.3936 | minADE: 0.3629 | minFDE: 0.4788
Ep  90 | loss: 0.3929 | minADE: 0.3629 | minFDE: 0.4788

Best minADE: 0.3477
```

### CI/CD
To automate pushing your trained weights to GitHub, configure `GH_TOKEN`, `GITHUB_USERNAME`, and `GITHUB_REPO`, then execute:
```bash
python scripts/github_push.py
```
