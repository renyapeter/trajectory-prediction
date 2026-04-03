import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.transforms import create_sequences, agent_centric_transform, add_velocity

class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'input': self.X[idx], 'target': self.y[idx]}

def get_dataloaders(annotation_path, past=4, future=6, batch_size=32, limit=3000):
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found at {annotation_path}. Please download the nuScenes mini dataset.")
        
    with open(annotation_path) as f:
        annotations = json.load(f)

    positions = []
    for ann in annotations[:limit]:
        x, y, _ = ann['translation']
        positions.append([x, y])

    data = np.array(positions)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Normalize
    data_norm = (data - mean) / std

    # Pipelines
    X, y = create_sequences(data_norm, past=past, future=future)
    X_ac, y_ac = agent_centric_transform(X, y)
    X_feat = add_velocity(X_ac)

    split_idx = int(len(X_feat) * 0.8)
    
    X_train, y_train = X_feat[:split_idx], y_ac[:split_idx]
    X_val, y_val = X_feat[split_idx:], y_ac[split_idx:]

    train_dataset = TrajectoryDataset(X_train, y_train)
    val_dataset = TrajectoryDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, (mean, std, X, y)
