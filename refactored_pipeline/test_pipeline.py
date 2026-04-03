import os
import torch
from data.dataset import get_dataloaders
from models.multimodal_lstm import MultiModalLSTM

def run_test():
    annotation_path = os.environ.get('NUSCENES_MINI_PATH', 'v1.0-mini/sample_annotation.json')
    print("Initializing Data Pipeline...")
    try:
        # Fetch dataloaders
        train_loader, _, _ = get_dataloaders(annotation_path, past=4, future=6, batch_size=8)
    except FileNotFoundError as e:
        print(e)
        print("Please download and extract nuScenes v1.0-mini dataset first.")
        return

    print("Loading Batch...")
    # Fetch a single real data batch
    batch = next(iter(train_loader))
    inputs = batch['input']
    targets = batch['target']
    
    print(f"Input Shape: {inputs.shape}  [Batch, Past Steps, Features (x, y, vx, vy)]")
    print(f"Target Shape: {targets.shape} [Batch, Future Steps, Features (x, y)]")

    print("\nInitializing MultiModal LSTM...")
    model = MultiModalLSTM(input_dim=4, hidden=256, K=3, future_steps=6)
    
    print("Running Forward Pass...")
    model.eval()
    with torch.no_grad():
        modes, conf = model(inputs)
    
    print(f"Output Modes Shape: {modes.shape} [Batch, K Modes, Future Steps, Features (x, y)]")
    print(f"Output Confidence Shape: {conf.shape} [Batch, K Modes Confidence]")
    
    print("\nEnd-to-End pipeline verified successfully!")

if __name__ == "__main__":
    run_test()
