import os
import torch
import torch.nn as nn
from data.dataset import get_dataloaders
from models.multimodal_lstm import MultiModalLSTM
from training.loss import wta_loss
from evaluate import compute_min_ade, compute_min_fde

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

def train_model(train_loader, val_loader, epochs=100):
    model = MultiModalLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_min_ade = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inp = batch['input'].to(DEVICE)   # (B, 4, 4)
            tgt = batch['target'].to(DEVICE)  # (B, 6, 2)

            modes, conf = model(inp)
            loss = wta_loss(modes, conf, tgt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        min_ades, min_fdes = [], []
        with torch.no_grad():
            for batch in val_loader:
                inp = batch['input'].to(DEVICE)
                tgt = batch['target'].to(DEVICE)
                modes, conf = model(inp)
                min_ades.append(compute_min_ade(modes, tgt))
                min_fdes.append(compute_min_fde(modes, tgt))

        avg_min_ade = sum(min_ades) / len(min_ades)
        avg_min_fde = sum(min_fdes) / len(min_fdes)
        scheduler.step(avg_min_ade)

        if avg_min_ade < best_min_ade:
            best_min_ade = avg_min_ade
            torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pt')

        if epoch % 10 == 0:
            print(f"Ep {epoch:3d} | loss: {train_loss/len(train_loader):.4f} | minADE: {avg_min_ade:.4f} | minFDE: {avg_min_fde:.4f}")

    print(f"\nBest minADE: {best_min_ade:.4f}")
    return model

if __name__ == "__main__":
    annotation_path = os.environ.get('NUSCENES_MINI_PATH', 'v1.0-mini/sample_annotation.json')
    try:
        train_loader, val_loader, _ = get_dataloaders(annotation_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)
        
    print(f"Starting training on {DEVICE}...")
    train_model(train_loader, val_loader, epochs=100)
