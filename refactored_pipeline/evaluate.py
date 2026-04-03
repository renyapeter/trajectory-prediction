import torch

def compute_min_ade(modes, target):
    """
    Minimum Average Displacement Error
    modes:  (B, K, T, 2)
    target: (B, T, 2)
    """
    target_exp = target.unsqueeze(1).expand_as(modes) # (B, K, T, 2)
    ade_k = torch.norm(modes - target_exp, dim=-1).mean(dim=-1) # (B, K)
    return ade_k.min(dim=1).values.mean().item()

def compute_min_fde(modes, target):
    """
    Minimum Final Displacement Error
    modes:  (B, K, T, 2)
    target: (B, T, 2)
    """
    pred_final = modes[:, :, -1, :]      # (B, K, 2)
    gt_final   = target[:, -1, :]        # (B, 2)
    fde_k = torch.norm(pred_final - gt_final.unsqueeze(1), dim=-1) # (B, K)
    return fde_k.min(dim=1).values.mean().item()

if __name__ == "__main__":
    import os
    from data.dataset import get_dataloaders
    from models.multimodal_lstm import MultiModalLSTM

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Needs a sample annotation JSON
    annotation_path = os.environ.get('NUSCENES_MINI_PATH', 'v1.0-mini/sample_annotation.json')
    
    try:
        _, val_loader, _ = get_dataloaders(annotation_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)
        
    model = MultiModalLSTM().to(DEVICE)
    ckpt_path = 'checkpoints/best_model.pt'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print("Loaded checkpoint.")
    else:
        print("No checkpoint found. Evaluating with random weights.")

    model.eval()
    all_modes = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            inp = batch['input'].to(DEVICE)
            tgt = batch['target'].to(DEVICE)
            modes, _ = model(inp)
            all_modes.append(modes)
            all_targets.append(tgt)
            
    modes_tensor = torch.cat(all_modes, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)
    
    ade = compute_min_ade(modes_tensor, target_tensor)
    fde = compute_min_fde(modes_tensor, target_tensor)
    
    print(f"Validation MinADE: {ade:.4f}")
    print(f"Validation MinFDE: {fde:.4f}")
