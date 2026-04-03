import torch
import torch.nn.functional as F

def wta_loss(modes, conf, target):
    """
    WTA (Winner-Takes-All) Loss — only backprop through the best mode.
    modes:  (B, K, future_steps, 2)
    conf:   (B, K)
    target: (B, future_steps, 2)
    """
    B, K, T, _ = modes.shape
    target_exp = target.unsqueeze(1).expand_as(modes) # (B, 1, T, 2) -> (B, K, T, 2)

    # ADE per mode: (B, K)
    # L2 norm over (x,y) then mean over timesteps
    ade_per_mode = torch.norm(modes - target_exp, dim=-1).mean(dim=-1)

    # Best mode index: (B,)
    best_idx = ade_per_mode.argmin(dim=1)

    # Best mode trajectory: (B, T, 2)
    best_traj = modes[torch.arange(B), best_idx]

    # Regression loss on best mode only
    reg_loss  = F.mse_loss(best_traj, target)

    # Confidence loss — teach model which mode is best
    conf_loss = F.cross_entropy(conf, best_idx)

    return reg_loss + 0.3 * conf_loss
