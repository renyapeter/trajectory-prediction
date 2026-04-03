import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden=256, K=3, future_steps=6):
        super().__init__()
        self.K = K
        self.future_steps = future_steps

        # Encoder reads past steps
        self.encoder = nn.LSTM(input_dim, hidden,
                               num_layers=2,
                               batch_first=True,
                               dropout=0.1)

        # K separate decoders — one per mode
        self.decoders = nn.ModuleList([
            nn.LSTM(2, hidden, num_layers=2, batch_first=True)
            for _ in range(K)
        ])

        # K output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(K)
        ])

        # Confidence score per mode
        self.confidence = nn.Linear(hidden, K)

    def forward(self, x):
        # x: (B, past_steps, 4) — 4 features
        B = x.shape[0]

        # Encode past
        _, (h, c) = self.encoder(x)

        # Starting input = last observed position
        dec_input = x[:, -1:, :2]  # (B, 1, 2)

        all_modes = []
        for k in range(self.K):
            hk = h.clone()
            ck = c.clone()
            preds = []
            inp = dec_input.clone()

            for _ in range(self.future_steps):
                out, (hk, ck) = self.decoders[k](inp, (hk, ck))
                pred = self.output_heads[k](out)  # (B, 1, 2)
                preds.append(pred)
                # Use predicted output as next input (autoregressive)
                inp = pred 

            mode = torch.cat(preds, dim=1)  # (B, future_steps, 2)
            all_modes.append(mode)

        # Stack all modes: (B, K, future_steps, 2)
        modes = torch.stack(all_modes, dim=1)

        # Confidence scores: (B, K)
        conf = F.softmax(self.confidence(h[-1]), dim=-1)

        return modes, conf
