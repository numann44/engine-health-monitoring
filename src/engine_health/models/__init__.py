import numpy as np
import torch
from torch import nn


class RULNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(46, 64, 2, batch_first=True, dropout=.1)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def predict_net(model, x, device="cpu", batch=512):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            value = model(torch.from_numpy(x[start:start+batch]).to(device)) * 100
            predictions.append(value.cpu().numpy())
    out = np.concatenate(predictions)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite model prediction")
    return out
