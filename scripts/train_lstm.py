import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from core.lstm_model import TurbulenceLSTM

BATCH = 64
EPOCHS = 80
LR = 3e-4

NORM = np.array([1.0, 5000.0, 8.0, 1.0], dtype=np.float32)

X = np.load("cache/train_X.npy") / NORM
y = np.load("cache/train_y.npy")

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

split = int(0.85 * len(X_t))

train_dl = DataLoader(
    TensorDataset(X_t[:split], y_t[:split]), batch_size=BATCH, shuffle=True
)

val_dl = DataLoader(TensorDataset(X_t[split:], y_t[split:]), batch_size=BATCH)

model = TurbulenceLSTM()

opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

best = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()

    for xb, yb in train_dl:
        opt.zero_grad()

        loss = criterion(model(xb), yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

    model.eval()

    val_loss = sum(criterion(model(xb), yb).item() for xb, yb in val_dl) / len(val_dl)

    sched.step()

    if val_loss < best:
        best = val_loss

        Path("models").mkdir(exist_ok=True)

        torch.save(model.state_dict(), "models/lstm_forecaster.pt")

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | val={val_loss:.4f}")

print(f"Done. Best val: {best:.4f}. Saved to models/lstm_forecaster.pt")
