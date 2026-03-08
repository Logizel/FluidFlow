import numpy as np
from pathlib import Path

np.random.seed(42)

N = 1200
SEQ_LEN = 60
PRED = 30


def simulate(kind: str) -> np.ndarray:
    T = SEQ_LEN + PRED
    t = np.linspace(0, 1, T)

    if kind == "laminar":
        density = 0.5 + 0.3 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.05, T)
        TI = 0.02 + 0.02 * np.random.rand(T)
        Re = 400 + 200 * np.random.rand(T)

    elif kind == "transitional":
        density = 2.5 + 0.8 * t + np.random.normal(0, 0.1, T)
        TI = 0.05 + 0.10 * t + np.random.normal(0, 0.01, T)
        Re = 1200 + 1500 * t + np.random.normal(0, 50, T)

    else:
        density = 3.5 + 2.0 * t**2 + np.random.normal(0, 0.15, T)
        TI = 0.10 + 0.40 * t**1.5 + np.random.normal(0, 0.02, T)
        Re = 2000 + 2500 * t**1.2 + np.random.normal(0, 100, T)

    pressure = np.clip(
        density * np.gradient(density) * 0.1 + np.random.normal(0, 0.01, T),
        0,
        None,
    )

    return np.stack(
        [
            np.clip(TI, 0, None),
            np.clip(Re, 0, None),
            np.clip(density, 0, None),
            pressure,
        ],
        axis=-1,
    ).astype(np.float32)


Path("cache").mkdir(exist_ok=True)

types = ["laminar", "transitional", "turbulent"]

X = []
y = []

for _ in range(N):
    data = simulate(np.random.choice(types))

    X.append(data[:SEQ_LEN])
    y.append(data[SEQ_LEN + PRED - 1, 0])
X = np.array(X)
y = np.array(y)

np.save("cache/train_X.npy", X)
np.save("cache/train_y.npy", y)

print(f"Saved {N} scenarios. Shape: {X.shape}")
