import torch
import numpy as np
from collections import deque
from core.lstm_model import TurbulenceLSTM


SEQ_LEN = 30
NORM_SCALE = np.array([1.0, 5000.0, 8.0, 1.0], dtype=np.float32)


class Forecaster:
    def __init__(self, model_path: str = "models/lstm_forecaster.pt "):
        self.model = TurbulenceLSTM()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.loaded = True
        except FileNotFoundError:
            self.loaded = False
            self.window: deque = deque(maxlen=SEQ_LEN)

    def update(
        self, TI: float, Re: float, density_max: float, pressure_max: float
    ) -> float:
        row = (
            np.array([TI, Re, density_max, pressure_max], dtype=np.float32) / NORM_SCALE
        )
        self.window.append(row)
        if len(self.window) < SEQ_LEN:
            return TI
        if not self.loaded:
            ti_series = np.array([r[0] for r in self.window])
            trend = np.polyfit(np.arange(SEQ_LEN), ti_series, 1)[0]
            return float(max(0, TI + trend * SEQ_LEN))
        x = torch.tensor(list(self.window)).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(x).item())
