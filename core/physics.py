import re
import cv2
import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter

MU_CROWD = 0.35
CORRIDOR_W = 5.0
GRID_N = 32


class PhysicsEngine:
    def __init__(self, fps: float = 25.0, meters_per_pixel: float = 0.02):
        self.fps = fps
        self.mpp = meters_per_pixel
        self.prev_velocity: np.ndarray | None = None
        self.density_history: deque = deque(maxlen=5)

    def _detect_shockwave(self, vx: np.ndarray, vy: np.ndarray) -> int:
        if len(self.density_history) < 2:
            return 0
        H, W = vx.shape
        prev = cv2.resize(self.density_history[-2], (W, H))
        curr = cv2.resize(self.density_history[-1], (W, H))
        py, px = np.where(prev > 4.0)
        cy, cx = np.where(curr > 4.0)
        if len(px) == 0 or len(cx) == 0:
            return 0
        dcx = np.mean(cx) - np.mean(px)
        dcy = np.mean(cy) - np.mean(py)
        dot = dcx * float(np.mean(vx)) + dcy * float(np.mean(vy))
        return int(dot < 0.5)

    def compute(self, density_grid: np.ndarray, flow: np.ndarray) -> dict:
        vx = flow[..., 0] * self.mpp * self.fps
        vy = flow[..., 1] * self.mpp * self.fps
        if self.prev_velocity is None:
            self.prev_velocity = np.stack([vx, vy], axis=-1)
            accel_mag = np.zeros_like(vx)
        else:
            dvx = vx - self.prev_velocity[..., 0]
            dvy = vy - self.prev_velocity[..., 1]
            accel_mag = np.sqrt(dvx**2 + dvy**2) * self.fps
            self.prev_velocity = np.stack([vx, vy], axis=-1)
        H, W = vx.shape
        density_resized = cv2.resize(density_grid, (W, H))
        pressure = density_resized * accel_mag
        pressure_norm = pressure / (pressure.max() + 1e-8)
        dvx_dx = np.gradient(vx, axis=1)
        dvy_dy = np.gradient(vy, axis=0)
        div_raw = dvx_dx + dvy_dy
        vx_sm = gaussian_filter(vx, sigma=3.0)
        vy_sm = gaussian_filter(vy, sigma=3.0)
        div_sm = np.gradient(vx_sm, axis=1) - np.gradient(vy_sm, axis=0)
        turbulence = div_raw - div_sm
        TI = float(np.percentile(np.abs(turbulence), 90))
        v_mean = float(np.mean(np.sqrt(vx**2 + vy**2)))
        rho_mean = float(np.mean(density_grid))
        Re = (rho_mean * v_mean * CORRIDOR_W) / MU_CROWD
        self.density_history.append(density_grid.copy())
        shockwave_flag = self._detect_shockwave(vx, vy)
        return {
            "density_grid": density_grid,
            "velocity_x": vx,
            "velocity_y": vy,
            "pressure_norm": pressure_norm,
            "turbulence_map": turbulence,
            "TI": TI,
            "Re": Re,
            "density_max": float(density_grid.max()),
            "shockwave_flag": shockwave_flag,
        }
