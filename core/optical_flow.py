from types import GeneratorType
import cv2
import numpy as np


FLOW_H, FLOW_W = 180, 320


class OpticalFlowEngine:
    FARNEBACK_PARAMS = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    def __init__(self, smooth_kernel: int = 21):
        self.prev_gray: np.ndarray | None = None
        self.smooth_kernel = smooth_kernel

    def update(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        small = cv2.resize(bgr_frame, (FLOW_H, FLOW_W))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.FARNEBACK_PARAMS
        )
        self.prev_gray = gray
        k = self.smooth_kernel
        flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (k, k), 0)
        flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (k, k), 0)
        return np.clip(flow, -25.0, 25.0)

        @staticmethod
        def to_magnitude(flow: np.ndarray) -> np.ndarray:
            return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
