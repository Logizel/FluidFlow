import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str, target_h: int = 360, target_w: int = 640):
        self.path = path
        self.target_h = target_h
        self.target_w = target_w
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video:{path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        frame = cv2.resize(frame, (self.target_w, self.target_h))
        self.frame_idx += 1
        return True, frame

    def to_rbg(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RBG)

    def to_gray(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
