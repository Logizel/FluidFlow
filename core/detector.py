import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

PERSON_CLASS_ID = 0


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PersonDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, bgr_frame: np.ndarray) -> list[Detection]:
        results = self.model(
            bgr_frame, classes=[PERSON_CLASS_ID], conf=self.conf, verbose=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(x1, y1, x2, y2, float(box.conf)))

        return detections

    def to_density_grid(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_w: int,
        grid_n: int = 32,
        meters_per_pixel: float = 0.02,
    ) -> np.ndarray:

        grid = np.zeros((grid_n, grid_n), dtype=np.float32)

        cell_h = frame_h / grid_n
        cell_w = frame_w / grid_n

        cell_area_m2 = max(
            (cell_h * meters_per_pixel) * (cell_w * meters_per_pixel), 0.01
        )

        for det in detections:
            cx, cy = det.center

            row = int(min(cy / cell_h, grid_n - 1))
            col = int(min(cx / cell_w, grid_n - 1))

            grid[row, col] += 1

        return grid / cell_area_m2
