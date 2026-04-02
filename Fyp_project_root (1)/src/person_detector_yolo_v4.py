from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOPersonDetector:
    """
    YOLOv8s-based person detector.
    Returns cropped person regions.
    """

    def __init__(self, model_name: str | None = None, conf: float = 0.4):
        from project_paths import yolo_weights

        weights = model_name or yolo_weights("yolov8s.pt")
        self.model = YOLO(weights)
        self.conf = conf
        # COCO class 0 = person
        self.person_cls = 0

    def detect_person_roi(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Largest COCO person box as (x1, y1, x2, y2) in pixels, or None."""
        h, w, _ = bgr.shape
        results = self.model.predict(
            bgr[..., ::-1],
            conf=self.conf,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        person_boxes = [
            xyxy[i] for i in range(len(xyxy)) if int(cls[i]) == self.person_cls
        ]
        if not person_boxes:
            return None
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in person_boxes]
        x1, y1, x2, y2 = person_boxes[int(np.argmax(areas))]
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w - 1)
        y2 = min(int(y2), h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def detect_and_crop(self, bgr: np.ndarray) -> np.ndarray:
        """
        Detects person in BGR frame and returns cropped BGR image.
        If no person, returns original frame.
        """
        h, w, _ = bgr.shape
        roi = self.detect_person_roi(bgr)
        if roi is None:
            return bgr
        x1, y1, x2, y2 = roi
        cropped = bgr[y1:y2, x1:x2].copy()
        if cropped.size == 0:
            return bgr
        return cropped
