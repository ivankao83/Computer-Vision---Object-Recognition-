from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import Detection


class YOLODetector:
    """Adapt a locally trained YOLO detection model to the webcam overlay."""

    def __init__(
        self,
        weights: Path,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"Missing custom model weights: {weights}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Install custom-model support: pip install -e ".[training]"') from exc

        self.model = YOLO(str(weights))
        if self.model.task != "detect":
            raise ValueError("Use an object detection model, not classification or segmentation weights.")
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        height, width = frame.shape[:2]
        detections = []
        for coords, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            left, top, right, bottom = coords
            box = (
                max(0, min(int(left), width - 1)),
                max(0, min(int(top), height - 1)),
                max(0, min(int(right), width - 1)),
                max(0, min(int(bottom), height - 1)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(Detection(str(result.names[int(class_id)]), float(confidence), box))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)
