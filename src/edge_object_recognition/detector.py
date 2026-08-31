from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .labels import MOBILENET_SSD_LABELS, label_for


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class MobileNetSSDDetector:
    """OpenCV DNN wrapper for the compact MobileNet-SSD demo model."""

    def __init__(
        self,
        prototxt_path: Path,
        model_path: Path,
        confidence_threshold: float = 0.5,
        labels: tuple[str, ...] = MOBILENET_SSD_LABELS,
    ) -> None:
        if not prototxt_path.exists():
            raise FileNotFoundError(f"Missing model config: {prototxt_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        self.confidence_threshold = confidence_threshold
        self.labels = labels

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results: list[Detection] = []
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(detections[0, 0, index, 1])
            left, top, right, bottom = detections[0, 0, index, 3:7] * np.array(
                [width, height, width, height]
            )
            box = self._clamp_box(
                int(left),
                int(top),
                int(right),
                int(bottom),
                width,
                height,
            )
            results.append(
                Detection(
                    label=label_for(class_id, self.labels),
                    confidence=confidence,
                    box=box,
                )
            )

        return sorted(results, key=lambda detection: detection.confidence, reverse=True)

    @staticmethod
    def _clamp_box(
        left: int,
        top: int,
        right: int,
        bottom: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        return (
            max(0, min(left, width - 1)),
            max(0, min(top, height - 1)),
            max(0, min(right, width - 1)),
            max(0, min(bottom, height - 1)),
        )

