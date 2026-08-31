from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .detector import Detection

BOX_COLOR = (0, 224, 255)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (28, 28, 28)
ACCENT_COLOR = (0, 170, 255)


class FpsMeter:
    def __init__(self) -> None:
        self._last_time = perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = perf_counter()
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed > 0:
            instant = 1.0 / elapsed
            self.fps = instant if self.fps == 0.0 else (self.fps * 0.85) + (instant * 0.15)
        return self.fps


def draw_overlay(frame: np.ndarray, detections: list[Detection], fps: float) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        left, top, right, bottom = detection.box
        cv2.rectangle(output, (left, top), (right, bottom), BOX_COLOR, 2)
        caption = f"{detection.label}: {detection.confidence:.0%}"
        _draw_label(output, caption, left, top)

    _draw_status_panel(output, detections, fps)
    return output


def save_frame(frame: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"capture-{int(perf_counter() * 1000)}.jpg"
    path = output_dir / filename
    cv2.imwrite(str(path), frame)
    return path


def _draw_label(frame: np.ndarray, text: str, left: int, top: int) -> None:
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_width, text_height = text_size
    label_top = max(top - text_height - 8, 0)
    cv2.rectangle(
        frame,
        (left, label_top),
        (left + text_width + 8, label_top + text_height + 8),
        BOX_COLOR,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (left + 4, label_top + text_height + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_status_panel(frame: np.ndarray, detections: list[Detection], fps: float) -> None:
    height, width = frame.shape[:2]
    panel_height = 92
    cv2.rectangle(frame, (0, height - panel_height), (width, height), PANEL_COLOR, cv2.FILLED)
    cv2.line(frame, (0, height - panel_height), (width, height - panel_height), ACCENT_COLOR, 2)

    best = detections[0] if detections else None
    label = best.label.upper() if best else "NO OBJECT"
    confidence = f"{best.confidence:.0%}" if best else "--"

    cv2.putText(
        frame,
        label,
        (18, height - 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence}",
        (18, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (18, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

