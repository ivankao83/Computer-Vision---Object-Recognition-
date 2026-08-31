import numpy as np

from edge_object_recognition.detector import Detection
from edge_object_recognition.ui import draw_overlay


def test_draw_overlay_keeps_frame_shape() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [Detection(label="bottle", confidence=0.93, box=(10, 20, 120, 180))]

    output = draw_overlay(frame, detections, fps=18.0)

    assert output.shape == frame.shape
    assert output.sum() > 0

