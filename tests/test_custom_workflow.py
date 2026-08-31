import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from edge_object_recognition.app import parse_args
from edge_object_recognition.collect import save_raw_frame
from edge_object_recognition.yolo_detector import YOLODetector


def test_raw_capture_is_readable_and_does_not_modify_source(tmp_path):
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    original = frame.copy()
    first = save_raw_frame(frame, tmp_path / "session")
    second = save_raw_frame(frame, tmp_path / "session")

    assert first != second
    assert first.is_file() and second.is_file()
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(cv2.imread(str(first)), original)


def test_failed_image_write_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Could not save"):
        save_raw_frame(np.zeros((10, 10, 3), dtype=np.uint8), tmp_path)


def test_custom_detection_uses_trained_names_and_pixel_coordinates(tmp_path, monkeypatch):
    def tensor(data):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: data))

    result = SimpleNamespace(
        names={0: "my-component", 1: "my-tool"},
        boxes=SimpleNamespace(
            xyxy=tensor([[-5, 10, 400, 200], [2, 3, 50, 60], [20, 20, 10, 10]]),
            conf=tensor([0.6, 0.9, 0.99]),
            cls=tensor([0, 1, 0]),
        ),
    )
    model = SimpleNamespace(task="detect", predict=Mock(return_value=[result]))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: model))
    weights = tmp_path / "best.pt"
    weights.touch()
    detector = YOLODetector(weights, confidence_threshold=0.55, device="mps")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert [item.label for item in detections] == ["my-tool", "my-component"]
    assert detections[0].box == (2, 3, 50, 60)
    assert detections[1].box == (0, 10, 319, 200)
    assert detections[0].confidence == 0.9
    assert model.predict.call_args.kwargs["source"] is frame
    assert model.predict.call_args.kwargs["device"] == "mps"
    assert model.predict.call_args.kwargs["conf"] == 0.55

    result.boxes = None
    assert detector.detect(frame) == []


def test_missing_weights_do_not_trigger_automatic_model_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing custom model"):
        YOLODetector(tmp_path / "missing.pt")


def test_classification_weights_are_rejected(tmp_path, monkeypatch):
    weights = tmp_path / "classifier.pt"
    weights.touch()
    monkeypatch.setitem(
        sys.modules, "ultralytics",
        SimpleNamespace(YOLO=lambda path: SimpleNamespace(task="classify")),
    )
    with pytest.raises(ValueError, match="object detection model"):
        YOLODetector(weights)


def test_default_cli_keeps_mobilenet_and_custom_weights_are_optional(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app"])
    assert parse_args().weights is None
    monkeypatch.setattr(sys, "argv", ["app", "--weights", "best.pt", "--device", "mps"])
    args = parse_args()
    assert args.weights.name == "best.pt"
    assert args.device == "mps"
