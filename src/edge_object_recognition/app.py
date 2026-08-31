from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .camera import Camera
from .detector import MobileNetSSDDetector
from .ui import FpsMeter, draw_overlay, save_frame

DEFAULT_PROTOTXT = "MobileNetSSD_deploy.prototxt"
DEFAULT_MODEL = "MobileNetSSD_deploy.caffemodel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live edge object recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0.")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--window-name", default="Edge Object Recognition")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector = MobileNetSSDDetector(
        prototxt_path=args.model_dir / DEFAULT_PROTOTXT,
        model_path=args.model_dir / DEFAULT_MODEL,
        confidence_threshold=args.confidence,
    )
    camera = Camera(args.camera, args.width, args.height)
    fps_meter = FpsMeter()

    print("Starting camera. Press q or Esc to quit, s to save an annotated frame.")
    try:
        while True:
            frame = camera.read()
            detections = detector.detect(frame)
            fps = fps_meter.tick()
            annotated = draw_overlay(frame, detections, fps)

            cv2.imshow(args.window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_frame(annotated, Path("captures"))
                print(f"Saved {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

