from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .camera import Camera


def save_raw_frame(frame: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{uuid4().hex}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not save image: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect unlabeled, raw camera images for training.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("datasets/raw"))
    args = parser.parse_args()

    # Keep sessions separate so near-identical views can stay in the same dataset split.
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output_dir = args.output / session
    camera = Camera(args.camera, args.width, args.height)
    print(f"Saving raw images to {output_dir}")
    print("Press s to capture, q or Esc to quit. Change angle/background between captures.")
    try:
        while True:
            frame = camera.read()
            cv2.imshow("Dataset Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print(f"Saved {save_raw_frame(frame, output_dir)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
