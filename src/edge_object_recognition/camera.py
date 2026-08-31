from __future__ import annotations

import cv2


class Camera:
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. Try a different --camera value."
            )

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera returned no frame.")
        return frame

    def release(self) -> None:
        self.capture.release()

