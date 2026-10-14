"""Face detection, landmarks and crops (YuNet, OpenCV Zoo — MIT licence).

All downstream layers work on a *context crop*: a square crop around the face
with a margin, so blending boundaries (face-swap) and the face/background
contrast (photometric challenge, print/replay) are inside the crop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

CROP_SIZE = 128
CONTEXT_SCALE = 1.35


@dataclass
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose tip, right mouth corner, left mouth corner
    score: float

    def as_yunet_row(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h, *self.landmarks.reshape(-1), self.score], dtype=np.float32)


class FaceDetector:
    """Thread-safe wrapper around ``cv2.FaceDetectorYN``."""

    def __init__(self, model_path: str, score_threshold: float = 0.6):
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 50)
        self._lock = threading.Lock()

    def detect(self, img_bgr: np.ndarray, pad_ratio: float = 0.25) -> FaceBox | None:
        """Return the most confident face. Pads the image first so tightly
        cropped selfies (face touching the borders) are still detected."""
        h, w = img_bgr.shape[:2]
        p = int(pad_ratio * max(h, w))
        fill = [int(v) for v in img_bgr.reshape(-1, 3).mean(axis=0)]
        padded = cv2.copyMakeBorder(img_bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=fill)
        with self._lock:
            self._det.setInputSize((padded.shape[1], padded.shape[0]))
            _, faces = self._det.detect(padded)
        if faces is None or len(faces) == 0:
            return None
        best = faces[np.argmax(faces[:, -1])]
        lmk = best[4:14].reshape(5, 2) - p
        return FaceBox(float(best[0] - p), float(best[1] - p), float(best[2]), float(best[3]), lmk, float(best[-1]))


def context_crop(
    img_bgr: np.ndarray, box: FaceBox, size: int = CROP_SIZE, scale: float = CONTEXT_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Square crop centred on the face; returns (crop, landmarks in crop coords).

    The crop is shrunk (down to the face box) before falling back to border
    replication, because replicated/reflected borders create artificial edges
    that a blending-artifact detector could latch onto.
    """
    H, W = img_bgr.shape[:2]
    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    side = scale * max(box.w, box.h)
    max_inside = 2.0 * min(cx, cy, W - cx, H - cy)
    side = max(min(side, max_inside), max(box.w, box.h))
    s = size / side
    M = np.array([[s, 0, size / 2.0 - s * cx], [0, s, size / 2.0 - s * cy]], dtype=np.float32)
    crop = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    lmk = (np.c_[box.landmarks, np.ones(5)] @ M.T).astype(np.float32)
    return crop, lmk


def face_ellipse_mask(shape: tuple[int, int], lmk: np.ndarray, grow: float = 1.0) -> np.ndarray:
    """Soft elliptical mask of the inner face from 5 landmarks (float32, 0..1)."""
    h, w = shape
    eyes = lmk[:2]
    mouth = lmk[3:5]
    iod = float(np.linalg.norm(eyes[1] - eyes[0])) + 1e-6
    angle = float(np.degrees(np.arctan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0])))
    center = (eyes.mean(axis=0) * 0.45 + mouth.mean(axis=0) * 0.55).astype(np.float32)
    axes = (int(round(iod * 1.05 * grow)), int(round(iod * 1.35 * grow)))
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(round(center[0])), int(round(center[1]))), axes, angle, 0, 360, 1.0, -1)
    k = max(3, int(iod * 0.25) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)
