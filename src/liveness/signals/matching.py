"""1:1 face matching with SFace (OpenCV Zoo, Apache-2.0).

Used to check the live capture against a reference (ID-document portrait or
enrolment photo). A successful face swap *defeats* this check by design —
the swapped face is the victim's — which is why matching alone is never a
liveness signal and why IAPMR is reported for face-swap species.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from src.liveness.face import FaceBox


class FaceMatcher:
    def __init__(self, model_path: str):
        self._rec = cv2.FaceRecognizerSF.create(model_path, "")
        self._lock = threading.Lock()

    def embed(self, img_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        with self._lock:
            aligned = self._rec.alignCrop(img_bgr, box.as_yunet_row())
            return self._rec.feature(aligned).copy()

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        with self._lock:
            return float(self._rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))
