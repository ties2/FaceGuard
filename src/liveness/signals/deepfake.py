"""Deepfake / face-swap layer: preprocessing + scoring of the served ONNX CNN."""

from __future__ import annotations

import numpy as np

MEAN, STD = 0.5, 0.25  # must match src/training/deepfake_net.py
FRAME_PICKS = (1 / 6, 1 / 2, 5 / 6)  # relative positions of the frames scored per session


def preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
    x = np.stack([c[..., ::-1] for c in crops_bgr]).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(0, 3, 1, 2).copy()


def pick_frames(n: int) -> list[int]:
    return [min(n - 1, int(round(p * n))) for p in FRAME_PICKS]


def score(onnx_model, crops_bgr: list[np.ndarray]) -> float:
    """Mean manipulation probability over the selected frames."""
    logits = onnx_model.run({onnx_model.input_names[0]: preprocess(crops_bgr)})[0].ravel()
    return float(np.mean(1.0 / (1.0 + np.exp(-logits))))
