"""Shared image-formation primitives for the attack harness.

These are deliberately simple, documented models — enough to exercise every
code path of the defence layers and to measure them reproducibly. They are not
a substitute for physical PAIs or real injection tooling (see docs/02).
"""

from __future__ import annotations

import cv2
import numpy as np


def to_float(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def to_u8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)


def jpeg(img_u8: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def sensor_noise(img_f: np.ndarray, rng: np.random.Generator, read_sigma: float, shot_gain: float) -> np.ndarray:
    """Poisson-Gaussian sensor noise in linear-ish space (values in 0..1).

    sigma(x) = sqrt(read^2 + gain * x), the standard heteroscedastic camera model.
    """
    sigma = np.sqrt((read_sigma / 255.0) ** 2 + shot_gain * np.clip(img_f, 0, 1) / 255.0)
    return img_f + rng.standard_normal(img_f.shape).astype(np.float32) * sigma


def small_motion(img: np.ndarray, rng: np.random.Generator, state: np.ndarray | None, strength: float = 1.0):
    """Smooth random-walk head motion (rotation, translation, scale).

    Returns the warped image and the new motion state [angle, tx, ty, log_scale].
    """
    if state is None:
        state = np.zeros(4, np.float32)
    step = rng.normal(0, [0.35, 0.45, 0.45, 0.004]).astype(np.float32) * strength
    state = 0.85 * state + step
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(state[0]), float(np.exp(state[3])))
    M[:, 2] += state[1:3]
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out, state


def warp_planar(img: np.ndarray, rng: np.random.Generator, max_shift: float = 0.04) -> np.ndarray:
    """Random homography — a flat artefact (paper / screen) held at an angle."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32) * np.float32([w, h])
    H = cv2.getPerspectiveTransform(src, src + jitter)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def color_transfer(src_f: np.ndarray, ref_f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reinhard colour transfer in Lab, statistics taken inside ``mask``."""
    src_lab = cv2.cvtColor(np.clip(src_f, 0, 1), cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(np.clip(ref_f, 0, 1), cv2.COLOR_BGR2LAB)
    m = mask > 0.5
    if m.sum() < 20:
        return src_f
    out = src_lab.copy()
    for c in range(3):
        s, r = src_lab[..., c][m], ref_lab[..., c][m]
        out[..., c] = (src_lab[..., c] - s.mean()) / (s.std() + 1e-6) * r.std() + r.mean()
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def random_deform_mask(mask: np.ndarray, rng: np.random.Generator, strength: float = 6.0) -> np.ndarray:
    """Elastic deformation + random blur of a blending mask (SBI-style)."""
    h, w = mask.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 12) * strength * 8
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 12) * strength * 8
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m = cv2.remap(mask, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    k = int(rng.choice([3, 5, 7, 9, 11, 15]))
    m = cv2.GaussianBlur(m, (k, k), 0)
    # SBI: random "blend ratio" so the boundary is sometimes very subtle
    return np.clip(m, 0, 1) * float(rng.choice([1.0, 1.0, 0.75, 0.5]))
