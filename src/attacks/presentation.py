"""Presentation attack instruments (ISO/IEC 30107 sense): the artefact is shown
*to a real camera*. Output is the appearance of the artefact (float BGR 0..1)
before the camera model is applied by the session simulator."""

from __future__ import annotations

import cv2
import numpy as np

from src.attacks.common import to_float, warp_planar


def print_photo(face_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Printed photograph: compressed dynamic range and gamut, halftone dots,
    paper texture, uneven illumination, held at an angle."""
    f = to_float(face_u8)
    h, w = f.shape[:2]
    contrast = rng.uniform(0.55, 0.85)
    black = rng.uniform(0.04, 0.14)
    f = black + contrast * f
    gray = f.mean(axis=2, keepdims=True)
    f = gray + rng.uniform(0.55, 0.9) * (f - gray)  # desaturation (ink gamut)
    # halftone: ordered dot pattern at a random pitch/angle
    pitch = rng.uniform(1.6, 3.2)
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    v = -xx * np.sin(ang) + yy * np.cos(ang)
    dots = np.cos(2 * np.pi * u / pitch) * np.cos(2 * np.pi * v / pitch)
    f = f * (1 + rng.uniform(0.02, 0.07) * dots[..., None])
    # paper fibre texture (low-frequency) + uneven light gradient
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), rng.uniform(0.8, 2.0))
    f = f * (1 + rng.uniform(0.01, 0.04) * tex[..., None])
    gx = np.linspace(-1, 1, w, dtype=np.float32)[None, :] * rng.uniform(-0.12, 0.12)
    gy = np.linspace(-1, 1, h, dtype=np.float32)[:, None] * rng.uniform(-0.12, 0.12)
    f = f * (1 + gx + gy)[..., None]
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.1))
    return warp_planar(np.clip(f, 0, 1), rng, 0.05)


def screen_replay(face_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photo/video shown on a phone or monitor: moiré from the pixel grid
    beating against the camera sensor, gamma / colour-temperature shift,
    specular glare and refresh banding."""
    f = to_float(face_u8)
    h, w = f.shape[:2]
    f = np.power(np.clip(f, 0, 1), rng.uniform(0.8, 1.25))
    f = f * np.float32([rng.uniform(1.0, 1.12), rng.uniform(0.97, 1.03), rng.uniform(0.86, 1.0)])  # bluish
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    moire = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        fr = rng.uniform(0.12, 0.42)  # cycles / pixel
        a = rng.uniform(0, np.pi)
        moire += np.cos(2 * np.pi * fr * (xx * np.cos(a) + yy * np.sin(a)) + rng.uniform(0, 2 * np.pi))
    f = f * (1 + rng.uniform(0.025, 0.08) * moire[..., None])
    # sub-pixel RGB stripes
    stripe = (np.arange(w) % 3)[None, :]
    sp = np.stack([(stripe == k).astype(np.float32) for k in (2, 1, 0)], axis=-1)
    f = f * (1 - rng.uniform(0.0, 0.05) + rng.uniform(0.0, 0.1) * sp)
    # glare blob
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    r = rng.uniform(0.15, 0.45) * w
    glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    f = f + rng.uniform(0.04, 0.22) * glare[..., None]
    # rolling refresh band
    band = 1 + rng.uniform(0.0, 0.05) * np.sin(2 * np.pi * yy / rng.uniform(20, 60) + rng.uniform(0, 6.28))
    f = f * band[..., None]
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 0.9))
    return warp_planar(np.clip(f, 0, 1), rng, 0.04)
