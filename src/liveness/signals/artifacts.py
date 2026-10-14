"""Hand-crafted face-manipulation forensics (complements the CNN).

A face swap pastes a *generated, resampled* face into a camera frame, so the
inner face and its surroundings disagree in ways that are cheap to measure:

* noise level (generated faces lack the sensor noise of the rest of the frame)
* sharpness (generator resolution bottleneck -> blurrier face)
* high-frequency energy and periodic spectral peaks (up-sampling layers)
* colour statistics across the blending seam

Ratios are used (inside / outside) so global camera quality cancels out.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.liveness.face import face_ellipse_mask

ART_FEATURES = [
    "noise_ratio",
    "sharp_ratio",
    "hf_ratio_face",
    "seam_grad_ratio",
    "seam_color_jump",
    "chroma_ratio",
    "periodic_peak",
    "noise_face",
]


def _noise_map(gray: np.ndarray) -> np.ndarray:
    k = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    return np.abs(cv2.filter2D(gray, -1, k))


def _periodic_peak(gray: np.ndarray) -> float:
    """Energy at the up-sampling frequencies (N/2, N/4) of the high-pass residual,
    relative to its neighbourhood — checkerboard / transposed-conv fingerprints."""
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    mag = np.abs(np.fft.fft2(hp))
    n = mag.shape[0]
    peaks = []
    for f in (n // 2, n // 4, 3 * n // 4):
        for a, b in ((f, 0), (0, f), (f, f)):
            peaks.append(mag[a % n, b % n])
    return float(np.log((np.mean(peaks) + 1e-6) / (np.median(mag) + 1e-6)))


def artifact_features(crop_bgr: np.ndarray, lmk: np.ndarray) -> np.ndarray:
    shape = crop_bgr.shape[:2]
    inner = face_ellipse_mask(shape, lmk, grow=0.75) > 0.5
    outer_e = face_ellipse_mask(shape, lmk, grow=1.3) > 0.5
    ring = (face_ellipse_mask(shape, lmk, grow=1.6) > 0.5) & ~outer_e
    seam_in = (face_ellipse_mask(shape, lmk, grow=1.0) > 0.5) & ~(face_ellipse_mask(shape, lmk, grow=0.85) > 0.5)
    seam_out = outer_e & ~(face_ellipse_mask(shape, lmk, grow=1.1) > 0.5)
    seam = (face_ellipse_mask(shape, lmk, grow=1.12) > 0.5) & ~(face_ellipse_mask(shape, lmk, grow=0.92) > 0.5)
    if ring.sum() < 50:
        ring = ~outer_e
    if ring.sum() < 50 or inner.sum() < 50 or seam.sum() < 20:
        return np.zeros(len(ART_FEATURES), np.float32)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    nm = _noise_map(gray)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gm = np.hypot(gx, gy)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def r(a, m1, m2, fn=np.median):
        return float(np.log((fn(a[m1]) + 1e-3) / (fn(a[m2]) + 1e-3)))

    # high-frequency share inside the face (resampled faces lose it)
    face_box = gray.copy()
    face_box[~inner] = gray[inner].mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(face_box - face_box.mean()))) ** 2
    yy, xx = np.indices(spec.shape)
    rad = np.hypot(yy - spec.shape[0] / 2, xx - spec.shape[1] / 2) / (min(spec.shape) / 2)
    hf = float(np.log10(spec[rad > 0.5].sum() / (spec.sum() + 1e-9) + 1e-9))

    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    color_jump = float(np.linalg.norm(lab[seam_in].mean(0) - lab[seam_out].mean(0)))
    return np.float32(
        [
            r(nm, inner, ring),
            r(lap, inner, ring, np.mean),
            hf,
            r(gm, seam, inner | ring, np.mean),
            color_jump,
            r(chroma, inner, ring, np.mean),
            _periodic_peak(gray),
            float(np.median(nm[inner])),
        ]
    )
