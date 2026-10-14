"""Frame / texture artefact analysis for presentation attacks (print, replay).

Hand-crafted, fast and explainable features — the classic colour-texture PAD
recipe (Boulkenafet et al., "Face Spoofing Detection Using Colour Texture
Analysis", IEEE TIFS 2016) plus frequency-domain moiré and recapture cues:

* uniform LBP histograms on Y, Cr, Cb, H, S, V channels (print/screen
  re-capture changes micro-texture and chroma distribution)
* radial (azimuthally averaged) log-spectrum + peak-to-median ratio of the
  mid/high band (moiré, halftone and pixel-grid peaks)
* saturation / dynamic-range / sharpness statistics (gamut compression, glare)
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

N_RADIAL = 24
_LBP_BINS = 59  # P=8 non-rotation-invariant uniform patterns


def _lbp_hist(ch: np.ndarray) -> np.ndarray:
    codes = local_binary_pattern(ch, P=8, R=1, method="nri_uniform")
    hist, _ = np.histogram(codes, bins=_LBP_BINS, range=(0, _LBP_BINS))
    return hist.astype(np.float32) / max(hist.sum(), 1)


def radial_spectrum(gray: np.ndarray, n_bins: int = N_RADIAL) -> tuple[np.ndarray, float]:
    g = gray.astype(np.float32) - gray.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g * win))))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    bins = np.clip((r * n_bins).astype(int), 0, n_bins)
    prof = np.bincount(bins.ravel(), mag.ravel(), minlength=n_bins + 1)[:n_bins]
    cnt = np.bincount(bins.ravel(), minlength=n_bins + 1)[:n_bins]
    prof = prof / np.maximum(cnt, 1)
    prof = prof - prof.mean()
    band = mag[(r > 0.35) & (r < 0.95)]
    peak = float(np.percentile(band, 99.7) - np.median(band))
    return prof.astype(np.float32), peak


def pad_features(crop_bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lbp = [_lbp_hist(ycc[..., i]) for i in range(3)] + [_lbp_hist(hsv[..., i]) for i in range(3)]
    gray = ycc[..., 0]
    prof, peak = radial_spectrum(gray)
    y = gray.astype(np.float32)
    s = hsv[..., 1].astype(np.float32) / 255.0
    stats = np.float32(
        [
            peak,
            s.mean(),
            s.std(),
            (np.percentile(y, 99) - np.percentile(y, 1)) / 255.0,
            np.log1p(cv2.Laplacian(y, cv2.CV_32F).var()),
            (y > 245).mean(),  # glare / clipped highlights
        ]
    )
    return np.concatenate(lbp + [prof, stats]).astype(np.float32)


PAD_FEATURE_DIM = 6 * _LBP_BINS + N_RADIAL + 6
