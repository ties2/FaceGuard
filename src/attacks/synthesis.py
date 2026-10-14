"""Digital face manipulation instruments: self-blended images (training-only
pseudo-fakes), cross-identity face swaps and GAN-style synthesis artefacts."""

from __future__ import annotations

import cv2
import numpy as np

from src.attacks.common import color_transfer, random_deform_mask, to_float
from src.liveness.face import face_ellipse_mask


def _jitter_appearance(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Colour + frequency augmentation applied to the SBI 'source' copy."""
    f = f * rng.uniform(0.9, 1.1, 3).astype(np.float32) + rng.uniform(-0.04, 0.04, 3).astype(np.float32)
    hsv = cv2.cvtColor(np.clip(f, 0, 1), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 1)
    f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    op = rng.integers(0, 3)
    if op == 0:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.3))
    elif op == 1:
        blur = cv2.GaussianBlur(f, (0, 0), 1.0)
        f = f + rng.uniform(0.3, 1.0) * (f - blur)
    else:
        h, w = f.shape[:2]
        s = rng.uniform(0.45, 0.8)
        f = cv2.resize(cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
    return np.clip(f, 0, 1)


def self_blend(face_u8: np.ndarray, lmk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Self-Blended Image (Shiohara & Yamasaki, CVPR 2022).

    Blend a face with a slightly transformed, re-coloured copy of *itself*.
    The result carries the generic artefacts of face swapping (blending
    boundary, colour/frequency mismatch, landmark misalignment) without any
    identity change — so a detector trained on SBIs learns the artefacts, not
    a particular deepfake generator. Returns float BGR 0..1.
    """
    tgt = to_float(face_u8)
    src = _jitter_appearance(tgt.copy(), rng)
    h, w = tgt.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), rng.uniform(0.96, 1.04))
    M[:, 2] += rng.uniform(-3, 3, 2)
    src = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    mask = face_ellipse_mask((h, w), lmk, grow=rng.uniform(0.85, 1.1))
    mask = random_deform_mask((mask > 0.5).astype(np.float32), rng, strength=rng.uniform(2, 6))
    return mask[..., None] * src + (1 - mask[..., None]) * tgt


def face_swap(
    target_u8: np.ndarray,
    target_lmk: np.ndarray,
    source_u8: np.ndarray,
    source_lmk: np.ndarray,
    rng: np.random.Generator,
    method: str | None = None,
    target_f: np.ndarray | None = None,
    blend_mask: np.ndarray | None = None,
    color_ref_f: np.ndarray | None = None,
) -> np.ndarray:
    """Put the *source* (victim) face on the *target* (attacker) head.

    Mimics an autoencoder/GAN face-swap pipeline: landmark-aligned warp,
    generator resolution bottleneck, Reinhard colour transfer to the target
    frame, then alpha-feathered or Poisson blending. ``target_f`` lets the
    real-time simulator pass an already-illuminated float frame;
    ``color_ref_f`` (default: the target) is where colour statistics come from.
    """
    tgt = to_float(target_u8) if target_f is None else target_f
    h, w = tgt.shape[:2]
    M, _ = cv2.estimateAffinePartial2D(source_lmk, target_lmk, method=cv2.LMEDS)
    if M is None:
        M = np.float32([[1, 0, 0], [0, 1, 0]])
    src = cv2.warpAffine(to_float(source_u8), M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # generator bottleneck: faces are synthesised at a lower resolution
    s = rng.uniform(0.45, 0.85)
    src = cv2.resize(cv2.resize(src, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
    if blend_mask is None:
        blend_mask = face_ellipse_mask((h, w), target_lmk, grow=rng.uniform(1.0, 1.15))
        blend_mask = random_deform_mask((blend_mask > 0.5).astype(np.float32), rng, strength=rng.uniform(0.5, 2.0))
        blend_mask = np.clip(blend_mask / max(blend_mask.max(), 1e-6), 0, 1)
    src = color_transfer(src, tgt if color_ref_f is None else color_ref_f, blend_mask)
    method = method or str(rng.choice(["alpha", "alpha", "poisson"]))
    if method == "poisson":
        m8 = (blend_mask > 0.5).astype(np.uint8) * 255
        ys, xs = np.nonzero(m8)
        if len(xs) > 50:
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            try:
                out = cv2.seamlessClone(
                    (np.clip(src, 0, 1) * 255).astype(np.uint8),
                    (np.clip(tgt, 0, 1) * 255).astype(np.uint8),
                    m8,
                    center,
                    cv2.NORMAL_CLONE,
                )
                return out.astype(np.float32) / 255.0
            except cv2.error:
                pass
    return blend_mask[..., None] * src + (1 - blend_mask[..., None]) * tgt


def gan_upsample(face_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fully synthetic face proxy: content re-generated through stride-2
    transposed-convolution-like upsamplers with slightly unequal taps, plus
    re-injected fine detail (generators produce sharp output). Leaves the
    periodic spectral peaks / faint checkerboard documented for GAN
    generators (Durall et al., CVPR 2020; Frank et al., ICML 2020)."""
    f = to_float(face_u8)
    h, w = f.shape[:2]
    stages = int(rng.choice([1, 1, 2]))
    x = cv2.resize(f, (w >> stages, h >> stages), interpolation=cv2.INTER_AREA)
    for _ in range(stages):
        hh, ww = x.shape[:2]
        up = np.zeros((hh * 2, ww * 2, 3), np.float32)
        up[::2, ::2] = x
        # bilinear transposed-conv kernel with perturbed (learned-looking) taps:
        # the output phases no longer receive equal weight -> checkerboard
        k1 = np.float32([0.25, 0.75, 0.75, 0.25]) * (1 + rng.uniform(-0.15, 0.15, 4)).astype(np.float32)
        k = np.outer(k1, k1)
        k = k * (4.0 / k.sum())
        x = cv2.filter2D(up, -1, k, borderType=cv2.BORDER_REFLECT_101)
    if x.shape[:2] != (h, w):
        x = cv2.resize(x, (w, h))
    detail = f - cv2.GaussianBlur(f, (0, 0), 1.2)
    x = x + rng.uniform(0.3, 0.8) * detail
    return np.clip(x, 0, 1)
