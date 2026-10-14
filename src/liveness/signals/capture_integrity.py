"""Capture-integrity / injection-attack detection.

Two complementary parts:

1. **Integrity rules** on device telemetry (deterministic, explainable):
   virtual-camera drivers, browser automation, emulators, hooking frameworks,
   failed platform attestation, track/frame resolution mismatch. Cheap and
   precise against commodity tooling — and trivially bypassed by a careful
   attacker who renames the driver, which is why part 2 exists.

2. **Stream-authenticity features** computed from the frames themselves:
   * photometric challenge response (does the face reflect *this session's*
     nonce colours, with camera-like latency, more than the background?)
   * frame-timing statistics (virtual cameras are metronome-perfect or
     duplicate frames when re-timing a file)
   * sensor-noise floor and effective resolution (re-encoded / upscaled
     streams lose the camera's noise and high-frequency content)
   A gradient-boosted model turns these into a calibrated attack probability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from src.liveness.face import face_ellipse_mask

MAX_LAG = 12
EXPECTED_LAG = (0, 1, 2)  # camera pipeline latency window, in frames (calibrate per platform)

VIRTUAL_CAMERA_PATTERNS = [
    r"\bobs\b",
    r"virtual",
    r"manycam",
    r"xsplit",
    r"snap camera",
    r"e2esoft",
    r"\bvcam\b",
    r"droidcam",
    r"splitcam",
    r"dummy video",
    r"v4l2loopback",
    r"camtwist",
    r"iriun",
    r"epoccam",
    r"youcam",
    r"fake",
]
# Virtual cameras that honest users run (background blur, presenters). Treated as
# a *soft* finding -> step-up instead of reject, to protect BPCER.
BENIGN_VIRTUAL_CAMERAS = [r"nvidia broadcast", r"mmhmm", r"camo\b"]
EMULATOR_FINGERPRINTS = [
    r"generic",
    r"sdk_gphone",
    r"emulator",
    r"ranchu",
    r"goldfish",
    r"vbox",
    r"genymotion",
    r"emu64",
]
GOOD_ATTESTATION = {"MEETS_DEVICE_INTEGRITY", "MEETS_STRONG_INTEGRITY", "APP_ATTEST_VALID"}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # "hard" -> reject-worthy on its own, "soft" -> raises friction
    detail: str


def _match(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return p
    return None


def integrity_rules(tel: dict) -> list[Finding]:
    out: list[Finding] = []
    label = str(tel.get("camera_label", ""))
    if _match(BENIGN_VIRTUAL_CAMERAS, label):
        out.append(Finding("VIRTUAL_CAMERA_BENIGN", "soft", f"camera label '{label}'"))
    elif (p := _match(VIRTUAL_CAMERA_PATTERNS, label)) is not None:
        out.append(Finding("VIRTUAL_CAMERA", "hard", f"camera label '{label}' matches /{p}/"))
    if tel.get("navigator_webdriver"):
        out.append(Finding("BROWSER_AUTOMATION", "hard", "navigator.webdriver is true"))
    emu = tel.get("emulator_signals") or {}
    fp = str(emu.get("build_fingerprint", ""))
    if emu.get("qemu") or _match(EMULATOR_FINGERPRINTS, fp):
        out.append(Finding("EMULATOR", "hard", f"emulator fingerprint '{fp}'"))
    elif tel.get("platform") == "android" and emu and int(emu.get("sensor_count", 99)) < 5:
        out.append(Finding("SENSORS_MISSING", "soft", f"only {emu.get('sensor_count')} sensors"))
    hooks = tel.get("hooking_signals") or {}
    found = [k for k, v in hooks.items() if v]
    if found:
        out.append(Finding("HOOKING_FRAMEWORK", "hard", f"detected: {', '.join(found)}"))
    att = tel.get("attestation")
    if tel.get("platform") in ("android", "ios"):
        if not att or not att.get("present"):
            out.append(Finding("ATTESTATION_MISSING", "soft", "no platform attestation token"))
        elif att.get("verdict") not in GOOD_ATTESTATION:
            out.append(Finding("ATTESTATION_FAILED", "hard", f"verdict {att.get('verdict')}"))
    ts, fs = tel.get("track_settings") or {}, tel.get("frame_size")
    if ts and fs and (int(ts.get("width", 0)), int(ts.get("height", 0))) != (int(fs[0]), int(fs[1])):
        out.append(Finding("RESOLUTION_MISMATCH", "soft", f"track {ts.get('width')}x{ts.get('height')} vs frames {fs}"))
    return out


# ---------------------------------------------------------------------------
# Stream features
# ---------------------------------------------------------------------------
CI_FEATURES = [
    "chal_corr_early",
    "chal_corr_late",
    "chal_best_lag",
    "chal_corr_face_early",
    "chal_amp_face",
    "chal_bg_ratio",
    "ts_std",
    "ts_mean_err",
    "dup_ratio",
    "noise_spatial",
    "noise_temporal",
    "hf_ratio",
]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def _immerkaer_sigma(gray: np.ndarray) -> float:
    """Fast noise-sigma estimate (Immerkaer 1996)."""
    k = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def lag_profile(frames: list[np.ndarray], lmk: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Correlation between the face-minus-background colour response and the
    challenge, for lags 0..MAX_LAG frames (used for plots / diagnostics)."""
    shape = frames[0].shape[:2]
    face = face_ellipse_mask(shape, lmk, grow=0.85) > 0.5
    bg = face_ellipse_mask(shape, lmk, grow=1.7) < 0.05
    if bg.sum() < 80:
        bg = np.zeros(shape, bool)
        bg[:6], bg[-6:], bg[:, :6], bg[:, -6:] = True, True, True, True
    fx = np.stack([f[face].reshape(-1, 3).astype(np.float32).mean(0) for f in frames])
    bx = np.stack([f[bg].reshape(-1, 3).astype(np.float32).mean(0) for f in frames])
    d = (fx / (fx.mean(0) + 1e-6)) - (bx / (bx.mean(0) + 1e-6))
    c = colors[: len(frames)] - colors[: len(frames)].mean(0)
    T = len(frames)
    return np.array([_pearson(d[L:].ravel(), c[: T - L].ravel()) for L in range(MAX_LAG + 1)])


def photometric_response(frames: list[np.ndarray], lmk: np.ndarray, colors: np.ndarray) -> dict[str, float]:
    shape = frames[0].shape[:2]
    face = face_ellipse_mask(shape, lmk, grow=0.85) > 0.5
    bg = face_ellipse_mask(shape, lmk, grow=1.7) < 0.05
    if bg.sum() < 80:  # face fills the crop: fall back to a thin border
        bg = np.zeros(shape, bool)
        bg[:6], bg[-6:], bg[:, :6], bg[:, -6:] = True, True, True, True
    fx = np.stack([f[face].reshape(-1, 3).astype(np.float32).mean(0) for f in frames])
    bx = np.stack([f[bg].reshape(-1, 3).astype(np.float32).mean(0) for f in frames])
    s_f = fx / (fx.mean(0) + 1e-6) - 1
    s_b = bx / (bx.mean(0) + 1e-6) - 1
    d = s_f - s_b
    c = colors[: len(frames)] - colors[: len(frames)].mean(0)
    T = len(frames)
    corr_d = np.array([_pearson(d[L:].ravel(), c[: T - L].ravel()) for L in range(MAX_LAG + 1)])
    corr_f = np.array([_pearson(s_f[L:].ravel(), c[: T - L].ravel()) for L in range(MAX_LAG + 1)])
    best = int(np.argmax(corr_d))
    early = max(EXPECTED_LAG)
    cc = c[: T - best].ravel()
    denom = float((cc * cc).sum()) + 1e-9
    amp_f = float((s_f[best:].ravel() * cc).sum() / denom)
    amp_b = float((s_b[best:].ravel() * cc).sum() / denom)
    return {
        "chal_corr_early": float(corr_d[: early + 1].max()),
        "chal_corr_late": float(corr_d[early + 1 :].max()),
        "chal_best_lag": float(best),
        "chal_corr_face_early": float(corr_f[: early + 1].max()),
        "chal_amp_face": amp_f,
        "chal_bg_ratio": float(np.clip(amp_b / (abs(amp_f) + 1e-4), -3, 3)),
    }


def stream_features(
    frames: list[np.ndarray], timestamps_ms: np.ndarray, lmk: np.ndarray, colors: np.ndarray, claimed_fps: float = 30.0
) -> dict[str, float]:
    feats = photometric_response(frames, lmk, colors)
    iv = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
    expected = 1000.0 / claimed_fps
    feats["ts_std"] = float(np.std(iv))
    feats["ts_mean_err"] = float(abs(np.median(iv) - expected) / expected)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = np.array([np.abs(grays[i].astype(np.int16) - grays[i - 1]).mean() for i in range(1, len(grays))])
    feats["dup_ratio"] = float((diffs < 0.25).mean())
    feats["noise_spatial"] = float(np.median([_immerkaer_sigma(g) for g in grays[::5]]))
    shape = frames[0].shape[:2]
    bg = face_ellipse_mask(shape, lmk, grow=1.7) < 0.05
    if bg.sum() < 80:
        bg = np.ones(shape, bool)
    td = [
        np.median(np.abs(grays[i].astype(np.float32) - grays[i - 1])[bg])
        for i in range(1, len(grays))
        if diffs[i - 1] >= 0.25
    ]
    feats["noise_temporal"] = float(np.median(td)) if td else 0.0
    mid = grays[len(grays) // 2].astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(mid - mid.mean()))) ** 2
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - mag.shape[0] / 2, xx - mag.shape[1] / 2) / (min(mag.shape) / 2)
    feats["hf_ratio"] = float(np.log10(mag[r > 0.5].sum() / (mag.sum() + 1e-9) + 1e-9))
    return feats


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in CI_FEATURES], dtype=np.float32)


# Delayed-reflection rule (added after red-team iteration 1, sized on the *val* split:
# 0 % of bona fide sessions trigger it). Physically: a reflection of this session's
# challenge that peaks later than the camera-latency window was re-rendered by a
# processing pipeline (real-time face swap, challenge-aware relighting).
DELAY_MIN_CORR = 0.4
DELAY_MARGIN = 0.15


def challenge_findings(feats: dict[str, float]) -> list[Finding]:
    """Decision-grade findings from the photometric challenge (soft -> step-up:
    an honest device with an unusually slow camera pipeline can retry natively)."""
    if (
        feats["chal_best_lag"] > max(EXPECTED_LAG)
        and feats["chal_corr_late"] >= DELAY_MIN_CORR
        and feats["chal_corr_late"] > feats["chal_corr_early"] + DELAY_MARGIN
    ):
        return [
            Finding(
                "CHALLENGE_REFLECTION_DELAYED",
                "soft",
                f"reflection peaks {int(feats['chal_best_lag'])} frames late (camera window <= {max(EXPECTED_LAG)})",
            )
        ]
    return []


def explain(feats: dict[str, float]) -> list[Finding]:
    """Human-readable reasons derived from the stream features (for reviewers)."""
    out: list[Finding] = []
    if feats["chal_corr_early"] < 0.25 and feats["chal_corr_late"] < 0.35:
        if feats["chal_corr_face_early"] > 0.4:
            out.append(Finding("FLAT_REFLECTION", "soft", "challenge reflected uniformly (flat artefact?)"))
        else:
            out.append(Finding("CHALLENGE_NOT_REFLECTED", "soft", "no reflection of this session's challenge"))
    if feats["ts_std"] < 0.3:
        out.append(Finding("SYNTHETIC_FRAME_TIMING", "soft", f"frame interval std {feats['ts_std']:.2f} ms"))
    if feats["dup_ratio"] > 0.05:
        out.append(Finding("DUPLICATED_FRAMES", "soft", f"{feats['dup_ratio']:.0%} duplicated frames"))
    return out
