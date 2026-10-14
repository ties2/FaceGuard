"""Capture-session simulator: bona fide sessions and every attack species.

A *session* is what the capture client sends after a challenge: a short burst
of face frames (context crops), their timestamps and device telemetry. Every
species is produced by the same image-formation steps so that differences
between classes come from the attack, not from the simulator:

    scene -> (screen light from the challenge) -> camera ISP -> encoder

Injection species skip the physical camera and enter after (or instead of) it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.attacks.common import jpeg, random_deform_mask, sensor_noise, small_motion, to_float, to_u8
from src.attacks.presentation import print_photo, screen_replay
from src.attacks.synthesis import face_swap, gan_upsample
from src.liveness.challenge import Challenge
from src.liveness.face import face_ellipse_mask

GENUINE_WEB_CAMERAS = [
    "FaceTime HD Camera",
    "Integrated Camera",
    "HD Pro Webcam C920",
    "Logitech BRIO",
    "USB2.0 HD UVC WebCam",
    "HP TrueVision HD Camera",
    "Integrated Webcam",
]
BENIGN_VIRTUAL_CAMERAS = ["NVIDIA Broadcast", "mmhmm Camera"]
VIRTUAL_CAMERAS = [
    "OBS Virtual Camera",
    "ManyCam Virtual Webcam",
    "XSplit VCam",
    "Snap Camera",
    "e2eSoft VCam",
    "DroidCam Source 3",
    "SplitCam Video Driver",
    "Dummy video device (0x0000)",
]


@dataclass
class CaptureSession:
    frames: list[np.ndarray]
    timestamps_ms: np.ndarray
    telemetry: dict
    species: str  # "bona_fide" or attack species id
    challenge: Challenge
    lmk: np.ndarray  # landmarks of the face in crop coordinates (first frame)
    meta: dict = field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        return self.species != "bona_fide"

    @property
    def audit_frame(self) -> np.ndarray:
        return self.frames[len(self.frames) // 2]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _isp_profile(rng: np.random.Generator) -> dict:
    """Real webcams vary a lot: some are noisy, many denoise aggressively."""
    if rng.random() < 0.55:
        return {
            "read": rng.uniform(1.5, 4.0),
            "shot": rng.uniform(0.3, 1.2),
            "denoise": 0.0,
            "q": int(rng.integers(72, 96)),
        }
    return {
        "read": rng.uniform(0.4, 1.4),
        "shot": rng.uniform(0.05, 0.4),
        "denoise": rng.uniform(0.4, 0.9),
        "q": int(rng.integers(60, 88)),
    }


def _camera(img_f: np.ndarray, rng: np.random.Generator, isp: dict) -> np.ndarray:
    x = sensor_noise(img_f, rng, isp["read"], isp["shot"])
    if isp["denoise"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), isp["denoise"])
    return jpeg(to_u8(x), isp["q"])


def _reflectance_map(shape, lmk, bg_level: float) -> np.ndarray:
    """Where the screen light lands: the face (close to the display) much more
    than the background. Heavily smoothed — real fall-off follows 3-D geometry,
    there is no sharp edge (unlike a blending seam)."""
    face = face_ellipse_mask(shape, lmk, grow=1.1)
    iod = float(np.linalg.norm(lmk[1] - lmk[0])) + 1e-6
    face = cv2.GaussianBlur(face, (0, 0), 0.35 * iod)
    face = face / max(float(face.max()), 1e-6)
    return face + bg_level * (1 - face)


def _lit(base_f: np.ndarray, R: np.ndarray, k: float, color: np.ndarray, mean_color: np.ndarray, awb: float):
    """Screen light reflected by the scene (multiplicative, albedo-scaled) and a
    partial auto-white-balance correction applied to the whole frame."""
    delta = (color - mean_color) * 2.0
    out = base_f * (1 + k * R[..., None] * delta[None, None, :])
    gain = 1.0 / (1 + awb * k * float(R.mean()) * delta)
    return out * gain[None, None, :]


def _natural_timestamps(n: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    base = 1000.0 / fps
    iv = base + rng.normal(0, rng.uniform(1.0, 4.0), n - 1)
    drops = rng.random(n - 1) < 0.04
    iv[drops] += base
    return np.concatenate([[0.0], np.cumsum(iv)])


def _synthetic_timestamps(n: int, fps: float, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """Virtual-camera timing: metronome-perfect, or a 25 fps file re-timed to 30 fps
    (every 6th frame duplicated). Returns timestamps and duplicated indices."""
    base = 1000.0 / fps
    ts = np.arange(n) * base + rng.normal(0, 0.05, n)
    dups = []
    if rng.random() < 0.5:
        dups = [i for i in range(n) if i % 6 == 5]
    return ts, dups


def _codec(frames_f: list[np.ndarray], rng: np.random.Generator, downscale: bool = True) -> list[np.ndarray]:
    """Video-file / streaming codec: lower effective resolution + stronger compression."""
    q = int(rng.integers(35, 72))
    s = rng.uniform(0.5, 0.85) if downscale and rng.random() < 0.6 else 1.0
    out = []
    for f in frames_f:
        x = to_u8(f)
        if s < 1.0:
            h, w = x.shape[:2]
            x = cv2.resize(cv2.resize(x, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
        out.append(jpeg(x, q))
    return out


def _browser_encode(frames_u8: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    q = int(rng.integers(75, 95))
    return [jpeg(f, q) for f in frames_u8]


# ---------------------------------------------------------------------------
# Telemetry profiles
# ---------------------------------------------------------------------------
def _genuine_telemetry(rng: np.random.Generator) -> dict:
    r = rng.random()
    if r < 0.7:
        w, h = [(1280, 720), (640, 480), (1920, 1080)][int(rng.integers(0, 3))]
        label = str(rng.choice(GENUINE_WEB_CAMERAS))
        fw, fh = w, h
        q = rng.random()
        if q < 0.03:  # honest users with background-blur / presenter virtual cameras
            label = str(rng.choice(BENIGN_VIRTUAL_CAMERAS))
        elif q < 0.04:  # webcam driver ignores the requested resolution
            fw, fh = 640, 480
        return {
            "platform": "web",
            "camera_label": label,
            "track_settings": {"width": w, "height": h, "frameRate": 30},
            "frame_size": [fw, fh],
            "media_devices_count": int(rng.integers(1, 3)),
            "navigator_webdriver": False,
        }
    platform = "android" if r < 0.9 else "ios"
    return {
        "platform": platform,
        "camera_label": "camera2 1, facing front" if platform == "android" else "Front Camera",
        "track_settings": {"width": 1280, "height": 720, "frameRate": 30},
        "frame_size": [1280, 720],
        "media_devices_count": int(rng.integers(2, 5)),
        "navigator_webdriver": False,
        "emulator_signals": {
            "qemu": False,
            "build_fingerprint": "google/oriole/oriole:14/release-keys",
            "sensor_count": int(rng.integers(12, 40)),
        },
        "hooking_signals": {"frida": False, "xposed": False, "magisk": False},
        # ~3 % of honest devices cannot produce an attestation token (old OS, flaky Play services)
        "attestation": {
            "present": bool(rng.random() > 0.03),
            "verdict": "MEETS_DEVICE_INTEGRITY" if platform == "android" else "APP_ATTEST_VALID",
        },
    }


def _vcam_telemetry(rng: np.random.Generator, stealth: bool) -> dict:
    if stealth:
        t = _genuine_telemetry(rng)
        if t["platform"] != "web":  # stealth desktop attack: force a desktop profile
            t = {
                "platform": "web",
                "camera_label": str(rng.choice(GENUINE_WEB_CAMERAS)),
                "track_settings": {"width": 1280, "height": 720, "frameRate": 30},
                "frame_size": [1280, 720],
                "media_devices_count": 1,
                "navigator_webdriver": False,
            }
        return t
    w, h = 1280, 720
    fw, fh = (w, h) if rng.random() < 0.6 else (640, 480)
    return {
        "platform": "web",
        "camera_label": str(rng.choice(VIRTUAL_CAMERAS)),
        "track_settings": {"width": w, "height": h, "frameRate": 30},
        "frame_size": [fw, fh],
        "media_devices_count": int(rng.integers(2, 5)),
        "navigator_webdriver": bool(rng.random() < 0.15),
    }


def _emulator_telemetry(rng: np.random.Generator) -> dict:
    emulator = rng.random() < 0.5
    hooks_visible = rng.random() < 0.6
    return {
        "platform": "android",
        "camera_label": "camera2 1, facing front",
        "track_settings": {"width": 1280, "height": 720, "frameRate": 30},
        "frame_size": [1280, 720],
        "media_devices_count": 2,
        "navigator_webdriver": False,
        "emulator_signals": {
            "qemu": bool(emulator and rng.random() < 0.7),
            "build_fingerprint": "google/sdk_gphone64_x86_64/emu64xa:14/userdebug"
            if emulator
            else "samsung/a52q/a52q:13/release-keys",
            "sensor_count": int(rng.integers(0, 4)) if emulator else int(rng.integers(15, 35)),
        },
        "hooking_signals": {
            "frida": False,
            "xposed": bool(not emulator and hooks_visible),
            "magisk": bool(not emulator and hooks_visible),
        },
        "attestation": {
            "present": True,
            "verdict": str(
                rng.choice(["MEETS_BASIC_INTEGRITY", "NO_INTEGRITY", "MEETS_DEVICE_INTEGRITY"], p=[0.5, 0.3, 0.2])
            ),
        },
    }


# ---------------------------------------------------------------------------
# Session generators
# ---------------------------------------------------------------------------
def _illumination_params(rng):
    return {"k": rng.uniform(0.03, 0.10), "awb": rng.uniform(0.0, 0.6), "lag": int(rng.integers(1, 3))}


def bona_fide(face, lmk, ch: Challenge, rng) -> CaptureSession:
    n, colors = ch.n_frames, ch.per_frame()
    mean_c = colors.mean(axis=0)
    p = _illumination_params(rng)
    isp = _isp_profile(rng)
    R = _reflectance_map(face.shape[:2], lmk, rng.uniform(0.0, 0.3))
    base = to_float(face)
    frames, state = [], None
    for t in range(n):
        lit = _lit(base, R, p["k"], colors[max(0, t - p["lag"])], mean_c, p["awb"])
        moved, state = small_motion(lit, rng, state)
        frames.append(_camera(moved, rng, isp))
    return CaptureSession(
        frames, _natural_timestamps(n, ch.fps, rng), _genuine_telemetry(rng), "bona_fide", ch, lmk, {**p, **isp}
    )


def _presentation(artefact_fn, face, lmk, donor, donor_lmk, ch, rng, uniform_reflect):
    n, colors = ch.n_frames, ch.per_frame()
    mean_c = colors.mean(axis=0)
    p = _illumination_params(rng)
    isp = _isp_profile(rng)
    seed = int(rng.integers(1 << 31))
    R = np.full(face.shape[:2], uniform_reflect, np.float32)  # flat artefact: same reflection everywhere
    frames, state = [], None
    for t in range(n):
        if artefact_fn is screen_replay:  # replayed *video*: victim moves on the screen
            content, state = small_motion(donor, rng, state, 0.8)
            art = screen_replay(content, np.random.default_rng(seed))
        else:
            if t == 0:
                art0 = print_photo(donor, np.random.default_rng(seed))
            art, state = small_motion(art0, rng, state, 1.3)  # whole sheet moves rigidly
        lit = _lit(art, R, p["k"], colors[max(0, t - p["lag"])], mean_c, p["awb"])
        frames.append(_camera(lit, rng, isp))
    return frames, _natural_timestamps(n, ch.fps, rng), _genuine_telemetry(rng), {**p, **isp}


def simulate(species: str, face, lmk, donor, donor_lmk, ch: Challenge, rng: np.random.Generator) -> CaptureSession:
    """Generate one session. ``face``/``lmk`` is the person in front of the
    camera (bona fide user or attacker); ``donor`` is the victim being
    impersonated (attack instruments are made from the victim's media)."""
    n, colors, fps = ch.n_frames, ch.per_frame(), ch.fps
    mean_c = colors.mean(axis=0)

    if species == "bona_fide":
        return bona_fide(face, lmk, ch, rng)

    if species == "print_photo":
        fr, ts, tel, meta = _presentation(print_photo, face, lmk, donor, donor_lmk, ch, rng, rng.uniform(0.6, 1.0))
        return CaptureSession(fr, ts, tel, species, ch, donor_lmk, meta)

    if species == "screen_replay":
        fr, ts, tel, meta = _presentation(screen_replay, face, lmk, donor, donor_lmk, ch, rng, rng.uniform(0.1, 0.4))
        return CaptureSession(fr, ts, tel, species, ch, donor_lmk, meta)

    if species in ("vcam_prerecorded", "vcam_prerecorded_stealth", "emulator_injection", "challenge_aware_injection"):
        stealth = species in ("vcam_prerecorded_stealth", "challenge_aware_injection")
        # the victim's earlier genuine recording: their camera's noise is baked in
        rec_isp = {"read": rng.uniform(0.5, 1.5), "shot": 0.1, "denoise": 0.0, "q": 90}
        base = to_float(donor)
        meta: dict = {}
        if species == "challenge_aware_injection":
            # attacker screen-captures the challenge and re-lights the face mask
            lag = int(rng.integers(2, 6))
            k = rng.uniform(0.03, 0.10)
            R = _reflectance_map(donor.shape[:2], donor_lmk, rng.uniform(0.0, 0.3))
            meta = {"lag": lag, "k": k}
        frames_f, state = [], None
        for t in range(n):
            x = base
            if species == "challenge_aware_injection":
                x = _lit(base, R, k, colors[max(0, t - lag)], mean_c, 0.0)
            moved, state = small_motion(x, rng, state)
            frames_f.append(to_float(_camera(moved, rng, rec_isp)))
        frames = _codec(frames_f, rng)
        if stealth:  # add sensor-like noise to defeat noise-floor checks
            frames = [
                to_u8(sensor_noise(to_float(f), rng, rng.uniform(1.5, 4.0), rng.uniform(0.3, 1.0))) for f in frames
            ]
            ts = _natural_timestamps(n, fps, rng)
            tel = _vcam_telemetry(rng, stealth=True)
        else:
            ts, dups = _synthetic_timestamps(n, fps, rng)
            for i in dups:
                frames[i] = frames[i - 1].copy()
            tel = _emulator_telemetry(rng) if species == "emulator_injection" else _vcam_telemetry(rng, stealth=False)
        return CaptureSession(_browser_encode(frames, rng), ts, tel, species, ch, donor_lmk, meta)

    if species == "vcam_offline_faceswap":
        # the attacker's driving video was shot with a real camera (noise baked in);
        # the victim's face is generated, resampled and blended on top of it
        cam_frame = to_float(_camera(to_float(face), rng, _isp_profile(rng)))
        swapped = face_swap(face, lmk, donor, donor_lmk, rng, target_f=cam_frame)
        frames_f, state = [], None
        for _ in range(n):
            moved, state = small_motion(swapped, rng, state)
            frames_f.append(moved)
        frames = _codec(frames_f, rng, downscale=False)
        ts, dups = _synthetic_timestamps(n, fps, rng)
        for i in dups:
            frames[i] = frames[i - 1].copy()
        return CaptureSession(_browser_encode(frames, rng), ts, _vcam_telemetry(rng, stealth=False), species, ch, lmk)

    if species == "vcam_realtime_faceswap":
        # attacker sits in front of a real camera (reflects the challenge) ...
        p = _illumination_params(rng)
        isp = _isp_profile(rng)
        R = _reflectance_map(face.shape[:2], lmk, rng.uniform(0.0, 0.3))
        base = to_float(face)
        cam = [
            to_float(_camera(_lit(base, R, p["k"], colors[max(0, t - p["lag"])], mean_c, p["awb"]), rng, isp))
            for t in range(n)
        ]
        # ... and a swap model re-renders every frame with processing latency
        latency = int(rng.integers(4, 11))
        alpha = rng.uniform(0.4, 1.0)
        swap_seed = int(rng.integers(1 << 31))
        out, state = [], None
        for t in range(n):
            src_frame = cam[max(0, t - latency)]
            # same swap parameters every frame (a stable model); colour statistics from the lit frame
            # (carries the reflected challenge) or from the unlit face (imperfect transfer) -> alpha mix
            sw_lit = face_swap(face, lmk, donor, donor_lmk, np.random.default_rng(swap_seed), target_f=src_frame)
            sw_ref = face_swap(
                face, lmk, donor, donor_lmk, np.random.default_rng(swap_seed), target_f=src_frame, color_ref_f=base
            )
            sw = alpha * sw_lit + (1 - alpha) * sw_ref
            moved, state = small_motion(sw, rng, state)
            out.append(to_u8(moved))
        stealth = rng.random() < 0.5
        if stealth:
            ts = _natural_timestamps(n, fps, rng)
        else:
            ts, _ = _synthetic_timestamps(n, fps, rng)
        tel = _vcam_telemetry(rng, stealth=stealth)
        return CaptureSession(
            _browser_encode(out, rng), ts, tel, species, ch, lmk, {"latency": latency, "alpha": alpha, **p}
        )

    if species == "challenge_aware_faceswap":
        # the capture page runs on a second (virtual) display, so the attacker's own face never
        # sees the flashes; their camera feed is face-swapped and the challenge is re-lit
        # synthetically from a screen capture, with the pipeline's 2-5 frame lag
        isp = _isp_profile(rng)
        lag = int(rng.integers(2, 6))
        k = rng.uniform(0.03, 0.10)
        R = _reflectance_map(face.shape[:2], lmk, rng.uniform(0.0, 0.3))
        base = to_float(face)
        swap_seed = int(rng.integers(1 << 31))
        out, state = [], None
        for t in range(n):
            cam = to_float(_camera(base, rng, isp))
            sw = face_swap(face, lmk, donor, donor_lmk, np.random.default_rng(swap_seed), target_f=cam)
            relit = _lit(sw, R, k, colors[max(0, t - lag)], mean_c, 0.0)
            moved, state = small_motion(np.clip(relit, 0, 1), rng, state)
            out.append(to_u8(moved))
        return CaptureSession(
            _browser_encode(out, rng),
            _natural_timestamps(n, fps, rng),
            _vcam_telemetry(rng, stealth=True),
            species,
            ch,
            lmk,
            {"lag": lag, "k": k},
        )

    if species == "vcam_synthetic_gan":
        synth = gan_upsample(donor, rng)
        frames_f, state = [], None
        for _ in range(n):
            moved, state = small_motion(synth, rng, state)
            frames_f.append(moved)
        frames = _codec(frames_f, rng, downscale=False)
        stealth = rng.random() < 0.5
        if stealth:
            frames = [to_u8(sensor_noise(to_float(f), rng, rng.uniform(1.5, 4.0), 0.5)) for f in frames]
            ts = _natural_timestamps(n, fps, rng)
        else:
            ts, dups = _synthetic_timestamps(n, fps, rng)
            for i in dups:
                frames[i] = frames[i - 1].copy()
        return CaptureSession(_browser_encode(frames, rng), ts, _vcam_telemetry(rng, stealth), species, ch, donor_lmk)

    raise ValueError(f"unknown species {species}")


def live_capture_frame(face_u8: np.ndarray, lmk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One frame as a real camera delivers it: optional challenge tint, head
    motion, camera ISP (noise, denoise, JPEG)."""
    from src.liveness.challenge import PALETTE

    x = to_float(face_u8)
    if rng.random() < 0.7:
        R = _reflectance_map(x.shape[:2], lmk, rng.uniform(0.0, 0.3))
        c = PALETTE[int(rng.integers(0, len(PALETTE)))]
        x = _lit(x, R, rng.uniform(0.03, 0.10), c, PALETTE.mean(axis=0), rng.uniform(0.0, 0.6))
    x, _ = small_motion(np.clip(x, 0, 1), rng, None, strength=2.0)
    return _camera(x, rng, _isp_profile(rng))


def deliver(frame_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Transport to the server: browser JPEG, or a streaming/virtual-camera codec."""
    if rng.random() < 0.5:
        return jpeg(frame_u8, int(rng.integers(75, 96)))
    return _browser_encode(_codec([to_float(frame_u8)], rng, downscale=rng.random() < 0.3), rng)[0]


def bona_fide_frame(face_u8: np.ndarray, lmk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return deliver(live_capture_frame(face_u8, lmk, rng), rng)


def injected_self_blend(face_u8: np.ndarray, lmk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Self-Blended Image produced the way an *injected* manipulation is made:
    the target is a real camera frame (sensor noise, ISP, tint), the blended-in
    face is generated (clean, re-coloured, resampled) and the result is
    delivered without passing through a camera again. A detector trained on
    these learns the camera/generated inconsistency, not one generator."""
    from src.attacks.synthesis import _jitter_appearance

    tgt = to_float(live_capture_frame(face_u8, lmk, rng))
    h, w = tgt.shape[:2]
    src = _jitter_appearance(to_float(face_u8), rng)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), rng.uniform(0.96, 1.04))
    M[:, 2] += rng.uniform(-3, 3, 2)
    src = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    s = rng.uniform(0.45, 0.9)
    src = cv2.resize(cv2.resize(src, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
    mask = face_ellipse_mask((h, w), lmk, grow=rng.uniform(0.85, 1.1))
    mask = random_deform_mask((mask > 0.5).astype(np.float32), rng, strength=rng.uniform(2, 6))
    out = mask[..., None] * src + (1 - mask[..., None]) * tgt
    return deliver(to_u8(out), rng)
