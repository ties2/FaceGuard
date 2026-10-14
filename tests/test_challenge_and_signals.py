"""Nonce challenge + capture-integrity signals (rules and stream physics)."""

import numpy as np
import pytest

from src.attacks.capture import simulate
from src.liveness.challenge import PALETTE, challenge_from_nonce, random_challenge
from src.liveness.signals.capture_integrity import challenge_findings, explain, integrity_rules, stream_features
from src.liveness.signals.texture import PAD_FEATURE_DIM, pad_features
from tests.helpers import astronaut_face


# ---------------------------------------------------------------- challenge
def test_challenge_is_deterministic_per_nonce_and_secret():
    a = challenge_from_nonce(b"n" * 16, b"secret")
    assert a == challenge_from_nonce(b"n" * 16, b"secret")
    assert a != challenge_from_nonce(b"n" * 16, b"other-secret")


def test_challenge_has_no_immediate_repeats_and_valid_palette():
    for i in range(200):
        c = challenge_from_nonce(i.to_bytes(16, "big"), b"k")
        assert all(c.slots[k] != c.slots[k + 1] for k in range(len(c.slots) - 1))
        assert all(0 <= s < len(PALETTE) for s in c.slots)
        assert c.per_frame().shape == (c.n_frames, 3)
        assert all(h.startswith("#") and len(h) == 7 for h in c.to_hex())


def test_challenges_are_unpredictable():
    seqs = {challenge_from_nonce(i.to_bytes(16, "big"), b"k").slots for i in range(500)}
    assert len(seqs) == 500


# ---------------------------------------------------------------- rules
@pytest.mark.parametrize(
    "tel,code,severity",
    [
        ({"camera_label": "OBS Virtual Camera"}, "VIRTUAL_CAMERA", "hard"),
        ({"camera_label": "ManyCam Virtual Webcam"}, "VIRTUAL_CAMERA", "hard"),
        ({"camera_label": "NVIDIA Broadcast"}, "VIRTUAL_CAMERA_BENIGN", "soft"),
        ({"camera_label": "FaceTime HD Camera", "navigator_webdriver": True}, "BROWSER_AUTOMATION", "hard"),
        (
            {"platform": "android", "emulator_signals": {"qemu": True, "build_fingerprint": "x", "sensor_count": 30}},
            "EMULATOR",
            "hard",
        ),
        (
            {"platform": "android", "emulator_signals": {"build_fingerprint": "google/sdk_gphone64_x86_64/emu"}},
            "EMULATOR",
            "hard",
        ),
        ({"platform": "android", "hooking_signals": {"frida": True}}, "HOOKING_FRAMEWORK", "hard"),
        (
            {"platform": "android", "attestation": {"present": True, "verdict": "MEETS_BASIC_INTEGRITY"}},
            "ATTESTATION_FAILED",
            "hard",
        ),
        ({"platform": "ios", "attestation": {"present": False}}, "ATTESTATION_MISSING", "soft"),
        (
            {"track_settings": {"width": 1280, "height": 720}, "frame_size": [640, 480]},
            "RESOLUTION_MISMATCH",
            "soft",
        ),
    ],
)
def test_integrity_rules(tel, code, severity):
    found = {f.code: f.severity for f in integrity_rules(tel)}
    assert found.get(code) == severity


def test_clean_desktop_telemetry_has_no_findings():
    tel = {
        "platform": "web",
        "camera_label": "Logitech BRIO",
        "track_settings": {"width": 1280, "height": 720, "frameRate": 30},
        "frame_size": [1280, 720],
        "navigator_webdriver": False,
    }
    assert integrity_rules(tel) == []


# ---------------------------------------------------------------- stream physics
def _session(species, seed=0):
    face, lmk = astronaut_face()
    rng = np.random.default_rng(seed)
    ch = random_challenge(rng)
    s = simulate(species, face, lmk, face[:, ::-1].copy(), lmk.copy(), ch, rng)
    return s, stream_features(s.frames, s.timestamps_ms, s.lmk, ch.per_frame())


def test_bona_fide_reflects_the_challenge_with_camera_latency():
    _, f = _session("bona_fide")
    assert f["chal_corr_early"] > 0.6
    assert f["chal_best_lag"] <= 2
    assert f["ts_std"] > 0.5 and f["dup_ratio"] == 0


def test_prerecorded_injection_does_not_reflect_the_challenge():
    _, f = _session("vcam_prerecorded", seed=1)
    assert f["chal_corr_early"] < 0.35
    assert "CHALLENGE_NOT_REFLECTED" in {e.code for e in explain(f)}


def test_print_reflects_uniformly():
    _, f = _session("print_photo", seed=2)
    assert f["chal_corr_face_early"] > 0.5  # the paper reflects ...
    assert f["chal_corr_early"] < 0.35  # ... but face and background equally


def test_realtime_faceswap_reflection_is_delayed():
    _, f = _session("vcam_realtime_faceswap", seed=3)
    assert f["chal_best_lag"] >= 4
    assert f["chal_corr_late"] > f["chal_corr_early"]
    assert [x.code for x in challenge_findings(f)] == ["CHALLENGE_REFLECTION_DELAYED"]


def test_bona_fide_never_triggers_delayed_rule():
    for seed in range(5):
        _, f = _session("bona_fide", seed=100 + seed)
        assert challenge_findings(f) == []


def test_pad_feature_dimension():
    face, _ = astronaut_face()
    assert pad_features(face).shape == (PAD_FEATURE_DIM,)
