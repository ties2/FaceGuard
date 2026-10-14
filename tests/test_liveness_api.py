"""End-to-end API tests with stand-in detector models (no dataset needed).

Covers the logic attacks that need no ML at all — session replay, token
forgery, client-side verdict tampering, probing/velocity, device farms — plus
the review queue and MSP-driven model promotion.
"""

import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.attacks.capture import simulate
from src.liveness import api as liveness_api
from src.liveness.challenge import PALETTE, Challenge
from src.liveness.fusion import Band, Policy
from src.liveness.providers.simulated import SimulatedVendorProvider
from src.serving.app import app
from src.serving.cache import model_cache
from src.serving.dependency import get_artifact_storage
from tests.helpers import (
    astronaut_face,
    b64jpeg,
    stand_in_ci_bytes,
    stand_in_forensics_bytes,
    stand_in_pad_bytes,
    tiny_image_onnx,
)

POLICY = Policy(
    version="test-policy",
    pad=Band(0.999, 0.9999),
    capture_integrity=Band(0.5, 0.9),
    deepfake=Band(0.5, 0.9),
    max_attempts_per_identity_24h=3,
    max_attempts_per_device_24h=50,
    max_identities_per_device_30d=2,
    calibration={"vendor_pad_threshold": 0.999},
)
VENDOR = SimulatedVendorProvider(pad_scorer=lambda _: 0.0, pass_below=0.5)


def _register(client, name, version, data, framework, fmt):
    uri = f"local://{name}/{version}/artifact"
    get_artifact_storage().save(uri, data)
    client.post("/models", json={"name": name})
    client.post(
        f"/models/{name}/versions",
        json={"version": version, "framework": framework, "model_format": fmt, "artifact_uri": uri},
    )
    r = client.post(f"/models/{name}/versions/{version}/promote")
    assert r.status_code == 200, r.text


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[liveness_api.get_policy] = lambda: POLICY
    app.dependency_overrides[liveness_api.get_vendor] = lambda: VENDOR
    model_cache.clear()
    with TestClient(app) as c:
        _register(c, "faceguard-pad-texture", "1.0.0", stand_in_pad_bytes(), "scikit-learn", "joblib")
        _register(c, "faceguard-capture-integrity", "1.0.0", stand_in_ci_bytes(), "scikit-learn", "joblib")
        _register(c, "faceguard-deepfake", "1.0.0", tiny_image_onnx(bias=-3.0), "pytorch", "onnx")
        _register(c, "faceguard-deepfake-forensics", "1.0.0", stand_in_forensics_bytes(), "scikit-learn", "joblib")
        yield c
    app.dependency_overrides.clear()


def _challenge(out: dict) -> Challenge:
    lut = {}
    for i, (b, g, r) in enumerate((PALETTE * 255).astype(int)):
        lut[f"#{r:02x}{g:02x}{b:02x}"] = i
    ch = out["challenge"]
    slots = tuple(lut[c] for c in ch["colors"])
    return Challenge(slots, slot_frames=ch["n_frames"] // len(slots), fps=ch["fps"])


def _new_session(client, user=None, device=None):
    r = client.post("/v1/liveness/sessions", json={"user_ref": user or uuid.uuid4().hex, "device_id": device})
    return r


def _payload(out, species="bona_fide", seed=0, label=None, claims=None):
    face, lmk = astronaut_face()
    rng = np.random.default_rng(seed)
    s = simulate(species, face, lmk, face[:, ::-1].copy(), lmk.copy(), _challenge(out), rng)
    tel = dict(s.telemetry)
    if label:
        tel["camera_label"] = label
    return {
        "session_token": out["session_token"],
        "frames": [b64jpeg(f) for f in s.frames],
        "timestamps_ms": [float(t) for t in s.timestamps_ms],
        "telemetry": tel,
        "vendor": {"face_scan": f"scan-{uuid.uuid4()}", "audit_trail_image": b64jpeg(s.audit_frame)},
        "client_claims": claims,
    }


def test_session_issues_nonce_challenge(client):
    r = _new_session(client)
    assert r.status_code == 200
    out = r.json()
    assert len(out["challenge"]["colors"]) == 10 and out["challenge"]["n_frames"] == 30
    assert out["vendor_session_token"]


def test_bona_fide_is_approved(client):
    out = _new_session(client).json()
    r = client.post(
        f"/v1/liveness/sessions/{out['session_id']}/verify", json=_payload(out, seed=1, label="Logitech BRIO")
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "APPROVE", body["reasons"]
    assert {layer["name"] for layer in body["layers"]} >= {"vendor_liveness", "capture_integrity", "deepfake"}
    assert body["model_versions"]["faceguard-deepfake"] == "1.0.0"


def test_session_replay_is_refused(client):
    out = _new_session(client).json()
    payload = _payload(out, seed=2, label="Logitech BRIO")
    assert client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=payload).status_code == 200
    r = client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=payload)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "SESSION_REPLAY"


def test_forged_session_token_is_refused(client):
    out = _new_session(client).json()
    payload = _payload(out, seed=3)
    payload["session_token"] = "forged"
    assert client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=payload).status_code == 401


def test_virtual_camera_injection_rejected_even_with_client_claims(client):
    out = _new_session(client).json()
    payload = _payload(out, "vcam_prerecorded", seed=4, label="OBS Virtual Camera", claims={"livenessPassed": True})
    body = client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=payload).json()
    assert body["decision"] == "REJECT"
    assert any("VIRTUAL_CAMERA" in r for r in body["reasons"])


def test_stealth_injection_caught_by_challenge(client):
    out = _new_session(client).json()
    payload = _payload(out, "vcam_prerecorded_stealth", seed=5, label="Logitech BRIO")
    body = client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=payload).json()
    assert body["decision"] in ("REJECT", "STEP_UP")
    ci = next(layer for layer in body["layers"] if layer["name"] == "capture_integrity")
    assert "CHALLENGE_NOT_REFLECTED" in ci["reasons"]


def test_wrong_frame_count_is_422(client):
    out = _new_session(client).json()
    payload = _payload(out, seed=6)
    payload["frames"], payload["timestamps_ms"] = payload["frames"][:5], payload["timestamps_ms"][:5]
    assert client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=payload).status_code == 422


def test_identity_velocity_limit(client):
    user = uuid.uuid4().hex
    for _ in range(3):
        assert _new_session(client, user).status_code == 200
    r = _new_session(client, user)
    assert r.status_code == 429 and r.json()["detail"]["code"] == "VELOCITY_IDENTITY"


def test_device_used_by_many_identities_is_blocked(client):
    dev = uuid.uuid4().hex
    assert _new_session(client, "u-a-" + dev, dev).status_code == 200
    assert _new_session(client, "u-b-" + dev, dev).status_code == 200
    r = _new_session(client, "u-c-" + dev, dev)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "DEVICE_MULTI_IDENTITY"


def test_manual_review_queue_flow(client):
    review_policy = Policy(**{**POLICY.__dict__, "deepfake": Band(0.0, 0.9999)})
    app.dependency_overrides[liveness_api.get_policy] = lambda: review_policy
    try:
        out = _new_session(client).json()
        body = client.post(
            f"/v1/liveness/sessions/{out['session_id']}/verify", json=_payload(out, seed=7, label="Logitech BRIO")
        ).json()
    finally:
        app.dependency_overrides[liveness_api.get_policy] = lambda: POLICY
    assert body["decision"] == "MANUAL_REVIEW"
    queue = client.get("/v1/liveness/reviews").json()
    item = next(q for q in queue if q["session_id"] == out["session_id"])
    r = client.post(
        f"/v1/liveness/reviews/{item['decision_id']}", json={"outcome": "confirmed_genuine", "reviewer": "analyst-1"}
    )
    assert r.status_code == 200
    again = client.post(
        f"/v1/liveness/reviews/{item['decision_id']}", json={"outcome": "confirmed_attack", "reviewer": "x"}
    )
    assert again.status_code == 409


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    assert "faceguard_decisions_total" in text
    assert "faceguard_layer_score_bucket" in text


def test_promoting_a_detector_version_takes_effect_immediately(client):
    """MSP lifecycle drives the defence layer: promote -> next request uses it."""
    _register(client, "faceguard-deepfake", "1.1.0", tiny_image_onnx(bias=+6.0), "pytorch", "onnx")
    out = _new_session(client).json()
    body = client.post(
        f"/v1/liveness/sessions/{out['session_id']}/verify", json=_payload(out, seed=8, label="Logitech BRIO")
    ).json()
    assert body["model_versions"]["faceguard-deepfake"] == "1.1.0"
    df = next(layer for layer in body["layers"] if layer["name"] == "deepfake")
    assert any(r.startswith("cnn=0.99") or r.startswith("cnn=1.00") for r in df["reasons"])
    assert body["decision"] in ("MANUAL_REVIEW", "REJECT")
