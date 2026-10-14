"""Integration test of the *trained* system through the public API.

Runs only when the trained detectors exist (``make train``) — CI without the
dataset skips it. Uses the calibrated ``configs/policy.yaml``.
"""

import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.attacks.capture import simulate
from src.liveness import api as liveness_api
from src.liveness.fusion import Policy
from src.serving.app import app
from src.serving.cache import model_cache
from src.training.register import MODELS, register_all
from tests.helpers import astronaut_face, b64jpeg
from tests.test_liveness_api import _challenge

ART = Path("artifacts/models")
POLICY = Path("configs/policy.yaml")
pytestmark = [
    pytest.mark.trained,
    pytest.mark.skipif(
        not (POLICY.exists() and all((ART / f).exists() for _, f, *_ in MODELS)),
        reason="trained artifacts not present (run `make train eval`)",
    ),
]


@pytest.fixture(scope="module")
def client():
    pol = Policy.load(POLICY)
    app.dependency_overrides[liveness_api.get_policy] = lambda: pol
    model_cache.clear()
    with TestClient(app) as c:
        register_all(c, f"trained-{uuid.uuid4().hex[:6]}")
        yield c
    app.dependency_overrides.clear()
    model_cache.clear()


def _run(client, species, seed, label=None, device=None):
    out = client.post("/v1/liveness/sessions", json={"user_ref": uuid.uuid4().hex, "device_id": device}).json()
    face, lmk = astronaut_face()
    rng = np.random.default_rng(seed)
    s = simulate(species, face, lmk, face[:, ::-1].copy(), lmk.copy(), _challenge(out), rng)
    tel = dict(s.telemetry)
    if label:
        tel["camera_label"] = label
    body = {
        "session_token": out["session_token"],
        "frames": [b64jpeg(f) for f in s.frames],
        "timestamps_ms": [float(t) for t in s.timestamps_ms],
        "telemetry": tel,
        "vendor": {"face_scan": f"scan-{uuid.uuid4()}", "audit_trail_image": b64jpeg(s.audit_frame)},
    }
    r = client.post(f"/v1/liveness/sessions/{out['session_id']}/verify", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_bona_fide_mostly_passes(client):
    decisions = [_run(client, "bona_fide", seed, label="Logitech BRIO")["decision"] for seed in range(4)]
    assert decisions.count("APPROVE") >= 3, decisions


@pytest.mark.parametrize(
    "species", ["vcam_prerecorded_stealth", "vcam_realtime_faceswap", "vcam_offline_faceswap", "screen_replay"]
)
def test_attacks_are_not_auto_approved(client, species):
    r = _run(client, species, seed=11, label="Logitech BRIO")
    assert r["decision"] != "APPROVE", r
