"""Decision policy + FaceTec adapter semantics."""

import httpx
import pytest

from src.liveness.fusion import Band, Decision, Policy, decide
from src.liveness.providers.base import VendorRequest
from src.liveness.providers.facetec import FaceTecServerAdapter, FaceTecUnavailable
from src.liveness.providers.simulated import SimulatedVendorProvider
from src.liveness.signals.capture_integrity import Finding

POL = Policy(
    version="test",
    pad=Band(0.6, 0.8),
    capture_integrity=Band(0.5, 0.9),
    deepfake=Band(0.4, 0.95),
)


def test_all_clear_approves():
    v = decide(POL, vendor_ok=True, pad=0.1, capture_integrity=0.1, deepfake=0.1)
    assert v.decision == Decision.APPROVE
    assert v.reasons == []


def test_vendor_failure_rejects():
    v = decide(POL, vendor_ok=False, vendor_reasons=["replayCheckSucceeded"], capture_integrity=0.0)
    assert v.decision == Decision.REJECT
    assert "replayCheckSucceeded" in v.reasons[0]


def test_hard_finding_rejects_soft_finding_steps_up():
    assert decide(POL, vendor_ok=True, findings=[Finding("VIRTUAL_CAMERA", "hard", "")]).decision == Decision.REJECT
    assert (
        decide(POL, vendor_ok=True, findings=[Finding("RESOLUTION_MISMATCH", "soft", "")]).decision == Decision.STEP_UP
    )


def test_review_bands_route_to_the_right_queue():
    assert decide(POL, vendor_ok=True, capture_integrity=0.6).decision == Decision.STEP_UP
    assert decide(POL, vendor_ok=True, deepfake=0.5).decision == Decision.MANUAL_REVIEW
    assert decide(POL, vendor_ok=True, pad=0.7).decision == Decision.MANUAL_REVIEW


def test_most_severe_outcome_wins():
    v = decide(
        POL,
        vendor_ok=True,
        findings=[Finding("ATTESTATION_MISSING", "soft", "")],
        capture_integrity=0.6,
        deepfake=0.99,
    )
    assert v.decision == Decision.REJECT
    assert len(v.reasons) == 3


def test_face_mismatch_rejects():
    assert decide(POL, vendor_ok=True, face_match=0.1).decision == Decision.REJECT
    assert decide(POL, vendor_ok=True, face_match=0.7).decision == Decision.APPROVE


def test_policy_roundtrip(tmp_path):
    p = tmp_path / "p.yaml"
    POL.save(p)
    q = Policy.load(p)
    assert q.deepfake.review == 0.4 and q.version == "test"


# ---------------------------------------------------------------- FaceTec adapter
def _adapter(handler):
    return FaceTecServerAdapter("https://facetec.internal", "dev-key", transport=httpx.MockTransport(handler))


ALL_OK = {
    "success": True,
    "wasProcessed": True,
    "error": False,
    "scanResultBlob": "opaque-blob",
    "faceScanSecurityChecks": {
        "sessionTokenCheckSucceeded": True,
        "replayCheckSucceeded": True,
        "auditTrailVerificationCheckSucceeded": True,
        "faceScanLivenessCheckSucceeded": True,
    },
}
REQ = VendorRequest("s1", "tok", "facescan", "audit")


def test_facetec_all_checks_pass():
    seen = {}

    def h(req: httpx.Request):
        seen["key"] = req.headers.get("X-Device-Key")
        return httpx.Response(200, json=ALL_OK)

    r = _adapter(h).verify(REQ)
    assert r.ok and r.failed_checks == []
    assert seen["key"] == "dev-key"
    assert "scanResultBlob" not in r.raw  # never persisted


def test_facetec_success_flag_alone_is_not_enough():
    body = {**ALL_OK, "faceScanSecurityChecks": {**ALL_OK["faceScanSecurityChecks"], "replayCheckSucceeded": False}}
    r = _adapter(lambda req: httpx.Response(200, json=body)).verify(REQ)
    assert not r.ok
    assert r.failed_checks == ["replayCheckSucceeded"]


def test_facetec_missing_checks_fail_closed():
    body = {"success": True, "wasProcessed": True}
    r = _adapter(lambda req: httpx.Response(200, json=body)).verify(REQ)
    assert not r.ok and len(r.failed_checks) >= 3


def test_facetec_outage_fails_closed():
    with pytest.raises(FaceTecUnavailable):
        _adapter(lambda req: httpx.Response(503)).verify(REQ)


def test_simulated_vendor_enforces_one_time_token_and_replay():
    v = SimulatedVendorProvider(pad_scorer=lambda _: 0.1, pass_below=0.5)
    tok = v.new_session_token()
    first = v.verify(VendorRequest("a", tok, "scan-1", "img"))
    assert first.ok
    again = v.verify(VendorRequest("b", tok, "scan-1", "img"))
    assert not again.security_checks["sessionTokenCheckSucceeded"]
    assert not again.security_checks["replayCheckSucceeded"]
