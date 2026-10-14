"""Offline stand-in for a liveness vendor (demo, tests, harness).

Behaves like a typical *presentation-attack-only* liveness check: its
liveness verdict comes from a frame-level PAD model on the audit-trail image,
and it implements the same security-check semantics as FaceTec Server
(one-time session token, payload replay detection). It intentionally has no
notion of virtual cameras or deepfakes — that is the gap FaceGuard measures.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections.abc import Callable

from src.liveness.providers.base import VendorRequest, VendorResult


class SimulatedVendorProvider:
    name = "simulated"

    def __init__(self, pad_scorer: Callable[[str], float] | None = None, pass_below: float = 0.5):
        self._pad_scorer = pad_scorer
        self._pass_below = pass_below
        self._issued: set[str] = set()
        self._seen_payloads: set[str] = set()
        self._lock = threading.Lock()

    def new_session_token(self) -> str:
        tok = secrets.token_urlsafe(24)
        with self._lock:
            self._issued.add(tok)
        return tok

    def verify(self, req: VendorRequest) -> VendorResult:
        digest = hashlib.sha256(((req.face_scan or "") + (req.audit_trail_image or "")).encode()).hexdigest()
        with self._lock:
            token_ok = req.vendor_session_token in self._issued
            self._issued.discard(req.vendor_session_token)  # one-time use
            replay_ok = digest not in self._seen_payloads
            self._seen_payloads.add(digest)
        score = self._pad_scorer(req.audit_trail_image) if (self._pad_scorer and req.audit_trail_image) else 0.0
        live = score < self._pass_below
        checks = {
            "sessionTokenCheckSucceeded": token_ok,
            "replayCheckSucceeded": replay_ok,
            "auditTrailVerificationCheckSucceeded": req.audit_trail_image is not None,
            "faceScanLivenessCheckSucceeded": live,
        }
        return VendorResult("simulated", True, live, checks, None, {"pad_score": score})
