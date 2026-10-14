"""FaceTec Server adapter (server-to-server).

The integration rules this adapter enforces are exactly the weak points found
most often in FaceTec deployments (see docs/01-vulnerability-assessment.md):

1. **Never trust the client.** The browser/app only forwards the encrypted
   FaceScan and audit-trail image; the liveness verdict is obtained by *this*
   server calling FaceTec Server. A ``{"livenessPassed": true}`` from the client
   is ignored by design.
2. **Check every security check, not just ``success``.** FaceTec Server
   reports separate checks (session token, replay, audit-trail verification,
   liveness). Partial integrations that read only ``success``/``wasProcessed``
   lose that granularity — and the reason codes needed for monitoring.
3. **Fail closed.** Timeouts / 5xx never become an approval.
4. **Keep the audit-trail image.** It is the frame FaceGuard's own deepfake
   and texture layers analyse, and the evidence a manual reviewer sees.

Field names follow the FaceTec Server REST responses used in FaceTec's public
sample apps (``/session-token``, ``/liveness-3d``, ``faceScanSecurityChecks``).
They are configurable because they differ between server versions — verify
them against the FaceTec Server build you operate before go-live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from src.liveness.providers.base import VendorRequest, VendorResult


@dataclass
class FaceTecFieldMap:
    session_token_path: str = "/session-token"
    liveness_path: str = "/liveness-3d"
    token_field: str = "sessionToken"
    processed_field: str = "wasProcessed"
    success_field: str = "success"
    error_field: str = "error"
    checks_field: str = "faceScanSecurityChecks"
    required_checks: tuple[str, ...] = (
        "sessionTokenCheckSucceeded",
        "replayCheckSucceeded",
        "auditTrailVerificationCheckSucceeded",
        "faceScanLivenessCheckSucceeded",
    )
    extra_headers: dict = field(default_factory=dict)


class FaceTecUnavailable(Exception):
    pass


class FaceTecServerAdapter:
    name = "facetec"

    def __init__(
        self,
        base_url: str,
        device_key: str,
        fields: FaceTecFieldMap | None = None,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.fields = fields or FaceTecFieldMap()
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            transport=transport,
            headers={"X-Device-Key": device_key, **self.fields.extra_headers},
        )

    def new_session_token(self) -> str | None:
        try:
            r = self._client.get(self.fields.session_token_path)
            r.raise_for_status()
            return r.json().get(self.fields.token_field)
        except (httpx.HTTPError, ValueError) as e:
            raise FaceTecUnavailable(f"session-token request failed: {e}") from e

    def verify(self, req: VendorRequest) -> VendorResult:
        f = self.fields
        payload = {
            "faceScan": req.face_scan,
            "auditTrailImage": req.audit_trail_image,
            "lowQualityAuditTrailImage": req.low_quality_audit_trail_image,
            "sessionId": req.session_id,
        }
        if req.external_ref:
            payload["externalDatabaseRefID"] = req.external_ref
        headers = {"X-User-Agent": req.user_agent} if req.user_agent else {}
        try:
            r = self._client.post(f.liveness_path, json=payload, headers=headers)
            r.raise_for_status()
            body = r.json()
        except (httpx.HTTPError, ValueError) as e:
            # fail closed: an unreachable vendor is never a pass
            raise FaceTecUnavailable(f"liveness request failed: {e}") from e

        checks_raw = body.get(f.checks_field) or {}
        checks = {name: bool(checks_raw.get(name, False)) for name in f.required_checks}
        processed = bool(body.get(f.processed_field, False)) and not bool(body.get(f.error_field, False))
        liveness = bool(body.get(f.success_field, False)) and checks.get("faceScanLivenessCheckSucceeded", True)
        # never keep the encrypted blob in our logs/DB; keep only what we need
        redacted = {k: v for k, v in body.items() if k not in ("scanResultBlob",)}
        return VendorResult("facetec", processed, liveness, checks, body.get("matchLevel"), redacted)
