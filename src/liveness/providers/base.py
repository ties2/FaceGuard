"""Vendor liveness provider abstraction.

FaceGuard is a defence layer *around* a liveness vendor, not a replacement.
The pipeline talks to the vendor through this Protocol so the same fusion and
monitoring work with FaceTec, another SDK, or the offline simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class VendorRequest:
    session_id: str
    vendor_session_token: str | None
    face_scan: str | None  # opaque, encrypted vendor payload (FaceTec: FaceScan)
    audit_trail_image: str | None  # base64 JPEG (FaceTec: auditTrail[0])
    low_quality_audit_trail_image: str | None = None
    user_agent: str | None = None
    external_ref: str | None = None


@dataclass
class VendorResult:
    provider: str
    processed: bool
    liveness_passed: bool
    security_checks: dict[str, bool] = field(default_factory=dict)
    match_level: int | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.processed and self.liveness_passed and all(self.security_checks.values())

    @property
    def failed_checks(self) -> list[str]:
        out = [k for k, v in self.security_checks.items() if not v]
        if not self.processed:
            out.append("not_processed")
        if not self.liveness_passed:
            out.append("liveness_failed")
        return out


class LivenessProvider(Protocol):
    name: str

    def new_session_token(self) -> str | None: ...

    def verify(self, req: VendorRequest) -> VendorResult: ...
