from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    user_ref: str = Field(..., min_length=1, max_length=128, description="Pseudonymous applicant / account id")
    device_id: str | None = Field(None, max_length=128, description="Hashed device fingerprint from the client SDK")


class ChallengeOut(BaseModel):
    colors: list[str]  # hex colours the client renders full-screen, in order
    slot_ms: float
    n_frames: int
    fps: float


class SessionOut(BaseModel):
    session_id: str
    session_token: str
    vendor_session_token: str | None
    challenge: ChallengeOut
    expires_at: datetime


class VendorPayload(BaseModel):
    face_scan: str | None = None
    audit_trail_image: str | None = None  # base64 JPEG
    low_quality_audit_trail_image: str | None = None
    user_agent: str | None = None


class VerifyRequest(BaseModel):
    session_token: str
    frames: list[str] = Field(..., description="base64 JPEG frames captured during the challenge")
    timestamps_ms: list[float]
    telemetry: dict = Field(default_factory=dict)
    vendor: VendorPayload = Field(default_factory=VendorPayload)
    reference_image: str | None = Field(None, description="base64 ID portrait / enrolment photo for 1:1 match")
    client_claims: dict | None = Field(None, description="Ignored by design — never trust client-side verdicts")


class LayerOut(BaseModel):
    name: str
    score: float | None
    band: str
    reasons: list[str] = []


class VerifyResponse(BaseModel):
    session_id: str
    decision: str
    risk_score: float
    reasons: list[str]
    layers: list[LayerOut]
    model_versions: dict
    policy_version: str
    latency_ms: float


class ReviewItem(BaseModel):
    decision_id: int
    session_id: str
    user_ref: str
    decision: str
    risk_score: float
    reasons: list
    layers: list
    created_at: datetime


class ReviewOutcome(BaseModel):
    outcome: str = Field(..., pattern="^(confirmed_attack|confirmed_genuine)$")
    reviewer: str = Field(..., min_length=1, max_length=64)
    note: str | None = Field(None, max_length=1000)
