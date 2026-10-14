"""Liveness API (FG-004 .. FG-007).

POST /v1/liveness/sessions                 issue challenge + one-time token
POST /v1/liveness/sessions/{id}/verify     analyse capture -> decision
GET  /v1/liveness/reviews                  manual-review queue
POST /v1/liveness/reviews/{decision_id}    reviewer outcome (feeds retraining)
GET  /v1/liveness/policy                   active thresholds
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.liveness import metrics
from src.liveness.db_models import DBLivenessDecision
from src.liveness.face import FaceDetector, context_crop
from src.liveness.fusion import Decision, Policy, decide
from src.liveness.pipeline import MODEL_PAD, LivenessPipeline, NoFaceError
from src.liveness.providers.base import LivenessProvider, VendorRequest
from src.liveness.providers.facetec import FaceTecServerAdapter, FaceTecUnavailable
from src.liveness.providers.simulated import SimulatedVendorProvider
from src.liveness.schemas import (
    ChallengeOut,
    ReviewItem,
    ReviewOutcome,
    SessionCreate,
    SessionOut,
    VerifyRequest,
    VerifyResponse,
)
from src.liveness.sessions import SessionError, SessionService
from src.liveness.signals.capture_integrity import Finding
from src.liveness.signals.matching import FaceMatcher
from src.liveness.signals.texture import pad_features
from src.serving.config import settings
from src.serving.database import SessionLocal, get_db
from src.serving.service import ModelRuntime

router = APIRouter(prefix="/v1/liveness", tags=["Liveness defence"])

MAX_FRAMES = 90
MAX_FRAME_BYTES = 2_000_000


# ---------------------------------------------------------------------------
# Dependencies (overridable in tests)
# ---------------------------------------------------------------------------
@lru_cache
def get_policy() -> Policy:
    return Policy.load(settings.faceguard_policy_path)


@lru_cache
def get_detector() -> FaceDetector:
    return FaceDetector(settings.faceguard_yunet_path)


@lru_cache
def get_pipeline() -> LivenessPipeline:
    try:
        matcher = FaceMatcher(settings.faceguard_sface_path)
    except Exception:  # matcher is optional
        matcher = None
    return LivenessPipeline(get_detector(), matcher)


def get_sessions(policy: Policy = Depends(get_policy)) -> SessionService:
    return SessionService(settings.faceguard_session_secret, policy)


def _decode_image(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64.split(",")[-1], validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(422, "invalid base64 image") from e
    if len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(413, "frame too large")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "undecodable image")
    return img


def _sim_pad_scorer(audit_b64: str) -> float:
    """Simulated vendor: PAD score of the audit-trail image (stand-in for a
    presentation-attack-only liveness product)."""
    img = _decode_image(audit_b64)
    box = get_detector().detect(img)
    if box is None:
        return 1.0
    crop, _ = context_crop(img, box)
    db = SessionLocal()
    try:
        _, model = ModelRuntime.get_production_model(db, MODEL_PAD)
    finally:
        db.close()
    return float(model.predict_proba(pad_features(crop).reshape(1, -1))[0, 1])


@lru_cache
def get_vendor() -> LivenessProvider:
    if settings.faceguard_vendor == "facetec":
        if not (settings.facetec_server_url and settings.facetec_device_key):
            raise RuntimeError("FACETEC_SERVER_URL and FACETEC_DEVICE_KEY are required for the FaceTec adapter")
        return FaceTecServerAdapter(settings.facetec_server_url, settings.facetec_device_key)
    pol = get_policy()
    return SimulatedVendorProvider(
        _sim_pad_scorer, pass_below=pol.calibration.get("vendor_pad_threshold", pol.pad.reject)
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=SessionOut)
def create_session(
    body: SessionCreate,
    db: Session = Depends(get_db),
    svc: SessionService = Depends(get_sessions),
    vendor: LivenessProvider = Depends(get_vendor),
):
    try:
        vendor_token = vendor.new_session_token()
    except FaceTecUnavailable as e:
        metrics.VENDOR_ERRORS.inc()
        raise HTTPException(503, "liveness vendor unavailable, retry later") from e
    try:
        s = svc.issue(db, body.user_ref, body.device_id, vendor_token)
    except SessionError as e:
        metrics.SESSION_REJECTED.labels(code=e.code).inc()
        raise HTTPException(e.http_status, {"code": e.code, "message": str(e)}) from e
    ch = s.challenge
    return SessionOut(
        session_id=s.session_id,
        session_token=s.session_token,
        vendor_session_token=s.vendor_token,
        challenge=ChallengeOut(colors=ch.to_hex(), slot_ms=ch.slot_ms, n_frames=ch.n_frames, fps=ch.fps),
        expires_at=s.expires_at,
    )


@router.post("/sessions/{session_id}/verify", response_model=VerifyResponse)
def verify_session(
    session_id: str,
    body: VerifyRequest,
    db: Session = Depends(get_db),
    svc: SessionService = Depends(get_sessions),
    vendor: LivenessProvider = Depends(get_vendor),
    pipeline: LivenessPipeline = Depends(get_pipeline),
    policy: Policy = Depends(get_policy),
):
    try:
        sess, challenge = svc.consume(db, session_id, body.session_token)
    except SessionError as e:
        metrics.SESSION_REJECTED.labels(code=e.code).inc()
        raise HTTPException(e.http_status, {"code": e.code, "message": str(e)}) from e

    if len(body.frames) != challenge.n_frames or len(body.timestamps_ms) != len(body.frames):
        metrics.SESSION_REJECTED.labels(code="FRAME_COUNT").inc()
        raise HTTPException(422, f"expected {challenge.n_frames} frames with timestamps")
    if len(body.frames) > MAX_FRAMES:
        raise HTTPException(413, "too many frames")
    frames = [_decode_image(f) for f in body.frames]

    # 1) vendor verdict, server-to-server. Client-side claims are ignored.
    extra: list[Finding] = []
    vendor_ok, vendor_reasons = None, []
    if body.vendor.face_scan or body.vendor.audit_trail_image:
        try:
            vr = vendor.verify(
                VendorRequest(
                    session_id=session_id,
                    vendor_session_token=sess.vendor_token,
                    face_scan=body.vendor.face_scan,
                    audit_trail_image=body.vendor.audit_trail_image,
                    low_quality_audit_trail_image=body.vendor.low_quality_audit_trail_image,
                    user_agent=body.vendor.user_agent,
                )
            )
            vendor_ok, vendor_reasons = vr.ok, vr.failed_checks
            for c in vr.failed_checks:
                metrics.VENDOR_CHECK_FAILED.labels(check=c).inc()
        except FaceTecUnavailable:
            metrics.VENDOR_ERRORS.inc()
            extra.append(Finding("VENDOR_UNAVAILABLE", "soft", "vendor unreachable - failing closed to step-up"))
    else:
        extra.append(Finding("VENDOR_PAYLOAD_MISSING", "soft", "no vendor FaceScan / audit trail supplied"))

    # 2) FaceGuard layers
    reference = _decode_image(body.reference_image) if body.reference_image else None
    try:
        with metrics.PIPELINE_LATENCY.time():
            a = pipeline.analyse(db, frames, body.timestamps_ms, body.telemetry, challenge, reference)
    except NoFaceError as e:
        raise HTTPException(422, str(e)) from e

    # 3) fusion
    verdict = decide(
        policy,
        vendor_ok=vendor_ok,
        vendor_reasons=vendor_reasons,
        findings=a.findings + extra,
        pad=a.pad,
        capture_integrity=a.capture_integrity,
        deepfake=a.deepfake,
        face_match=a.face_match,
        explanations=a.explanations,
        deepfake_detail=a.deepfake_parts,
    )
    platform = body.telemetry.get("platform")
    metrics.record_verdict(verdict, platform, a.findings + extra, a.explanations)

    layers = [asdict(layer) for layer in verdict.layers]
    rec = DBLivenessDecision(
        session_id=session_id,
        user_ref=sess.user_ref,
        device_id=sess.device_id,
        platform=platform,
        decision=verdict.decision.name,
        risk_score=verdict.risk_score,
        layers=layers,
        reasons=verdict.reasons,
        model_versions=a.model_versions,
        policy_version=verdict.policy_version,
        review_status="pending" if verdict.decision == Decision.MANUAL_REVIEW else None,
    )
    db.add(rec)
    db.commit()
    return VerifyResponse(
        session_id=session_id,
        decision=verdict.decision.name,
        risk_score=verdict.risk_score,
        reasons=verdict.reasons,
        layers=layers,
        model_versions=a.model_versions,
        policy_version=verdict.policy_version,
        latency_ms=a.latency_s * 1000,
    )


@router.get("/reviews", response_model=list[ReviewItem])
def review_queue(status: str = "pending", limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(DBLivenessDecision)
        .filter(DBLivenessDecision.review_status == status)
        .order_by(DBLivenessDecision.risk_score.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [
        ReviewItem(
            decision_id=r.id,
            session_id=r.session_id,
            user_ref=r.user_ref,
            decision=r.decision,
            risk_score=r.risk_score,
            reasons=r.reasons,
            layers=r.layers,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/reviews/{decision_id}")
def submit_review(decision_id: int, body: ReviewOutcome, db: Session = Depends(get_db)):
    r = db.get(DBLivenessDecision, decision_id)
    if r is None:
        raise HTTPException(404, "decision not found")
    if r.review_status != "pending":
        raise HTTPException(409, "decision is not pending review")
    r.review_status, r.reviewer, r.review_note = body.outcome, body.reviewer, body.note
    r.reviewed_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()
    metrics.REVIEW_OUTCOMES.labels(outcome=body.outcome).inc()
    return {"decision_id": decision_id, "review_status": r.review_status}


@router.get("/policy")
def active_policy(policy: Policy = Depends(get_policy)):
    return asdict(policy)
