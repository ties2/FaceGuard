"""Liveness sessions and decisions (audit trail + manual-review queue)."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String

from src.serving.database import Base


def _now():
    return datetime.now(UTC).replace(tzinfo=None)


class DBLivenessSession(Base):
    __tablename__ = "liveness_sessions"

    id = Column(String, primary_key=True)
    user_ref = Column(String, index=True, nullable=False)  # pseudonymous applicant id
    device_id = Column(String, index=True, nullable=True)  # hashed device fingerprint
    nonce = Column(String, nullable=False)
    challenge = Column(JSON, nullable=False)
    vendor_token = Column(String, nullable=True)
    status = Column(String, nullable=False, default="issued")  # issued | consumed
    created_at = Column(DateTime, default=_now, index=True)
    expires_at = Column(DateTime, nullable=False)
    consumed_at = Column(DateTime, nullable=True)


class DBLivenessDecision(Base):
    __tablename__ = "liveness_decisions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("liveness_sessions.id"), unique=True, nullable=False)
    user_ref = Column(String, index=True, nullable=False)
    device_id = Column(String, index=True, nullable=True)
    platform = Column(String, nullable=True)
    decision = Column(String, nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    layers = Column(JSON, nullable=False)
    reasons = Column(JSON, nullable=False)
    model_versions = Column(JSON, nullable=False)
    policy_version = Column(String, nullable=False)
    review_status = Column(String, nullable=True, index=True)  # pending | confirmed_attack | confirmed_genuine
    reviewer = Column(String, nullable=True)
    review_note = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now, index=True)
