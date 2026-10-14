"""Server-issued liveness sessions: challenge, one-time token, velocity limits.

Session security properties (each maps to an attack in the threat model):

* **Nonce-bound challenge** — the colour sequence is derived server-side from a
  fresh nonce, so a recording made before the session cannot contain it.
* **One-time, signed session token** — HMAC(session_id, nonce). Consumption is
  an atomic ``UPDATE ... WHERE status='issued'``, so two concurrent submissions
  of the same captured payload cannot both be processed (API replay).
* **Short TTL** — limits the window for relaying a challenge to a deepfake rig.
* **Velocity limits** — attempts per identity / per device and identities per
  device. Bounds adaptive "probe until it passes" attacks and flags device farms.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from src.liveness.challenge import Challenge, challenge_from_nonce
from src.liveness.db_models import DBLivenessSession
from src.liveness.fusion import Policy


class SessionError(Exception):
    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code, self.http_status = code, http_status


@dataclass
class IssuedSession:
    session_id: str
    session_token: str
    challenge: Challenge
    vendor_token: str | None
    expires_at: datetime


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def sign(secret: str, session_id: str, nonce: str) -> str:
    mac = hmac.new(secret.encode(), f"{session_id}.{nonce}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


class SessionService:
    def __init__(self, secret: str, policy: Policy):
        self.secret = secret
        self.policy = policy

    # -- velocity -------------------------------------------------------------
    def check_velocity(self, db: Session, user_ref: str, device_id: str | None) -> None:
        now = _now()
        q = db.query(func.count(DBLivenessSession.id))
        n_user = q.filter(
            DBLivenessSession.user_ref == user_ref, DBLivenessSession.created_at > now - timedelta(hours=24)
        ).scalar()
        if n_user >= self.policy.max_attempts_per_identity_24h:
            raise SessionError("VELOCITY_IDENTITY", "too many attempts for this identity in 24h", 429)
        if device_id:
            n_dev = (
                db.query(func.count(DBLivenessSession.id))
                .filter(
                    DBLivenessSession.device_id == device_id, DBLivenessSession.created_at > now - timedelta(hours=24)
                )
                .scalar()
            )
            if n_dev >= self.policy.max_attempts_per_device_24h:
                raise SessionError("VELOCITY_DEVICE", "too many attempts from this device in 24h", 429)
            ids = (
                db.query(func.count(func.distinct(DBLivenessSession.user_ref)))
                .filter(
                    DBLivenessSession.device_id == device_id,
                    DBLivenessSession.user_ref != user_ref,
                    DBLivenessSession.created_at > now - timedelta(days=30),
                )
                .scalar()
            )
            if ids >= self.policy.max_identities_per_device_30d:
                raise SessionError("DEVICE_MULTI_IDENTITY", "device already used by other identities (30d)", 403)

    # -- lifecycle ------------------------------------------------------------
    def issue(self, db: Session, user_ref: str, device_id: str | None, vendor_token: str | None) -> IssuedSession:
        self.check_velocity(db, user_ref, device_id)
        sid = uuid.uuid4().hex
        nonce = secrets.token_hex(16)
        ch = challenge_from_nonce(bytes.fromhex(nonce), self.secret.encode())
        expires = _now() + timedelta(seconds=self.policy.session_ttl_seconds)
        db.add(
            DBLivenessSession(
                id=sid,
                user_ref=user_ref,
                device_id=device_id,
                nonce=nonce,
                challenge={"slots": list(ch.slots), "slot_frames": ch.slot_frames, "fps": ch.fps},
                vendor_token=vendor_token,
                status="issued",
                expires_at=expires,
            )
        )
        db.commit()
        return IssuedSession(sid, sign(self.secret, sid, nonce), ch, vendor_token, expires)

    def consume(self, db: Session, session_id: str, token: str) -> tuple[DBLivenessSession, Challenge]:
        s = db.get(DBLivenessSession, session_id)
        if s is None:
            raise SessionError("SESSION_UNKNOWN", "unknown session", 404)
        if not hmac.compare_digest(sign(self.secret, s.id, s.nonce), token or ""):
            raise SessionError("SESSION_TOKEN_INVALID", "invalid session token", 401)
        if s.expires_at < _now():
            raise SessionError("SESSION_EXPIRED", "session expired", 410)
        # atomic one-time consumption (race-safe replay protection)
        res = db.execute(
            update(DBLivenessSession)
            .where(DBLivenessSession.id == session_id, DBLivenessSession.status == "issued")
            .values(status="consumed", consumed_at=_now())
        )
        db.commit()
        if res.rowcount != 1:
            raise SessionError("SESSION_REPLAY", "session already used", 409)
        db.refresh(s)
        c = s.challenge
        return s, Challenge(tuple(c["slots"]), slot_frames=c["slot_frames"], fps=c["fps"])
