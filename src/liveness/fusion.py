"""Risk fusion and decision policy.

Each defence layer reports a calibrated attack probability. The policy maps
them to one of four outcomes, most severe wins:

    REJECT         hard evidence (vendor security check failed, known injection
                   tooling, layer score above its reject threshold)
    MANUAL_REVIEW  manipulation suspected but not certain (deepfake / PAD in
                   the review band) -> human with the evidence pack
    STEP_UP        capture channel suspicious but the user is plausibly honest
                   (soft telemetry findings, capture-integrity review band)
                   -> retry in the native app / NFC document chip / video call
    APPROVE        everything below review thresholds

Thresholds live in ``configs/policy.yaml`` and are produced by
``src.evaluation.calibrate`` from bona fide score quantiles (a BPCER budget per
layer), so they can be re-derived whenever a model version is promoted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path

import yaml

from src.liveness.signals.capture_integrity import Finding


class Decision(IntEnum):
    APPROVE = 0
    STEP_UP = 1
    MANUAL_REVIEW = 2
    REJECT = 3


@dataclass
class Band:
    review: float
    reject: float


@dataclass
class Policy:
    version: str = "uncalibrated"
    pad: Band = field(default_factory=lambda: Band(0.5, 0.9))
    capture_integrity: Band = field(default_factory=lambda: Band(0.5, 0.9))
    deepfake: Band = field(default_factory=lambda: Band(0.5, 0.9))
    face_match_min_cosine: float = 0.363  # SFace operating point published by OpenCV Zoo
    # Velocity limits (anti-probing). See docs/04-recommendations.md.
    max_attempts_per_identity_24h: int = 3
    max_attempts_per_device_24h: int = 5
    max_identities_per_device_30d: int = 2
    session_ttl_seconds: int = 120
    calibration: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        p = Path(path)
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text()) or {}
        for k in ("pad", "capture_integrity", "deepfake"):
            if k in raw:
                raw[k] = Band(**raw[k])
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


@dataclass
class LayerResult:
    name: str
    score: float | None
    band: str  # pass | review | reject | n/a
    reasons: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    decision: Decision
    layers: list[LayerResult]
    reasons: list[str]
    policy_version: str

    @property
    def risk_score(self) -> float:
        scores = [layer.score for layer in self.layers if layer.score is not None]
        return float(max(scores)) if scores else 0.0


def _band(score: float, b: Band) -> str:
    if score >= b.reject:
        return "reject"
    if score >= b.review:
        return "review"
    return "pass"


def decide(
    policy: Policy,
    *,
    vendor_ok: bool | None,
    vendor_reasons: list[str] | None = None,
    findings: list[Finding] | None = None,
    pad: float | None = None,
    capture_integrity: float | None = None,
    deepfake: float | None = None,
    face_match: float | None = None,
    explanations: list[Finding] | None = None,
    deepfake_detail: dict | None = None,
) -> Verdict:
    findings = findings or []
    reasons: list[str] = []
    decision = Decision.APPROVE
    layers: list[LayerResult] = []

    def escalate(to: Decision, why: str):
        nonlocal decision
        reasons.append(f"{to.name}: {why}")
        decision = max(decision, to)

    # 1. vendor (FaceTec) verdict + its server-side security checks
    if vendor_ok is not None:
        layers.append(LayerResult("vendor_liveness", None, "pass" if vendor_ok else "reject", vendor_reasons or []))
        if not vendor_ok:
            escalate(Decision.REJECT, "vendor liveness / security checks failed: " + ", ".join(vendor_reasons or []))

    # 2. deterministic integrity findings
    for f in findings:
        escalate(Decision.REJECT if f.severity == "hard" else Decision.STEP_UP, f"{f.code} ({f.detail})")

    # 3. model layers
    for name, score, band_cfg, review_action in (
        ("pad_texture", pad, policy.pad, Decision.MANUAL_REVIEW),
        ("capture_integrity", capture_integrity, policy.capture_integrity, Decision.STEP_UP),
        ("deepfake", deepfake, policy.deepfake, Decision.MANUAL_REVIEW),
    ):
        if score is None:
            layers.append(LayerResult(name, None, "n/a"))
            continue
        band = _band(score, band_cfg)
        extra = [e.code for e in (explanations or [])] if name == "capture_integrity" else []
        if name == "deepfake" and deepfake_detail:
            extra = [f"{k}={v:.3f}" for k, v in deepfake_detail.items()]
        layers.append(LayerResult(name, float(score), band, extra))
        if band == "reject":
            escalate(Decision.REJECT, f"{name} score {score:.3f} >= {band_cfg.reject:.3f}")
        elif band == "review":
            escalate(review_action, f"{name} score {score:.3f} in review band")

    # 4. 1:1 face match against the reference (ID photo / enrolment)
    if face_match is not None:
        ok = face_match >= policy.face_match_min_cosine
        layers.append(LayerResult("face_match", float(face_match), "pass" if ok else "reject"))
        if not ok:
            escalate(Decision.REJECT, f"face match cosine {face_match:.3f} < {policy.face_match_min_cosine}")

    return Verdict(decision, layers, reasons, policy.version)
