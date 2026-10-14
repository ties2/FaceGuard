"""End-to-end analysis of one capture session.

Detector models are ordinary MSP-registered models resolved to their current
*production* version through the MSP cache, so a promote / rollback through
the registry API changes what runs here on the next request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from src.liveness.challenge import Challenge
from src.liveness.face import FaceBox, FaceDetector, context_crop
from src.liveness.signals import deepfake as df_layer
from src.liveness.signals.artifacts import artifact_features
from src.liveness.signals.capture_integrity import (
    Finding,
    challenge_findings,
    explain,
    feature_vector,
    integrity_rules,
    stream_features,
)
from src.liveness.signals.matching import FaceMatcher
from src.liveness.signals.texture import pad_features
from src.serving.service import ModelRuntime

MODEL_PAD = "faceguard-pad-texture"
MODEL_CI = "faceguard-capture-integrity"
MODEL_DF = "faceguard-deepfake"
MODEL_DF_FORENSICS = "faceguard-deepfake-forensics"


class NoFaceError(Exception):
    pass


@dataclass
class Analysis:
    pad: float | None
    capture_integrity: float | None
    deepfake: float | None
    face_match: float | None
    findings: list[Finding]
    explanations: list[Finding]
    stream: dict
    model_versions: dict = field(default_factory=dict)
    deepfake_parts: dict = field(default_factory=dict)
    latency_s: float = 0.0


def _proba(model, x: np.ndarray) -> float:
    return float(model.predict_proba(x.reshape(1, -1))[0, 1])


class LivenessPipeline:
    def __init__(self, detector: FaceDetector, matcher: FaceMatcher | None = None):
        self.detector = detector
        self.matcher = matcher

    def crops(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, FaceBox]:
        """Detect on the first and middle frame (a face must be present in both);
        crop every frame with the *first* frame's box so the face/background
        masks of the photometric check stay fixed in crop coordinates (heads
        move little within ~1 s)."""
        boxes = {}
        for k in (0, len(frames) // 2):
            b = self.detector.detect(frames[k])
            if b is None:
                raise NoFaceError(f"no face in frame {k}")
            boxes[k] = b
        crops, lmk0 = [], None
        for i, f in enumerate(frames):
            c, lm = context_crop(f, boxes[0])
            if i == 0:
                lmk0 = lm
            crops.append(c)
        return crops, lmk0, boxes[len(frames) // 2]

    def analyse(
        self,
        db: Session,
        frames: list[np.ndarray],
        timestamps_ms: list[float],
        telemetry: dict,
        challenge: Challenge,
        reference: np.ndarray | None = None,
        audit_crop: np.ndarray | None = None,
    ) -> Analysis:
        t0 = time.time()
        crops, lmk, mid_box = self.crops(frames)
        findings = integrity_rules(telemetry)
        feats = stream_features(crops, np.asarray(timestamps_ms), lmk, challenge.per_frame(), challenge.fps)
        findings += challenge_findings(feats)
        explanations = explain(feats)
        versions = {}

        v, ci_model = ModelRuntime.get_production_model(db, MODEL_CI)
        versions[MODEL_CI] = v
        ci = _proba(ci_model, feature_vector(feats))

        v, pad_model = ModelRuntime.get_production_model(db, MODEL_PAD)
        versions[MODEL_PAD] = v
        audit = audit_crop if audit_crop is not None else crops[len(crops) // 2]
        pad = _proba(pad_model, pad_features(audit))

        # deepfake layer = mean of two independent views of the same frames:
        # the CNN (learned) and the forensics model (noise/sharpness/seam ratios)
        picks = df_layer.pick_frames(len(crops))
        v, df_model = ModelRuntime.get_production_model(db, MODEL_DF)
        versions[MODEL_DF] = v
        cnn = df_layer.score(df_model, [crops[i] for i in picks])
        v, fx_model = ModelRuntime.get_production_model(db, MODEL_DF_FORENSICS)
        versions[MODEL_DF_FORENSICS] = v
        fx = float(np.mean(fx_model.predict_proba(np.stack([artifact_features(crops[i], lmk) for i in picks]))[:, 1]))
        dfs = 0.5 * (cnn + fx)

        match = None
        if reference is not None and self.matcher is not None:
            rb = self.detector.detect(reference)
            if rb is not None:
                match = self.matcher.cosine(
                    self.matcher.embed(frames[len(frames) // 2], mid_box), self.matcher.embed(reference, rb)
                )
        return Analysis(
            pad=pad,
            capture_integrity=ci,
            deepfake=dfs,
            face_match=match,
            findings=findings,
            explanations=explanations,
            stream=feats,
            model_versions=versions,
            deepfake_parts={"cnn": cnn, "forensics": fx},
            latency_s=time.time() - t0,
        )
