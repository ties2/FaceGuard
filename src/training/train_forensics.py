"""Train the hand-crafted forensics model (second half of the deepfake layer).

    python -m src.training.train_forensics

Same training principle as the CNN: bona fide camera frames vs injection-
faithful Self-Blended Images from the *train* split only (3 passes with fresh
randomness). Swap and GAN AUCs on *val* are reported for monitoring only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.attacks.capture import bona_fide_frame, injected_self_blend
from src.liveness.signals.artifacts import ART_FEATURES, artifact_features
from src.training.data import load_faces
from src.training.train_deepfake import eval_frames, flip, jitter_base

OUT = Path("artifacts/models")


def main(passes: int = 3) -> None:
    d = load_faces()
    tr = np.nonzero(d["split"] == "train")[0]
    va = np.nonzero(d["split"] == "val")[0]
    rng = np.random.default_rng(7)
    x, y = [], []
    for _ in range(passes):
        for i in tr:
            face, lmk = d["crops"][i], d["lmk"][i]
            if rng.random() < 0.5:
                face, lmk = flip(face, lmk)
            face, lmk = jitter_base(face, lmk, rng)
            x.append(artifact_features(bona_fide_frame(face, lmk, rng), lmk))
            y.append(0)
            x.append(artifact_features(injected_self_blend(face, lmk, rng), lmk))
            y.append(1)
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
    ).fit(np.array(x), np.array(y))

    val_auc = {}
    for k, kind in enumerate(("sbi", "swap", "gan")):
        frames = eval_frames(d["crops"][va], d["lmk"][va], kind, seed=200 + k)
        lm = np.repeat(d["lmk"][va], 2, axis=0)
        feats = np.array([artifact_features(f, lm[n]) for n, f in enumerate(frames)])
        val_auc[kind] = float(roc_auc_score(np.arange(len(frames)) % 2, model.predict_proba(feats)[:, 1]))

    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    (OUT / "deepfake-forensics.joblib").write_bytes(buf.getvalue())
    meta = {"n": len(y), "features": ART_FEATURES, "val_auc": val_auc}
    (OUT / "deepfake-forensics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
