"""Train the two tabular layers from the harness' *train* split.

    python -m src.training.train_tabular

* ``pad-texture``        bona fide vs print / screen replay (colour-texture + spectrum)
                         -> also the "vendor stand-in" in the baseline (B0)
* ``capture-integrity``  bona fide vs every attack species marked
                         ``seen_in_training`` (stream + challenge features)

Both are scikit-learn HistGradientBoosting models saved with joblib, i.e. the
exact artifact contract MSP's SklearnJoblibLoader already serves.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from src.attacks.catalog import PRESENTATION_SPECIES
from src.evaluation.harness import load_split
from src.liveness.signals.capture_integrity import CI_FEATURES

OUT = Path("artifacts/models")


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def _save(model, name: str, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    (OUT / f"{name}.joblib").write_bytes(buf.getvalue())
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(name, json.dumps(meta))


def main() -> None:
    df, pad, _ = load_split("train")

    m = df["species"].isin(["bona_fide", *PRESENTATION_SPECIES]).to_numpy()
    x, y = pad[m], df.loc[m, "is_attack"].to_numpy()
    model = _hgb(0)
    cv = cross_val_score(_hgb(0), x, y, cv=4, scoring="roc_auc")
    model.fit(x, y)
    _save(
        model,
        "pad-texture",
        {"n": int(len(y)), "n_attack": int(y.sum()), "cv_auc": float(np.mean(cv)), "features": int(x.shape[1])},
    )

    x = df[CI_FEATURES].to_numpy(np.float32)
    y = df["is_attack"].to_numpy()
    model = _hgb(1)
    cv = cross_val_score(_hgb(1), x, y, cv=4, scoring="roc_auc")
    model.fit(x, y)
    _save(
        model,
        "capture-integrity",
        {
            "n": int(len(y)),
            "n_attack": int(y.sum()),
            "cv_auc": float(np.mean(cv)),
            "features": CI_FEATURES,
            "species": sorted(df["species"].unique().tolist()),
        },
    )


if __name__ == "__main__":
    main()
