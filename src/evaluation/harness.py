"""Adversarial test harness: generate capture sessions for a split and extract
every layer's inputs.

    python -m src.evaluation.harness --split train
    python -m src.evaluation.harness --split val
    python -m src.evaluation.harness --split test

Per session we store: species, attacker/victim face ids, telemetry findings,
PAD features (audit frame), capture-integrity stream features, and three
frames (for the deepfake CNN) so models can be re-scored without regenerating.
Everything is seeded: the same command produces the same campaign.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.attacks.capture import simulate
from src.attacks.catalog import ALL_ATTACKS, SPECIES
from src.liveness.challenge import random_challenge
from src.liveness.signals.capture_integrity import CI_FEATURES, integrity_rules, stream_features
from src.liveness.signals.texture import pad_features
from src.training.data import load_faces

OUT = Path("data/sessions")
DF_FRAMES = (5, 15, 25)  # frames scored by the deepfake CNN (same indices as the API pipeline)
SEEN = [k for k, s in SPECIES.items() if s.seen_in_training]

PLAN = {
    # split: (bona fide sessions per face, attack species, attack sessions per face per species)
    "train": (2, SEEN, 1),
    "val": (3, ALL_ATTACKS, 1),
    "test": (3, ALL_ATTACKS, 1),
}
SPLIT_SEED = {"train": 11, "val": 22, "test": 33}

_FACES: dict | None = None
_DONORS: dict[int, int] = {}


def yaw_proxy(lmk: np.ndarray) -> float:
    eye_mid = lmk[:2].mean(axis=0)
    iod = float(np.linalg.norm(lmk[1] - lmk[0])) + 1e-6
    return float((lmk[2, 0] - eye_mid[0]) / iod)


def choose_donor(i: int, idx: np.ndarray, yaws: np.ndarray, rng: np.random.Generator) -> int:
    """Victim with a compatible head pose (attackers pick usable source media)."""
    cand = idx[(idx != i) & (np.abs(yaws[idx] - yaws[i]) < 0.15) & (np.abs(yaws[idx]) < 0.35)]
    if len(cand) == 0:
        cand = idx[idx != i]
    return int(rng.choice(cand))


def _run(task: tuple[int, int, str, int]) -> tuple[dict, np.ndarray, np.ndarray]:
    sid, i, species, seed = task
    rng = np.random.default_rng(seed)
    d = _FACES
    j = _DONORS[sid]
    ch = random_challenge(rng)
    s = simulate(species, d["crops"][i], d["lmk"][i], d["crops"][j], d["lmk"][j], ch, rng)
    feats = stream_features(s.frames, s.timestamps_ms, s.lmk, ch.per_frame())
    findings = integrity_rules(s.telemetry)
    rec = {
        "session_id": sid,
        "species": species,
        "is_attack": int(species != "bona_fide"),
        "face": i,
        "victim": j if species != "bona_fide" else i,
        "platform": s.telemetry.get("platform"),
        "camera_label": s.telemetry.get("camera_label"),
        "rule_codes": ";".join(f.code for f in findings),
        "rule_hard": int(any(f.severity == "hard" for f in findings)),
        "rule_soft": int(any(f.severity == "soft" for f in findings)),
        **{k: feats[k] for k in CI_FEATURES},
        **{f"lmk_{k}": float(v) for k, v in enumerate(np.asarray(s.lmk).ravel())},
        **{f"meta_{k}": v for k, v in s.meta.items() if isinstance(v, (int, float))},
    }
    return rec, pad_features(s.audit_frame), np.stack([s.frames[k] for k in DF_FRAMES])


def plan_tasks(split: str) -> tuple[list[tuple[int, int, str, int]], dict[int, int]]:
    d = load_faces()
    idx = np.nonzero(d["split"] == split)[0]
    yaws = np.array([yaw_proxy(lmk) for lmk in d["lmk"]])
    n_bf, species, n_att = PLAN[split]
    rng = np.random.default_rng(SPLIT_SEED[split])
    tasks, donors, sid = [], {}, 0
    for i in idx:
        for _ in range(n_bf):
            tasks.append((sid, int(i), "bona_fide", int(rng.integers(1 << 62))))
            donors[sid] = int(i)
            sid += 1
        for sp in species:
            for _ in range(n_att):
                tasks.append((sid, int(i), sp, int(rng.integers(1 << 62))))
                donors[sid] = choose_donor(int(i), idx, yaws, rng)
                sid += 1
    return tasks, donors


def _init_with(donors):
    global _DONORS, _FACES
    _DONORS = donors
    _FACES = load_faces()


def run(split: str, workers: int = 2) -> pd.DataFrame:
    tasks, donors = plan_tasks(split)
    recs, pads, frames = [], [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_with, initargs=(donors,)) as ex:
        for n, (rec, pad, fr) in enumerate(ex.map(_run, tasks, chunksize=16)):
            recs.append(rec)
            pads.append(pad)
            if split != "train":
                frames.append(fr)
            if n % 500 == 0:
                print(f"[{split}] {n}/{len(tasks)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs)
    df["split"] = split
    df.to_csv(OUT / f"{split}_records.csv", index=False)
    np.save(OUT / f"{split}_pad.npy", np.stack(pads).astype(np.float32))
    if frames:
        np.save(OUT / f"{split}_frames.npy", np.stack(frames))
    summary = df.groupby("species").size().to_dict()
    (OUT / f"{split}_summary.json").write_text(json.dumps({"sessions": len(df), "by_species": summary}, indent=2))
    return df


def load_split(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(OUT / f"{split}_records.csv", keep_default_na=False)
    pad = np.load(OUT / f"{split}_pad.npy")
    fp = OUT / f"{split}_frames.npy"
    frames = np.load(fp, mmap_mode="r") if fp.exists() else None
    return df, pad, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(PLAN), required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    df = run(args.split, args.workers)
    print(df.groupby("species").size())


if __name__ == "__main__":
    main()
