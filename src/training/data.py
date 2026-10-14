"""Build the bona fide face corpus used by training and the attack harness.

    python -m src.training.data --src data/raw/face-mask-detection/dataset/without_mask \
                                --src data/raw/deepface/tests/unit/dataset

Steps: detect (YuNet) -> context crop 128x128 + 5 landmarks -> near-duplicate
removal (64-bit dHash, Hamming <= 4) -> deterministic 60/20/20 split.
The split is by *image*; near-duplicate removal keeps augmented copies of the
same photo from leaking across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.liveness.face import FaceDetector, context_crop

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def dedupe(hashes: list[int], max_dist: int = 4) -> list[int]:
    keep: list[int] = []
    arr = np.array(hashes, dtype=np.uint64)
    kept_hashes: list[np.uint64] = []
    for i, h in enumerate(arr):
        if kept_hashes:
            x = np.bitwise_xor(np.array(kept_hashes, dtype=np.uint64), h)
            dist = np.array([bin(int(v)).count("1") for v in x])
            if dist.min() <= max_dist:
                continue
        keep.append(i)
        kept_hashes.append(h)
    return keep


def build(sources: list[Path], out: Path, yunet: str, min_face: int = 40) -> dict:
    det = FaceDetector(yunet, score_threshold=0.7)
    crops, lmks, names, hashes = [], [], [], []
    stats = {"seen": 0, "no_face": 0, "too_small": 0}
    files = sorted(p for s in sources for p in s.rglob("*") if p.suffix.lower() in IMG_EXT)
    for f in files:
        stats["seen"] += 1
        img = cv2.imread(str(f))
        if img is None:
            continue
        box = det.detect(img)
        if box is None:
            stats["no_face"] += 1
            continue
        if min(box.w, box.h) < min_face:
            stats["too_small"] += 1
            continue
        crop, lmk = context_crop(img, box)
        crops.append(crop)
        lmks.append(lmk)
        names.append(f"{f.parent.name}/{f.name}")
        hashes.append(dhash(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))

    keep = dedupe(hashes)
    stats["near_duplicates_removed"] = len(crops) - len(keep)
    crops = np.stack([crops[i] for i in keep])
    lmks = np.stack([lmks[i] for i in keep])
    names = [names[i] for i in keep]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(crops))
    n = len(order)
    split = np.empty(n, dtype="<U5")
    split[order[: int(0.6 * n)]] = "train"
    split[order[int(0.6 * n) : int(0.8 * n)]] = "val"
    split[order[int(0.8 * n) :]] = "test"

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, crops=crops, lmk=lmks, split=split, names=np.array(names))
    digest = hashlib.sha256(crops.tobytes()).hexdigest()[:16]
    stats.update(
        {
            "kept": int(n),
            "train": int((split == "train").sum()),
            "val": int((split == "val").sum()),
            "test": int((split == "test").sum()),
            "crop_size": int(crops.shape[1]),
            "sha256_16": digest,
        }
    )
    (out.parent / "faces_manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_faces(path: str | Path = "data/processed/faces.npz") -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/faces.npz"))
    ap.add_argument("--yunet", default="assets/models/face_detection_yunet_2023mar.onnx")
    args = ap.parse_args()
    print(json.dumps(build(args.src, args.out, args.yunet), indent=2))


if __name__ == "__main__":
    main()
