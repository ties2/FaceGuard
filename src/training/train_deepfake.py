"""Train the face-manipulation (deepfake / face-swap) detector and export ONNX.

    python -m src.training.train_deepfake --epochs 24

Training data: bona fide faces from the *train* split, delivered as a real
camera would (``bona_fide_frame``), and their *injection-faithful* Self-Blended
Images (``injected_self_blend``: a camera frame with a generated, resampled
copy of the same face blended in, delivered without a second camera pass).
No face-swap, GAN or real-time-swap sample is ever shown to the model — those
species are generalisation tests in the harness. Model selection
uses SBI on the *val* split only; swap/GAN AUCs on val are logged for
monitoring but never used to pick a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.attacks.capture import bona_fide_frame, deliver, injected_self_blend, live_capture_frame
from src.attacks.common import to_float, to_u8
from src.attacks.synthesis import face_swap, gan_upsample
from src.training.data import load_faces
from src.training.deepfake_net import MEAN, STD, DeepfakeNet

FLIP_IDX = [1, 0, 2, 4, 3]


def to_tensor(img_bgr_u8: np.ndarray) -> np.ndarray:
    x = img_bgr_u8[..., ::-1].astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1).copy()


def jitter_base(face: np.ndarray, lmk: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Appearance + geometry augmentation of the *base* image, applied before
    SBI. Without it a small network memorises the ~1k real training faces
    ("anything that deviates from the remembered real image is fake") instead
    of learning blending artefacts — train loss falls, val AUC does not."""
    h, w = face.shape[:2]
    ang = rng.uniform(-10, 10)
    sc = rng.uniform(0.9, 1.12)
    M = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-6, 6), h / 2 + rng.uniform(-6, 6)), ang, sc)
    out = cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    lm = (np.c_[lmk, np.ones(len(lmk))] @ M.T).astype(np.float32)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.75, 1.25) + rng.uniform(-15, 15), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3:
        g = rng.uniform(0.7, 1.4)
        out = (255 * (out / 255.0) ** g).astype(np.uint8)
    return out, lm


def flip(face: np.ndarray, lmk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = face.shape[1]
    lm = lmk[FLIP_IDX].copy()
    lm[:, 0] = w - 1 - lm[:, 0]
    return face[:, ::-1].copy(), lm


class SBIDataset(Dataset):
    def __init__(self, crops: np.ndarray, lmks: np.ndarray, seed: int):
        self.crops, self.lmks, self.seed, self.epoch = crops, lmks, seed, 0

    def __len__(self) -> int:
        return 2 * len(self.crops)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        face, lmk = self.crops[idx // 2], self.lmks[idx // 2]
        if rng.random() < 0.5:
            face, lmk = flip(face, lmk)
        face, lmk = jitter_base(face, lmk, rng)
        fake = idx % 2
        x = injected_self_blend(face, lmk, rng) if fake else bona_fide_frame(face, lmk, rng)
        return to_tensor(x), np.float32(fake)


def fixed_eval_set(crops, lmks, kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic evaluation tensors: one bona fide frame + one manipulation per face."""
    xs, ys = [], []
    for i, x in enumerate(eval_frames(crops, lmks, kind, seed)):
        xs.append(to_tensor(x))
        ys.append(i % 2)
    return np.stack(xs), np.array(ys)


def eval_frames(crops, lmks, kind: str, seed: int) -> list[np.ndarray]:
    """[real_0, fake_0, real_1, fake_1, ...] for kind in sbi | swap | gan."""
    rng = np.random.default_rng(seed)
    out, n = [], len(crops)
    for i in range(n):
        out.append(bona_fide_frame(crops[i], lmks[i], rng))
        if kind == "sbi":
            out.append(injected_self_blend(crops[i], lmks[i], rng))
        elif kind == "swap":  # unseen technique: cross-identity swap onto a camera frame
            j = (i + 1 + int(rng.integers(0, n - 1))) % n
            cam = to_float(live_capture_frame(crops[i], lmks[i], rng))
            out.append(deliver(to_u8(face_swap(crops[i], lmks[i], crops[j], lmks[j], rng, target_f=cam)), rng))
        else:  # unseen: fully synthetic face
            out.append(deliver(to_u8(gan_upsample(crops[i], rng)), rng))
    return out


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, bs: int = 128) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(x[i : i + bs]))).numpy().ravel() for i in range(0, len(x), bs)]
    return np.concatenate(out)


def export_onnx(model: nn.Module, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, 128, 128)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", type=Path, default=Path("artifacts/models/deepfake-detector.onnx"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)

    d = load_faces()
    tr, va = d["split"] == "train", d["split"] == "val"
    ds = SBIDataset(d["crops"][tr], d["lmk"][tr], seed=1)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=1, drop_last=True, persistent_workers=True)
    evals = {
        k: fixed_eval_set(d["crops"][va], d["lmk"][va], k, seed=100 + i) for i, k in enumerate(["sbi", "swap", "gan"])
    }

    model = DeepfakeNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.15)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_auc, best_state = [], -1.0, None
    for ep in range(args.epochs):
        ds.epoch = ep
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xb, yb in dl:
            loss = loss_fn(model(xb).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        aucs = {k: float(roc_auc_score(y, predict(model, x))) for k, (x, y) in evals.items()}
        history.append(
            {"epoch": ep + 1, "loss": tot / nb, **{f"val_auc_{k}": v for k, v in aucs.items()}, "sec": time.time() - t0}
        )
        print(json.dumps(history[-1]), flush=True)
        if aucs["sbi"] > best_auc:  # selection on SBI only (see module docstring)
            best_auc = aucs["sbi"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.out)
    torch.save(best_state, args.out.with_suffix(".pt"))
    best = max(history, key=lambda h: h["val_auc_sbi"])
    meta = {
        "model": "DeepfakeNet",
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best,
        "history": history,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("saved", args.out, "best", best)


if __name__ == "__main__":
    main()
