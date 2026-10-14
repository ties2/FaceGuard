"""Score the val/test campaigns, calibrate the policy on val, evaluate on test.

    python -m src.evaluation.run_eval

Outputs
  reports/scores_{val,test}.csv   per-session layer scores
  configs/policy.yaml             calibrated thresholds (consumed by the API)
  reports/results.json            every number quoted in docs/02 and docs/03

Decisions are produced by ``src.liveness.fusion.decide`` — the same function
the API calls — so the evaluated policy is the deployed policy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.attacks.catalog import ALL_ATTACKS, SPECIES
from src.evaluation import metrics as M
from src.evaluation.harness import load_split
from src.liveness.fusion import Band, Decision, Policy, decide
from src.liveness.signals import deepfake as df_layer
from src.liveness.signals.artifacts import artifact_features
from src.liveness.signals.capture_integrity import CI_FEATURES, Finding, challenge_findings
from src.serving.loader import OnnxRuntimeLoader

ART = Path("artifacts/models")
REPORTS = Path("reports")
HARD = {"VIRTUAL_CAMERA", "BROWSER_AUTOMATION", "EMULATOR", "HOOKING_FRAMEWORK", "ATTESTATION_FAILED"}

# BPCER budget per layer (share of bona fide sessions allowed above each threshold, on val)
BUDGET = {
    "vendor_pad": 0.010,
    "pad_reject": 0.005,
    "ci_review": 0.020,
    "ci_reject": 0.005,
    "df_review": 0.015,
    "df_reject": 0.005,
}

VARIANTS = {
    "B0": "Vendor liveness only (PAD stand-in)",
    "B1": "+ device / telemetry integrity rules",
    "B2": "+ capture-integrity model (nonce challenge, timing, sensor noise)",
    "B3": "+ deepfake detector = FaceGuard full",
}

LAYER_SPECIES = {
    "pad": ["print_photo", "screen_replay"],
    "ci": ALL_ATTACKS,
    # face-manipulation species; fully synthetic faces are reported per species (not this layer's scope)
    "df": ["vcam_offline_faceswap", "vcam_realtime_faceswap", "challenge_aware_faceswap"],
}


def score_split(split: str) -> pd.DataFrame:
    df, pad, frames = load_split(split)
    pad_model = joblib.load(ART / "pad-texture.joblib")
    ci_model = joblib.load(ART / "capture-integrity.joblib")
    df_model = OnnxRuntimeLoader().load((ART / "deepfake-detector.onnx").read_bytes())
    df["s_pad"] = pad_model.predict_proba(pad)[:, 1]
    df["s_ci"] = ci_model.predict_proba(df[CI_FEATURES].to_numpy(np.float32))[:, 1]
    out = []
    for i in range(0, len(frames), 64):
        batch = np.asarray(frames[i : i + 64])  # (B, 3, H, W, 3)
        b, k = batch.shape[:2]
        x = df_layer.preprocess(list(batch.reshape(b * k, *batch.shape[2:])))
        logits = df_model.run({df_model.input_names[0]: x})[0].reshape(b, k)
        out.append((1 / (1 + np.exp(-logits))).mean(axis=1))
    df["s_df_cnn"] = np.concatenate(out)
    fx_model = joblib.load(ART / "deepfake-forensics.joblib")
    lmk_cols = [f"lmk_{k}" for k in range(10)]
    fx = []
    for n in range(len(df)):
        lmk = df[lmk_cols].iloc[n].to_numpy(np.float32).reshape(5, 2)
        feats = np.stack([artifact_features(np.asarray(f), lmk) for f in frames[n]])
        fx.append(fx_model.predict_proba(feats)[:, 1].mean())
    df["s_df_fx"] = np.array(fx)
    df["s_df"] = 0.5 * (df["s_df_cnn"] + df["s_df_fx"])  # same ensemble as the API pipeline
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / f"scores_{split}.csv", index=False)
    return df


def calibrate(val: pd.DataFrame) -> Policy:
    bf = val[val.species == "bona_fide"]
    t = lambda col, b: M.threshold_for_bpcer(bf[col].to_numpy(), b)  # noqa: E731
    vendor_t = t("s_pad", BUDGET["vendor_pad"])
    pol = Policy(
        version=f"cal-{datetime.now(UTC):%Y%m%d}-val{len(bf)}",
        pad=Band(review=vendor_t, reject=max(vendor_t, t("s_pad", BUDGET["pad_reject"]))),
        capture_integrity=Band(review=t("s_ci", BUDGET["ci_review"]), reject=t("s_ci", BUDGET["ci_reject"])),
        deepfake=Band(review=t("s_df", BUDGET["df_review"]), reject=t("s_df", BUDGET["df_reject"])),
        calibration={
            "split": "val",
            "bona_fide_sessions": int(len(bf)),
            "bpcer_budget": BUDGET,
            "vendor_pad_threshold": vendor_t,
            "method": "per-layer bona fide quantiles (attack data not used)",
        },
    )
    return pol


def findings_of(codes: str) -> list[Finding]:
    return [Finding(c, "hard" if c in HARD else "soft", "") for c in codes.split(";") if c]


def decide_variant(row, pol: Policy, variant: str, delay_rule: bool = True) -> Decision:
    vendor_ok = bool(row.s_pad < pol.calibration["vendor_pad_threshold"])
    kw = {"vendor_ok": vendor_ok, "vendor_reasons": [] if vendor_ok else ["faceScanLivenessCheckSucceeded"]}
    if variant in ("B1", "B2", "B3"):
        kw["findings"] = findings_of(row.rule_codes)
    if variant in ("B2", "B3"):
        kw["capture_integrity"] = row.s_ci
        if delay_rule:
            kw["findings"] = kw["findings"] + challenge_findings({k: getattr(row, k) for k in CI_FEATURES})
    if variant == "B3":
        kw["deepfake"] = row.s_df
        kw["pad"] = row.s_pad
    return decide(pol, **kw).decision


def evaluate(test: pd.DataFrame, pol: Policy) -> dict:
    res: dict = {"variants": {}, "layers": {}, "campaign": {}}
    for v in VARIANTS:
        test[f"d_{v}"] = [decide_variant(r, pol, v) for r in test.itertuples()]
    bf = test[test.species == "bona_fide"]
    res["campaign"] = {
        "bona_fide_sessions": int(len(bf)),
        "bona_fide_subjects": int(bf.face.nunique()),
        "attack_sessions": int((test.species != "bona_fide").sum()),
        "per_species": {s: int((test.species == s).sum()) for s in ALL_ATTACKS},
    }
    for v, desc in VARIANTS.items():
        d = test[f"d_{v}"]
        out = {"description": desc, "bona_fide": {}, "species": {}}
        bfd = d[test.species == "bona_fide"]
        out["bona_fide"] = {
            "approve": M.rate(bfd == Decision.APPROVE),
            "step_up": M.rate(bfd == Decision.STEP_UP),
            "manual_review": M.rate(bfd == Decision.MANUAL_REVIEW),
            "reject": M.rate(bfd == Decision.REJECT),
            "bpcer_reject": M.rate(bfd == Decision.REJECT),
            "friction": M.rate(bfd != Decision.APPROVE),
        }
        for s in ALL_ATTACKS:
            ds = d[test.species == s]
            out["species"][s] = {
                "apcer": M.rate(ds == Decision.APPROVE),
                "step_up": M.rate(ds == Decision.STEP_UP),
                "manual_review": M.rate(ds == Decision.MANUAL_REVIEW),
                "reject": M.rate(ds == Decision.REJECT),
            }
        ap = [out["species"][s]["apcer"]["rate"] for s in ALL_ATTACKS]
        out["max_apcer"] = float(max(ap))
        out["max_apcer_species"] = ALL_ATTACKS[int(np.argmax(ap))]
        out["mean_apcer"] = float(np.mean(ap))
        res["variants"][v] = out

    for layer, col in (("pad", "s_pad"), ("ci", "s_ci"), ("df", "s_df"), ("df_cnn", "s_df_cnn"), ("df_fx", "s_df_fx")):
        b = bf[col].to_numpy()
        a = test[test.species.isin(LAYER_SPECIES[layer.split("_")[0]])][col].to_numpy()
        y = np.r_[np.zeros(len(b)), np.ones(len(a))]
        res["layers"][layer] = {
            "attack_species": LAYER_SPECIES[layer.split("_")[0]],
            "auc": float(roc_auc_score(y, np.r_[b, a])),
            "eer": M.eer(b, a),
            "bpcer_at_apcer_5": M.bpcer_at_apcer(b, a, 0.05),
            "bpcer_at_apcer_1": M.bpcer_at_apcer(b, a, 0.01),
            "per_species_auc": {
                s: float(
                    roc_auc_score(
                        np.r_[np.zeros(len(b)), np.ones((test.species == s).sum())],
                        np.r_[b, test[test.species == s][col]],
                    )
                )
                for s in ALL_ATTACKS
            },
        }
    # ablation on the same campaign: FaceGuard without the delayed-reflection rule (red-team run 1 policy)
    d0 = pd.Series([decide_variant(r, pol, "B3", delay_rule=False) for r in test.itertuples()], index=test.index)
    abl = {"species": {}, "bona_fide": {}}
    for s in ALL_ATTACKS:
        abl["species"][s] = {"apcer": M.rate(d0[test.species == s] == Decision.APPROVE)}
    bfd0 = d0[test.species == "bona_fide"]
    abl["bona_fide"] = {"bpcer_reject": M.rate(bfd0 == Decision.REJECT), "friction": M.rate(bfd0 != Decision.APPROVE)}
    ap0 = [abl["species"][s]["apcer"]["rate"] for s in ALL_ATTACKS]
    abl["mean_apcer"], abl["max_apcer"] = float(np.mean(ap0)), float(np.max(ap0))
    res["ablation_without_delay_rule"] = abl

    res["species_catalog"] = {k: vars(v) for k, v in SPECIES.items()}
    res["policy"] = {
        "version": pol.version,
        "vendor_pad_threshold": pol.calibration["vendor_pad_threshold"],
        "pad": vars(pol.pad),
        "capture_integrity": vars(pol.capture_integrity),
        "deepfake": vars(pol.deepfake),
    }
    # retry / probing: cumulative success with the per-attempt APCER of the worst species
    worst = res["variants"]["B3"]["max_apcer"]
    res["retry_risk"] = {
        "per_attempt_apcer": worst,
        "species": res["variants"]["B3"]["max_apcer_species"],
        "cumulative": {n: M.cumulative_success(worst, n) for n in (1, 3, 5, 10, 20, 50)},
    }
    return res


def matcher_eval(test: pd.DataFrame, frames: np.ndarray, n_max: int = 361) -> dict:
    """IAPMR: does the *matcher* accept face-swap attacks as the victim?

    Reference = the victim's source photo (the same image the swap was built
    from), which makes this an upper bound — with an independent ID-document
    portrait as reference IAPMR is typically lower. Zero-effort impostor
    acceptance (bona fide frame vs. another person) is reported for context.
    """
    from src.liveness.face import FaceDetector
    from src.liveness.signals.matching import FaceMatcher
    from src.serving.config import settings
    from src.training.data import load_faces

    faces = load_faces()["crops"]
    det = FaceDetector(settings.faceguard_yunet_path, score_threshold=0.5)
    fm = FaceMatcher(settings.faceguard_sface_path)
    thr = Policy().face_match_min_cosine
    cache: dict[int, np.ndarray | None] = {}

    def emb(img):
        b = det.detect(img)
        return None if b is None else fm.embed(img, b)

    def ref(i):
        if i not in cache:
            cache[i] = emb(faces[i])
        return cache[i]

    out: dict = {"threshold_cosine": thr, "reference": "victim source photo (upper bound)"}
    rng = np.random.default_rng(5)
    for species in ("vcam_offline_faceswap", "vcam_realtime_faceswap", "challenge_aware_faceswap"):
        pos = np.nonzero((test.species == species).to_numpy())[0][:n_max]
        vic, att = [], []
        for k in pos:
            e = emb(np.asarray(frames[k][1]))
            rv, ra = ref(int(test.victim.iloc[k])), ref(int(test.face.iloc[k]))
            if e is None or rv is None or ra is None:
                continue
            vic.append(fm.cosine(e, rv))
            att.append(fm.cosine(e, ra))
        vic, att = np.array(vic), np.array(att)
        out[species] = {
            "n": int(len(vic)),
            "iapmr": M.rate(vic >= thr),
            "matches_attacker": M.rate(att >= thr),
            "median_cos_victim": float(np.median(vic)),
            "median_cos_attacker": float(np.median(att)),
        }
    pos = np.nonzero((test.species == "bona_fide").to_numpy())[0][::3][:n_max]
    imp = []
    for k in pos:
        e = emb(np.asarray(frames[k][1]))
        other = int(rng.choice(test.face.unique()))
        if other == int(test.face.iloc[k]) or e is None or ref(other) is None:
            continue
        imp.append(fm.cosine(e, ref(other)))
    out["zero_effort_impostor"] = {"n": len(imp), "fmr": M.rate(np.array(imp) >= thr)}
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-scores", action="store_true", help="skip model scoring, reuse reports/scores_*.csv")
    args = ap.parse_args()
    if args.reuse_scores:
        val = pd.read_csv(REPORTS / "scores_val.csv", keep_default_na=False)
        test = pd.read_csv(REPORTS / "scores_test.csv", keep_default_na=False)
        test = test[[c for c in test.columns if not c.startswith("d_")]]
    else:
        val = score_split("val")
        test = score_split("test")
    pol = calibrate(val)
    pol.save("configs/policy.yaml")
    res = evaluate(test, pol)
    res["matching"] = matcher_eval(test, load_split("test")[2])
    res["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    test.to_csv(REPORTS / "scores_test.csv", index=False)
    (REPORTS / "results.json").write_text(json.dumps(res, indent=2, default=float))
    for v, o in res["variants"].items():
        print(
            f"{v}: BPCER(reject)={o['bona_fide']['bpcer_reject']['rate']:.3%} "
            f"friction={o['bona_fide']['friction']['rate']:.2%} "
            f"maxAPCER={o['max_apcer']:.2%} ({o['max_apcer_species']}) meanAPCER={o['mean_apcer']:.2%}"
        )
        for s in ALL_ATTACKS:
            print(f"    {s:28s} APCER={o['species'][s]['apcer']['rate']:.2%}")
    for layer, o in res["layers"].items():
        print(layer, f"AUC={o['auc']:.4f} EER={o['eer']:.3%} BPCER@APCER5={o['bpcer_at_apcer_5']['bpcer']:.2%}")
    print(json.dumps(res["matching"], indent=1, default=float))


if __name__ == "__main__":
    main()
