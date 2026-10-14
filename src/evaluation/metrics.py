"""PAD / injection-detection metrics (ISO/IEC 30107-3 vocabulary).

* APCER (per PAI species): share of attack presentations of that species that
  were *accepted* as bona fide. ISO/IEC 30107-3 reports it per species and
  the worst species ("max APCER") for a given attack-potential level.
* BPCER: share of bona fide presentations wrongly classified as attacks.
* BPCER@APCER=x: BPCER at the threshold where APCER (pooled) equals x —
  BPCER20 = BPCER at APCER 5 %, BPCER100 = BPCER at APCER 1 %.
* IAPMR: share of impostor attacks accepted by the *matcher* (used for
  face-swap: does the swapped face match the victim's reference?).

Every rate comes with a 95 % Wilson score interval — with a few hundred
presentations per species, an "0 %" is really "below ~1 %".
"""

from __future__ import annotations

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(mask: np.ndarray) -> dict:
    k, n = int(np.sum(mask)), int(np.size(mask))
    lo, hi = wilson(k, n)
    return {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci95": [lo, hi]}


def det_curve(bona_scores: np.ndarray, attack_scores: np.ndarray, n: int = 400) -> dict:
    """Scores are attack probabilities (higher = more likely attack)."""
    thr = np.unique(np.quantile(np.concatenate([bona_scores, attack_scores]), np.linspace(0, 1, n)))
    apcer = np.array([(attack_scores < t).mean() for t in thr])
    bpcer = np.array([(bona_scores >= t).mean() for t in thr])
    return {"thresholds": thr, "apcer": apcer, "bpcer": bpcer}


def eer(bona_scores: np.ndarray, attack_scores: np.ndarray) -> float:
    c = det_curve(bona_scores, attack_scores, 1000)
    i = int(np.argmin(np.abs(c["apcer"] - c["bpcer"])))
    return float((c["apcer"][i] + c["bpcer"][i]) / 2)


def bpcer_at_apcer(bona_scores: np.ndarray, attack_scores: np.ndarray, target_apcer: float) -> dict:
    """Threshold = (target_apcer)-quantile of attack scores; report BPCER there."""
    t = float(np.quantile(attack_scores, target_apcer))
    return {"threshold": t, "bpcer": float((bona_scores >= t).mean()), "apcer": float((attack_scores < t).mean())}


def threshold_for_bpcer(bona_scores: np.ndarray, target_bpcer: float) -> float:
    """Smallest threshold whose BPCER on these bona fide scores is <= target."""
    return float(np.quantile(bona_scores, 1 - target_bpcer, method="higher")) + 1e-9


def cumulative_success(p_single: float, attempts: int) -> float:
    """P(at least one success in n independent attempts) — retry/probing risk."""
    return 1 - (1 - p_single) ** attempts
