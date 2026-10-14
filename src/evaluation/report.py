"""Charts + rendered documents from ``reports/results.json``.

    python -m src.evaluation.report

Every number in docs/02..04 comes from results.json through ``<<key>>``
placeholders in docs/templates/*.md — nothing is typed by hand, so the
documents cannot drift from the measured results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.attacks.catalog import ALL_ATTACKS, SPECIES  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402

REPORTS = Path("reports")
ASSETS = Path("docs/assets")
TEMPLATES = Path("docs/templates")

# Reference palette (dataviz skill, light mode) — validated: adjacent CVD dE >= 9.1
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
STATUS = {"APPROVE": "#0ca30c", "STEP_UP": "#fab219", "MANUAL_REVIEW": "#ec835a", "REJECT": "#d03b3b"}

LABEL = {
    "bona_fide": "Bona fide",
    "print_photo": "Print photo",
    "screen_replay": "Screen replay",
    "vcam_prerecorded": "Virtual cam · pre-recorded",
    "vcam_prerecorded_stealth": "Virtual cam · stealth",
    "vcam_offline_faceswap": "Virtual cam · offline face swap",
    "vcam_realtime_faceswap": "Virtual cam · real-time face swap",
    "vcam_synthetic_gan": "Virtual cam · synthetic (GAN)",
    "emulator_injection": "Emulator / hooked device",
    "challenge_aware_injection": "Challenge-aware replay (white-box)",
    "challenge_aware_faceswap": "Challenge-aware face swap (white-box)",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": INK2,
            "font.size": 10,
            "font.family": ["Inter", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_apcer(res: dict) -> None:
    sp = ALL_ATTACKS[::-1]
    b0 = [res["variants"]["B0"]["species"][s]["apcer"]["rate"] for s in sp]
    b3 = [res["variants"]["B3"]["species"][s]["apcer"]["rate"] for s in sp]
    y = np.arange(len(sp))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    h = 0.38
    ax.barh(y + h / 2, b0, h, color=S2, edgecolor=SURFACE, linewidth=2, label="Baseline B0 · vendor liveness only")
    ax.barh(y - h / 2, b3, h, color=S1, edgecolor=SURFACE, linewidth=2, label="FaceGuard B3 · all layers")
    for yi, v in zip(y, b0, strict=True):
        ax.text(v + 0.01, yi + h / 2, pct(v), va="center", fontsize=8.5, color=INK2)
    for yi, v in zip(y, b3, strict=True):
        ax.text(v + 0.01, yi - h / 2, pct(v), va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(y, [LABEL[s] for s in sp])
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("APCER — share of attacks auto-approved (lower is better)")
    ax.set_title("Attacks that get through, per attack species (held-out test faces)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "apcer_by_species.png", dpi=160)
    plt.close(fig)


def chart_ladder(res: dict) -> None:
    vs = list(res["variants"])
    names = ["B0\nvendor", "B1\n+ rules", "B2\n+ capture", "B3\n+ deepfake"]
    mean_ap = [res["variants"][v]["mean_apcer"] for v in vs]
    max_ap = [res["variants"][v]["max_apcer"] for v in vs]
    friction = [res["variants"][v]["bona_fide"]["friction"]["rate"] for v in vs]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, vals, title in (
        (axes[0], mean_ap, f"Mean APCER across {len(ALL_ATTACKS)} species"),
        (axes[1], max_ap, "Worst-species APCER"),
        (axes[2], friction, "Bona fide friction (not auto-approved)"),
    ):
        ax.bar(range(4), vals, 0.6, color=S1, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, pct(v), ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(4), names, fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.22 + 1e-3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        "Defence-in-depth ladder: each layer added on top of the previous",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "defence_ladder.png", dpi=160)
    plt.close(fig)


def chart_decisions(res: dict) -> None:
    rows = ["bona_fide", *ALL_ATTACKS][::-1]
    v = res["variants"]["B3"]
    data = {}
    for r in rows:
        src = v["bona_fide"] if r == "bona_fide" else v["species"][r]
        a = src["approve"]["rate"] if r == "bona_fide" else src["apcer"]["rate"]
        data[r] = [a, src["step_up"]["rate"], src["manual_review"]["rate"], src["reject"]["rate"]]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    left = np.zeros(len(rows))
    for k, name in enumerate(["APPROVE", "STEP_UP", "MANUAL_REVIEW", "REJECT"]):
        vals = np.array([data[r][k] for r in rows])
        ax.barh(
            range(len(rows)),
            vals,
            0.62,
            left=left,
            color=STATUS[name],
            edgecolor=SURFACE,
            linewidth=2,
            label=name.replace("_", " ").title(),
        )
        for i, (lft, val) in enumerate(zip(left, vals, strict=True)):
            if val >= 0.08:
                ax.text(lft + val / 2, i, pct(val, 0), ha="center", va="center", fontsize=8, color=INK)
        left += vals
    ax.set_yticks(range(len(rows)), [LABEL[r] for r in rows])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.grid(visible=False)
    ax.set_title("FaceGuard decisions per class (B3, test split)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.06), fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "decision_mix.png", dpi=160)
    plt.close(fig)


def chart_det(scores: pd.DataFrame, res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    bf = scores[scores.species == "bona_fide"]
    for (layer, col), color, name in zip(
        (("pad", "s_pad"), ("ci", "s_ci"), ("df", "s_df")),
        (S1, S2, S3),
        (
            "PAD texture vs print / replay",
            f"Capture integrity vs all {len(ALL_ATTACKS)} species",
            "Deepfake layer vs face swaps",
        ),
        strict=True,
    ):
        att = scores[scores.species.isin(res["layers"][layer]["attack_species"])][col].to_numpy()
        c = M.det_curve(bf[col].to_numpy(), att, 600)
        ap, bp = np.clip(c["apcer"], 1e-4, 1), np.clip(c["bpcer"], 1e-4, 1)
        ax.plot(ap, bp, color=color, linewidth=2, label=f"{name} (EER {pct(res['layers'][layer]['eer'])})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("APCER")
    ax.set_ylabel("BPCER")
    ax.set_title("DET curves per layer (test split)")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(ASSETS / "det_curves.png", dpi=160)
    plt.close(fig)


def chart_challenge(profiles: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lags = np.arange(next(iter(profiles.values())).shape[1])
    for (sp, prof), color in zip(profiles.items(), (S1, S2, S3, S4), strict=False):
        m = prof.mean(0)
        name = LABEL[sp] + (" (4–10 frame latency, averaged)" if sp == "vcam_realtime_faceswap" else "")
        name += " (2–5 frame lag, averaged)" if sp == "challenge_aware_injection" else ""
        ax.plot(lags * 33.3, m, color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.axvspan(-5, 2.5 * 33.3, color=GRID, alpha=0.6, linewidth=0)
    ax.text(3, 0.93, "expected camera\nlatency window", fontsize=8.5, color=INK2, va="top")
    ax.set_xlabel("lag between displayed challenge and observed reflection (ms)")
    ax.set_ylabel("correlation (face − background)")
    ax.set_ylim(-0.25, 1)
    ax.set_title("Nonce-bound photometric challenge: who reflects this session's colours, and when")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "challenge_lag.png", dpi=160)
    plt.close(fig)


def chart_retry(res: dict) -> None:
    n = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for v, color, name in (("B0", S2, "Baseline B0"), ("B3", S1, "FaceGuard B3")):
        p = res["variants"][v]["max_apcer"]
        ax.plot(n, 1 - (1 - p) ** n, color=color, linewidth=2, label=f"{name}: worst species, {pct(p)} per attempt")
    ax.axvline(3, color=AXIS, linewidth=1)
    ax.text(3.6, 0.04, "policy: 3 attempts / identity / 24 h", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("attempts by one attacker")
    ax.set_ylabel("P(at least one success)")
    ax.set_title("Why velocity limits matter: cumulative success under retries")
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()
    fig.savefig(ASSETS / "retry_risk.png", dpi=160)
    plt.close(fig)


def challenge_profiles(n: int = 60) -> dict[str, np.ndarray]:
    from src.attacks.capture import simulate
    from src.liveness.challenge import random_challenge
    from src.liveness.signals.capture_integrity import lag_profile
    from src.training.data import load_faces

    d = load_faces()
    idx = np.nonzero(d["split"] == "test")[0]
    rng = np.random.default_rng(99)
    out = {}
    for sp in ("bona_fide", "vcam_prerecorded_stealth", "vcam_realtime_faceswap", "challenge_aware_injection"):
        rows = []
        for _ in range(n):
            i, j = rng.choice(idx, 2, replace=False)
            ch = random_challenge(rng)
            s = simulate(sp, d["crops"][i], d["lmk"][i], d["crops"][j], d["lmk"][j], ch, rng)
            rows.append(lag_profile(s.frames, s.lmk, ch.per_frame()))
        out[sp] = np.array(rows)
    return out


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def _ci(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{pct(r['rate'])} [{pct(lo)}–{pct(hi)}]"


def verdict(rate: float) -> str:
    if rate <= 0.02:
        return "✅ PASS"
    if rate <= 0.10:
        return "⚠️ PARTIAL"
    return "❌ FAIL"


def build_context(res: dict, junit: dict | None) -> dict[str, str]:
    v = res["variants"]
    ctx: dict[str, str] = {}
    c = res["campaign"]
    ctx["n_bf"] = str(c["bona_fide_sessions"])
    ctx["n_bf_subjects"] = str(c["bona_fide_subjects"])
    ctx["n_attack"] = str(c["attack_sessions"])
    ctx["n_per_species"] = str(next(iter(c["per_species"].values())))
    ctx["n_species"] = str(len(ALL_ATTACKS))
    ctx["generated_at"] = res.get("generated_at", "")
    for k in v:
        ctx[f"{k}_mean_apcer"] = pct(v[k]["mean_apcer"])
        ctx[f"{k}_max_apcer"] = pct(v[k]["max_apcer"])
        ctx[f"{k}_max_species"] = LABEL[v[k]["max_apcer_species"]]
        ctx[f"{k}_bpcer"] = pct(v[k]["bona_fide"]["bpcer_reject"]["rate"], 2)
        ctx[f"{k}_friction"] = pct(v[k]["bona_fide"]["friction"]["rate"])
        ctx[f"{k}_stepup"] = pct(v[k]["bona_fide"]["step_up"]["rate"])
        ctx[f"{k}_review"] = pct(v[k]["bona_fide"]["manual_review"]["rate"])
        for s in ALL_ATTACKS:
            ctx[f"{k}_{s}"] = pct(v[k]["species"][s]["apcer"]["rate"])

    # species catalogue table
    rows = [
        "| Species | Family | Instrument (PAI / IAI) | Delivery (method) | Seen in training | Attack potential |",
        "|---|---|---|---|---|---|",
    ]
    for s in ALL_ATTACKS:
        sp = SPECIES[s]
        rows.append(
            f"| {LABEL[s]} | {sp.family} | {sp.instrument} | {sp.method} | {'yes' if sp.seen_in_training else '**no** (generalisation)'} | {sp.attack_potential} |"
        )
    ctx["species_table"] = "\n".join(rows)

    # main APCER ladder table
    rows = [
        "| Attack species | B0 vendor only | B1 + device rules | B2 + capture integrity | **B3 FaceGuard** | 95% CI (B3) | Verdict (B3) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ALL_ATTACKS:
        r3 = v["B3"]["species"][s]["apcer"]
        rows.append(
            f"| {LABEL[s]}{'' if SPECIES[s].seen_in_training else ' †'} | {pct(v['B0']['species'][s]['apcer']['rate'])} | "
            f"{pct(v['B1']['species'][s]['apcer']['rate'])} | {pct(v['B2']['species'][s]['apcer']['rate'])} | "
            f"**{pct(r3['rate'])}** | {pct(r3['ci95'][0])}–{pct(r3['ci95'][1])} | {verdict(r3['rate'])} |"
        )
    rows.append(
        f"| **Mean / worst** | {pct(v['B0']['mean_apcer'])} / {pct(v['B0']['max_apcer'])} | {pct(v['B1']['mean_apcer'])} / {pct(v['B1']['max_apcer'])} | "
        f"{pct(v['B2']['mean_apcer'])} / {pct(v['B2']['max_apcer'])} | **{pct(v['B3']['mean_apcer'])} / {pct(v['B3']['max_apcer'])}** | | |"
    )
    ctx["apcer_table"] = "\n".join(rows)

    rows = [
        "| Variant | Approved | Step-up | Manual review | Rejected (BPCER) | Friction (not auto-approved) |",
        "|---|---|---|---|---|---|",
    ]
    for k, o in v.items():
        b = o["bona_fide"]
        rows.append(
            f"| {k} — {o['description']} | {pct(b['approve']['rate'])} | {pct(b['step_up']['rate'])} | {pct(b['manual_review']['rate'])} | {_ci(b['bpcer_reject'])} | {pct(b['friction']['rate'])} |"
        )
    ctx["bona_fide_table"] = "\n".join(rows)

    rows = ["| Attack species | Approve (APCER) | Step-up | Manual review | Reject |", "|---|---|---|---|---|"]
    for s in ALL_ATTACKS:
        o = v["B3"]["species"][s]
        rows.append(
            f"| {LABEL[s]} | {pct(o['apcer']['rate'])} | {pct(o['step_up']['rate'])} | {pct(o['manual_review']['rate'])} | {pct(o['reject']['rate'])} |"
        )
    ctx["decision_table"] = "\n".join(rows)

    names = {
        "pad": "PAD texture (frame)",
        "ci": "Capture integrity (session)",
        "df": "**Deepfake layer** (mean of the two below)",
        "df_cnn": "↳ CNN (PyTorch → ONNX)",
        "df_fx": "↳ Forensics GBM",
    }
    rows = [
        "| Layer | Evaluated against | AUC | EER | BPCER @ APCER 5% | BPCER @ APCER 1% |",
        "|---|---|---|---|---|---|",
    ]
    for k, o in res["layers"].items():
        rows.append(
            f"| {names[k]} | {', '.join(LABEL[s] for s in o['attack_species']) if k != 'ci' else f'all {len(ALL_ATTACKS)} species'} | {o['auc']:.3f} | {pct(o['eer'])} | "
            f"{pct(o['bpcer_at_apcer_5']['bpcer'])} | {pct(o['bpcer_at_apcer_1']['bpcer'])} |"
        )
    ctx["layer_table"] = "\n".join(rows)

    rows = [
        "| Attack species | PAD AUC | Capture-integrity AUC | Deepfake AUC (CNN / forensics) |",
        "|---|---|---|---|",
    ]
    for s in ALL_ATTACKS:
        L = res["layers"]
        rows.append(
            f"| {LABEL[s]}{'' if SPECIES[s].seen_in_training else ' †'} | {L['pad']['per_species_auc'][s]:.3f} | {L['ci']['per_species_auc'][s]:.3f} | "
            f"{L['df']['per_species_auc'][s]:.3f} ({L['df_cnn']['per_species_auc'][s]:.3f} / {L['df_fx']['per_species_auc'][s]:.3f}) |"
        )
    ctx["per_species_auc_table"] = "\n".join(rows)

    from src.liveness.fusion import Policy

    ctx["ttl"] = str(Policy().session_ttl_seconds)
    man = Path("data/processed/faces_manifest.json")
    if man.exists():
        m = json.loads(man.read_text())
        ctx["corpus_seen"] = f"{m['seen']:,}"
        ctx["corpus_kept"] = f"{m['kept']:,}"
        ctx["corpus_dupes"] = str(m["near_duplicates_removed"])
    lat = REPORTS / "latency.json"
    if lat.exists():
        lj = json.loads(lat.read_text())
        ctx["latency_p50"] = f"{lj['p50_ms']:.0f} ms"
        ctx["latency_p95"] = f"{lj['p95_ms']:.0f} ms"
    pol = res["policy"]
    ctx["policy_version"] = pol["version"]
    ctx["t_vendor"] = f"{pol['vendor_pad_threshold']:.4f}"
    ctx["t_ci_review"] = f"{pol['capture_integrity']['review']:.4f}"
    ctx["t_ci_reject"] = f"{pol['capture_integrity']['reject']:.4f}"
    ctx["t_df_review"] = f"{pol['deepfake']['review']:.4f}"
    ctx["t_df_reject"] = f"{pol['deepfake']['reject']:.4f}"
    ctx["t_pad_reject"] = f"{pol['pad']['reject']:.4f}"

    rr = res["retry_risk"]
    ctx["retry_p"] = pct(rr["per_attempt_apcer"])
    ctx["retry_species"] = LABEL[rr["species"]]
    for n, val in rr["cumulative"].items():
        ctx[f"retry_{n}"] = pct(val)
    pm = v["B3"]["mean_apcer"]
    ctx["retry_mean_3"] = pct(M.cumulative_success(pm, 3))
    ctx["retry_mean_50"] = pct(M.cumulative_success(pm, 50))
    p0 = v["B0"]["max_apcer"]
    for n in (3, 10, 50):
        ctx[f"retry_b0_{n}"] = pct(M.cumulative_success(p0, n))

    mt = res.get("matching", {})
    for s in ("vcam_offline_faceswap", "vcam_realtime_faceswap", "challenge_aware_faceswap"):
        if s in mt:
            ctx[f"iapmr_{s}"] = _ci(mt[s]["iapmr"])
            ctx[f"iapmr_n_{s}"] = str(mt[s]["n"])
            ctx[f"attacker_match_{s}"] = pct(mt[s]["matches_attacker"]["rate"])
    if "zero_effort_impostor" in mt:
        ctx["fmr_zero_effort"] = _ci(mt["zero_effort_impostor"]["fmr"])
        ctx["fmr_n"] = str(mt["zero_effort_impostor"]["n"])

    if "ablation_without_delay_rule" in res:
        r1 = res["ablation_without_delay_rule"]
        r2 = v["B3"]
        rows = [
            "| B3 FaceGuard, same test campaign | Run 1 policy (capture-integrity model only) | Run 2 policy (+ delayed-reflection rule) |",
            "|---|---|---|",
        ]
        for s in ("challenge_aware_injection", "challenge_aware_faceswap", "vcam_realtime_faceswap"):
            rows.append(
                f"| APCER · {LABEL[s]} | {pct(r1['species'][s]['apcer']['rate'])} | **{pct(r2['species'][s]['apcer']['rate'])}** |"
            )
        rows.append(
            f"| Mean APCER ({len(ALL_ATTACKS)} species) | {pct(r1['mean_apcer'])} | **{pct(r2['mean_apcer'])}** |"
        )
        rows.append(f"| Worst-species APCER | {pct(r1['max_apcer'])} | **{pct(r2['max_apcer'])}** |")
        rows.append(
            f"| Bona fide rejected / friction | {pct(r1['bona_fide']['bpcer_reject']['rate'], 2)} / {pct(r1['bona_fide']['friction']['rate'])} | "
            f"{pct(r2['bona_fide']['bpcer_reject']['rate'], 2)} / {pct(r2['bona_fide']['friction']['rate'])} |"
        )
        ctx["iteration_table"] = "\n".join(rows)
        ctx["it1_ca"] = pct(r1["species"]["challenge_aware_injection"]["apcer"]["rate"])
        ctx["it1_rt"] = pct(r1["species"]["vcam_realtime_faceswap"]["apcer"]["rate"])

    if junit:
        rows = ["| Logic attack (vulnerability) | Automated test | Result |", "|---|---|---|"]
        for desc, test, ok in junit["rows"]:
            rows.append(f"| {desc} | `{test}` | {'✅ blocked' if ok else '❌ NOT blocked'} |")
        ctx["logic_table"] = "\n".join(rows)
        ctx["logic_summary"] = f"{junit['passed']}/{junit['total']}"
        ctx["suite_summary"] = f"{junit['suite_passed']}/{junit['suite_total']}"
    return ctx


LOGIC_TESTS = [
    ("AV-12 API replay of a captured submission", "test_session_replay_is_refused"),
    ("Forged / guessed session token", "test_forged_session_token_is_refused"),
    (
        "AV-10 Client-side verdict tampering (`livenessPassed: true`) + virtual camera",
        "test_virtual_camera_injection_rejected_even_with_client_claims",
    ),
    ("AV-11 Only `success` checked — replay check failing", "test_facetec_success_flag_alone_is_not_enough"),
    ("AV-11 Vendor response missing security checks", "test_facetec_missing_checks_fail_closed"),
    ("Vendor outage must not approve (fail closed)", "test_facetec_outage_fails_closed"),
    ("Vendor session token reuse / payload replay", "test_simulated_vendor_enforces_one_time_token_and_replay"),
    ("AV-13 Probing: >3 attempts per identity / 24 h", "test_identity_velocity_limit"),
    ("AV-16 Account farming: one device, many identities", "test_device_used_by_many_identities_is_blocked"),
    (
        "AV-04 Stealth injection (renamed driver) caught by the nonce challenge",
        "test_stealth_injection_caught_by_challenge",
    ),
    ("Malformed capture (frame count ≠ challenge)", "test_wrong_frame_count_is_422"),
]


def parse_junit(path: Path) -> dict | None:
    if not path.exists():
        return None
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    status = {}
    for tc in root.iter("testcase"):
        failed = any(child.tag in ("failure", "error", "skipped") for child in tc)
        status[tc.get("name")] = not failed
    rows = [(d, t, status.get(t, False)) for d, t in LOGIC_TESTS]
    return {
        "rows": rows,
        "passed": sum(ok for *_, ok in rows),
        "total": len(rows),
        "suite_total": len(status),
        "suite_passed": sum(status.values()),
    }


def render(ctx: dict[str, str]) -> list[str]:
    written = []
    for tpl in sorted(TEMPLATES.glob("*.md")):
        text = tpl.read_text()
        missing = set(re.findall(r"<<(\w+)>>", text)) - set(ctx)
        if missing:
            raise KeyError(f"{tpl.name}: missing values {sorted(missing)}")
        out = re.sub(r"<<(\w+)>>", lambda m: ctx[m.group(1)], text)
        # "X.root.md" renders to the repository root (README), everything else to docs/
        dst = Path(tpl.name.replace(".root.md", ".md")) if tpl.name.endswith(".root.md") else Path("docs") / tpl.name
        dst.write_text(out)
        written.append(str(dst))
    return written


def main() -> None:
    _style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    res = json.loads((REPORTS / "results.json").read_text())
    scores = pd.read_csv(REPORTS / "scores_test.csv", keep_default_na=False)
    chart_apcer(res)
    chart_ladder(res)
    chart_decisions(res)
    chart_det(scores, res)
    chart_retry(res)
    chart_challenge(challenge_profiles())
    ctx = build_context(res, parse_junit(REPORTS / "junit.xml"))
    (REPORTS / "doc_context.json").write_text(json.dumps(ctx, indent=1))
    print("charts ->", ASSETS, "| docs ->", render(ctx))


if __name__ == "__main__":
    main()
