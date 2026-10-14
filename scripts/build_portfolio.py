"""Build the Upwork portfolio pack: 1600x1200 slides (PNG) + a case-study PDF.

    python scripts/build_portfolio.py        # after `make eval report`

Numbers come from reports/doc_context.json (same source as the docs).
Rendering: HTML/CSS -> headless Chromium (Playwright); Mermaid via mmdc.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "portfolio"
CTX = json.loads((ROOT / "reports/doc_context.json").read_text())
RES = json.loads((ROOT / "reports/results.json").read_text())
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{width:1600px;height:1200px;background:#0d1117;color:#e6edf3;font-family:Inter,system-ui,sans-serif;overflow:hidden}
.wrap{padding:84px 96px;height:100%;display:flex;flex-direction:column}
.kicker{font-size:22px;letter-spacing:.18em;text-transform:uppercase;color:#7d8590;font-weight:600}
h1{font-size:96px;line-height:1;font-weight:800;letter-spacing:-.02em;margin-top:18px}
h1 em{font-style:normal;color:#58a6ff}
h2{font-size:56px;line-height:1.08;font-weight:750;letter-spacing:-.015em;margin-top:14px}
.sub{font-size:32px;line-height:1.35;color:#c9d1d9;margin-top:22px;max-width:1300px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin-top:auto}
.tile{background:#161b22;border:1px solid #30363d;border-radius:18px;padding:26px 26px 24px}
.tile .v{font-size:54px;font-weight:800;letter-spacing:-.02em}
.tile .v small{font-size:26px;color:#7d8590;font-weight:600}
.tile .l{font-size:20px;color:#9da7b3;margin-top:8px;line-height:1.3}
.good{color:#3fb950}.bad{color:#f0883e}.blue{color:#58a6ff}
.tags{display:flex;flex-wrap:wrap;gap:12px;margin-top:34px}
.tag{font-size:20px;padding:9px 16px;border-radius:999px;border:1px solid #30363d;color:#c9d1d9;background:#11161d}
.foot{display:flex;justify-content:space-between;margin-top:34px;font-size:20px;color:#7d8590}
.img{margin-top:30px;flex:1;display:flex;align-items:center;justify-content:center;background:#fcfcfb;border-radius:18px;padding:18px}
.img img{max-width:100%;max-height:100%}
.cols{display:grid;grid-template-columns:repeat(2,1fr);gap:22px;margin-top:34px;flex:1}
.card{background:#161b22;border:1px solid #30363d;border-radius:18px;padding:30px}
.card .n{font-size:18px;color:#58a6ff;font-weight:700;letter-spacing:.12em}
.card h3{font-size:40px;margin:10px 0 18px;font-weight:750}
.card li{font-size:24px;color:#c9d1d9;line-height:1.5;margin:0 0 8px 26px}
"""


def page(body: str, extra_css: str = "") -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{BASE_CSS}{extra_css}</style></head><body>{body}</body></html>"


def img64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def shoot(html: str, name: str, browser) -> None:
    pg = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
    pg.set_content(html, wait_until="load")
    pg.screenshot(path=str(OUT / name), full_page=False)
    pg.close()


def cover() -> str:
    c = CTX
    tiles_data = [
        (c["B3_mean_apcer"], c["B0_mean_apcer"], f"mean attack pass rate (APCER), {c['n_species']} attack species"),
        (
            c["B3_vcam_prerecorded_stealth"],
            c["B0_vcam_prerecorded_stealth"],
            "stealth virtual-camera injection approved",
        ),
        (c["B3_vcam_realtime_faceswap"], c["B0_vcam_realtime_faceswap"], "real-time face swap approved"),
    ]
    tiles = "".join(
        f"<div class='tile'><div class='was'>vendor-only: <s>{was}</s></div><div class='v good'>{now}</div><div class='l'>{label}</div></div>"
        for now, was, label in tiles_data
    )
    tiles += (
        f"<div class='tile'><div class='was'>honest users</div><div class='v blue'>{c['B3_bpcer']}</div>"
        f"<div class='l'>rejected (BPCER) · {c['B3_friction']} stepped-up or reviewed</div></div>"
    )
    feats = [
        (
            "Nonce photometric challenge",
            "The screen flashes a per-session colour code. Live faces reflect it on time; injected video can't.",
        ),
        (
            "Deepfake CNN + forensics",
            "PyTorch → ONNX, trained without a single deepfake (self-blended images). Catches unseen face swaps.",
        ),
        (
            "Calibrated fusion & policy",
            "Approve · step-up · manual review · reject, with reason codes, alerts and a review queue.",
        ),
    ]
    feat_html = "".join(f"<div class='feat'><b>{t}</b><span>{d}</span></div>" for t, d in feats)
    tags = "".join(
        f"<span class='tag'>{t}</span>"
        for t in [
            "FaceTec integration",
            "Injection detection",
            "Deepfake detection",
            "PyTorch → ONNX",
            "OpenCV",
            "FastAPI",
            "Prometheus · Grafana",
            "ISO/IEC 30107-3 · CEN/TS 18099",
        ]
    )
    css = """
    .tile .was{font-size:19px;color:#7d8590}.tile .was s{color:#f0883e;text-decoration-thickness:2px}
    .tile .v{font-size:64px;margin-top:4px}
    .feats{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:46px}
    .feat{border-left:4px solid #1f6feb;padding:4px 0 4px 20px}
    .feat b{display:block;font-size:26px;margin-bottom:8px}
    .feat span{font-size:20px;color:#9da7b3;line-height:1.4}
    """
    body = f"""<div class='wrap'>
      <div class='kicker'>Biometric security · Face liveness · Case study</div>
      <h1>Face<em>Guard</em></h1>
      <div class='sub'>Deepfake, face-swap and camera-injection defence layer for a <b>FaceTec</b>-based identity
      verification flow: vulnerability assessment, adversarial test campaign, AI mitigation layers and an operating policy.</div>
      <div class='tags'>{tags}</div>
      <div class='feats'>{feat_html}</div>
      <div class='tiles'>{tiles}</div>
      <div class='foot'><span>Held-out test split · {c["n_attack"]} attack sessions · simulated attack instruments on real public face photos</span><span>Built on my ML Model Serving Platform</span></div>
    </div>"""
    return page(body, css)


def arch() -> str:
    css = """
    .flow{display:grid;grid-template-columns:300px 1fr 300px;gap:26px;margin-top:12px;flex:1}
    .heads{display:grid;grid-template-columns:300px 1fr 300px;gap:26px;margin-top:40px;font-size:18px;letter-spacing:.14em;text-transform:uppercase;color:#58a6ff;font-weight:700}
    .box{background:#161b22;border:1px solid #30363d;border-radius:16px;padding:20px 22px}
    .box b{display:block;font-size:24px;margin-bottom:6px}
    .box span{font-size:18px;color:#9da7b3;line-height:1.35;display:block}
    .col{display:flex;flex-direction:column;gap:16px}
    .mid{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-content:start}
    .accent{border-color:#1f6feb;background:#0f1d33}
    .out .box{padding:16px 22px}
    .d{font-size:22px;font-weight:700}
    .ok{color:#3fb950}.su{color:#d29922}.mr{color:#f0883e}.rj{color:#f85149}
    .msp{margin-top:22px;background:#11161d;border:1px dashed #30363d;border-radius:16px;padding:18px 24px;font-size:20px;color:#c9d1d9}
    .msp b{color:#58a6ff}
    """
    body = """<div class='wrap'>
      <div class='kicker'>Architecture</div>
      <h2>Defence in depth around FaceTec, not a replacement</h2>
      <div class='heads'><span>Session &amp; integration</span><span>Detection layers</span><span>Outcome</span></div>
      <div class='flow'>
        <div class='col'>
          <div class='box'><b>Capture client</b><span>FaceTec SDK 3D-liveness UX + FaceGuard add-on: server-issued colour challenge, frame timestamps, device telemetry</span></div>
          <div class='box accent'><b>Session service</b><span>128-bit nonce → colour sequence · HMAC one-time token · atomic consumption · 120 s TTL · velocity & device-farm limits</span></div>
          <div class='box'><b>FaceTec adapter</b><span>Server-to-server only · every security check required · client claims ignored · fail closed</span></div>
        </div>
        <div class='mid'>
          <div class='box'><b>Device & camera integrity</b><span>Virtual-camera drivers, automation, emulators, hooks, Play Integrity / App Attest</span></div>
          <div class='box accent'><b>Capture-integrity model</b><span>Photometric challenge: does the face reflect <i>this</i> nonce, at camera latency? + timing, duplicate frames, noise floor</span></div>
          <div class='box'><b>PAD texture model</b><span>Colour LBP (YCrCb/HSV) + moiré spectrum + gamut statistics on the audit image</span></div>
          <div class='box accent'><b>Deepfake layer</b><span>CNN (PyTorch → ONNX) + forensics GBM, trained only on injection-faithful Self-Blended Images</span></div>
          <div class='box' style='grid-column:1/3'><b>Fusion policy</b><span>Per-layer review / reject thresholds calibrated on bona fide quantiles · most severe wins · reason codes · audit trail · Prometheus</span></div>
        </div>
        <div class='col out'>
          <div class='box'><div class='d ok'>APPROVE</div><span>all layers below review</span></div>
          <div class='box'><div class='d su'>STEP-UP</div><span>native app + attestation, NFC chip, video call</span></div>
          <div class='box'><div class='d mr'>MANUAL REVIEW</div><span>analyst queue → labels for retraining</span></div>
          <div class='box'><div class='d rj'>REJECT</div><span>hard evidence + alert</span></div>
        </div>
      </div>
      <div class='msp'><b>Foundation: ML Model Serving Platform</b> — every detector is a registry-versioned model (PostgreSQL), loaded through a loader registry (scikit-learn · ONNX Runtime), cached, promoted / rolled back via API and monitored.</div>
    </div>"""
    return page(body, css)


def chart_slide(kicker: str, title: str, img: Path, note: str) -> str:
    body = f"""<div class='wrap'><div class='kicker'>{kicker}</div><h2>{title}</h2>
      <div class='img'><img src='{img64(img)}'></div>
      <div class='foot'><span>{note}</span><span>FaceGuard</span></div></div>"""
    return page(body)


def deliverables() -> str:
    c = CTX
    cards = [
        (
            "01",
            "Vulnerability assessment",
            [
                "19 attack vectors · likelihood × impact",
                "Trust boundaries of a FaceTec flow",
                "Certification gap: ISO 30107-3 vs CEN/TS 18099",
                "Prioritised P0–P2 roadmap",
                "Integration flaws first: more common than SDK bypasses",
            ],
        ),
        (
            "02",
            "Adversarial test report",
            [
                f"{c['n_attack']} attack sessions, {c['n_species']} species, held-out faces",
                "APCER per species with 95 % CIs, BPCER, DET, IAPMR",
                f"{c['logic_summary']} integration attacks blocked (pytest)",
                "Red-team iteration: finding → fix → re-test",
                "Limitations stated, fully reproducible",
            ],
        ),
        (
            "03",
            "Mitigation layer",
            [
                "Nonce photometric challenge vs injection",
                "Deepfake CNN + forensics (no deepfakes in training)",
                f"Mean APCER {c['B0_mean_apcer']} → {c['B3_mean_apcer']}",
                f"Honest users rejected {c['B3_bpcer']} · {c.get('latency_p50', 'n/a')} p50 analysis",
                "Versioned models on the serving platform",
            ],
        ),
        (
            "04",
            "Recommendations",
            [
                "Calibrated review / reject thresholds",
                "Step-up & manual-review routing rules",
                "Prometheus alerts + Grafana dashboard",
                "Reviewer playbook + model lifecycle on MSP",
                "Shadow-mode rollout plan for production",
            ],
        ),
    ]
    html = "".join(
        f"<div class='card'><div class='n'>DELIVERABLE {n}</div><h3>{t}</h3><ul>{''.join(f'<li>{b}</li>' for b in bs)}</ul></div>"
        for n, t, bs in cards
    )
    body = f"<div class='wrap'><div class='kicker'>Definition of done</div><h2>Exactly the four deliverables in the brief</h2><div class='cols'>{html}</div></div>"
    return page(body)


# ---------------------------------------------------------------------------
# Case-study PDF
# ---------------------------------------------------------------------------
PDF_CSS = """
@page{size:A4;margin:16mm 15mm 18mm}
body{font-family:Inter,system-ui,sans-serif;font-size:10pt;line-height:1.5;color:#1f2328}
h1{font-size:20pt;margin:0 0 6pt;line-height:1.2;page-break-before:always}
h2{font-size:14pt;margin:16pt 0 6pt;border-bottom:1px solid #d0d7de;padding-bottom:3pt}
h3{font-size:11.5pt;margin:12pt 0 4pt}
table{border-collapse:collapse;width:100%;margin:8pt 0;font-size:8.4pt;page-break-inside:auto}
th,td{border:1px solid #d0d7de;padding:3pt 5pt;vertical-align:top;text-align:left}
th{background:#f6f8fa}
tr{page-break-inside:avoid}
code{font-family:'DejaVu Sans Mono',monospace;font-size:8.4pt;background:#f6f8fa;padding:0 2pt;border-radius:3px}
pre{background:#f6f8fa;padding:8pt;border-radius:6px;font-size:8pt;white-space:pre-wrap;page-break-inside:avoid}
pre code{background:none;padding:0}
img{max-width:100%;page-break-inside:avoid;display:block;margin:8pt auto}
blockquote{border-left:3px solid #58a6ff;margin:8pt 0;padding:2pt 10pt;color:#424a53;background:#f6f8fa}
a{color:#0969da;text-decoration:none}
.cover{height:250mm;display:flex;flex-direction:column;justify-content:center;page-break-after:always}
.cover img{margin:0}
"""


def mermaid_to_png(src: str, name: str) -> Path:
    tmp = OUT / "_build"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"{name}.mmd").write_text(src)
    (tmp / "pp.json").write_text(json.dumps({"executablePath": CHROME, "args": ["--no-sandbox"]}))
    out = tmp / f"{name}.png"
    subprocess.run(
        [
            "mmdc",
            "-p",
            str(tmp / "pp.json"),
            "-i",
            str(tmp / f"{name}.mmd"),
            "-o",
            str(out),
            "-w",
            "1400",
            "-b",
            "white",
            "-s",
            "2",
        ],
        check=True,
        capture_output=True,
    )
    return out


def doc_html(md_path: Path, idx: int) -> str:
    text = md_path.read_text()
    blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    for k, b in enumerate(blocks):
        png = mermaid_to_png(b, f"doc{idx}_{k}")
        text = text.replace(f"```mermaid\n{b}```", f"![diagram]({png})", 1)
    text = text.replace("](assets/", f"]({ROOT / 'docs/assets'}/")
    html = subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "html5"], input=text, capture_output=True, text=True, check=True
    ).stdout
    return html


def build_pdf(browser) -> None:
    docs = [
        ROOT / "docs" / n
        for n in (
            "01-vulnerability-assessment.md",
            "02-adversarial-test-report.md",
            "03-mitigation-layer.md",
            "04-recommendations.md",
        )
    ]
    parts = [f"<div class='cover'><img src='{img64(OUT / '01-cover.png')}'></div>"]
    parts += [doc_html(d, i) for i, d in enumerate(docs)]
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{PDF_CSS}</style></head><body>{''.join(parts)}</body></html>"
    tmp = OUT / "_build" / "case-study.html"
    tmp.write_text(html)
    pg = browser.new_page()
    pg.goto(tmp.as_uri(), wait_until="load")
    pg.pdf(
        path=str(OUT / "FaceGuard-Case-Study.pdf"),
        format="A4",
        print_background=True,
        display_header_footer=True,
        header_template="<span></span>",
        footer_template="<div style='font-size:7pt;color:#8b949e;width:100%;text-align:center'>FaceGuard case study · <span class='pageNumber'></span>/<span class='totalPages'></span></div>",
        margin={"top": "16mm", "bottom": "18mm", "left": "15mm", "right": "15mm"},
    )
    pg.close()


def upwork_text() -> str:
    c = CTX
    desc = (
        f"Defense layer for a FaceTec-based identity check against deepfakes, face swaps and virtual-camera/emulator "
        f"injection. Vulnerability assessment, red-team campaign ({c['n_species']} attack types, ISO 30107-3 metrics), "
        f"AI layers (PyTorch→ONNX, OpenCV) and an operating policy. Attack pass rate {c['B0_mean_apcer']} → {c['B3_mean_apcer']}."
    )
    captions = [
        (
            "01-cover.png",
            f"FaceGuard: defence layer around FaceTec 3D liveness. Mean attack pass rate (APCER) {c['B0_mean_apcer']} → {c['B3_mean_apcer']} on a held-out test campaign; honest users rejected: {c['B3_bpcer']}.",
        ),
        (
            "02-architecture.png",
            "Architecture: server-issued nonce challenge + one-time tokens, server-to-server FaceTec adapter, device integrity rules, capture-integrity model, PAD texture model, deepfake CNN + forensics, calibrated fusion into approve / step-up / review / reject.",
        ),
        (
            "03-results.png",
            "Measured improvement per attack species (APCER). Vendor-only liveness vs all FaceGuard layers, held-out faces, 95% CIs in the report. Residual white-box risk is reported, not hidden.",
        ),
        (
            "04-challenge.png",
            "Core anti-injection idea: the screen flashes a colour sequence derived from a per-session nonce. Live faces reflect it at camera latency; recordings can't contain it; real-time face swaps reflect it late.",
        ),
        (
            "05-deliverables.png",
            "The four deliverables from the brief: vulnerability assessment, adversarial test report with pass/fail rates, mitigation layer with APCER/BPCER improvement, thresholds/monitoring/manual-review rules.",
        ),
        (
            "06-demo-injection.png",
            "Live demo: a pre-recorded video injected through the browser's camera API, automation flag hidden. Device checks see nothing; the nonce challenge rejects it (CHALLENGE_NOT_REFLECTED).",
        ),
    ]
    out = [
        "# Upwork portfolio item — copy/paste",
        "",
        "## Project title (≤ 40 characters)",
        "FaceGuard: Deepfake & Injection Defense",
        "",
        "Alternatives: `Deepfake & Injection Defense for FaceTec` (40) · `Liveness Anti-Spoofing & Deepfake Shield` (40)",
        "",
        "## Your role (≤ 100 characters)",
        "ML Security Engineer: threat modelling, computer vision / deep learning, MLOps (solo project)",
        "",
        f"## Project description ({len(desc)} characters; Upwork allows 600, ≤ 300 shows in full on the card)",
        desc,
        "",
        "## Skills and deliverables",
        "Computer Vision · Deep Learning · PyTorch · OpenCV · Biometrics · Face Recognition · Liveness Detection · "
        "Deepfake Detection · Python · FastAPI · MLOps",
        "",
        "## Content (upload in this order, each with its caption)",
    ]
    for f, cap in captions:
        out += [f"**{f}** ({len(cap)} chars)", "", cap, ""]
    out += [
        "**FaceGuard-Case-Study.pdf**: the four deliverables in one document",
        "",
        "**Link**: your GitHub repository URL (push the `FaceGuard` folder)",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    assets = ROOT / "docs/assets"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        shoot(cover(), "01-cover.png", b)
        shoot(arch(), "02-architecture.png", b)
        shoot(
            chart_slide(
                "Measured improvement",
                "Attacks that get through: vendor-only vs FaceGuard",
                assets / "apcer_by_species.png",
                "APCER per attack species · held-out test split · lower is better",
            ),
            "03-results.png",
            b,
        )
        shoot(
            chart_slide(
                "Anti-injection core",
                "A nonce-bound colour challenge that a recording cannot contain",
                assets / "challenge_lag.png",
                "Live faces reflect this session's colours at camera latency; real-time swaps reflect them late",
            ),
            "04-challenge.png",
            b,
        )
        shoot(deliverables(), "05-deliverables.png", b)
        build_pdf(b)
        b.close()
    (OUT / "UPWORK.md").write_text(upwork_text())
    print("portfolio ->", sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    sys.exit(main())
