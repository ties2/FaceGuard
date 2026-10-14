# Upwork portfolio item — copy/paste

## Project title (≤ 40 characters)
FaceGuard: Deepfake & Injection Defense

Alternatives: `Deepfake & Injection Defense for FaceTec` (40) · `Liveness Anti-Spoofing & Deepfake Shield` (40)

## Your role (≤ 100 characters)
ML Security Engineer: threat modelling, computer vision / deep learning, MLOps (solo project)

## Project description (290 characters; Upwork allows 600, ≤ 300 shows in full on the card)
Defense layer for a FaceTec-based identity check against deepfakes, face swaps and virtual-camera/emulator injection. Vulnerability assessment, red-team campaign (10 attack types, ISO 30107-3 metrics), AI layers (PyTorch→ONNX, OpenCV) and an operating policy. Attack pass rate 87.8% → 6.4%.

## Skills and deliverables
Computer Vision · Deep Learning · PyTorch · OpenCV · Biometrics · Face Recognition · Liveness Detection · Deepfake Detection · Python · FastAPI · MLOps

## Content (upload in this order, each with its caption)
**01-cover.png** (154 chars)

FaceGuard: defence layer around FaceTec 3D liveness. Mean attack pass rate (APCER) 87.8% → 6.4% on a held-out test campaign; honest users rejected: 2.40%.

**02-architecture.png** (250 chars)

Architecture: server-issued nonce challenge + one-time tokens, server-to-server FaceTec adapter, device integrity rules, capture-integrity model, PAD texture model, deepfake CNN + forensics, calibrated fusion into approve / step-up / review / reject.

**03-results.png** (182 chars)

Measured improvement per attack species (APCER). Vendor-only liveness vs all FaceGuard layers, held-out faces, 95% CIs in the report. Residual white-box risk is reported, not hidden.

**04-challenge.png** (204 chars)

Core anti-injection idea: the screen flashes a colour sequence derived from a per-session nonce. Live faces reflect it at camera latency; recordings can't contain it; real-time face swaps reflect it late.

**05-deliverables.png** (199 chars)

The four deliverables from the brief: vulnerability assessment, adversarial test report with pass/fail rates, mitigation layer with APCER/BPCER improvement, thresholds/monitoring/manual-review rules.

**06-demo-injection.png** (183 chars)

Live demo: a pre-recorded video injected through the browser's camera API, automation flag hidden. Device checks see nothing; the nonce challenge rejects it (CHALLENGE_NOT_REFLECTED).

**FaceGuard-Case-Study.pdf**: the four deliverables in one document

**Link**: your GitHub repository URL (push the `FaceGuard` folder)
