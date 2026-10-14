# Deliverable 4 — Recommendations: thresholds, monitoring, manual review and fallback rules

This is the operating manual for the defended flow. Each recommendation is tied to a measured result
([Deliverable 2](02-adversarial-test-report.md)) or to a vulnerability
([Deliverable 1](01-vulnerability-assessment.md)). Machine-readable versions ship with the code:
`configs/policy.yaml` (thresholds) and `monitoring/prometheus/alerts.yml` (alerts).

---

## 1. Thresholds

### 1.1 How to set them

1. **Budget BPCER per layer, not per system.** Each layer gets a share of the honest-user friction
   budget, and its thresholds are the matching **bona fide quantiles** on a validation set. Attack
   data is *not* used to set thresholds: attack mixes change weekly, honest users do not.
2. **Two thresholds per layer.** *Review* sends the case to a human or a stronger channel. *Reject*
   is for near-certain evidence. Only hard integrity findings (driver signatures, emulator, hooks,
   failed attestation) and above-reject scores auto-reject.
3. **Recalibrate on every model promotion and every capture-client release.** Thresholds are
   per-model-version artefacts. `src.evaluation.run_eval` regenerates them, and `policy.version` is
   stored with every decision.
4. **Per-platform thresholds after go-live.** Web and native have different camera pipelines
   (challenge latency, noise floors). Start global, split once each platform has ≥ 2,000 labelled
   bona fide sessions.

### 1.2 Recommended starting values (policy `cal-20261014-val1083`)

| Layer | Review → action | Reject | BPCER budget on val bona fide (review / reject) |
|---|---|---|---|
| Vendor liveness (FaceTec) | – | vendor decision + **all** security checks | non-negotiable |
| PAD texture (second opinion on the audit image) | 0.9477 → manual review | 0.9704 | 1.0 % / 0.5 % |
| Capture integrity | 0.9119 → **step-up** | 0.9874 | 2.0 % / 0.5 % |
| Deepfake (CNN + forensics mean) | 0.8427 → **manual review** | 0.9090 | 1.5 % / 0.5 % |
| Delayed challenge reflection (rule) | → **step-up** | – | 0 % on val (peak after the camera window, corr ≥ 0.4, +0.15 over in-window) |
| Face match (SFace cosine) | – | < 0.363 | OpenCV Zoo operating point; identity only, never liveness |

Measured on the held-out test split with these values: **BPCER 2.40% · friction 6.8% ·
mean APCER 6.4%** ([Deliverable 2](02-adversarial-test-report.md)).

> These numbers are calibrated on the simulated campaign. **On the client's flow, run a 2-week
> shadow mode first** (§5): log scores without acting on them, then set thresholds from real
> bona fide quantiles.

### 1.3 Integrity rules (deterministic)

| Code | Severity | Action | Note |
|---|---|---|---|
| `VIRTUAL_CAMERA` (OBS, ManyCam, XSplit, e2eSoft, DroidCam, v4l2loopback…) | hard | REJECT | Keep the pattern list in config. Review new labels weekly from metrics |
| `VIRTUAL_CAMERA_BENIGN` (NVIDIA Broadcast, mmhmm, Camo) | soft | STEP_UP | Honest users. Ask them to switch camera rather than rejecting |
| `BROWSER_AUTOMATION` (`navigator.webdriver`) | hard | REJECT | |
| `EMULATOR`, `HOOKING_FRAMEWORK` | hard | REJECT | Mobile: verify on the **server** from the attestation payload, not client flags |
| `ATTESTATION_FAILED` (Play Integrity / App Attest) | hard | REJECT | Require `MEETS_DEVICE_INTEGRITY` (or stronger) for onboarding |
| `ATTESTATION_MISSING`, `SENSORS_MISSING`, `RESOLUTION_MISMATCH` | soft | STEP_UP | ~3 % of honest mobile devices cannot attest |

---

## 2. Manual review and step-up rules

### 2.1 Routing

| Trigger | Route | SLA | Reviewer sees |
|---|---|---|---|
| Deepfake score in review band | **Manual review** (fraud analyst) | 4 h (onboarding), 15 min (high-value login) | Audit image + 3 scored frames, CNN vs forensics sub-scores and feature values, FaceTec result |
| PAD score in review band | Manual review | 4 h | Audit image, PAD score, FaceTec result |
| Capture-integrity review band, or any *soft* finding | **Step-up**: retry in native app with attestation, or NFC chip read, or scheduled video call | user-driven | Reason code shown to the analyst only |
| Vendor unavailable | Step-up (retry later) | – | Never approve on vendor error (fail closed) |
| ≥ 2 step-ups for one identity in 24 h | Manual review | 24 h | Full session history for the identity and device |
| Approved session, then a chargeback / ATO report on the account within 30 days | Retro-review + label | 72 h | Feeds the training set |

### 2.2 Reviewer decision rules (written guidance)

1. **Never approve on face similarity alone.** A face swap *is* the victim's face. In Deliverable 2,
   SFace accepted the victim in 91.1% [87.7%–93.6%] of real-time swaps.
2. Treat these as **confirmed attack**: a hard integrity finding on any session for the same
   device; `CHALLENGE_REFLECTION_DELAYED` together with deepfake score > review; identical
   background/lighting across different identities.
3. Signs that point to **genuine**: low deepfake score from *both* sub-models, the challenge
   reflected at camera latency, a native device with valid attestation.
4. Unsure → **step-up**, not reject. The cost of one extra NFC read is far below the cost of a
   false reject plus support contact.
5. Every review outcome (`confirmed_attack` / `confirmed_genuine`) is recorded through
   `POST /v1/liveness/reviews/{id}`. Those labels become the next training and calibration set.

### 2.3 Fallbacks: always stronger, never weaker

* ❌ "Upload a selfie instead", "skip liveness and verify by email", or "try another browser" with
  no extra checks. Attackers route to the weakest path (AV-14).
* ✅ Native app with attestation · NFC eMRTD chip portrait as the 1:1 reference (signed by the
  issuing state, which also closes AV-15) · live video call with a trained agent for the residue.

### 2.4 Velocity and anti-probing

| Limit | Value | Why |
|---|---|---|
| Liveness attempts per identity | 3 / 24 h | At the mean per-attempt APCER (6.4%) an attacker succeeds 18.0% of the time within 3 tries vs 96.3% within 50 |
| Attempts per device | 5 / 24 h | Blocks iterating across identities |
| Distinct identities per device | 2 / 30 days | Account farming (AV-16) |
| Client-side error detail | generic ("we couldn't verify you") | Reason codes only in the back-office. Never tell the attacker which layer fired |
| Session TTL | 120 s, one-time | Limits relaying the challenge to a remote rig (AV-09, AV-12) |

---

## 3. Monitoring

### 3.1 Attack signals

| Alert (Prometheus) | Condition | Owner | First response |
|---|---|---|---|
| `InjectionToolingSurge` | > 5 % of sessions with injection signatures, 10 min | Fraud | Check new camera labels / ASNs; raise to step-up for the affected cohort |
| `ChallengeFailureSurge` | > 8 % `CHALLENGE_NOT_REFLECTED` / `…_DELAYED`, 15 min | Fraud + Mobile | Distinguish a campaign (clustered devices/IPs) from a client regression (one app build) |
| `RejectRateSpike` | reject share > 2× 7-day baseline, 30 min | Fraud | Same |
| `SessionReplayAttempt` | any replayed / forged session | Fraud | Block device, review the account |
| `DeviceFarmSuspected` | > 5 multi-identity device blocks / h | Fraud | Link analysis on devices and accounts |

### 3.2 Model and quality signals

| Alert | Condition | Owner | Response |
|---|---|---|---|
| `BonaFideFrictionHigh` | step-up + review > 12 % for 1 h (≈ 1.4× the calibrated 6.8%) | ML | Check the capture-client release, camera mix, per-platform score shift. Recalibrate |
| `DeepfakeScoreDrift` | median deepfake score > 2× 7-day median for 2 h | ML | New camera/codec in the population, or a campaign. Check review outcomes |
| `ReviewersOverturnMostCases` | > 90 % of reviews `confirmed_genuine` in 24 h | ML | Review band too wide. Raise the threshold or retrain |
| `LivenessVendorUnavailable` | > 3 vendor errors / 5 min | Platform | Flow fails closed (step-up). Page on-call |
| `VendorSecurityCheckFailures` | any replay / token / audit-trail check failure | Fraud | Investigate the integration or an attack |
| `PipelineLatencyHigh` | p95 > 1.5 s, 10 min | Platform | Scale out, check the model cache |

Dashboard: `monitoring/grafana/dashboards/faceguard.json` (decision mix, friction, findings by code,
per-layer score quantiles, session rejections, vendor checks, latency, detector model versions).

### 3.3 Weekly review (30 min)

* Top new camera labels / user agents / ASNs among rejects.
* Reviewer agreement and turnaround. Confirmed-attack rate per route.
* Per-platform and (where lawful) per-demographic BPCER from step-up outcomes.
* Retraining queue: confirmed attacks the models scored low (false accepts found by retro-review).

---

## 4. Model lifecycle (on MSP)

1. Train → register as a new **staging** version (`src/training/register.py`).
2. **Shadow**: score live traffic with both versions and compare score distributions and review
   outcomes for ≥ 7 days.
3. Recalibrate thresholds for the new version and store them as a new `policy.version`.
4. **Promote** through the MSP API. The previous version is archived and the cache invalidated. The
   pipeline picks up the new version on the next request.
5. **Rollback**: re-register the previous artifact as a new version and promote it. MSP blocks
   promoting archived versions on purpose, keeping an auditable history.

---

## 5. Engagement plan (to take this from the simulated campaign to the client's production flow)

| Week | Activity | Output |
|---|---|---|
| 1 | Flow, code and log review against the checklist in Deliverable 1 §7. P0 integration fixes (server-side verdict, all checks, one-time tokens, velocity, fallbacks) | Confirmed findings + P0 PRs |
| 2 | FaceGuard capture add-on in **shadow mode**: scores logged, no decisions | Real bona fide score distributions |
| 3 | Authorised red-team with real tooling: OBS + pre-recorded video, DeepFaceLive-class swap, rooted Android + camera hook, print/screen, replayed submissions. Same harness and metrics | Real APCER per species, before/after |
| 4 | Calibrate on real bona fide quantiles, enable review/step-up routing, alerts and dashboard, reviewer guide | Live policy v1 |
| 5–8 | Weekly review loop, retraining from review labels, per-platform thresholds | Policy v2 + model v1.1 |
| Later | Independent CEN/TS 18099-aligned injection test, iBeta for any PAD changes | External attestation |
