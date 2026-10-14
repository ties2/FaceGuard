# Deliverable 2 — Adversarial Test Report
## Deepfake, face-swap, injection and presentation attacks: pass/fail rates

| | |
|---|---|
| **Systems under test** | B0 vendor liveness only (stand-in) · B1 + device/telemetry rules · B2 + capture-integrity model · **B3 FaceGuard (all layers)** |
| **Campaign** | <<n_bf>> bona fide sessions (<<n_bf_subjects>> subjects) + <<n_attack>> attack sessions, <<n_species>> attack species × <<n_per_species>> |
| **Subjects** | Held-out test split. No test face was used for training or threshold calibration |
| **Metrics** | ISO/IEC 30107-3 vocabulary: APCER per species, worst-species APCER, BPCER; DET, EER, BPCER@APCER; IAPMR for face swaps; 95 % Wilson intervals |
| **Policy** | `<<policy_version>>`, calibrated on the validation split's bona fide sessions only |
| **Generated** | <<generated_at>> from `reports/results.json` (`make eval report`) |

### Headline

| | Baseline B0 | FaceGuard B3 |
|---|---|---|
| Mean APCER across <<n_species>> attack species | <<B0_mean_apcer>> | **<<B3_mean_apcer>>** |
| Worst species | <<B0_max_apcer>> (<<B0_max_species>>) | **<<B3_max_apcer>>** (<<B3_max_species>>) |
| Bona fide rejected (BPCER) | <<B0_bpcer>> | **<<B3_bpcer>>** |
| Bona fide sent to step-up / manual review | <<B0_stepup>> / <<B0_review>> | <<B3_stepup>> / <<B3_review>> |
| Logic attacks blocked (API tests) | – | **<<logic_summary>>** |

---

## 1. Test setup

### 1.1 System under test

* **Vendor stand-in (B0).** No FaceTec licence is used in this portfolio project. The vendor is
  modelled by a frame-level PAD model (colour-texture + spectrum, trained on print/replay), the way
  a presentation-attack-only liveness check behaves. Its threshold is set at 1 % BPCER on
  validation. **FaceTec's 3D liveness is expected to be much stronger than this stand-in on
  print/replay** (it is iBeta-tested). The B0 rows that matter here are the **injection** rows, where
  the camera is bypassed and any camera-side PAD sees a clean, genuine-looking face.
* **FaceGuard (B1–B3)** adds its layers *on top of* the vendor decision, through the same
  `fusion.decide()` function the API calls. The evaluated policy is the deployed policy.

### 1.2 Bona fide corpus

<<corpus_seen>> public face photos (two MIT-licensed GitHub repositories) were processed: YuNet detection →
128 × 128 context crop + 5 landmarks → near-duplicate removal (64-bit dHash, Hamming ≤ 4, <<corpus_dupes>> removed) → <<corpus_kept>>
faces → deterministic 60/20/20 split. Each bona fide *session* is simulated through the full capture
chain: the screen's challenge light reflected by the face (more than by the background), head motion,
one of two camera ISP profiles (noisy, or denoised), JPEG, natural frame jitter with 4 % dropped
frames, and realistic telemetry. 3 % of honest web users run benign virtual cameras (NVIDIA Broadcast,
mmhmm), 1 % have webcams that ignore the requested resolution, and 3 % of mobile devices have no
attestation token. These are *not* idealised users.

### 1.3 Attack catalogue (PAI species / injection instruments)

<<species_table>>

Physical 3D masks (AV-03), document attacks (AV-15) and coerced users (AV-19) cannot be meaningfully
simulated in software and are excluded. They need iBeta-style lab testing and process controls.

### 1.4 Protocol

1. `src.evaluation.harness` generates seeded sessions per split: 30 frames at 30 fps, a fresh
   10-slot colour challenge per session, telemetry and timestamps.
2. The models are trained on the *train* split (deepfake models: bona fide + SBI only).
3. `src.evaluation.run_eval` scores *val* and *test*, calibrates every threshold on val bona fide
   quantiles, then evaluates on test.
4. An attack **passes** if the decision is `APPROVE`. `STEP_UP`, `MANUAL_REVIEW` and `REJECT` all
   stop it.

---

## 2. Results — attack pass rates (APCER)

<<apcer_table>>

† not seen in training by any model. Verdict on B3: ✅ PASS ≤ 2 % · ⚠️ PARTIAL ≤ 10 % · ❌ FAIL > 10 %.
95 % Wilson intervals: with <<n_per_species>> attempts per species, "0 %" means "below ~1 %", not "never".

![APCER by species](assets/apcer_by_species.png)

### 2.1 Decision distribution under FaceGuard (B3)

<<decision_table>>

![decision mix](assets/decision_mix.png)

### 2.2 Red-team iteration: a finding and its fix

The first run exposed a gap. The capture-integrity model had only been trained on species *without*
a delayed reflection, so real-time swaps and the challenge-aware attacker (whose reflection arrives
3–10 frames late) often scored below its review threshold, and the explicit lag check existed only
as an explanation string. The fix promotes that physical check to a decision rule:
`CHALLENGE_REFLECTION_DELAYED` (soft → step-up) fires when the reflection peaks after the camera
window and correlates ≥ 0.4, more than 0.15 above the in-window peak. Both constants were sized on
the **validation** split (0 % of bona fide sessions trigger it). Both policies on the final campaign:

<<iteration_table>>

What remains is the challenge-aware attacker whose pipeline lag falls *inside* the honest camera
window (2 frames ≈ 67 ms). In this simulation that attacker is physically indistinguishable from a live
user. It is the documented residual risk, closed by native capture with attestation (Deliverable 4).
For the **face-swap** variant of that attacker, the deepfake layer is the last barrier: it cuts the
pass rate from <<B2_challenge_aware_faceswap>> (B2) to <<B3_challenge_aware_faceswap>> (B3). This is the clearest
measured case for keeping a manipulation detector *behind* the challenge.
*Disclosure: the gap was found on an earlier test run. The rule's constants never saw test data, and
the final campaign was regenerated afterwards, with a new white-box species added (challenge-aware face swap).*

---

## 3. Bona fide impact (BPCER and friction)

<<bona_fide_table>>

Friction is the price of defence. Most of it comes from *soft* findings on honest users (benign
virtual cameras, missing attestation, resolution mismatches) and from review bands. Those users are
**stepped up** (retry in the native app, NFC document read), not rejected.

---

## 4. Per-layer analysis

<<layer_table>>

Per-species separability (AUC, bona fide vs. that species):

<<per_species_auc_table>>

![DET curves](assets/det_curves.png)

Reading the layers together:

* **PAD texture** separates print better than screen replay. Single-frame texture is the weakest
  layer and is not designed to see injections.
* **Capture integrity** separates almost every species, because the nonce challenge cannot be
  pre-recorded. Its weakest species is the white-box *challenge-aware* attacker, by design.
* **Deepfake (CNN + forensics)** is trained without any deepfake. It separates face-swap species it
  never saw, but not fully synthetic GAN-style faces: whole-frame synthesis has no blending seam and
  no camera/generated noise mismatch. The capture-integrity layer covers that species instead.
  This is why the layers are fused.

### 4.1 Face matching does not stop face swaps (IAPMR)

SFace 1:1 matching (OpenCV Zoo, cosine threshold 0.363) was run between a frame of each face-swap
attack and the **victim's** reference photo:

| Species | n | IAPMR (matcher accepts the attack as the victim) | Matches the attacker instead |
|---|---|---|---|
| Virtual cam · offline face swap | <<iapmr_n_vcam_offline_faceswap>> | <<iapmr_vcam_offline_faceswap>> | <<attacker_match_vcam_offline_faceswap>> |
| Virtual cam · real-time face swap | <<iapmr_n_vcam_realtime_faceswap>> | <<iapmr_vcam_realtime_faceswap>> | <<attacker_match_vcam_realtime_faceswap>> |

Zero-effort impostor acceptance for context: <<fmr_zero_effort>> (n = <<fmr_n>>). The reference here is
the victim's source photo, so this is an **upper bound**: with an independent ID-document portrait,
IAPMR is lower. The conclusion still stands: **once a swap reaches the matcher, identity checks help
the attacker**, so liveness and deepfake detection must sit in front of matching.

---

## 5. Logic and integration attacks (automated, `pytest`)

<<logic_table>>

These are the integration flaws from the vulnerability assessment (AV-10…AV-16). They need no ML,
and they run in CI on every commit.

---

## 6. Retry / probing risk

With per-attempt success *p*, the chance an attacker succeeds at least once in *n* tries is
1 − (1 − p)ⁿ. Using the worst species' per-attempt APCER:

| Attempts | B0 (p = <<B0_max_apcer>>) | FaceGuard B3 (p = <<retry_p>>, <<retry_species>>) |
|---|---|---|
| 3 (policy limit / identity / 24 h) | <<retry_b0_3>> | <<retry_3>> |
| 10 | <<retry_b0_10>> | <<retry_10>> |
| 50 (no velocity limit) | <<retry_b0_50>> | <<retry_50>> |

Velocity limits cap the damage of moderate species (a 0.6 % per-attempt real-time swap stays at
~2 % after 3 tries), but they cannot fix a species with a high per-attempt rate. The white-box
residual is closed by channel controls (native capture + attestation), not by counting attempts.

![retry risk](assets/retry_risk.png)

---

## 7. Limitations and threats to validity

* **Simulated instruments.** Print, replay, codecs, virtual-camera timing, emulator telemetry,
  face swaps and GAN artefacts are software models of real tooling. They exercise every code path
  and give reproducible, comparable numbers, but **absolute rates will differ on real attacks**.
  The same harness accepts real captured media. For a client this campaign would be re-run with
  OBS / DeepFaceLive / rooted-device captures recorded under written authorisation.
* **Vendor stand-in.** B0 is not FaceTec. Read improvements on injection rows as "beyond what
  any camera-side PAD can see", not as "better than FaceTec".
* **Same simulator for training and test** on seen species. Unseen species (†), held-out subjects
  and calibration without attack data are the safeguards. Seen-species results are optimistic.
* **Population.** Bona fide faces are internet photos (varied ethnicity, age and quality), not a
  controlled demographic sample. Per-demographic BPCER should be measured on client traffic.
* **Sample size.** <<n_per_species>> attacks per species gives ±~1–5 pp intervals. Rare-event
  claims (< 1 %) need larger campaigns.

---

## 8. Reproduce

```bash
make assets data sessions train eval report   # ~1.5 h on 2 CPU cores, no GPU
pytest -q --junitxml reports/junit.xml        # logic-attack table
```
