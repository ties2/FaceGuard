"""Attack catalogue — PAI species / injection instruments used by the harness.

Terminology follows ISO/IEC 30107-1/-3 (presentation attacks) and CEN/TS 18099
(injection attacks: Injection Attack Instrument = *what* is injected,
Injection Attack Method = *how* it reaches the system).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    id: str
    family: str  # presentation | injection | adaptive
    instrument: str  # PAI / IAI
    method: str  # how it reaches the system
    description: str
    seen_in_training: bool  # False = generalisation test (no model saw it)
    attack_potential: str  # rough ISO 18045-style effort: low / moderate / high


SPECIES: dict[str, Species] = {
    s.id: s
    for s in [
        Species(
            "print_photo",
            "presentation",
            "Printed photo of victim",
            "Held in front of a real camera",
            "Paper print: halftone, reduced gamut/dynamic range, flat geometry",
            True,
            "low",
        ),
        Species(
            "screen_replay",
            "presentation",
            "Photo/video of victim on a phone or monitor",
            "Held in front of a real camera",
            "Display recapture: moire, colour shift, glare, refresh banding",
            True,
            "low",
        ),
        Species(
            "vcam_prerecorded",
            "injection",
            "Pre-recorded genuine video of victim",
            "Virtual camera (OBS/ManyCam), driver name visible",
            "Clean replayed stream, codec-compressed, synthetic frame timing",
            True,
            "low",
        ),
        Species(
            "vcam_prerecorded_stealth",
            "injection",
            "Pre-recorded genuine video of victim",
            "Virtual camera renamed to a real webcam; added sensor-like noise and timing jitter",
            "Same instrument, metadata hardened against signature rules",
            False,
            "moderate",
        ),
        Species(
            "vcam_offline_faceswap",
            "injection",
            "Offline deepfake (victim face swapped onto attacker video)",
            "Virtual camera",
            "Face-swap with generator bottleneck, colour transfer, alpha/Poisson blending",
            True,
            "moderate",
        ),
        Species(
            "vcam_realtime_faceswap",
            "injection",
            "Real-time face swap (DeepFaceLive-style) of attacker's live camera",
            "Virtual camera; label spoofed in 50% of attempts",
            "Live attacker reflects the challenge, but the swap adds 130-330 ms latency",
            False,
            "high",
        ),
        Species(
            "vcam_synthetic_gan",
            "injection",
            "Fully synthetic face (GAN-style upsampling artefacts)",
            "Virtual camera; label spoofed in 50% of attempts",
            "Synthetic identity, no live subject",
            False,
            "moderate",
        ),
        Species(
            "emulator_injection",
            "injection",
            "Pre-recorded genuine video of victim",
            "Android emulator / rooted device with camera hook (VCAM-style module)",
            "Mobile camera API hooked; attestation usually fails",
            True,
            "moderate",
        ),
        Species(
            "challenge_aware_injection",
            "adaptive",
            "Pre-recorded genuine video of victim, re-lit in real time",
            "Stealth virtual camera + screen capture of the challenge, 2-5 frame pipeline lag",
            "White-box attacker who knows the photometric check and tries to fake it",
            False,
            "high",
        ),
        Species(
            "challenge_aware_faceswap",
            "adaptive",
            "Real-time face swap of a live attacker, re-lit in real time",
            "Stealth virtual camera + screen capture of the challenge, 2-5 frame pipeline lag",
            "Most capable attacker: hardened metadata and a faked challenge; only manipulation artefacts remain",
            False,
            "high",
        ),
    ]
}

PRESENTATION_SPECIES = [k for k, s in SPECIES.items() if s.family == "presentation"]
ALL_ATTACKS = list(SPECIES)
