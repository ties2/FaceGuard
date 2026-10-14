"""Nonce-bound photometric challenge.

The server issues a random nonce per session. The capture client turns the
screen into a light source and flashes a colour sequence derived from that
nonce (HMAC-SHA256 keystream). A real face *in front of the screen* reflects
those colours — more strongly than the background, which is further away —
with only camera-pipeline latency.

Why it matters for injection attacks: a pre-recorded or offline-generated
stream cannot contain a reflection of a sequence that did not exist when it was
made, and a real-time face-swap pipeline adds processing latency that shows up
as a lag in the reflected sequence. Security rests on the nonce's
unpredictability, not on any learned model.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np

# Saturated BGR palette (values 0..1) — easy for any display to render.
PALETTE = np.float32(
    [
        [0.0, 0.0, 1.0],  # red
        [0.0, 1.0, 0.0],  # green
        [1.0, 0.0, 0.0],  # blue
        [0.0, 1.0, 1.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [1.0, 1.0, 0.0],  # cyan
    ]
)
PALETTE_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan"]


@dataclass(frozen=True)
class Challenge:
    slots: tuple[int, ...]  # palette index per slot
    slot_frames: int = 3
    fps: float = 30.0

    @property
    def n_frames(self) -> int:
        return len(self.slots) * self.slot_frames

    def per_frame(self) -> np.ndarray:
        """(n_frames, 3) BGR colour displayed while each frame is exposed."""
        return np.repeat(PALETTE[list(self.slots)], self.slot_frames, axis=0)

    def to_hex(self) -> list[str]:
        out = []
        for i in self.slots:
            b, g, r = (PALETTE[i] * 255).astype(int)
            out.append(f"#{r:02x}{g:02x}{b:02x}")
        return out

    @property
    def slot_ms(self) -> float:
        return 1000.0 * self.slot_frames / self.fps


def challenge_from_nonce(nonce: bytes, secret: bytes, n_slots: int = 10, slot_frames: int = 3) -> Challenge:
    """Deterministically expand a nonce into a colour sequence (no immediate repeats)."""
    stream = b""
    counter = 0
    while len(stream) < n_slots * 2:
        stream += hmac.new(secret, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        counter += 1
    slots: list[int] = []
    for i in range(n_slots):
        idx = stream[i] % len(PALETTE)
        if slots and idx == slots[-1]:
            idx = (idx + 1 + stream[n_slots + i] % (len(PALETTE) - 1)) % len(PALETTE)
        slots.append(int(idx))
    return Challenge(tuple(slots), slot_frames=slot_frames)


def random_challenge(rng: np.random.Generator, n_slots: int = 10, slot_frames: int = 3) -> Challenge:
    return challenge_from_nonce(rng.bytes(16), b"harness", n_slots=n_slots, slot_frames=slot_frames)
