"""Server-side analysis latency on 640x480 frames (30-frame session, warm cache).

    python scripts/bench_latency.py --url http://localhost:8000 --n 20

Writes reports/latency.json (p50 / p95 of the pipeline time reported by the API).
"""

from __future__ import annotations

import argparse
import base64
import json
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]


def frames_640(n: int = 30) -> list[str]:
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[20:420, 56:456]
    out = []
    for t in range(n):
        canvas = np.full((480, 640, 3), 40, np.uint8)
        M = cv2.getRotationMatrix2D((200, 160), 0.5 * np.sin(t / 5), 1.0)
        canvas[40:440, 120:520] = cv2.warpAffine(img, M, (400, 400), borderMode=cv2.BORDER_REFLECT_101)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(base64.b64encode(buf.tobytes()).decode())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    fr = frames_640()
    lat = []
    with httpx.Client(base_url=args.url, timeout=60) as c:
        for i in range(args.n + 2):
            ids = {"user_ref": uuid.uuid4().hex, "device_id": uuid.uuid4().hex}
            s = c.post("/v1/liveness/sessions", json=ids).json()
            body = {
                "session_token": s["session_token"],
                "frames": fr,
                "timestamps_ms": [33.3 * k for k in range(len(fr))],
                "telemetry": {"platform": "web", "camera_label": "Logitech BRIO"},
                "vendor": {"face_scan": f"bench-{uuid.uuid4()}", "audit_trail_image": fr[15]},
            }
            r = c.post(f"/v1/liveness/sessions/{s['session_id']}/verify", json=body).json()
            if i >= 2:  # first calls warm the model cache
                lat.append(r["latency_ms"])
    res = {
        "n": len(lat),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "frames": "30 x 640x480",
    }
    (ROOT / "reports" / "latency.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
