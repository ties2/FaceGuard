"""Publish trained detectors to the MSP registry and promote them.

    # against a running stack (docker compose exec api ...)
    python -m src.training.register --version 1.0.0
    # or fully in-process (CI, notebooks)
    python -m src.training.register --version 1.0.0 --in-process

Flow is exactly MSP's: artifact -> storage backend -> register version ->
promote (promotion re-loads the artifact and refuses corrupt files).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.serving.dependency import get_artifact_storage

ARTIFACTS = Path("artifacts/models")
MODELS = [
    # (registry name, file, framework, format, description)
    ("faceguard-pad-texture", "pad-texture.joblib", "scikit-learn", "joblib", "Frame texture PAD (print/replay)"),
    (
        "faceguard-capture-integrity",
        "capture-integrity.joblib",
        "scikit-learn",
        "joblib",
        "Injection / stream authenticity",
    ),
    ("faceguard-deepfake", "deepfake-detector.onnx", "pytorch", "onnx", "Face-manipulation CNN (SBI-trained)"),
    (
        "faceguard-deepfake-forensics",
        "deepfake-forensics.joblib",
        "scikit-learn",
        "joblib",
        "Face-manipulation forensics",
    ),
]


def register_all(client, version: str, artifacts: Path = ARTIFACTS) -> list[dict]:
    storage = get_artifact_storage()
    out = []
    for name, fname, framework, fmt, desc in MODELS:
        uri = f"local://{name}/{version}/{fname}"
        storage.save(uri, (artifacts / fname).read_bytes())
        client.post("/models", json={"name": name, "description": desc})  # 400 if it exists: fine
        r = client.post(
            f"/models/{name}/versions",
            json={
                "version": version,
                "framework": framework,
                "model_format": fmt,
                "artifact_uri": uri,
                "status": "staging",
            },
        )
        if r.status_code not in (200, 400):
            raise RuntimeError(f"register {name}: {r.status_code} {r.text}")
        current = {v["version"]: v["status"] for v in client.get(f"/models/{name}/versions").json()}
        if current.get(version) == "production":  # already live: re-running is a no-op
            out.append({"model_name": name, "version": version, "status": "production"})
            continue
        if current.get(version) == "archived":
            raise RuntimeError(
                f"{name}:{version} is archived (MSP never re-promotes archived versions). "
                f"Register a new version, e.g. --version 1.0.1"
            )
        p = client.post(f"/models/{name}/versions/{version}/promote")
        if p.status_code != 200:
            raise RuntimeError(f"promote {name}: {p.status_code} {p.text}")
        out.append(p.json())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--in-process", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("MSP_BASE_URL", "http://localhost:8000"))
    args = ap.parse_args()
    if args.in_process:
        from fastapi.testclient import TestClient

        from src.serving.app import app

        with TestClient(app) as c:
            print(register_all(c, args.version))
    else:
        import httpx

        with httpx.Client(base_url=args.base_url, timeout=60) as c:
            print(register_all(c, args.version))


if __name__ == "__main__":
    main()