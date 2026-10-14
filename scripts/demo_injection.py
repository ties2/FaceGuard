"""Live injection demo: feed a pre-recorded video into the browser's camera and
watch FaceGuard reject it.

    make run &                       # API + /demo on :8000 with registered detectors
    python scripts/demo_injection.py # writes portfolio/06-demo-injection.png

Chromium's ``--use-file-for-fake-video-capture`` is a textbook virtual-camera
injection: the page receives a MediaStream that never touched a sensor. The
video is the public-domain NASA astronaut portrait (skimage), slowly zooming.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "portfolio"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def make_video(path: Path, seconds: int = 12, fps: int = 30) -> None:
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)  # 512x512
    img = img[20:420, 56:456]  # head and shoulders
    frames = []
    for t in range(seconds * fps):
        s = 1.0 + 0.04 * np.sin(2 * np.pi * t / (fps * 4))
        M = cv2.getRotationMatrix2D((200, 160), 1.5 * np.sin(2 * np.pi * t / (fps * 3)), s)
        f = cv2.warpAffine(img, M, (400, 400), borderMode=cv2.BORDER_REFLECT_101)
        canvas = np.full((480, 640, 3), 40, np.uint8)
        canvas[40:440, 120:520] = f
        frames.append(canvas)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "640x480",
            "-r",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def main(url: str = "http://localhost:8000/demo/") -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    video = OUT / "_build" / "injected.y4m"
    video.parent.mkdir(parents=True, exist_ok=True)
    make_video(video)
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path=CHROME,
            args=[
                "--no-sandbox",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={video}",
            ],
        )
        pg = b.new_page(viewport={"width": 1180, "height": 760})
        # a careful attacker hides the automation flag: detection must come from the challenge
        pg.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")
        pg.goto(url)
        pg.click("#start")
        done = "['done', 'error'].some(p => document.querySelector('#status').textContent.startsWith(p))"
        pg.wait_for_function(done, timeout=60000)
        pg.screenshot(path=str(OUT / "06-demo-injection.png"), full_page=True)
        print(pg.inner_text("#status"), "|", pg.inner_text("#decision"))
        b.close()


if __name__ == "__main__":
    main()
