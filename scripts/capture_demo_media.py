#!/usr/bin/env python3
"""Capture the README's screenshot and GIF from the static demo build.

    (cd frontend && npm run build:demo && npx vite preview --mode demo --outDir dist-demo --port 4174)
    python scripts/capture_demo_media.py --url http://localhost:4174/Synapse/

Uses the locally installed Google Chrome through Playwright (no browser
download). Writes:
    docs/screenshots/benchmarks.png   the Benchmarks page (top: tiles + speed chart)
    docs/screenshots/replay.gif       one replay-Playground topic: miss, repeat, reworded
The GIF is assembled from frames sampled while the replay plays, so the
streaming in it follows the recorded timing; the pauses between steps are
added for readability.
"""

import argparse
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "screenshots"
VIEWPORT = {"width": 1100, "height": 760}


def shot(page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def capture(url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2, color_scheme="dark")

        page.goto(f"{url}#/benchmarks")
        page.wait_for_selector(".recharts-surface")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "benchmarks.png"), clip={"x": 0, "y": 0, **VIEWPORT})

        # The GIF uses 1x pixels and a taller window: screenshots are fast
        # enough to catch the reply streaming in, and all three exchanges
        # (with their cache-hit chips) fit in the frame.
        page = browser.new_page(viewport={"width": 1000, "height": 960}, device_scale_factor=1, color_scheme="dark")
        page.goto(f"{url}#/playground")
        page.wait_for_selector("text=Replay, not a live model.")
        frames: list[tuple[Image.Image, int]] = [(shot(page), 1200)]
        for label in ("Ask", "Ask the same again", "Ask it reworded"):
            page.get_by_role("button", name=label, exact=True).click()
            # Sample while the reply streams in (the captured topic's longest
            # reply takes 0.45 s), then hold on the finished state.
            start = time.monotonic()
            while time.monotonic() - start < 1.0:
                frames.append((shot(page), 80))
            frames.append((shot(page), 1600))
        frames[-1] = (frames[-1][0], 3000)
        browser.close()

    # The Playground column is ~760 px wide; drop the empty space to its right.
    cropped = [img.crop((0, 0, 790, img.height)) for img, _ in frames]
    palette_frames = [img.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for img in cropped]
    palette_frames[0].save(OUT / "replay.gif", save_all=True, append_images=palette_frames[1:],
                           duration=[d for _, d in frames], loop=0, optimize=True)
    for name in ("benchmarks.png", "replay.gif"):
        print(f"wrote {OUT / name} ({(OUT / name).stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:4174/Synapse/")
    capture(parser.parse_args().url)
