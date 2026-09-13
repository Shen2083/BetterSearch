#!/usr/bin/env python3
"""Capture the README screenshots from the running catalogue demo.

Written as a script rather than captured by hand so the README images can be
regenerated when the interface changes, instead of becoming stale artefacts
nobody dares touch.

Each shot is a deliberate crop, not a full-page dump. The two catalogue shots
must be roughly equal in height to read as a comparison, and the collision shot
is worthless unless both author lines are legible.

    uvicorn api.main:app --port 8100      # in another shell, with
                                          # BETTERSEARCH_INDEX_PATH=.bettersearch/catalogue
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
OUT = Path("docs/screenshots")
WIDTH = 1280

# name, query, mode, crop height, y offset
SHOTS = [
    ("catalogue-keyword", "learning to be present", "keyword", 720, 0),
    ("catalogue-meaning", "learning to be present", "semantic", 720, 0),
    # Offset past the masthead, clear of the breadcrumb, so the author lines
    # carry the frame.
    ("catalogue-collision", "anand", "keyword", 760, 272),
]


async def capture(base_url: str) -> list[Path]:
    from playwright.async_api import async_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROMIUM)
        # 2x for sharpness on high-density displays; GitHub scales it down.
        page = await browser.new_page(
            viewport={"width": WIDTH, "height": 1000}, device_scale_factor=2
        )
        await page.goto(f"{base_url}/catalogue", wait_until="networkidle")

        for name, query, mode, height, offset in SHOTS:
            await page.fill("#q", query)
            await page.click(f"#m-{mode}")
            await page.wait_for_timeout(1500)
            path = OUT / f"{name}.png"
            await page.screenshot(
                path=str(path),
                full_page=True,
                clip={"x": 0, "y": offset, "width": WIDTH, "height": height},
            )
            written.append(path)

        await browser.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    args = parser.parse_args()

    for path in asyncio.run(capture(args.base_url)):
        kb = path.stat().st_size // 1024
        flag = "  <-- over 250KB budget" if kb > 250 else ""
        print(f"{path}  {kb} KB{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
