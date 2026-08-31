#!/usr/bin/env python3
"""Render an HTML file to PDF using the headless Chromium already on the box.

Chromium's print engine is used rather than a Python PDF library because these
documents are typographic, not programmatic: real fonts, controlled page breaks,
and CSS that can be previewed in a browser while writing.

    python scripts/render_pdf.py docs/approach-client.html
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


async def render(source: Path, out: Path) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROMIUM)
        page = await browser.new_page()
        await page.goto(source.resolve().as_uri(), wait_until="networkidle")
        await page.emulate_media(media="print")
        await page.pdf(
            path=str(out),
            format="A4",
            print_background=True,
            margin={"top": "18mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
        )
        await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="HTML files to render")
    args = parser.parse_args()

    for raw in args.sources:
        source = Path(raw)
        if not source.exists():
            print(f"missing: {source}", file=sys.stderr)
            return 1
        out = source.with_suffix(".pdf")
        asyncio.run(render(source, out))
        print(f"{source} -> {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
