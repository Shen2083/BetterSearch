#!/usr/bin/env python3
"""Render an HTML or markdown file to PDF using the headless Chromium on the box.

Chromium's print engine is used rather than a Python PDF library because these
documents are typographic, not programmatic: real fonts, controlled page breaks,
and CSS that can be previewed in a browser while writing.

    python scripts/render_pdf.py docs/approach-client.html
    python scripts/render_pdf.py README.md --out docs/README.pdf

A markdown source is converted to a self-contained HTML file first, rendered,
and the intermediate deleted. That intermediate is written *beside the markdown*
rather than beside the PDF: Chromium loads over file://, so relative paths
resolve against the HTML file's own directory. README.md refers to its images as
docs/screenshots/..., relative to the repo root, so rendering from docs/ would
look for docs/docs/screenshots/... and silently emit four broken images.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

#: Relative links are dead in a PDF that gets emailed to someone, so they are
#: rewritten to point at the repository.
REPO_BLOB = "https://github.com/Shen2083/BetterSearch/blob/main/"

# The widest line in README.md is 101 characters. A4 at 18mm margins leaves
# 493pt of content width, and DejaVu Sans Mono at 8pt advances ~4.8pt per
# character, so ~102 characters fit. pre-wrap is the safety net: a longer line
# added later must wrap visibly rather than clip off the page edge unnoticed.
PRINT_CSS = """
@page { size: A4; }
* { box-sizing: border-box; }
:root {
  --ink: #1A1F23; --muted: #55636B; --rule: #D8DEE2;
  --panel: #F4F6F7; --link: #1B4D3E;
  /* Locally installed faces only - no webfont fetch, so rendering is
     deterministic and works offline. */
  --sans: 'Liberation Sans', 'DejaVu Sans', Helvetica, sans-serif;
  --mono: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
}
body {
  margin: 0; background: #fff; color: var(--ink);
  font-family: var(--sans); font-size: 10.5pt; line-height: 1.5;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1, h2, h3, h4 {
  font-weight: 600; line-height: 1.2;
  break-after: avoid; page-break-after: avoid;
}
h1 { font-size: 21pt; margin: 0 0 10pt; }
h2 { font-size: 14pt; margin: 20pt 0 7pt; }
h3 { font-size: 11.5pt; margin: 14pt 0 5pt; }
h4 { font-size: 10.5pt; margin: 12pt 0 4pt; }
p { margin: 0 0 8pt; orphans: 2; widows: 2; }
/* A label immediately above a screenshot must travel with it, or it strands at
   the foot of a page while the image it introduces starts the next one. */
p:has(+ p > img) { break-after: avoid; page-break-after: avoid; }
a { color: var(--link); }
strong { font-weight: 600; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 16pt 0; }

ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 4pt; }

code {
  font-family: var(--mono); font-size: 8.8pt;
  background: var(--panel); padding: 0 3px; border-radius: 2px;
}
pre {
  font-family: var(--mono); font-size: 8pt; line-height: 1.45;
  background: var(--panel); border: 1px solid var(--rule);
  padding: 7pt 9pt; margin: 9pt 0;
  white-space: pre-wrap; overflow-wrap: anywhere;
  break-inside: avoid; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }

table {
  border-collapse: collapse; width: 100%; font-size: 9pt;
  margin: 9pt 0; break-inside: avoid; page-break-inside: avoid;
}
th, td {
  text-align: left; vertical-align: top;
  padding: 4pt 8pt 4pt 0; border-bottom: 1px solid var(--rule);
}
th {
  font-weight: 600; font-size: 8.5pt; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.03em;
}
td code, th code { font-size: 8.2pt; }

p:has(> img) { margin: 10pt 0; break-inside: avoid; page-break-inside: avoid; }
img {
  display: block; width: 100%; height: auto;
  border: 1px solid var(--rule);
  break-inside: avoid; page-break-inside: avoid;
}
"""

FOOTER_TEMPLATE = (
    '<div style="width:100%;font-family:sans-serif;font-size:8pt;color:#55636B;'
    'text-align:center;margin:0 18mm;">'
    '<span class="pageNumber"></span></div>'
)
EMPTY_HEADER = '<div style="display:none"></div>'

HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


# ----------------------------------------------------------------- markdown
def markdown_to_html(source: Path) -> str:
    """Convert a markdown file to a self-contained HTML document."""
    from markdown_it import MarkdownIt

    # "js-default" rather than "gfm-like": both give tables, but gfm-like also
    # enables linkify, and linkify-it-py is not installed - it raises on
    # construction.
    md = MarkdownIt("js-default")

    # add_render_rule binds these to the renderer, so they take `self` too.
    def render_image(self, tokens, idx, options, env):
        token = tokens[idx]
        src = token.attrGet("src") or ""
        if src.startswith(("http://", "https://", "//")):
            # Cannot be fetched offline; the GitHub CI badge 403s through the
            # proxy and would print as a broken-image box.
            return ""
        alt = self.renderInlineAsText(token.children or [], options, env)
        return f'<img src="{src}" alt="{_escape(alt)}">'

    def render_link_open(self, tokens, idx, options, env):
        token = tokens[idx]
        href = token.attrGet("href") or ""
        if not href.startswith(("http://", "https://", "//", "#", "mailto:")):
            token.attrSet("href", REPO_BLOB + href.lstrip("./"))
        return self.renderToken(tokens, idx, options, env)

    md.add_render_rule("image", render_image)
    md.add_render_rule("link_open", render_link_open)

    body = md.render(source.read_text(encoding="utf-8"))
    # Dropping the badge leaves <p><a href="..."></a></p> behind.
    body = re.sub(r"<p>\s*(<a [^>]*>\s*</a>\s*)+</p>", "", body)

    title = source.stem
    if match := re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S):
        title = re.sub(r"<[^>]+>", "", match.group(1)).strip() or title
    return HTML_SHELL.format(title=_escape(title), css=PRINT_CSS, body=body)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# -------------------------------------------------------------------- render
async def render(source: Path, out: Path, *, page_numbers: bool = False) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROMIUM)
        page = await browser.new_page()
        await page.goto(source.resolve().as_uri(), wait_until="networkidle")
        await page.emulate_media(media="print")
        extra = {}
        if page_numbers:
            extra = {
                "display_header_footer": True,
                "header_template": EMPTY_HEADER,
                "footer_template": FOOTER_TEMPLATE,
            }
        await page.pdf(
            path=str(out),
            format="A4",
            print_background=True,
            margin={"top": "18mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
            **extra,
        )
        await browser.close()


def render_source(source: Path, out: Path, *, keep_html: bool = False) -> None:
    """Render one source to `out`, converting markdown first if needed."""
    if source.suffix.lower() not in {".md", ".markdown"}:
        asyncio.run(render(source, out))
        return

    html = markdown_to_html(source)
    if keep_html:
        target = source.with_suffix(".print.html")
        target.write_text(html, encoding="utf-8")
        print(f"kept {target}")
        asyncio.run(render(target, out, page_numbers=True))
        return

    # Beside the markdown so relative image paths resolve - see module docstring.
    fd, raw = tempfile.mkstemp(dir=source.parent, prefix=".mdprint-", suffix=".html")
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(html)
        asyncio.run(render(tmp, out, page_numbers=True))
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="HTML or markdown files")
    parser.add_argument(
        "--out",
        help="output path (default: the source with a .pdf suffix); "
        "only valid with a single source",
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="keep the generated HTML next to a markdown source, for debugging "
        "the stylesheet without re-rendering",
    )
    args = parser.parse_args()

    if args.out and len(args.sources) > 1:
        print("--out takes a single source", file=sys.stderr)
        return 1

    for raw in args.sources:
        source = Path(raw)
        if not source.exists():
            print(f"missing: {source}", file=sys.stderr)
            return 1
        out = Path(args.out) if args.out else source.with_suffix(".pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        render_source(source, out, keep_html=args.keep_html)
        print(f"{source} -> {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
