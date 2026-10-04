"""Tests for the markdown-to-print conversion.

These cover the one failure mode this renderer has already shipped: output that
looks deliberate. A Mermaid fence that does not render prints as a code block,
and a code block in a PDF looks like a choice rather than a bug - which is how
docs/README.pdf sat stale for over a week without anyone noticing.

The delegation test earns its place. The first version of the fence rule
delegated non-Mermaid fences to `renderToken`, which emits only a token's
opening tag; a fence carries its content on the token, so every code block in
every document rendered as an empty `<code>`. Nothing about the PDF would have
looked broken enough to investigate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "render_pdf.py"
_spec = importlib.util.spec_from_file_location("render_pdf", SCRIPT)
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def body_of(tmp_path: Path, markdown: str) -> str:
    source = tmp_path / "doc.md"
    source.write_text(markdown, encoding="utf-8")
    html = rp.markdown_to_html(source)
    return html[html.index("<body>"):]


# ------------------------------------------------------------------- mermaid

def test_a_mermaid_fence_becomes_a_mermaid_pre(tmp_path):
    body = body_of(tmp_path, "```mermaid\nflowchart TD\n  A --> B\n```\n")
    assert 'class="mermaid"' in body
    assert "flowchart TD" in body
    # Not a code block: Mermaid's selector would miss it and it would print as
    # source, which is the whole failure being guarded against.
    assert "language-mermaid" not in body


def test_diagram_labels_are_escaped_not_injected(tmp_path):
    """Mermaid labels contain angle brackets; raw they would close the <pre>."""
    body = body_of(tmp_path, '```mermaid\nflowchart TD\n  A["a <b> c"]\n```\n')
    assert "&lt;b&gt;" in body
    assert "<b>" not in body


def test_an_info_string_with_arguments_still_matches(tmp_path):
    body = body_of(tmp_path, "```mermaid {theme=neutral}\nflowchart TD\n  A --> B\n```\n")
    assert 'class="mermaid"' in body


def test_a_lookalike_language_is_not_treated_as_mermaid(tmp_path):
    """`mermaidjs` is a different fence and must keep its code-block rendering."""
    body = body_of(tmp_path, "```mermaidjs\nnot a diagram\n```\n")
    assert 'class="mermaid"' not in body
    assert "not a diagram" in body


# ------------------------------------------------- every other fence survives

@pytest.mark.parametrize("info", ["python", "bash", "json", ""])
def test_a_non_mermaid_fence_keeps_its_content(tmp_path, info):
    body = body_of(tmp_path, f"```{info}\nkeep = 1\n```\n")
    assert "keep = 1" in body
    assert 'class="mermaid"' not in body


def test_a_non_mermaid_fence_keeps_its_language_class(tmp_path):
    body = body_of(tmp_path, "```python\nx = 1\n```\n")
    assert "language-python" in body


def test_code_content_is_still_escaped(tmp_path):
    body = body_of(tmp_path, "```python\nx = 1 < 2\n```\n")
    assert "x = 1 &lt; 2" in body


# ------------------------------------------------------- the fetch is lazy

def test_a_document_with_no_diagram_needs_no_mermaid(tmp_path, monkeypatch):
    """Most documents have no diagram and must render with no network at all."""
    def explode():
        raise AssertionError("fetched Mermaid for a document with no diagram")

    monkeypatch.setattr(rp, "mermaid_script", explode)
    body = body_of(tmp_path, "# Plain\n\nSome prose.\n")
    assert "Plain" in body
