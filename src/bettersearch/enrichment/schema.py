"""The shape of an enrichment record.

Structured fields rather than free prose, for three reasons:

* Free summaries generated from one prompt all sound alike, and homogeneous text
  embeds into a homogeneous cluster - which *reduces* discriminative power.
* Fields can be weighted independently when composing the text to embed, and
  faceted on later.
* ``recognised`` is the hallucination control. It lets the model tell us whether
  it actually knows the work, so we can refuse to index invented content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

#: Bumped whenever the prompt changes in a way that invalidates stored output.
#: Stored next to every enrichment so a prompt change is a costed, staged
#: migration rather than a silent inconsistency - the same discipline as
#: ``model_id`` on a vector, except regenerating costs money, not CPU.
PROMPT_VERSION = "v1"

ENRICHMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recognised": {
            "type": "boolean",
            "description": (
                "True only if you genuinely know this specific work. False if you "
                "are inferring from the title alone."
            ),
        },
        "synopsis": {
            "type": "string",
            "description": "Two to three sentences on what this work is about.",
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concepts a searcher might name when looking for this.",
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "People, places, events and organisations involved.",
        },
        "period": {
            "type": ["string", "null"],
            "description": "Historical period or date range, if applicable.",
        },
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Questions a reader could answer using this work.",
        },
        "audience": {"type": ["string", "null"]},
    },
    "required": [
        "recognised",
        "synopsis",
        "topics",
        "entities",
        "period",
        "questions",
        "audience",
    ],
    "additionalProperties": False,
}


def source_hash(record_text: str) -> str:
    """Hash of the input record, so re-runs only touch what changed."""
    return hashlib.sha256(record_text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Enrichment:
    item_id: str
    recognised: bool
    synopsis: str
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    period: str | None = None
    audience: str | None = None

    # Provenance. All three participate in the cache key.
    prompt_version: str = PROMPT_VERSION
    enrichment_model: str = ""
    source_hash: str = ""

    def embed_text(self, *, title: str) -> str:
        """Compose the text that actually gets embedded.

        Questions come first after the title because query-to-question matching
        is a closer shape match for natural-language search than
        query-to-document, and earlier text carries slightly more weight in most
        encoders.
        """
        parts = [title]
        if self.questions:
            parts.append(" ".join(self.questions))
        parts.append(self.synopsis)
        if self.topics:
            parts.append("Topics: " + ", ".join(self.topics))
        if self.entities:
            parts.append("Related: " + ", ".join(self.entities))
        if self.period:
            parts.append(f"Period: {self.period}")
        return "\n\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "recognised": self.recognised,
            "synopsis": self.synopsis,
            "topics": self.topics,
            "entities": self.entities,
            "questions": self.questions,
            "period": self.period,
            "audience": self.audience,
            "prompt_version": self.prompt_version,
            "enrichment_model": self.enrichment_model,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Enrichment":
        return cls(
            item_id=raw["item_id"],
            recognised=bool(raw["recognised"]),
            synopsis=raw["synopsis"],
            topics=list(raw.get("topics", [])),
            entities=list(raw.get("entities", [])),
            questions=list(raw.get("questions", [])),
            period=raw.get("period"),
            audience=raw.get("audience"),
            prompt_version=raw.get("prompt_version", PROMPT_VERSION),
            enrichment_model=raw.get("enrichment_model", ""),
            source_hash=raw.get("source_hash", ""),
        )

    @classmethod
    def from_model_output(
        cls,
        raw: dict[str, Any],
        *,
        item_id: str,
        model: str,
        source: str,
    ) -> "Enrichment":
        return cls(
            item_id=item_id,
            recognised=bool(raw.get("recognised", False)),
            synopsis=str(raw.get("synopsis", "")).strip(),
            topics=[str(t) for t in raw.get("topics") or []],
            entities=[str(e) for e in raw.get("entities") or []],
            questions=[str(q) for q in raw.get("questions") or []],
            period=raw.get("period") or None,
            audience=raw.get("audience") or None,
            prompt_version=PROMPT_VERSION,
            enrichment_model=model,
            source_hash=source_hash(source),
        )
