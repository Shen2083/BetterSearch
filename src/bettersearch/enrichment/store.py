"""Enrichment persistence.

Keyed by ``(item_id, prompt_version, enrichment_model, source_hash)``. All four
matter:

* ``prompt_version`` - a prompt change invalidates stored output
* ``enrichment_model`` - so a bake-off across model tiers can coexist on disk
* ``source_hash`` - so an edited catalogue record is re-enriched and an
  unedited one never is

At 100k-1M items a full run is a multi-day, four-figure job. Everything is
appended as it lands so a crash costs the in-flight batch and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Enrichment, source_hash


class EnrichmentStore:
    """Append-only JSONL store. Last write for a key wins."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: dict[tuple[str, str, str, str], Enrichment] = {}
        self._load()

    @property
    def path(self) -> Path:
        """Where this store lives. Callers need it to place sidecar files."""
        return self._path

    @staticmethod
    def _key(e: Enrichment) -> tuple[str, str, str, str]:
        return (e.item_id, e.prompt_version, e.enrichment_model, e.source_hash)

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                enrichment = Enrichment.from_dict(json.loads(line))
                self._entries[self._key(enrichment)] = enrichment

    def add(self, enrichments: list[Enrichment]) -> int:
        if not enrichments:
            return 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for enrichment in enrichments:
                handle.write(json.dumps(enrichment.to_dict(), ensure_ascii=False) + "\n")
                self._entries[self._key(enrichment)] = enrichment
        return len(enrichments)

    def get(
        self, item_id: str, *, prompt_version: str, model: str, record: str
    ) -> Enrichment | None:
        return self._entries.get(
            (item_id, prompt_version, model, source_hash(record))
        )

    def for_model(self, *, prompt_version: str, model: str) -> dict[str, Enrichment]:
        """Every enrichment produced by one prompt/model pair, keyed by item."""
        return {
            e.item_id: e
            for (item_id, pv, m, _), e in self._entries.items()
            if pv == prompt_version and m == model
        }

    def __len__(self) -> int:
        return len(self._entries)
