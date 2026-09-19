"""Resumable enrichment orchestration.

The only interesting behaviour here is what it refuses to redo. Anything already
in the store for this (prompt_version, model, source_hash) is skipped, so a
restart after a crash - or a re-run after adding 500 new catalogue items -
costs only the work that is genuinely outstanding.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass

from ..types import Document
from .client import CatalogueItem, EnrichmentClient
from .schema import PROMPT_VERSION, Enrichment
from .store import EnrichmentStore

_TERMINAL = {"ended", "canceled", "expired"}


@dataclass(slots=True)
class EnrichReport:
    items: int
    enriched: int
    skipped: int
    failed: int
    recognised: int
    model: str
    prompt_version: str

    @property
    def grounded(self) -> int:
        """Items the model did not recognise, enriched from the record alone."""
        return self.enriched - self.recognised

    def to_dict(self) -> dict:
        return {
            "items": self.items,
            "enriched": self.enriched,
            "skipped": self.skipped,
            "failed": self.failed,
            "recognised": self.recognised,
            "grounded": self.grounded,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


def to_items(documents: Sequence[Document]) -> list[CatalogueItem]:
    return [
        CatalogueItem(item_id=d.doc_id, title=d.title, record=d.text)
        for d in documents
    ]


def pending(
    items: Sequence[CatalogueItem], store: EnrichmentStore, *, model: str
) -> list[CatalogueItem]:
    return [
        item
        for item in items
        if store.get(
            item.item_id,
            prompt_version=PROMPT_VERSION,
            model=model,
            record=item.record,
        )
        is None
    ]


def _inflight_path(store: EnrichmentStore) -> Path:
    return Path(str(store.path) + ".inflight.json")


def _write_inflight(store: EnrichmentStore, batch_id: str, *, model: str) -> None:
    """Record the batch id beside the store the moment it is submitted.

    Until the results are collected, this file is the only thing that connects
    money already spent to the store it belongs in.
    """
    _inflight_path(store).write_text(
        json.dumps({"batch_id": batch_id, "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "submitted_at": time.time()}),
        encoding="utf-8",
    )


def _read_inflight(store: EnrichmentStore, *, model: str) -> str | None:
    path = _inflight_path(store)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    # A batch is only reusable for the model and prompt that produced it.
    if raw.get("model") != model or raw.get("prompt_version") != PROMPT_VERSION:
        return None
    return raw.get("batch_id")


def _clear_inflight(store: EnrichmentStore) -> None:
    _inflight_path(store).unlink(missing_ok=True)


def enrich(
    documents: Sequence[Document],
    *,
    store: EnrichmentStore,
    client: EnrichmentClient,
    use_batch: bool = True,
    poll_seconds: int = 30,
    max_wait_seconds: int = 86_400,
    progress: bool = False,
) -> EnrichReport:
    items = to_items(documents)
    outstanding = pending(items, store, model=client.model)

    if progress:
        print(
            f"{len(items)} items · {len(outstanding)} outstanding · "
            f"model {client.model} · prompt {PROMPT_VERSION}"
        )

    produced: list[Enrichment] = []
    failed: list[str] = []

    if outstanding:
        if use_batch:
            # Reattach to a batch this store already paid for, rather than
            # submitting a second one. A poller that dies between submission and
            # collection otherwise strands the results: they sit completed on the
            # server, their id exists only in a log file, and the next run
            # cheerfully bills the whole corpus again.
            batch_id = _read_inflight(store, model=client.model)
            if batch_id:
                if progress:
                    print(f"reattaching to in-flight batch {batch_id} "
                          f"(already submitted and paid for)")
            else:
                # One real request first: a bad request shape fails every item in
                # the batch, and the API only says so after the whole thing has
                # been submitted and processed.
                if progress:
                    print("preflight: validating the request shape on one item")
                client.preflight(outstanding[0])
                batch_id = client.submit_batch(outstanding)
                _write_inflight(store, batch_id, model=client.model)
                if progress:
                    print(f"batch {batch_id} submitted; polling every {poll_seconds}s")
            waited = 0
            while client.batch_status(batch_id) not in _TERMINAL:
                if waited >= max_wait_seconds:
                    raise TimeoutError(
                        f"Batch {batch_id} still running after {max_wait_seconds}s. "
                        f"It is not lost - re-run to resume from the store."
                    )
                time.sleep(poll_seconds)
                waited += poll_seconds
            produced, failed = client.collect_batch(batch_id, outstanding)
            _clear_inflight(store)
        else:
            for item in outstanding:
                try:
                    produced.append(client.enrich_one(item))
                except Exception:
                    failed.append(item.item_id)
                if progress and len(produced) % 25 == 0 and produced:
                    print(f"  {len(produced)}/{len(outstanding)}")

        # Persist before doing anything else with them.
        store.add(produced)

    return EnrichReport(
        items=len(items),
        enriched=len(produced),
        skipped=len(items) - len(outstanding),
        failed=len(failed),
        recognised=sum(1 for e in produced if e.recognised),
        model=client.model,
        prompt_version=PROMPT_VERSION,
    )


def enriched_documents(
    documents: Sequence[Document],
    *,
    store: EnrichmentStore,
    model: str,
) -> list[Document]:
    """Rebuild documents with enrichment folded into the searchable text.

    The original record is kept in the text so the keyword lane can still match
    exact author, title and subject-heading terms - the enrichment is added
    alongside it, never in place of it.
    """
    available = store.for_model(prompt_version=PROMPT_VERSION, model=model)
    rebuilt: list[Document] = []
    for document in documents:
        enrichment = available.get(document.doc_id)
        if enrichment is None:
            rebuilt.append(document)
            continue
        rebuilt.append(
            Document(
                doc_id=document.doc_id,
                title=document.title,
                text=document.text + "\n\n" + enrichment.embed_text(title=document.title),
                url=document.url,
                tags=document.tags,
            )
        )
    return rebuilt
