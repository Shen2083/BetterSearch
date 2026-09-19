"""Enrichment pipeline tests.

A fake client stands in for the Anthropic API, so every behaviour that matters -
caching, resumability, prompt invalidation, out-of-order batch results - is
verified with no key, no network and no cost.
"""

from __future__ import annotations

import pytest

from bettersearch.enrichment.client import CatalogueItem
from bettersearch.enrichment.pipeline import (
    enrich,
    enriched_documents,
    pending,
    to_items,
)
from bettersearch.enrichment.schema import (
    PROMPT_VERSION,
    Enrichment,
    source_hash,
)
from bettersearch.enrichment.store import EnrichmentStore
from bettersearch.types import Document


class FakeClient:
    """Records every item it was asked to enrich, so we can assert on cost."""

    def __init__(self, model: str = "fake-model", *, recognise: bool = True) -> None:
        self.model = model
        self.calls: list[str] = []
        self._recognise = recognise

    def _make(self, item: CatalogueItem) -> Enrichment:
        self.calls.append(item.item_id)
        return Enrichment(
            item_id=item.item_id,
            recognised=self._recognise,
            synopsis=f"A synopsis of {item.title}.",
            topics=["norman conquest", "1066"] if self._recognise else ["history"],
            entities=["William the Conqueror"] if self._recognise else [],
            questions=[f"What is {item.title} about?"],
            period="1066-1154" if self._recognise else None,
            prompt_version=PROMPT_VERSION,
            enrichment_model=self.model,
            source_hash=source_hash(item.record),
        )

    def preflight(self, item: CatalogueItem) -> None:
        """Real clients send one live request here to validate the shape."""
        self.preflighted = True

    def enrich_one(self, item: CatalogueItem) -> Enrichment:
        return self._make(item)

    def submit_batch(self, items) -> str:
        self._batch = list(items)
        return "batch_fake"

    def batch_status(self, batch_id: str) -> str:
        return "ended"

    def collect_batch(self, batch_id: str, items):
        # Return in reverse order deliberately: the real Batch API gives results
        # back in arbitrary order, and anything keyed by position is a bug.
        return [self._make(i) for i in reversed(list(items))], []


@pytest.fixture
def records() -> list[Document]:
    return [
        Document(
            doc_id="hastings",
            title="The Battle of Hastings",
            text="Author: Merrick, R. D.\nPublished: 1993\nSubjects: History; Great Britain",
        ),
        Document(
            doc_id="domesday",
            title="The Domesday Book",
            text="Author: Doyle, K.\nPublished: 1987\nSubjects: History; Great Britain",
        ),
    ]


@pytest.fixture
def store(tmp_path) -> EnrichmentStore:
    return EnrichmentStore(tmp_path / "enrichment.jsonl")


# ------------------------------------------------------------------ the schema
def test_embed_text_carries_the_bridging_vocabulary():
    """The whole mechanism: terms the record never contained become searchable."""
    e = Enrichment(
        item_id="x",
        recognised=True,
        synopsis="An account of the invasion of England.",
        topics=["Norman Conquest", "feudalism"],
        entities=["William the Conqueror"],
        questions=["What happened in 1066?"],
        period="1066-1154",
    )
    text = e.embed_text(title="A Study of Senlac")
    for term in ("Norman Conquest", "feudalism", "William the Conqueror", "1066"):
        assert term in text
    assert text.startswith("A Study of Senlac")


def test_embed_text_omits_empty_fields():
    e = Enrichment(item_id="x", recognised=False, synopsis="Thin.")
    text = e.embed_text(title="T")
    assert "Topics:" not in text
    assert "Period:" not in text


def test_enrichment_round_trips():
    e = Enrichment(
        item_id="x", recognised=True, synopsis="S", topics=["a"], questions=["q?"]
    )
    assert Enrichment.from_dict(e.to_dict()).to_dict() == e.to_dict()


# ------------------------------------------------------------------- the store
def test_store_persists_across_instances(tmp_path, records):
    path = tmp_path / "e.jsonl"
    client = FakeClient()
    enrich(records, store=EnrichmentStore(path), client=client, use_batch=False)

    reopened = EnrichmentStore(path)
    assert len(reopened) == 2
    assert reopened.for_model(prompt_version=PROMPT_VERSION, model="fake-model").keys() == {
        "hastings",
        "domesday",
    }


def test_lookup_is_keyed_on_the_source_record(store, records):
    client = FakeClient()
    enrich(records, store=store, client=client, use_batch=False)

    item = to_items(records)[0]
    assert store.get(
        item.item_id, prompt_version=PROMPT_VERSION, model="fake-model",
        record=item.record,
    ) is not None
    # An edited record must miss, so it gets re-enriched.
    assert store.get(
        item.item_id, prompt_version=PROMPT_VERSION, model="fake-model",
        record=item.record + " revised edition",
    ) is None


# ---------------------------------------------------------------- the pipeline
def test_first_run_enriches_everything(store, records):
    client = FakeClient()
    report = enrich(records, store=store, client=client, use_batch=False)
    assert report.items == 2
    assert report.enriched == 2
    assert report.skipped == 0
    assert len(client.calls) == 2


def test_rerun_costs_nothing(store, records):
    """The money question: a no-op re-run must make zero API calls."""
    client = FakeClient()
    enrich(records, store=store, client=client, use_batch=False)
    calls_after_first = len(client.calls)

    report = enrich(records, store=store, client=client, use_batch=False)

    assert report.enriched == 0
    assert report.skipped == 2
    assert len(client.calls) == calls_after_first


def test_only_changed_records_are_re_enriched(store, records):
    client = FakeClient()
    enrich(records, store=store, client=client, use_batch=False)
    calls_after_first = len(client.calls)

    edited = list(records)
    edited[0] = Document(
        doc_id="hastings",
        title=records[0].title,
        text=records[0].text + "; Normans",
    )
    report = enrich(edited, store=store, client=client, use_batch=False)

    assert report.enriched == 1
    assert report.skipped == 1
    assert len(client.calls) == calls_after_first + 1


def test_a_prompt_change_invalidates_everything(store, records, monkeypatch):
    """A prompt edit must force a full re-run - it is a costed migration."""
    client = FakeClient()
    enrich(records, store=store, client=client, use_batch=False)

    monkeypatch.setattr("bettersearch.enrichment.pipeline.PROMPT_VERSION", "v2")
    assert len(pending(to_items(records), store, model="fake-model")) == 2


def test_a_different_model_does_not_reuse_stored_output(store, records):
    enrich(records, store=store, client=FakeClient("model-a"), use_batch=False)
    outstanding = pending(to_items(records), store, model="model-b")
    assert len(outstanding) == 2, "a bake-off must not reuse another model's output"


def test_batch_results_are_matched_by_id_not_position(store, records):
    """FakeClient returns results reversed, as the real Batch API may."""
    client = FakeClient()
    enrich(records, store=store, client=client, use_batch=True)

    stored = store.for_model(prompt_version=PROMPT_VERSION, model="fake-model")
    assert "The Battle of Hastings" in stored["hastings"].synopsis
    assert "The Domesday Book" in stored["domesday"].synopsis


def test_report_separates_recognised_from_grounded(store, records):
    report = enrich(
        records, store=store, client=FakeClient(recognise=False), use_batch=False
    )
    assert report.recognised == 0
    assert report.grounded == 2


# ------------------------------------------------------- composing for indexing
def test_enrichment_is_added_alongside_the_record_not_instead_of_it(store, records):
    """The keyword lane still needs the original author and subject headings."""
    enrich(records, store=store, client=FakeClient(), use_batch=False)
    rebuilt = enriched_documents(records, store=store, model="fake-model")

    assert "Merrick, R. D." in rebuilt[0].text      # original record survives
    assert "Norman Conquest".lower() in rebuilt[0].text.lower()  # enrichment added
    assert rebuilt[0].doc_id == records[0].doc_id


def test_unenriched_documents_pass_through_unchanged(store, records):
    rebuilt = enriched_documents(records, store=store, model="fake-model")
    assert rebuilt[0].text == records[0].text


def test_batch_runs_preflight_before_submitting(store, records):
    """One bad request shape fails every item in a batch, and the API only says
    so after the whole batch has been processed. A 4,000-item run came back 100%
    errored on a parameter the model did not accept; one request would have
    caught it."""
    client = FakeClient()
    enrich(records, store=store, client=client, use_batch=True)
    assert getattr(client, "preflighted", False), (
        "batch submission must validate the request shape on one item first"
    )


def test_effort_is_omitted_for_models_that_reject_it():
    """Haiku 4.5 rejects output_config.effort outright, failing the request
    before inference. Effort is an optimisation, never a requirement."""
    from bettersearch.enrichment.client import _request_params, supports_effort

    item = CatalogueItem("x", "Title", "Author: A")
    assert supports_effort("claude-opus-5")
    assert not supports_effort("claude-haiku-4-5")
    assert "effort" in _request_params(item, "claude-opus-5")["output_config"]
    assert "effort" not in _request_params(item, "claude-haiku-4-5")["output_config"]
    # The schema must survive either way - that is what makes the output parse.
    for model in ("claude-opus-5", "claude-haiku-4-5"):
        assert _request_params(item, model)["output_config"]["format"]["type"] == "json_schema"


def test_a_dead_poller_does_not_lose_a_paid_batch(store, records, tmp_path):
    """A poller that dies between submitting and collecting must not cost the
    run twice. The batch id is recorded beside the store at submit time, so the
    next run reattaches to the batch already paid for instead of billing the
    whole corpus again."""
    from bettersearch.enrichment.pipeline import _read_inflight, _write_inflight

    class DyingClient(FakeClient):
        def collect_batch(self, batch_id, items):
            raise RuntimeError("poller died before collecting")

    dying = DyingClient()
    try:
        enrich(records, store=store, client=dying, use_batch=True)
    except RuntimeError:
        pass

    # The id survives the crash, scoped to the model that produced it.
    assert _read_inflight(store, model=dying.model) == dying.submit_batch([]), (
        "the submitted batch id must be recorded beside the store"
    )
    assert _read_inflight(store, model="a-different-model") is None, (
        "a batch must never be reused for a model that did not produce it"
    )

    # A second run reattaches rather than submitting again.
    recovering = FakeClient()
    recovering.submitted = 0
    original = FakeClient.submit_batch

    def counting(self, items):
        self.submitted += 1
        return original(self, items)

    FakeClient.submit_batch = counting
    try:
        enrich(records, store=store, client=recovering, use_batch=True)
    finally:
        FakeClient.submit_batch = original
    assert recovering.submitted == 0, "must reattach, not resubmit and pay twice"
    assert _read_inflight(store, model=recovering.model) is None, (
        "the sidecar must be cleared once results are collected"
    )
