"""Anthropic client for enrichment.

Two paths on purpose:

``enrich_one`` / ``enrich_sync``
    Straightforward request per item. Use for a few hundred items - a bake-off,
    a spot-check, a smoke test.

``submit_batch`` / ``collect_batch``
    The Batch API, at half the price. This is the only sane path for a
    100k-1M item catalogue. Results come back in **arbitrary order**, so they
    are keyed by ``custom_id`` and never by position.

Cost note: with thin records the input is tiny (~100 tokens against a cached
system prompt) and the output is ~300 tokens, so **output length dominates the
bill**. Synopsis length is the primary cost lever, not model tier.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .prompt import SYSTEM_PROMPT, build_user_message
from .schema import ENRICHMENT_JSON_SCHEMA, Enrichment

DEFAULT_MODEL = "claude-opus-5"
#: Enough for a 3-sentence synopsis plus lists, with headroom.
MAX_TOKENS = 1200


class MissingCredentialError(RuntimeError):
    """Raised when enrichment is requested with no API credentials configured."""


@dataclass(frozen=True, slots=True)
class CatalogueItem:
    item_id: str
    title: str
    record: str


#: Models that reject `output_config.effort` outright:
#:     "This model does not support the effort parameter."
#: Sending it anyway fails the whole request before inference, so a 4,000-item
#: batch comes back 100% errored. Effort is an optimisation, never a
#: requirement, so it is simply omitted where unsupported.
_NO_EFFORT_SUPPORT = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-3")


def supports_effort(model: str) -> bool:
    return not model.startswith(_NO_EFFORT_SUPPORT)


def _request_params(item: CatalogueItem, model: str) -> dict:
    # Descriptive metadata writing does not need deep reasoning, and thinking
    # tokens bill as output - which is the whole cost here. Where the model
    # cannot take the hint, the request still has to be valid.
    output_config: dict = {
        "format": {"type": "json_schema", "schema": ENRICHMENT_JSON_SCHEMA}
    }
    if supports_effort(model):
        output_config["effort"] = "low"

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # The system prompt is identical across every item in the run, so it is
        # worth a cache breakpoint. Verify with usage.cache_read_input_tokens -
        # a silent cache miss here is expensive at catalogue scale.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": build_user_message(title=item.title, record=item.record),
            }
        ],
        "output_config": output_config,
    }


def _parse(text: str, item: CatalogueItem, model: str) -> Enrichment:
    return Enrichment.from_model_output(
        json.loads(text), item_id=item.item_id, model=model, source=item.record
    )


class EnrichmentClient:
    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Enrichment needs the anthropic package. "
                'Install it with: pip install "bettersearch[enrich]"'
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Construction always succeeds; the SDK only resolves credentials when a
        # request is made, and it can also pick up an `ant auth login` profile
        # or workload identity that no env var reveals. So do not pre-judge here
        # - _guard() below turns the eventual failure into a clear message.
        self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self._anthropic = anthropic
        self.model = model

    @staticmethod
    def _guard(call):
        """Turn the SDK's auth TypeError into a message that says what to do."""
        try:
            return call()
        except TypeError as exc:
            if "authentication" in str(exc).lower():
                raise MissingCredentialError(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
                    "`ant auth login`. Enrichment is the only part of BetterSearch "
                    "that needs an API key - indexing and search do not."
                ) from exc
            raise

    # ------------------------------------------------------------ synchronous
    def enrich_one(self, item: CatalogueItem) -> Enrichment:
        response = self._guard(
            lambda: self._client.messages.create(**_request_params(item, self.model))
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return _parse(text, item, self.model)

    def enrich_sync(self, items: Sequence[CatalogueItem]) -> Iterator[Enrichment]:
        for item in items:
            yield self.enrich_one(item)

    # ------------------------------------------------------------------ batch
    def preflight(self, item: CatalogueItem) -> None:
        """Send one real request to prove the request shape is valid.

        A malformed request fails every item in a batch identically, and the
        Batch API reports that as 4,000 individual errors long after submission
        rather than as one rejection at submit time. A 4,000-item Haiku run came
        back 100% errored on "This model does not support the effort parameter",
        which one request would have caught in seconds.

        Costs a single item. Raises whatever the API raises.
        """
        self._guard(
            lambda: self._client.messages.create(**_request_params(item, self.model))
        )

    def submit_batch(self, items: Sequence[CatalogueItem]) -> str:
        """Submit a batch and return its id. Does not wait."""
        requests = [
            {"custom_id": item.item_id, "params": _request_params(item, self.model)}
            for item in items
        ]
        batch = self._guard(
            lambda: self._client.messages.batches.create(requests=requests)
        )
        return batch.id

    def batch_status(self, batch_id: str) -> str:
        return self._guard(
            lambda: self._client.messages.batches.retrieve(batch_id)
        ).processing_status

    def collect_batch(
        self, batch_id: str, items: Sequence[CatalogueItem]
    ) -> tuple[list[Enrichment], list[str]]:
        """Read a finished batch.

        Returns (enrichments, failed_item_ids). Results arrive in arbitrary
        order, so everything is matched by custom_id.
        """
        by_id = {item.item_id: item for item in items}
        enrichments: list[Enrichment] = []
        failed: list[str] = []

        results = self._guard(lambda: self._client.messages.batches.results(batch_id))
        for entry in results:
            item = by_id.get(entry.custom_id)
            if item is None:
                continue
            if entry.result.type != "succeeded":
                failed.append(entry.custom_id)
                continue
            try:
                text = "".join(
                    b.text for b in entry.result.message.content if b.type == "text"
                )
                enrichments.append(_parse(text, item, self.model))
            except (json.JSONDecodeError, KeyError, AttributeError):
                failed.append(entry.custom_id)

        return enrichments, failed
