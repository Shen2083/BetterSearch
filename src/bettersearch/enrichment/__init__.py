"""LLM enrichment for thin catalogue records.

A catalogue record names a shelf position; semantic search needs text that names
concepts. Enrichment manufactures that text, and can state concepts the record
never contains - which is the whole mechanism.

The generated text is a **retrieval bridge and is never displayed**. Match
against it, then show the real record. That one rule demotes a hallucination
from a false statement by the library to a mildly bad search result.
"""

from __future__ import annotations

from .client import CatalogueItem, EnrichmentClient, MissingCredentialError
from .pipeline import EnrichReport, enrich, enriched_documents, pending, to_items
from .schema import PROMPT_VERSION, Enrichment, source_hash
from .store import EnrichmentStore

__all__ = [
    "PROMPT_VERSION",
    "CatalogueItem",
    "EnrichReport",
    "Enrichment",
    "EnrichmentClient",
    "EnrichmentStore",
    "MissingCredentialError",
    "enrich",
    "enriched_documents",
    "pending",
    "source_hash",
    "to_items",
]
