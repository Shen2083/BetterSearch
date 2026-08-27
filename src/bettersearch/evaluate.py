"""Retrieval evaluation - the part that makes this a PoC rather than a demo.

A single impressive query proves nothing; the question is whether semantic
retrieval wins *on average* and where it loses. Three standard metrics, computed
at the document level (chunks are collapsed to their parent document, because a
user cares that the right article came back, not which chunk of it did):

Recall@5
    Of the relevant documents, what fraction appear in the top 5?
MRR@10
    1 / rank of the first relevant document. Rewards getting one right answer
    to the top.
nDCG@10
    Rank-discounted gain over all relevant documents. The most complete of the
    three, and the one to trust when they disagree.

Expect BM25 to win some rows. Exact names, dates and figures are what term
overlap is *for*; that result is worth having, and it is the argument for
shipping hybrid rather than pure vector search.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .search import MODES, Mode, Searcher
from .types import ScoredChunk


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query: str
    relevant_doc_ids: tuple[str, ...]
    note: str = ""


@dataclass(slots=True)
class ModeMetrics:
    mode: str
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    queries: int

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "recall@5": round(self.recall_at_5, 4),
            "mrr@10": round(self.mrr_at_10, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "queries": self.queries,
        }


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw["queries"] if isinstance(raw, dict) else raw
    return [
        EvalQuery(
            query=r["query"],
            relevant_doc_ids=tuple(r["relevant_doc_ids"]),
            note=r.get("note", ""),
        )
        for r in records
    ]


def _ranked_doc_ids(results: Sequence[ScoredChunk]) -> list[str]:
    """Collapse chunk hits to a de-duplicated document ranking."""
    seen: list[str] = []
    for scored in results:
        if scored.chunk.doc_id not in seen:
            seen.append(scored.chunk.doc_id)
    return seen


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant_set
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant_set), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def evaluate_mode(
    searcher: Searcher,
    queries: Sequence[EvalQuery],
    *,
    mode: Mode,
    top_k: int = 10,
) -> ModeMetrics:
    recalls: list[float] = []
    reciprocals: list[float] = []
    gains: list[float] = []

    for item in queries:
        response = searcher.search(item.query, mode=mode, top_k=top_k)
        ranked = _ranked_doc_ids(response.results)
        recalls.append(recall_at_k(ranked, item.relevant_doc_ids, 5))
        reciprocals.append(reciprocal_rank(ranked, item.relevant_doc_ids, 10))
        gains.append(ndcg_at_k(ranked, item.relevant_doc_ids, 10))

    count = len(queries) or 1
    return ModeMetrics(
        mode=mode,
        recall_at_5=sum(recalls) / count,
        mrr_at_10=sum(reciprocals) / count,
        ndcg_at_10=sum(gains) / count,
        queries=len(queries),
    )


def evaluate_all(
    searcher: Searcher,
    queries: Sequence[EvalQuery],
    *,
    modes: Sequence[Mode] = MODES,
    top_k: int = 10,
) -> list[ModeMetrics]:
    return [
        evaluate_mode(searcher, queries, mode=mode, top_k=top_k) for mode in modes
    ]


def per_query_breakdown(
    searcher: Searcher,
    queries: Sequence[EvalQuery],
    *,
    modes: Sequence[Mode] = MODES,
    top_k: int = 10,
) -> list[dict]:
    """Per-query nDCG for every mode - shows *where* each approach wins."""
    rows: list[dict] = []
    for item in queries:
        row: dict = {"query": item.query, "note": item.note}
        for mode in modes:
            response = searcher.search(item.query, mode=mode, top_k=top_k)
            ranked = _ranked_doc_ids(response.results)
            row[mode] = round(ndcg_at_k(ranked, item.relevant_doc_ids, 10), 3)
        rows.append(row)
    return rows
