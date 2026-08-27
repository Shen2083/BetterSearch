"""Command line interface.

    bettersearch ingest --corpus data/corpus_seed.json
    bettersearch search "Norman conquest" --mode semantic
    bettersearch compare "Norman conquest"
    bettersearch evaluate
    bettersearch stats
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .config import load_settings
from .evaluate import evaluate_all, load_eval_queries, per_query_breakdown
from .ingest import ingest_corpus, ingest_documents, load_corpus
from .search import MODES, Searcher
from .types import EmptyIndexError, ModelMismatchError

DEFAULT_CORPUS = "data/corpus_seed.json"
DEFAULT_EVAL = "data/eval_queries.json"
DEFAULT_ENRICHMENT_STORE = ".bettersearch/enrichment.jsonl"

_SNIPPET_WIDTH = 96


def _snippet(text: str, width: int = _SNIPPET_WIDTH) -> str:
    flat = " ".join(text.split())
    return textwrap.shorten(flat, width=width, placeholder=" ...")


def _print_results(results, *, indent: str = "  ") -> None:
    if not results:
        print(f"{indent}(no results)")
        return
    for scored in results:
        print(f"{indent}{scored.rank:>2}. [{scored.score:6.3f}] {scored.chunk.title}")
        print(f"{indent}    {_snippet(scored.chunk.text)}")


def _build_searcher() -> Searcher:
    return Searcher(settings=load_settings())


# --------------------------------------------------------------------- ingest
def cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    print(
        f"Provider: {settings.embedding_provider}   "
        f"Index: {settings.index_backend}   Corpus: {args.corpus}"
    )
    report = ingest_corpus(
        args.corpus, settings=settings, force=args.force, progress=args.progress
    )

    print(
        f"\nDocuments {report.documents}   chunks {report.chunks_total}   "
        f"embedded {report.chunks_embedded}   skipped (unchanged) "
        f"{report.chunks_skipped}"
    )
    print(f"Model: {report.model_id}  ({report.dimensions} dims)")
    if report.truncated_chunks:
        print(
            f"\nWarning: {report.truncated_chunks} chunk(s) exceed this provider's "
            f"input limit and were truncated by the encoder. This costs recall - "
            f"use a provider with a larger input window, or reduce "
            f"BETTERSEARCH_CHUNK_TOKENS."
        )
    if report.chunks_embedded == 0 and report.chunks_total:
        print("\nNothing changed - every chunk was already indexed.")
    return 0


# --------------------------------------------------------------------- search
def cmd_search(args: argparse.Namespace) -> int:
    searcher = _build_searcher()
    response = searcher.search(args.query, mode=args.mode, top_k=args.top_k)
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(f"\n{args.mode.upper()}  -  {args.query!r}\n")
    _print_results(response.results)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    searcher = _build_searcher()
    responses = searcher.compare(args.query, top_k=args.top_k)
    if args.json:
        print(
            json.dumps(
                {m: r.to_dict() for m, r in responses.items()},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"\nQuery: {args.query!r}")
    for mode in MODES:
        print(f"\n{'-' * 78}\n{mode.upper()}")
        _print_results(responses[mode].results)
    print()
    return 0


# ------------------------------------------------------------------- evaluate
def cmd_evaluate(args: argparse.Namespace) -> int:
    queries = load_eval_queries(args.queries)
    searcher = _build_searcher()

    metrics = evaluate_all(searcher, queries, top_k=args.top_k)
    print(f"\n{len(queries)} queries\n")
    print(f"{'mode':<10}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@10':>10}")
    print("-" * 40)
    for row in metrics:
        print(
            f"{row.mode:<10}{row.recall_at_5:>10.3f}"
            f"{row.mrr_at_10:>10.3f}{row.ndcg_at_10:>10.3f}"
        )

    if args.per_query:
        print(f"\n{'-' * 78}\nPer-query nDCG@10\n")
        header = f"{'query':<46}" + "".join(f"{m:>11}" for m in MODES)
        print(header)
        print("-" * len(header))
        for row in per_query_breakdown(searcher, queries, top_k=args.top_k):
            label = _snippet(row["query"], width=44)
            print(f"{label:<46}" + "".join(f"{row[m]:>11.3f}" for m in MODES))
    print()
    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    """Generate enrichment for a thin catalogue, then optionally index it."""
    from .enrichment import (
        EnrichmentClient,
        EnrichmentStore,
        enrich,
        enriched_documents,
    )

    settings = load_settings()
    documents = load_corpus(args.corpus)
    store = EnrichmentStore(args.store)
    client = EnrichmentClient(model=args.model)

    report = enrich(
        documents,
        store=store,
        client=client,
        use_batch=not args.sync,
        progress=True,
    )
    print("\n" + json.dumps(report.to_dict(), indent=2))

    if report.failed:
        print(
            f"\n{report.failed} item(s) failed. Re-run to retry only those.",
            file=sys.stderr,
        )
    if report.enriched:
        share = report.recognised / report.enriched
        print(
            f"\nRecognised {report.recognised}/{report.enriched} ({share:.0%}). "
            f"The remaining {report.grounded} were expanded from the record "
            f"alone, with no invented content."
        )

    if args.index:
        rebuilt = enriched_documents(documents, store=store, model=args.model)
        ingest_report = ingest_documents(rebuilt, settings=settings)
        print(
            f"\nIndexed {ingest_report.chunks_embedded} chunks "
            f"({ingest_report.chunks_skipped} unchanged)."
        )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    settings = load_settings()
    searcher = Searcher(settings=settings)
    print(json.dumps(searcher.stats().to_dict(), indent=2))
    return 0


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bettersearch",
        description="Semantic content search - retrieval by meaning, not keywords.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and index a corpus")
    p_ingest.add_argument("--corpus", default=DEFAULT_CORPUS)
    p_ingest.add_argument(
        "--force",
        action="store_true",
        help="re-embed every chunk, ignoring content hashes",
    )
    p_ingest.add_argument("--progress", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="run a single query")
    p_search.add_argument("query")
    p_search.add_argument("--mode", choices=MODES, default="semantic")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_compare = sub.add_parser(
        "compare", help="run one query through all three modes side by side"
    )
    p_compare.add_argument("query")
    p_compare.add_argument("--top-k", type=int, default=5)
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    p_eval = sub.add_parser("evaluate", help="score all modes on the labelled queries")
    p_eval.add_argument("--queries", default=DEFAULT_EVAL)
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument("--per-query", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_enrich = sub.add_parser(
        "enrich", help="generate LLM enrichment for a thin catalogue"
    )
    p_enrich.add_argument("--corpus", default="data/catalogue_thin.json")
    p_enrich.add_argument("--store", default=DEFAULT_ENRICHMENT_STORE)
    p_enrich.add_argument("--model", default="claude-opus-5")
    p_enrich.add_argument(
        "--sync",
        action="store_true",
        help="one request per item instead of the Batch API (half price); "
        "use only for small runs",
    )
    p_enrich.add_argument(
        "--index", action="store_true", help="ingest the enriched documents after"
    )
    p_enrich.set_defaults(func=cmd_enrich)

    p_stats = sub.add_parser("stats", help="describe the current index")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (EmptyIndexError, ModelMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: file not found - {exc.filename}", file=sys.stderr)
        return 1
    except (ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
