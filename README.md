# BetterSearch

Semantic content search: retrieval by **meaning** rather than by shared words.

Search a content library for `Norman conquest` and get back articles about
*1066*, the *Battle of Hastings* and the *Domesday Book* — none of which
contain the phrase you typed.

```
$ bettersearch compare "Norman conquest"
```

![Side-by-side comparison: the keyword column is empty while the semantic column returns Hastings, Harold Godwinson and William of Normandy](demo.png)

---

## Why this exists

Keyword search (BM25, Postgres full-text, Elasticsearch's default) ranks by term
overlap. If the user's words are not in the document, the document cannot be
found — however obviously relevant it is. That is the entire failure mode this
proof of concept addresses.

Semantic search embeds text as vectors positioned by meaning, so "Norman
conquest" lands near "the invading army of Duke William" without sharing a
single content word.

## Install and run

```bash
pip install -e ".[dev,local,api]"        # local backend: no API key needed
bettersearch ingest --corpus data/corpus_seed.json
bettersearch compare "Norman conquest"   # the demonstration
bettersearch evaluate --per-query        # the measurement
```

Then for the side-by-side UI:

```bash
uvicorn api.main:app --reload            # http://127.0.0.1:8000
```

Nothing above needs an API key, a database, or network access after the first
model download.

---

## The three modes

| Mode | How it ranks | Good at | Blind to |
|---|---|---|---|
| `keyword` | BM25 term overlap | exact names, dates, figures, jargon | anything phrased differently |
| `semantic` | cosine over embeddings | paraphrase, concepts, questions | precise tokens it blurs together |
| `hybrid` | reciprocal rank fusion of both | most real traffic | — |

`hybrid` is the mode you would normally reach for — though on this corpus it
measurably loses to pure `semantic`; see the results below. Fusion uses **rank**,
not raw score:
BM25 scores are unbounded and cosine scores sit in [-1, 1], so blending the
numbers directly needs normalisation constants that must be re-tuned whenever
the corpus or model changes. Ranks are comparable by construction.

---

## Measured results

Run `bettersearch evaluate` to reproduce. 112 documents, 32 hand-labelled
queries, `all-MiniLM-L6-v2` embeddings, numpy index.

| mode | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---|---|
| keyword | 0.355 | 0.625 | 0.415 |
| **semantic** | **0.841** | **0.984** | **0.890** |
| hybrid | 0.624 | 0.826 | 0.741 |

Semantic retrieval more than doubles nDCG over keyword and wins **30 of the 32
queries** outright. On eight queries — including `Norman conquest`,
`why does bread rise`, `father of genetics` and `vaccination` — keyword search
scores exactly **0.000**: not a poor ranking, but no relevant document retrieved
at any depth.

Two illustrative failures worth seeing in the UI:

- `Norman conquest` → BM25 returns **nothing**. No document contains either word.
- `why does bread rise` → BM25 returns **Inflation**, because that article says
  "a sustained *rise* in the general price level". Term overlap without meaning.

### Hybrid loses here, and that is a real result

Fusing a strong retriever with a weak one produced a *worse* ranking than the
strong one alone (0.741 vs 0.890). That is not a bug in the fusion — it is what
RRF does when one input list is mostly empty or noisy, which is the case on an
eval set deliberately weighted towards keyword-hostile phrasing.

The honest reading: **ship `semantic` as the default for this kind of content**,
and revisit `hybrid` only against a query mix with more exact-name, code and
figure lookups, where BM25 earns its half of the fusion. Sample queries where
keyword did contribute: `storing energy in a chemical form` (keyword 0.871 vs
semantic 0.828) and `1066` (0.870 vs 0.876). Run
`bettersearch evaluate --per-query` for the full breakdown.

**Read this before quoting the numbers.** The seed corpus and the eval labels
were written by the same author, so the absolute values are indicative, not
authoritative — the *gap between modes* is the signal. For a non-circular check,
build a corpus you did not write:

```bash
python scripts/fetch_wikipedia.py --count 2000 --out data/corpus_wikipedia.json
bettersearch ingest --corpus data/corpus_wikipedia.json
bettersearch evaluate
```

Note also that the eval set is deliberately weighted towards keyword-hostile
phrasing, because searching for words already in the text is the case keyword
search already handles. Two control queries (`Bayeux Tapestry`, `evolution by
natural selection`) use exact terminology, and BM25 is expected to win those.

---

## Architecture

```
Document ──chunk──► Chunk ──embed──► vector ──► VectorIndex
                      │                              │
                      └──────── BM25 ────────────────┤
                                                     ▼
                                    keyword │ semantic │ hybrid
```

```
src/bettersearch/
  chunking.py         paragraph-aware splitting, overlap, content hashing
  embeddings/         EmbeddingProvider protocol + local / OpenAI / Voyage
  index/              VectorIndex protocol + numpy / pgvector
  keyword.py          BM25 baseline
  fusion.py           reciprocal rank fusion
  search.py           Searcher — the public entry point
  ingest.py           incremental ingestion
  evaluate.py         Recall@5, MRR@10, nDCG@10
  cli.py
api/main.py           FastAPI wrapper (~80 lines)
web/index.html        side-by-side demo page
```

The library is framework-agnostic. `api/` and `cli.py` are thin callers, and
neither is imported by the core.

---

## Configuration

Everything is a `BETTERSEARCH_*` environment variable; no code change is needed
to switch provider or storage.

| Variable | Default | Notes |
|---|---|---|
| `BETTERSEARCH_EMBEDDING_PROVIDER` | `local` | `local` \| `openai` \| `voyage` |
| `BETTERSEARCH_INDEX` | `numpy` | `numpy` \| `pgvector` |
| `BETTERSEARCH_INDEX_PATH` | `.bettersearch/index` | numpy backend only |
| `BETTERSEARCH_DATABASE_URL` | — | pgvector backend; falls back to `DATABASE_URL` |
| `BETTERSEARCH_EMBEDDING_DIMENSIONS` | `512` | Matryoshka width for API providers |
| `BETTERSEARCH_CHUNK_TOKENS` | `450` | target chunk size |
| `BETTERSEARCH_CHUNK_OVERLAP` | `60` | overlap between chunks |

API keys are read from `OPENAI_API_KEY` / `VOYAGE_API_KEY` at the point of use
and never stored by this package.

---

## What actually breaks at scale

The embedding model is not the bottleneck people expect. Three other things are,
and each has a specific countermeasure in the code.

### 1. Changing embedding model invalidates the whole index

Vectors from two models occupy different spaces. Comparing them returns
confident nonsense — the worst kind of bug, because nothing errors.

Every vector is stored with its `model_id` and `dims`, and both index backends
raise `ModelMismatchError` rather than answer a query embedded by a different
model. That column is also what makes a *rolling* re-embed possible: without it,
changing model means taking search offline.

### 2. Re-ingesting must not re-embed everything

Every chunk carries a `content_hash` over its embed text. Ingestion skips
hashes already indexed, so a content update costs only the changed chunks.
Re-run `bettersearch ingest` twice — the second run reports zero embedded.

### 3. Vector *storage* is the real wall

At 1M chunks × 1536 dims × 4 bytes you are holding **6GB** of raw vectors before
index overhead. Three multiplicative fixes, all implemented:

| Lever | Effect |
|---|---|
| Matryoshka truncation (1536 → 512) | 3× smaller |
| `halfvec` fp16 storage | 2× smaller |
| HNSW rather than flat scan | sub-linear query time |

Together: ~6GB → ~500MB, which is the difference between needing a large managed
Postgres instance and fitting comfortably on a small one.

### Choosing a provider for a large library

| Provider | Cost to embed 1M chunks (~400M tokens) | Input limit | Ops burden |
|---|---|---|---|
| `local` MiniLM | £0, but hours of CPU or a GPU box | **256 tokens** | you run and scale it; ~600MB RSS per worker |
| `openai` text-embedding-3-small | ~$8 (~$4 batched) | 8,191 tokens | none |
| `voyage` voyage-3.5-lite | ~$8 | 32,000 tokens | none, but a new vendor |

The local model's **256-token input cap** is the quiet problem for a real
content library: the chunker targets 450 tokens, so the encoder silently
truncates most chunks. `bettersearch ingest` counts and warns about this rather
than letting it degrade recall unnoticed. It is the correct default for a PoC
and the wrong one at scale.

Voyage additionally supports asymmetric encoding (`input_type` of `document` vs
`query`), which is a real recall gain and is wired in.

---

## Using the pgvector backend

```bash
docker compose up -d
export BETTERSEARCH_DATABASE_URL="postgresql://bettersearch:bettersearch@localhost:5433/bettersearch"
export BETTERSEARCH_INDEX=pgvector
bettersearch ingest --corpus data/corpus_seed.json
bettersearch compare "Norman conquest"
```

Results should be identical to the numpy backend. The schema is created on first
write and includes a generated `tsvector` column with a GIN index — the query to
switch to when the corpus outgrows loading every chunk into memory.

---

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /health` | provider, backend, chunk count, `model_id` |
| `POST /search` | `{query, mode, top_k}` → ranked results |
| `POST /compare` | one query, all three modes |

Responses follow `{"status": "success", "data": {...}}`.

### Dropping this into a Django/DRF project

The core library has no framework dependency, so the whole integration is one
view:

```python
# search/views.py
from bettersearch import Searcher
from rest_framework.response import Response
from rest_framework.views import APIView

_searcher = Searcher()          # module-level: loads the model once

class SemanticSearchView(APIView):
    def post(self, request):
        query = request.data.get("query", "").strip()
        if not query:
            return Response({"status": "error", "message": "query is required"}, status=400)
        response = _searcher.search(
            query,
            mode=request.data.get("mode", "hybrid"),
            top_k=int(request.data.get("top_k", 10)),
        )
        return Response({"status": "success", "data": response.to_dict()})
```

Point `BETTERSEARCH_DATABASE_URL` at the existing database and the chunk table
lives alongside the rest of the schema. Embedding work belongs in a Celery task,
not the request cycle.

---

## Scope

Deliberately **not** included: LLM answer generation, query expansion, a
reranker, authentication, and multi-tenant scoping. This PoC answers one
question — does retrieval by meaning beat retrieval by keyword on this content —
and each of those additions would obscure the answer.

## Tests

```bash
pytest
```

Covers chunk boundaries and overlap, hash stability, the incremental-ingest
skip, model-mismatch refusal, BM25 ranking, RRF ordering, and numpy top-k
against hand-computed values. The tests use a deterministic fake embedding
provider, so they need no model, no network and no API key.
