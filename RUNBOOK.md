# Runbook

How to run, verify and deploy BetterSearch. Every number below was measured, not
estimated.

---

## 1. Run it locally (5 minutes)

```bash
git clone https://github.com/Shen2083/BetterSearch && cd BetterSearch
pip install -e ".[dev,local,api]"

bettersearch ingest --corpus data/corpus_seed.json
```

The first `ingest` downloads the embedding model (~80MB) and takes about 20
seconds for 112 documents. Later runs re-embed nothing.

> **A fresh clone has no index.** `.bettersearch/` is gitignored, so you must
> `ingest` before searching. Skip it and you get
> `Error: Index is empty - run 'bettersearch ingest' first.` rather than
> silently wrong results.

---

## 2. Verify it works

Four checks, in order of how much they tell you.

### The demonstration

```bash
bettersearch compare "Norman conquest"
```

**Expected:** the KEYWORD column is empty. The SEMANTIC column returns *The
Battle of Hastings*, *Harold Godwinson*, *Motte-and-Bailey Castles*, *William,
Duke of Normandy*. None of those articles contain the word "Norman" or
"conquest".

If the keyword column is empty and the semantic column is right, the core thesis
is working.

### The measurement

```bash
bettersearch evaluate
```

**Expected**, within a few thousandths:

```
mode        Recall@5    MRR@10   nDCG@10
keyword        0.355     0.625     0.415
semantic       0.841     0.984     0.890
hybrid         0.624     0.826     0.741
```

Semantic should roughly double keyword on nDCG. Hybrid scoring *below* semantic
is expected here and is not a bug — see the README on why.

To re-derive where the catalogue page cuts its result list, against the 4,000
real records and their judgements:

```bash
python scripts/tune_cutoff.py thin=.bettersearch/real \
    enriched=.bettersearch/real-enriched
```

**Expected**: `fixed top-20` is the best measured F1 on both arms, and the script
says so itself. It also marks the rows it cannot judge — any cut returning more
than the pool depth of 20 is scored partly against records no judge ever saw, so
its precision is not comparable. `api/catalogue.py:SEMANTIC_TOP_K` holds the
number the script picks; if you change one, re-run the other.

`BETTERSEARCH_TOP_K` overrides it per deployment. 20 is measured for the
4,000-record corpus and is not a universal constant — on a much smaller
catalogue it returns a sizeable fraction of the shelf. Anything you set there is
a guess unless `tune_cutoff.py` produced it from judgements for **that** corpus,
which is the whole reason the previous eyeballed number had to be replaced.

#### Comparing embedding models

The default encoder is `bge-base-en-v1.5` (768 dims, 512 tokens). It replaced
`all-MiniLM-L6-v2` on the numbers below. The encoder is a config change, not a
code change, so trying another is a second index and a second lane:

```bash
BETTERSEARCH_LOCAL_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
BETTERSEARCH_INDEX_PATH=.bettersearch/real-minilm \
    bettersearch ingest --corpus data/catalogue_real.json
```

**Changing the model invalidates every existing index.** The index records the
model it was built with and refuses a write from a different one — `--force`
does not override it, because a silent mismatch would write vectors from the
wrong model into a live index. Delete the `.json` and `.npz` for that index
path and re-ingest.

**An arm that was not in the pool cannot be compared.** `data/eval_real.json`
was pooled from three MiniLM and BM25 lanes; scoring a bge index against it
looked fair and was not, because 29% of what bge returned had never been shown
to a judge and unjudged counts as irrelevant. It cost bge 0.080 nDCG — enough
to invert the conclusion. Re-pool before comparing:

```bash
python scripts/build_eval_set.py \
    --lane "keyword=.bettersearch/real:keyword" \
    --lane "semantic-thin=.bettersearch/real:semantic" \
    --lane "semantic-enriched=.bettersearch/real-enriched:semantic" \
    --lane "semantic-bge=.bettersearch/real-bge:semantic@BAAI/bge-base-en-v1.5"
```

`@model` tells a lane which encoder its index was built with; without it every
lane shares the configured default and the mismatch either raises or, if the
dimensions happen to agree, silently scores one index with another's vectors.
`scripts/compare_arms.py` takes the same suffix.

**Expected**: roughly 2,378 judgements, each lane contributing records no other
lane found — if a new lane contributes none, it is not adding information and
the pool did not need it.

#### Pooling an arm that is not an index and a mode

A reranked lane is a retrieval plus a cross-encoder plus a depth, so there is no
index path that expresses it. Give it as precomputed rankings instead —
`{query: [doc_id, ...]}`, the shape `tune_reranking.py --dump` writes and
`score_rankings.py` reads:

```bash
python scripts/build_eval_set.py --per-lane 20 \
    --lane-file "rerank-bge-200=data/rankings/rerank-bge-200.json" \
    --lane-file "rerank-bge-200-promoted=data/rankings/rerank-bge-200-promoted.json"
```

Those two files are committed, because the judgements are only auditable if the
pool they were made over can be reconstructed — and a reranked lane has no index
path that re-derives it. `data/rankings/README.md` says what each arm is and how
to regenerate the full depth-200 dump (about 25 minutes of CPU, no network, no
key):

```bash
python scripts/tune_reranking.py --depths 200 --rerankers bge --dump /tmp/lane200
```

The dump is depth-long; `--per-lane` cuts it to the top K, in rank order. Any
future system — a Vespa ranking profile included — becomes poolable by dumping
its rankings in that shape, with no code change here.

**Expected** on the two lanes above: `399 new to judge`, about **$0.06** at
Haiku 4.5 batch rates, against 1,979 grades carried forward (that run has since
been made, so a repeat carries 2,378 forward and finds nothing new). The
per-lane line
reads `399 of 820 retrieved pairs were never judged` — that, not `unique_to`, is
the number that says whether a lane was worth pooling.

#### The pool only ever grows

`--reuse` (on by default, pointing at `data/eval_real.json`) carries existing
grades forward and judges only pairs no judge has seen, which is both cheaper
and the only way published figures stay comparable. Two things it does that are
worth understanding before you trust a re-pooled number.

**Reuse is keyed on the query's text, not its position.** The judgements file
keys grades `q{index}--{doc_id}`. Re-pooling invites editing the query list, and
if reuse read those keys, inserting one query would shift every index after it
and silently attach grades to the wrong queries. The ground truth would be
corrupt and nothing would look wrong.

**Records already judged stay in the pool even if no current lane retrieves
them** (`--no-keep-judged` turns this off, and you almost certainly do not want
to). Without it, a record the old lanes found and the new ones miss would drop
out of `relevant_doc_ids`, shrinking the denominator every published figure was
measured against — so every arm would appear to improve while nothing had. A
judgement is paid for and permanent.

A useful consequence: the lanes that built the previous pool do **not** have to
still exist on disk to keep contributing to it. `lanes_pooled` lists them
alongside this run's lanes, and `lanes_run` lists only the ones actually
executed. So the command above is reproducible even though `.bettersearch/real`,
`real-enriched` and `real-bge` are gitignored and do not survive a fresh clone.

To check reuse is wired up correctly, run it with no lanes at all:

```bash
python scripts/build_eval_set.py --validate 0 --out /tmp/inert.json
```

**Expected**: `2378 grades carried forward · 0 new to judge`, and `/tmp/inert.json`
identical to `data/eval_real.json` on every query's `relevant_doc_ids`, `grades`
and `note`. Only `judge_self_consistency` (null, since no judge ran),
`lanes_run` (empty) and one sentence of `description` may differ. If anything
else moves, reuse is mismatching keys and no number downstream can be trusted.

Add `--dry-run` to any of these to pool, report how many pairs are genuinely new
and what judging them would cost, and stop before spending anything.

### The tests

```bash
pytest -q      # 59 passed
```

No model, no network, no API key needed: the suite uses a fake embedding
provider and a fake enrichment client.

### The UI

```bash
uvicorn api.main:app --reload      # http://127.0.0.1:8000
```

`GET /health` should report `chunk_count: 112` and the MiniLM `model_id`.

### The catalogue prototype

A library-catalogue version of the same search, for showing to a non-technical
audience. It runs over its own 74-record demonstration catalogue:

```bash
export BETTERSEARCH_INDEX_PATH=.bettersearch/catalogue
bettersearch ingest --corpus data/catalogue_library.json
uvicorn api.main:app --reload      # http://127.0.0.1:8000/catalogue
```

Search `learning to be present` and switch between the two modes. **Catalogue
search returns nothing; Search by meaning returns six records.** Then try
`grewal`: catalogue search mixes a crime novelist into results plainly meant for
a spirituality author, because the two share a surname. Meaning-based search
does not.

The catalogue is a fictional service with invented records, so it can be shown
around without passing as a real library. Rebuild it with
`python scripts/build_catalogue.py`.

#### Events alongside the books

The catalogue can serve more than one collection. `--corpus` is repeatable, and
`BETTERSEARCH_CATALOGUE` takes a comma-separated list:

```bash
BETTERSEARCH_INDEX_PATH=.bettersearch/real-events \
    bettersearch ingest --corpus data/catalogue_real.json \
                        --corpus data/events_northfield.json

BETTERSEARCH_CATALOGUE="data/catalogue_real.json,data/events_northfield.json" \
BETTERSEARCH_INDEX_PATH=.bettersearch/real-events \
    uvicorn api.main:app --port 8000
```

**Expected**: `Documents 4114`. Try `getting my toddler to eat vegetables` — the
Toddler Mealtimes Workshop should be the first result, above the books — and
`somewhere to go on a Tuesday afternoon`, which should return six events, each
card saying how many other dates the session runs.

This uses the default encoder, `bge-base-en-v1.5`, and so does the standalone
page — see below. **If a query ranks differently here and in the file you send
someone, the two are not on the same index**; check `GET /health` for the
`model_id` before suspecting anything subtler.

Rebuild the events with `python scripts/build_events.py`. They are invented, and
the corpus file's own description says so; the standalone build reads that
description into the page footnote, so it cannot be shipped without the caveat.

To re-check that events are findable and not intrusive:

```bash
python scripts/check_events.py
```

**Expected**: 12 of 12 event-shaped queries reach an event, 0 of 5 known-item
lookups show one, and the script concludes that one index is enough. If reach
drops below 60% it says so and exits non-zero — that is the signal to retrieve
each collection separately and fuse by rank with
`bettersearch.fusion.reciprocal_rank_fusion`, which is unused today.

#### Showing it to someone without a server

`web/catalogue.html` is only a front end — every search POSTs to
`/catalogue/search`, so opening it from disk gives "Failed to fetch" (Safari
says "Load failed"). **`docs/catalogue-standalone.html` is the file to send.**
It opens on a double-click, works offline, needs no Python — and answers any
query typed into it, not a fixed list.

The whole corpus, its embedding vectors and the search all live in the file.
Keyword search is BM25 in JavaScript, ported from `src/bettersearch/keyword.py`.
Searching by meaning embeds the query in the browser with transformers.js.

**The page runs the same encoder as the server.** Corpus vectors and query must
come from the same model, so the page loads the ONNX build of `bge-base-en-v1.5`
— 110 MB, fetched on the first meaning search and then cached. Nothing about the
API changes.

This section used to say the page ran a smaller model *and had to*, because
`bge-base` was "too large to send to a browser". Nobody had looked at the file.
The quantised build is 110 MB against `bge-small`'s 34 MB — three times, not the
order of magnitude the claim assumed — and the price of the assumption was that
the demo you email behaved differently from the service you tested, with no
error anywhere to say so. It is also the weaker model: nDCG@10 0.610 against
0.658.

```bash
BETTERSEARCH_INDEX_PATH=.bettersearch/real-events-base \
    bettersearch ingest --corpus data/catalogue_real.json \
                        --corpus data/events_northfield.json

python scripts/build_standalone.py \
    --catalogue "data/catalogue_real.json,data/events_northfield.json" \
    --index .bettersearch/real-events-base
```

**Expected**: `4114 records, 4114 chunks`, and a file of about **7.0 MB**. The
build fails over 8 MB rather than quietly shipping something nobody will open
on a train — so the headroom is now about 1 MB, and a much larger corpus needs
either a smaller encoder or a different plan.

A deliberately lighter page is still one flag away — `--model
BAAI/bge-small-en-v1.5` against a `bge-small` index — at 4.8 MB and a 34 MB
download. Just do not then compare it to a `bge-base` server.

The vectors are taken out of the index rather than recomputed. That is the
point: re-embedding in the builder would be a second path to "the same"
numbers, and the day it drifted the page would rank differently from the API
with nothing to show for it. `--model` must name the model the index was built
with; the build refuses a mismatch, because scoring a query from one encoder
against vectors from another returns confident nonsense rather than an error.

#### Checking the page agrees with the API

The page is a second implementation of retrieval, so it gets checked against
the first over the whole eval set:

```bash
python scripts/check_browser_parity.py
```

**Expected**, measured on `bge-base` over the 41 judged queries:

| | browser | server | |
|---|---|---|---|
| top-20 overlap | — | — | **93.5%** mean, 70% worst |
| nDCG@10, meaning | 0.645 | 0.651 | **−0.006** |
| nDCG@10, keyword | 0.439 | 0.439 | **0.000** |
| ordering, keyword | — | — | **identical on 41/41** |
| promotion labels (`--blend`) | — | — | **50/50 records agree** |
| explanations | — | — | 85.3% of shared records |
| top-20 overlap, `--filter` | — | — | **94.9%** mean, 85% worst |
| ordering, keyword + `--filter` | — | — | **identical on 41/41** |

Run it with `--filter "format=Large print"` as well. Filtering now happens
*before* the cut on both sides, and the order is the kind of thing that is easy
to get right in one implementation and wrong in the other; an unfiltered run
cannot see that. The keyword lane with a filter is the sharpest check of all —
it is model-free, so anything short of 41/41 identical ordering is a porting
bug rather than quantisation.

Run the keyword and blended lanes too — `--mode keyword` and `--blend` — because
they are the model-free parts and **exact is the bar**: BM25 and exact matching
are string work, so anything short of 41/41 and 100% is a porting bug, not
quantisation.

Only two things stop the meaning lane being 100%, both measured rather than
waved away: int8 corpus vectors (98.8% overlap on their own) and the browser's
quantised ONNX model against fp32 Python, which accounts for the rest. That
second one is the price of a 110 MB download instead of 440 MB, and it does not
move quality — 0.006 nDCG, in the browser's favour on some queries.

Explanation agreement is a little lower than it was on `bge-small` (85.1%
against ~89%). That is the expected direction: 768 dimensions give near-tied
subject headings more places to cross over under quantisation than 384 do. It
is churn at the margin, not a porting fault — which is exactly why the
model-free lanes are checked separately, where no such excuse exists.

If overlap ever collapses rather than drifting, suspect the page and the index
no longer share a model.

#### Publishing it

`.github/workflows/pages.yml` rebuilds the file on pushes to `main` that touch
the corpus, the page or the search code, and publishes `docs/` to GitHub Pages.

**One repository setting has to be switched by hand**: Settings → Pages →
Build and deployment → Source → **GitHub Actions**. Until then the deploy step
fails with "Pages site not found" however green the build is.

---

## 3. Deploy to Render

```
Render dashboard → New → Blueprint → select this repo → Apply
```

`render.yaml` does the rest. First build takes roughly 5-10 minutes, mostly
installing torch and embedding the corpus.

**Verify the deploy:**

1. `GET https://<your-service>.onrender.com/health` returns `chunk_count: 112`.
2. Open the root URL and search `Norman conquest` — same result as local.
3. Check the build log shows the CPU-only torch install and **no** `nvidia-*`
   packages.

### Why the blueprint looks the way it does

Three decisions that are easy to get wrong:

| Decision | Reason |
|---|---|
| CPU-only torch installed first | The default wheel pulls ~2.7GB of CUDA packages that are useless on a CPU host |
| `bettersearch ingest` in the **build** command | Render's filesystem is ephemeral and the index is gitignored — built at runtime it would vanish, and the service would boot empty |
| `plan: standard` | Measured 844MB resident. Free and Starter are both 512MB and will OOM |

### The cost tradeoff

| Runtime shape | Memory | Render plan | Cost |
|---|---|---|---|
| Local MiniLM (default) | **844 MB** | Standard | ~$25/month |
| API embeddings | **32 MB** | Free | £0 hosting, pennies per ingest |

The API shape is small because the index is precomputed — only the *query* needs
embedding at request time, so no model sits in memory. The commented-out second
service in `render.yaml` is that configuration.

Choosing it means no self-hosting, but it sends your content to a third party
and adds a vendor dependency. For a corpus that is commercially or personally
sensitive, the $25 is usually the right trade.

**If you switch providers, re-ingest.** An index built by MiniLM will refuse an
OpenAI-embedded query — by design, because the vectors are not comparable.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Index is empty - run 'bettersearch ingest' first` | No index (fresh clone, or ingest missing from the build) | Run `ingest`; on Render check it is in `buildCommand` |
| `Index was built with X but the query was embedded with Y` | Embedding provider or model changed after indexing | Re-ingest, or set the provider back. This error is deliberate — it prevents silently meaningless results |
| Deploy OOMs or restarts on boot | 844MB model on a 512MB plan | Use `plan: standard`, or switch to API embeddings |
| Build image is several GB | CUDA torch got installed | Ensure CPU-only torch installs **before** `pip install -e` |
| `chunk(s) exceed this provider's input limit` | Chunks longer than the model's window | Lower `BETTERSEARCH_CHUNK_TOKENS`, or use a longer-context model |
| Enrichment: `No Anthropic credentials found` | `bettersearch enrich` needs a key | Set `ANTHROPIC_API_KEY`. Only enrichment needs one — search does not |

---

## 5. Regenerating the documents

Three PDFs in `docs/` are generated artefacts, committed so they can be sent to
someone without a checkout. They are built with the headless Chromium already on
the box — its print engine, not a Python PDF library, so the documents get real
fonts and controlled page breaks.

```bash
python scripts/render_pdf.py docs/approach-technical.html
python scripts/render_pdf.py docs/approach-client.html
python scripts/render_pdf.py README.md --out docs/README.pdf
python scripts/render_pdf.py docs/ARCHITECTURE.md --out docs/ARCHITECTURE.pdf
python scripts/render_pdf.py docs/VESPA-HANDOVER.md --out docs/VESPA-HANDOVER.pdf
python scripts/build_standalone.py          # docs/catalogue-standalone.html
```

The two approach documents are written as HTML and rendered directly.
`docs/README.pdf`, `docs/ARCHITECTURE.pdf` and `docs/VESPA-HANDOVER.pdf` are
converted from their markdown — **rebuild them after any substantive edit**, because nothing forces it and a
stale PDF is not visible in a diff.

**Mermaid renders.** A fenced ```mermaid block is drawn to SVG inside the print
browser, so README keeps its diagrams as diffable text and still produces a
clean PDF. The library is pinned and cached in `.cache/` on first use, so only
the first build needs the network — and a document with no diagram fetches
nothing at all. If a diagram fails to parse the build **fails** rather than
printing its source: that failure mode is invisible in a PDF, which is how
`docs/README.pdf` sat stale for over a week.

`docs/ARCHITECTURE.md` still draws from `docs/architecture/*.svg` rather than
Mermaid. That was originally forced — there was no Mermaid support — and is now
a choice: those three diagrams are hand-laid-out, and Mermaid's automatic layout
does not reproduce them. New diagrams should prefer Mermaid. A relative image
works in both targets: GitHub renders it, and Chromium resolves it against the
intermediate HTML, which is written beside the markdown. Use `--keep-html` to inspect the generated HTML in a browser
while changing the print stylesheet, which lives in `scripts/render_pdf.py`.

Two things the markdown path does deliberately, both of which only matter in a
PDF: it drops images it cannot fetch (the CI badge 403s and would otherwise
print as a broken-image box), and it rewrites relative links such as
`RUNBOOK.md` to full GitHub URLs, so they still work in a file that has been
emailed away from the repository.

The screenshots the README uses are themselves regenerated — see **The catalogue
prototype** above.

---

## 6. What is not verified

Two areas have never run against their real dependencies, so treat them as
untested rather than working:

- **The pgvector backend** — never connected to a real Postgres.
- **The OpenAI and Voyage embedding backends** — construct and validate
  correctly, but have never made an API call.

**Enrichment is now verified against the live API.** A full 112-record run went
through the Batch API on Claude Opus 5: 112 submitted, 112 succeeded, 0 errored,
0 failed, results matched back by `custom_id`. Structured output parsed cleanly
on every record, and the resumability check held — a second `bettersearch enrich`
reports every item skipped. The measured numbers are in the README.
