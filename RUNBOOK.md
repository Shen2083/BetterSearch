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
search returns nothing; Search by meaning returns eight records.** Then try
`anand`: catalogue search mixes a crime novelist into results plainly meant for
a spirituality author, because the two share a surname. Meaning-based search
does not.

The catalogue is a fictional service with invented records, so it can be shown
around without passing as a real library. Rebuild it with
`python scripts/build_catalogue.py`.

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
```

The two approach documents are written as HTML and rendered directly.
`docs/README.pdf` is converted from `README.md` — **rebuild it after any
substantive README edit**, because nothing forces it and a stale PDF is not
visible in a diff. Use `--keep-html` to inspect the generated HTML in a browser
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

Three areas have never run against their real dependencies, so treat them as
untested rather than working:

- **Enrichment against the live API** — tested only with a fake client.
- **The pgvector backend** — never connected to a real Postgres.
- **The OpenAI and Voyage embedding backends** — construct and validate
  correctly, but have never made an API call.
