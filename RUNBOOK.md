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

## 5. What is not verified

Three areas have never run against their real dependencies, so treat them as
untested rather than working:

- **Enrichment against the live API** — tested only with a fake client.
- **The pgvector backend** — never connected to a real Postgres.
- **The OpenAI and Voyage embedding backends** — construct and validate
  correctly, but have never made an API call.
