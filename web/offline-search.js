/* Live search in the browser for the standalone catalogue file.
 *
 * NOT loaded by the served page. `scripts/build_standalone.py` inlines this
 * file, the whole corpus, and the corpus embedding vectors into a copy of
 * catalogue.html, so the result answers **any** query from file:// with no
 * server. Earlier builds baked a fixed list of answers; this one does the
 * retrieval.
 *
 * Both lanes run here, and both are ports of the Python:
 *
 *   keyword   BM25, mirroring src/bettersearch/keyword.py - same parameters,
 *             same tokeniser, same stopwords, same IDF.
 *   meaning   the query is embedded by transformers.js and scored against the
 *             corpus vectors by dot product, mirroring the numpy index.
 *
 * Everything after retrieval - the top-k cut, collapsing repeated event dates,
 * facet counts, filtering, paging - mirrors run_search in api/catalogue.py.
 * Keep the two in step: scripts/check_browser_parity.py compares this file's
 * ranking against the API's over the whole eval set and will say when they
 * have drifted.
 *
 * THE ONE THING THAT MUST NOT CHANGE INDEPENDENTLY
 * The corpus vectors were produced by the model named in the vectors block.
 * The query must be embedded by that same model. Vectors from two different
 * encoders are not more or less similar to each other, they are unrelated, and
 * the page would return confident nonsense rather than failing. The build
 * refuses a mismatch; do not work around it here.
 */
(function () {
  "use strict";

  const readJSON = (id) => {
    const el = document.getElementById(id);
    return el ? JSON.parse(el.textContent) : null;
  };

  const META = readJSON("catalogue-meta") || { presets: [], description: "" };
  const RECORDS = readJSON("catalogue-records") || {};
  const VECTORS = readJSON("catalogue-vectors");
  // Read from api/catalogue.py at build time so the page and the API cannot
  // disagree about how long a result list is.
  const TOP_K = window.__SEMANTIC_TOP_K__ || 20;
  const EXPLAIN_HEADINGS = window.__EXPLAIN_HEADINGS__ || 2;
  // "toggle" keeps the two modes and the button pair; "blended" is one box,
  // always meaning, with named records promoted into it. Read from the build
  // rather than decided here, so one page source serves both demos.
  const INTERACTION = window.__INTERACTION__ || "toggle";
  const PHRASE_TOKENS = window.__PHRASE_TOKENS__ || 3;
  const MAX_PROMOTED = window.__MAX_PROMOTED__ || 3;

  function fromBase64(text) {
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // int8 codes, row-major, with one float32 scale per row (already divided by
  // 127 at build time so scoring is one multiply rather than two).
  const CODES = VECTORS ? new Int8Array(fromBase64(VECTORS.codes).buffer) : null;
  const SCALES = VECTORS ? new Float32Array(fromBase64(VECTORS.scales).buffer) : null;
  const DIMS = VECTORS ? VECTORS.dims : 0;
  const DOC_IDS = VECTORS ? VECTORS.doc_ids : [];

  // ---- status, so the page can say what it is waiting for -----------------
  let onStatus = () => {};
  const setStatus = (state, detail) => onStatus(state, detail || "");

  // ---- BM25, mirrored from src/bettersearch/keyword.py --------------------
  const K1 = 1.5;
  const B = 0.75;
  const STOPWORDS = new Set((
    "a an and are as at be been but by can did do does for from had has have he " +
    "her his how i if in into is it its of on or our that the their them then " +
    "there these they this to was were what when where which who why will with " +
    "you your"
  ).split(" "));

  function tokenize(text) {
    const words = String(text).toLowerCase().match(/[a-z0-9]+/g) || [];
    return words.filter((w) => !STOPWORDS.has(w));
  }

  let bm25 = null;
  function buildBM25() {
    // Built on first keyword search rather than at load: it costs a pass over
    // the whole corpus, and a reader who only ever searches by meaning should
    // not pay for it.
    const ids = Object.keys(RECORDS);
    const frequencies = [];
    const lengths = [];
    const documentFrequency = new Map();
    for (const id of ids) {
      const record = RECORDS[id];
      const terms = tokenize(`${record.title} ${record.text || ""}`);
      const counts = new Map();
      for (const t of terms) counts.set(t, (counts.get(t) || 0) + 1);
      frequencies.push(counts);
      lengths.push(terms.length);
      for (const t of counts.keys()) {
        documentFrequency.set(t, (documentFrequency.get(t) || 0) + 1);
      }
    }
    const total = ids.length;
    const idf = new Map();
    for (const [term, count] of documentFrequency) {
      idf.set(term, Math.log(1 + (total - count + 0.5) / (count + 0.5)));
    }
    const average = lengths.length
      ? lengths.reduce((a, b) => a + b, 0) / lengths.length
      : 0;
    bm25 = { ids, frequencies, lengths, idf, average };
  }

  function keywordRank(query) {
    if (!bm25) buildBM25();
    const terms = tokenize(query);
    if (!terms.length) return [];
    const hits = [];
    for (let i = 0; i < bm25.ids.length; i++) {
      const length = bm25.lengths[i];
      if (length === 0) continue;
      const norm = K1 * (1 - B + (B * length) / (bm25.average || 1));
      let score = 0;
      for (const term of terms) {
        const frequency = bm25.frequencies[i].get(term);
        if (!frequency) continue;
        score += (bm25.idf.get(term) || 0) * ((frequency * (K1 + 1)) / (frequency + norm));
      }
      // Python keeps only positive scores, so a record sharing no term with
      // the query is absent rather than present with score zero. That is what
      // makes a keyword result count mean something.
      if (score > 0) hits.push([bm25.ids[i], score]);
    }
    hits.sort((a, b) => b[1] - a[1]);
    return hits;
  }

  // ---- meaning, via transformers.js ---------------------------------------
  let embedder = null;
  let embedderPromise = null;

  function loadEmbedder() {
    if (embedderPromise) return embedderPromise;
    embedderPromise = (async () => {
      setStatus("loading");
      // Imported at first use, not at page load: it is a ~35 MB download and
      // the keyword lane needs none of it.
      const transformers = await import(
        "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1"
      );
      transformers.env.allowLocalModels = false;
      embedder = await transformers.pipeline(
        "feature-extraction", VECTORS.onnx_repo, { dtype: "q8" }
      );
      setStatus("ready");
      return embedder;
    })().catch((err) => {
      embedderPromise = null; // let a later search try again
      setStatus("error", err && err.message ? err.message : String(err));
      throw err;
    });
    return embedderPromise;
  }

  async function semanticRank(query) {
    if (!VECTORS) return [];
    const pipe = await loadEmbedder();
    // CLS pooling and L2 normalisation, which is how bge was trained and how
    // sentence-transformers encodes it on the server. Mean pooling here would
    // silently produce a different vector for the same words.
    //
    // The server does not prefix the query with bge's retrieval instruction,
    // so this does not either - matching the server matters more, and the
    // instruction was measured at 0.006 nDCG either way.
    const output = await pipe(query, { pooling: "cls", normalize: true });
    const q = output.data;

    // Dot product against int8 rows. Both sides are L2-normalised, so this is
    // cosine similarity, exactly as in the numpy index.
    const count = SCALES.length;
    const scored = new Array(count);
    for (let i = 0; i < count; i++) {
      const base = i * DIMS;
      let total = 0;
      for (let j = 0; j < DIMS; j++) total += CODES[base + j] * q[j];
      scored[i] = [DOC_IDS[i], total * SCALES[i]];
    }
    scored.sort((a, b) => b[1] - a[1]);
    // The cut happens here, on chunks, before anything is collapsed - which is
    // where searcher.semantic(top_k=...) cuts on the server. Collapsing first
    // would quietly return more than TOP_K records.
    //
    // The query vector is handed back rather than stashed in a closure, so
    // that two searches in flight at once cannot explain each other's cards.
    return { hits: scored.slice(0, TOP_K), vector: q };
  }

  // ---- why a record is here, mirroring api/catalogue.py -------------------
  function missingTerms(query, record) {
    const haystack = new Set(tokenize(
      `${record.title || ""} ${record.author || ""} ` +
      `${(record.subjects || []).join(" ")} ${record.text || ""}`
    ));
    const absent = [];
    for (const term of tokenize(query)) {
      if (!haystack.has(term) && !absent.includes(term)) absent.push(term);
    }
    return absent;
  }

  async function closestHeadings(queryVector, records) {
    // Only the page being shown is embedded, and each distinct heading only
    // once - see _closest_headings in api/catalogue.py for why this is a
    // description of the record rather than an explanation of the ranking.
    const vocabulary = [];
    const position = new Map();
    for (const record of records) {
      for (const heading of record.subjects || []) {
        if (!position.has(heading)) {
          position.set(heading, vocabulary.length);
          vocabulary.push(heading);
        }
      }
    }
    if (!vocabulary.length) return records.map(() => []);

    const pipe = await loadEmbedder();
    const output = await pipe(vocabulary, { pooling: "cls", normalize: true });
    const dims = output.dims[output.dims.length - 1];
    const data = output.data;
    const similarity = new Float32Array(vocabulary.length);
    for (let i = 0; i < vocabulary.length; i++) {
      const base = i * dims;
      let total = 0;
      for (let j = 0; j < dims; j++) total += data[base + j] * queryVector[j];
      similarity[i] = total;
    }
    // Array.prototype.sort is stable, as Python's sorted is, so headings of
    // equal similarity keep the order the record lists them in.
    return records.map((record) =>
      (record.subjects || [])
        .slice()
        .sort((a, b) => similarity[position.get(b)] - similarity[position.get(a)])
        .slice(0, EXPLAIN_HEADINGS)
    );
  }

  // ---- mirrored from api/catalogue.py -------------------------------------
  function decade(year) {
    const head = String(year || "").slice(0, 3);
    if (!head || !/^\d{3}$/.test(head)) return null;
    return head + "0s";
  }

  function availability(r) {
    const tags = [];
    if ((r.available || 0) > 0) tags.push("Available now");
    if (r.format === "eAudiobook") tags.push("Available online");
    return tags;
  }

  function recordType(r) {
    return [r.record_type === "event" ? "Events" : "Books"];
  }

  const FACET_SPEC = [
    ["record_type", "Books / Events", recordType],
    ["availability", "Availability", availability],
    ["format", "Format", (r) => (r.format ? [r.format] : [])],
    ["fiction", "Fiction / Non-fiction", (r) => [r.fiction ? "Fiction" : "Non-fiction"]],
    ["decade", "Publication date", (r) => { const d = decade(r.year); return d ? [d] : []; }],
    ["location", "Location", (r) => (r.location ? [r.location] : [])],
    ["language", "Language", (r) => (r.language ? [r.language] : [])],
  ];

  function buildFacets(records) {
    const facets = [];
    for (const [key, label, reader] of FACET_SPEC) {
      const counter = new Map();
      for (const record of records) {
        for (const value of reader(record)) {
          counter.set(value, (counter.get(value) || 0) + 1);
        }
      }
      if (!counter.size) continue;
      const ordered = [...counter.entries()].sort(
        (a, b) => b[1] - a[1] || a[0].localeCompare(b[0])
      );
      facets.push({ key, label, values: ordered.map(([value, count]) => ({ value, count })) });
    }
    return facets;
  }

  function matchesFilters(record, filters) {
    for (const [key, , reader] of FACET_SPEC) {
      const wanted = filters[key];
      if (!wanted || !wanted.length) continue;
      if (!reader(record).some((v) => wanted.includes(v))) return false;
    }
    return true;
  }

  function toCard(record, score, rank) {
    return {
      rank,
      score: Math.round(score * 1e4) / 1e4,
      doc_id: record.doc_id,
      title: record.title,
      author: record.author || "",
      author_dates: record.author_dates || "",
      year: record.year || "",
      publisher: record.publisher || "",
      format: record.format || "",
      record_type: record.record_type || "book",
      audience: record.audience || "",
      when: record.when || "",
      booking: record.booking || "",
      cost: record.cost || "",
      repeats: Number(record.repeats || 1),
      subjects: record.subjects || [],
      blurb: record.blurb,
      location: record.location || "",
      available: Number(record.available || 0),
      copies: Number(record.copies || 0),
      cover: record.cover || "#2E4A62",
      record_text: record.text || "",
    };
  }

  // ---- exact matches, mirroring src/bettersearch/exact.py ----------------
  //
  // Word order is the one signal neither lane sees: BM25 is a bag of words and
  // an embedding blurs proper nouns. These three predicates find records a
  // reader arguably *named* rather than described. Pure string work, so the
  // parity check should come back exact - anything less is a porting bug and
  // not quantisation.
  //
  // The title and author tokens are cut once here rather than on every search,
  // beside the BM25 index which already walks every record at load.
  const EXACT_INDEX = Object.values(RECORDS).map((record) => {
    const title = tokenize(record.title || "");
    const author = tokenize(record.author || "");
    return {
      doc_id: record.doc_id,
      title: title.join(" "),
      author: author.join(" "),
      authorLength: author.length,
    };
  });

  function runs(terms, length) {
    const out = new Set();
    for (let i = 0; i + length <= terms.length; i++) {
      out.add(terms.slice(i, i + length).join(" "));
    }
    return out;
  }

  function findExact(query) {
    const terms = tokenize(query);
    if (!terms.length) return [];
    const asked = terms.join(" ");
    // Stopwords are already gone, so this counts *content* words - `the secret
    // life of bees` is the run `secret life bees`, which is why it does not
    // match The Secret Life of Mermaids.
    const phrases = terms.length >= PHRASE_TOKENS
      ? runs(terms, PHRASE_TOKENS) : new Set();

    const titles = [];
    const authors = [];
    const contained = [];
    for (const entry of EXACT_INDEX) {
      if (entry.title && entry.title === asked) {
        titles.push({ doc_id: entry.doc_id, reason: "this exact title" });
        continue;
      }
      if (entry.authorLength) {
        const named = entry.author === asked
          || runs(terms, entry.authorLength).has(entry.author);
        if (named) {
          authors.push({ doc_id: entry.doc_id, reason: "by this author" });
          continue;
        }
      }
      if (phrases.size && entry.title) {
        for (const phrase of phrases) {
          if (entry.title.includes(phrase)) {
            contained.push({
              doc_id: entry.doc_id, reason: "contains your exact phrase",
            });
            break;
          }
        }
      }
    }
    return titles.concat(authors, contained);
  }

  function promoteExact(ranked, matches) {
    // Inserting is the point rather than a detail: reordering alone could only
    // shuffle what meaning-based search already found, and the case worth
    // fixing is the titled record sitting nowhere on page one. Nothing is
    // dropped - a result at rank 7 moves to rank 8.
    if (!matches.length) return [ranked, {}];
    const reasons = {};
    for (const match of matches.slice(0, MAX_PROMOTED)) {
      if (RECORDS[match.doc_id] && !(match.doc_id in reasons)) {
        reasons[match.doc_id] = match.reason;
      }
    }
    const ids = Object.keys(reasons);
    if (!ids.length) return [ranked, {}];

    const existing = new Map(ranked.map((row) => [row[0].doc_id, row]));
    // Inserted rows score 0: the page never shows a score, and inventing a
    // similarity for a record matched by its title would be making one up.
    const lifted = ids.map((id) => existing.get(id) || [RECORDS[id], 0]);
    const rest = ranked.filter((row) => !(row[0].doc_id in reasons));
    return [lifted.concat(rest), reasons];
  }

  function collapse(hits) {
    // One card per record, then one card per event series: a drop-in that runs
    // every Tuesday is one thing to show, not seven, and `repeats` carries the
    // rest for the card. Same order and same rules as run_search.
    const seen = new Set();
    const seriesAt = new Map();
    const ranked = [];
    for (const [docId, score] of hits) {
      if (seen.has(docId)) continue;
      seen.add(docId);
      const record = RECORDS[docId];
      if (!record) continue;
      const series = record.series_id;
      if (series != null) {
        const at = seriesAt.get(series);
        if (at != null) {
          ranked[at][0] = { ...ranked[at][0], repeats: (ranked[at][0].repeats || 1) + 1 };
          continue;
        }
        seriesAt.set(series, ranked.length);
      }
      ranked.push([record, score]);
    }
    return ranked;
  }

  // ---- the same shape /catalogue/search returns ---------------------------
  async function search(query, mode, filters, page, perPage, blend) {
    // The blended build has no toggle: every search is meaning plus promotion.
    if (INTERACTION === "blended") {
      mode = "semantic";
      blend = true;
    }
    let hits;
    let queryVector = null;
    if (mode === "keyword") {
      hits = keywordRank(query);
    } else {
      ({ hits, vector: queryVector } = await semanticRank(query));
    }
    let ranked = collapse(hits);

    // Blended mode lifts named records above the meaning ranking, before the
    // facets are counted so the sidebar describes the list actually shown.
    let promoted = {};
    if (blend) {
      [ranked, promoted] = promoteExact(ranked, findExact(query));
    }

    // Facets are counted before filtering, so the sidebar shows what you could
    // narrow to rather than only what is already selected.
    const facets = buildFacets(ranked.map(([record]) => record));

    const kept = ranked.filter(([record]) => matchesFilters(record, filters));
    const total = kept.length;
    const start = (page - 1) * perPage;
    const window_ = kept.slice(start, start + perPage);

    const cards = window_.map(([record, score], i) =>
      toCard(record, score, start + i + 1));

    if (mode === "keyword") {
      cards.forEach((card, i) => {
        card.missing_terms = missingTerms(query, window_[i][0]);
      });
    } else if (cards.length) {
      const headings = await closestHeadings(
        queryVector, window_.map(([record]) => record));
      cards.forEach((card, i) => { card.closest_headings = headings[i]; });
    }

    cards.forEach((card) => {
      if (card.doc_id in promoted) card.promoted = promoted[card.doc_id];
    });

    return {
      query,
      mode,
      total,
      page,
      per_page: perPage,
      showing: window_.length ? [start + 1, start + window_.length] : [0, 0],
      results: cards,
      facets,
    };
  }

  function meta() {
    return META;
  }

  function modelInfo() {
    return VECTORS
      ? { model: VECTORS.model, dims: VECTORS.dims, records: Object.keys(RECORDS).length }
      : null;
  }

  window.OFFLINE = {
    search,
    interaction: () => INTERACTION,
    meta,
    modelInfo,
    onStatus: (cb) => { onStatus = cb; },
  };

  // Say what this copy is. The page fills .footnote from the corpus
  // description after this script runs, so wait for that rather than appending
  // to an empty paragraph that is about to be overwritten.
  const NOTE = VECTORS
    ? "This is a self-contained copy: the catalogue, its embeddings and the" +
      " search all run in this page, with no server. Searching by meaning" +
      " downloads the " + VECTORS.model + " model once (about 35 MB) and then" +
      " works offline. The server uses a larger model, so its ranking can" +
      " differ slightly."
    : "This is a self-contained copy, but it was built without corpus vectors," +
      " so only keyword search works.";
  const footnote = document.querySelector(".footnote");
  if (footnote) {
    new MutationObserver((_, observer) => {
      if (footnote.textContent.includes("self-contained copy")) return;
      footnote.innerHTML += "<br>" + NOTE;
      observer.disconnect();
    }).observe(footnote, { childList: true, characterData: true, subtree: true });
  }
})();
