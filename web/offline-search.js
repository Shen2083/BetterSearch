/* Offline search for the standalone catalogue file.
 *
 * NOT loaded by the served page. `scripts/build_standalone.py` inlines this
 * file, plus a baked data blob, into a copy of catalogue.html so the result
 * works from file:// with no server.
 *
 * Only retrieval needs Python - embedding the query, the vector index, BM25.
 * That part is precomputed at build time into RANKINGS: for each example query
 * and mode, the ordered [doc_id, score] list the Searcher returned. Everything
 * after retrieval is plain data work, so it is ported here from
 * api/catalogue.py and runs live: facet counts, filtering and pagination all
 * behave exactly as they do against the real API.
 *
 * Keep this in step with api/catalogue.py - _FACET_SPEC, build_facets,
 * matches_filters and to_card are mirrored below and must agree, or the
 * standalone file will quietly disagree with the served one.
 */
(function () {
  "use strict";

  const RECORDS = window.__OFFLINE_RECORDS__;
  const RANKINGS = window.__OFFLINE_RANKINGS__;

  // Match on a loosened form of the query, so the comma in the visible label
  // "gentle crime novels, nothing too gory" still finds the baked ranking.
  // The key separator is a plain "|" deliberately: a normalised query is only
  // [a-z0-9 ], so "|" cannot collide, and unlike an exotic separator it
  // survives the HTML parser. A NUL here does not - the tokenizer rewrites
  // U+0000 in script data to U+FFFD, and every lookup then misses silently.
  const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

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
      // Python: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
      const ordered = [...counter.entries()].sort(
        (a, b) => b[1] - a[1] || a[0].localeCompare(b[0])
      );
      facets.push({
        key,
        label,
        values: ordered.map(([value, count]) => ({ value, count })),
      });
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

  // ---- the same shape /catalogue/search returns ---------------------------
  function search(query, mode, filters, page, perPage) {
    const ranking = RANKINGS[norm(query) + "|" + mode];
    if (!ranking) return null; // not a baked query - caller explains

    // Repeated occurrences of a weekly session were already collapsed by
    // run_search at build time - a drop-in that happens every Tuesday is one
    // thing to show, not seven - so the baked ranking holds one record per
    // series and carries the occurrence count as a third element. Collapsing
    // again here would be a second implementation of the same rule, free to
    // drift from the first.
    const ranked = ranking
      .map(([docId, score, repeats]) =>
        [RECORDS[docId] && { ...RECORDS[docId], repeats: repeats || 1 }, score])
      .filter(([record]) => record);

    // Facets are counted before filtering, so the sidebar shows what you could
    // narrow to rather than only what is already selected.
    const facets = buildFacets(ranked.map(([record]) => record));

    const kept = ranked.filter(([record]) => matchesFilters(record, filters));
    const total = kept.length;
    const start = (page - 1) * perPage;
    const window_ = kept.slice(start, start + perPage);

    return {
      query,
      mode,
      total,
      page,
      per_page: perPage,
      showing: window_.length ? [start + 1, start + window_.length] : [0, 0],
      results: window_.map(([record, score], i) => toCard(record, score, start + i + 1)),
      facets,
    };
  }

  function knownQueries() {
    return [...new Set(Object.keys(RANKINGS).map((k) => k.split("|")[0]))];
  }

  // The corpus metadata the served page fetches from /catalogue/meta. Baked in
  // here because there is no service behind this file.
  function meta() {
    return window.__OFFLINE_META__ || { presets: [], description: "" };
  }

  window.OFFLINE = { search, knownQueries, meta };

  // Say what this copy is, so nobody mistakes an unanswerable query for an
  // empty catalogue. The page fills .footnote from the corpus description
  // after this script runs, so wait for it rather than appending to an empty
  // paragraph that is about to be overwritten.
  const NOTE =
    "This is a self-contained offline copy: the searches below are" +
    " precomputed, so it answers the example queries only. The full version" +
    " runs every query live against the search service.";
  const footnote = document.querySelector(".footnote");
  if (footnote) {
    new MutationObserver((_, observer) => {
      if (footnote.textContent.includes(NOTE)) return;
      footnote.innerHTML += "<br>" + NOTE;
      observer.disconnect();
    }).observe(footnote, { childList: true, characterData: true, subtree: true });
  }
})();
