"""Thin HTTP wrapper around the bettersearch library.

Deliberately thin: this file holds no retrieval logic. It loads a Searcher once
at startup (so the embedding model is not reloaded per request) and translates
between JSON and the library's own types.

The README shows the equivalent Django/DRF view - the core library has no
framework dependency, so porting this is a single APIView.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bettersearch import Searcher, load_settings
from bettersearch.search import MODES
from bettersearch.types import EmptyIndexError, ModelMismatchError

from .catalogue import load_catalogue, run_search

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="BetterSearch",
    description="Semantic content search - retrieval by meaning, not keywords.",
    version="0.1.0",
)

_settings = load_settings()
# Module-level so the model loads once rather than on every request.
_searcher = Searcher(settings=_settings)
# Display metadata for the catalogue demo. Read once; the library never sees it.
_catalogue = load_catalogue()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["keyword", "semantic", "hybrid"] = "semantic"
    top_k: int = Field(default=10, ge=1, le=50)


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)


class CatalogueRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["keyword", "semantic"] = "keyword"
    page: int = Field(default=1, ge=1, le=100)
    per_page: int = Field(default=10, ge=1, le=50)
    #: facet key -> selected values, e.g. {"format": ["Large print"]}
    filters: dict[str, list[str]] = Field(default_factory=dict)


def _ok(data: dict) -> JSONResponse:
    return JSONResponse({"status": "success", "data": data})


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"status": "error", "message": message}, status_code=status)


@app.exception_handler(EmptyIndexError)
async def _handle_empty_index(request, exc: EmptyIndexError) -> JSONResponse:
    return _error(str(exc), status=409)


@app.exception_handler(ModelMismatchError)
async def _handle_model_mismatch(request, exc: ModelMismatchError) -> JSONResponse:
    return _error(str(exc), status=409)


@app.get("/health")
def health() -> JSONResponse:
    stats = _searcher.stats()
    return _ok(
        {
            "embedding_provider": _settings.embedding_provider,
            "modes": list(MODES),
            "index": stats.to_dict(),
        }
    )


@app.post("/search")
def search(request: SearchRequest) -> JSONResponse:
    response = _searcher.search(
        request.query, mode=request.mode, top_k=request.top_k
    )
    return _ok(response.to_dict())


@app.post("/compare")
def compare(request: CompareRequest) -> JSONResponse:
    responses = _searcher.compare(request.query, top_k=request.top_k)
    return _ok(
        {
            "query": request.query,
            "modes": {mode: r.to_dict() for mode, r in responses.items()},
        }
    )


@app.post("/catalogue/search")
def catalogue_search(request: CatalogueRequest) -> JSONResponse:
    """Catalogue-shaped results: records, facet counts and availability."""
    return _ok(
        run_search(
            _searcher,
            _catalogue,
            query=request.query,
            mode=request.mode,
            filters=request.filters,
            page=request.page,
            per_page=request.per_page,
        )
    )


@app.get("/catalogue")
def catalogue_page() -> FileResponse:
    return FileResponse(WEB_DIR / "catalogue.html")


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
