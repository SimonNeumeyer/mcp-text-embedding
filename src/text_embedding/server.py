"""MCP server exposing the keyed semantic embedding store as tools.

Thin wrapper: every tool delegates to an `EmbeddingStore`. There is one store per
*context* -- every tool takes an obligatory `context` and operates on that context's
own `.npz`. Stores are loaded lazily on first use and cached in `_stores`; a single
global lock guards both the cache and each store op, and writes persist atomically,
so concurrent tool calls can't corrupt a `.npz`. Because stores are held in memory,
edits made out-of-band (e.g. `text-embedding seed` via the CLI) are not picked up until the
server restarts.

Uses `fastmcp` (v2), matching the semantic-scholar-mcp repo's convention.
"""

from __future__ import annotations

import threading

from fastmcp import FastMCP

from .config import Config
from .store import EmbeddingStore

_cfg = Config.from_env()
_lock = threading.Lock()
_stores: dict[str, EmbeddingStore] = {}  # context -> store, lazily populated

mcp = FastMCP("mcp-text-embedding")


def _get_store(context: str) -> EmbeddingStore:
    """Load-or-return the store for `context`. Call only while holding `_lock`."""
    if context not in _stores:
        path = _cfg.path_for(context)  # validates the context name
        _stores[context] = EmbeddingStore.load(
            path, model=_cfg.model, revision=_cfg.revision
        )
    return _stores[context]


@mcp.tool()
def add_texts(
    context: str,
    items: list[dict],
    overwrite: bool = False,
) -> dict:
    """Embed and persist `(id, text)` pairs in `context` in one batched call.

    `items` is a list of `{"id": ..., "text": ..., "metadata": {...}}` objects
    (`metadata` optional, free-form; its `class` entry is what `classify` scores).
    For a single add, pass a one-element list. One batched encode and one write.
    The batch is validated up front -- a duplicate id within the batch, or an id
    already present when `overwrite=False`, rejects the *whole* batch and leaves the
    store unchanged. Set `overwrite=True` to replace existing keys (this replaces each
    matched key's whole metadata map too). Returns the number added, the ids, and the
    new total count.
    """
    triples = []
    for i, rec in enumerate(items):
        if not isinstance(rec, dict) or "id" not in rec or "text" not in rec:
            raise ValueError(f"item {i} must be an object with 'id' and 'text'")
        meta = rec.get("metadata", {})
        if not isinstance(meta, dict):
            raise ValueError(f"item {i} 'metadata' must be an object")
        triples.append((str(rec["id"]), str(rec["text"]), meta))
    with _lock:
        store = _get_store(context)
        n = store.add_many(triples, overwrite=overwrite)
        store.save(_cfg.path_for(context))
        return {
            "added": n,
            "ids": [t[0] for t in triples],
            "total": len(store.ids),
            "context": context,
        }


@mcp.tool()
def closest(
    context: str, k: int = 5, text: str | None = None, id: str | None = None
) -> list[dict]:
    """Return the keys of the k closest stored embeddings in `context` (cosine).

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own results). Results are ordered
    closest-first as `{"id": ..., "score": ...}`.
    """
    if (text is None) == (id is None):
        raise ValueError("provide exactly one of `text` or `id`")
    with _lock:
        store = _get_store(context)
        if id is not None:
            vec, exclude = store.vector_for(id), id
        else:
            vec, exclude = store.encode(text), None
        return [
            {"id": key, "score": score}
            for key, score in store.closest(vec, k, exclude=exclude)
        ]


@mcp.tool()
def classify(
    context: str,
    text: str | None = None,
    id: str | None = None,
    kappa: float = 10.0,
    prior: str = "uniform",
) -> list[dict]:
    """Estimate class probabilities for a query within `context` via density estimation.

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own estimate). Scores the `class`
    metadata of stored samples with a von Mises-Fisher kernel density estimate:
    `kappa` is the concentration/bandwidth (higher -> peakier) and `prior` is
    "uniform" (default) or "empirical". Returns `{"class": ..., "probability": ...}`
    ordered most-probable-first; empty if nothing in the context is classified.
    """
    if (text is None) == (id is None):
        raise ValueError("provide exactly one of `text` or `id`")
    with _lock:
        store = _get_store(context)
        if id is not None:
            vec, exclude = store.vector_for(id), id
        else:
            vec, exclude = store.encode(text), None
        return [
            {"class": cls_, "probability": prob}
            for cls_, prob in store.class_probabilities(
                vec, kappa=kappa, prior=prior, exclude=exclude
            )
        ]


@mcp.tool()
def density(
    context: str,
    text: str | None = None,
    id: str | None = None,
    kappa: float = 10.0,
    radius: float = 0.5,
) -> dict:
    """Estimate how crowded the embedding space of `context` is around a query.

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own estimate). Returns a von
    Mises-Fisher kernel density estimate over all stored points:
      - `density`: smooth mean kernel weight in (0, 1] -- higher means the query
        sits in a crowded region; comparable across queries.
      - `neighbors`: count of stored points within cosine `radius` of the query.
      - `count`: number of points the estimate ranges over.
    `kappa` is the concentration/bandwidth (higher -> more local). Useful for
    spotting novelty/outliers and gauging how well a context covers a region.
    """
    if (text is None) == (id is None):
        raise ValueError("provide exactly one of `text` or `id`")
    with _lock:
        store = _get_store(context)
        if id is not None:
            vec, exclude = store.vector_for(id), id
        else:
            vec, exclude = store.encode(text), None
        return store.density(vec, kappa=kappa, radius=radius, exclude=exclude)


@mcp.tool()
def list_keys(context: str) -> list[str]:
    """List all keys currently in `context`."""
    with _lock:
        return list(_get_store(context).ids)


@mcp.tool()
def list_contexts() -> list[str]:
    """List all contexts that currently have a store on disk."""
    return _cfg.list_contexts()


@mcp.tool()
def store_info(context: str) -> dict:
    """Report a context's pinned mechanism, size, and classes."""
    with _lock:
        store = _get_store(context)
        return {
            "context": context,
            "model": store.model,
            "revision": store.revision,
            "count": len(store.ids),
            "classified": store.num_classified,
            "classes": store.classes,
            "dim": store.dim,
        }


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
