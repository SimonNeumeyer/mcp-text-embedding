"""MCP server exposing the keyed semantic embedding store as tools.

Thin wrapper: every tool delegates to an `EmbeddingStore`. There is one store per
*context* -- every tool takes an obligatory `context` and operates on that context's
own `.npz`. Stores are loaded lazily on first use and cached in `_stores`; a single
global lock guards both the cache and each store op, and writes persist atomically,
so concurrent tool calls can't corrupt a `.npz`. A cached store is reloaded whenever
its `.npz` mtime changes on disk, so edits made out-of-band (e.g. `text-embedding seed`
via the CLI) are picked up on the next tool call -- no server restart required.

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
_mtimes: dict[str, float | None] = {}    # context -> .npz mtime when last loaded

mcp = FastMCP("mcp-text-embedding")


def _get_store(context: str) -> EmbeddingStore:
    """Load-or-reload the store for `context`. Reloads when the context's `.npz`
    mtime differs from when it was last loaded, so out-of-band CLI writes are picked
    up without a restart. Call only while holding `_lock`."""
    path = _cfg.path_for(context)  # validates the context name
    mtime = path.stat().st_mtime if path.exists() else None
    if context not in _stores or _mtimes.get(context) != mtime:
        _stores[context] = EmbeddingStore.load(
            path, model=_cfg.model, revision=_cfg.revision
        )
        _mtimes[context] = mtime
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
        path = _cfg.path_for(context)
        store.save(path)
        _mtimes[context] = path.stat().st_mtime  # our own write; don't reload it next call
        return {
            "added": n,
            "ids": [t[0] for t in triples],
            "total": len(store.ids),
            "context": context,
        }


@mcp.tool()
def closest(
    context: str,
    k: int = 5,
    text: str | None = None,
    id: str | None = None,
    filter: dict | None = None,
) -> list[dict]:
    """Return the keys of the k closest stored embeddings in `context` (cosine).

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own results). Results are ordered
    closest-first as `{"id": ..., "score": ...}`.

    `filter` optionally restricts results to neighbors whose metadata contains
    every key/value pair in `filter` (subset match).
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
            for key, score in store.closest(
                vec, k, exclude=exclude, metadata_filter=filter
            )
        ]


@mcp.tool()
def classify(
    context: str,
    text: str | None = None,
    id: str | None = None,
    kappa: float = 10.0,
    prior: str = "uniform",
    calibrate: bool = True,
) -> dict | list[dict]:
    """Estimate class probabilities for a query within `context` via density estimation.

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own estimate). Scores the `class`
    metadata of stored samples with a von Mises-Fisher kernel density estimate:
    `kappa` is the concentration/bandwidth (higher -> peakier) and `prior` is
    "uniform" (default) or "empirical".

    With `calibrate=True` (default) the ranking is computed in the validated geometry
    transform (de-anisotropised / de-hubbed; see `evaluate`) and the result is a dict
    `{"classes": [{"class", "probability"}, ...], "n_eff", "low_confidence"}`, where
    `n_eff` is the effective number of samples the estimate rests on. With
    `calibrate=False` it returns the raw-cosine `[{"class", "probability"}, ...]` list,
    byte-for-byte with the pre-calibration behaviour. Classes are most-probable-first;
    empty if nothing in the context is classified.
    """
    if (text is None) == (id is None):
        raise ValueError("provide exactly one of `text` or `id`")
    with _lock:
        store = _get_store(context)
        if id is not None:
            vec, exclude = store.vector_for(id), id
        else:
            vec, exclude = store.encode(text), None
        if not calibrate:
            return [
                {"class": cls_, "probability": prob}
                for cls_, prob in store.class_probabilities(
                    vec, kappa=kappa, prior=prior, exclude=exclude, calibrate=False
                )
            ]
        r = store.class_scores(vec, kappa=kappa, prior=prior, exclude=exclude, calibrate=True)
        return {
            "classes": [{"class": c, "probability": p} for c, p in r["classes"]],
            "n_eff": r["n_eff"],
            "low_confidence": r["low_confidence"],
        }


@mcp.tool()
def density(
    context: str,
    text: str | None = None,
    id: str | None = None,
    kappa: float = 10.0,
    radius: float = 0.5,
    calibrate: bool = True,
) -> dict:
    """Estimate how crowded the embedding space of `context` is around a query.

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own estimate). The reference is every
    non-background point. Always returns:
      - `density`: smooth mean kernel weight in (0, 1] -- higher means a crowded region.
      - `neighbors`: count of reference points within `radius` of the query.
      - `count`: number of points the estimate ranges over.
    With `calibrate=True` (default) the estimate is computed in the validated geometry
    transform and the result also carries an honesty layer -- `percentile` (rank of the
    query's local density against the reference), `lof_score`, `n_eff`, `rank_ci`
    (bootstrap 95% CI on the percentile) and `low_confidence`. With `calibrate=False`
    it is the raw-cosine estimate, byte-for-byte with the pre-calibration behaviour.
    `kappa` is the concentration/bandwidth. Useful for spotting novelty/outliers.
    """
    if (text is None) == (id is None):
        raise ValueError("provide exactly one of `text` or `id`")
    with _lock:
        store = _get_store(context)
        if id is not None:
            vec, exclude = store.vector_for(id), id
        else:
            vec, exclude = store.encode(text), None
        return store.density(
            vec, kappa=kappa, radius=radius, exclude=exclude, calibrate=calibrate
        )


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
            "background": store.num_background,
            "classes": store.classes,
            "dim": store.dim,
            "geometry_config": store.geometry_config,
        }


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
