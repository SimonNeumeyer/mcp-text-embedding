"""MCP server exposing the keyed semantic embedding store as tools.

Thin wrapper: every tool delegates to an `EmbeddingStore`. There is one store per
*context* -- every tool takes an obligatory `context` and operates on that context's
own `.npz`. Stores are loaded lazily on first use and cached in `_stores`; a single
global lock guards both the cache and each store op, and writes persist atomically,
so concurrent tool calls can't corrupt a `.npz`. Because stores are held in memory,
edits made out-of-band (e.g. `myembed seed` via the CLI) are not picked up until the
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

mcp = FastMCP("myembed")


def _get_store(context: str) -> EmbeddingStore:
    """Load-or-return the store for `context`. Call only while holding `_lock`."""
    if context not in _stores:
        path = _cfg.path_for(context)  # validates the context name
        _stores[context] = EmbeddingStore.load(
            path, model=_cfg.model, revision=_cfg.revision
        )
    return _stores[context]


@mcp.tool()
def add_text(
    context: str,
    id: str,
    text: str,
    overwrite: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Embed `text` with the pinned model and persist it under key `id` in `context`.

    Pass an optional `metadata` map (e.g. `{"class": "animal"}`); the `class` entry
    is what `classify` scores against. Set `overwrite=True` to replace an existing
    key; note this replaces the whole metadata map too, so omitting `metadata` on an
    overwrite clears it. Returns the key, its metadata, and the new total count.
    """
    with _lock:
        store = _get_store(context)
        store.add(id, text, overwrite=overwrite, metadata=metadata)
        store.save(_cfg.path_for(context))
        return {
            "id": id,
            "metadata": metadata or {},
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
