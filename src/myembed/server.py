"""MCP server exposing the keyed semantic embedding store as tools.

Thin wrapper: every tool delegates to `EmbeddingStore`. The store is loaded once
at startup and kept in memory; writes are serialized with a lock and persisted
atomically, so concurrent tool calls can't corrupt the `.npz`.

Uses `fastmcp` (v2), matching the semantic-scholar-mcp repo's convention.
"""

from __future__ import annotations

import threading

from fastmcp import FastMCP

from .config import Config
from .store import EmbeddingStore

_cfg = Config.from_env()
_lock = threading.Lock()
_store = EmbeddingStore.load(_cfg.store_path, model=_cfg.model, revision=_cfg.revision)

mcp = FastMCP("myembed")


@mcp.tool()
def add_text(id: str, text: str, overwrite: bool = False) -> dict:
    """Embed `text` with the pinned model and persist it under key `id`.

    Set `overwrite=True` to replace an existing key. Returns the key and the new
    total count.
    """
    with _lock:
        _store.add(id, text, overwrite=overwrite)
        _store.save(_cfg.store_path)
        return {"id": id, "total": len(_store.ids), "store": str(_cfg.store_path)}


@mcp.tool()
def closest(k: int = 5, text: str | None = None, id: str | None = None) -> list[dict]:
    """Return the keys of the k closest stored embeddings (cosine similarity).

    Provide exactly one of `text` (embed a fresh query) or `id` (use an embedding
    already in the store; it is excluded from its own results). Results are
    ordered closest-first as `{"id": ..., "score": ...}`.
    """
    if (text is None) == (id is None):
        raise ValueError("provide exactly one of `text` or `id`")
    with _lock:
        if id is not None:
            vec, exclude = _store.vector_for(id), id
        else:
            vec, exclude = _store.encode(text), None
        return [
            {"id": key, "score": score}
            for key, score in _store.closest(vec, k, exclude=exclude)
        ]


@mcp.tool()
def list_keys() -> list[str]:
    """List all keys currently in the store."""
    with _lock:
        return list(_store.ids)


@mcp.tool()
def store_info() -> dict:
    """Report the store's pinned mechanism and size."""
    with _lock:
        return {
            "model": _store.model,
            "revision": _store.revision,
            "count": len(_store.ids),
            "dim": _store.dim,
            "store": str(_cfg.store_path),
        }


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
