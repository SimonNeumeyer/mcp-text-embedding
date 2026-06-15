"""myembed — a keyed semantic text-embedding store + MCP server."""

from .store import EmbeddingStore, ModelMismatch

__all__ = ["EmbeddingStore", "ModelMismatch"]
