"""Keyed semantic text-embedding store.

All behavior lives here so the MCP server (and the original CLI) are thin
wrappers. The store is a single `.npz` keyed map:
  - ids:        object array of str keys
  - embeddings: (N, D) float32 matrix, row i = embedding of ids[i]
  - model:      SentenceTransformer name that produced the vectors
  - revision:   HF model revision ("" if unpinned)

The `model`/`revision` are persisted so the *same mechanism* is reused on every
add, and a mismatch against the running config is rejected (old and new vectors
would otherwise live in incomparable spaces).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class ModelMismatch(Exception):
    """Raised when a store's pinned model/revision differs from the config."""


class EmbeddingStore:
    def __init__(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        model: str,
        revision: str | None = None,
    ):
        self.ids = list(ids)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.ids:
            embeddings = embeddings.reshape(len(self.ids), -1)
        self.embeddings = embeddings
        self.model = model
        self.revision = revision or None
        self._encoder = None  # lazy: pure --id queries skip the model load

    # --- persistence -----------------------------------------------------
    @classmethod
    def load(
        cls, path: str | Path, model: str, revision: str | None = None
    ) -> "EmbeddingStore":
        path = Path(path)
        if not path.exists():
            # fresh store; mechanism is pinned to (model, revision)
            return cls([], np.empty((0, 0), np.float32), model, revision)

        data = np.load(path, allow_pickle=True)
        stored_model = str(data["model"])
        stored_rev = str(data["revision"]) if "revision" in data else ""
        stored_rev = stored_rev or None

        if stored_model != model:
            raise ModelMismatch(
                f"store {path} was built with model {stored_model!r} but config "
                f"requests {model!r}; point MYEMBED_STORE_PATH elsewhere or rebuild"
            )
        if revision and stored_rev and revision != stored_rev:
            raise ModelMismatch(
                f"store {path} pinned revision {stored_rev!r} but config requests "
                f"{revision!r}"
            )
        return cls(list(data["ids"]), data["embeddings"], stored_model,
                   stored_rev or revision)

    def save(self, path: str | Path) -> None:
        """Atomic write: serialize to a temp file, then rename into place."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # Pass a file handle so np.savez does not append its own .npz suffix.
        with open(tmp, "wb") as f:
            np.savez(
                f,
                ids=np.array(self.ids, dtype=object),
                embeddings=self.embeddings,
                model=self.model,
                revision=self.revision or "",
            )
        os.replace(tmp, path)

    # --- embedding mechanism ---------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model, revision=self.revision)
        return self._encoder.encode(text, convert_to_numpy=True).astype(np.float32)

    # --- component 1: add + persist --------------------------------------
    def add(self, key: str, text: str, overwrite: bool = False) -> np.ndarray:
        if key in self.ids and not overwrite:
            raise KeyError(f"id {key!r} already exists (use overwrite=True)")
        vec = self.encode(text)
        if key in self.ids:  # overwrite path
            self.embeddings[self.ids.index(key)] = vec
        else:
            if self.embeddings.size == 0:
                self.embeddings = vec.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, vec])
            self.ids.append(key)
        return vec

    # --- component 2: closest-k use case ---------------------------------
    def closest(
        self, vec: np.ndarray, k: int, exclude: str | None = None
    ) -> list[tuple[str, float]]:
        if len(self.ids) == 0:
            return []
        vec = vec.reshape(-1)
        mat = self.embeddings
        sims = (mat @ vec) / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-12
        )
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for i in order:
            if exclude is not None and self.ids[i] == exclude:
                continue
            out.append((self.ids[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    def vector_for(self, key: str) -> np.ndarray:
        if key not in self.ids:
            raise KeyError(f"id {key!r} not in store")
        return self.embeddings[self.ids.index(key)]

    # --- introspection ---------------------------------------------------
    @property
    def dim(self) -> int | None:
        return int(self.embeddings.shape[1]) if self.ids else None
