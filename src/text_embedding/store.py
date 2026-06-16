"""Keyed semantic text-embedding store.

All behavior lives here so the MCP server (and the CLI) are thin wrappers. One
store is a single `.npz` keyed map (the server keeps one such store per context):
  - ids:        object array of str keys
  - embeddings: (N, D) float32 matrix, row i = embedding of ids[i]
  - metadata:   object array of JSON strings, one key-value map per ids[i]
  - model:      SentenceTransformer name that produced the vectors
  - revision:   HF model revision ("" if unpinned)

Each sample carries an arbitrary metadata map; the class used by `classify` lives
under the `CLASS_KEY` entry of that map (samples without it are simply unclassified).

The `model`/`revision` are persisted so the *same mechanism* is reused on every
add, and a mismatch against the running config is rejected (old and new vectors
would otherwise live in incomparable spaces).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Metadata key under which a sample's class label is stored, read by `classify`.
CLASS_KEY = "class"

# One SentenceTransformer per (model, revision), shared across every store in the
# process so N contexts don't load N copies of the same multi-hundred-MB model.
_ENCODERS: dict[tuple[str, str | None], object] = {}


def _get_encoder(model: str, revision: str | None):
    key = (model, revision)
    if key not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[key] = SentenceTransformer(model, revision=revision)
    return _ENCODERS[key]


class ModelMismatch(Exception):
    """Raised when a store's pinned model/revision differs from the config."""


class EmbeddingStore:
    def __init__(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        model: str,
        revision: str | None = None,
        metadata: list[dict] | None = None,
    ):
        self.ids = list(ids)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.ids:
            embeddings = embeddings.reshape(len(self.ids), -1)
        self.embeddings = embeddings
        # one key-value map per id, parallel to `ids`; {} == no metadata
        if metadata is None:
            self.metadata = [{} for _ in self.ids]
        else:
            self.metadata = [dict(m) if m else {} for m in metadata]
        if len(self.metadata) != len(self.ids):
            raise ValueError("metadata must be the same length as ids")
        self.model = model
        self.revision = revision or None

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
        metadata = (
            [json.loads(str(x)) for x in data["metadata"]]
            if "metadata" in data
            else None
        )

        if stored_model != model:
            raise ModelMismatch(
                f"store {path} was built with model {stored_model!r} but config "
                f"requests {model!r}; point TEXT_EMBEDDING_STORE_DIR elsewhere or rebuild"
            )
        if revision and stored_rev and revision != stored_rev:
            raise ModelMismatch(
                f"store {path} pinned revision {stored_rev!r} but config requests "
                f"{revision!r}"
            )
        return cls(list(data["ids"]), data["embeddings"], stored_model,
                   stored_rev or revision, metadata=metadata)

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
                metadata=np.array([json.dumps(m or {}) for m in self.metadata],
                                  dtype=object),
                model=self.model,
                revision=self.revision or "",
            )
        os.replace(tmp, path)

    # --- embedding mechanism ---------------------------------------------
    def _load_encoder(self):
        # shared across stores with the same (model, revision); lazy so pure --id
        # queries never trigger a model load
        return _get_encoder(self.model, self.revision)

    def encode(self, text: str) -> np.ndarray:
        return self._load_encoder().encode(text, convert_to_numpy=True).astype(np.float32)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts in one batched call -> (len(texts), D) matrix."""
        return self._load_encoder().encode(texts, convert_to_numpy=True).astype(np.float32)

    # --- component 1: add + persist --------------------------------------
    def add(
        self,
        key: str,
        text: str,
        overwrite: bool = False,
        metadata: dict | None = None,
    ) -> np.ndarray:
        if key in self.ids and not overwrite:
            raise KeyError(f"id {key!r} already exists (use overwrite=True)")
        vec = self.encode(text)
        if key in self.ids:  # overwrite path: replaces embedding *and* metadata
            idx = self.ids.index(key)
            self.embeddings[idx] = vec
            self.metadata[idx] = dict(metadata) if metadata else {}
        else:
            if self.embeddings.size == 0:
                self.embeddings = vec.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, vec])
            self.ids.append(key)
            self.metadata.append(dict(metadata) if metadata else {})
        return vec

    def add_many(
        self,
        items: list[tuple[str, str, dict | None]],
        overwrite: bool = False,
    ) -> int:
        """Bulk add `(key, text, metadata)` triples: one batched encode, one vstack.

        The seeding path. Rejects in-batch duplicate keys and (unless `overwrite`)
        keys already in the store *before* encoding, so a bad batch fails fast and
        leaves the store untouched. Returns the number of items added.
        """
        items = list(items)
        if not items:
            return 0
        seen: set[str] = set()
        for key, _, _ in items:
            if key in seen:
                raise KeyError(f"duplicate id {key!r} in batch")
            seen.add(key)
            if key in self.ids and not overwrite:
                raise KeyError(f"id {key!r} already exists (use overwrite=True)")

        vecs = self.encode_many([text for _, text, _ in items])
        new_rows, new_ids, new_meta = [], [], []
        for (key, _, meta), vec in zip(items, vecs):
            meta = dict(meta) if meta else {}
            if key in self.ids:  # overwrite path: replaces embedding *and* metadata
                idx = self.ids.index(key)
                self.embeddings[idx] = vec
                self.metadata[idx] = meta
            else:
                new_rows.append(vec)
                new_ids.append(key)
                new_meta.append(meta)
        if new_rows:
            block = np.vstack(new_rows)
            self.embeddings = block if self.embeddings.size == 0 else np.vstack(
                [self.embeddings, block]
            )
            self.ids.extend(new_ids)
            self.metadata.extend(new_meta)
        return len(items)

    def delete_many(self, keys: list[str]) -> int:
        """Remove ids and their rows/metadata. Validates all keys exist *before*
        mutating, so a bad batch fails fast and leaves the store untouched. Rejects
        in-batch duplicates. Returns the number removed."""
        keys = list(keys)
        if not keys:
            return 0
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                raise KeyError(f"duplicate id {key!r} in batch")
            seen.add(key)
            if key not in self.ids:
                raise KeyError(f"id {key!r} not in store")
        drop = {self.ids.index(k) for k in keys}
        self.embeddings = np.delete(self.embeddings, sorted(drop), axis=0)
        self.ids = [k for i, k in enumerate(self.ids) if i not in drop]
        self.metadata = [m for i, m in enumerate(self.metadata) if i not in drop]
        return len(keys)

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

    # --- component 3: class-probability use case -------------------------
    def class_probabilities(
        self,
        vec: np.ndarray,
        kappa: float = 10.0,
        prior: str = "uniform",
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """von Mises-Fisher kernel density estimate over the classified points.

        A point is classified if its metadata carries a `CLASS_KEY` entry. Weights
        each such point by `exp(kappa * cosine(vec, point))` -- the vMF kernel, the
        spherical analogue of a Gaussian on the unit sphere. Aggregated per class and
        normalized into a posterior over classes:
          - prior="uniform":   posterior(c) proportional to the *mean* weight in c
          - prior="empirical": posterior(c) proportional to the *summed* weight in c
        `kappa` is the concentration/bandwidth (higher -> peakier, more like nearest
        neighbour). Unclassified points and `exclude` are dropped. Returns `(class,
        probability)` ordered most-probable-first, or `[]` if nothing is classified.
        """
        if prior not in ("uniform", "empirical"):
            raise ValueError("prior must be 'uniform' or 'empirical'")
        idx = [
            i
            for i in range(len(self.ids))
            if self.metadata[i].get(CLASS_KEY) is not None and self.ids[i] != exclude
        ]
        if not idx:
            return []
        vec = vec.reshape(-1)
        mat = self.embeddings[idx]
        sims = (mat @ vec) / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-12
        )
        # single global max-subtract keeps exp() finite and cancels on normalization
        weights = np.exp(kappa * (sims - sims.max()))

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for i, w in zip(idx, weights):
            cls_ = self.metadata[i][CLASS_KEY]
            sums[cls_] = sums.get(cls_, 0.0) + float(w)
            counts[cls_] = counts.get(cls_, 0) + 1
        scores = (
            {c: sums[c] / counts[c] for c in sums} if prior == "uniform" else sums
        )
        total = sum(scores.values())
        return sorted(
            ((c, s / total) for c, s in scores.items()), key=lambda t: -t[1]
        )

    # --- component 4: local-density use case -----------------------------
    def density(
        self,
        vec: np.ndarray,
        kappa: float = 10.0,
        radius: float = 0.5,
        exclude: str | None = None,
    ) -> dict:
        """Estimate how crowded the embedding space is around `vec`. Two views:

          - `density`: mean von Mises-Fisher kernel weight `exp(kappa*(cosine - 1))`
            over every stored point -- a smooth estimate in (0, 1]. 1.0 means every
            point sits right on the query; values near 0 mean it is isolated. Same
            kernel as `classify`, but the exponent is anchored at the theoretical max
            cosine of 1 rather than a per-query max, so the number is comparable
            *across* queries -- which is the whole point of a density.
          - `neighbors`: hard count of stored points with cosine >= `radius`, the
            intuitive "how many points fall in a ball around the query" view.

        `kappa` is the concentration/bandwidth (higher -> more local). `exclude`
        (the query's own key) is dropped, and `count` is how many points the
        estimate ranges over. Returns zeros when the context has no other points.
        """
        idx = [i for i in range(len(self.ids)) if self.ids[i] != exclude]
        if not idx:
            return {"density": 0.0, "neighbors": 0, "count": 0,
                    "kappa": kappa, "radius": radius}
        vec = vec.reshape(-1)
        mat = self.embeddings[idx]
        sims = (mat @ vec) / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-12
        )
        return {
            "density": float(np.exp(kappa * (sims - 1.0)).mean()),
            "neighbors": int((sims >= radius).sum()),
            "count": len(idx),
            "kappa": kappa,
            "radius": radius,
        }

    # --- introspection ---------------------------------------------------
    @property
    def dim(self) -> int | None:
        return int(self.embeddings.shape[1]) if self.ids else None

    @property
    def classes(self) -> list[str]:
        """Sorted distinct class values in the store (excludes unclassified)."""
        return sorted(
            {m[CLASS_KEY] for m in self.metadata if m.get(CLASS_KEY) is not None}
        )

    @property
    def num_classified(self) -> int:
        """How many samples carry a CLASS_KEY entry in their metadata."""
        return sum(m.get(CLASS_KEY) is not None for m in self.metadata)
