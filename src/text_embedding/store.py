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

from . import geometry

# Metadata key under which a sample's class label is stored, read by `classify`.
CLASS_KEY = "class"

# Metadata key marking a sample as unlabelled *background*: it feeds the unsupervised
# geometry transform (lifting n for estimation) but is excluded from classify voting
# and from the density reference set. Carries no CLASS_KEY.
BACKGROUND_KEY = "background"

# Calibration only fires once a context holds at least this many points (below it the
# geometry estimate is meaningless and we fall back to the raw cosine, byte-for-byte).
MIN_CALIBRATION_POINTS = 8

# Below this Kish effective sample size the estimate rests on too few points to trust.
MIN_EFFECTIVE_N = 5.0

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
        geometry_config: dict | None = None,
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
        # The pipeline `evaluate` selected for this context (None -> a sensible default
        # at fit time). Persisted so the selection survives the server's mtime reload.
        self.geometry_config = dict(geometry_config) if geometry_config else None
        # Lazily-fit unsupervised transform; invalidated on every mutation.
        self._geometry: geometry.Geometry | None = None

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
        geometry_config = (
            json.loads(str(data["geometry_config"]))
            if "geometry_config" in data and str(data["geometry_config"])
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
                   stored_rev or revision, metadata=metadata,
                   geometry_config=geometry_config)

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
                geometry_config=json.dumps(self.geometry_config or {}),
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
        self._geometry = None  # pool changed; refit lazily on next calibrated query
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
        self._geometry = None  # pool changed; refit lazily on next calibrated query
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
        self._geometry = None  # pool changed; refit lazily on next calibrated query
        return len(keys)

    # --- component 2: closest-k use case ---------------------------------
    def closest(
        self,
        vec: np.ndarray,
        k: int,
        exclude: str | None = None,
        metadata_filter: dict | None = None,
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
            if metadata_filter and not (
                metadata_filter.items() <= self.metadata[i].items()
            ):
                continue
            out.append((self.ids[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    def vector_for(self, key: str) -> np.ndarray:
        if key not in self.ids:
            raise KeyError(f"id {key!r} not in store")
        return self.embeddings[self.ids.index(key)]

    # --- geometry calibration --------------------------------------------
    def _get_geometry(self) -> geometry.Geometry | None:
        """Lazily fit (and cache) the unsupervised transform over *all* stored points.

        Returns None when the context is too small to estimate geometry, so callers
        fall back to the raw cosine. Fit on every point (labelled + background) but
        **never** on labels -- the fit is purely unsupervised. Invalidated on mutation.
        """
        if len(self.ids) < MIN_CALIBRATION_POINTS:
            return None
        if self._geometry is None:
            cfg = self.geometry_config or geometry.DEFAULT_CONFIG
            self._geometry = geometry.fit(self.embeddings.astype(np.float64), cfg)
        return self._geometry

    def _similarities(
        self, vec: np.ndarray, idx: list[int], geo: geometry.Geometry | None
    ) -> np.ndarray:
        """Query-vs-reference similarities for reference rows `idx`.

        Raw cosine when `geo` is None (the `calibrate=False` path, byte-for-byte with
        the original code); otherwise cosine in the transformed space, optionally
        rescored by mutual proximity when the fitted pipeline enables hubness reduction.
        """
        vec = vec.reshape(-1)
        if geo is None:
            mat = self.embeddings[idx]
            return (mat @ vec) / (
                np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-12
            )
        z_query = geo.transform_vec(vec)
        ref_z = geo.transform(self.embeddings[idx])
        base = ref_z @ z_query  # both L2-normalised -> cosine
        if geo.config.get("hubness") and geo.pool_z is not None:
            return geo.mp_similarity(z_query, np.asarray(idx), base)
        return base

    @staticmethod
    def _effective_n(weights: np.ndarray) -> float:
        """Kish effective sample size of a weighted estimate: (sum w)^2 / sum(w^2)."""
        s2 = float((weights**2).sum())
        return float(weights.sum() ** 2 / s2) if s2 > 0 else 0.0

    # --- component 3: class-probability use case -------------------------
    def class_probabilities(
        self,
        vec: np.ndarray,
        kappa: float = 10.0,
        prior: str = "uniform",
        exclude: str | None = None,
        calibrate: bool = True,
    ) -> list[tuple[str, float]]:
        """von Mises-Fisher kernel density estimate over the classified points.

        A point is classified if its metadata carries a `CLASS_KEY` entry (background
        points never do, so they never vote). Weights each such point by
        `exp(kappa * sim(vec, point))` -- the vMF kernel -- aggregated per class and
        normalized into a posterior:
          - prior="uniform":   posterior(c) proportional to the *mean* weight in c
          - prior="empirical": posterior(c) proportional to the *summed* weight in c
        With `calibrate=True` (default) `sim` is cosine in the validated transformed
        space (optionally mutual-proximity rescored); with `calibrate=False` it is the
        raw cosine, byte-for-byte with the original behaviour. `kappa` is the
        concentration/bandwidth. Unclassified points and `exclude` are dropped. Returns
        `(class, probability)` most-probable-first, or `[]` if nothing is classified.
        """
        return self.class_scores(vec, kappa, prior, exclude, calibrate)["classes"]

    def class_scores(
        self,
        vec: np.ndarray,
        kappa: float = 10.0,
        prior: str = "uniform",
        exclude: str | None = None,
        calibrate: bool = True,
    ) -> dict:
        """`class_probabilities` plus diagnostics (`n_eff`, `low_confidence`, `count`).

        Single source of truth for the ranking; the public method returns only the
        ranked list so its `calibrate=False` output stays byte-for-byte identical.
        """
        if prior not in ("uniform", "empirical"):
            raise ValueError("prior must be 'uniform' or 'empirical'")
        idx = [
            i
            for i in range(len(self.ids))
            if self.metadata[i].get(CLASS_KEY) is not None and self.ids[i] != exclude
        ]
        if not idx:
            return {"classes": [], "n_eff": 0.0, "low_confidence": True, "count": 0}
        geo = self._get_geometry() if calibrate else None
        sims = self._similarities(vec, idx, geo)
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
        classes = sorted(
            ((c, s / total) for c, s in scores.items()), key=lambda t: -t[1]
        )
        n_eff = self._effective_n(weights)
        return {
            "classes": classes,
            "n_eff": n_eff,
            "low_confidence": n_eff < MIN_EFFECTIVE_N,
            "count": len(idx),
        }

    # --- component 4: local-density use case -----------------------------
    def density(
        self,
        vec: np.ndarray,
        kappa: float = 10.0,
        radius: float = 0.5,
        exclude: str | None = None,
        calibrate: bool = True,
    ) -> dict:
        """Estimate how crowded the embedding space is around `vec`. Core views:

          - `density`: mean von Mises-Fisher kernel weight `exp(kappa*(sim - 1))` over
            the reference points -- a smooth estimate in (0, 1], comparable *across*
            queries because the exponent is anchored at the theoretical max sim of 1.
          - `neighbors`: hard count of reference points with `sim >= radius`.

        The reference set is every non-background point (background feeds the geometry
        transform but is not part of the density distribution) except `exclude`. With
        `calibrate=True` (default) `sim` is computed in the validated transformed space
        and the result also carries an **honesty layer**: `percentile` (rank of the
        query's local density against the reference via `LocalOutlierFactor`),
        `lof_score`, `n_eff` (Kish effective sample size), `rank_ci` (bootstrap 95% CI
        on the percentile) and a `low_confidence` flag. With `calibrate=False` the
        output is the raw cosine estimate, byte-for-byte with the original behaviour.
        Returns zeros when the reference set is empty.
        """
        idx = [
            i
            for i in range(len(self.ids))
            if self.ids[i] != exclude and not self.metadata[i].get(BACKGROUND_KEY)
        ]
        if not idx:
            return {"density": 0.0, "neighbors": 0, "count": 0,
                    "kappa": kappa, "radius": radius}
        geo = self._get_geometry() if calibrate else None
        sims = self._similarities(vec, idx, geo)
        weights = np.exp(kappa * (sims - 1.0))
        out = {
            "density": float(weights.mean()),
            "neighbors": int((sims >= radius).sum()),
            "count": len(idx),
            "kappa": kappa,
            "radius": radius,
        }
        if geo is not None:
            out.update(self._density_honesty(vec, idx, geo, weights))
        return out

    def _density_honesty(
        self, vec: np.ndarray, idx: list[int], geo: geometry.Geometry, weights: np.ndarray
    ) -> dict:
        """Percentile / LOF / n_eff / rank_ci / low_confidence for a calibrated density.

        Ranks the query's local density against the reference with `LocalOutlierFactor`
        in the novelty representation (`Geometry.transform_novelty`: GD-denoised position
        plus reconstruction residual, so a query far in *discarded* directions still reads
        as far). The query and reference are scored on equal footing -- one non-novelty
        LOF over reference+{query}, so their leave-one-out scores are comparable; a
        novelty-mode fit would score the *seen* reference as inliers and push every fresh
        query to percentile ~0. `n_eff` is the Kish effective sample size of the density
        kernel weights -- the honest count the estimate actually rests on.
        """
        n_eff = self._effective_n(weights)
        info: dict = {"n_eff": n_eff}
        ref_z = geo.transform_novelty(self.embeddings[idx])
        z_query = geo.transform_novelty(vec)
        rank_ci = None
        if len(idx) >= 3:
            from sklearn.neighbors import LocalOutlierFactor

            k = min(20, len(idx) - 1)
            # Reference baseline = each reference point's leave-one-out LOF (fit on the
            # reference alone). The query gets the same treatment via a fit on
            # reference+{query}, reading the appended point's score -- so query and
            # reference are both "held out" and their scores are comparable.
            ref_scores = LocalOutlierFactor(n_neighbors=k).fit(ref_z).negative_outlier_factor_
            q_score = float(
                LocalOutlierFactor(n_neighbors=k)
                .fit(np.vstack([ref_z, z_query]))
                .negative_outlier_factor_[-1]
            )
            info["lof_score"] = q_score
            info["percentile"] = float((ref_scores <= q_score).mean())
            # bootstrap the reference scores for a CI on that percentile
            rng = np.random.default_rng(0)
            boot = rng.choice(len(ref_scores), size=(1000, len(ref_scores)), replace=True)
            pct = np.sort((ref_scores[boot] <= q_score).mean(axis=1))
            rank_ci = (float(pct[25]), float(pct[975]))
            info["rank_ci"] = rank_ci
        wide = rank_ci is not None and (rank_ci[1] - rank_ci[0]) > 0.5
        info["low_confidence"] = n_eff < MIN_EFFECTIVE_N or wide
        return info

    # --- component 5: visualization use case -----------------------------
    def project(
        self, method: str = "pca", dim: int = 2, seed: int = 0
    ) -> tuple[list[str], np.ndarray, list[str | None]]:
        """Reduce all stored embeddings to `dim` dims for visualization.

        Returns (ids, coords, classes): coords is an (N, dim) float array, row i the
        projection of ids[i]; classes[i] is the CLASS_KEY value of ids[i] or None.
        `method` is "pca" (numpy SVD; linear, deterministic) or "tsne"
        (sklearn.manifold.TSNE; nonlinear, `seed` makes it reproducible).
        """
        if method not in ("pca", "tsne"):
            raise ValueError("method must be 'pca' or 'tsne'")
        n = len(self.ids)
        if n <= dim:
            raise ValueError(
                f"need more than {dim} embedding(s) to project to {dim}-D, have {n}"
            )

        if method == "pca":
            centered = self.embeddings - self.embeddings.mean(axis=0)
            u, s, _ = np.linalg.svd(centered, full_matrices=False)
            coords = u[:, :dim] * s[:dim]
        else:  # tsne; lazy import like the encoder, so non-tsne paths don't pay for it
            from sklearn.manifold import TSNE

            # t-SNE requires perplexity < n_samples; clamp for small contexts.
            perplexity = min(30.0, max(1.0, (n - 1) / 3))
            coords = TSNE(
                n_components=dim,
                init="pca",
                random_state=seed,
                perplexity=perplexity,
            ).fit_transform(self.embeddings)

        classes = [m.get(CLASS_KEY) for m in self.metadata]
        return list(self.ids), np.asarray(coords, dtype=np.float32), classes

    # --- component 6: validated calibration use case ---------------------
    def evaluate(
        self,
        *,
        k: int = 5,
        n_folds: int = 5,
        seed: int = 0,
        apply_recommended: bool = False,
    ) -> dict:
        """Leak-free nested-CV assessment of the geometry transform on the labels.

        Compares the raw-cosine baseline against the best validated pipeline (selected
        per outer fold, scored only on held-out points) and reports the accuracy delta
        with a bootstrap CI, hubness before/after, the selection-frequency table, and
        the config recommended for production. With `apply_recommended=True`, adopts
        that config as this context's `geometry_config` (persist with `save` to make it
        durable across the server's mtime reload). Labels are used only to score and
        select -- never to fit the transform. See `geometry.nested_cv`.
        """
        labelled_idx = [
            i for i in range(len(self.ids)) if self.metadata[i].get(CLASS_KEY) is not None
        ]
        labels = [self.metadata[i][CLASS_KEY] for i in labelled_idx]
        if len(labelled_idx) < 4 or len(set(labels)) < 2:
            raise ValueError(
                "evaluate needs at least 4 labelled points across at least 2 classes"
            )
        bg_idx = [i for i in range(len(self.ids)) if self.metadata[i].get(BACKGROUND_KEY)]
        report = geometry.nested_cv(
            self.embeddings[labelled_idx].astype(np.float64),
            labels,
            self.embeddings[bg_idx].astype(np.float64) if bg_idx else None,
            k=k,
            n_folds=n_folds,
            seed=seed,
        )
        if apply_recommended:
            self.geometry_config = report["recommended_config"]
            self._geometry = None
        return report

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

    @property
    def num_background(self) -> int:
        """How many samples are unlabelled background (feed geometry, never vote)."""
        return sum(bool(m.get(BACKGROUND_KEY)) for m in self.metadata)
