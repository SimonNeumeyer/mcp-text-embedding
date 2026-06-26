"""Unsupervised geometry calibration for embedding rankings.

The raw ranking of a query against a small set of stored embeddings is corrupted at
high dimension / low sample count by three forces:
  - **anisotropy** -- a dominant common-mode direction that reorders neighbours,
  - **distance concentration** -- the Marchenko-Pastur noise bulk drowns fine signal,
  - **hubness** -- a few points are everyone's neighbour regardless of meaning.

This module learns an *unsupervised* linear transform on a pooled set of embeddings
and an optional similarity rescoring:
  1. denoise to the signal subspace (Gavish-Donoho optimal hard threshold),
  2. drop the top `drop_top` common-mode directions (all-but-the-top),
  3. Ledoit-Wolf-shrunk whitening of the retained directions,
  4. L2-normalise,
  5. (at scoring time) mutual-proximity hubness rescoring.

**Labels never enter the fit** -- every step above is unsupervised. The labelled
points are used only by `nested_cv` to *select* which pipeline config to deploy and
to score it, behind nested cross-validation, so the selection never leaks into the
fit. `Geometry.fit` is the transform; `nested_cv` is the validated, leak-free selection.

References: Gavish & Donoho 2014 (optimal hard threshold); Mu & Viswanath 2018
(all-but-the-top); Su et al. 2021 (whitening sentence reps); Ledoit & Wolf (shrinkage);
Schnitzer et al. 2012 (mutual proximity / hubness).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

# A pipeline config is a plain dict so it round-trips through JSON in the NPZ.
# `drop_top` is the number of leading common-mode directions to discard.
DEFAULT_CONFIG: dict = {"denoise": True, "drop_top": 1, "whiten": True, "hubness": True}

# Vectorised standard-normal CDF (no scipy dependency); small arrays, so erf is fine.
_ERF = np.vectorize(erf, otypes=[float])


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(np.asarray(x, dtype=float) / sqrt(2.0)))


def optimal_hard_threshold_rank(s: np.ndarray, n_rows: int, n_cols: int) -> int:
    """Gavish-Donoho 2014 optimal hard threshold for singular values, unknown noise.

    Returns how many leading singular values to keep: those above
    `omega(beta) * median(s)`, where `beta = min/max` shape ratio and `omega` is the
    paper's polynomial approximation. Parameter-free, and the median makes it
    noise-level-agnostic. Always keeps at least the top component.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0
    beta = min(n_rows, n_cols) / max(n_rows, n_cols)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = omega * float(np.median(s))
    return max(int((s > tau).sum()), 1)


def _ledoit_wolf_whitener(Y: np.ndarray) -> np.ndarray:
    """Symmetric inverse-square-root of the Ledoit-Wolf-shrunk covariance of `Y`.

    `Y @ W` has (shrinkage-regularised) identity covariance, equalising the variance
    of the retained directions. Shrinkage keeps it well-conditioned at small n.
    """
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(Y).covariance_
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


@dataclass
class Geometry:
    """A fitted unsupervised transform plus (optional) hubness statistics.

    `transform` is a pointwise linear map (center -> project -> whiten -> L2). When
    `config["hubness"]` is set, `pool_z`/`pool_mu`/`pool_sigma` hold the transformed
    fit pool and each pool point's similarity distribution, so `mp_similarity` can
    rescore query-vs-reference cosines into mutual-proximity similarities at scoring time.
    """

    mean: np.ndarray  # (d,)
    proj: np.ndarray  # (d, k) retained principal axes (after denoise + drop_top)
    whitener: np.ndarray | None  # (k, k) whitening matrix, or None
    config: dict
    proj_denoise: np.ndarray  # (d, r) GD-denoised axes *before* drop_top (for novelty)
    denoise_std: np.ndarray  # (r,) per-axis std of the pool in that basis
    residual_mean: float  # mean of the pool's energy *outside* the denoised subspace
    residual_std: float  # std of that residual
    pool_z: np.ndarray | None = None  # (N, k) transformed fit pool
    pool_mu: np.ndarray | None = None  # (N,) mean similarity of each pool point to the pool
    pool_sigma: np.ndarray | None = None  # (N,) std of that distribution

    def transform(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Apply the pointwise map (center -> project -> whiten) to a vector or matrix.

        `normalize=True` (the default) L2-normalises the rows, the right representation
        for *directional* cosine ranking (classify, density kernel). `normalize=False`
        keeps the whitened coordinates with their radial magnitude intact -- required for
        *novelty/density* ranking (LOF), where how far a query sits from the cloud is the
        whole signal and normalising it onto the unit sphere would hide every outlier.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = (X - self.mean) @ self.proj
        if self.whitener is not None:
            Y = Y @ self.whitener
        if not normalize:
            return Y
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        return Y / (norms + 1e-12)

    def transform_vec(self, vec: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Transform a single vector -> (k,)."""
        return self.transform(vec, normalize=normalize)[0]

    def transform_novelty(self, X: np.ndarray) -> np.ndarray:
        """Representation for novelty/density ranking: GD-denoised coordinates plus a
        reconstruction-residual axis.

        The first `r` axes are the position within the GD-denoised subspace (standardised
        per axis). The final axis is the *excess* reconstruction residual -- the query's
        energy outside that subspace, standardised and clipped at the pool mean -- so a
        query that differs from the cloud in discarded directions reads as far, while one
        that is merely *cleaner* than the (noisy) reference is not penalised. Together they
        make LOF see both in-subspace position and out-of-subspace distance.
        """
        Xc = np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mean
        coords = (Xc @ self.proj_denoise) / self.denoise_std
        recon = (Xc @ self.proj_denoise) @ self.proj_denoise.T
        residual = np.linalg.norm(Xc - recon, axis=1, keepdims=True)
        excess = np.clip((residual - self.residual_mean) / self.residual_std, 0.0, None)
        return np.hstack([coords, excess])

    def mp_similarity(
        self, z_query: np.ndarray, ref_idx: np.ndarray, base_sims: np.ndarray
    ) -> np.ndarray:
        """Gaussian mutual-proximity rescoring of query-vs-reference similarities.

        `base_sims[j]` is the cosine (in transformed space) between the query and
        reference point `ref_idx[j]`. Returns MP(query, ref) in [0, 1]: the product of
        the probability that a random pool similarity falls below `base_sims[j]` from
        the query's view and from reference point j's view (Schnitzer 2012, Gaussian
        approximation). Demotes hubs, whose similarities are high to everyone.
        """
        q_sims = self.pool_z @ np.asarray(z_query, dtype=np.float64)
        mu_q = float(q_sims.mean())
        sigma_q = float(q_sims.std()) + 1e-9
        fq = _norm_cdf((base_sims - mu_q) / sigma_q)
        fi = _norm_cdf((base_sims - self.pool_mu[ref_idx]) / self.pool_sigma[ref_idx])
        return fq * fi


def fit(X: np.ndarray, config: dict, *, with_hubness_stats: bool = True) -> Geometry:
    """Fit the unsupervised geometry transform on pooled embeddings `X` (n, d).

    No labels are consulted. `config` toggles `denoise` (Gavish-Donoho), `drop_top`
    (int common-mode directions to discard), `whiten` (Ledoit-Wolf), and `hubness`
    (store per-point stats for `mp_similarity`). Degenerate configs (e.g. dropping
    every retained direction) fall back to keeping the top component.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    r = optimal_hard_threshold_rank(s, n, d) if config.get("denoise") else len(s)
    m = min(int(config.get("drop_top", 0)), max(r - 1, 0))  # never drop all retained
    proj = Vt[m:r].T  # (d, k)
    if proj.shape[1] == 0:  # fully degenerate; keep the leading direction
        proj = Vt[:1].T

    whitener = None
    if config.get("whiten"):
        whitener = _ledoit_wolf_whitener(Xc @ proj)

    # Novelty basis: the GD-denoised subspace *including* the dropped common-mode axes,
    # standardised per axis, plus the std of the pool's reconstruction residual (energy
    # outside that subspace). Keeps radial + out-of-subspace structure that `proj` discards.
    proj_denoise = Vt[:r].T
    coords = Xc @ proj_denoise
    denoise_std = coords.std(axis=0) + 1e-9
    residual_norm = np.linalg.norm(Xc - coords @ proj_denoise.T, axis=1)

    geo = Geometry(
        mean=mean,
        proj=proj,
        whitener=whitener,
        config=dict(config),
        proj_denoise=proj_denoise,
        denoise_std=denoise_std,
        residual_mean=float(residual_norm.mean()),
        residual_std=float(residual_norm.std()) + 1e-9,
    )

    if config.get("hubness") and with_hubness_stats and n >= 3:
        z = geo.transform(X)
        sims = z @ z.T
        np.fill_diagonal(sims, np.nan)
        geo.pool_z = z
        geo.pool_mu = np.nanmean(sims, axis=1)
        geo.pool_sigma = np.nanstd(sims, axis=1) + 1e-9
    return geo


# --- validation harness ---------------------------------------------------------


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def knn_scores(Z: np.ndarray, labels: list, k: int) -> tuple[float, float]:
    """Leave-one-out kNN on already-transformed rows `Z`.

    Returns `(accuracy, hit_rate)`: the fraction of points whose k-NN *majority* label
    matches their own (accuracy), and the mean fraction of each point's k neighbours
    sharing its label (neighbourhood hit-rate). Self is excluded via -inf diagonal.
    """
    Z = _l2(Z)
    sims = Z @ Z.T
    np.fill_diagonal(sims, -np.inf)
    n = len(labels)
    kk = max(1, min(k, n - 1))
    correct = 0
    hits = 0.0
    for i in range(n):
        nn = np.argpartition(-sims[i], kk - 1)[:kk]
        nn_labels = [labels[j] for j in nn]
        votes: dict = {}
        for lab in nn_labels:
            votes[lab] = votes.get(lab, 0) + 1
        pred = max(votes, key=votes.get)
        correct += pred == labels[i]
        hits += sum(lab == labels[i] for lab in nn_labels) / kk
    return correct / n, hits / n


def _stratified_folds(labels: list, n_folds: int, rng) -> list[np.ndarray]:
    """Indices for `n_folds` stratified folds: each class spread across folds."""
    folds: list[list] = [[] for _ in range(n_folds)]
    for lab in sorted(set(labels)):
        idx = [i for i, x in enumerate(labels) if x == lab]
        idx = list(rng.permutation(idx))
        for j, i in enumerate(idx):
            folds[j % n_folds].append(i)
    return [np.array(sorted(f)) for f in folds]


def candidate_configs() -> list[dict]:
    """The pipeline grid the nested CV selects over."""
    configs = []
    for denoise in (False, True):
        for drop_top in (0, 1, 2, 3):
            for whiten in (False, True):
                for hubness in (False, True):
                    configs.append(
                        {
                            "denoise": denoise,
                            "drop_top": drop_top,
                            "whiten": whiten,
                            "hubness": hubness,
                        }
                    )
    return configs


def _complexity(cfg: dict) -> int:
    """A parsimony score for the one-SE tie-break: fewer/cheaper steps win ties."""
    return (
        int(bool(cfg.get("denoise")))
        + int(cfg.get("drop_top", 0))
        + int(bool(cfg.get("whiten")))
        + int(bool(cfg.get("hubness")))
    )


def _score_config(
    fit_X: np.ndarray, train_X: np.ndarray, train_labels: list, cfg: dict, k: int
) -> float:
    """Inner-loop score: fit the transform on `fit_X` (unsupervised), then LOO kNN
    accuracy of `train_labels` over the transformed `train_X`. Labels touch only the
    score, never the fit."""
    geo = fit(fit_X, cfg, with_hubness_stats=False)
    acc, _ = knn_scores(geo.transform(train_X), train_labels, k)
    return acc


def nested_cv(
    labelled_X: np.ndarray,
    labels: list,
    background_X: np.ndarray | None = None,
    *,
    k: int = 5,
    n_folds: int = 5,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict:
    """Leak-free nested cross-validation over the labelled points.

    Outer stratified K-fold: each fold's test points are removed from **both** the
    transform fit and the inner selection. Inner loop selects the pipeline config
    maximising LOO kNN accuracy on the outer-train labels (one-SE parsimony tie-break),
    fit on `background + outer-train` (test excluded). The selected config is scored on
    the held-out outer-test fold, so the reported numbers reflect generalisation to
    unseen real queries -- the transform never saw them, in fit or selection.

    Returns a report: outer-CV accuracy/hit-rate for the raw-cosine baseline vs the
    selected pipeline, a paired bootstrap CI on the per-point accuracy delta, hubness
    skewness before/after, a selection-frequency table, and the recommended config
    (the one selected on the full data) for production refit.
    """
    rng = np.random.default_rng(seed)
    labelled_X = np.asarray(labelled_X, dtype=np.float64)
    labels = list(labels)
    n = len(labels)
    bg = (
        np.asarray(background_X, dtype=np.float64)
        if background_X is not None and len(background_X)
        else np.empty((0, labelled_X.shape[1]))
    )

    min_class = min(sum(x == c for x in labels) for c in set(labels))
    n_folds = max(2, min(n_folds, min_class))
    folds = _stratified_folds(labels, n_folds, rng)
    configs = candidate_configs()

    # Per outer-test point: was the baseline / selected pipeline's majority vote correct?
    base_correct = np.zeros(n)
    sel_correct = np.zeros(n)
    base_hit = np.zeros(n)
    sel_hit = np.zeros(n)
    selection_counts: dict = {}

    for test_idx in folds:
        train_idx = np.array([i for i in range(n) if i not in set(test_idx.tolist())])
        train_X, train_labels = labelled_X[train_idx], [labels[i] for i in train_idx]
        fit_X = np.vstack([labelled_X[train_idx], bg]) if len(bg) else labelled_X[train_idx]

        # inner: pick the config with the best LOO accuracy on outer-train (one-SE parsimony)
        scores = [_score_config(fit_X, train_X, train_labels, c, k) for c in configs]
        best = max(scores)
        se = np.std(scores) / sqrt(max(len(scores), 1))
        within = [c for c, s in zip(configs, scores) if s >= best - se]
        chosen = min(within, key=_complexity)
        selection_counts[_key(chosen)] = selection_counts.get(_key(chosen), 0) + 1

        # score baseline (raw cosine) and the chosen pipeline on the held-out test fold
        geo = fit(fit_X, chosen, with_hubness_stats=False)
        ref_z = geo.transform(train_X)
        raw_ref = _l2(train_X)
        for ti in test_idx:
            base_correct[ti], base_hit[ti] = _predict_one(
                _l2(labelled_X[ti]), raw_ref, train_labels, labels[ti], k
            )
            sel_correct[ti], sel_hit[ti] = _predict_one(
                geo.transform_vec(labelled_X[ti]), ref_z, train_labels, labels[ti], k
            )

    delta = sel_correct - base_correct
    boot = rng.choice(n, size=(n_boot, n), replace=True)
    boot_delta = np.sort(delta[boot].mean(axis=1))
    ci = (float(boot_delta[int(0.025 * n_boot)]), float(boot_delta[int(0.975 * n_boot)]))

    # config selected on the full data = production recommendation
    full_fit = np.vstack([labelled_X, bg]) if len(bg) else labelled_X
    full_scores = [_score_config(full_fit, labelled_X, labels, c, k) for c in configs]
    fbest = max(full_scores)
    fse = np.std(full_scores) / sqrt(max(len(full_scores), 1))
    recommended = min(
        [c for c, s in zip(configs, full_scores) if s >= fbest - fse], key=_complexity
    )

    return {
        "n_labelled": n,
        "n_background": int(len(bg)),
        "n_folds": int(n_folds),
        "k": k,
        "baseline": {"accuracy": float(base_correct.mean()), "hit_rate": float(base_hit.mean())},
        "selected": {"accuracy": float(sel_correct.mean()), "hit_rate": float(sel_hit.mean())},
        "delta_accuracy": float(delta.mean()),
        "delta_accuracy_ci95": ci,
        "hubness_skew_baseline": _hubness_skew(_l2(labelled_X), k),
        "hubness_skew_selected": _hubness_skew(
            fit(full_fit, recommended, with_hubness_stats=False).transform(labelled_X), k
        ),
        "selection_frequency": selection_counts,
        "recommended_config": recommended,
    }


def _key(cfg: dict) -> str:
    return f"denoise={int(bool(cfg['denoise']))},drop_top={cfg['drop_top']},whiten={int(bool(cfg['whiten']))},hubness={int(bool(cfg['hubness']))}"


def _predict_one(z_query, ref_z, ref_labels, true_label, k) -> tuple[float, float]:
    """Majority-vote correctness and same-label hit-rate of one query's k-NN in `ref_z`."""
    sims = _l2(ref_z) @ np.asarray(z_query, dtype=np.float64).reshape(-1)
    kk = max(1, min(k, len(ref_labels)))
    nn = np.argpartition(-sims, kk - 1)[:kk]
    nn_labels = [ref_labels[j] for j in nn]
    votes: dict = {}
    for lab in nn_labels:
        votes[lab] = votes.get(lab, 0) + 1
    pred = max(votes, key=votes.get)
    hit = sum(lab == true_label for lab in nn_labels) / kk
    return float(pred == true_label), float(hit)


def _hubness_skew(Z: np.ndarray, k: int) -> float:
    """Skewness of the k-occurrence distribution (how often each point is in others'
    k-NN). High positive skew == strong hubness; lower is better."""
    Z = _l2(Z)
    n = len(Z)
    if n < 3:
        return 0.0
    kk = max(1, min(k, n - 1))
    sims = Z @ Z.T
    np.fill_diagonal(sims, -np.inf)
    occ = np.zeros(n)
    for i in range(n):
        for j in np.argpartition(-sims[i], kk - 1)[:kk]:
            occ[j] += 1
    mu, sd = occ.mean(), occ.std()
    if sd < 1e-12:
        return 0.0
    return float(((occ - mu) ** 3).mean() / sd**3)
