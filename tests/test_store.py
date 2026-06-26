"""Tests for the geometry calibration, validation harness, and the calibrate switch.

No model download: every store is built from synthetic vectors via
`EmbeddingStore(ids, embeddings, model="test")`, and queries pass raw vectors straight
to the store methods. Determinism comes from fixed numpy Generators.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from text_embedding import geometry
from text_embedding.store import (
    BACKGROUND_KEY,
    CLASS_KEY,
    MIN_CALIBRATION_POINTS,
    EmbeddingStore,
)


def _planted_corpus(seed=0, d=48, n_per_class=25, n_bg_per_class=150):
    """Two classes separated by a small signal, swamped by a strong per-point common-mode
    axis (a genuine top-PC artifact that reorders raw-cosine neighbours)."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(d) * 4.0

    def make(label, n):
        sig = np.zeros(d)
        sig[0 if label == "a" else 1] = 1.0
        return [sig + 0.5 * rng.standard_normal(d) + rng.standard_normal() * common
                for _ in range(n)]

    ids, vecs, meta = [], [], []
    for lab in ("a", "b"):
        for i in range(n_per_class):
            ids.append(f"{lab}{i}"); vecs.append(make(lab, 1)[0]); meta.append({CLASS_KEY: lab})
    for lab in ("a", "b"):
        for i in range(n_bg_per_class):
            ids.append(f"bg{lab}{i}"); vecs.append(make(lab, 1)[0]); meta.append({BACKGROUND_KEY: "1"})
    return EmbeddingStore(ids, np.array(vecs, dtype=np.float32), model="test", metadata=meta)


# --- calibrate=False is byte-for-byte the original raw-cosine behaviour ----------

def test_calibrate_false_matches_raw_golden():
    rng = np.random.default_rng(3)
    n, d = 12, 16
    emb = rng.standard_normal((n, d)).astype(np.float32)
    meta = [{CLASS_KEY: "a" if i % 2 else "b"} for i in range(n)]
    s = EmbeddingStore([f"i{i}" for i in range(n)], emb, model="test", metadata=meta)
    q = rng.standard_normal(d).astype(np.float32)

    # independent raw-cosine reference for class_probabilities (uniform prior)
    sims = (emb @ q) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q) + 1e-12)
    w = np.exp(10.0 * (sims - sims.max()))
    sums, counts = {}, {}
    for i in range(n):
        c = meta[i][CLASS_KEY]
        sums[c] = sums.get(c, 0.0) + w[i]; counts[c] = counts.get(c, 0) + 1
    scores = {c: sums[c] / counts[c] for c in sums}
    tot = sum(scores.values())
    expected = sorted(((c, v / tot) for c, v in scores.items()), key=lambda t: -t[1])

    got = s.class_probabilities(q, calibrate=False)
    assert [c for c, _ in got] == [c for c, _ in expected]
    # float32 store path vs the float64 reference: equal to float32 precision
    np.testing.assert_allclose([p for _, p in got], [p for _, p in expected], rtol=1e-5)

    # density raw view matches and carries no honesty fields
    dr = s.density(q, calibrate=False)
    assert set(dr) == {"density", "neighbors", "count", "kappa", "radius"}
    np.testing.assert_allclose(dr["density"], float(np.exp(10.0 * (sims - 1.0)).mean()), rtol=1e-5)
    assert dr["neighbors"] == int((sims >= 0.5).sum())


# --- the core claim: validated transform raises purity and lowers hubness --------

def test_evaluate_improves_on_planted_artifact():
    s = _planted_corpus(seed=0)
    rep = s.evaluate(k=5, seed=0)
    assert rep["selected"]["accuracy"] > rep["baseline"]["accuracy"]
    assert rep["delta_accuracy_ci95"][0] >= 0.0  # CI lower bound does not dip below zero
    assert rep["hubness_skew_selected"] < rep["hubness_skew_baseline"]
    assert rep["recommended_config"]["denoise"] in (True, False)


# --- leak checks: labels never enter the fit -------------------------------------

def test_fit_signature_takes_no_labels():
    params = set(inspect.signature(geometry.fit).parameters)
    assert "labels" not in params and "y" not in params
    assert params == {"X", "config", "with_hubness_stats"}


def test_negative_control_random_labels_show_no_improvement():
    """If the harness leaked labels into the transform fit, even *random* labels would
    show spurious improvement. With the leak-free protocol they must not: the delta CI
    has to straddle zero."""
    rng = np.random.default_rng(1)
    d = 32
    X = rng.standard_normal((40, d))  # no class structure at all
    labels = ["a" if i % 2 else "b" for i in range(40)]
    rng.shuffle(labels)
    bg = rng.standard_normal((100, d))
    rep = geometry.nested_cv(X, labels, bg, k=5, seed=0)
    lo, hi = rep["delta_accuracy_ci95"]
    assert lo <= 0.0 <= hi  # no spurious gain survives the held-out evaluation


# --- background feeds geometry but never votes / never enters the density ref -----

def test_background_excluded_from_voting_and_density_reference():
    s = _planted_corpus(seed=2)
    n_labelled = s.num_classified
    assert s.num_background == 300
    q = np.zeros(s.dim, dtype=np.float32); q[0] = 1.0
    # density reference excludes background
    assert s.density(q, calibrate=False)["count"] == n_labelled
    # classify only ever ranges over labelled points
    assert s.class_scores(q, calibrate=False)["count"] == n_labelled
    # ...but the geometry transform is fit over *all* points (labelled + background)
    geo = s._get_geometry()
    assert geo is not None and geo.pool_z.shape[0] == len(s.ids)


# --- Gavish-Donoho denoising keeps the planted rank, drops the noise bulk ---------

def test_gavish_donoho_keeps_planted_rank():
    rng = np.random.default_rng(0)
    n, d, r = 300, 60, 4
    signal = rng.standard_normal((n, r)) @ (rng.standard_normal((r, d)) * 6.0)
    noise = rng.standard_normal((n, d))
    X = signal + noise
    _, sv, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    rank = geometry.optimal_hard_threshold_rank(sv, n, d)
    assert r <= rank < d  # recovers the signal, discards the noise bulk


# --- cache lifecycle and small-n guards ------------------------------------------

def test_cache_lifecycle():
    s = _planted_corpus(seed=4)
    g1 = s._get_geometry()
    assert g1 is not None and s._geometry is g1  # fitted and cached
    s.delete_many([])  # no-op: cache must survive
    assert s._geometry is g1
    s.delete_many(["a0"])  # real mutation: cache cleared
    assert s._geometry is None
    assert s._get_geometry() is not None  # refits lazily


@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_n_falls_back_to_raw(n):
    rng = np.random.default_rng(n)
    emb = rng.standard_normal((n, 8)).astype(np.float32)
    meta = [{CLASS_KEY: "a"} for _ in range(n)]
    s = EmbeddingStore([f"i{i}" for i in range(n)], emb, model="test", metadata=meta)
    assert n < MIN_CALIBRATION_POINTS
    assert s._get_geometry() is None  # too small to calibrate
    q = rng.standard_normal(8).astype(np.float32)
    # calibrate=True must not crash and must equal the raw path (geometry is None)
    assert s.class_probabilities(q, calibrate=True) == s.class_probabilities(q, calibrate=False)
    d = s.density(q, calibrate=True)
    assert "percentile" not in d  # no honesty layer without a transform


# --- n_eff / honesty layer behaviour ---------------------------------------------

def test_effective_n_is_kish():
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert EmbeddingStore._effective_n(w) == pytest.approx(4.0)  # uniform -> n
    w2 = np.array([1.0, 0.0, 0.0, 0.0])
    assert EmbeddingStore._effective_n(w2) == pytest.approx(1.0)  # one point -> 1


def test_density_novelty_ranks_outlier_below_inlier():
    s = _planted_corpus(seed=0)
    q_in = np.zeros(s.dim, dtype=np.float32); q_in[0] = 1.0
    rng = np.random.default_rng(99)
    q_far = (10.0 * rng.standard_normal(s.dim)).astype(np.float32)
    p_in = s.density(q_in, calibrate=True)["percentile"]
    p_far = s.density(q_far, calibrate=True)["percentile"]
    assert p_in > p_far
