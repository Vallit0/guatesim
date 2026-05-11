"""Tests offline de PPC y compare_posteriors.

No tocan PyMC: fabrican `IRLPosterior` sintéticos directamente.
Validan (a) que el PPC detecta cuándo el modelo Boltzmann ajusta y
cuándo no, (b) que `compare_posteriors` devuelve probabilidades
sensatas en casos donde la respuesta correcta es conocida por
construcción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.irl.bayesian_irl import IRLPosterior
from guatemala_sim.irl.boltzmann import (
    sample_boltzmann_choices,
    subtract_reference,
)
from guatemala_sim.irl.posterior_analysis import (
    PPCResult,
    PosteriorComparison,
    _hdi,
    compare_posteriors,
    posterior_predictive_check,
)


# --- helpers para IRLPosterior sintéticos -----------------------------------


def make_synthetic_posterior(
    w_true: np.ndarray,
    feature_names: tuple[str, ...] = ("f0", "f1", "f2", "f3", "f4", "f5"),
    n_obs: int = 8,
    n_candidates: int = 5,
    n_samples: int = 1000,
    sample_sigma: float = 0.1,
    seed: int = 0,
) -> IRLPosterior:
    """Construye un IRLPosterior sintético centrado en w_true.

    Para tests, no necesitamos que los samples vengan de NUTS real;
    basta que tengan la geometría correcta (centrados en w_true,
    con varianza pequeña → posterior concentrado).
    """
    d = len(w_true)
    rng = np.random.default_rng(seed)
    samples = w_true.reshape(1, d) + rng.normal(0, sample_sigma, size=(n_samples, d))
    w_mean = samples.mean(axis=0)
    hdi = np.array([_hdi(samples[:, k]) for k in range(d)])
    return IRLPosterior(
        feature_names=feature_names,
        n_observations=n_obs,
        n_candidates=n_candidates,
        w_mean=w_mean,
        w_hdi95=hdi,
        w_samples=samples,
        diverging=0,
        rhat_max=1.01,
        ess_bulk_min=500.0,
        prior_sigma=1.0,
    )


def make_synthetic_choices(
    w_true: np.ndarray,
    T: int = 8,
    K: int = 5,
    seed: int = 1,
):
    """Genera (features ref-subtracted, chosen) Boltzmann con w_true."""
    d = len(w_true)
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1, size=(T, K, d))
    features_rs = subtract_reference(features, ref_idx=0)
    chosen = sample_boltzmann_choices(features_rs, w_true, rng=rng)
    return features_rs, chosen


# --- _hdi ------------------------------------------------------------------


def test_hdi_normal_aprox_centrado():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, size=10000)
    lo, hi = _hdi(samples, prob=0.95)
    # Para una normal estándar, HDI95 ≈ ±1.96
    assert -2.1 < lo < -1.7
    assert 1.7 < hi < 2.1


def test_hdi_samples_vacios():
    lo, hi = _hdi(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


# --- PPC -------------------------------------------------------------------


def test_ppc_modelo_correcto_alta_accuracy():
    """Si el posterior contiene w_true y los datos vienen de Boltzmann
    con w_true, la accuracy debe ser claramente > random."""
    rng_seed = 11
    w_true = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    features, chosen = make_synthetic_choices(
        w_true, T=200, K=5, seed=rng_seed,
    )
    posterior = make_synthetic_posterior(
        w_true, n_obs=200, n_candidates=5, sample_sigma=0.05,
    )
    ppc = posterior_predictive_check(features, chosen, posterior, n_samples=200)
    assert isinstance(ppc, PPCResult)
    assert ppc.lift_over_random > 0.15  # bien arriba de 1/5 = 0.2
    assert ppc.accuracy_argmax_point > ppc.random_accuracy


def test_ppc_modelo_incorrecto_cerca_de_random():
    """Si el posterior está mal (w ortogonal al verdadero), accuracy
    no debería despegar mucho del random."""
    w_true = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w_wrong = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    features, chosen = make_synthetic_choices(w_true, T=200, K=5, seed=22)
    posterior = make_synthetic_posterior(w_wrong, n_obs=200, n_candidates=5)
    ppc = posterior_predictive_check(features, chosen, posterior, n_samples=200)
    # No exigimos lift = 0 exacto (los features son random, hay coincidencias),
    # pero sí que el lift sea pequeño comparado con el caso correcto
    assert ppc.lift_over_random < 0.15


def test_ppc_devuelve_random_baseline_correcto():
    w_true = np.array([1.0, 0.5])
    features, chosen = make_synthetic_choices(w_true, T=20, K=4, seed=3)
    posterior = make_synthetic_posterior(
        w_true, feature_names=("f0", "f1"), n_obs=20, n_candidates=4,
    )
    ppc = posterior_predictive_check(features, chosen, posterior, n_samples=100)
    assert ppc.random_accuracy == pytest.approx(0.25)
    assert ppc.n_turns == 20
    assert ppc.n_candidates == 4


def test_ppc_summary_text_no_crashea():
    w = np.array([1.0, 0.0])
    features, chosen = make_synthetic_choices(w, T=10, K=3, seed=7)
    posterior = make_synthetic_posterior(
        w, feature_names=("f0", "f1"), n_obs=10, n_candidates=3,
    )
    ppc = posterior_predictive_check(features, chosen, posterior, n_samples=50)
    assert "PPC" in ppc.summary_text("Claude")


def test_ppc_falla_en_shapes_invalidos():
    w = np.array([1.0, 0.0])
    posterior = make_synthetic_posterior(
        w, feature_names=("f0", "f1"), n_obs=10, n_candidates=3,
    )
    with pytest.raises(ValueError, match="3D"):
        posterior_predictive_check(
            np.zeros((5, 5)), np.zeros(5, dtype=int), posterior,
        )


# --- compare_posteriors ----------------------------------------------------


def test_compare_posteriors_dim_donde_a_es_mayor():
    """Si w_a tiene mayor magnitud en dim 0 que w_b, entonces
    P(w_a_0 > w_b_0) debe ser cercano a 1."""
    w_a = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pa = make_synthetic_posterior(w_a, sample_sigma=0.1, seed=10)
    pb = make_synthetic_posterior(w_b, sample_sigma=0.1, seed=11)
    cmp = compare_posteriors(pa, pb, label_a="A", label_b="B")
    assert isinstance(cmp, PosteriorComparison)
    row = cmp.per_dimension.loc["f0"]
    assert row["p_a_gt_b"] > 0.95
    assert bool(row["decisive"]) is True
    assert row["diff_mean"] > 1.5


def test_compare_posteriors_dims_iguales_p_cerca_de_05():
    """w_a == w_b → P(w_a_k > w_b_k) ≈ 0.5 en todas las dims."""
    w = np.array([1.0, 0.5, -0.3, 0.0, 0.7, -0.1])
    pa = make_synthetic_posterior(w, seed=20)
    pb = make_synthetic_posterior(w, seed=21)
    cmp = compare_posteriors(pa, pb)
    for f in cmp.feature_names:
        p = cmp.per_dimension.loc[f, "p_a_gt_b"]
        assert 0.35 < p < 0.65, f"feature {f}: p={p}, esperado ~0.5"
        assert bool(cmp.per_dimension.loc[f, "decisive"]) is False


def test_compare_posteriors_cosine_anti_aligned():
    """w_a = -w_b → cosine posterior ≈ -1, p_anti_aligned ≈ 1."""
    w_a = np.array([1.0, 0.5, -0.3, 0.0, 0.7, -0.1])
    pa = make_synthetic_posterior(w_a, sample_sigma=0.05, seed=30)
    pb = make_synthetic_posterior(-w_a, sample_sigma=0.05, seed=31)
    cmp = compare_posteriors(pa, pb)
    assert cmp.cosine_posterior_mean < -0.9
    assert cmp.p_anti_aligned > 0.9


def test_compare_posteriors_cosine_alineados():
    w = np.array([1.0, 0.5, -0.3, 0.0, 0.7, -0.1])
    pa = make_synthetic_posterior(w, sample_sigma=0.05, seed=40)
    pb = make_synthetic_posterior(w, sample_sigma=0.05, seed=41)
    cmp = compare_posteriors(pa, pb)
    assert cmp.cosine_posterior_mean > 0.95


def test_compare_posteriors_falla_si_features_difieren():
    pa = make_synthetic_posterior(
        np.zeros(2), feature_names=("f0", "f1"),
    )
    pb = make_synthetic_posterior(
        np.zeros(2), feature_names=("g0", "g1"),
    )
    with pytest.raises(ValueError, match="feature_names"):
        compare_posteriors(pa, pb)


def test_compare_posteriors_falla_decisive_threshold_invalido():
    pa = make_synthetic_posterior(np.zeros(2), feature_names=("f0", "f1"))
    pb = make_synthetic_posterior(np.zeros(2), feature_names=("f0", "f1"))
    with pytest.raises(ValueError, match="decisive_threshold"):
        compare_posteriors(pa, pb, decisive_threshold=0.4)


def test_compare_posteriors_summary_text():
    w_a = np.array([2.0, 0.0])
    pa = make_synthetic_posterior(w_a, feature_names=("f0", "f1"))
    pb = make_synthetic_posterior(np.zeros(2), feature_names=("f0", "f1"))
    cmp = compare_posteriors(pa, pb, label_a="Claude", label_b="GPT")
    txt = cmp.summary_text()
    assert "Claude" in txt and "GPT" in txt
    assert "decisivas" in txt
