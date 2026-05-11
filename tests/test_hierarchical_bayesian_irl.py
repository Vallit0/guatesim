"""Tests del IRL bayesiano jerárquico.

Dos categorías:
  - **offline** (siempre): fabrican `HierarchicalIRLPosterior` sintéticos
    sin tocar PyMC y validan la mecánica de `compare_constitutions`,
    `mu_table`, `tau_table`, `pooling_factor`.
  - **slow** (opt-in con `pytest -m slow`): corren `fit_hierarchical_bayesian_irl`
    real (NUTS) sobre datos sintéticos generados con μ_true, τ_true
    conocidos y verifican recovery dentro del HDI95.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.irl.boltzmann import sample_boltzmann_choices, subtract_reference
from guatemala_sim.irl.hierarchical_bayesian_irl import (
    HierarchicalComparison,
    HierarchicalIRLPosterior,
    compare_constitutions,
)
from guatemala_sim.irl.posterior_analysis import _hdi


# --- helper: fabricar HierarchicalIRLPosterior sintético --------------------


def make_synthetic_hierarchical_posterior(
    mu_true: np.ndarray,
    tau_true: np.ndarray,
    n_seeds: int = 20,
    n_turns_per_seed: int = 8,
    n_candidates: int = 5,
    n_samples: int = 1000,
    sample_sigma_mu: float = 0.1,
    sample_sigma_tau: float = 0.05,
    feature_names: tuple[str, ...] = ("f0", "f1", "f2", "f3", "f4", "f5"),
    seed: int = 0,
) -> HierarchicalIRLPosterior:
    """Construye un posterior jerárquico sintético centrado en (μ_true, τ_true).

    Útil para tests offline que no requieren correr NUTS — solo
    verifican que `compare_constitutions` y las tablas funcionan
    correctamente dado un posterior bien formado.
    """
    d = len(mu_true)
    rng = np.random.default_rng(seed)

    mu_samples = mu_true.reshape(1, d) + rng.normal(0, sample_sigma_mu, size=(n_samples, d))
    tau_samples = np.maximum(
        1e-6,
        tau_true.reshape(1, d) + rng.normal(0, sample_sigma_tau, size=(n_samples, d)),
    )
    # w_per_seed: cada seed es una realización adicional con ruido
    w_per_seed = np.zeros((n_samples, n_seeds, d))
    for i in range(n_samples):
        for s in range(n_seeds):
            w_per_seed[i, s] = mu_samples[i] + tau_samples[i] * rng.normal(0, 1, d)

    mu_mean = mu_samples.mean(axis=0)
    tau_mean = tau_samples.mean(axis=0)
    mu_hdi = np.array([_hdi(mu_samples[:, k]) for k in range(d)])
    tau_hdi = np.array([_hdi(tau_samples[:, k]) for k in range(d)])

    return HierarchicalIRLPosterior(
        feature_names=feature_names[:d],
        n_seeds=n_seeds,
        n_turns_per_seed=tuple([n_turns_per_seed] * n_seeds),
        n_candidates=n_candidates,
        mu_samples=mu_samples,
        tau_samples=tau_samples,
        w_samples_per_seed=w_per_seed,
        mu_mean=mu_mean,
        mu_hdi95=mu_hdi,
        tau_mean=tau_mean,
        tau_hdi95=tau_hdi,
        diverging=0,
        rhat_max=1.01,
        ess_bulk_min=500.0,
        prior_sigma_mu=1.0,
        prior_sigma_tau=0.5,
    )


# --- API básica del posterior -----------------------------------------------


def test_mu_tau_tables_estructura():
    mu = np.array([1.5, 0.2, 0.0, -0.3, 0.7, 0.0])
    tau = np.array([0.1, 0.5, 0.05, 0.3, 0.2, 0.6])
    post = make_synthetic_hierarchical_posterior(mu, tau)
    df_mu = post.mu_table()
    df_tau = post.tau_table()
    assert len(df_mu) == 6
    assert len(df_tau) == 6
    assert "mu_hdi95_excludes_zero" in df_mu.columns
    assert df_mu.loc["f0", "mu_mean"] == pytest.approx(post.mu_mean[0], abs=0.01)
    assert df_tau.loc["f1", "tau_mean"] == pytest.approx(post.tau_mean[1], abs=0.01)


def test_pooling_factor_alto_cuando_tau_pequeno():
    """τ chico ⇒ seeds homogéneos ⇒ pooling factor alto."""
    mu = np.array([1.0, 1.0, 1.0])
    tau = np.array([0.01, 0.01, 0.01])  # MUY chico
    post = make_synthetic_hierarchical_posterior(
        mu, tau, n_seeds=20, sample_sigma_mu=0.05, sample_sigma_tau=0.001,
        feature_names=("f0", "f1", "f2"),
    )
    pf = post.pooling_factor()
    assert pf.shape == (3,)
    # Con τ chico, var(w_s | μ) ≈ 0, var(w marginal) ≈ var(μ posterior) ≈ pequeña
    # → la varianza explicada por τ es chica → pooling alto (cerca de 1)
    assert (pf > 0.5).any()


def test_pooling_factor_en_rango_cero_uno():
    mu = np.zeros(6)
    tau = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    post = make_synthetic_hierarchical_posterior(mu, tau)
    pf = post.pooling_factor()
    assert (pf >= 0.0).all()
    assert (pf <= 1.0).all()


def test_diagnostics_ok():
    post = make_synthetic_hierarchical_posterior(
        np.zeros(6), np.full(6, 0.3),
    )
    assert post.diagnostics_ok() is True


# --- compare_constitutions --------------------------------------------------


def test_compare_constitutions_dim_decisiva_cuando_mu_difiere():
    """Si μ_a tiene mayor magnitud en dim 0 que μ_b, P(μ_a_0 > μ_b_0) ≈ 1."""
    pa = make_synthetic_hierarchical_posterior(
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.full(6, 0.3),
        sample_sigma_mu=0.1,
        seed=10,
    )
    pb = make_synthetic_hierarchical_posterior(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.full(6, 0.3),
        sample_sigma_mu=0.1,
        seed=11,
    )
    cmp = compare_constitutions(pa, pb, label_a="A", label_b="B")
    assert isinstance(cmp, HierarchicalComparison)
    row = cmp.constitution.loc["f0"]
    assert row["p_a_gt_b"] > 0.95
    assert bool(row["decisive"]) is True
    # En dims sin diferencia, no decisivo
    assert bool(cmp.constitution.loc["f1", "decisive"]) is False


def test_compare_volatilities_detecta_lln_mas_volatil():
    """Si τ_a > τ_b en una dim, P(τ_a_k > τ_b_k) ≈ 1 → A más volátil."""
    pa = make_synthetic_hierarchical_posterior(
        np.zeros(6),
        np.array([0.8, 0.1, 0.1, 0.1, 0.1, 0.1]),  # A muy volátil en dim 0
        sample_sigma_tau=0.02,
        seed=20,
    )
    pb = make_synthetic_hierarchical_posterior(
        np.zeros(6),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # B estable en todas
        sample_sigma_tau=0.02,
        seed=21,
    )
    cmp = compare_constitutions(pa, pb)
    row = cmp.volatility.loc["f0"]
    assert row["p_a_gt_b"] > 0.95
    assert bool(row["decisive"]) is True


def test_compare_cosine_alineado_cuando_mu_iguales():
    mu = np.array([1.0, 0.5, -0.3, 0.0, 0.7, -0.1])
    pa = make_synthetic_hierarchical_posterior(
        mu, np.full(6, 0.3), sample_sigma_mu=0.05, seed=30,
    )
    pb = make_synthetic_hierarchical_posterior(
        mu, np.full(6, 0.3), sample_sigma_mu=0.05, seed=31,
    )
    cmp = compare_constitutions(pa, pb)
    assert cmp.cosine_mu_mean > 0.95


def test_compare_cosine_anti_alineado():
    mu_a = np.array([1.0, 0.5, -0.3, 0.0, 0.7, -0.1])
    pa = make_synthetic_hierarchical_posterior(
        mu_a, np.full(6, 0.3), sample_sigma_mu=0.05, seed=40,
    )
    pb = make_synthetic_hierarchical_posterior(
        -mu_a, np.full(6, 0.3), sample_sigma_mu=0.05, seed=41,
    )
    cmp = compare_constitutions(pa, pb)
    assert cmp.cosine_mu_mean < -0.9
    assert cmp.p_anti_aligned_constitution > 0.9


def test_compare_falla_si_features_difieren():
    pa = make_synthetic_hierarchical_posterior(
        np.zeros(2), np.full(2, 0.3),
        feature_names=("f0", "f1"),
    )
    pb = make_synthetic_hierarchical_posterior(
        np.zeros(2), np.full(2, 0.3),
        feature_names=("g0", "g1"),
    )
    with pytest.raises(ValueError, match="feature_names"):
        compare_constitutions(pa, pb)


def test_compare_falla_decisive_threshold_invalido():
    pa = make_synthetic_hierarchical_posterior(
        np.zeros(2), np.full(2, 0.3), feature_names=("f0", "f1"),
    )
    pb = make_synthetic_hierarchical_posterior(
        np.zeros(2), np.full(2, 0.3), feature_names=("f0", "f1"),
    )
    with pytest.raises(ValueError, match="decisive_threshold"):
        compare_constitutions(pa, pb, decisive_threshold=0.4)


def test_compare_summary_text():
    pa = make_synthetic_hierarchical_posterior(
        np.array([2.0, 0.0]), np.full(2, 0.3), feature_names=("f0", "f1"),
    )
    pb = make_synthetic_hierarchical_posterior(
        np.zeros(2), np.full(2, 0.3), feature_names=("f0", "f1"),
    )
    cmp = compare_constitutions(pa, pb, label_a="Claude", label_b="GPT")
    txt = cmp.summary_text()
    assert "Claude" in txt and "GPT" in txt
    assert "constituciones" in txt
    assert "volatilidades" in txt


# --- end-to-end con PyMC (slow, opt-in) -------------------------------------


_HAS_PYMC = importlib.util.find_spec("pymc") is not None


@pytest.mark.skipif(not _HAS_PYMC, reason="PyMC no instalado")
@pytest.mark.slow
def test_hierarchical_recovery_synthetic():
    """Genera datos con μ_true, τ_true conocidos, ajusta el modelo
    y verifica recovery dentro del HDI95.

    No requiere muchas iteraciones porque solo testeamos que la
    mecánica corre y que el posterior captura la verdad ground;
    no es benchmark de convergencia.
    """
    from guatemala_sim.irl.hierarchical_bayesian_irl import (
        fit_hierarchical_bayesian_irl,
    )

    rng = np.random.default_rng(7)
    d = 4
    S = 10
    T = 8
    K = 5
    mu_true = np.array([1.5, -0.8, 0.0, 0.5])
    tau_true = np.array([0.2, 0.5, 0.05, 0.3])

    features_per_seed: list[np.ndarray] = []
    chosen_per_seed: list[np.ndarray] = []
    for s in range(S):
        w_s = mu_true + tau_true * rng.normal(0, 1, size=d)
        feats = rng.normal(0, 1, size=(T, K, d))
        feats_rs = subtract_reference(feats, ref_idx=0)
        ch = sample_boltzmann_choices(feats_rs, w_s, rng=rng)
        features_per_seed.append(feats_rs)
        chosen_per_seed.append(ch)

    post = fit_hierarchical_bayesian_irl(
        features_per_seed,
        chosen_per_seed,
        feature_names=tuple(f"f{k}" for k in range(d)),
        prior_sigma_mu=2.0,
        prior_sigma_tau=1.0,
        draws=500,
        tune=500,
        chains=2,
        seed=11,
        progressbar=False,
    )
    assert isinstance(post, HierarchicalIRLPosterior)
    assert post.n_seeds == S
    assert post.mu_samples.shape == (1000, d)
    # Recovery: μ_true dentro del HDI95 en al menos 3/4 dims
    in_hdi = sum(
        1 for k in range(d)
        if post.mu_hdi95[k, 0] <= mu_true[k] <= post.mu_hdi95[k, 1]
    )
    assert in_hdi >= 3, (
        f"recovery insuficiente: {in_hdi}/{d} dims con μ_true en HDI95. "
        f"mu_true={mu_true}, mu_mean={post.mu_mean}, hdi={post.mu_hdi95}"
    )
