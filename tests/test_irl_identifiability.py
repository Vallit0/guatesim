"""Tests del variance_ratio / identified_dimensions de IRLPosterior."""

from __future__ import annotations

import numpy as np
import pytest

from guatemala_sim.irl.bayesian_irl import IRLPosterior
from guatemala_sim.irl.features import OUTCOME_FEATURE_NAMES
from tests.test_irl_posterior_analysis import make_synthetic_posterior


def test_variance_ratio_dim_concentrada():
    """Si el posterior está MUY concentrado, variance_ratio << 1."""
    w_true = np.zeros(6)
    w_true[0] = 2.0
    posterior = make_synthetic_posterior(
        w_true,
        feature_names=OUTCOME_FEATURE_NAMES,
        sample_sigma=0.05,  # mucho menor que prior_sigma=1.0
    )
    ratios = posterior.variance_ratio
    assert ratios.shape == (6,)
    # Cada dim tiene var ≈ 0.05^2 = 0.0025 vs prior_var = 1
    assert np.all(ratios < 0.05)


def test_variance_ratio_dim_no_identificada():
    """Si los samples tienen varianza similar al prior, ratio ≈ 1."""
    rng = np.random.default_rng(0)
    # Samples con varianza idéntica al prior (1.0): la dim NO está
    # identificada. La forma honesta de simular esto es muestrear
    # del prior directamente.
    samples = rng.normal(0, 1.0, size=(2000, 6))
    posterior = IRLPosterior(
        feature_names=OUTCOME_FEATURE_NAMES,
        n_observations=8,
        n_candidates=5,
        w_mean=samples.mean(axis=0),
        w_hdi95=np.array([[s.min(), s.max()] for s in samples.T]),
        w_samples=samples,
        diverging=0,
        rhat_max=1.01,
        ess_bulk_min=500.0,
        prior_sigma=1.0,
    )
    ratios = posterior.variance_ratio
    # Cada dim tiene var ≈ 1, ratio ≈ 1
    assert np.all(ratios > 0.85)
    assert np.all(ratios < 1.15)


def test_identified_dimensions_devuelve_indices():
    w_true = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.05,
    )
    ids = posterior.identified_dimensions(threshold=0.5)
    # Todas las dims tienen sample_sigma=0.05 << prior_sigma=1
    # → todas identificadas
    assert ids == [0, 1, 2, 3, 4, 5]


def test_identified_dimensions_threshold_estricto():
    """Threshold cerca de 0 debe identificar menos dims."""
    rng = np.random.default_rng(0)
    # Mezcla: dims 0-2 muy concentradas, dims 3-5 medianamente
    samples = np.zeros((1000, 6))
    samples[:, :3] = rng.normal(0, 0.05, size=(1000, 3))   # ratio ~ 0.0025
    samples[:, 3:] = rng.normal(0, 0.4, size=(1000, 3))    # ratio ~ 0.16

    posterior = IRLPosterior(
        feature_names=OUTCOME_FEATURE_NAMES,
        n_observations=8, n_candidates=5,
        w_mean=samples.mean(axis=0),
        w_hdi95=np.zeros((6, 2)),
        w_samples=samples,
        diverging=0, rhat_max=1.01, ess_bulk_min=500.0,
        prior_sigma=1.0,
    )
    # Threshold 0.5: 6 dims identificadas (todas ratio < 0.5)
    assert posterior.n_dims_identified(threshold=0.5) == 6
    # Threshold 0.05: solo dims 0-2 identificadas (las concentradas)
    assert posterior.n_dims_identified(threshold=0.05) == 3


def test_identifiability_table_shape_y_columnas():
    posterior = make_synthetic_posterior(
        np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        feature_names=OUTCOME_FEATURE_NAMES,
        sample_sigma=0.05,
    )
    df = posterior.identifiability_table()
    assert len(df) == 6
    expected_cols = {
        "prior_variance", "posterior_variance", "variance_ratio", "identified",
    }
    assert expected_cols.issubset(df.columns)
    # prior_variance constante e igual a prior_sigma**2
    assert (df["prior_variance"] == posterior.prior_sigma ** 2).all()


def test_identified_dimensions_threshold_invalido():
    posterior = make_synthetic_posterior(np.zeros(6))
    with pytest.raises(ValueError):
        posterior.identified_dimensions(threshold=0.0)
    with pytest.raises(ValueError):
        posterior.identified_dimensions(threshold=1.5)


def test_variance_ratio_respeta_prior_sigma_distinto():
    """Si cambia prior_sigma, el ratio debe escalar acordemente."""
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.5, size=(1000, 6))  # var(samples) ≈ 0.25
    for prior_sigma in (0.5, 1.0, 2.0):
        posterior = IRLPosterior(
            feature_names=OUTCOME_FEATURE_NAMES,
            n_observations=8, n_candidates=5,
            w_mean=samples.mean(axis=0),
            w_hdi95=np.zeros((6, 2)),
            w_samples=samples,
            diverging=0, rhat_max=1.01, ess_bulk_min=500.0,
            prior_sigma=prior_sigma,
        )
        ratios = posterior.variance_ratio
        # En cada caso, ratio debe ser ≈ 0.25 / prior_sigma**2
        expected = 0.25 / (prior_sigma ** 2)
        assert np.allclose(ratios.mean(), expected, rtol=0.15), (
            f"prior_sigma={prior_sigma}: ratio mean = {ratios.mean()}, "
            f"esperado ≈ {expected}"
        )
