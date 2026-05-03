"""Tests del IRL bayesiano (PyMC).

Tests son lentos (~30-60 s cada uno por NUTS). Usamos draws bajos (300)
para que la suite sea tolerable. La validación seria de recovery se
hace en `irl_recovery_curve.py` con draws altos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Si PyMC no está, salteamos toda la suite. Esto matchea el patrón de
# tests/test_bayesian.py para BEST/Dirichlet.
pytest.importorskip("pymc")

from guatemala_sim.irl.bayesian_irl import (  # noqa: E402
    IRLPosterior,
    fit_bayesian_irl,
    fit_bayesian_irl_point_estimate,
)
from guatemala_sim.irl.recovery import generate_synthetic_dataset  # noqa: E402


# --- smoke tests rápidos ------------------------------------------------------


def test_fit_devuelve_irl_posterior_con_shapes_correctas():
    true_w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(true_w, n_turns=80, n_candidates=4, feature_seed=0, choice_seed=1)
    post = fit_bayesian_irl(
        ds.features, ds.chosen,
        feature_names=("a", "b", "c"),
        draws=300, tune=300, chains=2, seed=11,
    )
    assert isinstance(post, IRLPosterior)
    assert post.n_observations == 80
    assert post.n_candidates == 4
    assert post.w_mean.shape == (3,)
    assert post.w_hdi95.shape == (3, 2)
    assert post.w_samples.shape == (300 * 2, 3)
    assert post.feature_names == ("a", "b", "c")
    # HDI95: lo ≤ hi por construcción
    assert (post.w_hdi95[:, 0] <= post.w_hdi95[:, 1]).all()


def test_fit_validaciones_de_input():
    true_w = np.array([1.0, -0.5])
    ds = generate_synthetic_dataset(true_w, n_turns=20, n_candidates=3, feature_seed=0, choice_seed=1)
    # features 2D
    with pytest.raises(ValueError):
        fit_bayesian_irl(ds.features.reshape(-1, ds.features.shape[-1]), ds.chosen, draws=50, tune=50)
    # chosen mal shape
    with pytest.raises(ValueError):
        fit_bayesian_irl(ds.features, ds.chosen[:5], draws=50, tune=50)
    # chosen fuera de rango
    bad_chosen = ds.chosen.copy()
    bad_chosen[0] = 999
    with pytest.raises(ValueError):
        fit_bayesian_irl(ds.features, bad_chosen, draws=50, tune=50)
    # feature_names mal len
    with pytest.raises(ValueError):
        fit_bayesian_irl(ds.features, ds.chosen, feature_names=("a",), draws=50, tune=50)
    # prior_sigma inválido
    with pytest.raises(ValueError):
        fit_bayesian_irl(ds.features, ds.chosen, prior_sigma=-1.0, draws=50, tune=50)


# --- recovery: el test crítico ------------------------------------------------


@pytest.mark.slow
def test_recupera_w_estrella_dentro_del_hdi95():
    """Validación canónica: con N=600 y w* moderado, el HDI95 debe
    cubrir w* en al menos 4 de 5 dimensiones (cobertura nominal 0.95
    permite 1 fallo en 5 con probabilidad razonable)."""
    true_w = np.array([1.5, -0.8, 0.5, 1.0, -0.3])
    ds = generate_synthetic_dataset(
        true_w, n_turns=600, n_candidates=5,
        feature_seed=42, choice_seed=43,
    )
    post = fit_bayesian_irl(
        ds.features, ds.chosen,
        prior_sigma=2.0,
        draws=800, tune=800, chains=2, seed=11,
    )
    # cobertura por dimensión
    cubre = (true_w >= post.w_hdi95[:, 0]) & (true_w <= post.w_hdi95[:, 1])
    n_cubre = int(cubre.sum())
    assert n_cubre >= 4, (
        f"HDI95 cubre solo {n_cubre}/5 dims. w* = {true_w}, "
        f"HDI95 = {post.w_hdi95.tolist()}"
    )
    # Diagnostics: no divergencias y r-hat decente
    assert post.diverging == 0, f"NUTS divergió {post.diverging} transiciones"
    assert post.rhat_max < 1.05, f"R-hat máximo = {post.rhat_max:.3f}"


# --- API auxiliar -------------------------------------------------------------


def test_w_table_tiene_columnas_esperadas():
    true_w = np.array([1.0, 0.0, -0.5])
    ds = generate_synthetic_dataset(true_w, n_turns=80, feature_seed=0, choice_seed=1)
    post = fit_bayesian_irl(
        ds.features, ds.chosen,
        feature_names=("anti_pobreza", "neutral", "anti_deuda"),
        draws=300, tune=300, chains=2, seed=11,
    )
    df = post.w_table()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"mean", "hdi95_lo", "hdi95_hi", "hdi95_excludes_zero"}
    assert df.index.tolist() == ["anti_pobreza", "neutral", "anti_deuda"]
    # tipos
    assert df["hdi95_excludes_zero"].dtype == bool


def test_w_norm_y_direction_son_consistentes():
    true_w = np.array([2.0, -1.0, 0.5])
    ds = generate_synthetic_dataset(true_w, n_turns=200, feature_seed=0, choice_seed=1)
    post = fit_bayesian_irl(ds.features, ds.chosen, draws=300, tune=300, chains=2, seed=11)
    # dirección es vector unitario
    direction = post.w_direction_mean
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-6
    # norm_mean es positivo y razonable (no NaN, no infinito)
    assert post.w_norm_mean > 0
    assert np.isfinite(post.w_norm_mean)


def test_diagnostics_ok_metodo():
    true_w = np.array([1.0, -0.5])
    ds = generate_synthetic_dataset(true_w, n_turns=100, feature_seed=0, choice_seed=1)
    post = fit_bayesian_irl(ds.features, ds.chosen, draws=300, tune=300, chains=2, seed=11)
    # Con 100 turnos y modelo bien especificado, esperamos diagnostics sanos
    assert post.diagnostics_ok()


# --- compatibilidad con FitFunction de recovery.py ----------------------------


def test_point_estimate_compatible_con_fit_function():
    """fit_bayesian_irl_point_estimate debe tener la firma
    (features, chosen) -> w para servir de FitFunction en
    run_recovery_sweep."""
    true_w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(true_w, n_turns=80, feature_seed=0, choice_seed=1)
    w_hat = fit_bayesian_irl_point_estimate(
        ds.features, ds.chosen,
        draws=300, tune=300, chains=2, seed=11,
    )
    assert w_hat.shape == (3,)
    # No verifico recovery exacto acá — eso es trabajo de los tests de
    # recovery. Solo verifico la firma y que devuelve algo razonable.
    assert np.all(np.isfinite(w_hat))
