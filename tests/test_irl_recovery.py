"""Tests del harness de validación con sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.irl.recovery import (
    RecoveryDataset,
    RecoveryMetrics,
    compute_recovery_metrics,
    fit_mle_boltzmann,
    generate_synthetic_dataset,
    run_recovery_sweep,
)


# --- generate_synthetic_dataset ----------------------------------------------


def test_generate_synthetic_dataset_shapes():
    w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(w, n_turns=20, n_candidates=5)
    assert ds.features.shape == (20, 5, 3)
    assert ds.chosen.shape == (20,)
    assert ds.true_w.shape == (3,)
    assert ds.n_turns == 20
    assert ds.n_candidates == 5
    assert ds.n_features == 3


def test_generate_synthetic_dataset_features_ancladas_en_indice_cero():
    w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(w, n_turns=10, n_candidates=4)
    np.testing.assert_array_equal(ds.features[:, 0, :], np.zeros((10, 3)))


def test_generate_synthetic_dataset_es_determinista():
    w = np.array([1.0, -0.5, 0.3])
    a = generate_synthetic_dataset(w, n_turns=15, feature_seed=7, choice_seed=11)
    b = generate_synthetic_dataset(w, n_turns=15, feature_seed=7, choice_seed=11)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.chosen, b.chosen)


def test_generate_synthetic_dataset_seeds_distintos_dan_resultados_distintos():
    w = np.array([1.0, -0.5, 0.3])
    a = generate_synthetic_dataset(w, n_turns=50, feature_seed=1, choice_seed=2)
    b = generate_synthetic_dataset(w, n_turns=50, feature_seed=3, choice_seed=4)
    assert not np.array_equal(a.features, b.features)


def test_generate_synthetic_dataset_chosen_indices_validos():
    w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(w, n_turns=100, n_candidates=5)
    assert ((ds.chosen >= 0) & (ds.chosen < 5)).all()


def test_generate_synthetic_dataset_no_muta_true_w():
    w = np.array([1.0, -0.5, 0.3])
    snapshot = w.copy()
    _ = generate_synthetic_dataset(w, n_turns=10)
    np.testing.assert_array_equal(w, snapshot)


def test_generate_synthetic_dataset_rechaza_inputs_invalidos():
    w = np.array([1.0])
    with pytest.raises(ValueError):
        generate_synthetic_dataset(w, n_turns=0)
    with pytest.raises(ValueError):
        generate_synthetic_dataset(w, n_candidates=1)
    with pytest.raises(ValueError):
        generate_synthetic_dataset(np.zeros((2, 3)))  # true_w 2D


# --- compute_recovery_metrics ------------------------------------------------


def test_metrics_caso_identico_da_rmse_cero_y_cos_uno():
    w = np.array([1.0, -0.5, 2.0])
    m = compute_recovery_metrics(w, w.copy())
    assert m.rmse < 1e-12
    assert abs(m.cosine_similarity - 1.0) < 1e-12
    assert m.angle_degrees < 1e-6
    assert abs(m.norm_ratio - 1.0) < 1e-12


def test_metrics_caso_collinear_positivo_da_cos_uno():
    w_true = np.array([1.0, -0.5, 2.0])
    w_est = 3.0 * w_true  # mismo dirección, distinta magnitud
    m = compute_recovery_metrics(w_true, w_est)
    assert abs(m.cosine_similarity - 1.0) < 1e-10
    assert abs(m.norm_ratio - 3.0) < 1e-10


def test_metrics_caso_collinear_negativo_da_cos_menos_uno():
    w_true = np.array([1.0, -0.5, 2.0])
    w_est = -1.5 * w_true
    m = compute_recovery_metrics(w_true, w_est)
    assert abs(m.cosine_similarity + 1.0) < 1e-10
    assert abs(m.angle_degrees - 180.0) < 1e-6


def test_metrics_caso_ortogonal_da_cos_cero():
    w_true = np.array([1.0, 0.0])
    w_est = np.array([0.0, 1.0])
    m = compute_recovery_metrics(w_true, w_est)
    assert abs(m.cosine_similarity) < 1e-12
    assert abs(m.angle_degrees - 90.0) < 1e-6


def test_metrics_con_hdi95_calcula_cobertura():
    w_true = np.array([1.0, -0.5, 2.0, 0.3])
    w_est = np.array([0.9, -0.4, 1.8, 0.5])
    # HDI95 que cubre 3 de 4 dimensiones
    hdi = np.array([
        [0.5, 1.5],   # cubre w_true[0] = 1.0  ✓
        [-0.6, -0.3], # cubre w_true[1] = -0.5 ✓
        [1.5, 2.2],   # cubre w_true[2] = 2.0  ✓
        [0.6, 0.9],   # NO cubre w_true[3] = 0.3 ✗
    ])
    m = compute_recovery_metrics(w_true, w_est, estimated_w_hdi95=hdi)
    assert m.coverage_hdi95 == 0.75
    np.testing.assert_array_equal(m.coverage_per_dim, [True, True, True, False])


def test_metrics_rechaza_shape_mismatch():
    with pytest.raises(ValueError):
        compute_recovery_metrics(np.zeros(3), np.zeros(4))


def test_metrics_rechaza_hdi_mal_formado():
    with pytest.raises(ValueError):
        compute_recovery_metrics(np.zeros(3), np.zeros(3), estimated_w_hdi95=np.zeros((3, 3)))


# --- fit_mle_boltzmann -------------------------------------------------------


def test_mle_recupera_w_estrella_con_datos_abundantes():
    """El test crítico: con N=3000 muestras y w* moderado, MLE debe
    recuperar la dirección con cosine ≥ 0.95 y la magnitud dentro de
    ±30 %."""
    true_w = np.array([1.5, -0.8, 0.5, 0.2, 1.0, -0.3])
    ds = generate_synthetic_dataset(
        true_w, n_turns=3000, n_candidates=5,
        feature_seed=42, choice_seed=43,
    )
    w_hat = fit_mle_boltzmann(ds.features, ds.chosen)
    m = compute_recovery_metrics(true_w, w_hat)
    assert m.cosine_similarity > 0.95, (
        f"MLE no recuperó la dirección de w*: cos_sim = {m.cosine_similarity:.4f}, "
        f"w_hat = {w_hat.round(3)}, w* = {true_w}"
    )
    assert 0.7 < m.norm_ratio < 1.3, f"norm_ratio = {m.norm_ratio:.3f} fuera de [0.7, 1.3]"


def test_mle_devuelve_shape_correcta():
    true_w = np.array([1.0, -0.5, 0.3, 0.8])
    ds = generate_synthetic_dataset(true_w, n_turns=100)
    w_hat = fit_mle_boltzmann(ds.features, ds.chosen)
    assert w_hat.shape == (4,)


def test_mle_es_determinista():
    """L-BFGS-B desde mismo punto inicial sobre misma data debe converger
    al mismo óptimo."""
    true_w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(true_w, n_turns=200, feature_seed=0, choice_seed=1)
    a = fit_mle_boltzmann(ds.features, ds.chosen)
    b = fit_mle_boltzmann(ds.features, ds.chosen)
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_mle_rechaza_initial_w_shape_invalido():
    true_w = np.array([1.0, -0.5, 0.3])
    ds = generate_synthetic_dataset(true_w, n_turns=50)
    with pytest.raises(ValueError):
        fit_mle_boltzmann(ds.features, ds.chosen, initial_w=np.zeros(5))


def test_mle_con_l2_reg_acerca_a_cero():
    """Con regularización fuerte, el MLE debe contraerse hacia 0."""
    true_w = np.array([2.0, -1.5, 1.0])
    ds = generate_synthetic_dataset(true_w, n_turns=200)
    w_unreg = fit_mle_boltzmann(ds.features, ds.chosen, l2_reg=0.0)
    w_reg = fit_mle_boltzmann(ds.features, ds.chosen, l2_reg=10.0)
    assert np.linalg.norm(w_reg) < np.linalg.norm(w_unreg)


# --- run_recovery_sweep ------------------------------------------------------


def test_sweep_devuelve_dataframe_con_columnas_esperadas():
    true_w = np.array([1.0, -0.5, 0.3])
    df = run_recovery_sweep(
        true_w, sample_sizes=[50, 200], n_replications=2,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "N", "replication", "rmse", "cosine_similarity",
        "angle_degrees", "norm_ratio", "log_likelihood_at_w_hat",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 4  # 2 N × 2 replications


def test_sweep_rmse_decrece_con_n_en_promedio():
    """Test asintótico: el RMSE promedio debe ser menor para N más grande.

    Usamos N=100 vs N=2000, con 3 réplicas cada uno. La diferencia
    debería ser robusta."""
    true_w = np.array([1.5, -0.8, 0.5, 1.0])
    df = run_recovery_sweep(
        true_w, sample_sizes=[100, 2000], n_replications=3, base_seed=0,
    )
    rmse_pequeno = df[df["N"] == 100]["rmse"].mean()
    rmse_grande = df[df["N"] == 2000]["rmse"].mean()
    assert rmse_grande < rmse_pequeno, (
        f"RMSE no decreció con N: N=100 → {rmse_pequeno:.4f}, "
        f"N=2000 → {rmse_grande:.4f}"
    )
