"""Tests del likelihood Boltzmann para IRL bayesiano."""

from __future__ import annotations

import numpy as np
import pytest

from guatemala_sim.irl.boltzmann import (
    boltzmann_choice_probs,
    boltzmann_log_likelihood,
    boltzmann_log_probs,
    sample_boltzmann_choices,
    subtract_reference,
)


# --- subtract_reference ------------------------------------------------------


def test_subtract_reference_shape_preservada():
    f = np.random.default_rng(0).standard_normal((4, 5, 3))
    g = subtract_reference(f, ref_idx=0)
    assert g.shape == f.shape


def test_subtract_reference_referencia_es_cero():
    f = np.random.default_rng(0).standard_normal((4, 5, 3))
    for ref_idx in range(5):
        g = subtract_reference(f, ref_idx=ref_idx)
        np.testing.assert_array_equal(g[:, ref_idx, :], np.zeros((4, 3)))


def test_subtract_reference_otros_se_mueven_correctamente():
    f = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],   # turno 0
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], # turno 1
    ])  # shape (2, 3, 2)
    g = subtract_reference(f, ref_idx=0)
    # candidato 1 - candidato 0
    np.testing.assert_array_equal(g[0, 1, :], np.array([2.0, 2.0]))
    np.testing.assert_array_equal(g[0, 2, :], np.array([4.0, 4.0]))
    np.testing.assert_array_equal(g[1, 1, :], np.array([2.0, 2.0]))


def test_subtract_reference_rechaza_ref_idx_fuera_de_rango():
    f = np.zeros((2, 5, 3))
    with pytest.raises(ValueError):
        subtract_reference(f, ref_idx=5)
    with pytest.raises(ValueError):
        subtract_reference(f, ref_idx=-1)


def test_subtract_reference_rechaza_ndim_invalido():
    with pytest.raises(ValueError):
        subtract_reference(np.zeros((5, 3)))  # 2D
    with pytest.raises(ValueError):
        subtract_reference(np.zeros((2, 5, 3, 1)))  # 4D


# --- boltzmann_log_probs / choice_probs --------------------------------------


def test_log_probs_shape_correcta():
    f = np.random.default_rng(0).standard_normal((10, 5, 6))
    w = np.random.default_rng(1).standard_normal(6)
    lp = boltzmann_log_probs(f, w)
    assert lp.shape == (10, 5)


def test_probs_suman_uno_por_turno():
    rng = np.random.default_rng(42)
    f = rng.standard_normal((20, 5, 6))
    w = rng.standard_normal(6)
    probs = boltzmann_choice_probs(f, w)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(20), atol=1e-10)


def test_w_cero_da_distribucion_uniforme():
    f = np.random.default_rng(0).standard_normal((10, 5, 6))
    w = np.zeros(6)
    probs = boltzmann_choice_probs(f, w)
    np.testing.assert_allclose(probs, np.full((10, 5), 1.0 / 5), atol=1e-12)


def test_w_grande_concentra_en_argmax():
    """Cuando ‖w‖ → ∞, la Boltzmann colapsa a la indicador del argmax.

    Construimos features donde sabemos cuál candidato gana: features =
    one-hot por candidato, w positivo que privilegia el candidato 2.
    """
    # Para cada turno, candidato k tiene feature = (0, ..., k, ..., 0)
    # Con w = (1, 1, 1, 1, 1), utility = k → argmax es siempre el último.
    T, K, d = 3, 5, 5
    f = np.zeros((T, K, d))
    for k in range(K):
        f[:, k, k] = float(k)
    w = np.full(d, 100.0)  # gran escala
    probs = boltzmann_choice_probs(f, w)
    # candidato K-1 tiene utility = (K-1)·100 = 400 frente a 0, 100, 200, 300
    # En log-probs debería estar a ≥ 100 unidades del segundo, así que ≈ 1.0
    np.testing.assert_allclose(probs[:, K - 1], np.ones(T), atol=1e-10)


def test_log_probs_consistente_con_choice_probs():
    rng = np.random.default_rng(99)
    f = rng.standard_normal((5, 4, 3))
    w = rng.standard_normal(3)
    lp = boltzmann_log_probs(f, w)
    p = boltzmann_choice_probs(f, w)
    np.testing.assert_allclose(np.exp(lp), p)


def test_subtract_reference_no_cambia_choice_probs():
    """La sustracción de referencia es un cambio aditivo en utilidades —
    cancela en el softmax. Las choice_probs deben ser idénticas."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal((5, 4, 3))
    w = rng.standard_normal(3)
    p_raw = boltzmann_choice_probs(f, w)
    p_sub = boltzmann_choice_probs(subtract_reference(f, ref_idx=0), w)
    np.testing.assert_allclose(p_raw, p_sub, atol=1e-12)


def test_log_probs_rechaza_w_shape_invalida():
    f = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        boltzmann_log_probs(f, np.zeros(5))     # d incorrecto
    with pytest.raises(ValueError):
        boltzmann_log_probs(f, np.zeros((4, 1)))  # 2D


def test_log_probs_rechaza_features_2d():
    with pytest.raises(ValueError):
        boltzmann_log_probs(np.zeros((3, 4)), np.zeros(4))


# --- boltzmann_log_likelihood -----------------------------------------------


def test_log_likelihood_es_no_positivo():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((10, 5, 6))
    w = rng.standard_normal(6)
    chosen = rng.integers(0, 5, size=10)
    ll = boltzmann_log_likelihood(f, chosen, w)
    assert ll <= 1e-10  # tolerancia numérica


def test_log_likelihood_w_cero_es_minus_T_log_K():
    """Con w=0, P uniforme = 1/K; suma de T log(1/K) = -T·log(K)."""
    T, K, d = 7, 5, 4
    f = np.random.default_rng(0).standard_normal((T, K, d))
    chosen = np.array([0, 1, 2, 3, 4, 0, 1])
    ll = boltzmann_log_likelihood(f, chosen, np.zeros(d))
    np.testing.assert_allclose(ll, -T * np.log(K))


def test_log_likelihood_es_escalar_python_float():
    f = np.random.default_rng(0).standard_normal((3, 4, 2))
    chosen = np.array([0, 1, 2])
    ll = boltzmann_log_likelihood(f, chosen, np.zeros(2))
    assert isinstance(ll, float)


def test_log_likelihood_rechaza_chosen_invalido():
    f = np.zeros((3, 4, 2))
    w = np.zeros(2)
    with pytest.raises(ValueError):
        boltzmann_log_likelihood(f, np.array([0, 4, 1]), w)  # 4 fuera de [0,4)
    with pytest.raises(ValueError):
        boltzmann_log_likelihood(f, np.array([0, -1, 1]), w)
    with pytest.raises(ValueError):
        boltzmann_log_likelihood(f, np.array([0, 1]), w)  # shape distinta a T


# --- sample_boltzmann_choices -----------------------------------------------


def test_sample_devuelve_indices_validos():
    rng_data = np.random.default_rng(0)
    f = rng_data.standard_normal((100, 5, 6))
    w = rng_data.standard_normal(6)
    rng_sample = np.random.default_rng(1)
    chosen = sample_boltzmann_choices(f, w, rng_sample)
    assert chosen.shape == (100,)
    assert chosen.dtype.kind == "i"
    assert ((chosen >= 0) & (chosen < 5)).all()


def test_sample_es_determinista_bajo_mismo_seed():
    rng_data = np.random.default_rng(0)
    f = rng_data.standard_normal((50, 5, 6))
    w = rng_data.standard_normal(6)
    a = sample_boltzmann_choices(f, w, np.random.default_rng(11))
    b = sample_boltzmann_choices(f, w, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


def test_sample_seeds_distintos_dan_resultados_distintos():
    rng_data = np.random.default_rng(0)
    f = rng_data.standard_normal((50, 5, 6))
    w = rng_data.standard_normal(6)
    a = sample_boltzmann_choices(f, w, np.random.default_rng(1))
    b = sample_boltzmann_choices(f, w, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_sample_w_cero_da_frecuencias_aprox_uniformes():
    """Con w=0 todos los candidatos son equiprobables. Sobre N grande,
    las frecuencias empíricas deberían ≈ 1/K. Test asintótico, no
    determinístico — usamos N=5000 y tolerancia de 0.03 (≈ 3·sd)."""
    T, K, d = 5000, 5, 6
    f = np.random.default_rng(0).standard_normal((T, K, d))
    chosen = sample_boltzmann_choices(f, np.zeros(d), np.random.default_rng(123))
    counts = np.bincount(chosen, minlength=K)
    freq = counts / T
    np.testing.assert_allclose(freq, np.full(K, 1.0 / K), atol=0.03)


def test_sample_w_grande_siempre_elige_argmax():
    """Construimos un escenario donde el argmax es claro y w grande hace
    que la elección sea casi determinista."""
    T, K, d = 100, 5, 5
    f = np.zeros((T, K, d))
    for k in range(K):
        f[:, k, k] = float(k)
    w = np.full(d, 50.0)
    chosen = sample_boltzmann_choices(f, w, np.random.default_rng(0))
    # Casi todos deben ser K-1
    assert (chosen == K - 1).sum() >= T - 1


# --- integración con features.py + candidates.py -----------------------------


def test_pipeline_completo_con_simulador_real():
    """Smoke test: extraer features de candidatos reales, restar referencia,
    samplear elecciones desde un w sintético, computar log-likelihood.
    Este es el ciclo completo que va a ejecutar el modelo bayesiano."""
    from guatemala_sim.bootstrap import initial_state
    from guatemala_sim.irl import (
        N_OUTCOME_FEATURES,
        REFERENCE_CANDIDATE_INDEX,
        extract_outcome_features,
        generate_candidate_menu,
    )

    state = initial_state()
    menu = generate_candidate_menu()
    K = len(menu)
    T = 1  # un turno; este test es un smoke check, no validación estadística

    features = np.zeros((T, K, N_OUTCOME_FEATURES))
    for k, cand in enumerate(menu):
        features[0, k, :] = extract_outcome_features(
            state, cand.presupuesto, feature_seed=k, n_samples=5,
        )

    features_anchored = subtract_reference(features, ref_idx=REFERENCE_CANDIDATE_INDEX)
    np.testing.assert_array_equal(
        features_anchored[0, REFERENCE_CANDIDATE_INDEX, :],
        np.zeros(N_OUTCOME_FEATURES),
    )

    # un w sintético: valoramos pro_aprobacion y anti_pobreza
    w = np.array([1.0, 0.5, 1.0, 0.3, 0.2, 0.5])
    chosen = sample_boltzmann_choices(features_anchored, w, np.random.default_rng(0))
    assert chosen.shape == (T,)

    ll = boltzmann_log_likelihood(features_anchored, chosen, w)
    assert ll <= 1e-10
