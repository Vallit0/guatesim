"""Tests del extractor de features φ(s, a) para IRL."""

from __future__ import annotations

import numpy as np
import pytest

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.irl.candidates import generate_candidate_menu
from guatemala_sim.irl.features import (
    N_OUTCOME_FEATURES,
    OUTCOME_FEATURE_NAMES,
    extract_outcome_features,
)


def test_shape_es_seis():
    state = initial_state()
    cand = generate_candidate_menu()[0].presupuesto
    phi = extract_outcome_features(state, cand, feature_seed=0, n_samples=5)
    assert phi.shape == (N_OUTCOME_FEATURES,)
    assert phi.shape == (6,)


def test_nombres_de_features_y_n_son_consistentes():
    assert len(OUTCOME_FEATURE_NAMES) == N_OUTCOME_FEATURES == 6


def test_determinismo_bajo_mismo_seed():
    state = initial_state()
    cand = generate_candidate_menu()[1].presupuesto
    phi_a = extract_outcome_features(state, cand, feature_seed=42, n_samples=10)
    phi_b = extract_outcome_features(state, cand, feature_seed=42, n_samples=10)
    np.testing.assert_array_equal(phi_a, phi_b)


def test_seeds_distintos_dan_features_distintas():
    state = initial_state()
    cand = generate_candidate_menu()[1].presupuesto
    phi_a = extract_outcome_features(state, cand, feature_seed=1, n_samples=5)
    phi_b = extract_outcome_features(state, cand, feature_seed=2, n_samples=5)
    # Con n_samples=5 y mucho ruido gaussiano, deberían diferir en al menos
    # una componente.
    assert not np.allclose(phi_a, phi_b)


def test_n_samples_invalido_levanta():
    state = initial_state()
    cand = generate_candidate_menu()[0].presupuesto
    with pytest.raises(ValueError):
        extract_outcome_features(state, cand, feature_seed=0, n_samples=0)


def test_no_muta_state_input():
    state = initial_state()
    snapshot = state.model_dump_json()
    cand = generate_candidate_menu()[2].presupuesto
    _ = extract_outcome_features(state, cand, feature_seed=0, n_samples=3)
    assert state.model_dump_json() == snapshot


def test_candidatos_distintos_dan_features_distintas():
    """Si fiscal_prudente y desarrollo_humano dieran el mismo φ, el IRL
    no podría distinguirlos. Esa es una sanity-check del simulador."""
    state = initial_state()
    menu = generate_candidate_menu()
    fiscal = menu[1].presupuesto      # fiscal_prudente
    desarrollo = menu[2].presupuesto  # desarrollo_humano
    # n_samples alto para promediar el ruido y aislar el efecto de la decisión
    phi_fiscal = extract_outcome_features(state, fiscal, feature_seed=99, n_samples=50)
    phi_desarrollo = extract_outcome_features(state, desarrollo, feature_seed=99, n_samples=50)
    assert not np.allclose(phi_fiscal, phi_desarrollo, atol=1e-3)


def test_desarrollo_humano_reduce_pobreza_mas_que_fiscal_prudente():
    """Test sustantivo: el simulador penaliza menos pobreza cuando el
    presupuesto tiene mayor peso en gasto social. desarrollo_humano
    debería tener mayor `anti_pobreza` (componente 0) que fiscal_prudente.
    Si esto falla, hay un bug en `world.macro` o las features están mal
    firmadas."""
    state = initial_state()
    menu = {c.name: c.presupuesto for c in generate_candidate_menu()}
    # n_samples grande para que el promedio supere el ruido (σ ≈ 1)
    phi_fiscal = extract_outcome_features(
        state, menu["fiscal_prudente"], feature_seed=7, n_samples=200,
    )
    phi_desarrollo = extract_outcome_features(
        state, menu["desarrollo_humano"], feature_seed=7, n_samples=200,
    )
    # componente 0 = anti_pobreza (mayor = menos pobreza inducida)
    assert phi_desarrollo[0] > phi_fiscal[0], (
        f"desarrollo_humano debería reducir más pobreza que fiscal_prudente; "
        f"obtuve anti_pobreza desarrollo={phi_desarrollo[0]:.3f} "
        f"fiscal={phi_fiscal[0]:.3f}"
    )


def test_anti_desviacion_inflacion_es_no_positivo():
    """La componente 4 es -|inflación - 4|, así que nunca puede ser > 0."""
    state = initial_state()
    cand = generate_candidate_menu()[0].presupuesto
    phi = extract_outcome_features(state, cand, feature_seed=11, n_samples=30)
    # Promedio Monte Carlo de cantidades ≤ 0 puede ser ≈ 0 pero no positivo.
    assert phi[4] <= 1e-9
