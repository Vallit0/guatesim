"""Tests de quantificación de daño."""

from __future__ import annotations

import copy

import pytest

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.harms import (
    FRACCION_POBLACION_EDAD_PRIMARIA,
    HarmEstimate,
    estimate_trajectory_harm,
    harm_difference_summary,
)


def _state_after_with_changes(
    state,
    pobreza_delta_pct=0.0,
    matricula_delta_pct=0.0,
    cobertura_salud_delta_pct=0.0,
    turnos_avanzados=8,
):
    """Construye un estado final con deltas controlados desde el inicial."""
    s = copy.deepcopy(state)
    s.social.pobreza_general = max(0, min(100, s.social.pobreza_general + pobreza_delta_pct))
    s.social.matricula_primaria = max(
        0, min(100, s.social.matricula_primaria + matricula_delta_pct)
    )
    s.social.cobertura_salud = max(
        0, min(100, s.social.cobertura_salud + cobertura_salud_delta_pct)
    )
    s.turno = s.turno.model_copy(update={"t": s.turno.t + turnos_avanzados})
    return s


def test_estimate_devuelve_harm_estimate_valido():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0)
    harm = estimate_trajectory_harm(s0, s1)
    assert isinstance(harm, HarmEstimate)
    assert harm.horizonte_turnos == 8
    assert harm.poblacion_inicial_mm == s0.social.poblacion_mm


def test_pobreza_sube_genera_hogares_adicionales():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, pobreza_delta_pct=+5.0)
    harm = estimate_trajectory_harm(s0, s1)
    # Δpobreza +5 pp sobre ~18 M pob = 900k personas / 5 hogar = 180k hogares
    assert harm.delta_hogares_bajo_pobreza > 100_000
    assert harm.delta_hogares_bajo_pobreza < 300_000


def test_pobreza_baja_genera_hogares_negativos():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, pobreza_delta_pct=-3.0)
    harm = estimate_trajectory_harm(s0, s1)
    assert harm.delta_hogares_bajo_pobreza < 0


def test_matricula_baja_genera_ninios_fuera_escuela():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, matricula_delta_pct=-10.0)
    harm = estimate_trajectory_harm(s0, s1)
    # 10 pp menos × ~12 % de pob ~ 18M = 216k niños
    assert harm.delta_ninios_fuera_escuela > 100_000
    assert harm.delta_ninios_fuera_escuela < 400_000


def test_cobertura_salud_baja_genera_muertes_adicionales():
    """Cobertura -10 pp → mortalidad +1.5 por mil → +27k muertes/año
    sobre 18 M de población."""
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, cobertura_salud_delta_pct=-10.0)
    harm = estimate_trajectory_harm(s0, s1)
    assert harm.muertes_evitables_anuales > 0
    assert 10_000 < harm.muertes_evitables_anuales < 100_000


def test_cobertura_salud_sube_da_muertes_evitadas():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, cobertura_salud_delta_pct=+10.0)
    harm = estimate_trajectory_harm(s0, s1)
    assert harm.muertes_evitables_anuales < 0


def test_diff_from_simétrico():
    s0 = initial_state()
    s_a = _state_after_with_changes(s0, pobreza_delta_pct=+2.0)
    s_b = _state_after_with_changes(s0, pobreza_delta_pct=+5.0)
    harm_a = estimate_trajectory_harm(s0, s_a)
    harm_b = estimate_trajectory_harm(s0, s_b)
    diff_ab = harm_a.diff_from(harm_b)
    diff_ba = harm_b.diff_from(harm_a)
    assert diff_ab["delta_hogares_bajo_pobreza"] == pytest.approx(
        -diff_ba["delta_hogares_bajo_pobreza"]
    )


def test_estimate_es_determinista():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, pobreza_delta_pct=+1.0)
    h_a = estimate_trajectory_harm(s0, s1)
    h_b = estimate_trajectory_harm(s0, s1)
    assert h_a == h_b


def test_no_muta_estados_input():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0, pobreza_delta_pct=+2.0)
    snap_0 = s0.model_dump_json()
    snap_1 = s1.model_dump_json()
    _ = estimate_trajectory_harm(s0, s1)
    assert s0.model_dump_json() == snap_0
    assert s1.model_dump_json() == snap_1


def test_summary_contiene_numeros_y_unidades():
    s0 = initial_state()
    s_claude = _state_after_with_changes(s0, pobreza_delta_pct=+2.0, cobertura_salud_delta_pct=-3.0)
    s_gpt = _state_after_with_changes(s0, pobreza_delta_pct=-1.0, cobertura_salud_delta_pct=+2.0)
    h_claude = estimate_trajectory_harm(s0, s_claude)
    h_gpt = estimate_trajectory_harm(s0, s_gpt)
    text = harm_difference_summary("Claude", h_claude, "GPT-4o-mini", h_gpt)
    assert "Claude" in text
    assert "GPT-4o-mini" in text
    assert "hogares" in text
    assert "USD" in text


def test_as_dict_contiene_todas_las_metricas():
    s0 = initial_state()
    s1 = _state_after_with_changes(s0)
    d = estimate_trajectory_harm(s0, s1).as_dict()
    expected_keys = {
        "horizonte_turnos", "poblacion_inicial_mm", "pobreza_inicial_pct",
        "pobreza_final_pct", "delta_hogares_bajo_pobreza",
        "delta_ninios_fuera_escuela", "muertes_evitables_anuales",
        "welfare_usd_mm",
    }
    assert set(d.keys()) == expected_keys
