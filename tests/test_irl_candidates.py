"""Tests del menú de candidatos IRL."""

from __future__ import annotations

import pytest

from guatemala_sim.actions import PresupuestoAnual
from guatemala_sim.irl.candidates import (
    Candidate,
    REFERENCE_CANDIDATE_INDEX,
    generate_candidate_menu,
)


def _suma(p: PresupuestoAnual) -> float:
    return (
        p.salud
        + p.educacion
        + p.seguridad
        + p.infraestructura
        + p.agro_desarrollo_rural
        + p.proteccion_social
        + p.servicio_deuda
        + p.justicia
        + p.otros
    )


def test_menu_tiene_5_candidatos():
    menu = generate_candidate_menu()
    assert len(menu) == 5


def test_cada_candidato_es_Candidate():
    menu = generate_candidate_menu()
    for c in menu:
        assert isinstance(c, Candidate)
        assert isinstance(c.presupuesto, PresupuestoAnual)


def test_cada_candidato_suma_100_pp_tolerancia_1():
    menu = generate_candidate_menu()
    for c in menu:
        s = _suma(c.presupuesto)
        assert abs(s - 100.0) <= 1.0, f"{c.name}: suma = {s:.2f}"


def test_orden_estable_y_anchor_es_primero():
    menu = generate_candidate_menu()
    nombres = [c.name for c in menu]
    assert nombres == [
        "status_quo_uniforme",
        "fiscal_prudente",
        "desarrollo_humano",
        "seguridad_primero",
        "equilibrado",
    ]
    assert menu[REFERENCE_CANDIDATE_INDEX].name == "status_quo_uniforme"


def test_referencia_es_uniforme():
    menu = generate_candidate_menu()
    ref = menu[REFERENCE_CANDIDATE_INDEX].presupuesto
    valores = [
        ref.salud, ref.educacion, ref.seguridad, ref.infraestructura,
        ref.agro_desarrollo_rural, ref.proteccion_social, ref.servicio_deuda,
        ref.justicia, ref.otros,
    ]
    # Cada partida ≈ 100/9 ≈ 11.11; tolerancia 0.05
    for v in valores:
        assert abs(v - 100.0 / 9.0) < 0.05


def test_candidatos_son_distintos_entre_si():
    """Si dos candidatos fueran idénticos, el likelihood Boltzmann
    perdería identificabilidad sobre esa partida."""
    menu = generate_candidate_menu()
    presupuestos_dump = [tuple(sorted(c.presupuesto.model_dump().items())) for c in menu]
    assert len(set(presupuestos_dump)) == 5, "todos los candidatos deben ser únicos"


def test_arquetipos_tienen_la_orientacion_esperada():
    """Sanity check de que los nombres reflejan la asignación."""
    menu = {c.name: c.presupuesto for c in generate_candidate_menu()}

    # fiscal_prudente: deuda alta, social bajo
    assert menu["fiscal_prudente"].servicio_deuda > 20
    assert menu["fiscal_prudente"].salud < 12

    # desarrollo_humano: salud + educación dominantes, deuda baja
    assert menu["desarrollo_humano"].salud > 15
    assert menu["desarrollo_humano"].educacion > 15
    assert menu["desarrollo_humano"].servicio_deuda < 8

    # seguridad_primero: seguridad dominante
    assert menu["seguridad_primero"].seguridad > 20
    assert menu["seguridad_primero"].justicia > 10


def test_generate_candidate_menu_devuelve_lista_independiente():
    """Mutar la lista devuelta no debe afectar futuras llamadas."""
    m1 = generate_candidate_menu()
    m1.pop()
    m2 = generate_candidate_menu()
    assert len(m2) == 5
