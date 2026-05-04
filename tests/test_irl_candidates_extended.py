"""Tests del menú extendido K=5/7/9 y leave-one-out."""

from __future__ import annotations

import pytest

from guatemala_sim.actions import PresupuestoAnual
from guatemala_sim.irl.candidates import generate_candidate_menu
from guatemala_sim.irl.candidates_extended import (
    generate_candidate_menu_k,
    menu_leave_one_out,
)


def _suma(p: PresupuestoAnual) -> float:
    return (
        p.salud + p.educacion + p.seguridad + p.infraestructura
        + p.agro_desarrollo_rural + p.proteccion_social + p.servicio_deuda
        + p.justicia + p.otros
    )


def test_k5_es_identico_al_base():
    base = generate_candidate_menu()
    k5 = generate_candidate_menu_k(5)
    assert [c.name for c in k5] == [c.name for c in base]


def test_k7_extiende_k5():
    base = generate_candidate_menu()
    k7 = generate_candidate_menu_k(7)
    assert len(k7) == 7
    assert [c.name for c in k7[:5]] == [c.name for c in base]
    extras = [c.name for c in k7[5:]]
    assert "growth_first" in extras
    assert "rights_first" in extras


def test_k9_extiende_k7():
    k7 = generate_candidate_menu_k(7)
    k9 = generate_candidate_menu_k(9)
    assert len(k9) == 9
    assert [c.name for c in k9[:7]] == [c.name for c in k7]
    extras = [c.name for c in k9[7:]]
    assert "austerity_max" in extras
    assert "populist_high_spend" in extras


def test_todos_los_extras_suman_100_pp_tolerancia_1():
    for k in (7, 9):
        for c in generate_candidate_menu_k(k):
            s = _suma(c.presupuesto)
            assert abs(s - 100.0) <= 1.0, f"K={k} candidato {c.name}: suma = {s:.2f}"


def test_k_no_soportada_levanta():
    with pytest.raises(ValueError, match="k debe ser 5, 7 ó 9"):
        generate_candidate_menu_k(6)
    with pytest.raises(ValueError, match="k debe ser 5, 7 ó 9"):
        generate_candidate_menu_k(11)


def test_extras_son_distintos_de_los_base():
    base_names = {c.name for c in generate_candidate_menu()}
    for k in (7, 9):
        extras = [c for c in generate_candidate_menu_k(k) if c.name not in base_names]
        # los extras tienen presupuestos distintos a los base
        for e in extras:
            for b in generate_candidate_menu():
                assert e.presupuesto != b.presupuesto, (
                    f"extra {e.name} duplica base {b.name}"
                )


def test_leave_one_out_drop_0():
    sub = menu_leave_one_out(0)
    assert len(sub) == 4
    assert "status_quo_uniforme" not in [c.name for c in sub]


def test_leave_one_out_drop_3():
    sub = menu_leave_one_out(3)
    assert len(sub) == 4
    nombres = [c.name for c in sub]
    assert "seguridad_primero" not in nombres
    assert "status_quo_uniforme" in nombres  # el ancla debe seguir


def test_leave_one_out_indice_invalido():
    with pytest.raises(ValueError):
        menu_leave_one_out(-1)
    with pytest.raises(ValueError):
        menu_leave_one_out(5)


def test_growth_first_orientacion_esperada():
    k7 = generate_candidate_menu_k(7)
    by_name = {c.name: c.presupuesto for c in k7}
    g = by_name["growth_first"]
    # infraestructura + agro dominantes
    assert g.infraestructura >= 20
    assert g.agro_desarrollo_rural >= 15
    # menos peso a deuda/proteccion que en otros arquetipos similares
    assert g.servicio_deuda <= 12


def test_rights_first_orientacion_esperada():
    k7 = generate_candidate_menu_k(7)
    by_name = {c.name: c.presupuesto for c in k7}
    r = by_name["rights_first"]
    assert r.proteccion_social >= 20
    assert r.justicia >= 15


def test_austerity_max_orientacion_esperada():
    k9 = generate_candidate_menu_k(9)
    by_name = {c.name: c.presupuesto for c in k9}
    a = by_name["austerity_max"]
    # mucho más prudente que `fiscal_prudente`
    base_prudente = next(c for c in generate_candidate_menu() if c.name == "fiscal_prudente")
    assert a.servicio_deuda > base_prudente.presupuesto.servicio_deuda


def test_populist_orientacion_esperada():
    k9 = generate_candidate_menu_k(9)
    by_name = {c.name: c.presupuesto for c in k9}
    p = by_name["populist_high_spend"]
    assert p.proteccion_social >= 20
    assert p.servicio_deuda <= 5
