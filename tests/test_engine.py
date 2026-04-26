"""Tests del turn loop con decisión dummy."""

from __future__ import annotations

import numpy as np

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.engine import DummyDecisionMaker, run_turn


def test_un_turno_preserva_validez():
    state = initial_state()
    rng = np.random.default_rng(1)
    dm = DummyDecisionMaker(rng)
    new_state, rec = run_turn(state, dm, rng)
    assert new_state.turno.t == state.turno.t + 1
    assert new_state.turno.fecha.year == state.turno.fecha.year + 1
    assert rec.decision["exterior"]["alineamiento_priorizado"] == "multilateral"
    assert new_state.macro.pib_usd_mm > 0


def test_diez_turnos_no_explotan():
    state = initial_state()
    rng = np.random.default_rng(42)
    dm = DummyDecisionMaker(rng)
    for _ in range(10):
        state, _ = run_turn(state, dm, rng)
    assert state.turno.t == 10
    # chequeos de "no explotó"
    assert 10.0 <= state.social.pobreza_general <= 90.0
    assert -8.0 <= state.macro.crecimiento_pib <= 8.0
    assert 0.0 <= state.politico.aprobacion_presidencial <= 100.0


def test_determinista_con_misma_seed():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    s1 = initial_state()
    s2 = initial_state()
    dm1 = DummyDecisionMaker(np.random.default_rng(7))
    dm2 = DummyDecisionMaker(np.random.default_rng(7))
    for _ in range(5):
        s1, _ = run_turn(s1, dm1, rng1)
        s2, _ = run_turn(s2, dm2, rng2)
    assert s1.macro.pib_usd_mm == s2.macro.pib_usd_mm
