"""Tests del wrapper resiliente."""

from __future__ import annotations

import numpy as np

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.engine import DummyDecisionMaker
from guatemala_sim.resilient import ResilientDecisionMaker


class AlwaysFails:
    def decide(self, state):
        raise RuntimeError("no puedo gobernar")


class SometimesFails:
    def __init__(self):
        self.n = 0

    def decide(self, state):
        self.n += 1
        if self.n % 2 == 1:
            raise RuntimeError("turno impar: fallo")
        return DummyDecisionMaker(np.random.default_rng(0)).decide(state)


def test_fallback_cuando_primario_falla_siempre():
    rng = np.random.default_rng(1)
    dm = ResilientDecisionMaker(
        primario=AlwaysFails(),
        fallback=DummyDecisionMaker(rng),
    )
    d = dm.decide(initial_state())
    assert d is not None
    assert dm.n_llamadas == 1
    assert dm.n_fallos == 1
    assert dm.tasa_fallo == 100.0


def test_fallback_parcial():
    rng = np.random.default_rng(1)
    dm = ResilientDecisionMaker(
        primario=SometimesFails(),
        fallback=DummyDecisionMaker(rng),
    )
    for _ in range(4):
        dm.decide(initial_state())
    assert dm.n_llamadas == 4
    assert dm.n_fallos == 2  # t=1 y t=3 fallan
    assert dm.tasa_fallo == 50.0


def test_fallos_guardan_traza():
    rng = np.random.default_rng(1)
    dm = ResilientDecisionMaker(
        primario=AlwaysFails(),
        fallback=DummyDecisionMaker(rng),
    )
    dm.decide(initial_state())
    assert len(dm.fallos) == 1
    assert "RuntimeError" in dm.fallos[0]
    assert "no puedo gobernar" in dm.fallos[0]
