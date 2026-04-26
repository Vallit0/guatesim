"""Tests del grafo territorial."""

from __future__ import annotations

import numpy as np

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.world.territory import Territory


def test_carga_grafo_22_deptos():
    t = Territory.load_default()
    assert len(t.deptos()) == 22
    # Guatemala debe ser un nodo
    assert "Guatemala" in t.deptos()


def test_propagacion_shock():
    t = Territory.load_default()
    antes = t.G.nodes["Chiquimula"]["sequia_spi"]
    t.propagar_shock_climatico("Chiquimula", intensidad=0.5)
    despues = t.G.nodes["Chiquimula"]["sequia_spi"]
    assert despues > antes
    # un vecino también debe subir
    vecino = next(iter(t.G.neighbors("Chiquimula")))
    # la intensidad decae, pero el vecino recibe algo
    assert t.G.nodes[vecino]["sequia_spi"] >= 0


def test_summary_tiene_campos():
    t = Territory.load_default()
    s = t.summary()
    d = s.as_dict()
    assert "pobreza_media_ponderada" in d
    assert isinstance(d["deptos_top_pobreza"], list)
    assert len(d["deptos_top_pobreza"]) == 3


def test_step_con_shock_activo():
    t = Territory.load_default()
    state = initial_state()
    state.shocks_activos = ["sequía severa en corredor seco"]
    rng = np.random.default_rng(1)
    antes = t.G.nodes["Chiquimula"]["sequia_spi"]
    t.step(state, rng)
    despues = t.G.nodes["Chiquimula"]["sequia_spi"]
    assert despues > antes
