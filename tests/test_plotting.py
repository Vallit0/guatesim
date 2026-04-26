"""Tests de plotting: corren el pipeline completo y verifican que los archivos existen."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from guatemala_sim.agents import (
    AgentesModel,
    CACIF,
    CongresoOposicion,
    PartidoOficialista,
    ProtestaSocial,
)
from guatemala_sim.bootstrap import initial_state
from guatemala_sim.engine import DummyDecisionMaker, run_turn
from guatemala_sim.logging_ import JsonlLogger, read_run
from guatemala_sim.plotting import generar_todo
from guatemala_sim.world.territory import Territory


def test_pipeline_completo(tmp_path: Path):
    rng = np.random.default_rng(0)
    state = initial_state()
    territory = Territory.load_default()
    agentes = AgentesModel(
        [PartidoOficialista, CongresoOposicion, CACIF, ProtestaSocial], seed=0
    )
    dm = DummyDecisionMaker(rng)

    log_path = tmp_path / "corrida.jsonl"
    with JsonlLogger(log_path) as lg:
        for _ in range(4):
            state, _ = run_turn(
                state, dm, rng, hooks=[lg.log],
                agentes=agentes, territorio=territory,
            )

    records = read_run(log_path)
    assert len(records) == 4
    assert "indicadores" in records[0]
    assert "state_after" in records[0]

    fig_dir = tmp_path / "figs"
    outs = generar_todo(records, fig_dir)
    assert len(outs) == 8
    for o in outs:
        assert o.exists()
        assert o.stat().st_size > 0
