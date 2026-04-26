"""Tests del módulo de comparativa: pipeline completo con 2 corridas dummy."""

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
from guatemala_sim.comparison import (
    CorridaEtiquetada,
    generar_comparativa,
    tabla_comparativa,
)
from guatemala_sim.engine import DummyDecisionMaker, run_turn
from guatemala_sim.logging_ import JsonlLogger
from guatemala_sim.world.territory import Territory


def _corrida(tmp_path: Path, seed: int, label: str, turnos: int = 4) -> Path:
    rng = np.random.default_rng(seed)
    state = initial_state()
    territory = Territory.load_default()
    agentes = AgentesModel(
        [PartidoOficialista, CongresoOposicion, CACIF, ProtestaSocial], seed=seed
    )
    dm = DummyDecisionMaker(rng)
    path = tmp_path / f"{label}.jsonl"
    with JsonlLogger(path) as lg:
        for _ in range(turnos):
            state, _ = run_turn(
                state, dm, rng, hooks=[lg.log],
                agentes=agentes, territorio=territory,
            )
    return path


def test_tabla_comparativa(tmp_path: Path):
    p1 = _corrida(tmp_path, seed=1, label="a")
    p2 = _corrida(tmp_path, seed=2, label="b")
    c1 = CorridaEtiquetada.from_path("A", p1)
    c2 = CorridaEtiquetada.from_path("B", p2)
    df = tabla_comparativa([c1, c2])
    assert list(df.index) == ["A", "B"]
    for col in ("PIB_fin", "pobreza_fin", "coherencia_temporal",
                "bienestar_fin", "reformas_totales"):
        assert col in df.columns


def test_generar_comparativa_produce_archivos(tmp_path: Path):
    p1 = _corrida(tmp_path, seed=1, label="a")
    p2 = _corrida(tmp_path, seed=2, label="b")
    c1 = CorridaEtiquetada.from_path("seed=1", p1)
    c2 = CorridaEtiquetada.from_path("seed=2", p2)
    out_dir = tmp_path / "compare_out"
    outs = generar_comparativa([c1, c2], out_dir)
    assert len(outs) == 5  # 4 PNG + 1 MD
    for o in outs:
        assert o.exists()
        assert o.stat().st_size > 0
    # el reporte debe mencionar las 2 corridas
    md = (out_dir / "reporte.md").read_text(encoding="utf-8")
    assert "seed=1" in md and "seed=2" in md
