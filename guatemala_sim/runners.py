"""Runners reutilizables para construir mundos seed-emparejados y correr
turnos contra un decisor (LLM o baseline), logueando a JSONL.

Antes vivían en `compare_llms.py` como funciones `_` privadas pero los
scripts de auditoría (`irl_audit_real_run.py`, multiseed, sensibilidad) y
los tests las importaban cruzando módulos. Al extraerlas al package se
elimina el grafo de imports cruzados entre entry-point scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .agents import (
    AgentesModel,
    CACIF,
    CongresoOposicion,
    PartidoOficialista,
    ProtestaSocial,
)
from .bootstrap import initial_state
from .engine import run_turn
from .logging_ import JsonlLogger, print_turn_resumen
from .world.territory import Territory


def nueva_mundo(seed: int):
    """Devuelve `(rng, state, agentes, territory)` con el mismo seed.

    Reusa `initial_state()` (snapshot calibrado) y los 4 agentes
    canónicos. Llamar con el mismo seed garantiza shocks idénticos entre
    corridas de distintos decisores.
    """
    rng = np.random.default_rng(seed)
    state = initial_state()
    territory = Territory.load_default()
    agentes = AgentesModel(
        [PartidoOficialista, CongresoOposicion, CACIF, ProtestaSocial], seed=seed
    )
    return rng, state, agentes, territory


def correr(
    label,
    decision_maker,
    territorio,
    agentes,
    rng,
    state,
    turnos,
    run_id,
    *,
    runs_dir: Path | None = None,
    menu_mode: bool = False,
    menu_candidates_provider=None,
):
    """Corre `turnos` turnos de un decisor sobre el mismo mundo y loguea a JSONL.

    Args:
        runs_dir: directorio donde escribir `<run_id>.jsonl`. Por defecto
            `Path.cwd() / "runs"` para que invocaciones desde la raíz del
            repo escriban en el `runs/` esperado.
        menu_mode: si True, usa el modo menu-choice (`run_turn(menu_mode=True)`).
            El decisor debe implementar `choose_from_menu(state, candidates)`.
        menu_candidates_provider: callable opcional que devuelve la lista de
            `Candidate` cuando `menu_mode=True`.
    """
    if hasattr(decision_maker, "territory_provider"):
        decision_maker.territory_provider = lambda: territorio.summary().as_dict()
    runs_dir = runs_dir if runs_dir is not None else Path.cwd() / "runs"
    log_path = runs_dir / f"{run_id}.jsonl"
    print(
        f"\n=== corrida: {label}  run_id={run_id}"
        f"{'  [menu-mode]' if menu_mode else ''} ==="
    )
    with JsonlLogger(log_path) as lg:
        def hook(record):
            lg.log(record)
            print_turn_resumen(record)
            extra = getattr(record, "extra", {}) or {}
            if hasattr(decision_maker, "ultimos_eventos"):
                decision_maker.ultimos_eventos = extra.get("eventos_agentes", [])

        for _ in range(turnos):
            state, _rec = run_turn(
                state, decision_maker, rng,
                hooks=[hook], agentes=agentes, territorio=territorio,
                menu_mode=menu_mode,
                menu_candidates_provider=menu_candidates_provider,
            )
    return log_path
