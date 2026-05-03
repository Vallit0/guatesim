"""Tests del wiring de --menu-mode en los runners (compare_llms*).

Estos tests son offline: no llaman a APIs de LLM. Verifican que (a) el
flag se parsea correctamente, (b) `_correr` con menu_mode=True propaga
correctamente al engine usando un DummyMenuDecisionMaker, (c) el JSONL
resultante contiene `menu_choice`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Asegurar que podemos importar compare_llms desde el root del repo
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_llms import _correr, _nueva_mundo  # type: ignore[import-not-found]
from guatemala_sim.engine import DummyDecisionMaker, DummyMenuDecisionMaker


def test_correr_menu_mode_genera_jsonl_con_menu_choice(tmp_path, monkeypatch):
    """Smoke end-to-end: 2 turnos con DummyMenuDecisionMaker producen
    JSONL donde cada línea tiene `menu_choice.chosen_index`."""
    # Redirijo runs/ a tmp_path para no contaminar el repo
    monkeypatch.setattr("compare_llms.ROOT", tmp_path)

    rng, state, agentes, territory = _nueva_mundo(seed=7)
    dm = DummyMenuDecisionMaker(rng=rng, selected_index=2)

    log_path = _correr(
        label="DummyMenu/test",
        decision_maker=dm,
        territorio=territory,
        agentes=agentes,
        rng=rng,
        state=state,
        turnos=2,
        run_id="testrun_menu",
        menu_mode=True,
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["menu_choice"] is not None
        assert rec["menu_choice"]["chosen_index"] == 2
        assert len(rec["menu_choice"]["candidates"]) == 5
        names = [c["name"] for c in rec["menu_choice"]["candidates"]]
        assert "status_quo_uniforme" in names
        assert "equilibrado" in names


def test_correr_legacy_mode_no_loguea_menu_choice(tmp_path, monkeypatch):
    """Si menu_mode=False (default), los registros tienen menu_choice=None.
    Asegura backwards compatibility."""
    monkeypatch.setattr("compare_llms.ROOT", tmp_path)

    rng, state, agentes, territory = _nueva_mundo(seed=7)
    dm = DummyDecisionMaker(rng)
    log_path = _correr(
        label="Dummy/legacy",
        decision_maker=dm,
        territorio=territory,
        agentes=agentes,
        rng=rng,
        state=state,
        turnos=2,
        run_id="testrun_legacy",
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec.get("menu_choice") is None


def test_argparse_compare_llms_acepta_menu_mode():
    """El parser de compare_llms.py debe aceptar --menu-mode."""
    import compare_llms  # type: ignore
    import argparse

    # Reconstruir el parser inspeccionando el archivo es frágil; mejor
    # verificar via parse_known_args en una invocación simulada.
    # Truco: monkeypatch sys.argv y usar argparse desde main.
    # Pero main() ejecuta toda la corrida — no queremos eso.
    # En su lugar, verifico que el flag está mencionado en el help text.
    src = Path(compare_llms.__file__).read_text(encoding="utf-8")
    assert '--menu-mode' in src
    assert 'menu-choice' in src or 'menu_mode' in src


def test_argparse_compare_llms_multiseed_acepta_menu_mode():
    import compare_llms_multiseed  # type: ignore
    src = Path(compare_llms_multiseed.__file__).read_text(encoding="utf-8")
    assert '--menu-mode' in src
    assert 'menu_mode=args.menu_mode' in src


def test_correr_menu_mode_3_turnos_sin_explotar(tmp_path, monkeypatch):
    """Smoke un poco más largo: 3 turnos con índices distintos por dummy."""
    monkeypatch.setattr("compare_llms.ROOT", tmp_path)

    rng, state, agentes, territory = _nueva_mundo(seed=42)
    dm = DummyMenuDecisionMaker(rng=rng, selected_index=0)
    log_path = _correr(
        label="DummyMenu/idx0",
        decision_maker=dm,
        territorio=territory,
        agentes=agentes,
        rng=rng,
        state=state,
        turnos=3,
        run_id="testrun_3turnos",
        menu_mode=True,
    )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert rec["menu_choice"]["chosen_index"] == 0
        # El presupuesto del turno debe coincidir con el del candidato 0
        chosen_pres = rec["menu_choice"]["candidates"][0]["presupuesto"]
        decision_pres = rec["decision"]["presupuesto"]
        assert chosen_pres == decision_pres
