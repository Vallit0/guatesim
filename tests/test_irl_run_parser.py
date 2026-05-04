"""Tests del parser JSONL → (features, chosen) para IRL bayesiano.

Genera un JSONL con `DummyMenuDecisionMaker` (sin API), lo parsea, y
verifica los invariantes que el IRL bayesiano necesita: shapes, rangos,
reference subtraction y consistencia del menú.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Permite importar compare_llms desde el root del repo (mismo truco que
# tests/test_runners_menu_mode.py)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_llms import _correr, _nueva_mundo  # type: ignore[import-not-found]
from guatemala_sim.engine import DummyMenuDecisionMaker
from guatemala_sim.irl import (
    OUTCOME_FEATURE_NAMES,
    ParsedRun,
    RunFormatError,
    parse_menu_run,
)
from guatemala_sim.irl.candidates import REFERENCE_CANDIDATE_INDEX


def _generar_jsonl_dummy_menu(
    tmp_path: Path,
    monkeypatch,
    selected_index: int = 2,
    turnos: int = 3,
    seed: int = 7,
) -> Path:
    """Helper: corre `turnos` con DummyMenu y devuelve el JSONL producido."""
    monkeypatch.setattr("compare_llms.ROOT", tmp_path)
    rng, state, agentes, territory = _nueva_mundo(seed=seed)
    dm = DummyMenuDecisionMaker(rng=rng, selected_index=selected_index)
    return _correr(
        label="DummyMenu/parser-test",
        decision_maker=dm,
        territorio=territory,
        agentes=agentes,
        rng=rng,
        state=state,
        turnos=turnos,
        run_id=f"parsertest_dummy_{selected_index}",
        menu_mode=True,
    )


def test_parse_menu_run_devuelve_shapes_correctos(tmp_path, monkeypatch):
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, selected_index=2, turnos=3)
    parsed = parse_menu_run(log, n_samples=3)
    assert isinstance(parsed, ParsedRun)
    assert parsed.n_turns == 3
    assert parsed.n_candidates == 5
    assert parsed.n_features == 6
    assert parsed.features.shape == (3, 5, 6)
    assert parsed.chosen.shape == (3,)
    assert len(parsed.razonamientos) == 3
    assert parsed.feature_names == OUTCOME_FEATURE_NAMES


def test_chosen_coincide_con_selected_index(tmp_path, monkeypatch):
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, selected_index=4, turnos=2)
    parsed = parse_menu_run(log, n_samples=2)
    np.testing.assert_array_equal(parsed.chosen, np.array([4, 4]))


def test_features_estan_reference_subtracted(tmp_path, monkeypatch):
    """`features[:, 0, :]` debe ser exactamente cero por construcción."""
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=2)
    parsed = parse_menu_run(log, n_samples=2)
    np.testing.assert_array_equal(
        parsed.features[:, REFERENCE_CANDIDATE_INDEX, :],
        np.zeros((parsed.n_turns, parsed.n_features)),
    )


def test_chosen_dentro_de_rango(tmp_path, monkeypatch):
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=3)
    parsed = parse_menu_run(log, n_samples=2)
    assert (parsed.chosen >= 0).all()
    assert (parsed.chosen < parsed.n_candidates).all()


def test_candidate_names_contiene_los_5_arquetipos(tmp_path, monkeypatch):
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=1)
    parsed = parse_menu_run(log, n_samples=2)
    expected = {
        "status_quo_uniforme",
        "fiscal_prudente",
        "desarrollo_humano",
        "seguridad_primero",
        "equilibrado",
    }
    assert set(parsed.candidate_names) == expected
    # status_quo_uniforme debe ser el ancla (índice 0)
    assert parsed.candidate_names[REFERENCE_CANDIDATE_INDEX] == "status_quo_uniforme"


def test_feature_seed_es_determinista(tmp_path, monkeypatch):
    """Mismo JSONL + mismo feature_seed → mismas features."""
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=2)
    a = parse_menu_run(log, feature_seed=42, n_samples=3)
    b = parse_menu_run(log, feature_seed=42, n_samples=3)
    np.testing.assert_array_equal(a.features, b.features)


def test_feature_seeds_distintos_dan_features_distintas(tmp_path, monkeypatch):
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=2)
    a = parse_menu_run(log, feature_seed=1, n_samples=3)
    b = parse_menu_run(log, feature_seed=2, n_samples=3)
    # En el non-ref slice debe haber al menos un componente distinto
    diff_in_non_ref = np.abs(a.features - b.features)[:, 1:, :].sum()
    assert diff_in_non_ref > 0


def test_estados_inicial_y_final_son_validos(tmp_path, monkeypatch):
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=3)
    parsed = parse_menu_run(log, n_samples=2)
    # state_initial.turno.t < state_final.turno.t (avanza al menos 1 turno)
    assert parsed.state_initial.turno.t < parsed.state_final.turno.t


def test_jsonl_legacy_sin_menu_choice_levanta_run_format_error(tmp_path, monkeypatch):
    """Un JSONL sin records `menu_choice` debe levantar un error informativo."""
    from guatemala_sim.engine import DummyDecisionMaker

    monkeypatch.setattr("compare_llms.ROOT", tmp_path)
    rng, state, agentes, territory = _nueva_mundo(seed=7)
    dm = DummyDecisionMaker(rng)
    log = _correr(
        label="Dummy/legacy",
        decision_maker=dm,
        territorio=territory,
        agentes=agentes,
        rng=rng,
        state=state,
        turnos=2,
        run_id="parsertest_legacy",
        menu_mode=False,
    )
    with pytest.raises(RunFormatError) as excinfo:
        parse_menu_run(log, n_samples=2)
    assert "menu-mode" in str(excinfo.value)


def test_jsonl_inexistente_levanta_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_menu_run(tmp_path / "no_existe.jsonl", n_samples=2)


def test_label_se_extrae_del_nombre_del_archivo(tmp_path, monkeypatch):
    """`ParsedRun.label` heurística: segmento después del último '_'."""
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=1)
    # Renombrar el archivo a algo que parsee como `<id>_claude.jsonl`
    nuevo = log.parent / "20260101_000000_abcdef_claude.jsonl"
    log.rename(nuevo)
    parsed = parse_menu_run(nuevo, n_samples=2)
    assert parsed.label == "claude"


def test_chosen_index_fuera_de_rango_levanta(tmp_path, monkeypatch):
    """Si manualmente corrompemos el JSONL con chosen_index fuera de rango,
    el parser falla limpio."""
    log = _generar_jsonl_dummy_menu(tmp_path, monkeypatch, turnos=2)
    lines = log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["menu_choice"]["chosen_index"] = 99
    lines[0] = json.dumps(rec, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fuera de rango"):
        parse_menu_run(log, n_samples=2)
