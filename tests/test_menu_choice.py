"""Tests del modo menu-choice: schemas, dummy decisor, integración engine."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from guatemala_sim.actions import (
    ChosenDecision,
    DecisionTurno,
    Fiscal,
    PoliticaExterior,
    PresupuestoAnual,
    decision_from_choice,
)
from guatemala_sim.bootstrap import initial_state
from guatemala_sim.engine import DummyMenuDecisionMaker, TurnRecord, run_turn
from guatemala_sim.irl.candidates import generate_candidate_menu
from guatemala_sim.president import (
    MENU_SYSTEM_PROMPT,
    _format_menu,
    _menu_tool_schema,
)
from guatemala_sim.president_openai import _menu_openai_schema


# --- ChosenDecision schema ---------------------------------------------------


def _valid_chosen(**overrides) -> ChosenDecision:
    base = dict(
        razonamiento="elijo el equilibrado para mantener flexibilidad",
        chosen_index=2,
        fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0),
        exterior=PoliticaExterior(alineamiento_priorizado="multilateral"),
        respuestas_shocks=[],
        reformas=[],
        mensaje_al_pueblo="seguimos trabajando por la estabilidad",
    )
    base.update(overrides)
    return ChosenDecision(**base)


def test_chosen_decision_valida_con_indice_legal():
    c = _valid_chosen(chosen_index=0)
    assert c.chosen_index == 0


def test_chosen_decision_acepta_indice_4():
    c = _valid_chosen(chosen_index=4)
    assert c.chosen_index == 4


def test_chosen_decision_rechaza_indice_negativo():
    with pytest.raises(ValidationError):
        _valid_chosen(chosen_index=-1)


def test_chosen_decision_rechaza_indice_5_o_mas():
    with pytest.raises(ValidationError):
        _valid_chosen(chosen_index=5)


def test_chosen_decision_rechaza_razonamiento_vacio():
    with pytest.raises(ValidationError):
        _valid_chosen(razonamiento="")


def test_chosen_decision_rechaza_extra_fields():
    with pytest.raises(ValidationError):
        ChosenDecision(
            razonamiento="x",
            chosen_index=0,
            fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0),
            exterior=PoliticaExterior(alineamiento_priorizado="neutral"),
            respuestas_shocks=[],
            reformas=[],
            mensaje_al_pueblo="x",
            extra_field="should_fail",  # type: ignore[arg-type]
        )


def test_decision_from_choice_construye_decision_turno_valida():
    chosen = _valid_chosen(chosen_index=2)
    presup = generate_candidate_menu()[2].presupuesto
    dt = decision_from_choice(chosen, presup)
    assert isinstance(dt, DecisionTurno)
    assert dt.presupuesto == presup
    assert dt.razonamiento == chosen.razonamiento
    assert dt.fiscal == chosen.fiscal
    assert dt.exterior == chosen.exterior


def test_decision_from_choice_no_muta_chosen_listas():
    chosen = _valid_chosen()
    snapshot = chosen.model_dump_json()
    presup = generate_candidate_menu()[0].presupuesto
    dt = decision_from_choice(chosen, presup)
    # Modificar la decisión final no debe afectar la chosen
    dt.respuestas_shocks.clear()
    assert chosen.model_dump_json() == snapshot


# --- _menu_tool_schema (Anthropic) -------------------------------------------


def test_menu_tool_schema_anthropic_tiene_chosen_index():
    t = _menu_tool_schema()
    assert t["name"] == "elegir_y_decidir"
    assert "input_schema" in t
    schema = t["input_schema"]
    assert "chosen_index" in schema["properties"]
    cidx = schema["properties"]["chosen_index"]
    # Pydantic genera ge/le como minimum/maximum
    assert cidx.get("minimum") == 0
    assert cidx.get("maximum") == 4


def test_menu_tool_schema_anthropic_no_tiene_presupuesto_libre():
    """En menu mode no se compone presupuesto — confirmar."""
    t = _menu_tool_schema()
    assert "presupuesto" not in t["input_schema"]["properties"]


def test_menu_system_prompt_anthropic_pide_elegir():
    assert "menú" in MENU_SYSTEM_PROMPT or "candidatos" in MENU_SYSTEM_PROMPT
    assert "elegir_y_decidir" in MENU_SYSTEM_PROMPT


# --- _menu_openai_schema (OpenAI strict) -------------------------------------


def test_menu_openai_schema_es_strict_compliant():
    s = _menu_openai_schema()
    assert s["name"] == "ChosenDecision"
    assert s["strict"] is True
    inner = s["schema"]
    # strict-mode: additionalProperties=false en TODO objeto
    _check_strict_recursive(inner)


def _check_strict_recursive(obj):
    """Helper: todo dict con type=object debe tener additionalProperties=false."""
    if isinstance(obj, dict):
        if obj.get("type") == "object":
            assert obj.get("additionalProperties") is False, (
                f"objeto sin additionalProperties=false: {list(obj.keys())}"
            )
            assert "required" in obj
            assert set(obj["required"]) == set(obj.get("properties", {}).keys())
        for v in obj.values():
            _check_strict_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            _check_strict_recursive(item)


def test_menu_openai_schema_chosen_index_propiedad_requerida():
    s = _menu_openai_schema()
    inner = s["schema"]
    assert "chosen_index" in inner["properties"]
    assert "chosen_index" in inner["required"]


def test_menu_openai_schema_es_serializable_a_json():
    """Un schema con tipos no-JSON o referencias colgantes rompería."""
    s = _menu_openai_schema()
    blob = json.dumps(s)
    assert "ChosenDecision" in blob
    assert "additionalProperties" in blob


# --- _format_menu ------------------------------------------------------------


def test_format_menu_usa_nombres_provistos():
    menu = generate_candidate_menu()
    presupuestos = [c.presupuesto for c in menu]
    nombres = [c.name for c in menu]
    s = _format_menu(presupuestos, names=nombres)
    for n in nombres:
        assert n in s
    # Y los porcentajes principales aparecen
    assert "salud" in s
    assert "servicio deuda" in s


def test_format_menu_default_names_usa_indices():
    menu = generate_candidate_menu()
    s = _format_menu([c.presupuesto for c in menu])
    for i in range(5):
        assert f"Candidato {i}" in s


def test_format_menu_rechaza_names_de_largo_distinto():
    menu = generate_candidate_menu()
    with pytest.raises(ValueError):
        _format_menu([c.presupuesto for c in menu], names=["a", "b"])


# --- DummyMenuDecisionMaker --------------------------------------------------


def test_dummy_menu_devuelve_tupla_correcta():
    state = initial_state()
    candidates = [c.presupuesto for c in generate_candidate_menu()]
    dm = DummyMenuDecisionMaker(selected_index=2)
    idx, decision = dm.choose_from_menu(state, candidates)
    assert idx == 2
    assert isinstance(decision, DecisionTurno)
    assert decision.presupuesto == candidates[2]


def test_dummy_menu_selected_index_default_es_4():
    state = initial_state()
    candidates = [c.presupuesto for c in generate_candidate_menu()]
    dm = DummyMenuDecisionMaker()
    idx, _ = dm.choose_from_menu(state, candidates)
    assert idx == 4  # equilibrado


def test_dummy_menu_indices_se_envuelven_modulo():
    state = initial_state()
    candidates = [c.presupuesto for c in generate_candidate_menu()]
    dm = DummyMenuDecisionMaker(selected_index=12)  # 12 % 5 == 2
    idx, decision = dm.choose_from_menu(state, candidates)
    assert idx == 2
    assert decision.presupuesto == candidates[2]


def test_dummy_menu_rechaza_candidates_vacios():
    state = initial_state()
    dm = DummyMenuDecisionMaker()
    with pytest.raises(ValueError):
        dm.choose_from_menu(state, [])


def test_dummy_menu_decision_resultante_es_valida():
    """El DecisionTurno devuelto debe pasar validación Pydantic."""
    state = initial_state()
    candidates = [c.presupuesto for c in generate_candidate_menu()]
    dm = DummyMenuDecisionMaker(selected_index=0)
    _, decision = dm.choose_from_menu(state, candidates)
    # round-trip JSON valida
    rebuilt = DecisionTurno.model_validate_json(decision.model_dump_json())
    assert rebuilt == decision


# --- engine.run_turn con menu_mode ------------------------------------------


def test_run_turn_menu_mode_loguea_chosen_index():
    state = initial_state()
    rng = np.random.default_rng(1)
    dm = DummyMenuDecisionMaker(selected_index=3)
    new_state, rec = run_turn(state, dm, rng, menu_mode=True)
    assert rec.menu_choice is not None
    assert rec.menu_choice["chosen_index"] == 3
    assert len(rec.menu_choice["candidates"]) == 5
    # cada candidato tiene name + presupuesto
    for c in rec.menu_choice["candidates"]:
        assert "name" in c
        assert "presupuesto" in c
        assert isinstance(c["presupuesto"], dict)
    # backwards compat: el state avanzó
    assert new_state.turno.t == state.turno.t + 1


def test_run_turn_legacy_mode_no_loguea_menu_choice():
    """menu_mode=False (default) debe dejar menu_choice=None."""
    from guatemala_sim.engine import DummyDecisionMaker
    state = initial_state()
    rng = np.random.default_rng(1)
    dm = DummyDecisionMaker(rng)
    _, rec = run_turn(state, dm, rng)
    assert rec.menu_choice is None


def test_run_turn_menu_mode_5_turnos_no_explota():
    state = initial_state()
    rng = np.random.default_rng(42)
    dm = DummyMenuDecisionMaker(selected_index=2)
    for _ in range(5):
        state, rec = run_turn(state, dm, rng, menu_mode=True)
        assert rec.menu_choice["chosen_index"] == 2
    assert state.turno.t == 5


def test_run_turn_menu_mode_acepta_provider_custom():
    """Si paso menu_candidates_provider, el engine usa ESOS candidatos."""
    state = initial_state()
    rng = np.random.default_rng(0)
    dm = DummyMenuDecisionMaker(selected_index=1)

    # Provider que devuelve los 5 candidatos canónicos pero re-etiquetados
    def custom_provider():
        from guatemala_sim.irl.candidates import Candidate
        original = generate_candidate_menu()
        return [Candidate(name=f"custom_{c.name}", presupuesto=c.presupuesto) for c in original]

    _, rec = run_turn(state, dm, rng, menu_mode=True, menu_candidates_provider=custom_provider)
    nombres = [c["name"] for c in rec.menu_choice["candidates"]]
    assert all(n.startswith("custom_") for n in nombres)


def test_run_turn_menu_choice_record_es_json_serializable():
    """El menu_choice del record debe serializar — necesario para JSONL."""
    state = initial_state()
    rng = np.random.default_rng(7)
    dm = DummyMenuDecisionMaker(selected_index=0)
    _, rec = run_turn(state, dm, rng, menu_mode=True)
    blob = json.dumps(rec.menu_choice)
    assert "chosen_index" in blob
    assert "candidates" in blob
