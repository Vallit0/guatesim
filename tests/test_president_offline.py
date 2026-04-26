"""Tests del cliente de Claude que NO requieren conexión a la API.

Sólo verifican el schema de la tool y la construcción de contexto.
"""

from __future__ import annotations

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.president import _decision_tool_schema, build_context


def test_tool_schema_tiene_presupuesto():
    t = _decision_tool_schema()
    assert t["name"] == "tomar_decision"
    assert "input_schema" in t
    # el schema debe referir al presupuesto
    schema_str = str(t["input_schema"])
    assert "presupuesto" in schema_str
    assert "razonamiento" in schema_str


def test_build_context_menciona_shocks():
    s = initial_state()
    s.shocks_activos = ["sequía severa en corredor seco"]
    ctx = build_context(s, territory_summary=None)
    assert "Shocks" in ctx
    assert "sequía" in ctx


def test_build_context_con_territorio():
    s = initial_state()
    ts = {
        "n_deptos_en_crisis": 3,
        "regiones_criticas": ["Oriente", "Occidente"],
        "pobreza_media_ponderada": 62.5,
        "pobreza_p90": 83.0,
        "sequia_media": 0.3,
        "homicidios_p90": 32.0,
        "deptos_top_pobreza": ["Alta Verapaz", "Sololá", "Quiché"],
        "deptos_top_homicidios": ["Escuintla", "Izabal", "Zacapa"],
        "deptos_top_sequia": ["Chiquimula", "Zacapa", "El Progreso"],
    }
    ctx = build_context(s, territory_summary=ts)
    assert "Alta Verapaz" in ctx
    assert "Deptos en crisis" in ctx
