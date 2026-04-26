"""Tests del cliente OpenAI que NO requieren conexión a la API."""

from __future__ import annotations

import json

from guatemala_sim.president_openai import (
    GPTPresidente,
    _hardening,
    _openai_schema,
    qwen_via_dashscope,
    qwen_via_lmstudio,
    qwen_via_ollama,
)


def test_openai_schema_valido():
    s = _openai_schema()
    assert s["name"] == "DecisionTurno"
    assert s["strict"] is True
    inner = s["schema"]
    # debe contener los campos principales
    props = inner["properties"]
    for field_name in ("presupuesto", "fiscal", "exterior", "razonamiento",
                       "respuestas_shocks", "reformas", "mensaje_al_pueblo"):
        assert field_name in props
    # debe tener additionalProperties=False en el root
    assert inner["additionalProperties"] is False
    # required debe listar todos los campos
    assert set(inner["required"]) == set(props.keys())


def test_hardening_recursivo():
    """El hardening debe propagar additionalProperties=false a objetos anidados."""
    s = _openai_schema()
    stack = [s["schema"]]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert "required" in node
                assert set(node["required"]) == set(node.get("properties", {}).keys())
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def test_schema_serializa_a_json():
    """Debe poder mandarse por HTTP — o sea, ser JSON-serializable."""
    s = _openai_schema()
    # no debe romper
    json.dumps(s)


def test_modo_loose_inyecta_schema_al_system_prompt():
    """En json_object mode, el schema va en el system prompt como instrucción."""
    pres = GPTPresidente(structured_mode="json_object")
    sp = pres._system_prompt()
    assert "presupuesto" in sp
    assert "100" in sp
    # response_format debe ser json_object, no json_schema
    rf = pres._response_format()
    assert rf["type"] == "json_object"


def test_modo_strict_no_inyecta_schema_al_prompt():
    """En json_schema mode, el schema va en response_format, no en el prompt."""
    pres = GPTPresidente(structured_mode="json_schema")
    sp = pres._system_prompt()
    # el prompt no incluye el schema inline
    assert "\"additionalProperties\"" not in sp
    rf = pres._response_format()
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "DecisionTurno"


def test_factories_qwen_configuran_endpoint():
    q_ollama = qwen_via_ollama()
    assert q_ollama.base_url == "http://localhost:11434/v1"
    assert q_ollama.api_key == "ollama"
    assert q_ollama.structured_mode == "json_object"
    assert q_ollama.label is not None and "Qwen" in q_ollama.label

    q_lms = qwen_via_lmstudio()
    assert "1234" in q_lms.base_url

    q_ds = qwen_via_dashscope()
    assert "dashscope" in q_ds.base_url
    assert q_ds.structured_mode == "json_object"
