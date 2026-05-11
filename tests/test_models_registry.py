"""Tests offline del registry multi-modelo.

Verifican (a) cobertura de las 6 familias requeridas para test cross-vendor
del paper, (b) resolución y errores útiles, (c) construcción de clientes
sin tocar red, (d) parser CSV de la flag --models.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from guatemala_sim.models_registry import (
    MODEL_SPECS,
    ModelSpec,
    get_spec,
    list_specs,
    make_decision_maker,
    parse_models_csv,
)


# --- cobertura del registry --------------------------------------------------


def test_registry_cubre_seis_familias_para_paper():
    """El paper afirma cross-vendor evaluation; el registry debe cubrir
    las familias mínimas para sostener el claim Norte→Sur."""
    families = {s.family for s in MODEL_SPECS.values()}
    requeridas = {"anthropic", "openai", "google", "deepseek", "meta", "alibaba"}
    faltan = requeridas - families
    assert not faltan, (
        f"Faltan familias para sostener el claim cross-vendor: {sorted(faltan)}. "
        f"El paper depende de testear modelos del Norte (anthropic/openai/google/meta) "
        f"y del 'no-Norte' (deepseek/alibaba)."
    )


def test_specs_tienen_campos_minimos():
    for mid, spec in MODEL_SPECS.items():
        assert spec.model_id == mid, f"model_id key/value mismatch en {mid}"
        assert spec.display_name, f"{mid} sin display_name"
        assert spec.model, f"{mid} sin model string"
        assert spec.provider in ("anthropic", "openai", "openai_compat")
        assert spec.family, f"{mid} sin family"
        if spec.provider == "openai_compat":
            assert spec.base_url, f"{mid} es openai_compat pero no tiene base_url"
        if spec.env_key:
            assert spec.env_key.endswith("_API_KEY"), (
                f"{mid}: env_key={spec.env_key!r} no termina en _API_KEY (convención)"
            )


def test_get_spec_falla_util_en_id_desconocido():
    with pytest.raises(KeyError) as exc:
        get_spec("modelo-inexistente-42")
    msg = str(exc.value)
    assert "no registrado" in msg
    assert "Disponibles" in msg
    # El usuario debe ver al menos un id válido en el mensaje
    assert "claude-haiku-4-5" in msg or "gpt-4o-mini" in msg


def test_list_specs_filtra_por_familia():
    google = list_specs(family="google")
    assert google
    for s in google:
        assert s.family == "google"
    # Sin filtro devuelve todo
    assert len(list_specs()) == len(MODEL_SPECS)


# --- factory --------------------------------------------------------------


def test_make_decision_maker_anthropic_sin_red():
    """Anthropic: con env key seteada, construye ClaudePresidente sin tocar red."""
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        dm = make_decision_maker("claude-haiku-4-5")
    from guatemala_sim.president import ClaudePresidente
    assert isinstance(dm, ClaudePresidente)
    assert dm.model == "claude-haiku-4-5-20251001"
    # Cliente lazy: no se inicializó todavía
    assert dm._client is None


def test_make_decision_maker_openai_sin_red():
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
        dm = make_decision_maker("gpt-4o-mini")
    from guatemala_sim.president_openai import GPTPresidente
    assert isinstance(dm, GPTPresidente)
    assert dm.model == "gpt-4o-mini"
    assert dm.structured_mode == "json_schema"
    assert dm.base_url is None  # OpenAI cloud usa default


def test_make_decision_maker_gemini_compat_pasa_base_url():
    """Gemini debe entrar via base_url custom + GEMINI_API_KEY."""
    with mock.patch.dict(
        os.environ, {"GEMINI_API_KEY": "test-gemini-key"}, clear=False
    ):
        dm = make_decision_maker("gemini-2-0-flash")
    from guatemala_sim.president_openai import GPTPresidente
    assert isinstance(dm, GPTPresidente)
    assert dm.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert dm.api_key == "test-gemini-key"
    assert dm.structured_mode == "json_object"
    assert dm.label == "Gemini 2.0 Flash"


def test_make_decision_maker_deepseek_pasa_base_url():
    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-key"}, clear=False):
        dm = make_decision_maker("deepseek-v3")
    assert dm.base_url == "https://api.deepseek.com"
    assert dm.api_key == "ds-key"
    assert dm.model == "deepseek-chat"


def test_make_decision_maker_llama_via_together():
    with mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "tg-key"}, clear=False):
        dm = make_decision_maker("llama-3-1-405b")
    assert dm.base_url == "https://api.together.xyz/v1"
    assert dm.model == "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"


def test_make_decision_maker_falla_si_falta_api_key():
    """Sin la env var, debe fallar antes de construir el cliente."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError) as exc:
            make_decision_maker("gemini-2-0-flash")
        assert "GEMINI_API_KEY" in str(exc.value)
        assert "require_api_key=False" in str(exc.value)


def test_make_decision_maker_offline_no_valida_api_key():
    """En modo offline (tests), permitir construcción sin env keys."""
    with mock.patch.dict(os.environ, {}, clear=True):
        # No debe lanzar
        dm = make_decision_maker("deepseek-v3", require_api_key=False)
    # api_key será None pero el objeto se construye
    assert dm.model == "deepseek-chat"


def test_make_decision_maker_overrides_pasan_al_cliente():
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=False):
        dm = make_decision_maker("gpt-4o-mini", max_tokens=8000)
    assert dm.max_tokens == 8000


# --- parser CSV ------------------------------------------------------------


def test_parse_models_csv_simple():
    specs = parse_models_csv("gpt-4o-mini,claude-haiku-4-5")
    assert len(specs) == 2
    assert [s.model_id for s in specs] == ["gpt-4o-mini", "claude-haiku-4-5"]


def test_parse_models_csv_dedup_preserva_orden():
    specs = parse_models_csv("gpt-4o-mini, claude-haiku-4-5 , gpt-4o-mini")
    assert [s.model_id for s in specs] == ["gpt-4o-mini", "claude-haiku-4-5"]


def test_parse_models_csv_falla_temprano_en_id_invalido():
    """Preferible: fallar antes de un batch largo si hay un typo."""
    with pytest.raises(KeyError) as exc:
        parse_models_csv("gpt-4o-mini,gpt-typo,claude-haiku-4-5")
    assert "gpt-typo" in str(exc.value)


def test_parse_models_csv_vacia():
    with pytest.raises(ValueError):
        parse_models_csv("")
    with pytest.raises(ValueError):
        parse_models_csv(",,, ,")
