"""Registry declarativo de modelos LLM para auditoría multi-vendor.

El paper actual evalúa Claude Haiku 4.5 vs GPT-4o-mini. Para sostener la
hipótesis de transferencia cultural Norte→Sur el pipeline tiene que correr
contra modelos de varias familias (Anthropic, OpenAI, Google, DeepSeek,
Meta, Alibaba) sin tocar el código del runner. Este módulo da un único
punto de configuración: cada modelo es una `ModelSpec` y el factory
`make_decision_maker(model_id)` devuelve el cliente listo.

Aprovecha que `GPTPresidente` ya es OpenAI-compatible — basta con
`base_url` + `api_key` correctos. Sólo Anthropic usa SDK propio
(`ClaudePresidente`).

Para correr offline (tests, demos): `make_decision_maker("...", require_api_key=False)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal


Provider = Literal["anthropic", "openai", "openai_compat"]
StructuredMode = Literal["json_schema", "json_object"]


@dataclass(frozen=True)
class ModelSpec:
    """Configuración declarativa de un modelo evaluable.

    `model_id` es la clave canónica usada en CLI flags y reportes.
    `model` es el string que va al provider en cada request.
    """

    model_id: str
    display_name: str
    provider: Provider
    model: str
    base_url: str | None = None
    env_key: str = ""
    structured_mode: StructuredMode = "json_schema"
    family: str = ""
    notes: str = ""


# Registro central. Para agregar un modelo, agregalo acá; el resto del
# pipeline lo descubre por `model_id`. Mantener `model_id` estable porque
# aparece en paths de runs/figures y en tablas del paper.
MODEL_SPECS: dict[str, ModelSpec] = {
    # --- Anthropic (SDK nativo, tool_use) -----------------------------------
    "claude-haiku-4-5": ModelSpec(
        model_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        env_key="ANTHROPIC_API_KEY",
        family="anthropic",
    ),
    "claude-sonnet-4-6": ModelSpec(
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        model="claude-sonnet-4-6",
        env_key="ANTHROPIC_API_KEY",
        family="anthropic",
    ),
    "claude-opus-4-7": ModelSpec(
        model_id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        provider="anthropic",
        model="claude-opus-4-7",
        env_key="ANTHROPIC_API_KEY",
        family="anthropic",
    ),

    # --- OpenAI (SDK nativo, json_schema strict) ----------------------------
    "gpt-4o-mini": ModelSpec(
        model_id="gpt-4o-mini",
        display_name="GPT-4o-mini",
        provider="openai",
        model="gpt-4o-mini",
        env_key="OPENAI_API_KEY",
        structured_mode="json_schema",
        family="openai",
    ),
    "gpt-4o": ModelSpec(
        model_id="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        model="gpt-4o",
        env_key="OPENAI_API_KEY",
        structured_mode="json_schema",
        family="openai",
    ),
    "gpt-4-1-mini": ModelSpec(
        model_id="gpt-4-1-mini",
        display_name="GPT-4.1-mini",
        provider="openai",
        model="gpt-4.1-mini",
        env_key="OPENAI_API_KEY",
        structured_mode="json_schema",
        family="openai",
    ),

    # --- Google Gemini via OpenAI-compatible shim ---------------------------
    "gemini-2-0-flash": ModelSpec(
        model_id="gemini-2-0-flash",
        display_name="Gemini 2.0 Flash",
        provider="openai_compat",
        model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_key="GEMINI_API_KEY",
        structured_mode="json_object",
        family="google",
        notes="Shim OpenAI-compat de Google Generative AI; json_object es la opción robusta cross-version.",
    ),
    "gemini-2-5-flash": ModelSpec(
        model_id="gemini-2-5-flash",
        display_name="Gemini 2.5 Flash",
        provider="openai_compat",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_key="GEMINI_API_KEY",
        structured_mode="json_object",
        family="google",
    ),

    # --- DeepSeek (OpenAI-compat nativo) ------------------------------------
    "deepseek-v3": ModelSpec(
        model_id="deepseek-v3",
        display_name="DeepSeek V3 (chat)",
        provider="openai_compat",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
        structured_mode="json_object",
        family="deepseek",
        notes="V3-chat: json_object estable; json_schema strict no garantizado.",
    ),
    "deepseek-r1": ModelSpec(
        model_id="deepseek-r1",
        display_name="DeepSeek R1 (reasoner)",
        provider="openai_compat",
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
        structured_mode="json_object",
        family="deepseek",
        notes="Reasoner: emite chain-of-thought separado; útil para test de faithfulness.",
    ),

    # --- Meta Llama via Together AI -----------------------------------------
    "llama-3-1-405b": ModelSpec(
        model_id="llama-3-1-405b",
        display_name="Llama 3.1 405B Instruct Turbo (Together)",
        provider="openai_compat",
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        structured_mode="json_object",
        family="meta",
    ),
    "llama-3-3-70b": ModelSpec(
        model_id="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct Turbo (Together)",
        provider="openai_compat",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        structured_mode="json_object",
        family="meta",
    ),

    # --- Alibaba Qwen via DashScope (OpenAI-compat) -------------------------
    "qwen-2-5-72b": ModelSpec(
        model_id="qwen-2-5-72b",
        display_name="Qwen 2.5 72B Instruct",
        provider="openai_compat",
        model="qwen2.5-72b-instruct",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        structured_mode="json_object",
        family="alibaba",
    ),
    "qwen-plus": ModelSpec(
        model_id="qwen-plus",
        display_name="Qwen Plus",
        provider="openai_compat",
        model="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        structured_mode="json_object",
        family="alibaba",
    ),
}


def get_spec(model_id: str) -> ModelSpec:
    """Resuelve `model_id` a `ModelSpec` con error útil si no existe."""
    if model_id not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS.keys()))
        raise KeyError(
            f"model_id={model_id!r} no registrado. "
            f"Disponibles: {valid}. "
            f"Para agregar uno nuevo, editá guatemala_sim/models_registry.py."
        )
    return MODEL_SPECS[model_id]


def list_specs(family: str | None = None) -> list[ModelSpec]:
    """Lista de specs registradas, opcionalmente filtradas por `family`."""
    items = list(MODEL_SPECS.values())
    if family is not None:
        items = [s for s in items if s.family == family]
    return items


def make_decision_maker(
    model_id: str,
    *,
    require_api_key: bool = True,
    **overrides: Any,
) -> Any:
    """Construye el `DecisionMaker` para `model_id`.

    Args:
        model_id: clave del registry.
        require_api_key: si True (default), valida que la env var del
            provider esté seteada antes de devolver el cliente. En tests
            offline o demos sin red, pasar False.
        **overrides: kwargs forwardeados al constructor del cliente
            (`max_tokens`, `territory_provider`, etc).

    Returns:
        Instancia con `.decide(state)` y `.choose_from_menu(state, candidates)`.
    """
    spec = get_spec(model_id)

    if require_api_key and spec.env_key:
        if not os.environ.get(spec.env_key):
            raise RuntimeError(
                f"{spec.env_key} no está en el environment para "
                f"{spec.display_name} ({spec.model_id}). "
                f"Setealo en .env o pasá require_api_key=False para tests offline."
            )

    if spec.provider == "anthropic":
        from .president import ClaudePresidente
        return ClaudePresidente(model=spec.model, **overrides)

    if spec.provider == "openai":
        from .president_openai import GPTPresidente
        return GPTPresidente(
            model=spec.model,
            structured_mode=spec.structured_mode,
            label=spec.display_name,
            **overrides,
        )

    if spec.provider == "openai_compat":
        from .president_openai import GPTPresidente
        api_key = os.environ.get(spec.env_key) if spec.env_key else None
        return GPTPresidente(
            model=spec.model,
            base_url=spec.base_url,
            api_key=api_key,
            structured_mode=spec.structured_mode,
            label=spec.display_name,
            **overrides,
        )

    raise ValueError(f"provider desconocido en {spec.model_id}: {spec.provider!r}")


def parse_models_csv(csv: str) -> list[ModelSpec]:
    """Parsea 'gpt-4o-mini,claude-haiku-4-5,gemini-2-0-flash' → list[ModelSpec].

    Cada id se resuelve via `get_spec`. Falla con KeyError útil al primer
    id desconocido — preferimos fallar temprano que arrancar un batch
    largo y descubrir el typo a la mitad.
    """
    ids = [s.strip() for s in csv.split(",") if s.strip()]
    if not ids:
        raise ValueError(f"lista de modelos vacía: {csv!r}")
    seen: set[str] = set()
    specs: list[ModelSpec] = []
    for mid in ids:
        if mid in seen:
            continue
        seen.add(mid)
        specs.append(get_spec(mid))
    return specs
