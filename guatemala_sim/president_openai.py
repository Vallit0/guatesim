"""Cliente OpenAI-compatible como presidente de Guatemala.

Espejo simétrico de `president.py` (Claude) pero sobre la API OpenAI.
Reutilizable para cualquier backend OpenAI-compatible (Ollama, LM Studio,
vLLM, together.ai, OpenRouter, DashScope-OpenAI-compat, etc.) vía los
parámetros `base_url` + `api_key`.

Dos modos de structured output:
- `structured_mode='json_schema'`: strict (solo OpenAI cloud y algunos vLLM
  recientes). Restringe el sampler al schema.
- `structured_mode='json_object'`: loose (Ollama, LM Studio, la mayoría de
  locales). Fuerza JSON válido pero el contenido del schema va en el prompt.

Corre en paralelo con `ClaudePresidente` a través del mismo protocolo
`DecisionMaker.decide(state) -> DecisionTurno`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .actions import (
    ChosenDecision,
    DecisionTurno,
    PresupuestoAnual,
    decision_from_choice,
)
from .president import (
    MENU_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    _format_menu,
    build_context,
)
from .state import GuatemalaState


def _openai_schema() -> dict[str, Any]:
    """Schema JSON compatible con `response_format=json_schema` de OpenAI.

    OpenAI impone restricciones (strict mode):
      - additionalProperties: false en TODO objeto
      - required para cada propiedad declarada
      - sin default ni minLength/maxLength-only libres

    Adaptamos el Pydantic schema a ese shape.
    """
    schema = DecisionTurno.model_json_schema()
    schema = _hardening(schema)
    return {
        "name": "DecisionTurno",
        "schema": schema,
        "strict": True,
    }


def _menu_openai_schema() -> dict[str, Any]:
    """Schema OpenAI strict para `ChosenDecision` (menu-choice mode)."""
    schema = ChosenDecision.model_json_schema()
    schema = _hardening(schema)
    return {
        "name": "ChosenDecision",
        "schema": schema,
        "strict": True,
    }


def _hardening(s: Any) -> Any:
    """Recorre el schema y fuerza el shape que OpenAI strict acepta."""
    if isinstance(s, dict):
        # inlining de refs
        if "$defs" in s:
            defs = s["$defs"]
            s = _inline_refs(s, defs)
            s.pop("$defs", None)
        new: dict[str, Any] = {}
        for k, v in s.items():
            new[k] = _hardening(v)
        # objetos: strict requiere additionalProperties=false y required=[todas las props]
        if new.get("type") == "object":
            props = new.get("properties", {})
            new["additionalProperties"] = False
            new["required"] = sorted(props.keys())
        return new
    if isinstance(s, list):
        return [_hardening(x) for x in s]
    return s


def _inline_refs(obj: Any, defs: dict) -> Any:
    """Reemplaza {"$ref": "#/$defs/X"} por la definición inline."""
    if isinstance(obj, dict):
        if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
            name = obj["$ref"].split("/")[-1]
            resolved = defs.get(name, {})
            return _inline_refs(resolved, defs)
        return {k: _inline_refs(v, defs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_inline_refs(x, defs) for x in obj]
    return obj


StructuredMode = Literal["json_schema", "json_object"]


@dataclass
class GPTPresidente:
    """Tomador de decisiones basado en la API OpenAI-compatible.

    Configuración por defecto apunta a OpenAI cloud. Para otros backends:

    - **Ollama local**: `base_url="http://localhost:11434/v1"`,
      `api_key="ollama"`, `structured_mode="json_object"`.
    - **LM Studio**: `base_url="http://localhost:1234/v1"`,
      `api_key="lm-studio"`, `structured_mode="json_object"`.
    - **vLLM**: `base_url="http://localhost:8000/v1"`,
      `structured_mode="json_schema"` (si vLLM lo soporta).
    - **Together / OpenRouter**: `base_url` correspondiente + api_key real.
    """

    model: str = "gpt-4o-mini"
    max_tokens: int = 4_000
    ultimos_eventos: list[str] = field(default_factory=list)
    territory_provider: Any = None
    base_url: str | None = None
    api_key: str | None = None
    structured_mode: StructuredMode = "json_schema"
    label: str | None = None  # para logging / identificación
    _client: Any = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # lazy import
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "openai SDK no disponible; `pip install openai` primero."
                ) from e
            kwargs: dict[str, Any] = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = OpenAI(**kwargs)

    def _system_prompt(self) -> str:
        """Extiende el system prompt con el schema si estamos en modo loose."""
        if self.structured_mode == "json_object":
            schema = DecisionTurno.model_json_schema()
            return (
                SYSTEM_PROMPT
                + "\n\nDEBES responder UN SOLO objeto JSON válido "
                "(sin texto adicional, sin markdown, sin ```) que respete "
                "este schema:\n"
                + json.dumps(schema, ensure_ascii=False)
                + "\n\nReglas críticas: presupuesto (9 partidas) debe sumar 100. "
                "`fiscal.delta_iva_pp` en [-5, 5]. `fiscal.delta_isr_pp` en [-10, 10]. "
                "`reformas` ≤ 2 ítems. Campo `razonamiento` obligatorio y sustancial."
            )
        return SYSTEM_PROMPT

    def _menu_system_prompt(self) -> str:
        """System prompt para menu-choice mode. En modo loose inyecta el schema."""
        if self.structured_mode == "json_object":
            schema = ChosenDecision.model_json_schema()
            return (
                MENU_SYSTEM_PROMPT
                + "\n\nDEBES responder UN SOLO objeto JSON válido "
                "(sin texto adicional, sin markdown, sin ```) que respete "
                "este schema:\n"
                + json.dumps(schema, ensure_ascii=False)
                + "\n\nReglas críticas: `chosen_index` ∈ [0, 4]. "
                "`fiscal.delta_iva_pp` en [-5, 5]. `fiscal.delta_isr_pp` en [-10, 10]. "
                "`reformas` ≤ 2 ítems. Campo `razonamiento` obligatorio y sustancial."
            )
        return MENU_SYSTEM_PROMPT

    def _response_format(self) -> dict[str, Any]:
        if self.structured_mode == "json_schema":
            return {"type": "json_schema", "json_schema": _openai_schema()}
        # json_object: acepta cualquier JSON válido, schema va en el prompt
        return {"type": "json_object"}

    def _menu_response_format(self) -> dict[str, Any]:
        if self.structured_mode == "json_schema":
            return {"type": "json_schema", "json_schema": _menu_openai_schema()}
        return {"type": "json_object"}

    def decide(self, state: GuatemalaState) -> DecisionTurno:
        self._ensure_client()
        ts = self.territory_provider() if callable(self.territory_provider) else None
        user_msg = build_context(state, territory_summary=ts, eventos_pasados=self.ultimos_eventos)

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_msg},
        ]

        ultimo_err: str = ""
        for intento in range(3):
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                response_format=self._response_format(),
            )
            content = resp.choices[0].message.content or ""
            try:
                decision = DecisionTurno.model_validate_json(content)
                return decision
            except Exception as e:
                ultimo_err = str(e)[:400]
                feedback = (
                    f"Tu JSON no validó: {ultimo_err}. "
                    f"Verificá: presupuesto suma 100 (±1), fiscal rangos, "
                    f"≤2 reformas. Reenviá UN SOLO objeto JSON."
                )
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": feedback},
                ]
        raise RuntimeError(
            f"{self.label or 'OpenAI-compat'} no devolvió decisión válida tras 3 intentos. "
            f"Último error: {ultimo_err}"
        )

    def choose_from_menu(
        self,
        state: GuatemalaState,
        candidates: list[PresupuestoAnual],
        candidate_names: list[str] | None = None,
    ) -> tuple[int, DecisionTurno]:
        """Modo menu-choice: presenta candidates y devuelve (idx, decisión)."""
        if not candidates:
            raise ValueError("candidates no puede ser vacío")
        self._ensure_client()
        ts = self.territory_provider() if callable(self.territory_provider) else None
        user_msg = (
            build_context(state, territory_summary=ts, eventos_pasados=self.ultimos_eventos)
            + _format_menu(candidates, names=candidate_names)
            + "\nElegí el candidato y completá la decisión."
        )

        messages = [
            {"role": "system", "content": self._menu_system_prompt()},
            {"role": "user", "content": user_msg},
        ]

        ultimo_err: str = ""
        for intento in range(3):
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                response_format=self._menu_response_format(),
            )
            content = resp.choices[0].message.content or ""
            try:
                chosen = ChosenDecision.model_validate_json(content)
                if chosen.chosen_index >= len(candidates):
                    raise ValueError(
                        f"chosen_index={chosen.chosen_index} fuera del menú "
                        f"de {len(candidates)} candidatos"
                    )
                decision = decision_from_choice(chosen, candidates[chosen.chosen_index])
                return chosen.chosen_index, decision
            except Exception as e:
                ultimo_err = str(e)[:400]
                feedback = (
                    f"Tu JSON no validó: {ultimo_err}. "
                    f"Verificá: chosen_index ∈ [0, {len(candidates) - 1}], "
                    f"fiscal rangos, ≤2 reformas. Reenviá UN SOLO objeto JSON."
                )
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": feedback},
                ]
        raise RuntimeError(
            f"{self.label or 'OpenAI-compat'} no devolvió elección válida tras "
            f"3 intentos. Último error: {ultimo_err}"
        )


# --- factory helpers ---------------------------------------------------------


def qwen_via_ollama(
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    **kwargs,
) -> GPTPresidente:
    """Convenience: Qwen via Ollama (endpoint OpenAI-compatible local)."""
    return GPTPresidente(
        model=model,
        base_url=base_url,
        api_key=api_key,
        structured_mode="json_object",
        label=f"Qwen/{model}",
        **kwargs,
    )


def qwen_via_lmstudio(
    model: str = "qwen2.5-7b-instruct",
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    **kwargs,
) -> GPTPresidente:
    """Convenience: Qwen via LM Studio (endpoint OpenAI-compatible local)."""
    return GPTPresidente(
        model=model,
        base_url=base_url,
        api_key=api_key,
        structured_mode="json_object",
        label=f"Qwen/{model}",
        **kwargs,
    )


def qwen_via_dashscope(
    model: str = "qwen-plus",
    api_key: str | None = None,
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    **kwargs,
) -> GPTPresidente:
    """Convenience: Qwen via DashScope (Alibaba Cloud, OpenAI-compatible)."""
    return GPTPresidente(
        model=model,
        base_url=base_url,
        api_key=api_key,
        structured_mode="json_object",
        label=f"Qwen/{model}",
        **kwargs,
    )
