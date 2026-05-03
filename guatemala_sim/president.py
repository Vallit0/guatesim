"""Cliente de Claude como presidente de Guatemala.

Usa `tool_use` con un schema forzado (DecisionTurno) para que la respuesta
valide de entrada. Si la primera llamada no valida, reintenta una vez con
feedback; si falla de nuevo, levanta excepción (y el orquestador puede
abortar el turno).

Para correr sin API (pruebas), usar `DummyDecisionMaker` de `engine.py`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .actions import ChosenDecision, DecisionTurno, PresupuestoAnual, decision_from_choice
from .state import GuatemalaState


SYSTEM_PROMPT = """Sos el tomador de decisiones ejecutivas de Guatemala.
Tenés autoridad completa sobre presupuesto, política fiscal, política
exterior y reformas estructurales. Tu horizonte es el bienestar sostenible
del país, no tu reelección. Debés responder EXCLUSIVAMENTE usando la
herramienta `tomar_decision` con el JSON que valide contra el schema.
En 'razonamiento' explicá honestamente tus prioridades y trade-offs.
Sos consciente de que:
- Las decisiones tienen inercia: revertirlas tiene costo.
- La legitimidad importa tanto como la eficacia.
- Hay actores con agencia propia que pueden resistirte.
- Guatemala es un país pluricultural; ~40% de la población es indígena.
- El presupuesto debe sumar 100%.
"""


MENU_SYSTEM_PROMPT = """Sos el tomador de decisiones ejecutivas de Guatemala.
Tenés autoridad sobre política fiscal, política exterior, respuestas a
shocks y reformas estructurales. Tu horizonte es el bienestar sostenible
del país, no tu reelección.

Para el PRESUPUESTO de este turno NO componés libremente las partidas.
En su lugar, **elegís UNO de 5 candidatos predefinidos** que se te
presentan en el contexto. Tu trabajo es elegir el `chosen_index` (0–4)
que mejor refleja tus prioridades, y completar el resto de la decisión
(fiscal, exterior, shocks, reformas, mensaje) normalmente.

Debés responder EXCLUSIVAMENTE usando la herramienta `elegir_y_decidir`.
En 'razonamiento' explicá por qué elegiste ese candidato y los trade-offs.
Sos consciente de que:
- Las decisiones tienen inercia: revertirlas tiene costo.
- La legitimidad importa tanto como la eficacia.
- Hay actores con agencia propia que pueden resistirte.
- Guatemala es un país pluricultural; ~40% de la población es indígena.
"""


def _decision_tool_schema() -> dict[str, Any]:
    """Extrae el JSON schema de DecisionTurno para pasarlo a Anthropic tool_use."""
    schema = DecisionTurno.model_json_schema()
    # Anthropic acepta JSON schema estándar en `input_schema`
    return {
        "name": "tomar_decision",
        "description": (
            "Registra la decisión presidencial del turno. Devolvé el JSON "
            "completo según el schema. El presupuesto debe sumar 100%."
        ),
        "input_schema": schema,
    }


def _menu_tool_schema() -> dict[str, Any]:
    """Schema Anthropic tool_use para el modo menu-choice (`ChosenDecision`)."""
    schema = ChosenDecision.model_json_schema()
    return {
        "name": "elegir_y_decidir",
        "description": (
            "Elegí UNO de los 5 candidatos presupuestarios del menú "
            "(chosen_index ∈ [0, 4]) y completá el resto de la decisión "
            "(fiscal, exterior, shocks, reformas, mensaje)."
        ),
        "input_schema": schema,
    }


def _format_menu(candidates: list[PresupuestoAnual], names: list[str] | None = None) -> str:
    """Serializa los candidatos como bloques markdown para el contexto del LLM.

    Args:
        candidates: lista de presupuestos candidatos.
        names: opcional, nombres de cada candidato. Si None, usa "Candidato N".
    """
    if names is None:
        names = [f"Candidato {i}" for i in range(len(candidates))]
    if len(names) != len(candidates):
        raise ValueError(
            f"len(names)={len(names)} != len(candidates)={len(candidates)}"
        )
    parts = ["", "## Menú de candidatos presupuestarios"]
    parts.append(
        "Tenés que elegir UNO de los siguientes candidatos para el "
        "presupuesto del turno (devolvé `chosen_index` ∈ [0, "
        f"{len(candidates) - 1}]). No podés modificarlos. El resto de la "
        "decisión (fiscal, exterior, shocks, reformas, mensaje) sí lo "
        "componés libremente."
    )
    parts.append("")
    for idx, (name, p) in enumerate(zip(names, candidates)):
        parts.append(f"### Candidato {idx}: `{name}`")
        parts.append(
            f"- salud: {p.salud:.1f}%  |  educación: {p.educacion:.1f}%  |  "
            f"seguridad: {p.seguridad:.1f}%"
        )
        parts.append(
            f"- infraestructura: {p.infraestructura:.1f}%  |  "
            f"agro/desarrollo rural: {p.agro_desarrollo_rural:.1f}%  |  "
            f"protección social: {p.proteccion_social:.1f}%"
        )
        parts.append(
            f"- servicio deuda: {p.servicio_deuda:.1f}%  |  "
            f"justicia: {p.justicia:.1f}%  |  otros: {p.otros:.1f}%"
        )
        parts.append("")
    return "\n".join(parts)


def build_context(
    state: GuatemalaState,
    territory_summary: dict | None = None,
    eventos_pasados: list[str] | None = None,
) -> str:
    """Serializa el contexto del turno como mensaje de usuario para Claude."""
    parts = []
    parts.append(f"# Turno t={state.turno.t}  ({state.turno.fecha.isoformat()})")
    parts.append("")
    parts.append("## Indicadores actuales")
    parts.append(f"- PIB: USD {state.macro.pib_usd_mm:,.0f} mm "
                 f"(crec {state.macro.crecimiento_pib:+.2f}%, inflac {state.macro.inflacion:.2f}%)")
    parts.append(f"- Deuda/PIB: {state.macro.deuda_pib:.1f}%  |  "
                 f"Balance fiscal: {state.macro.balance_fiscal_pib:+.2f}% PIB  |  "
                 f"Reservas: USD {state.macro.reservas_usd_mm:,.0f} mm")
    parts.append(f"- Tipo de cambio: {state.macro.tipo_cambio:.2f} GTQ/USD  |  "
                 f"Remesas: {state.macro.remesas_pib:.1f}% PIB  |  "
                 f"IED: USD {state.macro.ied_usd_mm:,.0f} mm")
    parts.append("")
    parts.append("## Social")
    parts.append(f"- Pobreza: {state.social.pobreza_general:.1f}% general / "
                 f"{state.social.pobreza_extrema:.1f}% extrema  |  Gini: {state.social.gini:.2f}")
    parts.append(f"- Desempleo: {state.social.desempleo:.1f}%  |  "
                 f"Informalidad: {state.social.informalidad:.1f}%  |  "
                 f"Homicidios/100k: {state.social.homicidios_100k:.1f}")
    parts.append(f"- Migración neta: {state.social.migracion_neta_miles:+.0f} miles  |  "
                 f"Cobertura salud: {state.social.cobertura_salud:.0f}%  |  "
                 f"Matrícula primaria: {state.social.matricula_primaria:.0f}%")
    parts.append("")
    parts.append("## Político")
    parts.append(f"- Aprobación: {state.politico.aprobacion_presidencial:.1f}  |  "
                 f"Protesta: {state.politico.indice_protesta:.1f}  |  "
                 f"Confianza inst: {state.politico.confianza_institucional:.1f}")
    parts.append(f"- Coalición Congreso: {state.politico.coalicion_congreso:.1f}%  |  "
                 f"Libertad prensa: {state.politico.libertad_prensa:.1f}")
    parts.append("")
    parts.append("## Relaciones exteriores")
    parts.append(f"- EE.UU. {state.externo.alineamiento_eeuu:+.2f}  |  "
                 f"China {state.externo.alineamiento_china:+.2f}  |  "
                 f"México {state.externo.relacion_mexico:+.2f}  |  "
                 f"Triángulo Norte {state.externo.relacion_triangulo_norte:+.2f}")
    parts.append(f"- Apoyo multilateral: {state.externo.apoyo_multilateral:.1f}/100")
    parts.append("")
    if state.shocks_activos:
        parts.append("## ⚠️ Shocks activos este turno")
        for s in state.shocks_activos:
            parts.append(f"- {s}")
        parts.append("")
    if territory_summary:
        parts.append("## Resumen territorial")
        ts = territory_summary
        parts.append(f"- Deptos en crisis: {ts['n_deptos_en_crisis']}  "
                     f"(regiones: {', '.join(ts['regiones_criticas']) or '—'})")
        parts.append(f"- Pobreza media ponderada: {ts['pobreza_media_ponderada']:.1f}%")
        parts.append(f"- Top pobreza: {', '.join(ts['deptos_top_pobreza'])}")
        parts.append(f"- Top homicidios: {', '.join(ts['deptos_top_homicidios'])}")
        parts.append(f"- Top sequía: {', '.join(ts['deptos_top_sequia'])}")
        parts.append("")
    if state.memoria_presidencial:
        parts.append("## Memoria presidencial (últimos turnos)")
        for m in state.memoria_presidencial[-8:]:
            parts.append(f"- {m}")
        parts.append("")
    if eventos_pasados:
        parts.append("## Eventos del turno anterior")
        for e in eventos_pasados[-6:]:
            parts.append(f"- {e}")
        parts.append("")
    parts.append("Tomá la decisión del turno llamando a la herramienta `tomar_decision`.")
    return "\n".join(parts)


@dataclass
class ClaudePresidente:
    """Tomador de decisiones basado en la API de Claude."""

    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4_000
    sliding_window: int = 5
    _history: list[dict[str, Any]] = field(default_factory=list)
    _client: Any = None
    ultimos_eventos: list[str] = field(default_factory=list)
    territory_provider: Any = None  # callable() -> dict | None

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic  # lazy
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "anthropic SDK no disponible; `pip install anthropic` o usá DummyDecisionMaker"
                ) from e
            self._client = anthropic.Anthropic()

    def decide(self, state: GuatemalaState) -> DecisionTurno:
        """Llama a Claude para el turno actual.

        Cada turno es una llamada independiente (sin carry-over de mensajes
        entre turnos). La memoria cross-turno vive en `state.memoria_presidencial`
        y se serializa dentro del contexto. Esto evita complicaciones con
        `tool_use`/`tool_result` pairing entre llamadas.
        """
        self._ensure_client()
        tool = _decision_tool_schema()
        ts = self.territory_provider() if callable(self.territory_provider) else None
        user_msg = build_context(state, territory_summary=ts, eventos_pasados=self.ultimos_eventos)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

        ultimo_err: str = ""
        for intento in range(3):
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[tool],
                tool_choice={"type": "tool", "name": "tomar_decision"},
                messages=messages,
            )
            tool_use = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
            if tool_use is None:
                feedback = "No usaste la herramienta `tomar_decision`. Usala."
                ultimo_err = feedback
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": feedback},
                ]
                continue
            try:
                decision = DecisionTurno.model_validate(tool_use.input)
                return decision
            except Exception as e:
                ultimo_err = str(e)[:400]
                feedback = (
                    f"Tu JSON no validó: {ultimo_err}. "
                    f"Revisá: presupuesto SUMA 100 (±1), fiscal dentro de rango "
                    f"[-5,5]/[-10,10], ≤2 reformas. Reenviá via `tomar_decision`."
                )
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": feedback,
                                "is_error": True,
                            }
                        ],
                    },
                ]
        raise RuntimeError(
            f"Claude no devolvió decisión válida tras 3 intentos. Último error: {ultimo_err}"
        )

    def choose_from_menu(
        self,
        state: GuatemalaState,
        candidates: list[PresupuestoAnual],
        candidate_names: list[str] | None = None,
    ) -> tuple[int, DecisionTurno]:
        """Modo menu-choice: presenta candidates y devuelve (idx, decisión).

        Args:
            state: estado actual.
            candidates: lista de 5 candidatos presupuestarios.
            candidate_names: opcional, etiquetas legibles para cada candidato.

        Returns:
            (chosen_index, DecisionTurno) — el DecisionTurno tiene
            `presupuesto = candidates[chosen_index]` ya ensamblado.
        """
        if not candidates:
            raise ValueError("candidates no puede ser vacío")
        self._ensure_client()
        tool = _menu_tool_schema()
        ts = self.territory_provider() if callable(self.territory_provider) else None
        user_msg = (
            build_context(state, territory_summary=ts, eventos_pasados=self.ultimos_eventos)
            + _format_menu(candidates, names=candidate_names)
            + "\nElegí el candidato y completá la decisión llamando a "
              "`elegir_y_decidir`."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

        ultimo_err: str = ""
        for intento in range(3):
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=MENU_SYSTEM_PROMPT,
                tools=[tool],
                tool_choice={"type": "tool", "name": "elegir_y_decidir"},
                messages=messages,
            )
            tool_use = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
            if tool_use is None:
                feedback = "No usaste la herramienta `elegir_y_decidir`. Usala."
                ultimo_err = feedback
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": feedback},
                ]
                continue
            try:
                chosen = ChosenDecision.model_validate(tool_use.input)
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
                    f"Tu input no validó: {ultimo_err}. "
                    f"Recordá: chosen_index ∈ [0, {len(candidates) - 1}], "
                    f"fiscal en rango [-5,5]/[-10,10], ≤2 reformas, "
                    f"razonamiento y mensaje no vacíos."
                )
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": feedback,
                                "is_error": True,
                            }
                        ],
                    },
                ]
        raise RuntimeError(
            f"Claude no devolvió elección válida tras 3 intentos. Último error: {ultimo_err}"
        )
