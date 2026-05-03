"""Turn loop: orquesta macro, shocks, decisión y logging."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Protocol

import numpy as np

from .actions import (
    DecisionTurno,
    Fiscal,
    PoliticaExterior,
    PresupuestoAnual,
)
from .state import GuatemalaState, Turno
from .world.macro import MacroParams, PARAMS, step_macro
from .world.shocks import sample_shocks


class DecisionMaker(Protocol):
    """Protocolo que todo tomador de decisiones (Claude, dummy, histórico) cumple."""

    def decide(self, state: GuatemalaState) -> DecisionTurno: ...


@dataclass
class EngineConfig:
    seed: int = 42
    params: MacroParams = field(default_factory=lambda: PARAMS)


@dataclass
class TurnRecord:
    """Resumen de un turno — lo que se loguea."""

    t: int
    fecha_iso: str
    state_before: dict
    decision: dict
    state_after: dict
    shocks_activos: list[str]
    menu_choice: dict | None = None
    """Si el turno corrió en menu-choice mode, dict con
    `{"chosen_index": int, "candidates": [{"name": str, "presupuesto": dict}]}`.
    None en modo legacy (composición libre)."""


def advance_date(d: date, periodo: str) -> date:
    if periodo == "anual":
        return d.replace(year=d.year + 1)
    # trimestral
    month = d.month + 3
    year = d.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return date(year, month, 1)


def _aplicar_memoria(state: GuatemalaState, decision: DecisionTurno) -> GuatemalaState:
    """Deja trazas estructuradas en `memoria_presidencial`."""
    s = copy.deepcopy(state)
    entrada = (
        f"t={state.turno.t} align={decision.exterior.alineamiento_priorizado} "
        f"ΔIVA={decision.fiscal.delta_iva_pp:+.1f} "
        f"reformas={[r.area for r in decision.reformas]}"
    )
    s.memoria_presidencial = (s.memoria_presidencial + [entrada])[-50:]
    return s


def run_turn(
    state: GuatemalaState,
    decision_maker: DecisionMaker,
    rng: np.random.Generator,
    params: MacroParams = PARAMS,
    hooks: list[Callable[[TurnRecord], None]] | None = None,
    agentes=None,          # AgentesModel | None
    territorio=None,       # Territory | None
    menu_mode: bool = False,
    menu_candidates_provider: Callable[[], list] | None = None,
) -> tuple[GuatemalaState, TurnRecord]:
    """Avanza un turno completo y devuelve el nuevo state + el record del turno.

    Si `agentes` (AgentesModel) se pasa, las reacciones modifican el state
    post-macro. Si `territorio` (Territory) se pasa, sus shocks climáticos
    se propagan y su resumen se adjunta al record.

    Args:
        menu_mode: si True, el `decision_maker` debe implementar
            `choose_from_menu(state, candidates) -> (int, DecisionTurno)`
            en vez de `decide(state)`. Habilita el flujo de IRL bayesiano
            sobre elecciones discretas.
        menu_candidates_provider: callable opcional que devuelve la lista de
            `Candidate`. Default: `irl.candidates.generate_candidate_menu`.
            Sólo se usa si `menu_mode=True`.
    """
    hooks = hooks or []

    # 1. Shocks (para que el tomador de decisiones pueda responder)
    state_con_shocks = sample_shocks(state, rng)

    # 2. Territorio: step + resumen
    territory_summary = None
    if territorio is not None:
        territorio.step(state_con_shocks, rng)
        territory_summary = territorio.summary().as_dict()

    # 3. Decisión (legacy free-composition o nuevo menu-choice)
    menu_choice_record: dict | None = None
    if menu_mode:
        if menu_candidates_provider is None:
            from .irl.candidates import generate_candidate_menu
            menu_candidates_provider = generate_candidate_menu
        candidates_full = menu_candidates_provider()
        candidate_presupuestos = [c.presupuesto for c in candidates_full]
        candidate_names = [c.name for c in candidates_full]
        chosen_idx, decision = decision_maker.choose_from_menu(
            state_con_shocks, candidate_presupuestos, candidate_names
        )
        menu_choice_record = {
            "chosen_index": int(chosen_idx),
            "candidates": [
                {"name": c.name, "presupuesto": c.presupuesto.model_dump()}
                for c in candidates_full
            ],
        }
    else:
        decision = decision_maker.decide(state_con_shocks)

    # 4. Física del mundo con la decisión
    state_post_macro = step_macro(state_con_shocks, decision, rng, params=params)

    # 5. Agentes reaccionan a la decisión
    agentes_eventos: list[str] = []
    if agentes is not None:
        state_post_macro, impactos = agentes.step_con_decision(state_post_macro, decision)
        agentes_eventos = [i.evento for i in impactos if i.evento]

    # 6. Territorio: inversión rebaja pobreza rural (después de la decisión)
    if territorio is not None:
        p = decision.presupuesto.normalizado()
        peso_infra = (p.infraestructura + p.agro_desarrollo_rural) / 100.0
        territorio.aplicar_inversion_infraestructura(peso_infra)

    # 7. Memoria presidencial
    state_final = _aplicar_memoria(state_post_macro, decision)

    # 8. Avanzar turno
    nuevo_t = state_final.turno.t + 1
    nueva_fecha = advance_date(state_final.turno.fecha, state_final.turno.periodo)
    state_final = state_final.model_copy(
        update={"turno": Turno(t=nuevo_t, fecha=nueva_fecha, periodo=state_final.turno.periodo)}
    )

    record = TurnRecord(
        t=state.turno.t,
        fecha_iso=state.turno.fecha.isoformat(),
        state_before=state.model_dump(mode="json"),
        decision=decision.model_dump(mode="json"),
        state_after=state_final.model_dump(mode="json"),
        shocks_activos=list(state_con_shocks.shocks_activos),
        menu_choice=menu_choice_record,
    )
    # campos extra no tipados (territorio / eventos de agentes)
    record_extra = {
        "territorio": territory_summary,
        "eventos_agentes": agentes_eventos,
    }
    setattr(record, "extra", record_extra)
    for h in hooks:
        h(record)

    return state_final, record


# --- tomador de decisiones dummy (para pruebas sin API) ----------------------


class DummyDecisionMaker:
    """Devuelve una decisión razonable, constante, con ligeras variaciones."""

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def decide(self, state: GuatemalaState) -> DecisionTurno:
        # Presupuesto baseline "razonable"
        presupuesto = PresupuestoAnual(
            salud=12,
            educacion=18,
            seguridad=11,
            infraestructura=14,
            agro_desarrollo_rural=8,
            proteccion_social=10,
            servicio_deuda=10,
            justicia=5,
            otros=12,
        )
        fiscal = Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0)
        exterior = PoliticaExterior(
            alineamiento_priorizado="multilateral",
            acciones_diplomaticas=["diálogo con BID"],
        )
        # respuestas a shocks: 0.3% PIB por shock
        from .actions import RespuestaShock

        respuestas = [
            RespuestaShock(shock=sh, medida="paquete de emergencia", costo_fiscal_pib=0.3)
            for sh in state.shocks_activos
        ]
        return DecisionTurno(
            razonamiento=(
                "continuidad y estabilidad macro; priorizar infraestructura y "
                "educación. respuesta contracíclica a shocks activos."
            ),
            presupuesto=presupuesto,
            fiscal=fiscal,
            exterior=exterior,
            respuestas_shocks=respuestas,
            reformas=[],
            mensaje_al_pueblo=(
                "seguimos trabajando por la estabilidad y el bienestar de todas "
                "las familias guatemaltecas."
            ),
        )


# --- tomador de decisiones dummy en menu-choice mode (para tests sin API) ---


class DummyMenuDecisionMaker:
    """Análogo de `DummyDecisionMaker` pero para `menu_mode=True`.

    Implementa `choose_from_menu(state, candidates) -> (int, DecisionTurno)`
    eligiendo siempre el mismo índice configurable. Los demás campos de la
    decisión (fiscal, exterior, etc.) son neutros y constantes.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        selected_index: int = 4,
    ):
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.selected_index = int(selected_index)

    def choose_from_menu(
        self,
        state: GuatemalaState,
        candidates: list[PresupuestoAnual],
        candidate_names: list[str] | None = None,
    ) -> tuple[int, DecisionTurno]:
        if not candidates:
            raise ValueError("candidates no puede ser vacío")
        idx = self.selected_index % len(candidates)
        from .actions import RespuestaShock

        respuestas = [
            RespuestaShock(shock=sh, medida="paquete de emergencia", costo_fiscal_pib=0.3)
            for sh in state.shocks_activos
        ]
        decision = DecisionTurno(
            razonamiento=(
                f"dummy menu pick (selected_index={self.selected_index}): "
                f"continuidad operativa con respuesta contracíclica."
            ),
            presupuesto=candidates[idx],
            fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0),
            exterior=PoliticaExterior(
                alineamiento_priorizado="multilateral",
                acciones_diplomaticas=["diálogo con BID"],
            ),
            respuestas_shocks=respuestas,
            reformas=[],
            mensaje_al_pueblo=(
                "seguimos trabajando por la estabilidad y el bienestar."
            ),
        )
        return idx, decision
