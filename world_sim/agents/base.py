"""Infraestructura común de agentes.

Mesa v3 usa `Model` + `Agent`. Cada agente guarda un historial de reacciones
a la última decisión presidencial, y expone un método `impacto(state)` que
devuelve deltas a aplicar sobre el state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import mesa

from ..actions import DecisionTurno
from ..state import GuatemalaState


@dataclass
class Impacto:
    """Delta aditivo que un agente aplica sobre el state después de la decisión."""

    delta_aprobacion: float = 0.0
    delta_protesta: float = 0.0
    delta_coalicion: float = 0.0
    delta_confianza: float = 0.0
    delta_ied_mm: float = 0.0
    evento: str = ""


class AgenteBase(mesa.Agent):
    """Base de todos los agentes del sim."""

    nombre: str = "agente"

    def __init__(self, model: "AgentesModel"):
        super().__init__(model)
        self.historial: list[Impacto] = []

    def reaccionar(self, state: GuatemalaState, decision: DecisionTurno) -> Impacto:
        """Override por subclase."""
        return Impacto()


class AgentesModel(mesa.Model):
    """Modelo Mesa que corre los agentes en orden simultáneo cada turno."""

    def __init__(self, agent_classes: list[type[AgenteBase]], seed: int | None = None):
        super().__init__(seed=seed)
        self._by_name: dict[str, AgenteBase] = {}
        for cls in agent_classes:
            ag = cls(self)
            self._by_name[cls.nombre] = ag

    def step_con_decision(
        self, state: GuatemalaState, decision: DecisionTurno
    ) -> tuple[GuatemalaState, list[Impacto]]:
        impactos: list[Impacto] = []
        for ag in self._by_name.values():
            imp = ag.reaccionar(state, decision)
            ag.historial.append(imp)
            impactos.append(imp)
        # aplicar impactos agregados
        s = copy.deepcopy(state)
        for imp in impactos:
            s.politico.aprobacion_presidencial = max(
                0.0, min(100.0, s.politico.aprobacion_presidencial + imp.delta_aprobacion)
            )
            s.politico.indice_protesta = max(
                0.0, min(100.0, s.politico.indice_protesta + imp.delta_protesta)
            )
            s.politico.coalicion_congreso = max(
                0.0, min(100.0, s.politico.coalicion_congreso + imp.delta_coalicion)
            )
            s.politico.confianza_institucional = max(
                0.0, min(100.0, s.politico.confianza_institucional + imp.delta_confianza)
            )
            s.macro.ied_usd_mm = max(0.0, s.macro.ied_usd_mm + imp.delta_ied_mm)
            if imp.evento:
                s.eventos_turno.append(imp.evento)
        return s, impactos

    def get(self, nombre: str) -> AgenteBase:
        return self._by_name[nombre]
