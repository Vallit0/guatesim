"""Wrapper resiliente de DecisionMaker.

Envuelve a cualquier tomador de decisiones y, si falla en producir una
decisión válida, delega a un fallback (típicamente DummyDecisionMaker).
Cuenta los fallos para análisis posterior.

Útil para correr modelos inestables (Qwen 0.5b, Llama chicos, etc.) en
el loop sin abortar la corrida entera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import DecisionTurno
from .state import GuatemalaState


@dataclass
class ResilientDecisionMaker:
    primario: Any                       # tomador principal (puede fallar)
    fallback: Any                       # típicamente DummyDecisionMaker(rng)
    label: str = "Resilient"
    n_llamadas: int = 0
    n_fallos: int = 0
    fallos: list[str] = field(default_factory=list)
    ultimos_eventos: list[str] = field(default_factory=list)
    territory_provider: Any = None

    # propaga la config al primario también
    def __post_init__(self):
        if hasattr(self.primario, "territory_provider"):
            self.primario.territory_provider = self.territory_provider

    def decide(self, state: GuatemalaState) -> DecisionTurno:
        self.n_llamadas += 1
        # sync territory_provider y eventos por si cambian turno a turno
        if hasattr(self.primario, "territory_provider") and self.territory_provider:
            self.primario.territory_provider = self.territory_provider
        if hasattr(self.primario, "ultimos_eventos"):
            self.primario.ultimos_eventos = self.ultimos_eventos
        try:
            return self.primario.decide(state)
        except Exception as e:
            self.n_fallos += 1
            self.fallos.append(f"t={state.turno.t}: {type(e).__name__}: {str(e)[:200]}")
            return self.fallback.decide(state)

    @property
    def tasa_fallo(self) -> float:
        return 100.0 * self.n_fallos / self.n_llamadas if self.n_llamadas else 0.0
