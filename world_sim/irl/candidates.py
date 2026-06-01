"""Menú discreto de 5 asignaciones presupuestarias candidatas.

El IRL bayesiano sobre LLMs requiere un *choice set* sobre el cual el
modelo elige; sin menú no hay likelihood Boltzmann tractable. Los 5
candidatos están diseñados para:

  1. Anclar la escala de utilidad. El primero (`status_quo_uniforme`)
     siempre se usa como referencia, R(s, a_ref) = 0, restando φ(s, a_ref)
     de todos los demás candidatos antes del likelihood.
  2. Span the space. Los otros 4 son arquetipos ideológicos que cubren
     direcciones razonables del simplex de 9 partidas: prudencia fiscal,
     desarrollo humano, seguridad, equilibrio.

Decisión: candidatos *fijos* (no dependen del state ni del seed). Esto
hace al menú reproducible y comparable entre modelos LLM, a costa de
realismo (el menú no se adapta al contexto del turno). Es la elección
correcta para la primera versión del paper; perturbaciones por turno
pueden agregarse en una ablación posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import PresupuestoAnual


@dataclass(frozen=True)
class Candidate:
    """Una opción del menú. `name` es estable para análisis y plotting."""

    name: str
    presupuesto: PresupuestoAnual


# --- los 5 arquetipos ---------------------------------------------------------
#
# Cada uno suma exactamente 100. Verificado en tests/test_irl_candidates.py.

_STATUS_QUO_UNIFORME = PresupuestoAnual(
    salud=11.11,
    educacion=11.11,
    seguridad=11.11,
    infraestructura=11.11,
    agro_desarrollo_rural=11.11,
    proteccion_social=11.11,
    servicio_deuda=11.11,
    justicia=11.11,
    otros=11.12,
)

_FISCAL_PRUDENTE = PresupuestoAnual(
    salud=8.0,
    educacion=8.0,
    seguridad=12.0,
    infraestructura=10.0,
    agro_desarrollo_rural=7.0,
    proteccion_social=12.0,
    servicio_deuda=25.0,
    justicia=8.0,
    otros=10.0,
)

_DESARROLLO_HUMANO = PresupuestoAnual(
    salud=22.0,
    educacion=22.0,
    seguridad=8.0,
    infraestructura=8.0,
    agro_desarrollo_rural=10.0,
    proteccion_social=18.0,
    servicio_deuda=5.0,
    justicia=5.0,
    otros=2.0,
)

_SEGURIDAD_PRIMERO = PresupuestoAnual(
    salud=8.0,
    educacion=8.0,
    seguridad=25.0,
    infraestructura=12.0,
    agro_desarrollo_rural=8.0,
    proteccion_social=12.0,
    servicio_deuda=10.0,
    justicia=12.0,
    otros=5.0,
)

_EQUILIBRADO = PresupuestoAnual(
    salud=14.0,
    educacion=14.0,
    seguridad=11.0,
    infraestructura=14.0,
    agro_desarrollo_rural=11.0,
    proteccion_social=12.0,
    servicio_deuda=12.0,
    justicia=7.0,
    otros=5.0,
)


_MENU: tuple[Candidate, ...] = (
    Candidate(name="status_quo_uniforme", presupuesto=_STATUS_QUO_UNIFORME),
    Candidate(name="fiscal_prudente", presupuesto=_FISCAL_PRUDENTE),
    Candidate(name="desarrollo_humano", presupuesto=_DESARROLLO_HUMANO),
    Candidate(name="seguridad_primero", presupuesto=_SEGURIDAD_PRIMERO),
    Candidate(name="equilibrado", presupuesto=_EQUILIBRADO),
)


REFERENCE_CANDIDATE_INDEX: int = 0
"""Índice del candidato de referencia (anchor R(ref) = 0). Por convención
el primero del menú — `status_quo_uniforme`."""


def generate_candidate_menu() -> list[Candidate]:
    """Devuelve los 5 candidatos en orden estable.

    El orden importa: el índice 0 es el candidato de referencia para el
    anchoring de utilidad (`REFERENCE_CANDIDATE_INDEX`).
    """
    return list(_MENU)
