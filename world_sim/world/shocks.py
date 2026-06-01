"""Muestreo de shocks por turno.

Los shocks se inyectan en `state.shocks_activos` **antes** de mandar el
contexto al presidente, para que pueda responderlos en su decisión.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..state import GuatemalaState


@dataclass(frozen=True)
class ShockDef:
    nombre: str
    p_base: float
    etiqueta: str  # lo que se ve en shocks_activos


SHOCKS: list[ShockDef] = [
    ShockDef("sequia_corredor_seco", 0.15, "sequía severa en corredor seco"),
    ShockDef("huracan", 0.10, "huracán en costa atlántica"),
    ShockDef("caida_remesas", 0.08, "caída de remesas >10%"),
    ShockDef("deportaciones_masivas", 0.20, "deportaciones masivas desde EEUU"),
    ShockDef("escandalo_corrupcion", 0.25, "escándalo de corrupción"),
    ShockDef("crisis_gobernabilidad", 0.05, "crisis de gobernabilidad"),
    ShockDef("colapso_vecino", 0.06, "crisis política en vecino regional"),
]


def sample_shocks(state: GuatemalaState, rng) -> GuatemalaState:
    """Devuelve un estado con `shocks_activos` poblado para el turno."""
    nuevos: list[str] = []
    for sh in SHOCKS:
        p = sh.p_base
        # Ajustes contextuales:
        if sh.nombre == "escandalo_corrupcion":
            # menos justicia -> más probable
            p = sh.p_base * (1.5 - state.politico.confianza_institucional / 100.0)
        if sh.nombre == "crisis_gobernabilidad":
            p = sh.p_base + 0.3 * (state.politico.indice_protesta / 100.0) \
                + 0.3 * max(0.0, (50.0 - state.politico.aprobacion_presidencial) / 100.0)
        if sh.nombre == "caida_remesas":
            # más probable si remesas son muy altas (base de comparación)
            p = sh.p_base * (state.macro.remesas_pib / 20.0)
        p = max(0.0, min(1.0, p))
        if rng.random() < p:
            nuevos.append(sh.etiqueta)
    state = state.model_copy(update={"shocks_activos": nuevos})
    return state
