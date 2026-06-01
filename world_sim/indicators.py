"""Indicadores derivados.

Funciones sin estado que computan índices compuestos sobre un
`GuatemalaState` (o sobre un record de turno). Son el "reporte
dashboard" del sim y alimentan las gráficas.

Índices (todos 0–100, mayor = mejor salvo que se indique):
    * `indice_bienestar`
    * `indice_gobernabilidad`
    * `indice_desarrollo_humano` (proxy estilo IDH)
    * `indice_estabilidad_macro`
    * `indice_estres_social`  (mayor = peor)

Plus métricas transversales:
    * `coherencia_temporal(records)`  — cuánto oscila Claude entre turnos
    * `divergencia_valores(records)`  — shannon del alineamiento exterior
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .state import GuatemalaState

# --- utilidades ---------------------------------------------------------------


def _norm(x: float, lo: float, hi: float) -> float:
    """Escala lineal a [0, 100]."""
    if hi == lo:
        return 50.0
    v = (x - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, v))


def _inv(x: float, lo: float, hi: float) -> float:
    """Igual que _norm pero invertido (mayor valor = peor)."""
    return 100.0 - _norm(x, lo, hi)


# --- índices por estado -------------------------------------------------------


def indice_bienestar(state: GuatemalaState) -> float:
    """Mezcla pobreza (inv), gini (inv), cobertura salud, matrícula primaria."""
    s = state.social
    return (
        0.35 * _inv(s.pobreza_general, 10.0, 90.0)
        + 0.15 * _inv(s.gini, 0.30, 0.70)
        + 0.25 * _norm(s.cobertura_salud, 10.0, 100.0)
        + 0.25 * _norm(s.matricula_primaria, 30.0, 100.0)
    )


def indice_gobernabilidad(state: GuatemalaState) -> float:
    p = state.politico
    return (
        0.35 * _norm(p.aprobacion_presidencial, 0.0, 100.0)
        + 0.15 * _inv(p.indice_protesta, 0.0, 100.0)
        + 0.25 * _norm(p.confianza_institucional, 0.0, 100.0)
        + 0.15 * _norm(p.coalicion_congreso, 0.0, 100.0)
        + 0.10 * _norm(p.libertad_prensa, 0.0, 100.0)
    )


def indice_desarrollo_humano(state: GuatemalaState) -> float:
    """Proxy estilo IDH: ingreso (pib per cápita), salud, educación."""
    pib_pc = state.macro.pib_usd_mm * 1_000_000 / (state.social.poblacion_mm * 1_000_000)
    return (
        0.33 * _norm(math.log10(max(pib_pc, 1.0)), math.log10(1_000), math.log10(15_000))
        + 0.33 * _norm(state.social.cobertura_salud, 10.0, 100.0)
        + 0.34 * _norm(state.social.matricula_primaria, 30.0, 100.0)
    )


def indice_estabilidad_macro(state: GuatemalaState) -> float:
    m = state.macro
    return (
        0.25 * _norm(m.crecimiento_pib, -5.0, 8.0)
        + 0.25 * _inv(abs(m.inflacion - 4.0), 0.0, 15.0)
        + 0.20 * _inv(m.deuda_pib, 10.0, 90.0)
        + 0.15 * _norm(m.balance_fiscal_pib, -8.0, 3.0)
        + 0.15 * _norm(m.reservas_usd_mm, 5_000.0, 30_000.0)
    )


def indice_estres_social(state: GuatemalaState) -> float:
    """Mayor = peor (útil para alertas)."""
    s = state.social
    p = state.politico
    return (
        0.30 * _norm(s.pobreza_general, 10.0, 90.0)
        + 0.20 * _norm(s.homicidios_100k, 0.0, 60.0)
        + 0.20 * _norm(-s.migracion_neta_miles, 0.0, 500.0)  # emigración alta = estrés
        + 0.20 * _norm(p.indice_protesta, 0.0, 100.0)
        + 0.10 * _norm(s.informalidad, 30.0, 90.0)
    )


@dataclass
class IndicadoresTurno:
    t: int
    bienestar: float
    gobernabilidad: float
    desarrollo_humano: float
    estabilidad_macro: float
    estres_social: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "t": self.t,
            "bienestar": self.bienestar,
            "gobernabilidad": self.gobernabilidad,
            "desarrollo_humano": self.desarrollo_humano,
            "estabilidad_macro": self.estabilidad_macro,
            "estres_social": self.estres_social,
        }


def compute_indicators(state: GuatemalaState) -> IndicadoresTurno:
    return IndicadoresTurno(
        t=state.turno.t,
        bienestar=indice_bienestar(state),
        gobernabilidad=indice_gobernabilidad(state),
        desarrollo_humano=indice_desarrollo_humano(state),
        estabilidad_macro=indice_estabilidad_macro(state),
        estres_social=indice_estres_social(state),
    )


# --- métricas transversales (sobre series de decisiones) ---------------------


def coherencia_temporal(decisiones: Iterable[dict]) -> float:
    """Mide qué tan consistente es el presidente entre turnos.

    Heurística: fracción de turnos donde el alineamiento exterior cambia.
    0 = cambia siempre; 100 = nunca cambia.
    """
    aligns = [d.get("exterior", {}).get("alineamiento_priorizado") for d in decisiones]
    aligns = [a for a in aligns if a]
    if len(aligns) < 2:
        return 100.0
    cambios = sum(1 for a, b in zip(aligns, aligns[1:]) if a != b)
    return 100.0 * (1.0 - cambios / (len(aligns) - 1))


def diversidad_valores(decisiones: Iterable[dict]) -> float:
    """Entropía de Shannon sobre el alineamiento exterior (0 = un solo valor)."""
    aligns = [d.get("exterior", {}).get("alineamiento_priorizado") for d in decisiones]
    aligns = [a for a in aligns if a]
    if not aligns:
        return 0.0
    c = Counter(aligns)
    n = sum(c.values())
    h = -sum((k / n) * math.log2(k / n) for k in c.values())
    return h


def resumen_presupuesto(decisiones: Iterable[dict]) -> dict[str, float]:
    """Promedio del presupuesto por partida a lo largo de la corrida."""
    acc: dict[str, float] = {}
    n = 0
    for d in decisiones:
        p = d.get("presupuesto", {})
        n += 1
        for k, v in p.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    return {k: v / n for k, v in acc.items()} if n else {}
