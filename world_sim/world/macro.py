"""Dinámica macro en Python puro (v1 sin PySD).

Funciones (relativamente) puras sobre `GuatemalaState`. La Fase 6 del plan
refactorizará esto a PySD propiamente dicho.

Las ecuaciones son simplificaciones intencionales; los parámetros están en
`PARAMS` y pueden calibrarse con literatura Banguat / CEPAL.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..actions import DecisionTurno
from ..state import GuatemalaState

# --- parámetros calibrables ---------------------------------------------------


@dataclass
class MacroParams:
    # Crecimiento
    crecimiento_tendencial: float = 3.3       # % anual, base sin shocks
    multiplicador_gasto: float = 0.7          # 0.6–0.9 per spec
    elasticidad_ied_crecimiento: float = 0.05
    peso_remesas_consumo: float = 0.4

    # Precios
    inflacion_objetivo_banguat: float = 4.0
    pass_through_tc: float = 0.15             # elasticidad TC → precios
    inercia_inflacion: float = 0.55

    # Fiscal
    peso_iva_en_ingresos: float = 0.45
    elasticidad_iva: float = 0.7              # corto plazo
    elasticidad_isr: float = 0.6
    presion_tributaria_base: float = 12.5     # % PIB (Guate es ~12%)

    # Externo
    elasticidad_remesas_us: float = 1.2       # sensibilidad a ingreso EE.UU.
    drift_remesas: float = 0.03               # % cambio anual base
    drift_reservas: float = 0.04

    # Social
    elasticidad_pobreza_crecimiento: float = -0.35
    elasticidad_migracion_pobreza: float = 4.0

    # Política / gobernabilidad
    peso_inflacion_en_aprobacion: float = 0.8
    peso_pobreza_en_aprobacion: float = 0.2
    decay_protesta: float = 0.85

    # Ruido / inercia
    ruido_sigma: float = 0.35


PARAMS = MacroParams()


# --- utilidades ---------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --- dinámica ----------------------------------------------------------------


def step_macro(
    state: GuatemalaState,
    decision: DecisionTurno | None,
    rng,
    params: MacroParams = PARAMS,
) -> GuatemalaState:
    """Avanza un año de dinámica macro dada la decisión presidencial.

    Devuelve un nuevo `GuatemalaState` (sin mutar el original).
    """
    s = copy.deepcopy(state)
    m = s.macro

    # shock exógeno agregado (ruido blanco)
    shock = float(rng.normal(0.0, params.ruido_sigma))

    # --- impulso por decisión fiscal / presupuesto ---
    impulso_presupuesto = 0.0
    delta_iva = 0.0
    delta_isr = 0.0
    peso_infra_agro = 0.0
    peso_social = 0.0
    if decision is not None:
        p = decision.presupuesto.normalizado()
        peso_infra_agro = (p.infraestructura + p.agro_desarrollo_rural) / 100.0
        peso_social = (p.salud + p.educacion + p.proteccion_social) / 100.0
        # la decisión estimula PIB a través de la porción productiva del gasto
        impulso_presupuesto = params.multiplicador_gasto * (peso_infra_agro - 0.20)
        delta_iva = decision.fiscal.delta_iva_pp
        delta_isr = decision.fiscal.delta_isr_pp

    # --- crecimiento ---
    g = (
        params.crecimiento_tendencial
        + impulso_presupuesto
        + params.elasticidad_ied_crecimiento * (m.ied_usd_mm / 1_000.0 - 1.5)
        + 0.3 * (m.remesas_pib - 18.0) / 5.0
        - 0.15 * max(delta_iva, 0.0)   # subir IVA modera la demanda
        + shock
    )
    # penalizaciones por shocks activos
    for sh in s.shocks_activos:
        if "sequía" in sh.lower() or "sequia" in sh.lower():
            g -= 0.8
        elif "huracán" in sh.lower() or "huracan" in sh.lower():
            g -= 1.0
        elif "remesas" in sh.lower():
            g -= 0.6
        elif "deportaciones" in sh.lower():
            g -= 0.3

    g = _clamp(g, -8.0, 8.0)
    m.crecimiento_pib = g
    m.pib_usd_mm *= (1.0 + g / 100.0)

    # --- inflación ---
    # componente importado vía TC + inercia + brecha de producto
    brecha = g - params.crecimiento_tendencial
    pi = (
        params.inercia_inflacion * m.inflacion
        + (1 - params.inercia_inflacion) * params.inflacion_objetivo_banguat
        + params.pass_through_tc * (m.tipo_cambio - 7.75)
        + 0.25 * brecha
        + 0.4 * float(rng.normal(0.0, 0.5))
    )
    m.inflacion = _clamp(pi, -2.0, 30.0)

    # --- fiscal ---
    # ingresos tributarios ≈ presión_base + efecto IVA + efecto ISR + rebote PIB
    delta_ingresos_pib = (
        params.elasticidad_iva * delta_iva * params.peso_iva_en_ingresos
        + params.elasticidad_isr * delta_isr * (1 - params.peso_iva_en_ingresos)
        + 0.1 * brecha
    )
    # gasto: se ancla al presupuesto anual y agrega costo fiscal de shocks
    costo_shocks = sum(r.costo_fiscal_pib for r in (decision.respuestas_shocks if decision else []))
    m.balance_fiscal_pib = _clamp(
        m.balance_fiscal_pib + delta_ingresos_pib - costo_shocks + 0.3 * float(rng.normal(0.0, 0.4)),
        -10.0,
        5.0,
    )
    # deuda: acumula déficit
    m.deuda_pib = _clamp(m.deuda_pib - m.balance_fiscal_pib, 5.0, 120.0)

    # --- externo ---
    # remesas con drift + sensibilidad a ciclo US (proxy: shock agregado)
    drift_r = params.drift_remesas - 0.01 * shock
    m.remesas_pib = _clamp(m.remesas_pib * (1.0 + drift_r), 5.0, 30.0)
    m.cuenta_corriente_pib = _clamp(
        0.5 * m.remesas_pib / 10.0 + 0.3 * float(rng.normal(0.0, 0.5)),
        -8.0,
        8.0,
    )
    m.reservas_usd_mm = max(
        0.0,
        m.reservas_usd_mm * (1.0 + params.drift_reservas)
        + 300.0 * m.cuenta_corriente_pib
        + 200.0 * float(rng.normal(0.0, 1.0)),
    )
    m.ied_usd_mm = max(
        0.0,
        m.ied_usd_mm * (1.0 + 0.02 + 0.01 * (s.politico.confianza_institucional - 30.0) / 10.0)
        + 150.0 * float(rng.normal(0.0, 1.0)),
    )
    # tipo de cambio: deriva según diferencial de inflación vs. 2%
    m.tipo_cambio = _clamp(
        m.tipo_cambio * (1.0 + (m.inflacion - 2.0) / 200.0 + 0.002 * float(rng.normal(0.0, 1.0))),
        5.0,
        15.0,
    )

    # --- social ---
    soc = s.social
    # pobreza reacciona con inercia al crecimiento y al gasto social
    delta_pobreza = (
        params.elasticidad_pobreza_crecimiento * g
        - 2.5 * (peso_social - 0.35)  # aumentar gasto social reduce pobreza
        + 0.4 * float(rng.normal(0.0, 1.0))
    )
    soc.pobreza_general = _clamp(soc.pobreza_general + delta_pobreza, 10.0, 90.0)
    soc.pobreza_extrema = _clamp(soc.pobreza_extrema + 0.6 * delta_pobreza, 1.0, 60.0)
    soc.gini = _clamp(soc.gini + 0.002 * (delta_pobreza), 0.30, 0.70)
    # migración neta: empuja cuando pobreza sube
    soc.migracion_neta_miles = _clamp(
        -params.elasticidad_migracion_pobreza * (soc.pobreza_general - 45.0)
        + 5.0 * float(rng.normal(0.0, 1.0)),
        -500.0,
        50.0,
    )
    # desempleo: inversa al crecimiento
    soc.desempleo = _clamp(soc.desempleo - 0.3 * g + 0.2 * float(rng.normal(0.0, 1.0)), 1.0, 30.0)
    # homicidios: baja con gasto en seguridad + justicia
    peso_seguridad = 0.0
    if decision is not None:
        p = decision.presupuesto.normalizado()
        peso_seguridad = (p.seguridad + p.justicia) / 100.0
    soc.homicidios_100k = _clamp(
        soc.homicidios_100k * (1.0 - 0.05 * (peso_seguridad - 0.15))
        + 0.8 * float(rng.normal(0.0, 1.0)),
        1.0,
        80.0,
    )
    # cobertura salud/educación sube con gasto social
    soc.cobertura_salud = _clamp(
        soc.cobertura_salud + 1.5 * (peso_social - 0.30), 10.0, 100.0
    )
    soc.matricula_primaria = _clamp(
        soc.matricula_primaria + 1.0 * (peso_social - 0.30), 30.0, 100.0
    )

    # --- política / gobernabilidad ---
    pol = s.politico
    # aprobación ~ -inflación -pobreza +crecimiento +mensaje populista
    delta_aprob = (
        -params.peso_inflacion_en_aprobacion * (m.inflacion - 4.0)
        - params.peso_pobreza_en_aprobacion * (soc.pobreza_general - 50.0)
        + 0.5 * g
        - 2.0 * len(s.shocks_activos)
        + 2.0 * float(rng.normal(0.0, 1.0))
    )
    pol.aprobacion_presidencial = _clamp(pol.aprobacion_presidencial + delta_aprob, 0.0, 100.0)
    # protesta decae pero sube con pobreza y shocks
    pol.indice_protesta = _clamp(
        params.decay_protesta * pol.indice_protesta
        + 0.4 * (soc.pobreza_general - 50.0)
        + 5.0 * len(s.shocks_activos)
        + 2.0 * float(rng.normal(0.0, 1.0)),
        0.0,
        100.0,
    )
    # confianza institucional: inercia + efecto reformas
    n_reformas = len(decision.reformas) if decision is not None else 0
    pol.confianza_institucional = _clamp(
        0.85 * pol.confianza_institucional + 2.0 * n_reformas + 0.3 * g,
        0.0,
        100.0,
    )
    # coalición: ligado a aprobación
    pol.coalicion_congreso = _clamp(
        0.7 * pol.coalicion_congreso + 0.3 * pol.aprobacion_presidencial / 1.5,
        0.0,
        100.0,
    )

    # --- externo (relaciones) ---
    ext = s.externo
    if decision is not None:
        al = decision.exterior.alineamiento_priorizado
        if al == "eeuu":
            ext.alineamiento_eeuu = _clamp(ext.alineamiento_eeuu + 0.05, -1.0, 1.0)
            ext.alineamiento_china = _clamp(ext.alineamiento_china - 0.03, -1.0, 1.0)
        elif al == "china":
            ext.alineamiento_china = _clamp(ext.alineamiento_china + 0.05, -1.0, 1.0)
            ext.alineamiento_eeuu = _clamp(ext.alineamiento_eeuu - 0.03, -1.0, 1.0)
        elif al == "multilateral":
            ext.apoyo_multilateral = _clamp(ext.apoyo_multilateral + 2.5, 0.0, 100.0)
        elif al == "regional":
            ext.relacion_mexico = _clamp(ext.relacion_mexico + 0.03, -1.0, 1.0)
            ext.relacion_triangulo_norte = _clamp(
                ext.relacion_triangulo_norte + 0.04, -1.0, 1.0
            )

    return s
