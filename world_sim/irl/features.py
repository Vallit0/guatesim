"""Vector de features φ(s, a) para el IRL bayesiano.

φ se define sobre **outcomes esperados**, no sobre las shares
presupuestarias directamente. Esto es lo que diferencia este IRL del
Dirichlet-multinomial existente: en lugar de recuperar pesos sobre
partidas (que es trivialmente lo que el LLM eligió), recuperamos pesos
sobre dimensiones de bienestar — qué le importa al LLM más allá del
mecanismo.

Implementación:

  φ(s, a) = E_ω [ outcome(s, a, ω) ]

donde ω es el ruido del simulador (gaussiano + bernoulli) y la esperanza
se aproxima por Monte Carlo con `n_samples` corridas independientes del
mismo turno con el mismo state inicial y la misma decisión candidata.

Las 6 dimensiones de φ están firmadas para que **mayor sea siempre
mejor**, lo cual hace los pesos w directamente interpretables: w_k > 0
significa "el LLM valora positivamente la dimensión k".
"""

from __future__ import annotations

import copy

import numpy as np

from ..actions import (
    DecisionTurno,
    Fiscal,
    PoliticaExterior,
    PresupuestoAnual,
)
from ..state import GuatemalaState
from ..world.macro import MacroParams, PARAMS, step_macro


OUTCOME_FEATURE_NAMES: tuple[str, ...] = (
    "anti_pobreza",       # -Δ pobreza_general
    "anti_deuda",         # -Δ deuda_pib
    "pro_aprobacion",     # +Δ aprobación_presidencial
    "pro_crecimiento",    # +crecimiento_pib
    "anti_desviacion_inflacion",  # -|inflación - 4|
    "pro_confianza",      # +Δ confianza_institucional
)
"""Nombres de las 6 dimensiones de φ. Mayor = mejor en todas."""


N_OUTCOME_FEATURES: int = len(OUTCOME_FEATURE_NAMES)


def _minimal_decision(candidate: PresupuestoAnual) -> DecisionTurno:
    """Construye una DecisionTurno mínima usando el candidato como
    presupuesto, con todo lo demás en valores neutros.

    Razón: `step_macro` requiere un `DecisionTurno` válido pero solo usa
    presupuesto + fiscal + respuestas_shocks + reformas + alineamiento.
    Para extraer features queremos aislar el efecto del presupuesto, así
    que dejamos lo demás en cero/neutro.
    """
    return DecisionTurno(
        razonamiento="feature_extraction",
        presupuesto=candidate,
        fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0),
        exterior=PoliticaExterior(alineamiento_priorizado="multilateral"),
        respuestas_shocks=[],
        reformas=[],
        mensaje_al_pueblo="feature_extraction",
    )


def _outcome_vector(
    state_before: GuatemalaState,
    state_after: GuatemalaState,
) -> np.ndarray:
    """Convierte (state_before, state_after) en el vector φ de 6 dims.

    Todas las componentes están firmadas en dirección "mayor = mejor".
    """
    mb, sb, pb = state_before.macro, state_before.social, state_before.politico
    ma, sa, pa = state_after.macro, state_after.social, state_after.politico
    return np.array(
        [
            -(sa.pobreza_general - sb.pobreza_general),       # anti_pobreza
            -(ma.deuda_pib - mb.deuda_pib),                    # anti_deuda
            pa.aprobacion_presidencial - pb.aprobacion_presidencial,  # pro_aprobacion
            ma.crecimiento_pib,                                # pro_crecimiento
            -abs(ma.inflacion - 4.0),                          # anti_desviacion_inflacion
            pa.confianza_institucional - pb.confianza_institucional,  # pro_confianza
        ],
        dtype=float,
    )


def extract_outcome_features(
    state_before: GuatemalaState,
    candidate: PresupuestoAnual,
    feature_seed: int = 0,
    n_samples: int = 20,
    params: MacroParams = PARAMS,
) -> np.ndarray:
    """Estima φ(s, a) = E_ω[outcome(s, a, ω)] vía Monte Carlo.

    Args:
        state_before: estado del mundo antes del turno.
        candidate: la asignación presupuestaria candidata.
        feature_seed: semilla base; los n_samples sub-rngs se derivan
            determinísticamente de esta. Misma semilla → mismo vector φ.
        n_samples: número de simulaciones Monte Carlo a promediar.
            Default 20 (suficiente para reducir el ruido gaussiano del
            simulador a ~σ/√20 ≈ 22 % del original).
        params: parámetros del modelo macro.

    Returns:
        np.ndarray de shape (6,) con las features promediadas. Orden y
        signos: ver `OUTCOME_FEATURE_NAMES`.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples debe ser ≥ 1; recibí {n_samples}")

    decision = _minimal_decision(candidate)
    samples = np.zeros((n_samples, N_OUTCOME_FEATURES), dtype=float)
    for i in range(n_samples):
        # rng derivado determinísticamente de (feature_seed, i)
        rng = np.random.default_rng(np.uint64(feature_seed) * np.uint64(1_000_003) + np.uint64(i))
        # step_macro hace deepcopy internamente, así que no muta state_before
        state_after = step_macro(state_before, decision, rng, params=params)
        samples[i] = _outcome_vector(state_before, state_after)

    return samples.mean(axis=0)
