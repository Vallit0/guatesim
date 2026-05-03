"""Likelihood Boltzmann para IRL bayesiano sobre LLMs.

Modelo (Ramachandran & Amir 2007 — Bayesian IRL con racionalidad
boundedly Boltzmann; equivalente al conditional logit de McFadden 1974
cuando el menú de acción es discreto):

    P(a | s, w) = exp(wᵀ φ̃(s, a)) / Σ_{a' ∈ A(s)} exp(wᵀ φ̃(s, a'))

donde:
    - φ̃(s, a) = φ(s, a) - φ(s, a_ref)  son las features restando la
      referencia. La sustracción hace R(s, a_ref) = wᵀ φ̃(s, a_ref) = 0
      por construcción, anclando la escala absoluta de utilidad.
    - w ∈ ℝᵈ son los pesos sobre las dimensiones de bienestar.
    - La temperatura T del LLM está absorbida implícitamente en la
      norma de w: ‖w‖ grande ⇔ preferencias fuertes y/o sampling
      concentrado (T efectiva baja); ‖w‖ ≈ 0 ⇔ elecciones casi uniformes.
      Esto es la elección estándar en IRL para evitar la indeterminación
      conjunta (w, T) ↔ (cw, cT).

Este módulo es **NumPy puro**. La versión simbólica para PyMC vive en
`bayesian_irl.py` y replica esta misma matemática.

Referencias:
    - Ramachandran & Amir (2007), Bayesian Inverse Reinforcement Learning,
      IJCAI.
    - McFadden (1974), Conditional logit analysis of qualitative choice
      behavior.
    - Ziebart et al. (2008), Maximum Entropy Inverse Reinforcement
      Learning, AAAI — para la justificación principia de la forma
      Boltzmann como max-entropy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def subtract_reference(features: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """Resta φ(s, a_ref) de todos los candidatos.

    Es operacionalmente *opcional* para el likelihood (la normalización
    softmax cancela cualquier constante aditiva), pero **es necesaria
    para que la interpretación de w sea "utilidad relativa a la
    referencia"**.

    Args:
        features: shape (T, K, d). T turnos, K candidatos, d features.
        ref_idx: índice del candidato de referencia. Default 0
            (corresponde a `irl.candidates.REFERENCE_CANDIDATE_INDEX`).

    Returns:
        Array (T, K, d). Por construcción, el slice
        `result[:, ref_idx, :]` es exactamente cero.
    """
    if features.ndim != 3:
        raise ValueError(f"features debe ser 3D (T, K, d); tengo shape {features.shape}")
    K = features.shape[1]
    if not 0 <= ref_idx < K:
        raise ValueError(f"ref_idx={ref_idx} fuera de rango [0, {K})")
    return features - features[:, ref_idx : ref_idx + 1, :]


def boltzmann_log_probs(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Log-probabilidades Boltzmann por turno y candidato.

    Args:
        features: shape (T, K, d).
        w: shape (d,).

    Returns:
        log_probs: shape (T, K). Para cada t, exp(log_probs[t]) suma 1.

    Implementación: logsumexp robusto a overflow vía scipy.special.
    """
    if features.ndim != 3:
        raise ValueError(f"features debe ser 3D (T, K, d); tengo shape {features.shape}")
    if w.ndim != 1 or w.shape[0] != features.shape[2]:
        raise ValueError(
            f"w debe ser 1D con shape ({features.shape[2]},); tengo shape {w.shape}"
        )
    utilities = features @ w  # (T, K)
    return utilities - logsumexp(utilities, axis=-1, keepdims=True)


def boltzmann_choice_probs(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Probabilidades Boltzmann (en escala lineal). Shape (T, K)."""
    return np.exp(boltzmann_log_probs(features, w))


def boltzmann_log_likelihood(
    features: np.ndarray,
    chosen: np.ndarray,
    w: np.ndarray,
) -> float:
    """Log-verosimilitud total de las elecciones observadas.

    Args:
        features: shape (T, K, d).
        chosen: shape (T,), índices enteros en [0, K).
        w: shape (d,).

    Returns:
        Σ_t log P(chosen[t] | s_t, w) ∈ (-∞, 0].
    """
    chosen = np.asarray(chosen, dtype=int)
    T_, K = features.shape[0], features.shape[1]
    if chosen.ndim != 1 or chosen.shape[0] != T_:
        raise ValueError(f"chosen debe ser 1D con shape ({T_},); tengo shape {chosen.shape}")
    if (chosen < 0).any() or (chosen >= K).any():
        raise ValueError(
            f"chosen debe estar en [0, {K}); rango observado = "
            f"[{int(chosen.min())}, {int(chosen.max())}]"
        )
    log_probs = boltzmann_log_probs(features, w)  # (T, K)
    return float(log_probs[np.arange(T_), chosen].sum())


def sample_boltzmann_choices(
    features: np.ndarray,
    w: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Muestrea una elección por turno desde la política Boltzmann(w).

    Para validación con sintéticos: generás datos con w* conocido y
    verificás que el modelo bayesiano lo recupera dentro del HDI95.

    Args:
        features: shape (T, K, d).
        w: shape (d,).
        rng: numpy generator (para reproducibilidad).

    Returns:
        chosen: shape (T,), índices en [0, K).
    """
    probs = boltzmann_choice_probs(features, w)
    T_, K = probs.shape
    out = np.empty(T_, dtype=int)
    for t in range(T_):
        out[t] = rng.choice(K, p=probs[t])
    return out
