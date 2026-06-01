"""Validación del IRL bayesiano con datos sintéticos de ground-truth conocido.

El experimento canónico de validación de cualquier método de IRL/IRD:

    1. Fijás un w* verdadero conocido.
    2. Generás un dataset sintético muestreando elecciones desde
       Boltzmann(w*).
    3. Ajustás el modelo y obtenés ŵ.
    4. Verificás que ŵ recupera w* dentro de error esperado.
    5. Repetís sobre N ∈ {50, 100, ..., 5000} para caracterizar la
       curva error vs. tamaño de muestra → es la Figura 1 del paper.

Sin esta validación, ningún revisor de NeurIPS cree que el método
funciona sobre datos reales (donde no hay ground truth).

Este módulo provee:
  - `generate_synthetic_dataset`: features random gaussian + sampling
    Boltzmann. Independiente del simulador para aislar la inferencia.
  - `fit_mle_boltzmann`: estimador puntual MLE vía scipy.optimize.
    Sirve como baseline frecuentista contra el cual comparar el posterior
    bayesiano de `bayesian_irl.py`.
  - `compute_recovery_metrics`: RMSE, cosine similarity, ratio de normas
    y (si hay posterior) cobertura del HDI95.
  - `run_mle_recovery_sweep`: barrido sobre N para la curva del paper.

Diseño: la función de ajuste es un parámetro genérico, así que cuando
`bayesian_irl.fit_bayesian_irl` esté lista, se podrá reusar la misma
infraestructura sin tocar este módulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .boltzmann import boltzmann_log_likelihood, sample_boltzmann_choices, subtract_reference


# --- generación de datos sintéticos ------------------------------------------


@dataclass(frozen=True)
class RecoveryDataset:
    """Dataset sintético para validación de recovery.

    Campos:
        features: shape (T, K, d), ya con `subtract_reference` aplicado
            (ref_idx=0). features[:, 0, :] es exactamente cero.
        chosen: shape (T,), índices en [0, K) muestreados desde
            Boltzmann(true_w).
        true_w: shape (d,), los pesos verdaderos.
        feature_seed, choice_seed: semillas usadas, para reproducibilidad.
    """

    features: np.ndarray
    chosen: np.ndarray
    true_w: np.ndarray
    feature_seed: int
    choice_seed: int

    @property
    def n_turns(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[2])


def generate_synthetic_dataset(
    true_w: np.ndarray,
    n_turns: int = 200,
    n_candidates: int = 5,
    feature_seed: int = 0,
    choice_seed: int = 1,
    feature_scale: float = 1.0,
) -> RecoveryDataset:
    """Genera features random gaussian + elecciones desde Boltzmann(w*).

    Las features son random gaussian (no del simulador) intencionalmente:
    para validar el método de inferencia queremos aislar errores del
    estimador de errores del simulador. La validación end-to-end con
    features reales es un experimento separado.

    Args:
        true_w: shape (d,), los pesos verdaderos.
        n_turns: número de observaciones (T).
        n_candidates: tamaño del menú (K).
        feature_seed: semilla para generar las features.
        choice_seed: semilla para muestrear las elecciones.
        feature_scale: desvío estándar de las features gaussianas.
            Default 1.0; valores más altos hacen el problema "más fácil"
            (mayor separación entre candidatos).

    Returns:
        RecoveryDataset con features ya ancladas en candidato 0.
    """
    true_w = np.asarray(true_w, dtype=float)
    if true_w.ndim != 1:
        raise ValueError(f"true_w debe ser 1D; tengo shape {true_w.shape}")
    if n_turns < 1 or n_candidates < 2:
        raise ValueError(
            f"requiero n_turns ≥ 1 y n_candidates ≥ 2; tengo {n_turns}, {n_candidates}"
        )

    d = true_w.shape[0]
    rng_features = np.random.default_rng(feature_seed)
    features_raw = rng_features.standard_normal((n_turns, n_candidates, d)) * feature_scale
    features = subtract_reference(features_raw, ref_idx=0)

    rng_choice = np.random.default_rng(choice_seed)
    chosen = sample_boltzmann_choices(features, true_w, rng_choice)

    return RecoveryDataset(
        features=features,
        chosen=chosen,
        true_w=true_w.copy(),
        feature_seed=feature_seed,
        choice_seed=choice_seed,
    )


# --- estimación MLE puntual --------------------------------------------------


def fit_mle_boltzmann(
    features: np.ndarray,
    chosen: np.ndarray,
    initial_w: np.ndarray | None = None,
    l2_reg: float = 0.0,
    max_iter: int = 500,
) -> np.ndarray:
    """MLE de w por L-BFGS-B sobre la log-verosimilitud Boltzmann.

    Minimiza `-log L(w) + λ ‖w‖²`. Sirve como baseline frecuentista
    contra el cual validar el posterior bayesiano.

    Args:
        features: shape (T, K, d). Recomendado pasarlas ya
            reference-subtracted; si no, el resultado es igual hasta una
            constante aditiva en utilidad (irrelevante para la política).
        chosen: shape (T,), índices observados.
        initial_w: punto inicial. Default w=0 (siempre es razonable
            porque la log-likelihood en w=0 es -T log K, finita).
        l2_reg: coeficiente λ ≥ 0 de regularización L2. Default 0.
            Útil si los datos son pocos y la verosimilitud es plana.
        max_iter: máximo de iteraciones de L-BFGS-B.

    Returns:
        ŵ: shape (d,), el estimador MLE (o MAP con prior Gaussian si
        l2_reg > 0).

    Raises:
        RuntimeError si el optimizador no converge.
    """
    d = features.shape[2]
    w0 = np.zeros(d) if initial_w is None else np.asarray(initial_w, dtype=float)
    if w0.shape != (d,):
        raise ValueError(f"initial_w debe tener shape ({d},); tengo {w0.shape}")

    def neg_log_posterior(w: np.ndarray) -> float:
        ll = boltzmann_log_likelihood(features, chosen, w)
        if l2_reg > 0:
            ll = ll - l2_reg * float(np.dot(w, w))
        return -ll

    result = minimize(
        neg_log_posterior,
        w0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(
            f"MLE no convergió: {result.message}. nit={result.nit}, "
            f"final neg_log_lik={result.fun:.4f}"
        )
    return np.asarray(result.x, dtype=float)


# --- métricas de recovery ----------------------------------------------------


@dataclass(frozen=True)
class RecoveryMetrics:
    """Métricas comparando ŵ contra w*.

    Campos siempre presentes:
        rmse: ‖ŵ - w*‖₂ / √d  — error promedio por componente.
        cosine_similarity: ŵ·w* / (‖ŵ‖ ‖w*‖) ∈ [-1, 1]. 1 = misma
            dirección de preferencia.
        angle_degrees: arccos(cos_sim) en grados ∈ [0, 180].
        norm_ratio: ‖ŵ‖ / ‖w*‖. 1 = misma "concentración" o
            rationality. >1 = recovered es más "decisivo" que el real.

    Campos opcionales (sólo si se pasa el HDI95 del posterior):
        coverage_hdi95: fracción de dimensiones donde w*[k] ∈ HDI95[k].
            Cobertura nominal = 0.95.
        coverage_per_dim: array (d,) booleano por dimensión.
    """

    true_w: np.ndarray
    estimated_w: np.ndarray
    rmse: float
    cosine_similarity: float
    angle_degrees: float
    norm_ratio: float
    coverage_hdi95: float | None = None
    coverage_per_dim: np.ndarray | None = None


def compute_recovery_metrics(
    true_w: np.ndarray,
    estimated_w: np.ndarray,
    estimated_w_hdi95: np.ndarray | None = None,
) -> RecoveryMetrics:
    """Compara ŵ contra w*.

    Args:
        true_w: shape (d,).
        estimated_w: shape (d,).
        estimated_w_hdi95: shape (d, 2) opcional. Columna 0 = lo, columna
            1 = hi del HDI95 por componente.

    Returns:
        RecoveryMetrics.
    """
    true_w = np.asarray(true_w, dtype=float)
    estimated_w = np.asarray(estimated_w, dtype=float)
    if true_w.shape != estimated_w.shape:
        raise ValueError(
            f"shape mismatch: true_w {true_w.shape} vs estimated_w {estimated_w.shape}"
        )
    d = true_w.shape[0]

    diff = estimated_w - true_w
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    norm_t = float(np.linalg.norm(true_w))
    norm_e = float(np.linalg.norm(estimated_w))
    if norm_t < 1e-12 or norm_e < 1e-12:
        cos_sim = float("nan")
        angle = float("nan")
    else:
        cos_sim = float(np.dot(true_w, estimated_w) / (norm_t * norm_e))
        cos_sim = max(-1.0, min(1.0, cos_sim))  # clamp por estabilidad numérica
        angle = float(np.degrees(np.arccos(cos_sim)))
    norm_ratio = float(norm_e / norm_t) if norm_t > 1e-12 else float("nan")

    coverage_hdi95: float | None = None
    coverage_per_dim: np.ndarray | None = None
    if estimated_w_hdi95 is not None:
        hdi = np.asarray(estimated_w_hdi95, dtype=float)
        if hdi.shape != (d, 2):
            raise ValueError(f"hdi95 debe tener shape ({d}, 2); tengo {hdi.shape}")
        coverage_per_dim = (true_w >= hdi[:, 0]) & (true_w <= hdi[:, 1])
        coverage_hdi95 = float(coverage_per_dim.mean())

    return RecoveryMetrics(
        true_w=true_w.copy(),
        estimated_w=estimated_w.copy(),
        rmse=rmse,
        cosine_similarity=cos_sim,
        angle_degrees=angle,
        norm_ratio=norm_ratio,
        coverage_hdi95=coverage_hdi95,
        coverage_per_dim=coverage_per_dim,
    )


# --- barrido sobre N (la Figura 1 del paper) ---------------------------------


FitFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Tipo: una función de ajuste toma (features, chosen) y devuelve ŵ."""


def run_recovery_sweep(
    true_w: np.ndarray,
    sample_sizes: list[int],
    fit_fn: FitFunction = fit_mle_boltzmann,
    n_replications: int = 5,
    n_candidates: int = 5,
    feature_scale: float = 1.0,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Barrido sobre N para caracterizar la curva error de recovery vs N.

    Para cada N en sample_sizes, corre n_replications réplicas
    independientes (distintos feature_seed y choice_seed) y mide RMSE,
    cosine similarity, etc. El output es una tabla larga lista para
    plotting.

    Args:
        true_w: pesos verdaderos.
        sample_sizes: lista de N (turnos) a barrer.
        fit_fn: función (features, chosen) → ŵ. Default MLE.
            Pasale `bayesian_irl.fit_bayesian_irl_point_estimate`
            cuando esté lista para comparar.
        n_replications: réplicas independientes por N (con seeds
            distintos).
        n_candidates: tamaño del menú.
        feature_scale: escala de las features sintéticas.
        base_seed: semilla raíz; las réplicas usan offsets reproducibles.

    Returns:
        DataFrame con columnas: N, replication, rmse, cosine_similarity,
            angle_degrees, norm_ratio, log_likelihood_at_w_hat.
    """
    rows: list[dict] = []
    for N in sample_sizes:
        for rep in range(n_replications):
            ds = generate_synthetic_dataset(
                true_w=true_w,
                n_turns=N,
                n_candidates=n_candidates,
                feature_seed=base_seed + 1_000 * rep + N,
                choice_seed=base_seed + 1_000_003 * rep + N,
                feature_scale=feature_scale,
            )
            w_hat = fit_fn(ds.features, ds.chosen)
            metrics = compute_recovery_metrics(true_w, w_hat)
            ll = boltzmann_log_likelihood(ds.features, ds.chosen, w_hat)
            rows.append(
                {
                    "N": N,
                    "replication": rep,
                    "rmse": metrics.rmse,
                    "cosine_similarity": metrics.cosine_similarity,
                    "angle_degrees": metrics.angle_degrees,
                    "norm_ratio": metrics.norm_ratio,
                    "log_likelihood_at_w_hat": ll,
                }
            )
    return pd.DataFrame(rows)
