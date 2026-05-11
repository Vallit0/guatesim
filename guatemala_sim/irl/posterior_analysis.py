"""Análisis posterior del IRL bayesiano: PPC + comparación entre LLMs.

Cierra dos huecos críticos del paper actual:

  1. **Posterior Predictive Check (PPC)**: el revisor pregunta
     *"¿el modelo Boltzmann ajustado describe bien al LLM, o el LLM no
     es Boltzmann-racional y entonces el IRL es basura?"*. La métrica
     directa es la choice accuracy del modelo ajustado sobre las
     elecciones observadas. Si está apenas por encima del random
     baseline (1/K), el IRL no aprendió y todas las conclusiones
     downstream (audit, harms, consistency) están en jaque para ese LLM.

  2. **Bayesian model comparison**: el paper actual usa Wilcoxon sobre
     `w_mean` como punto, perdiendo toda la incertidumbre del
     posterior. La inferencia natural es
     `P(w_a_k > w_b_k | data_a, data_b)` computada sobre los samples
     directamente, asumiendo independencia entre los dos datasets
     (razonable cuando cada LLM corre en su propio batch). Esto es
     más informativo y no requiere las asunciones del Wilcoxon.

Ambas funciones operan sobre `IRLPosterior.w_samples` ya disponible —
solo no se estaba usando.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .bayesian_irl import IRLPosterior
from .boltzmann import boltzmann_log_probs


# --- HDI local (evitar import circular con bayesian.py) ----------------------


def _hdi(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Highest density interval (HDI) de una colección 1-D de samples."""
    s = np.sort(np.asarray(samples).ravel())
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"))
    interval_size = int(np.ceil(prob * n))
    if interval_size >= n:
        return (float(s[0]), float(s[-1]))
    n_intervals = n - interval_size
    widths = s[interval_size:] - s[:n_intervals]
    j = int(np.argmin(widths))
    return float(s[j]), float(s[j + interval_size])


# --- Posterior Predictive Check ---------------------------------------------


@dataclass(frozen=True)
class PPCResult:
    """PPC del modelo Boltzmann sobre las elecciones observadas.

    `accuracy_argmax_*`: fracción de turnos donde argmax_k Boltzmann(w)
    coincide con la elección observada. Es la métrica más estricta:
    "¿el modelo decide lo MISMO que el LLM?".

    `accuracy_expected_*`: P_Boltzmann(observed_choice | w), promediada
    sobre turnos. Más laxa: penaliza igual una asignación de 0.51 que
    una de 0.99, pero captura el "el modelo ASIGNA probabilidad alta a
    lo que el LLM eligió".

    Ambas métricas se reportan en dos versiones: con `w_mean` (punto)
    y con la distribución entera de samples (`_sample_*` → mean + HDI).

    `random_accuracy = 1/K` es el baseline trivial. La diferencia
    `accuracy_argmax_point − random_accuracy` operacionaliza cuánta
    información extrajo el IRL sobre las elecciones del LLM.
    """

    n_turns: int
    n_candidates: int
    accuracy_argmax_point: float
    accuracy_expected_point: float
    accuracy_argmax_sample_mean: float
    accuracy_argmax_sample_hdi95: tuple[float, float]
    accuracy_expected_sample_mean: float
    accuracy_expected_sample_hdi95: tuple[float, float]
    log_lik_total_mean: float
    log_lik_total_hdi95: tuple[float, float]
    random_accuracy: float

    @property
    def lift_over_random(self) -> float:
        """Diferencia argmax-accuracy menos random baseline. Si ≤ 0,
        el IRL no extrajo nada útil sobre este LLM."""
        return self.accuracy_argmax_point - self.random_accuracy

    def summary_text(self, model_label: str = "el modelo") -> str:
        """Texto quotable para reporte/paper."""
        if self.lift_over_random < 0.05:
            verdict = (
                f"⚠️ NO Boltzmann-coherente: el IRL apenas supera al random "
                f"baseline; el modelo Boltzmann ajustado no describe bien las "
                f"elecciones de {model_label}"
            )
        elif self.accuracy_argmax_point >= 0.7:
            verdict = (
                f"alto fit: el modelo Boltzmann reproduce {self.accuracy_argmax_point:.0%} "
                f"de las elecciones observadas"
            )
        else:
            verdict = (
                f"fit moderado: {self.accuracy_argmax_point:.0%} de las elecciones "
                f"reproducidas, lift {self.lift_over_random:+.2f} sobre random"
            )
        return (
            f"PPC de {model_label}: {verdict}. "
            f"Accuracy argmax = {self.accuracy_argmax_point:.3f} "
            f"(HDI95 sample = [{self.accuracy_argmax_sample_hdi95[0]:.3f}, "
            f"{self.accuracy_argmax_sample_hdi95[1]:.3f}]); "
            f"random baseline = {self.random_accuracy:.3f}; "
            f"log-lik total mean = {self.log_lik_total_mean:.2f}."
        )


def posterior_predictive_check(
    features: np.ndarray,
    chosen: np.ndarray,
    posterior: IRLPosterior,
    n_samples: int | None = 500,
    seed: int = 0,
) -> PPCResult:
    """PPC del modelo Boltzmann ajustado sobre las elecciones observadas.

    Args:
        features: shape (T, K, d). Pasarlas reference-subtracted (igual
            que en `fit_bayesian_irl`).
        chosen: shape (T,), ints en [0, K).
        posterior: salida de `fit_bayesian_irl`.
        n_samples: número de samples del posterior a usar para la
            distribución de accuracy/log-lik. None usa todos. 500 es
            balance entre estabilidad y velocidad.
        seed: para subsampleo reproducible cuando n_samples < total.

    Returns:
        PPCResult con métricas point + sample-based.
    """
    if features.ndim != 3:
        raise ValueError(f"features debe ser 3D; tengo {features.shape}")
    T, K, d = features.shape
    chosen = np.asarray(chosen, dtype=int)
    if chosen.shape != (T,):
        raise ValueError(f"chosen debe ser ({T},); tengo {chosen.shape}")

    # --- point estimate: w_mean ---
    log_probs_pt = boltzmann_log_probs(features, posterior.w_mean)
    probs_pt = np.exp(log_probs_pt)
    pred_pt = log_probs_pt.argmax(axis=-1)
    acc_argmax_pt = float((pred_pt == chosen).mean())
    acc_expected_pt = float(probs_pt[np.arange(T), chosen].mean())

    # --- sample-based ---
    all_samples = posterior.w_samples
    if n_samples is not None and n_samples < len(all_samples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_samples), n_samples, replace=False)
        ws = all_samples[idx]
    else:
        ws = all_samples

    accs_argmax = np.empty(len(ws))
    accs_expected = np.empty(len(ws))
    log_liks = np.empty(len(ws))
    for i, w in enumerate(ws):
        lp = boltzmann_log_probs(features, w)
        pred = lp.argmax(axis=-1)
        accs_argmax[i] = (pred == chosen).mean()
        accs_expected[i] = np.exp(lp[np.arange(T), chosen]).mean()
        log_liks[i] = lp[np.arange(T), chosen].sum()

    return PPCResult(
        n_turns=T,
        n_candidates=K,
        accuracy_argmax_point=acc_argmax_pt,
        accuracy_expected_point=acc_expected_pt,
        accuracy_argmax_sample_mean=float(accs_argmax.mean()),
        accuracy_argmax_sample_hdi95=_hdi(accs_argmax),
        accuracy_expected_sample_mean=float(accs_expected.mean()),
        accuracy_expected_sample_hdi95=_hdi(accs_expected),
        log_lik_total_mean=float(log_liks.mean()),
        log_lik_total_hdi95=_hdi(log_liks),
        random_accuracy=1.0 / K,
    )


# --- Bayesian model comparison ------------------------------------------------


@dataclass(frozen=True)
class PosteriorComparison:
    """Comparación bayesiana directa entre dos posteriors IRL.

    `per_dimension` columnas:
      - feature
      - p_a_gt_b: P(w_a_k > w_b_k | data) ∈ [0, 1]. Cerca de 0.5 = sin
        evidencia direccional; cerca de 0 ó 1 = decisivo.
      - diff_mean: E[w_a_k - w_b_k]
      - diff_hdi95_lo, diff_hdi95_hi
      - decisive: True si p_a_gt_b ≥ decisive_threshold o
        ≤ 1 - decisive_threshold.

    `cosine_posterior_*`: distribución de cosine entre direcciones de
    `w_a` y `w_b`, marginalizado sobre samples. Captura "qué tan
    parecidas son las constituciones reveladas".
    """

    label_a: str
    label_b: str
    feature_names: tuple[str, ...]
    per_dimension: pd.DataFrame
    cosine_posterior_mean: float
    cosine_posterior_hdi95: tuple[float, float]
    p_anti_aligned: float
    n_samples_paired: int

    def summary_text(self) -> str:
        n_dec = int(self.per_dimension["decisive"].sum())
        d = len(self.feature_names)
        return (
            f"Comparación bayesiana {self.label_a} vs {self.label_b}: "
            f"{n_dec}/{d} dimensiones decisivas (P ≥ 0.95 o ≤ 0.05). "
            f"Cosine posterior entre direcciones = "
            f"{self.cosine_posterior_mean:+.3f} "
            f"(HDI95 = [{self.cosine_posterior_hdi95[0]:+.3f}, "
            f"{self.cosine_posterior_hdi95[1]:+.3f}]). "
            f"P(direcciones anti-alineadas) = {self.p_anti_aligned:.3f}."
        )


def compare_posteriors(
    post_a: IRLPosterior,
    post_b: IRLPosterior,
    label_a: str = "A",
    label_b: str = "B",
    decisive_threshold: float = 0.95,
    seed: int = 0,
) -> PosteriorComparison:
    """Computa `P(w_a_k > w_b_k | data)` y cosine posterior.

    Asume independencia entre `data_a` y `data_b` — válido cuando los
    dos LLMs se evaluaron en runs separados con seeds independientes,
    que es nuestro setup multi-seed.

    Para emparejar los samples sin sesgo:
      n = min(len(samples_a), len(samples_b))
      Subsamplea cada uno a n con `seed` para reproducibilidad.
    """
    if post_a.feature_names != post_b.feature_names:
        raise ValueError(
            "feature_names deben coincidir entre los dos posteriors; "
            f"a={post_a.feature_names}, b={post_b.feature_names}"
        )
    sa = post_a.w_samples
    sb = post_b.w_samples
    if sa.shape[1] != sb.shape[1]:
        raise ValueError(
            f"dimensión inconsistente: a={sa.shape}, b={sb.shape}"
        )
    if not (0.5 < decisive_threshold < 1.0):
        raise ValueError(
            f"decisive_threshold debe estar en (0.5, 1.0); tengo {decisive_threshold}"
        )

    n = min(len(sa), len(sb))
    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(sa), n, replace=False)
    idx_b = rng.choice(len(sb), n, replace=False)
    sa_p = sa[idx_a]
    sb_p = sb[idx_b]

    diffs = sa_p - sb_p  # (n, d)
    p_gt = (diffs > 0).mean(axis=0)

    rows: list[dict] = []
    for k, name in enumerate(post_a.feature_names):
        diff_k = diffs[:, k]
        lo, hi = _hdi(diff_k)
        rows.append({
            "feature": name,
            "p_a_gt_b": float(p_gt[k]),
            "diff_mean": float(diff_k.mean()),
            "diff_hdi95_lo": lo,
            "diff_hdi95_hi": hi,
            "decisive": bool(
                (p_gt[k] >= decisive_threshold) or (p_gt[k] <= 1 - decisive_threshold)
            ),
        })
    df = pd.DataFrame(rows).set_index("feature")

    # cosine entre direcciones, posterior-by-posterior
    norm_a = np.linalg.norm(sa_p, axis=1, keepdims=True)
    norm_b = np.linalg.norm(sb_p, axis=1, keepdims=True)
    sa_n = sa_p / np.where(norm_a > 1e-12, norm_a, 1.0)
    sb_n = sb_p / np.where(norm_b > 1e-12, norm_b, 1.0)
    valid = ((norm_a > 1e-12).ravel()) & ((norm_b > 1e-12).ravel())
    cosines = (sa_n * sb_n).sum(axis=1)
    if valid.sum() == 0:
        cos_mean = float("nan")
        cos_hdi = (float("nan"), float("nan"))
        p_anti = float("nan")
    else:
        cosines_valid = cosines[valid]
        cos_mean = float(cosines_valid.mean())
        cos_hdi = _hdi(cosines_valid)
        p_anti = float((cosines_valid < 0).mean())

    return PosteriorComparison(
        label_a=label_a,
        label_b=label_b,
        feature_names=post_a.feature_names,
        per_dimension=df,
        cosine_posterior_mean=cos_mean,
        cosine_posterior_hdi95=cos_hdi,
        p_anti_aligned=p_anti,
        n_samples_paired=n,
    )
