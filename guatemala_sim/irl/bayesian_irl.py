"""Inferencia bayesiana de los pesos w del IRL Boltzmann via PyMC.

Modelo:

    w_k ~ Normal(0, prior_sigma)        para k = 1, ..., d
    chosen[t] | features[t], w  ~  Categorical(softmax(features[t] @ w))

Implementación con `pm.Potential` + `pt.special.logsumexp` para que la
log-verosimilitud sea estable numéricamente (mismo trick que
`boltzmann.py` usa en NumPy).

Devuelve un `IRLPosterior` con la media posterior, HDI95 por dimensión,
samples completos del posterior, y diagnostics canónicos (R-hat, ESS,
divergencias) — todo lo que un revisor de NeurIPS pide en un análisis
bayesiano serio.

`fit_bayesian_irl_point_estimate` es un wrapper que devuelve solo la
media posterior, compatible con la firma `FitFunction` de
`recovery.py`. Eso permite reutilizar `run_recovery_sweep` con el
método bayesiano sin tocar el harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..bayesian import _hdi, _require_pymc


@dataclass(frozen=True)
class IRLPosterior:
    """Posterior de w del IRL bayesiano."""

    feature_names: tuple[str, ...]
    n_observations: int                # T
    n_candidates: int                  # K
    w_mean: np.ndarray                 # shape (d,)
    w_hdi95: np.ndarray                # shape (d, 2): [lo, hi] por dim
    w_samples: np.ndarray              # shape (n_chains * n_draws, d)
    diverging: int                     # transiciones divergentes (NUTS)
    rhat_max: float                    # R-hat máximo sobre componentes (≤ 1.05 = sano)
    ess_bulk_min: float                # ESS bulk mínima sobre componentes
    prior_sigma: float

    def w_table(self) -> pd.DataFrame:
        """Tabla canónica para el paper: peso ± HDI95 por feature."""
        return pd.DataFrame(
            {
                "feature": list(self.feature_names),
                "mean": self.w_mean,
                "hdi95_lo": self.w_hdi95[:, 0],
                "hdi95_hi": self.w_hdi95[:, 1],
                "hdi95_excludes_zero": (
                    (self.w_hdi95[:, 0] > 0) | (self.w_hdi95[:, 1] < 0)
                ),
            }
        ).set_index("feature")

    @property
    def w_norm_mean(self) -> float:
        """E[‖w‖] sobre el posterior. Análogo a la "concentración" del
        Dirichlet del otro análisis: ‖w‖ alta = preferencias fuertes."""
        return float(np.linalg.norm(self.w_samples, axis=-1).mean())

    @property
    def w_direction_mean(self) -> np.ndarray:
        """Dirección promedio de w (vector unitario), separada de magnitud.
        Útil para comparar "estilos" entre LLMs sin que la rationality los
        confunda."""
        m = self.w_mean
        n = float(np.linalg.norm(m))
        return m / n if n > 1e-12 else m

    def diagnostics_ok(self, rhat_threshold: float = 1.05) -> bool:
        """True si los diagnostics están en niveles aceptables."""
        return self.rhat_max <= rhat_threshold and self.diverging == 0


def fit_bayesian_irl(
    features: np.ndarray,
    chosen: np.ndarray,
    feature_names: tuple[str, ...] | None = None,
    prior_sigma: float = 1.0,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 11,
    progressbar: bool = False,
) -> IRLPosterior:
    """Ajusta el posterior bayesiano de w via NUTS en PyMC.

    Args:
        features: shape (T, K, d). **Recomendado pasarlas ya
            reference-subtracted** con `subtract_reference`. Si no, los
            pesos están definidos hasta una constante aditiva
            irrelevante para la política pero confusa para la
            interpretación.
        chosen: shape (T,), índices enteros en [0, K).
        feature_names: nombres de las d dimensiones (para
            `IRLPosterior.w_table`). Si None se generan como
            `("w_0", ..., "w_{d-1}")`.
        prior_sigma: σ del prior Normal(0, σ) sobre cada w_k. Default
            1.0 — moderadamente informativo. Subir a 5 ó 10 para priors
            más débiles si tenés muchos datos.
        draws, tune, chains: parámetros de NUTS. Defaults razonables
            para producción (4000 muestras totales, ~4000 effective).
        seed: semilla.
        progressbar: si mostrar barra de progreso (False en tests).

    Returns:
        IRLPosterior con todo lo necesario para análisis y reporte.

    Raises:
        RuntimeError si PyMC no está instalado.
        ValueError en inputs malformados.
    """
    _require_pymc()
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    if features.ndim != 3:
        raise ValueError(f"features debe ser 3D (T, K, d); tengo shape {features.shape}")
    T, K, d = features.shape
    chosen = np.asarray(chosen, dtype=int)
    if chosen.shape != (T,):
        raise ValueError(f"chosen debe tener shape ({T},); tengo {chosen.shape}")
    if (chosen < 0).any() or (chosen >= K).any():
        raise ValueError(
            f"chosen debe estar en [0, {K}); rango = [{int(chosen.min())}, {int(chosen.max())}]"
        )

    if feature_names is None:
        feature_names = tuple(f"w_{k}" for k in range(d))
    if len(feature_names) != d:
        raise ValueError(
            f"len(feature_names)={len(feature_names)} no coincide con d={d}"
        )
    if prior_sigma <= 0:
        raise ValueError(f"prior_sigma debe ser > 0; tengo {prior_sigma}")

    features_arr = np.ascontiguousarray(features, dtype=np.float64)
    chosen_arr = np.ascontiguousarray(chosen, dtype=np.int64)

    with pm.Model():
        w = pm.Normal("w", mu=0.0, sigma=prior_sigma, shape=d)
        # utilities[t, k] = features[t, k, :] @ w
        utilities = pt.tensordot(features_arr, w, axes=[[2], [0]])  # (T, K)
        # log-softmax estable
        log_norm = pt.logsumexp(utilities, axis=-1, keepdims=True)
        log_probs = utilities - log_norm  # (T, K)
        # log-verosimilitud: indexo log_probs[t, chosen[t]] y sumo
        log_lik = log_probs[pt.arange(T), chosen_arr]
        pm.Potential("log_lik", log_lik.sum())

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=False,
        )

    w_samples = idata.posterior["w"].values.reshape(-1, d)  # (chains*draws, d)
    w_mean = w_samples.mean(axis=0)
    w_hdi95 = np.array([_hdi(w_samples[:, k]) for k in range(d)])

    diverging = int(idata.sample_stats.get("diverging", np.zeros(1)).values.sum())

    summary = az.summary(idata, var_names=["w"])
    rhat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())

    return IRLPosterior(
        feature_names=tuple(feature_names),
        n_observations=T,
        n_candidates=K,
        w_mean=w_mean,
        w_hdi95=w_hdi95,
        w_samples=w_samples,
        diverging=diverging,
        rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min,
        prior_sigma=prior_sigma,
    )


def fit_bayesian_irl_point_estimate(
    features: np.ndarray,
    chosen: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Wrapper compatible con `recovery.FitFunction`: devuelve solo la
    media posterior como estimador puntual.

    Permite reutilizar `recovery.run_recovery_sweep` con el método
    bayesiano y comparar contra el MLE sin cambiar el harness.
    """
    posterior = fit_bayesian_irl(features, chosen, **kwargs)
    return posterior.w_mean
