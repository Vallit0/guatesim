"""Análisis bayesiano del comportamiento de los LLMs.

Dos modelos principales, ambos con PyMC:

1. **BEST (Bayesian Estimation Supersedes the t-test, Kruschke 2013)**
   para comparar pareadamente una métrica entre dos modelos. Reemplaza al
   `t`-test con un modelo robusto Student-t que devuelve posterior completo
   de la diferencia de medias, HDI95, tamaño de efecto y probabilidad de
   estar dentro de un ROPE (region of practical equivalence).

2. **Dirichlet-multinomial jerárquico sobre el presupuesto** para
   recuperar la "constitución revelada" de cada LLM como una distribución
   posterior sobre las 9 partidas del presupuesto. Cada turno aporta una
   observación; el posterior de `α_modelo` da la asignación esperada con
   incertidumbre.

PyMC se importa de forma diferida: si no está instalado, las funciones
levantan `RuntimeError` con una guía clara al usuario. Esto permite
mantener PyMC como `extras = [bayes]` opcional sin romper el resto del
paquete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


PRESUPUESTO_PARTIDAS: tuple[str, ...] = (
    "salud",
    "educacion",
    "seguridad",
    "infraestructura",
    "agro_desarrollo_rural",
    "proteccion_social",
    "servicio_deuda",
    "justicia",
    "otros",
)


# --- helpers --------------------------------------------------------------


def _require_pymc():
    try:
        import pymc as pm  # noqa: F401
        import arviz as az  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Análisis bayesiano requiere PyMC. Instalá con "
            "`pip install -e .[bayes]` o `pip install pymc arviz`."
        ) from e


def _hdi(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Highest density interval por método de ventana mínima."""
    s = np.sort(np.asarray(samples).ravel())
    n = len(s)
    if n < 2:
        return (float("nan"), float("nan"))
    k = int(np.floor(prob * n))
    if k < 1:
        return (float(s[0]), float(s[-1]))
    widths = s[k:] - s[: n - k]
    i = int(np.argmin(widths))
    return (float(s[i]), float(s[i + k]))


# --- BEST -----------------------------------------------------------------


@dataclass
class BestResult:
    """Resultado de un BEST pareado (model_b − model_a)."""

    metric: str
    model_a: str
    model_b: str
    n_pairs: int
    posterior_diff_mean: float
    posterior_diff_median: float
    hdi95_lo: float
    hdi95_hi: float
    prob_b_gt_a: float          # P(μ_b > μ_a | datos)
    prob_in_rope: float         # P(|μ_b - μ_a| < rope_width)
    rope_width: float
    effect_size_mean: float     # (μ_b − μ_a) / sd_pooled, posterior mean
    effect_size_hdi95: tuple[float, float]
    nu_mean: float              # grados de libertad (robustez)
    diverging: int

    def to_row(self) -> dict:
        return {
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "diff_mean": self.posterior_diff_mean,
            "diff_median": self.posterior_diff_median,
            "hdi95_lo": self.hdi95_lo,
            "hdi95_hi": self.hdi95_hi,
            "prob_b_gt_a": self.prob_b_gt_a,
            "prob_in_rope": self.prob_in_rope,
            "rope_width": self.rope_width,
            "effect_size": self.effect_size_mean,
            "es_hdi95_lo": self.effect_size_hdi95[0],
            "es_hdi95_hi": self.effect_size_hdi95[1],
            "nu_mean": self.nu_mean,
            "diverging": self.diverging,
        }


def best_paired(
    x_a: np.ndarray,
    x_b: np.ndarray,
    metric: str = "metric",
    model_a: str = "A",
    model_b: str = "B",
    rope_width: float | None = None,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 11,
    progressbar: bool = False,
) -> BestResult:
    """BEST pareado para `x_b − x_a`.

    Modelo (Kruschke 2013):
        diff_i ~ StudentT(ν, μ, σ)
        μ ~ Normal(0, 10 · sd(diff))
        σ ~ HalfNormal(sd(diff) · 5)
        ν ~ Exponential(1/29) + 1   (mean ≈ 30, robusto a outliers)

    Trabajamos directamente sobre las diferencias pareadas (mismo seed →
    mismos shocks); equivale a un BEST de una muestra alrededor de 0.

    `rope_width`: ancho del ROPE simétrico alrededor de 0. Si `None`, se
    usa `0.1 · sd(diff)` (regla práctica).
    """
    _require_pymc()
    import pymc as pm
    import arviz as az

    a = np.asarray(x_a, dtype=float)
    b = np.asarray(x_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"x_a y x_b deben tener la misma forma: {a.shape} vs {b.shape}")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = b - a
    n = len(diff)
    if n < 2:
        raise ValueError(f"Necesito ≥ 2 pares válidos; tengo {n}")

    sd_diff = float(np.std(diff, ddof=1))
    if sd_diff == 0:
        sd_diff = 1.0  # evita HalfNormal(0)
    if rope_width is None:
        rope_width = 0.1 * sd_diff

    with pm.Model():
        mu = pm.Normal("mu", mu=0.0, sigma=10.0 * sd_diff)
        sigma = pm.HalfNormal("sigma", sigma=5.0 * sd_diff)
        nu_minus_one = pm.Exponential("nu_m1", lam=1.0 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1.0)
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=diff)
        # tamaño de efecto estandarizado (Cohen's d bayesiano)
        pm.Deterministic("effect", mu / sigma)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=seed, progressbar=progressbar,
            compute_convergence_checks=False,
        )

    mu_post = idata.posterior["mu"].values.ravel()
    eff_post = idata.posterior["effect"].values.ravel()
    nu_post = idata.posterior["nu"].values.ravel()
    hdi_lo, hdi_hi = _hdi(mu_post, 0.95)
    eff_lo, eff_hi = _hdi(eff_post, 0.95)
    prob_gt = float((mu_post > 0).mean())
    prob_rope = float((np.abs(mu_post) < rope_width).mean())
    diverging = int(idata.sample_stats.get("diverging", np.zeros(1)).values.sum())

    return BestResult(
        metric=metric,
        model_a=model_a,
        model_b=model_b,
        n_pairs=n,
        posterior_diff_mean=float(np.mean(mu_post)),
        posterior_diff_median=float(np.median(mu_post)),
        hdi95_lo=hdi_lo,
        hdi95_hi=hdi_hi,
        prob_b_gt_a=prob_gt,
        prob_in_rope=prob_rope,
        rope_width=float(rope_width),
        effect_size_mean=float(np.mean(eff_post)),
        effect_size_hdi95=(eff_lo, eff_hi),
        nu_mean=float(np.mean(nu_post)),
        diverging=diverging,
    )


def best_paired_table(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] | None = None,
    rope_widths: dict[str, float] | None = None,
    **best_kwargs,
) -> pd.DataFrame:
    """BEST sobre múltiples métricas. `df` con índice (seed, modelo) o
    (seed, replica, modelo); en el segundo caso, colapsamos por seed.

    Devuelve DataFrame indexado por métrica con las columnas de
    `BestResult.to_row()`.
    """
    from .multiseed import collapse_replicas

    if "replica" in df.index.names:
        df = collapse_replicas(df)

    a = df.xs(model_a, level="modelo")
    b = df.xs(model_b, level="modelo")
    seeds = a.index.intersection(b.index)
    if len(seeds) < 2:
        return pd.DataFrame()

    a = a.loc[seeds].select_dtypes(include=[np.number])
    b = b.loc[seeds].select_dtypes(include=[np.number])
    if metrics is None:
        metrics = list(a.columns)
    rope_widths = rope_widths or {}

    rows: list[dict] = []
    for m in metrics:
        if m not in a.columns or m not in b.columns:
            continue
        va = a[m].to_numpy(dtype=float)
        vb = b[m].to_numpy(dtype=float)
        try:
            r = best_paired(
                va, vb, metric=m, model_a=model_a, model_b=model_b,
                rope_width=rope_widths.get(m), **best_kwargs,
            )
            rows.append(r.to_row())
        except Exception as e:
            rows.append({
                "metric": m, "n_pairs": 0,
                "diff_mean": float("nan"), "diff_median": float("nan"),
                "hdi95_lo": float("nan"), "hdi95_hi": float("nan"),
                "prob_b_gt_a": float("nan"), "prob_in_rope": float("nan"),
                "rope_width": float("nan"),
                "effect_size": float("nan"),
                "es_hdi95_lo": float("nan"), "es_hdi95_hi": float("nan"),
                "nu_mean": float("nan"), "diverging": 0,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("metric")
    return out


# --- Dirichlet-multinomial sobre presupuesto ------------------------------


@dataclass
class BudgetPosterior:
    """Posterior del Dirichlet-multinomial jerárquico sobre el presupuesto
    de un LLM (o de cada LLM en una comparación)."""

    model_label: str
    n_obs: int
    partidas: tuple[str, ...]
    # E[p_k | data]: posterior mean de la asignación esperada
    expected_share: np.ndarray
    expected_share_hdi95: np.ndarray  # shape (K, 2)
    # α_k posterior mean (parámetros de concentración)
    alpha_mean: np.ndarray
    alpha_hdi95: np.ndarray  # shape (K, 2)
    # concentración total (suma de α): mayor = más "rígido", menor = más volátil
    concentration_mean: float
    concentration_hdi95: tuple[float, float]
    diverging: int

    def expected_share_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "partida": list(self.partidas),
            "E[share] (%)": self.expected_share * 100,
            "hdi95_lo (%)": self.expected_share_hdi95[:, 0] * 100,
            "hdi95_hi (%)": self.expected_share_hdi95[:, 1] * 100,
            "alpha_mean": self.alpha_mean,
        }).set_index("partida")


def fit_budget_dirichlet(
    observations: np.ndarray,
    model_label: str = "model",
    partidas: Sequence[str] = PRESUPUESTO_PARTIDAS,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 11,
    progressbar: bool = False,
) -> BudgetPosterior:
    """Ajusta un Dirichlet-multinomial al presupuesto observado de un LLM.

    Modelo:
        α_k ~ HalfNormal(σ=10),     k = 1, ..., K=9
        p_t ~ Dirichlet(α),          t = 1, ..., T   (un budget por turno)

    `observations` es un array `(T, K)` con las fracciones del presupuesto
    en [0, 1] que suman 1 en cada fila. (Si tus datos están en %, dividí
    por 100 antes de pasarlos.) Para evitar `log(0)` cuando algún `p_k = 0`,
    aplicamos un suavizado de Laplace ε = 1e-3 y renormalizamos.

    El posterior de `α_k / Σ α` es la "constitución revelada" del LLM con
    incertidumbre cuantificada — qué partidas prioriza y cuánto ruido hay
    en sus elecciones.
    """
    _require_pymc()
    import pymc as pm

    obs = np.asarray(observations, dtype=float)
    if obs.ndim != 2:
        raise ValueError(f"observations debe ser 2D (T, K); tengo {obs.shape}")
    T, K = obs.shape
    if K != len(partidas):
        raise ValueError(f"K={K} no coincide con len(partidas)={len(partidas)}")
    if T < 2:
        raise ValueError(f"Necesito ≥ 2 observaciones; tengo {T}")

    # Suavizado para evitar p_k == 0 (Dirichlet log-pdf diverge)
    eps = 1e-3
    obs = obs + eps
    obs = obs / obs.sum(axis=1, keepdims=True)

    with pm.Model():
        alpha = pm.HalfNormal("alpha", sigma=10.0, shape=K)
        pm.Dirichlet("p_obs", a=alpha, observed=obs)
        # cantidades derivadas
        concentration = pm.Deterministic("concentration", alpha.sum())
        pm.Deterministic("expected_share", alpha / concentration)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=seed, progressbar=progressbar,
            compute_convergence_checks=False,
        )

    alpha_post = idata.posterior["alpha"].values.reshape(-1, K)
    share_post = idata.posterior["expected_share"].values.reshape(-1, K)
    conc_post = idata.posterior["concentration"].values.ravel()

    alpha_mean = alpha_post.mean(axis=0)
    share_mean = share_post.mean(axis=0)
    alpha_hdi = np.array([_hdi(alpha_post[:, k]) for k in range(K)])
    share_hdi = np.array([_hdi(share_post[:, k]) for k in range(K)])
    conc_hdi = _hdi(conc_post)
    diverging = int(idata.sample_stats.get("diverging", np.zeros(1)).values.sum())

    return BudgetPosterior(
        model_label=model_label,
        n_obs=T,
        partidas=tuple(partidas),
        expected_share=share_mean,
        expected_share_hdi95=share_hdi,
        alpha_mean=alpha_mean,
        alpha_hdi95=alpha_hdi,
        concentration_mean=float(conc_post.mean()),
        concentration_hdi95=conc_hdi,
        diverging=diverging,
    )


def compare_budget_constitutions(
    df_long: pd.DataFrame,
    models: Sequence[str],
    partidas: Sequence[str] = PRESUPUESTO_PARTIDAS,
    **fit_kwargs,
) -> dict[str, BudgetPosterior]:
    """Ajusta un Dirichlet-multinomial separado por modelo y devuelve los
    posteriors lado a lado.

    `df_long` es el output de `multiseed.collect_turn_metrics`: una fila
    por (seed, replica, modelo, t) con columnas `presup_<partida>` que
    suman ~100. Acá las renormalizamos a [0, 1].
    """
    cols = [f"presup_{p}" for p in partidas]
    missing = [c for c in cols if c not in df_long.columns]
    if missing:
        raise ValueError(f"faltan columnas en df_long: {missing}")

    out: dict[str, BudgetPosterior] = {}
    for m in models:
        sub = df_long[df_long["modelo"] == m]
        if sub.empty:
            continue
        obs = sub[cols].to_numpy(dtype=float)
        # `tabla_comparativa` reporta en %, dividimos a [0, 1]
        if np.nanmax(obs) > 1.5:
            obs = obs / 100.0
        out[m] = fit_budget_dirichlet(
            obs, model_label=m, partidas=partidas, **fit_kwargs,
        )
    return out


def constitutions_to_dataframe(
    posteriors: dict[str, BudgetPosterior],
) -> pd.DataFrame:
    """Tabla comparativa: una fila por partida × modelo con E[share] y HDI95."""
    rows: list[dict] = []
    for label, post in posteriors.items():
        for k, partida in enumerate(post.partidas):
            rows.append({
                "modelo": label,
                "partida": partida,
                "E[share]": post.expected_share[k],
                "hdi95_lo": post.expected_share_hdi95[k, 0],
                "hdi95_hi": post.expected_share_hdi95[k, 1],
                "alpha": post.alpha_mean[k],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["modelo", "partida"])
