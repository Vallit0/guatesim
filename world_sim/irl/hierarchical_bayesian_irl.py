"""IRL bayesiano jerárquico — μ_LLM (constitución central) y τ_LLM (varianza cross-seed).

**Por qué jerárquico** (la pieza que mata la crítica más fuerte al paper):

El paper actual ajusta un posterior `w` independiente por (LLM, seed) y
después agrega via Wilcoxon. Esto tiene dos problemas:

  1. **Pierde poder estadístico**: cada seed solo tiene T=8 turnos, y
     muchas dimensiones quedan sin identificar (var ratio ≈ 1) según
     el módulo `posterior_analysis.identifiability_table`. Pooling
     cross-seed bajo un modelo jerárquico sí identifica esas dims
     porque comparte información entre seeds.

  2. **No modela el LLM como function-of-prompt**: tratar `w_LLM` como
     un punto fijo asume que el LLM tiene preferencias estables. No
     las tiene — son condicional al prompt. El modelo jerárquico lo
     captura formalmente:

         μ_LLM = constitución central (preferencia esperada del LLM)
         τ_LLM = varianza cross-seed (sensibilidad a contexto/prompt)

     Cada seed es una observación de `w_seed ~ Normal(μ_LLM, τ_LLM)`.
     `τ_LLM` chico ⇒ LLM con preferencias estables; `τ_LLM` grande ⇒
     LLM cuya `w` revelada depende fuertemente del prompt/seed.

**Modelo**:

    μ_k ~ Normal(0, σ_μ)           para k = 1..d
    τ_k ~ HalfNormal(σ_τ)          para k = 1..d
    z_{s,k} ~ Normal(0, 1)         (parameterización non-centered)
    w_{s,k} = μ_k + τ_k · z_{s,k}
    chosen_{s,t} ~ Categorical(softmax(φ_{s,t} · w_s))

**Parameterización non-centered** (Neal 2003, Betancourt & Girolami
2015): la versión "centered" `w_s ~ Normal(μ, τ)` produce la "funnel
pathology" — geometría con curvatura altísima que NUTS no puede
muestrear cuando τ → 0. Reescribir como `w_s = μ + τ·z` con `z ~ N(0,1)`
elimina la dependencia funcional entre samples y permite a NUTS
mover libremente.

**Referencias**:
- Chen et al. 2020. *Active Learning for Hierarchical Inverse RL*.
- Choi & Kim 2011. *Map Inference for Bayesian Inverse Reinforcement
  Learning*.
- Betancourt & Girolami 2015. *Hamiltonian Monte Carlo for Hierarchical
  Models*. arXiv:1312.0906.
- Neal 2003. *Slice Sampling*, Annals of Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..bayesian import _hdi, _require_pymc


@dataclass(frozen=True)
class HierarchicalIRLPosterior:
    """Posterior del IRL bayesiano jerárquico.

    Attributes:
        feature_names: nombres de las d features (igual orden que el IRL flat).
        n_seeds: S (número de seeds = grupos jerárquicos).
        n_turns_per_seed: lista (S,) con T_s.
        n_candidates: K (mismo entre seeds por contrato del simulador).
        mu_samples: shape (n_chains*n_draws, d). Constitución central
            posterior del LLM.
        tau_samples: shape (n_chains*n_draws, d). Varianza cross-seed
            posterior por dimensión.
        w_samples_per_seed: shape (n_chains*n_draws, S, d). Posteriors
            individuales por seed (después de re-parameterizar).
        mu_mean, mu_hdi95: estadísticas de μ.
        tau_mean, tau_hdi95: estadísticas de τ.
        diverging, rhat_max, ess_bulk_min: diagnósticos NUTS.
        prior_sigma_mu, prior_sigma_tau: hyperparámetros del prior.
    """

    feature_names: tuple[str, ...]
    n_seeds: int
    n_turns_per_seed: tuple[int, ...]
    n_candidates: int
    mu_samples: np.ndarray
    tau_samples: np.ndarray
    w_samples_per_seed: np.ndarray
    mu_mean: np.ndarray
    mu_hdi95: np.ndarray
    tau_mean: np.ndarray
    tau_hdi95: np.ndarray
    diverging: int
    rhat_max: float
    ess_bulk_min: float
    prior_sigma_mu: float
    prior_sigma_tau: float

    @property
    def d(self) -> int:
        return len(self.feature_names)

    def mu_table(self) -> pd.DataFrame:
        """Tabla canónica de μ: constitución central por feature."""
        return pd.DataFrame({
            "feature": list(self.feature_names),
            "mu_mean": self.mu_mean,
            "mu_hdi95_lo": self.mu_hdi95[:, 0],
            "mu_hdi95_hi": self.mu_hdi95[:, 1],
            "mu_hdi95_excludes_zero": (
                (self.mu_hdi95[:, 0] > 0) | (self.mu_hdi95[:, 1] < 0)
            ),
        }).set_index("feature")

    def tau_table(self) -> pd.DataFrame:
        """Tabla de τ: cuánta varianza induce el contexto/prompt por
        dimensión. τ alto ⇒ esa dim es sensible al prompt; τ bajo ⇒
        estable cross-seed (preferencia "core" del LLM)."""
        return pd.DataFrame({
            "feature": list(self.feature_names),
            "tau_mean": self.tau_mean,
            "tau_hdi95_lo": self.tau_hdi95[:, 0],
            "tau_hdi95_hi": self.tau_hdi95[:, 1],
        }).set_index("feature")

    def pooling_factor(self) -> np.ndarray:
        """Factor de pooling jerárquico por dimensión: 1 - τ²/(σ²_w)
        donde σ²_w es la varianza marginal de w_s, ≈ τ² + var(μ_post).

        Cerca de 1 = pooling completo (los seeds son ~iguales en esa
        dim, μ informativo); cerca de 0 = no pooling (seeds varían
        mucho, μ poco informativo).

        Estimador estable: usa la media de τ² posterior y compara con
        var(w_samples_per_seed). NaN-safe.
        """
        tau_sq_mean = (self.tau_samples ** 2).mean(axis=0)
        # var(w_s) marginal sobre seeds y samples
        ws = self.w_samples_per_seed.reshape(-1, self.d)
        marginal_var = ws.var(axis=0)
        denom = np.where(marginal_var > 1e-12, marginal_var, 1.0)
        pf = 1.0 - tau_sq_mean / denom
        return np.clip(pf, 0.0, 1.0)

    def diagnostics_ok(self, rhat_threshold: float = 1.05) -> bool:
        return self.rhat_max <= rhat_threshold and self.diverging == 0


def fit_hierarchical_bayesian_irl(
    features_per_seed: Sequence[np.ndarray],
    chosen_per_seed: Sequence[np.ndarray],
    feature_names: tuple[str, ...] | None = None,
    prior_sigma_mu: float = 1.0,
    prior_sigma_tau: float = 0.5,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 11,
    progressbar: bool = False,
    target_accept: float = 0.95,
) -> HierarchicalIRLPosterior:
    """Ajusta el modelo jerárquico via NUTS.

    Args:
        features_per_seed: lista de S arrays, cada uno (T_s, K, d).
            **Pasarlos ya reference-subtracted** con `subtract_reference`.
        chosen_per_seed: lista de S arrays, cada uno (T_s,) con ints en [0, K).
        feature_names: nombres de las d dimensiones.
        prior_sigma_mu: σ del prior `Normal(0, σ_μ)` sobre la constitución
            central μ. Default 1.0.
        prior_sigma_tau: σ del prior `HalfNormal(σ_τ)` sobre la varianza
            cross-seed τ. Default 0.5 — moderadamente informativo a
            favor de pooling. Subir a 1.0+ para permitir más
            heterogeneidad entre seeds.
        draws, tune, chains: parámetros NUTS. target_accept=0.95
            (más alto que default) porque jerárquico es más sensible.

    Returns:
        HierarchicalIRLPosterior con μ, τ, w_per_seed, diagnostics.

    Raises:
        RuntimeError si PyMC no está instalado.
        ValueError en inputs malformados.
    """
    _require_pymc()
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    # --- validación de inputs ---
    if len(features_per_seed) != len(chosen_per_seed):
        raise ValueError(
            f"features_per_seed (S={len(features_per_seed)}) y "
            f"chosen_per_seed (S={len(chosen_per_seed)}) deben tener mismo S"
        )
    S = len(features_per_seed)
    if S == 0:
        raise ValueError("se requiere al menos 1 seed")

    feats_validated: list[np.ndarray] = []
    chosen_validated: list[np.ndarray] = []
    K = None
    d = None
    for s, (f_s, c_s) in enumerate(zip(features_per_seed, chosen_per_seed)):
        f_s = np.asarray(f_s, dtype=np.float64)
        c_s = np.asarray(c_s, dtype=np.int64)
        if f_s.ndim != 3:
            raise ValueError(
                f"features_per_seed[{s}] debe ser 3D (T, K, d); tengo {f_s.shape}"
            )
        T_s, K_s, d_s = f_s.shape
        if K is None:
            K, d = K_s, d_s
        else:
            if (K_s, d_s) != (K, d):
                raise ValueError(
                    f"shape inconsistente entre seeds: seed 0 = (·, {K}, {d}), "
                    f"seed {s} = (·, {K_s}, {d_s})"
                )
        if c_s.shape != (T_s,):
            raise ValueError(
                f"chosen_per_seed[{s}] debe ser ({T_s},); tengo {c_s.shape}"
            )
        if (c_s < 0).any() or (c_s >= K).any():
            raise ValueError(
                f"chosen_per_seed[{s}] fuera de [0, {K}): "
                f"rango [{int(c_s.min())}, {int(c_s.max())}]"
            )
        feats_validated.append(np.ascontiguousarray(f_s))
        chosen_validated.append(np.ascontiguousarray(c_s))

    if feature_names is None:
        feature_names = tuple(f"w_{k}" for k in range(d))
    if len(feature_names) != d:
        raise ValueError(
            f"len(feature_names)={len(feature_names)} no coincide con d={d}"
        )
    if prior_sigma_mu <= 0 or prior_sigma_tau <= 0:
        raise ValueError("prior_sigma_mu y prior_sigma_tau deben ser > 0")

    # --- aplanado para una sola operación tensorial ---
    n_turns_per_seed = tuple(int(f.shape[0]) for f in feats_validated)
    T_total = sum(n_turns_per_seed)
    features_all = np.concatenate(feats_validated, axis=0)   # (T_total, K, d)
    chosen_all = np.concatenate(chosen_validated, axis=0)    # (T_total,)
    seed_idx = np.repeat(np.arange(S, dtype=np.int64), n_turns_per_seed)  # (T_total,)

    # --- modelo PyMC ---
    with pm.Model():
        mu = pm.Normal("mu", mu=0.0, sigma=prior_sigma_mu, shape=d)
        tau = pm.HalfNormal("tau", sigma=prior_sigma_tau, shape=d)
        # Non-centered: z ~ N(0,1), w_s = mu + tau * z_s
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(S, d))
        w = pm.Deterministic("w", mu[None, :] + tau[None, :] * z)  # (S, d)

        # utilities[t, k] = features_all[t, k, :] · w[seed_idx[t], :]
        # einsum tkd, td -> tk
        w_per_obs = w[seed_idx]  # (T_total, d)
        utilities = pt.batched_dot(features_all, w_per_obs[:, :, None])[:, :, 0]
        # equivalente a: pt.einsum("tkd,td->tk", features_all, w_per_obs)
        # pero batched_dot es más portable cross-versión

        log_norm = pt.logsumexp(utilities, axis=-1, keepdims=True)
        log_probs = utilities - log_norm  # (T_total, K)
        log_lik = log_probs[pt.arange(T_total), chosen_all]
        pm.Potential("log_lik", log_lik.sum())

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=progressbar,
            target_accept=target_accept,
            compute_convergence_checks=False,
        )

    # --- extracción de samples ---
    mu_samples = idata.posterior["mu"].values.reshape(-1, d)
    tau_samples = idata.posterior["tau"].values.reshape(-1, d)
    w_samples = idata.posterior["w"].values  # (chains, draws, S, d)
    n_total = w_samples.shape[0] * w_samples.shape[1]
    w_samples = w_samples.reshape(n_total, S, d)

    mu_mean = mu_samples.mean(axis=0)
    mu_hdi = np.array([_hdi(mu_samples[:, k]) for k in range(d)])
    tau_mean = tau_samples.mean(axis=0)
    tau_hdi = np.array([_hdi(tau_samples[:, k]) for k in range(d)])

    diverging = int(idata.sample_stats.get("diverging", np.zeros(1)).values.sum())
    summary = az.summary(idata, var_names=["mu", "tau"])
    rhat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())

    return HierarchicalIRLPosterior(
        feature_names=tuple(feature_names),
        n_seeds=S,
        n_turns_per_seed=n_turns_per_seed,
        n_candidates=K,
        mu_samples=mu_samples,
        tau_samples=tau_samples,
        w_samples_per_seed=w_samples,
        mu_mean=mu_mean,
        mu_hdi95=mu_hdi,
        tau_mean=tau_mean,
        tau_hdi95=tau_hdi,
        diverging=diverging,
        rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min,
        prior_sigma_mu=prior_sigma_mu,
        prior_sigma_tau=prior_sigma_tau,
    )


# --- Comparación bayesiana entre constituciones de LLMs ---------------------


@dataclass(frozen=True)
class HierarchicalComparison:
    """Comparación bayesiana de dos LLMs ajustados con jerárquico.

    Tres niveles de comparación:
      - **constituciones** (μ): preferencia central. Reemplaza el
        Wilcoxon agregado del paper actual.
      - **volatilidades** (τ): cuál LLM es más sensible al
        prompt/contexto.
      - **direcciones** (cosine entre μ_a y μ_b): qué tan similares
        son las constituciones en dirección.
    """

    label_a: str
    label_b: str
    feature_names: tuple[str, ...]
    constitution: pd.DataFrame   # μ comparison
    volatility: pd.DataFrame     # τ comparison
    cosine_mu_mean: float
    cosine_mu_hdi95: tuple[float, float]
    p_anti_aligned_constitution: float
    n_samples_paired: int

    def summary_text(self) -> str:
        d = len(self.feature_names)
        n_dec_mu = int(self.constitution["decisive"].sum())
        n_dec_tau = int(self.volatility["decisive"].sum())
        return (
            f"Hierarchical comparison {self.label_a} vs {self.label_b}: "
            f"constituciones {n_dec_mu}/{d} decisivas; "
            f"volatilidades {n_dec_tau}/{d} decisivas. "
            f"Cosine(μ_a, μ_b) = {self.cosine_mu_mean:+.3f} "
            f"(HDI95 [{self.cosine_mu_hdi95[0]:+.3f}, "
            f"{self.cosine_mu_hdi95[1]:+.3f}]). "
            f"P(constituciones anti-alineadas) = "
            f"{self.p_anti_aligned_constitution:.3f}."
        )


def compare_constitutions(
    post_a: HierarchicalIRLPosterior,
    post_b: HierarchicalIRLPosterior,
    label_a: str = "A",
    label_b: str = "B",
    decisive_threshold: float = 0.95,
    seed: int = 0,
) -> HierarchicalComparison:
    """Compara μ, τ y dirección entre dos posteriors jerárquicos.

    Asume independencia entre los datasets de A y B (válido cuando
    los LLMs corren en runs separados con seeds independientes).
    """
    if post_a.feature_names != post_b.feature_names:
        raise ValueError(
            f"feature_names difieren: a={post_a.feature_names}, "
            f"b={post_b.feature_names}"
        )
    if not (0.5 < decisive_threshold < 1.0):
        raise ValueError(
            f"decisive_threshold debe estar en (0.5, 1.0); tengo {decisive_threshold}"
        )

    n = min(len(post_a.mu_samples), len(post_b.mu_samples))
    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(post_a.mu_samples), n, replace=False)
    idx_b = rng.choice(len(post_b.mu_samples), n, replace=False)

    mu_a = post_a.mu_samples[idx_a]
    mu_b = post_b.mu_samples[idx_b]
    tau_a = post_a.tau_samples[idx_a]
    tau_b = post_b.tau_samples[idx_b]

    # --- constitución (μ) ---
    diffs_mu = mu_a - mu_b
    p_gt_mu = (diffs_mu > 0).mean(axis=0)
    rows_mu = []
    for k, name in enumerate(post_a.feature_names):
        d_k = diffs_mu[:, k]
        lo, hi = _hdi(d_k)
        rows_mu.append({
            "feature": name,
            "p_a_gt_b": float(p_gt_mu[k]),
            "diff_mean": float(d_k.mean()),
            "diff_hdi95_lo": lo,
            "diff_hdi95_hi": hi,
            "decisive": bool(
                (p_gt_mu[k] >= decisive_threshold)
                or (p_gt_mu[k] <= 1 - decisive_threshold)
            ),
        })
    df_mu = pd.DataFrame(rows_mu).set_index("feature")

    # --- volatilidad (τ) ---
    diffs_tau = tau_a - tau_b
    p_gt_tau = (diffs_tau > 0).mean(axis=0)
    rows_tau = []
    for k, name in enumerate(post_a.feature_names):
        d_k = diffs_tau[:, k]
        lo, hi = _hdi(d_k)
        rows_tau.append({
            "feature": name,
            "p_a_gt_b": float(p_gt_tau[k]),
            "diff_mean": float(d_k.mean()),
            "diff_hdi95_lo": lo,
            "diff_hdi95_hi": hi,
            "decisive": bool(
                (p_gt_tau[k] >= decisive_threshold)
                or (p_gt_tau[k] <= 1 - decisive_threshold)
            ),
        })
    df_tau = pd.DataFrame(rows_tau).set_index("feature")

    # --- cosine entre direcciones de μ ---
    norm_a = np.linalg.norm(mu_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(mu_b, axis=1, keepdims=True)
    valid = ((norm_a > 1e-12).ravel()) & ((norm_b > 1e-12).ravel())
    if valid.sum() == 0:
        cos_mean = float("nan")
        cos_hdi = (float("nan"), float("nan"))
        p_anti = float("nan")
    else:
        mu_a_n = mu_a / np.where(norm_a > 1e-12, norm_a, 1.0)
        mu_b_n = mu_b / np.where(norm_b > 1e-12, norm_b, 1.0)
        cosines = (mu_a_n * mu_b_n).sum(axis=1)
        cos_v = cosines[valid]
        cos_mean = float(cos_v.mean())
        cos_hdi = _hdi(cos_v)
        p_anti = float((cos_v < 0).mean())

    return HierarchicalComparison(
        label_a=label_a,
        label_b=label_b,
        feature_names=post_a.feature_names,
        constitution=df_mu,
        volatility=df_tau,
        cosine_mu_mean=cos_mean,
        cosine_mu_hdi95=cos_hdi,
        p_anti_aligned_constitution=p_anti,
        n_samples_paired=n,
    )
