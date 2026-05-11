"""Demo end-to-end: Hierarchical Bayesian IRL sobre el batch N=20 real.

Aplica el modelo jerárquico recién implementado a los JSONL ya commited
del batch 20260503_181558_dceacd_multiseed. Reporta:

  - μ_LLM (constitución central) por LLM
  - τ_LLM (volatilidad cross-seed) por LLM — la pieza nueva del paper
  - compare_constitutions: P(μ_claude_k > μ_openai_k | data) y
    P(τ_claude_k > τ_openai_k | data)

Persiste outputs en figures/hierarchical_real/
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from guatemala_sim.irl import (
    OUTCOME_FEATURE_NAMES,
    compare_constitutions,
    fit_hierarchical_bayesian_irl,
    parse_menu_run,
)


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "20260503_181558_dceacd_multiseed"
OUT_DIR = ROOT / "figures" / "hierarchical_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_seeds(model: str, n_seeds: int = 20, n_samples: int = 10):
    """Parsea los JSONL de los primeros n_seeds para `model` ('claude' o 'openai').

    Devuelve (features_per_seed, chosen_per_seed, seed_ids).
    """
    feats = []
    chosens = []
    ids = []
    for seed in range(1, n_seeds + 1):
        path = RUNS_DIR / f"seed{seed:03d}_{model}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} no existe")
            continue
        try:
            parsed = parse_menu_run(
                path, feature_seed=0, n_samples=n_samples,
            )
        except Exception as e:
            print(f"  [skip] seed {seed} {model}: {type(e).__name__}: {e}")
            continue
        feats.append(parsed.features)
        chosens.append(parsed.chosen)
        ids.append(seed)
    return feats, chosens, ids


def main():
    print(f"=== Hierarchical Bayesian IRL sobre batch N=20 ===")
    print(f"Outputs → {OUT_DIR}")
    print()

    posteriors = {}
    for model in ["claude", "openai"]:
        print(f"[{model}] parseando JSONL …")
        feats, chosens, ids = load_seeds(model)
        print(f"  {len(feats)} seeds cargadas (T_total={sum(f.shape[0] for f in feats)})")
        print(f"[{model}] ajustando hierarchical NUTS (puede tomar 3-5 min) …")
        post = fit_hierarchical_bayesian_irl(
            feats, chosens,
            feature_names=OUTCOME_FEATURE_NAMES,
            prior_sigma_mu=2.0,
            prior_sigma_tau=1.0,
            draws=1000,
            tune=500,
            chains=2,
            seed=11,
            progressbar=False,
        )
        posteriors[model] = post

        diag = "OK" if post.diagnostics_ok() else "WARN"
        print(f"  R-hat_max={post.rhat_max:.3f}  ESS={post.ess_bulk_min:.0f}  "
              f"diverging={post.diverging}  [{diag}]")

        # Persistir
        post.mu_table().to_csv(OUT_DIR / f"{model}_mu_table.csv")
        post.tau_table().to_csv(OUT_DIR / f"{model}_tau_table.csv")
        np.save(OUT_DIR / f"{model}_mu_samples.npy", post.mu_samples)
        np.save(OUT_DIR / f"{model}_tau_samples.npy", post.tau_samples)

        print(f"\n  μ ({model}, constitución central):")
        for k, name in enumerate(OUTCOME_FEATURE_NAMES):
            mu = post.mu_mean[k]
            lo, hi = post.mu_hdi95[k]
            excludes_zero = (lo > 0) or (hi < 0)
            mark = " *" if excludes_zero else ""
            print(f"    {name:30s}  {mu:+.3f}  [{lo:+.3f}, {hi:+.3f}]{mark}")

        print(f"\n  τ ({model}, volatilidad cross-seed):")
        for k, name in enumerate(OUTCOME_FEATURE_NAMES):
            t = post.tau_mean[k]
            lo, hi = post.tau_hdi95[k]
            print(f"    {name:30s}  {t:.3f}  [{lo:.3f}, {hi:.3f}]")

        pf = post.pooling_factor()
        print(f"\n  pooling factor por dim:")
        for k, name in enumerate(OUTCOME_FEATURE_NAMES):
            print(f"    {name:30s}  {pf[k]:.3f}")
        print()

    # ========== compare ==========
    print("=== Hierarchical comparison: Claude vs OpenAI ===")
    cmp = compare_constitutions(
        posteriors["claude"], posteriors["openai"],
        label_a="Claude", label_b="OpenAI",
    )
    print(cmp.summary_text())
    print()
    print("--- Constituciones (μ): P(Claude > OpenAI) por dim ---")
    print(cmp.constitution.to_string(float_format=lambda x: f"{x:+.3f}"))
    print()
    print("--- Volatilidades (τ): P(Claude más volátil que OpenAI) por dim ---")
    print(cmp.volatility.to_string(float_format=lambda x: f"{x:+.3f}"))

    cmp.constitution.to_csv(OUT_DIR / "comparison_constitution.csv")
    cmp.volatility.to_csv(OUT_DIR / "comparison_volatility.csv")
    (OUT_DIR / "summary.md").write_text(
        f"# Hierarchical Bayesian IRL: Claude vs OpenAI (N=20 seeds)\n\n"
        f"{cmp.summary_text()}\n\n"
        f"## Constituciones (μ)\n\n{cmp.constitution.to_markdown(floatfmt='+.3f')}\n\n"
        f"## Volatilidades (τ)\n\n{cmp.volatility.to_markdown(floatfmt='+.3f')}\n",
        encoding="utf-8",
    )
    print(f"\n[done] outputs en {OUT_DIR}")


if __name__ == "__main__":
    main()
