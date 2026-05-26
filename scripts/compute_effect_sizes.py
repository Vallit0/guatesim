"""Compute Cliff's delta and matched-pairs rank-biserial r_rb for the
13 pre-specified paired Wilcoxon comparisons of the multiseed batch.

Inputs (CSV) live under
  figures/20260503_181558_dceacd_multiseed_irl_multiseed/

Output: figures/20260503_181558_dceacd_multiseed_irl_multiseed/
        paired_effect_sizes.csv

Run from repo root:
  python scripts/compute_effect_sizes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BATCH_DIR = Path("figures/20260503_181558_dceacd_multiseed_irl_multiseed")


def cliffs_delta_paired(diff: np.ndarray) -> float:
    """Sign-based Cliff's delta on per-pair claude-openai differences.

    delta = (#pos - #neg) / n, ignoring exact ties.
    Range [-1, +1]; positive means model_a > model_b dominantly.
    """
    n = len(diff)
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    return (pos - neg) / n


def rank_biserial_paired(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from the Wilcoxon W.

    r_rb = (W_pos - W_neg) / (W_pos + W_neg), where W_+/W_- are the
    sums of ranks of positive / negative differences (ties dropped).
    Range [-1, +1].
    """
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom else 0.0


def magnitude(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def paired_series(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    claude = df[df["model"] == "claude"].sort_values("seed")[col].to_numpy()
    openai = df[df["model"] == "openai"].sort_values("seed")[col].to_numpy()
    assert len(claude) == len(openai), f"length mismatch for {col}"
    return claude, openai


def main() -> None:
    audit = pd.read_csv(BATCH_DIR / "audit_per_seed.csv")
    harms = pd.read_csv(BATCH_DIR / "harms_per_seed.csv")
    consistency = pd.read_csv(BATCH_DIR / "consistency_per_seed.csv")
    posteriors = pd.read_csv(BATCH_DIR / "posteriors_per_seed.csv")
    tests = pd.read_csv(BATCH_DIR / "tests_pareados.csv")

    sources = {
        "cosine_irl": (audit, "cosine_irl"),
        "w_norm": (audit, "w_norm"),
        "chosen_entropy": (audit, "chosen_entropy"),
        "delta_hogares": (harms, "delta_hogares"),
        "muertes_anuales": (harms, "muertes_anuales"),
        "welfare_usd_mm": (harms, "welfare_usd_mm"),
        "cosine_cot": (consistency, "cosine_cot"),
    }
    dim_map = {
        "w[anti_pobreza]": "anti_pobreza",
        "w[anti_deuda]": "anti_deuda",
        "w[pro_aprobacion]": "pro_aprobacion",
        "w[pro_crecimiento]": "pro_crecimiento",
        "w[anti_desviacion_inflacion]": "anti_desviacion_inflacion",
        "w[pro_confianza]": "pro_confianza",
    }

    rows = []
    for _, t in tests.iterrows():
        metric = t["metric"]
        if metric in sources:
            df, col = sources[metric]
            claude, openai = paired_series(df, col)
        elif metric in dim_map:
            dim = dim_map[metric]
            sub = posteriors[posteriors["dim"] == dim]
            claude, openai = paired_series(sub.rename(columns={"w_mean": "w"}), "w")
        else:
            raise ValueError(f"unknown metric {metric}")
        diff = claude - openai
        cd = cliffs_delta_paired(diff)
        rrb = rank_biserial_paired(diff)
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(diff),
                "median_diff_claude_minus_openai": float(np.median(diff)),
                "cliffs_delta": cd,
                "rank_biserial_r_rb": rrb,
                "magnitude_cliffs": magnitude(cd),
                "pvalue_wilcoxon": float(t["pvalue"]),
            }
        )

    out = pd.DataFrame(rows)
    out_path = BATCH_DIR / "paired_effect_sizes.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print()
    print(f"Wrote {out_path}")

    bucket = out["magnitude_cliffs"].value_counts()
    print("\nMagnitude counts (Cliff's delta):")
    print(bucket.to_string())


if __name__ == "__main__":
    main()
